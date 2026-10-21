/*
 * SPDX-FileCopyrightText: 2026 TheGammaSqueeze
 * SPDX-License-Identifier: MIT
 */

/*
 * Draws on the job manager (v7): vkCmdDraw and vkCmdDrawIndexed as a
 * 192-byte Vertex job, which runs the vertex shader once per vertex of the
 * range, and a 256-byte Tiler job that depends on it, in the open batch's
 * vtc chain. There is no IDVS job: the vertex shader writes its outputs
 * into varying buffers the draw allocates (position, point size and the
 * generic varyings, one Attribute Buffer each), and the tiler reads them
 * back.
 *
 * The jobs follow panvk's Bifrost draw (jm/panvk_vX_cmd_draw.c:
 * panvk_draw_prepare_vertex_job, panvk_draw_prepare_tiler_job) and varying
 * link (panvk_vX_shader.c: link_shaders). What depends only on the
 * pipeline is baked with it (mali_pipeline_state.c: the fragment RSD's
 * state words, the Draw's culling bits, the Primitive flags). Choices and
 * differences:
 *
 *  - the vertex shader's RSD is static, built at shader upload
 *    (mali_shader.c); the fragment RSD, with its Blend descriptors behind
 *    it, is rebuilt when the pipeline, blend or depth/stencil group is
 *    dirty;
 *  - the vertex attributes and the varying buffers are rebuilt every draw,
 *    as panvk does: they depend on the vertex and instance counts;
 *  - indexed draws: Bifrost shades the vertex range [min, max] of the
 *    indices, which the jobs carry. panvk finds it on the GPU (libpan
 *    draw_helper.cl); here the CPU reads the indices from the index
 *    buffer's mapping (mali_cmd_state.c) when the draw is recorded, so they
 *    must be in place by then and stay unchanged until the draw has run
 *    (index_range lists the writes it cannot see). Not supported, each
 *    logged once with the draw skipped: an index buffer without a CPU
 *    mapping, and indices that a buffer copy, fill or update or an image
 *    to buffer copy recorded earlier in the same command buffer writes;
 *  - descriptor sets: each stage's tables come from the builder the
 *    dispatches use too (mali_jm_desc_v7.c). The vertex shader's images
 *    go behind its vertex attributes, the fragment shader's in the Tiler
 *    job's Attribute pointers; the set addresses and the dynamic storage
 *    buffer table go through the stage's FAU block;
 *  - no occlusion queries, no multiview, layered passes draw into layer 0
 *    (as on v9: mali_jm_cmd_draw.c's file comment).
 *
 * Job dependencies (Mesa pan_jc.h rules): a stage whose tables were
 * rebuilt gets a descriptor copy job (a Compute job) in front of the
 * Vertex job, the vertex shader's before the fragment shader's. A stage
 * whose FAU block holds uniform buffer words gets a job that copies them
 * in, also in front of the Vertex job. The Vertex job's Dependency 1 is
 * the vertex shader's last copy job, the Tiler job's Dependency 1 the
 * Vertex job and its Dependency 2 the previous tiler-side job. A pending
 * barrier goes to the draw's first job, and to the Vertex job as well
 * when that first job is a fragment shader's copy.
 */

#include "mali_cmd_state.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "panfrost/compiler/pan_nir.h"
#include "util/blend.h"
#include "util/log.h"
#include "vk_log.h"

#include "mali_rsd_v7.h"

#if PAN_ARCH != 7
#error "mali_jm_cmd_draw_v7.c is the Bifrost (PAN_ARCH 7) job manager back half"
#endif

/* ---------------------------------------------------------------------- */
/* Mesa helpers (pan_encoder.h, Collabora, MIT)                            */

/* Mesa's pan_padded_vertex_count: the instance stride of attribute divisors
 * and varying buffers, a number of the form (1, 3, 5, 7 or 9) << n. */
static unsigned
padded_vertex_count(unsigned vertex_count)
{
   if (vertex_count < 10)
      return vertex_count;
   if (vertex_count < 20)
      return (vertex_count + 1) & ~1;

   const unsigned highest = 32 - __builtin_clz(vertex_count);
   const unsigned n = highest - 4;
   const unsigned nibble = (vertex_count >> n) & 0xf;
   const unsigned middle_two = (nibble >> 1) & 0x3;
   switch (middle_two) {
   case 0:
      return (nibble & 1) ? (1u << (n + 1)) * 5 : (1u << n) * 9;
   case 1:
      return (1u << (n + 2)) * 3;
   case 2:
      return (1u << (n + 1)) * 7;
   default:
      return 1u << (n + 4);
   }
}

/* Mesa's pan_compute_npot_divisor: the fixed-point reciprocal of a divisor
 * that is not a power of two. */
static unsigned
compute_npot_divisor(unsigned hw_divisor, unsigned *divisor_r, unsigned *divisor_e)
{
   const unsigned r = util_logbase2(hw_divisor);
   const uint64_t shift_hi = 32 + r;
   const uint64_t t = (uint64_t)1 << shift_hi;
   const uint64_t f0 = t + hw_divisor / 2;
   const uint64_t fi = f0 / hw_divisor;
   const uint64_t ff = f0 - fi * hw_divisor;
   const uint64_t d = fi - (1ull << 31);
   *divisor_r = r;
   *divisor_e = ff > hw_divisor / 2 ? 1 : 0;
   return d;
}

void
mali_jm_draw_tmpl_init(struct mali_jm_draw_tmpl *t)
{
   memset(t, 0, sizeof(*t));
}

/* The Primitive word's First provoking vertex bit (word 0 only; the count
 * is packed as 1 because the v7 packer encodes it minus one and asserts
 * it is at least 1). */
static inline uint32_t
prim_provoking_bit(void)
{
   struct mali_primitive_packed p;
   pan_pack_nodefaults(&p, PRIMITIVE, cfg) {
      cfg.first_provoking_vertex = true;
      cfg.index_count = 1;
   }
   return p.opaque[0];
}

/* The Primitive word's Index type field for `type` (word 0 only, the
 * count packed as in prim_provoking_bit). */
static inline uint32_t
prim_index_type(enum mali_index_type type)
{
   struct mali_primitive_packed p;
   pan_pack_nodefaults(&p, PRIMITIVE, cfg) {
      cfg.index_type = type;
      cfg.index_count = 1;
   }
   return p.opaque[0];
}

/* ---------------------------------------------------------------------- */
/* Varyings                                                                */

/* panvk's varying_buf_id. */
static inline unsigned
varying_buf(gl_varying_slot loc)
{
   switch (loc) {
   case VARYING_SLOT_POS: return MALI_V7_VARY_BUF_POSITION;
   case VARYING_SLOT_PSIZ: return MALI_V7_VARY_BUF_PSIZ;
   default: return MALI_V7_VARY_BUF_GENERAL;
   }
}

/* panvk's varying_format at PAN_ARCH 7. */
static inline uint32_t
varying_format(gl_varying_slot loc, enum pipe_format pfmt)
{
   switch (loc) {
   case VARYING_SLOT_PNTC:
   case VARYING_SLOT_PSIZ:
      return (MALI_R16F << 12) | MALI_RGB_COMPONENT_ORDER_R000;
   case VARYING_SLOT_POS:
      return (MALI_SNAP_4 << 12) | MALI_RGB_COMPONENT_ORDER_RGBA;
   default:
      return GENX(pan_format_from_pipe_format)(pfmt)->hw;
   }
}

/*
 * panvk's emit_varying_attrs: one Attribute descriptor per slot of a
 * stage's varying layout (descriptor i is the LEA_ATTR or LD_VAR index i
 * the compiler used). A generic slot the other stage does not have reads
 * or writes a constant zero. bit_sizes and buf_offsets are indexed by
 * location. *out = the descriptors' address, 0 for an empty layout.
 */
static bool
emit_varying_attrs(struct mali_cmd_buffer *cmd, const struct pan_varying_layout *layout,
                   const unsigned *bit_sizes, const unsigned *buf_offsets, uint64_t *out)
{
   *out = 0;
   if (!layout->count)
      return true;

   struct mali_ptr p = mali_cmd_alloc(cmd, layout->count * pan_size(ATTRIBUTE),
                                      pan_alignment(ATTRIBUTE));
   if (!p.cpu)
      return false;

   struct mali_attribute_packed *attrs = p.cpu;
   for (unsigned i = 0; i < layout->count; i++) {
      pan_pack(&attrs[i], ATTRIBUTE, cfg) {
         const struct pan_varying_slot *slot = pan_varying_layout_slot_at(layout, i);
         if (!slot)
            continue;
         const gl_varying_slot loc = slot->location;
         /* The hardware varyings (the position, point size and the like)
          * are the ones outside the generic section: a built-in the layout
          * keeps in it, such as the clip distances, links as a generic
          * varying (panvk tests the location against VARYING_SLOT_VAR0). */
         const bool is_hw = slot->section != PAN_VARYING_SECTION_GENERIC;
         const bool present = is_hw || bit_sizes[loc] != 0;
         enum pipe_format pfmt = PIPE_FORMAT_NONE;
         if (present && !is_hw) {
            const nir_alu_type base = nir_alu_type_get_base_type(slot->alu_type);
            pfmt = pan_varying_format(base | bit_sizes[loc], slot->ncomps);
         }
         if (!present) {
            cfg.format = (MALI_CONSTANT << 12) | MALI_RGB_COMPONENT_ORDER_0000;
         } else {
            cfg.buffer_index = varying_buf(loc);
            cfg.offset = buf_offsets[loc];
            cfg.format = varying_format(loc, pfmt);
         }
         cfg.offset_enable = false;
      }
   }
   *out = p.gpu;
   return true;
}

/*
 * panvk's link_shaders: the varying buffer strides and the two stages'
 * varying descriptors for the bound pipeline. The position and point size
 * have buffers of their own (the tiler reads them through the Draw's
 * position pointer and the Primitive Size section); the generic varyings
 * share the general buffer at the offsets of the vertex shader's layout,
 * which the fragment shader was compiled against. A generic varying is
 * carried only when the fragment shader reads it and the vertex shader
 * writes it.
 */
static bool
link_varyings(struct mali_cmd_buffer *cmd, const struct mali_graphics_pipeline *p,
              struct mali_jm_draw_tmpl *t)
{
   const struct pan_varying_layout *vl = &p->vs->info.varyings.formats;
   const struct pan_varying_layout *fl = p->fs ? &p->fs->info.varyings.formats : NULL;
   unsigned buf_offsets[VARYING_SLOT_MAX] = {0};
   unsigned bit_sizes[VARYING_SLOT_MAX] = {0};

   pan_varying_layout_require_layout(vl);
   memset(t->vary_stride, 0, sizeof(t->vary_stride));

   const struct pan_varying_slot *pos = pan_varying_layout_find_slot(vl, VARYING_SLOT_POS);
   const struct pan_varying_slot *psiz = pan_varying_layout_find_slot(vl, VARYING_SLOT_PSIZ);
   if (pos) {
      t->vary_stride[MALI_V7_VARY_BUF_POSITION] = 4 * sizeof(uint32_t);
      bit_sizes[VARYING_SLOT_POS] = nir_alu_type_get_type_size(pos->alu_type);
   }
   if (psiz) {
      t->vary_stride[MALI_V7_VARY_BUF_PSIZ] = sizeof(uint16_t);
      bit_sizes[VARYING_SLOT_PSIZ] = nir_alu_type_get_type_size(psiz->alu_type);
   }
   t->vary_stride[MALI_V7_VARY_BUF_GENERAL] = vl->generic_size_B;

   const uint32_t fs_count = fl ? fl->count : 0;
   for (uint32_t i = 0; i < fs_count; i++) {
      const struct pan_varying_slot *fslot = pan_varying_layout_slot_at(fl, i);
      if (!fslot || fslot->section != PAN_VARYING_SECTION_GENERIC)
         continue;
      assert(varying_buf(fslot->location) == MALI_V7_VARY_BUF_GENERAL);
      const struct pan_varying_slot *vslot = pan_varying_layout_find_slot(vl, fslot->location);
      if (vslot) {
         buf_offsets[fslot->location] = vslot->offset;
         bit_sizes[fslot->location] = nir_alu_type_get_type_size(vslot->alu_type);
      }
   }

   if (!emit_varying_attrs(cmd, vl, bit_sizes, buf_offsets, &t->vs_vary))
      return false;
   if (fl) {
      pan_varying_layout_require_format(fl);
      if (!emit_varying_attrs(cmd, fl, bit_sizes, buf_offsets, &t->fs_vary))
         return false;
   } else {
      t->fs_vary = 0;
   }
   return true;
}

/* ---------------------------------------------------------------------- */
/* Blend descriptors and the viewport                                      */

/* pan_fixup_blend_type: the type of an untyped colour output is the
 * render target format's base type at the output's size. */
static nir_alu_type
fixup_blend_type(nir_alu_type type, enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);
   return nir_alu_type_get_base_type(pan_unpacked_type_for_format(desc)) |
          nir_alu_type_get_type_size(type);
}

/* pan_blend.c is_factor_01: is the factor `val` (0 or 1) when the source
 * alpha is `srca`? */
static bool
blend_factor_is(enum pipe_blendfactor f, unsigned val, unsigned srca)
{
   switch (f) {
   case PIPE_BLENDFACTOR_ZERO: return val == 0;
   case PIPE_BLENDFACTOR_ONE: return val == 1;
   case PIPE_BLENDFACTOR_SRC_ALPHA: return val == srca;
   case PIPE_BLENDFACTOR_INV_SRC_ALPHA: return val == 1 - srca;
   default: return false;
   }
}

/* pan_blend_alpha_zero_nop: a source alpha of 0 leaves the destination. */
static bool
blend_alpha_zero_nop(const struct mali_blend_eq *eq)
{
   if (eq->rgb_func != PIPE_BLEND_ADD && eq->rgb_func != PIPE_BLEND_REVERSE_SUBTRACT)
      return false;
   if ((eq->color_mask & 0x8) && !blend_factor_is(eq->alpha_dst, 1, 0))
      return false;
   if (eq->color_mask & 0x7) {
      if (!blend_factor_is(eq->rgb_dst, 1, 0))
         return false;
      if (!blend_factor_is(eq->rgb_src, 0, 0))
         return false;
   }
   return true;
}

/* pan_blend_alpha_one_store: a source alpha of 1 stores the source. */
static bool
blend_alpha_one_store(const struct mali_blend_eq *eq)
{
   if (eq->rgb_func != PIPE_BLEND_ADD && eq->rgb_func != PIPE_BLEND_SUBTRACT)
      return false;
   if (eq->color_mask != 0xf)
      return false;
   return blend_factor_is(eq->rgb_src, 1, 1) && blend_factor_is(eq->alpha_src, 1, 1) &&
          blend_factor_is(eq->rgb_dst, 0, 1) && blend_factor_is(eq->alpha_dst, 0, 1);
}

/*
 * The v7 Blend descriptors (panvk emit_blend_desc): `count` of them packed
 * into `out`, which the draw places right behind the fragment Renderer
 * State Descriptor, where the hardware reads them. Which targets take a
 * blend shader is decided as for mali_gfx_blend (mali_gfx_blend_select),
 * and the shader returns into the fragment code at `fs_code`. A
 * fixed-function descriptor converts from the fragment shader's output
 * register format. The BLEND instructions read the internal words from the
 * hardware's blend FAU slots (bifrost_compile.c BIR_FAU_BLEND_0), so no
 * system value carries them on v7 (no fs.blend_descs).
 */
static bool
emit_blend_descs(struct mali_cmd_buffer *cmd, void *out, uint32_t count, uint64_t fs_code)
{
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   const struct mali_graphics_pipeline *p = d->pipeline;
   const struct mali_gfx_baked *bk = &p->baked;
   const struct mali_render_state *r = &cmd->gfx.render;

   uint64_t shader[MALI_MAX_RTS] = {0};
   bool opaque_fallback[MALI_MAX_RTS] = {false};
   const uint32_t ff_constant = mali_gfx_blend_select(cmd, p, count, shader, opaque_fallback);

   d->blend_shader_mask = 0;
   for (uint32_t i = 0; i < count; i++)
      d->blend_shader_mask |= shader[i] ? BITFIELD_BIT(i) : 0;
   struct mali_blend_packed off;
   pan_pack(&off, BLEND, cfg) {
      cfg.enable = false;
      cfg.internal.mode = MALI_BLEND_MODE_OFF;
      /* Not read in this mode; the v7 packer encodes it minus one and
       * asserts it is at least 1. */
      cfg.internal.fixed_function.num_comps = 4;
   }

   struct mali_blend_packed descs[MALI_MAX_RTS];
   assert(count <= MALI_MAX_RTS);
   for (uint32_t i = 0; i < count; i++) {
      struct mali_blend_packed *o = &descs[i];
      const struct mali_blend_rt_baked *rt = i < bk->rt_count ? &bk->blend[i] : NULL;
      const bool have_rt = i < r->desc.rt_count && r->desc.rt[i].image;

      if (!p->fs || !rt || rt->mode == MALI_BLEND_RT_OFF || !have_rt) {
         *o = off;
         continue;
      }

      const enum pipe_format fmt = r->desc.rt[i].format;
      const unsigned loc = rt->shader_location;
      if (shader[i]) {
         /* Where the fragment shader continues after this target's BLEND;
          * 0 when the BLEND ends it, and a return to 0 ends the thread. */
         const uint32_t ret = loc < MALI_MAX_RTS ? p->fs->info.bifrost.blend[loc].return_offset : 0;
         pan_pack(o, BLEND, cfg) {
            cfg.srgb = util_format_is_srgb(fmt);
            cfg.load_destination = rt->load_destination;
            cfg.round_to_fb_precision = true;
            cfg.blend_constant = ff_constant;
            cfg.internal.mode = MALI_BLEND_MODE_SHADER;
            cfg.internal.shader.pc = (uint32_t)shader[i];
            cfg.internal.shader.return_value = ret ? (uint32_t)(fs_code + ret) : 0;
         }
         continue;
      }

      const bool opaque = opaque_fallback[i] || rt->mode == MALI_BLEND_RT_OPAQUE;
      nir_alu_type out_type =
         loc < MALI_MAX_RTS ? p->fs->info.bifrost.blend[loc].type : nir_type_float32;
      if (p->fs->info.fs.untyped_color_outputs)
         out_type = fixup_blend_type(out_type, fmt);
      pan_pack(o, BLEND, cfg) {
         cfg.srgb = util_format_is_srgb(fmt);
         cfg.load_destination = rt->load_destination && !opaque;
         cfg.round_to_fb_precision = true;
         cfg.blend_constant = ff_constant;
         cfg.internal.mode = opaque ? MALI_BLEND_MODE_OPAQUE : MALI_BLEND_MODE_FIXED_FUNCTION;
         cfg.internal.fixed_function.num_comps = 4;
         cfg.internal.fixed_function.conversion.memory_format =
            GENX(pan_dithered_format_from_pipe_format)(fmt, false);
         if (cfg.internal.mode == MALI_BLEND_MODE_FIXED_FUNCTION &&
             (cfg.internal.fixed_function.conversion.memory_format & 0xff) ==
                MALI_RGB_COMPONENT_ORDER_RGB1) {
            /* Fixed-function blending does not take RGB1 (panvk). */
            cfg.internal.fixed_function.conversion.memory_format &= ~0xff;
            cfg.internal.fixed_function.conversion.memory_format |=
               MALI_RGB_COMPONENT_ORDER_RGBA;
         }
         cfg.internal.fixed_function.rt = i;
         cfg.internal.fixed_function.conversion.register_format =
            mali_v7_register_format(out_type);
         /* Only for a target that blends. With blending off Vulkan
          * ignores the attachment's factors, but the bake keeps them, and
          * factors left from a blending state would make a source alpha
          * of 0 skip a masked store. Mesa's gallium driver builds that
          * equation without factors (pan_cmdstream.c
          * panfrost_create_blend_state), so both flags stay clear. */
         if (!opaque && rt->eq.enable) {
            cfg.internal.fixed_function.alpha_zero_nop = blend_alpha_zero_nop(&rt->eq);
            cfg.internal.fixed_function.alpha_one_store = blend_alpha_one_store(&rt->eq);
         }
      }
      /* The equation word: the baked one, or a plain store with the colour
       * mask for a target whose blend shader is missing (as mali_gfx_blend
       * packs it). */
      if (opaque_fallback[i]) {
         struct mali_blend_equation_packed replace;
         pan_pack(&replace, BLEND_EQUATION, cfg) {
            cfg.rgb.a = MALI_BLEND_OPERAND_A_SRC;
            cfg.rgb.b = MALI_BLEND_OPERAND_B_SRC;
            cfg.rgb.c = MALI_BLEND_OPERAND_C_ZERO;
            cfg.alpha.a = MALI_BLEND_OPERAND_A_SRC;
            cfg.alpha.b = MALI_BLEND_OPERAND_B_SRC;
            cfg.alpha.c = MALI_BLEND_OPERAND_C_ZERO;
            cfg.color_mask = rt->eq.color_mask;
         }
         o->opaque[1] = replace.opaque[0];
      } else {
         o->opaque[1] = rt->equation;
      }
   }
   memcpy(out, descs, count * sizeof(descs[0]));
   return true;
}

/*
 * v7 has no Scissor word and no Minimum/Maximum Z in the Draw: the box and
 * the depth range go in a Viewport descriptor the Tiler job's Draw points at
 * (panvk_emit_viewport). The box is the one v9 packs into its Scissor word
 * (mali_gfx_scissor_rect); the float X/Y bounds keep their infinite
 * defaults, as in panvk. Returns its address, 0 on failure.
 */
static uint64_t
emit_viewport(struct mali_cmd_buffer *cmd)
{
   const VkViewport *vp = &cmd->vk.dynamic_graphics_state.vp.viewports[0];
   int minx, miny, maxx, maxy;
   mali_gfx_scissor_rect(cmd, &minx, &miny, &maxx, &maxy);

   struct mali_ptr p = mali_cmd_alloc(cmd, pan_size(VIEWPORT), pan_alignment(VIEWPORT));
   if (!p.cpu)
      return 0;
   pan_cast_and_pack(p.cpu, VIEWPORT, cfg) {
      cfg.scissor_minimum_x = CLAMP(minx, 0, UINT16_MAX);
      cfg.scissor_minimum_y = CLAMP(miny, 0, UINT16_MAX);
      cfg.scissor_maximum_x = CLAMP(maxx, 0, UINT16_MAX);
      cfg.scissor_maximum_y = CLAMP(maxy, 0, UINT16_MAX);
      cfg.minimum_z = MIN2(vp->minDepth, vp->maxDepth);
      cfg.maximum_z = MAX2(vp->minDepth, vp->maxDepth);
   }
   return p.gpu;
}

/* ---------------------------------------------------------------------- */
/* Per-draw descriptors                                                    */

struct draw_data {
   uint32_t vertex_range;       /* vertices the vertex job shades */
   uint32_t padded_count;       /* padded_vertex_count(range) when instanced, else the range */
   int32_t raw_vertex_offset;   /* Draw offset_start: firstVertex, or min index + vertexOffset */
   uint64_t indices;            /* GPU address of the first index, 0 for a plain draw */
   /* Outputs of the builders. */
   uint64_t attribs, attrib_bufs;
   uint64_t vary_bufs, position, psiz;
   /* The stages whose descriptor tables were rebuilt for this draw: their
    * copied entries are still to be filled (mali_v7_desc_copy), and the
    * buffer index the vertex shader's image Attributes start at. */
   bool vs_copy, fs_copy;
   uint32_t vs_img_buf_offset;
   /* The stages whose FAU block was rebuilt for this draw with uniform
    * buffer words still to be copied in (mali_v7_ubo_copy). */
   bool vs_ubo_copy, fs_ubo_copy;
};

/*
 * panvk_draw_prepare_varyings: the varying buffers of this draw in command
 * memory (padded count x instances x stride bytes each) and their
 * Attribute Buffers, plus the empty entry that stops the Bifrost
 * prefetcher.
 */
static bool
emit_varying_bufs(struct mali_cmd_buffer *cmd, const struct mali_jm_draw_tmpl *t,
                  const struct mali_draw_info *di, struct draw_data *dd)
{
   struct mali_ptr bufs =
      mali_cmd_alloc(cmd, (MALI_V7_VARY_BUF_COUNT + 1) * pan_size(ATTRIBUTE_BUFFER),
                     pan_alignment(ATTRIBUTE_BUFFER));
   if (!bufs.cpu)
      return false;

   struct mali_attribute_buffer_packed descs[MALI_V7_VARY_BUF_COUNT + 1];
   memset(descs, 0, sizeof(descs));
   uint64_t addrs[MALI_V7_VARY_BUF_COUNT] = {0};
   for (unsigned i = 0; i < MALI_V7_VARY_BUF_COUNT; i++) {
      const uint64_t size = (uint64_t)dd->padded_count * di->instance_count * t->vary_stride[i];
      if (size) {
         if (size > UINT32_MAX)
            return false;
         struct mali_ptr b = mali_cmd_alloc(cmd, size, 64);
         if (!b.cpu)
            return false;
         addrs[i] = b.gpu;
      }
      pan_pack(&descs[i], ATTRIBUTE_BUFFER, cfg) {
         cfg.stride = t->vary_stride[i];
         cfg.size = (uint32_t)size;
         cfg.pointer = addrs[i];
      }
   }
   memcpy(bufs.cpu, descs, sizeof(descs));
   dd->vary_bufs = bufs.gpu;
   dd->position = addrs[MALI_V7_VARY_BUF_POSITION];
   dd->psiz = addrs[MALI_V7_VARY_BUF_PSIZ];
   return true;
}

/* panvk_draw_emit_attrib_buf: a vertex binding as an Attribute Buffer pair
 * (the second entry is the NPOT divisor continuation, else zero). The
 * buffer address is 64-byte aligned down and the remainder moved into the
 * attributes' offsets. */
static void
emit_attrib_buf(const struct vk_vertex_binding_state *b, uint32_t stride, uint64_t addr,
                uint64_t size, uint32_t instance_count, uint32_t padded_count,
                struct mali_attribute_buffer_packed *desc)
{
   const uint64_t base = addr & ~63ull;
   const uint32_t bsize = (uint32_t)MIN2(size + (addr & 63), UINT32_MAX);
   const bool per_instance = b->input_rate == VK_VERTEX_INPUT_RATE_INSTANCE;
   const unsigned divisor = padded_count * b->divisor;
   bool continuation = false;

   if (instance_count <= 1) {
      pan_pack(&desc[0], ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.stride = per_instance ? 0 : stride;
         cfg.pointer = base;
         cfg.size = bsize;
      }
   } else if (!per_instance) {
      pan_pack(&desc[0], ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D_MODULUS;
         cfg.divisor = padded_count;
         cfg.stride = stride;
         cfg.pointer = base;
         cfg.size = bsize;
      }
   } else if (!divisor) {
      /* Divisor 0: every instance reads the first element. */
      pan_pack(&desc[0], ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.stride = 0;
         cfg.pointer = base;
         cfg.size = bsize;
      }
   } else if (util_is_power_of_two_or_zero(divisor)) {
      pan_pack(&desc[0], ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D_POT_DIVISOR;
         cfg.stride = stride;
         cfg.pointer = base;
         cfg.size = bsize;
         cfg.divisor_r = __builtin_ctz(divisor);
      }
   } else {
      unsigned divisor_r = 0, divisor_e = 0;
      const unsigned divisor_d = compute_npot_divisor(divisor, &divisor_r, &divisor_e);
      pan_pack(&desc[0], ATTRIBUTE_BUFFER, cfg) {
         cfg.type = MALI_ATTRIBUTE_TYPE_1D_NPOT_DIVISOR;
         cfg.stride = stride;
         cfg.pointer = base;
         cfg.size = bsize;
         cfg.divisor_r = divisor_r;
         cfg.divisor_e = divisor_e;
      }
      pan_cast_and_pack(&desc[1], ATTRIBUTE_BUFFER_CONTINUATION_NPOT, cfg) {
         cfg.divisor_numerator = divisor_d;
         cfg.divisor = b->divisor;
      }
      continuation = true;
   }
   if (!continuation)
      memset(&desc[1], 0, sizeof(desc[1]));
}

/*
 * panvk_draw_prepare_vs_attribs: the vertex attributes of this draw, one
 * Attribute descriptor per location (unbound locations read a constant
 * zero) and two Attribute Buffer entries per binding plus the empty
 * terminator. Rebuilt every draw: the instance count and base instance
 * are in the descriptors.
 *
 * A vertex shader's storage images and texel buffers are attributes too:
 * their Attribute descriptors follow the 16 locations and their Attribute
 * Buffer pairs the bindings', before the terminator. They are the shader's
 * image table, filled by its descriptor copy (mali_v7_desc_copy) in front
 * of the Vertex job. Image i names buffer 2 x (binding count + i), which
 * the image table's size limit keeps within the Attribute's 9-bit buffer
 * index (mali_shader.h).
 */
static_assert(2 * (MALI_MAX_VBS + MALI_V7_MAX_STAGE_IMG_ENTRIES) <= MALI_V7_ATTRIB_BUF_RECORDS,
              "a vertex shader's image pairs fit behind the vertex buffers' within the "
              "Attribute's buffer index");

static bool
emit_attributes(struct mali_cmd_buffer *cmd, const struct mali_graphics_pipeline *p,
                const struct mali_draw_info *di, struct draw_data *dd)
{
   const struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   const struct vk_vertex_input_state *vi = d->vi;
   const uint16_t *strides = d->vi_strides;
   struct mali_v7_desc_tables *vt = &cmd->jm.draw.vs_tab;
   const unsigned num_vbs = util_last_bit(vi->bindings_valid);
   const unsigned num_imgs = p->vs->v7.count[MALI_V7_DESC_TABLE_IMG];
   const unsigned num_attribs =
      num_imgs ? MALI_MAX_VS_ATTRIBS
               : MAX2(util_last_bit(vi->attributes_valid), p->vs->info.attribute_count);
   const unsigned buf_count = (num_vbs + num_imgs) * 2;
   assert(buf_count <= MALI_V7_ATTRIB_BUF_RECORDS);

   struct mali_ptr bufs = mali_cmd_alloc(cmd, (buf_count + 1) * pan_size(ATTRIBUTE_BUFFER),
                                         pan_alignment(ATTRIBUTE_BUFFER));
   if (!bufs.cpu)
      return false;
   struct mali_ptr attrs = {0};
   if (num_attribs + num_imgs) {
      attrs = mali_cmd_alloc(cmd, (num_attribs + num_imgs) * pan_size(ATTRIBUTE),
                             pan_alignment(ATTRIBUTE));
      if (!attrs.cpu)
         return false;
   }

   /* Packed in cached memory, then copied to write-combined command memory. */
   struct mali_attribute_buffer_packed bdescs[2 * MALI_MAX_VBS + 1];
   struct mali_attribute_packed adescs[MALI_MAX_VS_ATTRIBS];
   assert(num_vbs <= MALI_MAX_VBS && num_attribs <= MALI_MAX_VS_ATTRIBS);

   for (unsigned b = 0; b < num_vbs; b++) {
      if ((vi->bindings_valid & BITFIELD_BIT(b)) && d->vb[b].addr) {
         emit_attrib_buf(&vi->bindings[b], strides[b], d->vb[b].addr, d->vb[b].size,
                         di->instance_count, dd->padded_count, &bdescs[2 * b]);
      } else {
         memset(&bdescs[2 * b], 0, 2 * sizeof(bdescs[0]));
      }
   }
   /* The terminator (behind the images when the shader has any). */
   memset(&bdescs[2 * num_vbs], 0, sizeof(bdescs[0]));

   for (unsigned i = 0; i < num_attribs; i++) {
      if (!(vi->attributes_valid & BITFIELD_BIT(i))) {
         pan_pack(&adescs[i], ATTRIBUTE, cfg) {
            cfg.format = (MALI_CONSTANT << 12) | MALI_RGB_COMPONENT_ORDER_0000;
         }
         continue;
      }
      const struct vk_vertex_attribute_state *a = &vi->attributes[i];
      const struct vk_vertex_binding_state *b = &vi->bindings[a->binding];
      const bool per_instance = b->input_rate == VK_VERTEX_INPUT_RATE_INSTANCE;
      const uint64_t addr = d->vb[a->binding].addr;
      pan_pack(&adescs[i], ATTRIBUTE, cfg) {
         cfg.buffer_index = a->binding * 2;
         cfg.offset_enable = true;
         cfg.format = GENX(pan_format_from_pipe_format)(vk_format_to_pipe_format(a->format))->hw;
         cfg.offset = a->offset + (addr & 63) +
                      (per_instance ? (uint64_t)di->first_instance * strides[a->binding] : 0);
      }
   }

   if (!num_imgs) {
      memcpy(bufs.cpu, bdescs, (buf_count + 1) * sizeof(bdescs[0]));
   } else {
      memcpy(bufs.cpu, bdescs, 2 * num_vbs * sizeof(bdescs[0]));
      memcpy((uint8_t *)bufs.cpu + buf_count * pan_size(ATTRIBUTE_BUFFER), &bdescs[2 * num_vbs],
             sizeof(bdescs[0]));
      /* The vertex shader's image table: the copy fills it. */
      const uint64_t img_bufs = 2ull * num_vbs * pan_size(ATTRIBUTE_BUFFER);
      const uint64_t img_attrs = (uint64_t)MALI_MAX_VS_ATTRIBS * pan_size(ATTRIBUTE);
      vt->tables[MALI_V7_DESC_TABLE_IMG] = bufs.gpu + img_bufs;
      vt->img_attribs = attrs.gpu + img_attrs;
      dd->vs_img_buf_offset = 2 * num_vbs;
   }
   if (num_attribs)
      memcpy(attrs.cpu, adescs, num_attribs * sizeof(adescs[0]));
   dd->attrib_bufs = bufs.gpu;
   dd->attribs = attrs.gpu;
   return true;
}

/*
 * The fragment Renderer State Descriptor and one Blend descriptor per
 * render target behind it (panvk_draw_prepare_fs_rsd): the shader half
 * from the fragment shader's info, the state half (words 4 to 11) from the
 * bake (mali_pipeline_state.c pack_rsd_state, repacked for dynamic state).
 * The Blend descriptors are packed first: which targets got a blend shader
 * feeds the RSD under multisampling.
 */
static bool
build_fs_rsd(struct mali_cmd_buffer *cmd, const struct mali_graphics_pipeline *p,
             const struct mali_gfx_baked *bk, struct mali_jm_draw_tmpl *t)
{
   const uint32_t rt_count = cmd->gfx.render.rt_count;
   struct mali_ptr m = mali_cmd_alloc(cmd, pan_size(RENDERER_STATE) + rt_count * pan_size(BLEND),
                                      pan_alignment(RENDERER_STATE));
   if (!m.cpu)
      return false;

   if (!emit_blend_descs(cmd, (uint8_t *)m.cpu + pan_size(RENDERER_STATE), rt_count, t->fs_code))
      return false;

   /* Under multisampling a blend shader runs per sample and reads its
    * sample ID (r61). On Bifrost the RSD asks for that preload and for
    * per-sample shading, as Mesa's gallium driver does (pan_cmdstream.c);
    * bifrost_compile.c preloads it itself on v9+ only. The dynamic blend
    * constants can add blend shaders the bake does not know of, so this is
    * decided here. */
   const struct vk_dynamic_graphics_state *st = p->bind.dynamic ? &cmd->vk.dynamic_graphics_state
                                                                : &p->state;
   const bool per_sample_blend = cmd->gfx.draw.blend_shader_mask &&
                                 st->ms.rasterization_samples > 1;

   struct mali_renderer_state_packed rsd;
   pan_pack(&rsd, RENDERER_STATE, cfg) {
      if (p->fs) {
         mali_v7_rsd_shader(&p->fs->info, t->fs_code, &cfg);
         if (per_sample_blend)
            cfg.preload.fragment.sample_mask_id = true;
      }
   }
   STATIC_ASSERT(sizeof(bk->zsd) == MALI_V7_RSD_STATE_WORDS * sizeof(uint32_t));
   memcpy(&rsd.opaque[MALI_V7_RSD_STATE_WORD_FIRST], bk->zsd, sizeof(bk->zsd));
   if (per_sample_blend) {
      struct mali_multisample_misc_packed ms;
      pan_pack_nodefaults(&ms, MULTISAMPLE_MISC, cfg) {
         cfg.evaluate_per_sample = true;
      }
      rsd.opaque[MALI_V7_RSD_MULTISAMPLE_MISC_WORD] |= ms.opaque[0];
   }
   memcpy(m.cpu, &rsd, sizeof(rsd));
   t->fs_rsd = m.gpu;
   return true;
}

/* ---------------------------------------------------------------------- */
/* Draw state                                                              */

/*
 * The draw's state (mali_cmd_state.h, "A draw's state"), v7 shape: the
 * shared step 1, then each dirty group's descriptors into the template.
 */
static bool
prepare_draw(struct mali_cmd_buffer *cmd, const struct mali_draw_info *di, struct draw_data *dd)
{
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   struct mali_render_state *r = &cmd->gfx.render;
   struct mali_jm_draw_tmpl *t = &cmd->jm.draw;
   const struct mali_graphics_pipeline *p = d->pipeline;
   const struct mali_gfx_bind *bd = &p->bind;
   const struct mali_gfx_baked *dbk;
   uint64_t v;

   if (!mali_gfx_update_dirty(cmd, di, &dbk))
      return false;
   const struct mali_gfx_baked *bk = dbk ? dbk : &p->baked;

   /* A new raw vertex offset rebuilds the VS FAU block when the shader
    * reads it for gl_VertexIndex, or reads the noperspective mask, which
    * shares its FAU word. */
   if (t->raw_vertex_offset != dd->raw_vertex_offset &&
       BITSET_TEST(p->vs->fau.used_sysvals,
                   offsetof(struct mali_graphics_sysvals, vs.raw_vertex_offset) /
                      MALI_FAU_WORD_SIZE))
      d->dirty |= MALI_GFX_DIRTY_VS_FAU;

   if (d->dirty & MALI_GFX_DIRTY_PIPELINE) {
      if (!link_varyings(cmd, p, t))
         return false;
      t->prim0 = bk->tiler_flags;
      t->dcd0 = bk->dcd0[0];
      t->prim = bk->prim;
      t->lines = u_reduced_prim(bk->prim) == MESA_PRIM_LINES;
      t->writes_psiz = bk->prim == MESA_PRIM_POINTS && p->vs->info.vs.writes_point_size;
      /* The bind block's three words are the baked position, varying and
       * fragment entries (mali_pipeline.c fill_bind): on v7 the first is
       * the vertex RSD (one for points and triangles alike), the second is
       * 0 (no varying shader), the third the fragment code address. */
      t->vs_rsd = bd->spd[0];
      t->fs_code = bd->spd[2];
      assert(t->vs_rsd);
   }
   if (d->dirty & (MALI_GFX_DIRTY_PIPELINE | MALI_GFX_DIRTY_BLEND | MALI_GFX_DIRTY_ZSD)) {
      if (!build_fs_rsd(cmd, p, bk, t))
         return false;
   }

   /* Each stage's descriptor tables (mali_jm_desc_v7.c), rebuilt when a
    * set the stage reads was bound, when the stage's shader changed (the
    * tables' layout is the shader's own, which the pipeline's table keys
    * do not cover), and on every draw for a vertex shader with images,
    * whose entries are in the draw's attribute tables (emit_attributes).
    * A stage that reads the set addresses gets them through its FAU
    * block, so its FAU group follows. */
   if (t->vs_tab.shader != p->vs || p->vs->v7.count[MALI_V7_DESC_TABLE_IMG])
      d->dirty |= MALI_GFX_DIRTY_VS_SRT;
   if (t->fs_tab.shader != p->fs)
      d->dirty |= MALI_GFX_DIRTY_FS_SRT;
   if (d->dirty & MALI_GFX_DIRTY_VS_SRT) {
      /* The same sets with other dynamic offsets keep the copied tables;
       * with the same dynamic buffers too, the tables and the FAU block
       * stay. */
      bool same = false;
      if (!mali_v7_desc_refresh(cmd, p->vs, &cmd->gfx.desc, false, &t->vs_tab, &same)) {
         if (vk_command_buffer_has_error(&cmd->vk) ||
             !mali_v7_desc_prepare(cmd, p->vs, &cmd->gfx.desc, false, &t->vs_tab))
            return false;
         dd->vs_copy = true;
      }
      d->vs_srt_key = bd->vs_srt_key;
      d->vs_sets_dirty = 0;
      /* The set addresses and the uniform buffer words in FAU follow the
       * tables (mali_gfx_update_dirty leaves the second to this). */
      if (!same && (p->vs->v7.needs_set_addrs || p->vs->fau.ubo_push_count))
         d->dirty |= MALI_GFX_DIRTY_VS_FAU;
   }
   if (d->dirty & MALI_GFX_DIRTY_FS_SRT) {
      if (p->fs) {
         bool same = false;
         if (!mali_v7_desc_refresh(cmd, p->fs, &cmd->gfx.desc, true, &t->fs_tab, &same)) {
            if (vk_command_buffer_has_error(&cmd->vk) ||
                !mali_v7_desc_prepare(cmd, p->fs, &cmd->gfx.desc, true, &t->fs_tab))
               return false;
            dd->fs_copy = true;
         }
         if (!same && (p->fs->v7.needs_set_addrs || p->fs->fau.ubo_push_count))
            d->dirty |= MALI_GFX_DIRTY_FS_FAU;
      } else {
         memset(&t->fs_tab, 0, sizeof(t->fs_tab));
      }
      d->fs_srt_key = bd->fs_srt_key;
      d->fs_sets_dirty = 0;
   }

   struct mali_graphics_sysvals sv;
   if (d->dirty & (MALI_GFX_DIRTY_VS_FAU | MALI_GFX_DIRTY_FS_FAU)) {
      mali_gfx_fill_sysvals(cmd, &sv);
      sv.vs.raw_vertex_offset = dd->raw_vertex_offset;
   }
   if (d->dirty & MALI_GFX_DIRTY_VS_FAU) {
      /* The stage's set addresses and constant data (the v7_desc words
       * take the place of fs.blend_descs, which v7 shaders never read). */
      mali_v7_desc_sysvals(&t->vs_tab, p->vs, sv.v7_desc.sets);
      sv.common.constant_data = p->vs->v7.data_mem.gpu_va;
      if (!mali_gfx_build_vs_fau(cmd, &sv, &v))
         return false;
      /* The RSD's preload.uniform_count is the shader's FAU word count;
       * the block built here has as many words. */
      assert((v >> 56) == DIV_ROUND_UP(p->vs->info.fau.count, 2));
      t->vs_fau = v & BITFIELD64_MASK(56);
      dd->vs_ubo_copy = p->vs->fau.ubo_push_count && t->vs_fau;
      t->raw_vertex_offset = dd->raw_vertex_offset;
   }
   if (d->dirty & MALI_GFX_DIRTY_FS_FAU) {
      if (p->fs) {
         mali_v7_desc_sysvals(&t->fs_tab, p->fs, sv.v7_desc.sets);
         sv.common.constant_data = p->fs->v7.data_mem.gpu_va;
      }
      if (!mali_gfx_build_fs_fau(cmd, &sv, &v))
         return false;
      assert(!p->fs || (v >> 56) == DIV_ROUND_UP(p->fs->info.fau.count, 2));
      t->fs_fau = v & BITFIELD64_MASK(56);
      dd->fs_ubo_copy = p->fs && p->fs->fau.ubo_push_count && t->fs_fau;
   }
   if (d->dirty & MALI_GFX_DIRTY_VIEWPORT) {
      t->viewport = emit_viewport(cmd);
      if (!t->viewport)
         return false;
   }
   if (d->dirty & MALI_GFX_DIRTY_PASS) {
      /* Layer 0's context (file comment). */
      t->tiler = mali_jm_pass_tiler(r, 0);
      t->tsd = r->tsd;
   }

   /* Per draw: the vertex attributes and the varying buffers. */
   if (!emit_attributes(cmd, p, di, dd) || !emit_varying_bufs(cmd, t, di, dd))
      return false;

   d->dirty = 0;
   return true;
}

/* ---------------------------------------------------------------------- */
/* Indexed draws: the vertex range                                         */

/* The first of the n sorted, disjoint ranges r that ends at or after
 * addr; n if there is none. */
static unsigned
range_first_ending_at(const struct mali_v7_range *r, unsigned n, uint64_t addr)
{
   unsigned lo = 0, hi = n;
   while (lo < hi) {
      const unsigned mid = lo + (hi - lo) / 2;
      if (r[mid].end < addr)
         lo = mid + 1;
      else
         hi = mid;
   }
   return lo;
}

void
mali_v7_note_buffer_write(struct mali_cmd_buffer *cmd, uint64_t addr, uint64_t size)
{
   if (!size)
      return;
   struct util_dynarray *a = &cmd->jm.buf_writes;
   struct mali_v7_range *r = util_dynarray_begin(a);
   const unsigned n = util_dynarray_num_elements(a, struct mali_v7_range);
   struct mali_v7_range w = {addr, addr + size};

   /* The ranges w overlaps or touches, [first, last), become one. */
   const unsigned first = range_first_ending_at(r, n, w.start);
   unsigned last = first;
   for (; last < n && r[last].start <= w.end; last++) {
      w.start = MIN2(w.start, r[last].start);
      w.end = MAX2(w.end, r[last].end);
   }
   if (last == first) {
      if (!util_dynarray_grow(a, struct mali_v7_range, 1)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return;
      }
      r = util_dynarray_begin(a);
      memmove(&r[first + 1], &r[first], (n - first) * sizeof(*r));
   } else {
      memmove(&r[first + 1], &r[last], (n - last) * sizeof(*r));
      a->size -= (last - first - 1) * sizeof(*r);
   }
   r[first] = w;
}

/* Does a transfer recorded earlier in this command buffer write any of the
 * size bytes at addr (mali_v7_note_buffer_write)? */
static inline bool
buffer_written(const struct mali_cmd_buffer *cmd, uint64_t addr, uint64_t size)
{
   const unsigned n = util_dynarray_num_elements(&cmd->jm.buf_writes, struct mali_v7_range);
   if (likely(!n) || !size)
      return false;
   /* Only the first range that ends past addr can overlap. */
   const struct mali_v7_range *r = util_dynarray_begin(&cmd->jm.buf_writes);
   const unsigned i = range_first_ending_at(r, n, addr + 1);
   return i < n && r[i].start < addr + size;
}

/*
 * Min and max of n 16- or 32-bit indices, restart indices (all ones) left
 * out of the max when restart is on: the min needs no exclusion, a restart
 * index being the largest value, so a draw of nothing but restart indices
 * comes out with min > max. Index buffers are often in write-combined
 * memory (azahar's stream buffer), where every load goes to DRAM: the
 * loop issues eight 16-byte loads, which the compiler pairs into LDP,
 * before it uses any, so they are in flight together.
 */
#if defined(__aarch64__)
static void
scan_u16(const uint16_t *idx, uint32_t n, bool restart, uint32_t *min_out, uint32_t *max_out)
{
   uint16x8_t vmin = vdupq_n_u16(UINT16_MAX), vmax = vdupq_n_u16(0);
   const uint16x8_t all = vdupq_n_u16(UINT16_MAX);
   uint32_t i = 0;
   for (; i + 64 <= n; i += 64) {
      uint16x8_t v[8];
      for (unsigned k = 0; k < 8; k++)
         v[k] = vld1q_u16(idx + i + 8 * k);
      for (unsigned k = 0; k < 8; k++) {
         vmin = vminq_u16(vmin, v[k]);
         /* A restart index counts as 0 for the max. */
         vmax = vmaxq_u16(vmax, restart ? vbicq_u16(v[k], vceqq_u16(v[k], all)) : v[k]);
      }
   }
   uint32_t min = vminvq_u16(vmin), max = vmaxvq_u16(vmax);
   for (; i < n; i++) {
      const uint32_t v = idx[i];
      min = MIN2(min, v);
      if (!restart || v != UINT16_MAX)
         max = MAX2(max, v);
   }
   *min_out = min;
   *max_out = max;
}

static void
scan_u32(const uint32_t *idx, uint32_t n, bool restart, uint32_t *min_out, uint32_t *max_out)
{
   uint32x4_t vmin = vdupq_n_u32(UINT32_MAX), vmax = vdupq_n_u32(0);
   const uint32x4_t all = vdupq_n_u32(UINT32_MAX);
   uint32_t i = 0;
   for (; i + 32 <= n; i += 32) {
      uint32x4_t v[8];
      for (unsigned k = 0; k < 8; k++)
         v[k] = vld1q_u32(idx + i + 4 * k);
      for (unsigned k = 0; k < 8; k++) {
         vmin = vminq_u32(vmin, v[k]);
         vmax = vmaxq_u32(vmax, restart ? vbicq_u32(v[k], vceqq_u32(v[k], all)) : v[k]);
      }
   }
   uint32_t min = vminvq_u32(vmin), max = vmaxvq_u32(vmax);
   for (; i < n; i++) {
      const uint32_t v = idx[i];
      min = MIN2(min, v);
      if (!restart || v != UINT32_MAX)
         max = MAX2(max, v);
   }
   *min_out = min;
   *max_out = max;
}
#endif

/*
 * The smallest and largest index of the draw, read from the index
 * buffer's CPU view (restart indices skipped when primitive restart is
 * on); *empty when no index is in bounds or every index is a restart index
 * (nothing to shade).
 *
 * The indices are read when the draw is recorded, not when it runs, so
 * they must be in the buffer by then and stay unchanged until the draw has
 * run: an index outside the range found here makes the Tiler job read
 * positions and varyings the Vertex job did not write. Two cases are
 * caught, logged once and returned as false, and the draw is skipped: an
 * index buffer without a CPU view, and indices that a transfer recorded
 * earlier in this command buffer writes (mali_v7_note_buffer_write), which
 * runs after this read. Other writes are not detected: by a shader, by
 * another command buffer that has not finished by this call, by the host
 * after it, or by a transfer recorded later in this command buffer when it
 * runs again. In host-cached memory that is not host-coherent, indices
 * the GPU wrote can also be read from stale CPU cache lines.
 */
static bool
index_range(struct mali_cmd_buffer *cmd, const struct mali_draw_info *di, bool restart,
            uint32_t *min_out, uint32_t *max_out, bool *empty)
{
   const struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   const struct mali_jm_draw_tmpl *t = &cmd->jm.draw;
   const uint32_t size = d->ib.index_size;

   *empty = false;
   if (!t->ib_cpu || !size) {
      MALI_V7_UNSUPPORTED("an indexed draw whose index buffer has no CPU view");
      return false;
   }
   const uint64_t first = (uint64_t)di->first_index * size;
   if (first >= d->ib.size) {
      *empty = true;
      return true;
   }
   /* Indices past the end of the bound range are left out of the scan. */
   const uint32_t count = (uint32_t)MIN2(di->count, (d->ib.size - first) / size);
   if (buffer_written(cmd, d->ib.addr + first, (uint64_t)count * size)) {
      MALI_V7_UNSUPPORTED("an indexed draw whose indices a transfer earlier in the same "
                          "command buffer writes");
      return false;
   }
   const uint8_t *base = t->ib_cpu + first;
   uint32_t min = UINT32_MAX, max = 0;

#define SCAN(T, RESTART)                                                       \
   do {                                                                        \
      const T *idx = (const T *)base;                                          \
      for (uint32_t i = 0; i < count; i++) {                                   \
         const uint32_t v = idx[i];                                            \
         if (restart && v == (RESTART))                                        \
            continue;                                                          \
         min = MIN2(min, v);                                                   \
         max = MAX2(max, v);                                                   \
      }                                                                        \
   } while (0)
   switch (size) {
   case 1: SCAN(uint8_t, 0xff); break;
#if defined(__aarch64__)
   /* Index buffer offsets are a multiple of the index size. */
   case 2: scan_u16((const uint16_t *)base, count, restart, &min, &max); break;
   default: scan_u32((const uint32_t *)base, count, restart, &min, &max); break;
#else
   case 2: SCAN(uint16_t, 0xffff); break;
   default: SCAN(uint32_t, 0xffffffffu); break;
#endif
   }
#undef SCAN
   if (min > max) {
      *empty = true;
      return true;
   }
   *min_out = min;
   *max_out = max;
   return true;
}

/* ---------------------------------------------------------------------- */
/* The jobs                                                                */

static void
draw(struct mali_cmd_buffer *cmd, const struct mali_draw_info *di)
{
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   struct mali_render_state *r = &cmd->gfx.render;
   struct mali_jm_draw_tmpl *t = &cmd->jm.draw;

   if (!di->count || !di->instance_count)
      return;
   if (!r->active || !d->pipeline || !d->pipeline->vs) {
      vk_logw(VK_LOG_OBJS(cmd), "malisx2: draw outside a render pass or without a pipeline");
      return;
   }
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   const struct mali_graphics_pipeline *p = d->pipeline;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   /* States the bind does not copy into the command buffer come from the
    * pipeline unless it left them dynamic (mali_cmd_bind_graphics). */
   const struct vk_dynamic_graphics_state *st = p->bind.dynamic ? dyn : &p->state;

   struct draw_data dd;
   memset(&dd, 0, sizeof(dd));
   dd.vertex_range = di->count;
   dd.raw_vertex_offset = di->vertex_offset;
   if (di->indexed) {
      /* The vertex range, from the indices in the buffer at this point
       * of recording (index_range says what that needs); a draw whose
       * indices cannot be read here is skipped. */
      uint32_t min, max;
      bool empty;
      if (!index_range(cmd, di, st->ia.primitive_restart_enable, &min, &max, &empty) || empty)
         return;
      dd.vertex_range = max - min + 1;
      dd.raw_vertex_offset = (int32_t)min + di->vertex_offset;
      dd.indices = d->ib.addr + (uint64_t)di->first_index * d->ib.index_size;
   }
   dd.padded_count =
      di->instance_count > 1 ? padded_vertex_count(dd.vertex_range) : dd.vertex_range;

   if (!prepare_draw(cmd, di, &dd)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   /* Rasterizer discard: the Vertex job alone, as in panvk, which keeps
    * the Tiler job only for an occlusion query. */
   const bool tiling = !st->rs.rasterizer_discard_enable;

   /* The Invocation of both jobs (panvk_cmd_draw): (1, vertex range,
    * instances) workgroups of one invocation, with the graphics quirks. */
   struct mali_invocation_packed invocation;
   const uint32_t local[3] = {1, 1, 1};
   const uint32_t groups[3] = {1, dd.vertex_range, di->instance_count};
   mali_v7_pack_invocation(&invocation, local, groups, true);
   const uint32_t instance_size = di->instance_count > 1 ? dd.padded_count : 1;

   struct mali_jm_chain *c = mali_jm_cmd_pass_vtc(cmd);

   /* The descriptor copies of the stages whose tables were rebuilt and
    * their uniform buffer word copies go in front of the Vertex job
    * (panvk_cmd_draw), the vertex shader's before the fragment shader's.
    * The Vertex job depends on the vertex shader's. The fragment shader's
    * are for the fragment job, which follows the whole vtc atom, so no
    * job here waits for them. A barrier pending on the chain goes to the
    * first of these jobs, and to the Vertex job too when that is a
    * fragment shader's copy, which the Vertex job does not wait for.
    *
    * An earlier draw's Tiler job goes in behind this draw's Vertex job
    * (MALI_V7_TILER_DEFER, struct mali_jm_cmd::pend_tiler): the job manager
    * starts a chain's jobs in order and a job waits for its dependencies,
    * so a Tiler job right behind its own Vertex job would hold the next
    * draw's jobs until that vertex shader is done. While tiler_defer is
    * set, this draw's copy jobs and Vertex job go in ahead of the pending
    * Tiler jobs (mali_jm_cmd_add_job). The Tiler jobs stay in draw order
    * among themselves (Dependency 2). With a barrier pending they are
    * linked first, so that it orders this draw's jobs after them. */
   const bool pending = c->pending != 0;
   const bool defer = !pending;
   if (!defer)
      mali_v7_link_tilers(cmd, 0);
   cmd->jm.tiler_defer = defer;
   uint16_t vs_copy = 0, fs_copy = 0, fs_ubo = 0;
   if (dd.vs_copy &&
       !mali_v7_desc_copy(cmd, c, p->vs, &t->vs_tab, dd.vs_img_buf_offset, &vs_copy))
      goto fail;
   /* The vertex shader's uniform buffer words, which the Vertex job reads
    * from its FAU block: a copy job between the descriptor copy and it. */
   uint16_t vs_dep = vs_copy;
   if (dd.vs_ubo_copy &&
       !mali_v7_ubo_copy(cmd, c, p->vs, &cmd->gfx.desc, t->vs_fau, vs_copy, &vs_dep))
      goto fail;
   if (dd.fs_copy && !mali_v7_desc_copy(cmd, c, p->fs, &t->fs_tab, 0, &fs_copy))
      goto fail;
   /* The fragment shader's uniform buffer words, which only the fragment
    * job reads. */
   if (tiling && dd.fs_ubo_copy &&
       !mali_v7_ubo_copy(cmd, c, p->fs, &cmd->gfx.desc, t->fs_fau, 0, &fs_ubo))
      goto fail;
   struct mali_ptr vj = mali_jm_cmd_add_job(cmd, c, MALI_JOB_TYPE_VERTEX, pan_size(COMPUTE_JOB),
                                            pending && !vs_dep && (fs_copy || fs_ubo), vs_dep);
   cmd->jm.tiler_defer = false;
   if (!vj.cpu)
      return;
   const uint16_t vertex_index = c->index;
   /* The Tiler job of the draw MALI_V7_TILER_DEFER draws back goes in
    * behind this Vertex job. */
   if (defer)
      mali_v7_link_tilers(cmd, MALI_V7_TILER_DEFER - 1);

   /* The Vertex job (panvk_draw_prepare_vertex_job, panvk_emit_vertex_dcd). */
   memcpy(pan_section_ptr(vj.cpu, COMPUTE_JOB, INVOCATION), &invocation, sizeof(invocation));
   pan_section_pack(vj.cpu, COMPUTE_JOB, PARAMETERS, cfg) {
      cfg.job_task_split = 5;
   }
   pan_section_pack(vj.cpu, COMPUTE_JOB, DRAW, cfg) {
      cfg.state = t->vs_rsd;
      cfg.attributes = dd.attribs;
      cfg.attribute_buffers = dd.attrib_bufs;
      cfg.varyings = t->vs_vary;
      cfg.varying_buffers = dd.vary_bufs;
      cfg.thread_storage = t->tsd;
      cfg.offset_start = dd.raw_vertex_offset;
      cfg.instance_size = instance_size;
      cfg.push_uniforms = t->vs_fau;
      cfg.uniform_buffers = t->vs_tab.tables[MALI_V7_DESC_TABLE_UBO];
      cfg.textures = t->vs_tab.tables[MALI_V7_DESC_TABLE_TEXTURE];
      cfg.samplers = t->vs_tab.tables[MALI_V7_DESC_TABLE_SAMPLER];
   }

   if (tiling) {
      /* The Tiler job is written here, its header and link when it goes
       * into the chain (mali_v7_link_tilers): behind a later draw's Vertex
       * job, in front of any other job, or when the batch closes. */
      struct mali_ptr tj = mali_cmd_alloc(cmd, pan_size(TILER_JOB), 128);
      if (!tj.cpu)
         return;

      memcpy(pan_section_ptr(tj.cpu, TILER_JOB, INVOCATION), &invocation, sizeof(invocation));

      /* panvk_emit_tiler_primitive: the baked flags, this draw's index
       * type, the pass's provoking vertex, and the counts. */
      struct mali_primitive_packed prim;
      pan_pack(&prim, PRIMITIVE, cfg) {
         cfg.index_count = di->count;
         if (di->indexed) {
            cfg.indices = dd.indices;
            /* Index + base_vertex_offset selects the shaded vertex:
             * vertexOffset minus the shading range's start (libpan
             * draw_helper.cl). */
            cfg.base_vertex_offset = (int64_t)di->vertex_offset - dd.raw_vertex_offset;
         }
      }
      const enum mali_index_type index_type =
         di->indexed ? mali_gfx_index_type(d->ib.index_size) : MALI_INDEX_TYPE_NONE;
      prim.opaque[0] = (t->prim0 & ~prim_provoking_bit()) | prim_index_type(index_type) |
                       (r->first_provoking_vertex ? prim_provoking_bit() : 0);
      memcpy(pan_section_ptr(tj.cpu, TILER_JOB, PRIMITIVE), &prim, sizeof(prim));

      pan_section_pack(tj.cpu, TILER_JOB, PRIMITIVE_SIZE, cfg) {
         if (t->writes_psiz)
            cfg.size_array = dd.psiz;
         else
            cfg.fixed_sized = t->lines ? dyn->rs.line.width : 1.0f;
      }
      pan_section_pack(tj.cpu, TILER_JOB, TILER, cfg) {
         cfg.address = t->tiler;
      }
      pan_section_pack(tj.cpu, TILER_JOB, PADDING, cfg);

      /* panvk_emit_tiler_dcd, with the baked culling bits ORed into word 0. */
      const uint32_t prims_per_instance =
         DIV_ROUND_UP(dd.padded_count, mesa_vertices_per_prim(t->prim));
      struct mali_draw_packed dcd;
      pan_pack(&dcd, DRAW, cfg) {
         cfg.position = dd.position;
         cfg.state = t->fs_rsd;
         /* The fragment shader's images (panvk_emit_tiler_dcd). */
         cfg.attributes = t->fs_tab.img_attribs;
         cfg.attribute_buffers = t->fs_tab.tables[MALI_V7_DESC_TABLE_IMG];
         cfg.viewport = t->viewport;
         cfg.varyings = t->fs_vary;
         cfg.varying_buffers = t->fs_vary ? dd.vary_bufs : 0;
         cfg.thread_storage = t->tsd;
         cfg.offset_start = dd.raw_vertex_offset;
         cfg.instance_size = instance_size;
         cfg.instance_primitive_size = padded_vertex_count(MAX2(prims_per_instance, 1));
         cfg.push_uniforms = t->fs_fau;
         cfg.uniform_buffers = t->fs_tab.tables[MALI_V7_DESC_TABLE_UBO];
         cfg.textures = t->fs_tab.tables[MALI_V7_DESC_TABLE_TEXTURE];
         cfg.samplers = t->fs_tab.tables[MALI_V7_DESC_TABLE_SAMPLER];
         cfg.occlusion_query = MALI_OCCLUSION_MODE_DISABLED;
      }
      dcd.opaque[0] |= t->dcd0;
      memcpy(pan_section_ptr(tj.cpu, TILER_JOB, DRAW), &dcd, sizeof(dcd));

      assert(cmd->jm.pend_tilers < MALI_V7_TILER_DEFER);
      cmd->jm.pend_tiler[cmd->jm.pend_tilers] = tj;
      cmd->jm.pend_tiler_dep1[cmd->jm.pend_tilers] = vertex_index;
      cmd->jm.pend_tilers++;
      if (!defer)
         mali_v7_link_tilers(cmd, 0);
   }

   /* Heap use: polygon-list entries only, the varyings being in command
    * memory; a guess, as on v9. */
   mali_jm_cmd_heap_use(cmd, (uint64_t)di->count * di->instance_count * 32);
   cmd->jm.cur.draws++;
   r->draws++;
   cmd->draws++;
   return;

fail:
   cmd->jm.tiler_defer = false;
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdDraw)(VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount,
                       uint32_t firstVertex, uint32_t firstInstance)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   const struct mali_draw_info di = {
      .count = vertexCount,
      .instance_count = instanceCount,
      .vertex_offset = (int32_t)firstVertex,
      .first_instance = firstInstance,
   };
   draw(cmd, &di);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdDrawIndexed)(VkCommandBuffer commandBuffer, uint32_t indexCount,
                              uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset,
                              uint32_t firstInstance)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   const struct mali_draw_info di = {
      .count = indexCount,
      .instance_count = instanceCount,
      .first_index = firstIndex,
      .vertex_offset = vertexOffset,
      .first_instance = firstInstance,
      .indexed = true,
   };
   draw(cmd, &di);
}
