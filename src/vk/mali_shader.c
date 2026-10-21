/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * One pipeline stage from NIR to an uploaded shader (mali_shader.h), built
 * per arch (v7, v9 and v11).
 *
 * The lowering follows panvk's v9+ path (Mesa src/panfrost/vulkan/
 * panvk_vX_shader.c, MIT) step for step, because the shaders come from the
 * same compiler and the draw code follows panvk's FAU and descriptor
 * model; v7 follows its PAN_ARCH < 9 path. Left out: cooperative matrices,
 * YCbCr, tile images, pipeline executables. Where the code goes differs
 * from panvk: shader code in the device's executable pool (kbase GPU_EX
 * memory in the 4 GiB executable zone, 128-byte aligned), SPDs (RSDs on
 * v7) in the pipeline descriptor pool.
 *
 * v9 and v11 differ here only in the uniform-buffer words pushed into FAU
 * (v11 only, "Dynamic uniform buffers in FAU"); the SPD and Shader
 * Environment layouts are the same, and kraid takes the arch from the GPU
 * ID.
 *
 * v7 (Bifrost) compiles with Mesa's Bifrost backend (pan_use_kraid()
 * answers false below arch 9), and differs more:
 *  - descriptors go through per-type tables instead of resource tables
 *    (mali_shader.h enum mali_v7_desc_table, mali_nir_lower_descriptors_v7.c);
 *  - constant data is a buffer of its own, read through a system value,
 *    as Bifrost has no PC-relative constant pool;
 *  - the BLEND instructions read the Blend descriptors from the hardware's
 *    blend FAU slots, not from system values;
 *  - jobs point at Renderer State Descriptors (RSDs) instead of SPDs
 *    (shader_upload);
 *  - vertex and fragment shaders push uniform-buffer words into FAU too
 *    (lower_ubo_push), and clip distances are a discard in the fragment
 *    shader ("Clip distances").
 */

#ifndef PAN_ARCH
#error "mali_shader.c is built per arch: PAN_ARCH must be set"
#endif
#include "genxml/gen_macros.h"

#include "mali_shader.h"
#if PAN_ARCH >= 10
#include "mali_cs.h"
#endif
#if PAN_ARCH < 9
#include "mali_rsd_v7.h"
#include "pan_format.h"
#endif
#include "mali_vk.h"

#include <stdlib.h>
#include <string.h>

#include "nir_builder.h"
#include "compiler/spirv/nir_spirv.h"
#include "panfrost/compiler/bifrost/bifrost_compile.h"
#include "panfrost/compiler/pan_nir.h"
#include "util/blob.h"
#include "util/log.h"
#include "util/u_dynarray.h"

#include "vk_graphics_state.h"
#include "vk_alloc.h"
#include "vk_log.h"
#include "vk_pipeline.h"

#include "mali_compiler.h"

/* ---------------------------------------------------------------------- */
/* System values                                                           */

/* A magic push-constant base marking system values until the FAU packing
 * below gives everything its final offset. */
#define SYSVALS_PUSH_CONST_BASE MALI_SYSVALS_PUSH_CONST_BASE
/* And one marking the shader's constant FAU words (fau.consts), above any
 * sysval offset. */
#define CONSTS_PUSH_CONST_BASE (SYSVALS_PUSH_CONST_BASE + 4096)
static_assert(sizeof(struct mali_graphics_sysvals) < 4096 &&
                 sizeof(struct mali_compute_sysvals) < 4096,
              "sysvals below the constant base");

#define common_sysval_size(name) \
   sizeof(((struct mali_graphics_sysvals *)NULL)->common.name)
#define graphics_sysval_size(name) \
   sizeof(((struct mali_graphics_sysvals *)NULL)->name)
#define compute_sysval_size(name) \
   sizeof(((struct mali_compute_sysvals *)NULL)->name)
#define sysval_size(ptype, name) ptype##_sysval_size(name)

#define common_sysval_offset(name) \
   offsetof(struct mali_graphics_sysvals, common.name)
#define graphics_sysval_offset(name) \
   offsetof(struct mali_graphics_sysvals, name)
#define compute_sysval_offset(name) \
   offsetof(struct mali_compute_sysvals, name)
#define sysval_offset(ptype, name) ptype##_sysval_offset(name)

#define sysval_entry_size(ptype, name) \
   sizeof(((struct mali_##ptype##_sysvals *)NULL)->name[0])

#define sysval_fau_start(ptype, name) \
   (sysval_offset(ptype, name) / MALI_FAU_WORD_SIZE)
#define sysval_fau_end(ptype, name) \
   ((sysval_offset(ptype, name) + sysval_size(ptype, name) - 1) / MALI_FAU_WORD_SIZE)

#define load_sysval(b, ptype, bitsz, name)                                     \
   nir_load_push_constant(b, sysval_size(ptype, name) / ((bitsz) / 8), bitsz,  \
                          nir_imm_int(b, sysval_offset(ptype, name)),          \
                          .base = SYSVALS_PUSH_CONST_BASE)

#define load_sysval_entry(b, ptype, bitsz, name, dyn_idx)                      \
   nir_load_push_constant(                                                     \
      b, sysval_entry_size(ptype, name) / ((bitsz) / 8), bitsz,                \
      nir_imul_imm(b, dyn_idx, sysval_entry_size(ptype, name)),                \
      .base = SYSVALS_PUSH_CONST_BASE + sysval_offset(ptype, name),            \
      .range = sysval_size(ptype, name))

#define shader_use_sysval(shader, ptype, name)                                 \
   BITSET_SET_RANGE((shader)->fau.used_sysvals, sysval_fau_start(ptype, name), \
                    sysval_fau_end(ptype, name))

/* Common sysvals sit at the same offset for both kinds. */
static_assert(offsetof(struct mali_graphics_sysvals, common) ==
                 offsetof(struct mali_compute_sysvals, common),
              "common sysvals must be at the same offset");
static_assert(offsetof(struct mali_graphics_sysvals, blend) == 0,
              "blend constants must come first");
static_assert(sizeof(struct mali_graphics_sysvals) % MALI_FAU_WORD_SIZE == 0,
              "sysvals are whole FAU words");
static_assert(sizeof(struct mali_compute_sysvals) % MALI_FAU_WORD_SIZE == 0,
              "sysvals are whole FAU words");

struct lower_sysvals_ctx {
   const struct vk_graphics_pipeline_state *state;
};

static bool
lower_sysvals(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct lower_sysvals_ctx *ctx = data;
   unsigned bit_size = intr->def.bit_size;
   nir_def *val = NULL;

   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_load_base_workgroup_id:
      val = load_sysval(b, compute, bit_size, base);
      break;
   case nir_intrinsic_load_num_workgroups:
      val = load_sysval(b, compute, bit_size, num_work_groups);
      break;
   case nir_intrinsic_load_workgroup_size:
      val = load_sysval(b, compute, bit_size, local_group_size);
      break;
   case nir_intrinsic_load_viewport_scale:
   case nir_intrinsic_load_viewport_offset: {
      const bool scale = intr->intrinsic == nir_intrinsic_load_viewport_scale;
      assert(bit_size == 32);
      nir_def *comps[3];
      for (unsigned i = 0; i < 3; i++) {
         const unsigned off =
            scale ? offsetof(struct mali_graphics_sysvals, viewport[i].scale)
                  : offsetof(struct mali_graphics_sysvals, viewport[i].offset);
         comps[i] = nir_load_push_constant(b, 1, 32, nir_imm_int(b, off),
                                           .base = SYSVALS_PUSH_CONST_BASE);
      }
      val = nir_vec(b, comps, intr->def.num_components);
      break;
   }
   case nir_intrinsic_load_first_vertex:
      val = load_sysval(b, graphics, bit_size, vs.first_vertex);
      break;
   case nir_intrinsic_load_base_instance:
      val = load_sysval(b, graphics, bit_size, vs.base_instance);
      break;
   case nir_intrinsic_load_noperspective_varyings_pan:
      val = load_sysval(b, graphics, bit_size, vs.noperspective_varyings);
      break;
#if PAN_ARCH < 9
   case nir_intrinsic_load_raw_vertex_offset:
      /* gl_VertexIndex = raw vertex id + this (pan_nir_lower_vertex_id). */
      val = load_sysval(b, graphics, bit_size, vs.raw_vertex_offset);
      break;
   case nir_intrinsic_load_view_index:
   case nir_intrinsic_load_layer_id:
      /* v7 has no frame argument; panvk draws once per layer and passes the
       * layer as a sysval. Multiview is refused at compile time and a
       * layered pass is drawn into layer 0 only (mali_jm_cmd_draw_v7.c),
       * so both are 0. */
      val = nir_imm_zero(b, 1, 32);
      break;
#else
   case nir_intrinsic_load_view_index:
      if (!ctx->state || !ctx->state->mv || ctx->state->mv->view_mask == 0) {
         val = nir_imm_zero(b, 1, 32);
         break;
      }
      /* A vertex shader's view index is lowered by nir_lower_multiview();
       * in a fragment shader it is the layer. */
      if (b->shader->info.stage == MESA_SHADER_VERTEX)
         return false;
      FALLTHROUGH;
   case nir_intrinsic_load_layer_id:
      val = nir_extract_u8_imm(b, nir_u2u32(b, nir_load_frame_arg_pan(b)), 0);
      break;
#endif /* PAN_ARCH >= 9 */
   case nir_intrinsic_load_printf_buffer_address:
      val = load_sysval(b, common, bit_size, printf_buffer_address);
      break;
#if PAN_ARCH < 9
   case nir_intrinsic_load_blend_descriptor_pan:
      /* Bifrost reads the Blend descriptors from its blend FAU slots
       * (BIR_FAU_BLEND_0 + rt): the backend handles the intrinsic. */
      return false;
#else
   case nir_intrinsic_load_blend_descriptor_pan:
      val = load_sysval(b, graphics, bit_size, fs.blend_descs[nir_intrinsic_base(intr)]);
      break;
#endif
   case nir_intrinsic_load_input_attachment_target_pan: {
      const struct vk_input_attachment_location_state *ial =
         ctx->state ? ctx->state->ial : NULL;

      if (ial && nir_src_is_const(intr->src[0])) {
         /* Known at compile time from the render pass. */
         uint32_t index = nir_src_as_uint(intr->src[0]);
         uint32_t depth_idx = ial->depth_att == MESA_VK_ATTACHMENT_NO_INDEX ?
                                 0 : ial->depth_att + 1;
         uint32_t stencil_idx = ial->stencil_att == MESA_VK_ATTACHMENT_NO_INDEX ?
                                   0 : ial->stencil_att + 1;
         uint32_t target = ~0u;

         if (depth_idx == index || stencil_idx == index) {
            target = MALI_ZS_ATTACHMENT;
         } else {
            for (unsigned i = 0; i < ial->color_attachment_count; i++) {
               if (ial->color_map[i] != MESA_VK_ATTACHMENT_UNUSED &&
                   ial->color_map[i] + 1 == index) {
                  target = MALI_COLOR_ATTACHMENT(i);
                  break;
               }
            }
         }
         val = nir_imm_int(b, target);
      } else {
         nir_def *info = load_sysval_entry(b, graphics, bit_size, iam, intr->src[0].ssa);
         val = nir_channel(b, info, 0);
      }
      break;
   }
   case nir_intrinsic_load_input_attachment_conv_pan: {
      nir_def *info = load_sysval_entry(b, graphics, bit_size, iam, intr->src[0].ssa);
      val = nir_channel(b, info, 1);
      break;
   }
   case nir_intrinsic_load_ro_sink_address_poly:
      val = nir_imm_int64(b, PAN_SHADER_OOB_ADDRESS);
      break;
   default:
      return false;
   }

   assert(val->num_components == intr->def.num_components);
   nir_def_replace(&intr->def, val);
   return true;
}

/* ---------------------------------------------------------------------- */
/* FAU packing                                                             */

/* Where a byte of the sysval or push-constant block lands in the packed
 * FAU block: the used words before it, then the byte within its word. */
static unsigned
remapped_sysval_offset(const struct mali_shader *s, unsigned offset)
{
   return MALI_FAU_WORD_SIZE *
             BITSET_PREFIX_SUM(s->fau.used_sysvals, offset / MALI_FAU_WORD_SIZE) +
          offset % MALI_FAU_WORD_SIZE;
}

static unsigned
remapped_push_const_offset(const struct mali_shader *s, unsigned offset)
{
   return s->fau.sysval_count * MALI_FAU_WORD_SIZE +
          MALI_FAU_WORD_SIZE *
             BITSET_PREFIX_SUM(s->fau.used_push_consts, offset / MALI_FAU_WORD_SIZE) +
          offset % MALI_FAU_WORD_SIZE;
}

/*
 * Constant texture handles. After pan_postprocess_nir every texture
 * instruction carries its resource handle as a vec2 (handle, narrow index
 * bits), a constant for ARMSX2's descriptors. kraid promotes only 32-bit
 * immediates to FAU, so it builds each constant pair in registers before
 * every texture instruction (MOV_IMM + MOV, two instructions per texture
 * access, as many again in copies when the pair is reallocated). TEX_*
 * takes the handle as a 64-bit source that may be a FAU word, so the pair
 * becomes a driver FAU constant and the moves disappear.
 */
static bool
fau_tex_handle(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;

   struct mali_shader *s = data;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   int i = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   if (i < 0)
      return false;

   nir_def *h = tex->src[i].src.ssa;
   if (h->num_components != 2 || h->bit_size != 32 || !nir_src_is_const(tex->src[i].src))
      return false;

   const uint64_t v = nir_src_comp_as_uint(tex->src[i].src, 0) |
                      nir_src_comp_as_uint(tex->src[i].src, 1) << 32;
   unsigned idx;
   for (idx = 0; idx < s->fau.const_count; idx++) {
      if (s->fau.consts[idx] == v)
         break;
   }
   if (idx == s->fau.const_count) {
      if (idx == MALI_MAX_SHADER_FAU_CONSTS)
         return false;
      s->fau.consts[s->fau.const_count++] = v;
   }

   b->cursor = nir_before_instr(instr);
   nir_def *fau = nir_load_push_constant(b, 2, 32, nir_imm_int(b, 0),
                                         .base = CONSTS_PUSH_CONST_BASE +
                                                 idx * MALI_FAU_WORD_SIZE,
                                         .range = MALI_FAU_WORD_SIZE);
   nir_src_rewrite(&tex->src[i].src, fau);
   return true;
}

static bool
collect_push_constant(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_push_constant)
      return false;

   struct mali_shader *s = data;
   uint32_t base = nir_intrinsic_base(intr);
   if (base >= CONSTS_PUSH_CONST_BASE)
      return true; /* always present, fau.const_count */
   bool is_sysval = base >= SYSVALS_PUSH_CONST_BASE;
   uint32_t offset, size;

   if (is_sysval)
      base -= SYSVALS_PUSH_CONST_BASE;

   if (!nir_src_is_const(intr->src[0])) {
      /* A dynamic offset may touch the whole range, and is read from
       * memory through the push_uniforms address. */
      offset = base;
      size = nir_intrinsic_range(intr);
      shader_use_sysval(s, common, push_uniforms);
   } else {
      offset = base + nir_src_as_uint(intr->src[0]);
      size = (intr->def.bit_size / 8) * intr->def.num_components;
   }

   BITSET_WORD *used = is_sysval ? s->fau.used_sysvals : s->fau.used_push_consts;
   BITSET_SET_RANGE(used, offset / MALI_FAU_WORD_SIZE,
                    (offset + size - 1) / MALI_FAU_WORD_SIZE);
   return true;
}

static bool
move_push_constant(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_push_constant)
      return false;

   const struct mali_shader *s = data;
   unsigned base = nir_intrinsic_base(intr);

   b->cursor = nir_before_instr(&intr->instr);

   if (base >= CONSTS_PUSH_CONST_BASE) {
      const unsigned offset =
         (s->fau.sysval_count + BITSET_COUNT(s->fau.used_push_consts)) *
            MALI_FAU_WORD_SIZE +
         base - CONSTS_PUSH_CONST_BASE;
      nir_src_rewrite(&intr->src[0], nir_imm_int(b, offset));
      nir_intrinsic_set_base(intr, 0);
      nir_intrinsic_set_range(intr, 0);
      return true;
   }

   bool is_sysval = base >= SYSVALS_PUSH_CONST_BASE;

   if (is_sysval)
      base -= SYSVALS_PUSH_CONST_BASE;

   if (nir_src_is_const(intr->src[0])) {
      unsigned offset = base + nir_src_as_uint(intr->src[0]);
      offset = is_sysval ? remapped_sysval_offset(s, offset)
                         : remapped_push_const_offset(s, offset);
      nir_src_rewrite(&intr->src[0], nir_imm_int(b, offset));
      /* Nothing after this may use base or range. */
      nir_intrinsic_set_base(intr, 0);
      nir_intrinsic_set_range(intr, 0);
   } else {
      /* Dynamic offsets read the FAU block's copy in memory. */
      unsigned buf_offset =
         remapped_sysval_offset(s, sysval_offset(common, push_uniforms));
      nir_def *buf = nir_load_push_constant(b, 1, 64, nir_imm_int(b, buf_offset));
      unsigned pc_offset = is_sysval ? remapped_sysval_offset(s, base)
                                     : remapped_push_const_offset(s, base);
      nir_def *offset = nir_iadd_imm(b, intr->src[0].ssa, pc_offset);
      unsigned align = nir_combined_align(nir_intrinsic_align_mul(intr),
                                          nir_intrinsic_align_offset(intr));
      align = MIN2(align, MALI_FAU_WORD_SIZE);
      nir_def *value =
         nir_load_global(b, intr->def.num_components, intr->def.bit_size,
                         nir_iadd(b, buf, nir_u2u64(b, offset)), .align_mul = align);
      nir_def_replace(&intr->def, value);
   }
   return true;
}

static void
lower_load_push_consts(nir_shader *nir, struct mali_shader *s)
{
   /* Fold constant offset arithmetic first so fewer loads stay dynamic. */
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_cse);
      nir_opt_peephole_select_options peephole = {
         .limit = 64,
         .expensive_alu_ok = true,
      };
      NIR_PASS(progress, nir, nir_opt_peephole_select, &peephole);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
   } while (progress);

   /* Fragment shaders always reserve the blend constants: a blend shader
    * chosen at draw time may read them from FAU 0-1. */
   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      shader_use_sysval(s, graphics, blend.constants);

   progress = false;
   NIR_PASS(progress, nir, nir_shader_intrinsics_pass, collect_push_constant,
            nir_metadata_all, s);

   s->fau.sysval_count = BITSET_COUNT(s->fau.used_sysvals);
   /* 32 FAU words (256 bytes) are for API push constants. */
   assert(s->fau.sysval_count <= MALI_FAU_WORD_COUNT - 32);
   s->fau.total_count = s->fau.sysval_count + BITSET_COUNT(s->fau.used_push_consts) +
                        s->fau.const_count;
   assert(s->fau.total_count <= MALI_FAU_WORD_COUNT);

   if (progress)
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, move_push_constant,
               nir_metadata_control_flow, s);
}

/* ---------------------------------------------------------------------- */
/* Dynamic uniform buffers in FAU                                          */

#if PAN_ARCH >= 10 || PAN_ARCH < 9
/* Replaces a uniform buffer load with a read of its words in FAU, the
 * first of which is FAU word fau_word; offset is the load's byte offset
 * in the buffer. */
static void
ubo_push_replace(nir_builder *b, nir_intrinsic_instr *intr, unsigned fau_word,
                 uint32_t offset)
{
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *val = nir_load_push_constant(
      b, intr->def.num_components, intr->def.bit_size,
      nir_imm_int(b, fau_word * MALI_FAU_WORD_SIZE + offset % 8));
   nir_def_replace(&intr->def, val);
}

/* Lists the words set in `used` (size bits, words_per_buf for each
 * dynamic buffer in turn) in s->fau.ubo_push, in that order. */
static void
ubo_push_list(struct mali_shader *s, const BITSET_WORD *used, unsigned size,
              unsigned words_per_buf)
{
   unsigned w;
   BITSET_FOREACH_SET(w, used, size) {
      s->fau.ubo_push[s->fau.ubo_push_count++] = (struct mali_ubo_push_word){
         .dyn = w / words_per_buf,
         .offset = (w % words_per_buf) * 8,
      };
   }
}
#endif

#if PAN_ARCH >= 10

/* The FAU block may hold this many words once the uniform-buffer words are
 * added; the rest is left for constants the backend promotes. */
#define UBO_PUSH_FAU_LIMIT 48
#define UBO_PUSH_MAX_BYTES MALI_UBO_PUSH_MAX_BYTES

struct ubo_push_ctx {
   struct mali_shader *s;
   uint32_t first_handle;   /* table-0 handle of dynamic buffer copy 0 */
   BITSET_DECLARE(used, MALI_MAX_DYNAMIC_BUFFERS * (UBO_PUSH_MAX_BYTES / 8));
   bool lower;
};

/* The dynamic buffer copy and byte offset a load reads, or false when it
 * cannot be pushed: a load_ubo with a constant handle naming one of the
 * driver table's dynamic buffer copies and a constant, 4-byte-aligned
 * offset inside the first UBO_PUSH_MAX_BYTES. */
static bool
ubo_push_load(const struct ubo_push_ctx *ctx, nir_intrinsic_instr *intr,
              uint32_t *dyn, uint32_t *offset, uint32_t *bytes)
{
   if (intr->intrinsic != nir_intrinsic_load_ubo ||
       !nir_src_is_const(intr->src[0]) || !nir_src_is_const(intr->src[1]) ||
       intr->def.bit_size < 32)
      return false;

   uint32_t handle = nir_src_as_uint(intr->src[0]);
   if (pan_res_handle_get_table(handle) != 0)
      return false;
   uint32_t idx = pan_res_handle_get_index(handle);
   if (idx < ctx->first_handle || idx - ctx->first_handle >= ctx->s->desc.dyn_bufs.count)
      return false;

   *dyn = idx - ctx->first_handle;
   *offset = nir_src_as_uint(intr->src[1]);
   *bytes = intr->def.num_components * (intr->def.bit_size / 8);
   return *offset % 4 == 0 && *offset + *bytes <= UBO_PUSH_MAX_BYTES;
}

static bool
ubo_push_intr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct ubo_push_ctx *ctx = data;
   uint32_t dyn, offset, bytes;
   if (!ubo_push_load(ctx, intr, &dyn, &offset, &bytes))
      return false;

   const unsigned words_per_buf = UBO_PUSH_MAX_BYTES / 8;
   const unsigned first = dyn * words_per_buf + offset / 8;
   const unsigned last = dyn * words_per_buf + (offset + bytes - 1) / 8;

   if (!ctx->lower) {
      BITSET_SET_RANGE(ctx->used, first, last);
      return false;
   }

   /* The pushed words are in (buffer, offset) order, so the words of one
    * load are adjacent in FAU. */
   unsigned fau_word = ctx->s->fau.ubo_push_start + BITSET_PREFIX_SUM(ctx->used, first);
   ubo_push_replace(b, intr, fau_word, offset);
   return true;
}

/*
 * Replace a fragment shader's constant-offset reads of dynamic uniform
 * buffers with FAU reads. Each such read is otherwise an LD_PKA per warp
 * (an attribute-unit instruction and a load/store-unit read); in FAU the
 * value is an operand of the instruction that uses it. The draw fills the
 * words from the buffer when it is recorded (mali_cmd_gfx_fau).
 *
 * All or nothing: when the words do not fit, nothing is pushed.
 */
static void
lower_ubo_push(nir_shader *nir, struct mali_shader *s)
{
   s->fau.ubo_push_start = s->fau.total_count;
   s->fau.ubo_push_count = 0;

   if (nir->info.stage != MESA_SHADER_FRAGMENT || !s->desc.dyn_bufs.count)
      return;

   struct ubo_push_ctx ctx = {
      .s = s,
      .first_handle = s->desc.driver_table_prefix + 1,
   };
   nir_shader_intrinsics_pass(nir, ubo_push_intr, nir_metadata_all, &ctx);

   const unsigned n = BITSET_COUNT(ctx.used);
   if (!n || n > MALI_MAX_UBO_PUSH_FAUS || s->fau.total_count + n > UBO_PUSH_FAU_LIMIT)
      return;

   const unsigned words_per_buf = UBO_PUSH_MAX_BYTES / 8;
   ubo_push_list(s, ctx.used, MALI_MAX_DYNAMIC_BUFFERS * words_per_buf, words_per_buf);

   ctx.lower = true;
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, ubo_push_intr,
            nir_metadata_control_flow, &ctx);
   s->fau.total_count += n;
}
#elif PAN_ARCH < 9

/*
 * v7: as on v11, constant-offset reads of dynamic uniform buffers become
 * FAU reads, here in vertex and fragment shaders. The job manager has no
 * command-stream loads and stores, so a small Compute job in the vtc chain
 * copies the words into the draw's FAU block before the jobs that read it
 * (mali_jm_desc_v7.c mali_v7_ubo_copy). Each such read is otherwise a
 * uniform buffer LOAD, a load/store-unit message per warp, and azahar's
 * shaders in both stages are bound by them.
 *
 * Partial: when not every load's words fit, the ranges read by the most
 * load instructions go first, each one whose words still fit; the other
 * loads stay loads.
 */

/* The pushed words may fill the FAU block, as Mesa's Bifrost UBO push
 * (bi_opt_push_ubo) fills it: the Bifrost backend promotes no constants
 * into FAU (bifrost_compile.c does that from v9 on), so there is nothing
 * to leave room for, unlike v11's UBO_PUSH_FAU_LIMIT. */
#define V7_UBO_PUSH_FAU_LIMIT MALI_FAU_WORD_COUNT

/* Words are pushed from the first 4 KiB of a buffer. The window only says
 * where pushed words may come from; how many are pushed is the FAU budget
 * above, and the window costs only the `used` bitset below. 4 KiB covers a
 * 3DS emulator's whole vertex uniform block with room for larger ones:
 * azahar's PICA vertex uniforms (vs_pica_data, 96 vec4 from byte 80) end
 * at byte 1616. */
#define V7_UBO_PUSH_MAX_BYTES 4096
#define V7_UBO_PUSH_WORDS_PER_BUF (V7_UBO_PUSH_MAX_BYTES / 8)

/* A word range loads read, and how many load instructions read it. */
struct v7_ubo_push_range {
   uint16_t first, last;
   uint32_t uses;
};

struct v7_ubo_push_ctx {
   struct mali_shader *s;
   uint32_t first_index;   /* UBO table index of dynamic uniform buffer 0 */
   BITSET_DECLARE(used, MALI_MAX_DYNAMIC_UNIFORM_BUFFERS * V7_UBO_PUSH_WORDS_PER_BUF);
   struct util_dynarray ranges;   /* struct v7_ubo_push_range */
   bool lower;
};

/* The first and last 64-bit word a load reads, numbered as in `used`
 * (buffer, then offset), or false when it cannot be pushed: a non-constant
 * index or offset, components narrower than 32 bits, not a dynamic uniform
 * buffer, or an offset that is not 4-byte aligned or reads past
 * V7_UBO_PUSH_MAX_BYTES. */
static bool
v7_ubo_push_load(const struct v7_ubo_push_ctx *ctx, nir_intrinsic_instr *intr,
                 unsigned *first, unsigned *last, uint32_t *offset)
{
   if (intr->intrinsic != nir_intrinsic_load_ubo || !nir_src_is_const(intr->src[0]) ||
       !nir_src_is_const(intr->src[1]) || intr->def.bit_size < 32)
      return false;
   const uint32_t index = nir_src_as_uint(intr->src[0]);
   if (index < ctx->first_index || index - ctx->first_index >= ctx->s->v7.dyn_ubo_count ||
       index - ctx->first_index >= MALI_MAX_DYNAMIC_UNIFORM_BUFFERS)
      return false;
   const uint32_t dyn = index - ctx->first_index;
   *offset = nir_src_as_uint(intr->src[1]);
   const uint32_t bytes = intr->def.num_components * (intr->def.bit_size / 8);
   if (*offset % 4 || *offset + bytes > V7_UBO_PUSH_MAX_BYTES)
      return false;
   *first = dyn * V7_UBO_PUSH_WORDS_PER_BUF + *offset / 8;
   *last = dyn * V7_UBO_PUSH_WORDS_PER_BUF + (*offset + bytes - 1) / 8;
   return true;
}

static bool
v7_ubo_push_intr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct v7_ubo_push_ctx *ctx = data;
   unsigned first, last;
   uint32_t offset;
   if (!v7_ubo_push_load(ctx, intr, &first, &last, &offset))
      return false;

   if (!ctx->lower) {
      util_dynarray_foreach(&ctx->ranges, struct v7_ubo_push_range, r) {
         if (r->first == first && r->last == last) {
            r->uses++;
            return false;
         }
      }
      util_dynarray_append_typed(&ctx->ranges, struct v7_ubo_push_range,
                                 ((struct v7_ubo_push_range){first, last, 1}));
      return false;
   }

   for (unsigned w = first; w <= last; w++) {
      if (!BITSET_TEST(ctx->used, w))
         return false;
   }
   /* The pushed words are in (buffer, offset) order, so the words of one
    * load are adjacent in FAU. */
   const unsigned fau_word = ctx->s->fau.ubo_push_start + BITSET_PREFIX_SUM(ctx->used, first);
   ubo_push_replace(b, intr, fau_word, offset);
   return true;
}

/* More uses first, then the lower first word. */
static int
v7_ubo_push_range_cmp(const void *a, const void *b)
{
   const struct v7_ubo_push_range *x = a, *y = b;
   if (x->uses != y->uses)
      return x->uses > y->uses ? -1 : 1;
   return (int)x->first - (int)y->first;
}

static void
lower_ubo_push(nir_shader *nir, struct mali_shader *s)
{
   s->fau.ubo_push_start = s->fau.total_count;
   s->fau.ubo_push_count = 0;

   if ((nir->info.stage != MESA_SHADER_FRAGMENT && nir->info.stage != MESA_SHADER_VERTEX) ||
       !s->v7.dyn_ubo_count || s->fau.total_count >= V7_UBO_PUSH_FAU_LIMIT)
      return;

   struct v7_ubo_push_ctx ctx = {
      .s = s,
      .first_index = s->v7.count[MALI_V7_DESC_TABLE_UBO],
   };
   util_dynarray_init(&ctx.ranges, NULL);
   nir_shader_intrinsics_pass(nir, v7_ubo_push_intr, nir_metadata_all, &ctx);

   /* The most used ranges first, each one whose words still fit. */
   unsigned budget = MIN2(MALI_MAX_UBO_PUSH_FAUS, V7_UBO_PUSH_FAU_LIMIT - s->fau.total_count);
   const unsigned nr = util_dynarray_num_elements(&ctx.ranges, struct v7_ubo_push_range);
   if (nr)
      qsort(util_dynarray_begin(&ctx.ranges), nr, sizeof(struct v7_ubo_push_range),
            v7_ubo_push_range_cmp);
   util_dynarray_foreach(&ctx.ranges, struct v7_ubo_push_range, r) {
      unsigned need = 0;
      for (unsigned w = r->first; w <= r->last; w++)
         need += !BITSET_TEST(ctx.used, w);
      if (need <= budget) {
         BITSET_SET_RANGE(ctx.used, r->first, r->last);
         budget -= need;
      }
   }
   util_dynarray_fini(&ctx.ranges);

   const unsigned n = BITSET_COUNT(ctx.used);
   if (!n)
      return;
   assert(n <= MALI_MAX_UBO_PUSH_FAUS);
   ubo_push_list(s, ctx.used, MALI_MAX_DYNAMIC_UNIFORM_BUFFERS * V7_UBO_PUSH_WORDS_PER_BUF,
                 V7_UBO_PUSH_WORDS_PER_BUF);

   ctx.lower = true;
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, v7_ubo_push_intr,
            nir_metadata_control_flow, &ctx);
   s->fau.total_count += n;
}
#else
/* v9: nothing is pushed; shaders load from the buffer. The v9 draw
 * (mali_jm_cmd_draw.c) has no step that copies the words into the FAU
 * block: v11 copies them with command-stream loads and stores, which the
 * job manager does not have, and v7 with a Compute job of its own
 * (mali_jm_desc_v7.c mali_v7_ubo_copy). */
static void
lower_ubo_push(nir_shader *nir, struct mali_shader *s)
{
   s->fau.ubo_push_start = s->fau.total_count;
   s->fau.ubo_push_count = 0;
}
#endif

/* ---------------------------------------------------------------------- */
/* Options and lowering                                                    */

const struct nir_shader_compiler_options *
MALI_PER_ARCH(shader_nir_options)(struct mali_device *dev, mesa_shader_stage stage)
{
   return pan_get_nir_shader_compiler_options(PAN_ARCH, stage, false);
}

/* The buffer address formats (panvk_buffer_ubo_addr_format,
 * panvk_buffer_ssbo_addr_format): resource handles on v9+; on v7 a
 * Uniform Buffer table index, and a global address and size read from the
 * set, bounds-checked by nir_lower_explicit_io when storage buffers are
 * robust. */
static nir_address_format
ubo_addr_format(void)
{
#if PAN_ARCH < 9
   return nir_address_format_32bit_index_offset;
#else
   return nir_address_format_vec2_index_32bit_offset;
#endif
}

static nir_address_format
ssbo_addr_format(const struct vk_pipeline_robustness_state *rs)
{
#if PAN_ARCH < 9
   return rs->storage_buffers != VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT ?
             nir_address_format_64bit_bounded_global :
             nir_address_format_64bit_global_32bit_offset;
#else
   (void)rs;
   return nir_address_format_vec2_index_32bit_offset;
#endif
}

struct spirv_to_nir_options
MALI_PER_ARCH(shader_spirv_options)(const struct vk_pipeline_robustness_state *rs)
{
   return (struct spirv_to_nir_options){
      .ubo_addr_format = ubo_addr_format(),
      .ssbo_addr_format = ssbo_addr_format(rs),
      .phys_ssbo_addr_format = nir_address_format_64bit_global,
      .push_const_addr_format = nir_address_format_32bit_offset,
      .shared_addr_format = nir_address_format_32bit_offset,
      .min_ubo_alignment = 16,
      .min_ssbo_alignment = 16,
      /* In debug builds vtn raises SIGTRAP on a parse failure unless told
       * not to; we want the error return. */
      .skip_os_break_in_debug_build = true,
   };
}

void
MALI_PER_ARCH(shader_preprocess)(struct mali_device *dev, nir_shader *nir)
{
   const struct mali_physical_device *pdev = mali_device_physical(dev);

   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      NIR_PASS(_, nir, nir_opt_vectorize_io_vars, nir_var_shader_out);

   NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries,
            nir_shader_get_entrypoint(nir), nir_var_shader_out);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_opt_copy_prop_vars);
   NIR_PASS(_, nir, nir_opt_combine_stores, nir_var_all);
   NIR_PASS(_, nir, nir_opt_loop);
   NIR_PASS(_, nir, nir_opt_barrier_modes);
   NIR_PASS(_, nir, nir_opt_acquire_release_barriers, SCOPE_DEVICE);

   /* Texture lowering has to happen before descriptor lowering: some of it
    * produces size queries that descriptor lowering handles. */
   NIR_PASS(_, nir, nir_lower_system_values);

   nir_lower_compute_system_values_options cs_opts = {
      .has_base_workgroup_id = true,
      .shuffle_local_ids_for_quad_derivatives = true,
   };
   NIR_PASS(_, nir, nir_lower_compute_system_values, &cs_opts);

   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      NIR_PASS(_, nir, nir_lower_wpos_center);

   uint64_t core_max_id = util_last_bit64(pdev->props.shader_present) - 1;
   NIR_PASS(_, nir, nir_inline_sysval, nir_intrinsic_load_core_max_id_arm,
            core_max_id);

   pan_preprocess_nir(nir, pdev->props.gpu_id);
}

#if PAN_ARCH >= 9
/* The valhall backend wants the 32bit_index_offset format; we lower buffer
 * access with vec2_index_32bit_offset so the descriptor walk keeps the
 * array size for bounds checks, and fold the index back here. */
static bool
pack_buf_idx(nir_builder *b, nir_instr *instr, UNUSED void *data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   unsigned index_src;

   switch (intr->intrinsic) {
   case nir_intrinsic_load_ubo:
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
      index_src = 0;
      break;
   case nir_intrinsic_store_ssbo:
      index_src = 1;
      break;
   default:
      return false;
   }

   nir_def *index = intr->src[index_src].ssa;
   if (index->num_components == 1)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_src_rewrite(&intr->src[index_src],
                   nir_iadd(b, nir_channel(b, index, 0), nir_channel(b, index, 1)));
   return true;
}

static bool
is_robust_ssbo_intr(const nir_intrinsic_instr *intr, UNUSED const void *data)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
      return true;
   default:
      return false;
   }
}

/* get_ssbo_size reads the size word of the Buffer descriptor through the
 * resource-table-as-buffer table 62. */
static bool
lower_get_ssbo_size(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_get_ssbo_size)
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   nir_def *res_handle = nir_channel(b, intr->src[0].ssa, 0);
   nir_def *table_idx = nir_ushr_imm(b, res_handle, 24);
   nir_def *res_idx = nir_iand_imm(b, res_handle, BITFIELD_MASK(24));
   nir_def *res_table = nir_ior_imm(b, table_idx, pan_res_handle(62, 0));
   nir_def *buf_idx = nir_iadd(b, res_idx, nir_channel(b, intr->src[0].ssa, 1));
   nir_def *desc_offset = nir_imul_imm(b, buf_idx, MALI_DESCRIPTOR_SIZE);
   nir_def *size = nir_load_ubo(b, 1, 32, res_table, nir_iadd_imm(b, desc_offset, 4),
                                .range = ~0u, .align_mul = MALI_DESCRIPTOR_SIZE,
                                .align_offset = 4);

   nir_def_replace(&intr->def, size);
   return true;
}
#endif /* PAN_ARCH >= 9 */

static bool
mark_all_access_non_uniform(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      for (unsigned i = 0; i < tex->num_srcs; i++) {
         switch (tex->src[i].src_type) {
         case nir_tex_src_texture_offset:
         case nir_tex_src_texture_handle:
            tex->texture_non_uniform = true;
            break;
         case nir_tex_src_sampler_offset:
         case nir_tex_src_sampler_handle:
            tex->sampler_non_uniform = true;
            break;
         default:
            break;
         }
      }
      return true;
   }

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_load_ubo:
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
   case nir_intrinsic_image_load:
   case nir_intrinsic_image_sparse_load:
   case nir_intrinsic_image_store:
   case nir_intrinsic_image_atomic:
   case nir_intrinsic_image_atomic_swap:
   case nir_intrinsic_image_size:
   case nir_intrinsic_image_samples:
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_sparse_load:
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_image_deref_levels:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_image_deref_samples:
   case nir_intrinsic_image_deref_samples_identical:
      nir_intrinsic_set_access(intr, nir_intrinsic_access(intr) | ACCESS_NON_UNIFORM);
      return true;
   default:
      return false;
   }
}

static void
shared_type_info(const struct glsl_type *type, unsigned *size, unsigned *align)
{
   assert(glsl_type_is_vector_or_scalar(type));

   uint32_t comp_size = glsl_type_is_boolean(type) ? 4 : glsl_get_bit_size(type) / 8;
   unsigned length = glsl_get_vector_elements(type);
   *size = comp_size * length;
   *align = comp_size * (length == 3 ? 4 : length);
}

/* Resource lowering (panvk_lower_nir). On v7, a table map that cannot be
 * allocated sets s->v7.failed and ends the lowering early. */
static void
lower_nir(nir_shader *nir, const struct mali_shader_compile_info *info,
          struct mali_shader *s, bool allow_merging_workgroups)
{
   const struct vk_pipeline_robustness_state *rs = info->rs;

   NIR_PASS(_, nir, nir_opt_large_constants, NULL, 32);

   const nir_opt_access_options access_options = {
      .is_vulkan = true,
   };
   NIR_PASS(_, nir, nir_opt_access, &access_options);

#if PAN_ARCH < 9
   /* Bifrost reads a multisampled storage image as a 3D one, samples
    * folded into Z. Before the descriptor lowering, which lowers the
    * sample count query this adds (panvk_lower_nir). */
   NIR_PASS(_, nir, pan_nir_lower_image_ms);

   /* Per-type descriptor tables (mali_shader.h). */
   if (!mali_nir_lower_descriptors_v7(nir, rs, ssbo_addr_format(rs), info->layout, &s->desc,
                                      &s->v7)) {
      s->v7.failed = true;
      return;
   }
#else
   mali_nir_lower_descriptors(nir, rs, info->layout, &s->desc);
#endif

   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_memcpy);

   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ubo, ubo_addr_format());
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo, ssbo_addr_format(rs));
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const,
            nir_address_format_32bit_offset);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_global,
            nir_address_format_64bit_global);

#if PAN_ARCH >= 9
   if (rs->storage_buffers != VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT) {
      NIR_PASS(_, nir, nir_lower_robust_access, is_robust_ssbo_intr, NULL);
      NIR_PASS(_, nir, nir_opt_constant_folding);
      NIR_PASS(_, nir, nir_opt_dce);
   }
#else
   /* With the bounded global format, nir_lower_explicit_io bounds-checks
    * storage buffer accesses itself. */
#endif

   if (allow_merging_workgroups) {
      /* Accesses uniform in the source may not be once workgroups share a
       * warp; mark everything and let the analysis below sort it out. */
      NIR_PASS(_, nir, nir_shader_instructions_pass, mark_all_access_non_uniform,
               nir_metadata_all, NULL);
   }

   enum nir_lower_non_uniform_access_type non_uniform_types =
      nir_lower_non_uniform_ubo_access | nir_lower_non_uniform_ssbo_access |
      nir_lower_non_uniform_texture_access | nir_lower_non_uniform_texture_query |
      nir_lower_non_uniform_image_access | nir_lower_non_uniform_image_query |
      nir_lower_non_uniform_get_ssbo_size;
#if PAN_ARCH < 9
   /* As panvk on Bifrost: divergent texel offsets are made warp-uniform
    * too. */
   non_uniform_types |= nir_lower_non_uniform_texture_offset_access;
#endif

   if (allow_merging_workgroups ||
       nir_has_non_uniform_access(nir, non_uniform_types)) {
      NIR_PASS(_, nir, nir_opt_non_uniform_access);
      struct nir_lower_non_uniform_access_options opts = {
         .types = non_uniform_types,
      };
      NIR_PASS(_, nir, nir_lower_non_uniform_access, &opts);
   }

#if PAN_ARCH >= 9
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_get_ssbo_size,
            nir_metadata_control_flow, NULL);
   NIR_PASS(_, nir, nir_shader_instructions_pass, pack_buf_idx,
            nir_metadata_control_flow, NULL);
#else
   /* v7: nir_lower_explicit_io takes buffer sizes from the storage buffer
    * address, and load_ubo already carries a plain table index. */
#endif

   if (mesa_shader_stage_uses_workgroup(nir->info.stage)) {
      NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_mem_shared,
               shared_type_info);
      NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_shared,
               nir_address_format_32bit_offset);
   }

   if (nir->info.zero_initialize_shared_memory && nir->info.shared_size > 0) {
      nir->info.shared_size = align(nir->info.shared_size, 16);
      NIR_PASS(_, nir, nir_zero_initialize_shared_memory, nir->info.shared_size, 16);
      /* That pass emits load_invocation_id, which needs lowering again. */
      NIR_PASS(_, nir, nir_lower_compute_system_values, NULL);
   }

   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
}

static unsigned
io_type_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

static void
lower_nir_io(nir_shader *nir)
{
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees,
            nir_var_shader_in | nir_var_shader_out, UINT32_MAX);
   NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
            io_type_size, nir_lower_io_use_interpolated_input_intrinsics);
   /* Constant offsets for the constant-indexed derefs left above. */
   NIR_PASS(_, nir, nir_opt_constant_folding);
   pan_nir_lower_mediump_io(nir);
}

static bool
is_src1_factor(VkBlendFactor f)
{
   return f == VK_BLEND_FACTOR_SRC1_COLOR || f == VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR ||
          f == VK_BLEND_FACTOR_SRC1_ALPHA || f == VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
}

bool
MALI_PER_ARCH(shader_fs_keeps_dual_source)(const struct mali_device *dev,
                                           const struct vk_graphics_pipeline_state *state)
{
   if (!dev->vk.enabled_features.dualSrcBlend || !state || !state->cb)
      return false;
   if (BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_BLEND_ENABLES) ||
       BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_BLEND_EQUATIONS))
      return true;
   for (uint32_t i = 0; i < state->cb->attachment_count; i++) {
      const struct vk_color_blend_attachment_state *a = &state->cb->attachments[i];
      if (a->blend_enable &&
          (is_src1_factor(a->src_color_blend_factor) ||
           is_src1_factor(a->dst_color_blend_factor) ||
           is_src1_factor(a->src_alpha_blend_factor) ||
           is_src1_factor(a->dst_alpha_blend_factor)))
         return true;
   }
   return false;
}

/*
 * A fragment shader's second colour output (index 1) is read only by the
 * blend shader of a target that blends with a SRC1 factor. ARMSX2 writes
 * one in nearly every TFX shader; kept, it costs its own arithmetic and
 * the moves that put it in r4-r7, so it is dropped unless the pipeline's
 * blend state uses it (shader_fs_keeps_dual_source).
 */
static bool
drop_dual_source_output(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output ||
       nir_intrinsic_io_semantics(intr).dual_source_blend_index == 0)
      return false;

   nir_instr_remove(&intr->instr);
   return true;
}

static bool
is_const_integral(nir_alu_instr *alu, unsigned src)
{
   if (!nir_src_is_const(alu->src[src].src))
      return false;
   for (unsigned c = 0; c < alu->def.num_components; c++) {
      const double v = nir_src_comp_as_float(alu->src[src].src, alu->src[src].swizzle[c]);
      if (v != (double)(int64_t)v)
         return false;
   }
   return true;
}

/*
 * f2u32 / f2i32 round toward zero, and so does ftrunc; min and max with an
 * integer bound commute with truncation (trunc(min(x, C)) =
 * min(trunc(x), C) for integer C, NaN included since the bound wins
 * either way). So f2u(fmin(fmax(ftrunc(x), 0), 255)), ARMSX2's colour
 * clamp before its integer masking, loses the ftrunc: one FROUND per
 * channel. Every node of the chain must have this one use.
 */
static bool
drop_trunc_before_f2i(nir_builder *b, nir_alu_instr *alu, UNUSED void *data)
{
   if ((alu->op != nir_op_f2u32 && alu->op != nir_op_f2i32) ||
       nir_src_bit_size(alu->src[0].src) != 32)
      return false;

   nir_alu_instr *use = alu;
   unsigned use_src = 0;
   for (;;) {
      nir_def *def = use->src[use_src].src.ssa;
      nir_alu_instr *parent = nir_def_as_alu_or_null(def);
      if (!parent || !list_is_singular(&def->uses))
         return false;
      if (parent->op == nir_op_ftrunc) {
         nir_src_rewrite(&use->src[use_src].src, parent->src[0].src.ssa);
         /* ftrunc's own swizzle composes with the user's. */
         for (unsigned c = 0; c < NIR_MAX_VEC_COMPONENTS; c++)
            use->src[use_src].swizzle[c] =
               parent->src[0].swizzle[use->src[use_src].swizzle[c]];
         return true;
      }
      if (parent->op != nir_op_fmin && parent->op != nir_op_fmax)
         return false;
      if (is_const_integral(parent, 1))
         use_src = 0;
      else if (is_const_integral(parent, 0))
         use_src = 1;
      else
         return false;
      use = parent;
   }
}

/*
 * ftrunc commutes with fmin and fmax against an integral bound K:
 * min(trunc(x), K) = trunc(min(x, K)) for every x, NaN included (NIR's fmin
 * returns the bound for a NaN operand either way), and the same for max.
 * With K = 0 only the sign of a zero result can differ, which fmin and fmax
 * leave open unless the instruction preserves signed zeros. ARMSX2 clamps
 * its truncated colour arithmetic with max(trunc(x), 0) and min(., 255);
 * with the trunc moved outside the clamp, kraid folds max(., 0) into the
 * instruction that computes x (a clamp modifier) instead of issuing it as an
 * FADD, one instruction per channel, and an f2u/f2i of the result loses the
 * trunc altogether (drop_trunc_before_f2i).
 */
static bool
hoist_trunc_over_clamp(nir_builder *b, nir_alu_instr *alu, UNUSED void *data)
{
   if ((alu->op != nir_op_fmin && alu->op != nir_op_fmax) ||
       nir_alu_instr_is_signed_zero_preserve(alu))
      return false;

   for (unsigned t = 0; t < 2; t++) {
      const unsigned k = 1 - t;
      nir_alu_instr *trunc = nir_def_as_alu_or_null(alu->src[t].src.ssa);
      if (!trunc || trunc->op != nir_op_ftrunc || !list_is_singular(&trunc->def.uses) ||
          !is_const_integral(alu, k))
         continue;

      b->cursor = nir_before_instr(&alu->instr);
      nir_alu_src xs = { .src = nir_src_for_ssa(trunc->src[0].src.ssa) };
      for (unsigned c = 0; c < NIR_MAX_VEC_COMPONENTS; c++)
         xs.swizzle[c] = trunc->src[0].swizzle[alu->src[t].swizzle[c]];
      nir_def *x = nir_mov_alu(b, xs, alu->def.num_components);
      nir_def *bound = nir_mov_alu(b, alu->src[k], alu->def.num_components);
      nir_def *m = nir_build_alu2(b, alu->op, x, bound);
      nir_def_as_alu(m)->fp_math_ctrl = alu->fp_math_ctrl;
      nir_def *r = nir_ftrunc(b, m);
      nir_def_as_alu(r)->fp_math_ctrl = trunc->fp_math_ctrl;
      nir_def_replace(&alu->def, r);
      nir_instr_remove(&trunc->instr);
      return true;
   }
   return false;
}

/*
 * ATEST. pan_nir_lower_fs_outputs ends every fragment shader with an ATEST
 * of the colour's alpha. The blob's compiler emits one only in shaders
 * that discard, placed right after the discard with a constant 0 alpha,
 * and none otherwise (observed in the blob's code for ARMSX2's shaders).
 * ATEST hands the final coverage to the depth/stencil unit and does
 * alpha-to-coverage; without discards, coverage
 * writes, depth/stencil writes or reads, side effects or alpha-to-coverage
 * the coverage is the rasterizer's and the depth/stencil work is early, so
 * the shader may leave it out (the draw then forces early depth/stencil
 * and pixel kill, as panvk does for its ATEST-less frame shaders).
 */
static bool
alpha_to_coverage_possible(const struct vk_graphics_pipeline_state *state)
{
   if (!state)
      return true;
   if (BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_ALPHA_TO_COVERAGE_ENABLE))
      return true;
   return state->ms && state->ms->alpha_to_coverage_enable;
}

static bool
is_demote(const nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   switch (nir_instr_as_intrinsic(instr)->intrinsic) {
   case nir_intrinsic_demote:
   case nir_intrinsic_demote_if:
   case nir_intrinsic_terminate:
   case nir_intrinsic_terminate_if:
      return true;
   default:
      return false;
   }
}

static bool
fs_needs_atest(const nir_shader *nir, const struct vk_graphics_pipeline_state *state,
               uint32_t input_attachment_read, bool meta)
{
   if (meta) {
      return nir->info.fs.uses_discard ||
             (nir->info.outputs_written & (BITFIELD64_BIT(FRAG_RESULT_DEPTH) |
                                           BITFIELD64_BIT(FRAG_RESULT_STENCIL) |
                                           BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK)));
   }
   if (!state || alpha_to_coverage_possible(state))
      return true;
   if (state->rasterization_order_access &
       (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))
      return true;
   if (nir->info.fs.uses_discard || nir->info.writes_memory)
      return true;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (is_demote(instr))
               return true;
         }
      }
   }
   if (nir->info.outputs_written & (BITFIELD64_BIT(FRAG_RESULT_DEPTH) |
                                    BITFIELD64_BIT(FRAG_RESULT_STENCIL) |
                                    BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK)))
      return true;
   /* A depth or stencil input attachment read from the tile buffer makes
    * the depth/stencil update late. */
   if (input_attachment_read) {
      const struct vk_input_attachment_location_state *ial = state->ial;
      if (!ial)
         return true;
      const uint8_t zs[2] = { ial->depth_att, ial->stencil_att };
      for (unsigned i = 0; i < 2; i++) {
         const uint32_t bit = zs[i] == MESA_VK_ATTACHMENT_NO_INDEX ? BITFIELD_BIT(0) :
                              zs[i] != MESA_VK_ATTACHMENT_UNUSED   ? BITFIELD_BIT(zs[i] + 1) :
                                                                     0;
         if (input_attachment_read & bit)
            return true;
      }
   }
   return false;
}

/* Tile reads and the coverage preload read r60, which ATEST rewrites:
 * kraid's register allocator cannot give both values that register (an
 * assertion in ra.rs), so the ATEST stays after them. */
static bool
reads_coverage(const nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   switch (nir_instr_as_intrinsic(instr)->intrinsic) {
   case nir_intrinsic_load_tile_pan:
   case nir_intrinsic_load_tile_res_pan:
   case nir_intrinsic_load_cumulative_coverage_pan:
   case nir_intrinsic_load_sample_mask_in:
   case nir_intrinsic_load_sample_mask:
      return true;
   default:
      return false;
   }
}

/*
 * Without alpha-to-coverage, ATEST's alpha is not used: give it 0 so it
 * does not wait for the colour, and move it (with the coverage read) to
 * just after the last discard when every discard is in the shader's
 * top-level code, as the blob does, so the depth/stencil unit gets the
 * final coverage while the colour is still being computed.
 */
static void
place_atest_early(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_intrinsic_instr *atest = NULL;
   nir_instr *last_demote = NULL;
   bool nested_demote = false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_atest_pan)
            atest = nir_instr_as_intrinsic(instr);
      }
   }
   if (!atest)
      return;

   /* The last discard or coverage read before the ATEST (other than the
    * ATEST's own coverage read). */
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr == &atest->instr)
            goto found;
         if (instr == nir_def_instr(atest->src[0].ssa))
            continue;
         if (is_demote(instr) || reads_coverage(instr)) {
            if (block->cf_node.parent != &impl->cf_node)
               nested_demote = true;
            last_demote = instr;
         }
      }
   }
found:

   nir_builder b = nir_builder_at(nir_before_instr(&atest->instr));
   nir_src_rewrite(&atest->src[1], nir_imm_float(&b, 0.0f));

   nir_instr *cov = nir_def_instr(atest->src[0].ssa);
   if (nested_demote || atest->instr.block->cf_node.parent != &impl->cf_node ||
       cov->type != nir_instr_type_intrinsic ||
       nir_instr_as_intrinsic(cov)->intrinsic != nir_intrinsic_load_cumulative_coverage_pan)
      return;

   nir_instr *zero = nir_def_instr(atest->src[1].ssa);
   const nir_cursor at = last_demote ? nir_after_instr(last_demote)
                                     : nir_before_impl(impl);
   nir_instr_move(at, &atest->instr);
   nir_instr_move(nir_before_instr(&atest->instr), zero);
   nir_instr_move(nir_before_instr(zero), cov);
   nir_progress(true, impl, nir_metadata_control_flow);
}

#if PAN_ARCH < 9
/* ---------------------------------------------------------------------- */
/* Stencil tile reads                                                      */

/*
 * The conversion of a stencil tile read, S8 into 32-bit unsigned
 * registers, as panvk packs it at PAN_ARCH < 9
 * (panvk_vX_nir_lower_input_attachment_loads.c). The input attachment
 * lowering, compiled once, leaves the zero conversion of v9+ there, which
 * faults on v7 with INVALID_INSTR_ENC.
 */
static bool
v7_stencil_tile_conv(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_tile_pan ||
       nir_intrinsic_io_semantics(intr).location != FRAG_RESULT_STENCIL)
      return false;
   assert(nir_src_is_const(intr->src[2]) && nir_src_as_uint(intr->src[2]) == 0);

   struct mali_internal_conversion_packed conv;
   pan_pack(&conv, INTERNAL_CONVERSION, cfg) {
      cfg.register_format = MALI_REGISTER_FILE_FORMAT_U32;
      cfg.memory_format = GENX(pan_dithered_format_from_pipe_format)(PIPE_FORMAT_S8_UINT, false);
   }
   b->cursor = nir_before_instr(&intr->instr);
   nir_src_rewrite(&intr->src[2], nir_imm_int(b, conv.opaque[0]));
   return true;
}

/* ---------------------------------------------------------------------- */
/* Clip distances                                                          */

/*
 * Bifrost has no clip distance hardware. As Mesa's gallium driver does for
 * user clip planes (nir_lower_clip_fs), the distances are varyings, and the
 * fragment shader discards a fragment whose interpolated distance is
 * negative.
 *
 * The vertex shader keeps gl_ClipDistance as vk_spirv_to_nir leaves it: a
 * compact float array, whatever its name (a gl_PerVertex member is
 * ".gl_ClipDistance", and stripped SPIR-V has none). nir_lower_io stores
 * its element i in component i % 4 of VARYING_SLOT_CLIP_DIST0 + i / 4, so
 * the vertex shader's varying layout has a slot for each of the two
 * locations it writes, as wide as the components it writes there. The
 * fragment shader is compiled against that layout (its key includes the
 * vertex stage, mali_pipeline.c) and tests exactly those components.
 *
 * A pipeline without a fragment shader gets one that does nothing else
 * (mali_pipeline.c clip_fs_compile).
 *
 * The result is that of clipping the primitive, except that:
 *  - the distance is interpolated at the pixel centre, so under
 *    multisampling the clipped edge is not antialiased;
 *  - the discard makes the depth/stencil update late
 *    (mali_pipeline_state.c earlyzs_get), but with a fragment shader that
 *    forces early fragment tests, depth and stencil are written before the
 *    shader runs, so clipped fragments still write them.
 */

/*
 * The fragment shader's gl_ClipDistance input, a compact float array at
 * VARYING_SLOT_CLIP_DIST0 with at least `count` elements: the shader's
 * own, lengthened when it declares fewer, or a new one. Its elements lower
 * to the components the vertex shader writes, as the vertex shader's do.
 */
static nir_variable *
clip_distance_input(nir_shader *nir, unsigned count)
{
   const struct glsl_type *type = glsl_array_type(glsl_float_type(), count, 0);
   nir_variable *var =
      nir_find_variable_with_location(nir, nir_var_shader_in, VARYING_SLOT_CLIP_DIST0);

   if (!var) {
      var = nir_variable_create(nir, nir_var_shader_in, type, "gl_ClipDistance");
      var->data.location = VARYING_SLOT_CLIP_DIST0;
      var->data.compact = true;
   } else if (glsl_get_length(var->type) < count) {
      /* The variable's derefs carry its type too. */
      var->type = type;
      nir_foreach_function_impl(impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_deref)
                  continue;
               nir_deref_instr *deref = nir_instr_as_deref(instr);
               if (deref->deref_type == nir_deref_type_var && deref->var == var)
                  deref->type = type;
            }
         }
      }
   }
   assert(var->data.compact);

   nir->info.clip_distance_array_size =
      MAX2(nir->info.clip_distance_array_size, glsl_get_length(var->type));
   return var;
}

/* The discard for every clip distance the vertex shader writes
 * (vs_layout), before the fragment shader's input locations are assigned. */
static void
lower_clip_distance_fs(nir_shader *nir, const struct pan_varying_layout *vs_layout)
{
   if (!vs_layout)
      return;
   const struct pan_varying_slot *slots[2] = {
      pan_varying_layout_find_slot(vs_layout, VARYING_SLOT_CLIP_DIST0),
      pan_varying_layout_find_slot(vs_layout, VARYING_SLOT_CLIP_DIST1),
   };
   if (!slots[0] && !slots[1])
      return;

   nir_variable *in = clip_distance_input(nir, slots[1] ? 4 + slots[1]->ncomps
                                                        : slots[0]->ncomps);

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *clip = nir_imm_false(&b);
   for (unsigned i = 0; i < 2; i++) {
      for (unsigned c = 0; slots[i] && c < slots[i]->ncomps; c++) {
         nir_def *d = nir_load_array_var_imm(&b, in, 4 * i + c);
         clip = nir_ior(&b, clip, nir_flt_imm(&b, d, 0.0));
      }
   }
   nir_discard_if(&b, clip);
   nir->info.fs.uses_discard = true;
   nir_progress(true, impl, nir_metadata_control_flow);
}
#endif

/* ---------------------------------------------------------------------- */
/* Backend compile                                                         */

static VkResult
compile_nir(struct mali_device *dev, nir_shader *nir,
            const struct pan_compile_inputs *compile_inputs,
            const struct vk_graphics_pipeline_state *state, struct mali_shader *s)
{
   struct pan_compile_inputs inputs = *compile_inputs;

   bool hoisted;
   do {
      hoisted = false;
      NIR_PASS(hoisted, nir, nir_shader_alu_pass, hoist_trunc_over_clamp,
               nir_metadata_control_flow, NULL);
   } while (hoisted);
   NIR_PASS(_, nir, nir_shader_alu_pass, drop_trunc_before_f2i,
            nir_metadata_control_flow, NULL);
   pan_postprocess_nir(nir, &inputs, &s->info);

   struct lower_sysvals_ctx ctx = { .state = state };
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_sysvals,
            nir_metadata_control_flow, &ctx);

   NIR_PASS(_, nir, nir_shader_instructions_pass, fau_tex_handle,
            nir_metadata_control_flow, s);
   lower_load_push_consts(nir, s);
   lower_ubo_push(nir, s);

   /* The driver's words come first; the backend may put promoted
    * constants after them. */
   inputs.fau.reserved = s->fau.total_count * 2;
   inputs.fau.promote_immediates = true;

   struct util_dynarray binary;
   util_dynarray_init(&binary, NULL);
   char why[160];
   enum mali_compile_status st =
      mali_compile_nir(nir, &inputs, &binary, &s->info, why, sizeof(why));
   if (st != MALI_COMPILE_OK) {
      util_dynarray_fini(&binary);
      return vk_errorf(dev, VK_ERROR_INVALID_SHADER_NV, "%s shader: %s",
                       mesa_shader_stage_name(nir->info.stage), why);
   }

   /* The FAU size including promoted constants (info counts 32-bit
    * halves). */
   s->fau.total_count = DIV_ROUND_UP(s->info.fau.count, 2);
   s->info.fau.count = s->fau.total_count * 2;

   if (binary.size) {
      s->bin = malloc(binary.size);
      if (!s->bin) {
         util_dynarray_fini(&binary);
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
      memcpy(s->bin, binary.data, binary.size);
      s->bin_size = binary.size;
   }
   util_dynarray_fini(&binary);

   if (nir->info.stage == MESA_SHADER_COMPUTE) {
      s->cs.local_size[0] = nir->info.workgroup_size[0];
      s->cs.local_size[1] = nir->info.workgroup_size[1];
      s->cs.local_size[2] = nir->info.workgroup_size[2];
   }

   return VK_SUCCESS;
}

/* ---------------------------------------------------------------------- */
/* Bifrost Renderer State Descriptor counts                                */

#if PAN_ARCH < 9
/*
 * The shader info the RSD is packed from, patched after the compile as
 * panvk_compile_nir does at PAN_ARCH < 9: descriptor counts from the
 * driver's tables (mali_nir_lower_descriptors_v7.c), with the dummy
 * sampler for a stage with textures but no sampler; the vertex attribute
 * count up to the highest generic location read, holes included (one
 * descriptor per location); and images counted as attributes.
 */
static VkResult
v7_fixup_info(struct mali_device *dev, const nir_shader *nir, struct mali_shader *s)
{
   /* More images than the 9-bit Attribute Buffer index can address
    * (mali_shader.h). Only a layout over the per-stage limits gets here;
    * its indices would spill into the Attribute's Format bits. */
   if (s->v7.count[MALI_V7_DESC_TABLE_IMG] > MALI_V7_MAX_STAGE_IMG_ENTRIES)
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "%s shader: %u storage images and texel buffers, more than the %u a "
                       "Bifrost stage can address",
                       mesa_shader_stage_name(s->stage), s->v7.count[MALI_V7_DESC_TABLE_IMG],
                       MALI_V7_MAX_STAGE_IMG_ENTRIES);

   s->info.ubo_count = s->v7.count[MALI_V7_DESC_TABLE_UBO] + s->v7.dyn_ubo_count;
   s->info.texture_count = s->v7.count[MALI_V7_DESC_TABLE_TEXTURE];
   s->info.sampler_count = s->v7.count[MALI_V7_DESC_TABLE_SAMPLER];
   if (s->info.texture_count && !s->info.sampler_count)
      s->info.sampler_count = 1;

   if (nir->constant_data_size) {
      s->v7.data = malloc(nir->constant_data_size);
      if (!s->v7.data)
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      memcpy(s->v7.data, nir->constant_data, nir->constant_data_size);
      s->v7.data_size = nir->constant_data_size;
   }

   if (s->stage == MESA_SHADER_VERTEX) {
      const uint32_t generic =
         (s->info.attributes_read & VERT_BIT_GENERIC_ALL) >> VERT_ATTRIB_GENERIC0;
      s->info.attribute_count = util_last_bit(generic);
      /* bi_should_idvs() honours no_idvs (shader_compile). */
      assert(!s->info.vs.idvs && !s->info.vs.secondary_enable);
   } else {
      s->info.attribute_count = 0;
   }
   /* Images are attributes: behind the 16 vertex attributes in a vertex
    * shader, from 0 elsewhere (panvk_compile_nir). */
   if (s->v7.count[MALI_V7_DESC_TABLE_IMG])
      s->info.attribute_count = s->v7.count[MALI_V7_DESC_TABLE_IMG] +
                                (s->stage == MESA_SHADER_VERTEX ? MALI_MAX_VS_ATTRIBS : 0);
   return VK_SUCCESS;
}
#endif /* PAN_ARCH < 9 */

/* ---------------------------------------------------------------------- */
/* Upload                                                                  */

#if PAN_ARCH >= 9
static enum mali_flush_to_zero_mode
ftz_mode(const struct mali_shader *s)
{
   if (s->info.ftz_fp32)
      return s->info.ftz_fp16 ? MALI_FLUSH_TO_ZERO_MODE_ALWAYS
                              : MALI_FLUSH_TO_ZERO_MODE_DX11;
   /* No "flush FP16, preserve FP32" mode; APIs cannot ask for it. */
   return MALI_FLUSH_TO_ZERO_MODE_PRESERVE_SUBNORMALS;
}

static enum mali_shader_register_allocation
register_allocation(unsigned work_reg_count)
{
   return work_reg_count <= 32 ? MALI_SHADER_REGISTER_ALLOCATION_32_PER_THREAD
                               : MALI_SHADER_REGISTER_ALLOCATION_64_PER_THREAD;
}

static enum mali_shader_stage
spd_stage(mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:
      return MALI_SHADER_STAGE_VERTEX;
   case MESA_SHADER_FRAGMENT:
      return MALI_SHADER_STAGE_FRAGMENT;
   default:
      return MALI_SHADER_STAGE_COMPUTE;
   }
}
#endif /* PAN_ARCH >= 9 */

/*
 * Copy the code into the executable pool and build the SPDs, using
 * panvk's v9+ shapes (the same on v9 and v11); on v7, the Renderer State
 * Descriptors of panvk's PAN_ARCH < 9 path instead. The code is padded to
 * 128 bytes with zeros; kraid and the Bifrost backend pad their output
 * already, the memset covers the rest of the sub-allocation. Both pools
 * are CPU-uncached: no cache maintenance.
 */
#if PAN_ARCH >= 10
/*
 * The command-stream words that copy a fragment shader's pushed
 * uniform-buffer words (fau.ubo_push) into its FAU block when they all
 * come from one buffer: a wait for earlier loads and stores, then per
 * group of runs of adjacent words, the loads into the data registers, a
 * wait, and the stores. The draw prepends the two address moves
 * (mali_cmd_draw.c emit_ubo_push). Registers and offsets are fixed by the
 * shader, so this is built once.
 */
static_assert(MALI_UBO_COPY_SRC == MALI_CS_REG_SCRATCH_START &&
                 MALI_UBO_COPY_DST == MALI_CS_REG_SCRATCH_START + 2 &&
                 MALI_UBO_COPY_DATA == MALI_CS_REG_SCRATCH_START + 4 &&
                 MALI_UBO_COPY_DATA + MALI_UBO_COPY_DATA_REGS <=
                    MALI_CS_REG_SCRATCH_START + MALI_CS_SCRATCH_MEASURE,
              "the uniform-buffer copy uses command scratch registers 0-15");

static void
build_ubo_copy(struct mali_shader *s)
{
   s->ubo_copy.count = 0;
   const uint32_t n = s->fau.ubo_push_count;
   if (s->stage != MESA_SHADER_FRAGMENT || !n)
      return;
   for (uint32_t i = 1; i < n; i++)
      if (s->fau.ubo_push[i].dyn != s->fau.ubo_push[0].dyn)
         return;

   uint64_t *ins = s->ubo_copy.ins;
   unsigned c = 0;
   const uint64_t max = MALI_UBO_COPY_MAX_INS;
   struct mali_cs_wait_packed wait;
   pan_pack(&wait, CS_WAIT, I) {
      I.wait_mask = BITFIELD_BIT(MALI_SB_LS);
   }
   uint64_t wait_word;
   memcpy(&wait_word, &wait, sizeof(wait_word));

   struct {
      unsigned reg, count;
      int dst_off;
   } pend[MALI_UBO_COPY_DATA_REGS];
   unsigned npend = 0, used = 0;

   ins[c++] = wait_word;
   for (uint32_t i = 0; i < n;) {
      unsigned k = 1;
      while (i + k < n && 2 * (k + 1) <= MALI_UBO_COPY_DATA_REGS &&
             s->fau.ubo_push[i + k].offset == s->fau.ubo_push[i].offset + 8 * k)
         k++;
      const unsigned regs = 2 * k;
      if (used + regs > MALI_UBO_COPY_DATA_REGS) {
         if (c + 1 + npend > max)
            return;
         ins[c++] = wait_word;
         for (unsigned p = 0; p < npend; p++) {
            struct mali_cs_store_multiple_packed st;
            pan_pack(&st, CS_STORE_MULTIPLE, I) {
               I.base_register = pend[p].reg;
               I.address = MALI_UBO_COPY_DST;
               I.mask = BITFIELD_MASK(pend[p].count);
               I.offset = pend[p].dst_off;
            }
            memcpy(&ins[c++], &st, sizeof(st));
         }
         npend = used = 0;
      }
      if (c + 1 > max)
         return;
      struct mali_cs_load_multiple_packed ld;
      pan_pack(&ld, CS_LOAD_MULTIPLE, I) {
         I.base_register = MALI_UBO_COPY_DATA + used;
         I.address = MALI_UBO_COPY_SRC;
         I.mask = BITFIELD_MASK(regs);
         I.offset = s->fau.ubo_push[i].offset;
      }
      memcpy(&ins[c++], &ld, sizeof(ld));
      pend[npend].reg = MALI_UBO_COPY_DATA + used;
      pend[npend].count = regs;
      pend[npend].dst_off = (s->fau.ubo_push_start + i) * MALI_FAU_WORD_SIZE;
      npend++;
      used += regs;
      i += k;
   }
   if (c + 1 + npend > max)
      return;
   ins[c++] = wait_word;
   for (unsigned p = 0; p < npend; p++) {
      struct mali_cs_store_multiple_packed st;
      pan_pack(&st, CS_STORE_MULTIPLE, I) {
         I.base_register = pend[p].reg;
         I.address = MALI_UBO_COPY_DST;
         I.mask = BITFIELD_MASK(pend[p].count);
         I.offset = pend[p].dst_off;
      }
      memcpy(&ins[c++], &st, sizeof(st));
   }
   s->ubo_copy.end = s->fau.ubo_push[n - 1].offset + 8;
   s->ubo_copy.count = c;
}
#endif

static VkResult
shader_upload(struct mali_device *dev, struct mali_shader *s)
{
#if PAN_ARCH >= 10
   build_ubo_copy(s);
#else
   s->ubo_copy.count = 0;
#endif

#if PAN_ARCH < 9
   /* The table map, for the descriptor copy jobs (mali_jm_desc_v7.c),
    * whatever the size of the code: a copy job reads it on its own. */
   if (s->v7.map_count) {
      const uint64_t bytes = s->v7.map_count * sizeof(uint32_t);
      if (!s->v7.map ||
          mali_bo_pool_alloc(&dev->desc_pool, bytes, 16, &s->v7.map_mem) != MALI_KBASE_SUCCESS)
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      memcpy(s->v7.map_mem.cpu, s->v7.map, bytes);
   }
#endif

   if (!s->bin_size)
      return VK_SUCCESS;

   /* The blob rejects a stage whose per-thread TLS is over 0x800 16-byte
    * units. */
   if (s->info.tls_size > 0x800 * 16)
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "%s shader needs %u bytes of thread-local storage per thread",
                       mesa_shader_stage_name(s->stage), s->info.tls_size);

   uint64_t padded = align64(s->bin_size, MALI_SHADER_CODE_ALIGN);
   if (mali_bo_pool_alloc(&dev->exec_pool, padded, MALI_SHADER_CODE_ALIGN,
                          &s->code) != MALI_KBASE_SUCCESS)
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   memcpy(s->code.cpu, s->bin, s->bin_size);
   memset((uint8_t *)s->code.cpu + s->bin_size, 0, padded - s->bin_size);

   /* The inline constant pool is addressed as PC + a 32-bit offset. */
   assert(s->code.gpu_va >> 32 == (s->code.gpu_va + padded - 1) >> 32);

#if PAN_ARCH < 9
   /* The constant data common.constant_data points at, 64-byte aligned as
    * in panvk_shader_upload. */
   if (s->v7.data_size) {
      if (mali_bo_pool_alloc(&dev->desc_pool, s->v7.data_size, 64, &s->v7.data_mem) !=
          MALI_KBASE_SUCCESS)
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      memcpy(s->v7.data_mem.cpu, s->v7.data, s->v7.data_size);
   }

   /*
    * v7 has no Shader Program Descriptor: jobs point at a Renderer State
    * Descriptor (panvk_shader_upload at PAN_ARCH < 9). A vertex or compute
    * shader's RSD depends on the shader alone, so it is built here. A
    * compute shader's goes into spd[MALI_SPD_MAIN], the State of the
    * dispatch's Draw section (mali_jm_cmd_dispatch.c); a vertex shader's
    * into both position slots, as there is no point-size variant (the
    * point size goes to its own varying buffer) and no varying shader (no
    * IDVS). A fragment shader's RSD also carries depth/stencil, multisample
    * and pixel-kill state and is followed by the Blend descriptors, so it
    * is built per pipeline and draw (mali_pipeline_state.c,
    * mali_jm_cmd_draw_v7.c); spd[MALI_SPD_MAIN] holds its code address.
    */
   if (s->stage == MESA_SHADER_FRAGMENT) {
      s->spd[MALI_SPD_MAIN] = s->code.gpu_va;
      return VK_SUCCESS;
   }
   if (mali_bo_pool_alloc(&dev->desc_pool, pan_size(RENDERER_STATE),
                          pan_alignment(RENDERER_STATE), &s->spd_mem) != MALI_KBASE_SUCCESS)
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   pan_cast_and_pack(s->spd_mem.cpu, RENDERER_STATE, cfg) {
      mali_v7_rsd_shader(&s->info, s->code.gpu_va, &cfg);
   }
   if (s->stage == MESA_SHADER_COMPUTE) {
      s->spd[MALI_SPD_MAIN] = s->spd_mem.gpu_va;
      return VK_SUCCESS;
   }
   s->spd[MALI_SPD_POS_POINTS] = s->spd_mem.gpu_va;
   s->spd[MALI_SPD_POS_TRIANGLES] = s->spd_mem.gpu_va;
   s->spd[MALI_SPD_VARYING] = 0;
   return VK_SUCCESS;
#else
   const unsigned spd_count = s->stage == MESA_SHADER_VERTEX ? 3 : 1;
   if (mali_bo_pool_alloc(&dev->desc_pool, spd_count * MALI_SPD_SIZE, MALI_SPD_SIZE,
                          &s->spd_mem) != MALI_KBASE_SUCCESS)
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   struct mali_shader_program_packed *spds = s->spd_mem.cpu;
   const uint64_t code = s->code.gpu_va;

   if (s->stage != MESA_SHADER_VERTEX) {
      pan_pack(&spds[0], SHADER_PROGRAM, cfg) {
         cfg.stage = spd_stage(s->stage);
         if (cfg.stage == MALI_SHADER_STAGE_FRAGMENT) {
            cfg.fragment_coverage_bitmask_type = MALI_COVERAGE_BITMASK_TYPE_GL;
            cfg.requires_helper_threads = s->info.contains_barrier;
         }
         cfg.register_allocation = register_allocation(s->info.work_reg_count);
         cfg.binary = code;
         cfg.preload.r48_r63 = s->info.preload >> 48;
         cfg.flush_to_zero_mode = ftz_mode(s);
      }
      s->spd[MALI_SPD_MAIN] = s->spd_mem.gpu_va;
      return VK_SUCCESS;
   }

   /* Vertex: position for points, position for everything else (the
    * point-size write patched out), then the varying shader. */
   const uint64_t pos_offsets[2] = { 0, s->info.vs.no_psiz_offset };
   for (unsigned i = 0; i < 2; i++) {
      pan_pack(&spds[i], SHADER_PROGRAM, cfg) {
         cfg.stage = MALI_SHADER_STAGE_VERTEX;
         cfg.vertex_warp_limit = MALI_WARP_LIMIT_HALF;
         cfg.register_allocation = register_allocation(s->info.work_reg_count);
         cfg.binary = code + pos_offsets[i];
         cfg.preload.r48_r63 = s->info.preload >> 48;
         cfg.flush_to_zero_mode = ftz_mode(s);
      }
      s->spd[MALI_SPD_POS_POINTS + i] = s->spd_mem.gpu_va + i * MALI_SPD_SIZE;
   }

   if (s->info.vs.secondary_enable) {
      pan_pack(&spds[2], SHADER_PROGRAM, cfg) {
         cfg.stage = MALI_SHADER_STAGE_VERTEX;
         cfg.vertex_warp_limit = MALI_WARP_LIMIT_FULL;
         cfg.register_allocation =
            register_allocation(s->info.vs.secondary_work_reg_count);
         cfg.binary = code + s->info.vs.secondary_offset;
         cfg.preload.r48_r63 = s->info.vs.secondary_preload >> 48;
         cfg.flush_to_zero_mode = ftz_mode(s);
      }
      s->spd[MALI_SPD_VARYING] = s->spd_mem.gpu_va + 2 * MALI_SPD_SIZE;
   }

   return VK_SUCCESS;
#endif /* PAN_ARCH >= 9 */
}

/* ---------------------------------------------------------------------- */
/* Object                                                                  */

static struct mali_shader *
shader_alloc(struct mali_device *dev, mesa_shader_stage stage, const void *key,
             size_t key_size)
{
   struct mali_shader *s = vk_zalloc(&dev->vk.alloc, sizeof(*s), 8,
                                     VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!s)
      return NULL;

   assert(key_size == sizeof(s->key));
   memcpy(s->key, key, sizeof(s->key));
   vk_pipeline_cache_object_init(&dev->vk, &s->base, &MALI_PER_ARCH(shader_cache_ops),
                                 s->key, sizeof(s->key));
   s->stage = stage;
   return s;
}

static void
shader_destroy(struct mali_device *dev, struct mali_shader *s)
{
#if PAN_ARCH < 9
   mali_bo_pool_free(&dev->desc_pool, &s->v7.data_mem);
   mali_bo_pool_free(&dev->desc_pool, &s->v7.map_mem);
   free(s->v7.data);
   free(s->v7.map);
#endif
   mali_bo_pool_free(&dev->desc_pool, &s->spd_mem);
   mali_bo_pool_free(&dev->exec_pool, &s->code);
   free(s->bin);
   vk_pipeline_cache_object_finish(&s->base);
   vk_free(&dev->vk.alloc, s);
}

void
MALI_PER_ARCH(shader_unref)(struct mali_device *dev, struct mali_shader *s)
{
   if (s)
      vk_pipeline_cache_object_unref(&dev->vk, &s->base);
}

VkResult
MALI_PER_ARCH(shader_compile)(struct mali_device *dev,
                              const struct mali_shader_compile_info *info,
                              const blake3_hash key, struct mali_shader **out)
{
   const struct mali_physical_device *pdev = mali_device_physical(dev);
   nir_shader *nir = info->nir;
   const mesa_shader_stage stage = nir->info.stage;
   const struct vk_graphics_pipeline_state *state = info->state;
   VkResult result;

   *out = NULL;

   struct mali_shader *s = shader_alloc(dev, stage, key, sizeof(blake3_hash));
   if (!s) {
      ralloc_free(nir);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   nir_variable_mode robust_modes = 0;
   if (info->rs->uniform_buffers != VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT)
      robust_modes |= nir_var_mem_ubo;
   if (info->rs->storage_buffers != VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT)
      robust_modes |= nir_var_mem_ssbo;

   struct pan_compile_inputs inputs = {
      .gpu_id = pdev->props.gpu_id,
      .view_mask = (state && state->rp && state->mv) ? state->mv->view_mask : 0,
      .robust_modes = robust_modes,
      .robust_descriptors = dev->vk.enabled_features.nullDescriptor,
#if PAN_ARCH < 9
      /* One plain vertex shader per Vertex job, no IDVS pair: the v7 draw
       * is a Vertex and a Tiler job (mali_jm_cmd_draw_v7.c), the form
       * Mesa also uses on Bifrost when the shader writes gl_PointSize
       * (bi_should_idvs). */
      .no_idvs = true,
#endif
   };
   struct pan_varying_layout varying_layout;

   switch (stage) {
   case MESA_SHADER_VERTEX:
      s->desc.driver_table_prefix = MALI_MAX_VS_ATTRIBS;
      lower_nir(nir, info, s, false);

      if (inputs.view_mask) {
#if PAN_ARCH < 10
         /* The lowering below relies on the v10+ tiler's view mask; v7 and
          * v9 would need a draw per view (panvk's pre-v10 path), which the
          * job-manager back half does not do. ARMSX2 has no multiview. */
         result = vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                            "multiview pipelines are not supported on v" NIR_STRINGIZE(PAN_ARCH));
         goto fail;
#endif
         nir_lower_multiview_options mv = {
            .view_mask = inputs.view_mask,
            .allowed_per_view_outputs = ~0,
         };
         if (!nir_can_lower_multiview(nir, mv)) {
            result = vk_errorf(dev, VK_ERROR_INVALID_SHADER_NV,
                               "vertex shader cannot be lowered for multiview");
            goto fail;
         }
         NIR_PASS(_, nir, nir_lower_multiview, mv);
         /* Output writes out of the per-view loop, constant offsets. */
         NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries,
                  nir_shader_get_entrypoint(nir), nir_var_shader_out);
         NIR_PASS(_, nir, nir_lower_global_vars_to_local);
         NIR_PASS(_, nir, nir_split_var_copies);
      }

      /* driver_location = attribute location, holes included, so the
       * vertex input state's numbering applies directly. */
      nir_foreach_shader_in_variable(var, nir) {
         assert(var->data.location >= VERT_ATTRIB_GENERIC0 &&
                var->data.location <= VERT_ATTRIB_GENERIC15);
         var->data.driver_location = var->data.location - VERT_ATTRIB_GENERIC0;
      }
      nir_assign_io_var_locations(nir, nir_var_shader_out);
      lower_nir_io(nir);
      NIR_PASS(_, nir, nir_opt_constant_folding);

      pan_varying_collect_formats(&varying_layout, nir, inputs.gpu_id);
      pan_build_varying_layout_compact(&varying_layout, nir, inputs.gpu_id);
      inputs.varying_layout = &varying_layout;
      break;

   case MESA_SHADER_FRAGMENT:
      if (state && state->ms && state->ms->sample_shading_enable)
         nir->info.fs.uses_sample_shading = true;

      /* Input attachments before descriptors. */
      NIR_PASS(_, nir, mali_nir_lower_input_attachment_loads, PAN_ARCH,
               state, &s->fs.input_attachment_read);

#if PAN_ARCH < 9
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, v7_stencil_tile_conv,
               nir_metadata_control_flow, NULL);
      lower_clip_distance_fs(nir, info->vs_varying_layout);
#endif
      /* Input slots first: their count sizes the front of the driver
       * table (mali_shader.h). */
      nir_assign_io_var_locations(nir, nir_var_shader_in);
      s->desc.driver_table_prefix = nir->num_inputs;
      inputs.varying_layout = info->vs_varying_layout;

      lower_nir(nir, info, s, false);

      nir_assign_io_var_locations(nir, nir_var_shader_out);
      lower_nir_io(nir);
      if (!MALI_PER_ARCH(shader_fs_keeps_dual_source)(dev, state) &&
          nir_shader_intrinsics_pass(nir, drop_dual_source_output,
                                     nir_metadata_control_flow, NULL)) {
         nir->info.fs.color_is_dual_source = false;
         NIR_PASS(_, nir, nir_opt_dce);
      }
      /* "if (c) discard;" (ARMSX2's alpha and destination-alpha tests)
       * as demote_if(c): kraid emits one conditional DISCARD instead of a
       * compare, an inversion, a branch and an unconditional DISCARD. Limit
       * 0 flattens only branches that hold nothing but the demote. Then
       * the discards go as early as their inputs allow (the blob does
       * both). */
      const nir_opt_peephole_select_options demote_select = {
         .limit = 0,
         .discard_ok = true,
      };
      NIR_PASS(_, nir, nir_opt_peephole_select, &demote_select);
      NIR_PASS(_, nir, nir_opt_move_discards_to_top);
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

      /* Blend descriptors become driver-provided FAU values
       * (load_blend_descriptor_pan -> fs.blend_descs); on v7 the BLEND
       * instructions read them from the hardware's blend FAU slots
       * (lower_sysvals). */
      const bool atest = fs_needs_atest(nir, state, s->fs.input_attachment_read, info->meta);
      s->fs.no_atest = !atest;
      NIR_PASS(_, nir, pan_nir_lower_fs_outputs, !atest, 0);
      if (atest && !alpha_to_coverage_possible(state) &&
          !(nir->info.outputs_written & (BITFIELD64_BIT(FRAG_RESULT_DEPTH) |
                                         BITFIELD64_BIT(FRAG_RESULT_STENCIL) |
                                         BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK))))
         place_atest_early(nir);
      break;

   case MESA_SHADER_COMPUTE: {
      s->desc.driver_table_prefix = 0;
#if PAN_ARCH >= 9
      nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
      s->info.cs.allow_merging_workgroups = valhall_can_merge_workgroups(nir);
      if (s->info.cs.allow_merging_workgroups) {
         /* Threads of different workgroups may share a subgroup: other
          * divergence options, and the old analysis is stale. */
         nir->options = pan_get_nir_shader_compiler_options(PAN_ARCH,
                                                            MESA_SHADER_COMPUTE, true);
         nir_foreach_function_impl(impl, nir)
            nir_progress(true, impl, ~nir_metadata_divergence);
      }
#else
      /* No workgroup merging on Bifrost: the compiler has no merge_wg
       * options before v9 (pan_get_nir_shader_compiler_options), and
       * panvk gates this on PAN_ARCH >= 9. allow_merging_workgroups stays
       * false. */
#endif
      lower_nir(nir, info, s, s->info.cs.allow_merging_workgroups);
      break;
   }

   default:
      result = vk_errorf(dev, VK_ERROR_INVALID_SHADER_NV, "%s shaders are not supported",
                         mesa_shader_stage_name(stage));
      goto fail;
   }

#if PAN_ARCH < 9
   if (s->v7.failed) {
      result = vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto fail;
   }
#endif

   result = compile_nir(dev, nir, &inputs, state, s);
   if (result != VK_SUCCESS)
      goto fail;

   if (stage == MESA_SHADER_FRAGMENT)
      s->desc.needs_varying_descs = s->info.bifrost.uses_ld_var;

#if PAN_ARCH < 9
   result = v7_fixup_info(dev, nir, s);
   if (result != VK_SUCCESS)
      goto fail;
#endif

   ralloc_free(nir);
   nir = NULL;

   result = shader_upload(dev, s);
   if (result != VK_SUCCESS)
      goto fail;

   *out = s;
   return VK_SUCCESS;

fail:
   ralloc_free(nir);
   shader_destroy(dev, s);
   return result;
}

/* ---------------------------------------------------------------------- */
/* Pipeline cache                                                          */

/*
 * Serialized form, versioned by the pipeline cache UUID (it changes with
 * every build of the driver): stage, the compiler's info block, FAU and
 * descriptor info, stage extras, code. The GPU copy is never serialized;
 * deserializing uploads again.
 */
static bool
shader_serialize(struct vk_pipeline_cache_object *object, struct blob *blob)
{
   const struct mali_shader *s = container_of(object, struct mali_shader, base);

   blob_write_uint32(blob, s->stage);
   blob_write_bytes(blob, &s->info, sizeof(s->info));
   blob_write_bytes(blob, &s->fau, sizeof(s->fau));
   blob_write_bytes(blob, &s->desc, sizeof(s->desc));
   if (s->stage == MESA_SHADER_COMPUTE)
      blob_write_bytes(blob, s->cs.local_size, sizeof(s->cs.local_size));
   else if (s->stage == MESA_SHADER_FRAGMENT) {
      blob_write_uint32(blob, s->fs.input_attachment_read);
      blob_write_uint32(blob, s->fs.no_atest);
   }
   blob_write_uint32(blob, s->bin_size);
   blob_write_bytes(blob, s->bin, s->bin_size);
#if PAN_ARCH < 9
   /* The Bifrost descriptor tables and constant data. */
   blob_write_bytes(blob, s->v7.count, sizeof(s->v7.count));
   blob_write_uint32(blob, s->v7.dyn_ubo_count);
   blob_write_uint32(blob, s->v7.needs_set_addrs);
   blob_write_uint32(blob, s->v7.map_count);
   blob_write_bytes(blob, s->v7.map, s->v7.map_count * sizeof(uint32_t));
   blob_write_uint32(blob, s->v7.data_size);
   blob_write_bytes(blob, s->v7.data, s->v7.data_size);
#endif

   return !blob->out_of_memory;
}

#if PAN_ARCH < 9
/*
 * Whether a deserialized shader's v7 counts are ones a compile produces.
 * The dynamic buffer counts size host arrays (mali_jm_desc_v7.c
 * pack_dyn_ubos, build_dyn_ssbos), the table map has one entry per copied
 * table entry (the descriptor copy job reads that many), and the image
 * table is within MALI_V7_MAX_STAGE_IMG_ENTRIES (v7_fixup_info). The
 * pushed uniform buffer words size the copy table of mali_v7_ubo_copy.
 */
static bool
v7_desc_counts_valid(const struct mali_shader *s)
{
   uint64_t entries = 0;
   for (unsigned t = 0; t < MALI_V7_DESC_TABLE_COUNT; t++)
      entries += s->v7.count[t];

   return s->v7.dyn_ubo_count <= s->desc.dyn_bufs.count &&
          s->v7.dyn_ubo_count <= MALI_MAX_DYNAMIC_UNIFORM_BUFFERS &&
          s->desc.dyn_bufs.count - s->v7.dyn_ubo_count <= MALI_MAX_DYNAMIC_STORAGE_BUFFERS &&
          entries == s->v7.map_count &&
          s->v7.count[MALI_V7_DESC_TABLE_IMG] <= MALI_V7_MAX_STAGE_IMG_ENTRIES &&
          s->fau.ubo_push_count <= MALI_MAX_UBO_PUSH_FAUS;
}
#endif

static struct vk_pipeline_cache_object *
shader_deserialize(struct vk_pipeline_cache *cache, const void *key_data,
                   size_t key_size, struct blob_reader *blob)
{
   struct mali_device *dev = container_of(cache->base.device, struct mali_device, vk);

   if (key_size != sizeof(blake3_hash))
      return NULL;

   mesa_shader_stage stage = blob_read_uint32(blob);
   if (blob->overrun || (stage != MESA_SHADER_VERTEX && stage != MESA_SHADER_FRAGMENT &&
                         stage != MESA_SHADER_COMPUTE))
      return NULL;

   struct mali_shader *s = shader_alloc(dev, stage, key_data, key_size);
   if (!s)
      return NULL;

   blob_copy_bytes(blob, &s->info, sizeof(s->info));
   blob_copy_bytes(blob, &s->fau, sizeof(s->fau));
   blob_copy_bytes(blob, &s->desc, sizeof(s->desc));
   if (stage == MESA_SHADER_COMPUTE)
      blob_copy_bytes(blob, s->cs.local_size, sizeof(s->cs.local_size));
   else if (stage == MESA_SHADER_FRAGMENT) {
      s->fs.input_attachment_read = blob_read_uint32(blob);
      s->fs.no_atest = blob_read_uint32(blob);
   }
   s->bin_size = blob_read_uint32(blob);
   const void *bin = blob_read_bytes(blob, s->bin_size);
#if PAN_ARCH < 9
   blob_copy_bytes(blob, s->v7.count, sizeof(s->v7.count));
   s->v7.dyn_ubo_count = blob_read_uint32(blob);
   s->v7.needs_set_addrs = blob_read_uint32(blob);
   s->v7.map_count = blob_read_uint32(blob);
   const void *map = blob_read_bytes(blob, s->v7.map_count * sizeof(uint32_t));
   s->v7.data_size = blob_read_uint32(blob);
   const void *data = blob_read_bytes(blob, s->v7.data_size);
#endif

   if (blob->overrun || s->info.stage != stage ||
       s->desc.dyn_bufs.count > MALI_MAX_DYNAMIC_BUFFERS)
      goto fail;
#if PAN_ARCH < 9
   if (!v7_desc_counts_valid(s))
      goto fail;
#endif

   if (s->bin_size) {
      s->bin = malloc(s->bin_size);
      if (!s->bin)
         goto fail;
      memcpy(s->bin, bin, s->bin_size);
   }
#if PAN_ARCH < 9
   if (s->v7.map_count) {
      s->v7.map = malloc(s->v7.map_count * sizeof(uint32_t));
      if (!s->v7.map)
         goto fail;
      memcpy(s->v7.map, map, s->v7.map_count * sizeof(uint32_t));
   }
   if (s->v7.data_size) {
      s->v7.data = malloc(s->v7.data_size);
      if (!s->v7.data)
         goto fail;
      memcpy(s->v7.data, data, s->v7.data_size);
   }
#endif

   if (shader_upload(dev, s) != VK_SUCCESS)
      goto fail;

   return &s->base;

fail:
   shader_destroy(dev, s);
   return NULL;
}

static void
shader_cache_destroy(struct vk_device *vk_dev, struct vk_pipeline_cache_object *object)
{
   struct mali_device *dev = container_of(vk_dev, struct mali_device, vk);
   shader_destroy(dev, container_of(object, struct mali_shader, base));
}

const struct vk_pipeline_cache_object_ops MALI_PER_ARCH(shader_cache_ops) = {
   .serialize = shader_serialize,
   .deserialize = shader_deserialize,
   .destroy = shader_cache_destroy,
};

const struct vk_pipeline_cache_object_ops *const MALI_PER_ARCH(pipeline_cache_import_ops)[] = {
   &MALI_PER_ARCH(shader_cache_ops),
   NULL,
};
