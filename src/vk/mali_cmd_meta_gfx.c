/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Internal fragment shaders and their Draw descriptors: attachment
 * preloads (frame shaders), vkCmdClearAttachments (entry point too) and blits.
 *
 * The blob compiles its internal shaders from GLSL ES at run time; panvk
 * builds them in NIR (panvk_vX_cmd_frame_shaders.c, lib/pan_fb_nir.c).
 * Ours are built with
 * nir_builder from a small key and compiled through the normal path
 * (mali_shader_compile), so they use the same FAU and resource-table
 * model as application shaders: table 0 holds a sampler at index 0 and the
 * source textures from index 1, the constants come in push constants, and
 * the BLEND instructions read the blend descriptors' internal words from
 * the fs.blend_descs system values.
 *
 * The Draw descriptors follow panvk's frame-shader DCD (cmd_emit_dcd,
 * v9+): opaque blend descriptors for the targets written, a
 * depth/stencil descriptor that takes depth and stencil from the shader,
 * late ZS update when the shader writes depth or stencil.
 *
 * Built per arch. The v9 Draw descriptor has no Flags 2 (the tile-buffer
 * read and write masks), and takes the FAU count in its own field, as
 * v11's does.
 *
 * v7 (Bifrost) has no resource tables, Shader Environments or frame
 * argument: the shaders use texture i and sampler 0 of the draw's own
 * tables, the Draw section points at a Renderer State Descriptor with the
 * Blend descriptors behind it, and a layered frame shader takes the layer
 * from its push constants, one Draw per layer, as panvk's v7 frame shaders
 * do (panvk_vX_cmd_frame_shaders.c). With no Fullscreen job, in-pass draws
 * (clears) are Tiler jobs drawing a rectangle (mali_jm_cmd_render.c
 * mali_v7_cmd_meta_quads).
 */

#ifndef PAN_ARCH
#error "mali_cmd_meta_gfx.c is built per arch: PAN_ARCH must be set"
#endif

#include "genxml/gen_macros.h"

#if PAN_ARCH >= 10
#include "mali_cmd_buffer.h"
#else
#include "mali_jm.h"
#endif

#include <string.h>

#include "compiler/nir/nir_builder.h"
#include "util/format/u_format.h"
#include "util/mesa-blake3.h"
#include "vk_pipeline.h"

#include "panfrost/compiler/pan_compiler.h"

#include "mali_descriptor_set.h"
#include "mali_image.h"
#include "mali_vk.h"
#if PAN_ARCH < 9
#include "compiler/nir/nir_format_convert.h"
#include "mali_rsd_v7.h"
#include "pan_format.h"
#endif

#define META_FS_CACHE_SIZE 64

struct mali_meta_gfx {
#if PAN_ARCH < 9
   /* v7 reads a texture source, an AFBC plane, only in a copy format, so
    * copy[] holds the raw sources alone ([0][log2 block bytes]) and the
    * copies out of AFBC planes are kept by copy format. */
   struct mali_shader *copy[1][5];
   struct mali_shader *copy_tex_fmt[MALI_META_COPY_FMT_COUNT];
#else
   struct mali_shader *copy[2][5];   /* [source is a texture][log2 block bytes] */
#endif
   struct mali_shader *copy_quad[3]; /* [log2 block bytes] */
   unsigned count;
   struct {
      struct mali_meta_fs_key key;
      struct mali_shader *shader;
   } e[META_FS_CACHE_SIZE];
};

static nir_def *
push_load(nir_builder *b, unsigned comps, unsigned bits, size_t offset)
{
   return nir_load_push_constant(b, comps, bits, nir_imm_int(b, offset), .base = 0,
                                 .range = sizeof(struct mali_meta_fs_push));
}

static nir_alu_type
nir_type(enum mali_meta_fs_type t)
{
   switch (t) {
   case MALI_META_FS_UINT: return nir_type_uint32;
   case MALI_META_FS_SINT: return nir_type_int32;
   default: return nir_type_float32;
   }
}

static const struct glsl_type *
out_type(enum mali_meta_fs_type t)
{
   switch (t) {
   case MALI_META_FS_UINT: return glsl_uvec4_type();
   case MALI_META_FS_SINT: return glsl_ivec4_type();
   default: return glsl_vec4_type();
   }
}

/* A texture instruction reading texture `tex` with sampler 0: entries
 * 1 + tex and 0 of resource table 0, or on v7 entries tex and 0 of the
 * draw's or dispatch's Texture and Sampler tables. */
static nir_def *
tex_op(nir_builder *b, nir_texop op, unsigned tex, nir_def *coord, bool array,
       nir_alu_type type)
{
   nir_tex_instr *t = nir_tex_instr_create(b->shader, 2);
   t->op = op;
   t->sampler_dim = GLSL_SAMPLER_DIM_2D;
   t->is_array = array;
   t->dest_type = type;
#if PAN_ARCH < 9
   /* v7 has no resource tables: texture `tex` and sampler 0 of the draw's
    * tables (panvk_vX_cmd_frame_shaders.c compact_tex_indices). */
   t->texture_index = tex;
   t->sampler_index = 0;
#else
   t->texture_index = pan_res_handle(0, 1 + tex);
   t->sampler_index = pan_res_handle(0, 0);
#endif
   t->coord_components = coord->num_components;
   t->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   t->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, op == nir_texop_txf ? nir_imm_int(b, 0)
                                                                         : nir_imm_float(b, 0.0f));
   nir_def_init(&t->instr, &t->def, 4, 32);
   nir_builder_instr_insert(b, &t->instr);
   return &t->def;
}

static nir_def *
fetch(nir_builder *b, unsigned tex, nir_def *xy, nir_def *layer, nir_alu_type type)
{
   nir_def *c = layer ? nir_vec3(b, nir_channel(b, xy, 0), nir_channel(b, xy, 1), layer) : xy;
   return tex_op(b, nir_texop_txf, tex, c, layer != NULL, type);
}

static void
store_out(nir_builder *b, gl_frag_result loc, const struct glsl_type *type, nir_def *v)
{
   nir_variable *var = nir_variable_create(b->shader, nir_var_shader_out, type, "out");
   var->data.location = loc;
   nir_store_var(b, var, v, BITFIELD_MASK(v->num_components));
}

static nir_def *surface_addr(nir_builder *b, nir_def *base, nir_def *layer_stride,
                             nir_def *row_stride, nir_def *x, nir_def *y, nir_def *z,
                             nir_def *shift, unsigned elem);

/* Components and bit size of a raw load or store of one block. */
static void
raw_block_type(unsigned elem, unsigned *comps, unsigned *bits)
{
   *comps = elem >= 8 ? elem / 4 : 1;
   *bits = elem >= 4 ? 32 : elem * 8;
}

/* A block of 2^elem_log2 bytes as the uvec4 a uint render target of that
 * size takes (R8/R16/R32/R32G32/R32G32B32A32_UINT). */
static nir_def *
raw_to_uvec4(nir_builder *b, nir_def *v)
{
   if (v->bit_size != 32)
      v = nir_u2u32(b, v);
   return nir_pad_vector_imm_int(b, v, 0, 4);
}

/* COPY from a raw surface: the block at (x, y) of layer z. */
static nir_def *
copy_raw_load(nir_builder *b, const struct mali_meta_fs_key *key, nir_def *sxy, nir_def *z)
{
   const unsigned elem = 1u << key->copy_elem_log2;
   unsigned comps, bits;
   raw_block_type(elem, &comps, &bits);
   nir_def *addr = surface_addr(
      b, push_load(b, 1, 64, offsetof(struct mali_meta_fs_push, copy_src)),
      push_load(b, 1, 64, offsetof(struct mali_meta_fs_push, copy_layer_stride)),
      push_load(b, 1, 32, offsetof(struct mali_meta_fs_push, copy_row_stride)),
      nir_channel(b, sxy, 0), nir_channel(b, sxy, 1), z ? z : nir_imm_int(b, 0),
      push_load(b, 1, 32, offsetof(struct mali_meta_fs_push, copy_tile_shift)), elem);
   return raw_to_uvec4(b, nir_load_global(b, comps, bits, addr, .align_mul = MIN2(elem, 16)));
}

/* The BLIT source coordinate: normalized, with the source layer as z for
 * an array. */
static nir_def *
blit_coord(nir_builder *b, nir_def *pos, nir_def *layer)
{
   nir_def *scale = push_load(b, 2, 32, offsetof(struct mali_meta_fs_push, blit_scale));
   nir_def *offset = push_load(b, 2, 32, offsetof(struct mali_meta_fs_push, blit_offset));
   nir_def *uv = nir_ffma(b, pos, scale, offset);
   if (!layer)
      return uv;
   return nir_vec3(b, nir_channel(b, uv, 0), nir_channel(b, uv, 1),
                   nir_fadd(b, nir_u2f32(b, layer),
                            push_load(b, 1, 32, offsetof(struct mali_meta_fs_push, blit_layer))));
}

#if PAN_ARCH < 9
/* A depth or stencil BLIT (v7 only): the source plane (texture tex)
 * sampled at the transformed position inside the area, the target plane
 * (texture tex + 1) outside it. */
static nir_def *
blit_zs(nir_builder *b, unsigned tex, nir_def *pos, nir_def *xy, nir_def *layer,
        nir_def *inside, nir_alu_type type)
{
   nir_def *s = nir_channel(b, tex_op(b, nir_texop_txl, tex, blit_coord(b, pos, layer),
                                      layer != NULL, type), 0);
   nir_def *d = nir_channel(b, fetch(b, tex + 1, xy, layer, type), 0);
   return nir_bcsel(b, inside, s, d);
}
#endif

static nir_shader *
build_fs(struct mali_device *dev, const struct mali_meta_fs_key *key)
{
   nir_builder bld = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, MALI_PER_ARCH(shader_nir_options)(dev, MESA_SHADER_FRAGMENT),
      "mali_meta_fs");
   nir_builder *b = &bld;

   nir_def *fc = nir_load_frag_coord(b);
   nir_def *pos = nir_trim_vector(b, fc, 2);
   nir_def *xy = nir_f2i32(b, pos);
#if PAN_ARCH < 9
   /* v7 has no frame argument: the frame shader Draws of layer l carry l in
    * their push constants (panvk_vX_cmd_frame_shaders.c
    * panvk_fb_sysvals.layer_id). */
   nir_def *layer =
      key->layered ? push_load(b, 1, 32, offsetof(struct mali_meta_fs_push, layer)) : NULL;
#else
   nir_def *layer = key->layered ? nir_load_layer_id(b) : NULL;
#endif

   nir_def *inside = NULL;
   bool need_area = false;
   for (unsigned i = 0; i < 8; i++)
      need_area |= key->rt_op[i] == MALI_META_FS_CLEAR_IN_AREA ||
                   key->rt_op[i] == MALI_META_FS_BLIT ||
                   key->rt_op[i] == MALI_META_FS_COPY;
   need_area |= key->z_op == MALI_META_FS_CLEAR_IN_AREA ||
                key->s_op == MALI_META_FS_CLEAR_IN_AREA;
#if PAN_ARCH < 9
   need_area |= key->z_op == MALI_META_FS_BLIT || key->s_op == MALI_META_FS_BLIT;
#endif
   if (need_area) {
      nir_def *area = push_load(b, 4, 32, offsetof(struct mali_meta_fs_push, area));
      nir_def *x = nir_channel(b, xy, 0), *y = nir_channel(b, xy, 1);
      inside = nir_iand(b, nir_iand(b, nir_ige(b, x, nir_channel(b, area, 0)),
                                    nir_ige(b, y, nir_channel(b, area, 1))),
                        nir_iand(b, nir_ige(b, nir_channel(b, area, 2), x),
                                 nir_ige(b, nir_channel(b, area, 3), y)));
   }

   unsigned tex = 0;
   for (unsigned i = 0; i < 8; i++) {
      const enum mali_meta_fs_op op = key->rt_op[i];
      if (op == MALI_META_FS_NONE)
         continue;
      const nir_alu_type type = nir_type(key->rt_type[i]);
      nir_def *clear = push_load(b, 4, 32, offsetof(struct mali_meta_fs_push, clear_color) + 16 * i);
      nir_def *v;
      switch (op) {
      case MALI_META_FS_LOAD:
         v = fetch(b, tex++, xy, layer, type);
         break;
      case MALI_META_FS_CLEAR:
         v = clear;
         break;
      case MALI_META_FS_CLEAR_IN_AREA:
         v = nir_bcsel(b, inside, clear, fetch(b, tex++, xy, layer, type));
         break;
      case MALI_META_FS_BLIT: {
         nir_def *uv = blit_coord(b, pos, layer);
         nir_def *s = tex_op(b, nir_texop_txl, tex, uv, layer != NULL, type);
         nir_def *d = fetch(b, tex + 1, xy, layer, type);
         tex += 2;
         v = nir_bcsel(b, inside, s, d);
         break;
      }
      case MALI_META_FS_COPY: {
         /* An if, not a select: a raw load outside the area could read
          * anything. */
         nir_def *delta = push_load(b, 2, 32, offsetof(struct mali_meta_fs_push, copy_delta));
         nir_def *sxy = nir_iadd(b, xy, delta);
         nir_push_if(b, inside);
#if PAN_ARCH < 9
         /* v7: an AFBC source is read in its own format and packed into
          * the block's bits; an AFBC target takes them unpacked into its
          * own format (mali_meta_copy_fmt). Exact for these UNORM formats:
          * n / (2^k - 1) converts back to n. */
         nir_def *in;
         if (key->copy_src_fmt) {
            nir_def *t = fetch(b, tex, sxy, layer, nir_type_float32);
            in = raw_to_uvec4(
               b, nir_format_pack_rgba(b, mali_meta_copy_fmt_pipe(key->copy_src_fmt), t));
         } else {
            in = key->copy_src_tex ? fetch(b, tex, sxy, layer, nir_type_uint32)
                                   : copy_raw_load(b, key, sxy, layer);
         }
         if (key->copy_dst_fmt)
            in = nir_format_unpack_rgba(b, nir_channel(b, in, 0),
                                        mali_meta_copy_fmt_pipe(key->copy_dst_fmt));
         nir_push_else(b, NULL);
         nir_def *out = fetch(b, tex + (key->copy_src_tex ? 1 : 0), xy, layer,
                              key->copy_dst_fmt ? nir_type_float32 : nir_type_uint32);
#else
         nir_def *in = key->copy_src_tex ? fetch(b, tex, sxy, layer, nir_type_uint32)
                                         : copy_raw_load(b, key, sxy, layer);
         nir_push_else(b, NULL);
         nir_def *out = fetch(b, tex + (key->copy_src_tex ? 1 : 0), xy, layer,
                              nir_type_uint32);
#endif
         nir_pop_if(b, NULL);
         v = nir_if_phi(b, in, out);
         tex += key->copy_src_tex ? 2 : 1;
         break;
      }
      default:
         UNREACHABLE("meta fs op");
      }
      store_out(b, FRAG_RESULT_DATA0 + i, out_type(key->rt_type[i]), v);
   }

#if PAN_ARCH < 9
   /* v7 also blits depth and stencil (MALI_META_FS_BLIT on z_op / s_op:
    * two textures each, the source plane and the target plane). */
   if (key->z_op == MALI_META_FS_BLIT) {
      store_out(b, FRAG_RESULT_DEPTH, glsl_float_type(),
                blit_zs(b, tex, pos, xy, layer, inside, nir_type_float32));
      tex += 2;
   } else if (key->z_op != MALI_META_FS_NONE) {
#else
   if (key->z_op != MALI_META_FS_NONE) {
#endif
      nir_def *clear = push_load(b, 1, 32, offsetof(struct mali_meta_fs_push, clear_depth));
      nir_def *v = key->z_op == MALI_META_FS_CLEAR ? clear :
                   nir_channel(b, fetch(b, tex++, xy, layer, nir_type_float32), 0);
      if (key->z_op == MALI_META_FS_CLEAR_IN_AREA)
         v = nir_bcsel(b, inside, clear, v);
      store_out(b, FRAG_RESULT_DEPTH, glsl_float_type(), v);
   }
#if PAN_ARCH < 9
   if (key->s_op == MALI_META_FS_BLIT) {
      store_out(b, FRAG_RESULT_STENCIL, glsl_int_type(),
                blit_zs(b, tex, pos, xy, layer, inside, nir_type_uint32));
      tex += 2;
   } else if (key->s_op != MALI_META_FS_NONE) {
#else
   if (key->s_op != MALI_META_FS_NONE) {
#endif
      nir_def *clear = push_load(b, 1, 32, offsetof(struct mali_meta_fs_push, clear_stencil));
      nir_def *v = key->s_op == MALI_META_FS_CLEAR ? clear :
                   nir_channel(b, fetch(b, tex++, xy, layer, nir_type_uint32), 0);
      if (key->s_op == MALI_META_FS_CLEAR_IN_AREA)
         v = nir_bcsel(b, inside, clear, v);
      store_out(b, FRAG_RESULT_STENCIL, glsl_int_type(), v);
   }

   return b->shader;
}

const struct mali_shader *
MALI_PER_ARCH(meta_fs_get)(struct mali_cmd_buffer *cmd, const struct mali_meta_fs_key *key)
{
   struct mali_device *dev = cmd->dev;
   struct mali_shader *s = NULL;

   simple_mtx_lock(&dev->meta_lock);
   if (!dev->meta_gfx)
      dev->meta_gfx = calloc(1, sizeof(struct mali_meta_gfx));
   struct mali_meta_gfx *mg = dev->meta_gfx;
   if (mg) {
      for (unsigned i = 0; i < mg->count; i++) {
         if (!memcmp(&mg->e[i].key, key, sizeof(*key))) {
            s = mg->e[i].shader;
            break;
         }
      }
   }
   if (mg && !s && mg->count < META_FS_CACHE_SIZE) {
      nir_shader *nir = build_fs(dev, key);
      MALI_PER_ARCH(shader_preprocess)(dev, nir);

      const struct vk_pipeline_robustness_state rs = {
         .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
         .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
         .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
         .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
      };
      const struct mali_shader_compile_info info = {
         .nir = nir,
         .rs = &rs,
         .meta = true,
      };
      blake3_hash hash;
      struct mesa_blake3 h;
      _mesa_blake3_init(&h);
      _mesa_blake3_update(&h, "mali-meta-fs", 12);
      _mesa_blake3_update(&h, key, sizeof(*key));
      _mesa_blake3_final(&h, hash);

      if (MALI_PER_ARCH(shader_compile)(dev, &info, hash, &s) == VK_SUCCESS) {
         mg->e[mg->count].key = *key;
         mg->e[mg->count].shader = s;
         mg->count++;
      } else {
         s = NULL;
      }
   }
   simple_mtx_unlock(&dev->meta_lock);

   if (!s)
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   return s;
}

void
MALI_PER_ARCH(meta_gfx_finish)(struct mali_device *dev)
{
   struct mali_meta_gfx *mg = dev->meta_gfx;
   if (!mg)
      return;
   for (unsigned t = 0; t < ARRAY_SIZE(mg->copy); t++) {
      for (unsigned i = 0; i < ARRAY_SIZE(mg->copy[t]); i++)
         MALI_PER_ARCH(shader_unref)(dev, mg->copy[t][i]);
   }
   for (unsigned i = 0; i < ARRAY_SIZE(mg->copy_quad); i++)
      MALI_PER_ARCH(shader_unref)(dev, mg->copy_quad[i]);
   for (unsigned i = 0; i < mg->count; i++)
      MALI_PER_ARCH(shader_unref)(dev, mg->e[i].shader);
#if PAN_ARCH < 9
   for (unsigned i = 0; i < ARRAY_SIZE(mg->copy_tex_fmt); i++)
      MALI_PER_ARCH(shader_unref)(dev, mg->copy_tex_fmt[i]);
#endif
   free(mg);
   dev->meta_gfx = NULL;
}

bool
MALI_PER_ARCH(cmd_pack_plane_texture)(struct mali_cmd_buffer *cmd, const struct mali_image *image,
                                      unsigned plane, enum pipe_format format, unsigned level,
                                      unsigned first_layer, unsigned layer_count, uint32_t out[8])
{
   struct mali_ptr p =
      mali_cmd_alloc(cmd, mali_image_plane_texture_desc_size(image, layer_count), 64);
   if (!p.cpu)
      return false;
   MALI_PER_ARCH(image_pack_plane_texture)(image, plane, format, level, first_layer, layer_count,
                                           p.cpu, p.gpu, out);
   return true;
}

#if PAN_ARCH < 9
/*
 * Bifrost (v7) Draw sections for the internal fragment shaders, following
 * panvk's v7 frame-shader DCD (panvk_vX_cmd_frame_shaders.c cmd_emit_dcd)
 * and gallium's preload (pan_fb_preload.c pan_preload_emit_dcd). A frame
 * shader draws a full-framebuffer rectangle scissored to the pass's
 * bounding box grown to 32-pixel tiles, with Clean Fragment Write so that
 * tiles it only preloads stay clean; an in-pass draw gets its rectangle
 * and scissor per job (mali_v7_cmd_meta_quads).
 */

/* Opaque for the targets written, off for the rest (panvk fill_bds, gallium
 * pan_preload_emit_blend). */
static void
v7_meta_blends(const struct mali_render_state *r, const struct mali_meta_fs_key *key, void *out)
{
   for (unsigned i = 0; i < r->rt_count; i++) {
      void *o = (uint8_t *)out + i * pan_size(BLEND);
      const bool written = i < MALI_MAX_RTS && key->rt_op[i] != MALI_META_FS_NONE &&
                           i < r->desc.rt_count && r->desc.rt[i].image;
      if (!written) {
         pan_cast_and_pack(o, BLEND, cfg) {
            cfg.enable = false;
            cfg.internal.mode = MALI_BLEND_MODE_OFF;
            cfg.internal.fixed_function.num_comps = 4;
         }
         continue;
      }
      /* The opaque Blend every arch packs, with the register format of
       * the shader's output type, which v7's conversion carries. */
      struct mali_blend_packed b, conv;
      MALI_PER_ARCH(pack_opaque_blend)(r->desc.rt[i].format, i, b.opaque);
      pan_pack(&conv, BLEND, cfg) {
         cfg.internal.fixed_function.conversion.register_format =
            mali_v7_register_format(nir_type(key->rt_type[i]));
      }
      pan_merge(&b, &conv, BLEND);
      memcpy(o, &b, sizeof(b));
   }
}

/* pan_preload_emit_rsd at PAN_ARCH 7, with the blends behind it. The
 * texture and sampler counts are this draw's: the shader info counts
 * descriptor-set tables only. */
static uint64_t
v7_meta_rsd(struct mali_cmd_buffer *cmd, const struct mali_shader *fs,
            const struct mali_meta_fs_key *key, unsigned texture_count, bool frame_shader)
{
   const struct mali_render_state *r = &cmd->gfx.render;
   struct mali_ptr m = mali_cmd_alloc(cmd, pan_size(RENDERER_STATE) + r->rt_count * pan_size(BLEND),
                                      pan_alignment(RENDERER_STATE));
   if (!m.cpu)
      return 0;

   const bool zs = key->z_op != MALI_META_FS_NONE || key->s_op != MALI_META_FS_NONE;
   pan_cast_and_pack(m.cpu, RENDERER_STATE, cfg) {
      mali_v7_rsd_shader(&fs->info, fs->spd[MALI_SPD_MAIN], &cfg);
      cfg.shader.texture_count = texture_count;
      cfg.shader.sampler_count = 1;

      cfg.multisample_misc.sample_mask = 0xffff;
      cfg.multisample_misc.multisample_enable = false;
      cfg.multisample_misc.depth_function = MALI_FUNC_ALWAYS;
      cfg.multisample_misc.depth_write_mask = key->z_op != MALI_META_FS_NONE;

      cfg.stencil_mask_misc.stencil_enable = key->s_op != MALI_META_FS_NONE;
      cfg.stencil_mask_misc.stencil_mask_front = 0xff;
      cfg.stencil_mask_misc.stencil_mask_back = 0xff;
      cfg.stencil_front.compare_function = MALI_FUNC_ALWAYS;
      cfg.stencil_front.stencil_fail = MALI_STENCIL_OP_REPLACE;
      cfg.stencil_front.depth_fail = MALI_STENCIL_OP_REPLACE;
      cfg.stencil_front.depth_pass = MALI_STENCIL_OP_REPLACE;
      cfg.stencil_front.mask = 0xff;
      cfg.stencil_back = cfg.stencil_front;

      if (zs) {
         /* ZS_EMIT needs late update and kill. */
         cfg.properties.zs_update_operation = MALI_PIXEL_KILL_FORCE_LATE;
         cfg.properties.pixel_kill_operation = MALI_PIXEL_KILL_FORCE_LATE;
      } else {
         /* Skipping ATEST needs forced early depth/stencil. */
         cfg.properties.zs_update_operation = MALI_PIXEL_KILL_FORCE_EARLY;
         cfg.properties.pixel_kill_operation = MALI_PIXEL_KILL_FORCE_EARLY;
      }
      cfg.properties.allow_forward_pixel_to_kill = frame_shader && !zs;
   }
   v7_meta_blends(r, key, (uint8_t *)m.cpu + pan_size(RENDERER_STATE));
   return m.gpu;
}

/* A Viewport descriptor whose scissor is [minx, maxx] x [miny, maxy]
 * (inclusive); depth range 0..1 (the genxml defaults). The frame shaders
 * and the in-pass draws (mali_v7_cmd_meta_quads) use it. 0 on failure. */
uint64_t
mali_v7_meta_viewport(struct mali_cmd_buffer *cmd, uint32_t minx, uint32_t miny, uint32_t maxx,
                      uint32_t maxy)
{
   struct mali_ptr p = mali_cmd_alloc(cmd, pan_size(VIEWPORT), pan_alignment(VIEWPORT));
   if (!p.cpu)
      return 0;
   pan_cast_and_pack(p.cpu, VIEWPORT, cfg) {
      cfg.scissor_minimum_x = MIN2(minx, UINT16_MAX);
      cfg.scissor_minimum_y = MIN2(miny, UINT16_MAX);
      cfg.scissor_maximum_x = MIN2(maxx, UINT16_MAX);
      cfg.scissor_maximum_y = MIN2(maxy, UINT16_MAX);
   }
   return p.gpu;
}

/* Four screen-space positions (x, y, z, w) of the rectangle [x0, x1) x
 * [y0, y1), the vertices of a 4-vertex triangle strip (gallium
 * GENX(pan_preload_fb)). 0 on failure. */
uint64_t
mali_v7_meta_rect_positions(struct mali_cmd_buffer *cmd, float x0, float y0, float x1, float y1)
{
   const float rect[16] = {
      x0, y0, 0.0f, 1.0f,
      x1, y0, 0.0f, 1.0f,
      x0, y1, 0.0f, 1.0f,
      x1, y1, 0.0f, 1.0f,
   };
   struct mali_ptr p = mali_cmd_alloc(cmd, sizeof(rect), 64);
   if (!p.cpu)
      return 0;
   memcpy(p.cpu, rect, sizeof(rect));
   return p.gpu;
}
#endif /* PAN_ARCH < 9 */

bool
MALI_PER_ARCH(meta_fs_dcd)(struct mali_cmd_buffer *cmd, const struct mali_shader *fs,
                           const struct mali_meta_fs_key *key, const struct mali_meta_fs_push *push,
                           const uint32_t (*textures)[8], unsigned texture_count,
                           const uint32_t *sampler, bool frame_shader, void *out)
{
#if PAN_ARCH < 9
   struct mali_render_state *r = &cmd->gfx.render;
   assert(r->tsd);

   r->tls_size = MAX2(r->tls_size, fs->info.tls_size);

   const uint64_t rsd = v7_meta_rsd(cmd, fs, key, texture_count, frame_shader);
   /* Tables 64-byte aligned, as mali_jm_desc_v7.c allocates them. */
   struct mali_ptr samp = mali_cmd_alloc(cmd, pan_size(SAMPLER), 64);
   struct mali_ptr tex = {0};
   if (texture_count)
      tex = mali_cmd_alloc(cmd, texture_count * pan_size(TEXTURE), 64);
   if (!rsd || !samp.cpu || (texture_count && !tex.cpu))
      return false;
   if (sampler) {
      memcpy(samp.cpu, sampler, pan_size(SAMPLER));
   } else {
      pan_cast_and_pack(samp.cpu, SAMPLER, cfg) {
         cfg.seamless_cube_map = false;
         cfg.normalized_coordinates = false;
         cfg.clamp_integer_array_indices = false;
         cfg.minify_nearest = true;
         cfg.magnify_nearest = true;
      }
   }
   for (unsigned i = 0; i < texture_count; i++)
      memcpy((uint8_t *)tex.cpu + i * pan_size(TEXTURE), textures[i], pan_size(TEXTURE));

   uint64_t position = 0, viewport = 0;
   if (frame_shader) {
      const uint32_t minx = ROUND_DOWN_TO(r->minx, 32), miny = ROUND_DOWN_TO(r->miny, 32);
      const uint32_t maxx = MIN2(ALIGN_POT(r->maxx + 1, 32), r->desc.width) - 1;
      const uint32_t maxy = MIN2(ALIGN_POT(r->maxy + 1, 32), r->desc.height) - 1;
      viewport = mali_v7_meta_viewport(cmd, minx, miny, maxx, maxy);
      position = mali_v7_meta_rect_positions(cmd, 0.0f, 0.0f, (float)r->desc.width,
                                             (float)r->desc.height);
      if (!viewport || !position)
         return false;
   }

   /* No fs.blend_descs on v7: BLEND reads the Blend descriptors behind the
    * RSD through the blend FAU slots (bifrost_compile.c BIR_FAU_BLEND_0). */
   struct mali_graphics_sysvals sv;
   memset(&sv, 0, sizeof(sv));
   memset(sv.iam, 0xff, sizeof(sv.iam));

   const unsigned layers = frame_shader ? MAX2(r->desc.layer_count, 1) : 1;
   struct mali_meta_fs_push lp = *push;
   for (unsigned l = 0; l < layers; l++) {
      lp.layer = l;
      const uint64_t fau = MALI_PER_ARCH(cmd_gfx_fau)(cmd, fs, &sv, &lp, sizeof(lp));
      if (fs->fau.total_count && !fau)
         return false;
      /* The RSD's preload.uniform_count is the shader's FAU word count;
       * the block built here has as many words. */
      assert((fau >> 56) == DIV_ROUND_UP(fs->info.fau.count, 2));

      struct mali_draw_packed dcd;
      pan_pack(&dcd, DRAW, cfg) {
         cfg.state = rsd;
         cfg.textures = texture_count ? tex.gpu : 0;
         cfg.samplers = samp.gpu;
         cfg.push_uniforms = fau & BITFIELD64_MASK(56);
         cfg.thread_storage = r->tsd;
         cfg.position = position;
         cfg.viewport = viewport;
         cfg.occlusion_query = MALI_OCCLUSION_MODE_DISABLED;
         cfg.clean_fragment_write = frame_shader;
      }
      memcpy((uint8_t *)out + (uint64_t)l * MALI_FRAME_SHADER_DCDS * pan_size(DRAW), &dcd,
             sizeof(dcd));
   }
   return true;
#else
   struct mali_render_state *r = &cmd->gfx.render;
   assert(r->tsd);

   r->tls_size = MAX2(r->tls_size, fs->info.tls_size);

   /* Table 0: the sampler, then the textures. */
   const unsigned drv_count = 1 + texture_count;
   struct mali_ptr drv = mali_cmd_alloc(cmd, drv_count * MALI_DESCRIPTOR_SIZE,
                                        MALI_DESCRIPTOR_SIZE);
   struct mali_ptr table = mali_cmd_alloc(cmd, 4 * MALI_RESOURCE_SIZE, 64);
   if (!drv.cpu || !table.cpu)
      return false;
   if (sampler) {
      memcpy(drv.cpu, sampler, MALI_DESCRIPTOR_SIZE);
   } else {
      pan_cast_and_pack(drv.cpu, SAMPLER, cfg) {
         cfg.seamless_cube_map = false;
         cfg.normalized_coordinates = false;
         cfg.clamp_integer_array_indices = false;
         cfg.minify_nearest = true;
         cfg.magnify_nearest = true;
      }
   }
   for (unsigned i = 0; i < texture_count; i++)
      memcpy((uint8_t *)drv.cpu + (1 + i) * MALI_DESCRIPTOR_SIZE, textures[i],
             MALI_DESCRIPTOR_SIZE);
   memset(table.cpu, 0, 4 * MALI_RESOURCE_SIZE);
   pan_cast_and_pack(table.cpu, RESOURCE, cfg) {
      cfg.address = drv.gpu;
      cfg.size = drv_count * MALI_DESCRIPTOR_SIZE;
      cfg.contains_descriptors = true;
   }

   /* Blend: opaque for the targets written, off for the rest. */
   uint32_t written = 0;
   for (unsigned i = 0; i < 8; i++) {
      if (key->rt_op[i] != MALI_META_FS_NONE)
         written |= BITFIELD_BIT(i);
   }
   const bool zs = key->z_op != MALI_META_FS_NONE || key->s_op != MALI_META_FS_NONE;

   struct mali_graphics_sysvals sv;
   memset(&sv, 0, sizeof(sv));
   memset(sv.iam, 0xff, sizeof(sv.iam));

   uint64_t blend = 0;
   unsigned blend_count = 0;
   if (written) {
      blend_count = r->rt_count;
      struct mali_ptr bl = mali_cmd_alloc(cmd, blend_count * pan_size(BLEND), 64);
      if (!bl.cpu)
         return false;
      for (unsigned i = 0; i < blend_count; i++) {
         uint32_t *w = (uint32_t *)((uint8_t *)bl.cpu + i * pan_size(BLEND));
         if ((written & BITFIELD_BIT(i)) && i < r->desc.rt_count && r->desc.rt[i].image) {
            MALI_PER_ARCH(pack_opaque_blend)(r->desc.rt[i].format, i, w);
         } else {
            pan_cast_and_pack(w, BLEND, cfg) {
               cfg.enable = false;
               cfg.internal.mode = MALI_BLEND_MODE_OFF;
            }
         }
         sv.fs.blend_descs[i] = w[2] | (uint64_t)w[3] << 32;
      }
      blend = bl.gpu;
   }

   uint64_t fau = MALI_PER_ARCH(cmd_gfx_fau)(cmd, fs, &sv, push, sizeof(*push));
   if (fs->fau.total_count && !fau)
      return false;

   struct mali_ptr zsd = mali_cmd_alloc(cmd, pan_size(DEPTH_STENCIL), 32);
   if (!zsd.cpu)
      return false;
   const bool wz = key->z_op != MALI_META_FS_NONE;
   const bool ws = key->s_op != MALI_META_FS_NONE;
   pan_cast_and_pack(zsd.cpu, DEPTH_STENCIL, cfg) {
      cfg.depth_function = MALI_FUNC_ALWAYS;
      cfg.depth_write_enable = wz;
      if (wz)
         cfg.depth_source = MALI_DEPTH_SOURCE_SHADER;
      cfg.stencil_test_enable = ws;
      cfg.stencil_from_shader = ws;
      cfg.front_compare_function = MALI_FUNC_ALWAYS;
      cfg.front_stencil_fail = MALI_STENCIL_OP_REPLACE;
      cfg.front_depth_fail = MALI_STENCIL_OP_REPLACE;
      cfg.front_depth_pass = MALI_STENCIL_OP_REPLACE;
      cfg.front_write_mask = 0xff;
      cfg.front_value_mask = 0xff;
      cfg.back_compare_function = MALI_FUNC_ALWAYS;
      cfg.back_stencil_fail = MALI_STENCIL_OP_REPLACE;
      cfg.back_depth_fail = MALI_STENCIL_OP_REPLACE;
      cfg.back_depth_pass = MALI_STENCIL_OP_REPLACE;
      cfg.back_write_mask = 0xff;
      cfg.back_value_mask = 0xff;
      cfg.depth_cull_enable = false;
   }

   pan_cast_and_pack(out, DRAW, cfg) {
      /* ZS emit needs late update and kill; without it, early (panvk). */
      cfg.flags_0.zs_update_operation = zs ? MALI_PIXEL_KILL_FORCE_LATE
                                           : MALI_PIXEL_KILL_FORCE_EARLY;
      cfg.flags_0.pixel_kill_operation = zs ? MALI_PIXEL_KILL_FORCE_LATE
                                            : MALI_PIXEL_KILL_FORCE_EARLY;
      cfg.flags_0.allow_forward_pixel_to_kill = frame_shader && !zs;
      cfg.flags_0.allow_forward_pixel_to_be_killed = true;
      cfg.flags_0.clean_fragment_write = frame_shader;
      cfg.flags_0.multisample_enable = false;
      cfg.flags_0.occlusion_query = MALI_OCCLUSION_MODE_DISABLED;
      cfg.flags_1.sample_mask = 0xffff;
      cfg.flags_1.render_target_mask = written;
#if PAN_ARCH >= 10
      cfg.flags_2.write_mask = written;
      cfg.flags_2.read_mask = 0;
      cfg.flags_2.no_shader_depth_read = true;
      cfg.flags_2.no_shader_stencil_read = true;
#else
      /* A full-screen job's draw takes its vertex packet as a Malloc
       * Vertex draw does (Mesa's v9 jm_emit_tiler_draw); frame shaders
       * have none. */
      cfg.vertex_array.packet = !frame_shader;
#endif
      cfg.blend = blend;
      cfg.blend_count = blend_count;
      cfg.depth_stencil = zsd.gpu;
      cfg.maximum_z = 1.0f;
      cfg.shader.resources = table.gpu | 4;
      cfg.shader.shader = fs->spd[MALI_SPD_MAIN];
      cfg.shader.thread_storage = r->tsd;
      cfg.shader.fau = fau & BITFIELD64_MASK(56);
      cfg.shader.fau_count = fs->fau.total_count;
   }
   return true;
#endif /* PAN_ARCH >= 9 */
}

/* ---------------------------------------------------------------------- */
/* Raw block copies                                                        */

/* Bits k of v moved to bit 2k (4 bits). */
static nir_def *
spread4(nir_builder *b, nir_def *v)
{
   nir_def *r = nir_iand_imm(b, v, 1);
   r = nir_ior(b, r, nir_ishl_imm(b, nir_iand_imm(b, v, 2), 1));
   r = nir_ior(b, r, nir_ishl_imm(b, nir_iand_imm(b, v, 4), 2));
   return nir_ior(b, r, nir_ishl_imm(b, nir_iand_imm(b, v, 8), 3));
}

#define COPY_PUSH(field) offsetof(struct mali_meta_copy_push, field)

static nir_def *
copy_push(nir_builder *b, unsigned bits, size_t off)
{
   return nir_load_push_constant(b, 1, bits, nir_imm_int(b, off), .base = 0,
                                 .range = sizeof(struct mali_meta_copy_push));
}

/*
 * Byte address of block (x, y) of layer z. Within a u-interleaved tile the
 * block index is spread(y) * 3 ^ spread(x) (pan_tiling.c: bit_duplication
 * and space_4); with a tile shift of 0 the formula is the linear one.
 */
static nir_def *
surface_addr(nir_builder *b, nir_def *base, nir_def *layer_stride, nir_def *row_stride,
             nir_def *x, nir_def *y, nir_def *z, nir_def *shift, unsigned elem)
{
   nir_def *mask = nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), shift), -1);
   nir_def *tx = nir_ushr(b, x, shift);
   nir_def *ty = nir_ushr(b, y, shift);
   nir_def *idx = nir_ixor(b, nir_imul_imm(b, spread4(b, nir_iand(b, y, mask)), 3),
                           spread4(b, nir_iand(b, x, mask)));
   nir_def *blocks = nir_iadd(b, nir_ishl(b, tx, nir_ishl_imm(b, shift, 1)), idx);
   nir_def *off = nir_iadd(b, nir_imul(b, nir_u2u64(b, ty), nir_u2u64(b, row_stride)),
                           nir_u2u64(b, nir_imul_imm(b, blocks, elem)));
   off = nir_iadd(b, off, nir_imul(b, nir_u2u64(b, z), layer_stride));
   return nir_iadd(b, base, off);
}

static nir_shader *
build_copy(struct mali_device *dev, unsigned elem_log2, bool src_tex)
{
   const unsigned elem = 1u << elem_log2;
   unsigned comps, bits;
   raw_block_type(elem, &comps, &bits);

   nir_builder bld = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, MALI_PER_ARCH(shader_nir_options)(dev, MESA_SHADER_COMPUTE),
      "mali_meta_copy_%s%u", src_tex ? "tex" : "image", elem);
   nir_builder *b = &bld;
   b->shader->info.workgroup_size[0] = MALI_META_COPY_WG;
   b->shader->info.workgroup_size[1] = MALI_META_COPY_WG;
   b->shader->info.workgroup_size[2] = 1;

   nir_def *id = nir_load_global_invocation_id(b, 32);
   nir_def *gx = nir_channel(b, id, 0), *gy = nir_channel(b, id, 1), *gz = nir_channel(b, id, 2);
   nir_def *w = copy_push(b, 32, COPY_PUSH(width));
   nir_def *h = copy_push(b, 32, COPY_PUSH(height));

   nir_push_if(b, nir_iand(b, nir_ult(b, gx, w), nir_ult(b, gy, h)));
   {
      nir_def *sx = nir_iadd(b, gx, copy_push(b, 32, COPY_PUSH(src_x)));
      nir_def *sy = nir_iadd(b, gy, copy_push(b, 32, COPY_PUSH(src_y)));
      nir_def *v;
      if (src_tex) {
         /* txf in the uint format of the block size: the block's bits in
          * the first components. */
         v = fetch(b, 0, nir_vec2(b, sx, sy), gz, nir_type_uint32);
         v = nir_trim_vector(b, v, comps);
         if (bits < 32)
            v = nir_u2uN(b, v, bits);
      } else {
         nir_def *sa = surface_addr(
            b, copy_push(b, 64, COPY_PUSH(src)), copy_push(b, 64, COPY_PUSH(src_layer_stride)),
            copy_push(b, 32, COPY_PUSH(src_row_stride)), sx, sy, gz,
            copy_push(b, 32, COPY_PUSH(src_tile_shift)), elem);
         v = nir_load_global(b, comps, bits, sa, .align_mul = MIN2(elem, 16));
      }
      nir_def *da = surface_addr(
         b, copy_push(b, 64, COPY_PUSH(dst)), copy_push(b, 64, COPY_PUSH(dst_layer_stride)),
         copy_push(b, 32, COPY_PUSH(dst_row_stride)),
         nir_iadd(b, gx, copy_push(b, 32, COPY_PUSH(dst_x))),
         nir_iadd(b, gy, copy_push(b, 32, COPY_PUSH(dst_y))), gz,
         copy_push(b, 32, COPY_PUSH(dst_tile_shift)), elem);
      nir_store_global(b, v, da, .write_mask = BITFIELD_MASK(comps),
                       .align_mul = MIN2(elem, 16));
   }
   nir_pop_if(b, NULL);
   return b->shader;
}

/*
 * Linear -> u-interleaved, a 4x4 group of blocks per invocation. Within a
 * tile, index bit 2i is x_i ^ y_i and bit 2i+1 is y_i, so the group at an
 * aligned (x, y) fills indices 0..15 of its run: block k of the run is at
 * y0 = k1, x0 = k0 ^ y0, y1 = k3, x1 = k2 ^ y1.
 */
static nir_shader *
build_copy_quad(struct mali_device *dev, unsigned elem_log2)
{
   const unsigned elem = 1u << elem_log2;
   nir_builder bld = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, MALI_PER_ARCH(shader_nir_options)(dev, MESA_SHADER_COMPUTE),
      "mali_meta_copy_quad%u", elem);
   nir_builder *b = &bld;
   b->shader->info.workgroup_size[0] = MALI_META_COPY_WG;
   b->shader->info.workgroup_size[1] = MALI_META_COPY_WG;
   b->shader->info.workgroup_size[2] = 1;

   nir_def *id = nir_load_global_invocation_id(b, 32);
   nir_def *gx = nir_channel(b, id, 0), *gy = nir_channel(b, id, 1), *gz = nir_channel(b, id, 2);
   nir_def *w = copy_push(b, 32, COPY_PUSH(width));
   nir_def *h = copy_push(b, 32, COPY_PUSH(height));

   nir_push_if(b, nir_iand(b, nir_ult(b, gx, w), nir_ult(b, gy, h)));
   {
      nir_def *qx = nir_ishl_imm(b, gx, 2), *qy = nir_ishl_imm(b, gy, 2);
      nir_def *sx = nir_iadd(b, qx, copy_push(b, 32, COPY_PUSH(src_x)));
      nir_def *sy = nir_iadd(b, qy, copy_push(b, 32, COPY_PUSH(src_y)));
      nir_def *src_row = copy_push(b, 32, COPY_PUSH(src_row_stride));
      /* Row (sy + r) of the source starts at row0 + r * src_row. */
      nir_def *row0 = surface_addr(
         b, copy_push(b, 64, COPY_PUSH(src)), copy_push(b, 64, COPY_PUSH(src_layer_stride)),
         src_row, sx, sy, gz, nir_imm_int(b, 0), elem);
      nir_def *src_row64 = nir_u2u64(b, src_row);

      nir_def *words[16] = {0};
      const unsigned per_word = 4 / elem;
      for (unsigned k = 0; k < 16; k++) {
         const unsigned y0 = (k >> 1) & 1, x0 = (k & 1) ^ y0;
         const unsigned y1 = (k >> 3) & 1, x1 = ((k >> 2) & 1) ^ y1;
         const unsigned lx = x0 | (x1 << 1), ly = y0 | (y1 << 1);
         nir_def *a = nir_iadd(b, row0, nir_iadd_imm(b, nir_imul_imm(b, src_row64, ly), lx * elem));
         nir_def *v = nir_load_global(b, 1, elem * 8, a, .align_mul = elem);
         if (elem < 4)
            v = nir_u2u32(b, v);
         const unsigned wi = k / per_word, sh = (k % per_word) * elem * 8;
         if (sh)
            v = nir_ishl_imm(b, v, sh);
         words[wi] = words[wi] ? nir_ior(b, words[wi], v) : v;
      }

      nir_def *da = surface_addr(
         b, copy_push(b, 64, COPY_PUSH(dst)), copy_push(b, 64, COPY_PUSH(dst_layer_stride)),
         copy_push(b, 32, COPY_PUSH(dst_row_stride)),
         nir_iadd(b, qx, copy_push(b, 32, COPY_PUSH(dst_x))),
         nir_iadd(b, qy, copy_push(b, 32, COPY_PUSH(dst_y))), gz,
         copy_push(b, 32, COPY_PUSH(dst_tile_shift)), elem);
      const unsigned nwords = 4 * elem;
      for (unsigned i = 0; i < nwords; i += 4) {
         nir_def *v = nir_vec4(b, words[i], words[i + 1], words[i + 2], words[i + 3]);
         nir_store_global(b, v, nir_iadd_imm(b, da, i * 4), .write_mask = 0xf, .align_mul = 16);
      }
   }
   nir_pop_if(b, NULL);
   return b->shader;
}

static const struct mali_shader *
copy_get(struct mali_cmd_buffer *cmd, unsigned elem_log2, bool src_tex)
{
   struct mali_device *dev = cmd->dev;
   struct mali_shader *s = NULL;

   assert(elem_log2 < 5);
#if PAN_ARCH < 9
   /* v7 has no texture-source slots here (struct mali_meta_gfx). */
   assert(!src_tex);
#endif
   simple_mtx_lock(&dev->meta_lock);
   if (!dev->meta_gfx)
      dev->meta_gfx = calloc(1, sizeof(struct mali_meta_gfx));
   struct mali_meta_gfx *mg = dev->meta_gfx;
   if (mg) {
      s = mg->copy[src_tex][elem_log2];
      if (!s) {
         nir_shader *nir = build_copy(dev, elem_log2, src_tex);
         MALI_PER_ARCH(shader_preprocess)(dev, nir);
         const struct vk_pipeline_robustness_state rs = {
            .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
            .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
            .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
            .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
         };
         const struct mali_shader_compile_info info = {.nir = nir, .rs = &rs};
         blake3_hash hash;
         struct mesa_blake3 h;
         _mesa_blake3_init(&h);
         _mesa_blake3_update(&h, src_tex ? "mali-meta-copy-tex" : "mali-meta-copy-image",
                             src_tex ? 18 : 20);
         _mesa_blake3_update(&h, &elem_log2, sizeof(elem_log2));
         _mesa_blake3_final(&h, hash);
         if (MALI_PER_ARCH(shader_compile)(dev, &info, hash, &s) != VK_SUCCESS)
            s = NULL;
         mg->copy[src_tex][elem_log2] = s;
      }
   }
   simple_mtx_unlock(&dev->meta_lock);

   if (!s)
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   return s;
}

const struct mali_shader *
MALI_PER_ARCH(meta_copy_get)(struct mali_cmd_buffer *cmd, unsigned elem_log2)
{
   return copy_get(cmd, elem_log2, false);
}

const struct mali_shader *
MALI_PER_ARCH(meta_copy_quad_get)(struct mali_cmd_buffer *cmd, unsigned elem_log2)
{
   struct mali_device *dev = cmd->dev;
   struct mali_shader *s = NULL;

   assert(elem_log2 < 3);
   simple_mtx_lock(&dev->meta_lock);
   if (!dev->meta_gfx)
      dev->meta_gfx = calloc(1, sizeof(struct mali_meta_gfx));
   struct mali_meta_gfx *mg = dev->meta_gfx;
   if (mg) {
      s = mg->copy_quad[elem_log2];
      if (!s) {
         nir_shader *nir = build_copy_quad(dev, elem_log2);
         MALI_PER_ARCH(shader_preprocess)(dev, nir);
         const struct vk_pipeline_robustness_state rs = {
            .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
            .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
            .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
            .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
         };
         const struct mali_shader_compile_info info = {.nir = nir, .rs = &rs};
         blake3_hash hash;
         struct mesa_blake3 h;
         _mesa_blake3_init(&h);
         _mesa_blake3_update(&h, "mali-meta-copy-quad", 19);
         _mesa_blake3_update(&h, &elem_log2, sizeof(elem_log2));
         _mesa_blake3_final(&h, hash);
         if (MALI_PER_ARCH(shader_compile)(dev, &info, hash, &s) != VK_SUCCESS)
            s = NULL;
         mg->copy_quad[elem_log2] = s;
      }
   }
   simple_mtx_unlock(&dev->meta_lock);

   if (!s)
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   return s;
}

#if PAN_ARCH >= 9
const struct mali_shader *
MALI_PER_ARCH(meta_copy_tex_get)(struct mali_cmd_buffer *cmd, unsigned elem_log2)
{
   return copy_get(cmd, elem_log2, true);
}
#else
/*
 * The copy out of a v7 AFBC plane: build_copy with a texture source,
 * except that the texel is fetched in the copy format and packed into the
 * block's bits (one value of at most 32 bits) where build_copy fetches
 * the bits as uint. These shaders are cached by copy format, not by block
 * size as copy_get's are, so they have a builder and a getter of their
 * own.
 */
static nir_shader *
build_copy_tex_fmt(struct mali_device *dev, enum mali_meta_copy_fmt fmt)
{
   const enum pipe_format pf = mali_meta_copy_fmt_pipe(fmt);
   const unsigned elem = util_format_get_blocksize(pf);
   unsigned comps, bits;
   raw_block_type(elem, &comps, &bits);
   assert(comps == 1);

   nir_builder bld = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, MALI_PER_ARCH(shader_nir_options)(dev, MESA_SHADER_COMPUTE),
      "mali_meta_copy_tex_%s", util_format_short_name(pf));
   nir_builder *b = &bld;
   b->shader->info.workgroup_size[0] = MALI_META_COPY_WG;
   b->shader->info.workgroup_size[1] = MALI_META_COPY_WG;
   b->shader->info.workgroup_size[2] = 1;

   nir_def *id = nir_load_global_invocation_id(b, 32);
   nir_def *gx = nir_channel(b, id, 0), *gy = nir_channel(b, id, 1), *gz = nir_channel(b, id, 2);
   nir_def *w = copy_push(b, 32, COPY_PUSH(width));
   nir_def *h = copy_push(b, 32, COPY_PUSH(height));

   nir_push_if(b, nir_iand(b, nir_ult(b, gx, w), nir_ult(b, gy, h)));
   {
      nir_def *sx = nir_iadd(b, gx, copy_push(b, 32, COPY_PUSH(src_x)));
      nir_def *sy = nir_iadd(b, gy, copy_push(b, 32, COPY_PUSH(src_y)));
      nir_def *t = fetch(b, 0, nir_vec2(b, sx, sy), gz, nir_type_float32);
      nir_def *v = nir_format_pack_rgba(b, pf, t);
      if (bits < 32)
         v = nir_u2uN(b, v, bits);
      nir_def *da = surface_addr(
         b, copy_push(b, 64, COPY_PUSH(dst)), copy_push(b, 64, COPY_PUSH(dst_layer_stride)),
         copy_push(b, 32, COPY_PUSH(dst_row_stride)),
         nir_iadd(b, gx, copy_push(b, 32, COPY_PUSH(dst_x))),
         nir_iadd(b, gy, copy_push(b, 32, COPY_PUSH(dst_y))), gz,
         copy_push(b, 32, COPY_PUSH(dst_tile_shift)), elem);
      nir_store_global(b, v, da, .write_mask = 0x1, .align_mul = elem);
   }
   nir_pop_if(b, NULL);
   return b->shader;
}

const struct mali_shader *
MALI_PER_ARCH(meta_copy_tex_fmt_get)(struct mali_cmd_buffer *cmd, enum mali_meta_copy_fmt fmt)
{
   struct mali_device *dev = cmd->dev;
   struct mali_shader *s = NULL;

   assert(fmt > MALI_META_COPY_FMT_RAW && fmt < MALI_META_COPY_FMT_COUNT);
   simple_mtx_lock(&dev->meta_lock);
   if (!dev->meta_gfx)
      dev->meta_gfx = calloc(1, sizeof(struct mali_meta_gfx));
   struct mali_meta_gfx *mg = dev->meta_gfx;
   if (mg) {
      s = mg->copy_tex_fmt[fmt];
      if (!s) {
         nir_shader *nir = build_copy_tex_fmt(dev, fmt);
         MALI_PER_ARCH(shader_preprocess)(dev, nir);
         const struct vk_pipeline_robustness_state rs = {
            .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
            .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
            .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
            .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
         };
         const struct mali_shader_compile_info info = {.nir = nir, .rs = &rs};
         const uint32_t fmt_word = fmt;
         blake3_hash hash;
         struct mesa_blake3 h;
         _mesa_blake3_init(&h);
         _mesa_blake3_update(&h, "mali-meta-copy-tex-fmt", 22);
         _mesa_blake3_update(&h, &fmt_word, sizeof(fmt_word));
         _mesa_blake3_final(&h, hash);
         if (MALI_PER_ARCH(shader_compile)(dev, &info, hash, &s) != VK_SUCCESS)
            s = NULL;
         mg->copy_tex_fmt[fmt] = s;
      }
   }
   simple_mtx_unlock(&dev->meta_lock);

   if (!s)
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   return s;
}
#endif /* PAN_ARCH < 9 */

/* ---------------------------------------------------------------------- */
/* vkCmdClearAttachments                                                   */

static enum mali_meta_fs_type
fs_type(enum pipe_format f)
{
   if (util_format_is_pure_uint(f))
      return MALI_META_FS_UINT;
   if (util_format_is_pure_sint(f))
      return MALI_META_FS_SINT;
   return MALI_META_FS_FLOAT;
}

/*
 * A full-screen draw per rectangle with a shader that writes the clear
 * values (the blob's path; panvk goes through vk_meta and ends in the
 * same RUN_FULLSCREEN, Mesa's v9 driver in a Fullscreen job). Depth and
 * stencil are written by the shader (ZS emit), so no depth-bias trick is
 * needed. The back half's cmd_run_fullscreen draws it: RUN_FULLSCREEN on
 * v11, one Fullscreen job per layer on v9. v7 has no Fullscreen job:
 * mali_v7_cmd_meta_quads draws the rectangle as one Tiler job per layer.
 */
VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdClearAttachments)(VkCommandBuffer commandBuffer, uint32_t attachmentCount,
                                   const VkClearAttachment *pAttachments, uint32_t rectCount,
                                   const VkClearRect *pRects)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   struct mali_render_state *r = &cmd->gfx.render;

   if (!r->active || !rectCount || vk_command_buffer_has_error(&cmd->vk))
      return;

   struct mali_meta_fs_key key = {0};
   struct mali_meta_fs_push push = {0};
   bool any = false;

   for (uint32_t i = 0; i < attachmentCount; i++) {
      const VkClearAttachment *a = &pAttachments[i];
      if (a->aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) {
         const uint32_t rt = a->colorAttachment;
         if (rt == VK_ATTACHMENT_UNUSED || rt >= r->desc.rt_count || !r->desc.rt[rt].image)
            continue;
         key.rt_op[rt] = MALI_META_FS_CLEAR;
         key.rt_type[rt] = fs_type(r->desc.rt[rt].format);
         memcpy(push.clear_color[rt], &a->clearValue.color, 16);
         any = true;
      }
      if ((a->aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) && r->desc.z.image) {
         key.z_op = MALI_META_FS_CLEAR;
         push.clear_depth = a->clearValue.depthStencil.depth;
         any = true;
      }
      if ((a->aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT) && r->desc.s.image) {
         key.s_op = MALI_META_FS_CLEAR;
         push.clear_stencil = a->clearValue.depthStencil.stencil;
         any = true;
      }
   }
   if (!any)
      return;

   if (!MALI_PER_ARCH(cmd_render_tiler)(cmd)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }
   const struct mali_shader *fs = MALI_PER_ARCH(meta_fs_get)(cmd, &key);
#if PAN_ARCH < 9
   /* v7 has no Fullscreen job: each rectangle is one Tiler job per layer,
    * copying this Draw template with the rectangle as its positions and
    * scissor (mali_v7_cmd_meta_quads). */
   struct mali_draw_packed tmpl;
   if (!fs ||
       !MALI_PER_ARCH(meta_fs_dcd)(cmd, fs, &key, &push, NULL, 0, NULL, false, &tmpl)) {
      if (!vk_command_buffer_has_error(&cmd->vk))
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }
   for (uint32_t i = 0; i < rectCount; i++) {
      mali_v7_cmd_meta_quads(cmd, &tmpl, &pRects[i].rect, pRects[i].baseArrayLayer,
                             pRects[i].layerCount, true);
   }
#else
   struct mali_ptr dcd = mali_cmd_alloc(cmd, pan_size(DRAW), 64);
   if (!fs || !dcd.cpu ||
       !MALI_PER_ARCH(meta_fs_dcd)(cmd, fs, &key, &push, NULL, 0, NULL, false, dcd.cpu)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   for (uint32_t i = 0; i < rectCount; i++) {
      MALI_PER_ARCH(cmd_run_fullscreen)(cmd, dcd.gpu, &pRects[i].rect, pRects[i].baseArrayLayer,
                                        pRects[i].layerCount);
   }
#endif
}
