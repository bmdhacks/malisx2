/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Blend shaders (mali_blend.h), built per arch.
 *
 * The shader follows Mesa's pan_blend_create_shader (src/panfrost/lib/
 * pan_blend.c, MIT) for v9+: read the fragment shader's colour output(s)
 * from the blend input registers, read the destination from the tile
 * buffer, blend with nir_lower_blend's helpers (logic op, equation, colour
 * mask), write back with an opaque BLEND and return to the fragment shader.
 * panvk_vX_blend.c compiles it with is_blend and the blend constants in
 * the first four FAU words, which is where our fragment shaders keep them
 * (mali_graphics_sysvals.blend, reserved in every fragment shader).
 *
 * pan_blend.c is not vendored: its PAN_ARCH half includes pan_texture.h,
 * which pulls in the panthor kernel interface (kmod/pan_kmod.h). The
 * fixed-function half is in mali_pipeline_state.c.
 *
 * The same NIR serves v7, compiled with Mesa's Bifrost backend instead of
 * kraid (mali_compiler.c). There the shader returns to the Blend
 * descriptor's Return Value, which the draw sets per render target
 * (mali_jm_cmd_draw_v7.c emit_blend_descs), and the opaque Internal Blend
 * word also names the register format (pan_blend_get_internal_desc at
 * PAN_ARCH < 9).
 *
 * Placement: the code goes into the device's executable pool with the
 * pipelines' shaders, so it shares their
 * 4 GiB window; the Blend descriptor holds only the low 32 bits of its
 * address (shifted by 4, so 16-byte alignment; the pool gives 128).
 * The blob keeps a 128-entry LRU cache of blend shaders; ARMSX2 makes far
 * fewer distinct blends than that, so ours keeps every shader until the
 * device is destroyed.
 */

#ifndef PAN_ARCH
#error "mali_blend.c is built per arch: PAN_ARCH must be set"
#endif
#include "genxml/gen_macros.h"

#include "mali_blend.h"

#include <string.h>

#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_conversion_builder.h"
#include "compiler/nir/nir_lower_blend.h"
#include "panfrost/compiler/pan_compiler.h"
#include "panfrost/compiler/pan_nir.h"
#include "pan_format.h"
#include "util/format/u_format.h"
#include "util/hash_table.h"
#include "util/log.h"
#include "util/u_dynarray.h"
#include "vk_format.h"
#include "vk_log.h"

#include "mali_compiler.h"
#include "mali_vk.h"
#if PAN_ARCH < 9
#include "mali_rsd_v7.h"
#endif

#define BLEND_CODE_ALIGN 128

struct blend_shader {
   struct mali_blend_shader_key key;
   struct mali_bo_ref code;
};

struct blend_cache {
   struct hash_table *ht;        /* key -> struct blend_shader */
};

static uint32_t
key_hash(const void *key)
{
   return _mesa_hash_data(key, sizeof(struct mali_blend_shader_key));
}

static bool
key_equal(const void *a, const void *b)
{
   return !memcmp(a, b, sizeof(struct mali_blend_shader_key));
}

/* ---------------------------------------------------------------------- */
/* NIR                                                                     */

/* Formats the shader reads and writes in a wider form (pan_blend_shader_fmt
 * for v6+). */
static enum pipe_format
shader_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_A8_UNORM:
   case PIPE_FORMAT_L8A8_UNORM:
      return PIPE_FORMAT_R8G8B8A8_UNORM;
   default:
      return format;
   }
}

/* The Internal Blend word pair of an opaque store of `format` to render
 * target `rt` (pan_blend_get_internal_desc). The shader's own tile read
 * and BLEND use it. On v7 its conversion also names the register format
 * of `dest_type`, the type the shader blends in. */
static uint64_t
opaque_internal_desc(enum pipe_format format, unsigned rt, nir_alu_type dest_type)
{
   const struct util_format_description *desc = util_format_description(format);
   struct mali_internal_blend_packed res;

   pan_pack(&res, INTERNAL_BLEND, cfg) {
      cfg.mode = MALI_BLEND_MODE_OPAQUE;
      cfg.fixed_function.num_comps = desc->nr_channels;
      cfg.fixed_function.rt = rt;
#if PAN_ARCH < 9
      cfg.fixed_function.conversion.register_format = mali_v7_register_format(dest_type);
#else
      (void)dest_type;
#endif
      cfg.fixed_function.conversion.memory_format =
         GENX(pan_dithered_format_from_pipe_format)(format, false);
   }
   return res.opaque[0] | ((uint64_t)res.opaque[1] << 32);
}

nir_shader *
MALI_PER_ARCH(blend_shader_nir)(const struct mali_blend_shader_key *key)
{
   const nir_shader_compiler_options *options =
      pan_get_nir_shader_compiler_options(PAN_ARCH, MESA_SHADER_FRAGMENT, false);
   const enum pipe_format format = shader_format(key->format);
   nir_builder builder = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, options, "mali_blend(rt=%u,fmt=%s,samples=%u)", key->rt,
      util_format_short_name(format), key->nr_samples);
   nir_builder *b = &builder;
   const struct util_format_description *desc = util_format_description(format);

   /* Blend in a type that loses no precision; 8-bit channels are blended
    * as 16-bit (the conversion hardware does the rest). */
   nir_alu_type dest_type = pan_unpacked_type_for_format(desc);
   if (nir_alu_type_get_type_size(dest_type) == 8)
      dest_type = nir_alu_type_get_base_type(dest_type) | 16;
   const unsigned dest_bits = nir_alu_type_get_type_size(dest_type);
   const nir_alu_type dest_base = nir_alu_type_get_base_type(dest_type);

   const nir_alu_type src0_type = key->src0_type ? key->src0_type : nir_type_float32;
   const nir_alu_type src1_type = key->src1_type ? key->src1_type : nir_type_float32;

   nir_def *src0 = nir_load_blend_input_pan(
      b, 4, nir_alu_type_get_type_size(src0_type),
      .io_semantics.location = FRAG_RESULT_DATA0 + key->rt,
      .io_semantics.dual_source_blend_index = 0, .io_semantics.num_slots = 1,
      .dest_type = src0_type);
   nir_def *src1 = nir_load_blend_input_pan(
      b, 4, nir_alu_type_get_type_size(src1_type),
      .io_semantics.location = FRAG_RESULT_DATA0 + key->rt,
      .io_semantics.dual_source_blend_index = 1, .io_semantics.num_slots = 1,
      .dest_type = src1_type);

   /* The conversion hardware saturates integer stores itself. */
   src0 = nir_convert_with_rounding(b, src0, dest_base, dest_type, nir_rounding_mode_undef,
                                    false);
   src1 = nir_convert_with_rounding(b, src1, dest_base, dest_type, nir_rounding_mode_undef,
                                    false);

   if (key->alpha_to_one && dest_base == nir_type_float) {
      nir_def *one = nir_imm_floatN_t(b, 1.0, dest_bits);
      src0 = nir_vector_insert_imm(b, src0, one, 3);
      src1 = nir_vector_insert_imm(b, src1, one, 3);
   }

   const uint64_t opaque = opaque_internal_desc(format, key->rt, dest_type);
   nir_def *sample = key->nr_samples > 1 ? nir_load_sample_id(b) : nir_imm_int(b, 0);
   nir_def *dest = nir_load_tile_pan(
      b, 4, dest_bits, pan_nir_tile_rt_sample(b, nir_imm_int(b, key->rt), sample),
      pan_nir_tile_default_coverage(b), nir_imm_int(b, opaque >> 32), .dest_type = dest_type,
      .io_semantics.location = FRAG_RESULT_DATA0 + key->rt, .io_semantics.num_slots = 1);

   const struct mali_blend_eq *eq = &key->eq;
   nir_def *color = src0;
   if (key->logicop_enable) {
      color = nir_color_logicop(b, src0, dest, key->logicop_func, format);
   } else if (eq->enable) {
      const nir_lower_blend_rt rt = {
         .format = format,
         .rgb.func = eq->rgb_func,
         .rgb.src_factor = eq->rgb_src,
         .rgb.dst_factor = eq->rgb_dst,
         .alpha.func = eq->alpha_func,
         .alpha.src_factor = eq->alpha_src,
         .alpha.dst_factor = eq->alpha_dst,
         .colormask = eq->color_mask,
      };
      color = nir_color_blend(b, src0, src1, dest, &rt, false);
   }

   color = nir_color_mask(b, color, dest, eq->color_mask);
   /* Channels the format does not have are not written. */
   color = nir_color_mask(b, color, nir_undef(b, 4, dest_bits), util_format_colormask(desc));

   if (color != dest) {
      nir_blend_pan(b, nir_load_cumulative_coverage_pan(b), nir_imm_int64(b, opaque), color,
                    .src_type = dest_type, .io_semantics.location = FRAG_RESULT_DATA0 + key->rt,
                    .io_semantics.num_slots = 1);
   }
   nir_blend_return_pan(b);

   b->shader->info.io_lowered = true;
   return b->shader;
}

/* The blend constants are the first four 32-bit FAU words of the fragment
 * shader that calls the blend shader (panvk lower_load_blend_const). */
static bool
lower_blend_const(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_blend_const_color_rgba)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *c = nir_load_push_constant(b, intr->def.num_components, intr->def.bit_size,
                                       nir_imm_int(b, 0));
   nir_def_replace(&intr->def, c);
   return true;
}

/* ---------------------------------------------------------------------- */
/* Compile and cache                                                       */

void
MALI_PER_ARCH(blend_shader_key_init)(struct mali_blend_shader_key *key,
                                     const struct mali_gfx_baked *bk, unsigned rt,
                                     enum pipe_format format, unsigned samples,
                                     const struct pan_shader_info *fs_info)
{
   const struct mali_blend_rt_baked *b = &bk->blend[rt];

   memset(key, 0, sizeof(*key));
   key->format = format;
   key->src0_type = fs_info->bifrost.blend[b->shader_location].type;
   key->src1_type = fs_info->bifrost.blend_src1_type;
   key->rt = rt;
   key->nr_samples = samples;
   key->logicop_enable = bk->logicop_enable;
   key->logicop_func = bk->logicop_enable ? bk->logicop_func : 0;
   key->alpha_to_one = bk->alpha_to_one;
   key->eq = b->eq;
}

void
MALI_PER_ARCH(blend_shaders_prepare)(struct mali_device *dev, struct mali_graphics_pipeline *p)
{
   const struct mali_gfx_baked *bk = &p->baked;
   memset(p->blend_shader, 0, sizeof(p->blend_shader));
   if (!p->fs || !bk->needs_blend_shader || bk->uses_dynamic_state)
      return;

   const unsigned samples = MAX2(p->state.ms.rasterization_samples, 1);
   for (unsigned i = 0; i < bk->rt_count && i < MALI_MAX_RTS; i++) {
      if (bk->blend[i].mode != MALI_BLEND_RT_SHADER)
         continue;
      const enum pipe_format format = vk_format_to_pipe_format(bk->blend[i].format);
      struct mali_blend_shader_key key;
      MALI_PER_ARCH(blend_shader_key_init)(&key, bk, i, format, samples, &p->fs->info);
      uint64_t addr;
      if (MALI_PER_ARCH(blend_shader_get)(dev, &key, &addr) == VK_SUCCESS) {
         p->blend_shader[i].addr = addr;
         p->blend_shader[i].format = format;
         p->blend_shader[i].samples = samples;
      }
   }
}

static VkResult
compile_blend_shader(struct mali_device *dev, const struct mali_blend_shader_key *key,
                     struct mali_bo_ref *code)
{
   const struct mali_physical_device *pdev = mali_device_physical(dev);
   nir_shader *nir = MALI_PER_ARCH(blend_shader_nir)(key);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_blend_const, nir_metadata_control_flow,
            NULL);

   struct pan_compile_inputs inputs = {
      .gpu_id = pdev->props.gpu_id,
      .is_blend = true,
      /* The blend constants, 32-bit words. */
      .fau.reserved = 4,
   };
   struct pan_shader_info info;
   memset(&info, 0, sizeof(info));
   pan_preprocess_nir(nir, inputs.gpu_id);
   pan_postprocess_nir(nir, &inputs, &info);

   struct util_dynarray bin;
   util_dynarray_init(&bin, NULL);
   char why[160];
   enum mali_compile_status st = mali_compile_nir(nir, &inputs, &bin, &info, why, sizeof(why));
   ralloc_free(nir);
   if (st != MALI_COMPILE_OK || !bin.size) {
      util_dynarray_fini(&bin);
      return vk_errorf(dev, VK_ERROR_UNKNOWN, "blend shader (format %s, rt %u): %s",
                       util_format_short_name(key->format), key->rt,
                       st != MALI_COMPILE_OK ? why : "empty binary");
   }

   const uint64_t padded = align64(bin.size, BLEND_CODE_ALIGN);
   if (mali_bo_pool_alloc(&dev->exec_pool, padded, BLEND_CODE_ALIGN, code) !=
       MALI_KBASE_SUCCESS) {
      util_dynarray_fini(&bin);
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }
   memcpy(code->cpu, bin.data, bin.size);
   memset((uint8_t *)code->cpu + bin.size, 0, padded - bin.size);
   util_dynarray_fini(&bin);
   return VK_SUCCESS;
}

VkResult
MALI_PER_ARCH(blend_shader_get)(struct mali_device *dev, const struct mali_blend_shader_key *key,
                                uint64_t *addr)
{
   VkResult result = VK_SUCCESS;

   simple_mtx_lock(&dev->meta_lock);
   struct blend_cache *cache = dev->blend_shaders;
   if (!cache) {
      cache = calloc(1, sizeof(*cache));
      if (cache)
         cache->ht = _mesa_hash_table_create(NULL, key_hash, key_equal);
      if (!cache || !cache->ht) {
         free(cache);
         simple_mtx_unlock(&dev->meta_lock);
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
      dev->blend_shaders = cache;
   }

   struct hash_entry *he = _mesa_hash_table_search(cache->ht, key);
   if (he) {
      *addr = ((struct blend_shader *)he->data)->code.gpu_va;
   } else {
      struct blend_shader *bs = calloc(1, sizeof(*bs));
      if (!bs) {
         result = vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      } else {
         bs->key = *key;
         result = compile_blend_shader(dev, key, &bs->code);
         if (result == VK_SUCCESS) {
            _mesa_hash_table_insert(cache->ht, &bs->key, bs);
            *addr = bs->code.gpu_va;
         } else {
            free(bs);
         }
      }
   }
   simple_mtx_unlock(&dev->meta_lock);
   return result;
}

unsigned
MALI_PER_ARCH(blend_shader_count)(struct mali_device *dev)
{
   struct blend_cache *cache = dev->blend_shaders;
   return cache ? _mesa_hash_table_num_entries(cache->ht) : 0;
}

void
MALI_PER_ARCH(blend_shaders_finish)(struct mali_device *dev)
{
   struct blend_cache *cache = dev->blend_shaders;
   if (!cache)
      return;
   hash_table_foreach(cache->ht, he) {
      struct blend_shader *bs = he->data;
      mali_bo_pool_free(&dev->exec_pool, &bs->code);
      free(bs);
   }
   _mesa_hash_table_destroy(cache->ht, NULL);
   free(cache);
   dev->blend_shaders = NULL;
}
