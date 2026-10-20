/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * vkCreateGraphicsPipelines, vkCreateComputePipelines and pipeline
 * destruction.
 *
 * Mesa's runtime has a generic pipeline implementation (vk_pipeline.c,
 * which panvk uses) built on per-shader objects; we use its pieces
 * (SPIR-V to NIR for a stage, stage hashing, robustness, the pipeline
 * cache, graphics state parsing) but create the pipeline objects ourselves,
 * because a pipeline here bakes hardware words at creation, as the blob
 * does, and ARMSX2 needs only vertex + fragment and compute pipelines
 * without libraries.
 *
 * Shaders are cached per stage. Keys:
 *   vertex   = BLAKE3("vs", stage, layout, view mask)
 *   fragment = BLAKE3("fs", stage, layout, vertex stage, the state the
 *              fragment lowering reads)
 *   compute  = BLAKE3("cs", stage, layout)
 * and on v7, for the fragment shader that only clips (clip_fs_compile),
 *   BLAKE3("mali-v7-clip-fs", the vertex shader's varying layout, the
 *          state the fragment lowering reads)
 * "stage" is the runtime's hash of the stage create info (SPIR-V, entry
 * point, specialization, flags, robustness). The fragment shader is
 * compiled against the vertex shader's varying layout, so its key includes
 * the vertex stage. The vertex shader is not specialized on the fragment
 * shader: it reads the noperspective mask from a system value, so one
 * compiled vertex shader serves every pipeline that uses it.
 */

#ifndef PAN_ARCH
#error "mali_pipeline.c is built per arch: PAN_ARCH must be set"
#endif

#include "mali_blend.h"
#include "mali_pipeline.h"
#if PAN_ARCH >= 10
#include "mali_cmd_buffer.h"
#else
#include "mali_jm.h"
#endif
#include "mali_vk.h"

#include "mali_arch.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vk_alloc.h"
#include "compiler/spirv/nir_spirv.h"
#include "vk_command_buffer.h"
#include "vk_log.h"
#include "vk_shader_module.h"
#include "vk_util.h"

#include "panfrost/compiler/pan_nir.h"
#include "util/hash_table.h"
#include "util/os_time.h"
#include "util/simple_mtx.h"

#include "mali_compiler.h"
#include "mali_measure.h"

/* ---------------------------------------------------------------------- */
/* Destruction and bind                                                    */

static void
graphics_pipeline_destroy(struct vk_device *vk_dev, struct vk_pipeline *vk_pipeline,
                          const VkAllocationCallbacks *alloc)
{
   struct mali_device *dev = container_of(vk_dev, struct mali_device, vk);
   struct mali_graphics_pipeline *p = mali_graphics_pipeline(vk_pipeline);

   MALI_PER_ARCH(shader_unref)(dev, p->vs);
   MALI_PER_ARCH(shader_unref)(dev, p->fs);
   if (p->zsd_mem.size)
      mali_bo_pool_free(&dev->desc_pool, &p->zsd_mem);
   if (p->base.layout)
      vk_pipeline_layout_unref(vk_dev, p->base.layout);
   vk_pipeline_free(vk_dev, alloc, vk_pipeline);
}

static void
compute_pipeline_destroy(struct vk_device *vk_dev, struct vk_pipeline *vk_pipeline,
                         const VkAllocationCallbacks *alloc)
{
   struct mali_device *dev = container_of(vk_dev, struct mali_device, vk);
   struct mali_compute_pipeline *p = mali_compute_pipeline(vk_pipeline);

   MALI_PER_ARCH(shader_unref)(dev, p->cs);
   if (p->base.layout)
      vk_pipeline_layout_unref(vk_dev, p->base.layout);
   vk_pipeline_free(vk_dev, alloc, vk_pipeline);
}

/* The pipeline ops' bind (the runtime's vkCmdBindPipeline would call it;
 * the per-arch CmdBindPipeline entry point replaces that). The
 * dynamic-state part of a graphics bind is mali_cmd_bind_graphics'. */
static void
pipeline_cmd_bind(struct vk_command_buffer *cmd, struct vk_pipeline *pipeline)
{
   MALI_PER_ARCH(cmd_bind_pipeline)(container_of(cmd, struct mali_cmd_buffer, vk), pipeline);
}

static const struct vk_pipeline_ops graphics_pipeline_ops = {
   .destroy = graphics_pipeline_destroy,
   .cmd_bind = pipeline_cmd_bind,
};

static const struct vk_pipeline_ops compute_pipeline_ops = {
   .destroy = compute_pipeline_destroy,
   .cmd_bind = pipeline_cmd_bind,
};

/* ---------------------------------------------------------------------- */
/* Shader lookup and compile                                               */

/* Everything about a layout that the shader lowering reads. */
static void
hash_layout(struct mesa_blake3 *ctx, const struct vk_pipeline_layout *layout)
{
   static const blake3_hash zero;

   uint32_t set_count = layout ? layout->set_count : 0;
   _mesa_blake3_update(ctx, &set_count, sizeof(set_count));
   for (uint32_t i = 0; i < set_count; i++) {
      const struct vk_descriptor_set_layout *sl = layout->set_layouts[i];
      _mesa_blake3_update(ctx, sl ? sl->blake3 : zero, sizeof(blake3_hash));
   }
   /* Push constant ranges do not change the code (offsets are the API's),
    * so they are not part of the key. */
}

struct stage {
   const VkPipelineShaderStageCreateInfo *info;
   struct vk_pipeline_robustness_state rs;
   blake3_hash stage_hash;
   blake3_hash key;
   nir_shader *nir;
   struct mali_shader *shader;
   bool cache_hit;
   int64_t duration_ns;
};

static struct vk_pipeline_cache *
pick_cache(struct mali_device *dev, VkPipelineCache handle)
{
   VK_FROM_HANDLE(vk_pipeline_cache, cache, handle);
   return cache ? cache : dev->vk.mem_cache;
}

static void
stage_init(struct mali_device *dev, VkPipelineCreateFlags2KHR flags,
           const void *pipeline_pNext, const VkPipelineShaderStageCreateInfo *info,
           struct stage *st)
{
   memset(st, 0, sizeof(*st));
   st->info = info;
   vk_pipeline_robustness_state_fill(&dev->vk.robustness_state, &st->rs,
                                     pipeline_pNext, info->pNext);
   vk_pipeline_hash_shader_stage(flags, info, &st->rs, st->stage_hash);
}

static bool
stage_lookup(struct mali_device *dev, struct vk_pipeline_cache *cache, struct stage *st)
{
   if (!cache)
      return false;

   struct vk_pipeline_cache_object *obj =
      vk_pipeline_cache_lookup_object(cache, st->key, sizeof(st->key),
                                      &MALI_PER_ARCH(shader_cache_ops), &st->cache_hit);
   if (!obj) {
      st->cache_hit = false;
      return false;
   }
   st->shader = container_of(obj, struct mali_shader, base);
   st->cache_hit = true;
   p_atomic_inc(&dev->pipeline_stats.shader_cache_hits);
   return true;
}

/* The stage's SPIR-V, from its module or an inline VkShaderModuleCreateInfo. */
static bool
stage_spirv(const VkPipelineShaderStageCreateInfo *info, const uint32_t **words,
            size_t *size)
{
   VK_FROM_HANDLE(vk_shader_module, module, info->module);
   if (module) {
      *words = (const uint32_t *)module->data;
      *size = module->size;
      return true;
   }
   const VkShaderModuleCreateInfo *minfo =
      vk_find_struct_const(info->pNext, SHADER_MODULE_CREATE_INFO);
   if (!minfo)
      return false;
   *words = minfo->pCode;
   *size = minfo->codeSize;
   return true;
}

/* SPIR-V -> NIR -> generic lowering. */
static VkResult
stage_to_nir(struct mali_device *dev, VkPipelineCreateFlags2KHR flags, struct stage *st)
{
   if (st->nir)
      return VK_SUCCESS;

   const uint32_t *words;
   size_t size;
   if (!stage_spirv(st->info, &words, &size) || size % 4 != 0 ||
       !mali_spirv_structure_ok(words, size / 4))
      return vk_errorf(dev, VK_ERROR_INVALID_SHADER_NV,
                       "malformed SPIR-V for the %s stage",
                       mesa_shader_stage_name(vk_to_mesa_shader_stage(st->info->stage)));

   const mesa_shader_stage stage = vk_to_mesa_shader_stage(st->info->stage);
   const struct spirv_to_nir_options spirv_options = MALI_PER_ARCH(shader_spirv_options)(&st->rs);
   VkResult result =
      vk_pipeline_shader_stage_to_nir(&dev->vk, flags, st->info, &spirv_options,
                                      MALI_PER_ARCH(shader_nir_options)(dev, stage), NULL,
                                      &st->nir);
   if (result != VK_SUCCESS)
      return vk_errorf(dev, VK_ERROR_INVALID_SHADER_NV,
                       "the %s stage's SPIR-V was rejected by the SPIR-V front end",
                       mesa_shader_stage_name(stage));

   MALI_PER_ARCH(shader_preprocess)(dev, st->nir);
   return VK_SUCCESS;
}

/*
 * MALISX2_MEASURE=shaders: every stage compiled (not cache hits) is
 * written as <dir>/shaders/<key>.<stage>.spv and .bin, with a line in
 * index.txt, so the machine code of an application's shaders can be
 * disassembled offline.
 */
static void
stage_dump(const struct stage *st, const struct mali_shader *s)
{
   static simple_mtx_t lock = SIMPLE_MTX_INITIALIZER;
   const char *dir = mali_measure_shader_dir();
   if (!dir)
      return;
   simple_mtx_lock(&lock);

   const char *sname = s->stage == MESA_SHADER_VERTEX     ? "vs"
                       : s->stage == MESA_SHADER_FRAGMENT ? "fs"
                                                          : "cs";
   uint64_t key;
   memcpy(&key, st->key, sizeof(key));
   char path[512];
   const uint32_t *words;
   size_t size;
   if (stage_spirv(st->info, &words, &size)) {
      snprintf(path, sizeof(path), "%s/%016" PRIx64 ".%s.spv", dir, key, sname);
      FILE *f = fopen(path, "wb");
      if (f) {
         fwrite(words, 1, size, f);
         fclose(f);
      }
   }
   snprintf(path, sizeof(path), "%s/%016" PRIx64 ".%s.bin", dir, key, sname);
   FILE *f = fopen(path, "wb");
   if (f) {
      fwrite(s->bin, 1, s->bin_size, f);
      fclose(f);
   }
   snprintf(path, sizeof(path), "%s/index.txt", dir);
   f = fopen(path, "a");
   if (f) {
      fprintf(f, "%016" PRIx64 " %s bytes=%u regs=%u fau=%u entry=%s\n", key, sname,
              s->bin_size, s->info.work_reg_count, s->fau.total_count,
              st->info->pName);
      fclose(f);
   }
   simple_mtx_unlock(&lock);
}

static VkResult
stage_compile(struct mali_device *dev, struct vk_pipeline_cache *cache,
              VkPipelineCreateFlags2KHR flags, struct stage *st,
              const struct vk_pipeline_layout *layout,
              const struct vk_graphics_pipeline_state *state,
              const struct pan_varying_layout *vs_varying_layout)
{
   int64_t start = os_time_get_nano();

   VkResult result = stage_to_nir(dev, flags, st);
   if (result != VK_SUCCESS)
      return result;

   const struct mali_shader_compile_info info = {
      .nir = st->nir,
      .rs = &st->rs,
      .layout = layout,
      .state = state,
      .vs_varying_layout = vs_varying_layout,
   };
   st->nir = NULL; /* consumed */

   struct mali_shader *shader;
   result = MALI_PER_ARCH(shader_compile)(dev, &info, st->key, &shader);
   if (result != VK_SUCCESS)
      return result;
   p_atomic_inc(&dev->pipeline_stats.shaders_compiled);
   stage_dump(st, shader);

   if (cache) {
      struct vk_pipeline_cache_object *obj =
         vk_pipeline_cache_add_object(cache, &shader->base);
      shader = container_of(obj, struct mali_shader, base);
   }
   st->shader = shader;
   st->duration_ns = os_time_get_nano() - start;
   return VK_SUCCESS;
}

static void
stage_finish(struct mali_device *dev, struct stage *st)
{
   ralloc_free(st->nir);
   st->nir = NULL;
   MALI_PER_ARCH(shader_unref)(dev, st->shader);
   st->shader = NULL;
}

/* VkPipelineCreationFeedbackCreateInfo, if the application chained one. */
static void
write_feedback(const void *pNext, const struct stage *stages, uint32_t stage_count,
               bool app_cache, int64_t duration_ns)
{
   const VkPipelineCreationFeedbackCreateInfo *fb =
      vk_find_struct_const(pNext, PIPELINE_CREATION_FEEDBACK_CREATE_INFO);
   if (!fb)
      return;

   bool all_hit = stage_count > 0;
   for (uint32_t i = 0; i < stage_count; i++)
      all_hit &= stages[i].cache_hit;

   if (fb->pPipelineCreationFeedback) {
      fb->pPipelineCreationFeedback->flags =
         VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT |
         (all_hit && app_cache ?
             VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT : 0);
      fb->pPipelineCreationFeedback->duration = duration_ns;
   }

   for (uint32_t i = 0; i < fb->pipelineStageCreationFeedbackCount && i < stage_count; i++) {
      fb->pPipelineStageCreationFeedbacks[i].flags =
         VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT |
         (stages[i].cache_hit && app_cache ?
             VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT : 0);
      fb->pPipelineStageCreationFeedbacks[i].duration = stages[i].duration_ns;
   }
}

/* ---------------------------------------------------------------------- */
/* Graphics                                                                */

/*
 * Dynamic states that feed none of the baked words: the draw encoder
 * handles them itself (viewport and scissor registers, blend constants in
 * the blend descriptor and FAU, line width, vertex buffers).
 */
static bool
baked_words_use_dynamic_state(const BITSET_WORD *dynamic)
{
   BITSET_DECLARE(d, MESA_VK_DYNAMIC_GRAPHICS_STATE_ENUM_MAX);
   BITSET_COPY(d, dynamic);

   static const enum mesa_vk_dynamic_graphics_state independent[] = {
      MESA_VK_DYNAMIC_VI,
      MESA_VK_DYNAMIC_VI_BINDINGS_VALID,
      MESA_VK_DYNAMIC_VI_BINDING_STRIDES,
      MESA_VK_DYNAMIC_VP_VIEWPORT_COUNT,
      MESA_VK_DYNAMIC_VP_VIEWPORTS,
      MESA_VK_DYNAMIC_VP_SCISSOR_COUNT,
      MESA_VK_DYNAMIC_VP_SCISSORS,
      MESA_VK_DYNAMIC_RS_LINE_WIDTH,
      MESA_VK_DYNAMIC_CB_BLEND_CONSTANTS,
      /* The runtime marks advanced blending dynamic when the feature is
       * off (always, here); nothing reads it. */
      MESA_VK_DYNAMIC_CB_BLEND_ADVANCED,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(independent); i++)
      BITSET_CLEAR(d, independent[i]);

   return !BITSET_IS_EMPTY(d);
}

/* The graphics state the fragment lowering reads. */
static void
hash_fs_state(struct mesa_blake3 *ctx, const struct mali_device *dev,
              const struct vk_graphics_pipeline_state *state)
{
   bool sample_shading = state->ms && state->ms->sample_shading_enable;
   uint32_t view_mask = state->mv ? state->mv->view_mask : 0;
   /* Whether the fragment shader keeps its ATEST (mali_shader.c). */
   bool a2c = BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_ALPHA_TO_COVERAGE_ENABLE) ||
              (state->ms && state->ms->alpha_to_coverage_enable);
   uint32_t roa = state->rasterization_order_access;
   /* Whether it keeps its second colour output. */
   bool dual_src = MALI_PER_ARCH(shader_fs_keeps_dual_source)(dev, state);
   _mesa_blake3_update(ctx, &sample_shading, sizeof(sample_shading));
   _mesa_blake3_update(ctx, &view_mask, sizeof(view_mask));
   _mesa_blake3_update(ctx, &a2c, sizeof(a2c));
   _mesa_blake3_update(ctx, &roa, sizeof(roa));
   _mesa_blake3_update(ctx, &dual_src, sizeof(dual_src));
   if (state->ial)
      _mesa_blake3_update(ctx, state->ial, sizeof(*state->ial));
   if (state->cal)
      _mesa_blake3_update(ctx, state->cal, sizeof(*state->cal));
}

/* Descriptor sets a stage's resource table reads: the sets it indexes and
 * the sets its dynamic buffer copies come from. */
static uint16_t
stage_sets(const struct mali_shader *s)
{
   uint32_t m = s->desc.used_set_mask;
   for (uint32_t i = 0; i < s->desc.dyn_bufs.count; i++)
      m |= BITFIELD_BIT(MALI_COPY_DESC_HANDLE_SET(s->desc.dyn_bufs.map[i]));
   return (uint16_t)m;
}

/* Everything mali_gfx_vs_srt (mali_cmd_state.h) reads from the pipeline side,
 * with unused entries zero. */
struct vs_srt_key {
   struct mali_shader_desc_info desc;
   uint32_t dynamic_vi, dynamic_strides;
   uint32_t attributes_valid, bindings_valid;
   struct {
      uint32_t binding, format, offset;
   } attr[MESA_VK_MAX_VERTEX_ATTRIBUTES];
   struct {
      uint32_t input_rate, divisor, stride;
   } binding[MESA_VK_MAX_VERTEX_BINDINGS];
};

/* Everything mali_gfx_fs_srt reads from the pipeline side. */
struct fs_srt_key {
   uint32_t has_fs;
   struct mali_shader_desc_info desc;
   struct pan_varying_layout vs_formats, fs_formats;
};

static void
copy_varying_layout(struct pan_varying_layout *dst, const struct pan_varying_layout *src)
{
   dst->count = src->count;
   dst->known = src->known;
   dst->generic_size_B = src->generic_size_B;
   for (unsigned i = 0; i < src->count && i < ARRAY_SIZE(dst->slots); i++)
      dst->slots[i] = src->slots[i];
}

/* The bind block (struct mali_gfx_bind): register words, the static
 * states the draw reads, table keys, and on v9 and v11 a GPU copy of the
 * Depth/stencil descriptor when nothing dynamic feeds it. */
static VkResult
fill_bind(struct mali_device *dev, struct mali_graphics_pipeline *p)
{
   struct mali_gfx_bind *bd = &p->bind;
   const struct mali_gfx_baked *bk = &p->baked;
   const BITSET_WORD *set = p->state.set;

   bd->spd[0] = bk->vs_pos_spd;
   bd->spd[1] = bk->vs_var_spd;
   bd->spd[2] = bk->fs_spd;
   bd->tiler_flags = bk->tiler_flags;
   bd->dcd0 = bk->dcd0[0];
   bd->dcd1 = bk->dcd1;
   bd->dcd2 = bk->dcd2;
   bd->vary_size = p->fs ? p->vs->info.varyings.formats.generic_size_B : 0;
   bd->tls_size = bk->tls_size;
   bd->noperspective = p->fs ? p->fs->info.varyings.noperspective : 0;
   bd->dynamic = bk->uses_dynamic_state;
   /* A dynamic topology keeps the point size (mali_cmd_draw.c reads
    * baked.prim only when it is static). */
   bd->lines = bk->prim != MESA_PRIM_COUNT && u_reduced_prim(bk->prim) == MESA_PRIM_LINES;
   bd->has_fs = p->fs != NULL;
   bd->vs_sets = stage_sets(p->vs);
   bd->vs_first_vertex =
      BITSET_TEST(p->vs->fau.used_sysvals,
                  offsetof(struct mali_graphics_sysvals, vs.first_vertex) / MALI_FAU_WORD_SIZE);
   bd->vs_base_instance =
      BITSET_TEST(p->vs->fau.used_sysvals,
                  offsetof(struct mali_graphics_sysvals, vs.base_instance) / MALI_FAU_WORD_SIZE);
   bd->fs_sets = p->fs ? stage_sets(p->fs) : 0;

   bd->static_mask = 0;
   if (BITSET_TEST(set, MESA_VK_DYNAMIC_VI))
      bd->static_mask |= MALI_GFX_STATIC_VI;
   if (BITSET_TEST(set, MESA_VK_DYNAMIC_VI_BINDING_STRIDES))
      bd->static_mask |= MALI_GFX_STATIC_STRIDES;
   if (BITSET_TEST(set, MESA_VK_DYNAMIC_VP_VIEWPORTS))
      bd->static_mask |= MALI_GFX_STATIC_VIEWPORTS;
   if (BITSET_TEST(set, MESA_VK_DYNAMIC_VP_SCISSORS))
      bd->static_mask |= MALI_GFX_STATIC_SCISSORS;
   if (BITSET_TEST(set, MESA_VK_DYNAMIC_CB_BLEND_CONSTANTS))
      bd->static_mask |= MALI_GFX_STATIC_BLEND_CONSTANTS;
   if (BITSET_TEST(set, MESA_VK_DYNAMIC_RS_LINE_WIDTH))
      bd->static_mask |= MALI_GFX_STATIC_LINE_WIDTH;
   if (BITSET_TEST(set, MESA_VK_DYNAMIC_INPUT_ATTACHMENT_MAP))
      bd->static_mask |= MALI_GFX_STATIC_IAL;

   bd->line_width = p->state.rs.line.width;
   bd->ial = p->state.ial;

   struct vs_srt_key vkey;
   memset(&vkey, 0, sizeof(vkey));
   vkey.desc = p->vs->desc;
   vkey.dynamic_vi = !(bd->static_mask & MALI_GFX_STATIC_VI);
   vkey.dynamic_strides = !(bd->static_mask & MALI_GFX_STATIC_STRIDES);
   if (!vkey.dynamic_vi) {
      vkey.attributes_valid = p->vi.attributes_valid;
      vkey.bindings_valid = p->vi.bindings_valid;
      u_foreach_bit(a, p->vi.attributes_valid) {
         vkey.attr[a].binding = p->vi.attributes[a].binding;
         vkey.attr[a].format = p->vi.attributes[a].format;
         vkey.attr[a].offset = p->vi.attributes[a].offset;
      }
      u_foreach_bit(b, p->vi.bindings_valid) {
         vkey.binding[b].input_rate = p->vi.bindings[b].input_rate;
         vkey.binding[b].divisor = p->vi.bindings[b].divisor;
      }
   }
   if (!vkey.dynamic_strides) {
      for (unsigned b = 0; b < MESA_VK_MAX_VERTEX_BINDINGS; b++)
         vkey.binding[b].stride = p->state.vi_binding_strides[b];
   }
   bd->vs_srt_key = mali_device_intern_key(dev, &vkey, sizeof(vkey));

   struct fs_srt_key fkey;
   memset(&fkey, 0, sizeof(fkey));
   if (p->fs) {
      fkey.has_fs = 1;
      fkey.desc = p->fs->desc;
      if (p->fs->desc.needs_varying_descs) {
         copy_varying_layout(&fkey.vs_formats, &p->vs->info.varyings.formats);
         copy_varying_layout(&fkey.fs_formats, &p->fs->info.varyings.formats);
      }
   }
   bd->fs_srt_key = mali_device_intern_key(dev, &fkey, sizeof(fkey));

   bd->zsd = 0;
#if PAN_ARCH >= 9
   if (!bk->uses_dynamic_state) {
      if (mali_bo_pool_alloc(&dev->desc_pool, sizeof(bk->zsd), 32, &p->zsd_mem) !=
          MALI_KBASE_SUCCESS)
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      memcpy(p->zsd_mem.cpu, bk->zsd, sizeof(bk->zsd));
      bd->zsd = p->zsd_mem.gpu_va;
   }
#else
   /* v7 has no Depth/stencil descriptor: the draw copies baked.zsd into
    * its fragment Renderer State Descriptor (mali_jm_cmd_draw_v7.c
    * build_fs_rsd). */
#endif
   return VK_SUCCESS;
}

#if PAN_ARCH < 9
/*
 * v7 clips in the fragment shader (mali_shader.c, "Clip distances"), so a
 * pipeline whose vertex shader writes clip distances and that has no
 * fragment shader gets one that only clips: an empty fragment shader, to
 * which the fragment compile adds the clip discard. Like no fragment
 * shader, it writes no colour (every colour target stays off,
 * mali_pipeline_state.c pack_blend); its discard makes the depth/stencil
 * update late, so a clipped fragment writes no depth or stencil.
 */
static bool
vs_writes_clip_distances(const struct mali_shader *vs)
{
   const struct pan_varying_layout *vl = &vs->info.varyings.formats;
   return pan_varying_layout_find_slot(vl, VARYING_SLOT_CLIP_DIST0) ||
          pan_varying_layout_find_slot(vl, VARYING_SLOT_CLIP_DIST1);
}

/* Its cache key covers what its compile reads: the vertex shader's
 * varying layout and the fragment state (hash_fs_state). */
static void
clip_fs_key(const struct mali_device *dev, const struct vk_graphics_pipeline_state *state,
            const struct mali_shader *vs, blake3_hash key)
{
   struct pan_varying_layout vl;
   memset(&vl, 0, sizeof(vl));
   copy_varying_layout(&vl, &vs->info.varyings.formats);

   struct mesa_blake3 h;
   _mesa_blake3_init(&h);
   _mesa_blake3_update(&h, "mali-v7-clip-fs", 15);
   _mesa_blake3_update(&h, &vl, sizeof(vl));
   hash_fs_state(&h, dev, state);
   _mesa_blake3_final(&h, key);
}

static VkResult
clip_fs_compile(struct mali_device *dev, struct vk_pipeline_cache *cache,
                const struct vk_graphics_pipeline_state *state, const struct mali_shader *vs,
                struct stage *st)
{
   const int64_t start = os_time_get_nano();
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, MALI_PER_ARCH(shader_nir_options)(dev, MESA_SHADER_FRAGMENT),
      "mali_v7_clip_fs");
   MALI_PER_ARCH(shader_preprocess)(dev, b.shader);

   /* The shader reads no buffer or image. */
   const struct vk_pipeline_robustness_state rs = {
      .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
   };
   const struct mali_shader_compile_info info = {
      .nir = b.shader,
      .rs = &rs,
      .state = state,
      .vs_varying_layout = &vs->info.varyings.formats,
   };

   struct mali_shader *shader;
   VkResult result = MALI_PER_ARCH(shader_compile)(dev, &info, st->key, &shader);
   if (result != VK_SUCCESS)
      return result;
   p_atomic_inc(&dev->pipeline_stats.shaders_compiled);

   if (cache) {
      struct vk_pipeline_cache_object *obj =
         vk_pipeline_cache_add_object(cache, &shader->base);
      shader = container_of(obj, struct mali_shader, base);
   }
   st->shader = shader;
   st->duration_ns = os_time_get_nano() - start;
   return VK_SUCCESS;
}
#endif /* PAN_ARCH < 9 */

static VkResult
create_graphics_pipeline(struct mali_device *dev, VkPipelineCache cache_handle,
                         const VkGraphicsPipelineCreateInfo *ci,
                         const VkAllocationCallbacks *alloc, VkPipeline *out)
{
   const int64_t start = os_time_get_nano();
   const VkPipelineCreateFlags2KHR flags = vk_graphics_pipeline_create_flags(ci);
   VK_FROM_HANDLE(vk_pipeline_layout, layout, ci->layout);
   struct vk_pipeline_cache *cache = pick_cache(dev, cache_handle);
   struct stage stages[2];
   struct stage *vs = NULL, *fs = NULL;
#if PAN_ARCH < 9
   /* The fragment shader that only clips (clip_fs_compile), if needed. */
   struct stage clip = {0};
#endif
   uint32_t stage_count = 0;
   VkResult result;

   *out = VK_NULL_HANDLE;

   if (flags & VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR)
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "graphics pipeline libraries are not supported");

   for (uint32_t i = 0; i < ci->stageCount; i++) {
      const VkPipelineShaderStageCreateInfo *si = &ci->pStages[i];
      if (vk_pipeline_shader_stage_is_null(si))
         continue;
      switch (si->stage) {
      case VK_SHADER_STAGE_VERTEX_BIT:
         vs = &stages[stage_count++];
         stage_init(dev, flags, ci->pNext, si, vs);
         break;
      case VK_SHADER_STAGE_FRAGMENT_BIT:
         fs = &stages[stage_count++];
         stage_init(dev, flags, ci->pNext, si, fs);
         break;
      default:
         return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                          "shader stage 0x%x is not supported", si->stage);
      }
   }
   if (!vs)
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "graphics pipeline without a vertex shader");

   struct vk_graphics_pipeline_all_state all;
   struct vk_graphics_pipeline_state state = {0};
   result = vk_graphics_pipeline_state_fill(&dev->vk, &state, ci, NULL, NULL, 0, &all,
                                            NULL, 0, NULL);
   if (result != VK_SUCCESS)
      return result;

   /* Keys. */
   struct mesa_blake3 h;
   const uint32_t view_mask = state.mv ? state.mv->view_mask : 0;

   _mesa_blake3_init(&h);
   _mesa_blake3_update(&h, "vs", 2);
   _mesa_blake3_update(&h, vs->stage_hash, sizeof(vs->stage_hash));
   hash_layout(&h, layout);
   _mesa_blake3_update(&h, &view_mask, sizeof(view_mask));
   _mesa_blake3_final(&h, vs->key);

   if (fs) {
      _mesa_blake3_init(&h);
      _mesa_blake3_update(&h, "fs", 2);
      _mesa_blake3_update(&h, fs->stage_hash, sizeof(fs->stage_hash));
      hash_layout(&h, layout);
      _mesa_blake3_update(&h, vs->stage_hash, sizeof(vs->stage_hash));
      hash_fs_state(&h, dev, &state);
      _mesa_blake3_final(&h, fs->key);
   }

   stage_lookup(dev, cache, vs);
   if (fs)
      stage_lookup(dev, cache, fs);

   if ((flags & VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_KHR) &&
       (!vs->shader || (fs && !fs->shader))) {
      result = VK_PIPELINE_COMPILE_REQUIRED;
      goto out;
   }

   if (!vs->shader) {
      result = stage_compile(dev, cache, flags, vs, layout, &state, NULL);
      if (result != VK_SUCCESS)
         goto out;
   }
   if (fs && !fs->shader) {
      result = stage_compile(dev, cache, flags, fs, layout, &state,
                             &vs->shader->info.varyings.formats);
      if (result != VK_SUCCESS)
         goto out;
   }
#if PAN_ARCH < 9
   if (!fs && vs_writes_clip_distances(vs->shader)) {
      clip_fs_key(dev, &state, vs->shader, clip.key);
      if (!stage_lookup(dev, cache, &clip)) {
         if (flags & VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_KHR) {
            result = VK_PIPELINE_COMPILE_REQUIRED;
            goto out;
         }
         result = clip_fs_compile(dev, cache, &state, vs->shader, &clip);
         if (result != VK_SUCCESS)
            goto out;
         /* Creation feedback counts it with the vertex stage. */
         vs->cache_hit = false;
         vs->duration_ns += clip.duration_ns;
      }
   }
#endif

   struct mali_graphics_pipeline *p =
      vk_pipeline_zalloc(&dev->vk, &graphics_pipeline_ops,
                         VK_PIPELINE_BIND_POINT_GRAPHICS, flags, alloc, sizeof(*p));
   if (!p) {
      result = vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto out;
   }
   p->base.vk.stages = state.shader_stages;
   if (layout)
      p->base.layout = vk_pipeline_layout_ref(layout);
   p->vs = vs->shader;
   vs->shader = NULL;
   if (fs) {
      p->fs = fs->shader;
      fs->shader = NULL;
   }
#if PAN_ARCH < 9
   if (clip.shader) {
      p->fs = clip.shader;
      clip.shader = NULL;
   }
#endif

   /* The static state, and the words baked from it. */
   BITSET_COPY(p->dynamic, state.dynamic);
   vk_dynamic_graphics_state_init(&p->state);
   p->state.vi = &p->vi;
   p->state.ms.sample_locations = &p->sl;
   vk_dynamic_graphics_state_fill(&p->state, &state);
   if (state.rp)
      p->rp = *state.rp;

   p->baked.uses_dynamic_state = baked_words_use_dynamic_state(state.dynamic);
   const struct mali_gfx_pack_input pack = {
      .dyn = &p->state,
      .rp = &p->rp,
      .vs = p->vs,
      .fs = p->fs,
      .view_mask = view_mask,
   };
   MALI_PER_ARCH(gfx_pack_state)(&pack, &p->baked);
   MALI_PER_ARCH(blend_shaders_prepare)(dev, p);

   result = fill_bind(dev, p);
   if (result != VK_SUCCESS) {
      graphics_pipeline_destroy(&dev->vk, &p->base.vk, alloc);
      goto out;
   }

   *out = vk_pipeline_to_handle(&p->base.vk);
   result = VK_SUCCESS;

out:
   write_feedback(ci->pNext, stages, stage_count, cache_handle != VK_NULL_HANDLE,
                  os_time_get_nano() - start);
   for (uint32_t i = 0; i < stage_count; i++)
      stage_finish(dev, &stages[i]);
#if PAN_ARCH < 9
   stage_finish(dev, &clip);
#endif
   return result;
}

/* ---------------------------------------------------------------------- */
/* Compute                                                                 */

static VkResult
create_compute_pipeline(struct mali_device *dev, VkPipelineCache cache_handle,
                        const VkComputePipelineCreateInfo *ci,
                        const VkAllocationCallbacks *alloc, VkPipeline *out)
{
   const int64_t start = os_time_get_nano();
   const VkPipelineCreateFlags2KHR flags = vk_compute_pipeline_create_flags(ci);
   VK_FROM_HANDLE(vk_pipeline_layout, layout, ci->layout);
   struct vk_pipeline_cache *cache = pick_cache(dev, cache_handle);
   struct stage cs;
   VkResult result;

   *out = VK_NULL_HANDLE;

   stage_init(dev, flags, ci->pNext, &ci->stage, &cs);

   struct mesa_blake3 h;
   _mesa_blake3_init(&h);
   _mesa_blake3_update(&h, "cs", 2);
   _mesa_blake3_update(&h, cs.stage_hash, sizeof(cs.stage_hash));
   hash_layout(&h, layout);
   _mesa_blake3_final(&h, cs.key);

   stage_lookup(dev, cache, &cs);

   if (!cs.shader) {
      if (flags & VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT_KHR) {
         result = VK_PIPELINE_COMPILE_REQUIRED;
         goto out;
      }
      result = stage_compile(dev, cache, flags, &cs, layout, NULL, NULL);
      if (result != VK_SUCCESS)
         goto out;
   }

   struct mali_compute_pipeline *p =
      vk_pipeline_zalloc(&dev->vk, &compute_pipeline_ops,
                         VK_PIPELINE_BIND_POINT_COMPUTE, flags, alloc, sizeof(*p));
   if (!p) {
      result = vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto out;
   }
   p->base.vk.stages = VK_SHADER_STAGE_COMPUTE_BIT;
   if (layout)
      p->base.layout = vk_pipeline_layout_ref(layout);
   p->cs = cs.shader;
   cs.shader = NULL;

   *out = vk_pipeline_to_handle(&p->base.vk);
   result = VK_SUCCESS;

out:
   write_feedback(ci->pNext, &cs, 1, cache_handle != VK_NULL_HANDLE,
                  os_time_get_nano() - start);
   stage_finish(dev, &cs);
   return result;
}

/* ---------------------------------------------------------------------- */
/* Entry points                                                            */

/*
 * The multi-create rules of the spec: every handle is written, failures
 * are VK_NULL_HANDLE, the first error is returned, and
 * EARLY_RETURN_ON_FAILURE stops at the first failure.
 */
#define CREATE_PIPELINES(create_fn, infos, count, flags_fn)                    \
   VkResult result = VK_SUCCESS;                                               \
   uint32_t i = 0;                                                             \
   for (; i < (count); i++) {                                                  \
      VkResult r = create_fn(dev, pipelineCache, &(infos)[i], pAllocator,      \
                             &pPipelines[i]);                                  \
      if (r == VK_SUCCESS)                                                     \
         continue;                                                             \
      if (result == VK_SUCCESS)                                                \
         result = r;                                                           \
      if (flags_fn(&(infos)[i]) &                                              \
          VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT_KHR) {              \
         i++;                                                                  \
         break;                                                                \
      }                                                                        \
   }                                                                           \
   for (; i < (count); i++)                                                    \
      pPipelines[i] = VK_NULL_HANDLE;                                          \
   return result;

VKAPI_ATTR VkResult VKAPI_CALL
MALI_PER_ARCH(CreateGraphicsPipelines)(VkDevice _device, VkPipelineCache pipelineCache,
                                       uint32_t createInfoCount,
                                       const VkGraphicsPipelineCreateInfo *pCreateInfos,
                                       const VkAllocationCallbacks *pAllocator,
                                       VkPipeline *pPipelines)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   CREATE_PIPELINES(create_graphics_pipeline, pCreateInfos, createInfoCount,
                    vk_graphics_pipeline_create_flags)
}

VKAPI_ATTR VkResult VKAPI_CALL
MALI_PER_ARCH(CreateComputePipelines)(VkDevice _device, VkPipelineCache pipelineCache,
                                      uint32_t createInfoCount,
                                      const VkComputePipelineCreateInfo *pCreateInfos,
                                      const VkAllocationCallbacks *pAllocator,
                                      VkPipeline *pPipelines)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   CREATE_PIPELINES(create_compute_pipeline, pCreateInfos, createInfoCount,
                    vk_compute_pipeline_create_flags)
}
