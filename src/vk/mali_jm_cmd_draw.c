/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Draws on the job manager (v9): vkCmdDraw and vkCmdDrawIndexed as one
 * 384-byte Malloc Vertex job (IDVS) each, a tiler-side job in the open
 * batch's vtc chain. The binds, the dirty bits and the descriptors a draw
 * points at are shared with the v11 back half (mali_cmd_state.[ch]); this
 * file turns them into the words of the job.
 *
 * The job is built from a template (struct mali_jm_draw_tmpl): an image of
 * the next job in cached memory, in which each dirty group rewrites only
 * its own words, as the v11 draw rewrites only the staging registers whose
 * value changed. A draw then patches its counts, index pointer and header,
 * copies the 384 bytes into command memory in address order and stores
 * its address into the previous job's Next. Command memory is
 * write-combined and never read: the copy is six 64-byte stores, one per
 * line, in order, plus the 8-byte Next store behind them.
 *
 * The job's words follow Mesa's v9 gallium draw (pan_jm.c:
 * jm_emit_malloc_vertex_job, jm_emit_primitive, jm_emit_tiler_draw,
 * jm_emit_shader_env), "which source wins" for job encoding, with the
 * per-pipeline words our pipeline compile baked for v9
 * (mali_pipeline_state.c). The T820 blob writes the same job (its
 * draw_emit_malloc_vertex_job). Choices and differences:
 *
 *  - Allow rotating primitives is 0 in every draw (the baked Primitive
 *    word clears it; genxml's default is 1). Rotation would move the
 *    provoking vertex of flat-shaded triangles, and ARMSX2 shades flat a
 *    lot. The blob never sets it; Mesa sets it only without flat inputs.
 *    Whichever vertex is provoking (the Tiler Context's First provoking
 *    vertex, mali_jm_cmd_render.c) stays the one the pass's pipelines
 *    asked for, not one the hardware picked on its own;
 *  - the Draw descriptor's Vertex array is Packet = 1, pointer and
 *    strides 0 in every job: the hardware writes the vertex packet's
 *    address and strides there in Malloc Vertex mode (v9.xml), so a
 *    command buffer that runs again gets these words back with the header
 *    (mali_jm_cmd_note_reset); the blob's per-64-draw pre-pass re-arms
 *    the same words on the GPU, and we have no pre-pass;
 *  - the varying shader environment is written only when the Primitive
 *    word's Secondary Shader is set, with the position shader's resources,
 *    FAU and thread storage (Mesa; the blob writes the same), else it is
 *    zero; Allocation is the varying size + 16 and the varying size, or
 *    16 and 0 without a varying shader (Mesa's "no varyings" rule);
 *  - Instance offset is 0: zero-based instance IDs, firstInstance reaches
 *    the shader as a system value, as on v11;
 *  - the FAU block's word count goes into the Shader Environment's own
 *    byte (v9 has no count in the pointer); a zero count would drop the
 *    uniforms silently;
 *  - no occlusion query (ARMSX2 records none).
 *
 * Layered passes. Every layer has its own Tiler Context with layer offset
 * 0 (mali_jm_cmd_render.c). A primitive's layer is 0 unless the vertex
 * shader writes gl_Layer, which no v9 shader can: the device offers no
 * shader-output-layer feature and multiview is refused at compile time on
 * v9 (its tiler has no view mask). So a draw in a layered pass renders
 * into layer 0 only, as Vulkan says, and one job on layer 0's context is
 * the whole draw. The v11 back half runs one RUN_IDVS per 8-layer context
 * for the same result.
 */

#include "mali_cmd_state.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "util/log.h"
#include "vk_log.h"

#if PAN_ARCH != 9
#error "mali_jm_cmd_draw.c is the v9 (job manager) back half"
#endif

/* ---------------------------------------------------------------------- */
/* The template's layout                                                   */

/* 32-bit word indices into the Malloc Vertex job (v9.xml). */
#define MV_WORD(section) (MALI_MALLOC_VERTEX_JOB_SECTION_##section##_OFFSET / 4)
enum {
   TW_HDR_TYPE_INDEX = 4,              /* Type, Barrier, Index */
   TW_HDR_DEPS = 5,                    /* Dependency 1, Dependency 2 */
   TW_PRIM = MV_WORD(PRIMITIVE),       /* flags; +1 base vertex offset;
                                        * +2 instance offset; +3 index count */
   TW_INSTANCES = MV_WORD(INSTANCE_COUNT),
   TW_ALLOC = MV_WORD(ALLOCATION),
   TW_TILER = MV_WORD(TILER),          /* 64-bit */
   TW_SCISSOR = MV_WORD(SCISSOR),      /* 64-bit */
   TW_PSIZE = MV_WORD(PRIMITIVE_SIZE), /* the fixed size, a float */
   TW_INDICES = MV_WORD(INDICES),      /* 64-bit */
   TW_DRAW = MV_WORD(DRAW),
   /* The Draw descriptor's words. */
   TW_DCD0 = TW_DRAW + 0,
   TW_DCD1 = TW_DRAW + 1,
   TW_VARRAY = TW_DRAW + 2,            /* 3 words, GPU-written */
   TW_MIN_Z = TW_DRAW + 6,
   TW_MAX_Z = TW_DRAW + 7,
   TW_ZSD = TW_DRAW + 10,              /* 64-bit */
   TW_BLEND = TW_DRAW + 12,            /* 64-bit: address | count */
   TW_FS_ENV = TW_DRAW + 16,
   TW_POS_ENV = MV_WORD(POSITION),
   TW_VAR_ENV = MV_WORD(VARYING),
};

/* Shader Environment words (16 words). */
enum {
   ENV_FAU_COUNT = 1,
   ENV_RESOURCES = 8,                  /* 64-bit */
   ENV_SHADER = 10,                    /* 64-bit */
   ENV_TSD = 12,                       /* 64-bit */
   ENV_FAU = 14,                       /* 64-bit */
   ENV_WORDS = 16,
};

_Static_assert(MALI_MALLOC_VERTEX_JOB_LENGTH == sizeof(((struct mali_jm_draw_tmpl *)0)->w),
               "the template is one Malloc Vertex job");
_Static_assert(TW_PRIM == 8 && TW_INSTANCES == 12 && TW_ALLOC == 13 && TW_TILER == 14 &&
                  TW_SCISSOR == 26 && TW_PSIZE == 28 && TW_INDICES == 30 && TW_DRAW == 32 &&
                  TW_POS_ENV == 64 && TW_VAR_ENV == 80,
               "v9.xml Malloc Vertex Job layout");

static inline void
tset64(uint32_t *w, unsigned i, uint64_t v)
{
   memcpy(&w[i], &v, sizeof(v));
}

/* The FAU word of the builders (address | count << 56) into a Shader
 * Environment's FAU pointer and FAU count. */
static inline void
env_set_fau(uint32_t *env, uint64_t fau)
{
   env[ENV_FAU_COUNT] = (uint32_t)(fau >> 56);
   tset64(env, ENV_FAU, fau & BITFIELD64_MASK(56));
}

void
mali_jm_draw_tmpl_init(struct mali_jm_draw_tmpl *t)
{
   memset(t, 0, sizeof(*t));

   struct mali_job_header_packed h;
   pan_pack(&h, JOB_HEADER, cfg) {
      cfg.type = MALI_JOB_TYPE_MALLOC_VERTEX;
   }
   t->hdr4 = h.opaque[TW_HDR_TYPE_INDEX];

   /* The Vertex array words the hardware overwrites (file comment). */
   struct mali_vertex_array_packed va;
   pan_pack(&va, VERTEX_ARRAY, cfg) {
      cfg.packet = true;
   }
   memcpy(&t->w[TW_VARRAY], &va, sizeof(va));
}

/* The Primitive word's Secondary Shader bit. */
static inline uint32_t
prim_secondary_bit(void)
{
   struct mali_primitive_packed p;
   pan_pack_nodefaults(&p, PRIMITIVE, cfg) {
      cfg.secondary_shader = true;
   }
   return p.opaque[0];
}

/* The Primitive word's Point Size Array Format field: when it is not
 * NONE, the shader writes the point size and the job's Primitive Size
 * section holds the size array pointer instead of a fixed size. */
static inline uint32_t
prim_point_size_array_mask(void)
{
   struct mali_primitive_packed p;
   pan_pack_nodefaults(&p, PRIMITIVE, cfg) {
      cfg.point_size_array_format = MALI_POINT_SIZE_ARRAY_FORMAT_FP32;   /* both bits */
   }
   return p.opaque[0];
}

/* ---------------------------------------------------------------------- */
/* Draw state into the template                                            */

/*
 * The draw's state (mali_cmd_state.h, "A draw's state"): the shared
 * builders for every dirty group, each result stored into the template
 * right after it is built. Each group owns its words; words no group owns
 * are constant (mali_jm_draw_tmpl_init) or per draw.
 */
static bool
prepare_draw(struct mali_cmd_buffer *cmd, const struct mali_draw_info *di)
{
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   struct mali_render_state *r = &cmd->gfx.render;
   struct mali_jm_draw_tmpl *t = &cmd->jm.draw;
   uint32_t *w = t->w;
   uint32_t *const fs_env = &w[TW_FS_ENV];
   uint32_t *const pos_env = &w[TW_POS_ENV];
   const struct mali_graphics_pipeline *p = d->pipeline;
   const struct mali_gfx_bind *bd = &p->bind;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const struct mali_gfx_baked *dbk;
   uint64_t v;

   if (!mali_gfx_update_dirty(cmd, di, &dbk))
      return false;

   if (d->dirty & MALI_GFX_DIRTY_BLEND) {
      if (!mali_gfx_build_blend(cmd, &v))
         return false;
      tset64(w, TW_BLEND, v);
   }

   struct mali_graphics_sysvals sv;
   if (d->dirty & (MALI_GFX_DIRTY_VS_FAU | MALI_GFX_DIRTY_FS_FAU))
      mali_gfx_fill_sysvals(cmd, &sv);

   if (d->dirty & MALI_GFX_DIRTY_VS_FAU) {
      if (!mali_gfx_build_vs_fau(cmd, &sv, &v))
         return false;
      env_set_fau(pos_env, v);
   }
   if (d->dirty & MALI_GFX_DIRTY_FS_FAU) {
      if (!mali_gfx_build_fs_fau(cmd, &sv, &v))
         return false;
      /* v9 shaders keep no uniform-buffer words in FAU (no stream to
       * copy them with). */
      assert(!p->fs || !p->fs->fau.ubo_push_count);
      env_set_fau(fs_env, v);
   }
   if (d->dirty & MALI_GFX_DIRTY_VS_SRT) {
      if (!mali_gfx_build_vs_srt(cmd, &v))
         return false;
      tset64(pos_env, ENV_RESOURCES, v);
   }
   if (d->dirty & MALI_GFX_DIRTY_FS_SRT) {
      if (!mali_gfx_build_fs_srt(cmd, &v))
         return false;
      tset64(fs_env, ENV_RESOURCES, v);
   }
   if (d->dirty & MALI_GFX_DIRTY_ZSD) {
      if (!mali_gfx_build_zsd(cmd, dbk, &v))
         return false;
      tset64(w, TW_ZSD, v);
   }
   if (d->dirty & MALI_GFX_DIRTY_PIPELINE) {
      bool lines;
      uint64_t spd_var;
      if (unlikely(dbk)) {
         tset64(pos_env, ENV_SHADER, dbk->vs_pos_spd);
         tset64(fs_env, ENV_SHADER, dbk->fs_spd);
         spd_var = dbk->vs_var_spd;
         t->prim0 = dbk->tiler_flags;
         w[TW_DCD0] = dbk->dcd0[0];
         w[TW_DCD1] = dbk->dcd1;
         lines = u_reduced_prim(dbk->prim) == MESA_PRIM_LINES;
      } else {
         tset64(pos_env, ENV_SHADER, bd->spd[0]);
         tset64(fs_env, ENV_SHADER, bd->spd[2]);
         spd_var = bd->spd[1];
         t->prim0 = bd->tiler_flags;
         w[TW_DCD0] = bd->dcd0;
         w[TW_DCD1] = bd->dcd1;
         lines = bd->lines;
      }
      t->secondary = t->prim0 & prim_secondary_bit();
      /* The vertex packet: the varyings after a 16-byte position, or
       * position only. */
      t->packet_stride = t->secondary ? bd->vary_size + 16 : 16;
      w[TW_ALLOC] = t->packet_stride | (t->secondary ? bd->vary_size : 0) << 16;
      /* Lines: the line width. Points whose shader writes the size: the
       * field is the size array pointer, not used when the sizes come in
       * the vertex packets; 0, as Mesa writes, not a float's bits. Other
       * points: 1.0. */
      if (t->prim0 & prim_point_size_array_mask())
         w[TW_PSIZE] = 0;
      else
         w[TW_PSIZE] = fui(lines ? dyn->rs.line.width : 1.0f);
      tset64(&w[TW_VAR_ENV], ENV_SHADER, t->secondary ? spd_var : 0);
   }
   if (d->dirty & MALI_GFX_DIRTY_VIEWPORT) {
      const VkViewport *vp = &dyn->vp.viewports[0];
      tset64(w, TW_SCISSOR, mali_gfx_scissor_box(cmd));
      w[TW_MIN_Z] = fui(MIN2(vp->minDepth, vp->maxDepth));
      w[TW_MAX_Z] = fui(MAX2(vp->minDepth, vp->maxDepth));
   }
   if (d->dirty & MALI_GFX_DIRTY_PASS) {
      /* Layer 0's context (file comment). */
      tset64(w, TW_TILER, r->tiler);
      tset64(fs_env, ENV_TSD, r->tsd);
      tset64(pos_env, ENV_TSD, r->tsd);
   }
   /* The varying environment: the position one's resources, FAU and
    * thread storage with the varying shader, or nothing. */
   if (d->dirty & (MALI_GFX_DIRTY_PIPELINE | MALI_GFX_DIRTY_VS_FAU | MALI_GFX_DIRTY_VS_SRT |
                   MALI_GFX_DIRTY_PASS)) {
      uint32_t *var_env = &w[TW_VAR_ENV];
      if (t->secondary) {
         var_env[ENV_FAU_COUNT] = pos_env[ENV_FAU_COUNT];
         memcpy(&var_env[ENV_RESOURCES], &pos_env[ENV_RESOURCES], 8);
         memcpy(&var_env[ENV_TSD], &pos_env[ENV_TSD], 16);   /* TSD, FAU */
      } else {
         memset(var_env, 0, ENV_WORDS * 4);
      }
   }

   d->dirty = 0;
   return true;
}

/* ---------------------------------------------------------------------- */
/* Writing the job                                                         */

/* The 384 bytes as six 64-byte stores (one ST1 of four registers each, a
 * whole cache line of the 128-byte-aligned job) in address order: the
 * compiler may not reorder or split them (a library memcpy of this size
 * may store its last block first, or twice; clang reorders the stores of
 * an inlined one), so write-combined memory gets full lines, one after
 * the other. */
static inline void
copy_job(void *restrict dst, const void *restrict src)
{
#if defined(__aarch64__)
   for (unsigned i = 0; i < 6; i++) {
      const uint8x16x4_t v = vld1q_u8_x4((const uint8_t *)src + 64 * i);
      vst1q_u8_x4((uint8_t *)dst + 64 * i, v);
      __asm__ volatile("" ::: "memory");
   }
#else
   memcpy(dst, src, 384);
#endif
}

/* The template as the next job of chain c. */
static bool
emit_job(struct mali_cmd_buffer *cmd, struct mali_jm_chain *c)
{
   struct mali_jm_draw_tmpl *t = &cmd->jm.draw;
   struct mali_ptr job;

   if (likely(!c->pending && c->index < UINT16_MAX - 2)) {
      job = mali_cmd_alloc(cmd, pan_size(MALLOC_VERTEX_JOB), 128);
      if (!job.cpu)
         return false;
      const uint16_t index = ++c->index;
      t->w[TW_HDR_TYPE_INDEX] = t->hdr4 | (uint32_t)index << 16;
      t->w[TW_HDR_DEPS] = (uint32_t)c->tiler_dep << 16;   /* Dependency 2 */
      copy_job(job.cpu, t->w);
      mali_jm_chain_link(c, job, index, true);
      if (unlikely(cmd->jm.resubmit))
         mali_jm_cmd_note_reset(cmd, job.cpu, NULL, 16);
   } else {
      /* A barrier owed to this chain (a Cache Flush job first), or a
       * chain out of indices (an error): the general path. */
      job = mali_jm_cmd_add_job(cmd, c, MALI_JOB_TYPE_MALLOC_VERTEX,
                                pan_size(MALLOC_VERTEX_JOB), false, 0);
      if (!job.cpu)
         return false;
      memcpy((uint8_t *)job.cpu + 32, (const uint8_t *)t->w + 32,
             pan_size(MALLOC_VERTEX_JOB) - 32);
   }
   if (unlikely(cmd->jm.resubmit))
      mali_jm_cmd_note_reset(cmd, (uint8_t *)job.cpu + TW_VARRAY * 4, &t->w[TW_VARRAY], 12);
   return true;
}

static void
draw(struct mali_cmd_buffer *cmd, const struct mali_draw_info *di)
{
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   struct mali_render_state *r = &cmd->gfx.render;

   if (!di->count || !di->instance_count)
      return;
   if (!r->active || !d->pipeline || !d->pipeline->vs) {
      vk_logw(VK_LOG_OBJS(cmd), "malisx2: draw outside a render pass or without a pipeline");
      return;
   }
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   if (!prepare_draw(cmd, di)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   /* The draw's own words. */
   struct mali_jm_draw_tmpl *t = &cmd->jm.draw;
   uint32_t *w = t->w;
   uint64_t indices = 0;
   uint32_t index_type = MALI_INDEX_TYPE_NONE;
   if (di->indexed) {
      index_type = mali_gfx_index_type(d->ib.index_size);
      indices = d->ib.addr + (uint64_t)di->first_index * d->ib.index_size;
   }
   w[TW_PRIM] = t->prim0 | index_type << 8;   /* Index type, bits 8-10 */
   w[TW_PRIM + 1] = (uint32_t)di->vertex_offset;
   w[TW_PRIM + 3] = di->count;
   w[TW_INSTANCES] = di->instance_count;
   tset64(w, TW_INDICES, indices);

   if (!emit_job(cmd, mali_jm_cmd_pass_vtc(cmd)))
      return;

   /* Heap use: every vertex's packet (with an index count, an upper
    * bound) and about as much again for polygon-list entries. A guess
    * for closing batches early enough, tuned on the device. */
   mali_jm_cmd_heap_use(cmd, (uint64_t)di->count * di->instance_count *
                                (t->packet_stride + 16));
   cmd->jm.cur.draws++;
   r->draws++;
   cmd->draws++;
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
