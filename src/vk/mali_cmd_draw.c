/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Draws on the command-stream frontend: vkCmdDraw / vkCmdDrawIndexed as
 * RUN_IDVS on the vertex/tiler subqueue, full-screen draws (RUN_FULLSCREEN)
 * for vkCmdClearAttachments (mali_cmd_meta_gfx.c) and in-pass barriers. The
 * binds, the dirty bits and the descriptors a draw points at are shared
 * with the job-manager back half (mali_cmd_state.[ch]); this file turns
 * them into register moves.
 *
 * The encoding follows panvk v11 (csf/panvk_vX_cmd_draw.c: prepare_draw,
 * launch_draw, cmd_run_fullscreen, cmd_fb_barrier), "which source wins"
 * for command-stream encoding; the per-pipeline words are the ones our
 * pipeline compile baked in (mali_pipeline_state.c). Differences:
 *
 *  - RUN_IDVS sets bit 55 on v11, which v11.xml does not define, as every
 *    blob RUN_IDVS does (read as "Tiler Flags2 Enable", the RUN_FULLSCREEN
 *    field at the same bit). TILER_FLAGS2 is written as 0 before every
 *    draw either way. v10 has no TILER_FLAGS2 register (genxml v10.xml:
 *    the IDVS staging registers end below it, and RUN_FULLSCREEN has no
 *    such field), so neither the bit nor the register is written there.
 *  - The FAU blocks carry our sysval layout (mali_shader.h), built only
 *    when their inputs changed.
 *
 * Register use on the vertex/tiler subqueue is the IDVS staging set;
 * state registers persist between draws and are rewritten from dirty
 * bits (MALI_GFX_DIRTY_*). What a draw emits before its RUN_IDVS is in
 * mali_cmd_draw.h, shared with the indirect draws
 * (mali_cmd_draw_indirect.c).
 */

#include "mali_cmd_state.h"
#include "mali_measure.h"

#include "vk_log.h"

#include "mali_arch.h"
#include "mali_queue.h"

#if PAN_ARCH < 10
#error "mali_cmd_draw.c is written for the command-stream archs (v10, v11, v12)"
#endif

#include "mali_cmd_draw.h"

#if PAN_ARCH >= 11
/* Bit 55 of RUN_IDVS; the on-device smoke test clears it once to check
 * what it does. */
bool mali_idvs_bit55 = true;
#endif

static void
draw_pass(struct mali_cmd_buffer *cmd, const struct mali_draw_info *di)
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
   if (mali_gfx_discards(cmd))
      return;

   struct idvs_batch e;
   e.d = d;
   e.n = 0;
   if (!prepare_draw(cmd, di, &e)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_VERTEX_TILER);
   batch_move32(&e, IDVS(INDEX_COUNT), di->count);
   batch_move32(&e, IDVS(INSTANCE_COUNT), di->instance_count);
   batch_move32(&e, IDVS(INDEX_OFFSET), di->indexed ? di->first_index : 0);
   batch_move32(&e, IDVS(VERTEX_OFFSET), (uint32_t)di->vertex_offset);
   /* Zero-based instance IDs; firstInstance reaches the shader as a
    * system value (panvk). */
   batch_move32(&e, IDVS(INSTANCE_OFFSET), 0);

   struct mali_primitive_flags_packed ovr;
   pan_pack_nodefaults(&ovr, PRIMITIVE_FLAGS, cfg) {
      cfg.index_type = di->indexed ? mali_gfx_index_type(d->ib.index_size) : MALI_INDEX_TYPE_NONE;
   }

   /* Per-draw timing ("draws" mode). */
   uint32_t mh = 0;
   if (unlikely(cmd->measure) && cmd->measure->draws) {
      batch_flush(b, &e);
      mh = mali_measure_begin(cmd, MALI_MEASURE_DRAW, MALI_SUBQUEUE_VERTEX_TILER,
                              cmd->draws);
      mali_measure_info(cmd, mh, di->count, di->instance_count, di->indexed,
                        cmd->passes - 1);
      uint64_t key = 0;
      if (d->pipeline->fs)
         memcpy(&key, d->pipeline->fs->key, sizeof(key));
      mali_measure_shader(cmd, mh, key);
      mali_measure_extra(cmd, mh, d->blend_shader_mask);
   }

   /* One RUN_IDVS per tiler context: a layered pass has one per 8
    * layers, and the draw's primitives may go to any of them. */
   if (r->td_count > 1) {
      batch_flush(b, &e);
      struct cs_index ctx = cs_reg64(b, IDVS(TILER_CTX));
      for (uint32_t i = 0; i < r->td_count; i++) {
         if (i)
            cs_add_imm64(b, ctx, ctx, pan_size(TILER_CONTEXT));
         emit_run_idvs(b, ovr.opaque[0], -1);
      }
      cs_move64_to(b, ctx, r->tiler);
   } else {
      b->req_resource_mask |= CS_IDVS_RES;
      assert(e.n < ARRAY_SIZE(e.ins));
      e.ins[e.n++] = run_idvs_ins(ovr.opaque[0], -1);
      batch_flush(b, &e);
   }
   r->draws++;
   cmd->draws++;
   if (unlikely(mh))
      mali_measure_end(cmd, mh);
}

static void
draw_viewports(struct mali_cmd_buffer *cmd, const struct mali_draw_info *di)
{
   const uint32_t passes = viewport_passes(cmd);
   if (likely(passes == 1)) {
      draw_pass(cmd, di);
      return;
   }
   for (uint32_t i = 0; i < passes; i++) {
      viewport_pass_begin(cmd, i);
      draw_pass(cmd, di);
   }
   viewport_pass_begin(cmd, 0);
}

/*
 * Polygon mode line: the triangles' edges as a line list (the pipeline's
 * tiler flags draw lines, mali_pipeline_state.c polygon_prim). A compute
 * job on the vertex/tiler stream writes six indices per triangle, from the
 * bound index buffer or the vertex positions, and the draw is an indexed
 * one of those. The topology is a triangle list, strip or fan; a strip's
 * restart is honoured per triangle only.
 */
static enum mesa_prim
polygon_lines_topology(const struct mali_cmd_buffer *cmd)
{
   /* The polygon mode is never dynamic here (no extended dynamic state 3);
    * the topology may be. */
   const struct mali_graphics_pipeline *p = cmd->gfx.draw.pipeline;
   if (!p || p->state.rs.polygon_mode != VK_POLYGON_MODE_LINE)
      return MESA_PRIM_COUNT;
   const VkPrimitiveTopology topology =
      BITSET_TEST(p->dynamic, MESA_VK_DYNAMIC_IA_PRIMITIVE_TOPOLOGY)
         ? cmd->vk.dynamic_graphics_state.ia.primitive_topology
         : p->state.ia.primitive_topology;
   const enum mesa_prim prim = vk_topology_to_mesa(topology);
   return prim == MESA_PRIM_TRIANGLES || prim == MESA_PRIM_TRIANGLE_STRIP ||
                prim == MESA_PRIM_TRIANGLE_FAN
             ? prim
             : MESA_PRIM_COUNT;
}

static void
draw_polygon_lines(struct mali_cmd_buffer *cmd, const struct mali_draw_info *di,
                   enum mesa_prim topology)
{
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const uint32_t tris = topology == MESA_PRIM_TRIANGLES ? di->count / 3
                         : di->count >= 3                 ? di->count - 2
                                                          : 0;
   if (!tris || !di->instance_count)
      return;
   const struct mali_shader *s = MALI_PER_ARCH(cmd_meta_shader)(cmd, MALI_META_POLYGON_LINES);
   struct mali_ptr dst = mali_cmd_alloc(cmd, (uint64_t)tris * 24, 64);
   if (!s || !dst.cpu)
      return;

   const uint32_t isz = di->indexed ? d->ib.index_size : 4;
   const uint32_t restart = di->indexed && dyn->ia.primitive_restart_enable
                               ? (uint32_t)BITFIELD64_MASK(isz * 8)
                               : 0;
   const struct mali_meta_lines_push push = {
      .src = di->indexed ? d->ib.addr : 0,
      .dst = dst.gpu,
      .count = tris,
      .first = di->indexed ? di->first_index : (uint32_t)di->vertex_offset,
      .index_size = isz,
      .mode = topology,
      .restart = restart,
   };
   const uint32_t groups[3] = {DIV_ROUND_UP(tris, 64), 1, 1};
   MALI_PER_ARCH(cmd_vt_dispatch_shader)(cmd, s, &push, sizeof(push), groups);

   struct mali_draw_info ld = *di;
   ld.count = tris * 6;
   ld.indexed = true;
   ld.first_index = 0;
   ld.vertex_offset = di->indexed ? di->vertex_offset : 0;
   const __typeof__(d->ib) ib = d->ib;
   d->ib.addr = dst.gpu;
   d->ib.size = tris * 24;
   d->ib.index_size = 4;
   d->dirty |= MALI_GFX_DIRTY_INDEX;
   draw_viewports(cmd, &ld);
   d->ib = ib;
   d->dirty |= MALI_GFX_DIRTY_INDEX;
}

#if MALI_WITH_POLY
#include "poly/tessellator.h"

/* The draw's topology: the pipeline's, or the command buffer's when it is
 * dynamic. */
static enum mesa_prim
draw_topology(const struct mali_cmd_buffer *cmd)
{
   const struct mali_graphics_pipeline *p = cmd->gfx.draw.pipeline;
   const VkPrimitiveTopology t = BITSET_TEST(p->dynamic, MESA_VK_DYNAMIC_IA_PRIMITIVE_TOPOLOGY)
                                    ? cmd->vk.dynamic_graphics_state.ia.primitive_topology
                                    : p->state.ia.primitive_topology;
   return vk_topology_to_mesa(t);
}

static uint64_t
upload(struct mali_cmd_buffer *cmd, const void *data, size_t size)
{
   struct mali_ptr p = mali_cmd_alloc(cmd, size, 64);
   if (!p.cpu)
      return 0;
   memcpy(p.cpu, data, size);
   return p.gpu;
}

/*
 * A draw with a geometry shader (mali_shader.c, "Geometry shaders"): the
 * vertex shader and the geometry shader run as compute jobs on this
 * stream, then the draw of the geometry shader's primitives. Direct draws.
 */
static void
draw_gs(struct mali_cmd_buffer *cmd, const struct mali_draw_info *di)
{
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   const struct mali_gs_shaders *g = d->pipeline->gs;
   const struct poly_gs_info *info = &g->info;
   const enum mesa_prim prim = draw_topology(cmd);
   const uint32_t verts = di->count, inst = di->instance_count;
   const uint32_t prims = u_decomposed_prims_for_vertices(prim, verts);
   if (!prims || !inst)
      return;

   /* The vertex shader's outputs: a vec4 per output written, per vertex
    * and instance (poly_vertex_output_address). */
   const uint32_t outputs = util_bitcount64(g->vs_outputs);
   struct mali_ptr vout = mali_cmd_alloc(cmd, MAX2((uint64_t)verts * inst * outputs * 16, 16), 64);
   if (!vout.cpu)
      return;

   const uint32_t wg[3] = {64, 1, 1};
   struct poly_vertex_params vp;
   poly_vertex_params_init(&vp, g->vs_outputs, wg);
   poly_vertex_params_set_draw(&vp, verts, inst);
   vp.output_buffer = vout.gpu;
   if (di->indexed) {
      const uint32_t isz = d->ib.index_size;
      vp.index_buffer = d->ib.addr + (uint64_t)di->first_index * isz;
      vp.index_size_B = isz;
      vp.index_buffer_range_el = poly_index_buffer_range_el(d->ib.size / isz, di->first_index);
   }

   struct poly_geometry_params gp;
   poly_geometry_params_init(&gp, prim, wg);
   poly_geometry_params_set_draw(&gp, prim, info->shape, info->max_indices, verts, inst);
   gp.count_buffer_stride = info->count_words * 4;
   if (gp.count_buffer_stride) {
      struct mali_ptr c = mali_cmd_alloc(cmd, 16, 16);
      if (!c.cpu)
         return;
      memset(c.cpu, 0, 16);
      gp.count_buffer = c.gpu;
   }
   for (unsigned i = 0; i < POLY_MAX_SO_BUFFERS; i++)
      gp.xfb_offs_ptrs[i] = gp.xfb_base_original[i] = PAN_SHADER_OOB_ADDRESS;
   for (unsigned i = 0; i < POLY_MAX_VERTEX_STREAMS; i++)
      gp.xfb_prims_generated_counter[i] = gp.prims_generated_counter[i] =
         gp.xfb_overflow[i] = PAN_SHADER_OOB_ADDRESS;
   gp.xfb_any_overflow = PAN_SHADER_OOB_ADDRESS;

   uint64_t ib = 0;
   uint32_t ib_size = 0, isz = 0;
   if (info->shape == POLY_GS_SHAPE_DYNAMIC_INDEXED) {
      struct mali_ptr i = mali_cmd_alloc(cmd, (uint64_t)gp.draw.index_count * 4, 64);
      if (!i.cpu)
         return;
      gp.output_index_buffer = i.gpu;
      ib = i.gpu;
      ib_size = gp.draw.index_count * 4;
      isz = 4;
   } else if (info->shape == POLY_GS_SHAPE_STATIC_INDEXED) {
      ib = upload(cmd, info->topology, info->max_indices);
      ib_size = info->max_indices;
      isz = 1;
   }

   d->poly_vertex_params = upload(cmd, &vp, sizeof(vp));
   d->poly_geometry_params = upload(cmd, &gp, sizeof(gp));
   if (!d->poly_vertex_params || !d->poly_geometry_params || (isz && !ib))
      return;

   /* The vertex shader over (vertices, instances), then the geometry
    * shader over (primitives, instances), with this draw's base vertex and
    * instance (the rasterization draw's are 0). */
   d->first_vertex = di->vertex_offset;
   d->base_instance = di->first_instance;
   struct mali_graphics_sysvals sv;
   mali_gfx_fill_sysvals(cmd, &sv);
   const uint64_t vs_fau = MALI_PER_ARCH(cmd_gfx_fau)(cmd, g->sw_vs, &sv, cmd->push_constants,
                                                       sizeof(cmd->push_constants));
   const uint64_t vs_srt = mali_gfx_shader_vs_srt(cmd, g->sw_vs, NULL);
   const uint64_t gs_fau = MALI_PER_ARCH(cmd_gfx_fau)(cmd, g->main, &sv, cmd->push_constants,
                                                       sizeof(cmd->push_constants));
   const uint64_t gs_srt = mali_gfx_shader_vs_srt(cmd, g->main, NULL);
   if (!vs_srt || !gs_srt || (g->sw_vs->fau.total_count && !vs_fau) ||
       (g->main->fau.total_count && !gs_fau)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }
   const uint32_t vs_groups[3] = {DIV_ROUND_UP(verts, 64), inst, 1};
   MALI_PER_ARCH(cmd_vt_dispatch)(cmd, g->sw_vs, vs_srt, vs_fau, vs_groups);
   const uint32_t gs_groups[3] = {DIV_ROUND_UP(prims, 64), inst, 1};
   MALI_PER_ARCH(cmd_vt_dispatch)(cmd, g->main, gs_srt, gs_fau, gs_groups);

   /* The draw of what it emitted, by the rasterization shader. */
   struct mali_draw_info ld = {
      .count = poly_gs_rast_vertices(info->shape, info->max_indices, prims, inst),
      .instance_count = poly_gs_rast_instances(info->shape, info->max_indices, prims, inst),
      .indexed = isz != 0,
   };
   const __typeof__(d->ib) saved_ib = d->ib;
   if (isz) {
      d->ib.addr = ib;
      d->ib.size = ib_size;
      d->ib.index_size = isz;
      d->dirty |= MALI_GFX_DIRTY_INDEX;
   }
   d->dirty |= MALI_GFX_DIRTY_VS_FAU | MALI_GFX_DIRTY_VS_SRT;
   draw_viewports(cmd, &ld);
   if (isz) {
      d->ib = saved_ib;
      d->dirty |= MALI_GFX_DIRTY_INDEX;
   }
}

/* Bytes of the tessellator's heap, after its struct poly_heap. Asahi's
 * is 128 MiB. */
#define POLY_HEAP_SIZE (64u * 1024 * 1024)
#define POLY_HEAP_HEADER 64

/* The tessellator's heap (struct poly_heap's GPU address), emptied before
 * this command buffer's first use of it: everything that reads or writes it
 * runs on this stream, so once the stream has drained, what an earlier
 * command buffer allocated is no longer read. */
static uint64_t
poly_heap(struct mali_cmd_buffer *cmd)
{
   struct mali_device *dev = cmd->dev;
   simple_mtx_lock(&dev->meta_lock);
   if (!dev->poly_heap.size) {
      const struct mali_kbase_alloc_info ai = {
         .size = POLY_HEAP_HEADER + POLY_HEAP_SIZE,
         .flags = mali_kbase_mem_same_va_policy(MALI_KBASE_FLAGS_DEVICE_UNCACHED),
         .mem_class = MALI_KBASE_MEM_CLASS_DEVICE_CPU_UNCACHED,
      };
      if (mali_kbase_alloc(dev->kbase, &ai, &dev->poly_heap) == MALI_KBASE_SUCCESS) {
         struct poly_heap *h = dev->poly_heap.cpu;
         *h = (struct poly_heap){
            .base = dev->poly_heap.gpu_va + POLY_HEAP_HEADER,
            .size = POLY_HEAP_SIZE,
         };
      } else {
         dev->poly_heap = (struct mali_kbase_bo){0};
      }
   }
   simple_mtx_unlock(&dev->meta_lock);
   if (!dev->poly_heap.size)
      return 0;

   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   if (!d->poly_heap_reset) {
      struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_VERTEX_TILER);
      cs_wait_slots(b, cmd->dev->csf->sb.all_iters_mask);
      const struct cs_index addr = mali_cs_scratch_reg64(b, 0);
      const struct cs_index zero = mali_cs_scratch_reg32(b, 2);
      cs_move64_to(b, addr, dev->poly_heap.gpu_va + offsetof(struct poly_heap, bottom));
      cs_move32_to(b, zero, 0);
      cs_store32(b, zero, addr, 0);
      cs_flush_stores(b);
      d->poly_heap_reset = true;
   }
   return dev->poly_heap.gpu_va;
}

/*
 * A draw with tessellation (mali_shader.c, "Tessellation"): on this stream
 * the vertex shader, the control shader, the tessellator's count, offset
 * and index passes, then the indexed draw of its output, whose command the
 * tessellator wrote. Direct draws.
 */
static void
draw_tess(struct mali_cmd_buffer *cmd, const struct mali_draw_info *di)
{
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   const struct mali_graphics_pipeline *p = d->pipeline;
   const struct mali_tess_shaders *t = p->tess;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const uint32_t patch_size =
      BITSET_TEST(p->dynamic, MESA_VK_DYNAMIC_TS_PATCH_CONTROL_POINTS)
         ? dyn->ts.patch_control_points
         : p->state.ts.patch_control_points;
   const uint32_t verts = di->count, inst = di->instance_count;
   const uint32_t in_patches = patch_size ? verts / patch_size : 0;
   const uint32_t patches = in_patches * inst;
   if (!patches)
      return;

   const uint64_t heap = poly_heap(cmd);
   const uint32_t stride = t->tcs_output_stride;
   const uint64_t tcs_out = (uint64_t)patches * stride;
   const uint64_t size = tcs_out + 2ull * patches * 4 + 32;
   struct mali_ptr blob = mali_cmd_alloc(cmd, size, 64);
   const uint32_t outputs = util_bitcount64(t->vs_outputs);
   struct mali_ptr vout = mali_cmd_alloc(cmd, MAX2((uint64_t)verts * inst * outputs * 16, 16), 64);
   if (!heap || !blob.cpu || !vout.cpu) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }
   memset(blob.cpu, 0, size);

   const VkTessellationDomainOrigin origin =
      BITSET_TEST(p->dynamic, MESA_VK_DYNAMIC_TS_DOMAIN_ORIGIN) ? dyn->ts.domain_origin
                                                                : p->state.ts.domain_origin;
   struct poly_tess_params tp = {
      .heap = heap,
      .patch_coord_buffer = heap + POLY_HEAP_HEADER,
      .tcs_buffer = blob.gpu,
      .coord_allocs = blob.gpu + tcs_out,
      .counts = blob.gpu + tcs_out + patches * 4,
      .out_draws = blob.gpu + tcs_out + 2 * patches * 4,
      .statistic = 0,
      .tcs_per_vertex_outputs = t->tcs_per_vertex_outputs,
      .input_patch_size = patch_size,
      .output_patch_size = t->tcs_output_patch_size,
      .tcs_patch_constants = t->tcs_patch_outputs,
      .patches_per_instance = in_patches,
      .tcs_stride_el = stride / 4,
      .nr_patches = patches,
      .partitioning = t->spacing == TESS_SPACING_EQUAL ? POLY_TESS_PARTITIONING_INTEGER
                      : t->spacing == TESS_SPACING_FRACTIONAL_ODD
                         ? POLY_TESS_PARTITIONING_FRACTIONAL_ODD
                         : POLY_TESS_PARTITIONING_FRACTIONAL_EVEN,
      .points_mode = t->points,
      .isolines = t->mode == TESS_PRIMITIVE_ISOLINES,
   };
   if (!t->points && t->mode != TESS_PRIMITIVE_ISOLINES)
      tp.ccw = t->ccw ^ (origin == VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT);

   const uint32_t wg[3] = {64, 1, 1};
   struct poly_vertex_params vp;
   poly_vertex_params_init(&vp, t->vs_outputs, wg);
   poly_vertex_params_set_draw(&vp, verts, inst);
   vp.output_buffer = vout.gpu;
   if (di->indexed) {
      const uint32_t isz = d->ib.index_size;
      vp.index_buffer = d->ib.addr + (uint64_t)di->first_index * isz;
      vp.index_size_B = isz;
      vp.index_buffer_range_el = poly_index_buffer_range_el(d->ib.size / isz, di->first_index);
   }
   d->poly_vertex_params = upload(cmd, &vp, sizeof(vp));
   d->poly_tess_params = upload(cmd, &tp, sizeof(tp));
   if (!d->poly_vertex_params || !d->poly_tess_params)
      return;

   d->first_vertex = di->vertex_offset;
   d->base_instance = di->first_instance;
   struct mali_graphics_sysvals sv;
   mali_gfx_fill_sysvals(cmd, &sv);
   const uint64_t vs_fau = MALI_PER_ARCH(cmd_gfx_fau)(cmd, t->sw_vs, &sv, cmd->push_constants,
                                                       sizeof(cmd->push_constants));
   const uint64_t vs_srt = mali_gfx_shader_vs_srt(cmd, t->sw_vs, NULL);
   const uint64_t tcs_fau = MALI_PER_ARCH(cmd_gfx_fau)(cmd, t->tcs, &sv, cmd->push_constants,
                                                        sizeof(cmd->push_constants));
   const uint64_t tcs_srt = mali_gfx_shader_vs_srt(cmd, t->tcs, NULL);
   if (!vs_srt || !tcs_srt || (t->sw_vs->fau.total_count && !vs_fau) ||
       (t->tcs->fau.total_count && !tcs_fau)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }
   const uint32_t vs_groups[3] = {DIV_ROUND_UP(verts, 64), inst, 1};
   MALI_PER_ARCH(cmd_vt_dispatch)(cmd, t->sw_vs, vs_srt, vs_fau, vs_groups);
   const uint32_t tcs_groups[3] = {in_patches, inst, 1};
   MALI_PER_ARCH(cmd_vt_dispatch)(cmd, t->tcs, tcs_srt, tcs_fau, tcs_groups);

   const unsigned domain = t->mode == TESS_PRIMITIVE_ISOLINES ? 0
                           : t->mode == TESS_PRIMITIVE_QUADS  ? 2
                                                              : 1;
   const struct mali_meta_tess_push push = {.params = d->poly_tess_params};
   const uint32_t groups[3] = {DIV_ROUND_UP(patches, 64), 1, 1};
   const uint32_t one[3] = {1, 1, 1};
   const struct mali_shader *count =
      MALI_PER_ARCH(cmd_meta_shader)(cmd, MALI_META_TESS_COUNT_ISOLINES + domain);
   const struct mali_shader *finish = MALI_PER_ARCH(cmd_meta_shader)(cmd, MALI_META_TESS_FINISH);
   const struct mali_shader *write =
      MALI_PER_ARCH(cmd_meta_shader)(cmd, MALI_META_TESS_WRITE_ISOLINES + domain);
   if (!count || !finish || !write)
      return;
   MALI_PER_ARCH(cmd_vt_dispatch_shader)(cmd, count, &push, sizeof(push), groups);
   MALI_PER_ARCH(cmd_vt_dispatch_shader)(cmd, finish, &push, sizeof(push), one);
   MALI_PER_ARCH(cmd_vt_dispatch_shader)(cmd, write, &push, sizeof(push), groups);

   /* The draw of the tessellator's indices, into the heap. */
   const __typeof__(d->ib) saved_ib = d->ib;
   d->ib.addr = heap + POLY_HEAP_HEADER;
   d->ib.size = POLY_HEAP_SIZE;
   d->ib.index_size = 4;
   d->dirty |= MALI_GFX_DIRTY_INDEX | MALI_GFX_DIRTY_VS_FAU | MALI_GFX_DIRTY_VS_SRT;
   MALI_PER_ARCH(cmd_draw_indexed_indirect_one)(cmd, tp.out_draws);
   d->ib = saved_ib;
   d->dirty |= MALI_GFX_DIRTY_INDEX;
}
#endif /* MALI_WITH_POLY */

static void
draw(struct mali_cmd_buffer *cmd, const struct mali_draw_info *di)
{
#if MALI_WITH_POLY
   if (unlikely(cmd->gfx.draw.pipeline && cmd->gfx.draw.pipeline->gs)) {
      draw_gs(cmd, di);
      return;
   }
   if (unlikely(cmd->gfx.draw.pipeline && cmd->gfx.draw.pipeline->tess)) {
      draw_tess(cmd, di);
      return;
   }
#endif
   const enum mesa_prim lines = polygon_lines_topology(cmd);
   if (unlikely(lines != MESA_PRIM_COUNT)) {
      draw_polygon_lines(cmd, di, lines);
      return;
   }
   draw_viewports(cmd, di);
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

/* ---------------------------------------------------------------------- */
/* Full-screen draws                                                       */

void
MALI_PER_ARCH(cmd_run_fullscreen)(struct mali_cmd_buffer *cmd, uint64_t dcd, const VkRect2D *rect,
                                  uint32_t base_layer, uint32_t layer_count)
{
   struct mali_render_state *r = &cmd->gfx.render;
   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_VERTEX_TILER);

   if (!MALI_PER_ARCH(cmd_render_tiler)(cmd)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   int minx = MAX2(rect->offset.x, 0), miny = MAX2(rect->offset.y, 0);
   int maxx = MIN2(rect->offset.x + (int)rect->extent.width, (int)r->desc.width) - 1;
   int maxy = MIN2(rect->offset.y + (int)rect->extent.height, (int)r->desc.height) - 1;
   if (maxx < minx || maxy < miny)
      return;
   struct mali_scissor_packed sc;
   pan_pack(&sc, SCISSOR, cfg) {
      cfg.scissor_minimum_x = minx;
      cfg.scissor_minimum_y = miny;
      cfg.scissor_maximum_x = maxx;
      cfg.scissor_maximum_y = maxy;
   }
   cs_move64_to(b, cs_reg64(b, IDVS(SCISSOR_BOX)), sc.opaque[0] | (uint64_t)sc.opaque[1] << 32);
#if PAN_ARCH >= 11
   cs_move32_to(b, cs_reg32(b, IDVS(TILER_FLAGS2)), 0);
#endif

   struct cs_index dcd_reg = mali_cs_scratch_reg64(b, 0);
   cs_move64_to(b, dcd_reg, dcd);

   /* RUN_FULLSCREEN draws into the layers its tiler flags' view mask
    * selects inside the current tiler context (panvk cmd_run_fullscreen). */
   const uint32_t first_td = base_layer / MALI_LAYERS_PER_TILER_CTX;
   const uint32_t last_td = (base_layer + layer_count - 1) / MALI_LAYERS_PER_TILER_CTX;
   for (uint32_t td = first_td; td <= last_td && td < r->td_count; td++) {
      const uint32_t td_base = td * MALI_LAYERS_PER_TILER_CTX;
      const uint32_t first = MAX2(base_layer, td_base);
      const uint32_t last = MIN2(base_layer + layer_count, td_base + MALI_LAYERS_PER_TILER_CTX);
      struct mali_primitive_flags_packed tf;
      pan_pack(&tf, PRIMITIVE_FLAGS, cfg) {
         cfg.low_depth_cull = false;
         cfg.high_depth_cull = false;
         cfg.view_mask = BITFIELD_RANGE(first - td_base, last - first);
      }
      cs_move64_to(b, cs_reg64(b, IDVS(TILER_CTX)), r->tiler + td * pan_size(TILER_CONTEXT));
      cs_move32_to(b, cs_reg32(b, IDVS(TILER_FLAGS)), tf.opaque[0]);
      cs_run_fullscreen(b, 0, dcd_reg);
   }

   r->draws++;
   cmd->gfx.draw.dirty |= MALI_GFX_DIRTY_PIPELINE | MALI_GFX_DIRTY_VIEWPORT |
                          MALI_GFX_DIRTY_PASS;
   /* The scissor, tiler flags and tiler context registers changed behind
    * the draws' record of them. */
   BITSET_ZERO(cmd->gfx.draw.regs_valid);
}

/* A primitive barrier over the whole framebuffer: later fragments of the
 * pass wait for earlier ones in each tile (panvk cmd_fb_barrier). */
void
MALI_PER_ARCH(cmd_fb_barrier)(struct mali_cmd_buffer *cmd)
{
   struct mali_render_state *r = &cmd->gfx.render;
   struct mali_ptr zsd = mali_cmd_alloc(cmd, pan_size(DEPTH_STENCIL), 32);
   struct mali_ptr dcd = mali_cmd_alloc(cmd, pan_size(DRAW), 64);
   if (!zsd.cpu || !dcd.cpu)
      return;

   pan_cast_and_pack(zsd.cpu, DEPTH_STENCIL, cfg) {
      cfg.stencil_test_enable = false;
      cfg.depth_write_enable = false;
      cfg.depth_function = MALI_FUNC_ALWAYS;
   }
   pan_cast_and_pack(dcd.cpu, DRAW, cfg) {
      cfg.flags_0.allow_forward_pixel_to_kill = false;
      cfg.flags_0.allow_forward_pixel_to_be_killed = false;
      cfg.flags_0.primitive_barrier = true;
      cfg.flags_0.occlusion_query = MALI_OCCLUSION_MODE_DISABLED;
      cfg.flags_2.read_mask = 0;
      cfg.flags_2.write_mask = 0;
#if PAN_ARCH >= 11
      cfg.flags_2.no_shader_depth_read = true;
      cfg.flags_2.no_shader_stencil_read = true;
#endif
      cfg.depth_stencil = zsd.gpu;
   }
   const VkRect2D all = {{0, 0}, {r->desc.width, r->desc.height}};
   MALI_PER_ARCH(cmd_run_fullscreen)(cmd, dcd.gpu, &all, 0, MAX2(r->desc.layer_count, 1));
}
