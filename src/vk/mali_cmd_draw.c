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
 *  - RUN_IDVS sets bit 55, which v11.xml does not define, as every blob
 *    RUN_IDVS does (read as "Tiler Flags2 Enable", the RUN_FULLSCREEN
 *    field at the same bit). TILER_FLAGS2 is written as 0 before every
 *    draw either way.
 *  - The FAU blocks carry our sysval layout (mali_shader.h), built only
 *    when their inputs changed.
 *
 * Register use on the vertex/tiler subqueue is the v11 IDVS staging set;
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

#if PAN_ARCH != 11
#error "mali_cmd_draw.c is written for arch v11"
#endif

#include "mali_cmd_draw.h"

/* Bit 55 of RUN_IDVS; the on-device smoke test clears it once to check
 * what it does. */
bool mali_idvs_bit55 = true;

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
         emit_run_idvs(b, ovr.opaque[0]);
      }
      cs_move64_to(b, ctx, r->tiler);
   } else {
      b->req_resource_mask |= CS_IDVS_RES;
      assert(e.n < ARRAY_SIZE(e.ins));
      e.ins[e.n++] = run_idvs_ins(ovr.opaque[0]);
      batch_flush(b, &e);
   }
   r->draws++;
   cmd->draws++;
   if (unlikely(mh))
      mali_measure_end(cmd, mh);
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
   cs_move32_to(b, cs_reg32(b, IDVS(TILER_FLAGS2)), 0);

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
   cmd->gfx.draw.regs_valid = 0;
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
      cfg.flags_2.no_shader_depth_read = true;
      cfg.flags_2.no_shader_stencil_read = true;
      cfg.depth_stencil = zsd.gpu;
   }
   const VkRect2D all = {{0, 0}, {r->desc.width, r->desc.height}};
   MALI_PER_ARCH(cmd_run_fullscreen)(cmd, dcd.gpu, &all, 0, MAX2(r->desc.layer_count, 1));
}
