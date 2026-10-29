/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Query commands on the job manager (v7 and v9): vkCmdResetQueryPool,
 * vkCmdWriteTimestamp, and the occlusion queries' vkCmdBeginQuery and
 * vkCmdEndQuery. The pool object and the host reads are in mali_query.c. ARMSX2 turns GPU timing on for any device whose name is
 * not "Mali-G615", so the G57 and G52 builds need the timestamp commands;
 * the G615 build has its own (mali_cmd_query.c).
 *
 * A query is 16 bytes of host-visible, uncached device memory: the value
 * System Timestamp writes at +0, an Immediate 64 availability flag at +8
 * (our own layout; nothing reads it but our own vkGetQueryPoolResults).
 * Both words are Write Value jobs with the header Barrier bit (the blob's
 * own query and timestamp jobs are the same two types, same bit).
 * mali_jm_measure_begin/end (timing_jm.c) use the identical pair for
 * MALISX2_MEASURE's timing regions; this file does not share code with
 * them beyond mali_jm_cmd_write_value, because a query's two jobs are not
 * a "region" (no CSV row, no handle to end later from a different call).
 */

#include "mali_cmd_state.h"

#include <string.h>

#include "vk_log.h"

#include "mali_query.h"

void
MALI_PER_ARCH(cmd_query_fence)(struct mali_cmd_buffer *cmd, bool before)
{
   /* The dispatches are transfers on the vtc slot. */
   if (before)
      mali_jm_cmd_barrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                          VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                          VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
   else
      mali_jm_cmd_barrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                          VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdResetQueryPool)(VkCommandBuffer commandBuffer, VkQueryPool _pool,
                                 uint32_t firstQuery, uint32_t queryCount)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(mali_query_pool, pool, _pool);

   if (!queryCount)
      return;

   /* A few queries (ARMSX2 resets two at a time): Write Value jobs. */
   if (queryCount <= MALI_QUERY_RESET_JOBS_MAX) {
      /* Not valid inside a render pass instance (Vulkan). */
      struct mali_jm_chain *c = mali_jm_cmd_vtc(cmd);
      if (!c)
         return;
      for (uint32_t i = 0; i < queryCount; i++) {
         const uint64_t addr = mali_query_val_va(pool, firstQuery + i);
         if (!mali_jm_cmd_write_value(cmd, c, MALI_WRITE_VALUE_TYPE_ZERO, addr, 0, true) ||
             !mali_jm_cmd_write_value(cmd, c, MALI_WRITE_VALUE_TYPE_ZERO,
                                      mali_query_avail_va(pool, firstQuery + i), 0, true))
            return;
      }
      return;
   }

   /* Many (a pool of thousands is reset whole): one fill over the slots,
    * after whatever earlier work wrote them and before whatever uses them
    * next. */
   MALI_PER_ARCH(cmd_query_fence)(cmd, true);
   MALI_PER_ARCH(cmd_fill_addr)(cmd, mali_query_val_va(pool, firstQuery),
                                (uint64_t)queryCount * pool->val_stride, 0);
   MALI_PER_ARCH(cmd_query_fence)(cmd, false);
}

/* Which job chain a timestamp of pipelineStage waits for: the vtc chain
 * only when the stage is exclusively vertex/tiler/compute-side (as
 * mali_jm_queue.c's VTC_ONLY_STAGES classifies a semaphore wait's
 * destination); the fragment chain otherwise, including top/bottom of
 * pipe and "all commands", because a batch's fragment atom already
 * depends on its own vtc atom and the pass/dispatch recorded since asks
 * for the latest point reached so far either way. */
static bool
stage_is_vtc_only(VkPipelineStageFlags2 stage)
{
   const VkPipelineStageFlags2 vtc_only =
      VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT |
      VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT |
      VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
      VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT |
      VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT |
      VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
      VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT;
   return stage && !(stage & ~vtc_only);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdWriteTimestamp)(VkCommandBuffer commandBuffer, VkPipelineStageFlagBits pipelineStage,
                                 VkQueryPool _pool, uint32_t query)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(mali_query_pool, pool, _pool);

   struct mali_jm_chain *c = stage_is_vtc_only((VkPipelineStageFlags2)pipelineStage)
                                ? mali_jm_cmd_vtc(cmd)
                                : mali_jm_cmd_frag(cmd, false);
   if (!c)
      return;
   if (!mali_jm_cmd_write_value(cmd, c, MALI_WRITE_VALUE_TYPE_SYSTEM_TIMESTAMP,
                                mali_query_val_va(pool, query), 0, true))
      return;
   mali_jm_cmd_write_value(cmd, c, MALI_WRITE_VALUE_TYPE_IMMEDIATE_64,
                           mali_query_avail_va(pool, query), 1, true);
}

/* ---------------------------------------------------------------------- */
/* Occlusion queries                                                       */

/*
 * The hardware counts. A draw's Draw descriptor names a counter (the
 * Occlusion pointer) and a mode (DCD Flags 0); the fragment job adds the
 * samples that passed to it, tile by tile, so the count is only final when
 * the fragment jobs of every pass that drew into the query have finished.
 *
 *  - Begin: a Write Value job zeroes the counter, in the open batch's vtc
 *    chain. The batch's fragment atom starts after its vtc atom, so the
 *    counter is zero before any fragment job can add to it. The draws that
 *    follow carry the counter (MALI_GFX_DIRTY_OQ).
 *  - End: the draws stop carrying it, and the availability word is set by
 *    a Write Value job with the Barrier bit on the fragment chain, behind
 *    the fragment jobs recorded so far: the barrier makes it start when
 *    they are done, and it is in the same atom as they are or in a later
 *    one of the fragment slot, which runs after them. The count and the
 *    availability word are in one 16-byte slot (one cache line), so the
 *    cache clean at the end of the atom publishes both together.
 *
 *    When the query ends inside a render pass, the pass's fragment jobs
 *    do not exist yet (the pass writes them when it ends), so the word is
 *    set at the end of the pass (cmd_oq_pass_end), after them. A query
 *    that ends outside a pass is set at once, behind the fragment jobs of
 *    the passes it spanned.
 */

void
MALI_PER_ARCH(cmd_oq_pass_end)(struct mali_cmd_buffer *cmd)
{
   if (likely(!util_dynarray_num_elements(&cmd->jm.oq_done, uint64_t)))
      return;
   struct mali_jm_chain *c = mali_jm_cmd_frag(cmd, false);
   if (c) {
      util_dynarray_foreach(&cmd->jm.oq_done, uint64_t, addr) {
         if (!mali_jm_cmd_write_value(cmd, c, MALI_WRITE_VALUE_TYPE_IMMEDIATE_64, *addr, 1, true))
            break;
      }
   }
   util_dynarray_clear(&cmd->jm.oq_done);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdBeginQuery)(VkCommandBuffer commandBuffer, VkQueryPool _pool, uint32_t query,
                             VkQueryControlFlags flags)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(mali_query_pool, pool, _pool);
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;

   if (pool->vk.query_type != VK_QUERY_TYPE_OCCLUSION)
      return;

   /* The pass's vtc chain once it tiles (the batch may not close inside
    * it), else the open batch's. */
   const uint64_t counter = mali_query_val_va(pool, query);
   struct mali_jm_chain *c = cmd->gfx.render.active && cmd->gfx.render.tiler
                                ? mali_jm_cmd_pass_vtc(cmd)
                                : mali_jm_cmd_vtc(cmd);
   /* No Barrier bit: nothing in the chain reads or writes the counter, and
    * a barrier here would hold the pass's tiling back until every draw
    * before it was done. */
   if (!c || !mali_jm_cmd_write_value(cmd, c, MALI_WRITE_VALUE_TYPE_ZERO, counter, 0, false))
      return;

   d->oq_addr = counter;
   d->oq_precise = flags & VK_QUERY_CONTROL_PRECISE_BIT;
   d->dirty |= MALI_GFX_DIRTY_OQ;
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdEndQuery)(VkCommandBuffer commandBuffer, VkQueryPool _pool, uint32_t query)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(mali_query_pool, pool, _pool);
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;

   if (pool->vk.query_type != VK_QUERY_TYPE_OCCLUSION)
      return;

   d->oq_addr = 0;
   d->dirty |= MALI_GFX_DIRTY_OQ;

   const uint64_t avail = mali_query_avail_va(pool, query);
   if (cmd->gfx.render.active) {
      uint64_t *slot = util_dynarray_grow(&cmd->jm.oq_done, uint64_t, 1);
      if (slot)
         *slot = avail;
      else
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }
   struct mali_jm_chain *c = mali_jm_cmd_frag(cmd, false);
   if (c)
      mali_jm_cmd_write_value(cmd, c, MALI_WRITE_VALUE_TYPE_IMMEDIATE_64, avail, 1, true);
}
