/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Query commands on the job manager (v7 and v9): vkCmdResetQueryPool,
 * vkCmdWriteTimestamp. The pool object and the host reads are in
 * mali_query.c. ARMSX2 turns GPU timing on for any device whose name is
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
