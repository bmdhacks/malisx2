/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Query commands on the command-stream frontend (v11): vkCmdResetQueryPool
 * and vkCmdWriteTimestamp. (The job manager's are in mali_jm_query.c; the
 * pool object and the host reads are in mali_query.c.)
 *
 * A timestamp is two deferred instructions on one subqueue's stream:
 * STORE_STATE{Timestamp} into the query's value word, then SYNC_SET64 of
 * the availability word to 1 once the store is done. Both words are in CSF
 * event memory (mali_query.h), so the host sees them as they land. The
 * store waits for the stream's earlier jobs (every iterator scoreboard
 * entry) except for a top-of-pipe stamp, and for the previous timestamp's
 * set, so stamps written by one stream never go backwards.
 *
 * The subqueue is picked by the stage: the vertex/tiler stream for the
 * vertex-side stages and top of pipe, the compute stream for the compute
 * shader stage, the fragment stream for everything else (bottom of pipe
 * and all commands included). A stamp is therefore ordered after the work
 * of its own subqueue only; the others are not waited for.
 */

#include "mali_cmd_state.h"

#include "mali_arch.h"
#include "mali_query.h"
#include "mali_queue.h"

#if PAN_ARCH != 11
#error "mali_cmd_query.c is written for arch v11"
#endif

/* The memory dependency around a dispatch that touches query memory. A
 * "before" fence is a barrier from every subqueue to the compute subqueue
 * (the dispatches of mali_cmd_copy.c run there) whose sources also wait for
 * the deferred operations that set availability words: they signal
 * MALI_SB_DEFERRED_SYNC / MALI_SB_DEFERRED_FLUSH, which no other barrier
 * waits for. */
void
MALI_PER_ARCH(cmd_query_fence)(struct mali_cmd_buffer *cmd, bool before)
{
   const VkMemoryBarrier2 mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = before ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
                             : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                  VK_PIPELINE_STAGE_2_COPY_BIT,
      .srcAccessMask = before ? VK_ACCESS_2_MEMORY_WRITE_BIT
                              : VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                                   VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .dstStageMask = before ? VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                  VK_PIPELINE_STAGE_2_COPY_BIT
                             : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .dstAccessMask = before ? VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT
                              : VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
   };
   const VkDependencyInfo dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &mb,
   };
   struct mali_cs_deps deps = {0};
   mali_cmd_add_deps(cmd, &dep, &deps);
   if (before) {
      for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++) {
         if (deps.src[i].wait_sb_mask)
            deps.src[i].wait_sb_mask |= MALI_SB_MASK(MALI_SB_DEFERRED_SYNC) |
                                        MALI_SB_MASK(MALI_SB_DEFERRED_FLUSH);
      }
   }
   mali_cmd_emit_barrier(cmd, &deps);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdResetQueryPool)(VkCommandBuffer commandBuffer, VkQueryPool _pool,
                                 uint32_t firstQuery, uint32_t queryCount)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(mali_query_pool, pool, _pool);

   if (!queryCount)
      return;

   /* Fills of the slots, on the compute subqueue, after whatever earlier
    * work wrote them and before whatever uses them next. */
   MALI_PER_ARCH(cmd_query_fence)(cmd, true);
   MALI_PER_ARCH(cmd_fill_addr)(cmd, mali_query_val_va(pool, firstQuery),
                                (uint64_t)queryCount * pool->val_stride, 0);
   if (pool->avail_bo.size)
      MALI_PER_ARCH(cmd_fill_addr)(cmd, mali_query_avail_va(pool, firstQuery),
                                   (uint64_t)queryCount * pool->avail_stride, 0);
   MALI_PER_ARCH(cmd_query_fence)(cmd, false);
}

/* The vertex-side stages, which the vertex/tiler stream carries. */
#define VT_STAGES                                                              \
   (VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT | \
    VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT | \
    VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |                                    \
    VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT |                      \
    VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT |                   \
    VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT |                                  \
    VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT)

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdWriteTimestamp)(VkCommandBuffer commandBuffer,
                                 VkPipelineStageFlagBits pipelineStage, VkQueryPool _pool,
                                 uint32_t query)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(mali_query_pool, pool, _pool);
   const struct mali_csf_device *csf = cmd->dev->csf;
   const VkPipelineStageFlags2 stage = (VkPipelineStageFlags2)pipelineStage;

   const bool top = stage == VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT ||
                    stage == VK_PIPELINE_STAGE_2_NONE;
   enum mali_subqueue sq = MALI_SUBQUEUE_FRAGMENT;
   if (top || (stage && !(stage & ~VT_STAGES)))
      sq = MALI_SUBQUEUE_VERTEX_TILER;
   else if (stage == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
      sq = MALI_SUBQUEUE_COMPUTE;

   struct cs_builder *b = mali_cmd_cs(cmd, sq);
   struct cs_index addr = mali_cs_scratch_reg64(b, 0);
   struct cs_index one = mali_cs_scratch_reg64(b, 2);

   cs_move64_to(b, addr, mali_query_val_va(pool, query));
   cs_store_state(b, addr, 0, MALI_CS_STATE_TIMESTAMP,
                  cs_defer((top ? 0 : csf->sb.all_iters_mask) | MALI_SB_MASK(MALI_SB_DEFERRED_SYNC),
                           MALI_SB_DEFERRED_FLUSH));
   cs_move64_to(b, addr, mali_query_avail_va(pool, query));
   cs_move64_to(b, one, 1);
   cs_sync64_set(b, false, MALI_CS_SYNC_SCOPE_SYSTEM, one, addr,
                 cs_defer(MALI_SB_MASK(MALI_SB_DEFERRED_FLUSH), MALI_SB_DEFERRED_SYNC));
}
