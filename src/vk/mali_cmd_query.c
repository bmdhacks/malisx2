/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Query commands on the command-stream frontend (v11): vkCmdResetQueryPool,
 * vkCmdWriteTimestamp, and the occlusion queries' vkCmdBeginQuery and
 * vkCmdEndQuery. (The job manager's are in mali_jm_query.c; the
 * pool object and the host reads are in mali_query.c.)
 *
 * A timestamp is two deferred instructions on one subqueue's stream:
 * STORE_STATE{Timestamp} into the query's value word, then SYNC_SET64 of
 * the availability word to 1 once the store is done (mali_query.h for the
 * layout). The
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

/* ---------------------------------------------------------------------- */
/* Occlusion queries                                                       */

/*
 * The hardware counts. RUN_IDVS takes the counter's address in the OQ
 * register and the mode in DCD Flags 0 (MALI_GFX_DIRTY_OQ, mali_cmd_draw.c);
 * the fragment job adds the samples that passed to the counter, tile by
 * tile, so the count is only final when the fragment job of every pass that
 * drew into the query has finished. The counters are in ordinary GPU-cached
 * memory (the adds are atomic in L2), the availability words in a buffer of
 * their own (mali_query.h).
 *
 *  - Begin: the fragment stream stores a zero into the counter and waits for
 *    the store. The pass's RUN_FRAGMENT comes later in that stream, so it
 *    adds to a counter that is zero.
 *  - End: the availability word is set on the fragment stream by a deferred
 *    SYNC_SET64, after a deferred cache clean, after the fragment jobs
 *    issued so far (every iterator scoreboard entry). The clean puts the
 *    counter in memory before the word says it is final. A query that ends
 *    inside a render pass has no fragment jobs yet: the signal is emitted at
 *    the end of the pass (cmd_oq_pass_end), behind them.
 */

/* The availability words in addrs[0..n), set once the fragment work issued
 * so far has finished and its writes are in memory. */
static void
emit_oq_signals(struct mali_cmd_buffer *cmd, const uint64_t *addrs, unsigned n)
{
   const struct mali_csf_device *csf = cmd->dev->csf;
   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_FRAGMENT);
   struct cs_index flush_id = mali_cs_scratch_reg32(b, 0);
   struct cs_index addr = mali_cs_scratch_reg64(b, 2);
   struct cs_index one = mali_cs_scratch_reg64(b, 4);

   cs_move32_to(b, flush_id, 0);
   cs_flush_caches(b, MALI_CS_FLUSH_MODE_CLEAN, MALI_CS_FLUSH_MODE_CLEAN,
                   MALI_CS_OTHER_FLUSH_MODE_NONE, flush_id,
                   cs_defer(csf->sb.all_iters_mask, MALI_SB_DEFERRED_FLUSH));
   cs_move64_to(b, one, 1);
   for (unsigned i = 0; i < n; i++) {
      cs_move64_to(b, addr, addrs[i]);
      cs_sync64_set(b, false, MALI_CS_SYNC_SCOPE_SYSTEM, one, addr,
                    cs_defer(MALI_SB_MASK(MALI_SB_DEFERRED_FLUSH), MALI_SB_DEFERRED_SYNC));
   }
}

void
MALI_PER_ARCH(cmd_oq_pass_end)(struct mali_cmd_buffer *cmd)
{
   const unsigned n = util_dynarray_num_elements(&cmd->oq_done, uint64_t);
   if (likely(!n))
      return;
   emit_oq_signals(cmd, util_dynarray_begin(&cmd->oq_done), n);
   util_dynarray_clear(&cmd->oq_done);
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

   const uint64_t counter = mali_query_val_va(pool, query);
   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_FRAGMENT);
   struct cs_index addr = mali_cs_scratch_reg64(b, 0);
   struct cs_index zero = mali_cs_scratch_reg64(b, 2);
   cs_move64_to(b, addr, counter);
   cs_move64_to(b, zero, 0);
   cs_store64(b, zero, addr, 0);
   cs_flush_stores(b);

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
      uint64_t *slot = util_dynarray_grow(&cmd->oq_done, uint64_t, 1);
      if (slot)
         *slot = avail;
      else
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }
   emit_oq_signals(cmd, &avail, 1);
}
