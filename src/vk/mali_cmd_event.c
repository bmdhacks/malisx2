/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * VkEvent commands on the command-stream frontend (v11): vkCmdSetEvent2,
 * vkCmdResetEvent2, vkCmdWaitEvents2. The Vulkan 1.0 vkCmdSetEvent,
 * vkCmdResetEvent and vkCmdWaitEvents are Mesa's, which call these; an
 * application that enables synchronization2 names the "2" forms itself. The
 * object and the host side are mali_event.c; the job manager's commands are
 * mali_jm_event.c.
 *
 * A set or reset is a barrier from every subqueue into the compute
 * subqueue (mali_cmd_emit_barrier: each source waits for its own
 * scoreboard entries and signals, the compute stream waits for the
 * signals), then a SYNC_SET32 of the word on the compute stream at system
 * scope, as the blob's does for an event the host can see. The event is
 * one word, written by one stream, so a reset recorded after a set always
 * lands after it, which per-subqueue words (panvk's layout) would not
 * give without more waits. The first synchronization scope is all earlier
 * work whatever stage mask the application gives, which is the strictest
 * reading the specification allows. The barrier covers earlier command
 * buffers too: each ring runs its command buffers in order, and a stream
 * ends with all its work done (finish_stream), so a source's signal in
 * this command buffer comes after the earlier ones' work.
 *
 * A set also cleans L2 and the load/store caches on every source, as a
 * barrier into the host's HOST_READ would: an application polls an event
 * and then reads a buffer the GPU wrote (a texture readback), and may not
 * have asked for the clean itself. A reset has no such reader.
 */

#include "mali_cmd_buffer.h"

#include "util/log.h"
#include "vk_log.h"

#include "mali_event.h"
#include "mali_vk.h"

static void
emit_event_write(struct mali_cmd_buffer *cmd, const struct mali_event *ev, bool set)
{
   const VkMemoryBarrier2 barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                      (set ? VK_PIPELINE_STAGE_2_HOST_BIT : 0),
      .dstAccessMask = set ? VK_ACCESS_2_HOST_READ_BIT : 0,
   };
   const VkDependencyInfo info = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &barrier,
   };
   struct mali_cs_deps deps = {0};
   mali_cmd_add_deps(cmd, &info, &deps);
   mali_cmd_emit_barrier(cmd, &deps);

   /* The compute stream puts its barrier waits off until it records work
    * (mali_cmd_cs): this is that work. */
   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_COMPUTE);
   struct cs_index addr = mali_cs_scratch_reg64(b, 0);
   struct cs_index val = mali_cs_scratch_reg32(b, 2);
   cs_move64_to(b, addr, mali_event_va(ev));
   cs_move32_to(b, val, set ? 1 : 0);
   cs_sync32_set(b, false, MALI_CS_SYNC_SCOPE_SYSTEM, val, addr, cs_now());
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdSetEvent2)(VkCommandBuffer commandBuffer, VkEvent _event,
                            const VkDependencyInfo *pDependencyInfo)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(mali_event, ev, _event);

   /* Not valid inside a render pass instance (Vulkan). The dependency
    * info is not needed: the set waits for everything before it, and image
    * layout transitions change nothing in memory on this driver. */
   if (cmd->gfx.render.active) {
      mesa_logw("malisx2: vkCmdSetEvent inside a render pass instance ignored");
      return;
   }
   emit_event_write(cmd, ev, true);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdResetEvent2)(VkCommandBuffer commandBuffer, VkEvent _event,
                              VkPipelineStageFlags2 stageMask)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(mali_event, ev, _event);

   if (cmd->gfx.render.active) {
      mesa_logw("malisx2: vkCmdResetEvent inside a render pass instance ignored");
      return;
   }
   emit_event_write(cmd, ev, false);
}

/*
 * Each event's dependency info is a pipeline barrier, built by the same code
 * (inside a render pass that is the by-region framebuffer dependency a
 * subpass self-dependency asks for). Outside a render pass every stream
 * then waits for the word to be set (SYNC_WAIT32, greater than 0), so an
 * event the host sets after the command buffer was submitted holds the work
 * after the wait until then (vkSetEvent signals the kernel, which has the
 * firmware evaluate the waits again). Every stream waits, whatever the
 * destination stages: the strictest reading. The compute stream's put-off
 * barrier waits come first (mali_cmd_cs), then the fragment stream's
 * put-off signal (mali_cmd_frag_flush_pending): a set recorded earlier in
 * this command buffer waits on the compute stream for that signal, so
 * the signal must not come after a wait on the set.
 *
 * Inside a render pass only the barrier is recorded: an event waited for
 * there must have been set before (Vulkan forbids a host set after the
 * submit for a wait inside a render pass), by the host or earlier in the
 * queue.
 *
 * A wait on an event that nobody sets never ends: the streams stay in their
 * sync wait (kbase has no timeout for it), the submission's fences never
 * signal, and a host wait with a timeout returns VK_TIMEOUT.
 */
static void
emit_event_wait(struct mali_cmd_buffer *cmd, const struct mali_event *ev)
{
   static const enum mali_subqueue order[] = {
      MALI_SUBQUEUE_COMPUTE, MALI_SUBQUEUE_VERTEX_TILER, MALI_SUBQUEUE_FRAGMENT,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(order); i++) {
      struct cs_builder *b = mali_cmd_cs(cmd, order[i]);
      if (order[i] == MALI_SUBQUEUE_COMPUTE)
         mali_cmd_frag_flush_pending(cmd);
      struct cs_index addr = mali_cs_scratch_reg64(b, 0);
      struct cs_index ref = mali_cs_scratch_reg32(b, 2);
      cs_move64_to(b, addr, mali_event_va(ev));
      cs_move32_to(b, ref, 0);
      cs_sync32_wait(b, false, MALI_CS_CONDITION_GREATER, ref, addr);
   }
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdWaitEvents2)(VkCommandBuffer commandBuffer, uint32_t eventCount,
                              const VkEvent *pEvents, const VkDependencyInfo *pDependencyInfos)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);

   for (uint32_t i = 0; i < eventCount; i++) {
      VK_FROM_HANDLE(mali_event, ev, pEvents[i]);
      MALI_PER_ARCH(CmdPipelineBarrier2)(commandBuffer, &pDependencyInfos[i]);
      if (!cmd->gfx.render.active)
         emit_event_wait(cmd, ev);
   }
}
