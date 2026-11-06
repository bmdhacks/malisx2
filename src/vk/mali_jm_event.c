/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * VkEvent commands on the job manager (v7 and v9): vkCmdSetEvent2,
 * vkCmdResetEvent2, vkCmdWaitEvents2. The Vulkan 1.0 vkCmdSetEvent,
 * vkCmdResetEvent and vkCmdWaitEvents are Mesa's, which call these; an
 * application that enables synchronization2 names the "2" forms itself.
 * The object and the host side are mali_event.c.
 *
 * Each command is a batch of its own (mali_jm_cmd_event) that submit turns
 * into one kbase soft atom on the event's word (mali_jm_queue.c,
 * build_event), as the blob does:
 *
 *  - a set or reset (EVENT_SET, EVENT_RESET) runs when every atom
 *    submitted before it has completed, and the kernel writes the word
 *    then. The first synchronization scope is therefore all earlier work,
 *    whatever stage mask the application gives, which is the strictest
 *    reading the specification allows. The kernel cleans the caches when
 *    an atom completes, so what that work wrote is in memory before the
 *    word changes;
 *  - a wait (EVENT_WAIT) completes when the word is set: at once if it is,
 *    else when an EVENT_SET atom on the same word runs or the host calls
 *    vkSetEvent (mali_event.c, SOFT_EVENT_UPDATE). The atoms after it in
 *    the slots its destination stages use depend on it. The work after the
 *    wait is in later atoms, which start with clean caches, so the memory
 *    side of the dependency needs nothing more here; the image layout
 *    transitions that make CRCs stale are handled as a barrier handles
 *    them.
 *
 * A wait whose event is never set is cancelled by the kernel after its
 * soft-job timeout (3 s unless the system changed it); the work behind it
 * fails with it and the device is lost (mali_jm_queue.c).
 */

#include "mali_cmd_state.h"

#include "util/log.h"

#include "mali_event.h"

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdSetEvent2)(VkCommandBuffer commandBuffer, VkEvent _event,
                            const VkDependencyInfo *pDependencyInfo)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(mali_event, ev, _event);

   /* Not valid inside a render pass instance (Vulkan). */
   if (cmd->gfx.render.active) {
      mesa_logw("malisx2: vkCmdSetEvent inside a render pass instance ignored");
      return;
   }
   mali_jm_cmd_dep_crc(cmd, pDependencyInfo);
   mali_jm_cmd_event(cmd, MALI_JM_EVENT_SET, mali_event_va(ev), 0);
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
   mali_jm_cmd_event(cmd, MALI_JM_EVENT_RESET, mali_event_va(ev), 0);
}

/* The destination stages of a dependency info: what a wait on it holds. */
static VkPipelineStageFlags2
dst_stages(const VkDependencyInfo *info)
{
   VkPipelineStageFlags2 dst = 0;
   for (uint32_t i = 0; i < info->memoryBarrierCount; i++)
      dst |= info->pMemoryBarriers[i].dstStageMask;
   for (uint32_t i = 0; i < info->bufferMemoryBarrierCount; i++)
      dst |= info->pBufferMemoryBarriers[i].dstStageMask;
   for (uint32_t i = 0; i < info->imageMemoryBarrierCount; i++)
      dst |= info->pImageMemoryBarriers[i].dstStageMask;
   return dst;
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdWaitEvents2)(VkCommandBuffer commandBuffer, uint32_t eventCount,
                              const VkEvent *pEvents, const VkDependencyInfo *pDependencyInfos)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);

   for (uint32_t i = 0; i < eventCount; i++) {
      VK_FROM_HANDLE(mali_event, ev, pEvents[i]);
      mali_jm_cmd_dep_crc(cmd, &pDependencyInfos[i]);
      if (!mali_jm_cmd_event(cmd, MALI_JM_EVENT_WAIT, mali_event_va(ev),
                             dst_stages(&pDependencyInfos[i])))
         return;
   }
}
