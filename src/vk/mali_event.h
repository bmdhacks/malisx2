/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * VkEvent, on the command-stream frontend (v11). An event is one 32-bit
 * word (in a 16-byte slot) of CPU-uncached device memory, the memory type
 * the application sees as HOST_VISIBLE | HOST_COHERENT. Nonzero means set.
 *
 *  - The host sets and resets it with a store (vkSetEvent, vkResetEvent)
 *    and reads it for vkGetEventStatus.
 *  - The device writes it from the queue, in queue order after everything
 *    recorded before the command: vkCmdSetEvent2 and vkCmdResetEvent2, as
 *    SYNC_SET32 at system scope (mali_cmd_event.c).
 *  - vkCmdWaitEvents2 is a pipeline barrier built from its dependency
 *    infos. There is one queue and the event is set earlier in it (or by
 *    the host before the command buffer is submitted), so the barrier
 *    already orders what the wait would. An event set by the host after
 *    the waiting command buffer was submitted is not supported: nothing
 *    waits for it.
 *
 * The job-manager GPUs (v7, v9) have no events: their entry points are
 * waist stubs.
 *
 * VK_EVENT_CREATE_DEVICE_ONLY_BIT needs synchronization2, which the driver
 * does not expose, so the creation flags are not looked at: every event is
 * host-visible.
 */

#ifndef MALI_EVENT_H
#define MALI_EVENT_H

#include <stdint.h>

#include "vk_object.h"

#include "mali_bo_pool.h"

#define MALI_EVENT_SLOT_SIZE 16u

struct mali_event {
   struct vk_object_base base;
   struct mali_bo_ref mem;   /* MALI_EVENT_SLOT_SIZE bytes in the device's event pool */
};

VK_DEFINE_NONDISP_HANDLE_CASTS(mali_event, base, VkEvent, VK_OBJECT_TYPE_EVENT)

static inline volatile uint32_t *
mali_event_word(const struct mali_event *ev)
{
   return ev->mem.cpu;
}

static inline uint64_t
mali_event_va(const struct mali_event *ev)
{
   return ev->mem.gpu_va;
}

#endif
