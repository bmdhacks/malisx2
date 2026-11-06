/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * VkEvent, on every arch. An event is one 32-bit word (in a 16-byte slot)
 * of CPU-uncached device memory, the memory type the application sees as
 * HOST_VISIBLE | HOST_COHERENT. Nonzero means set.
 *
 *  - The host resets it with a store (vkResetEvent) and reads it for
 *    vkGetEventStatus. vkSetEvent stores 1; on the job manager it then
 *    tells the kernel (SOFT_EVENT_UPDATE), so that GPU work waiting on the
 *    event goes on (mali_event.c).
 *  - The device writes it in queue order, after everything recorded before
 *    the command (vkCmdSetEvent2, vkCmdResetEvent2): on v11 with a
 *    command-stream sync instruction (mali_cmd_event.c), on the job manager
 *    (v7, v9) with kbase soft atoms (EVENT_SET, EVENT_RESET), which also
 *    wait for it (EVENT_WAIT, vkCmdWaitEvents2; mali_jm_event.c). The job
 *    manager's kernel reads and writes only the word's first byte, which is
 *    the whole value in little-endian order. On v11 vkCmdWaitEvents2 is
 *    the pipeline barrier of its dependency infos (mali_cmd_event.c).
 *
 * VK_EVENT_CREATE_DEVICE_ONLY_BIT needs synchronization2, which every arch
 * offers; the creation flags are not looked at: every event is
 * host-visible, which a device-only event may also be.
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
