/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * VkEvent, on every arch. An event is one 32-bit word (in a 16-byte slot)
 * of memory the CPU and GPU see coherently without cache maintenance: CSF
 * event memory on v11, CPU-uncached device memory on the job manager
 * (mali_event.c, mali_event_pool_init). Nonzero means set.
 *
 *  - The host resets it with a store (vkResetEvent) and reads it for
 *    vkGetEventStatus. vkSetEvent stores 1 and then tells the kernel, so
 *    that GPU work waiting on the event goes on: SOFT_EVENT_UPDATE on the
 *    job manager, CS_EVENT_SIGNAL on the command-stream frontend
 *    (mali_event.c).
 *  - The device writes it in queue order, after everything recorded before
 *    the command (vkCmdSetEvent2, vkCmdResetEvent2), and waits for it
 *    (vkCmdWaitEvents2): on v11 with command-stream sync instructions
 *    (mali_cmd_event.c), on the job manager (v7, v9) with kbase soft atoms
 *    (EVENT_SET, EVENT_RESET, EVENT_WAIT; mali_jm_event.c). The job
 *    manager's kernel reads and writes only the word's first byte, which is
 *    the whole value in little-endian order.
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
