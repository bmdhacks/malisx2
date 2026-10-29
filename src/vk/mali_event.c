/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * VkEvent from the host's side: vkCreateEvent, vkDestroyEvent, vkSetEvent,
 * vkResetEvent, vkGetEventStatus. The object, and what the device does to
 * the word, are in mali_event.h; the vkCmd* that write and wait on events
 * are mali_cmd_event.c.
 *
 * Built once with the command-stream frontend, so these are the v11 table's
 * entries (MALI_PER_ARCH, as in mali_cmd_event.c). The job-manager GPUs
 * (v7, v9) have no events: their entry points stay waist stubs.
 *
 * The words come from one device-wide pool (dev->event_pool): ARMSX3
 * creates an event for every texture readback and polls it hot, so an
 * event cannot cost a kbase allocation, and a poll must not cost a system
 * call when the event is set.
 */

/* Built with the v11 files: MALI_PER_ARCH needs PAN_ARCH (mali_cs.h does
 * the same). */
#ifndef PAN_ARCH
#define PAN_ARCH MALI_PAN_ARCH
#endif

#include "mali_event.h"

#include "vk_alloc.h"
#include "vk_log.h"

#include "mali_vk.h"

/* One slab holds this many events; the pool frees a slab that is empty
 * unless it is the last. */
#define MALI_EVENT_SLAB_SIZE (16u << 10)

void
mali_event_pool_init(struct mali_device *dev)
{
   mali_bo_pool_init(&dev->event_pool, dev->kbase, "events",
                     mali_kbase_mem_same_va_policy(MALI_KBASE_FLAGS_DEVICE_UNCACHED),
                     MALI_KBASE_MEM_CLASS_DEVICE_CPU_UNCACHED, MALI_EVENT_SLAB_SIZE,
                     MALI_EVENT_SLOT_SIZE, false);
}

void
mali_event_pool_finish(struct mali_device *dev)
{
   mali_bo_pool_finish(&dev->event_pool);
}

VKAPI_ATTR VkResult VKAPI_CALL
MALI_PER_ARCH(CreateEvent)(VkDevice _device, const VkEventCreateInfo *pCreateInfo,
                 const VkAllocationCallbacks *pAllocator, VkEvent *pEvent)
{
   VK_FROM_HANDLE(mali_device, dev, _device);

   struct mali_event *ev =
      vk_object_zalloc(&dev->vk, pAllocator, sizeof(*ev), VK_OBJECT_TYPE_EVENT);
   if (!ev)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   const enum mali_kbase_result r =
      mali_bo_pool_alloc(&dev->event_pool, MALI_EVENT_SLOT_SIZE, MALI_EVENT_SLOT_SIZE, &ev->mem);
   if (r != MALI_KBASE_SUCCESS) {
      vk_object_free(&dev->vk, pAllocator, ev);
      return vk_error(dev, r == MALI_KBASE_ERROR_OUT_OF_HOST_MEMORY
                              ? VK_ERROR_OUT_OF_HOST_MEMORY
                              : VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }
   /* A slot that held an earlier event is not clean. */
   __atomic_store_n(mali_event_word(ev), 0, __ATOMIC_RELAXED);

   *pEvent = mali_event_to_handle(ev);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(DestroyEvent)(VkDevice _device, VkEvent _event, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   VK_FROM_HANDLE(mali_event, ev, _event);

   if (!ev)
      return;
   mali_bo_pool_free(&dev->event_pool, &ev->mem);
   vk_object_free(&dev->vk, pAllocator, ev);
}

VKAPI_ATTR VkResult VKAPI_CALL
MALI_PER_ARCH(SetEvent)(VkDevice _device, VkEvent _event)
{
   VK_FROM_HANDLE(mali_event, ev, _event);
   __atomic_store_n(mali_event_word(ev), 1, __ATOMIC_RELEASE);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
MALI_PER_ARCH(ResetEvent)(VkDevice _device, VkEvent _event)
{
   VK_FROM_HANDLE(mali_event, ev, _event);
   __atomic_store_n(mali_event_word(ev), 0, __ATOMIC_RELEASE);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
MALI_PER_ARCH(GetEventStatus)(VkDevice _device, VkEvent _event)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   VK_FROM_HANDLE(mali_event, ev, _event);

   if (__atomic_load_n(mali_event_word(ev), __ATOMIC_ACQUIRE))
      return VK_EVENT_SET;

   /* Not set: a poller must see a lost device (the event would never be
    * set). */
   if (vk_device_check_status(&dev->vk) != VK_SUCCESS)
      return VK_ERROR_DEVICE_LOST;
   return __atomic_load_n(mali_event_word(ev), __ATOMIC_ACQUIRE) ? VK_EVENT_SET : VK_EVENT_RESET;
}
