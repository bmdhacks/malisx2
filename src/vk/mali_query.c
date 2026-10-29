/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Query pools, the host side: vkCreateQueryPool, vkDestroyQueryPool and
 * vkGetQueryPoolResults, for timestamp and occlusion queries. Built once
 * per back half (v11 and the job manager's v7 and v9), like mali_sync.c:
 * what is recorded into command buffers differs (mali_cmd_query.c on the
 * command-stream frontend, mali_jm_query.c on the job manager), the pool
 * and its host reads do not. The layout is in mali_query.h.
 *
 * Pipeline statistics queries are not offered (pipelineStatisticsQuery is
 * false), nor are the other query types.
 */

#if defined(PAN_ARCH) && PAN_ARCH < 10
#include "mali_jm.h"
#else
#include "mali_queue.h"
#endif

#include <string.h>

#include "util/os_time.h"
#include "vk_alloc.h"
#include "vk_log.h"

#include "mali_arch.h"
#include "mali_query.h"

/* How long a host wait for one query lasts before the device counts as lost:
 * the job manager's kernel hard-stops a job after a second, so a GPU that
 * has not finished in this long is not going to. */
#define QUERY_WAIT_NS (10ll * 1000 * 1000 * 1000)
#define QUERY_WAIT_SLICE_NS (5ll * 1000 * 1000)

static VkResult
query_pool_alloc(struct mali_device *dev, struct mali_query_pool *pool, VkQueryType type)
{
   const uint32_t count = pool->vk.query_count;

#if PAN_ARCH < 10
   /* One buffer, value and availability in 16-byte slots. */
   const struct mali_kbase_alloc_info ai = {
      .size = (uint64_t)count * 16,
      .flags = mali_kbase_mem_same_va_policy(MALI_KBASE_FLAGS_DEVICE_UNCACHED),
      .mem_class = MALI_KBASE_MEM_CLASS_DESCRIPTOR_POOL,
   };
   if (mali_kbase_alloc(dev->kbase, &ai, &pool->bo) != MALI_KBASE_SUCCESS)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   memset(pool->bo.cpu, 0, ai.size);
   pool->val_stride = pool->avail_stride = 16;
   pool->val_va = pool->bo.gpu_va;
   pool->val_cpu = pool->bo.cpu;
   pool->avail_va = pool->bo.gpu_va + 8;
   pool->avail_cpu = (volatile uint64_t *)((uint8_t *)pool->bo.cpu + 8);
#else
   if (type == VK_QUERY_TYPE_TIMESTAMP) {
      /* Ordinary device memory, not CSF event memory (mali_query.h). A
       * 64-bit sync object is 16 bytes at a 16-byte aligned address (the
       * stream faults with CS_INVALID_INSTRUCTION on any other), so the
       * slot is 32 bytes: the value, then the availability object. */
      const struct mali_kbase_alloc_info ai = {
         .size = (uint64_t)count * 32,
         .flags = mali_kbase_mem_same_va_policy(MALI_KBASE_FLAGS_DEVICE_UNCACHED),
         .mem_class = MALI_KBASE_MEM_CLASS_DESCRIPTOR_POOL,
      };
      if (mali_kbase_alloc(dev->kbase, &ai, &pool->bo) != MALI_KBASE_SUCCESS)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      memset(pool->bo.cpu, 0, ai.size);
      pool->val_stride = pool->avail_stride = 32;
      pool->val_va = pool->bo.gpu_va;
      pool->val_cpu = pool->bo.cpu;
      pool->avail_va = pool->bo.gpu_va + 16;
      pool->avail_cpu = (volatile uint64_t *)((uint8_t *)pool->bo.cpu + 16);
   } else {
      const struct mali_kbase_alloc_info vi = {
         .size = (uint64_t)count * 8,
         .flags = mali_kbase_mem_same_va_policy(MALI_KBASE_FLAGS_DEVICE_UNCACHED),
         .mem_class = MALI_KBASE_MEM_CLASS_DESCRIPTOR_POOL,
      };
      /* The availability words are 64-bit sync objects: 16 bytes each, as
       * above. */
      const struct mali_kbase_alloc_info ai = {
         .size = (uint64_t)count * 16,
         .flags = mali_kbase_mem_same_va_policy(MALI_KBASE_FLAGS_DEVICE_UNCACHED),
         .mem_class = MALI_KBASE_MEM_CLASS_DESCRIPTOR_POOL,
      };
      if (mali_kbase_alloc(dev->kbase, &vi, &pool->bo) != MALI_KBASE_SUCCESS)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      if (mali_kbase_alloc(dev->kbase, &ai, &pool->avail_bo) != MALI_KBASE_SUCCESS) {
         mali_kbase_free(dev->kbase, &pool->bo);
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      }
      memset(pool->bo.cpu, 0, vi.size);
      memset(pool->avail_bo.cpu, 0, ai.size);
      pool->val_stride = 8;
      pool->avail_stride = 16;
      pool->val_va = pool->bo.gpu_va;
      pool->val_cpu = pool->bo.cpu;
      pool->avail_va = pool->avail_bo.gpu_va;
      pool->avail_cpu = pool->avail_bo.cpu;
   }
#endif
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
MALI_PER_ARCH(CreateQueryPool)(VkDevice _device, const VkQueryPoolCreateInfo *pCreateInfo,
                               const VkAllocationCallbacks *pAllocator, VkQueryPool *pQueryPool)
{
   VK_FROM_HANDLE(mali_device, dev, _device);

   if (pCreateInfo->queryType != VK_QUERY_TYPE_TIMESTAMP &&
       pCreateInfo->queryType != VK_QUERY_TYPE_OCCLUSION)
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "only timestamp and occlusion query pools are supported");
#if PAN_ARCH == 7
   /* Bifrost's counter is one per shader core and its Draw descriptor is
    * laid out differently; not done. */
   if (pCreateInfo->queryType == VK_QUERY_TYPE_OCCLUSION)
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "occlusion queries are not supported on this GPU");
#endif

   struct mali_query_pool *pool =
      vk_query_pool_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*pool));
   if (!pool)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult r = query_pool_alloc(dev, pool, pCreateInfo->queryType);
   if (r != VK_SUCCESS) {
      vk_query_pool_destroy(&dev->vk, pAllocator, &pool->vk);
      return vk_error(dev, r);
   }

   *pQueryPool = mali_query_pool_to_handle(pool);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(DestroyQueryPool)(VkDevice _device, VkQueryPool _pool,
                                const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   VK_FROM_HANDLE(mali_query_pool, pool, _pool);
   if (!pool)
      return;
   mali_kbase_free(dev->kbase, &pool->bo);
   if (pool->avail_bo.size)
      mali_kbase_free(dev->kbase, &pool->avail_bo);
   vk_query_pool_destroy(&dev->vk, pAllocator, &pool->vk);
}

/* Waits for the availability word to become non-zero. With dev->lock not
 * held. VK_ERROR_DEVICE_LOST if the device is lost or the GPU does not
 * finish in QUERY_WAIT_NS. */
static VkResult
wait_available(struct mali_device *dev, volatile uint64_t *avail)
{
   const int64_t deadline = os_time_get_nano() + QUERY_WAIT_NS;

   pthread_mutex_lock(&dev->lock);
   for (;;) {
      if (*avail) {
         pthread_mutex_unlock(&dev->lock);
         return VK_SUCCESS;
      }
      if (dev->lost) {
         pthread_mutex_unlock(&dev->lock);
         return VK_ERROR_DEVICE_LOST;
      }
      /* A fault that has not reached the log yet. */
      const char *lost_msg = MALI_PER_ARCH(queue_wait)(dev);
      if (lost_msg) {
         pthread_mutex_unlock(&dev->lock);
         mali_device_set_lost(dev, "%s", lost_msg);
         return VK_ERROR_DEVICE_LOST;
      }
      const int64_t now = os_time_get_nano();
      if (now >= deadline) {
         pthread_mutex_unlock(&dev->lock);
         mali_device_set_lost(dev, "a query did not become available in %lld s",
                              (long long)(QUERY_WAIT_NS / 1000000000ll));
         return VK_ERROR_DEVICE_LOST;
      }
      MALI_PER_ARCH(queue_sleep)(dev, MIN2(deadline, now + QUERY_WAIT_SLICE_NS));
   }
}

static inline void
put_result(void *out, unsigned index, bool wide, uint64_t v)
{
   if (wide)
      ((uint64_t *)out)[index] = v;
   else
      ((uint32_t *)out)[index] = (uint32_t)MIN2(v, (uint64_t)UINT32_MAX);
}

VKAPI_ATTR VkResult VKAPI_CALL
MALI_PER_ARCH(GetQueryPoolResults)(VkDevice _device, VkQueryPool _pool, uint32_t firstQuery,
                                   uint32_t queryCount, size_t dataSize, void *pData,
                                   VkDeviceSize stride, VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   VK_FROM_HANDLE(mali_query_pool, pool, _pool);
   const bool wide = flags & VK_QUERY_RESULT_64_BIT;
   VkResult result = VK_SUCCESS;

   if (vk_device_is_lost(&dev->vk))
      return VK_ERROR_DEVICE_LOST;

   for (uint32_t i = 0; i < queryCount; i++) {
      const uint32_t q = firstQuery + i;
      volatile uint64_t *avail_word = mali_query_avail_cpu(pool, q);
      void *out = (uint8_t *)pData + i * stride;

      bool avail = *avail_word != 0;
      if (!avail && (flags & VK_QUERY_RESULT_WAIT_BIT)) {
         VkResult r = wait_available(dev, avail_word);
         if (r != VK_SUCCESS)
            return r;
         avail = true;
      }

      /* Availability first, then the value it vouches for. */
      if (avail)
         __atomic_thread_fence(__ATOMIC_ACQUIRE);

      if (avail || (flags & VK_QUERY_RESULT_PARTIAL_BIT))
         put_result(out, 0, wide, mali_query_val_cpu(pool, q));
      else
         result = VK_NOT_READY;

      if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT)
         put_result(out, 1, wide, avail ? 1 : 0);
   }
   return result;
}
