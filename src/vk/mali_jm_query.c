/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Timestamp queries on the job manager (v7 and v9): vkCreateQueryPool,
 * vkDestroyQueryPool, vkCmdResetQueryPool, vkCmdWriteTimestamp,
 * vkGetQueryPoolResults. ARMSX2 turns GPU timing on for any device
 * whose name is not "Mali-G615", so the G57 and G52 builds need these;
 * the G615 build does not (they stay waist stubs there,
 * mali_v11_device_entrypoints never gets them -- see the file comment
 * in mali_jm.h).
 *
 * TIMESTAMP is the only query type: ARMSX2's other query use (OSD GPU
 * stats, VK_QUERY_TYPE_PIPELINE_STATISTICS) is gated on the
 * pipelineStatisticsQuery feature, which mali_physical_device.c
 * reports false (the counters it would need are deferred), so
 * ARMSX2 never creates that pool; occlusion queries are not in
 * ARMSX2's waist either (it only uses the timestamp and
 * pipeline-statistics query types). Implementing only TIMESTAMP here
 * matches the driver's ARMSX2-only scope.
 *
 * Each query is 16 bytes of host-visible, uncached device memory: the
 * value System Timestamp writes at +0, an Immediate 64 availability
 * flag at +8 (our own layout; nothing reads it but our own
 * vkGetQueryPoolResults). Both words are Write Value jobs with the
 * header Barrier bit (the blob's own query and timestamp jobs are
 * the same two types, same bit). mali_jm_measure_begin/end
 * (timing_jm.c) use the identical pair for MALISX2_MEASURE's timing
 * regions; this file does not share code with them beyond
 * mali_jm_cmd_write_value, because a query's two jobs are not a
 * "region" (no CSV row, no handle to end later from a different
 * call).
 */

#include "mali_cmd_state.h"

#include <string.h>

#include "util/os_time.h"
#include "vk_alloc.h"
#include "vk_log.h"
#include "vk_query_pool.h"

struct mali_query_pool {
   struct vk_query_pool vk;
   struct mali_kbase_bo bo; /* vk.query_count * MALI_QUERY_SLOT_SIZE bytes,
                              * CPU+GPU read/write, uncached */
};

VK_DEFINE_NONDISP_HANDLE_CASTS(mali_query_pool, vk.base, VkQueryPool, VK_OBJECT_TYPE_QUERY_POOL)

#define MALI_QUERY_SLOT_SIZE 16u /* value (+0), availability (+8) */

VKAPI_ATTR VkResult VKAPI_CALL
MALI_PER_ARCH(CreateQueryPool)(VkDevice _device, const VkQueryPoolCreateInfo *pCreateInfo,
                               const VkAllocationCallbacks *pAllocator, VkQueryPool *pQueryPool)
{
   VK_FROM_HANDLE(mali_device, dev, _device);

   if (pCreateInfo->queryType != VK_QUERY_TYPE_TIMESTAMP)
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "only timestamp query pools are supported");

   struct mali_query_pool *pool =
      vk_query_pool_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*pool));
   if (!pool)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   const struct mali_kbase_alloc_info ai = {
      .size = (uint64_t)pCreateInfo->queryCount * MALI_QUERY_SLOT_SIZE,
      .flags = mali_kbase_mem_same_va_policy(MALI_KBASE_FLAGS_DEVICE_UNCACHED),
      .mem_class = MALI_KBASE_MEM_CLASS_DESCRIPTOR_POOL,
   };
   if (mali_kbase_alloc(dev->kbase, &ai, &pool->bo) != MALI_KBASE_SUCCESS) {
      vk_query_pool_destroy(&dev->vk, pAllocator, &pool->vk);
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }
   memset(pool->bo.cpu, 0, ai.size);

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
   vk_query_pool_destroy(&dev->vk, pAllocator, &pool->vk);
}

static inline uint64_t
query_addr(const struct mali_query_pool *pool, uint32_t query)
{
   return pool->bo.gpu_va + (uint64_t)query * MALI_QUERY_SLOT_SIZE;
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdResetQueryPool)(VkCommandBuffer commandBuffer, VkQueryPool _pool,
                                 uint32_t firstQuery, uint32_t queryCount)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(mali_query_pool, pool, _pool);

   /* Not valid inside a render pass instance (Vulkan). */
   struct mali_jm_chain *c = mali_jm_cmd_vtc(cmd);
   if (!c)
      return;
   for (uint32_t i = 0; i < queryCount; i++) {
      const uint64_t addr = query_addr(pool, firstQuery + i);
      if (!mali_jm_cmd_write_value(cmd, c, MALI_WRITE_VALUE_TYPE_ZERO, addr, 0, true) ||
          !mali_jm_cmd_write_value(cmd, c, MALI_WRITE_VALUE_TYPE_ZERO, addr + 8, 0, true))
         return;
   }
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
   const uint64_t addr = query_addr(pool, query);
   if (!mali_jm_cmd_write_value(cmd, c, MALI_WRITE_VALUE_TYPE_SYSTEM_TIMESTAMP, addr, 0, true))
      return;
   mali_jm_cmd_write_value(cmd, c, MALI_WRITE_VALUE_TYPE_IMMEDIATE_64, addr + 8, 1, true);
}

VKAPI_ATTR VkResult VKAPI_CALL
MALI_PER_ARCH(GetQueryPoolResults)(VkDevice _device, VkQueryPool _pool, uint32_t firstQuery,
                                   uint32_t queryCount, size_t dataSize, void *pData,
                                   VkDeviceSize stride, VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   VK_FROM_HANDLE(mali_query_pool, pool, _pool);
   const uint64_t *slots = pool->bo.cpu;
   VkResult result = VK_SUCCESS;

   for (uint32_t i = 0; i < queryCount; i++) {
      const uint32_t q = firstQuery + i;
      uint8_t *out = (uint8_t *)pData + i * stride;
      uint64_t avail;

      for (;;) {
         avail = slots[(uint64_t)q * 2 + 1];
         if (avail || !(flags & VK_QUERY_RESULT_WAIT_BIT) || !dev->jm)
            break;
         pthread_mutex_lock(&dev->lock);
         mali_jm_wait_locked(dev, os_time_get_nano() + 5 * 1000 * 1000);
         pthread_mutex_unlock(&dev->lock);
         if (dev->lost)
            return vk_error(dev, VK_ERROR_DEVICE_LOST);
      }

      if (avail || (flags & VK_QUERY_RESULT_PARTIAL_BIT)) {
         const uint64_t value = slots[(uint64_t)q * 2];
         if (flags & VK_QUERY_RESULT_64_BIT)
            *(uint64_t *)out = value;
         else
            *(uint32_t *)out = (uint32_t)value;
      } else {
         result = VK_NOT_READY;
      }

      if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) {
         void *out_avail = out + (flags & VK_QUERY_RESULT_64_BIT ? 8 : 4);
         if (flags & VK_QUERY_RESULT_64_BIT)
            *(uint64_t *)out_avail = avail ? 1 : 0;
         else
            *(uint32_t *)out_avail = avail ? 1 : 0;
      }
   }
   return result;
}
