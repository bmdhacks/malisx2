/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * VkDeviceMemory and VkBuffer.
 */

#ifndef MALI_MEMORY_H
#define MALI_MEMORY_H

#include <stdint.h>

#include "vk_buffer.h"
#include "vk_device_memory.h"

#include "kbase/kbase.h"

struct mali_device;
struct mali_physical_device;

/* Memory type indices. */
enum mali_memory_type {
   MALI_MEMORY_TYPE_UNCACHED = 0,   /* DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT */
   MALI_MEMORY_TYPE_CACHED = 1,     /* ... | HOST_CACHED (+ HOST_COHERENT on ACE) */
   MALI_MEMORY_TYPE_TRANSIENT = 2,  /* DEVICE_LOCAL | LAZILY_ALLOCATED, GPU only */
   MALI_MEMORY_TYPE_COUNT = 3,
};

/* memoryTypeBits the blob reports: buffers and ordinary images may use
 * types 0 and 1; transient attachments only the lazily allocated type. */
#define MALI_MEMORY_TYPES_HOST_VISIBLE \
   ((1u << MALI_MEMORY_TYPE_UNCACHED) | (1u << MALI_MEMORY_TYPE_CACHED))
#define MALI_MEMORY_TYPES_TRANSIENT (1u << MALI_MEMORY_TYPE_TRANSIENT)

/* Buffer offsets and sizes the GPU needs no more than 64-byte aligned
 * (the blob reports 64). */
#define MALI_BUFFER_ALIGNMENT 64

struct mali_device_memory {
   struct vk_device_memory vk;

   /* The kbase region: one kernel allocation (or import) per
    * VkDeviceMemory, no sub-allocation (design note). */
   struct mali_kbase_bo bo;

   /* The memory type's property flags. */
   VkMemoryPropertyFlags props;

   /* An imported dma-buf: the fd we own from a successful import (-1
    * otherwise). kbase refuses CPU access through its own mapping of an
    * import, so vkMapMemory maps this fd instead. */
   int dmabuf_fd;
   void *dmabuf_map;          /* mapping of dmabuf_fd, bo.size bytes */
   /* A scanout buffer (VK_KHR_display): allocated as a GBM buffer object on
    * the display's DRM node and imported here; dmabuf_fd is ours to close. */
   bool scanout;

   /* Where the application's CPU view currently starts (vkMapMemory), NULL
    * when not mapped. */
   void *map;

   /* The range [lo, hi) ever mapped, invalidated before the memory is
    * freed when the type is not host-coherent. */
   uint64_t mapped_lo, mapped_hi;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(mali_device_memory, vk.base, VkDeviceMemory,
                               VK_OBJECT_TYPE_DEVICE_MEMORY)

/* GPU address of offset in the memory. */
static inline uint64_t
mali_device_memory_gpu_va(const struct mali_device_memory *mem, uint64_t offset)
{
   return mem->bo.gpu_va + offset;
}

/*
 * The kbase flags and memory class behind a memory type. Exposed for the
 * tests.
 */
void mali_memory_type_kbase_flags(const struct mali_physical_device *pdev,
                                  uint32_t type_index, uint64_t *flags,
                                  enum mali_kbase_mem_class *mem_class);

struct mali_buffer {
   struct vk_buffer vk;
   /* The bound memory, or NULL before vkBindBufferMemory. The GPU address
    * is vk.device_address (memory base + offset), as in the blob. */
   struct mali_device_memory *mem;
   uint64_t mem_offset;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(mali_buffer, vk.base, VkBuffer, VK_OBJECT_TYPE_BUFFER)

/* GPU address of offset in a bound buffer, 0 when unbound. */
static inline uint64_t
mali_buffer_gpu_va(const struct mali_buffer *buf, uint64_t offset)
{
   return buf->vk.device_address ? buf->vk.device_address + offset : 0;
}

#endif
