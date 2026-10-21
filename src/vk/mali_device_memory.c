/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * VkDeviceMemory over kbase. One kernel allocation per VkDeviceMemory, with
 * the blob's flags and memory class for each memory type. The host-visible
 * types are SAME_VA, so their CPU mapping exists from allocation on and
 * vkMapMemory makes no kernel call. Cache maintenance only for the
 * CPU-cached type when the GPU is not ACE-coherent. dma-buf imports are
 * GPU-mapped through kbase and CPU-mapped through the dma-buf fd.
 */

#include "mali_vk.h"
#include "mali_image.h"
#include "mali_memory.h"

#include <errno.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <linux/dma-buf.h>

#include "util/log.h"
#include "util/u_atomic.h"
#include "util/u_math.h"
#include "vk_alloc.h"
#include "vk_util.h"

void
mali_memory_type_kbase_flags(const struct mali_physical_device *pdev,
                             uint32_t type_index, uint64_t *flags,
                             enum mali_kbase_mem_class *mem_class)
{
   switch (type_index) {
   case MALI_MEMORY_TYPE_UNCACHED:
      *flags = mali_kbase_mem_same_va_policy(MALI_KBASE_FLAGS_DEVICE_UNCACHED);
      *mem_class = MALI_KBASE_MEM_CLASS_DEVICE_CPU_UNCACHED;
      break;
   case MALI_MEMORY_TYPE_CACHED:
      /* Kernel coherency mode 1 is full ACE: the blob then asks for
       * system coherency and reports the type as host-coherent. */
      *flags = mali_kbase_mem_same_va_policy(pdev->props.coherency_mode == 1 ?
                                                MALI_KBASE_FLAGS_DEVICE_CACHED_ACE :
                                                MALI_KBASE_FLAGS_DEVICE_CACHED);
      *mem_class = MALI_KBASE_MEM_CLASS_DEVICE_CPU_CACHED;
      break;
   default:
      /* The blob's GPU-only pool for lazily allocated memory: GPU read and
       * write, no CPU access, fully committed. SAME_VA as well: the r44p1
       * kernel adds it to every allocation of a 64-bit client that is
       * neither executable nor FIXED/FIXABLE (KCTX_FORCE_SAME_VA, seen on
       * the device), so asking for it says what happens; the region's CPU
       * mapping is PROT_NONE and only reserves the address. */
      *flags = mali_kbase_mem_same_va_policy(MALI_KBASE_FLAGS_DEVICE_TRANSIENT);
      *mem_class = MALI_KBASE_MEM_CLASS_DEVICE_TRANSIENT;
      break;
   }
}

/* Allocation failures: the blob turns every failed backing allocation into
 * "device out of memory"; only a failed host allocation is host
 * memory. */
static VkResult
alloc_result(enum mali_kbase_result r)
{
   return r == MALI_KBASE_ERROR_OUT_OF_HOST_MEMORY ? VK_ERROR_OUT_OF_HOST_MEMORY
                                                   : VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

/* DMA_BUF_IOCTL_SYNC around CPU access to an imported dma-buf. A file that
 * is not a dma-buf (the host tests use a memfd) answers ENOTTY; there is
 * nothing to synchronize then. */
static void
dmabuf_sync(struct mali_device_memory *mem, uint64_t flags)
{
   struct dma_buf_sync s = {.flags = flags | DMA_BUF_SYNC_RW};
   int ret;
   do {
      ret = ioctl(mem->dmabuf_fd, DMA_BUF_IOCTL_SYNC, &s);
   } while (ret == -1 && (errno == EINTR || errno == EAGAIN));
   if (ret == -1 && errno != ENOTTY)
      mesa_logw("malisx2: DMA_BUF_IOCTL_SYNC 0x%llx on fd %d failed: %s",
                (unsigned long long)s.flags, mem->dmabuf_fd, strerror(errno));
}

static VkResult
import_dmabuf(struct mali_device *dev, struct mali_device_memory *mem, int fd)
{
   /*
    * As the blob: MEM_IMPORT type UMM with the import flags (no memory
    * group: the kernel asks the memory group manager for imports), then a
    * sticky map so the pages are on the GPU from now on.
    */
   enum mali_kbase_result r =
      mali_kbase_import_dmabuf(dev->kbase, fd, MALI_KBASE_FLAGS_IMPORT, 0, true, &mem->bo);
   if (r != MALI_KBASE_SUCCESS)
      return vk_errorf(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "cannot import dma-buf fd %d: %s", fd, mali_kbase_result_str(r));

   if (mem->vk.size > mem->bo.size) {
      mali_kbase_free(dev->kbase, &mem->bo);
      return vk_errorf(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "dma-buf fd %d has %llu bytes, %llu requested", fd,
                       (unsigned long long)mem->bo.size,
                       (unsigned long long)mem->vk.size);
   }

   /* A successful import owns the fd (Vulkan spec); we keep it for CPU
    * mapping and close it when the memory is freed. */
   mem->dmabuf_fd = fd;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_AllocateMemory(VkDevice _device, const VkMemoryAllocateInfo *pAllocateInfo,
                    const VkAllocationCallbacks *pAllocator, VkDeviceMemory *pMem)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   struct mali_physical_device *pdev = mali_device_physical(dev);
   const uint32_t type = pAllocateInfo->memoryTypeIndex;

   if (type >= pdev->memory.memoryTypeCount)
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "memory type %u does not exist", type);

   const VkImportMemoryFdInfoKHR *fd_info =
      vk_find_struct_const(pAllocateInfo->pNext, IMPORT_MEMORY_FD_INFO_KHR);
   if (fd_info && !fd_info->handleType)
      fd_info = NULL;
   if (fd_info && fd_info->handleType != VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT)
      return vk_errorf(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "import of handle type 0x%x (only dma-buf is supported)",
                       fd_info->handleType);

   /*
    * Seam for the Android WSI: AHB import and export go through gralloc
    * and the same kbase import as dma-bufs. Refused here, before the
    * runtime's memory object (which would take an AHB reference).
    */
   if (__vk_find_struct((void *)pAllocateInfo->pNext,
                        VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID))
      return vk_errorf(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "AHardwareBuffer import is not implemented");
   const VkExportMemoryAllocateInfo *export_info =
      vk_find_struct_const(pAllocateInfo->pNext, EXPORT_MEMORY_ALLOCATE_INFO);
   const bool export_dmabuf =
      export_info && (export_info->handleTypes & VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
   if (export_info && export_info->handleTypes && !export_dmabuf)
      return vk_errorf(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "memory export (handle types 0x%x): only dma-buf is supported",
                       export_info->handleTypes);
   /* An exportable allocation is a swapchain image for VK_KHR_display: a
    * scanout buffer allocated by the display's DRM node (GBM) and imported
    * into kbase, with the dma-buf handed to the client. */
   const VkMemoryDedicatedAllocateInfo *dedicated_info =
      vk_find_struct_const(pAllocateInfo->pNext, MEMORY_DEDICATED_ALLOCATE_INFO);
   if (export_dmabuf && (!dedicated_info || dedicated_info->image == VK_NULL_HANDLE))
      return vk_errorf(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "an exportable allocation must be a dedicated image allocation");

   if (!fd_info && pAllocateInfo->allocationSize > pdev->vk.properties.maxMemoryAllocationSize)
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "allocation of %llu bytes is above maxMemoryAllocationSize",
                       (unsigned long long)pAllocateInfo->allocationSize);

   struct mali_device_memory *mem =
      vk_device_memory_create(&dev->vk, pAllocateInfo, pAllocator, sizeof(*mem));
   if (!mem)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   mem->dmabuf_fd = -1;
   mem->props = pdev->memory.memoryTypes[type].propertyFlags;

   VkResult result;
   if (fd_info) {
      result = import_dmabuf(dev, mem, fd_info->fd);
   } else if (export_dmabuf) {
#ifndef __ANDROID__
      struct mali_image *image =
         mali_image_from_handle(dedicated_info->image);
      const unsigned width = image->vk.extent.width;
      const unsigned height = image->vk.extent.height;
      uint32_t stride = 0;
      uint64_t size = 0;
      int fd = -1;
      result = mali_wsi_alloc_scanout(pdev, image->vk.format, width, height, &stride, &size, &fd);
      if (result != VK_SUCCESS)
         goto out;
      /* The image was laid out when it was created, and the display and
       * the GPU have to agree on the row pitch. */
      if (stride != image->planes[0].layout.slices[0].row_stride || size < mem->vk.size) {
         mesa_loge("malisx2: scanout buffer %ux%u: stride %u (image %u) or size %llu "
                   "(image %llu) disagrees with the image layout",
                   width, height, stride, image->planes[0].layout.slices[0].row_stride,
                   (unsigned long long)size, (unsigned long long)mem->vk.size);
         close(fd);
         result = vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         goto out;
      }
      const enum mali_kbase_result r =
         mali_kbase_import_dmabuf(dev->kbase, fd, MALI_KBASE_FLAGS_IMPORT, 0, true, &mem->bo);
      if (r != MALI_KBASE_SUCCESS) {
         close(fd);
         result = vk_errorf(dev, alloc_result(r), "import of the scanout dma-buf: %s",
                            mali_kbase_result_str(r));
         goto out;
      }
      mem->dmabuf_fd = fd;
      mem->scanout = true;
#else
      result = vk_errorf(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                         "memory export is not supported on Android");
      goto out;
#endif
   } else {
      struct mali_kbase_alloc_info ai = {.size = mem->vk.size};
      mali_memory_type_kbase_flags(pdev, type, &ai.flags, &ai.mem_class);
      enum mali_kbase_result r = mali_kbase_alloc(dev->kbase, &ai, &mem->bo);
      result = r == MALI_KBASE_SUCCESS ?
                  VK_SUCCESS :
                  vk_errorf(dev, alloc_result(r), "allocation of %llu bytes (type %u): %s",
                            (unsigned long long)mem->vk.size, type,
                            mali_kbase_result_str(r));
   }
out:
   if (result != VK_SUCCESS) {
      vk_device_memory_destroy(&dev->vk, pAllocator, &mem->vk);
      return result;
   }

   /* VK_EXT_memory_budget's heapUsage: the kbase region's real size (page
    * rounded), not the requested size. */
   p_atomic_add(&pdev->heap_used, (int64_t)mem->bo.size);

   *pMem = mali_device_memory_to_handle(mem);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
mali_FreeMemory(VkDevice _device, VkDeviceMemory _mem,
                const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   VK_FROM_HANDLE(mali_device_memory, mem, _mem);

   if (!mem)
      return;

   p_atomic_add(&mali_device_physical(dev)->heap_used, -(int64_t)mem->bo.size);

   if (mem->dmabuf_fd >= 0) {
      if (mem->map)
         dmabuf_sync(mem, DMA_BUF_SYNC_END);
      if (mem->dmabuf_map)
         munmap(mem->dmabuf_map, mem->vk.size);
   } else if (mem->mapped_hi > mem->mapped_lo &&
              (mem->props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
              !(mem->props & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      /*
       * As the blob: dirty CPU cache lines of what was mapped must not be
       * written back later into pages the kernel has handed to someone
       * else.
       */
      mali_kbase_bo_invalidate(dev->kbase, &mem->bo, mem->mapped_lo,
                               mem->mapped_hi - mem->mapped_lo);
   }

   mali_kbase_free(dev->kbase, &mem->bo);
   if (mem->dmabuf_fd >= 0)
      close(mem->dmabuf_fd);
   vk_device_memory_destroy(&dev->vk, pAllocator, &mem->vk);
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_MapMemory2KHR(VkDevice _device, const VkMemoryMapInfoKHR *pMemoryMapInfo,
                   void **ppData)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   VK_FROM_HANDLE(mali_device_memory, mem, pMemoryMapInfo->memory);

   if (!mem) {
      *ppData = NULL;
      return VK_SUCCESS;
   }
   if (!(mem->props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
      return vk_errorf(dev, VK_ERROR_MEMORY_MAP_FAILED, "memory is not host-visible");
   if (mem->map)
      return vk_errorf(dev, VK_ERROR_MEMORY_MAP_FAILED, "memory is already mapped");

   const uint64_t offset = pMemoryMapInfo->offset;
   const uint64_t size = vk_device_memory_range(&mem->vk, offset, pMemoryMapInfo->size);
   uint8_t *base;

   if (mem->dmabuf_fd >= 0) {
      if (!mem->dmabuf_map) {
         void *p = mmap(NULL, mem->vk.size, PROT_READ | PROT_WRITE, MAP_SHARED,
                        mem->dmabuf_fd, 0);
         if (p == MAP_FAILED)
            return vk_errorf(dev, VK_ERROR_MEMORY_MAP_FAILED,
                             "mmap of dma-buf fd %d failed: %s", mem->dmabuf_fd,
                             strerror(errno));
         mem->dmabuf_map = p;
      }
      dmabuf_sync(mem, DMA_BUF_SYNC_START);
      base = mem->dmabuf_map;
   } else {
      /* SAME_VA memory: the mapping made at allocation. */
      base = mem->bo.cpu;
      if (!base)
         return vk_errorf(dev, VK_ERROR_MEMORY_MAP_FAILED, "memory has no CPU mapping");
   }

   mem->map = base + offset;
   if (mem->mapped_hi == mem->mapped_lo) {
      mem->mapped_lo = offset;
      mem->mapped_hi = offset + size;
   } else {
      mem->mapped_lo = MIN2(mem->mapped_lo, offset);
      mem->mapped_hi = MAX2(mem->mapped_hi, offset + size);
   }
   *ppData = mem->map;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_UnmapMemory2KHR(VkDevice _device, const VkMemoryUnmapInfoKHR *pMemoryUnmapInfo)
{
   VK_FROM_HANDLE(mali_device_memory, mem, pMemoryUnmapInfo->memory);

   if (!mem || !mem->map)
      return VK_SUCCESS;
   /* No cache maintenance. The SAME_VA mapping stays: unmapping it
    * would free the memory. The dma-buf mapping stays until vkFreeMemory. */
   if (mem->dmabuf_fd >= 0)
      dmabuf_sync(mem, DMA_BUF_SYNC_END);
   mem->map = NULL;
   return VK_SUCCESS;
}

static VkResult
sync_ranges(struct mali_device *dev, uint32_t count, const VkMappedMemoryRange *ranges,
            bool flush)
{
   for (uint32_t i = 0; i < count; i++) {
      VK_FROM_HANDLE(mali_device_memory, mem, ranges[i].memory);

      /* Host-coherent types need nothing. */
      if (mem->props & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
         continue;
      const uint64_t size = vk_device_memory_range(&mem->vk, ranges[i].offset, ranges[i].size);
      if (!size)
         continue;

      if (mem->dmabuf_fd >= 0) {
         dmabuf_sync(mem, flush ? DMA_BUF_SYNC_END : DMA_BUF_SYNC_START);
         continue;
      }

      enum mali_kbase_result r =
         flush ? mali_kbase_bo_flush(dev->kbase, &mem->bo, ranges[i].offset, size)
               : mali_kbase_bo_invalidate(dev->kbase, &mem->bo, ranges[i].offset, size);
      if (r != MALI_KBASE_SUCCESS)
         return vk_errorf(dev, alloc_result(r), "%s of [0x%llx, +0x%llx) failed: %s",
                          flush ? "flush" : "invalidate",
                          (unsigned long long)ranges[i].offset, (unsigned long long)size,
                          mali_kbase_result_str(r));
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_FlushMappedMemoryRanges(VkDevice _device, uint32_t memoryRangeCount,
                             const VkMappedMemoryRange *pMemoryRanges)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   return sync_ranges(dev, memoryRangeCount, pMemoryRanges, true);
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_InvalidateMappedMemoryRanges(VkDevice _device, uint32_t memoryRangeCount,
                                  const VkMappedMemoryRange *pMemoryRanges)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   return sync_ranges(dev, memoryRangeCount, pMemoryRanges, false);
}

/*
 * VK_KHR_external_memory_fd: the dma-buf an exportable allocation was made
 * from. Only a scanout allocation (a VK_KHR_display swapchain image, made
 * from a GBM buffer object) has one; an ordinary kbase allocation cannot be
 * exported on the kernels this driver runs on (there is no fd ioctl for it).
 * wsi_common_display asks for it to turn each swapchain image into a DRM
 * framebuffer.
 */
VKAPI_ATTR VkResult VKAPI_CALL
mali_GetMemoryFdKHR(VkDevice _device, const VkMemoryGetFdInfoKHR *pGetFdInfo, int *pFd)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   VK_FROM_HANDLE(mali_device_memory, mem, pGetFdInfo->memory);

   if (pGetFdInfo->handleType != VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT)
      return vk_errorf(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "handle type 0x%x: only dma-buf can be exported",
                       pGetFdInfo->handleType);
   if (mem->dmabuf_fd < 0)
      return vk_errorf(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "memory was not allocated for export");
   const int fd = dup(mem->dmabuf_fd);
   if (fd < 0)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   *pFd = fd;
   return VK_SUCCESS;
}
