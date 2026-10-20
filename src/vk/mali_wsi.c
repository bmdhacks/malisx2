/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The Android window-system path without Android's headers: gralloc usage
 * for swapchain images, binding a gralloc buffer to a swapchain image, and
 * the acquire and release fences, matching the blob's behaviour.
 *
 * Mostly frontend-neutral: gralloc usage, binding a buffer and acquire
 * touch nothing that differs between the
 * command-stream frontend (v11) and the job manager (v7 and v9) -- acquire
 * goes through the runtime's own sync-file import, which already
 * dispatches to whichever vk_sync type the physical device chose -- so
 * they are compiled once, in the CSF build (mali_wsi_gralloc_usage,
 * mali_wsi_image_bind_buffer, mali_wsi_acquire below). Release reads a
 * mali_sync's payload directly (its req[] has a different meaning and size
 * per frontend, mali_jm.h vs mali_queue.h), so it is compiled per
 * architecture, like mali_sync.c;
 * mali_wsi_release (compiled once, next to the other three) dispatches to
 * the queue's frontend.
 */

#include "mali_wsi.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "drm-uapi/drm_fourcc.h"
#include "util/format/u_format.h"
#include "util/log.h"
#include "util/stack_array.h"
#include "util/u_math.h"
#include "vk_enum_to_str.h"
#include "vk_fence.h"
#include "vk_format.h"
#include "vk_log.h"
#include "vk_semaphore.h"
#include "vk_util.h"

#include "mali_image.h"
#include "mali_memory.h"
#if defined(PAN_ARCH) && PAN_ARCH < 10
#include "mali_jm.h"
#else
#include "mali_queue.h"
#endif
#include "mali_vk.h"

/* Failures on the swapchain path are always in the log (the runtime's
 * own message is debug-only): the loader turns them into a failed
 * vkCreateSwapchainKHR or present with no reason given. */
#define wsi_errorf(obj, result, ...)                                            \
   (mesa_loge("malisx2: WSI: " __VA_ARGS__), vk_errorf(obj, result, __VA_ARGS__))

#if PAN_ARCH >= 10

/* ---------------------------------------------------------------------- */
/* Gralloc usage                                                           */

uint64_t
mali_wsi_no_afbc_usage(const char *property_value, bool *fallback)
{
   uint64_t value = 0;
   if (property_value && property_value[0]) {
      char *end;
      errno = 0;
      const unsigned long long v = strtoull(property_value, &end, 0);
      if (errno == 0 && *end == '\0')
         value = v;
   }
   *fallback = value == 0;
   return value ? value : MALI_GRALLOC_USAGE_CPU_READ_RARELY;
}

uint64_t
mali_wsi_arch_no_afbc_usage(unsigned arch, const char *property_value)
{
   bool fallback;
   const uint64_t bits = mali_wsi_no_afbc_usage(property_value, &fallback);
   return fallback && arch == 11 ? 0x0200000000000000ull : bits;
}

#ifndef VK_USE_PLATFORM_ANDROID_KHR
uint64_t
mali_wsi_swapchain_no_afbc(unsigned arch)
{
   return mali_wsi_arch_no_afbc_usage(arch, NULL);
}
#endif

VkResult
mali_wsi_gralloc_usage(struct mali_physical_device *pdev, VkFormat format,
                       VkImageUsageFlags usage, uint64_t no_afbc,
                       uint64_t *consumer, uint64_t *producer)
{
   /* What a swapchain image can be used for here, and the format feature
    * each use needs (swapchain images are linear, whose features are the
    * optimal ones, mali_formats.c). */
   static const struct {
      VkImageUsageFlags usage;
      VkFormatFeatureFlags feature;
   } uses[] = {
      {VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_FORMAT_FEATURE_TRANSFER_SRC_BIT},
      {VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_FORMAT_FEATURE_TRANSFER_DST_BIT},
      {VK_IMAGE_USAGE_SAMPLED_BIT, VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT},
      {VK_IMAGE_USAGE_STORAGE_BIT, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT},
      {VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT},
      {VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT},
   };

   enum pipe_format planes[MALI_IMAGE_MAX_PLANES];
   if (mali_format_planes(format, planes) != 1 || vk_format_is_depth_or_stencil(format) ||
       util_format_is_compressed(planes[0]))
      return wsi_errorf(pdev, VK_ERROR_FORMAT_NOT_SUPPORTED,
                       "swapchain images of format %d are not supported", format);

   const VkFormatFeatureFlags feat = mali_format_image_features(pdev, format);
   VkImageUsageFlags left = usage;
   for (unsigned i = 0; i < ARRAY_SIZE(uses); i++) {
      if (!(usage & uses[i].usage))
         continue;
      if (!(feat & uses[i].feature))
         return wsi_errorf(pdev, VK_ERROR_FORMAT_NOT_SUPPORTED,
                          "swapchain images of format %d cannot have usage 0x%x", format,
                          uses[i].usage);
      left &= ~uses[i].usage;
   }
   if (left)
      return wsi_errorf(pdev, VK_ERROR_FORMAT_NOT_SUPPORTED,
                       "swapchain images cannot have usage 0x%x", left);

   *consumer = MALI_GRALLOC_USAGE_HW_TEXTURE;
   *producer = MALI_GRALLOC_USAGE_HW_RENDER | no_afbc;
   return VK_SUCCESS;
}

VkResult
mali_wsi_ahb_usage(struct mali_physical_device *pdev, const VkPhysicalDeviceImageFormatInfo2 *info,
                   uint64_t *usage)
{
   const VkImageCreateFlags mutable_format =
      VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
   if (info->type != VK_IMAGE_TYPE_2D || info->tiling != VK_IMAGE_TILING_OPTIMAL ||
       (info->flags & ~mutable_format))
      return wsi_errorf(pdev, VK_ERROR_FORMAT_NOT_SUPPORTED,
                       "hardware-buffer images of type %d, tiling %d, flags 0x%x are not "
                       "supported", info->type, info->tiling, info->flags);

   uint64_t consumer, producer;
   VkResult result = mali_wsi_gralloc_usage(pdev, info->format, info->usage,
                                            mali_wsi_swapchain_no_afbc(pdev->arch), &consumer,
                                            &producer);
   if (result != VK_SUCCESS)
      return result;
   /* The loader merges the two halves of vkGetSwapchainGrallocUsage2ANDROID
    * the same way; its gralloc1 conversion changes only the CPU_*_OFTEN
    * bits, which are not among these. */
   *usage = consumer | producer;
   return VK_SUCCESS;
}

/* ---------------------------------------------------------------------- */
/* Swapchain image memory                                                  */

VkResult
mali_wsi_image_bind_buffer(struct mali_device *dev, struct mali_image *image, int dmabuf_fd,
                           uint32_t stride_px, const VkAllocationCallbacks *alloc)
{
   const struct vk_image *vi = &image->vk;

   if (image->mem)
      return wsi_errorf(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "swapchain image is bound already");
   if (image->plane_count != 1 || vi->image_type != VK_IMAGE_TYPE_2D ||
       vi->mip_levels != 1 || vi->array_layers != 1 || vi->samples != 1 ||
       util_format_is_compressed(image->planes[0].format))
      return wsi_errorf(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "swapchain image: only single-level, single-layer, "
                       "single-sample 2D color images");
   if (dmabuf_fd < 0)
      return wsi_errorf(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "gralloc buffer without a dma-buf");

   /*
    * Linear with the buffer's stride. We asked gralloc for no AFBC
    * (mali_wsi_gralloc_usage) and take the stride the loader passes from
    * the ANativeWindowBuffer; the blob reads stride, offset and modifier
    * from gralloc metadata instead. Linear render targets and textures
    * need 64-byte aligned rows.
    */
   const enum pipe_format format = image->planes[0].format;
   const unsigned bpp = util_format_get_blocksize(format);
   const uint64_t row = (uint64_t)stride_px * bpp;
   if (stride_px < vi->extent.width || row % 64 || row > UINT32_MAX)
      return wsi_errorf(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "gralloc stride of %u pixels (%llu bytes) for a %u pixel wide "
                       "image is not usable (rows must be 64-byte aligned)",
                       stride_px, (unsigned long long)row, vi->extent.width);

   const struct mali_image_layout_info info = {
      .format = format,
      .modifier = DRM_FORMAT_MOD_LINEAR,
      .width = vi->extent.width,
      .height = vi->extent.height,
      .depth = 1,
      .samples = 1,
      .levels = 1,
      .layers = 1,
   };
   struct mali_image_plane_layout *pl = &image->planes[0].layout;
   if (!mali_image_plane_layout_init(&info, 0, pl))
      return wsi_errorf(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "no linear layout for a swapchain image of format %d", vi->format);
   struct mali_image_slice *s = &pl->slices[0];
   s->row_stride = (uint32_t)row;
   s->surface_stride = row * vi->extent.height;
   s->size = s->surface_stride;
   pl->array_stride = s->size;
   pl->data_size = s->size;
   image->size = s->size;
   image->vk.tiling = VK_IMAGE_TILING_LINEAR;
   image->vk.drm_format_mod = DRM_FORMAT_MOD_LINEAR;

   /* Imported as the blob imports native buffers: kbase MEM_IMPORT of the
    * dma-buf, then a sticky map. The fd belongs to gralloc; the memory
    * object owns a dup. */
   int fd = fcntl(dmabuf_fd, F_DUPFD_CLOEXEC, 0);
   if (fd < 0)
      return wsi_errorf(dev, errno == EMFILE ? VK_ERROR_TOO_MANY_OBJECTS :
                                              VK_ERROR_OUT_OF_HOST_MEMORY,
                       "cannot duplicate the gralloc buffer's fd: %s", strerror(errno));
   const VkImportMemoryFdInfoKHR import = {
      .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
      .fd = fd,
   };
   const VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &import,
      .allocationSize = image->size,
      .memoryTypeIndex = MALI_MEMORY_TYPE_UNCACHED,
   };
   VkDeviceMemory mem_h;
   VkResult result = mali_AllocateMemory(mali_device_to_handle(dev), &ai, alloc, &mem_h);
   if (result != VK_SUCCESS) {
      close(fd);
      return result;
   }

   struct mali_device_memory *mem = mali_device_memory_from_handle(mem_h);
   image->wsi_mem = mem;
   image->mem = mem;
   image->mem_offset = 0;
   image->base = mali_device_memory_gpu_va(mem, 0);
   return VK_SUCCESS;
}

/* ---------------------------------------------------------------------- */
/* Acquire                                                                 */

VkResult
mali_wsi_acquire(struct mali_device *dev, int fd, VkSemaphore semaphore, VkFence fence)
{
   VkDevice dev_h = mali_device_to_handle(dev);
   VkResult result = VK_SUCCESS;

   /*
    * The driver owns fd whatever happens (the VK_ANDROID_native_buffer
    * contract); a successful import owns what it was given. Like the blob,
    * nothing waits here: the semaphore's waiter makes the GPU wait (kcpu on
    * v11, a FENCE_WAIT atom on the job manager), the fence's waiter polls.
    */
   int sem_fd = -1, fence_fd = -1;
   if (fd >= 0) {
      if (semaphore && fence) {
         sem_fd = fd;
         fence_fd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
         if (fence_fd < 0) {
            int err = errno;
            close(fd);
            return wsi_errorf(dev, err == EMFILE ? VK_ERROR_TOO_MANY_OBJECTS :
                                                  VK_ERROR_OUT_OF_HOST_MEMORY,
                             "cannot duplicate the acquire fence: %s", strerror(err));
         }
      } else if (semaphore) {
         sem_fd = fd;
      } else if (fence) {
         fence_fd = fd;
      } else {
         close(fd);
      }
   }

   if (semaphore) {
      const VkImportSemaphoreFdInfoKHR info = {
         .sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
         .semaphore = semaphore,
         .flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT,
         .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
         .fd = sem_fd,
      };
      result = dev->vk.dispatch_table.ImportSemaphoreFdKHR(dev_h, &info);
      if (result == VK_SUCCESS)
         sem_fd = -1;
      else
         MALI_PER_ARCH(sync_file_log_error)(dev, "vkAcquireImageANDROID: semaphore import of",
                                            sem_fd, vk_Result_to_str(result));
   }
   if (result == VK_SUCCESS && fence) {
      const VkImportFenceFdInfoKHR info = {
         .sType = VK_STRUCTURE_TYPE_IMPORT_FENCE_FD_INFO_KHR,
         .fence = fence,
         .flags = VK_FENCE_IMPORT_TEMPORARY_BIT,
         .handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT,
         .fd = fence_fd,
      };
      result = dev->vk.dispatch_table.ImportFenceFdKHR(dev_h, &info);
      if (result == VK_SUCCESS)
         fence_fd = -1;
      else
         MALI_PER_ARCH(sync_file_log_error)(dev, "vkAcquireImageANDROID: fence import of",
                                            fence_fd, vk_Result_to_str(result));
   }

   if (sem_fd >= 0)
      close(sem_fd);
   if (fence_fd >= 0)
      close(fence_fd);
   return result;
}

#endif /* PAN_ARCH >= 10 */

/* ---------------------------------------------------------------------- */
/* Release (per architecture: see the file comment)                        */

/* How many uint64_t slots a mali_sync's req[] holds here: the done value
 * per CSF subqueue on v11, the submission number and tracker atom on v7
 * and v9 (mali_jm.h's MALI_SYNC_REQ_COUNT). Local to this function: the
 * rest of the file never looks inside a mali_sync. */
#if PAN_ARCH < 10
#define MALI_WSI_SYNC_REQ_COUNT MALI_SYNC_REQ_COUNT
#else
#define MALI_WSI_SYNC_REQ_COUNT MALI_SUBQUEUE_COUNT
#endif

/* Declared here, not in mali_queue.h/mali_jm.h: only this file defines and
 * calls the per-arch body (the dispatcher below declares the job-manager
 * variants itself, with MALI_PER_JM_ARCH_DECL). */
VkResult MALI_PER_ARCH(wsi_release)(struct mali_queue *queue, uint32_t count,
                                    const VkSemaphore *semaphores, int *out_fd);

VkResult
MALI_PER_ARCH(wsi_release)(struct mali_queue *queue, uint32_t count, const VkSemaphore *semaphores,
                           int *out_fd)
{
   struct mali_device *dev = container_of(queue->vk.base.device, struct mali_device, vk);
   VkResult result = VK_SUCCESS;

   *out_fd = -1;
   if (!count)
      return VK_SUCCESS;

   STACK_ARRAY(int, fds, count);
   uint32_t nfds = 0;
   uint64_t req[MALI_WSI_SYNC_REQ_COUNT] = {0};

   /*
    * The blob's release: kcpu waits for every semaphore's payload, then a
    * kcpu FENCE_SIGNAL whose sync file is the release fence. Here the
    * payloads are sync files (FENCE_WAIT) and GPU work (CQS waits on the
    * done slots on v11, the submission's tracker on the job manager), both
    * in the sync_file_create hook (mali_queue.h, mali_jm.h).
    */
   pthread_mutex_lock(&dev->lock);
   if (dev->lost) {
      result = VK_ERROR_DEVICE_LOST;
      goto out;
   }
   for (uint32_t i = 0; i < count; i++) {
      VK_FROM_HANDLE(vk_semaphore, sem, semaphores[i]);
      struct vk_sync *sync = vk_semaphore_get_active_sync(sem);
      if (sync->type != &MALI_PER_ARCH(sync_type) || (sync->flags & VK_SYNC_IS_TIMELINE))
         continue;   /* the spec allows binary semaphores only */
      struct mali_sync *s = container_of(sync, struct mali_sync, vk);
      if (s->host_signaled)
         continue;
      if (s->fd >= 0) {
         fds[nfds++] = s->fd;
      } else if (s->submitted) {
         for (unsigned j = 0; j < MALI_WSI_SYNC_REQ_COUNT; j++)
            req[j] = MAX2(req[j], s->req[j]);
      }
      /* A semaphore nothing signals: invalid usage; the blob waits for
       * nothing then, and so do we. */
   }
   result = MALI_PER_ARCH(sync_file_create)(dev, nfds, fds, req, out_fd);
   if (result != VK_SUCCESS) {
      MALI_PER_ARCH(sync_file_log_error)(dev, "vkQueueSignalReleaseImageANDROID: export to", -1,
                                         vk_Result_to_str(result));
      goto out;
   }

   /* Waiting consumes the payloads. */
   for (uint32_t i = 0; i < count; i++) {
      VK_FROM_HANDLE(vk_semaphore, sem, semaphores[i]);
      if (sem->temporary) {
         vk_semaphore_reset_temporary(&dev->vk, sem);
      } else if (sem->permanent.type == &MALI_PER_ARCH(sync_type) &&
                 !(sem->permanent.flags & VK_SYNC_IS_TIMELINE)) {
         MALI_PER_ARCH(sync_clear)(container_of(&sem->permanent, struct mali_sync, vk));
      }
   }

out:
   pthread_mutex_unlock(&dev->lock);
   STACK_ARRAY_FINISH(fds);
   return result;
}

#undef MALI_WSI_SYNC_REQ_COUNT

#if PAN_ARCH >= 10
/*
 * vkQueueSignalReleaseImageANDROID's entry point (mali_android.c) and the
 * host tests call this name; mali_v7_wsi_release and mali_v9_wsi_release
 * are the other variants (defined when this same file is built at
 * PAN_ARCH=7 and 9 into malisx2_vk_v7 and malisx2_vk_v9, always linked
 * next to this CSF build). Compiled once here, like the three functions
 * above.
 */
MALI_PER_JM_ARCH_DECL(VkResult, wsi_release, (struct mali_queue *queue, uint32_t count,
                                              const VkSemaphore *semaphores, int *out_fd));

VkResult
mali_wsi_release(struct mali_queue *queue, uint32_t count, const VkSemaphore *semaphores,
                 int *out_fd)
{
   if (queue->jm) {
      struct mali_device *dev = container_of(queue->vk.base.device, struct mali_device, vk);
      const uint32_t arch = mali_device_physical(dev)->arch;
      return mali_jm_arch_dispatch(arch, wsi_release, queue, count, semaphores, out_fd);
   }
   return MALI_PER_ARCH(wsi_release)(queue, count, semaphores, out_fd);
}
#endif
