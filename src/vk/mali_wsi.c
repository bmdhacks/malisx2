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
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "drm-uapi/drm_fourcc.h"
#include "util/format/u_format.h"
#include "util/log.h"
#include "util/stack_array.h"
#include "util/u_atomic.h"
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

/*
 * The image's layout: linear with the buffer's stride. We asked gralloc for
 * no AFBC (mali_wsi_gralloc_usage) and take the stride the loader passes
 * from the ANativeWindowBuffer; the blob reads stride, offset and modifier
 * from gralloc metadata instead. Linear render targets and textures need
 * 64-byte aligned rows.
 */
static VkResult
gralloc_layout(struct mali_device *dev, struct mali_image *image, uint32_t stride_px)
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
   return VK_SUCCESS;
}

/* Imported as the blob imports native buffers: kbase MEM_IMPORT of the
 * dma-buf, then a sticky map. The fd belongs to gralloc; the memory object
 * owns a dup. A refusal is logged where it happens (mali_device_memory.c). */
static VkResult
gralloc_import(struct mali_device *dev, struct mali_image *image, int dmabuf_fd,
               const VkAllocationCallbacks *alloc)
{
   int fd = fcntl(dmabuf_fd, F_DUPFD_CLOEXEC, 0);
   if (fd < 0) {
      const int err = errno;
      return wsi_errorf(dev, err == EMFILE ? VK_ERROR_TOO_MANY_OBJECTS :
                             err == EBADF ? VK_ERROR_INVALID_EXTERNAL_HANDLE :
                                            VK_ERROR_OUT_OF_HOST_MEMORY,
                       "cannot duplicate the gralloc buffer's fd %d: %s", dmabuf_fd,
                       strerror(err));
   }
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

VkResult
mali_wsi_image_bind_buffer(struct mali_device *dev, struct mali_image *image, int dmabuf_fd,
                           uint32_t stride_px, const VkAllocationCallbacks *alloc)
{
   VkResult result = gralloc_layout(dev, image, stride_px);
   if (result != VK_SUCCESS)
      return result;
   if (dmabuf_fd < 0)
      return wsi_errorf(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "gralloc buffer without a dma-buf");
   return gralloc_import(dev, image, dmabuf_fd, alloc);
}

/*
 * A gralloc handle holds several fds and their order and kind are the
 * gralloc's business: MediaTek's has an anonymous inode first, the dma-buf
 * second and a metadata memfd third; Arm-derived ones (Rockchip's) can hold
 * the buffer and a second dma-buf for the buffer's metadata. The blob asks
 * the gralloc mapper which is the buffer. We have no mapper, so we look at
 * what each fd is and how big it is:
 *
 *  - /proc/self/fd/N reads "/dmabuf:<name>" for a dma-buf, and just
 *    "/dmabuf:" for one nobody named (fstatfs for the dma-buf filesystem's
 *    magic is refused by SELinux on some devices);
 *  - lseek to the end gives a dma-buf's size, so a metadata buffer is
 *    recognised as too small for the image.
 *
 * Candidates are tried in this order, and the first import that works wins:
 * dma-bufs at least as big as the image, other fds that are not known to be
 * too small, then the fds that are too small (so a wrong size can never
 * make this worse than trying every fd).
 */
#define GRALLOC_MAX_FDS 8

struct gralloc_fd {
   int fd;
   bool named;        /* a dma-buf by its /proc link */
   int64_t size;      /* lseek, or -1 */
   VkResult tried;    /* the import's result, VK_NOT_READY if not tried */
   char link[40];
};

static void
gralloc_fd_describe(struct gralloc_fd *g, int fd)
{
   char path[32];

   g->fd = fd;
   g->tried = VK_NOT_READY;
   snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
   const ssize_t n = readlink(path, g->link, sizeof(g->link) - 1);
   g->link[n > 0 ? n : 0] = '\0';
   g->named = n >= 8 && !strncmp(g->link, "/dmabuf:", 8);
   g->size = mali_dmabuf_fd_size(fd);
}

static unsigned
gralloc_fd_class(const struct gralloc_fd *g, uint64_t wanted)
{
   if (g->size >= 0 && (uint64_t)g->size < wanted)
      return 2;
   return g->named ? 0 : 1;
}

/* The dma-buf exporter (the heap) from /proc/self/fdinfo, for the log. */
static void
gralloc_fd_exporter(int fd, char *out, size_t n)
{
   char path[40], line[128];

   snprintf(out, n, "?");
   snprintf(path, sizeof(path), "/proc/self/fdinfo/%d", fd);
   FILE *f = fopen(path, "re");
   if (!f)
      return;
   while (fgets(line, sizeof(line), f)) {
      if (strncmp(line, "exp_name:", 9))
         continue;
      const char *v = line + 9;
      while (*v == ' ' || *v == '\t')
         v++;
      snprintf(out, n, "%.*s", (int)strcspn(v, "\r\n"), v);
      break;
   }
   fclose(f);
}

static void
gralloc_fd_log(const struct gralloc_fd *g, unsigned index, bool error)
{
   char exporter[40];
   char line[256];

   gralloc_fd_exporter(g->fd, exporter, sizeof(exporter));
   snprintf(line, sizeof(line),
            "malisx2: WSI:   handle fd %u = %d: link \"%s\", size %lld, exporter %s, import %s",
            index, g->fd, g->link, (long long)g->size, exporter,
            g->tried == VK_NOT_READY ? "not tried" : vk_Result_to_str(g->tried));
   if (error)
      mesa_loge("%s", line);
   else
      mesa_logi("%s", line);
}

static void
gralloc_log_image(const struct mali_image *image, uint32_t stride_px, unsigned nfds, bool error)
{
   char line[256];

   snprintf(line, sizeof(line),
            "malisx2: WSI: %s a gralloc buffer for a %ux%u image of format %d: stride %u px "
            "(%u bytes per row), %llu bytes wanted, %u fd(s), CPU page size %ld",
            error ? "cannot bind" : "bound", image->vk.extent.width, image->vk.extent.height,
            image->vk.format, stride_px, image->planes[0].layout.slices[0].row_stride,
            (unsigned long long)image->size, nfds, sysconf(_SC_PAGESIZE));
   if (error)
      mesa_loge("%s", line);
   else
      mesa_logi("%s", line);
}

VkResult
mali_wsi_image_bind_gralloc_fds(struct mali_device *dev, struct mali_image *image,
                                const int *fds, unsigned nfds, uint32_t stride_px,
                                const VkAllocationCallbacks *alloc)
{
   static int logged_first;

   VkResult result = gralloc_layout(dev, image, stride_px);
   if (result != VK_SUCCESS)
      return result;
   if (!nfds)
      return wsi_errorf(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "gralloc buffer without a dma-buf");

   struct gralloc_fd cand[GRALLOC_MAX_FDS];
   const unsigned n = MIN2(nfds, GRALLOC_MAX_FDS);
   for (unsigned i = 0; i < n; i++)
      gralloc_fd_describe(&cand[i], fds[i]);

   result = VK_ERROR_INVALID_EXTERNAL_HANDLE;
   for (unsigned pass = 0; pass < 3; pass++) {
      for (unsigned i = 0; i < n; i++) {
         if (gralloc_fd_class(&cand[i], image->size) != pass)
            continue;
         cand[i].tried = gralloc_import(dev, image, cand[i].fd, alloc);
         if (cand[i].tried == VK_SUCCESS) {
            /* The first buffer of the process is logged whole: what the
             * gralloc hands out is the first thing to look at when a
             * device's picture is wrong. */
            if (p_atomic_cmpxchg(&logged_first, 0, 1) == 0) {
               gralloc_log_image(image, stride_px, nfds, false);
               gralloc_fd_log(&cand[i], i, false);
            }
            return VK_SUCCESS;
         }
         result = cand[i].tried;
         /* A dup or host memory failure does not depend on which fd. */
         if (result != VK_ERROR_INVALID_EXTERNAL_HANDLE)
            goto fail;
      }
   }

fail:
   gralloc_log_image(image, stride_px, nfds, true);
   for (unsigned i = 0; i < n; i++)
      gralloc_fd_log(&cand[i], i, true);
   return result;
}

/*
 * The stride of a buffer whose gralloc does not report one.
 *
 * Arm's gralloc 4 (Rockchip's is built from it) sets the stride it returns
 * from an allocation only for buffers the CPU can read or write; for a
 * swapchain buffer, which only the GPU and the display touch, it is 0, and
 * Android's gralloc wrapper and the Vulkan loader pass the 0 on. The
 * buffer's real layout is in its private handle, which is what the gralloc
 * mapper reads it from. The handle is two fds (the buffer's dma-buf and a
 * memfd of metadata) followed by 37 ints; this is the front of it.
 */
#define ARM_GRALLOC_VERSION  12          /* sizeof(native_handle_t) */
#define ARM_GRALLOC_NUM_FDS  2
#define ARM_GRALLOC_NUM_INTS 37          /* later builds append more */
#define ARM_GRALLOC_MAGIC    0x3141592

struct arm_gralloc_handle {
   int version, num_fds, num_ints;       /* native_handle_t */
   int share_fd, share_attr_fd;
   int magic;
   int width, height, req_format;
   uint64_t producer_usage, consumer_usage;
   int stride;                           /* the 0 the loader passes */
   struct {
      uint64_t base;
      uint32_t modifiers;                /* 0 for an uncompressed layout */
   } alloc_format;
   struct {
      uint32_t offset, byte_stride, alloc_width, alloc_height;
   } plane[3];
   int size, layer_count;
   uint64_t backing_store_id, attr_size, reserved_region_size;
   int type;
};

_Static_assert(offsetof(struct arm_gralloc_handle, magic) == 0x14, "arm gralloc magic");
_Static_assert(offsetof(struct arm_gralloc_handle, stride) == 0x38, "arm gralloc stride");
_Static_assert(offsetof(struct arm_gralloc_handle, alloc_format) == 0x40, "arm gralloc format");
_Static_assert(offsetof(struct arm_gralloc_handle, plane) == 0x50, "arm gralloc planes");
_Static_assert(offsetof(struct arm_gralloc_handle, size) == 0x80, "arm gralloc size");
_Static_assert(offsetof(struct arm_gralloc_handle, type) == 0xa0, "arm gralloc type");
_Static_assert(sizeof(struct arm_gralloc_handle) ==
                  3 * sizeof(int) + 4 * (ARM_GRALLOC_NUM_FDS + ARM_GRALLOC_NUM_INTS),
               "arm gralloc handle size");

/*
 * Plane 0's row stride in pixels from an Arm gralloc handle, or 0 with a
 * reason. A handle is taken for one only if its counts, magic and
 * dimensions are right, and for a buffer we can use only if it is a single
 * uncompressed layer that starts at offset 0 and holds the image: anything
 * else would be read with the wrong layout.
 */
static uint32_t
arm_gralloc_stride_px(const int *handle, const struct mali_image *image, const char **why)
{
   const unsigned bpp = util_format_get_blocksize(image->planes[0].format);
   struct arm_gralloc_handle h;

   if (handle[0] != ARM_GRALLOC_VERSION || handle[1] != ARM_GRALLOC_NUM_FDS ||
       handle[2] < ARM_GRALLOC_NUM_INTS || handle[2] > 256) {
      *why = "the handle is not an Arm gralloc 4 handle (its fd and int counts)";
      return 0;
   }
   memcpy(&h, handle, sizeof(h));
   if (h.magic != ARM_GRALLOC_MAGIC) {
      *why = "the handle is not an Arm gralloc 4 handle (its magic)";
      return 0;
   }
   if (h.width != (int)image->vk.extent.width || h.height != (int)image->vk.extent.height ||
       h.layer_count > 1) {
      *why = "the handle's size disagrees with the image";
      return 0;
   }
   if (h.alloc_format.modifiers) {
      *why = "gralloc allocated a compressed or tiled buffer";
      return 0;
   }
   const uint32_t pitch = h.plane[0].byte_stride;
   if (h.plane[0].offset || !pitch || pitch % bpp ||
       pitch < (uint64_t)image->vk.extent.width * bpp ||
       h.plane[0].alloc_height < image->vk.extent.height ||
       (h.size > 0 && (uint64_t)pitch * image->vk.extent.height > (uint32_t)h.size)) {
      *why = "the handle's plane 0 does not describe a linear buffer that holds the image";
      return 0;
   }
   return pitch / bpp;
}

VkResult
mali_wsi_image_bind_native_handle(struct mali_device *dev, struct mali_image *image,
                                  const int *handle, int stride_px,
                                  const VkAllocationCallbacks *alloc)
{
   static int logged_derived;

   if (!handle || handle[1] < 1 || handle[1] > 64 || handle[2] < 0 || handle[2] > 1024)
      return wsi_errorf(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "native buffer without a dma-buf fd");

   if (stride_px <= 0) {
      const char *why;
      const uint32_t derived = arm_gralloc_stride_px(handle, image, &why);
      if (!derived)
         return wsi_errorf(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                          "native buffer with stride %d and no layout to take instead: %s",
                          stride_px, why);
      if (p_atomic_cmpxchg(&logged_derived, 0, 1) == 0)
         mesa_logi("malisx2: WSI: gralloc reports no stride for the swapchain buffers; using "
                   "the row stride in the Arm gralloc handle, %u pixels", derived);
      stride_px = (int)derived;
   }

   return mali_wsi_image_bind_gralloc_fds(dev, image, handle + 3, (unsigned)handle[1],
                                          (uint32_t)stride_px, alloc);
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
