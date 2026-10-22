/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Android only: the HAL module the Android Vulkan loader opens, and the
 * VK_ANDROID_native_buffer entry points it drives swapchains with. The
 * work is in mali_wsi.c; this file unwraps Android's structures.
 *
 * The loader (/system/lib64/libvulkan.so) loads the driver with
 * android_load_sphal_library or android_dlopen_ext (adrenotools hooks both
 * to substitute this library), looks up the symbol HMI, and opens device
 * "vk0" through it; from then on it only uses the three functions of
 * hwvulkan_device_t and vkGetInstanceProcAddr. Surfaces and swapchains
 * are the loader's; per swapchain image it calls vkCreateImage with a
 * VkNativeBufferANDROID, per acquire vkAcquireImageANDROID, per present
 * vkQueueSignalReleaseImageANDROID.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/system_properties.h>
#include <unistd.h>

#include <hardware/hwvulkan.h>

#include "util/log.h"
#include "util/u_atomic.h"
#include "util/u_call_once.h"
#include "vk_log.h"
#include "vk_util.h"

#include "mali_image.h"
#include "mali_vk.h"
#include "mali_wsi.h"

#define wsi_errorf(obj, result, ...)                                            \
   (mesa_loge("malisx2: WSI: " __VA_ARGS__), vk_errorf(obj, result, __VA_ARGS__))

/* ---------------------------------------------------------------------- */
/* The HAL module                                                          */

static int mali_hal_open(const struct hw_module_t *mod, const char *id,
                         struct hw_device_t **dev);

static_assert(HWVULKAN_DISPATCH_MAGIC == ICD_LOADER_MAGIC,
              "dispatchable handles start with the loader magic");

static hw_module_methods_t mali_hal_methods = {
   .open = mali_hal_open,
};

__attribute__((visibility("default"))) struct hwvulkan_module_t HAL_MODULE_INFO_SYM = {
   .common = {
      .tag = HARDWARE_MODULE_TAG,
      .module_api_version = HWVULKAN_MODULE_API_VERSION_0_1,
      .hal_api_version = HARDWARE_MAKE_API_VERSION(1, 0),
      .id = HWVULKAN_HARDWARE_MODULE_ID,
      .name = "malisx2 Vulkan driver (kbase)",
      .author = "ARMSX2 and bmdhacks",
      .methods = &mali_hal_methods,
   },
};

static int
mali_hal_close(struct hw_device_t *dev)
{
   /* Called when the loader unloads the driver. */
   free(container_of(dev, hwvulkan_device_t, common));
   return 0;
}

static int
mali_hal_open(const struct hw_module_t *mod, const char *id, struct hw_device_t **dev)
{
   if (mod != &HAL_MODULE_INFO_SYM.common || strcmp(id, HWVULKAN_DEVICE_0) != 0)
      return -1;

   hwvulkan_device_t *hal = calloc(1, sizeof(*hal));
   if (!hal)
      return -1;
   *hal = (hwvulkan_device_t){
      .common = {
         .tag = HARDWARE_DEVICE_TAG,
         .version = HWVULKAN_DEVICE_API_VERSION_0_1,
         .module = &HAL_MODULE_INFO_SYM.common,
         .close = mali_hal_close,
      },
      .EnumerateInstanceExtensionProperties =
         (PFN_vkEnumerateInstanceExtensionProperties)vk_icdGetInstanceProcAddr(
            NULL, "vkEnumerateInstanceExtensionProperties"),
      .CreateInstance =
         (PFN_vkCreateInstance)vk_icdGetInstanceProcAddr(NULL, "vkCreateInstance"),
      .GetInstanceProcAddr =
         (PFN_vkGetInstanceProcAddr)vk_icdGetInstanceProcAddr(NULL, "vkGetInstanceProcAddr"),
   };
   *dev = &hal->common;
   return 0;
}

/* ---------------------------------------------------------------------- */
/* Gralloc usage                                                           */

static char no_afbc_property[PROP_VALUE_MAX];
static util_once_flag no_afbc_once = UTIL_ONCE_FLAG_INIT;

static void
no_afbc_usage_init(void)
{
   bool fallback;
   __system_property_get("ro.vendor.arm.gralloc.no_afbc_usage_flags", no_afbc_property);
   mali_wsi_no_afbc_usage(no_afbc_property, &fallback);
   if (fallback)
      mesa_logw("malisx2: ro.vendor.arm.gralloc.no_afbc_usage_flags is not readable or "
                "not a usable value (\"%s\")", no_afbc_property);
}

uint64_t
mali_wsi_swapchain_no_afbc(unsigned arch)
{
   util_call_once(&no_afbc_once, no_afbc_usage_init);
   return mali_wsi_arch_no_afbc_usage(arch, no_afbc_property);
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_GetSwapchainGrallocUsage2ANDROID(VkDevice _device, VkFormat format,
                                      VkImageUsageFlags imageUsage,
                                      VkSwapchainImageUsageFlagsANDROID swapchainImageUsage,
                                      uint64_t *grallocConsumerUsage,
                                      uint64_t *grallocProducerUsage)
{
   VK_FROM_HANDLE(mali_device, dev, _device);

   /* Shared (front-buffer) presentation is not offered: sharedImage is
    * false, so the loader never asks for it. */
   if (swapchainImageUsage & VK_SWAPCHAIN_IMAGE_USAGE_SHARED_BIT_ANDROID)
      return wsi_errorf(dev, VK_ERROR_FORMAT_NOT_SUPPORTED,
                       "shared presentable images are not supported");
   const VkResult result =
      mali_wsi_gralloc_usage(mali_device_physical(dev), format, imageUsage,
                             mali_wsi_swapchain_no_afbc(mali_device_physical(dev)->arch),
                             grallocConsumerUsage, grallocProducerUsage);
   /* Once per process: what gralloc is asked for decides whether a buffer
    * comes back linear, so it belongs next to the buffers' own log line. */
   static int logged_usage;
   if (result == VK_SUCCESS && p_atomic_cmpxchg(&logged_usage, 0, 1) == 0)
      mesa_logi("malisx2: WSI: gralloc usage for format %d, image usage 0x%x: consumer 0x%llx, "
                "producer 0x%llx (no-AFBC property \"%s\")", format, imageUsage,
                (unsigned long long)*grallocConsumerUsage,
                (unsigned long long)*grallocProducerUsage, no_afbc_property);
   return result;
}

/* The gralloc0 form: 32 bits, so no room for the no-AFBC bit (the blob
 * drops it too). The loader uses it only if ...Usage2 is missing. */
VKAPI_ATTR VkResult VKAPI_CALL
mali_GetSwapchainGrallocUsageANDROID(VkDevice _device, VkFormat format,
                                     VkImageUsageFlags imageUsage, int *grallocUsage)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   uint64_t consumer, producer;
   VkResult result = mali_wsi_gralloc_usage(mali_device_physical(dev), format, imageUsage,
                                            0, &consumer, &producer);
   if (result == VK_SUCCESS)
      *grallocUsage = (int)(consumer | producer);
   return result;
}

/* ---------------------------------------------------------------------- */
/* Swapchain images                                                        */

/*
 * What the loader handed over, for the log: the handle's counts and ints
 * (gralloc keeps format, size and usage in them, and the layout of the
 * buffer, AFBC or not), and the native buffer's own fields. Logged when a
 * bind fails, and once for the first buffer of the process.
 */
static void
log_native_buffer(const VkNativeBufferANDROID *nb, const struct mali_image *image, bool error)
{
   const native_handle_t *h = nb->handle;
   char ints[24 * 9 + 1] = "";
   char line[512];
   const int n = h->numInts < 24 ? h->numInts : 24;

   for (int i = 0; i < n; i++)
      snprintf(ints + strlen(ints), sizeof(ints) - strlen(ints), "%x ", h->data[h->numFds + i]);
   snprintf(line, sizeof(line),
            "malisx2: WSI: native buffer for a %ux%u image: stride %d, gralloc format %d, usage "
            "0x%x, usage3 0x%llx, handle version %d, %d fd(s), %d int(s) [%s%s]",
            image->vk.extent.width, image->vk.extent.height, nb->stride, nb->format,
            (unsigned)nb->usage, (unsigned long long)nb->usage3, h->version, h->numFds,
            h->numInts, ints, h->numInts > n ? "..." : "");
   if (error)
      mesa_loge("%s", line);
   else
      mesa_logi("%s", line);
}

VkResult
mali_android_bind_native_buffer(struct mali_device *dev, struct mali_image *image,
                                const VkNativeBufferANDROID *nb,
                                const VkAllocationCallbacks *alloc)
{
   static int logged_first;

   if (!nb->handle || nb->handle->numFds < 1)
      return wsi_errorf(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "native buffer without a dma-buf fd");
   if (nb->stride <= 0) {
      log_native_buffer(nb, image, true);
      return wsi_errorf(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "native buffer with stride %d", nb->stride);
   }

   /* Which fd is the buffer is decided there: see mali_wsi.c. */
   const VkResult result =
      mali_wsi_image_bind_gralloc_fds(dev, image, nb->handle->data, (unsigned)nb->handle->numFds,
                                      (uint32_t)nb->stride, alloc);
   if (result != VK_SUCCESS)
      log_native_buffer(nb, image, true);
   else if (p_atomic_cmpxchg(&logged_first, 0, 1) == 0)
      log_native_buffer(nb, image, false);
   return result;
}

/* ---------------------------------------------------------------------- */
/* Acquire and present                                                     */

VKAPI_ATTR VkResult VKAPI_CALL
mali_AcquireImageANDROID(VkDevice _device, VkImage image, int nativeFenceFd,
                         VkSemaphore semaphore, VkFence fence)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   return mali_wsi_acquire(dev, nativeFenceFd, semaphore, fence);
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_QueueSignalReleaseImageANDROID(VkQueue _queue, uint32_t waitSemaphoreCount,
                                    const VkSemaphore *pWaitSemaphores, VkImage image,
                                    int *pNativeFenceFd)
{
   VK_FROM_HANDLE(mali_queue, queue, _queue);
   return mali_wsi_release(queue, waitSemaphoreCount, pWaitSemaphores, pNativeFenceFd);
}
