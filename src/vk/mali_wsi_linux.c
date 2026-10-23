/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The Linux window system: Mesa's shared Vulkan WSI (src/vulkan/wsi) with
 * a driver behind it, so a Linux application can present straight to the
 * display with VK_KHR_display and no compositor. ARMSX2's SDL/kmsdrm
 * frontend (pcsx2-sdl) uses exactly that on a handheld.
 *
 * wsi_common owns the surfaces, modes and the swapchain; what it needs
 * from the driver is small:
 *
 *  - a wsi_device on the physical device (mali_physical_device.c), filled
 *    by wsi_device_init with a physical-device proc-addr callback and,
 *    below, whether a given DRM node belongs to this GPU;
 *  - VkImages created with the swapchain's LINEAR tiling and the DRM
 *    plane's format (mali_image.c), in memory the driver can export as a
 *    dma-buf for the kernel to scan out (mali_device_memory.c).
 *
 * The page flips, DRM master and the sync files between the display and
 * the swapchain image are wsi_common_display.c and wsi_common_drm.c; this
 * file only answers what they ask the driver.
 */

#include "mali_vk.h"

#include <vulkan/vk_icd.h>

#include "wsi_common.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <gbm.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#include "util/log.h"
#include "util/xmlconfig.h"

/*
 * The physical-device functions wsi_common asks for (memory and format
 * properties on the physical device it was given). The loader's ICD
 * interface resolves them from the same place our vk_icdGetInstanceProcAddr
 * does.
 */
static PFN_vkVoidFunction VKAPI_PTR
mali_wsi_proc_addr(VkPhysicalDevice pdevice, const char *pName)
{
   VK_FROM_HANDLE(mali_physical_device, pdev, pdevice);
   /* The instance's own table, then the physical-device and device
    * trampolines: wsi_common fetches the VkDevice functions it uses
    * (vkCreateImage, vkAllocateMemory, vkQueueSubmit2, ...) through this
    * callback, not only the physical-device ones. */
   return vk_instance_get_proc_addr_unchecked(pdev->vk.instance, pName);
}

/*
 * Whether a display surface on this DRM node can present on this GPU
 * without a copy: a platform device (an SoC's integrated GPU), as panvk
 * and the other Mesa drivers put it. A discrete card would need the PRIME
 * blit path, which the driver does not have.
 */
static bool
mali_wsi_can_present_on_device(VkPhysicalDevice pdevice, int fd)
{
   drmDevicePtr device;
   if (drmGetDevice2(fd, 0, &device) != 0)
      return false;
   const bool match = device->bustype == DRM_BUS_PLATFORM;
   drmFreeDevice(&device);
   return match;
}

VkResult
mali_wsi_init(struct mali_physical_device *pdev)
{
   const VkAllocationCallbacks *alloc = &pdev->vk.instance->alloc;

   struct wsi_device *wsi = vk_zalloc(alloc, sizeof(*wsi), 8,
                                      VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!wsi)
      return vk_error(pdev, VK_ERROR_OUT_OF_HOST_MEMORY);

   pdev->drm_fd = -1;
   /* The display's DRM node: wsi_common_display enumerates its connectors
    * and planes on this fd and takes DRM master on it while a swapchain is
    * bound to the display. The first card node that opens is the one; a
    * handheld has exactly one. */
   char path[32];
   for (unsigned i = 0; i < 4 && pdev->drm_fd < 0; i++) {
      snprintf(path, sizeof(path), "/dev/dri/card%u", i);
      const int fd = open(path, O_RDWR | O_CLOEXEC);
      if (fd < 0)
         continue;
      /* wsi_common_display only keeps an fd that is already DRM master,
       * and drops master again straight away so another client (a
       * compositor, a VT switch) can take it; it re-takes master while a
       * swapchain is bound. A display on a node somebody else is master of
       * is not available to us until they let go. */
      if (drmSetMaster(fd) != 0)
         mesa_logd("malisx2: %s: not DRM master (%s); VK_KHR_display needs it",
                   path, strerror(errno));
      pdev->drm_fd = fd;
   }

   VkResult result = wsi_device_init(wsi, mali_physical_device_to_handle(pdev),
                                     mali_wsi_proc_addr, alloc, pdev->drm_fd, NULL,
                                     &(struct wsi_device_options){.sw_device = false});
   if (result != VK_SUCCESS) {
      vk_free(alloc, wsi);
      return result;
   }

   /* Linear swapchain images for now: whatever DRM format the display's
    * plane offers, allocated so the kernel can scan it out
    * (mali_device_memory.c exports it as a dma-buf). */
   wsi->supports_modifiers = false;
   wsi->can_present_on_device = mali_wsi_can_present_on_device;
   /* kbase does not put its job fences on an imported dma-buf, so the
    * display cannot see when a frame is finished. Kernels with dma-buf
    * sync-file import get the present's sync file attached to the buffer
    * by wsi_common; older ones (the RK3562's 5.10) need it handed to the
    * flip as IN_FENCE_FD, or KMS scans out a half-rendered or stale image. */
   wsi->no_dma_buf_implicit_sync = true;

   pdev->wsi = wsi;
   pdev->vk.wsi_device = wsi;
   return VK_SUCCESS;
}

void
mali_wsi_finish(struct mali_physical_device *pdev)
{
   pdev->vk.wsi_device = NULL;
   if (pdev->wsi) {
      wsi_device_finish(pdev->wsi, &pdev->vk.instance->alloc);
      vk_free(&pdev->vk.instance->alloc, pdev->wsi);
      pdev->wsi = NULL;
   }
   if (pdev->gbm) {
      gbm_device_destroy(pdev->gbm);
      pdev->gbm = NULL;
   }
   if (pdev->drm_fd >= 0) {
      close(pdev->drm_fd);
      pdev->drm_fd = -1;
   }
}

/*
 * A scanout buffer for a swapchain image: a GBM buffer object on the
 * display's DRM node (the same allocation the desktop stacks use for KMS),
 * handed back as a dma-buf the driver imports into kbase and the display
 * scans out. wsi_common_display asks for the memory this way: an exportable,
 * dedicated allocation for a LINEAR image.
 *
 * *stride is the buffer's row pitch, which the image layout has to agree
 * with: the driver writes the image with the layout it computed when the
 * image was created.
 */
VkResult
mali_wsi_alloc_scanout(struct mali_physical_device *pdev, VkFormat format, unsigned width,
                       unsigned height, uint32_t *stride, uint64_t *size, int *fd)
{
   if (pdev->drm_fd < 0)
      return VK_ERROR_INITIALIZATION_FAILED;
   if (!pdev->gbm) {
      pdev->gbm = gbm_create_device(pdev->drm_fd);
      if (!pdev->gbm) {
         mesa_loge("malisx2: gbm_create_device failed: %s", strerror(errno));
         return VK_ERROR_INITIALIZATION_FAILED;
      }
   }

   /* The display's plane offers B8G8R8A8 (UNORM or SRGB); KMS names both
    * ARGB8888. */
   uint32_t gbm_format;
   switch (format) {
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_B8G8R8A8_SRGB:
      gbm_format = GBM_FORMAT_ARGB8888;
      break;
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SRGB:
      gbm_format = GBM_FORMAT_ABGR8888;
      break;
   case VK_FORMAT_B5G6R5_UNORM_PACK16:
      gbm_format = GBM_FORMAT_RGB565;
      break;
   default:
      mesa_loge("malisx2: no scanout buffer for format %d", format);
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   }

   struct gbm_bo *bo = gbm_bo_create(pdev->gbm, width, height, gbm_format,
                                     GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING);
   if (!bo) {
      mesa_loge("malisx2: gbm_bo_create %ux%u failed: %s", width, height, strerror(errno));
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }

   const int nfd = gbm_bo_get_fd(bo);
   *stride = gbm_bo_get_stride(bo);
   *size = (uint64_t)gbm_bo_get_stride(bo) * gbm_bo_get_height(bo);
   gbm_bo_destroy(bo);
   if (nfd < 0) {
      mesa_loge("malisx2: gbm_bo_get_fd failed: %s", strerror(errno));
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }
   *fd = nfd;
   return VK_SUCCESS;
}

/*
 * Mesa's XML-config option lookups, which wsi_common calls only when it was
 * given a driOptionCache (wsi_device_init's dri_options is NULL here: the
 * driver has no driconf). They exist for the link.
 */
unsigned char
driCheckOption(const driOptionCache *cache, const char *name, driOptionType type)
{
   return 0;
}

unsigned char
driQueryOptionb(const driOptionCache *cache, const char *name)
{
   return 0;
}
