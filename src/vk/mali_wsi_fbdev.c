/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * VK_KHR_display on a Linux framebuffer device, for systems with a kbase
 * GPU and no DRM display node: the Allwinner H700 handhelds' 4.9 BSP
 * kernels have /dev/fb0 and no /dev/dri at all. Without it a Linux
 * application finds no display to present to.
 *
 * mali_wsi_fbdev_init (from mali_wsi_init, after wsi_device_init) takes
 * over when the DRM search found no node and /dev/fb0 answers the
 * framebuffer ioctls:
 *
 *  - the VK_KHR_display queries (the mali_ entry points below, which win
 *    over wsi_common's in the physical-device table) answer one display,
 *    one mode at the framebuffer's resolution and refresh, and one plane;
 *    vkCreateDisplayPlaneSurfaceKHR stays wsi_common's (it only records
 *    the handles in the surface);
 *  - the DISPLAY platform's wsi_interface is replaced by the one below:
 *    surface queries describe the framebuffer, and its swapchain is CPU
 *    images (linear, host-mapped, WSI_IMAGE_TYPE_CPU without a blit), so
 *    the application renders straight into memory the CPU can read.
 *
 * Present: wsi_common has submitted the frame, which signals the image's
 * present fence (swapchain->fences[i]). queue_present only queues the
 * image; a present thread waits for the fence, copies the image into the
 * framebuffer page that is not on screen, and pans to it (FBIOPAN_DISPLAY,
 * which waits for the vertical blank on the H700's sunxi driver; then
 * FBIO_WAITFORVSYNC where a driver has it), so the application renders
 * the next frame meanwhile. The image is free again once it is copied.
 * With one page only it copies into the visible one. The copy is a row
 * memcpy when the swapchain format has the framebuffer's byte order
 * (B8G8R8A8 for the usual 32-bit ARGB layout, which the formats list
 * puts first), else a per-pixel conversion to 32 or 16 bpp. The images
 * are in host-cached memory (fb_create_image_mem), invalidated before
 * the copy reads them.
 *
 * Rotation: a panel mounted turned (the H700 handhelds' portrait panels in
 * a landscape case) says so through the console's rotation,
 * /sys/class/graphics/fbcon/rotate, set by fbcon=rotate:N on the kernel
 * command line or by the kernel from the panel's orientation. The display
 * is reported as the user sees it (width and height swapped for a quarter
 * turn), and the surface's currentTransform is the panel's turn. An
 * application that renders turned itself (preTransform = currentTransform,
 * image extent the framebuffer's own) gets a plain copy; for any other
 * the present copy turns each frame.
 */

#include "mali_vk.h"

#include <vulkan/vk_icd.h>

#include "vk_util.h"
#include "wsi_common.h"
#include "wsi_common_entrypoints.h"
#include "wsi_common_private.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/fb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "c11/threads.h"

#include "util/log.h"
#include "util/os_time.h"

#ifndef FBIO_WAITFORVSYNC
#define FBIO_WAITFORVSYNC _IOW('F', 0x20, uint32_t)
#endif

struct mali_fbdev {
   struct wsi_interface base;          /* the DISPLAY platform's interface */
   struct wsi_interface *drm_iface;    /* wsi_common_display's, restored at finish */
   char path[64];
   struct fb_var_screeninfo var;       /* as read at init */
   uint32_t refresh_mhz;
   /* Clockwise quarter turns from the picture the user sees to the
    * framebuffer (fbcon's rotate: 1 FB_ROTATE_CW, 3 FB_ROTATE_CCW), which
    * is also the surface's currentTransform. Odd: the display is reported
    * with width and height swapped. */
   unsigned rotate;
   /* The display, mode and plane exist once; their handles are the
    * addresses of these two bytes. */
   uint8_t display_tag, mode_tag;
};

static inline VkDisplayKHR
fb_display_handle(struct mali_fbdev *fb)
{
   return (VkDisplayKHR)(uintptr_t)&fb->display_tag;
}

static inline VkDisplayModeKHR
fb_mode_handle(struct mali_fbdev *fb)
{
   return (VkDisplayModeKHR)(uintptr_t)&fb->mode_tag;
}

/* The display as the application sees it: the framebuffer, turned. */
static inline VkExtent2D
fb_extent(const struct mali_fbdev *fb)
{
   return (fb->rotate & 1) ? (VkExtent2D){fb->var.yres, fb->var.xres}
                           : (VkExtent2D){fb->var.xres, fb->var.yres};
}

static VkSurfaceTransformFlagBitsKHR
fb_transform(unsigned quarter_turns)
{
   static const VkSurfaceTransformFlagBitsKHR t[4] = {
      VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR, VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR,
      VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR, VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR};
   return t[quarter_turns & 3];
}

/* The clockwise quarter turns a preTransform names (mirrors are not
 * offered). */
static unsigned
fb_transform_turns(VkSurfaceTransformFlagBitsKHR t)
{
   switch (t) {
   case VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR: return 1;
   case VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR: return 2;
   case VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR: return 3;
   default: return 0;
   }
}

/* The panel's turn from the console's rotation; 0 when there is no
 * console on the framebuffer. */
static unsigned
fb_panel_rotation(void)
{
   FILE *f = fopen("/sys/class/graphics/fbcon/rotate", "r");
   if (!f)
      return 0;
   unsigned r = 0;
   if (fscanf(f, "%u", &r) != 1)
      r = 0;
   fclose(f);
   return r & 3;
}

static struct mali_fbdev *
fb_of(VkPhysicalDevice physicalDevice)
{
   VK_FROM_HANDLE(mali_physical_device, pdev, physicalDevice);
   return pdev->fbdev;
}

static struct mali_fbdev *
fb_of_wsi(struct wsi_device *wsi_device)
{
   VK_FROM_HANDLE(mali_physical_device, pdev, wsi_device->pdevice);
   return pdev->fbdev;
}

/* 32-bit framebuffers with red at bit 16 are B8G8R8A8 in memory. */
static bool
fb_is_bgra(const struct fb_var_screeninfo *v)
{
   return v->bits_per_pixel == 32 && v->red.offset == 16 && v->green.offset == 8 &&
          v->blue.offset == 0;
}

static bool
fb_is_rgba(const struct fb_var_screeninfo *v)
{
   return v->bits_per_pixel == 32 && v->red.offset == 0 && v->green.offset == 8 &&
          v->blue.offset == 16;
}

static uint32_t
fb_refresh_mhz(const struct fb_var_screeninfo *v)
{
   const uint64_t htotal = (uint64_t)v->xres + v->left_margin + v->right_margin + v->hsync_len;
   const uint64_t vtotal = (uint64_t)v->yres + v->upper_margin + v->lower_margin + v->vsync_len;
   if (!v->pixclock || !htotal || !vtotal)
      return 60000;
   /* pixclock is picoseconds per pixel. */
   const uint64_t mhz = 1000000000000000ull / ((uint64_t)v->pixclock * htotal * vtotal);
   return mhz >= 10000 && mhz <= 240000 ? (uint32_t)mhz : 60000;
}

/* ---------------------------------------------------------------------- */
/* VK_KHR_display queries                                                  */

static void
fill_display_props(struct mali_fbdev *fb, VkDisplayPropertiesKHR *p)
{
   *p = (VkDisplayPropertiesKHR){
      .display = fb_display_handle(fb),
      .displayName = fb->path,
      .physicalDimensions = {fb->var.width < 10000 ? fb->var.width : 0,
                             fb->var.height < 10000 ? fb->var.height : 0},
      .physicalResolution = fb_extent(fb),
      .supportedTransforms = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
      .planeReorderPossible = VK_FALSE,
      .persistentContent = VK_FALSE,
   };
}

static void
fill_mode_props(struct mali_fbdev *fb, VkDisplayModePropertiesKHR *p)
{
   *p = (VkDisplayModePropertiesKHR){
      .displayMode = fb_mode_handle(fb),
      .parameters = {.visibleRegion = fb_extent(fb),
                     .refreshRate = fb->refresh_mhz},
   };
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_GetPhysicalDeviceDisplayPropertiesKHR(VkPhysicalDevice physicalDevice,
                                           uint32_t *pPropertyCount,
                                           VkDisplayPropertiesKHR *pProperties)
{
   struct mali_fbdev *fb = fb_of(physicalDevice);
   if (!fb)
      return wsi_GetPhysicalDeviceDisplayPropertiesKHR(physicalDevice, pPropertyCount,
                                                       pProperties);
   VK_OUTARRAY_MAKE_TYPED(VkDisplayPropertiesKHR, out, pProperties, pPropertyCount);
   vk_outarray_append_typed(VkDisplayPropertiesKHR, &out, p) fill_display_props(fb, p);
   return vk_outarray_status(&out);
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_GetPhysicalDeviceDisplayProperties2KHR(VkPhysicalDevice physicalDevice,
                                            uint32_t *pPropertyCount,
                                            VkDisplayProperties2KHR *pProperties)
{
   struct mali_fbdev *fb = fb_of(physicalDevice);
   if (!fb)
      return wsi_GetPhysicalDeviceDisplayProperties2KHR(physicalDevice, pPropertyCount,
                                                        pProperties);
   VK_OUTARRAY_MAKE_TYPED(VkDisplayProperties2KHR, out, pProperties, pPropertyCount);
   vk_outarray_append_typed(VkDisplayProperties2KHR, &out, p)
      fill_display_props(fb, &p->displayProperties);
   return vk_outarray_status(&out);
}

static void
fill_plane_props(struct mali_fbdev *fb, VkDisplayPlanePropertiesKHR *p)
{
   *p = (VkDisplayPlanePropertiesKHR){
      .currentDisplay = fb_display_handle(fb),
      .currentStackIndex = 0,
   };
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_GetPhysicalDeviceDisplayPlanePropertiesKHR(VkPhysicalDevice physicalDevice,
                                                uint32_t *pPropertyCount,
                                                VkDisplayPlanePropertiesKHR *pProperties)
{
   struct mali_fbdev *fb = fb_of(physicalDevice);
   if (!fb)
      return wsi_GetPhysicalDeviceDisplayPlanePropertiesKHR(physicalDevice, pPropertyCount,
                                                            pProperties);
   VK_OUTARRAY_MAKE_TYPED(VkDisplayPlanePropertiesKHR, out, pProperties, pPropertyCount);
   vk_outarray_append_typed(VkDisplayPlanePropertiesKHR, &out, p) fill_plane_props(fb, p);
   return vk_outarray_status(&out);
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_GetPhysicalDeviceDisplayPlaneProperties2KHR(VkPhysicalDevice physicalDevice,
                                                 uint32_t *pPropertyCount,
                                                 VkDisplayPlaneProperties2KHR *pProperties)
{
   struct mali_fbdev *fb = fb_of(physicalDevice);
   if (!fb)
      return wsi_GetPhysicalDeviceDisplayPlaneProperties2KHR(physicalDevice, pPropertyCount,
                                                             pProperties);
   VK_OUTARRAY_MAKE_TYPED(VkDisplayPlaneProperties2KHR, out, pProperties, pPropertyCount);
   vk_outarray_append_typed(VkDisplayPlaneProperties2KHR, &out, p)
      fill_plane_props(fb, &p->displayPlaneProperties);
   return vk_outarray_status(&out);
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_GetDisplayPlaneSupportedDisplaysKHR(VkPhysicalDevice physicalDevice, uint32_t planeIndex,
                                         uint32_t *pDisplayCount, VkDisplayKHR *pDisplays)
{
   struct mali_fbdev *fb = fb_of(physicalDevice);
   if (!fb)
      return wsi_GetDisplayPlaneSupportedDisplaysKHR(physicalDevice, planeIndex,
                                                     pDisplayCount, pDisplays);
   VK_OUTARRAY_MAKE_TYPED(VkDisplayKHR, out, pDisplays, pDisplayCount);
   if (planeIndex == 0) {
      vk_outarray_append_typed(VkDisplayKHR, &out, d) *d = fb_display_handle(fb);
   }
   return vk_outarray_status(&out);
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_GetDisplayModePropertiesKHR(VkPhysicalDevice physicalDevice, VkDisplayKHR display,
                                 uint32_t *pPropertyCount,
                                 VkDisplayModePropertiesKHR *pProperties)
{
   struct mali_fbdev *fb = fb_of(physicalDevice);
   if (!fb)
      return wsi_GetDisplayModePropertiesKHR(physicalDevice, display, pPropertyCount,
                                             pProperties);
   VK_OUTARRAY_MAKE_TYPED(VkDisplayModePropertiesKHR, out, pProperties, pPropertyCount);
   if (display == fb_display_handle(fb)) {
      vk_outarray_append_typed(VkDisplayModePropertiesKHR, &out, p) fill_mode_props(fb, p);
   }
   return vk_outarray_status(&out);
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_GetDisplayModeProperties2KHR(VkPhysicalDevice physicalDevice, VkDisplayKHR display,
                                  uint32_t *pPropertyCount,
                                  VkDisplayModeProperties2KHR *pProperties)
{
   struct mali_fbdev *fb = fb_of(physicalDevice);
   if (!fb)
      return wsi_GetDisplayModeProperties2KHR(physicalDevice, display, pPropertyCount,
                                              pProperties);
   VK_OUTARRAY_MAKE_TYPED(VkDisplayModeProperties2KHR, out, pProperties, pPropertyCount);
   if (display == fb_display_handle(fb)) {
      vk_outarray_append_typed(VkDisplayModeProperties2KHR, &out, p)
         fill_mode_props(fb, &p->displayModeProperties);
   }
   return vk_outarray_status(&out);
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_CreateDisplayModeKHR(VkPhysicalDevice physicalDevice, VkDisplayKHR display,
                          const VkDisplayModeCreateInfoKHR *pCreateInfo,
                          const VkAllocationCallbacks *pAllocator, VkDisplayModeKHR *pMode)
{
   struct mali_fbdev *fb = fb_of(physicalDevice);
   if (!fb)
      return wsi_CreateDisplayModeKHR(physicalDevice, display, pCreateInfo, pAllocator, pMode);
   /* The framebuffer's mode is the panel's; asking for that one is fine. */
   if (pCreateInfo->parameters.visibleRegion.width == fb_extent(fb).width &&
       pCreateInfo->parameters.visibleRegion.height == fb_extent(fb).height) {
      *pMode = fb_mode_handle(fb);
      return VK_SUCCESS;
   }
   return VK_ERROR_INITIALIZATION_FAILED;
}

static void
fill_plane_caps(struct mali_fbdev *fb, VkDisplayPlaneCapabilitiesKHR *c)
{
   const VkExtent2D e = fb_extent(fb);
   *c = (VkDisplayPlaneCapabilitiesKHR){
      .supportedAlpha = VK_DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR,
      .minSrcPosition = {0, 0},
      .maxSrcPosition = {0, 0},
      .minSrcExtent = e,
      .maxSrcExtent = e,
      .minDstPosition = {0, 0},
      .maxDstPosition = {0, 0},
      .minDstExtent = e,
      .maxDstExtent = e,
   };
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_GetDisplayPlaneCapabilitiesKHR(VkPhysicalDevice physicalDevice, VkDisplayModeKHR mode,
                                    uint32_t planeIndex,
                                    VkDisplayPlaneCapabilitiesKHR *pCapabilities)
{
   struct mali_fbdev *fb = fb_of(physicalDevice);
   if (!fb)
      return wsi_GetDisplayPlaneCapabilitiesKHR(physicalDevice, mode, planeIndex,
                                                pCapabilities);
   fill_plane_caps(fb, pCapabilities);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_GetDisplayPlaneCapabilities2KHR(VkPhysicalDevice physicalDevice,
                                     const VkDisplayPlaneInfo2KHR *pDisplayPlaneInfo,
                                     VkDisplayPlaneCapabilities2KHR *pCapabilities)
{
   struct mali_fbdev *fb = fb_of(physicalDevice);
   if (!fb)
      return wsi_GetDisplayPlaneCapabilities2KHR(physicalDevice, pDisplayPlaneInfo,
                                                 pCapabilities);
   fill_plane_caps(fb, &pCapabilities->capabilities);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_ReleaseDisplayEXT(VkPhysicalDevice physicalDevice, VkDisplayKHR display)
{
   struct mali_fbdev *fb = fb_of(physicalDevice);
   if (!fb)
      return wsi_ReleaseDisplayEXT(physicalDevice, display);
   return VK_SUCCESS;
}

/* ---------------------------------------------------------------------- */
/* Surface queries (the DISPLAY platform's wsi_interface)                  */

static VkResult
fb_surface_get_support(VkIcdSurfaceBase *surface, struct wsi_device *wsi_device,
                       uint32_t queueFamilyIndex, VkBool32 *pSupported)
{
   *pSupported = VK_TRUE;
   return VK_SUCCESS;
}

static VkResult
fb_surface_get_capabilities2(VkIcdSurfaceBase *surface, struct wsi_device *wsi_device,
                             const void *info_next, VkSurfaceCapabilities2KHR *caps)
{
   struct mali_fbdev *fb = fb_of_wsi(wsi_device);
   const VkExtent2D e = fb_extent(fb);
   VkSurfaceCapabilitiesKHR *c = &caps->surfaceCapabilities;
   /* Two images let the application render one while the other is
    * copied; the copy itself is synchronous, so more only add latency. */
   c->minImageCount = 2;
   c->maxImageCount = 0;
   c->currentExtent = e;
   c->minImageExtent = e;
   c->maxImageExtent = e;
   if (fb->rotate & 1) {
      /* An application that renders turned itself makes its images the
       * framebuffer's own size, e swapped. */
      const uint32_t lo = MIN2(e.width, e.height), hi = MAX2(e.width, e.height);
      c->minImageExtent = (VkExtent2D){lo, lo};
      c->maxImageExtent = (VkExtent2D){hi, hi};
   }
   c->maxImageArrayLayers = 1;
   c->currentTransform = fb_transform(fb->rotate);
   c->supportedTransforms = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR | c->currentTransform;
   c->supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
   c->supportedUsageFlags = wsi_caps_get_image_usage();

   vk_foreach_struct(sType, ext, caps->pNext) {
      switch (sType) {
      case VK_STRUCTURE_TYPE_SURFACE_PROTECTED_CAPABILITIES_KHR: {
         VkSurfaceProtectedCapabilitiesKHR *protected = (void *)ext;
         protected->supportsProtected = VK_FALSE;
         break;
      }
      case VK_STRUCTURE_TYPE_SURFACE_PRESENT_SCALING_CAPABILITIES_KHR: {
         VkSurfacePresentScalingCapabilitiesKHR *scaling = (void *)ext;
         scaling->supportedPresentScaling = 0;
         scaling->supportedPresentGravityX = 0;
         scaling->supportedPresentGravityY = 0;
         scaling->minScaledImageExtent = e;
         scaling->maxScaledImageExtent = e;
         break;
      }
      case VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_COMPATIBILITY_KHR: {
         VkSurfacePresentModeCompatibilityKHR *compat = (void *)ext;
         if (compat->pPresentModes && compat->presentModeCount)
            compat->pPresentModes[0] = VK_PRESENT_MODE_FIFO_KHR;
         compat->presentModeCount = 1;
         break;
      }
      default:
         break;
      }
   }
   return VK_SUCCESS;
}

/* The swapchain formats, the framebuffer's own byte order first. */
static unsigned
fb_formats(struct mali_fbdev *fb, VkFormat out[4])
{
   if (fb_is_rgba(&fb->var)) {
      out[0] = VK_FORMAT_R8G8B8A8_UNORM;
      out[1] = VK_FORMAT_R8G8B8A8_SRGB;
      out[2] = VK_FORMAT_B8G8R8A8_UNORM;
      out[3] = VK_FORMAT_B8G8R8A8_SRGB;
   } else {
      out[0] = VK_FORMAT_B8G8R8A8_UNORM;
      out[1] = VK_FORMAT_B8G8R8A8_SRGB;
      out[2] = VK_FORMAT_R8G8B8A8_UNORM;
      out[3] = VK_FORMAT_R8G8B8A8_SRGB;
   }
   return 4;
}

static VkResult
fb_surface_get_formats(VkIcdSurfaceBase *surface, struct wsi_device *wsi_device,
                       uint32_t *pSurfaceFormatCount, VkSurfaceFormatKHR *pSurfaceFormats)
{
   VkFormat f[4];
   const unsigned n = fb_formats(fb_of_wsi(wsi_device), f);
   VK_OUTARRAY_MAKE_TYPED(VkSurfaceFormatKHR, out, pSurfaceFormats, pSurfaceFormatCount);
   for (unsigned i = 0; i < n; i++) {
      vk_outarray_append_typed(VkSurfaceFormatKHR, &out, o) {
         o->format = f[i];
         o->colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
      }
   }
   return vk_outarray_status(&out);
}

static VkResult
fb_surface_get_formats2(VkIcdSurfaceBase *surface, struct wsi_device *wsi_device,
                        const void *info_next, uint32_t *pSurfaceFormatCount,
                        VkSurfaceFormat2KHR *pSurfaceFormats)
{
   VkFormat f[4];
   const unsigned n = fb_formats(fb_of_wsi(wsi_device), f);
   VK_OUTARRAY_MAKE_TYPED(VkSurfaceFormat2KHR, out, pSurfaceFormats, pSurfaceFormatCount);
   for (unsigned i = 0; i < n; i++) {
      vk_outarray_append_typed(VkSurfaceFormat2KHR, &out, o) {
         o->surfaceFormat.format = f[i];
         o->surfaceFormat.colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
      }
   }
   return vk_outarray_status(&out);
}

static const VkPresentModeKHR fb_present_modes[] = {
   VK_PRESENT_MODE_FIFO_KHR,      /* pan to the new page, wait for vsync */
   VK_PRESENT_MODE_IMMEDIATE_KHR, /* pan, no vsync wait */
};

static VkResult
fb_surface_get_present_modes(VkIcdSurfaceBase *surface, struct wsi_device *wsi_device,
                             uint32_t *pPresentModeCount, VkPresentModeKHR *pPresentModes)
{
   VK_OUTARRAY_MAKE_TYPED(VkPresentModeKHR, out, pPresentModes, pPresentModeCount);
   for (unsigned i = 0; i < ARRAY_SIZE(fb_present_modes); i++) {
      vk_outarray_append_typed(VkPresentModeKHR, &out, m) *m = fb_present_modes[i];
   }
   return vk_outarray_status(&out);
}

static VkResult
fb_surface_get_present_rectangles(VkIcdSurfaceBase *surface, struct wsi_device *wsi_device,
                                  uint32_t *pRectCount, VkRect2D *pRects)
{
   struct mali_fbdev *fb = fb_of_wsi(wsi_device);
   VK_OUTARRAY_MAKE_TYPED(VkRect2D, out, pRects, pRectCount);
   vk_outarray_append_typed(VkRect2D, &out, r) {
      *r = (VkRect2D){.offset = {0, 0}, .extent = fb_extent(fb)};
   }
   return vk_outarray_status(&out);
}

/* ---------------------------------------------------------------------- */
/* Swapchain                                                               */

struct mali_fb_image {
   struct wsi_image base;
   /* Acquired by the application, or presented and not yet copied to the
    * framebuffer. Under mali_fb_swapchain::lock. */
   bool busy;
};

struct mali_fb_swapchain {
   struct wsi_swapchain base;
   int fd;
   uint8_t *map;
   size_t map_size;
   struct fb_var_screeninfo var; /* as of swapchain creation */
   struct fb_fix_screeninfo fix;
   unsigned pages;               /* 2 when yres_virtual holds two screens */
   unsigned shown;               /* page on screen */
   bool same_order;              /* swapchain format = framebuffer byte order */
   bool image_rgba;              /* R8G8B8A8 image (else B8G8R8A8) */
   bool swap_rb;                 /* 32-bit fb in the image's order with red and blue swapped */
   /* Clockwise quarter turns the copy applies: the panel's, less what
    * the application's preTransform did. */
   unsigned rotate;
   uint32_t *rot_rows;           /* rotated copy: FB_ROT_ROWS framebuffer rows, converted */
   PFN_vkInvalidateMappedMemoryRanges invalidate;
   /* The present thread: queue_present queues the image, the thread waits
    * for its fence, copies it and pans, so the application renders the
    * next frame meanwhile. queue[] holds image indices in present order. */
   mtx_t lock;
   cnd_t cond;
   thrd_t thread;
   bool thread_started, quit;
   unsigned queue_head, queue_len;
   VkResult error;               /* sticky: the first failure in the thread */
   uint32_t *queue;
   struct mali_fb_image images[0];
};

static struct wsi_image *
fb_swapchain_get_wsi_image(struct wsi_swapchain *wsi_chain, uint32_t image_index)
{
   struct mali_fb_swapchain *chain = (struct mali_fb_swapchain *)wsi_chain;
   return &chain->images[image_index].base;
}

static VkResult
fb_swapchain_release_images(struct wsi_swapchain *wsi_chain, uint32_t count,
                            const uint32_t *indices)
{
   struct mali_fb_swapchain *chain = (struct mali_fb_swapchain *)wsi_chain;
   mtx_lock(&chain->lock);
   for (uint32_t i = 0; i < count; i++)
      chain->images[indices[i]].busy = false;
   cnd_broadcast(&chain->cond);
   mtx_unlock(&chain->lock);
   return VK_SUCCESS;
}

static VkResult
fb_swapchain_acquire_next_image(struct wsi_swapchain *wsi_chain,
                                const VkAcquireNextImageInfoKHR *info, uint32_t *image_index)
{
   struct mali_fb_swapchain *chain = (struct mali_fb_swapchain *)wsi_chain;
   /* An image is free once the present thread has copied it. */
   const uint64_t deadline = info->timeout == UINT64_MAX
                                ? UINT64_MAX
                                : os_time_get_absolute_timeout(info->timeout);
   mtx_lock(&chain->lock);
   VkResult r = VK_SUCCESS;
   for (;;) {
      if (chain->error != VK_SUCCESS) {
         r = chain->error;
         break;
      }
      uint32_t i = 0;
      while (i < chain->base.image_count && chain->images[i].busy)
         i++;
      if (i < chain->base.image_count) {
         chain->images[i].busy = true;
         *image_index = i;
         break;
      }
      if (info->timeout == 0) {
         r = VK_NOT_READY;
         break;
      }
      if (deadline == UINT64_MAX) {
         cnd_wait(&chain->cond, &chain->lock);
      } else {
         const uint64_t now = os_time_get_nano();
         if (now >= deadline) {
            r = VK_TIMEOUT;
            break;
         }
         struct timespec ts;
         timespec_get(&ts, TIME_UTC);
         const uint64_t abs = (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec + (deadline - now);
         ts.tv_sec = abs / 1000000000ull;
         ts.tv_nsec = abs % 1000000000ull;
         cnd_timedwait(&chain->cond, &chain->lock, &ts);
      }
   }
   mtx_unlock(&chain->lock);
   return r;
}

/* One image pixel (R8G8B8A8 or B8G8R8A8 in memory) as a framebuffer
 * pixel: the low 16 bits of the result for a 16 bpp framebuffer. */
static inline uint32_t
fb_pixel(const struct mali_fb_swapchain *chain, uint32_t p)
{
   const struct fb_var_screeninfo *v = &chain->var;
   if (chain->swap_rb) {
      const uint32_t a = v->transp.length == 8 ? 0xffu << v->transp.offset : 0;
      return (p & 0x0000ff00u) | (p >> 16 & 0xffu) | (p & 0xffu) << 16 | a;
   }
   uint32_t r = p & 0xff, g = (p >> 8) & 0xff, b = (p >> 16) & 0xff;
   if (!chain->image_rgba) {
      const uint32_t t = r;
      r = b;
      b = t;
   }
   uint32_t out = (r >> (8 - v->red.length)) << v->red.offset |
                  (g >> (8 - v->green.length)) << v->green.offset |
                  (b >> (8 - v->blue.length)) << v->blue.offset;
   if (v->bits_per_pixel == 32 && v->transp.length)
      out |= ((1u << v->transp.length) - 1) << v->transp.offset;
   return out;
}

static void fb_copy_rotated(struct mali_fb_swapchain *chain, const struct wsi_image *img,
                            uint8_t *dst);

static void
fb_copy(struct mali_fb_swapchain *chain, const struct wsi_image *img, uint8_t *dst)
{
   if (chain->rotate) {
      fb_copy_rotated(chain, img, dst);
      return;
   }
   const unsigned w = MIN2(chain->var.xres, chain->base.image_info.create.extent.width);
   const unsigned h = MIN2(chain->var.yres, chain->base.image_info.create.extent.height);
   const uint8_t *src = img->cpu_map;
   const uint32_t sp = img->row_pitches[0], dp = chain->fix.line_length;
   const struct fb_var_screeninfo *v = &chain->var;

   if (chain->same_order) {
      for (unsigned y = 0; y < h; y++)
         memcpy(dst + (size_t)y * dp, src + (size_t)y * sp, (size_t)w * 4);
      return;
   }

   if (chain->swap_rb) {
      /* An R8G8B8A8 image on a B8G8R8A8 framebuffer or the reverse: swap
       * red and blue (alpha opaque where the framebuffer has it) with one
       * 32-bit swizzle per pixel, which the compiler vectorizes. */
      for (unsigned y = 0; y < h; y++) {
         const uint32_t *s = (const uint32_t *)(src + (size_t)y * sp);
         uint32_t *d = (uint32_t *)(dst + (size_t)y * dp);
         for (unsigned x = 0; x < w; x++)
            d[x] = fb_pixel(chain, s[x]);
      }
      return;
   }

   /* Read whole rows first: the image may be write-combined memory, where
    * byte reads are very slow. */
   uint32_t row[4096];
   const unsigned rw = MIN2(w, ARRAY_SIZE(row));
   for (unsigned y = 0; y < h; y++) {
      memcpy(row, src + (size_t)y * sp, (size_t)rw * 4);
      uint8_t *d = dst + (size_t)y * dp;
      for (unsigned x = 0; x < rw; x++) {
         if (v->bits_per_pixel == 32)
            ((uint32_t *)d)[x] = fb_pixel(chain, row[x]);
         else
            ((uint16_t *)d)[x] = (uint16_t)fb_pixel(chain, row[x]);
      }
   }
}

/* Rows of the framebuffer gathered per pass of the rotated copy: for a
 * quarter turn, a framebuffer column is an image row, so each pass reads
 * FB_ROT_ROWS adjacent image pixels at a time. */
#define FB_ROT_ROWS 8

/* The copy that turns the picture chain->rotate quarter turns clockwise
 * onto the framebuffer: for an image W x H, framebuffer pixel (fx, fy)
 * shows image pixel (fy, H-1-fx) for one turn, (W-1-fx, H-1-fy) for two
 * and (W-1-fy, fx) for three: as an address, a fixed start plus fx and fy
 * steps. The rows are built in cached memory and written to the
 * framebuffer whole. */
static void
fb_copy_rotated(struct mali_fb_swapchain *chain, const struct wsi_image *img, uint8_t *dst)
{
   const struct fb_var_screeninfo *v = &chain->var;
   const unsigned iw = chain->base.image_info.create.extent.width;
   const unsigned ih = chain->base.image_info.create.extent.height;
   const unsigned fw = MIN2(v->xres, (chain->rotate & 1) ? ih : iw);
   const unsigned fh = MIN2(v->yres, (chain->rotate & 1) ? iw : ih);
   const uint8_t *src = img->cpu_map;
   const ptrdiff_t sp = img->row_pitches[0], dp = chain->fix.line_length;
   ptrdiff_t start, step_x, step_y;
   switch (chain->rotate) {
   case 1: start = (ptrdiff_t)(ih - 1) * sp; step_x = -sp; step_y = 4; break;
   case 2: start = (ptrdiff_t)(ih - 1) * sp + (ptrdiff_t)(iw - 1) * 4; step_x = -4; step_y = -sp; break;
   default: start = (ptrdiff_t)(iw - 1) * 4; step_x = sp; step_y = -4; break;
   }

   for (unsigned y0 = 0; y0 < fh; y0 += FB_ROT_ROWS) {
      const unsigned n = MIN2(FB_ROT_ROWS, fh - y0);
      uint32_t *rows = chain->rot_rows;
      /* Read along image rows: across the rows block for a quarter turn,
       * along each row (backwards) for a half turn. */
      if (chain->rotate == 2) {
         for (unsigned k = 0; k < n; k++) {
            const uint8_t *s = src + start + (ptrdiff_t)(y0 + k) * step_y;
            for (unsigned x = 0; x < fw; x++)
               rows[k * fw + x] = *(const uint32_t *)(s + (ptrdiff_t)x * step_x);
         }
      } else {
         for (unsigned x = 0; x < fw; x++) {
            const uint8_t *s = src + start + (ptrdiff_t)x * step_x + (ptrdiff_t)y0 * step_y;
            for (unsigned k = 0; k < n; k++)
               rows[k * fw + x] = *(const uint32_t *)(s + (ptrdiff_t)k * step_y);
         }
      }
      for (unsigned k = 0; k < n; k++) {
         uint32_t *r = rows + k * fw;
         uint8_t *d = dst + (size_t)(y0 + k) * dp;
         if (chain->same_order) {
            memcpy(d, r, (size_t)fw * 4);
         } else if (v->bits_per_pixel == 32) {
            for (unsigned x = 0; x < fw; x++)
               r[x] = fb_pixel(chain, r[x]);
            memcpy(d, r, (size_t)fw * 4);
         } else {
            for (unsigned x = 0; x < fw; x++)
               ((uint16_t *)d)[x] = (uint16_t)fb_pixel(chain, r[x]);
         }
      }
   }
}

/* The present thread's work for one image: wait for the frame, copy it
 * into the page not on screen, pan to it. */
static VkResult
fb_present_one(struct mali_fb_swapchain *chain, uint32_t image_index)
{
   struct mali_fb_image *image = &chain->images[image_index];
   const struct wsi_device *wsi = chain->base.wsi;

   /* The frame is finished when the present's fence is. wsi_common waits
    * on and resets this fence again only at the image's next present,
    * which cannot come before the image is free again below. */
   VkResult r = wsi->WaitForFences(chain->base.device, 1, &chain->base.fences[image_index],
                                   VK_TRUE, UINT64_MAX);
   if (r != VK_SUCCESS)
      return r;

   /* Cached, not coherent memory (fb_create_image_mem): drop what the
    * CPU may hold of it before reading the GPU's writes. */
   const VkMappedMemoryRange range = {.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                                      .memory = image->base.memory, .size = VK_WHOLE_SIZE};
   chain->invalidate(chain->base.device, 1, &range);
   const unsigned page = chain->pages > 1 ? chain->shown ^ 1 : 0;
   const size_t page_bytes = (size_t)chain->var.yres * chain->fix.line_length;
   fb_copy(chain, &image->base, chain->map + page * page_bytes);
   mtx_lock(&chain->lock);
   image->busy = false;
   cnd_broadcast(&chain->cond);
   mtx_unlock(&chain->lock);

   if (chain->pages > 1) {
      struct fb_var_screeninfo v = chain->var;
      v.xoffset = 0;
      v.yoffset = page * chain->var.yres;
      if (ioctl(chain->fd, FBIOPAN_DISPLAY, &v) == 0)
         chain->shown = page;
      else
         mesa_logw("malisx2: FBIOPAN_DISPLAY to page %u failed: %s", page, strerror(errno));
   }
   if (chain->base.present_mode == VK_PRESENT_MODE_FIFO_KHR) {
      uint32_t crtc = 0;
      ioctl(chain->fd, FBIO_WAITFORVSYNC, &crtc);
   }
   return VK_SUCCESS;
}

static int
fb_present_thread(void *data)
{
   struct mali_fb_swapchain *chain = data;
   mtx_lock(&chain->lock);
   for (;;) {
      while (!chain->quit && !chain->queue_len)
         cnd_wait(&chain->cond, &chain->lock);
      if (!chain->queue_len)
         break; /* quit, and nothing left to show */
      const uint32_t index = chain->queue[chain->queue_head];
      chain->queue_head = (chain->queue_head + 1) % chain->base.image_count;
      chain->queue_len--;
      mtx_unlock(&chain->lock);
      const VkResult r = fb_present_one(chain, index);
      mtx_lock(&chain->lock);
      if (r != VK_SUCCESS) {
         chain->images[index].busy = false;
         if (chain->error == VK_SUCCESS)
            chain->error = r;
         cnd_broadcast(&chain->cond);
      }
   }
   mtx_unlock(&chain->lock);
   return 0;
}

static VkResult
fb_swapchain_queue_present(struct wsi_swapchain *wsi_chain, uint32_t image_index,
                           uint64_t present_id, const VkPresentRegionKHR *damage)
{
   struct mali_fb_swapchain *chain = (struct mali_fb_swapchain *)wsi_chain;
   /* The image stays busy until the present thread has copied it. Every
    * image is in the queue at most once, so it never overflows. */
   mtx_lock(&chain->lock);
   const VkResult error = chain->error;
   if (error == VK_SUCCESS) {
      chain->queue[(chain->queue_head + chain->queue_len) % chain->base.image_count] =
         image_index;
      chain->queue_len++;
      cnd_broadcast(&chain->cond);
   } else {
      chain->images[image_index].busy = false;
   }
   mtx_unlock(&chain->lock);
   return error;
}

/* Swapchain image memory: linear (wants_linear), dedicated, CPU-mapped,
 * as wsi_common's CPU images, but from a host-cached memory type: the
 * copy reads every byte of it, and reads from write-combined memory run
 * at a few tens of MB/s on these CPUs. fb_present_one invalidates before
 * it reads. */
static VkResult
fb_create_image_mem(const struct wsi_swapchain *chain, const struct wsi_image_info *info,
                    struct wsi_image *image)
{
   const struct wsi_device *wsi = chain->wsi;
   VkMemoryRequirements reqs;
   wsi->GetImageMemoryRequirements(chain->device, image->image, &reqs);
   VkSubresourceLayout layout;
   wsi->GetImageSubresourceLayout(chain->device, image->image,
                                  &(VkImageSubresource){
                                     .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                  },
                                  &layout);

   uint32_t type = UINT32_MAX, fallback = UINT32_MAX;
   for (uint32_t i = 0; i < wsi->memory_props.memoryTypeCount; i++) {
      if (!(reqs.memoryTypeBits & (1u << i)))
         continue;
      const VkMemoryPropertyFlags f = wsi->memory_props.memoryTypes[i].propertyFlags;
      if (!(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
         continue;
      if ((f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) && type == UINT32_MAX)
         type = i;
      if (fallback == UINT32_MAX)
         fallback = i;
   }
   if (type == UINT32_MAX)
      type = fallback;
   if (type == UINT32_MAX)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   const VkMemoryDedicatedAllocateInfo dedicated = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
      .image = image->image,
   };
   const VkMemoryAllocateInfo mi = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &dedicated,
      .allocationSize = reqs.size,
      .memoryTypeIndex = type,
   };
   VkResult result = wsi->AllocateMemory(chain->device, &mi, &chain->alloc, &image->memory);
   if (result != VK_SUCCESS)
      return result;
   result = wsi->MapMemory(chain->device, image->memory, 0, VK_WHOLE_SIZE, 0, &image->cpu_map);
   if (result != VK_SUCCESS)
      return result;
   image->num_planes = 1;
   image->sizes[0] = reqs.size;
   image->row_pitches[0] = layout.rowPitch;
   image->offsets[0] = layout.offset;
   return VK_SUCCESS;
}

static VkResult
fb_swapchain_wait_for_present(struct wsi_swapchain *wsi_chain, uint64_t waitValue,
                              uint64_t timeout)
{
   return wsi_swapchain_wait_for_present_semaphore(wsi_chain, waitValue, timeout);
}

static VkResult
fb_swapchain_destroy(struct wsi_swapchain *wsi_chain, const VkAllocationCallbacks *pAllocator)
{
   struct mali_fb_swapchain *chain = (struct mali_fb_swapchain *)wsi_chain;
   /* The thread shows what is still queued, then stops. */
   if (chain->thread_started) {
      mtx_lock(&chain->lock);
      chain->quit = true;
      cnd_broadcast(&chain->cond);
      mtx_unlock(&chain->lock);
      thrd_join(chain->thread, NULL);
   }
   for (uint32_t i = 0; i < chain->base.image_count; i++) {
      if (chain->images[i].base.image != VK_NULL_HANDLE)
         wsi_destroy_image(&chain->base, &chain->images[i].base);
   }
   munmap(chain->map, chain->map_size);
   close(chain->fd);
   cnd_destroy(&chain->cond);
   mtx_destroy(&chain->lock);
   wsi_swapchain_finish(&chain->base);
   vk_free(pAllocator, chain->queue);
   vk_free(pAllocator, chain->rot_rows);
   vk_free(pAllocator, chain);
   return VK_SUCCESS;
}

static VkResult
fb_surface_create_swapchain(VkIcdSurfaceBase *icd_surface, VkDevice device,
                            struct wsi_device *wsi_device,
                            const VkSwapchainCreateInfoKHR *pCreateInfo,
                            const VkAllocationCallbacks *pAllocator,
                            struct wsi_swapchain **swapchain_out)
{
   struct mali_fbdev *fb = fb_of_wsi(wsi_device);
   const uint32_t num_images = MAX2(pCreateInfo->minImageCount, 2);
   const size_t size = sizeof(struct mali_fb_swapchain) + num_images * sizeof(struct mali_fb_image);
   struct mali_fb_swapchain *chain =
      vk_zalloc(pAllocator, size, 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!chain)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   chain->fd = -1;

   chain->fd = open(fb->path, O_RDWR | O_CLOEXEC);
   if (chain->fd < 0 || ioctl(chain->fd, FBIOGET_VSCREENINFO, &chain->var) ||
       ioctl(chain->fd, FBIOGET_FSCREENINFO, &chain->fix)) {
      mesa_loge("malisx2: %s: %s", fb->path, strerror(errno));
      if (chain->fd >= 0)
         close(chain->fd);
      vk_free(pAllocator, chain);
      return VK_ERROR_SURFACE_LOST_KHR;
   }
   chain->map_size = chain->fix.smem_len;
   chain->map = mmap(NULL, chain->map_size, PROT_READ | PROT_WRITE, MAP_SHARED, chain->fd, 0);
   if (chain->map == MAP_FAILED) {
      mesa_loge("malisx2: mmap of %s failed: %s", fb->path, strerror(errno));
      close(chain->fd);
      vk_free(pAllocator, chain);
      return VK_ERROR_SURFACE_LOST_KHR;
   }
   const size_t page_bytes = (size_t)chain->var.yres * chain->fix.line_length;
   chain->pages = chain->var.yres_virtual >= 2 * chain->var.yres &&
                        chain->map_size >= 2 * page_bytes
                     ? 2
                     : 1;
   chain->shown = chain->pages > 1 && chain->var.yoffset >= chain->var.yres ? 1 : 0;

   const VkFormat f = pCreateInfo->imageFormat;
   const bool image_bgra = f == VK_FORMAT_B8G8R8A8_UNORM || f == VK_FORMAT_B8G8R8A8_SRGB;
   chain->same_order = image_bgra ? fb_is_bgra(&chain->var) : fb_is_rgba(&chain->var);
   chain->image_rgba = !image_bgra;
   chain->swap_rb = image_bgra ? fb_is_rgba(&chain->var) : fb_is_bgra(&chain->var);
   chain->rotate = (fb->rotate - fb_transform_turns(pCreateInfo->preTransform)) & 3;
   if (chain->var.bits_per_pixel != 32 && chain->var.bits_per_pixel != 16) {
      mesa_loge("malisx2: %s is %u bpp; only 16 and 32 are supported", fb->path,
                chain->var.bits_per_pixel);
      munmap(chain->map, chain->map_size);
      close(chain->fd);
      vk_free(pAllocator, chain);
      return VK_ERROR_SURFACE_LOST_KHR;
   }

   chain->queue = vk_zalloc(pAllocator, num_images * sizeof(uint32_t), 8,
                            VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (chain->rotate)
      chain->rot_rows = vk_alloc(pAllocator, (size_t)FB_ROT_ROWS * chain->var.xres * 4, 8,
                                 VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   struct wsi_cpu_image_params cpu_params = {
      .base.image_type = WSI_IMAGE_TYPE_CPU,
   };
   VkResult result = chain->queue && (!chain->rotate || chain->rot_rows)
                        ? wsi_swapchain_init(wsi_device, &chain->base, device,
                                             pCreateInfo, &cpu_params.base, pAllocator)
                        : VK_ERROR_OUT_OF_HOST_MEMORY;
   if (result != VK_SUCCESS) {
      munmap(chain->map, chain->map_size);
      close(chain->fd);
      vk_free(pAllocator, chain->queue);
      vk_free(pAllocator, chain->rot_rows);
      vk_free(pAllocator, chain);
      return result;
   }
   mtx_init(&chain->lock, mtx_plain);
   cnd_init(&chain->cond);
   VK_FROM_HANDLE(vk_device, vk_dev, device);
   chain->invalidate = vk_dev->dispatch_table.InvalidateMappedMemoryRanges;
   chain->base.image_info.create_mem = fb_create_image_mem;

   chain->base.destroy = fb_swapchain_destroy;
   chain->base.get_wsi_image = fb_swapchain_get_wsi_image;
   chain->base.acquire_next_image = fb_swapchain_acquire_next_image;
   chain->base.release_images = fb_swapchain_release_images;
   chain->base.queue_present = fb_swapchain_queue_present;
   chain->base.wait_for_present = fb_swapchain_wait_for_present;
   chain->base.wait_for_present2 = fb_swapchain_wait_for_present;
   chain->base.present_mode = wsi_swapchain_get_present_mode(wsi_device, pCreateInfo);
   chain->base.image_count = num_images;

   for (uint32_t i = 0; i < num_images; i++) {
      result = wsi_create_image(&chain->base, &chain->base.image_info, &chain->images[i].base);
      if (result != VK_SUCCESS) {
         fb_swapchain_destroy(&chain->base, pAllocator);
         return result;
      }
   }
   if (thrd_create(&chain->thread, fb_present_thread, chain) != thrd_success) {
      fb_swapchain_destroy(&chain->base, pAllocator);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   chain->thread_started = true;

   mesa_logi("malisx2 fbdev swapchain on %s: %ux%u %u bpp, %u page%s, %u images, %s%s%s",
             fb->path, chain->var.xres, chain->var.yres, chain->var.bits_per_pixel,
             chain->pages, chain->pages > 1 ? "s" : "", num_images,
             chain->rotate ? "turned " : "",
             chain->same_order ? "row copy" : chain->swap_rb ? "red/blue swap copy" : "converting copy",
             chain->base.present_mode == VK_PRESENT_MODE_FIFO_KHR ? ", vsync" : "");
   *swapchain_out = &chain->base;
   return VK_SUCCESS;
}

/* ---------------------------------------------------------------------- */
/* Set-up                                                                  */

void
mali_wsi_fbdev_init(struct mali_physical_device *pdev)
{
   pdev->fbdev = NULL;
   if (pdev->drm_fd >= 0 || !pdev->wsi)
      return;
   const char *path = "/dev/fb0";

   const int fd = open(path, O_RDWR | O_CLOEXEC);
   if (fd < 0)
      return;
   struct fb_var_screeninfo var;
   const bool ok = ioctl(fd, FBIOGET_VSCREENINFO, &var) == 0 && var.xres && var.yres &&
                   (var.bits_per_pixel == 32 || var.bits_per_pixel == 16);
   close(fd);
   if (!ok)
      return;

   struct mali_fbdev *fb = vk_zalloc(&pdev->vk.instance->alloc, sizeof(*fb), 8,
                                     VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!fb)
      return;
   snprintf(fb->path, sizeof(fb->path), "%s", path);
   fb->var = var;
   fb->refresh_mhz = fb_refresh_mhz(&var);
   fb->rotate = fb_panel_rotation();
   fb->base.get_support = fb_surface_get_support;
   fb->base.get_capabilities2 = fb_surface_get_capabilities2;
   fb->base.get_formats = fb_surface_get_formats;
   fb->base.get_formats2 = fb_surface_get_formats2;
   fb->base.get_present_modes = fb_surface_get_present_modes;
   fb->base.get_present_rectangles = fb_surface_get_present_rectangles;
   fb->base.create_swapchain = fb_surface_create_swapchain;

   fb->drm_iface = pdev->wsi->wsi[VK_ICD_WSI_PLATFORM_DISPLAY];
   pdev->wsi->wsi[VK_ICD_WSI_PLATFORM_DISPLAY] = &fb->base;
   /* CPU images that are themselves linear (wsi_cpu_image_needs_buffer_blit):
    * the application renders into what the copy reads, with no GPU blit
    * into a separate buffer. Nothing else uses wants_linear without a DRM
    * node. */
   pdev->wsi->wants_linear = true;
   pdev->fbdev = fb;
   mesa_logi("malisx2 VK_KHR_display on %s: %ux%u %u bpp at %u.%03u Hz (no DRM node)%s", path,
             var.xres, var.yres, var.bits_per_pixel, fb->refresh_mhz / 1000,
             fb->refresh_mhz % 1000, fb->rotate ? ", turned" : "");
   if (fb->rotate)
      mesa_logi("malisx2 fbdev: the panel is turned %u quarter%s clockwise (fbcon rotate), "
                "the display is %ux%u",
                fb->rotate, fb->rotate > 1 ? "s" : "", fb_extent(fb).width, fb_extent(fb).height);
}

void
mali_wsi_fbdev_finish(struct mali_physical_device *pdev)
{
   struct mali_fbdev *fb = pdev->fbdev;
   if (!fb)
      return;
   /* wsi_device_finish frees the DISPLAY platform's wsi_display. */
   if (pdev->wsi)
      pdev->wsi->wsi[VK_ICD_WSI_PLATFORM_DISPLAY] = fb->drm_iface;
   pdev->fbdev = NULL;
   vk_free(&pdev->vk.instance->alloc, fb);
}
