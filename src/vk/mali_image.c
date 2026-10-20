/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * VkImage and VkImageView: layout choice, memory requirements, binding,
 * subresource layouts. Render targets and depth buffers are AFBC where
 * every path that touches them can handle it; other images are 16x16
 * u-interleaved unless linear is asked for or tiling cannot help.
 */

#include "mali_vk.h"
#include "mali_image.h"
#include "mali_memory.h"
#include "mali_wsi.h"
#include "mali_cmd_gfx.h"

#include <stdlib.h>

#include "drm-uapi/drm_fourcc.h"
#include "util/format/u_format.h"
#include "util/log.h"
#include <string.h>
#include "vk_format.h"
#include "vk_util.h"

/* Whether an image's memory is shared outside the device: external memory,
 * a swapchain image or an Android native buffer. Other processes and the
 * display read and write its bytes directly. */
static bool
external_memory(const VkImageCreateInfo *info)
{
   if (vk_find_struct_const(info->pNext, EXTERNAL_MEMORY_IMAGE_CREATE_INFO) ||
       vk_find_struct_const(info->pNext, IMAGE_SWAPCHAIN_CREATE_INFO_KHR))
      return true;
#ifdef VK_USE_PLATFORM_ANDROID_KHR
   if (vk_find_struct_const(info->pNext, NATIVE_BUFFER_ANDROID))
      return true;
#endif
   return false;
}

/*
 * AFBC (lossless framebuffer compression) for the images whose memory
 * traffic is the GPU's own: colour and depth/stencil attachments. Kept to
 * what ARMSX2 creates (single-sampled 2D, one level, one layer) and to
 * images nothing reads or writes as raw bytes behind the GPU's back:
 *
 *  - no storage (shader image stores cannot write AFBC), no host transfer;
 *  - no mutable or block-compatible formats and no aliasing: views keep
 *    the plane's compression mode;
 *  - every plane's format has an AFBC mode on this arch (not R32 formats,
 *    for one; v9 has no 16-bit-channel modes);
 *  - large enough for a header tile (panvk's threshold; smaller images
 *    gain little).
 *
 * Copies, clears and blits into and out of these images go through the
 * texture unit and the tile buffer (mali_cmd_image.c), never through the
 * raw compute copy. panvk makes the same choices (panvk_image_can_use_mod,
 * pan_mod_afbc_test_props).
 *
 * v9 and v11. On a G57 at 2x upscaling, uncompressed render targets made
 * fragment-bound dumps 20-25 % slower than with AFBC.
 *
 * v7 has narrower rules of its own, colour attachments only
 * (v7_can_use_afbc).
 */
static bool
afbc_mode_on_arch(enum mali_afbc_mode mode, unsigned arch)
{
   switch (mode) {
   case MALI_AFBC_NONE:
      return false;
   case MALI_AFBC_R16:
   case MALI_AFBC_R16G16:
   case MALI_AFBC_R16G16B16A16:
      /* The 16-bit-channel compression modes start at v10. */
      return arch >= 10;
   default:
      return true;
   }
}

static bool
can_use_afbc(const struct mali_image *image, const enum pipe_format *formats, unsigned arch)
{
   const VkImageUsageFlags usage = image->vk.usage | image->vk.stencil_usage;
   if (!(usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)))
      return false;
   if (usage & (VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_HOST_TRANSFER_BIT))
      return false;
   if (image->vk.create_flags & (VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
                                 VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT |
                                 VK_IMAGE_CREATE_ALIAS_BIT |
                                 VK_IMAGE_CREATE_DISJOINT_BIT))
      return false;
   if (image->vk.tiling != VK_IMAGE_TILING_OPTIMAL ||
       image->vk.image_type != VK_IMAGE_TYPE_2D || image->vk.samples != 1 ||
       image->vk.mip_levels != 1 || image->vk.array_layers != 1)
      return false;
   for (unsigned p = 0; p < image->plane_count; p++) {
      const unsigned min = mali_afbc_min_extent(formats[p]);
      if (!afbc_mode_on_arch(mali_afbc_mode(formats[p]), arch) ||
          image->vk.extent.width < min || image->vk.extent.height < min)
         return false;
   }
   return true;
}

/*
 * The plane formats v7 AFBC takes here: 8-bit UNORM R, RG and RGBA (sRGB
 * as a flag on RGBA8), whose component order the v7 texture unit accepts
 * with AFBC as it is. pan_afbc_format at PAN_ARCH 7 accepts more (RGB565,
 * RGB5A1, RGBA4, RGB10A2, the BGR orders); those are left uncompressed
 * here. The BGR orders also need pan_texture_afbc_reswizzle, with the
 * sampler border colours swizzled to match (panvk_vX_sampler.c).
 */
static bool
v7_afbc_format(enum pipe_format f)
{
   switch (f) {
   case PIPE_FORMAT_R8_UNORM:
   case PIPE_FORMAT_R8G8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_SRGB:
      return true;
   default:
      return false;
   }
}

/*
 * AFBC on v7 (panvk_image_can_use_mod and pan_mod_afbc_test_props at
 * PAN_ARCH 7, narrowed): a single-plane 2D colour attachment in a format
 * of v7_afbc_format, optimal tiling, single-sampled, one level and one
 * layer, at least half a header tile each way. Not for images whose bytes
 * are read or written uncompressed (storage, host transfer, external
 * memory, swapchain and Android native buffers), nor for mutable,
 * block-compatible or aliased ones (v7 cannot read an AFBC plane in
 * another format), nor for disjoint or sparse ones.
 */
static bool
v7_can_use_afbc(const struct mali_image *image, const enum pipe_format *formats,
                const VkImageCreateInfo *info)
{
   const VkImageUsageFlags usage = image->vk.usage | image->vk.stencil_usage;
   if (!(usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) ||
       (usage & (VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                 VK_IMAGE_USAGE_HOST_TRANSFER_BIT)))
      return false;
   if (image->plane_count != 1 || vk_format_is_depth_or_stencil(image->vk.format) ||
       !v7_afbc_format(formats[0]))
      return false;
   if (image->vk.create_flags &
       (VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT |
        VK_IMAGE_CREATE_ALIAS_BIT | VK_IMAGE_CREATE_DISJOINT_BIT |
        VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT |
        VK_IMAGE_CREATE_SPARSE_ALIASED_BIT))
      return false;
   if (image->vk.tiling != VK_IMAGE_TILING_OPTIMAL ||
       image->vk.image_type != VK_IMAGE_TYPE_2D || image->vk.samples != 1 ||
       image->vk.mip_levels != 1 || image->vk.array_layers != 1)
      return false;
   if (external_memory(info))
      return false;
   const unsigned min = mali_afbc_min_extent(formats[0]);
   return image->vk.extent.width >= min && image->vk.extent.height >= min;
}

/*
 * The layout of an image, by panvk's rules for the modifiers we use
 * (panvk_image_can_use_mod, pan_mod_u_tiled_test_props):
 *
 * - AFBC when can_use_afbc says so (v7_can_use_afbc on v7);
 * - linear when the application asks for linear tiling, for 1D images,
 *   and when the base level is a single texel wide or high (tiling cannot
 *   help then);
 * - linear for a compressed image that may be viewed with an uncompressed
 *   format (BLOCK_TEXEL_VIEW_COMPATIBLE): u-interleaving differs between
 *   the two;
 * - 16x16 u-interleaved otherwise.
 */
static uint64_t
choose_modifier(const struct mali_image *image, const enum pipe_format *formats, unsigned arch,
                const VkImageCreateInfo *info)
{
   if (arch == 7 ? v7_can_use_afbc(image, formats, info) : can_use_afbc(image, formats, arch))
      return mali_afbc_can_ytr(formats[0]) ? MALI_MOD_AFBC_YTR : MALI_MOD_AFBC;
   if (image->vk.tiling == VK_IMAGE_TILING_LINEAR ||
       image->vk.image_type == VK_IMAGE_TYPE_1D ||
       image->vk.extent.width < 2 || image->vk.extent.height < 2)
      return DRM_FORMAT_MOD_LINEAR;
   if (vk_format_is_compressed(image->vk.format) &&
       (image->vk.create_flags & VK_IMAGE_CREATE_BLOCK_TEXEL_VIEW_COMPATIBLE_BIT))
      return DRM_FORMAT_MOD_LINEAR;
   return DRM_FORMAT_MOD_ARM_16X16_BLOCK_U_INTERLEAVED;
}

/*
 * Which images get a CRC table for transaction elimination.
 * panvk_should_checksum's rules, plus the ones our CRC handling needs:
 *
 * - optimal tiling (u-interleaved): a linear image can be written through
 *   a host mapping behind the CRC's back;
 * - a single-sampled, single-layer 2D colour attachment of at most 4 bytes
 *   per pixel (panvk's limit); the CRC covers level 0 only;
 * - nothing that writes it outside render passes and copies: no storage
 *   or host-transfer usage, no depth/stencil usage;
 * - no mutable format (views of another format would write the same
 *   memory with other bits), no aliasing flag, no sparse binding;
 * - no external or swapchain memory (other processes write it) and no
 *   transient attachment (lazily allocated memory has no CPU mapping to
 *   initialize the CRC state through).
 */
bool
mali_image_wants_crc(const VkImageCreateInfo *info, uint64_t modifier)
{
   /* AFBC targets can carry CRC on v11 (panvk's pan_image_view_can_crc:
    * sparse AFBC whose superblock fits a tile; ours are 16x16 sparse, and a
    * pass only uses CRC with tiles of 16x16 or more). */
   if ((modifier != DRM_FORMAT_MOD_ARM_16X16_BLOCK_U_INTERLEAVED &&
        !mali_mod_is_afbc(modifier)) ||
       info->imageType != VK_IMAGE_TYPE_2D || info->arrayLayers != 1 ||
       info->samples != VK_SAMPLE_COUNT_1_BIT || info->extent.depth != 1)
      return false;
   if (!(info->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) ||
       (info->usage & (VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                       VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT |
                       VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT)))
      return false;
   if (info->flags & (VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
                      VK_IMAGE_CREATE_ALIAS_BIT))
      return false;
   if (external_memory(info))
      return false;
   if (vk_format_is_depth_or_stencil(info->format) || vk_format_is_compressed(info->format) ||
       vk_format_get_plane_count(info->format) != 1)
      return false;
   return vk_format_get_blocksize(info->format) <= 4;
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_CreateImage(VkDevice _device, const VkImageCreateInfo *pCreateInfo,
                 const VkAllocationCallbacks *pAllocator, VkImage *pImage)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   struct mali_physical_device *pdev = mali_device_physical(dev);
   VkResult result;

   /* Swapchain images (VkNativeBufferANDROID) are laid out and bound
    * below; AHB images are not supported. */
   if (pCreateInfo->tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT)
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "DRM format modifier tiling is not supported");

   struct mali_image *image =
      vk_image_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*image));
   if (!image)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   enum pipe_format formats[MALI_IMAGE_MAX_PLANES];
   image->plane_count = mali_format_planes(image->vk.format, formats);
   if (!image->plane_count) {
      result = vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                         "images of format %d are not supported", image->vk.format);
      goto fail;
   }
   const unsigned arch = mali_device_physical(dev)->arch;
   image->vk.drm_format_mod = choose_modifier(image, formats, arch, pCreateInfo);
   /* No CRC on v7 and v9 yet: the job manager cannot update the seed from the
    * GPU, and the CPU-side seed it needs is not written yet. */
   const bool crc = !mali_arch_is_jm(arch) &&
                    mali_image_wants_crc(pCreateInfo, image->vk.drm_format_mod);

   /* Planes follow each other; each starts 4 KiB aligned (data_size is
    * rounded so). */
   uint64_t offset = 0;
   for (unsigned p = 0; p < image->plane_count; p++) {
      const struct mali_image_layout_info info = {
         .format = formats[p],
         .modifier = image->vk.drm_format_mod,
         .width = image->vk.extent.width,
         .height = image->vk.extent.height,
         .depth = image->vk.extent.depth,
         .samples = image->vk.samples,
         .levels = image->vk.mip_levels,
         .layers = image->vk.array_layers,
         .crc = crc && p == 0,
      };
      image->planes[p].format = formats[p];
      if (!mali_image_plane_layout_init(&info, offset, &image->planes[p].layout)) {
         result = vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                            "no layout for plane %u of a %ux%ux%u image of format %d", p,
                            info.width, info.height, info.depth, image->vk.format);
         goto fail;
      }
      offset += image->planes[p].layout.data_size;
   }
   image->size = offset;

   /* maxResourceSize: an image has to fit one allocation. */
   if (image->size > pdev->vk.properties.maxMemoryAllocationSize) {
      result = vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                         "image of %llu bytes is above maxMemoryAllocationSize",
                         (unsigned long long)image->size);
      goto fail;
   }

#ifdef VK_USE_PLATFORM_ANDROID_KHR
   /* A swapchain image from the Android loader: the gralloc buffer's
    * memory, linear with gralloc's stride (mali_wsi.c). */
   const VkNativeBufferANDROID *nb =
      vk_find_struct_const(pCreateInfo->pNext, NATIVE_BUFFER_ANDROID);
   if (nb) {
      result = mali_android_bind_native_buffer(dev, image, nb, pAllocator);
      if (result != VK_SUCCESS)
         goto fail;
   }
#endif

   *pImage = mali_image_to_handle(image);
   return VK_SUCCESS;

fail:
   vk_image_destroy(&dev->vk, pAllocator, &image->vk);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
mali_DestroyImage(VkDevice _device, VkImage _image, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   VK_FROM_HANDLE(mali_image, image, _image);

   if (!image)
      return;
   if (image->wsi_mem)
      mali_FreeMemory(_device, mali_device_memory_to_handle(image->wsi_mem), pAllocator);
   vk_image_destroy(&dev->vk, pAllocator, &image->vk);
}

VKAPI_ATTR void VKAPI_CALL
mali_GetImageMemoryRequirements2(VkDevice _device, const VkImageMemoryRequirementsInfo2 *pInfo,
                                 VkMemoryRequirements2 *pMemoryRequirements)
{
   VK_FROM_HANDLE(mali_image, image, pInfo->image);

   /* Memory types as the blob: a transient attachment only in lazily
    * allocated memory, everything else in the host-visible types. No
    * disjoint planes, so VkImagePlaneMemoryRequirementsInfo does not
    * apply. */
   pMemoryRequirements->memoryRequirements = (VkMemoryRequirements) {
      .size = image->size,
      .alignment = MALI_IMAGE_ALIGNMENT,
      .memoryTypeBits = (image->vk.usage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) ?
                           MALI_MEMORY_TYPES_TRANSIENT : MALI_MEMORY_TYPES_HOST_VISIBLE,
   };

   vk_foreach_struct(stype, ext, pMemoryRequirements->pNext) {
      switch (stype) {
      case VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS: {
         /* Only AHB images need a dedicated allocation. */
         VkMemoryDedicatedRequirements *d = ext;
         d->prefersDedicatedAllocation = VK_FALSE;
         d->requiresDedicatedAllocation = VK_FALSE;
         break;
      }
      default:
         vk_debug_ignored_stype(stype);
         break;
      }
   }
}

/* The CRC knobs (mali_cmd_gfx.h): here, in code compiled once, so the
 * render-pass code of every arch reads the same ones. */
struct mali_crc_options mali_crc_options = {
   .empty_tile_read = true,
   .empty_tile_write = true,
};

/*
 * Start the image's CRC state from zero: seed 0 and a table of zeros, as
 * panvk does at bind. Whatever the memory held before (another image's
 * CRCs, pixels) must not be taken for CRCs of this image's contents. A
 * CRC of zero is as unlikely as any other 64-bit value. The write goes
 * through the allocation's CPU mapping; the GPU's L2 is cleaned and
 * invalidated at the start of every submit, so it sees it. Memory without
 * a CPU mapping (imports) leaves CRC off.
 */
static void
crc_init(struct mali_device *dev, struct mali_image *image)
{
   const struct mali_image_crc_layout *c = &image->planes[0].layout.crc;
   struct mali_device_memory *mem = image->mem;

   image->crc_header = 0;
   if (!c->header_offset || !mem->bo.cpu || mem->dmabuf_fd >= 0)
      return;
   const uint64_t off = image->mem_offset + c->header_offset;
   const uint64_t size = MALI_CRC_HEADER_SIZE + c->size;
   if (off + size > mem->bo.size)
      return;
   memset((uint8_t *)mem->bo.cpu + off, 0, size);
   if (!(mem->props & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
      mali_kbase_bo_flush(dev->kbase, &mem->bo, off, size);
   image->crc_header = image->base + c->header_offset;
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_BindImageMemory2(VkDevice _device, uint32_t bindInfoCount,
                      const VkBindImageMemoryInfo *pBindInfos)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   VkResult result = VK_SUCCESS;

   for (uint32_t i = 0; i < bindInfoCount; i++) {
      VK_FROM_HANDLE(mali_image, image, pBindInfos[i].image);
      VK_FROM_HANDLE(mali_device_memory, mem, pBindInfos[i].memory);
      const VkBindMemoryStatus *status =
         vk_find_struct_const(pBindInfos[i].pNext, BIND_MEMORY_STATUS);
      VkResult r = VK_SUCCESS;

      if (mem) {
         /* No kernel call: the image records where it lives. */
         image->mem = mem;
         image->mem_offset = pBindInfos[i].memoryOffset;
         image->base = mali_device_memory_gpu_va(mem, pBindInfos[i].memoryOffset);
         crc_init(dev, image);
      } else {
#ifdef VK_USE_PLATFORM_ANDROID_KHR
         /* An image created with VkImageSwapchainCreateInfoKHR bound to a
          * swapchain image: the loader adds the gralloc buffer
          * (VK_ANDROID_native_buffer spec version 8). */
         const VkNativeBufferANDROID *nb =
            vk_find_struct_const(pBindInfos[i].pNext, NATIVE_BUFFER_ANDROID);
         if (nb)
            r = mali_android_bind_native_buffer(dev, image, nb, NULL);
         else
#endif
            r = vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                          "binding an image without memory");
      }
      if (status)
         *status->pResult = r;
      if (r != VK_SUCCESS)
         result = r;
   }
   return result;
}

VKAPI_ATTR void VKAPI_CALL
mali_GetImageSubresourceLayout2(VkDevice _device, VkImage _image,
                                const VkImageSubresource2 *pSubresource,
                                VkSubresourceLayout2 *pLayout)
{
   VK_FROM_HANDLE(mali_image, image, _image);
   const VkImageSubresource *sub = &pSubresource->imageSubresource;
   const struct mali_image_plane_layout *pl =
      &image->planes[mali_image_aspect_plane(image, sub->aspectMask)].layout;
   const struct mali_image_slice *s = &pl->slices[sub->mipLevel];

   pLayout->subresourceLayout = (VkSubresourceLayout) {
      .offset = s->offset + sub->arrayLayer * pl->array_stride,
      .size = s->size,
      .rowPitch = s->row_stride,
      .arrayPitch = pl->array_stride,
      .depthPitch = s->surface_stride,
   };
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_CreateImageView(VkDevice _device, const VkImageViewCreateInfo *pCreateInfo,
                     const VkAllocationCallbacks *pAllocator, VkImageView *pView)
{
   VK_FROM_HANDLE(mali_device, dev, _device);

   struct mali_image_view *view =
      vk_image_view_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*view));
   if (!view)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result =
      mali_arch_dispatch(mali_device_physical(dev)->arch, image_view_init_descs, dev, view);
   if (result != VK_SUCCESS) {
      vk_image_view_destroy(&dev->vk, pAllocator, &view->vk);
      return result;
   }

   *pView = mali_image_view_to_handle(view);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
mali_DestroyImageView(VkDevice _device, VkImageView _view,
                      const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   VK_FROM_HANDLE(mali_image_view, view, _view);

   if (!view)
      return;
   mali_bo_pool_free(&dev->desc_pool, &view->plane_descs);
   vk_image_view_destroy(&dev->vk, pAllocator, &view->vk);
}
