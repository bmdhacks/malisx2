/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Format support: vkGetPhysicalDeviceFormatProperties(2) and
 * vkGetPhysicalDeviceImageFormatProperties(2).
 *
 * The answers come from Mesa's format table for the device's architecture
 * (pan_format.c, vendored; the per-arch half is mali_format_table.c) and
 * the GPU's compressed-format bits (TEXTURE_FEATURES_0), derived the way
 * panvk does (panvk_physical_device.c), because the descriptors that use
 * the formats are packed from the same table. Where the blob's answer
 * differs and the blob's is the safer one we follow it: integer formats
 * get no blend or linear-filter bit, and combined depth/stencil formats
 * have no linear-tiling features. YCbCr formats, sparse images and protected
 * images are not supported. External memory: on Android only the Android
 * loader's question about swapchain buffers (the hardware-buffer handle
 * type) is answered; elsewhere dma-bufs can be imported, as linear images
 * (DRM_FORMAT_MOD_LINEAR, the one modifier offered) or as buffers.
 */

#include "mali_vk.h"
#include "mali_image.h"
#include "mali_wsi.h"

#include "util/format/u_format.h"
#include "util/log.h"
#include "vk_format.h"
#include "vk_util.h"
#include "vulkan/vulkan_android.h"

unsigned
mali_format_planes(VkFormat format, enum pipe_format planes[MALI_IMAGE_MAX_PLANES])
{
   switch (format) {
   /* Combined depth/stencil in two planes, as panvk does on arch 9 and
    * later (Z32_S8X24 does not exist there) and as the blob keeps stencil
    * in a second slice array. */
   case VK_FORMAT_D32_SFLOAT_S8_UINT:
      planes[0] = PIPE_FORMAT_Z32_FLOAT;
      planes[1] = PIPE_FORMAT_S8_UINT;
      return 2;
   case VK_FORMAT_D24_UNORM_S8_UINT:
      planes[0] = PIPE_FORMAT_Z24X8_UNORM;
      planes[1] = PIPE_FORMAT_S8_UINT;
      return 2;
   default:
      break;
   }

   if (vk_format_get_plane_count(format) != 1)
      return 0;
   planes[0] = vk_format_to_pipe_format(format);
   return planes[0] == PIPE_FORMAT_NONE ? 0 : 1;
}

VkFormatFeatureFlags
mali_format_image_features(const struct mali_physical_device *pdev, VkFormat format)
{
   return mali_arch_dispatch(pdev->arch, format_image_features, pdev, format);
}

VkFormatFeatureFlags
mali_format_buffer_features(const struct mali_physical_device *pdev, VkFormat format)
{
   return mali_arch_dispatch(pdev->arch, format_buffer_features, pdev, format);
}

/* Linear images of a combined depth/stencil format are not offered (the
 * blob reports optimal tiling only for them). */
static VkFormatFeatureFlags
linear_features(const struct mali_physical_device *pdev, VkFormat format)
{
   enum pipe_format planes[MALI_IMAGE_MAX_PLANES];
   if (mali_format_planes(format, planes) > 1)
      return 0;
   return mali_format_image_features(pdev, format);
}

VKAPI_ATTR void VKAPI_CALL
mali_GetPhysicalDeviceFormatProperties2(VkPhysicalDevice physicalDevice, VkFormat format,
                                        VkFormatProperties2 *pFormatProperties)
{
   VK_FROM_HANDLE(mali_physical_device, pdev, physicalDevice);

   const VkFormatFeatureFlags linear = linear_features(pdev, format);
   pFormatProperties->formatProperties = (VkFormatProperties) {
      .linearTilingFeatures = linear,
      .optimalTilingFeatures = mali_format_image_features(pdev, format),
      .bufferFeatures = mali_format_buffer_features(pdev, format),
   };

   /* DRM format modifiers: DRM_FORMAT_MOD_LINEAR alone, with the linear
    * tiling features, for every format linear images can have. The list's
    * second form only differs in the width of the feature flags. */
   vk_foreach_struct(stype, ext, pFormatProperties->pNext) {
      switch (stype) {
      case VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT: {
         VkDrmFormatModifierPropertiesListEXT *list = ext;
         VK_OUTARRAY_MAKE_TYPED(VkDrmFormatModifierPropertiesEXT, out,
                                list->pDrmFormatModifierProperties,
                                &list->drmFormatModifierCount);
         if (linear) {
            vk_outarray_append_typed(VkDrmFormatModifierPropertiesEXT, &out, p) {
               *p = (VkDrmFormatModifierPropertiesEXT){
                  .drmFormatModifier = DRM_FORMAT_MOD_LINEAR,
                  .drmFormatModifierPlaneCount = 1,
                  .drmFormatModifierTilingFeatures = linear,
               };
            }
         }
         break;
      }
      case VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_2_EXT: {
         VkDrmFormatModifierPropertiesList2EXT *list = ext;
         VK_OUTARRAY_MAKE_TYPED(VkDrmFormatModifierProperties2EXT, out,
                                list->pDrmFormatModifierProperties,
                                &list->drmFormatModifierCount);
         if (linear) {
            vk_outarray_append_typed(VkDrmFormatModifierProperties2EXT, &out, p) {
               *p = (VkDrmFormatModifierProperties2EXT){
                  .drmFormatModifier = DRM_FORMAT_MOD_LINEAR,
                  .drmFormatModifierPlaneCount = 1,
                  .drmFormatModifierTilingFeatures = linear,
               };
            }
         }
         break;
      }
      default:
         vk_debug_ignored_stype(stype);
         break;
      }
   }
}

static VkResult
image_format_properties(const struct mali_physical_device *pdev,
                        const VkPhysicalDeviceImageFormatInfo2 *info,
                        VkImageFormatProperties *props)
{
   const VkImageStencilUsageCreateInfo *stencil_usage_info =
      vk_find_struct_const(info->pNext, IMAGE_STENCIL_USAGE_CREATE_INFO);
   const VkImageUsageFlags usage =
      info->usage | (stencil_usage_info ? stencil_usage_info->stencilUsage : 0);

   if (info->flags & (VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                      VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT |
                      VK_IMAGE_CREATE_SPARSE_ALIASED_BIT |
                      VK_IMAGE_CREATE_PROTECTED_BIT |
                      VK_IMAGE_CREATE_DISJOINT_BIT))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   VkFormatFeatureFlags feat;
   switch (info->tiling) {
   case VK_IMAGE_TILING_OPTIMAL:
      feat = mali_format_image_features(pdev, info->format);
      break;
   case VK_IMAGE_TILING_LINEAR:
      feat = linear_features(pdev, info->format);
      break;
   case VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT: {
      /* DRM_FORMAT_MOD_LINEAR is linear tiling under another name. */
      const VkPhysicalDeviceImageDrmFormatModifierInfoEXT *mod_info =
         vk_find_struct_const(info->pNext, PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT);
      if (!mod_info || mod_info->drmFormatModifier != DRM_FORMAT_MOD_LINEAR)
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
      feat = linear_features(pdev, info->format);
      break;
   }
   default:
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   }
   if (!feat)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   const struct vk_properties *lim = &pdev->vk.properties;
   VkExtent3D extent;
   uint32_t layers;
   switch (info->type) {
   case VK_IMAGE_TYPE_1D:
      extent = (VkExtent3D){lim->maxImageDimension1D, 1, 1};
      layers = lim->maxImageArrayLayers;
      break;
   case VK_IMAGE_TYPE_2D:
      extent = (VkExtent3D){lim->maxImageDimension2D, lim->maxImageDimension2D, 1};
      layers = lim->maxImageArrayLayers;
      break;
   case VK_IMAGE_TYPE_3D:
      extent = (VkExtent3D){lim->maxImageDimension3D, lim->maxImageDimension3D,
                            lim->maxImageDimension3D};
      layers = 1;
      break;
   default:
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   }

   /* Unless EXTENDED_USAGE lets views use other formats, every usage needs
    * its format feature (as panvk). */
   if (!(info->flags & VK_IMAGE_CREATE_EXTENDED_USAGE_BIT)) {
      const bool ds = vk_format_is_depth_or_stencil(info->format);
      if ((usage & VK_IMAGE_USAGE_SAMPLED_BIT) && !(feat & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT))
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
      if ((usage & VK_IMAGE_USAGE_STORAGE_BIT) && !(feat & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
      if ((usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) &&
          !(feat & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT))
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
      if ((usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) &&
          !(feat & VK_FORMAT_FEATURE_TRANSFER_DST_BIT))
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
      if (((usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) ||
           ((usage & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) && !ds)) &&
          !(feat & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT))
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
      if (((usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ||
           ((usage & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) && ds)) &&
          !(feat & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
   }

   /* Multisampling: optimal 2D attachments, not cube, not storage (panvk's
    * condition), with the blob's per-size rule. */
   VkSampleCountFlags samples = VK_SAMPLE_COUNT_1_BIT;
   if (info->tiling == VK_IMAGE_TILING_OPTIMAL && info->type == VK_IMAGE_TYPE_2D &&
       (feat & (VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) &&
       !(info->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) &&
       !(usage & VK_IMAGE_USAGE_STORAGE_BIT)) {
      enum pipe_format planes[MALI_IMAGE_MAX_PLANES];
      mali_format_planes(info->format, planes);
      samples = mali_physical_device_sample_counts(pdev, util_format_get_blocksize(planes[0]));
   }

   *props = (VkImageFormatProperties) {
      .maxExtent = extent,
      .maxMipLevels = MALI_IMAGE_MAX_LEVELS,
      .maxArrayLayers = layers,
      .sampleCounts = samples,
      /* An image has to fit in one allocation. */
      .maxResourceSize = pdev->vk.properties.maxMemoryAllocationSize,
   };
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_GetPhysicalDeviceImageFormatProperties2(VkPhysicalDevice physicalDevice,
                                             const VkPhysicalDeviceImageFormatInfo2 *pInfo,
                                             VkImageFormatProperties2 *pProps)
{
   VK_FROM_HANDLE(mali_physical_device, pdev, physicalDevice);

   /*
    * Two external handle types:
    *
    * - Android's hardware buffer, only for what the Android loader asks
    *   before it allocates swapchain buffers: their gralloc usage
    *   (mali_wsi_ahb_usage). The image itself is created with a
    *   VkNativeBufferANDROID, not by importing the hardware buffer, so
    *   nothing is importable or exportable.
    * - dma-buf (not on Android), importable into a 2D linear image (linear
    *   tiling or DRM_FORMAT_MOD_LINEAR), not exportable: we cannot allocate
    *   memory another process can open.
    *
    * Either image has one level, one layer and one sample.
    */
   const VkPhysicalDeviceExternalImageFormatInfo *ext_info =
      vk_find_struct_const(pInfo->pNext, PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO);
   const VkExternalMemoryHandleTypeFlagBits handle_type = ext_info ? ext_info->handleType : 0;
   const bool ahb =
      handle_type == VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
#ifdef VK_USE_PLATFORM_ANDROID_KHR
   const bool dmabuf = false;
#else
   const bool dmabuf = handle_type == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
#endif
   uint64_t ahb_usage = 0;
   VkResult result;
   if (ahb) {
      result = mali_wsi_ahb_usage(pdev, pInfo, &ahb_usage);
      if (result == VK_SUCCESS)
         result = image_format_properties(pdev, pInfo, &pProps->imageFormatProperties);
   } else if (dmabuf) {
      result = pInfo->type == VK_IMAGE_TYPE_2D && pInfo->tiling != VK_IMAGE_TILING_OPTIMAL ?
                  image_format_properties(pdev, pInfo, &pProps->imageFormatProperties) :
                  VK_ERROR_FORMAT_NOT_SUPPORTED;
   } else if (handle_type) {
      result = VK_ERROR_FORMAT_NOT_SUPPORTED;
   } else {
      result = image_format_properties(pdev, pInfo, &pProps->imageFormatProperties);
   }
   if (result != VK_SUCCESS) {
      pProps->imageFormatProperties = (VkImageFormatProperties){0};
      return result;
   }
   if (ahb || dmabuf) {
      VkImageFormatProperties *p = &pProps->imageFormatProperties;
      p->maxMipLevels = 1;
      p->maxArrayLayers = 1;
      p->sampleCounts = VK_SAMPLE_COUNT_1_BIT;
   }

   vk_foreach_struct(stype, ext, pProps->pNext) {
      switch (stype) {
      case VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES: {
         VkExternalImageFormatProperties *e = ext;
         e->externalMemoryProperties = (VkExternalMemoryProperties){
            .externalMemoryFeatures = dmabuf ? VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT : 0,
            .compatibleHandleTypes = handle_type,
         };
         break;
      }
      case VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_USAGE_ANDROID: {
         VkAndroidHardwareBufferUsageANDROID *u = ext;
         u->androidHardwareBufferUsage = ahb_usage;
         break;
      }
      case VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES: {
         VkSamplerYcbcrConversionImageFormatProperties *y = ext;
         y->combinedImageSamplerDescriptorCount = 1;
         break;
      }
      default:
         vk_debug_ignored_stype(stype);
         break;
      }
   }
   return VK_SUCCESS;
}

/* Buffers: a dma-buf can be imported as a buffer's memory (not on Android,
 * where nothing is imported this way); nothing is exportable. */
VKAPI_ATTR void VKAPI_CALL
mali_GetPhysicalDeviceExternalBufferProperties(
   VkPhysicalDevice physicalDevice, const VkPhysicalDeviceExternalBufferInfo *pExternalBufferInfo,
   VkExternalBufferProperties *pExternalBufferProperties)
{
#ifdef VK_USE_PLATFORM_ANDROID_KHR
   const bool dmabuf = false;
#else
   const bool dmabuf =
      pExternalBufferInfo->handleType == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
#endif
   pExternalBufferProperties->externalMemoryProperties = (VkExternalMemoryProperties){
      .externalMemoryFeatures = dmabuf ? VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT : 0,
      .compatibleHandleTypes = dmabuf ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT : 0,
   };
}
