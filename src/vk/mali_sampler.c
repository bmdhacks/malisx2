/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * vkCreateSampler / vkDestroySampler: the Sampler descriptor, packed once
 * at creation as panvk does (panvk_vX_sampler.c, MIT), built per arch. The
 * blob packs the same fields; the two value differences are settled in
 * panvk's favour (round-to-nearest-even, and the compare function of a
 * non-comparison sampler). No YCbCr conversion (not advertised). Custom
 * border colours (v7) come from Mesa's runtime like the standard ones.
 * v9's Sampler has no reduction mode (min/max filtering is not advertised
 * there).
 */

#ifndef PAN_ARCH
#error "mali_sampler.c is built per arch: PAN_ARCH must be set"
#endif

#include "mali_vk.h"
#include "mali_arch.h"
#include "mali_descriptor_set.h"

#include "genxml/gen_macros.h"

#include "vk_log.h"
#include "vk_sampler.h"

static enum mali_mipmap_mode
mipmap_mode(VkSamplerMipmapMode mode, bool aniso)
{
   if (mode == VK_SAMPLER_MIPMAP_MODE_NEAREST)
      return MALI_MIPMAP_MODE_NEAREST;
   return aniso ? MALI_MIPMAP_MODE_PERFORMANCE_TRILINEAR : MALI_MIPMAP_MODE_TRILINEAR;
}

static enum mali_wrap_mode
wrap_mode(VkSamplerAddressMode mode)
{
   switch (mode) {
   case VK_SAMPLER_ADDRESS_MODE_REPEAT:
      return MALI_WRAP_MODE_REPEAT;
   case VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT:
      return MALI_WRAP_MODE_MIRRORED_REPEAT;
   case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER:
      return MALI_WRAP_MODE_CLAMP_TO_BORDER;
   case VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE:
      return MALI_WRAP_MODE_MIRRORED_CLAMP_TO_EDGE;
   case VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE:
   default:
      return MALI_WRAP_MODE_CLAMP_TO_EDGE;
   }
}

/* The hardware compares the other way round: Vulkan's LESS is its
 * GREATER. VkCompareOp and the hardware's function enum have the same
 * numbering otherwise. */
static enum mali_func
compare_func(const VkSamplerCreateInfo *info)
{
   if (!info->compareEnable)
      return MALI_FUNC_NEVER;
   switch ((enum mali_func)info->compareOp) {
   case MALI_FUNC_LESS:
      return MALI_FUNC_GREATER;
   case MALI_FUNC_GREATER:
      return MALI_FUNC_LESS;
   case MALI_FUNC_LEQUAL:
      return MALI_FUNC_GEQUAL;
   case MALI_FUNC_GEQUAL:
      return MALI_FUNC_LEQUAL;
   default:
      return (enum mali_func)info->compareOp;
   }
}

#if PAN_ARCH >= 10
static enum mali_reduction_mode
reduction_mode(VkSamplerReductionMode mode)
{
   switch (mode) {
   case VK_SAMPLER_REDUCTION_MODE_MIN:
      return MALI_REDUCTION_MODE_MINIMUM;
   case VK_SAMPLER_REDUCTION_MODE_MAX:
      return MALI_REDUCTION_MODE_MAXIMUM;
   default:
      return MALI_REDUCTION_MODE_AVERAGE;
   }
}
#endif

static void
pack_sampler(const struct mali_sampler *sampler, const VkSamplerCreateInfo *info,
             struct mali_hw_desc *out)
{
   const bool aniso = info->anisotropyEnable && info->maxAnisotropy > 1.0f;
   VkFormat border_format;
   const VkClearColorValue border = vk_sampler_border_color_value(info, &border_format);

   STATIC_ASSERT(sizeof(*out) == pan_size(SAMPLER));
   pan_cast_and_pack(out, SAMPLER, cfg) {
      cfg.magnify_nearest = info->magFilter == VK_FILTER_NEAREST;
      cfg.minify_nearest = info->minFilter == VK_FILTER_NEAREST;
      cfg.mipmap_mode = mipmap_mode(info->mipmapMode, aniso);
      cfg.normalized_coordinates = !info->unnormalizedCoordinates;
      cfg.clamp_integer_array_indices = false;
      /* Nearest-only sampling rounds towards zero: with round-to-even the
       * top 2^-9 of each texel would pick the next one (panvk). */
      if (info->minFilter == VK_FILTER_NEAREST && info->magFilter == VK_FILTER_NEAREST)
         cfg.round_to_nearest_even = false;
      cfg.lod_bias = info->mipLodBias;
      cfg.minimum_lod = info->minLod;
      cfg.maximum_lod = info->maxLod;
      cfg.wrap_mode_s = wrap_mode(info->addressModeU);
      cfg.wrap_mode_t = wrap_mode(info->addressModeV);
      /* Unnormalized coordinates allow only 1D/2D single-layer views, so
       * R is unused; clamp-to-edge is what the hardware accepts then. */
      cfg.wrap_mode_r = info->unnormalizedCoordinates ?
                           MALI_WRAP_MODE_CLAMP_TO_EDGE :
                           wrap_mode(info->addressModeW);
      cfg.seamless_cube_map =
         !(info->flags & VK_SAMPLER_CREATE_NON_SEAMLESS_CUBE_MAP_BIT_EXT);
      cfg.compare_function = compare_func(info);
      cfg.border_color_r = border.uint32[0];
      cfg.border_color_g = border.uint32[1];
      cfg.border_color_b = border.uint32[2];
      cfg.border_color_a = border.uint32[3];
      if (aniso) {
         cfg.maximum_anisotropy = MIN2((unsigned)info->maxAnisotropy, 16);
         cfg.lod_algorithm = MALI_LOD_ALGORITHM_ANISOTROPIC;
      }
#if PAN_ARCH >= 10
      cfg.reduction_mode = reduction_mode(sampler->vk.reduction_mode);
#else
      assert(sampler->vk.reduction_mode == VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE);
#endif
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
MALI_PER_ARCH(CreateSampler)(VkDevice _device, const VkSamplerCreateInfo *pCreateInfo,
                             const VkAllocationCallbacks *pAllocator, VkSampler *pSampler)
{
   VK_FROM_HANDLE(mali_device, dev, _device);

   struct mali_sampler *sampler =
      vk_sampler_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*sampler));
   if (!sampler)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   if (sampler->vk.ycbcr_conversion) {
      vk_sampler_destroy(&dev->vk, pAllocator, &sampler->vk);
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "samplers with a YCbCr conversion are not supported");
   }

   pack_sampler(sampler, pCreateInfo, &sampler->desc);

   *pSampler = mali_sampler_to_handle(sampler);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(DestroySampler)(VkDevice _device, VkSampler _sampler,
                              const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   VK_FROM_HANDLE(mali_sampler, sampler, _sampler);

   if (sampler)
      vk_sampler_destroy(&dev->vk, pAllocator, &sampler->vk);
}

void
MALI_PER_ARCH(pack_dummy_sampler)(void *out)
{
   pan_cast_and_pack(out, SAMPLER, cfg) {
      cfg.clamp_integer_array_indices = false;
   }
}
