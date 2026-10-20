/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The per-arch half of format support (mali_formats.c): image and buffer
 * features from Mesa's format table of this arch (pan_format.c, built once
 * per arch) and the GPU's compressed-format bits. The arches' tables
 * differ in a few entries (v9 has no R10X6-style formats and numbers the
 * narrow ASTC formats differently from v11; v7 gives compressed formats
 * their own hardware format, not an interchange one); each arch answers
 * from its own.
 */

#ifndef PAN_ARCH
#error "mali_format_table.c is built per arch: PAN_ARCH must be set"
#endif

#include "mali_vk.h"
#include "mali_arch.h"
#include "mali_image.h"

#include "pan_format.h"

#include "util/format/u_format.h"
#include "vk_format.h"

static const struct pan_format *
pan_fmt(enum pipe_format p)
{
   return GENX(pan_format_from_pipe_format)(p);
}

/* Compressed formats are optional in the hardware: TEXTURE_FEATURES_0 has
 * one bit per compressed texture format (pan_query_compressed_formats). */
static uint32_t
compressed_format_bits(const struct mali_physical_device *pdev)
{
   return pdev->props.texture_features[0] ? pdev->props.texture_features[0]
                                          : pdev->props.raw_texture_features[0];
}

static bool
pipe_format_supported(const struct mali_physical_device *pdev, enum pipe_format p)
{
   if (p == PIPE_FORMAT_NONE || !pan_fmt(p)->hw)
      return false;
   /* No YCbCr conversion support (and Mesa's subsampled RGB formats are
    * YUV to the hardware). */
   if (pan_format_is_yuv(p))
      return false;
   if (util_format_is_compressed(p) &&
       !(compressed_format_bits(pdev) & BITFIELD_BIT(pan_fmt(p)->texfeat_bit)))
      return false;
   return true;
}

VkFormatFeatureFlags
MALI_PER_ARCH(format_image_features)(const struct mali_physical_device *pdev, VkFormat format)
{
   enum pipe_format planes[MALI_IMAGE_MAX_PLANES];
   const unsigned nplanes = mali_format_planes(format, planes);
   if (!nplanes)
      return 0;
   for (unsigned i = 0; i < nplanes; i++)
      if (!pipe_format_supported(pdev, planes[i]))
         return 0;

   const enum pipe_format p = vk_format_to_pipe_format(format);
   if (!pipe_format_supported(pdev, p))
      return 0;

   const struct pan_format *f = pan_fmt(p);
   const bool integer = util_format_is_pure_integer(p);
   VkFormatFeatureFlags feat = 0;

   if (f->bind & PAN_BIND_SAMPLER_VIEW) {
      feat |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_BLIT_SRC_BIT |
              VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
      /* Integer and scaled formats filter nearest only. */
      if (!integer && !util_format_is_scaled(p))
         feat |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
   }

   if (f->bind & PAN_BIND_RENDER_TARGET) {
      feat |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
      /* The blob reports no blending for integer formats (R32_UINT is
       * 0xcc87); Vulkan never blends them. */
      if (!integer)
         feat |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT;
   }

   if (f->bind & PAN_BIND_STORAGE_IMAGE) {
      feat |= VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
      if (p == PIPE_FORMAT_R32_UINT || p == PIPE_FORMAT_R32_SINT)
         feat |= VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT;
   }

   if (f->bind & PAN_BIND_DEPTH_STENCIL)
      feat |= VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
#if PAN_ARCH < 9
   /* v7 blits depth and stencil through a frame shader that writes them
    * (mali_cmd_image.c), so its sampleable depth/stencil formats are blit
    * destinations too (nearest only, identical formats, as Vulkan asks). */
   if ((f->bind & PAN_BIND_DEPTH_STENCIL) && (f->bind & PAN_BIND_SAMPLER_VIEW))
      feat |= VK_FORMAT_FEATURE_BLIT_DST_BIT;
#endif

   return feat;
}

VkFormatFeatureFlags
MALI_PER_ARCH(format_buffer_features)(const struct mali_physical_device *pdev, VkFormat format)
{
   const enum pipe_format p = vk_format_to_pipe_format(format);
   if (vk_format_get_plane_count(format) != 1 || !pipe_format_supported(pdev, p))
      return 0;

   const struct pan_format *f = pan_fmt(p);
   VkFormatFeatureFlags feat = 0;

   /* sRGB vertex formats are rejected, as panvk does (Vulkan-Docs issue
    * 2214). */
   if ((f->bind & PAN_BIND_VERTEX_BUFFER) && !util_format_is_srgb(p))
      feat |= VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT;
   if (f->bind & PAN_BIND_TEXEL_BUFFER)
      feat |= VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT |
              VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT;
   if ((f->bind & PAN_BIND_TEXEL_BUFFER) &&
       (p == PIPE_FORMAT_R32_UINT || p == PIPE_FORMAT_R32_SINT))
      feat |= VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT;
   return feat;
}

