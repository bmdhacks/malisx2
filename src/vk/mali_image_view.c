/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The Texture descriptors of an image view, and the plane descriptors they
 * point at, built per arch. Follows Mesa's pan_sampled_texture_emit /
 * pan_storage_texture_emit and the Generic and ASTC 2D plane emitters of
 * pan_texture.c (MIT) for the three layouts images have here, linear,
 * 16x16 u-interleaved and AFBC (emit_afbc_plane); the blob builds the
 * same descriptor. pan_texture.c itself is not vendored: it pulls in the
 * panthor kernel headers through pan_kmod.h and works on Mesa's image
 * structures, which we do not use.
 *
 * Plane descriptors: one per (layer, level) of the view, layer-major, in
 * the device's pipeline descriptor pool (CPU-uncached, GPU-readable). A
 * 3D view has one entry per level; its depth is in the Texture. Samples
 * of a multisampled image share one plane descriptor (slice stride =
 * sample stride).
 *
 * v9 and v11 differ only in the plane descriptors: v11's have a 48-bit
 * pointer with a size-hi word, a 27-bit row stride with the slice stride's
 * high bits above it, and the plane's width and height; v9's have a 64-bit
 * pointer, 32-bit sizes and strides and no extent (the T820 blob writes
 * the same v9 fields).
 *
 * v7 (Bifrost) has no plane descriptors: the Texture descriptor carries
 * the texel ordering and points at 16-byte Surface With Stride entries,
 * one per (layer, sample, level) in that nesting, cube faces counted as
 * layers and a 3D view as one layer (pan_texture.c
 * pan_emit_iview_texture_payload at PAN_ARCH 7). A v7 storage view is an
 * Attribute Buffer pair instead, which the descriptor copy of a draw or
 * dispatch turns into an image table entry (mali_jm_desc_v7.c).
 */

#ifndef PAN_ARCH
#error "mali_image_view.c is built per arch: PAN_ARCH must be set"
#endif

#include "mali_vk.h"
#include "mali_arch.h"
#include "mali_image.h"

#include <string.h>

#include "genxml/gen_macros.h"
#include "pan_format.h"

#include "drm-uapi/drm_fourcc.h"
#include "util/format/u_format.h"
#include "util/u_math.h"
#include "vk_format.h"
#include "vk_log.h"

#if PAN_ARCH < 9
#define PLANE_DESC_SIZE 16   /* Surface With Stride */
#else
#define PLANE_DESC_SIZE 32   /* a plane descriptor */
#endif

#if PAN_ARCH >= 9

/* Clump formats that are not raw N-byte texels (pan_texture.c's table,
 * less the YUV formats, which are not supported). */
static enum mali_clump_format
clump_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_X32_S8X24_UINT: return MALI_CLUMP_FORMAT_X32S8X24;
   case PIPE_FORMAT_X24S8_UINT: return MALI_CLUMP_FORMAT_X24S8;
   case PIPE_FORMAT_S8X24_UINT: return MALI_CLUMP_FORMAT_S8X24;
   case PIPE_FORMAT_S8_UINT: return MALI_CLUMP_FORMAT_S8;
   case PIPE_FORMAT_L4A4_UNORM: return MALI_CLUMP_FORMAT_L4A4;
   case PIPE_FORMAT_L8A8_UNORM:
   case PIPE_FORMAT_L8A8_UINT:
   case PIPE_FORMAT_L8A8_SINT: return MALI_CLUMP_FORMAT_L8A8;
   case PIPE_FORMAT_A8_UNORM:
   case PIPE_FORMAT_A8_UINT:
   case PIPE_FORMAT_A8_SINT: return MALI_CLUMP_FORMAT_A8;
   case PIPE_FORMAT_ETC1_RGB8:
   case PIPE_FORMAT_ETC2_RGB8:
   case PIPE_FORMAT_ETC2_SRGB8: return MALI_CLUMP_FORMAT_ETC2_RGB8;
   case PIPE_FORMAT_ETC2_RGB8A1:
   case PIPE_FORMAT_ETC2_SRGB8A1: return MALI_CLUMP_FORMAT_ETC2_RGB8A1;
   case PIPE_FORMAT_ETC2_RGBA8:
   case PIPE_FORMAT_ETC2_SRGBA8: return MALI_CLUMP_FORMAT_ETC2_RGBA8;
   case PIPE_FORMAT_ETC2_R11_UNORM: return MALI_CLUMP_FORMAT_ETC2_R11_UNORM;
   case PIPE_FORMAT_ETC2_R11_SNORM: return MALI_CLUMP_FORMAT_ETC2_R11_SNORM;
   case PIPE_FORMAT_ETC2_RG11_UNORM: return MALI_CLUMP_FORMAT_ETC2_RG11_UNORM;
   case PIPE_FORMAT_ETC2_RG11_SNORM: return MALI_CLUMP_FORMAT_ETC2_RG11_SNORM;
   default:
      break;
   }

   switch (util_format_get_blocksize(format)) {
   case 1: return MALI_CLUMP_FORMAT_RAW8;
   case 2: return MALI_CLUMP_FORMAT_RAW16;
   case 3: return MALI_CLUMP_FORMAT_RAW24;
   case 4: return MALI_CLUMP_FORMAT_RAW32;
   case 6: return MALI_CLUMP_FORMAT_RAW48;
   case 8: return MALI_CLUMP_FORMAT_RAW64;
   case 12: return MALI_CLUMP_FORMAT_RAW96;
   default: return MALI_CLUMP_FORMAT_RAW128;
   }
}

static enum mali_astc_2d_dimension
astc_dim(unsigned dim)
{
   switch (dim) {
   case 4: return MALI_ASTC_2D_DIMENSION_4;
   case 5: return MALI_ASTC_2D_DIMENSION_5;
   case 6: return MALI_ASTC_2D_DIMENSION_6;
   case 8: return MALI_ASTC_2D_DIMENSION_8;
   case 10: return MALI_ASTC_2D_DIMENSION_10;
   default: return MALI_ASTC_2D_DIMENSION_12;
   }
}

#endif /* PAN_ARCH >= 9 */

static enum mali_texture_dimension
texture_dim(VkImageViewType type)
{
   switch (type) {
   case VK_IMAGE_VIEW_TYPE_1D:
   case VK_IMAGE_VIEW_TYPE_1D_ARRAY:
      return MALI_TEXTURE_DIMENSION_1D;
   case VK_IMAGE_VIEW_TYPE_3D:
      return MALI_TEXTURE_DIMENSION_3D;
   case VK_IMAGE_VIEW_TYPE_CUBE:
   case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
      return MALI_TEXTURE_DIMENSION_CUBE;
   default:
      return MALI_TEXTURE_DIMENSION_2D;
   }
}

/* ---------------------------------------------------------------------- */
/* AFBC                                                                    */

#if PAN_ARCH >= 9

static enum mali_afbc_compression_mode
afbc_hw_mode(enum mali_afbc_mode m)
{
   switch (m) {
   case MALI_AFBC_R8: return MALI_AFBC_COMPRESSION_MODE_R8;
   case MALI_AFBC_R8G8: return MALI_AFBC_COMPRESSION_MODE_R8G8;
   case MALI_AFBC_R8G8B8A8: return MALI_AFBC_COMPRESSION_MODE_R8G8B8A8;
   case MALI_AFBC_R10G10B10A2: return MALI_AFBC_COMPRESSION_MODE_R10G10B10A2;
   case MALI_AFBC_R11G11B10: return MALI_AFBC_COMPRESSION_MODE_R11G11B10;
#if PAN_ARCH >= 10
   /* v9 has no 16-bit-channel AFBC modes; mali_image.c never makes such
    * an image AFBC there. */
   case MALI_AFBC_R16: return MALI_AFBC_COMPRESSION_MODE_R16;
   case MALI_AFBC_R16G16: return MALI_AFBC_COMPRESSION_MODE_R16G16;
   case MALI_AFBC_R16G16B16A16: return MALI_AFBC_COMPRESSION_MODE_R16G16B16A16;
#endif
   default: UNREACHABLE("no AFBC mode");
   }
}

unsigned
MALI_PER_ARCH(image_afbc_hw_mode)(const struct mali_image *image, unsigned plane)
{
   return afbc_hw_mode(mali_afbc_mode(image->planes[plane].format));
}

/* pan_afbc_decompression_mode: a stencil plane read as stencil. */
unsigned
MALI_PER_ARCH(image_afbc_hw_read_mode)(const struct mali_image *image, unsigned plane,
                                       enum pipe_format view)
{
   if (view == PIPE_FORMAT_S8_UINT)
      return MALI_AFBC_COMPRESSION_MODE_S8;
   return MALI_PER_ARCH(image_afbc_hw_mode)(image, plane);
}

#endif /* PAN_ARCH >= 9 */

/* What a view reads: one plane of the image, in one format. */
struct view_src {
   const struct mali_image *image;
   unsigned plane;
   enum pipe_format format;      /* the view's format for that plane */
   enum mali_texture_dimension dim;
   unsigned first_level, levels;
   unsigned first_layer, layers; /* 3D: first Z slice and slice count */
};

#if PAN_ARCH < 9

/* Surface entries a view's texture has: layers (one for 3D) x samples x
 * levels. */
static unsigned
view_plane_count(const struct view_src *src)
{
   const unsigned layers = src->dim == MALI_TEXTURE_DIMENSION_3D ? 1 : src->layers;
   return layers * MAX2(src->image->vk.samples, 1) * src->levels;
}

/* The ASTC block size tag of a surface pointer (pan_texture.c
 * astc_compression_tag). */
static unsigned
astc_tag(const struct util_format_description *desc)
{
   if (desc->block.depth > 1) {
      static const unsigned dims3d[7] = {
         [3] = MALI_ASTC_3D_DIMENSION_3, [4] = MALI_ASTC_3D_DIMENSION_4,
         [5] = MALI_ASTC_3D_DIMENSION_5, [6] = MALI_ASTC_3D_DIMENSION_6,
      };
      return (dims3d[desc->block.depth] << 4) | (dims3d[desc->block.height] << 2) |
             dims3d[desc->block.width];
   }
   static const unsigned dims2d[13] = {
      [4] = MALI_ASTC_2D_DIMENSION_4,   [5] = MALI_ASTC_2D_DIMENSION_5,
      [6] = MALI_ASTC_2D_DIMENSION_6,   [8] = MALI_ASTC_2D_DIMENSION_8,
      [10] = MALI_ASTC_2D_DIMENSION_10, [12] = MALI_ASTC_2D_DIMENSION_12,
   };
   return (dims2d[desc->block.height] << 3) | dims2d[desc->block.width];
}

/* One Surface With Stride: level `level`, layer or Z slice `layer`,
 * sample `sample` (get_linear_or_u_tiled_surface_props, or
 * get_afbc_surface_props for an AFBC image). */
static void
emit_surface(const struct view_src *src, unsigned level, unsigned layer, unsigned sample,
             void *out)
{
   const struct mali_image *image = src->image;
   const struct mali_image_plane_layout *pl = &image->planes[src->plane].layout;
   const struct mali_image_slice *s = &pl->slices[level];
   const struct util_format_description *desc = util_format_description(src->format);

   if (mali_image_is_afbc(image)) {
      /* get_afbc_surface_props at PAN_ARCH 7: the layer's first header
       * (4 KiB aligned) with the AFBC surface flags in the pointer's low
       * bits, the header row stride, and the surface stride (headers and
       * body), by which the payload range check bounds the headers' body
       * offsets. Read in the plane's own format, sRGB aside. */
      const uint64_t mod = image->vk.drm_format_mod;
      assert(image->vk.image_type == VK_IMAGE_TYPE_2D && sample == 0);
      assert(util_format_linear(src->format) ==
             util_format_linear(image->planes[src->plane].format));
      unsigned flags = MALI_AFBC_SURFACE_FLAG_PREFETCH | MALI_AFBC_SURFACE_FLAG_CHECK_PAYLOAD_RANGE;
      if (mod & AFBC_FORMAT_MOD_YTR)
         flags |= MALI_AFBC_SURFACE_FLAG_YTR;
      if (mod & AFBC_FORMAT_MOD_TILED)
         flags |= MALI_AFBC_SURFACE_FLAG_TILED_HEADER;
      const uint64_t header = image->base + s->offset + (uint64_t)layer * pl->array_stride;
      assert(!(header & (MALI_AFBC_ALIGN - 1)));
      assert(s->surface_stride <= UINT32_MAX);
      pan_cast_and_pack(out, SURFACE_WITH_STRIDE, cfg) {
         cfg.pointer = header | flags;
         cfg.row_stride = s->row_stride;
         cfg.surface_stride = (uint32_t)s->surface_stride;
      }
      return;
   }

   uint64_t addr = image->base + s->offset;
   if (image->vk.image_type == VK_IMAGE_TYPE_3D)
      addr += (uint64_t)layer * s->surface_stride;
   else
      addr += (uint64_t)layer * pl->array_stride + (uint64_t)sample * s->surface_stride;
   if (desc->layout == UTIL_FORMAT_LAYOUT_ASTC)
      addr |= astc_tag(desc);

   assert(s->surface_stride <= UINT32_MAX);
   pan_cast_and_pack(out, SURFACE_WITH_STRIDE, cfg) {
      cfg.pointer = addr;
      cfg.row_stride = s->row_stride;
      cfg.surface_stride = (uint32_t)s->surface_stride;
   }
}

/* The surfaces of a view: layers outer, then samples, levels inner
 * (pan_emit_iview_texture_payload at PAN_ARCH 7). Returns the bytes
 * written. */
static unsigned
emit_planes(const struct view_src *src, uint8_t *out)
{
   const unsigned layers = src->dim == MALI_TEXTURE_DIMENSION_3D ? 1 : src->layers;
   const unsigned samples = MAX2(src->image->vk.samples, 1);
   unsigned n = 0;
   for (unsigned l = 0; l < layers; l++) {
      for (unsigned smp = 0; smp < samples; smp++) {
         for (unsigned lvl = 0; lvl < src->levels; lvl++) {
            emit_surface(src, src->first_level + lvl, src->first_layer + l, smp,
                         out + n * PLANE_DESC_SIZE);
            n++;
         }
      }
   }
   return n * PLANE_DESC_SIZE;
}

#else

/* One plane descriptor: level `level`, layer or Z slice `layer`
 * (get_linear_or_u_tiled_plane_props + emit_generic_plane /
 * emit_astc_plane). */
static void
emit_plane(const struct view_src *src, unsigned level, unsigned layer, void *out)
{
   const struct mali_image *image = src->image;
   const struct mali_image_plane_layout *pl = &image->planes[src->plane].layout;
   const struct mali_image_slice *s = &pl->slices[level];
   const struct util_format_description *desc = util_format_description(src->format);
   const bool linear = image->vk.drm_format_mod == DRM_FORMAT_MOD_LINEAR;
   const enum mali_clump_ordering ordering =
      linear ? MALI_CLUMP_ORDERING_LINEAR : MALI_CLUMP_ORDERING_TILED_U_INTERLEAVED;

   uint64_t addr = image->base + s->offset;
   uint64_t size = s->size;
   uint64_t slice_stride;
   if (image->vk.image_type == VK_IMAGE_TYPE_3D) {
      addr += layer * s->surface_stride;
      size -= layer * s->surface_stride;
      slice_stride = s->surface_stride;
   } else {
      addr += layer * pl->array_stride;
      slice_stride = image->vk.samples > 1 ? s->surface_stride : 0;
   }
#if PAN_ARCH >= 10
   const uint32_t width = u_minify(image->vk.extent.width, level);
   const uint32_t height = u_minify(image->vk.extent.height, level);
#else
   /* 32-bit sizes and strides (no hi words on v9). */
   assert(size <= UINT32_MAX && slice_stride <= UINT32_MAX);
#endif

   if (mali_image_is_afbc(image)) {
      /* emit_afbc_plane: the layer's headers; 2D, single-sampled, so no
       * header slice stride. */
      const uint64_t mod = image->vk.drm_format_mod;
      pan_cast_and_pack(out, AFBC_PLANE, cfg) {
         cfg.superblock_size = MALI_AFBC_SUPERBLOCK_SIZE_16X16;
         cfg.ytr = mod & AFBC_FORMAT_MOD_YTR;
         cfg.split_block = false;
         cfg.tiled_header = mod & AFBC_FORMAT_MOD_TILED;
         cfg.prefetch = true;
         cfg.compression_mode =
            MALI_PER_ARCH(image_afbc_hw_read_mode)(image, src->plane, src->format);
         cfg.size = size & BITFIELD_MASK(32);
         cfg.pointer = addr;
         cfg.header_row_stride = s->row_stride;
         cfg.header_slice_size = s->afbc_header_size;
         cfg.header_slice_stride = 0;
#if PAN_ARCH >= 10
         cfg.size_hi = size >> 32;
         cfg.header_slice_stride_hi = 0;
         cfg.width = width;
         cfg.height = height;
#endif
      }
      return;
   }

   if (desc->layout == UTIL_FORMAT_LAYOUT_ASTC) {
      /* sRGB decodes to RGBA8; the rest to FP16 ("wide"), there being no
       * VK_EXT_astc_decode_mode. */
      pan_cast_and_pack(out, ASTC_2D_PLANE, cfg) {
         cfg.clump_ordering = ordering;
         cfg.decode_hdr = util_format_is_astc_hdr(src->format);
         cfg.decode_wide = desc->colorspace != UTIL_FORMAT_COLORSPACE_SRGB;
         cfg.block_width = astc_dim(desc->block.width);
         cfg.block_height = astc_dim(desc->block.height);
         cfg.size = size & BITFIELD_MASK(32);
         cfg.pointer = addr;
         cfg.row_stride = s->row_stride;
         cfg.slice_stride = slice_stride & BITFIELD_MASK(32);
#if PAN_ARCH >= 10
         cfg.size_hi = size >> 32;
         cfg.slice_stride_hi = slice_stride >> 32;
         cfg.width = width;
         cfg.height = height;
#endif
      }
      return;
   }

   pan_cast_and_pack(out, GENERIC_PLANE, cfg) {
      cfg.clump_ordering = ordering;
      cfg.clump_format = clump_format(src->format);
      cfg.size = size & BITFIELD_MASK(32);
      cfg.pointer = addr;
      cfg.row_stride = s->row_stride;
      cfg.slice_stride = slice_stride & BITFIELD_MASK(32);
#if PAN_ARCH >= 10
      cfg.size_hi = size >> 32;
      cfg.slice_stride_hi = slice_stride >> 32;
      cfg.width = width;
      cfg.height = height;
#endif
   }
}

/* The plane descriptors of a view: layers outer, levels inner
 * (pan_emit_iview_texture_payload). Returns the bytes written. */
static unsigned
emit_planes(const struct view_src *src, uint8_t *out)
{
   const unsigned layers = src->dim == MALI_TEXTURE_DIMENSION_3D ? 1 : src->layers;
   unsigned n = 0;
   for (unsigned l = 0; l < layers; l++) {
      for (unsigned lvl = 0; lvl < src->levels; lvl++) {
         emit_plane(src, src->first_level + lvl, src->first_layer + l,
                    out + n * PLANE_DESC_SIZE);
         n++;
      }
   }
   return n * PLANE_DESC_SIZE;
}

#endif /* PAN_ARCH >= 9 */

/* Extent at the view's base level; an uncompressed view of a compressed
 * image counts blocks (pan_texture_get_extent). */
static VkExtent3D
view_extent(const struct view_src *src)
{
   const struct mali_image *image = src->image;
   VkExtent3D e = {
      .width = u_minify(image->vk.extent.width, src->first_level),
      .height = u_minify(image->vk.extent.height, src->first_level),
      .depth = u_minify(image->vk.extent.depth, src->first_level),
   };
   const enum pipe_format ifmt = image->planes[src->plane].format;
   if (util_format_is_compressed(ifmt) && !util_format_is_compressed(src->format)) {
      e.width = DIV_ROUND_UP(e.width, util_format_get_blockwidth(ifmt));
      e.height = DIV_ROUND_UP(e.height, util_format_get_blockheight(ifmt));
      e.depth = DIV_ROUND_UP(e.depth, util_format_get_blockdepth(ifmt));
   }
   return e;
}

/* `storage` is never set on v7, where storage views are Attribute Buffer
 * pairs (v7_pack_storage_attr_bufs). */
static void
pack_texture(const struct view_src *src, const unsigned char swizzle[4],
             uint64_t planes, float min_lod, bool storage, uint32_t out[8])
{
   const struct mali_image *image = src->image;
   const VkExtent3D e = view_extent(src);
   unsigned array_size = src->dim == MALI_TEXTURE_DIMENSION_3D ? 1 : src->layers;
   if (src->dim == MALI_TEXTURE_DIMENSION_CUBE)
      array_size /= 6;

   unsigned sw = 0;
   for (unsigned i = 0; i < 4; i++)
      sw |= swizzle[i] << (3 * i);

   pan_cast_and_pack(out, TEXTURE, cfg) {
      cfg.dimension = src->dim;
      cfg.format = GENX(pan_format_from_pipe_format)(src->format)->hw;
      cfg.width = e.width;
      cfg.height = e.height;
      if (src->dim == MALI_TEXTURE_DIMENSION_3D)
         cfg.depth = storage ? src->layers : e.depth;
      else
         cfg.sample_count = image->vk.samples;
      cfg.swizzle = sw;
#if PAN_ARCH >= 9
      cfg.texel_interleave = image->vk.drm_format_mod != DRM_FORMAT_MOD_LINEAR ||
                             util_format_is_compressed(src->format);
#else
      /* pan_modifier_to_layout. */
      cfg.texel_ordering = mali_image_is_afbc(image) ? MALI_TEXTURE_LAYOUT_AFBC :
                           image->vk.drm_format_mod == DRM_FORMAT_MOD_LINEAR ?
                              MALI_TEXTURE_LAYOUT_LINEAR : MALI_TEXTURE_LAYOUT_TILED;
#endif
      cfg.levels = src->levels;
      cfg.array_size = array_size;
      cfg.surfaces = planes;
      if (storage) {
         cfg.minimum_lod = 0;
         cfg.maximum_lod = 0;
         cfg.minimum_level = 0;
      } else {
         /* The base level is in the plane pointers; LODs count from it. */
         cfg.minimum_lod = MAX2(0.0f, min_lod - src->first_level);
         cfg.maximum_lod = src->levels - 1;
      }
   }
}

#if PAN_ARCH < 9
/*
 * A v7 storage view (panvk_vX_image_view.c prepare_attr_buf_descs at
 * PAN_ARCH 7): shaders reach storage images through the attribute unit,
 * so its descriptor is an Attribute Buffer (3D Linear or 3D Interleaved)
 * at the view's first level and layer or Z slice, and its 3D
 * continuation. The hardware format belongs in the Attribute descriptor
 * that the descriptor copy of a draw or dispatch writes from this pair
 * (mali_jm_desc_v7.c), so, as panvk does, it is packed into the stride
 * word: the texel size in bits 0..6, log2(samples) in 7..9, the 22-bit
 * format from bit 10. The pointer field drops the low 6 bits: every level
 * and layer offset is a multiple of 64 (mali_image_layout.c).
 */
static void
v7_pack_storage_attr_bufs(const struct mali_image_view *view, uint32_t out[8])
{
   const struct mali_image *image = container_of(view->vk.image, struct mali_image, vk);
   const bool is_3d = image->vk.image_type == VK_IMAGE_TYPE_3D;
   const unsigned plane = 0;
   const struct mali_image_plane_layout *pl = &image->planes[plane].layout;
   const unsigned level = view->vk.base_mip_level;
   const struct mali_image_slice *sl = &pl->slices[level];
   const enum pipe_format format = vk_format_to_pipe_format(view->vk.view_format);
   const unsigned first = is_3d ? view->vk.storage.z_slice_offset : view->vk.base_array_layer;
   const unsigned count = is_3d ? view->vk.storage.z_slice_count : view->vk.layer_count;
   const unsigned samples = MAX2(image->vk.samples, 1);
   const uint64_t layer_stride = is_3d ? sl->surface_stride : pl->array_stride;
   const uint32_t blksize = util_format_get_blocksize(format);
   const uint32_t hw_fmt = GENX(pan_format_from_pipe_format)(format)->hw;

   assert(blksize <= BITFIELD_MASK(7));
   assert(util_logbase2(samples) <= BITFIELD_MASK(3));
   assert(hw_fmt <= BITFIELD_MASK(22));

   struct {
      struct mali_attribute_buffer_packed buf;
      struct mali_attribute_buffer_continuation_3d_packed cont;
   } desc;
   static_assert(sizeof(desc) == 8 * sizeof(uint32_t), "one slot");

   pan_pack(&desc.buf, ATTRIBUTE_BUFFER, cfg) {
      cfg.type = image->vk.drm_format_mod == DRM_FORMAT_MOD_LINEAR
                    ? MALI_ATTRIBUTE_TYPE_3D_LINEAR
                    : MALI_ATTRIBUTE_TYPE_3D_INTERLEAVED;
      cfg.pointer = image->base + sl->offset + (uint64_t)first * layer_stride;
      cfg.stride = blksize | (util_logbase2(samples) << 7) | (hw_fmt << 10);
      /* The bytes from the first layer's level to the end of the last
       * layer's. */
      const uint64_t size = is_3d ? (uint64_t)count * sl->surface_stride
                                  : (uint64_t)(count - 1) * layer_stride + sl->size;
      cfg.size = (uint32_t)MIN2(size, UINT32_MAX);
   }
   pan_pack(&desc.cont, ATTRIBUTE_BUFFER_CONTINUATION_3D, cfg) {
      /* The view's extent at its base level (in blocks for an
       * uncompressed view of a compressed image), as in panvk. */
      cfg.s_dimension = view->vk.extent.width;
      cfg.t_dimension = view->vk.extent.height;
      cfg.r_dimension = count;
      cfg.row_stride = sl->row_stride;
      if (count > 1 || samples > 1)
         cfg.slice_stride = (uint32_t)layer_stride;
      if (samples > 1) {
         cfg.r_dimension *= samples;
         cfg.slice_stride /= samples;
      }
   }
   memcpy(out, &desc, sizeof(desc));
}
#endif /* PAN_ARCH < 9 */

VkResult
MALI_PER_ARCH(image_view_init_descs)(struct mali_device *dev, struct mali_image_view *view)
{
   const struct mali_image *image = container_of(view->vk.image, struct mali_image, vk);
   const VkImageUsageFlags usage = view->vk.usage;
   const bool want_tex = usage & (VK_IMAGE_USAGE_SAMPLED_BIT |
                                  VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT);
#if PAN_ARCH < 9
   /* A v7 storage view is the Attribute Buffer pair above, with no plane
    * descriptors; the code below builds the sampled descriptor only. */
   if ((usage & VK_IMAGE_USAGE_STORAGE_BIT) && !vk_format_is_compressed(view->vk.view_format)) {
      if (!GENX(pan_format_from_pipe_format)(vk_format_to_pipe_format(view->vk.view_format))->hw)
         return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                          "no storage image format for view format %d", view->vk.view_format);
      v7_pack_storage_attr_bufs(view, view->storage_tex);
      view->has_storage_tex = true;
   }
   const bool want_storage = false;
#else
   const bool want_storage = (usage & VK_IMAGE_USAGE_STORAGE_BIT) &&
                             !vk_format_is_compressed(view->vk.view_format);
#endif
   if (!want_tex && !want_storage)
      return VK_SUCCESS;

   struct view_src src = {
      .image = image,
      .dim = texture_dim(view->vk.view_type),
      .first_level = view->vk.base_mip_level,
      .levels = view->vk.level_count,
   };

   /* Depth/stencil views read one plane: depth if the view has the depth
    * aspect (a view with both is only an attachment), else stencil. */
   unsigned char swizzle[4];
   vk_component_mapping_to_pipe_swizzle(view->vk.swizzle, swizzle);
   if (view->vk.aspects & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
      const VkImageAspectFlags aspect = (view->vk.aspects & VK_IMAGE_ASPECT_DEPTH_BIT) ?
                                           VK_IMAGE_ASPECT_DEPTH_BIT :
                                           VK_IMAGE_ASPECT_STENCIL_BIT;
      src.plane = mali_image_aspect_plane(image, aspect);
      src.format = image->planes[src.plane].format;
      /* Vulkan reads depth or stencil as (D, 0, 0, 1). */
      static const unsigned char r001[4] = {
         PIPE_SWIZZLE_X, PIPE_SWIZZLE_0, PIPE_SWIZZLE_0, PIPE_SWIZZLE_1,
      };
      unsigned char composed[4];
      util_format_compose_swizzles(r001, swizzle, composed);
      memcpy(swizzle, composed, sizeof(swizzle));
   } else {
      src.plane = 0;
      src.format = vk_format_to_pipe_format(view->vk.view_format);
   }

   if (!GENX(pan_format_from_pipe_format)(src.format)->hw)
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "no texture format for view format %d", view->vk.view_format);

   if (src.dim == MALI_TEXTURE_DIMENSION_3D) {
      src.first_layer = 0;
      src.layers = u_minify(image->vk.extent.depth, src.first_level);
   } else {
      src.first_layer = view->vk.base_array_layer;
      src.layers = view->vk.layer_count;
   }

   /* A storage view of a 3D image sees its Z-slice range. */
   struct view_src ssrc = src;
   if (src.dim == MALI_TEXTURE_DIMENSION_3D) {
      ssrc.first_layer = view->vk.storage.z_slice_offset;
      ssrc.layers = view->vk.storage.z_slice_count;
   }

#if PAN_ARCH < 9
   const unsigned per_view = view_plane_count(&src) * PLANE_DESC_SIZE;
#else
   const unsigned per_view =
      (src.dim == MALI_TEXTURE_DIMENSION_3D ? 1 : src.layers) * src.levels * PLANE_DESC_SIZE;
#endif
   const uint64_t size = (want_tex ? per_view : 0) + (want_storage ? per_view : 0);

   enum mali_kbase_result r =
      mali_bo_pool_alloc(&dev->desc_pool, size, PLANE_DESC_SIZE, &view->plane_descs);
   if (r != MALI_KBASE_SUCCESS)
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "no memory for %llu bytes of plane descriptors",
                       (unsigned long long)size);

   uint8_t *cpu = view->plane_descs.cpu;
   uint64_t gpu = view->plane_descs.gpu_va;
   if (want_tex) {
      emit_planes(&src, cpu);
      pack_texture(&src, swizzle, gpu, view->vk.min_lod, false, view->tex);
      view->has_tex = true;
      cpu += per_view;
      gpu += per_view;
   }
   if (want_storage) {
      static const unsigned char rgba[4] = {
         PIPE_SWIZZLE_X, PIPE_SWIZZLE_Y, PIPE_SWIZZLE_Z, PIPE_SWIZZLE_W,
      };
      emit_planes(&ssrc, cpu);
      pack_texture(&ssrc, rgba, gpu, 0.0f, true, view->storage_tex);
      view->has_storage_tex = true;
   }
   return VK_SUCCESS;
}

void
MALI_PER_ARCH(image_pack_plane_texture)(const struct mali_image *image, unsigned plane,
                                        enum pipe_format format, unsigned level,
                                        unsigned first_layer, unsigned layers,
                                        void *planes_cpu, uint64_t planes_gpu, uint32_t out[8])
{
   const struct view_src src = {
      .image = image,
      .plane = plane,
      .format = format,
      .dim = MALI_TEXTURE_DIMENSION_2D,
      .first_level = level,
      .levels = 1,
      .first_layer = first_layer,
      .layers = layers,
   };
   static const unsigned char rgba[4] = {
      PIPE_SWIZZLE_X, PIPE_SWIZZLE_Y, PIPE_SWIZZLE_Z, PIPE_SWIZZLE_W,
   };
   emit_planes(&src, planes_cpu);
   pack_texture(&src, rgba, planes_gpu, 0.0f, false, out);
}
