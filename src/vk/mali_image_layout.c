/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Plane layouts for linear, 16x16 u-interleaved and AFBC images, with the
 * rules of Mesa's pan_mod.c (linear, u_tiled and afbc init_slice_layout),
 * pan_afbc.h and pan_layout.c (pan_image_layout_init) for arch 11, without
 * explicit (imported) layouts:
 *
 * - rows and slices are 64-byte aligned (arch >= 7 row alignment for
 *   non-YUV formats, raised to a cache line);
 * - a u-interleaved tile is 16x16 texels, or 4x4 blocks of a compressed
 *   format; a tile row is the tiles covering the width;
 * - multisampled images store the samples like depth slices;
 * - mip levels follow each other; array layers repeat the whole mip chain
 *   at a 64-byte aligned stride; the plane is rounded up to 4 KiB;
 * - with CRC (u-interleaved or AFBC), level 0 is followed (at a 64-byte
 *   boundary) by a 64-byte header and the CRC table, and level 1 starts
 *   after the table.
 * - AFBC (MALI_MOD_AFBC*, single-sampled only): see afbc_slice below.
 *
 * Kept equal to Mesa's so the texture and attachment descriptors of later
 * units, which follow panvk, describe exactly these layouts.
 */

#include "mali_image.h"

#include "drm-uapi/drm_fourcc.h"
#include "util/format/u_format.h"
#include "util/macros.h"
#include "util/u_math.h"

#define ROW_ALIGN 64
#define PLANE_ALIGN 4096

/* ---------------------------------------------------------------------- */
/* AFBC                                                                    */

#define AFBC_SB 16

enum mali_afbc_mode
mali_afbc_mode(enum pipe_format format)
{
   /* sRGB and the channel type do not change the compressed bits on v9+
    * (pan_afbc_format); a depth or stencil plane compresses as the colour
    * format of its size. Only formats a render target or depth buffer of
    * ours can have are listed. */
   switch (util_format_linear(format)) {
   case PIPE_FORMAT_R8_UNORM:
   case PIPE_FORMAT_R8_SNORM:
   case PIPE_FORMAT_R8_UINT:
   case PIPE_FORMAT_R8_SINT:
   case PIPE_FORMAT_S8_UINT:
      return MALI_AFBC_R8;
   case PIPE_FORMAT_R8G8_UNORM:
   case PIPE_FORMAT_R8G8_SNORM:
   case PIPE_FORMAT_R8G8_UINT:
   case PIPE_FORMAT_R8G8_SINT:
      return MALI_AFBC_R8G8;
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_SNORM:
   case PIPE_FORMAT_R8G8B8A8_UINT:
   case PIPE_FORMAT_R8G8B8A8_SINT:
   case PIPE_FORMAT_Z32_FLOAT:
      return MALI_AFBC_R8G8B8A8;
   case PIPE_FORMAT_R10G10B10A2_UNORM:
   case PIPE_FORMAT_R10G10B10A2_UINT:
      return MALI_AFBC_R10G10B10A2;
   case PIPE_FORMAT_R11G11B10_FLOAT:
      return MALI_AFBC_R11G11B10;
   case PIPE_FORMAT_R16_UNORM:
   case PIPE_FORMAT_R16_SNORM:
   case PIPE_FORMAT_R16_UINT:
   case PIPE_FORMAT_R16_SINT:
   case PIPE_FORMAT_R16_FLOAT:
      return MALI_AFBC_R16;
   case PIPE_FORMAT_R16G16_UNORM:
   case PIPE_FORMAT_R16G16_SNORM:
   case PIPE_FORMAT_R16G16_UINT:
   case PIPE_FORMAT_R16G16_SINT:
   case PIPE_FORMAT_R16G16_FLOAT:
      return MALI_AFBC_R16G16;
   case PIPE_FORMAT_R16G16B16A16_UNORM:
   case PIPE_FORMAT_R16G16B16A16_SNORM:
   case PIPE_FORMAT_R16G16B16A16_UINT:
   case PIPE_FORMAT_R16G16B16A16_SINT:
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
      return MALI_AFBC_R16G16B16A16;
   default:
      return MALI_AFBC_NONE;
   }
}

bool
mali_afbc_can_ytr(enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);
   return (desc->nr_channels == 3 || desc->nr_channels == 4) &&
          desc->colorspace == UTIL_FORMAT_COLORSPACE_RGB;
}

unsigned
mali_afbc_header_tile_sbs(enum pipe_format format)
{
   return util_format_get_blocksizebits(format) <= 32 ? 8 : 4;
}

unsigned
mali_afbc_min_extent(enum pipe_format format)
{
   return AFBC_SB * mali_afbc_header_tile_sbs(format) / 2;
}

/*
 * One AFBC surface (pan_mod_afbc_init_slice_layout with tiled headers, no
 * explicit layout): the extent is padded to whole header tiles; a header
 * row covers one row of header tiles and is aligned to 1 KiB (256 bytes
 * above 32 bits per pixel); the headers, 16 bytes per superblock, come
 * first, and the bodies, one uncompressed-size slot per superblock
 * (sparse), start at the next 4 KiB.
 */
static void
afbc_slice(enum pipe_format f, unsigned w, unsigned h, struct mali_image_slice *s)
{
   const unsigned ts = mali_afbc_header_tile_sbs(f);
   const unsigned aw = ALIGN_POT(w, AFBC_SB * ts);
   const unsigned ah = ALIGN_POT(h, AFBC_SB * ts);
   const unsigned row_align = util_format_get_blocksizebits(f) <= 32 ? 1024 : 256;

   s->row_stride = ALIGN_POT((aw / AFBC_SB) * ts * MALI_AFBC_HEADER_BYTES, row_align);
   const unsigned row_sbs = s->row_stride / (MALI_AFBC_HEADER_BYTES * ts);
   const uint64_t sbs = (uint64_t)row_sbs * (ah / AFBC_SB);
   const uint64_t payload = AFBC_SB * AFBC_SB * util_format_get_blocksize(f);

   s->afbc_header_size = sbs * MALI_AFBC_HEADER_BYTES;
   s->surface_stride = ALIGN_POT((uint64_t)s->afbc_header_size, MALI_AFBC_ALIGN) + sbs * payload;
}

/* ---------------------------------------------------------------------- */

bool
mali_image_plane_layout_init(const struct mali_image_layout_info *info,
                             uint64_t offset, struct mali_image_plane_layout *layout)
{
   const enum pipe_format f = info->format;
   const bool tiled = info->modifier == DRM_FORMAT_MOD_ARM_16X16_BLOCK_U_INTERLEAVED;
   const bool afbc = mali_mod_is_afbc(info->modifier);

   if (!tiled && !afbc && info->modifier != DRM_FORMAT_MOD_LINEAR)
      return false;
   if (afbc && (mali_afbc_mode(f) == MALI_AFBC_NONE || info->samples != 1 ||
                ((info->modifier & AFBC_FORMAT_MOD_YTR) && !mali_afbc_can_ytr(f))))
      return false;
   if (util_format_get_num_planes(f) != 1 || info->levels == 0 ||
       info->levels > MALI_IMAGE_MAX_LEVELS || !info->width || !info->height ||
       !info->depth || !info->samples || !info->layers ||
       (info->depth > 1 && info->samples > 1))
      return false;

   const unsigned bw = util_format_get_blockwidth(f);
   const unsigned bh = util_format_get_blockheight(f);
   const unsigned bd = util_format_get_blockdepth(f);
   const unsigned block_bytes = util_format_get_blocksize(f);
   const bool compressed = util_format_is_compressed(f);

   /* U-interleaved tile in blocks: 4x4 for compressed formats, else 16x16
    * texels (pan_u_interleaved_tile_size_el). */
   unsigned tile_w = 1, tile_h = 1;
   if (tiled) {
      if (compressed) {
         tile_w = tile_h = 4;
      } else {
         if (16 % bw || 16 % bh)
            return false;
         tile_w = 16 / bw;
         tile_h = 16 / bh;
      }
   }
   const unsigned tile_bytes = tile_w * tile_h * block_bytes;

   *layout = (struct mali_image_plane_layout){0};
   uint64_t pos = ALIGN_POT(offset, afbc ? MALI_AFBC_ALIGN : ROW_ALIGN);
   const uint64_t start = pos;

   for (unsigned l = 0; l < info->levels; l++) {
      struct mali_image_slice *s = &layout->slices[l];
      const unsigned w_el = DIV_ROUND_UP(u_minify(info->width, l), bw);
      const unsigned h_el = DIV_ROUND_UP(u_minify(info->height, l), bh);
      const unsigned d_el = DIV_ROUND_UP(u_minify(info->depth, l), bd);

      if (afbc) {
         s->offset = ALIGN_POT(pos, MALI_AFBC_ALIGN);
         afbc_slice(f, w_el, h_el, s);
         s->size = s->surface_stride * d_el;
      } else {
         s->offset = ALIGN_POT(pos, ROW_ALIGN);
         if (tiled) {
            s->row_stride = ALIGN_POT(tile_bytes * DIV_ROUND_UP(w_el, tile_w), ROW_ALIGN);
            s->surface_stride = ALIGN_POT((uint64_t)s->row_stride * DIV_ROUND_UP(h_el, tile_h),
                                          ROW_ALIGN);
         } else {
            s->row_stride = ALIGN_POT(w_el * block_bytes, ROW_ALIGN);
            s->surface_stride = ALIGN_POT((uint64_t)s->row_stride * h_el, ROW_ALIGN);
         }
         s->size = s->surface_stride * d_el * info->samples;
      }
      pos = s->offset + s->size;

      if (l == 0 && info->crc) {
         /* 8 bytes per 16x16 tile, the tile rows and columns rounded up
          * to whole 32x32 regions. */
         const unsigned tiles_per_region = MALI_CRC_REGION / 16;
         const unsigned tx = tiles_per_region * DIV_ROUND_UP(info->width, MALI_CRC_REGION);
         const unsigned ty = tiles_per_region * DIV_ROUND_UP(info->height, MALI_CRC_REGION);
         const uint64_t header = ALIGN_POT(pos, 64);
         layout->crc.header_offset = header;
         layout->crc.row_stride = ALIGN_POT(tx * 8, 64);
         layout->crc.size = layout->crc.row_stride * ty;
         /* The slice's size stays the pixel data only, so texture
          * descriptors and vkGetImageSubresourceLayout do not cover the
          * CRC state. */
         pos = header + MALI_CRC_HEADER_SIZE + layout->crc.size;
      }
   }

   layout->array_stride = ALIGN_POT(pos - start, ROW_ALIGN);
   layout->data_size = ALIGN_POT(layout->array_stride * info->layers, PLANE_ALIGN);
   return true;
}
