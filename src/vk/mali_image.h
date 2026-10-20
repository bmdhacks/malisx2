/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * VkImage, VkImageView, image layouts and format support. Layouts follow
 * Mesa's rules for linear, 16x16 u-interleaved and AFBC images (pan_mod.c,
 * pan_layout.c, pan_afbc.h) because the texture and render-target
 * descriptors are panvk's. AFBC is used for render targets and depth
 * buffers only. u-interleaved colour render targets can carry a CRC table
 * for transaction elimination.
 */

#ifndef MALI_IMAGE_H
#define MALI_IMAGE_H

#include <stdbool.h>
#include <stdint.h>

#include "drm-uapi/drm_fourcc.h"
#include "util/format/u_formats.h"
#include "vk_image.h"

#include "mali_arch.h"
#include "mali_bo_pool.h"

struct mali_device_memory;
struct mali_physical_device;

/* Depth and stencil of a combined format live in two planes (as panvk and
 * the blob do). Multi-planar YUV is not supported. */
#define MALI_IMAGE_MAX_PLANES 2

/* log2(16384) + 1: maxImageDimension* is 16384. */
#define MALI_IMAGE_MAX_LEVELS 15

/* Images are placed at 4 KiB in memory, as panvk does with the same layout
 * rules (the blob uses 64 bytes with its own layouts). */
#define MALI_IMAGE_ALIGNMENT 4096

/*
 * The AFBC layouts we use: 16x16 superblocks, sparse (every superblock has
 * a fixed body slot, so the GPU can write any of them in any order), tiled
 * headers (headers of 8x8 superblocks together), solid-colour blocks
 * allowed; with the YUV-like transform for 3- and 4-channel RGB formats.
 * These are the modifiers panvk picks for a non-WSI render target on v11
 * (pan_format.h PAN_SUPPORTED_MODIFIERS, pan_mod_afbc_test_props).
 */
#define MALI_MOD_AFBC                                                            \
   DRM_FORMAT_MOD_ARM_AFBC(AFBC_FORMAT_MOD_BLOCK_SIZE_16x16 | AFBC_FORMAT_MOD_TILED | \
                           AFBC_FORMAT_MOD_SC | AFBC_FORMAT_MOD_SPARSE)
#define MALI_MOD_AFBC_YTR (MALI_MOD_AFBC | AFBC_FORMAT_MOD_YTR)

static inline bool
mali_mod_is_afbc(uint64_t mod)
{
   return mod == MALI_MOD_AFBC || mod == MALI_MOD_AFBC_YTR;
}

/* How a plane format is compressed (pan_afbc_format for arch 11, reduced to
 * the formats we let AFBC take); MALI_AFBC_NONE when it cannot be. The
 * hardware enum is picked from this in mali_image_view.c. */
enum mali_afbc_mode {
   MALI_AFBC_NONE = 0,
   MALI_AFBC_R8,
   MALI_AFBC_R8G8,
   MALI_AFBC_R8G8B8A8,
   MALI_AFBC_R10G10B10A2,
   MALI_AFBC_R11G11B10,
   MALI_AFBC_R16,
   MALI_AFBC_R16G16,
   MALI_AFBC_R16G16B16A16,
};

enum mali_afbc_mode mali_afbc_mode(enum pipe_format format);

/* Whether the YUV-like transform applies to a format (pan_afbc_can_ytr). */
bool mali_afbc_can_ytr(enum pipe_format format);

/* Bytes of header per 16x16 superblock (AFBC_HEADER_BYTES_PER_TILE in
 * pan_afbc.h). */
#define MALI_AFBC_HEADER_BYTES 16

/* Superblocks per side of a header tile (pan_afbc_tile_size with tiled
 * headers): 8, or 4 above 32 bits per pixel. */
unsigned mali_afbc_header_tile_sbs(enum pipe_format format);

/* Width and height below which a plane is not worth AFBC with tiled
 * headers (pan_mod_afbc_test_props: half a header tile of 8x8 superblocks,
 * 4x4 above 32 bits per pixel). */
unsigned mali_afbc_min_extent(enum pipe_format format);

/* Headers and bodies start 4 KiB aligned with tiled headers
 * (pan_afbc_header_align / pan_afbc_body_offset). */
#define MALI_AFBC_ALIGN 4096

/* One mip level of one plane. The fields mean what the same fields of
 * Mesa's pan_image_slice_layout mean for linear and u-interleaved images;
 * for AFBC, what its afbc fields mean. */
struct mali_image_slice {
   uint64_t offset;          /* from the image's start, plane offset included;
                                AFBC: the first header */
   uint64_t size;            /* all depth slices and samples of the level */
   uint32_t row_stride;      /* bytes between rows of 16x16 tiles (u-interleaved)
                                or rows of blocks (linear); AFBC: bytes of
                                header per row of header tiles */
   uint64_t surface_stride;  /* bytes between depth slices or samples; AFBC:
                                headers and body of one surface */
   uint32_t afbc_header_size; /* AFBC: bytes of header of one surface; the
                                 body starts at the next 4 KiB */
};

/* The CRC state of level 0 (Mesa pan_layout.c init_slice_crc_info): a
 * 64-byte header, then one 8-byte CRC per 16x16 tile, in rows padded to
 * 32x32 regions (the hardware prefetches CRCs by region on v11). The
 * header's first word holds the CRC seed (see mali_cmd_render.c). */
#define MALI_CRC_HEADER_SIZE 64
#define MALI_CRC_REGION 32

struct mali_image_crc_layout {
   uint64_t header_offset;   /* from the image's start; 0 = no CRC */
   uint32_t row_stride;      /* bytes between rows of 16x16 tiles */
   uint32_t size;            /* bytes of the table, header not included */
};
static inline uint32_t
mali_afbc_body_offset(const struct mali_image_slice *s)
{
   return (s->afbc_header_size + MALI_AFBC_ALIGN - 1) & ~(uint32_t)(MALI_AFBC_ALIGN - 1);
}

struct mali_image_plane_layout {
   struct mali_image_slice slices[MALI_IMAGE_MAX_LEVELS];
   uint64_t array_stride;    /* bytes between array layers */
   uint64_t data_size;       /* bytes of the plane, all layers */
   struct mali_image_crc_layout crc;
};

/* What a plane layout depends on. */
struct mali_image_layout_info {
   enum pipe_format format;
   uint64_t modifier;        /* DRM_FORMAT_MOD_LINEAR,
                                DRM_FORMAT_MOD_ARM_16X16_BLOCK_U_INTERLEAVED
                                or MALI_MOD_AFBC(_YTR) */
   uint32_t width, height, depth;
   uint32_t samples;
   uint32_t levels;
   uint32_t layers;
   bool crc;                 /* add a CRC table after level 0 */
};

/*
 * Lay out one plane starting at offset (a multiple of 64). Returns false
 * for a modifier or format the rules do not cover.
 */
bool mali_image_plane_layout_init(const struct mali_image_layout_info *info,
                                  uint64_t offset,
                                  struct mali_image_plane_layout *layout);

struct mali_image_plane {
   enum pipe_format format;
   struct mali_image_plane_layout layout;
};

struct mali_image {
   struct vk_image vk;   /* vk.drm_format_mod holds the chosen layout */

   uint8_t plane_count;
   struct mali_image_plane planes[MALI_IMAGE_MAX_PLANES];
   uint64_t size;        /* bytes of memory the image needs */

   /* Set by vkBindImageMemory: the memory and the GPU address of the
    * image's start. Plane p, level l begins at
    * base + planes[p].layout.slices[l].offset. */
   struct mali_device_memory *mem;
   uint64_t mem_offset;
   uint64_t base;

   /* A swapchain image's imported gralloc buffer, owned by the image and
    * freed with it (mali_wsi.c). */
   struct mali_device_memory *wsi_mem;

   /* GPU address of plane 0's CRC header once the image is bound and its
    * CRC state initialized; 0 when the image has no CRC. Render passes
    * enable transaction elimination on it; every other write of level 0
    * bumps the seed in the header (mali_cmd_crc_invalidate). */
   uint64_t crc_header;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(mali_image, vk.base, VkImage, VK_OBJECT_TYPE_IMAGE)

/* Should an image get a CRC table (mali_image.c)? Exposed for the tests. */
bool mali_image_wants_crc(const VkImageCreateInfo *info, uint64_t modifier);
static inline bool
mali_image_is_afbc(const struct mali_image *image)
{
   return mali_mod_is_afbc(image->vk.drm_format_mod);
}

/* The hardware AFBC compression mode (enum mali_afbc_compression_mode) of
 * a plane of an AFBC image, and the one a texture read of it as `view`
 * uses (a stencil plane read as S8 has its own). mali_image_view.c. */
MALI_PER_ARCH_DECL(unsigned, image_afbc_hw_mode,
                   (const struct mali_image *image, unsigned plane));
MALI_PER_ARCH_DECL(unsigned, image_afbc_hw_read_mode,
                   (const struct mali_image *image, unsigned plane, enum pipe_format view));

/* The plane an aspect lives in. */
static inline unsigned
mali_image_aspect_plane(const struct mali_image *image, VkImageAspectFlags aspect)
{
   if (aspect == VK_IMAGE_ASPECT_STENCIL_BIT)
      return image->plane_count - 1;
   return 0;
}

/*
 * A view keeps the runtime's view state and, for views a shader can read,
 * the Texture descriptors descriptor sets copy (mali_image_view.c).
 * Attachment descriptors are built by render passes.
 */
struct mali_image_view {
   struct vk_image_view vk;

   /* Texture descriptor for sampled images, input attachments and the
    * image half of combined image/samplers (views with SAMPLED or
    * INPUT_ATTACHMENT usage), and its storage-image form (STORAGE usage).
    * Zero when the view has no such usage. Each points at plane
    * descriptors in plane_descs. */
   uint32_t tex[8];
   uint32_t storage_tex[8];
   bool has_tex, has_storage_tex;
   struct mali_bo_ref plane_descs;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(mali_image_view, vk.base, VkImageView,
                               VK_OBJECT_TYPE_IMAGE_VIEW)

/* Build a view's Texture descriptors (mali_image_view.c, per arch); called
 * by vkCreateImageView. The image must be bound. vkDestroyImageView frees
 * plane_descs. */
struct mali_device;
MALI_PER_ARCH_DECL(VkResult, image_view_init_descs,
                   (struct mali_device *dev, struct mali_image_view *view));

/* Bytes of plane descriptors behind a Texture descriptor with `layers`
 * layers of one level of `image`: one 32-byte plane descriptor per layer,
 * or on v7 one 16-byte Surface With Stride per layer and sample
 * (mali_image_view.c). Per-arch code only. */
#if defined(PAN_ARCH)
static inline unsigned
mali_image_plane_texture_desc_size(const struct mali_image *image, unsigned layers)
{
#if PAN_ARCH < 9
   return layers * MAX2(image->vk.samples, 1) * 16;
#else
   return layers * 32;
#endif
}
#endif

/* Internal reads of an image plane (render-pass preloads, blits): a 2D
 * Texture descriptor for one level and `layers` layers from first_layer,
 * read as `format` with an identity swizzle. The plane descriptors
 * (mali_image_plane_texture_desc_size bytes, 32-byte aligned, 16-byte on
 * v7) are written to planes_cpu / planes_gpu. */
MALI_PER_ARCH_DECL(void, image_pack_plane_texture,
                   (const struct mali_image *image, unsigned plane, enum pipe_format format,
                    unsigned level, unsigned first_layer, unsigned layers, void *planes_cpu,
                    uint64_t planes_gpu, uint32_t out[8]));

/* ---------------------------------------------------------------------- */
/* Formats (mali_formats.c)                                                */

/*
 * Image features of a format for linear and optimal tiling (the same set),
 * and buffer features. From Mesa's format table for the device's arch and
 * the GPU's compressed-format bits, the way panvk derives them. The
 * mali_format_* functions pick the arch's table (mali_format_table.c,
 * built per arch).
 */
VkFormatFeatureFlags mali_format_image_features(const struct mali_physical_device *pdev,
                                                VkFormat format);
VkFormatFeatureFlags mali_format_buffer_features(const struct mali_physical_device *pdev,
                                                 VkFormat format);
MALI_PER_ARCH_DECL(VkFormatFeatureFlags, format_image_features,
                   (const struct mali_physical_device *pdev, VkFormat format));
MALI_PER_ARCH_DECL(VkFormatFeatureFlags, format_buffer_features,
                   (const struct mali_physical_device *pdev, VkFormat format));

/* The formats of each plane of an image of this format; returns the plane
 * count, 0 for a format images cannot have. */
unsigned mali_format_planes(VkFormat format, enum pipe_format planes[MALI_IMAGE_MAX_PLANES]);

/* Sample counts for framebuffer formats of this many bytes per pixel
 * (defined in mali_physical_device.c). */
VkSampleCountFlags mali_physical_device_sample_counts(const struct mali_physical_device *pdev,
                                                      unsigned bytes_per_pixel);

/* The colour tile buffer in bytes (16 KiB on the G52, 8 KiB on the G52 r1;
 * mali_physical_device.c). */
unsigned mali_physical_device_tilebuf_color_bytes(const struct mali_physical_device *pdev);

#endif
