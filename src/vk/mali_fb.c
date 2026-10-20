/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The parts of a render pass that do not depend on how the GPU is fed
 * (command streams on v11, job chains on v7 and v9): the pass's tile-buffer
 * layout, bounding box and preloads, the transaction elimination (CRC)
 * target, the frame shaders that preload and clear, the framebuffer
 * descriptors with their render targets and depth/stencil extension, the
 * pass's Local Storage descriptor, and vkCmdBeginRendering /
 * vkCmdEndRendering. The back half (mali_cmd_render.c on v11) does the
 * tiler side and the fragment jobs.
 *
 * Descriptor shapes follow panvk (lib/pan_fb.c, pan_desc.c,
 * csf/panvk_vX_cmd_draw.c get_fb_descs); see mali_cmd_render.c for where
 * the blob's choices were taken instead. MSAA and multiview are not used.
 */

#include "mali_cmd_state.h"

#include "drm-uapi/drm_fourcc.h"
#include "util/format_srgb.h"
#include "util/rounding.h"
#include "util/u_pack_color.h"
#include "vk_log.h"

#include "model/pan_model.h"

#include "mali_arch.h"
#include "mali_image.h"

/* Frame shader DCD slots (panvk: colour preload 0, depth/stencil 1). On v7
 * every layer has its own three (mali_cmd_gfx.h). */
#define FS_DCD_COLOR 0
#define FS_DCD_ZS    1
#define FS_DCD_COUNT MALI_FRAME_SHADER_DCDS

/* AFBC superblock size: the GPU writes whole superblocks of an AFBC target. */
#define AFBC_BLOCK 16

/* ---------------------------------------------------------------------- */
/* Formats                                                                 */

static enum mali_zs_format
zs_write_format(enum pipe_format f)
{
   switch (f) {
   case PIPE_FORMAT_Z16_UNORM: return MALI_ZS_FORMAT_D16;
   case PIPE_FORMAT_Z24X8_UNORM: return MALI_ZS_FORMAT_D24X8;
   case PIPE_FORMAT_Z24_UNORM_S8_UINT: return MALI_ZS_FORMAT_D24S8;
   default: return MALI_ZS_FORMAT_D32;
   }
}

static enum mali_z_internal_format
z_internal_format(enum pipe_format f)
{
   switch (f) {
   case PIPE_FORMAT_Z16_UNORM: return MALI_Z_INTERNAL_FORMAT_D16;
   case PIPE_FORMAT_Z24X8_UNORM:
   case PIPE_FORMAT_Z24_UNORM_S8_UINT: return MALI_Z_INTERNAL_FORMAT_D24;
   default: return MALI_Z_INTERNAL_FORMAT_D32;
   }
}

/* Bytes a colour target takes per pixel in the tile buffer. */
static unsigned
tib_bytes(enum pipe_format f)
{
   const struct pan_blendable_format *bf = GENX(pan_blendable_format_from_pipe_format)(f);
   return pan_format_tib_size(f, bf->internal);
}

static enum mali_color_format
raw_writeback_format(unsigned bits)
{
   switch (bits) {
   case 8: return MALI_COLOR_FORMAT_RAW8;
   case 16: return MALI_COLOR_FORMAT_RAW16;
   case 24: return MALI_COLOR_FORMAT_RAW24;
   case 32: return MALI_COLOR_FORMAT_RAW32;
   case 48: return MALI_COLOR_FORMAT_RAW48;
   case 64: return MALI_COLOR_FORMAT_RAW64;
   case 96: return MALI_COLOR_FORMAT_RAW96;
   default: return MALI_COLOR_FORMAT_RAW128;
   }
}

/* Writeback format, tile-buffer format and swizzle of a render target
 * (pan_desc.c get_rt_formats: the render swizzle is the inverse of the
 * texture one). */
static void
rt_formats(enum pipe_format f, uint32_t *writeback, uint32_t *internal, uint32_t *swizzle)
{
   const struct util_format_description *desc = util_format_description(f);
   const struct pan_blendable_format *bf = GENX(pan_blendable_format_from_pipe_format)(f);
   unsigned char sw[4] = {PIPE_SWIZZLE_X, PIPE_SWIZZLE_Y, PIPE_SWIZZLE_Z, PIPE_SWIZZLE_W};

   if (bf->internal) {
      *internal = bf->internal;
      *writeback = bf->writeback;
      for (unsigned c = 0; c < 4; c++) {
         if (desc->swizzle[c] < 4)
            sw[desc->swizzle[c]] = c;
      }
   } else {
      /* Raw formats, the internal size a power of two from RAW8. */
      unsigned bits = desc->block.bits;
      *internal = MALI_COLOR_BUFFER_INTERNAL_FORMAT_RAW8 + util_logbase2_ceil(bits) - 3;
      *writeback = raw_writeback_format(bits);
   }
   /* Mali channel numbers are the pipe swizzle values. */
   *swizzle = sw[0] | sw[1] << 3 | sw[2] << 6 | sw[3] << 9;
}

/* The tile-buffer layout of a clear colour (pan_clear.c). */
struct tib_layout {
   unsigned int_r, frac_r, int_g, frac_g, int_b, frac_b, int_a, frac_a;
};

static const struct tib_layout tib_layouts[] = {
   [MALI_COLOR_BUFFER_INTERNAL_FORMAT_R8G8B8A8] = {8, 0, 8, 0, 8, 0, 8, 0},
   [MALI_COLOR_BUFFER_INTERNAL_FORMAT_R10G10B10A2] = {10, 0, 10, 0, 10, 0, 2, 0},
   [MALI_COLOR_BUFFER_INTERNAL_FORMAT_R8G8B8A2] = {8, 2, 8, 2, 8, 2, 2, 0},
   [MALI_COLOR_BUFFER_INTERNAL_FORMAT_R4G4B4A4] = {4, 4, 4, 4, 4, 4, 4, 4},
   [MALI_COLOR_BUFFER_INTERNAL_FORMAT_R5G6B5A0] = {5, 5, 6, 4, 5, 5, 0, 2},
   [MALI_COLOR_BUFFER_INTERNAL_FORMAT_R5G5B5A1] = {5, 5, 5, 5, 5, 5, 1, 1},
};

static uint32_t
to_fixed(float f, unsigned bits_int, unsigned bits_frac)
{
   uint32_t m = (1u << bits_int) - 1;
   return (uint32_t)_mesa_roundevenf(f * (float)m) << bits_frac;
}

/* A clear value in the render target's tile-buffer format (pan_pack_color,
 * not dithered). */
static void
pack_clear_color(enum pipe_format f, const VkClearColorValue *v, uint32_t out[4])
{
   const struct pan_blendable_format *bf = GENX(pan_blendable_format_from_pipe_format)(f);
   const unsigned size = util_format_get_blocksize(f);

   if (util_format_is_float(f) || !bf->internal || util_format_is_pure_integer(f)) {
      union util_color c = {0};
      if (util_format_is_pure_integer(f)) {
         util_format_pack_rgba(f, &c, v->uint32, 1);
      } else {
         util_pack_color(v->float32, f, &c);
      }
      if (size == 1) {
         uint32_t b = c.ui[0] & 0xff;
         b |= b << 8;
         b |= b << 16;
         out[0] = out[1] = out[2] = out[3] = b;
      } else if (size == 2) {
         uint32_t w = c.ui[0] & 0xffff;
         out[0] = out[1] = out[2] = out[3] = w | w << 16;
      } else if (size <= 4) {
         out[0] = out[1] = out[2] = out[3] = c.ui[0];
      } else if (size <= 8) {
         out[0] = out[2] = c.ui[0];
         out[1] = out[3] = c.ui[1];
      } else {
         memcpy(out, c.ui, 16);
      }
      return;
   }

   float r = SATURATE(v->float32[0]), g = SATURATE(v->float32[1]);
   float b = SATURATE(v->float32[2]), a = SATURATE(v->float32[3]);
   if (!util_format_has_alpha(f))
      a = 1.0f;
   if (util_format_is_srgb(f)) {
      r = util_format_linear_to_srgb_float(r);
      g = util_format_linear_to_srgb_float(g);
      b = util_format_linear_to_srgb_float(b);
   }
   assert(bf->internal < ARRAY_SIZE(tib_layouts));
   const struct tib_layout l = tib_layouts[bf->internal];
   const unsigned cr = l.int_r + l.frac_r;
   const unsigned cg = cr + l.int_g + l.frac_g;
   const unsigned cb = cg + l.int_b + l.frac_b;
   uint32_t w = to_fixed(r, l.int_r, l.frac_r) | to_fixed(g, l.int_g, l.frac_g) << cr |
                to_fixed(b, l.int_b, l.frac_b) << cg | to_fixed(a, l.int_a, l.frac_a) << cb;
   out[0] = out[1] = out[2] = out[3] = w;
}

void
MALI_PER_ARCH(pack_opaque_blend)(enum pipe_format format, unsigned rt, uint32_t out[4])
{
   pan_cast_and_pack(out, BLEND, cfg) {
      cfg.round_to_fb_precision = true;
      cfg.srgb = util_format_is_srgb(format);
      cfg.internal.mode = MALI_BLEND_MODE_OPAQUE;
      cfg.equation.rgb.a = MALI_BLEND_OPERAND_A_SRC;
      cfg.equation.rgb.b = MALI_BLEND_OPERAND_B_SRC;
      cfg.equation.rgb.c = MALI_BLEND_OPERAND_C_ZERO;
      cfg.equation.alpha.a = MALI_BLEND_OPERAND_A_SRC;
      cfg.equation.alpha.b = MALI_BLEND_OPERAND_B_SRC;
      cfg.equation.alpha.c = MALI_BLEND_OPERAND_C_ZERO;
      cfg.equation.color_mask = 0xf;
      cfg.internal.fixed_function.num_comps = 4;
      cfg.internal.fixed_function.conversion.memory_format =
         GENX(pan_dithered_format_from_pipe_format)(format, false);
      cfg.internal.fixed_function.rt = rt;
   }
}

/* ---------------------------------------------------------------------- */
/* Begin                                                                   */

/* Width and height of a tile of `pixels` (a power of two). */
static void
tile_dims(uint32_t pixels, uint32_t *w, uint32_t *h)
{
   unsigned log2 = util_logbase2(pixels);
   *w = 1u << DIV_ROUND_UP(log2, 2);
   *h = pixels / *w;
}

/* The colour and depth tile buffer sizes of every arch 9 part. Mesa's
 * model table lists the G57 and G68 with these sizes and has no G77, G78
 * or G78AE entry, and Arm's drivers use one 16 KiB colour budget for all
 * arch 9 parts. The table is not consulted on v9: a part it lacks would
 * get the v11 budgets below, twice these. */
#define V9_TILEBUF_COLOR 16384
#define V9_TILEBUF_Z     8192

/* The depth tile buffer size of the G52 and the G52 r1 (Mesa's model
 * table). Their colour tile buffers differ (mali_physical_device.c). */
#define V7_TILEBUF_Z     8192

/* pan_select_fb_tile_size for single-sampled framebuffers: the largest
 * power-of-two tile whose colour and depth fit half the tile buffer (the
 * other half lets the GPU pipeline tiles). */
static void
select_tile_size(struct mali_cmd_buffer *cmd, struct mali_render_state *r)
{
   uint32_t rt_budget, z_budget;
   if (PAN_ARCH == 7) {
      /* The colour tile buffer is 16 KiB on the G52 and 8 KiB on the G52
       * r1 (Mesa's model table), the depth tile buffer 8 KiB on both. Half
       * of each, as pan_query_optimal_tib_size does on v7. */
      rt_budget = mali_physical_device_tilebuf_color_bytes(mali_device_physical(cmd->dev)) / 2;
      z_budget = V7_TILEBUF_Z / 2;
   } else if (PAN_ARCH < 10) {
      rt_budget = V9_TILEBUF_COLOR / 2;
      z_budget = V9_TILEBUF_Z / 2;
   } else {
      const struct mali_physical_device *pdev = mali_device_physical(cmd->dev);
      const struct pan_model *model = pan_get_model(pdev->props.gpu_id, 0);
      rt_budget = model ? model->tilebuffer.color_size / 2 : 16384;
      z_budget = model ? model->tilebuffer.z_size / 2 : 8192;
   }

   uint32_t rt_bytes = 0;
   for (unsigned i = 0; i < r->rt_count; i++) {
      if (r->desc.rt[i].image)
         rt_bytes += tib_bytes(r->desc.rt[i].format);
   }
   rt_bytes = MAX2(rt_bytes, 1);

   uint32_t tile = MIN2(rt_budget >> util_logbase2_ceil(rt_bytes), z_budget >> 2);
   if (tile < 16)
      tile *= 2;
   /* The effective tile size limit (pan_max_effective_tile_size): 32x32 on
    * v10/v11, 16x16 on v7 and v9. */
   tile = MIN2(tile, PAN_ARCH >= 10 ? 32 * 32 : 16 * 16);
   r->tile_size = tile;
   r->cbuf_alloc = ALIGN_POT(rt_bytes * tile, 1024);

   uint32_t off = 0;
   for (unsigned i = 0; i < r->rt_count; i++) {
      r->rt_offset[i] = off;
      if (r->desc.rt[i].image)
         off += tib_bytes(r->desc.rt[i].format) * tile;
   }
}

static bool
target_needs_border_load(const struct mali_fb_target *t)
{
   return t->image && t->load != MALI_ATT_LOAD_LOAD;
}

static bool
target_is_afbc(const struct mali_fb_target *t)
{
   return t->image && mali_image_is_afbc(t->image);
}

bool
MALI_PER_ARCH(fb_begin)(struct mali_cmd_buffer *cmd, const struct mali_render_desc *desc)
{
   struct mali_render_state *r = &cmd->gfx.render;

   memset(r, 0, sizeof(*r));
   r->active = true;
   r->first_provoking_vertex = true;
   r->desc = *desc;
   r->rt_count = MAX2(desc->rt_count, 1);
   r->crc_rt = -1;
   cmd->passes++;

   select_tile_size(cmd, r);

   /* Bounding box: the render area inside the framebuffer. */
   const VkRect2D *a = &desc->area;
   r->minx = MIN2((uint32_t)MAX2(a->offset.x, 0), desc->width - 1);
   r->miny = MIN2((uint32_t)MAX2(a->offset.y, 0), desc->height - 1);
   r->maxx = MIN2((uint32_t)MAX2(a->offset.x, 0) + a->extent.width, desc->width) - 1;
   r->maxy = MIN2((uint32_t)MAX2(a->offset.y, 0) + a->extent.height, desc->height) - 1;
   if (r->maxx < r->minx)
      r->maxx = r->minx;
   if (r->maxy < r->miny)
      r->maxy = r->miny;

   struct mali_fb_target *targets[10];
   unsigned n = 0;
   for (unsigned i = 0; i < desc->rt_count; i++)
      targets[n++] = &r->desc.rt[i];
   targets[n++] = &r->desc.z;
   targets[n++] = &r->desc.s;

   /* The fragment job writes back whole tiles; pixels of a partly covered
    * tile outside the render area must keep their contents, so a target
    * that is not loaded anyway gets loaded there (panvk's border load).
    * An AFBC target is written in whole superblocks, so the bounding box
    * grows to them and a tile smaller than a superblock counts as one. */
   uint32_t tw, th;
   tile_dims(r->tile_size, &tw, &th);
#if PAN_ARCH < 9
   /* pan_force_clean_write_on: tiles smaller than an AFBC superblock share
    * superblocks, which only compress right when every tile of the target
    * is written back, clean or not; the frame shaders then run on every
    * tile (fix_frame_mode). Only several or wide targets make the v7 tile
    * that small. */
   if (tw < AFBC_BLOCK || th < AFBC_BLOCK) {
      for (unsigned i = 0; i < desc->rt_count; i++) {
         struct mali_fb_target *t = &r->desc.rt[i];
         if (target_is_afbc(t) && t->store)
            t->always_write = true;
      }
   }
#endif
   bool afbc_store = false;
   for (unsigned i = 0; i < n; i++)
      afbc_store |= target_is_afbc(targets[i]) && targets[i]->store;
   if (afbc_store) {
      tw = MAX2(tw, AFBC_BLOCK);
      th = MAX2(th, AFBC_BLOCK);
   }
   r->partial_tiles = (r->minx % tw) || (r->miny % th) ||
                      ((r->maxx + 1) % tw && r->maxx + 1 != desc->width) ||
                      ((r->maxy + 1) % th && r->maxy + 1 != desc->height);
   if (afbc_store) {
      r->minx = ROUND_DOWN_TO(r->minx, AFBC_BLOCK);
      r->miny = ROUND_DOWN_TO(r->miny, AFBC_BLOCK);
      r->maxx = MIN2(ALIGN_POT(r->maxx + 1, AFBC_BLOCK), desc->width) - 1;
      r->maxy = MIN2(ALIGN_POT(r->maxy + 1, AFBC_BLOCK), desc->height) - 1;
   }

   bool any_preload = false;
   for (unsigned i = 0; i < n; i++) {
      struct mali_fb_target *t = targets[i];
      if (!t->image)
         continue;
      t->preload = t->load == MALI_ATT_LOAD_LOAD ||
                   (r->partial_tiles && target_needs_border_load(t));
      any_preload |= t->preload;
   }

   cmd->gfx.draw.dirty = MALI_GFX_DIRTY_ALL;
   return any_preload;
}

bool
MALI_PER_ARCH(fb_pass_has_work)(const struct mali_render_state *r)
{
   /* Nothing tiled and nothing cleared or border-loaded: the attachments
    * keep their contents, no fragment job is needed. */
   bool work = r->tiler != 0;
   const struct mali_fb_target *ts[] = {&r->desc.z, &r->desc.s};
   for (unsigned i = 0; i < 2; i++)
      work |= ts[i]->image && ts[i]->store && ts[i]->load == MALI_ATT_LOAD_CLEAR;
#if PAN_ARCH < 9
   /* v7 also blits depth and stencil (mali_cmd_image.c), as always-written
    * targets. */
   for (unsigned i = 0; i < 2; i++)
      work |= ts[i]->image && ts[i]->store && ts[i]->always_write;
#endif
   for (unsigned i = 0; i < r->desc.rt_count; i++) {
      const struct mali_fb_target *t = &r->desc.rt[i];
      work |= t->image && t->store && (t->load == MALI_ATT_LOAD_CLEAR || t->always_write);
   }
   return work;
}

/* ---------------------------------------------------------------------- */
/* CRC (transaction elimination): which target keeps its CRCs              */

/* The seed scheme and the fragment-side seed updates are described in
 * mali_cmd_render.c ("CRC"). The knobs (mali_crc_options) are in
 * mali_image.c, compiled once for every arch. */

/* Can target t keep its CRCs in this pass? panvk pan_fb_store_target_
 * should_crc on v11: level 0 of an image with CRC state, stored, tiles of
 * at least 16x16 pixels. One layer, since CRC images have one. */
static bool
target_can_crc(const struct mali_render_state *r, const struct mali_fb_target *t)
{
   return t->image && t->image->crc_header && t->store && t->level == 0 && t->plane == 0 &&
          r->tile_size >= 16 * 16 && MAX2(r->desc.layer_count, 1) == 1;
}

/* The hardware keeps CRCs for one render target: the first that can. */
static int
select_crc_rt(const struct mali_render_state *r)
{
   if (mali_crc_options.disable)
      return -1;
   for (unsigned i = 0; i < r->desc.rt_count; i++) {
      if (target_can_crc(r, &r->desc.rt[i]))
         return i;
   }
   return -1;
}

/*
 * The CRC Clear Color (pan_fb_crc_clear_color): a hash of the packed clear
 * values of the targets the pass clears, above bit 16, with bit 62 set so
 * it is never zero. With empty-tile write the hardware stores it as the
 * CRC of a tile that has no geometry and is only cleared; with empty-tile
 * read such a tile is not written when its stored CRC already equals it.
 * Bits 0-15 are the seed; the back half puts the image's current seed
 * there before the pass runs (the fragment stream does it on v11).
 */
static uint64_t
crc_clear_color(const struct mali_render_state *r)
{
   static const uint64_t primes[4] = {16381, 16369, 16363, 16361};
   uint64_t hash = 0;
   for (unsigned i = 0; i < r->desc.rt_count; i++) {
      const struct mali_fb_target *t = &r->desc.rt[i];
      if (!t->image || t->load != MALI_ATT_LOAD_CLEAR)
         continue;
      uint32_t clear[4];
      pack_clear_color(t->format, &r->desc.clear_color[i], clear);
      for (unsigned c = 0; c < 4; c++)
         hash ^= primes[c] * clear[c] * (i + 1);
   }
   return ((1ull << 46) | hash) << 16;
}

/* ---------------------------------------------------------------------- */
/* Framebuffer descriptors                                                 */

static void
target_mem(const struct mali_fb_target *t, unsigned layer, uint64_t *base,
           uint32_t *row_stride, uint64_t *surf_stride)
{
   const struct mali_image *img = t->image;
   const struct mali_image_plane_layout *pl = &img->planes[t->plane].layout;
   const struct mali_image_slice *s = &pl->slices[t->level];

   *base = img->base + s->offset;
   if (img->vk.image_type == VK_IMAGE_TYPE_3D)
      *base += (t->layer + layer) * s->surface_stride;
   else
      *base += (uint64_t)(t->layer + layer) * pl->array_stride;
   *row_stride = s->row_stride;
   *surf_stride = s->surface_stride;
}

/* get_afbc_att_mem_props: the layer's first header, the body offset from
 * it and the header row stride. */
static void
target_afbc_mem(const struct mali_fb_target *t, unsigned layer, uint64_t *header,
                uint32_t *body_offset, uint32_t *row_stride)
{
   const struct mali_image *img = t->image;
   const struct mali_image_plane_layout *pl = &img->planes[t->plane].layout;
   const struct mali_image_slice *s = &pl->slices[t->level];

   *header = img->base + s->offset + (uint64_t)(t->layer + layer) * pl->array_stride;
   *body_offset = mali_afbc_body_offset(s);
   *row_stride = s->row_stride;
}

/* pan_needs_afbc_reverse_issue_order: a tile smaller than a superblock in
 * either direction shares superblocks with its neighbours. */
static bool
afbc_reverse_issue_order(const struct mali_render_state *r)
{
   uint32_t tw, th;
   tile_dims(r->tile_size, &tw, &th);
   return tw < AFBC_BLOCK || th < AFBC_BLOCK;
}

static enum mali_block_format
target_block_format(const struct mali_fb_target *t)
{
   if (mali_image_is_afbc(t->image))
      return MALI_BLOCK_FORMAT_AFBC_TILED;
   return t->image->vk.drm_format_mod == DRM_FORMAT_MOD_LINEAR ?
             MALI_BLOCK_FORMAT_LINEAR : MALI_BLOCK_FORMAT_TILED_U_INTERLEAVED;
}

/* Does the colour frame shader clear some target (so every tile has to
 * be written back)? */
static bool
clears_in_area(const struct mali_fb_target *t)
{
   return t->image && t->preload && t->load == MALI_ATT_LOAD_CLEAR;
}

/* Tiles without geometry are written back only for targets that are
 * cleared (by the tile buffer or by the frame shader). */
static bool
clean_tile_write(const struct mali_fb_target *t)
{
   return t->image && t->store && (t->load == MALI_ATT_LOAD_CLEAR || t->always_write);
}

static uint32_t
fbd_size(const struct mali_render_state *r, bool has_zs)
{
   return pan_size(FRAMEBUFFER) + (has_zs ? pan_size(ZS_CRC_EXTENSION) : 0) +
          r->rt_count * pan_size(RENDER_TARGET);
}

/* A single-sampled sample position table (pan_samples.c: positions in
 * 1/256 pixel, centred; 64 entries of 4 bytes). */
static uint64_t
sample_positions(struct mali_cmd_buffer *cmd)
{
   struct mali_ptr p = mali_cmd_alloc(cmd, 64 * 4, 64);
   if (!p.cpu)
      return 0;
   uint16_t pos[128];
   memset(pos, 0, sizeof(pos));
   pos[0] = pos[1] = 128;       /* sample 0 */
   pos[64] = pos[65] = 128;     /* origin */
   memcpy(p.cpu, pos, sizeof(pos));
   return p.gpu;
}

/* The frame shaders of the pass: colour preload in DCD slot 0 (Intersect,
 * or Always when a clear inside the area is done by the shader), depth and
 * stencil in slot 1 (Early ZS Always), as panvk v9+. */
static bool
build_frame_shaders(struct mali_cmd_buffer *cmd, struct mali_render_state *r,
                    uint64_t *dcds, enum mali_pre_post_frame_shader_mode modes[3])
{
   struct mali_meta_fs_key ckey = {0}, zkey = {0};
   uint32_t ctex[10][8], ztex[2][8];
   unsigned nctex = 0, nztex = 0;
   bool color = false, zs = false, color_always = false;
   const bool layered = r->desc.layer_count > 1;

   struct mali_meta_fs_push push = {
      .area = {(int32_t)r->desc.area.offset.x, (int32_t)r->desc.area.offset.y,
               (int32_t)(r->desc.area.offset.x + r->desc.area.extent.width) - 1,
               (int32_t)(r->desc.area.offset.y + r->desc.area.extent.height) - 1},
      .clear_depth = r->desc.clear_depth,
      .clear_stencil = r->desc.clear_stencil,
   };

   for (unsigned i = 0; i < r->desc.rt_count && !r->desc.blit; i++) {
      const struct mali_fb_target *t = &r->desc.rt[i];
      if (!t->image || !t->preload)
         continue;
      ckey.rt_op[i] = t->load == MALI_ATT_LOAD_CLEAR ? MALI_META_FS_CLEAR_IN_AREA
                                                      : MALI_META_FS_LOAD;
      ckey.rt_type[i] = util_format_is_pure_uint(t->format) ? MALI_META_FS_UINT :
                        util_format_is_pure_sint(t->format) ? MALI_META_FS_SINT :
                                                              MALI_META_FS_FLOAT;
      memcpy(push.clear_color[i], &r->desc.clear_color[i], 16);
      if (!MALI_PER_ARCH(cmd_pack_plane_texture)(cmd, t->image, t->plane, t->format, t->level,
                                                 t->layer, MAX2(r->desc.layer_count, 1),
                                                 ctex[nctex++]))
         return false;
      color = true;
      color_always |= clears_in_area(t);
   }
   ckey.layered = zkey.layered = layered;

   const struct mali_fb_target *zt = &r->desc.z, *st = &r->desc.s;
#if PAN_ARCH < 9
   /* A v7 blit's frame shader writes its depth and stencil targets itself,
    * fetching them outside the blit area, so they get no separate
    * preload. */
   const bool zs_preload = !r->desc.blit;
#else
   const bool zs_preload = true;
#endif
   if (zs_preload && zt->image && zt->preload) {
      zkey.z_op = zt->load == MALI_ATT_LOAD_CLEAR ? MALI_META_FS_CLEAR_IN_AREA
                                                  : MALI_META_FS_LOAD;
      if (!MALI_PER_ARCH(cmd_pack_plane_texture)(cmd, zt->image, zt->plane, zt->format, zt->level,
                                                 zt->layer, MAX2(r->desc.layer_count, 1),
                                                 ztex[nztex++]))
         return false;
      zs = true;
   }
   if (zs_preload && st->image && st->preload) {
      zkey.s_op = st->load == MALI_ATT_LOAD_CLEAR ? MALI_META_FS_CLEAR_IN_AREA
                                                  : MALI_META_FS_LOAD;
      if (!MALI_PER_ARCH(cmd_pack_plane_texture)(cmd, st->image, st->plane, st->format, st->level,
                                                 st->layer, MAX2(r->desc.layer_count, 1),
                                                 ztex[nztex++]))
         return false;
      zs = true;
   }

   if (!color && !zs && !r->desc.blit)
      return true;

#if PAN_ARCH < 9
   /* v7: one triple per layer, as panvk_vX_cmd_frame_shaders.c
    * alloc_pre_post_dcds allocates before v9; each layer's FBD points at
    * its own (fb_build) and meta_fs_dcd writes every layer's copy. */
   const uint64_t dcd_bytes =
      (uint64_t)FS_DCD_COUNT * MAX2(r->desc.layer_count, 1) * pan_size(DRAW);
   struct mali_ptr p = mali_cmd_alloc(cmd, dcd_bytes, 64);
   if (!p.cpu)
      return false;
   memset(p.cpu, 0, dcd_bytes);
#else
   struct mali_ptr p = mali_cmd_alloc(cmd, FS_DCD_COUNT * pan_size(DRAW), 64);
   if (!p.cpu)
      return false;
   memset(p.cpu, 0, FS_DCD_COUNT * pan_size(DRAW));
#endif

   if (r->desc.blit) {
      const struct mali_render_blit *bl = r->desc.blit;
#if PAN_ARCH < 9
      /* A v7 depth/stencil blit takes the depth/stencil slot, where panvk
       * and gallium (pan_fb_preload.c) put frame shaders that write depth
       * or stencil. */
      const unsigned slot = bl->key->rt_op[0] == MALI_META_FS_NONE &&
                                  (bl->key->z_op != MALI_META_FS_NONE ||
                                   bl->key->s_op != MALI_META_FS_NONE)
                               ? FS_DCD_ZS
                               : FS_DCD_COLOR;
#else
      const unsigned slot = FS_DCD_COLOR;
#endif
      const struct mali_shader *fs = MALI_PER_ARCH(meta_fs_get)(cmd, bl->key);
      if (!fs || !MALI_PER_ARCH(meta_fs_dcd)(cmd, fs, bl->key, bl->push, bl->textures,
                                             bl->texture_count, bl->sampler, true,
                                             (uint8_t *)p.cpu + slot * pan_size(DRAW)))
         return false;
      modes[slot] = MALI_PRE_POST_FRAME_SHADER_MODE_ALWAYS;
   } else if (color) {
      const struct mali_shader *fs = MALI_PER_ARCH(meta_fs_get)(cmd, &ckey);
      if (!fs || !MALI_PER_ARCH(meta_fs_dcd)(cmd, fs, &ckey, &push, (const uint32_t(*)[8])ctex,
                                             nctex, NULL, true,
                                             (uint8_t *)p.cpu + FS_DCD_COLOR * pan_size(DRAW)))
         return false;
      modes[FS_DCD_COLOR] = color_always ? MALI_PRE_POST_FRAME_SHADER_MODE_ALWAYS
                                         : MALI_PRE_POST_FRAME_SHADER_MODE_INTERSECT;
   }
   if (zs) {
      const struct mali_shader *fs = MALI_PER_ARCH(meta_fs_get)(cmd, &zkey);
      if (!fs || !MALI_PER_ARCH(meta_fs_dcd)(cmd, fs, &zkey, &push, (const uint32_t(*)[8])ztex,
                                             nztex, NULL, true,
                                             (uint8_t *)p.cpu + FS_DCD_ZS * pan_size(DRAW)))
         return false;
      /* Intersect: tiles without geometry keep their memory and need no
       * reload. panvk uses Early ZS Always (the reload runs a tile or more
       * ahead) without having measured it; on ARMSX2's many small passes it
       * reloads and writes back every tile of the render area in every
       * pass. A clear inside the area has to reach every tile. */
      const bool zs_clears = (zt->image && zt->preload && zt->load == MALI_ATT_LOAD_CLEAR) ||
                             (st->image && st->preload && st->load == MALI_ATT_LOAD_CLEAR);
      modes[FS_DCD_ZS] = zs_clears ? MALI_PRE_POST_FRAME_SHADER_MODE_ALWAYS
                                   : MALI_PRE_POST_FRAME_SHADER_MODE_INTERSECT;
   }
   *dcds = p.gpu;
   return true;
}

static void
pack_zs_ext(const struct mali_render_state *r, unsigned layer, uint64_t crc_clear, void *out)
{
   const struct mali_fb_target *zt = &r->desc.z, *st = &r->desc.s;
   struct mali_zs_crc_extension_packed desc, part;

   pan_pack(&desc, ZS_CRC_EXTENSION, cfg) {
      cfg.zs.msaa = MALI_MSAA_SINGLE;
      cfg.s.msaa = MALI_MSAA_SINGLE;
      cfg.zs.clean_tile_write_enable = clean_tile_write(zt);
      cfg.s.clean_tile_write_enable = clean_tile_write(st);
   }

   if (r->crc_rt >= 0) {
      const struct mali_image *img = r->desc.rt[r->crc_rt].image;
      const struct mali_image_crc_layout *c = &img->planes[0].layout.crc;
      pan_pack(&part, ZS_CRC_EXTENSION, cfg) {
         cfg.crc.base = img->crc_header + MALI_CRC_HEADER_SIZE;
         cfg.crc.row_stride = c->row_stride;
         cfg.crc.render_target = r->crc_rt;
         cfg.crc.clear_color = crc_clear;
      }
      pan_merge(&desc, &part, ZS_CRC_EXTENSION);
   }

#if PAN_ARCH < 9
   /* v7 depth/stencil images are never AFBC (mali_image.c v7_can_use_afbc). */
   assert(!target_is_afbc(zt) && !target_is_afbc(st));
   if (zt->image && zt->store) {
#else
   if (zt->image && zt->store && target_is_afbc(zt)) {
      uint64_t header;
      uint32_t body, row;
      target_afbc_mem(zt, layer, &header, &body, &row);
      pan_pack(&part, ZS_CRC_EXTENSION, cfg) {
         cfg.afbc_zs.write_format = zs_write_format(zt->format);
         cfg.afbc_zs.block_format = MALI_BLOCK_FORMAT_AFBC_TILED;
         cfg.afbc_zs.reverse_issue_order = afbc_reverse_issue_order(r);
         cfg.afbc_zs.header = header;
         cfg.afbc_zs.body_offset = body;
         cfg.afbc_zs.header_row_stride = row;
      }
      pan_merge(&desc, &part, ZS_CRC_EXTENSION);
   } else if (zt->image && zt->store) {
#endif
      uint64_t base, surf;
      uint32_t row;
      target_mem(zt, layer, &base, &row, &surf);
      pan_pack(&part, ZS_CRC_EXTENSION, cfg) {
         cfg.zs.write_format = zs_write_format(zt->format);
         cfg.zs.block_format = target_block_format(zt);
         cfg.zs.base = base;
         cfg.zs.row_stride = row;
         cfg.zs.surface_stride = (uint32_t)surf;
#if PAN_ARCH >= 10
         cfg.zs.surface_stride_hi = (uint32_t)(surf >> 32);
#else
         assert(surf <= UINT32_MAX);
#endif
      }
      pan_merge(&desc, &part, ZS_CRC_EXTENSION);
   }
#if PAN_ARCH < 9
   if (st->image && st->store) {
#else
   if (st->image && st->store && target_is_afbc(st)) {
      uint64_t header;
      uint32_t body, row;
      target_afbc_mem(st, layer, &header, &body, &row);
      pan_pack(&part, ZS_CRC_EXTENSION, cfg) {
         cfg.afbc_s.write_format = MALI_S_FORMAT_S8;
         cfg.afbc_s.block_format = MALI_BLOCK_FORMAT_AFBC_TILED;
         cfg.afbc_s.reverse_issue_order = afbc_reverse_issue_order(r);
         cfg.afbc_s.header = header;
         cfg.afbc_s.body_offset = body;
         cfg.afbc_s.header_row_stride = row;
      }
      pan_merge(&desc, &part, ZS_CRC_EXTENSION);
   } else if (st->image && st->store) {
#endif
      uint64_t base, surf;
      uint32_t row;
      target_mem(st, layer, &base, &row, &surf);
      pan_pack(&part, ZS_CRC_EXTENSION, cfg) {
         cfg.s.write_format = MALI_S_FORMAT_S8;
         cfg.s.block_format = target_block_format(st);
         cfg.s.base = base;
         cfg.s.row_stride = row;
         cfg.s.surface_stride = (uint32_t)surf;
#if PAN_ARCH >= 10
         cfg.s.surface_stride_hi = (uint32_t)(surf >> 32);
#else
         assert(surf <= UINT32_MAX);
#endif
      }
      pan_merge(&desc, &part, ZS_CRC_EXTENSION);
   }
   memcpy(out, &desc, sizeof(desc));
}

/*
 * pan_emit_afbc_color_attachment. v7 has an absolute body pointer, the
 * header row stride in superblocks (pan_afbc_stride_blocks), no
 * compression mode word (the writeback format is the compression format)
 * and no RAW32 stand-in for RAW24.
 */
static void
pack_rt_afbc(const struct mali_render_state *r, const struct mali_fb_target *t, unsigned i,
             unsigned layer, const uint32_t clear[4], void *out)
{
   uint64_t header;
   uint32_t body, row;
   target_afbc_mem(t, layer, &header, &body, &row);
#if PAN_ARCH < 9
   const enum pipe_format pf = t->image->planes[t->plane].format;
   const uint32_t row_blocks = row / (MALI_AFBC_HEADER_BYTES * mali_afbc_header_tile_sbs(pf));
   assert(util_format_linear(t->format) == util_format_linear(pf));
#endif

   pan_cast_and_pack(out, AFBC_RGB_RENDER_TARGET, cfg) {
      cfg.internal_buffer_offset = r->rt_offset[i];
      cfg.dithering_enable = true;
      cfg.clean_tile_write_enable = clean_tile_write(t);
      cfg.clear.color_0 = clear[0];
      cfg.clear.color_1 = clear[1];
      cfg.clear.color_2 = clear[2];
      cfg.clear.color_3 = clear[3];
      cfg.write_enable = t->store;
      rt_formats(t->format, &cfg.writeback_format, &cfg.internal_format, &cfg.swizzle);
#if PAN_ARCH >= 9
      /* No AFBC with RAW24 on v9+: RAW32, the compression mode says 24. */
      if (cfg.writeback_format == MALI_COLOR_FORMAT_RAW24)
         cfg.writeback_format = MALI_COLOR_FORMAT_RAW32;
#endif
      cfg.srgb = util_format_is_srgb(t->format);
      cfg.writeback_block_format = MALI_BLOCK_FORMAT_AFBC_TILED;
      cfg.writeback_msaa = MALI_MSAA_SINGLE;
      cfg.yuv_transform = t->image->vk.drm_format_mod & AFBC_FORMAT_MOD_YTR;
      cfg.wide_block = false;
      cfg.split_block = false;
      cfg.reverse_issue_order = afbc_reverse_issue_order(r);
      cfg.header = header;
#if PAN_ARCH >= 9
      cfg.body_offset = body;
      cfg.row_stride = row;
      cfg.compression_mode = MALI_PER_ARCH(image_afbc_hw_mode)(t->image, t->plane);
#else
      cfg.body = header + body;
      cfg.row_stride = row_blocks;
#endif
   }
}

static void
pack_rt(const struct mali_render_state *r, unsigned i, unsigned layer, void *out)
{
   const struct mali_fb_target *t = i < r->desc.rt_count ? &r->desc.rt[i] : NULL;

   if (!t || !t->image) {
      /* The hardware wants at least one target; a missing one sits past
       * the allocation, where writes are dropped (pan_fb.c). */
      pan_cast_and_pack(out, RGB_RENDER_TARGET, cfg) {
         cfg.internal_buffer_offset = r->cbuf_alloc;
         cfg.internal_format = MALI_COLOR_BUFFER_INTERNAL_FORMAT_R8G8B8A8;
         cfg.write_enable = false;
         cfg.writeback_block_format = MALI_BLOCK_FORMAT_TILED_U_INTERLEAVED;
      }
      return;
   }

   uint32_t clear[4] = {0};
   if (t->load == MALI_ATT_LOAD_CLEAR)
      pack_clear_color(t->format, &r->desc.clear_color[i], clear);

   if (target_is_afbc(t)) {
      pack_rt_afbc(r, t, i, layer, clear, out);
      return;
   }

   uint64_t base, surf;
   uint32_t row;
   target_mem(t, layer, &base, &row, &surf);

   pan_cast_and_pack(out, RGB_RENDER_TARGET, cfg) {
      cfg.internal_buffer_offset = r->rt_offset[i];
      cfg.dithering_enable = true;
      cfg.clean_tile_write_enable = clean_tile_write(t);
      cfg.clear.color_0 = clear[0];
      cfg.clear.color_1 = clear[1];
      cfg.clear.color_2 = clear[2];
      cfg.clear.color_3 = clear[3];
      cfg.write_enable = t->store;
      cfg.writeback_block_format = target_block_format(t);
      rt_formats(t->format, &cfg.writeback_format, &cfg.internal_format, &cfg.swizzle);
      cfg.srgb = util_format_is_srgb(t->format);
      cfg.writeback_msaa = MALI_MSAA_SINGLE;
      cfg.writeback_buffer.base = base;
      cfg.writeback_buffer.row_stride = row;
      cfg.writeback_buffer.surface_stride = (uint32_t)surf;
#if PAN_ARCH >= 10
      cfg.writeback_buffer.surface_stride_hi = (uint32_t)(surf >> 32);
#else
      assert(surf <= UINT32_MAX);
#endif
   }
}

static enum mali_pre_post_frame_shader_mode
fix_frame_mode(enum mali_pre_post_frame_shader_mode mode, bool clean_tile)
{
   /* pan_fix_frame_shader_mode: with clean-tile writes, every tile runs. */
   if (clean_tile && mode == MALI_PRE_POST_FRAME_SHADER_MODE_INTERSECT)
      return MALI_PRE_POST_FRAME_SHADER_MODE_ALWAYS;
   return mode;
}

uint64_t
MALI_PER_ARCH(fb_build)(struct mali_cmd_buffer *cmd, struct mali_render_state *r,
                        uint32_t *size_out)
{
   const struct mali_fb_target *zt = &r->desc.z, *st = &r->desc.s;
   r->crc_rt = select_crc_rt(r);
   const bool crc = r->crc_rt >= 0;
   const bool has_zs = zt->image || st->image || crc;
   const uint32_t layers = MAX2(r->desc.layer_count, 1);
   const uint32_t size = fbd_size(r, has_zs);
   const uint64_t crc_clear = crc ? crc_clear_color(r) : 0;
   r->crc_clear_lo = (uint32_t)crc_clear;

   uint64_t dcds = 0;
   enum mali_pre_post_frame_shader_mode modes[3] = {0};
   if (!build_frame_shaders(cmd, r, &dcds, modes))
      return 0;

   bool clean = clean_tile_write(zt) || clean_tile_write(st);
   for (unsigned i = 0; i < r->desc.rt_count; i++)
      clean |= clean_tile_write(&r->desc.rt[i]);
   /* panvk runs the preloads of a pass with a CRC target on every tile
    * (pan_emit_fb_desc: "Pre-frame shaders that preload a CRC-enabled RT
    * must run in ALWAYS mode"). We keep Intersect: on the device a LOAD
    * pass with CRC leaves tiles without geometry alone and does not stamp
    * them (vk_render_smoke part 9), and Always would read back every tile
    * of ARMSX2's full-target passes. The option restores panvk's rule. */
   const bool always_preload = clean || (crc && mali_crc_options.always_preload);

   uint64_t samples = sample_positions(cmd);
   struct mali_ptr p = mali_cmd_alloc(cmd, (uint64_t)size * layers, 64);
   if (!p.cpu || !samples)
      return 0;

   for (uint32_t l = 0; l < layers; l++) {
      uint8_t *fbd = (uint8_t *)p.cpu + (uint64_t)l * size;

      pan_section_pack(fbd, FRAMEBUFFER, PARAMETERS, cfg) {
         cfg.pre_frame_0 = fix_frame_mode(modes[0], always_preload);
         cfg.pre_frame_1 = fix_frame_mode(modes[1], always_preload);
         cfg.post_frame = MALI_PRE_POST_FRAME_SHADER_MODE_NEVER;
#if PAN_ARCH < 9
         /* Layer l's own frame shader DCDs (jm/panvk_vX_cmd_buffer.c:
          * dcd_pointer + layer * 3 Draw descriptors). */
         cfg.frame_shader_dcds =
            dcds ? dcds + (uint64_t)l * FS_DCD_COUNT * pan_size(DRAW) : 0;
#else
         cfg.frame_shader_dcds = dcds;
#endif
         cfg.sample_locations = samples;
#if PAN_ARCH >= 9
         /* The layer inside its tiler context selects the primitive list;
          * the frame argument reaches the fragment shader as the layer. */
         cfg.internal_layer_index = l % MALI_LAYERS_PER_TILER_CTX;
         cfg.frame_argument = l;
#endif
         cfg.width = r->desc.width;
         cfg.height = r->desc.height;
         cfg.bound_min_x = r->minx;
         cfg.bound_min_y = r->miny;
         cfg.bound_max_x = r->maxx;
         cfg.bound_max_y = r->maxy;
         cfg.sample_count = 1;
         cfg.sample_pattern = MALI_SAMPLE_PATTERN_SINGLE_SAMPLED;
         cfg.tie_break_rule = MALI_TIE_BREAK_RULE_MINUS_180_IN_0_OUT;
         cfg.effective_tile_size = r->tile_size;
#if PAN_ARCH >= 9
         cfg.point_sprite_coord_origin_max_y = false;
         cfg.first_provoking_vertex = r->first_provoking_vertex;
#else
         /* v7 has neither field: the provoking vertex is a Primitive bit
          * of each tiler job (jm/panvk_vX_cmd_draw.c). */
#endif
         cfg.render_target_count = r->rt_count;
         cfg.color_buffer_allocation = r->cbuf_alloc;
         if (st->image) {
            cfg.s_clear = st->load == MALI_ATT_LOAD_CLEAR ? r->desc.clear_stencil : 0;
            cfg.s_write_enable = st->store;
         }
         if (zt->image) {
            cfg.z_internal_format = z_internal_format(zt->format);
            cfg.z_clear = zt->load == MALI_ATT_LOAD_CLEAR ? r->desc.clear_depth : 0;
            cfg.z_write_enable = zt->store;
         } else {
            cfg.z_internal_format = MALI_Z_INTERNAL_FORMAT_D24;
         }
         cfg.has_zs_crc_extension = has_zs;
         if (crc) {
            cfg.crc_read_enable = true;
            cfg.crc_write_enable = true;
            cfg.empty_tile_read_enable = mali_crc_options.empty_tile_read && r->rt_count == 1;
            cfg.empty_tile_write_enable = mali_crc_options.empty_tile_write;
         }
         cfg.tiler = r->tiler ? r->tiler + (l / MALI_LAYERS_PER_TILER_CTX) *
                                              pan_size(TILER_CONTEXT) : 0;
      }
      pan_section_pack(fbd, FRAMEBUFFER, PADDING, cfg);

      uint8_t *next = fbd + pan_size(FRAMEBUFFER);
      if (has_zs) {
         pack_zs_ext(r, l, crc_clear, next);
         next += pan_size(ZS_CRC_EXTENSION);
      }
      for (unsigned i = 0; i < r->rt_count; i++)
         pack_rt(r, i, l, next + i * pan_size(RENDER_TARGET));
   }

   struct mali_framebuffer_pointer_packed tag;
   pan_pack(&tag, FRAMEBUFFER_POINTER, cfg) {
      cfg.zs_crc_extension_present = has_zs;
      cfg.render_target_count = r->rt_count;
   }
   *size_out = size;
   return p.gpu | tag.opaque[0];
}

void
MALI_PER_ARCH(fb_fill_tsd)(struct mali_cmd_buffer *cmd, struct mali_render_state *r)
{
   if (!r->tsd_cpu)
      return;
   uint64_t tls = r->tls_size ? MALI_PER_ARCH(cmd_tls_buffer)(cmd, r->tls_size) : 0;
   pan_cast_and_pack(r->tsd_cpu, LOCAL_STORAGE, cfg) {
      if (tls) {
         cfg.tls_size = util_logbase2_ceil(DIV_ROUND_UP(r->tls_size, 16));
#if PAN_ARCH >= 9
         cfg.tls_address_mode = MALI_ADDRESS_MODE_PACKED;
         cfg.tls_base_pointer = tls >> 8;
#else
         /* v7 has no address mode: the full pointer (Mesa pan_emit_tls). */
         cfg.tls_base_pointer = tls;
#endif
      }
      cfg.wls_instances = MALI_LOCAL_STORAGE_NO_WORKGROUP_MEM;
   }
}

/* ---------------------------------------------------------------------- */
/* vkCmdBeginRendering / vkCmdEndRendering                                 */

static enum mali_att_load
att_load(VkAttachmentLoadOp op)
{
   switch (op) {
   case VK_ATTACHMENT_LOAD_OP_LOAD: return MALI_ATT_LOAD_LOAD;
   case VK_ATTACHMENT_LOAD_OP_CLEAR: return MALI_ATT_LOAD_CLEAR;
   default: return MALI_ATT_LOAD_DONT_CARE;
   }
}

static bool
att_store(const VkRenderingAttachmentInfo *att)
{
   return att->storeOp != VK_ATTACHMENT_STORE_OP_DONT_CARE &&
          att->storeOp != VK_ATTACHMENT_STORE_OP_NONE;
}

static void
set_target(struct mali_fb_target *t, const struct mali_image_view *view, unsigned plane,
           enum pipe_format format, const VkRenderingAttachmentInfo *att)
{
   const struct mali_image *image = container_of(view->vk.image, struct mali_image, vk);
   t->image = image;
   t->plane = plane;
   t->level = view->vk.base_mip_level;
   t->layer = view->vk.base_array_layer;
   t->format = format;
   t->load = att_load(att->loadOp);
   t->store = att_store(att);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdBeginRendering)(VkCommandBuffer commandBuffer, const VkRenderingInfo *info)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   struct mali_render_desc d = {
      .area = info->renderArea,
      .layer_count = info->viewMask ? util_last_bit(info->viewMask) : info->layerCount,
      .rt_count = info->colorAttachmentCount,
   };
   uint32_t w = UINT32_MAX, h = UINT32_MAX;
   bool bound = false;

   if (info->flags & (VK_RENDERING_SUSPENDING_BIT | VK_RENDERING_RESUMING_BIT))
      vk_logw(VK_LOG_OBJS(cmd), "malisx2: suspended/resumed rendering is not supported");
   if (info->viewMask)
      vk_logw(VK_LOG_OBJS(cmd), "malisx2: multiview rendering is not supported");

   assert(info->colorAttachmentCount <= MALI_MAX_RTS);
   for (uint32_t i = 0; i < info->colorAttachmentCount; i++) {
      const VkRenderingAttachmentInfo *att = &info->pColorAttachments[i];
      VK_FROM_HANDLE(mali_image_view, view, att->imageView);
      if (!view)
         continue;
      if (view->vk.image->samples > 1)
         vk_logw(VK_LOG_OBJS(cmd), "malisx2: multisampled attachments are not supported");
      set_target(&d.rt[i], view, 0, vk_format_to_pipe_format(view->vk.view_format), att);
      d.clear_color[i] = att->clearValue.color;
      w = MIN2(w, view->vk.extent.width);
      h = MIN2(h, view->vk.extent.height);
      bound = true;
   }

   const VkRenderingAttachmentInfo *za = info->pDepthAttachment;
   const VkRenderingAttachmentInfo *sa = info->pStencilAttachment;
   if (za && za->imageView) {
      VK_FROM_HANDLE(mali_image_view, view, za->imageView);
      const struct mali_image *img = container_of(view->vk.image, struct mali_image, vk);
      if (vk_format_has_depth(img->vk.format)) {
         unsigned plane = mali_image_aspect_plane(img, VK_IMAGE_ASPECT_DEPTH_BIT);
         set_target(&d.z, view, plane, img->planes[plane].format, za);
         d.clear_depth = za->clearValue.depthStencil.depth;
      }
      w = MIN2(w, view->vk.extent.width);
      h = MIN2(h, view->vk.extent.height);
      bound = true;
   }
   if (sa && sa->imageView) {
      VK_FROM_HANDLE(mali_image_view, view, sa->imageView);
      const struct mali_image *img = container_of(view->vk.image, struct mali_image, vk);
      /* A depth-only view of a depth/stencil image still names the
       * image's stencil plane (ARMSX2 attaches D32S8 through one). */
      if (vk_format_has_stencil(img->vk.format)) {
         unsigned plane = mali_image_aspect_plane(img, VK_IMAGE_ASPECT_STENCIL_BIT);
         set_target(&d.s, view, plane, img->planes[plane].format, sa);
         d.clear_stencil = sa->clearValue.depthStencil.stencil;
      }
      w = MIN2(w, view->vk.extent.width);
      h = MIN2(h, view->vk.extent.height);
      bound = true;
   }

   if (bound) {
      d.width = w;
      d.height = h;
   } else {
      /* No attachments: the framebuffer is the render area (panvk). */
      d.width = info->renderArea.offset.x + info->renderArea.extent.width;
      d.height = info->renderArea.offset.y + info->renderArea.extent.height;
   }
   d.width = MAX2(d.width, 1);
   d.height = MAX2(d.height, 1);
   d.layer_count = MAX2(d.layer_count, 1);

   MALI_PER_ARCH(cmd_render_begin)(cmd, &d);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdEndRendering)(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   MALI_PER_ARCH(cmd_render_end)(cmd);
}
