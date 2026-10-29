/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * How much of a render pass's area its draws can reach, kept as a coarse
 * grid of cells. The frame-shader mode of a depth preload depends on it
 * (mali_fb.c build_frame_shaders): a pass whose draws cover most of the
 * area wants every tile preloaded, a pass whose draws cover little wants
 * only the tiles that have geometry.
 *
 * The grid is 8x8 cells laid over the pass's bounding box. Each draw's
 * scissor box (the viewport clipped by the scissor, the most a draw can
 * touch) sets every cell it overlaps, so the result is an upper bound that
 * is exact for boxes aligned to cell edges. Plain C with no driver
 * dependencies, so the host tests include it directly.
 */

#ifndef MALI_COVER_H
#define MALI_COVER_H

#include <stdbool.h>
#include <stdint.h>

#define MALI_COVER_GRID  8
#define MALI_COVER_CELLS (MALI_COVER_GRID * MALI_COVER_GRID)

struct mali_cover {
   uint64_t cells;          /* bit 8 * row + column */
   int32_t x0, y0, x1, y1;  /* the area, inclusive */
   uint32_t sx, sy;         /* cells per pixel, 16.16, rounded up */
   uint32_t total;          /* cells the area has: 64, fewer if under 8 pixels wide or high */
};

/* Start an empty grid over the inclusive box. */
static inline void
mali_cover_init(struct mali_cover *c, int32_t minx, int32_t miny, int32_t maxx, int32_t maxy)
{
   const uint32_t w = maxx >= minx ? (uint32_t)(maxx - minx) + 1 : 1;
   const uint32_t h = maxy >= miny ? (uint32_t)(maxy - miny) + 1 : 1;
   const uint32_t gx = w < MALI_COVER_GRID ? w : MALI_COVER_GRID;
   const uint32_t gy = h < MALI_COVER_GRID ? h : MALI_COVER_GRID;
   c->cells = 0;
   c->x0 = minx;
   c->y0 = miny;
   c->x1 = maxx >= minx ? maxx : minx;
   c->y1 = maxy >= miny ? maxy : miny;
   c->sx = ((gx << 16) + w - 1) / w;
   c->sy = ((gy << 16) + h - 1) / h;
   c->total = gx * gy;
}

/* Mark the cells an inclusive box overlaps. A box outside the area, or an
 * empty one (max < min), marks nothing. */
static inline void
mali_cover_add(struct mali_cover *c, int32_t minx, int32_t miny, int32_t maxx, int32_t maxy)
{
   if (maxx < minx || maxy < miny || maxx < c->x0 || maxy < c->y0 || minx > c->x1 ||
       miny > c->y1)
      return;
   minx = minx < c->x0 ? c->x0 : minx;
   miny = miny < c->y0 ? c->y0 : miny;
   maxx = maxx > c->x1 ? c->x1 : maxx;
   maxy = maxy > c->y1 ? c->y1 : maxy;

   uint32_t cx0 = ((uint64_t)(minx - c->x0) * c->sx) >> 16;
   uint32_t cy0 = ((uint64_t)(miny - c->y0) * c->sy) >> 16;
   uint32_t cx1 = ((uint64_t)(maxx - c->x0) * c->sx) >> 16;
   uint32_t cy1 = ((uint64_t)(maxy - c->y0) * c->sy) >> 16;
   cx0 = cx0 > MALI_COVER_GRID - 1 ? MALI_COVER_GRID - 1 : cx0;
   cy0 = cy0 > MALI_COVER_GRID - 1 ? MALI_COVER_GRID - 1 : cy0;
   cx1 = cx1 > MALI_COVER_GRID - 1 ? MALI_COVER_GRID - 1 : cx1;
   cy1 = cy1 > MALI_COVER_GRID - 1 ? MALI_COVER_GRID - 1 : cy1;

   /* One byte per row: the columns of the box, repeated for its rows. */
   const uint64_t cols = (((uint64_t)1 << (cx1 - cx0 + 1)) - 1) << cx0;
   const uint64_t rows = (~(uint64_t)0 << (8 * cy0)) & (~(uint64_t)0 >> (8 * (7 - cy1)));
   c->cells |= rows & (cols * 0x0101010101010101ull);
}

/* Cells set, 0..MALI_COVER_CELLS. */
static inline unsigned
mali_cover_count(const struct mali_cover *c)
{
   return (unsigned)__builtin_popcountll(c->cells);
}

/* Do the draws reach at least half of the area's cells? */
static inline bool
mali_cover_most(const struct mali_cover *c)
{
   return mali_cover_count(c) * 2 >= c->total;
}

#endif
