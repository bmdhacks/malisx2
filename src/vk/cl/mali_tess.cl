/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The tessellator's compute kernels (mali_cmd_draw.c draw_tess), after
 * Asahi's libagx tessellator.cl and prefix_sum_tess: per patch, count its
 * indices or write them, for an isoline, triangle or quad domain; and in
 * between, one invocation that turns the counts into offsets and allocates
 * the index buffer. Compiled to SPIR-V at build time and by the driver at
 * run time (mali_cmd_copy.c), not into NIR builder functions: the
 * tessellator inlined and unrolled is millions of instructions.
 */

#include "poly/geometry.h"
#include "mali_tessellator.h"

/* Counting: an invocation per patch. Writing: MALI_TESS_THREADS per patch,
 * which share its points and stitches (mali_tessellator.h), each in a warp
 * of its own: a block of 16 x 16 invocations covers 16 patches, with the
 * patch as the lane and the share as the warp, so that a warp's lanes take
 * the same branches and the shares run at the same time. */
#define MALI_TESS_THREADS 16
#define MALI_TESS_LANES 16

#define TESS_KERNEL(name, process, mode, threads)                              \
   KERNEL(64)                                                                  \
   name(constant struct poly_tess_params *p)                                  \
   {                                                                           \
      uint g = cl_global_id.x, patch = g, tid = 0;                             \
      if (threads > 1) {                                                       \
         uint block = g / (MALI_TESS_LANES * threads);                         \
         tid = (g / MALI_TESS_LANES) % threads;                                \
         patch = block * MALI_TESS_LANES + g % MALI_TESS_LANES;                \
      }                                                                        \
      if (patch >= p->nr_patches)                                              \
         return;                                                               \
      for (uint target = tid;; target += threads) {                            \
         uint total = process(p, patch, tid, threads, target, mode);           \
         if (target + threads >= total)                                        \
            break;                                                             \
      }                                                                        \
   }

TESS_KERNEL(mali_tess_count_isolines, poly_tess_isoline_process, POLY_TESS_MODE_COUNT, 1)
/* Triangles and quads: counting records the patch's stitches, which
 * mali_tess_stitch does, and mali_tess_points_* places the points. */
#define TESS_COUNT_KERNEL(name, process)                                       \
   KERNEL(64)                                                                  \
   name(constant struct poly_tess_params *p)                                  \
   {                                                                           \
      uint patch = cl_global_id.x;                                             \
      if (patch < p->nr_patches) {                                             \
         tess_record_count(p)[patch] = 0;                                      \
         process(p, patch, 0, 1, TESS_TARGET_RECORD, POLY_TESS_MODE_COUNT);    \
      }                                                                        \
   }
TESS_KERNEL(mali_tess_count_triangles, poly_tess_tri_process, POLY_TESS_MODE_COUNT, 1)
TESS_KERNEL(mali_tess_count_quads, poly_tess_quad_process, POLY_TESS_MODE_COUNT, 1)
TESS_COUNT_KERNEL(mali_tess_record_triangles, poly_tess_tri_process)
TESS_COUNT_KERNEL(mali_tess_record_quads, poly_tess_quad_process)

#define TESS_POINTS_KERNEL(name, process)                                      \
   KERNEL(64)                                                                  \
   name(constant struct poly_tess_params *p)                                  \
   {                                                                           \
      uint g = cl_global_id.x;                                                 \
      uint block = g / (MALI_TESS_LANES * MALI_TESS_THREADS);                  \
      uint tid = (g / MALI_TESS_LANES) % MALI_TESS_THREADS;                    \
      uint patch = block * MALI_TESS_LANES + g % MALI_TESS_LANES;              \
      if (patch < p->nr_patches)                                               \
         process(p, patch, tid, MALI_TESS_THREADS, TESS_TARGET_POINTS,         \
                 POLY_TESS_MODE_WITH_COUNTS);                                  \
   }
TESS_POINTS_KERNEL(mali_tess_points_triangles, poly_tess_tri_process)
TESS_POINTS_KERNEL(mali_tess_points_quads, poly_tess_quad_process)

/* Record s of the patch. */
static void
tess_stitch(constant struct poly_tess_params *p, uint patch, uint s)
{
   if (s >= tess_record_count(p)[patch])
      return;
   global uint4 *r = (global uint4 *)(tess_records(p, patch) + s * TESS_RECORD_WORDS);
   uint4 r0 = r[0], r1 = r[1], r2 = r[2], r3 = r[3], r4 = r[4];
   struct CHWTessellator ctx = {
      .mode = POLY_TESS_MODE_WITH_COUNTS,
      .index_bias = patch * POLY_TES_PATCH_ID_STRIDE,
      /* Not p->index_buffer: mali_tess_finish just wrote it, and p is
       * read through the constant cache; the draw it wrote is global. */
      .Index = (global uint32_t *)((uintptr_t)p->heap->base) +
               ((global uint32_t *)p->out_draws)[2] +
               (patch ? p->counts[patch - 1] : 0),
   };
   ctx.def.kind = r0.x;
   ctx.def.a[0] = r0.y;
   ctx.def.a[1] = r0.z;
   ctx.def.a[2] = r0.w;
   ctx.def.a[3] = r1.x;
   ctx.def.a[4] = r1.y;
   ctx.def.a[5] = r1.z;
   ctx.def.odd0 = r1.w & 1;
   ctx.def.odd1 = (r1.w >> 1) & 1;
   ctx.def.trapezoid = (r1.w >> 2) & 1;
   ctx.def.patched = (r1.w >> 3) & 1;
   ctx.def.patched2 = (r1.w >> 4) & 1;
   ctx.def.diag = r2.x;
   ctx.def.pc.insidePointIndexDeltaToRealValue = r2.y;
   ctx.def.pc.insidePointIndexBadValue = r2.z;
   ctx.def.pc.insidePointIndexReplacementValue = r2.w;
   ctx.def.pc.outsidePointIndexPatchBase = r3.x;
   ctx.def.pc.outsidePointIndexDeltaToRealValue = r3.y;
   ctx.def.pc.outsidePointIndexBadValue = r3.z;
   ctx.def.pc.outsidePointIndexReplacementValue = r3.w;
   ctx.def.pc2.baseIndexToInvert = r4.x;
   ctx.def.pc2.indexInversionEndPoint = r4.y;
   ctx.def.pc2.cornerCaseBadValue = r4.z;
   ctx.def.pc2.cornerCaseReplacementValue = r4.w;
   tess_run_deferred(&ctx);
}

/* A recorded stitch: (record, patch) is the invocation's (x, y). */
KERNEL(64)
mali_tess_stitch(constant struct poly_tess_params *p)
{
   if (cl_global_id.y < p->nr_patches)
      tess_stitch(p, cl_global_id.y, cl_global_id.x);
}

/* The points and the stitches of a patch in one pass: invocations x 0..15
 * of row y place its points, x from 64 do its records. */
#define TESS_EMIT_KERNEL(name, process)                                        \
   KERNEL(64)                                                                  \
   name(constant struct poly_tess_params *p)                                  \
   {                                                                           \
      uint x = cl_global_id.x, patch = cl_global_id.y;                         \
      if (patch >= p->nr_patches)                                              \
         return;                                                               \
      if (x < MALI_TESS_THREADS)                                               \
         process(p, patch, x, MALI_TESS_THREADS, TESS_TARGET_POINTS,           \
                 POLY_TESS_MODE_WITH_COUNTS);                                  \
      else if (x >= 64)                                                        \
         tess_stitch(p, patch, x - 64);                                        \
   }
TESS_EMIT_KERNEL(mali_tess_emit_triangles, poly_tess_tri_process)
TESS_EMIT_KERNEL(mali_tess_emit_quads, poly_tess_quad_process)
TESS_KERNEL(mali_tess_write_isolines, poly_tess_isoline_process, POLY_TESS_MODE_WITH_COUNTS,
            MALI_TESS_THREADS)
TESS_KERNEL(mali_tess_write_triangles, poly_tess_tri_process, POLY_TESS_MODE_WITH_COUNTS,
            MALI_TESS_THREADS)
TESS_KERNEL(mali_tess_write_quads, poly_tess_quad_process, POLY_TESS_MODE_WITH_COUNTS,
            MALI_TESS_THREADS)

/* The inclusive prefix sum of the per-patch index counts, the index buffer
 * for all of them, and the draw of it; and the patches' domain points, at
 * the offsets their point counts give. p is global or constant. */
#define TESS_FINISH_BODY(p, alloc_B)                                           \
   do {                                                                        \
      uint total = 0, points = 0;                                              \
      for (uint i = 0; i < p->nr_patches; i++) {                               \
         total += p->counts[i];                                                \
         p->counts[i] = total;                                                 \
         uint n = p->coord_allocs[i];                                          \
         p->coord_allocs[i] = points;                                          \
         points += n;                                                          \
      }                                                                        \
      uint points_B = poly_heap_alloc_offs(                                    \
         p->heap, points * sizeof(struct poly_tess_point), 16);                \
      for (uint i = 0; i < p->nr_patches; i++)                                 \
         p->coord_allocs[i] += points_B / sizeof(struct poly_tess_point);      \
      alloc_B = poly_heap_alloc_offs(p->heap, total * 4, 16);                  \
      global uint32_t *desc = p->out_draws;                                    \
      desc[0] = total; /* index count */                                       \
      desc[1] = 1;     /* instance count */                                    \
      desc[2] = alloc_B / 4;                                                   \
      desc[3] = 0; /* vertex offset */                                         \
      desc[4] = 0; /* first instance */                                        \
   } while (0)

KERNEL(1)
mali_tess_finish(global struct poly_tess_params *p)
{
   uint alloc_B;
   TESS_FINISH_BODY(p, alloc_B);
   p->index_buffer = (global uint32_t *)(((uintptr_t)p->heap->base) + alloc_B);
}

/* Counting with records and the offsets in one workgroup, for draws of at
 * most 64 patches: an invocation per patch, then invocation 0 sums. */
#define TESS_RECORD_FINISH_KERNEL(name, process)                               \
   KERNEL(64)                                                                  \
   name(constant struct poly_tess_params *p)                                  \
   {                                                                           \
      uint patch = cl_local_id.x;                                              \
      if (patch < p->nr_patches) {                                             \
         tess_record_count(p)[patch] = 0;                                      \
         process(p, patch, 0, 1, TESS_TARGET_RECORD, POLY_TESS_MODE_COUNT);    \
      }                                                                        \
      barrier(CLK_GLOBAL_MEM_FENCE);                                           \
      if (patch == 0) {                                                        \
         uint alloc_B;                                                         \
         TESS_FINISH_BODY(p, alloc_B);                                         \
      }                                                                        \
   }
TESS_RECORD_FINISH_KERNEL(mali_tess_record_finish_triangles, poly_tess_tri_process)
TESS_RECORD_FINISH_KERNEL(mali_tess_record_finish_quads, poly_tess_quad_process)
