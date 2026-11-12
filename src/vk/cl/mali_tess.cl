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
#include "poly/cl/tessellator.h"

#define TESS_KERNEL(name, process, mode)                                       \
   KERNEL(64)                                                                  \
   name(constant struct poly_tess_params *p)                                  \
   {                                                                           \
      uint patch = cl_global_id.x;                                             \
      if (patch < p->nr_patches)                                               \
         process(p, patch, mode);                                              \
   }

TESS_KERNEL(mali_tess_count_isolines, poly_tess_isoline_process, POLY_TESS_MODE_COUNT)
TESS_KERNEL(mali_tess_count_triangles, poly_tess_tri_process, POLY_TESS_MODE_COUNT)
TESS_KERNEL(mali_tess_count_quads, poly_tess_quad_process, POLY_TESS_MODE_COUNT)
TESS_KERNEL(mali_tess_write_isolines, poly_tess_isoline_process, POLY_TESS_MODE_WITH_COUNTS)
TESS_KERNEL(mali_tess_write_triangles, poly_tess_tri_process, POLY_TESS_MODE_WITH_COUNTS)
TESS_KERNEL(mali_tess_write_quads, poly_tess_quad_process, POLY_TESS_MODE_WITH_COUNTS)

/* The inclusive prefix sum of the per-patch index counts, the index buffer
 * for all of them, and the draw of it. */
KERNEL(1)
mali_tess_finish(global struct poly_tess_params *p)
{
   uint total = 0;
   for (uint i = 0; i < p->nr_patches; i++) {
      total += p->counts[i];
      p->counts[i] = total;
   }

   uint alloc_B = poly_heap_alloc_offs(p->heap, total * 4, 16);
   p->index_buffer = (global uint32_t *)(((uintptr_t)p->heap->base) + alloc_B);

   global uint32_t *desc = p->out_draws;
   desc[0] = total;     /* index count */
   desc[1] = 1;         /* instance count */
   desc[2] = alloc_B / 4;
   desc[3] = 0;         /* vertex offset */
   desc[4] = 0;         /* first instance */
}
