/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * A tessellation pipeline whose control shader writes constant levels has
 * the same tessellation for every patch: its domain points and indices,
 * computed once on the CPU (mali_tess_cpu.c). A draw of it needs none of
 * the tessellator's jobs: every patch takes the same points, and its
 * indices are the pattern's plus the patch's POLY_TES_PATCH_ID_STRIDE.
 */

#ifndef MALI_TESS_PATTERN_H
#define MALI_TESS_PATTERN_H

#include <stdbool.h>
#include <stdint.h>

struct poly_tess_point;

struct mali_tess_pattern {
   uint32_t *indices;
   struct poly_tess_point *points;
   uint32_t index_count, point_count;
};

/* domain: 0 isolines, 1 triangles, 2 quads; partitioning: enum
 * poly_tess_partitioning; factors: outer 0..3, inner 0..1, as the control
 * shader writes them. False if the patch cannot be done this way. */
bool mali_tess_pattern_build(struct mali_tess_pattern *out, unsigned domain,
                             unsigned partitioning, bool points, const float factors[6]);
void mali_tess_pattern_finish(struct mali_tess_pattern *t);

#endif
