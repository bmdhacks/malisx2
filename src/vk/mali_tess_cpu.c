/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The tessellator (vk/cl/mali_tessellator.h, OpenCL C) built for the CPU,
 * to tessellate a patch whose levels are known when the pipeline is created
 * (mali_tess_pattern.h). The OpenCL address-space keywords are empty here,
 * its built-ins are C functions, and the parameter struct has host pointers.
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* The enums and constants, from the host side of the headers. */
#include "poly/geometry.h"
#include "poly/tessellator.h"

#include "mali_tess_pattern.h"

#define global
#define private
#define constant const
typedef unsigned int uint;

static inline int imin_(int a, int b) { return a < b ? a : b; }
static inline int imax_(int a, int b) { return a > b ? a : b; }
static inline unsigned umin_(unsigned a, unsigned b) { return a < b ? a : b; }
static inline unsigned umax_(unsigned a, unsigned b) { return a > b ? a : b; }
static inline float fmin_(float a, float b) { return fminf(a, b); }
static inline float fmax_(float a, float b) { return fmaxf(a, b); }

#define min(a, b)                                                              \
   _Generic((a) + (b), float: fmin_, double: fmin_, unsigned: umin_, default: imin_)(a, b)
#define max(a, b)                                                              \
   _Generic((a) + (b), float: fmax_, double: fmax_, unsigned: umax_, default: imax_)(a, b)
#define clamp(x, lo, hi) min(max((x), (lo)), (hi))
#define ceil(x)  ceilf(x)
#define floor(x) floorf(x)

static inline float
mad(float a, float b, float c)
{
   return a * b + c;
}

static inline uint
clz(uint x)
{
   return x ? __builtin_clz(x) : 32;
}

static inline uint
sub_sat(uint a, uint b)
{
   return a > b ? a - b : 0;
}

/* The tessellator's view of struct poly_tess_params, with host pointers. */
struct host_tess_params {
   struct poly_tess_point *patch_coord_buffer;
   uint32_t *coord_allocs;
   uint32_t *out_draws;
   float *tcs_buffer;
   uint32_t *counts;
   uint32_t *index_buffer;
   uint32_t tcs_stride_el;
   uint64_t statistic;   /* the GPU passes' stitch records; never recorded here */
   uint32_t nr_patches;
   enum poly_tess_partitioning partitioning;
   bool points_mode;
};

#define poly_tess_params host_tess_params
#define MALI_TESS_HOST 1
#include "cl/mali_tessellator.h"
#undef poly_tess_params

static uint32_t
process(constant struct host_tess_params *p, unsigned domain, uint target,
        enum poly_tess_mode mode)
{
   switch (domain) {
   case 0:
      return poly_tess_isoline_process(p, 0, 0, 1, target, mode);
   case 1:
      return poly_tess_tri_process(p, 0, 0, 1, target, mode);
   default:
      return poly_tess_quad_process(p, 0, 0, 1, target, mode);
   }
}

bool
mali_tess_pattern_build(struct mali_tess_pattern *out, unsigned domain,
                        unsigned partitioning, bool points, const float factors[6])
{
   memset(out, 0, sizeof(*out));
   float tcs[6];
   memcpy(tcs, factors, sizeof(tcs));
   uint32_t counts[1] = {0}, coord_allocs[1] = {0}, out_draws[5] = {0};
   struct host_tess_params p = {
      .coord_allocs = coord_allocs,
      .out_draws = out_draws,
      .tcs_buffer = tcs,
      .counts = counts,
      .tcs_stride_el = 6,
      .partitioning = partitioning,
      .points_mode = points,
   };

   process(&p, domain, 0, POLY_TESS_MODE_COUNT);
   const uint32_t indices = counts[0], vertices = coord_allocs[0];
   if (vertices > POLY_TES_PATCH_ID_STRIDE)
      return false;

   out->indices = calloc(MAX2(indices, 1), sizeof(uint32_t));
   out->points = calloc(MAX2(vertices, 1), sizeof(struct poly_tess_point));
   if (!out->indices || !out->points) {
      mali_tess_pattern_finish(out);
      return false;
   }

   /* One patch: its indices at 0, its points at 0. */
   p.index_buffer = out->indices;
   p.patch_coord_buffer = out->points;
   coord_allocs[0] = 0;
   for (uint target = 0;; target++) {
      if (process(&p, domain, target, POLY_TESS_MODE_WITH_COUNTS) <= target + 1)
         break;
   }
   out->index_count = indices;
   out->point_count = vertices;
   return true;
}

void
mali_tess_pattern_finish(struct mali_tess_pattern *t)
{
   free(t->indices);
   free(t->points);
   memset(t, 0, sizeof(*t));
}
