/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Bounds-checked storage buffer loads of v7 (Bifrost) vertex shaders.
 *
 * A v7 Vertex job shades every vertex of the draw's index range, including
 * ones no index names, so a vertex shader that reads its vertices out of a
 * storage buffer can compute an address far outside the buffer and fault
 * the GPU context. Such shaders are compiled with robust storage buffer
 * access (mali_pipeline.c, stage_init). With the bounded global address
 * format nir_lower_explicit_io puts every load behind an if, one per
 * component, and the backend cannot merge loads that are behind branches:
 * ARMSX2's sprite expansion vertex shader grew from 1488 to 2752 bytes.
 *
 * Vertex shaders therefore get nir_shader_compiler_options with
 * has_load_global_bounded (mali_shader.c), which makes nir_lower_explicit_io
 * emit load_global_bounded(base, offset, bound) instead of the if. This
 * pass turns that into a plain load at base + min(offset, last), where last
 * is the highest offset at which the access still fits in the buffer. No
 * branch, and one umin per load on top of a value shared by all loads of
 * the same size from the same descriptor.
 *
 * An access that does not fit therefore reads the last position that does
 * (the first one if the buffer is smaller than the access), not zero.
 * robustBufferAccess allows either: an out-of-bounds load returns zero or
 * a value from anywhere in the memory bound to the buffer. The
 * out-of-bounds value of a vertex nobody draws goes unused.
 *
 * The clamp is rounded down to the alignment of the loaded component, so a
 * clamped access is as aligned as the original (offsets of in-bounds
 * accesses are multiples of the component size, and are not changed).
 */

#include "mali_shader.h"

#include "nir_builder.h"

static bool
lower_bounded_load(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_global_bounded)
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   nir_def *base = intr->src[0].ssa;
   nir_def *offset = intr->src[1].ssa;
   nir_def *bound = intr->src[2].ssa;
   const unsigned comp_bytes = intr->def.bit_size / 8;
   const unsigned size = comp_bytes * intr->def.num_components;

   /* bound - size, or 0 when the buffer is smaller than the access. */
   nir_def *last = nir_iadd_imm(b, nir_umax_imm(b, bound, size), -(int)size);
   last = nir_iand_imm(b, last, ~(uint32_t)(comp_bytes - 1));

   nir_def *clamped = nir_umin(b, offset, last);
   nir_def *res = nir_load_global(b, intr->def.num_components, intr->def.bit_size,
                                  nir_iadd(b, base, nir_u2u64(b, clamped)),
                                  .access = nir_intrinsic_access(intr),
                                  .align_mul = comp_bytes, .align_offset = 0);
   nir_def_replace(&intr->def, res);
   return true;
}

bool
mali_nir_lower_bounded_loads(nir_shader *nir)
{
   return nir_shader_intrinsics_pass(nir, lower_bounded_load, nir_metadata_control_flow, NULL);
}
