/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * vkCmdCopyBuffer, vkCmdFillBuffer and vkCmdUpdateBuffer (and the query
 * pools' reset fill and result copy, below), as compute
 * dispatches of small internal shaders: on the compute subqueue (CSF, v11)
 * or as Compute jobs in the open batch's vtc chain (job manager, v7 and v9).
 * Built once per arch.
 *
 * Why compute and not the command stream itself: CS LOAD_MULTIPLE /
 * STORE_MULTIPLE move at most 64 bytes per instruction pair through the
 * CS front end, which is a microcontroller; a shader moves 16 bytes per
 * invocation on every core. The blob does the same: one compute dispatch
 * per copy region, with the element size the largest power of two up to
 * 16 that divides both addresses. ARMSX2 uses vkCmdCopyBuffer for
 * uploads; fill and update are outside its default path but cost nothing
 * extra here.
 *
 * The shaders are built with nir_builder (no descriptors: addresses come
 * in push constants, loads and stores are global), compiled through the
 * normal path on first use, and kept for the device's lifetime.
 */

#include "mali_blend.h"
#if PAN_ARCH >= 10
#include "mali_cmd_buffer.h"
#else
#include "mali_jm.h"
#endif

#include <stdlib.h>
#include <string.h>

#include "compiler/nir/nir_builder.h"
#include "util/mesa-blake3.h"
#include "vk_buffer.h"
#include "vk_pipeline.h"

#include "mali_arch.h"
#include "mali_memory.h"
#include "mali_query.h"
#include "mali_vk.h"

#define META_WG_SIZE 64
/* Elements per dispatch, so JOB_SIZE stays well inside 32 bits. */
#define META_MAX_ELEMS (1u << 30)

struct meta_copy_push {
   uint64_t src;
   uint64_t dst;
   uint32_t count;
   uint32_t pad;
};

struct meta_fill_push {
   uint64_t dst;
   uint32_t count;
   uint32_t value;
};

/* vkCmdCopyQueryPoolResults: one invocation per query. The pointers are
 * the first query's value, availability word and destination; the strides
 * step them per query. */
struct meta_query_push {
   uint64_t val;
   uint64_t avail;
   uint64_t dst;
   uint32_t val_stride;
   uint32_t avail_stride;
   uint32_t dst_stride;
   uint32_t count;
   uint32_t flags;   /* VkQueryResultFlags */
   uint32_t pad;
};

static unsigned
meta_elem_size(enum mali_meta_shader m)
{
   switch (m) {
   case MALI_META_COPY_16:
   case MALI_META_FILL_16:
      return 16;
   case MALI_META_COPY_4:
   case MALI_META_FILL_4:
      return 4;
   default:
      return 1;
   }
}

static nir_shader *
build_meta_nir(struct mali_device *dev, enum mali_meta_shader m)
{
   static const char *const names[] = {
      [MALI_META_COPY_16] = "mali_meta_copy16",
      [MALI_META_COPY_4] = "mali_meta_copy4",
      [MALI_META_COPY_1] = "mali_meta_copy1",
      [MALI_META_FILL_16] = "mali_meta_fill16",
      [MALI_META_FILL_4] = "mali_meta_fill4",
   };
   const bool fill = m == MALI_META_FILL_16 || m == MALI_META_FILL_4;
   const unsigned elem = meta_elem_size(m);
   const unsigned comps = elem == 16 ? 4 : 1;
   const unsigned bits = elem == 1 ? 8 : 32;

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, MALI_PER_ARCH(shader_nir_options)(dev, MESA_SHADER_COMPUTE), "%s",
      names[m]);
   b.shader->info.workgroup_size[0] = META_WG_SIZE;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   const unsigned push_size =
      fill ? sizeof(struct meta_fill_push) : sizeof(struct meta_copy_push);
   nir_def *id = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *count = nir_load_push_constant(
      &b, 1, 32, nir_imm_int(&b, fill ? offsetof(struct meta_fill_push, count)
                                      : offsetof(struct meta_copy_push, count)),
      .base = 0, .range = push_size);

   nir_push_if(&b, nir_ult(&b, id, count));
   {
      nir_def *off = nir_u2u64(&b, nir_imul_imm(&b, id, elem));
      nir_def *dst = nir_load_push_constant(
         &b, 1, 64, nir_imm_int(&b, fill ? offsetof(struct meta_fill_push, dst)
                                         : offsetof(struct meta_copy_push, dst)),
         .base = 0, .range = push_size);
      nir_def *value;
      if (fill) {
         nir_def *v = nir_load_push_constant(
            &b, 1, 32, nir_imm_int(&b, offsetof(struct meta_fill_push, value)),
            .base = 0, .range = push_size);
         value = comps == 4 ? nir_vec4(&b, v, v, v, v) : v;
      } else {
         nir_def *src = nir_load_push_constant(
            &b, 1, 64, nir_imm_int(&b, offsetof(struct meta_copy_push, src)),
            .base = 0, .range = push_size);
         value = nir_load_global(&b, comps, bits, nir_iadd(&b, src, off),
                                 .align_mul = elem);
      }
      nir_store_global(&b, value, nir_iadd(&b, dst, off),
                       .write_mask = BITFIELD_MASK(comps), .align_mul = elem);
   }
   nir_pop_if(&b, NULL);
   return b.shader;
}

#define QPUSH(b, field, bits)                                                  \
   nir_load_push_constant(b, 1, bits, nir_imm_int(b, offsetof(struct meta_query_push, field)), \
                          .base = 0, .range = sizeof(struct meta_query_push))

/*
 * What vkCmdCopyQueryPoolResults says: for each query, the value if the
 * query is available or VK_QUERY_RESULT_PARTIAL_BIT is set (a 32-bit
 * result that does not fit is clamped), then the availability, as a 32 or
 * 64-bit word each. Waiting for the queries is not the shader's: the
 * dispatch is ordered after the work that sets them (cmd_query_fence).
 *
 * `reports` is the number of 64-bit counters a query's value is the sum of
 * (mali_query.h): 1 is the shader every pool had before occlusion queries
 * on v7, which sums the entries of one shader core ID each. The count is a
 * property of the device, so it is part of the shader, not a push constant.
 */
nir_shader *
MALI_PER_ARCH(build_query_copy_nir)(struct mali_device *dev, unsigned reports)
{
   assert(reports >= 1);
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, MALI_PER_ARCH(shader_nir_options)(dev, MESA_SHADER_COMPUTE),
      "mali_meta_query_copy");
   b.shader->info.workgroup_size[0] = META_WG_SIZE;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *id = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_push_if(&b, nir_ult(&b, id, QPUSH(&b, count, 32)));
   {
      nir_def *id64 = nir_u2u64(&b, id);
      nir_def *val_addr =
         nir_iadd(&b, QPUSH(&b, val, 64), nir_imul(&b, id64, nir_u2u64(&b, QPUSH(&b, val_stride, 32))));
      nir_def *avail_addr =
         nir_iadd(&b, QPUSH(&b, avail, 64),
                  nir_imul(&b, id64, nir_u2u64(&b, QPUSH(&b, avail_stride, 32))));
      nir_def *dst_addr =
         nir_iadd(&b, QPUSH(&b, dst, 64), nir_imul(&b, id64, nir_u2u64(&b, QPUSH(&b, dst_stride, 32))));
      nir_def *flags = QPUSH(&b, flags, 32);

      nir_def *avail = nir_ine_imm(&b, nir_load_global(&b, 1, 64, avail_addr, .align_mul = 8), 0);
      nir_def *wide = nir_ine_imm(&b, nir_iand_imm(&b, flags, VK_QUERY_RESULT_64_BIT), 0);
      nir_def *partial = nir_ine_imm(&b, nir_iand_imm(&b, flags, VK_QUERY_RESULT_PARTIAL_BIT), 0);

      nir_push_if(&b, nir_ior(&b, avail, partial));
      {
         nir_def *v = nir_load_global(&b, 1, 64, val_addr, .align_mul = 8);
         for (unsigned i = 1; i < reports; i++) {
            v = nir_iadd(&b, v, nir_load_global(&b, 1, 64, nir_iadd_imm(&b, val_addr, i * 8),
                                                .align_mul = 8));
         }
         nir_push_if(&b, wide);
         nir_store_global(&b, v, dst_addr, .write_mask = 1, .align_mul = 8);
         nir_push_else(&b, NULL);
         nir_store_global(&b, nir_u2u32(&b, nir_umin(&b, v, nir_imm_int64(&b, UINT32_MAX))),
                          dst_addr, .write_mask = 1, .align_mul = 4);
         nir_pop_if(&b, NULL);
      }
      nir_pop_if(&b, NULL);

      nir_push_if(&b, nir_ine_imm(&b, nir_iand_imm(&b, flags, VK_QUERY_RESULT_WITH_AVAILABILITY_BIT), 0));
      {
         nir_push_if(&b, wide);
         nir_store_global(&b, nir_b2i64(&b, avail), nir_iadd_imm(&b, dst_addr, 8),
                          .write_mask = 1, .align_mul = 8);
         nir_push_else(&b, NULL);
         nir_store_global(&b, nir_b2i32(&b, avail), nir_iadd_imm(&b, dst_addr, 4),
                          .write_mask = 1, .align_mul = 4);
         nir_pop_if(&b, NULL);
      }
      nir_pop_if(&b, NULL);
   }
   nir_pop_if(&b, NULL);
   return b.shader;
}

#if PAN_ARCH < 10
#define IPUSH(b, field, bits)                                                  \
   nir_load_push_constant(b, 1, bits,                                          \
                          nir_imm_int(b, offsetof(struct mali_meta_indirect_push, field)), \
                          .base = 0, .range = sizeof(struct mali_meta_indirect_push))

#if PAN_ARCH < 9
/* ceil(log2(v)) for v >= 1, as util_logbase2_ceil. */
static nir_def *
nir_logbase2_ceil(nir_builder *b, nir_def *v)
{
   nir_def *msb = nir_iadd_imm(b, nir_ufind_msb(b, nir_iadd_imm(b, v, -1)), 1);
   return nir_bcsel(b, nir_uge_imm(b, v, 2), msb, nir_imm_int(b, 0));
}
#endif

/*
 * vkCmdDispatchIndirect on the job manager (panlib_indirect_dispatch): one
 * invocation reads the VkDispatchIndirectCommand and makes the dispatch's
 * Compute job a Null job when a count is 0, else writes the counts into
 * the job and into the FAU words of num_work_groups the shader reads. v7
 * writes them into the Invocation section (mali_v7_pack_invocation, with
 * the size half done on the CPU); v9 into the three Workgroup count words
 * of the Compute payload, at push.inv. The job's type is written on every
 * run, so a re-submitted command buffer is patched again.
 */
static nir_shader *
build_indirect_dispatch_nir(struct mali_device *dev)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, MALI_PER_ARCH(shader_nir_options)(dev, MESA_SHADER_COMPUTE),
      "mali_meta_indirect_dispatch");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *cmd = IPUSH(&b, cmd, 64);
   nir_def *job = IPUSH(&b, job, 64);
   nir_def *n[3];
   for (unsigned i = 0; i < 3; i++)
      n[i] = nir_load_global(&b, 1, 32, nir_iadd_imm(&b, cmd, 4 * i), .align_mul = 4);
   nir_def *none = nir_ior(&b, nir_ieq_imm(&b, n[0], 0),
                           nir_ior(&b, nir_ieq_imm(&b, n[1], 0), nir_ieq_imm(&b, n[2], 0)));

   /* Job Header word 4: Type is bits 1-7 (Null 1, Compute 4). */
   nir_def *hdr = nir_iadd_imm(&b, job, 16);
   nir_def *w4 = nir_load_global(&b, 1, 32, hdr, .align_mul = 4);
   nir_def *type = nir_bcsel(&b, none, nir_imm_int(&b, 1 << 1), nir_imm_int(&b, 4 << 1));
   nir_store_global(&b, nir_ior(&b, nir_iand_imm(&b, w4, ~0xfeu), type), hdr,
                    .write_mask = 1, .align_mul = 4);

   nir_push_if(&b, nir_inot(&b, none));
   {
#if PAN_ARCH >= 9
      nir_def *counts = IPUSH(&b, inv, 64);
      for (unsigned i = 0; i < 3; i++)
         nir_store_global(&b, n[i], nir_iadd_imm(&b, counts, 4 * i), .write_mask = 1,
                          .align_mul = 4);
#else
      nir_def *s3 = IPUSH(&b, wgx_shift, 32);
      nir_def *s4 = nir_iadd(&b, s3, nir_logbase2_ceil(&b, n[0]));
      nir_def *s5 = nir_iadd(&b, s4, nir_logbase2_ceil(&b, n[1]));
      nir_def *inv0 = IPUSH(&b, inv0, 32);
      inv0 = nir_ior(&b, inv0, nir_ishl(&b, nir_iadd_imm(&b, n[0], -1), s3));
      inv0 = nir_ior(&b, inv0, nir_ishl(&b, nir_iadd_imm(&b, n[1], -1), s4));
      inv0 = nir_ior(&b, inv0, nir_ishl(&b, nir_iadd_imm(&b, n[2], -1), s5));
      /* Workgroups Y shift at bit 16, Workgroups Z shift at bit 22. */
      nir_def *inv1 = nir_ior(&b, IPUSH(&b, inv1, 32),
                              nir_ior(&b, nir_ishl_imm(&b, s4, 16), nir_ishl_imm(&b, s5, 22)));
      nir_store_global(&b, nir_vec2(&b, inv0, inv1), IPUSH(&b, inv, 64),
                       .write_mask = 0x3, .align_mul = 8);
#endif

      for (unsigned i = 0; i < 3; i++) {
         nir_def *a = nir_load_push_constant(
            &b, 1, 64, nir_imm_int(&b, offsetof(struct mali_meta_indirect_push, nwg) + 8 * i),
            .base = 0, .range = sizeof(struct mali_meta_indirect_push));
         nir_push_if(&b, nir_ine_imm(&b, a, 0));
         nir_store_global(&b, n[i], a, .write_mask = 1, .align_mul = 4);
         nir_pop_if(&b, NULL);
      }
   }
   nir_pop_if(&b, NULL);
   return b.shader;
}
#endif /* PAN_ARCH < 10 */

#if MALI_WITH_POLY && PAN_ARCH >= 10
#include "compiler/nir/nir_precompiled.h"
#include "compiler/spirv/nir_spirv.h"
#include "mali_tess_spv.h"

/*
 * The tessellator's kernels (vk/cl/mali_tess.cl): SPIR-V to a compute
 * shader the normal compile path takes, after Mesa's panfrost_compile
 * (src/panfrost/clc/pan_compile.c, which does it at build time for panvk's
 * internal kernels): the library, then a shader per kernel with its
 * arguments in push constants (struct mali_meta_tess_push), its functions
 * inlined and its memory access lowered to the generic address format.
 */
static const struct spirv_to_nir_options cl_spirv_options = {
   .environment = NIR_SPIRV_OPENCL,
   .shared_addr_format = nir_address_format_62bit_generic,
   .global_addr_format = nir_address_format_62bit_generic,
   .temp_addr_format = nir_address_format_62bit_generic,
   .constant_addr_format = nir_address_format_64bit_global,
   .create_library = true,
   .printf = true,
};

static void
cl_optimize(nir_shader *nir)
{
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_split_var_copies);
      NIR_PASS(progress, nir, nir_lower_var_copies);
      NIR_PASS(progress, nir, nir_lower_vars_to_ssa);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_deref);
      NIR_PASS(progress, nir, nir_opt_copy_prop_vars);
      NIR_PASS(progress, nir, nir_opt_undef);
      NIR_PASS(progress, nir, nir_lower_undef_to_zero, NULL);
   } while (progress);
}

static nir_def *
cl_kernel_arg(nir_builder *b, unsigned num_components, unsigned bit_size, unsigned offset_B)
{
   return nir_load_push_constant(b, num_components, bit_size, nir_imm_int(b, offset_B),
                                 .base = 0, .range = sizeof(struct mali_meta_tess_push));
}

static nir_shader *
build_tess_nir(struct mali_device *dev, enum mali_meta_shader m)
{
   static const char *const names[] = {
      [MALI_META_TESS_COUNT_ISOLINES] = "mali_tess_count_isolines",
      [MALI_META_TESS_COUNT_TRIANGLES] = "mali_tess_count_triangles",
      [MALI_META_TESS_COUNT_QUADS] = "mali_tess_count_quads",
      [MALI_META_TESS_WRITE_ISOLINES] = "mali_tess_write_isolines",
      [MALI_META_TESS_WRITE_TRIANGLES] = "mali_tess_write_triangles",
      [MALI_META_TESS_WRITE_QUADS] = "mali_tess_write_quads",
      [MALI_META_TESS_FINISH] = "mali_tess_finish",
   };
   const nir_shader_compiler_options *options =
      MALI_PER_ARCH(shader_nir_options)(dev, MESA_SHADER_COMPUTE);

   nir_shader *lib = spirv_to_nir(mali_tess_spv, ARRAY_SIZE(mali_tess_spv), NULL,
                                  MESA_SHADER_KERNEL, "library", &cl_spirv_options, options);
   if (!lib)
      return NULL;
   nir_fixup_is_exported(lib);
   NIR_PASS(_, lib, nir_lower_system_values);
   NIR_PASS(_, lib, nir_lower_calls_to_builtins);
   nir_lower_compute_system_values_options cs = {.global_id_is_32bit = true};
   NIR_PASS(_, lib, nir_lower_compute_system_values, &cs);
   NIR_PASS(_, lib, nir_lower_printf, &(const struct nir_lower_printf_options){0});
   NIR_PASS(_, lib, nir_lower_variable_initializers, nir_var_function_temp);
   NIR_PASS(_, lib, nir_lower_returns);
   NIR_PASS(_, lib, nir_inline_functions);
   NIR_PASS(_, lib, nir_opt_copy_prop);
   NIR_PASS(_, lib, nir_opt_deref);
   nir_lower_constant_to_temp(lib);
   NIR_PASS(_, lib, nir_lower_variable_initializers, ~0);
   NIR_PASS(_, lib, nir_lower_vec3_to_vec4,
            nir_var_shader_temp | nir_var_function_temp | nir_var_mem_shared |
               nir_var_mem_global | nir_var_mem_constant);
   NIR_PASS(_, lib, nir_lower_vars_to_explicit_types, nir_var_uniform,
            glsl_get_cl_type_size_align);
   NIR_PASS(_, lib, nir_lower_explicit_io, nir_var_uniform,
            nir_address_format_32bit_offset_as_64bit);

   nir_function *kernel = NULL;
   nir_foreach_entrypoint(f, lib) {
      if (!strcmp(f->name, names[m]))
         kernel = f;
   }
   if (!kernel) {
      ralloc_free(lib);
      return NULL;
   }

   const struct nir_precomp_opts opt = {.arg_align_B = 4};
   nir_shader *s = nir_precompiled_build_variant(kernel, MESA_SHADER_COMPUTE, 0, options, &opt,
                                                 cl_kernel_arg);
   nir_link_shader_functions(s, lib);
   NIR_PASS(_, s, nir_inline_functions);
   nir_remove_non_entrypoints(s);
   NIR_PASS(_, s, nir_opt_deref);
   NIR_PASS(_, s, nir_lower_vars_to_ssa);
   NIR_PASS(_, s, nir_remove_dead_derefs);
   NIR_PASS(_, s, nir_remove_dead_variables, nir_var_function_temp | nir_var_shader_temp, NULL);
   NIR_PASS(_, s, nir_lower_vars_to_explicit_types,
            nir_var_shader_temp | nir_var_function_temp | nir_var_mem_shared | nir_var_mem_global,
            glsl_get_cl_type_size_align);
   cl_optimize(s);
   NIR_PASS(_, s, nir_remove_dead_variables, nir_var_all, NULL);
   NIR_PASS(_, s, nir_lower_vars_to_explicit_types,
            nir_var_shader_temp | nir_var_function_temp | nir_var_mem_shared |
               nir_var_mem_global | nir_var_mem_constant,
            glsl_get_cl_type_size_align);
   NIR_PASS(_, s, nir_lower_memcpy);
   NIR_PASS(_, s, nir_lower_explicit_io, nir_var_mem_constant, nir_address_format_64bit_global);
   NIR_PASS(_, s, nir_lower_convert_alu_types, NULL);
   NIR_PASS(_, s, nir_opt_if, 0);
   NIR_PASS(_, s, nir_opt_idiv_const, 16);
   NIR_PASS(_, s, nir_opt_deref);
   NIR_PASS(_, s, nir_lower_vars_to_ssa);
   NIR_PASS(_, s, nir_lower_explicit_io,
            nir_var_shader_temp | nir_var_function_temp | nir_var_mem_shared | nir_var_mem_global,
            nir_address_format_62bit_generic);
   cl_optimize(s);
   NIR_PASS(_, s, nir_lower_vars_to_explicit_types, nir_var_shader_temp | nir_var_function_temp,
            glsl_get_cl_type_size_align);
   NIR_PASS(_, s, nir_lower_vars_to_explicit_types, nir_var_mem_shared,
            glsl_get_cl_type_size_align);
   ralloc_free(lib);
   return s;
}
#endif

#if PAN_ARCH >= 10
#define LPUSH(b, field, bits)                                                  \
   nir_load_push_constant(b, 1, bits,                                          \
                          nir_imm_int(b, offsetof(struct mali_meta_lines_push, field)), \
                          .base = 0, .range = sizeof(struct mali_meta_lines_push))

static nir_def *
lines_index(nir_builder *b, nir_def *src, nir_def *size, nir_def *pos)
{
   nir_def *addr = nir_iadd(b, src, nir_u2u64(b, nir_imul(b, pos, size)));
   nir_def *v;
   nir_push_if(b, nir_ieq_imm(b, size, 4));
   nir_def *v4 = nir_load_global(b, 1, 32, addr, .align_mul = 4);
   nir_push_else(b, NULL);
   nir_push_if(b, nir_ieq_imm(b, size, 2));
   nir_def *v2 = nir_u2u32(b, nir_load_global(b, 1, 16, addr, .align_mul = 2));
   nir_push_else(b, NULL);
   nir_def *v1 = nir_u2u32(b, nir_load_global(b, 1, 8, addr, .align_mul = 1));
   nir_pop_if(b, NULL);
   v = nir_if_phi(b, v2, v1);
   nir_pop_if(b, NULL);
   v = nir_if_phi(b, v4, v);
   return v;
}

/* Polygon mode line (mali_cmd_draw.c): one invocation per triangle. */
static nir_shader *
build_polygon_lines_nir(struct mali_device *dev)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, MALI_PER_ARCH(shader_nir_options)(dev, MESA_SHADER_COMPUTE),
      "mali_meta_polygon_lines");
   b.shader->info.workgroup_size[0] = META_WG_SIZE;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *t = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_push_if(&b, nir_ult(&b, t, LPUSH(&b, count, 32)));
   {
      nir_def *mode = LPUSH(&b, mode, 32);
      nir_def *first = LPUSH(&b, first, 32);
      nir_def *src = LPUSH(&b, src, 64);
      nir_def *size = LPUSH(&b, index_size, 32);
      nir_def *restart = LPUSH(&b, restart, 32);
      nir_def *list = nir_ieq_imm(&b, mode, MESA_PRIM_TRIANGLES);
      nir_def *fan = nir_ieq_imm(&b, mode, MESA_PRIM_TRIANGLE_FAN);
      /* List: 3t, 3t + 1, 3t + 2. Strip: t, t + 1, t + 2. Fan: 0, t + 1,
       * t + 2. */
      nir_def *base = nir_bcsel(&b, list, nir_imul_imm(&b, t, 3), t);
      nir_def *pos[3] = {
         nir_bcsel(&b, fan, nir_imm_int(&b, 0), base),
         nir_bcsel(&b, fan, nir_iadd_imm(&b, t, 1), nir_iadd_imm(&b, base, 1)),
         nir_bcsel(&b, fan, nir_iadd_imm(&b, t, 2), nir_iadd_imm(&b, base, 2)),
      };
      nir_def *idx[3];
      nir_def *restarted = nir_imm_false(&b);
      for (unsigned i = 0; i < 3; i++) {
         nir_def *p = nir_iadd(&b, first, pos[i]);
         nir_push_if(&b, nir_ine_imm(&b, src, 0));
         nir_def *loaded = lines_index(&b, src, size, p);
         nir_pop_if(&b, NULL);
         idx[i] = nir_if_phi(&b, loaded, p);
         restarted = nir_ior(&b, restarted, nir_iand(&b, nir_ine_imm(&b, restart, 0),
                                                     nir_ieq(&b, idx[i], restart)));
      }
      nir_def *none = nir_imm_int(&b, ~0u);
      nir_def *edges[6] = {idx[0], idx[1], idx[1], idx[2], idx[2], idx[0]};
      for (unsigned i = 0; i < 6; i++)
         edges[i] = nir_bcsel(&b, restarted, none, edges[i]);
      nir_def *dst = nir_iadd(&b, LPUSH(&b, dst, 64), nir_u2u64(&b, nir_imul_imm(&b, t, 24)));
      for (unsigned i = 0; i < 6; i++)
         nir_store_global(&b, edges[i], nir_iadd_imm(&b, dst, 4 * i), .write_mask = 0x1,
                          .align_mul = 4);
   }
   nir_pop_if(&b, NULL);
   return b.shader;
}
#endif

/* The counters per query of the device's occlusion pools that have more than
 * one (v7: one per shader core ID, mali_query.c query_pool_alloc). */
static unsigned
query_copy_reports(const struct mali_device *dev)
{
   return MAX2(1, util_last_bit64(dev->kbase->props.shader_present));
}

static const struct mali_shader *
get_meta(struct mali_cmd_buffer *cmd, enum mali_meta_shader m)
{
   struct mali_device *dev = cmd->dev;

   simple_mtx_lock(&dev->meta_lock);
   struct mali_shader *s = dev->meta[m];
   if (!s) {
      const unsigned reports = m == MALI_META_QUERY_COPY_SUM ? query_copy_reports(dev) : 1;
      nir_shader *nir = m == MALI_META_QUERY_COPY || m == MALI_META_QUERY_COPY_SUM
                           ? MALI_PER_ARCH(build_query_copy_nir)(dev, reports)
#if PAN_ARCH < 10
                        : m == MALI_META_INDIRECT_DISPATCH
                           ? build_indirect_dispatch_nir(dev)
                        : m == MALI_META_INDIRECT_DRAW
                           ? MALI_PER_ARCH(build_indirect_draw_nir)(dev)
#endif
#if PAN_ARCH < 9
                        : m == MALI_META_INDEX_MINMAX
                           ? MALI_PER_ARCH(build_index_minmax_nir)(dev)
#endif
#if PAN_ARCH >= 10
                        : m == MALI_META_POLYGON_LINES
                           ? build_polygon_lines_nir(dev)
#endif
#if MALI_WITH_POLY && PAN_ARCH >= 10
                        : m >= MALI_META_TESS_COUNT_ISOLINES && m <= MALI_META_TESS_FINISH
                           ? build_tess_nir(dev, m)
#endif
                           : build_meta_nir(dev, m);
      MALI_PER_ARCH(shader_preprocess)(dev, nir);

      const struct vk_pipeline_robustness_state rs = {
         .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
         .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
         .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
         .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
      };
      const struct mali_shader_compile_info info = {
         .nir = nir,
         .rs = &rs,
      };
      blake3_hash key;
      struct mesa_blake3 h;
      _mesa_blake3_init(&h);
      _mesa_blake3_update(&h, "mali-meta", 9);
      _mesa_blake3_update(&h, &m, sizeof(m));
      if (m == MALI_META_QUERY_COPY_SUM)
         _mesa_blake3_update(&h, &reports, sizeof(reports));
      _mesa_blake3_final(&h, key);

      VkResult r = MALI_PER_ARCH(shader_compile)(dev, &info, key, &s);
      if (r != VK_SUCCESS)
         s = NULL;
      dev->meta[m] = s;
   }
   simple_mtx_unlock(&dev->meta_lock);

   if (!s)
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   return s;
}

const struct mali_shader *
MALI_PER_ARCH(cmd_meta_shader)(struct mali_cmd_buffer *cmd, unsigned m)
{
   return get_meta(cmd, m);
}

void
MALI_PER_ARCH(meta_finish)(struct mali_device *dev)
{
   for (unsigned i = 0; i < MALI_META_COUNT; i++) {
      MALI_PER_ARCH(shader_unref)(dev, dev->meta[i]);
      dev->meta[i] = NULL;
   }
   MALI_PER_ARCH(meta_gfx_finish)(dev);
   MALI_PER_ARCH(blend_shaders_finish)(dev);
}

static void
dispatch_elems(struct mali_cmd_buffer *cmd, enum mali_meta_shader m,
               void *push, uint32_t push_size, uint32_t *count_field,
               uint64_t elems, uint64_t *src, uint64_t *dst)
{
   const struct mali_shader *s = get_meta(cmd, m);
   if (!s)
      return;
   const unsigned elem = meta_elem_size(m);
#if PAN_ARCH < 9
   /* The buffer memory written, for the v7 indexed draws
    * (mali_v7_note_buffer_write). */
   mali_v7_note_buffer_write(cmd, *dst, elems * elem);
#endif

   while (elems) {
      uint32_t n = (uint32_t)MIN2(elems, META_MAX_ELEMS);
      *count_field = n;
      const uint32_t base[3] = {0, 0, 0};
      const uint32_t groups[3] = {DIV_ROUND_UP(n, META_WG_SIZE), 1, 1};
      MALI_PER_ARCH(cmd_dispatch_shader)(cmd, s, NULL, push, push_size, base, groups);
      elems -= n;
      if (src)
         *src += (uint64_t)n * elem;
      *dst += (uint64_t)n * elem;
   }
}

static void
record_copy(struct mali_cmd_buffer *cmd, uint64_t src, uint64_t dst, uint64_t size)
{
   if (!size)
      return;
   enum mali_meta_shader m;
   uint64_t bits = src | dst | size;
   if (!(bits & 15))
      m = MALI_META_COPY_16;
   else if (!(bits & 3))
      m = MALI_META_COPY_4;
   else
      m = MALI_META_COPY_1;

   struct meta_copy_push push = {.src = src, .dst = dst};
   dispatch_elems(cmd, m, &push, sizeof(push), &push.count,
                  size / meta_elem_size(m), &push.src, &push.dst);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdCopyBuffer2)(VkCommandBuffer commandBuffer, const VkCopyBufferInfo2 *info)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_buffer, src, info->srcBuffer);
   VK_FROM_HANDLE(vk_buffer, dst, info->dstBuffer);

   /* One dispatch per region, as the blob. */
   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkBufferCopy2 *r = &info->pRegions[i];
      record_copy(cmd, vk_buffer_address(src, r->srcOffset),
                  vk_buffer_address(dst, r->dstOffset), r->size);
   }
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdFillBuffer)(VkCommandBuffer commandBuffer, VkBuffer dstBuffer,
                             VkDeviceSize dstOffset, VkDeviceSize size, uint32_t data)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_buffer, dst, dstBuffer);

   /* VK_WHOLE_SIZE: to the end, rounded down to a multiple of 4. */
   size = vk_buffer_range(dst, dstOffset, size) & ~3ull;
   if (!size)
      return;

   uint64_t addr = vk_buffer_address(dst, dstOffset);
   enum mali_meta_shader m =
      !((addr | size) & 15) ? MALI_META_FILL_16 : MALI_META_FILL_4;
   struct meta_fill_push push = {.dst = addr, .value = data};
   dispatch_elems(cmd, m, &push, sizeof(push), &push.count,
                  size / meta_elem_size(m), NULL, &push.dst);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdUpdateBuffer)(VkCommandBuffer commandBuffer, VkBuffer dstBuffer,
                               VkDeviceSize dstOffset, VkDeviceSize dataSize,
                               const void *pData)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_buffer, dst, dstBuffer);

   if (!dataSize)
      return;
   /* The data goes into command-buffer memory now, and a copy moves it
    * when the command buffer runs. */
   struct mali_ptr p = mali_cmd_alloc(cmd, dataSize, 16);
   if (!p.cpu)
      return;
   memcpy(p.cpu, pData, dataSize);
   record_copy(cmd, p.gpu, vk_buffer_address(dst, dstOffset), dataSize);
}

/* ---------------------------------------------------------------------- */
/* Query pools                                                             */

void
MALI_PER_ARCH(cmd_fill_addr)(struct mali_cmd_buffer *cmd, uint64_t addr, uint64_t size,
                             uint32_t value)
{
   if (!size)
      return;
   const enum mali_meta_shader m =
      !((addr | size) & 15) ? MALI_META_FILL_16 : MALI_META_FILL_4;
   struct meta_fill_push push = {.dst = addr, .value = value};
   dispatch_elems(cmd, m, &push, sizeof(push), &push.count, size / meta_elem_size(m), NULL,
                  &push.dst);
}

/* vkCmdCopyQueryPoolResults reaches here through the runtime's
 * vk_common_CmdCopyQueryPoolResults. */
VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdCopyQueryPoolResultsToMemoryKHR)(VkCommandBuffer commandBuffer,
                                                  VkQueryPool _pool, uint32_t firstQuery,
                                                  uint32_t queryCount,
                                                  const VkStridedDeviceAddressRangeKHR *pDstRange,
                                                  VkAddressCommandFlagsKHR dstFlags,
                                                  VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(mali_query_pool, pool, _pool);

   if (!queryCount)
      return;
   const struct mali_shader *s =
      get_meta(cmd, pool->val_count > 1 ? MALI_META_QUERY_COPY_SUM : MALI_META_QUERY_COPY);
   if (!s)
      return;
   assert(pool->val_count == 1 || pool->val_count == query_copy_reports(cmd->dev));

   /* A copy that waits is after every query operation recorded so far. */
   if (flags & VK_QUERY_RESULT_WAIT_BIT)
      MALI_PER_ARCH(cmd_query_fence)(cmd, true);

#if PAN_ARCH < 9
   const uint64_t result_size = flags & VK_QUERY_RESULT_64_BIT ? 8 : 4;
   const uint64_t per_query = result_size * (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT ? 2 : 1);
   mali_v7_note_buffer_write(cmd, pDstRange->address,
                             (queryCount - 1) * pDstRange->stride + per_query);
#endif

#if PAN_ARCH < 9
   /* The compute copy waits on the vtc slot for the fragment work that set
    * the queries (cmd_query_fence), and every later vtc atom behind it: on
    * the fragment side it follows that work in the slot order and holds up
    * no tiling (mali_cmd_image.c copy_to_buffer_frag). */
   if (mali_jm_cmd_transfer_waits_frag(cmd) && !cmd->gfx.render.active &&
       queryCount <= (uint32_t)MALI_META_QUERY_ROW * 4096) {
      struct mali_meta_fs_key key = {
         .query_reports = (uint8_t)pool->val_count,
      };
      key.rt_op[0] = MALI_META_FS_QUERY;
      key.rt_type[0] = MALI_META_FS_UINT;
      struct mali_meta_fs_push fp = {
         .query_val = mali_query_val_va(pool, firstQuery),
         .query_avail = mali_query_avail_va(pool, firstQuery),
         .query_dst = pDstRange->address,
         .query_val_stride = pool->val_stride,
         .query_avail_stride = pool->avail_stride,
         .query_dst_stride = (uint32_t)pDstRange->stride,
         .query_count = queryCount,
         .query_flags = flags,
      };
      const struct mali_render_blit blit = {.key = &key, .push = &fp};
      const uint32_t w = MIN2(queryCount, (uint32_t)MALI_META_QUERY_ROW);
      const uint32_t h = DIV_ROUND_UP(queryCount, (uint32_t)MALI_META_QUERY_ROW);
      /* At least a 16x16 framebuffer: the fragments past the queries do
       * nothing. */
      struct mali_render_desc d = {
         .width = MAX2(w, 16),
         .height = MAX2(h, 16),
         .area = {{0, 0}, {w, h}},
         .layer_count = 1,
         .rt_count = 0,
         .blit = &blit,
      };
      MALI_PER_ARCH(cmd_render_begin)(cmd, &d);
      MALI_PER_ARCH(cmd_render_end)(cmd);
      mali_jm_cmd_mark_frag_transfer(cmd);
      return;
   }
#endif

   struct meta_query_push push = {
      .val = mali_query_val_va(pool, firstQuery),
      .avail = mali_query_avail_va(pool, firstQuery),
      .dst = pDstRange->address,
      .val_stride = pool->val_stride,
      .avail_stride = pool->avail_stride,
      .dst_stride = (uint32_t)pDstRange->stride,
      .flags = flags,
   };
   uint32_t done = 0;
   while (done < queryCount) {
      const uint32_t n = MIN2(queryCount - done, META_MAX_ELEMS);
      push.count = n;
      const uint32_t base[3] = {0, 0, 0};
      const uint32_t groups[3] = {DIV_ROUND_UP(n, META_WG_SIZE), 1, 1};
      MALI_PER_ARCH(cmd_dispatch_shader)(cmd, s, NULL, &push, sizeof(push), base, groups);
      push.val += (uint64_t)n * pool->val_stride;
      push.avail += (uint64_t)n * pool->avail_stride;
      push.dst += (uint64_t)n * pDstRange->stride;
      done += n;
   }
}
