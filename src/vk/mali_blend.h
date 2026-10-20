/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Blend shaders: the blends the fixed-function blender cannot do
 * (MIN/MAX, factor pairs it has no operand for, formats it cannot blend,
 * logic ops, alpha-to-one, inhomogeneous blend constants) run as a small
 * shader the fragment shader's BLEND instruction calls. Built from NIR
 * the way Mesa's pan_blend_create_shader does, compiled through kraid,
 * placed in the device's executable pool and cached per device by key.
 * Built per arch (v9 and v11 blend the same way; the shader is compiled
 * for the device's GPU). v7 compiles it with Mesa's Bifrost backend, and
 * its Blend descriptor carries the address the shader returns to.
 */

#ifndef MALI_BLEND_H
#define MALI_BLEND_H

#include <stdint.h>

#include "vulkan/vulkan_core.h"

#include "compiler/nir/nir.h"
#include "util/format/u_formats.h"

#include "mali_arch.h"
#include "mali_pipeline.h"

struct mali_device;

/*
 * Everything a blend shader depends on. Built with memset(0) first: the
 * cache hashes and compares the bytes.
 */
struct mali_blend_shader_key {
   uint32_t format;              /* enum pipe_format of the render target */
   uint32_t src0_type;           /* nir_alu_type of the colour output */
   uint32_t src1_type;           /* of the second (dual-source) output, or 0 */
   uint8_t rt;                   /* render target index */
   uint8_t nr_samples;
   uint8_t logicop_enable;
   uint8_t logicop_func;         /* enum pipe_logicop */
   uint8_t alpha_to_one;
   struct mali_blend_eq eq;
};

/* The key for render target `rt` of a pipeline with baked blend state
 * `bk` and fragment shader info `fs_info`, drawing to `format` with
 * `samples` samples. */
MALI_PER_ARCH_DECL(void, blend_shader_key_init,
                   (struct mali_blend_shader_key *key, const struct mali_gfx_baked *bk,
                    unsigned rt, enum pipe_format format, unsigned samples,
                    const struct pan_shader_info *fs_info));

/* Compiles the blend shaders a new pipeline's static state asks for and
 * keeps their addresses in p->blend_shader, so that its draws need no
 * cache lookup (the blob builds them at pipeline creation too). A failure
 * is left for the draw to report. */
struct mali_graphics_pipeline;
MALI_PER_ARCH_DECL(void, blend_shaders_prepare,
                   (struct mali_device *dev, struct mali_graphics_pipeline *p));

/* The blend shader for `key` in NIR (exposed for the host test). */
MALI_PER_ARCH_DECL(nir_shader *, blend_shader_nir, (const struct mali_blend_shader_key *key));

/* The GPU address of the compiled blend shader for `key`, compiled and
 * uploaded on first use. Thread-safe. */
MALI_PER_ARCH_DECL(VkResult, blend_shader_get,
                   (struct mali_device *dev, const struct mali_blend_shader_key *key,
                    uint64_t *addr));

/* Number of blend shaders the device has compiled. */
MALI_PER_ARCH_DECL(unsigned, blend_shader_count, (struct mali_device *dev));

/* Frees the device's blend shaders (vkDestroyDevice). */
MALI_PER_ARCH_DECL(void, blend_shaders_finish, (struct mali_device *dev));

#endif /* MALI_BLEND_H */
