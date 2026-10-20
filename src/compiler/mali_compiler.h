/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The backend end of the shader compile path: Mesa's panfrost compiler with
 * kraid as the backend on v9 and v11, and Mesa's Bifrost backend on v7
 * (pan_use_kraid() answers false below arch 9). The Vulkan layer
 * (src/vk/mali_shader.c) takes SPIR-V through vtn, the Vulkan runtime and
 * its own NIR lowering; this file does the checks around that and the
 * final NIR-to-machine-code step.
 *
 * Two things here exist because the vendored compiler can take the
 * process down:
 *
 *  - mali_spirv_structure_ok() checks a module's instruction stream before
 *    vtn parses it: vtn at the pinned Mesa commit dereferences NULL or
 *    asserts on a few malformed shapes instead of returning an error.
 *  - mali_compile_nir() runs the backend with a guard in front of kraid:
 *    kraid panics on NIR it does not handle, and a Rust panic across the C
 *    boundary aborts. The guard refuses such a shader instead.
 */

#ifndef MALI_COMPILER_H
#define MALI_COMPILER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "panfrost/compiler/pan_compiler.h"
#include "util/u_dynarray.h"

#ifdef __cplusplus
extern "C" {
#endif

/* GPU_ID register value of the Mali-G615 MC6 in the RG 477V: arch 11.8,
 * product 3, r1p3. */
#define MALI_GPU_ID_G615_R1P3 0xb8a31030u

enum mali_compile_status {
   MALI_COMPILE_OK = 0,
   /* The SPIR-V module is malformed or uses something vtn rejects. */
   MALI_COMPILE_INVALID_SPIRV,
   /* Valid input that this compiler path does not handle. */
   MALI_COMPILE_UNSUPPORTED,
   /* The Mesa compiler did not come up configured for kraid. */
   MALI_COMPILE_NO_KRAID,
   MALI_COMPILE_OUT_OF_MEMORY,
};

const char *mali_compile_status_string(enum mali_compile_status status);

/*
 * One-time process setup (GLSL type singleton), then, from arch 9, a check
 * that Mesa's compiler selects kraid for every stage of this architecture.
 * Below arch 9 there is nothing to check: Mesa's Bifrost backend compiles.
 * Thread-safe; cheap after the first call. Returns MALI_COMPILE_NO_KRAID
 * if the build lost the kraid selection
 * (mesa-glue/src/panfrost/compiler/meson.build).
 */
enum mali_compile_status mali_compiler_init(uint64_t gpu_id);

/*
 * Structural check of a SPIR-V module before vtn sees it: header (magic,
 * version, id bound, schema), instruction lengths, OpFunction /
 * OpFunctionEnd pairing, and entry points that name a function. Not a
 * validator: semantically invalid SPIR-V is outside the Vulkan contract.
 */
bool mali_spirv_structure_ok(const uint32_t *words, size_t word_count);

/*
 * pan_shader_compile() behind the kraid guard, which only v9 and v11
 * shaders reach: on v7 pan_shader_compile() never calls kraid. Consumes
 * nothing: the caller still owns nir. On MALI_COMPILE_UNSUPPORTED, why (if
 * not NULL) says what kraid would not have handled.
 */
enum mali_compile_status
mali_compile_nir(nir_shader *nir, struct pan_compile_inputs *inputs,
                 struct util_dynarray *binary, struct pan_shader_info *info,
                 char *why, size_t why_size);

/* Disassemble compiled code for the given GPU: kraid's on v9 and v11, the
 * Bifrost backend's on v7 (pan_disassemble picks by arch). */
void mali_disassemble(FILE *fp, uint64_t gpu_id, const void *code, size_t size);

/* Number of shaders the kraid guard refused, over the whole process
 * (tests). */
unsigned mali_kraid_guard_refusals(void);

#ifdef __cplusplus
}
#endif

#endif /* MALI_COMPILER_H */
