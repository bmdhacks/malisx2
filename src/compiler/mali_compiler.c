/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

#include "mali_compiler.h"

#include <stdlib.h>
#include <string.h>

#include "compiler/glsl_types.h"
#include "compiler/nir/nir.h"
#include "panfrost/compiler/kraid/kraid.h"
#include "panfrost/model/pan_model.h"
#include "util/bitset.h"
#include "util/log.h"
#include "util/u_atomic.h"
#include "util/u_call_once.h"

/* Generated from kraid/nir.rs by gen_kraid_support.py. */
#include "mali_kraid_support.h"

#define SPIRV_MAGIC 0x07230203u
#define SPIRV_HEADER_WORDS 5

/*
 * Which NIR kraid's translator accepts, as bitsets over NIR's enums. Built
 * once from the generated lists.
 */
static struct {
   BITSET_DECLARE(alu, nir_num_opcodes);
   BITSET_DECLARE(intrinsics, nir_num_intrinsics);
   BITSET_DECLARE(texops, 64);
   BITSET_DECLARE(tex_srcs, nir_num_tex_src_types);
   BITSET_DECLARE(instr_types, 16);
} kraid_support;

static util_once_flag compiler_once = UTIL_ONCE_FLAG_INIT;
static unsigned guard_refusals;

static void
compiler_init_once(void)
{
   glsl_type_singleton_init_or_ref();

   for (unsigned i = 0; i < ARRAY_SIZE(mali_kraid_alu_ops); i++)
      BITSET_SET(kraid_support.alu, mali_kraid_alu_ops[i]);
   for (unsigned i = 0; i < ARRAY_SIZE(mali_kraid_intrinsics); i++)
      BITSET_SET(kraid_support.intrinsics, mali_kraid_intrinsics[i]);
   for (unsigned i = 0; i < ARRAY_SIZE(mali_kraid_texops); i++)
      BITSET_SET(kraid_support.texops, mali_kraid_texops[i]);
   for (unsigned i = 0; i < ARRAY_SIZE(mali_kraid_tex_srcs); i++)
      BITSET_SET(kraid_support.tex_srcs, mali_kraid_tex_srcs[i]);
   for (unsigned i = 0; i < ARRAY_SIZE(mali_kraid_instr_types); i++)
      BITSET_SET(kraid_support.instr_types, mali_kraid_instr_types[i]);
}

/*
 * The panfrost compiler picks kraid per stage through pan_use_kraid(). Our
 * build of pan_compiler.c answers "all" for the PAN_USE_KRAID option
 * without looking at the environment (mesa-glue/src/panfrost/compiler/
 * meson.build), so this only checks that the build did its job.
 */
enum mali_compile_status
mali_compiler_init(uint64_t gpu_id)
{
   util_call_once(&compiler_once, compiler_init_once);

   const unsigned arch = pan_arch(gpu_id);
   /* No kraid on v7: pan_use_kraid() answers false below arch 9, so
    * pan_shader_compile() uses Mesa's Bifrost backend. Nothing to check. */
   if (arch < 9)
      return MALI_COMPILE_OK;
   const mesa_shader_stage stages[] = {
      MESA_SHADER_VERTEX, MESA_SHADER_FRAGMENT, MESA_SHADER_COMPUTE,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(stages); i++) {
      if (!pan_use_kraid(arch, stages[i], false) ||
          !pan_use_kraid(arch, stages[i], true))
         return MALI_COMPILE_NO_KRAID;
   }
   return MALI_COMPILE_OK;
}

/* ---------------------------------------------------------------------- */
/* The kraid guard                                                          */

/*
 * The final NIR only exists inside pan_shader_compile(): bifrost_nir.c runs
 * its late passes and then calls kraid_compile_nir(). The link wraps that
 * call (-Wl,--wrap=kraid_compile_nir, src/meson.build), so the check below
 * sees exactly what kraid would translate. A refused shader produces no
 * code; the thread-local state tells mali_compile_nir() why.
 */
struct kraid_guard {
   bool refused;
   char why[128];
};

static __thread struct kraid_guard *tls_guard;

void __real_kraid_compile_nir(nir_shader *nir,
                              const struct pan_compile_inputs *inputs,
                              struct util_dynarray *binary,
                              struct pan_shader_info *info,
                              enum kraid_idvs_mode idvs);
void __wrap_kraid_compile_nir(nir_shader *nir,
                              const struct pan_compile_inputs *inputs,
                              struct util_dynarray *binary,
                              struct pan_shader_info *info,
                              enum kraid_idvs_mode idvs);

/* NULL if kraid's translator handles every instruction of nir, else the
 * first one it does not, printed into buf. */
static const char *
kraid_unsupported(const nir_shader *nir, char *buf, size_t size)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (!BITSET_TEST(kraid_support.instr_types, instr->type)) {
               snprintf(buf, size, "instruction type %u", instr->type);
               return buf;
            }
            switch (instr->type) {
            case nir_instr_type_alu: {
               const nir_alu_instr *alu = nir_instr_as_alu(instr);
               if (!BITSET_TEST(kraid_support.alu, alu->op)) {
                  snprintf(buf, size, "ALU op %s", nir_op_infos[alu->op].name);
                  return buf;
               }
               break;
            }
            case nir_instr_type_intrinsic: {
               const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (!BITSET_TEST(kraid_support.intrinsics, intr->intrinsic)) {
                  snprintf(buf, size, "intrinsic %s",
                           nir_intrinsic_infos[intr->intrinsic].name);
                  return buf;
               }
               break;
            }
            case nir_instr_type_tex: {
               const nir_tex_instr *tex = nir_instr_as_tex(instr);
               if (tex->op >= 64 || !BITSET_TEST(kraid_support.texops, tex->op)) {
                  snprintf(buf, size, "texture op %u", tex->op);
                  return buf;
               }
               for (unsigned i = 0; i < tex->num_srcs; i++) {
                  if (!BITSET_TEST(kraid_support.tex_srcs, tex->src[i].src_type)) {
                     snprintf(buf, size, "texture source type %u",
                              tex->src[i].src_type);
                     return buf;
                  }
               }
               break;
            }
            default:
               break;
            }
         }
      }
   }
   return NULL;
}

void
__wrap_kraid_compile_nir(nir_shader *nir, const struct pan_compile_inputs *inputs,
                         struct util_dynarray *binary, struct pan_shader_info *info,
                         enum kraid_idvs_mode idvs)
{
   struct kraid_guard *g = tls_guard;
   char buf[128];

   /* After a refusal, the second IDVS variant is refused too. */
   if (g && g->refused)
      return;

   const char *why = kraid_unsupported(nir, buf, sizeof(buf));
   if (why) {
      p_atomic_inc(&guard_refusals);
      if (g) {
         g->refused = true;
         snprintf(g->why, sizeof(g->why), "%s", why);
      } else {
         mesa_loge("kraid guard: refusing a %s shader outside mali_compile_nir: %s",
                   mesa_shader_stage_name(nir->info.stage), why);
      }
      return;
   }

   __real_kraid_compile_nir(nir, inputs, binary, info, idvs);
}

unsigned
mali_kraid_guard_refusals(void)
{
   return p_atomic_read(&guard_refusals);
}

enum mali_compile_status
mali_compile_nir(nir_shader *nir, struct pan_compile_inputs *inputs,
                 struct util_dynarray *binary, struct pan_shader_info *info,
                 char *why, size_t why_size)
{
   struct kraid_guard guard = {0};

   tls_guard = &guard;
   pan_shader_compile(nir, inputs, binary, info);
   tls_guard = NULL;

   if (guard.refused) {
      if (why)
         snprintf(why, why_size, "kraid does not handle %s", guard.why);
      return MALI_COMPILE_UNSUPPORTED;
   }
   return MALI_COMPILE_OK;
}

/* ---------------------------------------------------------------------- */
/* SPIR-V structure check                                                   */

/*
 * vtn reports most problems through its error path, but a few malformed
 * shapes reach plain assert()s or out-of-bounds reads instead (a module
 * truncated inside a function body, for one), and at the pinned commit
 * vtn_create_builder() reports a bad version or schema word through
 * vtn_err() before its options pointer is set and crashes. This walks the
 * instruction stream and rejects:
 *   - a header with a bad magic, version, id bound or schema word;
 *   - an instruction with a zero word count or one running past the end;
 *   - an OpFunction without its OpFunctionEnd, or nested functions;
 *   - an OpEntryPoint whose target is not an OpFunction, or no entry point.
 */
#define SPIRV_OP_ENTRY_POINT  15
#define SPIRV_OP_FUNCTION     54
#define SPIRV_OP_FUNCTION_END 56
#define SPIRV_MAX_ID_BOUND    (1u << 22)

bool
mali_spirv_structure_ok(const uint32_t *words, size_t word_count)
{
   if (!words || word_count <= SPIRV_HEADER_WORDS || words[0] != SPIRV_MAGIC ||
       words[1] < 0x10000 || words[1] > 0x10600 || words[4] != 0)
      return false;

   const uint32_t bound = words[3];
   if (bound == 0 || bound > SPIRV_MAX_ID_BOUND)
      return false;

   uint8_t *is_function = calloc((bound + 7) / 8, 1);
   size_t entry_cap = 16, entry_count = 0;
   uint32_t *entries = malloc(entry_cap * sizeof(*entries));
   bool ok = is_function && entries;
   bool in_function = false;

   for (size_t i = SPIRV_HEADER_WORDS; ok && i < word_count;) {
      const uint32_t opcode = words[i] & 0xffff;
      const uint32_t len = words[i] >> 16;

      if (len == 0 || len > word_count - i) {
         ok = false;
         break;
      }

      switch (opcode) {
      case SPIRV_OP_ENTRY_POINT:
         if (len < 4) {
            ok = false;
            break;
         }
         if (entry_count == entry_cap) {
            entry_cap *= 2;
            uint32_t *grown = realloc(entries, entry_cap * sizeof(*entries));
            if (!grown) {
               ok = false;
               break;
            }
            entries = grown;
         }
         entries[entry_count++] = words[i + 2];
         break;
      case SPIRV_OP_FUNCTION:
         if (len != 5 || in_function || words[i + 2] >= bound) {
            ok = false;
            break;
         }
         is_function[words[i + 2] / 8] |= 1u << (words[i + 2] % 8);
         in_function = true;
         break;
      case SPIRV_OP_FUNCTION_END:
         if (!in_function)
            ok = false;
         in_function = false;
         break;
      default:
         break;
      }
      i += len;
   }

   if (ok && (in_function || entry_count == 0))
      ok = false;
   for (size_t e = 0; ok && e < entry_count; e++) {
      const uint32_t id = entries[e];
      if (id >= bound || !(is_function[id / 8] & (1u << (id % 8))))
         ok = false;
   }

   free(entries);
   free(is_function);
   return ok;
}

/* ---------------------------------------------------------------------- */

void
mali_disassemble(FILE *fp, uint64_t gpu_id, const void *code, size_t size)
{
   pan_disassemble(fp, code, size, gpu_id, false);
}

const char *
mali_compile_status_string(enum mali_compile_status status)
{
   switch (status) {
   case MALI_COMPILE_OK:
      return "ok";
   case MALI_COMPILE_INVALID_SPIRV:
      return "invalid SPIR-V";
   case MALI_COMPILE_UNSUPPORTED:
      return "unsupported";
   case MALI_COMPILE_NO_KRAID:
      return "kraid backend not selected";
   case MALI_COMPILE_OUT_OF_MEMORY:
      return "out of memory";
   }
   return "unknown";
}
