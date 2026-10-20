/*
 * SPDX-FileCopyrightText: 2026 TheGammaSqueeze
 * SPDX-License-Identifier: MIT
 */

/*
 * The fields of the Bifrost (v7) Renderer State Descriptor that come from
 * a compiled shader, for the static vertex and compute RSDs of the shader
 * upload (mali_shader.c) and the fragment RSD, built per pipeline and draw
 * (mali_pipeline_state.c, mali_jm_cmd_draw_v7.c).
 *
 * Ported from Mesa's src/panfrost/lib/pan_shader.h at PAN_ARCH 7
 * (pan_shader_prepare_rsd and its helpers; MIT, Collabora). That header
 * is not included because it pulls in pan_props.h and the kmod layer.
 *
 * Words of the 64-byte descriptor (v7.xml):
 *   0-1  shader pointer, 2 sampler and texture counts, 3 attribute and
 *        varying counts
 *   4    properties (shader and draw bits mixed)
 *   5-7  depth units, factor, bias clamp
 *   8    multisample misc, 9 stencil mask misc, 10-11 stencil front/back
 *   12   preload, 13 secondary preload, 14-15 secondary shader (word 15
 *        shared with the two message preloads)
 * The pipeline bakes words 4 to 11 from its static state and the fragment
 * shader; the draw packs the shader's fields and copies the baked words
 * over its own words 4 to 11.
 *
 * Also the register file format of a colour type, which the Blend
 * descriptors behind the fragment RSD (mali_jm_cmd_draw_v7.c) and the
 * blend shaders' tile conversions (mali_blend.c) name.
 */

#ifndef MALI_RSD_V7_H
#define MALI_RSD_V7_H

#if !defined(PAN_ARCH) || PAN_ARCH >= 9
#error "mali_rsd_v7.h is for the Bifrost (PAN_ARCH 7) build only"
#endif

#include "genxml/gen_macros.h"

#include "panfrost/compiler/pan_compiler.h"
#include "util/macros.h"

#define MALI_V7_RSD_STATE_WORD_FIRST 4
#define MALI_V7_RSD_STATE_WORDS 8
/* Word 8, Multisample Misc: the draw ORs in per-sample evaluation for a
 * blend shader under multisampling (mali_jm_cmd_draw_v7.c). */
#define MALI_V7_RSD_MULTISAMPLE_MISC_WORD 8

/* pan_blend.c get_register_format: the register file format of a colour
 * type (8-bit types use 16-bit registers). Any other type, such as 0 for
 * a colour output the fragment shader does not write, takes F32. */
static inline enum mali_register_file_format
mali_v7_register_format(nir_alu_type type)
{
   switch (type) {
   case nir_type_float16: return MALI_REGISTER_FILE_FORMAT_F16;
   case nir_type_int8:
   case nir_type_int16: return MALI_REGISTER_FILE_FORMAT_I16;
   case nir_type_uint8:
   case nir_type_uint16: return MALI_REGISTER_FILE_FORMAT_U16;
   case nir_type_int32: return MALI_REGISTER_FILE_FORMAT_I32;
   case nir_type_uint32: return MALI_REGISTER_FILE_FORMAT_U32;
   case nir_type_float32:
   default: return MALI_REGISTER_FILE_FORMAT_F32;
   }
}

static inline enum mali_shader_register_allocation
mali_v7_register_allocation(unsigned work_reg_count)
{
   return work_reg_count <= 32 ? MALI_SHADER_REGISTER_ALLOCATION_32_PER_THREAD
                               : MALI_SHADER_REGISTER_ALLOCATION_64_PER_THREAD;
}

/* pan_make_preload: which of r55..r62 the shader expects preloaded. */
static inline void
mali_v7_preload(mesa_shader_stage stage, uint64_t preload, struct MALI_PRELOAD *out)
{
#define PRELOADS(reg) ((preload & BITFIELD64_BIT(reg)) != 0)
   switch (stage) {
   case MESA_SHADER_VERTEX:
      out->vertex.position_result_address_lo = PRELOADS(58);
      out->vertex.position_result_address_hi = PRELOADS(59);
      out->vertex.vertex_id = PRELOADS(61);
      out->vertex.instance_id = PRELOADS(62);
      break;
   case MESA_SHADER_FRAGMENT:
      out->fragment.primitive_id = PRELOADS(57);
      out->fragment.primitive_flags = PRELOADS(58);
      out->fragment.fragment_position = PRELOADS(59);
      out->fragment.sample_mask_id = PRELOADS(61);
      out->fragment.coverage = true;
      break;
   default:
      out->compute.local_invocation_xy = PRELOADS(55);
      out->compute.local_invocation_z = PRELOADS(56);
      out->compute.work_group_x = PRELOADS(57);
      out->compute.work_group_y = PRELOADS(58);
      out->compute.work_group_z = PRELOADS(59);
      out->compute.global_invocation_x = PRELOADS(60);
      out->compute.global_invocation_y = PRELOADS(61);
      out->compute.global_invocation_z = PRELOADS(62);
      break;
   }
#undef PRELOADS
}

/* pan_pack_message_preload: an LD_VAR or VAR_TEX the hardware issues
 * before the fragment shader starts (bi_opt_message_preload). */
static inline void
mali_v7_message_preload(struct MALI_MESSAGE_PRELOAD *cfg, const struct bifrost_message_preload *msg)
{
   const enum mali_message_preload_register_format regfmt =
      msg->fp16 ? MALI_MESSAGE_PRELOAD_REGISTER_FORMAT_F16
                : MALI_MESSAGE_PRELOAD_REGISTER_FORMAT_F32;

   if (msg->enabled && msg->texture) {
      cfg->type = MALI_MESSAGE_TYPE_VAR_TEX;
      cfg->var_tex.varying_index = msg->varying_index;
      cfg->var_tex.texture_index = msg->texture_index;
      cfg->var_tex.register_format = regfmt;
      cfg->var_tex.skip = msg->skip;
      cfg->var_tex.zero_lod = msg->zero_lod;
   } else if (msg->enabled) {
      cfg->type = MALI_MESSAGE_TYPE_LD_VAR;
      cfg->ld_var.varying_index = msg->varying_index;
      cfg->ld_var.register_format = regfmt;
      cfg->ld_var.num_components = msg->num_components;
   } else {
      cfg->type = MALI_MESSAGE_TYPE_DISABLED;
   }
}

/*
 * pan_shader_prepare_rsd for Bifrost: the shader's own fields. `code` is
 * the GPU address of the machine code (0 in the pipeline bake, which
 * keeps only the state words). The counts are the ones the compile
 * patched (mali_shader.c v7_fixup_info). Vertex shaders are compiled
 * without IDVS, so there is no secondary shader.
 */
static inline void
mali_v7_rsd_shader(const struct pan_shader_info *info, uint64_t code,
                   struct MALI_RENDERER_STATE *rsd)
{
   rsd->shader.shader = code;
   rsd->shader.attribute_count = info->attribute_count;
   rsd->shader.varying_count = info->varyings.formats.count;
   rsd->shader.texture_count = info->texture_count;
   rsd->shader.sampler_count = info->sampler_count;
   rsd->properties.shader_contains_barrier = info->contains_barrier;
   rsd->properties.uniform_buffer_count = info->ubo_count;

   if (info->stage == MESA_SHADER_FRAGMENT) {
      rsd->properties.stencil_from_shader = info->fs.writes_stencil;
      rsd->properties.depth_source = info->fs.writes_depth ? MALI_DEPTH_SOURCE_SHADER
                                                           : MALI_DEPTH_SOURCE_FIXED_FUNCTION;
      /* For a draw, the pipeline bake sets this bit (mali_pipeline_state.c). */
      rsd->multisample_misc.evaluate_per_sample = info->fs.sample_shading;
   }

   /* pan_shader_prepare_bifrost_rsd. */
   rsd->preload.uniform_count = DIV_ROUND_UP(info->fau.count, 2);
   rsd->properties.shader_register_allocation = mali_v7_register_allocation(info->work_reg_count);
   mali_v7_preload(info->stage, info->preload, &rsd->preload);

   if (info->stage == MESA_SHADER_FRAGMENT) {
      rsd->properties.shader_modifies_coverage = info->fs.writes_coverage || info->fs.can_discard;
      rsd->properties.allow_forward_pixel_to_be_killed = !info->writes_global;
      rsd->properties.shader_wait_dependency_6 = info->bifrost.wait_6;
      rsd->properties.shader_wait_dependency_7 = info->bifrost.wait_7;
      mali_v7_message_preload(&rsd->message_preload_1, &info->bifrost.messages[0]);
      mali_v7_message_preload(&rsd->message_preload_2, &info->bifrost.messages[1]);
   } else if (info->stage == MESA_SHADER_VERTEX) {
      /* Compiled without IDVS (mali_shader.c). */
      assert(!info->vs.secondary_enable);
   }
}

#endif
