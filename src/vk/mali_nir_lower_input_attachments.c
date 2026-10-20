/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * subpassLoad(), following panvk (Mesa src/panfrost/vulkan/
 * panvk_vX_nir_lower_input_attachment_loads.c, MIT). ARMSX2 reads its
 * colour and depth attachments this way on Mali ("feedback" descriptors).
 * On v7 the depth and stencil tile reads also take a conversion word: the
 * map's for depth, a constant the v7 compile adds for stencil.
 *
 * An input attachment that is also a colour or depth/stencil attachment of
 * the current render pass is read from the tile buffer (LD_TILE); anything
 * else is read as an image. Which case applies is decided at draw time
 * through the iam system values (struct mali_graphics_sysvals): target ~0
 * means "not an attachment of this pass, read the image", MALI_ZS_ATTACHMENT
 * the depth/stencil tile, anything else a colour tile with the given
 * conversion. A colour attachment the shader never writes is read with the
 * read-only variant.
 */

#include "mali_shader.h"

#include "nir_builder.h"
#include "panfrost/compiler/pan_nir.h"
#include "vk_graphics_state.h"
#include "util/u_dynarray.h"

struct ia_ctx {
   unsigned arch;
   uint32_t ro_color_mask;
   uint32_t input_attachment_read;
};

static bool
collect_frag_writes(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_deref)
      return false;

   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   if (deref->modes != nir_var_shader_out)
      return false;

   nir_variable *var = nir_deref_instr_get_variable(deref);
   if (var->data.location < FRAG_RESULT_DATA0 ||
       var->data.location > FRAG_RESULT_DATA7)
      return false;

   *(uint32_t *)data |= BITFIELD_BIT(var->data.location - FRAG_RESULT_DATA0);
   return true;
}

/* Colour attachments mapped as input attachments that the shader does not
 * write. */
static uint32_t
readonly_color_mask(nir_shader *nir, const struct vk_graphics_pipeline_state *state)
{
   if (!state || !state->ial || !state->cal)
      return 0;

   uint32_t in_mask = 0, out_mask = 0;
   for (uint32_t i = 0; i < ARRAY_SIZE(state->ial->color_map) &&
                        i < state->ial->color_attachment_count; i++) {
      if (state->ial->color_map[i] != MESA_VK_ATTACHMENT_UNUSED)
         in_mask |= BITFIELD_BIT(i);
   }

   nir_shader_intrinsics_pass(nir, collect_frag_writes, nir_metadata_all, &out_mask);

   for (uint32_t i = 0; i < ARRAY_SIZE(state->cal->color_map); i++) {
      if (state->ial->color_map[i] == MESA_VK_ATTACHMENT_UNUSED)
         out_mask &= ~BITFIELD_BIT(i);
   }

   return in_mask & ~out_mask;
}

static bool
lower_load(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct ia_ctx *ctx = data;

   if (intr->intrinsic != nir_intrinsic_image_deref_load &&
       intr->intrinsic != nir_intrinsic_image_deref_sparse_load)
      return false;

   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   enum glsl_sampler_dim dim = glsl_get_sampler_dim(deref->type);
   if (dim != GLSL_SAMPLER_DIM_SUBPASS && dim != GLSL_SAMPLER_DIM_SUBPASS_MS)
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   uint32_t index_imm = 0, range = 1;
   nir_def *index_ssa = NULL;
   if (deref->deref_type == nir_deref_type_array) {
      nir_deref_instr *parent = nir_deref_instr_parent(deref);
      if (nir_src_is_const(deref->arr.index)) {
         index_imm = nir_src_as_uint(deref->arr.index);
      } else {
         index_ssa = deref->arr.index.ssa;
         range = glsl_array_size(parent->type);
      }
      deref = parent;
   }

   assert(deref->deref_type == nir_deref_type_var);
   nir_variable *var = deref->var;

   const unsigned base_idx =
      var->data.index != NIR_VARIABLE_NO_INDEX ? var->data.index + 1 : 0;
   index_imm += base_idx;
   index_ssa = index_ssa ? nir_iadd_imm(b, index_ssa, base_idx)
                         : nir_imm_int(b, index_imm);

   nir_alu_type dest_type = nir_intrinsic_dest_type(intr);

   /* Zero means a runtime-sized array. */
   range = range == 0 ? 9 - index_imm : range;
   ctx->input_attachment_read |= BITFIELD_RANGE(index_imm, range);

   nir_def *target = nir_load_input_attachment_target_pan(b, index_ssa);
   nir_def *load_img, *load_output;

   nir_push_if(b, nir_ine_imm(b, target, ~0));
   {
      nir_def *is_color = nir_ilt_imm(b, target, 8);
      nir_def *load_color, *load_zs;
      nir_io_semantics iosem = {0};
      iosem.fb_fetch_output = true;
      iosem.fb_fetch_output_coherent =
         !!(nir_intrinsic_access(intr) & ACCESS_COHERENT);

      nir_push_if(b, is_color);
      {
         nir_def *conversion = nir_load_input_attachment_conv_pan(b, index_ssa);
         iosem.location = FRAG_RESULT_DATA0;
         if (!ctx->ro_color_mask) {
            load_color = nir_load_tile_pan(
               b, intr->def.num_components, intr->def.bit_size,
               pan_nir_tile_rt_sample(b, target, intr->src[2].ssa),
               pan_nir_tile_default_coverage(b), conversion,
               .dest_type = dest_type, .access = nir_intrinsic_access(intr),
               .io_semantics = iosem);
         } else {
            nir_def *is_read_only = nir_i2b(
               b, nir_iand_imm(b, nir_ishl(b, nir_imm_int(b, 1), target), ctx->ro_color_mask));
            nir_def *load_ro, *load_rw;
            nir_push_if(b, is_read_only);
            {
               load_ro = nir_load_tile_res_pan(
                  b, intr->def.num_components, intr->def.bit_size,
                  pan_nir_tile_rt_sample(b, target, intr->src[2].ssa),
                  pan_nir_tile_default_coverage(b), conversion,
                  .dest_type = dest_type, .access = nir_intrinsic_access(intr),
                  .io_semantics = iosem);
            }
            nir_push_else(b, NULL);
            {
               load_rw = nir_load_tile_pan(
                  b, intr->def.num_components, intr->def.bit_size,
                  pan_nir_tile_rt_sample(b, target, intr->src[2].ssa),
                  pan_nir_tile_default_coverage(b), conversion,
                  .dest_type = dest_type, .access = nir_intrinsic_access(intr),
                  .io_semantics = iosem);
            }
            nir_pop_if(b, NULL);
            load_color = nir_if_phi(b, load_ro, load_rw);
         }
      }
      nir_push_else(b, NULL);
      {
         /* Depth or stencil: the conversion is zero on v9+. v7 needs the
          * format, as a zero conversion faults with INVALID_INSTR_ENC
          * (panvk_v7): depth's comes with the attachment map. Stencil's is
          * a constant, which the v7 compile puts in place of the zero
          * (mali_shader.c v7_stencil_tile_conv). */
         iosem.location = dest_type == nir_type_float32 ? FRAG_RESULT_DEPTH
                                                        : FRAG_RESULT_STENCIL;
         nir_def *v7_conv = NULL;
         if (ctx->arch < 9 && iosem.location == FRAG_RESULT_DEPTH)
            v7_conv = nir_load_input_attachment_conv_pan(b, index_ssa);
         load_zs = nir_load_tile_pan(
            b, intr->def.num_components, intr->def.bit_size,
            pan_nir_tile_location_sample(b, iosem.location, intr->src[2].ssa),
            pan_nir_tile_default_coverage(b), v7_conv ? v7_conv : nir_imm_int(b, 0),
            .dest_type = dest_type, .access = nir_intrinsic_access(intr),
            .io_semantics = iosem);

         /* A stencil load may have garbage in the upper 24 bits. */
         if (iosem.location == FRAG_RESULT_STENCIL)
            load_zs = nir_iand_imm(b, load_zs, BITFIELD_MASK(8));
      }
      nir_pop_if(b, NULL);

      load_output = nir_if_phi(b, load_color, load_zs);
   }
   nir_push_else(b, NULL);
   {
      nir_instr *clone = nir_instr_clone(b->shader, &intr->instr);
      nir_builder_instr_insert(b, clone);
      load_img = &nir_instr_as_intrinsic(clone)->def;
   }
   nir_pop_if(b, NULL);

   nir_def_replace(&intr->def, nir_if_phi(b, load_output, load_img));
   return true;
}

static bool
is_subpass_load(const nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic != nir_intrinsic_image_deref_load)
      return false;
   const nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   const enum glsl_sampler_dim dim = glsl_get_sampler_dim(deref->type);
   return dim == GLSL_SAMPLER_DIM_SUBPASS || dim == GLSL_SAMPLER_DIM_SUBPASS_MS;
}

/* Per component: the same scalar, equal constants, or both undefined. */
static bool
same_value(nir_def *a, nir_def *b)
{
   if (a == b)
      return true;
   if (a->num_components != b->num_components || a->bit_size != b->bit_size)
      return false;
   for (unsigned c = 0; c < a->num_components; c++) {
      nir_scalar sa = nir_scalar_resolved(a, c);
      nir_scalar sb = nir_scalar_resolved(b, c);
      if (nir_scalar_equal(sa, sb))
         continue;
      if (nir_scalar_is_undef(sa) && nir_scalar_is_undef(sb))
         continue;
      if (nir_scalar_is_const(sa) && nir_scalar_is_const(sb) &&
          nir_scalar_as_uint(sa) == nir_scalar_as_uint(sb))
         continue;
      return false;
   }
   return true;
}

static bool
same_subpass_load(const nir_intrinsic_instr *a, const nir_intrinsic_instr *b)
{
   if (a->def.num_components != b->def.num_components ||
       a->def.bit_size != b->def.bit_size ||
       nir_intrinsic_dest_type(a) != nir_intrinsic_dest_type(b))
      return false;
   /* Same attachment: the same variable and constant array index. */
   const nir_deref_instr *da = nir_src_as_deref(a->src[0]);
   const nir_deref_instr *db = nir_src_as_deref(b->src[0]);
   if (da->deref_type != db->deref_type)
      return false;
   if (da->deref_type == nir_deref_type_array) {
      if (!nir_src_is_const(da->arr.index) || !nir_src_is_const(db->arr.index) ||
          nir_src_as_uint(da->arr.index) != nir_src_as_uint(db->arr.index))
         return false;
      da = nir_deref_instr_parent(da);
      db = nir_deref_instr_parent(db);
   }
   if (da->deref_type != nir_deref_type_var || db->deref_type != nir_deref_type_var ||
       da->var != db->var)
      return false;
   /* Same coordinate and sample (vtn gives subpass loads constant
    * coordinates; the sample is gl_SampleID or 0). */
   for (unsigned s = 1; s <= 2; s++) {
      if (!same_value(a->src[s].ssa, b->src[s].ssa))
         return false;
   }
   return true;
}

/*
 * A fragment shader invocation writes its attachments only through its
 * outputs, after it ends, so every subpassLoad() of one attachment at one
 * sample returns the same value (with rasterization-order access the
 * earlier fragments' writes are all visible before the first read). ARMSX2
 * reads its colour target once for blending and again inside the
 * alpha-test branch; the second read is another LD_TILE message and its
 * conversion. A load dominated by an identical one reuses its value.
 */
static bool
dedup_subpass_loads(nir_shader *nir)
{
   bool progress = false;

   nir_foreach_function_impl(impl, nir) {
      nir_metadata_require(impl, nir_metadata_dominance);
      struct util_dynarray seen;
      util_dynarray_init(&seen, NULL);
      bool impl_progress = false;

      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (!is_subpass_load(instr))
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            nir_intrinsic_instr *dom = NULL;
            util_dynarray_foreach(&seen, nir_intrinsic_instr *, prev) {
               if (same_subpass_load(*prev, intr) &&
                   nir_block_dominates((*prev)->instr.block, block)) {
                  dom = *prev;
                  break;
               }
            }
            if (dom) {
               nir_def_replace(&intr->def, &dom->def);
               impl_progress = true;
            } else {
               util_dynarray_append(&seen, intr);
            }
         }
      }
      util_dynarray_fini(&seen);
      progress |= nir_progress(impl_progress, impl, nir_metadata_control_flow);
   }
   return progress;
}

bool
mali_nir_lower_input_attachment_loads(nir_shader *nir, unsigned arch,
                                      const struct vk_graphics_pipeline_state *state,
                                      uint32_t *input_attachment_read)
{
   bool progress = false;
   /* The read-only tile load (LD_TILE with the resource wait) exists from
    * v10 on; v9 reads every colour attachment with the plain tile load. */
   struct ia_ctx ctx = {
      .arch = arch,
      .ro_color_mask = arch >= 10 ? readonly_color_mask(nir, state) : 0,
   };

   NIR_PASS(progress, nir, dedup_subpass_loads);

   NIR_PASS(progress, nir, nir_shader_intrinsics_pass, lower_load,
            nir_metadata_none, &ctx);

   if (input_attachment_read)
      *input_attachment_read = ctx.input_attachment_read;

   /* The image path of the loads above, and anything else. */
   const struct nir_input_attachment_options opts = {0};
   NIR_PASS(progress, nir, nir_lower_input_attachments, &opts);

   return progress;
}
