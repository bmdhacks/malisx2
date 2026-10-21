/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Descriptor accesses in NIR to v11 resource-table accesses, following
 * panvk's v9+ lowering (Mesa src/panfrost/vulkan/
 * panvk_vX_nir_lower_descriptors.c, MIT), because the descriptor layout is
 * panvk's (mali_descriptor_set_layout.h):
 *
 *  - set N is resource table N + 1; a binding element's slot comes from the
 *    set layout (mali_desc_index);
 *  - table 0 is the driver's: vertex attributes (vertex shaders, 16
 *    entries) or varying descriptors (fragment shaders, one per input
 *    slot), then a dummy sampler, then copies of the dynamic buffers the
 *    shader uses. The caller sets desc_info->driver_table_prefix to the
 *    first part's size before calling. panvk puts a fragment shader's dummy
 *    sampler at 0 when it compiles and after the varying descriptors when
 *    it draws; we reserve the varying slots at compile time instead, so the
 *    two always agree;
 *  - uniform and storage buffers use the vec2-index + 32-bit offset address
 *    format through the resource handle; images and textures get their
 *    resource handle as texture/sampler index or offset source;
 *  - descriptor reads (buffer sizes) load the 32-byte descriptor through
 *    the special resource table 62, which addresses resource tables as
 *    buffers.
 *
 * The Bifrost paths, YCbCr conversion and immutable-sampler folding of the
 * original are left out: v11 only, and no YCbCr.
 */

#include "mali_shader.h"

#include "nir_builder.h"
#include "panfrost/compiler/pan_nir.h"
#include "util/hash_table.h"

#include "vk_pipeline.h"

#define RESOURCE_TABLE_IDX 62

struct desc_map {
   /* Entry i: MALI_COPY_DESC_HANDLE(set, dynamic buffer index). */
   uint32_t *map;
   uint32_t count;
};

struct lower_desc_ctx {
   mesa_shader_stage stage;
   const struct mali_descriptor_set_layout *set_layouts[MALI_MAX_SETS];
   struct hash_table_u64 *ht;
   bool add_bounds_checks;
   uint32_t used_set_mask;
   uint32_t dummy_sampler_handle;
   uint32_t dyn_bufs_start;
   struct desc_map dyn_bufs;
   uint32_t driver_table_prefix;
};

struct desc_id {
   union {
      struct {
         uint32_t binding;
         uint32_t set : 4;
         uint32_t sampler_subdesc : 1;
         uint32_t pad : 27;
      };
      uint64_t ht_key;
   };
};

static const struct mali_descriptor_set_binding_layout *
binding_layout(const struct lower_desc_ctx *ctx, uint32_t set, uint32_t binding)
{
   assert(set < MALI_MAX_SETS && ctx->set_layouts[set]);
   assert(binding < ctx->set_layouts[set]->binding_count);
   return &ctx->set_layouts[set]->bindings[binding];
}

/* Resource handle of (set, binding, sub-descriptor), element 0. */
static uint32_t
shader_desc_idx(const struct lower_desc_ctx *ctx, uint32_t set, uint32_t binding,
                enum mali_subdesc sub)
{
   const struct mali_descriptor_set_binding_layout *bl =
      binding_layout(ctx, set, binding);

   /* Non-dynamic descriptors are read straight from their set's table. */
   if (!vk_descriptor_type_is_dynamic(bl->type))
      return pan_res_handle(set + 1, mali_desc_index(bl, 0, sub));

   /* Dynamic buffers: their copies in the driver table. */
   struct desc_id id = {
      .set = set,
      .sampler_subdesc = sub == MALI_SUBDESC_SAMPLER,
      .binding = binding,
   };
   uint32_t *entry = _mesa_hash_table_u64_search(ctx->ht, id.ht_key);
   assert(entry && entry >= ctx->dyn_bufs.map &&
          entry < ctx->dyn_bufs.map + ctx->dyn_bufs.count);
   return pan_res_handle(0, ctx->dyn_bufs_start + (entry - ctx->dyn_bufs.map));
}

/* Vulkan resource index: vec3(first handle, array index, array size - 1). */
static nir_def *
build_res_index(nir_builder *b, const struct lower_desc_ctx *ctx, uint32_t set,
                uint32_t binding, nir_def *array_index)
{
   const struct mali_descriptor_set_binding_layout *bl =
      binding_layout(ctx, set, binding);
   uint32_t desc_idx = shader_desc_idx(ctx, set, binding, MALI_SUBDESC_NONE);

   return nir_vec3(b, nir_imm_int(b, desc_idx), array_index,
                   nir_imm_int(b, bl->desc_count - 1));
}

static nir_def *
build_res_reindex(nir_builder *b, nir_def *orig, nir_def *delta)
{
   return nir_vec3(b, nir_channel(b, orig, 0),
                   nir_iadd(b, nir_channel(b, orig, 1), delta),
                   nir_channel(b, orig, 2));
}

static nir_def *
build_buffer_addr_for_res_index(nir_builder *b, nir_def *res_index,
                                const struct lower_desc_ctx *ctx)
{
   nir_def *first_desc_index = nir_channel(b, res_index, 0);
   nir_def *array_index = nir_channel(b, res_index, 1);
   nir_def *array_max = nir_channel(b, res_index, 2);

   if (ctx->add_bounds_checks)
      array_index = nir_umin(b, array_index, array_max);

   return nir_vec3(b, first_desc_index, array_index, nir_imm_int(b, 0));
}

static bool
lower_res_intrinsic(nir_builder *b, nir_intrinsic_instr *intrin,
                    const struct lower_desc_ctx *ctx)
{
   b->cursor = nir_before_instr(&intrin->instr);

   nir_def *res;
   switch (intrin->intrinsic) {
   case nir_intrinsic_vulkan_resource_index:
      res = build_res_index(b, ctx, nir_intrinsic_desc_set(intrin),
                            nir_intrinsic_binding(intrin), intrin->src[0].ssa);
      break;
   case nir_intrinsic_vulkan_resource_reindex:
      res = build_res_reindex(b, intrin->src[0].ssa, intrin->src[1].ssa);
      break;
   case nir_intrinsic_load_vulkan_descriptor:
      res = build_buffer_addr_for_res_index(b, intrin->src[0].ssa, ctx);
      break;
   default:
      UNREACHABLE("unhandled resource intrinsic");
   }

   assert(intrin->def.bit_size == res->bit_size);
   assert(intrin->def.num_components == res->num_components);
   nir_def_replace(&intrin->def, res);
   return true;
}

static void
get_resource_deref_binding(nir_deref_instr *deref, uint32_t *set,
                           uint32_t *binding, uint32_t *index_imm,
                           nir_def **index_ssa, uint32_t *max_idx)
{
   *index_imm = 0;
   *max_idx = 0;
   *index_ssa = NULL;

   if (deref->deref_type == nir_deref_type_array) {
      if (nir_src_is_const(deref->arr.index)) {
         *index_imm = nir_src_as_uint(deref->arr.index);
         *max_idx = *index_imm;
      } else {
         *index_ssa = deref->arr.index.ssa;
         /* Zero means a runtime-sized array; minus one gives UINT32_MAX. */
         *max_idx = ((uint32_t)glsl_array_size(nir_deref_instr_parent(deref)->type)) - 1;
      }
      deref = nir_deref_instr_parent(deref);
   }

   assert(deref->deref_type == nir_deref_type_var);
   *set = deref->var->data.descriptor_set;
   *binding = deref->var->data.binding;
}

static bool
lower_tex(nir_builder *b, nir_tex_instr *tex, const struct lower_desc_ctx *ctx)
{
   bool progress = false;

   b->cursor = nir_before_instr(&tex->instr);

   /* Valhall takes complete handles in nir_tex_src_texture_handle and
    * nir_tex_src_sampler_handle and nothing in texture_index and
    * sampler_index (pan_nir_lower_tex.c). */
   int sampler_src_idx = nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref);
   if (sampler_src_idx >= 0) {
      nir_def *plane = nir_steal_tex_src(tex, nir_tex_src_plane);
      assert(!plane || nir_src_as_uint(nir_src_for_ssa(plane)) == 0);
      (void)plane;

      nir_deref_instr *deref = nir_src_as_deref(tex->src[sampler_src_idx].src);
      nir_tex_instr_remove_src(tex, sampler_src_idx);

      uint32_t set, binding, index_imm, max_idx;
      nir_def *index_ssa;
      get_resource_deref_binding(deref, &set, &binding, &index_imm, &index_ssa,
                                 &max_idx);

      const struct mali_descriptor_set_binding_layout *bl =
         binding_layout(ctx, set, binding);
      uint32_t stride = mali_desc_stride(bl);
      uint32_t sampler_index =
         shader_desc_idx(ctx, set, binding, MALI_SUBDESC_SAMPLER) + index_imm * stride;

      nir_def *handle = nir_imm_int(b, sampler_index);
      if (index_ssa)
         handle = nir_iadd(b, handle, nir_imul_imm(b, index_ssa, stride));
      nir_tex_instr_add_src(tex, nir_tex_src_sampler_handle, handle);
      progress = true;
   } else {
      /* Texel fetches still name a sampler on Valhall. */
      nir_tex_instr_add_src(tex, nir_tex_src_sampler_handle,
                            nir_imm_int(b, ctx->dummy_sampler_handle));
      progress = true;
   }

   int tex_src_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   if (tex_src_idx >= 0) {
      nir_deref_instr *deref = nir_src_as_deref(tex->src[tex_src_idx].src);
      nir_tex_instr_remove_src(tex, tex_src_idx);

      uint32_t set, binding, index_imm, max_idx;
      nir_def *index_ssa;
      get_resource_deref_binding(deref, &set, &binding, &index_imm, &index_ssa,
                                 &max_idx);

      const struct mali_descriptor_set_binding_layout *bl =
         binding_layout(ctx, set, binding);
      uint32_t stride = mali_desc_stride(bl);
      uint32_t texture_index =
         shader_desc_idx(ctx, set, binding, MALI_SUBDESC_TEXTURE) + index_imm * stride;

      nir_def *handle = nir_imm_int(b, texture_index);
      if (index_ssa)
         handle = nir_iadd(b, handle, nir_imul_imm(b, index_ssa, stride));
      nir_tex_instr_add_src(tex, nir_tex_src_texture_handle, handle);
      progress = true;
   }

   return progress;
}

static nir_def *
get_img_index(nir_builder *b, nir_deref_instr *deref,
              const struct lower_desc_ctx *ctx)
{
   uint32_t set, binding, index_imm, max_idx;
   nir_def *index_ssa;
   get_resource_deref_binding(deref, &set, &binding, &index_imm, &index_ssa,
                              &max_idx);

   ASSERTED const struct mali_descriptor_set_binding_layout *bl =
      binding_layout(ctx, set, binding);
   assert(bl->type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ||
          bl->type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER ||
          bl->type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER ||
          bl->type == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT);

   unsigned img_offset = shader_desc_idx(ctx, set, binding, MALI_SUBDESC_NONE);

   if (!index_ssa)
      return nir_imm_int(b, img_offset + index_imm);
   assert(index_imm == 0);
   return nir_iadd_imm(b, index_ssa, img_offset);
}

static bool
lower_img_intrinsic(nir_builder *b, nir_intrinsic_instr *intr,
                    const struct lower_desc_ctx *ctx)
{
   b->cursor = nir_before_instr(&intr->instr);
   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   nir_rewrite_image_intrinsic(intr, get_img_index(b, deref, ctx),
                               nir_image_intrinsic_type_default);
   return true;
}

/* Constant data lives in the binary's inline constant pool, addressed from
 * the program counter. */
static bool
lower_load_constant(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_constant)
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   nir_def *offset = nir_iadd_imm(b, intr->src[0].ssa, nir_intrinsic_base(intr));
   nir_def *addr = nir_iadd(b, nir_load_constant_base_ptr(b, 1, 64),
                            nir_u2u64(b, offset));
   nir_def *val = nir_load_global_constant(
      b, intr->def.num_components, intr->def.bit_size, addr,
      .access = ACCESS_CAN_REORDER | ACCESS_NON_WRITEABLE,
      .align_mul = nir_intrinsic_align_mul(intr),
      .align_offset = nir_intrinsic_align_offset(intr));

   nir_def_replace(&intr->def, val);
   return true;
}

static bool
lower_descriptors_instr(nir_builder *b, nir_instr *instr, void *data)
{
   struct lower_desc_ctx *ctx = data;

   if (instr->type == nir_instr_type_tex)
      return lower_tex(b, nir_instr_as_tex(instr), ctx);

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_vulkan_resource_index:
   case nir_intrinsic_vulkan_resource_reindex:
   case nir_intrinsic_load_vulkan_descriptor:
      return lower_res_intrinsic(b, intr, ctx);
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_image_deref_samples:
      return lower_img_intrinsic(b, intr, ctx);
   default:
      return false;
   }
}

/* First pass: which sets the shader uses, and how many elements of each
 * dynamic buffer binding it can reach. */
static void
record_binding(struct lower_desc_ctx *ctx, uint32_t set, uint32_t binding,
               enum mali_subdesc sub, uint32_t max_idx)
{
   const struct mali_descriptor_set_binding_layout *bl =
      binding_layout(ctx, set, binding);

   ctx->used_set_mask |= BITFIELD_BIT(set);

   if (!vk_descriptor_type_is_dynamic(bl->type))
      return;

   struct desc_id id = {
      .set = set,
      .sampler_subdesc = sub == MALI_SUBDESC_SAMPLER,
      .binding = binding,
   };
   uint32_t old_count =
      (uintptr_t)_mesa_hash_table_u64_search(ctx->ht, id.ht_key);
   uint32_t new_count = max_idx == UINT32_MAX ? bl->desc_count : max_idx + 1;
   assert(new_count <= bl->desc_count);

   if (old_count >= new_count)
      return;

   _mesa_hash_table_u64_insert(ctx->ht, id.ht_key, (void *)(uintptr_t)new_count);
   ctx->dyn_bufs.count += new_count - old_count;
}

static bool
collect_desc_access(nir_builder *b, nir_instr *instr, void *data)
{
   struct lower_desc_ctx *ctx = data;
   uint32_t set, binding, index_imm, max_idx;
   nir_def *index_ssa;

   if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      bool recorded = false;

      int idx = nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref);
      if (idx >= 0) {
         get_resource_deref_binding(nir_src_as_deref(tex->src[idx].src), &set,
                                    &binding, &index_imm, &index_ssa, &max_idx);
         record_binding(ctx, set, binding, MALI_SUBDESC_SAMPLER, max_idx);
         recorded = true;
      }
      idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
      if (idx >= 0) {
         get_resource_deref_binding(nir_src_as_deref(tex->src[idx].src), &set,
                                    &binding, &index_imm, &index_ssa, &max_idx);
         record_binding(ctx, set, binding, MALI_SUBDESC_TEXTURE, max_idx);
         recorded = true;
      }
      return recorded;
   }

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_vulkan_resource_index:
      record_binding(ctx, nir_intrinsic_desc_set(intr), nir_intrinsic_binding(intr),
                     MALI_SUBDESC_NONE, UINT32_MAX);
      return true;
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_image_deref_samples:
      get_resource_deref_binding(nir_src_as_deref(intr->src[0]), &set, &binding,
                                 &index_imm, &index_ssa, &max_idx);
      record_binding(ctx, set, binding, MALI_SUBDESC_NONE, max_idx);
      return true;
   default:
      return false;
   }
}

/* Lay out the driver table (mali_shader.h, struct mali_shader_desc_info):
 * the dummy sampler after the attributes or varyings, the dynamic buffer
 * copies after it. */
static void
create_copy_table(nir_shader *nir, struct lower_desc_ctx *ctx)
{
   uint32_t dummy_sampler_idx = ctx->driver_table_prefix;
   ctx->dummy_sampler_handle = pan_res_handle(0, dummy_sampler_idx);

   uint32_t copy_count = ctx->dyn_bufs.count;
   if (copy_count == 0)
      return;

   ctx->dyn_bufs_start = dummy_sampler_idx + 1;
   ctx->dyn_bufs.map = rzalloc_array(ctx->ht, uint32_t, copy_count);
   ctx->dyn_bufs.count = 0;

   hash_table_u64_foreach(ctx->ht, he) {
      uint32_t count = (uintptr_t)he.data;
      struct desc_id id = { .ht_key = he.key };
      const struct mali_descriptor_set_binding_layout *bl =
         binding_layout(ctx, id.set, id.binding);

      uint32_t *first = &ctx->dyn_bufs.map[ctx->dyn_bufs.count];
      for (uint32_t i = 0; i < count; i++)
         ctx->dyn_bufs.map[ctx->dyn_bufs.count++] =
            MALI_COPY_DESC_HANDLE(id.set, bl->desc_idx + i);

      /* From here on the table maps (set, binding) to its first copy. */
      _mesa_hash_table_u64_replace(ctx->ht, &he, first);
   }
}

void
mali_nir_lower_descriptors(nir_shader *nir,
                           const struct vk_pipeline_robustness_state *rs,
                           const struct vk_pipeline_layout *layout,
                           struct mali_shader_desc_info *desc_info)
{
   struct lower_desc_ctx ctx = {
      .stage = nir->info.stage,
      .add_bounds_checks =
         rs->storage_buffers != VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT ||
         rs->uniform_buffers != VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT ||
         rs->images != VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
      .driver_table_prefix = desc_info->driver_table_prefix,
   };
   bool progress = false;

   ctx.ht = _mesa_hash_table_u64_create(NULL);

   for (uint32_t i = 0; layout && i < layout->set_count && i < MALI_MAX_SETS; i++) {
      if (layout->set_layouts[i])
         ctx.set_layouts[i] = mali_descriptor_set_layout(layout->set_layouts[i]);
   }

   NIR_PASS(progress, nir, nir_shader_instructions_pass, collect_desc_access,
            nir_metadata_all, &ctx);
   if (progress) {
      create_copy_table(nir, &ctx);

      assert(ctx.dyn_bufs.count <= ARRAY_SIZE(desc_info->dyn_bufs.map));
      desc_info->dyn_bufs.count = ctx.dyn_bufs.count;
      memcpy(desc_info->dyn_bufs.map, ctx.dyn_bufs.map,
             ctx.dyn_bufs.count * sizeof(*desc_info->dyn_bufs.map));
      desc_info->used_set_mask = ctx.used_set_mask;

      NIR_PASS(_, nir, nir_shader_instructions_pass, lower_descriptors_instr,
               nir_metadata_control_flow, &ctx);
   }

   _mesa_hash_table_u64_destroy(ctx.ht);

   if (nir->constant_data_size)
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_load_constant,
               nir_metadata_control_flow, NULL);
}
