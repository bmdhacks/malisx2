/*
 * SPDX-FileCopyrightText: 2026 TheGammaSqueeze
 * SPDX-License-Identifier: MIT
 */

/*
 * Descriptor accesses in NIR for the Bifrost (v7) build, following panvk's
 * PAN_ARCH < 9 lowering (Mesa src/panfrost/vulkan/
 * panvk_vX_nir_lower_descriptors.c, MIT), on this driver's set layout
 * (mali_descriptor_set_layout.h: one 32-byte slot per descriptor, a
 * combined image/sampler's texture first and sampler second, dynamic
 * buffers without a slot):
 *
 *  - Bifrost has no resource tables. A stage gets per-type tables built at
 *    draw or dispatch time (mali_shader.h enum mali_v7_desc_table): the
 *    Uniform Buffer descriptors the shader's load_ubo indices name, the
 *    Texture and Sampler descriptors its texture and sampler indices name,
 *    and image Attribute descriptors. This pass hands out those indices,
 *    binding by binding in hash table order, and records for each entry
 *    which set slot it is copied from (mali_v7_shader_desc::map);
 *  - uniform buffers use the 32-bit index + offset address format (the
 *    index is the UBO table position); dynamic uniform buffers sit behind
 *    the static ones in the table, in the order of desc.dyn_bufs;
 *  - storage buffers are not copied: a load_vulkan_descriptor reads the
 *    buffer's address and size pair straight out of the set (64-bit
 *    global + 32-bit offset format, or the bounded one with robustness),
 *    through the set's GPU address in the sysvals (mali_graphics_sysvals
 *    v7_desc.sets). Dynamic storage buffers come from a per-stage table
 *    the draw or dispatch writes with the offsets applied, at index
 *    MALI_V7_SETS_DYN_SSBOS of the same sysval array;
 *  - texture and image size, level and sample queries read the
 *    descriptor's words from the set the same way (Mesa's compiler
 *    lowers these queries on v9+ only);
 *  - constant data lives in a buffer the shader upload fills, addressed
 *    through the common.constant_data sysval (Bifrost has no PC-relative
 *    constant pool).
 *
 * Left out of panvk's: YCbCr conversion and multi-plane descriptors, null
 * descriptors.
 */

#include "mali_shader.h"

#include <stdlib.h>

#include "nir_builder.h"
#include "util/hash_table.h"

#include "vk_pipeline.h"

struct desc_map {
   /* Entry i: MALI_COPY_DESC_HANDLE(set, slot or dynamic buffer index). */
   uint32_t *map;
   uint32_t count;
};

struct lower_desc_ctx {
   mesa_shader_stage stage;
   const struct mali_descriptor_set_layout *set_layouts[MALI_MAX_SETS];
   struct hash_table_u64 *ht;
   bool add_bounds_checks;
   nir_address_format ssbo_addr_format;
   uint32_t used_set_mask;
   bool needs_set_addrs;
   struct desc_map dyn_ubos;
   struct desc_map dyn_ssbos;
   struct desc_map others[MALI_V7_DESC_TABLE_COUNT];
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

/* Which per-type table a binding's descriptors are copied into; -1 for
 * the types that are not copied (storage buffers, dynamic buffers). */
static int
table_of(const struct mali_descriptor_set_binding_layout *bl, bool sampler_subdesc)
{
   switch (bl->type) {
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      return sampler_subdesc ? MALI_V7_DESC_TABLE_SAMPLER : MALI_V7_DESC_TABLE_TEXTURE;
   case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
   case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
      return MALI_V7_DESC_TABLE_TEXTURE;
   case VK_DESCRIPTOR_TYPE_SAMPLER:
      return MALI_V7_DESC_TABLE_SAMPLER;
   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
   case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
   case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      return MALI_V7_DESC_TABLE_IMG;
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
   case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK:
      return MALI_V7_DESC_TABLE_UBO;
   default:
      return -1;
   }
}

static struct desc_map *
map_of(struct lower_desc_ctx *ctx, const struct mali_descriptor_set_binding_layout *bl,
       bool sampler_subdesc)
{
   if (bl->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC)
      return &ctx->dyn_ubos;
   if (bl->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
      return &ctx->dyn_ssbos;
   const int t = table_of(bl, sampler_subdesc);
   assert(t >= 0);
   return &ctx->others[t];
}

/*
 * The index the shader uses for (set, binding, sub-descriptor), element
 * 0: a storage buffer's slot in its set, a dynamic storage buffer's entry
 * in the stage's dynamic table, anything else its position in its
 * per-type table.
 */
static uint32_t
shader_desc_idx(struct lower_desc_ctx *ctx, uint32_t set, uint32_t binding,
                enum mali_subdesc sub)
{
   const struct mali_descriptor_set_binding_layout *bl = binding_layout(ctx, set, binding);

   if (bl->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
      return bl->desc_idx;

   const bool sampler_subdesc = sub == MALI_SUBDESC_SAMPLER;
   struct desc_id id = {
      .set = set,
      .sampler_subdesc = sampler_subdesc,
      .binding = binding,
   };
   const uint32_t *entry = _mesa_hash_table_u64_search(ctx->ht, id.ht_key);
   const struct desc_map *map = map_of(ctx, bl, sampler_subdesc);
   assert(entry && entry >= map->map && entry < map->map + map->count);

   uint32_t idx = entry - map->map;
   if (bl->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC) {
      /* Dynamic uniform buffers follow the static ones in the UBO table. */
      idx += ctx->others[MALI_V7_DESC_TABLE_UBO].count;
   } else if (map == &ctx->others[MALI_V7_DESC_TABLE_IMG]) {
      /* Images follow the vertex attributes in the attribute table. */
      idx += ctx->stage == MESA_SHADER_VERTEX ? MALI_MAX_VS_ATTRIBS : 0;
   }
   return idx;
}

/* ---------------------------------------------------------------------- */
/* Sysval loads                                                            */

/* The GPU address of descriptor set `table` (or the stage's dynamic
 * storage buffer table), from the sysvals' v7_desc.sets. A non-constant
 * index reads the FAU block's copy in memory. */
static nir_def *
load_set_addr(nir_builder *b, struct lower_desc_ctx *ctx, nir_def *table)
{
   const unsigned off = ctx->stage == MESA_SHADER_COMPUTE
                           ? offsetof(struct mali_compute_sysvals, v7_desc.sets)
                           : offsetof(struct mali_graphics_sysvals, v7_desc.sets);
   ctx->needs_set_addrs = true;
   return nir_load_push_constant(
      b, 1, 64, nir_imul_imm(b, table, sizeof(uint64_t)),
      .base = MALI_SYSVALS_PUSH_CONST_BASE + off,
      .range = sizeof(((struct mali_graphics_sysvals *)NULL)->v7_desc.sets));
}

static nir_def *
load_constant_data_addr(nir_builder *b)
{
   return nir_load_push_constant(
      b, 1, 64, nir_imm_int(b, 0),
      .base = MALI_SYSVALS_PUSH_CONST_BASE +
              offsetof(struct mali_graphics_sysvals, common.constant_data));
}

/* ---------------------------------------------------------------------- */
/* Buffers                                                                 */

static uint32_t
ssbo_table(const struct lower_desc_ctx *ctx, uint32_t set, uint32_t binding)
{
   const struct mali_descriptor_set_binding_layout *bl = binding_layout(ctx, set, binding);
   assert(bl->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
          bl->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC);
   return bl->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC ? MALI_V7_SETS_DYN_SSBOS : set;
}

static bool
is_ubo_type(VkDescriptorType type)
{
   return type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
          type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
          type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK;
}

/* Uniform buffers: vec2(array size - 1 << 16 | first index, array index).
 * Storage buffers: vec4(table, first slot, array index, array size - 1). */
static nir_def *
build_res_index(nir_builder *b, struct lower_desc_ctx *ctx, uint32_t set, uint32_t binding,
                nir_def *array_index)
{
   const struct mali_descriptor_set_binding_layout *bl = binding_layout(ctx, set, binding);
   const uint32_t array_size =
      bl->type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK ? 1 : bl->desc_count;
   const uint32_t desc_idx = shader_desc_idx(ctx, set, binding, MALI_SUBDESC_NONE);

   if (is_ubo_type(bl->type)) {
      assert(desc_idx < BITFIELD_BIT(16) && array_size - 1 < BITFIELD_BIT(16));
      return nir_vec2(b, nir_imm_int(b, (array_size - 1) << 16 | desc_idx), array_index);
   }
   return nir_vec4(b, nir_imm_int(b, ssbo_table(ctx, set, binding)), nir_imm_int(b, desc_idx),
                   array_index, nir_imm_int(b, array_size - 1));
}

/* The array index moves by delta and the bound stays the binding's, as in
 * panvk's v9+ reindex (its PAN_ARCH < 9 one puts a constant 3 in the
 * bound). */
static nir_def *
build_res_reindex(nir_builder *b, nir_def *orig, nir_def *delta, nir_address_format fmt)
{
   if (fmt == nir_address_format_32bit_index_offset) {
      return nir_vec2(b, nir_channel(b, orig, 0), nir_iadd(b, nir_channel(b, orig, 1), delta));
   }
   return nir_vec4(b, nir_channel(b, orig, 0), nir_channel(b, orig, 1),
                   nir_iadd(b, nir_channel(b, orig, 2), delta), nir_channel(b, orig, 3));
}

static nir_def *
build_buffer_addr_for_res_index(nir_builder *b, struct lower_desc_ctx *ctx, nir_def *res_index,
                                nir_address_format fmt)
{
   if (fmt == nir_address_format_32bit_index_offset) {
      nir_def *packed = nir_channel(b, res_index, 0);
      nir_def *array_index = nir_channel(b, res_index, 1);
      nir_def *first = nir_extract_u16(b, packed, nir_imm_int(b, 0));
      nir_def *array_max = nir_extract_u16(b, packed, nir_imm_int(b, 1));
      if (ctx->add_bounds_checks)
         array_index = nir_umin(b, array_index, array_max);
      return nir_vec2(b, nir_iadd(b, first, array_index), nir_imm_int(b, 0));
   }

   nir_def *table = nir_channel(b, res_index, 0);
   nir_def *first = nir_channel(b, res_index, 1);
   nir_def *array_index = nir_channel(b, res_index, 2);
   nir_def *array_max = nir_channel(b, res_index, 3);
   if (ctx->add_bounds_checks)
      array_index = nir_umin(b, array_index, array_max);

   /* The address pair in its 32-byte slot: base lo, base hi, size, zero. */
   nir_def *offset = nir_imul_imm(b, nir_iadd(b, array_index, first), MALI_DESCRIPTOR_SIZE);
   nir_def *addr = nir_iadd(b, load_set_addr(b, ctx, table), nir_u2u64(b, offset));
   nir_def *desc = nir_load_global(b, 4, 32, addr, .align_mul = MALI_DESCRIPTOR_SIZE);
   return nir_vec4(b, nir_channel(b, desc, 0), nir_channel(b, desc, 1), nir_channel(b, desc, 2),
                   nir_imm_int(b, 0));
}

static nir_address_format
addr_format_for(const struct lower_desc_ctx *ctx, nir_descriptor_type type)
{
   return type == nir_descriptor_type_uniform_buffer ? nir_address_format_32bit_index_offset
                                                     : ctx->ssbo_addr_format;
}

static bool
lower_res_intrinsic(nir_builder *b, nir_intrinsic_instr *intrin, struct lower_desc_ctx *ctx)
{
   b->cursor = nir_before_instr(&intrin->instr);
   const nir_address_format fmt = addr_format_for(ctx, nir_intrinsic_desc_type(intrin));

   nir_def *res;
   switch (intrin->intrinsic) {
   case nir_intrinsic_vulkan_resource_index:
      res = build_res_index(b, ctx, nir_intrinsic_desc_set(intrin),
                            nir_intrinsic_binding(intrin), intrin->src[0].ssa);
      break;
   case nir_intrinsic_vulkan_resource_reindex:
      res = build_res_reindex(b, intrin->src[0].ssa, intrin->src[1].ssa, fmt);
      break;
   case nir_intrinsic_load_vulkan_descriptor:
      res = build_buffer_addr_for_res_index(b, ctx, intrin->src[0].ssa, fmt);
      break;
   default:
      UNREACHABLE("unhandled resource intrinsic");
   }

   assert(intrin->def.bit_size == res->bit_size);
   assert(intrin->def.num_components == res->num_components);
   nir_def_replace(&intrin->def, res);
   return true;
}

/* ---------------------------------------------------------------------- */
/* Textures and images                                                     */

static void
get_resource_deref_binding(nir_deref_instr *deref, uint32_t *set, uint32_t *binding,
                           uint32_t *index_imm, nir_def **index_ssa, uint32_t *max_idx)
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

/*
 * The words from byte desc_offset of the set slot a deref names (the
 * Texture descriptor of a sampled image, the attribute pair of a storage
 * image or texel buffer), read from the set through its sysval address.
 */
static nir_def *
load_resource_deref_desc(nir_builder *b, struct lower_desc_ctx *ctx, nir_deref_instr *deref,
                         enum mali_subdesc sub, unsigned desc_offset, unsigned num_components,
                         unsigned bit_size)
{
   uint32_t set, binding, index_imm, max_idx;
   nir_def *index_ssa;
   get_resource_deref_binding(deref, &set, &binding, &index_imm, &index_ssa, &max_idx);

   const struct mali_descriptor_set_binding_layout *bl = binding_layout(ctx, set, binding);
   assert(index_ssa == NULL || index_imm == 0);
   if (!index_ssa)
      index_ssa = nir_imm_int(b, index_imm);

   nir_def *slot = nir_iadd_imm(b, nir_imul_imm(b, index_ssa, mali_desc_stride(bl)),
                                mali_desc_index(bl, 0, sub));
   nir_def *offset = nir_iadd_imm(b, nir_imul_imm(b, slot, MALI_DESCRIPTOR_SIZE), desc_offset);
   nir_def *addr = nir_iadd(b, load_set_addr(b, ctx, nir_imm_int(b, set)), nir_u2u64(b, offset));
   const unsigned align = 1u << (ffs(MALI_DESCRIPTOR_SIZE + desc_offset) - 1);
   return nir_load_global(b, num_components, bit_size, addr, .align_mul = align);
}

/* Texture descriptor fields (v7.xml): width and height at byte 4, array
 * size at byte 24 and depth at byte 28, all 16-bit minus one; levels
 * minus one in word 2 bits 16..20; log2 of the sample count in word 3
 * bits 13..15. */
static nir_def *
load_tex_size(nir_builder *b, struct lower_desc_ctx *ctx, nir_deref_instr *deref,
              enum glsl_sampler_dim dim, bool is_array)
{
   if (dim == GLSL_SAMPLER_DIM_BUF) {
      /* A texel buffer is an attribute pair: stride at byte 8, size at 12
       * of the Attribute Buffer. Both are unsigned: a view of 2 GiB or
       * more is within maxTexelBufferElements. */
      nir_def *stride_size =
         load_resource_deref_desc(b, ctx, deref, MALI_SUBDESC_NONE, 8, 2, 32);
      return nir_udiv(b, nir_channel(b, stride_size, 1), nir_channel(b, stride_size, 0));
   }

   nir_def *w_h = load_resource_deref_desc(b, ctx, deref, MALI_SUBDESC_TEXTURE, 4, 2, 16);
   nir_def *depth_or_layers = load_resource_deref_desc(
      b, ctx, deref, MALI_SUBDESC_TEXTURE, dim == GLSL_SAMPLER_DIM_3D ? 28 : 24, 1, 16);
   nir_def *sz = is_array && dim == GLSL_SAMPLER_DIM_1D
                    ? nir_vec2(b, nir_channel(b, w_h, 0), depth_or_layers)
                    : nir_vec3(b, nir_channel(b, w_h, 0), nir_channel(b, w_h, 1),
                               depth_or_layers);
   sz = nir_pad_vector_imm_int(b, sz, 0, 4);
   return nir_iadd_imm(b, nir_u2u32(b, sz), 1);
}

static nir_def *
load_tex_levels(nir_builder *b, struct lower_desc_ctx *ctx, nir_deref_instr *deref)
{
   nir_def *word2 = load_resource_deref_desc(b, ctx, deref, MALI_SUBDESC_TEXTURE, 8, 1, 32);
   return nir_iadd_imm(b, nir_iand_imm(b, nir_ushr_imm(b, word2, 16), 0x1f), 1);
}

static nir_def *
load_tex_samples(nir_builder *b, struct lower_desc_ctx *ctx, nir_deref_instr *deref)
{
   nir_def *word3 = load_resource_deref_desc(b, ctx, deref, MALI_SUBDESC_TEXTURE, 12, 1, 32);
   return nir_ishl(b, nir_imm_int(b, 1), nir_iand_imm(b, nir_ushr_imm(b, word3, 13), 0x7));
}

/* An image's attribute pair (panvk_v7 prepare_attr_buf_descs): the
 * Attribute Buffer's stride word at byte 8 packs the texel size, log2 of
 * the sample count in bits 7..9 and the format; the 3D continuation at
 * byte 16 holds the dimensions minus one as 16-bit values from byte 18. */
static nir_def *
load_img_size(nir_builder *b, struct lower_desc_ctx *ctx, nir_deref_instr *deref,
              enum glsl_sampler_dim dim, bool is_array)
{
   if (dim == GLSL_SAMPLER_DIM_BUF) {
      /* As a texel buffer's size in load_tex_size. */
      nir_def *stride_size =
         load_resource_deref_desc(b, ctx, deref, MALI_SUBDESC_NONE, 8, 2, 32);
      return nir_udiv(b, nir_channel(b, stride_size, 1), nir_channel(b, stride_size, 0));
   }

   nir_def *sz = load_resource_deref_desc(b, ctx, deref, MALI_SUBDESC_NONE, 18, 3, 16);
   if (is_array && dim == GLSL_SAMPLER_DIM_CUBE)
      sz = nir_vector_insert_imm(b, sz, nir_udiv_imm(b, nir_channel(b, sz, 2), 6), 2);
   if (is_array && dim == GLSL_SAMPLER_DIM_1D)
      sz = nir_vec2(b, nir_channel(b, sz, 0), nir_channel(b, sz, 2));
   sz = nir_pad_vector_imm_int(b, sz, 0, 4);
   sz = nir_iadd_imm(b, nir_u2u32(b, sz), 1);

   if (is_array && dim == GLSL_SAMPLER_DIM_MS) {
      nir_def *samples = load_resource_deref_desc(b, ctx, deref, MALI_SUBDESC_NONE, 8, 1, 32);
      samples = nir_iand_imm(b, nir_ushr_imm(b, samples, 7), BITFIELD_MASK(3));
      /* The layer count was multiplied by the sample count to make the
       * multisampled array look like a 3D image. */
      sz = nir_vector_insert_imm(b, sz, nir_ushr(b, nir_channel(b, sz, 2), samples), 2);
   }
   return sz;
}

static nir_def *
load_img_samples(nir_builder *b, struct lower_desc_ctx *ctx, nir_deref_instr *deref)
{
   nir_def *samples = load_resource_deref_desc(b, ctx, deref, MALI_SUBDESC_NONE, 8, 1, 32);
   samples = nir_iand_imm(b, nir_ushr_imm(b, samples, 7), BITFIELD_MASK(3));
   return nir_ishl(b, nir_imm_int(b, 1), samples);
}

static bool
lower_tex(nir_builder *b, nir_tex_instr *tex, struct lower_desc_ctx *ctx)
{
   bool progress = false;

   b->cursor = nir_before_instr(&tex->instr);

   if (tex->op == nir_texop_txs || tex->op == nir_texop_query_levels ||
       tex->op == nir_texop_texture_samples) {
      int tex_src_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
      assert(tex_src_idx >= 0);
      nir_deref_instr *deref = nir_src_as_deref(tex->src[tex_src_idx].src);

      nir_def *res;
      switch (tex->op) {
      case nir_texop_txs:
         res = nir_channels(b, load_tex_size(b, ctx, deref, tex->sampler_dim, tex->is_array),
                            nir_component_mask(tex->def.num_components));
         break;
      case nir_texop_query_levels:
         assert(tex->def.num_components == 1);
         res = load_tex_levels(b, ctx, deref);
         break;
      default:
         assert(tex->def.num_components == 1);
         res = load_tex_samples(b, ctx, deref);
         break;
      }
      nir_def_replace(&tex->def, res);
      return true;
   }

   int sampler_src_idx = nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref);
   if (sampler_src_idx >= 0) {
      nir_def *plane = nir_steal_tex_src(tex, nir_tex_src_plane);
      assert(!plane || nir_src_as_uint(nir_src_for_ssa(plane)) == 0);
      (void)plane;

      nir_deref_instr *deref = nir_src_as_deref(tex->src[sampler_src_idx].src);
      nir_tex_instr_remove_src(tex, sampler_src_idx);

      uint32_t set, binding, index_imm, max_idx;
      nir_def *index_ssa;
      get_resource_deref_binding(deref, &set, &binding, &index_imm, &index_ssa, &max_idx);

      /* One Sampler per element in the per-type table (no planes). */
      const uint32_t sampler_index =
         shader_desc_idx(ctx, set, binding, MALI_SUBDESC_SAMPLER) + index_imm;
      if (index_ssa) {
         nir_tex_instr_add_src(tex, nir_tex_src_sampler_offset,
                               nir_iadd_imm(b, index_ssa, sampler_index));
      } else {
         tex->sampler_index = sampler_index;
      }
      progress = true;
   }
   /* A texel fetch names no sampler: index 0, a dummy one the draw or
    * dispatch provides when the stage has textures but no samplers. */

   int tex_src_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   if (tex_src_idx >= 0) {
      nir_deref_instr *deref = nir_src_as_deref(tex->src[tex_src_idx].src);
      nir_tex_instr_remove_src(tex, tex_src_idx);

      uint32_t set, binding, index_imm, max_idx;
      nir_def *index_ssa;
      get_resource_deref_binding(deref, &set, &binding, &index_imm, &index_ssa, &max_idx);

      const uint32_t texture_index =
         shader_desc_idx(ctx, set, binding, MALI_SUBDESC_TEXTURE) + index_imm;
      if (index_ssa) {
         nir_tex_instr_add_src(tex, nir_tex_src_texture_offset,
                               nir_iadd_imm(b, index_ssa, texture_index));
      } else {
         tex->texture_index = texture_index;
      }
      progress = true;
   }

   return progress;
}

static nir_def *
get_img_index(nir_builder *b, nir_deref_instr *deref, struct lower_desc_ctx *ctx)
{
   uint32_t set, binding, index_imm, max_idx;
   nir_def *index_ssa;
   get_resource_deref_binding(deref, &set, &binding, &index_imm, &index_ssa, &max_idx);

   ASSERTED const struct mali_descriptor_set_binding_layout *bl =
      binding_layout(ctx, set, binding);
   assert(bl->type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ||
          bl->type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER ||
          bl->type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER);

   const unsigned img_offset = shader_desc_idx(ctx, set, binding, MALI_SUBDESC_NONE);
   if (!index_ssa)
      return nir_imm_int(b, img_offset + index_imm);
   assert(index_imm == 0);
   return nir_iadd_imm(b, index_ssa, img_offset);
}

static bool
lower_img_intrinsic(nir_builder *b, nir_intrinsic_instr *intr, struct lower_desc_ctx *ctx)
{
   b->cursor = nir_before_instr(&intr->instr);
   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);

   if (intr->intrinsic == nir_intrinsic_image_deref_size ||
       intr->intrinsic == nir_intrinsic_image_deref_samples) {
      const enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
      const bool is_array = nir_intrinsic_image_array(intr);
      nir_def *res;
      if (intr->intrinsic == nir_intrinsic_image_deref_size)
         res = nir_channels(b, load_img_size(b, ctx, deref, dim, is_array),
                            nir_component_mask(intr->def.num_components));
      else
         res = load_img_samples(b, ctx, deref);
      nir_def_replace(&intr->def, res);
      return true;
   }

   nir_rewrite_image_intrinsic(intr, get_img_index(b, deref, ctx),
                               nir_image_intrinsic_type_default);
   return true;
}

static bool
lower_load_constant(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_constant)
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   nir_def *offset = nir_iadd_imm(b, intr->src[0].ssa, nir_intrinsic_base(intr));
   nir_def *addr = nir_iadd(b, load_constant_data_addr(b), nir_u2u64(b, offset));
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

/* ---------------------------------------------------------------------- */
/* First pass: which bindings the shader reaches, and how many elements   */

static void
record_binding(struct lower_desc_ctx *ctx, uint32_t set, uint32_t binding, enum mali_subdesc sub,
               uint32_t max_idx)
{
   const struct mali_descriptor_set_binding_layout *bl = binding_layout(ctx, set, binding);

   ctx->used_set_mask |= BITFIELD_BIT(set);

   /* Storage buffers are read from the set directly. */
   if (bl->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
      return;

   const bool sampler_subdesc = sub == MALI_SUBDESC_SAMPLER;
   struct desc_id id = {
      .set = set,
      .sampler_subdesc = sampler_subdesc,
      .binding = binding,
   };
   const uint32_t old_count = (uintptr_t)_mesa_hash_table_u64_search(ctx->ht, id.ht_key);
   /* An inline uniform block is one Uniform Buffer descriptor (its first
    * slot); its other slots are the data. */
   const uint32_t elements =
      bl->type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK ? 1 : bl->desc_count;
   const uint32_t new_count = max_idx == UINT32_MAX ? elements : max_idx + 1;
   assert(new_count <= elements);
   if (old_count >= new_count)
      return;

   _mesa_hash_table_u64_insert(ctx->ht, id.ht_key, (void *)(uintptr_t)new_count);
   map_of(ctx, bl, sampler_subdesc)->count += new_count - old_count;
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
         get_resource_deref_binding(nir_src_as_deref(tex->src[idx].src), &set, &binding,
                                    &index_imm, &index_ssa, &max_idx);
         record_binding(ctx, set, binding, MALI_SUBDESC_SAMPLER, max_idx);
         recorded = true;
      }
      idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
      if (idx >= 0) {
         get_resource_deref_binding(nir_src_as_deref(tex->src[idx].src), &set, &binding,
                                    &index_imm, &index_ssa, &max_idx);
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
      get_resource_deref_binding(nir_src_as_deref(intr->src[0]), &set, &binding, &index_imm,
                                 &index_ssa, &max_idx);
      record_binding(ctx, set, binding, MALI_SUBDESC_NONE, max_idx);
      return true;
   default:
      return false;
   }
}

/* The copy map entries of one binding: count elements, one entry each. */
static uint32_t *
fill_copy_descs(struct lower_desc_ctx *ctx, uint32_t set, uint32_t binding, bool sampler_subdesc,
                uint32_t count)
{
   const struct mali_descriptor_set_binding_layout *bl = binding_layout(ctx, set, binding);
   struct desc_map *map = map_of(ctx, bl, sampler_subdesc);
   uint32_t *first = &map->map[map->count];

   assert(count && count <= bl->desc_count);
   assert(bl->type != VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK || count == 1);
   for (uint32_t i = 0; i < count; i++) {
      /* A dynamic buffer's index among the set's dynamic buffers, anything
       * else's slot. */
      const uint32_t src = vk_descriptor_type_is_dynamic(bl->type) ?
                              bl->desc_idx + i :
                              mali_desc_index(bl, i, sampler_subdesc ? MALI_SUBDESC_SAMPLER :
                                                                        MALI_SUBDESC_TEXTURE);
      map->map[map->count++] = MALI_COPY_DESC_HANDLE(set, src);
   }
   return first;
}

/* One array for every map, filled binding by binding in hash table order;
 * from here on the hash table maps each binding to its first entry.
 * *table_out is the array (the caller owns it), NULL for nothing; false
 * when it could not be allocated. */
static bool
create_copy_table(struct lower_desc_ctx *ctx, uint32_t **table_out)
{
   uint32_t total = ctx->dyn_ubos.count + ctx->dyn_ssbos.count;
   for (unsigned i = 0; i < MALI_V7_DESC_TABLE_COUNT; i++)
      total += ctx->others[i].count;
   *table_out = NULL;
   if (!total)
      return true;

   uint32_t *table = calloc(total, sizeof(uint32_t));
   if (!table)
      return false;
   *table_out = table;

   uint32_t *p = table;
   ctx->dyn_ubos.map = p;
   p += ctx->dyn_ubos.count;
   ctx->dyn_ubos.count = 0;
   ctx->dyn_ssbos.map = p;
   p += ctx->dyn_ssbos.count;
   ctx->dyn_ssbos.count = 0;
   for (unsigned i = 0; i < MALI_V7_DESC_TABLE_COUNT; i++) {
      ctx->others[i].map = p;
      p += ctx->others[i].count;
      ctx->others[i].count = 0;
   }

   hash_table_u64_foreach(ctx->ht, he) {
      const uint32_t count = (uintptr_t)he.data;
      struct desc_id id = { .ht_key = he.key };
      uint32_t *first = fill_copy_descs(ctx, id.set, id.binding, id.sampler_subdesc, count);
      _mesa_hash_table_u64_replace(ctx->ht, &he, first);
   }
   return true;
}

bool
mali_nir_lower_descriptors_v7(nir_shader *nir, const struct vk_pipeline_robustness_state *rs,
                              nir_address_format ssbo_addr_format,
                              const struct vk_pipeline_layout *layout,
                              struct mali_shader_desc_info *desc_info,
                              struct mali_v7_shader_desc *v7)
{
   bool ok = true;
   struct lower_desc_ctx ctx = {
      .stage = nir->info.stage,
      .add_bounds_checks =
         rs->storage_buffers != VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT ||
         rs->uniform_buffers != VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT ||
         rs->images != VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
      .ssbo_addr_format = ssbo_addr_format,
   };
   bool progress = false;

   memset(v7->count, 0, sizeof(v7->count));
   v7->dyn_ubo_count = 0;
   v7->map = NULL;
   v7->map_count = 0;
   v7->needs_set_addrs = false;

   ctx.ht = _mesa_hash_table_u64_create(NULL);

   for (uint32_t i = 0; layout && i < layout->set_count && i < MALI_MAX_SETS; i++) {
      if (layout->set_layouts[i])
         ctx.set_layouts[i] = mali_descriptor_set_layout(layout->set_layouts[i]);
   }

   NIR_PASS(progress, nir, nir_shader_instructions_pass, collect_desc_access,
            nir_metadata_all, &ctx);
   uint32_t *table = NULL;
   if (progress && !create_copy_table(&ctx, &table)) {
      ok = false;
      progress = false;
   }
   if (progress) {
      /* The dynamic buffers, uniform ones first, go to the shared map; the
       * per-type tables keep the rest. */
      assert(ctx.dyn_ubos.count + ctx.dyn_ssbos.count <= ARRAY_SIZE(desc_info->dyn_bufs.map));
      desc_info->dyn_bufs.count = 0;
      for (uint32_t i = 0; i < ctx.dyn_ubos.count; i++)
         desc_info->dyn_bufs.map[desc_info->dyn_bufs.count++] = ctx.dyn_ubos.map[i];
      for (uint32_t i = 0; i < ctx.dyn_ssbos.count; i++)
         desc_info->dyn_bufs.map[desc_info->dyn_bufs.count++] = ctx.dyn_ssbos.map[i];
      v7->dyn_ubo_count = ctx.dyn_ubos.count;

      uint32_t others = 0;
      for (unsigned i = 0; i < MALI_V7_DESC_TABLE_COUNT; i++) {
         v7->count[i] = ctx.others[i].count;
         others += ctx.others[i].count;
      }
      if (others) {
         v7->map = malloc(others * sizeof(uint32_t));
         if (v7->map) {
            /* The per-type maps are consecutive behind the dynamic ones. */
            memcpy(v7->map, ctx.others[0].map, others * sizeof(uint32_t));
            v7->map_count = others;
         } else {
            ok = false;
         }
      }
      desc_info->used_set_mask = ctx.used_set_mask;

      if (ok)
         NIR_PASS(_, nir, nir_shader_instructions_pass, lower_descriptors_instr,
                  nir_metadata_control_flow, &ctx);
   }
   free(table);

   v7->needs_set_addrs = ctx.needs_set_addrs;
   _mesa_hash_table_u64_destroy(ctx.ht);

   if (ok && nir->constant_data_size)
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_load_constant,
               nir_metadata_control_flow, NULL);
   return ok;
}
