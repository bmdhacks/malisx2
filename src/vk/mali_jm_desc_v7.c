/*
 * SPDX-FileCopyrightText: 2026 TheGammaSqueeze
 * SPDX-License-Identifier: MIT
 */

/*
 * The Bifrost (v7) descriptor tables of a draw or dispatch, built by one
 * builder for both (struct mali_v7_desc_tables, mali_jm.h), and the job
 * that copies uniform buffer words into a FAU block. Follows panvk's
 * Bifrost job-manager path (Mesa panvk_vX_cmd_desc_state.c and
 * bifrost/panvk_vX_meta_desc_copy.c) on this driver's command memory,
 * shader objects and job chains.
 *
 * The model, the same for the vertex and fragment stage of a draw and for
 * a dispatch:
 *
 *  - mali_v7_desc_prepare allocates the stage's tables in command memory
 *    and writes what only the CPU knows: the dynamic uniform buffers
 *    (behind the regular ones in the Uniform Buffer table) and the dynamic
 *    storage buffer table, both with the bound dynamic offsets applied, the
 *    dummy sampler, and the bound sets' addresses;
 *  - mali_v7_desc_copy fills the entries that come from the sets (uniform
 *    buffers, textures, samplers, images and texel buffers, in the order of
 *    the shader's copy map) with one small Compute job in the same chain.
 *    The job that reads the tables depends on it: the Vertex job on the
 *    vertex shader's, the dispatch on the compute shader's. A fragment
 *    shader's tables are read by the fragment job, which starts only once
 *    the whole vtc atom is done, so the Tiler job needs no dependency on
 *    that copy job (panvk adds none either);
 *  - storage buffers and size queries are not copied: the shader reads the
 *    set itself through the set addresses in its system values
 *    (mali_v7_desc_sysvals).
 *
 * The copy runs on the GPU, as panvk's does, so the tables hold the sets as
 * they are when the command buffer executes; a copy on the CPU would read
 * set memory, which is CPU-uncached, one entry at a time.
 */

#include "mali_cmd_state.h"

#if PAN_ARCH >= 9
#error "mali_jm_desc_v7.c is the Bifrost (PAN_ARCH 7) descriptor model"
#endif

#include <string.h>

#include "nir_builder.h"
#include "util/mesa-blake3.h"
#include "vk_pipeline.h"

#include "mali_arch.h"

/* ---------------------------------------------------------------------- */
/* The bound sets                                                          */

/* The bound set holding the dynamic buffer a copy handle names, with the
 * set's number and the buffer's index in it. NULL when that set is not
 * bound or has no such buffer. */
static inline const struct mali_descriptor_set *
dyn_set(const struct mali_desc_state *desc, uint32_t handle, uint32_t *set, uint32_t *idx)
{
   *set = MALI_COPY_DESC_HANDLE_SET(handle);
   *idx = MALI_COPY_DESC_HANDLE_INDEX(handle);
   const struct mali_descriptor_set *s = desc && *set < MALI_MAX_SETS ? desc->sets[*set] : NULL;
   return s && *idx < s->layout->dyn_buf_count ? s : NULL;
}

/* The dynamic buffer a copy handle names: its address with the bound
 * dynamic offset applied, and its range. False when dyn_set finds no
 * buffer. */
static bool
dyn_buf(const struct mali_desc_state *desc, uint32_t handle, uint64_t *addr, uint64_t *range)
{
   uint32_t set, idx;
   const struct mali_descriptor_set *s = dyn_set(desc, handle, &set, &idx);
   if (!s)
      return false;
   *addr = s->dyn_bufs[idx].addr + desc->dyn_offsets[set][idx];
   *range = s->dyn_bufs[idx].range;
   return true;
}

/* dyn_buf, with 0 and 0 when there is no buffer. */
static inline void
dyn_buf_or_none(const struct mali_desc_state *desc, uint32_t handle, uint64_t *addr,
                uint64_t *range)
{
   if (!dyn_buf(desc, handle, addr, range))
      *addr = *range = 0;
}

/* ---------------------------------------------------------------------- */
/* The tables                                                              */

static unsigned
table_entry_size(unsigned t)
{
   return t == MALI_V7_DESC_TABLE_UBO ? pan_size(UNIFORM_BUFFER) : MALI_DESCRIPTOR_SIZE;
}

/* The dynamic uniform buffers' entries, offsets applied
 * (panvk_cmd_fill_dyn_ubos), packed into dst. One bound to nothing (or to
 * the null buffer, whose sink address does not fit the descriptor) reads
 * as one zero entry. */
static void
pack_dyn_ubos(const struct mali_shader *s, const struct mali_desc_state *desc, void *dst)
{
   const uint32_t dyn_ubo_count = s->v7.dyn_ubo_count;
   const uint32_t *dyn_ubo_map = s->desc.dyn_bufs.map;
   /* Packed in cached memory and copied out once: command memory is
    * uncached. */
   struct mali_uniform_buffer_packed dyn[MALI_MAX_DYNAMIC_UNIFORM_BUFFERS];
   assert(dyn_ubo_count <= ARRAY_SIZE(dyn));
   for (uint32_t i = 0; i < dyn_ubo_count; i++) {
      uint64_t addr = 0, range = 0;
      if (!dyn_buf(desc, dyn_ubo_map[i], &addr, &range) || (addr >> 56))
         addr = range = 0;
      pan_pack(&dyn[i], UNIFORM_BUFFER, cfg) {
         cfg.pointer = addr;
         cfg.entries = mali_v7_ubo_entries(range);
      }
   }
   memcpy(dst, dyn, dyn_ubo_count * sizeof(dyn[0]));
}

/* The dynamic storage buffer table: address and size, offset applied
 * (cmd_prepare_dyn_ssbos). One bound to nothing, or with no bytes, points
 * at the null buffer. 0 on failure. */
static uint64_t
build_dyn_ssbos(struct mali_cmd_buffer *cmd, const struct mali_shader *s,
                const struct mali_desc_state *desc)
{
   const uint32_t dyn_ubo_count = s->v7.dyn_ubo_count;
   const uint32_t dyn_ssbo_count = s->desc.dyn_bufs.count - dyn_ubo_count;
   const uint32_t *dyn_ssbo_map = s->desc.dyn_bufs.map + dyn_ubo_count;
   struct mali_ptr p =
      mali_cmd_alloc(cmd, (uint64_t)dyn_ssbo_count * sizeof(struct mali_v7_ssbo_addr), 64);
   if (!p.cpu)
      return 0;
   struct mali_v7_ssbo_addr entries[MALI_MAX_DYNAMIC_STORAGE_BUFFERS];
   assert(dyn_ssbo_count <= ARRAY_SIZE(entries));
   for (uint32_t i = 0; i < dyn_ssbo_count; i++) {
      uint64_t addr = 0, range = 0;
      if (!dyn_buf(desc, dyn_ssbo_map[i], &addr, &range) || !range)
         addr = MALI_NULL_BUFFER_ADDR;
      entries[i] = mali_v7_ssbo_addr(addr, range);
   }
   memcpy(p.cpu, entries, dyn_ssbo_count * sizeof(entries[0]));
   return p.gpu;
}

static inline void
save_dyn_bufs(const struct mali_shader *s, const struct mali_desc_state *desc,
              struct mali_v7_desc_tables *t)
{
   for (uint32_t i = 0; i < s->desc.dyn_bufs.count; i++)
      dyn_buf_or_none(desc, s->desc.dyn_bufs.map[i], &t->dyn_addrs[i], &t->dyn_ranges[i]);
}

bool
mali_v7_desc_refresh(struct mali_cmd_buffer *cmd, const struct mali_shader *s,
                     const struct mali_desc_state *desc, bool img,
                     struct mali_v7_desc_tables *t, bool *unchanged)
{
   const struct mali_v7_shader_desc *v7 = &s->v7;
   *unchanged = false;
   if (t->shader != s || v7->count[MALI_V7_DESC_TABLE_UBO] ||
       (!img && v7->count[MALI_V7_DESC_TABLE_IMG]))
      return false;
   for (unsigned i = 0; i < MALI_MAX_SETS; i++) {
      const struct mali_descriptor_set *set = desc ? desc->sets[i] : NULL;
      if (t->sets[i] != (set ? set->gpu : 0) ||
          t->set_desc_counts[i] != (set ? set->layout->desc_count : 0))
         return false;
   }

   /* The same sets and the same dynamic buffers: the tables stand as they
    * are. The buffers are compared, not the offsets: two sets without
    * descriptor slots both have the address 0. */
   bool same = true;
   for (uint32_t i = 0; i < s->desc.dyn_bufs.count && same; i++) {
      uint64_t addr, range;
      dyn_buf_or_none(desc, s->desc.dyn_bufs.map[i], &addr, &range);
      same = t->dyn_addrs[i] == addr && t->dyn_ranges[i] == range;
   }
   if (same) {
      *unchanged = true;
      return true;
   }

   const uint32_t dyn_ubo_count = v7->dyn_ubo_count;
   const uint32_t dyn_ssbo_count = s->desc.dyn_bufs.count - dyn_ubo_count;
   uint64_t ubo = t->tables[MALI_V7_DESC_TABLE_UBO];
   if (dyn_ubo_count) {
      struct mali_ptr p =
         mali_cmd_alloc(cmd, (uint64_t)dyn_ubo_count * pan_size(UNIFORM_BUFFER), 64);
      if (!p.cpu)
         return false;
      pack_dyn_ubos(s, desc, p.cpu);
      ubo = p.gpu;
   }
   uint64_t dyn_ssbos = t->dyn_ssbos;
   if (dyn_ssbo_count) {
      dyn_ssbos = build_dyn_ssbos(cmd, s, desc);
      if (!dyn_ssbos)
         return false;
   }
   t->tables[MALI_V7_DESC_TABLE_UBO] = ubo;
   t->dyn_ssbos = dyn_ssbos;
   save_dyn_bufs(s, desc, t);
   return true;
}

bool
mali_v7_desc_prepare(struct mali_cmd_buffer *cmd, const struct mali_shader *s,
                     const struct mali_desc_state *desc, bool img,
                     struct mali_v7_desc_tables *out)
{
   const struct mali_v7_shader_desc *v7 = &s->v7;
   /* desc.dyn_bufs.map: the dynamic uniform buffers first, then the
    * dynamic storage buffers (mali_nir_lower_descriptors_v7.c). */
   const uint32_t dyn_ubo_count = v7->dyn_ubo_count;
   const uint32_t dyn_ssbo_count = s->desc.dyn_bufs.count - dyn_ubo_count;

   memset(out, 0, sizeof(*out));
   out->shader = s;
   if (!s->desc.used_set_mask && !v7->map_count && !s->desc.dyn_bufs.count)
      return true;

   /* Every bound set: the copy shader's sources (panvk passes them all). */
   for (unsigned i = 0; i < MALI_MAX_SETS; i++) {
      const struct mali_descriptor_set *set = desc ? desc->sets[i] : NULL;
      out->sets[i] = set ? set->gpu : 0;
      out->set_desc_counts[i] = set ? set->layout->desc_count : 0;
   }

   for (unsigned t = 0; t < MALI_V7_DESC_TABLE_COUNT; t++) {
      const uint32_t regular = v7->count[t];
      const uint32_t count = regular + (t == MALI_V7_DESC_TABLE_UBO ? dyn_ubo_count : 0);
      const unsigned entry = table_entry_size(t);
      if (!count || (t == MALI_V7_DESC_TABLE_IMG && !img))
         continue;

      /* The image table is an Attribute Buffer table: an empty entry after
       * the pairs stops the Bifrost prefetcher, as behind the vertex and
       * varying buffers of a draw. */
      const uint64_t tail = t == MALI_V7_DESC_TABLE_IMG ? pan_size(ATTRIBUTE_BUFFER) : 0;
      struct mali_ptr p = mali_cmd_alloc(cmd, (uint64_t)count * entry + tail, 64);
      if (!p.cpu)
         return false;
      out->tables[t] = p.gpu;

      /* The dynamic uniform buffers behind the regular ones. */
      if (t == MALI_V7_DESC_TABLE_UBO && dyn_ubo_count)
         pack_dyn_ubos(s, desc, (uint8_t *)p.cpu + (uint64_t)regular * entry);

      if (t == MALI_V7_DESC_TABLE_IMG) {
         memset((uint8_t *)p.cpu + (uint64_t)count * entry, 0, tail);
         /* One Attribute descriptor per image, in a table of its own that
          * the copy writes with the Attribute Buffer pairs. */
         struct mali_ptr a = mali_cmd_alloc(cmd, (uint64_t)count * pan_size(ATTRIBUTE), 64);
         if (!a.cpu)
            return false;
         out->img_attribs = a.gpu;
      }
   }

   /* A shader with textures and no sampler of its own gets one: texel
    * fetches still name sampler 0 (panvk). */
   if (v7->count[MALI_V7_DESC_TABLE_TEXTURE] && !v7->count[MALI_V7_DESC_TABLE_SAMPLER]) {
      struct mali_ptr p = mali_cmd_alloc(cmd, pan_size(SAMPLER), 64);
      if (!p.cpu)
         return false;
      MALI_PER_ARCH(pack_dummy_sampler)(p.cpu);
      out->tables[MALI_V7_DESC_TABLE_SAMPLER] = p.gpu;
   }

   /* The dynamic storage buffers. */
   if (dyn_ssbo_count) {
      out->dyn_ssbos = build_dyn_ssbos(cmd, s, desc);
      if (!out->dyn_ssbos)
         return false;
   }
   save_dyn_bufs(s, desc, out);
   return true;
}

void
mali_v7_desc_sysvals(const struct mali_v7_desc_tables *t, const struct mali_shader *s,
                     uint64_t sets[MALI_MAX_SETS + 1])
{
   for (unsigned i = 0; i < MALI_MAX_SETS; i++)
      sets[i] = (s->desc.used_set_mask & BITFIELD_BIT(i)) ? t->sets[i] : 0;
   sets[MALI_V7_SETS_DYN_SSBOS] = t->dyn_ssbos;
}

/* ---------------------------------------------------------------------- */
/* The copy jobs                                                           */

/*
 * The shader of a copy job, kept in *slot (struct mali_jm_device). On
 * first use, build makes its NIR and the normal path compiles it, keyed
 * by a hash of name; the shader then stays until the device is destroyed,
 * as the copy and fill shaders of mali_cmd_copy.c do. NULL on failure,
 * with the command buffer's error set. A compiled compute shader always
 * has its Renderer State Descriptor (spd[MALI_SPD_MAIN], mali_shader.c).
 */
static const struct mali_shader *
get_copy_shader(struct mali_cmd_buffer *cmd, struct mali_shader **slot,
                nir_shader *(*build)(struct mali_device *dev), const char *name)
{
   struct mali_device *dev = cmd->dev;

   /* Set once under the lock and kept until the device is destroyed, so a
    * caller that finds it set needs no lock. */
   struct mali_shader *s = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
   if (likely(s))
      return s;
   simple_mtx_lock(&dev->meta_lock);
   s = *slot;
   if (!s) {
      nir_shader *nir = build(dev);
      MALI_PER_ARCH(shader_preprocess)(dev, nir);

      const struct vk_pipeline_robustness_state rs = {
         .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
         .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
         .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
         .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
      };
      const struct mali_shader_compile_info info = {.nir = nir, .rs = &rs};
      blake3_hash key;
      struct mesa_blake3 h;
      _mesa_blake3_init(&h);
      _mesa_blake3_update(&h, name, strlen(name));
      _mesa_blake3_final(&h, key);
      if (MALI_PER_ARCH(shader_compile)(dev, &info, key, &s) != VK_SUCCESS)
         s = NULL;
      __atomic_store_n(slot, s, __ATOMIC_RELEASE);
   }
   simple_mtx_unlock(&dev->meta_lock);

   if (!s)
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   assert(!s || s->spd[MALI_SPD_MAIN]);
   return s;
}

/* The Local Storage descriptor of a copy job: the same for every copy job
 * of the command buffer when the copy shader has no thread or workgroup
 * storage, so made once per command buffer. */
static uint64_t
copy_tsd(struct mali_cmd_buffer *cmd, const struct mali_shader *copy, const uint32_t groups[3])
{
   if (copy->info.tls_size || copy->info.wls_size) {
      const struct mali_thread_props tp = mali_thread_props(cmd->dev);
      return mali_compute_tsd(cmd, copy, &tp, groups);
   }
   if (!cmd->jm.copy_tsd) {
      const struct mali_thread_props tp = mali_thread_props(cmd->dev);
      cmd->jm.copy_tsd = mali_compute_tsd(cmd, copy, &tp, groups);
   }
   return cmd->jm.copy_tsd;
}

/*
 * Adds a Compute job running copy shader copy, with push as its push
 * constants, on chain c behind job dep (0: none). It runs count workgroups
 * of one invocation, the workgroup size the copy shaders are compiled with
 * (panvk runs its descriptor copy as one workgroup of count). *index (when
 * not NULL): the job's index. False on failure, with the command buffer's
 * error set.
 */
static bool
add_copy_job(struct mali_cmd_buffer *cmd, struct mali_jm_chain *c, const struct mali_shader *copy,
             const void *push, uint32_t push_size, uint32_t count, uint16_t dep, uint16_t *index)
{
   const uint32_t base[3] = {0, 0, 0};
   const uint32_t groups[3] = {count, 1, 1};
   const uint32_t local[3] = {1, 1, 1};
   const uint64_t fau =
      mali_compute_fau(cmd, copy, NULL, push, push_size, base, groups) & BITFIELD64_MASK(56);
   const uint64_t tsd = copy_tsd(cmd, copy, groups);
   if (vk_command_buffer_has_error(&cmd->vk) || !tsd || (copy->fau.total_count && !fau))
      return false;

   struct mali_ptr job =
      mali_jm_cmd_add_job(cmd, c, MALI_JOB_TYPE_COMPUTE, pan_size(COMPUTE_JOB), false, dep);
   if (!job.cpu)
      return false;
   mali_v7_compute_invocation(pan_section_ptr(job.cpu, COMPUTE_JOB, INVOCATION), local, groups);
   pan_section_pack(job.cpu, COMPUTE_JOB, PARAMETERS, cfg) {
      cfg.job_task_split = mali_v7_compute_task_split(local);
   }
   pan_section_pack(job.cpu, COMPUTE_JOB, DRAW, cfg) {
      cfg.state = copy->spd[MALI_SPD_MAIN];
      cfg.push_uniforms = fau;
      cfg.thread_storage = tsd;
   }
   if (index)
      *index = c->index;
   return true;
}

/* ---------------------------------------------------------------------- */
/* The descriptor copy shader                                              */

/*
 * Mesa's Bifrost descriptor copy shader (src/panfrost/vulkan/bifrost/
 * panvk_vX_meta_desc_copy.c, Collabora, MIT): desc_copy_info and its
 * input macros, get_array_entry, set_to_table_copy, set_to_table_img_copy
 * and single_desc_copy follow it, on this driver's tables and set slots.
 */

/* Its push constants (FAU). */
struct desc_copy_info {
   uint64_t sets[MALI_MAX_SETS];
   uint64_t tables[MALI_V7_DESC_TABLE_COUNT];
   uint64_t img_attrib_table;
   struct {
      uint64_t table;                             /* the shader's copy map */
      uint32_t limits[MALI_V7_DESC_TABLE_COUNT];  /* cumulative entry counts */
      uint32_t attrib_buf_idx_offset;
   } desc_copy;
   uint32_t set_desc_counts[MALI_MAX_SETS];
};
static_assert(sizeof(struct desc_copy_info) <= MALI_MAX_PUSH_CONSTANTS_SIZE,
              "the copy shader's inputs fit the push constant block");

#define get_input_field(b, name)                                               \
   nir_load_push_constant(b, 1, sizeof(((struct desc_copy_info *)0)->name) * 8,   \
                          nir_imm_int(b, offsetof(struct desc_copy_info, name)))

/* Entry idx of a push-constant array, selected from all its entries (as
 * panvk does) so that every load keeps a constant offset and reads FAU. */
static nir_def *
get_array_entry(nir_builder *b, unsigned array_offset, unsigned array_size,
                unsigned array_stride, nir_def *idx)
{
   assert(array_size > 0);
   assert(array_stride == 4 || array_stride == 8);

   nir_def *lut[MALI_MAX_SETS];
   assert(array_size <= ARRAY_SIZE(lut));
   for (unsigned i = 0; i < array_size; i++) {
      lut[i] = nir_load_push_constant(b, 1, array_stride * 8,
                                      nir_imm_int(b, i * array_stride + array_offset));
   }
   for (unsigned lut_stride = BITFIELD_BIT(util_last_bit(array_size - 1) - 1);
        lut_stride > 0; lut_stride >>= 1) {
      nir_def *bit_is_set = nir_i2b(b, nir_iand_imm(b, idx, lut_stride));
      for (unsigned i = 0; i < lut_stride && i + lut_stride < array_size; i++)
         lut[i] = nir_bcsel(b, bit_is_set, lut[i + lut_stride], lut[i]);
   }
   return lut[0];
}

#define get_input_array_slot(b, name, index)                                   \
   get_array_entry(b, offsetof(struct desc_copy_info, name),                   \
                   ARRAY_SIZE(((struct desc_copy_info *)0)->name),             \
                   sizeof(((struct desc_copy_info *)0)->name[0]), index)

/* The first element_size bytes of set slot src_desc_idx (the whole slot,
 * or the Uniform Buffer descriptor at its start) to table entry
 * dst_desc_idx; zero for a slot past the set's end (a set that is not
 * bound has no slots). */
static void
set_to_table_copy(nir_builder *b, nir_def *set_ptr, nir_def *set_desc_count,
                  nir_def *src_desc_idx, nir_def *table_ptr, nir_def *dst_desc_idx,
                  unsigned element_size)
{
   nir_def *dst_offset = nir_u2u64(b, nir_imul_imm(b, dst_desc_idx, element_size));

   nir_push_if(b, nir_ult(b, src_desc_idx, set_desc_count));
   {
      nir_def *src_offset = nir_u2u64(b, nir_imul_imm(b, src_desc_idx, MALI_DESCRIPTOR_SIZE));
      nir_def *desc = nir_load_global(b, element_size / 4, 32, nir_iadd(b, set_ptr, src_offset),
                                      .align_mul = element_size);
      nir_store_global(b, desc, nir_iadd(b, table_ptr, dst_offset), .align_mul = element_size);
   }
   nir_push_else(b, NULL);
   {
      nir_def *zero = nir_imm_zero(b, element_size / 4, 32);
      nir_store_global(b, zero, nir_iadd(b, table_ptr, dst_offset), .align_mul = element_size);
   }
   nir_pop_if(b, NULL);
}

/*
 * The fields set_to_table_img_copy masks in the 32-bit words of a slot
 * (v7.xml): the Attribute Buffer's 6-bit Type (word 0), the 9-bit Buffer
 * index of a texel buffer's Attribute (word 4), which it replaces, and in
 * the stride word (word 2) the stride itself, below the sample count and
 * format an image folds in at bits 7 and 10 (mali_image_view.c). Bit 10
 * is where the Attribute's Format starts, so the format moves over as is.
 */
#define SLOT_TYPE_MASK BITFIELD_MASK(6)
#define SLOT_BUFFER_INDEX_MASK BITFIELD_MASK(9)
#define SLOT_STRIDE_MASK BITFIELD_MASK(7)
#define SLOT_FORMAT_SHIFT 10

/*
 * An image slot (an Attribute Buffer and its 3D continuation, the format
 * folded into the stride word: mali_image_view.c) or a texel buffer slot (a
 * 1D Attribute Buffer and an Attribute: mali_descriptor_set.c) to image
 * entry dst_desc_idx: the pair into the attribute buffer table, and an
 * Attribute descriptor naming buffer 2 * dst_desc_idx + attrib_buf_idx_offset
 * into the attribute table.
 */
static void
set_to_table_img_copy(nir_builder *b, nir_def *set_ptr, nir_def *set_desc_count,
                      nir_def *src_desc_idx, nir_def *attrib_table_ptr,
                      nir_def *attrib_buf_table_ptr, nir_def *dst_desc_idx)
{
   const unsigned element_size = pan_size(ATTRIBUTE_BUFFER) * 2;
   const unsigned attrib_buf_comps = element_size / 4;
   const unsigned attrib_comps = pan_size(ATTRIBUTE) / 4;
   nir_def *attrib_offset = nir_u2u64(b, nir_imul_imm(b, dst_desc_idx, pan_size(ATTRIBUTE)));
   nir_def *attrib_buf_offset = nir_u2u64(b, nir_imul_imm(b, dst_desc_idx, element_size));

   nir_push_if(b, nir_ult(b, src_desc_idx, set_desc_count));
   {
      nir_def *attr_buf_idx_offset = get_input_field(b, desc_copy.attrib_buf_idx_offset);
      nir_def *src_offset = nir_u2u64(b, nir_imul_imm(b, src_desc_idx, MALI_DESCRIPTOR_SIZE));
      nir_def *src_desc = nir_load_global(b, element_size / 4, 32, nir_iadd(b, set_ptr, src_offset),
                                          .align_mul = element_size);
      nir_def *attr_buf_type = nir_iand_imm(b, nir_channel(b, src_desc, 0), SLOT_TYPE_MASK);
      nir_def *buf_idx = nir_iadd(b, nir_imul_imm(b, dst_desc_idx, 2), attr_buf_idx_offset);

      /* Texel buffer. */
      nir_push_if(b, nir_ieq_imm(b, attr_buf_type, MALI_ATTRIBUTE_TYPE_1D));
      {
         nir_def *fmt_offsetenable =
            nir_iand_imm(b, nir_channel(b, src_desc, 4), ~SLOT_BUFFER_INDEX_MASK);
         nir_def *attrib_desc = nir_vec2(b, nir_ior(b, fmt_offsetenable, buf_idx),
                                         nir_channel(b, src_desc, 5));
         nir_store_global(b, attrib_desc, nir_iadd(b, attrib_table_ptr, attrib_offset),
                          .align_mul = pan_size(ATTRIBUTE),
                          .write_mask = nir_component_mask(attrib_comps));

         nir_def *null_desc = nir_imm_zero(b, attrib_buf_comps / 2, 32);
         nir_def *attrib_buf_desc = nir_vec8(
            b, nir_channel(b, src_desc, 0), nir_channel(b, src_desc, 1),
            nir_iand_imm(b, nir_channel(b, src_desc, 2), SLOT_STRIDE_MASK),
            nir_channel(b, src_desc, 3), nir_channel(b, null_desc, 0),
            nir_channel(b, null_desc, 1), nir_channel(b, null_desc, 2),
            nir_channel(b, null_desc, 3));
         nir_store_global(b, attrib_buf_desc, nir_iadd(b, attrib_buf_table_ptr, attrib_buf_offset),
                          .align_mul = element_size,
                          .write_mask = nir_component_mask(attrib_buf_comps));
      }
      /* Image. */
      nir_push_else(b, NULL);
      {
         nir_def *fmt =
            nir_iand_imm(b, nir_channel(b, src_desc, 2), ~BITFIELD_MASK(SLOT_FORMAT_SHIFT));
         nir_def *attrib_desc = nir_vec2(b, nir_ior(b, buf_idx, fmt), nir_imm_int(b, 0));
         nir_store_global(b, attrib_desc, nir_iadd(b, attrib_table_ptr, attrib_offset),
                          .align_mul = pan_size(ATTRIBUTE),
                          .write_mask = nir_component_mask(attrib_comps));

         nir_def *attrib_buf_desc = nir_vec8(
            b, nir_channel(b, src_desc, 0), nir_channel(b, src_desc, 1),
            nir_iand_imm(b, nir_channel(b, src_desc, 2), SLOT_STRIDE_MASK),
            nir_channel(b, src_desc, 3), nir_channel(b, src_desc, 4),
            nir_channel(b, src_desc, 5), nir_channel(b, src_desc, 6),
            nir_channel(b, src_desc, 7));
         nir_store_global(b, attrib_buf_desc, nir_iadd(b, attrib_buf_table_ptr, attrib_buf_offset),
                          .align_mul = element_size,
                          .write_mask = nir_component_mask(attrib_buf_comps));
      }
      nir_pop_if(b, NULL);
   }
   nir_push_else(b, NULL);
   {
      /* panvk swaps the two alignments here; an Attribute entry is only
       * 8-byte aligned. */
      nir_def *desc = nir_imm_zero(b, MAX2(attrib_buf_comps, attrib_comps), 32);
      nir_store_global(b, desc, nir_iadd(b, attrib_buf_table_ptr, attrib_buf_offset),
                       .align_mul = element_size,
                       .write_mask = nir_component_mask(attrib_buf_comps));
      nir_store_global(b, desc, nir_iadd(b, attrib_table_ptr, attrib_offset),
                       .align_mul = pan_size(ATTRIBUTE),
                       .write_mask = nir_component_mask(attrib_comps));
   }
   nir_pop_if(b, NULL);
}

/* One invocation: copy map entry desc_copy_idx (a MALI_COPY_DESC_HANDLE)
 * into whichever table the cumulative limits place it in. */
static void
single_desc_copy(nir_builder *b, nir_def *desc_copy_idx)
{
   nir_def *desc_copy_offset = nir_imul_imm(b, desc_copy_idx, sizeof(uint32_t));
   nir_def *desc_copy_ptr = nir_iadd(b, get_input_field(b, desc_copy.table),
                                     nir_u2u64(b, desc_copy_offset));
   nir_def *src_copy_handle = nir_load_global(b, 1, 32, desc_copy_ptr, .align_mul = 4);

   nir_def *set_idx = nir_ushr_imm(b, src_copy_handle, MALI_COPY_DESC_HANDLE_SET_SHIFT);
   nir_def *src_desc_idx =
      nir_iand_imm(b, src_copy_handle, BITFIELD_MASK(MALI_COPY_DESC_HANDLE_SET_SHIFT));

   nir_def *set_ptr = get_input_array_slot(b, sets, set_idx);
   nir_def *set_desc_count = get_input_array_slot(b, set_desc_counts, set_idx);
   nir_def *ubo_end = get_input_field(b, desc_copy.limits[MALI_V7_DESC_TABLE_UBO]);
   nir_def *img_end = get_input_field(b, desc_copy.limits[MALI_V7_DESC_TABLE_IMG]);
   nir_def *tex_end = get_input_field(b, desc_copy.limits[MALI_V7_DESC_TABLE_TEXTURE]);
   nir_def *sampler_end = get_input_field(b, desc_copy.limits[MALI_V7_DESC_TABLE_SAMPLER]);

   nir_push_if(b, nir_ult(b, desc_copy_idx, ubo_end));
   {
      nir_def *table_ptr = get_input_field(b, tables[MALI_V7_DESC_TABLE_UBO]);
      set_to_table_copy(b, set_ptr, set_desc_count, src_desc_idx, table_ptr, desc_copy_idx,
                        pan_size(UNIFORM_BUFFER));
   }
   nir_push_else(b, NULL);
   {
      nir_push_if(b, nir_ult(b, desc_copy_idx, img_end));
      {
         nir_def *table_ptr = get_input_field(b, tables[MALI_V7_DESC_TABLE_IMG]);
         nir_def *attrib_table_ptr = get_input_field(b, img_attrib_table);
         set_to_table_img_copy(b, set_ptr, set_desc_count, src_desc_idx, attrib_table_ptr,
                               table_ptr, nir_isub(b, desc_copy_idx, ubo_end));
      }
      nir_push_else(b, NULL);
      {
         nir_push_if(b, nir_ult(b, desc_copy_idx, tex_end));
         {
            nir_def *table_ptr = get_input_field(b, tables[MALI_V7_DESC_TABLE_TEXTURE]);
            set_to_table_copy(b, set_ptr, set_desc_count, src_desc_idx, table_ptr,
                              nir_isub(b, desc_copy_idx, img_end), pan_size(TEXTURE));
         }
         nir_push_else(b, NULL);
         {
            nir_push_if(b, nir_ult(b, desc_copy_idx, sampler_end));
            {
               nir_def *table_ptr = get_input_field(b, tables[MALI_V7_DESC_TABLE_SAMPLER]);
               set_to_table_copy(b, set_ptr, set_desc_count, src_desc_idx, table_ptr,
                                 nir_isub(b, desc_copy_idx, tex_end), pan_size(SAMPLER));
            }
            nir_pop_if(b, NULL);
         }
         nir_pop_if(b, NULL);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);
}

static nir_shader *
build_desc_copy_nir(struct mali_device *dev)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, MALI_PER_ARCH(shader_nir_options)(dev, MESA_SHADER_COMPUTE), "%s",
      "mali_v7_desc_copy");
   /* One invocation per workgroup: the job dispatches as many workgroups
    * as the map has entries (mali_v7_desc_copy). */
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *id = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   single_desc_copy(&b, id);
   return b.shader;
}

/* ---------------------------------------------------------------------- */
/* Uniform buffer words in FAU                                             */

/*
 * The copy job of a shader whose dynamic uniform buffer reads became FAU
 * reads (mali_shader.c lower_ubo_push, fau.ubo_push): one invocation per
 * 32-bit half of a pushed word, dst[i] = *table[i], where table[i] is the
 * address of that half in the bound buffer, offset applied, or of a zero
 * word for a half past the binding's range. A vertex shader's copy runs
 * between its descriptor copy and the Vertex job. A fragment shader's
 * needs no dependency: the fragment job reads the FAU block after the
 * whole vtc atom.
 */
struct ubo_copy_info {
   uint64_t table;
   uint64_t dst;
};

static nir_shader *
build_ubo_copy_nir(struct mali_device *dev)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, MALI_PER_ARCH(shader_nir_options)(dev, MESA_SHADER_COMPUTE), "%s",
      "mali_v7_ubo_copy");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *id = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *table =
      nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, offsetof(struct ubo_copy_info, table)));
   nir_def *dst =
      nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, offsetof(struct ubo_copy_info, dst)));
   nir_def *src = nir_load_global(
      &b, 1, 64, nir_iadd(&b, table, nir_u2u64(&b, nir_imul_imm(&b, id, 8))), .align_mul = 8);
   nir_def *v = nir_load_global(&b, 1, 32, src, .align_mul = 4);
   nir_store_global(&b, v, nir_iadd(&b, dst, nir_u2u64(&b, nir_imul_imm(&b, id, 4))),
                    .align_mul = 4);
   return b.shader;
}

bool
mali_v7_ubo_copy(struct mali_cmd_buffer *cmd, struct mali_jm_chain *c, const struct mali_shader *s,
                 const struct mali_desc_state *desc, uint64_t fau_va, uint16_t dep, uint16_t *index)
{
   const uint32_t n = s->fau.ubo_push_count;
   if (index)
      *index = dep;
   if (!n || !fau_va)
      return true;
   const struct mali_shader *copy =
      get_copy_shader(cmd, &cmd->dev->jm->v7_ubo_copy, build_ubo_copy_nir, "mali-v7-ubo-copy");
   if (!copy)
      return false;

   /* 2n source addresses, then the zero word they fall back to. */
   struct mali_ptr p = mali_cmd_alloc(cmd, (2 * n + 1) * sizeof(uint64_t), 16);
   if (!p.cpu)
      return false;
   const uint64_t zero = p.gpu + 2 * n * sizeof(uint64_t);
   uint64_t table[2 * MALI_MAX_UBO_PUSH_FAUS + 1];
   uint32_t dyn_cached = UINT32_MAX;
   uint64_t addr = 0, range = 0;
   for (uint32_t i = 0; i < n; i++) {
      const struct mali_ubo_push_word *w = &s->fau.ubo_push[i];
      if (w->dyn != dyn_cached) {
         dyn_cached = w->dyn;
         if (w->dyn >= s->v7.dyn_ubo_count ||
             !dyn_buf(desc, s->desc.dyn_bufs.map[w->dyn], &addr, &range) || (addr >> 56))
            addr = range = 0;
      }
      for (unsigned h = 0; h < 2; h++) {
         const uint64_t off = w->offset + 4 * h;
         table[2 * i + h] = addr && off + 4 <= range ? addr + off : zero;
      }
   }
   table[2 * n] = 0;
   memcpy(p.cpu, table, (2 * n + 1) * sizeof(uint64_t));

   const struct ubo_copy_info info = {
      .table = p.gpu,
      .dst = fau_va + (uint64_t)s->fau.ubo_push_start * MALI_FAU_WORD_SIZE,
   };
   return add_copy_job(cmd, c, copy, &info, sizeof(info), 2 * n, dep, index);
}

void
mali_v7_desc_finish(struct mali_device *dev)
{
   struct mali_jm_device *jd = dev->jm;
   MALI_PER_ARCH(shader_unref)(dev, jd->v7_desc_copy);
   jd->v7_desc_copy = NULL;
   MALI_PER_ARCH(shader_unref)(dev, jd->v7_ubo_copy);
   jd->v7_ubo_copy = NULL;
}

/* ---------------------------------------------------------------------- */
/* Filling the copied entries                                              */

bool
mali_v7_desc_copy(struct mali_cmd_buffer *cmd, struct mali_jm_chain *c,
                  const struct mali_shader *s, const struct mali_v7_desc_tables *t,
                  uint32_t attrib_buf_idx_offset, uint16_t *index)
{
   *index = 0;
   assert(t->shader == s);
   if (!s->v7.map_count)
      return true;
   /* The map is in GPU memory: shader_upload uploads it whatever the size
    * of the code, or fails the shader (mali_shader.c). */
   assert(s->v7.map_mem.gpu_va);
   /* Every table the map fills exists (a vertex shader's image tables are
    * the draw's, mali_jm_cmd_draw_v7.c). */
   for (unsigned tab = 0; tab < MALI_V7_DESC_TABLE_COUNT; tab++)
      assert(!s->v7.count[tab] || t->tables[tab]);
   assert(!s->v7.count[MALI_V7_DESC_TABLE_IMG] || t->img_attribs);
   /* The last image pair stays within the 9-bit Attribute buffer index
    * (mali_shader.h; v7_fixup_info refuses a bigger table). */
   assert(attrib_buf_idx_offset + 2 * s->v7.count[MALI_V7_DESC_TABLE_IMG] <=
          MALI_V7_ATTRIB_BUF_RECORDS);

   const struct mali_shader *copy =
      get_copy_shader(cmd, &cmd->dev->jm->v7_desc_copy, build_desc_copy_nir, "mali-v7-desc-copy");
   if (!copy)
      return false;

   struct desc_copy_info info;
   memset(&info, 0, sizeof(info));
   for (unsigned i = 0; i < MALI_MAX_SETS; i++) {
      info.sets[i] = t->sets[i];
      info.set_desc_counts[i] = t->set_desc_counts[i];
   }
   for (unsigned i = 0; i < MALI_V7_DESC_TABLE_COUNT; i++) {
      info.tables[i] = s->v7.count[i] ? t->tables[i] : 0;
      info.desc_copy.limits[i] = s->v7.count[i] + (i ? info.desc_copy.limits[i - 1] : 0);
   }
   info.img_attrib_table = t->img_attribs;
   info.desc_copy.table = s->v7.map_mem.gpu_va;
   info.desc_copy.attrib_buf_idx_offset = attrib_buf_idx_offset;
   const uint32_t copy_count = info.desc_copy.limits[MALI_V7_DESC_TABLE_COUNT - 1];
   assert(copy_count == s->v7.map_count && copy_count > 0);
   return add_copy_job(cmd, c, copy, &info, sizeof(info), copy_count, 0, index);
}
