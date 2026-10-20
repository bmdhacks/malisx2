/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Samplers, buffer views, descriptor pools and descriptor sets.
 *
 * A set is a run of 32-byte hardware descriptors in GPU memory, laid out
 * by its set layout (mali_descriptor_set_layout.h, panvk's layout). Every
 * descriptor is packed once, when the object it describes is created
 * (sampler, image view, buffer view); vkUpdateDescriptorSets only copies
 * those 32 bytes into the set, except for uniform and storage buffers,
 * whose Buffer descriptor is packed at the write. Dynamic buffers have no
 * slot: their address and range stay in the set's host memory and the
 * command buffer writes them, with the dynamic offset applied, into the
 * driver's resource table (table 0) at draw or dispatch time. v7 has no
 * resource tables: a dynamic uniform buffer goes into the stage's Uniform
 * Buffer table behind the regular ones, a dynamic storage buffer into the
 * stage's dynamic storage buffer table (mali_jm_desc_v7.c).
 *
 * Set memory is CPU-uncached and GPU read-only (the blob's descriptor pool
 * without dynamic bindings), so a CPU write needs no cache maintenance and
 * the GPU reads a set in place.
 */

#ifndef MALI_DESCRIPTOR_SET_H
#define MALI_DESCRIPTOR_SET_H

#include <stdint.h>

#include "util/bitset.h"
#include "util/u_math.h"
#include "util/vma.h"
#include "vk_buffer_view.h"
#include "vk_object.h"
#include "vk_sampler.h"

#include "kbase/kbase.h"

#include "mali_arch.h"
#include "mali_descriptor_set_layout.h"

struct mali_device;

struct mali_sampler {
   struct vk_sampler vk;
   struct mali_hw_desc desc;    /* Sampler */
};

VK_DEFINE_NONDISP_HANDLE_CASTS(mali_sampler, vk.base, VkSampler,
                               VK_OBJECT_TYPE_SAMPLER)

struct mali_buffer_view {
   struct vk_buffer_view vk;
   /* Buffer, type Structure; on v7 a 1D Attribute Buffer and an Attribute
    * (Bifrost reads texel buffers as images). */
   struct mali_hw_desc desc;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(mali_buffer_view, vk.base, VkBufferView,
                               VK_OBJECT_TYPE_BUFFER_VIEW)

struct mali_descriptor_pool;

struct mali_descriptor_set {
   struct vk_object_base base;
   struct mali_descriptor_pool *pool;
   /* The pool holds a reference to it (mali_descriptor_pool.held), or the
    * set does when own_layout_ref is set. */
   const struct mali_descriptor_set_layout *layout;
   bool own_layout_ref;

   /* The set's slots: layout->desc_count descriptors. Both 0/NULL for a
    * layout without slots (dynamic buffers only, or empty). */
   uint64_t gpu;
   void *cpu;

   /* Dynamic uniform/storage buffers, by the index the layout gives them
    * (binding desc_idx + array element). */
   struct {
      uint64_t addr;
      uint64_t range;
   } dyn_bufs[MALI_MAX_DYNAMIC_BUFFERS];
};

VK_DEFINE_NONDISP_HANDLE_CASTS(mali_descriptor_set, base, VkDescriptorSet,
                               VK_OBJECT_TYPE_DESCRIPTOR_SET)

struct mali_descriptor_pool {
   struct vk_object_base base;

   /* All set memory: one kbase allocation, sub-allocated by a heap. No
    * allocation (gpu_va 0) when the pool holds only dynamic buffers. */
   struct mali_kbase_bo bo;
   struct util_vma_heap heap;
   uint64_t heap_size;

   /* Descriptors of each type still available (mali_desc_type_index). */
   uint32_t avail[MALI_DESC_TYPE_COUNT];
   uint32_t size[MALI_DESC_TYPE_COUNT];

   /* Layouts the pool holds a reference to until it is destroyed, so that
    * allocating and freeing sets of them costs no reference count atomics
    * (ARMSX2 allocates sets per draw from a few layouts). */
   struct mali_descriptor_set_layout *held[8];
   uint32_t held_count;

   uint32_t max_sets;
   uint32_t live_sets;
   BITSET_WORD *free_sets;      /* bit i: sets[i] is free */
   struct mali_descriptor_set *sets;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(mali_descriptor_pool, base, VkDescriptorPool,
                               VK_OBJECT_TYPE_DESCRIPTOR_POOL)

/* Bytes of GPU memory a set of this layout occupies. */
static inline uint64_t
mali_descriptor_set_size(const struct mali_descriptor_set *set)
{
   return (uint64_t)set->layout->desc_count * MALI_DESCRIPTOR_SIZE;
}

/* Where a dynamic buffer bound to nothing points: an address the GPU's
 * memory sink absorbs (panvk). */
#define MALI_NULL_BUFFER_ADDR (0x8ull << 60)

/*
 * v7: a storage buffer's set slot, and an entry of a stage's dynamic
 * storage buffer table (panvk_ssbo_addr): what the shader's
 * load_vulkan_descriptor reads (mali_nir_lower_descriptors_v7.c).
 */
struct mali_v7_ssbo_addr {
   uint64_t base_addr;
   uint32_t size;
   uint32_t zero[5];
};
static_assert(sizeof(struct mali_v7_ssbo_addr) == MALI_DESCRIPTOR_SIZE, "one slot");

/* The v7 storage buffer entry of range bytes at addr. The size is rounded
 * up to 4 bytes, as panvk's, and kept within its 32 bits. */
static inline struct mali_v7_ssbo_addr
mali_v7_ssbo_addr(uint64_t addr, uint64_t range)
{
   return (struct mali_v7_ssbo_addr){
      .base_addr = addr,
      .size = (uint32_t)MIN2(align64(range, 4), UINT32_MAX & ~3ull),
   };
}

/* The Entries of a v7 Uniform Buffer descriptor of range bytes: 16-byte
 * entries, at least one (the field is the count minus one) and at most
 * 4096: 64 KiB, this arch's maxUniformBufferRange. */
static inline uint32_t
mali_v7_ubo_entries(uint64_t range)
{
   return CLAMP(DIV_ROUND_UP(range, 16), 1, 1u << 12);
}

/* ---------------------------------------------------------------------- */
/* For the command buffer: what a bound set contributes to the shader
 * resource tables of a draw or dispatch. None of these touch the command
 * stream; they pack descriptors into memory the caller provides.        */

/* Size of one Resource entry of a resource table. */
#define MALI_RESOURCE_SIZE 16

/*
 * The Resource entry for a bound set (resource table set + 1): the set's
 * memory, Contains descriptors set, size in bytes. set may be NULL (an
 * unbound or unused set number): an empty entry. out: 16 bytes, 16-byte
 * aligned in a 64-byte aligned table. v9 and v11 only: v7 has no resource
 * tables.
 */
MALI_PER_ARCH_DECL(void, descriptor_set_pack_resource,
                   (const struct mali_descriptor_set *set, void *out));

/*
 * The Buffer descriptor for dynamic buffer idx of set with its dynamic
 * offset applied, for the driver table (table 0) slot the shader's
 * mali_shader_desc_info::dyn_bufs map names. Sizes are rounded up to 16
 * bytes for uniform buffers and 4 for storage buffers (panvk). out: 32
 * bytes. v9 and v11 only: v7 packs its dynamic buffers into the stage's
 * tables (mali_jm_desc_v7.c).
 */
MALI_PER_ARCH_DECL(void, descriptor_set_pack_dyn_buf,
                   (const struct mali_descriptor_set *set, uint32_t idx,
                    uint32_t dynamic_offset, void *out));

/* The dummy sampler of the driver table, or on v7 of a stage's Sampler
 * table (texel fetches still name a sampler on Valhall and Bifrost). out:
 * 32 bytes. */
MALI_PER_ARCH_DECL(void, pack_dummy_sampler, (void *out));

/* ---------------------------------------------------------------------- */
/* For push descriptors (VK_KHR_push_descriptor, mali_cmd_state.c): the
 * same per-type packers vkUpdateDescriptorSets uses, for a set whose
 * memory the command buffer owns instead of a pool. */

/* Writes the slots set's own layout fixes (an inline uniform block's
 * self-referencing Buffer descriptor, a binding's immutable samplers).
 * Idempotent: safe to call again on a set a caller is not sure was
 * already initialized. set->layout, ->cpu and ->gpu must already be set. */
MALI_PER_ARCH_DECL(void, descriptor_set_init_fixed_slots,
                   (struct mali_descriptor_set *set));

/* Applies write_count writes to set, as vkUpdateDescriptorSets would; a
 * write's dstSet is ignored, as the spec has it for a push. */
MALI_PER_ARCH_DECL(void, descriptor_set_write,
                   (struct mali_descriptor_set *set, uint32_t write_count,
                    const VkWriteDescriptorSet *writes));

#endif
