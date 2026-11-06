/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * VkDescriptorSetLayout: where each binding's descriptors sit in a set.
 *
 * The layout is panvk's v9+ software layout (the slot assignment is a
 * contract between the driver and Mesa's compiler, so it must be
 * panvk's, not the blob's):
 *
 *  - every descriptor is one 32-byte hardware descriptor slot;
 *  - bindings take consecutive slots in binding-number order;
 *  - a combined image/sampler takes two slots per element, texture first,
 *    sampler second;
 *  - an inline uniform block takes one Buffer descriptor slot for itself
 *    plus enough slots for its bytes;
 *  - dynamic uniform/storage buffers take no slot in the set; they are
 *    numbered separately (desc_idx counts dynamic buffers) and the command
 *    buffer writes them into the driver's own resource table at bind time.
 *
 * Written for the pipeline code, which needs it to lower descriptor
 * accesses in shaders; sets, pools and updates are in mali_descriptor_set.h.
 */

#ifndef MALI_DESCRIPTOR_SET_LAYOUT_H
#define MALI_DESCRIPTOR_SET_LAYOUT_H

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

#include "vk_descriptor_set_layout.h"

/* One hardware descriptor. */
#define MALI_DESCRIPTOR_SIZE 32

/* A packed 32-byte hardware descriptor (Sampler, Texture, Buffer, ...),
 * kept on the CPU and copied into set memory. */
struct mali_hw_desc {
   uint32_t words[MALI_DESCRIPTOR_SIZE / 4];
};

/* Per-type descriptor counts, indexed by VkDescriptorType for the core
 * types (0..10); inline uniform blocks count bytes, at the last index. */
#define MALI_DESC_TYPE_IUB 11
#define MALI_DESC_TYPE_COUNT 12

static inline unsigned
mali_desc_type_index(VkDescriptorType type)
{
   return type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK ? MALI_DESC_TYPE_IUB :
                                                            (unsigned)type;
}

/* Sets a pipeline layout may have (panvk v11: Valhall allows 16 resource
 * tables, the driver keeps some for itself). */
#define MALI_MAX_SETS 7

#define MALI_MAX_DYNAMIC_UNIFORM_BUFFERS 16
#define MALI_MAX_DYNAMIC_STORAGE_BUFFERS 8
#define MALI_MAX_DYNAMIC_BUFFERS \
   (MALI_MAX_DYNAMIC_UNIFORM_BUFFERS + MALI_MAX_DYNAMIC_STORAGE_BUFFERS)

/* Hardware limit is 2^24 per set. */
#define MALI_MAX_DESCS_PER_SET (1u << 24)

/* Descriptors of one type per set and per shader stage, ordinary and
 * update-after-bind, as reported on v9 and v11 (panvk's MAX_DESCRIPTORS);
 * a combined image sampler counts as a sampler and a sampled image, and
 * takes two slots. v7 has less (mali_physical_device.c). */
#define MALI_MAX_DESCS_PER_TYPE (1u << 20)

struct mali_descriptor_set_binding_layout {
   VkDescriptorType type;
   VkDescriptorBindingFlags flags;
   /* Array elements (inline uniform block: slots, see above). */
   uint32_t desc_count;
   /* First slot in the set; for dynamic buffers, the index among the set's
    * dynamic buffers. */
   uint32_t desc_idx;
   uint32_t textures_per_desc;  /* 1 if the type has a texture, else 0 */
   uint32_t samplers_per_desc;  /* 1 if the type has a sampler, else 0 */
   /* The Sampler descriptors of pImmutableSamplers (desc_count entries),
    * copied at layout creation so the samplers may be destroyed afterwards,
    * or NULL. Written into every set at allocation; the shader side does
    * not use them (no YCbCr conversion). */
   const struct mali_hw_desc *immutable_samplers;
};

struct mali_descriptor_set_layout {
   struct vk_descriptor_set_layout vk;
   VkDescriptorSetLayoutCreateFlags flags;
   uint32_t desc_count;         /* slots in the set */
   uint32_t dyn_buf_count;
   uint32_t dyn_ssbos;          /* bit i: dynamic buffer i is a storage buffer */
   uint32_t binding_count;      /* highest binding number + 1 */
   struct mali_descriptor_set_binding_layout *bindings;
   /* Descriptors of each type a set of this layout takes from its pool
    * (mali_desc_type_index), the variable-count binding at its full size. */
   uint32_t type_counts[MALI_DESC_TYPE_COUNT];
   bool has_immutable_samplers;
   /* The binding with VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT,
    * if has_variable_count. It is the layout's last binding, not a dynamic
    * buffer or an inline uniform block; its desc_count is the largest count
    * a set may be given, and a set takes only the count it is allocated
    * with (mali_descriptor_set::variable_count). */
   bool has_variable_count;
   uint32_t variable_binding;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(mali_descriptor_set_layout, vk.base,
                               VkDescriptorSetLayout,
                               VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT)

static inline const struct mali_descriptor_set_layout *
mali_descriptor_set_layout(const struct vk_descriptor_set_layout *layout)
{
   return container_of(layout, const struct mali_descriptor_set_layout, vk);
}

/* Slots one array element takes. */
static inline uint32_t
mali_desc_stride(const struct mali_descriptor_set_binding_layout *b)
{
   return b->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ?
             b->textures_per_desc + b->samplers_per_desc : 1;
}

/* Which half of a combined image/sampler element. */
enum mali_subdesc {
   MALI_SUBDESC_NONE,     /* the descriptor itself (not combined) */
   MALI_SUBDESC_TEXTURE,
   MALI_SUBDESC_SAMPLER,
};

static inline uint32_t
mali_subdesc_offset(const struct mali_descriptor_set_binding_layout *b,
                    enum mali_subdesc sub)
{
   if (b->type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
      return 0;
   return sub == MALI_SUBDESC_SAMPLER ? b->textures_per_desc : 0;
}

/* Slot of element elem (not for dynamic buffers). */
static inline uint32_t
mali_desc_index(const struct mali_descriptor_set_binding_layout *b,
                uint32_t elem, enum mali_subdesc sub)
{
   return b->desc_idx + elem * mali_desc_stride(b) + mali_subdesc_offset(b, sub);
}

/* Slots a set of this layout takes when its variable-count binding has
 * variable_count elements (variable_count is ignored by a layout without
 * one). */
static inline uint32_t
mali_desc_set_slots(const struct mali_descriptor_set_layout *l, uint32_t variable_count)
{
   if (!l->has_variable_count)
      return l->desc_count;
   const struct mali_descriptor_set_binding_layout *b = &l->bindings[l->variable_binding];
   assert(variable_count <= b->desc_count);
   return l->desc_count - (b->desc_count - variable_count) * mali_desc_stride(b);
}

/* Slot of the layout's first sampler (of its lowest sampler or combined
 * image sampler binding), -1 if it has none: what a texel fetch reads in a
 * v7 stage that reads the set as its tables (mali_nir_lower_descriptors_v7.c
 * choose_direct_set). */
static inline int
mali_desc_set_first_sampler_slot(const struct mali_descriptor_set_layout *l)
{
   for (uint32_t i = 0; i < l->binding_count; i++) {
      const struct mali_descriptor_set_binding_layout *b = &l->bindings[i];
      if (b->desc_count && (b->type == VK_DESCRIPTOR_TYPE_SAMPLER ||
                            b->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER))
         return mali_desc_index(b, 0, MALI_SUBDESC_SAMPLER);
   }
   return -1;
}

/* Slots an inline uniform block of size bytes takes. */
static inline uint32_t
mali_iub_desc_count(uint32_t size)
{
   return DIV_ROUND_UP(size, MALI_DESCRIPTOR_SIZE) + 1;
}

#endif
