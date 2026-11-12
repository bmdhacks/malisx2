/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Descriptor pools, descriptor sets, vkUpdateDescriptorSets, buffer views,
 * and the packers the command buffer uses to bind sets
 * (mali_descriptor_set.h). Follows panvk's v9+ descriptor sets
 * (panvk_vX_descriptor_set.c, panvk_vX_buffer_view.c, MIT) for what goes
 * into each slot, because Mesa's compiler reads the slots; pool memory
 * and accounting follow the blob.
 *
 * Built per arch. The one v9/v11 difference: v9's Buffer descriptor has
 * no size-hi word (v11 word 5), so buffer sizes are 32 bits there; the
 * T820 blob writes the same v9 fields.
 *
 * v7 (Bifrost) keeps the slot layout with panvk_v7's contents
 * (panvk_vX_descriptor_set.c at PAN_ARCH < 9): a Uniform Buffer
 * descriptor at the start of a uniform buffer's slot, an address and size
 * for a storage buffer, attribute descriptors for a storage image
 * (mali_image_view.c) or a buffer view (below), zero for an unused slot.
 * Draws and dispatches copy the slots a shader uses, storage buffers
 * aside, into per-type tables (mali_jm_desc_v7.c): on the CPU, when the
 * command is recorded, from a host copy of the set that v7 keeps next to
 * the set memory (set_store), or in a copy job when the command executes.
 */

#ifndef PAN_ARCH
#error "mali_descriptor_set.c is built per arch: PAN_ARCH must be set"
#endif

#include "mali_vk.h"
#include "mali_arch.h"
#include "mali_descriptor_set.h"
#include "mali_image.h"
#include "mali_memory.h"

#include <string.h>

#include "genxml/gen_macros.h"
#include "pan_format.h"

#include "util/format/u_format.h"
#include "util/u_math.h"
#include "vk_alloc.h"
#include "vk_descriptors.h"
#include "vk_descriptor_update_template.h"
#include "vk_format.h"
#include "vk_log.h"
#include "vk_util.h"

static_assert(sizeof(struct mali_hw_desc) == MALI_DESCRIPTOR_SIZE, "descriptor size");
static_assert(pan_size(TEXTURE) == MALI_DESCRIPTOR_SIZE, "descriptor size");
#if PAN_ARCH >= 9
static_assert(pan_size(BUFFER) == MALI_DESCRIPTOR_SIZE, "descriptor size");
static_assert(pan_size(NULL_DESCRIPTOR) == MALI_DESCRIPTOR_SIZE, "descriptor size");
static_assert(pan_size(RESOURCE) == MALI_RESOURCE_SIZE, "descriptor size");
#endif

/* Set memory: CPU read/write, GPU read-only, CPU-uncached, SAME_VA, class
 * DESCRIPTOR_POOL (the blob's pool B). The CPU never reads a set back, so
 * the blob's cached variant for sets with dynamic bindings is not
 * needed. */
#define MALI_DESC_POOL_FLAGS (KB_MEM_PROT_CPU_RD | KB_MEM_PROT_CPU_WR | KB_MEM_PROT_GPU_RD)

/* ---------------------------------------------------------------------- */
/* Buffer views                                                            */

/* A Buffer descriptor of type Structure over whole texels
 * (pan_buffer_texture_emit), or on v7 a 1D Attribute Buffer and an
 * Attribute (below). The Buffer's conversion word is read by shader code,
 * not the hardware, so it comes from Mesa's format table like the
 * compiler's expectations. */
VKAPI_ATTR VkResult VKAPI_CALL
MALI_PER_ARCH(CreateBufferView)(VkDevice _device, const VkBufferViewCreateInfo *pCreateInfo,
                                const VkAllocationCallbacks *pAllocator, VkBufferView *pView)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   VK_FROM_HANDLE(mali_buffer, buffer, pCreateInfo->buffer);

   struct mali_buffer_view *view =
      vk_buffer_view_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*view));
   if (!view)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

#if PAN_ARCH < 9
   /*
    * Bifrost reads a texel buffer as an image: a 1D Attribute Buffer in
    * bytes 0-15 of the slot and its Attribute in 16-23, as the descriptor
    * copy takes them (mali_jm_desc_v7.c set_to_table_img_copy; Mesa
    * pan_buffer.c pan_buffer_texture_emit at PAN_ARCH 6..7). The pointer
    * is 64-byte aligned, so the address's low 6 bits go into the
    * Attribute's offset.
    */
   const enum pipe_format pfmt = vk_format_to_pipe_format(view->vk.format);
   const unsigned stride = util_format_get_blocksize(pfmt);
   const uint64_t addr = mali_buffer_gpu_va(buffer, pCreateInfo->offset);
   const uint64_t base = addr & ~0x3full;
   const uint32_t offset = addr & 0x3f;
   /* The size field is 32 bits: a view that does not fit keeps the whole
    * texels that do, as on v9. Only maxTexelBufferElements (2^28) texels
    * of 16 bytes reach 4 GiB. */
   const uint64_t texels = MIN2(view->vk.elements, (UINT32_MAX - offset) / stride);

   struct {
      struct mali_attribute_buffer_packed buf;
      struct mali_attribute_packed attrib;
      uint32_t pad[2];
   } desc = {0};
   static_assert(sizeof(desc) == MALI_DESCRIPTOR_SIZE, "one slot");
   pan_pack(&desc.buf, ATTRIBUTE_BUFFER, cfg) {
      cfg.type = MALI_ATTRIBUTE_TYPE_1D;
      cfg.pointer = base;
      cfg.stride = stride;
      cfg.size = (uint32_t)(texels * stride + offset);
   }
   pan_pack(&desc.attrib, ATTRIBUTE, cfg) {
      cfg.format = GENX(pan_format_from_pipe_format)(pfmt)->hw;
      cfg.offset = offset;
      cfg.offset_enable = offset != 0;
   }
   memcpy(&view->desc, &desc, sizeof(desc));

   *pView = mali_buffer_view_to_handle(view);
   return VK_SUCCESS;
#else
   const enum pipe_format pfmt = vk_format_to_pipe_format(view->vk.format);
   const unsigned stride = util_format_get_blocksize(pfmt);
   const uint64_t size = view->vk.elements * stride;

   struct mali_buffer_packed desc;
   pan_pack(&desc, BUFFER, cfg) {
      cfg.buffer_type = MALI_BUFFER_TYPE_STRUCTURE;
#if PAN_ARCH >= 11
      cfg.size = size & BITFIELD_MASK(32);
      cfg.size_hi = size >> 32;
#else
      /* 32 bits of size on v9 and v10: whole texels below 4 GiB. */
      cfg.size = MIN2(size, (UINT32_MAX / stride) * (uint64_t)stride);
#endif
      cfg.address = mali_buffer_gpu_va(buffer, pCreateInfo->offset);
      cfg.stride = stride;
      cfg.conversion.memory_format = GENX(pan_format_from_pipe_format)(pfmt)->hw;
      cfg.conversion.raw = false;
   }
   memcpy(&view->desc, &desc, sizeof(desc));

   *pView = mali_buffer_view_to_handle(view);
   return VK_SUCCESS;
#endif
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(DestroyBufferView)(VkDevice _device, VkBufferView _view,
                                 const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   VK_FROM_HANDLE(mali_buffer_view, view, _view);

   if (view)
      vk_buffer_view_destroy(&dev->vk, pAllocator, &view->vk);
}

/* ---------------------------------------------------------------------- */
/* Writing slots                                                           */

static void *
slot_ptr(const struct mali_descriptor_set *set, uint32_t binding, uint32_t elem,
         enum mali_subdesc sub)
{
   const struct mali_descriptor_set_binding_layout *bl = &set->layout->bindings[binding];
   const uint32_t idx = mali_desc_index(bl, elem, sub);
   assert(idx < set->desc_count);
   return (uint8_t *)set->cpu + (size_t)idx * MALI_DESCRIPTOR_SIZE;
}

/* Store size bytes at dst, a pointer into set->cpu, and on v7 the same
 * bytes into the host copy. v9 and v11 have no host copy. */
static void
set_store(const struct mali_descriptor_set *set, void *dst, const void *src, size_t size)
{
#if PAN_ARCH < 9
   if (set->host && set->host != set->cpu)
      memmove((uint8_t *)set->host + ((uint8_t *)dst - (uint8_t *)set->cpu), src, size);
#endif
   memmove(dst, src, size);
}

/* Where bytes at p, a pointer into set->cpu, are best read: the host
 * copy when there is one (set memory is CPU-uncached). */
static const void *
set_load_ptr(const struct mali_descriptor_set *set, const void *p)
{
#if PAN_ARCH < 9
   if (set->host)
      return (const uint8_t *)set->host + ((const uint8_t *)p - (const uint8_t *)set->cpu);
#endif
   return p;
}

static void
write_slot(const struct mali_descriptor_set *set, uint32_t binding, uint32_t elem,
           enum mali_subdesc sub, const void *desc)
{
   set_store(set, slot_ptr(set, binding, elem, sub), desc, MALI_DESCRIPTOR_SIZE);
}

static void
write_null(const struct mali_descriptor_set *set, uint32_t binding, uint32_t elem,
           enum mali_subdesc sub)
{
#if PAN_ARCH >= 9
   struct mali_null_descriptor_packed null_desc;
   pan_pack(&null_desc, NULL_DESCRIPTOR, cfg);
   write_slot(set, binding, elem, sub, &null_desc);
#else
   /* No Null descriptor on Bifrost (panvk_v7 writes none either): the slot
    * is zeroed. Reading through it is not robust, and nullDescriptor is
    * not exposed. */
   const uint8_t null_desc[MALI_DESCRIPTOR_SIZE] = {0};
   write_slot(set, binding, elem, sub, null_desc);
#endif
}

static enum mali_subdesc
tex_subdesc(VkDescriptorType type)
{
   return type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ? MALI_SUBDESC_TEXTURE :
                                                              MALI_SUBDESC_NONE;
}

static enum mali_subdesc
sampler_subdesc(VkDescriptorType type)
{
   return type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ? MALI_SUBDESC_SAMPLER :
                                                              MALI_SUBDESC_NONE;
}

/* Samplers of a binding with immutable samplers are written at allocation
 * and ignored by updates (the spec's rule). */
static void
write_sampler(const struct mali_descriptor_set *set, const VkDescriptorImageInfo *info,
              uint32_t binding, uint32_t elem)
{
   const struct mali_descriptor_set_binding_layout *bl = &set->layout->bindings[binding];
   if (bl->immutable_samplers)
      return;

   VK_FROM_HANDLE(mali_sampler, sampler, info->sampler);
   if (sampler)
      write_slot(set, binding, elem, sampler_subdesc(bl->type), &sampler->desc);
   else
      write_null(set, binding, elem, sampler_subdesc(bl->type));
}

static void
write_image(const struct mali_descriptor_set *set, const VkDescriptorImageInfo *info,
            uint32_t binding, uint32_t elem, VkDescriptorType type)
{
   VK_FROM_HANDLE(mali_image_view, view, info->imageView);
   const enum mali_subdesc sub = tex_subdesc(type);

   if (view && type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE && view->has_storage_tex)
      write_slot(set, binding, elem, sub, view->storage_tex);
   else if (view && type != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE && view->has_tex)
      write_slot(set, binding, elem, sub, view->tex);
   else
      write_null(set, binding, elem, sub);
}

/* Sizes are rounded up to 16 bytes (uniform) or 4 (storage), as panvk:
 * the compiler's bounds checks assume it (drift doc, "Buffer, Simple").
 * v9 has 32 bits of size: a larger range is clamped to the largest size
 * the rounding allows (maxStorageBufferRange is below 4 GiB there). */
static void
pack_buffer(uint64_t addr, uint64_t range, bool ssbo, void *out)
{
#if PAN_ARCH < 9
   /* panvk_v7 write_buffer_desc: a storage buffer is the address and size
    * pair the shader reads from the set (struct mali_v7_ssbo_addr), a
    * uniform buffer a Uniform Buffer descriptor at the start of a zeroed
    * slot. */
   if (ssbo) {
      const struct mali_v7_ssbo_addr entry = mali_v7_ssbo_addr(addr, range);
      memcpy(out, &entry, sizeof(entry));
   } else {
      memset(out, 0, MALI_DESCRIPTOR_SIZE);
      pan_cast_and_pack(out, UNIFORM_BUFFER, cfg) {
         cfg.pointer = addr;
         cfg.entries = mali_v7_ubo_entries(range);
      }
   }
#else
   const uint64_t size = align64(range, ssbo ? 4 : 16);
   pan_cast_and_pack(out, BUFFER, cfg) {
      cfg.address = addr;
#if PAN_ARCH >= 11
      cfg.size = size & BITFIELD_MASK(32);
      cfg.size_hi = size >> 32;
#else
      /* 32 bits of size on v9 and v10. */
      cfg.size = MIN2(size, UINT32_MAX & ~(uint64_t)(ssbo ? 3 : 15));
#endif
   }
#endif
}

static void
write_buffer(const struct mali_descriptor_set *set, const VkDescriptorBufferInfo *info,
             uint32_t binding, uint32_t elem, VkDescriptorType type)
{
   VK_FROM_HANDLE(mali_buffer, buffer, info->buffer);
   if (!buffer) {
      write_null(set, binding, elem, MALI_SUBDESC_NONE);
      return;
   }

   struct mali_hw_desc desc;
   pack_buffer(mali_buffer_gpu_va(buffer, info->offset),
               vk_buffer_range(&buffer->vk, info->offset, info->range),
               type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &desc);
   write_slot(set, binding, elem, MALI_SUBDESC_NONE, &desc);
}

static void
write_dynamic_buffer(struct mali_descriptor_set *set, const VkDescriptorBufferInfo *info,
                     uint32_t binding, uint32_t elem)
{
   const uint32_t idx = set->layout->bindings[binding].desc_idx + elem;
   VK_FROM_HANDLE(mali_buffer, buffer, info->buffer);

   assert(idx < set->layout->dyn_buf_count);
   if (buffer) {
      set->dyn_bufs[idx].addr = mali_buffer_gpu_va(buffer, info->offset);
      set->dyn_bufs[idx].range = vk_buffer_range(&buffer->vk, info->offset, info->range);
   } else {
      set->dyn_bufs[idx].addr = MALI_NULL_BUFFER_ADDR;
      set->dyn_bufs[idx].range = 0;
   }
}

static void
write_buffer_view(const struct mali_descriptor_set *set, VkBufferView _view,
                  uint32_t binding, uint32_t elem)
{
   VK_FROM_HANDLE(mali_buffer_view, view, _view);
   if (view)
      write_slot(set, binding, elem, MALI_SUBDESC_NONE, &view->desc);
   else
      write_null(set, binding, elem, MALI_SUBDESC_NONE);
}

/* Inline uniform block bytes start in the binding's second slot; the first
 * holds the Buffer descriptor pointing at them (panvk). */
static uint8_t *
iub_data(const struct mali_descriptor_set *set, uint32_t binding)
{
   return (uint8_t *)slot_ptr(set, binding, 0, MALI_SUBDESC_NONE) + MALI_DESCRIPTOR_SIZE;
}

static void
init_iub(const struct mali_descriptor_set *set, uint32_t binding)
{
   const struct mali_descriptor_set_binding_layout *bl = &set->layout->bindings[binding];
   const uint32_t first = mali_desc_index(bl, 0, MALI_SUBDESC_NONE);
   struct mali_hw_desc desc;

   pack_buffer(set->gpu + (uint64_t)(first + 1) * MALI_DESCRIPTOR_SIZE,
               (uint64_t)(bl->desc_count - 1) * MALI_DESCRIPTOR_SIZE, false, &desc);
   write_slot(set, binding, 0, MALI_SUBDESC_NONE, &desc);
}

/* The slots a set's own layout fixes regardless of what the application
 * writes: an inline uniform block's self-referencing Buffer descriptor,
 * and a binding's immutable samplers. Idempotent, so a caller that is not
 * sure whether it ran already (a push descriptor set reusing its
 * container after a fresh allocation) can just call it again. */
static void
init_fixed_slots(struct mali_descriptor_set *set)
{
   const struct mali_descriptor_set_layout *layout = set->layout;

   for (uint32_t b = 0; b < layout->binding_count; b++) {
      const struct mali_descriptor_set_binding_layout *bl = &layout->bindings[b];
      if (bl->type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK && bl->desc_count) {
         init_iub(set, b);
      } else if (bl->immutable_samplers) {
         /* A variable-count binding has only the elements the set was
          * allocated with. */
         const uint32_t count =
            layout->has_variable_count && b == layout->variable_binding ? set->variable_count :
                                                                          bl->desc_count;
         for (uint32_t e = 0; e < count; e++)
            write_slot(set, b, e, sampler_subdesc(bl->type), &bl->immutable_samplers[e]);
      }
   }
}

static void
write_descriptors(struct mali_descriptor_set *set, const VkWriteDescriptorSet *w)
{
   if (w->descriptorType == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK) {
      /* dstArrayElement and descriptorCount are byte offset and size. */
      const VkWriteDescriptorSetInlineUniformBlock *iub =
         vk_find_struct_const(w->pNext, WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK);
      if (iub)
         set_store(set, iub_data(set, w->dstBinding) + w->dstArrayElement, iub->pData,
                   w->descriptorCount);
      return;
   }

   for (uint32_t j = 0; j < w->descriptorCount; j++) {
      const uint32_t elem = w->dstArrayElement + j;
      switch (w->descriptorType) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
         write_sampler(set, &w->pImageInfo[j], w->dstBinding, elem);
         break;
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
         write_image(set, &w->pImageInfo[j], w->dstBinding, elem, w->descriptorType);
         write_sampler(set, &w->pImageInfo[j], w->dstBinding, elem);
         break;
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
         write_image(set, &w->pImageInfo[j], w->dstBinding, elem, w->descriptorType);
         break;
      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         write_buffer_view(set, w->pTexelBufferView[j], w->dstBinding, elem);
         break;
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
         write_buffer(set, &w->pBufferInfo[j], w->dstBinding, elem, w->descriptorType);
         break;
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
         write_dynamic_buffer(set, &w->pBufferInfo[j], w->dstBinding, elem);
         break;
      default:
         UNREACHABLE("descriptor type refused by the set layout");
      }
   }
}

static void
copy_descriptors(const VkCopyDescriptorSet *c)
{
   VK_FROM_HANDLE(mali_descriptor_set, src, c->srcSet);
   VK_FROM_HANDLE(mali_descriptor_set, dst, c->dstSet);
   const struct mali_descriptor_set_binding_layout *sbl = &src->layout->bindings[c->srcBinding];
   const struct mali_descriptor_set_binding_layout *dbl = &dst->layout->bindings[c->dstBinding];

   assert(sbl->type == dbl->type);

   switch (sbl->type) {
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
      memmove(&dst->dyn_bufs[dbl->desc_idx + c->dstArrayElement],
              &src->dyn_bufs[sbl->desc_idx + c->srcArrayElement],
              c->descriptorCount * sizeof(dst->dyn_bufs[0]));
      break;
   case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK:
      set_store(dst, iub_data(dst, c->dstBinding) + c->dstArrayElement,
                set_load_ptr(src, iub_data(src, c->srcBinding) + c->srcArrayElement),
                c->descriptorCount);
      break;
   default: {
      /* Whole elements (both halves of a combined image/sampler). A
       * destination binding with immutable samplers keeps its own. */
      const uint32_t stride = mali_desc_stride(sbl);
      for (uint32_t i = 0; i < c->descriptorCount; i++) {
         const uint8_t *s = slot_ptr(src, c->srcBinding, c->srcArrayElement + i,
                                     MALI_SUBDESC_NONE);
         uint8_t *d = slot_ptr(dst, c->dstBinding, c->dstArrayElement + i,
                               MALI_SUBDESC_NONE);
         uint32_t n = stride;
         if (dbl->immutable_samplers && dbl->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
            n = dbl->textures_per_desc;
         else if (dbl->immutable_samplers)
            n = 0;
         set_store(dst, d, set_load_ptr(src, s), (size_t)n * MALI_DESCRIPTOR_SIZE);
      }
      break;
   }
   }
}

/* Set memory is CPU-uncached: the GPU sees these writes without any cache
 * maintenance. On v9 and v11 it reads the set in place when a command
 * buffer that bound it executes, so a write after the bind is seen, as
 * with the blob. On v7 a draw's tables are filled when it is recorded,
 * from the set as it is then, unless the set's layout is update-after-bind
 * (mali_jm_desc_v7.c): a write after the bind is seen there only for such
 * a set, as the specification allows for no other. */
VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(UpdateDescriptorSets)(VkDevice _device, uint32_t descriptorWriteCount,
                                    const VkWriteDescriptorSet *pDescriptorWrites,
                                    uint32_t descriptorCopyCount,
                                    const VkCopyDescriptorSet *pDescriptorCopies)
{
   for (uint32_t i = 0; i < descriptorWriteCount; i++) {
      VK_FROM_HANDLE(mali_descriptor_set, set, pDescriptorWrites[i].dstSet);
      write_descriptors(set, &pDescriptorWrites[i]);
   }
   for (uint32_t i = 0; i < descriptorCopyCount; i++)
      copy_descriptors(&pDescriptorCopies[i]);
}

/*
 * Template updates (Granite, i.e. parallel-RDP, writes every set this way).
 * Each element is written alone: the template's stride need not match the
 * Vulkan info structs', as in CmdPushDescriptorSetWithTemplate2KHR
 * (mali_cmd_state.c). An inline uniform block entry is one byte range.
 */
VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(UpdateDescriptorSetWithTemplate)(VkDevice _device, VkDescriptorSet descriptorSet,
                                               VkDescriptorUpdateTemplate descriptorUpdateTemplate,
                                               const void *pData)
{
   VK_FROM_HANDLE(mali_descriptor_set, set, descriptorSet);
   VK_FROM_HANDLE(vk_descriptor_update_template, template, descriptorUpdateTemplate);

   for (uint32_t i = 0; i < template->entry_count; i++) {
      const struct vk_descriptor_template_entry *e = &template->entries[i];
      const uint8_t *base = (const uint8_t *)pData + e->offset;

      if (e->type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK) {
         const VkWriteDescriptorSetInlineUniformBlock iub = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK,
            .dataSize = e->array_count,
            .pData = base,
         };
         const VkWriteDescriptorSet w = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = &iub,
            .dstBinding = e->binding,
            .dstArrayElement = e->array_element,
            .descriptorCount = e->array_count,
            .descriptorType = e->type,
         };
         write_descriptors(set, &w);
         continue;
      }

      for (uint32_t j = 0; j < e->array_count; j++) {
         const void *src = base + (size_t)j * e->stride;
         VkWriteDescriptorSet w = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding = e->binding,
            .dstArrayElement = e->array_element + j,
            .descriptorCount = 1,
            .descriptorType = e->type,
         };
         switch (e->type) {
         case VK_DESCRIPTOR_TYPE_SAMPLER:
         case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
         case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
         case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
            w.pImageInfo = src;
            break;
         case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
         case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
            w.pTexelBufferView = src;
            break;
         default:
            w.pBufferInfo = src;
            break;
         }
         write_descriptors(set, &w);
      }
   }
}

/* ---------------------------------------------------------------------- */
/* Pools and sets                                                          */

/* Descriptors of each type a set of layout takes from its pool when its
 * variable-count binding has variable_count elements: the layout's counts
 * (which have that binding at its full size) themselves, or those less the
 * elements the set does not have, built in scratch. */
static const uint32_t *
set_type_counts(const struct mali_descriptor_set_layout *layout, uint32_t variable_count,
                uint32_t scratch[MALI_DESC_TYPE_COUNT])
{
   if (!layout->has_variable_count)
      return layout->type_counts;
   const struct mali_descriptor_set_binding_layout *b = &layout->bindings[layout->variable_binding];
   memcpy(scratch, layout->type_counts, sizeof(layout->type_counts));
   scratch[mali_desc_type_index(b->type)] -= b->desc_count - variable_count;
   return scratch;
}

static void
pool_release_set(struct mali_device *dev, struct mali_descriptor_pool *pool,
                 struct mali_descriptor_set *set, bool heap_free)
{
   const uint32_t i = set - pool->sets;
   assert(i < pool->max_sets);
   if (BITSET_TEST(pool->free_sets, i))
      return;

   struct mali_descriptor_set_layout *layout =
      (struct mali_descriptor_set_layout *)set->layout;
   uint32_t type_counts[MALI_DESC_TYPE_COUNT];
   if (set->gpu && heap_free)
      util_vma_heap_free(&pool->heap, set->gpu, mali_descriptor_set_size(set));
   const uint32_t *counts = set_type_counts(layout, set->variable_count, type_counts);
   for (unsigned t = 0; t < MALI_DESC_TYPE_COUNT; t++)
      pool->avail[t] += counts[t];

   if (set->own_layout_ref)
      vk_descriptor_set_layout_unref(&dev->vk, &layout->vk);
   vk_object_base_finish(&set->base);
   memset(set, 0, sizeof(*set));
   BITSET_SET(pool->free_sets, i);
   pool->live_sets--;
}

static void
pool_free_set(struct mali_device *dev, struct mali_descriptor_pool *pool,
              struct mali_descriptor_set *set)
{
   /* A linear pool's set memory is not the heap's: back at the reset. */
   pool_release_set(dev, pool, set, !pool->linear);
}

/* Every set freed; the set memory heap starts over whole instead of taking
 * the sets back one by one. */
static void
pool_reset(struct mali_device *dev, struct mali_descriptor_pool *pool)
{
   if (pool->linear) {
      /* Only sets[0, next) were ever taken. */
      for (uint32_t i = 0; i < pool->next && pool->live_sets; i++)
         pool_release_set(dev, pool, &pool->sets[i], false);
      assert(pool->live_sets == 0);
      pool->next = 0;
      pool->bump = 0;
      return;
   }
   if (!pool->live_sets)
      return;
   for (uint32_t i = 0; i < pool->max_sets && pool->live_sets; i++)
      pool_release_set(dev, pool, &pool->sets[i], false);
   assert(pool->live_sets == 0);
   if (pool->bo.gpu_va) {
      util_vma_heap_finish(&pool->heap);
      util_vma_heap_init(&pool->heap, pool->bo.gpu_va, pool->heap_size);
   }
}

static void
pool_destroy(struct mali_device *dev, const VkAllocationCallbacks *alloc,
             struct mali_descriptor_pool *pool)
{
   pool_reset(dev, pool);
   for (uint32_t i = 0; i < pool->held_count; i++)
      vk_descriptor_set_layout_unref(&dev->vk, &pool->held[i]->vk);
   pool->held_count = 0;
   if (pool->bo.gpu_va) {
      util_vma_heap_finish(&pool->heap);
      mali_kbase_free(dev->kbase, &pool->bo);
   }
   vk_free2(&dev->vk.alloc, alloc, pool->host);
   vk_object_free(&dev->vk, alloc, pool);
}

VKAPI_ATTR VkResult VKAPI_CALL
MALI_PER_ARCH(CreateDescriptorPool)(VkDevice _device, const VkDescriptorPoolCreateInfo *pCreateInfo,
                                    const VkAllocationCallbacks *pAllocator,
                                    VkDescriptorPool *pDescriptorPool)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   const uint32_t max_sets = pCreateInfo->maxSets;

   VK_MULTIALLOC(ma);
   VK_MULTIALLOC_DECL(&ma, struct mali_descriptor_pool, pool, 1);
   VK_MULTIALLOC_DECL(&ma, BITSET_WORD, free_sets, BITSET_WORDS(max_sets));
   VK_MULTIALLOC_DECL(&ma, struct mali_descriptor_set, sets, max_sets);
   if (!vk_object_multizalloc(&dev->vk, &ma, pAllocator, VK_OBJECT_TYPE_DESCRIPTOR_POOL))
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   pool->max_sets = max_sets;
   pool->sets = sets;
   pool->free_sets = free_sets;
   BITSET_SET_COUNT(free_sets, 0, max_sets);
   pool->linear = !(pCreateInfo->flags & VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

   /* Slots: one per descriptor, two per combined image/sampler; an inline
    * uniform block's bytes plus one Buffer slot per block. Dynamic buffers
    * take no memory. */
   uint64_t slots = 0;
   for (uint32_t i = 0; i < pCreateInfo->poolSizeCount; i++) {
      const VkDescriptorPoolSize *ps = &pCreateInfo->pPoolSizes[i];
      const unsigned t = mali_desc_type_index(ps->type);
      if (t >= MALI_DESC_TYPE_COUNT)
         continue;   /* a type no layout can have */
      pool->size[t] += ps->descriptorCount;

      if (vk_descriptor_type_is_dynamic(ps->type))
         continue;
      if (ps->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
         slots += 2ull * ps->descriptorCount;
      else if (ps->type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK)
         slots += DIV_ROUND_UP(ps->descriptorCount, MALI_DESCRIPTOR_SIZE);
      else
         slots += ps->descriptorCount;
   }
   if (pool->size[MALI_DESC_TYPE_IUB]) {
      const VkDescriptorPoolInlineUniformBlockCreateInfo *iub =
         vk_find_struct_const(pCreateInfo->pNext,
                              DESCRIPTOR_POOL_INLINE_UNIFORM_BLOCK_CREATE_INFO);
      /* Per block: its Buffer slot and the rounding of its bytes. */
      slots += 2ull * (iub ? iub->maxInlineUniformBlockBindings : max_sets);
   }
   memcpy(pool->avail, pool->size, sizeof(pool->avail));

   if (slots) {
      const struct mali_kbase_alloc_info ai = {
         .size = slots * MALI_DESCRIPTOR_SIZE,
         .flags = mali_kbase_mem_same_va_policy(MALI_DESC_POOL_FLAGS),
         .mem_class = MALI_KBASE_MEM_CLASS_DESCRIPTOR_POOL,
      };
      enum mali_kbase_result r = mali_kbase_alloc(dev->kbase, &ai, &pool->bo);
      if (r != MALI_KBASE_SUCCESS) {
         vk_object_free(&dev->vk, pAllocator, pool);
         return vk_errorf(dev, r == MALI_KBASE_ERROR_OUT_OF_HOST_MEMORY ?
                                  VK_ERROR_OUT_OF_HOST_MEMORY :
                                  VK_ERROR_OUT_OF_DEVICE_MEMORY,
                          "descriptor pool of %llu bytes: %s",
                          (unsigned long long)ai.size, mali_kbase_result_str(r));
      }
      pool->heap_size = slots * MALI_DESCRIPTOR_SIZE;
      util_vma_heap_init(&pool->heap, pool->bo.gpu_va, pool->heap_size);
#if PAN_ARCH < 9
      /* The host copy v7 fills its draws' tables from. None for an
       * update-after-bind pool: its sets are read by the GPU as they are
       * when the draw executes, so the tables of a draw are copied by a
       * job there (mali_jm_desc_v7.c). */
      if (!(pCreateInfo->flags & VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT)) {
         pool->host = vk_zalloc2(&dev->vk.alloc, pAllocator, pool->heap_size, 8,
                                 VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
         if (!pool->host) {
            pool_destroy(dev, pAllocator, pool);
            return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
         }
      }
#endif
   }

   *pDescriptorPool = mali_descriptor_pool_to_handle(pool);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(DestroyDescriptorPool)(VkDevice _device, VkDescriptorPool _pool,
                                     const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   VK_FROM_HANDLE(mali_descriptor_pool, pool, _pool);

   if (pool)
      pool_destroy(dev, pAllocator, pool);
}

VKAPI_ATTR VkResult VKAPI_CALL
MALI_PER_ARCH(ResetDescriptorPool)(VkDevice _device, VkDescriptorPool _pool,
                                   VkDescriptorPoolResetFlags flags)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   VK_FROM_HANDLE(mali_descriptor_pool, pool, _pool);

   pool_reset(dev, pool);
   return VK_SUCCESS;
}

/*
 * One set, with the blob's checks in its order: the pool's set count,
 * then each type's remaining descriptors (both
 * VK_ERROR_OUT_OF_POOL_MEMORY), then memory
 * (VK_ERROR_OUT_OF_POOL_MEMORY when too little is free in total,
 * VK_ERROR_FRAGMENTED_POOL when it is free but not contiguous). A layout's
 * variable-count binding takes the pool descriptors and the set memory of
 * the count the set is allocated with, not of its largest. Set memory is
 * not cleared (neither the blob nor panvk does); immutable samplers and
 * inline-uniform-block headers are written.
 */
/* Takes the pool's reference to a layout, once per layout; false when the
 * pool already holds as many as it keeps. */
static bool
pool_hold_layout(struct mali_descriptor_pool *pool, struct mali_descriptor_set_layout *layout)
{
   for (uint32_t i = 0; i < pool->held_count; i++) {
      if (pool->held[i] == layout)
         return true;
   }
   if (pool->held_count == ARRAY_SIZE(pool->held))
      return false;
   vk_descriptor_set_layout_ref(&layout->vk);
   pool->held[pool->held_count++] = layout;
   return true;
}

static VkResult
pool_alloc_set(struct mali_device *dev, struct mali_descriptor_pool *pool,
               struct mali_descriptor_set_layout *layout, uint32_t variable_count,
               struct mali_descriptor_set **out)
{
   if (pool->live_sets >= pool->max_sets)
      return VK_ERROR_OUT_OF_POOL_MEMORY;
   uint32_t type_counts[MALI_DESC_TYPE_COUNT];
   const uint32_t *counts = set_type_counts(layout, variable_count, type_counts);
   for (unsigned t = 0; t < MALI_DESC_TYPE_COUNT; t++) {
      if (counts[t] > pool->avail[t])
         return VK_ERROR_OUT_OF_POOL_MEMORY;
   }

   const uint32_t slots = mali_desc_set_slots(layout, variable_count);
   const uint64_t size = (uint64_t)slots * MALI_DESCRIPTOR_SIZE;
   uint64_t gpu = 0;
   if (pool->linear) {
      /* In order: no heap, no search (mali_descriptor_pool::linear). */
      if (pool->next >= pool->max_sets)
         return VK_ERROR_OUT_OF_POOL_MEMORY;
      if (size) {
         if (!pool->bo.gpu_va || pool->bump + size > pool->heap_size)
            return VK_ERROR_OUT_OF_POOL_MEMORY;
         gpu = pool->bo.gpu_va + pool->bump;
         pool->bump += size;
      }
   } else if (size) {
      if (!pool->bo.gpu_va || pool->heap.free_size < size)
         return VK_ERROR_OUT_OF_POOL_MEMORY;
      gpu = util_vma_heap_alloc(&pool->heap, size, MALI_DESCRIPTOR_SIZE);
      if (!gpu)
         return VK_ERROR_FRAGMENTED_POOL;
   }

   const uint32_t i = pool->linear ? pool->next++
                                   : __bitset_ffs(pool->free_sets, BITSET_WORDS(pool->max_sets)) - 1;
   assert(i < pool->max_sets);
   struct mali_descriptor_set *set = &pool->sets[i];
   BITSET_CLEAR(pool->free_sets, i);
   pool->live_sets++;
   for (unsigned t = 0; t < MALI_DESC_TYPE_COUNT; t++)
      pool->avail[t] -= counts[t];

   vk_object_base_init(&dev->vk, &set->base, VK_OBJECT_TYPE_DESCRIPTOR_SET);
   set->own_layout_ref = !pool_hold_layout(pool, layout);
   if (set->own_layout_ref)
      vk_descriptor_set_layout_ref(&layout->vk);
   set->pool = pool;
   set->layout = layout;
   set->desc_count = slots;
   set->variable_count = layout->has_variable_count ? variable_count : 0;
   set->gpu = gpu;
   set->cpu = gpu ? (uint8_t *)pool->bo.cpu + (gpu - pool->bo.gpu_va) : NULL;
   set->host = gpu && pool->host ? pool->host + (gpu - pool->bo.gpu_va) : NULL;

#if PAN_ARCH < 9
   /* A texel fetch in a v7 stage that reads this set as its tables reads
    * the set's first sampler slot (mali_desc_set_first_sampler_slot): a
    * sampler until the application writes its own there, else the GPU
    * faults on whatever the slot holds. Only a set laid out for update
    * after bind can be read that way. Immutable samplers, written next,
    * replace it. */
   if (gpu && (layout->flags & VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT)) {
      const int fetch_sampler = mali_desc_set_first_sampler_slot(layout);
      if (fetch_sampler >= 0) {
         uint8_t dummy[MALI_DESCRIPTOR_SIZE];
         MALI_PER_ARCH(pack_dummy_sampler)(dummy);
         set_store(set, (uint8_t *)set->cpu + (size_t)fetch_sampler * MALI_DESCRIPTOR_SIZE, dummy,
                   sizeof(dummy));
      }
   }
#endif
   init_fixed_slots(set);

   *out = set;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
MALI_PER_ARCH(AllocateDescriptorSets)(VkDevice _device, const VkDescriptorSetAllocateInfo *pAllocateInfo,
                                      VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   VK_FROM_HANDLE(mali_descriptor_pool, pool, pAllocateInfo->descriptorPool);
   VkResult result = VK_SUCCESS;
   uint32_t i;
   /* A linear pool's position, put back if this call fails. */
   const uint32_t next = pool->next;
   const uint64_t bump = pool->bump;
   /* Without the structure, or with a count of zero sets in it, the
    * variable counts are zero. */
   const VkDescriptorSetVariableDescriptorCountAllocateInfo *var_counts =
      vk_find_struct_const(pAllocateInfo->pNext,
                           DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO);

   for (i = 0; i < pAllocateInfo->descriptorSetCount; i++) {
      VK_FROM_HANDLE(mali_descriptor_set_layout, layout, pAllocateInfo->pSetLayouts[i]);
      struct mali_descriptor_set *set;
      uint32_t variable_count = 0;
      if (layout->has_variable_count && var_counts && var_counts->descriptorSetCount > 0) {
         /* The specification keeps it within the binding's size; a larger
          * value would run past the set's memory, so it is cut. */
         variable_count = MIN2(var_counts->pDescriptorCounts[i],
                               layout->bindings[layout->variable_binding].desc_count);
      }
      result = pool_alloc_set(dev, pool, layout, variable_count, &set);
      if (result != VK_SUCCESS)
         break;
      pDescriptorSets[i] = mali_descriptor_set_to_handle(set);
   }

   if (result == VK_SUCCESS)
      return VK_SUCCESS;

   /* Undo this call's sets; every output is null (the blob does the
    * same). Pool exhaustion is not logged: ARMSX2 grows its chain of
    * pools on it. */
   for (uint32_t j = 0; j < i; j++)
      pool_free_set(dev, pool, mali_descriptor_set_from_handle(pDescriptorSets[j]));
   /* This call's sets were a linear pool's last ones: their slots and
    * memory go back too, not only at the next reset. */
   if (pool->linear) {
      pool->next = next;
      pool->bump = bump;
   }
   for (uint32_t j = 0; j < pAllocateInfo->descriptorSetCount; j++)
      pDescriptorSets[j] = VK_NULL_HANDLE;
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
MALI_PER_ARCH(FreeDescriptorSets)(VkDevice _device, VkDescriptorPool _pool, uint32_t count,
                                  const VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   VK_FROM_HANDLE(mali_descriptor_pool, pool, _pool);

   for (uint32_t i = 0; i < count; i++) {
      VK_FROM_HANDLE(mali_descriptor_set, set, pDescriptorSets[i]);
      if (set)
         pool_free_set(dev, pool, set);
   }
   return VK_SUCCESS;
}

/* ---------------------------------------------------------------------- */
/* Binding (for the command buffer)                                        */

/* The resource table entries, on v9 and v11; v7 has no resource tables
 * (mali_jm_desc_v7.c packs its dynamic buffers). */
#if PAN_ARCH >= 9
void
MALI_PER_ARCH(descriptor_set_pack_resource)(const struct mali_descriptor_set *set, void *out)
{
   pan_cast_and_pack(out, RESOURCE, cfg) {
      if (set && set->gpu) {
         cfg.address = set->gpu;
         cfg.size = mali_descriptor_set_size(set);
         cfg.contains_descriptors = true;
      } else {
         cfg.address = 0;
         cfg.size = 0;
         cfg.contains_descriptors = false;
      }
   }
}

void
MALI_PER_ARCH(descriptor_set_pack_dyn_buf)(const struct mali_descriptor_set *set, uint32_t idx,
                                           uint32_t dynamic_offset, void *out)
{
   assert(idx < set->layout->dyn_buf_count);
   pack_buffer(set->dyn_bufs[idx].addr + dynamic_offset, set->dyn_bufs[idx].range,
               set->layout->dyn_ssbos & BITFIELD_BIT(idx), out);
}
#endif

/* ---------------------------------------------------------------------- */
/* Push descriptors (VK_KHR_push_descriptor): the command buffer           */
/* (mali_cmd_state.c) owns the set's storage and its lifetime; this is     */
/* only the write path, identical to a pool-backed set's.                 */

void
MALI_PER_ARCH(descriptor_set_init_fixed_slots)(struct mali_descriptor_set *set)
{
   init_fixed_slots(set);
}

void
MALI_PER_ARCH(descriptor_set_write)(struct mali_descriptor_set *set, uint32_t write_count,
                                    const VkWriteDescriptorSet *writes)
{
   for (uint32_t i = 0; i < write_count; i++)
      write_descriptors(set, &writes[i]);
}
