/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * vkCreateDescriptorSetLayout (layout rules in mali_descriptor_set_layout.h)
 * and vkGetDescriptorSetLayoutSupport. Destruction is the runtime's
 * reference-counted vk_common_ one.
 */

#include "mali_vk.h"

#include <assert.h>
#include <stdlib.h>

#include "vk_descriptors.h"
#include "vk_alloc.h"
#include "vk_log.h"
#include "vk_util.h"

#include "util/mesa-blake3.h"

#include "mali_descriptor_set.h"
#include "mali_descriptor_set_layout.h"

static bool
has_texture(VkDescriptorType type)
{
   switch (type) {
   case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
   case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      return true;
   default:
      return false;
   }
}

static bool
has_sampler(VkDescriptorType type)
{
   return type == VK_DESCRIPTOR_TYPE_SAMPLER ||
          type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
}

static bool
type_supported(VkDescriptorType type)
{
   switch (type) {
   case VK_DESCRIPTOR_TYPE_SAMPLER:
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
   case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
   case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
   case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
   case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK:
      return true;
   default:
      return false;
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_CreateDescriptorSetLayout(VkDevice _device,
                               const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
                               const VkAllocationCallbacks *pAllocator,
                               VkDescriptorSetLayout *pSetLayout)
{
   VK_FROM_HANDLE(mali_device, dev, _device);
   const VkDescriptorSetLayoutBindingFlagsCreateInfo *flags_info =
      vk_find_struct_const(pCreateInfo->pNext,
                           DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO);
   VkDescriptorSetLayoutBinding *bindings = NULL;
   VkDescriptorBindingFlags *binding_flags = NULL;
   uint32_t binding_count = 0, immutable_count = 0;
   VkResult result;

   for (uint32_t i = 0; i < pCreateInfo->bindingCount; i++) {
      const VkDescriptorSetLayoutBinding *b = &pCreateInfo->pBindings[i];
      if (!type_supported(b->descriptorType))
         return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                          "descriptor type %d is not supported", b->descriptorType);
      binding_count = MAX2(binding_count, b->binding + 1);
      if (has_sampler(b->descriptorType) && b->pImmutableSamplers)
         immutable_count += b->descriptorCount;
   }

   if (pCreateInfo->bindingCount) {
      result = vk_create_sorted_bindings(pCreateInfo->pBindings,
                                         pCreateInfo->bindingCount, &bindings,
                                         flags_info, &binding_flags);
      if (result != VK_SUCCESS)
         return vk_error(dev, result);
   }

   VK_MULTIALLOC(ma);
   VK_MULTIALLOC_DECL(&ma, struct mali_descriptor_set_layout, layout, 1);
   VK_MULTIALLOC_DECL(&ma, struct mali_descriptor_set_binding_layout, blayouts,
                      binding_count);
   VK_MULTIALLOC_DECL(&ma, struct mali_hw_desc, samplers, immutable_count);

   if (!vk_descriptor_set_layout_multizalloc(&dev->vk, &ma, pCreateInfo)) {
      free(bindings);
      free(binding_flags);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   layout->flags = pCreateInfo->flags;
   layout->bindings = blayouts;
   layout->binding_count = binding_count;

   uint32_t desc_idx = 0, dyn_idx = 0, dyn_ubos = 0, dyn_ssbos_count = 0;
   for (uint32_t i = 0; i < pCreateInfo->bindingCount; i++) {
      const VkDescriptorSetLayoutBinding *b = &bindings[i];
      struct mali_descriptor_set_binding_layout *bl = &layout->bindings[b->binding];

      if (b->descriptorCount == 0)
         continue;

      /* The spec forbids dynamic buffers in a push-descriptor layout
       * (vkCmdPushDescriptorSetKHR has nowhere to take a dynamic offset
       * from), so a push set's slots are never dynamic buffers. */
      assert(!(layout->flags & VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR) ||
             !vk_descriptor_type_is_dynamic(b->descriptorType));

      bl->type = b->descriptorType;
      bl->flags = binding_flags ? binding_flags[i] : 0;
      /* Only the highest binding may have a variable count (the
       * specification's rule), and not a dynamic buffer: those have no
       * slots. An inline uniform block is not offered. */
      if ((bl->flags & VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT) &&
          i == pCreateInfo->bindingCount - 1 && !vk_descriptor_type_is_dynamic(bl->type) &&
          bl->type != VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK) {
         layout->has_variable_count = true;
         layout->variable_binding = b->binding;
      }
      bl->desc_count = b->descriptorType == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK ?
                          mali_iub_desc_count(b->descriptorCount) :
                          b->descriptorCount;
      bl->textures_per_desc = has_texture(bl->type) ? 1 : 0;
      bl->samplers_per_desc = has_sampler(bl->type) ? 1 : 0;

      layout->type_counts[mali_desc_type_index(bl->type)] += b->descriptorCount;

      if (has_sampler(bl->type) && b->pImmutableSamplers) {
         bl->immutable_samplers = samplers;
         for (uint32_t j = 0; j < b->descriptorCount; j++) {
            VK_FROM_HANDLE(mali_sampler, sampler, b->pImmutableSamplers[j]);
            samplers[j] = sampler->desc;
         }
         samplers += b->descriptorCount;
         layout->has_immutable_samplers = true;
      }

      if (vk_descriptor_type_is_dynamic(bl->type)) {
         bl->desc_idx = dyn_idx;
         if (bl->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC) {
            layout->dyn_ssbos |= BITFIELD_RANGE(dyn_idx, bl->desc_count);
            dyn_ssbos_count += bl->desc_count;
         } else {
            dyn_ubos += bl->desc_count;
         }
         dyn_idx += bl->desc_count;
      } else {
         bl->desc_idx = desc_idx;
         desc_idx += mali_desc_stride(bl) * bl->desc_count;
      }
   }

   free(bindings);
   free(binding_flags);

   if (dyn_ubos > MALI_MAX_DYNAMIC_UNIFORM_BUFFERS ||
       dyn_ssbos_count > MALI_MAX_DYNAMIC_STORAGE_BUFFERS ||
       desc_idx > MALI_MAX_DESCS_PER_SET) {
      vk_descriptor_set_layout_unref(&dev->vk, &layout->vk);
      return vk_errorf(dev, VK_ERROR_OUT_OF_HOST_MEMORY,
                       "descriptor set layout exceeds the limits "
                       "(%u slots, %u dynamic uniform, %u dynamic storage buffers)",
                       desc_idx, dyn_ubos, dyn_ssbos_count);
   }

   layout->desc_count = desc_idx;
   layout->dyn_buf_count = dyn_idx;
   layout->vk.dynamic_descriptor_count = dyn_idx;

   /* The hash identifies the layout in pipeline cache keys: everything the
    * shader lowering reads. That includes the layout's flags: v7 compiles
    * a stage that reads a set laid out for update after bind differently
    * (mali_nir_lower_descriptors_v7.c choose_direct_set), and a shader
    * built for the one cannot run on the other. */
   struct mesa_blake3 ctx;
   _mesa_blake3_init(&ctx);
   _mesa_blake3_update(&ctx, &layout->flags, sizeof(layout->flags));
   _mesa_blake3_update(&ctx, &layout->binding_count, sizeof(layout->binding_count));
   _mesa_blake3_update(&ctx, &layout->desc_count, sizeof(layout->desc_count));
   _mesa_blake3_update(&ctx, &layout->dyn_buf_count, sizeof(layout->dyn_buf_count));
   for (uint32_t i = 0; i < binding_count; i++) {
      const struct mali_descriptor_set_binding_layout *bl = &layout->bindings[i];
      _mesa_blake3_update(&ctx, &bl->type, sizeof(bl->type));
      _mesa_blake3_update(&ctx, &bl->flags, sizeof(bl->flags));
      _mesa_blake3_update(&ctx, &bl->desc_count, sizeof(bl->desc_count));
      _mesa_blake3_update(&ctx, &bl->textures_per_desc, sizeof(bl->textures_per_desc));
      _mesa_blake3_update(&ctx, &bl->samplers_per_desc, sizeof(bl->samplers_per_desc));
   }
   _mesa_blake3_final(&ctx, layout->vk.blake3);

   *pSetLayout = mali_descriptor_set_layout_to_handle(layout);
   return VK_SUCCESS;
}

/* The checks of vkCreateDescriptorSetLayout without creating anything, and
 * for a binding with a variable descriptor count the most elements it can
 * have: what is left of a set's slots after the other bindings, over the
 * slots one element takes. */
VKAPI_ATTR void VKAPI_CALL
mali_GetDescriptorSetLayoutSupport(VkDevice _device,
                                   const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
                                   VkDescriptorSetLayoutSupport *pSupport)
{
   const VkDescriptorSetLayoutBindingFlagsCreateInfo *flags_info =
      vk_find_struct_const(pCreateInfo->pNext,
                           DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO);
   VkDescriptorSetVariableDescriptorCountLayoutSupport *var =
      vk_find_struct(pSupport->pNext, DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_LAYOUT_SUPPORT);
   if (var)
      var->maxVariableDescriptorCount = 0;

   pSupport->supported = false;

   uint64_t slots = 0, fixed_slots = 0;
   uint32_t dyn_ubos = 0, dyn_ssbos = 0, variable_stride = 0;
   for (uint32_t i = 0; i < pCreateInfo->bindingCount; i++) {
      const VkDescriptorSetLayoutBinding *b = &pCreateInfo->pBindings[i];
      if (!type_supported(b->descriptorType))
         return;

      /* A variable count binding with a count of zero is taken as one
       * element for this query (the specification's rule). */
      const bool variable =
         flags_info && i < flags_info->bindingCount &&
         (flags_info->pBindingFlags[i] & VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT) &&
         !vk_descriptor_type_is_dynamic(b->descriptorType) &&
         b->descriptorType != VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK;
      const uint32_t count = variable ? MAX2(b->descriptorCount, 1) : b->descriptorCount;

      uint64_t binding_slots = 0;
      switch (b->descriptorType) {
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
         dyn_ubos += count;
         break;
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
         dyn_ssbos += count;
         break;
      case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK:
         if (count)
            binding_slots = mali_iub_desc_count(count);
         break;
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
         binding_slots = 2ull * count;
         break;
      default:
         binding_slots = count;
         break;
      }
      slots += binding_slots;
      if (variable)
         variable_stride = b->descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ? 2 : 1;
      else
         fixed_slots += binding_slots;
   }

   pSupport->supported = dyn_ubos <= MALI_MAX_DYNAMIC_UNIFORM_BUFFERS &&
                         dyn_ssbos <= MALI_MAX_DYNAMIC_STORAGE_BUFFERS &&
                         slots <= MALI_MAX_DESCS_PER_SET;
   /* v7 does not offer a variable descriptor count: its tables are filled
    * for a layout's whole count. */
   VK_FROM_HANDLE(mali_device, dev, _device);
   if (var && pSupport->supported && variable_stride && mali_device_physical(dev)->arch != 7) {
      var->maxVariableDescriptorCount =
         MIN2((MALI_MAX_DESCS_PER_SET - fixed_slots) / variable_stride, MALI_MAX_DESCS_PER_TYPE);
   }
}
