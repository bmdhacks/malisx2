/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * vkCmdCopyBuffer, vkCmdFillBuffer and vkCmdUpdateBuffer, as compute
 * dispatches of small internal shaders: on the compute subqueue (CSF, v11)
 * or as Compute jobs in the open batch's vtc chain (job manager, v7 and v9).
 * Built once per arch.
 *
 * Why compute and not the command stream itself: CS LOAD_MULTIPLE /
 * STORE_MULTIPLE move at most 64 bytes per instruction pair through the
 * CS front end, which is a microcontroller; a shader moves 16 bytes per
 * invocation on every core. The blob does the same: one compute dispatch
 * per copy region, with the element size the largest power of two up to
 * 16 that divides both addresses. ARMSX2 uses vkCmdCopyBuffer for
 * uploads; fill and update are outside its default path but cost nothing
 * extra here.
 *
 * The shaders are built with nir_builder (no descriptors: addresses come
 * in push constants, loads and stores are global), compiled through the
 * normal path on first use, and kept for the device's lifetime.
 */

#include "mali_blend.h"
#if PAN_ARCH >= 10
#include "mali_cmd_buffer.h"
#else
#include "mali_jm.h"
#endif

#include <string.h>

#include "compiler/nir/nir_builder.h"
#include "util/mesa-blake3.h"
#include "vk_buffer.h"
#include "vk_pipeline.h"

#include "mali_arch.h"
#include "mali_memory.h"
#include "mali_vk.h"

#define META_WG_SIZE 64
/* Elements per dispatch, so JOB_SIZE stays well inside 32 bits. */
#define META_MAX_ELEMS (1u << 30)

struct meta_copy_push {
   uint64_t src;
   uint64_t dst;
   uint32_t count;
   uint32_t pad;
};

struct meta_fill_push {
   uint64_t dst;
   uint32_t count;
   uint32_t value;
};

static unsigned
meta_elem_size(enum mali_meta_shader m)
{
   switch (m) {
   case MALI_META_COPY_16:
   case MALI_META_FILL_16:
      return 16;
   case MALI_META_COPY_4:
   case MALI_META_FILL_4:
      return 4;
   default:
      return 1;
   }
}

static nir_shader *
build_meta_nir(struct mali_device *dev, enum mali_meta_shader m)
{
   static const char *const names[] = {
      [MALI_META_COPY_16] = "mali_meta_copy16",
      [MALI_META_COPY_4] = "mali_meta_copy4",
      [MALI_META_COPY_1] = "mali_meta_copy1",
      [MALI_META_FILL_16] = "mali_meta_fill16",
      [MALI_META_FILL_4] = "mali_meta_fill4",
   };
   const bool fill = m == MALI_META_FILL_16 || m == MALI_META_FILL_4;
   const unsigned elem = meta_elem_size(m);
   const unsigned comps = elem == 16 ? 4 : 1;
   const unsigned bits = elem == 1 ? 8 : 32;

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, MALI_PER_ARCH(shader_nir_options)(dev, MESA_SHADER_COMPUTE), "%s",
      names[m]);
   b.shader->info.workgroup_size[0] = META_WG_SIZE;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   const unsigned push_size =
      fill ? sizeof(struct meta_fill_push) : sizeof(struct meta_copy_push);
   nir_def *id = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *count = nir_load_push_constant(
      &b, 1, 32, nir_imm_int(&b, fill ? offsetof(struct meta_fill_push, count)
                                      : offsetof(struct meta_copy_push, count)),
      .base = 0, .range = push_size);

   nir_push_if(&b, nir_ult(&b, id, count));
   {
      nir_def *off = nir_u2u64(&b, nir_imul_imm(&b, id, elem));
      nir_def *dst = nir_load_push_constant(
         &b, 1, 64, nir_imm_int(&b, fill ? offsetof(struct meta_fill_push, dst)
                                         : offsetof(struct meta_copy_push, dst)),
         .base = 0, .range = push_size);
      nir_def *value;
      if (fill) {
         nir_def *v = nir_load_push_constant(
            &b, 1, 32, nir_imm_int(&b, offsetof(struct meta_fill_push, value)),
            .base = 0, .range = push_size);
         value = comps == 4 ? nir_vec4(&b, v, v, v, v) : v;
      } else {
         nir_def *src = nir_load_push_constant(
            &b, 1, 64, nir_imm_int(&b, offsetof(struct meta_copy_push, src)),
            .base = 0, .range = push_size);
         value = nir_load_global(&b, comps, bits, nir_iadd(&b, src, off),
                                 .align_mul = elem);
      }
      nir_store_global(&b, value, nir_iadd(&b, dst, off),
                       .write_mask = BITFIELD_MASK(comps), .align_mul = elem);
   }
   nir_pop_if(&b, NULL);
   return b.shader;
}

static const struct mali_shader *
get_meta(struct mali_cmd_buffer *cmd, enum mali_meta_shader m)
{
   struct mali_device *dev = cmd->dev;

   simple_mtx_lock(&dev->meta_lock);
   struct mali_shader *s = dev->meta[m];
   if (!s) {
      nir_shader *nir = build_meta_nir(dev, m);
      MALI_PER_ARCH(shader_preprocess)(dev, nir);

      const struct vk_pipeline_robustness_state rs = {
         .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
         .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
         .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
         .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
      };
      const struct mali_shader_compile_info info = {
         .nir = nir,
         .rs = &rs,
      };
      blake3_hash key;
      struct mesa_blake3 h;
      _mesa_blake3_init(&h);
      _mesa_blake3_update(&h, "mali-meta", 9);
      _mesa_blake3_update(&h, &m, sizeof(m));
      _mesa_blake3_final(&h, key);

      VkResult r = MALI_PER_ARCH(shader_compile)(dev, &info, key, &s);
      if (r != VK_SUCCESS)
         s = NULL;
      dev->meta[m] = s;
   }
   simple_mtx_unlock(&dev->meta_lock);

   if (!s)
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   return s;
}

void
MALI_PER_ARCH(meta_finish)(struct mali_device *dev)
{
   for (unsigned i = 0; i < MALI_META_COUNT; i++) {
      MALI_PER_ARCH(shader_unref)(dev, dev->meta[i]);
      dev->meta[i] = NULL;
   }
   MALI_PER_ARCH(meta_gfx_finish)(dev);
   MALI_PER_ARCH(blend_shaders_finish)(dev);
}

static void
dispatch_elems(struct mali_cmd_buffer *cmd, enum mali_meta_shader m,
               void *push, uint32_t push_size, uint32_t *count_field,
               uint64_t elems, uint64_t *src, uint64_t *dst)
{
   const struct mali_shader *s = get_meta(cmd, m);
   if (!s)
      return;
   const unsigned elem = meta_elem_size(m);
#if PAN_ARCH < 9
   /* The buffer memory written, for the v7 indexed draws
    * (mali_v7_note_buffer_write). */
   mali_v7_note_buffer_write(cmd, *dst, elems * elem);
#endif

   while (elems) {
      uint32_t n = (uint32_t)MIN2(elems, META_MAX_ELEMS);
      *count_field = n;
      const uint32_t base[3] = {0, 0, 0};
      const uint32_t groups[3] = {DIV_ROUND_UP(n, META_WG_SIZE), 1, 1};
      MALI_PER_ARCH(cmd_dispatch_shader)(cmd, s, NULL, push, push_size, base, groups);
      elems -= n;
      if (src)
         *src += (uint64_t)n * elem;
      *dst += (uint64_t)n * elem;
   }
}

static void
record_copy(struct mali_cmd_buffer *cmd, uint64_t src, uint64_t dst, uint64_t size)
{
   if (!size)
      return;
   enum mali_meta_shader m;
   uint64_t bits = src | dst | size;
   if (!(bits & 15))
      m = MALI_META_COPY_16;
   else if (!(bits & 3))
      m = MALI_META_COPY_4;
   else
      m = MALI_META_COPY_1;

   struct meta_copy_push push = {.src = src, .dst = dst};
   dispatch_elems(cmd, m, &push, sizeof(push), &push.count,
                  size / meta_elem_size(m), &push.src, &push.dst);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdCopyBuffer2)(VkCommandBuffer commandBuffer, const VkCopyBufferInfo2 *info)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_buffer, src, info->srcBuffer);
   VK_FROM_HANDLE(vk_buffer, dst, info->dstBuffer);

   /* One dispatch per region, as the blob. */
   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkBufferCopy2 *r = &info->pRegions[i];
      record_copy(cmd, vk_buffer_address(src, r->srcOffset),
                  vk_buffer_address(dst, r->dstOffset), r->size);
   }
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdFillBuffer)(VkCommandBuffer commandBuffer, VkBuffer dstBuffer,
                             VkDeviceSize dstOffset, VkDeviceSize size, uint32_t data)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_buffer, dst, dstBuffer);

   /* VK_WHOLE_SIZE: to the end, rounded down to a multiple of 4. */
   size = vk_buffer_range(dst, dstOffset, size) & ~3ull;
   if (!size)
      return;

   uint64_t addr = vk_buffer_address(dst, dstOffset);
   enum mali_meta_shader m =
      !((addr | size) & 15) ? MALI_META_FILL_16 : MALI_META_FILL_4;
   struct meta_fill_push push = {.dst = addr, .value = data};
   dispatch_elems(cmd, m, &push, sizeof(push), &push.count,
                  size / meta_elem_size(m), NULL, &push.dst);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdUpdateBuffer)(VkCommandBuffer commandBuffer, VkBuffer dstBuffer,
                               VkDeviceSize dstOffset, VkDeviceSize dataSize,
                               const void *pData)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_buffer, dst, dstBuffer);

   if (!dataSize)
      return;
   /* The data goes into command-buffer memory now, and a copy moves it
    * when the command buffer runs. */
   struct mali_ptr p = mali_cmd_alloc(cmd, dataSize, 16);
   if (!p.cpu)
      return;
   memcpy(p.cpu, pData, dataSize);
   record_copy(cmd, p.gpu, vk_buffer_address(dst, dstOffset), dataSize);
}
