/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The command-buffer state both back halves keep the same way, out of
 * line (the per-draw builders are inline, in mali_cmd_state.h): graphics
 * pipeline binds and their dirty bits, vertex and index buffer binds, the
 * FAU block of a graphics-stage shader, the command buffer's TLS buffer,
 * and the pipeline, descriptor-set and push-constant entry points.
 */

#include "mali_cmd_state.h"

#include "vk_buffer.h"
#include "vk_command_pool.h"
#include "vk_descriptor_update_template.h"
#include "vk_pipeline.h"
#include "vk_pipeline_layout.h"

#include "mali_arch.h"
#include "mali_memory.h"

/* ---------------------------------------------------------------------- */
/* Binding                                                                 */

/*
 * The static states other than vertex input that the draws read, copied
 * into the command buffer's dynamic state the way the runtime's
 * vk_dynamic_graphics_state_copy does it (value, then the dirty and set
 * bits when it changed), for the few states that need it.
 */
static void
bind_static_state(struct vk_dynamic_graphics_state *dyn, const struct mali_graphics_pipeline *p,
                  uint32_t m)
{
   const struct vk_dynamic_graphics_state *src = &p->state;
#define MARK(STATE)                                                            \
   do {                                                                        \
      BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_##STATE);                         \
      BITSET_SET(dyn->set, MESA_VK_DYNAMIC_##STATE);                           \
   } while (0)

   if (m & MALI_GFX_STATIC_VIEWPORTS) {
      const uint32_t n = src->vp.viewport_count;
      if (dyn->vp.viewport_count != n ||
          memcmp(dyn->vp.viewports, src->vp.viewports, n * sizeof(src->vp.viewports[0]))) {
         dyn->vp.viewport_count = n;
         memcpy(dyn->vp.viewports, src->vp.viewports, n * sizeof(src->vp.viewports[0]));
         MARK(VP_VIEWPORT_COUNT);
         MARK(VP_VIEWPORTS);
      }
   }
   if (m & MALI_GFX_STATIC_SCISSORS) {
      const uint32_t n = src->vp.scissor_count;
      if (dyn->vp.scissor_count != n ||
          memcmp(dyn->vp.scissors, src->vp.scissors, n * sizeof(src->vp.scissors[0]))) {
         dyn->vp.scissor_count = n;
         memcpy(dyn->vp.scissors, src->vp.scissors, n * sizeof(src->vp.scissors[0]));
         MARK(VP_SCISSOR_COUNT);
         MARK(VP_SCISSORS);
      }
   }
   if ((m & MALI_GFX_STATIC_BLEND_CONSTANTS) &&
       memcmp(dyn->cb.blend_constants, src->cb.blend_constants, sizeof(src->cb.blend_constants))) {
      memcpy(dyn->cb.blend_constants, src->cb.blend_constants, sizeof(src->cb.blend_constants));
      MARK(CB_BLEND_CONSTANTS);
   }
   /* These two from the bind block's copies: every ARMSX2 pipeline sets
    * them, and the full state is on cache lines a bind otherwise does not
    * touch. */
   if ((m & MALI_GFX_STATIC_LINE_WIDTH) && dyn->rs.line.width != p->bind.line_width) {
      dyn->rs.line.width = p->bind.line_width;
      MARK(RS_LINE_WIDTH);
   }
   if ((m & MALI_GFX_STATIC_IAL) && memcmp(&dyn->ial, &p->bind.ial, sizeof(p->bind.ial))) {
      dyn->ial = p->bind.ial;
      MARK(INPUT_ATTACHMENT_MAP);
   }
#undef MARK
}

/*
 * vkCmdBindPipeline for a graphics pipeline. A pipeline whose baked words
 * use no dynamic state (every ARMSX2 pipeline) does not go through the
 * runtime's copy of its whole static state: the draws read only vertex
 * input (pointed to), viewport, scissor, blend constants, line width and
 * the input attachment map (copied when the pipeline sets them). The
 * runtime's state is left stale for the rest, which nothing reads: a later
 * pipeline that repacks from dynamic state gets the full copy, and state
 * it leaves dynamic must be set again after its bind (Vulkan's rule for
 * state a bound pipeline had static).
 */
void
MALI_PER_ARCH(cmd_bind_graphics)(struct mali_cmd_buffer *cmd, struct mali_graphics_pipeline *p)
{
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const struct mali_gfx_bind *bd = &p->bind;

   if (unlikely(bd->dynamic)) {
      vk_cmd_set_dynamic_graphics_state(&cmd->vk, &p->state);
      d->vi = dyn->vi;
      d->vi_strides = dyn->vi_binding_strides;
   } else {
      const uint32_t m = bd->static_mask;
      d->vi = (m & MALI_GFX_STATIC_VI) ? &p->vi : dyn->vi;
      d->vi_strides = (m & MALI_GFX_STATIC_STRIDES) ? p->state.vi_binding_strides
                                                     : dyn->vi_binding_strides;
      if (m & ~(MALI_GFX_STATIC_VI | MALI_GFX_STATIC_STRIDES))
         bind_static_state(dyn, p, m);
   }

   if (d->pipeline == p)
      return;
   d->pipeline = p;
   d->dirty |= MALI_GFX_DIRTY_PIPELINE | MALI_GFX_DIRTY_VS_FAU | MALI_GFX_DIRTY_FS_FAU |
               MALI_GFX_DIRTY_BLEND | MALI_GFX_DIRTY_ZSD;
   /* Tables whose inputs from the pipeline side did not change stay. */
   if (!bd->vs_srt_key || bd->vs_srt_key != d->vs_srt_key)
      d->dirty |= MALI_GFX_DIRTY_VS_SRT;
   if (!bd->fs_srt_key || bd->fs_srt_key != d->fs_srt_key)
      d->dirty |= MALI_GFX_DIRTY_FS_SRT;
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdBindVertexBuffers2)(VkCommandBuffer commandBuffer, uint32_t firstBinding,
                                     uint32_t bindingCount, const VkBuffer *pBuffers,
                                     const VkDeviceSize *pOffsets, const VkDeviceSize *pSizes,
                                     const VkDeviceSize *pStrides)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;

   assert(firstBinding + bindingCount <= MALI_MAX_VBS);
   for (uint32_t i = 0; i < bindingCount; i++) {
      VK_FROM_HANDLE(vk_buffer, buf, pBuffers[i]);
      const uint32_t b = firstBinding + i;
      if (!buf) {
         d->vb[b].addr = 0;
         d->vb[b].size = 0;
         continue;
      }
      d->vb[b].addr = vk_buffer_address(buf, pOffsets[i]);
      d->vb[b].size = vk_buffer_range(buf, pOffsets[i], pSizes ? pSizes[i] : VK_WHOLE_SIZE);
   }
   if (pStrides)
      vk_cmd_set_vertex_binding_strides(&cmd->vk, firstBinding, bindingCount, pStrides);
#if PAN_ARCH >= 9
   /* v9+: the attributes live in the VS table (v7 packs them per draw). */
   d->dirty |= MALI_GFX_DIRTY_VS_SRT;
#endif
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdBindIndexBuffer2KHR)(VkCommandBuffer commandBuffer, VkBuffer buffer,
                                      VkDeviceSize offset, VkDeviceSize size, VkIndexType indexType)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_buffer, buf, buffer);
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;

   d->ib.index_size = vk_index_type_to_bytes(indexType);
   if (buf) {
      d->ib.addr = vk_buffer_address(buf, offset);
      d->ib.size = (uint32_t)MIN2(vk_buffer_range(buf, offset, size), UINT32_MAX);
   } else {
      d->ib.addr = 0;
      d->ib.size = 0;
   }
#if PAN_ARCH < 9
   /* v7 runs the vertex shader over the index range of an indexed draw,
    * which the v7 draw finds by reading the indices on the CPU
    * (mali_jm_cmd_draw_v7.c): the buffer's CPU view, NULL for an imported
    * dma-buf, which has none. */
   const struct mali_buffer *mb = buf ? container_of(buf, struct mali_buffer, vk) : NULL;
   cmd->jm.draw.ib_cpu = (mb && mb->mem && mb->mem->bo.cpu)
                            ? (const uint8_t *)mb->mem->bo.cpu + mb->mem_offset + offset
                            : NULL;
#endif
   d->dirty |= MALI_GFX_DIRTY_INDEX;
}

/* ---------------------------------------------------------------------- */
/* FAU                                                                     */

uint64_t
MALI_PER_ARCH(cmd_gfx_fau)(struct mali_cmd_buffer *cmd, const struct mali_shader *s,
                           const void *sysvals, const void *push, uint32_t push_size)
{
   if (!s || !s->fau.total_count)
      return 0;

   const uint32_t total = s->fau.total_count;
   struct mali_ptr p = mali_cmd_alloc(cmd, total * 8, 16);
   if (!p.cpu)
      return 0;

   /* The sysval words the shader reads, with the block's own address in
    * common.push_uniforms, then its push-constant words (zero past the
    * end of push). Read in place: this runs twice a draw. */
   const unsigned push_uniforms_word =
      offsetof(struct mali_graphics_sysvals, common.push_uniforms) / MALI_FAU_WORD_SIZE;
   uint64_t tmp[MALI_FAU_WORD_COUNT];
   unsigned n = 0, w;
   BITSET_FOREACH_SET(w, s->fau.used_sysvals, MALI_MAX_SYSVAL_FAUS) {
      tmp[n++] = w == push_uniforms_word ?
                    p.gpu :
                    mali_sysval_word(sysvals, sizeof(struct mali_graphics_sysvals), w);
   }
   const uint32_t push_len = MIN2(push_size, MALI_MAX_PUSH_CONST_FAUS * MALI_FAU_WORD_SIZE);
   BITSET_FOREACH_SET(w, s->fau.used_push_consts, MALI_MAX_PUSH_CONST_FAUS) {
      uint64_t v = 0;
      const uint32_t off = w * MALI_FAU_WORD_SIZE;
      if (off < push_len)
         memcpy(&v, (const uint8_t *)push + off, MIN2(push_len - off, sizeof(v)));
      tmp[n++] = v;
   }
   n += mali_shader_fau_consts(&s->fau, &tmp[n]);
   /* Pushed uniform-buffer words: zero here, copied in by the GPU
    * (mali_shader.h struct mali_shader_fau). */
   assert(!s->fau.ubo_push_count || n == s->fau.ubo_push_start);
   for (unsigned i = n; i < total; i++)
      tmp[i] = 0;

   pan_fau_foreach_imm(&s->info.fau, i) {
      bool hi = i & 1;
      unsigned idx = i / 2;
      assert(idx < total);
      tmp[idx] = (tmp[idx] & ((uint64_t)UINT32_MAX << (32 * !hi))) |
                 ((uint64_t)s->info.fau.words[i].constant << (32 * hi));
   }

   memcpy(p.cpu, tmp, total * sizeof(tmp[0]));
   return p.gpu | ((uint64_t)total << 56);
}

/* ---------------------------------------------------------------------- */
/* Thread-local storage                                                    */

uint64_t
MALI_PER_ARCH(cmd_tls_buffer)(struct mali_cmd_buffer *cmd, uint32_t tls_size)
{
   struct mali_device *dev = cmd->dev;
   const struct mali_thread_props tp = mali_thread_props(dev);

   /* Per thread: the next power of two of the size, at least 16 bytes; for
    * every thread slot of every core (pan_get_total_stack_size). */
   unsigned per_thread = util_next_power_of_two(ALIGN_POT(tls_size, 16));
#if PAN_ARCH >= 10
   const unsigned slots = tp.max_threads_per_core;
#else
   /* On the job manager the hardware indexes the buffer by TLS thread
    * slot, and THREAD_TLS_ALLOC (the slot count per core) may be larger
    * than the thread count, which it rounds up to a power of two. The
    * kernel reports 0 when the register does not exist; the thread count
    * is then the slot count (Mesa and kbase both fall back the same way). */
   const unsigned slots = MAX2(dev->kbase->props.thread_tls_alloc, tp.max_threads_per_core);
#endif
   if (per_thread > cmd->tls.size) {
      uint64_t total = (uint64_t)per_thread * slots * tp.core_id_range;
      /* TLS memory: GPU group 9. */
      struct mali_cmd_slab *s = calloc(1, sizeof(*s));
      const struct mali_kbase_alloc_info ai = {
         .size = total,
         .flags = mali_kbase_mem_same_va_policy(MALI_KBASE_FLAGS_TLS),
         .mem_class = MALI_KBASE_MEM_CLASS_INTERNAL_TLS,
      };
      if (!s || mali_kbase_alloc(dev->kbase, &ai, &s->bo) != MALI_KBASE_SUCCESS) {
         free(s);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return 0;
      }
      list_addtail(&s->link, &cmd->slabs);
      cmd->tls.size = per_thread;
      cmd->tls.gpu = s->bo.gpu_va;
   }
   return cmd->tls.gpu;
}

/* ---------------------------------------------------------------------- */
/* Pipeline, descriptor-set and push-constant binds (both back halves)    */

void
MALI_PER_ARCH(cmd_bind_pipeline)(struct mali_cmd_buffer *cmd, struct vk_pipeline *pipeline)
{
   if (pipeline->bind_point == VK_PIPELINE_BIND_POINT_COMPUTE)
      cmd->compute.shader = mali_compute_pipeline(pipeline)->cs;
   else if (pipeline->bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS)
      MALI_PER_ARCH(cmd_bind_graphics)(cmd, mali_graphics_pipeline(pipeline));
}

/* Replaces the runtime's vkCmdBindPipeline, which only calls through the
 * pipeline's ops to the same place. */
VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdBindPipeline)(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
                               VkPipeline _pipeline)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_pipeline, pipeline, _pipeline);
   assert(pipeline->bind_point == pipelineBindPoint);
   MALI_PER_ARCH(cmd_bind_pipeline)(cmd, pipeline);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdBindDescriptorSets2KHR)(VkCommandBuffer commandBuffer,
                                         const VkBindDescriptorSetsInfoKHR *info)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_pipeline_layout, layout, info->layout);

   struct mali_desc_state *states[2];
   unsigned n = 0;
   if (info->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT)
      states[n++] = &cmd->compute.desc;
   if (info->stageFlags & VK_SHADER_STAGE_ALL_GRAPHICS)
      states[n++] = &cmd->gfx.desc;

   if (info->stageFlags & VK_SHADER_STAGE_ALL_GRAPHICS) {
      /* The draw rebuilds a stage's table when it reads one of these. */
      const uint32_t sets = BITFIELD_RANGE(info->firstSet, info->descriptorSetCount);
      cmd->gfx.draw.vs_sets_dirty |= sets;
      cmd->gfx.draw.fs_sets_dirty |= sets;
   }

   for (unsigned s = 0; s < n; s++) {
      struct mali_desc_state *st = states[s];
      uint32_t dyn = 0;
      for (uint32_t i = 0; i < info->descriptorSetCount; i++) {
         uint32_t idx = info->firstSet + i;
         VK_FROM_HANDLE(mali_descriptor_set, set, info->pDescriptorSets[i]);
         assert(idx < MALI_MAX_SETS);
         st->sets[idx] = set;
         const struct mali_descriptor_set_layout *sl =
            set ? set->layout :
                  (layout && layout->set_layouts[idx] ?
                      mali_descriptor_set_layout(layout->set_layouts[idx]) : NULL);
         uint32_t count = sl ? sl->dyn_buf_count : 0;
         for (uint32_t d = 0; d < count; d++) {
            st->dyn_offsets[idx][d] =
               dyn < info->dynamicOffsetCount ? info->pDynamicOffsets[dyn] : 0;
            dyn++;
         }
      }
   }
}

/* The Vulkan 1.0 entry point, straight to the one above (the runtime's
 * version goes through the dispatch table). */
VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdBindDescriptorSets)(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint,
                                     VkPipelineLayout layout, uint32_t firstSet,
                                     uint32_t descriptorSetCount, const VkDescriptorSet *pDescriptorSets,
                                     uint32_t dynamicOffsetCount, const uint32_t *pDynamicOffsets)
{
   const VkBindDescriptorSetsInfoKHR info = {
      .sType = VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO_KHR,
      .stageFlags = vk_shader_stages_from_bind_point(pipelineBindPoint),
      .layout = layout,
      .firstSet = firstSet,
      .descriptorSetCount = descriptorSetCount,
      .pDescriptorSets = pDescriptorSets,
      .dynamicOffsetCount = dynamicOffsetCount,
      .pDynamicOffsets = pDynamicOffsets,
   };
   MALI_PER_ARCH(CmdBindDescriptorSets2KHR)(commandBuffer, &info);
}

/*
 * A push descriptor set (VK_KHR_push_descriptor): a set whose storage is
 * the command buffer's, not a pool's. Pushing always allocates a fresh
 * block for the set's slots (mali_cmd_alloc, freed only when the command
 * buffer is reset), so a draw already recorded keeps reading the memory
 * its resource table pointed at even after a later push changes what is
 * bound at that set index.
 *
 * A binding the push does not mention carries over from the set's last
 * push in this command buffer: push[idx] is reused as the container across
 * pushes, so carrying forward is "this is still the same push, add to it"
 * when sets[idx] is still push[idx] and its layout has not changed; a real
 * vkCmdBindDescriptorSets at idx (which points sets[idx] elsewhere) or a
 * command buffer reset (which clears push[idx].layout) both make the next
 * push start over, from zeroed slots.
 *
 * The slots are written in the set's cached copy (struct mali_push_shadow:
 * set->cpu points there, set->gpu at the new block), which holds what the
 * previous push left, and push_end copies the finished set into the block.
 * Command memory is never read by the CPU.
 */
struct push_target {
   struct mali_descriptor_set *set;
   void *dst;                /* the new block's mapping; NULL: no slots */
   uint32_t size;
};

static bool
push_begin(struct mali_cmd_buffer *cmd, struct mali_desc_state *st, unsigned shadow_bp,
           const struct mali_descriptor_set_layout *layout, uint32_t idx,
           struct push_target *out)
{
   struct mali_descriptor_set *set = &st->push[idx];
   struct mali_push_shadow *sh = &cmd->push_shadow.s[shadow_bp][idx];
   const bool carry = st->sets[idx] == set && set->layout == layout;
   const uint32_t size = layout->desc_count * MALI_DESCRIPTOR_SIZE;

   out->set = set;
   out->dst = NULL;
   out->size = size;
   if (size) {
      if (size > sh->size) {
         /* Only a new layout grows it, and a new layout carries nothing. */
         assert(!carry);
         vk_free(&cmd->vk.pool->alloc, sh->data);
         sh->size = 0;
         sh->data = vk_alloc(&cmd->vk.pool->alloc, size, 16,
                             VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
         if (!sh->data) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
            return false;
         }
         sh->size = size;
      }
      struct mali_ptr p = mali_cmd_alloc(cmd, size, MALI_DESCRIPTOR_SIZE);
      if (!p.cpu) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return false;
      }
      if (!carry)
         memset(sh->data, 0, size);
      set->cpu = sh->data;
      set->gpu = p.gpu;
      out->dst = p.cpu;
   } else {
      set->cpu = NULL;
      set->gpu = 0;
   }
   set->layout = layout;
   st->sets[idx] = set;
   MALI_PER_ARCH(descriptor_set_init_fixed_slots)(set);
   return true;
}

static void
push_end(const struct push_target *t)
{
   if (t->dst)
      memcpy(t->dst, t->set->cpu, t->size);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdPushDescriptorSet2KHR)(VkCommandBuffer commandBuffer,
                                        const VkPushDescriptorSetInfoKHR *info)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_pipeline_layout, layout, info->layout);

   assert(info->set < MALI_MAX_SETS);
   const struct mali_descriptor_set_layout *sl =
      mali_descriptor_set_layout(layout->set_layouts[info->set]);

   struct mali_desc_state *states[2];
   unsigned shadow_bp[2];
   unsigned n = 0;
   if (info->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT) {
      shadow_bp[n] = MALI_PUSH_SHADOW_COMPUTE;
      states[n++] = &cmd->compute.desc;
   }
   if (info->stageFlags & VK_SHADER_STAGE_ALL_GRAPHICS) {
      shadow_bp[n] = MALI_PUSH_SHADOW_GFX;
      states[n++] = &cmd->gfx.desc;
   }

   if (info->stageFlags & VK_SHADER_STAGE_ALL_GRAPHICS) {
      const uint32_t bit = BITFIELD_BIT(info->set);
      cmd->gfx.draw.vs_sets_dirty |= bit;
      cmd->gfx.draw.fs_sets_dirty |= bit;
   }

   for (unsigned s = 0; s < n; s++) {
      struct push_target t;
      if (!push_begin(cmd, states[s], shadow_bp[s], sl, info->set, &t))
         return;
      MALI_PER_ARCH(descriptor_set_write)(t.set, info->descriptorWriteCount,
                                          info->pDescriptorWrites);
      push_end(&t);
   }
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdPushDescriptorSetWithTemplate2KHR)(
   VkCommandBuffer commandBuffer,
   const VkPushDescriptorSetWithTemplateInfoKHR *info)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_pipeline_layout, layout, info->layout);
   VK_FROM_HANDLE(vk_descriptor_update_template, template, info->descriptorUpdateTemplate);

   assert(info->set < MALI_MAX_SETS);
   assert(template->bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS ||
          template->bind_point == VK_PIPELINE_BIND_POINT_COMPUTE);
   const struct mali_descriptor_set_layout *sl =
      mali_descriptor_set_layout(layout->set_layouts[info->set]);

   const bool compute = template->bind_point == VK_PIPELINE_BIND_POINT_COMPUTE;
   struct mali_desc_state *st = compute ? &cmd->compute.desc : &cmd->gfx.desc;

   if (template->bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS) {
      const uint32_t bit = BITFIELD_BIT(info->set);
      cmd->gfx.draw.vs_sets_dirty |= bit;
      cmd->gfx.draw.fs_sets_dirty |= bit;
   }

   struct push_target t;
   if (!push_begin(cmd, st, compute ? MALI_PUSH_SHADOW_COMPUTE : MALI_PUSH_SHADOW_GFX, sl,
                   info->set, &t))
      return;

   for (uint32_t i = 0; i < template->entry_count; i++) {
      const struct vk_descriptor_template_entry *e = &template->entries[i];
      const VkWriteDescriptorSet w = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstBinding = e->binding,
         .dstArrayElement = e->array_element,
         .descriptorCount = e->array_count,
         .descriptorType = e->type,
         /*
          * The template's stride need not match the Vulkan struct's, so
          * these can only be used for a one-element write (descriptorCount
          * 1); descriptor_set_write reads pImageInfo/pBufferInfo as
          * e->array_count-element arrays only for a plain
          * vkUpdateDescriptorSets call, never for a template, so writing
          * one element at a time here is required, not just simplest.
          */
      };
      for (uint32_t j = 0; j < e->array_count; j++) {
         const void *src = (const uint8_t *)info->pData + e->offset + (size_t)j * e->stride;
         VkWriteDescriptorSet w1 = w;
         w1.dstArrayElement = e->array_element + j;
         w1.descriptorCount = 1;
         switch (e->type) {
         case VK_DESCRIPTOR_TYPE_SAMPLER:
         case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
         case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
         case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
            w1.pImageInfo = src;
            break;
         case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
         case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
            w1.pTexelBufferView = src;
            break;
         case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK:
            /* Push-descriptor layouts have no inline uniform blocks
             * ARMSX2 uses (mali_descriptor_set_layout.c); not reached by
             * anything this driver serves. */
            UNREACHABLE("inline uniform blocks are not pushed");
         default:
            w1.pBufferInfo = src;
            break;
         }
         MALI_PER_ARCH(descriptor_set_write)(t.set, 1, &w1);
      }
   }
   push_end(&t);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdPushConstants2KHR)(VkCommandBuffer commandBuffer,
                                    const VkPushConstantsInfoKHR *info)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   assert(info->offset + info->size <= sizeof(cmd->push_constants));
   memcpy(cmd->push_constants + info->offset, info->pValues, info->size);
   cmd->gfx.draw.dirty |= MALI_GFX_DIRTY_VS_FAU | MALI_GFX_DIRTY_FS_FAU;
}
