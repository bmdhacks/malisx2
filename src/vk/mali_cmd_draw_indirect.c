/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Indirect draws (vkCmdDrawIndirect, vkCmdDrawIndexedIndirect and their
 * count forms) loop in the command stream, as panvk's launch_indirect_draw
 * and the blob do: each command is loaded into INDEX_COUNT..INSTANCE_OFFSET,
 * and a draw with no vertices or no instances runs no RUN_IDVS. What the
 * vertex stage reads of the command (the first-vertex and base-instance
 * system values, and the offsets of per-instance attributes, which carry
 * firstInstance x stride as in a direct draw) is stored by the stream
 * into a per-draw copy of the vertex FAU block and of the vertex resource
 * table, so draws in flight never see each other's values. The copies are
 * a ring of at most INDIRECT_RING_BYTES; when a loop has more draws than
 * the ring holds, the stream waits for the draws already started before it
 * reuses a copy.
 */

#include "mali_cmd_draw.h"
#include "mali_measure.h"

#include "vk_buffer.h"
#include "vk_log.h"

#include "mali_arch.h"
#include "mali_queue.h"

/* ---------------------------------------------------------------------- */
/* Indirect draws                                                          */

/* Most bytes of per-draw FAU block and resource table copies one indirect
 * draw takes (half a command-buffer slab). */
#define INDIRECT_RING_BYTES (32 * 1024)

/* The first vertex and the base instance are one FAU word, in the order of
 * VERTEX_OFFSET and INSTANCE_OFFSET, so one STORE_MULTIPLE of the two
 * registers writes both. */
static_assert(offsetof(struct mali_graphics_sysvals, vs.first_vertex) % MALI_FAU_WORD_SIZE == 0,
              "first_vertex starts an FAU word");
static_assert(offsetof(struct mali_graphics_sysvals, vs.base_instance) ==
                 offsetof(struct mali_graphics_sysvals, vs.first_vertex) + 4,
              "base_instance follows first_vertex");
static_assert(MALI_IDVS_SR_INSTANCE_OFFSET == MALI_IDVS_SR_VERTEX_OFFSET + 1,
              "INSTANCE_OFFSET follows VERTEX_OFFSET");

/* Byte offset in the vertex FAU block of the word holding first_vertex and
 * base_instance, -1 if the shader reads neither. */
static int
vs_draw_sysval_offset(const struct mali_shader *vs)
{
   const unsigned w = offsetof(struct mali_graphics_sysvals, vs.first_vertex) / MALI_FAU_WORD_SIZE;
   if (!BITSET_TEST(vs->fau.used_sysvals, w))
      return -1;
   unsigned rank = 0;
   for (unsigned i = 0; i < w; i++)
      rank += BITSET_TEST(vs->fau.used_sysvals, i);
   return rank * MALI_FAU_WORD_SIZE;
}

/* The per-instance attributes whose offset depends on firstInstance. */
static uint32_t
instance_attribs(const struct mali_gfx_draw_state *d)
{
   const struct vk_vertex_input_state *vi = d->vi;
   uint32_t mask = 0;
   u_foreach_bit(i, vi->attributes_valid & BITFIELD_MASK(MALI_MAX_VS_ATTRIBS)) {
      const uint32_t binding = vi->attributes[i].binding;
      if (vi->bindings[binding].input_rate == VK_VERTEX_INPUT_RATE_INSTANCE &&
          d->vi_strides[binding])
         mask |= BITFIELD_BIT(i);
   }
   return mask;
}

/* acc += fi * stride, mod 2^32, for the fi in a register: v11's stream has
 * no multiply or shift, so each bit of fi that can change the product is
 * tested with AND and adds stride << bit. */
static void
emit_mul_add(struct cs_builder *b, struct cs_index acc, struct cs_index fi, uint32_t stride,
             struct cs_index bit, struct cs_index tmp)
{
   const unsigned bits = 32 - ffs(stride) + 1;   /* stride << i is 0 from here */
   for (unsigned i = 0; i < bits; i++) {
      cs_move32_to(b, bit, 1u << i);
      cs_and32(b, tmp, fi, bit);
      cs_if(b, MALI_CS_CONDITION_NEQUAL, tmp)
         cs_add_imm32(b, acc, acc, (int32_t)(stride << i));
   }
   /* Close the last if here: the caller's own if ends right after. */
   cs_flush_pending_if(b);
}

/* Invalidate the read-only caches the vertex jobs read the FAU blocks and
 * attribute descriptors through, before the stream stores new values into
 * copies a job may have read already (in an earlier run of this command
 * buffer, or an earlier lap of the ring), and wait for it. */
static void
emit_invalidate_ro(struct cs_builder *b, uint32_t wait_sb_mask)
{
   struct cs_index flush_id = mali_cs_scratch_reg32(b, 9);
   cs_move32_to(b, flush_id, 0);
   cs_flush_caches(b, MALI_CS_FLUSH_MODE_NONE, MALI_CS_FLUSH_MODE_NONE,
                   MALI_CS_OTHER_FLUSH_MODE_INVALIDATE, flush_id,
                   cs_defer(wait_sb_mask, MALI_SB_IMM_FLUSH));
   cs_wait_slots(b, wait_sb_mask | MALI_SB_MASK(MALI_SB_IMM_FLUSH));
}

/*
 * One indirect draw command, or a loop of them: max_draws commands stride
 * bytes apart from params, or with count_va the smaller of the count there
 * and max_draws.
 */
static void
draw_indirect(struct mali_cmd_buffer *cmd, bool indexed, uint64_t params, uint32_t stride,
              uint64_t count_va, uint32_t max_draws)
{
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   struct mali_render_state *r = &cmd->gfx.render;

   /* The count is compared as a signed word in the stream. */
   max_draws = MIN2(max_draws, INT32_MAX);
   if (!max_draws)
      return;
   if (!r->active || !d->pipeline || !d->pipeline->vs) {
      vk_logw(VK_LOG_OBJS(cmd), "malisx2: draw outside a render pass or without a pipeline");
      return;
   }
   if (vk_command_buffer_has_error(&cmd->vk))
      return;
   if (mali_gfx_discards(cmd))
      return;

   /* The state of a draw with first vertex 0 and first instance 0: what
    * the stream then stores into the per-draw copies is the rest. */
   const struct mali_draw_info di = {
      .count = 1,
      .instance_count = 1,
      .indexed = indexed,
   };
   struct idvs_batch e;
   e.d = d;
   e.n = 0;
   if (!prepare_draw(cmd, &di, &e)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   const struct mali_shader *vs = d->pipeline->vs;
   const int sv_off = vs_draw_sysval_offset(vs);
   const uint32_t attribs = instance_attribs(d);

   /* One ring entry: [FAU block][resource table][driver set], each part
    * only when it changes from draw to draw. */
   const uint32_t fau_bytes = sv_off >= 0 ? ALIGN_POT(vs->fau.total_count * 8, 64) : 0;
   uint32_t vb_offset = 0, vb_count = 0, drv_count = 0, rt_count = 0;
   uint32_t rt_off = fau_bytes, drv_off = fau_bytes;
   uint32_t entry = fau_bytes;
   if (attribs) {
      drv_count = mali_gfx_vs_drv_count(vs, d->vi, &vb_offset, &vb_count);
      rt_count = mali_gfx_res_table_count(vs);
      drv_off = rt_off + ALIGN_POT(rt_count * MALI_RESOURCE_SIZE, 64);
      entry = drv_off + ALIGN_POT(drv_count * MALI_DESCRIPTOR_SIZE, 64);
   }
   /* Store offsets are 16-bit. */
   assert(entry < 32768);

   uint32_t ring = 0;
   struct mali_ptr p = {0};
   if (entry) {
      ring = MIN2(max_draws, MAX2(INDIRECT_RING_BYTES / entry, 1));
      p = mali_cmd_alloc(cmd, (uint64_t)ring * entry, 64);
      if (!p.cpu)
         return;

      struct mali_graphics_sysvals sv;
      mali_gfx_fill_sysvals(cmd, &sv);
      for (uint32_t k = 0; k < ring; k++) {
         uint8_t *cpu = (uint8_t *)p.cpu + (uint64_t)k * entry;
         const uint64_t gpu = p.gpu + (uint64_t)k * entry;
         if (sv_off >= 0)
            MALI_PER_ARCH(gfx_fau_fill)(vs, &sv, cmd->push_constants,
                                        sizeof(cmd->push_constants), gpu, cpu);
         if (attribs) {
            mali_gfx_pack_vs_drv(cmd, vs, d->vi, cpu + drv_off, vb_offset, vb_count);
            mali_gfx_pack_res_table(vs, &cmd->gfx.desc, gpu + drv_off, drv_count, rt_count,
                                    cpu + rt_off);
         }
      }
   }
   const uint64_t fau0 = p.gpu | ((uint64_t)vs->fau.total_count << 56);
   const uint64_t srt0 = (p.gpu + rt_off) | rt_count;
   if (sv_off >= 0)
      batch_move64(&e, IDVS(FAU_0), fau0);
   if (attribs)
      batch_move64(&e, IDVS(SRT_0), srt0);
   if (!indexed)
      batch_move32(&e, IDVS(INDEX_OFFSET), 0);

   struct mali_primitive_flags_packed ovr;
   pan_pack_nodefaults(&ovr, PRIMITIVE_FLAGS, cfg) {
      cfg.index_type = indexed ? mali_gfx_index_type(d->ib.index_size) : MALI_INDEX_TYPE_NONE;
   }

   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_VERTEX_TILER);
   batch_flush(b, &e);
   /* The scratch registers below may be the sources of the uniform-buffer
    * copies' stores (emit_ubo_push). */
   cs_flush_stores(b);
   if (r->ubo_stores) {
      cs_wait_slot(b, MALI_SB_LS);
      r->ubo_stores = false;
   }

   uint32_t mh = 0;
   if (unlikely(cmd->measure) && cmd->measure->draws) {
      /* The counts are in the buffer, not known here. */
      mh = mali_measure_begin(cmd, MALI_MEASURE_DRAW, MALI_SUBQUEUE_VERTEX_TILER, cmd->draws);
      mali_measure_info(cmd, mh, 0, 0, indexed, cmd->passes - 1);
      uint64_t key = 0;
      if (d->pipeline->fs)
         memcpy(&key, d->pipeline->fs->key, sizeof(key));
      mali_measure_shader(cmd, mh, key);
      mali_measure_extra(cmd, mh, d->blend_shader_mask);
   }

   const struct cs_index addr = mali_cs_scratch_reg64(b, 0);   /* the command */
   const struct cs_index base = mali_cs_scratch_reg64(b, 2);   /* the ring entry */
   const struct cs_index count = mali_cs_scratch_reg32(b, 4);
   const struct cs_index tmp = mali_cs_scratch_reg32(b, 5);
   const struct cs_index left = mali_cs_scratch_reg32(b, 6);   /* entries left in the ring */
   const struct cs_index acc = mali_cs_scratch_reg32(b, 7);
   const struct cs_index bit = mali_cs_scratch_reg32(b, 8);
   const struct cs_index first_instance = cs_reg32(b, IDVS(INSTANCE_OFFSET));
   const bool laps = entry && ring < max_draws;

   if (count_va) {
      cs_move64_to(b, addr, count_va);
      cs_load32_to(b, count, addr, 0);
      cs_move32_to(b, tmp, max_draws);
      cs_umin32(b, count, count, tmp);
   } else {
      cs_move32_to(b, count, max_draws);
   }
   cs_move64_to(b, addr, params);
   if (entry) {
      cs_move64_to(b, base, p.gpu);
      emit_invalidate_ro(b, 0);
   }
   if (laps)
      cs_move32_to(b, left, ring);

   cs_while(b, MALI_CS_CONDITION_GREATER, count) {
      if (laps) {
         /* Back to the first copy, once the draws that read the copies
          * are done. */
         cs_if(b, MALI_CS_CONDITION_EQUAL, left) {
            emit_invalidate_ro(b, cmd->dev->csf->sb.all_iters_mask);
            cs_move64_to(b, base, p.gpu);
            if (sv_off >= 0)
               cs_move64_to(b, cs_reg64(b, IDVS(FAU_0)), fau0);
            if (attribs)
               cs_move64_to(b, cs_reg64(b, IDVS(SRT_0)), srt0);
            cs_move32_to(b, left, ring);
         }
      }

      /* VkDrawIndexedIndirectCommand is the five registers in order;
       * VkDrawIndirectCommand has no firstIndex, so its last two words go
       * to VERTEX_OFFSET and INSTANCE_OFFSET by a second load. */
      if (indexed) {
         cs_load_to(b, cs_reg_tuple(b, IDVS(INDEX_COUNT), 5), addr, BITFIELD_MASK(5), 0);
      } else {
         cs_load_to(b, cs_reg_tuple(b, IDVS(INDEX_COUNT), 2), addr, BITFIELD_MASK(2), 0);
         cs_load_to(b, cs_reg_tuple(b, IDVS(VERTEX_OFFSET), 2), addr, BITFIELD_MASK(2), 8);
      }

      if (sv_off >= 0)
         cs_store(b, cs_reg_tuple(b, IDVS(VERTEX_OFFSET), 2), base, BITFIELD_MASK(2), sv_off);
      u_foreach_bit(i, attribs) {
         const struct vk_vertex_attribute_state *a = &d->vi->attributes[i];
         /* The offset a direct draw with this first instance packs
          * (mali_gfx_pack_attrib). */
         cs_flush_stores(b);   /* acc may still be a store's source */
         cs_move32_to(b, acc, a->offset);
         cs_if(b, MALI_CS_CONDITION_NEQUAL, first_instance)
            emit_mul_add(b, acc, first_instance, d->vi_strides[a->binding], bit, tmp);
         cs_store32(b, acc, base, drv_off + i * MALI_DESCRIPTOR_SIZE + 2 * 4);
      }
      cs_flush_stores(b);

      /* Zero-based instance IDs, as for a direct draw. */
      cs_move32_to(b, first_instance, 0);

      cs_umin32(b, tmp, cs_reg32(b, IDVS(INDEX_COUNT)), cs_reg32(b, IDVS(INSTANCE_COUNT)));
      cs_if(b, MALI_CS_CONDITION_NEQUAL, tmp) {
         if (r->td_count > 1) {
            struct cs_index ctx = cs_reg64(b, IDVS(TILER_CTX));
            for (uint32_t i = 0; i < r->td_count; i++) {
               if (i)
                  cs_add_imm64(b, ctx, ctx, pan_size(TILER_CONTEXT));
               emit_run_idvs(b, ovr.opaque[0]);
            }
            cs_move64_to(b, ctx, r->tiler);
         } else {
            emit_run_idvs(b, ovr.opaque[0]);
         }
      }

      cs_add_imm32(b, count, count, -1);
      cs_add_imm64(b, addr, addr, stride);
      if (entry) {
         cs_add_imm64(b, base, base, entry);
         if (sv_off >= 0)
            cs_add_imm64(b, cs_reg64(b, IDVS(FAU_0)), cs_reg64(b, IDVS(FAU_0)), entry);
         if (attribs)
            cs_add_imm64(b, cs_reg64(b, IDVS(SRT_0)), cs_reg64(b, IDVS(SRT_0)), entry);
      }
      if (laps)
         cs_add_imm32(b, left, left, -1);
   }

   /* The registers the loop changed behind the draws' record, and the
    * blocks the next draw has to build again: the ones here hold the last
    * command's values. */
   d->regs_valid &= ~(BITFIELD64_RANGE(IDVS(INDEX_COUNT), 5) |
                      BITFIELD64_RANGE(IDVS(FAU_0), 2) | BITFIELD64_RANGE(IDVS(SRT_0), 2));
   if (sv_off >= 0)
      d->dirty |= MALI_GFX_DIRTY_VS_FAU;
   if (attribs)
      d->dirty |= MALI_GFX_DIRTY_VS_SRT;

   r->draws++;
   cmd->draws++;
   if (unlikely(mh))
      mali_measure_end(cmd, mh);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdDrawIndirect)(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                               VkDeviceSize offset, uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_buffer, buffer, _buffer);
   draw_indirect(cmd, false, vk_buffer_address(buffer, offset), stride, 0, drawCount);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdDrawIndexedIndirect)(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                                      VkDeviceSize offset, uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_buffer, buffer, _buffer);
   draw_indirect(cmd, true, vk_buffer_address(buffer, offset), stride, 0, drawCount);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdDrawIndirectCount)(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                                    VkDeviceSize offset, VkBuffer countBuffer,
                                    VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                                    uint32_t stride)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_buffer, buffer, _buffer);
   VK_FROM_HANDLE(vk_buffer, count, countBuffer);
   draw_indirect(cmd, false, vk_buffer_address(buffer, offset), stride,
                 vk_buffer_address(count, countBufferOffset), maxDrawCount);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdDrawIndexedIndirectCount)(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                                           VkDeviceSize offset, VkBuffer countBuffer,
                                           VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                                           uint32_t stride)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_buffer, buffer, _buffer);
   VK_FROM_HANDLE(vk_buffer, count, countBuffer);
   draw_indirect(cmd, true, vk_buffer_address(buffer, offset), stride,
                 vk_buffer_address(count, countBufferOffset), maxDrawCount);
}
