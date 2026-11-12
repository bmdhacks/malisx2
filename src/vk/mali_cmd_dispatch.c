/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Compute dispatch (vkCmdDispatch, vkCmdDispatchBase, and the internal
 * copy/fill shaders) on the command-stream frontend. Encoding follows panvk
 * (Mesa csf/panvk_vX_cmd_dispatch.c), with what our compiled shaders
 * need the command buffer to provide (the FAU block, resource table and
 * TSD are built by mali_cmd_state.h, shared with the job-manager back
 * half):
 *
 *  - FAU: the used words of struct mali_compute_sysvals, then the used
 *    push-constant words, then the constants the compiler promoted
 *    (pan_fau_foreach_imm); pointer | word count << 56 in the FAU register
 *    (the v11 packing);
 *  - resource table: entry 0 the driver set (the dummy sampler, then the
 *    dynamic buffers in the shader's dyn_bufs order with their offsets),
 *    entry N + 1 descriptor set N;
 *  - TSD: a Local Storage descriptor with the command buffer's TLS buffer
 *    and, for shaders with shared memory, a per-dispatch WLS buffer;
 *  - WG_SIZE, JOB_OFFSET (0: the base workgroup is a system value), JOB_SIZE,
 *    and RUN_COMPUTE with the task axis/increment panvk computes, register
 *    group 0 (the blob uses group 3; either is valid).
 *
 * The iterator scoreboard entry moves on before each RUN_COMPUTE
 * (cs_iter_sb_update: NEXT_SB_ENTRY on v11, SET_SB_ENTRY with the entry
 * known at record time on v10), so consecutive dispatches can overlap;
 * barriers and the end of the command buffer wait for them.
 *
 * vkCmdDispatchIndirect loads the three counts from the buffer into the
 * JOB_SIZE registers, stores them into the FAU words the shader reads as
 * num_work_groups (panvk csf cmd_dispatch_shader), and runs the job with
 * task axis X: RUN_COMPUTE_INDIRECT has been seen to hang the GPU, and a
 * task increment larger than the count along the axis is clamped by the
 * hardware. A count of 0 is not given to the hardware: the stream branches
 * over RUN_COMPUTE when any of the three is 0, as Vulkan says such a
 * dispatch does nothing.
 */

#include "mali_cmd_state.h"
#include "mali_measure.h"

#include "mali_arch.h"
#include "mali_queue.h"
#include "vk_buffer.h"

#define COMPUTE_SRT MALI_COMPUTE_SR_SRT_0
#define COMPUTE_FAU MALI_COMPUTE_SR_FAU_0
#define COMPUTE_SPD MALI_COMPUTE_SR_SPD_0
#define COMPUTE_TSD MALI_COMPUTE_SR_TSD_0

/* ---------------------------------------------------------------------- */
/* Encoding                                                                */

/* cs_iter_sb_update with nothing inside: select the next iterator
 * scoreboard entry for the next asynchronous job; on v11 also make it the
 * indirect wait mask, and keep the current one out of the stream mask. */
static void
next_iter_sb(struct mali_cmd_buffer *cmd, struct cs_builder *b)
{
#if PAN_ARCH == 10
   (void)b;
   mali_cmd_next_iter_sb(cmd, MALI_SUBQUEUE_COMPUTE);
#else
   const struct mali_csf_device *csf = cmd->dev->csf;
   struct cs_index next_sb = mali_cs_scratch_reg32(b, 0);
   struct cs_index sb_mask = mali_cs_scratch_reg32(b, 1);

   cs_next_sb_entry(b, next_sb, MALI_CS_SCOREBOARD_TYPE_ENDPOINT,
                    MALI_CS_NEXT_SB_ENTRY_FORMAT_INDEX);
   cs_move32_to(b, sb_mask, 0);
   cs_bit_set32(b, sb_mask, sb_mask, next_sb);
   cs_set_state(b, MALI_CS_SET_STATE_TYPE_SB_MASK_WAIT, sb_mask);
   cs_move32_to(b, sb_mask, csf->sb.all_iters_mask);
   cs_bit_clear32(b, sb_mask, sb_mask, next_sb);
   cs_set_state(b, MALI_CS_SET_STATE_TYPE_SB_MASK_STREAM, sb_mask);
#endif
}

/* indirect: GPU address of a VkDispatchIndirectCommand, 0 for a direct
 * dispatch of `groups`. The groups of an indirect dispatch are given as
 * {1, 1, 1} here, where they only fill the FAU words the stream overwrites
 * with the real counts. */
static void
dispatch(struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
         const struct mali_desc_state *desc, const void *push, uint32_t push_size,
         const uint32_t base[3], const uint32_t groups[3], const uint32_t (*textures)[8],
         unsigned texture_count, uint64_t indirect)
{
   if (!cs || !cs->spd[MALI_SPD_MAIN])
      return;
   if (!indirect && (!groups[0] || !groups[1] || !groups[2]))
      return;

   const struct mali_thread_props tp = mali_thread_props(cmd->dev);

   uint64_t fau = mali_compute_fau(cmd, cs, push, push_size, base, groups);
   uint64_t srt = mali_compute_res_table(cmd, cs, desc, textures, texture_count);
   uint64_t tsd = mali_compute_tsd(cmd, cs, &tp, indirect ? NULL : groups);
   if (vk_command_buffer_has_error(&cmd->vk) || !srt || !tsd ||
       (cs->fau.total_count && !fau))
      return;

   unsigned axis, inc;
   mali_compute_task_axis(&tp, cs, indirect ? NULL : groups, &axis, &inc);

   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_COMPUTE);

   cs_move64_to(b, cs_reg64(b, COMPUTE_SRT), srt);
   cs_move64_to(b, cs_reg64(b, COMPUTE_FAU), fau);
   cs_move64_to(b, cs_reg64(b, COMPUTE_SPD), cs->spd[MALI_SPD_MAIN]);
   cs_move64_to(b, cs_reg64(b, COMPUTE_TSD), tsd);
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, GLOBAL_ATTRIBUTE_OFFSET), 0);

   struct mali_compute_size_workgroup_packed wg;
   pan_pack(&wg, COMPUTE_SIZE_WORKGROUP, cfg) {
      cfg.workgroup_size_x = cs->cs.local_size[0];
      cfg.workgroup_size_y = cs->cs.local_size[1];
      cfg.workgroup_size_z = cs->cs.local_size[2];
      cfg.allow_merging_workgroups = cs->info.cs.allow_merging_workgroups;
   }
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, WG_SIZE), wg.opaque[0]);

   /* The base workgroup is added by the shader (a system value); the job
    * offset does not apply to the workgroup ID on Mali (panvk). */
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_OFFSET_X), 0);
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_OFFSET_Y), 0);
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_OFFSET_Z), 0);
   if (indirect) {
      /* The counts, into the staging registers and into the FAU words
       * the shader reads. The stores are waited for before the job
       * starts (cs_flush_stores); the FAU block is this dispatch's own,
       * so a command buffer submitted again stores the counts again. */
      const struct cs_index addr = mali_cs_scratch_reg64(b, 0);
      cs_move64_to(b, addr, indirect);
      cs_load_to(b, cs_sr_reg_tuple(b, COMPUTE, JOB_SIZE_X, 3), addr, BITFIELD_MASK(3), 0);
      bool stores = false;
      for (unsigned i = 0; i < 3; i++) {
         const int off = mali_compute_fau_sysval_offset(
            cs, offsetof(struct mali_compute_sysvals, num_work_groups) + 4 * i);
         if (off < 0)
            continue;
         if (!stores)
            cs_move64_to(b, addr, fau & BITFIELD64_MASK(56));
         stores = true;
         cs_store32(b, cs_reg32(b, MALI_COMPUTE_SR_JOB_SIZE_X + i), addr, off);
      }
      cs_flush_stores(b);
      /* Loads still in flight when no store waited for them. */
      cs_flush_loads(b);
   } else {
      cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_SIZE_X), groups[0]);
      cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_SIZE_Y), groups[1]);
      cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_SIZE_Z), groups[2]);
   }

   /* Timing. */
   uint32_t mh = 0;
   if (unlikely(cmd->measure)) {
      mh = mali_measure_begin(cmd, MALI_MEASURE_DISPATCH, MALI_SUBQUEUE_COMPUTE,
                              cmd->dispatches);
      uint64_t key;
      memcpy(&key, cs->key, sizeof(key));
      /* The counts of an indirect dispatch are in the buffer, not known
       * here. */
      mali_measure_info(cmd, mh, indirect ? 0 : groups[0], indirect ? 0 : groups[1],
                        indirect ? 0 : groups[2],
                        cs->cs.local_size[0] * cs->cs.local_size[1] * cs->cs.local_size[2]);
      mali_measure_shader(cmd, mh, key);
   }

   next_iter_sb(cmd, b);
   if (indirect) {
      /* No job for a count of 0: the smallest of the three is 0 then. */
      const struct cs_index min = mali_cs_scratch_reg32(b, 2);
      cs_umin32(b, min, cs_sr_reg32(b, COMPUTE, JOB_SIZE_X), cs_sr_reg32(b, COMPUTE, JOB_SIZE_Y));
      cs_umin32(b, min, min, cs_sr_reg32(b, COMPUTE, JOB_SIZE_Z));
      cs_if(b, MALI_CS_CONDITION_NEQUAL, min)
         cs_run_compute(b, inc, axis, cs_shader_res_sel(0, 0, 0, 0));
   } else {
      cs_run_compute(b, inc, axis, cs_shader_res_sel(0, 0, 0, 0));
   }
   cmd->dispatches++;
   if (unlikely(mh))
      mali_measure_end(cmd, mh);
}

/*
 * An internal compute job on the vertex/tiler stream, for what a draw needs
 * computed before it runs (polygon mode line, mali_cmd_draw.c): the stream
 * waits for it before going on, so the next draw reads what it wrote. It
 * signals the pass's iterator entry like the draws, so the wait is for the
 * draws before it too. The compute staging registers are the IDVS ones, so
 * the draws' record of those is gone (every group dirty).
 */
void
MALI_PER_ARCH(cmd_vt_dispatch_shader)(struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
                                      const void *push, uint32_t push_size,
                                      const uint32_t groups[3])
{
   const uint32_t base[3] = {0, 0, 0};
   if (!cs || !cs->spd[MALI_SPD_MAIN] || !groups[0] || !groups[1] || !groups[2])
      return;
   uint64_t fau = mali_compute_fau(cmd, cs, push, push_size, base, groups);
   uint64_t srt = mali_compute_res_table(cmd, cs, NULL, NULL, 0);
   if (vk_command_buffer_has_error(&cmd->vk) || !srt || (cs->fau.total_count && !fau))
      return;
   MALI_PER_ARCH(cmd_vt_dispatch)(cmd, cs, srt, fau, groups);
}

/* The same with the resource table and FAU block given: a shader compiled
 * against the graphics system values (the geometry shader's compute
 * shaders, mali_cmd_draw.c). */
void
MALI_PER_ARCH(cmd_vt_dispatch)(struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
                               uint64_t srt, uint64_t fau, const uint32_t groups[3])
{
   if (!cs || !cs->spd[MALI_SPD_MAIN] || !groups[0] || !groups[1] || !groups[2])
      return;
   const struct mali_thread_props tp = mali_thread_props(cmd->dev);
   uint64_t tsd = mali_compute_tsd(cmd, cs, &tp, groups);
   if (vk_command_buffer_has_error(&cmd->vk) || !tsd)
      return;
   unsigned axis, inc;
   mali_compute_task_axis(&tp, cs, groups, &axis, &inc);

   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_VERTEX_TILER);
   cs_move64_to(b, cs_reg64(b, COMPUTE_SRT), srt);
   cs_move64_to(b, cs_reg64(b, COMPUTE_FAU), fau);
   cs_move64_to(b, cs_reg64(b, COMPUTE_SPD), cs->spd[MALI_SPD_MAIN]);
   cs_move64_to(b, cs_reg64(b, COMPUTE_TSD), tsd);
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, GLOBAL_ATTRIBUTE_OFFSET), 0);
   struct mali_compute_size_workgroup_packed wg;
   pan_pack(&wg, COMPUTE_SIZE_WORKGROUP, cfg) {
      cfg.workgroup_size_x = cs->cs.local_size[0];
      cfg.workgroup_size_y = cs->cs.local_size[1];
      cfg.workgroup_size_z = cs->cs.local_size[2];
      cfg.allow_merging_workgroups = cs->info.cs.allow_merging_workgroups;
   }
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, WG_SIZE), wg.opaque[0]);
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_OFFSET_X), 0);
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_OFFSET_Y), 0);
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_OFFSET_Z), 0);
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_SIZE_X), groups[0]);
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_SIZE_Y), groups[1]);
   cs_move32_to(b, cs_sr_reg32(b, COMPUTE, JOB_SIZE_Z), groups[2]);
   cs_run_compute(b, inc, axis, cs_shader_res_sel(0, 0, 0, 0));
   cs_wait_slots(b, cmd->dev->csf->sb.all_iters_mask);

   BITSET_ZERO(cmd->gfx.draw.regs_valid);
   cmd->gfx.draw.dirty |= MALI_GFX_DIRTY_ALL;
   cmd->dispatches++;
}

void
MALI_PER_ARCH(cmd_dispatch_shader)(struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
                         const struct mali_desc_state *desc, const void *push,
                         uint32_t push_size, const uint32_t base[3],
                         const uint32_t groups[3])
{
   dispatch(cmd, cs, desc, push, push_size, base, groups, NULL, 0, 0);
}

void
MALI_PER_ARCH(cmd_dispatch_meta)(struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
                       const void *push, uint32_t push_size, const uint32_t groups[3],
                       const uint32_t (*textures)[8], unsigned texture_count)
{
   const uint32_t base[3] = {0, 0, 0};
   dispatch(cmd, cs, NULL, push, push_size, base, groups, textures, texture_count, 0);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdDispatchBase)(VkCommandBuffer commandBuffer, uint32_t baseGroupX,
                               uint32_t baseGroupY, uint32_t baseGroupZ,
                               uint32_t groupCountX, uint32_t groupCountY,
                               uint32_t groupCountZ)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   const uint32_t base[3] = {baseGroupX, baseGroupY, baseGroupZ};
   const uint32_t groups[3] = {groupCountX, groupCountY, groupCountZ};
   MALI_PER_ARCH(cmd_dispatch_shader)(cmd, cmd->compute.shader, &cmd->compute.desc,
                            cmd->push_constants, sizeof(cmd->push_constants),
                            base, groups);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdDispatchIndirect)(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                                   VkDeviceSize offset)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_buffer, buffer, _buffer);
   const uint32_t base[3] = {0, 0, 0};
   const uint32_t groups[3] = {1, 1, 1};
   dispatch(cmd, cmd->compute.shader, &cmd->compute.desc, cmd->push_constants,
            sizeof(cmd->push_constants), base, groups, NULL, 0,
            vk_buffer_address(buffer, offset));
}
