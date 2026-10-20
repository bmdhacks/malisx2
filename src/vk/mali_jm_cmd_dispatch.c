/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Compute dispatch on the job manager (v7 and v9): one Compute job per
 * dispatch in the open batch's vtc chain (Mesa gallium pan_jm.c
 * jm_launch_grid; the T820 blob writes the same job). The FAU block,
 * resource table and Local Storage descriptor come from the builders
 * shared with the v11 back half (mali_cmd_state.h); what differs on v9,
 * where the job is 128 bytes, is where they go:
 *
 *  - workgroup size, counts and task axis/increment in the Compute
 *    payload; offsets 0, as on v11 (the base workgroup is a system value);
 *  - one Shader Environment: the resource table, the shader, the TSD, and
 *    the FAU block as a pointer and a separate 8-bit count of 64-bit words
 *    (the builders return v11's packed FAU word, address | count << 56);
 *  - no Barrier bit of its own: dispatches without a barrier between them
 *    may overlap, as Vulkan allows; a recorded barrier puts the bit (and a
 *    Cache Flush job) in front of the next job (mali_jm_cmd_buffer.c).
 *
 * The internal dispatches (cmd_dispatch_shader, cmd_dispatch_meta) are the
 * transfers done by compute (copies, fills, updates) and meet the barriers
 * whose destination is only the transfer stages; vkCmdDispatch does not.
 *
 * On v7 (Bifrost) the Compute job is 192 bytes and has no Shader
 * Environment: its Draw section points at the shader's Renderer State
 * Descriptor and at descriptor tables built for each dispatch
 * (mali_jm_desc_v7.c; the v7 half of dispatch below).
 */

#include "mali_cmd_state.h"

#include "mali_measure.h"
#if PAN_ARCH < 9
#include "mali_rsd_v7.h"
#endif

static void
dispatch(struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
         const struct mali_desc_state *desc, const void *push, uint32_t push_size,
         const uint32_t base[3], const uint32_t groups[3], const uint32_t (*textures)[8],
         unsigned texture_count, bool xfer)
{
   if (!cs || !cs->spd[MALI_SPD_MAIN])
      return;
   if (!groups[0] || !groups[1] || !groups[2])
      return;

#if PAN_ARCH < 9
   /*
    * As panvk_v7's cmd_dispatch (jm/panvk_vX_cmd_dispatch.c): the workgroup
    * size and count as Invocation shifts, the job task split in
    * Parameters, and a Draw section pointing at the RSD, the FAU block, the
    * Local Storage descriptor and the descriptor tables. The tables'
    * entries from the sets are written by a copy job in front, which this
    * job depends on. The tables are built anew for every dispatch: compute
    * keeps no dirty state.
    *
    * Internal dispatches that read textures (the AFBC plane copy,
    * mali_cmd_image.c) use no descriptor sets and name texture i of
    * `textures` and sampler 0 by literal index. The shader's RSD counts no
    * textures, so such a dispatch packs an RSD of its own with the counts
    * of its Texture table and its one nearest sampler.
    */
   const struct mali_thread_props tp = mali_thread_props(cmd->dev);
   struct mali_v7_desc_tables tables;
   if (!mali_v7_desc_prepare(cmd, cs, desc, true, &tables))
      return;
   uint64_t rsd = cs->spd[MALI_SPD_MAIN];
   if (texture_count) {
      assert(!desc && !cs->desc.used_set_mask && !cs->v7.map_count);
      struct mali_ptr tex = mali_cmd_alloc(cmd, (uint64_t)texture_count * pan_size(TEXTURE), 64);
      struct mali_ptr samp = mali_cmd_alloc(cmd, pan_size(SAMPLER), 64);
      struct mali_ptr state =
         mali_cmd_alloc(cmd, pan_size(RENDERER_STATE), pan_alignment(RENDERER_STATE));
      if (!tex.cpu || !samp.cpu || !state.cpu)
         return;
      for (unsigned i = 0; i < texture_count; i++)
         memcpy((uint8_t *)tex.cpu + i * pan_size(TEXTURE), textures[i], pan_size(TEXTURE));
      pan_cast_and_pack(samp.cpu, SAMPLER, cfg) {
         cfg.seamless_cube_map = false;
         cfg.normalized_coordinates = false;
         cfg.clamp_integer_array_indices = false;
         cfg.minify_nearest = true;
         cfg.magnify_nearest = true;
      }
      pan_cast_and_pack(state.cpu, RENDERER_STATE, cfg) {
         mali_v7_rsd_shader(&cs->info, mali_shader_code_va(cs), &cfg);
         cfg.shader.texture_count = texture_count;
         cfg.shader.sampler_count = 1;
      }
      tables.tables[MALI_V7_DESC_TABLE_TEXTURE] = tex.gpu;
      tables.tables[MALI_V7_DESC_TABLE_SAMPLER] = samp.gpu;
      rsd = state.gpu;
   }
   const uint64_t fau =
      mali_compute_fau(cmd, cs, &tables, push, push_size, base, groups) & BITFIELD64_MASK(56);
   const uint64_t tsd = mali_compute_tsd(cmd, cs, &tp, groups);
   if (vk_command_buffer_has_error(&cmd->vk) || !tsd || (cs->fau.total_count && !fau))
      return;
   /* The RSD's preload.uniform_count is the shader's FAU word count; the
    * block built here has as many words. */
   assert(cs->fau.total_count == DIV_ROUND_UP(cs->info.fau.count, 2));

   struct mali_jm_chain *c = xfer ? mali_jm_cmd_vtc_transfer(cmd) : mali_jm_cmd_vtc(cmd);
   if (!c)
      return;

   uint16_t copy_dep = 0;
   if (!mali_v7_desc_copy(cmd, c, cs, &tables, 0, &copy_dep))
      return;

   /* Timing. */
   uint32_t mh = 0;
   if (unlikely(cmd->jm.measure)) {
      mh = mali_jm_measure_begin(cmd, MALI_MEASURE_DISPATCH, c, MALI_JM_SLOT_VTC,
                                 cmd->dispatches);
      uint64_t key;
      memcpy(&key, cs->key, sizeof(key));
      mali_jm_measure_info(cmd, mh, groups[0], groups[1], groups[2],
                          cs->cs.local_size[0] * cs->cs.local_size[1] * cs->cs.local_size[2]);
      mali_jm_measure_shader(cmd, mh, key);
   }

   struct mali_ptr job = mali_jm_cmd_add_job(cmd, c, MALI_JOB_TYPE_COMPUTE,
                                             pan_size(COMPUTE_JOB), false, copy_dep);
   if (!job.cpu) {
      if (unlikely(mh))
         mali_jm_measure_end(cmd, mh, c);
      return;
   }

   mali_v7_compute_invocation(pan_section_ptr(job.cpu, COMPUTE_JOB, INVOCATION),
                              cs->cs.local_size, groups);
   pan_section_pack(job.cpu, COMPUTE_JOB, PARAMETERS, cfg) {
      cfg.job_task_split = mali_v7_compute_task_split(cs->cs.local_size);
   }
   pan_section_pack(job.cpu, COMPUTE_JOB, DRAW, cfg) {
      cfg.state = rsd;
      cfg.attributes = tables.img_attribs;
      cfg.attribute_buffers = tables.tables[MALI_V7_DESC_TABLE_IMG];
      cfg.thread_storage = tsd;
      cfg.uniform_buffers = tables.tables[MALI_V7_DESC_TABLE_UBO];
      cfg.push_uniforms = fau;
      cfg.textures = tables.tables[MALI_V7_DESC_TABLE_TEXTURE];
      cfg.samplers = tables.tables[MALI_V7_DESC_TABLE_SAMPLER];
   }
   cmd->dispatches++;
   if (unlikely(mh))
      mali_jm_measure_end(cmd, mh, c);
#else
   const struct mali_thread_props tp = mali_thread_props(cmd->dev);

   const uint64_t fau = mali_compute_fau(cmd, cs, push, push_size, base, groups);
   const uint64_t srt = mali_compute_res_table(cmd, cs, desc, textures, texture_count);
   const uint64_t tsd = mali_compute_tsd(cmd, cs, &tp, groups);
   if (vk_command_buffer_has_error(&cmd->vk) || !srt || !tsd ||
       (cs->fau.total_count && !fau))
      return;

   unsigned axis, inc;
   mali_compute_task_axis(&tp, cs, groups, &axis, &inc);

   struct mali_jm_chain *c = xfer ? mali_jm_cmd_vtc_transfer(cmd) : mali_jm_cmd_vtc(cmd);
   if (!c)
      return;

   /* Timing. */
   uint32_t mh = 0;
   if (unlikely(cmd->jm.measure)) {
      mh = mali_jm_measure_begin(cmd, MALI_MEASURE_DISPATCH, c, MALI_JM_SLOT_VTC,
                                 cmd->dispatches);
      uint64_t key;
      memcpy(&key, cs->key, sizeof(key));
      mali_jm_measure_info(cmd, mh, groups[0], groups[1], groups[2],
                          cs->cs.local_size[0] * cs->cs.local_size[1] * cs->cs.local_size[2]);
      mali_jm_measure_shader(cmd, mh, key);
   }

   struct mali_ptr job =
      mali_jm_cmd_add_job(cmd, c, MALI_JOB_TYPE_COMPUTE, pan_size(COMPUTE_JOB), false, 0);
   if (!job.cpu) {
      if (unlikely(mh))
         mali_jm_measure_end(cmd, mh, c);
      return;
   }

   pan_section_pack(job.cpu, COMPUTE_JOB, PAYLOAD, cfg) {
      cfg.workgroup_size_x = cs->cs.local_size[0];
      cfg.workgroup_size_y = cs->cs.local_size[1];
      cfg.workgroup_size_z = cs->cs.local_size[2];
      cfg.allow_merging_workgroups = cs->info.cs.allow_merging_workgroups;
      /* 14 bits; the builder's increment is a few workgroups per task. */
      cfg.task_increment = MIN2(inc, (1u << 14) - 1);
      cfg.task_axis = axis;
      cfg.workgroup_count_x = groups[0];
      cfg.workgroup_count_y = groups[1];
      cfg.workgroup_count_z = groups[2];
      cfg.compute.resources = srt;
      cfg.compute.shader = cs->spd[MALI_SPD_MAIN];
      cfg.compute.thread_storage = tsd;
      cfg.compute.fau = fau & BITFIELD64_MASK(56);
      cfg.compute.fau_count = fau >> 56;
   }
   cmd->dispatches++;
   if (unlikely(mh))
      mali_jm_measure_end(cmd, mh, c);
#endif /* PAN_ARCH >= 9 */
}

void
MALI_PER_ARCH(cmd_dispatch_shader)(struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
                                   const struct mali_desc_state *desc, const void *push,
                                   uint32_t push_size, const uint32_t base[3],
                                   const uint32_t groups[3])
{
   dispatch(cmd, cs, desc, push, push_size, base, groups, NULL, 0, true);
}

void
MALI_PER_ARCH(cmd_dispatch_meta)(struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
                                 const void *push, uint32_t push_size, const uint32_t groups[3],
                                 const uint32_t (*textures)[8], unsigned texture_count)
{
   const uint32_t base[3] = {0, 0, 0};
   dispatch(cmd, cs, NULL, push, push_size, base, groups, textures, texture_count, true);
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
   dispatch(cmd, cmd->compute.shader, &cmd->compute.desc, cmd->push_constants,
            sizeof(cmd->push_constants), base, groups, NULL, 0, false);
}
