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
#include "vk_buffer.h"
#if PAN_ARCH < 9
#include "mali_rsd_v7.h"
#endif

#if PAN_ARCH < 9
static void dispatch(struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
                     const struct mali_desc_state *desc, const void *push, uint32_t push_size,
                     const uint32_t base[3], const uint32_t groups[3],
                     const uint32_t (*textures)[8], unsigned texture_count, bool xfer,
                     uint64_t indirect, struct mali_jm_chain *chain);

/* GPU address of the 32-bit field `off` bytes into struct
 * mali_compute_sysvals in a dispatch's FAU block at fau (the used sysval
 * words first, in order: mali_compute_fau), 0 if the shader reads none. */
static uint64_t
fau_sysval_addr(const struct mali_shader *cs, uint64_t fau, unsigned off)
{
   const unsigned w = off / MALI_FAU_WORD_SIZE;
   if (!fau || w >= MALI_MAX_SYSVAL_FAUS || !BITSET_TEST(cs->fau.used_sysvals, w))
      return 0;
   unsigned rank = 0;
   for (unsigned i = 0; i < w; i++)
      rank += BITSET_TEST(cs->fau.used_sysvals, i);
   return fau + rank * MALI_FAU_WORD_SIZE + off % MALI_FAU_WORD_SIZE;
}

/*
 * The jobs of a v7 vkCmdDispatchIndirect, as panvk_v7's cmd_dispatch:
 * the dispatch's Compute job is allocated first and given to a patch job
 * (MALI_META_INDIRECT_DISPATCH, mali_cmd_copy.c), which waits for the
 * descriptor copy job (copy_dep) and has Suppress Prefetch set, so the
 * job manager reads the Compute job only after the patch wrote it. The
 * Compute job goes in behind it, Not Started until patched, depending on
 * it. fau is the FAU block's address, built with counts of 1.
 */
static void
dispatch_indirect(struct mali_cmd_buffer *cmd, struct mali_jm_chain *c,
                  const struct mali_shader *cs, uint64_t rsd,
                  const struct mali_v7_desc_tables *tables, uint64_t fau, uint64_t tsd,
                  uint16_t copy_dep, uint64_t indirect)
{
   const struct mali_shader *pc =
      MALI_PER_ARCH(cmd_meta_shader)(cmd, MALI_META_INDIRECT_DISPATCH);
   struct mali_ptr job = mali_cmd_alloc(cmd, pan_size(COMPUTE_JOB), 128);
   if (!pc || !job.cpu)
      return;

   const uint32_t one[3] = {1, 1, 1};
   struct mali_invocation_packed inv;
   mali_v7_compute_invocation(&inv, cs->cs.local_size, one);
   uint32_t inv_words[2];
   memcpy(inv_words, &inv, sizeof(inv_words));
   unsigned wgx_shift = 0;
   for (unsigned i = 0; i < 3; i++)
      wgx_shift += util_logbase2_ceil(cs->cs.local_size[i]);

   struct mali_meta_indirect_push push = {
      .cmd = indirect,
      .job = job.gpu,
      .inv = job.gpu + pan_section_offset(COMPUTE_JOB, INVOCATION),
      /* Counts of 1 pack as zero bits, so word 0 is the size half. */
      .inv0 = inv_words[0],
      .inv1 = inv_words[1] & ~((0x3fu << 16) | (0x3fu << 22)),
      .wgx_shift = wgx_shift,
   };
   for (unsigned i = 0; i < 3; i++)
      push.nwg[i] = fau_sysval_addr(
         cs, fau, offsetof(struct mali_compute_sysvals, num_work_groups) + 4 * i);

   /* The patch job goes onto the chain the copy job is on. Taking the
    * vtc chain again could close a batch that has just reached its job
    * limit, which would put the patch and Compute jobs in a different batch
    * from the copy job they depend on (job indices only mean something
    * within one chain). A few jobs past the limit are fine, as for the
    * other emitters that add several jobs per call. */
   const uint16_t before = c->index;
   const uint32_t zero[3] = {0, 0, 0};
   dispatch(cmd, pc, NULL, &push, sizeof(push), zero, one, NULL, 0, false, 0, c);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;
   /* The patch job is the chain's newest job. */
   if (c->index <= before || !c->prev_next) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_UNKNOWN);
      return;
   }
   const uint16_t patch_index = c->index;
   uint8_t *patch_hdr = (uint8_t *)c->prev_next - 24;
   uint32_t w;
   memcpy(&w, patch_hdr + 16, sizeof(w));
   w |= 1u << 11; /* Suppress Prefetch */
   memcpy(patch_hdr + 16, &w, sizeof(w));
   if (copy_dep) {
      memcpy(&w, patch_hdr + 20, sizeof(w));
      if (!(w & 0xffff)) {
         w |= copy_dep; /* Dependency 1 */
         memcpy(patch_hdr + 20, &w, sizeof(w));
      }
   }

   memcpy(pan_section_ptr(job.cpu, COMPUTE_JOB, INVOCATION), &inv, sizeof(inv));
   pan_section_pack(job.cpu, COMPUTE_JOB, PARAMETERS, cfg) {
      cfg.job_task_split = mali_v7_compute_task_split(cs->cs.local_size);
   }
   pan_section_pack(job.cpu, COMPUTE_JOB, DRAW, cfg) {
      cfg.state = rsd;
      cfg.attributes = tables->img_attribs;
      cfg.attribute_buffers = tables->tables[MALI_V7_DESC_TABLE_IMG];
      cfg.thread_storage = tsd;
      cfg.uniform_buffers = tables->tables[MALI_V7_DESC_TABLE_UBO];
      cfg.push_uniforms = fau;
      cfg.textures = tables->tables[MALI_V7_DESC_TABLE_TEXTURE];
      cfg.samplers = tables->tables[MALI_V7_DESC_TABLE_SAMPLER];
   }
   if (!mali_jm_cmd_add_job_at(cmd, c, job, MALI_JOB_TYPE_NOT_STARTED, false, patch_index))
      return;
   cmd->dispatches++;
}
#endif

static void
dispatch(struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
         const struct mali_desc_state *desc, const void *push, uint32_t push_size,
         const uint32_t base[3], const uint32_t groups[3], const uint32_t (*textures)[8],
         unsigned texture_count, bool xfer, uint64_t indirect, struct mali_jm_chain *chain)
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
   /* An indirect dispatch's count is not known here: no groups, which
    * gives its WLS the most instances the cores hold at once. */
   const uint64_t tsd = mali_compute_tsd(cmd, cs, &tp, indirect ? NULL : groups);
   if (vk_command_buffer_has_error(&cmd->vk) || !tsd || (cs->fau.total_count && !fau))
      return;
   /* The RSD's preload.uniform_count is the shader's FAU word count; the
    * block built here has as many words. */
   assert(cs->fau.total_count == DIV_ROUND_UP(cs->info.fau.count, 2));

   struct mali_jm_chain *c =
      chain ? chain : xfer ? mali_jm_cmd_vtc_transfer(cmd) : mali_jm_cmd_vtc(cmd);
   if (!c)
      return;

   uint16_t copy_dep = 0;
   if (!mali_v7_desc_copy(cmd, c, cs, &tables, 0, &copy_dep))
      return;

   if (indirect) {
      dispatch_indirect(cmd, c, cs, rsd, &tables, fau, tsd, copy_dep, indirect);
      return;
   }

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
   dispatch(cmd, cs, desc, push, push_size, base, groups, NULL, 0, true, 0, NULL);
}

void
MALI_PER_ARCH(cmd_dispatch_meta)(struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
                                 const void *push, uint32_t push_size, const uint32_t groups[3],
                                 const uint32_t (*textures)[8], unsigned texture_count)
{
   const uint32_t base[3] = {0, 0, 0};
   dispatch(cmd, cs, NULL, push, push_size, base, groups, textures, texture_count, true, 0,
            NULL);
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
            sizeof(cmd->push_constants), base, groups, NULL, 0, false, 0, NULL);
}

#if PAN_ARCH < 9
VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdDispatchIndirect)(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                                   VkDeviceSize offset)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_buffer, buffer, _buffer);
   const uint32_t base[3] = {0, 0, 0};
   const uint32_t groups[3] = {1, 1, 1};
   dispatch(cmd, cmd->compute.shader, &cmd->compute.desc, cmd->push_constants,
            sizeof(cmd->push_constants), base, groups, NULL, 0, false,
            vk_buffer_address(buffer, offset), NULL);
}
#endif
