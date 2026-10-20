/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * GPU timing of passes and dispatches on the job manager (v7 and v9): the
 * job-manager counterpart of timing.c.
 *
 * timing.c is compiled once, at this build's fixed CSF (v11) struct
 * mali_cmd_buffer layout and mali_cmd_alloc(); calling its functions with
 * a job-manager cmd pointer would read cmd->measure, and allocate command
 * memory, at the wrong offsets. This file is therefore its own
 * implementation of the pieces that touch cmd directly (region allocation
 * out of command memory, cmd->jm.measure, the pending-submit queue), built
 * at PAN_ARCH 7 and 9 against mali_jm.h's struct mali_cmd_buffer.
 * Everything else -- struct mali_measure_cmd / mali_measure_region, the
 * configuration and output writers in config.c, and mali_measure_harvest /
 * mali_measure_collect in timing.c -- is the one shared implementation both
 * frontends call; none of it touches cmd.
 *
 * Recording: a region is a System Timestamp Write Value job with the
 * header Barrier bit into the chain the caller names (mali_jm_cmd_write_value,
 * mali_jm_cmd_buffer.c): the open batch's vtc chain for a pass's
 * vertex/tiler side or a dispatch, the pass's current fragment segment
 * for its fragment side. The Barrier bit is what makes the timestamp wait
 * for every earlier job of that chain -- JM has no deferred-store
 * scoreboard to lean on the way the CSF side's STORE_STATE does, so the
 * ordering has to come from the job header itself (the blob sets the
 * same bit on its own Write Value jobs).
 *
 * Reading: the same discipline as CSF (vkQueueSubmit queues every command
 * buffer that has regions on the device's pending list; a command buffer
 * being reset or destroyed reads its own regions first, since Vulkan
 * guarantees its work has completed), except "has this submission's work
 * finished" is the job-manager queue's completed_seq counter rather than
 * per-subqueue sync-object slots (mali_jm_measure_reached, mali_jm_queue.c
 * -- also called from timing.c's completed(), so a command buffer
 * measured from either frontend is read correctly by the one shared
 * mali_measure_collect).
 */

#include "mali_measure.h"

#include <stdlib.h>
#include <string.h>

#include "mali_jm.h"

#define SLOT_BLOCK 4096

void
mali_jm_measure_cmd_create(struct mali_cmd_buffer *cmd)
{
   struct mali_measure *m = cmd->dev->measure;
   if (!m || !m->cfg.timing)
      return;
   struct mali_measure_cmd *mc = calloc(1, sizeof(*mc));
   if (!mc)
      return;
   mc->m = m;
   list_inithead(&mc->link);
   util_dynarray_init(&mc->regions, NULL);
   cmd->jm.measure = mc;
}

/* With m->lock not held: harvest takes it itself. */
static void
cmd_clear(struct mali_cmd_buffer *cmd)
{
   struct mali_measure_cmd *mc = cmd->jm.measure;
   struct mali_measure *m = mc->m;

   pthread_mutex_lock(&m->lock);
   if (mc->pending)
      mali_measure_harvest(m, mc);
   pthread_mutex_unlock(&m->lock);

   util_dynarray_clear(&mc->regions);
   mc->slots = NULL;
   mc->slots_va = 0;
   mc->slots_left = 0;
}

void
mali_jm_measure_cmd_reset(struct mali_cmd_buffer *cmd)
{
   if (cmd->jm.measure)
      cmd_clear(cmd);
}

void
mali_jm_measure_cmd_destroy(struct mali_cmd_buffer *cmd)
{
   struct mali_measure_cmd *mc = cmd->jm.measure;
   if (!mc)
      return;
   cmd_clear(cmd);
   util_dynarray_fini(&mc->regions);
   free(mc);
   cmd->jm.measure = NULL;
}

uint32_t
mali_jm_measure_begin(struct mali_cmd_buffer *cmd, enum mali_measure_kind kind,
                      struct mali_jm_chain *chain, uint8_t slot, uint32_t index)
{
   struct mali_measure_cmd *mc = cmd->jm.measure;

   if (!mc->slots_left) {
      struct mali_ptr p = mali_cmd_alloc(cmd, SLOT_BLOCK, 64);
      if (!p.cpu)
         return 0;
      memset(p.cpu, 0, SLOT_BLOCK);
      mc->slots = p.cpu;
      mc->slots_va = p.gpu;
      mc->slots_left = SLOT_BLOCK / 16;
   }

   struct mali_measure_region r = {
      .kind = kind,
      .sq = slot,
      .index = index,
      .slot = mc->slots,
      .slot_va = mc->slots_va,
   };
   mc->slots += 2;
   mc->slots_va += 16;
   mc->slots_left--;

   util_dynarray_append_typed(&mc->regions, struct mali_measure_region, r);
   mali_jm_cmd_write_value(cmd, chain, MALI_WRITE_VALUE_TYPE_SYSTEM_TIMESTAMP, r.slot_va, 0,
                           true);
   return util_dynarray_num_elements(&mc->regions, struct mali_measure_region);
}

static struct mali_measure_region *
region(struct mali_cmd_buffer *cmd, uint32_t handle)
{
   struct mali_measure_cmd *mc = cmd->jm.measure;
   if (!mc || !handle ||
       handle > util_dynarray_num_elements(&mc->regions, struct mali_measure_region))
      return NULL;
   return util_dynarray_element(&mc->regions, struct mali_measure_region, handle - 1);
}

void
mali_jm_measure_end(struct mali_cmd_buffer *cmd, uint32_t handle, struct mali_jm_chain *chain)
{
   struct mali_measure_region *r = region(cmd, handle);
   if (r)
      mali_jm_cmd_write_value(cmd, chain, MALI_WRITE_VALUE_TYPE_SYSTEM_TIMESTAMP,
                              r->slot_va + 8, 0, true);
}

void
mali_jm_measure_info(struct mali_cmd_buffer *cmd, uint32_t handle, uint32_t a, uint32_t b,
                     uint32_t c, uint32_t d)
{
   struct mali_measure_region *r = region(cmd, handle);
   if (r) {
      r->info[0] = a;
      r->info[1] = b;
      r->info[2] = c;
      r->info[3] = d;
   }
}

void
mali_jm_measure_shader(struct mali_cmd_buffer *cmd, uint32_t handle, uint64_t key)
{
   struct mali_measure_region *r = region(cmd, handle);
   if (r)
      r->shader = key;
}

void
mali_jm_measure_submit(struct mali_device *dev, struct vk_queue_submit *submit, uint64_t seqno)
{
   struct mali_measure *m = dev->measure;

   pthread_mutex_lock(&m->lock);
   m->submits = seqno;
   mali_measure_collect(m, false);
   mali_measure_rotate(m, seqno);

   const bool active = m->cfg.timing && mali_measure_window(m, seqno);
   for (uint32_t c = 0; c < submit->command_buffer_count; c++) {
      struct mali_cmd_buffer *cmd =
         container_of(submit->command_buffers[c], struct mali_cmd_buffer, vk);
      struct mali_measure_cmd *mc = cmd->jm.measure;
      if (!mc || !util_dynarray_num_elements(&mc->regions, struct mali_measure_region))
         continue;
      if (mc->pending) {
         /* Submitted again before its last results were read: read them
          * if they are there, else they are lost. */
         if (mali_jm_measure_reached(dev->jm, mc->submit)) {
            mali_measure_harvest(m, mc);
         } else {
            m->dropped++;
            list_del(&mc->link);
            mc->pending = false;
         }
      }
      mc->submit = seqno;
      mc->submit_pos = c;
      mc->active = active;
      mc->pending = true;
      list_addtail(&mc->link, &m->pending);
   }
   pthread_mutex_unlock(&m->lock);
}
