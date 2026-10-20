/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * GPU timing of passes, dispatches and draws.
 *
 * Recording: a region is two STORE_STATE{Timestamp} instructions on one
 * subqueue's stream, each deferred on every iterator scoreboard entry
 * (panvk's end-of-work timestamps, panvk_vX_utrace.c, and the blob's own
 * STORE_STATE timestamp). A deferred store runs
 * when the jobs issued before it on that stream have finished, and the
 * stream goes on issuing meanwhile, so nothing waits for it. The begin
 * timestamp is thus "when the previous work on this subqueue finished, or
 * when this region was reached, whichever is later", and regions on one
 * subqueue do not overlap. The end store also waits for the begin store
 * (emit_timestamp). The stores signal the deferred-flush and deferred-sync
 * entries, which only the end of the command buffer waits for.
 *
 * The address goes in r82:83 (scratch 16), which no command code uses
 * (mali_cs.h). Slots are 16 bytes (begin, end) in blocks of 4 KiB of
 * command memory, zeroed when allocated, so a timestamp that was never
 * written reads 0.
 *
 * Reading: vkQueueSubmit queues the command buffers on the device's
 * pending list; each later submit reads those whose subqueues' done slots
 * show the submit finished. A reset or destroy of a command buffer reads
 * it first (its work has completed by Vulkan's rules), and so does
 * vkDestroyDevice.
 */

#include "mali_measure.h"

#include <string.h>

#include "vk_queue.h"

#include "mali_cmd_buffer.h"
#include "mali_queue.h"
#include "mali_vk.h"

#define SLOT_BLOCK 4096

void
mali_measure_cmd_create(struct mali_cmd_buffer *cmd)
{
   struct mali_measure *m = cmd->dev->measure;
   if (!m || !m->cfg.timing)
      return;
   struct mali_measure_cmd *mc = calloc(1, sizeof(*mc));
   if (!mc)
      return;
   mc->m = m;
   mc->draws = m->cfg.draws;
   list_inithead(&mc->link);
   util_dynarray_init(&mc->regions, NULL);
   cmd->measure = mc;
}

/* With m->lock held. */
void
mali_measure_harvest(struct mali_measure *m, struct mali_measure_cmd *mc)
{
   if (mc->active) {
      util_dynarray_foreach(&mc->regions, struct mali_measure_region, r)
         mali_measure_row(m, mc, r);
   }
   list_del(&mc->link);
   list_inithead(&mc->link);
   mc->pending = false;
}

/*
 * completed(): has every region mc holds finished on the device? dev->jm
 * (non-NULL only on a job-manager device) asks the JM queue's completed
 * submission counter (mali_jm_measure_reached, mali_jm_queue.c -- jd is
 * only ever passed back to it, never dereferenced here, so this needs no
 * job-manager type); a CSF device (dev->csf) checks the subqueues'
 * sync-object "done" slots mc->sq_mask names, as before. Neither path
 * touches cmd, so this is safe for a mc created by either frontend.
 */
static bool
completed(struct mali_measure *m, const struct mali_measure_cmd *mc)
{
   if (m->dev->jm) {
      const uint32_t arch = mali_device_physical(m->dev)->arch;
      return mali_jm_arch_dispatch(arch, jm_measure_reached, m->dev->jm, mc->submit);
   }
   struct mali_csf_queue *q = m->dev->csf ? m->dev->csf->queue : NULL;
   if (!q)
      return true;
   volatile struct mali_cs_sync64 *done = mali_queue_done(q);
   u_foreach_bit(i, mc->sq_mask) {
      if (done[i].seqno < mc->submit)
         return false;
   }
   return true;
}

void
mali_measure_collect(struct mali_measure *m, bool force)
{
   list_for_each_entry_safe(struct mali_measure_cmd, mc, &m->pending, link) {
      if (force || completed(m, mc))
         mali_measure_harvest(m, mc);
   }
   if (m->timing)
      fflush(m->timing);
}

static void
cmd_clear(struct mali_cmd_buffer *cmd)
{
   struct mali_measure_cmd *mc = cmd->measure;
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
mali_measure_cmd_reset(struct mali_cmd_buffer *cmd)
{
   if (cmd->measure)
      cmd_clear(cmd);
}

void
mali_measure_cmd_destroy(struct mali_cmd_buffer *cmd)
{
   struct mali_measure_cmd *mc = cmd->measure;
   if (!mc)
      return;
   cmd_clear(cmd);
   util_dynarray_fini(&mc->regions);
   free(mc);
   cmd->measure = NULL;
}

/*
 * begin: wait for the iterators, signal the deferred-flush entry (which no
 * command code uses). end: wait for the iterators and for the begin store,
 * signal the deferred-sync entry. Without the wait on the begin store the
 * two deferred stores of a short region completed in either order on the
 * device (end up to 7 us before begin).
 */
static void
emit_timestamp(struct mali_cmd_buffer *cmd, enum mali_subqueue sq, uint64_t va, bool end)
{
   const struct mali_csf_device *csf = cmd->dev->csf;
   struct cs_builder *b = mali_cmd_cs(cmd, sq);
   struct cs_index addr = mali_cs_scratch_reg64(b, MALI_CS_SCRATCH_MEASURE);

   const struct cs_async_op op =
      end ? cs_defer(csf->sb.all_iters_mask | MALI_SB_MASK(MALI_SB_DEFERRED_FLUSH),
                     MALI_SB_DEFERRED_SYNC)
          : cs_defer(csf->sb.all_iters_mask, MALI_SB_DEFERRED_FLUSH);
   cs_move64_to(b, addr, va);
   cs_store_state(b, addr, 0, MALI_CS_STATE_TIMESTAMP, op);
}

/*
 * cmd->measure's region allocation is specific to this file's (CSF,
 * v11-fixed) struct mali_cmd_buffer layout and mali_cmd_alloc(): the
 * job-manager back half has its own copy of this logic in timing_jm.c
 * against its own struct mali_cmd_buffer (mali_jm.h), because a function
 * compiled once here would read cmd->measure, and call mali_cmd_alloc(),
 * at this file's (v11) offsets and definition even when handed a
 * job-manager cmd pointer. Only mali_measure_harvest and completed() are
 * shared (above): they never touch cmd, only the frontend-neutral struct
 * mali_measure_cmd a cmd->measure already points at.
 */
uint32_t
mali_measure_begin(struct mali_cmd_buffer *cmd, enum mali_measure_kind kind,
                   enum mali_subqueue sq, uint32_t index)
{
   struct mali_measure_cmd *mc = cmd->measure;

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
      .sq = sq,
      .index = index,
      .slot = mc->slots,
      .slot_va = mc->slots_va,
   };
   mc->slots += 2;
   mc->slots_va += 16;
   mc->slots_left--;

   util_dynarray_append_typed(&mc->regions, struct mali_measure_region, r);
   emit_timestamp(cmd, sq, r.slot_va, false);
   return util_dynarray_num_elements(&mc->regions, struct mali_measure_region);
}

static struct mali_measure_region *
region(struct mali_cmd_buffer *cmd, uint32_t handle)
{
   struct mali_measure_cmd *mc = cmd->measure;
   if (!mc || !handle ||
       handle > util_dynarray_num_elements(&mc->regions, struct mali_measure_region))
      return NULL;
   return util_dynarray_element(&mc->regions, struct mali_measure_region, handle - 1);
}

void
mali_measure_end(struct mali_cmd_buffer *cmd, uint32_t handle)
{
   struct mali_measure_region *r = region(cmd, handle);
   if (r)
      emit_timestamp(cmd, r->sq, r->slot_va + 8, true);
}

void
mali_measure_info(struct mali_cmd_buffer *cmd, uint32_t handle, uint32_t a, uint32_t b,
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
mali_measure_extra(struct mali_cmd_buffer *cmd, uint32_t handle, uint32_t extra)
{
   struct mali_measure_region *r = region(cmd, handle);
   if (r)
      r->extra = extra;
}

void
mali_measure_shader(struct mali_cmd_buffer *cmd, uint32_t handle, uint64_t key)
{
   struct mali_measure_region *r = region(cmd, handle);
   if (r)
      r->shader = key;
}

void
mali_measure_submit(struct mali_device *dev, struct mali_queue *queue,
                    struct vk_queue_submit *submit, uint64_t seqno)
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
      struct mali_measure_cmd *mc = cmd->measure;
      if (!mc || !util_dynarray_num_elements(&mc->regions, struct mali_measure_region))
         continue;
      if (mc->pending) {
         /* Submitted again before its last results were read: read them
          * if they are there, else they are lost. */
         if (completed(m, mc)) {
            mali_measure_harvest(m, mc);
         } else {
            m->dropped++;
            list_del(&mc->link);
            mc->pending = false;
         }
      }
      mc->submit = seqno;
      mc->submit_pos = c;
      mc->sq_mask = mali_cmd_buffer_subqueue_mask(cmd);
      mc->active = active;
      mc->pending = true;
      list_addtail(&mc->link, &m->pending);
   }
   pthread_mutex_unlock(&m->lock);
}
