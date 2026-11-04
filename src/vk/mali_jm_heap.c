/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The tiler heap ring of the job-manager (v7 and v9) back half:
 * growable GPU regions (MALI_JM_HEAP_SLOTS at first, more when submissions
 * keep wrapping the ring; mali_jm.h) the tiler allocates polygon lists
 * from, and on v9 vertex packets too (mali_jm.h has the scheme). The
 * driver owns the memory and writes the Tiler Heap descriptor itself when
 * it records, as Mesa's gallium driver does; there is no JIT allocation
 * and no GPU job that sets the heap up. What the heap must look like to
 * the GPU:
 *
 *  - Size: the region's size; Base and Bottom: its start;
 *  - Top on v7: the region's end. v7's tiler fails a batch with
 *    OUT_OF_MEMORY (0x60) once its polygon lists would pass Top, and the
 *    region is backed only as far as the tiler has used it (measured on a
 *    G52 r1 with kbase r34p0: one triangle backs 2 MiB, four million
 *    small ones 40 MiB). With Top at the end of the first 2 MiB chunk, as
 *    on v9, ARMSX2's batches of upscaled passes ran out (issue #5);
 *  - Top on v9: the end of the first 2 MiB chunk, not the end of the
 *    region. The tiler allocates chunks from both the bottom and the top
 *    pointer and moves either on to a new chunk when its part is used up.
 *    A Top at the region's end would make the first top-side allocation
 *    fault at the region's last page, and a GROW_ON_GPF fault commits
 *    every page from the current commit up to the faulting one: the whole
 *    region.
 *    TILER_ALIGN_TOP with a 2 MiB commit and a 2 MiB growth step makes the
 *    kernel place the region so that start + commit is 2 MiB aligned, so
 *    start + 2 MiB is the end of the committed chunk;
 *  - word 0 on v9: Buffer, Tiler heap, 2 MiB chunks, dynamic partitioning
 *    (the genxml defaults, 0x329); v7's descriptor has no fields there.
 *
 * The GPU moves Bottom and Top in the descriptor as it allocates, so the
 * passes of a batch that share one descriptor continue from each other, and
 * a command buffer that runs again gets the descriptor restored first.
 *
 * Slots are taken round-robin by batch, at record time; the dependency on
 * the slot's previous user is the queue's business at submit
 * (mali_jm_queue.c, heap_last_frag), so command buffers recorded on other
 * threads or submitted out of order stay correct.
 */

#include "mali_cmd_state.h"

#include "util/log.h"
#include "vk_log.h"

void
mali_jm_heap_init(struct mali_jm_device *jd)
{
   simple_mtx_init(&jd->heap.lock, mtx_plain);
   jd->heap.next = 0;
   jd->heap.count = MALI_JM_HEAP_SLOTS;
   jd->heap.win_len = jd->heap.win_hits = jd->heap.win_max = 0;
}

void
mali_jm_heap_finish(struct mali_jm_device *jd)
{
   for (unsigned i = 0; i < MALI_JM_HEAP_SLOTS_MAX; i++) {
      if (jd->heap.bo[i].size)
         mali_kbase_free(jd->kb, &jd->heap.bo[i]);
   }
   simple_mtx_destroy(&jd->heap.lock);
}

/* A slot's memory: GPU-only and growable. On a 64-bit job-manager context
 * the kernel makes it SAME_VA, so mali_kbase_alloc maps it whole with no CPU
 * access (PROT_NONE) to get its address; each slot takes 256 MiB of the
 * process's address space and commits 2 MiB until the tiler grows it. */
static bool
slot_alloc(struct mali_jm_device *jd, struct mali_kbase_bo *bo)
{
   const struct mali_kbase_alloc_info ai = {
      .size = MALI_JM_HEAP_SLOT_SIZE,
      .commit_size = MALI_JM_HEAP_CHUNK,
      .extension_pages = MALI_JM_HEAP_CHUNK / KB_PAGE_SIZE,
      /* COHERENT_LOCAL (inner shareable on the GPU), as the kernel gives
       * its own JIT regions. On v9 the tiler reads the positions the
       * shader cores write into this memory during the same Malloc Vertex
       * job, which it does not see otherwise. On v7 the shader cores write
       * the varyings to command memory and only the tiler writes here (its
       * polygon lists); the flag stays as on the JIT regions. */
      .flags = MALI_KBASE_FLAGS_DEVICE_TRANSIENT | KB_MEM_GROW_ON_GPF | KB_MEM_TILER_ALIGN_TOP,
      .mem_class = MALI_KBASE_MEM_CLASS_DEVICE_TRANSIENT,
   };
   if (mali_kbase_alloc(jd->kb, &ai, bo) != MALI_KBASE_SUCCESS) {
      mesa_loge("malisx2: the tiler heap region (%llu MiB of VA) could not be allocated",
                (unsigned long long)(MALI_JM_HEAP_SLOT_SIZE >> 20));
      return false;
   }
   return true;
}

bool
mali_jm_heap_take(struct mali_cmd_buffer *cmd, struct mali_jm_batch *b)
{
   struct mali_jm_device *jd = cmd->dev->jm;
   struct mali_jm_heap *h = &jd->heap;

   simple_mtx_lock(&h->lock);
   const uint32_t slot = h->next;
   bool ok = h->bo[slot].size || slot_alloc(jd, &h->bo[slot]);
   if (ok)
      h->next = (slot + 1) % h->count;
   const uint64_t base = h->bo[slot].gpu_va;
   const uint64_t size = h->bo[slot].size;
   simple_mtx_unlock(&h->lock);
   if (!ok) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return false;
   }

   struct mali_ptr p = mali_cmd_alloc(cmd, pan_size(TILER_HEAP), 64);
   if (!p.cpu)
      return false;
   struct mali_tiler_heap_packed desc;
   pan_pack(&desc, TILER_HEAP, cfg) {
      cfg.size = (uint32_t)size;
      cfg.base = base;
      cfg.bottom = base;
#if PAN_ARCH == 7
      cfg.top = base + size;
#else
      cfg.top = base + MIN2(MALI_JM_HEAP_CHUNK, size);
#endif
   }
   memcpy(p.cpu, &desc, sizeof(desc));
   /* The GPU advances Bottom and Top. */
   mali_jm_cmd_note_reset(cmd, p.cpu, &desc, sizeof(desc));

   STATIC_ASSERT(MALI_JM_HEAP_SLOTS_GROWN <= MALI_JM_HEAP_SLOTS_MAX);
   STATIC_ASSERT(MALI_JM_HEAP_SLOT_SIZE <= UINT32_MAX);
   b->heap_slot = (int16_t)slot;
   b->heap_desc = p.gpu;
   return true;
}

void
mali_jm_heap_note_submit(struct mali_jm_device *jd, uint32_t tiled)
{
   struct mali_jm_heap *h = &jd->heap;
   if (h->count >= MALI_JM_HEAP_SLOTS_GROWN)
      return;
   if (tiled > h->count) {
      h->win_hits++;
      h->win_max = MAX2(h->win_max, tiled);
   }
   if (++h->win_len < MALI_JM_HEAP_GROW_WINDOW)
      return;
   if (h->win_hits >= MALI_JM_HEAP_GROW_HITS) {
      const uint32_t want = MIN2(ALIGN_POT(h->win_max, 4), MALI_JM_HEAP_SLOTS_GROWN);
      simple_mtx_lock(&h->lock);
      const uint32_t old = h->count;
      h->count = MAX2(h->count, want);
      simple_mtx_unlock(&h->lock);
      mesa_logi("malisx2: tiler heap ring %u -> %u slots (%u of the last %u submissions "
                "wrapped it, up to %u tiled passes)",
                old, want, h->win_hits, h->win_len, h->win_max);
   }
   h->win_len = h->win_hits = h->win_max = 0;
}
