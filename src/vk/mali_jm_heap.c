/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The tiler heap ring of the job-manager (v7 and v9) back half:
 * HEAP_SLOTS growable GPU regions (4, 16 on v7) the tiler allocates
 * polygon lists from, and on v9 vertex packets too (mali_jm.h has the
 * scheme). The driver owns the memory and writes the Tiler Heap descriptor
 * itself when it records, as Mesa's gallium driver does; there is no JIT
 * allocation and no GPU job that sets the heap up. What the heap must look
 * like to the GPU:
 *
 *  - Size: the region's size; Base and Bottom: its start;
 *  - Top: the end of the first 2 MiB chunk, not the end of the region. The
 *    tiler allocates chunks from both the bottom and the top pointer and
 *    moves either on to a new chunk when its part is used up. A Top at the
 *    region's end would make the first top-side allocation fault at the
 *    region's last page, and a GROW_ON_GPF fault commits every page from
 *    the current commit up to the faulting one: the whole region.
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

/* The ring's size on this arch (mali_jm.h). */
#if PAN_ARCH < 9
#define HEAP_SLOTS MALI_JM_HEAP_SLOTS_V7
#else
#define HEAP_SLOTS MALI_JM_HEAP_SLOTS
#endif

void
mali_jm_heap_init(struct mali_jm_device *jd)
{
   simple_mtx_init(&jd->heap.lock, mtx_plain);
   jd->heap.next = 0;
}

void
mali_jm_heap_finish(struct mali_jm_device *jd)
{
   for (unsigned i = 0; i < HEAP_SLOTS; i++) {
      if (jd->heap.bo[i].size)
         mali_kbase_free(jd->kb, &jd->heap.bo[i]);
   }
   simple_mtx_destroy(&jd->heap.lock);
}

/* A slot's memory: GPU-only and growable. On a 64-bit job-manager context
 * the kernel makes it SAME_VA, so mali_kbase_alloc maps it whole with no CPU
 * access (PROT_NONE) to get its address; each slot takes 256 MiB of the
 * process's address space and commits 2 MiB until the tiler grows it (or
 * MALI_JM_HEAP_PRECOMMIT, below). */
static bool
slot_alloc(struct mali_jm_device *jd, unsigned slot, struct mali_kbase_bo *bo)
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
   /* Back the first MALI_JM_HEAP_PRECOMMIT now (the kernel takes at most
    * one growth step as the initial commit of a TILER_ALIGN_TOP region),
    * so that ordinary frames never fault on the heap. A GROW_ON_GPF fault
    * stalls the GPU while the kernel finds the pages, which with the
    * system short of memory can take long. Not fatal if the kernel
    * refuses: the tiler then grows the region as before. */
   /* The first MALI_JM_HEAP_SLOTS only: v7's further slots (mali_jm.h)
    * mostly take small passes and grow if a big one needs more. */
   if (MALI_JM_HEAP_PRECOMMIT > MALI_JM_HEAP_CHUNK && slot < MALI_JM_HEAP_SLOTS &&
       !mali_debug_flag("heapgrow") &&
       mali_kbase_mem_commit(jd->kb, bo, MALI_JM_HEAP_PRECOMMIT / KB_PAGE_SIZE) !=
          MALI_KBASE_SUCCESS)
      mesa_logw("malisx2: the tiler heap region could not be backed up front; it grows on use");
   return true;
}

/* Log the heap regions whose backed size grew since the last look: the
 * tiler grows a region by faulting on it, and each fault stalls the GPU.
 * The queue calls this every MALI_JM_HEAP_CHECK_PERIOD submissions. */
void
mali_jm_heap_check(struct mali_jm_device *jd, uint64_t seq)
{
   struct mali_jm_heap *h = &jd->heap;
   simple_mtx_lock(&h->lock);
   for (unsigned i = 0; i < HEAP_SLOTS; i++) {
      uint64_t pages = 0;
      if (!h->bo[i].size ||
          mali_kbase_mem_query(jd->kb, h->bo[i].gpu_va, KB_MEM_QUERY_COMMIT_SIZE, &pages) !=
             MALI_KBASE_SUCCESS)
         continue;
      /* The kernel answers in pages. */
      const uint64_t bytes = pages * KB_PAGE_SIZE;
      if (bytes <= h->committed[i])
         continue;
      mesa_logi("malisx2: tiler heap region %u backs %llu KiB (was %llu) by submission %llu",
                i, (unsigned long long)(bytes >> 10),
                (unsigned long long)(h->committed[i] >> 10), (unsigned long long)seq);
      h->committed[i] = bytes;
   }
   simple_mtx_unlock(&h->lock);
}

bool
mali_jm_heap_take(struct mali_cmd_buffer *cmd, struct mali_jm_batch *b)
{
   struct mali_jm_device *jd = cmd->dev->jm;
   struct mali_jm_heap *h = &jd->heap;

   simple_mtx_lock(&h->lock);
   const uint32_t slot = h->next;
   bool ok = h->bo[slot].size || slot_alloc(jd, slot, &h->bo[slot]);
   if (ok)
      h->next = (slot + 1) % HEAP_SLOTS;
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
      cfg.top = base + MIN2(MALI_JM_HEAP_CHUNK, size);
   }
   memcpy(p.cpu, &desc, sizeof(desc));
   /* The GPU advances Bottom and Top. */
   mali_jm_cmd_note_reset(cmd, p.cpu, &desc, sizeof(desc));

   STATIC_ASSERT(HEAP_SLOTS <= MALI_JM_HEAP_SLOTS_MAX);
   STATIC_ASSERT(MALI_JM_HEAP_SLOT_SIZE <= UINT32_MAX);
   b->heap_slot = (int16_t)slot;
   b->heap_desc = p.gpu;
   return true;
}
