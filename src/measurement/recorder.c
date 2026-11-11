/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Command-stream capture at the submit splice: vkQueueSubmit is the one
 * place where every instruction the GPU will run is reachable.
 * tools/mali_cs_decode.py reads the format.
 *
 * One file per captured submit, csf-SSSSSSSS.bin (S = submit number):
 *
 *   header   "MALICSF1", u32 version (1), u32 GPU product ID,
 *            u64 submit number, u64 GPU timestamp rate
 *   records  u32 type, u32 arg, u64 a, u64 b, u64 size, then size bytes
 *            padded to 8:
 *     1 RING    arg = subqueue, a = its subqueue context address (the
 *               value of r90:91), payload = the words this submit put in
 *               its ring (waits, cache flush, CALLs, done signal)
 *     2 MEM     arg = 0 command memory, 1 the queue's sync memory;
 *               a = GPU address, payload = the bytes
 *     3 CMDBUF  arg = position in the submit, a = subqueue mask,
 *               b = VkCommandBufferUsageFlags, payload = per subqueue
 *               {u64 stream address, u32 size, u32 0}, then u32
 *               dispatches, draws, passes, 0
 *     0xffffffff END
 *
 * The command memory holds the recorded streams (and every chunk they
 * JUMP to) and everything the driver writes per command buffer:
 * framebuffer and tiler descriptors, FAU, resource tables, blend and
 * depth/stencil descriptors, full-screen draw descriptors. Descriptor
 * sets, shader descriptors, shader code and application buffers are not
 * captured.
 *
 * The file is written before the rings are published, so the memory is as
 * recorded, not as the GPU leaves it.
 */

#include "mali_measure.h"

#include <inttypes.h>
#include <string.h>

#include "util/log.h"
#include "vk_queue.h"

#include "mali_cmd_buffer.h"
#include "mali_queue.h"
#include "mali_vk.h"

#define CAP_RING   1u
#define CAP_MEM    2u
#define CAP_CMDBUF 3u
#define CAP_END    0xffffffffu

struct cap_header {
   char magic[8];
   uint32_t version;
   uint32_t product_id;
   uint64_t submit;
   uint64_t timestamp_hz;
};

struct cap_record {
   uint32_t type;
   uint32_t arg;
   uint64_t a;
   uint64_t b;
   uint64_t size;
};

struct cap_cmdbuf {
   struct {
      uint64_t addr;
      uint32_t size;
      uint32_t pad;
   } span[MALI_SUBQUEUE_COUNT];
   uint32_t dispatches, draws, passes, pad;
};

struct mali_measure_capture {
   uint64_t seqno;
   struct mali_csf_queue *q;
   struct vk_queue_submit *submit;
   struct util_dynarray ring[MALI_SUBQUEUE_COUNT];   /* uint64_t words */
};

struct mali_measure_capture *
mali_measure_capture_begin(struct mali_device *dev, struct mali_queue *queue,
                           struct vk_queue_submit *submit, uint64_t seqno)
{
   struct mali_measure *m = dev->measure;
   /* A submit without command buffers (a fence, vkQueueWaitIdle) has
    * nothing worth a capture. */
   if (!m->cfg.csf || !submit->command_buffer_count)
      return NULL;

   pthread_mutex_lock(&m->lock);
   bool take = mali_measure_window(m, seqno) &&
               (!m->cfg.csf_max || m->captures < m->cfg.csf_max);
   pthread_mutex_unlock(&m->lock);
   if (!take)
      return NULL;

   struct mali_measure_capture *cap = calloc(1, sizeof(*cap));
   if (!cap)
      return NULL;
   cap->seqno = seqno;
   cap->q = queue->csf;
   cap->submit = submit;
   for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++)
      util_dynarray_init(&cap->ring[i], NULL);
   return cap;
}

void
mali_measure_capture_ring(struct mali_measure_capture *cap, unsigned sq,
                          const uint64_t *words, uint32_t count)
{
   util_dynarray_append_array(&cap->ring[sq], uint64_t, words, count);
}

static bool
put(FILE *fp, const void *p, size_t size, uint64_t *bytes)
{
   if (size && fwrite(p, 1, size, fp) != size)
      return false;
   *bytes += size;
   return true;
}

static bool
put_record(FILE *fp, uint32_t type, uint32_t arg, uint64_t a, uint64_t b, const void *data,
           uint64_t size, uint64_t *bytes)
{
   const struct cap_record r = {type, arg, a, b, size};
   static const uint8_t zeros[8];
   return put(fp, &r, sizeof(r), bytes) && put(fp, data, size, bytes) &&
          put(fp, zeros, (8 - size % 8) % 8, bytes);
}

static bool
write_capture(struct mali_device *dev, struct mali_measure_capture *cap, FILE *fp,
              uint64_t *bytes)
{
   struct mali_measure *m = dev->measure;
   struct cap_header h = {
      .magic = "MALICSF1",
      .version = 1,
      .product_id = (uint32_t)dev->kbase->props.product_id,
      .submit = cap->seqno,
      .timestamp_hz = m->hz,
   };
   if (!put(fp, &h, sizeof(h), bytes))
      return false;

   for (uint32_t c = 0; c < cap->submit->command_buffer_count; c++) {
      struct mali_cmd_buffer *cmd =
         container_of(cap->submit->command_buffers[c], struct mali_cmd_buffer, vk);
      struct cap_cmdbuf cb = {
         .dispatches = cmd->dispatches,
         .draws = cmd->draws,
         .passes = cmd->passes,
      };
      uint32_t mask = 0;
      for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++) {
         if (mali_cmd_buffer_span(cmd, i, &cb.span[i].addr, &cb.span[i].size))
            mask |= 1u << i;
      }
      if (!put_record(fp, CAP_CMDBUF, c, mask, cmd->usage, &cb, sizeof(cb), bytes))
         return false;
      list_for_each_entry(struct mali_cmd_slab, s, &cmd->slabs, link) {
         if (!put_record(fp, CAP_MEM, 0, s->bo.gpu_va, 0, s->bo.cpu, s->bo.size, bytes))
            return false;
      }
   }

   const struct mali_kbase_bo *sync = &cap->q->sync_mem;
   if (!put_record(fp, CAP_MEM, 1, sync->gpu_va, 0, sync->cpu, sync->size, bytes))
      return false;

   for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++) {
      const uint32_t n = util_dynarray_num_elements(&cap->ring[i], uint64_t);
      if (!n)
         continue;
      if (!put_record(fp, CAP_RING, i, sync->gpu_va + MALI_QUEUE_CTX_OFFSET(i), 0,
                      cap->ring[i].data, n * 8ull, bytes))
         return false;
   }
   return put_record(fp, CAP_END, 0, 0, 0, NULL, 0, bytes);
}

void
mali_measure_capture_end(struct mali_device *dev, struct mali_measure_capture *cap)
{
   struct mali_measure *m = dev->measure;
   char name[40], path[600];
   snprintf(name, sizeof(name), "csf-%08" PRIu64 ".bin", cap->seqno);
   mali_measure_path(m, name, path, sizeof(path));

   uint64_t bytes = 0;
   FILE *fp = mali_measure_fopen(&m->cfg, path, "wb");
   bool ok = fp && write_capture(dev, cap, fp, &bytes);
   if (fp && fclose(fp) != 0)
      ok = false;

   pthread_mutex_lock(&m->lock);
   if (ok) {
      mali_measure_add_capture(m, name, cap->seqno, bytes);
   } else {
      if (!m->write_failed)
         mesa_loge("malisx2: measurement: cannot write %s", path);
      m->write_failed = true;
   }
   pthread_mutex_unlock(&m->lock);

   for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++)
      util_dynarray_fini(&cap->ring[i]);
   free(cap);
}
