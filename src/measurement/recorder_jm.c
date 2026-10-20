/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Job-manager command-stream capture: MALISX2_MEASURE's "csf" mode keeps
 * that name on a job-manager device too -- job-manager capture is a mode
 * of the same knob, not a second one. One file per captured vkQueueSubmit,
 * jm-SSSSSSSS.bin (S = submit number):
 *
 *   header   "MALIJM01", u32 version (1), u32 GPU product ID,
 *            u64 submit number, u64 GPU timestamp rate
 *   records  u32 type, u32 arg, u64 a, u64 b, u64 size, then size bytes
 *            padded to 8 -- the same record shape as csf-*.bin
 *     1 ATOMS   arg = count, payload = count raw 64-byte struct kb_jm_atom
 *               (kbase/uapi_jm.h), as submitted. More than one ATOMS
 *               record only when the submission needed more than one
 *               JOB_SUBMIT call (atom numbers ran out mid-submit,
 *               which forces a flush)
 *     2 MEM     arg = 0 (command memory), a = GPU address (same-VA, so
 *               also the CPU address), payload = the bytes
 *     3 CMDBUF  arg = position in the submit, payload = u32 dispatches,
 *               draws, passes, 0
 *     0xffffffff END
 *
 * tools/mali_jc_decode.py walks each atom's chain through its job
 * headers' Next pointers and decodes every job with v9.xml, the way
 * tools/mali_cs_decode.py walks a CSF capture's rings with v11.xml. A G52
 * capture holds v7 jobs and descriptors, which only v7.xml describes; the
 * header's GPU product ID says which arch a capture is.
 *
 * Command memory is captured before the kernel sees the submission
 * (mali_jm_measure_capture_begin runs before mali_jm_build_begin in
 * MALI_PER_ARCH(queue_submit)), so it is as recorded, not as the job
 * manager leaves it; the atoms are plain CPU data (struct kb_jm_atom)
 * either way.
 */

#include "mali_measure.h"

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include "util/log.h"
#include "vk_queue.h"

#include "mali_jm.h"

#define CAP_ATOMS  1u
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
   uint32_t dispatches, draws, passes, pad;
};

struct mali_jm_measure_capture {
   uint64_t seqno;
   struct vk_queue_submit *submit;
   struct util_dynarray atoms; /* struct kb_jm_atom, as built across every
                                 * JOB_SUBMIT call of this vkQueueSubmit */
};

struct mali_jm_measure_capture *
mali_jm_measure_capture_begin(struct mali_device *dev, struct vk_queue_submit *submit,
                              uint64_t seqno)
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

   struct mali_jm_measure_capture *cap = calloc(1, sizeof(*cap));
   if (!cap)
      return NULL;
   cap->seqno = seqno;
   cap->submit = submit;
   util_dynarray_init(&cap->atoms, NULL);
   return cap;
}

void
mali_jm_measure_capture_atoms(struct mali_jm_measure_capture *cap, const struct kb_jm_atom *atoms,
                              unsigned n)
{
   if (cap && n)
      util_dynarray_append_array(&cap->atoms, struct kb_jm_atom, atoms, n);
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
write_capture(struct mali_device *dev, struct mali_jm_measure_capture *cap, FILE *fp,
              uint64_t *bytes)
{
   struct mali_measure *m = dev->measure;
   struct cap_header h = {
      .magic = "MALIJM01",
      .version = 1,
      .product_id = (uint32_t)dev->kbase->props.product_id,
      .submit = cap->seqno,
      .timestamp_hz = m->hz,
   };
   if (!put(fp, &h, sizeof(h), bytes))
      return false;

   const unsigned natoms = util_dynarray_num_elements(&cap->atoms, struct kb_jm_atom);
   if (natoms && !put_record(fp, CAP_ATOMS, natoms, 0, 0, util_dynarray_begin(&cap->atoms),
                             natoms * sizeof(struct kb_jm_atom), bytes))
      return false;

   for (uint32_t c = 0; c < cap->submit->command_buffer_count; c++) {
      struct mali_cmd_buffer *cmd =
         container_of(cap->submit->command_buffers[c], struct mali_cmd_buffer, vk);
      const struct cap_cmdbuf cb = {
         .dispatches = cmd->dispatches,
         .draws = cmd->draws,
         .passes = cmd->passes,
      };
      if (!put_record(fp, CAP_CMDBUF, c, 0, 0, &cb, sizeof(cb), bytes))
         return false;
      list_for_each_entry(struct mali_cmd_slab, s, &cmd->slabs, link) {
         if (!put_record(fp, CAP_MEM, 0, s->bo.gpu_va, 0, s->bo.cpu, s->bo.size, bytes))
            return false;
      }
   }
   return put_record(fp, CAP_END, 0, 0, 0, NULL, 0, bytes);
}

void
mali_jm_measure_capture_end(struct mali_device *dev, struct mali_jm_measure_capture *cap)
{
   if (!cap)
      return;
   struct mali_measure *m = dev->measure;
   char name[40], path[600];
   snprintf(name, sizeof(name), "jm-%08" PRIu64 ".bin", cap->seqno);
   mali_measure_path(m, name, path, sizeof(path));

   uint64_t bytes = 0;
   FILE *fp = fopen(path, "wb");
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

   util_dynarray_fini(&cap->atoms);
   free(cap);
}
