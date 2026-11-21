/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The job-manager frontend: atom-number bookkeeping, JOB_SUBMIT, the
 * non-blocking event read and poll-based wait, POST_TERM, and the
 * sync-file stream a FENCE_TRIGGER soft atom's base_fence names. This is
 * the JM counterpart of csf.c; it is compiled unconditionally (like
 * context.c and mem.c) but every entry point here is only ever called on
 * a context mali_kbase_create negotiated as MALI_KBASE_FRONTEND_JM.
 *
 * What is NOT here: anything that interprets core_req, dependencies or
 * udata, and anything that builds job chains. The queue/command-buffer
 * layer above this one owns that; this layer only moves bytes to and from
 * the kernel.
 */

#include <errno.h>
#include <poll.h>
#include <stdio.h>
#include <string.h>

#include "kbase_priv.h"

/* ---------------------------------------------------------------------- */
/* Atom numbers                                                            */

void
mali_kbase_jm_atom_ids_init(struct mali_kbase_jm_atom_ids *ids)
{
   memset(ids, 0, sizeof(*ids));
   /* Atom number 0 means "no dependency" in a job-chain's dependency
    * fields and is never handed out: mark it permanently in use. */
   ids->bitmap[0] |= 1u;
}

static inline bool
bit_test(const uint8_t *bitmap, unsigned n)
{
   return (bitmap[n / 8] >> (n % 8)) & 1u;
}

static inline void
bit_set(uint8_t *bitmap, unsigned n)
{
   bitmap[n / 8] |= (uint8_t)(1u << (n % 8));
}

static inline void
bit_clear(uint8_t *bitmap, unsigned n)
{
   bitmap[n / 8] &= (uint8_t)~(1u << (n % 8));
}

unsigned
mali_kbase_jm_atom_ids_alloc(struct mali_kbase_jm_atom_ids *ids, uint8_t *out, unsigned n)
{
   unsigned got = 0;
   for (unsigned id = 1; id < 256 && got < n; id++) {
      if (bit_test(ids->bitmap, id))
         continue;
      bit_set(ids->bitmap, id);
      out[got++] = (uint8_t)id;
   }
   return got;
}

void
mali_kbase_jm_atom_ids_free(struct mali_kbase_jm_atom_ids *ids, uint8_t id)
{
   if (id != 0)
      bit_clear(ids->bitmap, id);
}

bool
mali_kbase_jm_atom_ids_used(const struct mali_kbase_jm_atom_ids *ids, uint8_t id)
{
   return bit_test(ids->bitmap, id);
}

unsigned
mali_kbase_jm_atom_ids_free_count(const struct mali_kbase_jm_atom_ids *ids)
{
   unsigned used = 0;
   for (unsigned id = 0; id < 256; id++)
      used += bit_test(ids->bitmap, id);
   return 255 - (used - 1); /* id 0 is permanently "used" and not a number */
}

/* ---------------------------------------------------------------------- */
/* Submission                                                              */

/* At most 256 atoms per JOB_SUBMIT call. Stock kbase has no per-call
 * limit, but the T820's kernel (Unisoc's kbase_jd_submit) records the
 * number of every atom of one call in a fixed array of 256 entries,
 * without a bounds check: a longer call would write past the array into
 * kernel memory. Do not raise this. */
#define MALI_KBASE_JM_MAX_ATOMS_PER_SUBMIT 256

/* One JOB_SUBMIT call of n <= 256 atoms, each as the first `size` bytes of
 * the kernel's atom struct followed by zeros up to `stride`. size 64 is
 * base_jd_atom, which is struct kb_jm_atom; a smaller size is
 * base_jd_atom_v2, which starts at jc. Returns the ioctl's result. */
static long
jm_job_submit(struct mali_kbase *kb, const struct kb_jm_atom *atoms, unsigned n,
              unsigned size, unsigned stride)
{
   struct kb_jm_job_submit s = {
      .addr = (uint64_t)(uintptr_t)atoms,
      .nr_atoms = n,
      .stride = stride,
   };
   const bool as_is = size == sizeof(struct kb_jm_atom) && stride == size;
   const unsigned from = size == sizeof(struct kb_jm_atom) ? 0 : offsetof(struct kb_jm_atom, jc);
   /* At most 18 KiB (256 atoms of 72 bytes), only when the kernel's atom
    * is not struct kb_jm_atom as it is. */
   uint8_t packed[as_is ? 1 : n * stride];
   if (!as_is) {
      memset(packed, 0, n * stride);
      for (unsigned i = 0; i < n; i++)
         memcpy(packed + i * stride, (const uint8_t *)&atoms[i] + from, size);
      s.addr = (uint64_t)(uintptr_t)packed;
   }
   return kb_ioctl(kb, KB_JM_IOCTL_JOB_SUBMIT, &s);
}

enum mali_kbase_result
mali_kbase_jm_submit(struct mali_kbase *kb, const struct kb_jm_atom *atoms, unsigned n)
{
   if (!kb || (n && !atoms))
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;

   for (unsigned off = 0; off < n; off += MALI_KBASE_JM_MAX_ATOMS_PER_SUBMIT) {
      unsigned batch = n - off;
      if (batch > MALI_KBASE_JM_MAX_ATOMS_PER_SUBMIT)
         batch = MALI_KBASE_JM_MAX_ATOMS_PER_SUBMIT;
      long ret = jm_job_submit(kb, atoms + off, batch, kb->jm_atom_size, kb->jm_atom_stride);
      if (ret < 0) {
         KB_LOG(kb, "JOB_SUBMIT of %u atoms (offset %u of %u, stride %u, kernel UK %u.%u) "
                    "failed: %s",
                batch, off, n, kb->jm_atom_stride, kb->uk_major, kb->uk_minor,
                strerror((int)-ret));
         return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
      }
   }
   return MALI_KBASE_SUCCESS;
}

/* ---------------------------------------------------------------------- */
/* Atom layout probe                                                       */

/* The probe atom's number and udata. The number is free again once its
 * event is read, before anyone else can submit on the context. */
#define PROBE_ATOM_NUMBER 1
#define PROBE_UDATA0 0x6d616c6973783270ull /* arbitrary, checked on return */
#define PROBE_UDATA1 0x61746f6d2d70726full

/* A dependency-only atom with no dependencies completes inside the
 * JOB_SUBMIT call (jd_submit_atom resolves it at once), so its event is
 * pending by the time the ioctl returns; this only bounds the wait. */
#define PROBE_EVENT_TIMEOUT_MS 1000

/* The bytes a vendor field on the end of the atom adds: MediaTek's u32
 * frame_nr, rounded up to the struct's 8-byte alignment. */
#define VENDOR_TAIL 8

/* Reads back the event of the probe atom the kernel accepted at the
 * context's stride, and checks that the kernel read the atom's fields
 * where we put them. */
static enum mali_kbase_result
probe_event(struct mali_kbase *kb)
{
   struct kb_jm_event ev;
   unsigned n = 0;
   const int slice = 50;
   for (int waited = 0; !n && waited <= PROBE_EVENT_TIMEOUT_MS; waited += slice) {
      enum mali_kbase_result r = mali_kbase_jm_read_events(kb, &ev, 1, &n, NULL);
      if (r != MALI_KBASE_SUCCESS)
         return r;
      if (!n && mali_kbase_jm_poll(kb, slice) < 0 && errno != EINTR) {
         KB_LOG(kb, "poll for the atom-layout probe's event failed: %s", strerror(errno));
         return MALI_KBASE_ERROR_KERNEL;
      }
   }
   if (!n) {
      /* Atom 1 would still be in flight in the kernel and its event
       * would turn up later as one nobody submitted. */
      KB_LOG(kb, "the kernel accepted a %u-byte atom but posted no event for it "
                 "within %d ms; refusing the device",
             kb->jm_atom_stride, PROBE_EVENT_TIMEOUT_MS);
      return MALI_KBASE_ERROR_INCOMPATIBLE_KERNEL;
   }
   kb->jm_probe_event = ev.event_code;
   if (ev.atom_number != PROBE_ATOM_NUMBER || ev.udata[0] != PROBE_UDATA0 ||
       ev.udata[1] != PROBE_UDATA1) {
      /* Accepted, but not read where we put the fields: a layout we do
       * not know. Submitting real work would run garbage. */
      KB_LOG(kb, "the kernel accepted a %u-byte atom but returned atom %u udata "
                 "0x%llx 0x%llx (event 0x%x); unknown atom layout, refusing the device",
             kb->jm_atom_stride, ev.atom_number, (unsigned long long)ev.udata[0],
             (unsigned long long)ev.udata[1], ev.event_code);
      return MALI_KBASE_ERROR_INCOMPATIBLE_KERNEL;
   }
   /* Any code is fine here: the kernel parsed the atom where we put it.
    * A code other than DONE (some of these kernels are said to terminate
    * the first atom of a new context) is reported by
    * mali_kbase_jm_atom_layout_str. */
   return MALI_KBASE_SUCCESS;
}

enum mali_kbase_result
kb_jm_probe_atom_layout(struct mali_kbase *kb)
{
   /* The minor picks the struct (uapi_jm.h, KB_JM_UK_MIN_MINOR); the
    * stride is that struct's size or, on a kernel with a vendor field on
    * the end of the atom, that plus VENDOR_TAIL. Below 11.22 the stock
    * kernels check the size exactly and a vendor kernel refuses it, so
    * the stock size goes first.
    *
    * From 11.22 the vendor stride (72) goes first: MediaTek's frame_nr
    * kernels also accept 64 but read it as base_jd_atom_v2 + frame_nr, a
    * stock atom 8 bytes off. A stock kernel refuses 72, which settles it
    * at 64 without submitting anything. */
   const unsigned v2_from = offsetof(struct kb_jm_atom, jc);
   const unsigned size = kb->uk_minor >= KB_JM_UK_ATOM_V3_MINOR ? sizeof(struct kb_jm_atom)
                         : kb->uk_minor >= KB_JM_UK_ATOM_RP_MINOR
                            ? sizeof(struct kb_jm_atom) - v2_from
                            : offsetof(struct kb_jm_atom, renderpass_id) - v2_from;
   const bool v3 = size == sizeof(struct kb_jm_atom);
   const unsigned strides[2] = {v3 ? size + VENDOR_TAIL : size, v3 ? size : size + VENDOR_TAIL};

   struct kb_jm_atom probe;
   memset(&probe, 0, sizeof(probe));
   probe.atom_number = PROBE_ATOM_NUMBER;
   probe.prio = KB_JM_PRIO_MEDIUM;
   probe.core_req = KB_JM_REQ_DEP;
   probe.udata[0] = PROBE_UDATA0;
   probe.udata[1] = PROBE_UDATA1;

   kb->jm_atom_size = size;
   kb->jm_probe_errno = 0;
   for (unsigned i = 0; i < 2; i++) {
      kb->jm_atom_stride = strides[i];
      long ret = jm_job_submit(kb, &probe, 1, size, strides[i]);
      if (ret == -EINVAL) {
         /* The stride check comes before any atom is read: nothing was
          * submitted, nothing will complete. */
         kb->jm_probe_errno = EINVAL;
         if (v3) {
            kb->jm_atom_stride = size;
            return MALI_KBASE_SUCCESS;
         }
         continue;
      }
      if (ret < 0) {
         KB_LOG(kb, "JOB_SUBMIT of the atom-layout probe at stride %u failed: %s", strides[i],
                strerror((int)-ret));
         return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
      }
      return probe_event(kb);
   }
   KB_LOG(kb, "the kernel (UK %u.%u) refused a %u-byte atom at stride %u and at stride %u (%s); "
              "unknown atom layout, refusing the device",
          kb->uk_major, kb->uk_minor, size, strides[0], strides[1], strerror(EINVAL));
   return MALI_KBASE_ERROR_INCOMPATIBLE_KERNEL;
}

const char *
mali_kbase_jm_atom_layout_str(const struct mali_kbase *kb, char *buf, size_t size)
{
   const char *code = kb->jm_probe_event == KB_JM_EVENT_DONE         ? " (done)"
                      : kb->jm_probe_event == KB_JM_EVENT_TERMINATED ? " (terminated)"
                                                                     : "";
   if (kb->jm_atom_size == sizeof(struct kb_jm_atom) && kb->jm_atom_stride == kb->jm_atom_size) {
      /* The one layout taken without a probe atom. */
      snprintf(buf, size, "64-byte atoms (stock kbase): the kernel refused stride 72 (%s)",
               strerror(kb->jm_probe_errno));
   } else if (kb->jm_atom_size == sizeof(struct kb_jm_atom)) {
      snprintf(buf, size,
               "72-byte atoms with frame_nr (MediaTek kbase): the kernel accepted "
               "stride 72, probe atom completed with 0x%x%s",
               kb->jm_probe_event, code);
   } else {
      snprintf(buf, size,
               "%u-byte atoms: %u-byte base_jd_atom_v2%s (UK %u.%u), "
               "probe atom completed with 0x%x%s",
               kb->jm_atom_stride, kb->jm_atom_size,
               kb->jm_atom_stride > kb->jm_atom_size ? " and a vendor field" : "", kb->uk_major,
               kb->uk_minor, kb->jm_probe_event, code);
   }
   return buf;
}

/* ---------------------------------------------------------------------- */
/* Completion                                                              */

enum mali_kbase_result
mali_kbase_jm_read_events(struct mali_kbase *kb, struct kb_jm_event *ev, unsigned max,
                          unsigned *n, bool *terminated)
{
   if (!kb || !ev || !max || !n)
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;
   *n = 0;
   if (terminated)
      *terminated = false;

   long r = kb->backend->read(kb->backend->priv, kb->fd, ev,
                              (size_t)max * sizeof(*ev));
   if (r < 0) {
      int err = errno;
      if (err == EAGAIN || err == EWOULDBLOCK)
         return MALI_KBASE_SUCCESS;
      if (err == EPIPE) {
         /* kbase_event_close/kbase_event_pending, JM build: once
          * POST_TERM has drained the queued events, read() answers
          * EPIPE forever. */
         if (terminated)
            *terminated = true;
         return MALI_KBASE_SUCCESS;
      }
      KB_LOG(kb, "reading job-manager events failed: %s", strerror(err));
      return kb_result_from_errno(err, KB_ERRNO_GENERIC);
   }

   *n = (unsigned)(r / (long)sizeof(*ev));
   return MALI_KBASE_SUCCESS;
}

int
mali_kbase_jm_poll(struct mali_kbase *kb, int timeout_ms)
{
   if (!kb->backend->poll) {
      errno = ENOSYS;
      return -1;
   }
   struct pollfd pfd = {.fd = kb->fd, .events = POLLIN};
   return kb->backend->poll(kb->backend->priv, &pfd, 1, timeout_ms);
}

enum mali_kbase_result
mali_kbase_jm_post_term(struct mali_kbase *kb)
{
   long ret = kb_ioctl(kb, KB_JM_IOCTL_POST_TERM, NULL);
   if (ret < 0) {
      KB_LOG(kb, "POST_TERM failed: %s", strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
   }
   return MALI_KBASE_SUCCESS;
}

enum mali_kbase_result
mali_kbase_jm_soft_event_update(struct mali_kbase *kb, uint64_t gpu_va, bool set)
{
   struct kb_jm_soft_event_update u = {
      .event = gpu_va,
      .new_status = set ? 1 : 0,
   };
   long ret = kb_ioctl(kb, KB_JM_IOCTL_SOFT_EVENT_UPDATE, &u);
   if (ret < 0) {
      KB_LOG(kb, "SOFT_EVENT_UPDATE failed: %s", strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
   }
   return MALI_KBASE_SUCCESS;
}

/* ---------------------------------------------------------------------- */
/* Sync-file streams                                                       */

enum mali_kbase_result
mali_kbase_stream_create(struct mali_kbase *kb, const char *name, int *out_fd)
{
   if (!kb || !out_fd)
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;

   struct kb_jm_stream_create sc;
   memset(&sc, 0, sizeof(sc));
   if (name)
      strncpy(sc.name, name, sizeof(sc.name) - 1);

   long ret = kb_ioctl(kb, KB_JM_IOCTL_STREAM_CREATE, &sc);
   if (ret < 0) {
      KB_LOG(kb, "STREAM_CREATE(\"%s\") failed: %s", sc.name, strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
   }
   *out_fd = (int)ret;
   return MALI_KBASE_SUCCESS;
}
