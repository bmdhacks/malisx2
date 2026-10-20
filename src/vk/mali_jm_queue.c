/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The job-manager queue: atoms, submit, completion and device loss.
 *
 * One vkQueueSubmit becomes one JOB_SUBMIT holding the whole atom graph of
 * its command buffers, with the dependencies given to the kernel as atom
 * pre_deps, so no dependency inside a submission costs a CPU round trip
 * (more than one JOB_SUBMIT only when atom numbers run out). The graph
 * (mali_jm.h has the per-batch inputs):
 *
 *   vtc(b):  slot 1, core_req CS|T|CF|COHERENT_GROUP
 *            pre_dep[0] the previous atom in slot 1's order
 *            pre_dep[1] the newest fragment-slot atom it has to wait for:
 *                       the batch's vtc_after_frag (or a barrier at the
 *                       end of an earlier command buffer,
 *                       mali_jm_cmd::end_req), the last fragment reader of
 *                       its tiler heap slot, and (first vtc atom of a
 *                       submission) semaphore waits whose destination
 *                       stages include vtc-slot stages
 *   frag(b): slot 0, core_req FS, one per fragment segment
 *            pre_dep[0] vtc(b) (or, with frag_after_vtc and no vtc chain,
 *                       the newest vtc-slot atom)
 *            pre_dep[1] the previous atom in slot 0's order
 *
 * Every dependency is DATA except one on a FENCE_WAIT atom, which is
 * ORDER (dep_type). With DATA the kernel fails an atom whose dependency
 * failed, with the same event code, without running it, so a GPU fault
 * travels along the graph: to the submission's tracker and to the
 * FENCE_TRIGGER of a sync file exported for it, which then signals with
 * an error instead of success. A waited sync file that signalled an
 * error (the display's acquire fence) cancels its FENCE_WAIT, and the
 * ORDER dependency keeps that from failing the work behind it. The one
 * gap: a fault does not travel through a FENCE_WAIT that sits in a slot
 * order between the faulting atom and later ones (the device is lost
 * anyway; only that sync file's error status is missed).
 *
 * Every atom depends on its slot's previous one, so each slot runs in
 * submission order like a CSF ring and any "everything on slot S up to
 * here" collapses to one dependency on the newest atom of S. Waits on
 * imported sync files are FENCE_WAIT soft atoms placed into the order of
 * the slot their destination stages need (both slots: into the fragment
 * order, and the first vtc atom waits for it too), so an acquire fence
 * waited at COLOR_ATTACHMENT_OUTPUT holds only fragment work.
 *
 * Atoms in a slot order carry EVENT_COALESCE. Each JOB_SUBMIT call ends in
 * one non-coalesced atom that depends on all of them (the last one if it
 * does, else a dependency-only join); for the last call of a submission it
 * covers both slots' newest atoms and is the submission's "tracker": its
 * event (udata[0] = the submission number) means that submission and every
 * earlier one have finished. Coalesced events arrive with it, so their
 * atom numbers come back with it, and no atom number is ever stuck behind
 * a coalesced event nobody will flush.
 *
 * Atom numbers are freed only when their event is read. A dependency is
 * written only on an atom whose event has not been read (mali_jm_ref_live),
 * so it never names a reused number. Events are read on demand, with no
 * event thread: without blocking at every submit, status check and host
 * wait, and blocking (poll() on the kbase fd) by a host waiter or a submit
 * that ran out of numbers. One thread at a time polls (the reader); the
 * others sleep on the device condition variable, which it broadcasts.
 *
 * Device loss: any event code other than DONE, except a FENCE_WAIT
 * cancelled because the waited fence signalled an error, makes the device
 * lost, sticky, reported at the next submit, wait or status query.
 * Nothing is masked, TERMINATED included; that may be revisited later.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "mali_jm.h"

#include <assert.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "util/log.h"
#include "util/os_time.h"
#include "vk_alloc.h"
#include "vk_log.h"

#include "mali_measure.h"

/* One JOB_SUBMIT's worth of atoms being built. jc of a soft fence atom
 * points at its fences[] entry, which the kernel reads (and, for
 * FENCE_TRIGGER, writes the new fd into) during the call. */
struct mali_jm_build {
   uint64_t seq;
   unsigned n;
   struct kb_jm_atom atoms[256];
   struct kb_fence fences[256];
   int trigger_idx;              /* FENCE_TRIGGER among atoms[0..n), or -1 */
   int trigger_fd;
   mali_jm_ref last;             /* the last atom built, if in atoms[0..n) */
   mali_jm_ref slot_last[MALI_JM_SLOT_COUNT]; /* newest slot-order atoms in atoms[0..n) */
   bool any;
};

/* ---------------------------------------------------------------------- */
/* Faults                                                                  */

static void
jm_set_fault(struct mali_jm_device *jd, const char *fmt, ...)
   __attribute__((format(printf, 2, 3)));

static void
jm_set_fault(struct mali_jm_device *jd, const char *fmt, ...)
{
   if (jd->fault)
      return;
   va_list ap;
   va_start(ap, fmt);
   vsnprintf(jd->fault_msg, sizeof(jd->fault_msg), fmt, ap);
   va_end(ap);
   jd->fault = true;
}

static const char *
event_code_name(uint32_t code)
{
   switch (code) {
   case KB_JM_EVENT_STOPPED: return "stopped";
   case KB_JM_EVENT_TERMINATED: return "terminated";
   case KB_JM_EVENT_JOB_CONFIG_FAULT: return "job config fault";
   case KB_JM_EVENT_JOB_POWER_FAULT: return "job power fault";
   case KB_JM_EVENT_JOB_READ_FAULT: return "job read fault";
   case KB_JM_EVENT_JOB_WRITE_FAULT: return "job write fault";
   case KB_JM_EVENT_JOB_AFFINITY_FAULT: return "job affinity fault";
   case KB_JM_EVENT_JOB_BUS_FAULT: return "job bus fault";
   case KB_JM_EVENT_DATA_INVALID_FAULT: return "data invalid fault";
   case KB_JM_EVENT_TILE_RANGE_FAULT: return "tile range fault";
   case KB_JM_EVENT_STATE_FAULT: return "state fault";
   case KB_JM_EVENT_OUT_OF_MEMORY: return "out of memory";
   case KB_JM_EVENT_UNKNOWN: return "unknown fault";
   case KB_JM_EVENT_DELAYED_BUS_FAULT: return "delayed bus fault";
   case KB_JM_EVENT_SHAREABILITY_FAULT: return "shareability fault";
   case KB_JM_EVENT_PERMISSION_FAULT: return "permission fault";
   case KB_JM_EVENT_ACCESS_FLAG: return "access flag fault";
   case KB_JM_EVENT_MEM_GROWTH_FAILED: return "memory growth failed";
   case KB_JM_EVENT_JOB_CANCELLED: return "job cancelled";
   case KB_JM_EVENT_JOB_INVALID: return "job invalid";
   default:
      if (code >= 0xc0 && code <= 0xc7)
         return "translation fault";
      return "error";
   }
}

static const char *
atom_kind_name(unsigned kind)
{
   switch (kind) {
   case MALI_JM_ATOM_VTC: return "vertex/tiler/compute";
   case MALI_JM_ATOM_FRAG: return "fragment";
   case MALI_JM_ATOM_JOIN: return "dependency-only";
   case MALI_JM_ATOM_FENCE_WAIT: return "fence wait";
   case MALI_JM_ATOM_FENCE_TRIGGER: return "fence trigger";
   default: return "unknown";
   }
}

/* ---------------------------------------------------------------------- */
/* Events                                                                  */

/* With dev->lock held. The event code is all the kernel tells us about a
 * job fault: no fault address or status reaches userspace on the job
 * manager. */
static void
record_fault(struct mali_device *dev, const struct kb_jm_event *ev, const char *what)
{
   struct mali_device_fault f = {
      .address_type = VK_DEVICE_FAULT_ADDRESS_TYPE_NONE_EXT,
      .code = ev->event_code,
      .data = ev->atom_number,
   };
   snprintf(f.what, sizeof(f.what), "%s", what);
   mali_device_record_fault(dev, &f);
}

static void
process_event(struct mali_device *dev, const struct kb_jm_event *ev)
{
   struct mali_jm_device *jd = dev->jm;
   const uint8_t n = ev->atom_number;
   jd->stats.events++;
   dev->stats.kernel_events++;

   if (n == 0 || ev->event_code == KB_JM_EVENT_DRV_TERMINATED) {
      record_fault(dev, ev, "the kernel ended the job-manager context");
      jm_set_fault(jd, "the kernel ended the job-manager context (event 0x%x)",
                   ev->event_code);
      return;
   }
   struct mali_jm_atom_info *a = &jd->atoms[n];
   if (!a->in_flight) {
      jm_set_fault(jd, "event 0x%x for atom %u, which is not in flight", ev->event_code, n);
      return;
   }
   a->in_flight = false;
   mali_kbase_jm_atom_ids_free(&jd->ids, n);
   jd->free_ids++;
   jd->in_flight--;

   const uint64_t u = ev->udata[1];
   const unsigned kind = MALI_JM_UDATA_KIND(u);
   if (ev->event_code != KB_JM_EVENT_DONE) {
      if (kind == MALI_JM_ATOM_FENCE_WAIT && ev->event_code == KB_JM_EVENT_JOB_CANCELLED) {
         /* The waited sync file signalled an error (the display's acquire
          * fence). Its followers depend on it with ORDER, so the error
          * goes no further. */
         jd->stats.fence_wait_errors++;
         mesa_logw("malisx2: a waited sync file signalled an error (submission %llu); "
                   "continuing", (unsigned long long)ev->udata[0]);
      } else {
         char what[VK_MAX_DESCRIPTION_SIZE];
         snprintf(what, sizeof(what), "%s: %s atom %u (submission %llu)",
                  event_code_name(ev->event_code), atom_kind_name(kind), n,
                  (unsigned long long)ev->udata[0]);
         record_fault(dev, ev, what);
         jm_set_fault(jd,
                      "%s atom %u (submission %llu, command buffer %u, batch %u) "
                      "completed with 0x%x (%s)",
                      atom_kind_name(kind), n, (unsigned long long)ev->udata[0],
                      MALI_JM_UDATA_CMD(u), MALI_JM_UDATA_BATCH(u), ev->event_code,
                      event_code_name(ev->event_code));
      }
   }
   if ((u & MALI_JM_UDATA_TRACKER) && jd->queue && ev->udata[0] > jd->queue->completed_seq)
      jd->queue->completed_seq = ev->udata[0];
}

/* Read every pending event without blocking. With dev->lock held. */
static void
read_events_locked(struct mali_device *dev)
{
   struct mali_jm_device *jd = dev->jm;
   struct kb_jm_event ev[MALI_JM_EVENTS_PER_READ];
   bool any = false;

   while (jd->in_flight && !jd->fault) {
      unsigned n = 0;
      bool terminated = false;
      enum mali_kbase_result r =
         mali_kbase_jm_read_events(jd->kb, ev, MALI_JM_EVENTS_PER_READ, &n, &terminated);
      if (r != MALI_KBASE_SUCCESS) {
         jm_set_fault(jd, "reading job-manager events failed: %s", mali_kbase_result_str(r));
         break;
      }
      if (terminated) {
         jm_set_fault(jd, "the kbase event channel was closed");
         break;
      }
      for (unsigned i = 0; i < n; i++)
         process_event(dev, &ev[i]);
      any |= n != 0;
      if (n < MALI_JM_EVENTS_PER_READ)
         break;
   }
   if (any || jd->fault)
      pthread_cond_broadcast(&dev->cond);
}

void
mali_jm_drain_locked(struct mali_device *dev)
{
   struct mali_jm_device *jd = dev->jm;
   if (!jd || jd->reader)
      return;
   read_events_locked(dev);
}

/* A reader's poll() never sleeps longer than this, so a host signal (which
 * cannot interrupt it) is seen within it, as with the G615's waits. */
#define READER_SLICE_NS (20ll * 1000 * 1000)

void
mali_jm_wait_locked(struct mali_device *dev, int64_t abs_ns)
{
   struct mali_jm_device *jd = dev->jm;
   int64_t now = os_time_get_nano();
   int64_t until = MIN2(abs_ns, now + READER_SLICE_NS);

   if (!jd || jd->reader || !jd->in_flight || jd->fault) {
      /* Nothing to read for us: sleep on the condition variable (a reader,
       * a submit or a host signal broadcasts it). */
      struct timespec ts = {
         .tv_sec = until / 1000000000ll,
         .tv_nsec = until % 1000000000ll,
      };
      pthread_cond_timedwait(&dev->cond, &dev->lock, &ts);
      return;
   }

   jd->reader = true;
   int ms = until > now ? (int)((until - now + 999999) / 1000000) : 0;
   pthread_mutex_unlock(&dev->lock);
   int r = mali_kbase_jm_poll(jd->kb, ms);
   int err = errno;
   pthread_mutex_lock(&dev->lock);
   jd->reader = false;
   if (r < 0 && err != EINTR)
      jm_set_fault(jd, "poll on the kbase fd failed: %s", strerror(err));
   else if (r > 0)
      read_events_locked(dev);
   /* Whoever sleeps on the condition may become the reader now. */
   pthread_cond_broadcast(&dev->cond);
}

/* ---------------------------------------------------------------------- */
/* Hooks for mali_sync.c and the runtime                                   */

const char *
MALI_PER_ARCH(queue_wait)(struct mali_device *dev)
{
   struct mali_jm_device *jd = dev->jm;
   if (!jd || dev->lost)
      return NULL;
   mali_jm_drain_locked(dev);
   return jd->fault ? jd->fault_msg : NULL;
}

bool
MALI_PER_ARCH(queue_reached)(struct mali_device *dev, const uint64_t req[MALI_SYNC_REQ_COUNT])
{
   struct mali_jm_device *jd = dev->jm;
   if (!jd || !jd->queue)
      return true;
   return jd->queue->completed_seq >= req[0];
}

void
MALI_PER_ARCH(queue_describe)(struct mali_device *dev, char *buf, size_t size)
{
   const struct mali_jm_device *jd = dev->jm;
   if (!jd) {
      snprintf(buf, size, "no job-manager state");
      return;
   }
   const struct mali_jm_queue *q = jd->queue;
   snprintf(buf, size,
            "submission %llu sent, %llu completed; %u atoms in flight, %u numbers free; "
            "%llu JOB_SUBMIT calls, %llu kernel events read, %llu fence waits, "
            "%llu fence triggers%s%s",
            (unsigned long long)(q ? q->seq : 0), (unsigned long long)(q ? q->completed_seq : 0),
            jd->in_flight, jd->free_ids, (unsigned long long)jd->stats.job_submits,
            (unsigned long long)jd->stats.events, (unsigned long long)jd->stats.fence_waits,
            (unsigned long long)jd->stats.fence_triggers, jd->fault ? "; fault: " : "",
            jd->fault ? jd->fault_msg : "");
}

/* The slow-wait log for the job manager's own blocking loops: once per
 * wait, after MALI_SLOW_WAIT_NS. With dev->lock held. */
static void
jm_note_slow(struct mali_device *dev, int64_t start, bool *warned, const char *what)
{
   if (*warned || os_time_get_nano() - start < MALI_SLOW_WAIT_NS)
      return;
   *warned = true;
   uint32_t n;
   if (!mali_diag_should_log(&dev->diag.slow_waits, &n))
      return;
   char state[320];
   MALI_PER_ARCH(queue_describe)(dev, state, sizeof(state));
   mesa_logw("malisx2: %s has blocked for over 1 s (slow wait %u): %s", what, n, state);
}

/* Shared with timing.c's completed(), for a mc created by either
 * frontend. jd is NULL on a CSF device, in which case completed() never
 * calls this. */
bool
mali_jm_measure_reached(struct mali_jm_device *jd, uint64_t seq)
{
   return !jd->queue || jd->queue->completed_seq >= seq;
}

VkResult
MALI_PER_ARCH(device_check_status)(struct vk_device *vkdev)
{
   struct mali_device *dev = container_of(vkdev, struct mali_device, vk);
   pthread_mutex_lock(&dev->lock);
   const char *msg = MALI_PER_ARCH(queue_wait)(dev);
   char copy[sizeof(dev->jm->fault_msg)];
   if (msg)
      snprintf(copy, sizeof(copy), "%s", msg);
   bool lost = dev->lost;
   pthread_mutex_unlock(&dev->lock);
   if (msg) {
      mali_device_set_lost(dev, "%s", copy);
      lost = true;
   }
   return lost ? VK_ERROR_DEVICE_LOST : VK_SUCCESS;
}

/* After dropping dev->lock: report a fault that was read (once; later
 * callers just see dev->lost). */
static void
report_fault(struct mali_device *dev, const char *msg)
{
   if (msg)
      mali_device_set_lost(dev, "%s", msg);
}

/* ---------------------------------------------------------------------- */
/* Building atoms                                                          */

static bool
covers(const struct mali_jm_device *jd, mali_jm_ref by, mali_jm_ref r)
{
   if (!mali_jm_ref_live(jd, r))
      return true;
   if (!by)
      return false;
   const struct mali_jm_atom_info *b = &jd->atoms[mali_jm_ref_num(by)];
   const struct mali_jm_atom_info *a = &jd->atoms[mali_jm_ref_num(r)];
   if (a->slot >= MALI_JM_SLOT_COUNT)
      return by == r;
   return mali_jm_ref_serial(b->cov[a->slot]) >= mali_jm_ref_serial(r);
}

static mali_jm_ref
newer(mali_jm_ref a, mali_jm_ref b)
{
   return mali_jm_ref_serial(a) >= mali_jm_ref_serial(b) ? a : b;
}

void
mali_jm_build_begin(struct mali_device *dev, uint64_t seq)
{
   struct mali_jm_device *jd = dev->jm;
   while (jd->building)
      pthread_cond_wait(&dev->cond, &dev->lock);
   jd->building = true;
   struct mali_jm_build *b = jd->build;
   b->seq = seq;
   b->n = 0;
   b->trigger_idx = -1;
   b->trigger_fd = -1;
   b->last = 0;
   memset(b->slot_last, 0, sizeof(b->slot_last));
   b->any = false;
}

/* JOB_SUBMIT of atoms[0..n). */
static VkResult
flush(struct mali_device *dev)
{
   struct mali_jm_device *jd = dev->jm;
   struct mali_jm_build *b = jd->build;
   if (!b->n)
      return VK_SUCCESS;

   /* MALISX2_MEASURE's "csf" mode (recorder_jm.c): the atoms of this
    * JOB_SUBMIT call, before they are handed to the kernel. More than
    * one call per vkQueueSubmit only happens when atom numbers run out;
    * the capture accumulates every call's atoms. */
   if (unlikely(jd->measure_cap))
      mali_jm_measure_capture_atoms(jd->measure_cap, b->atoms, b->n);

   /* The chains were written through write-combined mappings and never
    * read back: make the stores visible before the GPU reads them. */
#if defined(__aarch64__)
   __asm__ volatile("dsb st" ::: "memory");
#else
   __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
   enum mali_kbase_result r = mali_kbase_jm_submit(jd->kb, b->atoms, b->n);
   jd->stats.job_submits++;
   if (b->trigger_idx >= 0) {
      b->trigger_fd = b->fences[b->trigger_idx].fd;
      b->trigger_idx = -1;
   }
   b->n = 0;
   b->last = 0;
   memset(b->slot_last, 0, sizeof(b->slot_last));
   if (r != MALI_KBASE_SUCCESS) {
      /* Some atoms of the call may be in the kernel and some not; there
       * is no way back from that. */
      jm_set_fault(jd, "JOB_SUBMIT failed: %s", mali_kbase_result_str(r));
      return VK_ERROR_DEVICE_LOST;
   }
   return VK_SUCCESS;
}

static mali_jm_ref build_atom(struct mali_device *dev, enum mali_jm_atom_kind kind,
                              uint8_t slot, uint32_t core_req, uint64_t jc,
                              const struct kb_fence *fence, uint64_t udata1,
                              mali_jm_ref dep0, mali_jm_ref dep1, unsigned reserve,
                              VkResult *result);

/*
 * Make atoms[0..n) end in a non-coalesced atom that waits for all of them
 * (and, for a tracker, for the newest atom of both slot orders), building a
 * join atom when the last one does not. Every slot-order atom of the call
 * is at or before the call's newest atom of its slot, so covering those
 * covers them all; atoms outside the slot orders report their own events.
 */
static VkResult
make_terminal(struct mali_device *dev, bool tracker, mali_jm_ref *out)
{
   struct mali_jm_device *jd = dev->jm;
   struct mali_jm_build *b = jd->build;
   const mali_jm_ref *must = tracker ? jd->queue->last : b->slot_last;

   *out = 0;
   mali_jm_ref t = b->n ? b->last : 0;
   bool need = false;
   for (unsigned s = 0; s < MALI_JM_SLOT_COUNT; s++)
      need |= !covers(jd, t, must[s]);
   if (!need && !t)
      return VK_SUCCESS;
   if (need) {
      VkResult result = VK_SUCCESS;
      /* The number kept in reserve for exactly this (take_number). */
      t = build_atom(dev, MALI_JM_ATOM_JOIN, MALI_JM_SLOT_NONE, KB_JM_REQ_DEP, 0, NULL, 0,
                     must[MALI_JM_SLOT_VTC], must[MALI_JM_SLOT_FRAG], 0, &result);
      if (!t)
         return result;
      jd->stats.joins++;
   }
   struct kb_jm_atom *a = &b->atoms[b->n - 1];
   assert(a->atom_number == mali_jm_ref_num(t));
   a->core_req &= ~KB_JM_REQ_EVENT_COALESCE;
   a->udata[1] |= MALI_JM_UDATA_TERMINAL | (tracker ? MALI_JM_UDATA_TRACKER : 0);
   *out = t;
   return VK_SUCCESS;
}

/*
 * An atom number, keeping `reserve` free for a terminal join. When there
 * are none: read events; if that is not enough, submit what is built so
 * far (ending in a terminal that brings its events back) and block
 * reading events until numbers come back. Every JOB_SUBMIT ends in a
 * non-coalesced atom that waits for the call's other atoms, so the
 * numbers in flight do come back as the GPU finishes.
 */
static bool
take_number(struct mali_device *dev, unsigned reserve, uint8_t *num, VkResult *result)
{
   struct mali_jm_device *jd = dev->jm;
   struct mali_jm_build *b = jd->build;
   const int64_t start = os_time_get_nano();
   bool warned = false;
   for (;;) {
      if (dev->lost || jd->fault) {
         *result = VK_ERROR_DEVICE_LOST;
         return false;
      }
      if (jd->free_ids > reserve) {
         unsigned got = mali_kbase_jm_atom_ids_alloc(&jd->ids, num, 1);
         assert(got == 1);
         (void)got;
         jd->free_ids--;
         jd->in_flight++;
         return true;
      }
      mali_jm_drain_locked(dev);
      if (jd->free_ids > reserve)
         continue;
      if (b->n) {
         mali_jm_ref t;
         VkResult r = make_terminal(dev, false, &t);
         if (r == VK_SUCCESS)
            r = flush(dev);
         if (r != VK_SUCCESS) {
            *result = r;
            return false;
         }
         jd->stats.partial_flushes++;
         continue;
      }
      jd->stats.number_waits++;
      jm_note_slow(dev, start, &warned, "waiting for a free job-manager atom number");
      mali_jm_wait_locked(dev, os_time_get_nano() + READER_SLICE_NS);
   }
}

/* The type of a dependency on in-flight atom r (file comment): DATA, so a
 * failure reaches the dependent; ORDER on a FENCE_WAIT. */
static uint8_t
dep_type(const struct mali_jm_device *jd, mali_jm_ref r)
{
   return jd->atoms[mali_jm_ref_num(r)].kind == MALI_JM_ATOM_FENCE_WAIT ? KB_JM_DEP_ORDER
                                                                        : KB_JM_DEP_DATA;
}

static mali_jm_ref
build_atom(struct mali_device *dev, enum mali_jm_atom_kind kind, uint8_t slot,
           uint32_t core_req, uint64_t jc, const struct kb_fence *fence, uint64_t udata1,
           mali_jm_ref dep0, mali_jm_ref dep1, unsigned reserve, VkResult *result)
{
   struct mali_jm_device *jd = dev->jm;
   struct mali_jm_build *b = jd->build;
   assert(jd->building);

   uint8_t num;
   if (!take_number(dev, reserve, &num, result))
      return 0;

   const mali_jm_ref ref = (++jd->next_serial << 8) | num;
   const unsigned i = b->n++;
   struct kb_jm_atom *a = &b->atoms[i];
   memset(a, 0, sizeof(*a));
   a->seq_nr = b->seq;
   a->jc = jc;
   if (fence) {
      b->fences[i] = *fence;
      a->jc = (uint64_t)(uintptr_t)&b->fences[i];
      if (kind == MALI_JM_ATOM_FENCE_TRIGGER)
         b->trigger_idx = (int)i;
   }
   a->udata[0] = b->seq;
   a->udata[1] = (udata1 & ~0xffull) | kind;
   a->atom_number = num;
   a->prio = KB_JM_PRIO_MEDIUM;
   a->core_req = core_req | (slot < MALI_JM_SLOT_COUNT ? KB_JM_REQ_EVENT_COALESCE : 0);

   struct mali_jm_atom_info *info = &jd->atoms[num];
   info->ref = ref;
   info->in_flight = true;
   info->kind = kind;
   info->slot = slot;
   memset(info->cov, 0, sizeof(info->cov));

   const mali_jm_ref deps[2] = {dep0, dep1};
   unsigned nd = 0;
   for (unsigned d = 0; d < 2; d++) {
      if (!mali_jm_ref_live(jd, deps[d]) || (d == 1 && deps[1] == deps[0]))
         continue;
      const struct mali_jm_atom_info *p = &jd->atoms[mali_jm_ref_num(deps[d])];
      a->pre_dep[nd].atom_id = mali_jm_ref_num(deps[d]);
      a->pre_dep[nd].dependency_type = dep_type(jd, deps[d]);
      nd++;
      for (unsigned s = 0; s < MALI_JM_SLOT_COUNT; s++)
         info->cov[s] = newer(info->cov[s], p->cov[s]);
   }
   if (slot < MALI_JM_SLOT_COUNT) {
      info->cov[slot] = ref;
      jd->queue->last[slot] = ref;
      b->slot_last[slot] = ref;
   }

   b->last = ref;
   b->any = true;
   jd->stats.atoms++;
   return ref;
}

mali_jm_ref
mali_jm_build_atom(struct mali_device *dev, enum mali_jm_atom_kind kind, uint8_t slot,
                   uint32_t core_req, uint64_t jc, const struct kb_fence *fence,
                   uint64_t udata1, mali_jm_ref dep0, mali_jm_ref dep1, VkResult *result)
{
   return build_atom(dev, kind, slot, core_req, jc, fence, udata1, dep0, dep1, 1, result);
}

unsigned
mali_jm_build_join_all(struct mali_device *dev, const mali_jm_ref *deps, unsigned n,
                       mali_jm_ref out[2], VkResult *result)
{
   struct mali_jm_device *jd = dev->jm;
   mali_jm_ref live[64];
   unsigned m = 0;
   for (unsigned i = 0; i < n && m < ARRAY_SIZE(live); i++) {
      if (mali_jm_ref_live(jd, deps[i]))
         live[m++] = deps[i];
   }
   /* Pairwise, until two are left for the consumer's two pre_deps. */
   while (m > 2) {
      unsigned k = 0;
      for (unsigned i = 0; i < m; i += 2) {
         if (i + 1 == m || m - i + k <= 2) {
            live[k++] = live[i];
            if (i + 1 < m)
               live[k++] = live[i + 1];
            continue;
         }
         mali_jm_ref j = mali_jm_build_atom(dev, MALI_JM_ATOM_JOIN, MALI_JM_SLOT_NONE,
                                            KB_JM_REQ_DEP, 0, NULL, 0, live[i], live[i + 1],
                                            result);
         if (!j)
            return 0;
         jd->stats.joins++;
         live[k++] = j;
      }
      m = k;
   }
   out[0] = m > 0 ? live[0] : 0;
   out[1] = m > 1 ? live[1] : 0;
   return m;
}

/* Give back the numbers of atoms built but not submitted. */
static void
discard_unsubmitted(struct mali_jm_device *jd)
{
   struct mali_jm_build *b = jd->build;
   for (unsigned i = 0; i < b->n; i++) {
      const uint8_t num = b->atoms[i].atom_number;
      jd->atoms[num].in_flight = false;
      mali_kbase_jm_atom_ids_free(&jd->ids, num);
      jd->free_ids++;
      jd->in_flight--;
   }
   b->n = 0;
   b->last = 0;
   memset(b->slot_last, 0, sizeof(b->slot_last));
}

static void
build_finish(struct mali_device *dev, mali_jm_ref t, mali_jm_ref *terminal, bool *built,
             int *trigger_fd)
{
   struct mali_jm_device *jd = dev->jm;
   struct mali_jm_build *b = jd->build;
   if (terminal)
      *terminal = t;
   if (built)
      *built = b->any;
   if (trigger_fd)
      *trigger_fd = b->trigger_fd;
   else if (b->trigger_fd >= 0)
      close(b->trigger_fd);
   jd->building = false;
   pthread_cond_broadcast(&dev->cond);
}

VkResult
mali_jm_build_end(struct mali_device *dev, bool tracker, mali_jm_ref *terminal, bool *built,
                  int *trigger_fd)
{
   struct mali_jm_device *jd = dev->jm;
   VkResult result = VK_ERROR_DEVICE_LOST;
   mali_jm_ref t = 0;

   if (!dev->lost && !jd->fault) {
      result = make_terminal(dev, tracker, &t);
      if (result == VK_SUCCESS)
         result = flush(dev);
   }
   if (result != VK_SUCCESS) {
      discard_unsubmitted(jd);
      t = 0;
   }
   build_finish(dev, t, terminal, built, trigger_fd);
   return result;
}

void
mali_jm_build_abort(struct mali_device *dev)
{
   struct mali_jm_device *jd = dev->jm;
   /* Atoms of earlier JOB_SUBMIT calls of this build stay in the kernel
    * (each call ended in its own terminal, so their numbers come back);
    * nothing marks a submission complete. */
   discard_unsubmitted(jd);
   jd->build->any = false;
   build_finish(dev, 0, NULL, NULL, NULL);
}

/* ---------------------------------------------------------------------- */
/* Submit                                                                  */

/* Destination stages that run on the fragment slot only, and on the vtc
 * slot only. Anything else (transfer, which either side may do, all
 * commands, top or bottom of pipe, none) waits on both. */
#define FRAG_ONLY_STAGES                                                     \
   (VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |                           \
    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |                                \
    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |                            \
    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT)
#define VTC_ONLY_STAGES                                                      \
   (VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT | \
    VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |                                  \
    VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT |                    \
    VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT |                 \
    VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | \
    VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT |                                    \
    VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT |                         \
    VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT)

#define GATE_FRAG 1u
#define GATE_VTC  2u

/* The slots a wait with this destination stage mask gates. On v7 a
 * fragment shader's pushed uniform buffer words are read by a copy job on
 * the vtc chain (mali_v7_ubo_copy), so a mask with the fragment shader
 * stage gates the vtc slot as well, as a barrier into that stage orders
 * the copy job (mali_jm_cmd_buffer.c ubo_copy_dep). */
static unsigned
stage_gate(VkPipelineStageFlags2 mask)
{
#if PAN_ARCH < 9
   if (mask & VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)
      return GATE_FRAG | GATE_VTC;
#endif
   if (mask && !(mask & ~FRAG_ONLY_STAGES))
      return GATE_FRAG;
   if (mask && !(mask & ~VTC_ONLY_STAGES))
      return GATE_VTC;
   return GATE_FRAG | GATE_VTC;
}

static struct mali_sync *
to_sync(struct vk_sync *s)
{
   assert(s->type == &MALI_PER_ARCH(sync_type));
   return container_of(s, struct mali_sync, vk);
}

static uint64_t
udata_of(unsigned cmd, unsigned batch)
{
   return ((uint64_t)(cmd & 0xffff) << 32) | ((uint64_t)(batch & 0xffff) << 16);
}

/* Extra core_req bits a chain needs: PERMON when it writes system
 * timestamps (queries, MALISX2_MEASURE), which keeps the GPU's timestamp
 * counter running while the atom is on the GPU, as the blob does for its
 * query chains. */
static inline uint32_t
chain_req(const struct mali_jm_chain *c)
{
   return c->timestamps ? KB_JM_REQ_PERMON : 0;
}

/* The atom graph of the submit's waits and command buffers (file
 * comment). With dev->lock held and the build begun. batch_frag has room
 * for the largest command buffer's batch count, allocated before the
 * build so that nothing but device loss can stop it half way (the queue
 * state it updates, q->carry included, would not match the atoms that
 * reached the kernel). */
static VkResult
build_submit(struct mali_device *dev, struct vk_queue_submit *submit, mali_jm_ref *batch_frag)
{
   struct mali_jm_device *jd = dev->jm;
   struct mali_jm_queue *q = jd->queue;
   VkResult result = VK_SUCCESS;

   /* The newest fragment-slot atom the submission's first vtc atom has to
    * wait for (semaphore waits with vtc-slot destination stages). */
   mali_jm_ref frag_req = 0;

   for (uint32_t i = 0; i < submit->wait_count; i++) {
      struct mali_sync *s = to_sync(submit->waits[i].sync);
      const unsigned gate = stage_gate(submit->waits[i].stage_mask);
      if (s->fd >= 0) {
         if (mali_sync_file_signaled(s->fd))
            continue;
         /* The kernel takes its own reference to the fence at submit. */
         const struct kb_fence f = {.fd = s->fd, .stream_fd = -1};
         const uint8_t slot = gate == GATE_VTC ? MALI_JM_SLOT_VTC : MALI_JM_SLOT_FRAG;
         mali_jm_ref w = mali_jm_build_atom(dev, MALI_JM_ATOM_FENCE_WAIT, slot,
                                            KB_JM_REQ_SOFT_FENCE_WAIT, 0, &f, udata_of(0, i),
                                            q->last[slot], 0, &result);
         if (!w)
            return result;
         jd->stats.fence_waits++;
         if (gate == (GATE_FRAG | GATE_VTC))
            frag_req = newer(frag_req, w);
         continue;
      }
      if (s->host_signaled || !s->submitted || q->completed_seq >= s->req[0])
         continue;
      /* Same queue: each slot is already in order behind the signalling
       * submission; what is left is the other slot. Vtc-slot work waits
       * for its fragment work, which is what its tracker covers on the
       * fragment slot. Fragment-slot work waits for its vtc work: a batch
       * with a vtc chain has its fragment atoms depend on that, which
       * follows the signaller's in the slot order; the first fragment atom
       * of a batch without one is told to wait for the newest vtc atom
       * (the carried requirement a barrier at the end of a command buffer
       * also sets). */
      if (!mali_jm_ref_live(jd, s->req[1]))
         continue;
      const struct mali_jm_atom_info *t = &jd->atoms[mali_jm_ref_num(s->req[1])];
      if (gate & GATE_VTC)
         frag_req = newer(frag_req, t->cov[MALI_JM_SLOT_FRAG]);
      if ((gate & GATE_FRAG) && mali_jm_ref_live(jd, t->cov[MALI_JM_SLOT_VTC]))
         q->carry |= MALI_JM_REQ_FRAG_AFTER_VTC;
   }

   for (uint32_t c = 0; c < submit->command_buffer_count; c++) {
      struct mali_cmd_buffer *cmd =
         container_of(submit->command_buffers[c], struct mali_cmd_buffer, vk);
      uint32_t count;
      const struct mali_jm_batch *batches = mali_jm_cmd_batches(cmd, &count);
      const struct mali_jm_frag_seg *segs = util_dynarray_begin(&cmd->jm.frags);

      for (uint32_t k = 0; k < count; k++) {
         const struct mali_jm_batch *bt = &batches[k];
         const uint64_t ud = udata_of(c, k);
         mali_jm_ref vtc = 0;

         if (bt->vtc.first) {
            mali_jm_ref req = frag_req;
            if (bt->vtc_after_frag == MALI_JM_AFTER_EARLIER ||
                (q->carry & MALI_JM_REQ_VTC_AFTER_FRAG)) {
               req = newer(req, q->last[MALI_JM_SLOT_FRAG]);
            } else {
               if (bt->vtc_after_frag == MALI_JM_AFTER_EARLIER_XFER ||
                   (q->carry & MALI_JM_REQ_VTC_AFTER_FRAG_XFER))
                  req = newer(req, q->last_frag_xfer);
               if (bt->vtc_after_frag && bt->vtc_after_frag < MALI_JM_AFTER_EARLIER_XFER &&
                   bt->vtc_after_frag - 1u < k)
                  req = newer(req, batch_frag[bt->vtc_after_frag - 1u]);
            }
            if (bt->heap_slot >= 0 && bt->heap_slot < MALI_JM_HEAP_SLOTS_MAX)
               req = newer(req, q->heap_last_frag[bt->heap_slot]);
            /* A transfer-only requirement carried from an earlier command
             * buffer: met by the first vtc chain with a transfer job, or
             * by any vtc atom that waits for as much anyway (later vtc
             * atoms follow it in the slot order). */
            if (q->xvtc_frag) {
               if (bt->vtc_xfer)
                  req = newer(req, q->xvtc_frag);
               if (req && newer(req, q->xvtc_frag) == req)
                  q->xvtc_frag = 0;
            }
            vtc = mali_jm_build_atom(dev, MALI_JM_ATOM_VTC, MALI_JM_SLOT_VTC,
                                     MALI_JM_REQ_VTC | chain_req(&bt->vtc), bt->vtc.first,
                                     NULL, ud, q->last[MALI_JM_SLOT_VTC], req, &result);
            if (!vtc)
               return result;
            frag_req = 0;
            q->carry &= ~(MALI_JM_REQ_VTC_AFTER_FRAG | MALI_JM_REQ_VTC_AFTER_FRAG_XFER);
         }

         mali_jm_ref frag = 0;
         for (uint32_t j = 0; j < bt->frag_count; j++) {
            const struct mali_jm_frag_seg *seg = &segs[bt->frag_first + j];
            if (!seg->chain.first)
               continue;
            mali_jm_ref d = vtc;
            if (!d && (bt->frag_after_vtc || (q->carry & MALI_JM_REQ_FRAG_AFTER_VTC)))
               d = q->last[MALI_JM_SLOT_VTC];
            frag = mali_jm_build_atom(dev, MALI_JM_ATOM_FRAG, MALI_JM_SLOT_FRAG,
                                      MALI_JM_REQ_FRAG | chain_req(&seg->chain),
                                      seg->chain.first, NULL, ud, d,
                                      q->last[MALI_JM_SLOT_FRAG], &result);
            if (!frag)
               return result;
            q->carry &= ~MALI_JM_REQ_FRAG_AFTER_VTC;
         }
         if (frag && bt->heap_slot >= 0 && bt->heap_slot < MALI_JM_HEAP_SLOTS_MAX)
            q->heap_last_frag[bt->heap_slot] = frag;
         if (frag && bt->frag_xfer)
            q->last_frag_xfer = frag;
         /* The newest fragment-slot atom at or before this batch. */
         batch_frag[k] = q->last[MALI_JM_SLOT_FRAG];
      }
      /* Barriers at its end order the next atoms of later command
       * buffers (and submissions). */
      q->carry |= cmd->jm.end_req & (MALI_JM_REQ_VTC_AFTER_FRAG | MALI_JM_REQ_FRAG_AFTER_VTC |
                                     MALI_JM_REQ_VTC_AFTER_FRAG_XFER);
      if (cmd->jm.end_req & MALI_JM_REQ_XVTC_AFTER_FRAG)
         q->xvtc_frag = newer(q->xvtc_frag, q->last[MALI_JM_SLOT_FRAG]);
      else if (cmd->jm.end_req & MALI_JM_REQ_XVTC_AFTER_FRAG_XFER)
         q->xvtc_frag = newer(q->xvtc_frag, q->last_frag_xfer);
   }
   return VK_SUCCESS;
}

/*
 * Command buffers that ran before and may run again (recorded without
 * ONE_TIME_SUBMIT): wait until their last run has completed, which only a
 * SIMULTANEOUS_USE command buffer may not have (one job chain cannot run
 * twice at once, so such submissions run one after the other), then
 * restore the words the GPU wrote (mali_jm_cmd_prepare_submit). With
 * dev->lock held; drops it while waiting.
 */
static VkResult
prepare_command_buffers(struct mali_device *dev, struct vk_queue_submit *submit)
{
   struct mali_jm_device *jd = dev->jm;
   struct mali_jm_queue *q = jd->queue;

   for (uint32_t c = 0; c < submit->command_buffer_count; c++) {
      struct mali_cmd_buffer *cmd =
         container_of(submit->command_buffers[c], struct mali_cmd_buffer, vk);
      if (!cmd->jm.resubmit)
         continue;
      for (uint32_t d = 0; d < c; d++) {
         if (submit->command_buffers[d] == submit->command_buffers[c])
            return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                             "a command buffer twice in one submission (SIMULTANEOUS_USE) "
                             "is not supported on the job manager");
      }
      if (!cmd->jm.submitted)
         continue;
      const int64_t start = os_time_get_nano();
      bool warned = false;
      while (q->completed_seq < cmd->jm.last_seq && !jd->fault && !dev->lost) {
         jm_note_slow(dev, start, &warned,
                      "vkQueueSubmit, waiting for a command buffer's previous run");
         mali_jm_wait_locked(dev, os_time_get_nano() + READER_SLICE_NS);
      }
      if (jd->fault || dev->lost)
         return VK_ERROR_DEVICE_LOST;
      mali_jm_cmd_prepare_submit(cmd);
   }
   return VK_SUCCESS;
}

VkResult
MALI_PER_ARCH(queue_submit)(struct vk_queue *vkq, struct vk_queue_submit *submit)
{
   struct mali_device *dev = container_of(vkq->base.device, struct mali_device, vk);
   struct mali_jm_device *jd = dev->jm;
   struct mali_jm_queue *q = jd->queue;

   if (vk_queue_submit_has_bind(submit))
      return vk_errorf(vkq, VK_ERROR_FEATURE_NOT_PRESENT, "sparse binding is not supported");

   pthread_mutex_lock(&dev->lock);
   /* Free atom numbers and notice faults first. */
   const char *msg = MALI_PER_ARCH(queue_wait)(dev);
   if (msg || dev->lost) {
      char copy[sizeof(jd->fault_msg)];
      snprintf(copy, sizeof(copy), "%s", msg ? msg : "");
      pthread_mutex_unlock(&dev->lock);
      if (msg)
         report_fault(dev, copy);
      return VK_ERROR_DEVICE_LOST;
   }

   VkResult result = prepare_command_buffers(dev, submit);
   if (result != VK_SUCCESS) {
      char copy[sizeof(jd->fault_msg)];
      const bool fault = jd->fault && !dev->lost;
      if (fault)
         snprintf(copy, sizeof(copy), "%s", jd->fault_msg);
      pthread_mutex_unlock(&dev->lock);
      if (fault)
         report_fault(dev, copy);
      return result;
   }

   const uint64_t seqno = q->seq + 1;

   /* Measurement: queue the command buffers' timed regions for
    * reading, and capture this submit's atoms and command memory if
    * asked to. */
   if (unlikely(dev->measure)) {
      mali_jm_measure_submit(dev, submit, seqno);
      jd->measure_cap = mali_jm_measure_capture_begin(dev, submit, seqno);
   }

   /* The per-batch scratch of build_submit, before anything is built. */
   uint32_t max_batches = 1;
   for (uint32_t c = 0; c < submit->command_buffer_count; c++) {
      uint32_t count;
      mali_jm_cmd_batches(container_of(submit->command_buffers[c], struct mali_cmd_buffer, vk),
                          &count);
      max_batches = MAX2(max_batches, count);
   }
   mali_jm_ref *batch_frag = malloc(max_batches * sizeof(*batch_frag));
   if (!batch_frag) {
      if (unlikely(jd->measure_cap)) {
         mali_jm_measure_capture_end(dev, jd->measure_cap);
         jd->measure_cap = NULL;
      }
      pthread_mutex_unlock(&dev->lock);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   mali_jm_build_begin(dev, seqno);
   result = build_submit(dev, submit, batch_frag);
   free(batch_frag);
   mali_jm_ref tracker = 0;
   bool built = false;
   if (result == VK_SUCCESS)
      result = mali_jm_build_end(dev, true, &tracker, &built, NULL);
   else
      mali_jm_build_abort(dev);

   if (unlikely(jd->measure_cap)) {
      mali_jm_measure_capture_end(dev, jd->measure_cap);
      jd->measure_cap = NULL;
   }

   if (result == VK_SUCCESS) {
      if (built) {
         q->seq++;
         q->tracker = tracker;
         /* Everything already finished and was read meanwhile. */
         if (!tracker && q->completed_seq < q->seq)
            q->completed_seq = q->seq;
      }
      for (uint32_t c = 0; c < submit->command_buffer_count; c++) {
         struct mali_cmd_buffer *cmd =
            container_of(submit->command_buffers[c], struct mali_cmd_buffer, vk);
         cmd->jm.submitted = true;
         cmd->jm.last_seq = q->seq;
      }

      /* Binary semaphore waits consume the payload; signals are pending on
       * this submission, or on the last one with atoms if this one has
       * none (its work is then everything before it). */
      for (uint32_t i = 0; i < submit->wait_count; i++) {
         struct mali_sync *s = to_sync(submit->waits[i].sync);
         if (!(s->vk.flags & VK_SYNC_IS_TIMELINE))
            MALI_PER_ARCH(sync_clear)(s);
      }
      for (uint32_t i = 0; i < submit->signal_count; i++) {
         struct mali_sync *s = to_sync(submit->signals[i].sync);
         MALI_PER_ARCH(sync_clear)(s);
         s->submitted = true;
         s->req[0] = q->seq;
         s->req[1] = q->tracker;
      }
      dev->stats.submits++;
      pthread_cond_broadcast(&dev->cond);
   }

   char copy[sizeof(jd->fault_msg)];
   const bool fault = jd->fault && !dev->lost;
   if (fault)
      snprintf(copy, sizeof(copy), "%s", jd->fault_msg);
   pthread_mutex_unlock(&dev->lock);
   if (fault)
      report_fault(dev, copy);
   return result;
}

/* ---------------------------------------------------------------------- */
/* Device and queue                                                        */

VkResult
MALI_PER_ARCH(device_init)(struct mali_device *dev)
{
   struct mali_jm_device *jd =
      vk_zalloc(&dev->vk.alloc, sizeof(*jd), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   struct mali_jm_build *b =
      vk_zalloc(&dev->vk.alloc, sizeof(*b), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!jd || !b) {
      vk_free(&dev->vk.alloc, jd);
      vk_free(&dev->vk.alloc, b);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   jd->dev = dev;
   jd->kb = dev->kbase;
   jd->build = b;
   jd->stream_fd = -1;
   /* JS_FEATURE_CACHE_FLUSH_JOB of the fragment slot. */
   jd->frag_cache_flush = (dev->kbase->props.js_features[0] >> 3) & 1;
   mali_kbase_jm_atom_ids_init(&jd->ids);
   jd->free_ids = mali_kbase_jm_atom_ids_free_count(&jd->ids);
   mali_jm_heap_init(jd);

   dev->jm = jd;
   dev->fe = jd;
   dev->vk.check_status = MALI_PER_ARCH(device_check_status);
   return VK_SUCCESS;
}

void
MALI_PER_ARCH(device_finish)(struct mali_device *dev)
{
   struct mali_jm_device *jd = dev->jm;
   if (!jd)
      return;
   /* MALI_PER_ARCH(device_quiesce) ran first: nothing is in flight, or
    * the context was abandoned and the frees below leave memory alone.
    * mali_kbase_destroy ends the context with POST_TERM. */
   pthread_mutex_lock(&dev->lock);
   mali_jm_drain_locked(dev);
   pthread_mutex_unlock(&dev->lock);
   mali_jm_sync_file_finish(jd);
   mali_jm_heap_finish(jd);
#if PAN_ARCH < 9
   mali_v7_desc_finish(dev);
#endif

   dev->vk.check_status = NULL;
   dev->jm = NULL;
   dev->fe = NULL;
   vk_free(&dev->vk.alloc, jd->build);
   vk_free(&dev->vk.alloc, jd);
}

void
MALI_PER_ARCH(device_quiesce)(struct mali_device *dev)
{
   struct mali_jm_device *jd = dev->jm;
   if (!jd)
      return;

   pthread_mutex_lock(&dev->lock);
   const int64_t deadline = os_time_get_nano() + MALI_JM_TEARDOWN_WAIT_NS;
   bool readable = true;
   while (jd->in_flight && readable) {
      const int64_t now = os_time_get_nano();
      if (now >= deadline)
         break;
      const int ms = (int)MIN2((deadline - now + 999999) / 1000000, READER_SLICE_NS / 1000000);
      const int r = mali_kbase_jm_poll(jd->kb, ms);
      if (r < 0 && errno != EINTR)
         break;
      if (r <= 0)
         continue;
      /* Unlike read_events_locked, go on after a fault: a lost device's
       * atoms still complete, and their events still come back. */
      for (;;) {
         struct kb_jm_event ev[MALI_JM_EVENTS_PER_READ];
         unsigned n = 0;
         bool terminated = false;
         if (mali_kbase_jm_read_events(jd->kb, ev, MALI_JM_EVENTS_PER_READ, &n, &terminated) !=
                MALI_KBASE_SUCCESS ||
             terminated) {
            readable = false;
            break;
         }
         for (unsigned i = 0; i < n; i++)
            process_event(dev, &ev[i]);
         if (n < MALI_JM_EVENTS_PER_READ)
            break;
      }
   }
   const unsigned left = jd->in_flight;
   pthread_mutex_unlock(&dev->lock);

   if (left) {
      mesa_logw("malisx2: %u job-manager atoms are still in flight at device destruction; "
                "leaving the context and its memory to the kernel until the process exits",
                left);
      mali_kbase_abandon(jd->kb);
   }
}

VkResult
MALI_PER_ARCH(queue_init)(struct mali_device *dev, struct mali_queue *queue)
{
   struct mali_jm_device *jd = dev->jm;
   struct mali_jm_queue *q =
      vk_zalloc(&dev->vk.alloc, sizeof(*q), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!q)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   q->dev = dev;

   pthread_mutex_lock(&dev->lock);
   jd->queue = q;
   pthread_mutex_unlock(&dev->lock);
   queue->jm = q;
   queue->fe = q;
   return VK_SUCCESS;
}

void
MALI_PER_ARCH(queue_finish)(struct mali_device *dev, struct mali_queue *queue)
{
   struct mali_jm_queue *q = queue->jm;
   if (!q)
      return;
   pthread_mutex_lock(&dev->lock);
   mali_jm_drain_locked(dev);
   if (dev->jm && dev->jm->queue == q)
      dev->jm->queue = NULL;
   pthread_mutex_unlock(&dev->lock);
   queue->jm = NULL;
   queue->fe = NULL;
   vk_free(&dev->vk.alloc, q);
}
