/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The CSF side of a VkDevice and its VkQueue:
 *
 *  - struct mali_csf_device: per device. Scoreboard facts, the event
 *    thread that reads the kbase event channel, the queue group handle,
 *    and the kcpu queue. The state every frontend needs alike (the
 *    command-buffer slab cache, internal shader caches, the sync-object
 *    lock and condition, and device loss) lives on struct mali_device
 *    instead (mali_vk.h).
 *  - struct mali_csf_queue: per VkQueue. One kbase queue group with one CS
 *    queue (ring) per subqueue, the tiler heap, the GPU sync objects, and
 *    the submit counter.
 *  - struct mali_sync: the vk_sync type behind VkFence and binary
 *    VkSemaphore. A signal operation of submit N makes the sync "pending
 *    on N"; it is signalled once every subqueue has finished submit N's
 *    work, which each subqueue reports by writing N into its own "done"
 *    sync object in GPU memory.
 *
 * mali_sync.c and mali_wsi.c are mostly frontend-neutral; the few
 * operations that genuinely differ per frontend (whether a submit's work
 * has been reached, noticing a fault before the next event, and sync-file
 * import/export, how a host wait sleeps) are declared below as
 * MALI_PER_ARCH() hooks. This header has the v11 (CSF) bodies, defined in
 * mali_queue.c, mali_sync_file.c and inline below; the v9 (JM) bodies are
 * in mali_jm.h, mali_jm_queue.c and mali_jm_sync_file.c.
 */

#ifndef MALI_QUEUE_H
#define MALI_QUEUE_H

#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "util/list.h"
#include "util/simple_mtx.h"
#include "vk_queue.h"
#include "vk_sync.h"

#include "kbase/kbase.h"

#include "mali_cs.h"
/* mali_cs.h sets PAN_ARCH from MALI_PAN_ARCH, which mali_arch.h needs. */
#include "mali_arch.h"
#include "mali_vk.h"

struct mali_shader;

struct mali_csf_device {
   struct mali_device *dev;
   struct mali_kbase *kb;
   struct mali_cs_sb_info sb;
   uint8_t nr_registers;          /* CS work registers (96) */
   /* The vertex/tiler -> fragment pass dependency goes through a shared
    * scoreboard entry (mali_cs.h) instead of the vertex/tiler barrier
    * counter: gpu_features bit 3, as the blob decides. */
   bool shared_sb;

   /* The event thread: reads the kbase event channel, wakes waiters,
    * turns a queue-group error into device loss. */
   pthread_t thread;
   bool thread_running;
   int wake_fd;                   /* eventfd that stops the thread */
   uint8_t group_handle;          /* the group whose errors mean loss */
   bool group_valid;

   /* The device's one queue (MALI_QUEUE_COUNT is 1). */
   struct mali_csf_queue *queue;

   /* Sync-file import and export (mali_sync_file.c): the kcpu queue,
    * created on first use. Under dev->lock. */
   struct mali_kcpu *kcpu;
};

/* Where the queue's sync memory keeps what (one page, CSF event memory). */
#define MALI_QUEUE_SYNCOBJS_OFFSET 0x000
#define MALI_QUEUE_DONE_OFFSET     0x040
/* The kcpu queue sets this sync64 to N once the N-th sync file a submit
 * waits for has signalled (mali_sync_file.c). */
#define MALI_QUEUE_FENCE_OFFSET    0x080
/* Per subqueue, a sync64 its ring adds one to after a command buffer when
 * a later command buffer of the same submit must follow it (mali_queue.c,
 * "Compute-subqueue ordering across command buffers"). */
#define MALI_QUEUE_HANDOFF_OFFSET  0x0a0
#define MALI_QUEUE_CTX_OFFSET(sq)  (0x100 + (sq) * sizeof(struct mali_cs_subqueue_ctx))

/* The tiler heap block: a 64 KiB geometry buffer, then the Tiler Heap
 * descriptor. */
#define MALI_TILER_GEOM_BUF_SIZE   (64 * 1024)
#define MALI_TILER_HEAP_DESC_OFFSET MALI_TILER_GEOM_BUF_SIZE
#define MALI_TILER_HEAP_CHUNK_SIZE (2 * 1024 * 1024)
#define MALI_TILER_HEAP_MAX_CHUNKS 552

#define MALI_RING_SIZE (64 * 1024)
#define MALI_RING_PRIORITY 8

struct mali_csf_queue {
   struct mali_device *dev;

   uint8_t group;
   bool group_created;

   struct mali_kbase_bo heap_block;
   uint64_t heap_ctx;               /* from CS_TILER_HEAP_INIT */
   bool heap_created;

   struct mali_kbase_bo sync_mem;
   struct mali_kbase_queue sq[MALI_SUBQUEUE_COUNT];

   /* Submits handed to the rings so far; submit N writes N into done[]. */
   uint64_t seqno;
   /* The last submit number each subqueue will write into its done slot;
    * a subqueue that had no work since keeps its old value. */
   uint64_t last_signal[MALI_SUBQUEUE_COUNT];

   /* Compute-subqueue ordering across command buffers (mali_queue.c):
    * done values of each subqueue that a stream relying on the compute
    * subqueue's earlier work (prior) or the compute subqueue's next work
    * (after_lazy) must wait for; ring_waited[k][j], the highest done value
    * of j ring k already waited for; handoff[j], signals ring j made on
    * its handoff counter so far. */
   uint64_t compute_prior[MALI_SUBQUEUE_COUNT];
   uint64_t compute_after_lazy[MALI_SUBQUEUE_COUNT];
   uint64_t ring_waited[MALI_SUBQUEUE_COUNT][MALI_SUBQUEUE_COUNT];
   uint64_t handoff[MALI_SUBQUEUE_COUNT];
};

static inline volatile struct mali_cs_sync64 *
mali_queue_done(const struct mali_csf_queue *q)
{
   return (volatile struct mali_cs_sync64 *)((uint8_t *)q->sync_mem.cpu +
                                             MALI_QUEUE_DONE_OFFSET);
}

/* The barrier counter of subqueue sq (struct mali_cs_sync64). The device
 * has one queue (MALI_QUEUE_COUNT), so command buffers use its address as
 * an immediate instead of loading it from the subqueue context. */
static inline uint64_t
mali_queue_syncobj_va(const struct mali_csf_queue *q, unsigned sq)
{
   return q->sync_mem.gpu_va + MALI_QUEUE_SYNCOBJS_OFFSET +
          sq * sizeof(struct mali_cs_sync64);
}

/* The tiler heap's Tiler Heap descriptor and the geometry buffer. */
static inline uint64_t
mali_queue_heap_desc_va(const struct mali_csf_queue *q)
{
   return q->heap_block.gpu_va + MALI_TILER_HEAP_DESC_OFFSET;
}

static inline uint64_t
mali_queue_geom_buf_va(const struct mali_csf_queue *q)
{
   return q->heap_block.gpu_va;
}

static inline uint64_t
mali_queue_done_va(const struct mali_csf_queue *q, unsigned sq)
{
   return q->sync_mem.gpu_va + MALI_QUEUE_DONE_OFFSET +
          sq * sizeof(struct mali_cs_sync64);
}

VkResult mali_csf_device_init(struct mali_device *dev);
void mali_csf_device_finish(struct mali_device *dev);

VkResult mali_csf_queue_init(struct mali_device *dev, struct mali_queue *queue);
void mali_csf_queue_finish(struct mali_device *dev, struct mali_queue *queue);

/* vk_queue::driver_submit. */
VkResult mali_queue_submit(struct vk_queue *vkq, struct vk_queue_submit *submit);

/* vk_device::check_status: DEVICE_LOST after a queue-group error or a
 * faulted sync object. */
VkResult mali_device_check_status(struct vk_device *vkdev);

/* Report device loss to the runtime and the log (once), then mark it and
 * wake every waiter. Takes dev->lock; call without it. */
void mali_device_set_lost(struct mali_device *dev, const char *fmt, ...)
   __attribute__((format(printf, 2, 3)));

/*
 * With dev->lock held: does this device's frontend have a fault that has
 * not been reported yet? CSF: a done slot with its error word set (a
 * stream faulted, whether or not the kernel's event has been read yet).
 * JM will scan atom events on demand instead. Returns the message for
 * mali_device_set_lost (called after the lock is dropped), or NULL.
 * Shared by mali_device_check_status, mali_queue_submit
 * and a host wait (mali_sync.c), so a wait notices a fault without
 * depending on the event thread to have run first.
 */
const char *MALI_PER_ARCH(queue_wait)(struct mali_device *dev);

/* ---------------------------------------------------------------------- */
/* Sync objects                                                            */

struct mali_sync {
   struct vk_sync vk;
   /* Under mali_device::lock. */
   bool host_signaled;
   bool submitted;                  /* a submit will signal it */
   uint64_t req[MALI_SUBQUEUE_COUNT];   /* ... when done[i] >= req[i] for all i */
   int fd;                          /* an imported sync file, or -1 */
};

extern const struct vk_sync_type MALI_PER_ARCH(sync_type);

/* The device's sync types, binary first and then the timeline semaphore
 * type wrapped around it (mali_sync.c): what vk_physical_device_init gets. */
const struct vk_sync_type *const *MALI_PER_ARCH(sync_types)(void);

/* Whether sync is one value of a timeline semaphore (a point sync): the
 * queue does not consume those when it waits on them. */
bool MALI_PER_ARCH(sync_is_timeline_point)(const struct vk_sync *sync);

/* Drop a sync's payload (close its sync file). With dev->lock held. */
void MALI_PER_ARCH(sync_clear)(struct mali_sync *s);

/* Log a failed sync-file import or export (mali_sync.c; thinned by
 * mali_diag_should_log). op says which ("import of", "export to", ...);
 * fd is -1 when there is none to name. Frontend-neutral. */
void MALI_PER_ARCH(sync_file_log_error)(struct mali_device *dev, const char *op, int fd,
                                        const char *why);

/* With dev->lock held: has every subqueue reached req? (a fault in a done
 * slot counts as device loss, not as "reached"; mali_sync.c's own
 * dev->lost check covers that separately). */
bool MALI_PER_ARCH(queue_reached)(struct mali_device *dev, const uint64_t req[MALI_SUBQUEUE_COUNT]);

/* For the slow-wait log (mali_sync.c): the queue's state as one line.
 * With dev->lock held. */
static inline void
MALI_PER_ARCH(queue_describe)(struct mali_device *dev, char *buf, size_t size)
{
   snprintf(buf, size, "%llu submits, %llu kernel events read, event thread %s",
            (unsigned long long)dev->stats.submits,
            (unsigned long long)dev->stats.kernel_events,
            dev->csf && dev->csf->thread_running ? "running" : "not running");
}

/* Does an event thread wake host waits when the GPU finishes? (Decides
 * only how often a sleeping wait re-checks; mali_sync.c.) */
static inline bool
MALI_PER_ARCH(queue_has_notifier)(struct mali_device *dev)
{
   return dev->csf && dev->csf->thread_running;
}

/* A host wait's sleep, with dev->lock held: until a broadcast of
 * dev->cond (the event thread, a submit, a host signal) or until_ns. */
static inline void
MALI_PER_ARCH(queue_sleep)(struct mali_device *dev, int64_t until_ns)
{
   struct timespec ts = {
      .tv_sec = until_ns / 1000000000ll,
      .tv_nsec = until_ns % 1000000000ll,
   };
   pthread_cond_timedwait(&dev->cond, &dev->lock, &ts);
}

/* ---------------------------------------------------------------------- */
/* Sync files (mali_sync_file.c)                                           */

/*
 * GPU wait on a sync file: the kcpu queue waits for fd, then sets the
 * queue's fence sync64 to *value; a ring waits for "fence > *value - 1".
 * The kernel takes its own reference, so fd may be closed afterwards.
 * With csf->dev->lock held.
 */
VkResult mali_sync_file_gpu_wait(struct mali_csf_device *csf, int fd, uint64_t *value);

/*
 * A new sync file that signals once every sync file in fds has signalled
 * and every subqueue has reached req (req may be NULL). *out = -1 when
 * nothing is left to wait for. With csf->dev->lock held.
 */
VkResult mali_sync_file_create(struct mali_csf_device *csf, uint32_t fd_count,
                               const int *fds, const uint64_t *req, int *out);

/* Delete the kcpu queue (before the queue memory it points at goes). */
void mali_sync_file_finish(struct mali_csf_device *csf);

/* Is the sync file signalled (poll without waiting)? */
bool mali_sync_file_signaled(int fd);

/* The queue's fence sync64 GPU address. */
static inline uint64_t
mali_queue_handoff_va(const struct mali_csf_queue *q, unsigned sq)
{
   return q->sync_mem.gpu_va + MALI_QUEUE_HANDOFF_OFFSET + sq * sizeof(struct mali_cs_sync64);
}

static inline uint64_t
mali_queue_fence_va(const struct mali_csf_queue *q)
{
   return q->sync_mem.gpu_va + MALI_QUEUE_FENCE_OFFSET;
}

/*
 * Frontend-neutral callers (mali_sync.c, mali_wsi.c) reach the sync-file
 * primitives above through these, so the call sites do not name the CSF
 * type. mali_sync_file.c stays CSF-only; a JM frontend gets its own
 * mali_jm_sync_file.c and its own MALI_PER_ARCH bodies, not these inline
 * ones.
 */
static inline VkResult
MALI_PER_ARCH(sync_file_gpu_wait)(struct mali_device *dev, int fd, uint64_t *value)
{
   return mali_sync_file_gpu_wait(dev->csf, fd, value);
}

static inline VkResult
MALI_PER_ARCH(sync_file_create)(struct mali_device *dev, uint32_t fd_count, const int *fds,
                                const uint64_t *req, int *out)
{
   return mali_sync_file_create(dev->csf, fd_count, fds, req, out);
}

#endif
