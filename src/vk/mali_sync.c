/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The vk_sync type behind VkFence and binary VkSemaphore. There is one
 * queue per device, so a sync is fully described by "signalled from the
 * host", or "pending on the done values the queue's subqueues will reach"
 * (mali_queue.c), or neither. Host waits read the done slots in GPU memory
 * and sleep on the device's condition variable, which the event thread
 * broadcasts on every kernel event. Timeline semaphores are not supported:
 * ARMSX2 does not use them and the device does not report the feature.
 *
 * A third kind of payload is an imported sync file (the Android acquire
 * fence): the sync owns a dup of the fd. Host waits poll it; GPU waits go
 * through the kcpu queue (mali_sync_file.c, mali_queue.c). Export makes a
 * sync file for whatever the payload is.
 *
 * This file is mostly frontend-neutral: the lock, condition and
 * device-loss state it uses live on struct mali_device, not the CSF
 * device. The handful of operations that genuinely differ per frontend
 * (has a submit's work been reached, noticing a fault
 * before the next event, sync-file import/export) go through the
 * MALI_PER_ARCH() hooks declared in mali_queue.h (CSF, v11) and mali_jm.h
 * (job manager, v7 and v9). The file is built once per arch; on the job
 * manager a GPU signal's payload is a submission number and its tracker
 * atom instead of per-subqueue done values, and a sleeping host wait reads
 * kbase events itself (mali_jm_queue.c) instead of waiting for an event
 * thread.
 */

#if defined(PAN_ARCH) && PAN_ARCH < 10
#include "mali_jm.h"
#else
#include "mali_queue.h"
#endif

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "util/log.h"
#include "util/os_time.h"

#include "vk_enum_to_str.h"

#include "mali_vk.h"

/* A failed sync-file import or export, in the log (see
 * mali_diag_should_log); fd is -1 when there is none to name. */
void
MALI_PER_ARCH(sync_file_log_error)(struct mali_device *dev, const char *op, int fd,
                                   const char *why)
{
   uint32_t n;
   if (!mali_diag_should_log(&dev->diag.sync_file_errors, &n))
      return;
   if (fd >= 0)
      mesa_logw("malisx2: %s sync file fd %d failed: %s (sync-file error %u)", op, fd, why, n);
   else
      mesa_logw("malisx2: %s a sync file failed: %s (sync-file error %u)", op, why, n);
}

static struct mali_sync *
to_sync(struct vk_sync *s)
{
   return container_of(s, struct mali_sync, vk);
}

static struct mali_device *
dev_of(struct vk_device *vkdev)
{
   return container_of(vkdev, struct mali_device, vk);
}

void
MALI_PER_ARCH(sync_clear)(struct mali_sync *s)
{
   s->host_signaled = false;
   s->submitted = false;
   if (s->fd >= 0) {
      close(s->fd);
      s->fd = -1;
   }
}

static VkResult
sync_init(struct vk_device *vkdev, struct vk_sync *vs, uint64_t initial_value)
{
   struct mali_sync *s = to_sync(vs);
   s->host_signaled = initial_value != 0;
   s->submitted = false;
   s->fd = -1;
   memset(s->req, 0, sizeof(s->req));
   return VK_SUCCESS;
}

static void
sync_finish(struct vk_device *vkdev, struct vk_sync *vs)
{
   struct mali_sync *s = to_sync(vs);
   if (s->fd >= 0)
      close(s->fd);
   s->fd = -1;
}

static VkResult
sync_signal(struct vk_device *vkdev, struct vk_sync *vs, uint64_t value)
{
   struct mali_device *dev = dev_of(vkdev);
   struct mali_sync *s = to_sync(vs);
   pthread_mutex_lock(&dev->lock);
   MALI_PER_ARCH(sync_clear)(s);
   s->host_signaled = true;
   pthread_cond_broadcast(&dev->cond);
   pthread_mutex_unlock(&dev->lock);
   return VK_SUCCESS;
}

static VkResult
sync_reset(struct vk_device *vkdev, struct vk_sync *vs)
{
   struct mali_device *dev = dev_of(vkdev);
   struct mali_sync *s = to_sync(vs);
   pthread_mutex_lock(&dev->lock);
   MALI_PER_ARCH(sync_clear)(s);
   pthread_mutex_unlock(&dev->lock);
   return VK_SUCCESS;
}

static VkResult
sync_move(struct vk_device *vkdev, struct vk_sync *dst, struct vk_sync *src)
{
   struct mali_device *dev = dev_of(vkdev);
   struct mali_sync *d = to_sync(dst), *s = to_sync(src);
   pthread_mutex_lock(&dev->lock);
   MALI_PER_ARCH(sync_clear)(d);
   d->host_signaled = s->host_signaled;
   d->submitted = s->submitted;
   memcpy(d->req, s->req, sizeof(d->req));
   d->fd = s->fd;
   s->fd = -1;
   MALI_PER_ARCH(sync_clear)(s);
   pthread_mutex_unlock(&dev->lock);
   return VK_SUCCESS;
}

/* With dev->lock held. A sync file found signalled becomes a host
 * signal (its fd is closed). */
static bool
sync_is_signaled(struct mali_device *dev, struct mali_sync *s, bool pending)
{
   if (s->host_signaled)
      return true;
   if (s->fd >= 0) {
      if (pending)
         return true;
      if (!mali_sync_file_signaled(s->fd))
         return false;
      MALI_PER_ARCH(sync_clear)(s);
      s->host_signaled = true;
      return true;
   }
   if (!s->submitted)
      return false;
   /* In immediate submit mode a submitted signal is already "pending". */
   return pending || MALI_PER_ARCH(queue_reached)(dev, s->req);
}

/*
 * The slow-wait log: what a host wait is still waiting for once it has
 * blocked for MALI_SLOW_WAIT_NS (once per wait; mali_diag_should_log
 * thins repeats). With dev->lock held.
 */
static void
note_slow_wait(struct mali_device *dev, uint32_t count, const struct vk_sync_wait *waits,
               bool any, uint64_t abs_timeout_ns, int64_t waited_ns, uint32_t *report)
{
   uint32_t n;
   if (!mali_diag_should_log(&dev->diag.slow_waits, &n))
      return;
   *report = n;

   char what[192];
   size_t len = 0;
   unsigned listed = 0, more = 0;
   what[0] = 0;
   for (uint32_t i = 0; i < count; i++) {
      struct mali_sync *s = to_sync(waits[i].sync);
      if (s->host_signaled)
         continue;
      if (listed == 4) {
         more++;
         continue;
      }
      int r;
      if (s->fd >= 0)
         r = snprintf(what + len, sizeof(what) - len, "%ssync file fd %d", listed ? ", " : "",
                      s->fd);
      else if (s->submitted)
         r = snprintf(what + len, sizeof(what) - len, "%sGPU work up to submission %llu",
                      listed ? ", " : "", (unsigned long long)s->req[0]);
      else
         r = snprintf(what + len, sizeof(what) - len, "%sa sync nothing has signalled yet",
                      listed ? ", " : "");
      if (r > 0)
         len = MIN2(len + (size_t)r, sizeof(what) - 1);
      listed++;
   }
   if (more)
      snprintf(what + len, sizeof(what) - len, " and %u more", more);

   char timeout[48];
   if (abs_timeout_ns >= (uint64_t)INT64_MAX)
      snprintf(timeout, sizeof(timeout), "no timeout");
   else
      snprintf(timeout, sizeof(timeout), "timeout in %lld ms",
               (long long)(((int64_t)abs_timeout_ns - os_time_get_nano()) / 1000000));
   char state[320];
   MALI_PER_ARCH(queue_describe)(dev, state, sizeof(state));
   mesa_logw("malisx2: a host wait (%s of %u, %s) has blocked for %lld ms (slow wait %u); "
             "waiting for %s; queue: %s",
             any ? "any" : "all", count, timeout, (long long)(waited_ns / 1000000), n,
             listed ? what : "nothing pending", state);
}

/* Without kernel events a sleeping wait re-reads the done slots this
 * often; with them, this only bounds the damage of a missed event. */
#define WAIT_SLICE_NO_EVENTS_NS (1ll * 1000 * 1000)
#define WAIT_SLICE_EVENTS_NS (20ll * 1000 * 1000)

static VkResult
sync_wait_many(struct vk_device *vkdev, uint32_t count,
               const struct vk_sync_wait *waits, enum vk_sync_wait_flags flags,
               uint64_t abs_timeout_ns)
{
   struct mali_device *dev = dev_of(vkdev);
   const bool any = flags & VK_SYNC_WAIT_ANY;
   const bool pending = flags & VK_SYNC_WAIT_PENDING;
   /* Whether something wakes a sleeping wait when the GPU finishes (CSF's
    * event thread; on JM the sleeping wait reads events itself) decides
    * only how long a sleep waits between re-checks. */
   const bool has_notifier = MALI_PER_ARCH(queue_has_notifier)(dev);
   const int64_t slice = has_notifier ? WAIT_SLICE_EVENTS_NS : WAIT_SLICE_NO_EVENTS_NS;
   VkResult result;
   bool slept = false;
   const int64_t start = os_time_get_nano();
   bool warned = false;
   uint32_t report = 0;

   pthread_mutex_lock(&dev->lock);
   for (;;) {
      if (dev->lost) {
         result = VK_ERROR_DEVICE_LOST;
         break;
      }

      /* A fault that has not reached the log yet (queue_wait's CSF body
       * scans the done slots directly; JM will drain events on demand). */
      const char *lost_msg = MALI_PER_ARCH(queue_wait)(dev);
      if (lost_msg) {
         pthread_mutex_unlock(&dev->lock);
         mali_device_set_lost(dev, "%s", lost_msg);
         return VK_ERROR_DEVICE_LOST;
      }

      /* Sync files are waited for with poll(), everything else on the
       * condition variable. */
      struct pollfd fds[16];
      uint32_t nfds = 0, signaled = 0;
      bool gpu_pending = false;
      for (uint32_t i = 0; i < count; i++) {
         struct mali_sync *s = to_sync(waits[i].sync);
         if (sync_is_signaled(dev, s, pending)) {
            signaled++;
         } else if (s->fd >= 0 && nfds < ARRAY_SIZE(fds)) {
            fds[nfds++] = (struct pollfd){.fd = s->fd, .events = POLLIN};
         } else {
            gpu_pending = true;
         }
      }
      if ((any && signaled) || signaled == count) {
         result = VK_SUCCESS;
         break;
      }

      int64_t now = os_time_get_nano();
      if ((uint64_t)now >= abs_timeout_ns) {
         result = VK_TIMEOUT;
         break;
      }

      if (!warned && now - start >= MALI_SLOW_WAIT_NS) {
         warned = true;
         note_slow_wait(dev, count, waits, any, abs_timeout_ns, now - start, &report);
      }

      int64_t until = MIN2((int64_t)MIN2(abs_timeout_ns, (uint64_t)INT64_MAX), now + slice);
      if (!slept) {
         dev->stats.waits++;
         slept = true;
      }
      if (nfds) {
         /* A sync file cannot wake the condition variable. With GPU work
          * to watch as well, poll in short slices. The fds stay valid:
          * only a reset, signal or import of these syncs closes them, and
          * the application may not do that during the wait. */
         int64_t len = (gpu_pending ? MIN2(until, now + WAIT_SLICE_NO_EVENTS_NS) : until) - now;
         struct timespec ts = {
            .tv_sec = len / 1000000000ll,
            .tv_nsec = len % 1000000000ll,
         };
         pthread_mutex_unlock(&dev->lock);
         ppoll(fds, nfds, &ts, NULL);
         pthread_mutex_lock(&dev->lock);
      } else {
         MALI_PER_ARCH(queue_sleep)(dev, until);
      }
      dev->stats.wakeups++;
   }
   pthread_mutex_unlock(&dev->lock);
   if (report) {
      mesa_logw("malisx2: slow wait %u ended after %lld ms: %s", report,
                (long long)((os_time_get_nano() - start) / 1000000),
                result == VK_SUCCESS ? "signalled" :
                result == VK_TIMEOUT ? "timed out" : "device lost");
   }
   return result;
}

/*
 * Import: the sync takes a dup of the fd (the runtime closes the caller's
 * after a successful import), after the blob's check that it is a sync
 * file (KBASE_IOCTL_FENCE_VALIDATE). A sync file that has signalled
 * already is kept as a host signal. fd -1 never gets here (the runtime
 * signals the sync instead).
 */
static VkResult
sync_import_sync_file(struct vk_device *vkdev, struct vk_sync *vs, int fd)
{
   struct mali_device *dev = dev_of(vkdev);
   struct mali_sync *s = to_sync(vs);

   int dup_fd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
   if (dup_fd < 0) {
      const int err = errno;
      MALI_PER_ARCH(sync_file_log_error)(dev, "import of", fd, strerror(err));
      return vk_errorf(vkdev, err == EMFILE  ? VK_ERROR_TOO_MANY_OBJECTS :
                              err == EBADF ? VK_ERROR_INVALID_EXTERNAL_HANDLE :
                                             VK_ERROR_OUT_OF_HOST_MEMORY,
                       "cannot duplicate sync file %d: %s", fd, strerror(err));
   }
   const enum mali_kbase_result vr = mali_kbase_fence_validate(dev->kbase, dup_fd);
   if (vr != MALI_KBASE_SUCCESS) {
      close(dup_fd);
      MALI_PER_ARCH(sync_file_log_error)(dev, "import of", fd, mali_kbase_result_str(vr));
      return vk_errorf(vkdev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "fd %d is not a sync file", fd);
   }

   pthread_mutex_lock(&dev->lock);
   MALI_PER_ARCH(sync_clear)(s);
   if (mali_sync_file_signaled(dup_fd)) {
      close(dup_fd);
      s->host_signaled = true;
   } else {
      s->fd = dup_fd;
   }
   pthread_cond_broadcast(&dev->cond);
   pthread_mutex_unlock(&dev->lock);
   return VK_SUCCESS;
}

/*
 * Export: a dup of an imported sync file, -1 for a payload that has
 * signalled (the Android and Vulkan conventions for "already signalled"),
 * or a new sync file from the kcpu queue for pending GPU work. The runtime
 * resets a binary semaphore afterwards (copy transference).
 */
static VkResult
sync_export_sync_file(struct vk_device *vkdev, struct vk_sync *vs, int *pfd)
{
   struct mali_device *dev = dev_of(vkdev);
   struct mali_sync *s = to_sync(vs);
   VkResult result = VK_SUCCESS;

   pthread_mutex_lock(&dev->lock);
   if (dev->lost) {
      result = VK_ERROR_DEVICE_LOST;
   } else if (sync_is_signaled(dev, s, false)) {
      *pfd = -1;
   } else if (s->fd >= 0) {
      *pfd = fcntl(s->fd, F_DUPFD_CLOEXEC, 0);
      if (*pfd < 0)
         result = vk_errorf(vkdev, errno == EMFILE ? VK_ERROR_TOO_MANY_OBJECTS :
                                                     VK_ERROR_OUT_OF_HOST_MEMORY,
                            "cannot duplicate a sync file: %s", strerror(errno));
   } else if (s->submitted) {
      result = MALI_PER_ARCH(sync_file_create)(dev, 0, NULL, s->req, pfd);
   } else {
      /* Nothing will signal it: invalid usage. The blob answers the same. */
      result = vk_errorf(vkdev, VK_ERROR_OUT_OF_HOST_MEMORY,
                         "exporting a sync file from a sync nothing will signal");
   }
   pthread_mutex_unlock(&dev->lock);
   if (result != VK_SUCCESS)
      MALI_PER_ARCH(sync_file_log_error)(dev, "export to", -1, vk_Result_to_str(result));
   return result;
}

const struct vk_sync_type MALI_PER_ARCH(sync_type) = {
   .size = sizeof(struct mali_sync),
   .features = VK_SYNC_FEATURE_BINARY | VK_SYNC_FEATURE_GPU_WAIT |
               VK_SYNC_FEATURE_CPU_WAIT |
               VK_SYNC_FEATURE_CPU_RESET | VK_SYNC_FEATURE_CPU_SIGNAL |
               VK_SYNC_FEATURE_WAIT_ANY | VK_SYNC_FEATURE_WAIT_PENDING,
   .init = sync_init,
   .finish = sync_finish,
   .signal = sync_signal,
   .reset = sync_reset,
   .move = sync_move,
   .wait_many = sync_wait_many,
   .import_sync_file = sync_import_sync_file,
   .export_sync_file = sync_export_sync_file,
};
