/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Sync files (Linux sync_file fds, what Android calls native fences) over
 * a kbase kcpu queue, the blob's mechanism:
 *
 *  - GPU wait on a sync file: kcpu FENCE_WAIT on the fd, then a kcpu
 *    CQS_SET_OPERATION of the queue's fence sync64 to the next value; the
 *    ring waits for that value with SYNC_WAIT64 (mali_queue.c). Nothing
 *    waits on the CPU.
 *  - A sync file for GPU work: kcpu CQS_WAIT_OPERATION on the done slots
 *    the work will reach, then FENCE_SIGNAL, which makes the kernel create
 *    a sync file on the kcpu queue's fence context at enqueue time; it
 *    signals when the queue gets past the waits.
 *
 * The kcpu queue executes in order, so a FENCE_WAIT that has not signalled
 * holds back every later command, the exports included. That is the
 * blob's behaviour; an export only ever depends on work that comes after
 * the earlier imports anyway.
 *
 * All of it runs under mali_device::lock.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "mali_queue.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <string.h>
#include <unistd.h>

#include "util/log.h"
#include "vk_alloc.h"
#include "vk_log.h"

#include "mali_vk.h"

struct mali_kcpu {
   uint8_t id;
   /* Value the fence sync64 is set to by the last FENCE_WAIT's CQS set. */
   uint64_t fence_seq;
   struct {
      uint64_t waits, exports, sets;
   } stats;
};

bool
mali_sync_file_signaled(int fd)
{
   struct pollfd p = {.fd = fd, .events = POLLIN};
   int r;
   do {
      r = poll(&p, 1, 0);
   } while (r < 0 && errno == EINTR);
   /* An error on the fd (POLLERR, POLLNVAL) counts as signalled: a sync
    * file with an error has signalled with that error. */
   return r > 0;
}

static VkResult
kcpu_get(struct mali_csf_device *csf, struct mali_kcpu **out)
{
   struct mali_device *dev = csf->dev;
   if (!csf->queue)
      return vk_errorf(dev, VK_ERROR_INITIALIZATION_FAILED, "sync files need the queue");
   if (!csf->kcpu) {
      struct mali_kcpu *k =
         vk_zalloc(&dev->vk.alloc, sizeof(*k), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
      if (!k)
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      enum mali_kbase_result r = mali_kbase_kcpu_queue_create(csf->kb, &k->id);
      if (r != MALI_KBASE_SUCCESS) {
         vk_free(&dev->vk.alloc, k);
         return vk_errorf(dev, VK_ERROR_INITIALIZATION_FAILED,
                          "cannot create the kcpu queue for sync files: %s",
                          mali_kbase_result_str(r));
      }
      /* The fence sync64 starts at 0 (the queue memory is zeroed). */
      csf->kcpu = k;
   }
   *out = csf->kcpu;
   return VK_SUCCESS;
}

static VkResult
enqueue(struct mali_csf_device *csf, struct mali_kcpu *k, const struct kb_kcpu_command *cmd)
{
   enum mali_kbase_result r = mali_kbase_kcpu_enqueue(csf->kb, k->id, cmd);
   if (r == MALI_KBASE_SUCCESS)
      return VK_SUCCESS;
   /* A refused FENCE_WAIT is a bad fd; anything else is the kernel. */
   if (cmd->type == KB_KCPU_COMMAND_FENCE_WAIT)
      return vk_errorf(csf->dev, VK_ERROR_INVALID_EXTERNAL_HANDLE,
                       "the kernel refused to wait for a sync file: %s",
                       mali_kbase_result_str(r));
   return vk_errorf(csf->dev, r == MALI_KBASE_ERROR_OUT_OF_HOST_MEMORY ?
                                 VK_ERROR_OUT_OF_HOST_MEMORY :
                                 VK_ERROR_INITIALIZATION_FAILED,
                    "kcpu command %u failed: %s", cmd->type, mali_kbase_result_str(r));
}

static VkResult
enqueue_fence_wait(struct mali_csf_device *csf, struct mali_kcpu *k, int fd)
{
   /* The kernel reads the fd during the call (sync_file_get_fence). */
   struct kb_fence fence = {.fd = fd, .stream_fd = -1};
   struct kb_kcpu_command cmd;
   memset(&cmd, 0, sizeof(cmd));
   cmd.type = KB_KCPU_COMMAND_FENCE_WAIT;
   cmd.info.fence.fence = (uint64_t)(uintptr_t)&fence;
   VkResult result = enqueue(csf, k, &cmd);
   if (result == VK_SUCCESS)
      k->stats.waits++;
   return result;
}

VkResult
mali_sync_file_gpu_wait(struct mali_csf_device *csf, int fd, uint64_t *value)
{
   struct mali_kcpu *k;
   VkResult result = kcpu_get(csf, &k);
   if (result != VK_SUCCESS)
      return result;

   result = enqueue_fence_wait(csf, k, fd);
   if (result != VK_SUCCESS)
      return result;

   /* SET (not ADD): the queue runs in order, so the value only grows. The
    * kernel writes the kcpu queue's error state into the error word and
    * rings the firmware so blocked SYNC_WAITs look again. */
   struct kb_cqs_operation set = {
      .addr = mali_queue_fence_va(csf->queue),
      .val = k->fence_seq + 1,
      .operation = KB_CQS_SET_OPERATION_SET,
      .data_type = KB_CQS_DATA_TYPE_U64,
   };
   struct kb_kcpu_command cmd;
   memset(&cmd, 0, sizeof(cmd));
   cmd.type = KB_KCPU_COMMAND_CQS_SET_OPERATION;
   cmd.info.cqs.objs = (uint64_t)(uintptr_t)&set;
   cmd.info.cqs.nr_objs = 1;
   result = enqueue(csf, k, &cmd);
   if (result != VK_SUCCESS) {
      /* The FENCE_WAIT is in the queue; later sets keep counting from its
       * value, so nothing waits for a value that never comes. */
      return result;
   }
   k->fence_seq++;
   k->stats.sets++;
   *value = k->fence_seq;
   return VK_SUCCESS;
}

VkResult
mali_sync_file_create(struct mali_csf_device *csf, uint32_t fd_count, const int *fds,
                      const uint64_t *req, int *out)
{
   struct mali_csf_queue *q = csf->queue;
   *out = -1;

   /* What is still pending. Sync files that have signalled and subqueues
    * that are there already need no command. */
   struct kb_cqs_operation waits[MALI_SUBQUEUE_COUNT];
   uint32_t nwaits = 0;
   if (req && q) {
      volatile struct mali_cs_sync64 *done = mali_queue_done(q);
      for (unsigned j = 0; j < MALI_SUBQUEUE_COUNT; j++) {
         if (!req[j] || done[j].seqno >= req[j])
            continue;
         waits[nwaits++] = (struct kb_cqs_operation){
            .addr = mali_queue_done_va(q, j),
            .val = req[j] - 1,
            .operation = KB_CQS_WAIT_OPERATION_GT,
            .data_type = KB_CQS_DATA_TYPE_U64,
         };
      }
   }
   uint32_t pending_fds = 0;
   for (uint32_t i = 0; i < fd_count; i++)
      pending_fds += fds[i] >= 0 && !mali_sync_file_signaled(fds[i]);
   if (!nwaits && !pending_fds)
      return VK_SUCCESS;

   struct mali_kcpu *k;
   VkResult result = kcpu_get(csf, &k);
   if (result != VK_SUCCESS)
      return result;

   for (uint32_t i = 0; i < fd_count; i++) {
      if (fds[i] < 0 || mali_sync_file_signaled(fds[i]))
         continue;
      result = enqueue_fence_wait(csf, k, fds[i]);
      if (result != VK_SUCCESS)
         return result;
   }

   struct kb_kcpu_command cmd;
   /* One CQS_WAIT_OPERATION per object. The r44p1 kernel prints a pending
    * command into a fixed 256-byte buffer when a kcpu command times out
    * (kbasep_csf_sync_kcpu_dump_locked); a command with three objects
    * overflows it and the kernel dies in a BRK, turning a GPU hang into a
    * device reboot. A queue executes in order, so several single-object
    * waits mean the same as one with all of them. */
   for (uint32_t w = 0; w < nwaits; w++) {
      /* A done slot's error word (a faulted stream) puts the kcpu queue in
       * error; the device is lost then anyway. */
      memset(&cmd, 0, sizeof(cmd));
      cmd.type = KB_KCPU_COMMAND_CQS_WAIT_OPERATION;
      cmd.info.cqs.objs = (uint64_t)(uintptr_t)&waits[w];
      cmd.info.cqs.nr_objs = 1;
      cmd.info.cqs.inherit_err_flags = 1;
      result = enqueue(csf, k, &cmd);
      if (result != VK_SUCCESS)
         return result;
   }

   struct kb_fence fence = {.fd = -1, .stream_fd = -1};
   memset(&cmd, 0, sizeof(cmd));
   cmd.type = KB_KCPU_COMMAND_FENCE_SIGNAL;
   cmd.info.fence.fence = (uint64_t)(uintptr_t)&fence;
   result = enqueue(csf, k, &cmd);
   if (result != VK_SUCCESS)
      return result;
   if (fence.fd < 0)
      return vk_errorf(csf->dev, VK_ERROR_INITIALIZATION_FAILED,
                       "the kernel created no sync file");
   k->stats.exports++;
   *out = fence.fd;
   return VK_SUCCESS;
}

void
mali_sync_file_finish(struct mali_csf_device *csf)
{
   pthread_mutex_lock(&csf->dev->lock);
   struct mali_kcpu *k = csf->kcpu;
   csf->kcpu = NULL;
   pthread_mutex_unlock(&csf->dev->lock);
   if (!k)
      return;
   /* The kernel drains what is still queued without waiting (sync-file
    * waits are cancelled, the sync files it created signal). */
   mali_kbase_kcpu_queue_destroy(csf->kb, k->id);
   if (k->stats.waits || k->stats.exports)
      mesa_logi("malisx2: kcpu queue: %llu sync-file waits, %llu sync files created",
                (unsigned long long)k->stats.waits, (unsigned long long)k->stats.exports);
   vk_free(&csf->dev->vk.alloc, k);
}
