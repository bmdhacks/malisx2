/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The CSF queue: queue group, rings, tiler heap, submit, the event thread
 * and device loss.
 *
 * Kernel side, the blob's way: one kbase queue group per VkQueue, one ring
 * per subqueue registered with CS_QUEUE_REGISTER and bound with
 * CS_QUEUE_BIND, the tiler heap from CS_TILER_HEAP_INIT, publishing by
 * CS_INSERT + doorbell or CS_QUEUE_KICK, and a thread reading the kbase
 * event channel.
 *
 * Command-stream side, panvk's way (Mesa csf/panvk_vX_gpu_queue.c): three
 * subqueues, each with an initial stream that requests its resources, sets
 * up the scoreboard entries and loads the subqueue context. Where panthor
 * took a list of command-buffer streams per submit, we write into the ring
 * ourselves, per subqueue that has work:
 *
 *   1. waits on the other subqueues' done slots for the semaphores the
 *      submit waits on (SYNC_WAIT64);
 *   2. a full cache flush (the blob's submit-start FLUSH_CACHE2; panvk
 *      gets it from panthor);
 *   3. a CALL to each command buffer's recorded stream (always CALL,
 *      never copy);
 *   4. SYNC_SET64 of the submit number into the subqueue's done slot, with
 *      system scope so the kernel hears of it, and error propagation.
 *
 * Each command buffer ends with a wait on all its scoreboard entries and a
 * cache clean (mali_cmd_buffer.c), so by step 4 the subqueue's work is
 * complete and visible. A submit is complete when every subqueue's done
 * slot has reached the value it will be set to for that submit
 * (mali_sync.req).
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "mali_queue.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/eventfd.h>
#include <time.h>
#include <unistd.h>

#include "util/log.h"
#include "util/os_time.h"
#include "util/u_atomic.h"
#include "vk_alloc.h"
#include "vk_command_buffer.h"
#include "vk_log.h"

#include "mali_cmd_buffer.h"
#include "mali_measure.h"
#include "mali_vk.h"

/* ---------------------------------------------------------------------- */
/* Device loss                                                             */

/*
 * The runtime is told first (vk_device_set_lost, once), then waiters see
 * dev->lost: the runtime asserts that anything returning
 * VK_ERROR_DEVICE_LOST has already reported the loss.
 */
void
mali_device_set_lost(struct mali_device *dev, const char *fmt, ...)
{
   char msg[256];
   va_list ap;
   va_start(ap, fmt);
   vsnprintf(msg, sizeof(msg), fmt, ap);
   va_end(ap);

   if (p_atomic_cmpxchg(&dev->lost_reported, 0, 1) == 0) {
      /* Always in the log, whatever the build (the runtime's own message
       * is debug-only). Device loss is never hidden. */
      mesa_loge("malisx2: device lost: %s", msg);
      vk_device_set_lost(&dev->vk, "%s", msg);
      mali_measure_event(dev, "device lost: %s", msg);
   }

   pthread_mutex_lock(&dev->lock);
   if (!dev->lost_msg[0])
      snprintf(dev->lost_msg, sizeof(dev->lost_msg), "%s", msg);
   dev->lost = true;
   pthread_cond_broadcast(&dev->cond);
   pthread_mutex_unlock(&dev->lock);
}

/* With dev->lock held. A done slot with its error word set means a stream
 * faulted; that is device loss whether or not the kernel told us. Returns
 * the message for mali_device_set_lost (called after the lock is
 * dropped), or NULL. The CSF body of the queue_wait hook (mali_queue.h);
 * also used directly by mali_sync.c's host wait. */
const char *
MALI_PER_ARCH(queue_wait)(struct mali_device *dev)
{
   struct mali_csf_device *csf = dev->csf;
   struct mali_csf_queue *q = csf ? csf->queue : NULL;
   if (!q || dev->lost)
      return NULL;
   volatile struct mali_cs_sync64 *done = mali_queue_done(q);
   for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++) {
      if (done[i].error)
         return "a command stream reported an error in its done slot";
   }
   return NULL;
}

bool
MALI_PER_ARCH(queue_reached)(struct mali_device *dev, const uint64_t req[MALI_SUBQUEUE_COUNT])
{
   struct mali_csf_queue *q = dev->csf->queue;
   if (!q)
      return true;
   volatile struct mali_cs_sync64 *done = mali_queue_done(q);
   for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++) {
      if (done[i].seqno < req[i])
         return false;
   }
   return true;
}

VkResult
mali_device_check_status(struct vk_device *vkdev)
{
   struct mali_device *dev = container_of(vkdev, struct mali_device, vk);
   if (!dev->csf)
      return VK_SUCCESS;

   pthread_mutex_lock(&dev->lock);
   const char *msg = MALI_PER_ARCH(queue_wait)(dev);
   bool lost = dev->lost;
   pthread_mutex_unlock(&dev->lock);
   if (msg) {
      mali_device_set_lost(dev, "%s", msg);
      lost = true;
   }
   return lost ? VK_ERROR_DEVICE_LOST : VK_SUCCESS;
}

/* ---------------------------------------------------------------------- */
/* The event thread                                                        */

static const char *
group_error_name(uint8_t type)
{
   switch (type) {
   case KB_GPU_QUEUE_GROUP_ERROR_FATAL: return "fatal group error";
   case KB_GPU_QUEUE_GROUP_QUEUE_ERROR_FATAL: return "fatal queue error";
   case KB_GPU_QUEUE_GROUP_ERROR_TIMEOUT: return "progress timeout";
   case KB_GPU_QUEUE_GROUP_ERROR_TILER_HEAP_OOM: return "tiler heap out of memory";
   default: return "unknown error";
   }
}

/* Exception types: the low byte of CS_FATAL (a queue error) or of the
 * fault status of a group error (MMU or GPU fault status, or one of the
 * kernel's own software codes). Arm's names. */
static const char *
exception_name(uint8_t type)
{
   switch (type) {
   case 0x05: return "KABOOM";
   case 0x0f: return "CS_RESOURCE_TERMINATED";
   case 0x40: return "CS_CONFIG_FAULT";
   case 0x41: return "CS_UNRECOVERABLE";
   case 0x44: return "CS_ENDPOINT_FAULT";
   case 0x48: return "CS_BUS_FAULT";
   case 0x49: return "CS_INVALID_INSTRUCTION";
   case 0x4a: return "CS_CALL_STACK_OVERFLOW";
   case 0x4b: return "CS_INHERIT_FAULT";
   case 0x50: return "INSTR_INVALID_PC";
   case 0x51: return "INSTR_INVALID_ENC";
   case 0x55: return "INSTR_BARRIER_FAULT";
   case 0x58: return "DATA_INVALID_FAULT";
   case 0x59: return "TILE_RANGE_FAULT";
   case 0x5a: return "ADDR_RANGE_FAULT";
   case 0x5b: return "IMPRECISE_FAULT";
   case 0x68: return "FIRMWARE_INTERNAL_ERROR";
   case 0x69: return "RESOURCE_EVICTION_TIMEOUT";
   case 0x70: return "SW_FAULT_0";
   case 0x71: return "SW_FAULT_1";
   case 0x72: return "SW_FAULT_2";
   case 0x80: return "GPU_BUS_FAULT";
   case 0x88: return "GPU_SHAREABILITY_FAULT";
   case 0x89: return "SYSTEM_SHAREABILITY_FAULT";
   case 0x8a: return "GPU_CACHEABILITY_FAULT";
   case 0xc0: case 0xc1: case 0xc2: case 0xc3: case 0xc4:
      return "TRANSLATION_FAULT";
   case 0xc8: case 0xc9: case 0xca: case 0xcb:
      return "PERMISSION_FAULT";
   case 0xd9: case 0xda: case 0xdb:
      return "ACCESS_FLAG";
   case 0xe0: return "ADDRESS_SIZE_FAULT_IN";
   case 0xe4: case 0xe5: case 0xe6: case 0xe7:
      return "ADDRESS_SIZE_FAULT_OUT";
   case 0xe8: case 0xe9: case 0xea: case 0xeb:
      return "MEMORY_ATTRIBUTE_FAULT";
   default: return "UNKNOWN";
   }
}

/* A GPU page fault reaches us as a group error whose status is the MMU's
 * fault status and whose sideband is the faulting GPU virtual address.
 * Bits 9:8 of the status are the access type. */
static bool
is_mmu_fault(uint8_t type)
{
   return type >= 0xc0 && type <= 0xeb;
}

static VkDeviceFaultAddressTypeEXT
mmu_access_type(uint32_t status)
{
   switch ((status >> 8) & 3) {
   case 1: return VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT;
   case 2: return VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT;
   default: return VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT;   /* write, atomic */
   }
}

static const char *
mmu_access_name(uint32_t status)
{
   static const char *const names[] = {"atomic", "execute", "read", "write"};
   return names[(status >> 8) & 3];
}

/* The kernel's description of a group error as a device fault. */
static void
group_error_fault(const struct kb_gpu_queue_group_error *e, struct mali_device_fault *f)
{
   memset(f, 0, sizeof(*f));
   f->address_type = VK_DEVICE_FAULT_ADDRESS_TYPE_NONE_EXT;
   switch (e->error_type) {
   case KB_GPU_QUEUE_GROUP_QUEUE_ERROR_FATAL: {
      /* status is CS_FATAL (exception type, then exception data in bits
       * 31:8), sideband is CS_FATAL_INFO. */
      uint32_t st = e->payload.fatal_queue.status;
      f->code = st & 0xff;
      f->data = e->payload.fatal_queue.sideband;
      snprintf(f->what, sizeof(f->what), "%s: %s (0x%02x) on CS %u, exception data 0x%06x",
               group_error_name(e->error_type), exception_name(st & 0xff), st & 0xff,
               e->payload.fatal_queue.csi_index, st >> 8);
      break;
   }
   case KB_GPU_QUEUE_GROUP_ERROR_FATAL: {
      uint32_t st = e->payload.fatal_group.status;
      f->code = st & 0xff;
      f->data = e->payload.fatal_group.sideband;
      if (is_mmu_fault(st & 0xff)) {
         f->address_type = mmu_access_type(st);
         f->address = e->payload.fatal_group.sideband;
         snprintf(f->what, sizeof(f->what), "%s: %s (0x%02x), %s access, source 0x%04x",
                  group_error_name(e->error_type), exception_name(st & 0xff), st & 0xff,
                  mmu_access_name(st), st >> 16);
      } else {
         snprintf(f->what, sizeof(f->what), "%s: %s (0x%02x)", group_error_name(e->error_type),
                  exception_name(st & 0xff), st & 0xff);
      }
      break;
   }
   default:
      /* Progress timeout and tiler heap out of memory carry no status. */
      snprintf(f->what, sizeof(f->what), "%s", group_error_name(e->error_type));
      break;
   }
}

static void
handle_group_error(struct mali_csf_device *csf, const struct kb_csf_notification *n)
{
   const struct kb_gpu_queue_group_error *e = &n->payload.csg_error.error;
   uint8_t handle = n->payload.csg_error.handle;
   struct mali_device *dev = csf->dev;
   struct mali_device_fault f;
   group_error_fault(e, &f);

   pthread_mutex_lock(&dev->lock);
   dev->stats.group_errors++;
   bool ours = csf->group_valid && handle == csf->group_handle;
   mali_device_record_fault(dev, &f);
   pthread_mutex_unlock(&dev->lock);

   /* The kernel has terminated the group; every error type, the tiler
    * heap one included, is treated as device loss. */
   if (e->error_type == KB_GPU_QUEUE_GROUP_QUEUE_ERROR_FATAL) {
      mali_device_set_lost(dev,
                           "queue group %u: %s: CS %u status 0x%08x sideband 0x%016llx (%s)%s",
                           handle, group_error_name(e->error_type),
                           e->payload.fatal_queue.csi_index,
                           e->payload.fatal_queue.status,
                           (unsigned long long)e->payload.fatal_queue.sideband,
                           exception_name(e->payload.fatal_queue.status & 0xff),
                           ours ? "" : " (not our group)");
   } else if (e->error_type == KB_GPU_QUEUE_GROUP_ERROR_FATAL) {
      mali_device_set_lost(dev, "queue group %u: %s: status 0x%08x sideband 0x%016llx (%s)%s",
                           handle, group_error_name(e->error_type),
                           e->payload.fatal_group.status,
                           (unsigned long long)e->payload.fatal_group.sideband,
                           exception_name(e->payload.fatal_group.status & 0xff),
                           ours ? "" : " (not our group)");
   } else {
      mali_device_set_lost(dev, "queue group %u: %s%s", handle,
                           group_error_name(e->error_type), ours ? "" : " (not our group)");
   }
}

static void *
event_thread_main(void *arg)
{
   struct mali_csf_device *csf = arg;
   struct mali_device *dev = csf->dev;
   struct mali_kbase *kb = csf->kb;

   for (;;) {
      int m = mali_kbase_event_wait(kb, csf->wake_fd, 1000);
      if (m < 0) {
         mali_device_set_lost(dev, "poll on the kbase event channel failed: %s",
                              strerror(errno));
         break;
      }
      if (m & 2)
         break;
      if (m & 4) {
         mali_device_set_lost(dev, "the kbase event channel reported an error");
         break;
      }
      if (!(m & 1))
         continue;

      struct kb_csf_notification n;
      if (mali_kbase_event_read(kb, &n) != MALI_KBASE_SUCCESS) {
         mali_device_set_lost(dev, "reading the kbase event channel failed");
         break;
      }

      switch (n.type) {
      case KB_CSF_NOTIFICATION_EVENT:
         pthread_mutex_lock(&dev->lock);
         dev->stats.kernel_events++;
         pthread_cond_broadcast(&dev->cond);
         pthread_mutex_unlock(&dev->lock);
         break;
      case KB_CSF_NOTIFICATION_GPU_QUEUE_GROUP_ERROR:
         handle_group_error(csf, &n);
         break;
      case KB_CSF_NOTIFICATION_CPU_QUEUE_DUMP:
         mali_kbase_cpu_queue_dump(kb);
         break;
      default:
         mesa_logw("malisx2: unknown kbase notification type %u", n.type);
         break;
      }
   }
   return NULL;
}

/* ---------------------------------------------------------------------- */
/* Device                                                                  */

VkResult
mali_csf_device_init(struct mali_device *dev)
{
   struct mali_csf_device *csf =
      vk_zalloc(&dev->vk.alloc, sizeof(*csf), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!csf)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   csf->dev = dev;
   csf->kb = dev->kbase;
   csf->wake_fd = -1;
   csf->nr_registers = dev->kbase->glb.cs_work_registers;

   /* STREAM_FEATURES bits 15:8: scoreboard entries per CS. The layout
    * needs the three fixed entries plus at least one iterator. */
   unsigned sb_count = 0;
   if (dev->kbase->glb.streams && dev->kbase->glb.total_stream_num)
      sb_count = (dev->kbase->glb.streams[0].features >> 8) & 0xff;
   if (sb_count <= MALI_SB_ITER_START)
      sb_count = 8;
   sb_count = MIN2(sb_count, 16);
   csf->sb.count = sb_count;
   csf->sb.all_mask = (uint16_t)BITFIELD_MASK(sb_count);
   csf->sb.iter_count = sb_count - MALI_SB_ITER_START;
   csf->sb.all_iters_mask = (uint16_t)BITFIELD_RANGE(MALI_SB_ITER_START, csf->sb.iter_count);

   csf->shared_sb = dev->kbase->props.gpu_features & 0x8;

   dev->csf = csf;
   dev->fe = csf;
   dev->vk.check_status = mali_device_check_status;

   if (mali_kbase_has_events(dev->kbase)) {
      csf->wake_fd = eventfd(0, EFD_CLOEXEC);
      if (csf->wake_fd < 0 ||
          pthread_create(&csf->thread, NULL, event_thread_main, csf) != 0) {
         mali_csf_device_finish(dev);
         return vk_errorf(dev, VK_ERROR_INITIALIZATION_FAILED,
                          "cannot start the kbase event thread");
      }
      csf->thread_running = true;
   }
   return VK_SUCCESS;
}

void
mali_csf_device_finish(struct mali_device *dev)
{
   struct mali_csf_device *csf = dev->csf;
   if (!csf)
      return;

   if (csf->thread_running) {
      uint64_t one = 1;
      ssize_t w = write(csf->wake_fd, &one, sizeof(one));
      (void)w;
      pthread_join(csf->thread, NULL);
   }
   if (csf->wake_fd >= 0)
      close(csf->wake_fd);

   dev->vk.check_status = NULL;
   dev->csf = NULL;
   dev->fe = NULL;
   vk_free(&dev->vk.alloc, csf);
}

/* ---------------------------------------------------------------------- */
/* Ring words                                                              */

/*
 * A cs_builder over a stack buffer, for words the queue writes straight
 * into a ring. Registers 92..95 are free between command buffers (they are
 * cs_builder's own temporaries inside them); the builder normally refuses
 * them, so it is told the register file is three larger than it is.
 */
struct ring_words {
   struct cs_builder b;
   uint64_t buf[128];
};

static void
ring_words_init(struct ring_words *w, const struct mali_csf_device *csf)
{
   const struct cs_builder_conf conf = {
      .nr_registers = csf->nr_registers + 3,
      .nr_kernel_registers = 3,
      .ls_sb_slot = MALI_SB_LS,
   };
   cs_builder_init(&w->b, &conf,
                   (struct cs_buffer){.cpu = w->buf, .capacity = ARRAY_SIZE(w->buf)});
}

static uint32_t
ring_words_end(struct ring_words *w)
{
   cs_end(&w->b);
   assert(cs_is_valid(&w->b));
   uint32_t n = cs_root_chunk_size(&w->b) / 8;
   cs_builder_fini(&w->b);
   return n;
}

/* Wait for ring space; the firmware frees it as it executes. */
static VkResult
ring_reserve(struct mali_csf_device *csf, struct mali_kbase_queue *rq, uint32_t words)
{
   uint64_t need = (uint64_t)words * 8;
   if (mali_kbase_queue_space(rq) >= need)
      return VK_SUCCESS;

   /* What is in the ring must be running before space can appear. */
   mali_kbase_queue_publish(csf->kb, rq);
   int64_t deadline = os_time_get_nano() + 10ll * 1000 * 1000 * 1000;
   while (mali_kbase_queue_space(rq) < need) {
      if (p_atomic_read(&csf->dev->lost))
         return VK_ERROR_DEVICE_LOST;
      if (os_time_get_nano() > deadline) {
         mali_device_set_lost(csf->dev, "a command stream ring stayed full for 10 s: "
                                        "the GPU is not consuming it");
         return VK_ERROR_DEVICE_LOST;
      }
      usleep(100);
   }
   return VK_SUCCESS;
}

/* cap: the submit's capture, or NULL. */
static VkResult
ring_emit(struct mali_csf_device *csf, struct mali_kbase_queue *rq,
          struct ring_words *w, struct mali_measure_capture *cap, unsigned sq)
{
   uint32_t n = ring_words_end(w);
   if (unlikely(cap))
      mali_measure_capture_ring(cap, sq, w->buf, n);
   VkResult r = ring_reserve(csf, rq, n);
   if (r != VK_SUCCESS)
      return r;
   mali_kbase_queue_write(rq, w->buf, n);
   return VK_SUCCESS;
}

/* ---------------------------------------------------------------------- */
/* Queue                                                                   */

static VkResult
queue_setup_memory(struct mali_device *dev, struct mali_csf_queue *q)
{
   struct mali_kbase *kb = dev->kbase;
   enum mali_kbase_result kr;

   /* Sync objects: CSF event memory, which the kernel keeps mapped so it
    * can evaluate a blocked SYNC_WAIT. */
   const struct mali_kbase_alloc_info sync_info = {
      .size = 4096,
      .flags = mali_kbase_mem_same_va_policy(MALI_KBASE_FLAGS_INTERNAL_CSF_EVENT),
      .mem_class = MALI_KBASE_MEM_CLASS_INTERNAL,
   };
   kr = mali_kbase_alloc(kb, &sync_info, &q->sync_mem);
   if (kr != MALI_KBASE_SUCCESS)
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "cannot allocate the queue's sync memory: %s",
                       mali_kbase_result_str(kr));
   memset(q->sync_mem.cpu, 0, 4096);

   /* Barrier counters start at 1: waits are "greater than" a register
    * value that starts at 0 (panvk). */
   struct mali_cs_sync64 *syncobjs =
      (void *)((uint8_t *)q->sync_mem.cpu + MALI_QUEUE_SYNCOBJS_OFFSET);
   for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++)
      syncobjs[i].seqno = 1;

   /* Geometry buffer + Tiler Heap descriptor, fully backed and CPU
    * readable, as CS_TILER_HEAP_INIT wants of buf_desc_va. */
   const struct mali_kbase_alloc_info heap_info = {
      .size = MALI_TILER_GEOM_BUF_SIZE + 64,
      .flags = mali_kbase_mem_same_va_policy(MALI_KBASE_FLAGS_DEVICE_UNCACHED),
      .mem_class = MALI_KBASE_MEM_CLASS_INTERNAL,
   };
   kr = mali_kbase_alloc(kb, &heap_info, &q->heap_block);
   if (kr != MALI_KBASE_SUCCESS)
      return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                       "cannot allocate the tiler heap block: %s",
                       mali_kbase_result_str(kr));
   memset(q->heap_block.cpu, 0, q->heap_block.size);

   const uint64_t desc_va = q->heap_block.gpu_va + MALI_TILER_HEAP_DESC_OFFSET;
   const struct mali_kbase_tiler_heap_info hi = {
      .chunk_size = MALI_TILER_HEAP_CHUNK_SIZE,
      .initial_chunks = 1,
      .max_chunks = MALI_TILER_HEAP_MAX_CHUNKS,
      .target_in_flight = 0xffff,
      /* The blob takes the group from config option 0x3b, as for
       * MEM_JIT_INIT; 0 as there (kbase/context.c). */
      .group_id = 0,
      .buf_desc_va = desc_va,
   };
   uint64_t first_chunk;
   kr = mali_kbase_tiler_heap_init(kb, &hi, &q->heap_ctx, &first_chunk);
   if (kr != MALI_KBASE_SUCCESS)
      return vk_errorf(dev, VK_ERROR_INITIALIZATION_FAILED,
                       "cannot create the tiler heap: %s", mali_kbase_result_str(kr));
   q->heap_created = true;

   pan_cast_and_pack((uint8_t *)q->heap_block.cpu + MALI_TILER_HEAP_DESC_OFFSET,
                     TILER_HEAP, cfg) {
      cfg.size = MALI_TILER_HEAP_CHUNK_SIZE;
      cfg.base = first_chunk;
      cfg.bottom = first_chunk + 64;
      cfg.top = first_chunk + MALI_TILER_HEAP_CHUNK_SIZE;
   }

   for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++) {
      struct mali_cs_subqueue_ctx *ctx =
         (void *)((uint8_t *)q->sync_mem.cpu + MALI_QUEUE_CTX_OFFSET(i));
      *ctx = (struct mali_cs_subqueue_ctx){
         .syncobjs = q->sync_mem.gpu_va + MALI_QUEUE_SYNCOBJS_OFFSET,
         .done = q->sync_mem.gpu_va + MALI_QUEUE_DONE_OFFSET,
      };
      if (i != MALI_SUBQUEUE_COMPUTE) {
         ctx->tiler_heap = desc_va;
         /* The size (64 KiB in 4 KiB units) in the low bits, so the value
          * goes into a tiler context as it is (panvk, blob alike). */
         ctx->geom_buf = q->heap_block.gpu_va | (MALI_TILER_GEOM_BUF_SIZE >> 12);
      }
   }
   return VK_SUCCESS;
}

/* The first words of each subqueue (panvk's init_subqueue): its resources,
 * the subqueue context, the scoreboard entries, zeroed progress registers
 * and, for the two subqueues that tile and render, the tiler heap. */
static void
queue_initial_words(struct mali_csf_queue *q, unsigned sq, struct ring_words *w)
{
   struct mali_csf_device *csf = q->dev->csf;
   struct cs_builder *b = &w->b;

   cs_req_res(b, mali_subqueue_resources(sq));
   cs_move64_to(b, mali_cs_subqueue_ctx_reg(b),
                q->sync_mem.gpu_va + MALI_QUEUE_CTX_OFFSET(sq));

   const uint32_t iter0 = MALI_SB_ITER_START;
   cs_set_state_imm32(b, MALI_CS_SET_STATE_TYPE_SB_SEL_ENDPOINT, iter0);
   cs_set_state_imm32(b, MALI_CS_SET_STATE_TYPE_SB_MASK_WAIT, MALI_SB_MASK(iter0));
   cs_set_state_imm32(b, MALI_CS_SET_STATE_TYPE_SB_SEL_OTHER, MALI_SB_LS);
   cs_set_state_imm32(b, MALI_CS_SET_STATE_TYPE_SB_SEL_DEFERRED, MALI_SB_DEFERRED_SYNC);
   cs_set_state_imm32(b, MALI_CS_SET_STATE_TYPE_SB_MASK_STREAM,
                      csf->sb.all_iters_mask & ~MALI_SB_MASK(iter0));

   for (unsigned j = 0; j < MALI_SUBQUEUE_COUNT; j++)
      cs_move64_to(b, mali_cs_progress_seqno_reg(b, j), 0);

   if (sq != MALI_SUBQUEUE_COMPUTE) {
      struct cs_index heap = cs_reg64(b, MALI_CS_RING_REG0);
      cs_move64_to(b, heap, q->heap_ctx);
      cs_heap_set(b, heap);
   }
}

static void
queue_teardown(struct mali_device *dev, struct mali_csf_queue *q)
{
   struct mali_kbase *kb = dev->kbase;

   if (dev->csf) {
      pthread_mutex_lock(&dev->lock);
      dev->csf->group_valid = false;
      if (dev->csf->queue == q)
         dev->csf->queue = NULL;
      pthread_mutex_unlock(&dev->lock);
   }

   /* The blob's order: group, then the queues, then the heap. */
   if (q->group_created)
      mali_kbase_group_terminate(kb, q->group);
   for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++)
      mali_kbase_queue_destroy(kb, &q->sq[i]);
   if (q->heap_created)
      mali_kbase_tiler_heap_term(kb, q->heap_ctx);
   if (q->heap_block.gpu_va)
      mali_kbase_free(kb, &q->heap_block);
   if (q->sync_mem.gpu_va)
      mali_kbase_free(kb, &q->sync_mem);
}

VkResult
mali_csf_queue_init(struct mali_device *dev, struct mali_queue *queue)
{
   struct mali_csf_device *csf = dev->csf;
   struct mali_kbase *kb = dev->kbase;
   VkResult result;

   struct mali_csf_queue *q =
      vk_zalloc(&dev->vk.alloc, sizeof(*q), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!q)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   q->dev = dev;

   result = queue_setup_memory(dev, q);
   if (result != VK_SUCCESS)
      goto fail;

   const struct mali_kbase_group_create_info gi = {
      .cs_count = MALI_SUBQUEUE_COUNT,
      .priority = KB_QUEUE_GROUP_PRIORITY_MEDIUM,
   };
   enum mali_kbase_result kr = mali_kbase_group_create(kb, &gi, &q->group);
   if (kr != MALI_KBASE_SUCCESS) {
      result = vk_errorf(dev, VK_ERROR_INITIALIZATION_FAILED,
                         "cannot create the queue group: %s", mali_kbase_result_str(kr));
      goto fail;
   }
   q->group_created = true;

   for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++) {
      kr = mali_kbase_queue_create(kb, MALI_RING_SIZE, MALI_RING_PRIORITY, &q->sq[i]);
      if (kr == MALI_KBASE_SUCCESS)
         kr = mali_kbase_queue_bind(kb, &q->sq[i], q->group, (uint8_t)i);
      if (kr != MALI_KBASE_SUCCESS) {
         result = vk_errorf(dev, VK_ERROR_INITIALIZATION_FAILED,
                            "cannot set up command stream queue %u: %s", i,
                            mali_kbase_result_str(kr));
         goto fail;
      }
   }

   pthread_mutex_lock(&dev->lock);
   csf->queue = q;
   csf->group_handle = q->group;
   csf->group_valid = true;
   pthread_mutex_unlock(&dev->lock);

   /* Nothing waits for these: the rings run in order, so any submit comes
    * after them. */
   for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++) {
      struct ring_words w;
      ring_words_init(&w, csf);
      queue_initial_words(q, i, &w);
      result = ring_emit(csf, &q->sq[i], &w, NULL, i);
      if (result != VK_SUCCESS)
         goto fail;
      mali_kbase_queue_publish(kb, &q->sq[i]);
   }

   queue->csf = q;
   queue->fe = q;
   return VK_SUCCESS;

fail:
   queue_teardown(dev, q);
   vk_free(&dev->vk.alloc, q);
   return result;
}

void
mali_csf_queue_finish(struct mali_device *dev, struct mali_queue *queue)
{
   struct mali_csf_queue *q = queue->csf;
   if (!q)
      return;
   /* The kcpu queue points into the queue's sync memory. */
   mali_sync_file_finish(dev->csf);
   queue_teardown(dev, q);
   vk_free(&dev->vk.alloc, q);
   queue->csf = NULL;
   queue->fe = NULL;
}

/* ---------------------------------------------------------------------- */
/* Submit                                                                  */

static struct mali_sync *
to_mali_sync(struct vk_sync *s)
{
   assert(s->type == &MALI_PER_ARCH(sync_type) || MALI_PER_ARCH(sync_is_timeline_point)(s));
   return container_of(s, struct mali_sync, vk);
}

/*
 * Compute-subqueue ordering across command buffers.
 *
 * A command buffer's compute stream starts idle (mali_cmd_buffer.c,
 * init_streams): a barrier that names the compute subqueue as a source
 * does not make it signal while it has recorded nothing, and a stream that
 * relied on its earlier work that way (compute_prior_mask) waits for that
 * work here instead, before the command buffer's span. The compute
 * subqueue's barrier waits that were still put off at the end of a command
 * buffer are dropped there (compute_end_lazy, one bit per subqueue they
 * named); its next work waits here for everything those subqueues were
 * given up to then, and so does a stream relying on its earlier state (the
 * chain through it). Across
 * submits these are waits on the done slots, left out when the host
 * already sees them reached or the ring waited for them before; inside a
 * submit, on handoff counters that the rings add one to after a command
 * buffer. Before this (2026-09-29), each command buffer had the compute
 * stream signal and the fragment stream wait on it at the first transfer
 * barrier, and the fragment stream signal at its end for the compute
 * stream's put-off waits: 20-50 us of the fragment subqueue per command
 * buffer on ARMSX2, whose compute subqueue is idle in most frames.
 */
struct cmd_order {
   uint32_t n;
   bool intra;                 /* handoffs inside this submit */
   uint64_t base[MALI_SUBQUEUE_COUNT];
};

static bool
cmd_has_span(struct mali_cmd_buffer *cmd, unsigned sq)
{
   uint64_t a;
   uint32_t sz;
   return mali_cmd_buffer_span(cmd, sq, &a, &sz);
}

static struct mali_cmd_buffer *
submit_cmd(struct vk_queue_submit *submit, uint32_t c)
{
   return container_of(submit->command_buffers[c], struct mali_cmd_buffer, vk);
}

/* Ring i's handoff value after the command buffers before c (0: none). */
static uint64_t
handoff_before(struct mali_csf_queue *q, const struct cmd_order *o,
               struct vk_queue_submit *submit, unsigned i, uint32_t c)
{
   if (!o->intra)
      return 0;
   uint64_t n = 0;
   for (uint32_t k = 0; k < c; k++)
      n += cmd_has_span(submit_cmd(submit, k), i);
   return n ? o->base[i] + n : 0;
}

static void
ring_wait_ge(struct cs_builder *b, uint64_t va, uint64_t v)
{
   struct cs_index addr = cs_reg64(b, MALI_CS_RING_REG0);
   struct cs_index val = cs_reg64(b, MALI_CS_RING_REG0 + 2);
   cs_move64_to(b, addr, va);
   cs_move64_to(b, val, v - 1);
   cs_sync64_wait(b, false, MALI_CS_CONDITION_GREATER, val, addr);
}

/* The waits ring i makes before command buffer c. */
static void
ring_order_waits(struct mali_csf_queue *q, const struct cmd_order *o,
                 struct vk_queue_submit *submit, unsigned i, uint32_t c,
                 struct cs_builder *b)
{
   const unsigned C = MALI_SUBQUEUE_COMPUTE;
   struct mali_cmd_buffer *cmd = submit_cmd(submit, c);
   const bool prior = i != C && (cmd->compute_prior_mask & BITFIELD_BIT(i));
   const bool after_lazy = i == C && cmd->compute_work;
   if (!prior && !after_lazy)
      return;

   volatile struct mali_cs_sync64 *done = mali_queue_done(q);
   const uint64_t *need = prior ? q->compute_prior : q->compute_after_lazy;
   for (unsigned j = 0; j < MALI_SUBQUEUE_COUNT; j++) {
      const uint64_t v = need[j];
      if (j == i || !v || v <= q->ring_waited[i][j] || v <= done[j].seqno)
         continue;
      ring_wait_ge(b, mali_queue_done_va(q, j), v);
      q->ring_waited[i][j] = v;
   }

   if (!o->intra)
      return;
   uint32_t lazy_before = 0;
   bool compute_before = false;
   for (uint32_t k = 0; k < c; k++) {
      lazy_before |= submit_cmd(submit, k)->compute_end_lazy;
      compute_before |= submit_cmd(submit, k)->compute_work;
   }
   for (unsigned j = 0; j < MALI_SUBQUEUE_COUNT; j++) {
      if (j == i)
         continue;
      if (!(lazy_before & BITFIELD_BIT(j)) && !(prior && j == C && compute_before))
         continue;
      const uint64_t v = handoff_before(q, o, submit, j, c);
      if (v)
         ring_wait_ge(b, mali_queue_handoff_va(q, j), v);
   }
}

VkResult
mali_queue_submit(struct vk_queue *vkq, struct vk_queue_submit *submit)
{
   struct mali_queue *queue = container_of(vkq, struct mali_queue, vk);
   struct mali_device *dev = container_of(vkq->base.device, struct mali_device, vk);
   struct mali_csf_device *csf = dev->csf;
   struct mali_csf_queue *q = queue->csf;
   VkResult result = VK_SUCCESS;

   if (vk_queue_submit_has_bind(submit))
      return vk_errorf(vkq, VK_ERROR_FEATURE_NOT_PRESENT, "sparse binding is not supported");

   /* Which subqueues have command-buffer work. */
   uint32_t work = 0;
   for (uint32_t c = 0; c < submit->command_buffer_count; c++) {
      struct mali_cmd_buffer *cmd =
         container_of(submit->command_buffers[c], struct mali_cmd_buffer, vk);
      work |= mali_cmd_buffer_subqueue_mask(cmd);
   }

   /* What the waits need, per subqueue: the largest done value any waited
    * sync is pending on and has not reached yet. Syncs signalled from the
    * host, or never submitted, need nothing. */
   uint64_t need[MALI_SUBQUEUE_COUNT] = {0};
   /* Imported sync files (the Android acquire fence): the kcpu queue
    * waits for them and sets the queue's fence sync64, which the rings
    * wait for (mali_sync_file.c). */
   uint64_t fence_need = 0;
   pthread_mutex_lock(&dev->lock);
   const char *lost_msg = MALI_PER_ARCH(queue_wait)(dev);
   bool lost = dev->lost;
   if (!lost) {
      volatile struct mali_cs_sync64 *done = mali_queue_done(q);
      for (uint32_t i = 0; i < submit->wait_count && result == VK_SUCCESS; i++) {
         struct mali_sync *s = to_mali_sync(submit->waits[i].sync);
         if (s->fd >= 0) {
            uint64_t v;
            if (!mali_sync_file_signaled(s->fd)) {
               result = mali_sync_file_gpu_wait(csf, s->fd, &v);
               fence_need = MAX2(fence_need, v);
            }
            continue;
         }
         if (s->host_signaled || !s->submitted)
            continue;
         for (unsigned j = 0; j < MALI_SUBQUEUE_COUNT; j++) {
            if (s->req[j] > done[j].seqno)
               need[j] = MAX2(need[j], s->req[j]);
         }
      }
   }
   pthread_mutex_unlock(&dev->lock);
   if (lost_msg) {
      mali_device_set_lost(dev, "%s", lost_msg);
      lost = true;
   }
   if (lost)
      return VK_ERROR_DEVICE_LOST;
   if (result != VK_SUCCESS)
      return result;
   /* A submit without command buffers still has to hold its signals back
    * until the sync files have signalled: the compute ring does the
    * waiting. */
   if (fence_need && !work)
      work = BITFIELD_BIT(MALI_SUBQUEUE_COMPUTE);

   const uint64_t seqno = q->seqno + 1;

   /* Measurement: queue the command buffers' timestamps for
    * reading, and capture the submit's streams if asked to. */
   struct mali_measure_capture *cap = NULL;
   if (unlikely(dev->measure)) {
      mali_measure_submit(dev, queue, submit, seqno);
      cap = mali_measure_capture_begin(dev, queue, submit, seqno);
   }

   struct cmd_order order = {.n = submit->command_buffer_count};
   if (order.n > 1) {
      for (uint32_t c = 0; c < order.n; c++) {
         struct mali_cmd_buffer *cmd = submit_cmd(submit, c);
         order.intra |= cmd->compute_work || cmd->compute_end_lazy;
      }
   }
   memcpy(order.base, q->handoff, sizeof(order.base));

   for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++) {
      if (!(work & BITFIELD_BIT(i)))
         continue;

      struct mali_kbase_queue *rq = &q->sq[i];
      struct ring_words w;
      ring_words_init(&w, csf);
      struct cs_builder *b = &w.b;
      struct cs_index addr = cs_reg64(b, MALI_CS_RING_REG0);
      struct cs_index val = cs_reg64(b, MALI_CS_RING_REG0 + 2);
      struct cs_index r32 = cs_reg32(b, MALI_CS_RING_REG0);

      /* 1. Semaphore waits on the other subqueues. Our own subqueue is in
       *    order: its earlier submits have finished by the time the ring
       *    gets here. */
      for (unsigned j = 0; j < MALI_SUBQUEUE_COUNT; j++) {
         if (j == i || !need[j])
            continue;
         cs_move64_to(b, addr, mali_queue_done_va(q, j));
         cs_move64_to(b, val, need[j] - 1);
         cs_sync64_wait(b, false, MALI_CS_CONDITION_GREATER, val, addr);
      }
      if (fence_need) {
         cs_move64_to(b, addr, mali_queue_fence_va(q));
         cs_move64_to(b, val, fence_need - 1);
         cs_sync64_wait(b, false, MALI_CS_CONDITION_GREATER, val, addr);
      }

      /* 2. Everything the host wrote since the last submit becomes visible:
       *    clean and invalidate L2 and the load/store caches, invalidate
       *    the others. Flush ID 0: unconditional. */
      cs_move32_to(b, r32, 0);
      cs_flush_caches(b, MALI_CS_FLUSH_MODE_CLEAN_AND_INVALIDATE,
                      MALI_CS_FLUSH_MODE_CLEAN_AND_INVALIDATE,
                      MALI_CS_OTHER_FLUSH_MODE_INVALIDATE, r32,
                      cs_defer(0, MALI_SB_IMM_FLUSH));
      cs_wait_slot(b, MALI_SB_IMM_FLUSH);

      result = ring_emit(csf, rq, &w, cap, i);
      if (result != VK_SUCCESS)
         goto out;

      /* 3. The command buffers. */
      for (uint32_t c = 0; c < submit->command_buffer_count; c++) {
         struct mali_cmd_buffer *cmd =
            container_of(submit->command_buffers[c], struct mali_cmd_buffer, vk);
         uint64_t span;
         uint32_t size;
         if (!mali_cmd_buffer_span(cmd, i, &span, &size))
            continue;
         ring_words_init(&w, csf);
         ring_order_waits(q, &order, submit, i, c, &w.b);
         struct cs_index call_addr = cs_reg64(&w.b, MALI_CS_RING_REG0 + 2);
         struct cs_index call_len = cs_reg32(&w.b, MALI_CS_RING_REG0);
         cs_move64_to(&w.b, call_addr, span);
         cs_move32_to(&w.b, call_len, size);
         cs_call(&w.b, call_addr, call_len);
         if (order.intra && c + 1 < order.n) {
            /* The stream ends with all its work done (finish_stream). */
            struct cs_index addr = cs_reg64(&w.b, MALI_CS_RING_REG0);
            struct cs_index one = cs_reg64(&w.b, MALI_CS_RING_REG0 + 2);
            cs_move64_to(&w.b, addr, mali_queue_handoff_va(q, i));
            cs_move64_to(&w.b, one, 1);
            cs_sync64_add(&w.b, true, MALI_CS_SYNC_SCOPE_CSG, one, addr, cs_now());
         }
         result = ring_emit(csf, rq, &w, cap, i);
         if (result != VK_SUCCESS)
            goto out;
      }

      /* 4. Done: the submit number into our done slot. System scope, so
       *    the kernel gets an event and wakes host waiters. */
      ring_words_init(&w, csf);
      b = &w.b;
      addr = cs_reg64(b, MALI_CS_RING_REG0);
      val = cs_reg64(b, MALI_CS_RING_REG0 + 2);
      cs_move64_to(b, addr, mali_queue_done_va(q, i));
      cs_move64_to(b, val, seqno);
      cs_sync64_set(b, true, MALI_CS_SYNC_SCOPE_SYSTEM, val, addr, cs_now());
      result = ring_emit(csf, rq, &w, cap, i);
      if (result != VK_SUCCESS)
         goto out;
   }

   /* The capture is written before the GPU can touch the memory. */
   if (unlikely(cap))
      mali_measure_capture_end(dev, cap);

   /* Publish every ring that got words. */
   for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++)
      mali_kbase_queue_publish(dev->kbase, &q->sq[i]);

   q->seqno = seqno;
   for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++) {
      if (work & BITFIELD_BIT(i))
         q->last_signal[i] = seqno;
   }

   /* What later command buffers order against (see ring_order_waits). */
   for (uint32_t c = 0; c < order.n; c++) {
      struct mali_cmd_buffer *cmd = submit_cmd(submit, c);
      if (order.intra && c + 1 < order.n) {
         for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++)
            q->handoff[i] += cmd_has_span(cmd, i);
      }
      if (cmd->compute_work)
         q->compute_prior[MALI_SUBQUEUE_COMPUTE] = seqno;
      u_foreach_bit(j, cmd->compute_end_lazy) {
         if (j != MALI_SUBQUEUE_COMPUTE) {
            q->compute_prior[j] = MAX2(q->compute_prior[j], q->last_signal[j]);
            q->compute_after_lazy[j] = MAX2(q->compute_after_lazy[j], q->last_signal[j]);
         }
      }
   }

   /* Signals: pending until every subqueue reaches what it will have
    * written by the end of this submit. Binary semaphore waits consume
    * the payload (a timeline semaphore's point sync does not: the timeline
    * still needs it to know when that value completed). */
   pthread_mutex_lock(&dev->lock);
   for (uint32_t i = 0; i < submit->wait_count; i++) {
      struct mali_sync *s = to_mali_sync(submit->waits[i].sync);
      if (!(s->vk.flags & VK_SYNC_IS_TIMELINE) &&
          !MALI_PER_ARCH(sync_is_timeline_point)(&s->vk))
         MALI_PER_ARCH(sync_clear)(s);
   }
   for (uint32_t i = 0; i < submit->signal_count; i++) {
      struct mali_sync *s = to_mali_sync(submit->signals[i].sync);
      MALI_PER_ARCH(sync_clear)(s);
      s->submitted = true;
      memcpy(s->req, q->last_signal, sizeof(s->req));
   }
   dev->stats.submits++;
   pthread_mutex_unlock(&dev->lock);
   return VK_SUCCESS;

out:
   /* Words already in the rings stay there and run with the next publish;
    * nothing in them depends on this submit's signals. */
   if (unlikely(cap))
      mali_measure_capture_end(dev, cap);
   return result;
}
