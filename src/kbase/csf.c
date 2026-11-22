/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * CSF objects on the kbase context: queue groups, command-stream queues
 * (ring, registration, binding, publish/kick), the tiler heap and the event
 * channel. The kernel calls, their arguments and their order are the
 * blob's, without the parts the blob adds for its CINSTR tracing (no
 * REGISTER_EX, no trace buffer, no 8x ring) and without its kcpu queues.
 */

#include <string.h>
#include <sys/mman.h>

#include "kbase_priv.h"

/* The Mali user I/O and ring memory is Normal non-cacheable or device
 * memory: order the ring writes before CS_INSERT, and CS_INSERT before the
 * doorbell, for an outside observer. */
static inline void
io_barrier(void)
{
#if defined(__aarch64__)
   __asm__ volatile("dsb sy" ::: "memory");
#else
   __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

/* ---------------------------------------------------------------------- */
/* Groups                                                                  */

enum mali_kbase_result
mali_kbase_group_create(struct mali_kbase *kb,
                        const struct mali_kbase_group_create_info *info,
                        uint8_t *handle)
{
   /* Tiler endpoint 0 only, every fragment and compute endpoint, at most
    * 1 tiler and 64 fragment/compute endpoints, no CSI exception handlers
    * (no incremental rendering). A 1.18 kernel knows only the 40-byte
    * form of the ioctl; the 112-byte form adds fields we leave zero. */
   union kb_ioctl_cs_queue_group_create gc;
   memset(&gc, 0, sizeof(gc));
   gc.in.tiler_mask = 1;
   gc.in.fragment_mask = ~0ull;
   gc.in.compute_mask = ~0ull;
   gc.in.cs_min = info->cs_count;
   gc.in.priority = info->priority;
   gc.in.tiler_max = 1;
   gc.in.fragment_max = 64;
   gc.in.compute_max = 64;

   long ret;
   if (kb->uk_minor < KB_UK_GROUP_CREATE_112_MINOR) {
      /* The 1.18 form is the first 40 bytes of the current one. */
      union kb_ioctl_cs_queue_group_create_1_18 old;
      memcpy(&old, &gc, sizeof(old));
      ret = kb_ioctl(kb, KB_IOCTL_CS_QUEUE_GROUP_CREATE_1_18, &old);
      memcpy(&gc.out, &old.out, sizeof(old.out));
   } else {
      ret = kb_ioctl(kb, KB_IOCTL_CS_QUEUE_GROUP_CREATE, &gc);
   }
   if (ret < 0) {
      KB_LOG(kb, "CS_QUEUE_GROUP_CREATE (UK 1.%u form, %u CSs, priority %u) failed: %s",
             kb->uk_minor, info->cs_count, info->priority, strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
   }
   *handle = gc.out.group_handle;
   return MALI_KBASE_SUCCESS;
}

void
mali_kbase_group_terminate(struct mali_kbase *kb, uint8_t handle)
{
   struct kb_ioctl_cs_queue_group_term gt = {.group_handle = handle};
   long ret = kb_ioctl(kb, KB_IOCTL_CS_QUEUE_GROUP_TERMINATE, &gt);
   if (ret < 0)
      KB_LOG(kb, "CS_QUEUE_GROUP_TERMINATE %u failed: %s", handle,
             strerror((int)-ret));
}

/* ---------------------------------------------------------------------- */
/* Queues                                                                  */

enum mali_kbase_result
mali_kbase_queue_create(struct mali_kbase *kb, uint32_t ring_size,
                        uint8_t priority, struct mali_kbase_queue *q)
{
   memset(q, 0, sizeof(*q));
   if (ring_size < KB_PAGE_SIZE || (ring_size & (ring_size - 1)) ||
       priority > KB_QUEUE_MAX_PRIORITY)
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;

   /* The blob's ring flags 0x200f: CPU and GPU read/write, SAME_VA, fully
    * committed, CPU-uncached. */
   const struct mali_kbase_alloc_info ai = {
      .size = ring_size,
      .flags = mali_kbase_mem_same_va_policy(MALI_KBASE_FLAGS_PROT_ALL),
      .mem_class = MALI_KBASE_MEM_CLASS_INTERNAL,
   };
   enum mali_kbase_result r = mali_kbase_alloc(kb, &ai, &q->ring);
   if (r != MALI_KBASE_SUCCESS)
      return r;

   struct kb_ioctl_cs_queue_register reg = {
      .buffer_gpu_addr = q->ring.gpu_va,
      .buffer_size = ring_size,
      .priority = priority,
   };
   long ret = kb_ioctl(kb, KB_IOCTL_CS_QUEUE_REGISTER, &reg);
   if (ret < 0) {
      KB_LOG(kb, "CS_QUEUE_REGISTER 0x%llx (%u bytes) failed: %s",
             (unsigned long long)q->ring.gpu_va, ring_size, strerror((int)-ret));
      mali_kbase_free(kb, &q->ring);
      return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
   }
   q->registered = true;
   return MALI_KBASE_SUCCESS;
}

enum mali_kbase_result
mali_kbase_queue_bind(struct mali_kbase *kb, struct mali_kbase_queue *q,
                      uint8_t group, uint8_t csi)
{
   union kb_ioctl_cs_queue_bind bind;
   memset(&bind, 0, sizeof(bind));
   bind.in.buffer_gpu_addr = q->ring.gpu_va;
   bind.in.group_handle = group;
   bind.in.csi_index = csi;
   long ret = kb_ioctl(kb, KB_IOCTL_CS_QUEUE_BIND, &bind);
   if (ret < 0) {
      KB_LOG(kb, "CS_QUEUE_BIND 0x%llx to group %u slot %u failed: %s",
             (unsigned long long)q->ring.gpu_va, group, csi, strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
   }

   /* The kernel accepts exactly three pages for this handle. */
   void *io = kb->backend->mmap(kb->backend->priv, KB_CS_IO_PAGES * KB_PAGE_SIZE,
                                PROT_READ | PROT_WRITE, kb->fd,
                                bind.out.mmap_handle);
   if (io == MAP_FAILED) {
      KB_LOG(kb, "mmap of the user I/O pages (handle 0x%llx) failed: %s",
             (unsigned long long)bind.out.mmap_handle, strerror(errno));
      return MALI_KBASE_ERROR_MAP_FAILED;
   }
   q->io = io;
   q->doorbell = (volatile uint32_t *)io;
   q->input = (volatile uint64_t *)((uint8_t *)io + KB_PAGE_SIZE);
   q->output = (const volatile uint64_t *)((uint8_t *)io + 2 * KB_PAGE_SIZE);
   return MALI_KBASE_SUCCESS;
}

void
mali_kbase_queue_destroy(struct mali_kbase *kb, struct mali_kbase_queue *q)
{
   if (q->io)
      kb->backend->munmap(kb->backend->priv, q->io, KB_CS_IO_PAGES * KB_PAGE_SIZE);
   if (q->registered) {
      struct kb_ioctl_cs_queue_terminate t = {.buffer_gpu_addr = q->ring.gpu_va};
      long ret = kb_ioctl(kb, KB_IOCTL_CS_QUEUE_TERMINATE, &t);
      if (ret < 0)
         KB_LOG(kb, "CS_QUEUE_TERMINATE 0x%llx failed: %s",
                (unsigned long long)q->ring.gpu_va, strerror((int)-ret));
   }
   if (q->ring.gpu_va)
      mali_kbase_free(kb, &q->ring);
   memset(q, 0, sizeof(*q));
}

uint64_t
mali_kbase_queue_extract(const struct mali_kbase_queue *q)
{
   return q->output ? q->output[KB_CS_OUTPUT_EXTRACT / 8] : 0;
}

uint64_t
mali_kbase_queue_space(const struct mali_kbase_queue *q)
{
   uint64_t used = q->insert - mali_kbase_queue_extract(q);
   uint64_t size = q->ring.size;
   /* Keep 64 bytes for the NOP padding of the next publish. */
   return used + 64 >= size ? 0 : size - used - 64;
}

void
mali_kbase_queue_write(struct mali_kbase_queue *q, const uint64_t *words,
                       uint32_t count)
{
   uint64_t *ring = q->ring.cpu;
   uint64_t mask = q->ring.size / 8 - 1;
   for (uint32_t i = 0; i < count; i++)
      ring[(q->insert / 8 + i) & mask] = words[i];
   q->insert += (uint64_t)count * 8;
}

void
mali_kbase_queue_publish(struct mali_kbase *kb, struct mali_kbase_queue *q)
{
   if (q->insert == q->published)
      return;

   /* Pad to 64 bytes with NOPs, as the blob does before every publish. */
   static const uint64_t nop = 0;
   while (q->insert & 63)
      mali_kbase_queue_write(q, &nop, 1);

   io_barrier();
   q->input[KB_CS_INPUT_INSERT / 8] = q->insert;
   q->published = q->insert;
   q->publishes++;
   io_barrier();

   /* If the firmware is running the CS, the doorbell is enough; otherwise
    * ask the kernel to schedule it. */
   if (q->kicked && (q->output[KB_CS_OUTPUT_ACTIVE / 8] & 1)) {
      *q->doorbell = 1;
      io_barrier();
      if (q->output[KB_CS_OUTPUT_ACTIVE / 8] & 1) {
         q->doorbells++;
         return;
      }
   }

   struct kb_ioctl_cs_queue_kick k = {.buffer_gpu_addr = q->ring.gpu_va};
   long ret = kb_ioctl(kb, KB_IOCTL_CS_QUEUE_KICK, &k);
   q->kick_ioctls++;
   if (ret < 0)
      KB_LOG(kb, "CS_QUEUE_KICK 0x%llx failed: %s",
             (unsigned long long)q->ring.gpu_va, strerror((int)-ret));
   q->kicked = true;
}

/* ---------------------------------------------------------------------- */
/* Tiler heap                                                              */

enum mali_kbase_result
mali_kbase_tiler_heap_init(struct mali_kbase *kb,
                           const struct mali_kbase_tiler_heap_info *info,
                           uint64_t *heap_ctx_va, uint64_t *first_chunk_va)
{
   union kb_ioctl_cs_tiler_heap_init hi;
   memset(&hi, 0, sizeof(hi));
   hi.in.chunk_size = info->chunk_size;
   hi.in.initial_chunks = info->initial_chunks;
   hi.in.max_chunks = info->max_chunks;
   hi.in.target_in_flight = info->target_in_flight;
   hi.in.group_id = info->group_id;
   hi.in.buf_desc_va = info->buf_desc_va;
   long ret = kb_ioctl(kb, KB_IOCTL_CS_TILER_HEAP_INIT, &hi);
   if (ret < 0) {
      KB_LOG(kb, "CS_TILER_HEAP_INIT (%u x %u bytes, max %u) failed: %s",
             info->initial_chunks, info->chunk_size, info->max_chunks,
             strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_DEVICE_MEMORY);
   }
   *heap_ctx_va = hi.out.gpu_heap_va;
   *first_chunk_va = hi.out.first_chunk_va;
   return MALI_KBASE_SUCCESS;
}

void
mali_kbase_tiler_heap_term(struct mali_kbase *kb, uint64_t heap_ctx_va)
{
   struct kb_ioctl_cs_tiler_heap_term t = {.gpu_heap_va = heap_ctx_va};
   long ret = kb_ioctl(kb, KB_IOCTL_CS_TILER_HEAP_TERM, &t);
   if (ret < 0)
      KB_LOG(kb, "CS_TILER_HEAP_TERM 0x%llx failed: %s",
             (unsigned long long)heap_ctx_va, strerror((int)-ret));
}

/* ---------------------------------------------------------------------- */
/* Events                                                                  */

void
mali_kbase_event_signal(struct mali_kbase *kb)
{
   long ret = kb_ioctl(kb, KB_IOCTL_CS_EVENT_SIGNAL, NULL);
   if (ret < 0)
      KB_LOG(kb, "CS_EVENT_SIGNAL failed: %s", strerror((int)-ret));
}

bool
mali_kbase_has_events(const struct mali_kbase *kb)
{
   return kb->backend->poll && kb->backend->read;
}

int
mali_kbase_event_wait(struct mali_kbase *kb, int wake_fd, int timeout_ms)
{
   struct pollfd fds[2] = {
      {.fd = kb->fd, .events = POLLIN},
      {.fd = wake_fd, .events = POLLIN},
   };
   unsigned nfds = wake_fd >= 0 ? 2 : 1;
   int n = kb->backend->poll(kb->backend->priv, fds, nfds, timeout_ms);
   if (n < 0)
      return errno == EINTR ? 0 : -1;

   int mask = 0;
   if (fds[0].revents & POLLIN)
      mask |= 1;
   if (fds[0].revents & (POLLERR | POLLHUP | POLLNVAL))
      mask |= 4;
   if (nfds > 1 && (fds[1].revents & POLLIN))
      mask |= 2;
   return mask;
}

enum mali_kbase_result
mali_kbase_event_read(struct mali_kbase *kb, struct kb_csf_notification *n)
{
   long got = kb->backend->read(kb->backend->priv, kb->fd, n, sizeof(*n));
   if (got != (long)sizeof(*n)) {
      if (got < 0)
         KB_LOG(kb, "read of the event channel failed: %s", strerror(errno));
      else
         KB_LOG(kb, "short read of the event channel: %ld bytes", got);
      return MALI_KBASE_ERROR_KERNEL;
   }
   return MALI_KBASE_SUCCESS;
}

void
mali_kbase_cpu_queue_dump(struct mali_kbase *kb)
{
   struct kb_ioctl_cs_cpu_queue_info info = {0};
   long ret = kb_ioctl(kb, KB_IOCTL_CS_CPU_QUEUE_DUMP, &info);
   if (ret < 0)
      KB_LOG(kb, "CS_CPU_QUEUE_DUMP failed: %s", strerror((int)-ret));
}
