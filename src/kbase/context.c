/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Context creation and teardown. The order follows the blob's context
 * init, minus the steps that are the blob's own bookkeeping or belong to
 * the command-stream layer:
 *
 *   blob step                          here
 *   2  open + VERSION_CHECK            open, version handshake
 *   3  SET_FLAGS                       SET_FLAGS
 *   4  tracking page (UK < 1.18)       not needed at UK 1.20
 *   5  GET_GPUPROPS                    GET_GPUPROPS + decode
 *   7  CS_GET_GLB_IFACE (twice)        same
 *   8  GPU ID vs supported list        left to the caller
 *   9  map the user register page      same
 *   10-12 page pool, event thread      command-stream layer
 *   (after the context) MEM_EXEC_INIT  MEM_EXEC_INIT
 *   (after the context) MEM_JIT_INIT   MEM_JIT_INIT (context flags bit 0 or
 *                                      2, which the Vulkan context has)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#include "kbase_priv.h"

#define DEFAULT_DEVICE_PATH "/dev/mali0"

static enum mali_kbase_result
open_device(struct mali_kbase *kb, const char *path)
{
   int fd = kb->backend->open(kb->backend->priv, path);
   if (fd >= 0) {
      kb->fd = fd;
      return MALI_KBASE_SUCCESS;
   }

   int err = errno;
   KB_LOG(kb, "cannot open %s: %s", path, strerror(err));
   switch (err) {
   case ENOENT:
   case ENODEV:
   case ENXIO:
      return MALI_KBASE_ERROR_DEVICE_NOT_FOUND;
   case EACCES:
   case EPERM:
      return MALI_KBASE_ERROR_PERMISSION_DENIED;
   default:
      return MALI_KBASE_ERROR_KERNEL;
   }
}

/*
 * We propose UK 1.20, the r44p1 CSF interface, and accept only 1.20 back.
 * The kernel answers min(its minor, ours) when the major matches and its
 * own version otherwise, so "1.20" means "a CSF kernel at 1.20 or newer"
 * (the RG 477V is 1.21). The blob accepts older minors and switches ioctl
 * variants for them; we implement only the 1.20 variants (MEM_ALLOC_EX,
 * the 112-byte group create, tiler heap init with buf_desc_va, no
 * tracking page), so an older kernel is refused here, loudly, instead of
 * failing later in some ioctl.
 *
 * If that check fails with EPERM, the kernel is a job-manager one: it
 * reserves this ioctl number and answers it with EPERM without touching
 * the file's setup state. Only then do we try the job-manager handshake,
 * on the same fd: VERSION_CHECK ioctl 0 proposing 11.36. A CSF kernel
 * that answers with another version, or a failure other than EPERM, is
 * refused at once, as before there was a job-manager path, so a CSF
 * device sees exactly the ioctls it always did. We accept only exactly 11.36 back, the same "kernel at
 * this minor or newer" pin as the CSF side (the kernel answers
 * min(ours, its own) for a matching major). Only one of kb->uk_major/minor
 * and kb->frontend is meaningful once this returns success; the other
 * frontend's fields in struct mali_kbase stay zero.
 */
static enum mali_kbase_result
version_handshake(struct mali_kbase *kb)
{
   struct kb_ioctl_version_check vc = {
      .major = KB_UK_VERSION_MAJOR,
      .minor = KB_UK_VERSION_MINOR,
   };
   long ret = kb_ioctl(kb, KB_IOCTL_VERSION_CHECK, &vc);
   if (ret >= 0 && vc.major == KB_UK_VERSION_MAJOR && vc.minor == KB_UK_VERSION_MINOR) {
      kb->frontend = MALI_KBASE_FRONTEND_CSF;
      kb->uk_major = vc.major;
      kb->uk_minor = vc.minor;
      return MALI_KBASE_SUCCESS;
   }
   if (ret < 0 && ret != -EPERM) {
      KB_LOG(kb, "refusing device: VERSION_CHECK (CSF ioctl 52) failed: %s; "
                 "not a CSF kbase kernel?",
             strerror((int)-ret));
      return MALI_KBASE_ERROR_INCOMPATIBLE_KERNEL;
   }
   if (ret >= 0) {
      KB_LOG(kb, "refusing device: kernel answered UK %u.%u, this driver "
                 "speaks only UK %u.%u (kbase r44p1)",
             vc.major, vc.minor, KB_UK_VERSION_MAJOR, KB_UK_VERSION_MINOR);
      return MALI_KBASE_ERROR_INCOMPATIBLE_KERNEL;
   }

   char csf_detail[80];
   snprintf(csf_detail, sizeof(csf_detail), "CSF VERSION_CHECK (ioctl 52) failed: %s",
            strerror((int)-ret));

   struct kb_ioctl_version_check jvc = {
      .major = KB_JM_UK_VERSION_MAJOR,
      .minor = KB_JM_UK_VERSION_MINOR,
   };
   long jret = kb_ioctl(kb, KB_JM_IOCTL_VERSION_CHECK, &jvc);
   if (jret >= 0 && jvc.major == KB_JM_UK_VERSION_MAJOR &&
       jvc.minor == KB_JM_UK_VERSION_MINOR) {
      kb->frontend = MALI_KBASE_FRONTEND_JM;
      kb->uk_major = jvc.major;
      kb->uk_minor = jvc.minor;
      return MALI_KBASE_SUCCESS;
   }

   if (jret < 0)
      KB_LOG(kb, "refusing device: %s (this driver speaks CSF UK %u.%u); "
                 "job-manager VERSION_CHECK (ioctl 0) failed too: %s (this driver "
                 "speaks job-manager UK %u.%u)",
             csf_detail, KB_UK_VERSION_MAJOR, KB_UK_VERSION_MINOR, strerror((int)-jret),
             KB_JM_UK_VERSION_MAJOR, KB_JM_UK_VERSION_MINOR);
   else
      KB_LOG(kb, "refusing device: %s (this driver speaks CSF UK %u.%u); "
                 "job-manager VERSION_CHECK (ioctl 0) answered UK %u.%u (this driver "
                 "speaks job-manager UK %u.%u)",
             csf_detail, KB_UK_VERSION_MAJOR, KB_UK_VERSION_MINOR, jvc.major, jvc.minor,
             KB_JM_UK_VERSION_MAJOR, KB_JM_UK_VERSION_MINOR);
   return MALI_KBASE_ERROR_INCOMPATIBLE_KERNEL;
}

static enum mali_kbase_result
set_flags(struct mali_kbase *kb, uint32_t mmu_group)
{
   /* The blob sends its context flags masked to what the kernel accepts:
    * only the MMU group (bits 3..6) is ever set, and it is 0 on the
    * device. CCTX_EMBEDDED and CSF_EVENT_THREAD are user-space only. */
   struct kb_ioctl_set_flags sf = {
      .create_flags = (mmu_group << KB_CONTEXT_MMU_GROUP_ID_SHIFT) &
                      KB_CONTEXT_MMU_GROUP_ID_MASK,
   };
   long ret = kb_ioctl(kb, KB_IOCTL_SET_FLAGS, &sf);
   if (ret < 0) {
      KB_LOG(kb, "SET_FLAGS 0x%x failed: %s", sf.create_flags, strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
   }
   kb->context_created = true;
   return MALI_KBASE_SUCCESS;
}

static enum mali_kbase_result
load_gpu_props(struct mali_kbase *kb)
{
   /* First call with no buffer returns the stream size. */
   struct kb_ioctl_get_gpuprops gp = {0};
   long size = kb_ioctl(kb, KB_IOCTL_GET_GPUPROPS, &gp);
   if (size <= 0) {
      KB_LOG(kb, "GET_GPUPROPS size query failed: %s",
             size < 0 ? strerror((int)-size) : "empty stream");
      return size < 0 ? kb_result_from_errno((int)-size, KB_ERRNO_GENERIC)
                      : MALI_KBASE_ERROR_MALFORMED_PROPERTIES;
   }

   void *buf = calloc(1, (size_t)size);
   if (!buf)
      return MALI_KBASE_ERROR_OUT_OF_HOST_MEMORY;

   gp.buffer = (uint64_t)(uintptr_t)buf;
   gp.size = (uint32_t)size;
   long got = kb_ioctl(kb, KB_IOCTL_GET_GPUPROPS, &gp);
   enum mali_kbase_result r;
   if (got < 0) {
      KB_LOG(kb, "GET_GPUPROPS failed: %s", strerror((int)-got));
      r = kb_result_from_errno((int)-got, KB_ERRNO_GENERIC);
   } else {
      if (got > size)
         got = size;
      r = mali_kbase_gpu_props_decode(buf, (size_t)got, &kb->props);
      if (r != MALI_KBASE_SUCCESS)
         KB_LOG(kb, "GET_GPUPROPS stream of %ld bytes is truncated", got);
   }
   free(buf);
   return r;
}

static enum mali_kbase_result
load_glb_iface(struct mali_kbase *kb)
{
   /* First call with zero counts returns the header and the counts. */
   union kb_ioctl_cs_get_glb_iface q;
   memset(&q, 0, sizeof(q));
   long ret = kb_ioctl(kb, KB_IOCTL_CS_GET_GLB_IFACE, &q);
   if (ret < 0) {
      KB_LOG(kb, "CS_GET_GLB_IFACE failed: %s", strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
   }

   struct mali_kbase_glb_iface *g = &kb->glb;
   uint32_t groups = q.out.group_num;
   uint32_t streams = q.out.total_stream_num;
   g->groups = calloc(groups ? groups : 1, sizeof(*g->groups));
   g->streams = calloc(streams ? streams : 1, sizeof(*g->streams));
   if (!g->groups || !g->streams)
      return MALI_KBASE_ERROR_OUT_OF_HOST_MEMORY;

   memset(&q, 0, sizeof(q));
   q.in.max_group_num = groups;
   q.in.max_total_stream_num = streams;
   q.in.groups_ptr = (uint64_t)(uintptr_t)g->groups;
   q.in.streams_ptr = (uint64_t)(uintptr_t)g->streams;
   ret = kb_ioctl(kb, KB_IOCTL_CS_GET_GLB_IFACE, &q);
   if (ret < 0) {
      KB_LOG(kb, "CS_GET_GLB_IFACE failed: %s", strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
   }

   g->version = q.out.glb_version;
   g->features = q.out.features;
   g->prfcnt_size = q.out.prfcnt_size;
   g->instr_features = q.out.instr_features;
   g->group_num = groups < q.out.group_num ? groups : q.out.group_num;
   g->total_stream_num =
      streams < q.out.total_stream_num ? streams : q.out.total_stream_num;
   /* STREAM_FEATURES bits 7:0 hold the work register count minus one: the
    * RG 477V reports 0x5f and the blob uses r0..r95 (upstream panthor
    * decodes the field the same way). */
   g->cs_work_registers =
      g->total_stream_num ? (g->streams[0].features & 0xff) + 1 : 0;
   return MALI_KBASE_SUCCESS;
}

static enum mali_kbase_result
map_user_reg_page(struct mali_kbase *kb)
{
   void *p = kb->backend->mmap(kb->backend->priv, KB_PAGE_SIZE, PROT_READ, kb->fd,
                               KB_MEM_CSF_USER_REG_PAGE_HANDLE);
   if (p == MAP_FAILED) {
      KB_LOG(kb, "mapping the user register page failed: %s", strerror(errno));
      return MALI_KBASE_ERROR_MAP_FAILED;
   }
   kb->user_reg_page = p;
   return MALI_KBASE_SUCCESS;
}

/*
 * BASE_MEM_MAP_TRACKING_HANDLE: on this job-manager kernel (UK 11.36, older
 * than 11.38) an allocation before this mapping exists fails with EINVAL
 * (kbase_mem_allow_alloc), and munmapping it again blocks allocation once
 * more. We never touch the mapping's contents (PROT_NONE); mali_kbase_destroy
 * unmaps it.
 */
static enum mali_kbase_result
jm_map_tracking_page(struct mali_kbase *kb)
{
   void *p = kb->backend->mmap(kb->backend->priv, KB_PAGE_SIZE, PROT_NONE, kb->fd,
                               KB_MEM_MAP_TRACKING_HANDLE);
   if (p == MAP_FAILED) {
      KB_LOG(kb, "mapping the job-manager tracking page failed: %s", strerror(errno));
      return MALI_KBASE_ERROR_MAP_FAILED;
   }
   kb->jm_tracking_page = p;
   return MALI_KBASE_SUCCESS;
}

static enum mali_kbase_result
exec_init(struct mali_kbase *kb)
{
   /* On r44p1 CSF kernels the executable zone always exists and this only
    * validates the size; the blob issues it anyway, with the full 4 GiB,
    * and so do we. */
   struct kb_ioctl_mem_exec_init ei = {.va_pages = KB_EXEC_VA_PAGES};
   long ret = kb_ioctl(kb, KB_IOCTL_MEM_EXEC_INIT, &ei);
   if (ret < 0) {
      KB_LOG(kb, "MEM_EXEC_INIT failed: %s", strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
   }
   return MALI_KBASE_SUCCESS;
}

static enum mali_kbase_result
jit_init(struct mali_kbase *kb)
{
   /* The blob issues this for its Vulkan contexts (create flags bits 0 and
    * 2). We do not use JIT memory, but on a 64-bit CSF context this is
    * what creates the CUSTOM_VA zone, and the kernel allocates tiler heap
    * contexts and chunks there: without it CS_TILER_HEAP_INIT fails with
    * ENOMEM (found on the RG 477V). It must come before the first
    * allocation. Group 0. (The G615's blob takes the group from its
    * config option 0x3b, the T820's job-manager blob from its option 0x36;
    * neither default is known.)
    *
    * On a job-manager context we issue it for the same reason the CSF
    * side does (it costs only a VA reservation, and keeps the context set
    * up like the one the kernel was tested with); our driver keeps its own
    * tiler-heap ring instead of using JIT memory, so JIT memory itself is
    * unused there too. The T820 blob's trim level is 0, not the CSF side's
    * 5. */
   struct kb_ioctl_mem_jit_init ji = {
      .va_pages = KB_JIT_VA_PAGES,
      .max_allocations = KB_JIT_MAX_ALLOCATIONS,
      .trim_level = kb->frontend == MALI_KBASE_FRONTEND_JM ? KB_JM_JIT_TRIM_LEVEL
                                                            : KB_JIT_TRIM_LEVEL,
      .group_id = 0,
      .phys_pages = KB_JIT_VA_PAGES,
   };
   long ret = kb_ioctl(kb, KB_IOCTL_MEM_JIT_INIT, &ji);
   if (ret == -ENOMEM && kb->frontend == MALI_KBASE_FRONTEND_JM) {
      /* The kernel cuts the JIT zone from the end of the SAME_VA zone (at
       * most 2^(GPU VA bits) bytes, less the 4 GiB executable zone) and
       * answers ENOMEM when KB_JIT_VA_PAGES does not fit
       * (kbase_region_tracker_init_jit_64), as on the RK3568's Mali-G52.
       * The job-manager path issues no JIT soft jobs and allocates only
       * from the SAME_VA and executable zones (mem.c), so it goes on
       * without the zone (mali_device.c logs it). A CSF context still
       * fails here: its tiler heap chunks come from that zone. */
      kb->jm_no_jit = true;
      return MALI_KBASE_SUCCESS;
   }
   if (ret < 0) {
      KB_LOG(kb, "MEM_JIT_INIT failed: %s", strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
   }
   return MALI_KBASE_SUCCESS;
}

static void
release(struct mali_kbase *kb)
{
   if (kb->frontend == MALI_KBASE_FRONTEND_JM && kb->context_created) {
      /* Best effort: a failure here does not change that we are about to
       * close the fd anyway. Without SET_FLAGS there is no context to
       * terminate (the kernel would answer EPERM). */
      enum mali_kbase_result r = mali_kbase_jm_post_term(kb);
      if (r != MALI_KBASE_SUCCESS)
         KB_LOG(kb, "POST_TERM during teardown failed (ignored): %s",
                mali_kbase_result_str(r));
   }
   /* An abandoned context keeps every mapping (mali_kbase_abandon). */
   if (kb->jm_tracking_page && !kb->abandoned)
      kb->backend->munmap(kb->backend->priv, kb->jm_tracking_page, KB_PAGE_SIZE);
   if (kb->user_reg_page)
      kb->backend->munmap(kb->backend->priv, (void *)kb->user_reg_page, KB_PAGE_SIZE);
   free(kb->glb.groups);
   free(kb->glb.streams);
   if (kb->fd >= 0)
      kb->backend->close(kb->backend->priv, kb->fd);
   free(kb);
}

enum mali_kbase_result
mali_kbase_create(const struct mali_kbase_create_info *info, struct mali_kbase **out)
{
   static const struct mali_kbase_create_info defaults = {0};
   if (!out)
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;
   *out = NULL;
   if (!info)
      info = &defaults;
   if (info->mmu_group > 15)
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;

   struct mali_kbase *kb = calloc(1, sizeof(*kb));
   if (!kb)
      return MALI_KBASE_ERROR_OUT_OF_HOST_MEMORY;
   kb->fd = -1;
   kb->backend = info->backend ? info->backend : mali_kbase_os_backend();
   kb->log = info->log;
   kb->log_user = info->log_user;

   long page = kb->backend->page_size(kb->backend->priv);
   if (page != KB_PAGE_SIZE) {
      KB_LOG(kb, "refusing device: page size is %ld, only 4096 is supported", page);
      release(kb);
      return MALI_KBASE_ERROR_INCOMPATIBLE_KERNEL;
   }

   const char *path = info->device_path ? info->device_path : DEFAULT_DEVICE_PATH;
   enum mali_kbase_result r = open_device(kb, path);
   if (r == MALI_KBASE_SUCCESS)
      r = version_handshake(kb);
   if (r == MALI_KBASE_SUCCESS)
      r = set_flags(kb, info->mmu_group);
   if (r == MALI_KBASE_SUCCESS && kb->frontend == MALI_KBASE_FRONTEND_JM)
      r = jm_map_tracking_page(kb);
   if (r == MALI_KBASE_SUCCESS)
      r = load_gpu_props(kb);
   if (r == MALI_KBASE_SUCCESS && kb->frontend == MALI_KBASE_FRONTEND_CSF)
      r = load_glb_iface(kb);
   if (r == MALI_KBASE_SUCCESS && kb->frontend == MALI_KBASE_FRONTEND_CSF)
      r = map_user_reg_page(kb);
   if (r == MALI_KBASE_SUCCESS)
      r = exec_init(kb);
   if (r == MALI_KBASE_SUCCESS)
      r = jit_init(kb);
   if (r == MALI_KBASE_SUCCESS && kb->frontend == MALI_KBASE_FRONTEND_JM)
      r = kb_jm_probe_atom_layout(kb);

   if (r != MALI_KBASE_SUCCESS) {
      release(kb);
      return r;
   }
   *out = kb;
   return MALI_KBASE_SUCCESS;
}

void
mali_kbase_destroy(struct mali_kbase *kb)
{
   if (kb)
      release(kb);
}

void
mali_kbase_abandon(struct mali_kbase *kb)
{
   if (kb)
      kb->abandoned = true;
}

uint32_t
mali_kbase_latest_flush_id(const struct mali_kbase *kb)
{
   return kb->user_reg_page[KB_USER_REG_LATEST_FLUSH / 4];
}

enum mali_kbase_result
mali_kbase_timeinfo(struct mali_kbase *kb, uint64_t *mono_ns, uint64_t *gpu_ts)
{
   union kb_ioctl_get_cpu_gpu_timeinfo t;
   memset(&t, 0, sizeof(t));
   t.in.request_flags = KB_TIMEINFO_MONOTONIC | KB_TIMEINFO_TIMESTAMP;
   long ret = kb_ioctl(kb, KB_IOCTL_GET_CPU_GPU_TIMEINFO, &t);
   if (ret < 0) {
      KB_LOG(kb, "GET_CPU_GPU_TIMEINFO failed: %s", strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
   }
   *mono_ns = t.out.sec * 1000000000ull + t.out.nsec;
   *gpu_ts = t.out.timestamp;
   return MALI_KBASE_SUCCESS;
}
