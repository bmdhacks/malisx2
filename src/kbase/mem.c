/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * GPU memory: allocate, import, CPU map, free, cache maintenance, query,
 * commit. The rules come from the blob's kernel wrappers and the r44p1
 * kernel:
 *
 * - Allocations use MEM_ALLOC_EX; flags are normalized first (the output
 *   bit NEED_MMAP is dropped, GPU_EX implies GPU_RD) and get the memory
 *   group of their class in bits 22..25 (group 0 for every class on the
 *   job manager).
 * - SAME_VA allocations come back as a cookie; mmap at offset = cookie
 *   gives the memory its address, CPU and GPU alike. Such a region is
 *   freed by munmap alone. Everything else is freed with MEM_FREE.
 * - Non-SAME_VA memory is CPU-mapped by mmap at offset = its GPU VA.
 * - Cache maintenance is needed only for CPU-cached memory that is not
 *   system-coherent, and goes through MEM_SYNC only when the kernel flagged
 *   the region KERNEL_SYNC; otherwise DC CVAC / DC CIVAC from user space.
 */

#include <string.h>
#include <sys/mman.h>

#include "kbase_priv.h"

unsigned
mali_kbase_mem_class_group(enum mali_kbase_mem_class c)
{
   /* Defaults of the blob's per-class config options, none overridden on
    * the RG 477V. */
   switch (c) {
   case MALI_KBASE_MEM_CLASS_NONE:
   case MALI_KBASE_MEM_CLASS_MMU:
      return 0;
   case MALI_KBASE_MEM_CLASS_INTERNAL_TLS:
      return 9;
   default:
      return 6;
   }
}

uint64_t
mali_kbase_mem_same_va_policy(uint64_t flags)
{
   const uint64_t no_same_va =
      KB_MEM_PROT_GPU_EX | KB_MEM_FIXED | KB_MEM_FIXABLE | KB_MEM_PROTECTED;
   if (flags & no_same_va)
      return flags;
   if (!(flags & (KB_MEM_PROT_GPU_RD | KB_MEM_PROT_GPU_WR)))
      return flags;
   return flags | KB_MEM_SAME_VA;
}

static int
cpu_prot(uint64_t flags)
{
   int prot = PROT_NONE;
   if (flags & KB_MEM_PROT_CPU_RD)
      prot |= PROT_READ;
   if (flags & KB_MEM_PROT_CPU_WR)
      prot |= PROT_WRITE;
   return prot;
}

static uint32_t
attrs_from_flags(uint64_t flags)
{
   uint32_t a = 0;
   if (flags & KB_MEM_CACHED_CPU)
      a |= MALI_KBASE_BO_CPU_CACHED;
   if (flags & (KB_MEM_COHERENT_SYSTEM | KB_MEM_COHERENT_SYSTEM_REQUIRED))
      a |= MALI_KBASE_BO_SYSTEM_COHERENT;
   if (flags & KB_MEM_KERNEL_SYNC)
      a |= MALI_KBASE_BO_KERNEL_SYNC;
   if (flags & KB_MEM_GROW_ON_GPF)
      a |= MALI_KBASE_BO_GROW_ON_GPF;
   return a;
}

static void
mem_free_ioctl(struct mali_kbase *kb, uint64_t gpu_va)
{
   struct kb_ioctl_mem_free mf = {.gpu_addr = gpu_va};
   long ret = kb_ioctl(kb, KB_IOCTL_MEM_FREE, &mf);
   if (ret < 0)
      KB_LOG(kb, "MEM_FREE 0x%llx failed: %s", (unsigned long long)gpu_va,
             strerror((int)-ret));
}

/*
 * After MEM_ALLOC_EX or MEM_IMPORT: map the region if the kernel handed
 * back a cookie (SAME_VA, or NEED_MMAP for imports), else optionally map
 * it at its GPU VA. On mmap failure the region is released again.
 */
static enum mali_kbase_result
finish_region(struct mali_kbase *kb, struct mali_kbase_bo *bo, bool cookie,
              bool want_map)
{
   if (cookie) {
      void *p = kb->backend->mmap(kb->backend->priv, bo->size, cpu_prot(bo->flags),
                                  kb->fd, bo->gpu_va);
      if (p == MAP_FAILED) {
         KB_LOG(kb, "mmap of %llu pages at cookie 0x%llx failed: %s",
                (unsigned long long)bo->va_pages, (unsigned long long)bo->gpu_va,
                strerror(errno));
         mem_free_ioctl(kb, bo->gpu_va);
         return MALI_KBASE_ERROR_MAP_FAILED;
      }
      bo->cpu = p;
      bo->gpu_va = (uint64_t)(uintptr_t)p;
      bo->attrs |= MALI_KBASE_BO_SAME_VA | MALI_KBASE_BO_CPU_MAPPED;
      return MALI_KBASE_SUCCESS;
   }

   if (want_map && cpu_prot(bo->flags) != PROT_NONE) {
      enum mali_kbase_result r = mali_kbase_map(kb, bo);
      if (r != MALI_KBASE_SUCCESS) {
         mem_free_ioctl(kb, bo->gpu_va);
         return r;
      }
   }
   return MALI_KBASE_SUCCESS;
}

/*
 * kbase_check_alloc_flags / kbase_check_alloc_sizes, job-manager build
 * (mali_kbase_mem.c in the kernel source; the same checks on the T820
 * kernel as on an r44p1 job-manager kernel). Returns NULL when flags
 * and the extension pass, else the reason (for the log line); every
 * one of these is caught here rather than left to the kernel's blanket
 * ENOMEM, so a flag mistake does not look like an out-of-memory
 * failure.
 */
static const char *
jm_check_alloc(uint64_t flags, uint64_t commit_pages, uint64_t extension)
{
   if (flags & KB_MEM_JM_RESERVED)
      return "bit 8 or 19 is reserved on a job-manager context";
   if (flags & KB_MEM_JM_KERNEL_ONLY)
      return "bit 5, 7, 27 or 29 is for the kernel's own allocations";
   if (!(flags & (KB_MEM_PROT_GPU_RD | KB_MEM_PROT_GPU_WR)))
      return "no GPU access";
   if (!(flags & (KB_MEM_PROT_CPU_RD | KB_MEM_PROT_GPU_RD)))
      return "neither CPU nor GPU reads it";
   if (!(flags & (KB_MEM_PROT_CPU_WR | KB_MEM_PROT_GPU_WR)))
      return "neither CPU nor GPU writes it";
   if ((flags & KB_MEM_GPU_VA_SAME_4GB_PAGE) &&
       (flags & (KB_MEM_PROT_GPU_EX | KB_MEM_TILER_ALIGN_TOP)))
      return "SAME_4GB_PAGE with GPU_EX or TILER_ALIGN_TOP";
   if ((flags & KB_MEM_PROT_GPU_EX) &&
       (flags & (KB_MEM_PROT_GPU_WR | KB_MEM_GROW_ON_GPF | KB_MEM_TILER_ALIGN_TOP)))
      return "GPU_EX with GPU_WR, GROW_ON_GPF or TILER_ALIGN_TOP";
   if ((flags & KB_MEM_GROW_ON_GPF) && !extension)
      return "GROW_ON_GPF with a zero extension";
   if (flags & KB_MEM_TILER_ALIGN_TOP) {
      if (!extension || (extension & (extension - 1)))
         return "TILER_ALIGN_TOP extension is not a nonzero power of two";
      if (extension > KB_MEM_TILER_ALIGN_TOP_EXT_MAX_PAGES)
         return "TILER_ALIGN_TOP extension over 2 MiB";
      if (commit_pages > extension)
         return "TILER_ALIGN_TOP commit larger than the extension";
   }
   if (!(flags & (KB_MEM_GROW_ON_GPF | KB_MEM_TILER_ALIGN_TOP)) && extension)
      return "a nonzero extension without GROW_ON_GPF or TILER_ALIGN_TOP";
   return NULL;
}

/*
 * MEM_ALLOC (nr 5, the plain 32-byte union): the only allocation ioctl a
 * job-manager kernel has (no MEM_ALLOC_EX). flags has already had the
 * generic normalization (NEED_MMAP stripped, GPU_EX implies GPU_RD, the
 * GPU_EX+SAME_VA exclusivity check, the memory class's group bits) done by
 * the caller.
 */
static enum mali_kbase_result
jm_alloc(struct mali_kbase *kb, const struct mali_kbase_alloc_info *info, uint64_t va_pages,
         uint64_t commit_pages, uint64_t flags, struct mali_kbase_bo *bo)
{
   if (info->fixed_address) {
      KB_LOG(kb, "alloc: fixed_address is a CSF feature, not available on a "
                 "job-manager context");
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;
   }
   /* A 64-bit job-manager context forces SAME_VA on everything except
    * executable memory (KCTX_FORCE_SAME_VA,
    * context/backend/mali_kbase_context_jm.c): the context's executable
    * zone is already set up by mali_kbase_create (exec_init), before any
    * allocation can reach here. */
   if (!(flags & KB_MEM_PROT_GPU_EX))
      flags |= KB_MEM_SAME_VA;

   /* Unlike the CSF side (which only ever has GROW_ON_GPF to worry about
    * and silently drops a stray extension), a nonzero extension without
    * GROW_ON_GPF or TILER_ALIGN_TOP is a rule we validate rather than
    * paper over: pass extension_pages through as given. */
   uint64_t extension = info->extension_pages;
   const char *why = jm_check_alloc(flags, commit_pages, extension);
   if (why) {
      KB_LOG(kb, "alloc: %s (flags 0x%llx)", why, (unsigned long long)flags);
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;
   }

   union kb_ioctl_mem_alloc a;
   memset(&a, 0, sizeof(a));
   a.in.va_pages = va_pages;
   a.in.commit_pages = commit_pages;
   a.in.extension = extension;
   a.in.flags = flags;
   long ret = kb_ioctl(kb, KB_IOCTL_MEM_ALLOC, &a);
   if (ret < 0) {
      KB_LOG(kb, "MEM_ALLOC of %llu pages, flags 0x%llx failed: %s",
             (unsigned long long)va_pages, (unsigned long long)flags, strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_DEVICE_MEMORY);
   }

   bo->gpu_va = a.out.gpu_va;
   bo->va_pages = va_pages;
   bo->size = va_pages << KB_PAGE_SHIFT;
   bo->flags = a.out.flags;
   bo->attrs = attrs_from_flags(a.out.flags);

   enum mali_kbase_result r =
      finish_region(kb, bo, (a.out.flags & KB_MEM_SAME_VA) != 0, info->cpu_map);
   if (r != MALI_KBASE_SUCCESS)
      memset(bo, 0, sizeof(*bo));
   return r;
}

enum mali_kbase_result
mali_kbase_alloc(struct mali_kbase *kb, const struct mali_kbase_alloc_info *info,
                 struct mali_kbase_bo *bo)
{
   if (!kb || !info || !bo)
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;
   memset(bo, 0, sizeof(*bo));

   uint64_t va_pages = (info->size + KB_PAGE_SIZE - 1) >> KB_PAGE_SHIFT;
   uint64_t commit_pages =
      info->commit_size ? (info->commit_size + KB_PAGE_SIZE - 1) >> KB_PAGE_SHIFT
                        : va_pages;
   if (va_pages == 0 || commit_pages > va_pages)
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;

   uint64_t flags = info->flags & ~KB_MEM_NEED_MMAP;
   if (flags & ~KB_MEM_FLAGS_INPUT_MASK) {
      KB_LOG(kb, "alloc: flags 0x%llx have bits above bit 29",
             (unsigned long long)info->flags);
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;
   }
   if (flags & KB_MEM_PROT_GPU_EX)
      flags |= KB_MEM_PROT_GPU_RD;
   /* SAME_VA wins over GPU_EX in the kernel's zone choice, which would
    * put code outside the executable zone. */
   if ((flags & KB_MEM_PROT_GPU_EX) && (flags & KB_MEM_SAME_VA)) {
      KB_LOG(kb, "alloc: GPU_EX with SAME_VA would miss the executable zone");
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;
   }
   if (((flags & KB_MEM_FIXED) != 0) != (info->fixed_address != 0))
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;
   if (info->mem_class != MALI_KBASE_MEM_CLASS_NONE) {
      flags &= ~KB_MEM_GROUP_ID_MASK;
      /* The job-manager devices (Unisoc) have no memory group manager:
       * every class goes to group 0 until the T820 blob's class table is
       * decoded. */
      if (kb->frontend != MALI_KBASE_FRONTEND_JM)
         flags |= KB_MEM_GROUP_ID(mali_kbase_mem_class_group(info->mem_class));
   }

   if (kb->frontend == MALI_KBASE_FRONTEND_JM)
      return jm_alloc(kb, info, va_pages, commit_pages, flags, bo);

   union kb_ioctl_mem_alloc_ex a;
   memset(&a, 0, sizeof(a));
   a.in.va_pages = va_pages;
   a.in.commit_pages = commit_pages;
   a.in.extension = (flags & KB_MEM_GROW_ON_GPF) ? info->extension_pages : 0;
   a.in.flags = flags;
   a.in.fixed_address = info->fixed_address;
   long ret = kb_ioctl(kb, KB_IOCTL_MEM_ALLOC_EX, &a);
   if (ret < 0) {
      KB_LOG(kb, "MEM_ALLOC_EX of %llu pages, flags 0x%llx failed: %s",
             (unsigned long long)va_pages, (unsigned long long)flags,
             strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_DEVICE_MEMORY);
   }

   bo->gpu_va = a.out.gpu_va;
   bo->va_pages = va_pages;
   bo->size = va_pages << KB_PAGE_SHIFT;
   bo->flags = a.out.flags;
   bo->attrs = attrs_from_flags(a.out.flags);

   enum mali_kbase_result r =
      finish_region(kb, bo, (a.out.flags & KB_MEM_SAME_VA) != 0, info->cpu_map);
   if (r != MALI_KBASE_SUCCESS)
      memset(bo, 0, sizeof(*bo));
   return r;
}

enum mali_kbase_result
mali_kbase_alloc_program(struct mali_kbase *kb, uint64_t size, struct mali_kbase_bo *bo)
{
   struct mali_kbase_alloc_info info = {
      .size = size,
      .flags = MALI_KBASE_FLAGS_PROGRAM,
      .mem_class = MALI_KBASE_MEM_CLASS_PROGRAM,
      .cpu_map = true,
   };
   return mali_kbase_alloc(kb, &info, bo);
}

enum mali_kbase_result
mali_kbase_map(struct mali_kbase *kb, struct mali_kbase_bo *bo)
{
   if (!kb || !bo || (bo->attrs & MALI_KBASE_BO_SAME_VA))
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;
   if (bo->attrs & MALI_KBASE_BO_CPU_MAPPED)
      return MALI_KBASE_SUCCESS;

   void *p = kb->backend->mmap(kb->backend->priv, bo->size, cpu_prot(bo->flags), kb->fd,
                               bo->gpu_va);
   if (p == MAP_FAILED) {
      KB_LOG(kb, "mmap of GPU VA 0x%llx (%llu pages) failed: %s",
             (unsigned long long)bo->gpu_va, (unsigned long long)bo->va_pages,
             strerror(errno));
      return MALI_KBASE_ERROR_MAP_FAILED;
   }
   bo->cpu = p;
   bo->attrs |= MALI_KBASE_BO_CPU_MAPPED;
   return MALI_KBASE_SUCCESS;
}

void
mali_kbase_unmap(struct mali_kbase *kb, struct mali_kbase_bo *bo)
{
   /* Unmapping a SAME_VA region would free it. */
   if (!kb || !bo || (bo->attrs & MALI_KBASE_BO_SAME_VA) ||
       !(bo->attrs & MALI_KBASE_BO_CPU_MAPPED))
      return;
   kb->backend->munmap(kb->backend->priv, bo->cpu, bo->size);
   bo->cpu = NULL;
   bo->attrs &= ~MALI_KBASE_BO_CPU_MAPPED;
}

static long
sticky(struct mali_kbase *kb, unsigned long request, uint64_t gpu_va)
{
   uint64_t addr = gpu_va;
   struct kb_ioctl_sticky_resource s = {
      .count = 1,
      .address = (uint64_t)(uintptr_t)&addr,
   };
   return kb_ioctl(kb, request, &s);
}

void
mali_kbase_free(struct mali_kbase *kb, struct mali_kbase_bo *bo)
{
   if (!kb || !bo || bo->size == 0)
      return;
   if (kb->abandoned) {
      /* mali_kbase_abandon: the GPU may still use it. */
      memset(bo, 0, sizeof(*bo));
      return;
   }

   if (bo->attrs & MALI_KBASE_BO_STICKY) {
      long ret = sticky(kb, KB_IOCTL_STICKY_RESOURCE_UNMAP, bo->gpu_va);
      if (ret < 0)
         KB_LOG(kb, "STICKY_RESOURCE_UNMAP 0x%llx failed: %s",
                (unsigned long long)bo->gpu_va, strerror((int)-ret));
   }

   if (bo->attrs & MALI_KBASE_BO_SAME_VA) {
      /* The kernel frees the region with its last CPU mapping, which sits at
       * the GPU address. */
      kb->backend->munmap(kb->backend->priv, (void *)(uintptr_t)bo->gpu_va, bo->size);
   } else {
      if (bo->attrs & MALI_KBASE_BO_CPU_MAPPED)
         kb->backend->munmap(kb->backend->priv, bo->cpu, bo->size);
      mem_free_ioctl(kb, bo->gpu_va);
   }
   memset(bo, 0, sizeof(*bo));
}

enum mali_kbase_result
mali_kbase_import_dmabuf(struct mali_kbase *kb, int fd, uint64_t flags,
                         uint32_t padding_pages, bool sticky_map,
                         struct mali_kbase_bo *bo)
{
   if (!kb || !bo || fd < 0)
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;
   memset(bo, 0, sizeof(*bo));

   flags &= ~KB_MEM_NEED_MMAP;
   if (flags & ~KB_MEM_FLAGS_INPUT_MASK)
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;

   /* phandle is a pointer to the fd, not the fd. */
   int handle = fd;
   union kb_ioctl_mem_import im;
   memset(&im, 0, sizeof(im));
   im.in.flags = flags;
   im.in.phandle = (uint64_t)(uintptr_t)&handle;
   im.in.type = KB_MEM_IMPORT_TYPE_UMM;
   im.in.padding = padding_pages;
   long ret = kb_ioctl(kb, KB_IOCTL_MEM_IMPORT, &im);

   /*
    * Some Huawei job-manager kernels (including older Kirin 990 builds)
    * use a 32-byte MEM_IMPORT ioctl request instead of the 24-byte layout.
    * The ioctl number encodes the struct size, so ENOTTY from the default
    * request can mean that the older layout is required.
    *
    * Only retry for a JM kernel reporting ENOTTY. Keep both the original
    * request and the normal behavior untouched on all other kernels.
    */
   if (ret == -ENOTTY && kb->frontend == MALI_KBASE_FRONTEND_JM) {
      union {
         struct {
            uint64_t flags;
            uint64_t phandle;
            uint32_t type;
            uint32_t padding;
            uint64_t padding2;
         } in;
         struct {
            uint64_t flags;
            uint64_t gpu_va;
            uint64_t va_pages;
            uint64_t padding2;
         } out;
      } legacy;
      _Static_assert(sizeof(legacy) == 32,
                     "legacy Huawei MEM_IMPORT ioctl must be 32 bytes");
      memset(&legacy, 0, sizeof(legacy));
      legacy.in.flags = flags;
      legacy.in.phandle = (uint64_t)(uintptr_t)&handle;
      legacy.in.type = KB_MEM_IMPORT_TYPE_UMM;
      legacy.in.padding = padding_pages;
      ret = kb_ioctl(kb, _IOWR(KB_IOCTL_TYPE, 22, __typeof__(legacy)), &legacy);
      if (ret >= 0) {
         im.out.flags = legacy.out.flags;
         im.out.gpu_va = legacy.out.gpu_va;
         im.out.va_pages = legacy.out.va_pages;
         KB_LOG(kb, "MEM_IMPORT: 32-byte legacy job-manager layout succeeded");
      } else {
         KB_LOG(kb, "MEM_IMPORT: 32-byte legacy job-manager retry failed: %s",
                strerror((int)-ret));
      }
   }
   if (ret < 0) {
      KB_LOG(kb, "MEM_IMPORT of dma-buf fd %d, flags 0x%llx failed: %s", fd,
             (unsigned long long)flags, strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_DEVICE_MEMORY);
   }

   bo->gpu_va = im.out.gpu_va;
   bo->va_pages = im.out.va_pages;
   bo->size = im.out.va_pages << KB_PAGE_SHIFT;
   bo->flags = im.out.flags;
   bo->attrs = attrs_from_flags(im.out.flags) | MALI_KBASE_BO_IMPORTED;

   bool cookie = (im.out.flags & (KB_MEM_SAME_VA | KB_MEM_NEED_MMAP)) != 0;
   enum mali_kbase_result r = finish_region(kb, bo, cookie, false);
   if (r != MALI_KBASE_SUCCESS) {
      memset(bo, 0, sizeof(*bo));
      return r;
   }
   /* The mapping only reserves the address: kbase answers CPU faults on a
    * SAME_VA mapping of a dma-buf with SIGBUS. */
   bo->cpu = NULL;
   bo->attrs &= ~MALI_KBASE_BO_CPU_MAPPED;

   if (sticky_map) {
      ret = sticky(kb, KB_IOCTL_STICKY_RESOURCE_MAP, bo->gpu_va);
      if (ret < 0) {
         KB_LOG(kb, "STICKY_RESOURCE_MAP 0x%llx failed: %s",
                (unsigned long long)bo->gpu_va, strerror((int)-ret));
         mali_kbase_free(kb, bo);
         return kb_result_from_errno((int)-ret, KB_ERRNO_DEVICE_MEMORY);
      }
      bo->attrs |= MALI_KBASE_BO_STICKY;
   }
   return MALI_KBASE_SUCCESS;
}

static enum mali_kbase_result
cache_op(struct mali_kbase *kb, const struct mali_kbase_bo *bo, uint64_t offset,
         uint64_t size, uint8_t op)
{
   if (!kb || !bo || offset > bo->size || size > bo->size - offset)
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;
   if (!(bo->attrs & MALI_KBASE_BO_CPU_CACHED) ||
       (bo->attrs & MALI_KBASE_BO_SYSTEM_COHERENT) || size == 0)
      return MALI_KBASE_SUCCESS;
   if (!bo->cpu)
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;

   uint8_t *addr = (uint8_t *)bo->cpu + offset;
   if (bo->attrs & MALI_KBASE_BO_KERNEL_SYNC) {
      /* The kernel keeps separate CPU and GPU pages for this region; only
       * it can move the data. */
      struct kb_ioctl_mem_sync ms = {
         .handle = bo->gpu_va & ~(uint64_t)(KB_PAGE_SIZE - 1),
         .user_addr = (uint64_t)(uintptr_t)addr,
         .size = size,
         .type = op,
      };
      kb->stats.mem_sync_calls++;
      long ret = kb_ioctl(kb, KB_IOCTL_MEM_SYNC, &ms);
      if (ret < 0) {
         KB_LOG(kb, "MEM_SYNC type %u failed: %s", op, strerror((int)-ret));
         return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
      }
      return MALI_KBASE_SUCCESS;
   }

   if (op == KB_SYNCSET_OP_MSYNC) {
      mali_kbase_cpu_clean(addr, size);
      kb->stats.dc_clean_bytes += size;
   } else {
      mali_kbase_cpu_clean_invalidate(addr, size);
      kb->stats.dc_clean_inv_bytes += size;
   }
   return MALI_KBASE_SUCCESS;
}

enum mali_kbase_result
mali_kbase_bo_flush(struct mali_kbase *kb, const struct mali_kbase_bo *bo,
                    uint64_t offset, uint64_t size)
{
   return cache_op(kb, bo, offset, size, KB_SYNCSET_OP_MSYNC);
}

enum mali_kbase_result
mali_kbase_bo_invalidate(struct mali_kbase *kb, const struct mali_kbase_bo *bo,
                         uint64_t offset, uint64_t size)
{
   return cache_op(kb, bo, offset, size, KB_SYNCSET_OP_CSYNC);
}

enum mali_kbase_result
mali_kbase_mem_query(struct mali_kbase *kb, uint64_t gpu_va, uint64_t query,
                     uint64_t *value)
{
   if (!kb || !value)
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;
   union kb_ioctl_mem_query q;
   memset(&q, 0, sizeof(q));
   q.in.gpu_addr = gpu_va;
   q.in.query = query;
   long ret = kb_ioctl(kb, KB_IOCTL_MEM_QUERY, &q);
   if (ret < 0) {
      KB_LOG(kb, "MEM_QUERY %llu of 0x%llx failed: %s", (unsigned long long)query,
             (unsigned long long)gpu_va, strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
   }
   *value = q.out.value;
   return MALI_KBASE_SUCCESS;
}

enum mali_kbase_result
mali_kbase_mem_commit(struct mali_kbase *kb, const struct mali_kbase_bo *bo,
                      uint64_t pages)
{
   if (!kb || !bo || pages > bo->va_pages)
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;
   struct kb_ioctl_mem_commit c = {.gpu_addr = bo->gpu_va, .pages = pages};
   long ret = kb_ioctl(kb, KB_IOCTL_MEM_COMMIT, &c);
   if (ret < 0) {
      KB_LOG(kb, "MEM_COMMIT of 0x%llx to %llu pages failed: %s",
             (unsigned long long)bo->gpu_va, (unsigned long long)pages,
             strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_DEVICE_MEMORY);
   }
   return MALI_KBASE_SUCCESS;
}

enum mali_kbase_result
mali_kbase_mem_flags_change(struct mali_kbase *kb, uint64_t gpu_va, uint64_t flags,
                            uint64_t mask)
{
   if (!kb)
      return MALI_KBASE_ERROR_INVALID_ARGUMENT;
   struct kb_ioctl_mem_flags_change fc = {
      .gpu_va = gpu_va & ~(uint64_t)(KB_PAGE_SIZE - 1),
      .flags = flags,
      .mask = mask,
   };
   long ret = kb_ioctl(kb, KB_IOCTL_MEM_FLAGS_CHANGE, &fc);
   if (ret < 0) {
      KB_LOG(kb, "MEM_FLAGS_CHANGE of 0x%llx failed: %s", (unsigned long long)gpu_va,
             strerror((int)-ret));
      return kb_result_from_errno((int)-ret, KB_ERRNO_GENERIC);
   }
   return MALI_KBASE_SUCCESS;
}
