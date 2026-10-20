/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The kbase layer: one kbase context (an open /dev/mali<N> file) and the
 * memory operations on it. Everything above this (Vulkan memory, queues)
 * talks to the kernel only through here.
 *
 * Scope: open, UK version handshake, context flags, GPU properties, the CSF
 * global interface, the user register page, the executable zone, and GPU
 * memory (allocate, free, CPU map, cache maintenance, dma-buf import,
 * query, commit), and the CSF objects the queues build on: queue groups,
 * command-stream queues and their rings, the tiler heap, and the event
 * channel (csf.c), and one kcpu queue for sync files (kcpu.c).
 *
 * mali_kbase_create also speaks the older job-manager (JM) frontend the
 * Mali-G57 uses instead of CSF: the handshake tries CSF first, so the
 * G615 is unaffected, and falls back to the JM version check on EPERM. On
 * a JM context, memory allocation takes the JM rules (mem.c), and
 * submission goes through job-manager atoms instead of command-stream
 * rings (jm.c: atom numbers, JOB_SUBMIT, event read, POST_TERM, sync-file
 * streams) — there is no JM equivalent of queue groups, command-stream
 * queues or the tiler heap in this file.
 *
 * All kernel access goes through a backend (struct mali_kbase_backend), so
 * the host tests can run this code against a fake kernel.
 *
 * Thread safety: the calls hold no locks of their own and keep no shared
 * mutable state except the statistics counters; concurrent calls on one
 * context are as safe as the kernel makes them. Creating and destroying a
 * context must not race with other calls on it.
 */

#ifndef MALI_KBASE_H
#define MALI_KBASE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <poll.h>

#include "uapi.h"
#include "uapi_jm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------- */
/* Results                                                                 */

/*
 * Our own result codes. The blob collapses kernel errors into 0 / 1
 * (generic) / 2 (host memory) / 3 (device memory) / 0x46 (busy); the first
 * five values here correspond to those, the rest are finer causes that the
 * Vulkan layer may still collapse the same way (generic failures become
 * VK_ERROR_INITIALIZATION_FAILED).
 */
enum mali_kbase_result {
   MALI_KBASE_SUCCESS = 0,
   MALI_KBASE_ERROR_KERNEL,               /* ioctl failed with another errno */
   MALI_KBASE_ERROR_OUT_OF_HOST_MEMORY,   /* malloc failed, or ENOMEM from a
                                             non-memory ioctl */
   MALI_KBASE_ERROR_OUT_OF_DEVICE_MEMORY, /* ENOMEM from an allocation,
                                             commit or import */
   MALI_KBASE_ERROR_BUSY,                 /* EBUSY (blob code 0x46) */
   MALI_KBASE_ERROR_DEVICE_NOT_FOUND,     /* no such node / not a char device */
   MALI_KBASE_ERROR_PERMISSION_DENIED,    /* EACCES / EPERM opening the node */
   MALI_KBASE_ERROR_INCOMPATIBLE_KERNEL,  /* UK version or page size refused */
   MALI_KBASE_ERROR_INVALID_ARGUMENT,     /* rejected before any kernel call */
   MALI_KBASE_ERROR_MAP_FAILED,           /* mmap of GPU memory failed */
   MALI_KBASE_ERROR_MALFORMED_PROPERTIES, /* GET_GPUPROPS stream is truncated */
};

const char *mali_kbase_result_str(enum mali_kbase_result r);

/* ---------------------------------------------------------------------- */
/* Backend: how the layer reaches the kernel                               */

/*
 * POSIX semantics: open returns an fd or -1, ioctl returns >= 0 or -1, mmap
 * returns MAP_FAILED on failure, munmap and close return 0 or -1; failures
 * set errno. open is expected to open read/write, non-blocking, close-on-exec
 * and to fail with ENODEV if the path is not a character device.
 */
struct mali_kbase_backend {
   void *priv;
   int (*open)(void *priv, const char *path);
   int (*close)(void *priv, int fd);
   int (*ioctl)(void *priv, int fd, unsigned long request, void *arg);
   void *(*mmap)(void *priv, size_t length, int prot, int fd, uint64_t offset);
   int (*munmap)(void *priv, void *addr, size_t length);
   /* The CPU page size of the process; kbase counts in kernel pages and we
    * only support kernels with 4 KiB pages. */
   long (*page_size)(void *priv);
   /* poll(2) over fds, of which the kbase fd may be one; read(2) on the
    * kbase fd (the event channel). Optional: without them there is no
    * event thread and waits poll memory on a timer. */
   int (*poll)(void *priv, struct pollfd *fds, unsigned nfds, int timeout_ms);
   long (*read)(void *priv, int fd, void *buf, size_t size);
};

/* The real one: open(2), ioctl(2), mmap(2) and so on. */
const struct mali_kbase_backend *mali_kbase_os_backend(void);

/* ---------------------------------------------------------------------- */
/* GPU properties (GET_GPUPROPS, decoded)                                   */

#define MALI_KBASE_MAX_COHERENT_GROUPS 16

/*
 * One field per kbase property key we keep (uapi.h, enum kb_gpuprop_key).
 * Keys the kernel sends that we do not keep (JS_FEATURES of slots past the
 * third, anything newer) are skipped, as the blob skips unknown keys.
 */
struct mali_kbase_gpu_props {
   /* core */
   uint32_t product_id;            /* GPU_ID bits 31:16, 0xb8a3 on the G615 */
   uint32_t version_status;
   uint32_t minor_revision;
   uint32_t major_revision;
   uint32_t gpu_freq_khz_max;
   uint32_t log2_program_counter_size;
   uint32_t texture_features[4];
   uint64_t gpu_available_memory_size;
   uint32_t num_exec_engines;
   /* L2 and tiler */
   uint32_t l2_log2_line_size;
   uint32_t l2_log2_cache_size;
   uint32_t l2_num_l2_slices;
   uint32_t tiler_bin_size_bytes;
   uint32_t tiler_max_active_levels;
   /* threads */
   uint32_t max_threads;
   uint32_t max_workgroup_size;
   uint32_t max_barrier_size;
   uint32_t max_registers;
   uint32_t max_task_queue;
   uint32_t max_thread_group_split;
   uint32_t impl_tech;
   uint32_t tls_alloc;
   /* raw registers */
   uint64_t shader_present;
   uint64_t tiler_present;
   uint64_t l2_present;
   uint64_t stack_present;
   uint32_t l2_features;
   uint32_t core_features;
   uint32_t mem_features;
   uint32_t mmu_features;
   uint32_t as_present;
   uint32_t js_present;
   /* job manager only: JS_FEATURES_0..2 (keys 35..37), one per job slot.
    * A G57 MC4 has three slots (fragment, vtc, compute-only); ARMSX2 never
    * needs a fourth. */
   uint32_t js_features[3];
   uint32_t tiler_features;
   uint32_t raw_texture_features[4];
   uint64_t gpu_id;
   uint32_t thread_max_threads;
   uint32_t thread_max_workgroup_size;
   uint32_t thread_max_barrier_size;
   uint32_t thread_features;
   uint32_t coherency_mode;        /* 0 ACE-Lite, 1 ACE, 31 none */
   uint32_t thread_tls_alloc;
   uint64_t gpu_features;
   /* coherent groups */
   uint32_t num_coherent_groups;
   uint32_t num_core_groups;
   uint32_t coherency;
   uint64_t coherent_group_core_mask[MALI_KBASE_MAX_COHERENT_GROUPS];

   /* Derived after decoding (not kernel keys). */
   uint32_t arch_major;            /* product_id bits 15:12 */
   uint32_t arch_minor;            /* bits 11:8 */
   uint32_t arch_rev;              /* bits 7:4 */
   uint32_t product_major;         /* bits 3:0 */
   uint32_t core_count;            /* popcount(shader_present) */
   uint32_t coherent_group_num_cores[MALI_KBASE_MAX_COHERENT_GROUPS];
   uint32_t va_bits;               /* mmu_features bits 7:0 */

   /* Which keys were present in the stream: bit k of word k / 64. */
   uint64_t keys_seen[2];
};

/*
 * Decode a GET_GPUPROPS stream. Zeroes *props first. Returns
 * MALI_KBASE_ERROR_MALFORMED_PROPERTIES when a record runs past the end
 * (fields decoded before that point are kept).
 */
enum mali_kbase_result
mali_kbase_gpu_props_decode(const void *stream, size_t size,
                            struct mali_kbase_gpu_props *props);

static inline bool
mali_kbase_gpu_props_has(const struct mali_kbase_gpu_props *p, unsigned key)
{
   return key < 128 && (p->keys_seen[key / 64] >> (key % 64)) & 1;
}

/* ---------------------------------------------------------------------- */
/* The CSF global interface (CS_GET_GLB_IFACE)                              */

struct mali_kbase_glb_iface {
   uint32_t version;          /* firmware global interface version */
   uint32_t features;
   uint32_t prfcnt_size;      /* 31:16 firmware, 15:0 hardware counter bytes */
   uint32_t instr_features;   /* 7:4 log2 max trace event size, 3:0 offset
                                 update rate */
   uint32_t group_num;
   uint32_t total_stream_num;
   struct kb_cs_group_control *groups;    /* group_num entries */
   struct kb_cs_stream_control *streams;  /* total_stream_num entries */
   /* Work registers per command stream: stream 0 features bits 7:0, plus
    * one (the field holds the count minus one). 96 on the RG 477V. */
   uint32_t cs_work_registers;
};

/* ---------------------------------------------------------------------- */
/* Context                                                                 */

/*
 * Which kbase frontend the context negotiated. mali_kbase_create works out
 * which one the kernel is: CSF-only fields (glb, user_reg_page) stay zero
 * on a job-manager context, and the job-manager tracking-page mapping stays
 * NULL on a CSF context.
 */
enum mali_kbase_frontend {
   MALI_KBASE_FRONTEND_CSF = 0,
   MALI_KBASE_FRONTEND_JM = 1,
};

/*
 * The atom layout JOB_SUBMIT takes on a job-manager context (uapi_jm.h,
 * struct kb_jm_atom_frame_nr, says how the two differ).
 */
enum mali_kbase_jm_atom_layout {
   MALI_KBASE_JM_ATOM_STOCK = 0,  /* base_jd_atom, stride 64 */
   MALI_KBASE_JM_ATOM_FRAME_NR,   /* base_jd_atom + u32 frame_nr, stride 72 */
};

typedef void (*mali_kbase_log_fn)(void *user, const char *msg);

struct mali_kbase_create_info {
   const char *device_path;                  /* NULL: "/dev/mali0" */
   uint32_t mmu_group;                       /* 0..15, the blob uses 0 */
   const struct mali_kbase_backend *backend; /* NULL: the OS backend */
   mali_kbase_log_fn log;                    /* NULL: stderr */
   void *log_user;
};

/* Counters for tests and measurement. Updated without atomics: treat as
 * approximate when calls run concurrently. */
struct mali_kbase_stats {
   uint64_t dc_clean_bytes;       /* DC CVAC from user space */
   uint64_t dc_clean_inv_bytes;   /* DC CIVAC from user space */
   uint64_t mem_sync_calls;       /* KBASE_IOCTL_MEM_SYNC */
};

/* Fields are read-only for users of the layer. */
struct mali_kbase {
   const struct mali_kbase_backend *backend;
   int fd;
   enum mali_kbase_frontend frontend;
   uint16_t uk_major, uk_minor;  /* negotiated: 1.20 on CSF, 11.36 on JM */
   struct mali_kbase_gpu_props props;
   struct mali_kbase_glb_iface glb;          /* CSF only; zero on JM */
   /* The GPU's user register page, mapped read-only. CSF only; NULL on JM. */
   const volatile uint32_t *user_reg_page;
   /* The job-manager tracking-page mapping (BASE_MEM_MAP_TRACKING_HANDLE),
    * kept only so mali_kbase_destroy can munmap it. Job manager only; NULL
    * on CSF. */
   void *jm_tracking_page;
   /* Job manager only: the atom layout mali_kbase_create's probe found,
    * and what the probe saw. jm_probe_errno is the errno of the stride-72
    * JOB_SUBMIT (0: accepted); jm_probe_event the event code its atom
    * completed with when it was accepted. */
   enum mali_kbase_jm_atom_layout jm_atom_layout;
   int jm_probe_errno;
   uint32_t jm_probe_event;
   /* Job manager only: MEM_JIT_INIT failed with ENOMEM (no room for the
    * zone in the GPU's VA) and the context runs without a JIT zone. */
   bool jm_no_jit;
   /* SET_FLAGS succeeded: the kernel context exists. */
   bool context_created;
   /* mali_kbase_abandon was called. */
   bool abandoned;
   mali_kbase_log_fn log;
   void *log_user;
   struct mali_kbase_stats stats;
};

/*
 * Open the device and set up a context. mali_kbase_create works out which
 * frontend the kernel speaks:
 *
 *   1. VERSION_CHECK (ioctl 52) proposing 1.20. On a matching answer the
 *      frontend is CSF and the rest of the sequence is the blob's CSF
 *      order: SET_FLAGS, GET_GPUPROPS, CS_GET_GLB_IFACE, map the user
 *      register page, MEM_EXEC_INIT, MEM_JIT_INIT (trim level 5).
 *   2. Otherwise, VERSION_CHECK (ioctl 0) proposing 11.36 on the same fd:
 *      each frontend's kernel answers the other's version-check ioctl
 *      with EPERM without touching setup state, so trying both on one fd
 *      is valid. On a matching answer the frontend is job-manager and the
 *      rest follows the T820 blob's order: SET_FLAGS, map the tracking
 *      page (BASE_MEM_MAP_TRACKING_HANDLE, required before any
 *      allocation on this kernel), GET_GPUPROPS, MEM_EXEC_INIT,
 *      MEM_JIT_INIT (trim level 0; we do not use JIT memory, see
 *      mali_kbase_alloc). Then one JOB_SUBMIT of a dependency-only atom
 *      at stride 72 picks the atom layout: EINVAL means a stock kernel
 *      (64-byte base_jd_atom); acceptance means the MediaTek frame_nr
 *      layout, and the atom's completion event is read back before
 *      returning, so no event or atom number is left over for the
 *      caller. An ENOMEM answer to MEM_JIT_INIT, from a GPU VA too narrow
 *      for the zone, leaves the context without one (jm_no_jit).
 *   3. If neither answer matches, INCOMPATIBLE_KERNEL, logging both
 *      refusals.
 *
 * Any other failure undoes everything done so far and logs why.
 */
enum mali_kbase_result
mali_kbase_create(const struct mali_kbase_create_info *info,
                  struct mali_kbase **out);

/* Free every allocation first: SAME_VA memory keeps the kernel context
 * alive until its CPU mapping is gone. */
void mali_kbase_destroy(struct mali_kbase *kb);

/*
 * The GPU may still be running work that uses this context's memory and
 * user space cannot stop it (job-manager atoms that never completed). From
 * now on mali_kbase_free leaves every region and its mapping in place, and
 * mali_kbase_destroy closes the fd without unmapping anything. The
 * mappings keep the kernel context alive until the process exits; the
 * kernel then stops the context's jobs before it frees their memory,
 * instead of the jobs faulting on memory freed under them. Leaks the
 * context's memory until then.
 */
void mali_kbase_abandon(struct mali_kbase *kb);

/* LATEST_FLUSH from the user register page (the flush ID FLUSH_CACHE2
 * compares against). */
uint32_t mali_kbase_latest_flush_id(const struct mali_kbase *kb);

/* GET_CPU_GPU_TIMEINFO: CLOCK_MONOTONIC in ns and the GPU system timestamp
 * (the counter STORE_STATE{Timestamp} writes), sampled together. The
 * timestamp runs at the CPU's architected timer frequency (CNTFRQ_EL0). */
enum mali_kbase_result mali_kbase_timeinfo(struct mali_kbase *kb, uint64_t *mono_ns,
                                           uint64_t *gpu_ts);

/* ---------------------------------------------------------------------- */
/* Memory classes and groups                                               */

/*
 * The blob tags every allocation with a memory class; the class selects the
 * kbase memory group (flag bits 22..25) through config options that are
 * not overridden on the RG 477V. Numbers are the blob's class numbers;
 * only the classes the Vulkan driver uses are listed.
 */
enum mali_kbase_mem_class {
   MALI_KBASE_MEM_CLASS_NONE = -1,              /* no group bits (imports) */
   MALI_KBASE_MEM_CLASS_PROGRAM = 5,
   MALI_KBASE_MEM_CLASS_DEVICE_CPU_UNCACHED = 39,
   MALI_KBASE_MEM_CLASS_DEVICE_CPU_CACHED = 40,
   MALI_KBASE_MEM_CLASS_DEVICE_TRANSIENT = 41,
   MALI_KBASE_MEM_CLASS_DEVICE_PROTECTED = 42,
   MALI_KBASE_MEM_CLASS_INTERNAL = 45,
   MALI_KBASE_MEM_CLASS_INTERNAL_TLS = 46,
   MALI_KBASE_MEM_CLASS_INTERNAL_TRANSIENT = 49,
   MALI_KBASE_MEM_CLASS_DESCRIPTOR_POOL = 50,
   MALI_KBASE_MEM_CLASS_COMMAND_ALLOCATOR = 51,
   MALI_KBASE_MEM_CLASS_MMU = 60,
};

/* The kbase memory group of a class: 9 for TLS, 0 for MMU, 6 otherwise. */
unsigned mali_kbase_mem_class_group(enum mali_kbase_mem_class c);

/*
 * The blob's kbase flags per kind of memory, without group bits and
 * without the SAME_VA the blob's pools add (see
 * mali_kbase_mem_same_va_policy).
 */
#define MALI_KBASE_FLAGS_PROT_ALL \
   (KB_MEM_PROT_CPU_RD | KB_MEM_PROT_CPU_WR | KB_MEM_PROT_GPU_RD | KB_MEM_PROT_GPU_WR)
/* Vulkan memory type 0: host visible, host coherent, CPU-uncached. */
#define MALI_KBASE_FLAGS_DEVICE_UNCACHED (MALI_KBASE_FLAGS_PROT_ALL | KB_MEM_COHERENT_LOCAL)
/* Vulkan memory type 1, kernel coherency mode not ACE (the RG 477V). */
#define MALI_KBASE_FLAGS_DEVICE_CACHED \
   (MALI_KBASE_FLAGS_DEVICE_UNCACHED | KB_MEM_CACHED_CPU)
/* Vulkan memory type 1, kernel coherency mode ACE. */
#define MALI_KBASE_FLAGS_DEVICE_CACHED_ACE \
   (MALI_KBASE_FLAGS_PROT_ALL | KB_MEM_CACHED_CPU | KB_MEM_COHERENT_SYSTEM | \
    KB_MEM_COHERENT_SYSTEM_REQUIRED)
/* Vulkan memory type 2: lazily allocated, GPU only. */
#define MALI_KBASE_FLAGS_DEVICE_TRANSIENT \
   (KB_MEM_PROT_GPU_RD | KB_MEM_PROT_GPU_WR | KB_MEM_COHERENT_LOCAL)
/* Driver-internal CPU-written memory (descriptors, command streams). */
#define MALI_KBASE_FLAGS_INTERNAL (MALI_KBASE_FLAGS_PROT_ALL | KB_MEM_CACHED_CPU)
/* CSF sync/event memory. */
#define MALI_KBASE_FLAGS_INTERNAL_CSF_EVENT \
   (MALI_KBASE_FLAGS_PROT_ALL | KB_MEM_COHERENT_SYSTEM | KB_MEM_CSF_EVENT)
/* Thread-local storage. */
#define MALI_KBASE_FLAGS_TLS MALI_KBASE_FLAGS_PROT_ALL
/* Shader code: lands in the 4 GiB executable zone; never SAME_VA, never
 * GPU-writable, CPU-uncached. */
#define MALI_KBASE_FLAGS_PROGRAM \
   (KB_MEM_PROT_CPU_RD | KB_MEM_PROT_CPU_WR | KB_MEM_PROT_GPU_RD | KB_MEM_PROT_GPU_EX)
/* dma-buf imports. */
#define MALI_KBASE_FLAGS_IMPORT (MALI_KBASE_FLAGS_PROT_ALL | KB_MEM_COHERENT_LOCAL)

/*
 * The blob's pool rule for SAME_VA: add it unless the flags have GPU_EX,
 * FIXED, FIXABLE or PROTECTED, or have neither GPU read nor GPU write.
 */
uint64_t mali_kbase_mem_same_va_policy(uint64_t flags);

/* ---------------------------------------------------------------------- */
/* Memory                                                                  */

/* struct mali_kbase_bo attribute bits. */
#define MALI_KBASE_BO_IMPORTED        (1u << 0)
/* The CPU mapping made at allocation/import owns the region: CPU address ==
 * GPU address, and munmap alone frees it (the kernel frees SAME_VA regions
 * when the last CPU mapping goes away; MEM_FREE on top would double-free). */
#define MALI_KBASE_BO_SAME_VA         (1u << 1)
#define MALI_KBASE_BO_CPU_MAPPED      (1u << 2)
#define MALI_KBASE_BO_CPU_CACHED      (1u << 3) /* returned CACHED_CPU */
#define MALI_KBASE_BO_SYSTEM_COHERENT (1u << 4) /* returned COHERENT_SYSTEM(_REQUIRED) */
#define MALI_KBASE_BO_KERNEL_SYNC     (1u << 5) /* returned KERNEL_SYNC */
#define MALI_KBASE_BO_GROW_ON_GPF     (1u << 6)
#define MALI_KBASE_BO_STICKY          (1u << 7) /* STICKY_RESOURCE_MAP done */

struct mali_kbase_bo {
   uint64_t gpu_va;
   void *cpu;            /* NULL when not CPU-accessible (always NULL for
                            imports, see mali_kbase_import_dmabuf) */
   uint64_t va_pages;
   uint64_t size;        /* va_pages * KB_PAGE_SIZE */
   uint64_t flags;       /* flags as the kernel returned them */
   uint32_t attrs;       /* MALI_KBASE_BO_* */
};

struct mali_kbase_alloc_info {
   uint64_t size;              /* bytes of GPU VA; rounded up to pages */
   uint64_t commit_size;       /* bytes backed now; 0 means all of size */
   uint64_t extension_pages;   /* growth step, only with GROW_ON_GPF */
   uint64_t flags;             /* KB_MEM_*, no group bits */
   enum mali_kbase_mem_class mem_class;
   uint64_t fixed_address;     /* only with KB_MEM_FIXED */
   /* Non-SAME_VA memory with CPU access: also mmap it now (at offset = GPU
    * VA). SAME_VA memory is always mapped: that mmap is what gives it its
    * address. */
   bool cpu_map;
};

/* MEM_ALLOC_EX (UK >= 1.9, which is all we accept), plus the mmap. */
enum mali_kbase_result
mali_kbase_alloc(struct mali_kbase *kb, const struct mali_kbase_alloc_info *info,
                 struct mali_kbase_bo *bo);

/* Shader code: MALI_KBASE_FLAGS_PROGRAM, class PROGRAM, CPU-mapped. The
 * kernel places it in the executable zone [2^47, 2^47 + 4 GiB). */
enum mali_kbase_result
mali_kbase_alloc_program(struct mali_kbase *kb, uint64_t size,
                         struct mali_kbase_bo *bo);

/* Undo alloc/import: sticky unmap, then munmap alone for SAME_VA regions,
 * munmap (if mapped) + MEM_FREE otherwise. Clears *bo. */
void mali_kbase_free(struct mali_kbase *kb, struct mali_kbase_bo *bo);

/* CPU-map / unmap a non-SAME_VA allocation after the fact. */
enum mali_kbase_result mali_kbase_map(struct mali_kbase *kb, struct mali_kbase_bo *bo);
void mali_kbase_unmap(struct mali_kbase *kb, struct mali_kbase_bo *bo);

/*
 * Import a dma-buf (MEM_IMPORT, type UMM). The kernel takes its own
 * reference; the caller still owns fd. padding_pages adds GPU VA after the
 * buffer. With sticky, the import is also mapped on the GPU now
 * (STICKY_RESOURCE_MAP), as the blob does for every import; without it the
 * pages are mapped only when a kcpu MAP_IMPORT asks. The r44p1 kernel
 * reports every import failure (bad fd included) as ENOMEM, so a failure
 * here is MALI_KBASE_ERROR_OUT_OF_DEVICE_MEMORY whatever the cause.
 *
 * The kernel asks for an mmap of the import (NEED_MMAP), which gives it its
 * GPU address (in the SAME_VA zone, equal to the mapping's CPU address), but
 * kbase refuses CPU page faults on that mapping (SIGBUS, "Invalid CPU
 * access to UMM memory"). So bo->cpu stays NULL: CPU access to an imported
 * buffer goes through an mmap of the dma-buf fd itself. The mapping is
 * still what keeps the region alive; mali_kbase_free unmaps it.
 */
enum mali_kbase_result
mali_kbase_import_dmabuf(struct mali_kbase *kb, int fd, uint64_t flags,
                         uint32_t padding_pages, bool sticky,
                         struct mali_kbase_bo *bo);

/*
 * CPU cache maintenance on [offset, offset + size) of a mapped bo. Nothing
 * to do unless the memory is CPU-cached and not system-coherent; then
 * MEM_SYNC if the kernel flagged the region KERNEL_SYNC, else DC CVAC
 * (flush) / DC CIVAC (invalidate) from user space, then DSB SY.
 * flush: CPU writes -> GPU. invalidate: GPU writes -> CPU.
 */
enum mali_kbase_result mali_kbase_bo_flush(struct mali_kbase *kb,
                                           const struct mali_kbase_bo *bo,
                                           uint64_t offset, uint64_t size);
enum mali_kbase_result mali_kbase_bo_invalidate(struct mali_kbase *kb,
                                                const struct mali_kbase_bo *bo,
                                                uint64_t offset, uint64_t size);

/* The raw user-space cache operations (aarch64; no-ops elsewhere). */
void mali_kbase_cpu_clean(const void *addr, size_t size);
void mali_kbase_cpu_clean_invalidate(const void *addr, size_t size);

/* MEM_QUERY (KB_MEM_QUERY_COMMIT_SIZE / VA_SIZE / FLAGS). */
enum mali_kbase_result mali_kbase_mem_query(struct mali_kbase *kb, uint64_t gpu_va,
                                            uint64_t query, uint64_t *value);

/* MEM_COMMIT: change the backed page count of a growable allocation. */
enum mali_kbase_result mali_kbase_mem_commit(struct mali_kbase *kb,
                                             const struct mali_kbase_bo *bo,
                                             uint64_t pages);

/* MEM_FLAGS_CHANGE (the kernel accepts only DONT_NEED and the coherency
 * bits). */
enum mali_kbase_result mali_kbase_mem_flags_change(struct mali_kbase *kb,
                                                   uint64_t gpu_va, uint64_t flags,
                                                   uint64_t mask);

/* ---------------------------------------------------------------------- */
/* Job manager: atom numbers, submission, events, sync-file streams (jm.c) */

/*
 * 255 atom numbers (1..255; 0 means "no dependency" and is never handed
 * out), kept in a bitmap. A number may be reused only once its completion
 * event has been read: a dependency on a number whose event has not been
 * read, but that was freed anyway, would resolve against whatever atom
 * the kernel gives that number next. This type has no lock of its own;
 * whatever lock serializes a queue's submits and event reads also covers
 * its atom-number allocator.
 */
struct mali_kbase_jm_atom_ids {
   uint8_t bitmap[32]; /* bit n set: atom number n is in use */
};

void mali_kbase_jm_atom_ids_init(struct mali_kbase_jm_atom_ids *ids);

/*
 * Take up to n free numbers into out[0..], lowest first. Returns the
 * number actually taken, which is less than n once fewer than n are free
 * (0 if none are). Every number returned is nonzero and marked in use.
 */
unsigned mali_kbase_jm_atom_ids_alloc(struct mali_kbase_jm_atom_ids *ids, uint8_t *out,
                                      unsigned n);
/* Marks id free again. id must be nonzero and in use. */
void mali_kbase_jm_atom_ids_free(struct mali_kbase_jm_atom_ids *ids, uint8_t id);
bool mali_kbase_jm_atom_ids_used(const struct mali_kbase_jm_atom_ids *ids, uint8_t id);
unsigned mali_kbase_jm_atom_ids_free_count(const struct mali_kbase_jm_atom_ids *ids);

/*
 * One line saying which atom layout the context uses and why, for the
 * driver's log. Returns buf.
 */
const char *mali_kbase_jm_atom_layout_str(const struct mali_kbase *kb, char *buf, size_t size);

/*
 * JOB_SUBMIT: submits exactly the atoms the caller built (dependencies,
 * core_req, atom numbers already filled in; this layer does not interpret
 * them). More than 256 atoms, the T820 kernel's limit per ioctl call (jm.c
 * says why), are split across as many JOB_SUBMIT calls as needed; nothing
 * else is done between them (no per-edge round trip — the whole
 * dependency graph is submitted with pre_dep already resolved by the
 * caller). n may be 0 (no-op). The atoms are always built as struct
 * kb_jm_atom; on a MALI_KBASE_JM_ATOM_FRAME_NR context each is copied into
 * the 72-byte layout (frame_nr 0) on the way in.
 */
enum mali_kbase_result
mali_kbase_jm_submit(struct mali_kbase *kb, const struct kb_jm_atom *atoms, unsigned n);

/*
 * Non-blocking read() of pending completion events (24-byte
 * base_jd_event_v2 records). *n is set to the number actually read (0 if
 * none are pending yet: EAGAIN). After mali_kbase_jm_post_term the kernel
 * answers EPIPE once every queued event has been drained; that is
 * reported as *n == 0 with *terminated set true (terminated may be NULL
 * if the caller never calls post_term on this context, or does not need
 * to know).
 */
enum mali_kbase_result
mali_kbase_jm_read_events(struct mali_kbase *kb, struct kb_jm_event *ev, unsigned max,
                          unsigned *n, bool *terminated);

/*
 * poll(2) on the kbase fd alone (POLLIN). Returns 1 when the fd is
 * readable, 0 on timeout, -1 on failure (errno set). Unlike the CSF side's
 * mali_kbase_event_wait, there is no wake_fd and no event thread at this
 * layer: job-manager completion has one reader at a time by the caller's
 * own choice and lock, not ours.
 */
int mali_kbase_jm_poll(struct mali_kbase *kb, int timeout_ms);

/*
 * POST_TERM: tells the kernel this context will submit no more atoms.
 * mali_kbase_destroy calls this itself (best effort) before closing the
 * fd. A caller that keeps reading events past that point sees
 * mali_kbase_jm_read_events report *terminated once they drain, with
 * no more to come.
 */
enum mali_kbase_result mali_kbase_jm_post_term(struct mali_kbase *kb);

/*
 * STREAM_CREATE: a new sync-file timeline fd, the stream_fd a
 * FENCE_TRIGGER soft atom's base_fence names. name is copied in, truncated
 * to 31 bytes plus a NUL as the kernel requires;
 * NULL gives it an empty name.
 */
enum mali_kbase_result
mali_kbase_stream_create(struct mali_kbase *kb, const char *name, int *out_fd);

/* ---------------------------------------------------------------------- */
/* CSF: queue groups, command-stream queues, tiler heap, events (csf.c)    */

/*
 * One queue group, the kernel's unit of GPU scheduling. The arguments are
 * the blob's, except the CS count, which is the caller's.
 */
struct mali_kbase_group_create_info {
   uint8_t cs_count;       /* CSs the group needs (cs_min) */
   uint8_t priority;       /* KB_QUEUE_GROUP_PRIORITY_* */
};

enum mali_kbase_result
mali_kbase_group_create(struct mali_kbase *kb,
                        const struct mali_kbase_group_create_info *info,
                        uint8_t *handle);

/* CS_QUEUE_GROUP_TERMINATE. The kernel unbinds the group's queues. */
void mali_kbase_group_terminate(struct mali_kbase *kb, uint8_t handle);

/*
 * A command-stream queue: its ring (the kernel's queue buffer, which the
 * firmware executes) and the three user I/O pages of its binding. The
 * producer offsets are byte counts that only grow; the
 * ring is written at offset & (size - 1). Not thread-safe: one producer.
 */
struct mali_kbase_queue {
   struct mali_kbase_bo ring;
   uint64_t insert;                  /* bytes written */
   uint64_t published;               /* last CS_INSERT given to the GPU */
   void *io;                         /* KB_CS_IO_PAGES pages, or NULL */
   volatile uint32_t *doorbell;      /* page 0 */
   volatile uint64_t *input;         /* page 1 */
   const volatile uint64_t *output;  /* page 2 */
   bool registered;
   bool kicked;                      /* KICK issued at least once */
   /* Statistics: publishes, KICK ioctls, doorbell-only publishes. */
   uint64_t publishes, kick_ioctls, doorbells;
};

/*
 * Allocate a ring of ring_size bytes (a power of two >= 4 KiB; flags
 * CPU/GPU read-write, SAME_VA, CPU-uncached as the blob's 0x200f) and
 * register it (CS_QUEUE_REGISTER) with the given priority (0..15).
 */
enum mali_kbase_result
mali_kbase_queue_create(struct mali_kbase *kb, uint32_t ring_size,
                        uint8_t priority, struct mali_kbase_queue *q);

/* CS_QUEUE_BIND to slot csi of group, then map the user I/O pages. */
enum mali_kbase_result
mali_kbase_queue_bind(struct mali_kbase *kb, struct mali_kbase_queue *q,
                      uint8_t group, uint8_t csi);

/* Unmap the I/O pages, CS_QUEUE_TERMINATE, free the ring. Terminate the
 * group first (the blob's order). Clears *q. */
void mali_kbase_queue_destroy(struct mali_kbase *kb, struct mali_kbase_queue *q);

/* CS_EXTRACT: bytes the firmware has consumed (0 before the first bind). */
uint64_t mali_kbase_queue_extract(const struct mali_kbase_queue *q);

/* Bytes that can be written now without overtaking the firmware, keeping
 * a 64-byte reserve for the publish padding. */
uint64_t mali_kbase_queue_space(const struct mali_kbase_queue *q);

/* Append count instruction words. The caller checked the space. */
void mali_kbase_queue_write(struct mali_kbase_queue *q, const uint64_t *words,
                            uint32_t count);

/*
 * Hand what was written to the firmware: pad to 64 bytes with NOPs, write
 * CS_INSERT, then ring the doorbell if the CS is already running, else
 * CS_QUEUE_KICK. Nothing if nothing was written.
 */
void mali_kbase_queue_publish(struct mali_kbase *kb, struct mali_kbase_queue *q);

/* The tiler heap. */
struct mali_kbase_tiler_heap_info {
   uint32_t chunk_size;
   uint32_t initial_chunks;
   uint32_t max_chunks;
   uint16_t target_in_flight;
   uint8_t group_id;
   uint64_t buf_desc_va;   /* the heap's buffer descriptor, or 0 */
};

enum mali_kbase_result
mali_kbase_tiler_heap_init(struct mali_kbase *kb,
                           const struct mali_kbase_tiler_heap_info *info,
                           uint64_t *heap_ctx_va, uint64_t *first_chunk_va);
void mali_kbase_tiler_heap_term(struct mali_kbase *kb, uint64_t heap_ctx_va);

/* KBASE_IOCTL_CS_EVENT_SIGNAL: make the firmware re-evaluate sync waits
 * after the CPU wrote sync memory. */
void mali_kbase_event_signal(struct mali_kbase *kb);

/* Whether the backend can wait for kernel events (poll + read). */
bool mali_kbase_has_events(const struct mali_kbase *kb);

/*
 * Wait until the kbase fd is readable or wake_fd (another fd, -1 for none)
 * is. Returns a mask: 1 = kbase event pending, 2 = wake_fd readable,
 * 4 = the kbase fd reported an error or hang-up; 0 on timeout; -1 on
 * failure.
 */
int mali_kbase_event_wait(struct mali_kbase *kb, int wake_fd, int timeout_ms);

/* Read one notification record (the kernel always returns one). */
enum mali_kbase_result mali_kbase_event_read(struct mali_kbase *kb,
                                             struct kb_csf_notification *n);

/* Answer a CPU_QUEUE_DUMP request with an empty dump (we have no CPU
 * queues). */
void mali_kbase_cpu_queue_dump(struct mali_kbase *kb);

/* ---------------------------------------------------------------------- */
/* kcpu queue and sync files (kcpu.c)                                      */

/*
 * A kcpu queue: commands the kernel executes in order on the CPU side
 * (wait for a sync file, wait for or set CQS objects in CSF event memory,
 * create a sync file that signals when the queue gets there). Used for
 * sync-file import and export, as the blob does.
 */
enum mali_kbase_result mali_kbase_kcpu_queue_create(struct mali_kbase *kb, uint8_t *id);
void mali_kbase_kcpu_queue_destroy(struct mali_kbase *kb, uint8_t id);

/*
 * KCPU_QUEUE_ENQUEUE of one command (the r44p1 kernel takes exactly one
 * per call). The kernel copies the command and what it points to during
 * the call; after a FENCE_SIGNAL the struct kb_fence holds the new
 * sync-file fd, owned by the caller. A full queue (EBUSY) is retried for
 * up to two seconds, as the blob retries.
 */
enum mali_kbase_result mali_kbase_kcpu_enqueue(struct mali_kbase *kb, uint8_t id,
                                               const struct kb_kcpu_command *cmd);

/* KBASE_IOCTL_FENCE_VALIDATE: is fd a sync file (the blob's check on
 * import)? */
enum mali_kbase_result mali_kbase_fence_validate(struct mali_kbase *kb, int fd);

#ifdef __cplusplus
}
#endif

#endif /* MALI_KBASE_H */
