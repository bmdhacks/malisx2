/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The kbase kernel interface (Arm's Mali kernel driver, /dev/mali<N>), CSF
 * flavour, UK version 1.20 (DDK r44p1), plus the one 1.18 variant we use
 * on older kernels: the ioctl argument structures,
 * ioctl numbers and flag values this driver uses.
 *
 * Licensing choice: these are our own definitions, written from the r44p1
 * kbase UAPI headers (GPL-2.0 WITH Linux-syscall-note) rather than copied
 * from them, so this file stays MIT. Names use a KB_/kb_ prefix so they can
 * never collide with the kernel's KBASE_/kbase_ names. The layouts are
 * checked twice: the static asserts below pin every size and field offset
 * on every build, and tests/test_kbase_uapi_abi.c compiles these
 * definitions next to the kernel's own headers and compares sizes, offsets
 * and ioctl numbers (built when the headers are available, see
 * meson.options).
 *
 * Only the ioctls the driver issues are here.
 */

#ifndef MALI_KBASE_UAPI_H
#define MALI_KBASE_UAPI_H

#include <stddef.h>
#include <stdint.h>
#include <linux/ioctl.h>

/* The UK (user-kernel) interface version we speak. The kernel answers a
 * VERSION_CHECK with min(its minor, our minor) for major 1. We propose 1.20
 * and accept 1.18 and up back (KB_UK_MIN_MINOR). Between 1.18 and 1.20 the
 * structures and ioctl numbers we use are the same except for queue group
 * creation, whose 112-byte form is new in 1.19 (KB_UK_GROUP_CREATE_NE_MINOR);
 * below that we issue the older 40-byte form. 1.20's only other change, the
 * refusal of file operations on an inherited fd, does not affect us: we
 * open with O_CLOEXEC and never use the fd from a child. Below 1.18 a
 * context needs the tracking page mapped before any allocation, which we
 * do not do, so 1.18 is the floor. */
#define KB_UK_VERSION_MAJOR 1
#define KB_UK_VERSION_MINOR 20
#define KB_UK_MIN_MINOR 18
#define KB_UK_GROUP_CREATE_NE_MINOR 19

#define KB_IOCTL_TYPE 0x80

/* The kernel's page size for va_pages/commit_pages counts. The RG 477V
 * kernel uses 4 KiB pages; the context checks this at creation. */
#define KB_PAGE_SHIFT 12
#define KB_PAGE_SIZE (1u << KB_PAGE_SHIFT)

/* ---------------------------------------------------------------------- */
/* Argument structures                                                     */

struct kb_ioctl_version_check {
   uint16_t major;
   uint16_t minor;
};

struct kb_ioctl_set_flags {
   uint32_t create_flags;
};

struct kb_ioctl_get_gpuprops {
   uint64_t buffer; /* user pointer; 0 to ask for the stream size */
   uint32_t size;
   uint32_t flags;  /* must be 0 */
};

union kb_ioctl_mem_alloc {
   struct {
      uint64_t va_pages;
      uint64_t commit_pages;
      uint64_t extension;
      uint64_t flags;
   } in;
   struct {
      uint64_t flags;
      uint64_t gpu_va;
   } out;
};

/* UK >= 1.9. The blob uses this one whenever the kernel allows it. */
union kb_ioctl_mem_alloc_ex {
   struct {
      uint64_t va_pages;
      uint64_t commit_pages;
      uint64_t extension;
      uint64_t flags;
      uint64_t fixed_address; /* only with KB_MEM_FIXED, else 0 */
      uint64_t extra[3];      /* must be 0 */
   } in;
   struct {
      uint64_t flags;
      uint64_t gpu_va;
   } out;
};

union kb_ioctl_mem_query {
   struct {
      uint64_t gpu_addr;
      uint64_t query;
   } in;
   struct {
      uint64_t value;
   } out;
};

struct kb_ioctl_mem_free {
   uint64_t gpu_addr;
};

struct kb_ioctl_mem_sync {
   uint64_t handle;    /* GPU VA of the region (page aligned) */
   uint64_t user_addr; /* CPU address of the range */
   uint64_t size;
   uint8_t type;       /* KB_SYNCSET_OP_* */
   uint8_t padding[7];
};

struct kb_ioctl_mem_commit {
   uint64_t gpu_addr;
   uint64_t pages;
};

union kb_ioctl_mem_import {
   struct {
      uint64_t flags;
      uint64_t phandle; /* user pointer to the handle (an int fd for UMM) */
      uint32_t type;    /* KB_MEM_IMPORT_TYPE_* */
      uint32_t padding; /* extra VA pages after the buffer (UMM only) */
   } in;
   struct {
      uint64_t flags;
      uint64_t gpu_va;
      uint64_t va_pages;
   } out;
};

struct kb_ioctl_mem_flags_change {
   uint64_t gpu_va;
   uint64_t flags;
   uint64_t mask;
};

struct kb_ioctl_mem_exec_init {
   uint64_t va_pages;
};

/* MEM_JIT_INIT: on CSF kernels this is also what creates the CUSTOM_VA
 * zone of a 64-bit context, which the kernel's own allocations (tiler heap
 * contexts and chunks) come from. */
struct kb_ioctl_mem_jit_init {
   uint64_t va_pages;
   uint8_t max_allocations;
   uint8_t trim_level;
   uint8_t group_id;
   uint8_t padding[5];
   uint64_t phys_pages;
};

/* STICKY_RESOURCE_MAP and _UNMAP share this layout: address points at an
 * array of count u64 GPU addresses. */
struct kb_ioctl_sticky_resource {
   uint64_t count;
   uint64_t address;
};

union kb_ioctl_cs_get_glb_iface {
   struct {
      uint32_t max_group_num;
      uint32_t max_total_stream_num;
      uint64_t groups_ptr;
      uint64_t streams_ptr;
   } in;
   struct {
      uint32_t glb_version;
      uint32_t features;
      uint32_t group_num;
      uint32_t prfcnt_size;
      uint32_t total_stream_num;
      uint32_t instr_features;
   } out;
};

/* Elements of the two arrays CS_GET_GLB_IFACE fills. */
struct kb_cs_group_control {
   uint32_t features;
   uint32_t stream_num;
   uint32_t suspend_size;
   uint32_t padding;
};

struct kb_cs_stream_control {
   uint32_t features;
   uint32_t padding;
};

/* CSF queues, groups, tiler heap. */

struct kb_ioctl_cs_queue_register {
   uint64_t buffer_gpu_addr;  /* ring, page aligned */
   uint32_t buffer_size;      /* power of two, >= 4 KiB */
   uint8_t priority;          /* 0..15 within the group */
   uint8_t padding[3];
};

struct kb_ioctl_cs_queue_kick {
   uint64_t buffer_gpu_addr;
};

union kb_ioctl_cs_queue_bind {
   struct {
      uint64_t buffer_gpu_addr;
      uint8_t group_handle;
      uint8_t csi_index;
      uint8_t padding[6];
   } in;
   struct {
      uint64_t mmap_handle;  /* mmap offset of the three user I/O pages */
   } out;
};

struct kb_ioctl_cs_queue_terminate {
   uint64_t buffer_gpu_addr;
};

/* UK >= 1.19 form (nr 58). */
union kb_ioctl_cs_queue_group_create {
   struct {
      uint64_t tiler_mask;
      uint64_t fragment_mask;
      uint64_t compute_mask;
      uint8_t cs_min;
      uint8_t priority;       /* KB_QUEUE_GROUP_PRIORITY_* */
      uint8_t tiler_max;
      uint8_t fragment_max;
      uint8_t compute_max;
      uint8_t csi_handlers;   /* exception handler flags, 0 */
      uint16_t reserved;
      uint64_t dvs_buf;
      uint64_t padding[9];
   } in;
   struct {
      uint8_t group_handle;
      uint8_t padding[3];
      uint32_t group_uid;
   } out;
};

/* UK 1.6 to 1.18 form (also nr 58; the ioctl number differs by its size).
 * Kernels at 1.19 and newer still accept it. */
union kb_ioctl_cs_queue_group_create_1_18 {
   struct {
      uint64_t tiler_mask;
      uint64_t fragment_mask;
      uint64_t compute_mask;
      uint8_t cs_min;
      uint8_t priority;
      uint8_t tiler_max;
      uint8_t fragment_max;
      uint8_t compute_max;
      uint8_t csi_handlers;
      uint8_t padding[2];
      uint64_t dvs_buf;
   } in;
   struct {
      uint8_t group_handle;
      uint8_t padding[3];
      uint32_t group_uid;
   } out;
};

struct kb_ioctl_cs_queue_group_term {
   uint8_t group_handle;
   uint8_t padding[7];
};

/* UK >= 1.14 form, with the buffer descriptor address. */
union kb_ioctl_cs_tiler_heap_init {
   struct {
      uint32_t chunk_size;
      uint32_t initial_chunks;
      uint32_t max_chunks;
      uint16_t target_in_flight;
      uint8_t group_id;
      uint8_t padding;
      uint64_t buf_desc_va;
   } in;
   struct {
      uint64_t gpu_heap_va;    /* the heap context, for HEAP_SET */
      uint64_t first_chunk_va;
   } out;
};

struct kb_ioctl_cs_tiler_heap_term {
   uint64_t gpu_heap_va;
};

struct kb_ioctl_cs_cpu_queue_info {
   uint64_t buffer;
   uint64_t size;
};

#define KB_QUEUE_GROUP_PRIORITY_HIGH     0
#define KB_QUEUE_GROUP_PRIORITY_MEDIUM   1
#define KB_QUEUE_GROUP_PRIORITY_LOW      2
#define KB_QUEUE_GROUP_PRIORITY_REALTIME 3

#define KB_QUEUE_MAX_PRIORITY 15

/* The user I/O pages of a bound queue (three pages at the bind's
 * mmap_handle): page 0 the hardware doorbell, page 1 the input page, page 2
 * the output page. Offsets within the input and output pages: */
#define KB_CS_IO_PAGES 3
#define KB_CS_INPUT_INSERT        0x00  /* u64, written by us */
#define KB_CS_INPUT_EXTRACT_INIT  0x08  /* u64 */
#define KB_CS_OUTPUT_EXTRACT      0x00  /* u64, written by the firmware */
#define KB_CS_OUTPUT_ACTIVE       0x08  /* u32, bit 0 */

/* ---------------------------------------------------------------------- */
/* Event channel: read(2) on the kbase fd returns one 64-byte record.      */

#define KB_CSF_NOTIFICATION_EVENT                 0
#define KB_CSF_NOTIFICATION_GPU_QUEUE_GROUP_ERROR 1
#define KB_CSF_NOTIFICATION_CPU_QUEUE_DUMP        2

#define KB_GPU_QUEUE_GROUP_ERROR_FATAL          0
#define KB_GPU_QUEUE_GROUP_QUEUE_ERROR_FATAL    1
#define KB_GPU_QUEUE_GROUP_ERROR_TIMEOUT        2
#define KB_GPU_QUEUE_GROUP_ERROR_TILER_HEAP_OOM 3

struct kb_gpu_queue_group_error {
   uint8_t error_type;       /* KB_GPU_QUEUE_GROUP_ERROR_* */
   uint8_t padding[7];
   union {
      struct {
         uint64_t sideband;
         uint32_t status;
         uint32_t padding;
      } fatal_group;
      struct {
         uint64_t sideband;
         uint32_t status;
         uint8_t csi_index;
         uint8_t padding[3];
      } fatal_queue;
   } payload;
};

struct kb_csf_notification {
   uint8_t type;             /* KB_CSF_NOTIFICATION_* */
   uint8_t padding[7];
   union {
      struct {
         uint8_t handle;
         uint8_t padding[7];
         struct kb_gpu_queue_group_error error;
      } csg_error;
      uint8_t align[56];
   } payload;
};

/* GET_CPU_GPU_TIMEINFO: a CLOCK_MONOTONIC time and the GPU's system
 * timestamp (and cycle counter) sampled together by the kernel. The
 * timestamp counts at the CPU's architected timer frequency on CSF GPUs
 * (kbase mali_kbase_time.c). */
#define KB_TIMEINFO_MONOTONIC     (1u << 0)
#define KB_TIMEINFO_TIMESTAMP     (1u << 1)
#define KB_TIMEINFO_CYCLE_COUNTER (1u << 2)
#define KB_TIMEINFO_USER_SOURCE   (1u << 31)

union kb_ioctl_get_cpu_gpu_timeinfo {
   struct {
      uint32_t request_flags;
      uint32_t paddings[7];
   } in;
   struct {
      uint64_t sec;
      uint32_t nsec;
      uint32_t padding;
      uint64_t timestamp;
      uint64_t cycle_counter;
   } out;
};

/* ---------------------------------------------------------------------- */
/* Ioctl numbers                                                           */

#define KB_IOCTL_SET_FLAGS        _IOW(KB_IOCTL_TYPE, 1, struct kb_ioctl_set_flags)
#define KB_IOCTL_GET_GPUPROPS     _IOW(KB_IOCTL_TYPE, 3, struct kb_ioctl_get_gpuprops)
#define KB_IOCTL_MEM_ALLOC        _IOWR(KB_IOCTL_TYPE, 5, union kb_ioctl_mem_alloc)
#define KB_IOCTL_MEM_QUERY        _IOWR(KB_IOCTL_TYPE, 6, union kb_ioctl_mem_query)
#define KB_IOCTL_MEM_FREE         _IOW(KB_IOCTL_TYPE, 7, struct kb_ioctl_mem_free)
#define KB_IOCTL_MEM_JIT_INIT     _IOW(KB_IOCTL_TYPE, 14, struct kb_ioctl_mem_jit_init)
#define KB_IOCTL_MEM_SYNC         _IOW(KB_IOCTL_TYPE, 15, struct kb_ioctl_mem_sync)
#define KB_IOCTL_MEM_COMMIT       _IOW(KB_IOCTL_TYPE, 20, struct kb_ioctl_mem_commit)
#define KB_IOCTL_MEM_IMPORT       _IOWR(KB_IOCTL_TYPE, 22, union kb_ioctl_mem_import)
#define KB_IOCTL_MEM_FLAGS_CHANGE _IOW(KB_IOCTL_TYPE, 23, struct kb_ioctl_mem_flags_change)
#define KB_IOCTL_STICKY_RESOURCE_MAP   _IOW(KB_IOCTL_TYPE, 29, struct kb_ioctl_sticky_resource)
#define KB_IOCTL_STICKY_RESOURCE_UNMAP _IOW(KB_IOCTL_TYPE, 30, struct kb_ioctl_sticky_resource)
#define KB_IOCTL_CS_QUEUE_REGISTER  _IOW(KB_IOCTL_TYPE, 36, struct kb_ioctl_cs_queue_register)
#define KB_IOCTL_CS_QUEUE_KICK      _IOW(KB_IOCTL_TYPE, 37, struct kb_ioctl_cs_queue_kick)
#define KB_IOCTL_CS_QUEUE_BIND      _IOWR(KB_IOCTL_TYPE, 39, union kb_ioctl_cs_queue_bind)
#define KB_IOCTL_CS_QUEUE_TERMINATE _IOW(KB_IOCTL_TYPE, 41, struct kb_ioctl_cs_queue_terminate)
#define KB_IOCTL_CS_QUEUE_GROUP_TERMINATE \
   _IOW(KB_IOCTL_TYPE, 43, struct kb_ioctl_cs_queue_group_term)
#define KB_IOCTL_CS_EVENT_SIGNAL    _IO(KB_IOCTL_TYPE, 44)
#define KB_IOCTL_CS_TILER_HEAP_INIT _IOWR(KB_IOCTL_TYPE, 48, union kb_ioctl_cs_tiler_heap_init)
#define KB_IOCTL_CS_TILER_HEAP_TERM _IOW(KB_IOCTL_TYPE, 49, struct kb_ioctl_cs_tiler_heap_term)
#define KB_IOCTL_CS_CPU_QUEUE_DUMP  _IOW(KB_IOCTL_TYPE, 53, struct kb_ioctl_cs_cpu_queue_info)
#define KB_IOCTL_CS_QUEUE_GROUP_CREATE \
   _IOWR(KB_IOCTL_TYPE, 58, union kb_ioctl_cs_queue_group_create)
#define KB_IOCTL_CS_QUEUE_GROUP_CREATE_1_18 \
   _IOWR(KB_IOCTL_TYPE, 58, union kb_ioctl_cs_queue_group_create_1_18)
#define KB_IOCTL_MEM_EXEC_INIT    _IOW(KB_IOCTL_TYPE, 38, struct kb_ioctl_mem_exec_init)
#define KB_IOCTL_CS_GET_GLB_IFACE _IOWR(KB_IOCTL_TYPE, 51, union kb_ioctl_cs_get_glb_iface)
#define KB_IOCTL_VERSION_CHECK    _IOWR(KB_IOCTL_TYPE, 52, struct kb_ioctl_version_check)
#define KB_IOCTL_MEM_ALLOC_EX     _IOWR(KB_IOCTL_TYPE, 59, union kb_ioctl_mem_alloc_ex)
#define KB_IOCTL_GET_CPU_GPU_TIMEINFO _IOWR(KB_IOCTL_TYPE, 50, union kb_ioctl_get_cpu_gpu_timeinfo)

/* ---------------------------------------------------------------------- */
/* Memory allocation flags (base_mem_alloc_flags, 30 bits)                 */

#define KB_MEM_PROT_CPU_RD                (1ull << 0)
#define KB_MEM_PROT_CPU_WR                (1ull << 1)
#define KB_MEM_PROT_GPU_RD                (1ull << 2)
#define KB_MEM_PROT_GPU_WR                (1ull << 3)
#define KB_MEM_PROT_GPU_EX                (1ull << 4)
#define KB_MEM_GPU_VA_SAME_4GB_PAGE       (1ull << 6)
#define KB_MEM_FIXED                      (1ull << 8)  /* CSF */
#define KB_MEM_GROW_ON_GPF                (1ull << 9)
#define KB_MEM_COHERENT_SYSTEM            (1ull << 10)
#define KB_MEM_COHERENT_LOCAL             (1ull << 11)
#define KB_MEM_CACHED_CPU                 (1ull << 12)
#define KB_MEM_SAME_VA                    (1ull << 13)
#define KB_MEM_NEED_MMAP                  (1ull << 14) /* output only */
#define KB_MEM_COHERENT_SYSTEM_REQUIRED   (1ull << 15)
#define KB_MEM_PROTECTED                  (1ull << 16)
#define KB_MEM_DONT_NEED                  (1ull << 17)
#define KB_MEM_IMPORT_SHARED              (1ull << 18)
#define KB_MEM_CSF_EVENT                  (1ull << 19) /* CSF */
#define KB_MEM_UNCACHED_GPU               (1ull << 21)
#define KB_MEM_GROUP_ID_SHIFT             22
#define KB_MEM_GROUP_ID_MASK              (0xfull << KB_MEM_GROUP_ID_SHIFT)
#define KB_MEM_IMPORT_SYNC_ON_MAP_UNMAP   (1ull << 26)
#define KB_MEM_KERNEL_SYNC                (1ull << 28) /* output: CPU cache
                                                          maintenance must go
                                                          through MEM_SYNC */
#define KB_MEM_FIXABLE                    (1ull << 29) /* CSF */
#define KB_MEM_FLAGS_NR_BITS              30
#define KB_MEM_FLAGS_INPUT_MASK \
   (((1ull << KB_MEM_FLAGS_NR_BITS) - 1) & ~KB_MEM_NEED_MMAP)

#define KB_MEM_GROUP_ID(g) (((uint64_t)(g) & 0xf) << KB_MEM_GROUP_ID_SHIFT)

/* Special mmap offsets ("handles") on the kbase fd. */
#define KB_MEM_MAP_TRACKING_HANDLE        (3ull << KB_PAGE_SHIFT)
#define KB_MEM_WRITE_ALLOC_PAGES_HANDLE   (4ull << KB_PAGE_SHIFT)
#define KB_MEM_CSF_USER_REG_PAGE_HANDLE   (47ull << KB_PAGE_SHIFT)
#define KB_MEM_CSF_USER_IO_PAGES_HANDLE   (48ull << KB_PAGE_SHIFT)
#define KB_MEM_COOKIE_BASE                (64ull << KB_PAGE_SHIFT)
/* 64 cookies on a 64-bit kernel; addresses at or above this are real VAs. */
#define KB_MEM_FIRST_FREE_ADDRESS         ((64ull << KB_PAGE_SHIFT) + KB_MEM_COOKIE_BASE)

/* Register offsets inside the user register page. */
#define KB_USER_REG_LATEST_FLUSH          0x0000

/* MEM_QUERY queries. */
#define KB_MEM_QUERY_COMMIT_SIZE 1
#define KB_MEM_QUERY_VA_SIZE     2
#define KB_MEM_QUERY_FLAGS       3

/* MEM_SYNC types. */
#define KB_SYNCSET_OP_MSYNC 1 /* CPU caches to memory (clean) */
#define KB_SYNCSET_OP_CSYNC 2 /* memory to CPU (invalidate) */

/* MEM_IMPORT types. */
#define KB_MEM_IMPORT_TYPE_UMM         2 /* dma-buf */
#define KB_MEM_IMPORT_TYPE_USER_BUFFER 3

/* MEM_JIT_INIT as the blob's Vulkan context issues it: 0x2000000 pages
 * (128 GiB) of VA, 255 allocations, trim level 5, the physical limit
 * equal to the VA. */
#define KB_JIT_VA_PAGES        0x2000000ull
#define KB_JIT_MAX_ALLOCATIONS 255
#define KB_JIT_TRIM_LEVEL      5

/* MEM_EXEC_INIT: the blob asks for the whole 4 GiB executable zone. */
#define KB_EXEC_VA_PAGES (1ull << (32 - KB_PAGE_SHIFT))

/* ---------------------------------------------------------------------- */
/* Context creation flags (SET_FLAGS)                                      */

#define KB_CONTEXT_SYSTEM_MONITOR_SUBMIT_DISABLED (1u << 1)
#define KB_CONTEXT_MMU_GROUP_ID_SHIFT 3
#define KB_CONTEXT_MMU_GROUP_ID_MASK  (0xfu << KB_CONTEXT_MMU_GROUP_ID_SHIFT)
/* Everything else is refused by the kernel with EINVAL. */
#define KB_CONTEXT_KERNEL_FLAGS \
   (KB_CONTEXT_SYSTEM_MONITOR_SUBMIT_DISABLED | KB_CONTEXT_MMU_GROUP_ID_MASK)

/* ---------------------------------------------------------------------- */
/* GET_GPUPROPS stream: a sequence of (u32 token, value) records with      */
/* token = key << 2 | size code; size code 0..3 = 1, 2, 4, 8 bytes.        */

#define KB_GPUPROP_SIZE_U8  0
#define KB_GPUPROP_SIZE_U16 1
#define KB_GPUPROP_SIZE_U32 2
#define KB_GPUPROP_SIZE_U64 3

enum kb_gpuprop_key {
   KB_GPUPROP_PRODUCT_ID = 1,
   KB_GPUPROP_VERSION_STATUS = 2,
   KB_GPUPROP_MINOR_REVISION = 3,
   KB_GPUPROP_MAJOR_REVISION = 4,
   KB_GPUPROP_GPU_FREQ_KHZ_MAX = 6,
   KB_GPUPROP_LOG2_PROGRAM_COUNTER_SIZE = 8,
   KB_GPUPROP_TEXTURE_FEATURES_0 = 9,
   KB_GPUPROP_TEXTURE_FEATURES_1 = 10,
   KB_GPUPROP_TEXTURE_FEATURES_2 = 11,
   KB_GPUPROP_GPU_AVAILABLE_MEMORY_SIZE = 12,
   KB_GPUPROP_L2_LOG2_LINE_SIZE = 13,
   KB_GPUPROP_L2_LOG2_CACHE_SIZE = 14,
   KB_GPUPROP_L2_NUM_L2_SLICES = 15,
   KB_GPUPROP_TILER_BIN_SIZE_BYTES = 16,
   KB_GPUPROP_TILER_MAX_ACTIVE_LEVELS = 17,
   KB_GPUPROP_MAX_THREADS = 18,
   KB_GPUPROP_MAX_WORKGROUP_SIZE = 19,
   KB_GPUPROP_MAX_BARRIER_SIZE = 20,
   KB_GPUPROP_MAX_REGISTERS = 21,
   KB_GPUPROP_MAX_TASK_QUEUE = 22,
   KB_GPUPROP_MAX_THREAD_GROUP_SPLIT = 23,
   KB_GPUPROP_IMPL_TECH = 24,
   KB_GPUPROP_RAW_SHADER_PRESENT = 25,
   KB_GPUPROP_RAW_TILER_PRESENT = 26,
   KB_GPUPROP_RAW_L2_PRESENT = 27,
   KB_GPUPROP_RAW_STACK_PRESENT = 28,
   KB_GPUPROP_RAW_L2_FEATURES = 29,
   KB_GPUPROP_RAW_CORE_FEATURES = 30,
   KB_GPUPROP_RAW_MEM_FEATURES = 31,
   KB_GPUPROP_RAW_MMU_FEATURES = 32,
   KB_GPUPROP_RAW_AS_PRESENT = 33,
   KB_GPUPROP_RAW_JS_PRESENT = 34,
   KB_GPUPROP_RAW_JS_FEATURES_0 = 35, /* .. 50, job manager only */
   KB_GPUPROP_RAW_TILER_FEATURES = 51,
   KB_GPUPROP_RAW_TEXTURE_FEATURES_0 = 52,
   KB_GPUPROP_RAW_TEXTURE_FEATURES_1 = 53,
   KB_GPUPROP_RAW_TEXTURE_FEATURES_2 = 54,
   KB_GPUPROP_RAW_GPU_ID = 55,
   KB_GPUPROP_RAW_THREAD_MAX_THREADS = 56,
   KB_GPUPROP_RAW_THREAD_MAX_WORKGROUP_SIZE = 57,
   KB_GPUPROP_RAW_THREAD_MAX_BARRIER_SIZE = 58,
   KB_GPUPROP_RAW_THREAD_FEATURES = 59,
   KB_GPUPROP_RAW_COHERENCY_MODE = 60,
   KB_GPUPROP_COHERENCY_NUM_GROUPS = 61,
   KB_GPUPROP_COHERENCY_NUM_CORE_GROUPS = 62,
   KB_GPUPROP_COHERENCY_COHERENCY = 63,
   KB_GPUPROP_COHERENCY_GROUP_0 = 64, /* .. 79 */
   KB_GPUPROP_TEXTURE_FEATURES_3 = 80,
   KB_GPUPROP_RAW_TEXTURE_FEATURES_3 = 81,
   KB_GPUPROP_NUM_EXEC_ENGINES = 82,
   KB_GPUPROP_RAW_THREAD_TLS_ALLOC = 83,
   KB_GPUPROP_TLS_ALLOC = 84,
   KB_GPUPROP_RAW_GPU_FEATURES = 85,
   KB_GPUPROP_KEY_COUNT = 86,
};

/* ---------------------------------------------------------------------- */
/* Layout checks (the kernel headers are the reference; see the test).     */

#define KB_CHECK_SIZE(t, n) _Static_assert(sizeof(t) == (n), #t " size")
#define KB_CHECK_OFF(t, f, n) \
   _Static_assert(offsetof(t, f) == (n), #t "." #f " offset")

KB_CHECK_SIZE(struct kb_ioctl_version_check, 4);
KB_CHECK_OFF(struct kb_ioctl_version_check, minor, 2);
KB_CHECK_SIZE(struct kb_ioctl_set_flags, 4);
KB_CHECK_SIZE(struct kb_ioctl_get_gpuprops, 16);
KB_CHECK_OFF(struct kb_ioctl_get_gpuprops, size, 8);
KB_CHECK_OFF(struct kb_ioctl_get_gpuprops, flags, 12);
KB_CHECK_SIZE(union kb_ioctl_mem_alloc, 32);
KB_CHECK_OFF(union kb_ioctl_mem_alloc, in.flags, 24);
KB_CHECK_OFF(union kb_ioctl_mem_alloc, out.gpu_va, 8);
KB_CHECK_SIZE(union kb_ioctl_mem_alloc_ex, 64);
KB_CHECK_OFF(union kb_ioctl_mem_alloc_ex, in.flags, 24);
KB_CHECK_OFF(union kb_ioctl_mem_alloc_ex, in.fixed_address, 32);
KB_CHECK_OFF(union kb_ioctl_mem_alloc_ex, in.extra, 40);
KB_CHECK_OFF(union kb_ioctl_mem_alloc_ex, out.gpu_va, 8);
KB_CHECK_SIZE(union kb_ioctl_mem_query, 16);
KB_CHECK_OFF(union kb_ioctl_mem_query, in.query, 8);
KB_CHECK_SIZE(struct kb_ioctl_mem_free, 8);
KB_CHECK_SIZE(struct kb_ioctl_mem_sync, 32);
KB_CHECK_OFF(struct kb_ioctl_mem_sync, size, 16);
KB_CHECK_OFF(struct kb_ioctl_mem_sync, type, 24);
KB_CHECK_SIZE(struct kb_ioctl_mem_commit, 16);
KB_CHECK_SIZE(union kb_ioctl_mem_import, 24);
KB_CHECK_OFF(union kb_ioctl_mem_import, in.phandle, 8);
KB_CHECK_OFF(union kb_ioctl_mem_import, in.type, 16);
KB_CHECK_OFF(union kb_ioctl_mem_import, in.padding, 20);
KB_CHECK_OFF(union kb_ioctl_mem_import, out.va_pages, 16);
KB_CHECK_SIZE(struct kb_ioctl_mem_flags_change, 24);
KB_CHECK_SIZE(struct kb_ioctl_mem_exec_init, 8);
KB_CHECK_SIZE(struct kb_ioctl_mem_jit_init, 24);
KB_CHECK_OFF(struct kb_ioctl_mem_jit_init, group_id, 10);
KB_CHECK_OFF(struct kb_ioctl_mem_jit_init, phys_pages, 16);
KB_CHECK_SIZE(struct kb_ioctl_sticky_resource, 16);
KB_CHECK_SIZE(union kb_ioctl_cs_get_glb_iface, 24);
KB_CHECK_OFF(union kb_ioctl_cs_get_glb_iface, in.groups_ptr, 8);
KB_CHECK_OFF(union kb_ioctl_cs_get_glb_iface, in.streams_ptr, 16);
KB_CHECK_OFF(union kb_ioctl_cs_get_glb_iface, out.total_stream_num, 16);
KB_CHECK_OFF(union kb_ioctl_cs_get_glb_iface, out.instr_features, 20);
KB_CHECK_SIZE(struct kb_cs_group_control, 16);
KB_CHECK_SIZE(struct kb_cs_stream_control, 8);

KB_CHECK_SIZE(struct kb_ioctl_cs_queue_register, 16);
KB_CHECK_OFF(struct kb_ioctl_cs_queue_register, buffer_size, 8);
KB_CHECK_OFF(struct kb_ioctl_cs_queue_register, priority, 12);
KB_CHECK_SIZE(struct kb_ioctl_cs_queue_kick, 8);
KB_CHECK_SIZE(union kb_ioctl_cs_queue_bind, 16);
KB_CHECK_OFF(union kb_ioctl_cs_queue_bind, in.group_handle, 8);
KB_CHECK_OFF(union kb_ioctl_cs_queue_bind, in.csi_index, 9);
KB_CHECK_SIZE(struct kb_ioctl_cs_queue_terminate, 8);
KB_CHECK_SIZE(union kb_ioctl_cs_queue_group_create, 112);
KB_CHECK_OFF(union kb_ioctl_cs_queue_group_create, in.cs_min, 24);
KB_CHECK_OFF(union kb_ioctl_cs_queue_group_create, in.csi_handlers, 29);
KB_CHECK_OFF(union kb_ioctl_cs_queue_group_create, in.dvs_buf, 32);
KB_CHECK_OFF(union kb_ioctl_cs_queue_group_create, out.group_uid, 4);
KB_CHECK_SIZE(union kb_ioctl_cs_queue_group_create_1_18, 40);
KB_CHECK_OFF(union kb_ioctl_cs_queue_group_create_1_18, in.cs_min, 24);
KB_CHECK_OFF(union kb_ioctl_cs_queue_group_create_1_18, in.csi_handlers, 29);
KB_CHECK_OFF(union kb_ioctl_cs_queue_group_create_1_18, in.dvs_buf, 32);
KB_CHECK_OFF(union kb_ioctl_cs_queue_group_create_1_18, out.group_uid, 4);
KB_CHECK_SIZE(struct kb_ioctl_cs_queue_group_term, 8);
KB_CHECK_SIZE(union kb_ioctl_cs_tiler_heap_init, 24);
KB_CHECK_OFF(union kb_ioctl_cs_tiler_heap_init, in.target_in_flight, 12);
KB_CHECK_OFF(union kb_ioctl_cs_tiler_heap_init, in.group_id, 14);
KB_CHECK_OFF(union kb_ioctl_cs_tiler_heap_init, in.buf_desc_va, 16);
KB_CHECK_OFF(union kb_ioctl_cs_tiler_heap_init, out.first_chunk_va, 8);
KB_CHECK_SIZE(struct kb_ioctl_cs_tiler_heap_term, 8);
KB_CHECK_SIZE(struct kb_ioctl_cs_cpu_queue_info, 16);
KB_CHECK_SIZE(struct kb_gpu_queue_group_error, 24);
KB_CHECK_OFF(struct kb_gpu_queue_group_error, payload.fatal_group.status, 16);
KB_CHECK_OFF(struct kb_gpu_queue_group_error, payload.fatal_queue.csi_index, 20);
KB_CHECK_SIZE(struct kb_csf_notification, 64);
KB_CHECK_OFF(struct kb_csf_notification, payload.csg_error.error, 16);

/* The ioctl numbers the blob was seen issuing, as an independent check of
 * the encoding. */
_Static_assert(KB_IOCTL_SET_FLAGS == 0x40048001, "SET_FLAGS");
_Static_assert(KB_IOCTL_GET_GPUPROPS == 0x40108003, "GET_GPUPROPS");
_Static_assert(KB_IOCTL_MEM_ALLOC == 0xc0208005, "MEM_ALLOC");
_Static_assert(KB_IOCTL_MEM_QUERY == 0xc0108006, "MEM_QUERY");
_Static_assert(KB_IOCTL_MEM_FREE == 0x40088007, "MEM_FREE");
_Static_assert(KB_IOCTL_MEM_SYNC == 0x4020800f, "MEM_SYNC");
_Static_assert(KB_IOCTL_MEM_COMMIT == 0x40108014, "MEM_COMMIT");
_Static_assert(KB_IOCTL_MEM_IMPORT == 0xc0188016, "MEM_IMPORT");
_Static_assert(KB_IOCTL_MEM_FLAGS_CHANGE == 0x40188017, "MEM_FLAGS_CHANGE");
_Static_assert(KB_IOCTL_STICKY_RESOURCE_MAP == 0x4010801d, "STICKY_MAP");
_Static_assert(KB_IOCTL_STICKY_RESOURCE_UNMAP == 0x4010801e, "STICKY_UNMAP");
_Static_assert(KB_IOCTL_MEM_EXEC_INIT == 0x40088026, "MEM_EXEC_INIT");
_Static_assert(KB_IOCTL_MEM_JIT_INIT == 0x4018800e, "MEM_JIT_INIT");
_Static_assert(KB_IOCTL_CS_GET_GLB_IFACE == 0xc0188033, "CS_GET_GLB_IFACE");
_Static_assert(KB_IOCTL_VERSION_CHECK == 0xc0048034, "VERSION_CHECK");
_Static_assert(KB_IOCTL_MEM_ALLOC_EX == 0xc040803b, "MEM_ALLOC_EX");
KB_CHECK_SIZE(union kb_ioctl_get_cpu_gpu_timeinfo, 32);
_Static_assert(KB_IOCTL_GET_CPU_GPU_TIMEINFO == 0xc0208032, "GET_CPU_GPU_TIMEINFO");
_Static_assert(KB_IOCTL_CS_QUEUE_GROUP_CREATE == 0xc070803a, "GROUP_CREATE");
_Static_assert(KB_IOCTL_CS_QUEUE_GROUP_CREATE_1_18 == 0xc028803a, "GROUP_CREATE_1_18");
_Static_assert(KB_IOCTL_CS_QUEUE_GROUP_TERMINATE == 0x4008802b, "GROUP_TERMINATE");
_Static_assert(KB_IOCTL_CS_TILER_HEAP_INIT == 0xc0188030, "TILER_HEAP_INIT");
_Static_assert(KB_IOCTL_CS_EVENT_SIGNAL == 0x802c, "CS_EVENT_SIGNAL");

/* ---------------------------------------------------------------------- */
/* kcpu queues and sync files (sync-file import and export, as the blob    */
/* does)                                                                   */

/* The kernel's own fence: FENCE_SIGNAL writes a new sync-file fd into fd
 * at enqueue time; FENCE_WAIT reads fd. stream_fd is not used by the CSF
 * kernel. */
struct kb_fence {
   int32_t fd;
   int32_t stream_fd;
};

#define KB_KCPU_COMMAND_FENCE_SIGNAL       0
#define KB_KCPU_COMMAND_FENCE_WAIT         1
#define KB_KCPU_COMMAND_CQS_WAIT_OPERATION 4
#define KB_KCPU_COMMAND_CQS_SET_OPERATION  5

#define KB_CQS_DATA_TYPE_U32 0
#define KB_CQS_DATA_TYPE_U64 1
#define KB_CQS_WAIT_OPERATION_LE 0
#define KB_CQS_WAIT_OPERATION_GT 1
#define KB_CQS_SET_OPERATION_ADD 0
#define KB_CQS_SET_OPERATION_SET 1

/* At most this many objects per CQS command (BASEP_KCPU_CQS_MAX_NUM_OBJS). */
#define KB_KCPU_CQS_MAX_OBJS 32

/* One object of a CQS_WAIT_OPERATION or CQS_SET_OPERATION. The object is
 * a sync32/sync64 in CSF event memory: value, then a 32-bit error word at
 * +4 / +8; aligned to 8 / 16 bytes. */
struct kb_cqs_operation {
   uint64_t addr;
   uint64_t val;
   uint8_t operation;        /* KB_CQS_WAIT_OPERATION_* / KB_CQS_SET_OPERATION_* */
   uint8_t data_type;        /* KB_CQS_DATA_TYPE_* */
   uint8_t padding[6];
};

struct kb_kcpu_command {
   uint8_t type;             /* KB_KCPU_COMMAND_* */
   uint8_t padding[7];
   union {
      struct {
         uint64_t fence;     /* user pointer to a struct kb_fence */
      } fence;
      struct {
         uint64_t objs;      /* user pointer to struct kb_cqs_operation[] */
         uint32_t nr_objs;
         uint32_t inherit_err_flags;  /* waits: bit i takes object i's error */
      } cqs;
      uint64_t words[2];
   } info;
};

struct kb_ioctl_kcpu_queue_new {
   uint8_t id;
   uint8_t padding[7];
};

struct kb_ioctl_kcpu_queue_delete {
   uint8_t id;
   uint8_t padding[7];
};

struct kb_ioctl_kcpu_queue_enqueue {
   uint64_t addr;            /* user pointer to struct kb_kcpu_command[] */
   uint32_t nr_commands;
   uint8_t id;
   uint8_t padding[3];
};

struct kb_ioctl_fence_validate {
   int32_t fd;
};

#define KB_IOCTL_FENCE_VALIDATE     _IOW(KB_IOCTL_TYPE, 25, struct kb_ioctl_fence_validate)
#define KB_IOCTL_KCPU_QUEUE_CREATE  _IOR(KB_IOCTL_TYPE, 45, struct kb_ioctl_kcpu_queue_new)
#define KB_IOCTL_KCPU_QUEUE_DELETE  _IOW(KB_IOCTL_TYPE, 46, struct kb_ioctl_kcpu_queue_delete)
#define KB_IOCTL_KCPU_QUEUE_ENQUEUE _IOW(KB_IOCTL_TYPE, 47, struct kb_ioctl_kcpu_queue_enqueue)

KB_CHECK_SIZE(struct kb_fence, 8);
KB_CHECK_SIZE(struct kb_cqs_operation, 24);
KB_CHECK_OFF(struct kb_cqs_operation, operation, 16);
KB_CHECK_OFF(struct kb_cqs_operation, data_type, 17);
KB_CHECK_SIZE(struct kb_kcpu_command, 24);
KB_CHECK_OFF(struct kb_kcpu_command, info.cqs.nr_objs, 16);
KB_CHECK_OFF(struct kb_kcpu_command, info.cqs.inherit_err_flags, 20);
KB_CHECK_SIZE(struct kb_ioctl_kcpu_queue_new, 8);
KB_CHECK_SIZE(struct kb_ioctl_kcpu_queue_delete, 8);
KB_CHECK_SIZE(struct kb_ioctl_kcpu_queue_enqueue, 16);
KB_CHECK_OFF(struct kb_ioctl_kcpu_queue_enqueue, id, 12);
KB_CHECK_SIZE(struct kb_ioctl_fence_validate, 4);
/* The numbers the blob issues. */
_Static_assert(KB_IOCTL_FENCE_VALIDATE == 0x40048019, "FENCE_VALIDATE");
_Static_assert(KB_IOCTL_KCPU_QUEUE_ENQUEUE == 0x4010802f, "KCPU_QUEUE_ENQUEUE");

#endif /* MALI_KBASE_UAPI_H */
