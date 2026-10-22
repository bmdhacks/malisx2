/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The kbase kernel interface, job-manager (JM) flavour: Arm's older
 * frontend, used by the Mali-G57 (arch v9) instead of the G615's CSF. UK
 * version 11.36 (DDK r40p0, the Unisoc T820's kernel; its job-manager UAPI
 * has no difference from the r44p1 Exynos tree this driver already
 * targets on CSF).
 *
 * Licensing and evidence: as uapi.h. Our own MIT text written from the
 * r40p0 kbase UAPI headers (GPL-2.0 WITH Linux-syscall-note), never
 * copied; names use the KB_JM_/kb_jm_ prefix for anything that does not
 * already exist in uapi.h. A struct, ioctl number or flag shared with the
 * CSF side (SET_FLAGS, GET_GPUPROPS, MEM_ALLOC, MEM_EXEC_INIT,
 * MEM_JIT_INIT, MEM_IMPORT, MEM_FREE, MEM_SYNC, MEM_COMMIT,
 * MEM_FLAGS_CHANGE, STICKY_RESOURCE_MAP/UNMAP, GET_CPU_GPU_TIMEINFO,
 * FENCE_VALIDATE, the base_mem_alloc_flags bits) is the same struct and
 * number on both kernels and is reused from uapi.h rather than redefined
 * here.
 *
 * tests/test_kbase_uapi_abi_jm.c compiles this header next to the r40p0
 * kernel's own headers, built with -DMALI_USE_CSF=0, and compares sizes,
 * offsets, ioctl numbers and flag values, the same way
 * test_kbase_uapi_abi.c checks uapi.h against r44p1.
 */

#ifndef MALI_KBASE_UAPI_JM_H
#define MALI_KBASE_UAPI_JM_H

#include <stddef.h>
#include <stdint.h>
#include <linux/ioctl.h>

#include "uapi.h"

/* The JM UK version we propose: 11.36 (DDK r40p0, the Unisoc T820's
 * kernel). The kernel answers min(ours, its own) when the major matches,
 * so a kernel at or above 11.36 shows up as 11.36 and an older one as its
 * own minor. Proposing less would buy nothing on a newer kernel.
 *
 * KB_JM_UK_MIN_MINOR is the oldest minor we accept back. 11.22 is the
 * first kernel whose JOB_SUBMIT takes the 64-byte base_jd_atom that
 * struct kb_jm_atom is (11.19 to 11.21 take only 48 or 56 bytes and answer
 * EINVAL to every real submit). From 11.22 to 11.36 every ioctl we use has
 * the same number and size, so nothing inside the range branches on the
 * minor. The MediaTek 72-byte atom layout is not a function of the minor
 * and is found by the probe in mali_kbase_create, not by this check. */
#define KB_JM_UK_VERSION_MAJOR 11
#define KB_JM_UK_VERSION_MINOR 36
#define KB_JM_UK_MIN_MINOR     22

/* MEM_JIT_INIT's trim level for a job-manager context; the CSF side's
 * KB_JIT_TRIM_LEVEL (5) is unrelated. */
#define KB_JM_JIT_TRIM_LEVEL 0

/* ---------------------------------------------------------------------- */
/* Memory allocation flags that only exist, or only matter, on JM          */

/* Bits 8 and 19 are BASE_MEM_FIXED and BASE_MEM_CSF_EVENT on the CSF side
 * (uapi.h: KB_MEM_FIXED, KB_MEM_CSF_EVENT); on a job-manager kernel they
 * are plain reserved bits (BASE_MEM_RESERVED_BIT_8/_19) that a client must
 * never set. */
#define KB_MEM_JM_RESERVED (KB_MEM_FIXED | KB_MEM_CSF_EVENT)

/* BASEP_MEM_FLAGS_KERNEL_ONLY of the job-manager kernel: bits 5
 * (PERMANENT_KERNEL_MAPPING), 7 (NO_USER_FREE), 27 (FLAG_MAP_FIXED) and
 * 29 (PERFORM_JIT_TRIM). MEM_ALLOC refuses them. */
#define KB_MEM_JM_KERNEL_ONLY \
   ((1ull << 5) | (1ull << 7) | (1ull << 27) | (1ull << 29))

/* BASE_MEM_TILER_ALIGN_TOP: memory past the initial commit is aligned to
 * an `extension`-page boundary (job-manager only; the equivalent on CSF
 * memory is simply not offered). */
#define KB_MEM_TILER_ALIGN_TOP ((uint64_t)1 << 20)

/* BASE_MEM_TILER_ALIGN_TOP_EXTENSION_MAX_PAGES: 2 MiB. */
#define KB_MEM_TILER_ALIGN_TOP_EXT_MAX_PAGES (1ull << (21 - KB_PAGE_SHIFT))

/* ---------------------------------------------------------------------- */
/* The handshake and job submission                                        */

#define KB_JM_IOCTL_VERSION_CHECK _IOWR(KB_IOCTL_TYPE, 0, struct kb_ioctl_version_check)

struct kb_jm_job_submit {
   uint64_t addr;     /* user pointer to an array of struct kb_jm_atom */
   uint32_t nr_atoms;
   uint32_t stride;   /* sizeof(struct kb_jm_atom), or on a kernel with the
                         frame-number field sizeof(struct
                         kb_jm_atom_frame_nr); see below */
};

#define KB_JM_IOCTL_JOB_SUBMIT _IOW(KB_IOCTL_TYPE, 2, struct kb_jm_job_submit)
#define KB_JM_IOCTL_POST_TERM  _IO(KB_IOCTL_TYPE, 4)

struct kb_jm_soft_event_update {
   uint64_t event;      /* GPU address of the event byte */
   uint32_t new_status;
   uint32_t flags;      /* must be 0 */
};

#define KB_JM_IOCTL_SOFT_EVENT_UPDATE \
   _IOW(KB_IOCTL_TYPE, 28, struct kb_jm_soft_event_update)

/* STREAM_CREATE: the sync-file timeline a FENCE_TRIGGER soft atom's
 * base_fence.stream_fd names. Ioctl 24 is a shared number on both
 * kernels; the CSF side never issues it (its sync-file export goes
 * through the kcpu queue instead), so it is defined here rather than in
 * uapi.h. */
struct kb_jm_stream_create {
   char name[32]; /* NUL-terminated; the rest must be NUL too */
};

#define KB_JM_IOCTL_STREAM_CREATE _IOW(KB_IOCTL_TYPE, 24, struct kb_jm_stream_create)

/* struct base_dependency: one entry of an atom's pre_dep[2]. */
struct kb_jm_dep {
   uint8_t atom_id;
   uint8_t dependency_type; /* KB_JM_DEP_* */
};

#define KB_JM_DEP_INVALID 0u
#define KB_JM_DEP_DATA    (1u << 0)
#define KB_JM_DEP_ORDER   (1u << 1)

/*
 * struct base_jd_atom ("v3": base_jd_atom_v2 plus a leading seq_nr), the
 * 64-byte layout this driver builds and submits on stock kernels, with
 * every offset below checked against both the r40p0 and r44p1 kernel
 * headers' structs. kb_jm_atom_frame_nr below is the same atom as
 * MediaTek kernels with the frame-number field want it.
 */
struct kb_jm_atom {
   uint64_t seq_nr;
   uint64_t jc;            /* job chain GPU VA, or a soft job's own argument */
   uint64_t udata[2];      /* returned verbatim in the completion event */
   uint64_t extres_list;
   uint16_t nr_extres;
   uint8_t jit_id[2];      /* must be zero unless used for JIT soft jobs */
   struct kb_jm_dep pre_dep[2];
   uint8_t atom_number;    /* 1..255; 0 means "no dependency" */
   uint8_t prio;           /* KB_JM_PRIO_* */
   uint8_t device_nr;
   uint8_t jobslot;        /* only with KB_JM_REQ_JOB_SLOT */
   uint32_t core_req;      /* KB_JM_REQ_* */
   uint8_t renderpass_id;  /* incremental rendering; not used (compiled out
                              of the T820 kernel) */
   uint8_t padding[7];     /* must be zero */
};

/*
 * MediaTek's kbase builds with the job-manager variant of their GPU
 * bandwidth monitor enabled add a u32 frame_nr at the end of both atom
 * structs. base_jd_atom becomes 72 bytes (64 + 4, rounded up to its 8-byte
 * alignment) and base_jd_atom_v2 becomes 64. Those kernels accept stride
 * 64 and 72 only, and read stride 64 as base_jd_atom_v2: a stock 64-byte
 * atom submitted there is read 8 bytes off (seq_nr as jc, and so on).
 * Stock kernels refuse stride 72 with EINVAL before looking at any atom,
 * which is how mali_kbase_create tells the two apart.
 *
 * The kernel copies frame_nr into its atom and hands it, with the context
 * id and an atom serial, to the SoC's bandwidth-monitor shared memory when
 * the atom starts on job slot 0. Nothing in the job path reads it.
 */
struct kb_jm_atom_frame_nr {
   struct kb_jm_atom atom;
   uint32_t frame_nr;
   uint32_t tail;          /* the struct's trailing alignment padding; the
                              kernel copies it but does not check it */
};

#define KB_JM_PRIO_MEDIUM   0
#define KB_JM_PRIO_HIGH     1
#define KB_JM_PRIO_LOW      2
#define KB_JM_PRIO_REALTIME 3
#define KB_JM_PRIO_INVALID  255

/* base_jd_core_req bits, checked against both kernels' headers.
 */
#define KB_JM_REQ_DEP                   0u
#define KB_JM_REQ_FS                     (1u << 0)
#define KB_JM_REQ_CS                     (1u << 1)
#define KB_JM_REQ_T                      (1u << 2)
#define KB_JM_REQ_CF                     (1u << 3)
#define KB_JM_REQ_V                      (1u << 4)
#define KB_JM_REQ_EVENT_COALESCE         (1u << 5)
#define KB_JM_REQ_COHERENT_GROUP         (1u << 6)
#define KB_JM_REQ_PERMON                 (1u << 7)
#define KB_JM_REQ_EXTERNAL_RESOURCES     (1u << 8)
#define KB_JM_REQ_SOFT_JOB               (1u << 9)
#define KB_JM_REQ_SOFT_DUMP_CPU_GPU_TIME (KB_JM_REQ_SOFT_JOB | 0x1u)
#define KB_JM_REQ_SOFT_FENCE_TRIGGER     (KB_JM_REQ_SOFT_JOB | 0x2u)
#define KB_JM_REQ_SOFT_FENCE_WAIT        (KB_JM_REQ_SOFT_JOB | 0x3u)
#define KB_JM_REQ_SOFT_EVENT_WAIT        (KB_JM_REQ_SOFT_JOB | 0x5u)
#define KB_JM_REQ_SOFT_EVENT_SET         (KB_JM_REQ_SOFT_JOB | 0x6u)
#define KB_JM_REQ_SOFT_EVENT_RESET       (KB_JM_REQ_SOFT_JOB | 0x7u)
#define KB_JM_REQ_SOFT_JIT_ALLOC         (KB_JM_REQ_SOFT_JOB | 0x9u)
#define KB_JM_REQ_SOFT_JIT_FREE          (KB_JM_REQ_SOFT_JOB | 0xau)
#define KB_JM_REQ_SOFT_EXT_RES_MAP       (KB_JM_REQ_SOFT_JOB | 0xbu)
#define KB_JM_REQ_SOFT_EXT_RES_UNMAP     (KB_JM_REQ_SOFT_JOB | 0xcu)
#define KB_JM_REQ_ONLY_COMPUTE           (1u << 10)
#define KB_JM_REQ_SPECIFIC_COHERENT_GROUP (1u << 11)
#define KB_JM_REQ_EVENT_ONLY_ON_FAILURE  (1u << 12)
#define KB_JM_REQ_FS_AFBC                (1u << 13)
#define KB_JM_REQ_EVENT_NEVER            (1u << 14) /* BASEP_, kernel-visible only */
#define KB_JM_REQ_SKIP_CACHE_START       (1u << 15)
#define KB_JM_REQ_SKIP_CACHE_END         (1u << 16)
#define KB_JM_REQ_JOB_SLOT               (1u << 17)
#define KB_JM_REQ_START_RENDERPASS       (1u << 18)
#define KB_JM_REQ_END_RENDERPASS         (1u << 19)
#define KB_JM_REQ_LIMITED_CORE_MASK      (1u << 20)
#define KB_JM_REQ_SOFT_JOB_TYPE          (KB_JM_REQ_SOFT_JOB | 0x1fu)
#define KB_JM_REQ_ATOM_TYPE \
   (KB_JM_REQ_FS | KB_JM_REQ_CS | KB_JM_REQ_T | KB_JM_REQ_CF | KB_JM_REQ_V | \
    KB_JM_REQ_SOFT_JOB | KB_JM_REQ_ONLY_COMPUTE)

/* enum base_jd_event_code: the codes read() can return (not
 * exhaustive). */
#define KB_JM_EVENT_NOT_STARTED        0x00u
#define KB_JM_EVENT_DONE               0x01u
#define KB_JM_EVENT_STOPPED            0x03u
#define KB_JM_EVENT_TERMINATED         0x04u
#define KB_JM_EVENT_ACTIVE             0x08u
#define KB_JM_EVENT_JOB_CONFIG_FAULT   0x40u
#define KB_JM_EVENT_JOB_POWER_FAULT    0x41u
#define KB_JM_EVENT_JOB_READ_FAULT     0x42u
#define KB_JM_EVENT_JOB_WRITE_FAULT    0x43u
#define KB_JM_EVENT_JOB_AFFINITY_FAULT 0x44u
#define KB_JM_EVENT_JOB_BUS_FAULT      0x48u
#define KB_JM_EVENT_DATA_INVALID_FAULT 0x58u
#define KB_JM_EVENT_TILE_RANGE_FAULT   0x59u
#define KB_JM_EVENT_STATE_FAULT        0x5au
#define KB_JM_EVENT_OUT_OF_MEMORY      0x60u
#define KB_JM_EVENT_UNKNOWN            0x7fu
#define KB_JM_EVENT_DELAYED_BUS_FAULT  0x80u
#define KB_JM_EVENT_SHAREABILITY_FAULT 0x88u
#define KB_JM_EVENT_TRANSLATION_FAULT_LEVEL1 0xc1u
#define KB_JM_EVENT_PERMISSION_FAULT   0xc8u
#define KB_JM_EVENT_ACCESS_FLAG        0xd8u
#define KB_JM_EVENT_MEM_GROWTH_FAILED  0x4000u
#define KB_JM_EVENT_JOB_CANCELLED      0x4002u
#define KB_JM_EVENT_JOB_INVALID        0x4003u
#define KB_JM_EVENT_DRV_TERMINATED     0x7000u

/* struct base_jd_event_v2: what read() on the kbase fd returns, 24 bytes. */
struct kb_jm_event {
   uint32_t event_code;
   uint8_t atom_number;
   uint8_t padding[3];
   uint64_t udata[2];
};

/* ---------------------------------------------------------------------- */
/* JIT soft jobs. Our tiler-heap design is a driver-owned ring, not JIT,
 * so nothing in this layer builds these; they are defined for
 * completeness of the split and in case a later unit needs the blob's
 * scheme as a fallback. */

#define KB_JM_JIT_ALLOC_MEM_TILER_ALIGN_TOP (1u << 0)
#define KB_JM_JIT_ALLOC_HEAP_INFO_IS_SIZE   (1u << 1)
#define KB_JM_JIT_MAX_ALLOCATIONS           255

struct kb_jm_jit_alloc_info {
   uint64_t gpu_alloc_addr; /* the kernel writes the allocation's GPU VA here */
   uint64_t va_pages;
   uint64_t commit_pages;
   uint64_t extension;
   uint8_t id;              /* nonzero; pairs an alloc with its free */
   uint8_t bin_id;
   uint8_t max_allocations;
   uint8_t flags;           /* KB_JM_JIT_ALLOC_* */
   uint8_t padding[2];
   uint16_t usage_id;
   uint64_t heap_info_gpu_addr;
};

/* struct base_external_resource / BASE_EXT_RES_COUNT_MAX: external
 * resources for BASE_JD_REQ_EXTERNAL_RESOURCES and the ext-res soft jobs.
 * Not used by this driver (sticky mapping covers our dma-buf imports, as
 * on the CSF side); defined for completeness. */
struct kb_jm_external_resource {
   uint64_t ext_resource;
};
#define KB_JM_EXT_RES_COUNT_MAX 10

/* ---------------------------------------------------------------------- */
/* Layout checks (the kernel headers are the reference; see
 * test_kbase_uapi_abi_jm.c). */

KB_CHECK_SIZE(struct kb_jm_job_submit, 16);
KB_CHECK_OFF(struct kb_jm_job_submit, nr_atoms, 8);
KB_CHECK_OFF(struct kb_jm_job_submit, stride, 12);
KB_CHECK_SIZE(struct kb_jm_soft_event_update, 16);
KB_CHECK_OFF(struct kb_jm_soft_event_update, new_status, 8);
KB_CHECK_SIZE(struct kb_jm_stream_create, 32);
KB_CHECK_SIZE(struct kb_jm_dep, 2);
KB_CHECK_OFF(struct kb_jm_dep, dependency_type, 1);

KB_CHECK_SIZE(struct kb_jm_atom, 64);
KB_CHECK_OFF(struct kb_jm_atom, jc, 8);
KB_CHECK_OFF(struct kb_jm_atom, udata, 16);
KB_CHECK_OFF(struct kb_jm_atom, extres_list, 32);
KB_CHECK_OFF(struct kb_jm_atom, nr_extres, 40);
KB_CHECK_OFF(struct kb_jm_atom, jit_id, 42);
KB_CHECK_OFF(struct kb_jm_atom, pre_dep, 44);
KB_CHECK_OFF(struct kb_jm_atom, atom_number, 48);
KB_CHECK_OFF(struct kb_jm_atom, prio, 49);
KB_CHECK_OFF(struct kb_jm_atom, device_nr, 50);
KB_CHECK_OFF(struct kb_jm_atom, jobslot, 51);
KB_CHECK_OFF(struct kb_jm_atom, core_req, 52);
KB_CHECK_OFF(struct kb_jm_atom, renderpass_id, 56);
KB_CHECK_OFF(struct kb_jm_atom, padding, 57);
/* base_jd_atom_v2 is base_jd_atom from jc on: the other two accepted
 * JOB_SUBMIT strides. */
_Static_assert(sizeof(struct kb_jm_atom) - 8 == 56, "base_jd_atom_v2 size");
_Static_assert(offsetof(struct kb_jm_atom, renderpass_id) - 8 == 48,
               "base_jd_atom_v2.renderpass_id");

KB_CHECK_SIZE(struct kb_jm_atom_frame_nr, 72);
KB_CHECK_OFF(struct kb_jm_atom_frame_nr, frame_nr, 64);

KB_CHECK_SIZE(struct kb_jm_event, 24);
KB_CHECK_OFF(struct kb_jm_event, atom_number, 4);
KB_CHECK_OFF(struct kb_jm_event, udata, 8);

KB_CHECK_SIZE(struct kb_jm_jit_alloc_info, 48);
KB_CHECK_OFF(struct kb_jm_jit_alloc_info, va_pages, 8);
KB_CHECK_OFF(struct kb_jm_jit_alloc_info, commit_pages, 16);
KB_CHECK_OFF(struct kb_jm_jit_alloc_info, extension, 24);
KB_CHECK_OFF(struct kb_jm_jit_alloc_info, id, 32);
KB_CHECK_OFF(struct kb_jm_jit_alloc_info, bin_id, 33);
KB_CHECK_OFF(struct kb_jm_jit_alloc_info, max_allocations, 34);
KB_CHECK_OFF(struct kb_jm_jit_alloc_info, flags, 35);
KB_CHECK_OFF(struct kb_jm_jit_alloc_info, usage_id, 38);
KB_CHECK_OFF(struct kb_jm_jit_alloc_info, heap_info_gpu_addr, 40);

KB_CHECK_SIZE(struct kb_jm_external_resource, 8);

/* The ioctl numbers the T820 blob and our test fixtures were built
 * against, as an independent check of the encoding (uapi.h's
 * KB_CHECK_SIZE/KB_CHECK_OFF/_Static_assert pattern). */
_Static_assert(KB_JM_IOCTL_VERSION_CHECK == 0xc0048000, "JM VERSION_CHECK");
_Static_assert(KB_JM_IOCTL_JOB_SUBMIT == 0x40108002, "JOB_SUBMIT");
_Static_assert(KB_JM_IOCTL_POST_TERM == 0x8004, "POST_TERM");
_Static_assert(KB_JM_IOCTL_SOFT_EVENT_UPDATE == 0x4010801c, "SOFT_EVENT_UPDATE");
_Static_assert(KB_JM_IOCTL_STREAM_CREATE == 0x40208018, "STREAM_CREATE");

#endif /* MALI_KBASE_UAPI_JM_H */
