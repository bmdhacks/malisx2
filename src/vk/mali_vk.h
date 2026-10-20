/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The Vulkan front half: instance, physical device, device and queue, built
 * on Mesa's Vulkan runtime.
 */

#ifndef MALI_VK_H
#define MALI_VK_H

#include <pthread.h>
#include <stdint.h>

#include "util/list.h"
#include "util/simple_mtx.h"

#include "vk_device.h"
#include "vk_instance.h"
#include "vk_log.h"
#include "vk_physical_device.h"
#include "vk_queue.h"

#include "kbase/kbase.h"

#include "mali_arch.h"
#include "mali_bo_pool.h"

#include "mali_entrypoints.h"

/* Highest number of physical devices (kbase nodes /dev/mali0..N-1) we look
 * at; the blob creates at most four. */
#define MALI_MAX_PHYSICAL_DEVICES 4

/* The Vulkan version we implement: ARMSX2 needs 1.1 and uses nothing
 * newer. */
#define MALI_API_VERSION VK_MAKE_API_VERSION(0, 1, 1, VK_HEADER_VERSION)

/* One queue family, one queue. */
#define MALI_QUEUE_COUNT 1

struct mali_instance {
   struct vk_instance vk;
   /* How the kbase layer reaches the kernel; the OS backend except in
    * tests (mali_instance_set_kbase_backend). */
   const struct mali_kbase_backend *kbase_backend;
};

VK_DEFINE_HANDLE_CASTS(mali_instance, vk.base, VkInstance, VK_OBJECT_TYPE_INSTANCE)

/*
 * Tests only: kbase backend for instances created after this call (NULL
 * restores the OS backend). Not exported from the driver .so.
 */
void mali_instance_set_kbase_backend(const struct mali_kbase_backend *backend);

struct mali_physical_device {
   struct vk_physical_device vk;

   char path[32];                      /* "/dev/mali0" */
   struct mali_kbase_gpu_props props;  /* decoded GET_GPUPROPS */
   uint32_t cs_work_registers;

   /* props.arch_major, cached here so code outside the kbase layer picks
    * the per-arch device entry points (mali_CreateDevice) and internals
    * (mali_arch_dispatch) without reaching back into props. */
   uint32_t arch;

   VkPhysicalDeviceMemoryProperties memory;
   uint64_t timestamp_hz;              /* 0: no timestamps */

   /* VK_EXT_memory_budget's heapUsage: bytes in live VkDeviceMemory
    * allocations on this physical device, summed across every VkDevice.
    * Atomic: mali_AllocateMemory/mali_FreeMemory touch it without the
    * device lock. */
   uint64_t heap_used;
   /* The system's available memory for heapBudget, read from the kernel
    * at most once a second (mali_physical_device.c), and when it was read. */
   uint64_t avail_mem;
   uint64_t avail_mem_ns;
};

VK_DEFINE_HANDLE_CASTS(mali_physical_device, vk.base, VkPhysicalDevice,
                       VK_OBJECT_TYPE_PHYSICAL_DEVICE)

/* vk_instance::physical_devices.enumerate / .destroy */
VkResult mali_physical_devices_enumerate(struct vk_instance *instance);
void mali_physical_device_destroy(struct vk_physical_device *pdev);

/* The deviceName for a GPU, or false if it is not a GPU this driver was
 * built for. */
bool mali_device_name(const struct mali_kbase_gpu_props *p, char *buf, size_t size);

/* Log sink for the kbase layer: kbase messages go to the Vulkan log. */
void mali_kbase_log_to_vk(void *user, const char *msg);

struct mali_csf_queue;
struct mali_csf_device;
struct mali_jm_queue;
struct mali_jm_device;
struct mali_measure;
struct mali_shader;

/*
 * Measurement (timing.c, shared by both frontends): has submission seq
 * completed on jd's queue? Defined in mali_jm_queue.c (v7 and v9); called
 * from timing.c only when dev->jm is non-NULL, so the CSF build never
 * calls it. jd is never dereferenced outside JM files, so this needs no
 * frontend-specific type here.
 */
MALI_PER_JM_ARCH_DECL(bool, jm_measure_reached, (struct mali_jm_device *jd, uint64_t seq));

/* Internal compute/fragment shaders for copies, fills and blits
 * (mali_cmd_copy.c, mali_cmd_meta_gfx.c). Frontend-neutral: every GPU
 * architecture needs the same internal shaders, compiled on first use and
 * cached for the device's lifetime. */
enum mali_meta_shader {
   MALI_META_COPY_16,     /* 16 bytes per invocation */
   MALI_META_COPY_4,
   MALI_META_COPY_1,
   MALI_META_FILL_16,
   MALI_META_FILL_4,
   MALI_META_COUNT,
};

/* A fault as the kernel described it, for vkGetDeviceFaultInfoEXT. code
 * and data become vendorFaultCode and vendorFaultData; address is a GPU
 * virtual address when address_type is not NONE. */
struct mali_device_fault {
   bool valid;
   VkDeviceFaultAddressTypeEXT address_type;
   uint64_t address;
   uint64_t code;
   uint64_t data;
   char what[VK_MAX_DESCRIPTION_SIZE];
};

struct mali_queue {
   struct vk_queue vk;
   /* Queue group, rings, submit state (mali_queue.h). */
   struct mali_csf_queue *csf;
   /* Job-manager submit state (mali_jm.h), on a v7 or v9 device instead. */
   struct mali_jm_queue *jm;
   /* Same object as csf or jm above, untyped: lets frontend-neutral code
    * hold a queue pointer without the frontend's type. Set by the
    * frontend's queue-init path. */
   void *fe;
};

VK_DEFINE_HANDLE_CASTS(mali_queue, vk.base, VkQueue, VK_OBJECT_TYPE_QUEUE)

struct mali_device {
   struct vk_device vk;

   /* This device's kbase context (a VkDevice owns one, as in the blob). */
   struct mali_kbase *kbase;

   struct mali_queue queues[MALI_QUEUE_COUNT];
   uint32_t queue_count;

   /* Shader memory: code in the executable zone, SPDs and other pipeline
    * descriptors. */
   struct mali_bo_pool exec_pool;
   struct mali_bo_pool desc_pool;

   /*
    * Device state every GPU architecture needs, regardless of frontend
    * (CSF or JM): command-buffer memory, internal shaders, the sync
    * waiter lock and device-loss state, and submit statistics. This used
    * to live in struct mali_csf_device; it moved here because none of it
    * is CSF-specific (the JM backend will need the same bookkeeping).
    * Initialized by mali_device.c before the frontend's own device-init
    * runs, so it exists under either frontend.
    */

   /* Command-buffer memory: free slabs (64 KiB, and on v7 also 1 MiB),
    * shared by every command pool of the device. */
   simple_mtx_t slab_lock;
   struct list_head free_slabs;
   unsigned free_slab_count;

   /* Internal shaders, compiled on first use. */
   simple_mtx_t meta_lock;
   struct mali_shader *meta[MALI_META_COUNT];
   /* Internal fragment shaders (mali_cmd_meta_gfx.c), by key. */
   void *meta_gfx;
   /* Blend shaders (mali_blend.c), by key. Under meta_lock. */
   void *blend_shaders;

   /* Guards every mali_sync's state and the device's loss state; waiters
    * sleep on cond, which submits, host signals and kernel (or, on JM,
    * on-demand) events broadcast. */
   pthread_mutex_t lock;
   pthread_cond_t cond;
   bool lost;
   uint32_t lost_reported;        /* vk_device_set_lost called (atomic) */
   /* What vkGetDeviceFaultInfoEXT reports, under lock: the message of the
    * first device loss and the first fault the kernel described. */
   char lost_msg[256];
   struct mali_device_fault fault;

   struct {
      uint64_t kernel_events;       /* EVENT notifications read */
      uint64_t group_errors;
      uint64_t waits;               /* host waits that had to sleep */
      uint64_t wakeups;             /* condition wake-ups while waiting */
      uint64_t submits;
   } stats;

   /* How many of each diagnostic the log has been offered
    * (mali_diag_should_log), atomics. */
   struct {
      uint32_t slow_waits;          /* host or submit waits over 1 s */
      uint32_t sync_file_errors;    /* sync-file import/export failures */
   } diag;

   /* Command-stream state specific to the device's frontend: struct
    * mali_csf_device (CSF, v11) today; struct mali_jm_device (JM, v9)
    * once that frontend exists. Scoreboard facts, the event thread, the
    * queue group and the kcpu queue stay inside it (mali_queue.h). */
   struct mali_csf_device *csf;
   /* The job-manager state on a v7 or v9 device (mali_jm.h): atom numbers,
    * on-demand event reading, the sync-file stream. */
   struct mali_jm_device *jm;
   /* Same object as csf or jm above, untyped: lets frontend-neutral code
    * hold a device pointer without the frontend's type. */
   void *fe;

   /* Timing and command-stream capture (src/measurement/); NULL unless
    * MALISX2_MEASURE or debug.malisx2.measure turns them on. */
   struct mali_measure *measure;

   /* Interned byte strings (mali_device_intern_key): small ids that
    * pipelines compare at bind instead of the data behind them. */
   struct {
      simple_mtx_t lock;
      struct hash_table *table;
      uint32_t count;
   } keys;

   /* Shader compiles and pipeline cache hits since device creation. */
   struct {
      uint32_t shaders_compiled;
      uint32_t shader_cache_hits;
   } pipeline_stats;
};

VK_DEFINE_HANDLE_CASTS(mali_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)

/*
 * The job-manager back half (v7 and v9), as code built once (mali_device.c,
 * mali_physical_device.c) reaches it: device and queue setup
 * (mali_jm_queue.c), command buffers (mali_jm_cmd_buffer.c) and the sync
 * type (mali_sync.c built at v7 and v9). The v11 counterparts are in
 * mali_queue.h and mali_cmd_buffer.h.
 */
struct vk_command_buffer_ops;
struct vk_queue_submit;
struct vk_sync_type;
MALI_PER_JM_ARCH_DECL(VkResult, device_init, (struct mali_device *dev));
MALI_PER_JM_ARCH_DECL(void, device_finish, (struct mali_device *dev));
MALI_PER_JM_ARCH_DECL(void, device_quiesce, (struct mali_device *dev));
MALI_PER_JM_ARCH_DECL(VkResult, queue_init, (struct mali_device *dev, struct mali_queue *queue));
MALI_PER_JM_ARCH_DECL(void, queue_finish, (struct mali_device *dev, struct mali_queue *queue));
MALI_PER_JM_ARCH_DECL(VkResult, queue_submit,
                      (struct vk_queue *vkq, struct vk_queue_submit *submit));
extern const struct vk_command_buffer_ops mali_v7_cmd_buffer_ops;
extern const struct vk_command_buffer_ops mali_v9_cmd_buffer_ops;
extern const struct vk_sync_type mali_v7_sync_type;
extern const struct vk_sync_type mali_v9_sync_type;
extern const struct vk_sync_type mali_v11_sync_type;

static inline struct mali_physical_device *
mali_device_physical(struct mali_device *dev)
{
   return container_of(dev->vk.physical, struct mali_physical_device, vk);
}

/* With dev->lock held: keep f unless a fault is kept already. */
static inline void
mali_device_record_fault(struct mali_device *dev, const struct mali_device_fault *f)
{
   if (!dev->fault.valid) {
      dev->fault = *f;
      dev->fault.valid = true;
   }
}

/*
 * Diagnostics that go to the log (logcat tag MESA on Android) whatever
 * the build: a wait that blocks for MALI_SLOW_WAIT_NS or more, and each
 * sync-file import or export that fails. Each kind is counted in
 * mali_device::diag; the first 8 of a kind are logged, then every 100th,
 * so something that repeats every frame does not flood the log. Returns
 * whether to log this one; *n is its number (from 1).
 */
#define MALI_SLOW_WAIT_NS (1000ll * 1000 * 1000)

static inline bool
mali_diag_should_log(uint32_t *counter, uint32_t *n)
{
   const uint32_t i = __atomic_fetch_add(counter, 1, __ATOMIC_RELAXED);
   *n = i + 1;
   return i < 8 || (i + 1) % 100 == 0;
}

#if defined(PAN_ARCH) && PAN_ARCH == 7
#include "util/log.h"

/*
 * A path v7 does not support: logged once per call site, after which the
 * caller skips that work. The command buffer records no error for it: an
 * error recorded there is what vkEndCommandBuffer returns, and that may not
 * be VK_ERROR_FEATURE_NOT_PRESENT.
 */
#define MALI_V7_UNSUPPORTED(what)                                              \
   do {                                                                        \
      static bool mali_v7_unsupported_said_;                                   \
      if (!mali_v7_unsupported_said_) {                                        \
         mali_v7_unsupported_said_ = true;                                     \
         mesa_logw("malisx2: %s: %s is not supported on v7", __func__, what);  \
      }                                                                        \
   } while (0)
#endif

#endif
