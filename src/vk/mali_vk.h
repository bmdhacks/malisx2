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

#ifndef __ANDROID__
/* Mesa's shared Vulkan WSI device state (src/vulkan/wsi/wsi_common.h). */
struct wsi_device;
struct gbm_device;
struct pan_model;
#endif

struct mali_physical_device {
   struct vk_physical_device vk;

   char path[32];                      /* "/dev/mali0" */
   struct mali_kbase_gpu_props props;  /* decoded GET_GPUPROPS */
   uint32_t cs_work_registers;

   /* props.arch_major, cached here so code outside the kbase layer picks
    * the per-arch device entry points (mali_CreateDevice) and internals
    * (mali_arch_dispatch) without reaching back into props. */
   uint32_t arch;

   /* Mesa's model-table entry for this GPU (core count and feature bits
    * included in the match), or, for a part the table does not list, our
    * own built from the GPU's ID (mali_physical_device.c fallback_model):
    * set for every GPU we serve. The compiler takes it with the arch:
    * Bifrost's workarounds are chosen from the model's architecture minor
    * version. */
   const struct pan_model *model;

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

#ifndef __ANDROID__
   /* The shared Vulkan WSI (Mesa's src/vulkan/wsi): VK_KHR_display's
    * surfaces, modes and swapchains, and the page flips themselves
    * (mali_wsi_linux.c). Android's window system has its own path
    * (mali_wsi.c, mali_android.c). */
   struct wsi_device *wsi;
   /* The display's DRM node, or -1, and the GBM device on it (opened on
    * first use; mali_wsi_linux.c). */
   int drm_fd;
   struct gbm_device *gbm;
#endif
};

VK_DEFINE_HANDLE_CASTS(mali_physical_device, vk.base, VkPhysicalDevice,
                       VK_OBJECT_TYPE_PHYSICAL_DEVICE)

/* vk_instance::physical_devices.enumerate / .destroy */
VkResult mali_physical_devices_enumerate(struct vk_instance *instance);
void mali_physical_device_destroy(struct vk_physical_device *pdev);

#ifndef __ANDROID__
/* The Linux window system (mali_wsi_linux.c): Mesa's shared WSI. */
VkResult mali_wsi_init(struct mali_physical_device *pdev);
void mali_wsi_finish(struct mali_physical_device *pdev);
/* A scanout buffer for a WSI image (mali_wsi_linux.c): a GBM buffer object
 * on the display's DRM node, as a dma-buf. */
VkResult mali_wsi_alloc_scanout(struct mali_physical_device *pdev, VkFormat format,
                                unsigned width, unsigned height, uint32_t *stride,
                                uint64_t *size, int *fd);
#endif

/* The deviceName for a GPU, or false if it is not a GPU this driver was
 * built for. */
bool mali_device_name(const struct mali_kbase_gpu_props *p, char *buf, size_t size);

/* Whether Mesa's Bifrost compiler targets the GPU's quad-only CLPER form
 * (arch 7 below arch minor 2: the G51 and G31), where a shader that needs
 * read_invocation cannot be compiled (mali_physical_device.c). */
bool mali_physical_device_limited_clper(const struct mali_physical_device *pdev);

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
   MALI_META_QUERY_COPY,  /* vkCmdCopyQueryPoolResults, one invocation per query */
   MALI_META_QUERY_COPY_SUM, /* the same for pools whose query is a sum of
                              * counters (v7 occlusion: one per shader core) */
   MALI_META_INDIRECT_DISPATCH, /* v7 and v9 vkCmdDispatchIndirect: patches the Compute job */
   MALI_META_INDIRECT_DRAW,     /* v7 and v9 indirect draws: patches a draw's jobs */
   MALI_META_INDEX_MINMAX,      /* v7 indirect indexed draws: the indices' range */
   MALI_META_COUNT,
};

/* Push constants of MALI_META_INDIRECT_DISPATCH (mali_jm_cmd_dispatch.c):
 * the VkDispatchIndirectCommand, the Compute job's header, the place the
 * counts go in the job, and the FAU addresses of num_work_groups.xyz (0
 * where the shader does not read one). On v7 inv is the Invocation
 * section, whose words inv0 and inv1 have the workgroup size packed and
 * the counts left out, and wgx_shift is where the counts start. On v9 inv
 * is the first of the three Workgroup count words of the Compute payload,
 * and the other three fields are not used. */
struct mali_meta_indirect_push {
   uint64_t cmd;
   uint64_t job;
   uint64_t inv;
   uint64_t nwg[3];
   uint32_t inv0;
   uint32_t inv1;
   uint32_t wgx_shift;
   uint32_t pad;
};

/*
 * One draw of a vkCmdDraw*Indirect* on the job manager (v7 and v9), for its
 * patch job (MALI_META_INDIRECT_DRAW, mali_jm_idraw_nir.c), which reads it
 * from command memory through the one 64-bit push constant it has. What
 * the command buffer knows when it records the draw: where the draw's
 * parameters and the draw count are, and where in the draw's jobs and
 * descriptors each value that comes from them goes. Addresses that are 0
 * name nothing to patch. v9 uses the fields up to attr_stride.
 */
#define MALI_IDRAW_MAX_ATTRIBS 16
struct mali_jm_idraw_rec {
   uint64_t cmd;            /* the VkDraw(Indexed)IndirectCommand */
   uint64_t count;          /* the draw count (indirect count draws), else 0 */
   uint32_t draw_index;     /* this draw's place in the call */
   uint32_t max_draws;      /* drawCount or maxDrawCount */
   uint32_t indexed;
   uint32_t index_shift;    /* log2 of the index size */
   uint64_t indices;        /* the bound index buffer, its offset applied */
   uint64_t job;            /* v9: the Malloc Vertex job; v7: the Vertex job */
   uint64_t tiler_job;      /* v7: the Tiler job, 0 without one */
   uint64_t first_vertex;   /* FAU words of the vertex shader's system values */
   uint64_t base_instance;
   uint64_t raw_vertex_offset;   /* v7 */
   /* Per-instance attributes: their offset is base + firstInstance x stride.
    * v9: in the vertex shader's driver table (32-byte Attribute
    * descriptors, offset in word 2); v7: in the draw's Attribute table
    * (8-byte descriptors, offset in word 1). */
   uint64_t attribs;
   uint32_t inst_attr_mask;
   /* v7: of those, the ones whose buffer is a 1D POT Divisor: the attribute
    * unit adds the draw's Offset start to their element too, so their
    * offset is base + (firstInstance - Offset start) x stride. */
   uint32_t inst_attr_raw_mask;
   uint32_t attr_base[MALI_IDRAW_MAX_ATTRIBS];
   uint32_t attr_stride[MALI_IDRAW_MAX_ATTRIBS];
   /* v7 */
   uint64_t minmax;         /* indexed: the smallest and largest index */
   uint64_t attrib_bufs;    /* the Attribute Buffer table, two per binding */
   uint32_t mod_mask;       /* bindings packed 1D Modulus (per vertex) */
   uint32_t pot_mask;       /* bindings packed 1D POT Divisor (per instance) */
   uint32_t pot_shift[MALI_IDRAW_MAX_ATTRIBS];   /* log2 of their divisor */
   uint64_t vary_bufs;      /* the varying Attribute Buffers */
   uint32_t vary_stride[3];
   uint32_t vary_max[3];    /* the most vertices that fit the arena at that stride */
   uint64_t arena;          /* the varying arena: header, then arena_size bytes */
   uint32_t arena_size;
   uint32_t prim_verts;     /* vertices per primitive: 1, 2 or 3 */
   uint32_t inv1;           /* Invocation word 1 without the Workgroups Z shift */
   uint32_t writes_psiz;
};

/* The header of a v7 command buffer's varying arena (mali_jm_cmd_draw_v7.c):
 * the bytes handed out so far, and the draws that did not fit. */
struct mali_jm_idraw_arena_hdr {
   uint32_t used;
   uint32_t dropped;
};
#define MALI_IDRAW_ARENA_HDR 64

/* Push constants of MALI_META_INDEX_MINMAX: the smallest and largest of a
 * VkDrawIndexedIndirectCommand's indices into result[0] and result[1]
 * (atomic min and max), restart indices left out when restart is set. elems
 * is the bound index buffer's size in indices; each invocation reads
 * per_thread of them. */
struct mali_meta_minmax_push {
   uint64_t indices;
   uint64_t cmd;
   uint64_t result;
   uint32_t elems;
   uint32_t per_thread;
   uint32_t index_shift;
   uint32_t restart;
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

   /* VkEvent words (mali_event.h), host-visible; set up by
    * mali_event_pool_init. */
   struct mali_bo_pool event_pool;
};

VK_DEFINE_HANDLE_CASTS(mali_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)

/* The event pool (mali_event.c): set up after the kbase context
 * exists, torn down before it goes. Slabs are allocated by the first
 * event. */
void mali_event_pool_init(struct mali_device *dev);
void mali_event_pool_finish(struct mali_device *dev);

/*
 * The job-manager back half (v7 and v9), as code built once (mali_device.c,
 * mali_physical_device.c) reaches it: device and queue setup
 * (mali_jm_queue.c), command buffers (mali_jm_cmd_buffer.c) and the sync
 * type (mali_sync.c built at v7 and v9). The command-stream (v10, v11, v12)
 * counterparts are in mali_queue.h and below.
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
extern const struct vk_sync_type mali_v10_sync_type;
extern const struct vk_sync_type mali_v11_sync_type;
extern const struct vk_sync_type mali_v12_sync_type;

/* Each arch's sync types, binary first and then the timeline semaphore type
 * wrapped around it (mali_sync.c); the physical device picks by arch. */
const struct vk_sync_type *const *mali_v7_sync_types(void);
const struct vk_sync_type *const *mali_v9_sync_types(void);
const struct vk_sync_type *const *mali_v10_sync_types(void);
const struct vk_sync_type *const *mali_v11_sync_types(void);
const struct vk_sync_type *const *mali_v12_sync_types(void);

/* Log a failed sync-file import or export (mali_sync.c, every arch; the
 * per-arch headers declare each build's own). */
MALI_PER_ARCH_DECL(void, sync_file_log_error,
                   (struct mali_device *dev, const char *op, int fd, const char *why));

/* The command-stream back half's command buffers (mali_cmd_buffer.c, built
 * at v10, v11 and v12). */
extern const struct vk_command_buffer_ops mali_v10_cmd_buffer_ops;
extern const struct vk_command_buffer_ops mali_v11_cmd_buffer_ops;
extern const struct vk_command_buffer_ops mali_v12_cmd_buffer_ops;

/* Report device loss to the runtime and the log (once), then mark it and
 * wake every waiter. Takes dev->lock; call without it. Every back half
 * calls it (mali_device.c). */
void mali_device_set_lost(struct mali_device *dev, const char *fmt, ...)
   __attribute__((format(printf, 2, 3)));

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

/* malisx2's own command, handed out by vkGetDeviceProcAddr on v7
 * (mali_device.c): the index range of the next indexed draw, which must
 * cover all its indices but the restart index (mali_jm_cmd_draw_v7.c). */
typedef void(VKAPI_PTR *PFN_vkCmdSetIndexRangeMALISX2)(VkCommandBuffer commandBuffer,
                                                      uint32_t firstIndex, uint32_t indexCount,
                                                      uint32_t minIndex, uint32_t maxIndex);
VKAPI_ATTR void VKAPI_CALL mali_v7_CmdSetIndexRangeMALISX2(VkCommandBuffer commandBuffer,
                                                           uint32_t firstIndex,
                                                           uint32_t indexCount,
                                                           uint32_t minIndex, uint32_t maxIndex);

/*
 * malisx2's own command on v7, from vkGetDeviceProcAddr: what happens to
 * the fragments of the draws recorded after it, in this command buffer,
 * whose depth falls outside the viewport's depth range. With
 * VK_DEPTH_CULL_OVERRIDE_BIT_MALISX2 set, those below minDepth are
 * discarded when VK_DEPTH_CULL_BELOW_BIT_MALISX2 is set and clamped to it
 * when not, and those above maxDepth likewise with
 * VK_DEPTH_CULL_ABOVE_BIT_MALISX2 (which keeps up to 1e-5 past maxDepth:
 * float rounding puts geometry modelled on the far plane a ULP or two past
 * it); the pipeline's depthClampEnable and depth clip state no longer
 * decide it. 0 returns to the pipeline's state. A command buffer starts at
 * 0.
 *
 * This is the near and far handling of the fixed-function tiler (the
 * Primitive's low and high depth cull), which Vulkan only offers coupled
 * (depth clamp or depth clip on both sides). An emulator that needs
 * fragments in front of the near plane clipped but those behind the far
 * plane clamped otherwise writes clip distances, which v7 can only apply as
 * a fragment discard (mali_shader.c lower_clip_distance_fs): that turns off
 * forward pixel kill and early depth updates for every draw.
 */
enum {
   VK_DEPTH_CULL_OVERRIDE_BIT_MALISX2 = 1u << 0,
   VK_DEPTH_CULL_BELOW_BIT_MALISX2 = 1u << 1,
   VK_DEPTH_CULL_ABOVE_BIT_MALISX2 = 1u << 2,
};
typedef void(VKAPI_PTR *PFN_vkCmdSetDepthCullMALISX2)(VkCommandBuffer commandBuffer,
                                                     uint32_t flags);
VKAPI_ATTR void VKAPI_CALL mali_v7_CmdSetDepthCullMALISX2(VkCommandBuffer commandBuffer,
                                                         uint32_t flags);

/*
 * malisx2's own device extension on v7, in place of
 * VK_EXT_descriptor_indexing (whose required features and 500000
 * update-after-bind limits v7's 16-bit descriptor tables cannot meet). v9
 * and v11 offer VK_EXT_descriptor_indexing itself (mali_physical_device.c)
 * and not this one; v7 offers this one and not the EXT.
 *
 * - With it offered, VkPhysicalDeviceDescriptorIndexingFeatures and
 *   VkPhysicalDeviceDescriptorIndexingProperties may be chained into
 *   vkGetPhysicalDeviceFeatures2 and vkGetPhysicalDeviceProperties2. They
 *   report the part v7 has (runtime arrays, partially bound bindings,
 *   sampled images updated after bind, non-uniform sampled-image indexing)
 *   and its own update-after-bind limits.
 * - With it enabled, an application may use those features (in
 *   VkPhysicalDeviceDescriptorIndexingFeatures at vkCreateDevice too),
 *   VkDescriptorSetLayoutBindingFlagsCreateInfo, the update-after-bind
 *   layout and pool flags, and SPV_EXT_descriptor_indexing's capabilities
 *   for them, as VK_EXT_descriptor_indexing defines them, within those
 *   limits.
 * - One deviation: a non-uniform index into a sampled-image or sampler
 *   array must be the same for the four invocations of each 2x2 fragment
 *   quad, and dynamically uniform in other stages. That holds for every
 *   operation of the texture unit, texelFetch, explicit level of detail and
 *   gather included, so it is wider than what the reported
 *   quadDivergentImplicitLod (false) excuses, which is only implicit level
 *   of detail. (hakuX indexes per draw, which a quad never straddles.)
 * - A stage can use at most 240 storage images, storage texel buffers and
 *   uniform texel buffers together (the image table), whatever
 *   maxPerStageDescriptorUpdateAfterBindSampledImages says of the last
 *   kind; pipeline creation fails with VK_ERROR_FEATURE_NOT_PRESENT past
 *   that.
 */
#define VK_MALISX2_DESCRIPTOR_INDEXING_EXTENSION_NAME "VK_MALISX2_descriptor_indexing"
#define VK_MALISX2_DESCRIPTOR_INDEXING_SPEC_VERSION 1

/* Whether descriptor indexing is offered to applications under malisx2's own
 * name (v7) and not VK_EXT_descriptor_indexing's (v9, v11). Mesa's runtime
 * keys the feature structs on the EXT, so it is supported to the runtime
 * either way. */
static inline bool
mali_descriptor_indexing_renamed(uint32_t arch)
{
   return arch == 7;
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
