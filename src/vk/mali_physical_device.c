/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * VkPhysicalDevice: one per kbase device node. What we report matches
 * the blob's values, including the memory types.
 */

#include "mali_descriptor_set_layout.h"
#include "mali_image.h"
#include "mali_queue.h"
#include "mali_vk.h"
#include "mali_shader.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "util/log.h"
#include "util/macros.h"
#include "util/mesa-blake3.h"
#include "util/u_atomic.h"
#include "util/u_math.h"
#include "util/os_misc.h"
#include "util/os_time.h"
#include "model/pan_model.h"
#include "vk_alloc.h"
#include "vk_limits.h"
#include "vk_util.h"

#include "mali_version.h"

#ifndef __ANDROID__
#include <link.h>

/* The shared WSI's physical-device entry points (VK_KHR_display). */
#include "wsi_common.h"
#endif
#include "mali_shader_cache_id.h"

/* Arm's PCI-style vendor ID, as the blob reports it. */
#define MALI_VENDOR_ID 0x13b5

/* The r44p1 blob's driverVersion, 44.1.0 in VK_MAKE_VERSION layout.
 * ARMSX2's Arm driver rules match on exactly this. */
#define MALI_DRIVER_VERSION VK_MAKE_VERSION(44, 1, 0)

/* driverInfo keeps the blob's "v1.r44p1-" prefix (ARMSX2 looks for
 * "r44p1"; the blob's own version parse still yields 44.1) and then says
 * what we are. It ends in the shader cache ID, not the commit: ARMSX2 keys
 * its pipeline cache file to driverInfo as well as pipelineCacheUUID, so a
 * commit here would empty the cache on every driver update. The commit is
 * logged when the physical device is created instead. */
#define MALI_DRIVER_INFO \
   "v1.r44p1-malisx2." MALI_VERSION_STRING ".s" MALI_SHADER_CACHE_ID_SHORT

/*
 * What identity the G57 build (arch 9) reports to ARMSX2: the
 * working choice is to report the same scheme the G615 build uses
 * above, not the T820 blob's own values (driverVersion 40.0.0,
 * driverInfo "v1.r40p0-01eac0.<hash>"). Either way driverID is 9
 * (VK_DRIVER_ID_ARM_PROPRIETARY), so ARMSX2 treats both the same
 * way for that check. Switching to the blob's own values instead
 * of the G615 scheme is a one-line change to these two
 * definitions.
 */
#define MALI_V9_DRIVER_VERSION MALI_DRIVER_VERSION
/* The same scheme, ending in the v9 build's own shader cache ID. */
#define MALI_V9_DRIVER_INFO \
   "v1.r44p1-malisx2." MALI_VERSION_STRING ".s" MALI_SHADER_CACHE_ID_V9_SHORT
/* The G52 build (arch 7): the same scheme, ending in the v7 build's own
 * shader cache ID. */
#define MALI_V7_DRIVER_VERSION MALI_DRIVER_VERSION
#define MALI_V7_DRIVER_INFO \
   "v1.r44p1-malisx2." MALI_VERSION_STRING ".s" MALI_SHADER_CACHE_ID_V7_SHORT
/* The G610 and G710 (arch 10): the G615's scheme too. Arm's own drivers for
 * these parts report the same driverID and their DDK release in the same
 * fields, so the scheme says nothing an Arm driver could not. */
#define MALI_V10_DRIVER_VERSION MALI_DRIVER_VERSION
#define MALI_V10_DRIVER_INFO \
   "v1.r44p1-malisx2." MALI_VERSION_STRING ".s" MALI_SHADER_CACHE_ID_V10_SHORT

static inline const char *
driver_info_for_arch(uint32_t arch)
{
   return arch == 7    ? MALI_V7_DRIVER_INFO
          : arch == 9  ? MALI_V9_DRIVER_INFO
          : arch == 10 ? MALI_V10_DRIVER_INFO
                       : MALI_DRIVER_INFO;
}

static inline uint32_t
driver_version_for_arch(uint32_t arch)
{
   return arch == 7    ? MALI_V7_DRIVER_VERSION
          : arch == 9  ? MALI_V9_DRIVER_VERSION
          : arch == 10 ? MALI_V10_DRIVER_VERSION
                       : MALI_DRIVER_VERSION;
}

/* VkFence and binary VkSemaphore, then the timeline semaphore type built on
 * them (mali_sync.c, built once per arch; MALI_PER_ARCH(sync_types)() also
 * fills in the point type both share). */

void
mali_kbase_log_to_vk(void *user, const char *msg)
{
   mesa_logw("malisx2: kbase: %s", msg);
}

/* ---------------------------------------------------------------------- */
/* Identity                                                                */

/*
 * Arch 9 (first-generation Valhall, job manager). The product major in
 * product_id picks the part, and the names are the ones Arm's userspace
 * drivers print:
 *
 *   0x9000  Mali-G77; the r32p1 driver (the only one we have for G77
 *           silicon) calls it G57 instead when the GPU has fewer than 7
 *           shader cores, and appends " MC<cores>" either way
 *   0x9001  Mali-G57
 *   0x9002  Mali-G78
 *   0x9003  Mali-G57 (the second G57 product ID; each Arm driver build
 *           names only one of 0x9001 and 0x9003)
 *   0x9004  Mali-G68
 *   0x9005  Mali-G78AE
 *
 * The G57 IDs get the bare "Mali-G57" that the Arm driver on the Unisoc
 * T820 prints, and G78, G68 and G78AE are bare in every Arm driver we
 * know of. An arch 9 product that is not listed is named as 0x9000 is:
 * every arch 9 part is a first-generation Valhall, and ARMSX2 takes the
 * family and the core count from the name.
 */
static void
v9_device_name(const struct mali_kbase_gpu_props *p, char *buf, size_t size)
{
   switch (p->product_id & 0xf00f) {
   case 0x9001:
   case 0x9003:
      snprintf(buf, size, "Mali-G57");
      return;
   case 0x9002:
      snprintf(buf, size, "Mali-G78");
      return;
   case 0x9004:
      snprintf(buf, size, "Mali-G68");
      return;
   case 0x9005:
      snprintf(buf, size, "Mali-G78AE");
      return;
   case 0x9000:
      break;
   default:
      mesa_logw("malisx2: unknown arch 9 product 0x%04x, naming it by core count",
                p->product_id);
      break;
   }

   /* No shader cores reported (a property stream without the register):
    * there is no count to choose a family by. */
   if (!p->core_count) {
      snprintf(buf, size, "Mali-G77");
      return;
   }
   snprintf(buf, size, "Mali-%s MC%u", p->core_count < 7 ? "G57" : "G77", p->core_count);
}

/*
 * Arch 7 (Bifrost, job manager). The product major in product_id picks the
 * part, as kbase's product table has them:
 *
 *   0x7000  Mali-G51 (TSIX)
 *   0x7001  Mali-G76 (TNOX)
 *   0x7002  Mali-G52 (TGOX; arch minor 2, and the G52 r1 at arch minor 4)
 *   0x7003  Mali-G31 (TDVX)
 *
 * Bare names, as for the G52 until now. An arch 7 product kbase does not
 * list is served too and named "Mali-G52"; its per-part values come from
 * fallback_model.
 */
static void
v7_device_name(const struct mali_kbase_gpu_props *p, char *buf, size_t size)
{
   const char *model;
   switch (p->product_id & 0xf00f) {
   case 0x7000:
      model = "G51";
      break;
   case 0x7001:
      model = "G76";
      break;
   case 0x7002:
      model = "G52";
      break;
   case 0x7003:
      model = "G31";
      break;
   default:
      mesa_logw("malisx2: unknown arch 7 product 0x%04x, naming it Mali-G52", p->product_id);
      model = "G52";
      break;
   }
   snprintf(buf, size, "Mali-%s", model);
}

/*
 * Arch 10 (command-stream Valhall). The product major in product_id picks
 * the part, as kbase's product table has them:
 *
 *   0xa002  Mali-G710 (TODX)
 *   0xa003  Mali-G510 (TGRX)
 *   0xa004  Mali-G310 (TVAX; Mesa's model table has it as
 *           PAN_PROD_ID(10, 12, 4), one entry per core variant)
 *   0xa007  Mali-G610 (LODX, the RK3588's; PAN_PROD_ID(10, 8, 7))
 *
 * Named like the G615 below, with the core count. An arch 10 product kbase
 * does not list is named as the G610 and G710 are split: G610 below seven
 * cores, G710 from seven.
 */
static void
v10_device_name(const struct mali_kbase_gpu_props *p, char *buf, size_t size)
{
   const char *model;
   switch (p->product_id & 0xf00f) {
   case 0xa007:
      model = "G610";
      break;
   case 0xa002:
      model = "G710";
      break;
   case 0xa003:
      model = "G510";
      break;
   case 0xa004:
      model = "G310";
      break;
   default:
      mesa_logw("malisx2: unknown arch 10 product 0x%04x, naming it by core count",
                p->product_id);
      model = p->core_count < 7 ? "G610" : "G710";
      break;
   }
   snprintf(buf, size, "Mali-%s MC%u", model, p->core_count);
}

/*
 * Arch 11. The blob's naming: product 0xb003 (LTUX) is the G615; 0xb002
 * (TTUX) is the G715, named G615 when it has fewer than 7 cores. The blob's
 * "G715-Immortalis" variant is not reproduced. An arch 11 product kbase does
 * not list is named as 0xb002 is.
 */
static void
v11_device_name(const struct mali_kbase_gpu_props *p, char *buf, size_t size)
{
   const unsigned product = p->product_id & 0xf00f;
   if (product != 0xb002 && product != 0xb003)
      mesa_logw("malisx2: unknown arch 11 product 0x%04x, naming it by core count",
                p->product_id);
   const char *model = product == 0xb003 || p->core_count < 7 ? "G615" : "G715";
   snprintf(buf, size, "Mali-%s MC%u", model, p->core_count);
}

/* Every product of the arches built in (7, 9, 10, 11) has a name; any other
 * arch (Midgard, the arch 6 G71 and G72, arch 12 on) has no back half here
 * and gets none. */
bool
mali_device_name(const struct mali_kbase_gpu_props *p, char *buf, size_t size)
{
   switch (p->arch_major) {
   case 7:
      v7_device_name(p, buf, size);
      return true;
   case 9:
      v9_device_name(p, buf, size);
      return true;
   case 10:
      v10_device_name(p, buf, size);
      return true;
   case 11:
      v11_device_name(p, buf, size);
      return true;
   default:
      return false;
   }
}

/* ---------------------------------------------------------------------- */
/* Timestamps                                                              */

/*
 * GPU timestamps count the Arm generic timer (the blob takes 13 MHz
 * from its platform config on the RG 477V); CNTFRQ_EL0 gives the same
 * frequency without a config store.
 */
static uint64_t
timestamp_frequency(void)
{
#if defined(__aarch64__)
   uint64_t hz;
   __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(hz));
   return hz;
#else
   return 0;
#endif
}

/* ---------------------------------------------------------------------- */
/* Memory                                                                  */

/*
 * One DEVICE_LOCAL heap whose size is the smaller of the memory kbase
 * reports and 2^VA bits, at least 1 GiB; three types (uncached coherent,
 * cached, lazily allocated). The cached type is host-coherent only when
 * the kernel runs the GPU in full ACE coherency (coherency mode 1). No
 * protected type.
 */
static uint64_t
heap_size(const struct mali_kbase_gpu_props *p)
{
   uint64_t size = p->gpu_available_memory_size;
   if (p->va_bits && p->va_bits < 64)
      size = MIN2(size, 1ull << p->va_bits);
   return MAX2(size, 1ull << 30);
}

static void
init_memory(struct mali_physical_device *pdev)
{
   const struct mali_kbase_gpu_props *p = &pdev->props;
   VkPhysicalDeviceMemoryProperties *m = &pdev->memory;
   const bool ace = p->coherency_mode == 1;

   m->memoryHeapCount = 1;
   m->memoryHeaps[0] = (VkMemoryHeap) {
      .size = heap_size(p),
      .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT,
   };

   m->memoryTypeCount = 3;
   m->memoryTypes[0] = (VkMemoryType) {
      .propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      .heapIndex = 0,
   };
   m->memoryTypes[1] = (VkMemoryType) {
      .propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
                       (ace ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT : 0),
      .heapIndex = 0,
   };
   m->memoryTypes[2] = (VkMemoryType) {
      .propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                       VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT,
      .heapIndex = 0,
   };
}

/*
 * Available system memory for VK_EXT_memory_budget. VMA asks for the budget
 * once a frame, and reading /proc/meminfo each time costs several percent
 * of a light frame, so the value is kept for a second. Two threads may both
 * refresh it; either result is fine.
 */
static uint64_t
available_memory(struct mali_physical_device *pdev)
{
   const uint64_t now = os_time_get_nano();
   const uint64_t then = p_atomic_read(&pdev->avail_mem_ns);
   if (then && now - then < 1000000000ull)
      return p_atomic_read(&pdev->avail_mem);

   /* If it cannot be read, report the whole heap rather than a budget
    * no larger than our own usage. */
   uint64_t avail;
   if (!os_get_available_system_memory(&avail))
      avail = pdev->memory.memoryHeaps[0].size;
   p_atomic_set(&pdev->avail_mem, avail);
   p_atomic_set(&pdev->avail_mem_ns, now);
   return avail;
}

/* ---------------------------------------------------------------------- */
/* Features, properties, extensions                                        */

/*
 * Only what ARMSX2 and the Android loader need.
 * VK_KHR_driver_properties is what makes ARMSX2 read driverID/driverInfo.
 */
static const struct vk_device_extension_table mali_device_extensions = {
   .KHR_driver_properties = true,
   /* Sync-file import and export: the Android loader signals present
    * fences by importing the release fence (swapchain_maintenance1). */
   .KHR_external_fence_fd = true,
   .KHR_external_semaphore_fd = true,
   /* ARMSX2 logs the fault records after VK_ERROR_DEVICE_LOST. */
   .EXT_device_fault = true,
#ifdef VK_USE_PLATFORM_ANDROID_KHR
   /* Without it the Android loader builds no swapchain. */
   .ANDROID_native_buffer = true,
   /* Buffers' memory from BLOB hardware buffers (mali_android.c): an
    * emulator's guest RAM read in place. Images are not importable. */
   .ANDROID_external_memory_android_hardware_buffer = true,
   .EXT_queue_family_foreign = true,
#else
   /* The Linux window system: Mesa's WSI owns the swapchain for a
    * VK_KHR_display surface (mali_wsi_linux.c). */
   .KHR_swapchain = true,
   /* What Arm's WSI layer (vulkan-wsi-layer, which ROCKNIX installs) needs
    * for Wayland swapchains: it allocates linear dma-bufs itself, imports
    * them with an explicit DRM format modifier layout, and calibrates
    * present timing against the device clock. Only DRM_FORMAT_MOD_LINEAR
    * is offered, and image formats report dma-bufs as importable only
    * (mali_formats.c): export is limited to the display swapchain's
    * dedicated GBM allocations (mali_device_memory.c). */
   .KHR_external_memory_fd = true,
   .EXT_external_memory_dma_buf = true,
   .EXT_image_drm_format_modifier = true,
   .KHR_image_format_list = true,
   .KHR_calibrated_timestamps = true,
   /* Vulkan 1.1 core; for an application that asks for Vulkan 1.0 the
    * layer enables them by name. */
   .KHR_bind_memory2 = true,
   .KHR_dedicated_allocation = true,
   .KHR_external_fence = true,
   .KHR_external_memory = true,
   .KHR_external_semaphore = true,
   .KHR_get_memory_requirements2 = true,
#endif
   /* Vulkan 1.1 core, listed by name for applications that ask for Vulkan
    * 1.0: Vita3K takes no device without both (and swapchain), and with
    * no device selected it crashes at boot. vkTrimCommandPool is Mesa's
    * common one; storage_buffer_storage_class is SPIR-V only. */
   .KHR_maintenance1 = true,
   .KHR_storage_buffer_storage_class = true,
   /* ARMSX2's in-tile destination read (the blob advertises both names).
    * Without it ARMSX2 reads the destination through an input attachment
    * with a by-region barrier between primitives, splitting draws (3.5
    * times the draw calls on Stuntman) and running up to 6.9 times the
    * blob's frame time. Input attachments are tile reads already; with
    * the rasterization-order flags the pipeline turns off forward pixel
    * kill and early depth/stencil updates (mali_pipeline_state.c), as
    * panvk does on v10+. v9's draw descriptor has no DCD Flags 2 word,
    * but that word holds tile-buffer read/write masks, which this
    * extension does not touch; the Flags 0 bits it needs exist on v9
    * the same as v11, and mali_jm_cmd_draw.c already copies them into
    * the job. On v7 the same bits are in the Renderer State Descriptor's
    * properties word (mali_pipeline_state.c). Advertised on every arch. */
   .EXT_rasterization_order_attachment_access = true,
   .ARM_rasterization_order_attachment_access = true,

   /* ARMSX2 selects Bresenham rasterization on every line pipeline; the
    * promoted KHR name and the original EXT name are separate table
    * entries but the same feature and property structs. */
   .KHR_line_rasterization = true,
   .EXT_line_rasterization = true,

   /* VMA (ARMSX2's allocator) only asks for a memory budget when this is
    * listed. */
   .EXT_memory_budget = true,

   /* Without it ARMSX2 de-indexes every flat-shaded draw on the CPU to
    * fake a last-vertex convention (up to 3x the vertices) instead of
    * just setting the pipeline's mode. Same hardware bit on v9 and v11;
    * on v7 it is a bit of each tiler job's Primitive word
    * (mali_jm_cmd_draw_v7.c). */
   .EXT_provoking_vertex = true,

   /* ARMSX2 pushes its texture set instead of allocating and writing a new
    * descriptor set per draw; every arch shares the same descriptor-set
    * code (mali_descriptor_set.c), so this needs no per-arch gate. */
   .KHR_push_descriptor = true,

   /* Descriptor indexing: the sets are read in place through the resource
    * tables on v9 and v11, which is what update after bind needs, and v7
    * copies a stage's descriptors from them when the draw executes
    * (get_descriptor_indexing_features has what each arch has). v7 cannot
    * meet the extension's limits for what it does not have, so it is
    * listed there under malisx2's own name (mali_vk.h): Mesa's runtime keys
    * the feature structs on this entry, which stays set, and
    * mali_EnumerateDeviceExtensionProperties and mali_CreateDevice swap the
    * name (mali_descriptor_indexing_renamed). */
   .EXT_descriptor_indexing = true,

   /* The timeline semaphore type is Mesa's emulated one over our binary
    * sync (mali_sync.c), so the device runs in Mesa's emulated timeline
    * mode. */
   .KHR_timeline_semaphore = true,

   /* Dynamic rendering, and the two extensions the Vulkan 1.1 form of it
    * depends on. The driver implements vkCmdBeginRendering /
    * vkCmdEndRendering on every arch; the render pass object API
    * (vkCreateRenderPass2, vkCmdBeginRenderPass2, ...) is Mesa's common
    * one (vk_render_pass.c), which lowers each subpass to those two. */
   .KHR_dynamic_rendering = true,
   .KHR_depth_stencil_resolve = true,
   .KHR_create_renderpass2 = true,

   /* What Vulkan 1.3 made core, on every arch. The device stays 1.1: 1.2
    * and 1.3 also require host query reset, buffer device address,
    * maintenance4, inline uniform blocks and subgroup size control, which
    * no arch has.
    *
    *  - synchronization2: the driver's barrier, submit and timestamp
    *    commands are the 2 forms (the 1.0 forms are the runtime's wrappers
    *    around them), so the 64-bit stage and access masks reach the same
    *    code, the event commands included (every arch has VkEvent);
    *  - extended dynamic state 1 and 2 and vertex input dynamic state: the
    *    runtime keeps the state, and a pipeline that leaves any of it
    *    dynamic has its hardware words packed again at the draw from the
    *    command buffer's state (mali_cmd_state.h mali_gfx_update_dirty).
    *    Rasterizer discard drops the draw. No logic op and no patch control
    *    points (no logicOp, no tessellation);
    *  - demote to helper invocation: every back end's discard is a demote;
    *  - private data: the runtime's, on every object. */
   .KHR_synchronization2 = true,
   .EXT_extended_dynamic_state = true,
   .EXT_extended_dynamic_state2 = true,
   .EXT_vertex_input_dynamic_state = true,
   .EXT_shader_demote_to_helper_invocation = true,
   .EXT_private_data = true,

   /* Reads of the bound colour and depth/stencil attachments inside a
    * dynamic rendering pass, and the maps that decide which output location
    * feeds which attachment and which input index reads which. Every arch
    * has the three parts it needs: input attachments are tile reads
    * (mali_nir_lower_input_attachments.c, panvk's scheme), where the maps
    * are the pipeline's (VkRenderingAttachmentLocationInfo and
    * VkRenderingInputAttachmentIndexInfo, kept by the runtime, which also
    * implements the two vkCmdSetRendering* commands); a blend descriptor
    * and the BLEND of an output location go with the attachment the
    * location feeds (mali_pipeline_state.c pack_blend; v7 in mali_shader.c
    * lower_sysvals); and a by-region barrier inside a pass is a draw with a
    * primitive barrier (cmd_fb_barrier). */
   .KHR_dynamic_rendering_local_read = true,
};

/*
 * 8- and 16-bit storage buffer and push constant access, and 8-bit integer
 * arithmetic, as panvk reports them. v7's shaders go through Mesa's Bifrost
 * backend, which lowers narrow loads and stores itself (bifrost_nir.c,
 * nir_lower_mem_access_bit_sizes); kraid (v9, v11) loads and stores bytes
 * and half words directly (8- and 16-bit OpLoad and OpStore) and has the
 * narrow integer ALU. parallel-RDP (N64) requires the storage features.
 * One predicate per extension, so that a part that fails on a device can be
 * turned off alone.
 */
static bool
storage8_for_arch(uint32_t arch)      /* VK_KHR_8bit_storage */
{
   return arch == 7 || arch == 9 || arch == 10 || arch == 11;
}

static bool
storage16_for_arch(uint32_t arch)     /* VK_KHR_16bit_storage */
{
   return arch == 7 || arch == 9 || arch == 10 || arch == 11;
}

static bool
int8_for_arch(uint32_t arch)          /* shaderInt8 of VK_KHR_shader_float16_int8 */
{
   return arch == 7 || arch == 9 || arch == 10 || arch == 11;
}

/*
 * Indirect draws with more than one command (multiDrawIndirect), a first
 * instance taken from the command (drawIndirectFirstInstance), and the
 * indirect count forms (VK_KHR_draw_indirect_count). v11 runs the commands
 * in a command-stream loop (mali_cmd_draw_indirect.c). On the job manager
 * (v7, v9) each draw of a call is its own jobs behind a patch job that
 * reads its parameters (mali_jm_idraw_nir.c), so maxDrawCount draws are
 * recorded for an indirect count call whatever the count turns out to be.
 */
static bool
draw_indirect_for_arch(uint32_t arch)
{
   return arch == 7 || arch == 9 || arch == 10 || arch == 11;
}

/*
 * Buffer device addresses (VK_KHR_buffer_device_address), without capture
 * and replay. A bound buffer's address is its memory's GPU address plus
 * the offset (mali_BindBufferMemory2), which Mesa's common
 * vkGetBufferDeviceAddress returns, and every allocation is mapped on the
 * GPU, so the allocate flag changes nothing. Physical storage buffer
 * pointers are 64bit_global (shader_spirv_options) and become global loads
 * and stores on every arch. Vita3K maps the guest's memory through these;
 * without them it runs with memory mapping disabled.
 */
static bool
bda_for_arch(uint32_t arch)
{
   return arch == 7 || arch == 9 || arch == 10 || arch == 11;
}

/*
 * std430 layout in uniform buffers (VK_KHR_uniform_buffer_standard_layout).
 * The offsets come from the SPIR-V; uniform buffer loads and the FAU push of
 * their words (lower_ubo_push) take any 4-byte-aligned offset. Vita3K
 * requires it with buffer device addresses.
 */
static bool
ubo_std_layout_for_arch(uint32_t arch)
{
   return arch == 7 || arch == 9 || arch == 10 || arch == 11;
}

/* maxDrawIndirectCount. v10, v11: the stream compares the count as a signed
 * word. v7, v9: the smallest limit Vulkan allows with multiDrawIndirect,
 * which bounds what one call records. */
static uint32_t
max_draw_indirect_count_for_arch(uint32_t arch)
{
   if (!draw_indirect_for_arch(arch))
      return 1;
   return arch >= 10 ? INT32_MAX : UINT16_MAX;
}

#ifndef __ANDROID__
/*
 * Arm's vulkan-wsi-layer, the Wayland and display layer some Linux
 * distributions ship for Mali (ROCKNIX's RK3588 image among them), puts
 * VK_KHR_sampler_ycbcr_conversion in the device extensions its Wayland
 * backend requires, whatever API version the application asks for, in its
 * releases before upstream commit 4a6f61a; a device that does not list it
 * fails vkCreateDevice. The layer calls none of the extension's commands:
 * it wants the disjoint image flag the extension brought, and uses that
 * only for a format whose modifier reports VK_FORMAT_FEATURE_DISJOINT_BIT,
 * which no format here does.
 *
 * So the name is listed while that layer is loaded in the process (the
 * loader loads implicit and enabled layers at vkCreateInstance, before it
 * enumerates physical devices), and never otherwise. samplerYcbcrConversion
 * stays false, which the extension does not allow (vk.xml requires the
 * feature of every device that lists it): an application that checks the
 * feature, as it must before creating a conversion, sees none, a device
 * that enables the feature is refused, and a sampler with a conversion is
 * refused (mali_sampler.c). The commands are the Vulkan 1.1 core ones every
 * device already resolves.
 */
static int
is_arm_wsi_layer(struct dl_phdr_info *info, size_t size, void *data)
{
   static const char layer[] = "libVkLayer_window_system_integration.so";
   const char *name = info->dlpi_name;
   if (!name)
      return 0;
   const char *base = strrchr(name, '/');
   base = base ? base + 1 : name;
   return strncmp(base, layer, sizeof(layer) - 1) == 0;
}

static bool
arm_wsi_layer_loaded(void)
{
   return dl_iterate_phdr(is_arm_wsi_layer, NULL) != 0;
}
#endif

static struct vk_device_extension_table
device_extensions_for_arch(uint32_t arch)
{
   struct vk_device_extension_table ext = mali_device_extensions;

   ext.KHR_draw_indirect_count = draw_indirect_for_arch(arch);

   /* 8- and 16-bit storage and int8 (see storage8_for_arch). */
   ext.KHR_8bit_storage = storage8_for_arch(arch);
   ext.KHR_16bit_storage = storage16_for_arch(arch);
   ext.KHR_shader_float16_int8 = int8_for_arch(arch);

   ext.KHR_buffer_device_address = bda_for_arch(arch);
   ext.KHR_uniform_buffer_standard_layout = ubo_std_layout_for_arch(arch);
#ifndef __ANDROID__
   /* For Arm's older WSI layer only (arm_wsi_layer_loaded). */
   ext.KHR_sampler_ycbcr_conversion = arm_wsi_layer_loaded();
#endif
   return ext;
}

/*
 * Dual-source blending. No arch has a second-source operand in the
 * fixed-function blender: a render target whose equation uses a SRC1
 * factor runs a blend shader, and only pipelines with such an equation
 * keep the fragment shader's second colour output (mali_shader.c). The
 * fragment shader's BLEND passes that colour in r4-r7 (blend2_pan) and the
 * blend shader reads it from there (load_blend_input_pan); on v7 Mesa's
 * Bifrost backend does both, as for panvk_v7.
 *
 * Reported on v7, v9, v10 and v11. ARMSX2 then hardware-blends some draws into
 * 16-bit PS2 frames without the PS2's colour truncation, a few levels too
 * bright, as on every other driver that reports it; it saves those draws
 * a framebuffer read. Without it ARMSX2 also takes its render-target-read
 * path for stencil shadows (fast_stencil_shadow needs it); on a G52 those
 * reads cost a pass and a copy each.
 */
static bool
dual_src_blend_for_arch(uint32_t arch)
{
   switch (arch) {
   case 7:
   case 9:
   case 10:
   case 11:
      return true;
   default:
      return false;
   }
}

/* Whether every format from first to last is a sampled image, linearly
 * filtered and a blit source (Vulkan's required format support for the
 * texture compression features). */
static bool
formats_complete(const struct mali_physical_device *pdev, VkFormat first, VkFormat last)
{
   const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                     VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                                     VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
   for (VkFormat f = first; f <= last; f++) {
      if ((mali_format_image_features(pdev, f) & need) != need)
         return false;
   }
   return true;
}

/* textureCompressionBC: v7 and v9 have every BCn format when the GPU has
 * all of BC1-7 (mali_format_table.c); v10 and v11 answer no for every BCn
 * format. */
static bool
bc_formats_complete(const struct mali_physical_device *pdev)
{
   return formats_complete(pdev, VK_FORMAT_BC1_RGB_UNORM_BLOCK, VK_FORMAT_BC7_SRGB_BLOCK);
}

/* textureCompressionETC2 and textureCompressionASTC_LDR: the GPU's
 * TEXTURE_FEATURES_0 has a bit per compressed format (mali_format_table.c),
 * so a part without them says so. The G615 (0xc7fe001e), the G52 and the
 * G57 (0xf7fffffe) have every ETC2, EAC and 2D ASTC LDR format. */
static bool
etc2_formats_complete(const struct mali_physical_device *pdev)
{
   return formats_complete(pdev, VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK,
                           VK_FORMAT_EAC_R11G11_SNORM_BLOCK);
}

static bool
astc_ldr_formats_complete(const struct mali_physical_device *pdev)
{
   return formats_complete(pdev, VK_FORMAT_ASTC_4x4_UNORM_BLOCK,
                           VK_FORMAT_ASTC_12x12_SRGB_BLOCK);
}

/*
 * VkPhysicalDeviceDescriptorIndexingFeatures (VK_EXT_descriptor_indexing on
 * v9 and v11, VK_MALISX2_descriptor_indexing on v7: mali_vk.h).
 *
 * v9 and v11: a set is read in place through its resource table entry
 * when a job runs, so a descriptor written after the set is bound, or never
 * written when the shader does not reach it, needs nothing from the
 * driver; the variable-count binding is the set's size (mali_descriptor_set.c).
 * Every kind of array a stage can index is lowered with
 * nir_lower_non_uniform_access (mali_shader.c). Input attachments are tile
 * reads with a compile-time attachment (mali_nir_lower_input_attachments.c),
 * so arrays of them are not offered. Dynamic uniform and storage buffers
 * are not update-after-bind (a specification rule) and their addresses are
 * taken when the command is recorded.
 *
 * v7: runtime arrays, partially bound bindings, sampled images updated
 * after bind and non-uniform sampled-image indexing, within what its
 * descriptor tables, copied by a job when the draw runs, can do: no
 * variable count, no update after bind of buffers and storage images, no
 * update while pending.
 */
static void
get_descriptor_indexing_features(struct vk_features *f, uint32_t arch)
{
   if (arch == 7) {
      f->runtimeDescriptorArray = true;
      f->descriptorBindingPartiallyBound = true;
      f->descriptorBindingSampledImageUpdateAfterBind = true;
      f->shaderSampledImageArrayNonUniformIndexing = true;
      return;
   }

   f->shaderUniformTexelBufferArrayDynamicIndexing = true;
   f->shaderStorageTexelBufferArrayDynamicIndexing = true;
   f->shaderUniformBufferArrayNonUniformIndexing = true;
   f->shaderSampledImageArrayNonUniformIndexing = true;
   f->shaderStorageBufferArrayNonUniformIndexing = true;
   f->shaderStorageImageArrayNonUniformIndexing = true;
   f->shaderUniformTexelBufferArrayNonUniformIndexing = true;
   f->shaderStorageTexelBufferArrayNonUniformIndexing = true;
   f->descriptorBindingUniformBufferUpdateAfterBind = true;
   f->descriptorBindingSampledImageUpdateAfterBind = true;
   f->descriptorBindingStorageImageUpdateAfterBind = true;
   f->descriptorBindingStorageBufferUpdateAfterBind = true;
   f->descriptorBindingUniformTexelBufferUpdateAfterBind = true;
   f->descriptorBindingStorageTexelBufferUpdateAfterBind = true;
   f->descriptorBindingUpdateUnusedWhilePending = true;
   f->descriptorBindingPartiallyBound = true;
   f->descriptorBindingVariableDescriptorCount = true;
   f->runtimeDescriptorArray = true;
}

/*
 * The features we intend to implement for ARMSX2 on this GPU, not the
 * blob's full list. Values that decide ARMSX2's paths follow the blob
 * where we can back them.
 */
static void
get_features(struct vk_features *f, uint32_t arch)
{
   *f = (struct vk_features) {
      /* Vulkan 1.0 */
      .robustBufferAccess = true,
      .fullDrawIndexUint32 = true,
      .imageCubeArray = true,
      .independentBlend = true,
      .geometryShader = false,        /* blob: true; no GS in our compiler */
      .tessellationShader = false,
      .sampleRateShading = true,
      .dualSrcBlend = dual_src_blend_for_arch(arch),
      .logicOp = false,
      .multiDrawIndirect = draw_indirect_for_arch(arch),
      .drawIndirectFirstInstance = draw_indirect_for_arch(arch),
      /* Vulkan 1.2's name for VK_KHR_draw_indirect_count. */
      .drawIndirectCount = draw_indirect_for_arch(arch),
      .depthClamp = true,
      .depthBiasClamp = true,
      /* v7, v10 and v11 (not v9): clip distances are varyings the fragment
       * shader tests and discards on (mali_shader.c, "Clip distances").
       * The test is per pixel, at the pixel centre, so under
       * multisampling the clipped edge is not antialiased; and with a
       * fragment shader that forces early fragment tests and also
       * discards or writes memory, depth or stencil, clipped fragments
       * still write depth and stencil. Cull distances are not
       * supported. */
      .shaderClipDistance = arch == 7 || arch == 10 || arch == 11,
      .fillModeNonSolid = false,
      .depthBounds = false,
      .wideLines = false,
      .largePoints = true,
      .alphaToOne = false,
      .multiViewport = false,
      /* These four follow the GPU (physical_device_create): the model's
       * anisotropy revision and the compressed formats it samples. */
      .samplerAnisotropy = true,
      .textureCompressionETC2 = true,
      .textureCompressionASTC_LDR = true,
      .textureCompressionBC = false,
      .occlusionQueryPrecise = true,
      .pipelineStatisticsQuery = false,
      .vertexPipelineStoresAndAtomics = false,
      .fragmentStoresAndAtomics = true,
      .shaderImageGatherExtended = true,
      .shaderStorageImageExtendedFormats = true,
      .shaderStorageImageReadWithoutFormat = true,
      .shaderStorageImageWriteWithoutFormat = true,
      .shaderUniformBufferArrayDynamicIndexing = true,
      .shaderSampledImageArrayDynamicIndexing = true,
      .shaderStorageBufferArrayDynamicIndexing = true,
      .shaderStorageImageArrayDynamicIndexing = true,
      .shaderInt16 = true,

      /* VK_KHR_8bit_storage, VK_KHR_16bit_storage and the int8 half of
       * VK_KHR_shader_float16_int8 (storage8_for_arch and the two after
       * it). */
      .storageBuffer8BitAccess = storage8_for_arch(arch),
      .uniformAndStorageBuffer8BitAccess = storage8_for_arch(arch),
      .storagePushConstant8 = storage8_for_arch(arch),
      .storageBuffer16BitAccess = storage16_for_arch(arch),
      .uniformAndStorageBuffer16BitAccess = storage16_for_arch(arch),
      .storagePushConstant16 = storage16_for_arch(arch),
      .shaderInt8 = int8_for_arch(arch),

      /* VK_KHR_buffer_device_address and
       * VK_KHR_uniform_buffer_standard_layout (bda_for_arch and the one
       * after it). */
      .bufferDeviceAddress = bda_for_arch(arch),
      .bufferDeviceAddressCaptureReplay = false,
      .bufferDeviceAddressMultiDevice = false,
      .uniformBufferStandardLayout = ubo_std_layout_for_arch(arch),

      /* Vulkan 1.1: multiview is required of a 1.1 device. */
      .multiview = true,

      /* VK_EXT_rasterization_order_attachment_access, every arch: the
       * draw descriptor's Flags 0 word (mali_pipeline_state.c) carries
       * this on v9 the same as v11, the Renderer State Descriptor's
       * properties word on v7. */
      .rasterizationOrderColorAttachmentAccess = true,
      .rasterizationOrderDepthAttachmentAccess = true,
      .rasterizationOrderStencilAttachmentAccess = true,

      /* VK_EXT_line_rasterization: the aligned-line-ends DCD bit
       * (mali_pipeline_state.c) already follows the pipeline's line mode.
       * Bresenham is what ARMSX2 asks for; the non-Bresenham modes are the
       * hardware's one fixed algorithm already, which panvk also reports
       * as rectangular. No smooth (coverage-based) lines, no stipple. */
      .rectangularLines = true,
      .bresenhamLines = true,
      .smoothLines = false,
      .stippledRectangularLines = false,
      .stippledBresenhamLines = false,
      .stippledSmoothLines = false,

      /* VK_EXT_provoking_vertex. No transform feedback, so the second
       * bit is moot either way. */
      .provokingVertexLast = true,
      .transformFeedbackPreservesProvokingVertex = false,

      /* VK_EXT_device_fault */
      .deviceFaultEXT = true,
      .deviceFaultVendorBinaryEXT = false,

      /* VK_KHR_push_descriptor */
      .pushDescriptor = true,

      /* VK_KHR_timeline_semaphore. The binary sync type wrapped by Mesa's
       * vk_sync_timeline (mali_sync.c): each signalled value is a point
       * sync the queue submits like a binary semaphore. ARMSX2 does not use
       * it; hakuX's GPU clock is one timeline semaphore (frame completion,
       * readbacks, slot reuse and present) and a floor feature there. */
      .timelineSemaphore = true,

      /* VK_KHR_dynamic_rendering, and the two extensions it needs below
       * Vulkan 1.2 (KHR_depth_stencil_resolve and KHR_create_renderpass2).
       * Begin and end rendering and the pipeline's dynamic rendering state
       * are what malisx2 has always used (mali_fb.c); the render pass object
       * API behind those two extensions is Mesa's common one, which lowers
       * render passes to them (vk_render_pass.c). Advertised for hakuX,
       * whose only rendering path on a 1.1 device is vkCmdBeginRendering
       * and which requires all three names. */
      .dynamicRendering = true,

      /* The extensions of mali_device_extensions that Vulkan 1.3 made
       * core. Extended dynamic state 2 reports neither of its two optional
       * parts. */
      .synchronization2 = true,
      .privateData = true,
      .shaderDemoteToHelperInvocation = true,
      .extendedDynamicState = true,
      .extendedDynamicState2 = true,
      .extendedDynamicState2LogicOp = false,
      .extendedDynamicState2PatchControlPoints = false,
      .vertexInputDynamicState = true,

      /* VK_KHR_dynamic_rendering_local_read (the table above). */
      .dynamicRenderingLocalRead = true,
   };

   get_descriptor_indexing_features(f, arch);
}

/* The sample counts we report. Rendering is single-sampled (mali_fb.c) and
 * vkCmdResolveImage does nothing, so only 1 is offered. That is below the
 * minimum Vulkan requires (1|4 for the framebuffer and sampledImage limits).
 * Back to 1|4 when multisampled rendering and resolve exist. */
#define MALI_SUPPORTED_SAMPLE_COUNTS VK_SAMPLE_COUNT_1_BIT

/* Sample counts for a format of the given bytes per pixel: 1 and 4
 * samples when bytes x 512 fits the tile buffer, 8 at bytes x 1024, 16 at
 * bytes x 2048, limited to MALI_SUPPORTED_SAMPLE_COUNTS. */
static VkSampleCountFlags
sample_counts(unsigned bytes_per_pixel, unsigned tilebuf_bytes)
{
   VkSampleCountFlags s = VK_SAMPLE_COUNT_1_BIT;
   if (bytes_per_pixel * 512 <= tilebuf_bytes)
      s |= VK_SAMPLE_COUNT_4_BIT;
   if (bytes_per_pixel * 1024 <= tilebuf_bytes)
      s |= VK_SAMPLE_COUNT_8_BIT;
   if (bytes_per_pixel * 2048 <= tilebuf_bytes)
      s |= VK_SAMPLE_COUNT_16_BIT;
   return s & MALI_SUPPORTED_SAMPLE_COUNTS;
}

/*
 * Per-part values (tile buffer sizes, the anisotropy revision, the Bifrost
 * compiler's choice of instruction forms) come from Mesa's model table,
 * looked up by GPU_ID and, on the command-stream parts, the CORE_FEATURES
 * variant (pan_get_model). A part the table does not list gets an entry of
 * our own (fallback_model), so pdev->model is set for every GPU we serve.
 *
 * Tile buffer, colour and depth/stencil, in bytes:
 *
 * - v9: 16 KiB and 8 KiB for every part. The four Arm drivers we have for
 *   arch 9 (G77, and G57 from three vendors) all store a 16 KiB budget
 *   whatever core_features says, and Mesa's model table gives the G57 and
 *   G68 16 and 8 KiB.
 * - Otherwise the model's sizes: on v7 16/8 KiB for the G52 (arch minor 2)
 *   and G76, 8/8 KiB for the G52 r1 (arch minor 4), G51 and G31; on v10
 *   32/16 KiB for the G610 and the G310's variants 3 and 4, 16/8 KiB for its
 *   variants 0 to 2; on v11 32/16 KiB for the G615 and G715 (variant 4).
 * - A part the table does not list (fallback_tilebuf): 32/16 KiB for the
 *   arch 10 and 11 products whose every listed entry has that (product
 *   majors 2 and 7 on v10, the G710 and G610, which share their shader core;
 *   2 and 3 on v11, the G715 and G615), which is what the tile size choice
 *   used for them before the table was consulted; for any other product the
 *   smallest sizes the table gives any part of the arch (8/8 KiB on v7,
 *   16/8 KiB on v10 and v11). The tile size is chosen to fit half the
 *   buffer, so a size below the hardware's costs only tile count, while one
 *   above it would overflow the tile buffer. The G510 is one of these: Mesa
 *   does not list it. (Arm's r44p1 driver sizes its sample-count budget on
 *   arch 10 and 11 by the variant alone, 32 KiB for variants 3 and 4 and
 *   16 KiB otherwise; Mesa's entries agree with that for the variants they
 *   list, apart from the G610's 32 KiB at variant 0.)
 */
struct mali_tilebuf {
   unsigned color;
   unsigned z;
};

static struct mali_tilebuf
fallback_tilebuf(const struct mali_kbase_gpu_props *p)
{
   const unsigned product_major = p->product_id & 0xf;
   switch (p->arch_major) {
   case 7:
      return (struct mali_tilebuf){8192, 8192};
   case 10:
      if (product_major == 2 || product_major == 7)
         return (struct mali_tilebuf){32768, 16384};
      return (struct mali_tilebuf){16384, 8192};
   case 11:
      if (product_major == 2 || product_major == 3)
         return (struct mali_tilebuf){32768, 16384};
      return (struct mali_tilebuf){16384, 8192};
   default:
      return (struct mali_tilebuf){16384, 8192};
   }
}

static struct mali_tilebuf
tilebuf_sizes(const struct mali_kbase_gpu_props *p, const struct pan_model *model)
{
   if (p->arch_major == 9)
      return (struct mali_tilebuf){16384, 8192};
   if (model)
      return (struct mali_tilebuf){model->tilebuffer.color_size, model->tilebuffer.z_size};
   return fallback_tilebuf(p);
}

/*
 * The model-table entry for a part Mesa does not list (the G510, G310
 * variants past 4, the G710, the G77, G78 and G78AE, and any product kbase
 * does not list either), from the GPU's own ID:
 *
 * - gpu_prod_id is the part's arch major, arch minor and product major, so
 *   the Bifrost compiler's choice of instruction forms (bifrost_get_quirks,
 *   which reads the arch minor from it: the full CLPER form exists from
 *   arch 7.2 on) follows the hardware. Without an entry it assumes arch
 *   minor 0, the oldest forms.
 * - Anisotropic filtering at every revision on Valhall, where every part in
 *   Mesa's table has it, and at none on an unlisted Bifrost part, since the
 *   table's Bifrost entries differ (the G51 has it from r1p1).
 * - The tile buffer of fallback_tilebuf.
 * - Rates that are only read for the compiler's statistics, which divide
 *   by them: per field, the smallest the table gives a part of arch 7 or
 *   9 to 11.
 */
static void
fallback_model(struct pan_model *m, const struct mali_kbase_gpu_props *p)
{
   const struct mali_tilebuf tb = fallback_tilebuf(p);
   *m = (struct pan_model){
      .gpu_prod_id = PAN_PROD_ID(p->arch_major, p->arch_minor, p->product_major),
      .gpu_variant = p->arch_major >= 10 ? p->core_features & 0xff : 0,
      .name = "Mali (not in Mesa's model table)",
      .min_rev_anisotropic = p->arch_major == 7 ? ~0u : 0,
      .tilebuffer = {.color_size = tb.color, .z_size = tb.z},
      .rates = {.pixel = 2, .texel = 2, .varying = 8, .cvt = 16, .fma = 8, .sfu = 4},
   };
}

/* Mesa's Bifrost compiler emits the old, quad-only CLPER form on arch 7
 * below arch minor 2 (the G51 and G31; BIFROST_LIMITED_CLPER in
 * bifrost_get_quirks, which reads the arch minor from the model), except in
 * read_invocation, which the waterfall loops of non-uniform resource
 * indexing and of divergent texel offsets use: that always emits the full
 * form, which those GPUs do not have (mali_shader.c refuses such a
 * shader). */
bool
mali_physical_device_limited_clper(const struct mali_physical_device *pdev)
{
   return pdev->arch == 7 && pdev->model && ((pdev->model->gpu_prod_id >> 8) & 0xff) < 2;
}

/* For image format queries (mali_formats.c, declared in mali_image.h). */
VkSampleCountFlags mali_physical_device_sample_counts(const struct mali_physical_device *pdev,
                                                      unsigned bytes_per_pixel);

VkSampleCountFlags
mali_physical_device_sample_counts(const struct mali_physical_device *pdev,
                                   unsigned bytes_per_pixel)
{
   return sample_counts(bytes_per_pixel, tilebuf_sizes(&pdev->props, pdev->model).color);
}

/* For the tile size choice (mali_fb.c, declared in mali_image.h). */
unsigned mali_physical_device_tilebuf_color_bytes(const struct mali_physical_device *pdev);
unsigned mali_physical_device_tilebuf_z_bytes(const struct mali_physical_device *pdev);

unsigned
mali_physical_device_tilebuf_color_bytes(const struct mali_physical_device *pdev)
{
   return tilebuf_sizes(&pdev->props, pdev->model).color;
}

unsigned
mali_physical_device_tilebuf_z_bytes(const struct mali_physical_device *pdev)
{
   return tilebuf_sizes(&pdev->props, pdev->model).z;
}

static void
hash_uuid(uint8_t out[VK_UUID_SIZE], const void *data, size_t size)
{
   blake3_hash h;
   _mesa_blake3_compute(data, size, h);
   memcpy(out, h, VK_UUID_SIZE);
}

/*
 * pipelineCacheUUID must change whenever our compiler or the shader binary
 * format changes, and should not change otherwise: ARMSX2 throws its
 * whole pipeline cache away when the UUID differs, so a UUID tied to the
 * commit made every driver update start with an empty cache. It hashes
 * the architecture's shader cache ID (a hash of every source that can
 * change that architecture's shader binaries, src/vk/gen_shader_cache_id.py:
 * MALI_SHADER_CACHE_ID for arch 11, MALI_SHADER_CACHE_ID_V10,
 * MALI_SHADER_CACHE_ID_V9 and MALI_SHADER_CACHE_ID_V7) and the GPU ID. The
 * GPU ID matters within an arch: Mesa's Bifrost compiler emits different
 * instructions below arch 7.2 (the G51 and G31; bifrost_get_quirks), so a
 * G52's cache must not load on a G31. kraid's output depends on the arch
 * alone. vk_pipeline_cache also refuses data whose deviceID (the GPU ID)
 * differs.
 */
static void
pipeline_cache_uuid(uint8_t out[VK_UUID_SIZE], const char *cache_id, size_t cache_id_size,
                    uint64_t gpu_id)
{
   struct mesa_blake3 ctx;
   _mesa_blake3_init(&ctx);
   _mesa_blake3_update(&ctx, cache_id, cache_id_size);
   _mesa_blake3_update(&ctx, &gpu_id, sizeof(gpu_id));

   blake3_hash h;
   _mesa_blake3_final(&ctx, h);
   memcpy(out, h, VK_UUID_SIZE);
}

static void
get_properties(const struct mali_physical_device *pdev, const char *name,
               struct vk_properties *props)
{
   const struct mali_kbase_gpu_props *p = &pdev->props;
   const VkSampleCountFlags samples =
      sample_counts(4, tilebuf_sizes(p, pdev->model).color);

   /* min(max_workgroup_size, max_barrier_size); 1024 on the device.
    * Fall back to Vulkan's minimum if the kernel did not say. */
   uint32_t wg = MIN2(p->max_workgroup_size, p->max_barrier_size);
   if (!wg)
      wg = 128;

   const uint64_t max_alloc = MIN2(heap_size(p), 8ull << 30);
   const float ts_period = pdev->timestamp_hz ? 1e9f / (float)pdev->timestamp_hz : 1.0f;

   /* The G57 and G52 builds' own identity schemes (above). */
   const uint32_t driver_version = driver_version_for_arch(p->arch_major);
   const char *const driver_info = driver_info_for_arch(p->arch_major);

   *props = (struct vk_properties) {
      .apiVersion = MALI_API_VERSION,
      .driverVersion = driver_version,
      .vendorID = MALI_VENDOR_ID,
      .deviceID = (uint32_t)p->gpu_id,
      .deviceType = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,

      /* Hardware limits: the blob's values. */
      .maxImageDimension1D = 16384,
      .maxImageDimension2D = 16384,
      .maxImageDimension3D = 16384,
      .maxImageDimensionCube = 16384,
      .maxImageArrayLayers = 4096,
      .maxTexelBufferElements = 1u << 28,
      .maxStorageBufferRange = 1u << 31,
      .maxMemoryAllocationCount = 16384,
      .maxSamplerAllocationCount = 4000,
      .bufferImageGranularity = 1,
      .sparseAddressSpaceSize = 0,
      .maxComputeSharedMemorySize = 32768,
      .maxComputeWorkGroupCount = { 65535, 65535, 65535 },
      .maxComputeWorkGroupInvocations = wg,
      .maxComputeWorkGroupSize = { wg, wg, wg },
      .subPixelPrecisionBits = 8,
      .subTexelPrecisionBits = 8,
      .mipmapPrecisionBits = 8,
      .lineSubPixelPrecisionBits = 8,     /* same as panvk on this hardware */
      .maxDrawIndexedIndexValue = UINT32_MAX,
      .maxDrawIndirectCount = max_draw_indirect_count_for_arch(pdev->arch),
      .maxSamplerLodBias = 126.0f,
      .maxSamplerAnisotropy = 16.0f,
      .maxViewports = 1,
      .maxViewportDimensions = { 16384, 16384 },
      .viewportBoundsRange = { -32768.0f, 32767.0f },
      .viewportSubPixelBits = 0,
      .minMemoryMapAlignment = 64,
      .minTexelBufferOffsetAlignment = 64,
      .minUniformBufferOffsetAlignment = 16,
      .minStorageBufferOffsetAlignment = 64,
      .minTexelOffset = -8,
      .maxTexelOffset = 7,
      .minTexelGatherOffset = -8,
      .maxTexelGatherOffset = 7,
      .minInterpolationOffset = -0.5f,
      .maxInterpolationOffset = 0.5f,
      .subPixelInterpolationOffsetBits = 4,
      .maxFramebufferWidth = 16384,
      .maxFramebufferHeight = 16384,
      .maxFramebufferLayers = 256,
      .framebufferColorSampleCounts = samples,
      .framebufferDepthSampleCounts = samples,
      .framebufferStencilSampleCounts = samples,
      .framebufferNoAttachmentsSampleCounts = samples,
      .maxColorAttachments = 8,
      .sampledImageColorSampleCounts = samples,
      .sampledImageIntegerSampleCounts = samples,
      .sampledImageDepthSampleCounts = samples,
      .sampledImageStencilSampleCounts = samples,
      .storageImageSampleCounts = VK_SAMPLE_COUNT_1_BIT,
      .maxSampleMaskWords = 1,
      .timestampComputeAndGraphics = pdev->timestamp_hz != 0,
      .timestampPeriod = ts_period,
      .maxClipDistances = 0,
      .maxCullDistances = 0,
      .maxCombinedClipAndCullDistances = 0,
      .discreteQueuePriorities = 2,
      .pointSizeRange = { 1.0f, 1024.0f },
      .lineWidthRange = { 1.0f, 1.0f },
      .pointSizeGranularity = 0.0625f,
      .lineWidthGranularity = 0.0f,
      .strictLines = true,
      .standardSampleLocations = true,
      .optimalBufferCopyOffsetAlignment = 64,
      .optimalBufferCopyRowPitchAlignment = 64,
      .nonCoherentAtomSize = 64,

      /*
       * Limits that follow from our descriptor and shader-interface design,
       * which is panvk's for v11: 7 sets, 2^20 descriptors per type, 16
       * dynamic uniform and 8 dynamic storage buffers, 16 vertex buffers,
       * 1 MiB uniform buffers (12-bit entry count of 16-byte entries),
       * 256 bytes of push constants.
       */
      .maxUniformBufferRange = 1u << 20,
      .maxPushConstantsSize = 256,
      .maxBoundDescriptorSets = 7,

      /* VK_KHR_push_descriptor. panvk's value; ARMSX2 needs 7 (its TFX
       * texture set). */
      .maxPushDescriptors = 32,
      .maxPerStageDescriptorSamplers = MALI_MAX_DESCS_PER_TYPE,
      .maxPerStageDescriptorUniformBuffers = MALI_MAX_DESCS_PER_TYPE,
      .maxPerStageDescriptorStorageBuffers = MALI_MAX_DESCS_PER_TYPE,
      .maxPerStageDescriptorSampledImages = MALI_MAX_DESCS_PER_TYPE,
      .maxPerStageDescriptorStorageImages = MALI_MAX_DESCS_PER_TYPE,
      .maxPerStageDescriptorInputAttachments = 9,
      .maxPerStageResources = 5 * MALI_MAX_DESCS_PER_TYPE + 9,
      .maxDescriptorSetSamplers = MALI_MAX_DESCS_PER_TYPE,
      .maxDescriptorSetUniformBuffers = MALI_MAX_DESCS_PER_TYPE,
      .maxDescriptorSetUniformBuffersDynamic = 16,
      .maxDescriptorSetStorageBuffers = MALI_MAX_DESCS_PER_TYPE,
      .maxDescriptorSetStorageBuffersDynamic = 8,
      .maxDescriptorSetSampledImages = MALI_MAX_DESCS_PER_TYPE,
      .maxDescriptorSetStorageImages = MALI_MAX_DESCS_PER_TYPE,
      .maxDescriptorSetInputAttachments = 9,
      .maxVertexInputAttributes = 16,
      .maxVertexInputBindings = 16,
      .maxVertexInputAttributeOffset = UINT32_MAX,
      .maxVertexInputBindingStride = MESA_VK_MAX_VERTEX_BINDING_STRIDE,
      .maxVertexOutputComponents = 128,
      .maxFragmentInputComponents = 128,
      .maxFragmentOutputAttachments = 8,
      /* kraid takes one second colour per shader (location 0). */
      .maxFragmentDualSrcAttachments = dual_src_blend_for_arch(pdev->arch) ? 1 : 0,
      .maxFragmentCombinedOutputResources = 8 + (1u << 12) + (1u << 8),

      /* Vulkan 1.1 */
      .subgroupSize = 16,
      /*
       * Only what a 1.1 device must have until we have checked which
       * subgroup operations kraid (Mesa's Bifrost backend on v7) handles.
       * Not in the vertex stage: the hardware may run vertex invocations
       * for indices that are never drawn (panvk's reasoning).
       */
      .subgroupSupportedStages = VK_SHADER_STAGE_FRAGMENT_BIT |
                                 VK_SHADER_STAGE_COMPUTE_BIT,
      .subgroupSupportedOperations = VK_SUBGROUP_FEATURE_BASIC_BIT,
      .subgroupQuadOperationsInAllStages = false,
      .pointClippingBehavior = VK_POINT_CLIPPING_BEHAVIOR_ALL_CLIP_PLANES,
      .maxMultiviewViewCount = 8,
      .maxMultiviewInstanceIndex = UINT32_MAX,
      .protectedNoFault = false,
      .maxPerSetDescriptors = UINT16_MAX,
      .maxMemoryAllocationSize = max_alloc,

      /* VK_KHR_timeline_semaphore: Mesa's emulated timelines
       * (vk_sync_timeline) keep 64-bit values with no limit on the gap. */
      .maxTimelineSemaphoreValueDifference = UINT64_MAX,

      /* VK_KHR_driver_properties */
      .driverID = VK_DRIVER_ID_ARM_PROPRIETARY,
      .conformanceVersion = { 0, 0, 0, 0 },

      /* VK_KHR_depth_stencil_resolve (advertised with
       * VK_KHR_dynamic_rendering, which depends on it below Vulkan 1.2).
       * Vulkan requires SAMPLE_ZERO in both sets; the arithmetic modes are
       * not implemented, since rendering is single-sampled
       * (MALI_SUPPORTED_SAMPLE_COUNTS). */
      .supportedDepthResolveModes = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT,
      .supportedStencilResolveModes = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT,
      .independentResolveNone = false,
      .independentResolve = false,

      /* VK_EXT_provoking_vertex: the hardware bit is per render pass, not
       * per pipeline (panvk reports the same). */
      .provokingVertexModePerPipeline = false,
      .transformFeedbackPreservesTriangleFanProvokingVertex = false,

      /* VK_KHR_dynamic_rendering_local_read (get_features): depth and
       * stencil are tile reads like colour; rendering is single-sampled
       * (MALI_SUPPORTED_SAMPLE_COUNTS), so there is no multisampled
       * attachment to read. */
      .dynamicRenderingLocalReadDepthStencilAttachments = true,
      .dynamicRenderingLocalReadMultisampledAttachments = false,
   };

   /*
    * VK_EXT_descriptor_indexing's properties on v9 and v11 (v7 has its own
    * below). An update-after-bind limit counts the descriptors of sets with
    * and without the update-after-bind bit, so none is below its ordinary
    * limit; with the sets read in place there is no cost to a descriptor
    * being update-after-bind, so they are the ordinary limits (as panvk's).
    * No pool-wide count is kept for update-after-bind pools: a pool is as
    * large as the memory it can allocate.
    *
    * No quad-divergent implicit level of detail: an implicit-lod operation
    * whose image or sampler differs inside a fragment quad is not defined
    * (nir_lower_non_uniform_access makes the index uniform in the warp, not
    * in the quad, mali_shader.c), and nothing is native: every non-uniform
    * index goes through that lowering.
    */
   if (p->arch_major != 7) {
      props->maxUpdateAfterBindDescriptorsInAllPools = UINT32_MAX;
      props->shaderUniformBufferArrayNonUniformIndexingNative = false;
      props->shaderSampledImageArrayNonUniformIndexingNative = false;
      props->shaderStorageBufferArrayNonUniformIndexingNative = false;
      props->shaderStorageImageArrayNonUniformIndexingNative = false;
      props->shaderInputAttachmentArrayNonUniformIndexingNative = false;
      props->robustBufferAccessUpdateAfterBind = true;
      props->quadDivergentImplicitLod = false;
      props->maxPerStageDescriptorUpdateAfterBindSamplers = props->maxPerStageDescriptorSamplers;
      props->maxPerStageDescriptorUpdateAfterBindUniformBuffers =
         props->maxPerStageDescriptorUniformBuffers;
      props->maxPerStageDescriptorUpdateAfterBindStorageBuffers =
         props->maxPerStageDescriptorStorageBuffers;
      props->maxPerStageDescriptorUpdateAfterBindSampledImages =
         props->maxPerStageDescriptorSampledImages;
      props->maxPerStageDescriptorUpdateAfterBindStorageImages =
         props->maxPerStageDescriptorStorageImages;
      props->maxPerStageDescriptorUpdateAfterBindInputAttachments =
         props->maxPerStageDescriptorInputAttachments;
      props->maxPerStageUpdateAfterBindResources = props->maxPerStageResources;
      props->maxDescriptorSetUpdateAfterBindSamplers = props->maxDescriptorSetSamplers;
      props->maxDescriptorSetUpdateAfterBindUniformBuffers = props->maxDescriptorSetUniformBuffers;
      props->maxDescriptorSetUpdateAfterBindUniformBuffersDynamic =
         props->maxDescriptorSetUniformBuffersDynamic;
      props->maxDescriptorSetUpdateAfterBindStorageBuffers = props->maxDescriptorSetStorageBuffers;
      props->maxDescriptorSetUpdateAfterBindStorageBuffersDynamic =
         props->maxDescriptorSetStorageBuffersDynamic;
      props->maxDescriptorSetUpdateAfterBindSampledImages = props->maxDescriptorSetSampledImages;
      props->maxDescriptorSetUpdateAfterBindStorageImages = props->maxDescriptorSetStorageImages;
      props->maxDescriptorSetUpdateAfterBindInputAttachments =
         props->maxDescriptorSetInputAttachments;
   }

   /*
    * The Mali-G52 (v7) has no resource tables and takes panvk_v7's limits:
    * per set, those of the Renderer State's 16-bit sampler and texture
    * counts and 8-bit uniform buffer count, the Uniform Buffer's 12-bit
    * entry count and the Attribute's 9-bit buffer index
    * (panvk_descriptor_set_layout.h); per stage, panvk_shader.h's. The
    * exception is the per-stage sampled image limit: storage images,
    * storage texel buffers and uniform texel buffers share a stage's image
    * table of MALI_V7_MAX_STAGE_IMG_ENTRIES (mali_shader.h), and Vulkan
    * counts uniform texel buffers as sampled images, so that limit is what
    * the storage images leave of the table. Uniform buffers hold at most
    * 4096 16-byte entries (genxml v7.xml). Warps are 8 lanes
    * (pan_subgroup_size).
    */
   if (p->arch_major == 10 || p->arch_major == 11) {
      props->maxClipDistances = MALI_V11_MAX_CLIP_DISTANCES;
      props->maxCombinedClipAndCullDistances = MALI_V11_MAX_CLIP_DISTANCES;
   }
   if (p->arch_major == 7) {
      props->maxClipDistances = MALI_V7_MAX_CLIP_DISTANCES;
      props->maxCombinedClipAndCullDistances = MALI_V7_MAX_CLIP_DISTANCES;
      props->maxUniformBufferRange = 1u << 16;
      props->maxPerStageDescriptorSamplers = 128;
      props->maxPerStageDescriptorUniformBuffers = UINT8_MAX - 32;
      props->maxPerStageDescriptorStorageBuffers = 64;
      props->maxPerStageDescriptorStorageImages = 32;
      props->maxPerStageDescriptorSampledImages =
         MALI_V7_MAX_STAGE_IMG_ENTRIES - props->maxPerStageDescriptorStorageImages;
      props->maxPerStageResources =
         props->maxPerStageDescriptorSamplers + props->maxPerStageDescriptorUniformBuffers +
         props->maxPerStageDescriptorStorageBuffers + props->maxPerStageDescriptorSampledImages +
         props->maxPerStageDescriptorStorageImages + props->maxPerStageDescriptorInputAttachments;
      props->maxDescriptorSetSamplers = UINT16_MAX;
      props->maxDescriptorSetUniformBuffers = UINT8_MAX - 32;
      props->maxDescriptorSetStorageBuffers = 1u << 12;
      props->maxDescriptorSetSampledImages = UINT16_MAX;
      props->maxDescriptorSetStorageImages = 1u << 8;
      props->subgroupSize = 8;

      /* VK_MALISX2_descriptor_indexing (get_descriptor_indexing_features):
       * the limits start as the ordinary ones; per-stage ones are raised
       * below. A non-uniform index must be uniform within a fragment quad
       * even for explicit level of detail, which quadDivergentImplicitLod
       * does not cover (mali_vk.h). */
      props->maxUpdateAfterBindDescriptorsInAllPools = UINT32_MAX / 2;
      props->quadDivergentImplicitLod = false;
      props->shaderSampledImageArrayNonUniformIndexingNative = false;
      props->maxPerStageDescriptorUpdateAfterBindSamplers = props->maxPerStageDescriptorSamplers;
      props->maxPerStageDescriptorUpdateAfterBindUniformBuffers =
         props->maxPerStageDescriptorUniformBuffers;
      props->maxPerStageDescriptorUpdateAfterBindStorageBuffers =
         props->maxPerStageDescriptorStorageBuffers;
      props->maxPerStageDescriptorUpdateAfterBindSampledImages =
         props->maxPerStageDescriptorSampledImages;
      props->maxPerStageDescriptorUpdateAfterBindStorageImages =
         props->maxPerStageDescriptorStorageImages;
      props->maxPerStageDescriptorUpdateAfterBindInputAttachments =
         props->maxPerStageDescriptorInputAttachments;
      props->maxPerStageUpdateAfterBindResources = props->maxPerStageResources;
      props->maxDescriptorSetUpdateAfterBindSamplers = props->maxDescriptorSetSamplers;
      props->maxDescriptorSetUpdateAfterBindUniformBuffers = props->maxDescriptorSetUniformBuffers;
      props->maxDescriptorSetUpdateAfterBindUniformBuffersDynamic =
         props->maxDescriptorSetUniformBuffersDynamic;
      props->maxDescriptorSetUpdateAfterBindStorageBuffers = props->maxDescriptorSetStorageBuffers;
      props->maxDescriptorSetUpdateAfterBindStorageBuffersDynamic =
         props->maxDescriptorSetStorageBuffersDynamic;
      props->maxDescriptorSetUpdateAfterBindSampledImages = props->maxDescriptorSetSampledImages;
      props->maxDescriptorSetUpdateAfterBindStorageImages = props->maxDescriptorSetStorageImages;
      props->maxDescriptorSetUpdateAfterBindInputAttachments =
         props->maxDescriptorSetInputAttachments;

      /*
       * VK_MALISX2_descriptor_indexing (get_descriptor_indexing_features).
       * An update-after-bind limit counts the descriptors of sets with and
       * without the update-after-bind bit, so none is below its ordinary
       * counterpart: per set they are the ordinary limits.
       *
       * Per stage, a stage whose textures and samplers all come from one
       * update-after-bind set reads that set as its tables
       * (mali_nir_lower_descriptors_v7.c direct sets), so a bindless array
       * costs nothing per draw there. The tables have 16-bit counts, so the
       * set must have fewer than 65536 slots: up to 16384 images and 1024
       * samplers (hakuX asks for 3 x 4096 and 256, and then drops its
       * per-draw sets: about a third of the driver's time per draw on the
       * Mali-G52).
       *
       * The sampled image limit counts uniform texel buffers too, and those
       * are not read through the Texture table: each takes an entry of the
       * stage's image table, which storage images and storage texel buffers
       * share and which holds MALI_V7_MAX_STAGE_IMG_ENTRIES whatever the
       * limits say. A pipeline whose stage needs more fails to be created
       * with VK_ERROR_FEATURE_NOT_PRESENT (mali_shader.c v7_fixup_info).
       */
      props->maxPerStageDescriptorUpdateAfterBindSampledImages = 16384;
      props->maxPerStageDescriptorUpdateAfterBindSamplers = 1024;
      props->maxPerStageUpdateAfterBindResources =
         props->maxPerStageDescriptorUpdateAfterBindSamplers +
         props->maxPerStageDescriptorUpdateAfterBindUniformBuffers +
         props->maxPerStageDescriptorUpdateAfterBindStorageBuffers +
         props->maxPerStageDescriptorUpdateAfterBindSampledImages +
         props->maxPerStageDescriptorUpdateAfterBindStorageImages +
         props->maxPerStageDescriptorUpdateAfterBindInputAttachments;
   }

   snprintf(props->deviceName, sizeof(props->deviceName), "%s", name);
   /* The blob reports the device name as driverName too. */
   snprintf(props->driverName, sizeof(props->driverName), "%s", name);
   snprintf(props->driverInfo, sizeof(props->driverInfo), "%s", driver_info);

   /* deviceUUID: gpu_id little-endian, then 1. */
   uint32_t id = (uint32_t)p->gpu_id;
   memcpy(props->deviceUUID, &id, sizeof(id));
   props->deviceUUID[4] = 1;
   /* driverUUID: a hash of driverInfo, so never the blob's. */
   hash_uuid(props->driverUUID, driver_info, strlen(driver_info) + 1);
   if (p->arch_major == 7)
      pipeline_cache_uuid(props->pipelineCacheUUID, MALI_SHADER_CACHE_ID_V7,
                          sizeof(MALI_SHADER_CACHE_ID_V7), p->gpu_id);
   else if (p->arch_major == 9)
      pipeline_cache_uuid(props->pipelineCacheUUID, MALI_SHADER_CACHE_ID_V9,
                          sizeof(MALI_SHADER_CACHE_ID_V9), p->gpu_id);
   else if (p->arch_major == 10)
      pipeline_cache_uuid(props->pipelineCacheUUID, MALI_SHADER_CACHE_ID_V10,
                          sizeof(MALI_SHADER_CACHE_ID_V10), p->gpu_id);
   else
      pipeline_cache_uuid(props->pipelineCacheUUID, MALI_SHADER_CACHE_ID,
                          sizeof(MALI_SHADER_CACHE_ID), p->gpu_id);
}

/* ---------------------------------------------------------------------- */
/* Queries we implement ourselves                                          */

/* Mesa's list, with VK_EXT_descriptor_indexing shown as malisx2's own
 * VK_MALISX2_descriptor_indexing on v7 (device_extensions_for_arch,
 * mali_vk.h). */
VKAPI_ATTR VkResult VKAPI_CALL
mali_EnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice,
                                        const char *pLayerName,
                                        uint32_t *pPropertyCount,
                                        VkExtensionProperties *pProperties)
{
   VK_FROM_HANDLE(mali_physical_device, pdev, physicalDevice);
   VK_OUTARRAY_MAKE_TYPED(VkExtensionProperties, out, pProperties, pPropertyCount);

   for (int i = 0; i < VK_DEVICE_EXTENSION_COUNT; i++) {
      if (!pdev->vk.supported_extensions.extensions[i])
         continue;
#ifdef ANDROID_STRICT
      if (!vk_android_allowed_device_extensions.extensions[i])
         continue;
#endif
      vk_outarray_append_typed(VkExtensionProperties, &out, prop) {
         *prop = vk_device_extensions[i];
         if (mali_descriptor_indexing_renamed(pdev->arch) &&
             !strcmp(prop->extensionName, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)) {
            snprintf(prop->extensionName, sizeof(prop->extensionName), "%s",
                     VK_MALISX2_DESCRIPTOR_INDEXING_EXTENSION_NAME);
            prop->specVersion = VK_MALISX2_DESCRIPTOR_INDEXING_SPEC_VERSION;
         }
      }
   }

   return vk_outarray_status(&out);
}

VKAPI_ATTR void VKAPI_CALL
mali_GetPhysicalDeviceQueueFamilyProperties2(VkPhysicalDevice physicalDevice,
                                             uint32_t *pQueueFamilyPropertyCount,
                                             VkQueueFamilyProperties2 *pQueueFamilyProperties)
{
   VK_FROM_HANDLE(mali_physical_device, pdev, physicalDevice);
   VK_OUTARRAY_MAKE_TYPED(VkQueueFamilyProperties2, out, pQueueFamilyProperties,
                          pQueueFamilyPropertyCount);

   vk_outarray_append_typed(VkQueueFamilyProperties2, &out, p) {
      p->queueFamilyProperties = (VkQueueFamilyProperties) {
         .queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT |
                       VK_QUEUE_TRANSFER_BIT,
         .queueCount = MALI_QUEUE_COUNT,
         .timestampValidBits = pdev->timestamp_hz ? 64 : 0,
         .minImageTransferGranularity = { 1, 1, 1 },
      };
      /* No pNext structure we know is filled: we expose no global
       * priorities and no other queue-family extension. */
   }
}

VKAPI_ATTR void VKAPI_CALL
mali_GetPhysicalDeviceMemoryProperties2(VkPhysicalDevice physicalDevice,
                                        VkPhysicalDeviceMemoryProperties2 *pMemoryProperties)
{
   VK_FROM_HANDLE(mali_physical_device, pdev, physicalDevice);
   pMemoryProperties->memoryProperties = pdev->memory;

   vk_foreach_struct(stype, ext, pMemoryProperties->pNext) {
      switch (stype) {
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT: {
         VkPhysicalDeviceMemoryBudgetPropertiesEXT *b = ext;
         const uint64_t used = p_atomic_read(&pdev->heap_used);

         for (unsigned i = 0; i < VK_MAX_MEMORY_HEAPS; i++) {
            b->heapBudget[i] = 0;
            b->heapUsage[i] = 0;
         }
         /* One heap (init_memory above). The budget never drops below our
          * own usage and never exceeds the heap: min(heap size, usage +
          * available system memory), no margin held back. */
         b->heapUsage[0] = used;
         b->heapBudget[0] = vk_physical_device_heap_budget(
            available_memory(pdev), 1.0f, pdev->memory.memoryHeaps[0].size, used);
         break;
      }
      default:
         vk_debug_ignored_stype(stype);
         break;
      }
   }
}

/* ---------------------------------------------------------------------- */
/* Creation                                                                */

/* Collects the kbase layer's messages while probing a node, so a node
 * that does not exist costs no log line. */
struct probe_log {
   char text[512];
   size_t len;
};

static void
probe_log_append(void *user, const char *msg)
{
   struct probe_log *l = user;
   int n = snprintf(l->text + l->len, sizeof(l->text) - l->len, "%s%s",
                    l->len ? "; " : "", msg);
   if (n > 0)
      l->len = MIN2(l->len + (size_t)n, sizeof(l->text) - 1);
}

static VkResult
physical_device_create(struct mali_instance *instance, const char *path,
                       struct vk_physical_device **out)
{
   struct probe_log log = { .len = 0 };
   struct mali_kbase *kb;

   /*
    * Like the blob, open a kbase context just to read the GPU
    * properties, then close it again. Each VkDevice opens its own.
    */
   const struct mali_kbase_create_info info = {
      .device_path = path,
      .backend = instance->kbase_backend,
      .log = probe_log_append,
      .log_user = &log,
   };
   enum mali_kbase_result r = mali_kbase_create(&info, &kb);
   if (r == MALI_KBASE_ERROR_DEVICE_NOT_FOUND)
      return VK_ERROR_INCOMPATIBLE_DRIVER;
   if (r != MALI_KBASE_SUCCESS) {
      mesa_logw("malisx2: skipping %s: %s (%s)", path, mali_kbase_result_str(r),
                log.text);
      return VK_ERROR_INCOMPATIBLE_DRIVER;
   }

   /* The physical device and the model entry fallback_model fills when
    * Mesa's table has none; freed together (mali_physical_device_destroy
    * frees the first member's address). */
   struct pdev_alloc {
      struct mali_physical_device pdev;
      struct pan_model model;
   } *pa = vk_zalloc(&instance->vk.alloc, sizeof(*pa), 8, VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   struct mali_physical_device *pdev = pa ? &pa->pdev : NULL;
   if (!pdev) {
      mali_kbase_destroy(kb);
      return vk_error(instance, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   snprintf(pdev->path, sizeof(pdev->path), "%s", path);
   pdev->props = kb->props;
   pdev->cs_work_registers = kb->glb.cs_work_registers;
   pdev->arch = pdev->props.arch_major;
   /* Mesa's model table keys the command-stream parts by the low byte of
    * CORE_FEATURES (what panthor reports as the variant) and lists the
    * job-manager parts as variant 0 (the panfrost kernel driver reports
    * none). */
   pdev->model = pan_get_model(pdev->props.gpu_id,
                               pdev->arch >= 10 ? pdev->props.core_features & 0xff : 0,
                               pdev->props.core_count, pdev->props.gpu_features);
   if (!pdev->model) {
      fallback_model(&pa->model, &pdev->props);
      pdev->model = &pa->model;
   }
   const enum mali_kbase_frontend frontend = kb->frontend;
   mali_kbase_destroy(kb);

   /* The arch has to match the frontend the kernel speaks: CSF with v10
    * and v11, the job manager with v7 and v9. */
   const bool arch_ok =
      (mali_arch_is_csf(pdev->arch) && frontend == MALI_KBASE_FRONTEND_CSF) ||
      (mali_arch_is_jm(pdev->arch) && frontend == MALI_KBASE_FRONTEND_JM);
   char name[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE];
   if (!arch_ok || !mali_device_name(&pdev->props, name, sizeof(name))) {
      mesa_logw("malisx2: skipping %s: GPU product 0x%04x (arch %u) is not "
                "one this driver was built for (arch 7 or 9 on a job-manager "
                "kernel, arch 10 or 11 on a command-stream kernel)", path,
                pdev->props.product_id, pdev->props.arch_major);
      vk_free(&instance->vk.alloc, pdev);
      return VK_ERROR_INCOMPATIBLE_DRIVER;
   }

   pdev->timestamp_hz = timestamp_frequency();
   init_memory(pdev);
   /* driverInfo carries the shader cache ID, not the commit; say which build
    * this is here. No "malisx2:" prefix: the corpus tools count those lines
    * as warnings. */
   mesa_logi("malisx2 build %s, driverInfo %s", MALI_GIT_SHA,
            driver_info_for_arch(pdev->arch));
   /* One bit per compressed texture format the GPU samples (BCn, ETC2,
    * ASTC; pan_format texfeat_bit), to tell which a device can take. */
   mesa_logi("malisx2 texture features 0x%08x 0x%08x 0x%08x 0x%08x (raw 0x%08x)",
            pdev->props.texture_features[0], pdev->props.texture_features[1],
            pdev->props.texture_features[2], pdev->props.texture_features[3],
            pdev->props.raw_texture_features[0]);

   struct vk_features features;
   struct vk_properties props;
   get_features(&features, pdev->arch);
   features.textureCompressionBC = bc_formats_complete(pdev);
   features.textureCompressionETC2 = etc2_formats_complete(pdev);
   features.textureCompressionASTC_LDR = astc_ldr_formats_complete(pdev);
   /* Mesa's model table: the revision anisotropic filtering starts at
    * (r1p1 on the G51, every revision on the other parts we serve). */
   features.samplerAnisotropy = pan_rev(pdev->props.gpu_id) >= pdev->model->min_rev_anisotropic;
   if (mali_physical_device_limited_clper(pdev)) {
      /* The two features whose shaders need read_invocation, which the
       * G51 and G31 cannot run (mali_physical_device_limited_clper):
       * non-uniform sampled-image indexing, and texel offsets that are not
       * constants (the Offset and ConstOffsets operands, which can differ
       * between invocations). */
      features.shaderSampledImageArrayNonUniformIndexing = false;
      features.shaderImageGatherExtended = false;
   }
   get_properties(pdev, name, &props);

   const struct vk_device_extension_table extensions =
      device_extensions_for_arch(pdev->arch);

   struct vk_physical_device_dispatch_table dispatch;
   vk_physical_device_dispatch_table_from_entrypoints(
      &dispatch, &mali_physical_device_entrypoints, true);
#ifndef __ANDROID__
   vk_physical_device_dispatch_table_from_entrypoints(
      &dispatch, &wsi_physical_device_entrypoints, false);
#endif

   VkResult result = vk_physical_device_init(&pdev->vk, &instance->vk,
                                             &extensions, &features,
                                             &props, &dispatch);
   if (result != VK_SUCCESS) {
      vk_free(&instance->vk.alloc, pdev);
      return result;
   }
   vk_physical_device_dispatch_table_from_entrypoints(
      &pdev->vk.dispatch_table, &mali_waist_physical_device_entrypoints, false);

   pdev->vk.supported_sync_types = pdev->arch == 7    ? mali_v7_sync_types()
                                   : pdev->arch == 9  ? mali_v9_sync_types()
                                   : pdev->arch == 10 ? mali_v10_sync_types()
                                                      : mali_v11_sync_types();

   /* Compiled shaders are what pipeline caches hold. */
   pdev->vk.pipeline_cache_import_ops =
      pdev->arch == 7    ? mali_v7_pipeline_cache_import_ops
      : pdev->arch == 9  ? mali_v9_pipeline_cache_import_ops
      : pdev->arch == 10 ? mali_v10_pipeline_cache_import_ops
                         : mali_v11_pipeline_cache_import_ops;

#ifndef __ANDROID__
   /* Last: wsi_device_init asks the runtime what the device can export,
    * which needs the sync types and pipeline-cache ops above. */
   result = mali_wsi_init(pdev);
   if (result != VK_SUCCESS) {
      vk_physical_device_finish(&pdev->vk);
      vk_free(&instance->vk.alloc, pdev);
      return result;
   }
#endif

   *out = &pdev->vk;
   return VK_SUCCESS;
}

VkResult
mali_physical_devices_enumerate(struct vk_instance *vk_instance)
{
   struct mali_instance *instance = container_of(vk_instance, struct mali_instance, vk);
   const bool os_backend = instance->kbase_backend == mali_kbase_os_backend();

   for (unsigned i = 0; i < MALI_MAX_PHYSICAL_DEVICES; i++) {
      char path[32];
      snprintf(path, sizeof(path), "/dev/mali%u", i);

      /* The blob checks each node with access() first. With a test
       * backend the node exists only in the fake, so we ask it. */
      if (os_backend && access(path, F_OK) != 0)
         continue;

      struct vk_physical_device *pdev;
      VkResult result = physical_device_create(instance, path, &pdev);
      if (result == VK_ERROR_INCOMPATIBLE_DRIVER)
         continue;
      if (result != VK_SUCCESS)
         return result;
      list_addtail(&pdev->link, &vk_instance->physical_devices.list);
   }
   return VK_SUCCESS;
}

void
mali_physical_device_destroy(struct vk_physical_device *vk_pdev)
{
   struct mali_physical_device *pdev =
      container_of(vk_pdev, struct mali_physical_device, vk);
#ifndef __ANDROID__
   mali_wsi_finish(pdev);
#endif
   vk_physical_device_finish(&pdev->vk);
   vk_free(&pdev->vk.instance->alloc, pdev);
}
