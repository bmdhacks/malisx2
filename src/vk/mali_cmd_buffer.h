/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Command buffers. Every vkCmd* writes its command-stream instructions
 * when it is recorded, with Mesa's cs_builder, into one stream per
 * subqueue; vkQueueSubmit only CALLs the recorded streams from the kernel
 * rings (mali_queue.c).
 *
 * Pools and the command-buffer objects are Mesa's runtime (vk_command_pool,
 * vk_command_buffer); this file set is the recording.
 *
 * Seam for render passes and draws: the vertex/tiler and fragment streams
 * exist in every command buffer (cs[MALI_SUBQUEUE_VERTEX_TILER],
 * cs[MALI_SUBQUEUE_FRAGMENT]) and are CALLed like the compute stream when
 * they are not empty; the subqueue context holds the tiler heap and
 * geometry buffer; barriers already handle all three subqueues;
 * mali_cmd_alloc gives descriptor memory; graphics descriptor sets are
 * bound into gfx.desc.
 */

#ifndef MALI_CMD_BUFFER_H
#define MALI_CMD_BUFFER_H

#include <stdint.h>

#include "util/list.h"
#include "util/u_dynarray.h"
#include "vk_command_buffer.h"

#include "kbase/kbase.h"

#include "mali_cmd_gfx.h"
#include "mali_cs.h"
/* After mali_cs.h, which sets PAN_ARCH: MALI_PER_ARCH below. */
#include "mali_arch.h"
#include "mali_descriptor_set_layout.h"
#include "mali_shader.h"

struct mali_descriptor_set;
struct mali_device;
struct vk_pipeline;

/* A cs_builder chunk: 16 KiB = 2048 instructions. Command-buffer memory
 * (slabs, struct mali_ptr) is described in mali_cmd_gfx.h. */
#define MALI_CS_CHUNK_SIZE (16 * 1024)

struct mali_cache_flush {
   enum mali_cs_flush_mode l2;
   enum mali_cs_flush_mode lsc;
   enum mali_cs_other_flush_mode others;
};

struct mali_cmd_cs {
   struct cs_builder b;
   /* Barrier signals this stream made since the progress registers were
    * last brought up to date (panvk's relative sync point). */
   int32_t relative_sync_point;
   /* Fragment stream only: the scoreboard wait and cache flush that
    * barriers asked of it and that are put off until its next fragment
    * job (mali_cmd_emit_barrier, mali_cmd_frag_flush_pending). */
   uint32_t pending_wait;
   struct mali_cache_flush pending_flush;
   /* Fragment stream only: a barrier signal on its counter, put off with
    * the wait and flush above (its number is already counted in
    * relative_sync_point). */
   bool pending_signal;

   /* Barrier bookkeeping for leaving out waits that are already satisfied
    * (mali_cmd_emit_barrier). Values are relative sync points of this
    * command buffer (1 = its first signal on that subqueue).
    *
    * idle: nothing has been recorded on this stream since its last barrier
    *       signal, so that signal still covers all its work (false at the
    *       start: earlier command buffers may have left work running);
    * flushed: the cache clean that went with that signal;
    * carried[j]: the highest signal of subqueue j this stream waited on
    *       since its last signal (not covered by that signal);
    * waited[j]: the highest signal of subqueue j this stream waited on in
    *       this command buffer. */
   bool idle;
   struct mali_cache_flush flushed;
   int32_t carried[MALI_SUBQUEUE_COUNT];
   int32_t waited[MALI_SUBQUEUE_COUNT];

   /* Compute stream only: barrier waits put off until the stream records
    * work (mali_cmd_compute_flush_waits). lazy[j]: a signal of subqueue j
    * already made or counted; lazy_frag: a fragment-subqueue signal that
    * does not exist yet, covering the fragment work recorded before the
    * barrier, with the cache clean lazy_frag_flush. */
   int32_t lazy[MALI_SUBQUEUE_COUNT];
   bool lazy_pending;   /* any of lazy[] or lazy_frag */
   bool lazy_frag;
   struct mali_cache_flush lazy_frag_flush;
};

struct mali_cmd_buffer {
   struct vk_command_buffer vk;
   struct mali_device *dev;
   VkCommandBufferUsageFlags usage;

   /* GPU memory: slabs in use, the one being filled. */
   struct list_head slabs;
   struct mali_cmd_slab *cur;
   uint64_t cur_offset;
   /* cur's mapping and GPU address (NULL/0 without one), for the inline
    * part of mali_cmd_alloc. */
   uint8_t *cur_cpu;
   uint64_t cur_gpu;

   struct mali_cmd_cs cs[MALI_SUBQUEUE_COUNT];

   /* What the queue must order around this command buffer at submit
    * (mali_queue.c, "Compute-subqueue ordering across command buffers"):
    * compute_work: work was recorded on the compute stream;
    * compute_prior_mask: bit k, stream k relies on the compute subqueue's
    *    work from before this command buffer being complete (the compute
    *    stream starts every command buffer as idle, with no signal);
    * compute_end_lazy: bit j, the compute subqueue still had waits for
    *    subqueue j put off when the command buffer ended; its next work
    *    must follow everything j recorded up to here. */
   bool compute_work;
   uint32_t compute_end_lazy;
   uint32_t compute_prior_mask;

   uint8_t push_constants[MALI_MAX_PUSH_CONSTANTS_SIZE];

   /* Storage behind vk.dynamic_graphics_state.vi / .ms.sample_locations. */
   struct vk_vertex_input_state dyn_vi;
   struct vk_sample_locations_state dyn_sl;

   struct {
      const struct mali_shader *shader;
      struct mali_desc_state desc;
   } compute;

   struct {
      struct mali_desc_state desc;
      /* Pipeline, vertex/index buffers, dirty registers (mali_cmd_draw.c). */
      struct mali_gfx_draw_state draw;
      /* The render pass instance being recorded (mali_cmd_render.c). */
      struct mali_render_state render;
   } gfx;

   /* Cached copies of push descriptor sets (outside gfx/compute, which a
    * reset zeroes; freed at destroy). */
   struct mali_push_shadows push_shadow;

   /* Thread-local storage shared by every dispatch of the command buffer
    * (per-thread slots, so concurrent dispatches do not collide). */
   struct {
      uint32_t size;        /* per-thread bytes the buffer was sized for */
      uint64_t gpu;
   } tls;

   /* RUN_COMPUTEs, RUN_IDVSs and render passes recorded (tests,
    * statistics). */
   uint32_t dispatches;
   uint32_t draws;
   uint32_t passes;

   /* Timing regions (src/measurement/); NULL unless timing is on. */
   struct mali_measure_cmd *measure;

   /* Occlusion queries that ended inside the open render pass: the GPU
    * addresses (uint64_t) of their availability words, set behind the
    * pass's fragment jobs (mali_cmd_query.c). */
   struct util_dynarray oq_done;
};

VK_DEFINE_HANDLE_CASTS(mali_cmd_buffer, vk.base, VkCommandBuffer,
                       VK_OBJECT_TYPE_COMMAND_BUFFER)

extern const struct vk_command_buffer_ops mali_cmd_buffer_ops;

void mali_cmd_compute_flush_waits(struct mali_cmd_buffer *cmd);

static inline struct cs_builder *
mali_cmd_cs(struct mali_cmd_buffer *cmd, enum mali_subqueue sq)
{
   /* Every caller records work: the stream's last barrier signal no longer
    * covers everything it did, and the waits put off until it records
    * work are due. */
   if (sq == MALI_SUBQUEUE_COMPUTE) {
      if (unlikely(cmd->cs[sq].lazy_pending))
         mali_cmd_compute_flush_waits(cmd);
      cmd->compute_work = true;
   }
   cmd->cs[sq].idle = false;
   return &cmd->cs[sq].b;
}

/* The recorded stream of subqueue sq: its first chunk and that chunk's size
 * in bytes. False when the stream is empty. */
bool mali_cmd_buffer_span(struct mali_cmd_buffer *cmd, unsigned sq,
                          uint64_t *addr, uint32_t *size);

/* Bit i: subqueue i has recorded work. */
uint32_t mali_cmd_buffer_subqueue_mask(struct mali_cmd_buffer *cmd);


/* size bytes of CPU-mapped, CPU-uncached GPU memory at a multiple of align
 * (a power of two), alive until the command buffer is reset. On failure
 * the command buffer records VK_ERROR_OUT_OF_DEVICE_MEMORY and {0} comes
 * back. Inline for the common case, a piece of the current slab (a draw
 * makes several of these). */
static inline struct mali_ptr
mali_cmd_alloc(struct mali_cmd_buffer *cmd, uint64_t size, uint64_t align)
{
   const uint64_t off = (cmd->cur_offset + align - 1) & ~(align - 1);
   if (likely(cmd->cur_cpu && size && size <= MALI_CMD_SLAB_SIZE / 2 &&
              off + size <= MALI_CMD_SLAB_SIZE)) {
      cmd->cur_offset = off + size;
      return (struct mali_ptr){cmd->cur_cpu + off, cmd->cur_gpu + off};
   }
   return MALI_PER_ARCH(cmd_alloc_slow)(cmd, size, align);
}

/* Frees the device's cached slabs. */
void mali_cmd_slabs_finish(struct mali_device *dev);

/* vkCmdBindPipeline, and the pipeline ops' bind (mali_pipeline.c). */
void MALI_PER_ARCH(cmd_bind_pipeline)(struct mali_cmd_buffer *cmd, struct vk_pipeline *pipeline);

/* ---------------------------------------------------------------------- */
/* Barriers (mali_cmd_buffer.c), shared with later command code            */

struct mali_cs_deps {
   struct {
      uint32_t wait_sb_mask;
      struct mali_cache_flush flush;
   } src[MALI_SUBQUEUE_COUNT];
   struct {
      uint32_t wait_subqueue_mask;
   } dst[MALI_SUBQUEUE_COUNT];
};

void mali_cmd_add_deps(struct mali_cmd_buffer *cmd, const VkDependencyInfo *info,
                       struct mali_cs_deps *deps);
void mali_cmd_emit_barrier(struct mali_cmd_buffer *cmd, const struct mali_cs_deps *deps);
/* Emits the fragment stream's put-off barrier wait and flush, if any.
 * Called before a fragment job and at the end of the command buffer. */
void mali_cmd_frag_flush_pending(struct mali_cmd_buffer *cmd);

/* Compute dispatch (mali_cmd_dispatch.c) and the internal copy shaders
 * (mali_cmd_copy.c): mali_cmd_gfx.h, shared with the job-manager back
 * half. */

#endif
