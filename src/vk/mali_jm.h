/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The job-manager (JM) back half of a VkDevice, for the Mali-G57 (arch v9)
 * and the Mali-G52 (arch v7, Bifrost). Compiled at PAN_ARCH 7 and 9; the
 * CSF counterpart is mali_queue.h.
 *
 * What the command-buffer side and the queue agree on:
 *
 *  - A command buffer is a list of batches (struct mali_jm_batch). A batch
 *    is one vertex/tiler/compute ("vtc") job chain and zero or more
 *    fragment segments, each a fragment job chain. Each non-empty chain
 *    becomes one kbase atom at submit: the vtc chain on job slot 1
 *    (core_req CS|T|CF|COHERENT_GROUP), a fragment segment on slot 0
 *    (core_req FS). The chains are finished when recorded; submit only
 *    numbers atoms and fills in their dependencies.
 *  - Ordering the recorder asks for, per batch: vtc_after_frag (the vtc
 *    chain must wait for fragment work recorded earlier: in an earlier
 *    batch of the same command buffer, or anywhere before), frag_after_vtc
 *    (a batch with no vtc chain whose fragment work must wait for earlier
 *    vtc work), and heap_slot (the tiler heap region the batch tiles into;
 *    its vtc chain waits for the region's previous user's fragment work).
 *
 * The queue's rules (mali_jm_queue.c): each atom depends on the previous
 * atom of its slot, so each slot runs in submission order, as a CSF ring
 * does; a fragment atom depends on its batch's vtc atom; a vtc atom's one
 * other dependency is the newest fragment-slot atom it must wait for.
 * Dependencies are DATA (a fault fails what depends on it), except on a
 * FENCE_WAIT atom (ORDER). Atoms in a slot order carry EVENT_COALESCE; each JOB_SUBMIT
 * ends in one that does not and that waits for the others, and the last
 * one of a submission (its "tracker") says "submission N and everything
 * before it is done". Completion is read on demand by whichever thread
 * needs it, with no event thread.
 *
 * Locking: everything here is under mali_device::lock.
 */

#ifndef MALI_JM_H
#define MALI_JM_H

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "util/simple_mtx.h"
#include "util/u_dynarray.h"
#include "vk_command_buffer.h"
#include "vk_queue.h"
#include "vk_sync.h"

#include "kbase/kbase.h"

#if !defined(PAN_ARCH) || (PAN_ARCH != 7 && PAN_ARCH != 9)
#error "mali_jm.h is the job manager back half: build with PAN_ARCH=7 or 9"
#endif
#include "genxml/gen_macros.h"

#include "mali_arch.h"
#include "mali_cmd_gfx.h"
#include "mali_shader.h"
#include "mali_vk.h"

struct mali_measure_cmd;
struct mali_jm_measure_capture;

/* Job slots (kbase_js_get_slot): fragment work runs on slot 0, everything
 * else we submit on slot 1. */
#define MALI_JM_SLOT_FRAG  0
#define MALI_JM_SLOT_VTC   1
#define MALI_JM_SLOT_COUNT 2
#define MALI_JM_SLOT_NONE  0xff

/* core_req of the two hardware atom kinds (the T820 blob's values,
 * confirmed on the device). */
#define MALI_JM_REQ_VTC  (KB_JM_REQ_CS | KB_JM_REQ_T | KB_JM_REQ_CF | KB_JM_REQ_COHERENT_GROUP)
#define MALI_JM_REQ_FRAG KB_JM_REQ_FS

/* ---------------------------------------------------------------------- */
/* Command buffers: batches and job chains                                 */

/* mali_jm_chain::pending: what the chain's next job has to wait for (an
 * in-chain barrier). BARRIER: the next job sets the
 * header's Barrier bit (it starts after every earlier job of the chain).
 * FLUSH: a Cache Flush job with Barrier and "Invalidate Shader Core Other"
 * goes first, for reads through a shader core's read-only caches (texture,
 * attribute) of what earlier jobs wrote. */
#define MALI_JM_PENDING_BARRIER (1u << 0)
#define MALI_JM_PENDING_FLUSH   (1u << 1)

/* One job chain = one atom. */
struct mali_jm_chain {
   uint64_t first;          /* GPU VA of the first job; 0: empty */
   uint32_t *prev_next;     /* CPU pointer to the last header's Next (written only) */
   uint16_t index;          /* last job index (Mesa's scheme: unique, increasing) */
   uint16_t tiler_dep;      /* index of the last tiler-side job */
   uint32_t jobs;
   uint8_t pending;         /* MALI_JM_PENDING_* */
   /* The chain writes system timestamps: its atom asks for PERMON, without
    * which the kernel leaves the GPU's timestamp counter stopped. */
   bool timestamps;
};

/* A run of fragment jobs submitted as one fragment atom. */
struct mali_jm_frag_seg {
   struct mali_jm_chain chain;
};

/* mali_jm_batch::vtc_after_frag: 0 = no requirement; this value = every
 * fragment-slot atom submitted before this batch (the barrier's first
 * scope includes earlier submissions); otherwise 1 + the index of a batch
 * of the same command buffer whose fragment work must finish first. */
#define MALI_JM_AFTER_NONE    0
#define MALI_JM_AFTER_EARLIER 0xffff
/* Every earlier fragment-slot atom of a batch with fragment-side transfer
 * work (mali_jm_batch::frag_xfer): what a barrier whose fragment-slot
 * source stages are only transfer stages waits for. */
#define MALI_JM_AFTER_EARLIER_XFER 0xfffe

struct mali_jm_batch {
   struct mali_jm_chain vtc;
   /* Fragment segments: frag_count entries of mali_jm_cmd::frags from
    * frag_first (indices, not a pointer: the array grows while recording). */
   uint32_t frag_first;
   uint32_t frag_count;
   int16_t heap_slot;       /* -1: the batch tiles nothing */
   uint16_t vtc_after_frag; /* MALI_JM_AFTER_* or 1 + batch index */
   bool frag_after_vtc;     /* no vtc chain of its own, but its fragment work
                               must follow the vtc work submitted before it */
   bool frag_xfer;          /* its fragment segments carry transfer work
                               (MALI_JM_AFTER_EARLIER_XFER) */
   bool vtc_xfer;           /* its vtc chain has a transfer job (a compute
                               copy, fill or update): it meets a transfer-
                               only requirement carried from an earlier
                               command buffer (mali_jm_queue::xvtc_frag) */
   uint32_t passes, draws;
   uint64_t est_heap_bytes; /* estimated tiler heap use, for batch closing */
   uint64_t heap_desc;      /* the Tiler Heap descriptor every pass of the
                               batch tiles through (0 until one tiles) */
};

/*
 * Batch limits. A batch is closed before a job that would take its vtc
 * chain past MALI_JM_BATCH_MAX_JOBS jobs, outside a render pass (a pass's
 * tiling stays in one batch; the render pass side checks the limit when a
 * pass begins). The limit keeps a chain's GPU time far below the T820
 * kernel's 1 s hard-stop (soft-stops resume a chain at a job boundary, so
 * only one long job is at risk) and the 16-bit job index far from
 * wrapping. 1024 vtc jobs are 3-6 ms of tiling at ARMSX2's draw sizes,
 * tuned on the device.
 */
#define MALI_JM_BATCH_MAX_JOBS   1024
#define MALI_JM_BATCH_MAX_PASSES 16

/*
 * Inside a render pass the batch cannot close, so one pass's tiling would
 * be one serial chain of any length: past about 65,530 jobs the 16-bit job
 * index runs out (the command buffer fails), and long before that one atom
 * runs for long enough that only soft-stop keeps it under the kernel's 1 s
 * hard-stop. So a pass's vtc chain is split every MALI_JM_PASS_MAX_JOBS
 * jobs (mali_jm_cmd_pass_vtc): the open batch ends there, and the pass
 * continues in a new batch on the same tiler contexts, heap slot and heap
 * descriptor (the tiler's state is in that memory). The new batch's vtc
 * atom follows the old one in the slot order, and its fragment work (the
 * rest of the pass) depends on it. ARMSX2's passes are far shorter; 8192
 * jobs are a few tens of milliseconds of tiling.
 */
#define MALI_JM_PASS_MAX_JOBS 8192

/* mali_jm_cmd::end_req: ordering a barrier recorded in this command buffer
 * asks of work after it, in later command buffers (the queue carries it to
 * the next atom of the slot): the next vtc atom waits for the newest
 * fragment atom, the next fragment atom of a batch without a vtc chain
 * waits for the newest vtc atom. */
#define MALI_JM_REQ_VTC_AFTER_FRAG      (1u << 0)
#define MALI_JM_REQ_FRAG_AFTER_VTC      (1u << 1)
#define MALI_JM_REQ_VTC_AFTER_FRAG_XFER (1u << 2)
/* The same two requirements of barriers whose vtc-slot destination stages
 * are only transfer stages: only the next vtc chain with a transfer job
 * (mali_jm_batch::vtc_xfer) waits, not the next frame's tiling. */
#define MALI_JM_REQ_XVTC_AFTER_FRAG      (1u << 3)
#define MALI_JM_REQ_XVTC_AFTER_FRAG_XFER (1u << 4)

/* A GPU-written range to restore before a command buffer runs again
 * (mali_jm_cmd::resets): size bytes at dst, from data_off in
 * mali_jm_cmd::reset_data, or zero when data_off is UINT32_MAX. */
struct mali_jm_reset {
   void *dst;
   uint32_t data_off;
   uint32_t size;
};

#if PAN_ARCH >= 9
/*
 * The draw template (mali_jm_cmd_draw.c): a CPU-side image of the next
 * Malloc Vertex job in cached memory. The draw state's dirty groups
 * rewrite their words of it; a draw patches its own counts and pointers
 * and the header, and copies the 384 bytes into command memory in one
 * pass.
 */
struct mali_jm_draw_tmpl {
   uint32_t w[96] __attribute__((aligned(16)));   /* 16: the most vk_alloc guarantees */
   uint32_t hdr4;           /* header word 4 without the index: the job type */
   uint32_t prim0;          /* Primitive word 0 without the index type */
   uint32_t packet_stride;  /* Allocation's vertex packet stride, for the heap estimate */
   bool secondary;          /* the varying shader runs (Primitive "Secondary Shader") */
};
#else
/* A v7 draw's varying buffers, in the order of panvk's PANVK_VARY_BUF_*. */
#define MALI_V7_VARY_BUF_GENERAL 0
#define MALI_V7_VARY_BUF_POSITION 1
#define MALI_V7_VARY_BUF_PSIZ 2
#define MALI_V7_VARY_BUF_COUNT 3

/*
 * The v7 descriptor tables of one stage of a draw or dispatch
 * (mali_shader.h, "Bifrost (v7) descriptor model"): what the Draw
 * section's Uniform buffers, Textures, Samplers, Attribute buffers and
 * Attributes pointers take, plus the dynamic storage buffer table and the
 * set addresses the stage's system values carry. Built by
 * mali_v7_desc_prepare; the entries that come from the sets are written by
 * a copy job (mali_v7_desc_copy).
 */
struct mali_v7_desc_tables {
   /* MALI_V7_DESC_TABLE_*, 0 for a table the shader does not have: 8-byte
    * Uniform Buffer descriptors (the dynamic ones behind the others), the
    * images' Attribute Buffer pairs, Texture and Sampler descriptors (a
    * dummy Sampler for a shader with textures and no sampler). */
   uint64_t tables[MALI_V7_DESC_TABLE_COUNT];
   uint64_t img_attribs;                      /* the images' Attribute descriptors */
   uint64_t dyn_ssbos;                        /* the dynamic storage buffer table */
   uint64_t sets[MALI_MAX_SETS];              /* the bound sets' memory, 0 when unbound */
   uint32_t set_desc_counts[MALI_MAX_SETS];   /* their slot counts */
   /* The shader the tables were built for (their layout is its own). */
   const struct mali_shader *shader;
   /* The dynamic buffers they were built with, one per entry of the
    * shader's dynamic buffer map: the address with the bound offset
    * applied, and the range (mali_v7_desc_refresh). */
   uint64_t dyn_addrs[MALI_MAX_DYNAMIC_BUFFERS];
   uint64_t dyn_ranges[MALI_MAX_DYNAMIC_BUFFERS];
};

/*
 * The v7 draw state (mali_jm_cmd_draw_v7.c): what a draw's Vertex and Tiler
 * jobs point at that outlives one draw, rebuilt by the dirty groups named
 * below. A draw packs its two jobs from these and its own counts.
 */
struct mali_jm_draw_tmpl {
   /* Per pipeline (MALI_GFX_DIRTY_PIPELINE). */
   uint32_t prim0;          /* Primitive word 0 without index type and provoking vertex */
   uint32_t dcd0;           /* Draw word 0: the baked culling / front face / flat bits */
   uint64_t vs_rsd;         /* the vertex shader's Renderer State Descriptor */
   uint64_t fs_code;        /* the fragment shader's code address, 0 without one */
   uint64_t vs_vary;        /* varying Attribute descriptors the vertex shader writes */
   uint64_t fs_vary;        /* ... and the fragment shader reads */
   uint32_t vary_stride[MALI_V7_VARY_BUF_COUNT];
   enum mesa_prim prim;
   bool writes_psiz;        /* points with a shader-written size */
   bool lines;
   /* Per state group. */
   uint64_t fs_rsd;         /* fragment RSD + Blend descriptors (PIPELINE, BLEND, ZSD) */
   struct mali_v7_desc_tables vs_tab, fs_tab;   /* descriptor tables (VS_SRT, FS_SRT) */
   uint64_t vs_fau, fs_fau; /* push uniform blocks (VS_FAU, FS_FAU) */
   uint64_t viewport;       /* Viewport descriptor (VIEWPORT) */
   uint64_t tiler, tsd;     /* layer 0's Tiler Context, the pass's Local Storage (PASS) */
   int32_t raw_vertex_offset;   /* the value the VS FAU block was built with */
   /* The bound index buffer's CPU view (mali_cmd_state.c), NULL when none. */
   const uint8_t *ib_cpu;
};
#endif

#if PAN_ARCH < 9
/* How many draws later a draw's Tiler job goes into the chain
 * (mali_jm_cmd_draw_v7.c), and so how many can be pending. The job manager
 * starts a chain's jobs in order and holds a Tiler job, and every job
 * behind it, until the Tiler job's Vertex job is done. With 2 the Tiler
 * job goes in behind the Vertex jobs of the next two draws, which keep the
 * shader cores busy through that wait; with 1, behind only the next
 * draw's, which can finish first and leave them idle. */
#define MALI_V7_TILER_DEFER 2

/* A range [start, end) of GPU addresses. */
struct mali_v7_range {
   uint64_t start, end;
};
#endif

/* Inside the job-manager command buffer: what submit reads, and the open
 * batch. */
struct mali_jm_cmd {
   struct mali_jm_draw_tmpl draw;

   struct util_dynarray batches; /* struct mali_jm_batch, closed batches */
   struct util_dynarray frags;   /* struct mali_jm_frag_seg */

   /* The batch being recorded (when `open`); its fragment segments are
    * already in frags, from cur.frag_first. */
   struct mali_jm_batch cur;
   bool open;
   /* Barrier requirements recorded and not yet applied to a job of the
    * slot they order (mali_jm_cmd_buffer.c), and what is left of them at
    * the end of the command buffer (MALI_JM_REQ_*), for the queue. */
   uint16_t req;
   uint8_t end_req;

   /* Re-submission: command buffers recorded without ONE_TIME_SUBMIT
    * note every GPU-written word, and submit restores them before every
    * run after the first. */
   bool resubmit;
   bool submitted;
   uint64_t last_seq;            /* the submission that ran it last */
   struct util_dynarray resets;  /* struct mali_jm_reset */
   struct util_dynarray reset_data;

   /* Timing regions (measurement/timing_jm.c); NULL unless
    * MALISX2_MEASURE's "timing" mode is on. */
   struct mali_measure_cmd *measure;
#if PAN_ARCH < 9
   /* Tiler jobs written and waiting to be linked into the open batch's
    * vtc chain (mali_v7_link_tilers). Each goes in behind the Vertex job
    * of the draw MALI_V7_TILER_DEFER draws later, in front of any other
    * job, or at batch close, whichever comes first. dep1 is each one's
    * Vertex job. tiler_defer is set while a draw adds its copy jobs and
    * its Vertex job, the only jobs that go in ahead of the pending ones
    * (mali_jm_cmd_add_job). */
   struct mali_ptr pend_tiler[MALI_V7_TILER_DEFER];
   uint16_t pend_tiler_dep1[MALI_V7_TILER_DEFER];
   uint8_t pend_tilers;          /* pending, oldest first */
   bool tiler_defer;
   /* The Local Storage descriptor of the copy jobs (mali_jm_desc_v7.c
    * copy_tsd), 0 until the first one. */
   uint64_t copy_tsd;
   /* The buffer memory that the transfers recorded so far write
    * (mali_v7_note_buffer_write): struct mali_v7_range, sorted, disjoint
    * and not touching. Empty unless the command buffer has such a
    * transfer. */
   struct util_dynarray buf_writes;
#endif
};

/*
 * The job-manager command buffer: what submit reads (jm), and the
 * recording state the builders shared with the v11 back half read, under
 * the same member names as the v11 struct (mali_cmd_state.h lists them).
 */
struct mali_cmd_buffer {
   struct vk_command_buffer vk;
   struct mali_device *dev;
   struct mali_jm_cmd jm;

   /* GPU memory: slabs in use, the one being filled (mali_cmd_alloc). */
   struct list_head slabs;
   struct mali_cmd_slab *cur;
   uint64_t cur_offset;
   uint8_t *cur_cpu;
   uint64_t cur_gpu;

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
      /* regs/regs_valid are the v11 register record; unused here. */
      struct mali_gfx_draw_state draw;
      struct mali_render_state render;
   } gfx;

   /* Cached copies of push descriptor sets (outside gfx/compute, which a
    * reset zeroes; freed at destroy). */
   struct mali_push_shadows push_shadow;

   /* Thread-local storage shared by every job of the command buffer. */
   struct {
      uint32_t size;
      uint64_t gpu;
   } tls;

   uint32_t dispatches;
   uint32_t draws;
   uint32_t passes;

#if PAN_ARCH < 9
   /* The 1 MiB slab being filled with blocks over half a slab
    * (mali_jm_cmd_buffer.c). Like every slab it is on the slabs list,
    * which releases it at reset. */
   uint8_t *big_cpu;
   uint64_t big_gpu;
   uint64_t big_offset;
#endif
};

VK_DEFINE_HANDLE_CASTS(mali_cmd_buffer, vk.base, VkCommandBuffer,
                       VK_OBJECT_TYPE_COMMAND_BUFFER)

/* size bytes of CPU-mapped, CPU-uncached GPU memory at a multiple of align
 * (a power of two), alive until the command buffer is reset; {0} with the
 * command buffer's error set on failure. The same as the v11 one
 * (mali_cmd_buffer.h). */
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

/* vkCmdBindPipeline, and the pipeline ops' bind (mali_cmd_state.c). */
void MALI_PER_ARCH(cmd_bind_pipeline)(struct mali_cmd_buffer *cmd, struct vk_pipeline *pipeline);

extern const struct vk_command_buffer_ops MALI_PER_ARCH(cmd_buffer_ops);

static inline struct mali_jm_batch *
mali_jm_cmd_batches(struct mali_cmd_buffer *cmd, uint32_t *count)
{
   *count = util_dynarray_num_elements(&cmd->jm.batches, struct mali_jm_batch);
   return util_dynarray_begin(&cmd->jm.batches);
}

/* Append a closed batch (and its fragment segments) behind the batches
 * recorded so far. For tests that build chains by hand; the open batch, if
 * any, is closed first. Returns false on allocation failure. */
bool mali_jm_cmd_add_batch(struct mali_cmd_buffer *cmd, const struct mali_jm_batch *batch,
                           const struct mali_jm_frag_seg *frags, uint32_t frag_count);

/* ---------------------------------------------------------------------- */
/* Recording job chains (mali_jm_cmd_buffer.c)                             */

/* The open batch, opened if there is none. NULL on allocation failure
 * (the command buffer has the error). */
struct mali_jm_batch *mali_jm_cmd_batch(struct mali_cmd_buffer *cmd);

/* Close the open batch: it joins the closed batches if it has any job,
 * with the barrier requirements recorded for its work. */
void mali_jm_cmd_batch_close(struct mali_cmd_buffer *cmd);

/* The open batch's vtc chain, for a job outside a render pass: closes a
 * full batch first (MALI_JM_BATCH_MAX_JOBS). NULL on failure. */
struct mali_jm_chain *mali_jm_cmd_vtc(struct mali_cmd_buffer *cmd);

/* The same for a transfer done by compute (buffer and image copies, fills,
 * updates): it also waits for what barriers whose destination stages are
 * only transfer stages asked for. Plain vtc jobs (dispatches, draws) are
 * not in those barriers' second scope, so such a barrier does not close a
 * batch at the next draw. */
struct mali_jm_chain *mali_jm_cmd_vtc_transfer(struct mali_cmd_buffer *cmd);

/* Would a transfer job on the vtc chain close the open batch now (a
 * recorded barrier makes it wait for this batch's fragment work)? An
 * image copy then goes to the fragment side instead (mali_cmd_image.c). */
bool mali_jm_cmd_transfer_needs_frag(struct mali_cmd_buffer *cmd);

/* The open batch's current fragment segment, started if the batch has
 * none; with new_segment, always a new one (a new fragment atom). Applies
 * the barriers recorded for fragment work. NULL on failure. The vtc
 * variant above applies those for vtc work: call it before the first job
 * of a render pass's tiling (a batch can still close there, since nothing
 * of the pass is in it yet), and before each job outside a pass. */
struct mali_jm_chain *mali_jm_cmd_frag(struct mali_cmd_buffer *cmd, bool new_segment);

/* The open batch's fragment work includes a transfer (a fragment-side
 * copy, blit, resolve or image clear): barriers from transfer stages wait
 * for it. */
void mali_jm_cmd_mark_frag_transfer(struct mali_cmd_buffer *cmd);

/*
 * Append a job to chain c: size bytes (the whole job, its header included)
 * at 128-byte alignment from command memory, with the header written
 * (type, the next index, Barrier from the chain's pending barrier or
 * `barrier`, Dependency 1 = dep1, Dependency 2 = the previous tiler-side
 * job for a tiler-side job, Next = 0) and linked behind the chain's last
 * job. A pending Cache Flush job goes in first. On v7 the pending Tiler
 * jobs (struct mali_jm_cmd::pend_tiler) are linked into the open batch's
 * vtc chain before anything else, whatever chain c is, unless a draw holds
 * them back (tiler_defer). The caller writes the payload, every byte of it
 * (slabs are recycled without clearing). {0} on failure, with the command
 * buffer's error set.
 */
struct mali_ptr mali_jm_cmd_add_job(struct mali_cmd_buffer *cmd, struct mali_jm_chain *c,
                                    enum mali_job_type type, unsigned size, bool barrier,
                                    uint16_t dep1);

/*
 * The link half of mali_jm_cmd_add_job, for writers that pack the header
 * themselves (the draws copy whole jobs from a cached template): job, whose
 * header already has Index = `index` (taken with ++c->index) and, for a
 * tiler-side job, Dependency 2 = c->tiler_dep from before this call, goes
 * behind the chain's last job. Stores the job's address into the previous
 * header's Next; reads no command memory. The caller notes the header for
 * re-submission (mali_jm_cmd_note_reset) and handles c->pending first.
 */
static inline void
mali_jm_chain_link(struct mali_jm_chain *c, struct mali_ptr job, uint16_t index, bool tiler_side)
{
   if (c->prev_next)
      memcpy(c->prev_next, &job.gpu, sizeof(job.gpu));
   else
      c->first = job.gpu;
   c->prev_next = (uint32_t *)((uint8_t *)job.cpu + 24);
   c->jobs++;
   if (tiler_side)
      c->tiler_dep = index;
}

#if PAN_ARCH < 9
/* Links the pending Tiler jobs (struct mali_jm_cmd::pend_tiler), oldest
 * first, into the open batch's vtc chain until at most keep are left. */
void mali_v7_link_tilers(struct mali_cmd_buffer *cmd, unsigned keep);
#endif

/* A Write Value job on chain c (timestamps, availability, tests). */
bool mali_jm_cmd_write_value(struct mali_cmd_buffer *cmd, struct mali_jm_chain *c,
                             enum mali_write_value_type type, uint64_t addr, uint64_t value,
                             bool barrier);

/* Note size bytes at dst as GPU-written: restored from tmpl (or zeroed,
 * tmpl NULL) before every run after the first. Free for ONE_TIME_SUBMIT
 * command buffers. */
void mali_jm_cmd_note_reset(struct mali_cmd_buffer *cmd, void *dst, const void *tmpl,
                            uint32_t size);

/*
 * The JM barrier: what VkPipelineStageFlags2/VkAccessFlags2 on each side
 * require of the job chains: an in-chain barrier on the vtc or fragment
 * chain, a closed batch with the next one waiting for the fragment atom,
 * or a requirement the queue resolves at submit.
 * Outside a render pass; vkCmdPipelineBarrier2 handles the in-pass case.
 */
void mali_jm_cmd_barrier(struct mali_cmd_buffer *cmd, VkPipelineStageFlags2 src_stages,
                         VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stages,
                         VkAccessFlags2 dst_access);

/* Before a submission runs cmd: restore what earlier runs changed. With
 * dev->lock held, the command buffer not in flight. */
void mali_jm_cmd_prepare_submit(struct mali_cmd_buffer *cmd);

/* The draw template's words that no draw state owns (mali_jm_cmd_draw.c),
 * at command buffer creation. On v7 (mali_jm_cmd_draw_v7.c) it clears the
 * draw state, at creation and at every reset. */
void mali_jm_draw_tmpl_init(struct mali_jm_draw_tmpl *t);

/* ---------------------------------------------------------------------- */
/* Render passes (mali_jm_cmd_render.c) and the tiler heap (mali_jm_heap.c) */

/*
 * What a draw (mali_jm_cmd_draw.c on v9, mali_jm_cmd_draw_v7.c on v7) and
 * the internal draws of a pass use of the pass, after
 * MALI_PER_ARCH(cmd_render_tiler) has returned true for it:
 *  - the chain its jobs go into: mali_jm_cmd_pass_vtc (the open batch's
 *    vtc chain; the batch closes inside a pass only at the pass job limit,
 *    MALI_JM_PASS_MAX_JOBS, and the barriers owed to the pass's vtc work
 *    were applied by cmd_render_tiler, so a draw takes the chain as it
 *    is). Call it once before jobs that depend on each other, such as a v7
 *    draw's copy jobs, Vertex job and Tiler job: a split between them
 *    would leave a dependency pointing into the closed chain (the chain
 *    may pass the limit by those few jobs). Independent jobs, such as one
 *    Tiler job per layer, may take it once each;
 *  - one Tiler Context per layer: mali_jm_pass_tiler(r, layer), layers
 *    0 .. r->td_count - 1 (MALI_LAYERS_PER_TILER_CTX is 1 on v7 and v9);
 *  - the pass's Local Storage descriptor: r->tsd (filled at the end of the
 *    pass from r->tls_size, which the draw raises to its shaders' need);
 *  - the batch's heap estimate: mali_jm_cmd_heap_use(cmd, bytes) per draw,
 *    a guess at its use of the tiler heap: vertices x instances x (packet
 *    stride + 16) on v9 (vertex packets and polygon lists), vertices x
 *    instances x 32 on v7 (polygon lists only: the varyings are in command
 *    memory). The next pass's begin closes the batch past half a heap slot.
 */
void mali_jm_cmd_pass_split(struct mali_cmd_buffer *cmd);

static inline struct mali_jm_chain *
mali_jm_cmd_pass_vtc(struct mali_cmd_buffer *cmd)
{
   assert(cmd->jm.open && cmd->gfx.render.tiler);
   if (unlikely(cmd->jm.cur.vtc.jobs >= MALI_JM_PASS_MAX_JOBS))
      mali_jm_cmd_pass_split(cmd);
   return &cmd->jm.cur.vtc;
}

static inline uint64_t
mali_jm_pass_tiler(const struct mali_render_state *r, uint32_t layer)
{
   assert(layer < r->td_count);
   return r->tiler + (uint64_t)layer * pan_size(TILER_CONTEXT);
}

static inline void
mali_jm_cmd_heap_use(struct mali_cmd_buffer *cmd, uint64_t bytes)
{
   cmd->jm.cur.est_heap_bytes += bytes;
}

struct mali_jm_device;

/* The ring's lock; slots are allocated on first use. */
void mali_jm_heap_init(struct mali_jm_device *jd);
/* Frees every slot. The GPU must be idle. */
void mali_jm_heap_finish(struct mali_jm_device *jd);

/* Give the open batch b a heap slot (the next in the ring, allocated if it
 * is new) and its Tiler Heap descriptor in command memory (b->heap_slot,
 * b->heap_desc). False on failure, with the command buffer's error set. */
bool mali_jm_heap_take(struct mali_cmd_buffer *cmd, struct mali_jm_batch *b);

#if PAN_ARCH < 9
/* ---------------------------------------------------------------------- */
/* Bifrost descriptor tables (mali_jm_desc_v7.c)                           */

/*
 * Allocates a shader's tables in command memory and writes what the CPU
 * packs (panvk_v7 cmd_prepare_shader_desc_tables): the dynamic uniform
 * buffers behind the regular ones, the dynamic storage buffer table, the
 * dummy sampler and the bound sets' addresses. With img false the image
 * tables are left out: a vertex shader's images live behind the vertex
 * buffers in the draw's own Attribute Buffer and Attribute tables, which
 * the draw points tables[MALI_V7_DESC_TABLE_IMG] and img_attribs at. desc
 * may be NULL for a shader without descriptor sets. False on failure, with
 * the command buffer's error set.
 */
bool mali_v7_desc_prepare(struct mali_cmd_buffer *cmd, const struct mali_shader *s,
                          const struct mali_desc_state *desc, bool img,
                          struct mali_v7_desc_tables *out);

/*
 * Updates *t for new dynamic offsets when it was built for this shader and
 * the same bound sets: the dynamic uniform buffer entries and the dynamic
 * storage buffer table are packed anew and the copied entries kept, so the
 * draw needs no copy job. A bound set cannot change while the command
 * buffer records (no update-after-bind is exposed), and a push set gets
 * new memory with every push. *unchanged when the shader's dynamic buffers
 * (addresses with the offsets applied, and ranges) are the same too. False,
 * with *t unchanged, when the tables must be rebuilt
 * (another shader or set, regular uniform buffers, which share the table
 * with the dynamic ones and come from the copy, or a vertex shader with
 * images), or on an allocation failure, with the command buffer's error
 * set.
 */
bool mali_v7_desc_refresh(struct mali_cmd_buffer *cmd, const struct mali_shader *s,
                          const struct mali_desc_state *desc, bool img,
                          struct mali_v7_desc_tables *t, bool *unchanged);

/*
 * Writes the table entries that come from the sets (the shader's copy map)
 * with a Compute job on chain c, one invocation per entry (panvk_v7
 * meta_get_copy_desc_job). *index: that job's index, the Dependency 1 of
 * the job that reads the tables; 0 when there is nothing to copy.
 * attrib_buf_idx_offset: the buffer index the image Attribute descriptors
 * start at (twice the vertex binding count for a vertex shader, else 0).
 * False on failure, with the command buffer's error set.
 */
bool mali_v7_desc_copy(struct mali_cmd_buffer *cmd, struct mali_jm_chain *c,
                       const struct mali_shader *s, const struct mali_v7_desc_tables *t,
                       uint32_t attrib_buf_idx_offset, uint16_t *index);

/*
 * A shader's uniform buffer words in FAU (mali_shader.c lower_ubo_push): a
 * Compute job on chain c, depending on job dep (0: none), that copies them
 * from the bound dynamic uniform buffers into the draw's FAU block at
 * fau_va (the block's address, without the count in the top byte). *index
 * (when not NULL): that job's index, or dep when there is nothing to copy.
 * False on failure, with the command buffer's error set.
 */
bool mali_v7_ubo_copy(struct mali_cmd_buffer *cmd, struct mali_jm_chain *c,
                      const struct mali_shader *s, const struct mali_desc_state *desc,
                      uint64_t fau_va, uint16_t dep, uint16_t *index);

/* The v7_desc.sets system values of one stage: the addresses of the bound
 * sets the shader reads and, at MALI_V7_SETS_DYN_SSBOS, its dynamic
 * storage buffer table. */
void mali_v7_desc_sysvals(const struct mali_v7_desc_tables *t, const struct mali_shader *s,
                          uint64_t sets[MALI_MAX_SETS + 1]);

/*
 * The Invocation section of groups[] workgroups of local[] invocations
 * (Mesa pan_encoder.h pan_pack_work_groups_compute, without indirect
 * dispatch). graphics is Mesa's quirk_graphics, for a draw's Vertex and
 * Tiler jobs: a workgroup Z shift of 32 for one instance and the minimum
 * efficient thread group split. Otherwise the split equals the workgroup
 * X shift, which barriers need.
 */
static inline void
mali_v7_pack_invocation(struct mali_invocation_packed *out, const uint32_t local[3],
                        const uint32_t groups[3], bool graphics)
{
   const unsigned values[6] = {local[0], local[1], local[2], groups[0], groups[1], groups[2]};
   unsigned shifts[7] = {0};
   uint32_t packed = 0;

   for (unsigned i = 0; i < 6; i++) {
      assert(values[i] >= 1);
      packed |= (values[i] - 1) << shifts[i];
      shifts[i + 1] = shifts[i] + util_logbase2_ceil(values[i]);
   }
   pan_pack(out, INVOCATION, cfg) {
      cfg.invocations = packed;
      cfg.size_y_shift = shifts[1];
      cfg.size_z_shift = shifts[2];
      cfg.workgroups_x_shift = shifts[3];
      cfg.workgroups_y_shift = shifts[4];
      cfg.workgroups_z_shift = (graphics && groups[2] <= 1) ? 32 : shifts[5];
      cfg.thread_group_split = graphics ? MALI_SPLIT_MIN_EFFICIENT : cfg.workgroups_x_shift;
   }
}

/* Compute job sizing: the Invocation section of a dispatch or copy job
 * (no graphics quirks) and the Parameters' job task split. */
static inline void
mali_v7_compute_invocation(struct mali_invocation_packed *out, const uint32_t local[3],
                           const uint32_t groups[3])
{
   mali_v7_pack_invocation(out, local, groups, false);
}

/* panvk's job_task_split for a dispatch (jm/panvk_vX_cmd_dispatch.c). */
static inline unsigned
mali_v7_compute_task_split(const uint32_t local[3])
{
   return util_logbase2_ceil(local[0] + 1) + util_logbase2_ceil(local[1] + 1) +
          util_logbase2_ceil(local[2] + 1);
}

/* Frees the device's descriptor and uniform buffer word copy shaders. */
void mali_v7_desc_finish(struct mali_device *dev);

/* ---------------------------------------------------------------------- */
/* Internal draws inside a Bifrost pass (mali_jm_cmd_render.c)             */

/*
 * v7 has no Fullscreen job: an internal draw inside the current pass
 * (vkCmdClearAttachments' fragment shader, the in-pass fragment barrier's
 * Renderer State Descriptor without a shader) is one Tiler job per layer
 * in [base_layer, base_layer + layer_count), drawing rect as a 4-vertex
 * triangle strip from positions the CPU writes, with no Vertex job in
 * front. draw is the Draw section's template (a struct mali_draw_packed);
 * each job gets the rectangle as its position and scissor. barrier sets
 * the job header's Barrier bit.
 */
void mali_v7_cmd_meta_quads(struct mali_cmd_buffer *cmd, const void *draw, const VkRect2D *rect,
                            uint32_t base_layer, uint32_t layer_count, bool barrier);

/* The four positions (x, y, 0, 1) of the rectangle [x0, x1) x [y0, y1) as
 * a triangle strip, in command memory (mali_cmd_meta_gfx.c). 0 on failure. */
uint64_t mali_v7_meta_rect_positions(struct mali_cmd_buffer *cmd, float x0, float y0, float x1,
                                     float y1);

/* A Viewport descriptor whose scissor is [minx, maxx] x [miny, maxy]
 * (inclusive, each bound clamped to 16 bits) and whose depth range is 0
 * to 1, in command memory (mali_cmd_meta_gfx.c). 0 on failure. */
uint64_t mali_v7_meta_viewport(struct mali_cmd_buffer *cmd, uint32_t minx, uint32_t miny,
                               uint32_t maxx, uint32_t maxy);

/* ---------------------------------------------------------------------- */
/* Indexed draws (mali_jm_cmd_draw_v7.c)                                   */

/*
 * Notes that a transfer recorded in cmd writes the size bytes of buffer
 * memory at GPU address addr (struct mali_jm_cmd::buf_writes); the buffer
 * copies, fills and updates and the image to buffer copies call it. The
 * v7 draw reads an indexed draw's indices on the CPU when the draw is
 * recorded, before such a write runs, and skips a draw whose indices one
 * of them writes (index_range). On failure the command buffer's error is
 * set.
 */
void mali_v7_note_buffer_write(struct mali_cmd_buffer *cmd, uint64_t addr, uint64_t size);
#endif /* PAN_ARCH < 9 */

/* ---------------------------------------------------------------------- */
/* The device and queue                                                    */

/* A reference to a submitted atom that stays meaningful after its number
 * is reused: (serial << 8) | number. Serials increase with every atom. 0
 * names nothing. */
typedef uint64_t mali_jm_ref;

static inline uint8_t
mali_jm_ref_num(mali_jm_ref r)
{
   return (uint8_t)r;
}

static inline uint64_t
mali_jm_ref_serial(mali_jm_ref r)
{
   return r >> 8;
}

enum mali_jm_atom_kind {
   MALI_JM_ATOM_VTC = 1,
   MALI_JM_ATOM_FRAG,
   MALI_JM_ATOM_JOIN,           /* dependency-only (core_req 0) */
   MALI_JM_ATOM_FENCE_WAIT,     /* soft: waits for an imported sync file */
   MALI_JM_ATOM_FENCE_TRIGGER,  /* soft: signals a sync file we hand out */
};

/* udata[1] of every atom: its kind, flags, and where it came from (for
 * the device-loss message). udata[0] is the submission number. */
#define MALI_JM_UDATA_KIND(u)   ((unsigned)((u) & 0xff))
#define MALI_JM_UDATA_TRACKER   (1ull << 8)   /* completes submission udata[0] */
#define MALI_JM_UDATA_TERMINAL  (1ull << 9)   /* last atom of a JOB_SUBMIT call */
#define MALI_JM_UDATA_BATCH(u)  ((unsigned)(((u) >> 16) & 0xffff))
#define MALI_JM_UDATA_CMD(u)    ((unsigned)(((u) >> 32) & 0xffff))

/* What the driver knows about one atom number. */
struct mali_jm_atom_info {
   mali_jm_ref ref;         /* the atom holding the number now (or last) */
   bool in_flight;          /* numbered and its event not read yet */
   uint8_t kind;            /* enum mali_jm_atom_kind */
   uint8_t slot;            /* MALI_JM_SLOT_*: in that slot's order, or NONE */
   /* Per slot, the newest atom in that slot's order this one waits for,
    * directly or through other atoms (itself, for its own slot). */
   mali_jm_ref cov[MALI_JM_SLOT_COUNT];
};

/*
 * The tiler heap ring (mali_jm_heap.c). The tiler allocates polygon lists
 * from the heap, and on v9 vertex packets (varyings) too; v7 keeps its
 * varyings in command memory. The kernel does not order atoms that share
 * memory, so a heap region is reused only after the fragment work that
 * reads it. The ring has MALI_JM_HEAP_SLOTS regions; a batch that tiles
 * takes the next one when its first pass starts tiling, and every pass and
 * layer of the batch tiles through one Tiler Heap descriptor in its
 * command memory. The queue keeps, per slot, the last fragment atom that
 * read it, and the vtc atom of the next batch using the slot waits for it.
 *
 * Each slot is a separate growable allocation (a GROW_ON_GPF fault commits
 * every page up to the faulting one, so windows inside one region would
 * commit the gaps between them): GPU read/write, no CPU access,
 * TILER_ALIGN_TOP, 2 MiB committed, 2 MiB growth steps, 256 MiB of VA (the
 * T820 blob's big JIT heaps are 360 MiB).
 */
#define MALI_JM_HEAP_SLOTS_MAX 32
#define MALI_JM_HEAP_SLOTS 4
#define MALI_JM_HEAP_CHUNK (2ull << 20)
#define MALI_JM_HEAP_SLOT_SIZE (256ull << 20)

struct mali_jm_heap {
   simple_mtx_t lock;
   uint32_t next;           /* the slot the next batch takes */
   struct mali_kbase_bo bo[MALI_JM_HEAP_SLOTS];
};

#define MALI_JM_EVENTS_PER_READ 32

struct mali_jm_queue;
struct mali_jm_build;

struct mali_jm_device {
   struct mali_device *dev;
   struct mali_kbase *kb;

   /* Atom numbers (1..255) and what each one is. A number is freed when
    * its event is read, never earlier, so a dependency on an atom whose
    * event has not been read names that atom. */
   struct mali_kbase_jm_atom_ids ids;
   unsigned free_ids;
   uint64_t next_serial;
   struct mali_jm_atom_info atoms[256];
   unsigned in_flight;

   /* One thread at a time blocks in poll() on the kbase fd (the
    * "reader"); the others sleep on mali_device::cond, which the reader
    * broadcasts. Non-blocking reads happen under the lock, when no reader
    * is polling. */
   bool reader;
   /* A thread is building and submitting atoms. Set while it may drop
    * the lock to wait for atom numbers, so no other thread numbers atoms
    * in between. */
   bool building;

   /* A fault read from an event that is not reported yet (the message
    * for mali_device_set_lost, which is called without the lock). */
   bool fault;
   char fault_msg[192];

   /* Job slot 0 (fragment) runs Cache Flush jobs (JS0_FEATURES bit 3).
    * Without it, a fragment-side barrier that needs a cache flush starts a
    * new fragment atom instead (mali_jm_cmd_frag). */
   bool frag_cache_flush;

   /* STREAM_CREATE's timeline fd for FENCE_TRIGGER (first use), or -1. */
   int stream_fd;

   /* The device's one queue (MALI_QUEUE_COUNT is 1). */
   struct mali_jm_queue *queue;

   /* Tiler heap regions; their own lock (recording threads take slots). */
   struct mali_jm_heap heap;

   /* Atoms of the JOB_SUBMIT being built (only while building). */
   struct mali_jm_build *build;

#if PAN_ARCH < 9
   /* The descriptor copy shader (mali_jm_desc_v7.c), compiled on first
    * use under mali_device::meta_lock, unreferenced at device_finish. */
   struct mali_shader *v7_desc_copy;
   /* The uniform buffer word copy shader (mali_v7_ubo_copy), the same way. */
   struct mali_shader *v7_ubo_copy;
   /* How many of mali_device::free_slabs are 1 MiB slabs for large blocks
    * (mali_jm_cmd_buffer.c); under mali_device::slab_lock. */
   unsigned free_big_slabs;
#endif

   /* Set for the duration of one vkQueueSubmit when MALISX2_MEASURE's
    * "csf" mode is on (recorder_jm.c calls it "csf" too: one capture
    * knob); flush() (mali_jm_queue.c) appends every JOB_SUBMIT call's
    * atoms to it. */
   struct mali_jm_measure_capture *measure_cap;

   struct {
      uint64_t atoms, job_submits, events, joins, partial_flushes, number_waits;
      uint64_t fence_waits, fence_triggers, fence_wait_errors;
   } stats;
};

struct mali_jm_queue {
   struct mali_device *dev;
   /* The newest atom in each slot's order. */
   mali_jm_ref last[MALI_JM_SLOT_COUNT];
   /* Per tiler heap slot, the last fragment atom that used it. */
   mali_jm_ref heap_last_frag[MALI_JM_HEAP_SLOTS_MAX];
   /* Submissions with atoms so far, the last one's tracker, and the
    * newest submission whose tracker event has been read. */
   uint64_t seq;
   mali_jm_ref tracker;
   uint64_t completed_seq;
   /* The newest fragment atom of a batch with fragment-side transfer
    * work (MALI_JM_AFTER_EARLIER_XFER). */
   mali_jm_ref last_frag_xfer;
   /* Barrier requirements (MALI_JM_REQ_*) recorded at the end of a
    * command buffer and not yet met by a later atom; the transfer-only
    * ones are kept as xvtc_frag instead. */
   uint8_t carry;
   /* The newest fragment atom the next vtc chain with a transfer job has
    * to wait for (MALI_JM_REQ_XVTC_*: everything, or the fragment-side
    * transfers, up to the end of the command buffer that asked); 0 when
    * nothing is owed. */
   mali_jm_ref xvtc_frag;
};

VkResult MALI_PER_ARCH(device_init)(struct mali_device *dev);
void MALI_PER_ARCH(device_finish)(struct mali_device *dev);

/*
 * First step of device destruction, before any GPU memory is freed: read
 * events until no atom is in flight, for at most MALI_JM_TEARDOWN_WAIT_NS.
 * The application has waited for its work, but after device loss, or with
 * a fence wait on a sync file that never signals, atoms can still be
 * queued or running, and freeing memory under them would fault. Atoms
 * still in flight after the wait cannot be stopped from user space: the
 * kbase context is then abandoned (mali_kbase_abandon), its memory left
 * in place until the process exits.
 */
void MALI_PER_ARCH(device_quiesce)(struct mali_device *dev);
#define MALI_JM_TEARDOWN_WAIT_NS (1000ll * 1000 * 1000)
VkResult MALI_PER_ARCH(queue_init)(struct mali_device *dev, struct mali_queue *queue);
void MALI_PER_ARCH(queue_finish)(struct mali_device *dev, struct mali_queue *queue);

/* vk_queue::driver_submit. */
VkResult MALI_PER_ARCH(queue_submit)(struct vk_queue *vkq, struct vk_queue_submit *submit);

/* vk_device::check_status. */
VkResult MALI_PER_ARCH(device_check_status)(struct vk_device *vkdev);

/* Shared with the CSF side (mali_queue.c): report device loss once, then
 * mark it and wake every waiter. Takes dev->lock; call without it. */
void mali_device_set_lost(struct mali_device *dev, const char *fmt, ...)
   __attribute__((format(printf, 2, 3)));

/* ---------------------------------------------------------------------- */
/* Atom building (mali_jm_queue.c), also used by mali_jm_sync_file.c       */

/* Is the atom r names still in flight (numbered, event not read)? */
static inline bool
mali_jm_ref_live(const struct mali_jm_device *jd, mali_jm_ref r)
{
   if (!r)
      return false;
   const struct mali_jm_atom_info *a = &jd->atoms[mali_jm_ref_num(r)];
   return a->in_flight && a->ref == r;
}

/* Start building atoms for one JOB_SUBMIT (more past the atom-number
 * limit), as submission seq (0 for atoms outside any submission). Waits
 * while another thread builds. With dev->lock held. */
void mali_jm_build_begin(struct mali_device *dev, uint64_t seq);

/*
 * Add one atom. slot: the slot order it joins (MALI_JM_SLOT_*: it follows
 * nothing by itself, the caller passes the slot's last atom as a
 * dependency) or MALI_JM_SLOT_NONE. Atoms in a slot order carry
 * EVENT_COALESCE; others report their own event. deps (up to two, 0 for
 * none) are dropped when no longer in flight; each is DATA, or ORDER on a
 * FENCE_WAIT atom (mali_jm_queue.c explains why). For soft fence atoms,
 * fence is the base_fence the atom's jc points at (copied). Returns the
 * atom's reference, or 0 with *result set (device lost, or the kernel
 * refused a submit). May drop dev->lock to wait for atom numbers.
 */
mali_jm_ref mali_jm_build_atom(struct mali_device *dev, enum mali_jm_atom_kind kind,
                               uint8_t slot, uint32_t core_req, uint64_t jc,
                               const struct kb_fence *fence, uint64_t udata1,
                               mali_jm_ref dep0, mali_jm_ref dep1, VkResult *result);

/* At most two dependencies standing for all of deps[0..n) (in flight or
 * not), joining pairs with dependency-only atoms while more than two are
 * in flight. Returns how many of out[] are set (0 with *result set on
 * failure, or when none is in flight). */
unsigned mali_jm_build_join_all(struct mali_device *dev, const mali_jm_ref *deps, unsigned n,
                                mali_jm_ref out[2], VkResult *result);

/*
 * Finish: the last atom built, or a join atom after it, becomes the
 * non-coalesced terminal of the JOB_SUBMIT (with tracker set: covering
 * every atom of both slot orders, flagged as the tracker of the build's
 * seq), and what is left is submitted. *terminal: its reference (0 when
 * nothing needs one); *built: whether the build numbered any atom;
 * *trigger_fd (may be NULL): the sync file of a FENCE_TRIGGER in the
 * build, -1 if none. Ends the build, also on failure.
 */
VkResult mali_jm_build_end(struct mali_device *dev, bool tracker, mali_jm_ref *terminal,
                           bool *built, int *trigger_fd);

/* End a build that failed: atoms not submitted yet are dropped. */
void mali_jm_build_abort(struct mali_device *dev);

/* With dev->lock held: read and process pending events without
 * blocking, unless another thread is blocked reading them. */
void mali_jm_drain_locked(struct mali_device *dev);

/* With dev->lock held: wait for events until abs_ns (CLOCK_MONOTONIC),
 * as the reader or on the condition variable. Returns early on any
 * event. */
void mali_jm_wait_locked(struct mali_device *dev, int64_t abs_ns);

/* ---------------------------------------------------------------------- */
/* Sync objects (mali_sync.c, built per arch)                              */

/*
 * The vk_sync type behind VkFence and binary VkSemaphore on the job
 * manager (v7 and v9). The payload of a GPU signal is req[0], the
 * submission number (reached when the queue's completed_seq is at least
 * that), and req[1], that submission's tracker atom, which a same-queue
 * wait or a sync-file export depends on while it is in flight.
 */
#define MALI_SYNC_REQ_COUNT 2

struct mali_sync {
   struct vk_sync vk;
   /* Under mali_device::lock. */
   bool host_signaled;
   bool submitted;
   uint64_t req[MALI_SYNC_REQ_COUNT];
   int fd;                          /* an imported sync file, or -1 */
};

extern const struct vk_sync_type MALI_PER_ARCH(sync_type);

void MALI_PER_ARCH(sync_clear)(struct mali_sync *s);

/* Log a failed sync-file import or export (mali_sync.c; thinned by
 * mali_diag_should_log). op says which ("import of", "export to", ...);
 * fd is -1 when there is none to name. Frontend-neutral. */
void MALI_PER_ARCH(sync_file_log_error)(struct mali_device *dev, const char *op, int fd,
                                        const char *why);

/* Hooks mali_sync.c calls (see mali_queue.h for the CSF bodies). */
const char *MALI_PER_ARCH(queue_wait)(struct mali_device *dev);
bool MALI_PER_ARCH(queue_reached)(struct mali_device *dev, const uint64_t req[MALI_SYNC_REQ_COUNT]);

/* For the slow-wait log (mali_sync.c, mali_jm_queue.c): what the queue
 * is waiting for, as one line (submissions, completions, atoms in flight,
 * events read). With dev->lock held. */
void MALI_PER_ARCH(queue_describe)(struct mali_device *dev, char *buf, size_t size);

/* Host waits re-check every 20 ms even without an event, as on the
 * G615: the reader's poll() covers GPU completion, the slice covers host
 * signals that cannot interrupt it. */
static inline bool
MALI_PER_ARCH(queue_has_notifier)(struct mali_device *dev)
{
   return true;
}

static inline void
MALI_PER_ARCH(queue_sleep)(struct mali_device *dev, int64_t until_ns)
{
   mali_jm_wait_locked(dev, until_ns);
}

/* ---------------------------------------------------------------------- */
/* Sync files (mali_jm_sync_file.c)                                        */

/* Is the sync file signalled (poll without waiting)? Shared with the CSF
 * side (mali_sync_file.c); frontend-neutral. */
bool mali_sync_file_signaled(int fd);

/*
 * A new sync file that signals once every sync file in fds has signalled
 * and the submission req names has completed (req may be NULL): FENCE_WAIT
 * atoms for the fds, a FENCE_TRIGGER after them and the submission's
 * tracker. *out = -1 when nothing is left to wait for. With dev->lock
 * held.
 */
VkResult MALI_PER_ARCH(sync_file_create)(struct mali_device *dev, uint32_t fd_count,
                                         const int *fds, const uint64_t *req, int *out);

/* Close the device's sync-file stream. */
void mali_jm_sync_file_finish(struct mali_jm_device *jd);

#endif
