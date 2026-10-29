/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Render passes on the command-stream frontend: the tiler contexts, the
 * vertex/tiler -> fragment synchronization, the fragment jobs, the tiler
 * heap protocol and the CRC seed updates. What does not depend on the
 * frontend (the pass's tile-buffer layout and preloads, the framebuffer
 * descriptors, the frame shaders, vkCmdBeginRendering) is in mali_fb.c.
 *
 * Tiler descriptor shapes, scoreboards and the vertex/tiler -> fragment
 * synchronization follow panvk v11 (csf/panvk_vX_cmd_draw.c:
 * get_tiler_desc, flush_tiling, issue_fragment_jobs). Where the blob and
 * panvk differ and both are valid, the blob's choice is taken when it is
 * simpler or is what this kernel was tested with:
 *
 *  - no tiler out-of-memory exception handler and no incremental
 *    rendering (the queue group has none);
 *  - the tiler heap and geometry buffer belong to the queue; the device
 *    has one queue, so a ONE_TIME_SUBMIT command buffer gets them packed
 *    into its tiler contexts on the CPU, and any other command buffer
 *    stores them (and zeroes the words the tiler writes) from the stream
 *    at run time, as the blob does for every pass, so it can run again;
 *  - exactly three heap-counter steps per pass that tiles, in order:
 *    HEAP_OPERATION{Vertex/Tiler Started} before the first draw, a
 *    deferred HEAP_OPERATION{Vertex/Tiler Completed} after FINISH_TILING,
 *    and FINISH_FRAGMENT with the increment bit on the last tiler context
 *    (the blob; panvk uses a separate HEAP_OPERATION{Fragment Completed}
 *    for more than one tiler context). kbase's tiler out-of-memory handler
 *    rejects the event unless frag_end <= vt_end < vt_start. A pass that
 *    draws nothing does not touch the counters at all.
 *
 * Transaction elimination (CRC) follows panvk v11's seed scheme; see
 * "CRC" below.
 */

#include "mali_cmd_state.h"
#include "mali_measure.h"
#include "mali_query.h"

#include "mali_arch.h"
#include "mali_image.h"
#include "mali_queue.h"

#if PAN_ARCH != 11
#error "mali_cmd_render.c is written for arch v11"
#endif

/* ---------------------------------------------------------------------- */
/* Begin                                                                   */

void
MALI_PER_ARCH(cmd_render_begin)(struct mali_cmd_buffer *cmd, const struct mali_render_desc *desc)
{
   const bool any_preload = MALI_PER_ARCH(fb_begin)(cmd, desc);

   if (any_preload) {
      /* Loads go through the texture unit: invalidate its caches after
       * earlier attachment writes (panvk cmd_init_render_state; the blob
       * sets r49 for the same invalidate). */
      const VkMemoryBarrier2 mb = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                         VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
                         VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
         .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                          VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
         .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
      };
      const VkDependencyInfo dep = {
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .memoryBarrierCount = 1,
         .pMemoryBarriers = &mb,
      };
      struct mali_cs_deps deps = {0};
      mali_cmd_add_deps(cmd, &dep, &deps);
      mali_cmd_emit_barrier(cmd, &deps);
   }
}

/* ---------------------------------------------------------------------- */
/* Tiler contexts                                                          */

/* v11 cs_iter_sb_update with nothing inside (panvk cs_next_iter_sb). */
static void
next_iter_sb(struct mali_cmd_buffer *cmd, struct cs_builder *b)
{
   const struct mali_csf_device *csf = cmd->dev->csf;
   struct cs_index next_sb = mali_cs_scratch_reg32(b, 0);
   struct cs_index sb_mask = mali_cs_scratch_reg32(b, 1);

   cs_next_sb_entry(b, next_sb, MALI_CS_SCOREBOARD_TYPE_ENDPOINT,
                    MALI_CS_NEXT_SB_ENTRY_FORMAT_INDEX);
   cs_move32_to(b, sb_mask, 0);
   cs_bit_set32(b, sb_mask, sb_mask, next_sb);
   cs_set_state(b, MALI_CS_SET_STATE_TYPE_SB_MASK_WAIT, sb_mask);
   cs_move32_to(b, sb_mask, csf->sb.all_iters_mask);
   cs_bit_clear32(b, sb_mask, sb_mask, next_sb);
   cs_set_state(b, MALI_CS_SET_STATE_TYPE_SB_MASK_STREAM, sb_mask);
}

/*
 * One Tiler Context's static words: everything but what the stream stores
 * or zeroes at run time for a command buffer that may run again (the
 * heap, geometry buffer and polygon list). Takes only data that stays the
 * same for the pass's lifetime (r->desc, the device's queue), so it can
 * be called again later to fix up a field without redoing allocation or
 * drifting from the initial build.
 */
static void
pack_tiler_context(struct mali_cmd_buffer *cmd, const struct mali_render_state *r, uint32_t i,
                   uint32_t layers, bool once, uint8_t *dst)
{
   const struct mali_csf_queue *q = cmd->dev->csf->queue;
   pan_cast_and_pack(dst, TILER_CONTEXT, cfg) {
      if (once) {
         cfg.heap = mali_queue_heap_desc_va(q);
         cfg.geometry_buffer_size = MALI_TILER_GEOM_BUF_SIZE;
         cfg.geometry_buffer = mali_queue_geom_buf_va(q);
      }
      cfg.hierarchy_mask = MALI_TILER_HIERARCHY_MASK;
      cfg.sample_pattern = MALI_SAMPLE_PATTERN_SINGLE_SAMPLED;
      cfg.first_provoking_vertex = r->first_provoking_vertex;
      cfg.fb_width = r->desc.width;
      cfg.fb_height = r->desc.height;
      cfg.layer_count = MIN2(layers - i * MALI_LAYERS_PER_TILER_CTX,
                             MALI_LAYERS_PER_TILER_CTX);
      cfg.layer_offset = -(int32_t)(i * MALI_LAYERS_PER_TILER_CTX);
   }
}

bool
MALI_PER_ARCH(cmd_render_tiler)(struct mali_cmd_buffer *cmd)
{
   struct mali_render_state *r = &cmd->gfx.render;
   assert(r->active);
   if (r->tiler)
      return true;
   if (!mali_fb_alloc_tsd(cmd, r))
      return false;

   const uint32_t layers = MAX2(r->desc.layer_count, 1);
   const uint32_t td_count = DIV_ROUND_UP(layers, MALI_LAYERS_PER_TILER_CTX);
   struct mali_ptr p = mali_cmd_alloc(cmd, td_count * pan_size(TILER_CONTEXT), 64);
   if (!p.cpu)
      return false;

   /* Words 2-4 on the CPU (as the blob). Polygon list, completed
    * top/bottom and the private state start at zero; the tiler writes
    * them. A command buffer that runs once gets the queue's heap and
    * geometry buffer here too and needs no stream work; one that may run
    * again has the stream below store them and zero the rest before
    * every run. */
   STATIC_ASSERT(MALI_QUEUE_COUNT == 1);
   const bool once = cmd->usage & VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
   for (uint32_t i = 0; i < td_count; i++)
      pack_tiler_context(cmd, r, i, layers, once,
                        (uint8_t *)p.cpu + i * pan_size(TILER_CONTEXT));
   r->tiler = p.gpu;
   r->tiler_cpu = p.cpu;
   r->td_count = td_count;

   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_VERTEX_TILER);
   if (!once) {
      struct cs_index zero_lo = mali_cs_scratch_reg_tuple(b, 0, 2);
      struct cs_index heap_geom = mali_cs_scratch_reg_tuple(b, 2, 4);
      struct cs_index zeros = mali_cs_scratch_reg_tuple(b, 6, 8);
      struct cs_index addr = mali_cs_scratch_reg64(b, 14);

      cs_move64_to(b, zero_lo, 0);
      for (unsigned i = 0; i < 8; i += 2)
         cs_move64_to(b, mali_cs_scratch_reg64(b, 6 + i), 0);
      STATIC_ASSERT(offsetof(struct mali_cs_subqueue_ctx, geom_buf) ==
                    offsetof(struct mali_cs_subqueue_ctx, tiler_heap) + 8);
      cs_load_to(b, heap_geom, mali_cs_subqueue_ctx_reg(b), BITFIELD_MASK(4),
                 offsetof(struct mali_cs_subqueue_ctx, tiler_heap));

      for (uint32_t i = 0; i < td_count; i++) {
         cs_move64_to(b, addr, r->tiler + i * pan_size(TILER_CONTEXT));
         cs_store(b, zero_lo, addr, BITFIELD_MASK(2), 0);          /* polygon list */
         cs_store(b, heap_geom, addr, BITFIELD_MASK(4), 6 * 4);    /* heap, geometry buffer */
         cs_store(b, mali_cs_scratch_reg_tuple(b, 6, 4), addr, BITFIELD_MASK(4),
                  10 * 4);                                         /* completed top/bottom */
         cs_store(b, zeros, addr, BITFIELD_MASK(8), 24 * 4);       /* private state */
      }
      cs_flush_stores(b);
   }

   /* Timing: the vertex/tiler side of the pass starts here. */
   if (unlikely(cmd->measure))
      r->measure_vt = mali_measure_begin(cmd, MALI_MEASURE_PASS_VT,
                                         MALI_SUBQUEUE_VERTEX_TILER, cmd->passes - 1);

   /* The endpoint scoreboard entry the pass's draws signal, then the first
    * heap-counter step. */
   next_iter_sb(cmd, b);
   cs_vt_start(b, cs_now());

   cmd->gfx.draw.dirty |= MALI_GFX_DIRTY_PASS;
   return true;
}

void
MALI_PER_ARCH(cmd_render_tiler_set_provoking_vertex)(struct mali_cmd_buffer *cmd)
{
   struct mali_render_state *r = &cmd->gfx.render;
   const uint32_t layers = MAX2(r->desc.layer_count, 1);
   const bool once = cmd->usage & VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
   for (uint32_t i = 0; i < r->td_count; i++)
      pack_tiler_context(cmd, r, i, layers, once,
                        (uint8_t *)r->tiler_cpu + (uint64_t)i * pan_size(TILER_CONTEXT));
}

/* ---------------------------------------------------------------------- */
/* CRC (transaction elimination)                                           */

/*
 * The fragment unit can keep a 64-bit CRC per 16x16 tile of one colour
 * target in a table next to the image (mali_image_layout.c). With CRC
 * read on, a tile whose new CRC equals the stored one is not written to
 * memory; with CRC write on, the new CRCs are stored.
 *
 * A stored CRC is only right while the tile's memory holds what that CRC
 * was computed from. Everything that writes the image other than a pass
 * that keeps its CRCs must therefore make the stored CRCs useless. panvk
 * v11 does this without touching the table: the low 16 bits of the CRC
 * Clear Color word are the seed the hardware starts each CRC from, taken
 * from a word in the image's CRC header when the pass runs; an invalidation
 * adds one to that word, so every CRC stored before it stops matching. The
 * blob does the same with a device-wide stamp counter per slice. We take
 * panvk's per-image counter: no device-global state and one
 * load/add/store per invalidation.
 *
 * The seed is 16 bits, so a CRC stored 65,536 invalidations ago matches
 * again if its tile was not rendered since and the pass renders exactly the
 * tile contents it was computed from while memory holds something else
 * (panvk has the same limit).
 */

/* Seed word in the CRC header. */
#define CRC_SEED_OFFSET 0
#define CRC_SEED_MASK 0xffff
/* Fragment-stream scratch registers of the seed code: above the ones the
 * tiling wait and the barrier flush use (0-3), so the pass patch's store
 * can still be reading them while those run. */
#define CRC_SCRATCH 8

void
MALI_PER_ARCH(cmd_crc_invalidate)(struct mali_cmd_buffer *cmd, const struct mali_image *image)
{
   if (!image || !image->crc_header)
      return;
   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_FRAGMENT);
   struct cs_index addr = mali_cs_scratch_reg64(b, CRC_SCRATCH);
   struct cs_index seed = mali_cs_scratch_reg32(b, CRC_SCRATCH + 2);

   /* A pending store may be the pass patch or an earlier bump of the same
    * word. The store is waited for at once: cs_builder does not track the
    * registers a pending store reads, and bumps are rare. */
   cs_flush_stores(b);
   cs_move64_to(b, addr, image->crc_header);
   cs_load32_to(b, seed, addr, CRC_SEED_OFFSET);
   cs_add_imm32(b, seed, seed, 1);
   cs_store32(b, seed, addr, CRC_SEED_OFFSET);
   cs_flush_stores(b);
}

/*
 * Fragment stream, before the pass's fragment job: put the CRC target's
 * current seed into the CRC Clear Color of the FBD, and invalidate every
 * other stored target that has CRC state (its tiles are written without
 * CRCs). fbd is the FBD's address, untagged.
 */
static void
emit_crc_updates(struct mali_cmd_buffer *cmd, const struct mali_render_state *r, uint64_t fbd)
{
   for (unsigned i = 0; i < r->desc.rt_count; i++) {
      const struct mali_fb_target *t = &r->desc.rt[i];
      if ((int)i != r->crc_rt && t->image && t->store && t->level == 0 && t->plane == 0)
         MALI_PER_ARCH(cmd_crc_invalidate)(cmd, t->image);
   }
   if (r->crc_rt < 0)
      return;

   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_FRAGMENT);
   struct cs_index addr = mali_cs_scratch_reg64(b, CRC_SCRATCH);
   struct cs_index seed = mali_cs_scratch_reg32(b, CRC_SCRATCH + 2);
   struct cs_index mask = mali_cs_scratch_reg32(b, CRC_SCRATCH + 3);
   struct cs_index lo = mali_cs_scratch_reg32(b, CRC_SCRATCH + 4);

   cs_flush_stores(b);
   cs_move64_to(b, addr, r->desc.rt[r->crc_rt].image->crc_header);
   cs_load32_to(b, seed, addr, CRC_SEED_OFFSET);
   cs_move32_to(b, mask, CRC_SEED_MASK);
   cs_move32_to(b, lo, r->crc_clear_lo & ~CRC_SEED_MASK);
   cs_and32(b, seed, seed, mask);
   cs_or32(b, lo, lo, seed);
   /* CRC Clear Color is word 2 of the ZS/CRC extension, which follows
    * the Framebuffer Parameters. */
   cs_move64_to(b, addr, fbd + pan_size(FRAMEBUFFER) + 2 * 4);
   cs_store32(b, lo, addr, 0);
   /* The caller waits for the store before the fragment job. */
}

/* ---------------------------------------------------------------------- */
/* End                                                                     */

/* Vertex/tiler side of the end (panvk flush_tiling): finish tiling, the
 * deferred HEAP_OPERATION{Vertex/Tiler Completed} behind every draw of the
 * pass, and the signal the fragment side waits on: a deferred increment of
 * the shared scoreboard entry (the blob on this GPU), or without shared
 * entries a deferred SYNC_ADD64 on the vertex/tiler barrier counter
 * (panvk). */
static void
flush_tiling(struct mali_cmd_buffer *cmd)
{
   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_VERTEX_TILER);

   /* The draws' copies of uniform-buffer words into fragment FAU blocks
    * (mali_cmd_draw.c emit_ubo_push) must land before the fragment job
    * reads them. */
   cs_flush_stores(b);
   if (cmd->gfx.render.ubo_stores) {
      cs_wait_slot(b, MALI_SB_LS);
      cmd->gfx.render.ubo_stores = false;
   }
   cs_finish_tiling(b);
   if (cmd->dev->csf->shared_sb) {
      cs_vt_end(b, cs_defer_indirect());
      mali_cs_shared_sb_inc(b, MALI_SHARED_SB_VT_TO_FRAG, cs_defer_indirect());
      return;
   }

   struct cs_index addr = mali_cs_scratch_reg64(b, 0);
   struct cs_index one = mali_cs_scratch_reg64(b, 2);
   cs_move64_to(b, addr, mali_queue_syncobj_va(cmd->dev->csf->queue,
                                                MALI_SUBQUEUE_VERTEX_TILER));
   cs_move64_to(b, one, 1);
   cs_vt_end(b, cs_defer_indirect());
   cs_sync64_add(b, true, MALI_CS_SYNC_SCOPE_CSG, one, addr, cs_defer_indirect());
   cmd->cs[MALI_SUBQUEUE_VERTEX_TILER].relative_sync_point++;
}

static void
issue_fragment_jobs(struct mali_cmd_buffer *cmd, struct mali_render_state *r,
                    uint64_t fbd_ptr, uint32_t fbd_sz)
{
   const struct mali_csf_device *csf = cmd->dev->csf;
   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_FRAGMENT);

   cs_move32_to(b, cs_sr_reg32(b, FRAGMENT, BBOX_MIN), r->miny << 16 | r->minx);
   cs_move32_to(b, cs_sr_reg32(b, FRAGMENT, BBOX_MAX), r->maxy << 16 | r->maxx);

   /* The CRC seed patch first: seeds are only read and bumped on this
    * stream, so stream order is all it needs, and its load and store then
    * overlap the tiling wait. */
   emit_crc_updates(cmd, r, fbd_ptr & ~(uint64_t)63);

   /* Wait for the tiling of this pass: take one off the shared entry the
    * vertex/tiler side increments per pass (a pass that tiled nothing has
    * no increment and waits for nothing), or, without shared entries,
    * panvk's wait_finish_tiling: the vertex/tiler counter passes progress +
    * signals so far. Then the barrier wait and cache flush put off until
    * this job; in this order the tiling wait, which the firmware takes
    * microseconds to pass even when it is already satisfied, overlaps the
    * previous fragment job. */
   if (csf->shared_sb) {
      if (r->tiler)
         mali_cs_shared_sb_dec(b, MALI_SHARED_SB_VT_TO_FRAG);
   } else {
      struct cs_index addr = mali_cs_scratch_reg64(b, 0);
      struct cs_index ref = mali_cs_scratch_reg64(b, 2);
      cs_move64_to(b, addr, mali_queue_syncobj_va(cmd->dev->csf->queue,
                                                  MALI_SUBQUEUE_VERTEX_TILER));
      cs_add_imm64(b, ref, mali_cs_progress_seqno_reg(b, MALI_SUBQUEUE_VERTEX_TILER),
                   cmd->cs[MALI_SUBQUEUE_VERTEX_TILER].relative_sync_point);
      cs_sync64_wait(b, false, MALI_CS_CONDITION_GREATER, ref, addr);
   }
   mali_cmd_frag_flush_pending(cmd);
   /* The fragment job reads the patched FBD from memory. */
   if (r->crc_rt >= 0)
      cs_flush_stores(b);

   /* Timing: the fragment side, from the end of the tiling wait to
    * the end of the pass's fragment jobs. */
   if (unlikely(cmd->measure))
      r->measure_frag = mali_measure_begin(cmd, MALI_MEASURE_PASS_FRAG,
                                           MALI_SUBQUEUE_FRAGMENT, cmd->passes - 1);

   struct cs_index fbd = cs_sr_reg64(b, FRAGMENT, FBD_POINTER);
   cs_move64_to(b, fbd, fbd_ptr);
   const uint32_t layers = MAX2(r->desc.layer_count, 1);
   for (uint32_t l = 0; l < layers; l++) {
      if (l)
         cs_add_imm64(b, fbd, fbd, fbd_sz);
      cs_run_fragment(b, false, MALI_TILE_RENDER_ORDER_Z_ORDER);
   }
   if (unlikely(r->measure_frag))
      mali_measure_end(cmd, r->measure_frag);

   /* The heap chunks of the pass go back after the fragment job, in pass
    * order: wait on the current iterator entry (the RUN_FRAGMENTs and the
    * previous pass's FINISH_FRAGMENT), signal the next (panvk). Exactly one
    * fragment-completed increment per tiled pass, on the last context. */
   struct cs_index next_sb = mali_cs_scratch_reg32(b, 0);
   struct cs_index sb_mask = mali_cs_scratch_reg32(b, 1);
   struct cs_index tiler = mali_cs_scratch_reg64(b, 2);
   struct cs_index completed = mali_cs_scratch_reg_tuple(b, 4, 4);
   struct cs_index top = mali_cs_scratch_reg64(b, 4);
   struct cs_index bottom = mali_cs_scratch_reg64(b, 6);

   cs_next_sb_entry(b, next_sb, MALI_CS_SCOREBOARD_TYPE_ENDPOINT,
                    MALI_CS_NEXT_SB_ENTRY_FORMAT_INDEX);
   cs_set_state(b, MALI_CS_SET_STATE_TYPE_SB_SEL_DEFERRED, next_sb);
   if (r->td_count) {
      cs_move64_to(b, tiler, r->tiler);
      for (uint32_t i = 0; i < r->td_count; i++) {
         cs_load_to(b, completed, tiler, BITFIELD_MASK(4),
                    i * pan_size(TILER_CONTEXT) + 10 * 4);
         cs_finish_fragment(b, i == r->td_count - 1, top, bottom, cs_defer_indirect());
      }
   }
   cs_set_state_imm32(b, MALI_CS_SET_STATE_TYPE_SB_SEL_DEFERRED, MALI_SB_DEFERRED_SYNC);

   /* panvk also adds one to the fragment counter here, for its descriptor
    * ring buffer; nothing waits on that counter per pass in this driver
    * (barriers signal it themselves), so the signal is left out, as the
    * blob does. */

   cs_move32_to(b, sb_mask, 0);
   cs_bit_set32(b, sb_mask, sb_mask, next_sb);
   cs_set_state(b, MALI_CS_SET_STATE_TYPE_SB_MASK_WAIT, sb_mask);
   cs_move32_to(b, sb_mask, csf->sb.all_iters_mask);
   cs_bit_clear32(b, sb_mask, sb_mask, next_sb);
   cs_set_state(b, MALI_CS_SET_STATE_TYPE_SB_MASK_STREAM, sb_mask);
}

void
MALI_PER_ARCH(cmd_render_end)(struct mali_cmd_buffer *cmd)
{
   struct mali_render_state *r = &cmd->gfx.render;
   if (!r->active)
      return;

   if (MALI_PER_ARCH(fb_pass_has_work)(r) && !vk_command_buffer_has_error(&cmd->vk)) {
      bool ok = mali_fb_alloc_tsd(cmd, r);
      uint32_t fbd_sz = 0;
      uint64_t fbds = ok ? MALI_PER_ARCH(fb_build)(cmd, r, &fbd_sz) : 0;
      if (!fbds) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      } else {
         MALI_PER_ARCH(fb_fill_tsd)(cmd, r);
         if (r->tiler)
            flush_tiling(cmd);
         if (unlikely(r->measure_vt))
            mali_measure_end(cmd, r->measure_vt);
         issue_fragment_jobs(cmd, r, fbds, fbd_sz);
      }
   }
   if (!vk_command_buffer_has_error(&cmd->vk))
      MALI_PER_ARCH(cmd_oq_pass_end)(cmd);
   if (unlikely(cmd->measure)) {
      const uint32_t w = r->desc.width, h = r->desc.height;
      mali_measure_info(cmd, r->measure_vt, w, h, r->desc.rt_count, r->draws);
      mali_measure_info(cmd, r->measure_frag, w, h, r->desc.rt_count, r->draws);
   }

   r->active = false;
   cmd->gfx.draw.dirty = MALI_GFX_DIRTY_ALL;
}
