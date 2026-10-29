/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Draws on the command-stream frontend: vkCmdDraw / vkCmdDrawIndexed as
 * RUN_IDVS on the vertex/tiler subqueue, full-screen draws (RUN_FULLSCREEN)
 * for vkCmdClearAttachments (mali_cmd_meta_gfx.c) and in-pass barriers. The
 * binds, the dirty bits and the descriptors a draw points at are shared
 * with the job-manager back half (mali_cmd_state.[ch]); this file turns
 * them into register moves.
 *
 * The encoding follows panvk v11 (csf/panvk_vX_cmd_draw.c: prepare_draw,
 * launch_draw, cmd_run_fullscreen, cmd_fb_barrier), "which source wins"
 * for command-stream encoding; the per-pipeline words are the ones our
 * pipeline compile baked in (mali_pipeline_state.c). Differences:
 *
 *  - RUN_IDVS sets bit 55, which v11.xml does not define, as every blob
 *    RUN_IDVS does (read as "Tiler Flags2 Enable", the RUN_FULLSCREEN
 *    field at the same bit). TILER_FLAGS2 is written as 0 before every
 *    draw either way.
 *  - The FAU blocks carry our sysval layout (mali_shader.h), built only
 *    when their inputs changed.
 *
 * Register use on the vertex/tiler subqueue is the v11 IDVS staging set;
 * state registers persist between draws and are rewritten from dirty
 * bits (MALI_GFX_DIRTY_*).
 */

#include "mali_cmd_state.h"
#include "mali_measure.h"

#include "vk_log.h"

#include "mali_arch.h"
#include "mali_queue.h"

#if PAN_ARCH != 11
#error "mali_cmd_draw.c is written for arch v11"
#endif

#define IDVS(reg) MALI_IDVS_SR_##reg

/* ---------------------------------------------------------------------- */
/* FAU                                                                     */

/*
 * The fragment shader's pushed uniform-buffer words (mali_shader_fau::
 * ubo_push): the vertex/tiler stream copies them from the buffer into the
 * FAU block at fau_va (LOAD_MULTIPLE into scratch registers,
 * STORE_MULTIPLE into the block), ahead of the draw's RUN_IDVS. The GPU
 * reads the buffer when it runs the command buffer, as for any uniform
 * buffer read; the CPU never reads it (ARMSX2's rings are write-combined,
 * and CPU reads of them cost the GS thread up to 25 % on draw-heavy
 * dumps). Nothing waits for the stores here: the
 * fragment shader runs in the pass's fragment job, after flush_tiling has
 * waited for them. Runs of adjacent words share one load and one store;
 * words past the binding's range are not copied and stay zero (the CPU
 * wrote the block).
 */
static void
emit_ubo_push_general(struct mali_cmd_buffer *cmd, const struct mali_shader *s,
                      uint64_t fau_va)
{
   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_VERTEX_TILER);
   const struct mali_desc_state *desc = &cmd->gfx.desc;
   /* Scratch 0-1 the source address, 2-3 the FAU block, 4-15 data. */
   enum { SRC = 0, DST = 2, DATA = 4, DATA_REGS = 12 };
   struct {
      unsigned reg, count;
      int dst_off;
   } pend[DATA_REGS];
   unsigned npend = 0, used = 0;
   uint64_t cur_src = 0;

   /* The data registers may still be the source of the previous draw's
    * stores. */
   cs_flush_stores(b);

#define FLUSH_PENDING()                                                        \
   do {                                                                        \
      if (npend) {                                                             \
         cs_flush_loads(b);                                                    \
         cs_move64_to(b, mali_cs_scratch_reg64(b, DST), fau_va);               \
         for (unsigned k = 0; k < npend; k++)                                  \
            cs_store(b, mali_cs_scratch_reg_tuple(b, pend[k].reg, pend[k].count), \
                     mali_cs_scratch_reg64(b, DST), BITFIELD_MASK(pend[k].count), \
                     pend[k].dst_off);                                         \
      }                                                                        \
      npend = used = 0;                                                        \
   } while (0)

   const uint32_t n = s->fau.ubo_push_count;
   for (uint32_t i = 0; i < n;) {
      const struct mali_ubo_push_word w0 = s->fau.ubo_push[i];
      const uint32_t h = s->desc.dyn_bufs.map[w0.dyn];
      const uint32_t set = MALI_COPY_DESC_HANDLE_SET(h);
      const uint32_t idx = MALI_COPY_DESC_HANDLE_INDEX(h);
      const struct mali_descriptor_set *ds = desc->sets[set];
      const uint64_t range = ds ? ds->dyn_bufs[idx].range : 0;
      const uint64_t addr = ds ? ds->dyn_bufs[idx].addr + desc->dyn_offsets[set][idx] : 0;

      /* A run: consecutive words of one buffer, whole words inside the
       * range, the last may be half a word. */
      unsigned regs = 0, k = 0;
      while (i + k < n && regs < DATA_REGS) {
         const struct mali_ubo_push_word w = s->fau.ubo_push[i + k];
         if (w.dyn != w0.dyn || w.offset != w0.offset + 8 * k)
            break;
         const unsigned valid = w.offset >= range ? 0 :
                                range - w.offset >= 8 ? 2 :
                                range - w.offset >= 4 ? 1 : 0;
         if (!valid || regs + valid > DATA_REGS)
            break;
         regs += valid;
         k++;
         if (valid < 2)
            break;
      }
      if (!k || !addr) {
         i++;   /* nothing of this word is inside the binding */
         continue;
      }

      if (used + regs > DATA_REGS)
         FLUSH_PENDING();
      if (addr != cur_src) {
         cs_move64_to(b, mali_cs_scratch_reg64(b, SRC), addr);
         cur_src = addr;
      }
      cs_load_to(b, mali_cs_scratch_reg_tuple(b, DATA + used, regs),
                 mali_cs_scratch_reg64(b, SRC), BITFIELD_MASK(regs), w0.offset);
      pend[npend].reg = DATA + used;
      pend[npend].count = regs;
      pend[npend].dst_off = (s->fau.ubo_push_start + i) * MALI_FAU_WORD_SIZE;
      npend++;
      used += regs;
      i += k;
   }
   FLUSH_PENDING();
#undef FLUSH_PENDING
}

/* ---------------------------------------------------------------------- */
/* Draw                                                                    */

/* Bit 55 of RUN_IDVS; the on-device smoke test clears it once to check
 * what it does. */
bool mali_idvs_bit55 = true;

/* RUN_IDVS with bit 55 set. Selects as panvk v11: varying stage from
 * group 0/1 (SPD_1), fragment from group 2, both with TSD_0. */
static uint64_t
run_idvs_ins(uint32_t flags_override)
{
   struct mali_cs_run_idvs_packed ins;
   pan_pack(&ins, CS_RUN_IDVS, I) {
      I.flags_override = flags_override;
      I.malloc_enable = true;
      I.draw_id_register_enable = false;
      I.varying_srt_select = false;
      I.varying_fau_select = false;
      I.varying_tsd_select = false;
      I.fragment_srt_select = true;
      I.fragment_tsd_select = false;
   }
   if (mali_idvs_bit55)
      ins.opaque[1] |= 1u << (55 - 32);
   uint64_t v;
   memcpy(&v, &ins, sizeof(v));
   return v;
}

static void
emit_run_idvs(struct cs_builder *b, uint32_t flags_override)
{
   cs_flush_loads(b);
   b->req_resource_mask |= CS_IDVS_RES;
   const uint64_t ins = run_idvs_ins(flags_override);
   memcpy(cs_alloc_ins(b), &ins, sizeof(ins));
}

/*
 * The instructions of one draw, collected here and copied into the stream
 * in one piece (one space check instead of one per instruction). Only
 * register moves to the IDVS staging registers and RUN_IDVS go through it,
 * and a move is left out when the register already holds the value
 * (mali_gfx_draw_state.regs).
 */
struct idvs_batch {
   struct mali_gfx_draw_state *d;
   unsigned n;
   uint64_t ins[128];
};

static inline void
batch_move32(struct idvs_batch *e, unsigned reg, uint32_t v)
{
   struct mali_gfx_draw_state *d = e->d;
   if ((d->regs_valid & BITFIELD64_BIT(reg)) && d->regs[reg] == v)
      return;
   d->regs_valid |= BITFIELD64_BIT(reg);
   d->regs[reg] = v;
   struct mali_cs_move32_packed m;
   pan_pack(&m, CS_MOVE32, I) {
      I.destination = reg;
      I.immediate = v;
   }
   assert(e->n < ARRAY_SIZE(e->ins));
   memcpy(&e->ins[e->n++], &m, sizeof(m));
}

static inline void
batch_move64(struct idvs_batch *e, unsigned reg, uint64_t v)
{
   struct mali_gfx_draw_state *d = e->d;
   const uint32_t lo = (uint32_t)v, hi = (uint32_t)(v >> 32);
   const bool lo_ok = (d->regs_valid & BITFIELD64_BIT(reg)) && d->regs[reg] == lo;
   const bool hi_ok = (d->regs_valid & BITFIELD64_BIT(reg + 1)) && d->regs[reg + 1] == hi;
   if (lo_ok && hi_ok)
      return;
   if (!lo_ok && !hi_ok && v < (1ull << 48)) {
      /* MOVE48 zero-extends into the high register. */
      d->regs_valid |= BITFIELD64_BIT(reg) | BITFIELD64_BIT(reg + 1);
      d->regs[reg] = lo;
      d->regs[reg + 1] = hi;
      struct mali_cs_move48_packed m;
      pan_pack(&m, CS_MOVE48, I) {
         I.destination = reg;
         I.immediate = v;
      }
      assert(e->n < ARRAY_SIZE(e->ins));
      memcpy(&e->ins[e->n++], &m, sizeof(m));
      return;
   }
   batch_move32(e, reg, lo);
   batch_move32(e, reg + 1, hi);
}

static void
batch_flush(struct cs_builder *b, struct idvs_batch *e)
{
   if (!e->n)
      return;
   /* A move must not overtake a pending load into its register; waiting
    * for all loads is what the RUN_IDVS after it does anyway. */
   cs_flush_loads(b);
   cs_flush_pending_if(b);
   uint64_t *dst = cs_alloc_ins_block(b, e->n);
   if (dst)
      memcpy(dst, e->ins, e->n * sizeof(e->ins[0]));
   e->n = 0;
}

/* The uniform-buffer copy for the fragment FAU block at fau_va. ARMSX2's
 * case, one buffer whose binding covers every pushed word, appends the
 * shader's prebuilt words (mali_shader::ubo_copy) to the draw's batch
 * after the two address moves; anything else goes through the stream
 * builder. */
static void
emit_ubo_push(struct mali_cmd_buffer *cmd, struct idvs_batch *e, const struct mali_shader *s,
              uint64_t fau_va)
{
   const struct mali_desc_state *desc = &cmd->gfx.desc;
   const uint32_t h = s->desc.dyn_bufs.map[s->fau.ubo_push[0].dyn];
   const uint32_t set = MALI_COPY_DESC_HANDLE_SET(h);
   const uint32_t idx = MALI_COPY_DESC_HANDLE_INDEX(h);
   const struct mali_descriptor_set *ds = desc->sets[set];

   if (!s->ubo_copy.count || !ds || ds->dyn_bufs[idx].range < s->ubo_copy.end ||
       !ds->dyn_bufs[idx].addr ||
       e->n + 2 + s->ubo_copy.count + 32 > ARRAY_SIZE(e->ins)) {
      emit_ubo_push_general(cmd, s, fau_va);
      return;
   }

   const uint64_t addrs[2] = {ds->dyn_bufs[idx].addr + desc->dyn_offsets[set][idx], fau_va};
   const unsigned regs[2] = {MALI_UBO_COPY_SRC, MALI_UBO_COPY_DST};
   for (unsigned i = 0; i < 2; i++) {
      struct mali_cs_move48_packed m;
      pan_pack(&m, CS_MOVE48, I) {
         I.destination = regs[i];
         I.immediate = addrs[i];
      }
      memcpy(&e->ins[e->n++], &m, sizeof(m));
   }
   memcpy(&e->ins[e->n], s->ubo_copy.ins, s->ubo_copy.count * sizeof(uint64_t));
   e->n += s->ubo_copy.count;
   cmd->gfx.render.ubo_stores = true;
}

/* The draw's state (mali_cmd_state.h, "A draw's state"): the shared
 * builders for every dirty group, each word moved into its IDVS staging
 * register as it is built. */
static bool
prepare_draw(struct mali_cmd_buffer *cmd, const struct mali_draw_info *di, struct idvs_batch *e)
{
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   struct mali_render_state *r = &cmd->gfx.render;
   struct mali_graphics_pipeline *p = d->pipeline;
   const struct mali_gfx_bind *bd = &p->bind;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const struct mali_gfx_baked *dbk;
   uint64_t v;

   if (!mali_gfx_update_dirty(cmd, di, &dbk))
      return false;

   if (d->dirty & MALI_GFX_DIRTY_BLEND) {
      if (!mali_gfx_build_blend(cmd, &v))
         return false;
      batch_move64(e, IDVS(BLEND_DESC), v);
   }

   struct mali_graphics_sysvals sv;
   if (d->dirty & (MALI_GFX_DIRTY_VS_FAU | MALI_GFX_DIRTY_FS_FAU))
      mali_gfx_fill_sysvals(cmd, &sv);

   if (d->dirty & MALI_GFX_DIRTY_VS_FAU) {
      if (!mali_gfx_build_vs_fau(cmd, &sv, &v))
         return false;
      batch_move64(e, IDVS(FAU_0), v);
   }
   if (d->dirty & MALI_GFX_DIRTY_FS_FAU) {
      if (!mali_gfx_build_fs_fau(cmd, &sv, &v))
         return false;
      /* Straight into the stream, ahead of the batched moves and RUN_IDVS. */
      if (p->fs && p->fs->fau.ubo_push_count)
         emit_ubo_push(cmd, e, p->fs, v & BITFIELD64_MASK(56));
      batch_move64(e, IDVS(FAU_2), v);
   }
   if (d->dirty & MALI_GFX_DIRTY_VS_SRT) {
      if (!mali_gfx_build_vs_srt(cmd, &v))
         return false;
      batch_move64(e, IDVS(SRT_0), v);
   }
   if (d->dirty & MALI_GFX_DIRTY_FS_SRT) {
      if (!mali_gfx_build_fs_srt(cmd, &v))
         return false;
      batch_move64(e, IDVS(SRT_2), v);
   }
   if (d->dirty & MALI_GFX_DIRTY_ZSD) {
      if (!mali_gfx_build_zsd(cmd, dbk, &v))
         return false;
      batch_move64(e, IDVS(ZSD), v);
   }
   if (d->dirty & MALI_GFX_DIRTY_PIPELINE) {
      bool lines;
      if (unlikely(dbk)) {
         batch_move64(e, IDVS(SPD_0), dbk->vs_pos_spd);
         batch_move64(e, IDVS(SPD_1), dbk->vs_var_spd);
         batch_move64(e, IDVS(SPD_2), dbk->fs_spd);
         batch_move32(e, IDVS(TILER_FLAGS), dbk->tiler_flags);
         batch_move32(e, IDVS(DCD0), mali_gfx_dcd0(d, dbk));
         batch_move32(e, IDVS(DCD1), dbk->dcd1);
         batch_move32(e, IDVS(DCD2), dbk->dcd2);
         lines = u_reduced_prim(dbk->prim) == MESA_PRIM_LINES;
      } else {
         batch_move64(e, IDVS(SPD_0), bd->spd[0]);
         batch_move64(e, IDVS(SPD_1), bd->spd[1]);
         batch_move64(e, IDVS(SPD_2), bd->spd[2]);
         batch_move32(e, IDVS(TILER_FLAGS), bd->tiler_flags);
         batch_move32(e, IDVS(DCD0), mali_gfx_dcd0(d, NULL));
         batch_move32(e, IDVS(DCD1), bd->dcd1);
         batch_move32(e, IDVS(DCD2), bd->dcd2);
         lines = bd->lines;
      }
      batch_move32(e, IDVS(TILER_FLAGS2), 0);
      batch_move32(e, IDVS(VARY_SIZE), bd->vary_size);
      /* Point size 1.0 unless the shader writes it; line width. */
      batch_move32(e, IDVS(PRIMITIVE_SIZE), fui(lines ? dyn->rs.line.width : 1.0f));
      batch_move32(e, IDVS(GLOBAL_ATTRIBUTE_OFFSET), 0);
      batch_move64(e, IDVS(OQ), d->oq_addr);
   } else if (d->dirty & MALI_GFX_DIRTY_OQ) {
      /* Only the query changed. */
      batch_move32(e, IDVS(DCD0), mali_gfx_dcd0(d, dbk));
      batch_move64(e, IDVS(OQ), d->oq_addr);
   }
   if (d->dirty & MALI_GFX_DIRTY_VIEWPORT) {
      const VkViewport *vp = &dyn->vp.viewports[0];
      batch_move64(e, IDVS(SCISSOR_BOX), mali_gfx_scissor_box(cmd));
      batch_move32(e, IDVS(LOW_DEPTH_CLAMP), fui(MIN2(vp->minDepth, vp->maxDepth)));
      batch_move32(e, IDVS(HIGH_DEPTH_CLAMP), fui(MAX2(vp->minDepth, vp->maxDepth)));
   }
   if (d->dirty & MALI_GFX_DIRTY_PASS) {
      batch_move64(e, IDVS(TILER_CTX), r->tiler);
      batch_move64(e, IDVS(TSD_0), r->tsd);
   }
   if (di->indexed && (d->dirty & MALI_GFX_DIRTY_INDEX)) {
      batch_move64(e, IDVS(INDEX_BUFFER), d->ib.addr);
      batch_move32(e, IDVS(INDEX_BUFFER_SIZE), d->ib.size);
      d->dirty &= ~MALI_GFX_DIRTY_INDEX;
   }

   d->dirty &= MALI_GFX_DIRTY_INDEX;
   return true;
}

static void
draw(struct mali_cmd_buffer *cmd, const struct mali_draw_info *di)
{
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   struct mali_render_state *r = &cmd->gfx.render;

   if (!di->count || !di->instance_count)
      return;
   if (!r->active || !d->pipeline || !d->pipeline->vs) {
      vk_logw(VK_LOG_OBJS(cmd), "malisx2: draw outside a render pass or without a pipeline");
      return;
   }
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   struct idvs_batch e;
   e.d = d;
   e.n = 0;
   if (!prepare_draw(cmd, di, &e)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_VERTEX_TILER);
   batch_move32(&e, IDVS(INDEX_COUNT), di->count);
   batch_move32(&e, IDVS(INSTANCE_COUNT), di->instance_count);
   batch_move32(&e, IDVS(INDEX_OFFSET), di->indexed ? di->first_index : 0);
   batch_move32(&e, IDVS(VERTEX_OFFSET), (uint32_t)di->vertex_offset);
   /* Zero-based instance IDs; firstInstance reaches the shader as a
    * system value (panvk). */
   batch_move32(&e, IDVS(INSTANCE_OFFSET), 0);

   struct mali_primitive_flags_packed ovr;
   pan_pack_nodefaults(&ovr, PRIMITIVE_FLAGS, cfg) {
      cfg.index_type = di->indexed ? mali_gfx_index_type(d->ib.index_size) : MALI_INDEX_TYPE_NONE;
   }

   /* Per-draw timing ("draws" mode). */
   uint32_t mh = 0;
   if (unlikely(cmd->measure) && cmd->measure->draws) {
      batch_flush(b, &e);
      mh = mali_measure_begin(cmd, MALI_MEASURE_DRAW, MALI_SUBQUEUE_VERTEX_TILER,
                              cmd->draws);
      mali_measure_info(cmd, mh, di->count, di->instance_count, di->indexed,
                        cmd->passes - 1);
      uint64_t key = 0;
      if (d->pipeline->fs)
         memcpy(&key, d->pipeline->fs->key, sizeof(key));
      mali_measure_shader(cmd, mh, key);
      mali_measure_extra(cmd, mh, d->blend_shader_mask);
   }

   /* One RUN_IDVS per tiler context: a layered pass has one per 8
    * layers, and the draw's primitives may go to any of them. */
   if (r->td_count > 1) {
      batch_flush(b, &e);
      struct cs_index ctx = cs_reg64(b, IDVS(TILER_CTX));
      for (uint32_t i = 0; i < r->td_count; i++) {
         if (i)
            cs_add_imm64(b, ctx, ctx, pan_size(TILER_CONTEXT));
         emit_run_idvs(b, ovr.opaque[0]);
      }
      cs_move64_to(b, ctx, r->tiler);
   } else {
      b->req_resource_mask |= CS_IDVS_RES;
      assert(e.n < ARRAY_SIZE(e.ins));
      e.ins[e.n++] = run_idvs_ins(ovr.opaque[0]);
      batch_flush(b, &e);
   }
   r->draws++;
   cmd->draws++;
   if (unlikely(mh))
      mali_measure_end(cmd, mh);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdDraw)(VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount,
                       uint32_t firstVertex, uint32_t firstInstance)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   const struct mali_draw_info di = {
      .count = vertexCount,
      .instance_count = instanceCount,
      .vertex_offset = (int32_t)firstVertex,
      .first_instance = firstInstance,
   };
   draw(cmd, &di);
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdDrawIndexed)(VkCommandBuffer commandBuffer, uint32_t indexCount,
                              uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset,
                              uint32_t firstInstance)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   const struct mali_draw_info di = {
      .count = indexCount,
      .instance_count = instanceCount,
      .first_index = firstIndex,
      .vertex_offset = vertexOffset,
      .first_instance = firstInstance,
      .indexed = true,
   };
   draw(cmd, &di);
}

/* ---------------------------------------------------------------------- */
/* Full-screen draws                                                       */

void
MALI_PER_ARCH(cmd_run_fullscreen)(struct mali_cmd_buffer *cmd, uint64_t dcd, const VkRect2D *rect,
                                  uint32_t base_layer, uint32_t layer_count)
{
   struct mali_render_state *r = &cmd->gfx.render;
   struct cs_builder *b = mali_cmd_cs(cmd, MALI_SUBQUEUE_VERTEX_TILER);

   if (!MALI_PER_ARCH(cmd_render_tiler)(cmd)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   int minx = MAX2(rect->offset.x, 0), miny = MAX2(rect->offset.y, 0);
   int maxx = MIN2(rect->offset.x + (int)rect->extent.width, (int)r->desc.width) - 1;
   int maxy = MIN2(rect->offset.y + (int)rect->extent.height, (int)r->desc.height) - 1;
   if (maxx < minx || maxy < miny)
      return;
   struct mali_scissor_packed sc;
   pan_pack(&sc, SCISSOR, cfg) {
      cfg.scissor_minimum_x = minx;
      cfg.scissor_minimum_y = miny;
      cfg.scissor_maximum_x = maxx;
      cfg.scissor_maximum_y = maxy;
   }
   cs_move64_to(b, cs_reg64(b, IDVS(SCISSOR_BOX)), sc.opaque[0] | (uint64_t)sc.opaque[1] << 32);
   cs_move32_to(b, cs_reg32(b, IDVS(TILER_FLAGS2)), 0);

   struct cs_index dcd_reg = mali_cs_scratch_reg64(b, 0);
   cs_move64_to(b, dcd_reg, dcd);

   /* RUN_FULLSCREEN draws into the layers its tiler flags' view mask
    * selects inside the current tiler context (panvk cmd_run_fullscreen). */
   const uint32_t first_td = base_layer / MALI_LAYERS_PER_TILER_CTX;
   const uint32_t last_td = (base_layer + layer_count - 1) / MALI_LAYERS_PER_TILER_CTX;
   for (uint32_t td = first_td; td <= last_td && td < r->td_count; td++) {
      const uint32_t td_base = td * MALI_LAYERS_PER_TILER_CTX;
      const uint32_t first = MAX2(base_layer, td_base);
      const uint32_t last = MIN2(base_layer + layer_count, td_base + MALI_LAYERS_PER_TILER_CTX);
      struct mali_primitive_flags_packed tf;
      pan_pack(&tf, PRIMITIVE_FLAGS, cfg) {
         cfg.low_depth_cull = false;
         cfg.high_depth_cull = false;
         cfg.view_mask = BITFIELD_RANGE(first - td_base, last - first);
      }
      cs_move64_to(b, cs_reg64(b, IDVS(TILER_CTX)), r->tiler + td * pan_size(TILER_CONTEXT));
      cs_move32_to(b, cs_reg32(b, IDVS(TILER_FLAGS)), tf.opaque[0]);
      cs_run_fullscreen(b, 0, dcd_reg);
   }

   r->draws++;
   cmd->gfx.draw.dirty |= MALI_GFX_DIRTY_PIPELINE | MALI_GFX_DIRTY_VIEWPORT |
                          MALI_GFX_DIRTY_PASS;
   /* The scissor, tiler flags and tiler context registers changed behind
    * the draws' record of them. */
   cmd->gfx.draw.regs_valid = 0;
}

/* A primitive barrier over the whole framebuffer: later fragments of the
 * pass wait for earlier ones in each tile (panvk cmd_fb_barrier). */
void
MALI_PER_ARCH(cmd_fb_barrier)(struct mali_cmd_buffer *cmd)
{
   struct mali_render_state *r = &cmd->gfx.render;
   struct mali_ptr zsd = mali_cmd_alloc(cmd, pan_size(DEPTH_STENCIL), 32);
   struct mali_ptr dcd = mali_cmd_alloc(cmd, pan_size(DRAW), 64);
   if (!zsd.cpu || !dcd.cpu)
      return;

   pan_cast_and_pack(zsd.cpu, DEPTH_STENCIL, cfg) {
      cfg.stencil_test_enable = false;
      cfg.depth_write_enable = false;
      cfg.depth_function = MALI_FUNC_ALWAYS;
   }
   pan_cast_and_pack(dcd.cpu, DRAW, cfg) {
      cfg.flags_0.allow_forward_pixel_to_kill = false;
      cfg.flags_0.allow_forward_pixel_to_be_killed = false;
      cfg.flags_0.primitive_barrier = true;
      cfg.flags_0.occlusion_query = MALI_OCCLUSION_MODE_DISABLED;
      cfg.flags_2.read_mask = 0;
      cfg.flags_2.write_mask = 0;
      cfg.flags_2.no_shader_depth_read = true;
      cfg.flags_2.no_shader_stencil_read = true;
      cfg.depth_stencil = zsd.gpu;
   }
   const VkRect2D all = {{0, 0}, {r->desc.width, r->desc.height}};
   MALI_PER_ARCH(cmd_run_fullscreen)(cmd, dcd.gpu, &all, 0, MAX2(r->desc.layer_count, 1));
}
