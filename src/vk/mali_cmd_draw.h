/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * What a draw puts in the vertex/tiler stream before its RUN_IDVS
 * (RUN_IDVS2 on v12), for
 * the direct draws (mali_cmd_draw.c) and the indirect ones
 * (mali_cmd_draw_indirect.c): the staging-register moves of the draw's
 * dirty state, collected and copied into the stream in one piece, the
 * fragment shader's pushed uniform-buffer words, and RUN_IDVS itself.
 *
 * Everything here is static and included by those two files only. The
 * indirect draws are a file of their own so that the direct draw, which
 * every ARMSX2 draw runs, is compiled with these helpers having one caller
 * each, as it always was (inlined the same way).
 */

#ifndef MALI_CMD_DRAW_H
#define MALI_CMD_DRAW_H

#include "mali_cmd_state.h"

#if PAN_ARCH < 10
#error "mali_cmd_draw.h is written for the command-stream archs (v10, v11, v12)"
#endif

#define IDVS(reg) MALI_IDVS_SR_##reg

/* The staging registers of a draw's resources. v12 names them per stage
 * (one vertex shader, no position/varying pair: IDVS2); on v11 the vertex
 * stage reads group 0 for position and group 1 (SPD_1) for varyings, the
 * fragment stage group 2, both TSD_0 (run_idvs_ins). */
#if PAN_ARCH >= 12
#define IDVS_VS_FAU IDVS(VERTEX_FAU)
#define IDVS_FS_FAU IDVS(FRAGMENT_FAU)
#define IDVS_VS_SRT IDVS(VERTEX_SRT)
#define IDVS_FS_SRT IDVS(FRAGMENT_SRT)
#define IDVS_FS_SPD IDVS(FRAGMENT_SPD)
#else
#define IDVS_VS_FAU IDVS(FAU_0)
#define IDVS_FS_FAU IDVS(FAU_2)
#define IDVS_VS_SRT IDVS(SRT_0)
#define IDVS_FS_SRT IDVS(SRT_2)
#define IDVS_FS_SPD IDVS(SPD_2)
#endif

/* The staging registers RUN_IDVS reads, by bit in
 * mali_gfx_draw_state::regs_valid: 0..65 on v12, whose last is OQ, and
 * 0..63 on v10 and v11, where OQ (46) is followed by VARY_SIZE, BLEND_DESC,
 * ZSD, INDEX_BUFFER, TILER_FLAGS, the DCD words and PRIMITIVE_SIZE. */
#if PAN_ARCH >= 12
#define MALI_IDVS_SR_COUNT (IDVS(OQ) + 2)
#else
#define MALI_IDVS_SR_COUNT 64
#endif

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

#if PAN_ARCH >= 12
/* RUN_IDVS2 with the vertex stage shaded early, as panvk v12. The stages
 * read their own staging registers (IDVS_VS_*, IDVS_FS_*). gl_DrawID is
 * the register draw_id_reg names, or 0 when it is negative. */
static uint64_t
run_idvs_ins(uint32_t flags_override, int draw_id_reg)
{
   struct mali_cs_run_idvs2_packed ins;
   pan_pack(&ins, CS_RUN_IDVS2, I) {
      I.flags_override = flags_override;
      I.malloc_enable = MALI_CS_MEMORY_ALLOCATION_ENABLE_MALLOC;
      if (draw_id_reg >= 0) {
         I.draw_id_register_enable = MALI_CS_DRAW_ID_REGISTER_ENABLE_DRAWID;
         I.draw_id = draw_id_reg;
      } else {
         I.draw_id_register_enable = MALI_CS_DRAW_ID_REGISTER_ENABLE_NO_DRAWID;
      }
      I.vertex_shading_mode = MALI_IDVS_SHADING_MODE_EARLY;
   }
   uint64_t v;
   memcpy(&v, &ins, sizeof(v));
   return v;
}
#else
/* RUN_IDVS, with bit 55 set on v11 (mali_cmd_draw.c). Selects as panvk: varying stage from
 * group 0/1 (SPD_1), fragment from group 2, both with TSD_0. gl_DrawID is
 * the register draw_id_reg names, or 0 when it is negative. */
static uint64_t
run_idvs_ins(uint32_t flags_override, int draw_id_reg)
{
   struct mali_cs_run_idvs_packed ins;
   pan_pack(&ins, CS_RUN_IDVS, I) {
      I.flags_override = flags_override;
      I.malloc_enable = true;
      I.draw_id_register_enable = draw_id_reg >= 0;
      if (draw_id_reg >= 0)
         I.draw_id = draw_id_reg;
      I.varying_srt_select = false;
      I.varying_fau_select = false;
      I.varying_tsd_select = false;
      I.fragment_srt_select = true;
      I.fragment_tsd_select = false;
   }
#if PAN_ARCH >= 11
   if (mali_idvs_bit55)
      ins.opaque[1] |= 1u << (55 - 32);
#endif
   uint64_t v;
   memcpy(&v, &ins, sizeof(v));
   return v;
}
#endif

static void
emit_run_idvs(struct cs_builder *b, uint32_t flags_override, int draw_id_reg)
{
   cs_flush_loads(b);
   b->req_resource_mask |= CS_IDVS_RES;
   const uint64_t ins = run_idvs_ins(flags_override, draw_id_reg);
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
   assert(reg < MALI_IDVS_SR_COUNT);
   if (BITSET_TEST(d->regs_valid, reg) && d->regs[reg] == v)
      return;
   BITSET_SET(d->regs_valid, reg);
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
   assert(reg + 1 < MALI_IDVS_SR_COUNT);
   const bool lo_ok = BITSET_TEST(d->regs_valid, reg) && d->regs[reg] == lo;
   const bool hi_ok = BITSET_TEST(d->regs_valid, reg + 1) && d->regs[reg + 1] == hi;
   if (lo_ok && hi_ok)
      return;
   if (!lo_ok && !hi_ok && v < (1ull << 48)) {
      /* MOVE48 zero-extends into the high register. */
      BITSET_SET(d->regs_valid, reg);
      BITSET_SET(d->regs_valid, reg + 1);
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
      batch_move64(e, IDVS_VS_FAU, v);
   }
   if (d->dirty & MALI_GFX_DIRTY_FS_FAU) {
      if (!mali_gfx_build_fs_fau(cmd, &sv, &v))
         return false;
      /* Straight into the stream, ahead of the batched moves and RUN_IDVS. */
      if (p->fs && p->fs->fau.ubo_push_count)
         emit_ubo_push(cmd, e, p->fs, v & BITFIELD64_MASK(56));
      batch_move64(e, IDVS_FS_FAU, v);
   }
   if (d->dirty & MALI_GFX_DIRTY_VS_SRT) {
      if (!mali_gfx_build_vs_srt(cmd, &v))
         return false;
      batch_move64(e, IDVS_VS_SRT, v);
   }
   if (d->dirty & MALI_GFX_DIRTY_FS_SRT) {
      if (!mali_gfx_build_fs_srt(cmd, &v))
         return false;
      batch_move64(e, IDVS_FS_SRT, v);
   }
   if (d->dirty & MALI_GFX_DIRTY_ZSD) {
      if (!mali_gfx_build_zsd(cmd, dbk, &v))
         return false;
      batch_move64(e, IDVS(ZSD), v);
   }
   if (d->dirty & MALI_GFX_DIRTY_PIPELINE) {
      bool lines;
      if (unlikely(dbk)) {
#if PAN_ARCH >= 12
         batch_move64(e, IDVS(VERTEX_SPD), dbk->vs_pos_spd);
#else
         batch_move64(e, IDVS(SPD_0), dbk->vs_pos_spd);
         batch_move64(e, IDVS(SPD_1), dbk->vs_var_spd);
#endif
         batch_move64(e, IDVS_FS_SPD, dbk->fs_spd);
         batch_move32(e, IDVS(TILER_FLAGS), dbk->tiler_flags);
         batch_move32(e, IDVS(DCD0), mali_gfx_dcd0(d, dbk));
         batch_move32(e, IDVS(DCD1), dbk->dcd1);
         batch_move32(e, IDVS(DCD2), dbk->dcd2);
         lines = u_reduced_prim(dbk->prim) == MESA_PRIM_LINES;
      } else {
#if PAN_ARCH >= 12
         batch_move64(e, IDVS(VERTEX_SPD), bd->spd[0]);
#else
         batch_move64(e, IDVS(SPD_0), bd->spd[0]);
         batch_move64(e, IDVS(SPD_1), bd->spd[1]);
#endif
         batch_move64(e, IDVS_FS_SPD, bd->spd[2]);
         batch_move32(e, IDVS(TILER_FLAGS), bd->tiler_flags);
         batch_move32(e, IDVS(DCD0), mali_gfx_dcd0(d, NULL));
         batch_move32(e, IDVS(DCD1), bd->dcd1);
         batch_move32(e, IDVS(DCD2), bd->dcd2);
         lines = bd->lines;
      }
#if PAN_ARCH >= 11
      batch_move32(e, IDVS(TILER_FLAGS2), 0);
#endif
      batch_move32(e, IDVS(VARY_SIZE), bd->vary_size);
      /* Point size 1.0 unless the shader writes it; line width. */
      batch_move32(e, IDVS(PRIMITIVE_SIZE),
                   fui(lines ? CLAMP(dyn->rs.line.width, 1.0f, MALI_MAX_LINE_WIDTH) : 1.0f));
      batch_move32(e, IDVS(GLOBAL_ATTRIBUTE_OFFSET), 0);
      batch_move64(e, IDVS(OQ), d->oq_addr);
   } else if (d->dirty & MALI_GFX_DIRTY_OQ) {
      /* Only the query changed. */
      batch_move32(e, IDVS(DCD0), mali_gfx_dcd0(d, dbk));
      batch_move64(e, IDVS(OQ), d->oq_addr);
   }
   if (d->dirty & MALI_GFX_DIRTY_VIEWPORT) {
      batch_move64(e, IDVS(SCISSOR_BOX), mali_gfx_scissor_box(cmd));
#if PAN_ARCH >= 12
      uint64_t words[2];
      mali_gfx_viewport(cmd, words);
      batch_move64(e, IDVS(VIEWPORT_HIGH), words[0]);
      batch_move64(e, IDVS(VIEWPORT_LOW), words[1]);
#else
      const VkViewport *vp = &dyn->vp.viewports[d->vp_pass];
      batch_move32(e, IDVS(LOW_DEPTH_CLAMP), fui(MIN2(vp->minDepth, vp->maxDepth)));
      batch_move32(e, IDVS(HIGH_DEPTH_CLAMP), fui(MAX2(vp->minDepth, vp->maxDepth)));
#endif
   }
   if (d->dirty & MALI_GFX_DIRTY_PASS) {
      batch_move64(e, IDVS(TILER_CTX), r->tiler);
#if PAN_ARCH >= 12
      batch_move64(e, IDVS(VERTEX_TSD), r->tsd);
      batch_move64(e, IDVS(FRAGMENT_TSD), r->tsd);
#else
      batch_move64(e, IDVS(TSD_0), r->tsd);
#endif
   }
   if (di->indexed && (d->dirty & MALI_GFX_DIRTY_INDEX)) {
      batch_move64(e, IDVS(INDEX_BUFFER), d->ib.addr);
      batch_move32(e, IDVS(INDEX_BUFFER_SIZE), d->ib.size);
      d->dirty &= ~MALI_GFX_DIRTY_INDEX;
   }

   d->dirty &= MALI_GFX_DIRTY_INDEX;
   return true;
}

/* How many times a draw runs: once per viewport for a pipeline whose
 * vertex shader writes gl_ViewportIndex (mali_shader.c, "Viewport
 * index"), else once. */
static inline uint32_t
viewport_passes(const struct mali_cmd_buffer *cmd)
{
   const struct mali_graphics_pipeline *p = cmd->gfx.draw.pipeline;
   const uint32_t n = cmd->vk.dynamic_graphics_state.vp.viewport_count;
   return p && p->vs && p->vs->vs.viewport_index && n > 1 ? n : 1;
}

/* Before pass i of a draw that runs once per viewport, and after the last
 * one (i = 0): the viewport, the sysvals that name it, and the scissor. */
static inline void
viewport_pass_begin(struct mali_cmd_buffer *cmd, uint32_t i)
{
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   if (d->vp_pass == i)
      return;
   d->vp_pass = i;
   d->dirty |= MALI_GFX_DIRTY_VIEWPORT | MALI_GFX_DIRTY_VS_FAU | MALI_GFX_DIRTY_FS_FAU;
}

#endif
