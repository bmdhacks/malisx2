/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Render passes on the job manager (v7 and v9): the tiler side (heap slot,
 * Tiler Contexts, the pass's Local Storage descriptor), the fragment jobs,
 * and the full-screen jobs (in-pass clears and the in-pass by-region
 * barrier).
 * What does not depend on the frontend (tile-buffer layout, preloads and
 * frame shaders, framebuffer descriptors, vkCmdBeginRendering) is in
 * mali_fb.c, shared with the v11 back half.
 *
 * A pass lives inside one batch (mali_jm.h), unless it reaches the pass
 * job limit (MALI_JM_PASS_MAX_JOBS): then it goes on in a new batch on the
 * same tiler memory (mali_jm_cmd_pass_split). Its order of events:
 *
 *  - begin: the open batch is closed first if it is full (its pass count,
 *    vtc job count or heap estimate); then the shared layout. A pass that
 *    preloads reads earlier attachment writes through the texture unit,
 *    which a fragment -> fragment barrier with a cache flush on the
 *    fragment chain covers (the v11 pass does the same with its stream
 *    barrier);
 *  - first tiled job (a draw, a full-screen job): the barriers owed to vtc
 *    work are applied here and not at begin (mali_jm_cmd_vtc), because
 *    the batch may still close at this point (nothing of the pass is in it
 *    yet). Then the batch takes a heap slot if it has none
 *    (mali_jm_heap.c), and the pass gets one 192-byte Tiler Context per
 *    layer, every word written, all pointing at the batch's one heap
 *    descriptor. A pass that never tiles has no contexts (the FBD's tiler
 *    pointer stays 0, as Mesa's gallium driver does for a clear-only
 *    batch) and no heap slot;
 *  - end: the framebuffer descriptors (mali_fb.c, one per layer) and one
 *    128-byte Fragment job per layer in the open batch's current fragment
 *    segment, bounding box in 16-pixel tiles. Pending fragment-side
 *    barriers are applied by mali_jm_cmd_frag (a Barrier bit and Cache
 *    Flush job in the chain, or a new fragment atom where the fragment
 *    slot runs no Cache Flush jobs). A pass with nothing to store writes
 *    nothing.
 *
 * Differences from the blob's passes and why:
 *
 *  - no heap-setup compute job and no JIT atoms: the CPU owns the heap
 *    memory and writes the descriptor (mali_jm_heap.c);
 *  - one Tiler Context per layer, not two: the blob points full-screen jobs
 *    at a second context per layer (hierarchy mask 0x1000); we point them
 *    at the layer's own context, as Mesa does (to be confirmed on the
 *    device);
 *  - Layer offset 0 in every context. Each layer has its own context and
 *    polygon lists, and the FBD selects internal layer 0. The blob writes
 *    -i into layer i's draw context, so that a primitive's layer index
 *    picks its context. A full-screen job has no shader to give its
 *    primitive a layer; if the tiler takes it as layer 0, an offset of -i
 *    would drop it from every layer but the first. Noysz's arch-9 port
 *    runs one context per layer with offset 0 and passes the layered
 *    rendering tests on a G57. To be confirmed on the device. A draw of a
 *    layered pass goes to layer 0's context only: no job-manager shader
 *    writes a layer, so all its primitives are in layer 0
 *    (mali_jm_cmd_draw.c, mali_jm_cmd_draw_v7.c);
 *  - all layers' fragment jobs in one fragment chain (the blob has a chain
 *    per layer): they need no order between them;
 *  - transaction elimination (CRC) is off on the job manager until the
 *    device shows it pays: no image has CRC state, so no pass picks a CRC
 *    target. Turning it on means putting the image's seed into the CRC
 *    Clear Color here, on the CPU, from a per-image seed counter bumped at
 *    record time.
 *
 * Bifrost (v7) has no Fullscreen job: its in-pass clears and fragment
 * barrier are Tiler jobs drawing a rectangle from CPU-written positions
 * (mali_v7_cmd_meta_quads), and its frame shaders get one Draw triple per
 * layer (mali_fb.c, mali_cmd_meta_gfx.c).
 */

#include "mali_cmd_state.h"

#include "vk_log.h"

#include "mali_measure.h"

#if PAN_ARCH != 7 && PAN_ARCH != 9
#error "mali_jm_cmd_render.c is the job manager back half (PAN_ARCH 7 or 9)"
#endif

/* What one tiled pass adds to its batch's heap estimate on top of its
 * draws: bin pointers and the first polygon-list blocks. A guess, to be
 * measured on the device; it only matters for batches of hundreds
 * of passes, which the pass limit closes first. */
#define MALI_JM_HEAP_PASS_EST (64 * 1024)

_Static_assert(MALI_LAYERS_PER_TILER_CTX == 1, "one Tiler Context per layer on v7 and v9");

/* ---------------------------------------------------------------------- */
/* Begin                                                                   */

void
MALI_PER_ARCH(cmd_render_begin)(struct mali_cmd_buffer *cmd, const struct mali_render_desc *desc)
{
   /* The batch limits (mali_jm.h): a pass's tiling stays in one
    * batch, so a full batch closes before the pass starts. */
   if (cmd->jm.open) {
      const struct mali_jm_batch *b = &cmd->jm.cur;
      if (b->passes >= MALI_JM_BATCH_MAX_PASSES || b->vtc.jobs >= MALI_JM_BATCH_MAX_JOBS ||
          b->est_heap_bytes >= MALI_JM_HEAP_SLOT_SIZE / 2)
         mali_jm_cmd_batch_close(cmd);
   }

   if (MALI_PER_ARCH(fb_begin)(cmd, desc)) {
      /* Preloads go through the texture unit: earlier attachment writes of
       * this batch must be done and its caches invalidated (applied to the
       * pass's first fragment job, mali_jm_cmd_frag). */
      mali_jm_cmd_barrier(cmd,
                          VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                             VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
                             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                          VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                          VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
   }
}

/* ---------------------------------------------------------------------- */
/* Tiler side                                                              */

/* The tiler's hierarchy levels: the blob's fixed 0xaa (levels 1, 3, 5, 7)
 * when the tiler can keep four levels active, which covers framebuffers
 * up to 2048 pixels; otherwise Mesa's spread for fewer levels
 * (pan_select_tiler_hierarchy_mask). To be checked against the G57's
 * TILER_FEATURES on the device. */
static uint32_t
hierarchy_mask(struct mali_device *dev)
{
   static const uint32_t fewer[4] = {MALI_TILER_HIERARCHY_MASK, 0x80, 0x82, 0xa2};
   const unsigned levels = mali_device_physical(dev)->props.tiler_max_active_levels;
   return levels < 4 ? fewer[levels] : MALI_TILER_HIERARCHY_MASK;
}

/*
 * One layer's Tiler Context, packed on the CPU and copied into command
 * memory (write-combined, never read back), noted so a command buffer
 * that runs again gets it restored. Takes only data that stays the same
 * for the pass's lifetime (r->desc, the pass's hierarchy mask and heap
 * descriptor), so it can be called again later to fix up a field without
 * redoing allocation or drifting from the initial build: a second
 * note_reset for the same destination is applied after the first on the
 * next resubmit (mali_jm_cmd_note_reset appends, mali_jm_cmd_prepare_submit
 * runs the list in order), so its template wins.
 */
static void
pack_tiler_context(struct mali_cmd_buffer *cmd, const struct mali_render_state *r,
                   uint32_t mask, uint64_t heap, void *dst)
{
   struct mali_tiler_context_packed tc;
   pan_pack(&tc, TILER_CONTEXT, cfg) {
      cfg.polygon_list = 0;
      cfg.hierarchy_mask = mask;
      cfg.sample_pattern = MALI_SAMPLE_PATTERN_SINGLE_SAMPLED;
#if PAN_ARCH >= 9
      cfg.first_provoking_vertex = r->first_provoking_vertex;
#endif
      cfg.fb_width = r->desc.width;
      cfg.fb_height = r->desc.height;
#if PAN_ARCH >= 9
      cfg.layer_count = 1;
      cfg.layer_offset = 0;
#else
      /* v7: no layer fields (one context per layer either way), and the
       * provoking vertex is a Primitive bit of each Tiler job (Mesa's
       * pan_jm.c jm_emit_tiler_desc sets it here only on v9). */
#endif
      cfg.heap = heap;
   }
   memcpy(dst, &tc, sizeof(tc));
   mali_jm_cmd_note_reset(cmd, dst, &tc, sizeof(tc));
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

   /* The barriers owed to vtc work; the batch may close here. */
   if (!mali_jm_cmd_vtc(cmd))
      return false;
   struct mali_jm_batch *b = &cmd->jm.cur;
   if (b->heap_slot < 0 && !mali_jm_heap_take(cmd, b))
      return false;

   const uint32_t layers = MAX2(r->desc.layer_count, 1);
   struct mali_ptr p = mali_cmd_alloc(cmd, (uint64_t)layers * pan_size(TILER_CONTEXT), 64);
   if (!p.cpu)
      return false;

   const uint32_t mask = hierarchy_mask(cmd->dev);
   for (uint32_t i = 0; i < layers; i++)
      pack_tiler_context(cmd, r, mask, b->heap_desc,
                        (uint8_t *)p.cpu + (uint64_t)i * pan_size(TILER_CONTEXT));
   r->tiler = p.gpu;
   r->tiler_cpu = p.cpu;
   r->td_count = layers;
   b->est_heap_bytes += MALI_JM_HEAP_PASS_EST;

   /* Timing: the vertex/tiler side of the pass starts here, a System
    * Timestamp Write Value job in the batch's vtc chain (the same chain
    * the pass's draws go into, mali_jm_cmd_pass_vtc). */
   if (unlikely(cmd->jm.measure))
      r->measure_vt =
         mali_jm_measure_begin(cmd, MALI_MEASURE_PASS_VT, &b->vtc, MALI_JM_SLOT_VTC,
                               cmd->passes - 1);

   cmd->gfx.draw.dirty |= MALI_GFX_DIRTY_PASS;
   return true;
}

void
MALI_PER_ARCH(cmd_render_tiler_set_provoking_vertex)(struct mali_cmd_buffer *cmd)
{
   struct mali_render_state *r = &cmd->gfx.render;
   const uint32_t mask = hierarchy_mask(cmd->dev);
   /* Pass-split carries heap_slot/heap_desc over unchanged (mali_jm_cmd_pass_split
    * below), so the current batch's heap descriptor is still the pass's one. */
   const uint64_t heap = cmd->jm.cur.heap_desc;
   for (uint32_t i = 0; i < r->td_count; i++)
      pack_tiler_context(cmd, r, mask, heap,
                        (uint8_t *)r->tiler_cpu + (uint64_t)i * pan_size(TILER_CONTEXT));
}

/* The pass job limit (MALI_JM_PASS_MAX_JOBS, mali_jm.h): end the open
 * batch's vtc chain here and go on in a new batch that tiles through the
 * same heap slot and descriptor. The pass's tiler contexts and Local
 * Storage descriptor are in command memory and stay as they are. */
void
mali_jm_cmd_pass_split(struct mali_cmd_buffer *cmd)
{
   const struct mali_jm_batch old = cmd->jm.cur;
   mali_jm_cmd_batch_close(cmd);
   struct mali_jm_batch *b = mali_jm_cmd_batch(cmd);
   b->heap_slot = old.heap_slot;
   b->heap_desc = old.heap_desc;
   b->est_heap_bytes = old.est_heap_bytes;
}

/* ---------------------------------------------------------------------- */
/* Full-screen jobs                                                        */

#if PAN_ARCH < 9
/* The job's fields follow the Tiler job of Mesa's pre-Bifrost preload
 * (gallium pan_fb_preload.c pan_preload_emit_tiler_job): a triangle strip
 * of 4, job task split 6, fixed primitive size 1, Invocation (1, 4, 1). */
void
mali_v7_cmd_meta_quads(struct mali_cmd_buffer *cmd, const void *draw, const VkRect2D *rect,
                       uint32_t base_layer, uint32_t layer_count, bool barrier)
{
   struct mali_render_state *r = &cmd->gfx.render;
   if (!MALI_PER_ARCH(cmd_render_tiler)(cmd)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   const int minx = MAX2(rect->offset.x, 0), miny = MAX2(rect->offset.y, 0);
   const int maxx = MIN2(rect->offset.x + (int)rect->extent.width, (int)r->desc.width) - 1;
   const int maxy = MIN2(rect->offset.y + (int)rect->extent.height, (int)r->desc.height) - 1;
   if (maxx < minx || maxy < miny)
      return;

   const uint64_t position = mali_v7_meta_rect_positions(cmd, (float)minx, (float)miny,
                                                         (float)(maxx + 1), (float)(maxy + 1));
   const uint64_t viewport = mali_v7_meta_viewport(cmd, minx, miny, maxx, maxy);
   if (!position || !viewport)
      return;

   /* Bytes 32-255 of the job: the sections, zero between them. */
   uint8_t tmpl[256] __attribute__((aligned(16)));
   STATIC_ASSERT(sizeof(tmpl) == pan_size(TILER_JOB));
   memset(tmpl, 0, sizeof(tmpl));
   /* One instance of four vertices, with the graphics quirks. */
   const uint32_t local[3] = {1, 1, 1}, groups[3] = {1, 4, 1};
   mali_v7_pack_invocation(pan_section_ptr(tmpl, TILER_JOB, INVOCATION), local, groups, true);
   pan_section_pack(tmpl, TILER_JOB, PRIMITIVE, cfg) {
      cfg.draw_mode = MALI_DRAW_MODE_TRIANGLE_STRIP;
      cfg.index_count = 4;
      cfg.job_task_split = 6;
      cfg.first_provoking_vertex = r->first_provoking_vertex;
   }
   pan_section_pack(tmpl, TILER_JOB, PRIMITIVE_SIZE, cfg) {
      cfg.fixed_sized = 1.0f;
   }
   pan_section_pack(tmpl, TILER_JOB, PADDING, cfg);

   /* The template has position and viewport 0 and the genxml defaults
    * elsewhere (the callers pack it with pan_pack), so ORing in a Draw
    * packed with the same defaults and these two pointers sets just them.
    * (pan_pack_nodefaults cannot pack a Draw: the padded instance sizes
    * have no encoding for 0.) */
   struct mali_draw_packed dcd, part;
   memcpy(&dcd, draw, sizeof(dcd));
   pan_pack(&part, DRAW, cfg) {
      cfg.position = position;
      cfg.viewport = viewport;
   }
   pan_merge(&dcd, &part, DRAW);
   memcpy(pan_section_ptr(tmpl, TILER_JOB, DRAW), &dcd, sizeof(dcd));

   const uint32_t end = MIN2(base_layer + layer_count, r->td_count);
   for (uint32_t l = base_layer; l < end; l++) {
      pan_section_pack(tmpl, TILER_JOB, TILER, cfg) {
         cfg.address = mali_jm_pass_tiler(r, l);
      }
      struct mali_jm_chain *c = mali_jm_cmd_pass_vtc(cmd);
      struct mali_ptr job =
         mali_jm_cmd_add_job(cmd, c, MALI_JOB_TYPE_TILER, pan_size(TILER_JOB), barrier, 0);
      if (!job.cpu)
         return;
      memcpy((uint8_t *)job.cpu + 32, tmpl + 32, sizeof(tmpl) - 32);
      /* Polygon-list entries for two triangles, the draws' guess. */
      mali_jm_cmd_heap_use(cmd, 2 * 32);
   }
   r->draws++;
}
#else

/*
 * Fullscreen jobs (type 12, 128 bytes) running the Draw descriptor dcd over
 * rect in layers [base_layer, base_layer + layer_count), one per layer, on
 * the layer's tiler context. They are tiler-side jobs in the batch's vtc
 * chain (Dependency 2 = the previous tiler-side job). barrier: the header's
 * Barrier bit, which the blob sets on its in-pass clears.
 */
static void
fullscreen_jobs(struct mali_cmd_buffer *cmd, uint64_t dcd, const VkRect2D *rect,
                uint32_t base_layer, uint32_t layer_count, bool barrier)
{
   struct mali_render_state *r = &cmd->gfx.render;
   if (!MALI_PER_ARCH(cmd_render_tiler)(cmd)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   const int minx = MAX2(rect->offset.x, 0), miny = MAX2(rect->offset.y, 0);
   const int maxx = MIN2(rect->offset.x + (int)rect->extent.width, (int)r->desc.width) - 1;
   const int maxy = MIN2(rect->offset.y + (int)rect->extent.height, (int)r->desc.height) - 1;
   if (maxx < minx || maxy < miny)
      return;

   /* Bytes 32-127 of the job: the sections, and zero between them. */
   uint8_t tmpl[128] __attribute__((aligned(16)));
   memset(tmpl, 0, sizeof(tmpl));
   /* As Mesa's v9 full-screen draw (jm_launch_draw_fullscreen): no draw
    * mode, no indices, the genxml defaults otherwise. */
   pan_section_pack(tmpl, FULLSCREEN_JOB, PRIMITIVE, cfg) {
      cfg.scissor_array_enable = false;
   }
   pan_section_pack(tmpl, FULLSCREEN_JOB, DCD, cfg) {
      cfg.address = dcd;
   }
   pan_section_pack(tmpl, FULLSCREEN_JOB, SCISSOR, cfg) {
      cfg.scissor_minimum_x = minx;
      cfg.scissor_minimum_y = miny;
      cfg.scissor_maximum_x = maxx;
      cfg.scissor_maximum_y = maxy;
   }

   const uint32_t end = MIN2(base_layer + layer_count, r->td_count);
   for (uint32_t l = base_layer; l < end; l++) {
      pan_section_pack(tmpl, FULLSCREEN_JOB, TILER, cfg) {
         cfg.address = mali_jm_pass_tiler(r, l);
      }
      struct mali_jm_chain *c = mali_jm_cmd_pass_vtc(cmd);
      struct mali_ptr job = mali_jm_cmd_add_job(cmd, c, MALI_JOB_TYPE_FULLSCREEN,
                                                pan_size(FULLSCREEN_JOB), barrier, 0);
      if (!job.cpu)
         return;
      memcpy((uint8_t *)job.cpu + 32, tmpl + 32, sizeof(tmpl) - 32);
   }
   r->draws++;
}

void
MALI_PER_ARCH(cmd_run_fullscreen)(struct mali_cmd_buffer *cmd, uint64_t dcd, const VkRect2D *rect,
                                  uint32_t base_layer, uint32_t layer_count)
{
   fullscreen_jobs(cmd, dcd, rect, base_layer, layer_count, true);
}
#endif /* PAN_ARCH >= 9 */

/*
 * A by-region fragment -> fragment barrier inside a pass: later
 * fragments of the pass wait in each tile for earlier ones. One draw per
 * layer whose Draw descriptor sets Primitive Barrier, with no fragment
 * shader, both forward-pixel-kill bits clear and a depth/stencil
 * descriptor that tests ALWAYS and writes nothing (panvk's v11
 * cmd_fb_barrier and the blob both). The blob draws it as a 4-vertex quad
 * in a Malloc Vertex job with an internal vertex shader; a Fullscreen job
 * carrying the same descriptor is Mesa's shape and needs no shader. No
 * Cache Flush job and no header Barrier bit: the ordering is the tiler's
 * and the fragment front end's, inside each tile.
 */
void
MALI_PER_ARCH(cmd_fb_barrier)(struct mali_cmd_buffer *cmd)
{
#if PAN_ARCH < 9
   /* v7: the same draw as a Tiler job rectangle over the framebuffer
    * (mali_v7_cmd_meta_quads), with Primitive Barrier in the Draw section
    * and a Renderer State Descriptor without a shader (panvk_v7's
    * panvk_draw_prepare_fs_rsd without a fragment shader): fixed-function
    * depth tested ALWAYS and not written, stencil off, every Blend
    * descriptor off, and both forward-pixel-kill bits clear as on v9. */
   struct mali_render_state *r = &cmd->gfx.render;
   if (!MALI_PER_ARCH(cmd_render_tiler)(cmd)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }
   struct mali_ptr rsd =
      mali_cmd_alloc(cmd, pan_size(RENDERER_STATE) + r->rt_count * pan_size(BLEND),
                     pan_alignment(RENDERER_STATE));
   if (!rsd.cpu)
      return;
   pan_cast_and_pack(rsd.cpu, RENDERER_STATE, cfg) {
      cfg.properties.depth_source = MALI_DEPTH_SOURCE_FIXED_FUNCTION;
      cfg.properties.allow_forward_pixel_to_kill = false;
      cfg.properties.allow_forward_pixel_to_be_killed = false;
      cfg.properties.zs_update_operation = MALI_PIXEL_KILL_FORCE_EARLY;
      cfg.properties.pixel_kill_operation = MALI_PIXEL_KILL_FORCE_EARLY;
      cfg.multisample_misc.sample_mask = 0xffff;
      cfg.multisample_misc.depth_function = MALI_FUNC_ALWAYS;
      cfg.multisample_misc.depth_write_mask = false;
      cfg.stencil_mask_misc.stencil_enable = false;
   }
   for (unsigned i = 0; i < r->rt_count; i++) {
      pan_cast_and_pack((uint8_t *)rsd.cpu + pan_size(RENDERER_STATE) + i * pan_size(BLEND), BLEND,
                        cfg) {
         cfg.enable = false;
         cfg.internal.mode = MALI_BLEND_MODE_OFF;
         /* Not read in this mode; the v7 packer asserts it is at least 1. */
         cfg.internal.fixed_function.num_comps = 4;
      }
   }
   struct mali_draw_packed dcd;
   pan_pack(&dcd, DRAW, cfg) {
      cfg.primitive_barrier = true;
      cfg.occlusion_query = MALI_OCCLUSION_MODE_DISABLED;
      cfg.state = rsd.gpu;
      cfg.thread_storage = r->tsd;
   }
   const VkRect2D all = {{0, 0}, {r->desc.width, r->desc.height}};
   mali_v7_cmd_meta_quads(cmd, &dcd, &all, 0, MAX2(r->desc.layer_count, 1), false);
#else
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
      cfg.vertex_array.packet = true;
      cfg.minimum_z = 0.0f;
      cfg.maximum_z = 1.0f;
      cfg.depth_stencil = zsd.gpu;
   }
   const VkRect2D all = {{0, 0}, {r->desc.width, r->desc.height}};
   fullscreen_jobs(cmd, dcd.gpu, &all, 0, MAX2(r->desc.layer_count, 1), false);
#endif /* PAN_ARCH >= 9 */
}

/* ---------------------------------------------------------------------- */
/* End                                                                     */

/* One Fragment job per layer: bounding box in 16-pixel tiles (Mesa
 * pan_emit_fragment_job_payload), the layer's FBD pointer, tagged. */
static void
fragment_jobs(struct mali_cmd_buffer *cmd, struct mali_render_state *r, uint64_t fbds,
              uint32_t fbd_size)
{
   struct mali_jm_chain *c = mali_jm_cmd_frag(cmd, false);
   if (!c)
      return;

   /* Timing: the fragment side, from here (the current fragment segment,
    * not forced new -- a timed pass that preloads through a pending
    * barrier still gets that barrier on its first job below) to the end
    * of this pass's fragment jobs. */
   if (unlikely(cmd->jm.measure))
      r->measure_frag =
         mali_jm_measure_begin(cmd, MALI_MEASURE_PASS_FRAG, c, MALI_JM_SLOT_FRAG,
                               cmd->passes - 1);

   uint8_t tmpl[128] __attribute__((aligned(16)));
   memset(tmpl, 0, sizeof(tmpl));
   const uint32_t layers = MAX2(r->desc.layer_count, 1);
   for (uint32_t l = 0; l < layers; l++) {
      pan_section_pack(tmpl, FRAGMENT_JOB, PAYLOAD, cfg) {
         cfg.bound_min_x = r->minx >> MALI_TILE_SHIFT;
         cfg.bound_min_y = r->miny >> MALI_TILE_SHIFT;
         cfg.bound_max_x = r->maxx >> MALI_TILE_SHIFT;
         cfg.bound_max_y = r->maxy >> MALI_TILE_SHIFT;
         cfg.framebuffer = fbds + (uint64_t)l * fbd_size;
      }
      struct mali_ptr job =
         mali_jm_cmd_add_job(cmd, c, MALI_JOB_TYPE_FRAGMENT, sizeof(tmpl), false, 0);
      if (!job.cpu) {
         if (unlikely(r->measure_frag))
            mali_jm_measure_end(cmd, r->measure_frag, c);
         return;
      }
      memcpy((uint8_t *)job.cpu + 32, tmpl + 32, sizeof(tmpl) - 32);
   }
   if (unlikely(r->measure_frag))
      mali_jm_measure_end(cmd, r->measure_frag, c);
}

void
MALI_PER_ARCH(cmd_render_end)(struct mali_cmd_buffer *cmd)
{
   struct mali_render_state *r = &cmd->gfx.render;
   if (!r->active)
      return;

   if (MALI_PER_ARCH(fb_pass_has_work)(r) && !vk_command_buffer_has_error(&cmd->vk)) {
      bool ok = mali_fb_alloc_tsd(cmd, r);
      uint32_t fbd_size = 0;
      const uint64_t fbds = ok ? MALI_PER_ARCH(fb_build)(cmd, r, &fbd_size) : 0;
      if (!fbds) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      } else {
         /* No CRC state on the job manager (see the file comment). */
         assert(r->crc_rt < 0);
         /* Each layer's descriptors are 64-byte aligned, so the tag of the
          * first pointer holds for all. */
         assert(fbd_size % 64 == 0);
         MALI_PER_ARCH(fb_fill_tsd)(cmd, r);
         if (unlikely(r->measure_vt))
            mali_jm_measure_end(cmd, r->measure_vt, &cmd->jm.cur.vtc);
         fragment_jobs(cmd, r, fbds, fbd_size);
         cmd->jm.cur.passes++;
      }
   }

   if (unlikely(cmd->jm.measure)) {
      const uint32_t w = r->desc.width, h = r->desc.height;
      mali_jm_measure_info(cmd, r->measure_vt, w, h, r->desc.rt_count, r->draws);
      mali_jm_measure_info(cmd, r->measure_frag, w, h, r->desc.rt_count, r->draws);
   }

   r->active = false;
   cmd->gfx.draw.dirty = MALI_GFX_DIRTY_ALL;
}

/* CRCs (transaction elimination) are off on the job manager until the
 * device shows they pay; with no CRC state there is nothing to
 * invalidate. */
void
MALI_PER_ARCH(cmd_crc_invalidate)(struct mali_cmd_buffer *cmd, const struct mali_image *image)
{
}
