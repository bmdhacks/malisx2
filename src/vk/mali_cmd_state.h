/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Builders shared by the two back halves of the command buffer: the
 * command-stream one (CSF, v11: mali_cmd_draw.c, mali_cmd_dispatch.c,
 * mali_cmd_render.c) and the job-chain one (JM, v9). They pack the
 * descriptors and memory blocks a draw or a dispatch points the GPU at
 * (FAU blocks, resource tables, Blend descriptors, Local Storage
 * descriptors) and keep the draw state's dirty bits. How the result reaches
 * the GPU is the back half's business: register moves into the command
 * stream on CSF, words of a job descriptor on JM.
 *
 * Everything here is static inline and compiled into the back half's draw
 * and dispatch files, because a draw runs these builders every time
 * (the per-draw CPU cost was tuned with them inlined). The per-pass and
 * per-bind parts, which are not hot, are functions in mali_fb.c and
 * mali_cmd_state.c.
 *
 * What the builders need from the back half's struct mali_cmd_buffer (CSF:
 * mali_cmd_buffer.h; JM: mali_jm.h), by member name:
 *
 *  - dev, vk (the runtime's command buffer and its dynamic graphics state),
 *    push_constants;
 *  - gfx.desc (struct mali_desc_state), gfx.draw (struct
 *    mali_gfx_draw_state: everything but regs/regs_valid, which are the
 *    CSF register record), gfx.render (struct mali_render_state);
 *  - slabs and tls (mali_cmd_tls_buffer);
 *  - mali_cmd_alloc(): CPU-mapped, uncached GPU memory that lives until the
 *    command buffer is reset; {0} with the command buffer's error set on
 *    failure;
 *  - mali_cmd_render_tiler(): sets up the pass's tiler side before its
 *    first tiled job and marks MALI_GFX_DIRTY_PASS.
 *
 * Two pointer conventions:
 *  - an FAU word is the block's GPU address with its 64-bit word count in
 *    bits 56..63 (the v11 FAU register; v9 has separate pointer and count
 *    fields in the Shader Environment);
 *  - a resource table pointer is the table's address (64-byte aligned)
 *    with its entry count in the low bits, the form both archs' Resources
 *    fields take.
 */

#ifndef MALI_CMD_STATE_H
#define MALI_CMD_STATE_H

#ifndef PAN_ARCH
#define PAN_ARCH MALI_PAN_ARCH
#endif

#if PAN_ARCH >= 10
#include "mali_cmd_buffer.h"
#else
#include "mali_jm.h"
#endif

#include <string.h>

#include "util/bitscan.h"
#include "util/format/u_format.h"
#include "util/log.h"
#include "util/u_math.h"
#include "vk_format.h"

#include "pan_format.h"

#include "mali_blend.h"
#include "mali_descriptor_set.h"
#include "mali_image.h"
#include "mali_pipeline.h"
#include "mali_vk.h"

/* ---------------------------------------------------------------------- */
/* Compute                                                                 */

/* GPU facts the dispatch sizing needs (panthor's names, kbase's values). */
struct mali_thread_props {
   unsigned max_threads_per_wg;
   unsigned max_threads_per_core;
   unsigned max_tasks_per_core;
   unsigned registers_per_core;
   unsigned core_count;
   unsigned core_id_range;
};

static inline struct mali_thread_props
mali_thread_props(const struct mali_device *dev)
{
   const struct mali_kbase_gpu_props *p = &dev->kbase->props;
   struct mali_thread_props t = {
      .max_threads_per_wg = p->thread_max_workgroup_size ? p->thread_max_workgroup_size
                                                         : p->max_workgroup_size,
      .max_threads_per_core = p->thread_max_threads ? p->thread_max_threads : p->max_threads,
      .max_tasks_per_core = p->thread_features >> 24,
#if PAN_ARCH >= 10
      .registers_per_core = p->thread_features & 0x3fffff,
#else
      /* Before v10, THREAD_FEATURES has the register count in bits 0-15
       * only; bits 16-23 are the task queue depth. */
      .registers_per_core = p->thread_features & 0xffff,
#endif
      .core_count = util_bitcount64(p->shader_present),
      .core_id_range = util_last_bit64(p->shader_present),
   };
   /* Fallbacks for a property stream without the THREAD_* registers (the
    * host tests' fake): the G615's values. */
   if (!t.max_threads_per_wg)
      t.max_threads_per_wg = 1024;
   if (!t.max_threads_per_core)
      t.max_threads_per_core = 2048;
   if (!t.max_tasks_per_core)
      t.max_tasks_per_core = 4;
   if (!t.registers_per_core)
      t.registers_per_core = 65536;
   if (!t.core_count)
      t.core_count = t.core_id_range = 1;
   return t;
}

#if PAN_ARCH >= 9
/* panvk calculate_task_axis_and_increment: spread workgroups evenly over
 * the cores, with tasks small enough that other clients can share them. */
static inline void
mali_compute_task_axis(const struct mali_thread_props *tp, const struct mali_shader *cs,
                       const uint32_t groups[3], unsigned *axis, unsigned *inc)
{
   const unsigned regs = cs->info.work_reg_count <= 32 ? 32 : 64;
   const unsigned max_threads =
      MIN3(tp->max_threads_per_wg, tp->max_threads_per_core, tp->registers_per_core / regs);
   const unsigned threads_per_wg =
      cs->cs.local_size[0] * cs->cs.local_size[1] * cs->cs.local_size[2];
   const uint64_t total = (uint64_t)groups[0] * groups[1] * groups[2];

   *axis = MALI_TASK_AXIS_X;
   *inc = 1;
   if (!total || !threads_per_wg)
      return;

   const unsigned wgs_per_core = (unsigned)MIN2(DIV_ROUND_UP(total, tp->core_count), UINT32_MAX);
   const unsigned threads_per_task = DIV_ROUND_UP(max_threads, tp->max_tasks_per_core);
   unsigned wgs_per_task = CLAMP(threads_per_task / threads_per_wg, 1, wgs_per_core);

   *inc = wgs_per_task;
   for (unsigned i = 0; i < 2; i++) {
      if (*inc <= groups[i])
         break;
      (*axis)++;
      *inc /= groups[i];
   }
   *inc = MAX2(*inc, 1);
}
#endif

/*
 * The FAU block of a dispatch: the used words of struct
 * mali_compute_sysvals, then the used push-constant words, then the
 * constants the compiler promoted (pan_fau_foreach_imm). Returns the FAU
 * word (address | count << 56), 0 when the shader has no FAU or on
 * failure.
 *
 * On v7 the system values also hold the shader's constant data and, from
 * t, the stage's descriptor set addresses (mali_v7_desc_sysvals); t is
 * NULL for an internal shader without descriptor sets. A v7 job takes the
 * address alone.
 */
static inline uint64_t
mali_compute_fau(struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
#if PAN_ARCH < 9
                 const struct mali_v7_desc_tables *t,
#endif
                 const void *push, uint32_t push_size, const uint32_t base[3],
                 const uint32_t groups[3])
{
   if (!cs->fau.total_count)
      return 0;

   struct mali_ptr p = mali_cmd_alloc(cmd, cs->fau.total_count * 8, 16);
   if (!p.cpu)
      return 0;

   struct mali_compute_sysvals sv;
   memset(&sv, 0, sizeof(sv));
   sv.base.x = base[0];
   sv.base.y = base[1];
   sv.base.z = base[2];
   sv.num_work_groups.x = groups[0];
   sv.num_work_groups.y = groups[1];
   sv.num_work_groups.z = groups[2];
   sv.local_group_size.x = cs->cs.local_size[0];
   sv.local_group_size.y = cs->cs.local_size[1];
   sv.local_group_size.z = cs->cs.local_size[2];
   sv.common.push_uniforms = p.gpu;
#if PAN_ARCH < 9
   sv.common.constant_data = cs->v7.data_mem.gpu_va;
   if (t)
      mali_v7_desc_sysvals(t, cs, sv.v7_desc.sets);
#endif

   uint64_t pc[MALI_MAX_PUSH_CONST_FAUS];
   memset(pc, 0, sizeof(pc));
   memcpy(pc, push, MIN2(push_size, sizeof(pc)));

   uint64_t *out = p.cpu;
   uint64_t tmp[MALI_FAU_WORD_COUNT];
   unsigned n = 0, w;
   BITSET_FOREACH_SET(w, cs->fau.used_sysvals, MALI_MAX_SYSVAL_FAUS) {
      tmp[n++] = mali_sysval_word(&sv, sizeof(sv), w);
   }
   BITSET_FOREACH_SET(w, cs->fau.used_push_consts, MALI_MAX_PUSH_CONST_FAUS)
      tmp[n++] = pc[w];
   n += mali_shader_fau_consts(&cs->fau, &tmp[n]);
   for (unsigned i = n; i < cs->fau.total_count; i++)
      tmp[i] = 0;

   /* Constants the backend promoted into the FAU, in 32-bit halves. */
   pan_fau_foreach_imm(&cs->info.fau, i) {
      bool hi = i & 1;
      unsigned idx = i / 2;
      assert(idx < cs->fau.total_count);
      tmp[idx] = (tmp[idx] & ((uint64_t)UINT32_MAX << (32 * !hi))) |
                 ((uint64_t)cs->info.fau.words[i].constant << (32 * hi));
   }

   /* One pass of 64-bit stores into uncached memory. */
   for (unsigned i = 0; i < cs->fau.total_count; i++)
      out[i] = tmp[i];
   return p.gpu | ((uint64_t)cs->fau.total_count << 56);
}

#if PAN_ARCH >= 9
/*
 * Resource table of a dispatch: entry 0 the driver set (the dummy sampler,
 * the dynamic buffers in the shader's dyn_bufs order with their offsets,
 * then `textures` for internal shaders, which have no dynamic buffers),
 * entry N + 1 descriptor set N. desc may be NULL. 0 on failure.
 */
static inline uint64_t
mali_compute_res_table(struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
                       const struct mali_desc_state *desc, const uint32_t (*textures)[8],
                       unsigned texture_count)
{
   const struct mali_shader_desc_info *di = &cs->desc;

   uint32_t drv_count = 1 + di->dyn_bufs.count + texture_count;
   struct mali_ptr drv = mali_cmd_alloc(cmd, drv_count * MALI_DESCRIPTOR_SIZE,
                                        MALI_DESCRIPTOR_SIZE);
   if (!drv.cpu)
      return 0;
   MALI_PER_ARCH(pack_dummy_sampler)(drv.cpu);
   for (uint32_t i = 0; i < di->dyn_bufs.count; i++) {
      uint32_t h = di->dyn_bufs.map[i];
      uint32_t set = MALI_COPY_DESC_HANDLE_SET(h);
      uint32_t idx = MALI_COPY_DESC_HANDLE_INDEX(h);
      const struct mali_descriptor_set *s = desc ? desc->sets[set] : NULL;
      uint8_t *out = (uint8_t *)drv.cpu + (1 + i) * MALI_DESCRIPTOR_SIZE;
      if (s)
         MALI_PER_ARCH(descriptor_set_pack_dyn_buf)(s, idx, desc->dyn_offsets[set][idx], out);
      else
         memset(out, 0, MALI_DESCRIPTOR_SIZE);
   }
   for (unsigned i = 0; i < texture_count; i++)
      memcpy((uint8_t *)drv.cpu + (1 + di->dyn_bufs.count + i) * MALI_DESCRIPTOR_SIZE,
             textures[i], MALI_DESCRIPTOR_SIZE);

   uint32_t last = util_last_bit(di->used_set_mask);
   uint32_t count = ALIGN_POT(1 + last, MALI_RESOURCE_TABLE_SIZE_ALIGNMENT);
   struct mali_ptr t = mali_cmd_alloc(cmd, count * MALI_RESOURCE_SIZE, 64);
   if (!t.cpu)
      return 0;

   pan_cast_and_pack(t.cpu, RESOURCE, cfg) {
      cfg.address = drv.gpu;
      cfg.size = drv_count * MALI_DESCRIPTOR_SIZE;
      cfg.contains_descriptors = true;
   }
   for (uint32_t i = 1; i < count; i++) {
      uint32_t set = i - 1;
      const struct mali_descriptor_set *s =
         (set < MALI_MAX_SETS && (di->used_set_mask & BITFIELD_BIT(set)) && desc) ?
            desc->sets[set] : NULL;
      MALI_PER_ARCH(descriptor_set_pack_resource)(s, (uint8_t *)t.cpu + i * MALI_RESOURCE_SIZE);
   }
   return t.gpu | count;
}
#endif

/* Local Storage descriptor of a dispatch (pan_emit_tls): the command
 * buffer's TLS buffer and, for shaders with shared memory, a per-dispatch
 * WLS buffer. 0 on failure. */
static inline uint64_t
mali_compute_tsd(struct mali_cmd_buffer *cmd, const struct mali_shader *cs,
                 const struct mali_thread_props *tp, const uint32_t groups[3])
{
   struct mali_device *dev = cmd->dev;
   uint64_t tls_ptr = 0, wls_ptr = 0;
   unsigned wls_instances = 0, wls_size = 0;

   if (cs->info.tls_size) {
      tls_ptr = MALI_PER_ARCH(cmd_tls_buffer)(cmd, cs->info.tls_size);
      if (!tls_ptr)
         return 0;
   }

   if (cs->info.wls_size) {
      /* pan_calc_wls_instances / pan_calc_total_wls_size. */
      unsigned threads_per_wg =
         cs->cs.local_size[0] * cs->cs.local_size[1] * cs->cs.local_size[2];
      unsigned wg_per_task =
         DIV_ROUND_UP(tp->max_threads_per_core / tp->max_tasks_per_core, threads_per_wg);
      unsigned max_per_core = util_next_power_of_two(wg_per_task * tp->max_tasks_per_core);
      unsigned dispatch = util_next_power_of_two(groups[0]) *
                          util_next_power_of_two(groups[1]) *
                          util_next_power_of_two(groups[2]);
      wls_instances = MIN2(dispatch, max_per_core);
      wls_size = util_next_power_of_two(MAX2(cs->info.wls_size, 128));
      uint64_t total = (uint64_t)wls_size * wls_instances * tp->core_id_range;

      /* 4 KiB aligned and inside one 4 GiB window (the descriptor's
       * requirement): its own allocation, kept in one 4 GiB page. */
      struct mali_cmd_slab *s = calloc(1, sizeof(*s));
      const struct mali_kbase_alloc_info ai = {
         .size = total,
         .flags = mali_kbase_mem_same_va_policy(MALI_KBASE_FLAGS_TLS |
                                                KB_MEM_GPU_VA_SAME_4GB_PAGE),
         .mem_class = MALI_KBASE_MEM_CLASS_INTERNAL_TLS,
      };
      if (!s || mali_kbase_alloc(dev->kbase, &ai, &s->bo) != MALI_KBASE_SUCCESS) {
         free(s);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return 0;
      }
      list_addtail(&s->link, &cmd->slabs);
      wls_ptr = s->bo.gpu_va;
   }

   struct mali_ptr p = mali_cmd_alloc(cmd, pan_size(LOCAL_STORAGE), 64);
   if (!p.cpu)
      return 0;
   pan_cast_and_pack(p.cpu, LOCAL_STORAGE, cfg) {
      if (tls_ptr) {
         cfg.tls_size = util_logbase2_ceil(DIV_ROUND_UP(cs->info.tls_size, 16));
#if PAN_ARCH >= 9
         cfg.tls_address_mode = MALI_ADDRESS_MODE_PACKED;
         cfg.tls_base_pointer = tls_ptr >> 8;
#else
         /* v7 has no address mode and takes the full pointer (pan_emit_tls). */
         cfg.tls_base_pointer = tls_ptr;
#endif
      }
      if (wls_ptr) {
         cfg.wls_base_pointer = wls_ptr;
         cfg.wls_instances = wls_instances;
         cfg.wls_size_scale = util_logbase2(wls_size) + 1;
      } else {
         cfg.wls_instances = MALI_LOCAL_STORAGE_NO_WORKGROUP_MEM;
      }
   }
   return p.gpu;
}

/* ---------------------------------------------------------------------- */
/* Graphics: FAU (the block itself is mali_cmd_gfx_fau, in mali_cmd_state.c) */

/* The sysvals every graphics stage of the current draw sees. */
static inline void
mali_gfx_fill_sysvals(struct mali_cmd_buffer *cmd, struct mali_graphics_sysvals *sv)
{
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   const struct mali_graphics_pipeline *p = d->pipeline;

   memset(sv, 0, sizeof(*sv));
   for (unsigned i = 0; i < 4; i++)
      sv->blend.constants[i] = dyn->cb.blend_constants[i];

   /* Viewport transform (the vertex shader applies it): scale (w/2, h/2,
    * zmax - zmin), offset (x + w/2, y + h/2, zmin). */
   const VkViewport *vp = &dyn->vp.viewports[0];
   sv->viewport[0].scale = 0.5f * vp->width;
   sv->viewport[1].scale = 0.5f * vp->height;
   sv->viewport[2].scale = vp->maxDepth - vp->minDepth;
   sv->viewport[0].offset = 0.5f * vp->width + vp->x;
   sv->viewport[1].offset = 0.5f * vp->height + vp->y;
   sv->viewport[2].offset = vp->minDepth;

   sv->vs.first_vertex = d->first_vertex;
   sv->vs.base_instance = d->base_instance;
   sv->vs.noperspective_varyings = p->bind.noperspective;
#if PAN_ARCH >= 9
   /* On v7 these words are v7_desc.sets, which the v7 draw fills: BLEND
    * reads the Blend descriptors from the blend FAU slots (mali_shader.h). */
   memcpy(sv->fs.blend_descs, d->blend_descs, sizeof(sv->fs.blend_descs));
#endif

   /* Input attachments: colour targets through the tile buffer with their
    * conversion, depth/stencil as the ZS target (panvk
    * prepare_iam_sysvals). */
   memset(sv->iam, 0xff, sizeof(sv->iam));
   const struct vk_input_attachment_location_state *ial = &dyn->ial;
   const struct mali_render_state *r = &cmd->gfx.render;
   const uint32_t catt = ial->color_attachment_count == MESA_VK_COLOR_ATTACHMENT_COUNT_UNKNOWN ?
                            MALI_MAX_RTS : ial->color_attachment_count;
   for (uint32_t i = 0; i < catt && i < MALI_MAX_RTS; i++) {
      if (ial->color_map[i] == MESA_VK_ATTACHMENT_UNUSED || i >= r->desc.rt_count ||
          !r->desc.rt[i].image)
         continue;
      const uint32_t idx = ial->color_map[i] + 1;
      if (idx >= MALI_INPUT_ATTACHMENT_MAP_SIZE)
         continue;
      struct mali_internal_conversion_packed conv;
      pan_pack(&conv, INTERNAL_CONVERSION, cfg) {
         cfg.memory_format =
            GENX(pan_dithered_format_from_pipe_format)(r->desc.rt[i].format, false);
#if PAN_ARCH < 9
         /* The v7 tile load converts into the register format the
          * conversion names (panvk prepare_iam_sysvals). */
         const enum pipe_format f = r->desc.rt[i].format;
         cfg.register_format = util_format_is_pure_uint(f)   ? MALI_REGISTER_FILE_FORMAT_U32
                               : util_format_is_pure_sint(f) ? MALI_REGISTER_FILE_FORMAT_I32
                                                             : MALI_REGISTER_FILE_FORMAT_F32;
#endif
      }
      sv->iam[idx].target = MALI_COLOR_ATTACHMENT(i);
      sv->iam[idx].conversion = conv.opaque[0];
   }
   if (ial->depth_att != MESA_VK_ATTACHMENT_UNUSED) {
      uint32_t idx = ial->depth_att == MESA_VK_ATTACHMENT_NO_INDEX ? 0 : ial->depth_att + 1;
      if (idx < MALI_INPUT_ATTACHMENT_MAP_SIZE) {
         sv->iam[idx].target = MALI_ZS_ATTACHMENT;
         sv->iam[idx].conversion = 0;
#if PAN_ARCH < 9
         /* v7 needs the depth format in the conversion; a zero one
          * faults with INVALID_INSTR_ENC (panvk prepare_iam_sysvals). */
         if (r->desc.z.image) {
            struct mali_internal_conversion_packed zconv;
            pan_pack(&zconv, INTERNAL_CONVERSION, cfg) {
               cfg.register_format = MALI_REGISTER_FILE_FORMAT_F32;
               cfg.memory_format =
                  GENX(pan_dithered_format_from_pipe_format)(r->desc.z.format, false);
            }
            sv->iam[idx].conversion = zconv.opaque[0];
         }
#endif
      }
   }
   if (ial->stencil_att != MESA_VK_ATTACHMENT_UNUSED) {
      uint32_t idx = ial->stencil_att == MESA_VK_ATTACHMENT_NO_INDEX ? 0 : ial->stencil_att + 1;
      if (idx < MALI_INPUT_ATTACHMENT_MAP_SIZE) {
#if PAN_ARCH < 9
         /* A v7 stencil read uses a constant conversion (mali_shader.c
          * v7_stencil_tile_conv): leave depth's when both share the index. */
         if (sv->iam[idx].target != MALI_ZS_ATTACHMENT)
            sv->iam[idx].conversion = 0;
         sv->iam[idx].target = MALI_ZS_ATTACHMENT;
#else
         sv->iam[idx].target = MALI_ZS_ATTACHMENT;
         sv->iam[idx].conversion = 0;
#endif
      }
   }
}

/* ---------------------------------------------------------------------- */
/* Graphics: resource tables                                               */

#if PAN_ARCH >= 9

static inline void
mali_gfx_pack_dyn_bufs(const struct mali_shader_desc_info *di, const struct mali_desc_state *desc,
                       uint8_t *out)
{
   for (uint32_t i = 0; i < di->dyn_bufs.count; i++) {
      uint32_t h = di->dyn_bufs.map[i];
      uint32_t set = MALI_COPY_DESC_HANDLE_SET(h);
      uint32_t idx = MALI_COPY_DESC_HANDLE_INDEX(h);
      const struct mali_descriptor_set *s = desc->sets[set];
      uint8_t *o = out + i * MALI_DESCRIPTOR_SIZE;
      if (s)
         MALI_PER_ARCH(descriptor_set_pack_dyn_buf)(s, idx, desc->dyn_offsets[set][idx], o);
      else
         memset(o, 0, MALI_DESCRIPTOR_SIZE);
   }
}

/* Resource table: entry 0 the driver set (already built), entry N + 1
 * descriptor set N. */
static inline uint64_t
mali_gfx_res_table(struct mali_cmd_buffer *cmd, const struct mali_shader *s,
                   const struct mali_desc_state *desc, uint64_t drv, uint32_t drv_count)
{
   const struct mali_shader_desc_info *di = &s->desc;
   uint32_t last = util_last_bit(di->used_set_mask);
   uint32_t count = ALIGN_POT(1 + last, MALI_RESOURCE_TABLE_SIZE_ALIGNMENT);
   struct mali_ptr t = mali_cmd_alloc(cmd, count * MALI_RESOURCE_SIZE, 64);
   if (!t.cpu)
      return 0;

   pan_cast_and_pack(t.cpu, RESOURCE, cfg) {
      cfg.address = drv;
      cfg.size = drv_count * MALI_DESCRIPTOR_SIZE;
      cfg.contains_descriptors = true;
   }
   for (uint32_t i = 1; i < count; i++) {
      uint32_t set = i - 1;
      const struct mali_descriptor_set *ds =
         (set < MALI_MAX_SETS && (di->used_set_mask & BITFIELD_BIT(set))) ? desc->sets[set]
                                                                          : NULL;
      MALI_PER_ARCH(descriptor_set_pack_resource)(ds, (uint8_t *)t.cpu + i * MALI_RESOURCE_SIZE);
   }
   return t.gpu | count;
}

/* One vertex attribute (panvk emit_vs_attrib). */
static inline void
mali_gfx_pack_attrib(const struct vk_vertex_input_state *vi, const uint16_t *strides, uint32_t i,
                     uint32_t vb_offset, uint32_t base_instance, void *out)
{
   const struct vk_vertex_attribute_state *a = &vi->attributes[i];
   const struct vk_vertex_binding_state *bs = &vi->bindings[a->binding];
   const uint32_t stride = strides[a->binding];
   const bool per_instance = bs->input_rate == VK_VERTEX_INPUT_RATE_INSTANCE;
   const enum pipe_format f = vk_format_to_pipe_format(a->format);

   pan_cast_and_pack(out, ATTRIBUTE, cfg) {
      cfg.offset = a->offset + (per_instance ? base_instance * stride : 0);
      cfg.format = GENX(pan_format_from_pipe_format)(f)->hw;
      cfg.table = 0;
      cfg.buffer_index = vb_offset + a->binding;
      cfg.stride = stride;
      if (!per_instance) {
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_VERTEX;
         cfg.offset_enable = true;
      } else if (bs->divisor == 1) {
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_INSTANCE;
      } else if (bs->divisor == 0) {
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_INSTANCE;
         cfg.stride = 0;
      } else if (util_is_power_of_two_or_zero(bs->divisor)) {
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_1D_POT_DIVISOR;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_INSTANCE;
         cfg.divisor_r = __builtin_ctz(bs->divisor);
      } else {
         /* Not reachable for ARMSX2 (no divisors); the NPOT magic numbers
          * are not implemented. */
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_INSTANCE;
      }
   }
}

/* Vertex stage resource table. Table 0: 16 attributes, the dummy sampler,
 * the dynamic buffers, then one Buffer per vertex binding (panvk
 * prepare_vs_driver_set). 0 on failure. */
static inline uint64_t
mali_gfx_vs_srt(struct mali_cmd_buffer *cmd)
{
   const struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   const struct mali_shader *vs = d->pipeline->vs;
   const struct vk_vertex_input_state *vi = d->vi;

   uint32_t vb_count = 0;
   u_foreach_bit(i, vi->attributes_valid)
      vb_count = MAX2(vi->attributes[i].binding + 1, vb_count);
   const uint32_t vb_offset = MALI_MAX_VS_ATTRIBS + 1 + vs->desc.dyn_bufs.count;
   const uint32_t count = vb_offset + vb_count;

   struct mali_ptr p = mali_cmd_alloc(cmd, count * MALI_DESCRIPTOR_SIZE, MALI_DESCRIPTOR_SIZE);
   if (!p.cpu)
      return 0;
   uint8_t *descs = p.cpu;

   for (uint32_t i = 0; i < MALI_MAX_VS_ATTRIBS; i++) {
      void *o = descs + i * MALI_DESCRIPTOR_SIZE;
      if (vi->attributes_valid & BITFIELD_BIT(i)) {
         mali_gfx_pack_attrib(vi, d->vi_strides, i, vb_offset, d->base_instance, o);
      } else {
         /* An invalid table makes reads out of bounds (panvk). */
         pan_cast_and_pack(o, ATTRIBUTE, cfg) {
            cfg.table = 17;
            cfg.format = (MALI_R16F << 12) | MALI_RGB_COMPONENT_ORDER_RGBA;
         }
      }
   }
   MALI_PER_ARCH(pack_dummy_sampler)(descs + MALI_MAX_VS_ATTRIBS * MALI_DESCRIPTOR_SIZE);
   mali_gfx_pack_dyn_bufs(&vs->desc, &cmd->gfx.desc,
                          descs + (MALI_MAX_VS_ATTRIBS + 1) * MALI_DESCRIPTOR_SIZE);
   for (uint32_t i = 0; i < vb_count; i++) {
      void *o = descs + (vb_offset + i) * MALI_DESCRIPTOR_SIZE;
      if ((vi->bindings_valid & BITFIELD_BIT(i)) && d->vb[i].addr) {
         pan_cast_and_pack(o, BUFFER, cfg) {
            cfg.address = d->vb[i].addr;
            cfg.size = (uint32_t)MIN2(d->vb[i].size, UINT32_MAX);
         }
      } else {
         pan_cast_and_pack(o, NULL_DESCRIPTOR, cfg);
      }
   }
   return mali_gfx_res_table(cmd, vs, &cmd->gfx.desc, p.gpu, count);
}

/* Fragment varyings read with LD_VAR (panvk emit_varying_descs). */
static inline void
mali_gfx_pack_varying_descs(const struct mali_shader *vs, const struct mali_shader *fs,
                            uint8_t *out)
{
   const struct pan_varying_layout *vl = &vs->info.varyings.formats;
   const struct pan_varying_layout *fl = &fs->info.varyings.formats;

   for (uint32_t i = 0; i < fl->count && i < fs->desc.driver_table_prefix; i++) {
      const struct pan_varying_slot *fslot = pan_varying_layout_slot_at(fl, i);
      if (!fslot || fslot->section != PAN_VARYING_SECTION_GENERIC)
         continue;
      unsigned offset = 0;
      enum pipe_format format = PIPE_FORMAT_NONE;
      const struct pan_varying_slot *vslot = pan_varying_layout_find_slot(vl, fslot->location);
      if (vslot) {
         nir_alu_type base = nir_alu_type_get_base_type(fslot->alu_type);
         nir_alu_type bits = nir_alu_type_get_type_size(vslot->alu_type);
         offset = vslot->offset;
         format = pan_varying_format(base | bits, vslot->ncomps);
      }
      pan_cast_and_pack(out + i * MALI_DESCRIPTOR_SIZE, ATTRIBUTE, cfg) {
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_VERTEX_PACKET;
         cfg.offset_enable = false;
         cfg.format = GENX(pan_format_from_pipe_format)(format)->hw;
         cfg.table = 61;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_VERTEX;
         cfg.offset = 1024 + offset;
         cfg.buffer_index = 0;
         cfg.attribute_stride = vl->generic_size_B;
         cfg.packet_stride = vl->generic_size_B + 16;
      }
   }
}

/* Fragment stage resource table. Table 0: the varying descriptors, the
 * dummy sampler, the dynamic buffers. 0 on failure. */
static inline uint64_t
mali_gfx_fs_srt(struct mali_cmd_buffer *cmd)
{
   const struct mali_graphics_pipeline *p = cmd->gfx.draw.pipeline;
   const struct mali_shader *fs = p->fs;
   const uint32_t prefix = fs->desc.driver_table_prefix;
   const uint32_t count = prefix + 1 + fs->desc.dyn_bufs.count;

   struct mali_ptr t = mali_cmd_alloc(cmd, count * MALI_DESCRIPTOR_SIZE, MALI_DESCRIPTOR_SIZE);
   if (!t.cpu)
      return 0;
   uint8_t *descs = t.cpu;
   memset(descs, 0, prefix * MALI_DESCRIPTOR_SIZE);
   if (fs->desc.needs_varying_descs)
      mali_gfx_pack_varying_descs(p->vs, fs, descs);
   MALI_PER_ARCH(pack_dummy_sampler)(descs + prefix * MALI_DESCRIPTOR_SIZE);
   mali_gfx_pack_dyn_bufs(&fs->desc, &cmd->gfx.desc, descs + (prefix + 1) * MALI_DESCRIPTOR_SIZE);
   return mali_gfx_res_table(cmd, fs, &cmd->gfx.desc, t.gpu, count);
}

#else /* PAN_ARCH < 9 */

/* v7 has no resource tables: the v7 draw builds its attribute and varying
 * descriptors itself (mali_jm_cmd_draw_v7.c), and each stage's
 * descriptors go through per-type tables, built the same way for draws
 * and dispatches (mali_jm_desc_v7.c). */

#endif /* PAN_ARCH < 9 */

/* ---------------------------------------------------------------------- */
/* Graphics: blend                                                         */

/* The blend constant for fixed-function blending: UNORM of the target's
 * channel size, in the top bits of 16 (pan_pack_blend_constant). Blend
 * factors of UNORM targets are clamped to [0, 1] (Vulkan, "Blend
 * Factors"); ARMSX2 passes AFIX / 128, up to 1.99. */
static inline uint16_t
mali_gfx_pack_blend_constant(enum pipe_format format, float c)
{
   const struct util_format_description *desc = util_format_description(format);
   unsigned size = 0;
   for (unsigned i = 0; i < desc->nr_channels; i++)
      size = MAX2(desc->channel[0].size, size);
   if (!size || size > 16)
      return 0;
   float factor = (float)(((1u << size) - 1) << (16 - size));
   return (uint16_t)(CLAMP(c, 0.0f, 1.0f) * factor);
}

/* The blend constant channels `mask` reads are all the same value
 * (fixed-function blending has one constant). */
static inline bool
mali_gfx_homogeneous_constant(unsigned mask, const float *consts)
{
   const float c = consts[ffs(mask) - 1];
   u_foreach_bit(i, mask) {
      if (consts[i] != c)
         return false;
   }
   return true;
}

/* The address of the blend shader for render target `i`, 0 if there is
 * none (compile failure, logged). */
static inline uint64_t
mali_gfx_blend_shader_for(struct mali_cmd_buffer *cmd, const struct mali_graphics_pipeline *p,
                          unsigned i)
{
   const struct mali_render_state *r = &cmd->gfx.render;
   const enum pipe_format format = r->desc.rt[i].format;
   const unsigned samples = r->desc.rt[i].image->vk.samples;

   uint64_t addr = 0;
   if (p->blend_shader[i].addr && p->blend_shader[i].format == format &&
       p->blend_shader[i].samples == samples) {
      /* Compiled with the pipeline for this target's format and samples. */
      addr = p->blend_shader[i].addr;
   } else {
      struct mali_blend_shader_key key;
      MALI_PER_ARCH(blend_shader_key_init)(&key, &p->baked, i, format, samples, &p->fs->info);
      if (MALI_PER_ARCH(blend_shader_get)(cmd->dev, &key, &addr) != VK_SUCCESS) {
         static bool warned;
         if (!warned) {
            warned = true;
            mesa_logw("malisx2: a blend shader failed to compile; the colour is stored "
                      "unblended");
         }
         return 0;
      }
   }
   /* The descriptor holds 32 bits of the address; the fragment shader's
    * jump supplies the rest. */
   assert(addr >> 32 == p->fs->code.gpu_va >> 32);
   assert(!(addr & 15));
   return addr;
}

/*
 * Which of the first `count` render targets of pipeline p (the bound one)
 * take a blend shader in the current pass, for the Blend descriptor
 * builders (mali_gfx_blend, and on v7 mali_jm_cmd_draw_v7.c). The caller
 * passes shader[] and opaque_fallback[] zeroed: shader[i] becomes the
 * target's blend shader address, and opaque_fallback[i] is set for a
 * target whose blend shader is missing, which then stores the source
 * unblended. Returns the fixed-function blend constant, 0 when no target
 * uses one.
 *
 * A target uses a blend shader when the pipeline says so, or when it uses
 * the blend constant in fixed function and either its channels disagree or
 * the value differs from the one an earlier target already uses: the
 * hardware has a single constant (panvk blend_needs_shader).
 */
static inline uint32_t
mali_gfx_blend_select(struct mali_cmd_buffer *cmd, const struct mali_graphics_pipeline *p,
                      uint32_t count, uint64_t *shader, bool *opaque_fallback)
{
   const struct mali_gfx_baked *bk = &p->baked;
   const struct mali_render_state *r = &cmd->gfx.render;
   const float *consts = cmd->vk.dynamic_graphics_state.cb.blend_constants;

   /* The fixed-function constant, all ones until a target takes one. */
   uint32_t ff_constant = ~0u;
   for (uint32_t i = 0; i < count && i < bk->rt_count; i++) {
      const struct mali_blend_rt_baked *rt = &bk->blend[i];
      const bool have_rt = i < r->desc.rt_count && r->desc.rt[i].image;
      if (!p->fs || rt->mode == MALI_BLEND_RT_OFF || !have_rt)
         continue;

      bool use_shader = rt->mode == MALI_BLEND_RT_SHADER;
      if (rt->mode == MALI_BLEND_RT_FIXED_FUNCTION && rt->constant_mask) {
         const uint32_t c = mali_gfx_pack_blend_constant(r->desc.rt[i].format,
                                                         consts[ffs(rt->constant_mask) - 1]);
         if (!mali_gfx_homogeneous_constant(rt->constant_mask, consts) ||
             (ff_constant != ~0u && c != ff_constant))
            use_shader = true;
         else
            ff_constant = c;
      }
      if (use_shader) {
         shader[i] = mali_gfx_blend_shader_for(cmd, p, i);
         opaque_fallback[i] = !shader[i];
      }
   }
   if (ff_constant == ~0u)
      ff_constant = 0;
   return ff_constant;
}

#if PAN_ARCH >= 9

/*
 * The Blend descriptors of the current pipeline for the current render
 * pass (panvk blend_emit_descs with the equations our pipeline compile
 * baked in), and the internal words the fragment shader's BLEND
 * instructions read (by output location), kept in gfx.draw.blend_descs for
 * the fragment FAU. Returns their address, *count_out descriptors; 0 on
 * failure. The targets that take a blend shader: mali_gfx_blend_select.
 */
static inline uint64_t
mali_gfx_blend(struct mali_cmd_buffer *cmd, uint32_t *count_out)
{
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   const struct mali_graphics_pipeline *p = d->pipeline;
   const struct mali_gfx_baked *bk = &p->baked;
   const struct mali_render_state *r = &cmd->gfx.render;
   const uint32_t count = r->rt_count;

   struct mali_ptr m = mali_cmd_alloc(cmd, count * pan_size(BLEND), 64);
   if (!m.cpu)
      return 0;

   uint64_t shader[MALI_MAX_RTS] = {0};
   bool opaque_fallback[MALI_MAX_RTS] = {false};
   const uint32_t ff_constant = mali_gfx_blend_select(cmd, p, count, shader, opaque_fallback);

   d->blend_shader_mask = 0;
   for (uint32_t i = 0; i < count; i++)
      d->blend_shader_mask |= shader[i] ? BITFIELD_BIT(i) : 0;

   memset(d->blend_descs, 0, sizeof(d->blend_descs));
   struct mali_blend_packed off;
   pan_pack(&off, BLEND, cfg) {
      cfg.enable = false;
      cfg.internal.mode = MALI_BLEND_MODE_OFF;
   }
   for (unsigned loc = 0; loc < MALI_MAX_RTS; loc++)
      d->blend_descs[loc] = off.opaque[2] | (uint64_t)off.opaque[3] << 32;

   /* Packed here and copied out at the end: m is uncached, and reading
    * back from it (the internal words below) stalls for a memory round
    * trip. */
   struct mali_blend_packed descs[MALI_MAX_RTS];
   assert(count <= MALI_MAX_RTS);
   for (uint32_t i = 0; i < count; i++) {
      struct mali_blend_packed *out = &descs[i];
      const struct mali_blend_rt_baked *rt = i < bk->rt_count ? &bk->blend[i] : NULL;
      const bool have_rt = i < r->desc.rt_count && r->desc.rt[i].image;

      if (!p->fs || !rt || rt->mode == MALI_BLEND_RT_OFF || !have_rt) {
         *out = off;
         continue;
      }

      const enum pipe_format fmt = r->desc.rt[i].format;
      if (shader[i]) {
         pan_pack(out, BLEND, cfg) {
            cfg.srgb = util_format_is_srgb(fmt);
            cfg.load_destination = rt->load_destination;
            cfg.round_to_fb_precision = true;
            cfg.blend_constant = ff_constant;
            cfg.internal.mode = MALI_BLEND_MODE_SHADER;
            cfg.internal.shader.pc = (uint32_t)shader[i];
         }
      } else {
         enum mali_blend_rt_mode mode = opaque_fallback[i] ? MALI_BLEND_RT_OPAQUE : rt->mode;
         pan_pack(out, BLEND, cfg) {
            cfg.srgb = util_format_is_srgb(fmt);
            cfg.load_destination = rt->load_destination && mode == MALI_BLEND_RT_FIXED_FUNCTION;
            cfg.round_to_fb_precision = true;
            cfg.blend_constant = ff_constant;
            cfg.internal.mode = mode == MALI_BLEND_RT_OPAQUE ? MALI_BLEND_MODE_OPAQUE
                                                             : MALI_BLEND_MODE_FIXED_FUNCTION;
            cfg.internal.fixed_function.num_comps = 4;
            cfg.internal.fixed_function.conversion.memory_format =
               GENX(pan_dithered_format_from_pipe_format)(fmt, false);
            if (cfg.internal.mode == MALI_BLEND_MODE_FIXED_FUNCTION &&
                (cfg.internal.fixed_function.conversion.memory_format & 0xff) ==
                   MALI_RGB_COMPONENT_ORDER_RGB1) {
               /* Fixed-function blending does not take RGB1 (panvk). */
               cfg.internal.fixed_function.conversion.memory_format &= ~0xff;
               cfg.internal.fixed_function.conversion.memory_format |=
                  MALI_RGB_COMPONENT_ORDER_RGBA;
            }
            cfg.internal.fixed_function.rt = i;
         }
         /* The baked equation word (descriptor word 1); a failed blend
          * shader stores the source, colour mask applied. */
         if (rt->mode == MALI_BLEND_RT_SHADER) {
            struct mali_blend_equation_packed replace;
            pan_pack(&replace, BLEND_EQUATION, cfg) {
               cfg.rgb.a = MALI_BLEND_OPERAND_A_SRC;
               cfg.rgb.b = MALI_BLEND_OPERAND_B_SRC;
               cfg.rgb.c = MALI_BLEND_OPERAND_C_ZERO;
               cfg.alpha.a = MALI_BLEND_OPERAND_A_SRC;
               cfg.alpha.b = MALI_BLEND_OPERAND_B_SRC;
               cfg.alpha.c = MALI_BLEND_OPERAND_C_ZERO;
               cfg.color_mask = rt->eq.color_mask;
            }
            out->opaque[1] = replace.opaque[0];
         } else {
            out->opaque[1] = rt->equation;
         }
      }

      if (rt->shader_location < MALI_MAX_RTS)
         d->blend_descs[rt->shader_location] =
            out->opaque[2] | (uint64_t)out->opaque[3] << 32;
   }

   memcpy(m.cpu, descs, count * sizeof(descs[0]));
   *count_out = count;
   return m.gpu;
}

#endif

/* ---------------------------------------------------------------------- */
/* Graphics: viewport, index type                                          */

/*
 * The scissor box: the viewport rectangle clipped by the scissor (panvk
 * v11 prepare_vp), as inclusive bounds; the hardware clips to the box. The
 * caller clamps them to the 16 bits of the Scissor (v9+) or of the v7
 * Viewport descriptor's scissor.
 */
static inline void
mali_gfx_scissor_rect(const struct mali_cmd_buffer *cmd, int *minx_out, int *miny_out,
                      int *maxx_out, int *maxy_out)
{
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const VkViewport *vp = &dyn->vp.viewports[0];
   const VkRect2D *sc = &dyn->vp.scissors[0];

   int minx = (int)vp->x;
   int maxx = (int)(vp->x + vp->width);
   int miny = (int)MIN2(vp->y, vp->y + vp->height);
   int maxy = (int)MAX2(vp->y, vp->y + vp->height);
   minx = MAX2(sc->offset.x, minx);
   miny = MAX2(sc->offset.y, miny);
   maxx = MIN2(sc->offset.x + (int)sc->extent.width, maxx);
   maxy = MIN2(sc->offset.y + (int)sc->extent.height, maxy);
#if PAN_ARCH < 10
   /* Clamp the box to the render area, as Mesa's v9 driver does, rather
    * than rely on the tiler context's framebuffer size to clip a viewport
    * that extends past it. */
   const struct mali_render_state *r = &cmd->gfx.render;
   minx = MIN2(minx, (int)r->desc.width);
   miny = MIN2(miny, (int)r->desc.height);
   maxx = MIN2(maxx, (int)r->desc.width);
   maxy = MIN2(maxy, (int)r->desc.height);
#endif
   maxx = maxx > minx ? maxx - 1 : maxx;
   maxy = maxy > miny ? maxy - 1 : maxy;
   *minx_out = minx;
   *miny_out = miny;
   *maxx_out = maxx;
   *maxy_out = maxy;
}

#if PAN_ARCH >= 9

/* The packed Scissor (64 bits) of the scissor box (mali_gfx_scissor_rect). */
static inline uint64_t
mali_gfx_scissor_box(const struct mali_cmd_buffer *cmd)
{
   int minx, miny, maxx, maxy;
   mali_gfx_scissor_rect(cmd, &minx, &miny, &maxx, &maxy);

   struct mali_scissor_packed s;
   pan_pack(&s, SCISSOR, cfg) {
      cfg.scissor_minimum_x = CLAMP(minx, 0, UINT16_MAX);
      cfg.scissor_minimum_y = CLAMP(miny, 0, UINT16_MAX);
      cfg.scissor_maximum_x = CLAMP(maxx, 0, UINT16_MAX);
      cfg.scissor_maximum_y = CLAMP(maxy, 0, UINT16_MAX);
   }
   return s.opaque[0] | (uint64_t)s.opaque[1] << 32;
}

#endif

static inline enum mali_index_type
mali_gfx_index_type(uint32_t size)
{
   switch (size) {
   case 1: return MALI_INDEX_TYPE_UINT8;
   case 2: return MALI_INDEX_TYPE_UINT16;
   case 4: return MALI_INDEX_TYPE_UINT32;
   default: return MALI_INDEX_TYPE_NONE;
   }
}

/* ---------------------------------------------------------------------- */
/* Graphics: the draw's state                                              */

struct mali_draw_info {
   uint32_t count;
   uint32_t instance_count;
   uint32_t first_index;       /* indexed only */
   int32_t vertex_offset;      /* firstVertex or vertexOffset */
   uint32_t first_instance;
   bool indexed;
};

#if PAN_ARCH < 9
/* Whether a dirty dynamic state may feed the pipeline bake (gfx_pack_state):
 * any but the viewports, scissors, blend constants and vertex input, which
 * the draw reads through groups of their own. */
static inline bool
mali_v7_dyn_bake_dirty(const struct vk_dynamic_graphics_state *dyn)
{
   BITSET_DECLARE(d, MESA_VK_DYNAMIC_GRAPHICS_STATE_ENUM_MAX);
   memcpy(d, dyn->dirty, sizeof(d));
   BITSET_CLEAR(d, MESA_VK_DYNAMIC_VP_VIEWPORT_COUNT);
   BITSET_CLEAR(d, MESA_VK_DYNAMIC_VP_VIEWPORTS);
   BITSET_CLEAR(d, MESA_VK_DYNAMIC_VP_SCISSOR_COUNT);
   BITSET_CLEAR(d, MESA_VK_DYNAMIC_VP_SCISSORS);
   BITSET_CLEAR(d, MESA_VK_DYNAMIC_CB_BLEND_CONSTANTS);
   BITSET_CLEAR(d, MESA_VK_DYNAMIC_VI);
   BITSET_CLEAR(d, MESA_VK_DYNAMIC_VI_BINDINGS_VALID);
   BITSET_CLEAR(d, MESA_VK_DYNAMIC_VI_BINDING_STRIDES);
   return !BITSET_IS_EMPTY(d);
}
#endif

/* Dynamic states this driver reads at draw time, per dirty group. */
static inline void
mali_gfx_collect_dynamic_dirty(struct mali_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;

   if (BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_VP_VIEWPORTS) ||
       BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_VP_SCISSORS)) {
      d->dirty |= MALI_GFX_DIRTY_VIEWPORT | MALI_GFX_DIRTY_VS_FAU;
   }
   if (BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_CONSTANTS))
      d->dirty |= MALI_GFX_DIRTY_BLEND | MALI_GFX_DIRTY_FS_FAU;
   if (BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_RS_LINE_WIDTH))
      d->dirty |= MALI_GFX_DIRTY_PIPELINE;
#if PAN_ARCH >= 9
   /* v9+: the attributes live in the VS table (v7 packs them per draw). */
   if (BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_VI) ||
       BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_VI_BINDING_STRIDES) ||
       BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_VI_BINDINGS_VALID))
      d->dirty |= MALI_GFX_DIRTY_VS_SRT;
#endif
   if (BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_INPUT_ATTACHMENT_MAP))
      d->dirty |= MALI_GFX_DIRTY_FS_FAU;
   if (d->pipeline && d->pipeline->bind.dynamic) {
      /* Words baked from state the pipeline left dynamic: repack. On v7
       * only when a state the bake reads changed: the runtime marks a
       * state dirty only when its value changes, and a new pipeline, a
       * pass and meta operations mark these groups themselves. */
#if PAN_ARCH < 9
      if (mali_v7_dyn_bake_dirty(dyn))
#endif
         d->dirty |= MALI_GFX_DIRTY_PIPELINE | MALI_GFX_DIRTY_ZSD | MALI_GFX_DIRTY_BLEND |
                     MALI_GFX_DIRTY_FS_FAU;
   }
   vk_dynamic_graphics_state_clear_dirty(dyn);
}

/*
 * A draw's state, in two steps that the back half's draw runs in this
 * order:
 *
 *  1. mali_gfx_update_dirty: the pass's tiler side (the back half's
 *     mali_cmd_render_tiler, on the pass's first draw), then the draw's
 *     parameters and the dynamic state folded into gfx.draw.dirty, and the
 *     pipeline's dynamic words repacked.
 *  2. For each group whose bit is set in gfx.draw.dirty, in the order
 *     blend, the sysvals (when either FAU group is dirty), VS FAU, FS FAU,
 *     VS table, FS table, depth/stencil: its mali_gfx_build_* function,
 *     which packs the group's descriptors and returns the word the back
 *     half stores (blend first: it dirties the FS FAU, which carries the
 *     blend descriptors' internal words). Then the back half writes the
 *     groups that need no building (pipeline words, viewport, pass, index
 *     buffer) and clears the bits.
 *
 * The back half stores each word right after its build: the builders are
 * inlined into its draw, and holding the words until the end costs
 * register spills on every draw.
 */

/* Step 1. *dbk_out: the pipeline's words repacked from dynamic state
 * (gfx.draw.dyn_baked) when the pipeline has any, else NULL (the bind
 * block's words apply). False on allocation failure. */
static inline bool
mali_gfx_update_dirty(struct mali_cmd_buffer *cmd, const struct mali_draw_info *di,
                      const struct mali_gfx_baked **dbk_out)
{
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   struct mali_render_state *r = &cmd->gfx.render;
   struct mali_graphics_pipeline *p = d->pipeline;
   const struct mali_gfx_bind *bd = &p->bind;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   mali_gfx_collect_dynamic_dirty(cmd);

   /* The pass's provoking-vertex mode (provokingVertexModePerPipeline is
    * false): take it from the first app pipeline bound in the pass,
    * before the tiler context or framebuffer descriptors are built. Read
    * the pipeline's own copy, not cmd->vk.dynamic_graphics_state -- a
    * pipeline with no dynamic state (every ARMSX2 pipeline) never copies
    * this into it, since nothing else here reads it from there.
    *
    * Only an app draw reaches this (a full-screen draw -- a clear, the
    * in-pass primitive barrier -- calls cmd_render_tiler directly, never
    * this function; cmd_run_fullscreen's own comment says why it does
    * not need a mode). One of those may have already built the tiler
    * context, with the default, before any app pipeline said otherwise;
    * fix it up now that this is the first time we know. */
   const bool pv_first =
      p->state.rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT;
   if (!r->first_provoking_vertex_set) {
      r->first_provoking_vertex = pv_first;
      r->first_provoking_vertex_set = true;
      if (r->tiler)
         MALI_PER_ARCH(cmd_render_tiler_set_provoking_vertex)(cmd);
   } else {
      assert(r->first_provoking_vertex == pv_first);
   }

   if (!MALI_PER_ARCH(cmd_render_tiler)(cmd))
      return false;

   const struct mali_gfx_baked *dbk = NULL;
   if (unlikely(bd->dynamic)) {
      if (d->dirty & MALI_GFX_DIRTY_PIPELINE) {
         const struct mali_gfx_pack_input in = {
            .dyn = dyn,
            .rp = &p->rp,
            .vs = p->vs,
            .fs = p->fs,
         };
         d->dyn_baked.uses_dynamic_state = true;
         MALI_PER_ARCH(gfx_pack_state)(&in, &d->dyn_baked);
      }
      dbk = &d->dyn_baked;
   }
   *dbk_out = dbk;

   if (d->first_vertex != di->vertex_offset && bd->vs_first_vertex)
      d->dirty |= MALI_GFX_DIRTY_VS_FAU;
   if (d->base_instance != di->first_instance) {
#if PAN_ARCH >= 9
      /* The instanced attributes' offsets (in the VS table on v9+). */
      d->dirty |= MALI_GFX_DIRTY_VS_SRT;
#endif
      if (bd->vs_base_instance)
         d->dirty |= MALI_GFX_DIRTY_VS_FAU;
   }
   d->first_vertex = di->vertex_offset;
   d->base_instance = di->first_instance;

   /* Descriptor sets bound since a stage's table was built. */
   if (d->vs_sets_dirty & bd->vs_sets)
      d->dirty |= MALI_GFX_DIRTY_VS_SRT;
   if (d->fs_sets_dirty & bd->fs_sets)
      d->dirty |= MALI_GFX_DIRTY_FS_SRT;
#if PAN_ARCH < 9
   /* v7: the draw marks a stage's FAU group for its uniform-buffer words
    * when the stage's tables or dynamic offsets change
    * (mali_jm_cmd_draw_v7.c). */
#else
   /* A fragment shader with uniform-buffer words in FAU needs them again
    * whenever its sets or dynamic offsets change. */
   if ((d->dirty & MALI_GFX_DIRTY_FS_SRT) && p->fs && p->fs->fau.ubo_push_count)
      d->dirty |= MALI_GFX_DIRTY_FS_FAU;
#endif

   r->tls_size = MAX2(r->tls_size, bd->tls_size);
   return true;
}

#if PAN_ARCH >= 9
/* MALI_GFX_DIRTY_BLEND: *out = the Blend descriptors' address | their
 * count (v11 BLEND_DESC register, v9 Draw descriptor Blend word). Marks
 * the FS FAU dirty. */
static inline bool
mali_gfx_build_blend(struct mali_cmd_buffer *cmd, uint64_t *out)
{
   uint32_t n = 0;
   uint64_t bl = mali_gfx_blend(cmd, &n);
   if (!bl)
      return false;
   *out = bl | n;
   cmd->gfx.draw.dirty |= MALI_GFX_DIRTY_FS_FAU;
   return true;
}
#endif

/* MALI_GFX_DIRTY_VS_FAU / FS_FAU: *out = the FAU word of the stage's
 * block. sv is the draw's sysvals (mali_gfx_fill_sysvals). */
static inline bool
mali_gfx_build_vs_fau(struct mali_cmd_buffer *cmd, const struct mali_graphics_sysvals *sv,
                      uint64_t *out)
{
   const struct mali_shader *vs = cmd->gfx.draw.pipeline->vs;
   uint64_t fau = MALI_PER_ARCH(cmd_gfx_fau)(cmd, vs, sv, cmd->push_constants,
                                             sizeof(cmd->push_constants));
   if (vs->fau.total_count && !fau)
      return false;
   *out = fau;
   return true;
}

static inline bool
mali_gfx_build_fs_fau(struct mali_cmd_buffer *cmd, const struct mali_graphics_sysvals *sv,
                      uint64_t *out)
{
   const struct mali_shader *fs = cmd->gfx.draw.pipeline->fs;
   uint64_t fau = fs ? MALI_PER_ARCH(cmd_gfx_fau)(cmd, fs, sv, cmd->push_constants,
                                                  sizeof(cmd->push_constants)) : 0;
   if (fs && fs->fau.total_count && !fau)
      return false;
   *out = fau;
   return true;
}

#if PAN_ARCH >= 9
/* MALI_GFX_DIRTY_VS_SRT / FS_SRT: *out = the stage's resource table
 * pointer (0 for a pipeline without a fragment shader). */
static inline bool
mali_gfx_build_vs_srt(struct mali_cmd_buffer *cmd, uint64_t *out)
{
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   uint64_t srt = mali_gfx_vs_srt(cmd);
   if (!srt)
      return false;
   *out = srt;
   d->vs_srt_key = d->pipeline->bind.vs_srt_key;
   d->vs_sets_dirty = 0;
   return true;
}

static inline bool
mali_gfx_build_fs_srt(struct mali_cmd_buffer *cmd, uint64_t *out)
{
   struct mali_gfx_draw_state *d = &cmd->gfx.draw;
   const struct mali_graphics_pipeline *p = d->pipeline;
   uint64_t srt = p->fs ? mali_gfx_fs_srt(cmd) : 0;
   if (p->fs && !srt)
      return false;
   *out = srt;
   d->fs_srt_key = p->bind.fs_srt_key;
   d->fs_sets_dirty = 0;
   return true;
}

/* MALI_GFX_DIRTY_ZSD: *out = the Depth/stencil descriptor: the pipeline's
 * own, or a copy of the repacked one (dbk from mali_gfx_update_dirty). */
static inline bool
mali_gfx_build_zsd(struct mali_cmd_buffer *cmd, const struct mali_gfx_baked *dbk, uint64_t *out)
{
   uint64_t zsd = cmd->gfx.draw.pipeline->bind.zsd;
   if (!zsd) {
      struct mali_ptr z = mali_cmd_alloc(cmd, sizeof(dbk->zsd), 32);
      if (!z.cpu)
         return false;
      memcpy(z.cpu, dbk->zsd, sizeof(dbk->zsd));
      zsd = z.gpu;
   }
   *out = zsd;
   return true;
}
#endif /* PAN_ARCH >= 9 */

/* ---------------------------------------------------------------------- */
/* Render pass                                                             */

/* The pass's Local Storage descriptor, allocated on first use and filled
 * at the end of the pass (mali_fb_fill_tsd). */
static inline bool
mali_fb_alloc_tsd(struct mali_cmd_buffer *cmd, struct mali_render_state *r)
{
   if (r->tsd)
      return true;
   struct mali_ptr p = mali_cmd_alloc(cmd, pan_size(LOCAL_STORAGE), 64);
   if (!p.cpu)
      return false;
   r->tsd = p.gpu;
   r->tsd_cpu = p.cpu;
   return true;
}

#endif
