/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Uniform buffer words in FAU on the job manager (v7 and v9), and the copy
 * job that puts them there.
 *
 * mali_shader.c (lower_ubo_push) turns a shader's constant-offset reads of
 * dynamic uniform buffers into FAU reads and leaves, in fau.ubo_push, the
 * (buffer, offset) of every word it wants there. v11 fills those words with
 * command-stream loads and stores. The job manager has no command stream,
 * so the draw's vtc chain gets one small Compute job per stage that copies
 * them from the bound buffers into the FAU block before the job that reads
 * it: the Vertex job depends on the vertex shader's copy, and the fragment
 * job starts only after the whole vtc atom, so the fragment shader's copy
 * (also on the vtc chain) needs no dependency. A word whose half lies past
 * the binding's range, or a buffer bound to nothing, is copied from a zero
 * word instead.
 *
 * The copy shader is built here (a compute shader of one invocation per
 * 32-bit half, dst[i] = *table[i]) and kept per device, as the descriptor
 * and other meta shaders are.
 */

#include "mali_vk.h"
#include "mali_jm.h"
#include "mali_shader.h"

#include "mali_cmd_state.h"

#include <string.h>

#include "nir_builder.h"
#include "util/mesa-blake3.h"
#include "vk_pipeline.h"

#include "mali_arch.h"

/*
 * The shader of a meta job, kept in *slot (struct mali_jm_device). On first
 * use, build makes its NIR and the normal path compiles it, keyed by a hash
 * of name; the shader then stays until the device is destroyed, as the copy
 * and fill shaders of mali_cmd_copy.c do. NULL on failure, with the command
 * buffer's error set. A compiled compute shader always has its Renderer
 * State Descriptor (spd[MALI_SPD_MAIN], mali_shader.c).
 */
const struct mali_shader *
mali_jm_meta_shader(struct mali_cmd_buffer *cmd, struct mali_shader **slot,
                    nir_shader *(*build)(struct mali_device *dev), const char *name)
{
   struct mali_device *dev = cmd->dev;

   /* Set once under the lock and kept until the device is destroyed, so a
    * caller that finds it set needs no lock. */
   struct mali_shader *s = __atomic_load_n(slot, __ATOMIC_ACQUIRE);
   if (likely(s))
      return s;
   simple_mtx_lock(&dev->meta_lock);
   s = *slot;
   if (!s) {
      nir_shader *nir = build(dev);
      MALI_PER_ARCH(shader_preprocess)(dev, nir);

      const struct vk_pipeline_robustness_state rs = {
         .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
         .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
         .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
         .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
      };
      const struct mali_shader_compile_info info = {.nir = nir, .rs = &rs};
      blake3_hash key;
      struct mesa_blake3 h;
      _mesa_blake3_init(&h);
      _mesa_blake3_update(&h, name, strlen(name));
      _mesa_blake3_final(&h, key);
      if (MALI_PER_ARCH(shader_compile)(dev, &info, key, &s) != VK_SUCCESS)
         s = NULL;
      __atomic_store_n(slot, s, __ATOMIC_RELEASE);
   }
   simple_mtx_unlock(&dev->meta_lock);

   if (!s)
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   assert(!s || s->spd[MALI_SPD_MAIN]);
   return s;
}

/*
 * The Local Storage descriptor of a copy job: the same for every copy job
 * of the command buffer when the copy shader has no thread or workgroup
 * storage, so made once per command buffer.
 */
uint64_t
mali_jm_copy_tsd(struct mali_cmd_buffer *cmd, const struct mali_shader *copy,
                 const uint32_t groups[3])
{
   if (copy->info.tls_size || copy->info.wls_size) {
      const struct mali_thread_props tp = mali_thread_props(cmd->dev);
      return mali_compute_tsd(cmd, copy, &tp, groups);
   }
   if (!cmd->jm.copy_tsd) {
      const struct mali_thread_props tp = mali_thread_props(cmd->dev);
      cmd->jm.copy_tsd = mali_compute_tsd(cmd, copy, &tp, groups);
   }
   return cmd->jm.copy_tsd;
}

#if PAN_ARCH < 9
/*
 * Adds a Compute job running copy shader copy, with push as its push
 * constants, on chain c behind job dep (0: none). It runs count workgroups
 * of one invocation, the workgroup size the copy shaders are compiled with
 * (panvk runs its descriptor copy as one workgroup of count). *index (when
 * not NULL): the job's index. False on failure, with the command buffer's
 * error set.
 */
bool
mali_jm_add_copy_job(struct mali_cmd_buffer *cmd, struct mali_jm_chain *c,
                     const struct mali_shader *copy, const void *push, uint32_t push_size,
                     uint32_t count, uint16_t dep, uint16_t *index)
{
   const uint32_t base[3] = {0, 0, 0};
   const uint32_t groups[3] = {count, 1, 1};
   const uint32_t local[3] = {1, 1, 1};
   const uint64_t fau =
      mali_compute_fau(cmd, copy, NULL, push, push_size, base, groups) & BITFIELD64_MASK(56);
   const uint64_t tsd = mali_jm_copy_tsd(cmd, copy, groups);
   if (vk_command_buffer_has_error(&cmd->vk) || !tsd || (copy->fau.total_count && !fau))
      return false;

   struct mali_ptr job =
      mali_jm_cmd_add_job(cmd, c, MALI_JOB_TYPE_COMPUTE, pan_size(COMPUTE_JOB), false, dep);
   if (!job.cpu)
      return false;
   mali_v7_compute_invocation(pan_section_ptr(job.cpu, COMPUTE_JOB, INVOCATION), local, groups);
   pan_section_pack(job.cpu, COMPUTE_JOB, PARAMETERS, cfg) {
      cfg.job_task_split = mali_v7_compute_task_split(local);
   }
   pan_section_pack(job.cpu, COMPUTE_JOB, DRAW, cfg) {
      cfg.state = copy->spd[MALI_SPD_MAIN];
      cfg.push_uniforms = fau;
      cfg.thread_storage = tsd;
   }
   if (index)
      *index = c->index;
   return true;
}
#else
/*
 * The v9 form of the same: one Compute job of count workgroups of one
 * invocation, with the task split and resource table a v9 compute job has
 * (mali_jm_cmd_dispatch.c). The copy shader has no descriptor sets, so its
 * resource table is the driver's (the dummy sampler).
 */
bool
mali_jm_add_copy_job(struct mali_cmd_buffer *cmd, struct mali_jm_chain *c,
                     const struct mali_shader *copy, const void *push, uint32_t push_size,
                     uint32_t count, uint16_t dep, uint16_t *index)
{
   const struct mali_thread_props tp = mali_thread_props(cmd->dev);
   const uint32_t base[3] = {0, 0, 0};
   const uint32_t groups[3] = {count, 1, 1};
   const uint64_t fau = mali_compute_fau(cmd, copy, push, push_size, base, groups);
   const uint64_t srt = mali_compute_res_table(cmd, copy, NULL, NULL, 0);
   const uint64_t tsd = mali_jm_copy_tsd(cmd, copy, groups);
   if (vk_command_buffer_has_error(&cmd->vk) || !srt || !tsd ||
       (copy->fau.total_count && !fau))
      return false;

   unsigned axis, inc;
   mali_compute_task_axis(&tp, copy, groups, &axis, &inc);

   struct mali_ptr job =
      mali_jm_cmd_add_job(cmd, c, MALI_JOB_TYPE_COMPUTE, pan_size(COMPUTE_JOB), false, dep);
   if (!job.cpu)
      return false;

   pan_section_pack(job.cpu, COMPUTE_JOB, PAYLOAD, cfg) {
      cfg.workgroup_size_x = copy->cs.local_size[0];
      cfg.workgroup_size_y = copy->cs.local_size[1];
      cfg.workgroup_size_z = copy->cs.local_size[2];
      cfg.allow_merging_workgroups = copy->info.cs.allow_merging_workgroups;
      /* 14 bits; the builder's increment is a few workgroups per task. */
      cfg.task_increment = MIN2(inc, (1u << 14) - 1);
      cfg.task_axis = axis;
      cfg.workgroup_count_x = groups[0];
      cfg.workgroup_count_y = groups[1];
      cfg.workgroup_count_z = groups[2];
      cfg.compute.resources = srt;
      cfg.compute.shader = copy->spd[MALI_SPD_MAIN];
      cfg.compute.thread_storage = tsd;
      cfg.compute.fau = fau & BITFIELD64_MASK(56);
      cfg.compute.fau_count = fau >> 56;
   }
   if (index)
      *index = c->index;
   return true;
}
#endif

/* ---------------------------------------------------------------------- */
/* The uniform buffer word copy job                                        */

/* The copy shader's push constants (FAU). */
struct ubo_copy_info {
   uint64_t table;
   uint64_t dst;
};

static nir_shader *
build_ubo_copy_nir(struct mali_device *dev)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, MALI_PER_ARCH(shader_nir_options)(dev, MESA_SHADER_COMPUTE), "%s",
      "mali-jm-ubo-copy");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *id = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *table =
      nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, offsetof(struct ubo_copy_info, table)));
   nir_def *dst =
      nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, offsetof(struct ubo_copy_info, dst)));
   nir_def *src = nir_load_global(
      &b, 1, 64, nir_iadd(&b, table, nir_u2u64(&b, nir_imul_imm(&b, id, 8))), .align_mul = 8);
   nir_def *v = nir_load_global(&b, 1, 32, src, .align_mul = 4);
   nir_store_global(&b, v, nir_iadd(&b, dst, nir_u2u64(&b, nir_imul_imm(&b, id, 4))),
                    .align_mul = 4);
   return b.shader;
}

bool
mali_jm_ubo_copy(struct mali_cmd_buffer *cmd, struct mali_jm_chain *c, const struct mali_shader *s,
                 const struct mali_desc_state *desc, uint64_t fau_va, uint16_t dep,
                 uint16_t *index)
{
   const uint32_t n = s->fau.ubo_push_count;
   if (index)
      *index = dep;

   if (!n || !fau_va)
      return true;
   const struct mali_shader *copy =
      mali_jm_meta_shader(cmd, &cmd->dev->jm->ubo_copy, build_ubo_copy_nir, "mali-jm-ubo-copy");
   if (!copy)
      return false;

#if PAN_ARCH < 9
   const uint32_t dyn_count = s->v7.dyn_ubo_count;
#else
   /* Only dynamic uniform buffer loads become pushed words, and those are
    * the first entries of desc.dyn_bufs, so every w->dyn is inside it. */
   const uint32_t dyn_count = s->desc.dyn_bufs.count;
#endif

   /* 2n source addresses, then the zero word they fall back to. */
   struct mali_ptr p = mali_cmd_alloc(cmd, (2 * n + 1) * sizeof(uint64_t), 16);
   if (!p.cpu)
      return false;
   const uint64_t zero = p.gpu + 2 * n * sizeof(uint64_t);
   uint64_t table[2 * MALI_MAX_UBO_PUSH_FAUS + 1];
   uint32_t dyn_cached = UINT32_MAX;
   uint64_t addr = 0, range = 0;
   for (uint32_t i = 0; i < n; i++) {
      const struct mali_ubo_push_word *w = &s->fau.ubo_push[i];
      if (w->dyn != dyn_cached) {
         dyn_cached = w->dyn;
         if (w->dyn >= dyn_count ||
             !mali_jm_desc_dyn_buf(desc, s->desc.dyn_bufs.map[w->dyn], &addr, &range) ||
             (addr >> 56))
            addr = range = 0;
      }
      for (unsigned h = 0; h < 2; h++) {
         const uint64_t off = w->offset + 4 * h;
         table[2 * i + h] = addr && off + 4 <= range ? addr + off : zero;
      }
   }
   table[2 * n] = 0;
   memcpy(p.cpu, table, (2 * n + 1) * sizeof(uint64_t));

   const struct ubo_copy_info info = {
      .table = p.gpu,
      .dst = fau_va + (uint64_t)s->fau.ubo_push_start * MALI_FAU_WORD_SIZE,
   };
   return mali_jm_add_copy_job(cmd, c, copy, &info, sizeof(info), 2 * n, dep, index);
}

void
mali_jm_ubo_finish(struct mali_device *dev)
{
   struct mali_jm_device *jd = dev->jm;
   MALI_PER_ARCH(shader_unref)(dev, jd->ubo_copy);
   jd->ubo_copy = NULL;
}
