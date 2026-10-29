/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Command buffers: memory, the per-subqueue streams, begin/end, barriers.
 * (The pipeline, descriptor-set and push-constant binds are in
 * mali_cmd_state.c, shared with the job-manager back half.)
 *
 * The recording follows panvk v11 (Mesa csf/panvk_vX_cmd_buffer.c), which
 * "which source wins" makes the reference for command-stream encoding,
 * scoreboards and the subqueue-to-stage mapping; the blob records the same
 * way in substance. Every stream ends with a wait on all its scoreboard
 * entries and an L2/LSC clean, so a command buffer's writes are complete
 * and in memory when its stream returns to the ring.
 */

#include "mali_cmd_buffer.h"

#include <stdlib.h>
#include <string.h>

#include "vk_alloc.h"
#include "vk_command_pool.h"
#include "vk_log.h"
#include "vk_pipeline.h"
#include "vk_synchronization.h"
#include "vk_util.h"

#include "mali_arch.h"
#include "mali_descriptor_set.h"
#include "mali_image.h"
#include "mali_measure.h"
#include "mali_pipeline.h"
#include "mali_queue.h"
#include "mali_vk.h"

/* ---------------------------------------------------------------------- */
/* Memory                                                                  */

/* How many free slabs the device keeps for reuse (32 MiB). A frame of
 * several thousand draws uses about 100 slabs, and ARMSX2 keeps a few
 * frames in flight; with a smaller cache every command buffer reset gave
 * slabs back to the kernel and the next frame mapped them again, about a
 * tenth of the GS thread's time on Xenosaga. */
#define MALI_CMD_FREE_SLABS_MAX 512

static struct mali_cmd_slab *
slab_new(struct mali_device *dev, uint64_t size, bool recycle)
{
   if (recycle) {
      simple_mtx_lock(&dev->slab_lock);
      struct mali_cmd_slab *s = NULL;
      if (!list_is_empty(&dev->free_slabs)) {
         s = list_first_entry(&dev->free_slabs, struct mali_cmd_slab, link);
         list_del(&s->link);
         dev->free_slab_count--;
      }
      simple_mtx_unlock(&dev->slab_lock);
      if (s)
         return s;
   }

   struct mali_cmd_slab *s = calloc(1, sizeof(*s));
   if (!s)
      return NULL;
   /* CPU and GPU read/write, CPU-uncached (the blob's payload pool; its CS
    * chunks are CPU-cached and cleaned at vkEndCommandBuffer instead).
    * Uncached needs no maintenance. */
   const struct mali_kbase_alloc_info ai = {
      .size = size,
      .flags = mali_kbase_mem_same_va_policy(MALI_KBASE_FLAGS_DEVICE_UNCACHED),
      .mem_class = MALI_KBASE_MEM_CLASS_COMMAND_ALLOCATOR,
   };
   if (mali_kbase_alloc(dev->kbase, &ai, &s->bo) != MALI_KBASE_SUCCESS) {
      free(s);
      return NULL;
   }
   s->recycle = recycle;
   return s;
}

static void
slab_release(struct mali_device *dev, struct mali_cmd_slab *s)
{
   if (s->recycle) {
      simple_mtx_lock(&dev->slab_lock);
      if (dev->free_slab_count < MALI_CMD_FREE_SLABS_MAX) {
         list_add(&s->link, &dev->free_slabs);
         dev->free_slab_count++;
         s = NULL;
      }
      simple_mtx_unlock(&dev->slab_lock);
      if (!s)
         return;
   }
   mali_kbase_free(dev->kbase, &s->bo);
   free(s);
}

void
mali_cmd_slabs_finish(struct mali_device *dev)
{
   list_for_each_entry_safe(struct mali_cmd_slab, s, &dev->free_slabs, link) {
      list_del(&s->link);
      mali_kbase_free(dev->kbase, &s->bo);
      free(s);
   }
   dev->free_slab_count = 0;
}

struct mali_ptr
MALI_PER_ARCH(cmd_alloc_slow)(struct mali_cmd_buffer *cmd, uint64_t size, uint64_t align)
{
   assert(util_is_power_of_two_nonzero64(align) && align <= 4096);
   size = MAX2(size, 1);

   if (size > MALI_CMD_SLAB_SIZE / 2) {
      struct mali_cmd_slab *s = slab_new(cmd->dev, size, false);
      if (!s)
         goto oom;
      list_addtail(&s->link, &cmd->slabs);
      return (struct mali_ptr){s->bo.cpu, s->bo.gpu_va};
   }

   uint64_t off = ALIGN_POT(cmd->cur_offset, align);
   if (!cmd->cur || off + size > MALI_CMD_SLAB_SIZE) {
      struct mali_cmd_slab *s = slab_new(cmd->dev, MALI_CMD_SLAB_SIZE, true);
      if (!s)
         goto oom;
      list_addtail(&s->link, &cmd->slabs);
      cmd->cur = s;
      cmd->cur_cpu = s->bo.cpu;
      cmd->cur_gpu = s->bo.gpu_va;
      off = 0;
   }
   cmd->cur_offset = off + size;
   return (struct mali_ptr){
      (uint8_t *)cmd->cur->bo.cpu + off,
      cmd->cur->bo.gpu_va + off,
   };

oom:
   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   return (struct mali_ptr){0};
}

static void
cmd_release_memory(struct mali_cmd_buffer *cmd)
{
   list_for_each_entry_safe(struct mali_cmd_slab, s, &cmd->slabs, link) {
      list_del(&s->link);
      slab_release(cmd->dev, s);
   }
   cmd->cur = NULL;
   cmd->cur_cpu = NULL;
   cmd->cur_gpu = 0;
   cmd->cur_offset = 0;
   cmd->tls.size = 0;
   cmd->tls.gpu = 0;
}

/* ---------------------------------------------------------------------- */
/* Streams                                                                 */

static struct cs_buffer
cs_alloc_chunk(void *cookie)
{
   struct mali_cmd_buffer *cmd = cookie;
   struct mali_ptr p = mali_cmd_alloc(cmd, MALI_CS_CHUNK_SIZE, 64);
   return (struct cs_buffer){
      .cpu = p.cpu,
      .gpu = p.gpu,
      .capacity = p.cpu ? MALI_CS_CHUNK_SIZE / 8 : 0,
   };
}

static void
init_streams(struct mali_cmd_buffer *cmd)
{
   const struct mali_csf_device *csf = cmd->dev->csf;
   for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++) {
      const struct cs_builder_conf conf = {
         .nr_registers = csf->nr_registers,
         .nr_kernel_registers = MALI_CS_KERNEL_REGS,
         .alloc_buffer = cs_alloc_chunk,
         .cookie = cmd,
         .ls_sb_slot = MALI_SB_LS,
      };
      /* The root chunk is allocated with the first instruction. */
      cs_builder_init(&cmd->cs[i].b, &conf, (struct cs_buffer){0});
      cmd->cs[i].relative_sync_point = 0;
      cmd->cs[i].pending_wait = 0;
      cmd->cs[i].pending_flush = (struct mali_cache_flush){0};
      cmd->cs[i].pending_signal = false;
      cmd->cs[i].idle = false;
      cmd->cs[i].flushed = (struct mali_cache_flush){0};
      memset(cmd->cs[i].carried, 0, sizeof(cmd->cs[i].carried));
      memset(cmd->cs[i].waited, 0, sizeof(cmd->cs[i].waited));
      memset(cmd->cs[i].lazy, 0, sizeof(cmd->cs[i].lazy));
      cmd->cs[i].lazy_pending = false;
      cmd->cs[i].lazy_frag = false;
      cmd->cs[i].lazy_frag_flush = (struct mali_cache_flush){0};
   }
   /* The queue orders the compute subqueue's work from earlier command
    * buffers before any stream of this one that relies on it
    * (compute_prior_mask), so the compute stream starts idle: a barrier
    * that names it as a source needs no signal from it until it records
    * work. */
   cmd->cs[MALI_SUBQUEUE_COMPUTE].idle = true;
   cmd->compute_work = false;
   cmd->compute_end_lazy = 0;
   cmd->compute_prior_mask = 0;
}

static void
fini_streams(struct mali_cmd_buffer *cmd)
{
   for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++)
      cs_builder_fini(&cmd->cs[i].b);
}

bool
mali_cmd_buffer_span(struct mali_cmd_buffer *cmd, unsigned sq, uint64_t *addr,
                     uint32_t *size)
{
   struct cs_builder *b = &cmd->cs[sq].b;
   if (!cs_is_valid(b) || !b->root_chunk.buffer.gpu || !b->root_chunk.size)
      return false;
   *addr = cs_root_chunk_gpu_addr(b);
   *size = cs_root_chunk_size(b);
   return true;
}

uint32_t
mali_cmd_buffer_subqueue_mask(struct mali_cmd_buffer *cmd)
{
   uint32_t mask = 0;
   for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++) {
      uint64_t a;
      uint32_t s;
      if (mali_cmd_buffer_span(cmd, i, &a, &s))
         mask |= BITFIELD_BIT(i);
   }
   return mask;
}

/* ---------------------------------------------------------------------- */
/* Object                                                                  */

static void
cmd_reset_state(struct mali_cmd_buffer *cmd)
{
   memset(cmd->push_constants, 0, sizeof(cmd->push_constants));
   memset(&cmd->compute, 0, sizeof(cmd->compute));
   memset(&cmd->gfx, 0, sizeof(cmd->gfx));
   cmd->gfx.draw.dirty = MALI_GFX_DIRTY_ALL;
   cmd->dispatches = 0;
   cmd->draws = 0;
   cmd->passes = 0;
}

static VkResult
cmd_create(struct vk_command_pool *pool, VkCommandBufferLevel level,
           struct vk_command_buffer **out)
{
   struct mali_device *dev = container_of(pool->base.device, struct mali_device, vk);
   struct mali_cmd_buffer *cmd =
      vk_zalloc(&pool->alloc, sizeof(*cmd), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!cmd)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result = vk_command_buffer_init(pool, &cmd->vk, &mali_cmd_buffer_ops, level);
   if (result != VK_SUCCESS) {
      vk_free(&pool->alloc, cmd);
      return result;
   }
   cmd->dev = dev;
   cmd->vk.dynamic_graphics_state.vi = &cmd->dyn_vi;
   cmd->vk.dynamic_graphics_state.ms.sample_locations = &cmd->dyn_sl;
   list_inithead(&cmd->slabs);
   util_dynarray_init(&cmd->oq_done, NULL);
   init_streams(cmd);
   cmd_reset_state(cmd);
   mali_measure_cmd_create(cmd);
   *out = &cmd->vk;
   return VK_SUCCESS;
}

static void
cmd_reset(struct vk_command_buffer *vk_cmd, VkCommandBufferResetFlags flags)
{
   struct mali_cmd_buffer *cmd = container_of(vk_cmd, struct mali_cmd_buffer, vk);
   vk_command_buffer_reset(&cmd->vk);
   /* Timestamps live in command memory: read them before it goes. */
   mali_measure_cmd_reset(cmd);
   util_dynarray_clear(&cmd->oq_done);
   fini_streams(cmd);
   cmd_release_memory(cmd);
   init_streams(cmd);
   cmd_reset_state(cmd);
}

static void
cmd_destroy(struct vk_command_buffer *vk_cmd)
{
   struct mali_cmd_buffer *cmd = container_of(vk_cmd, struct mali_cmd_buffer, vk);
   mali_measure_cmd_destroy(cmd);
   util_dynarray_fini(&cmd->oq_done);
   fini_streams(cmd);
   cmd_release_memory(cmd);
   mali_push_shadows_fini(&cmd->push_shadow, &cmd->vk.pool->alloc);
   vk_command_buffer_finish(&cmd->vk);
   vk_free(&cmd->vk.pool->alloc, cmd);
}

const struct vk_command_buffer_ops mali_cmd_buffer_ops = {
   .create = cmd_create,
   .reset = cmd_reset,
   .destroy = cmd_destroy,
};

VKAPI_ATTR VkResult VKAPI_CALL
MALI_PER_ARCH(BeginCommandBuffer)(VkCommandBuffer commandBuffer,
                                  const VkCommandBufferBeginInfo *pBeginInfo)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);

   /* Resets a command buffer recorded before (implicit reset). */
   vk_command_buffer_begin(&cmd->vk, pBeginInfo);
   cmd->usage = pBeginInfo->flags;
   /* The recorded streams are not patched at submit (the per-submit words
    * live in the ring), so SIMULTANEOUS_USE needs nothing more; the blob
    * re-records those at every submit instead. */
   return VK_SUCCESS;
}

/* Bring every stream's progress registers up to date with the barrier
 * signals of this command buffer (panvk flush_sync_points): each subqueue
 * keeps the same view of every other subqueue's signal count. */
static void
flush_sync_points(struct mali_cmd_buffer *cmd)
{
   bool any = false;
   for (unsigned j = 0; j < MALI_SUBQUEUE_COUNT; j++)
      any |= cmd->cs[j].relative_sync_point != 0;
   if (!any)
      return;

   for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++) {
      struct cs_builder *b = &cmd->cs[i].b;
      for (unsigned j = 0; j < MALI_SUBQUEUE_COUNT; j++) {
         int32_t rel = cmd->cs[j].relative_sync_point;
         if (rel)
            cs_add_imm64(b, mali_cs_progress_seqno_reg(b, j),
                         mali_cs_progress_seqno_reg(b, j), rel);
      }
   }
   for (unsigned j = 0; j < MALI_SUBQUEUE_COUNT; j++)
      cmd->cs[j].relative_sync_point = 0;
}

/* The end of a stream: all its asynchronous work finished, and its writes
 * cleaned from L2 and the load/store caches to memory. Command memory is
 * recycled, and results are read by the host, so this is always needed
 * (panvk finish_cs; the blob's end-of-submit deferred clean does the
 * same). */
static void
finish_stream(struct mali_cmd_buffer *cmd, unsigned sq)
{
   const struct mali_csf_device *csf = cmd->dev->csf;
   struct cs_builder *b = &cmd->cs[sq].b;

   cs_wait_slots(b, csf->sb.all_mask);
   struct cs_index flush_id = mali_cs_scratch_reg32(b, 0);
   cs_move32_to(b, flush_id, 0);
   cs_flush_caches(b, MALI_CS_FLUSH_MODE_CLEAN, MALI_CS_FLUSH_MODE_CLEAN,
                   MALI_CS_OTHER_FLUSH_MODE_NONE, flush_id,
                   cs_defer(0, MALI_SB_IMM_FLUSH));
   cs_wait_slot(b, MALI_SB_IMM_FLUSH);
}

VKAPI_ATTR VkResult VKAPI_CALL
MALI_PER_ARCH(EndCommandBuffer)(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);

   /* Waits the compute subqueue put off until it records work are left
    * to the queue: its next work waits for everything recorded here
    * (compute_end_lazy). A barrier's put-off wait, flush and signal on
    * the fragment stream still apply to what follows in later command
    * buffers. */
   struct mali_cmd_cs *cc = &cmd->cs[MALI_SUBQUEUE_COMPUTE];
   if (cc->lazy_pending) {
      for (unsigned j = 0; j < MALI_SUBQUEUE_COUNT; j++) {
         if (cc->lazy[j] > cc->waited[j])
            cmd->compute_end_lazy |= BITFIELD_BIT(j);
      }
      if (cc->lazy_frag)
         cmd->compute_end_lazy |= BITFIELD_BIT(MALI_SUBQUEUE_FRAGMENT);
      cc->lazy_pending = false;
      cc->lazy_frag = false;
      cc->lazy_frag_flush = (struct mali_cache_flush){0};
      memset(cc->lazy, 0, sizeof(cc->lazy));
   }
   mali_cmd_frag_flush_pending(cmd);
   flush_sync_points(cmd);

   for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++) {
      struct cs_builder *b = &cmd->cs[i].b;
      if (!cs_is_valid(b)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         continue;
      }
      /* Streams nothing was recorded into stay empty and are not CALLed. */
      if (!cs_is_empty(b))
         finish_stream(cmd, i);
      cs_end(b);
      if (!cs_is_valid(b))
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

   return vk_command_buffer_end(&cmd->vk);
}

/* ---------------------------------------------------------------------- */
/* Barriers                                                                */

/* Stages each subqueue carries (panvk_get_subqueue_stages). */
static VkPipelineStageFlags2
subqueue_stages(unsigned sq)
{
   switch (sq) {
   case MALI_SUBQUEUE_VERTEX_TILER:
      return VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
             VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT |
             VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT |
             VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
   case MALI_SUBQUEUE_FRAGMENT:
      return VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
             VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
             VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_RESOLVE_BIT |
             VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT;
   default:
      return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
             VK_PIPELINE_STAGE_2_COPY_BIT;
   }
}

/* Execution dependency (panvk add_execution_dependency): each destination
 * subqueue waits for every source subqueue, and every source subqueue waits
 * for its own work when anyone (a subqueue or the host) depends on it. */
static void
add_execution_dependency(uint32_t wait_masks[MALI_SUBQUEUE_COUNT],
                         VkPipelineStageFlags2 src_stages,
                         VkPipelineStageFlags2 dst_stages)
{
   uint32_t src = 0, dst = 0;
   for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++) {
      if (src_stages & subqueue_stages(i))
         src |= BITFIELD_BIT(i);
      if (dst_stages & subqueue_stages(i))
         dst |= BITFIELD_BIT(i);
   }
   const bool dst_host = dst_stages & VK_PIPELINE_STAGE_2_HOST_BIT;
   if (!src || (!dst && !dst_host))
      return;

   u_foreach_bit(i, dst) {
      uint32_t mask = src;
      if (i == MALI_SUBQUEUE_VERTEX_TILER &&
          (src_stages & subqueue_stages(i)) == VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT)
         mask &= ~BITFIELD_BIT(i);   /* indirect reads are synchronous */
      if (i == MALI_SUBQUEUE_FRAGMENT)
         mask &= ~BITFIELD_BIT(MALI_SUBQUEUE_VERTEX_TILER); /* always waits */
      wait_masks[i] |= mask;
   }
   u_foreach_bit(i, src)
      wait_masks[i] |= BITFIELD_BIT(i);
}

/* Memory dependency (panvk add_memory_dependency): L2 is unified and the
 * load/store caches are coherent with it, so only the host domain and the
 * read-only L1 caches need maintenance. */
static void
add_memory_dependency(struct mali_cache_flush *f, VkAccessFlags2 src_access,
                      VkAccessFlags2 dst_access)
{
   const VkAccessFlags2 ro_l1 =
      VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
      VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
      VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT;

   if (dst_access & ro_l1)
      f->others |= MALI_CS_OTHER_FLUSH_MODE_INVALIDATE;
   if (src_access & VK_ACCESS_2_HOST_WRITE_BIT) {
      f->l2 |= MALI_CS_FLUSH_MODE_CLEAN_AND_INVALIDATE;
      f->lsc |= MALI_CS_FLUSH_MODE_CLEAN_AND_INVALIDATE;
      f->others |= MALI_CS_OTHER_FLUSH_MODE_INVALIDATE;
   }
   if (dst_access & (VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_HOST_WRITE_BIT)) {
      f->l2 |= MALI_CS_FLUSH_MODE_CLEAN;
      f->lsc |= MALI_CS_FLUSH_MODE_CLEAN;
   }
}

static void
collect_deps(struct mali_cmd_buffer *cmd, VkPipelineStageFlags2 src_stages,
             VkAccessFlags2 src_access, VkPipelineStageFlags2 dst_stages,
             VkAccessFlags2 dst_access, struct mali_cs_deps *deps)
{
   const struct mali_csf_device *csf = cmd->dev->csf;

   src_stages = vk_expand_src_stage_flags2(src_stages);
   src_access = vk_filter_src_access_flags2(src_stages, src_access);
   dst_stages = vk_expand_dst_stage_flags2(dst_stages);
   dst_access = vk_filter_dst_access_flags2(dst_stages, dst_access);

   uint32_t wait_masks[MALI_SUBQUEUE_COUNT] = {0};
   add_execution_dependency(wait_masks, src_stages, dst_stages);

   for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++) {
      /* The iterator scoreboard entries move, so a self-wait waits for all
       * of them. */
      if (wait_masks[i] & BITFIELD_BIT(i))
         deps->src[i].wait_sb_mask |= csf->sb.all_iters_mask;
      deps->dst[i].wait_subqueue_mask |= wait_masks[i];

      /* Cache maintenance where the dependency is: on a source subqueue
       * after its work, or on a destination before its reads. panvk flushes
       * on every subqueue; a flush on one that takes no part only makes an
       * otherwise empty stream run. */
      if (!((src_stages | dst_stages) & subqueue_stages(i)))
         continue;
      const VkPipelineStageFlags2 sq_stages =
         subqueue_stages(i) | VK_PIPELINE_STAGE_2_HOST_BIT;
      add_memory_dependency(&deps->src[i].flush,
                            vk_filter_src_access_flags2(sq_stages, src_access),
                            vk_filter_dst_access_flags2(sq_stages, dst_access));
   }
}

/*
 * Image barriers after which the image's CRCs may not describe its memory
 * (panvk image_barrier_invalidates_crc, plus UNDEFINED): contents from
 * before a transition from UNDEFINED are discarded and the memory may have
 * been written through an alias in between; PREINITIALIZED contents were
 * written by the host; an acquire from an external or foreign queue family
 * follows writes by someone else. Only level 0 of the colour aspect has
 * CRC state.
 */
static bool
image_barrier_invalidates_crc(const VkImageMemoryBarrier2 *m)
{
   if (m->subresourceRange.baseMipLevel != 0 ||
       !(m->subresourceRange.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT))
      return false;
   if (m->oldLayout == VK_IMAGE_LAYOUT_UNDEFINED ||
       m->oldLayout == VK_IMAGE_LAYOUT_PREINITIALIZED)
      return true;
   return m->srcQueueFamilyIndex != m->dstQueueFamilyIndex &&
          (m->srcQueueFamilyIndex == VK_QUEUE_FAMILY_EXTERNAL ||
           m->srcQueueFamilyIndex == VK_QUEUE_FAMILY_FOREIGN_EXT);
}

void
mali_cmd_add_deps(struct mali_cmd_buffer *cmd, const VkDependencyInfo *info,
                  struct mali_cs_deps *deps)
{
   for (uint32_t i = 0; i < info->memoryBarrierCount; i++) {
      const VkMemoryBarrier2 *m = &info->pMemoryBarriers[i];
      collect_deps(cmd, m->srcStageMask, m->srcAccessMask, m->dstStageMask,
                   m->dstAccessMask, deps);
   }
   for (uint32_t i = 0; i < info->bufferMemoryBarrierCount; i++) {
      const VkBufferMemoryBarrier2 *m = &info->pBufferMemoryBarriers[i];
      /* One queue family: ownership transfers only matter for the
       * external/foreign families, which count as the host. */
      VkPipelineStageFlags2 src = m->srcStageMask, dst = m->dstStageMask;
      VkAccessFlags2 srca = m->srcAccessMask, dsta = m->dstAccessMask;
      if (m->srcQueueFamilyIndex != m->dstQueueFamilyIndex) {
         if (m->srcQueueFamilyIndex == VK_QUEUE_FAMILY_FOREIGN_EXT) {
            src = VK_PIPELINE_STAGE_2_HOST_BIT;
            srca = VK_ACCESS_2_HOST_WRITE_BIT;
         } else if (m->srcQueueFamilyIndex == VK_QUEUE_FAMILY_EXTERNAL) {
            src = VK_PIPELINE_STAGE_2_NONE;
            srca = 0;
         }
         if (m->dstQueueFamilyIndex == VK_QUEUE_FAMILY_FOREIGN_EXT) {
            dst = VK_PIPELINE_STAGE_2_HOST_BIT;
            dsta = VK_ACCESS_2_HOST_WRITE_BIT;
         } else if (m->dstQueueFamilyIndex == VK_QUEUE_FAMILY_EXTERNAL) {
            dst = VK_PIPELINE_STAGE_2_NONE;
            dsta = 0;
         }
      }
      collect_deps(cmd, src, srca, dst, dsta, deps);
   }
   for (uint32_t i = 0; i < info->imageMemoryBarrierCount; i++) {
      const VkImageMemoryBarrier2 *m = &info->pImageMemoryBarriers[i];
      VkPipelineStageFlags2 src = m->srcStageMask, dst = m->dstStageMask;
      VkAccessFlags2 srca = m->srcAccessMask, dsta = m->dstAccessMask;
      if (m->srcQueueFamilyIndex != m->dstQueueFamilyIndex) {
         if (m->srcQueueFamilyIndex == VK_QUEUE_FAMILY_FOREIGN_EXT) {
            src = VK_PIPELINE_STAGE_2_HOST_BIT;
            srca = VK_ACCESS_2_HOST_WRITE_BIT;
         } else if (m->srcQueueFamilyIndex == VK_QUEUE_FAMILY_EXTERNAL) {
            src = VK_PIPELINE_STAGE_2_NONE;
            srca = 0;
         }
         if (m->dstQueueFamilyIndex == VK_QUEUE_FAMILY_FOREIGN_EXT) {
            dst = VK_PIPELINE_STAGE_2_HOST_BIT;
            dsta = VK_ACCESS_2_HOST_WRITE_BIT;
         } else if (m->dstQueueFamilyIndex == VK_QUEUE_FAMILY_EXTERNAL) {
            dst = VK_PIPELINE_STAGE_2_NONE;
            dsta = 0;
         }
      }
      /* Layout transitions change nothing in memory: AFBC images stay
       * compressed in every layout. A CRC table may need invalidating
       * (image_barrier_invalidates_crc, below). */
      collect_deps(cmd, src, srca, dst, dsta, deps);
      if (image_barrier_invalidates_crc(m))
         MALI_PER_ARCH(cmd_crc_invalidate)(cmd, container_of(vk_image_from_handle(m->image),
                                                   struct mali_image, vk));
   }
}

static bool
flush_is_nop(const struct mali_cache_flush *f)
{
   return f->l2 == MALI_CS_FLUSH_MODE_NONE && f->lsc == MALI_CS_FLUSH_MODE_NONE &&
          f->others == MALI_CS_OTHER_FLUSH_MODE_NONE;
}

/* Wait for the scoreboard entries in wait_sb_mask, then flush. The flush
 * is deferred behind those entries rather than issued after a WAIT for
 * them, so it starts as soon as they are done instead of when the stream
 * gets past the WAIT; the stream then waits once, for the flush. */
static void
emit_wait_and_flush(struct cs_builder *b, uint32_t wait_sb_mask,
                    const struct mali_cache_flush *flush)
{
   if (flush_is_nop(flush)) {
      if (wait_sb_mask)
         cs_wait_slots(b, wait_sb_mask);
      return;
   }
   struct cs_index flush_id = mali_cs_scratch_reg32(b, 0);
   cs_move32_to(b, flush_id, 0);
   cs_flush_caches(b, flush->l2, flush->lsc, flush->others, flush_id,
                   cs_defer(wait_sb_mask & ~MALI_SB_MASK(MALI_SB_IMM_FLUSH),
                            MALI_SB_IMM_FLUSH));
   cs_wait_slots(b, wait_sb_mask | MALI_SB_MASK(MALI_SB_IMM_FLUSH));
}

void
mali_cmd_frag_flush_pending(struct mali_cmd_buffer *cmd)
{
   struct mali_cmd_cs *cs = &cmd->cs[MALI_SUBQUEUE_FRAGMENT];
   if (!cs->pending_wait && flush_is_nop(&cs->pending_flush) && !cs->pending_signal)
      return;
   struct cs_builder *b = &cs->b;
   emit_wait_and_flush(b, cs->pending_wait, &cs->pending_flush);
   if (cs->pending_signal) {
      /* The barrier signal other subqueues wait for; its number was taken
       * when the barrier was recorded (mali_cmd_emit_barrier). */
      struct cs_index addr = mali_cs_scratch_reg64(b, 0);
      struct cs_index one = mali_cs_scratch_reg64(b, 2);
      cs_move64_to(b, addr, mali_queue_syncobj_va(cmd->dev->csf->queue,
                                                   MALI_SUBQUEUE_FRAGMENT));
      cs_move64_to(b, one, 1);
      cs_sync64_add(b, true, MALI_CS_SYNC_SCOPE_CSG, one, addr, cs_now());
      cs->pending_signal = false;
   }
   cs->pending_wait = 0;
   cs->pending_flush = (struct mali_cache_flush){0};
}

/* The clean that went with a stream's last signal covers what f asks of it
 * for other subqueues. Only L2 and load/store cleans matter to them: an
 * "other" invalidate is for the stream's own reads. */
static bool
flush_covered(const struct mali_cache_flush *done, const struct mali_cache_flush *f)
{
   return (unsigned)f->l2 <= (unsigned)done->l2 && (unsigned)f->lsc <= (unsigned)done->lsc;
}

/*
 * panvk emit_barrier_csf. Per source subqueue: wait for its scoreboard
 * entries, flush caches, and if another subqueue depends on it, add one to
 * its barrier counter. Per destination: wait until each source's counter
 * passes the value this signal gives it (progress register + signals so
 * far in this command buffer).
 *
 * Departures:
 *
 * - When no other subqueue waits on the fragment subqueue, its own wait
 *   and cache flush are put off until just before its next fragment job
 *   (or the end of the command buffer), as the blob does with its fragment
 *   pending mask and its r49 invalidate request. Consecutive barriers and
 *   the preload invalidate of mali_cmd_render_begin then become one wait
 *   and one flush, and the pass's wait for its tiling runs while the
 *   previous fragment job is still finishing.
 *
 * - A source that has recorded nothing since its last signal in this
 *   command buffer (struct mali_cmd_cs: idle) does not signal again: its
 *   destinations wait for that last signal, and for whatever it waited on
 *   since (carried), unless they already have. ARMSX2's transfer barriers
 *   name the compute subqueue (transfer stages run there too) on both
 *   sides although it usually has no work; without this, every such
 *   barrier made the fragment subqueue wait for a signal from the idle
 *   compute subqueue, which first had to see the fragment subqueue's own
 *   signal: two firmware-evaluated sync waits in a row on the fragment
 *   subqueue's critical path.
 *
 * - When only the compute subqueue waits for the fragment subqueue, the
 *   fragment subqueue's signal is put off with its wait and flush
 *   (pending_signal) instead of forcing them out at the barrier. Waits on
 *   the fragment and vertex/tiler subqueues are emitted only after it
 *   (emit_sync_wait), so nothing the fragment subqueue waits for before
 *   emitting it can depend on it.
 *
 * - The compute subqueue's waits are put off until it records work
 *   (lazy[], mali_cmd_compute_flush_waits), and a fragment-subqueue signal
 *   that only the compute subqueue would wait for is not made until then
 *   (lazy_frag). ARMSX2's transfer barriers name the compute subqueue as a
 *   destination (transfer stages run there too), but its transfers run on
 *   the fragment subqueue or not at all in most frames: before this, each
 *   such barrier cost the fragment subqueue a SYNC_ADD before its next
 *   fragment job, 10-25 us on the device, for a compute stream that only
 *   waited and did nothing. A
 *   subqueue that depends on the compute subqueue inherits its put-off
 *   waits, so dependency chains through it still hold.
 */

/* A barrier wait of stream k for signal v of subqueue j. */
static void
emit_sync_wait(struct mali_cmd_buffer *cmd, unsigned k, unsigned j, int32_t v)
{
   const struct mali_csf_queue *q = cmd->dev->csf->queue;
   struct mali_cmd_cs *ck = &cmd->cs[k];
   struct cs_builder *b = &ck->b;

   if (j == k || v <= 0 || v <= ck->waited[j])
      return;
   /* A wait on the fragment or vertex/tiler subqueue goes after the
    * fragment subqueue's put-off signal: the fragment subqueue waits for
    * the vertex/tiler subqueue's tiling before it emits that signal, and
    * for a source of a barrier before its next job, so a wait there on
    * something that waits for the signal would never end. Only the
    * compute subqueue waits for a put-off signal. */
   if (k != MALI_SUBQUEUE_COMPUTE && cmd->cs[MALI_SUBQUEUE_FRAGMENT].pending_signal)
      mali_cmd_frag_flush_pending(cmd);
   struct cs_index addr = mali_cs_scratch_reg64(b, 0);
   struct cs_index ref = mali_cs_scratch_reg64(b, 2);
   cs_move64_to(b, addr, mali_queue_syncobj_va(q, j));
   cs_add_imm64(b, ref, mali_cs_progress_seqno_reg(b, j), v);
   cs_sync64_wait(b, false, MALI_CS_CONDITION_GREATER, ref, addr);
   ck->waited[j] = v;
   ck->carried[j] = MAX2(ck->carried[j], v);
}

/* A fragment-subqueue signal after all fragment work recorded so far: the
 * put-off one if there is one (no fragment job was issued since it was
 * made), else a new one, put off the same way. Returns its number. */
static int32_t
frag_signal(struct mali_cmd_buffer *cmd, const struct mali_cache_flush *flush)
{
   struct mali_cmd_cs *cs = &cmd->cs[MALI_SUBQUEUE_FRAGMENT];
   cs->pending_wait |= cmd->dev->csf->sb.all_iters_mask;
   cs->pending_flush.l2 |= flush->l2;
   cs->pending_flush.lsc |= flush->lsc;
   cs->pending_flush.others |= flush->others;
   if (!cs->pending_signal) {
      cs->pending_signal = true;
      cs->relative_sync_point++;
      cs->idle = true;
      memset(cs->carried, 0, sizeof(cs->carried));
   }
   cs->flushed = cs->pending_flush;
   return cs->relative_sync_point;
}

static void
lazy_add_frag(struct mali_cmd_cs *c, const struct mali_cache_flush *flush)
{
   c->lazy_frag = true;
   c->lazy_pending = true;
   c->lazy_frag_flush.l2 |= flush->l2;
   c->lazy_frag_flush.lsc |= flush->lsc;
   c->lazy_frag_flush.others |= flush->others;
}

void
mali_cmd_compute_flush_waits(struct mali_cmd_buffer *cmd)
{
   struct mali_cmd_cs *c = &cmd->cs[MALI_SUBQUEUE_COMPUTE];
   if (!c->lazy_pending)
      return;
   c->lazy_pending = false;
   if (c->lazy_frag) {
      const int32_t v = frag_signal(cmd, &c->lazy_frag_flush);
      c->lazy[MALI_SUBQUEUE_FRAGMENT] = MAX2(c->lazy[MALI_SUBQUEUE_FRAGMENT], v);
      c->lazy_frag = false;
      c->lazy_frag_flush = (struct mali_cache_flush){0};
   }
   for (unsigned j = 0; j < MALI_SUBQUEUE_COUNT; j++) {
      emit_sync_wait(cmd, MALI_SUBQUEUE_COMPUTE, j, c->lazy[j]);
      c->lazy[j] = 0;
   }
}

void
mali_cmd_emit_barrier(struct mali_cmd_buffer *cmd, const struct mali_cs_deps *in)
{
   const struct mali_csf_queue *q = cmd->dev->csf->queue;
   const unsigned C = MALI_SUBQUEUE_COMPUTE, F = MALI_SUBQUEUE_FRAGMENT;
   struct mali_cs_deps deps = *in;
   uint32_t waited = 0;
   /* need[k][j]: a signal of j that k must wait for without a new one. */
   int32_t need[MALI_SUBQUEUE_COUNT][MALI_SUBQUEUE_COUNT] = {{0}};
   /* need_frag[k]: k needs a fragment signal after the fragment work
    * recorded so far, which does not exist yet (inherited from the compute
    * subqueue's put-off waits). */
   bool need_frag[MALI_SUBQUEUE_COUNT] = {false};
   struct mali_cache_flush need_frag_flush = {0};

   for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++) {
      /* A self-wait covers what a wait on our own counter would. */
      if (deps.src[i].wait_sb_mask)
         deps.dst[i].wait_subqueue_mask &= ~BITFIELD_BIT(i);
   }

   /* Whoever waits for the compute subqueue waits for what it put off. */
   const struct mali_cmd_cs *cc = &cmd->cs[C];
   if (cc->lazy_pending) {
      for (unsigned k = 0; k < MALI_SUBQUEUE_COUNT; k++) {
         if (k == C || !(deps.dst[k].wait_subqueue_mask & BITFIELD_BIT(C)))
            continue;
         for (unsigned j = 0; j < MALI_SUBQUEUE_COUNT; j++) {
            if (j != k)
               need[k][j] = MAX2(need[k][j], cc->lazy[j]);
         }
         if (cc->lazy_frag) {
            need_frag[k] = true;
            need_frag_flush.l2 |= cc->lazy_frag_flush.l2;
            need_frag_flush.lsc |= cc->lazy_frag_flush.lsc;
            need_frag_flush.others |= cc->lazy_frag_flush.others;
         }
      }
   }

   for (unsigned k = 0; k < MALI_SUBQUEUE_COUNT; k++) {
      u_foreach_bit(i, deps.dst[k].wait_subqueue_mask) {
         const struct mali_cmd_cs *si = &cmd->cs[i];
         if (!si->idle || !flush_covered(&si->flushed, &deps.src[i].flush))
            continue;
         deps.dst[k].wait_subqueue_mask &= ~BITFIELD_BIT(i);
         /* The compute stream with no signal yet: what it did before
          * this command buffer is the queue's to order. */
         if (i == C && !si->relative_sync_point)
            cmd->compute_prior_mask |= BITFIELD_BIT(k);
         need[k][i] = MAX2(need[k][i], si->relative_sync_point);
         for (unsigned j = 0; j < MALI_SUBQUEUE_COUNT; j++) {
            if (j != k)
               need[k][j] = MAX2(need[k][j], si->carried[j]);
         }
         /* carried[k] (i waited for k's own signal) needs nothing: k
          * drained its work before that signal, in its own stream,
          * so everything k records later already follows it. */
      }
   }

   /* The compute subqueue's wait for the fragment subqueue is put off
    * (lazy_frag): it makes no fragment signal now. */
   if (deps.dst[C].wait_subqueue_mask & BITFIELD_BIT(F)) {
      deps.dst[C].wait_subqueue_mask &= ~BITFIELD_BIT(F);
      need_frag[C] = true;
      need_frag_flush.l2 |= deps.src[F].flush.l2;
      need_frag_flush.lsc |= deps.src[F].flush.lsc;
      need_frag_flush.others |= deps.src[F].flush.others;
   }

   for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++)
      waited |= deps.dst[i].wait_subqueue_mask;

   for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++) {
      struct mali_cmd_cs *cs = &cmd->cs[i];
      struct cs_builder *b = &cs->b;

      if (i == F) {
         const bool is_waited = waited & BITFIELD_BIT(i);
         cs->pending_wait |= deps.src[i].wait_sb_mask;
         cs->pending_flush.l2 |= deps.src[i].flush.l2;
         cs->pending_flush.lsc |= deps.src[i].flush.lsc;
         cs->pending_flush.others |= deps.src[i].flush.others;
         if (is_waited) {
            /* One put-off signal serves every barrier recorded until it
             * is emitted: it comes after all of their work. */
            assert(deps.src[i].wait_sb_mask);
            if (!cs->pending_signal) {
               cs->pending_signal = true;
               cs->relative_sync_point++;
            }
            cs->idle = true;
            cs->flushed = cs->pending_flush;
            memset(cs->carried, 0, sizeof(cs->carried));
         }
         continue;
      }

      emit_wait_and_flush(b, deps.src[i].wait_sb_mask, &deps.src[i].flush);

      if (waited & BITFIELD_BIT(i)) {
         assert(deps.src[i].wait_sb_mask);
         struct cs_index addr = mali_cs_scratch_reg64(b, 0);
         struct cs_index one = mali_cs_scratch_reg64(b, 2);
         cs_move64_to(b, addr, mali_queue_syncobj_va(q, i));
         cs_move64_to(b, one, 1);
         cs_sync64_add(b, true, MALI_CS_SYNC_SCOPE_CSG, one, addr, cs_now());
         cs->relative_sync_point++;
         /* Everything the stream did is behind this signal now (the
          * compute subqueue's put-off waits are not: its waiters
          * inherited them above). */
         cs->idle = true;
         cs->flushed = deps.src[i].flush;
         memset(cs->carried, 0, sizeof(cs->carried));
      }
   }

   for (unsigned k = 0; k < MALI_SUBQUEUE_COUNT; k++) {
      struct mali_cmd_cs *ck = &cmd->cs[k];
      if (k == C) {
         /* Put off until the compute subqueue records work. */
         for (unsigned j = 0; j < MALI_SUBQUEUE_COUNT; j++) {
            int32_t v = need[k][j];
            if (deps.dst[k].wait_subqueue_mask & BITFIELD_BIT(j))
               v = MAX2(v, cmd->cs[j].relative_sync_point);
            if (j == k || v <= 0 || v <= ck->waited[j])
               continue;
            ck->lazy[j] = MAX2(ck->lazy[j], v);
            ck->lazy_pending = true;
         }
         if (need_frag[k])
            lazy_add_frag(ck, &need_frag_flush);
         continue;
      }
      /* The fragment subqueue's own earlier work is behind its drain. */
      if (need_frag[k] && k == F)
         cmd->cs[F].pending_wait |= cmd->dev->csf->sb.all_iters_mask;
      else if (need_frag[k])
         need[k][F] = MAX2(need[k][F], frag_signal(cmd, &need_frag_flush));
      for (unsigned j = 0; j < MALI_SUBQUEUE_COUNT; j++) {
         int32_t v = need[k][j];
         if (deps.dst[k].wait_subqueue_mask & BITFIELD_BIT(j))
            v = MAX2(v, cmd->cs[j].relative_sync_point);
         emit_sync_wait(cmd, k, j, v);
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
MALI_PER_ARCH(CmdPipelineBarrier2)(VkCommandBuffer commandBuffer,
                                   const VkDependencyInfo *pDependencyInfo)
{
   VK_FROM_HANDLE(mali_cmd_buffer, cmd, commandBuffer);
   struct mali_cs_deps deps = {0};
   mali_cmd_add_deps(cmd, pDependencyInfo, &deps);

   /* Inside a render pass that has tiled work (panvk collect_cs_deps):
    * the pass's fragment job is emitted at its end, so a by-region
    * dependency between fragment stages is a primitive barrier in the
    * tiler stream, not a wait on the fragment subqueue; and the
    * vertex/tiler subqueue does not wait for its own draws, which share
    * one scoreboard entry. */
   if (cmd->gfx.render.active && cmd->gfx.render.tiler) {
      const unsigned vt = MALI_SUBQUEUE_VERTEX_TILER, frag = MALI_SUBQUEUE_FRAGMENT;
      if (pDependencyInfo->dependencyFlags & VK_DEPENDENCY_BY_REGION_BIT) {
         MALI_PER_ARCH(cmd_fb_barrier)(cmd);
         deps.dst[frag].wait_subqueue_mask &= ~BITFIELD_BIT(frag);
         deps.src[frag].wait_sb_mask = 0;
      }
      bool vt_waited = false;
      for (unsigned i = 0; i < MALI_SUBQUEUE_COUNT; i++)
         vt_waited |= i != vt && (deps.dst[i].wait_subqueue_mask & BITFIELD_BIT(vt));
      deps.dst[vt].wait_subqueue_mask &= ~BITFIELD_BIT(vt);
      if (!vt_waited)
         deps.src[vt].wait_sb_mask = 0;
   }
   mali_cmd_emit_barrier(cmd, &deps);
}
