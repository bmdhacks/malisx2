/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The debug switch "cpuprof" (mali_debug_flag): how much CPU time each of
 * the application's threads spends inside the driver, and in which entry
 * points. The device's dispatch table gets a timing wrapper in front of
 * the entry points an emulator calls per draw, per pass and per frame
 * (the ones vkGetDeviceProcAddr hands out after vkCreateDevice); each
 * thread adds up its own calls and logs them about once a second:
 *
 *   malisx2 cpu: tid 1234: 8.1 ms/s in the driver (1.00 s): draw 37000
 *   calls 3.2 ms, binddesc 18000 1.1 ms, ...
 *
 * Off, nothing is wrapped and the table is as vk_device_init built it.
 * One device a process: the wrapped entry points' targets are kept in
 * file-scope pointers.
 */

#include "mali_vk.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "util/log.h"
#include "util/os_time.h"

/* X(name, short name, return, parameters, arguments). The name is the
 * dispatch table field; ALIAS(name, alias) also wraps the KHR/EXT field
 * of the same entry point. */
#define CPUPROF_VOID(X)                                                          \
   X(CmdDraw, "draw", (VkCommandBuffer a, uint32_t b, uint32_t c, uint32_t d,   \
                       uint32_t e), (a, b, c, d, e))                             \
   X(CmdDrawIndexed, "drawidx", (VkCommandBuffer a, uint32_t b, uint32_t c,     \
                                 uint32_t d, int32_t e, uint32_t f),             \
     (a, b, c, d, e, f))                                                         \
   X(CmdBindPipeline, "pipe", (VkCommandBuffer a, VkPipelineBindPoint b,        \
                               VkPipeline c), (a, b, c))                         \
   X(CmdBindDescriptorSets, "binddesc",                                          \
     (VkCommandBuffer a, VkPipelineBindPoint b, VkPipelineLayout c, uint32_t d,  \
      uint32_t e, const VkDescriptorSet *f, uint32_t g, const uint32_t *h),      \
     (a, b, c, d, e, f, g, h))                                                   \
   X(CmdPushDescriptorSet, "pushdesc",                                           \
     (VkCommandBuffer a, VkPipelineBindPoint b, VkPipelineLayout c, uint32_t d,  \
      uint32_t e, const VkWriteDescriptorSet *f), (a, b, c, d, e, f))            \
   X(CmdPushConstants, "pushconst",                                              \
     (VkCommandBuffer a, VkPipelineLayout b, VkShaderStageFlags c, uint32_t d,   \
      uint32_t e, const void *f), (a, b, c, d, e, f))                            \
   X(CmdBindVertexBuffers, "vb",                                                 \
     (VkCommandBuffer a, uint32_t b, uint32_t c, const VkBuffer *d,              \
      const VkDeviceSize *e), (a, b, c, d, e))                                   \
   X(CmdBindVertexBuffers2, "vb2",                                               \
     (VkCommandBuffer a, uint32_t b, uint32_t c, const VkBuffer *d,              \
      const VkDeviceSize *e, const VkDeviceSize *f, const VkDeviceSize *g),      \
     (a, b, c, d, e, f, g))                                                      \
   X(CmdBindIndexBuffer, "ib", (VkCommandBuffer a, VkBuffer b, VkDeviceSize c,  \
                                VkIndexType d), (a, b, c, d))                    \
   X(CmdSetVertexInputEXT, "vi",                                                 \
     (VkCommandBuffer a, uint32_t b, const VkVertexInputBindingDescription2EXT *c, \
      uint32_t d, const VkVertexInputAttributeDescription2EXT *e),               \
     (a, b, c, d, e))                                                            \
   X(CmdBeginRendering, "begin", (VkCommandBuffer a, const VkRenderingInfo *b), \
     (a, b))                                                                     \
   X(CmdEndRendering, "end", (VkCommandBuffer a), (a))                           \
   X(CmdPipelineBarrier2, "barrier", (VkCommandBuffer a, const VkDependencyInfo *b), \
     (a, b))                                                                     \
   X(CmdBeginQuery, "qbegin", (VkCommandBuffer a, VkQueryPool b, uint32_t c,    \
                               VkQueryControlFlags d), (a, b, c, d))             \
   X(CmdEndQuery, "qend", (VkCommandBuffer a, VkQueryPool b, uint32_t c),       \
     (a, b, c))                                                                  \
   X(CmdResetQueryPool, "qreset", (VkCommandBuffer a, VkQueryPool b, uint32_t c,\
                                   uint32_t d), (a, b, c, d))                    \
   X(CmdCopyQueryPoolResults, "qcopy",                                           \
     (VkCommandBuffer a, VkQueryPool b, uint32_t c, uint32_t d, VkBuffer e,      \
      VkDeviceSize f, VkDeviceSize g, VkQueryResultFlags h),                     \
     (a, b, c, d, e, f, g, h))                                                   \
   X(CmdCopyBufferToImage, "upload",                                             \
     (VkCommandBuffer a, VkBuffer b, VkImage c, VkImageLayout d, uint32_t e,     \
      const VkBufferImageCopy *f), (a, b, c, d, e, f))                           \
   X(CmdCopyBuffer, "copybuf",                                                   \
     (VkCommandBuffer a, VkBuffer b, VkBuffer c, uint32_t d,                     \
      const VkBufferCopy *e), (a, b, c, d, e))                                   \
   X(CmdDispatch, "dispatch", (VkCommandBuffer a, uint32_t b, uint32_t c,       \
                               uint32_t d), (a, b, c, d))                        \
   X(UpdateDescriptorSets, "updatedesc",                                         \
     (VkDevice a, uint32_t b, const VkWriteDescriptorSet *c, uint32_t d,         \
      const VkCopyDescriptorSet *e), (a, b, c, d, e))

#define CPUPROF_RESULT(X)                                                        \
   X(BeginCommandBuffer, "cbbegin", (VkCommandBuffer a,                          \
                                     const VkCommandBufferBeginInfo *b), (a, b)) \
   X(EndCommandBuffer, "cbend", (VkCommandBuffer a), (a))                        \
   X(QueueSubmit, "submit1", (VkQueue a, uint32_t b, const VkSubmitInfo *c,     \
                              VkFence d), (a, b, c, d))                          \
   X(QueueSubmit2, "submit", (VkQueue a, uint32_t b, const VkSubmitInfo2 *c,    \
                              VkFence d), (a, b, c, d))                          \
   X(QueuePresentKHR, "present", (VkQueue a, const VkPresentInfoKHR *b), (a, b)) \
   X(AcquireNextImageKHR, "acquire",                                             \
     (VkDevice a, VkSwapchainKHR b, uint64_t c, VkSemaphore d, VkFence e,        \
      uint32_t *f), (a, b, c, d, e, f))                                          \
   X(WaitForFences, "waitfence", (VkDevice a, uint32_t b, const VkFence *c,     \
                                  VkBool32 d, uint64_t e), (a, b, c, d, e))      \
   X(GetFenceStatus, "fencestat", (VkDevice a, VkFence b), (a, b))              \
   X(ResetFences, "resetfence", (VkDevice a, uint32_t b, const VkFence *c),     \
     (a, b, c))                                                                  \
   X(WaitSemaphores, "waitsem", (VkDevice a, const VkSemaphoreWaitInfo *b,      \
                                 uint64_t c), (a, b, c))                         \
   X(GetSemaphoreCounterValue, "semval", (VkDevice a, VkSemaphore b, uint64_t *c), \
     (a, b, c))                                                                  \
   X(GetQueryPoolResults, "qresults",                                            \
     (VkDevice a, VkQueryPool b, uint32_t c, uint32_t d, size_t e, void *f,      \
      VkDeviceSize g, VkQueryResultFlags h), (a, b, c, d, e, f, g, h))           \
   X(AllocateDescriptorSets, "allocdesc",                                        \
     (VkDevice a, const VkDescriptorSetAllocateInfo *b, VkDescriptorSet *c),     \
     (a, b, c))                                                                  \
   X(ResetDescriptorPool, "resetpool", (VkDevice a, VkDescriptorPool b,         \
                                        VkDescriptorPoolResetFlags c), (a, b, c)) \
   X(ResetCommandPool, "resetcmdpool", (VkDevice a, VkCommandPool b,            \
                                        VkCommandPoolResetFlags c), (a, b, c))   \
   X(FlushMappedMemoryRanges, "flush", (VkDevice a, uint32_t b,                 \
                                        const VkMappedMemoryRange *c), (a, b, c))

/* Entry points that also have a KHR or EXT name of their own in the table. */
#define CPUPROF_ALIASES(A)                                                       \
   A(CmdPushDescriptorSet, CmdPushDescriptorSetKHR)                              \
   A(CmdBindVertexBuffers2, CmdBindVertexBuffers2EXT)                            \
   A(CmdBeginRendering, CmdBeginRenderingKHR)                                    \
   A(CmdEndRendering, CmdEndRenderingKHR)                                        \
   A(CmdPipelineBarrier2, CmdPipelineBarrier2KHR)                                \
   A(QueueSubmit2, QueueSubmit2KHR)                                              \
   A(WaitSemaphores, WaitSemaphoresKHR)                                          \
   A(GetSemaphoreCounterValue, GetSemaphoreCounterValueKHR)

enum {
#define X(name, s, p, a) E_##name,
   CPUPROF_VOID(X) CPUPROF_RESULT(X)
#undef X
   E_COUNT
};

static const char *const names[E_COUNT] = {
#define X(name, s, p, a) [E_##name] = s,
   CPUPROF_VOID(X) CPUPROF_RESULT(X)
#undef X
};

static const char *const part_names[MALI_CPUPROF_PART_COUNT] = {
   [MALI_CPUPROF_IDXSCAN] = "(idxscan",
   [MALI_CPUPROF_DRAWPREP] = "(drawprep",
};

struct thread_stats {
   uint64_t ns[E_COUNT];
   uint32_t calls[E_COUNT];
   uint64_t part_ns[MALI_CPUPROF_PART_COUNT];
   uint32_t part_calls[MALI_CPUPROF_PART_COUNT];
   uint64_t since;
};

static _Thread_local struct thread_stats stats;

bool mali_cpuprof_on;

static void
report(struct thread_stats *t, uint64_t now)
{
   char line[1536];
   uint64_t total = 0;
   for (unsigned i = 0; i < E_COUNT; i++)
      total += t->ns[i];
   const double secs = (now - t->since) / 1e9;
   int n = snprintf(line, sizeof(line), "malisx2 cpu: tid %d: %.1f ms/s in the driver (%.2f s):",
                    (int)gettid(), total / 1e6 / secs, secs);
   for (unsigned i = 0; i < E_COUNT && n < (int)sizeof(line) - 48; i++) {
      if (!t->calls[i])
         continue;
      n += snprintf(line + n, sizeof(line) - n, " %s %u %.2f", names[i], t->calls[i],
                    t->ns[i] / 1e6 / secs);
   }
   /* Sections, in parentheses: already in their entry points' times. */
   for (unsigned i = 0; i < MALI_CPUPROF_PART_COUNT && n < (int)sizeof(line) - 48; i++) {
      if (!t->part_calls[i])
         continue;
      n += snprintf(line + n, sizeof(line) - n, " %s %u %.2f)", part_names[i],
                    t->part_calls[i], t->part_ns[i] / 1e6 / secs);
   }
   mesa_logi("%s", line);
   memset(t, 0, sizeof(*t));
   t->since = now;
}

void
mali_cpuprof_part(enum mali_cpuprof_part part, uint64_t t0)
{
   struct thread_stats *t = &stats;
   t->part_ns[part] += os_time_get_nano() - t0;
   t->part_calls[part]++;
}

static inline void
account(unsigned e, uint64_t t0)
{
   const uint64_t now = os_time_get_nano();
   struct thread_stats *t = &stats;
   if (unlikely(!t->since))
      t->since = t0;
   t->ns[e] += now - t0;
   t->calls[e]++;
   if (unlikely(now - t->since >= 1000000000ull))
      report(t, now);
}

#define X(name, s, params, args)                                                 \
   static PFN_vk##name orig_##name;                                              \
   static VKAPI_ATTR void VKAPI_CALL w_##name params                             \
   {                                                                             \
      const uint64_t t0 = os_time_get_nano();                                    \
      orig_##name args;                                                          \
      account(E_##name, t0);                                                     \
   }
CPUPROF_VOID(X)
#undef X

#define X(name, s, params, args)                                                 \
   static PFN_vk##name orig_##name;                                              \
   static VKAPI_ATTR VkResult VKAPI_CALL w_##name params                         \
   {                                                                             \
      const uint64_t t0 = os_time_get_nano();                                    \
      const VkResult r = orig_##name args;                                       \
      account(E_##name, t0);                                                     \
      return r;                                                                  \
   }
CPUPROF_RESULT(X)
#undef X

void
mali_cpuprof_install(struct vk_device_dispatch_table *t)
{
   if (!mali_debug_flag("cpuprof"))
      return;
   mali_cpuprof_on = true;
#define A(name, alias)                                                           \
   if (!t->name && t->alias)                                                     \
      t->name = t->alias;
   CPUPROF_ALIASES(A)
#undef A
#define X(name, s, p, a)                                                         \
   if (t->name) {                                                                \
      orig_##name = t->name;                                                     \
      t->name = w_##name;                                                        \
   }
   CPUPROF_VOID(X) CPUPROF_RESULT(X)
#undef X
#define A(name, alias)                                                           \
   if (t->alias)                                                                 \
      t->alias = t->name;
   CPUPROF_ALIASES(A)
#undef A
   mesa_logi("malisx2: cpuprof: driver CPU time per thread, logged about once a second");
}
