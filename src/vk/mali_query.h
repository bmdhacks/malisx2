/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Query pools: the object both back halves share (mali_query.c), and the
 * hooks the command-buffer side of each back half provides.
 *
 * A pool holds, for each query, a 64-bit value and a 64-bit availability
 * word. Availability is non-zero once the query's result is final.
 * The two arrays are addressed separately (a base and a stride each)
 * because where they live differs:
 *
 *  - job manager (v7, v9): one buffer, 16-byte slots, value at +0 and
 *    availability at +8. The GPU writes both from Write Value jobs (a
 *    System Timestamp, an Immediate 64) and, for occlusion, the fragment
 *    hardware accumulates into the value word. The kernel cleans the GPU
 *    caches when an atom ends, so the host sees both once the atom has
 *    completed;
 *  - command-stream frontend (v11), timestamp pools: one buffer, 32-byte
 *    slots: the value at +0, the availability word at +16. A STORE_STATE
 *    writes the value, a SYNC_SET64 (system scope) the availability word;
 *    the host sees the value once it sees the availability word;
 *  - v11 occlusion pools: the counters in one buffer, 8 bytes each (the
 *    fragment hardware adds to them from every shader core, which wants
 *    the L2 cache to do the atomics), and the availability words in a
 *    second buffer, 16 bytes each. The stream cleans the caches before it
 *    sets an availability word, so a host that sees availability sees the
 *    counter.
 *
 * All of it is ordinary device memory. CSF event memory (the flag the
 * queue's own sync objects use) would let the stream's writes reach the
 * host without going through the GPU caches, but a shader reads zeros from
 * it on the G615: vkCmdCopyQueryPoolResults, which reads the availability
 * words in a shader, saw every query as unavailable.
 *
 * On v11 the availability word is a 64-bit sync object, the target of a
 * SYNC_SET64: 16 bytes at a 16-byte aligned address (the value, then an
 * error word the driver leaves alone). A SYNC_SET64 anywhere else stops the
 * queue group with CS_INVALID_INSTRUCTION, which is why the strides are not
 * 8 and the availability word is not at +8.
 */

#ifndef MALI_QUERY_H
#define MALI_QUERY_H

#include <stdbool.h>
#include <stdint.h>

#include "vk_query_pool.h"

#include "kbase/kbase.h"

#include "mali_arch.h"

struct mali_device;
struct mali_cmd_buffer;

struct mali_query_pool {
   struct vk_query_pool vk;
   /* The values (and, where val_stride says so, the availability words). */
   struct mali_kbase_bo bo;
   /* v11 occlusion pools: the availability words. Unused (size 0) otherwise. */
   struct mali_kbase_bo avail_bo;

   uint64_t val_va, avail_va;
   uint8_t *val_cpu;
   volatile uint64_t *avail_cpu;
   uint32_t val_stride, avail_stride;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(mali_query_pool, vk.base, VkQueryPool, VK_OBJECT_TYPE_QUERY_POOL)

static inline uint64_t
mali_query_val_va(const struct mali_query_pool *pool, uint32_t q)
{
   return pool->val_va + (uint64_t)q * pool->val_stride;
}

static inline uint64_t
mali_query_avail_va(const struct mali_query_pool *pool, uint32_t q)
{
   return pool->avail_va + (uint64_t)q * pool->avail_stride;
}

static inline volatile uint64_t *
mali_query_avail_cpu(const struct mali_query_pool *pool, uint32_t q)
{
   return (volatile uint64_t *)((volatile uint8_t *)pool->avail_cpu +
                                (uint64_t)q * pool->avail_stride);
}

static inline uint64_t
mali_query_val_cpu(const struct mali_query_pool *pool, uint32_t q)
{
   return *(const volatile uint64_t *)(pool->val_cpu + (uint64_t)q * pool->val_stride);
}

/* vkCmdResetQueryPool zeroes at most this many queries with Write Value
 * jobs (job manager) before it switches to a fill dispatch: the 16 or so
 * ARMSX2 resets per frame stay the jobs they always were, and a reset of
 * a whole pool does not become tens of thousands of jobs. */
#define MALI_QUERY_RESET_JOBS_MAX 16

/*
 * The ordering around a dispatch that reads or writes query memory, for the
 * queries' own commands (reset, copy). before: the dispatch follows
 * everything recorded so far, including the writes of availability words
 * that were still on their way (a copy with VK_QUERY_RESULT_WAIT_BIT reads
 * a result that is final); after: what is recorded next follows the
 * dispatch. Each back half's query file (mali_jm_query.c,
 * mali_cmd_query.c) implements it.
 */
MALI_PER_ARCH_DECL(void, cmd_query_fence, (struct mali_cmd_buffer *cmd, bool before));

/*
 * The end of a render pass, after its fragment jobs: sets the availability
 * words of the occlusion queries that ended inside it (mali_jm_query.c,
 * mali_cmd_query.c). Called whether or not the pass had fragment work.
 */
MALI_PER_ARCH_DECL(void, cmd_oq_pass_end, (struct mali_cmd_buffer *cmd));

/*
 * Fills size bytes at the GPU address addr with the 32-bit word `value`,
 * with the internal fill shader (mali_cmd_copy.c). addr and size are
 * multiples of 4.
 */
MALI_PER_ARCH_DECL(void, cmd_fill_addr,
                   (struct mali_cmd_buffer *cmd, uint64_t addr, uint64_t size, uint32_t value));

#endif
