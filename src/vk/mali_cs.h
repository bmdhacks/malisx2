/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * What the command buffers and the queue agree on about the command
 * streams: the subqueues, the scoreboard entries, the CS register
 * layout, and the GPU-visible sync objects.
 *
 * The layout is panvk's for v10 and v11 (Mesa src/panfrost/vulkan/csf/
 * panvk_cmd_buffer.h): panvk gives the subqueue-to-stage mapping and the
 * scoreboard use, and its register layout for both stays inside the 96
 * registers the G615 has. The part that does not depend on the arch is in
 * mali_cs_defs.h. What differs from panvk is only what panthor
 * did for it and kbase does not: the queue writes the submit-time words
 * into the kernel rings itself (mali_queue.c).
 */

#ifndef MALI_CS_H
#define MALI_CS_H

/* Built per command-stream arch (10, 11 and 12), never once. */
#ifndef PAN_ARCH
#error "mali_cs.h needs PAN_ARCH"
#endif

#include <stdint.h>

#include "mali_cs_defs.h"

#include "genxml/gen_macros.h"
#include "genxml/cs_builder.h"

#if PAN_ARCH >= 11
/*
 * Shared scoreboard entries (v11 SHARED_SB_INC / SHARED_SB_DEC, opcodes
 * 30/31; v10 has neither and uses the barrier counter instead). A second set of counters, shared by the streams of one queue
 * group: SHARED_SB_INC on one stream adds one to an entry once the
 * scoreboard entries it waits on are done (immediately, or deferred like
 * any asynchronous instruction); SHARED_SB_DEC on another stream blocks
 * that stream until the entry is non-zero and takes one off. kbase counts
 * a stream blocked on a decrement as idle (CS_REQ_IDLE_SHARED_SB_DEC,
 * CS_STATUS_BLOCKED_ON_SB_WAIT), so the wait is handled by the front end,
 * not by a firmware sync-object check.
 *
 * The blob hands out one entry per (source, destination) subqueue pair
 * from 16, starting at 4 (2 and 3 are fixed for its transfer subqueue,
 * 0 and 1 unused). Only the vertex/tiler -> fragment pass dependency
 * uses one here.
 */
#define MALI_SHARED_SB_VT_TO_FRAG 4

static inline void
mali_cs_shared_sb_inc(struct cs_builder *b, unsigned entry, struct cs_async_op async)
{
   cs_emit(b, SHARED_SB_INC, I) {
      I.shared_entry = entry;
      if (async.indirect)
         I.defer_mode = MALI_CS_SHARED_SB_INCREMENT_DEFER_MODE_DEFER_INDIRECT;
      else
         I.sb_mask = async.wait_mask;
   }
}

static inline void
mali_cs_shared_sb_dec(struct cs_builder *b, unsigned entry)
{
   cs_emit(b, SHARED_SB_DEC, I) {
      I.shared_entry = entry;
   }
}
#endif

static inline struct cs_index
mali_cs_scratch_reg_tuple(struct cs_builder *b, unsigned start, unsigned count)
{
   assert(MALI_CS_REG_SCRATCH_START + start + count <= MALI_CS_REG_SCRATCH_END + 1);
   return cs_reg_tuple(b, MALI_CS_REG_SCRATCH_START + start, count);
}

static inline struct cs_index
mali_cs_scratch_reg32(struct cs_builder *b, unsigned reg)
{
   return mali_cs_scratch_reg_tuple(b, reg, 1);
}

static inline struct cs_index
mali_cs_scratch_reg64(struct cs_builder *b, unsigned reg)
{
   assert(reg % 2 == 0);
   return mali_cs_scratch_reg_tuple(b, reg, 2);
}

static inline struct cs_index
mali_cs_progress_seqno_reg(struct cs_builder *b, enum mali_subqueue sq)
{
   return cs_reg64(b, MALI_CS_REG_PROGRESS_SEQNO_START + 2 * sq);
}

static inline struct cs_index
mali_cs_subqueue_ctx_reg(struct cs_builder *b)
{
   return cs_reg64(b, MALI_CS_REG_SUBQUEUE_CTX);
}

/* The resources each subqueue asks the firmware for (panvk). */
static inline uint32_t
mali_subqueue_resources(enum mali_subqueue sq)
{
   switch (sq) {
   case MALI_SUBQUEUE_VERTEX_TILER: return CS_IDVS_RES | CS_TILER_RES;
   case MALI_SUBQUEUE_FRAGMENT: return CS_FRAG_RES;
   default: return CS_COMPUTE_RES;
   }
}

#endif
