/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The part of the command-stream layout that does not depend on the GPU
 * architecture: the subqueues, the scoreboard entry numbers, the CS
 * register numbers and the sync objects and subqueue context in GPU
 * memory. Code built once (device creation, sync files, the measurement
 * output) includes this; the per-arch command-stream code includes
 * mali_cs.h, which adds Mesa's cs_builder for its PAN_ARCH.
 */

#ifndef MALI_CS_DEFS_H
#define MALI_CS_DEFS_H

#include <stdint.h>

/* The three subqueues, one kernel CS queue each, bound at CSI slots 0..2 of
 * the device's queue group (panvk's split; the blob uses five). */
enum mali_subqueue {
   MALI_SUBQUEUE_VERTEX_TILER = 0,
   MALI_SUBQUEUE_FRAGMENT,
   MALI_SUBQUEUE_COMPUTE,
   MALI_SUBQUEUE_COUNT,
};

#define MALI_SUBQUEUE_ALL ((1u << MALI_SUBQUEUE_COUNT) - 1)

/* Scoreboard entries (panvk). Entry 0 tracks loads and stores and the
 * immediate cache flushes, entry 1 the deferred sync operations, entry 2
 * deferred flushes, entries 3.. the iterators (dispatches, draws): up to
 * 13 of them on v11, 5 on v10, which has 8 entries per stream. */
enum mali_sb {
   MALI_SB_LS = 0,
   MALI_SB_IMM_FLUSH = 0,
   MALI_SB_DEFERRED_SYNC = 1,
   MALI_SB_DEFERRED_FLUSH = 2,
   MALI_SB_ITER_START = 3,
};

#define MALI_SB_MASK(x) (1u << (x))

/*
 * Registers. 0..63 are the RUN_* staging registers (RUN_COMPUTE uses 0..39,
 * RUN_IDVS 0..63 on v11, 0..60 on v10), 66..83 scratch for command-buffer code. The next
 * ones keep their value across command buffers and submits:
 *
 *   84..89  progress sequence numbers: for each subqueue j, a 64-bit count
 *           of the barrier signals subqueue j made in earlier command
 *           buffers. Every subqueue keeps the same view (panvk's scheme,
 *           cs_progress_seqno_reg).
 *   90..91  the subqueue context address (struct mali_cs_subqueue_ctx).
 *
 * 92..95 are what cs_builder calls kernel registers: its chunk-chaining
 * JUMPs use them inside command buffers, and the queue uses them for the
 * words it writes into the ring between command buffers.
 */
#define MALI_CS_REG_SCRATCH_START 66
#define MALI_CS_REG_SCRATCH_END   83
/* Scratch 16..17 (r82:83) belong to the measurement layer's timestamp
 * address (src/measurement/timing.c); command code uses 0..15. */
#define MALI_CS_SCRATCH_MEASURE 16
#define MALI_CS_REG_PROGRESS_SEQNO_START 84
#define MALI_CS_REG_SUBQUEUE_CTX 90
#define MALI_CS_KERNEL_REGS 4
#define MALI_CS_RING_REG0 92   /* 92..95: ring-level temporaries */

/* A 64-bit GPU sync object (panvk_cs_sync64). error is set by sync
 * operations with error propagation when the stream has faulted. */
struct mali_cs_sync64 {
   uint64_t seqno;
   uint32_t error;
   uint32_t pad;
};

/*
 * Per-subqueue context in GPU memory, its address in MALI_CS_REG_SUBQUEUE_CTX
 * from the subqueue's first words on. Render passes add the tiler
 * fields' users; the queue fills them now.
 */
struct mali_cs_subqueue_ctx {
   uint64_t syncobjs;     /* struct mali_cs_sync64[MALI_SUBQUEUE_COUNT]:
                             barrier counters, one per subqueue */
   uint64_t done;         /* struct mali_cs_sync64[MALI_SUBQUEUE_COUNT]:
                             last submit completed on each subqueue */
   uint64_t tiler_heap;   /* the heap's Tiler Heap descriptor */
   uint64_t geom_buf;     /* geometry buffer, size code in the low bits */
   uint32_t last_error;
   uint32_t pad[7];
} __attribute__((aligned(64)));

/* Scoreboard facts of the device's CS interface. */
struct mali_cs_sb_info {
   uint8_t count;           /* entries per CS, at most 16 */
   uint16_t all_mask;
   uint8_t iter_count;
   uint16_t all_iters_mask;
};

#endif
