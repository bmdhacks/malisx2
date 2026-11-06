/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The internal shaders of indirect draws on the job manager (v7 and v9).
 *
 * The job manager cannot take a draw's counts from memory, so every draw of
 * a vkCmdDraw*Indirect* is recorded with placeholder counts and a patch job
 * in front of it (MALI_META_INDIRECT_DRAW): one invocation that reads the
 * draw's VkDraw(Indexed)IndirectCommand, and the draw count for the
 * indirect count forms, and writes the values that depend on them into the
 * draw's jobs and descriptors, or turns the jobs into Null jobs when the
 * draw draws nothing. The jobs it patches go into the chain as Not Started
 * jobs that depend on it, and it has Suppress Prefetch set, so the job
 * manager reads them after it is done (mali_jm_cmd_draw.c,
 * mali_jm_cmd_draw_v7.c). What to patch is in a record the command buffer
 * writes for each draw (struct mali_jm_idraw_rec); the shader's one push
 * constant is the record's address. The job types are written on every
 * run, so a command buffer that runs again is patched again.
 *
 * v9 (IDVS, Malloc Vertex job): the index count, instance count, base
 * vertex offset and index pointer of the job, the vertex shader's
 * first_vertex and base_instance system values, and the offsets of the
 * per-instance vertex attributes (base + firstInstance x stride), as a
 * direct draw has them.
 *
 * v7 (Bifrost: a Vertex job and a Tiler job, varyings in memory): a
 * Vertex job shades a range of vertices, and the tiler reads their outputs
 * back from varying buffers sized by that range times the instances. The
 * range of an indexed draw is [min, max] of its indices, which
 * MALI_META_INDEX_MINMAX finds on the GPU in front of the patch job. The
 * patch job takes the varying buffers from the command buffer's varying
 * arena; a draw that does not fit what is left of it is skipped (both jobs
 * Null) and counted in the arena's header, which the command buffer logs
 * when it is reset. Instanced draws use a power-of-two instance stride
 * (the padded vertex count), so the per-vertex attributes' modulus and the
 * per-instance attributes' divisor are shifts the shader can write.
 */

#include "mali_jm.h"

#include "compiler/nir/nir_builder.h"
#include "genxml/gen_macros.h"

#include "mali_arch.h"
#include "mali_vk.h"

#define REC32(b, rec, field)                                                   \
   nir_load_global(b, 1, 32, nir_iadd_imm(b, rec, offsetof(struct mali_jm_idraw_rec, field)), \
                   .align_mul = 4)
#define REC64(b, rec, field)                                                   \
   nir_load_global(b, 1, 64, nir_iadd_imm(b, rec, offsetof(struct mali_jm_idraw_rec, field)), \
                   .align_mul = 8)
#define REC32_AT(b, rec, field, i)                                             \
   nir_load_global(b, 1, 32,                                                   \
                   nir_iadd_imm(b, rec, offsetof(struct mali_jm_idraw_rec, field) + 4 * (i)), \
                   .align_mul = 4)

static void
store32(nir_builder *b, nir_def *v, nir_def *addr, int64_t off)
{
   nir_store_global(b, v, nir_iadd_imm(b, addr, off), .write_mask = 1, .align_mul = 4);
}

static nir_def *
load32(nir_builder *b, nir_def *addr, int64_t off)
{
   return nir_load_global(b, 1, 32, nir_iadd_imm(b, addr, off), .align_mul = 4);
}

/* Store v (32 bits) at addr + off when addr is not 0. */
static void
store32_if(nir_builder *b, nir_def *v, nir_def *addr)
{
   nir_push_if(b, nir_ine_imm(b, addr, 0));
   store32(b, v, addr, 0);
   nir_pop_if(b, NULL);
}

/* Job Header word 4, Type in bits 1-7 (v7 and v9): `type` when keep, else
 * Null. */
static void
set_job_type(nir_builder *b, nir_def *job, nir_def *keep, unsigned type)
{
   nir_def *w4 = load32(b, job, 16);
   nir_def *t = nir_bcsel(b, keep, nir_imm_int(b, type << 1), nir_imm_int(b, MALI_JOB_TYPE_NULL << 1));
   store32(b, nir_ior(b, nir_iand_imm(b, w4, ~0xfeu), t), job, 16);
}

/* The parameters of the draw: whether it draws, and the command's fields.
 * vertex_offset is firstVertex or vertexOffset, first is firstVertex or
 * firstIndex. */
struct idraw_params {
   nir_def *draws;          /* this draw is within the draw count */
   nir_def *count, *instances, *first, *vertex_offset, *first_instance;
};

static struct idraw_params
load_params(nir_builder *b, nir_def *rec, nir_def *indexed)
{
   struct idraw_params p;

   /* The draw count, for the indirect count forms. */
   nir_def *count_addr = REC64(b, rec, count);
   nir_def *max = REC32(b, rec, max_draws);
   nir_push_if(b, nir_ine_imm(b, count_addr, 0));
   nir_def *n_buf = nir_umin(b, load32(b, count_addr, 0), max);
   nir_pop_if(b, NULL);
   nir_def *n = nir_if_phi(b, n_buf, max);
   p.draws = nir_ult(b, REC32(b, rec, draw_index), n);

   /* The command: VkDrawIndirectCommand (4 words) or
    * VkDrawIndexedIndirectCommand (5); the fifth word is read only for an
    * indexed draw, the buffer may end after the fourth. The command is not
    * read for a draw past the draw count, whose command may not exist. */
   nir_def *cmd = REC64(b, rec, cmd);
   nir_def *zero = nir_imm_int(b, 0);
   nir_push_if(b, p.draws);
   nir_def *c[4];
   for (unsigned i = 0; i < 4; i++)
      c[i] = load32(b, cmd, 4 * i);
   nir_push_if(b, indexed);
   nir_def *c4 = load32(b, cmd, 16);
   nir_pop_if(b, NULL);
   nir_def *c4p = nir_if_phi(b, c4, c[3]);
   nir_pop_if(b, NULL);
   for (unsigned i = 0; i < 4; i++)
      c[i] = nir_if_phi(b, c[i], zero);
   c4p = nir_if_phi(b, c4p, zero);

   p.count = c[0];
   p.instances = c[1];
   p.first = c[2];
   p.vertex_offset = nir_bcsel(b, indexed, c[3], c[2]);
   p.first_instance = c4p;   /* c[3] for a non-indexed draw (the phi above) */
   p.draws = nir_iand(b, p.draws,
                      nir_iand(b, nir_ine_imm(b, p.count, 0), nir_ine_imm(b, p.instances, 0)));
   return p;
}

/* attr_base[a] + first_instance x attr_stride[a] into the per-instance
 * attributes' offsets: `size` bytes per descriptor, the offset at byte
 * `off` of it. */
static void
patch_inst_attribs(nir_builder *b, nir_def *rec, nir_def *first_instance, unsigned size,
                   unsigned off)
{
   nir_def *mask = REC32(b, rec, inst_attr_mask);
   nir_push_if(b, nir_ine_imm(b, mask, 0));
   nir_def *attribs = REC64(b, rec, attribs);
   for (unsigned a = 0; a < MALI_IDRAW_MAX_ATTRIBS; a++) {
      nir_push_if(b, nir_ine_imm(b, nir_iand_imm(b, mask, 1u << a), 0));
      nir_def *v = nir_iadd(b, REC32_AT(b, rec, attr_base, a),
                            nir_imul(b, first_instance, REC32_AT(b, rec, attr_stride, a)));
      store32(b, v, attribs, a * size + off);
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);
}

static nir_builder
init_shader(struct mali_device *dev, const char *name, unsigned wg)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, MALI_PER_ARCH(shader_nir_options)(dev, MESA_SHADER_COMPUTE), "%s",
      name);
   b.shader->info.workgroup_size[0] = wg;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   return b;
}

#if PAN_ARCH >= 9
/* Malloc Vertex job words (v9.xml): Primitive at 32 (base vertex offset
 * +4, index count +12), Instance Count at 48, Indices at 120. */
#define MV_PRIM 32
#define MV_INSTANCES 48
#define MV_INDICES 120

nir_shader *
MALI_PER_ARCH(build_indirect_draw_nir)(struct mali_device *dev)
{
   nir_builder b = init_shader(dev, "mali_meta_indirect_draw", 1);
   nir_def *rec = nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, 0), .base = 0, .range = 8);
   nir_def *indexed = nir_ine_imm(&b, REC32(&b, rec, indexed), 0);
   const struct idraw_params p = load_params(&b, rec, indexed);

   nir_def *job = REC64(&b, rec, job);
   set_job_type(&b, job, p.draws, MALI_JOB_TYPE_MALLOC_VERTEX);

   nir_push_if(&b, p.draws);
   {
      store32(&b, p.vertex_offset, job, MV_PRIM + 4);
      store32(&b, p.count, job, MV_PRIM + 12);
      store32(&b, p.instances, job, MV_INSTANCES);
      nir_push_if(&b, indexed);
      {
         nir_def *off = nir_ishl(&b, nir_u2u64(&b, p.first), REC32(&b, rec, index_shift));
         nir_store_global(&b, nir_iadd(&b, REC64(&b, rec, indices), off),
                          nir_iadd_imm(&b, job, MV_INDICES), .write_mask = 1, .align_mul = 8);
      }
      nir_pop_if(&b, NULL);
      store32_if(&b, p.vertex_offset, REC64(&b, rec, first_vertex));
      store32_if(&b, p.first_instance, REC64(&b, rec, base_instance));
      /* Attribute descriptors are 32 bytes, the offset in word 2. */
      patch_inst_attribs(&b, rec, p.first_instance, MALI_DESCRIPTOR_SIZE, 8);
   }
   nir_pop_if(&b, NULL);
   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   return b.shader;
}
#else
/* ceil(log2(v)) for v >= 1. */
static nir_def *
logbase2_ceil(nir_builder *b, nir_def *v)
{
   nir_def *msb = nir_iadd_imm(b, nir_ufind_msb(b, nir_iadd_imm(b, v, -1)), 1);
   return nir_bcsel(b, nir_uge_imm(b, v, 2), msb, nir_imm_int(b, 0));
}

/* Read-modify-write of bits [shift, shift + 8) of the word at addr + off. */
static void
store_byte(nir_builder *b, nir_def *v, nir_def *addr, int64_t off, unsigned shift)
{
   nir_def *w = load32(b, addr, off);
   w = nir_ior(b, nir_iand_imm(b, w, ~(0xffu << shift)), nir_ishl_imm(b, v, shift));
   store32(b, w, addr, off);
}

/* Job offsets (v7.xml): Invocation at 32 in both jobs; the Vertex job's
 * Draw at 64; the Tiler job's Primitive at 40, Primitive Size at 64,
 * Draw at 128. Draw word 0 has Instance Size in bits 16-23 and Instance
 * Primitive Size in bits 24-31, word 1 Offset start, words 4-5 Position.
 * Primitive word 1 is Base vertex offset, word 3 Index count minus one,
 * words 4-5 Indices. */
#define J_INVOCATION 32
#define VJ_DRAW 64
#define TJ_PRIM 40
#define TJ_PSIZE 64
#define TJ_DRAW 128
/* Invocation word 1: Workgroups Z shift in bits 22-27. */
#define INV1_WGZ_SHIFT 22
/* The most vertex-instances (bits of the Invocation's packed counts) an
 * indirect draw shades. */
#define MAX_COUNT_BITS 26

/*
 * Bytes off the arena: header->used grows by `size` when it stays within
 * `limit`; the old value, and whether it did. A compare-and-swap loop, so
 * the count never passes the limit: draws that do not fit leave it as it
 * was.
 */
static nir_def *
arena_take(nir_builder *b, nir_def *arena, nir_def *size, nir_def *limit, nir_def **ok_out)
{
   nir_variable *cur_v = nir_local_variable_create(b->impl, glsl_uint_type(), "cur");
   nir_variable *ok_v = nir_local_variable_create(b->impl, glsl_bool_type(), "ok");
   nir_store_var(b, cur_v, nir_imm_int(b, 0), 1);
   nir_store_var(b, ok_v, nir_imm_false(b), 1);
   nir_push_loop(b);
   {
      nir_def *cur = nir_load_var(b, cur_v);
      nir_def *end = nir_iadd(b, cur, size);
      nir_break_if(b, nir_ior(b, nir_ult(b, limit, end), nir_ult(b, end, cur)));
      nir_def *prev = nir_global_atomic_swap(b, 32, arena, cur, end,
                                             .atomic_op = nir_atomic_op_cmpxchg);
      nir_push_if(b, nir_ieq(b, prev, cur));
      {
         nir_store_var(b, ok_v, nir_imm_true(b), 1);
         nir_jump(b, nir_jump_break);
      }
      nir_pop_if(b, NULL);
      nir_store_var(b, cur_v, prev, 1);
   }
   nir_pop_loop(b, NULL);
   *ok_out = nir_load_var(b, ok_v);
   return nir_load_var(b, cur_v);
}

nir_shader *
MALI_PER_ARCH(build_indirect_draw_nir)(struct mali_device *dev)
{
   nir_builder b = init_shader(dev, "mali_meta_indirect_draw", 1);
   nir_def *rec = nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, 0), .base = 0, .range = 8);
   nir_def *indexed = nir_ine_imm(&b, REC32(&b, rec, indexed), 0);
   struct idraw_params p = load_params(&b, rec, indexed);

   /* The vertex range: the draw's vertices, or [min, max] of its indices
    * (an empty range when there were none to read, or only restart
    * indices). raw is the first shaded vertex (the Draws' offset start),
    * base the base vertex offset that maps an index onto it. */
   nir_def *mm0 = nir_imm_ivec2(&b, 0, 0);
   nir_push_if(&b, indexed);
   nir_def *mm = nir_load_global(&b, 2, 32, REC64(&b, rec, minmax), .align_mul = 8);
   nir_pop_if(&b, NULL);
   mm = nir_if_phi(&b, mm, mm0);
   nir_def *mn = nir_channel(&b, mm, 0), *mx = nir_channel(&b, mm, 1);
   p.draws = nir_iand(&b, p.draws, nir_ior(&b, nir_inot(&b, indexed), nir_uge(&b, mx, mn)));
   nir_def *range = nir_bcsel(&b, indexed, nir_iadd_imm(&b, nir_isub(&b, mx, mn), 1), p.count);
   nir_def *raw = nir_bcsel(&b, indexed, nir_iadd(&b, mn, p.vertex_offset), p.first);
   nir_def *base = nir_bcsel(&b, indexed, nir_ineg(&b, mn), nir_imm_int(&b, 0));

   /* Instance stride P = 2^lp >= range when instanced, the range itself
    * for one instance. range is 0 here only for [0, 2^32 - 1]. */
   nir_def *lp = logbase2_ceil(&b, range);
   nir_def *li = logbase2_ceil(&b, p.instances);
   nir_def *instanced = nir_uge_imm(&b, p.instances, 2);
   nir_def *per_instance = nir_bcsel(&b, instanced, nir_ishl(&b, nir_imm_int(&b, 1), lp), range);
   nir_def *fits = nir_iand(&b, nir_ine_imm(&b, range, 0),
                            nir_ult_imm(&b, nir_iadd(&b, lp, li), MAX_COUNT_BITS + 1));
   nir_def *verts = nir_imul(&b, per_instance, p.instances);

   /* The varying buffers' sizes, each a multiple of 64 bytes, and where
    * each starts in the draw's share of the arena. */
   nir_def *size[MALI_V7_VARY_BUF_COUNT], *start[MALI_V7_VARY_BUF_COUNT];
   nir_def *total = nir_imm_int(&b, 0);
   for (unsigned i = 0; i < MALI_V7_VARY_BUF_COUNT; i++) {
      nir_def *stride = REC32_AT(&b, rec, vary_stride, i);
      fits = nir_iand(&b, fits, nir_uge(&b, REC32_AT(&b, rec, vary_max, i), verts));
      size[i] = nir_imul(&b, verts, stride);
      start[i] = total;
      total = nir_iadd(&b, total, nir_iand_imm(&b, nir_iadd_imm(&b, size[i], 63), ~63u));
   }

   nir_def *arena = REC64(&b, rec, arena);
   nir_def *no = nir_imm_false(&b), *zero = nir_imm_int(&b, 0);
   nir_push_if(&b, nir_iand(&b, p.draws, fits));
   nir_def *ok;
   nir_def *at = arena_take(&b, arena, total, REC32(&b, rec, arena_size), &ok);
   nir_pop_if(&b, NULL);
   ok = nir_if_phi(&b, ok, no);
   at = nir_if_phi(&b, at, zero);

   /* A draw that draws but does not fit is counted. */
   nir_push_if(&b, nir_iand(&b, p.draws, nir_inot(&b, ok)));
   nir_global_atomic(&b, 32,
                     nir_iadd_imm(&b, arena, offsetof(struct mali_jm_idraw_arena_hdr, dropped)),
                     nir_imm_int(&b, 1), .atomic_op = nir_atomic_op_iadd);
   nir_pop_if(&b, NULL);

   nir_def *vjob = REC64(&b, rec, job);
   nir_def *tjob = REC64(&b, rec, tiler_job);
   nir_def *tiling = nir_ine_imm(&b, tjob, 0);
   set_job_type(&b, vjob, ok, MALI_JOB_TYPE_VERTEX);
   nir_push_if(&b, tiling);
   set_job_type(&b, tjob, ok, MALI_JOB_TYPE_TILER);
   nir_pop_if(&b, NULL);

   nir_push_if(&b, ok);
   {
      nir_def *data = nir_iadd_imm(&b, arena, MALI_IDRAW_ARENA_HDR);
      nir_def *addr[MALI_V7_VARY_BUF_COUNT];
      nir_def *vary_bufs = REC64(&b, rec, vary_bufs);
      for (unsigned i = 0; i < MALI_V7_VARY_BUF_COUNT; i++) {
         addr[i] = nir_iadd(&b, data, nir_u2u64(&b, nir_iadd(&b, at, start[i])));
         /* Attribute Buffer: Type in bits 0-5 and the pointer above them in
          * the first 64 bits, Size in word 3. A buffer of stride 0 keeps
          * its null pointer and size. */
         nir_push_if(&b, nir_ine_imm(&b, REC32_AT(&b, rec, vary_stride, i), 0));
         {
            nir_def *d = nir_iadd_imm(&b, vary_bufs, i * pan_size(ATTRIBUTE_BUFFER));
            nir_def *w01 = nir_load_global(&b, 1, 64, d, .align_mul = 8);
            nir_store_global(&b, nir_ior(&b, nir_iand_imm(&b, w01, 63), addr[i]), d,
                             .write_mask = 1, .align_mul = 8);
            store32(&b, size[i], d, 12);
         }
         nir_pop_if(&b, NULL);
      }

      /* The Invocation of both jobs: (1, range, instances) workgroups of
       * one invocation, with the graphics quirks (mali_v7_pack_invocation):
       * the range's bits, then the instances' from bit lp, Workgroups Z
       * shift lp, or 32 for one instance. */
      nir_def *inv0 = nir_ior(&b, nir_iadd_imm(&b, range, -1),
                              nir_ishl(&b, nir_iadd_imm(&b, p.instances, -1), lp));
      nir_def *zs = nir_bcsel(&b, instanced, lp, nir_imm_int(&b, 32));
      nir_def *inv = nir_vec2(&b, inv0,
                              nir_ior(&b, REC32(&b, rec, inv1), nir_ishl_imm(&b, zs, INV1_WGZ_SHIFT)));
      nir_store_global(&b, inv, nir_iadd_imm(&b, vjob, J_INVOCATION), .write_mask = 0x3,
                       .align_mul = 8);

      /* Instance Size: P as a padded count (a power of two packs as its
       * shift), or 1 (0) for one instance. */
      nir_def *isz = nir_bcsel(&b, instanced, lp, nir_imm_int(&b, 0));
      store_byte(&b, isz, vjob, VJ_DRAW, 16);
      store32(&b, raw, vjob, VJ_DRAW + 4);

      nir_push_if(&b, tiling);
      {
         nir_store_global(&b, inv, nir_iadd_imm(&b, tjob, J_INVOCATION), .write_mask = 0x3,
                          .align_mul = 8);
         /* Instance Primitive Size: a power of two at least the
          * primitives of one instance. */
         nir_def *prims =
            nir_udiv(&b, nir_iadd(&b, per_instance, nir_iadd_imm(&b, REC32(&b, rec, prim_verts), -1)),
                     REC32(&b, rec, prim_verts));
         nir_def *ips = logbase2_ceil(&b, nir_umax(&b, prims, nir_imm_int(&b, 1)));
         nir_def *w0 = load32(&b, tjob, TJ_DRAW);
         w0 = nir_ior(&b, nir_iand_imm(&b, w0, 0xffffu),
                      nir_ior(&b, nir_ishl_imm(&b, isz, 16), nir_ishl_imm(&b, ips, 24)));
         store32(&b, w0, tjob, TJ_DRAW);
         store32(&b, raw, tjob, TJ_DRAW + 4);
         nir_store_global(&b, addr[MALI_V7_VARY_BUF_POSITION], nir_iadd_imm(&b, tjob, TJ_DRAW + 16),
                          .write_mask = 1, .align_mul = 8);
         /* The Primitive: base vertex offset (indexed), index count. */
         store32(&b, base, tjob, TJ_PRIM + 4);
         store32(&b, nir_iadd_imm(&b, p.count, -1), tjob, TJ_PRIM + 12);
         nir_push_if(&b, indexed);
         {
            nir_def *off = nir_ishl(&b, nir_u2u64(&b, p.first), REC32(&b, rec, index_shift));
            nir_store_global(&b, nir_iadd(&b, REC64(&b, rec, indices), off),
                             nir_iadd_imm(&b, tjob, TJ_PRIM + 16), .write_mask = 1,
                             .align_mul = 8);
         }
         nir_pop_if(&b, NULL);
         nir_push_if(&b, nir_ine_imm(&b, REC32(&b, rec, writes_psiz), 0));
         nir_store_global(&b, addr[MALI_V7_VARY_BUF_PSIZ], nir_iadd_imm(&b, tjob, TJ_PSIZE),
                          .write_mask = 1, .align_mul = 8);
         nir_pop_if(&b, NULL);
      }
      nir_pop_if(&b, NULL);

      /* The vertex buffers: a per-vertex binding is 1D Modulus by P, a
       * per-instance one 1D POT Divisor by P x its divisor; both fields
       * are the byte at bits 24-31 of the Attribute Buffer's word 1. */
      nir_def *mod = REC32(&b, rec, mod_mask), *pot = REC32(&b, rec, pot_mask);
      nir_push_if(&b, nir_ine_imm(&b, nir_ior(&b, mod, pot), 0));
      {
         nir_def *bufs = REC64(&b, rec, attrib_bufs);
         for (unsigned i = 0; i < MALI_IDRAW_MAX_ATTRIBS; i++) {
            const unsigned off = 2 * i * pan_size(ATTRIBUTE_BUFFER) + 4;
            nir_push_if(&b, nir_ine_imm(&b, nir_iand_imm(&b, mod, 1u << i), 0));
            store_byte(&b, lp, bufs, off, 24);
            nir_pop_if(&b, NULL);
            nir_push_if(&b, nir_ine_imm(&b, nir_iand_imm(&b, pot, 1u << i), 0));
            store_byte(&b, nir_iadd(&b, lp, REC32_AT(&b, rec, pot_shift, i)), bufs, off, 24);
            nir_pop_if(&b, NULL);
         }
      }
      nir_pop_if(&b, NULL);
      /* Attribute descriptors are 8 bytes, the offset in word 1. */
      patch_inst_attribs(&b, rec, p.first_instance, pan_size(ATTRIBUTE), 4);

      store32_if(&b, p.vertex_offset, REC64(&b, rec, first_vertex));
      store32_if(&b, p.first_instance, REC64(&b, rec, base_instance));
      store32_if(&b, raw, REC64(&b, rec, raw_vertex_offset));
   }
   nir_pop_if(&b, NULL);

   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   return b.shader;
}

#define MPUSH(b, field, bits)                                                  \
   nir_load_push_constant(b, 1, bits, nir_imm_int(b, offsetof(struct mali_meta_minmax_push, field)), \
                          .base = 0, .range = sizeof(struct mali_meta_minmax_push))

/*
 * The smallest and largest index of a VkDrawIndexedIndirectCommand
 * (panvk's panlib_draw_index_minmax_search_helper): each invocation reads
 * per_thread indices, from firstIndex + its id x per_thread, the count
 * clamped to the bound buffer as the CPU scan of direct draws clamps it,
 * and folds its result in with atomic min and max. The result words start
 * as {~0, 0} (the command buffer restores them before every run), so a
 * draw with no index in the buffer, or only restart indices, ends with
 * min > max.
 */
nir_shader *
MALI_PER_ARCH(build_index_minmax_nir)(struct mali_device *dev)
{
   nir_builder b = init_shader(dev, "mali_meta_index_minmax", 64);

   nir_def *cmd = MPUSH(&b, cmd, 64);
   nir_def *elems = MPUSH(&b, elems, 32);
   nir_def *per = MPUSH(&b, per_thread, 32);
   nir_def *shift = MPUSH(&b, index_shift, 32);
   nir_def *count = load32(&b, cmd, 0);
   nir_def *first = load32(&b, cmd, 8);
   nir_def *avail = nir_bcsel(&b, nir_ult(&b, first, elems), nir_isub(&b, elems, first),
                              nir_imm_int(&b, 0));
   count = nir_umin(&b, count, avail);

   nir_def *id = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *begin = nir_imul(&b, id, per);
   nir_push_if(&b, nir_ult(&b, begin, count));
   {
      nir_def *end = nir_iadd(&b, begin, nir_umin(&b, per, nir_isub(&b, count, begin)));
      /* Index bits (8, 16 or 32), the restart value all ones of them. */
      nir_def *bits = nir_ishl(&b, nir_imm_int(&b, 8), shift);
      nir_def *all = nir_bcsel(&b, nir_ieq_imm(&b, bits, 32), nir_imm_int(&b, ~0),
                               nir_iadd_imm(&b, nir_ishl(&b, nir_imm_int(&b, 1), bits), -1));
      nir_def *restart = nir_ine_imm(&b, MPUSH(&b, restart, 32), 0);
      nir_def *indices = MPUSH(&b, indices, 64);

      nir_variable *i_v = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
      nir_variable *mn_v = nir_local_variable_create(b.impl, glsl_uint_type(), "mn");
      nir_variable *mx_v = nir_local_variable_create(b.impl, glsl_uint_type(), "mx");
      nir_store_var(&b, i_v, begin, 1);
      nir_store_var(&b, mn_v, nir_imm_int(&b, ~0), 1);
      nir_store_var(&b, mx_v, nir_imm_int(&b, 0), 1);
      nir_push_loop(&b);
      {
         nir_def *i = nir_load_var(&b, i_v);
         nir_break_if(&b, nir_uge(&b, i, end));
         /* The index, from the aligned 32-bit word that holds it: index
          * buffer offsets are a multiple of the index size. */
         nir_def *byte = nir_iadd(&b, indices,
                                  nir_ishl(&b, nir_u2u64(&b, nir_iadd(&b, first, i)), shift));
         nir_def *word = nir_load_global(&b, 1, 32, nir_iand_imm(&b, byte, ~3ull), .align_mul = 4);
         nir_def *sh = nir_ishl_imm(&b, nir_u2u32(&b, nir_iand_imm(&b, byte, 3)), 3);
         nir_def *v = nir_iand(&b, nir_ushr(&b, word, sh), all);
         nir_def *skip = nir_iand(&b, restart, nir_ieq(&b, v, all));
         nir_push_if(&b, nir_inot(&b, skip));
         {
            nir_store_var(&b, mn_v, nir_umin(&b, nir_load_var(&b, mn_v), v), 1);
            nir_store_var(&b, mx_v, nir_umax(&b, nir_load_var(&b, mx_v), v), 1);
         }
         nir_pop_if(&b, NULL);
         nir_store_var(&b, i_v, nir_iadd_imm(&b, i, 1), 1);
      }
      nir_pop_loop(&b, NULL);

      nir_def *mn = nir_load_var(&b, mn_v), *mx = nir_load_var(&b, mx_v);
      nir_push_if(&b, nir_uge(&b, mx, mn));
      {
         nir_def *res = MPUSH(&b, result, 64);
         nir_global_atomic(&b, 32, res, mn, .atomic_op = nir_atomic_op_umin);
         nir_global_atomic(&b, 32, nir_iadd_imm(&b, res, 4), mx,
                           .atomic_op = nir_atomic_op_umax);
      }
      nir_pop_if(&b, NULL);
   }
   nir_pop_if(&b, NULL);

   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   return b.shader;
}
#endif
