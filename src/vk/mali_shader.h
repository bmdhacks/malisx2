/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Compiled shaders: one pipeline stage taken from NIR through panvk-style
 * lowering and kraid to v9 or v11 machine code, or Mesa's Bifrost backend
 * to v7 machine code (mali_shader.c is built per arch), uploaded to the
 * device's executable pool, with its Shader Program Descriptors (SPDs)
 * built (on v7, a vertex or compute shader's Renderer State Descriptor).
 *
 * A mali_shader is a pipeline cache object: pipelines share them through
 * the application's VkPipelineCache or the device's own in-memory cache,
 * and they are serialized into vkGetPipelineCacheData.
 *
 * The resource model is panvk's for v9+ (the same on v9 and v11), because
 * the shaders come from Mesa's compiler and that is the model its lowering
 * produces:
 *
 *  - FAU (fast access uniforms, 64 words of 64 bits) hold, in this order,
 *    the system values the shader uses and then the push-constant words it
 *    uses, both compacted (fau below says which);
 *  - resource table 0 is the driver's (vertex attributes / varyings, a
 *    dummy sampler, dynamic buffers), table N + 1 is descriptor set N;
 *  - graphics and compute system values have the layouts of
 *    struct mali_graphics_sysvals / mali_compute_sysvals.
 *
 * v7 keeps the FAU block and the system values (with the descriptor set
 * addresses in v7_desc) but has no resource tables: panvk's PAN_ARCH < 9
 * per-type tables take their place (enum mali_v7_desc_table below).
 */

#ifndef MALI_SHADER_H
#define MALI_SHADER_H

#include <stdint.h>
#include <string.h>

#include "vk_pipeline_cache.h"
#include "vk_pipeline_layout.h"

#include "compiler/nir/nir.h"
#include "panfrost/compiler/pan_compiler.h"
#include "util/bitset.h"
#include "util/mesa-blake3.h"

#include "mali_arch.h"
#include "mali_bo_pool.h"
#include "mali_descriptor_set_layout.h"

struct mali_device;
struct vk_graphics_pipeline_state;
struct vk_pipeline_robustness_state;

#define MALI_MAX_RTS 8
#define MALI_MAX_VS_ATTRIBS 16
#define MALI_MAX_PUSH_CONSTANTS_SIZE 256
#define MALI_FAU_WORD_SIZE 8
#define MALI_FAU_WORD_COUNT 64

/* Input attachment targets in the iam system values. */
#define MALI_COLOR_ATTACHMENT(x) (x)
#define MALI_ZS_ATTACHMENT 255
/* One per colour attachment, depth, stencil, and one for an input
 * attachment without an InputAttachmentIndex. */
#define MALI_INPUT_ATTACHMENT_MAP_SIZE 11

#define mali_aligned_u64 __attribute__((aligned(8))) uint64_t

/* System values common to graphics and compute, at the same offset in both
 * (so a shader compiles without knowing where it runs). */
struct mali_common_sysvals_inner {
   /* Address of the sysval + push-constant block, for indirect loads. */
   mali_aligned_u64 push_uniforms;
   mali_aligned_u64 printf_buffer_address;
   mali_aligned_u64 constant_data;
} __attribute__((aligned(8)));

struct mali_input_attachment_info {
   uint32_t target;      /* MALI_COLOR_ATTACHMENT(i), MALI_ZS_ATTACHMENT or ~0 */
   /* Internal Conversion word for colour targets, and on v7 for the depth
    * target too (mali_cmd_state.h mali_gfx_fill_sysvals). */
   uint32_t conversion;
};

struct mali_graphics_sysvals {
   /* Blend constants first: blend shaders read them from FAU 0-1 whatever
    * the fragment shader's packing. */
   struct {
      float constants[4];
   } blend;
   struct mali_common_sysvals_inner common;
   /* Scale and offset of one axis share a 64-bit FAU word: the vertex
    * shader's viewport transform FMA(pos, scale, offset) may read only one
    * FAU word, so with the two in separate words kraid copied one into a
    * register first (a MOV per axis per vertex). */
   struct {
      float scale, offset;
   } viewport[3];
   struct {
      int32_t first_vertex;
      int32_t base_instance;
      uint32_t noperspective_varyings;
      /* v7: what the vertex shader adds to the raw vertex id to get
       * gl_VertexIndex (pan_nir_lower_vertex_id): firstVertex, or the
       * smallest index plus vertexOffset for an indexed draw, as in the
       * Draw descriptor's offset_start. It fills padding, so the other
       * words keep their offsets. */
      int32_t raw_vertex_offset;
   } vs;
   union {
      struct {
         mali_aligned_u64 blend_descs[MALI_MAX_RTS];
      } fs;
      /* v7: Bifrost reads the Blend descriptors from its blend FAU slots,
       * so these words hold the addresses of the bound descriptor sets,
       * for storage buffers and texture and image queries
       * (mali_nir_lower_descriptors_v7.c), and at MALI_V7_SETS_DYN_SSBOS
       * the stage's dynamic storage buffer table (mali_jm_desc_v7.c). */
      struct {
         mali_aligned_u64 sets[MALI_MAX_SETS + 1];
      } v7_desc;
   };
   struct mali_input_attachment_info iam[MALI_INPUT_ATTACHMENT_MAP_SIZE];
} __attribute__((aligned(8)));

static_assert(sizeof(((struct mali_graphics_sysvals *)NULL)->v7_desc) ==
                 sizeof(((struct mali_graphics_sysvals *)NULL)->fs),
              "the v7 set addresses take exactly the blend descriptor words");

/* The v7_desc.sets index of the dynamic storage buffer table, behind the
 * user sets (panvk_v7's PANVK_DESC_TABLE_*_DYN_SSBOS). Each stage has its
 * own sysvals, so one index serves every stage. It indexes the set
 * addresses, not the per-type tables of enum mali_v7_desc_table. */
#define MALI_V7_SETS_DYN_SSBOS MALI_MAX_SETS

/* The push-constant base that marks system values until the FAU packing
 * gives everything its final offset (mali_shader.c); the v7 descriptor
 * lowering uses it too. */
#define MALI_SYSVALS_PUSH_CONST_BASE MALI_MAX_PUSH_CONSTANTS_SIZE

/*
 * The arch this header is compiled for: PAN_ARCH in the per-arch files,
 * MALI_PAN_ARCH (the CSF arch) in the files built once. Only a v7 build
 * adds the compute v7_desc words, so the v9 and v11 struct keeps its
 * layout everywhere.
 */
#if defined(PAN_ARCH)
#define MALI_SHADER_ARCH PAN_ARCH
#elif defined(MALI_PAN_ARCH)
#define MALI_SHADER_ARCH MALI_PAN_ARCH
#else
#error "mali_shader.h needs PAN_ARCH or MALI_PAN_ARCH"
#endif
#define MALI_SHADER_V7 (MALI_SHADER_ARCH < 9)

struct mali_compute_sysvals {
   struct {
      uint32_t x, y, z;
   } base;
   uint32_t _pad;
   struct mali_common_sysvals_inner common;
   struct {
      uint32_t x, y, z;
   } num_work_groups;
   struct {
      uint32_t x, y, z;
   } local_group_size;
#if MALI_SHADER_V7
   /* v7: as the graphics v7_desc, the bound sets' addresses and the
    * dispatch's dynamic storage buffer table (mali_jm_desc_v7.c). */
   struct {
      mali_aligned_u64 sets[MALI_MAX_SETS + 1];
   } v7_desc;
#endif
} __attribute__((aligned(8)));

#define MALI_MAX_SYSVAL_FAUS                                                   \
   (MAX2(sizeof(struct mali_graphics_sysvals),                                 \
         sizeof(struct mali_compute_sysvals)) / MALI_FAU_WORD_SIZE)
#define MALI_MAX_PUSH_CONST_FAUS (MALI_MAX_PUSH_CONSTANTS_SIZE / MALI_FAU_WORD_SIZE)

/*
 * FAU word `w` of a sysval block (zero past its end). The blocks are
 * structs of 32-bit fields: reading them through a uint64_t pointer breaks
 * C's aliasing rules, and the NDK's clang did move such reads ahead of the
 * stores that fill the struct, so the words are copied out with memcpy.
 */
static inline uint64_t
mali_sysval_word(const void *sysvals, size_t size, unsigned w)
{
   uint64_t v = 0;
   if ((w + 1) * MALI_FAU_WORD_SIZE <= size)
      memcpy(&v, (const uint8_t *)sysvals + w * MALI_FAU_WORD_SIZE, sizeof(v));
   return v;
}

/* Most 64-bit words of dynamic uniform buffers one shader keeps in FAU,
 * and on v11 the part of a buffer they may come from (ARMSX2's constant
 * buffers are smaller). v7 has its own byte limit (mali_shader.c
 * V7_UBO_PUSH_MAX_BYTES). */
#define MALI_MAX_UBO_PUSH_FAUS 32
#define MALI_UBO_PUSH_MAX_BYTES 1024

/* The uniform-buffer copy in a draw's vertex/tiler stream: scratch
 * registers (mali_cs.h: 66 + n) for the source and FAU addresses and the
 * data, and the most instruction words a shader's prebuilt copy may have. */
#define MALI_UBO_COPY_SRC      66
#define MALI_UBO_COPY_DST      68
#define MALI_UBO_COPY_DATA     70
#define MALI_UBO_COPY_DATA_REGS 12
#define MALI_UBO_COPY_MAX_INS  40

/* One pushed 64-bit word: bytes [offset, offset + 8) of the dynamic buffer
 * in driver-table copy `dyn` (index into mali_shader_desc_info::dyn_bufs). */
struct mali_ubo_push_word {
   uint16_t dyn;
   uint16_t offset;
};

/* Constant 64-bit words the driver puts in a shader's FAU block (texture
 * handles, mali_shader.c "Constant texture handles"). */
#define MALI_MAX_SHADER_FAU_CONSTS 8

/*
 * Which FAU words the shader reads. The command buffer builds the
 * shader's FAU block as: the used sysval words in order, then the used
 * push-constant words in order, then the const_count words of consts[],
 * then the pushed uniform-buffer words (ubo_push, from FAU word
 * ubo_push_start on): total_count words of 64 bits, with kraid's promoted
 * constants after them on v9 and v11.
 *
 * Pushed uniform-buffer words come from dynamic uniform buffers. On v11
 * they are pushed in fragment shaders, and the draw copies them into the
 * FAU block with command-stream loads and stores (mali_cmd_draw.c
 * emit_ubo_push). On v7 they are pushed in vertex and fragment shaders,
 * and a Compute job ahead of the job that reads the block copies them
 * (mali_jm_desc_v7.c mali_v7_ubo_copy). v9 shaders load from the buffer
 * (ubo_push_count 0).
 *
 * The block's GPU address and total_count reach the hardware as one FAU
 * word on v11 (address | count << 56, the FAU register), as the Shader
 * Environment's separate FAU pointer and FAU count fields on v9, where a
 * count of 0 drops the uniforms, and on v7 as the job's Draw push
 * uniforms pointer and the Renderer State's preload uniform count.
 */
struct mali_shader_fau {
   BITSET_DECLARE(used_sysvals, MALI_MAX_SYSVAL_FAUS);
   BITSET_DECLARE(used_push_consts, MALI_MAX_PUSH_CONST_FAUS);
   uint32_t sysval_count;
   uint32_t total_count;
   uint32_t const_count;
   uint64_t consts[MALI_MAX_SHADER_FAU_CONSTS];
   uint32_t ubo_push_start;
   uint32_t ubo_push_count;
   struct mali_ubo_push_word ubo_push[MALI_MAX_UBO_PUSH_FAUS];
};

/* Writes the shader's constant FAU words at out[0 .. const_count). */
static inline unsigned
mali_shader_fau_consts(const struct mali_shader_fau *fau, uint64_t *out)
{
   for (unsigned i = 0; i < fau->const_count; i++)
      out[i] = fau->consts[i];
   return fau->const_count;
}

/*
 * What the shader reads from descriptor sets, and the layout of the
 * driver's resource table (table 0) it was compiled against:
 *
 *   [0, prefix)            vertex shader: the 16 vertex attribute
 *                          descriptors; fragment shader: one varying
 *                          attribute descriptor per input slot (written
 *                          only if needs_varying_descs); compute: none
 *   prefix                 a dummy sampler (texel fetches name one)
 *   prefix + 1 + i         copy of dynamic buffer map[i], with the dynamic
 *                          offset applied, for i < dyn_bufs.count
 *
 * map[i] is MALI_COPY_DESC_HANDLE(set, dynamic buffer index in the set):
 * the index in the low 28 bits, the set above them.
 */
#define MALI_COPY_DESC_HANDLE_SET_SHIFT        28
#define MALI_COPY_DESC_HANDLE(set, idx)        (((set) << MALI_COPY_DESC_HANDLE_SET_SHIFT) | (idx))
#define MALI_COPY_DESC_HANDLE_SET(handle)      ((handle) >> MALI_COPY_DESC_HANDLE_SET_SHIFT)
#define MALI_COPY_DESC_HANDLE_INDEX(handle)                                    \
   ((handle) & BITFIELD_MASK(MALI_COPY_DESC_HANDLE_SET_SHIFT))

struct mali_shader_desc_info {
   uint32_t used_set_mask;
   uint32_t driver_table_prefix;
   /* v7: the dynamic uniform buffers first (v7.dyn_ubo_count of them),
    * then the dynamic storage buffers. */
   struct {
      uint32_t map[MALI_MAX_DYNAMIC_BUFFERS];
      uint32_t count;
   } dyn_bufs;
   /* Fragment shaders: the shader loads some varyings with LD_VAR, which
    * reads the attribute descriptors at the start of table 0. */
   bool needs_varying_descs;
};

/*
 * The Bifrost (v7) descriptor model, panvk_v7's
 * (panvk_vX_nir_lower_descriptors.c, PANVK_BIFROST_DESC_TABLE_*). There
 * are no resource tables: each stage of a draw or dispatch gets per-type
 * tables of Uniform Buffer descriptors (8 bytes), image Attribute Buffer
 * pairs with their Attributes (storage images and texel buffers), and
 * Texture and Sampler descriptors (32 bytes), and the shader's indices are
 * positions in them. The lowering (mali_nir_lower_descriptors_v7.c) picks
 * the positions and records in `map` which set slot fills each entry;
 * mali_jm_desc_v7.c builds the tables, copying the slots with a small job
 * in front of the job that reads them. Storage buffers are not copied:
 * the shader reads their address and size, as it reads texture and image
 * sizes, from the set through v7_desc.sets.
 */
enum mali_v7_desc_table {
   MALI_V7_DESC_TABLE_UBO = 0,     /* 8-byte entries, dynamic ones last */
   MALI_V7_DESC_TABLE_IMG,         /* storage images and texel buffers */
   MALI_V7_DESC_TABLE_TEXTURE,
   MALI_V7_DESC_TABLE_SAMPLER,
   MALI_V7_DESC_TABLE_COUNT,
};

/*
 * The image table's size limit. An Attribute names its Attribute Buffer
 * with a 9-bit index (v7.xml), so a stage sees 512 records, and each image
 * takes a pair: from record 0 in a fragment or compute shader, behind the
 * pairs of up to 16 vertex buffers in a vertex shader
 * (mali_jm_cmd_draw_v7.c). That leaves 512 / 2 - 16 = 240 images. A larger
 * index would spill into the Attribute's Offset enable and Format bits
 * unnoticed, so the per-stage limits (mali_physical_device.c) keep the
 * table within it: maxPerStageDescriptorStorageImages counts storage
 * images and storage texel buffers, maxPerStageDescriptorSampledImages
 * uniform texel buffers. panvk_v7's limits (32 + 256) overflow the field.
 */
#define MALI_V7_ATTRIB_BUF_RECORDS 512
#define MALI_V7_MAX_STAGE_IMG_ENTRIES (MALI_V7_ATTRIB_BUF_RECORDS / 2 - 16)

/* Clip distances on Bifrost: varyings the fragment shader tests, four in
 * each of VARYING_SLOT_CLIP_DIST0 and 1 (mali_shader.c, "Clip
 * distances"). */
#define MALI_V7_MAX_CLIP_DISTANCES 8

struct mali_v7_shader_desc {
   /* Entries of each table copied from sets (the dynamic uniform buffers
    * from desc.dyn_bufs follow count[UBO]). */
   uint32_t count[MALI_V7_DESC_TABLE_COUNT];
   /* The first dyn_ubo_count entries of desc.dyn_bufs.map are uniform
    * buffers, the rest storage buffers. */
   uint32_t dyn_ubo_count;
   /* The tables' sources, one MALI_COPY_DESC_HANDLE(set, slot) per entry,
    * table after table in enum order (map_count entries, malloc'd). */
   uint32_t *map;
   uint32_t map_count;
   /* The shader reads v7_desc.sets: storage buffers, texture or image size
    * queries. */
   bool needs_set_addrs;
   /* The lowering could not allocate its map (mali_shader.c lower_nir). */
   bool failed;
   /* The shader's constant data (host copy, for serialization) and its GPU
    * copy, which common.constant_data points at. */
   void *data;
   uint32_t data_size;
   struct mali_bo_ref data_mem;
   /* The GPU copy of `map`, read by the descriptor copy jobs
    * (mali_jm_desc_v7.c). */
   struct mali_bo_ref map_mem;
};

/* SPD slots. A vertex shader has three (panvk v11): position for points,
 * position for other primitives (point-size writes removed), varyings.
 * On v7 the slots hold the Renderer State Descriptor of a vertex or
 * compute shader (both position slots alike, no varying shader) and the
 * code address of a fragment shader (mali_shader.c shader_upload). */
enum mali_spd {
   MALI_SPD_MAIN = 0,          /* fragment or compute */
   MALI_SPD_POS_POINTS = 0,
   MALI_SPD_POS_TRIANGLES = 1,
   MALI_SPD_VARYING = 2,
   MALI_SPD_COUNT = 3,
};

#define MALI_SHADER_CODE_ALIGN 128
#define MALI_SPD_SIZE 32

struct mali_shader {
   struct vk_pipeline_cache_object base;
   blake3_hash key;

   mesa_shader_stage stage;
   struct pan_shader_info info;
   struct mali_shader_fau fau;
   struct mali_shader_desc_info desc;

   union {
      struct {
         uint32_t local_size[3];
      } cs;
      struct {
         /* Bit i: input attachment i (index + 1 for InputAttachmentIndex
          * i; bit 0 for one without) is read. */
         uint32_t input_attachment_read;
         /* Compiled without ATEST (mali_shader.c, "ATEST"): the draw
          * must force early depth/stencil and pixel kill. */
         uint32_t no_atest;
      } fs;
   };

   /* Machine code, host copy (serialization). */
   void *bin;
   uint32_t bin_size;

   /* Fragment shaders with pushed uniform-buffer words from one dynamic
    * buffer (v11 only): the command-stream words that copy them into the
    * draw's FAU block (mali_cmd_draw.c), built at upload. The draw puts the buffer
    * address in MALI_UBO_COPY_SRC and the FAU block's in MALI_UBO_COPY_DST
    * first; valid while the binding covers bytes [0, end). count 0: no
    * such words, or the draw copies them another way. */
   struct {
      uint32_t count;
      uint32_t end;
      uint64_t ins[MALI_UBO_COPY_MAX_INS];
   } ubo_copy;

   /* GPU copies: code in the executable pool (128-byte aligned), SPDs in
    * the pipeline descriptor pool (32-byte aligned; on v7 a 64-byte RSD,
    * none for a fragment shader). */
   struct mali_bo_ref code;
   struct mali_bo_ref spd_mem;
   uint64_t spd[MALI_SPD_COUNT];   /* GPU addresses; 0 if absent */
#if MALI_SHADER_V7
   /* v7: the per-type descriptor tables and the constant data. */
   struct mali_v7_shader_desc v7;
#endif
};

static inline uint64_t
mali_shader_code_va(const struct mali_shader *s)
{
   return s ? s->code.gpu_va : 0;
}

/* The cache object operations, per arch (a v9 shader object uploads and
 * builds v9 SPDs); also the physical device's pipeline_cache_import_ops
 * entry. */
extern const struct vk_pipeline_cache_object_ops mali_v7_shader_cache_ops;
extern const struct vk_pipeline_cache_object_ops mali_v9_shader_cache_ops;
extern const struct vk_pipeline_cache_object_ops mali_v11_shader_cache_ops;
/* NULL-terminated lists for vk_physical_device::pipeline_cache_import_ops. */
extern const struct vk_pipeline_cache_object_ops *const mali_v7_pipeline_cache_import_ops[];
extern const struct vk_pipeline_cache_object_ops *const mali_v9_pipeline_cache_import_ops[];
extern const struct vk_pipeline_cache_object_ops *const mali_v11_pipeline_cache_import_ops[];

static inline struct mali_shader *
mali_shader_ref(struct mali_shader *s)
{
   vk_pipeline_cache_object_ref(&s->base);
   return s;
}

/* Drops a reference (s may be NULL); the last one frees the shader. */
MALI_PER_ARCH_DECL(void, shader_unref, (struct mali_device *dev, struct mali_shader *s));

/* NIR and SPIR-V options for the compile path. */
MALI_PER_ARCH_DECL(const struct nir_shader_compiler_options *, shader_nir_options,
                   (struct mali_device *dev, mesa_shader_stage stage));
MALI_PER_ARCH_DECL(struct spirv_to_nir_options, shader_spirv_options,
                   (const struct vk_pipeline_robustness_state *rs));

struct mali_shader_compile_info {
   /* Consumed (freed) by mali_shader_compile whatever the result. */
   nir_shader *nir;
   const struct vk_pipeline_robustness_state *rs;
   const struct vk_pipeline_layout *layout;
   /* Graphics state (NULL for compute). */
   const struct vk_graphics_pipeline_state *state;
   /* Fragment: the vertex shader's varying layout, if known. */
   const struct pan_varying_layout *vs_varying_layout;
   /* An internal fragment shader (mali_cmd_meta_gfx.c) whose draw forces
    * early depth/stencil and pixel kill unless it writes depth or
    * stencil: it gets no ATEST in that case, as panvk's frame shaders. */
   bool meta;
};

/* The fragment shader keeps its second colour output (location 0,
 * index 1): dualSrcBlend is enabled and some colour attachment blends
 * with a SRC1 factor, or the blend equations are dynamic. Otherwise the
 * output is dropped before compiling. Part of the shader's cache key. */
MALI_PER_ARCH_DECL(bool, shader_fs_keeps_dual_source,
                   (const struct mali_device *dev,
                    const struct vk_graphics_pipeline_state *state));

/* The generic lowering that depends only on the shader and the GPU
 * (panvk_preprocess_nir); run on the NIR from vtn before
 * mali_shader_compile. */
MALI_PER_ARCH_DECL(void, shader_preprocess, (struct mali_device *dev, nir_shader *nir));

/*
 * Lower, compile (kraid on v9 and v11, Mesa's Bifrost backend on v7),
 * upload. A vertex shader reads the fragment shader's noperspective mask
 * from the vs.noperspective_varyings system value, so it does not depend
 * on the fragment shader. On success *out holds one reference and its
 * cache key is key. VK_ERROR_INVALID_SHADER_NV if the compiler refuses the
 * shader (logged).
 */
MALI_PER_ARCH_DECL(VkResult, shader_compile,
                   (struct mali_device *dev, const struct mali_shader_compile_info *info,
                    const blake3_hash key, struct mali_shader **out));

/* Driver-side NIR lowering entry points (mali_nir_*.c). */
void mali_nir_lower_descriptors(nir_shader *nir,
                                const struct vk_pipeline_robustness_state *rs,
                                const struct vk_pipeline_layout *layout,
                                struct mali_shader_desc_info *desc_info);
/* The v7 variant, with per-type tables (mali_nir_lower_descriptors_v7.c).
 * ssbo_addr_format is the storage buffer address format the shader was
 * translated with (shader_spirv_options); rs says whether to bounds-check
 * descriptor indices. False when the table map could not be allocated
 * (v7->map is NULL). */
bool mali_nir_lower_descriptors_v7(nir_shader *nir,
                                   const struct vk_pipeline_robustness_state *rs,
                                   nir_address_format ssbo_addr_format,
                                   const struct vk_pipeline_layout *layout,
                                   struct mali_shader_desc_info *desc_info,
                                   struct mali_v7_shader_desc *v7);
bool mali_nir_lower_input_attachment_loads(nir_shader *nir, unsigned arch,
                                           const struct vk_graphics_pipeline_state *state,
                                           uint32_t *input_attachment_read);

#endif
