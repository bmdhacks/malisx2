/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Graphics and compute pipelines.
 *
 * A pipeline holds references to its layout and its compiled shaders (it
 * keeps its layout and its shader binaries alive) and, for graphics, the
 * state the draw encoder needs:
 *
 *  - the pipeline's static state as a vk_dynamic_graphics_state (only the
 *    states the pipeline does not declare dynamic are set in it); the bind
 *    copies all of it into the command buffer only for pipelines with
 *    baked.uses_dynamic_state, and otherwise only what the draws read
 *    (mali_cmd_bind_graphics);
 *  - the bind block (struct mali_gfx_bind), what binds and draws read;
 *  - the hardware words the blob bakes at pipeline creation: TILER_FLAGS,
 *    DCD0 (occlusion query off and on), DCD1, DCD2, the Depth/stencil
 *    descriptor, and per render target the blend equation and mode. They
 *    are packed for the device's arch (mali_pipeline_state.c is built per
 *    arch); on v9 TILER_FLAGS is word 0 of the Malloc Vertex job's
 *    Primitive section and DCD2 is 0 (v9's Draw descriptor has no Flags 2).
 *    On v7 TILER_FLAGS is word 0 of the Tiler job's Primitive section,
 *    DCD0 the low half of its Draw word 0, DCD1 and DCD2 are 0, and the
 *    Depth/stencil words are words 4 to 11 of the fragment Renderer State
 *    Descriptor. mali_gfx_pack_state() packs them from state; pipeline
 *    creation runs it with the pipeline's static state, and the draw
 *    encoder runs it again with the command buffer's state when
 *    baked.uses_dynamic_state says a dynamic state feeds them.
 *
 * Nothing here encodes a draw.
 */

#ifndef MALI_PIPELINE_H
#define MALI_PIPELINE_H

#include <stdint.h>

#include "vk_graphics_state.h"
#include "vk_pipeline.h"
#include "vk_pipeline_layout.h"

#include "mali_arch.h"
#include "mali_shader.h"

struct vk_command_buffer;

enum mali_blend_rt_mode {
   MALI_BLEND_RT_OFF = 0,        /* unused, or nothing written */
   MALI_BLEND_RT_OPAQUE,         /* plain store */
   MALI_BLEND_RT_FIXED_FUNCTION, /* fixed-function equation below */
   MALI_BLEND_RT_SHADER,         /* needs a blend shader (mali_blend.c) */
};

/*
 * A render target's blend equation in Gallium terms (factors carry their
 * invert bit, which util_blendfactor_* understands), after the
 * format-dependent simplifications of Mesa's pan_blend_optimize_equation.
 * Byte fields: it is part of the blend-shader cache key (mali_blend.h).
 */
struct mali_blend_eq {
   uint8_t enable;
   uint8_t is_float;
   uint8_t rgb_func, alpha_func;            /* enum pipe_blend_func */
   uint8_t rgb_src, rgb_dst;                /* enum pipe_blendfactor */
   uint8_t alpha_src, alpha_dst;
   uint8_t color_mask;
};

/*
 * One render target's blend state. The Blend descriptor itself also needs
 * the render target's Internal Conversion word (from the attachment format)
 * and, if constant_mask is set, the blend constant packed for that format;
 * both come from the format tables, so the draw encoder packs the final
 * descriptor from this.
 */
struct mali_blend_rt_baked {
   enum mali_blend_rt_mode mode;
   VkFormat format;
   uint32_t equation;          /* packed v11 Blend Equation (descriptor word 1) */
   bool load_destination;
   bool srgb;
   uint8_t constant_mask;      /* blend constant channels read (RGB = 0x7, A = 0x8) */
   uint8_t shader_location;    /* fragment output location feeding this RT */
   struct mali_blend_eq eq;    /* for a blend shader */
};

struct mali_gfx_baked {
   /* Packed from dynamic state that is dynamic in this pipeline (other than
    * viewport, scissor, blend constants, line width and vertex input, which
    * feed none of these words): the draw encoder must repack. */
   bool uses_dynamic_state;

   enum mesa_prim prim;           /* static topology, MESA_PRIM_COUNT if dynamic */
   uint32_t tiler_flags;          /* v11 Primitive Flags, v9 and v7 Primitive
                                   * word 0; index type is per draw */
   uint32_t dcd0[2];              /* [0] occlusion query off, [1] counting;
                                   * v7: the low half of the Tiler job's Draw
                                   * word 0, both alike */
   uint32_t dcd1;                 /* 0 on v7 */
   uint32_t dcd2;                 /* v11 only; 0 on v7 and v9 */
   uint32_t zsd[8];               /* Depth/stencil descriptor; v7: words 4 to
                                   * 11 of the fragment Renderer State
                                   * Descriptor (pack_rsd_state) */

   uint8_t rt_count;
   uint8_t rt_written;            /* colour attachments the fragment shader writes */
   uint8_t rt_read;               /* colour attachments it reads as input attachments */
   bool any_dest_read;            /* some RT blends with the destination */
   bool needs_blend_shader;
   /* Blend state that forces a blend shader on every target written. */
   bool logicop_enable;
   uint8_t logicop_func;          /* enum pipe_logicop */
   bool alpha_to_one;
   struct mali_blend_rt_baked blend[MALI_MAX_RTS];

   /* SPDs the draw uses: position (the points variant for point lists),
    * varying (0 when the vertex shader has none), fragment (0 without one).
    * v7: the vertex shader's RSD, 0, the fragment shader's code address. */
   uint64_t vs_pos_spd;
   uint64_t vs_var_spd;
   uint64_t fs_spd;
   /* Largest per-thread TLS of the stages, bytes. */
   uint32_t tls_size;
};

struct mali_pipeline {
   struct vk_pipeline vk;
   struct vk_pipeline_layout *layout;   /* reference */
};

/* Static states the draw path reads, when the pipeline sets them
 * (mali_gfx_bind.static_mask). */
enum mali_gfx_static {
   MALI_GFX_STATIC_VI              = 1u << 0,   /* vi, including bindings_valid */
   MALI_GFX_STATIC_STRIDES         = 1u << 1,
   MALI_GFX_STATIC_VIEWPORTS       = 1u << 2,   /* count and values */
   MALI_GFX_STATIC_SCISSORS        = 1u << 3,
   MALI_GFX_STATIC_BLEND_CONSTANTS = 1u << 4,
   MALI_GFX_STATIC_LINE_WIDTH      = 1u << 5,
   MALI_GFX_STATIC_IAL             = 1u << 6,   /* input attachment map */
};

/*
 * What vkCmdBindPipeline and the draws read of a graphics pipeline, kept
 * together so that switching pipelines touches two cache lines of it
 * instead of the whole object (ARMSX2 switches at almost every draw).
 * The register words are copies of `baked`'s; a pipeline with
 * baked.uses_dynamic_state takes them from the command buffer's repack.
 */
struct mali_gfx_bind {
   uint64_t spd[3];               /* SPD_0, SPD_1, SPD_2 */
   uint64_t zsd;                  /* GPU copy of baked.zsd, 0 if dynamic or
                                   * on v7 */
   uint32_t tiler_flags;
   uint32_t dcd0, dcd1, dcd2;
   uint32_t vary_size;            /* VARY_SIZE */
   uint32_t tls_size;
   uint32_t noperspective;        /* the fragment shader's noperspective mask */
   uint32_t static_mask;          /* MALI_GFX_STATIC_* */

   /* Equal keys mean equal table contents for equal bound sets and vertex
    * buffers (interned at creation, never 0): vertex stage table (vertex
    * input, strides, the shader's descriptor use), fragment stage table
    * (the shader's descriptor use and the varying layouts it reads). */
   uint32_t vs_srt_key, fs_srt_key;
   /* Descriptor sets each stage's resource table reads. */
   uint16_t vs_sets, fs_sets;
   /* The vertex shader reads these system values (FAU inputs that change
    * per draw). */
   bool vs_first_vertex, vs_base_instance;
   bool dynamic;                  /* baked.uses_dynamic_state */
   bool lines;                    /* PRIMITIVE_SIZE is the line width */
   bool has_fs;
   /* Copies of the static line width and input attachment map, when
    * static_mask has them. */
   float line_width;
   struct vk_input_attachment_location_state ial;
} __attribute__((aligned(64)));

struct mali_graphics_pipeline {
   struct mali_pipeline base;

   struct mali_gfx_bind bind;

   struct mali_shader *vs;              /* reference */
   struct mali_shader *fs;              /* reference, NULL without a fragment stage */
   struct mali_bo_ref zsd_mem;          /* bind.zsd */

   /* Which states are dynamic (MESA_VK_DYNAMIC_*). */
   BITSET_DECLARE(dynamic, MESA_VK_DYNAMIC_GRAPHICS_STATE_ENUM_MAX);
   /* The pipeline's static state, for the bind. */
   struct vk_dynamic_graphics_state state;
   struct vk_vertex_input_state vi;
   struct vk_sample_locations_state sl;
   struct vk_render_pass_state rp;

   struct mali_gfx_baked baked;

   /* The blend shaders of the static blend state, compiled at creation
    * (mali_blend.c blend_shaders_prepare), per render target in shader
    * mode, with the target format and sample count they were built for.
    * A draw into a matching target takes the address from here instead
    * of the device's cache; addr 0 means none. */
   struct {
      uint64_t addr;
      uint32_t format;        /* enum pipe_format */
      uint32_t samples;
   } blend_shader[MALI_MAX_RTS];
};

struct mali_compute_pipeline {
   struct mali_pipeline base;
   struct mali_shader *cs;              /* reference */
};

static inline struct mali_graphics_pipeline *
mali_graphics_pipeline(struct vk_pipeline *p)
{
   assert(p->bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS);
   return container_of(p, struct mali_graphics_pipeline, base.vk);
}

static inline struct mali_compute_pipeline *
mali_compute_pipeline(struct vk_pipeline *p)
{
   assert(p->bind_point == VK_PIPELINE_BIND_POINT_COMPUTE);
   return container_of(p, struct mali_compute_pipeline, base.vk);
}

/* What mali_gfx_pack_state() packs from. */
struct mali_gfx_pack_input {
   const struct vk_dynamic_graphics_state *dyn;
   const struct vk_render_pass_state *rp;
   const struct mali_shader *vs;
   const struct mali_shader *fs;       /* may be NULL */
   uint32_t view_mask;                 /* multiview, 0 without */
};

MALI_PER_ARCH_DECL(void, gfx_pack_state,
                   (const struct mali_gfx_pack_input *in, struct mali_gfx_baked *out));

struct mali_device;

/* A small id for a byte string, the same for equal strings for the
 * device's lifetime; 0 on allocation failure (compares unequal to
 * everything at bind). mali_device.c, compiled once. */
uint32_t mali_device_intern_key(struct mali_device *dev, const void *data, size_t size);
void mali_device_keys_init(struct mali_device *dev);
void mali_device_keys_finish(struct mali_device *dev);

#endif
