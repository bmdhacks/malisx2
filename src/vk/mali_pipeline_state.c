/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The per-pipeline hardware words (mali_pipeline.h, struct mali_gfx_baked).
 *
 * What goes into each word follows panvk's v11 draw code (Mesa
 * src/panfrost/vulkan/csf/panvk_vX_cmd_draw.c: build_dcd_flags, build_zsd,
 * set_tiler_idvs_flags; panvk_vX_blend.c), which computes the same words at
 * draw time. The blob computes them at pipeline creation instead;
 * ARMSX2's pipelines leave only
 * viewport, scissor, blend constants and line width dynamic, none of which
 * feed these words, so baking them once is exact for it.
 *
 * The fixed-function blend translation and the early-ZS decision are
 * written after Mesa's src/panfrost/lib/pan_blend.c and pan_earlyzs.c
 * (MIT), which are not vendored.
 *
 * Built per arch. On v9 (gallium's pan_jm.c and the T820 blob for the
 * job-manager words; genxml v9.xml for the layouts):
 *  - TILER_FLAGS is word 0 of the Malloc Vertex job's Primitive section,
 *    which has no position FIFO format or view mask (v9 has no multiview
 *    in the tiler) and whose "Allow rotating primitives" defaults to set:
 *    it is cleared, as the blob never sets it and as v11 has it;
 *  - DCD Flags 0 has no conservative rasterization or cull-zero-area,
 *    DCD Flags 1 an 8-bit render target mask, and there is no DCD Flags 2
 *    (dcd2 stays 0). Flags 2 holds the tile-buffer read/write masks on
 *    v11, which rasterization-order attachment access does not use
 *    either way; v9 has no per-render-target mask to narrow tile-read
 *    ordering against, so it may stall more conservatively there than on
 *    v11 -- a speed question, not a correctness one;
 *  - the Depth/stencil descriptor has no separated dependency tracking;
 *  - the depth/stencil update may be weak-early (pan_earlyzs.c forces it
 *    early from v11 only).
 */

#ifndef PAN_ARCH
#error "mali_pipeline_state.c is built per arch: PAN_ARCH must be set"
#endif
#include "genxml/gen_macros.h"

#include "mali_pipeline.h"

#include <string.h>

#include "util/blend.h"
#include "util/format/u_format.h"

#include "vk_blend.h"
#include "vk_format.h"

#include "mali_arch.h"
#if PAN_ARCH < 9
#include "mali_rsd_v7.h"
#endif

/* ---------------------------------------------------------------------- */
/* Blend                                                                    */


/* Formats the fixed-function blender handles (Mesa's blendable format
 * table for v9+: 8-bit UNORM/sRGB, 565, 4444, 5551, 10-10-10-2 and the
 * 16-bit/11-11-10 float formats). The float formats are v9+ only: on v7
 * a blended float target takes a blend shader. */
static bool
format_supports_hw_blend(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_R8_UNORM:
   case VK_FORMAT_R8_SRGB:
   case VK_FORMAT_R8G8_UNORM:
   case VK_FORMAT_R8G8_SRGB:
   case VK_FORMAT_R8G8B8_UNORM:
   case VK_FORMAT_R8G8B8_SRGB:
   case VK_FORMAT_B8G8R8_UNORM:
   case VK_FORMAT_B8G8R8_SRGB:
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SRGB:
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_B8G8R8A8_SRGB:
   case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
   case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
   case VK_FORMAT_R5G6B5_UNORM_PACK16:
   case VK_FORMAT_B5G6R5_UNORM_PACK16:
   case VK_FORMAT_R4G4B4A4_UNORM_PACK16:
   case VK_FORMAT_B4G4R4A4_UNORM_PACK16:
   case VK_FORMAT_R5G5B5A1_UNORM_PACK16:
   case VK_FORMAT_B5G5R5A1_UNORM_PACK16:
   case VK_FORMAT_A1R5G5B5_UNORM_PACK16:
   case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
   case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
      return true;
#if PAN_ARCH >= 9
   case VK_FORMAT_R16_SFLOAT:
   case VK_FORMAT_R16G16_SFLOAT:
   case VK_FORMAT_R16G16B16A16_SFLOAT:
   case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
      return true;
#endif
   default:
      return false;
   }
}

static bool
factor_is_supported(enum pipe_blendfactor f)
{
   f = util_blendfactor_without_invert(f);
   return f != PIPE_BLENDFACTOR_SRC_ALPHA_SATURATE &&
          f != PIPE_BLENDFACTOR_SRC1_COLOR && f != PIPE_BLENDFACTOR_SRC1_ALPHA;
}

/* Float render targets allow far fewer factors. */
static bool
factor_is_supported_for_float(enum pipe_blendfactor f)
{
   return f == PIPE_BLENDFACTOR_ZERO || f == PIPE_BLENDFACTOR_ONE ||
          f == PIPE_BLENDFACTOR_SRC_ALPHA || f == PIPE_BLENDFACTOR_INV_SRC_ALPHA;
}

/* src*dst + dst*src, which the hardware does as 0 + dst * (2 * src). */
static bool
is_2srcdest(enum pipe_blend_func func, enum pipe_blendfactor src,
            enum pipe_blendfactor dst, bool is_alpha)
{
   return func == PIPE_BLEND_ADD &&
          (src == PIPE_BLENDFACTOR_DST_COLOR ||
           (src == PIPE_BLENDFACTOR_DST_ALPHA && is_alpha)) &&
          (dst == PIPE_BLENDFACTOR_SRC_COLOR ||
           (dst == PIPE_BLENDFACTOR_SRC_ALPHA && is_alpha));
}

static bool
can_fixed_function_part(enum pipe_blend_func func, enum pipe_blendfactor src,
                        enum pipe_blendfactor dst, bool is_alpha, bool is_float)
{
   /* Add and subtract only; no min/max. */
   if (func != PIPE_BLEND_ADD && func != PIPE_BLEND_SUBTRACT &&
       func != PIPE_BLEND_REVERSE_SUBTRACT)
      return false;

   if (is_float) {
      if (func == PIPE_BLEND_ADD && src == PIPE_BLENDFACTOR_ZERO &&
          (dst == PIPE_BLENDFACTOR_INV_SRC_COLOR || dst == PIPE_BLENDFACTOR_DST_ALPHA))
         return true;
      return factor_is_supported_for_float(src) && factor_is_supported_for_float(dst);
   }

   if (is_2srcdest(func, src, dst, is_alpha))
      return true;

   if (!factor_is_supported(src) || !factor_is_supported(dst))
      return false;

   /* The two factors must match up to an invert, or one must be 0/1. */
   enum pipe_blendfactor s = util_blendfactor_without_invert(src);
   enum pipe_blendfactor d = util_blendfactor_without_invert(dst);
   return s == d || s == PIPE_BLENDFACTOR_ONE || d == PIPE_BLENDFACTOR_ONE;
}

static bool
can_fixed_function(const struct mali_blend_eq *eq)
{
   return !eq->enable ||
          (can_fixed_function_part(eq->rgb_func, eq->rgb_src, eq->rgb_dst, false,
                                   eq->is_float) &&
           can_fixed_function_part(eq->alpha_func, eq->alpha_src, eq->alpha_dst,
                                   true, eq->is_float));
}

static unsigned
factor_constant_mask(enum pipe_blendfactor f)
{
   f = util_blendfactor_without_invert(f);
   if (f == PIPE_BLENDFACTOR_CONST_COLOR)
      return 0x7;
   if (f == PIPE_BLENDFACTOR_CONST_ALPHA)
      return 0x8;
   return 0;
}

static unsigned
constant_mask(const struct mali_blend_eq *eq)
{
   if (!eq->enable)
      return 0;
   return factor_constant_mask(eq->rgb_src) | factor_constant_mask(eq->rgb_dst) |
          factor_constant_mask(eq->alpha_src) | factor_constant_mask(eq->alpha_dst);
}

static bool
is_opaque(const struct mali_blend_eq *eq)
{
   /* A masked channel needs the tile buffer read, so no opaque mode. */
   if (eq->color_mask != 0xf)
      return false;
   if (!eq->enable)
      return true;
   return eq->rgb_src == PIPE_BLENDFACTOR_ONE && eq->rgb_dst == PIPE_BLENDFACTOR_ZERO &&
          (eq->rgb_func == PIPE_BLEND_ADD || eq->rgb_func == PIPE_BLEND_SUBTRACT) &&
          eq->alpha_src == PIPE_BLENDFACTOR_ONE && eq->alpha_dst == PIPE_BLENDFACTOR_ZERO &&
          (eq->alpha_func == PIPE_BLEND_ADD || eq->alpha_func == PIPE_BLEND_SUBTRACT);
}

static bool
is_dest_factor(enum pipe_blendfactor f, bool alpha)
{
   f = util_blendfactor_without_invert(f);
   return f == PIPE_BLENDFACTOR_DST_ALPHA || f == PIPE_BLENDFACTOR_DST_COLOR ||
          (f == PIPE_BLENDFACTOR_SRC_ALPHA_SATURATE && !alpha);
}

static bool
reads_dest(const struct mali_blend_eq *eq)
{
   if (eq->color_mask && eq->color_mask != 0xf)
      return true;
   if (!eq->enable)
      return false;
   if (eq->rgb_func == PIPE_BLEND_MIN || eq->rgb_func == PIPE_BLEND_MAX ||
       eq->alpha_func == PIPE_BLEND_MIN || eq->alpha_func == PIPE_BLEND_MAX)
      return true;
   return is_dest_factor(eq->rgb_src, false) || is_dest_factor(eq->alpha_src, true) ||
          eq->rgb_dst != PIPE_BLENDFACTOR_ZERO || eq->alpha_dst != PIPE_BLENDFACTOR_ZERO;
}

static enum mali_blend_operand_c
to_c_factor(enum pipe_blendfactor f)
{
   switch (util_blendfactor_without_invert(f)) {
   case PIPE_BLENDFACTOR_ONE:
      /* Zero, inverted by the caller. */
      return MALI_BLEND_OPERAND_C_ZERO;
   case PIPE_BLENDFACTOR_SRC_ALPHA:
      return MALI_BLEND_OPERAND_C_SRC_ALPHA;
   case PIPE_BLENDFACTOR_DST_ALPHA:
      return MALI_BLEND_OPERAND_C_DEST_ALPHA;
   case PIPE_BLENDFACTOR_SRC_COLOR:
      return MALI_BLEND_OPERAND_C_SRC;
   case PIPE_BLENDFACTOR_DST_COLOR:
      return MALI_BLEND_OPERAND_C_DEST;
   case PIPE_BLENDFACTOR_CONST_COLOR:
   case PIPE_BLENDFACTOR_CONST_ALPHA:
      return MALI_BLEND_OPERAND_C_CONSTANT;
   default:
      UNREACHABLE("unsupported blend factor");
   }
}

/* One channel group of the fixed-function equation A + B * C. */
static void
to_mali_function(enum pipe_blend_func func, enum pipe_blendfactor src,
                 enum pipe_blendfactor dst, bool is_alpha,
                 struct MALI_BLEND_FUNCTION *fn)
{
   /* The hardware has 0 and can invert it to 1; Gallium's ONE is the
    * uninverted form, hence the extra invert. */
   bool src_inv = util_blendfactor_is_inverted(src) ^
                  (util_blendfactor_without_invert(src) == PIPE_BLENDFACTOR_ONE);
   bool dst_inv = util_blendfactor_is_inverted(dst) ^
                  (util_blendfactor_without_invert(dst) == PIPE_BLENDFACTOR_ONE);

   if (src == PIPE_BLENDFACTOR_ZERO) {
      fn->a = MALI_BLEND_OPERAND_A_ZERO;
      fn->b = MALI_BLEND_OPERAND_B_DEST;
      fn->negate_b = func == PIPE_BLEND_SUBTRACT;
      fn->invert_c = dst_inv;
      fn->c = to_c_factor(dst);
   } else if (src == PIPE_BLENDFACTOR_ONE) {
      fn->a = MALI_BLEND_OPERAND_A_SRC;
      fn->b = MALI_BLEND_OPERAND_B_DEST;
      fn->negate_b = func == PIPE_BLEND_SUBTRACT;
      fn->negate_a = func == PIPE_BLEND_REVERSE_SUBTRACT;
      fn->invert_c = dst_inv;
      fn->c = to_c_factor(dst);
   } else if (dst == PIPE_BLENDFACTOR_ZERO) {
      fn->a = MALI_BLEND_OPERAND_A_ZERO;
      fn->b = MALI_BLEND_OPERAND_B_SRC;
      fn->negate_b = func == PIPE_BLEND_REVERSE_SUBTRACT;
      fn->invert_c = src_inv;
      fn->c = to_c_factor(src);
   } else if (dst == PIPE_BLENDFACTOR_ONE) {
      fn->a = MALI_BLEND_OPERAND_A_DEST;
      fn->b = MALI_BLEND_OPERAND_B_SRC;
      fn->negate_a = func == PIPE_BLEND_SUBTRACT;
      fn->negate_b = func == PIPE_BLEND_REVERSE_SUBTRACT;
      fn->invert_c = src_inv;
      fn->c = to_c_factor(src);
   } else if (src == dst) {
      fn->a = MALI_BLEND_OPERAND_A_ZERO;
      fn->invert_c = src_inv;
      fn->c = to_c_factor(src);
      if (func == PIPE_BLEND_ADD) {
         fn->b = MALI_BLEND_OPERAND_B_SRC_PLUS_DEST;
      } else {
         fn->b = MALI_BLEND_OPERAND_B_SRC_MINUS_DEST;
         fn->negate_b = func == PIPE_BLEND_REVERSE_SUBTRACT;
      }
   } else if (is_2srcdest(func, src, dst, is_alpha)) {
      fn->a = MALI_BLEND_OPERAND_A_ZERO;
      fn->b = MALI_BLEND_OPERAND_B_DEST;
      fn->c = MALI_BLEND_OPERAND_C_SRC_X_2;
   } else {
      /* Same factor, one inverted: src*f + dst*(1-f) = dst + (src-dst)*f. */
      assert(util_blendfactor_without_invert(src) ==
                util_blendfactor_without_invert(dst) && src_inv != dst_inv);
      fn->a = MALI_BLEND_OPERAND_A_DEST;
      fn->invert_c = src_inv;
      fn->c = to_c_factor(src);
      switch (func) {
      case PIPE_BLEND_ADD:
         fn->b = MALI_BLEND_OPERAND_B_SRC_MINUS_DEST;
         break;
      case PIPE_BLEND_REVERSE_SUBTRACT:
         fn->b = MALI_BLEND_OPERAND_B_SRC_PLUS_DEST;
         fn->negate_b = true;
         break;
      case PIPE_BLEND_SUBTRACT:
         fn->b = MALI_BLEND_OPERAND_B_SRC_PLUS_DEST;
         fn->negate_a = true;
         break;
      default:
         UNREACHABLE("invalid blend function");
      }
   }
}

static uint32_t
pack_equation(const struct mali_blend_eq *eq)
{
   struct mali_blend_equation_packed out;

   pan_pack(&out, BLEND_EQUATION, cfg) {
      cfg.color_mask = eq->color_mask;
      if (!eq->enable) {
         /* Replace. */
         cfg.rgb.a = MALI_BLEND_OPERAND_A_SRC;
         cfg.rgb.b = MALI_BLEND_OPERAND_B_SRC;
         cfg.rgb.c = MALI_BLEND_OPERAND_C_ZERO;
         cfg.alpha.a = MALI_BLEND_OPERAND_A_SRC;
         cfg.alpha.b = MALI_BLEND_OPERAND_B_SRC;
         cfg.alpha.c = MALI_BLEND_OPERAND_C_ZERO;
      } else {
         to_mali_function(eq->rgb_func, eq->rgb_src, eq->rgb_dst, false, &cfg.rgb);
         to_mali_function(eq->alpha_func, eq->alpha_src, eq->alpha_dst, true, &cfg.alpha);
      }
   }
   return out.opaque[0];
}

static bool
is_min_max(enum pipe_blend_func func)
{
   return func == PIPE_BLEND_MIN || func == PIPE_BLEND_MAX;
}

/*
 * The simplifications of Mesa's pan_blend_optimize_equation that depend
 * only on the format (the ones that depend on the blend constants are not
 * made: the constants are dynamic in ARMSX2's pipelines). They let more
 * targets use the opaque or fixed-function modes, e.g. a single-channel
 * R32F target written with an R-only mask and no blending is opaque.
 */
static void
optimize_equation(struct mali_blend_eq *eq, enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);
   const unsigned comp_mask = util_format_colormask(desc);

   /* Masked-off channels the format does not have do not matter. */
   if (!(~eq->color_mask & comp_mask))
      eq->color_mask = 0xf;

   if (!eq->enable)
      return;

   /* COLOR factors in the alpha equation mean alpha. */
   eq->alpha_src = util_blendfactor_to_alpha(eq->alpha_src);
   eq->alpha_dst = util_blendfactor_to_alpha(eq->alpha_dst);

   /* MIN and MAX ignore the factors. */
   if (is_min_max(eq->rgb_func)) {
      eq->rgb_src = PIPE_BLENDFACTOR_ONE;
      eq->rgb_dst = PIPE_BLENDFACTOR_ONE;
   }
   if (is_min_max(eq->alpha_func)) {
      eq->alpha_src = PIPE_BLENDFACTOR_ONE;
      eq->alpha_dst = PIPE_BLENDFACTOR_ONE;
   }

   /* Without an alpha channel, destination alpha reads as one. */
   if (!(comp_mask & 0x8)) {
      eq->rgb_src = util_blend_dst_alpha_to_one(eq->rgb_src);
      eq->rgb_dst = util_blend_dst_alpha_to_one(eq->rgb_dst);
      eq->alpha_src = util_blend_dst_alpha_to_one(eq->alpha_src);
      eq->alpha_dst = util_blend_dst_alpha_to_one(eq->alpha_dst);
   }

   if (is_opaque(eq))
      eq->enable = false;
}

static void
pack_blend(const struct mali_gfx_pack_input *in, struct mali_gfx_baked *out)
{
   const struct vk_dynamic_graphics_state *dyn = in->dyn;
   const struct vk_color_blend_state *cb = &dyn->cb;
   const struct vk_render_pass_state *rp = in->rp;
   const uint64_t fs_outputs = in->fs ? in->fs->info.outputs_written : 0;

   out->rt_count = rp->color_attachment_count;
   out->logicop_enable = cb->logic_op_enable;
   out->logicop_func = vk_logic_op_to_pipe(cb->logic_op);
   out->alpha_to_one = dyn->ms.alpha_to_one_enable;

   for (uint32_t i = 0; i < rp->color_attachment_count && i < MALI_MAX_RTS; i++) {
      struct mali_blend_rt_baked *rt = &out->blend[i];
      const struct vk_color_blend_attachment_state *att = &cb->attachments[i];
      const VkFormat format = rp->color_attachment_formats[i];
      const uint8_t loc = dyn->cal.color_map[i];

      rt->mode = MALI_BLEND_RT_OFF;
      rt->format = format;
      rt->shader_location = loc;

      if (loc == MESA_VK_ATTACHMENT_UNUSED || format == VK_FORMAT_UNDEFINED ||
          !(fs_outputs & BITFIELD64_BIT(FRAG_RESULT_DATA0 + loc)) ||
          !(cb->color_write_enables & BITFIELD_BIT(i)) || !att->write_mask)
         continue;

      const enum pipe_format pfmt = vk_format_to_pipe_format(format);

      /* Logic op NOOP leaves integer and UNORM targets alone (panvk). */
      if (out->logicop_enable && out->logicop_func == PIPE_LOGICOP_NOOP &&
          !util_format_is_float(pfmt) && !util_format_is_srgb(pfmt))
         continue;

      struct mali_blend_eq eq = {
         .enable = att->blend_enable,
         .is_float = util_format_is_float(pfmt),
         .rgb_func = vk_blend_op_to_pipe(att->color_blend_op),
         .alpha_func = vk_blend_op_to_pipe(att->alpha_blend_op),
         .rgb_src = vk_blend_factor_to_pipe(att->src_color_blend_factor),
         .rgb_dst = vk_blend_factor_to_pipe(att->dst_color_blend_factor),
         .alpha_src = vk_blend_factor_to_pipe(att->src_alpha_blend_factor),
         .alpha_dst = vk_blend_factor_to_pipe(att->dst_alpha_blend_factor),
         .color_mask = att->write_mask,
      };
      optimize_equation(&eq, pfmt);
      rt->eq = eq;

      rt->srgb = util_format_is_srgb(pfmt);
      rt->load_destination = reads_dest(&eq);
      rt->constant_mask = constant_mask(&eq);
      out->any_dest_read |= rt->load_destination;

      if (out->logicop_enable || out->alpha_to_one) {
         rt->mode = MALI_BLEND_RT_SHADER;
      } else if (is_opaque(&eq)) {
         rt->mode = MALI_BLEND_RT_OPAQUE;
         rt->equation = pack_equation(&eq);
      } else if (!format_supports_hw_blend(format) || !can_fixed_function(&eq)) {
         rt->mode = MALI_BLEND_RT_SHADER;
      } else {
         rt->mode = MALI_BLEND_RT_FIXED_FUNCTION;
         rt->equation = pack_equation(&eq);
      }

      /* Fixed-function blending has one constant per draw (v7+ take it from
       * render target 0); whether the channels used agree depends on the
       * dynamic blend constants, so the draw (mali_cmd_state.h mali_gfx_blend)
       * checks and falls back to a blend shader then. */
      out->needs_blend_shader |= rt->mode == MALI_BLEND_RT_SHADER;
   }
}

/* ---------------------------------------------------------------------- */
/* Early depth/stencil                                                      */

enum zs_read {
   ZS_NOT_READ,
   ZS_READ,
};

struct earlyzs {
   enum mali_pixel_kill update, kill;
};

/*
 * When depth/stencil testing and the pixel kill happen relative to the
 * shader (pan_earlyzs.c: from arch 11 the update can never be weak-early;
 * the read-only ZS optimisation exists only on v10).
 */
static struct earlyzs
earlyzs_get(const struct pan_shader_info *s, bool writes_zs_or_oq,
            bool alpha_to_coverage, bool zs_always_passes, enum zs_read zs_read)
{
   const bool shader_writes_zs = s->fs.writes_depth || s->fs.writes_stencil;
   bool late_update = shader_writes_zs || alpha_to_coverage;
   bool late_kill = shader_writes_zs;
   const bool force_early_update = PAN_ARCH >= 11;
   const bool force_early_kill = s->fs.early_fragment_tests;

   /* Discards and coverage writes change which samples get written. */
   const bool late_coverage =
      s->fs.writes_coverage || s->fs.can_discard || alpha_to_coverage;
   late_update |= late_coverage && writes_zs_or_oq;

   /* Side effects must happen before the thread may be killed. */
   late_kill |= s->writes_global;

   /* Reading the ZS tile buffer needs late tests. */
   if (zs_read != ZS_NOT_READ)
      late_update = true;

   late_update &= !s->fs.early_fragment_tests;
   late_kill &= !s->fs.early_fragment_tests;

#define EARLY(force) \
   ((zs_always_passes && !(force)) ? MALI_PIXEL_KILL_WEAK_EARLY \
                                   : MALI_PIXEL_KILL_FORCE_EARLY)
   return (struct earlyzs){
      .update = late_update ? MALI_PIXEL_KILL_FORCE_LATE : EARLY(force_early_update),
      .kill = late_kill ? MALI_PIXEL_KILL_FORCE_LATE : EARLY(force_early_kill),
   };
#undef EARLY
}

/* ---------------------------------------------------------------------- */
/* Depth/stencil                                                            */

static bool
has_depth(const struct vk_render_pass_state *rp)
{
   return rp->attachments & MESA_VK_RP_ATTACHMENT_DEPTH_BIT;
}

static bool
has_stencil(const struct vk_render_pass_state *rp)
{
   return rp->attachments & MESA_VK_RP_ATTACHMENT_STENCIL_BIT;
}

static bool
writes_depth(const struct mali_gfx_pack_input *in)
{
   const struct vk_depth_stencil_state *ds = &in->dyn->ds;
   return has_depth(in->rp) && ds->depth.test_enable && ds->depth.write_enable &&
          ds->depth.compare_op != VK_COMPARE_OP_NEVER;
}

static bool
face_writes_stencil(const struct vk_stencil_test_face_state *f)
{
   return f->write_mask && (f->op.fail != VK_STENCIL_OP_KEEP ||
                            f->op.pass != VK_STENCIL_OP_KEEP ||
                            f->op.depth_fail != VK_STENCIL_OP_KEEP);
}

static bool
writes_stencil(const struct mali_gfx_pack_input *in)
{
   const struct vk_depth_stencil_state *ds = &in->dyn->ds;
   return has_stencil(in->rp) && ds->stencil.test_enable &&
          (face_writes_stencil(&ds->stencil.front) ||
           face_writes_stencil(&ds->stencil.back));
}

static bool
zs_test_always_passes(const struct mali_gfx_pack_input *in)
{
   const struct vk_depth_stencil_state *ds = &in->dyn->ds;

   if (!has_depth(in->rp))
      return true;
   if (ds->depth.test_enable && ds->depth.compare_op != VK_COMPARE_OP_ALWAYS)
      return false;
   if (ds->stencil.test_enable &&
       (ds->stencil.front.op.compare != VK_COMPARE_OP_ALWAYS ||
        ds->stencil.back.op.compare != VK_COMPARE_OP_ALWAYS))
      return false;
   return true;
}

static enum mali_stencil_op
stencil_op(VkStencilOp op)
{
   switch (op) {
   case VK_STENCIL_OP_KEEP:
      return MALI_STENCIL_OP_KEEP;
   case VK_STENCIL_OP_ZERO:
      return MALI_STENCIL_OP_ZERO;
   case VK_STENCIL_OP_REPLACE:
      return MALI_STENCIL_OP_REPLACE;
   case VK_STENCIL_OP_INCREMENT_AND_CLAMP:
      return MALI_STENCIL_OP_INCR_SAT;
   case VK_STENCIL_OP_DECREMENT_AND_CLAMP:
      return MALI_STENCIL_OP_DECR_SAT;
   case VK_STENCIL_OP_INCREMENT_AND_WRAP:
      return MALI_STENCIL_OP_INCR_WRAP;
   case VK_STENCIL_OP_DECREMENT_AND_WRAP:
      return MALI_STENCIL_OP_DECR_WRAP;
   case VK_STENCIL_OP_INVERT:
      return MALI_STENCIL_OP_INVERT;
   default:
      UNREACHABLE("invalid stencil op");
   }
}

/* VkCompareOp and the hardware's Func enum have the same values. */
static enum mali_func
compare_func(VkCompareOp op)
{
   STATIC_ASSERT(VK_COMPARE_OP_LESS == (VkCompareOp)MALI_FUNC_LESS);
   STATIC_ASSERT(VK_COMPARE_OP_GREATER_OR_EQUAL == (VkCompareOp)MALI_FUNC_GEQUAL);
   STATIC_ASSERT(VK_COMPARE_OP_ALWAYS == (VkCompareOp)MALI_FUNC_ALWAYS);
   return (enum mali_func)op;
}

#if PAN_ARCH >= 9
static void
pack_zsd(const struct mali_gfx_pack_input *in, struct mali_gfx_baked *out)
{
   const struct vk_depth_stencil_state *ds = &in->dyn->ds;
   const struct vk_rasterization_state *rs = &in->dyn->rs;
   const bool test_s = has_stencil(in->rp) && ds->stencil.test_enable;
   const bool test_z = has_depth(in->rp) && ds->depth.test_enable;

   pan_pack((struct mali_depth_stencil_packed *)out->zsd, DEPTH_STENCIL, cfg) {
      cfg.stencil_test_enable = test_s;
      if (test_s) {
         cfg.front_compare_function = compare_func(ds->stencil.front.op.compare);
         cfg.front_stencil_fail = stencil_op(ds->stencil.front.op.fail);
         cfg.front_depth_fail = stencil_op(ds->stencil.front.op.depth_fail);
         cfg.front_depth_pass = stencil_op(ds->stencil.front.op.pass);
         cfg.back_compare_function = compare_func(ds->stencil.back.op.compare);
         cfg.back_stencil_fail = stencil_op(ds->stencil.back.op.fail);
         cfg.back_depth_fail = stencil_op(ds->stencil.back.op.depth_fail);
         cfg.back_depth_pass = stencil_op(ds->stencil.back.op.pass);
      }

      cfg.stencil_from_shader = in->fs ? in->fs->info.fs.writes_stencil : false;
      cfg.front_write_mask = ds->stencil.front.write_mask;
      cfg.back_write_mask = ds->stencil.back.write_mask;
      cfg.front_value_mask = ds->stencil.front.compare_mask;
      cfg.back_value_mask = ds->stencil.back.compare_mask;
      cfg.front_reference_value = ds->stencil.front.reference;
      cfg.back_reference_value = ds->stencil.back.reference;

      cfg.depth_cull_enable = vk_rasterization_state_depth_clip_enable(rs);
      if (rs->depth_clamp_enable)
         cfg.depth_clamp_mode = MALI_DEPTH_CLAMP_MODE_BOUNDS;

      if (in->fs) {
#if PAN_ARCH >= 10
         cfg.separated_dependency_tracking = true;
#endif
         cfg.depth_source = in->fs->info.fs.writes_depth ?
                               MALI_DEPTH_SOURCE_SHADER :
                               MALI_DEPTH_SOURCE_FIXED_FUNCTION;
      }

      cfg.depth_write_enable = test_z && ds->depth.write_enable;
      cfg.depth_bias_enable = rs->depth_bias.enable;
      cfg.depth_function = test_z ? compare_func(ds->depth.compare_op) : MALI_FUNC_ALWAYS;
      cfg.depth_units = rs->depth_bias.constant_factor;
      cfg.depth_factor = rs->depth_bias.slope_factor;
      cfg.depth_bias_clamp = rs->depth_bias.clamp;
   }
}
#endif /* PAN_ARCH >= 9 */

/* ---------------------------------------------------------------------- */
/* DCD flags and tiler flags                                                */

static uint32_t
color_attachment_written_mask(const struct mali_shader *fs,
                              const struct vk_color_attachment_location_state *cal)
{
   uint32_t by_shader = (fs->info.outputs_written >> FRAG_RESULT_DATA0) & BITFIELD_MASK(8);
   uint32_t mask = 0;

   for (uint32_t i = 0; i < MALI_MAX_RTS; i++) {
      if (cal->color_map[i] != MESA_VK_ATTACHMENT_UNUSED &&
          (by_shader & BITFIELD_BIT(cal->color_map[i])))
         mask |= BITFIELD_BIT(i);
   }
   return mask;
}

static uint32_t
color_attachment_read_mask(const struct mali_shader *fs,
                           const struct vk_input_attachment_location_state *ial,
                           uint8_t color_mask)
{
   uint32_t count = ial->color_attachment_count == MESA_VK_COLOR_ATTACHMENT_COUNT_UNKNOWN ?
                       util_last_bit(color_mask) : ial->color_attachment_count;
   uint32_t mask = 0;

   for (uint32_t i = 0; i < count && i < MALI_MAX_RTS; i++) {
      if (ial->color_map[i] != MESA_VK_ATTACHMENT_UNUSED &&
          (fs->fs.input_attachment_read & BITFIELD_BIT(ial->color_map[i] + 1)))
         mask |= BITFIELD_BIT(i);
   }
   return mask;
}

static bool
attachment_read(const struct mali_shader *fs, uint8_t att)
{
   uint32_t bit = att == MESA_VK_ATTACHMENT_NO_INDEX ? BITFIELD_BIT(0) :
                  att != MESA_VK_ATTACHMENT_UNUSED   ? BITFIELD_BIT(att + 1) : 0;
   return fs->fs.input_attachment_read & bit;
}

static enum mali_draw_mode
draw_mode(enum mesa_prim prim)
{
   switch (prim) {
   case MESA_PRIM_POINTS:
      return MALI_DRAW_MODE_POINTS;
   case MESA_PRIM_LINES:
      return MALI_DRAW_MODE_LINES;
   case MESA_PRIM_LINE_STRIP:
      return MALI_DRAW_MODE_LINE_STRIP;
   case MESA_PRIM_TRIANGLES:
      return MALI_DRAW_MODE_TRIANGLES;
   case MESA_PRIM_TRIANGLE_STRIP:
      return MALI_DRAW_MODE_TRIANGLE_STRIP;
   case MESA_PRIM_TRIANGLE_FAN:
      return MALI_DRAW_MODE_TRIANGLE_FAN;
   default:
      return MALI_DRAW_MODE_NONE;
   }
}

static void
pack_flags(const struct mali_gfx_pack_input *in, struct mali_gfx_baked *out)
{
   const struct vk_dynamic_graphics_state *dyn = in->dyn;
   const struct vk_rasterization_state *rs = &dyn->rs;
   const struct mali_shader *fs = in->fs;
   const struct mali_shader *vs = in->vs;
   const bool a2c = dyn->ms.alpha_to_coverage_enable;
   const bool writes_zs = writes_depth(in) || writes_stencil(in);
   const uint8_t rt_mask = in->rp->attachments & MESA_VK_RP_ATTACHMENT_ANY_COLOR_BITS;
   bool modifies_coverage = false;

   const enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);
   const enum mesa_prim reduced = u_reduced_prim(prim);
   bool msaa = dyn->ms.rasterization_samples > 1;
   /* Bresenham lines are rasterized at pixel centres. */
   if (reduced == MESA_PRIM_LINES && rs->line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM)
      msaa = false;

   out->prim = prim;
   if (fs) {
      out->rt_written = color_attachment_written_mask(fs, &dyn->cal);
      out->rt_read = color_attachment_read_mask(fs, &dyn->ial, rt_mask);
      modifies_coverage = fs->info.fs.writes_coverage || fs->info.fs.can_discard || a2c;
   }

#if PAN_ARCH < 9
   /*
    * v7 has no DCD flag words: their early-ZS, multisample and pixel-kill
    * state is in the fragment RSD (pack_rsd_state). dcd0 is the low half
    * of the Tiler job's Draw word 0: culling, front face and flat-shading
    * vertex (panvk_emit_tiler_dcd), alike for both occlusion variants (no
    * occlusion queries on v7). The per-draw instance sizes in the upper
    * half are packed as 1 (the padded encoding has no zero) and masked off.
    */
   {
      const bool polygon = reduced == MESA_PRIM_TRIANGLES;
      struct mali_draw_packed d;
      pan_pack(&d, DRAW, cfg) {
         cfg.front_face_ccw = rs->front_face == VK_FRONT_FACE_COUNTER_CLOCKWISE;
         cfg.cull_front_face = polygon && (rs->cull_mode & VK_CULL_MODE_FRONT_BIT);
         cfg.cull_back_face = polygon && (rs->cull_mode & VK_CULL_MODE_BACK_BIT);
         /* Lines take their flat-shading vertex from this bit; every other
          * primitive from the Primitive section's First provoking vertex
          * (panvk). */
         cfg.flat_shading_vertex = reduced == MESA_PRIM_LINES;
         cfg.occlusion_query = MALI_OCCLUSION_MODE_DISABLED;
         cfg.instance_size = 1;
         cfg.instance_primitive_size = 1;
      }
      out->dcd0[0] = out->dcd0[1] = d.opaque[0] & 0xffff;
   }
   (void)writes_zs;
   (void)modifies_coverage;
   (void)msaa;
   out->dcd1 = 0;
   out->dcd2 = 0;
#else
   for (unsigned oq = 0; oq < 2; oq++) {
      struct mali_dcd_flags_0_packed dcd0;

      pan_pack(&dcd0, DCD_FLAGS_0, cfg) {
         if (fs) {
            bool zs_read = attachment_read(fs, dyn->ial.depth_att) ||
                           attachment_read(fs, dyn->ial.stencil_att) ||
                           (dyn->rasterization_order_access &
                            (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT));
            bool roa_color = dyn->rasterization_order_access & VK_IMAGE_ASPECT_COLOR_BIT;

            cfg.allow_forward_pixel_to_kill =
               fs->info.fs.can_fpk && !(rt_mask & ~out->rt_written) &&
               !(out->rt_read & out->rt_written) && !a2c && !out->any_dest_read &&
               !roa_color;
            cfg.allow_forward_pixel_to_be_killed = !fs->info.writes_global;

            struct earlyzs ezs = earlyzs_get(&fs->info, writes_zs || oq, a2c,
                                             zs_test_always_passes(in),
                                             zs_read ? ZS_READ : ZS_NOT_READ);
            cfg.pixel_kill_operation = ezs.kill;
            cfg.zs_update_operation = ezs.update;
            if (fs->fs.no_atest) {
               /* Nothing late is possible without an ATEST (the shader
                * was compiled so); panvk forces early for its ATEST-less
                * frame shaders the same way. */
               assert(ezs.kill != MALI_PIXEL_KILL_FORCE_LATE &&
                      ezs.update != MALI_PIXEL_KILL_FORCE_LATE);
               cfg.pixel_kill_operation = MALI_PIXEL_KILL_FORCE_EARLY;
               cfg.zs_update_operation = MALI_PIXEL_KILL_FORCE_EARLY;
            }

            /* Per-sample shading for sample shading, and for a blend shader
             * with multisampling (it stores one sample per invocation). */
            cfg.evaluate_per_sample =
               (fs->info.fs.sample_shading || out->needs_blend_shader) &&
               dyn->ms.rasterization_samples > 1;
            cfg.shader_modifies_coverage = modifies_coverage;
         } else {
            cfg.allow_forward_pixel_to_kill = true;
            cfg.allow_forward_pixel_to_be_killed = true;
            cfg.pixel_kill_operation = MALI_PIXEL_KILL_FORCE_EARLY;
            cfg.zs_update_operation = MALI_PIXEL_KILL_FORCE_EARLY;
            cfg.overdraw_alpha0 = true;
            cfg.overdraw_alpha1 = true;
         }

         cfg.aligned_line_ends = rs->line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM;
         cfg.front_face_ccw = rs->front_face == VK_FRONT_FACE_COUNTER_CLOCKWISE;
         /* Face culling is for polygons only. */
         const bool polygon = reduced == MESA_PRIM_TRIANGLES;
         cfg.cull_front_face = polygon && (rs->cull_mode & VK_CULL_MODE_FRONT_BIT);
         cfg.cull_back_face = polygon && (rs->cull_mode & VK_CULL_MODE_BACK_BIT);
         cfg.multisample_enable = msaa;
         cfg.occlusion_query = oq ? MALI_OCCLUSION_MODE_COUNTER : MALI_OCCLUSION_MODE_DISABLED;
         cfg.alpha_to_coverage = a2c;
         cfg.scissor_to_bounding_box = true;
#if PAN_ARCH >= 10
         cfg.conservative_rast_mode =
            rs->conservative_mode == VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT ?
               MALI_CONSERVATIVE_RAST_MODE_OVER_ESTIMATE :
               MALI_CONSERVATIVE_RAST_MODE_DISABLED;
         cfg.cull_zero_area = true;
#else
         assert(rs->conservative_mode == VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT);
#endif
      }
      out->dcd0[oq] = dcd0.opaque[0];
   }

   struct mali_dcd_flags_1_packed dcd1;
   pan_pack(&dcd1, DCD_FLAGS_1, cfg) {
      cfg.sample_mask = dyn->ms.sample_mask;
      cfg.render_target_mask = out->rt_written;
   }
   out->dcd1 = dcd1.opaque[0];

#if PAN_ARCH >= 10
   struct mali_dcd_flags_2_packed dcd2;
   pan_pack(&dcd2, DCD_FLAGS_2, cfg) {
      cfg.read_mask = out->rt_read;
      cfg.write_mask = out->rt_written;
      if (fs) {
         cfg.no_shader_depth_read = !attachment_read(fs, dyn->ial.depth_att);
         cfg.no_shader_stencil_read = !attachment_read(fs, dyn->ial.stencil_att);
      }
   }
   out->dcd2 = dcd2.opaque[0];
#else
   /* No Flags 2 on v9: the tile-buffer read and write masks have no home. */
   out->dcd2 = 0;
#endif
#endif /* PAN_ARCH >= 9 */

   /* Tiler flags; the index type is set per draw. */
   const bool points = prim == MESA_PRIM_POINTS;
   const bool writes_psiz = vs->info.vs.writes_point_size && points;
   const bool fs_reads_prim_id = fs && fs->info.fs.reads_primitive_id;
#if PAN_ARCH >= 10
   struct mali_primitive_flags_packed tiler;
   pan_pack(&tiler, PRIMITIVE_FLAGS, cfg) {
#elif PAN_ARCH == 9
   /* The Primitive section of the Malloc Vertex job; word 0 is the flags. */
   struct mali_primitive_packed tiler;
   pan_pack(&tiler, PRIMITIVE, cfg) {
      cfg.allow_rotating_primitives = false;
#else
   /* v7: the Primitive section of the Tiler job; word 0 is the flags, with
    * no rotation, index override or layer bits. The draw sets the index
    * type and the pass's first_provoking_vertex bit in it. */
   struct mali_primitive_packed tiler;
   pan_pack(&tiler, PRIMITIVE, cfg) {
      /* panvk_emit_tiler_primitive: the tiler's task split, and the
       * primitive index written back for a fragment shader that reads it. */
      cfg.job_task_split = 6;
      cfg.primitive_index_writeback = fs_reads_prim_id;
      /* The counts are per draw and only word 0 is kept; the packer
       * stores the index count minus one and asserts it is at least 1. */
      cfg.index_count = 1;
#endif
      cfg.draw_mode = draw_mode(prim);
      cfg.primitive_index_enable = fs_reads_prim_id;
#if PAN_ARCH >= 9
      cfg.primitive_index_override =
         fs_reads_prim_id && (vs->info.outputs_written & VARYING_BIT_PRIMITIVE_ID);
#endif
      cfg.point_size_array_format = writes_psiz ? MALI_POINT_SIZE_ARRAY_FORMAT_FP16
                                                : MALI_POINT_SIZE_ARRAY_FORMAT_NONE;
#if PAN_ARCH >= 9
      cfg.layer_index_enable = vs->info.outputs_written & VARYING_BIT_LAYER;
#endif
#if PAN_ARCH >= 10
      cfg.position_fifo_format = (writes_psiz || vs->info.vs.needs_extended_fifo) ?
                                    MALI_FIFO_FORMAT_EXTENDED : MALI_FIFO_FORMAT_BASIC;
#endif
      cfg.low_depth_cull = cfg.high_depth_cull =
         vk_rasterization_state_depth_clip_enable(rs);
      cfg.secondary_shader = vs->info.vs.secondary_enable && fs != NULL;
#if PAN_ARCH >= 9
      cfg.primitive_restart = dyn->ia.primitive_restart_enable;
#else
      /* An enum on v7: the implicit restart index of the index type
       * (panvk_emit_tiler_primitive). */
      cfg.primitive_restart = dyn->ia.primitive_restart_enable ? MALI_PRIMITIVE_RESTART_IMPLICIT
                                                               : MALI_PRIMITIVE_RESTART_NONE;
#endif
#if PAN_ARCH >= 10
      cfg.view_mask = in->view_mask;
#else
      assert(!in->view_mask);
#endif
   }
   out->tiler_flags = tiler.opaque[0];

   out->vs_pos_spd = points ? vs->spd[MALI_SPD_POS_POINTS] : vs->spd[MALI_SPD_POS_TRIANGLES];
   out->vs_var_spd = fs ? vs->spd[MALI_SPD_VARYING] : 0;
   out->fs_spd = fs ? fs->spd[MALI_SPD_MAIN] : 0;
   out->tls_size = MAX2(vs->info.tls_size, fs ? fs->info.tls_size : 0);
}

#if PAN_ARCH < 9
/*
 * v7: words 4 to 11 of the fragment Renderer State Descriptor (properties,
 * depth bias, multisample misc, stencil), packed into out->zsd as
 * panvk_draw_prepare_fs_rsd packs them at draw time. The draw overlays
 * them on the shader half (mali_rsd_v7.h); the properties word also takes
 * the shader's fields here, so it is complete.
 */
static void
pack_rsd_state(const struct mali_gfx_pack_input *in, struct mali_gfx_baked *out)
{
   const struct vk_dynamic_graphics_state *dyn = in->dyn;
   const struct vk_depth_stencil_state *ds = &dyn->ds;
   const struct vk_rasterization_state *rs = &dyn->rs;
   const struct mali_shader *fs = in->fs;
   const bool test_s = has_stencil(in->rp) && ds->stencil.test_enable;
   const bool test_z = has_depth(in->rp) && ds->depth.test_enable;
   const bool writes_z = writes_depth(in);
   const bool writes_zs = writes_z || writes_stencil(in);
   const bool a2c = dyn->ms.alpha_to_coverage_enable;
   const uint8_t rt_mask = in->rp->attachments & MESA_VK_RP_ATTACHMENT_ANY_COLOR_BITS;
   const enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);
   const bool bresenham = u_reduced_prim(prim) == MESA_PRIM_LINES &&
                          rs->line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM;
   /* Bresenham lines are rasterized at pixel centres. */
   const bool msaa = dyn->ms.rasterization_samples > 1 && !bresenham;

   struct mali_renderer_state_packed rsd;
   pan_pack(&rsd, RENDERER_STATE, cfg) {
      if (fs) {
         mali_v7_rsd_shader(&fs->info, 0, &cfg);

         const bool zs_read = attachment_read(fs, dyn->ial.depth_att) ||
                              attachment_read(fs, dyn->ial.stencil_att) ||
                              (dyn->rasterization_order_access &
                               (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT));
         const bool roa_color = dyn->rasterization_order_access & VK_IMAGE_ASPECT_COLOR_BIT;

         cfg.properties.allow_forward_pixel_to_kill =
            fs->info.fs.can_fpk && !(rt_mask & ~out->rt_written) &&
            !(out->rt_read & out->rt_written) && !a2c && !out->any_dest_read && !roa_color;
         /* The hardware does the early ZS check for FPK on v7 (panvk). */
         cfg.properties.allow_forward_pixel_to_be_killed = !fs->info.writes_global;

         struct earlyzs ezs = earlyzs_get(&fs->info, writes_zs, a2c, zs_test_always_passes(in),
                                          zs_read ? ZS_READ : ZS_NOT_READ);
         cfg.properties.pixel_kill_operation = ezs.kill;
         cfg.properties.zs_update_operation = ezs.update;
         if (fs->fs.no_atest) {
            /* No ATEST (mali_shader.c): nothing late is possible, so force
             * early, as on v9+. */
            assert(ezs.kill != MALI_PIXEL_KILL_FORCE_LATE &&
                   ezs.update != MALI_PIXEL_KILL_FORCE_LATE);
            cfg.properties.pixel_kill_operation = MALI_PIXEL_KILL_FORCE_EARLY;
            cfg.properties.zs_update_operation = MALI_PIXEL_KILL_FORCE_EARLY;
         }
         /* Sample shading only (panvk). The draw adds per-sample evaluation
          * for a blend shader under multisampling once it knows which
          * targets got one (mali_jm_cmd_draw_v7.c build_fs_rsd). */
         cfg.multisample_misc.evaluate_per_sample =
            fs->info.fs.sample_shading && dyn->ms.rasterization_samples > 1;
      } else {
         cfg.properties.depth_source = MALI_DEPTH_SOURCE_FIXED_FUNCTION;
         cfg.properties.allow_forward_pixel_to_kill = true;
         cfg.properties.allow_forward_pixel_to_be_killed = true;
         cfg.properties.zs_update_operation = MALI_PIXEL_KILL_FORCE_EARLY;
      }

      cfg.multisample_misc.multisample_enable = msaa;
      cfg.multisample_misc.sample_mask = dyn->ms.sample_mask;
      cfg.multisample_misc.depth_function =
         test_z ? compare_func(ds->depth.compare_op) : MALI_FUNC_ALWAYS;
      cfg.multisample_misc.depth_write_mask = writes_z;
      cfg.multisample_misc.fixed_function_near_discard =
         cfg.multisample_misc.fixed_function_far_discard =
            vk_rasterization_state_depth_clip_enable(rs);
      cfg.multisample_misc.fixed_function_depth_range_fixed = !rs->depth_clamp_enable;
      cfg.multisample_misc.shader_depth_range_fixed = true;

      cfg.stencil_mask_misc.stencil_enable = test_s;
      cfg.stencil_mask_misc.alpha_to_coverage = a2c;
      cfg.stencil_mask_misc.alpha_test_compare_function = MALI_FUNC_ALWAYS;
      cfg.stencil_mask_misc.front_facing_depth_bias = rs->depth_bias.enable;
      cfg.stencil_mask_misc.back_facing_depth_bias = rs->depth_bias.enable;
      cfg.stencil_mask_misc.aligned_line_ends =
         rs->line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM;

      cfg.depth_units = rs->depth_bias.constant_factor;
      cfg.depth_factor = rs->depth_bias.slope_factor;
      cfg.depth_bias_clamp = rs->depth_bias.clamp;

      cfg.stencil_front.mask = ds->stencil.front.compare_mask;
      cfg.stencil_back.mask = ds->stencil.back.compare_mask;
      cfg.stencil_mask_misc.stencil_mask_front = ds->stencil.front.write_mask;
      cfg.stencil_mask_misc.stencil_mask_back = ds->stencil.back.write_mask;
      cfg.stencil_front.reference_value = ds->stencil.front.reference;
      cfg.stencil_back.reference_value = ds->stencil.back.reference;
      if (test_s) {
         cfg.stencil_front.compare_function = compare_func(ds->stencil.front.op.compare);
         cfg.stencil_front.stencil_fail = stencil_op(ds->stencil.front.op.fail);
         cfg.stencil_front.depth_fail = stencil_op(ds->stencil.front.op.depth_fail);
         cfg.stencil_front.depth_pass = stencil_op(ds->stencil.front.op.pass);
         cfg.stencil_back.compare_function = compare_func(ds->stencil.back.op.compare);
         cfg.stencil_back.stencil_fail = stencil_op(ds->stencil.back.op.fail);
         cfg.stencil_back.depth_fail = stencil_op(ds->stencil.back.op.depth_fail);
         cfg.stencil_back.depth_pass = stencil_op(ds->stencil.back.op.pass);
      }
   }

   STATIC_ASSERT(sizeof(out->zsd) == MALI_V7_RSD_STATE_WORDS * sizeof(uint32_t));
   memcpy(out->zsd, &rsd.opaque[MALI_V7_RSD_STATE_WORD_FIRST], sizeof(out->zsd));
}
#endif /* PAN_ARCH < 9 */

void
MALI_PER_ARCH(gfx_pack_state)(const struct mali_gfx_pack_input *in, struct mali_gfx_baked *out)
{
   const bool keep_dynamic = out->uses_dynamic_state;

   memset(out, 0, sizeof(*out));
   out->uses_dynamic_state = keep_dynamic;

   /* Blend first: DCD0 depends on whether any RT reads the destination or
    * needs a blend shader. */
   pack_blend(in, out);
#if PAN_ARCH >= 9
   pack_zsd(in, out);
   pack_flags(in, out);
#else
   /* v7: pack_rsd_state reads the render target masks pack_flags computes. */
   pack_flags(in, out);
   pack_rsd_state(in, out);
#endif
}
