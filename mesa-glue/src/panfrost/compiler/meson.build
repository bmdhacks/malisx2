# SPDX-License-Identifier: MIT
# Derived from Mesa src/panfrost/compiler/meson.build, with Rust (kraid)
# always enabled.

_vpc = mesa_src / 'panfrost' / 'compiler'

_pc_names = [
  'pan_nir_collect_varyings.c', 'pan_nir_fuse_io_cvt.c',
  'pan_nir_lower_bool_to_bitsize.c', 'pan_nir_lower_divergent_scratch.c',
  'pan_nir_lower_framebuffer.c', 'pan_nir_lower_fs_inputs.c',
  'pan_nir_lower_fs_outputs.c', 'pan_nir_lower_helper_invocation.c',
  'pan_nir_lower_image.c', 'pan_nir_lower_image_64bit.c',
  'pan_nir_lower_image_ms.c', 'pan_nir_lower_mem_to_global.c',
  'pan_nir_lower_noperspective.c', 'pan_nir_lower_sample_position.c',
  'pan_nir_lower_tex.c', 'pan_nir_lower_var_special.c',
  'pan_nir_lower_vertex_id.c', 'pan_nir_lower_vs_inputs.c',
  'pan_nir_lower_vs_outputs.c', 'pan_nir_opt_push_ubo.c',
  'pan_nir_resize_varying_io.c', 'pan_nir_xfb.c',
]
libpanfrost_compiler_files = []
foreach n : _pc_names
  libpanfrost_compiler_files += files(_vpc / n)
endforeach

libpanfrost_compiler_c_args = [mesa_c_args, '-DWITH_PANFROST_RUST']

subdir('kraid')
subdir('bifrost')
subdir('midgard')

# kraid selection without the environment. pan_compiler.c reads the backend
# choice from PAN_USE_KRAID through debug_get_option_cached(). kraid is the
# only backend we use on v9 and v11 (below arch 9 pan_use_kraid() answers
# false whatever the option says, and v7 compiles through the Bifrost
# backend), and setting the variable at run time would race with other
# threads of the application reading the environment. So this one file is
# built with that call renamed to mali_pan_debug_option_cached()
# (pan_kraid_always.c), which answers "all" for PAN_USE_KRAID and forwards
# every other name. Mesa's source is unchanged.
_libpanfrost_compiler_core = static_library(
  'panfrost_compiler_core',
  files(_vpc / 'pan_compiler.c'),
  include_directories : [inc_include, inc_src, inc_valhall, inc_panfrost],
  dependencies : [idep_nir, idep_mesautil, idep_valhall_enums_h, idep_bi_opcodes_h],
  c_args : [libpanfrost_compiler_c_args,
            '-Ddebug_get_option_cached=mali_pan_debug_option_cached'],
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)
libpanfrost_compiler_files += files('pan_kraid_always.c')

libpanfrost_compiler = static_library(
  'panfrost_compiler',
  libpanfrost_compiler_files,
  include_directories : [inc_include, inc_src, inc_valhall, inc_panfrost],
  dependencies : [idep_nir, idep_mesautil, idep_valhall_enums_h, idep_bi_opcodes_h],
  link_with : [_libpanfrost_compiler_core, libpanfrost_midgard,
               libpanfrost_bifrost, libpanfrost_kraid],
  c_args : libpanfrost_compiler_c_args,
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)
