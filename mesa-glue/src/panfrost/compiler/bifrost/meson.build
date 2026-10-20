# SPDX-License-Identifier: MIT
# Derived from Mesa src/panfrost/compiler/bifrost/meson.build. The bifrost
# backend is the compile path on v7; on v9 and v11 kraid compiles, and
# pan_compiler.c and kraid link against this library.

_vb = mesa_src / 'panfrost' / 'compiler' / 'bifrost'

subdir('valhall')

inc_valhall = include_directories(
  '.', 'valhall',
  '../../../../../mesa/src/panfrost/compiler/bifrost',
  '../../../../../mesa/src/panfrost/compiler/bifrost/valhall',
)

_bi_names = [
  'bi_helper_invocations.c', 'bi_layout.c', 'bi_liveness.c',
  'bi_lower_divergent_indirects.c', 'bi_lower_mkvec_swz.c',
  'bi_lower_spill.c', 'bi_lower_swizzle.c', 'bi_print.c',
  'bi_opt_control_flow.c', 'bi_opt_constant_fold.c', 'bi_opt_copy_prop.c',
  'bi_opt_dce.c', 'bi_opt_cse.c', 'bi_opt_push_ubo.c', 'bi_opt_mod_props.c',
  'bi_opt_dual_tex.c', 'bi_iterator_schedule.c', 'bi_pressure_schedule.c',
  'bi_ra.c', 'bi_ra_ssa.c', 'bi_spill_ssa.c', 'bi_reindex_ssa.c',
  'bi_repair_ssa.c', 'bi_validate.c', 'bi_dominance.c', 'bi_debug.c',
  'bir.c', 'bifrost_compile.c', 'bifrost_nir.c',
  'bifrost/bi_opt_message_preload.c', 'bifrost/bi_pack.c',
  'bifrost/bi_schedule.c', 'bifrost/bi_scoreboard.c',
  'valhall/va_gather_hsr_info.c', 'valhall/va_insert_flow.c',
  'valhall/va_lower_constants.c', 'valhall/va_lower_isel.c',
  'valhall/va_lower_split_64bit.c', 'valhall/va_optimize.c',
  'valhall/va_mark_last.c', 'valhall/va_merge_flow.c', 'valhall/va_pack.c',
  'valhall/va_perf.c', 'valhall/va_validate.c',
]
libpanfrost_bifrost_files = []
foreach n : _bi_names
  libpanfrost_bifrost_files += files(_vb / n)
endforeach

_isa_in = files(_vb / 'IR_pseudo.xml', _vb / 'bifrost' / 'ISA.xml', _vb / 'valhall' / 'ISA.xml')
_isa_dep = files(_vb / 'bifrost_isa.py')

bifrost_gen_disasm_c = custom_target(
  'bifrost_gen_disasm.c',
  input : files(_vb / 'gen_disasm.py', _vb / 'bifrost' / 'ISA.xml'),
  output : 'bifrost_gen_disasm.c',
  command : [prog_python, '@INPUT@'],
  capture : true,
  depend_files : _isa_dep,
)

bi_opcodes_c = custom_target(
  'bi_opcodes.c',
  input : [files(_vb / 'bi_opcodes.c.py'), _isa_in],
  output : 'bi_opcodes.c',
  command : [prog_python, '@INPUT@'],
  capture : true,
  depend_files : _isa_dep,
)

bi_swizzles_c = custom_target(
  'bi_swizzles.c',
  input : files(_vb / 'bi_swizzles.c.py', _vb / 'IR_pseudo.xml', _vb / 'bifrost' / 'ISA.xml'),
  output : 'bi_swizzles.c',
  command : [prog_python, '@INPUT@'],
  capture : true,
  depend_files : _isa_dep,
)

bi_printer_c = custom_target(
  'bi_printer.c',
  input : [files(_vb / 'bi_printer.c.py'), _isa_in],
  output : 'bi_printer.c',
  command : [prog_python, '@INPUT@'],
  capture : true,
  depend_files : _isa_dep,
)

bi_packer_c = custom_target(
  'bi_packer.c',
  input : [files(_vb / 'bi_packer.c.py'), _isa_in],
  output : 'bi_packer.c',
  command : [prog_python, '@INPUT@'],
  capture : true,
  depend_files : _isa_dep,
)

bi_opcodes_h = custom_target(
  'bi_opcodes.h',
  input : [files(_vb / 'bi_opcodes.h.py'), _isa_in],
  output : 'bi_opcodes.h',
  command : [prog_python, '@INPUT@'],
  capture : true,
  depend_files : _isa_dep,
)

idep_bi_opcodes_h = declare_dependency(
  sources : [bi_opcodes_h],
  include_directories : inc_valhall,
)

bi_builder_h = custom_target(
  'bi_builder.h',
  input : [files(_vb / 'bi_builder.h.py'), _isa_in],
  output : 'bi_builder.h',
  command : [prog_python, '@INPUT@'],
  capture : true,
  depend_files : _isa_dep,
)

idep_bi_builder_h = declare_dependency(
  sources : [bi_builder_h],
  include_directories : inc_valhall,
)

bifrost_nir_algebraic_c = custom_target(
  'bifrost_nir_algebraic.c',
  input : files(_vb / 'bifrost_nir_algebraic.py'),
  output : 'bifrost_nir_algebraic.c',
  command : [prog_python, '@INPUT@', '-p', dir_compiler_nir],
  capture : true,
  depend_files : nir_algebraic_depends,
)

libpanfrost_bifrost_disasm = static_library(
  'panfrost_bifrost_disasm',
  [files(_vb / 'bifrost' / 'disassemble.c', _vb / 'bi_print_common.c'), bifrost_gen_disasm_c],
  include_directories : [inc_include, inc_src, inc_valhall],
  dependencies : [idep_nir],
  c_args : mesa_c_args,
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

libpanfrost_bifrost = static_library(
  'panfrost_bifrost',
  [libpanfrost_bifrost_files, bi_opcodes_c, bi_swizzles_c, bi_printer_c,
   bi_packer_c, bifrost_nir_algebraic_c, valhall_c],
  include_directories : [inc_include, inc_src, inc_valhall, inc_panfrost],
  dependencies : [idep_nir, idep_bi_opcodes_h, idep_bi_builder_h,
                  idep_valhall_enums_h, libpanfrost_model_dep],
  link_with : [libpanfrost_bifrost_disasm, libpanfrost_valhall_disasm],
  c_args : libpanfrost_compiler_c_args,
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)
