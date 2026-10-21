# SPDX-License-Identifier: MIT
# Derived from Mesa src/compiler/spirv/meson.build. SPIRV-Tools is not used
# (Mesa only uses it for debug validation of SPIR-V).

_vs = mesa_src / 'compiler' / 'spirv'

vtn_gather_types_c = custom_target(
  'vtn_gather_types.c',
  input : files(_vs / 'vtn_gather_types_c.py', _vs / 'spirv.core.grammar.json'),
  output : 'vtn_gather_types.c',
  command : [prog_python, '@INPUT0@', '@INPUT1@', '@OUTPUT@'],
)

spirv_info = custom_target(
  'spirv_info',
  input : files(_vs / 'spirv_info_gen.py', _vs / 'spirv.core.grammar.json'),
  output : ['spirv_info.h', 'spirv_info.c'],
  command : [prog_python, '@INPUT0@', '--json', '@INPUT1@',
             '--out-h', '@OUTPUT0@', '--out-c', '@OUTPUT1@'],
)
spirv_info_h = spirv_info[0]

vtn_generator_ids_h = custom_target(
  'vtn_generator_ids.h',
  input : files(_vs / 'vtn_generator_ids_h.py', _vs / 'spir-v.xml'),
  output : 'vtn_generator_ids.h',
  command : [prog_python, '@INPUT0@', '@INPUT1@', '@OUTPUT@'],
)

files_libvtn = []
foreach n : ['gl_spirv.c', 'spirv_to_nir.c', 'vtn_alu.c', 'vtn_amd.c',
             'vtn_cfg.c', 'vtn_cmat.c', 'vtn_constants.c', 'vtn_debug.c',
             'vtn_glsl450.c',
             'vtn_opencl.c', 'vtn_structured_cfg.c', 'vtn_subgroup.c',
             'vtn_variables.c']
  files_libvtn += files(_vs / n)
endforeach

_libvtn = static_library(
  'vtn',
  [files_libvtn, spirv_info, vtn_gather_types_c, vtn_generator_ids_h],
  include_directories : [inc_include, inc_src,
                         include_directories('.', '../../../../mesa/src/compiler/spirv')],
  c_args : mesa_c_args,
  gnu_symbol_visibility : 'hidden',
  dependencies : [idep_nir],
  build_by_default : false,
)

idep_vtn = declare_dependency(
  dependencies : [idep_nir, idep_mesautil],
  link_with : _libvtn,
  sources : [spirv_info_h],
)
