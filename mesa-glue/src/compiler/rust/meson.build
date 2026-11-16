# SPDX-License-Identifier: MIT
# Derived from Mesa src/compiler/rust/meson.build.

_vr = mesa_src / 'compiler' / 'rust'

_compiler_rs_names = [
  'as_slice.rs', 'cfg.rs', 'dataflow.rs', 'depth_first_search.rs',
  'float16.rs', 'lower_bounded.rs', 'memstream.rs', 'nir_instr_printer.rs',
  'nir.rs', 'smallvec.rs', 'union_find.rs', 'vec_pair.rs',
]

_compiler_binding_types = [
  'exec_list', 'exec_node', 'float_controls', 'gc_ctx',
  'gl_access_qualifier', 'gl_frag_result', 'gl_interp_mode',
  'mesa_shader_stage', 'gl_subgroup_size', 'gl_system_value',
  'gl_tess_spacing', 'gl_varying_slot', 'gl_vert_attrib',
  'glsl_sampler_dim', 'glsl_type', 'glsl_matrix_layout', 'nir_.*',
  'mesa_scope', 'mesa_prim', 'mesa_shader_stage', 'shader_info',
  'tess_primitive_mode', 'u_printf_info', 'util_dynarray',
]

_compiler_rust_args = [
  rust_global_args,
  rust_2024_lint_args,
  '-Anon_snake_case',
]

_compiler_bindgen_args = [
  bindgen_output_args,
  util_rs_bindgen_args,
  '--allowlist-var', 'NIR_.*',
  '--allowlist-var', 'nir_.*_infos',
  '--allowlist-var', 'rust_.*',
  '--allowlist-function', 'glsl_.*',
  '--allowlist-function', '_mesa_shader_stage_to_string',
  '--allowlist-function', '_mesa_.*half.*',
  '--allowlist-function', '_mesa_.*float16.*',
  '--allowlist-function', 'nir_.*',
  '--allowlist-function', 'compiler_rs.*',
  '--allowlist-function', 'u_memstream.*',
  '--allowlist-function', 'util_dynarray.*',
  '--allowlist-type', 'u_memstream',
  '--no-prepend-enum-name',
  '--with-derive-default',
  '--wrap-static-fns-suffix', '_compiler_rs_extern',
]
foreach type : _compiler_binding_types
  _compiler_bindgen_args += ['--allowlist-type', type]
endforeach

_inc_rust_helpers = include_directories('../../../../mesa/src/compiler/rust')

_libcompiler_c = static_library(
  'compiler_c_helpers',
  files(_vr / 'rust_helpers.c'),
  include_directories : [inc_include, inc_src],
  c_args : mesa_c_args,
  gnu_symbol_visibility : 'hidden',
)

_compiler_bindings_rs = rust.bindgen(
  input : files(_vr / 'bindings.h'),
  output : 'bindings.rs',
  output_inline_wrapper : 'bindings.c',
  c_args : [pre_args, '-DMESA_BINDGEN=1'],
  include_directories : [_inc_rust_helpers, inc_nir],
  args : _compiler_bindgen_args,
  dependencies : [idep_nir_headers, idep_mesautil, idep_mesa_util_rs],
)

_libcompiler_bindings = static_library(
  'compiler_bindings',
  [_compiler_bindings_rs[1]],
  gnu_symbol_visibility : 'hidden',
  include_directories : [_inc_rust_helpers, inc_nir],
  c_args : [mesa_c_args, cc.get_supported_arguments('-Wno-missing-prototypes')],
  dependencies : [idep_nir_headers, idep_mesautil],
)

compiler_rs_bindgen_blocklist = []
foreach type : _compiler_binding_types
  compiler_rs_bindgen_blocklist += ['--blocklist-type', type]
endforeach

_compiler_rs_srcs = [files(_vr / 'lib.rs'), _compiler_bindings_rs[0]]
foreach n : _compiler_rs_names
  _compiler_rs_srcs += files(_vr / n)
endforeach

_libcompiler_rs = static_library(
  'compiler',
  structured_sources(_compiler_rs_srcs),
  gnu_symbol_visibility : 'hidden',
  rust_abi : 'rust',
  link_with : [_libcompiler_bindings, _libcompiler_c],
  dependencies : [idep_mesa_util_rs],
  rust_args : _compiler_rust_args,
)

idep_compiler_rs = declare_dependency(link_with : _libcompiler_rs)

_libcompiler_proc_rs = static_library(
  'compiler_proc',
  files(_vr / 'proc' / 'lib.rs'),
  gnu_symbol_visibility : 'hidden',
  dependencies : [dep_syn],
  rust_abi : 'rust',
  rust_args : rust_global_args,
  native : true,
)

idep_compiler_proc_rs = declare_dependency(link_with : _libcompiler_proc_rs)
