# SPDX-License-Identifier: MIT
# Derived from Mesa src/util/rust/meson.build. Not built: the util_rs unit
# tests.

_vur = mesa_src / 'util' / 'rust'

_util_rust_args = [
  rust_global_args,
  rust_2024_lint_args,
  '-Aclippy::identity_op',
  '-Aclippy::len_zero',
  '-Aclippy::manual_range_contains',
  '-Aclippy::needless_range_loop',
  '-Aclippy::redundant_field_names',
  '-Aclippy::upper_case_acronyms',
  '-Aclippy::vec_box',
  '-Aclippy::write_with_newline',
  '-Aclippy::not_unsafe_ptr_arg_deref',
  '-Anon_snake_case',
]

_util_binding_types = [
  'pipe_format',
  'util_format_*',
]

_util_bindgen_args = [
  bindgen_output_args,
  '--allowlist-function', 'util_.*format_.*',
  '--rustified-enum', 'pipe_format',
  '--rustified-enum', 'util_format_layout',
  '--allowlist-var', 'PIPE_FORMAT_.*',
  '--no-prepend-enum-name',
  '--clang-macro-fallback',
  # The fallback's temp files default to fixed names in the cwd, which
  # races with other bindgen invocations in a parallel build.
  '--clang-macro-fallback-build-dir', '@PRIVATE_DIR@',
  '--wrap-static-fns-suffix', '_mesa_util_rs_extern',
]
foreach type : _util_binding_types
  _util_bindgen_args += ['--allowlist-type', type]
endforeach

_inc_util_rust = include_directories('../../../../mesa/src/util/rust')

_util_rust_bindgen = rust.bindgen(
  input : files(_vur / 'util_bindgen.h'),
  output : 'util_bindings.rs',
  output_inline_wrapper : 'util_bindings.c',
  c_args : pre_args,
  include_directories : [_inc_util_rust, inc_util],
  dependencies : [idep_mesautil, idep_mesautilformat],
  args : _util_bindgen_args,
)

_libutil_rust_bindgen_c = static_library(
  'util_rust_bindgen_c',
  _util_rust_bindgen[1],
  include_directories : [_inc_util_rust, inc_util],
  dependencies : [idep_mesautilformat],
  c_args : [mesa_c_args, cc.get_supported_arguments('-Wno-missing-prototypes')],
  gnu_symbol_visibility : 'hidden',
)

# The generated bindings are their own crate, as in Mesa.
_libmesa_util_bindings_rs = static_library(
  'mesa_util_bindings',
  structured_sources([files(_vur / 'bindings' / 'lib.rs'), _util_rust_bindgen[0]]),
  gnu_symbol_visibility : 'hidden',
  rust_abi : 'rust',
  link_with : [_libutil_rust_bindgen_c],
)

_libmesa_util_rs = static_library(
  'mesa_util',
  files(_vur / 'lib.rs'),
  gnu_symbol_visibility : 'hidden',
  rust_abi : 'rust',
  rust_args : _util_rust_args,
  link_with : [_libmesa_util_bindings_rs],
)

dep_syn = dependency('syn',
  version : '>= 2.0.15',
  fallback : ['syn-2-rs', 'dep_syn'],
  required : true,
)

_libmesa_util_proc_rs = rust.proc_macro(
  'mesa_util_proc',
  files(_vur / 'proc' / 'lib.rs'),
  dependencies : [dep_syn],
)

idep_mesa_util_rs = declare_dependency(
  link_with : [_libmesa_util_rs, _libmesa_util_proc_rs],
)

util_rs_bindgen_args = [
  '--raw-line', 'use mesa_util::format::pipe_format;'
]
foreach type : _util_binding_types
  util_rs_bindgen_args += ['--blocklist-type', type]
endforeach
