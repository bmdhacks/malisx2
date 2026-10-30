# SPDX-License-Identifier: MIT
# Derived from Mesa src/panfrost/compiler/kraid/meson.build. Not built:
# the pan-dis and isa-gen tools (kraid_disassemble() covers our needs) and
# kraid's own Rust unit tests (they need the hardware runner).

_vk = mesa_src / 'panfrost' / 'compiler' / 'kraid'

dep_bitflags = dependency('bitflags',
  version : '>= 2.9.1',
  fallback : ['bitflags-2-rs', 'dep_bitflags'],
  required : true,
)

dep_paste = dependency('paste',
  version : '>= 1.0.14',
  fallback : ['paste-1-rs', 'dep_paste'],
  required : true,
)

dep_rustc_hash = dependency('rustc-hash',
  version : '>= 2.0',
  fallback : ['rustc-hash-2-rs', 'dep_rustc_hash'],
  required : true,
)

dep_xml_rs = dependency('xml',
  version : '>= 1.2.1',
  fallback : ['xml-rs', 'dep_xml_rs'],
  required : true,
  native : true,
)

_inc_kraid = include_directories('../../../../../mesa/src/panfrost/compiler/kraid')

_kraid_bindings_deps = [idep_mesautil, idep_nir_headers]

_kraid_bindings_rs = rust.bindgen(
  input : files(_vk / 'bindings.h'),
  output : 'kraid_bindings.rs',
  output_inline_wrapper : 'kraid_bindings.c',
  c_args : pre_args,
  include_directories : [inc_src, inc_include, _inc_kraid],
  args : [
    bindgen_output_args,
    compiler_rs_bindgen_blocklist,
    '--raw-line', 'use compiler::bindings::*;',
    '--allowlist-type', 'bi_.*',
    '--allowlist-type', 'pan_.*',
    '--allowlist-function', 'pan_.*',
    '--allowlist-function', 'kraid_.*',
    '--allowlist-function', 'val_.*',
    '--allowlist-function', 'fwrite|fflush',
    '--allowlist-var', 'BI_VA_LOD_MODE_.*',
    '--no-prepend-enum-name',
    '--with-derive-default',
  ],
  dependencies : _kraid_bindings_deps,
)

_libkraid_binding_wrappers = static_library(
  'kraid_binding_wrappers',
  [_kraid_bindings_rs[1]],
  gnu_symbol_visibility : 'hidden',
  include_directories : [inc_src, inc_include, _inc_kraid],
  c_args : [mesa_c_args, cc.get_supported_arguments('-Wno-missing-prototypes')],
  dependencies : _kraid_bindings_deps,
)

_libkraid_bindings_rs = static_library(
  'kraid_bindings',
  _kraid_bindings_rs[0],
  gnu_symbol_visibility : 'hidden',
  dependencies : [idep_compiler_rs],
  link_with : [_libkraid_binding_wrappers],
  rust_abi : 'rust',
)

_libkraid_proc_rs = static_library(
  'kraid_proc',
  files(_vk / 'proc' / 'lib.rs'),
  gnu_symbol_visibility : 'hidden',
  dependencies : [dep_syn, dep_paste, dep_xml_rs],
  rust_args : rust_global_args,
  rust_abi : 'rust',
  native : true,
)

_libkraid_proc_macros_rs = rust.proc_macro(
  'kraid_proc_macros',
  files(_vk / 'proc' / 'macros.rs'),
  dependencies : [idep_compiler_proc_rs, dep_syn],
  link_with : [_libkraid_proc_rs],
  rust_args : rust_global_args,
)

libpanfrost_kraid = static_library(
  'kraid',
  files(_vk / 'lib.rs'),
  gnu_symbol_visibility : 'hidden',
  rust_abi : 'c',
  rust_args : [
    panfrost_rust_args,
    '--cfg', 'kraid_isa_xml_path="' + _vk / 'isa-v9-v14.xml' + '"',
    '--cfg', 'kraid_disasm',
    '-Clink-arg=-Wno-unused-command-line-argument',
  ],
  dependencies : [dep_bitflags, dep_paste, dep_rustc_hash, idep_compiler_rs,
                  idep_mesa_util_rs],
  link_with : [_libkraid_bindings_rs, _libkraid_proc_macros_rs],
)
