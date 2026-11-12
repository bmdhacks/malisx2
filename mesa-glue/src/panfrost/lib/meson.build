# SPDX-License-Identifier: MIT
# Derived from Mesa src/panfrost/lib/meson.build. The pixel-format tables
# (pan_format.c) are built once per entry of pan_archs; idep_pan_format
# below names the pan_arch one (the tests' arch).

libpanfrost_pixel_format_by_arch = {}
idep_pan_format_by_arch = {}
foreach arch : pan_archs
  _key = '@0@'.format(arch)
  libpanfrost_pixel_format_by_arch += {_key : static_library(
    'pan-format-v@0@'.format(arch),
    [files(mesa_src / 'panfrost' / 'lib' / 'pan_format.c'), pan_packers_by_arch[_key]],
    include_directories : [inc_src, inc_include, inc_panfrost],
    c_args : [mesa_c_args, '-DPAN_ARCH=@0@'.format(arch)],
    gnu_symbol_visibility : 'hidden',
    dependencies : [idep_pan_packers_by_arch[_key], idep_mesautil],
    build_by_default : false,
  )}
  idep_pan_format_by_arch += {_key : declare_dependency(
    link_with : libpanfrost_pixel_format_by_arch[_key],
    include_directories : [inc_panfrost],
    dependencies : [idep_pan_packers_by_arch[_key]],
  )}
endforeach

# The format library for pan_arch (the tests' arch).
libpanfrost_pixel_format = libpanfrost_pixel_format_by_arch['@0@'.format(pan_arch)]
idep_pan_format = idep_pan_format_by_arch['@0@'.format(pan_arch)]

# Mesa's pandecode for v9 (decode.c and decode_jm.c, the job-chain path),
# for host tests only: never linked into the driver. Tests of the v9 job
# manager hand it the fake kernel's memory to dump a submitted chain.
# decode_common.c (built without PAN_ARCH, as upstream does) dispatches to
# every arch; pandecode_v9_only.c aborts for the archs not built here.
_vg = mesa_src / 'panfrost' / 'genxml'
_libpandecode_v9_arch = static_library(
  'pandecode-arch-v9',
  [files(_vg / 'decode.c', _vg / 'decode_jm.c'), pan_packers_by_arch['9']],
  include_directories : [inc_include, inc_src, inc_panfrost],
  c_args : [mesa_c_args, '-DPAN_ARCH=9'],
  dependencies : [idep_pan_packers_by_arch['9'], idep_nir_headers, idep_mesautil],
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)
_libpandecode_v9 = static_library(
  'pandecode-v9',
  [files(_vg / 'decode_common.c', '../genxml/pandecode_v9_only.c'), common_pack_h],
  include_directories : [inc_include, inc_src, inc_panfrost],
  c_args : mesa_c_args,
  dependencies : [idep_nir_headers, idep_mesautil],
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
  link_with : [_libpandecode_v9_arch, libpanfrost_pixel_format_by_arch['9']],
)
# Include lib/wrap.h for the entry points (pandecode_create_context,
# pandecode_inject_mmap, pandecode_jc).
idep_pandecode_v9 = declare_dependency(
  link_with : [_libpandecode_v9, _libpandecode_v9_arch, libpanfrost_pixel_format_by_arch['9']],
  dependencies : [libpanfrost_model_dep],
  include_directories : [inc_panfrost],
)
