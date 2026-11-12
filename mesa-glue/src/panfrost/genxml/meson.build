# SPDX-License-Identifier: MIT
# Derived from Mesa src/panfrost/genxml/meson.build. The common packer and
# one v<N>_pack.h per entry of pan_archs are generated; idep_pan_packers
# below is pan_arch's (the arch the tests build their command-stream code
# for; the driver itself takes each arch's from the _by_arch tables).

_vg = mesa_src / 'panfrost' / 'genxml'

common_pack_h = custom_target(
  'common_pack.h',
  input : files(_vg / 'gen_pack.py', _vg / 'common.xml'),
  output : 'common_pack.h',
  command : [prog_python, '@INPUT@'],
  capture : true,
)

# v<N>_pack.h and the [common, v<N>] pair idep_pan_packers_by_arch wraps,
# keyed by arch as a string (Meson dict keys cannot be ints).
v_pack_h_by_arch = {}
pan_packers_by_arch = {}
idep_pan_packers_by_arch = {}
foreach arch : pan_archs
  _key = '@0@'.format(arch)
  v_pack_h_by_arch += {_key : custom_target(
    'v@0@_pack.h'.format(arch),
    input : files(_vg / 'gen_pack.py', _vg / 'v@0@.xml'.format(arch)),
    output : 'v@0@_pack.h'.format(arch),
    command : [prog_python, '@INPUT@'],
    capture : true,
  )}
  pan_packers_by_arch += {_key : [common_pack_h, v_pack_h_by_arch[_key]]}
  idep_pan_packers_by_arch += {_key : declare_dependency(
    sources : pan_packers_by_arch[_key],
    include_directories : include_directories('.', '../../../../mesa/src/panfrost/genxml'),
  )}
endforeach

# The packers for pan_arch (the tests' arch).
pan_packers = pan_packers_by_arch['@0@'.format(pan_arch)]
idep_pan_packers = idep_pan_packers_by_arch['@0@'.format(pan_arch)]
