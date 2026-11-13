# SPDX-License-Identifier: MIT
# Derived from Mesa src/poly/cl/meson.build. The generated libpoly.h sits
# at poly/cl/ in this build directory, where Mesa's poly sources look for
# it.

_vpc = mesa_src / 'poly' / 'cl'
_libpoly_cl = files(_vpc / 'geometry.cl', _vpc / 'tessellation.cl')

_libpoly_spv = custom_target(
  'libpoly.spv',
  input : _libpoly_cl,
  output : 'libpoly.spv',
  command : [
    prog_mesa_clc, '-o', '@OUTPUT@', '--depfile', '@DEPFILE@', _libpoly_cl, '--',
    '-I' + (mesa_src / '..' / 'include'), '-I' + (mesa_src / 'compiler' / 'libcl'),
    '-I' + _vpc, '-I' + mesa_src,
    '-cl-std=cl2.0', '-D__OPENCL_VERSION__=200',
    '-DHAVE___BUILTIN_FFS', '-DHAVE___BUILTIN_CLZ', '-DNDEBUG',
  ],
  depfile : 'libpoly_shaders.h.d',
)

libpoly_shaders = custom_target(
  'libpoly',
  input : _libpoly_spv,
  output : ['libpoly.cpp', 'libpoly.h'],
  command : [prog_vtn_bindgen2, '@INPUT@', '@OUTPUT0@', '@OUTPUT1@'],
)
