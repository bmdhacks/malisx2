# SPDX-License-Identifier: MIT
# Mesa's libpoly (src/poly): geometry and tessellation shaders in software,
# as compute shaders around a draw, for GPUs without those stages. Its
# OpenCL C helpers become NIR at build time through two host tools Mesa
# builds and installs with -Dmesa-clc=enabled -Dinstall-mesa-clc=true (its
# own cross-build recipe): mesa_clc (OpenCL C to SPIR-V) and vtn_bindgen2
# (SPIR-V to the NIR builder functions the passes call). Without them the
# driver is built without geometry and tessellation shaders.

prog_mesa_clc = find_program('mesa_clc', native : true, required : get_option('poly'))
prog_vtn_bindgen2 = find_program('vtn_bindgen2', native : true, required : get_option('poly'))
with_poly = prog_mesa_clc.found() and prog_vtn_bindgen2.found()

if with_poly
  _vp = mesa_src / 'poly'
  _inc_poly = include_directories('../../../mesa/src/poly', '../../../mesa/src/poly/nir')
  subdir('cl')

  _libpoly = static_library(
    'poly',
    files(_vp / 'nir' / 'poly_nir_lower_gs.c', _vp / 'nir' / 'poly_nir_lower_sysvals.c',
          _vp / 'nir' / 'poly_nir_lower_tess.c', _vp / 'nir' / 'poly_nir_lower_vs.c',
          _vp / 'nir' / 'poly_nir_passthrough_gs.c'),
    libpoly_shaders,
    include_directories : [inc_include, inc_src, _inc_poly],
    c_args : mesa_c_args,
    cpp_args : mesa_c_args,
    dependencies : [idep_nir, idep_mesautil],
    gnu_symbol_visibility : 'hidden',
  )

  idep_poly = declare_dependency(
    sources : [libpoly_shaders[1]],
    link_with : _libpoly,
    include_directories : _inc_poly,
  )
else
  idep_poly = dependency('', required : false)
endif
