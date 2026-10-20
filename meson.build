# SPDX-License-Identifier: MIT
#
# malisx2: Vulkan driver for Arm Mali (kbase kernel driver).
#
# Layout:
#   src/           the driver
#   mesa-glue/     our meson build for the Mesa sources the driver uses; its
#                  tree mirrors Mesa's src/ so generated headers land where
#                  the Mesa sources expect them
#   mesa/          Mesa, not in this repository (see mesa_commit below)
#   subprojects/   Rust crates kraid needs (meson downloads them)
#   cross/         meson cross files

project(
  'malisx2',
  ['c', 'cpp'],
  version : '0.1.5',
  license : 'MIT',
  meson_version : '>= 1.7.0',
  default_options : [
    'buildtype=debugoptimized',
    'b_ndebug=if-release',
    'c_std=c11',
    'cpp_std=c++17',
    'rust_std=2021',
    'build.rust_std=2021',
    'warning_level=1',
  ],
)

# The GPU architecture of the command-stream (CSF) back half and of the
# code compiled once (Mesa numbering): the G615 in the RG 477V reports
# arch 11.
pan_arch = 11

# Every arch we generate genxml packers and a pixel-format library for, and
# build the per-arch vk/ files for (src/meson.build): 7 for the G52's and
# 9 for the G57's job-manager back half, 11 for the G615.
pan_archs = [7, 9, pan_arch]

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')
prog_python = import('python').find_installation('python3', modules : ['mako', 'yaml'])
fs = import('fs')

# kraid, the shader compiler backend on v9 and v11 (v7 compiles through
# Mesa's Bifrost backend), is Rust.
add_languages('rust', required : true, native : false)
add_languages('rust', required : true, native : true)
rustc = meson.get_compiler('rust')
rust = import('rust')

# Mesa: branch malisx2 of github.com/bmdhacks/armsx2-turnip (upstream Mesa
# plus our kraid changes) at this commit, checked out (or symlinked) at
# mesa/. Only the sources are used; Mesa's own build is not run.
mesa_commit = '730a6284cb7a88f6c19a07f1caea35c7a2ae64f2'
if not fs.is_dir('mesa' / 'src' / 'panfrost')
  error('''Mesa is missing. Fetch it with:
  git init mesa && git -C mesa fetch --depth 1 https://github.com/bmdhacks/armsx2-turnip.git @0@ && git -C mesa checkout FETCH_HEAD'''.format(mesa_commit))
endif
_mesa_head = run_command('git', '-C', meson.project_source_root() / 'mesa',
                         'rev-parse', 'HEAD', check : false)
if _mesa_head.returncode() == 0 and _mesa_head.stdout().strip() != mesa_commit
  warning('mesa/ is at @0@, not the pinned @1@'.format(
          _mesa_head.stdout().strip(), mesa_commit))
endif

subdir('mesa-glue')
subdir('src')
if fs.is_dir('tests')
  subdir('tests')
endif
