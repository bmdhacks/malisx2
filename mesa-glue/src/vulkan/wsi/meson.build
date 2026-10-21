# SPDX-License-Identifier: MIT
# Derived from Mesa src/vulkan/wsi/meson.build: the shared Vulkan WSI
# (VK_KHR_surface and its platforms, the swapchain object and present),
# which talks to the kernel through DRM and to the driver through
# wsi_device / the driver's vk_queue and VkImages.
#
# Built for Linux only, and only the platforms we have a use for so far:
# VK_KHR_display (direct KMS scanout, what ARMSX2's SDL/kmsdrm frontend
# uses on a handheld) and VK_EXT_headless_surface (the device tests). X11
# and Wayland are not built; Android has its own window system in the
# driver (mali_wsi.c).

_vvw = mesa_src / 'vulkan' / 'wsi'

dep_libdrm = dependency('libdrm', required : false)
dep_libudev = dependency('libudev', required : false)

_vk_wsi_files = files(_vvw / 'wsi_common.c', _vvw / 'wsi_common_headless.c')
if dep_libdrm.found()
  _vk_wsi_files += files(_vvw / 'wsi_common_drm.c', _vvw / 'wsi_common_display.c')
endif

wsi_entrypoints = custom_target(
  'wsi_entrypoints',
  input : [vk_entrypoints_gen, vk_api_xml],
  output : ['wsi_common_entrypoints.h', 'wsi_common_entrypoints.c'],
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@', '--proto', '--weak',
    '--out-h', '@OUTPUT0@', '--out-c', '@OUTPUT1@',
    '--prefix', 'wsi',
    '--beta', with_vulkan_beta.to_string()
  ],
  depend_files : vk_entrypoints_gen_depend_files,
)

_libvulkan_wsi = static_library(
  'vulkan_wsi',
  [_vk_wsi_files, wsi_entrypoints],
  include_directories : [inc_include, inc_src, inc_vulkan_runtime,
                         include_directories('.')],
  dependencies : [
    vulkan_wsi_deps,
    dep_libdrm,
    dep_libudev,
    idep_vulkan_util,
    idep_vulkan_lite_runtime_headers,
    idep_mesautil,
    idep_blake3,
  ],
  c_args : [mesa_c_args, '-DHAVE_LIBDRM'],
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

idep_vulkan_wsi_headers = declare_dependency(
  sources : wsi_entrypoints[0],
  dependencies : [vulkan_wsi_deps, dep_libdrm],
  # The generated entry points (here, in the build directory) and Mesa's
  # headers next to the sources, which are outside this project's source
  # tree (an -I because include_directories() will not leave it).
  include_directories : include_directories('.'),
  compile_args : ['-I' + (mesa_src / 'vulkan' / 'wsi'), '-DHAVE_LIBDRM'],
)

# As in Mesa: users link the WSI with --whole-archive, so its weak
# entry-point overloads resolve.
idep_vulkan_wsi = declare_dependency(
  link_whole : _libvulkan_wsi,
  dependencies : [idep_vulkan_wsi_headers, dep_libudev, dep_libdrm],
)
