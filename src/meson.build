# SPDX-License-Identifier: MIT

inc_malisx2 = include_directories('.')

malisx2_c_args = cc.get_supported_arguments([
  '-Werror=implicit-function-declaration',
  '-Werror=missing-prototypes',
  '-Werror=return-type',
  '-Werror=incompatible-pointer-types',
  '-Wno-unused-parameter',
  '-Wno-missing-field-initializers',
])

# The compiler backend step: kraid selection check, SPIR-V structure check,
# NIR -> kraid -> v9 or v11 machine code behind the kraid guard, and
# NIR -> Mesa's Bifrost backend -> v7 machine code.
#
# The guard sits between Mesa's bifrost_nir.c and kraid: every link that
# contains the compiler wraps kraid_compile_nir (idep_mali_compiler carries
# the flag). The list of NIR kraid translates is generated from kraid's
# own source.
mali_kraid_support_h = custom_target(
  'mali_kraid_support.h',
  input : [files('compiler/gen_kraid_support.py'),
           mesa_src / 'panfrost' / 'compiler' / 'kraid' / 'nir.rs'],
  output : 'mali_kraid_support.h',
  command : [prog_python, '@INPUT0@', '--nir-rs', '@INPUT1@',
             '--nir-dir', mesa_src / 'compiler' / 'nir', '--out-h', '@OUTPUT@'],
  depend_files : nir_depends,
)

malisx2_compiler = static_library(
  'mali_compiler',
  files('compiler/mali_compiler.c'), mali_kraid_support_h,
  include_directories : [inc_include, inc_src, inc_panfrost],
  c_args : [pre_args, malisx2_c_args],
  dependencies : [idep_vtn, idep_nir, idep_mesautil, idep_valhall_enums_h,
                  idep_bi_opcodes_h, libpanfrost_model_dep],
  link_with : [libpanfrost_compiler],
  gnu_symbol_visibility : 'hidden',
)

_kraid_wrap = '-Wl,--wrap=kraid_compile_nir'
idep_mali_compiler = declare_dependency(
  link_with : malisx2_compiler,
  include_directories : [include_directories('compiler'), inc_panfrost],
  dependencies : [idep_nir, idep_vtn, idep_mesautil, idep_valhall_enums_h,
                  idep_bi_opcodes_h, libpanfrost_model_dep],
  link_args : _kraid_wrap,
)

# The kbase kernel interface: context, GPU properties, memory. jm.c is the
# job-manager (Mali-G52 and Mali-G57) frontend's atom submission and events,
# alongside csf.c's CSF frontend; both are compiled in unconditionally (the
# frontend is a runtime choice, mali_kbase_create) but only one is ever
# called on a given context.
malisx2_kbase = static_library(
  'mali_kbase',
  files('kbase/cache.c', 'kbase/context.c', 'kbase/csf.c', 'kbase/device.c',
        'kbase/jm.c', 'kbase/kcpu.c', 'kbase/mem.c', 'kbase/props.c'),
  c_args : [pre_args, malisx2_c_args],
  gnu_symbol_visibility : 'hidden',
)

idep_mali_kbase = declare_dependency(
  link_with : malisx2_kbase,
  include_directories : inc_malisx2,
)

# The Vulkan front half: our entry points
# (prefix mali_) and the waist stubs (prefix mali_waist_), both tabled by
# Mesa's generator; the stubs themselves come from waist.txt.
# Device-level commands (vkCmd*, vkQueue*, vkCreate{Sampler,ImageView,
# BufferView,*Pipelines}, vkUpdateDescriptorSets, ...) get a dispatch table
# per GPU architecture, --device-prefix mali_v7/mali_v9/mali_v11, so every
# arch's MALI_PER_ARCH-named functions can sit in the same .so without
# colliding; mali_CreateDevice picks the table that matches the device
# (mali_device.c).
# Instance- and physical-device-level commands keep one shared table under
# --prefix, because an arch is not known until a physical device is probed.
mali_entrypoints = custom_target(
  'mali_entrypoints',
  input : [vk_entrypoints_gen, vk_api_xml],
  output : ['mali_entrypoints.h', 'mali_entrypoints.c'],
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@', '--proto', '--weak',
    '--out-h', '@OUTPUT0@', '--out-c', '@OUTPUT1@',
    '--prefix', 'mali', '--prefix', 'mali_waist',
    '--device-prefix', 'mali_v7', '--device-prefix', 'mali_v9',
    '--device-prefix', 'mali_v11',
    '--beta', with_vulkan_beta.to_string(),
  ],
  depend_files : vk_entrypoints_gen_depend_files,
)

mali_waist_stubs = custom_target(
  'mali_waist_stubs',
  input : [files('vk/gen_waist_stubs.py'), vk_api_xml, files('vk/waist.txt')],
  output : 'mali_waist_stubs.c',
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@', '--waist', '@INPUT2@',
    '--vk-util', mesa_src / 'vulkan' / 'util', '--out-c', '@OUTPUT@',
  ],
  depend_files : vk_entrypoints_gen_depend_files,
)

mali_version_h = vcs_tag(
  input : files('vk/mali_version.h.in'),
  output : 'mali_version.h',
  command : ['git', '-C', meson.project_source_root(), 'rev-parse', '--short=12', 'HEAD'],
  fallback : 'unknown',
)

# pipelineCacheUUID comes from a hash of the sources that can change shader
# binaries (vk/gen_shader_cache_id.py), not from the commit, so a driver
# update that leaves the compiler alone keeps applications' pipeline caches.
# One hash per architecture: a change only the v9 build sees leaves the
# v11 one alone.
# The script runs on every build and rewrites the header only on a change.
mali_shader_cache_id_h = custom_target(
  'mali_shader_cache_id.h',
  output : 'mali_shader_cache_id.h',
  command : [prog_python, files('vk/gen_shader_cache_id.py'),
             '--root', meson.project_source_root(),
             '--arch', '@0@'.format(pan_arch), '--arch', '9', '--arch', '7',
             '--out-h', '@OUTPUT@'],
  build_always_stale : true,
)

_vk_c_args = [
  pre_args, malisx2_c_args,
  '-DMALI_PAN_ARCH=@0@'.format(pan_arch),
  '-DMALI_VERSION_STRING="@0@"'.format(meson.project_version()),
]

# The files built once per GPU architecture:
# descriptor, sampler, view, format-table, shader, blend-shader and
# pipeline packing, the command-buffer builders both back halves share
# (mali_cmd_state.c, mali_fb.c, mali_cmd_meta_gfx.c), the buffer copies
# (mali_cmd_copy.c, on each back half's dispatch) and the image copies,
# clears and blits (mali_cmd_image.c, on each back half's dispatch and
# render pass). Each copy names its
# functions mali_v7_*, mali_v9_* or mali_v11_* (MALI_PER_ARCH, vk/mali_arch.h).
_vk_per_arch_files = files(
  'vk/mali_blend.c', 'vk/mali_cmd_copy.c', 'vk/mali_cmd_image.c', 'vk/mali_cmd_meta_gfx.c',
  'vk/mali_cmd_state.c',
  'vk/mali_descriptor_set.c', 'vk/mali_fb.c', 'vk/mali_format_table.c',
  'vk/mali_image_view.c', 'vk/mali_pipeline.c', 'vk/mali_pipeline_state.c',
  'vk/mali_sampler.c', 'vk/mali_shader.c',
)
_vk_inc = [inc_include, inc_src, inc_malisx2, include_directories('vk'),
           include_directories('measurement')]
_vk_per_arch_deps = [idep_vulkan_lite_runtime_headers, idep_vulkan_util_headers,
                     idep_mesautil, idep_nir_headers, idep_mali_compiler]

# Everything of the front half except the per-arch files and what the waist
# stubs do when called (mali_waist.c), which the driver and the tests pick
# separately. The command-stream (CSF) back half here is the v11 one.
malisx2_vk = static_library(
  'mali_vk',
  files('vk/mali_device.c', 'vk/mali_icd.c', 'vk/mali_instance.c',
        'vk/mali_physical_device.c'),
  # Memory, buffers, images, formats.
  files('vk/mali_buffer.c', 'vk/mali_device_memory.c', 'vk/mali_formats.c',
        'vk/mali_image.c', 'vk/mali_image_layout.c'),
  # Shader lowering and memory pools, descriptor set layouts.
  files('vk/mali_bo_pool.c', 'vk/mali_descriptor_set_layout.c',
        'vk/mali_nir_lower_descriptors.c', 'vk/mali_nir_lower_input_attachments.c'),
  # Command buffers, the CSF queue, submit and sync.
  files('vk/mali_cmd_buffer.c', 'vk/mali_cmd_dispatch.c', 'vk/mali_queue.c',
        'vk/mali_sync.c'),
  # Render passes and draws (the CSF side; the shared builders and the
  # image transfers are per-arch files).
  files('vk/mali_cmd_draw.c', 'vk/mali_cmd_render.c'),
  # Sync files and the Android window system; mali_android.c only on Android.
  files('vk/mali_sync_file.c', 'vk/mali_wsi.c'),
  with_platform_android ? files('vk/mali_android.c') : [],
  # Timing and command-stream capture (MALISX2_MEASURE).
  files('measurement/config.c', 'measurement/recorder.c', 'measurement/timing.c'),
  mali_entrypoints, mali_waist_stubs, mali_version_h, mali_shader_cache_id_h,
  include_directories : [inc_include, inc_src, inc_malisx2, include_directories('vk'),
                         include_directories('measurement')],
  c_args : _vk_c_args,
  dependencies : [idep_vulkan_lite_runtime_headers, idep_vulkan_util_headers,
                  idep_mesautil, idep_nir_headers, idep_pan_format,
                  idep_mali_compiler, idep_pan_packers],
  gnu_symbol_visibility : 'hidden',
)

# The per-arch files at v11, for the G615 (mali_v11_*).
malisx2_vk_v11 = static_library(
  'mali_vk_v11',
  _vk_per_arch_files,
  mali_entrypoints[0],   # the header only; the tables are malisx2_vk's
  include_directories : _vk_inc,
  c_args : [pre_args, malisx2_c_args, '-DPAN_ARCH=11', '-DMALI_PAN_ARCH=11'],
  dependencies : [_vk_per_arch_deps, idep_pan_format_by_arch['11'],
                  idep_pan_packers_by_arch['11']],
  gnu_symbol_visibility : 'hidden',
)

# The files both job-manager back halves build, at PAN_ARCH 7 and 9:
# command buffers, dispatches, render passes, the tiler heap, queries, the
# queue, submit and sync files, and mali_sync.c and mali_wsi.c a second
# time. The draws are per arch (mali_jm_cmd_draw.c at 9,
# mali_jm_cmd_draw_v7.c at 7).
_vk_jm_files = files(
  'vk/mali_jm_cmd_buffer.c', 'vk/mali_jm_cmd_dispatch.c',
  'vk/mali_jm_cmd_render.c', 'vk/mali_jm_heap.c', 'vk/mali_jm_query.c',
  'vk/mali_jm_queue.c', 'vk/mali_jm_sync_file.c', 'vk/mali_sync.c', 'vk/mali_wsi.c',
  # Timing and command-stream capture on the job manager (MALISX2_MEASURE):
  # their own implementation of what touches struct mali_cmd_buffer
  # directly, because that struct (mali_jm.h) differs from the CSF one
  # timing.c/recorder.c are compiled against; the device-level
  # configuration, output writers and mali_measure_harvest/collect in those
  # two files (built once, in malisx2_vk) are shared.
  'measurement/timing_jm.c', 'measurement/recorder_jm.c',
)

# The job-manager (v9, Mali-G57) back half: the job-manager files, its
# draws and the per-arch files, at PAN_ARCH 9. Linked into the driver next
# to malisx2_vk; the device's arch picks which back half a VkDevice uses
# (mali_device.c).
malisx2_vk_v9 = static_library(
  'mali_vk_v9',
  _vk_jm_files, files('vk/mali_jm_cmd_draw.c'),
  _vk_per_arch_files,
  mali_entrypoints[0],   # the header only; the tables are malisx2_vk's
  include_directories : _vk_inc,
  c_args : [pre_args, malisx2_c_args, '-DPAN_ARCH=9',
            '-DMALI_VERSION_STRING="@0@"'.format(meson.project_version())],
  dependencies : [_vk_per_arch_deps, idep_pan_format_by_arch['9'],
                  idep_pan_packers_by_arch['9']],
  gnu_symbol_visibility : 'hidden',
)

# The job-manager back half a second time at PAN_ARCH 7, for the Bifrost
# Mali-G52 (mali_v7_*). The same kbase JM frontend and job-chain model as
# v9; the descriptors are Bifrost's (attribute and varying buffers, the
# Renderer State Descriptor, per-type descriptor tables), after panvk_v7.
malisx2_vk_v7 = static_library(
  'mali_vk_v7',
  _vk_jm_files,
  # The Bifrost draws (a Vertex and a Tiler job each, in place of v9's
  # mali_jm_cmd_draw.c); the descriptor tables and the copy jobs, shared by
  # the v7 draws and dispatches; the Bifrost descriptor lowering.
  files('vk/mali_jm_cmd_draw_v7.c', 'vk/mali_jm_desc_v7.c',
        'vk/mali_nir_lower_descriptors_v7.c'),
  _vk_per_arch_files,
  mali_entrypoints[0],   # the header only; the tables are malisx2_vk's
  include_directories : _vk_inc,
  c_args : [pre_args, malisx2_c_args, '-DPAN_ARCH=7',
            '-DMALI_VERSION_STRING="@0@"'.format(meson.project_version())],
  dependencies : [_vk_per_arch_deps, idep_pan_format_by_arch['7'],
                  idep_pan_packers_by_arch['7']],
  gnu_symbol_visibility : 'hidden',
)

_waist_variants = {}
foreach measure : [false, true]
  _waist_variants += {measure.to_string() : static_library(
    measure ? 'mali_waist_measure' : 'mali_waist',
    files('vk/mali_waist.c'),
    include_directories : [inc_include, inc_src, include_directories('vk')],
    c_args : [_vk_c_args, '-DMALI_MEASUREMENT_BUILD=@0@'.format(measure.to_int())],
    dependencies : [idep_mesautil],
    gnu_symbol_visibility : 'hidden',
  )}
endforeach

# For the driver .so and the tests: the front half, the runtime, kbase, the
# compiler. The waist behaviour is added by the user of the dependency.
idep_mali_vk = declare_dependency(
  link_whole : [malisx2_vk, malisx2_vk_v11, malisx2_vk_v9, malisx2_vk_v7],
  link_with : [malisx2_kbase, libpanfrost_pixel_format_by_arch['11'],
               libpanfrost_pixel_format_by_arch['9'],
               libpanfrost_pixel_format_by_arch['7']],
  sources : [mali_entrypoints[0]],
  include_directories : [inc_malisx2, include_directories('vk'),
                         include_directories('measurement')],
  dependencies : [idep_vulkan_runtime, idep_vulkan_util, idep_mesautil,
                  idep_mali_compiler, idep_pan_packers],
)
malisx2_waist = _waist_variants[get_option('measurement').to_string()]
malisx2_waist_log = _waist_variants['false']
malisx2_waist_abort = _waist_variants['true']

# The driver. Everything is hidden except the loader entry points listed in
# the version script; PT_LOAD segments are aligned for 16 KB pages. The
# build ID feeds the pipeline cache UUID.
# Android also exports HMI, the HAL module the Android loader opens.
_icd_sym = files(with_platform_android ? 'vk/mali_icd_android.sym' : 'vk/mali_icd.sym')
libvulkan_malisx2 = shared_library(
  'vulkan_malisx2',
  dependencies : [idep_mali_vk],
  link_whole : [malisx2_compiler, malisx2_waist],
  link_args : [
    '-Wl,--version-script=@0@'.format(_icd_sym[0].full_path()),
    '-Wl,-z,max-page-size=16384',
    '-Wl,--build-id=sha1',
  ],
  link_depends : _icd_sym,
  gnu_symbol_visibility : 'hidden',
  install : true,
)

# ICD manifest for running against the build tree:
#   VK_DRIVER_FILES=<builddir>/mali_devenv_icd.json
# (or `meson devenv -C <builddir>`, which sets it).
_icd_conf = configuration_data()
_icd_conf.set('LIBRARY_PATH', libvulkan_malisx2.full_path())
mali_devenv_icd = configure_file(
  input : '../mali_icd.json.in',
  output : 'mali_devenv_icd.json',
  configuration : _icd_conf,
)
_icd_conf_installed = configuration_data()
_icd_conf_installed.set('LIBRARY_PATH', 'libvulkan_malisx2.so')
configure_file(
  input : '../mali_icd.json.in',
  output : 'mali_icd.@0@.json'.format(host_machine.cpu_family()),
  configuration : _icd_conf_installed,
  install_dir : get_option('datadir') / 'vulkan' / 'icd.d',
)

devenv = environment()
devenv.set('VK_DRIVER_FILES', meson.current_build_dir() / 'mali_devenv_icd.json')
meson.add_devenv(devenv)
