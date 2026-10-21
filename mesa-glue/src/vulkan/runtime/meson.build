# SPDX-License-Identifier: MIT
# Derived from Mesa src/vulkan/runtime/meson.build. The "lite" runtime
# (object bases, dispatch, common entry points, logging, sync framework,
# render-pass emulation, command-buffer and pool bases) and, at the end of
# this file, the pipeline pieces the driver uses (vk_pipeline, vk_shader, shader
# modules, pipeline caches). vk_meta, ETC2 emulation, vk_drm_syncobj.c
# (libdrm) and vk_android.c (the driver has its own Android part) are
# not built.

_vvr = mesa_src / 'vulkan' / 'runtime'

_lite_names = [
  'rmv/vk_rmv_common.c',
  'rmv/vk_rmv_exporter.c',
  'vk_blend.c',
  'vk_buffer.c',
  'vk_buffer_view.c',
  'vk_cmd_copy.c',
  'vk_cmd_enqueue.c',
  'vk_command_buffer.c',
  'vk_command_pool.c',
  'vk_debug_report.c',
  'vk_debug_utils.c',
  'vk_deferred_operation.c',
  'vk_descriptor_set_layout.c',
  'vk_descriptors.c',
  'vk_descriptor_update_template.c',
  'vk_device.c',
  'vk_device_generated_commands.c',
  'vk_device_memory.c',
  'vk_fence.c',
  'vk_framebuffer.c',
  'vk_frame_pacer.c',
  'vk_graphics_state.c',
  'vk_image.c',
  'vk_log.c',
  'vk_meta_object_list.c',
  'vk_object.c',
  'vk_physical_device.c',
  'vk_pipeline_layout.c',
  'vk_query_pool.c',
  'vk_queue.c',
  'vk_render_pass.c',
  'vk_sampler.c',
  'vk_semaphore.c',
  'vk_standard_sample_locations.c',
  'vk_sync.c',
  'vk_sync_binary.c',
  'vk_sync_dummy.c',
  'vk_sync_timeline.c',
  'vk_synchronization.c',
  'vk_video.c',
  'vk_ycbcr_conversion.c',
]
vulkan_lite_runtime_files = []
foreach n : _lite_names
  vulkan_lite_runtime_files += files(_vvr / n)
endforeach

inc_vulkan_runtime = include_directories('.', '../../../../mesa/src/vulkan/runtime')

vulkan_lite_runtime_deps = [
  vulkan_wsi_deps,
  idep_mesautil,
  idep_nir_headers,
  idep_vulkan_util,
]

vk_common_entrypoints = custom_target(
  'vk_common_entrypoints',
  input : [vk_entrypoints_gen, vk_api_xml],
  output : ['vk_common_entrypoints.h', 'vk_common_entrypoints.c'],
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@', '--proto', '--weak',
    '--out-h', '@OUTPUT0@', '--out-c', '@OUTPUT1@', '--prefix', 'vk_common',
    '--beta', with_vulkan_beta.to_string()
  ],
  depend_files : vk_entrypoints_gen_depend_files,
)

vk_cmd_queue = custom_target(
  'vk_cmd_queue',
  input : [vk_cmd_queue_gen, vk_api_xml],
  output : ['vk_cmd_queue.c', 'vk_cmd_queue.h'],
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@',
    '--out-c', '@OUTPUT0@', '--out-h', '@OUTPUT1@',
    '--beta', with_vulkan_beta.to_string()
  ],
  depend_files : vk_cmd_queue_gen_depend_files,
)

vk_cmd_enqueue_entrypoints = custom_target(
  'vk_cmd_enqueue_entrypoints',
  input : [vk_entrypoints_gen, vk_api_xml],
  output : ['vk_cmd_enqueue_entrypoints.h', 'vk_cmd_enqueue_entrypoints.c'],
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@', '--proto', '--weak',
    '--out-h', '@OUTPUT0@', '--out-c', '@OUTPUT1@',
    '--prefix', 'vk_cmd_enqueue', '--prefix', 'vk_cmd_enqueue_unless_primary',
    '--beta', with_vulkan_beta.to_string()
  ],
  depend_files : vk_entrypoints_gen_depend_files,
)

vk_dispatch_trampolines = custom_target(
  'vk_dispatch_trampolines',
  input : [vk_dispatch_trampolines_gen, vk_api_xml],
  output : ['vk_dispatch_trampolines.c', 'vk_dispatch_trampolines.h'],
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@',
    '--out-c', '@OUTPUT0@', '--out-h', '@OUTPUT1@',
    '--beta', with_vulkan_beta.to_string()
  ],
  depend_files : vk_dispatch_trampolines_gen_depend_files,
)

vk_physical_device_features = custom_target(
  'vk_physical_device_features',
  input : [vk_physical_device_features_gen, vk_api_xml],
  output : ['vk_physical_device_features.c', 'vk_physical_device_features.h'],
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@',
    '--out-c', '@OUTPUT0@', '--out-h', '@OUTPUT1@',
    '--beta', with_vulkan_beta.to_string()
  ],
  depend_files : vk_physical_device_features_gen_depend_files,
)

vk_physical_device_properties = custom_target(
  'vk_physical_device_properties',
  input : [vk_physical_device_properties_gen, vk_api_xml],
  output : ['vk_physical_device_properties.c', 'vk_physical_device_properties.h'],
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@',
    '--out-c', '@OUTPUT0@', '--out-h', '@OUTPUT1@',
    '--beta', with_vulkan_beta.to_string()
  ],
  depend_files : vk_physical_device_properties_gen_depend_files,
)

vk_physical_device_spirv_caps = custom_target(
  'vk_physical_device_spirv_caps',
  input : [vk_physical_device_spirv_caps_gen, vk_api_xml],
  output : 'vk_physical_device_spirv_caps.c',
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@',
    '--out-c', '@OUTPUT0@', '--beta', with_vulkan_beta.to_string()
  ],
  depend_files : vk_physical_device_spirv_caps_gen_depend_files,
)

vk_synchronization_helpers = custom_target(
  'vk_synchronization_helpers',
  input : [vk_synchronization_helpers_gen, vk_api_xml],
  output : 'vk_synchronization_helpers.c',
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@',
    '--out-c', '@OUTPUT0@',
    '--beta', with_vulkan_beta.to_string()
  ],
  depend_files : vk_synchronization_helpers_gen_depend_files,
)

vk_format_info = custom_target(
  'vk_format_info',
  input : [files(_vvr / 'vk_format_info_gen.py'), vk_api_xml],
  output : ['vk_format_info.c', 'vk_format_info.h'],
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@',
    '--out-c', '@OUTPUT0@', '--out-h', '@OUTPUT1@'
  ],
)

vulkan_lite_runtime_files += [
  vk_cmd_enqueue_entrypoints,
  vk_cmd_queue,
  vk_common_entrypoints,
  vk_dispatch_trampolines,
  vk_format_info,
  vk_physical_device_features,
  vk_physical_device_properties,
  vk_physical_device_spirv_caps,
  vk_synchronization_helpers,
]

vulkan_lite_runtime_header_gen_deps = declare_dependency(
  sources : [
    spirv_info_h,
    vk_cmd_enqueue_entrypoints[0],
    vk_cmd_queue[1],
    vk_common_entrypoints[0],
    vk_dispatch_trampolines[1],
    vk_format_info[1],
    vk_physical_device_features[1],
    vk_physical_device_properties[1],
  ],
)
vulkan_lite_runtime_deps += vulkan_lite_runtime_header_gen_deps

_vk_runtime_c_args = [mesa_c_args,
                      '-DMESA_VK_LOG=@0@'.format(get_option('debug').to_int())]

_libvulkan_lite_runtime = static_library(
  'vulkan_lite_runtime',
  vulkan_lite_runtime_files,
  include_directories : [inc_include, inc_src, inc_vulkan_runtime],
  dependencies : vulkan_lite_runtime_deps,
  c_args : _vk_runtime_c_args,
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

_libvulkan_lite_instance = static_library(
  'vulkan_lite_instance',
  files(_vvr / 'vk_instance.c'),
  include_directories : [inc_include, inc_src, inc_vulkan_runtime],
  dependencies : vulkan_lite_runtime_deps,
  c_args : [mesa_c_args, '-DVK_LITE_RUNTIME_INSTANCE=1'],
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

idep_vulkan_lite_runtime_headers = declare_dependency(
  sources : [
    spirv_info_h,
    vk_cmd_enqueue_entrypoints[0],
    vk_cmd_queue[1],
    vk_common_entrypoints[0],
    vk_physical_device_features[1],
    vk_physical_device_properties[1],
  ],
  include_directories : inc_vulkan_runtime,
)

# As in Mesa: users link the runtime with --whole-archive, otherwise the
# weak vk_common_* references in the entry-point tables may resolve to NULL.
idep_vulkan_lite_runtime = declare_dependency(
  link_whole : [_libvulkan_lite_runtime, _libvulkan_lite_instance],
  dependencies : [idep_vulkan_lite_runtime_headers, vulkan_lite_runtime_deps],
)

# The rest of the runtime that the driver's pipelines use: SPIR-V to NIR
# for pipeline stages, shader modules, pipeline caches and the vk_pipeline
# base object. vk_meta and ETC2 emulation are still not built. The instance
# is rebuilt without VK_LITE_RUNTIME_INSTANCE so it holds a reference on
# the GLSL type singleton that NIR needs, as in Mesa's full runtime.
_runtime_names = [
  'vk_nir.c',
  'vk_nir_lower_descriptor_heaps.c',
  'vk_pipeline.c',
  'vk_pipeline_cache.c',
  'vk_shader.c',
  'vk_shader_module.c',
]
vulkan_runtime_files = []
foreach n : _runtime_names
  vulkan_runtime_files += files(_vvr / n)
endforeach

vulkan_runtime_deps = [vulkan_lite_runtime_deps, idep_nir, idep_vtn]

_libvulkan_runtime = static_library(
  'vulkan_runtime',
  vulkan_runtime_files,
  include_directories : [inc_include, inc_src, inc_vulkan_runtime],
  dependencies : vulkan_runtime_deps,
  c_args : _vk_runtime_c_args,
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

_libvulkan_instance = static_library(
  'vulkan_instance',
  files(_vvr / 'vk_instance.c'),
  include_directories : [inc_include, inc_src, inc_vulkan_runtime],
  dependencies : vulkan_runtime_deps,
  c_args : [mesa_c_args, '-DVK_LITE_RUNTIME_INSTANCE=0'],
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

idep_vulkan_runtime_headers = idep_vulkan_lite_runtime_headers

# Replaces idep_vulkan_lite_runtime; link only one of the two.
idep_vulkan_runtime = declare_dependency(
  link_whole : [_libvulkan_lite_runtime, _libvulkan_runtime, _libvulkan_instance],
  dependencies : [idep_vulkan_runtime_headers, vulkan_runtime_deps],
)
