# SPDX-License-Identifier: MIT
# Derived from Mesa src/util/meson.build.

_vu = mesa_src / 'util'
inc_util = [inc_include, inc_src]

subdir('blake3')

_util_names = [
  'anon_file.c', 'bitscan.c', 'blob.c', 'build_id.c', 'cnd_monotonic.c',
  'compress.c', 'thread_sched.c', 'crc32.c', 'dag.c', 'disk_cache.c',
  'disk_cache_os.c', 'double.c', 'fast_idiv_by_const.c', 'float8.c',
  'fossilize_db.c', 'futex.c', 'half_float.c', 'hash_table.c', 'helpers.c',
  'u_idalloc.c', 'log.c', 'lut.c', 'memstream.c', 'mesa-blake3.c',
  'os_time.c', 'os_file.c', 'os_file_notify.c', 'os_memory_fd.c',
  'os_misc.c', 'os_socket.c', 'pb_slab.c', 'perf/u_trace.c', 'u_process.c',
  'u_qsort.cpp', 'rwlock.c', 'ralloc.c', 'rand_xor.c',
  'range_minimum_query.c', 'rb_tree.c', 'register_allocate.c', 'rgtc.c',
  'set.c', 'simple_mtx.c', 'slab.c', 'softfloat.c', 'sparse_array.c',
  'string_buffer.c', 'strndup.c', 'strtod.c', 'texcompress_astc_luts.cpp',
  'texcompress_astc_luts_wrap.cpp', 'texcompress_astc.cpp',
  'texcompress_etc.c', 'u_atomic.c',
  'u_call_once.c', 'u_dl.c', 'u_dynarray.c', 'u_hash_table.c', 'u_queue.c',
  'u_range_remap.c', 'u_shader_variant_cache.c', 'u_string.c',
  'u_thread.c', 'u_vector.c', 'u_math.c', 'u_mm.c', 'u_debug.c',
  'u_debug_memory.c', 'u_cpu_detect.c', 'u_printf.c', 'u_worklist.c',
  'vl_zscan_data.c', 'vma.c', 'mesa_cache_db.c', 'mesa_cache_db_multipart.c',
]
if host_machine.cpu_family() == 'aarch64'
  _util_names += 'cache_ops_aarch64.c'
else
  _util_names += 'cache_ops_null.c'
endif
# Mesa builds u_debug_stack_android.cpp on Android, which needs AOSP's
# libbacktrace (not in the NDK). We build the generic files everywhere; on
# Android u_debug_stack.c compiles to nothing, which is fine as long as
# nothing we link calls debug_backtrace_*().
_util_names += ['u_debug_stack.c', 'u_debug_symbol.c']

# On Android, os_misc.c reads Mesa's options as properties as well as from
# the environment, several times per shader compile for some. It is built
# alone with __system_property_get() renamed to mali_os_property_get()
# (os_misc_android_props.c), which skips the vendor.* names the app's SELinux
# domain denies (a logcat line each) and reads every other name once. Mesa's
# source is unchanged.
files_mesa_util = []
foreach n : _util_names
  if not (with_platform_android and n == 'os_misc.c')
    files_mesa_util += files(_vu / n)
  endif
endforeach

format_srgb = custom_target(
  'format_srgb',
  input : files(_vu / 'format_srgb.py'),
  output : 'format_srgb.c',
  command : [prog_python, '@INPUT0@'],
  capture : true,
)

shader_stats_h = custom_target(
  input : files(_vu / 'process_shader_stats.py', _vu / 'shader_stats.rnc',
                _vu / 'shader_stats.xml'),
  output : 'shader_stats.h',
  command : [prog_python, '@INPUT@'],
  capture : true,
)
files_mesa_util += shader_stats_h

subdir('format')
files_mesa_util += files_mesa_format

dep_android_log = dependency('', required : false)
if with_platform_android
  dep_android_log = cc.find_library('log')  # util/log.c uses android/log.h
endif

deps_for_libmesa_util = [
  dep_android_log,
  dep_thread,
  dep_atomic,
  dep_m,
  dep_dl,
  idep_blake3,
  idep_mesautilc11,
  idep_mesautilformat,
]

_libmesa_util_simd = static_library(
  'mesa_util_simd',
  files(_vu / 'streaming-load-memcpy.c'),
  c_args : mesa_c_args,
  include_directories : inc_util,
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

_libmesa_util_os_misc = []
if with_platform_android
  _libmesa_util_os_misc = static_library(
    'mesa_util_os_misc',
    files(_vu / 'os_misc.c', 'os_misc_android_props.c'),
    include_directories : inc_util,
    dependencies : [dep_android_log, dep_thread, idep_mesautilc11],
    c_args : [mesa_c_args, '-D__system_property_get=mali_os_property_get'],
    gnu_symbol_visibility : 'hidden',
    build_by_default : false,
  )
endif

_libmesa_util = static_library(
  'mesa_util',
  [files_mesa_util, format_srgb],
  include_directories : [inc_util,
                         include_directories('../../../mesa/src/util'),
                         include_directories('format'),
                         include_directories('../../../mesa/src/util/format')],
  dependencies : deps_for_libmesa_util,
  link_with : [_libmesa_util_simd, _libmesa_util_os_misc],
  c_args : mesa_c_args,
  cpp_args : mesa_cpp_args,
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

idep_mesautil = declare_dependency(
  sources : [shader_stats_h],
  link_with : _libmesa_util,
  compile_args : ['-DXXH_FORCE_ALIGN_CHECK=0', '-DXXH_FORCE_MEMORY_ACCESS=0'],
  include_directories : inc_util,
  dependencies : deps_for_libmesa_util,
)

subdir('rust')
