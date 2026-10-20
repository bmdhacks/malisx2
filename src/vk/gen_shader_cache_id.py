#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Writes mali_shader_cache_id.h: a hash of every source file that can change
# the shader binaries our pipeline cache stores. pipelineCacheUUID is derived
# from it, so an application's pipeline cache survives a driver update that
# does not touch the compiler, and is thrown away when one does.
#
# There is one hash per GPU architecture the driver is built for (--arch),
# because each GPU only runs shaders built for its own: a file only some
# architectures' builds compile or use (ARCH_ONLY below) is left out of the
# other architectures' hashes, so a change to it does not empty the other
# GPUs' pipeline caches. The first --arch is the one MALI_SHADER_CACHE_ID
# names; each also gets MALI_SHADER_CACHE_ID_V<n>.
#
# Hashing too much only costs a needless recompile; hashing too little lets a
# stale shader binary be reused. So the hash covers the whole source tree and
# leaves out only files that cannot reach a shader: the kernel interface, the
# measurement layer, and the command-buffer, queue, memory and window-system
# code in src/vk. Headers are always included.
#
# The output file is only rewritten when the hash changes, so the target can
# run on every build without forcing a relink.

import argparse
import hashlib
import os

# Directories and top-level files hashed, relative to the source root. Of
# Mesa, only the directories the build compiles from.
ROOTS = ['src', 'mesa-glue', 'cross', 'meson.build', 'meson.options',
         'mesa/include', 'mesa/src/c11', 'mesa/src/compiler', 'mesa/src/panfrost',
         'mesa/src/util', 'mesa/src/vulkan']

# Files under src/ that do not affect shader compilation or the serialized
# shader format. Only .c files and build helpers; headers stay in.
EXCLUDE_DIRS = ['src/kbase', 'src/measurement']
EXCLUDE_FILES = {
    'src/vk/mali_android.c',
    'src/vk/mali_bo_pool.c',
    'src/vk/mali_buffer.c',
    'src/vk/mali_cmd_buffer.c',
    'src/vk/mali_cmd_copy.c',
    'src/vk/mali_cmd_dispatch.c',
    'src/vk/mali_cmd_draw.c',
    'src/vk/mali_cmd_image.c',
    'src/vk/mali_cmd_render.c',
    'src/vk/mali_cmd_state.c',
    'src/vk/mali_descriptor_set.c',
    'src/vk/mali_device_memory.c',
    'src/vk/mali_fb.c',
    'src/vk/mali_format_table.c',
    'src/vk/mali_formats.c',
    'src/vk/mali_icd.c',
    'src/vk/mali_icd.sym',
    'src/vk/mali_icd_android.sym',
    'src/vk/mali_image.c',
    'src/vk/mali_image_layout.c',
    'src/vk/mali_image_view.c',
    'src/vk/mali_instance.c',
    'src/vk/mali_jm_cmd_buffer.c',
    'src/vk/mali_jm_cmd_dispatch.c',
    'src/vk/mali_jm_queue.c',
    'src/vk/mali_jm_sync_file.c',
    'src/vk/mali_queue.c',
    'src/vk/mali_sampler.c',
    'src/vk/mali_sync.c',
    'src/vk/mali_sync_file.c',
    'src/vk/mali_waist.c',
    'src/vk/mali_wsi.c',
    'src/vk/waist.txt',
    'src/vk/gen_waist_stubs.py',
    'src/vk/mali_version.h.in',
    'src/vk/gen_shader_cache_id.py',
}


# Files only some architectures' builds compile or read: a test on the path
# relative to the source root, and those architectures. A file several
# tests match is hashed only for the architectures every one of them names.
# The job-manager back half (mali_jm*) is built at 7 and 9, except its
# draws: mali_jm_cmd_draw.c at 9 and mali_jm_cmd_draw_v7.c at 7. The
# v7-only sources are named *_v7.c and *_v7.h. Shaders take their
# descriptor lowering from mali_nir_lower_descriptors.c on 9 and 11 and
# from mali_nir_lower_descriptors_v7.c on 7 (mali_shader.c).
ARCH_ONLY = [
    (lambda rel: rel.startswith('src/vk/mali_jm'), {7, 9}),
    (lambda rel: rel == 'src/vk/mali_jm_cmd_draw.c', {9}),
    (lambda rel: rel == 'src/vk/mali_nir_lower_descriptors.c', {9, 11}),
    (lambda rel: rel == 'mesa/src/panfrost/genxml/v9.xml', {9}),
    (lambda rel: (rel == 'mesa/src/panfrost/genxml/v7.xml' or
                  rel.endswith(('_v7.c', '_v7.h'))), {7}),
]


def excluded(rel, arch):
    if rel in EXCLUDE_FILES:
        return True
    if '__pycache__' in rel.split('/'):
        return True
    if any(arch not in archs and only(rel) for only, archs in ARCH_ONLY):
        return True
    return any(rel.startswith(d + '/') for d in EXCLUDE_DIRS)


def tree_files(root, arch):
    out = []
    for r in ROOTS:
        p = os.path.join(root, r)
        if os.path.isfile(p):
            out.append(r)
            continue
        for d, dirs, files in os.walk(p):
            dirs.sort()
            for f in files:
                rel = os.path.relpath(os.path.join(d, f), root).replace(os.sep, '/')
                if not excluded(rel, arch):
                    out.append(rel)
    return sorted(out)


def arch_hash(root, arch, extras):
    h = hashlib.sha256()
    for extra in ['pan_arch=%d' % arch] + extras:
        h.update(b'extra\0' + extra.encode() + b'\0')
    for rel in tree_files(root, arch):
        h.update(rel.encode() + b'\0')
        path = os.path.join(root, rel)
        # A symlink inside a tree (Mesa has a few, into its docs/) counts by
        # its target path: the target may not be copied along.
        if os.path.islink(path):
            data = b'symlink\0' + os.readlink(path).encode()
        else:
            with open(path, 'rb') as f:
                data = f.read()
        h.update(len(data).to_bytes(8, 'little'))
        h.update(data)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--arch', type=int, action='append', required=True,
                    help='a GPU architecture (PAN_ARCH) the driver is built for; '
                         'the first is the main one')
    ap.add_argument('--extra', action='append', default=[],
                    help='a string that also goes into every hash (build options)')
    ap.add_argument('--out-h', required=True)
    args = ap.parse_args()

    text = ('/* Generated by gen_shader_cache_id.py: hash of the sources that\n'
            ' * can change shader binaries, per GPU architecture. */\n')
    for i, arch in enumerate(args.arch):
        digest = arch_hash(args.root, arch, args.extra)
        names = ['MALI_SHADER_CACHE_ID_V%d' % arch]
        if i == 0:
            names.insert(0, 'MALI_SHADER_CACHE_ID')
        for name in names:
            text += '#define %s "%s"\n#define %s_SHORT "%s"\n' % (
                name, digest, name, digest[:12])
    try:
        with open(args.out_h) as f:
            if f.read() == text:
                return
    except OSError:
        pass
    with open(args.out_h, 'w') as f:
        f.write(text)


if __name__ == '__main__':
    main()
