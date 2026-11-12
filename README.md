# malisx2

A Vulkan driver for Arm Mali GPUs that talks to Arm's **kbase** kernel
driver, the one Android phones and handhelds ship. It is written for one
application, the [ARMSX2](https://github.com/ARMSX2/ARMSX2) PlayStation 2
emulator, which loads it in place of the system driver. It implements what
ARMSX2 and the Android Vulkan loader use; every other entry point logs and
fails.

Target: Mali-G615 (Mesa architecture v11, command-stream frontend), on the
Anbernic RG 477V. The same library serves every GPU of the architectures
it is built for, picked by the GPU's architecture when the device is
created: v11 (Mali-G615, G715), v10 (Mali-G310, G510, G610, G710;
command-stream frontend), v9 (Mali-G57, G68, G77, G78, G78AE; job
manager) and v7 (Mali-G31, G51, G52, G76; job manager). Most of these have
not been tried by us. Per-part values (tile buffer sizes, anisotropic
filtering, the Bifrost compiler's instruction forms) come from Mesa's GPU
model table. Shaders are compiled with Mesa's kraid compiler (Mesa's
Bifrost compiler on v7); the driver is built on Mesa's shared Vulkan
runtime. Midgard, the arch 6 Bifrost parts (G71, G72) and arch 12 on
(G720 and later) are not served.

## Building

The driver builds against Mesa sources checked out at `mesa/`: branch
`malisx2` of [bmdhacks/armsx2-turnip](https://github.com/bmdhacks/armsx2-turnip)
at the commit `meson.build` pins (`mesa_commit`). Only the sources are used;
this repository's `mesa-glue/` builds the parts the driver needs.

    git init mesa
    git -C mesa fetch --depth 1 https://github.com/bmdhacks/armsx2-turnip.git <mesa_commit>
    git -C mesa checkout FETCH_HEAD

Host (Linux):

    meson setup build
    ninja -C build

Needs meson >= 1.7, ninja, a C/C++ compiler, rustc >= 1.85, bindgen >= 0.71.1
(not 0.72.0, 0.73.0 or 0.73.1) and Python 3 with mako and PyYAML.
`meson setup` downloads eight small Rust crates (`subprojects/`).

Android (arm64, API level 29) is a cross build with the NDK r28c at
`~/Android/Sdk/ndk/28.2.13676358` (see `cross/android-aarch64.ini`):

    meson setup build-android --cross-file cross/android-aarch64.ini
    ninja -C build-android
    llvm-strip -o libvulkan_malisx2.so build-android/src/libvulkan_malisx2.so

The result is a Vulkan HAL module (`HMI`) with 16 KiB-aligned segments,
needing only `libc`, `libm`, `libdl` and `liblog`. ARMSX2 loads it as an
adrenotools driver pack: the library plus a `meta.json` in a zip.

ROCKNIX (arm64 Linux, RK3566/RK3568 devices running ROCKNIX's
`mali_kbase`) is a cross build against the toolchain of a ROCKNIX build
tree, with the build machine's clang and lld. Name the tree's toolchain
directory in a machine file of your own and pass it first (see
`cross/rocknix-aarch64.ini`):

    # ~/rocknix-toolchain.ini
    [constants]
    toolchain = '/path/to/distribution/build.ROCKNIX-RK3566.aarch64/toolchain'

    meson setup build-rocknix \
        --cross-file ~/rocknix-toolchain.ini --cross-file cross/rocknix-aarch64.ini \
        --native-file ~/rocknix-toolchain.ini --native-file cross/rocknix-native.ini
    ninja -C build-rocknix

With libclang 22, bindgen 0.71.x generates broken bindings; use 0.72.1.

On Linux the driver presents `VK_KHR_display` swapchains itself, through
Mesa's WSI code built into the library. Wayland swapchains come from Arm's
[vulkan-wsi-layer](https://gitlab.freedesktop.org/mesa/vulkan-wsi-layer)
(MIT licensed), a separate implicit Vulkan layer that the system installs,
as ROCKNIX does. The layer allocates linear dma-bufs from a dma-buf heap and
imports them into the driver (`VK_EXT_image_drm_format_modifier` with
`DRM_FORMAT_MOD_LINEAR`, `VK_EXT_external_memory_dma_buf`); none of its code
is part of this repository or linked into the library.

## Measurement

`MALISX2_MEASURE=timing` (or `draws`, `csf`, `both`, `shaders`, `cpuprof`;
on Android the property `debug.malisx2.measure`) makes the driver write GPU
timestamps per render pass, dispatch and draw, captures of the submitted
command streams, or the shaders it compiles. `cpuprof` writes no files:
each application thread logs, about once a second, how much CPU time it
spent inside the driver and in which entry points. It wraps the device's
entry points, so it adds a little time of its own to every call.

## Rules

- Nothing decompiled goes in: no decompiled code, no Arm binaries.
- License: MIT (`LICENSE`). Mesa is not part of this repository and keeps
  its own licenses.
