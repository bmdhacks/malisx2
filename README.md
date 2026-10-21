# malisx2

A Vulkan driver for Arm Mali GPUs that talks to Arm's **kbase** kernel
driver, the one Android phones and handhelds ship. It is written for one
application, the [ARMSX2](https://github.com/ARMSX2/ARMSX2) PlayStation 2
emulator, which loads it in place of the system driver. It implements what
ARMSX2 and the Android Vulkan loader use; every other entry point logs and
fails.

Target: Mali-G615 (Mesa architecture v11, command-stream frontend), on the
Anbernic RG 477V. Shaders are compiled with Mesa's kraid compiler; the
driver is built on Mesa's shared Vulkan runtime.

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
On Linux the driver presents through Mesa's WSI (`VK_KHR_display`) and
through Arm's WSI layer (Wayland), which imports linear dma-bufs.

## Measurement

`MALISX2_MEASURE=timing` (or `draws`, `csf`, `both`, `shaders`; on Android
the property `debug.malisx2.measure`) makes the driver write GPU timestamps
per render pass, dispatch and draw, captures of the submitted command
streams, or the shaders it compiles.

## Rules

- Nothing decompiled goes in: no decompiled code, no Arm binaries.
- License: MIT (`LICENSE`). Mesa is not part of this repository and keeps
  its own licenses.
