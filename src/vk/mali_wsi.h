/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * What the Android window system needs from the driver, in terms that do
 * not need Android's headers, so the host tests can call it. The Android
 * entry points in mali_android.c unwrap the loader's structures and call
 * these.
 */

#ifndef MALI_WSI_H
#define MALI_WSI_H

#include <stdbool.h>
#include <stdint.h>

#include "vulkan/vulkan_core.h"

struct mali_device;
struct mali_image;
struct mali_physical_device;
struct mali_queue;

/* gralloc usage bits (hardware/gralloc.h, gralloc1.h; the 64-bit gralloc 4
 * BufferUsage has the same values). */
#define MALI_GRALLOC_USAGE_HW_TEXTURE 0x100ull  /* GPU_TEXTURE (consumer) */
#define MALI_GRALLOC_USAGE_HW_RENDER  0x200ull  /* GPU_RENDER_TARGET (producer) */

#define MALI_GRALLOC_USAGE_CPU_READ_RARELY 0x2ull

/*
 * The usage bits that keep gralloc from choosing AFBC for a swapchain
 * image, from the value of ro.vendor.arm.gralloc.no_afbc_usage_flags (NULL
 * if the property could not be read). The property names the vendor usage
 * bit that this device's gralloc takes as "no AFBC". Vendors pick
 * different bits (0x20000000 on Unisoc, 0x0200000000000000 on MediaTek),
 * and a bit gralloc does not know is ignored, which leaves AFBC on.
 *
 * The property is not always readable from an application (SELinux), or
 * set. Then no vendor bit is guessed, since a wrong one is ignored or means
 * something else to another vendor's gralloc. The usage is CPU read
 * instead: Arm's gralloc allocates buffers the CPU can read as linear,
 * whatever the vendor's encoding of "no AFBC" is. *fallback is set when the
 * property gave no usable value (missing, not a number, or 0).
 */
uint64_t mali_wsi_no_afbc_usage(const char *property_value, bool *fallback);

/*
 * mali_wsi_no_afbc_usage for a GPU of this arch. Without a usable property,
 * arch 11 keeps the bit earlier releases used there (MediaTek's gralloc,
 * the G615 devices this driver has shipped on); other GPUs ask for linear
 * buffers by CPU read, which any Arm gralloc honours.
 */
uint64_t mali_wsi_arch_no_afbc_usage(unsigned arch, const char *property_value);

/*
 * The no-AFBC usage bits swapchain buffers get on this system: on Android
 * from ro.vendor.arm.gralloc.no_afbc_usage_flags, read once
 * (mali_android.c); elsewhere there is no property service, so the answer
 * for an unreadable property.
 */
uint64_t mali_wsi_swapchain_no_afbc(unsigned arch);

/*
 * The gralloc usage for swapchain images of this format and image usage
 * (vkGetSwapchainGrallocUsage2ANDROID semantics). As the blob: GPU texture
 * as consumer usage and GPU render target as producer usage whatever the
 * image usage. Unlike the blob, the producer usage always carries
 * no_afbc: we render linear images only, so gralloc must not pick AFBC.
 * VK_ERROR_FORMAT_NOT_SUPPORTED for a format or usage we cannot render or
 * write to linear images with.
 */
VkResult mali_wsi_gralloc_usage(struct mali_physical_device *pdev, VkFormat format,
                                VkImageUsageFlags usage, uint64_t no_afbc,
                                uint64_t *consumer, uint64_t *producer);

/*
 * The answer to vkGetPhysicalDeviceImageFormatProperties2 with the
 * Android-hardware-buffer handle type: the gralloc usage for a buffer
 * behind an image of this description. Android's loader asks this, before
 * or instead of vkGetSwapchainGrallocUsage2ANDROID, for the usage of every
 * swapchain buffer it allocates; the buffers then come back to the driver
 * as VkNativeBufferANDROID. So the answer is what
 * vkGetSwapchainGrallocUsage2ANDROID gives for the same format and usage,
 * consumer and producer usage combined, no-AFBC bits included. Only what
 * a swapchain image can be is supported: 2D, optimal tiling, no create
 * flags but the two a mutable-format swapchain adds.
 */
VkResult mali_wsi_ahb_usage(struct mali_physical_device *pdev,
                            const VkPhysicalDeviceImageFormatInfo2 *info, uint64_t *usage);

/*
 * Give a swapchain image the memory of a gralloc buffer: a linear layout
 * with the buffer's row stride (in pixels, as the loader passes it), the
 * dma-buf imported through kbase (a dup of dmabuf_fd; the caller keeps
 * its fd) and bound. The image owns the import from then on.
 */
VkResult mali_wsi_image_bind_buffer(struct mali_device *dev, struct mali_image *image,
                                    int dmabuf_fd, uint32_t stride_px,
                                    const VkAllocationCallbacks *alloc);

/*
 * The same for a gralloc handle's fds, when it is not known which of them
 * is the buffer: candidates are tried dma-bufs first and big enough ones
 * first (mali_wsi.c has the order), and the first import that works is
 * bound. On failure the log says what each fd is. fds are the caller's;
 * at most 8 are looked at.
 */
VkResult mali_wsi_image_bind_gralloc_fds(struct mali_device *dev, struct mali_image *image,
                                         const int *fds, unsigned nfds, uint32_t stride_px,
                                         const VkAllocationCallbacks *alloc);

/*
 * vkAcquireImageANDROID: the display's release fence (a sync file, or -1
 * for "already signalled") becomes the payload of the semaphore and/or
 * the fence. Consumes fd on every path, success or not.
 */
VkResult mali_wsi_acquire(struct mali_device *dev, int fd, VkSemaphore semaphore,
                          VkFence fence);

/*
 * vkQueueSignalReleaseImageANDROID: one sync file that signals when every
 * semaphore's payload has signalled; the semaphores are waited on
 * (binary semaphore payloads are consumed). *out_fd is -1 when nothing is
 * pending.
 */
VkResult mali_wsi_release(struct mali_queue *queue, uint32_t count,
                          const VkSemaphore *semaphores, int *out_fd);

#ifdef VK_USE_PLATFORM_ANDROID_KHR
#include "vulkan/vk_android_native_buffer.h"

/* A swapchain image created or bound with a VkNativeBufferANDROID
 * (mali_android.c): mali_wsi_image_bind_buffer on the buffer's dma-buf. */
VkResult mali_android_bind_native_buffer(struct mali_device *dev, struct mali_image *image,
                                         const VkNativeBufferANDROID *nb,
                                         const VkAllocationCallbacks *alloc);
#endif

#endif
