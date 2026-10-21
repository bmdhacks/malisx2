/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * VkInstance. Mesa's vk_instance does the work (extension checks, app info,
 * debug utils, physical-device list); we supply the extension table, the
 * version and the kbase enumeration callback.
 */

#include "mali_vk.h"

#ifndef __ANDROID__
#include "wsi_common.h"
#endif

#include "util/u_atomic.h"
#include "vk_alloc.h"

static const struct mali_kbase_backend *kbase_backend_override;

void
mali_instance_set_kbase_backend(const struct mali_kbase_backend *backend)
{
   p_atomic_set(&kbase_backend_override, backend);
}

/*
 * Instance extensions. The blob exposes six; ARMSX2 enables only
 * VK_EXT_debug_utils, and only with its debug settings. The runtime
 * implements debug utils entirely.
 * get_physical_device_properties2 is core in 1.1; listing it is free and
 * helps applications that check for it. The surface extensions come from
 * the Android loader, not from us.
 */
static const struct vk_instance_extension_table mali_instance_extensions = {
   .KHR_get_physical_device_properties2 = true,
   .EXT_debug_utils = true,
#ifndef __ANDROID__
   /* Mesa's shared WSI, VK_KHR_display only for now (mali_wsi_linux.c):
    * the SDL/kmsdrm frontend of ARMSX2 presents straight to the display
    * with it. Android's loader owns surfaces there. */
   .KHR_surface = true,
   .KHR_display = true,
   .KHR_get_display_properties2 = true,
   .EXT_headless_surface = true,
#endif
};

VKAPI_ATTR VkResult VKAPI_CALL
mali_EnumerateInstanceVersion(uint32_t *pApiVersion)
{
   *pApiVersion = MALI_API_VERSION;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_EnumerateInstanceExtensionProperties(const char *pLayerName,
                                          uint32_t *pPropertyCount,
                                          VkExtensionProperties *pProperties)
{
   if (pLayerName)
      return vk_error(NULL, VK_ERROR_LAYER_NOT_PRESENT);

   return vk_enumerate_instance_extension_properties(&mali_instance_extensions,
                                                     pPropertyCount, pProperties);
}

/* Class d in the waist list, but it has an obvious correct answer (no
 * layers), and a loader that gets an error here may give up on the ICD. */
VKAPI_ATTR VkResult VKAPI_CALL
mali_EnumerateInstanceLayerProperties(uint32_t *pPropertyCount,
                                      VkLayerProperties *pProperties)
{
   *pPropertyCount = 0;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
mali_CreateInstance(const VkInstanceCreateInfo *pCreateInfo,
                    const VkAllocationCallbacks *pAllocator,
                    VkInstance *pInstance)
{
   assert(pCreateInfo->sType == VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);

   if (!pAllocator)
      pAllocator = vk_default_allocator();

   struct mali_instance *instance =
      vk_zalloc(pAllocator, sizeof(*instance), 8, VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!instance)
      return vk_error(NULL, VK_ERROR_OUT_OF_HOST_MEMORY);

   /* Driver entry points first; vk_instance_init adds Mesa's common ones
    * where we have none; the waist stubs fill what is still empty. */
   struct vk_instance_dispatch_table dispatch;
   vk_instance_dispatch_table_from_entrypoints(&dispatch, &mali_instance_entrypoints,
                                               true);
#ifndef __ANDROID__
   vk_instance_dispatch_table_from_entrypoints(&dispatch, &wsi_instance_entrypoints,
                                               false);
#endif

   VkResult result = vk_instance_init(&instance->vk, &mali_instance_extensions,
                                      &dispatch, pCreateInfo, pAllocator);
   if (result != VK_SUCCESS) {
      vk_free(pAllocator, instance);
      return result;
   }
   vk_instance_dispatch_table_from_entrypoints(&instance->vk.dispatch_table,
                                               &mali_waist_instance_entrypoints,
                                               false);

   instance->kbase_backend = p_atomic_read(&kbase_backend_override);
   if (!instance->kbase_backend)
      instance->kbase_backend = mali_kbase_os_backend();

   instance->vk.physical_devices.enumerate = mali_physical_devices_enumerate;
   instance->vk.physical_devices.destroy = mali_physical_device_destroy;

   *pInstance = mali_instance_to_handle(instance);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
mali_DestroyInstance(VkInstance _instance, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(mali_instance, instance, _instance);

   if (!instance)
      return;

   vk_instance_finish(&instance->vk);
   vk_free(&instance->vk.alloc, instance);
}

PFN_vkVoidFunction
mali_GetInstanceProcAddr(VkInstance _instance, const char *pName)
{
   VK_FROM_HANDLE(mali_instance, instance, _instance);
   return vk_instance_get_proc_addr(instance ? &instance->vk : NULL,
                                    &mali_instance_entrypoints, pName);
}
