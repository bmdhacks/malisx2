/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * The option lookup pan_compiler.c is built with (see meson.build here).
 * PAN_USE_KRAID is always "all": on v9 and v11 every stage and internal
 * shaders go through kraid, whatever the environment says (below arch 9
 * pan_use_kraid() does not read the option). Other names go to Mesa's
 * normal lookup.
 */

#include <string.h>

#include "util/u_debug.h"

const char *mali_pan_debug_option_cached(const char *name, const char *dfault);

const char *
mali_pan_debug_option_cached(const char *name, const char *dfault)
{
   if (strcmp(name, "PAN_USE_KRAID") == 0)
      return "all";
   return debug_get_option_cached(name, dfault);
}
