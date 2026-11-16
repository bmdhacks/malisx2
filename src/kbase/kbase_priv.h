/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/* Helpers shared by the kbase layer's source files. */

#ifndef MALI_KBASE_PRIV_H
#define MALI_KBASE_PRIV_H

#include <errno.h>

#include "kbase.h"

#if defined(__GNUC__)
#define KB_PRINTF(a, b) __attribute__((format(printf, a, b)))
#else
#define KB_PRINTF(a, b)
#endif

/* Log one line through the context's log hook ("malisx2/kbase: " prefix). */
void kb_log(mali_kbase_log_fn fn, void *user, const char *fmt, ...) KB_PRINTF(3, 4);

#define KB_LOG(kb, ...) kb_log((kb)->log, (kb)->log_user, __VA_ARGS__)

/*
 * errno -> result, the blob's three-way split (EBUSY, ENOMEM, other) with
 * ENOMEM made specific: device memory for allocation-type ioctls, host
 * memory for the rest.
 */
enum kb_errno_kind { KB_ERRNO_GENERIC, KB_ERRNO_DEVICE_MEMORY };
enum mali_kbase_result kb_result_from_errno(int err, enum kb_errno_kind kind);

/* ioctl through the backend; returns the ioctl's value or -errno. */
static inline long
kb_ioctl(struct mali_kbase *kb, unsigned long request, void *arg)
{
   int ret = kb->backend->ioctl(kb->backend->priv, kb->fd, request, arg);
   return ret < 0 ? -(long)errno : ret;
}

/* Whether the context has an executable VA zone (MEM_EXEC_INIT): every CSF
 * kernel and job-manager kernels from 11.13 (uapi_jm.h). Without one,
 * shader code is SAME_VA. */
static inline bool
mali_kbase_has_exec_zone(const struct mali_kbase *kb)
{
   return kb->frontend != MALI_KBASE_FRONTEND_JM || kb->uk_minor >= KB_JM_UK_EXEC_INIT_MINOR;
}

/* jm.c: pick the context's atom layout with one stride-72 JOB_SUBMIT
 * (mali_kbase_create, job manager only, last step). */
enum mali_kbase_result kb_jm_probe_atom_layout(struct mali_kbase *kb);

#endif /* MALI_KBASE_PRIV_H */
