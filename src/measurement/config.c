/*
 * SPDX-FileCopyrightText: 2026 ARMSX2 and bmdhacks
 * SPDX-License-Identifier: MIT
 */

/*
 * Measurement configuration, the device-level state, the output directory
 * and its manifest.
 *
 * Output directory layout:
 *
 *   manifest.json        what was measured and where it is; rewritten when
 *                        a file is added or rotated out and at the end
 *   timing-NNNN.csv      timing rows, `chunk` submits per file
 *   csf-SSSSSSSS.bin     one capture per submit (recorder.c)
 *
 * With ring=N at most N timing files and N captures are kept: the oldest
 * is deleted when a new one would exceed that, and the manifest marks it
 * deleted. Files are written with stdio (fopen and friends) so that
 * libadrenotools' file redirection can move them inside an app.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "mali_measure.h"

#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "util/simple_mtx.h"

#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif

#include "util/log.h"
#include "util/macros.h"

#include "kbase/kbase.h"
#include "mali_queue.h"
#include "mali_version.h"
#include "mali_vk.h"

/* ---------------------------------------------------------------------- */
/* Configuration                                                           */

static void
config_defaults(struct mali_measure_config *c)
{
   memset(c, 0, sizeof(*c));
#ifdef __ANDROID__
   /* The shell user's scratch area; an app sets dir= to its own. */
   snprintf(c->dir, sizeof(c->dir), "/data/local/tmp/malisx2-measure");
#else
   snprintf(c->dir, sizeof(c->dir), "malisx2-measure");
#endif
   c->chunk = 256;
   c->ring = 64;
   c->start = 1;
   c->csf_max = 16;
}

static bool
parse_u64(const char *s, size_t len, uint64_t *out)
{
   if (!len || len > 20)
      return false;
   uint64_t v = 0;
   for (size_t i = 0; i < len; i++) {
      if (s[i] < '0' || s[i] > '9')
         return false;
      v = v * 10 + (uint64_t)(s[i] - '0');
   }
   *out = v;
   return true;
}

static bool
copy_str(char *dst, size_t size, const char *s, size_t len)
{
   if (!len || len >= size)
      return false;
   memcpy(dst, s, len);
   dst[len] = 0;
   return true;
}

bool
mali_measure_config_parse(const char *str, struct mali_measure_config *c,
                          const char **err)
{
   config_defaults(c);
   *err = NULL;
   if (!str)
      return true;

   const char *p = str;
   while (*p) {
      size_t len = strcspn(p, ",");
      const char *eq = memchr(p, '=', len);
      const char *word = p;
      bool ok = true;

      if (!eq) {
         if (len == 0) {
            /* ",," or a trailing comma */
         } else if (len == 6 && !strncmp(p, "timing", 6)) {
            c->timing = true;
         } else if (len == 5 && !strncmp(p, "draws", 5)) {
            c->timing = c->draws = true;
         } else if (len == 3 && !strncmp(p, "csf", 3)) {
            c->csf = true;
         } else if (len == 4 && !strncmp(p, "both", 4)) {
            c->timing = c->csf = true;
         } else if (len == 7 && !strncmp(p, "shaders", 7)) {
            c->shaders = true;
         } else if (len == 3 && !strncmp(p, "off", 3)) {
            c->timing = c->draws = c->csf = c->shaders = false;
         } else {
            ok = false;
         }
      } else {
         const size_t klen = eq - p;
         const char *v = eq + 1;
         const size_t vlen = len - klen - 1;
         uint64_t n = 0;
#define KEY(k) (klen == sizeof(k) - 1 && !strncmp(p, k, klen))
         if (KEY("dir")) {
            ok = copy_str(c->dir, sizeof(c->dir), v, vlen);
         } else if (KEY("trigger")) {
            ok = copy_str(c->trigger, sizeof(c->trigger), v, vlen);
         } else if (KEY("chunk")) {
            ok = parse_u64(v, vlen, &n) && n > 0 && n <= UINT32_MAX;
            c->chunk = (uint32_t)n;
         } else if (KEY("ring")) {
            ok = parse_u64(v, vlen, &n) && n <= UINT32_MAX;
            c->ring = (uint32_t)n;
         } else if (KEY("start")) {
            ok = parse_u64(v, vlen, &n);
            c->start = n;
         } else if (KEY("count")) {
            ok = parse_u64(v, vlen, &n);
            c->count = n;
         } else if (KEY("csfmax")) {
            ok = parse_u64(v, vlen, &n) && n <= UINT32_MAX;
            c->csf_max = (uint32_t)n;
         } else {
            ok = false;
         }
#undef KEY
      }
      if (!ok) {
         *err = word;
         return false;
      }
      p += len;
      if (*p == ',')
         p++;
   }
   return true;
}

static bool
config_read(struct mali_measure_config *c)
{
   const char *s = getenv("MALISX2_MEASURE");
#ifdef __ANDROID__
   char prop[PROP_VALUE_MAX] = "";
   if (!s || !*s) {
      if (__system_property_get("debug.malisx2.measure", prop) > 0)
         s = prop;
   }
#endif
   if (!s || !*s)
      return false;

   const char *err;
   if (!mali_measure_config_parse(s, c, &err)) {
      mesa_loge("malisx2: measurement off: cannot parse \"%s\" at \"%s\"", s, err);
      return false;
   }
   return true;
}

bool
mali_measure_config_get(struct mali_measure_config *c)
{
   return config_read(c) && (c->timing || c->csf);
}

/* ---------------------------------------------------------------------- */
/* Clock                                                                   */

static uint64_t
read_cntfrq(void)
{
#if defined(__aarch64__)
   uint64_t v;
   __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(v));
   return v;
#else
   return 0;
#endif
}

/*
 * The rate of the GPU timestamp. kbase converts GPU timestamps with the
 * CPU's architected timer frequency (CNTFRQ_EL0, mali_kbase_time.c), so
 * that is the expected rate; two GET_CPU_GPU_TIMEINFO samples 5 ms apart
 * check it. When the two disagree by more than 1 % (or CNTFRQ_EL0 cannot
 * be read) the measured rate is used and the manifest says so.
 */
static void
clock_init(struct mali_measure *m)
{
   struct mali_kbase *kb = m->dev->kbase;
   uint64_t mono0, ts0, mono1, ts1;

   m->cntfrq = read_cntfrq();
   if (mali_kbase_timeinfo(kb, &mono0, &ts0) == MALI_KBASE_SUCCESS) {
      const struct timespec d = {.tv_nsec = 5 * 1000 * 1000};
      nanosleep(&d, NULL);
      if (mali_kbase_timeinfo(kb, &mono1, &ts1) == MALI_KBASE_SUCCESS &&
          mono1 > mono0 && ts1 > ts0) {
         m->measured_hz =
            (uint64_t)((double)(ts1 - ts0) * 1e9 / (double)(mono1 - mono0) + 0.5);
         m->cal_ts = ts1;
         m->cal_mono = mono1;
         m->cal_ok = true;
      }
   }

   if (m->cntfrq && (!m->measured_hz ||
                     llabs((int64_t)m->measured_hz - (int64_t)m->cntfrq) * 100 <
                        (int64_t)m->cntfrq)) {
      m->hz = m->cntfrq;
      m->hz_source = "cntfrq_el0";
   } else if (m->measured_hz) {
      m->hz = m->measured_hz;
      m->hz_source = "measured";
      if (m->cntfrq)
         mesa_logi("malisx2: measurement: GPU timestamps run at %" PRIu64
                   " Hz, not CNTFRQ_EL0's %" PRIu64 " Hz; using the measured rate",
                   m->measured_hz, m->cntfrq);
   } else {
      /* Nothing to go on: the RG 477V's timer rate. Rows still order
       * correctly; durations may be off (the manifest says so). */
      m->hz = 13000000;
      m->hz_source = "assumed";
   }
}

/* ---------------------------------------------------------------------- */
/* Files                                                                   */

void
mali_measure_path(const struct mali_measure *m, const char *name, char *out, size_t size)
{
   snprintf(out, size, "%s/%s", m->cfg.dir, name);
}

static bool
make_dirs(const char *path)
{
   char buf[512];
   if (snprintf(buf, sizeof(buf), "%s", path) >= (int)sizeof(buf))
      return false;
   for (char *p = buf + 1; *p; p++) {
      if (*p != '/')
         continue;
      *p = 0;
      if (mkdir(buf, 0775) < 0 && errno != EEXIST)
         return false;
      *p = '/';
   }
   return mkdir(buf, 0775) == 0 || errno == EEXIST;
}

bool
mali_measure_window(struct mali_measure *m, uint64_t seqno)
{
   if (seqno < m->cfg.start)
      return false;
   if (m->cfg.count && seqno - m->cfg.start >= m->cfg.count)
      return false;
   if (m->cfg.trigger[0] && access(m->cfg.trigger, F_OK) != 0)
      return false;
   return true;
}

static void
write_failed(struct mali_measure *m, const char *what)
{
   if (!m->write_failed)
      mesa_loge("malisx2: measurement: cannot write %s in %s: %s", what, m->cfg.dir,
                strerror(errno));
   m->write_failed = true;
}

/* Delete the oldest live file of one kind when there are more than
 * ring of them. */
static void
ring_trim(struct mali_measure *m, bool capture)
{
   uint32_t *live = capture ? &m->live_captures : &m->live_timing;
   if (!m->cfg.ring || *live <= m->cfg.ring)
      return;
   util_dynarray_foreach(&m->files, struct mali_measure_file, f) {
      if (f->capture != capture || f->deleted)
         continue;
      char path[600];
      mali_measure_path(m, f->name, path, sizeof(path));
      unlink(path);
      f->deleted = true;
      (*live)--;
      if (*live <= m->cfg.ring)
         break;
   }
}

static struct mali_measure_file *
open_timing_file(struct mali_measure *m, uint64_t seqno)
{
   struct mali_measure_file f = {.first_submit = seqno, .last_submit = seqno};
   snprintf(f.name, sizeof(f.name), "timing-%04u.csv", m->timing_index++);
   char path[600];
   mali_measure_path(m, f.name, path, sizeof(path));
   m->timing = fopen(path, "w");
   if (!m->timing) {
      write_failed(m, f.name);
      return NULL;
   }
   fprintf(m->timing, "submit,cmd,subqueue,kind,index,begin_ticks,end_ticks,"
                      "begin_ns,end_ns,dur_ns,info\n");
   util_dynarray_append_typed(&m->files, struct mali_measure_file, f);
   m->live_timing++;
   ring_trim(m, false);
   mali_measure_write_manifest(m);
   return util_dynarray_top_ptr(&m->files, struct mali_measure_file);
}

static struct mali_measure_file *
current_timing_file(struct mali_measure *m)
{
   for (int i = (int)util_dynarray_num_elements(&m->files, struct mali_measure_file) - 1;
        i >= 0; i--) {
      struct mali_measure_file *f =
         util_dynarray_element(&m->files, struct mali_measure_file, i);
      if (!f->capture)
         return f;
   }
   return NULL;
}

void
mali_measure_rotate(struct mali_measure *m, uint64_t seqno)
{
   if (m->timing && seqno >= m->timing_first + m->cfg.chunk) {
      fclose(m->timing);
      m->timing = NULL;
   }
}

static const char *const kind_names[MALI_MEASURE_KIND_COUNT] = {
   [MALI_MEASURE_PASS_VT] = "pass",
   [MALI_MEASURE_PASS_FRAG] = "pass",
   [MALI_MEASURE_DISPATCH] = "dispatch",
   [MALI_MEASURE_DRAW] = "draw",
};

static const char *const sq_names[MALI_SUBQUEUE_COUNT] = {
   [MALI_SUBQUEUE_VERTEX_TILER] = "vt",
   [MALI_SUBQUEUE_FRAGMENT] = "frag",
   [MALI_SUBQUEUE_COMPUTE] = "compute",
};

/*
 * A job-manager region's mali_measure_region::sq is MALI_JM_SLOT_FRAG
 * (0) or MALI_JM_SLOT_VTC (1) (mali_jm.h), not an enum mali_subqueue
 * value, and these rows use "vtc"/"frag" labels on JM rather than the
 * CSF names above. This file builds once, at this build's fixed v11
 * MALI_PAN_ARCH, so it cannot include the job-manager mali_jm.h; the two
 * slot values 0 and 1 are repeated here as plain indices instead of
 * the symbolic ones mali_jm.h defines.
 */
static const char *const jm_sq_names[2] = {"frag", "vtc"};

static uint64_t
ticks_to_mono(const struct mali_measure *m, uint64_t ticks)
{
   const int64_t d = (int64_t)(ticks - m->cal_ts);
   return m->cal_mono + (int64_t)((double)d * 1e9 / (double)m->hz);
}

void
mali_measure_row(struct mali_measure *m, const struct mali_measure_cmd *mc,
                 const struct mali_measure_region *r)
{
   if (!m->timing) {
      m->timing_first = mc->submit - (mc->submit - 1) % m->cfg.chunk;
      if (!open_timing_file(m, mc->submit))
         return;
   }
   struct mali_measure_file *f = current_timing_file(m);

   const uint64_t b = r->slot[0];
   uint64_t e = r->slot[1];
   char info[128];
   switch (r->kind) {
   case MALI_MEASURE_PASS_VT:
   case MALI_MEASURE_PASS_FRAG:
      snprintf(info, sizeof(info), "%ux%u rt=%u draws=%u", r->info[0], r->info[1],
               r->info[2], r->info[3]);
      break;
   case MALI_MEASURE_DISPATCH:
      snprintf(info, sizeof(info), "groups=%ux%ux%u local=%u shader=%016" PRIx64,
               r->info[0], r->info[1], r->info[2], r->info[3], r->shader);
      break;
   case MALI_MEASURE_DRAW:
      snprintf(info, sizeof(info), "count=%u inst=%u %s pass=%u fs=%016" PRIx64 " bs=%x",
               r->info[0], r->info[1], r->info[2] ? "indexed" : "direct", r->info[3],
               r->shader, r->extra);
      break;
   default:
      info[0] = 0;
      break;
   }

   const char *sq_name = m->dev->jm ? jm_sq_names[r->sq] : sq_names[r->sq];

   if (b && e) {
      /* The two deferred stores of a region with (almost) no work of its
       * own can complete in either order; seen on the device as an end a
       * few ticks before the begin. Such a region took no measurable time:
       * duration 0, marked "reordered". */
      const bool reordered = e < b;
      if (reordered) {
         e = b;
         m->rows_reordered++;
      }
      const uint64_t bn = ticks_to_mono(m, b), en = ticks_to_mono(m, e);
      fprintf(m->timing, "%" PRIu64 ",%u,%s,%s,%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64
                         ",%" PRIu64 ",%" PRIu64 ",%s%s\n",
              mc->submit, mc->submit_pos, sq_name, kind_names[r->kind],
              r->index, b, r->slot[1], bn, en, en - bn, reordered ? "reordered " : "", info);
   } else {
      /* Not written (the submit failed, or the device was lost). */
      fprintf(m->timing, "%" PRIu64 ",%u,%s,%s,%u,%" PRIu64 ",%" PRIu64 ",,,,missing %s\n",
              mc->submit, mc->submit_pos, sq_name, kind_names[r->kind],
              r->index, b, e, info);
      m->rows_missing++;
   }
   m->rows++;
   if (f) {
      f->rows++;
      f->first_submit = MIN2(f->first_submit, mc->submit);
      f->last_submit = MAX2(f->last_submit, mc->submit);
   }
}

void
mali_measure_add_capture(struct mali_measure *m, const char *name, uint64_t seqno,
                         uint64_t bytes)
{
   struct mali_measure_file f = {
      .capture = true,
      .first_submit = seqno,
      .last_submit = seqno,
      .rows = bytes,
   };
   snprintf(f.name, sizeof(f.name), "%s", name);
   util_dynarray_append_typed(&m->files, struct mali_measure_file, f);
   m->live_captures++;
   m->captures++;
   ring_trim(m, true);
   mali_measure_write_manifest(m);
}

static void
json_str(FILE *fp, const char *s)
{
   fputc('"', fp);
   for (; *s; s++) {
      if (*s == '"' || *s == '\\')
         fprintf(fp, "\\%c", *s);
      else if ((unsigned char)*s < 0x20)
         fprintf(fp, "\\u%04x", *s);
      else
         fputc(*s, fp);
   }
   fputc('"', fp);
}

void
mali_measure_write_manifest(struct mali_measure *m)
{
   char path[600], tmp[610];
   mali_measure_path(m, "manifest.json", path, sizeof(path));
   snprintf(tmp, sizeof(tmp), "%s.tmp", path);
   FILE *fp = fopen(tmp, "w");
   if (!fp) {
      write_failed(m, "manifest.json");
      return;
   }

   fprintf(fp, "{\n  \"format\": \"malisx2-measure 1\",\n");
   fprintf(fp, "  \"driver\": \"%s %s\",\n", MALI_VERSION_STRING, MALI_GIT_SHA);
   fprintf(fp, "  \"gpu_product_id\": \"0x%04x\",\n", (unsigned)m->dev->kbase->props.product_id);
   fprintf(fp, "  \"pid\": %d,\n  \"config\": ", (int)getpid());
   json_str(fp, m->cfg_str);
   fprintf(fp, ",\n  \"timestamp_hz\": %" PRIu64 ",\n  \"timestamp_hz_source\": \"%s\",\n",
           m->hz, m->hz_source);
   fprintf(fp, "  \"cntfrq_el0\": %" PRIu64 ",\n  \"measured_hz\": %" PRIu64 ",\n",
           m->cntfrq, m->measured_hz);
   fprintf(fp, "  \"calibration\": [");
   if (m->cal_ok)
      fprintf(fp, "{\"gpu_ticks\": %" PRIu64 ", \"mono_ns\": %" PRIu64 "}", m->cal_ts,
              m->cal_mono);
   if (m->end_ok)
      fprintf(fp, ", {\"gpu_ticks\": %" PRIu64 ", \"mono_ns\": %" PRIu64 "}", m->end_ts,
              m->end_mono);
   fprintf(fp, "],\n");
   fprintf(fp, "  \"submits\": %" PRIu64 ",\n  \"rows\": %" PRIu64
               ",\n  \"rows_missing\": %" PRIu64 ",\n  \"rows_reordered\": %" PRIu64
               ",\n  \"dropped\": %" PRIu64 ",\n",
           m->submits, m->rows, m->rows_missing, m->rows_reordered, m->dropped);
   fprintf(fp, "  \"files\": [");
   bool first = true;
   util_dynarray_foreach(&m->files, struct mali_measure_file, f) {
      fprintf(fp, "%s\n    {\"kind\": \"%s\", \"file\": \"%s\", \"first_submit\": %" PRIu64
                  ", \"last_submit\": %" PRIu64 ", \"%s\": %" PRIu64 ", \"deleted\": %s}",
              first ? "" : ",", f->capture ? "csf" : "timing", f->name, f->first_submit,
              f->last_submit, f->capture ? "bytes" : "rows", f->rows,
              f->deleted ? "true" : "false");
      first = false;
   }
   fprintf(fp, "\n  ],\n  \"events\": [");
   first = true;
   util_dynarray_foreach(&m->events, char *, e) {
      fprintf(fp, "%s\n    ", first ? "" : ",");
      json_str(fp, *e);
      first = false;
   }
   fprintf(fp, "\n  ]\n}\n");
   bool ok = fclose(fp) == 0;
   if (!ok || rename(tmp, path) != 0)
      write_failed(m, "manifest.json");
}

/* ---------------------------------------------------------------------- */
/* Device                                                                  */

void
mali_measure_init(struct mali_device *dev)
{
   struct mali_measure_config cfg;
   if (!mali_measure_config_get(&cfg))
      return;
   if (!make_dirs(cfg.dir)) {
      mesa_loge("malisx2: measurement off: cannot create %s: %s", cfg.dir, strerror(errno));
      return;
   }

   struct mali_measure *m = calloc(1, sizeof(*m));
   if (!m)
      return;
   m->dev = dev;
   m->cfg = cfg;
   const char *s = getenv("MALISX2_MEASURE");
#ifdef __ANDROID__
   char prop[PROP_VALUE_MAX] = "";
   if (!s || !*s) {
      __system_property_get("debug.malisx2.measure", prop);
      s = prop;
   }
#endif
   snprintf(m->cfg_str, sizeof(m->cfg_str), "%s", s ? s : "");
   pthread_mutex_init(&m->lock, NULL);
   list_inithead(&m->pending);
   util_dynarray_init(&m->files, NULL);
   util_dynarray_init(&m->events, NULL);
   clock_init(m);

   pthread_mutex_lock(&m->lock);
   mali_measure_write_manifest(m);
   pthread_mutex_unlock(&m->lock);
   dev->measure = m;
   mesa_logi("malisx2: measurement on (%s) into %s, timestamps at %" PRIu64 " Hz (%s)",
             m->cfg_str, cfg.dir, m->hz, m->hz_source);
}

void
mali_measure_finish(struct mali_device *dev)
{
   struct mali_measure *m = dev->measure;
   if (!m)
      return;

   pthread_mutex_lock(&m->lock);
   mali_measure_collect(m, true);
   if (mali_kbase_timeinfo(dev->kbase, &m->end_mono, &m->end_ts) == MALI_KBASE_SUCCESS)
      m->end_ok = true;
   if (m->timing) {
      fclose(m->timing);
      m->timing = NULL;
   }
   mali_measure_write_manifest(m);
   pthread_mutex_unlock(&m->lock);

   util_dynarray_foreach(&m->events, char *, e)
      free(*e);
   util_dynarray_fini(&m->events);
   util_dynarray_fini(&m->files);
   pthread_mutex_destroy(&m->lock);
   free(m);
   dev->measure = NULL;
}

void
mali_measure_event(struct mali_device *dev, const char *fmt, ...)
{
   struct mali_measure *m = dev->measure;
   if (!m)
      return;
   char *s = NULL;
   va_list ap;
   va_start(ap, fmt);
   int n = vasprintf(&s, fmt, ap);
   va_end(ap);
   if (n < 0)
      return;
   pthread_mutex_lock(&m->lock);
   util_dynarray_append_typed(&m->events, char *, s);
   mali_measure_write_manifest(m);
   pthread_mutex_unlock(&m->lock);
}

const char *
mali_measure_shader_dir(void)
{
   static char dir[600];
   static bool ok;
   static simple_mtx_t lock = SIMPLE_MTX_INITIALIZER;
   static bool read;

   simple_mtx_lock(&lock);
   if (!read) {
      struct mali_measure_config c;
      read = true;
      if (config_read(&c) && c.shaders) {
         snprintf(dir, sizeof(dir), "%s/shaders", c.dir);
         ok = make_dirs(dir);
         if (!ok)
            mesa_loge("malisx2: cannot create %s", dir);
      }
   }
   simple_mtx_unlock(&lock);
   return ok ? dir : NULL;
}
