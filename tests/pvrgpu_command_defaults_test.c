/* SPDX-License-Identifier: MIT */
/* Test the real command submission code with a deterministic API endpoint. */
#include <dlfcn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/gallium/drivers/pvrgpu/pvrgpu_systemc_compute_api.h"

static unsigned submitted_mode, submitted_count;
static int capture_compute(const struct pvrgpu_systemc_compute_dispatch *dispatch,
                           struct pvrgpu_systemc_compute_stats *stats,
                           char *error, size_t error_size)
{
   (void)error; (void)error_size;
   submitted_mode = dispatch->memory_mode;
   ++submitted_count;
   memset(stats, 0, sizeof(*stats));
   return 0;
}
static void *capture_dlopen(const char *path, int flags)
{ (void)path; (void)flags; return (void *)1; }
static void *capture_dlsym(void *handle, const char *name)
{
   (void)handle;
   return strcmp(name, "pvrgpu_systemc_submit_compute") == 0 ?
      (void *)capture_compute : NULL;
}
static char *capture_dlerror(void) { return NULL; }
#define dlopen capture_dlopen
#define dlsym capture_dlsym
#define dlerror capture_dlerror
#include "../src/gallium/drivers/pvrgpu/pvrgpu_cmd.c"
#undef dlopen
#undef dlsym
#undef dlerror

void pvrgpu_counter_eventf(const char *event, const char *format, ...)
{ (void)event; (void)format; }

static unsigned checks;
#define CHECK(condition) do { ++checks; if (!(condition)) { \
   fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #condition); return 1; \
} } while (0)

int main(void)
{
   const float depth_ranges[][2] = {
      {0, 1}, {1, 0}, {.25f, .75f}, {.75f, .25f},
      {0, 0}, {.375f, .375f}, {1, 1},
   };
   for (unsigned i = 0; i < sizeof(depth_ranges) / sizeof(depth_ranges[0]); ++i) {
      const float n = depth_ranges[i][0], f = depth_ranges[i][1];
      float scale[3] = {40, -30, (f - n) * .5f};
      float translate[3] = {40, 30, (f + n) * .5f};
      uint32_t scale_bits[3], translate_bits[3];
      memcpy(scale_bits, scale, sizeof(scale));
      memcpy(translate_bits, translate, sizeof(translate));
      CHECK(pvrgpu_cmd_viewport_scale_matches(scale_bits, 80, 60, true));
      CHECK(pvrgpu_cmd_viewport_depth_range_valid(scale_bits, translate_bits));
      CHECK(pvrgpu_cmd_viewport_scale_matches(scale_bits, 80, 60, false) == (i == 0));
   }
   const float bad_depth[][2] = {
      {.5f, .25f}, {.5f, .75f}, {-.75f, .5f}, {0, -0.125f}, {0, 1.125f},
      {INFINITY, .5f}, {.5f, NAN},
   };
   for (unsigned i = 0; i < sizeof(bad_depth) / sizeof(bad_depth[0]); ++i) {
      float scale[3] = {40, 30, bad_depth[i][0]};
      float translate[3] = {40, 30, bad_depth[i][1]};
      uint32_t scale_bits[3], translate_bits[3];
      memcpy(scale_bits, scale, sizeof(scale));
      memcpy(translate_bits, translate, sizeof(translate));
      CHECK(!pvrgpu_cmd_viewport_depth_range_valid(scale_bits, translate_bits));
   }
   CHECK(setenv("PVRGPU_SYSTEMC_API_LIB", "/test/capture-compute-api", 1) == 0);
   struct pvrgpu_systemc_compute_dispatch dispatch = {0};
   struct pvrgpu_systemc_compute_stats stats;
   char error[128];
   const char *modes[] = {NULL, "", "cache", "direct", "bypass"};
   const unsigned expected[] = {2, 2, 2, 0, 1};
   for (unsigned i = 0; i < sizeof(modes) / sizeof(modes[0]); ++i) {
      if (modes[i]) CHECK(setenv("PVRGPU_MODEL_MEMORY_MODE", modes[i], 1) == 0);
      else CHECK(unsetenv("PVRGPU_MODEL_MEMORY_MODE") == 0);
      const unsigned before = submitted_count;
      CHECK(pvrgpu_submit_compute_command(&dispatch, &stats, error, sizeof(error)));
      CHECK(submitted_count == before + 1 && submitted_mode == expected[i]);
      CHECK(dispatch.memory_mode == 0); /* caller command is immutable */
      CHECK(error[0] == '\0');
      struct pvrgpu_systemc_submit_info graphics;
      pvrgpu_systemc_submit_info_init(&graphics, NULL);
      CHECK(modes[i] && modes[i][0] ?
         graphics.memory_mode && strcmp(graphics.memory_mode, modes[i]) == 0 :
         graphics.memory_mode == NULL); /* NULL graphics mode means cache */
   }
   CHECK(setenv("PVRGPU_MODEL_MEMORY_MODE", "invalid", 1) == 0);
   unsigned before = submitted_count;
   CHECK(!pvrgpu_submit_compute_command(&dispatch, &stats, error, sizeof(error)));
   CHECK(strstr(error, "invalid compute memory_mode") != NULL);
   CHECK(submitted_count == before);
   CHECK(unsetenv("PVRGPU_SYSTEMC_API_LIB") == 0);
   CHECK(!pvrgpu_submit_compute_command(&dispatch, &stats, error, sizeof(error)));
   CHECK(strstr(error, "requires PVRGPU_SYSTEMC_API_LIB") != NULL);
   CHECK(submitted_count == before);
   printf("command defaults: %u checks passed\n", checks);
   return 0;
}
