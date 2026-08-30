/* SPDX-License-Identifier: MIT */

#include "pvrgpu_counter.h"
#include "pvrgpu_cmd.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

static const char *
pvrgpu_counter_output_path(void)
{
   const char *path = getenv("PVRGPU_DRIVER_COUNTER_OUT");
   if (path && path[0] != '\0')
      return path;
   return NULL;
}

void
pvrgpu_counter_event(const char *event, const char *details)
{
   const char *path = pvrgpu_counter_output_path();
   if (!path)
      return;

   FILE *file = fopen(path, "a");
   if (!file)
      return;

   fprintf(file,
           "schema=%s producer=%s event=%s",
           PVRGPU_DRIVER_COUNTER_SCHEMA,
           PVRGPU_DRIVER_COMMAND_PRODUCER,
           event ? event : "unknown");
   if (details && details[0] != '\0')
      fprintf(file, " %s", details);
   fprintf(file, "\n");
   fclose(file);
}

void
pvrgpu_counter_eventf(const char *event, const char *format, ...)
{
   if (!pvrgpu_counter_output_path())
      return;

   char details[512];
   va_list args;
   va_start(args, format);
   vsnprintf(details, sizeof(details), format ? format : "", args);
   va_end(args);

   pvrgpu_counter_event(event, details);
}
