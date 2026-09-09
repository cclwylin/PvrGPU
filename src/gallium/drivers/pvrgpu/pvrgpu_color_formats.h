/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_COLOR_FORMATS_H
#define PVRGPU_COLOR_FORMATS_H
#include <stdint.h>
#include <string.h>

static inline int
pvrgpu_is_explicit_color_format(const char *format)
{
   return format && (!strcmp(format, "PIPE_FORMAT_R8G8B8A8_UNORM") ||
                     !strcmp(format, "PIPE_FORMAT_R10G10B10A2_UNORM") ||
                     !strcmp(format, "PIPE_FORMAT_B10G10R10A2_UNORM"));
}

static inline const char *
pvrgpu_color_formats_error(const char *format, uint32_t targets, uint32_t count,
                          const char *const formats[4])
{
   if (!format || !formats || count > 4 ||
       (count && count != (targets ? targets : 1)))
      return "color attachment format count or common format";
   for (uint32_t target = 0; target < 4; ++target) {
      if (target < count) {
         if (!pvrgpu_is_explicit_color_format(formats[target]))
            return "unsupported explicit color attachment format";
      } else if (formats[target]) {
         return "inactive color attachment format is not NULL";
      }
   }
   if (count && strcmp(format, formats[0]))
      return "color attachment zero/common format mismatch";
   return NULL;
}

/* Use only after validation and with target below the effective RT count. */
static inline const char *
pvrgpu_color_format_at(const char *format, uint32_t count,
                      const char *const formats[4], unsigned target)
{
   return count ? formats[target] : format;
}
#endif
