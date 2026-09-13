/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_COLOR_FORMATS_H
#define PVRGPU_COLOR_FORMATS_H
#include <stdint.h>
#include <string.h>

static inline int
pvrgpu_is_explicit_color_format(const char *format)
{
   return format && (!strcmp(format, "PIPE_FORMAT_R8G8B8A8_UNORM") ||
                     !strcmp(format, "PIPE_FORMAT_R8_UNORM") ||
                     !strcmp(format, "PIPE_FORMAT_R8G8_UNORM") ||
                     !strcmp(format, "PIPE_FORMAT_R8G8B8X8_UNORM") ||
                     !strcmp(format, "PIPE_FORMAT_B8G8R8X8_UNORM") ||
                     !strcmp(format, "PIPE_FORMAT_R5G6B5_UNORM") ||
                     !strcmp(format, "PIPE_FORMAT_B5G6R5_UNORM") ||
                     !strcmp(format, "PIPE_FORMAT_R16_UNORM") ||
                     !strcmp(format, "PIPE_FORMAT_R16G16_UNORM") ||
                     !strcmp(format, "PIPE_FORMAT_R16G16B16A16_UNORM") ||
                     !strcmp(format, "PIPE_FORMAT_R32G32B32A32_UNORM") ||
                     !strcmp(format, "PIPE_FORMAT_R8_UINT") ||
                     !strcmp(format, "PIPE_FORMAT_R8G8_UINT") ||
                     !strcmp(format, "PIPE_FORMAT_R8G8B8A8_UINT") ||
                     !strcmp(format, "PIPE_FORMAT_R16_UINT") ||
                     !strcmp(format, "PIPE_FORMAT_R16G16_UINT") ||
                     !strcmp(format, "PIPE_FORMAT_R16G16B16A16_UINT") ||
                     !strcmp(format, "PIPE_FORMAT_R10G10B10A2_UINT") ||
                     !strcmp(format, "PIPE_FORMAT_B10G10R10A2_UINT") ||
                     !strcmp(format, "PIPE_FORMAT_R8_SINT") ||
                     !strcmp(format, "PIPE_FORMAT_R8G8_SINT") ||
                     !strcmp(format, "PIPE_FORMAT_R8G8B8A8_SINT") ||
                     !strcmp(format, "PIPE_FORMAT_R16_SINT") ||
                     !strcmp(format, "PIPE_FORMAT_R16G16_SINT") ||
                     !strcmp(format, "PIPE_FORMAT_R16G16B16A16_SINT") ||
                     !strcmp(format, "PIPE_FORMAT_R8_SNORM") ||
                     !strcmp(format, "PIPE_FORMAT_R8G8_SNORM") ||
                     !strcmp(format, "PIPE_FORMAT_R8G8B8A8_SNORM") ||
                     !strcmp(format, "PIPE_FORMAT_R16_SNORM") ||
                     !strcmp(format, "PIPE_FORMAT_R16G16_SNORM") ||
                     !strcmp(format, "PIPE_FORMAT_R16G16B16A16_SNORM") ||
                     !strcmp(format, "PIPE_FORMAT_R11G11B10_FLOAT") ||
                     !strcmp(format, "PIPE_FORMAT_R16_FLOAT") ||
                     !strcmp(format, "PIPE_FORMAT_R16G16_FLOAT") ||
                     !strcmp(format, "PIPE_FORMAT_R16G16B16A16_FLOAT") ||
                     !strcmp(format, "PIPE_FORMAT_R32_FLOAT") ||
                     !strcmp(format, "PIPE_FORMAT_R32G32_FLOAT") ||
                     !strcmp(format, "PIPE_FORMAT_R8G8B8A8_SRGB") ||
                     !strcmp(format, "PIPE_FORMAT_B8G8R8A8_SRGB") ||
                     !strcmp(format, "PIPE_FORMAT_R10G10B10A2_UNORM") ||
                     !strcmp(format, "PIPE_FORMAT_B10G10R10A2_UNORM") ||
                     !strcmp(format, "PIPE_FORMAT_R32_UINT") ||
                     !strcmp(format, "PIPE_FORMAT_R32_SINT") ||
                     !strcmp(format, "PIPE_FORMAT_R32G32_UINT") ||
                     !strcmp(format, "PIPE_FORMAT_R32G32_SINT") ||
                     !strcmp(format, "PIPE_FORMAT_R32G32B32A32_UINT") ||
                     !strcmp(format, "PIPE_FORMAT_R32G32B32A32_SINT") ||
                     !strcmp(format, "PIPE_FORMAT_R32G32B32A32_FLOAT"));
}

static inline int
pvrgpu_color_format_uses_integer_codec(const char *format)
{
   return format && (!strcmp(format, "PIPE_FORMAT_R8_UINT") ||
                     !strcmp(format, "PIPE_FORMAT_R8G8_UINT") ||
                     !strcmp(format, "PIPE_FORMAT_R8G8B8A8_UINT") ||
                     !strcmp(format, "PIPE_FORMAT_R16_UINT") ||
                     !strcmp(format, "PIPE_FORMAT_R16G16_UINT") ||
                     !strcmp(format, "PIPE_FORMAT_R16G16B16A16_UINT") ||
                     !strcmp(format, "PIPE_FORMAT_R32_UINT") ||
                     !strcmp(format, "PIPE_FORMAT_R32G32_UINT") ||
                     !strcmp(format, "PIPE_FORMAT_R32G32B32A32_UINT") ||
                     !strcmp(format, "PIPE_FORMAT_R10G10B10A2_UINT") ||
                     !strcmp(format, "PIPE_FORMAT_B10G10R10A2_UINT") ||
                     !strcmp(format, "PIPE_FORMAT_R8_SINT") ||
                     !strcmp(format, "PIPE_FORMAT_R8G8_SINT") ||
                     !strcmp(format, "PIPE_FORMAT_R8G8B8A8_SINT") ||
                     !strcmp(format, "PIPE_FORMAT_R16_SINT") ||
                     !strcmp(format, "PIPE_FORMAT_R16G16_SINT") ||
                     !strcmp(format, "PIPE_FORMAT_R16G16B16A16_SINT") ||
                     !strcmp(format, "PIPE_FORMAT_R32_SINT") ||
                     !strcmp(format, "PIPE_FORMAT_R32G32_SINT") ||
                     !strcmp(format, "PIPE_FORMAT_R32G32B32A32_SINT"));
}

/* UNORM formats of at most eight bits per channel travel as logical RGBA8:
 * the model keeps each native code bit-replicated to a byte, which Mesa's
 * 8unorm pack/unpack round-trips exactly. */
static inline int
pvrgpu_color_format_uses_unorm8_transport(const char *format)
{
   return format && (!strcmp(format, "PIPE_FORMAT_R8_UNORM") ||
                     !strcmp(format, "PIPE_FORMAT_R8G8_UNORM") ||
                     !strcmp(format, "PIPE_FORMAT_R8G8B8X8_UNORM") ||
                     !strcmp(format, "PIPE_FORMAT_B8G8R8X8_UNORM") ||
                     !strcmp(format, "PIPE_FORMAT_R5G6B5_UNORM") ||
                     !strcmp(format, "PIPE_FORMAT_B5G6R5_UNORM"));
}

static inline int
pvrgpu_color_format_uses_canonical_float(const char *format)
{
   return pvrgpu_is_explicit_color_format(format) &&
          !pvrgpu_color_format_uses_integer_codec(format) &&
          !pvrgpu_color_format_uses_unorm8_transport(format) &&
          strcmp(format, "PIPE_FORMAT_R32G32B32A32_UNORM") &&
          strcmp(format, "PIPE_FORMAT_R8G8B8A8_UNORM") &&
          strcmp(format, "PIPE_FORMAT_R8G8B8A8_SRGB") &&
          strcmp(format, "PIPE_FORMAT_B8G8R8A8_SRGB") &&
          strcmp(format, "PIPE_FORMAT_R10G10B10A2_UNORM") &&
          strcmp(format, "PIPE_FORMAT_B10G10R10A2_UNORM") &&
          strcmp(format, "PIPE_FORMAT_R32_UINT") &&
          strcmp(format, "PIPE_FORMAT_R32_SINT") &&
          strcmp(format, "PIPE_FORMAT_R32G32_UINT") &&
          strcmp(format, "PIPE_FORMAT_R32G32_SINT") &&
          strcmp(format, "PIPE_FORMAT_R32G32B32A32_UINT") &&
          strcmp(format, "PIPE_FORMAT_R32G32B32A32_SINT") &&
          strcmp(format, "PIPE_FORMAT_R32G32B32A32_FLOAT");
}

static inline int
pvrgpu_color_format_uses_canonical_double(const char *format)
{
   return format && !strcmp(format, "PIPE_FORMAT_R32G32B32A32_UNORM");
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
