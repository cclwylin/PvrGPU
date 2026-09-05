/* SPDX-License-Identifier: MIT */

#include "pvrgpu_context.h"
#include "pvrgpu_cmd.h"
#include "pvrgpu_counter.h"
#include "pvrgpu_resource.h"

#include "pipe/p_defines.h"
#include "util/format/u_format.h"
#include "util/u_debug.h"
#include "util/u_math.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Widest packed color block the driver lowers (32-bit RGBA/10:10:10:2). */
#define PVRGPU_MAX_COLOR_BLOCK_SIZE 4u

static const char *
pvrgpu_command_output_path(void)
{
   const char *path = getenv("PVRGPU_DRIVER_COMMAND_OUT");
   if (path && path[0] != '\0')
      return path;
   const char *jsonl_path = getenv("PVRGPU_SYSTEMC_JSONL_OUT");
   if (jsonl_path && jsonl_path[0] != '\0') {
      static char fallback_path[PATH_MAX];
      const char *slash = strrchr(jsonl_path, '/');
      if (slash && slash != jsonl_path) {
         const int dir_length = (int)(slash - jsonl_path);
         const int written = snprintf(fallback_path,
                                      sizeof(fallback_path),
                                      "%.*s/driver-command.txt",
                                      dir_length,
                                      jsonl_path);
         if (written > 0 && (size_t)written < sizeof(fallback_path))
            return fallback_path;
      }
   }
   return NULL;
}

static bool
pvrgpu_parse_env_uint(const char *name, unsigned minimum, unsigned *value)
{
   const char *text = getenv(name);
   if (!value || !text || text[0] == '\0')
      return false;

   errno = 0;
   char *end = NULL;
   const unsigned long parsed = strtoul(text, &end, 10);
   if (errno != 0 ||
       end == text ||
       *end != '\0' ||
       parsed < minimum ||
       parsed > UINT_MAX)
      return false;

   *value = (unsigned)parsed;
   return true;
}

static bool
pvrgpu_trace_has_no_draw_actions(void)
{
   unsigned draw_actions = 0;
   return pvrgpu_parse_env_uint("PVRGPU_RDC_TRACE_DRAW_ACTIONS",
                                0,
                                &draw_actions) &&
          draw_actions == 0;
}

static bool
pvrgpu_string_has_prefix(const char *text, const char *prefix)
{
   return text && prefix && strncmp(text, prefix, strlen(prefix)) == 0;
}

static const char *
pvrgpu_clear_command_case_name(void)
{
   const char *case_name = getenv("PVRGPU_RDC_CASE_NAME");
   if (!case_name || case_name[0] == '\0')
      return "phase1.clear.gallium";

   for (const char *cursor = case_name; *cursor; ++cursor) {
      if (*cursor == '\n' || *cursor == '\r')
         return "phase1.clear.gallium";
   }
   return case_name;
}

static bool
pvrgpu_rdc_output_extent(unsigned *width, unsigned *height)
{
   return pvrgpu_parse_env_uint("PVRGPU_RDC_OUTPUT_WIDTH", 1, width) &&
          pvrgpu_parse_env_uint("PVRGPU_RDC_OUTPUT_HEIGHT", 1, height);
}

static bool
pvrgpu_apply_zero_draw_output_extent(unsigned *width, unsigned *height)
{
   if (!width || !height || !pvrgpu_trace_has_no_draw_actions())
      return false;

   unsigned output_width = 0;
   unsigned output_height = 0;
   if (!pvrgpu_rdc_output_extent(&output_width, &output_height))
      return false;

   const bool changed = *width != output_width || *height != output_height;
   if (*width != output_width || *height != output_height) {
      pvrgpu_counter_eventf("rdc_output_extent",
                            "source=%ux%u output=%ux%u",
                            *width,
                            *height,
                            output_width,
                            output_height);
   }
   *width = output_width;
   *height = output_height;
   return changed;
}

static bool
pvrgpu_can_lower_clear_color_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8X8_UNORM:
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_B8G8R8X8_UNORM:
   case PIPE_FORMAT_R5G6B5_UNORM:
   case PIPE_FORMAT_B5G6R5_UNORM:
   case PIPE_FORMAT_R10G10B10A2_UNORM:
   case PIPE_FORMAT_B10G10R10A2_UNORM:
      /*
       * The command capsule carries the logical RGBA clear color.  8:8:8:8 X
       * formats are lowered as RGBA8 commands because the ignored alpha lane is
       * not visible to the model; 10:10:10:2 formats are carried explicitly so
       * RenderDoc replay can keep the captured default-framebuffer format.
       */
      return true;
   default:
      return false;
   }
}

static bool
pvrgpu_rgba8_cbuf_bound(const struct pipe_framebuffer_state *fb)
{
   return fb &&
          fb->nr_cbufs == 1 &&
          fb->cbufs[0].texture &&
          pvrgpu_can_lower_clear_color_format(fb->cbufs[0].format) &&
          fb->width != 0 &&
          fb->height != 0;
}

static bool
pvrgpu_surface_level_valid(const struct pvrgpu_resource *resource,
                           unsigned level)
{
   if (!resource)
      return false;
   if (resource->base.target == PIPE_BUFFER)
      return level == 0;

   const unsigned level_count = resource->level_count ? resource->level_count : 1;
   return level < level_count && level < PIPE_MAX_TEXTURE_LEVELS;
}

static unsigned
pvrgpu_surface_level_width(const struct pipe_resource *resource,
                           unsigned level)
{
   if (!resource)
      return 0;
   if (resource->target == PIPE_BUFFER)
      return resource->width0;
   return u_minify(resource->width0, level);
}

static unsigned
pvrgpu_surface_level_height(const struct pipe_resource *resource,
                            unsigned level)
{
   if (!resource || resource->target == PIPE_BUFFER)
      return 1;

   switch (resource->target) {
   case PIPE_TEXTURE_1D:
   case PIPE_TEXTURE_1D_ARRAY:
      return 1;
   default:
      return u_minify(resource->height0, level);
   }
}

static unsigned
pvrgpu_surface_level_layer_count(const struct pipe_resource *resource,
                                 unsigned level)
{
   if (!resource || resource->target == PIPE_BUFFER)
      return 1;

   switch (resource->target) {
   case PIPE_TEXTURE_3D:
      return u_minify(resource->depth0, level);
   case PIPE_TEXTURE_CUBE:
      return 6;
   case PIPE_TEXTURE_CUBE_ARRAY:
      return resource->array_size >= 6 ? resource->array_size : 6;
   case PIPE_TEXTURE_1D_ARRAY:
   case PIPE_TEXTURE_2D_ARRAY:
      return resource->array_size > 0 ? resource->array_size : 1;
   default:
      return 1;
   }
}

static bool
pvrgpu_color_surface_rect_supported(const struct pipe_surface *surface,
                                    unsigned dstx,
                                    unsigned dsty,
                                    unsigned width,
                                    unsigned height)
{
   if (!surface ||
       !surface->texture ||
       !pvrgpu_can_lower_clear_color_format(surface->format) ||
       width == 0 ||
       height == 0 ||
       util_format_get_blocksize(surface->format) == 0)
      return false;

   struct pvrgpu_resource *resource = pvrgpu_resource(surface->texture);
   if (!resource ||
       !resource->data ||
       !pvrgpu_surface_level_valid(resource, surface->level))
      return false;

   const unsigned level_width =
      pvrgpu_surface_level_width(surface->texture, surface->level);
   const unsigned level_height =
      pvrgpu_surface_level_height(surface->texture, surface->level);
   const unsigned layer_count =
      pvrgpu_surface_level_layer_count(surface->texture, surface->level);

   return level_width != 0 &&
          level_height != 0 &&
          layer_count != 0 &&
          surface->first_layer <= surface->last_layer &&
          surface->last_layer < layer_count &&
          dstx <= level_width &&
          dsty <= level_height &&
          width <= level_width - dstx &&
          height <= level_height - dsty &&
          resource->level_strides[surface->level] != 0 &&
          resource->level_layer_strides[surface->level] != 0;
}

static bool
pvrgpu_depth_surface_rect_supported(const struct pipe_surface *surface,
                                    unsigned dstx,
                                    unsigned dsty,
                                    unsigned width,
                                    unsigned height)
{
   if (!surface || !surface->texture || width == 0 || height == 0)
      return false;
   switch (surface->format) {
   case PIPE_FORMAT_Z16_UNORM:
   case PIPE_FORMAT_Z24X8_UNORM:
   case PIPE_FORMAT_X8Z24_UNORM:
   case PIPE_FORMAT_Z24_UNORM_S8_UINT:
   case PIPE_FORMAT_S8_UINT_Z24_UNORM:
   case PIPE_FORMAT_Z32_UNORM:
   case PIPE_FORMAT_Z32_FLOAT:
      break;
   default:
      return false;
   }

   struct pvrgpu_resource *resource = pvrgpu_resource(surface->texture);
   if (!resource || !resource->data ||
       !pvrgpu_surface_level_valid(resource, surface->level))
      return false;
   const unsigned level_width =
      pvrgpu_surface_level_width(surface->texture, surface->level);
   const unsigned level_height =
      pvrgpu_surface_level_height(surface->texture, surface->level);
   const unsigned layer_count =
      pvrgpu_surface_level_layer_count(surface->texture, surface->level);
   return surface->texture->target == PIPE_TEXTURE_2D &&
          level_width != 0 && level_height != 0 && layer_count == 1 &&
          surface->first_layer == 0 && surface->last_layer == 0 &&
          dstx <= level_width && dsty <= level_height &&
          width <= level_width - dstx &&
          height <= level_height - dsty &&
          resource->level_strides[surface->level] != 0 &&
          resource->level_layer_strides[surface->level] != 0;
}

/*
 * Write the stencil plane of a combined attachment, leaving the depth alone.
 *
 * Only the packed 24/8 formats have one; every other depth format reports no
 * stencil and is left untouched rather than being guessed at.
 */
static bool
pvrgpu_fill_surface_rect_with_clear_stencil(struct pipe_surface *surface,
                                            unsigned dstx,
                                            unsigned dsty,
                                            unsigned width,
                                            unsigned height,
                                            unsigned stencil)
{
   if (!pvrgpu_depth_surface_rect_supported(surface, dstx, dsty, width,
                                            height))
      return false;
   if (surface->format != PIPE_FORMAT_Z24_UNORM_S8_UINT &&
       surface->format != PIPE_FORMAT_S8_UINT_Z24_UNORM)
      return false;

   struct pvrgpu_resource *resource = pvrgpu_resource(surface->texture);
   const unsigned level = surface->level;
   const unsigned block_size = util_format_get_blocksize(surface->format);
   if (block_size != sizeof(uint32_t))
      return false;
   const uint32_t value = stencil & 0xffu;
   uint8_t *base = resource->data + resource->level_offsets[level];
   for (unsigned y = 0; y < height; ++y) {
      uint8_t *row = base +
                     (uintptr_t)(dsty + y) * resource->level_strides[level] +
                     (uintptr_t)dstx * block_size;
      for (unsigned x = 0; x < width; ++x) {
         uint8_t *pixel = row + (uintptr_t)x * block_size;
         uint32_t packed = 0;
         memcpy(&packed, pixel, sizeof(packed));
         if (surface->format == PIPE_FORMAT_Z24_UNORM_S8_UINT) {
            packed = (packed & UINT32_C(0x00ffffff)) | (value << 24);
         } else {
            packed = (packed & UINT32_C(0xffffff00)) | value;
         }
         memcpy(pixel, &packed, sizeof(packed));
      }
   }
   return true;
}

static bool
pvrgpu_fill_surface_rect_with_clear_depth(struct pipe_surface *surface,
                                          unsigned dstx,
                                          unsigned dsty,
                                          unsigned width,
                                          unsigned height,
                                          double depth)
{
   if (!pvrgpu_depth_surface_rect_supported(surface,
                                            dstx,
                                            dsty,
                                            width,
                                            height) ||
       !isfinite(depth))
      return false;

   const double clamped = depth < 0.0 ? 0.0 : depth > 1.0 ? 1.0 : depth;
   struct pvrgpu_resource *resource = pvrgpu_resource(surface->texture);
   const unsigned level = surface->level;
   const unsigned block_size = util_format_get_blocksize(surface->format);
   if (block_size == 0)
      return false;
   uint8_t *base = resource->data + resource->level_offsets[level];
   for (unsigned y = 0; y < height; ++y) {
      uint8_t *row = base +
                     (uintptr_t)(dsty + y) * resource->level_strides[level] +
                     (uintptr_t)dstx * block_size;
      for (unsigned x = 0; x < width; ++x) {
         uint8_t *pixel = row + (uintptr_t)x * block_size;
         switch (surface->format) {
         case PIPE_FORMAT_Z16_UNORM: {
            const uint16_t packed = (uint16_t)llround(clamped * 65535.0);
            memcpy(pixel, &packed, sizeof(packed));
            break;
         }
         case PIPE_FORMAT_Z24X8_UNORM: {
            const uint32_t packed =
               (uint32_t)llround(clamped * 16777215.0);
            memcpy(pixel, &packed, sizeof(packed));
            break;
         }
         case PIPE_FORMAT_X8Z24_UNORM: {
            const uint32_t packed =
               (uint32_t)llround(clamped * 16777215.0) << 8;
            memcpy(pixel, &packed, sizeof(packed));
            break;
         }
         case PIPE_FORMAT_Z24_UNORM_S8_UINT: {
            uint32_t packed = 0;
            memcpy(&packed, pixel, sizeof(packed));
            packed = (packed & UINT32_C(0xff000000)) |
                     (uint32_t)llround(clamped * 16777215.0);
            memcpy(pixel, &packed, sizeof(packed));
            break;
         }
         case PIPE_FORMAT_S8_UINT_Z24_UNORM: {
            uint32_t packed = 0;
            memcpy(&packed, pixel, sizeof(packed));
            packed = (packed & UINT32_C(0x000000ff)) |
                     ((uint32_t)llround(clamped * 16777215.0) << 8);
            memcpy(pixel, &packed, sizeof(packed));
            break;
         }
         case PIPE_FORMAT_Z32_UNORM: {
            const uint32_t packed =
               (uint32_t)llround(clamped * 4294967295.0);
            memcpy(pixel, &packed, sizeof(packed));
            break;
         }
         case PIPE_FORMAT_Z32_FLOAT: {
            const float packed = (float)clamped;
            memcpy(pixel, &packed, sizeof(packed));
            break;
         }
         default:
            return false;
         }
      }
   }
   return true;
}

static uint8_t
pvrgpu_float_to_unorm8(float value)
{
   if (!(value > 0.0f))
      return 0;
   if (value >= 1.0f)
      return 255;
   return (uint8_t)(value * 255.0f + 0.5f);
}

static uint32_t
pvrgpu_float_to_unorm10(float value)
{
   if (!(value > 0.0f))
      return 0;
   if (value >= 1.0f)
      return 1023;
   return (uint32_t)(value * 1023.0f + 0.5f);
}

static uint32_t
pvrgpu_float_to_unorm2(float value)
{
   if (!(value > 0.0f))
      return 0;
   if (value >= 1.0f)
      return 3;
   return (uint32_t)(value * 3.0f + 0.5f);
}

static const char *
pvrgpu_command_format_for_color_surface(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R8G8B8X8_UNORM:
      return PVRGPU_DRIVER_COMMAND_FORMAT_RGBX8;
   case PIPE_FORMAT_B8G8R8X8_UNORM:
      return PVRGPU_DRIVER_COMMAND_FORMAT_BGRX8;
   case PIPE_FORMAT_R5G6B5_UNORM:
      return PVRGPU_DRIVER_COMMAND_FORMAT_R5G6B5;
   case PIPE_FORMAT_B5G6R5_UNORM:
      return PVRGPU_DRIVER_COMMAND_FORMAT_B5G6R5;
   case PIPE_FORMAT_R10G10B10A2_UNORM:
      return PVRGPU_DRIVER_COMMAND_FORMAT_R10G10B10A2;
   case PIPE_FORMAT_B10G10R10A2_UNORM:
      return PVRGPU_DRIVER_COMMAND_FORMAT_B10G10R10A2;
   default:
      return PVRGPU_DRIVER_COMMAND_FORMAT_RGBA8;
   }
}

static void
pvrgpu_store_clear_color_pixel(enum pipe_format format,
                               uint8_t *pixel,
                               uint8_t r,
                               uint8_t g,
                               uint8_t b,
                               uint8_t a,
                               uint32_t r10,
                               uint32_t g10,
                               uint32_t b10,
                               uint32_t a2)
{
   switch (format) {
   case PIPE_FORMAT_R10G10B10A2_UNORM: {
      const uint32_t packed = r10 | (g10 << 10) | (b10 << 20) | (a2 << 30);
      memcpy(pixel, &packed, sizeof(packed));
      break;
   }
   case PIPE_FORMAT_B10G10R10A2_UNORM: {
      const uint32_t packed = b10 | (g10 << 10) | (r10 << 20) | (a2 << 30);
      memcpy(pixel, &packed, sizeof(packed));
      break;
   }
   case PIPE_FORMAT_R5G6B5_UNORM: {
      const uint16_t r5 = (uint16_t)((r * 31u + 127u) / 255u);
      const uint16_t g6 = (uint16_t)((g * 63u + 127u) / 255u);
      const uint16_t b5 = (uint16_t)((b * 31u + 127u) / 255u);
      const uint16_t packed = r5 | (g6 << 5) | (b5 << 11);
      memcpy(pixel, &packed, sizeof(packed));
      break;
   }
   case PIPE_FORMAT_B5G6R5_UNORM: {
      const uint16_t r5 = (uint16_t)((r * 31u + 127u) / 255u);
      const uint16_t g6 = (uint16_t)((g * 63u + 127u) / 255u);
      const uint16_t b5 = (uint16_t)((b * 31u + 127u) / 255u);
      const uint16_t packed = b5 | (g6 << 5) | (r5 << 11);
      memcpy(pixel, &packed, sizeof(packed));
      break;
   }
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_B8G8R8X8_UNORM:
      pixel[0] = b;
      pixel[1] = g;
      pixel[2] = r;
      pixel[3] = format == PIPE_FORMAT_B8G8R8X8_UNORM ? 255 : a;
      break;
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8X8_UNORM:
   default:
      pixel[0] = r;
      pixel[1] = g;
      pixel[2] = b;
      pixel[3] = format == PIPE_FORMAT_R8G8B8X8_UNORM ? 255 : a;
      break;
   }
}

static void
pvrgpu_clear_color_write_mask(enum pipe_format format,
                              unsigned colormask,
                              uint8_t *mask_pixel)
{
   /*
    * Every lowered color format packs its channels into disjoint bit ranges,
    * so storing the maximum value for the enabled channels and zero for the
    * disabled ones yields the exact write mask of the packed pixel.  The
    * ignored alpha lane of an X8 format always reads back as one, so its mask
    * bits stay set regardless of the requested colormask.
    */
   pvrgpu_store_clear_color_pixel(format,
                                  mask_pixel,
                                  (colormask & PIPE_MASK_R) ? 255 : 0,
                                  (colormask & PIPE_MASK_G) ? 255 : 0,
                                  (colormask & PIPE_MASK_B) ? 255 : 0,
                                  (colormask & PIPE_MASK_A) ? 255 : 0,
                                  (colormask & PIPE_MASK_R) ? 1023 : 0,
                                  (colormask & PIPE_MASK_G) ? 1023 : 0,
                                  (colormask & PIPE_MASK_B) ? 1023 : 0,
                                  (colormask & PIPE_MASK_A) ? 3 : 0);
}

static void
pvrgpu_fill_surface_rect_with_clear_color(struct pipe_surface *surface,
                                          unsigned dstx,
                                          unsigned dsty,
                                          unsigned width,
                                          unsigned height,
                                          unsigned colormask,
                                          const union pipe_color_union *color)
{
   struct pvrgpu_resource *resource = pvrgpu_resource(surface->texture);
   if (!resource ||
       !resource->data ||
       !pvrgpu_color_surface_rect_supported(surface, dstx, dsty, width, height))
      return;

   const unsigned level = surface->level;
   const unsigned block_size = util_format_get_blocksize(surface->format);
   if (block_size == 0 || block_size > PVRGPU_MAX_COLOR_BLOCK_SIZE)
      return;

   uint8_t value_pixel[PVRGPU_MAX_COLOR_BLOCK_SIZE] = {0};
   uint8_t mask_pixel[PVRGPU_MAX_COLOR_BLOCK_SIZE] = {0};
   pvrgpu_store_clear_color_pixel(surface->format,
                                  value_pixel,
                                  pvrgpu_float_to_unorm8(color->f[0]),
                                  pvrgpu_float_to_unorm8(color->f[1]),
                                  pvrgpu_float_to_unorm8(color->f[2]),
                                  pvrgpu_float_to_unorm8(color->f[3]),
                                  pvrgpu_float_to_unorm10(color->f[0]),
                                  pvrgpu_float_to_unorm10(color->f[1]),
                                  pvrgpu_float_to_unorm10(color->f[2]),
                                  pvrgpu_float_to_unorm2(color->f[3]));
   pvrgpu_clear_color_write_mask(surface->format, colormask, mask_pixel);
   for (unsigned byte = 0; byte < block_size; ++byte)
      value_pixel[byte] &= mask_pixel[byte];

   const unsigned stride = resource->level_strides[level];
   const uintptr_t layer_stride = resource->level_layer_strides[level];
   const uintptr_t level_offset = resource->level_offsets[level];
   for (unsigned layer = surface->first_layer; layer <= surface->last_layer;
        ++layer) {
      uint8_t *layer_base = resource->data + level_offset +
                            (uintptr_t)layer * layer_stride;
      for (unsigned y = 0; y < height; ++y) {
         uint8_t *row = layer_base + (uintptr_t)(dsty + y) * stride +
                        (uintptr_t)dstx * block_size;
         for (unsigned x = 0; x < width; ++x) {
            uint8_t *pixel = row + (uintptr_t)x * block_size;
            for (unsigned byte = 0; byte < block_size; ++byte) {
               pixel[byte] = (uint8_t)((pixel[byte] & ~mask_pixel[byte]) |
                                       value_pixel[byte]);
            }
         }
      }
   }
}

/*
 * The v1 clear_color capsule can only describe a framebuffer that holds a
 * single color.  A scissored or channel-masked clear still leaves the surface
 * uniform in the common dEQP color_clear sequences, so read the resolved color
 * back from the surface the driver just wrote instead of guessing it from the
 * requested clear value.
 */
static bool
pvrgpu_surface_uniform_color(const struct pipe_surface *surface,
                             unsigned width,
                             unsigned height,
                             float *rgba)
{
   if (!surface || !rgba || width == 0 || height == 0 ||
       surface->first_layer != surface->last_layer ||
       !pvrgpu_color_surface_rect_supported(surface, 0, 0, width, height))
      return false;

   const struct pvrgpu_resource *resource =
      pvrgpu_resource((struct pipe_resource *)surface->texture);
   const unsigned level = surface->level;
   const unsigned block_size = util_format_get_blocksize(surface->format);
   const unsigned stride = resource->level_strides[level];
   const uint8_t *base = resource->data + resource->level_offsets[level] +
                         (uintptr_t)surface->first_layer *
                            resource->level_layer_strides[level];

   const size_t row_size = (size_t)width * block_size;
   uint8_t *reference = malloc(row_size);
   if (!reference)
      return false;
   for (unsigned x = 0; x < width; ++x)
      memcpy(reference + (size_t)x * block_size, base, block_size);

   bool uniform = true;
   for (unsigned y = 0; uniform && y < height; ++y)
      uniform = memcmp(base + (uintptr_t)y * stride, reference, row_size) == 0;
   free(reference);
   if (!uniform)
      return false;

   util_format_unpack_rgba(surface->format, rgba, base, 1);
   return true;
}

static void
pvrgpu_emit_clear_color_command(unsigned width,
                                unsigned height,
                                enum pipe_format format,
                                const union pipe_color_union *color)
{
   const char *path = pvrgpu_command_output_path();
   if (!path) {
      debug_printf("pvrgpu: PVRGPU_DRIVER_COMMAND_OUT is not set\n");
      pvrgpu_counter_eventf("clear_color_command_skip",
                            "reason=missing_command_path");
      return;
   }

   const bool output_target_changed =
      pvrgpu_apply_zero_draw_output_extent(&width, &height);

   struct pvrgpu_clear_color_command command;
   memset(&command, 0, sizeof(command));
   command.case_name = pvrgpu_clear_command_case_name();
   command.frame = 1;
   command.width = width;
   command.height = height;
   command.format = pvrgpu_command_format_for_color_surface(format);
   if (output_target_changed) {
      command.clear_color_bits[0] = 0;
      command.clear_color_bits[1] = 0;
      command.clear_color_bits[2] = 0;
      command.clear_color_bits[3] = UINT32_C(0x3f800000);
   } else {
      command.clear_color_bits[0] = fui(color->f[0]);
      command.clear_color_bits[1] = fui(color->f[1]);
      command.clear_color_bits[2] = fui(color->f[2]);
      command.clear_color_bits[3] = fui(color->f[3]);
   }

   char error[256];
   if (!pvrgpu_write_clear_color_command(path, &command, error, sizeof(error))) {
      debug_printf("pvrgpu: %s\n", error);
      pvrgpu_counter_eventf("clear_color_command_error", "reason=%s", error);
      return;
   }
   pvrgpu_counter_eventf("clear_color_command",
                         "width=%u height=%u format=%s changed_extent=%u",
                         command.width,
                         command.height,
                         command.format,
                         output_target_changed ? 1 : 0);
}

/*
 * Reduce the requested clear region to the surface rectangle the driver will
 * write.  Mesa hands the scissor box in Gallium's Y=0=top convention, already
 * clamped against the framebuffer on the state-tracker side, so the only work
 * left is a defensive clamp and an empty-region test.
 */
static bool
pvrgpu_clear_rect(const struct pipe_framebuffer_state *fb,
                  const struct pipe_scissor_state *scissor,
                  unsigned *x,
                  unsigned *y,
                  unsigned *width,
                  unsigned *height)
{
   if (!fb || fb->width == 0 || fb->height == 0)
      return false;

   unsigned minx = 0;
   unsigned miny = 0;
   unsigned maxx = fb->width;
   unsigned maxy = fb->height;
   if (scissor) {
      minx = MIN2(scissor->minx, fb->width);
      miny = MIN2(scissor->miny, fb->height);
      maxx = MIN2(scissor->maxx, fb->width);
      maxy = MIN2(scissor->maxy, fb->height);
   }
   if (minx >= maxx || miny >= maxy)
      return false;

   *x = minx;
   *y = miny;
   *width = maxx - minx;
   *height = maxy - miny;
   return true;
}

void
pvrgpu_clear(struct pipe_context *pipe,
             unsigned buffers,
             uint32_t color_clear_mask,
             uint8_t stencil_clear_mask,
             const struct pipe_scissor_state *scissor_state,
             const union pipe_color_union *color,
             double depth,
             unsigned stencil)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);

   const bool clear_color = (buffers & PIPE_CLEAR_COLOR0) != 0;
   const bool clear_depth = (buffers & PIPE_CLEAR_DEPTH) != 0;
   const bool clear_stencil = (buffers & PIPE_CLEAR_STENCIL) != 0;
   const unsigned supported_buffers =
      PIPE_CLEAR_COLOR0 | PIPE_CLEAR_DEPTH | PIPE_CLEAR_STENCIL;
   /* Gallium packs four colormask bits per draw buffer; only cbuf0 is lowered. */
   const unsigned colormask = color_clear_mask & PIPE_MASK_RGBA;
   if ((buffers & ~supported_buffers) != 0 ||
       (!clear_color && !clear_depth && !clear_stencil)) {
      debug_printf("pvrgpu: unsupported clear flags; fail closed\n");
      if (clear_depth)
         pvrgpu_invalidate_full_depth_clear(ctx);
      return;
   }

   unsigned rect_x = 0;
   unsigned rect_y = 0;
   unsigned rect_width = 0;
   unsigned rect_height = 0;
   if (!pvrgpu_clear_rect(&ctx->framebuffer,
                          scissor_state,
                          &rect_x,
                          &rect_y,
                          &rect_width,
                          &rect_height)) {
      pvrgpu_counter_eventf("clear_scissored_empty",
                            "framebuffer=%ux%u buffers=0x%x",
                            ctx->framebuffer.width,
                            ctx->framebuffer.height,
                            buffers);
      if (clear_depth)
         pvrgpu_invalidate_full_depth_clear(ctx);
      return;
   }
   const bool full_surface_rect = rect_x == 0 && rect_y == 0 &&
                                  rect_width == ctx->framebuffer.width &&
                                  rect_height == ctx->framebuffer.height;

   if (clear_color &&
       (!color || !pvrgpu_rgba8_cbuf_bound(&ctx->framebuffer))) {
      debug_printf("pvrgpu: unsupported clear target; fail closed\n");
      if (clear_depth)
         pvrgpu_invalidate_full_depth_clear(ctx);
      return;
   }
   if (clear_color &&
       !pvrgpu_color_surface_rect_supported(&ctx->framebuffer.cbufs[0],
                                            rect_x,
                                            rect_y,
                                            rect_width,
                                            rect_height)) {
      debug_printf("pvrgpu: unsupported clear surface rect; fail closed\n");
      if (clear_depth)
         pvrgpu_invalidate_full_depth_clear(ctx);
      return;
   }
   if (clear_depth &&
       !pvrgpu_depth_surface_rect_supported(&ctx->framebuffer.zsbuf,
                                            rect_x,
                                            rect_y,
                                            rect_width,
                                            rect_height)) {
      debug_printf("pvrgpu: unsupported clear depth target; fail closed\n");
      pvrgpu_invalidate_full_depth_clear(ctx);
      return;
   }
   const bool depth_backing_written =
      !clear_depth ||
      pvrgpu_fill_surface_rect_with_clear_depth(&ctx->framebuffer.zsbuf,
                                                rect_x,
                                                rect_y,
                                                rect_width,
                                                rect_height,
                                                depth);

   /*
    * The model starts a sequence's stencil plane from the value the last
    * whole-surface clear wrote, so record it here.  A partial clear leaves the
    * plane non-uniform, which the capsule cannot describe; it still updates the
    * driver's own surface and is reported.
    */
   if (clear_stencil) {
      (void)pvrgpu_fill_surface_rect_with_clear_stencil(&ctx->framebuffer.zsbuf,
                                                        rect_x,
                                                        rect_y,
                                                        rect_width,
                                                        rect_height,
                                                        stencil);
      if (full_surface_rect)
         ctx->stencil_clear_value = stencil & 0xffu;
      pvrgpu_counter_eventf("clear_stencil",
                            "res=%p x=%u y=%u width=%u height=%u format=%s "
                            "stencil=%u full_surface=%u mask=0x%x",
                            (void *)ctx->framebuffer.zsbuf.texture,
                            rect_x,
                            rect_y,
                            rect_width,
                            rect_height,
                            util_format_name(ctx->framebuffer.zsbuf.format),
                            stencil,
                            full_surface_rect ? 1u : 0u,
                            stencil_clear_mask);
   }

   /*
    * A whole-surface clear is stated as the draw's own clear value and
    * supersedes every scissored one before it: the plane is uniform again, so
    * replaying the earlier rectangles would only undo it.  A scissored one is
    * remembered until a draw inherits it.
    */
   {
      const float clamped_depth =
         depth < 0.0 ? 0.0f : depth > 1.0 ? 1.0f : (float)depth;
      uint32_t depth_bits = 0;
      memcpy(&depth_bits, &clamped_depth, sizeof(depth_bits));
      const unsigned aspects =
         (clear_depth ? PVRGPU_SYSTEMC_CLEAR_ASPECT_DEPTH : 0u) |
         (clear_stencil ? PVRGPU_SYSTEMC_CLEAR_ASPECT_STENCIL : 0u);
      if (aspects != 0) {
         if (full_surface_rect) {
            ctx->pending_attachment_clear_count = 0;
         } else if (!pvrgpu_note_pending_attachment_clear(ctx,
                                                          rect_x,
                                                          rect_y,
                                                          rect_width,
                                                          rect_height,
                                                          aspects,
                                                          depth_bits,
                                                          stencil & 0xffu)) {
            pvrgpu_counter_eventf("clear_attachment_pending_overflow",
                                  "count=%u aspects=0x%x x=%u y=%u "
                                  "width=%u height=%u",
                                  ctx->pending_attachment_clear_count,
                                  aspects,
                                  rect_x,
                                  rect_y,
                                  rect_width,
                                  rect_height);
         }
      }
   }

   if (clear_depth) {
      if (full_surface_rect) {
         const float clamped =
            depth < 0.0 ? 0.0f : depth > 1.0 ? 1.0f : (float)depth;
         memcpy(&ctx->depth_clear_bits, &clamped, sizeof(ctx->depth_clear_bits));
      }
      if (depth == 1.0 && full_surface_rect) {
         pvrgpu_note_full_depth_clear_one(ctx,
                                          &ctx->framebuffer.zsbuf,
                                          ctx->framebuffer.width,
                                          ctx->framebuffer.height);
      } else {
         pvrgpu_invalidate_full_depth_clear(ctx);
      }
      pvrgpu_counter_eventf("clear_depth",
                            "res=%p x=%u y=%u width=%u height=%u format=%s "
                            "level=%u layers=%u-%u depth=%f combined_color=%u "
                            "backing_written=%u",
                            (void *)ctx->framebuffer.zsbuf.texture,
                            rect_x,
                            rect_y,
                            rect_width,
                            rect_height,
                            util_format_name(ctx->framebuffer.zsbuf.format),
                            ctx->framebuffer.zsbuf.level,
                            ctx->framebuffer.zsbuf.first_layer,
                            ctx->framebuffer.zsbuf.last_layer,
                            depth,
                            clear_color ? 1 : 0,
                            depth_backing_written ? 1 : 0);
   }

   if (!clear_color)
      return;
   if (colormask == 0) {
      pvrgpu_counter_eventf("clear_color_masked_out",
                            "res=%p width=%u height=%u",
                            (void *)ctx->framebuffer.cbufs[0].texture,
                            rect_width,
                            rect_height);
      return;
   }

   pvrgpu_fill_surface_rect_with_clear_color(&ctx->framebuffer.cbufs[0],
                                             rect_x,
                                             rect_y,
                                             rect_width,
                                             rect_height,
                                             colormask,
                                             color);
   /*
    * Record whether the model can still describe this surface.  A whole-surface
    * RGBA clear it can: the sequence it runs starts from the same colour.  A
    * scissored or masked one it cannot, and until the model can be told about
    * those, its framebuffer must not be copied back over them.
    */
   {
      struct pvrgpu_resource *cbuf0 =
         pvrgpu_resource(ctx->framebuffer.cbufs[0].texture);
      if (cbuf0) {
         cbuf0->driver_writes_model_cannot_reproduce =
            !full_surface_rect || colormask != PIPE_MASK_RGBA;
      }
   }
   pvrgpu_counter_eventf("clear_color",
                         "res=%p x=%u y=%u width=%u height=%u format=%s "
                         "level=%u layers=%u-%u colormask=0x%x scissored=%u "
                         "rgba=%u,%u,%u,%u floats=%f,%f,%f,%f",
                         (void *)ctx->framebuffer.cbufs[0].texture,
                         rect_x,
                         rect_y,
                         rect_width,
                         rect_height,
                         util_format_name(ctx->framebuffer.cbufs[0].format),
                         ctx->framebuffer.cbufs[0].level,
                         ctx->framebuffer.cbufs[0].first_layer,
                         ctx->framebuffer.cbufs[0].last_layer,
                         colormask,
                         full_surface_rect ? 0 : 1,
                         pvrgpu_float_to_unorm8(color->f[0]),
                         pvrgpu_float_to_unorm8(color->f[1]),
                         pvrgpu_float_to_unorm8(color->f[2]),
                         pvrgpu_float_to_unorm8(color->f[3]),
                         color->f[0],
                         color->f[1],
                         color->f[2],
                         color->f[3]);
   if (pvrgpu_case_reserves_native_pco_sequence() ||
       ctx->driver_draw_command_emitted ||
       pvrgpu_driver_draw_command_has_been_emitted())
      return;
   if (full_surface_rect && colormask == PIPE_MASK_RGBA) {
      pvrgpu_emit_clear_color_command(ctx->framebuffer.width,
                                      ctx->framebuffer.height,
                                      ctx->framebuffer.cbufs[0].format,
                                      color);
      return;
   }

   union pipe_color_union resolved;
   memset(&resolved, 0, sizeof(resolved));
   if (pvrgpu_surface_uniform_color(&ctx->framebuffer.cbufs[0],
                                    ctx->framebuffer.width,
                                    ctx->framebuffer.height,
                                    resolved.f)) {
      pvrgpu_emit_clear_color_command(ctx->framebuffer.width,
                                      ctx->framebuffer.height,
                                      ctx->framebuffer.cbufs[0].format,
                                      &resolved);
      return;
   }
   pvrgpu_counter_eventf("clear_color_command_skip",
                         "reason=non_uniform_color_surface x=%u y=%u "
                         "width=%u height=%u colormask=0x%x",
                         rect_x,
                         rect_y,
                         rect_width,
                         rect_height,
                         colormask);
}

void
pvrgpu_clear_render_target(struct pipe_context *pipe,
                           struct pipe_surface *dst,
                           const union pipe_color_union *color,
                           unsigned dstx,
                           unsigned dsty,
                           unsigned width,
                           unsigned height,
                           bool render_condition_enabled)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   if (render_condition_enabled) {
      debug_printf("pvrgpu: conditional clear_render_target is unsupported\n");
      return;
   }
   if (!color ||
       !pvrgpu_color_surface_rect_supported(dst, dstx, dsty, width, height)) {
      debug_printf("pvrgpu: unsupported clear_render_target; fail closed\n");
      return;
   }
   pvrgpu_fill_surface_rect_with_clear_color(dst,
                                             dstx,
                                             dsty,
                                             width,
                                             height,
                                             PIPE_MASK_RGBA,
                                             color);
   pvrgpu_counter_eventf("clear_render_target",
                         "res=%p x=%u y=%u width=%u height=%u format=%s "
                         "level=%u layers=%u-%u rgba=%u,%u,%u,%u "
                         "floats=%f,%f,%f,%f",
                         (void *)dst->texture,
                         dstx,
                         dsty,
                         width,
                         height,
                         util_format_name(dst->format),
                         dst->level,
                         dst->first_layer,
                         dst->last_layer,
                         pvrgpu_float_to_unorm8(color->f[0]),
                         pvrgpu_float_to_unorm8(color->f[1]),
                         pvrgpu_float_to_unorm8(color->f[2]),
                         pvrgpu_float_to_unorm8(color->f[3]),
                         color->f[0],
                         color->f[1],
                         color->f[2],
                         color->f[3]);
   if (!pvrgpu_case_reserves_native_pco_sequence() &&
       !ctx->driver_draw_command_emitted &&
       !pvrgpu_driver_draw_command_has_been_emitted()) {
      pvrgpu_emit_clear_color_command(width, height, dst->format, color);
   }
}

void
pvrgpu_clear_depth_stencil(struct pipe_context *pipe,
                           struct pipe_surface *dst,
                           unsigned clear_flags,
                           double depth,
                           unsigned stencil,
                           unsigned dstx,
                           unsigned dsty,
                           unsigned width,
                           unsigned height,
                           bool render_condition_enabled)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);

   if (render_condition_enabled) {
      debug_printf("pvrgpu: conditional clear_depth_stencil is unsupported\n");
      pvrgpu_invalidate_full_depth_clear(ctx);
      return;
   }
   if ((clear_flags & PIPE_CLEAR_DEPTH) == 0 ||
       (clear_flags & ~PIPE_CLEAR_DEPTH) != 0 ||
       !pvrgpu_depth_surface_rect_supported(dst,
                                            dstx,
                                            dsty,
                                            width,
                                            height)) {
      debug_printf("pvrgpu: unsupported clear_depth_stencil target; fail closed\n");
      pvrgpu_invalidate_full_depth_clear(ctx);
      return;
   }

   const bool backing_written =
      pvrgpu_fill_surface_rect_with_clear_depth(dst,
                                                dstx,
                                                dsty,
                                                width,
                                                height,
                                                depth);
   const unsigned level_width =
      pvrgpu_surface_level_width(dst->texture, dst->level);
   const unsigned level_height =
      pvrgpu_surface_level_height(dst->texture, dst->level);
   if (dstx == 0 && dsty == 0 && width == level_width &&
       height == level_height && depth == 1.0) {
      pvrgpu_note_full_depth_clear_one(ctx, dst, width, height);
   } else {
      pvrgpu_invalidate_full_depth_clear(ctx);
   }
   pvrgpu_counter_eventf("clear_depth_stencil",
                         "res=%p flags=0x%x x=%u y=%u width=%u height=%u "
                         "format=%s depth=%f stencil=%u backing_written=%u",
                         (void *)dst->texture,
                         clear_flags,
                         dstx,
                         dsty,
                         width,
                         height,
                         util_format_name(dst->format),
                         depth,
                         stencil,
                         backing_written ? 1 : 0);
}
