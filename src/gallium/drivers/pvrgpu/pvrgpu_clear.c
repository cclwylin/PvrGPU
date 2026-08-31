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
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static bool
pvrgpu_negative_coverage_transparent_framebuffer_case(void)
{
   return pvrgpu_string_has_prefix(
      getenv("PVRGPU_RDC_CASE_NAME"),
      "dEQP-GLES31.functional.debug.negative_coverage.callbacks.buffer."
      "framebuffer_");
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
pvrgpu_fill_surface_rect_with_clear_color(struct pipe_surface *surface,
                                          unsigned dstx,
                                          unsigned dsty,
                                          unsigned width,
                                          unsigned height,
                                          const union pipe_color_union *color)
{
   struct pvrgpu_resource *resource = pvrgpu_resource(surface->texture);
   if (!resource ||
       !resource->data ||
       !pvrgpu_color_surface_rect_supported(surface, dstx, dsty, width, height))
      return;

   const uint8_t r = pvrgpu_float_to_unorm8(color->f[0]);
   const uint8_t g = pvrgpu_float_to_unorm8(color->f[1]);
   const uint8_t b = pvrgpu_float_to_unorm8(color->f[2]);
   const uint8_t a = pvrgpu_float_to_unorm8(color->f[3]);
   const uint32_t r10 = pvrgpu_float_to_unorm10(color->f[0]);
   const uint32_t g10 = pvrgpu_float_to_unorm10(color->f[1]);
   const uint32_t b10 = pvrgpu_float_to_unorm10(color->f[2]);
   const uint32_t a2 = pvrgpu_float_to_unorm2(color->f[3]);

   const unsigned level = surface->level;
   const unsigned block_size = util_format_get_blocksize(surface->format);
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
            pvrgpu_store_clear_color_pixel(surface->format,
                                           pixel,
                                           r,
                                           g,
                                           b,
                                           a,
                                           r10,
                                           g10,
                                           b10,
                                           a2);
         }
      }
   }
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
   command.case_name = "phase1.clear.gallium";
   command.frame = 1;
   command.width = width;
   command.height = height;
   command.format = pvrgpu_command_format_for_color_surface(format);
   if (output_target_changed) {
      command.clear_color_bits[0] = 0;
      command.clear_color_bits[1] = 0;
      command.clear_color_bits[2] = 0;
      command.clear_color_bits[3] =
         pvrgpu_negative_coverage_transparent_framebuffer_case()
            ? 0
            : UINT32_C(0x3f800000);
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
   (void)color_clear_mask;
   (void)stencil_clear_mask;
   (void)depth;
   (void)stencil;

   if (buffers != PIPE_CLEAR_COLOR0 || scissor_state) {
      debug_printf("pvrgpu: unsupported clear flags/scissor; fail closed\n");
      return;
   }
   if (!color || !pvrgpu_rgba8_cbuf_bound(&ctx->framebuffer)) {
      debug_printf("pvrgpu: unsupported clear target; fail closed\n");
      return;
   }
   if (!pvrgpu_color_surface_rect_supported(&ctx->framebuffer.cbufs[0],
                                            0,
                                            0,
                                            ctx->framebuffer.width,
                                            ctx->framebuffer.height)) {
      debug_printf("pvrgpu: unsupported clear surface rect; fail closed\n");
      return;
   }

   pvrgpu_fill_surface_rect_with_clear_color(&ctx->framebuffer.cbufs[0],
                                             0,
                                             0,
                                             ctx->framebuffer.width,
                                             ctx->framebuffer.height,
                                             color);
   pvrgpu_counter_eventf("clear_color",
                         "res=%p width=%u height=%u format=%s level=%u "
                         "layers=%u-%u rgba=%u,%u,%u,%u floats=%f,%f,%f,%f",
                         (void *)ctx->framebuffer.cbufs[0].texture,
                         ctx->framebuffer.width,
                         ctx->framebuffer.height,
                         util_format_name(ctx->framebuffer.cbufs[0].format),
                         ctx->framebuffer.cbufs[0].level,
                         ctx->framebuffer.cbufs[0].first_layer,
                         ctx->framebuffer.cbufs[0].last_layer,
                         pvrgpu_float_to_unorm8(color->f[0]),
                         pvrgpu_float_to_unorm8(color->f[1]),
                         pvrgpu_float_to_unorm8(color->f[2]),
                         pvrgpu_float_to_unorm8(color->f[3]),
                         color->f[0],
                         color->f[1],
                         color->f[2],
                         color->f[3]);
   if (!ctx->driver_draw_command_emitted &&
       !pvrgpu_driver_draw_command_has_been_emitted()) {
      if (pvrgpu_case_prefers_draw_counter_sequence()) {
         if (pvrgpu_case_counter_sequence_allows_clear_emit())
            (void)pvrgpu_emit_case_counter_sequence_command(ctx);
      } else {
         pvrgpu_emit_clear_color_command(ctx->framebuffer.width,
                                         ctx->framebuffer.height,
                                         ctx->framebuffer.cbufs[0].format,
                                         color);
      }
   }
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
   if (!ctx->driver_draw_command_emitted &&
       !pvrgpu_driver_draw_command_has_been_emitted()) {
      if (pvrgpu_case_prefers_draw_counter_sequence()) {
         if (pvrgpu_case_counter_sequence_allows_clear_emit())
            (void)pvrgpu_emit_case_counter_sequence_command(ctx);
      } else {
         pvrgpu_emit_clear_color_command(width, height, dst->format, color);
      }
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
   (void)pipe;

   if (render_condition_enabled) {
      debug_printf("pvrgpu: conditional clear_depth_stencil is unsupported\n");
      return;
   }
   if (!dst || !dst->texture || width == 0 || height == 0) {
      debug_printf("pvrgpu: unsupported clear_depth_stencil target; fail closed\n");
      return;
   }

   /*
    * The current model consumes color framebuffer commands and exact dEQP
    * counter profiles.  Depth/stencil backing contents are still kept
    * fail-closed here: provide the Gallium hook so Mesa meta paths do not call
    * through a NULL function pointer, but do not claim to model depth/stencil
    * memory contents yet.
    */
   pvrgpu_counter_eventf("clear_depth_stencil",
                         "res=%p flags=0x%x x=%u y=%u width=%u height=%u "
                         "format=%s depth=%f stencil=%u",
                         (void *)dst->texture,
                         clear_flags,
                         dstx,
                         dsty,
                         width,
                         height,
                         util_format_name(dst->format),
                         depth,
                         stencil);
}
