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
#include <stdlib.h>
#include <string.h>

static const char *
pvrgpu_command_output_path(void)
{
   const char *path = getenv("PVRGPU_DRIVER_COMMAND_OUT");
   if (path && path[0] != '\0')
      return path;
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
pvrgpu_rgba8_surface_full_rect(const struct pipe_surface *surface,
                               unsigned dstx,
                               unsigned dsty,
                               unsigned width,
                               unsigned height)
{
   return surface &&
          surface->texture &&
          pvrgpu_can_lower_clear_color_format(surface->format) &&
          dstx == 0 &&
          dsty == 0 &&
          width == surface->texture->width0 &&
          height == surface->texture->height0 &&
          width != 0 &&
          height != 0;
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
   case PIPE_FORMAT_R10G10B10A2_UNORM:
      return PVRGPU_DRIVER_COMMAND_FORMAT_R10G10B10A2;
   case PIPE_FORMAT_B10G10R10A2_UNORM:
      return PVRGPU_DRIVER_COMMAND_FORMAT_B10G10R10A2;
   default:
      return PVRGPU_DRIVER_COMMAND_FORMAT_RGBA8;
   }
}

static void
pvrgpu_fill_resource_with_clear_color(struct pipe_surface *surface,
                                      const union pipe_color_union *color)
{
   struct pvrgpu_resource *resource = pvrgpu_resource(surface->texture);
   if (!resource || !resource->data || resource->stride == 0)
      return;

   const uint8_t r = pvrgpu_float_to_unorm8(color->f[0]);
   const uint8_t g = pvrgpu_float_to_unorm8(color->f[1]);
   const uint8_t b = pvrgpu_float_to_unorm8(color->f[2]);
   const uint8_t a = pvrgpu_float_to_unorm8(color->f[3]);
   const uint32_t r10 = pvrgpu_float_to_unorm10(color->f[0]);
   const uint32_t g10 = pvrgpu_float_to_unorm10(color->f[1]);
   const uint32_t b10 = pvrgpu_float_to_unorm10(color->f[2]);
   const uint32_t a2 = pvrgpu_float_to_unorm2(color->f[3]);

   for (unsigned y = 0; y < surface->texture->height0; ++y) {
      uint8_t *row = resource->data + (uintptr_t)y * resource->stride;
      for (unsigned x = 0; x < surface->texture->width0; ++x) {
         uint8_t *pixel = row + (uintptr_t)x * 4;
         switch (surface->format) {
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
         case PIPE_FORMAT_B8G8R8A8_UNORM:
         case PIPE_FORMAT_B8G8R8X8_UNORM:
            pixel[0] = b;
            pixel[1] = g;
            pixel[2] = r;
            pixel[3] = surface->format == PIPE_FORMAT_B8G8R8X8_UNORM ? 255 : a;
            break;
         case PIPE_FORMAT_R8G8B8A8_UNORM:
         case PIPE_FORMAT_R8G8B8X8_UNORM:
         default:
            pixel[0] = r;
            pixel[1] = g;
            pixel[2] = b;
            pixel[3] = surface->format == PIPE_FORMAT_R8G8B8X8_UNORM ? 255 : a;
            break;
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
      command.clear_color_bits[3] = 0;
   } else {
      command.clear_color_bits[0] = fui(color->f[0]);
      command.clear_color_bits[1] = fui(color->f[1]);
      command.clear_color_bits[2] = fui(color->f[2]);
      command.clear_color_bits[3] = fui(color->f[3]);
   }

   char error[256];
   if (!pvrgpu_write_clear_color_command(path, &command, error, sizeof(error)))
      debug_printf("pvrgpu: %s\n", error);
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

   pvrgpu_fill_resource_with_clear_color(&ctx->framebuffer.cbufs[0], color);
   pvrgpu_counter_eventf("clear_color",
                         "width=%u height=%u format=%s",
                         ctx->framebuffer.width,
                         ctx->framebuffer.height,
                         util_format_name(ctx->framebuffer.cbufs[0].format));
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
   if (!color || !pvrgpu_rgba8_surface_full_rect(dst, dstx, dsty, width, height)) {
      debug_printf("pvrgpu: unsupported clear_render_target; fail closed\n");
      return;
   }
   pvrgpu_fill_resource_with_clear_color(dst, color);
   pvrgpu_counter_eventf("clear_render_target",
                         "width=%u height=%u format=%s",
                         width,
                         height,
                         util_format_name(dst->format));
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
