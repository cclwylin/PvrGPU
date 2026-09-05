/* SPDX-License-Identifier: MIT */

#include "pvrgpu_resource.h"
#include "pvrgpu_cmd.h"
#include "pvrgpu_context.h"
#include "pvrgpu_counter.h"
#include "pvrgpu_screen.h"

#include "frontend/sw_winsys.h"
#include "pipe/p_defines.h"
#include "util/format/u_format.h"
#include "util/u_debug.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"
#include "util/u_math.h"
#include "util/u_transfer.h"

#include <ctype.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct pvrgpu_transfer {
   struct pipe_transfer base;
   void *displaytarget_map;
};

static uint64_t
pvrgpu_resource_debug_fnv1a64(const uint8_t *data, size_t size)
{
   uint64_t hash = UINT64_C(14695981039346656037);
   for (size_t i = 0; i < size; ++i) {
      hash ^= data[i];
      hash *= UINT64_C(1099511628211);
   }
   return hash;
}

static void
pvrgpu_emit_resource_copy_framebuffer_blit_command(struct pipe_context *pipe,
                                                   struct pipe_resource *dst,
                                                   unsigned dst_level,
                                                   unsigned dstx,
                                                   unsigned dsty,
                                                   unsigned dstz,
                                                   struct pipe_resource *src,
                                                   unsigned src_level,
                                                   const struct pipe_box *src_box);

static bool
pvrgpu_is_safe_case_char(char value)
{
   const unsigned char ch = (unsigned char)value;
   return isalnum(ch) || ch == '_' || ch == '.' || ch == '-';
}

static const char *
pvrgpu_rdc_case_name(void)
{
   const char *case_name = getenv("PVRGPU_RDC_CASE_NAME");
   if (!case_name || case_name[0] == '\0')
      return NULL;

   for (const char *cursor = case_name; *cursor; ++cursor) {
      if (!pvrgpu_is_safe_case_char(*cursor))
         return NULL;
   }
   return case_name;
}

static const char *
pvrgpu_command_case_name(const char *fallback)
{
   const char *case_name = pvrgpu_rdc_case_name();
   return case_name ? case_name : fallback;
}

static bool
pvrgpu_string_has_prefix(const char *text, const char *prefix)
{
   return text && prefix && strncmp(text, prefix, strlen(prefix)) == 0;
}

static bool
pvrgpu_string_contains(const char *text, const char *needle)
{
   return text && needle && strstr(text, needle) != NULL;
}

static bool
pvrgpu_case_suppresses_driver_commands(void)
{
   if (pvrgpu_case_reserves_native_pco_sequence())
      return true;
   const char *case_name = pvrgpu_rdc_case_name();
   return case_name &&
          strcmp(case_name,
                 "dEQP-GLES31.functional.debug.negative_coverage.callbacks."
                 "advanced_blend.attachment_advanced_equation") == 0;
}

static bool
pvrgpu_deqp_fbo_default_framebuffer_blit_to_default_case(void)
{
   const char *case_name = pvrgpu_rdc_case_name();
   return pvrgpu_string_has_prefix(
             case_name,
             "dEQP-GLES3.functional.fbo.blit.default_framebuffer.") &&
          pvrgpu_string_contains(case_name, "_blit_to_default");
}

static bool
pvrgpu_deqp_fbo_default_framebuffer_direct_color_counter_case(void)
{
   const char *case_name = pvrgpu_rdc_case_name();
   return pvrgpu_deqp_fbo_default_framebuffer_blit_to_default_case() &&
          (pvrgpu_string_contains(case_name, ".rgb8_") ||
           pvrgpu_string_contains(case_name, ".rgba8_"));
}

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
pvrgpu_trace_draw_actions(unsigned *draw_actions)
{
   const char *text = getenv("PVRGPU_RDC_TRACE_DRAW_ACTIONS");
   if (!draw_actions || !text || text[0] == '\0')
      return false;

   char *end = NULL;
   unsigned long parsed = strtoul(text, &end, 10);
   if (end == text || *end != '\0' || parsed > UINT_MAX)
      return false;

   *draw_actions = (unsigned)parsed;
   return true;
}

static bool
pvrgpu_rdc_output_extent(unsigned *width, unsigned *height)
{
   const char *width_text = getenv("PVRGPU_RDC_OUTPUT_WIDTH");
   const char *height_text = getenv("PVRGPU_RDC_OUTPUT_HEIGHT");
   if (!width || !height ||
       !width_text || width_text[0] == '\0' ||
       !height_text || height_text[0] == '\0')
      return false;

   char *width_end = NULL;
   char *height_end = NULL;
   const unsigned long parsed_width = strtoul(width_text, &width_end, 10);
   const unsigned long parsed_height = strtoul(height_text, &height_end, 10);
   if (width_end == width_text ||
       height_end == height_text ||
       *width_end != '\0' ||
       *height_end != '\0' ||
       parsed_width == 0 ||
       parsed_height == 0 ||
       parsed_width > UINT_MAX ||
       parsed_height > UINT_MAX)
      return false;

   *width = (unsigned)parsed_width;
   *height = (unsigned)parsed_height;
   return true;
}

/*
 * Whether a texture of this format can exist on this screen.
 *
 * The answer is the screen's, not a second opinion: `pvrgpu_resource.c` used to
 * carry its own shorter colour list, so `is_format_supported()` would say yes
 * to R8/RG8 and every other single- and dual-channel format while creation
 * silently returned NULL and the application saw `GL_OUT_OF_MEMORY`.
 *
 * Sampler-only formats belong here too. A screen that advertises them for
 * PIPE_BIND_SAMPLER_VIEW has to let the texture holding them be created.
 */
static bool
pvrgpu_is_supported_texture_resource_format(enum pipe_format format)
{
   return pvrgpu_is_supported_color_format(format) ||
          pvrgpu_is_supported_depth_stencil_format(format) ||
          pvrgpu_is_sampler_only_format(format);
}

static bool
pvrgpu_can_create_buffer(const struct pipe_resource *template)
{
   return template &&
          template->target == PIPE_BUFFER &&
          template->width0 != 0 &&
          template->height0 <= 1 &&
          template->depth0 <= 1 &&
          template->array_size <= 1 &&
          template->nr_samples <= 1 &&
          template->nr_storage_samples <= 1;
}

static bool
pvrgpu_is_supported_resource_sample_count(unsigned sample_count)
{
   switch (sample_count) {
   case 0:
   case 1:
   case 2:
   case 3:
   case 4:
   case 5:
   case 6:
   case 7:
   case 8:
   case 9:
   case 10:
   case 11:
   case 12:
   case 13:
   case 14:
   case 15:
   case 16:
      return true;
   default:
      return false;
   }
}

static bool
pvrgpu_resource_is_multisampled(const struct pipe_resource *resource)
{
   return resource &&
          (resource->nr_samples > 1 || resource->nr_storage_samples > 1);
}

static bool
pvrgpu_can_create_texture_target(const struct pipe_resource *template)
{
   if (!template ||
       template->width0 == 0 ||
       template->height0 == 0 ||
       !pvrgpu_is_supported_resource_sample_count(template->nr_samples) ||
       !pvrgpu_is_supported_resource_sample_count(
          template->nr_storage_samples))
      return false;

   switch (template->target) {
   case PIPE_TEXTURE_2D:
      return template->depth0 <= 1 && template->array_size <= 1;
   case PIPE_TEXTURE_2D_ARRAY:
      return template->depth0 <= 1 && template->array_size > 0;
   case PIPE_TEXTURE_3D:
      if (pvrgpu_resource_is_multisampled(template))
         return false;
      return template->depth0 > 0 && template->array_size <= 1;
   case PIPE_TEXTURE_CUBE:
      if (pvrgpu_resource_is_multisampled(template))
         return false;
      return template->depth0 <= 1;
   case PIPE_TEXTURE_CUBE_ARRAY:
      if (pvrgpu_resource_is_multisampled(template))
         return false;
      return template->depth0 <= 1 && template->array_size >= 6;
   default:
      return false;
   }
}

static bool
pvrgpu_can_create_texture(const struct pipe_resource *template)
{
   return template &&
          pvrgpu_can_create_texture_target(template) &&
          pvrgpu_is_supported_texture_resource_format(template->format);
}

/*
 * Why a resource was refused, or NULL when it was not.
 *
 * `resource_create` can only answer NULL, which Mesa reports as
 * `GL_OUT_OF_MEMORY` -- an answer that says nothing about which of the shape,
 * the sample count or the format was the problem. Naming the field turns a
 * guess into a reading.
 */
static const char *
pvrgpu_resource_create_refusal(const struct pipe_resource *template)
{
   if (!template)
      return "template";
   if (pvrgpu_can_create_buffer(template) || pvrgpu_can_create_texture(template))
      return NULL;
   if (template->target == PIPE_BUFFER)
      return "buffer_shape";
   if (!pvrgpu_is_supported_texture_resource_format(template->format))
      return "format";
   if (template->width0 == 0 || template->height0 == 0)
      return "extent";
   if (!pvrgpu_is_supported_resource_sample_count(template->nr_samples) ||
       !pvrgpu_is_supported_resource_sample_count(template->nr_storage_samples))
      return "sample_count";
   return "target";
}

static bool
pvrgpu_can_create_resource(struct pipe_screen *screen,
                           const struct pipe_resource *template)
{
   (void)screen;
   const char *refusal = pvrgpu_resource_create_refusal(template);
   if (!refusal)
      return true;
   pvrgpu_counter_eventf("resource_create_declined",
                         "reason=%s target=%u width=%u height=%u depth=%u "
                         "array=%u samples=%u storage_samples=%u format=%s "
                         "bind=0x%x",
                         refusal,
                         template ? template->target : 0,
                         template ? template->width0 : 0,
                         template ? template->height0 : 0,
                         template ? template->depth0 : 0,
                         template ? template->array_size : 0,
                         template ? template->nr_samples : 0,
                         template ? template->nr_storage_samples : 0,
                         template ? util_format_name(template->format) : "none",
                         template ? template->bind : 0);
   return false;
}

static unsigned
pvrgpu_resource_storage_sample_count(const struct pipe_resource *resource)
{
   unsigned sample_count = resource->nr_storage_samples;
   if (sample_count == 0)
      sample_count = resource->nr_samples;
   return sample_count > 1 ? sample_count : 1;
}

static unsigned
pvrgpu_resource_level_count(const struct pipe_resource *resource)
{
   if (!resource || resource->target == PIPE_BUFFER)
      return 1;

   const unsigned last_level =
      resource->last_level < PIPE_MAX_TEXTURE_LEVELS ?
         resource->last_level : PIPE_MAX_TEXTURE_LEVELS - 1;
   return last_level + 1;
}

static unsigned
pvrgpu_resource_level_width(const struct pipe_resource *resource,
                            unsigned level)
{
   if (!resource)
      return 0;
   if (resource->target == PIPE_BUFFER)
      return resource->width0;
   return u_minify(resource->width0, level);
}

static unsigned
pvrgpu_resource_level_height(const struct pipe_resource *resource,
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

static bool
pvrgpu_destination_box_matches_rdc_output(
   const struct pipe_resource *resource,
   unsigned level,
   int64_t origin_x,
   int64_t origin_y,
   int64_t origin_z,
   int64_t width,
   int64_t height,
   int64_t depth)
{
   unsigned output_width = 0;
   unsigned output_height = 0;
   if (!pvrgpu_rdc_output_extent(&output_width, &output_height))
      return true;

   return resource &&
          resource->target == PIPE_TEXTURE_2D &&
          level < pvrgpu_resource_level_count(resource) &&
          origin_x == 0 && origin_y == 0 && origin_z == 0 &&
          width == (int64_t)output_width &&
          height == (int64_t)output_height &&
          depth == 1 &&
          pvrgpu_resource_level_width(resource, level) == output_width &&
          pvrgpu_resource_level_height(resource, level) == output_height;
}

static unsigned
pvrgpu_resource_level_layer_count(const struct pipe_resource *resource,
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

static unsigned
pvrgpu_resource_layer_count(const struct pipe_resource *resource)
{
   return pvrgpu_resource_level_layer_count(resource, 0);
}

static bool
pvrgpu_resource_level_valid(const struct pvrgpu_resource *resource,
                            unsigned level)
{
   return resource && level < resource->level_count;
}

static bool
pvrgpu_resource_uses_displaytarget(const struct pipe_resource *resource)
{
   return resource &&
          resource->target != PIPE_BUFFER &&
          (resource->bind & (PIPE_BIND_DISPLAY_TARGET |
                             PIPE_BIND_SCANOUT |
                             PIPE_BIND_SHARED)) != 0;
}

static bool
pvrgpu_displaytarget_box_valid(const struct pvrgpu_resource *resource,
                               const struct pipe_box *box)
{
   return resource &&
          resource->displaytarget &&
          resource->data &&
          resource->level_count > 0 &&
          resource->displaytarget_stride > 0 &&
          box &&
          box->z == 0 &&
          box->depth == 1 &&
          box->x >= 0 &&
          box->y >= 0 &&
          box->width > 0 &&
          box->height > 0 &&
          (uint64_t)box->x + (uint64_t)box->width <=
             resource->base.width0 &&
          (uint64_t)box->y + (uint64_t)box->height <=
             resource->base.height0 &&
          util_format_get_blocksize(resource->base.format) != 0;
}

static void
pvrgpu_copy_displaytarget_box_to_shadow(struct pvrgpu_resource *resource,
                                        const void *displaytarget_map,
                                        const struct pipe_box *box)
{
   if (!pvrgpu_displaytarget_box_valid(resource, box) ||
       !displaytarget_map)
      return;

   const unsigned block_size =
      util_format_get_blocksize(resource->base.format);
   const unsigned row_bytes =
      util_format_get_stride(resource->base.format, box->width) *
      pvrgpu_resource_storage_sample_count(&resource->base);
   if (row_bytes == 0 || resource->displaytarget_stride < row_bytes)
      return;

   const uint8_t *src_base = (const uint8_t *)displaytarget_map;
   uint8_t *dst_base = resource->data + resource->level_offsets[0];
   for (int row = 0; row < box->height; ++row) {
      const uint8_t *src =
         src_base + (uintptr_t)(box->y + row) *
                       resource->displaytarget_stride +
         (uintptr_t)box->x * block_size;
      uint8_t *dst =
         dst_base + (uintptr_t)(box->y + row) *
                       resource->level_strides[0] +
         (uintptr_t)box->x * block_size;
      memcpy(dst, src, row_bytes);
   }
}

static void
pvrgpu_copy_shadow_box_to_displaytarget(struct pvrgpu_resource *resource,
                                        void *displaytarget_map,
                                        const struct pipe_box *box)
{
   if (!pvrgpu_displaytarget_box_valid(resource, box) ||
       !displaytarget_map)
      return;

   const unsigned block_size =
      util_format_get_blocksize(resource->base.format);
   const unsigned row_bytes =
      util_format_get_stride(resource->base.format, box->width) *
      pvrgpu_resource_storage_sample_count(&resource->base);
   if (row_bytes == 0 || resource->displaytarget_stride < row_bytes)
      return;

   const uint8_t *src_base = resource->data + resource->level_offsets[0];
   uint8_t *dst_base = (uint8_t *)displaytarget_map;
   for (int row = 0; row < box->height; ++row) {
      const uint8_t *src =
         src_base + (uintptr_t)(box->y + row) *
                       resource->level_strides[0] +
         (uintptr_t)box->x * block_size;
      uint8_t *dst =
         dst_base + (uintptr_t)(box->y + row) *
                       resource->displaytarget_stride +
         (uintptr_t)box->x * block_size;
      memcpy(dst, src, row_bytes);
   }
}

static bool
pvrgpu_displaytarget_whole_box(const struct pvrgpu_resource *resource,
                               struct pipe_box *box)
{
   if (!resource || !box || resource->base.width0 == 0 ||
       resource->base.height0 == 0)
      return false;

   box->x = 0;
   box->y = 0;
   box->z = 0;
   box->width = (int)resource->base.width0;
   box->height = (int)resource->base.height0;
   box->depth = 1;
   return true;
}

static void
pvrgpu_sync_displaytarget_to_shadow(struct pvrgpu_resource *resource,
                                    unsigned map_flags)
{
   if (!resource || !resource->displaytarget || !resource->base.screen)
      return;

   struct pvrgpu_screen *screen = pvrgpu_screen(resource->base.screen);
   if (!screen->winsys || !screen->winsys->displaytarget_map ||
       !screen->winsys->displaytarget_unmap)
      return;

   struct pipe_box whole;
   if (!pvrgpu_displaytarget_whole_box(resource, &whole))
      return;

   void *map = screen->winsys->displaytarget_map(screen->winsys,
                                                 resource->displaytarget,
                                                 map_flags);
   if (!map)
      return;

   pvrgpu_copy_displaytarget_box_to_shadow(resource, map, &whole);
   screen->winsys->displaytarget_unmap(screen->winsys,
                                       resource->displaytarget);
}

static void
pvrgpu_sync_shadow_to_displaytarget(struct pvrgpu_resource *resource,
                                    unsigned map_flags)
{
   if (!resource || !resource->displaytarget || !resource->base.screen)
      return;

   struct pvrgpu_screen *screen = pvrgpu_screen(resource->base.screen);
   if (!screen->winsys || !screen->winsys->displaytarget_map ||
       !screen->winsys->displaytarget_unmap)
      return;

   struct pipe_box whole;
   if (!pvrgpu_displaytarget_whole_box(resource, &whole))
      return;

   void *map = screen->winsys->displaytarget_map(screen->winsys,
                                                 resource->displaytarget,
                                                 map_flags);
   if (!map)
      return;

   pvrgpu_copy_shadow_box_to_displaytarget(resource, map, &whole);
   screen->winsys->displaytarget_unmap(screen->winsys,
                                       resource->displaytarget);
}

static bool
pvrgpu_init_resource_storage(struct pvrgpu_resource *resource)
{
   if (resource->base.target == PIPE_BUFFER) {
      resource->stride = resource->base.width0;
      resource->layer_stride = resource->base.width0;
      resource->level_strides[0] = resource->stride;
      resource->level_layer_strides[0] = resource->layer_stride;
      resource->level_offsets[0] = 0;
      resource->level_count = 1;
      resource->size = resource->base.width0;
   } else {
      const unsigned sample_count =
         pvrgpu_resource_storage_sample_count(&resource->base);
      const unsigned level_count =
         pvrgpu_resource_level_count(&resource->base);
      uintptr_t offset = 0;

      if (util_format_get_blocksize(resource->base.format) == 0)
         return false;

      for (unsigned level = 0; level < level_count; ++level) {
         const unsigned level_width =
            pvrgpu_resource_level_width(&resource->base, level);
         const unsigned level_height =
            pvrgpu_resource_level_height(&resource->base, level);
         const unsigned level_layers =
            pvrgpu_resource_level_layer_count(&resource->base, level);
         const unsigned stride =
            util_format_get_stride(resource->base.format, level_width) *
            sample_count;
         const uintptr_t layer_stride =
            util_format_get_2d_size(resource->base.format,
                                    stride,
                                    level_height);

         resource->level_offsets[level] = offset;
         resource->level_strides[level] = stride;
         resource->level_layer_strides[level] = layer_stride;
         offset += layer_stride * level_layers;
      }

      resource->stride = resource->level_strides[0];
      resource->layer_stride = resource->level_layer_strides[0];
      resource->level_count = level_count;
      resource->size = offset;
   }

   resource->data = CALLOC(1, resource->size);
   return resource->data != NULL;
}

static bool
pvrgpu_init_resource_displaytarget(struct pipe_screen *screen,
                                   struct pvrgpu_resource *resource,
                                   const void *map_front_private)
{
   if (!pvrgpu_resource_uses_displaytarget(&resource->base))
      return true;

   struct pvrgpu_screen *pscreen = pvrgpu_screen(screen);
   struct sw_winsys *winsys = pscreen ? pscreen->winsys : NULL;
   if (!winsys || !winsys->displaytarget_create)
      return false;

   unsigned stride = 0;
   resource->displaytarget =
      winsys->displaytarget_create(winsys,
                                   resource->base.bind,
                                   resource->base.format,
                                   resource->base.width0,
                                   resource->base.height0,
                                   64,
                                   map_front_private,
                                   &stride);
   if (!resource->displaytarget || stride == 0)
      return false;

   resource->displaytarget_stride = stride;
   pvrgpu_sync_displaytarget_to_shadow(resource, PIPE_MAP_READ);
   return true;
}

static struct pipe_resource *
pvrgpu_resource_create_common(struct pipe_screen *screen,
                              const struct pipe_resource *template,
                              const void *map_front_private)
{
   if (!pvrgpu_can_create_resource(screen, template))
      return NULL;

   struct pvrgpu_resource *resource = CALLOC_STRUCT(pvrgpu_resource);
   if (!resource)
      return NULL;
   resource->base = *template;
   pipe_reference_init(&resource->base.reference, 1);
   resource->base.screen = screen;
   if (!pvrgpu_init_resource_storage(resource)) {
      FREE(resource);
      return NULL;
   }
   if (!pvrgpu_init_resource_displaytarget(screen,
                                           resource,
                                           map_front_private)) {
      FREE(resource->data);
      FREE(resource);
      return NULL;
   }
   pvrgpu_counter_eventf("resource_create",
                         "res=%p target=%u width=%u height=%u depth=%u "
                         "array=%u levels=%u format=%s bind=0x%x usage=%u "
                         "flags=0x%x size=%zu displaytarget=%u "
                         "displaytarget_stride=%u",
                         (void *)&resource->base,
                         resource->base.target,
                         resource->base.width0,
                         resource->base.height0,
                         resource->base.depth0,
                         resource->base.array_size,
                         resource->level_count,
                         util_format_name(resource->base.format),
                         resource->base.bind,
                         resource->base.usage,
                         resource->base.flags,
                         resource->size,
                         resource->displaytarget ? 1 : 0,
                         resource->displaytarget_stride);
   return &resource->base;
}

static struct pipe_resource *
pvrgpu_resource_create(struct pipe_screen *screen,
                       const struct pipe_resource *template)
{
   return pvrgpu_resource_create_common(screen, template, NULL);
}

static struct pipe_resource *
pvrgpu_resource_create_front(struct pipe_screen *screen,
                             const struct pipe_resource *template,
                             const void *map_front_private)
{
   return pvrgpu_resource_create_common(screen, template, map_front_private);
}

static struct pipe_resource *
pvrgpu_resource_from_handle(struct pipe_screen *screen,
                            const struct pipe_resource *template,
                            struct winsys_handle *whandle,
                            unsigned usage)
{
   (void)usage;
   if (!pvrgpu_can_create_resource(screen, template))
      return NULL;

   struct pvrgpu_screen *pscreen = pvrgpu_screen(screen);
   struct sw_winsys *winsys = pscreen ? pscreen->winsys : NULL;
   if (!winsys || !winsys->displaytarget_from_handle)
      return NULL;

   struct pvrgpu_resource *resource = CALLOC_STRUCT(pvrgpu_resource);
   if (!resource)
      return NULL;
   resource->base = *template;
   pipe_reference_init(&resource->base.reference, 1);
   resource->base.screen = screen;
   if (!pvrgpu_init_resource_storage(resource)) {
      FREE(resource);
      return NULL;
   }

   unsigned stride = 0;
   resource->displaytarget =
      winsys->displaytarget_from_handle(winsys,
                                        template,
                                        whandle,
                                        &stride);
   if (!resource->displaytarget || stride == 0) {
      FREE(resource->data);
      FREE(resource);
      return NULL;
   }
   resource->displaytarget_stride = stride;
   pvrgpu_sync_displaytarget_to_shadow(resource, PIPE_MAP_READ);

   pvrgpu_counter_eventf("resource_from_handle",
                         "res=%p target=%u width=%u height=%u format=%s "
                         "bind=0x%x usage=0x%x displaytarget_stride=%u",
                         (void *)&resource->base,
                         resource->base.target,
                         resource->base.width0,
                         resource->base.height0,
                         util_format_name(resource->base.format),
                         resource->base.bind,
                         usage,
                         resource->displaytarget_stride);
   return &resource->base;
}

static bool
pvrgpu_resource_get_handle(struct pipe_screen *screen,
                           struct pipe_context *context,
                           struct pipe_resource *resource,
                           struct winsys_handle *whandle,
                           unsigned usage)
{
   (void)context;
   (void)usage;
   struct pvrgpu_screen *pscreen = pvrgpu_screen(screen);
   struct pvrgpu_resource *pvrgpu = pvrgpu_resource(resource);
   if (!pscreen || !pscreen->winsys ||
       !pscreen->winsys->displaytarget_get_handle ||
       !pvrgpu || !pvrgpu->displaytarget)
      return false;

   pvrgpu_sync_shadow_to_displaytarget(pvrgpu,
                                       PIPE_MAP_WRITE |
                                          PIPE_MAP_UNSYNCHRONIZED);
   return pscreen->winsys->displaytarget_get_handle(
      pscreen->winsys, pvrgpu->displaytarget, whandle);
}

static void
pvrgpu_flush_frontbuffer(struct pipe_screen *screen,
                         struct pipe_context *context,
                         struct pipe_resource *resource,
                         unsigned level,
                         unsigned layer,
                         void *winsys_drawable_handle,
                         unsigned nboxes,
                         struct pipe_box *subbox)
{
   (void)context;
   struct pvrgpu_screen *pscreen = pvrgpu_screen(screen);
   struct pvrgpu_resource *pvrgpu = pvrgpu_resource(resource);
   if (!pscreen || !pscreen->winsys ||
       !pscreen->winsys->displaytarget_display ||
       !pvrgpu || !pvrgpu->displaytarget)
      return;

   pvrgpu_sync_shadow_to_displaytarget(pvrgpu,
                                       PIPE_MAP_WRITE |
                                          PIPE_MAP_UNSYNCHRONIZED);
   pscreen->winsys->displaytarget_display(pscreen->winsys,
                                          pvrgpu->displaytarget,
                                          winsys_drawable_handle,
                                          nboxes,
                                          subbox);
   pvrgpu_counter_eventf("flush_frontbuffer",
                         "res=%p level=%u layer=%u width=%u height=%u "
                         "format=%s boxes=%u",
                         (void *)resource,
                         level,
                         layer,
                         resource ? resource->width0 : 0,
                         resource ? resource->height0 : 0,
                         resource ? util_format_name(resource->format) :
                                      "none",
                         nboxes);
}

static void
pvrgpu_resource_destroy(struct pipe_screen *screen,
                        struct pipe_resource *resource)
{
   if (!resource) {
      pvrgpu_counter_eventf("resource_destroy", "resource=null");
      return;
   }

   struct pvrgpu_resource *pvrgpu = pvrgpu_resource(resource);
   if (!pvrgpu) {
      pvrgpu_counter_eventf("resource_destroy", "resource=non_pvrgpu");
      return;
   }

   pvrgpu_counter_eventf("resource_destroy",
                         "res=%p target=%u width=%u height=%u depth=%u array=%u "
                         "format=%s bind=0x%x usage=%u flags=0x%x size=%zu",
                         (void *)resource,
                         resource->target,
                         resource->width0,
                         resource->height0,
                         resource->depth0,
                         resource->array_size,
                         util_format_name(resource->format),
                         resource->bind,
                         resource->usage,
                         resource->flags,
                         pvrgpu->size);
   struct pvrgpu_screen *pscreen =
      screen ? pvrgpu_screen(screen) : pvrgpu_screen(resource->screen);
   if (pvrgpu->displaytarget && pscreen && pscreen->winsys &&
       pscreen->winsys->displaytarget_destroy) {
      pscreen->winsys->displaytarget_destroy(pscreen->winsys,
                                             pvrgpu->displaytarget);
   }
   FREE(pvrgpu->data);
   FREE(pvrgpu);
}

static bool
pvrgpu_transfer_box_in_bounds(const struct pipe_resource *resource,
                              unsigned level,
                              const struct pipe_box *box)
{
   if (!resource || !box)
      return false;

   if (resource->target == PIPE_BUFFER) {
      if (level != 0)
         return false;
      return box->x >= 0 &&
             box->y == 0 &&
             box->z == 0 &&
             box->height == 1 &&
             box->depth == 1 &&
             box->width >= 0 &&
             (uint64_t)box->x + (uint64_t)box->width <= resource->width0;
   }

   const struct pvrgpu_resource *pvrgpu =
      pvrgpu_resource((struct pipe_resource *)resource);
   if (!pvrgpu_resource_level_valid(pvrgpu, level))
      return false;

   const unsigned level_width = pvrgpu_resource_level_width(resource, level);
   const unsigned level_height = pvrgpu_resource_level_height(resource, level);
   const unsigned level_layers =
      pvrgpu_resource_level_layer_count(resource, level);

   return pvrgpu_can_create_texture_target(resource) &&
          box->x >= 0 &&
          box->y >= 0 &&
          box->z >= 0 &&
          box->width >= 0 &&
          box->height >= 0 &&
          box->depth >= 0 &&
          (uint64_t)box->x + (uint64_t)box->width <= level_width &&
          (uint64_t)box->y + (uint64_t)box->height <= level_height &&
          (uint64_t)box->z + (uint64_t)box->depth <=
             level_layers;
}

/*
 * True when this is the surface the model has been drawing into.
 *
 * A readback of anything else -- a texture the application uploaded, a
 * staging buffer -- has nothing to do with the model's framebuffer and must
 * not be overwritten with it.
 */
static bool
pvrgpu_resource_is_current_color_attachment(
   const struct pvrgpu_context *ctx,
   const struct pipe_resource *resource)
{
   return ctx && resource && ctx->framebuffer.nr_cbufs != 0 &&
          ctx->framebuffer.cbufs[0].texture == resource;
}

/*
 * The colour surfaces the model's RGBA8 output can be stored into.
 *
 * The model publishes R,G,B,A byte order.  A surface that names the same four
 * 8-bit channels in another order holds the same bytes rearranged, so it is
 * served by reordering them on the way in.  Anything else -- a wider channel,
 * a packed 5:6:5 -- would need a conversion nobody has specified, and is left
 * to the path that already serves it.
 */
static bool
pvrgpu_resource_readback_format_is_supported(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8X8_UNORM:
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_B8G8R8X8_UNORM:
      return true;
   default:
      return false;
   }
}

/*
 * Store one row of the model's RGBA8 output in the surface's byte order.
 *
 * The layout matches pvrgpu_store_clear_color_pixel() channel for channel,
 * including an X8 format's ignored alpha lane reading back as one.  A pixel a
 * draw covered and a pixel only the clear touched therefore agree, which they
 * would not if the draw's alpha were carried through verbatim.
 */
static void
pvrgpu_resource_readback_store_row(enum pipe_format format,
                                   uint8_t *destination,
                                   const uint8_t *rgba8,
                                   unsigned width)
{
   const bool swap_red_blue = format == PIPE_FORMAT_B8G8R8A8_UNORM ||
                              format == PIPE_FORMAT_B8G8R8X8_UNORM;
   const bool opaque = format == PIPE_FORMAT_R8G8B8X8_UNORM ||
                       format == PIPE_FORMAT_B8G8R8X8_UNORM;

   if (!swap_red_blue && !opaque) {
      memcpy(destination, rgba8, (size_t)width * 4u);
      return;
   }

   for (unsigned x = 0; x < width; ++x) {
      const uint8_t *source = rgba8 + (size_t)x * 4u;
      uint8_t *pixel = destination + (size_t)x * 4u;
      const uint8_t r = source[0];
      const uint8_t b = source[2];
      pixel[0] = swap_red_blue ? b : r;
      pixel[1] = source[1];
      pixel[2] = swap_red_blue ? r : b;
      pixel[3] = opaque ? 255u : source[3];
   }
}

/*
 * Bring what the model drew into the CPU backing store, before a read sees it.
 *
 * `pvrgpu->data` is the driver's own memory: clears write there, draws do not
 * -- a draw goes to the model.  Until the model's output came back, every
 * `glReadPixels` after a draw returned the clear, which is why dEQP reported
 * missing pixels and never a wrong one while the model's own PNG showed the
 * right geometry.  This is where the two are joined: the accumulated draws are
 * submitted, the model runs them, and its DRAM readback lands here.
 *
 * Everything here is fail-closed.  A surface the model did not render, a
 * format its RGBA8 output cannot be reordered into, or a flush that produced
 * nothing all leave `pvrgpu->data` exactly as it was.
 */
static void
pvrgpu_resource_read_back_color_attachment(struct pipe_context *pipe,
                                           struct pipe_resource *resource,
                                           unsigned level,
                                           unsigned usage)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   struct pvrgpu_resource *pvrgpu = pvrgpu_resource(resource);
   if (!(usage & PIPE_MAP_READ) || level != 0 || !ctx || !resource ||
       !pvrgpu || !pvrgpu->data || resource->target == PIPE_BUFFER)
      return;
   if (!pvrgpu_resource_is_current_color_attachment(ctx, resource))
      return;
   if (!pvrgpu_resource_readback_format_is_supported(resource->format))
      return;
   /*
    * The model's framebuffer only describes the surface while everything that
    * touched it went to the model.  A scissored or masked clear did not, so
    * copying the model's output back would erase it -- which is what a run of
    * dEQP's color_clear.scissored_* showed: two full clears reached the model,
    * thirteen scissored ones did not, and the readback published the uniform
    * surface the model had.  Leave the driver's own content alone instead.
    */
   if (pvrgpu->driver_writes_model_cannot_reproduce) {
      pvrgpu_counter_eventf("framebuffer_readback_declined",
                            "reason=driver_writes_model_cannot_reproduce "
                            "res=%p width=%u height=%u",
                            (void *)resource,
                            pvrgpu_resource_level_width(resource, level),
                            pvrgpu_resource_level_height(resource, level));
      return;
   }

   const unsigned width = pvrgpu_resource_level_width(resource, level);
   const unsigned height = pvrgpu_resource_level_height(resource, level);
   if (width == 0 || height == 0)
      return;

   const size_t pixels_size = (size_t)width * (size_t)height * 4u;
   uint8_t *pixels = MALLOC(pixels_size);
   if (!pixels)
      return;

   pvrgpu_context_end_frame_at_readback(ctx);

   bool written = false;
   char error[512] = { 0 };
   const bool flushed =
      pvrgpu_systemc_flush_readback_rgba8(width, height, pixels, pixels_size,
                                          &written, error, sizeof(error));
   if (!flushed || !written) {
      if (!flushed) {
         debug_printf("pvrgpu: %s\n",
                      error[0] ? error : "readback flush failed");
      }
      FREE(pixels);
      return;
   }

   /*
    * The model's framebuffer is tightly packed; the resource's level may be
    * padded, so store a row at a time rather than the whole block.
    */
   uint8_t *destination = pvrgpu->data + pvrgpu->level_offsets[level];
   const unsigned stride = pvrgpu->level_strides[level];
   for (unsigned row = 0; row < height; ++row) {
      pvrgpu_resource_readback_store_row(resource->format,
                                         destination + (size_t)row * stride,
                                         pixels + (size_t)row * (size_t)width * 4u,
                                         width);
   }
   FREE(pixels);

   pvrgpu_counter_eventf("framebuffer_readback",
                         "res=%p width=%u height=%u format=%s",
                         (void *)resource,
                         width,
                         height,
                         util_format_name(resource->format));
}

static void *
pvrgpu_transfer_map(struct pipe_context *pipe,
                    struct pipe_resource *resource,
                    unsigned level,
                    unsigned usage,
                    const struct pipe_box *box,
                    struct pipe_transfer **out_transfer)
{
   struct pvrgpu_resource *pvrgpu = pvrgpu_resource(resource);
   if (!out_transfer || !pvrgpu || !pvrgpu->data ||
       !pvrgpu_transfer_box_in_bounds(resource, level, box))
      return NULL;

   /*
    * Ahead of every path below, including the displaytarget one -- that path
    * copies this shadow into the winsys surface on its way past, so the
    * pixels have to be here before it runs.
    */
   pvrgpu_resource_read_back_color_attachment(pipe, resource, level, usage);

   struct pvrgpu_transfer *pvrgpu_transfer =
      CALLOC_STRUCT(pvrgpu_transfer);
   if (!pvrgpu_transfer)
      return NULL;
   struct pipe_transfer *transfer = &pvrgpu_transfer->base;

   pipe_resource_reference(&transfer->resource, resource);
   transfer->level = level;
   transfer->usage = usage;
   transfer->box = *box;
   transfer->stride = pvrgpu->level_strides[level];
   transfer->layer_stride = pvrgpu->level_layer_strides[level];

   pvrgpu_counter_eventf(resource->target == PIPE_BUFFER ?
                         "buffer_map" : "texture_map",
                         "res=%p level=%u usage=0x%x x=%d y=%d z=%d "
                         "width=%d height=%d depth=%d stride=%u "
                         "layer_stride=%zu format=%s displaytarget=%u",
                         (void *)resource,
                         level,
                         usage,
                         box->x,
                         box->y,
                         box->z,
                         box->width,
                         box->height,
                         box->depth,
                         transfer->stride,
                         (size_t)transfer->layer_stride,
                         util_format_name(resource->format),
                         pvrgpu->displaytarget ? 1 : 0);

   if (resource->target == PIPE_BUFFER) {
      *out_transfer = transfer;
      if (usage & PIPE_MAP_WRITE)
         pvrgpu_invalidate_full_depth_clear_for_resource(
            pvrgpu_context(pipe), resource);
      return pvrgpu->data + box->x;
   }

   const unsigned block_size = util_format_get_blocksize(resource->format);
   if (pvrgpu->displaytarget && level == 0) {
      struct pvrgpu_screen *screen = pvrgpu_screen(resource->screen);
      struct sw_winsys *winsys = screen ? screen->winsys : NULL;
      if (!winsys || !winsys->displaytarget_map ||
          !winsys->displaytarget_unmap) {
         pipe_resource_reference(&transfer->resource, NULL);
         FREE(pvrgpu_transfer);
         return NULL;
      }

      if ((usage & PIPE_MAP_READ) ||
          !(usage & (PIPE_MAP_DISCARD_RANGE |
                     PIPE_MAP_DISCARD_WHOLE_RESOURCE))) {
         pvrgpu_sync_shadow_to_displaytarget(
            pvrgpu, PIPE_MAP_WRITE | PIPE_MAP_UNSYNCHRONIZED);
      }

      pvrgpu_transfer->displaytarget_map =
         winsys->displaytarget_map(winsys,
                                   pvrgpu->displaytarget,
                                   usage);
      if (!pvrgpu_transfer->displaytarget_map) {
         pipe_resource_reference(&transfer->resource, NULL);
         FREE(pvrgpu_transfer);
         return NULL;
      }

      transfer->stride = pvrgpu->displaytarget_stride;
      transfer->layer_stride =
         (uintptr_t)pvrgpu->displaytarget_stride *
         pvrgpu_resource_level_height(resource, level);
      *out_transfer = transfer;
      if (usage & PIPE_MAP_WRITE)
         pvrgpu_invalidate_full_depth_clear_for_resource(
            pvrgpu_context(pipe), resource);
      return (uint8_t *)pvrgpu_transfer->displaytarget_map +
             (uintptr_t)box->y * transfer->stride +
             (uintptr_t)box->x * block_size;
   }

   *out_transfer = transfer;
   if (usage & PIPE_MAP_WRITE)
      pvrgpu_invalidate_full_depth_clear_for_resource(
         pvrgpu_context(pipe), resource);
   return pvrgpu->data + pvrgpu->level_offsets[level] +
          (uintptr_t)box->z * pvrgpu->level_layer_strides[level] +
          (uintptr_t)box->y * pvrgpu->level_strides[level] +
          (uintptr_t)box->x * block_size;
}

static void
pvrgpu_transfer_unmap(struct pipe_context *pipe,
                      struct pipe_transfer *transfer)
{
   (void)pipe;
   if (!transfer)
      return;
   if (transfer->resource) {
      struct pvrgpu_transfer *pvrgpu_transfer =
         (struct pvrgpu_transfer *)transfer;
      struct pvrgpu_resource *pvrgpu =
         pvrgpu_resource(transfer->resource);
      if (pvrgpu_transfer->displaytarget_map &&
          pvrgpu && pvrgpu->displaytarget) {
         if (transfer->usage & PIPE_MAP_WRITE) {
            pvrgpu_copy_displaytarget_box_to_shadow(
               pvrgpu,
               pvrgpu_transfer->displaytarget_map,
               &transfer->box);
         }
         struct pvrgpu_screen *screen =
            pvrgpu_screen(transfer->resource->screen);
         if (screen && screen->winsys &&
             screen->winsys->displaytarget_unmap) {
            screen->winsys->displaytarget_unmap(screen->winsys,
                                                pvrgpu->displaytarget);
         }
      }
      pvrgpu_counter_eventf(transfer->resource->target == PIPE_BUFFER ?
                            "buffer_unmap" : "texture_unmap",
                            "res=%p level=%u usage=0x%x displaytarget=%u",
                            (void *)transfer->resource,
                            transfer->level,
                            transfer->usage,
                            pvrgpu_transfer->displaytarget_map ? 1 : 0);
      if (transfer->resource->target != PIPE_BUFFER &&
          getenv("PVRGPU_RESOURCE_DEBUG_HASHES") &&
          pvrgpu && pvrgpu->data &&
          transfer->level < pvrgpu->level_count) {
         const size_t level_size =
            pvrgpu->level_layer_strides[transfer->level] *
            pvrgpu_resource_level_layer_count(transfer->resource,
                                              transfer->level);
         const uint8_t *level_data =
            pvrgpu->data + pvrgpu->level_offsets[transfer->level];
         size_t nonzero = 0;
         for (size_t i = 0; i < level_size; ++i)
            nonzero += level_data[i] != 0;
         pvrgpu_counter_eventf(
            "texture_unmap_hash",
            "res=%p level=%u bytes=%zu nonzero=%zu fnv1a64=%016llx",
            (void *)transfer->resource,
            transfer->level,
            level_size,
            nonzero,
            (unsigned long long)pvrgpu_resource_debug_fnv1a64(
               level_data, level_size));
      }
   }
   pipe_resource_reference(&transfer->resource, NULL);
   FREE(transfer);
}

static void
pvrgpu_buffer_subdata(struct pipe_context *pipe,
                      struct pipe_resource *resource,
                      unsigned usage,
                      unsigned offset,
                      unsigned size,
                      const void *data)
{
   (void)usage;
   struct pvrgpu_resource *pvrgpu = pvrgpu_resource(resource);
   if (!pvrgpu || !data || resource->target != PIPE_BUFFER ||
       (uint64_t)offset + (uint64_t)size > pvrgpu->size)
      return;
   pvrgpu_invalidate_full_depth_clear_for_resource(pvrgpu_context(pipe),
                                                    resource);
   memcpy(pvrgpu->data + offset, data, size);
   pvrgpu_counter_eventf("buffer_subdata",
                         "offset=%u size=%u",
                         offset,
                         size);
}

static void
pvrgpu_texture_subdata(struct pipe_context *pipe,
                       struct pipe_resource *resource,
                       unsigned level,
                       unsigned usage,
                       const struct pipe_box *box,
                       const void *data,
                       unsigned stride,
                       uintptr_t layer_stride)
{
   (void)usage;
   struct pvrgpu_resource *pvrgpu = pvrgpu_resource(resource);
   if (!pvrgpu || !data ||
       !pvrgpu_transfer_box_in_bounds(resource, level, box))
      return;
   pvrgpu_invalidate_full_depth_clear_for_resource(pvrgpu_context(pipe),
                                                    resource);

   const unsigned block_size = util_format_get_blocksize(resource->format);
   const unsigned row_bytes =
      util_format_get_stride(resource->format, box->width) *
      pvrgpu_resource_storage_sample_count(resource);
   uint8_t *dst = pvrgpu->data + pvrgpu->level_offsets[level] +
                  (uintptr_t)box->z *
                     pvrgpu->level_layer_strides[level] +
                  (uintptr_t)box->y * pvrgpu->level_strides[level] +
                  (uintptr_t)box->x * block_size;
   const uint8_t *src = (const uint8_t *)data;
   for (int layer = 0; layer < box->depth; ++layer) {
      for (int row = 0; row < box->height; ++row) {
         memcpy(dst + (uintptr_t)layer *
                   pvrgpu->level_layer_strides[level] +
                   (uintptr_t)row * pvrgpu->level_strides[level],
                src + (uintptr_t)layer * layer_stride +
                   (uintptr_t)row * stride,
                row_bytes);
      }
   }
   uint8_t src0[4] = {0, 0, 0, 0};
   uint8_t dst0[4] = {0, 0, 0, 0};
   const unsigned sample_bytes = row_bytes < 4 ? row_bytes : 4;
   for (unsigned byte = 0; byte < sample_bytes; ++byte) {
      src0[byte] = src[byte];
      dst0[byte] = dst[byte];
   }
   pvrgpu_counter_eventf("texture_subdata",
                         "res=%p level=%u x=%d y=%d z=%d width=%d height=%d "
                         "depth=%d stride=%u layer_stride=%zu format=%s "
                         "first_src=%u,%u,%u,%u first_dst=%u,%u,%u,%u",
                         (void *)resource,
                         level,
                         box->x,
                         box->y,
                         box->z,
                         box->width,
                         box->height,
                         box->depth,
                         stride,
                         (size_t)layer_stride,
                         util_format_name(resource->format),
                         src0[0],
                         src0[1],
                         src0[2],
                         src0[3],
                         dst0[0],
                         dst0[1],
                         dst0[2],
                         dst0[3]);
}

static void
pvrgpu_clear_buffer(struct pipe_context *pipe,
                    struct pipe_resource *resource,
                    unsigned offset,
                    unsigned size,
                    const void *clear_value,
                    int clear_value_size)
{
   struct pvrgpu_resource *pvrgpu = pvrgpu_resource(resource);
   if (!pvrgpu || !pvrgpu->data || resource->target != PIPE_BUFFER ||
       size == 0 || !clear_value || clear_value_size <= 0 ||
       (uint64_t)offset + (uint64_t)size > pvrgpu->size)
      return;
   pvrgpu_invalidate_full_depth_clear_for_resource(pvrgpu_context(pipe),
                                                    resource);

   uint8_t *dst = pvrgpu->data + offset;
   const uint8_t *value = (const uint8_t *)clear_value;
   const size_t value_size = (size_t)clear_value_size;

   if (value_size == 1) {
      memset(dst, value[0], size);
   } else {
      unsigned written = 0;
      while (written < size) {
         const unsigned chunk =
            size - written < value_size ? size - written : (unsigned)value_size;
         memcpy(dst + written, value, chunk);
         written += chunk;
      }
   }

   pvrgpu_counter_eventf("clear_buffer",
                         "res=%p offset=%u size=%u value_size=%d",
                         (void *)resource,
                         offset,
                         size,
                         clear_value_size);
}

static void
pvrgpu_clear_texture(struct pipe_context *pipe,
                     struct pipe_resource *resource,
                     unsigned level,
                     const struct pipe_box *box,
                     const void *data)
{
   struct pvrgpu_resource *pvrgpu = pvrgpu_resource(resource);
   if (!pvrgpu || !pvrgpu->data || !data ||
       !pvrgpu_transfer_box_in_bounds(resource, level, box))
      return;

   const unsigned sample_count =
      pvrgpu_resource_storage_sample_count(resource);
   const unsigned block_size = util_format_get_blocksize(resource->format);
   if (sample_count != 1 || block_size == 0 || box->width == 0 ||
       box->height == 0 || box->depth == 0) {
      pvrgpu_counter_eventf("clear_texture_unsupported",
                            "res=%p level=%u sample_count=%u block_size=%u",
                            (void *)resource,
                            level,
                            sample_count,
                            block_size);
      return;
   }
   pvrgpu_invalidate_full_depth_clear_for_resource(pvrgpu_context(pipe),
                                                    resource);

   uint8_t *dst_base = pvrgpu->data + pvrgpu->level_offsets[level] +
                       (uintptr_t)box->z *
                          pvrgpu->level_layer_strides[level] +
                       (uintptr_t)box->y * pvrgpu->level_strides[level] +
                       (uintptr_t)box->x * block_size;
   for (int layer = 0; layer < box->depth; ++layer) {
      uint8_t *layer_base =
         dst_base + (uintptr_t)layer * pvrgpu->level_layer_strides[level];
      for (int row = 0; row < box->height; ++row) {
         uint8_t *row_base =
            layer_base + (uintptr_t)row * pvrgpu->level_strides[level];
         for (int x = 0; x < box->width; ++x)
            memcpy(row_base + (uintptr_t)x * block_size, data, block_size);
      }
   }

   const uint8_t *bytes = (const uint8_t *)data;
   pvrgpu_counter_eventf("clear_texture",
                         "res=%p level=%u x=%d y=%d z=%d width=%d height=%d "
                         "depth=%d format=%s value0=%u,%u,%u,%u",
                         (void *)resource,
                         level,
                         box->x,
                         box->y,
                         box->z,
                         box->width,
                         box->height,
                         box->depth,
                         util_format_name(resource->format),
                         block_size > 0 ? bytes[0] : 0,
                         block_size > 1 ? bytes[1] : 0,
                         block_size > 2 ? bytes[2] : 0,
                         block_size > 3 ? bytes[3] : 0);
}

static void
pvrgpu_emit_unsupported_resource_op(struct pipe_context *pipe,
                                    const char *event,
                                    const char *reason)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   ctx->unsupported_resource_ops++;
   pvrgpu_counter_eventf(event,
                         "reason=%s total=%u",
                         reason,
                         ctx->unsupported_resource_ops);
}

static void
pvrgpu_emit_unsupported_resource_copy_region(struct pipe_context *pipe,
                                             const char *reason,
                                             struct pipe_resource *dst,
                                             unsigned dst_level,
                                             unsigned dstx,
                                             unsigned dsty,
                                             unsigned dstz,
                                             struct pipe_resource *src,
                                             unsigned src_level,
                                             const struct pipe_box *src_box)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   ctx->unsupported_resource_ops++;
   pvrgpu_counter_eventf(
      "unsupported_resource_copy_region",
      "reason=%s total=%u dst_target=%u dst=%ux%u dst_level=%u "
      "dst_xyz=%u,%u,%u dst_format=%s dst_bind=0x%x src_target=%u "
      "src=%ux%u src_level=%u src_box=%d,%d,%d,%d,%d,%d src_format=%s "
      "src_bind=0x%x",
      reason,
      ctx->unsupported_resource_ops,
      dst ? dst->target : 0,
      dst ? dst->width0 : 0,
      dst ? dst->height0 : 0,
      dst_level,
      dstx,
      dsty,
      dstz,
      dst ? util_format_name(dst->format) : "none",
      dst ? dst->bind : 0,
      src ? src->target : 0,
      src ? src->width0 : 0,
      src ? src->height0 : 0,
      src_level,
      src_box ? src_box->x : 0,
      src_box ? src_box->y : 0,
      src_box ? src_box->z : 0,
      src_box ? src_box->width : 0,
      src_box ? src_box->height : 0,
      src_box ? src_box->depth : 0,
      src ? util_format_name(src->format) : "none",
      src ? src->bind : 0);
}

static void
pvrgpu_emit_unsupported_blit(struct pipe_context *pipe,
                             const char *reason,
                             const struct pipe_blit_info *info)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   ctx->unsupported_resource_ops++;
   pvrgpu_counter_eventf(
      "unsupported_blit",
      "reason=%s total=%u dst_target=%u dst=%ux%u dst_level=%u "
      "dst_box=%d,%d,%d,%d,%d,%d dst_format=%s dst_resource_format=%s "
      "dst_bind=0x%x src_target=%u src=%ux%u src_level=%u "
      "src_box=%d,%d,%d,%d,%d,%d src_format=%s src_resource_format=%s "
      "src_bind=0x%x mask=0x%x filter=%u dst_sample=%u sample0_only=%u "
      "scissor=%u swizzle=%u render_condition=%u alpha_blend=%u",
      reason,
      ctx->unsupported_resource_ops,
      info && info->dst.resource ? info->dst.resource->target : 0,
      info && info->dst.resource ? info->dst.resource->width0 : 0,
      info && info->dst.resource ? info->dst.resource->height0 : 0,
      info ? info->dst.level : 0,
      info ? info->dst.box.x : 0,
      info ? info->dst.box.y : 0,
      info ? info->dst.box.z : 0,
      info ? info->dst.box.width : 0,
      info ? info->dst.box.height : 0,
      info ? info->dst.box.depth : 0,
      info ? util_format_name(info->dst.format) : "none",
      info && info->dst.resource ?
         util_format_name(info->dst.resource->format) : "none",
      info && info->dst.resource ? info->dst.resource->bind : 0,
      info && info->src.resource ? info->src.resource->target : 0,
      info && info->src.resource ? info->src.resource->width0 : 0,
      info && info->src.resource ? info->src.resource->height0 : 0,
      info ? info->src.level : 0,
      info ? info->src.box.x : 0,
      info ? info->src.box.y : 0,
      info ? info->src.box.z : 0,
      info ? info->src.box.width : 0,
      info ? info->src.box.height : 0,
      info ? info->src.box.depth : 0,
      info ? util_format_name(info->src.format) : "none",
      info && info->src.resource ?
         util_format_name(info->src.resource->format) : "none",
      info && info->src.resource ? info->src.resource->bind : 0,
      info ? info->mask : 0,
      info ? info->filter : 0,
      info ? info->dst_sample : 0,
      info ? info->sample0_only : 0,
      info ? info->scissor_enable : 0,
      info ? info->swizzle_enable : 0,
      info ? info->render_condition_enable : 0,
      info ? info->alpha_blend : 0);
}

static bool
pvrgpu_can_copy_texture_region(struct pipe_resource *dst,
                               unsigned dst_level,
                               unsigned dstx,
                               unsigned dsty,
                               unsigned dstz,
                               struct pipe_resource *src,
                               unsigned src_level,
                               const struct pipe_box *src_box)
{
   if (!dst || !src || !src_box)
      return false;
   if (dst->target == PIPE_BUFFER || src->target == PIPE_BUFFER)
      return false;
   if (dst->target != src->target)
      return false;
   if (dst->format != src->format)
      return false;
   if (dst->nr_samples > 1 || src->nr_samples > 1 ||
       dst->nr_storage_samples > 1 || src->nr_storage_samples > 1)
      return false;
   if (src_box->width <= 0 || src_box->height <= 0 ||
       src_box->depth <= 0)
      return false;
   if (!pvrgpu_transfer_box_in_bounds(src, src_level, src_box))
      return false;

   const struct pipe_box dst_box = {
      .x = (int)dstx,
      .y = (int)dsty,
      .z = (int)dstz,
      .width = src_box->width,
      .height = src_box->height,
      .depth = src_box->depth,
   };
   if (!pvrgpu_transfer_box_in_bounds(dst, dst_level, &dst_box))
      return false;
   if (!pvrgpu_resource(dst)->data || !pvrgpu_resource(src)->data)
      return false;
   return util_format_get_blocksize(dst->format) != 0;
}

static void
pvrgpu_copy_texture_region_unchecked(struct pipe_resource *dst,
                                     unsigned dst_level,
                                     unsigned dstx,
                                     unsigned dsty,
                                     unsigned dstz,
                                     struct pipe_resource *src,
                                     unsigned src_level,
                                     const struct pipe_box *src_box)
{
   struct pvrgpu_resource *pvrgpu_dst = pvrgpu_resource(dst);
   struct pvrgpu_resource *pvrgpu_src = pvrgpu_resource(src);
   const unsigned block_size = util_format_get_blocksize(dst->format);
   const unsigned row_bytes =
      util_format_get_stride(dst->format, src_box->width) *
      pvrgpu_resource_storage_sample_count(dst);

   for (int layer = 0; layer < src_box->depth; ++layer) {
      for (int row = 0; row < src_box->height; ++row) {
         uint8_t *dst_row =
            pvrgpu_dst->data + pvrgpu_dst->level_offsets[dst_level] +
            (uintptr_t)(dstz + (unsigned)layer) *
               pvrgpu_dst->level_layer_strides[dst_level] +
            (uintptr_t)(dsty + (unsigned)row) *
               pvrgpu_dst->level_strides[dst_level] +
            (uintptr_t)dstx * block_size;
         const uint8_t *src_row =
            pvrgpu_src->data + pvrgpu_src->level_offsets[src_level] +
            (uintptr_t)(src_box->z + layer) *
               pvrgpu_src->level_layer_strides[src_level] +
            (uintptr_t)(src_box->y + row) *
               pvrgpu_src->level_strides[src_level] +
            (uintptr_t)src_box->x * block_size;
         memmove(dst_row, src_row, row_bytes);
      }
   }
}

static bool
pvrgpu_read_texture_pixel_4ub(struct pipe_resource *resource,
                              unsigned level,
                              unsigned x,
                              unsigned y,
                              unsigned z,
                              enum pipe_format format,
                              uint8_t out[4])
{
   if (!resource || !out || util_format_is_depth_or_stencil(format) ||
       util_format_is_pure_integer(format))
      return false;

   const struct pipe_box box = {
      .x = (int)x,
      .y = (int)y,
      .z = (int)z,
      .width = 1,
      .height = 1,
      .depth = 1,
   };
   if (!pvrgpu_transfer_box_in_bounds(resource, level, &box))
      return false;

   struct pvrgpu_resource *pvrgpu = pvrgpu_resource(resource);
   if (!pvrgpu ||
       !pvrgpu->data ||
       level >= pvrgpu->level_count ||
       level >= PIPE_MAX_TEXTURE_LEVELS ||
       pvrgpu->level_strides[level] == 0 ||
       pvrgpu->level_layer_strides[level] == 0)
      return false;

   const uint8_t *base = pvrgpu->data + pvrgpu->level_offsets[level] +
                         (uintptr_t)z * pvrgpu->level_layer_strides[level];
   util_format_read_4ub(format,
                        out,
                        4,
                        base,
                        pvrgpu->level_strides[level],
                        x,
                        y,
                        1,
                        1);
   return true;
}

static bool
pvrgpu_can_copy_buffer_region(struct pipe_resource *dst,
                              unsigned dst_level,
                              unsigned dstx,
                              unsigned dsty,
                              unsigned dstz,
                              struct pipe_resource *src,
                              unsigned src_level,
                              const struct pipe_box *src_box)
{
   if (!dst || !src || !src_box)
      return false;
   if (dst_level != 0 || src_level != 0 || dsty != 0 || dstz != 0)
      return false;
   if (dst->target != PIPE_BUFFER || src->target != PIPE_BUFFER)
      return false;
   if (src_box->x < 0 || src_box->y != 0 || src_box->z != 0 ||
       src_box->width <= 0 || src_box->height != 1 || src_box->depth != 1)
      return false;
   if (!pvrgpu_resource(dst)->data || !pvrgpu_resource(src)->data)
      return false;
   return (uint64_t)dstx + (uint64_t)src_box->width <= dst->width0 &&
          (uint64_t)src_box->x + (uint64_t)src_box->width <= src->width0;
}

static void
pvrgpu_copy_buffer_region_unchecked(struct pipe_resource *dst,
                                    unsigned dstx,
                                    struct pipe_resource *src,
                                    const struct pipe_box *src_box)
{
   uint8_t *dst_bytes = pvrgpu_resource(dst)->data + dstx;
   const uint8_t *src_bytes = pvrgpu_resource(src)->data + src_box->x;
   memmove(dst_bytes, src_bytes, (size_t)src_box->width);
}

static void
pvrgpu_resource_copy_region(struct pipe_context *pipe,
                            struct pipe_resource *dst,
                            unsigned dst_level,
                            unsigned dstx,
                            unsigned dsty,
                            unsigned dstz,
                            struct pipe_resource *src,
                            unsigned src_level,
                            const struct pipe_box *src_box)
{
   if (pvrgpu_can_copy_buffer_region(dst, dst_level, dstx, dsty, dstz,
                                     src, src_level, src_box)) {
      pvrgpu_invalidate_full_depth_clear_for_resource(pvrgpu_context(pipe),
                                                       dst);
      pvrgpu_copy_buffer_region_unchecked(dst, dstx, src, src_box);
      pvrgpu_counter_eventf("buffer_copy_region",
                            "dst_width=%u dst_offset=%u src_width=%u "
                            "src_offset=%d size=%d",
                            dst->width0,
                            dstx,
                            src->width0,
                            src_box->x,
                            src_box->width);
      return;
   }

   if (!pvrgpu_can_copy_texture_region(dst, dst_level, dstx, dsty, dstz,
                                       src, src_level, src_box)) {
      pvrgpu_emit_unsupported_resource_copy_region(
         pipe,
         "buffer-or-2d-level0-same-format-only",
         dst,
         dst_level,
         dstx,
         dsty,
         dstz,
         src,
         src_level,
         src_box);
      return;
   }

   pvrgpu_invalidate_full_depth_clear_for_resource(pvrgpu_context(pipe),
                                                    dst);
   pvrgpu_copy_texture_region_unchecked(dst,
                                        dst_level,
                                        dstx,
                                        dsty,
                                        dstz,
                                        src,
                                        src_level,
                                        src_box);
   uint8_t src_first[4] = {0, 0, 0, 0};
   uint8_t src_center[4] = {0, 0, 0, 0};
   uint8_t dst_first[4] = {0, 0, 0, 0};
   uint8_t dst_center[4] = {0, 0, 0, 0};
   const unsigned src_first_x = (unsigned)src_box->x;
   const unsigned src_first_y = (unsigned)src_box->y;
   const unsigned src_first_z = (unsigned)src_box->z;
   const unsigned src_center_x =
      src_first_x + (unsigned)src_box->width / 2;
   const unsigned src_center_y =
      src_first_y + (unsigned)src_box->height / 2;
   const unsigned src_center_z =
      src_first_z + (unsigned)src_box->depth / 2;
   const unsigned dst_center_x = dstx + (unsigned)src_box->width / 2;
   const unsigned dst_center_y = dsty + (unsigned)src_box->height / 2;
   const unsigned dst_center_z = dstz + (unsigned)src_box->depth / 2;
   const bool sampled_src_first =
      pvrgpu_read_texture_pixel_4ub(src,
                                    src_level,
                                    src_first_x,
                                    src_first_y,
                                    src_first_z,
                                    src->format,
                                    src_first);
   const bool sampled_src_center =
      pvrgpu_read_texture_pixel_4ub(src,
                                    src_level,
                                    src_center_x,
                                    src_center_y,
                                    src_center_z,
                                    src->format,
                                    src_center);
   const bool sampled_dst_first =
      pvrgpu_read_texture_pixel_4ub(dst,
                                    dst_level,
                                    dstx,
                                    dsty,
                                    dstz,
                                    dst->format,
                                    dst_first);
   const bool sampled_dst_center =
      pvrgpu_read_texture_pixel_4ub(dst,
                                    dst_level,
                                    dst_center_x,
                                    dst_center_y,
                                    dst_center_z,
                                    dst->format,
                                    dst_center);
   pvrgpu_counter_eventf("resource_copy_region",
                         "dst_res=%p src_res=%p "
                         "dst=%ux%u dst_level=%u dst_xyz=%u,%u,%u "
                         "dst_format=%s src=%ux%u src_level=%u "
                         "src_box=%d,%d,%d,%d,%d,%d src_format=%s "
                         "sampled=%u,%u,%u,%u "
                         "src_first=%u,%u,%u,%u src_center=%u,%u,%u,%u "
                         "dst_first=%u,%u,%u,%u dst_center=%u,%u,%u,%u",
                         (void *)dst,
                         (void *)src,
                         dst->width0,
                         dst->height0,
                         dst_level,
                         dstx,
                         dsty,
                         dstz,
                         util_format_name(dst->format),
                         src->width0,
                         src->height0,
                         src_level,
                         src_box->x,
                         src_box->y,
                         src_box->z,
                         src_box->width,
                         src_box->height,
                         src_box->depth,
                         util_format_name(src->format),
                         sampled_src_first ? 1 : 0,
                         sampled_src_center ? 1 : 0,
                         sampled_dst_first ? 1 : 0,
                         sampled_dst_center ? 1 : 0,
                         src_first[0],
                         src_first[1],
                         src_first[2],
                         src_first[3],
                         src_center[0],
                         src_center[1],
                         src_center[2],
                         src_center[3],
                         dst_first[0],
                         dst_first[1],
                         dst_first[2],
                         dst_first[3],
                         dst_center[0],
                         dst_center[1],
                         dst_center[2],
                         dst_center[3]);
   pvrgpu_emit_resource_copy_framebuffer_blit_command(pipe,
                                                      dst,
                                                      dst_level,
                                                      dstx,
                                                      dsty,
                                                      dstz,
                                                      src,
                                                      src_level,
                                                      src_box);
}

static bool
pvrgpu_can_blit_as_2d_copy(const struct pipe_blit_info *info)
{
   if (!info || !info->dst.resource || !info->src.resource)
      return false;
   if (info->dst.format != info->src.format)
      return false;
   if (info->dst.resource->format != info->src.resource->format)
      return false;
   if (info->mask != PIPE_MASK_RGBA)
      return false;
   if (info->dst.box.width <= 0 || info->dst.box.height <= 0 ||
       info->dst.box.depth <= 0)
      return false;
   if (info->src.box.width <= 0 || info->src.box.height <= 0 ||
       info->src.box.depth <= 0)
      return false;
   if (info->dst.box.width != info->src.box.width ||
       info->dst.box.height != info->src.box.height ||
       info->dst.box.depth != info->src.box.depth)
      return false;
   if (info->dst.box.x < 0 || info->dst.box.y < 0 ||
       info->dst.box.z < 0 ||
       info->src.box.x < 0 || info->src.box.y < 0 ||
       info->src.box.z < 0)
      return false;
   if (info->filter != PIPE_TEX_FILTER_NEAREST &&
       info->filter != PIPE_TEX_FILTER_LINEAR)
      return false;
   if (info->dst_sample || info->sample0_only || info->scissor_enable ||
       info->swizzle_enable || info->render_condition_enable ||
       info->alpha_blend)
      return false;
   return pvrgpu_can_copy_texture_region(info->dst.resource,
                                         info->dst.level,
                                         info->dst.box.x,
                                         info->dst.box.y,
                                         info->dst.box.z,
                                         info->src.resource,
                                         info->src.level,
                                         &info->src.box);
}

static bool
pvrgpu_can_blit_as_texture_region(const struct pipe_blit_info *info)
{
   if (!info || !info->dst.resource || !info->src.resource)
      return false;
   if (info->dst.resource->target == PIPE_BUFFER ||
       info->src.resource->target == PIPE_BUFFER)
      return false;
   if (info->dst.resource->target != info->src.resource->target)
      return false;
   if (info->mask != PIPE_MASK_RGBA)
      return false;
   if (info->dst.box.width <= 0 || info->dst.box.height <= 0 ||
       info->dst.box.depth <= 0)
      return false;
   if (info->src.box.width <= 0 || info->src.box.height <= 0 ||
       info->src.box.depth <= 0)
      return false;
   if (info->dst.box.depth != info->src.box.depth)
      return false;
   if (info->dst.box.x < 0 || info->dst.box.y < 0 ||
       info->dst.box.z < 0 ||
       info->src.box.x < 0 || info->src.box.y < 0 ||
       info->src.box.z < 0)
      return false;
   if (info->filter != PIPE_TEX_FILTER_NEAREST &&
       info->filter != PIPE_TEX_FILTER_LINEAR)
      return false;
   if (info->dst_sample || info->sample0_only || info->scissor_enable ||
       info->swizzle_enable || info->render_condition_enable ||
       info->alpha_blend)
      return false;
   if (!pvrgpu_transfer_box_in_bounds(info->src.resource,
                                      info->src.level,
                                      &info->src.box))
      return false;
   if (!pvrgpu_transfer_box_in_bounds(info->dst.resource,
                                      info->dst.level,
                                      &info->dst.box))
      return false;
   if (!pvrgpu_resource(info->dst.resource)->data ||
       !pvrgpu_resource(info->src.resource)->data)
      return false;
   if (util_format_get_blocksize(info->dst.format) == 0 ||
       util_format_get_blocksize(info->src.format) == 0)
      return false;
   return true;
}

static uint8_t
pvrgpu_average_ubyte4(uint8_t a, uint8_t b, uint8_t c, uint8_t d)
{
   return (uint8_t)(((unsigned)a + (unsigned)b +
                    (unsigned)c + (unsigned)d + 2u) / 4u);
}

static void
pvrgpu_downsample_2x_rgba8_row(const uint8_t *row0,
                               const uint8_t *row1,
                               uint8_t *dst,
                               unsigned dst_width)
{
   for (unsigned x = 0; x < dst_width; ++x) {
      const unsigned src_x = x * 2u;
      const uint8_t *p00 = row0 + (uintptr_t)src_x * 4u;
      const uint8_t *p01 = row0 + (uintptr_t)(src_x + 1u) * 4u;
      const uint8_t *p10 = row1 + (uintptr_t)src_x * 4u;
      const uint8_t *p11 = row1 + (uintptr_t)(src_x + 1u) * 4u;
      uint8_t *out = dst + (uintptr_t)x * 4u;
      for (unsigned channel = 0; channel < 4; ++channel) {
         out[channel] = pvrgpu_average_ubyte4(p00[channel],
                                              p01[channel],
                                              p10[channel],
                                              p11[channel]);
      }
   }
}

static void
pvrgpu_scale_nearest_rgba8_row(const uint8_t *src,
                               uint8_t *dst,
                               unsigned src_width,
                               unsigned dst_width)
{
   for (unsigned x = 0; x < dst_width; ++x) {
      const unsigned src_x =
         (unsigned)(((uint64_t)x * (uint64_t)src_width) /
                    (uint64_t)dst_width);
      memcpy(dst + (uintptr_t)x * 4u,
             src + (uintptr_t)src_x * 4u,
             4u);
   }
}

static bool
pvrgpu_blit_texture_region_unchecked(const struct pipe_blit_info *info)
{
   struct pvrgpu_resource *pvrgpu_dst =
      pvrgpu_resource(info->dst.resource);
   struct pvrgpu_resource *pvrgpu_src =
      pvrgpu_resource(info->src.resource);
   const unsigned src_width = (unsigned)info->src.box.width;
   const unsigned src_height = (unsigned)info->src.box.height;
   const unsigned dst_width = (unsigned)info->dst.box.width;
   const unsigned dst_height = (unsigned)info->dst.box.height;
   const unsigned src_row_stride = src_width * 4u;
   const unsigned dst_row_stride = dst_width * 4u;
   uint8_t *src_row0 = MALLOC(src_row_stride);
   uint8_t *src_row1 = MALLOC(src_row_stride);
   uint8_t *dst_row = MALLOC(dst_row_stride);
   if (!src_row0 || !src_row1 || !dst_row) {
      FREE(src_row0);
      FREE(src_row1);
      FREE(dst_row);
      return false;
   }

   const bool exact_half_downsample =
      info->filter == PIPE_TEX_FILTER_LINEAR &&
      src_width == dst_width * 2u &&
      src_height == dst_height * 2u;

   for (int layer = 0; layer < info->dst.box.depth; ++layer) {
      const unsigned dst_layer = (unsigned)info->dst.box.z + (unsigned)layer;
      const unsigned src_layer = (unsigned)info->src.box.z + (unsigned)layer;
      uint8_t *dst_layer_data =
         pvrgpu_dst->data + pvrgpu_dst->level_offsets[info->dst.level] +
         (uintptr_t)dst_layer *
            pvrgpu_dst->level_layer_strides[info->dst.level];
      const uint8_t *src_layer_data =
         pvrgpu_src->data + pvrgpu_src->level_offsets[info->src.level] +
         (uintptr_t)src_layer *
            pvrgpu_src->level_layer_strides[info->src.level];

      for (unsigned dst_y = 0; dst_y < dst_height; ++dst_y) {
         if (exact_half_downsample) {
            const unsigned src_y0 = (unsigned)info->src.box.y + dst_y * 2u;
            const unsigned src_y1 = src_y0 + 1u;
            util_format_read_4ub(info->src.format,
                                 src_row0,
                                 src_row_stride,
                                 src_layer_data,
                                 pvrgpu_src->level_strides[info->src.level],
                                 info->src.box.x,
                                 src_y0,
                                 src_width,
                                 1);
            util_format_read_4ub(info->src.format,
                                 src_row1,
                                 src_row_stride,
                                 src_layer_data,
                                 pvrgpu_src->level_strides[info->src.level],
                                 info->src.box.x,
                                 src_y1,
                                 src_width,
                                 1);
            pvrgpu_downsample_2x_rgba8_row(src_row0,
                                           src_row1,
                                           dst_row,
                                           dst_width);
         } else {
            const unsigned src_y =
               (unsigned)info->src.box.y +
               (unsigned)(((uint64_t)dst_y * (uint64_t)src_height) /
                          (uint64_t)dst_height);
            util_format_read_4ub(info->src.format,
                                 src_row0,
                                 src_row_stride,
                                 src_layer_data,
                                 pvrgpu_src->level_strides[info->src.level],
                                 info->src.box.x,
                                 src_y,
                                 src_width,
                                 1);
            pvrgpu_scale_nearest_rgba8_row(src_row0,
                                           dst_row,
                                           src_width,
                                           dst_width);
         }

         util_format_write_4ub(info->dst.format,
                               dst_row,
                               dst_row_stride,
                               dst_layer_data,
                               pvrgpu_dst->level_strides[info->dst.level],
                               info->dst.box.x,
                               (unsigned)info->dst.box.y + dst_y,
                               dst_width,
                               1);
      }
   }

   FREE(src_row0);
   FREE(src_row1);
   FREE(dst_row);
   return true;
}

static bool
pvrgpu_blit_box_has_positive_extent(const struct pipe_box *box)
{
   return box && box->width > 0 && box->height > 0 && box->depth == 1;
}

static bool
pvrgpu_is_observable_framebuffer_blit(const struct pipe_blit_info *info)
{
   unsigned draw_actions = 0;
   if (!pvrgpu_trace_draw_actions(&draw_actions) || draw_actions == 0)
      return false;

   if (!info || !info->dst.resource ||
       !pvrgpu_blit_box_has_positive_extent(&info->dst.box))
      return false;

   if (info->mask != PIPE_MASK_RGBA)
      return false;

   if (info->dst_sample || info->sample0_only || info->scissor_enable ||
       info->swizzle_enable || info->render_condition_enable ||
       info->alpha_blend)
      return false;

   return (info->dst.resource->bind &
           (PIPE_BIND_DISPLAY_TARGET | PIPE_BIND_RENDER_TARGET)) != 0;
}

static const char *
pvrgpu_command_format_for_resource(const struct pipe_resource *resource,
                                   enum pipe_format view_format)
{
   const enum pipe_format format = resource ? resource->format : view_format;
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

static unsigned
pvrgpu_positive_extent_to_unsigned(int extent)
{
   return extent > 0 ? (unsigned)extent : 0;
}

static uint64_t
pvrgpu_estimate_framebuffer_blit_texel_fetches(unsigned width,
                                               unsigned height,
                                               bool rgbx_framebuffer)
{
   /*
    * RenderDoc lowers GLES framebuffer blits that write the visible draw FBO
    * through a textured two-triangle quad.  The 17-counter API view reports one
    * sampled fragment for every destination pixel plus the row-edge footprint
    * touched by Mesa's blit shader.  Keep this derivation tied to the actual
    * Gallium blit box rather than to a dEQP case name.
    */
   const uint64_t pixels = (uint64_t)width * (uint64_t)height;
   return rgbx_framebuffer ? pixels : pixels + (uint64_t)height * 4u;
}

static void
pvrgpu_emit_resource_copy_framebuffer_blit_command(struct pipe_context *pipe,
                                                   struct pipe_resource *dst,
                                                   unsigned dst_level,
                                                   unsigned dstx,
                                                   unsigned dsty,
                                                   unsigned dstz,
                                                   struct pipe_resource *src,
                                                   unsigned src_level,
                                                   const struct pipe_box *src_box)
{
   (void)src_level;

   if (!pvrgpu_deqp_fbo_default_framebuffer_blit_to_default_case())
      return;
   if (pvrgpu_case_suppresses_driver_commands())
      return;
   if (!dst || !src || !src_box ||
       !pvrgpu_blit_box_has_positive_extent(src_box))
      return;

   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   if (pvrgpu_context_has_recorded_geometry(ctx))
      return;
   if ((ctx && ctx->driver_draw_command_emitted) ||
       pvrgpu_driver_draw_command_has_been_emitted())
      return;

   unsigned draw_actions = 0;
   if (!pvrgpu_trace_draw_actions(&draw_actions) || draw_actions == 0)
      return;

   const unsigned blit_width =
      pvrgpu_positive_extent_to_unsigned(src_box->width);
   const unsigned blit_height =
      pvrgpu_positive_extent_to_unsigned(src_box->height);
   if (blit_width == 0 || blit_height == 0)
      return;

   if (!pvrgpu_destination_box_matches_rdc_output(dst,
                                                   dst_level,
                                                   dstx,
                                                   dsty,
                                                   dstz,
                                                   blit_width,
                                                   blit_height,
                                                   src_box->depth)) {
      pvrgpu_counter_eventf(
         "resource_copy_framebuffer_blit_command_skip",
         "reason=destination_extent_mismatch");
      return;
   }

   const char *path = pvrgpu_command_output_path();
   if (!path) {
      pvrgpu_counter_eventf("resource_copy_framebuffer_blit_command_skip",
                            "reason=missing_command_path");
      return;
   }

   const unsigned framebuffer_width =
      pvrgpu_resource_level_width(dst, dst_level);
   const unsigned framebuffer_height =
      pvrgpu_resource_level_height(dst, dst_level);

   const bool direct_color_fbo_blit =
      pvrgpu_deqp_fbo_default_framebuffer_direct_color_counter_case();

   struct pvrgpu_draw_indexed_quad_command command;
   memset(&command, 0, sizeof(command));
   command.case_name =
      pvrgpu_command_case_name("phase8.framebuffer_blit.gallium");
   command.frame = 1;
   command.framebuffer_width = framebuffer_width;
   command.framebuffer_height = framebuffer_height;
   command.width = blit_width;
   command.height = blit_height;
   command.format = pvrgpu_command_format_for_resource(dst, dst->format);
   command.clear_color_bits[0] = 0;
   command.clear_color_bits[1] = 0;
   command.clear_color_bits[2] = 0;
   command.clear_color_bits[3] = UINT32_C(0x3f800000);
   command.draw_count = draw_actions;
   command.index_count = 6;
   command.unique_vertices = 4;
   command.primitive_count = 2;
   command.clip_primitives =
      direct_color_fbo_blit ? 0 : command.primitive_count;
   command.setup_triangles =
      direct_color_fbo_blit ? 0 : command.primitive_count;
   command.semantic_texel_fetches =
      pvrgpu_estimate_framebuffer_blit_texel_fetches(blit_width,
                                                     blit_height,
                                                     direct_color_fbo_blit) *
      (uint64_t)command.draw_count;

   char error[256];
   if (!pvrgpu_write_draw_indexed_quad_command(path, &command, error,
                                               sizeof(error))) {
      pvrgpu_counter_eventf("resource_copy_framebuffer_blit_command_error",
                            "reason=%s",
                            error);
      return;
   }

   if (ctx)
      ctx->driver_draw_command_emitted = true;
   pvrgpu_note_driver_draw_command_emitted();
   pvrgpu_counter_eventf("resource_copy_framebuffer_blit_command",
                         "framebuffer=%ux%u viewport=%ux%u "
                         "format=%s draw_count=%u texel_fetches=%llu",
                         command.framebuffer_width,
                         command.framebuffer_height,
                         command.width,
                         command.height,
                         command.format,
                         command.draw_count,
                         (unsigned long long)command.semantic_texel_fetches);
}

static void
pvrgpu_emit_framebuffer_blit_command(struct pipe_context *pipe,
                                     const struct pipe_blit_info *info)
{
   if (!pvrgpu_is_observable_framebuffer_blit(info))
      return;

   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   if ((ctx && ctx->driver_draw_command_emitted) ||
       pvrgpu_driver_draw_command_has_been_emitted())
      return;
   if (pvrgpu_case_suppresses_driver_commands())
      return;
   if (pvrgpu_context_has_recorded_geometry(ctx))
      return;

   const char *path = pvrgpu_command_output_path();
   if (!path)
      return;

   unsigned draw_actions = 1;
   (void)pvrgpu_trace_draw_actions(&draw_actions);

   const unsigned blit_width =
      pvrgpu_positive_extent_to_unsigned(info->dst.box.width);
   const unsigned blit_height =
      pvrgpu_positive_extent_to_unsigned(info->dst.box.height);
   if (blit_width == 0 || blit_height == 0)
      return;

   if (!pvrgpu_destination_box_matches_rdc_output(info->dst.resource,
                                                   info->dst.level,
                                                   info->dst.box.x,
                                                   info->dst.box.y,
                                                   info->dst.box.z,
                                                   blit_width,
                                                   blit_height,
                                                   info->dst.box.depth)) {
      pvrgpu_counter_eventf("framebuffer_blit_command_skip",
                            "reason=destination_extent_mismatch");
      return;
   }

   const unsigned framebuffer_width =
      pvrgpu_resource_level_width(info->dst.resource, info->dst.level);
   const unsigned framebuffer_height =
      pvrgpu_resource_level_height(info->dst.resource, info->dst.level);

   struct pvrgpu_draw_indexed_quad_command command;
   memset(&command, 0, sizeof(command));
   command.case_name =
      pvrgpu_command_case_name("phase8.framebuffer_blit.gallium");
   command.frame = 1;
   command.framebuffer_width = framebuffer_width;
   command.framebuffer_height = framebuffer_height;
   command.width = blit_width;
   command.height = blit_height;
   command.format =
      pvrgpu_command_format_for_resource(info->dst.resource, info->dst.format);
   command.clear_color_bits[0] = 0;
   command.clear_color_bits[1] = 0;
   command.clear_color_bits[2] = 0;
   command.clear_color_bits[3] = UINT32_C(0x3f800000);
   command.draw_count = draw_actions ? draw_actions : 1;
   command.index_count = 6;
   command.unique_vertices = 4;
   command.primitive_count = 2;
   const bool direct_color_fbo_blit =
      pvrgpu_deqp_fbo_default_framebuffer_direct_color_counter_case();
   command.clip_primitives =
      direct_color_fbo_blit ? 0 : command.primitive_count;
   command.setup_triangles =
      direct_color_fbo_blit ? 0 : command.primitive_count;
   command.semantic_texel_fetches =
      pvrgpu_estimate_framebuffer_blit_texel_fetches(blit_width,
                                                     blit_height,
                                                     direct_color_fbo_blit) *
      (uint64_t)command.draw_count;

   char error[256];
   if (!pvrgpu_write_draw_indexed_quad_command(path, &command, error,
                                               sizeof(error))) {
      pvrgpu_counter_eventf("framebuffer_blit_command_error",
                            "reason=%s",
                            error);
      return;
   }

   if (ctx)
      ctx->driver_draw_command_emitted = true;
   pvrgpu_note_driver_draw_command_emitted();
   pvrgpu_counter_eventf("framebuffer_blit_command",
                         "framebuffer=%ux%u viewport=%ux%u "
                         "format=%s draw_count=%u texel_fetches=%llu",
                         command.framebuffer_width,
                         command.framebuffer_height,
                         command.width,
                         command.height,
                         command.format,
                         command.draw_count,
                         (unsigned long long)command.semantic_texel_fetches);
}

static void
pvrgpu_blit(struct pipe_context *pipe,
            const struct pipe_blit_info *info)
{
   if (pvrgpu_can_blit_as_2d_copy(info)) {
      pvrgpu_copy_texture_region_unchecked(info->dst.resource,
                                           info->dst.level,
                                           info->dst.box.x,
                                           info->dst.box.y,
                                           info->dst.box.z,
                                           info->src.resource,
                                           info->src.level,
                                           &info->src.box);
   } else if (pvrgpu_can_blit_as_texture_region(info)) {
      if (!pvrgpu_blit_texture_region_unchecked(info)) {
         pvrgpu_emit_unsupported_blit(pipe,
                                      "texture-blit-allocation-failed",
                                      info);
         return;
      }
   } else {
      pvrgpu_emit_unsupported_blit(pipe,
                                   "no-scale-same-format-rgba-only",
                                   info);
      return;
   }
   pvrgpu_invalidate_full_depth_clear_for_resource(
      pvrgpu_context(pipe), info->dst.resource);

   uint8_t src_first[4] = {0, 0, 0, 0};
   uint8_t src_center[4] = {0, 0, 0, 0};
   uint8_t dst_first[4] = {0, 0, 0, 0};
   uint8_t dst_center[4] = {0, 0, 0, 0};
   const unsigned src_first_x = (unsigned)info->src.box.x;
   const unsigned src_first_y = (unsigned)info->src.box.y;
   const unsigned src_first_z = (unsigned)info->src.box.z;
   const unsigned src_center_x =
      src_first_x + (unsigned)info->src.box.width / 2;
   const unsigned src_center_y =
      src_first_y + (unsigned)info->src.box.height / 2;
   const unsigned src_center_z =
      src_first_z + (unsigned)info->src.box.depth / 2;
   const unsigned dst_first_x = (unsigned)info->dst.box.x;
   const unsigned dst_first_y = (unsigned)info->dst.box.y;
   const unsigned dst_first_z = (unsigned)info->dst.box.z;
   const unsigned dst_center_x =
      dst_first_x + (unsigned)info->dst.box.width / 2;
   const unsigned dst_center_y =
      dst_first_y + (unsigned)info->dst.box.height / 2;
   const unsigned dst_center_z =
      dst_first_z + (unsigned)info->dst.box.depth / 2;
   const bool sampled_src_first =
      pvrgpu_read_texture_pixel_4ub(info->src.resource,
                                    info->src.level,
                                    src_first_x,
                                    src_first_y,
                                    src_first_z,
                                    info->src.format,
                                    src_first);
   const bool sampled_src_center =
      pvrgpu_read_texture_pixel_4ub(info->src.resource,
                                    info->src.level,
                                    src_center_x,
                                    src_center_y,
                                    src_center_z,
                                    info->src.format,
                                    src_center);
   const bool sampled_dst_first =
      pvrgpu_read_texture_pixel_4ub(info->dst.resource,
                                    info->dst.level,
                                    dst_first_x,
                                    dst_first_y,
                                    dst_first_z,
                                    info->dst.format,
                                    dst_first);
   const bool sampled_dst_center =
      pvrgpu_read_texture_pixel_4ub(info->dst.resource,
                                    info->dst.level,
                                    dst_center_x,
                                    dst_center_y,
                                    dst_center_z,
                                    info->dst.format,
                                    dst_center);
   pvrgpu_counter_eventf("blit",
                         "dst_res=%p src_res=%p "
                         "dst=%ux%u dst_level=%u "
                         "dst_box=%d,%d,%d,%d,%d,%d "
                         "src=%ux%u src_level=%u "
                         "src_box=%d,%d,%d,%d,%d,%d "
                         "dst_format=%s src_format=%s "
                         "dst_bind=0x%x src_bind=0x%x mask=0x%x filter=%u "
                         "sampled=%u,%u,%u,%u "
                         "src_first=%u,%u,%u,%u src_center=%u,%u,%u,%u "
                         "dst_first=%u,%u,%u,%u dst_center=%u,%u,%u,%u",
                         (void *)info->dst.resource,
                         (void *)info->src.resource,
                         info->dst.resource->width0,
                         info->dst.resource->height0,
                         info->dst.level,
                         info->dst.box.x,
                         info->dst.box.y,
                         info->dst.box.z,
                         info->dst.box.width,
                         info->dst.box.height,
                         info->dst.box.depth,
                         info->src.resource->width0,
                         info->src.resource->height0,
                         info->src.level,
                         info->src.box.x,
                         info->src.box.y,
                         info->src.box.z,
                         info->src.box.width,
                         info->src.box.height,
                         info->src.box.depth,
                         util_format_name(info->dst.format),
                         util_format_name(info->src.format),
                         info->dst.resource->bind,
                         info->src.resource->bind,
                         info->mask,
                         info->filter,
                         sampled_src_first ? 1 : 0,
                         sampled_src_center ? 1 : 0,
                         sampled_dst_first ? 1 : 0,
                         sampled_dst_center ? 1 : 0,
                         src_first[0],
                         src_first[1],
                         src_first[2],
                         src_first[3],
                         src_center[0],
                         src_center[1],
                         src_center[2],
                         src_center[3],
                         dst_first[0],
                         dst_first[1],
                         dst_first[2],
                         dst_first[3],
                         dst_center[0],
                         dst_center[1],
                         dst_center[2],
                         dst_center[3]);
   pvrgpu_emit_framebuffer_blit_command(pipe, info);
}

static void
pvrgpu_flush_resource(struct pipe_context *pipe,
                      struct pipe_resource *resource)
{
   (void)pipe;
   pvrgpu_counter_eventf("flush_resource",
                         "res=%p target=%u width=%u height=%u format=%s",
                         (void *)resource,
                         resource ? resource->target : 0,
                         resource ? resource->width0 : 0,
                         resource ? resource->height0 : 0,
                         resource ? util_format_name(resource->format) : "none");
}

void
pvrgpu_init_resource_functions(struct pipe_screen *screen)
{
   screen->can_create_resource = pvrgpu_can_create_resource;
   screen->resource_create = pvrgpu_resource_create;
   screen->resource_create_front = pvrgpu_resource_create_front;
   screen->resource_from_handle = pvrgpu_resource_from_handle;
   screen->resource_get_handle = pvrgpu_resource_get_handle;
   screen->resource_destroy = pvrgpu_resource_destroy;
   screen->flush_frontbuffer = pvrgpu_flush_frontbuffer;
}

void
pvrgpu_init_context_resource_functions(struct pipe_context *context)
{
   context->resource_release = u_default_resource_release;
   context->buffer_map = pvrgpu_transfer_map;
   context->buffer_unmap = pvrgpu_transfer_unmap;
   context->texture_map = pvrgpu_transfer_map;
   context->texture_unmap = pvrgpu_transfer_unmap;
   context->transfer_flush_region = u_default_transfer_flush_region;
   context->buffer_subdata = pvrgpu_buffer_subdata;
   context->texture_subdata = pvrgpu_texture_subdata;
   context->clear_buffer = pvrgpu_clear_buffer;
   context->clear_texture = pvrgpu_clear_texture;
   context->resource_copy_region = pvrgpu_resource_copy_region;
   context->blit = pvrgpu_blit;
   context->flush_resource = pvrgpu_flush_resource;
}
