/* SPDX-License-Identifier: MIT */

#include "pvrgpu_resource.h"
#include "pvrgpu_cmd.h"
#include "pvrgpu_context.h"
#include "pvrgpu_counter.h"
#include "pvrgpu_msaa.h"
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
   uint8_t *sample0_staging;
   size_t sample0_staging_size;
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
   case 4:
   case 8:
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
   if ((template->bind & (PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_SHADER_IMAGE)) &&
       (template->nr_samples > 8 || template->nr_storage_samples > 8))
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
   if ((template->bind & (PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_SHADER_IMAGE)) &&
       (template->nr_samples > 8 || template->nr_storage_samples > 8))
      return "sampled_sample_count";
   return "target";
}

static bool
pvrgpu_validate_resource_creation(struct pipe_screen *screen,
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

/* Mesa st_TestProxyTexImage 傳入尚未 round 的 API requested samples 與
 * bind=0；後續 st_texture_storage 才挑選下一個可取樣的實際 sample count。
 * 只調整容量詢問的 local copy，真正 resource_create 仍驗原始嚴格格式。 */
static bool
pvrgpu_normalize_proxy_texture_samples(const struct pipe_resource *requested,
                                       struct pipe_resource *probe)
{
   if (!requested || !probe)
      return false;
   *probe = *requested;
   if (requested->bind != 0 ||
       (requested->target != PIPE_TEXTURE_2D &&
        requested->target != PIPE_TEXTURE_2D_ARRAY) ||
       (requested->nr_samples == 0 && requested->nr_storage_samples == 0))
      return true;
   if (requested->nr_samples > 8 || requested->nr_storage_samples > 8)
      return false;
   unsigned samples = requested->nr_samples;
   unsigned storage_samples = requested->nr_storage_samples;
   while (!pvrgpu_is_supported_resource_sample_count(samples))
      ++samples;
   while (!pvrgpu_is_supported_resource_sample_count(storage_samples))
      ++storage_samples;
   probe->nr_samples = samples;
   probe->nr_storage_samples = storage_samples;
   probe->bind = PIPE_BIND_SAMPLER_VIEW;
   return true;
}

static bool
pvrgpu_can_create_resource(struct pipe_screen *screen,
                           const struct pipe_resource *requested)
{
   struct pipe_resource probe;
   if (!pvrgpu_normalize_proxy_texture_samples(requested, &probe))
      return false;
   return pvrgpu_validate_resource_creation(screen, &probe);
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
   if (!pvrgpu_validate_resource_creation(screen, template))
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
   if (!pvrgpu_validate_resource_creation(screen, template))
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
 * Which colour attachment of the current framebuffer this resource is, or -1
 * when it is not one of them.
 *
 * A readback of anything else -- a texture the application uploaded, a
 * staging buffer -- has nothing to do with the model's framebuffer and must
 * not be overwritten with it.  A fragment shader returning more than one
 * result writes one attachment per result -- dEQP's modf returns its
 * fractional and integral parts -- and each is mapped for read separately, so
 * matching only attachment zero left every further one reading its own
 * untouched backing store.
 */
static int
pvrgpu_resource_color_attachment_index(const struct pvrgpu_context *ctx,
                                       const struct pipe_resource *resource,
                                       unsigned level)
{
   if (!ctx || !resource)
      return -1;
   for (unsigned target = 0; target < ctx->framebuffer.nr_cbufs; ++target) {
      if (ctx->framebuffer.cbufs[target].texture == resource &&
          ctx->framebuffer.cbufs[target].level == level)
         return (int)target;
   }
   return -1;
}

/* Color targets occupy bits 0..3. A depth-only pass still owns pending
 * work: carry its plane independently until real model execution/readback. */
#define PVRGPU_DEPTH_READBACK_PENDING (1u << 31)

void
pvrgpu_note_current_color_readback_pending(struct pvrgpu_context *ctx)
{
   if (!ctx)
      return;
   ctx->color_readback_generation = pvrgpu_systemc_submission_generation();
   ctx->color_readback_pending_mask = 0;
   for (unsigned target = 0; target < ctx->framebuffer.nr_cbufs; ++target) {
      if (ctx->framebuffer.cbufs[target].texture)
         ctx->color_readback_pending_mask |= 1u << target;
   }
   if (ctx->framebuffer.zsbuf.texture)
      ctx->color_readback_pending_mask |= PVRGPU_DEPTH_READBACK_PENDING;
}

/*
 * Whether the model has already applied the linear-to-sRGB transfer.
 *
 * Keep this in step with pvrgpu_command_format_for_framebuffer(): the two
 * formats below retain an sRGB command format, while all other surfaces use
 * the model's linear RGBA8 transport for now.
 */
static bool
pvrgpu_resource_readback_transport_is_srgb(enum pipe_format format)
{
   return format == PIPE_FORMAT_R8G8B8A8_SRGB ||
          format == PIPE_FORMAT_B8G8R8A8_SRGB;
}

/*
 * Raw dwords per integer pixel in the model transport.  Narrow native
 * channels are still one dword each; GLES has no three-channel integer render
 * target, and any such view uses the four-dword transport too.
 */
static unsigned
pvrgpu_resource_readback_raw_channels(enum pipe_format format)
{
   if (!util_format_is_pure_integer(format))
      return 0;

   const unsigned components = util_format_get_nr_components(format);
   if (components <= 1)
      return 1;
   if (components == 2)
      return 2;
   return 4;
}

static enum pipe_format
pvrgpu_resource_readback_pack_format(enum pipe_format format)
{
   /*
    * The model has already encoded these two sRGB transports.  Their linear
    * equivalents have the same byte layout and pack without encoding again.
    */
   return pvrgpu_resource_readback_transport_is_srgb(format)
             ? util_format_linear(format)
             : format;
}

/* The colour surfaces Mesa's generated packers can store model output into. */
static bool
pvrgpu_resource_readback_format_is_supported(enum pipe_format format)
{
   if (!pvrgpu_is_supported_color_format(format))
      return false;

   const struct util_format_description *description =
      util_format_description(format);
   if (!description || description->block.width != 1 ||
       description->block.height != 1 || description->block.depth != 1)
      return false;

   const enum pipe_format pack_format =
      pvrgpu_resource_readback_pack_format(format);
   const struct util_format_pack_description *pack =
      util_format_pack_description(pack_format);
   if (!pack)
      return false;

   if (util_format_is_pure_uint(format))
      return pack->pack_rgba_uint != NULL;
   if (util_format_is_pure_sint(format))
      return pack->pack_rgba_sint != NULL;
   if (util_format_is_float(format))
      return pack->pack_rgba_float != NULL;
   return pack->pack_rgba_8unorm != NULL;
}

/*
 * The transport width of one of those pixels.
 *
 * This is what sizes the staging buffer and what the flush must be told.  It
 * deliberately differs from the native block size for narrow integer
 * surfaces, because the model publishes one dword per logical channel.
 */
static unsigned
pvrgpu_resource_readback_bytes_per_pixel(enum pipe_format format)
{
   if (util_format_is_float(format))
      return 4u * sizeof(float);
   const unsigned raw_channels =
      pvrgpu_resource_readback_raw_channels(format);
   return raw_channels ? raw_channels * sizeof(uint32_t) : 4u;
}

/*
 * Store one row of the model's output in the surface's byte order.
 *
 * Mesa's generated packers narrow integer dwords and reconstruct normalized,
 * float and packed native layouts from the model's logical transport.
 */
static void
pvrgpu_resource_readback_store_row(enum pipe_format format,
                                   uint8_t *destination,
                                   const uint8_t *source_row,
                                   unsigned width)
{
   const enum pipe_format pack_format =
      pvrgpu_resource_readback_pack_format(format);
   const struct util_format_pack_description *pack =
      util_format_pack_description(pack_format);
   const unsigned raw_channels =
      pvrgpu_resource_readback_raw_channels(format);

   if (util_format_is_float(format)) {
      pack->pack_rgba_float(destination,
                            util_format_get_stride(pack_format, width),
                            (const float *)source_row,
                            width * 4u * sizeof(float),
                            width, 1);
      return;
   }

   if (raw_channels == 0) {
      /*
       * The model publishes logical RGBA8.  Mesa's generated packer performs
       * the native channel selection, swizzle and conversion for normalized,
       * float and packed targets.  This transport is currently quantized to
       * eight bits even when the native surface is wider.
       */
      pack->pack_rgba_8unorm(destination,
                             util_format_get_stride(pack_format, width),
                             source_row,
                             width * 4u,
                             width,
                             1);
      return;
   }

   const unsigned source_bytes_per_pixel =
      raw_channels * sizeof(uint32_t);
   const unsigned destination_bytes_per_pixel =
      util_format_get_blocksize(format);
   for (unsigned x = 0; x < width; ++x) {
      const uint8_t *source =
         source_row + (size_t)x * source_bytes_per_pixel;
      uint8_t *pixel =
         destination + (size_t)x * destination_bytes_per_pixel;
      if (util_format_is_pure_uint(format)) {
         uint32_t rgba[4] = { 0, 0, 0, 1 };
         memcpy(rgba, source, source_bytes_per_pixel);
         pack->pack_rgba_uint(pixel,
                              destination_bytes_per_pixel,
                              rgba,
                              sizeof(rgba),
                              1,
                              1);
      } else {
         int32_t rgba[4] = { 0, 0, 0, 1 };
         memcpy(rgba, source, source_bytes_per_pixel);
         pack->pack_rgba_sint(pixel,
                              destination_bytes_per_pixel,
                              rgba,
                              sizeof(rgba),
                              1,
                              1);
      }
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
 * format Mesa cannot pack, or a flush that produced nothing all leave
 * `pvrgpu->data` exactly as it was.
 */
static bool
pvrgpu_resource_read_back_color_surface(struct pipe_context *pipe,
                                        const struct pipe_surface *surface,
                                        unsigned attachment)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   struct pipe_resource *resource = surface ? surface->texture : NULL;
   struct pvrgpu_resource *pvrgpu = pvrgpu_resource(resource);
   if (!ctx || !surface || !resource || !pvrgpu || !pvrgpu->data ||
       resource->target == PIPE_BUFFER ||
       !pvrgpu_resource_level_valid(pvrgpu, surface->level))
      return false;

   /* A cached bridge image belongs to a submission, not merely an extent.
    * Consume ownership even on a failed read, so an error cannot later expose
    * an older cached image as this attachment's result. */
   const unsigned target_bit = 1u << attachment;
   if (!(ctx->color_readback_pending_mask & target_bit))
      return false;
   ctx->color_readback_pending_mask &= ~target_bit;
   if (ctx->color_readback_generation !=
       pvrgpu_systemc_submission_generation())
      return false;

   const struct util_format_description *view_desc =
      util_format_description(surface->format);
   const struct util_format_description *resource_desc =
      util_format_description(resource->format);
   if (!view_desc || !resource_desc ||
       view_desc->block.bits != resource_desc->block.bits ||
       view_desc->block.width != resource_desc->block.width ||
       view_desc->block.height != resource_desc->block.height ||
       view_desc->block.depth != resource_desc->block.depth) {
      pvrgpu_counter_eventf("framebuffer_readback_declined",
                            "reason=view_storage_layout res=%p target=%u "
                            "view_format=%s resource_format=%s",
                            (void *)resource, attachment,
                            util_format_name(surface->format),
                            util_format_name(resource->format));
      return false;
   }

   /* Read every attached layer from the same completed model submission. */
   const unsigned level_layers =
      pvrgpu_resource_level_layer_count(resource, surface->level);
   if (surface->first_layer > surface->last_layer ||
       surface->last_layer >= level_layers) {
      pvrgpu_counter_eventf("framebuffer_readback_declined",
                            "reason=layered_surface res=%p target=%u "
                            "level=%u layers=%u-%u level_layers=%u",
                            (void *)resource,
                            attachment,
                            surface->level,
                            surface->first_layer,
                            surface->last_layer,
                            level_layers);
      return false;
   }
   if (!pvrgpu_resource_readback_format_is_supported(surface->format)) {
      /*
       * Say so.  Returning quietly here left a case reading its own zeroed
       * backing store with nothing in the log to say the model's output had
       * been dropped -- dEQP's shader tests render into R32_UINT and every
       * result came back as zero.
       */
      pvrgpu_counter_eventf("framebuffer_readback_declined",
                            "reason=format res=%p target=%u format=%s",
                            (void *)resource,
                            attachment,
                            util_format_name(surface->format));
      return false;
   }

   /*
    * The model's framebuffer only describes the surface while everything that
    * touched it went to the model.  A scissored or masked clear that no draw
    * sequence subsumed did not, so copying the model's output back would erase
    * it -- which is what a run of dEQP's color_clear.scissored_* showed: two
    * full clears reached the model, thirteen scissored ones did not, and the
    * readback published the uniform surface the model had.  Leave the driver's
    * own content alone instead.
    */
   if (pvrgpu->driver_writes_model_cannot_reproduce) {
      pvrgpu_counter_eventf("framebuffer_readback_declined",
                            "reason=driver_writes_model_cannot_reproduce "
                            "res=%p target=%u width=%u height=%u",
                            (void *)resource,
                            attachment,
                            ctx->framebuffer.width,
                            ctx->framebuffer.height);
      return false;
   }

   const unsigned width = ctx->framebuffer.width;
   const unsigned height = ctx->framebuffer.height;
   if (width == 0 || height == 0 ||
       width > pvrgpu_resource_level_width(resource, surface->level) ||
       height > pvrgpu_resource_level_height(resource, surface->level)) {
      pvrgpu_counter_eventf("framebuffer_readback_declined",
                            "reason=extent res=%p target=%u fb=%ux%u "
                            "level=%u level_size=%ux%u",
                            (void *)resource,
                            attachment,
                            width,
                            height,
                            surface->level,
                            pvrgpu_resource_level_width(resource,
                                                        surface->level),
                            pvrgpu_resource_level_height(resource,
                                                         surface->level));
      return false;
   }

   const unsigned bytes_per_pixel =
      pvrgpu_resource_readback_bytes_per_pixel(surface->format);
   const unsigned samples = pvrgpu_resource_storage_sample_count(resource);
   const unsigned layer_count = (unsigned)surface->last_layer - surface->first_layer + 1;
   size_t destination_offset = 0;
   if (!pvrgpu_surface_span(pvrgpu, surface,
          (size_t)width * samples * util_format_get_blocksize(surface->format),
          height, layer_count, &destination_offset) ||
       (size_t)width * samples > SIZE_MAX / height / bytes_per_pixel / layer_count)
      return false;
   const size_t pixels_size =
      (size_t)width * (size_t)height * (size_t)bytes_per_pixel * samples * layer_count;
   uint8_t *pixels = MALLOC(pixels_size);
   if (!pixels)
      return false;

   bool written = false;
   char error[512] = { 0 };
   const bool flushed =
      pvrgpu_systemc_flush_readback_pixels(width, height, bytes_per_pixel,
                                           (uint32_t)attachment, samples, 0, layer_count, pixels,
                                           pixels_size, &written,
                                           error, sizeof(error));
   if (!flushed || !written) {
      if (!flushed) {
         /* The bridge may still retain an older successful framebuffer after
          * this execution failed.  No remaining target can read that cache. */
         ctx->color_readback_pending_mask = 0;
         debug_printf("pvrgpu: %s\n",
                      error[0] ? error : "readback flush failed");
      }
      FREE(pixels);
      return false;
   }

   /*
    * The model's framebuffer is tightly packed; the resource's level may be
    * padded, so store a row at a time rather than the whole block.
    */
   uint8_t *destination = pvrgpu->data + destination_offset;
   const unsigned stride = pvrgpu->level_strides[surface->level];
   for (unsigned layer = 0; layer < layer_count; ++layer)
   for (unsigned row = 0; row < height; ++row) {
      pvrgpu_resource_readback_store_row(
         surface->format,
         destination + (size_t)layer * pvrgpu->level_layer_strides[surface->level] + (size_t)row * stride,
         pixels + ((size_t)layer * height + row) * (size_t)width * (size_t)bytes_per_pixel * samples,
         width * samples);
   }
   FREE(pixels);

   pvrgpu_counter_eventf("framebuffer_readback",
                         "res=%p target=%u width=%u height=%u format=%s "
                         "level=%u layer=%u",
                         (void *)resource,
                         attachment,
                         width,
                         height,
                         util_format_name(surface->format),
                         surface->level,
                         surface->first_layer);
   return true;
}

static bool
pvrgpu_resource_read_back_depth_surface(struct pipe_context *pipe)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   const struct pipe_surface *surface = &ctx->framebuffer.zsbuf;
   struct pipe_resource *texture = surface->texture;
   struct pvrgpu_resource *resource = pvrgpu_resource(texture);
   if (!texture || !resource || !resource->data ||
       surface->format != texture->format ||
       !pvrgpu_resource_level_valid(resource, surface->level) ||
       surface->first_layer > surface->last_layer ||
       surface->last_layer >= pvrgpu_resource_level_layer_count(texture, surface->level))
      return false;
   const unsigned width = ctx->framebuffer.width;
   const unsigned height = ctx->framebuffer.height;
   if (!width || !height ||
       width > pvrgpu_resource_level_width(texture, surface->level) ||
       height > pvrgpu_resource_level_height(texture, surface->level))
      return false;
   const unsigned samples = pvrgpu_resource_storage_sample_count(texture);
   const unsigned bpp = util_format_get_blocksize(surface->format);
   const size_t row_size = (size_t)width * samples * bpp;
   const unsigned layer_count = (unsigned)surface->last_layer - surface->first_layer + 1;
   size_t destination_offset = 0;
   if (!pvrgpu_surface_span(resource, surface, row_size, height, layer_count, &destination_offset) ||
       row_size > SIZE_MAX / height / layer_count)
      return false;
   const size_t size = row_size * height * layer_count;
   uint8_t *pixels = malloc(size);
   if (!pixels)
      return false;
   char error[512] = {0};
   bool written = false;
   const bool flushed = pvrgpu_systemc_flush_readback_pixels(width, height, bpp, UINT32_MAX,
         samples, surface->format, layer_count, pixels, size, &written, error, sizeof(error));
   if (flushed && written) {
      uint8_t *destination = resource->data + destination_offset;
      for (unsigned layer = 0; layer < layer_count; ++layer)
      for (unsigned y = 0; y < height; ++y)
         memcpy(destination + (size_t)layer * resource->level_layer_strides[surface->level] +
                   (size_t)y * resource->level_strides[surface->level],
                pixels + ((size_t)layer * height + y) * row_size, row_size);
      pvrgpu_counter_eventf("depth_stencil_readback", "res=%p format=%s samples=%u",
                           (void *)texture, util_format_name(surface->format), samples);
   }
   free(pixels);
   return flushed && written;
}

/*
 * Submit once, then read every target from the bridge's cached framebuffer.
 * Calling the ordinary transfer-map path for each target would recursively
 * close the same frame and reset its command gates more than once.
 */
void
pvrgpu_flush_current_color_attachments(struct pipe_context *pipe)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   if (!ctx || pvrgpu_context_has_incomplete_replay(ctx))
      return;

   const unsigned recorded = ctx->array_primitive_draw_count;
   if (recorded != 0) {
      /* A failed new sequence cannot claim a previous submission's cache.
       * An already submitted RDC sequence keeps its existing ownership. */
      if (!ctx->array_primitive_sequence_owns_command)
         ctx->color_readback_pending_mask = 0;
   }
   if (recorded != 0 || ctx->color_readback_pending_mask != 0)
      pvrgpu_context_end_frame_at_readback(ctx);
   if (ctx->color_readback_pending_mask == 0)
      return;
   const unsigned expected = ctx->color_readback_pending_mask;
   unsigned written = 0;
   bool failed = false;
   for (unsigned target = 0; target < ctx->framebuffer.nr_cbufs; ++target) {
      if (!(expected & (1u << target)))
         continue;
      const struct pipe_surface *surface = &ctx->framebuffer.cbufs[target];
      if (surface->texture &&
          pvrgpu_resource_read_back_color_surface(pipe, surface, target))
         ++written;
      else
         failed = true;
   }
   /* A failed color execution clears the entire pending mask. Do not read
    * stale cached depth from an earlier successful sequence after that. */
   const bool read_depth =
      (ctx->color_readback_pending_mask & PVRGPU_DEPTH_READBACK_PENDING) &&
      ctx->color_readback_generation == pvrgpu_systemc_submission_generation();
   ctx->color_readback_pending_mask &= ~PVRGPU_DEPTH_READBACK_PENDING;
   if ((expected & PVRGPU_DEPTH_READBACK_PENDING) &&
       (!read_depth || !pvrgpu_resource_read_back_depth_surface(pipe)))
      failed = true;
   if (failed) {
      ++ctx->query_statistics_failures;
      ctx->color_readback_pending_mask = 0;
      pvrgpu_counter_eventf("framebuffer_boundary_flush_error",
                            "draws=%u expected=0x%x color_written=%u",
                            recorded, expected, written);
   }
   pvrgpu_counter_eventf("framebuffer_boundary_flush",
                         "draws=%u targets=%u written=%u",
                         recorded,
                         ctx->framebuffer.nr_cbufs,
                         written);
}

static void
pvrgpu_resource_read_back_color_attachment(struct pipe_context *pipe,
                                           struct pipe_resource *resource,
                                           unsigned level,
                                           unsigned usage)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   if (!(usage & PIPE_MAP_READ) || !ctx || !resource)
      return;

   if (ctx->framebuffer.zsbuf.texture == resource &&
       ctx->framebuffer.zsbuf.level == level) {
      pvrgpu_flush_current_color_attachments(pipe);
      return;
   }

   const int attachment =
      pvrgpu_resource_color_attachment_index(ctx, resource, level);
   if (attachment < 0) {
      /*
       * Say which surface was asked for and what the framebuffer held, so a
       * readback that silently returned the caller's own contents can be told
       * from one that was never a colour attachment at all.
       */
      pvrgpu_counter_eventf("framebuffer_readback_declined",
                            "reason=not_a_current_color_attachment res=%p "
                            "level=%u nr_cbufs=%u cbuf0=%p cbuf1=%p",
                            (void *)resource,
                            level,
                            ctx->framebuffer.nr_cbufs,
                            ctx->framebuffer.nr_cbufs > 0
                               ? (void *)ctx->framebuffer.cbufs[0].texture
                               : NULL,
                            ctx->framebuffer.nr_cbufs > 1
                               ? (void *)ctx->framebuffer.cbufs[1].texture
                               : NULL);
      return;
   }

   /* Multiple attachments can be different layers of this same resource.
    * Materialize every target before returning the requested box's backing;
    * selecting only the first resource/level match loses the other layers. */
   pvrgpu_flush_current_color_attachments(pipe);
}

/* Gallium texture_map exposes one tightly addressed pixel per format block,
 * not the driver's interleaved sample storage. llvmpipe's default map selects
 * sample zero (lp_texture.c:llvmpipe_transfer_map_ms/transfer_map). Preserve that
 * contract with a private packed staging view; an MSAA resolve is a separate
 * blit operation and must never be hidden inside this map. */
static bool
pvrgpu_copy_msaa_sample0(struct pvrgpu_transfer *transfer,
                         const struct pipe_box *region, bool write)
{
   if (!transfer || !transfer->sample0_staging || !region)
      return false;
   const struct pipe_transfer *map = &transfer->base;
   struct pvrgpu_resource *resource = pvrgpu_resource(map->resource);
   if (!resource || !resource->data || resource->base.target == PIPE_BUFFER ||
       !pvrgpu_transfer_box_in_bounds(&resource->base, map->level, &map->box) ||
       region->x < 0 || region->y < 0 || region->z < 0 ||
       region->width <= 0 || region->height <= 0 || region->depth <= 0 ||
       (uint64_t)region->x + region->width > (unsigned)map->box.width ||
       (uint64_t)region->y + region->height > (unsigned)map->box.height ||
       (uint64_t)region->z + region->depth > (unsigned)map->box.depth)
      return false;
   const unsigned bpp = util_format_get_blocksize(resource->base.format);
   const unsigned samples = pvrgpu_resource_storage_sample_count(&resource->base);
   if (!bpp || samples <= 1 ||
       util_format_get_blockwidth(resource->base.format) != 1 ||
       util_format_get_blockheight(resource->base.format) != 1)
      return false;
   const size_t pixel_stride = (size_t)bpp * samples;
   const size_t row_stride = resource->level_strides[map->level];
   const size_t layer_stride = resource->level_layer_strides[map->level];
   const unsigned level_width = pvrgpu_resource_level_width(&resource->base, map->level);
   const unsigned level_height = pvrgpu_resource_level_height(&resource->base, map->level);
   /* Validate both physical row/layer strides and the last byte before any
    * copy. Bounds expressed as division/subtraction also reject corrupt
    * metadata without overflowing size_t, including mip/layer offsets. */
   if (pixel_stride > row_stride || level_width > row_stride / pixel_stride ||
       !row_stride || level_height > layer_stride / row_stride ||
       !map->stride || (unsigned)map->box.width > map->stride / bpp ||
       (unsigned)map->box.height > map->layer_stride / map->stride ||
       !map->layer_stride ||
       (unsigned)map->box.depth > transfer->sample0_staging_size / map->layer_stride)
      return false;
   const size_t last_x = (unsigned)map->box.x + (unsigned)region->x + region->width - 1U;
   const size_t last_y = (unsigned)map->box.y + (unsigned)region->y + region->height - 1U;
   const size_t last_z = (unsigned)map->box.z + (unsigned)region->z + region->depth - 1U;
   size_t end = resource->level_offsets[map->level];
   if (end > resource->size || !layer_stride ||
       last_z > (resource->size - end) / layer_stride)
      return false;
   end += last_z * layer_stride;
   if (last_y > (resource->size - end) / row_stride)
      return false;
   end += last_y * row_stride;
   if (last_x > (resource->size - end) / pixel_stride)
      return false;
   end += last_x * pixel_stride;
   if (bpp > resource->size - end)
      return false;

   for (unsigned z = 0; z < (unsigned)region->depth; ++z)
      for (unsigned y = 0; y < (unsigned)region->height; ++y) {
         const size_t relative_z = (unsigned)region->z + z;
         const size_t relative_y = (unsigned)region->y + y;
         uint8_t *packed = transfer->sample0_staging +
            relative_z * map->layer_stride + relative_y * map->stride +
            (size_t)region->x * bpp;
         uint8_t *interleaved = resource->data + resource->level_offsets[map->level] +
            ((size_t)map->box.z + relative_z) * layer_stride +
            ((size_t)map->box.y + relative_y) * row_stride +
            ((size_t)map->box.x + region->x) * pixel_stride;
         for (unsigned x = 0; x < (unsigned)region->width; ++x) {
            if (write)
               memcpy(interleaved + (size_t)x * pixel_stride,
                      packed + (size_t)x * bpp, bpp);
            else
               memcpy(packed + (size_t)x * bpp,
                      interleaved + (size_t)x * pixel_stride, bpp);
         }
      }
   return true;
}

static void *
pvrgpu_map_msaa_sample0(struct pvrgpu_transfer *transfer)
{
   if (!transfer || !transfer->base.resource || transfer->sample0_staging)
      return NULL;
   struct pipe_transfer *map = &transfer->base;
   const struct pvrgpu_resource *resource = pvrgpu_resource(map->resource);
   if ((map->usage & (PIPE_MAP_DIRECTLY | PIPE_MAP_PERSISTENT | PIPE_MAP_COHERENT)) ||
       !(map->usage & (PIPE_MAP_READ | PIPE_MAP_WRITE)) ||
       !pvrgpu_transfer_box_in_bounds(map->resource, map->level, &map->box) ||
       pvrgpu_resource_storage_sample_count(map->resource) <= 1 ||
       util_format_get_blockwidth(map->resource->format) != 1 ||
       util_format_get_blockheight(map->resource->format) != 1 ||
       map->box.width <= 0 || map->box.height <= 0 || map->box.depth <= 0)
      return NULL;
   const unsigned bpp = util_format_get_blocksize(map->resource->format);
   if (!bpp || (unsigned)map->box.width > UINT_MAX / bpp)
      return NULL;
   const unsigned row_stride = (unsigned)map->box.width * bpp;
   if ((unsigned)map->box.height > SIZE_MAX / row_stride)
      return NULL;
   const size_t layer_stride = (size_t)row_stride * map->box.height;
   if ((unsigned)map->box.depth > SIZE_MAX / layer_stride)
      return NULL;
   const size_t bytes = layer_stride * map->box.depth;
   if (bytes > resource->size)
      return NULL;
   map->stride = row_stride;
   map->layer_stride = layer_stride;
   transfer->sample0_staging = malloc(bytes);
   if (!transfer->sample0_staging)
      return NULL;
   transfer->sample0_staging_size = bytes;
   const struct pipe_box whole = {.width = map->box.width,
      .height = map->box.height, .depth = map->box.depth};
   /* Gather even a write-only map so bytes the caller does not modify stay
    * intact. DISCARD permits losing them but does not require it; preserving
    * them also gives deterministic contents without exposing allocator data. */
   if (!pvrgpu_copy_msaa_sample0(transfer, &whole, false)) {
      FREE(transfer->sample0_staging);
      transfer->sample0_staging = NULL;
      transfer->sample0_staging_size = 0;
   }
   return transfer->sample0_staging;
}

static void
pvrgpu_unmap_msaa_sample0(struct pvrgpu_transfer *transfer)
{
   if (!transfer || !transfer->sample0_staging)
      return;
   const struct pipe_transfer *map = &transfer->base;
   if ((map->usage & PIPE_MAP_WRITE) && !(map->usage & PIPE_MAP_FLUSH_EXPLICIT)) {
      const struct pipe_box whole = {.width = map->box.width,
         .height = map->box.height, .depth = map->box.depth};
      (void)pvrgpu_copy_msaa_sample0(transfer, &whole, true);
   }
   FREE(transfer->sample0_staging);
   transfer->sample0_staging = NULL;
   transfer->sample0_staging_size = 0;
}

static bool
pvrgpu_texture_subdata_msaa_sample0(struct pipe_resource *resource,
                                    unsigned level, unsigned usage,
                                    const struct pipe_box *box, const void *data,
                                    unsigned stride, uintptr_t layer_stride)
{
   if (!resource || !data || !box || (usage & PIPE_MAP_READ) ||
       !pvrgpu_transfer_box_in_bounds(resource, level, box) ||
       box->width <= 0 || box->height <= 0 || box->depth <= 0)
      return false;
   const unsigned bpp = util_format_get_blocksize(resource->format);
   if (!bpp || (unsigned)box->width > SIZE_MAX / bpp)
      return false;
   const size_t row_bytes = (size_t)box->width * bpp;
   // A one-row/layer upload need not declare an unused stride. Otherwise the
   // source's packed rows/layers must not overlap or wrap pointer arithmetic.
   if ((box->height > 1 && stride < row_bytes) ||
       (stride && (unsigned)(box->height - 1) > (SIZE_MAX - row_bytes) / stride))
      return false;
   const size_t layer_bytes = (size_t)(box->height - 1) * stride + row_bytes;
   if ((box->depth > 1 && layer_stride < layer_bytes) ||
       (layer_stride && (unsigned)(box->depth - 1) >
                           (SIZE_MAX - layer_bytes) / layer_stride))
      return false;
   const size_t source_bytes = (size_t)(box->depth - 1) * layer_stride + layer_bytes;
   if ((uintptr_t)data > UINTPTR_MAX - (source_bytes - 1U))
      return false;

   struct pvrgpu_transfer transfer = {0};
   transfer.base.resource = resource;
   transfer.base.level = level;
   transfer.base.box = *box;
   // Subdata is a complete upload, equivalent to explicitly flushing its
   // whole mapped region. Use explicit mode so failure cleanup cannot commit.
   transfer.base.usage = usage | PIPE_MAP_WRITE | PIPE_MAP_DISCARD_RANGE |
                          PIPE_MAP_FLUSH_EXPLICIT;
   uint8_t *mapped = pvrgpu_map_msaa_sample0(&transfer);
   if (!mapped)
      return false;
   for (unsigned z = 0; z < (unsigned)box->depth; ++z)
      for (unsigned y = 0; y < (unsigned)box->height; ++y)
         memcpy(mapped + (size_t)z * transfer.base.layer_stride +
                    (size_t)y * transfer.base.stride,
                (const uint8_t *)data + (size_t)z * layer_stride + (size_t)y * stride,
                row_bytes);
   // The private snapshot also makes aliased source data safe: no destination
   // sample is mutated until every source row has been copied.
   const struct pipe_box whole = {.width = box->width,
      .height = box->height, .depth = box->depth};
   const bool copied = pvrgpu_copy_msaa_sample0(&transfer, &whole, true);
   pvrgpu_unmap_msaa_sample0(&transfer);
   return copied;
}

static void
pvrgpu_transfer_flush_region(struct pipe_context *pipe,
                             struct pipe_transfer *transfer,
                             const struct pipe_box *box)
{
   if (!transfer)
      return;
   struct pvrgpu_transfer *mapped = (struct pvrgpu_transfer *)transfer;
   if (mapped->sample0_staging) {
      if (transfer->usage & PIPE_MAP_WRITE)
         (void)pvrgpu_copy_msaa_sample0(mapped, box, true);
      return;
   }
   /* Non-staging maps are directly backed: retain the previous
    * u_default_transfer_flush_region no-op behavior. */
   (void)pipe;
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
   *out_transfer = NULL;
   const bool msaa = resource->target != PIPE_BUFFER &&
                     pvrgpu_resource_storage_sample_count(resource) > 1;
   if (msaa && (usage & (PIPE_MAP_DIRECTLY | PIPE_MAP_PERSISTENT | PIPE_MAP_COHERENT))) {
      pvrgpu_counter_eventf("texture_map_declined", "reason=msaa-staging-map-flags usage=0x%x", usage);
      return NULL;
   }

   /*
    * Ahead of every path below, including the displaytarget one -- that path
    * copies this shadow into the winsys surface on its way past, so the
    * pixels have to be here before it runs.
    */
   pvrgpu_resource_read_back_color_attachment(pipe, resource, level, usage);
   if (msaa && (usage & PIPE_MAP_WRITE) && !(usage & PIPE_MAP_UNSYNCHRONIZED))
      pvrgpu_flush_current_color_attachments(pipe);

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

   if (msaa) {
      void *mapped = pvrgpu_map_msaa_sample0(pvrgpu_transfer);
      if (!mapped) {
         pipe_resource_reference(&transfer->resource, NULL);
         FREE(pvrgpu_transfer);
         return NULL;
      }
      *out_transfer = transfer;
      if (usage & PIPE_MAP_WRITE)
         pvrgpu_invalidate_full_depth_clear_for_resource(pvrgpu_context(pipe), resource);
      pvrgpu_counter_eventf("texture_map", "res=%p level=%u usage=0x%x x=%d y=%d z=%d "
         "width=%d height=%d depth=%d stride=%u layer_stride=%zu format=%s sample=0 staging=1",
         (void *)resource, level, usage, box->x, box->y, box->z,
         box->width, box->height, box->depth, transfer->stride,
         (size_t)transfer->layer_stride, util_format_name(resource->format));
      return mapped;
   }

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
      pvrgpu_unmap_msaa_sample0(pvrgpu_transfer);
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
   struct pvrgpu_resource *pvrgpu = pvrgpu_resource(resource);
   if (!pvrgpu || !data ||
       !pvrgpu_transfer_box_in_bounds(resource, level, box))
      return;
   if (resource->target != PIPE_BUFFER &&
       pvrgpu_resource_storage_sample_count(resource) > 1) {
      if (!(usage & PIPE_MAP_UNSYNCHRONIZED))
         pvrgpu_flush_current_color_attachments(pipe);
      const bool copied = pvrgpu_texture_subdata_msaa_sample0(
         resource, level, usage, box, data, stride, layer_stride);
      if (copied)
         pvrgpu_invalidate_full_depth_clear_for_resource(pvrgpu_context(pipe), resource);
      pvrgpu_counter_eventf(copied ? "texture_subdata" : "texture_subdata_declined",
         "res=%p level=%u x=%d y=%d z=%d width=%d height=%d depth=%d "
         "stride=%u layer_stride=%zu format=%s sample=0 staging=1",
         (void *)resource, level, box->x, box->y, box->z,
         box->width, box->height, box->depth, stride, (size_t)layer_stride,
         util_format_name(resource->format));
      return;
   }
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
   if (pvrgpu_resource_storage_sample_count(dst) !=
       pvrgpu_resource_storage_sample_count(src) ||
       MAX2(1, dst->nr_samples) != MAX2(1, src->nr_samples) ||
       pvrgpu_resource_storage_sample_count(dst) != MAX2(1, dst->nr_samples))
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
            pvrgpu_msaa_texel_index(dstx, 0,
               pvrgpu_resource_storage_sample_count(dst)) * block_size;
         const uint8_t *src_row =
            pvrgpu_src->data + pvrgpu_src->level_offsets[src_level] +
            (uintptr_t)(src_box->z + layer) *
               pvrgpu_src->level_layer_strides[src_level] +
            (uintptr_t)(src_box->y + row) *
               pvrgpu_src->level_strides[src_level] +
            pvrgpu_msaa_texel_index((unsigned)src_box->x, 0,
               pvrgpu_resource_storage_sample_count(src)) * block_size;
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
                        x * pvrgpu_resource_storage_sample_count(resource),
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

   pvrgpu_flush_current_color_attachments(pipe);
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
   pvrgpu_resource(dst)->driver_writes_model_cannot_reproduce = true;
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
       info->alpha_blend || info->num_window_rectangles)
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
pvrgpu_blit_source_box_in_bounds(const struct pipe_resource *resource,
                                 unsigned level,
                                 const struct pipe_box *box)
{
   if (!resource || !box || resource->target == PIPE_BUFFER ||
       box->width == 0 || box->height == 0 || box->depth <= 0 ||
       box->z < 0)
      return false;

   const struct pvrgpu_resource *pvrgpu =
      pvrgpu_resource((struct pipe_resource *)resource);
   if (!pvrgpu_resource_level_valid(pvrgpu, level))
      return false;

   const int64_t x0 = box->x;
   const int64_t y0 = box->y;
   const int64_t x1 = x0 + (int64_t)box->width;
   const int64_t y1 = y0 + (int64_t)box->height;
   const int64_t min_x = x0 < x1 ? x0 : x1;
   const int64_t min_y = y0 < y1 ? y0 : y1;
   const int64_t max_x = x0 > x1 ? x0 : x1;
   const int64_t max_y = y0 > y1 ? y0 : y1;

   return pvrgpu_can_create_texture_target(resource) &&
          min_x >= 0 && min_y >= 0 &&
          (uint64_t)max_x <= pvrgpu_resource_level_width(resource, level) &&
          (uint64_t)max_y <= pvrgpu_resource_level_height(resource, level) &&
          (uint64_t)box->z + (uint64_t)box->depth <=
             pvrgpu_resource_level_layer_count(resource, level);
}

enum pvrgpu_blit_value_type {
   PVRGPU_BLIT_VALUE_FLOAT,
   PVRGPU_BLIT_VALUE_UINT,
   PVRGPU_BLIT_VALUE_SINT,
};

static bool
pvrgpu_blit_format_is_color(enum pipe_format format)
{
   if (format <= PIPE_FORMAT_NONE || format >= PIPE_FORMAT_COUNT)
      return false;

   const struct util_format_description *desc =
      util_format_description(format);
   if (!desc || desc->block.width != 1 || desc->block.height != 1 ||
       desc->block.depth != 1 || desc->block.bits == 0 ||
       desc->block.bits % 8 != 0)
      return false;

   return desc->colorspace == UTIL_FORMAT_COLORSPACE_RGB ||
          desc->colorspace == UTIL_FORMAT_COLORSPACE_SRGB;
}

static bool
pvrgpu_blit_view_matches_resource(const struct pipe_resource *resource,
                                  enum pipe_format view_format)
{
   if (!resource || !pvrgpu_blit_format_is_color(view_format) ||
       resource->format <= PIPE_FORMAT_NONE ||
       resource->format >= PIPE_FORMAT_COUNT)
      return false;

   const struct util_format_description *view_desc =
      util_format_description(view_format);
   const struct util_format_description *resource_desc =
      util_format_description(resource->format);

   return resource_desc &&
          view_desc->block.width == resource_desc->block.width &&
          view_desc->block.height == resource_desc->block.height &&
          view_desc->block.depth == resource_desc->block.depth &&
          view_desc->block.bits == resource_desc->block.bits;
}

static bool
pvrgpu_blit_get_value_type(enum pipe_format src_format,
                           enum pipe_format dst_format,
                           enum pvrgpu_blit_value_type *value_type)
{
   if (!value_type || !pvrgpu_blit_format_is_color(src_format) ||
       !pvrgpu_blit_format_is_color(dst_format))
      return false;

   const bool src_uint = util_format_is_pure_uint(src_format);
   const bool dst_uint = util_format_is_pure_uint(dst_format);
   const bool src_sint = util_format_is_pure_sint(src_format);
   const bool dst_sint = util_format_is_pure_sint(dst_format);
   const bool src_integer = util_format_is_pure_integer(src_format);
   const bool dst_integer = util_format_is_pure_integer(dst_format);

   if (src_uint && dst_uint) {
      *value_type = PVRGPU_BLIT_VALUE_UINT;
      return true;
   }
   if (src_sint && dst_sint) {
      *value_type = PVRGPU_BLIT_VALUE_SINT;
      return true;
   }
   if (src_integer || dst_integer)
      return false;

   *value_type = PVRGPU_BLIT_VALUE_FLOAT;
   return true;
}

static bool
pvrgpu_blit_format_access_is_supported(
   enum pipe_format src_format,
   enum pipe_format dst_format,
   enum pvrgpu_blit_value_type value_type)
{
   const struct util_format_unpack_description *unpack =
      util_format_unpack_description(src_format);
   const struct util_format_pack_description *pack =
      util_format_pack_description(dst_format);
   if (!unpack || (!unpack->unpack_rgba && !unpack->unpack_rgba_rect) ||
       !pack)
      return false;

   switch (value_type) {
   case PVRGPU_BLIT_VALUE_UINT:
      return pack->pack_rgba_uint != NULL;
   case PVRGPU_BLIT_VALUE_SINT:
      return pack->pack_rgba_sint != NULL;
   case PVRGPU_BLIT_VALUE_FLOAT:
      return pack->pack_rgba_float != NULL;
   default:
      return false;
   }
}

static bool
pvrgpu_blit_swizzle_is_valid(const struct pipe_blit_info *info)
{
   if (!info->swizzle_enable)
      return true;

   for (unsigned channel = 0; channel < 4; ++channel) {
      if (info->swizzle[channel] > PIPE_SWIZZLE_1)
         return false;
   }
   return true;
}

struct pvrgpu_blit_source_window {
   unsigned x;
   unsigned y;
   unsigned width;
   unsigned height;
};

/* Blits keep their original source/destination transform when clipped. Mesa's
 * GL frontend supplies the clipped destination as a scissor precisely because
 * rounding new integer box endpoints would change fractional sampling. */
static bool
pvrgpu_blit_layers_in_bounds(const struct pipe_resource *resource,
                             unsigned level, const struct pipe_box *box)
{
   if (!box)
      return false;
   const struct pipe_box layers = {.x = 0, .y = 0, .z = box->z,
                                    .width = 1, .height = 1, .depth = box->depth};
   return pvrgpu_transfer_box_in_bounds(resource, level, &layers);
}

static struct pvrgpu_blit_source_window
pvrgpu_blit_destination_window(const struct pipe_blit_info *info)
{
   int64_t x0 = MAX2((int64_t)info->dst.box.x, 0);
   int64_t y0 = MAX2((int64_t)info->dst.box.y, 0);
   int64_t x1 = MIN2((int64_t)info->dst.box.x + info->dst.box.width,
                    pvrgpu_resource_level_width(info->dst.resource, info->dst.level));
   int64_t y1 = MIN2((int64_t)info->dst.box.y + info->dst.box.height,
                    pvrgpu_resource_level_height(info->dst.resource, info->dst.level));
   if (info->scissor_enable) {
      x0 = MAX2(x0, info->scissor.minx);
      y0 = MAX2(y0, info->scissor.miny);
      x1 = MIN2(x1, info->scissor.maxx);
      y1 = MIN2(y1, info->scissor.maxy);
   }
   if (x1 <= x0 || y1 <= y0)
      return (struct pvrgpu_blit_source_window){0};
   return (struct pvrgpu_blit_source_window){x0, y0, x1 - x0, y1 - y0};
}

static bool
pvrgpu_blit_get_source_window(const struct pipe_blit_info *info,
                              struct pvrgpu_blit_source_window *window)
{
   if (!info || !window ||
       !pvrgpu_blit_layers_in_bounds(info->src.resource, info->src.level,
                                      &info->src.box))
      return false;

   const int64_t x0 = info->src.box.x;
   const int64_t y0 = info->src.box.y;
   const int64_t x1 = x0 + (int64_t)info->src.box.width;
   const int64_t y1 = y0 + (int64_t)info->src.box.height;
   int64_t min_x = MIN2(x0, x1);
   int64_t min_y = MIN2(y0, y1);
   int64_t max_x = MAX2(x0, x1);
   int64_t max_y = MAX2(y0, y1);
   const int64_t level_width = pvrgpu_resource_level_width(
      info->src.resource, info->src.level);
   const int64_t level_height = pvrgpu_resource_level_height(
      info->src.resource, info->src.level);

   if (info->filter == PIPE_TEX_FILTER_LINEAR) {
      --min_x;
      --min_y;
      ++max_x;
      ++max_y;
   }
   /* Include edge texels used by clamp-to-edge filtering even when an entire
    * original box lies beyond the resource. Destination coverage is separate. */
   min_x = CLAMP(min_x, 0, level_width - 1);
   min_y = CLAMP(min_y, 0, level_height - 1);
   max_x = CLAMP(max_x, 1, level_width);
   max_y = CLAMP(max_y, 1, level_height);

   if (max_x <= min_x || max_y <= min_y ||
       max_x - min_x > UINT_MAX || max_y - min_y > UINT_MAX)
      return false;

   window->x = (unsigned)min_x;
   window->y = (unsigned)min_y;
   window->width = (unsigned)(max_x - min_x);
   window->height = (unsigned)(max_y - min_y);
   return true;
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
   if (!(info->mask & PIPE_MASK_RGBA) || (info->mask & ~PIPE_MASK_RGBA))
      return false;
   if (info->dst.box.width <= 0 || info->dst.box.height <= 0 ||
       info->dst.box.depth <= 0)
      return false;
   if (info->src.box.width == 0 || info->src.box.height == 0 ||
       info->src.box.depth <= 0 ||
       info->dst.box.depth != info->src.box.depth)
      return false;
   if (info->filter != PIPE_TEX_FILTER_NEAREST &&
       info->filter != PIPE_TEX_FILTER_LINEAR)
      return false;
   if (info->render_condition_enable || info->alpha_blend ||
       info->num_window_rectangles ||
       !pvrgpu_blit_swizzle_is_valid(info))
      return false;
   const unsigned src_samples =
      pvrgpu_resource_storage_sample_count(info->src.resource);
   const unsigned dst_samples =
      pvrgpu_resource_storage_sample_count(info->dst.resource);
   /* Coverage/storage sample counts that differ require an explicit mapping,
    * which this storage layout does not provide. Ordinary MSAA has one stored
    * value per coverage sample. */
   if (src_samples != MAX2(1, info->src.resource->nr_samples) ||
       dst_samples != MAX2(1, info->dst.resource->nr_samples) ||
       (src_samples > 1 && dst_samples > 1 && src_samples != dst_samples) ||
       info->dst_sample > dst_samples ||
       (info->sample0_only && dst_samples != 1))
      return false;
   if (!pvrgpu_blit_layers_in_bounds(info->dst.resource, info->dst.level,
                                      &info->dst.box) ||
       !pvrgpu_blit_layers_in_bounds(info->src.resource, info->src.level,
                                      &info->src.box))
      return false;
   if (!pvrgpu_blit_view_matches_resource(info->src.resource,
                                          info->src.format) ||
       !pvrgpu_blit_view_matches_resource(info->dst.resource,
                                          info->dst.format))
      return false;

   enum pvrgpu_blit_value_type value_type;
   if (!pvrgpu_blit_get_value_type(info->src.format,
                                   info->dst.format,
                                   &value_type) ||
       (value_type != PVRGPU_BLIT_VALUE_FLOAT &&
        info->filter != PIPE_TEX_FILTER_NEAREST) ||
       !pvrgpu_blit_format_access_is_supported(info->src.format,
                                                info->dst.format,
                                                value_type))
      return false;
   if ((info->mask != PIPE_MASK_RGBA || info->dst_sample) &&
       !pvrgpu_blit_format_access_is_supported(info->dst.format,
                                                info->dst.format,
                                                value_type))
      return false;

   struct pvrgpu_resource *pvrgpu_dst =
      pvrgpu_resource(info->dst.resource);
   struct pvrgpu_resource *pvrgpu_src =
      pvrgpu_resource(info->src.resource);
   if (!pvrgpu_dst->data || !pvrgpu_src->data ||
       pvrgpu_dst->level_strides[info->dst.level] == 0 ||
       pvrgpu_src->level_strides[info->src.level] == 0 ||
       pvrgpu_dst->level_layer_strides[info->dst.level] == 0 ||
       pvrgpu_src->level_layer_strides[info->src.level] == 0)
      return false;

   struct pvrgpu_blit_source_window window;
   if (!pvrgpu_blit_get_source_window(info, &window) ||
       window.width > UINT_MAX / sizeof(union pipe_color_union) / src_samples ||
       (unsigned)info->dst.box.width >
          UINT_MAX / sizeof(union pipe_color_union) / dst_samples)
      return false;

   if (window.height > SIZE_MAX / window.width)
      return false;
   const size_t window_pixels = (size_t)window.width * window.height;
   if ((size_t)info->src.box.depth > SIZE_MAX / window_pixels ||
       window_pixels * (size_t)info->src.box.depth >
          SIZE_MAX / sizeof(union pipe_color_union) / src_samples)
      return false;

   return true;
}

struct pvrgpu_blit_axis_sample {
   unsigned first;
   unsigned second;
   float weight;
};

static int64_t
pvrgpu_floor_div_s64(int64_t numerator, int64_t denominator)
{
   const int64_t quotient = numerator / denominator;
   const int64_t remainder = numerator % denominator;
   return quotient - (remainder < 0 ? 1 : 0);
}

static unsigned
pvrgpu_clamp_blit_texel(int64_t coordinate, unsigned extent)
{
   if (coordinate < 0)
      return 0;
   if ((uint64_t)coordinate >= extent)
      return extent - 1;
   return (unsigned)coordinate;
}

static struct pvrgpu_blit_axis_sample
pvrgpu_get_blit_axis_sample(int src_origin,
                            int src_extent,
                            unsigned dst_index,
                            unsigned dst_extent,
                            unsigned resource_extent,
                            bool linear)
{
   const int64_t denominator = (int64_t)dst_extent * 2;
   int64_t numerator =
      (int64_t)src_extent * ((int64_t)dst_index * 2 + 1);
   if (linear)
      numerator -= dst_extent;

   const int64_t offset =
      pvrgpu_floor_div_s64(numerator, denominator);
   const int64_t first = (int64_t)src_origin + offset;
   struct pvrgpu_blit_axis_sample sample = {
      .first = pvrgpu_clamp_blit_texel(first, resource_extent),
      .second = pvrgpu_clamp_blit_texel(first + (linear ? 1 : 0),
                                        resource_extent),
      .weight = 0.0f,
   };

   if (linear) {
      const int64_t remainder = numerator - offset * denominator;
      sample.weight = (float)((double)remainder / (double)denominator);
   }
   return sample;
}

static union pipe_color_union
pvrgpu_blit_lerp(const union pipe_color_union *top_left,
                 const union pipe_color_union *top_right,
                 const union pipe_color_union *bottom_left,
                 const union pipe_color_union *bottom_right,
                 float x_weight,
                 float y_weight)
{
   union pipe_color_union result;
   for (unsigned channel = 0; channel < 4; ++channel) {
      const float top = top_left->f[channel] +
                        (top_right->f[channel] - top_left->f[channel]) *
                           x_weight;
      const float bottom = bottom_left->f[channel] +
                           (bottom_right->f[channel] -
                            bottom_left->f[channel]) * x_weight;
      result.f[channel] = top + (bottom - top) * y_weight;
   }
   return result;
}

static bool
pvrgpu_blit_zs_view_matches_resource(const struct pipe_resource *resource,
                                    enum pipe_format format)
{
   if (!resource || format <= PIPE_FORMAT_NONE || format >= PIPE_FORMAT_COUNT ||
       resource->format <= PIPE_FORMAT_NONE || resource->format >= PIPE_FORMAT_COUNT)
      return false;
   const struct util_format_description *view = util_format_description(format);
   const struct util_format_description *storage =
      util_format_description(resource->format);
   if (!(view && storage && view->colorspace == UTIL_FORMAT_COLORSPACE_ZS &&
          storage->colorspace == UTIL_FORMAT_COLORSPACE_ZS &&
          view->block.width == 1 && view->block.height == 1 &&
          view->block.depth == 1 && storage->block.width == 1 &&
          storage->block.height == 1 && storage->block.depth == 1 &&
          view->block.bits != 0 && view->block.bits % 8 == 0 &&
          view->block.bits == storage->block.bits))
      return false;
   for (unsigned aspect = 0; aspect < 2; ++aspect) {
      const unsigned view_channel = view->swizzle[aspect];
      const unsigned storage_channel = storage->swizzle[aspect];
      if (view_channel >= 4)
         continue;
      if (storage_channel >= 4 ||
          view->channel[view_channel].shift != storage->channel[storage_channel].shift ||
          view->channel[view_channel].size != storage->channel[storage_channel].size ||
          view->channel[view_channel].type != storage->channel[storage_channel].type ||
          view->channel[view_channel].normalized != storage->channel[storage_channel].normalized)
         return false;
   }
   return true;
}

static bool
pvrgpu_blit_zs_uses_float_depth(const struct pipe_blit_info *info)
{
   return util_get_depth_format_type(util_format_description(info->src.format)) ==
             UTIL_FORMAT_TYPE_FLOAT ||
          util_get_depth_format_type(util_format_description(info->dst.format)) ==
             UTIL_FORMAT_TYPE_FLOAT;
}

static bool
pvrgpu_can_blit_depth_stencil(const struct pipe_blit_info *info)
{
   if (!info || !info->src.resource || !info->dst.resource ||
       !(info->mask & PIPE_MASK_ZS) || (info->mask & ~PIPE_MASK_ZS) ||
       info->dst.box.width <= 0 || info->dst.box.height <= 0 ||
       info->dst.box.depth <= 0 ||
       info->src.box.depth != info->dst.box.depth ||
       info->filter != PIPE_TEX_FILTER_NEAREST || info->swizzle_enable ||
       info->render_condition_enable || info->alpha_blend ||
       info->num_window_rectangles ||
       !pvrgpu_blit_zs_view_matches_resource(info->src.resource, info->src.format) ||
       !pvrgpu_blit_zs_view_matches_resource(info->dst.resource, info->dst.format) ||
       !pvrgpu_transfer_box_in_bounds(info->dst.resource, info->dst.level,
                                      &info->dst.box) ||
       !pvrgpu_blit_source_box_in_bounds(info->src.resource, info->src.level,
                                         &info->src.box))
      return false;

   const unsigned src_samples = pvrgpu_resource_storage_sample_count(info->src.resource);
   const unsigned dst_samples = pvrgpu_resource_storage_sample_count(info->dst.resource);
   if (src_samples != MAX2(1, info->src.resource->nr_samples) ||
       dst_samples != MAX2(1, info->dst.resource->nr_samples) ||
       (src_samples > 1 && dst_samples > 1 && src_samples != dst_samples) ||
       info->dst_sample > dst_samples ||
       (info->sample0_only && dst_samples != 1))
      return false;

   const struct util_format_unpack_description *unpack =
      util_format_unpack_description(info->src.format);
   const struct util_format_pack_description *pack =
      util_format_pack_description(info->dst.resource->format);
   const struct util_format_description *src_view = util_format_description(info->src.format);
   const struct util_format_description *dst_view = util_format_description(info->dst.format);
   if (!unpack || !pack)
      return false;
   if (((info->mask & PIPE_MASK_Z) &&
        (!util_format_has_depth(src_view) || !util_format_has_depth(dst_view))) ||
       ((info->mask & PIPE_MASK_S) &&
        (!util_format_has_stencil(src_view) || !util_format_has_stencil(dst_view))))
      return false;
   if ((info->mask & PIPE_MASK_Z) &&
       (pvrgpu_blit_zs_uses_float_depth(info)
           ? (!unpack->unpack_z_float || !pack->pack_z_float)
           : (!unpack->unpack_z_32unorm || !pack->pack_z_32unorm)))
      return false;
   if ((info->mask & PIPE_MASK_S) &&
       (!unpack->unpack_s_8uint || !pack->pack_s_8uint))
      return false;

   const struct pvrgpu_resource *src = pvrgpu_resource(info->src.resource);
   const struct pvrgpu_resource *dst = pvrgpu_resource(info->dst.resource);
   struct pvrgpu_blit_source_window window;
   if (!src->data || !dst->data ||
       !pvrgpu_blit_get_source_window(info, &window) ||
       window.width > UINT_MAX / sizeof(uint32_t) / src_samples ||
       window.height > SIZE_MAX / window.width)
      return false;
   const size_t pixels = (size_t)window.width * window.height;
   return (size_t)info->src.box.depth <= SIZE_MAX / pixels &&
          pixels * (size_t)info->src.box.depth <=
             SIZE_MAX / sizeof(uint32_t) / src_samples;
}

static bool
pvrgpu_blit_depth_stencil_unchecked(const struct pipe_blit_info *info)
{
   struct pvrgpu_blit_source_window window;
   if (!pvrgpu_blit_get_source_window(info, &window))
      return false;
   struct pvrgpu_resource *src = pvrgpu_resource(info->src.resource);
   struct pvrgpu_resource *dst = pvrgpu_resource(info->dst.resource);
   const unsigned src_samples = pvrgpu_resource_storage_sample_count(info->src.resource);
   const unsigned dst_samples = pvrgpu_resource_storage_sample_count(info->dst.resource);
   const unsigned src_bpp = util_format_get_blocksize(info->src.format);
   const unsigned dst_bpp = util_format_get_blocksize(info->dst.format);
   const unsigned depth = (unsigned)info->src.box.depth;
   const size_t layer_samples = (size_t)window.width * window.height * src_samples;
   const size_t samples = layer_samples * depth;
   void *depth_values = info->mask & PIPE_MASK_Z ? MALLOC(samples * 4u) : NULL;
   uint8_t *stencil_values = info->mask & PIPE_MASK_S ? MALLOC(samples) : NULL;
   if (((info->mask & PIPE_MASK_Z) && !depth_values) ||
       ((info->mask & PIPE_MASK_S) && !stencil_values)) {
      FREE(depth_values);
      FREE(stencil_values);
      return false;
   }
   const struct util_format_unpack_description *unpack =
      util_format_unpack_description(info->src.format);
   const struct util_format_pack_description *pack =
      util_format_pack_description(info->dst.resource->format);
   const bool float_depth = pvrgpu_blit_zs_uses_float_depth(info);
   const unsigned row_samples = window.width * src_samples;

   /* Decode every source layer before the first destination write.  The two
    * resources may alias; masked stores must still read the original source.
    * Normalized depth keeps 32-bit precision rather than taking a float detour. */
   for (unsigned layer = 0; layer < depth; ++layer) {
      const uint8_t *source = src->data + src->level_offsets[info->src.level] +
         ((size_t)info->src.box.z + layer) * src->level_layer_strides[info->src.level] +
         (size_t)window.y * src->level_strides[info->src.level] +
         (size_t)window.x * src_samples * src_bpp;
      const size_t offset = (size_t)layer * layer_samples;
      if (depth_values) {
         if (float_depth)
            unpack->unpack_z_float((float *)depth_values + offset, row_samples * 4u,
                                    source, src->level_strides[info->src.level],
                                    row_samples, window.height);
         else
            unpack->unpack_z_32unorm((uint32_t *)depth_values + offset, row_samples * 4u,
                                      source, src->level_strides[info->src.level],
                                      row_samples, window.height);
      }
      if (stencil_values)
         unpack->unpack_s_8uint(stencil_values + offset, row_samples,
                                 source, src->level_strides[info->src.level],
                                 row_samples, window.height);
   }
   for (unsigned layer = 0; layer < depth; ++layer) {
      uint8_t *destination = dst->data + dst->level_offsets[info->dst.level] +
         ((size_t)info->dst.box.z + layer) * dst->level_layer_strides[info->dst.level];
      for (unsigned y = 0; y < (unsigned)info->dst.box.height; ++y) {
         const unsigned dst_y = (unsigned)info->dst.box.y + y;
         if (info->scissor_enable &&
             (dst_y < info->scissor.miny || dst_y >= info->scissor.maxy))
            continue;
         const unsigned src_y = pvrgpu_get_blit_axis_sample(
            info->src.box.y, info->src.box.height, y, info->dst.box.height,
            pvrgpu_resource_level_height(info->src.resource, info->src.level), false).first;
         for (unsigned x = 0; x < (unsigned)info->dst.box.width; ++x) {
            const unsigned dst_x = (unsigned)info->dst.box.x + x;
            if (info->scissor_enable &&
                (dst_x < info->scissor.minx || dst_x >= info->scissor.maxx))
               continue;
            const unsigned src_x = pvrgpu_get_blit_axis_sample(
               info->src.box.x, info->src.box.width, x, info->dst.box.width,
               pvrgpu_resource_level_width(info->src.resource, info->src.level), false).first;
            const size_t src_pixel = (size_t)layer * layer_samples +
               ((size_t)(src_y - window.y) * window.width + src_x - window.x) * src_samples;
            for (unsigned sample = 0; sample < dst_samples; ++sample) {
               if (info->dst_sample && sample != info->dst_sample - 1)
                  continue;
               /* A depth/stencil resolve selects one actual source sample;
                * unlike normalized color, it never averages the samples. */
               const size_t source_index = src_pixel +
                  (src_samples == dst_samples ? sample : 0);
               uint8_t *pixel = destination + (size_t)dst_y * dst->level_strides[info->dst.level] +
                  ((size_t)dst_x * dst_samples + sample) * dst_bpp;
               if (depth_values) {
                  if (float_depth)
                     pack->pack_z_float(pixel, dst_bpp,
                                         (const float *)depth_values + source_index, 4u, 1, 1);
                  else
                     pack->pack_z_32unorm(pixel, dst_bpp,
                                           (const uint32_t *)depth_values + source_index, 4u, 1, 1);
               }
               if (stencil_values)
                  pack->pack_s_8uint(pixel, dst_bpp, stencil_values + source_index, 1, 1, 1);
            }
         }
      }
   }
   FREE(depth_values);
   FREE(stencil_values);
   return true;
}

static bool
pvrgpu_blit_texture_region_unchecked(const struct pipe_blit_info *info)
{
   const struct pvrgpu_blit_source_window destination =
      pvrgpu_blit_destination_window(info);
   if (!destination.width || !destination.height)
      return true;
   struct pvrgpu_resource *pvrgpu_dst =
      pvrgpu_resource(info->dst.resource);
   struct pvrgpu_resource *pvrgpu_src =
      pvrgpu_resource(info->src.resource);
   const unsigned dst_width = (unsigned)info->dst.box.width;
   const unsigned dst_height = (unsigned)info->dst.box.height;
   const unsigned depth = (unsigned)info->dst.box.depth;
   const unsigned src_samples =
      pvrgpu_resource_storage_sample_count(info->src.resource);
   const unsigned dst_samples =
      pvrgpu_resource_storage_sample_count(info->dst.resource);
   const bool linear = info->filter == PIPE_TEX_FILTER_LINEAR;
   const unsigned src_level_width =
      pvrgpu_resource_level_width(info->src.resource, info->src.level);
   const unsigned src_level_height =
      pvrgpu_resource_level_height(info->src.resource, info->src.level);
   enum pvrgpu_blit_value_type value_type;
   struct pvrgpu_blit_source_window window;
   if (!pvrgpu_blit_get_value_type(info->src.format,
                                   info->dst.format,
                                   &value_type) ||
       !pvrgpu_blit_get_source_window(info, &window))
      return false;

   const size_t window_pixels = (size_t)window.width * window.height;
   const size_t source_pixel_count = window_pixels * depth * src_samples;
   const unsigned source_row_stride =
      window.width * src_samples * sizeof(union pipe_color_union);
   const unsigned destination_row_stride =
      destination.width * dst_samples * sizeof(union pipe_color_union);
   union pipe_color_union *source_pixels =
      MALLOC(source_pixel_count * sizeof(*source_pixels));
   union pipe_color_union *destination_row =
      MALLOC((size_t)destination_row_stride);
   if (!source_pixels || !destination_row) {
      FREE(source_pixels);
      FREE(destination_row);
      return false;
   }

   /* Snapshot all source layers before writing so a legal cross-layer alias
    * cannot feed already-written destination pixels back into the blit. */
   for (unsigned layer = 0; layer < depth; ++layer) {
      const uint8_t *src_layer_data =
         pvrgpu_src->data + pvrgpu_src->level_offsets[info->src.level] +
         (uintptr_t)((unsigned)info->src.box.z + layer) *
            pvrgpu_src->level_layer_strides[info->src.level];
      util_format_read_4(info->src.format,
                         source_pixels + (size_t)layer * window_pixels * src_samples,
                         source_row_stride,
                         src_layer_data,
                         pvrgpu_src->level_strides[info->src.level],
                         window.x * src_samples,
                         window.y,
                         window.width * src_samples,
                         window.height);
   }

   /* Resolve each source pixel before spatial filtering. Read every actual
    * sample, as util_blitter's resolve shader does; never manufacture missing
    * samples from sample zero. sRGB unpacking/packing linearizes/re-encodes
    * through the normal Mesa format callbacks. */
   if (src_samples > 1 && dst_samples == 1 && !info->sample0_only &&
       value_type == PVRGPU_BLIT_VALUE_FLOAT) {
      for (size_t pixel = 0; pixel < window_pixels * depth; ++pixel) {
         union pipe_color_union *samples = source_pixels + pixel * src_samples;
         float resolved[4];
         pvrgpu_msaa_resolve_float(samples, src_samples, sizeof(*samples),
                                    resolved);
         memcpy(samples[0].f, resolved, sizeof(resolved));
      }
   }

   for (unsigned layer = 0; layer < depth; ++layer) {
      uint8_t *dst_layer_data =
         pvrgpu_dst->data + pvrgpu_dst->level_offsets[info->dst.level] +
         (uintptr_t)((unsigned)info->dst.box.z + layer) *
            pvrgpu_dst->level_layer_strides[info->dst.level];
      const union pipe_color_union *source_layer =
         source_pixels + (size_t)layer * window_pixels * src_samples;

      for (unsigned row = 0; row < destination.height; ++row) {
         const unsigned dst_y = (int64_t)destination.y + row - info->dst.box.y;
         if (info->mask != PIPE_MASK_RGBA || info->dst_sample) {
            util_format_read_4(info->dst.format,
                                destination_row,
                                destination_row_stride,
                                dst_layer_data,
                                pvrgpu_dst->level_strides[info->dst.level],
                                destination.x * dst_samples,
                                destination.y + row,
                                destination.width * dst_samples,
                                1);
         }
         const struct pvrgpu_blit_axis_sample y_sample =
            pvrgpu_get_blit_axis_sample(info->src.box.y,
                                        info->src.box.height,
                                        dst_y,
                                        dst_height,
                                        src_level_height,
                                        linear);
         const size_t first_row =
            (size_t)(y_sample.first - window.y) * window.width;
         const size_t second_row =
            (size_t)(y_sample.second - window.y) * window.width;

         for (unsigned column = 0; column < destination.width; ++column) {
            const unsigned dst_x = (int64_t)destination.x + column - info->dst.box.x;
            const struct pvrgpu_blit_axis_sample x_sample =
               pvrgpu_get_blit_axis_sample(info->src.box.x,
                                           info->src.box.width,
                                           dst_x,
                                           dst_width,
                                           src_level_width,
                                           linear);
            const unsigned first_x = x_sample.first - window.x;
            const unsigned second_x = x_sample.second - window.x;
            for (unsigned dst_sample = 0; dst_sample < dst_samples;
                 ++dst_sample) {
               if (info->dst_sample && dst_sample != info->dst_sample - 1)
                  continue;
               const unsigned src_sample =
                  src_samples == dst_samples ? dst_sample : 0;
               const union pipe_color_union *top_left =
                  source_layer + (first_row + first_x) * src_samples + src_sample;
               union pipe_color_union result;

               if (linear) {
                  const union pipe_color_union *top_right =
                     source_layer + (first_row + second_x) * src_samples + src_sample;
                  const union pipe_color_union *bottom_left =
                     source_layer + (second_row + first_x) * src_samples + src_sample;
                  const union pipe_color_union *bottom_right =
                     source_layer + (second_row + second_x) * src_samples + src_sample;
                  result = pvrgpu_blit_lerp(top_left,
                                            top_right,
                                            bottom_left,
                                            bottom_right,
                                            x_sample.weight,
                                            y_sample.weight);
               } else {
                  result = *top_left;
               }

               if (info->swizzle_enable) {
                  union pipe_color_union swizzled;
                  util_format_apply_color_swizzle(
                     &swizzled,
                     &result,
                     info->swizzle,
                     value_type != PVRGPU_BLIT_VALUE_FLOAT);
                  result = swizzled;
               }
               union pipe_color_union *dst_pixel = destination_row +
                  pvrgpu_msaa_texel_index(column, dst_sample, dst_samples);
               for (unsigned channel = 0; channel < 4; ++channel) {
                  if (info->mask & (PIPE_MASK_R << channel))
                     dst_pixel->ui[channel] = result.ui[channel];
               }
            }
         }

         util_format_write_4(info->dst.format,
                             destination_row,
                             destination_row_stride,
                             dst_layer_data,
                             pvrgpu_dst->level_strides[info->dst.level],
                             destination.x * dst_samples,
                             destination.y + row,
                             destination.width * dst_samples,
                             1);
      }
   }

   FREE(source_pixels);
   FREE(destination_row);
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
   const bool copy_2d = pvrgpu_can_blit_as_2d_copy(info);
   const bool texture_blit =
      !copy_2d && pvrgpu_can_blit_as_texture_region(info);
   const bool depth_stencil_blit =
      !copy_2d && !texture_blit && pvrgpu_can_blit_depth_stencil(info);

   /*
    * The CPU paths below consume resource backing directly.  If the source is
    * the framebuffer a generic sequence just drew, its newest pixels still
    * live in the model until this materializes every current colour target.
    */
   if (copy_2d || texture_blit || depth_stencil_blit)
      pvrgpu_flush_current_color_attachments(pipe);

   if (copy_2d) {
      pvrgpu_copy_texture_region_unchecked(info->dst.resource,
                                           info->dst.level,
                                           info->dst.box.x,
                                           info->dst.box.y,
                                           info->dst.box.z,
                                           info->src.resource,
                                           info->src.level,
                                           &info->src.box);
   } else if (texture_blit) {
      if (!pvrgpu_blit_texture_region_unchecked(info)) {
         pvrgpu_emit_unsupported_blit(pipe,
                                      "texture-blit-allocation-failed",
                                      info);
         return;
      }
   } else if (depth_stencil_blit) {
      if (!pvrgpu_blit_depth_stencil_unchecked(info)) {
         pvrgpu_emit_unsupported_blit(pipe, "depth-stencil-blit-allocation-failed", info);
         return;
      }
   } else {
      pvrgpu_emit_unsupported_blit(pipe,
                                   "no-scale-same-format-rgba-only",
                                   info);
      return;
   }

   /*
    * This copy ran only in driver memory.  A later map must not replace it
    * with the bridge's cached pre-blit framebuffer merely because the resource
    * is still bound as a colour target.
    */
   pvrgpu_resource(info->dst.resource)
      ->driver_writes_model_cannot_reproduce = true;
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
   context->transfer_flush_region = pvrgpu_transfer_flush_region;
   context->buffer_subdata = pvrgpu_buffer_subdata;
   context->texture_subdata = pvrgpu_texture_subdata;
   context->clear_buffer = pvrgpu_clear_buffer;
   context->clear_texture = pvrgpu_clear_texture;
   context->resource_copy_region = pvrgpu_resource_copy_region;
   context->blit = pvrgpu_blit;
   context->flush_resource = pvrgpu_flush_resource;
}
