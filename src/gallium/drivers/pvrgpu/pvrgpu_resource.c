/* SPDX-License-Identifier: MIT */

#include "pvrgpu_resource.h"
#include "pvrgpu_cmd.h"
#include "pvrgpu_context.h"
#include "pvrgpu_counter.h"

#include "pipe/p_defines.h"
#include "util/format/u_format.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"
#include "util/u_transfer.h"

#include <ctype.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

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
pvrgpu_deqp_fbo_default_framebuffer_direct_color_counter_case(void)
{
   const char *case_name = pvrgpu_rdc_case_name();
   return pvrgpu_string_has_prefix(
             case_name,
             "dEQP-GLES3.functional.fbo.blit.default_framebuffer.") &&
          (pvrgpu_string_contains(case_name, ".rgb8_") ||
           pvrgpu_string_contains(case_name, ".rgba8_")) &&
          pvrgpu_string_contains(case_name, "_blit_to_default");
}

static const char *
pvrgpu_command_output_path(void)
{
   const char *path = getenv("PVRGPU_DRIVER_COMMAND_OUT");
   if (path && path[0] != '\0')
      return path;
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
pvrgpu_is_supported_color_resource_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8X8_UNORM:
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_B8G8R8X8_UNORM:
   case PIPE_FORMAT_R10G10B10A2_UNORM:
   case PIPE_FORMAT_B10G10R10A2_UNORM:
      return true;
   default:
      return false;
   }
}

static bool
pvrgpu_is_supported_depth_stencil_resource_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_Z32_FLOAT:
   case PIPE_FORMAT_Z32_UNORM:
   case PIPE_FORMAT_Z32_FLOAT_S8X24_UINT:
      return true;
   default:
      return false;
   }
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
   case 4:
   case 8:
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

   if (pvrgpu_resource_is_multisampled(template) &&
       template->target != PIPE_TEXTURE_2D)
      return false;

   switch (template->target) {
   case PIPE_TEXTURE_2D:
      return template->depth0 <= 1 && template->array_size <= 1;
   case PIPE_TEXTURE_2D_ARRAY:
      return template->depth0 <= 1 && template->array_size > 0;
   case PIPE_TEXTURE_3D:
      return template->depth0 > 0 && template->array_size <= 1;
   case PIPE_TEXTURE_CUBE:
      return template->depth0 <= 1;
   default:
      return false;
   }
}

static bool
pvrgpu_can_create_texture(const struct pipe_resource *template)
{
   return template &&
          pvrgpu_can_create_texture_target(template) &&
          (pvrgpu_is_supported_color_resource_format(template->format) ||
           (template->target == PIPE_TEXTURE_2D &&
            pvrgpu_is_supported_depth_stencil_resource_format(template->format)));
}

static bool
pvrgpu_can_create_resource(struct pipe_screen *screen,
                           const struct pipe_resource *template)
{
   (void)screen;
   return pvrgpu_can_create_buffer(template) ||
          pvrgpu_can_create_texture(template);
}

static unsigned
pvrgpu_resource_layer_count(const struct pipe_resource *resource)
{
   if (resource->target == PIPE_TEXTURE_3D)
      return resource->depth0 > 0 ? resource->depth0 : 1;

   if (resource->target == PIPE_TEXTURE_CUBE && resource->array_size < 6)
      return 6;

   return resource->array_size > 0 ? resource->array_size : 1;
}

static unsigned
pvrgpu_resource_storage_sample_count(const struct pipe_resource *resource)
{
   unsigned sample_count = resource->nr_storage_samples;
   if (sample_count == 0)
      sample_count = resource->nr_samples;
   return sample_count > 1 ? sample_count : 1;
}

static bool
pvrgpu_init_resource_storage(struct pvrgpu_resource *resource)
{
   if (resource->base.target == PIPE_BUFFER) {
      resource->stride = resource->base.width0;
      resource->layer_stride = resource->base.width0;
      resource->size = resource->base.width0;
   } else {
      const unsigned block_size =
         util_format_get_blocksize(resource->base.format);
      if (block_size == 0)
         return false;

      resource->stride =
         resource->base.width0 * block_size *
         pvrgpu_resource_storage_sample_count(&resource->base);
      resource->layer_stride =
         (uintptr_t)resource->stride * resource->base.height0;
      resource->size =
         resource->layer_stride * pvrgpu_resource_layer_count(&resource->base);
   }

   resource->data = CALLOC(1, resource->size);
   return resource->data != NULL;
}

static struct pipe_resource *
pvrgpu_resource_create(struct pipe_screen *screen,
                       const struct pipe_resource *template)
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
   pvrgpu_counter_eventf("resource_create",
                         "target=%u width=%u height=%u depth=%u array=%u "
                         "format=%s bind=0x%x usage=%u flags=0x%x size=%zu",
                         resource->base.target,
                         resource->base.width0,
                         resource->base.height0,
                         resource->base.depth0,
                         resource->base.array_size,
                         util_format_name(resource->base.format),
                         resource->base.bind,
                         resource->base.usage,
                         resource->base.flags,
                         resource->size);
   return &resource->base;
}

static struct pipe_resource *
pvrgpu_resource_create_front(struct pipe_screen *screen,
                             const struct pipe_resource *template,
                             const void *map_front_private)
{
   (void)map_front_private;
   return pvrgpu_resource_create(screen, template);
}

static void
pvrgpu_resource_destroy(struct pipe_screen *screen,
                        struct pipe_resource *resource)
{
   (void)screen;
   pvrgpu_counter_eventf("resource_destroy",
                         "target=%u width=%u height=%u depth=%u array=%u "
                         "format=%s bind=0x%x usage=%u flags=0x%x size=%zu",
                         resource->target,
                         resource->width0,
                         resource->height0,
                         resource->depth0,
                         resource->array_size,
                         util_format_name(resource->format),
                         resource->bind,
                         resource->usage,
                         resource->flags,
                         pvrgpu_resource(resource)->size);
   FREE(pvrgpu_resource(resource)->data);
   FREE(pvrgpu_resource(resource));
}

static bool
pvrgpu_transfer_box_in_bounds(const struct pipe_resource *resource,
                              unsigned level,
                              const struct pipe_box *box)
{
   if (!resource || !box || level != 0)
      return false;

   if (resource->target == PIPE_BUFFER) {
      return box->x >= 0 &&
             box->y == 0 &&
             box->z == 0 &&
             box->height == 1 &&
             box->depth == 1 &&
             box->width >= 0 &&
             (uint64_t)box->x + (uint64_t)box->width <= resource->width0;
   }

   return pvrgpu_can_create_texture_target(resource) &&
          box->x >= 0 &&
          box->y >= 0 &&
          box->z >= 0 &&
          box->width >= 0 &&
          box->height >= 0 &&
          box->depth >= 0 &&
          (uint64_t)box->x + (uint64_t)box->width <= resource->width0 &&
          (uint64_t)box->y + (uint64_t)box->height <= resource->height0 &&
          (uint64_t)box->z + (uint64_t)box->depth <=
             pvrgpu_resource_layer_count(resource);
}

static void *
pvrgpu_transfer_map(struct pipe_context *pipe,
                    struct pipe_resource *resource,
                    unsigned level,
                    unsigned usage,
                    const struct pipe_box *box,
                    struct pipe_transfer **out_transfer)
{
   (void)pipe;
   struct pvrgpu_resource *pvrgpu = pvrgpu_resource(resource);
   if (!out_transfer || !pvrgpu || !pvrgpu->data ||
       !pvrgpu_transfer_box_in_bounds(resource, level, box))
      return NULL;

   struct pipe_transfer *transfer = CALLOC_STRUCT(pipe_transfer);
   if (!transfer)
      return NULL;

   pipe_resource_reference(&transfer->resource, resource);
   transfer->level = level;
   transfer->usage = usage;
   transfer->box = *box;
   transfer->stride = pvrgpu->stride;
   transfer->layer_stride = pvrgpu->layer_stride;
   *out_transfer = transfer;

   pvrgpu_counter_eventf(resource->target == PIPE_BUFFER ?
                         "buffer_map" : "texture_map",
                         "usage=0x%x x=%d y=%d z=%d width=%d height=%d depth=%d",
                         usage,
                         box->x,
                         box->y,
                         box->z,
                         box->width,
                         box->height,
                         box->depth);

   if (resource->target == PIPE_BUFFER)
      return pvrgpu->data + box->x;

   const unsigned block_size = util_format_get_blocksize(resource->format);
   return pvrgpu->data + (uintptr_t)box->z * pvrgpu->layer_stride +
          (uintptr_t)box->y * pvrgpu->stride +
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
      pvrgpu_counter_event(transfer->resource->target == PIPE_BUFFER ?
                           "buffer_unmap" : "texture_unmap",
                           "");
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
   (void)pipe;
   (void)usage;
   struct pvrgpu_resource *pvrgpu = pvrgpu_resource(resource);
   if (!pvrgpu || !data || resource->target != PIPE_BUFFER ||
       (uint64_t)offset + (uint64_t)size > pvrgpu->size)
      return;
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
   (void)pipe;
   (void)usage;
   struct pvrgpu_resource *pvrgpu = pvrgpu_resource(resource);
   if (!pvrgpu || !data ||
       !pvrgpu_transfer_box_in_bounds(resource, level, box))
      return;

   const unsigned block_size = util_format_get_blocksize(resource->format);
   const unsigned row_bytes = (unsigned)box->width * block_size;
   uint8_t *dst = pvrgpu->data + (uintptr_t)box->z * pvrgpu->layer_stride +
                  (uintptr_t)box->y * pvrgpu->stride +
                  (uintptr_t)box->x * block_size;
   const uint8_t *src = (const uint8_t *)data;
   for (int layer = 0; layer < box->depth; ++layer) {
      for (int row = 0; row < box->height; ++row) {
         memcpy(dst + (uintptr_t)layer * pvrgpu->layer_stride +
                   (uintptr_t)row * pvrgpu->stride,
                src + (uintptr_t)layer * layer_stride +
                   (uintptr_t)row * stride,
                row_bytes);
      }
   }
   pvrgpu_counter_eventf("texture_subdata",
                         "x=%d y=%d z=%d width=%d height=%d depth=%d "
                         "stride=%u layer_stride=%zu",
                         box->x,
                         box->y,
                         box->z,
                         box->width,
                         box->height,
                         box->depth,
                         stride,
                         (size_t)layer_stride);
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

static bool
pvrgpu_can_copy_2d_region(struct pipe_resource *dst,
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
   if (dst_level != 0 || src_level != 0 || dstz != 0)
      return false;
   if (dst->target != PIPE_TEXTURE_2D || src->target != PIPE_TEXTURE_2D)
      return false;
   if (dst->format != src->format)
      return false;
   if (dst->nr_samples > 1 || src->nr_samples > 1 ||
       dst->nr_storage_samples > 1 || src->nr_storage_samples > 1)
      return false;
   if (src_box->width <= 0 || src_box->height <= 0 ||
       src_box->depth != 1)
      return false;
   if (!pvrgpu_transfer_box_in_bounds(src, src_level, src_box))
      return false;
   if ((uint64_t)dstx + (uint64_t)src_box->width > dst->width0 ||
       (uint64_t)dsty + (uint64_t)src_box->height > dst->height0)
      return false;
   if (!pvrgpu_resource(dst)->data || !pvrgpu_resource(src)->data)
      return false;
   return util_format_get_blocksize(dst->format) != 0;
}

static void
pvrgpu_copy_2d_region_unchecked(struct pipe_resource *dst,
                                unsigned dstx,
                                unsigned dsty,
                                struct pipe_resource *src,
                                const struct pipe_box *src_box)
{
   struct pvrgpu_resource *pvrgpu_dst = pvrgpu_resource(dst);
   struct pvrgpu_resource *pvrgpu_src = pvrgpu_resource(src);
   const unsigned block_size = util_format_get_blocksize(dst->format);
   const unsigned row_bytes = (unsigned)src_box->width * block_size;

   for (int row = 0; row < src_box->height; ++row) {
      uint8_t *dst_row =
         pvrgpu_dst->data + (uintptr_t)(dsty + row) * pvrgpu_dst->stride +
         (uintptr_t)dstx * block_size;
      const uint8_t *src_row =
         pvrgpu_src->data +
         (uintptr_t)(src_box->y + row) * pvrgpu_src->stride +
         (uintptr_t)src_box->x * block_size;
      memmove(dst_row, src_row, row_bytes);
   }
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

   if (!pvrgpu_can_copy_2d_region(dst, dst_level, dstx, dsty, dstz,
                                  src, src_level, src_box)) {
      pvrgpu_emit_unsupported_resource_op(pipe,
                                          "unsupported_resource_copy_region",
                                          "buffer-or-2d-level0-same-format-only");
      return;
   }

   pvrgpu_copy_2d_region_unchecked(dst, dstx, dsty, src, src_box);
   pvrgpu_counter_eventf("resource_copy_region",
                         "dst=%ux%u dst_xyz=%u,%u,%u dst_format=%s "
                         "src=%ux%u src_box=%d,%d,%d,%d,%d,%d src_format=%s",
                         dst->width0,
                         dst->height0,
                         dstx,
                         dsty,
                         dstz,
                         util_format_name(dst->format),
                         src->width0,
                         src->height0,
                         src_box->x,
                         src_box->y,
                         src_box->z,
                         src_box->width,
                         src_box->height,
                         src_box->depth,
                         util_format_name(src->format));
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
       info->dst.box.depth != 1)
      return false;
   if (info->src.box.width <= 0 || info->src.box.height <= 0 ||
       info->src.box.depth != 1)
      return false;
   if (info->dst.box.width != info->src.box.width ||
       info->dst.box.height != info->src.box.height)
      return false;
   if (info->dst.box.x < 0 || info->dst.box.y < 0 ||
       info->dst.box.z != 0)
      return false;
   if (info->filter != PIPE_TEX_FILTER_NEAREST &&
       info->filter != PIPE_TEX_FILTER_LINEAR)
      return false;
   if (info->dst_sample || info->sample0_only || info->scissor_enable ||
       info->swizzle_enable || info->render_condition_enable ||
       info->alpha_blend)
      return false;
   return pvrgpu_can_copy_2d_region(info->dst.resource,
                                    info->dst.level,
                                    info->dst.box.x,
                                    info->dst.box.y,
                                    info->dst.box.z,
                                    info->src.resource,
                                    info->src.level,
                                    &info->src.box);
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
   case PIPE_FORMAT_R10G10B10A2_UNORM:
      return PVRGPU_DRIVER_COMMAND_FORMAT_R10G10B10A2;
   case PIPE_FORMAT_B10G10R10A2_UNORM:
      return PVRGPU_DRIVER_COMMAND_FORMAT_B10G10R10A2;
   default:
      return PVRGPU_DRIVER_COMMAND_FORMAT_RGBA8;
   }
}

static unsigned
pvrgpu_max_unsigned(unsigned a, unsigned b)
{
   return a > b ? a : b;
}

static unsigned
pvrgpu_positive_extent_to_unsigned(int extent)
{
   return extent > 0 ? (unsigned)extent : 0;
}

static unsigned
pvrgpu_box_end_unsigned(int origin, int extent)
{
   if (origin < 0 || extent <= 0)
      return pvrgpu_positive_extent_to_unsigned(extent);
   return (unsigned)origin + (unsigned)extent;
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
pvrgpu_emit_framebuffer_blit_command(struct pipe_context *pipe,
                                     const struct pipe_blit_info *info)
{
   if (!pvrgpu_is_observable_framebuffer_blit(info))
      return;

   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   if (ctx && ctx->driver_draw_command_emitted)
      return;
   if (pvrgpu_case_prefers_draw_counter_sequence())
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

   unsigned framebuffer_width =
      info->dst.resource ? info->dst.resource->width0 : blit_width;
   unsigned framebuffer_height =
      info->dst.resource ? info->dst.resource->height0 : blit_height;
   framebuffer_width =
      pvrgpu_max_unsigned(framebuffer_width,
                          pvrgpu_box_end_unsigned(info->dst.box.x,
                                                  info->dst.box.width));
   framebuffer_height =
      pvrgpu_max_unsigned(framebuffer_height,
                          pvrgpu_box_end_unsigned(info->dst.box.y,
                                                  info->dst.box.height));
   if (ctx) {
      framebuffer_width = pvrgpu_max_unsigned(framebuffer_width,
                                              ctx->framebuffer.width);
      framebuffer_height = pvrgpu_max_unsigned(framebuffer_height,
                                               ctx->framebuffer.height);
      framebuffer_width = pvrgpu_max_unsigned(framebuffer_width,
                                              ctx->max_framebuffer_width);
      framebuffer_height = pvrgpu_max_unsigned(framebuffer_height,
                                               ctx->max_framebuffer_height);
   }

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
   if (!pvrgpu_can_blit_as_2d_copy(info)) {
      pvrgpu_emit_unsupported_resource_op(pipe,
                                          "unsupported_blit",
                                          "no-scale-same-format-rgba-only");
      return;
   }

   pvrgpu_copy_2d_region_unchecked(info->dst.resource,
                                   info->dst.box.x,
                                   info->dst.box.y,
                                   info->src.resource,
                                   &info->src.box);
   pvrgpu_counter_eventf("blit",
                         "dst=%ux%u dst_box=%d,%d,%d,%d,%d,%d "
                         "src=%ux%u src_box=%d,%d,%d,%d,%d,%d "
                         "format=%s dst_bind=0x%x src_bind=0x%x mask=0x%x "
                         "filter=%u",
                         info->dst.resource->width0,
                         info->dst.resource->height0,
                         info->dst.box.x,
                         info->dst.box.y,
                         info->dst.box.z,
                         info->dst.box.width,
                         info->dst.box.height,
                         info->dst.box.depth,
                         info->src.resource->width0,
                         info->src.resource->height0,
                         info->src.box.x,
                         info->src.box.y,
                         info->src.box.z,
                         info->src.box.width,
                         info->src.box.height,
                         info->src.box.depth,
                         util_format_name(info->dst.format),
                         info->dst.resource->bind,
                         info->src.resource->bind,
                         info->mask,
                         info->filter);
   pvrgpu_emit_framebuffer_blit_command(pipe, info);
}

static void
pvrgpu_flush_resource(struct pipe_context *pipe,
                      struct pipe_resource *resource)
{
   (void)pipe;
   pvrgpu_counter_eventf("flush_resource",
                         "target=%u width=%u height=%u format=%s",
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
   screen->resource_destroy = pvrgpu_resource_destroy;
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
   context->resource_copy_region = pvrgpu_resource_copy_region;
   context->blit = pvrgpu_blit;
   context->flush_resource = pvrgpu_flush_resource;
}
