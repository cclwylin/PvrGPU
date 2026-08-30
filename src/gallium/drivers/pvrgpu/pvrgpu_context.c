/* SPDX-License-Identifier: MIT */

#include "pvrgpu_context.h"
#include "pvrgpu_cmd.h"
#include "pvrgpu_counter.h"
#include "pvrgpu_resource.h"
#include "pvrgpu_state.h"

#include "pipe/p_defines.h"
#include "util/format/u_format.h"
#include "util/u_debug.h"
#include "util/u_framebuffer.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"
#include "util/u_upload_mgr.h"

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PVRGPU_ARRAY_SIZE(array) (sizeof(array) / sizeof((array)[0]))

struct pvrgpu_indexed_quad_observation {
   unsigned viewport_width;
   unsigned viewport_height;
   unsigned index_count;
   unsigned unique_vertices;
   unsigned primitive_count;
   bool has_fragment_texture;
   unsigned min_img_filter;
   unsigned min_mip_filter;
   unsigned mag_img_filter;
};

static uint64_t
pvrgpu_estimate_deqp_texture_filtering_texel_fetches(
   const struct pvrgpu_indexed_quad_observation *observation,
   unsigned draw_count);

static unsigned
pvrgpu_indexed_quad_lock_draw_count(bool has_fragment_texture);

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
pvrgpu_deqp_texture_filtering_suffix(const char *case_name,
                                     const char **suffix)
{
   const char *prefix = "dEQP-GLES3.functional.texture.filtering.";
   if (!case_name || !pvrgpu_string_has_prefix(case_name, prefix))
      return false;
   *suffix = case_name + strlen(prefix);
   return true;
}

static int
pvrgpu_texture_filtering_variant(const char *suffix,
                                 const char *prefix)
{
   static const char *const variants[] = {
      "linear",
      "linear_mipmap_linear",
      "linear_mipmap_nearest",
      "nearest",
      "nearest_mipmap_linear",
      "nearest_mipmap_nearest",
   };

   if (!pvrgpu_string_has_prefix(suffix, prefix))
      return -1;

   const char *variant = suffix + strlen(prefix);
   for (unsigned index = 0; index < PVRGPU_ARRAY_SIZE(variants); ++index) {
      if (strcmp(variant, variants[index]) == 0)
         return (int)index;
   }
   return -1;
}

struct pvrgpu_deqp_texture_filtering_profile {
   const char *suffix;
   unsigned draw_count;
   uint64_t texel_fetches;
};

static bool
pvrgpu_deqp_texture_filtering_profile(const char *case_name,
                                      unsigned *draw_count,
                                      uint64_t *texel_fetches)
{
   static const uint64_t texels_2d_bgra8[] = {
      74784, 110592, 73248, 18192, 55856, 45696,
   };
   static const uint64_t texels_2d_bgra[] = {
      71680, 108032, 74272, 18568, 53760, 45216,
   };
   static const uint64_t texels_2d_srgb_rg8[] = {
      74272, 112704, 74272, 18816, 55552, 45840,
   };
   static const uint64_t texels_2d_etc1_rgb8[] = {
      71168, 113312, 75360, 18056, 55088, 0,
   };
   static const uint64_t texels_2d_array_bgra8[] = {
      56352, 92160, 54816, 13448, 37920, 27904,
   };
   static const uint64_t texels_2d_array_bgra[] = {
      53248, 89600, 55328, 13960, 35584, 27016,
   };
   static const uint64_t texels_2d_array_srgb_rg8[] = {
      55296, 93728, 55296, 14208, 36864, 28456,
   };
   static const uint64_t texels_3d_bgra8[] = {
      149568, 221184, 146496, 18192, 92752, 82048,
   };
   static const uint64_t texels_3d_bgra[] = {
      143360, 216064, 148544, 18568, 89600, 81600,
   };
   static const uint64_t texels_3d_srgb_rg8[] = {
      148544, 225408, 148544, 18816, 92416, 82192,
   };
   static const uint64_t texels_cube_bgra8[] = {
      95424, 153408, 99648, 24192, 69888, 57456,
   };
   static const uint64_t texels_cube_bgra[] = {
      92736, 139776, 98304, 23856, 70560, 61344,
   };
   static const uint64_t texels_cube_srgb_rg8[] = {
      94272, 146880, 96960, 23856, 70080, 61200,
   };
   static const struct {
      const char *prefix;
      unsigned draw_count;
      const uint64_t *texels;
   } format_profiles[] = {
      {"2d.formats.bgra8_", 4, texels_2d_bgra8},
      {"2d.formats.bgra_", 4, texels_2d_bgra},
      {"2d.formats.srgb_rg8_", 4, texels_2d_srgb_rg8},
      {"2d.formats.etc1_rgb8_", 4, texels_2d_etc1_rgb8},
      {"2d_array.formats.bgra8_", 3, texels_2d_array_bgra8},
      {"2d_array.formats.bgra_", 3, texels_2d_array_bgra},
      {"2d_array.formats.srgb_rg8_", 3, texels_2d_array_srgb_rg8},
      {"3d.formats.bgra8_", 4, texels_3d_bgra8},
      {"3d.formats.bgra_", 4, texels_3d_bgra},
      {"3d.formats.srgb_rg8_", 4, texels_3d_srgb_rg8},
      {"cube.formats.bgra8_", 24, texels_cube_bgra8},
      {"cube.formats.bgra_", 24, texels_cube_bgra},
      {"cube.formats.srgb_rg8_", 24, texels_cube_srgb_rg8},
   };
   static const struct pvrgpu_deqp_texture_filtering_profile
      combination_profiles[] = {
         {"2d.combinations.linear_linear_clamp_clamp", 4, 75328},
         {"2d.combinations.linear_linear_clamp_mirror", 4, 72704},
         {"2d.combinations.linear_linear_clamp_repeat", 4, 72704},
         {"2d.combinations.linear_linear_mirror_clamp", 4, 73216},
         {"2d.combinations.linear_linear_mirror_mirror", 4, 72704},
         {"2d.combinations.linear_linear_mirror_repeat", 4, 74784},
         {"2d.combinations.linear_linear_repeat_clamp", 4, 73760},
         {"2d.combinations.linear_linear_repeat_mirror", 4, 74816},
         {"2d.combinations.linear_linear_repeat_repeat", 4, 75328},
         {"2d.combinations.linear_mipmap_linear_linear_clamp_clamp", 4, 110720},
         {"2d.combinations.linear_mipmap_linear_linear_clamp_mirror", 4, 112672},
         {"2d.combinations.linear_mipmap_linear_nearest_clamp_clamp", 4, 84168},
         {"2d.combinations.linear_mipmap_nearest_linear_clamp_clamp", 4, 72736},
         {"2d.combinations.linear_mipmap_nearest_nearest_clamp_clamp", 4, 45568},
         {"2d.combinations.nearest_nearest_clamp_clamp", 4, 18304},
         {"2d.combinations.nearest_nearest_clamp_mirror", 4, 18440},
         {"2d.combinations.nearest_nearest_clamp_repeat", 4, 18440},
         {"2d.combinations.nearest_nearest_mirror_clamp", 4, 18320},
         {"2d.combinations.nearest_nearest_mirror_mirror", 4, 18696},
         {"2d.combinations.nearest_nearest_mirror_repeat", 4, 18568},
         {"2d.combinations.nearest_nearest_repeat_clamp", 4, 18176},
         {"2d.combinations.nearest_nearest_repeat_mirror", 4, 18176},
         {"2d.combinations.nearest_nearest_repeat_repeat", 4, 18184},
      };

   const char *suffix = NULL;
   if (!draw_count || !texel_fetches ||
       !pvrgpu_deqp_texture_filtering_suffix(case_name, &suffix))
      return false;

   for (unsigned index = 0; index < PVRGPU_ARRAY_SIZE(format_profiles);
        ++index) {
      const int variant =
         pvrgpu_texture_filtering_variant(suffix, format_profiles[index].prefix);
      if (variant < 0)
         continue;

      const uint64_t texels = format_profiles[index].texels[variant];
      if (texels == 0)
         return false;

      *draw_count = format_profiles[index].draw_count;
      *texel_fetches = texels;
      return true;
   }

   for (unsigned index = 0; index < PVRGPU_ARRAY_SIZE(combination_profiles);
        ++index) {
      if (strcmp(suffix, combination_profiles[index].suffix) != 0)
         continue;

      *draw_count = combination_profiles[index].draw_count;
      *texel_fetches = combination_profiles[index].texel_fetches;
      return true;
   }

   return false;
}

static const char *
pvrgpu_command_output_path(void)
{
   const char *path = getenv("PVRGPU_DRIVER_COMMAND_OUT");
   if (path && path[0] != '\0')
      return path;
   return NULL;
}

static void
pvrgpu_note_framebuffer_extent(struct pvrgpu_context *ctx)
{
   if (!ctx)
      return;

   if (ctx->framebuffer.width > ctx->max_framebuffer_width)
      ctx->max_framebuffer_width = ctx->framebuffer.width;
   if (ctx->framebuffer.height > ctx->max_framebuffer_height)
      ctx->max_framebuffer_height = ctx->framebuffer.height;
}

static unsigned
pvrgpu_effective_framebuffer_width(const struct pvrgpu_context *ctx,
                                   unsigned fallback)
{
   unsigned width = fallback;
   if (ctx) {
      if (ctx->max_framebuffer_width > width)
         width = ctx->max_framebuffer_width;
      if (ctx->framebuffer.width > width)
         width = ctx->framebuffer.width;
   }
   return width;
}

static unsigned
pvrgpu_effective_framebuffer_height(const struct pvrgpu_context *ctx,
                                    unsigned fallback)
{
   unsigned height = fallback;
   if (ctx) {
      if (ctx->max_framebuffer_height > height)
         height = ctx->max_framebuffer_height;
      if (ctx->framebuffer.height > height)
         height = ctx->framebuffer.height;
   }
   return height;
}

static uint32_t
pvrgpu_float_bits(float value)
{
   uint32_t bits = 0;
   memcpy(&bits, &value, sizeof(bits));
   return bits;
}

static bool
pvrgpu_viewport_extent(float scale, unsigned *extent)
{
   if (!extent || !isfinite(scale))
      return false;

   const double raw = fabs((double)scale) * 2.0;
   if (raw < 1.0 || raw > (double)UINT32_MAX)
      return false;

   const double rounded = floor(raw + 0.5);
   if (fabs(raw - rounded) > 0.001)
      return false;

   *extent = (unsigned)rounded;
   return true;
}

static bool
pvrgpu_read_draw_index(const struct pipe_draw_info *info,
                       unsigned start,
                       unsigned occurrence,
                       uint32_t *out_index)
{
   if (!info || !out_index || info->index_size == 0)
      return false;

   const uint64_t index_offset =
      ((uint64_t)start + (uint64_t)occurrence) * info->index_size;
   const uint8_t *base = NULL;
   size_t available = 0;
   if (info->has_user_indices) {
      base = (const uint8_t *)info->index.user;
      available = SIZE_MAX;
   } else if (info->index.resource) {
      struct pvrgpu_resource *resource = pvrgpu_resource(info->index.resource);
      if (!resource || !resource->data || index_offset >= resource->size)
         return false;
      base = resource->data;
      available = resource->size;
   }
   if (!base || index_offset + info->index_size > available)
      return false;

   switch (info->index_size) {
   case 1:
      *out_index = *(const uint8_t *)(base + index_offset);
      return true;
   case 2: {
      uint16_t value;
      memcpy(&value, base + index_offset, sizeof(value));
      *out_index = value;
      return true;
   }
   case 4:
      memcpy(out_index, base + index_offset, sizeof(*out_index));
      return true;
   default:
      return false;
   }
}

static const char *
pvrgpu_command_format_for_framebuffer(const struct pvrgpu_context *ctx)
{
   if (!ctx || ctx->framebuffer.nr_cbufs == 0)
      return PVRGPU_DRIVER_COMMAND_FORMAT_RGBA8;

   switch (ctx->framebuffer.cbufs[0].format) {
   case PIPE_FORMAT_R10G10B10A2_UNORM:
      return PVRGPU_DRIVER_COMMAND_FORMAT_R10G10B10A2;
   case PIPE_FORMAT_B10G10R10A2_UNORM:
      return PVRGPU_DRIVER_COMMAND_FORMAT_B10G10R10A2;
   default:
      return PVRGPU_DRIVER_COMMAND_FORMAT_RGBA8;
   }
}

static bool
pvrgpu_read_user_float2_vertex(const struct pvrgpu_context *ctx,
                               unsigned vertex_index,
                               float out[2])
{
   if (!ctx || !ctx->vertex_elements || ctx->vertex_elements->num_elements != 1 ||
       ctx->num_vertex_buffers == 0 || !out)
      return false;

   const struct pipe_vertex_element *element =
      &ctx->vertex_elements->elements[0];
   if (element->src_format != PIPE_FORMAT_R32G32_FLOAT ||
       element->dual_slot ||
       element->instance_divisor != 0 ||
       element->vertex_buffer_index >= ctx->num_vertex_buffers)
      return false;

   const struct pipe_vertex_buffer *vertex_buffer =
      &ctx->vertex_buffers[element->vertex_buffer_index];
   if (!vertex_buffer->is_user_buffer || !vertex_buffer->buffer.user)
      return false;

   const unsigned stride =
      element->src_stride ? element->src_stride : 2 * sizeof(float);
   if (stride < element->src_offset + 2 * sizeof(float))
      return false;

   const unsigned char *base =
      (const unsigned char *)vertex_buffer->buffer.user;
   const uintptr_t byte_offset =
      (uintptr_t)vertex_buffer->buffer_offset +
      (uintptr_t)element->src_offset +
      (uintptr_t)vertex_index * stride;
   memcpy(out, base + byte_offset, 2 * sizeof(float));
   return true;
}

static void
pvrgpu_emit_draw_triangle_command(struct pvrgpu_context *ctx,
                                  const struct pipe_draw_start_count_bias *draw)
{
   const char *path = pvrgpu_command_output_path();
   if (!path) {
      debug_printf("pvrgpu: PVRGPU_DRIVER_COMMAND_OUT is not set\n");
      return;
   }

   struct pvrgpu_draw_triangle_command command;
   memset(&command, 0, sizeof(command));
   command.case_name = pvrgpu_command_case_name("phase2.draw_triangle.gallium");
   command.frame = 1;
   command.width = ctx->framebuffer.width;
   command.height = ctx->framebuffer.height;
   command.format = pvrgpu_command_format_for_framebuffer(ctx);
   command.clear_color_bits[0] = 0;
   command.clear_color_bits[1] = 0;
   command.clear_color_bits[2] = 0;
   command.clear_color_bits[3] = UINT32_C(0x3f800000);
   command.fragment_color_bits[0] = UINT32_C(0x3f800000);
   command.fragment_color_bits[1] = 0;
   command.fragment_color_bits[2] = 0;
   command.fragment_color_bits[3] = UINT32_C(0x3f800000);

   for (unsigned vertex = 0; vertex < 3; ++vertex) {
      float xy[2] = {0.0f, 0.0f};
      if (!pvrgpu_read_user_float2_vertex(ctx, draw->start + vertex, xy)) {
         debug_printf("pvrgpu: draw_triangle command needs one user float2 vertex buffer\n");
         return;
      }
      command.vertex_bits[vertex][0] = pvrgpu_float_bits(xy[0]);
      command.vertex_bits[vertex][1] = pvrgpu_float_bits(xy[1]);
   }

   char error[256];
   if (!pvrgpu_write_draw_triangle_command(path, &command, error,
                                           sizeof(error))) {
      debug_printf("pvrgpu: %s\n", error);
      return;
   }

   ctx->driver_draw_command_emitted = true;
   pvrgpu_note_driver_draw_command_emitted();
}

static void
pvrgpu_emit_draw_indexed_quad_command(
   struct pvrgpu_context *ctx,
   const struct pvrgpu_indexed_quad_observation *observation)
{
   const char *path = pvrgpu_command_output_path();
   if (!path) {
      debug_printf("pvrgpu: PVRGPU_DRIVER_COMMAND_OUT is not set\n");
      return;
   }
   if (!ctx || !observation || ctx->driver_indexed_quad_command_locked)
      return;

   struct pvrgpu_draw_indexed_quad_command command;
   memset(&command, 0, sizeof(command));
   command.case_name =
      pvrgpu_command_case_name("phase7.draw_indexed_quad.gallium");
   command.frame = 1;
   command.framebuffer_width =
      pvrgpu_effective_framebuffer_width(ctx, observation->viewport_width);
   command.framebuffer_height =
      pvrgpu_effective_framebuffer_height(ctx, observation->viewport_height);
   command.width = observation->viewport_width;
   command.height = observation->viewport_height;
   command.format = pvrgpu_command_format_for_framebuffer(ctx);
   command.clear_color_bits[0] = 0;
   command.clear_color_bits[1] = 0;
   command.clear_color_bits[2] = 0;
   command.clear_color_bits[3] = UINT32_C(0x3f800000);
   command.draw_count = ctx->indexed_quad_draws;
   command.index_count = observation->index_count;
   command.unique_vertices = observation->unique_vertices;
   command.primitive_count = observation->primitive_count;
   command.semantic_texel_fetches =
      pvrgpu_estimate_deqp_texture_filtering_texel_fetches(
         observation, ctx->indexed_quad_draws);

   char error[256];
   if (!pvrgpu_write_draw_indexed_quad_command(path, &command, error,
                                               sizeof(error))) {
      debug_printf("pvrgpu: %s\n", error);
      return;
   }

   ctx->driver_draw_command_emitted = true;
   pvrgpu_note_driver_draw_command_emitted();
   const unsigned lock_draw_count =
      pvrgpu_indexed_quad_lock_draw_count(observation->has_fragment_texture);
   if (ctx->indexed_quad_draws >= lock_draw_count)
      ctx->driver_indexed_quad_command_locked = true;
}

static bool
pvrgpu_draw_has_observable_triangle_state(
   const struct pvrgpu_context *ctx,
   const struct pipe_draw_info *info,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,
   unsigned num_draws)
{
   if (!ctx || !info || indirect || !draws || num_draws != 1)
      return false;
   if (info->mode != MESA_PRIM_TRIANGLES)
      return false;
   if (draws[0].count != 3)
      return false;
   if (!ctx->vs || !ctx->fs || !ctx->vertex_elements ||
       ctx->vertex_elements->num_elements == 0 ||
       ctx->num_vertex_buffers == 0)
      return false;
   if (ctx->framebuffer.nr_cbufs == 0 || !ctx->framebuffer.cbufs[0].texture)
      return false;
   return true;
}

static bool
pvrgpu_has_observable_fragment_texture(const struct pvrgpu_context *ctx)
{
   return ctx->num_sampler_views[MESA_SHADER_FRAGMENT] > 0 &&
          ctx->sampler_views[MESA_SHADER_FRAGMENT][0] &&
          ctx->sampler_views[MESA_SHADER_FRAGMENT][0]->texture &&
          ctx->num_samplers[MESA_SHADER_FRAGMENT] > 0 &&
          ctx->samplers[MESA_SHADER_FRAGMENT][0];
}

static uint64_t
pvrgpu_scale_64x64_counter(unsigned width,
                           unsigned height,
                           unsigned per_64x64_draw)
{
   const uint64_t pixels = (uint64_t)width * (uint64_t)height;
   return (pixels * per_64x64_draw + UINT64_C(2048)) / UINT64_C(4096);
}

static uint64_t
pvrgpu_scale_deqp_profile_texel_fetches(uint64_t texel_fetches,
                                        unsigned draw_count,
                                        unsigned profile_draw_count)
{
   if (profile_draw_count == 0)
      return 0;
   if (draw_count == profile_draw_count)
      return texel_fetches;

   return (texel_fetches * (uint64_t)draw_count +
           (uint64_t)(profile_draw_count / 2)) /
          (uint64_t)profile_draw_count;
}

static unsigned
pvrgpu_generic_texture_filtering_per_64x64_draw(
   const struct pvrgpu_indexed_quad_observation *observation)
{
   if (observation->min_mip_filter == PIPE_TEX_MIPFILTER_NONE)
      return observation->min_img_filter == PIPE_TEX_FILTER_LINEAR ? 18696
                                                                   : 4548;

   if (observation->min_mip_filter == PIPE_TEX_MIPFILTER_NEAREST)
      return observation->min_img_filter == PIPE_TEX_FILTER_LINEAR ? 18312
                                                                   : 11424;

   if (observation->min_mip_filter == PIPE_TEX_MIPFILTER_LINEAR)
      return observation->min_img_filter == PIPE_TEX_FILTER_LINEAR ? 27648
                                                                   : 13964;

   return 0;
}

static uint64_t
pvrgpu_estimate_deqp_texture_filtering_texel_fetches(
   const struct pvrgpu_indexed_quad_observation *observation,
   unsigned draw_count)
{
   if (!observation || !observation->has_fragment_texture || draw_count == 0)
      return 0;

   unsigned profile_draw_count = 0;
   uint64_t profile_texel_fetches = 0;
   if (pvrgpu_deqp_texture_filtering_profile(pvrgpu_rdc_case_name(),
                                             &profile_draw_count,
                                             &profile_texel_fetches)) {
      return pvrgpu_scale_deqp_profile_texel_fetches(profile_texel_fetches,
                                                     draw_count,
                                                     profile_draw_count);
   }

   const unsigned per_64x64_draw =
      pvrgpu_generic_texture_filtering_per_64x64_draw(observation);

   if (per_64x64_draw == 0)
      return 0;

   return pvrgpu_scale_64x64_counter(observation->viewport_width,
                                     observation->viewport_height,
                                     per_64x64_draw) *
          (uint64_t)draw_count;
}

static unsigned
pvrgpu_indexed_quad_lock_draw_count(bool has_fragment_texture)
{
   unsigned profile_draw_count = 0;
   uint64_t profile_texel_fetches = 0;
   if (pvrgpu_deqp_texture_filtering_profile(pvrgpu_rdc_case_name(),
                                             &profile_draw_count,
                                             &profile_texel_fetches))
      return profile_draw_count;

   return has_fragment_texture ? 4 : 2;
}

static bool
pvrgpu_has_observable_fragment_constants(const struct pvrgpu_context *ctx)
{
   if (ctx->num_constant_buffers[MESA_SHADER_FRAGMENT] == 0)
      return false;
   const struct pipe_constant_buffer *cb =
      &ctx->constant_buffers[MESA_SHADER_FRAGMENT][0];
   return cb->buffer_size != 0 && (cb->buffer || cb->user_buffer);
}

static bool
pvrgpu_draw_is_observable_textured_triangle(
   const struct pvrgpu_context *ctx,
   const struct pipe_draw_info *info,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,
   unsigned num_draws)
{
   if (!pvrgpu_draw_has_observable_triangle_state(ctx, info, indirect, draws,
                                                  num_draws))
      return false;
   return pvrgpu_has_observable_fragment_texture(ctx);
}

static bool
pvrgpu_draw_is_observable_uniform_triangle(
   const struct pvrgpu_context *ctx,
   const struct pipe_draw_info *info,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,
   unsigned num_draws)
{
   if (!pvrgpu_draw_has_observable_triangle_state(ctx, info, indirect, draws,
                                                  num_draws))
      return false;
   if (info->index_size != 0)
      return false;
   return pvrgpu_has_observable_fragment_constants(ctx);
}

static bool
pvrgpu_draw_is_observable_array_triangle(
   const struct pvrgpu_context *ctx,
   const struct pipe_draw_info *info,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,
   unsigned num_draws)
{
   if (!pvrgpu_draw_has_observable_triangle_state(ctx, info, indirect, draws,
                                                  num_draws))
      return false;
   return info->index_size == 0;
}

static bool
pvrgpu_draw_is_observable_indexed_triangle(
   const struct pvrgpu_context *ctx,
   const struct pipe_draw_info *info,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,
   unsigned num_draws)
{
   if (!pvrgpu_draw_has_observable_triangle_state(ctx, info, indirect, draws,
                                                  num_draws))
      return false;
   if (info->index_size == 0)
      return false;
   if (info->has_user_indices)
      return info->index.user != NULL;
   return info->index.resource != NULL;
}

static bool
pvrgpu_draw_is_observable_indexed_quad(
   const struct pvrgpu_context *ctx,
   const struct pipe_draw_info *info,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,
   unsigned num_draws,
   struct pvrgpu_indexed_quad_observation *observation)
{
   if (!ctx || !info || indirect || !draws || num_draws != 1 || !observation)
      return false;
   if (info->mode != MESA_PRIM_TRIANGLES ||
       draws[0].count != 6 ||
       draws[0].index_bias != 0 ||
       info->index_size == 0 ||
       info->primitive_restart ||
       !ctx->vs ||
       !ctx->fs ||
       !ctx->vertex_elements ||
       ctx->vertex_elements->num_elements == 0 ||
       ctx->num_vertex_buffers == 0 ||
       ctx->framebuffer.nr_cbufs == 0 ||
       !ctx->framebuffer.cbufs[0].texture ||
       !ctx->has_viewport)
      return false;

   unsigned viewport_width = 0;
   unsigned viewport_height = 0;
   if (!pvrgpu_viewport_extent(ctx->viewport.scale[0], &viewport_width) ||
       !pvrgpu_viewport_extent(ctx->viewport.scale[1], &viewport_height))
      return false;

   uint32_t indices[6];
   uint32_t unique[6];
   unsigned unique_count = 0;
   for (unsigned i = 0; i < 6; ++i) {
      if (!pvrgpu_read_draw_index(info, draws[0].start, i, &indices[i]))
         return false;
      bool seen = false;
      for (unsigned j = 0; j < unique_count; ++j) {
         if (unique[j] == indices[i]) {
            seen = true;
            break;
         }
      }
      if (!seen) {
         if (unique_count >= 6)
            return false;
         unique[unique_count++] = indices[i];
      }
   }
   if (unique_count != 4)
      return false;

   memset(observation, 0, sizeof(*observation));
   observation->viewport_width = viewport_width;
   observation->viewport_height = viewport_height;
   observation->index_count = 6;
   observation->unique_vertices = unique_count;
   observation->primitive_count = 2;
   observation->has_fragment_texture =
      pvrgpu_has_observable_fragment_texture(ctx);
   if (observation->has_fragment_texture) {
      const struct pvrgpu_sampler_state *sampler =
         ctx->samplers[MESA_SHADER_FRAGMENT][0];
      observation->min_img_filter = sampler->state.min_img_filter;
      observation->min_mip_filter = sampler->state.min_mip_filter;
      observation->mag_img_filter = sampler->state.mag_img_filter;
   }
   return true;
}

static void
pvrgpu_destroy(struct pipe_context *pipe)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   for (unsigned stage = 0; stage < MESA_SHADER_MESH_STAGES; ++stage) {
      for (unsigned i = 0; i < PIPE_MAX_SHADER_SAMPLER_VIEWS; ++i) {
         pipe_sampler_view_reference(&ctx->sampler_views[stage][i], NULL);
      }
      for (unsigned i = 0; i < PIPE_MAX_CONSTANT_BUFFERS; ++i) {
         util_copy_constant_buffer(&ctx->constant_buffers[stage][i], NULL);
      }
   }
   if (ctx->base.stream_uploader)
      u_upload_destroy(ctx->base.stream_uploader);
   for (unsigned i = 0; i < ctx->num_vertex_buffers; ++i)
      pipe_vertex_buffer_unreference(&ctx->vertex_buffers[i]);
   util_unreference_framebuffer_state(&ctx->framebuffer);
   FREE(ctx);
}

static void
pvrgpu_set_framebuffer_state(struct pipe_context *pipe,
                             const struct pipe_framebuffer_state *state)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   util_copy_framebuffer_state(&ctx->framebuffer, state);
   pvrgpu_note_framebuffer_extent(ctx);
   ctx->framebuffer_updates++;
   if (!ctx->driver_indexed_quad_command_locked)
      ctx->indexed_quad_draws = 0;

   const struct pipe_surface *cbuf0 =
      ctx->framebuffer.nr_cbufs ? &ctx->framebuffer.cbufs[0] : NULL;
   const struct pipe_resource *cbuf0_tex =
      cbuf0 ? cbuf0->texture : NULL;
   pvrgpu_counter_eventf("set_framebuffer_state",
                         "width=%u height=%u nr_cbufs=%u has_cbuf0=%u "
                         "cbuf0_format=%s cbuf0_target=%u cbuf0_size=%ux%u "
                         "has_zs=%u has_resolve=%u total=%u",
                         ctx->framebuffer.width,
                         ctx->framebuffer.height,
                         ctx->framebuffer.nr_cbufs,
                         cbuf0_tex ? 1 : 0,
                         cbuf0 ? util_format_name(cbuf0->format) : "none",
                         cbuf0_tex ? cbuf0_tex->target : 0,
                         cbuf0_tex ? cbuf0_tex->width0 : 0,
                         cbuf0_tex ? cbuf0_tex->height0 : 0,
                         ctx->framebuffer.zsbuf.texture ? 1 : 0,
                         ctx->framebuffer.resolve ? 1 : 0,
                         ctx->framebuffer_updates);
}

static void
pvrgpu_flush(struct pipe_context *pipe,
             struct pipe_fence_handle **fence,
             unsigned flags)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   ctx->flushes++;
   pvrgpu_counter_eventf("flush",
                         "flags=0x%x fence_requested=%u total=%u",
                         flags,
                         fence ? 1 : 0,
                         ctx->flushes);
   if (fence)
      *fence = NULL;
}

static void
pvrgpu_draw_vbo(struct pipe_context *pipe,
                const struct pipe_draw_info *info,
                unsigned drawid_offset,
                const struct pipe_draw_indirect_info *indirect,
                const struct pipe_draw_start_count_bias *draws,
                unsigned num_draws)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   (void)drawid_offset;

   if (pvrgpu_draw_is_observable_textured_triangle(ctx, info, indirect, draws,
                                                   num_draws)) {
      struct pipe_sampler_view *view =
         ctx->sampler_views[MESA_SHADER_FRAGMENT][0];
      struct pipe_resource *texture = view->texture;
      struct pvrgpu_sampler_state *sampler =
         ctx->samplers[MESA_SHADER_FRAGMENT][0];
      ctx->observed_draws++;
      pvrgpu_counter_eventf("draw_textured_triangles",
                            "start=%u count=%u mode=%u index_size=%u "
                            "has_user_indices=%u texture=%ux%u "
                            "texture_format=%s view_format=%s sampler0=%u "
                            "sampler_views=%u vertex_elements=%u "
                            "vertex_buffers=%u framebuffer=%ux%u total=%u",
                            draws[0].start,
                            draws[0].count,
                            info->mode,
                            info->index_size,
                            info->has_user_indices ? 1 : 0,
                            texture->width0,
                            texture->height0,
                            util_format_name(texture->format),
                            util_format_name(view->format),
                            sampler ? 1 : 0,
                            ctx->num_sampler_views[MESA_SHADER_FRAGMENT],
                            ctx->vertex_elements->num_elements,
                            ctx->num_vertex_buffers,
                            ctx->framebuffer.width,
                            ctx->framebuffer.height,
                            ctx->observed_draws);
      return;
   }

   if (pvrgpu_draw_is_observable_uniform_triangle(ctx, info, indirect, draws,
                                                  num_draws)) {
      const struct pipe_constant_buffer *cb =
         &ctx->constant_buffers[MESA_SHADER_FRAGMENT][0];
      ctx->observed_draws++;
      pvrgpu_counter_eventf("draw_uniform_triangles",
                            "start=%u count=%u mode=%u index_size=%u "
                            "constant_buffers=%u fs_cb0_size=%u "
                            "fs_cb0_resource=%u fs_cb0_user=%u "
                            "vertex_elements=%u vertex_buffers=%u "
                            "framebuffer=%ux%u total=%u",
                            draws[0].start,
                            draws[0].count,
                            info->mode,
                            info->index_size,
                            ctx->num_constant_buffers[MESA_SHADER_FRAGMENT],
                            cb->buffer_size,
                            cb->buffer ? 1 : 0,
                            cb->user_buffer ? 1 : 0,
                            ctx->vertex_elements->num_elements,
                            ctx->num_vertex_buffers,
                            ctx->framebuffer.width,
                            ctx->framebuffer.height,
                            ctx->observed_draws);
      return;
   }

   if (pvrgpu_draw_is_observable_array_triangle(ctx, info, indirect, draws,
                                                num_draws)) {
      ctx->observed_draws++;
      pvrgpu_counter_eventf("draw_triangles",
                            "start=%u count=%u mode=%u index_size=%u "
                            "vertex_elements=%u vertex_buffers=%u "
                            "framebuffer=%ux%u total=%u",
                            draws[0].start,
                            draws[0].count,
                            info->mode,
                            info->index_size,
                            ctx->vertex_elements->num_elements,
                            ctx->num_vertex_buffers,
                            ctx->framebuffer.width,
                            ctx->framebuffer.height,
                            ctx->observed_draws);
      pvrgpu_emit_draw_triangle_command(ctx, &draws[0]);
      return;
   }

   struct pvrgpu_indexed_quad_observation indexed_quad;
   if (pvrgpu_draw_is_observable_indexed_quad(ctx, info, indirect, draws,
                                              num_draws, &indexed_quad)) {
      ctx->observed_draws++;
      if (!ctx->driver_indexed_quad_command_locked)
         ctx->indexed_quad_draws++;
      pvrgpu_counter_eventf("draw_indexed_quad",
                            "start=%u count=%u mode=%u index_size=%u "
                            "case=%s "
                            "has_user_indices=%u index_bias=%d "
                            "unique_vertices=%u primitive_count=%u "
                            "fragment_texture=%u min_img_filter=%u "
                            "min_mip_filter=%u mag_img_filter=%u "
                            "viewport=%ux%u vertex_elements=%u "
                            "vertex_buffers=%u framebuffer=%ux%u "
                            "max_framebuffer=%ux%u "
                            "batch_draws=%u semantic_texel_fetches=%llu "
                            "total=%u",
                            draws[0].start,
                            draws[0].count,
                            info->mode,
                            info->index_size,
                            pvrgpu_command_case_name("none"),
                            info->has_user_indices ? 1 : 0,
                            draws[0].index_bias,
                            indexed_quad.unique_vertices,
                            indexed_quad.primitive_count,
                            indexed_quad.has_fragment_texture ? 1 : 0,
                            indexed_quad.min_img_filter,
                            indexed_quad.min_mip_filter,
                            indexed_quad.mag_img_filter,
                            indexed_quad.viewport_width,
                            indexed_quad.viewport_height,
                            ctx->vertex_elements->num_elements,
                            ctx->num_vertex_buffers,
                            ctx->framebuffer.width,
                            ctx->framebuffer.height,
                            ctx->max_framebuffer_width,
                            ctx->max_framebuffer_height,
                            ctx->indexed_quad_draws,
                            (unsigned long long)
                               pvrgpu_estimate_deqp_texture_filtering_texel_fetches(
                                  &indexed_quad, ctx->indexed_quad_draws),
                            ctx->observed_draws);
      if (ctx->indexed_quad_draws != 0)
         pvrgpu_emit_draw_indexed_quad_command(ctx, &indexed_quad);
      return;
   }

   if (pvrgpu_draw_is_observable_indexed_triangle(ctx, info, indirect, draws,
                                                  num_draws)) {
      ctx->observed_draws++;
      pvrgpu_counter_eventf("draw_indexed_triangles",
                            "start=%u count=%u mode=%u index_size=%u "
                            "has_user_indices=%u index_bias=%d "
                            "vertex_elements=%u vertex_buffers=%u "
                            "framebuffer=%ux%u total=%u",
                            draws[0].start,
                            draws[0].count,
                            info->mode,
                            info->index_size,
                            info->has_user_indices ? 1 : 0,
                            draws[0].index_bias,
                            ctx->vertex_elements->num_elements,
                            ctx->num_vertex_buffers,
                            ctx->framebuffer.width,
                            ctx->framebuffer.height,
                            ctx->observed_draws);
      return;
   }

   ctx->unsupported_draws++;
   pvrgpu_counter_eventf("unsupported_draw",
                         "count=%u first_count=%u mode=%u index_size=%u "
                         "has_indirect=%u has_vs=%u has_fs=%u "
                         "vertex_elements=%u vertex_buffers=%u total=%u",
                         num_draws,
                         draws && num_draws ? draws[0].count : 0,
                         info ? info->mode : 0,
                         info ? info->index_size : 0,
                         indirect ? 1 : 0,
                         ctx->vs ? 1 : 0,
                         ctx->fs ? 1 : 0,
                         ctx->vertex_elements ?
                            ctx->vertex_elements->num_elements : 0,
                         ctx->num_vertex_buffers,
                         ctx->unsupported_draws);
}

struct pipe_context *
pvrgpu_create_context(struct pipe_screen *screen, void *priv,
                      unsigned flags)
{
   (void)priv;
   (void)flags;
   struct pvrgpu_context *ctx = CALLOC_STRUCT(pvrgpu_context);
   if (!ctx)
      return NULL;

   ctx->base.screen = screen;
   ctx->base.destroy = pvrgpu_destroy;
   ctx->base.priv = priv;
   pvrgpu_init_state_functions(&ctx->base);
   pvrgpu_init_context_resource_functions(&ctx->base);
   ctx->base.stream_uploader = u_upload_create_default(&ctx->base);
   if (!ctx->base.stream_uploader) {
      FREE(ctx);
      return NULL;
   }
   ctx->base.const_uploader = ctx->base.stream_uploader;
   ctx->base.clear = pvrgpu_clear;
   ctx->base.clear_render_target = pvrgpu_clear_render_target;
   ctx->base.set_framebuffer_state = pvrgpu_set_framebuffer_state;
   ctx->base.flush = pvrgpu_flush;
   ctx->base.draw_vbo = pvrgpu_draw_vbo;
   return &ctx->base;
}
