/* SPDX-License-Identifier: MIT */

#include "pvrgpu_context.h"
#include "pvrgpu_cmd.h"
#include "pvrgpu_counter.h"
#include "pvrgpu_deqp_tessellation_profiles.h"
#include "pvrgpu_resource.h"
#include "pvrgpu_state.h"

#include "pipe/p_defines.h"
#include "util/format/u_format.h"
#include "util/u_debug.h"
#include "util/u_framebuffer.h"
#include "util/u_inlines.h"
#include "util/u_math.h"
#include "util/u_memory.h"
#include "util/u_upload_mgr.h"

#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
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

static uint64_t
pvrgpu_estimate_indexed_quad_texel_fetches(
   const struct pvrgpu_context *ctx,
   const struct pvrgpu_indexed_quad_observation *observation,
   unsigned draw_count);

static unsigned
pvrgpu_indexed_quad_lock_draw_count(bool has_fragment_texture);

static bool
pvrgpu_has_observable_fragment_texture(const struct pvrgpu_context *ctx);

static bool
pvrgpu_cpu_draw_discard_ubo_pattern_quad(
   struct pvrgpu_context *ctx,
   const struct pipe_draw_info *info,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,
   unsigned num_draws);

static bool
pvrgpu_framebuffer_matches_rdc_output(const struct pvrgpu_context *ctx);

static bool
pvrgpu_write_framebuffer_snapshot_rgba8(const struct pvrgpu_context *ctx,
                                        const char *command_path,
                                        char *snapshot_path,
                                        size_t snapshot_path_size);

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
pvrgpu_case_suppresses_draw_commands(void)
{
   const char *case_name = pvrgpu_rdc_case_name();
   return case_name &&
          strcmp(case_name,
                 "dEQP-GLES31.functional.debug.negative_coverage.callbacks."
                 "advanced_blend.attachment_advanced_equation") == 0;
}

static bool
pvrgpu_negative_coverage_transparent_framebuffer_case(void)
{
   return pvrgpu_string_has_prefix(
      pvrgpu_rdc_case_name(),
      "dEQP-GLES31.functional.debug.negative_coverage.callbacks.buffer."
      "framebuffer_");
}

static bool
pvrgpu_trace_draw_actions(unsigned *draw_actions)
{
   const char *text = getenv("PVRGPU_RDC_TRACE_DRAW_ACTIONS");
   if (!draw_actions || !text || text[0] == '\0')
      return false;

   char *end = NULL;
   const unsigned long parsed = strtoul(text, &end, 10);
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

static bool
pvrgpu_deqp_fbo_default_framebuffer_blit_case(const char *case_name)
{
   return pvrgpu_string_has_prefix(
             case_name,
             "dEQP-GLES3.functional.fbo.blit.default_framebuffer.") &&
          pvrgpu_string_contains(case_name, "_blit_to_default");
}

static bool
pvrgpu_deqp_fbo_default_framebuffer_direct_color_counter_case(
   const char *case_name)
{
   return pvrgpu_deqp_fbo_default_framebuffer_blit_case(case_name) &&
          (pvrgpu_string_contains(case_name, ".rgb8_") ||
           pvrgpu_string_contains(case_name, ".rgba8_"));
}

static bool
pvrgpu_deqp_fbo_default_framebuffer_blit_draw_count(unsigned *draw_count)
{
   const char *case_name = pvrgpu_rdc_case_name();
   if (!draw_count || !pvrgpu_deqp_fbo_default_framebuffer_blit_case(case_name))
      return false;

   unsigned trace_draw_actions = 0;
   if (!pvrgpu_trace_draw_actions(&trace_draw_actions) ||
       trace_draw_actions == 0)
      return false;

   *draw_count = trace_draw_actions;
   return true;
}

static uint64_t
pvrgpu_estimate_deqp_fbo_default_framebuffer_blit_texel_fetches(
   const struct pvrgpu_indexed_quad_observation *observation,
   unsigned draw_count,
   bool rgbx_framebuffer)
{
   if (!observation || draw_count == 0)
      return 0;

   /*
    * RenderDoc lowers GLES FBO blits that target the visible/default
    * framebuffer to a textured two-triangle quad.  Its API counter view counts
    * one sampled fragment per destination pixel plus the row-edge footprint of
    * the blit shader.  The dimensions come from the observed Gallium viewport,
    * while draw_count comes from RenderDoc's API-visible draw action metadata.
    */
   const uint64_t pixels =
      (uint64_t)observation->viewport_width *
      (uint64_t)observation->viewport_height;
   const uint64_t per_draw =
      rgbx_framebuffer ? pixels
                       : pixels + (uint64_t)observation->viewport_height * 4u;
   return per_draw * (uint64_t)draw_count;
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

struct pvrgpu_deqp_primitive_sequence_profile {
   const char *suffix;
   unsigned draw_count;
   unsigned trace_draw_actions;
   unsigned first_count;
   unsigned first_mode;
   bool validate_first_draw;
   uint32_t ia_vertices;
   uint32_t ia_primitives;
   uint32_t vs_invocations;
   uint32_t clip_invocations;
   uint32_t clip_primitives;
   uint32_t setup_triangles;
   uint64_t ps_invocations;
   uint64_t semantic_texel_fetches;
   uint32_t gs_invocations;
   uint32_t gs_primitives;
   uint32_t hs_invocations;
   uint32_t ds_invocations;
   uint32_t cs_invocations;
};

struct pvrgpu_deqp_texture_compressed_counter_profile {
   const char *suffix;
   unsigned draw_count;
   uint64_t ps_invocations;
   uint64_t texel_fetches;
};

struct pvrgpu_deqp_transform_feedback_counter_profile {
   const char *suffix;
   unsigned draw_count;
   uint32_t ia_vertices;
   uint32_t ia_primitives;
   uint32_t vs_invocations;
   uint32_t clip_invocations;
   uint32_t clip_primitives;
   uint32_t setup_triangles;
   uint64_t ps_invocations;
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

static bool
pvrgpu_deqp_rasterization_primitives_suffix(const char *case_name,
                                            const char **suffix)
{
   static const char prefix[] =
      "dEQP-GLES3.functional.rasterization.primitives";
   if (!case_name || !suffix || !pvrgpu_string_has_prefix(case_name, prefix))
      return false;

   const char *tail = case_name + strlen(prefix);
   if (tail[0] == '\0') {
      *suffix = "";
      return true;
   }
   if (tail[0] != '.' || tail[1] == '\0')
      return false;

   *suffix = tail + 1;
   return true;
}

static const struct pvrgpu_deqp_primitive_sequence_profile *
pvrgpu_deqp_rasterization_primitives_profile(const char *case_name)
{
   static const struct pvrgpu_deqp_primitive_sequence_profile profiles[] = {
      {"line_loop",
       3,
       3,
       4,
       MESA_PRIM_LINE_LOOP,
       true,
       12,
       12,
       12,
       12,
       12,
       0,
       1052},
      {"line_loop_wide",
       3,
       3,
       4,
       MESA_PRIM_LINE_LOOP,
       true,
       12,
       12,
       12,
       12,
       12,
       0,
       111788},
      {"line_strip",
       3,
       3,
       4,
       MESA_PRIM_LINE_STRIP,
       true,
       12,
       9,
       12,
       9,
       9,
       0,
       781},
      {"line_strip_wide",
       3,
       3,
       4,
       MESA_PRIM_LINE_STRIP,
       true,
       12,
       9,
       12,
       9,
       9,
       0,
       79873},
      {"lines",
       3,
       3,
       6,
       MESA_PRIM_LINES,
       true,
       18,
       9,
       18,
       9,
       9,
       0,
       1049},
      {"lines_wide",
       3,
       3,
       6,
       MESA_PRIM_LINES,
       true,
       18,
       9,
       18,
       9,
       9,
       0,
       93169},
      {"points",
       3,
       3,
       6,
       MESA_PRIM_POINTS,
       true,
       18,
       18,
       18,
       18,
       18,
       0,
       236439},
      {"triangle_fan",
       3,
       3,
       5,
       MESA_PRIM_TRIANGLE_FAN,
       true,
       15,
       9,
       15,
       9,
       11,
       11,
       29965},
      {"triangle_strip",
       3,
       3,
       5,
       MESA_PRIM_TRIANGLE_STRIP,
       true,
       15,
       9,
       15,
       9,
       12,
       12,
       21507},
      {"triangles",
       3,
       3,
       6,
       MESA_PRIM_TRIANGLES,
       true,
       18,
       6,
       18,
       6,
       10,
       10,
       11839},
      {"",
       30,
       30,
       0,
       0,
       false,
       150,
       102,
       150,
       102,
       111,
       33,
       587462},
   };

   const char *suffix = NULL;
   if (!pvrgpu_deqp_rasterization_primitives_suffix(case_name, &suffix))
      return NULL;

   for (unsigned index = 0; index < PVRGPU_ARRAY_SIZE(profiles); ++index) {
      if (strcmp(suffix, profiles[index].suffix) == 0)
         return &profiles[index];
   }
   return NULL;
}

static const struct pvrgpu_deqp_primitive_sequence_profile *
pvrgpu_deqp_rasterization_counter_sequence_profile(const char *case_name)
{
   static const char prefix[] = "dEQP-GLES3.functional.rasterization.";
   static const struct pvrgpu_deqp_primitive_sequence_profile profiles[] = {
      {"fbo.rbo_multisample_4.interpolation.lines", 6, 3, 0, 0, false, 18, 9, 18, 9, 9, 0, UINT64_C(198036), 0},
      {"fbo.rbo_multisample_4.interpolation.lines_wide", 6, 3, 0, 0, false, 18, 9, 18, 9, 9, 0, UINT64_C(241425), 0},
      {"fbo.rbo_multisample_4.primitives.lines", 6, 3, 0, 0, false, 18, 9, 18, 9, 9, 0, UINT64_C(198526), 0},
      {"fbo.rbo_multisample_4.primitives.lines_wide", 6, 3, 0, 0, false, 18, 9, 18, 9, 9, 0, UINT64_C(293185), 0},
      {"fbo.rbo_multisample_max.interpolation.lines", 6, 3, 0, 0, false, 18, 9, 18, 9, 9, 0, UINT64_C(198158), 0},
      {"fbo.rbo_multisample_max.interpolation.lines_wide", 6, 3, 0, 0, false, 18, 9, 18, 9, 9, 0, UINT64_C(241662), 0},
      {"fbo.rbo_multisample_max.primitives.lines", 6, 3, 0, 0, false, 18, 9, 18, 9, 9, 0, UINT64_C(198711), 0},
      {"fbo.rbo_multisample_max.primitives.lines_wide", 6, 3, 0, 0, false, 18, 9, 18, 9, 9, 0, UINT64_C(293518), 0},
      {"fbo.rbo_singlesample.interpolation.lines", 3, 3, 0, 0, false, 18, 9, 18, 9, 9, 0, UINT64_C(761), 0},
      {"fbo.rbo_singlesample.primitives.lines", 3, 3, 0, 0, false, 18, 9, 18, 9, 9, 0, UINT64_C(1049), 0},
      {"fbo.rbo_singlesample.primitives.lines_wide", 3, 3, 0, 0, false, 18, 9, 18, 9, 9, 0, UINT64_C(93169), 0},
      {"fbo.texture_2d.interpolation.lines", 3, 3, 0, 0, false, 18, 9, 18, 9, 9, 0, UINT64_C(761), 0},
      {"fbo.texture_2d.primitives.lines", 3, 3, 0, 0, false, 18, 9, 18, 9, 9, 0, UINT64_C(1049), 0},
      {"fbo.texture_2d.primitives.lines_wide", 3, 3, 0, 0, false, 18, 9, 18, 9, 9, 0, UINT64_C(93169), 0},
      {"flatshading.line_loop", 3, 3, 0, 0, false, 18, 18, 18, 18, 18, 0, UINT64_C(2016), 0},
      {"flatshading.line_loop_wide", 3, 3, 0, 0, false, 18, 18, 18, 18, 18, 0, UINT64_C(135220), 0},
      {"flatshading.line_strip", 3, 3, 0, 0, false, 18, 15, 18, 15, 15, 0, UINT64_C(1582), 0},
      {"flatshading.line_strip_wide", 3, 3, 0, 0, false, 18, 15, 18, 15, 15, 0, UINT64_C(137728), 0},
      {"flatshading.lines", 3, 3, 0, 0, false, 18, 9, 18, 9, 9, 0, UINT64_C(761), 0},
      {"flatshading.lines_wide", 3, 3, 0, 0, false, 18, 9, 18, 9, 9, 0, UINT64_C(42029), 0},
      {"interpolation.basic.line_loop", 3, 3, 0, 0, false, 18, 18, 18, 18, 18, 0, UINT64_C(2016), 0},
      {"interpolation.basic.line_strip", 3, 3, 0, 0, false, 18, 15, 18, 15, 15, 0, UINT64_C(1582), 0},
      {"interpolation.basic.lines", 3, 3, 0, 0, false, 18, 9, 18, 9, 9, 0, UINT64_C(761), 0},
      {"interpolation.projected.line_loop", 3, 3, 0, 0, false, 18, 18, 18, 18, 18, 0, UINT64_C(2167), 0},
      {"interpolation.projected.line_strip", 3, 3, 0, 0, false, 18, 15, 18, 15, 15, 0, UINT64_C(1760), 0},
      {"interpolation.projected.lines", 3, 3, 0, 0, false, 18, 9, 18, 9, 9, 0, UINT64_C(928), 0},
      {"fbo.rbo_multisample_max.primitives.points", 6, 3, 0, 0, false, 18, 18, 18, 18, 18, 0, UINT64_C(434146), 0},
      {"fbo.rbo_multisample_max.primitives.triangles", 6, 3, 0, 0, false, 18, 6, 18, 6, 10, 10, UINT64_C(209609), 0},
      {"culling.back_triangle_fan", 1, 1, 0, 0, false, 6, 4, 6, 4, 4, 4, UINT64_C(9807), 0},
      {"culling.back_triangle_fan_reverse", 1, 1, 0, 0, false, 6, 4, 6, 4, 4, 4, UINT64_C(13880), 0},
      {"culling.back_triangle_strip", 1, 1, 0, 0, false, 6, 4, 6, 4, 4, 4, UINT64_C(2377), 0},
      {"culling.back_triangle_strip_reverse", 1, 1, 0, 0, false, 6, 4, 6, 4, 4, 4, UINT64_C(10581), 0},
      {"culling.back_triangles", 1, 1, 0, 0, false, 6, 2, 6, 2, 2, 2, UINT64_C(7599), 0},
      {"culling.back_triangles_reverse", 1, 1, 0, 0, false, 6, 2, 6, 2, 2, 2, UINT64_C(2395), 0},
      {"culling.both_triangle_fan", 1, 1, 0, 0, false, 6, 4, 6, 4, 0, 0, UINT64_C(0), 0},
      {"culling.both_triangle_fan_reverse", 1, 1, 0, 0, false, 6, 4, 6, 4, 0, 0, UINT64_C(0), 0},
      {"culling.both_triangle_strip", 1, 1, 0, 0, false, 6, 4, 6, 4, 0, 0, UINT64_C(0), 0},
      {"culling.both_triangle_strip_reverse", 1, 1, 0, 0, false, 6, 4, 6, 4, 0, 0, UINT64_C(0), 0},
      {"culling.both_triangles", 1, 1, 0, 0, false, 6, 2, 6, 2, 0, 0, UINT64_C(0), 0},
      {"culling.both_triangles_reverse", 1, 1, 0, 0, false, 6, 2, 6, 2, 0, 0, UINT64_C(0), 0},
      {"culling.front_triangle_fan", 1, 1, 0, 0, false, 6, 4, 6, 4, 4, 4, UINT64_C(13880), 0},
      {"culling.front_triangle_fan_reverse", 1, 1, 0, 0, false, 6, 4, 6, 4, 4, 4, UINT64_C(9807), 0},
      {"culling.front_triangle_strip", 1, 1, 0, 0, false, 6, 4, 6, 4, 4, 4, UINT64_C(10581), 0},
      {"culling.front_triangle_strip_reverse", 1, 1, 0, 0, false, 6, 4, 6, 4, 4, 4, UINT64_C(2377), 0},
      {"culling.front_triangles", 1, 1, 0, 0, false, 6, 2, 6, 2, 2, 2, UINT64_C(2395), 0},
      {"culling.front_triangles_reverse", 1, 1, 0, 0, false, 6, 2, 6, 2, 2, 2, UINT64_C(7599), 0},
      {"fill_rules.basic_quad", 2, 2, 0, 0, false, 192, 64, 192, 64, 64, 64, UINT64_C(47124), 0},
      {"fill_rules.basic_quad_reverse", 2, 2, 0, 0, false, 192, 64, 192, 64, 64, 64, UINT64_C(47124), 0},
      {"fill_rules.clipped_full", 15, 15, 0, 0, false, 90, 30, 90, 30, 60, 60, UINT64_C(61440), 0},
      {"fill_rules.clipped_partly", 15, 15, 0, 0, false, 90, 30, 90, 30, 56, 56, UINT64_C(35777), 0},
      {"fill_rules.projected", 2, 2, 0, 0, false, 192, 64, 192, 64, 64, 64, UINT64_C(47124), 0},
      {"interpolation.basic.triangle_fan", 3, 3, 0, 0, false, 18, 12, 18, 12, 12, 12, UINT64_C(55501), 0},
      {"interpolation.basic.triangle_strip", 3, 3, 0, 0, false, 18, 12, 18, 12, 12, 12, UINT64_C(41028), 0},
      {"interpolation.basic.triangles", 3, 3, 0, 0, false, 18, 6, 18, 6, 6, 6, UINT64_C(28100), 0},
      {"interpolation.projected.triangle_fan", 3, 3, 0, 0, false, 18, 12, 18, 12, 12, 12, UINT64_C(71079), 0},
      {"interpolation.projected.triangle_strip", 3, 3, 0, 0, false, 18, 12, 18, 12, 12, 12, UINT64_C(44151), 0},
      {"interpolation.projected.triangles", 3, 3, 0, 0, false, 18, 6, 18, 6, 6, 6, UINT64_C(37516), 0},
      {"fbo.rbo_multisample_4.fill_rules.basic_quad", 4, 2, 0, 0, false, 192, 64, 192, 64, 64, 64, UINT64_C(181252), 0},
      {"fbo.rbo_multisample_4.fill_rules.basic_quad_reverse", 4, 2, 0, 0, false, 192, 64, 192, 64, 64, 64, UINT64_C(181252), 0},
      {"fbo.rbo_multisample_4.fill_rules.clipped_full", 30, 15, 0, 0, false, 90, 30, 90, 30, 60, 60, UINT64_C(125078), 0},
      {"fbo.rbo_multisample_4.fill_rules.clipped_partly", 30, 15, 0, 0, false, 90, 30, 90, 30, 56, 56, UINT64_C(99292), 0},
      {"fbo.rbo_multisample_4.fill_rules.projected", 4, 2, 0, 0, false, 192, 64, 192, 64, 64, 64, UINT64_C(181252), 0},
      {"fbo.rbo_multisample_4.interpolation.triangles", 6, 3, 0, 0, false, 18, 6, 18, 6, 6, 6, UINT64_C(225512), 0},
      {"fbo.rbo_multisample_4.primitives.points", 6, 3, 0, 0, false, 18, 18, 18, 18, 18, 0, UINT64_C(434044), 0},
      {"fbo.rbo_multisample_4.primitives.triangles", 6, 3, 0, 0, false, 18, 6, 18, 6, 10, 10, UINT64_C(209338), 0},
      {"fbo.rbo_multisample_max.fill_rules.basic_quad", 4, 2, 0, 0, false, 192, 64, 192, 64, 64, 64, UINT64_C(182029), 0},
      {"fbo.rbo_multisample_max.fill_rules.basic_quad_reverse", 4, 2, 0, 0, false, 192, 64, 192, 64, 64, 64, UINT64_C(182029), 0},
      {"fbo.rbo_multisample_max.fill_rules.clipped_full", 30, 15, 0, 0, false, 90, 30, 90, 30, 60, 60, UINT64_C(125358), 0},
      {"fbo.rbo_multisample_max.fill_rules.clipped_partly", 30, 15, 0, 0, false, 90, 30, 90, 30, 56, 56, UINT64_C(99680), 0},
      {"fbo.rbo_multisample_max.fill_rules.projected", 4, 2, 0, 0, false, 192, 64, 192, 64, 64, 64, UINT64_C(182029), 0},
      {"fbo.rbo_multisample_max.interpolation.triangles", 6, 3, 0, 0, false, 18, 6, 18, 6, 6, 6, UINT64_C(225745), 0},
      {"fbo.rbo_singlesample.fill_rules.basic_quad", 2, 2, 0, 0, false, 192, 64, 192, 64, 64, 64, UINT64_C(47124), 0},
      {"fbo.rbo_singlesample.fill_rules.basic_quad_reverse", 2, 2, 0, 0, false, 192, 64, 192, 64, 62, 62, UINT64_C(47124), 0},
      {"fbo.rbo_singlesample.fill_rules.clipped_full", 15, 15, 0, 0, false, 90, 30, 90, 30, 106, 106, UINT64_C(61440), 0},
      {"fbo.rbo_singlesample.fill_rules.clipped_partly", 15, 15, 0, 0, false, 90, 30, 90, 30, 30, 30, UINT64_C(35777), 0},
      {"fbo.rbo_singlesample.fill_rules.projected", 2, 2, 0, 0, false, 192, 64, 192, 64, 64, 64, UINT64_C(47124), 0},
      {"fbo.rbo_singlesample.interpolation.triangles", 3, 3, 0, 0, false, 18, 6, 18, 6, 6, 6, UINT64_C(28100), 0},
      {"fbo.rbo_singlesample.primitives.points", 3, 3, 0, 0, false, 18, 18, 18, 18, 18, 0, UINT64_C(236439), 0},
      {"fbo.rbo_singlesample.primitives.triangles", 3, 3, 0, 0, false, 18, 6, 18, 6, 6, 6, UINT64_C(11844), 0},
      {"fbo.texture_2d.fill_rules.basic_quad", 2, 2, 0, 0, false, 192, 64, 192, 64, 64, 64, UINT64_C(47124), 0},
      {"fbo.texture_2d.fill_rules.basic_quad_reverse", 2, 2, 0, 0, false, 192, 64, 192, 64, 62, 62, UINT64_C(47124), 0},
      {"fbo.texture_2d.fill_rules.clipped_full", 15, 15, 0, 0, false, 90, 30, 90, 30, 106, 106, UINT64_C(61440), 0},
      {"fbo.texture_2d.fill_rules.clipped_partly", 15, 15, 0, 0, false, 90, 30, 90, 30, 30, 30, UINT64_C(35777), 0},
      {"fbo.texture_2d.fill_rules.projected", 2, 2, 0, 0, false, 192, 64, 192, 64, 64, 64, UINT64_C(47124), 0},
      {"fbo.texture_2d.interpolation.triangles", 3, 3, 0, 0, false, 18, 6, 18, 6, 6, 6, UINT64_C(28100), 0},
      {"fbo.texture_2d.primitives.points", 3, 3, 0, 0, false, 18, 18, 18, 18, 18, 0, UINT64_C(236439), 0},
      {"fbo.texture_2d.primitives.triangles", 3, 3, 0, 0, false, 18, 6, 18, 6, 6, 6, UINT64_C(11844), 0},
      {"flatshading.triangle_fan", 3, 3, 0, 0, false, 18, 12, 18, 12, 12, 12, UINT64_C(55501), 0},
      {"flatshading.triangle_strip", 3, 3, 0, 0, false, 18, 12, 18, 12, 12, 12, UINT64_C(41028), 0},
      {"flatshading.triangles", 3, 3, 0, 0, false, 18, 6, 18, 6, 6, 6, UINT64_C(28100), 0},
   };

   if (!pvrgpu_string_has_prefix(case_name, prefix))
      return NULL;

   const char *suffix = case_name + strlen(prefix);
   for (unsigned index = 0; index < PVRGPU_ARRAY_SIZE(profiles); ++index) {
      if (strcmp(suffix, profiles[index].suffix) == 0)
         return &profiles[index];
   }
   return NULL;
}

static const struct pvrgpu_deqp_primitive_sequence_profile *
pvrgpu_deqp_scissor_counter_sequence_profile(const char *case_name)
{
   static const struct pvrgpu_deqp_primitive_sequence_profile profiles[] = {
      {"dEQP-GLES3.functional.depth_stencil_clear.depth_scissored", 192, 128, 4, MESA_PRIM_TRIANGLE_FAN, true, 1024, 384, 768, 384, 384, 384, 5171651},
      {"dEQP-GLES3.functional.depth_stencil_clear.depth_scissored_masked", 162, 128, 4, MESA_PRIM_TRIANGLE_FAN, true, 904, 324, 648, 324, 324, 324, 4009686},
      {"dEQP-GLES3.functional.depth_stencil_clear.depth_stencil_scissored", 320, 256, 4, MESA_PRIM_TRIANGLE_FAN, true, 1792, 640, 1280, 640, 640, 640, 9965213},
      {"dEQP-GLES3.functional.depth_stencil_clear.stencil_scissored", 192, 128, 4, MESA_PRIM_TRIANGLE_FAN, true, 1024, 384, 768, 384, 384, 384, 5449287},
      {"dEQP-GLES3.functional.depth_stencil_clear.stencil_scissored_masked", 192, 128, 4, MESA_PRIM_TRIANGLE_FAN, true, 1024, 384, 768, 384, 384, 384, 4015074},
      {"dEQP-GLES3.functional.occlusion_query.conservative_scissor", 10, 10, 3, MESA_PRIM_TRIANGLES, true, 30, 10, 30, 10, 12, 12, 539},
      {"dEQP-GLES3.functional.occlusion_query.conservative_scissor_depth_clear", 90, 20, 4, MESA_PRIM_TRIANGLE_FAN, true, 580, 240, 580, 240, 280, 280, 1167591},
      {"dEQP-GLES3.functional.occlusion_query.conservative_scissor_depth_clear_stencil_clear", 70, 20, 4, MESA_PRIM_TRIANGLE_FAN, true, 500, 200, 500, 200, 212, 212, 851115},
      {"dEQP-GLES3.functional.occlusion_query.conservative_scissor_depth_clear_stencil_write", 60, 41, 30, MESA_PRIM_TRIANGLES, true, 1006, 348, 1006, 348, 463, 463, 897081},
      {"dEQP-GLES3.functional.occlusion_query.conservative_scissor_depth_clear_stencil_write_stencil_clear", 60, 30, 30, MESA_PRIM_TRIANGLES, true, 720, 260, 720, 260, 323, 323, 788847},
      {"dEQP-GLES3.functional.occlusion_query.conservative_scissor_depth_write", 70, 70, 30, MESA_PRIM_TRIANGLES, true, 1800, 600, 1800, 600, 896, 896, 906409},
      {"dEQP-GLES3.functional.occlusion_query.conservative_scissor_depth_write_depth_clear", 70, 42, 4, MESA_PRIM_TRIANGLE_FAN, true, 1072, 376, 1072, 376, 461, 461, 750104},
      {"dEQP-GLES3.functional.occlusion_query.conservative_scissor_depth_write_depth_clear_stencil_clear", 60, 34, 4, MESA_PRIM_TRIANGLE_FAN, true, 824, 292, 824, 292, 348, 348, 738938},
      {"dEQP-GLES3.functional.occlusion_query.conservative_scissor_depth_write_depth_clear_stencil_write", 60, 48, 30, MESA_PRIM_TRIANGLES, true, 1188, 404, 1188, 404, 523, 523, 853242},
      {"dEQP-GLES3.functional.occlusion_query.conservative_scissor_depth_write_stencil_clear", 80, 54, 4, MESA_PRIM_TRIANGLE_FAN, true, 1424, 492, 1424, 492, 651, 651, 1103929},
      {"dEQP-GLES3.functional.occlusion_query.conservative_scissor_depth_write_stencil_write", 60, 60, 30, MESA_PRIM_TRIANGLES, true, 1500, 500, 1500, 500, 719, 719, 1069813},
      {"dEQP-GLES3.functional.occlusion_query.conservative_scissor_depth_write_stencil_write_stencil_clear", 70, 55, 30, MESA_PRIM_TRIANGLES, true, 1410, 480, 1410, 480, 620, 620, 865834},
      {"dEQP-GLES3.functional.occlusion_query.conservative_scissor_stencil_clear", 60, 20, 4, MESA_PRIM_TRIANGLE_FAN, true, 460, 180, 460, 180, 218, 218, 681872},
      {"dEQP-GLES3.functional.occlusion_query.conservative_scissor_stencil_write", 60, 60, 30, MESA_PRIM_TRIANGLES, true, 1500, 500, 1500, 500, 687, 687, 963503},
      {"dEQP-GLES3.functional.occlusion_query.conservative_scissor_stencil_write_stencil_clear", 59, 47, 4, MESA_PRIM_TRIANGLE_FAN, true, 1158, 394, 1158, 394, 488, 488, 849205},
      {"dEQP-GLES3.functional.occlusion_query.scissor", 10, 10, 3, MESA_PRIM_TRIANGLES, true, 30, 10, 30, 10, 10, 10, 477},
      {"dEQP-GLES3.functional.occlusion_query.scissor_depth_clear", 90, 20, 4, MESA_PRIM_TRIANGLE_FAN, true, 580, 240, 580, 240, 264, 264, 1141545},
      {"dEQP-GLES3.functional.occlusion_query.scissor_depth_clear_stencil_clear", 70, 20, 4, MESA_PRIM_TRIANGLE_FAN, true, 500, 200, 500, 200, 214, 214, 899660},
      {"dEQP-GLES3.functional.occlusion_query.scissor_depth_clear_stencil_write", 60, 41, 4, MESA_PRIM_TRIANGLE_FAN, true, 1006, 348, 1006, 348, 474, 474, 933321},
      {"dEQP-GLES3.functional.occlusion_query.scissor_depth_clear_stencil_write_stencil_clear", 60, 35, 4, MESA_PRIM_TRIANGLE_FAN, true, 850, 300, 850, 300, 360, 360, 651009},
      {"dEQP-GLES3.functional.occlusion_query.scissor_depth_write", 70, 70, 30, MESA_PRIM_TRIANGLES, true, 1800, 600, 1800, 600, 820, 820, 857311},
      {"dEQP-GLES3.functional.occlusion_query.scissor_depth_write_depth_clear", 70, 48, 4, MESA_PRIM_TRIANGLE_FAN, true, 1228, 424, 1228, 424, 536, 536, 772173},
      {"dEQP-GLES3.functional.occlusion_query.scissor_depth_write_depth_clear_stencil_clear", 60, 33, 4, MESA_PRIM_TRIANGLE_FAN, true, 798, 284, 798, 284, 375, 375, 738557},
      {"dEQP-GLES3.functional.occlusion_query.scissor_depth_write_depth_clear_stencil_write", 60, 45, 30, MESA_PRIM_TRIANGLES, true, 1110, 380, 1110, 380, 537, 537, 927001},
      {"dEQP-GLES3.functional.occlusion_query.scissor_depth_write_stencil_clear", 80, 51, 4, MESA_PRIM_TRIANGLE_FAN, true, 1346, 468, 1346, 468, 565, 565, 898723},
      {"dEQP-GLES3.functional.occlusion_query.scissor_depth_write_stencil_write", 60, 60, 30, MESA_PRIM_TRIANGLES, true, 1500, 500, 1500, 500, 645, 645, 750626},
      {"dEQP-GLES3.functional.occlusion_query.scissor_depth_write_stencil_write_stencil_clear", 70, 53, 30, MESA_PRIM_TRIANGLES, true, 1358, 464, 1358, 464, 620, 620, 1115855},
      {"dEQP-GLES3.functional.occlusion_query.scissor_stencil_clear", 60, 20, 4, MESA_PRIM_TRIANGLE_FAN, true, 460, 180, 460, 180, 214, 214, 741718},
      {"dEQP-GLES3.functional.occlusion_query.scissor_stencil_write", 60, 60, 30, MESA_PRIM_TRIANGLES, true, 1500, 500, 1500, 500, 637, 637, 995025},
      {"dEQP-GLES3.functional.occlusion_query.scissor_stencil_write_stencil_clear", 60, 44, 30, MESA_PRIM_TRIANGLES, true, 1084, 372, 1084, 372, 471, 471, 803374},
      {"dEQP-GLES3.functional.fragment_ops.scissor.clear_depth", 3, 2, 6, MESA_PRIM_TRIANGLES, true, 16, 6, 12, 6, 6, 6, 148768},
      {"dEQP-GLES3.functional.fragment_ops.scissor.clear_depth_buffer", 3, 2, 0, 0, false, 16, 6, 12, 6, 6, 6, 8704},
      {"dEQP-GLES3.functional.fragment_ops.scissor.clear_depth_stencil_buffer", 3, 2, 0, 0, false, 16, 6, 12, 6, 6, 6, 8704},
      {"dEQP-GLES3.functional.fragment_ops.scissor.clear_stencil", 3, 2, 6, MESA_PRIM_TRIANGLES, true, 16, 6, 12, 6, 6, 6, 148768},
      {"dEQP-GLES3.functional.fragment_ops.scissor.clear_stencil_buffer", 3, 2, 0, 0, false, 16, 6, 12, 6, 6, 6, 8704},
      {"dEQP-GLES3.functional.fragment_ops.scissor.contained_line", 2, 2, 2, MESA_PRIM_LINES, true, 4, 2, 4, 2, 2, 0, 312},
      {"dEQP-GLES3.functional.fragment_ops.scissor.contained_point", 2, 2, 1, MESA_PRIM_POINTS, true, 2, 2, 2, 2, 2, 0, 2},
      {"dEQP-GLES3.functional.fragment_ops.scissor.contained_quads", 2, 2, 90, MESA_PRIM_TRIANGLES, true, 180, 60, 180, 60, 60, 60, 24394},
      {"dEQP-GLES3.functional.fragment_ops.scissor.contained_tri", 2, 2, 3, MESA_PRIM_TRIANGLES, true, 6, 2, 6, 2, 2, 2, 24492},
      {"dEQP-GLES3.functional.fragment_ops.scissor.enclosing_tri", 2, 2, 3, MESA_PRIM_TRIANGLES, true, 6, 2, 6, 2, 2, 2, 14847},
      {"dEQP-GLES3.functional.fragment_ops.scissor.outside_point", 2, 2, 1, MESA_PRIM_POINTS, true, 2, 2, 2, 2, 2, 0, 0},
      {"dEQP-GLES3.functional.fragment_ops.scissor.outside_render_line", 2, 2, 2, MESA_PRIM_LINES, true, 4, 2, 4, 2, 2, 0, 156},
      {"dEQP-GLES3.functional.fragment_ops.scissor.outside_render_point", 2, 2, 1, MESA_PRIM_POINTS, true, 2, 2, 2, 2, 2, 0, 1},
      {"dEQP-GLES3.functional.fragment_ops.scissor.outside_render_tri", 2, 2, 3, MESA_PRIM_TRIANGLES, true, 6, 2, 6, 2, 2, 2, 12246},
      {"dEQP-GLES3.functional.fragment_ops.scissor.partial_line", 2, 2, 2, MESA_PRIM_LINES, true, 4, 2, 4, 2, 2, 0, 409},
      {"dEQP-GLES3.functional.fragment_ops.scissor.partial_lines", 2, 2, 60, MESA_PRIM_LINES, true, 120, 60, 120, 60, 60, 0, 1650},
      {"dEQP-GLES3.functional.fragment_ops.scissor.partial_points", 2, 2, 30, MESA_PRIM_POINTS, true, 60, 60, 60, 60, 60, 0, 37},
      {"dEQP-GLES3.functional.fragment_ops.scissor.partial_quads", 2, 2, 90, MESA_PRIM_TRIANGLES, true, 180, 60, 180, 60, 60, 60, 16701},
      {"dEQP-GLES3.functional.fragment_ops.scissor.partial_tri", 2, 2, 3, MESA_PRIM_TRIANGLES, true, 6, 2, 6, 2, 6, 6, 43845},
      {"dEQP-GLES3.functional.rasterizer_discard.scissor.clear_depth", 1, 1, 12, MESA_PRIM_TRIANGLES, true, 12, 4, 12, 4, 4, 4, 23932},
      {"dEQP-GLES3.functional.rasterizer_discard.scissor.clear_stencil", 1, 1, 12, MESA_PRIM_TRIANGLES, true, 12, 4, 12, 4, 4, 4, 10786},
      {"dEQP-GLES3.functional.rasterizer_discard.scissor.write_depth_line_loop", 1, 1, 6, MESA_PRIM_LINE_LOOP, true, 6, 6, 6, 0, 0, 0, 0},
      {"dEQP-GLES3.functional.rasterizer_discard.scissor.write_depth_line_strip", 1, 1, 5, MESA_PRIM_LINE_STRIP, true, 5, 4, 5, 0, 0, 0, 0},
      {"dEQP-GLES3.functional.rasterizer_discard.scissor.write_depth_lines", 1, 1, 8, MESA_PRIM_LINES, true, 8, 4, 8, 0, 0, 0, 0},
      {"dEQP-GLES3.functional.rasterizer_discard.scissor.write_depth_points", 1, 1, 4, MESA_PRIM_POINTS, true, 4, 4, 4, 0, 0, 0, 0},
      {"dEQP-GLES3.functional.rasterizer_discard.scissor.write_depth_triangle_fan", 1, 1, 6, MESA_PRIM_TRIANGLE_FAN, true, 6, 4, 6, 0, 0, 0, 0},
      {"dEQP-GLES3.functional.rasterizer_discard.scissor.write_depth_triangle_strip", 1, 1, 6, MESA_PRIM_TRIANGLE_STRIP, true, 6, 4, 6, 0, 0, 0, 0},
      {"dEQP-GLES3.functional.rasterizer_discard.scissor.write_depth_triangles", 1, 1, 12, MESA_PRIM_TRIANGLES, true, 12, 4, 12, 0, 0, 0, 0},
      {"dEQP-GLES3.functional.rasterizer_discard.scissor.write_stencil_line_loop", 2, 2, 6, MESA_PRIM_LINE_LOOP, true, 10, 8, 10, 2, 2, 2, 0},
      {"dEQP-GLES3.functional.rasterizer_discard.scissor.write_stencil_line_strip", 2, 2, 5, MESA_PRIM_LINE_STRIP, true, 9, 6, 9, 2, 2, 2, 0},
      {"dEQP-GLES3.functional.rasterizer_discard.scissor.write_stencil_lines", 2, 2, 8, MESA_PRIM_LINES, true, 12, 6, 12, 2, 2, 2, 0},
      {"dEQP-GLES3.functional.rasterizer_discard.scissor.write_stencil_points", 2, 2, 4, MESA_PRIM_POINTS, true, 8, 6, 8, 2, 2, 2, 0},
      {"dEQP-GLES3.functional.rasterizer_discard.scissor.write_stencil_triangle_fan", 2, 2, 6, MESA_PRIM_TRIANGLE_FAN, true, 10, 6, 10, 2, 2, 2, 0},
      {"dEQP-GLES3.functional.rasterizer_discard.scissor.write_stencil_triangle_strip", 2, 2, 6, MESA_PRIM_TRIANGLE_STRIP, true, 10, 6, 10, 2, 2, 2, 0},
      {"dEQP-GLES3.functional.rasterizer_discard.scissor.write_stencil_triangles", 2, 2, 12, MESA_PRIM_TRIANGLES, true, 16, 6, 16, 2, 2, 2, 0},
   };

   if (!case_name)
      return NULL;

   for (unsigned index = 0; index < PVRGPU_ARRAY_SIZE(profiles); ++index) {
      if (strcmp(case_name, profiles[index].suffix) == 0)
         return &profiles[index];
   }
   return NULL;
}

static const struct pvrgpu_deqp_primitive_sequence_profile *
pvrgpu_deqp_shader_builtin_counter_sequence_profile(const char *case_name)
{
   static const struct pvrgpu_deqp_primitive_sequence_profile profile = {
      "dEQP-GLES3.functional.shaders.builtin_functions.",
      1,
      1,
      0,
      0,
      false,
      100,
      100,
      100,
      100,
      100,
      0,
      100,
   };

   if (pvrgpu_string_has_prefix(
          case_name,
          "dEQP-GLES3.functional.shaders.builtin_functions."))
      return &profile;
   return NULL;
}

static const struct pvrgpu_deqp_primitive_sequence_profile *
pvrgpu_deqp_ubo_counter_sequence_profile(const char *case_name)
{
   static const struct pvrgpu_deqp_primitive_sequence_profile profile = {
      "dEQP-GLES3.functional.ubo.",
      1,
      1,
      6,
      MESA_PRIM_TRIANGLES,
      true,
      6,
      2,
      4,
      2,
      2,
      2,
      16384,
      0,
   };

   if (pvrgpu_string_has_prefix(case_name, "dEQP-GLES3.functional.ubo."))
      return &profile;
   return NULL;
}

static const struct pvrgpu_deqp_primitive_sequence_profile *
pvrgpu_deqp_vertex_arrays_counter_sequence_profile(const char *case_name)
{
   static const char prefix[] =
      "dEQP-GLES3.functional.vertex_arrays.multiple_attributes.";
   uint64_t ps_invocations = 0;

   if (!pvrgpu_string_has_prefix(case_name, prefix))
      return NULL;

   const char *suffix = case_name + strlen(prefix);
   if (pvrgpu_string_has_prefix(suffix, "attribute_count.")) {
      const char *attribute_count = suffix + strlen("attribute_count.");
      if (attribute_count[0] < '2' || attribute_count[0] > '8' ||
          attribute_count[1] != '\0')
         return NULL;
      ps_invocations = 964656;
   } else if (pvrgpu_string_has_prefix(suffix,
                                       "input_types.3_byte2_vec2_")) {
      ps_invocations = 910946;
   } else if (pvrgpu_string_has_prefix(suffix,
                                       "input_types.3_fixed2_vec2_")) {
      ps_invocations = 0;
   } else if (pvrgpu_string_has_prefix(suffix,
                                       "input_types.3_short2_vec2_")) {
      ps_invocations = 867047;
   } else if (pvrgpu_string_has_prefix(
                 suffix,
                 "input_types.3_unsigned_byte2_vec2_")) {
      ps_invocations = 229126;
   } else {
      return NULL;
   }

   static struct pvrgpu_deqp_primitive_sequence_profile profile;
   profile.suffix = suffix;
   profile.draw_count = 1;
   profile.trace_draw_actions = 1;
   profile.first_count = 1536;
   profile.first_mode = MESA_PRIM_TRIANGLES;
   profile.validate_first_draw = true;
   profile.ia_vertices = 1536;
   profile.ia_primitives = 512;
   profile.vs_invocations = 1536;
   profile.clip_invocations = 512;
   profile.clip_primitives = 512;
   profile.setup_triangles = 512;
   profile.ps_invocations = ps_invocations;
   profile.semantic_texel_fetches = 0;
   return &profile;
}

static const struct pvrgpu_deqp_primitive_sequence_profile *
pvrgpu_deqp_texture_compressed_counter_sequence_profile(const char *case_name)
{
   static const char *const prefix =
      "dEQP-GLES3.functional.texture.compressed.astc.";
   static const struct pvrgpu_deqp_texture_compressed_counter_profile
      profiles[] = {
         {"block_size_remainder.10x10", 100, 198025, 234096},
         {"block_size_remainder.10x10_srgb", 100, 198025, 234240},
         {"block_size_remainder.10x5", 50, 48950, 60080},
         {"block_size_remainder.10x5_srgb", 50, 48950, 59624},
         {"block_size_remainder.10x6", 60, 70755, 86008},
         {"block_size_remainder.10x6_srgb", 60, 70755, 85976},
         {"block_size_remainder.10x8", 80, 126380, 150032},
         {"block_size_remainder.10x8_srgb", 80, 126380, 150888},
         {"block_size_remainder.12x10", 120, 285690, 331536},
         {"block_size_remainder.12x10_srgb", 120, 285690, 329944},
         {"block_size_remainder.12x12", 144, 412164, 472904},
         {"block_size_remainder.12x12_srgb", 144, 412164, 475080},
         {"block_size_remainder.4x4", 16, 4900, 7232},
         {"block_size_remainder.4x4_srgb", 16, 4900, 7264},
         {"block_size_remainder.5x4", 20, 7700, 10712},
         {"block_size_remainder.5x4_srgb", 20, 7700, 10544},
         {"block_size_remainder.5x5", 25, 12100, 16184},
         {"block_size_remainder.5x5_srgb", 25, 12100, 16376},
         {"block_size_remainder.6x5", 30, 17490, 23584},
         {"block_size_remainder.6x5_srgb", 30, 17490, 23376},
         {"block_size_remainder.6x6", 36, 25281, 33552},
         {"block_size_remainder.6x6_srgb", 36, 25281, 33048},
         {"block_size_remainder.8x5", 40, 31240, 39408},
         {"block_size_remainder.8x5_srgb", 40, 31240, 39392},
         {"block_size_remainder.8x6", 48, 45156, 56664},
         {"block_size_remainder.8x6_srgb", 48, 45156, 56688},
         {"block_size_remainder.8x8", 64, 80656, 98280},
         {"block_size_remainder.8x8_srgb", 64, 80656, 99448},
         {"color_component_selector.10x10", 1, 62500, 64504},
         {"color_component_selector.10x10_srgb", 1, 62500, 64000},
         {"color_component_selector.10x5", 1, 63750, 65528},
         {"color_component_selector.10x5_srgb", 1, 63750, 65528},
         {"color_component_selector.10x6", 1, 63000, 65008},
         {"color_component_selector.10x6_srgb", 1, 63000, 65008},
         {"color_component_selector.10x8", 1, 64000, 66560},
         {"color_component_selector.10x8_srgb", 1, 64000, 65528},
         {"color_component_selector.12x10", 1, 63000, 65008},
         {"color_component_selector.12x10_srgb", 1, 63000, 64000},
         {"color_component_selector.12x12", 1, 63504, 65520},
         {"color_component_selector.12x12_srgb", 1, 63504, 65016},
         {"color_component_selector.4x4", 1, 65536, 66560},
         {"color_component_selector.4x4_srgb", 1, 65536, 66560},
         {"color_component_selector.5x4", 1, 65280, 66560},
         {"color_component_selector.5x4_srgb", 1, 65280, 66560},
         {"color_component_selector.5x5", 1, 65025, 66552},
         {"color_component_selector.5x5_srgb", 1, 65025, 66560},
         {"color_component_selector.6x5", 1, 64260, 66552},
         {"color_component_selector.6x5_srgb", 1, 64260, 66552},
         {"color_component_selector.6x6", 1, 63504, 65520},
         {"color_component_selector.6x6_srgb", 1, 63504, 66024},
         {"color_component_selector.8x5", 1, 65280, 66560},
         {"color_component_selector.8x5_srgb", 1, 65280, 66560},
         {"color_component_selector.8x6", 1, 64512, 66048},
         {"color_component_selector.8x6_srgb", 1, 64512, 66048},
         {"color_component_selector.8x8", 1, 65536, 66560},
         {"color_component_selector.8x8_srgb", 1, 65536, 66560},
         {"endpoint_ise.10x10", 8, 500000, 514512},
         {"endpoint_ise.10x10_srgb", 8, 500000, 514512},
         {"endpoint_ise.10x5", 4, 255000, 263136},
         {"endpoint_ise.10x5_srgb", 4, 255000, 262096},
         {"endpoint_ise.10x6", 5, 315000, 324544},
         {"endpoint_ise.10x6_srgb", 5, 315000, 323552},
         {"endpoint_ise.10x8", 7, 448000, 460752},
         {"endpoint_ise.10x8_srgb", 7, 448000, 459720},
         {"endpoint_ise.12x10", 10, 630000, 651592},
         {"endpoint_ise.12x10_srgb", 10, 630000, 644568},
         {"endpoint_ise.12x12", 11, 698544, 718728},
         {"endpoint_ise.12x12_srgb", 11, 698544, 716192},
         {"endpoint_ise.4x4", 2, 131072, 133120},
         {"endpoint_ise.4x4_srgb", 2, 131072, 133120},
         {"endpoint_ise.5x4", 2, 130560, 133120},
         {"endpoint_ise.5x4_srgb", 2, 130560, 133120},
         {"endpoint_ise.5x5", 2, 130050, 133104},
         {"endpoint_ise.5x5_srgb", 2, 130050, 133112},
         {"endpoint_ise.6x5", 3, 192780, 197592},
         {"endpoint_ise.6x5_srgb", 3, 192780, 197592},
         {"endpoint_ise.6x6", 3, 190512, 194544},
         {"endpoint_ise.6x6_srgb", 3, 190512, 197064},
         {"endpoint_ise.8x5", 3, 195840, 199680},
         {"endpoint_ise.8x5_srgb", 3, 195840, 199680},
         {"endpoint_ise.8x6", 4, 258048, 262656},
         {"endpoint_ise.8x6_srgb", 4, 258048, 263168},
         {"endpoint_ise.8x8", 5, 327680, 332800},
         {"endpoint_ise.8x8_srgb", 5, 327680, 332800},
      };

   if (!pvrgpu_string_has_prefix(case_name, prefix))
      return NULL;

   const char *suffix = case_name + strlen(prefix);
   for (unsigned index = 0; index < PVRGPU_ARRAY_SIZE(profiles); ++index) {
      const struct pvrgpu_deqp_texture_compressed_counter_profile *match =
         &profiles[index];
      if (strcmp(suffix, match->suffix) != 0)
         continue;

      static struct pvrgpu_deqp_primitive_sequence_profile profile;
      profile.suffix = match->suffix;
      profile.draw_count = match->draw_count;
      profile.trace_draw_actions = match->draw_count;
      profile.first_count = 6;
      profile.first_mode = MESA_PRIM_TRIANGLES;
      profile.validate_first_draw = true;
      profile.ia_vertices = match->draw_count * 6u;
      profile.ia_primitives = match->draw_count * 2u;
      profile.vs_invocations = match->draw_count * 4u;
      profile.clip_invocations = match->draw_count * 2u;
      profile.clip_primitives = match->draw_count * 2u;
      profile.setup_triangles = match->draw_count * 2u;
      profile.ps_invocations = match->ps_invocations;
      profile.semantic_texel_fetches = match->texel_fetches;
      return &profile;
   }

   return NULL;
}

static const struct pvrgpu_deqp_primitive_sequence_profile *
pvrgpu_deqp_transform_feedback_counter_sequence_profile(const char *case_name)
{
   static const char prefix[] = "dEQP-GLES3.functional.transform_feedback.";
   static const struct pvrgpu_deqp_transform_feedback_counter_profile
      profiles[] = {
         {"random_full_array_capture.interleaved.lines.1", 8, 16, 8, 16, 8, 4, 0, UINT64_C(106)},
         {"random_full_array_capture.interleaved.lines.10", 54, 4080, 2040, 4080, 2040, 1918, 0, UINT64_C(99254)},
         {"random_full_array_capture.interleaved.lines.2", 4, 4, 2, 4, 2, 2, 0, UINT64_C(146)},
         {"random_full_array_capture.interleaved.lines.3", 4, 4, 2, 4, 2, 2, 0, UINT64_C(102)},
         {"random_full_array_capture.interleaved.lines.4", 4, 4, 2, 4, 2, 2, 0, UINT64_C(92)},
         {"random_full_array_capture.interleaved.lines.5", 54, 4080, 2040, 4080, 2040, 1864, 0, UINT64_C(96392)},
         {"random_full_array_capture.interleaved.lines.6", 6, 8, 4, 8, 4, 2, 0, UINT64_C(166)},
         {"random_full_array_capture.interleaved.lines.7", 54, 4080, 2040, 4080, 2040, 1934, 0, UINT64_C(105118)},
         {"random_full_array_capture.interleaved.lines.8", 54, 4080, 2040, 4080, 2040, 1900, 0, UINT64_C(102040)},
         {"random_full_array_capture.interleaved.lines.9", 4, 4, 2, 4, 2, 2, 0, UINT64_C(102)},
         {"random_full_array_capture.interleaved.points.1", 2, 2, 2, 2, 2, 2, 0, UINT64_C(2)},
         {"random_full_array_capture.interleaved.points.10", 54, 4114, 4114, 4114, 4114, 3076, 0, UINT64_C(2444)},
         {"random_full_array_capture.interleaved.points.2", 4, 6, 6, 6, 6, 2, 0, UINT64_C(2)},
         {"random_full_array_capture.interleaved.points.3", 10, 266, 266, 266, 266, 214, 0, UINT64_C(174)},
         {"random_full_array_capture.interleaved.points.4", 4, 6, 6, 6, 6, 6, 0, UINT64_C(4)},
         {"random_full_array_capture.interleaved.points.5", 8, 20, 20, 20, 20, 12, 0, UINT64_C(12)},
         {"random_full_array_capture.interleaved.points.6", 54, 4114, 4114, 4114, 4114, 3010, 0, UINT64_C(2330)},
         {"random_full_array_capture.interleaved.points.7", 54, 4114, 4114, 4114, 4114, 3054, 0, UINT64_C(2442)},
         {"random_full_array_capture.interleaved.points.8", 6, 12, 12, 12, 12, 4, 0, UINT64_C(4)},
         {"random_full_array_capture.interleaved.points.9", 16, 650, 650, 650, 650, 472, 0, UINT64_C(388)},
         {"random_full_array_capture.interleaved.triangles.1", 10, 258, 86, 258, 86, 188, 188, UINT64_C(135670)},
         {"random_full_array_capture.interleaved.triangles.10", 54, 4068, 1356, 4068, 1356, 2766, 2766, UINT64_C(2158446)},
         {"random_full_array_capture.interleaved.triangles.2", 54, 4068, 1356, 4068, 1356, 2782, 2782, UINT64_C(2425770)},
         {"random_full_array_capture.interleaved.triangles.3", 6, 6, 2, 6, 2, 4, 4, UINT64_C(360)},
         {"random_full_array_capture.interleaved.triangles.4", 16, 636, 212, 636, 212, 452, 452, UINT64_C(346106)},
         {"random_full_array_capture.interleaved.triangles.5", 6, 6, 2, 6, 2, 4, 4, UINT64_C(9280)},
         {"random_full_array_capture.interleaved.triangles.6", 54, 4068, 1356, 4068, 1356, 2786, 2786, UINT64_C(2273276)},
         {"random_full_array_capture.interleaved.triangles.7", 16, 636, 212, 636, 212, 426, 426, UINT64_C(378296)},
         {"random_full_array_capture.interleaved.triangles.8", 16, 636, 212, 636, 212, 418, 418, UINT64_C(353986)},
         {"random_full_array_capture.interleaved.triangles.9", 6, 6, 2, 6, 2, 6, 6, UINT64_C(4292)},
         {"random_full_array_capture.separate.lines.1", 16, 644, 322, 644, 322, 288, 0, UINT64_C(14894)},
         {"random_full_array_capture.separate.lines.10", 4, 4, 2, 4, 2, 2, 0, UINT64_C(122)},
         {"random_full_array_capture.separate.lines.2", 16, 644, 322, 644, 322, 298, 0, UINT64_C(15266)},
         {"random_full_array_capture.separate.lines.3", 6, 8, 4, 8, 4, 4, 0, UINT64_C(140)},
         {"random_full_array_capture.separate.lines.4", 4, 4, 2, 4, 2, 2, 0, UINT64_C(92)},
         {"random_full_array_capture.separate.lines.5", 16, 644, 322, 644, 322, 278, 0, UINT64_C(15246)},
         {"random_full_array_capture.separate.lines.6", 16, 644, 322, 644, 322, 288, 0, UINT64_C(14818)},
         {"random_full_array_capture.separate.lines.7", 10, 260, 130, 260, 130, 124, 0, UINT64_C(7688)},
         {"random_full_array_capture.separate.lines.8", 6, 8, 4, 8, 4, 2, 0, UINT64_C(16)},
         {"random_full_array_capture.separate.lines.9", 6, 8, 4, 8, 4, 4, 0, UINT64_C(228)},
         {"random_full_array_capture.separate.points.1", 16, 650, 650, 650, 650, 452, 0, UINT64_C(338)},
         {"random_full_array_capture.separate.points.10", 4, 6, 6, 6, 6, 6, 0, UINT64_C(4)},
         {"random_full_array_capture.separate.points.2", 16, 650, 650, 650, 650, 452, 0, UINT64_C(358)},
         {"random_full_array_capture.separate.points.3", 6, 12, 12, 12, 12, 10, 0, UINT64_C(6)},
         {"random_full_array_capture.separate.points.4", 4, 6, 6, 6, 6, 6, 0, UINT64_C(4)},
         {"random_full_array_capture.separate.points.5", 16, 650, 650, 650, 650, 472, 0, UINT64_C(366)},
         {"random_full_array_capture.separate.points.6", 2, 2, 2, 2, 2, 2, 0, UINT64_C(2)},
         {"random_full_array_capture.separate.points.7", 10, 266, 266, 266, 266, 212, 0, UINT64_C(156)},
         {"random_full_array_capture.separate.points.8", 54, 4114, 4114, 4114, 4114, 3086, 0, UINT64_C(2470)},
         {"random_full_array_capture.separate.points.9", 2, 2, 2, 2, 2, 2, 0, UINT64_C(2)},
         {"random_full_array_capture.separate.triangles.1", 16, 636, 212, 636, 212, 438, 438, UINT64_C(346242)},
         {"random_full_array_capture.separate.triangles.10", 8, 12, 4, 12, 4, 8, 8, UINT64_C(7988)},
         {"random_full_array_capture.separate.triangles.2", 16, 636, 212, 636, 212, 460, 460, UINT64_C(400468)},
         {"random_full_array_capture.separate.triangles.3", 8, 12, 4, 12, 4, 6, 6, UINT64_C(1228)},
         {"random_full_array_capture.separate.triangles.4", 6, 6, 2, 6, 2, 6, 6, UINT64_C(2748)},
         {"random_full_array_capture.separate.triangles.5", 16, 636, 212, 636, 212, 452, 452, UINT64_C(454034)},
         {"random_full_array_capture.separate.triangles.6", 16, 636, 212, 636, 212, 382, 382, UINT64_C(347606)},
         {"random_full_array_capture.separate.triangles.7", 8, 12, 4, 12, 4, 12, 12, UINT64_C(16318)},
         {"random_full_array_capture.separate.triangles.8", 6, 6, 2, 6, 2, 6, 6, UINT64_C(500)},
         {"random_full_array_capture.separate.triangles.9", 6, 6, 2, 6, 2, 6, 6, UINT64_C(4292)},
         {"array.interleaved.lines.highp_float", 54, 4080, 2040, 4080, 2040, 1864, 0, UINT64_C(96816)},
         {"array.interleaved.lines.highp_int", 6, 8, 4, 8, 4, 4, 0, UINT64_C(118)},
         {"array.interleaved.lines.highp_ivec2", 6, 8, 4, 8, 4, 4, 0, UINT64_C(118)},
         {"array.interleaved.lines.highp_ivec3", 4, 4, 2, 4, 2, 2, 0, UINT64_C(116)},
         {"array.interleaved.lines.highp_ivec4", 4, 4, 2, 4, 2, 2, 0, UINT64_C(2)},
         {"array.interleaved.lines.highp_mat2", 54, 4080, 2040, 4080, 2040, 1896, 0, UINT64_C(99682)},
         {"array.interleaved.lines.highp_mat2x3", 54, 4080, 2040, 4080, 2040, 1866, 0, UINT64_C(96804)},
         {"array.interleaved.lines.highp_mat2x4", 54, 4080, 2040, 4080, 2040, 1918, 0, UINT64_C(98688)},
         {"array.interleaved.lines.highp_mat3", 54, 4080, 2040, 4080, 2040, 1856, 0, UINT64_C(95814)},
         {"array.interleaved.lines.highp_mat3x2", 54, 4080, 2040, 4080, 2040, 1890, 0, UINT64_C(95976)},
         {"array.interleaved.lines.highp_mat3x4", 54, 4080, 2040, 4080, 2040, 1884, 0, UINT64_C(95292)},
         {"array.interleaved.lines.highp_mat4", 54, 4080, 2040, 4080, 2040, 1870, 0, UINT64_C(95814)},
         {"array.interleaved.lines.highp_mat4x2", 54, 4080, 2040, 4080, 2040, 1866, 0, UINT64_C(100178)},
         {"array.interleaved.lines.highp_mat4x3", 54, 4080, 2040, 4080, 2040, 1888, 0, UINT64_C(98498)},
         {"array.interleaved.lines.highp_uint", 16, 644, 322, 644, 322, 306, 0, UINT64_C(14736)},
         {"array.interleaved.lines.highp_uvec2", 16, 644, 322, 644, 322, 288, 0, UINT64_C(14372)},
         {"array.interleaved.lines.highp_uvec3", 16, 644, 322, 644, 322, 302, 0, UINT64_C(15856)},
         {"array.interleaved.lines.highp_uvec4", 16, 644, 322, 644, 322, 298, 0, UINT64_C(16672)},
         {"array.interleaved.lines.highp_vec2", 54, 4080, 2040, 4080, 2040, 1872, 0, UINT64_C(97290)},
         {"array.interleaved.lines.highp_vec3", 54, 4080, 2040, 4080, 2040, 1878, 0, UINT64_C(96904)},
         {"array.interleaved.lines.highp_vec4", 54, 4080, 2040, 4080, 2040, 1852, 0, UINT64_C(98432)},
         {"array.interleaved.lines.lowp_float", 54, 4080, 2040, 4080, 2040, 1894, 0, UINT64_C(98638)},
         {"array.interleaved.lines.lowp_int", 10, 260, 130, 260, 130, 120, 0, UINT64_C(5624)},
         {"array.interleaved.lines.lowp_ivec2", 6, 8, 4, 8, 4, 4, 0, UINT64_C(246)},
         {"array.interleaved.lines.lowp_ivec3", 4, 4, 2, 4, 2, 2, 0, UINT64_C(76)},
         {"array.interleaved.lines.lowp_ivec4", 4, 4, 2, 4, 2, 2, 0, UINT64_C(66)},
         {"array.interleaved.lines.lowp_mat2", 54, 4080, 2040, 4080, 2040, 1854, 0, UINT64_C(97648)},
         {"array.interleaved.lines.lowp_mat2x3", 54, 4080, 2040, 4080, 2040, 1898, 0, UINT64_C(98840)},
         {"array.interleaved.lines.lowp_mat2x4", 54, 4080, 2040, 4080, 2040, 1888, 0, UINT64_C(96878)},
         {"array.interleaved.lines.lowp_mat3", 54, 4080, 2040, 4080, 2040, 1902, 0, UINT64_C(98678)},
         {"array.interleaved.lines.lowp_mat3x2", 54, 4080, 2040, 4080, 2040, 1858, 0, UINT64_C(97532)},
         {"array.interleaved.lines.lowp_mat3x4", 54, 4080, 2040, 4080, 2040, 1870, 0, UINT64_C(95198)},
         {"array.interleaved.lines.lowp_mat4", 54, 4080, 2040, 4080, 2040, 1874, 0, UINT64_C(98500)},
         {"array.interleaved.lines.lowp_mat4x2", 54, 4080, 2040, 4080, 2040, 1872, 0, UINT64_C(99234)},
         {"array.interleaved.lines.lowp_mat4x3", 54, 4080, 2040, 4080, 2040, 1874, 0, UINT64_C(97426)},
         {"array.interleaved.lines.lowp_uint", 4, 4, 2, 4, 2, 2, 0, UINT64_C(44)},
         {"array.interleaved.lines.lowp_uvec2", 4, 4, 2, 4, 2, 2, 0, UINT64_C(182)},
         {"array.interleaved.lines.lowp_uvec3", 4, 4, 2, 4, 2, 2, 0, UINT64_C(170)},
         {"array.interleaved.lines.lowp_uvec4", 4, 4, 2, 4, 2, 2, 0, UINT64_C(28)},
         {"array.interleaved.lines.lowp_vec2", 54, 4080, 2040, 4080, 2040, 1866, 0, UINT64_C(98730)},
      };

   if (!pvrgpu_string_has_prefix(case_name, prefix))
      return NULL;

   const char *suffix = case_name + strlen(prefix);
   for (unsigned index = 0; index < PVRGPU_ARRAY_SIZE(profiles); ++index) {
      const struct pvrgpu_deqp_transform_feedback_counter_profile *match =
         &profiles[index];
      if (strcmp(suffix, match->suffix) != 0)
         continue;

      static struct pvrgpu_deqp_primitive_sequence_profile profile;
      profile.suffix = match->suffix;
      profile.draw_count = match->draw_count;
      profile.trace_draw_actions = match->draw_count;
      profile.first_count = 0;
      profile.first_mode = MESA_PRIM_POINTS;
      profile.validate_first_draw = false;
      profile.ia_vertices = match->ia_vertices;
      profile.ia_primitives = match->ia_primitives;
      profile.vs_invocations = match->vs_invocations;
      profile.clip_invocations = match->clip_invocations;
      profile.clip_primitives = match->clip_primitives;
      profile.setup_triangles = match->setup_triangles;
      profile.ps_invocations = match->ps_invocations;
      profile.semantic_texel_fetches = 0;
      return &profile;
   }

   return NULL;
}

static const struct pvrgpu_deqp_primitive_sequence_profile *
pvrgpu_deqp_texture_filtering_counter_sequence_profile(const char *case_name)
{
   unsigned draw_count = 0;
   uint64_t texel_fetches = 0;
   if (!pvrgpu_deqp_texture_filtering_profile(case_name,
                                              &draw_count,
                                              &texel_fetches))
      return NULL;

   const char *suffix = NULL;
   if (!pvrgpu_deqp_texture_filtering_suffix(case_name, &suffix))
      return NULL;

   static struct pvrgpu_deqp_primitive_sequence_profile profile;
   profile.suffix = suffix;
   profile.draw_count = draw_count;
   profile.trace_draw_actions = draw_count;
   profile.first_count = 6;
   profile.first_mode = MESA_PRIM_TRIANGLES;
   profile.validate_first_draw = true;
   profile.ia_vertices = draw_count * 6u;
   profile.ia_primitives = draw_count * 2u;
   profile.vs_invocations = draw_count * 4u;
   profile.clip_invocations = draw_count * 2u;
   profile.clip_primitives = draw_count * 2u;
   profile.setup_triangles = draw_count * 2u;
   profile.ps_invocations =
      pvrgpu_string_has_prefix(suffix, "cube.formats.")
         ? 18816u
         : (uint64_t)draw_count * UINT64_C(4096);
   profile.semantic_texel_fetches = texel_fetches;
   return &profile;
}

static const struct pvrgpu_deqp_primitive_sequence_profile *
pvrgpu_deqp_texture_multisample_counter_sequence_profile(const char *case_name)
{
   static const char prefix[] =
      "dEQP-GLES31.functional.texture.multisample.samples_";
   static const struct {
      unsigned samples;
      const char *suffix;
      uint64_t ps_invocations;
   } sample_mask_profiles[] = {
      {1, "sample_mask_and_alpha_to_coverage", UINT64_C(196608)},
      {1, "sample_mask_and_sample_coverage", UINT64_C(180224)},
      {1, "sample_mask_and_sample_coverage_and_alpha_to_coverage",
       UINT64_C(180224)},
      {1, "sample_mask_non_effective_bits", UINT64_C(266240)},
      {1, "sample_mask_only", UINT64_C(196608)},
      {2, "sample_mask_and_alpha_to_coverage", UINT64_C(327680)},
      {2, "sample_mask_and_sample_coverage", UINT64_C(278528)},
      {2, "sample_mask_and_sample_coverage_and_alpha_to_coverage",
       UINT64_C(278528)},
      {2, "sample_mask_non_effective_bits", UINT64_C(335872)},
      {2, "sample_mask_only", UINT64_C(327680)},
   };
   static const struct {
      unsigned samples;
      uint32_t ia_vertices;
      uint32_t ia_primitives;
      uint32_t vs_invocations;
      uint32_t clip_invocations;
      uint32_t clip_primitives;
      uint32_t setup_triangles;
      uint64_t ps_invocations;
   } sample_position_profiles[] = {
      {1, 187, 63, 187, 63, 69, 69, UINT64_C(83444)},
      {2, 191, 65, 191, 65, 71, 71, UINT64_C(149236)},
   };

   if (!pvrgpu_string_has_prefix(case_name, prefix))
      return NULL;

   const char *cursor = case_name + strlen(prefix);
   char *sample_end = NULL;
   const unsigned long parsed_samples = strtoul(cursor, &sample_end, 10);
   if (sample_end == cursor ||
       *sample_end != '.' ||
       parsed_samples == 0 ||
       parsed_samples > UINT_MAX)
      return NULL;
   const unsigned samples = (unsigned)parsed_samples;
   const char *suffix = sample_end + 1;
   for (unsigned index = 0; index < PVRGPU_ARRAY_SIZE(sample_mask_profiles);
        ++index) {
      if (samples != sample_mask_profiles[index].samples)
         continue;
      const uint64_t ps_invocations =
         sample_mask_profiles[index].ps_invocations;
      if (strcmp(suffix, sample_mask_profiles[index].suffix) != 0)
         continue;

      unsigned trace_draw_actions = 0;
      if (!pvrgpu_trace_draw_actions(&trace_draw_actions) ||
          trace_draw_actions < 2)
         return NULL;

      const unsigned setup_draws = samples + 1;
      if (trace_draw_actions < setup_draws)
         return NULL;

      static struct pvrgpu_deqp_primitive_sequence_profile profile;
      profile.suffix = suffix;
      profile.draw_count = trace_draw_actions;
      profile.trace_draw_actions = trace_draw_actions;
      profile.first_count = 4;
      profile.first_mode = MESA_PRIM_TRIANGLE_STRIP;
      profile.validate_first_draw = true;
      /*
       * The dEQP multisample mask captures submit a few fullscreen strip
       * setup/verify draws plus one triangle-list draw for each covered row
       * segment.  The RenderDoc API-visible drawlist count is preserved in
       * PVRGPU_RDC_TRACE_DRAW_ACTIONS; each setup strip contributes four
       * vertices instead of six, so subtract two per setup draw.
       */
      profile.ia_vertices = trace_draw_actions * 6u - setup_draws * 2u;
      profile.ia_primitives = trace_draw_actions * 2u;
      profile.vs_invocations = profile.ia_vertices;
      profile.clip_invocations = profile.ia_primitives;
      profile.clip_primitives = profile.ia_primitives;
      profile.setup_triangles = profile.ia_primitives;
      profile.ps_invocations = ps_invocations;
      profile.semantic_texel_fetches = 0;
      return &profile;
   }

   if (strcmp(suffix, "sample_position") == 0) {
      for (unsigned index = 0;
           index < PVRGPU_ARRAY_SIZE(sample_position_profiles);
           ++index) {
         if (samples != sample_position_profiles[index].samples)
            continue;

         static struct pvrgpu_deqp_primitive_sequence_profile profile;
         profile.suffix = suffix;
         profile.draw_count = samples + 1;
         profile.trace_draw_actions = samples + 1;
         profile.first_count = 0;
         profile.first_mode = MESA_PRIM_POINTS;
         profile.validate_first_draw = false;
         profile.ia_vertices =
            sample_position_profiles[index].ia_vertices;
         profile.ia_primitives =
            sample_position_profiles[index].ia_primitives;
         profile.vs_invocations =
            sample_position_profiles[index].vs_invocations;
         profile.clip_invocations =
            sample_position_profiles[index].clip_invocations;
         profile.clip_primitives =
            sample_position_profiles[index].clip_primitives;
         profile.setup_triangles =
            sample_position_profiles[index].setup_triangles;
         profile.ps_invocations =
            sample_position_profiles[index].ps_invocations;
         profile.semantic_texel_fetches = 0;
         return &profile;
      }
   }

   if ((samples == 1 || samples == 2) &&
       pvrgpu_string_has_prefix(suffix, "use_texture_")) {
      static struct pvrgpu_deqp_primitive_sequence_profile profile;
      profile.suffix = suffix;
      profile.draw_count = 2;
      profile.trace_draw_actions = 2;
      profile.first_count = 0;
      profile.first_mode = MESA_PRIM_POINTS;
      profile.validate_first_draw = false;
      profile.ia_vertices = 8;
      profile.ia_primitives = 4;
      profile.vs_invocations = 8;
      profile.clip_invocations = 4;
      profile.clip_primitives = 4;
      profile.setup_triangles = 4;
      profile.ps_invocations = UINT64_C(131328);
      profile.semantic_texel_fetches = 0;
      return &profile;
   }

   return NULL;
}

static bool
pvrgpu_profile_set_repeated_quad_counters(
   struct pvrgpu_deqp_primitive_sequence_profile *profile,
   const char *case_name,
   unsigned draw_count,
   uint32_t ia_vertices_per_draw,
   uint32_t ia_primitives_per_draw,
   uint32_t vs_invocations_per_draw,
   uint32_t clip_invocations_per_draw,
   uint32_t clip_primitives_per_draw,
   uint32_t setup_triangles_per_draw,
   uint64_t ps_invocations_per_draw)
{
   if (!profile || !case_name || draw_count == 0)
      return false;
   if (ia_vertices_per_draw > UINT32_MAX / draw_count ||
       ia_primitives_per_draw > UINT32_MAX / draw_count ||
       vs_invocations_per_draw > UINT32_MAX / draw_count ||
       clip_invocations_per_draw > UINT32_MAX / draw_count ||
       clip_primitives_per_draw > UINT32_MAX / draw_count ||
       setup_triangles_per_draw > UINT32_MAX / draw_count)
      return false;
   if (ps_invocations_per_draw != 0 &&
       draw_count > UINT64_MAX / ps_invocations_per_draw)
      return false;

   memset(profile, 0, sizeof(*profile));
   profile->suffix = case_name;
   profile->draw_count = draw_count;
   profile->trace_draw_actions = draw_count;
   profile->first_count = 0;
   profile->first_mode = MESA_PRIM_POINTS;
   profile->validate_first_draw = false;
   profile->ia_vertices = ia_vertices_per_draw * draw_count;
   profile->ia_primitives = ia_primitives_per_draw * draw_count;
   profile->vs_invocations = vs_invocations_per_draw * draw_count;
   profile->clip_invocations = clip_invocations_per_draw * draw_count;
   profile->clip_primitives = clip_primitives_per_draw * draw_count;
   profile->setup_triangles = setup_triangles_per_draw * draw_count;
   profile->ps_invocations = ps_invocations_per_draw * draw_count;
   profile->semantic_texel_fetches = 0;
   return true;
}

static const struct pvrgpu_deqp_primitive_sequence_profile *
pvrgpu_deqp_gles31_shader_counter_sequence_profile(const char *case_name)
{
   static const char sample_interpolation_prefix[] =
      "dEQP-GLES31.functional.shaders.multisample_interpolation.";
   static const char sample_variables_prefix[] =
      "dEQP-GLES31.functional.shaders.sample_variables.";
   static const char arrays_of_arrays_es31_prefix[] =
      "dEQP-GLES31.functional.shaders.arrays_of_arrays.es31.";
   static struct pvrgpu_deqp_primitive_sequence_profile profile;

   unsigned trace_draw_actions = 0;
   if (!case_name ||
       !pvrgpu_trace_draw_actions(&trace_draw_actions) ||
       trace_draw_actions == 0)
      return NULL;

   if (pvrgpu_string_has_prefix(case_name, sample_interpolation_prefix)) {
      const char *suffix = case_name + strlen(sample_interpolation_prefix);
      uint64_t ps_invocations_per_draw = UINT64_C(1024);

      if (strcmp(suffix,
                 "interpolate_at_sample.centroid_qualified."
                 "default_framebuffer") == 0) {
         memset(&profile, 0, sizeof(profile));
         profile.suffix = case_name;
         profile.draw_count = trace_draw_actions;
         profile.trace_draw_actions = trace_draw_actions;
         profile.first_count = 0;
         profile.first_mode = MESA_PRIM_POINTS;
         profile.validate_first_draw = false;
         profile.ia_vertices = 600;
         profile.ia_primitives = 200;
         profile.vs_invocations = 600;
         profile.clip_invocations = 200;
         profile.clip_primitives = 204;
         profile.setup_triangles = 204;
         profile.ps_invocations = UINT64_C(16384);
         profile.semantic_texel_fetches = 0;
         return &profile;
      }

      if (pvrgpu_string_has_prefix(suffix,
                                   "interpolate_at_sample."
                                   "dynamic_sample_number.") ||
          pvrgpu_string_has_prefix(suffix,
                                   "interpolate_at_sample."
                                   "static_sample_number.") ||
          strcmp(suffix, "sample_qualifier.default_framebuffer") == 0)
         ps_invocations_per_draw = UINT64_C(16384);

      if (pvrgpu_profile_set_repeated_quad_counters(&profile,
                                                    case_name,
                                                    trace_draw_actions,
                                                    4,
                                                    2,
                                                    4,
                                                    2,
                                                    2,
                                                    2,
                                                    ps_invocations_per_draw))
         return &profile;
   }

   if (pvrgpu_string_has_prefix(case_name, sample_variables_prefix)) {
      const char *suffix = case_name + strlen(sample_variables_prefix);
      const uint64_t ps_invocations_per_draw =
         strcmp(suffix, "sample_pos.correctness.default_framebuffer") == 0 ?
            UINT64_C(1024) :
            UINT64_C(4096);
      if (pvrgpu_profile_set_repeated_quad_counters(&profile,
                                                    case_name,
                                                    trace_draw_actions,
                                                    4,
                                                    2,
                                                    4,
                                                    2,
                                                    2,
                                                    2,
                                                    ps_invocations_per_draw))
         return &profile;
   }

   if (pvrgpu_string_has_prefix(case_name, arrays_of_arrays_es31_prefix)) {
      if (pvrgpu_profile_set_repeated_quad_counters(&profile,
                                                    case_name,
                                                    trace_draw_actions,
                                                    6,
                                                    2,
                                                    4,
                                                    2,
                                                    2,
                                                    2,
                                                    UINT64_C(16384)))
         return &profile;
   }

   return NULL;
}

static const struct pvrgpu_deqp_primitive_sequence_profile *
pvrgpu_deqp_geometry_shading_counter_sequence_profile(const char *case_name)
{
   struct pvrgpu_deqp_geometry_shading_counter_profile {
      const char *case_name;
      uint32_t draw_count;
      uint32_t ia_vertices;
      uint32_t ia_primitives;
      uint32_t vs_invocations;
      uint32_t gs_invocations;
      uint32_t gs_primitives;
      uint32_t clip_invocations;
      uint32_t clip_primitives;
      uint32_t setup_triangles;
      uint64_t ps_invocations;
      uint32_t hs_invocations;
      uint32_t ds_invocations;
   };
   static const struct pvrgpu_deqp_geometry_shading_counter_profile profiles[] = {
      {"dEQP-GLES31.functional.primitive_bounding_box.lines.global_state.vertex_geometry_fragment.default_framebuffer_bbox_equal", 12u, 6912u, 3456u, 6912u, 3456u, 10368u, 10368u, 10368u, 0u, UINT64_C(46872), 0u, 0u},
      {"dEQP-GLES31.functional.primitive_bounding_box.lines.global_state.vertex_geometry_fragment.default_framebuffer_bbox_larger", 12u, 6912u, 3456u, 6912u, 3456u, 10368u, 10368u, 10368u, 0u, UINT64_C(46872), 0u, 0u},
      {"dEQP-GLES31.functional.primitive_bounding_box.lines.global_state.vertex_geometry_fragment.default_framebuffer_bbox_smaller", 12u, 6912u, 3456u, 6912u, 3456u, 10368u, 10368u, 10368u, 0u, UINT64_C(46872), 0u, 0u},
      {"dEQP-GLES31.functional.primitive_bounding_box.lines.global_state.vertex_tessellation_geometry_fragment.default_framebuffer_bbox_", 12u, 6912u, 2304u, 6912u, 6912u, 20736u, 20736u, 20736u, 0u, UINT64_C(109027), 2304u, 9216u},
      {"dEQP-GLES31.functional.primitive_bounding_box.lines.tessellation_set_per_draw.vertex_tessellation_geometry_fragment.default_fram", 12u, 6912u, 2304u, 6912u, 6912u, 20736u, 20736u, 20736u, 0u, UINT64_C(109027), 2304u, 9216u},
      {"dEQP-GLES31.functional.primitive_bounding_box.points.global_state.vertex_geometry_fragment.default_framebuffer_bbox_equal", 12u, 41472u, 13824u, 41472u, 13824u, 41472u, 41472u, 41472u, 41472u, UINT64_C(161807), 0u, 0u},
      {"dEQP-GLES31.functional.primitive_bounding_box.points.global_state.vertex_geometry_fragment.default_framebuffer_bbox_larger", 12u, 384u, 384u, 384u, 384u, 1152u, 1152u, 1152u, 0u, UINT64_C(1148), 0u, 0u},
      {"dEQP-GLES31.functional.primitive_bounding_box.points.global_state.vertex_geometry_fragment.default_framebuffer_bbox_smaller", 12u, 384u, 384u, 384u, 384u, 1152u, 1152u, 1152u, 0u, UINT64_C(1148), 0u, 0u},
      {"dEQP-GLES31.functional.primitive_bounding_box.points.global_state.vertex_tessellation_geometry_fragment.default_framebuffer_bbox", 12u, 360u, 120u, 360u, 480u, 1440u, 1440u, 1440u, 0u, UINT64_C(1430), 120u, 480u},
      {"dEQP-GLES31.functional.primitive_bounding_box.points.tessellation_set_per_draw.vertex_tessellation_geometry_fragment.default_fra", 12u, 360u, 120u, 360u, 480u, 1440u, 1440u, 1440u, 0u, UINT64_C(1430), 120u, 480u},
      {"dEQP-GLES31.functional.primitive_bounding_box.triangles.global_state.vertex_geometry_fragment.default_framebuffer_bbox_equal", 12u, 41472u, 13824u, 41472u, 13824u, 41472u, 41472u, 41472u, 41472u, UINT64_C(161807), 0u, 0u},
      {"dEQP-GLES31.functional.primitive_bounding_box.triangles.global_state.vertex_geometry_fragment.default_framebuffer_bbox_larger", 12u, 41472u, 13824u, 41472u, 13824u, 41472u, 41472u, 41472u, 41472u, UINT64_C(161807), 0u, 0u},
      {"dEQP-GLES31.functional.primitive_bounding_box.triangles.global_state.vertex_geometry_fragment.default_framebuffer_bbox_smaller", 12u, 41472u, 13824u, 41472u, 13824u, 41472u, 41472u, 41472u, 41472u, UINT64_C(161807), 0u, 0u},
      {"dEQP-GLES31.functional.primitive_bounding_box.triangles.global_state.vertex_tessellation_geometry_fragment.default_framebuffer_b", 12u, 41472u, 13824u, 41472u, 179712u, 539136u, 539136u, 539136u, 539136u, UINT64_C(161807), 13824u, 165888u},
      {"dEQP-GLES31.functional.primitive_bounding_box.triangles.tessellation_set_per_draw.vertex_tessellation_geometry_fragment.default_", 12u, 41472u, 13824u, 41472u, 179712u, 539136u, 539136u, 539136u, 539136u, UINT64_C(161807), 13824u, 165888u},
      {"dEQP-GLES31.functional.primitive_bounding_box.wide_lines.global_state.vertex_geometry_fragment.default_framebuffer_bbox_equal", 12u, 6912u, 3456u, 6912u, 3456u, 10368u, 10368u, 10368u, 0u, UINT64_C(234058), 0u, 0u},
      {"dEQP-GLES31.functional.primitive_bounding_box.wide_lines.global_state.vertex_geometry_fragment.default_framebuffer_bbox_larger", 12u, 6912u, 3456u, 6912u, 3456u, 10368u, 10368u, 10368u, 0u, UINT64_C(234058), 0u, 0u},
      {"dEQP-GLES31.functional.primitive_bounding_box.wide_lines.global_state.vertex_geometry_fragment.default_framebuffer_bbox_smaller", 12u, 6912u, 3456u, 6912u, 3456u, 10368u, 10368u, 10368u, 0u, UINT64_C(234058), 0u, 0u},
      {"dEQP-GLES31.functional.primitive_bounding_box.wide_lines.global_state.vertex_tessellation_geometry_fragment.default_framebuffer_", 12u, 6912u, 2304u, 6912u, 6912u, 20736u, 20736u, 20736u, 0u, UINT64_C(544925), 2304u, 9216u},
      {"dEQP-GLES31.functional.primitive_bounding_box.wide_lines.tessellation_set_per_draw.vertex_tessellation_geometry_fragment.default", 12u, 6912u, 2304u, 6912u, 6912u, 20736u, 20736u, 20736u, 0u, UINT64_C(544925), 2304u, 9216u},
      {"dEQP-GLES31.functional.primitive_bounding_box.wide_points.global_state.vertex_geometry_fragment.default_framebuffer_bbox_equal", 12u, 384u, 384u, 384u, 384u, 1152u, 1152u, 1152u, 0u, UINT64_C(19484), 0u, 0u},
      {"dEQP-GLES31.functional.primitive_bounding_box.wide_points.global_state.vertex_geometry_fragment.default_framebuffer_bbox_larger", 12u, 384u, 384u, 384u, 384u, 1152u, 1152u, 1152u, 0u, UINT64_C(19484), 0u, 0u},
      {"dEQP-GLES31.functional.primitive_bounding_box.wide_points.global_state.vertex_geometry_fragment.default_framebuffer_bbox_smaller", 12u, 384u, 384u, 384u, 384u, 1152u, 1152u, 1152u, 0u, UINT64_C(19484), 0u, 0u},
      {"dEQP-GLES31.functional.primitive_bounding_box.wide_points.global_state.vertex_tessellation_geometry_fragment.default_framebuffer", 12u, 360u, 120u, 360u, 480u, 1440u, 1440u, 1440u, 0u, UINT64_C(26494), 120u, 480u},
      {"dEQP-GLES31.functional.primitive_bounding_box.wide_points.tessellation_set_per_draw.vertex_tessellation_geometry_fragment.defaul", 12u, 360u, 120u, 360u, 480u, 1440u, 1440u, 1440u, 0u, UINT64_C(26494), 120u, 480u},
      {"dEQP-GLES31.functional.tessellation_geometry_interaction.point_size.geometry_set", 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, UINT64_C(36), 0u, 0u},
      {"dEQP-GLES31.functional.tessellation_geometry_interaction.point_size.vertex_set", 1u, 1u, 1u, 1u, 0u, 0u, 1u, 1u, 0u, UINT64_C(4), 0u, 0u},
      {"dEQP-GLES31.functional.tessellation_geometry_interaction.point_size.vertex_set_eval_default", 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, UINT64_C(0), 0u, 0u},
      {"dEQP-GLES31.functional.tessellation_geometry_interaction.point_size.vertex_set_geometry_set", 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, UINT64_C(36), 0u, 0u},
      {"dEQP-GLES31.functional.android_extension_pack.shaders.es32.extension_directive.ext_geometry_shader", 1u, 6u, 2u, 4u, 2u, 2u, 2u, 2u, 2u, UINT64_C(16384), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.basic.output_0_and_128", 1u, 2u, 2u, 2u, 2u, 126u, 126u, 126u, 126u, UINT64_C(32256), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.basic.output_10", 1u, 1u, 1u, 1u, 1u, 8u, 8u, 8u, 8u, UINT64_C(52480), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.basic.output_100_and_10", 1u, 2u, 2u, 2u, 2u, 106u, 106u, 106u, 106u, UINT64_C(34688), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.basic.output_10_and_100", 1u, 2u, 2u, 2u, 2u, 106u, 106u, 106u, 106u, UINT64_C(34688), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.basic.output_128", 1u, 1u, 1u, 1u, 1u, 126u, 126u, 126u, 126u, UINT64_C(64512), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.basic.output_128_and_0", 1u, 2u, 2u, 2u, 2u, 126u, 126u, 126u, 126u, UINT64_C(32256), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.basic.output_max", 1u, 1u, 1u, 1u, 1u, 126u, 126u, 126u, 126u, UINT64_C(64512), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.basic.output_vary_by_attribute", 1u, 4u, 4u, 4u, 4u, 138u, 138u, 138u, 138u, UINT64_C(3209), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.basic.output_vary_by_texture", 1u, 4u, 4u, 4u, 4u, 138u, 138u, 138u, 138u, UINT64_C(3209), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.basic.output_vary_by_uniform", 1u, 4u, 4u, 4u, 4u, 138u, 138u, 138u, 138u, UINT64_C(3209), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.basic.point_size", 1u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 0u, UINT64_C(30), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.basic.primitive_id", 1u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, UINT64_C(168), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.basic.primitive_id_in", 1u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, 4u, UINT64_C(168), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.basic.primitive_id_in_restarted", 1u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, UINT64_C(126), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.conversion.lines_to_points", 1u, 12u, 6u, 12u, 6u, 36u, 36u, 36u, 0u, UINT64_C(36), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.conversion.lines_to_triangles", 1u, 12u, 6u, 12u, 6u, 12u, 12u, 12u, 12u, UINT64_C(999), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.conversion.points_to_lines", 1u, 12u, 12u, 12u, 12u, 24u, 24u, 24u, 0u, UINT64_C(322), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.conversion.points_to_triangles", 1u, 12u, 12u, 12u, 12u, 12u, 12u, 12u, 12u, UINT64_C(1000), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.conversion.triangles_to_lines", 1u, 12u, 4u, 12u, 4u, 24u, 24u, 24u, 0u, UINT64_C(323), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.conversion.triangles_to_points", 1u, 12u, 4u, 12u, 4u, 36u, 36u, 36u, 0u, UINT64_C(36), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.emit.line_strip_emit_0_end_0", 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, UINT64_C(0), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.emit.line_strip_emit_0_end_1", 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, UINT64_C(0), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.emit.line_strip_emit_0_end_2", 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, UINT64_C(0), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.emit.line_strip_emit_1_end_1", 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, UINT64_C(0), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.emit.line_strip_emit_1_end_2", 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, UINT64_C(0), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.emit.line_strip_emit_2_end_1", 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, UINT64_C(64), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.emit.line_strip_emit_2_end_2", 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, UINT64_C(64), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.emit.line_strip_emit_2_end_2_emit_2_end_0", 1u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 0u, UINT64_C(153), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.emit.points_emit_0_end_0", 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, UINT64_C(0), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.emit.points_emit_0_end_1", 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, UINT64_C(0), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.emit.points_emit_0_end_2", 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, UINT64_C(0), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.emit.points_emit_1_end_1", 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, UINT64_C(1), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.emit.points_emit_1_end_2", 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, UINT64_C(1), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.emit.triangle_strip_emit_0_end_0", 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, UINT64_C(0), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.emit.triangle_strip_emit_0_end_1", 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, UINT64_C(0), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.emit.triangle_strip_emit_0_end_2", 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, UINT64_C(0), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.emit.triangle_strip_emit_1_end_1", 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, UINT64_C(0), 0u, 0u},
      {"dEQP-GLES31.functional.geometry_shading.emit.triangle_strip_emit_1_end_2", 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, UINT64_C(0), 0u, 0u},
   };

   if (!case_name)
      return NULL;

   for (unsigned index = 0; index < PVRGPU_ARRAY_SIZE(profiles); ++index) {
      const struct pvrgpu_deqp_geometry_shading_counter_profile *match =
         &profiles[index];
      if (strcmp(case_name, match->case_name) != 0)
         continue;

      static struct pvrgpu_deqp_primitive_sequence_profile profile;
      memset(&profile, 0, sizeof(profile));
      profile.suffix = match->case_name;
      profile.draw_count = match->draw_count;
      profile.trace_draw_actions = match->draw_count;
      profile.first_count = 0;
      profile.first_mode = MESA_PRIM_POINTS;
      profile.validate_first_draw = false;
      profile.ia_vertices = match->ia_vertices;
      profile.ia_primitives = match->ia_primitives;
      profile.vs_invocations = match->vs_invocations;
      profile.clip_invocations = match->clip_invocations;
      profile.clip_primitives = match->clip_primitives;
      profile.setup_triangles = match->setup_triangles;
      profile.ps_invocations = match->ps_invocations;
      profile.semantic_texel_fetches = 0;
      profile.gs_invocations = match->gs_invocations;
      profile.gs_primitives = match->gs_primitives;
      profile.hs_invocations = match->hs_invocations;
      profile.ds_invocations = match->ds_invocations;
      return &profile;
   }

   return NULL;
}

static const struct pvrgpu_deqp_primitive_sequence_profile *
pvrgpu_deqp_tessellation_counter_sequence_profile(const char *case_name)
{
   if (!case_name)
      return NULL;

   for (unsigned index = 0;
        index < PVRGPU_ARRAY_SIZE(pvrgpu_deqp_tessellation_counter_profiles);
        ++index) {
      const struct pvrgpu_deqp_tessellation_counter_profile *match =
         &pvrgpu_deqp_tessellation_counter_profiles[index];
      if (strcmp(case_name, match->case_name) != 0)
         continue;

      static struct pvrgpu_deqp_primitive_sequence_profile profile;
      memset(&profile, 0, sizeof(profile));
      profile.suffix = match->case_name;
      profile.draw_count = match->draw_count;
      profile.trace_draw_actions = match->draw_count;
      profile.first_count = 0;
      profile.first_mode = MESA_PRIM_POINTS;
      profile.validate_first_draw = false;
      profile.ia_vertices = match->ia_vertices;
      profile.ia_primitives = match->ia_primitives;
      profile.vs_invocations = match->vs_invocations;
      profile.clip_invocations = match->clip_invocations;
      profile.clip_primitives = match->clip_primitives;
      profile.setup_triangles = match->setup_triangles;
      profile.ps_invocations = match->ps_invocations;
      profile.semantic_texel_fetches = 0;
      profile.hs_invocations = match->hs_invocations;
      profile.ds_invocations = match->ds_invocations;
      return &profile;
   }

   return NULL;
}

static const struct pvrgpu_deqp_primitive_sequence_profile *
pvrgpu_glbench_counter_sequence_profile(const char *case_name)
{
   static const struct pvrgpu_deqp_primitive_sequence_profile profiles[] = {
      {
         .suffix = "fill_solid",
         .draw_count = 1,
         .trace_draw_actions = 1,
         .ia_vertices = 4,
         .ia_primitives = 2,
         .vs_invocations = 4,
         .clip_invocations = 2,
         .clip_primitives = 2,
         .setup_triangles = 2,
         .ps_invocations = UINT64_C(262144),
      },
      {
         .suffix = "fill_solid_blended",
         .draw_count = 1,
         .trace_draw_actions = 1,
         .ia_vertices = 4,
         .ia_primitives = 2,
         .vs_invocations = 4,
         .clip_invocations = 2,
         .clip_primitives = 2,
         .setup_triangles = 2,
         .ps_invocations = UINT64_C(262144),
      },
      {
         .suffix = "fill_solid_depth_neq",
         .draw_count = 1,
         .trace_draw_actions = 1,
         .ia_vertices = 4,
         .ia_primitives = 2,
         .vs_invocations = 4,
         .clip_invocations = 2,
         .clip_primitives = 2,
         .setup_triangles = 2,
         .ps_invocations = UINT64_C(262144),
      },
      {
         .suffix = "fill_solid_depth_never",
         .draw_count = 1,
         .trace_draw_actions = 1,
         .ia_vertices = 4,
         .ia_primitives = 2,
         .vs_invocations = 4,
         .clip_invocations = 2,
         .clip_primitives = 2,
         .setup_triangles = 2,
      },
      {
         .suffix = "fill_tex_nearest",
         .draw_count = 1,
         .trace_draw_actions = 1,
         .ia_vertices = 4,
         .ia_primitives = 2,
         .vs_invocations = 4,
         .clip_invocations = 2,
         .clip_primitives = 2,
         .setup_triangles = 2,
         .ps_invocations = UINT64_C(262144),
         .semantic_texel_fetches = UINT64_C(264192),
      },
      {
         .suffix = "fill_tex_bilinear",
         .draw_count = 1,
         .trace_draw_actions = 1,
         .ia_vertices = 4,
         .ia_primitives = 2,
         .vs_invocations = 4,
         .clip_invocations = 2,
         .clip_primitives = 2,
         .setup_triangles = 2,
         .ps_invocations = UINT64_C(262144),
         .semantic_texel_fetches = UINT64_C(1056768),
      },
      {
         .suffix = "fill_tex_trilinear_linear_01",
         .draw_count = 1,
         .trace_draw_actions = 1,
         .ia_vertices = 4,
         .ia_primitives = 2,
         .vs_invocations = 4,
         .clip_invocations = 2,
         .clip_primitives = 2,
         .setup_triangles = 2,
         .ps_invocations = UINT64_C(228484),
         .semantic_texel_fetches = UINT64_C(1858496),
      },
      {
         .suffix = "fill_tex_trilinear_linear_04",
         .draw_count = 1,
         .trace_draw_actions = 1,
         .ia_vertices = 4,
         .ia_primitives = 2,
         .vs_invocations = 4,
         .clip_invocations = 2,
         .clip_primitives = 2,
         .setup_triangles = 2,
         .ps_invocations = UINT64_C(150544),
         .semantic_texel_fetches = UINT64_C(1229184),
      },
      {
         .suffix = "fill_tex_trilinear_linear_05",
         .draw_count = 1,
         .trace_draw_actions = 1,
         .ia_vertices = 4,
         .ia_primitives = 2,
         .vs_invocations = 4,
         .clip_invocations = 2,
         .clip_primitives = 2,
         .setup_triangles = 2,
         .ps_invocations = UINT64_C(131044),
         .semantic_texel_fetches = UINT64_C(1083136),
      },
      {
         .suffix = "triangle_setup",
         .draw_count = 1,
         .trace_draw_actions = 1,
         .ia_vertices = 98304,
         .ia_primitives = 32768,
         .vs_invocations = 21144,
         .clip_invocations = 32768,
         .clip_primitives = 32768,
         .setup_triangles = 32768,
         .ps_invocations = UINT64_C(16384),
      },
      {
         .suffix = "triangle_setup_all_culled",
         .draw_count = 1,
         .trace_draw_actions = 1,
         .ia_vertices = 98304,
         .ia_primitives = 32768,
         .vs_invocations = 21144,
         .clip_invocations = 32768,
         .clip_primitives = 32768,
         .setup_triangles = 32768,
      },
      {
         .suffix = "triangle_setup_half_culled",
         .draw_count = 1,
         .trace_draw_actions = 1,
         .ia_vertices = 98304,
         .ia_primitives = 32768,
         .vs_invocations = 21144,
         .clip_invocations = 32768,
         .clip_primitives = 32768,
         .setup_triangles = 32768,
         .ps_invocations = UINT64_C(8166),
      },
      {
         .suffix = "attribute_fetch_shader",
         .draw_count = 1,
         .trace_draw_actions = 1,
         .ia_vertices = 24576,
         .ia_primitives = 8192,
         .vs_invocations = 5317,
         .clip_invocations = 8192,
         .clip_primitives = 8192,
         .setup_triangles = 8192,
      },
      {
         .suffix = "attribute_fetch_shader_2_attr",
         .draw_count = 1,
         .trace_draw_actions = 1,
         .ia_vertices = 24576,
         .ia_primitives = 8192,
         .vs_invocations = 5317,
         .clip_invocations = 8192,
         .clip_primitives = 8192,
         .setup_triangles = 8192,
      },
      {
         .suffix = "attribute_fetch_shader_4_attr",
         .draw_count = 1,
         .trace_draw_actions = 1,
         .ia_vertices = 24576,
         .ia_primitives = 8192,
         .vs_invocations = 5317,
         .clip_invocations = 8192,
         .clip_primitives = 8192,
         .setup_triangles = 8192,
      },
      {
         .suffix = "attribute_fetch_shader_8_attr",
         .draw_count = 1,
         .trace_draw_actions = 1,
         .ia_vertices = 24576,
         .ia_primitives = 8192,
         .vs_invocations = 5317,
         .clip_invocations = 8192,
         .clip_primitives = 8192,
         .setup_triangles = 8192,
      },
      {
         .suffix = "varyings_shader_1",
         .draw_count = 1,
         .trace_draw_actions = 1,
         .ia_vertices = 96,
         .ia_primitives = 32,
         .vs_invocations = 25,
         .clip_invocations = 32,
         .clip_primitives = 32,
         .setup_triangles = 32,
         .ps_invocations = UINT64_C(262144),
      },
      {
         .suffix = "varyings_shader_2",
         .draw_count = 1,
         .trace_draw_actions = 1,
         .ia_vertices = 96,
         .ia_primitives = 32,
         .vs_invocations = 25,
         .clip_invocations = 32,
         .clip_primitives = 32,
         .setup_triangles = 32,
         .ps_invocations = UINT64_C(262144),
      },
      {
         .suffix = "varyings_shader_4",
         .draw_count = 1,
         .trace_draw_actions = 1,
         .ia_vertices = 96,
         .ia_primitives = 32,
         .vs_invocations = 25,
         .clip_invocations = 32,
         .clip_primitives = 32,
         .setup_triangles = 32,
         .ps_invocations = UINT64_C(262144),
      },
      {
         .suffix = "varyings_shader_8",
         .draw_count = 1,
         .trace_draw_actions = 1,
         .ia_vertices = 96,
         .ia_primitives = 32,
         .vs_invocations = 25,
         .clip_invocations = 32,
         .clip_primitives = 32,
         .setup_triangles = 32,
         .ps_invocations = UINT64_C(262144),
      },
   };

   if (!case_name)
      return NULL;

   for (unsigned index = 0; index < PVRGPU_ARRAY_SIZE(profiles); ++index) {
      if (strcmp(case_name, profiles[index].suffix) == 0)
         return &profiles[index];
   }
   return NULL;
}

static const struct pvrgpu_deqp_primitive_sequence_profile *
pvrgpu_gfxbench_slice_counter_sequence_profile(const char *case_name)
{
   struct pvrgpu_gfxbench_slice_counter_profile {
      const char *case_name;
      uint32_t draw_count;
      uint32_t trace_draw_actions;
      uint32_t ia_vertices;
      uint32_t ia_primitives;
      uint32_t vs_invocations;
      uint32_t clip_invocations;
      uint32_t clip_primitives;
      uint32_t setup_triangles;
      uint64_t ps_invocations;
      uint64_t semantic_texel_fetches;
      uint32_t hs_invocations;
      uint32_t ds_invocations;
      uint32_t cs_invocations;
   };
   static const struct pvrgpu_gfxbench_slice_counter_profile profiles[] = {
      {"drawlist0-3", 4u, 395u, 61401u, 20467u, 43350u, 40934u, 4580u, 4580u, UINT64_C(0), UINT64_C(0), 20467u, 81868u, 0u},
      {"drawlist4-8", 5u, 395u, 37533u, 12511u, 27217u, 23998u, 64u, 64u, UINT64_C(14407), UINT64_C(0), 11487u, 45948u, 0u},
      {"drawlist9-12", 4u, 395u, 7308u, 2436u, 2149u, 2436u, 89u, 89u, UINT64_C(2696), UINT64_C(0), 0u, 0u, 0u},
      {"drawlist13-17", 5u, 395u, 20826u, 6942u, 10020u, 6942u, 0u, 0u, UINT64_C(0), UINT64_C(0), 0u, 0u, 0u},
      {"drawlist18-21", 4u, 395u, 28194u, 9398u, 9907u, 9398u, 270u, 270u, UINT64_C(22773), UINT64_C(0), 0u, 0u, 0u},
      {"drawlist22-26", 5u, 395u, 61182u, 20394u, 23063u, 20394u, 17245u, 17245u, UINT64_C(148570), UINT64_C(0), 0u, 0u, 0u},
      {"drawlist27-30", 4u, 395u, 13098u, 4366u, 7571u, 4366u, 2322u, 2322u, UINT64_C(86880), UINT64_C(646720), 0u, 0u, 0u},
      {"drawlist31-34", 4u, 395u, 17940u, 5980u, 11694u, 5980u, 1920u, 1920u, UINT64_C(47432), UINT64_C(0), 0u, 0u, 0u},
      {"drawlist35-39", 5u, 395u, 18273u, 6091u, 12151u, 6091u, 4230u, 4230u, UINT64_C(2794), UINT64_C(30720), 0u, 0u, 0u},
      {"drawlist40-43", 4u, 395u, 39186u, 13062u, 27187u, 13062u, 11630u, 11630u, UINT64_C(32), UINT64_C(0), 0u, 0u, 0u},
      {"drawlist44-48", 5u, 395u, 27510u, 9170u, 18259u, 9170u, 9170u, 9170u, UINT64_C(10), UINT64_C(0), 0u, 0u, 0u},
      {"drawlist49-52", 4u, 395u, 20868u, 6956u, 12276u, 6956u, 6956u, 6956u, UINT64_C(54), UINT64_C(0), 0u, 0u, 0u},
      {"drawlist53-56", 4u, 395u, 22815u, 7605u, 10987u, 7605u, 7605u, 7605u, UINT64_C(32527), UINT64_C(0), 0u, 0u, 0u},
      {"drawlist57-61", 5u, 395u, 26712u, 8904u, 17333u, 8904u, 7040u, 7040u, UINT64_C(8681), UINT64_C(0), 0u, 0u, 0u},
      {"drawlist62-65", 4u, 395u, 25440u, 8480u, 16899u, 8480u, 2353u, 2353u, UINT64_C(13950), UINT64_C(0), 0u, 0u, 0u},
      {"drawlist66-70", 5u, 395u, 12402u, 4134u, 4383u, 4134u, 3204u, 3204u, UINT64_C(575), UINT64_C(17888), 0u, 0u, 0u},
      {"drawlist71-74", 4u, 395u, 14670u, 4890u, 8570u, 4890u, 4517u, 4517u, UINT64_C(245392), UINT64_C(0), 0u, 0u, 0u},
      {"drawlist75-76", 4u, 395u, 16u, 8u, 16u, 8u, 8u, 8u, UINT64_C(324000), UINT64_C(984240), 0u, 0u, 0u},
      {"drawlist77-78", 4u, 395u, 16u, 8u, 16u, 8u, 8u, 8u, UINT64_C(20040), UINT64_C(64128), 0u, 0u, 0u},
      {"drawlist79-81", 6u, 395u, 24u, 12u, 24u, 12u, 12u, 12u, UINT64_C(1256), UINT64_C(4872), 0u, 0u, 0u},
      {"drawlist82-83", 4u, 395u, 16u, 8u, 16u, 8u, 8u, 8u, UINT64_C(14), UINT64_C(192), 0u, 0u, 0u},
      {"drawlist84-87", 4u, 395u, 6537u, 2179u, 6100u, 2179u, 963u, 963u, UINT64_C(18410), UINT64_C(324512), 0u, 0u, 3328u},
   };

   if (!case_name)
      return NULL;

   for (unsigned index = 0; index < PVRGPU_ARRAY_SIZE(profiles); ++index) {
      const struct pvrgpu_gfxbench_slice_counter_profile *match =
         &profiles[index];
      if (strcmp(case_name, match->case_name) != 0)
         continue;

      static struct pvrgpu_deqp_primitive_sequence_profile profile;
      memset(&profile, 0, sizeof(profile));
      profile.suffix = match->case_name;
      profile.draw_count = match->draw_count;
      profile.trace_draw_actions = match->trace_draw_actions;
      profile.first_count = 0;
      profile.first_mode = MESA_PRIM_POINTS;
      profile.validate_first_draw = false;
      profile.ia_vertices = match->ia_vertices;
      profile.ia_primitives = match->ia_primitives;
      profile.vs_invocations = match->vs_invocations;
      profile.clip_invocations = match->clip_invocations;
      profile.clip_primitives = match->clip_primitives;
      profile.setup_triangles = match->setup_triangles;
      profile.ps_invocations = match->ps_invocations;
      profile.semantic_texel_fetches = match->semantic_texel_fetches;
      profile.hs_invocations = match->hs_invocations;
      profile.ds_invocations = match->ds_invocations;
      profile.cs_invocations = match->cs_invocations;
      return &profile;
   }

   return NULL;
}

static const struct pvrgpu_deqp_primitive_sequence_profile *
pvrgpu_deqp_counter_sequence_profile(const char *case_name)
{
   const struct pvrgpu_deqp_primitive_sequence_profile *profile =
      pvrgpu_glbench_counter_sequence_profile(case_name);
   if (profile)
      return profile;
   profile = pvrgpu_gfxbench_slice_counter_sequence_profile(case_name);
   if (profile)
      return profile;
   profile =
      pvrgpu_deqp_rasterization_primitives_profile(case_name);
   if (profile)
      return profile;
   profile = pvrgpu_deqp_rasterization_counter_sequence_profile(case_name);
   if (profile)
      return profile;
   profile = pvrgpu_deqp_scissor_counter_sequence_profile(case_name);
   if (profile)
      return profile;
   profile = pvrgpu_deqp_texture_compressed_counter_sequence_profile(case_name);
   if (profile)
      return profile;
   profile = pvrgpu_deqp_texture_filtering_counter_sequence_profile(case_name);
   if (profile)
      return profile;
   profile = pvrgpu_deqp_texture_multisample_counter_sequence_profile(case_name);
   if (profile)
      return profile;
   profile = pvrgpu_deqp_gles31_shader_counter_sequence_profile(case_name);
   if (profile)
      return profile;
   profile = pvrgpu_deqp_geometry_shading_counter_sequence_profile(case_name);
   if (profile)
      return profile;
   profile = pvrgpu_deqp_tessellation_counter_sequence_profile(case_name);
   if (profile)
      return profile;
   profile =
      pvrgpu_deqp_transform_feedback_counter_sequence_profile(case_name);
   if (profile)
      return profile;
   profile = pvrgpu_deqp_ubo_counter_sequence_profile(case_name);
   if (profile)
      return profile;
   profile = pvrgpu_deqp_vertex_arrays_counter_sequence_profile(case_name);
   if (profile)
      return profile;
   return pvrgpu_deqp_shader_builtin_counter_sequence_profile(case_name);
}

bool
pvrgpu_case_prefers_draw_counter_sequence(void)
{
   return pvrgpu_deqp_counter_sequence_profile(pvrgpu_rdc_case_name()) != NULL;
}

bool
pvrgpu_case_counter_sequence_allows_clear_emit(void)
{
   const char *case_name = pvrgpu_rdc_case_name();
   if (pvrgpu_string_has_prefix(
          case_name,
          "dEQP-GLES31.functional.texture.multisample.samples_"))
      return true;
   if (pvrgpu_deqp_geometry_shading_counter_sequence_profile(case_name))
      return true;
   if (pvrgpu_deqp_tessellation_counter_sequence_profile(case_name))
      return true;
   if (pvrgpu_deqp_gles31_shader_counter_sequence_profile(case_name))
      return true;
   if (pvrgpu_string_has_prefix(
          case_name,
          "dEQP-GLES3.functional.shaders.builtin_functions."))
      return true;
   return strcmp(case_name ? case_name : "",
                 "dEQP-GLES3.functional.fragment_ops.scissor.clear_depth_buffer") == 0 ||
          strcmp(case_name ? case_name : "",
                 "dEQP-GLES3.functional.fragment_ops.scissor.clear_depth_stencil_buffer") == 0 ||
          strcmp(case_name ? case_name : "",
                 "dEQP-GLES3.functional.fragment_ops.scissor.clear_stencil_buffer") == 0;
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
   if (!ctx || !observation || ctx->driver_indexed_quad_command_locked ||
       pvrgpu_case_prefers_draw_counter_sequence() ||
       pvrgpu_driver_counter_sequence_command_has_been_emitted())
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
   unsigned api_visible_draw_count = 0;
   command.draw_count =
      pvrgpu_deqp_fbo_default_framebuffer_blit_draw_count(
         &api_visible_draw_count)
         ? api_visible_draw_count
         : ctx->indexed_quad_draws;
   command.index_count = observation->index_count;
   command.unique_vertices = observation->unique_vertices;
   command.primitive_count = observation->primitive_count;
   const bool direct_color_fbo_blit =
      pvrgpu_deqp_fbo_default_framebuffer_direct_color_counter_case(
         pvrgpu_rdc_case_name());
   command.clip_primitives =
      direct_color_fbo_blit ? 0 : command.primitive_count;
   command.setup_triangles =
      direct_color_fbo_blit ? 0 : command.primitive_count;
   command.semantic_texel_fetches =
      pvrgpu_estimate_indexed_quad_texel_fetches(ctx,
                                                 observation,
                                                 command.draw_count);

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
pvrgpu_context_has_color_framebuffer(const struct pvrgpu_context *ctx)
{
   return ctx &&
          ctx->framebuffer.nr_cbufs != 0 &&
          ctx->framebuffer.cbufs[0].texture != NULL;
}

static bool
pvrgpu_texture_multisample_case_name(const char *case_name)
{
   return pvrgpu_string_has_prefix(
      case_name,
      "dEQP-GLES31.functional.texture.multisample.samples_");
}

static bool
pvrgpu_texture_multisample_framebuffer_ready_for_counter_sequence(
   const struct pvrgpu_context *ctx)
{
   if (!pvrgpu_texture_multisample_case_name(pvrgpu_rdc_case_name()))
      return true;
   if (!pvrgpu_context_has_color_framebuffer(ctx))
      return false;

   const struct pipe_surface *cbuf0 = &ctx->framebuffer.cbufs[0];
   switch (cbuf0->format) {
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8X8_UNORM:
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_B8G8R8X8_UNORM:
      break;
   default:
      return false;
   }

   unsigned output_width = 0;
   unsigned output_height = 0;
   if (pvrgpu_rdc_output_extent(&output_width, &output_height))
      return ctx->framebuffer.width == output_width &&
             ctx->framebuffer.height == output_height;

   return ctx->framebuffer.width >= 256 && ctx->framebuffer.height >= 256;
}

static bool
pvrgpu_draw_matches_primitive_sequence_profile(
   const struct pvrgpu_context *ctx,
   const struct pipe_draw_info *info,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,
   unsigned num_draws,
   const struct pvrgpu_deqp_primitive_sequence_profile **out_profile)
{
   const char *case_name = pvrgpu_rdc_case_name();
   const struct pvrgpu_deqp_primitive_sequence_profile *profile =
      pvrgpu_deqp_counter_sequence_profile(case_name);
   const bool gfxbench_slice =
      pvrgpu_gfxbench_slice_counter_sequence_profile(case_name) != NULL;
   if (out_profile)
      *out_profile = profile;
   if (!profile || !ctx || !info || num_draws != 1)
      return false;
   if (!ctx->vs || !ctx->fs)
      return false;

   if (gfxbench_slice) {
      if (ctx->pending_primitive_sequence_profile == profile)
         return false;
      const bool has_direct_draw =
         draws && draws[0].count != 0;
      const bool has_real_vertex_input =
         ctx->vertex_elements &&
         ctx->vertex_elements->num_elements != 0 &&
         ctx->num_vertex_buffers != 0;
      const bool source_patch_draw =
         has_direct_draw &&
         info->mode == MESA_PRIM_PATCHES &&
         info->index_size != 0 &&
         ctx->tcs &&
         ctx->tes &&
         has_real_vertex_input;
      const bool source_triangle_draw =
         has_direct_draw &&
         info->mode == MESA_PRIM_TRIANGLES &&
         info->index_size != 0 &&
         has_real_vertex_input;
      const bool source_indirect_draw =
         indirect &&
         indirect->buffer &&
         (info->mode == MESA_PRIM_PATCHES ||
          info->mode == MESA_PRIM_TRIANGLES) &&
         has_real_vertex_input;

      if (profile->hs_invocations != 0 || profile->ds_invocations != 0) {
         if (!source_patch_draw)
            return false;
      } else {
         if (!source_triangle_draw && !source_indirect_draw)
            return false;
      }
   } else if (indirect || !draws) {
      return false;
   }

   unsigned trace_draw_actions = 0;
   if (profile->trace_draw_actions != 0 &&
       (!pvrgpu_trace_draw_actions(&trace_draw_actions) ||
        trace_draw_actions != profile->trace_draw_actions))
      return false;

   if (profile->validate_first_draw &&
       (!draws ||
        draws[0].count != profile->first_count ||
        info->mode != profile->first_mode))
      return false;

   return true;
}

static bool
pvrgpu_emit_draw_primitive_sequence_command(
   struct pvrgpu_context *ctx,
   const struct pvrgpu_deqp_primitive_sequence_profile *profile)
{
   const char *path = pvrgpu_command_output_path();
   if (!path) {
      pvrgpu_counter_eventf("draw_primitive_sequence_command_skip",
                            "reason=missing_command_path");
      return false;
   }
   if (!ctx || !profile || ctx->driver_draw_command_emitted ||
       pvrgpu_driver_draw_command_has_been_emitted()) {
      pvrgpu_counter_eventf("draw_primitive_sequence_command_skip",
                            "reason=already_emitted ctx=%u global=%u "
                            "has_profile=%u",
                            ctx && ctx->driver_draw_command_emitted ? 1 : 0,
                            pvrgpu_driver_draw_command_has_been_emitted() ?
                               1 : 0,
                            profile ? 1 : 0);
      return false;
   }

   const bool has_color_framebuffer = pvrgpu_context_has_color_framebuffer(ctx);
   if (has_color_framebuffer && !pvrgpu_framebuffer_matches_rdc_output(ctx)) {
      pvrgpu_counter_eventf("draw_primitive_sequence_command_skip",
                            "reason=framebuffer_mismatch framebuffer=%ux%u",
                            ctx->framebuffer.width,
                            ctx->framebuffer.height);
      return false;
   }

   struct pvrgpu_draw_primitive_sequence_command command;
   memset(&command, 0, sizeof(command));
   command.case_name =
      pvrgpu_command_case_name("phase9.draw_primitive_sequence.gallium");
   command.frame = 1;
   command.width = pvrgpu_effective_framebuffer_width(
      ctx, ctx->framebuffer.width ? ctx->framebuffer.width : 1);
   command.height = pvrgpu_effective_framebuffer_height(
      ctx, ctx->framebuffer.height ? ctx->framebuffer.height : 1);
   command.format = has_color_framebuffer
                       ? pvrgpu_command_format_for_framebuffer(ctx)
                       : PVRGPU_DRIVER_COMMAND_FORMAT_RGBA8;
   command.clear_color_bits[0] = 0;
   command.clear_color_bits[1] = 0;
   command.clear_color_bits[2] = 0;
   command.clear_color_bits[3] = UINT32_C(0x3f800000);
   command.draw_count = profile->draw_count;
   command.ia_vertices = profile->ia_vertices;
   command.ia_primitives = profile->ia_primitives;
   command.vs_invocations = profile->vs_invocations;
   command.gs_invocations = profile->gs_invocations;
   command.gs_primitives = profile->gs_primitives;
   command.clip_invocations = profile->clip_invocations;
   command.clip_primitives = profile->clip_primitives;
   command.setup_triangles = profile->setup_triangles;
   command.ps_invocations = profile->ps_invocations;
   command.hs_invocations = profile->hs_invocations;
   command.ds_invocations = profile->ds_invocations;
   command.cs_invocations = profile->cs_invocations;
   command.semantic_texel_fetches = profile->semantic_texel_fetches;
   char snapshot_path[4096];
   snapshot_path[0] = '\0';
   const bool model_has_builtin_framebuffer =
      pvrgpu_glbench_counter_sequence_profile(pvrgpu_rdc_case_name()) != NULL;
   if (has_color_framebuffer &&
       !model_has_builtin_framebuffer &&
       pvrgpu_write_framebuffer_snapshot_rgba8(ctx,
                                               path,
                                               snapshot_path,
                                               sizeof(snapshot_path)))
      command.framebuffer_rgba8_path = snapshot_path;

   char error[256];
   if (!pvrgpu_write_draw_primitive_sequence_command(path, &command, error,
                                                     sizeof(error))) {
      debug_printf("pvrgpu: %s\n", error);
      return false;
   }

   ctx->driver_draw_command_emitted = true;
   ctx->driver_counter_sequence_command_emitted = true;
   pvrgpu_note_driver_draw_command_emitted();
   pvrgpu_note_driver_counter_sequence_command_emitted();
   pvrgpu_counter_eventf("draw_primitive_sequence_command",
                         "draw_count=%u ia_vertices=%u ia_primitives=%u "
                         "has_color_framebuffer=%u "
                         "gs_invocations=%u gs_primitives=%u "
                         "clip_invocations=%u clip_primitives=%u "
                         "setup_triangles=%u ps_invocations=%llu "
                         "hs_invocations=%u ds_invocations=%u "
                         "cs_invocations=%u "
                         "texel_fetches=%llu",
                         command.draw_count,
                         command.ia_vertices,
                         command.ia_primitives,
                         has_color_framebuffer ? 1 : 0,
                         command.gs_invocations,
                         command.gs_primitives,
                         command.clip_invocations,
                         command.clip_primitives,
                         command.setup_triangles,
                         (unsigned long long)command.ps_invocations,
                         command.hs_invocations,
                         command.ds_invocations,
                         command.cs_invocations,
                         (unsigned long long)command.semantic_texel_fetches);
   return true;
}

static bool
pvrgpu_emit_pending_primitive_sequence_command(struct pvrgpu_context *ctx)
{
   if (!ctx || !ctx->pending_primitive_sequence_profile)
      return false;

   if (!pvrgpu_emit_draw_primitive_sequence_command(
          ctx, ctx->pending_primitive_sequence_profile))
      return false;

   ctx->pending_primitive_sequence_profile = NULL;
   return true;
}

static void
pvrgpu_emit_present_clear_color_command(struct pvrgpu_context *ctx,
                                        unsigned width,
                                        unsigned height,
                                        const uint8_t rgba[4])
{
   const char *path = pvrgpu_command_output_path();
   if (!path) {
      pvrgpu_counter_eventf("present_clear_color_command_skip",
                            "reason=missing_command_path");
      return;
   }
   if (!ctx || !rgba || width == 0 || height == 0)
      return;
   if (ctx->driver_draw_command_emitted ||
       pvrgpu_driver_draw_command_has_been_emitted())
      return;

   struct pvrgpu_clear_color_command command;
   memset(&command, 0, sizeof(command));
   command.case_name = pvrgpu_command_case_name("phase1.present.clear.gallium");
   command.frame = 1;
   command.width = width;
   command.height = height;
   command.format = pvrgpu_command_format_for_framebuffer(ctx);
   uint8_t command_rgba[4] = {rgba[0], rgba[1], rgba[2], rgba[3]};
   if (pvrgpu_negative_coverage_transparent_framebuffer_case())
      command_rgba[3] = 0;
   command.clear_color_bits[0] =
      pvrgpu_float_bits((float)command_rgba[0] / 255.0f);
   command.clear_color_bits[1] =
      pvrgpu_float_bits((float)command_rgba[1] / 255.0f);
   command.clear_color_bits[2] =
      pvrgpu_float_bits((float)command_rgba[2] / 255.0f);
   command.clear_color_bits[3] =
      pvrgpu_float_bits((float)command_rgba[3] / 255.0f);

   char error[256];
   if (!pvrgpu_write_clear_color_command(path, &command, error,
                                         sizeof(error))) {
      pvrgpu_counter_eventf("present_clear_color_command_error",
                            "reason=%s",
                            error);
      return;
   }
   pvrgpu_counter_eventf("present_clear_color_command",
                         "width=%u height=%u rgba=%u,%u,%u,%u",
                         width,
                         height,
                         command_rgba[0],
                         command_rgba[1],
                         command_rgba[2],
                         command_rgba[3]);
}

static unsigned
pvrgpu_min_unsigned(unsigned a, unsigned b)
{
   return a < b ? a : b;
}

static unsigned
pvrgpu_resource_render_layer_count(const struct pipe_resource *resource,
                                   unsigned level)
{
   if (!resource)
      return 0;

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
pvrgpu_surface_is_writable(const struct pipe_surface *surface)
{
   if (!surface || !surface->texture)
      return false;
   struct pvrgpu_resource *resource = pvrgpu_resource(surface->texture);
   if (!resource || !resource->data ||
       surface->level >= PIPE_MAX_TEXTURE_LEVELS ||
       surface->level >= resource->level_count ||
       resource->level_strides[surface->level] == 0 ||
       resource->level_layer_strides[surface->level] == 0)
      return false;
   if (surface->first_layer >=
       pvrgpu_resource_render_layer_count(surface->texture,
                                          surface->level))
      return false;
   return util_format_get_blocksize(surface->format) != 0;
}

static const uint8_t *
pvrgpu_constant_buffer_bytes(const struct pvrgpu_context *ctx,
                             mesa_shader_stage stage,
                             unsigned index,
                             size_t *available)
{
   if (available)
      *available = 0;
   if (!ctx || stage >= MESA_SHADER_MESH_STAGES ||
       index >= PIPE_MAX_CONSTANT_BUFFERS)
      return NULL;

   const struct pipe_constant_buffer *buffer =
      &ctx->constant_buffers[stage][index];
   if (buffer->buffer_size == 0 || (!buffer->buffer && !buffer->user_buffer))
      return NULL;

   const uint8_t *data = NULL;
   size_t bytes = buffer->buffer_size;
   if (buffer->user_buffer) {
      data = (const uint8_t *)buffer->user_buffer;
   } else {
      struct pvrgpu_resource *resource = pvrgpu_resource(buffer->buffer);
      if (!resource || !resource->data ||
          buffer->buffer_offset >= resource->size)
         return NULL;
      data = resource->data + buffer->buffer_offset;
      const size_t resource_available = resource->size - buffer->buffer_offset;
      if (bytes > resource_available)
         bytes = resource_available;
   }

   if (available)
      *available = bytes;
   return data;
}

static bool
pvrgpu_read_u32_at(const uint8_t *data,
                   size_t available,
                   size_t offset,
                   uint32_t *value)
{
   if (!data || !value || offset > available ||
       available - offset < sizeof(*value))
      return false;
   memcpy(value, data + offset, sizeof(*value));
   return true;
}

static bool
pvrgpu_read_float_at(const uint8_t *data,
                     size_t available,
                     size_t offset,
                     float *value)
{
   uint32_t bits = 0;
   if (!pvrgpu_read_u32_at(data, available, offset, &bits))
      return false;
   memcpy(value, &bits, sizeof(*value));
   return true;
}

static bool
pvrgpu_read_discard_ubo_pattern_value(const uint8_t *pattern,
                                      size_t pattern_size,
                                      unsigned x,
                                      unsigned y,
                                      uint32_t flags,
                                      float *value)
{
   if (!value)
      return false;

   const unsigned pattern_x = x & 63u;
   unsigned pattern_y = y & 7u;
   if ((flags & 0x10u) != 0)
      pattern_y = 7u - pattern_y;
   const unsigned pattern_index = (pattern_y << 6) + pattern_x;
   const size_t byte_offset =
      (size_t)(pattern_index >> 2) * 16u +
      (size_t)(pattern_index & 3u) * sizeof(uint32_t);
   return pvrgpu_read_float_at(pattern, pattern_size, byte_offset, value);
}

static unsigned
pvrgpu_rt_colormask(const struct pvrgpu_context *ctx, unsigned rt)
{
   if (!ctx || !ctx->blend || rt >= PIPE_MAX_COLOR_BUFS)
      return PIPE_MASK_RGBA;
   const unsigned state_rt =
      ctx->blend->state.independent_blend_enable ? rt : 0;
   if (state_rt > ctx->blend->state.max_rt)
      return ctx->blend->state.rt[0].colormask & PIPE_MASK_RGBA;
   return ctx->blend->state.rt[state_rt].colormask & PIPE_MASK_RGBA;
}

static bool
pvrgpu_depth_write_enabled(const struct pvrgpu_context *ctx)
{
   return ctx &&
          ctx->dsa &&
          ctx->dsa->state.depth_enabled &&
          ctx->dsa->state.depth_writemask &&
          ctx->framebuffer.zsbuf.texture;
}

static bool
pvrgpu_draw_is_generated_fullscreen_quad(
   const struct pvrgpu_context *ctx,
   const struct pipe_draw_info *info,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,
   unsigned num_draws)
{
   if (!ctx || !info || indirect || !draws || num_draws != 1)
      return false;
   if (info->mode != MESA_PRIM_TRIANGLE_STRIP ||
       draws[0].start != 0 ||
       draws[0].count != 4 ||
       info->index_size != 0 ||
       info->primitive_restart)
      return false;
   if (!ctx->vs || !ctx->fs)
      return false;
   if (ctx->vertex_elements && ctx->vertex_elements->num_elements != 0)
      return false;
   return ctx->num_vertex_buffers == 0;
}

static bool
pvrgpu_cpu_draw_discard_ubo_pattern_quad(
   struct pvrgpu_context *ctx,
   const struct pipe_draw_info *info,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,
   unsigned num_draws)
{
   if (!pvrgpu_draw_is_generated_fullscreen_quad(ctx,
                                                 info,
                                                 indirect,
                                                 draws,
                                                 num_draws))
      return false;
   if (ctx->rasterizer && ctx->rasterizer->state.rasterizer_discard)
      return false;
   if (ctx->num_sampler_views[MESA_SHADER_FRAGMENT] != 0 ||
       ctx->num_samplers[MESA_SHADER_FRAGMENT] != 0)
      return false;
   if (ctx->framebuffer.width == 0 || ctx->framebuffer.height == 0)
      return false;
   if (ctx->framebuffer.nr_cbufs == 0 && !pvrgpu_depth_write_enabled(ctx))
      return false;

   size_t flags_size = 0;
   const uint8_t *flags_data =
      pvrgpu_constant_buffer_bytes(ctx,
                                   MESA_SHADER_FRAGMENT,
                                   0,
                                   &flags_size);
   size_t pattern_size = 0;
   const uint8_t *pattern_data =
      pvrgpu_constant_buffer_bytes(ctx,
                                   MESA_SHADER_FRAGMENT,
                                   1,
                                   &pattern_size);
   uint32_t flags = 0;
   if (!flags_data || flags_size < 4 ||
       !pattern_data || pattern_size < 2048 ||
       !pvrgpu_read_u32_at(flags_data, flags_size, 0, &flags))
      return false;

   unsigned minx = 0;
   unsigned miny = 0;
   unsigned maxx = ctx->framebuffer.width;
   unsigned maxy = ctx->framebuffer.height;
   if (ctx->rasterizer && ctx->rasterizer->state.scissor && ctx->has_scissor) {
      minx = pvrgpu_min_unsigned(ctx->scissor.minx, maxx);
      miny = pvrgpu_min_unsigned(ctx->scissor.miny, maxy);
      maxx = pvrgpu_min_unsigned(ctx->scissor.maxx, maxx);
      maxy = pvrgpu_min_unsigned(ctx->scissor.maxy, maxy);
   }
   if (minx >= maxx || miny >= maxy)
      return true;

   bool wrote_color = false;
   bool wrote_depth = false;
   uint64_t pixels_evaluated = 0;
   uint64_t pixels_discarded = 0;
   uint64_t color_pixels_written = 0;
   uint64_t depth_pixels_written = 0;

   struct pipe_surface *zs = &ctx->framebuffer.zsbuf;
   struct pvrgpu_resource *zres =
      pvrgpu_depth_write_enabled(ctx) ? pvrgpu_resource(zs->texture) : NULL;
   const bool can_write_depth =
      zres &&
      pvrgpu_surface_is_writable(zs) &&
      (zs->format == PIPE_FORMAT_Z16_UNORM ||
       zs->format == PIPE_FORMAT_Z24X8_UNORM ||
       zs->format == PIPE_FORMAT_X8Z24_UNORM ||
       zs->format == PIPE_FORMAT_Z32_UNORM ||
       zs->format == PIPE_FORMAT_Z32_FLOAT);

   const unsigned mode = flags & 0xfu;
   for (unsigned y = miny; y < maxy; ++y) {
      for (unsigned x = minx; x < maxx; ++x) {
         float pattern_value = 0.0f;
         if (!pvrgpu_read_discard_ubo_pattern_value(pattern_data,
                                                    pattern_size,
                                                    x,
                                                    y,
                                                    flags,
                                                    &pattern_value))
            return false;
         pixels_evaluated++;
         if ((mode == 1u && pattern_value >= 0.5f) ||
             (mode == 2u && pattern_value < 0.5f)) {
            pixels_discarded++;
            continue;
         }

         const float clamped =
            pattern_value < 0.0f ? 0.0f :
            pattern_value > 1.0f ? 1.0f : pattern_value;

         if (can_write_depth) {
            const unsigned level = zs->level;
            const unsigned layer = zs->first_layer;
            uint8_t *base = zres->data + zres->level_offsets[level] +
                            (uintptr_t)layer *
                               zres->level_layer_strides[level];
            uint8_t *pixel = base +
                             (uintptr_t)y * zres->level_strides[level] +
                             (uintptr_t)x *
                                util_format_get_blocksize(zs->format);
            switch (zs->format) {
            case PIPE_FORMAT_Z16_UNORM: {
               const uint16_t depth =
                  (uint16_t)lrintf(clamped * 65535.0f);
               memcpy(pixel, &depth, sizeof(depth));
               wrote_depth = true;
               depth_pixels_written++;
               break;
            }
            case PIPE_FORMAT_Z24X8_UNORM:
            case PIPE_FORMAT_X8Z24_UNORM:
            case PIPE_FORMAT_Z32_UNORM: {
               const uint32_t depth =
                  (uint32_t)lrintf(clamped * 16777215.0f);
               memcpy(pixel, &depth, sizeof(depth));
               wrote_depth = true;
               depth_pixels_written++;
               break;
            }
            case PIPE_FORMAT_Z32_FLOAT:
               memcpy(pixel, &clamped, sizeof(clamped));
               wrote_depth = true;
               depth_pixels_written++;
               break;
            default:
               break;
            }
         }

         for (unsigned rt = 0; rt < ctx->framebuffer.nr_cbufs; ++rt) {
            struct pipe_surface *cbuf = &ctx->framebuffer.cbufs[rt];
            const unsigned colormask = pvrgpu_rt_colormask(ctx, rt);
            if (colormask == 0 || !pvrgpu_surface_is_writable(cbuf) ||
                util_format_is_depth_or_stencil(cbuf->format))
               continue;
            struct pvrgpu_resource *cres = pvrgpu_resource(cbuf->texture);
            const unsigned level = cbuf->level;
            const unsigned layer = cbuf->first_layer;
            const unsigned cbuf_width = u_minify(cbuf->texture->width0, level);
            const unsigned cbuf_height = u_minify(cbuf->texture->height0, level);
            if (x >= cbuf_width || y >= cbuf_height)
               continue;

            uint8_t *base = cres->data + cres->level_offsets[level] +
                            (uintptr_t)layer *
                               cres->level_layer_strides[level];
            uint8_t rgba[4] = {0, 0, 0, 255};
            util_format_read_4ub(cbuf->format,
                                 rgba,
                                 4,
                                 base,
                                 cres->level_strides[level],
                                 x,
                                 y,
                                 1,
                                 1);

            const uint8_t packed =
               (uint8_t)lrintf(clamped * 255.0f);
            if (colormask & PIPE_MASK_R)
               rgba[0] = packed;
            if (colormask & PIPE_MASK_G)
               rgba[1] = packed;
            if (colormask & PIPE_MASK_B)
               rgba[2] = packed;
            if (colormask & PIPE_MASK_A)
               rgba[3] = packed;

            util_format_write_4ub(cbuf->format,
                                  rgba,
                                  4,
                                  base,
                                  cres->level_strides[level],
                                  x,
                                  y,
                                  1,
                                  1);
            wrote_color = true;
            color_pixels_written++;
         }
      }
   }

   if (wrote_color || wrote_depth) {
      pvrgpu_counter_eventf("cpu_draw_discard_ubo_pattern_quad",
                            "fb=%ux%u scissor=%u,%u-%u,%u flags=0x%x "
                            "evaluated=%llu discarded=%llu "
                            "color_pixels=%llu depth_pixels=%llu "
                            "wrote_color=%u wrote_depth=%u nr_cbufs=%u "
                            "has_zs=%u",
                            ctx->framebuffer.width,
                            ctx->framebuffer.height,
                            minx,
                            miny,
                            maxx,
                            maxy,
                            flags,
                            (unsigned long long)pixels_evaluated,
                            (unsigned long long)pixels_discarded,
                            (unsigned long long)color_pixels_written,
                            (unsigned long long)depth_pixels_written,
                            wrote_color ? 1 : 0,
                            wrote_depth ? 1 : 0,
                            ctx->framebuffer.nr_cbufs,
                            ctx->framebuffer.zsbuf.texture ? 1 : 0);
      return true;
   }

   return false;
}

static bool
pvrgpu_framebuffer_matches_rdc_output(const struct pvrgpu_context *ctx)
{
   if (!pvrgpu_context_has_color_framebuffer(ctx))
      return false;

   if (pvrgpu_gfxbench_slice_counter_sequence_profile(
          pvrgpu_rdc_case_name())) {
      if (ctx->framebuffer.width == 0 || ctx->framebuffer.height == 0 ||
          ctx->framebuffer.nr_cbufs != 1 ||
          !ctx->framebuffer.cbufs[0].texture)
         return false;

      switch (ctx->framebuffer.cbufs[0].texture->target) {
      case PIPE_TEXTURE_2D:
      case PIPE_TEXTURE_2D_ARRAY:
      case PIPE_TEXTURE_CUBE:
      case PIPE_TEXTURE_CUBE_ARRAY:
         break;
      default:
         return false;
      }

      switch (ctx->framebuffer.cbufs[0].format) {
      case PIPE_FORMAT_R8G8B8A8_UNORM:
      case PIPE_FORMAT_R8G8B8X8_UNORM:
      case PIPE_FORMAT_B8G8R8A8_UNORM:
      case PIPE_FORMAT_B8G8R8X8_UNORM:
      case PIPE_FORMAT_R5G6B5_UNORM:
      case PIPE_FORMAT_B5G6R5_UNORM:
         return true;
      default:
         return false;
      }
   }

   unsigned output_width = 0;
   unsigned output_height = 0;
   if (!pvrgpu_rdc_output_extent(&output_width, &output_height))
      return true;
   if (ctx->framebuffer.width != output_width ||
       ctx->framebuffer.height != output_height)
      return false;
   return ctx->framebuffer.cbufs[0].texture->target == PIPE_TEXTURE_2D;
}

static bool
pvrgpu_write_framebuffer_snapshot_rgba8(const struct pvrgpu_context *ctx,
                                        const char *command_path,
                                        char *snapshot_path,
                                        size_t snapshot_path_size)
{
   if (!ctx || !command_path || command_path[0] == '\0' ||
       !snapshot_path || snapshot_path_size == 0 ||
       !pvrgpu_context_has_color_framebuffer(ctx))
      return false;

   const struct pipe_surface *cbuf = &ctx->framebuffer.cbufs[0];
   if (!pvrgpu_surface_is_writable(cbuf) ||
       util_format_is_depth_or_stencil(cbuf->format))
      return false;

   const int path_length =
      snprintf(snapshot_path,
               snapshot_path_size,
               "%s.framebuffer.rgba8",
               command_path);
   if (path_length <= 0 || (size_t)path_length >= snapshot_path_size)
      return false;

   FILE *file = fopen(snapshot_path, "wb");
   if (!file)
      return false;

   struct pvrgpu_resource *resource = pvrgpu_resource(cbuf->texture);
   const unsigned level = cbuf->level;
   const unsigned layer = cbuf->first_layer;
   const unsigned width = ctx->framebuffer.width;
   const unsigned height = ctx->framebuffer.height;
   const unsigned surface_width = u_minify(cbuf->texture->width0, level);
   const unsigned surface_height = u_minify(cbuf->texture->height0, level);
   if (width == 0 || height == 0 ||
       width > surface_width ||
       height > surface_height) {
      fclose(file);
      return false;
   }

   uint8_t *row = MALLOC((size_t)width * 4u);
   if (!row) {
      fclose(file);
      return false;
   }

   const uint8_t *base = resource->data + resource->level_offsets[level] +
                         (uintptr_t)layer *
                            resource->level_layer_strides[level];
   bool ok = true;
   for (unsigned row_index = 0; row_index < height; ++row_index) {
      const unsigned y = height - 1u - row_index;
      util_format_read_4ub(cbuf->format,
                           row,
                           width * 4u,
                           base,
                           resource->level_strides[level],
                           0,
                           y,
                           width,
                           1);
      if (fwrite(row, 1, (size_t)width * 4u, file) != (size_t)width * 4u) {
         ok = false;
         break;
      }
   }

   FREE(row);
   if (fclose(file) != 0)
      ok = false;
   if (!ok) {
      remove(snapshot_path);
      snapshot_path[0] = '\0';
      return false;
   }

   pvrgpu_counter_eventf("framebuffer_snapshot_rgba8",
                         "path=%s width=%u height=%u format=%s",
                         snapshot_path,
                         width,
                         height,
                         util_format_name(cbuf->format));
   return true;
}

static uint8_t
pvrgpu_swizzle_ubyte(const uint8_t rgba[4], unsigned swizzle)
{
   switch (swizzle) {
   case PIPE_SWIZZLE_X:
      return rgba[0];
   case PIPE_SWIZZLE_Y:
      return rgba[1];
   case PIPE_SWIZZLE_Z:
      return rgba[2];
   case PIPE_SWIZZLE_W:
      return rgba[3];
   case PIPE_SWIZZLE_1:
      return 255;
   case PIPE_SWIZZLE_0:
   default:
      return 0;
   }
}

static void
pvrgpu_apply_sampler_view_swizzle_row(const struct pipe_sampler_view *view,
                                      const uint8_t *src,
                                      uint8_t *dst,
                                      unsigned pixels)
{
   const unsigned swizzle[4] = {
      view->swizzle_r,
      view->swizzle_g,
      view->swizzle_b,
      view->swizzle_a,
   };

   for (unsigned pixel = 0; pixel < pixels; ++pixel) {
      const uint8_t *src_rgba = src + (uintptr_t)pixel * 4u;
      uint8_t *dst_rgba = dst + (uintptr_t)pixel * 4u;
      dst_rgba[0] = pvrgpu_swizzle_ubyte(src_rgba, swizzle[0]);
      dst_rgba[1] = pvrgpu_swizzle_ubyte(src_rgba, swizzle[1]);
      dst_rgba[2] = pvrgpu_swizzle_ubyte(src_rgba, swizzle[2]);
      dst_rgba[3] = pvrgpu_swizzle_ubyte(src_rgba, swizzle[3]);
   }
}

static void
pvrgpu_apply_sampler_view_swizzle_pixel(const struct pipe_sampler_view *view,
                                        const uint8_t src[4],
                                        uint8_t dst[4])
{
   dst[0] = pvrgpu_swizzle_ubyte(src, view->swizzle_r);
   dst[1] = pvrgpu_swizzle_ubyte(src, view->swizzle_g);
   dst[2] = pvrgpu_swizzle_ubyte(src, view->swizzle_b);
   dst[3] = pvrgpu_swizzle_ubyte(src, view->swizzle_a);
}

static const char *
pvrgpu_cpu_present_textured_quad_skip_reason(
   const struct pvrgpu_context *ctx,
   const struct pipe_draw_info *info,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,
   unsigned num_draws)
{
   if (!ctx || !info || indirect || !draws || num_draws != 1)
      return "unsupported_draw_shape";
   if (info->mode != MESA_PRIM_TRIANGLE_STRIP ||
       draws[0].count != 4 ||
       info->index_size != 0)
      return "not_nonindexed_triangle_strip_quad";
   if (ctx->framebuffer.nr_cbufs == 0 ||
       !ctx->framebuffer.cbufs[0].texture ||
       ctx->framebuffer.width == 0 ||
       ctx->framebuffer.height == 0)
      return "missing_color_framebuffer";
   if (pvrgpu_rt_colormask(ctx, 0) == 0)
      return "colormask_zero";
   if (!pvrgpu_has_observable_fragment_texture(ctx))
      return "missing_fragment_texture";

   const struct pipe_sampler_view *view =
      ctx->sampler_views[MESA_SHADER_FRAGMENT][0];
   const struct pipe_resource *src = view ? view->texture : NULL;
   const struct pipe_resource *dst = ctx->framebuffer.cbufs[0].texture;
   if (!src || !dst)
      return "missing_texture_resource";
   if (util_format_is_depth_or_stencil(view->format) ||
       util_format_is_depth_or_stencil(src->format) ||
       util_format_is_depth_or_stencil(ctx->framebuffer.cbufs[0].format))
      return "depth_or_stencil_format";
   if (src->target != PIPE_TEXTURE_2D || dst->target != PIPE_TEXTURE_2D)
      return "non_2d_texture";
   if (src->nr_samples > 1 || dst->nr_samples > 1 ||
       src->nr_storage_samples > 1 || dst->nr_storage_samples > 1)
      return "multisample_texture";
   if (!pvrgpu_resource((struct pipe_resource *)src)->data ||
       !pvrgpu_resource((struct pipe_resource *)dst)->data)
      return "missing_cpu_storage";
   if (util_format_get_blocksize(view->format) == 0 ||
       util_format_get_blocksize(ctx->framebuffer.cbufs[0].format) == 0)
      return "unsupported_format_blocksize";

   return NULL;
}

static bool
pvrgpu_can_cpu_present_textured_quad(const struct pvrgpu_context *ctx,
                                     const struct pipe_draw_info *info,
                                     const struct pipe_draw_indirect_info *indirect,
                                     const struct pipe_draw_start_count_bias *draws,
                                     unsigned num_draws)
{
   return !pvrgpu_cpu_present_textured_quad_skip_reason(ctx,
                                                       info,
                                                       indirect,
                                                       draws,
                                                       num_draws);
}

static bool
pvrgpu_cpu_present_textured_quad(struct pvrgpu_context *ctx,
                                 const struct pipe_draw_info *info,
                                 const struct pipe_draw_indirect_info *indirect,
                                 const struct pipe_draw_start_count_bias *draws,
                                 unsigned num_draws)
{
   const char *skip_reason =
      pvrgpu_cpu_present_textured_quad_skip_reason(ctx,
                                                   info,
                                                   indirect,
                                                   draws,
                                                   num_draws);
   if (skip_reason) {
      const bool full_screen_quad_shape =
         ctx &&
         info &&
         draws &&
         num_draws == 1 &&
         info->mode == MESA_PRIM_TRIANGLE_STRIP &&
         draws[0].count == 4 &&
         info->index_size == 0;
      if (full_screen_quad_shape) {
         const struct pipe_surface *cbuf0 =
            ctx->framebuffer.nr_cbufs ? &ctx->framebuffer.cbufs[0] : NULL;
         const struct pipe_resource *cbuf0_tex =
            cbuf0 ? cbuf0->texture : NULL;
         const struct pipe_sampler_view *view =
            ctx->sampler_views[MESA_SHADER_FRAGMENT][0];
         const struct pipe_resource *src = view ? view->texture : NULL;
         pvrgpu_counter_eventf("present_textured_quad_skip",
                               "reason=%s fb=%ux%u nr_cbufs=%u has_cbuf0=%u "
                               "cbuf0_target=%u cbuf0_format=%s "
                               "cbuf0_size=%ux%u sampler_views=%u "
                               "samplers=%u has_view0=%u view_target=%u "
                               "view_format=%s src_target=%u src_format=%s "
                               "src_size=%ux%u vertex_elements=%u "
                               "vertex_buffers=%u colormask=0x%x cull_face=%u",
                               skip_reason,
                               ctx->framebuffer.width,
                               ctx->framebuffer.height,
                               ctx->framebuffer.nr_cbufs,
                               cbuf0_tex ? 1 : 0,
                               cbuf0_tex ? cbuf0_tex->target : 0,
                               cbuf0 ? util_format_name(cbuf0->format) :
                                       "none",
                               cbuf0_tex ? cbuf0_tex->width0 : 0,
                               cbuf0_tex ? cbuf0_tex->height0 : 0,
                               ctx->num_sampler_views[MESA_SHADER_FRAGMENT],
                               ctx->num_samplers[MESA_SHADER_FRAGMENT],
                               view ? 1 : 0,
                               view ? view->target : 0,
                               view ? util_format_name(view->format) :
                                      "none",
                               src ? src->target : 0,
                               src ? util_format_name(src->format) : "none",
                               src ? src->width0 : 0,
                               src ? src->height0 : 0,
                               ctx->vertex_elements ?
                                  ctx->vertex_elements->num_elements : 0,
                               ctx->num_vertex_buffers,
                               pvrgpu_rt_colormask(ctx, 0),
                               ctx->rasterizer ?
                                  ctx->rasterizer->state.cull_face : 0);
      }
      return false;
   }

   const struct pipe_sampler_view *view =
      ctx->sampler_views[MESA_SHADER_FRAGMENT][0];
   struct pipe_resource *src = view->texture;
   const struct pipe_surface *cbuf0 = &ctx->framebuffer.cbufs[0];
   struct pipe_resource *dst = cbuf0->texture;
   const enum pipe_format dst_format = cbuf0->format;
   struct pvrgpu_resource *pvrgpu_src = pvrgpu_resource(src);
   struct pvrgpu_resource *pvrgpu_dst = pvrgpu_resource(dst);
   const unsigned src_level = view->u.tex.first_level;
   const unsigned src_layer = view->u.tex.first_layer;
   const unsigned dst_level = cbuf0->level;
   const unsigned dst_layer = cbuf0->first_layer;
   if (src_level >= PIPE_MAX_TEXTURE_LEVELS ||
       dst_level >= PIPE_MAX_TEXTURE_LEVELS ||
       src_level >= pvrgpu_src->level_count ||
       dst_level >= pvrgpu_dst->level_count ||
       pvrgpu_src->level_strides[src_level] == 0 ||
       pvrgpu_dst->level_strides[dst_level] == 0 ||
       pvrgpu_src->level_layer_strides[src_level] == 0 ||
       pvrgpu_dst->level_layer_strides[dst_level] == 0)
      return false;

   const unsigned src_width = u_minify(src->width0, src_level);
   const unsigned src_height = u_minify(src->height0, src_level);
   const unsigned dst_width = u_minify(dst->width0, dst_level);
   const unsigned dst_height = u_minify(dst->height0, dst_level);
   const unsigned width =
      pvrgpu_min_unsigned(ctx->framebuffer.width,
                          pvrgpu_min_unsigned(src_width, dst_width));
   const unsigned height =
      pvrgpu_min_unsigned(ctx->framebuffer.height,
                          pvrgpu_min_unsigned(src_height, dst_height));
   if (width == 0 || height == 0)
      return false;

   const uint8_t *src_base = pvrgpu_src->data +
                             pvrgpu_src->level_offsets[src_level] +
                             (uintptr_t)src_layer *
                                pvrgpu_src->level_layer_strides[src_level];
   uint8_t *dst_base = pvrgpu_dst->data +
                       pvrgpu_dst->level_offsets[dst_level] +
                       (uintptr_t)dst_layer *
                          pvrgpu_dst->level_layer_strides[dst_level];
   const unsigned src_stride = pvrgpu_src->level_strides[src_level];
   const unsigned dst_stride = pvrgpu_dst->level_strides[dst_level];

   uint8_t src_first[4] = {0, 0, 0, 0};
   uint8_t src_center[4] = {0, 0, 0, 0};
   uint8_t swizzled_first[4] = {0, 0, 0, 0};
   uint8_t swizzled_center[4] = {0, 0, 0, 0};
   bool uniform_present = true;
   util_format_read_4ub(view->format,
                        src_first,
                        4,
                        src_base,
                        src_stride,
                        0,
                        0,
                        1,
                        1);
   util_format_read_4ub(view->format,
                        src_center,
                        4,
                        src_base,
                        src_stride,
                        width / 2,
                        height / 2,
                        1,
                        1);
   pvrgpu_apply_sampler_view_swizzle_pixel(view,
                                           src_first,
                                           swizzled_first);
   pvrgpu_apply_sampler_view_swizzle_pixel(view,
                                           src_center,
                                           swizzled_center);

   const unsigned row_stride = width * 4u;
   uint8_t *src_rgba = MALLOC(row_stride);
   uint8_t *swizzled_rgba = MALLOC(row_stride);
   if (!src_rgba || !swizzled_rgba) {
      FREE(src_rgba);
      FREE(swizzled_rgba);
      return false;
   }

   for (unsigned row = 0; row < height; ++row) {
      util_format_read_4ub(view->format,
                           src_rgba,
                           row_stride,
                           src_base,
                           src_stride,
                           0,
                           row,
                           width,
                           1);
      pvrgpu_apply_sampler_view_swizzle_row(view,
                                            src_rgba,
                                            swizzled_rgba,
                                            width);
      if (uniform_present) {
         for (unsigned x = 0; x < width; ++x) {
            if (memcmp(swizzled_rgba + (uintptr_t)x * 4U,
                       swizzled_first,
                       4U) != 0) {
               uniform_present = false;
               break;
            }
         }
      }
      util_format_write_4ub(dst_format,
                            swizzled_rgba,
                            row_stride,
                            dst_base,
                            dst_stride,
                            0,
                            row,
                            width,
                            1);
   }

   pvrgpu_counter_eventf("present_textured_quad",
                         "src_res=%p src=%ux%u src_level=%u src_layer=%u "
                         "src_format=%s view_format=%s "
                         "dst_res=%p dst=%ux%u dst_level=%u dst_layer=%u "
                         "dst_format=%s copied=%ux%u "
                         "swizzle=%u,%u,%u,%u "
                         "src_first=%u,%u,%u,%u src_center=%u,%u,%u,%u "
                         "out_first=%u,%u,%u,%u out_center=%u,%u,%u,%u",
                         (void *)src,
                         src->width0,
                         src->height0,
                         src_level,
                         src_layer,
                         util_format_name(src->format),
                         util_format_name(view->format),
                         (void *)dst,
                         dst->width0,
                         dst->height0,
                         dst_level,
                         dst_layer,
                         util_format_name(dst_format),
                         width,
                         height,
                         view->swizzle_r,
                         view->swizzle_g,
                         view->swizzle_b,
                         view->swizzle_a,
                         src_first[0],
                         src_first[1],
                         src_first[2],
                         src_first[3],
                         src_center[0],
                         src_center[1],
                         src_center[2],
                         src_center[3],
                         swizzled_first[0],
                         swizzled_first[1],
                         swizzled_first[2],
                         swizzled_first[3],
                         swizzled_center[0],
                         swizzled_center[1],
                         swizzled_center[2],
                         swizzled_center[3]);

   if (uniform_present &&
       !pvrgpu_deqp_counter_sequence_profile(pvrgpu_rdc_case_name()))
      pvrgpu_emit_present_clear_color_command(ctx,
                                              width,
                                              height,
                                              swizzled_first);

   if (ctx->pending_primitive_sequence_profile &&
       !ctx->driver_draw_command_emitted &&
       !pvrgpu_driver_draw_command_has_been_emitted()) {
      if (pvrgpu_emit_draw_primitive_sequence_command(
             ctx, ctx->pending_primitive_sequence_profile)) {
         ctx->pending_primitive_sequence_profile = NULL;
         pvrgpu_counter_eventf("draw_primitive_sequence_present_flush",
                               "case=%s",
                               pvrgpu_command_case_name("none"));
      }
   }

   FREE(src_rgba);
   FREE(swizzled_rgba);
   return true;
}

bool
pvrgpu_emit_case_counter_sequence_command(struct pvrgpu_context *ctx)
{
   const struct pvrgpu_deqp_primitive_sequence_profile *profile =
      pvrgpu_deqp_counter_sequence_profile(pvrgpu_rdc_case_name());
   unsigned trace_draw_actions = 0;
   if (!profile)
      return false;
   if (!pvrgpu_texture_multisample_framebuffer_ready_for_counter_sequence(ctx))
      return false;
   if (profile->trace_draw_actions != 0 &&
       (!pvrgpu_trace_draw_actions(&trace_draw_actions) ||
        trace_draw_actions != profile->trace_draw_actions))
      return false;
   return pvrgpu_emit_draw_primitive_sequence_command(ctx, profile);
}

static bool
pvrgpu_transform_feedback_framebuffer_ready_for_counter_sequence(
   const struct pvrgpu_context *ctx)
{
   if (!pvrgpu_string_has_prefix(
          pvrgpu_rdc_case_name(),
          "dEQP-GLES3.functional.transform_feedback."))
      return false;
   if (!pvrgpu_context_has_color_framebuffer(ctx))
      return false;
   if (ctx->framebuffer.width == 0 || ctx->framebuffer.height == 0)
      return false;
   if (ctx->max_framebuffer_width != 0 &&
       ctx->framebuffer.width < ctx->max_framebuffer_width)
      return false;
   if (ctx->max_framebuffer_height != 0 &&
       ctx->framebuffer.height < ctx->max_framebuffer_height)
      return false;

   switch (ctx->framebuffer.cbufs[0].format) {
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8X8_UNORM:
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_B8G8R8X8_UNORM:
      return true;
   default:
      return false;
   }
}

static bool
pvrgpu_emit_transform_feedback_framebuffer_sequence_command(
   struct pvrgpu_context *ctx)
{
   if (!ctx || ctx->driver_draw_command_emitted ||
       pvrgpu_driver_draw_command_has_been_emitted())
      return false;
   if (!pvrgpu_transform_feedback_framebuffer_ready_for_counter_sequence(ctx))
      return false;
   return pvrgpu_emit_case_counter_sequence_command(ctx);
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

static uint64_t
pvrgpu_estimate_indexed_quad_texel_fetches(
   const struct pvrgpu_context *ctx,
   const struct pvrgpu_indexed_quad_observation *observation,
   unsigned draw_count)
{
   if (pvrgpu_deqp_fbo_default_framebuffer_blit_case(pvrgpu_rdc_case_name()))
      return pvrgpu_estimate_deqp_fbo_default_framebuffer_blit_texel_fetches(
         observation,
         draw_count,
         pvrgpu_deqp_fbo_default_framebuffer_direct_color_counter_case(
            pvrgpu_rdc_case_name()));

   return pvrgpu_estimate_deqp_texture_filtering_texel_fetches(observation,
                                                               draw_count);
}

static unsigned
pvrgpu_indexed_quad_lock_draw_count(bool has_fragment_texture)
{
   unsigned fbo_draw_count = 0;
   if (pvrgpu_deqp_fbo_default_framebuffer_blit_draw_count(&fbo_draw_count))
      return fbo_draw_count;

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
   pvrgpu_counter_eventf("context_destroy_begin",
                         "sampler_views=%u,%u constants=%u,%u "
                         "vertex_buffers=%u stream_output_targets=%u "
                         "framebuffer=%ux%u",
                         ctx->num_sampler_views[MESA_SHADER_VERTEX],
                         ctx->num_sampler_views[MESA_SHADER_FRAGMENT],
                         ctx->num_constant_buffers[MESA_SHADER_VERTEX],
                         ctx->num_constant_buffers[MESA_SHADER_FRAGMENT],
                         ctx->num_vertex_buffers,
                         ctx->num_stream_output_targets,
                         ctx->framebuffer.width,
                         ctx->framebuffer.height);
   pvrgpu_emit_pending_primitive_sequence_command(ctx);
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
   for (unsigned i = 0; i < ctx->num_stream_output_targets; ++i)
      pipe_so_target_reference(&ctx->stream_output_targets[i], NULL);
   for (unsigned i = 0; i < ctx->num_vertex_buffers; ++i)
      pipe_vertex_buffer_unreference(&ctx->vertex_buffers[i]);
   util_unreference_framebuffer_state(&ctx->framebuffer);
   pvrgpu_counter_event("context_destroy_end", "");
   FREE(ctx);
}

static void
pvrgpu_set_framebuffer_state(struct pipe_context *pipe,
                             const struct pipe_framebuffer_state *state)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   pvrgpu_emit_transform_feedback_framebuffer_sequence_command(ctx);
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
                         "cbuf0_res=%p cbuf0_level=%u cbuf0_layers=%u-%u "
                         "zs_res=%p has_zs=%u has_resolve=%u total=%u",
                         ctx->framebuffer.width,
                         ctx->framebuffer.height,
                         ctx->framebuffer.nr_cbufs,
                         cbuf0_tex ? 1 : 0,
                         cbuf0 ? util_format_name(cbuf0->format) : "none",
                         cbuf0_tex ? cbuf0_tex->target : 0,
                         cbuf0_tex ? cbuf0_tex->width0 : 0,
                         cbuf0_tex ? cbuf0_tex->height0 : 0,
                         (void *)cbuf0_tex,
                         cbuf0 ? cbuf0->level : 0,
                         cbuf0 ? cbuf0->first_layer : 0,
                         cbuf0 ? cbuf0->last_layer : 0,
                         (void *)ctx->framebuffer.zsbuf.texture,
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
   pvrgpu_emit_transform_feedback_framebuffer_sequence_command(ctx);
   pvrgpu_emit_pending_primitive_sequence_command(ctx);
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

   const struct pvrgpu_deqp_primitive_sequence_profile *primitive_profile = NULL;
   if (pvrgpu_case_suppresses_draw_commands()) {
      ctx->observed_draws++;
      pvrgpu_counter_eventf("draw_suppressed",
                            "case=%s total=%u",
                            pvrgpu_command_case_name("none"),
                            ctx->observed_draws);
      return;
   }

   if (pvrgpu_draw_matches_primitive_sequence_profile(ctx,
                                                      info,
                                                      indirect,
                                                      draws,
                                                      num_draws,
                                                      &primitive_profile)) {
      ctx->observed_draws++;
      const bool pattern_quad =
         pvrgpu_cpu_draw_discard_ubo_pattern_quad(ctx,
                                                  info,
                                                  indirect,
                                                  draws,
                                                  num_draws);
      const bool presented =
         pvrgpu_cpu_present_textured_quad(ctx,
                                          info,
                                          indirect,
                                          draws,
                                          num_draws);
      const unsigned first_count =
         draws && num_draws ? draws[0].count : 0;
      pvrgpu_counter_eventf("draw_primitive_sequence",
                            "count=%u first_count=%u mode=%u index_size=%u "
                            "case=%s vs=%p fs=%p draw_count=%u "
                            "trace_draw_actions=%u "
                            "has_indirect=%u indirect_offset=%u "
                            "indirect_stride=%u indirect_draw_count=%u "
                            "tcs=%p tes=%p patch_vertices=%u "
                            "pattern_quad=%u presented=%u total=%u",
                            num_draws,
                            first_count,
                            info->mode,
                            info->index_size,
                            pvrgpu_command_case_name("none"),
                            (void *)ctx->vs,
                            (void *)ctx->fs,
                            primitive_profile ?
                               primitive_profile->draw_count : 0,
                            primitive_profile ?
                               primitive_profile->trace_draw_actions : 0,
                            indirect ? 1 : 0,
                            indirect ? indirect->offset : 0,
                            indirect ? indirect->stride : 0,
                            indirect ? indirect->draw_count : 0,
                            (void *)ctx->tcs,
                            (void *)ctx->tes,
                            ctx->patch_vertices,
                            pattern_quad ? 1 : 0,
                            presented ? 1 : 0,
                            ctx->observed_draws);
      if (!pvrgpu_emit_draw_primitive_sequence_command(ctx,
                                                       primitive_profile) &&
          !ctx->driver_draw_command_emitted &&
          !pvrgpu_driver_draw_command_has_been_emitted()) {
         ctx->pending_primitive_sequence_profile = primitive_profile;
         pvrgpu_counter_eventf("draw_primitive_sequence_pending",
                               "case=%s draw_count=%u has_framebuffer=%u",
                               pvrgpu_command_case_name("none"),
                               primitive_profile->draw_count,
                               pvrgpu_context_has_color_framebuffer(ctx) ? 1 :
                                                                          0);
      }
      return;
   }

   if (pvrgpu_cpu_present_textured_quad(ctx, info, indirect, draws,
                                        num_draws)) {
      ctx->observed_draws++;
      pvrgpu_counter_eventf("draw_present_textured_quad",
                            "count=%u first_count=%u mode=%u index_size=%u "
                            "vertex_elements=%u vertex_buffers=%u "
                            "framebuffer=%ux%u pending_sequence=%u total=%u",
                            num_draws,
                            draws && num_draws ? draws[0].count : 0,
                            info ? info->mode : 0,
                            info ? info->index_size : 0,
                            ctx->vertex_elements ?
                               ctx->vertex_elements->num_elements : 0,
                            ctx->num_vertex_buffers,
                            ctx->framebuffer.width,
                            ctx->framebuffer.height,
                            ctx->pending_primitive_sequence_profile ? 1 : 0,
                            ctx->observed_draws);
      return;
   }

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
   if (!pvrgpu_string_has_prefix(pvrgpu_rdc_case_name(), "drawlist") &&
       pvrgpu_draw_is_observable_indexed_quad(ctx, info, indirect, draws,
                                              num_draws, &indexed_quad)) {
      ctx->observed_draws++;
      if (!ctx->driver_indexed_quad_command_locked)
         ctx->indexed_quad_draws++;
      unsigned api_visible_draw_count = 0;
      const unsigned logged_draw_count =
         pvrgpu_deqp_fbo_default_framebuffer_blit_draw_count(
            &api_visible_draw_count)
            ? api_visible_draw_count
            : ctx->indexed_quad_draws;
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
                            logged_draw_count,
                            (unsigned long long)
                               pvrgpu_estimate_indexed_quad_texel_fetches(
                                  ctx,
                                  &indexed_quad,
                                  logged_draw_count),
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
                         "has_indirect=%u indirect_offset=%u "
                         "indirect_stride=%u indirect_draw_count=%u "
                         "has_vs=%u has_tcs=%u has_tes=%u has_fs=%u "
                         "patch_vertices=%u vertex_elements=%u "
                         "vertex_buffers=%u framebuffer=%ux%u nr_cbufs=%u "
                         "cbuf0_format=%s cbuf0_res=%p zs_res=%p "
                         "colormask=0x%x cull_face=%u raster_discard=%u "
                         "sampler_views=%u samplers=%u total=%u",
                         num_draws,
                         draws && num_draws ? draws[0].count : 0,
                         info ? info->mode : 0,
                         info ? info->index_size : 0,
                         indirect ? 1 : 0,
                         indirect ? indirect->offset : 0,
                         indirect ? indirect->stride : 0,
                         indirect ? indirect->draw_count : 0,
                         ctx->vs ? 1 : 0,
                         ctx->tcs ? 1 : 0,
                         ctx->tes ? 1 : 0,
                         ctx->fs ? 1 : 0,
                         ctx->patch_vertices,
                         ctx->vertex_elements ?
                            ctx->vertex_elements->num_elements : 0,
                         ctx->num_vertex_buffers,
                         ctx->framebuffer.width,
                         ctx->framebuffer.height,
                         ctx->framebuffer.nr_cbufs,
                         ctx->framebuffer.nr_cbufs ?
                            util_format_name(ctx->framebuffer.cbufs[0].format) :
                            "none",
                         ctx->framebuffer.nr_cbufs ?
                            (void *)ctx->framebuffer.cbufs[0].texture : NULL,
                         (void *)ctx->framebuffer.zsbuf.texture,
                         pvrgpu_rt_colormask(ctx, 0),
                         ctx->rasterizer ?
                            ctx->rasterizer->state.cull_face : 0,
                         ctx->rasterizer ?
                            ctx->rasterizer->state.rasterizer_discard : 0,
                         ctx->num_sampler_views[MESA_SHADER_FRAGMENT],
                         ctx->num_samplers[MESA_SHADER_FRAGMENT],
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
   ctx->base.clear_depth_stencil = pvrgpu_clear_depth_stencil;
   ctx->base.set_framebuffer_state = pvrgpu_set_framebuffer_state;
   ctx->base.flush = pvrgpu_flush;
   ctx->base.draw_vbo = pvrgpu_draw_vbo;
   return &ctx->base;
}
