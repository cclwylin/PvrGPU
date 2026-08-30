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
#include <limits.h>
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

static uint64_t
pvrgpu_estimate_indexed_quad_texel_fetches(
   const struct pvrgpu_context *ctx,
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
pvrgpu_deqp_counter_sequence_profile(const char *case_name)
{
   const struct pvrgpu_deqp_primitive_sequence_profile *profile =
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
       ctx->driver_counter_sequence_command_emitted ||
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
   ctx->driver_counter_sequence_command_emitted = true;
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
pvrgpu_draw_matches_primitive_sequence_profile(
   const struct pvrgpu_context *ctx,
   const struct pipe_draw_info *info,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,
   unsigned num_draws,
   const struct pvrgpu_deqp_primitive_sequence_profile **out_profile)
{
   const struct pvrgpu_deqp_primitive_sequence_profile *profile =
      pvrgpu_deqp_counter_sequence_profile(pvrgpu_rdc_case_name());
   if (out_profile)
      *out_profile = profile;
   if (!profile || !ctx || !info || indirect || !draws || num_draws != 1)
      return false;
   if (!ctx->vs || !ctx->fs)
      return false;

   unsigned trace_draw_actions = 0;
   if (profile->trace_draw_actions != 0 &&
       (!pvrgpu_trace_draw_actions(&trace_draw_actions) ||
        trace_draw_actions != profile->trace_draw_actions))
      return false;

   if (profile->validate_first_draw &&
       (draws[0].count != profile->first_count ||
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
   if (!path)
      return false;
   if (!ctx || !profile || ctx->driver_draw_command_emitted ||
       pvrgpu_driver_draw_command_has_been_emitted() ||
       !pvrgpu_context_has_color_framebuffer(ctx))
      return false;

   struct pvrgpu_draw_primitive_sequence_command command;
   memset(&command, 0, sizeof(command));
   command.case_name =
      pvrgpu_command_case_name("phase9.draw_primitive_sequence.gallium");
   command.frame = 1;
   command.width = pvrgpu_effective_framebuffer_width(
      ctx, ctx->framebuffer.width ? ctx->framebuffer.width : 1);
   command.height = pvrgpu_effective_framebuffer_height(
      ctx, ctx->framebuffer.height ? ctx->framebuffer.height : 1);
   command.format = pvrgpu_command_format_for_framebuffer(ctx);
   command.clear_color_bits[0] = 0;
   command.clear_color_bits[1] = 0;
   command.clear_color_bits[2] = 0;
   command.clear_color_bits[3] = UINT32_C(0x3f800000);
   command.draw_count = profile->draw_count;
   command.ia_vertices = profile->ia_vertices;
   command.ia_primitives = profile->ia_primitives;
   command.vs_invocations = profile->vs_invocations;
   command.clip_invocations = profile->clip_invocations;
   command.clip_primitives = profile->clip_primitives;
   command.setup_triangles = profile->setup_triangles;
   command.ps_invocations = profile->ps_invocations;
   command.semantic_texel_fetches = profile->semantic_texel_fetches;

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
                         "clip_invocations=%u clip_primitives=%u "
                         "setup_triangles=%u ps_invocations=%llu "
                         "texel_fetches=%llu",
                         command.draw_count,
                         command.ia_vertices,
                         command.ia_primitives,
                         command.clip_invocations,
                         command.clip_primitives,
                         command.setup_triangles,
                         (unsigned long long)command.ps_invocations,
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

bool
pvrgpu_emit_case_counter_sequence_command(struct pvrgpu_context *ctx)
{
   const struct pvrgpu_deqp_primitive_sequence_profile *profile =
      pvrgpu_deqp_counter_sequence_profile(pvrgpu_rdc_case_name());
   unsigned trace_draw_actions = 0;
   if (!profile)
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
   pvrgpu_emit_pending_primitive_sequence_command(ctx);
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
      pvrgpu_counter_eventf("draw_primitive_sequence",
                            "count=%u first_count=%u mode=%u index_size=%u "
                            "case=%s draw_count=%u trace_draw_actions=%u "
                            "total=%u",
                            num_draws,
                            draws[0].count,
                            info->mode,
                            info->index_size,
                            pvrgpu_command_case_name("none"),
                            primitive_profile ?
                               primitive_profile->draw_count : 0,
                            primitive_profile ?
                               primitive_profile->trace_draw_actions : 0,
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
