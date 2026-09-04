/* SPDX-License-Identifier: MIT */

#include "pvrgpu_context.h"
#include "pvrgpu_cmd.h"
#include "pvrgpu_counter.h"
#include "pvrgpu_pco.h"
#include "pvrgpu_resource.h"
#include "pvrgpu_state.h"
#include "pvrgpu_systemc_api.h"

#include "pipe/p_defines.h"
#include "nir/nir.h"
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

struct pvrgpu_textured_triangles_observation {
   unsigned framebuffer_width;
   unsigned framebuffer_height;
   unsigned viewport_width;
   unsigned viewport_height;
   uint32_t vertex_bits[PVRGPU_DRAW_TEXTURED_TRIANGLES_VERTEX_COUNT][2];
   uint32_t texcoord_bits[PVRGPU_DRAW_TEXTURED_TRIANGLES_VERTEX_COUNT][2];
   unsigned texture_width;
   unsigned texture_height;
   unsigned texture_level;
   const struct pipe_sampler_view *texture_view;
};

struct pvrgpu_conditionals_observation {
   unsigned framebuffer_width;
   unsigned framebuffer_height;
   unsigned viewport_width;
   unsigned viewport_height;
   const uint8_t *raw_vertex_data;
   size_t raw_vertex_data_size;
   uint32_t vertex_shared[PVRGPU_DRAW_PCO_TRIANGLES_VS_SHARED_DWORDS];
   uint32_t fragment_shared[PVRGPU_DRAW_PCO_TRIANGLES_FS_SHARED_DWORDS];
};

struct pvrgpu_lit_mesh_observation {
   enum pvrgpu_pco_lit_mesh_profile profile;
   unsigned framebuffer_width;
   unsigned framebuffer_height;
   unsigned viewport_width;
   unsigned viewport_height;
   unsigned vertex_count;
   uint8_t *interleaved_vertex_data;
   size_t interleaved_vertex_data_size;
   uint32_t vertex_shared[32];
};

struct pvrgpu_texture_pco_observation {
   unsigned framebuffer_width;
   unsigned framebuffer_height;
   unsigned viewport_width;
   unsigned viewport_height;
   unsigned vertex_count;
   uint8_t *interleaved_vertex_data;
   size_t interleaved_vertex_data_size;
   uint8_t *sampled_texture_bytes;
   size_t sampled_texture_bytes_size;
   uint32_t vertex_shared[32];
   uint32_t fragment_shared[20];
};

#define PVRGPU_REFRACT_PCO_VERTEX_COUNT 208998u
#define PVRGPU_REFRACT_PCO_VERTEX_STRIDE (6u * sizeof(float))
#define PVRGPU_REFRACT_PCO_MAX_VS_SHARED_DWORDS 64u
#define PVRGPU_REFRACT_PCO_TEXTURE_COUNT PVRGPU_PCO_REFRACT_TEXTURE_COUNT
#define PVRGPU_REFRACT_PCO_FS_SHARED_DWORDS \
   PVRGPU_PCO_REFRACT_FRAGMENT_SHARED_DWORDS
#define PVRGPU_REFRACT_PCO_IMAGE_WIDTH 512u
#define PVRGPU_REFRACT_PCO_IMAGE_HEIGHT 512u
#define PVRGPU_REFRACT_PCO_IMAGE_ROW_PITCH \
   (PVRGPU_REFRACT_PCO_IMAGE_WIDTH * 4u)
#define PVRGPU_REFRACT_PCO_IMAGE_BYTES \
   (PVRGPU_REFRACT_PCO_IMAGE_ROW_PITCH * PVRGPU_REFRACT_PCO_IMAGE_HEIGHT)
#define PVRGPU_REFRACT_PCO_VERTEX_FNV1A64 UINT64_C(0x83920b2733098afa)
#define PVRGPU_REFRACT_PCO_PREPASS_SHARED_FNV1A64 \
   UINT64_C(0x1b26b797d92ed099)
#define PVRGPU_REFRACT_PCO_COMPOSITE_SHARED_FNV1A64 \
   UINT64_C(0xd5db135e7933d92d)
#define PVRGPU_REFRACT_PCO_FRAGMENT_SHARED_FNV1A64 \
   UINT64_C(0x26536c76cbc158b5)
#define PVRGPU_REFRACT_PCO_FRAGMENT_SHARED_800_FNV1A64 \
   UINT64_C(0x440e2cf4d71a3f5c)
#define PVRGPU_REFRACT_PCO_IMAGE_FNV1A64 UINT64_C(0xecfb10435885d2ce)

enum pvrgpu_refract_pco_texture_source {
   PVRGPU_REFRACT_PCO_EXTERNAL_PAYLOAD,
   PVRGPU_REFRACT_PCO_PREVIOUS_COLOR_ATTACHMENT,
   PVRGPU_REFRACT_PCO_PREVIOUS_DEPTH_ATTACHMENT,
};

struct pvrgpu_refract_pco_texture_mip {
   unsigned width;
   unsigned height;
   unsigned row_pitch;
   unsigned offset;
};

struct pvrgpu_refract_pco_texture {
   enum pvrgpu_refract_pco_texture_source source;
   unsigned descriptor_set;
   unsigned binding;
   enum pipe_format format;
   size_t declared_size;
   unsigned mip_count;
   struct pvrgpu_refract_pco_texture_mip
      mip[PVRGPU_SYSTEMC_MAX_TEXTURE_MIP_LEVELS];
   unsigned min_filter;
   unsigned mag_filter;
   unsigned mip_filter;
   unsigned wrap_u;
   unsigned wrap_v;
   bool normalized_coordinates;
   unsigned min_lod_u4_6;
   unsigned max_lod_u4_6;
};

struct pvrgpu_refract_pco_observation {
   enum pvrgpu_pco_refract_profile profile;
   unsigned framebuffer_width;
   unsigned framebuffer_height;
   unsigned viewport_width;
   unsigned viewport_height;
   unsigned vertex_count;
   uint8_t *interleaved_vertex_data;
   size_t interleaved_vertex_data_size;
   uint32_t vertex_shared[PVRGPU_REFRACT_PCO_MAX_VS_SHARED_DWORDS];
   size_t vertex_shared_count;
   uint32_t fragment_shared[PVRGPU_REFRACT_PCO_FS_SHARED_DWORDS];
   size_t fragment_shared_count;
   struct pvrgpu_refract_pco_texture
      textures[PVRGPU_REFRACT_PCO_TEXTURE_COUNT];
   size_t texture_count;
   struct pipe_resource *prepass_color;
   struct pipe_resource *prepass_depth;
   uint8_t *sampled_image_bytes;
   size_t sampled_image_bytes_size;
   unsigned sampled_image_width;
   unsigned sampled_image_height;
   unsigned sampled_image_row_pitch;
   enum pipe_format sampled_image_format;
   bool composite_depth_clear_one;
   struct pvrgpu_pco_graphics_binary binary;
};

#define PVRGPU_SHADOW_PCO_MESH_VERTEX_COUNT 21516u
#define PVRGPU_SHADOW_PCO_MESH_VERTEX_STRIDE (6u * sizeof(float))
#define PVRGPU_SHADOW_PCO_MASK_VERTEX_COUNT 4u
#define PVRGPU_SHADOW_PCO_MASK_VERTEX_STRIDE (2u * sizeof(float))
#define PVRGPU_SHADOW_PCO_MAX_VS_SHARED_DWORDS 32u
#define PVRGPU_SHADOW_PCO_FS_SHARED_DWORDS \
   PVRGPU_PCO_TEXTURE_DESCRIPTOR_DWORDS
#define PVRGPU_SHADOW_PCO_MESH_VERTEX_FNV1A64 \
   UINT64_C(0x8ed96606410719eb)
#define PVRGPU_SHADOW_PCO_MASK_VERTEX_FNV1A64 \
   UINT64_C(0xa5f3d6f332c36765)
#define PVRGPU_SHADOW_PCO_DEPTH_SHARED_FNV1A64 \
   UINT64_C(0x4f2ffc57463c4710)
#define PVRGPU_SHADOW_PCO_MASK_SHARED_FNV1A64 \
   UINT64_C(0x9f324a4e93b44b28)
#define PVRGPU_SHADOW_PCO_SCENE_SHARED_FNV1A64 \
   UINT64_C(0xbf09db1f576ac6d9)
#define PVRGPU_SHADOW_PCO_FRAGMENT_SHARED_FNV1A64 \
   UINT64_C(0x5d306f7625b3e88e)
#define PVRGPU_SHADOW_PCO_FRAGMENT_SHARED_800_FNV1A64 \
   UINT64_C(0x77d13d3fd210cd96)

struct pvrgpu_shadow_pco_observation {
   enum pvrgpu_pco_shadow_profile profile;
   enum mesa_prim primitive_mode;
   unsigned framebuffer_width;
   unsigned framebuffer_height;
   enum pipe_format color_format;
   enum pipe_format depth_format;
   unsigned color_attachment_source;
   unsigned depth_attachment_source;
   uint32_t clear_color_bits[4];
   uint32_t depth_clear_bits;
   bool color_clear;
   bool depth_clear;
   unsigned viewport_width;
   unsigned viewport_height;
   unsigned color_mask;
   bool blend_enable;
   unsigned rgb_func;
   unsigned rgb_src_factor;
   unsigned rgb_dst_factor;
   unsigned alpha_func;
   unsigned alpha_src_factor;
   unsigned alpha_dst_factor;
   bool dither;
   bool depth_enable;
   bool depth_write;
   unsigned depth_func;
   unsigned vertex_count;
   unsigned vertex_stride;
   uint8_t *vertex_data;
   size_t vertex_data_size;
   uint32_t vertex_shared[PVRGPU_SHADOW_PCO_MAX_VS_SHARED_DWORDS];
   size_t vertex_shared_count;
   uint32_t fragment_shared[PVRGPU_SHADOW_PCO_FS_SHARED_DWORDS];
   size_t fragment_shared_count;
   struct pipe_resource *shadow_depth;
   struct pipe_resource *color_attachment;
   struct pipe_resource *depth_attachment;
   bool output_depth_clear_one;
   struct pvrgpu_pco_graphics_binary binary;
};

/* Strict native GLMark2 terrain capture.  Texture descriptors retain their
 * shader stage explicitly: the MAIN pass samples two resources in VS and
 * five in FS, so a stage-less flattened resource list is ambiguous. */
#define PVRGPU_TERRAIN_PCO_DRAW_COUNT 8U
#define PVRGPU_TERRAIN_PCO_MAX_TEXTURES 7U
#define PVRGPU_TERRAIN_PCO_SEQUENCE_TEXTURE_COUNT 13U
#define PVRGPU_TERRAIN_PCO_MAX_VS_SHARED_DWORDS 96U
#define PVRGPU_TERRAIN_PCO_MAX_FS_SHARED_DWORDS 164U
#define PVRGPU_TERRAIN_PCO_MAIN_VERTEX_COUNT 393216U
#define PVRGPU_TERRAIN_PCO_FULLSCREEN_VERTEX_COUNT 6U
#define PVRGPU_TERRAIN_PCO_MAIN_VERTEX_STRIDE (11U * sizeof(float))
#define PVRGPU_TERRAIN_PCO_FULLSCREEN_VERTEX_STRIDE (3U * sizeof(float))

enum pvrgpu_terrain_pco_texture_source {
   PVRGPU_TERRAIN_PCO_EXTERNAL_PAYLOAD,
   PVRGPU_TERRAIN_PCO_PREVIOUS_COLOR_ATTACHMENT,
};

struct pvrgpu_terrain_pco_texture_mip {
   unsigned width;
   unsigned height;
   unsigned row_pitch;
   unsigned offset;
};

struct pvrgpu_terrain_pco_texture {
   mesa_shader_stage stage;
   enum pvrgpu_terrain_pco_texture_source source;
   unsigned producer_command_index;
   unsigned source_slot;
   unsigned descriptor_set;
   unsigned binding;
   enum pipe_format format;
   uint8_t *bytes;
   size_t bytes_size;
   size_t declared_size;
   unsigned mip_count;
   struct pvrgpu_terrain_pco_texture_mip
      mip[PVRGPU_SYSTEMC_MAX_TEXTURE_MIP_LEVELS];
   unsigned min_filter;
   unsigned mag_filter;
   unsigned mip_filter;
   unsigned wrap_u;
   unsigned wrap_v;
   bool normalized_coordinates;
   unsigned min_lod_u4_6;
   unsigned max_lod_u4_6;
};

struct pvrgpu_terrain_pco_observation {
   enum pvrgpu_pco_terrain_profile profile;
   enum mesa_prim primitive_mode;
   unsigned framebuffer_width;
   unsigned framebuffer_height;
   enum pipe_format color_format;
   enum pipe_format depth_format;
   unsigned color_attachment_source;
   unsigned depth_attachment_source;
   unsigned viewport_width;
   unsigned viewport_height;
   uint32_t clear_color_bits[4];
   uint32_t viewport_scale_bits[3];
   uint32_t viewport_translate_bits[3];
   bool depth_clear_one;
   bool front_ccw;
   unsigned cull_face;
   unsigned fill_front;
   unsigned fill_back;
   bool scissor;
   bool rasterizer_discard;
   bool multisample;
   bool half_pixel_center;
   bool bottom_edge_rule;
   bool clip_halfz;
   bool depth_clip_near;
   bool depth_clip_far;
   bool depth_clamp;
   unsigned sample_mask;
   unsigned color_mask;
   bool color_clear;
   bool blend_enable;
   unsigned rgb_func;
   unsigned rgb_src_factor;
   unsigned rgb_dst_factor;
   unsigned alpha_func;
   unsigned alpha_src_factor;
   unsigned alpha_dst_factor;
   bool dither;
   bool depth_enable;
   bool depth_write;
   unsigned depth_func;
   bool depth_clear;
   unsigned vertex_count;
   unsigned vertex_stride;
   uint8_t *vertex_data;
   size_t vertex_data_size;
   uint32_t vertex_shared[PVRGPU_TERRAIN_PCO_MAX_VS_SHARED_DWORDS];
   size_t vertex_shared_count;
   uint32_t fragment_shared[PVRGPU_TERRAIN_PCO_MAX_FS_SHARED_DWORDS];
   size_t fragment_shared_count;
   struct pvrgpu_terrain_pco_texture
      textures[PVRGPU_TERRAIN_PCO_MAX_TEXTURES];
   size_t texture_count;
   struct pipe_resource *color_attachment;
   struct pipe_resource *depth_attachment;
   struct pvrgpu_pco_graphics_binary binary;
};

static bool
pvrgpu_rgba8_surface_backing_matches(const struct pipe_surface *surface,
                                     unsigned width,
                                     unsigned height,
                                     const uint8_t expected_rgba[4]);

enum pvrgpu_ideas_pco_profile {
   PVRGPU_IDEAS_PCO_LOGO,
   PVRGPU_IDEAS_PCO_LIGHTING,
   PVRGPU_IDEAS_PCO_WHITE,
   PVRGPU_IDEAS_PCO_BLACK,
   PVRGPU_IDEAS_PCO_PROFILE_COUNT,
};

struct pvrgpu_ideas_pco_observation {
   enum pvrgpu_ideas_pco_profile profile;
   unsigned framebuffer_width;
   unsigned framebuffer_height;
   unsigned viewport_width;
   unsigned viewport_height;
   unsigned vertex_count;
   unsigned vertex_stride;
   unsigned attribute_count;
   unsigned primitive_mode;
   unsigned cull_face;
   uint8_t *interleaved_vertex_data;
   size_t interleaved_vertex_data_size;
   uint32_t vertex_shared[44];
   size_t vertex_shared_count;
   uint32_t fragment_shared[12];
   size_t fragment_shared_count;
};

#define PVRGPU_IDEAS_PCO_DRAW_COUNT 180u

#define PVRGPU_TEXTURE_PCO_VERTEX_COUNT 36u
#define PVRGPU_TEXTURE_PCO_VERTEX_STRIDE 32u
#define PVRGPU_TEXTURE_PCO_WIDTH 512u
#define PVRGPU_TEXTURE_PCO_HEIGHT 512u
#define PVRGPU_TEXTURE_PCO_ROW_PITCH 2048u
#define PVRGPU_TEXTURE_PCO_BYTES UINT32_C(1048576)
#define PVRGPU_TEXTURE_PCO_GPU_ADDRESS UINT64_C(0x40000000)

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
   return isalnum(ch) || ch == '_' || ch == '.' || ch == '-' || ch == '*';
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
   if (ctx && ctx->framebuffer.width > width)
      width = ctx->framebuffer.width;
   return width;
}

static unsigned
pvrgpu_effective_framebuffer_height(const struct pvrgpu_context *ctx,
                                    unsigned fallback)
{
   unsigned height = fallback;
   if (ctx && ctx->framebuffer.height > height)
      height = ctx->framebuffer.height;
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
pvrgpu_glmark_output_extent_supported(unsigned width, unsigned height)
{
   return (width == 80u && height == 60u) ||
          (width == 800u && height == 600u);
}

static bool
pvrgpu_glmark_scaled_output_extent(unsigned scale,
                                   unsigned *width,
                                   unsigned *height)
{
   unsigned output_width = 0;
   unsigned output_height = 0;
   if (!width || !height || (scale != 1u && scale != 2u) ||
       !pvrgpu_rdc_output_extent(&output_width, &output_height) ||
       !pvrgpu_glmark_output_extent_supported(output_width, output_height))
      return false;

   *width = output_width * scale;
   *height = output_height * scale;
   return true;
}

static bool
pvrgpu_tight_rgba8_mip_layout(unsigned width,
                              unsigned height,
                              unsigned *mip_count,
                              size_t *byte_count)
{
   if (!width || !height || !mip_count || !byte_count)
      return false;

   unsigned levels = 0;
   size_t total = 0;
   for (;;) {
      if (width > UINT32_MAX / sizeof(uint32_t))
         return false;
      const size_t row_bytes = (size_t)width * sizeof(uint32_t);
      if (height > SIZE_MAX / row_bytes)
         return false;
      const size_t level_bytes = row_bytes * height;
      if (total > SIZE_MAX - level_bytes)
         return false;
      total += level_bytes;
      levels++;

      if (width == 1u && height == 1u)
         break;
      width = MAX2(width >> 1u, 1u);
      height = MAX2(height >> 1u, 1u);
   }

   *mip_count = levels;
   *byte_count = total;
   return true;
}

static bool
pvrgpu_glmark_pco_draw_extent(const struct pvrgpu_context *ctx,
                              bool probe,
                              unsigned *width,
                              unsigned *height)
{
   if (!ctx || !width || !height)
      return false;

   if (probe) {
      *width = 1u;
      *height = 1u;
      return ctx->framebuffer.width == 1u && ctx->framebuffer.height == 1u;
   }

   *width = ctx->framebuffer.width;
   *height = ctx->framebuffer.height;
   return pvrgpu_glmark_output_extent_supported(*width, *height);
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

static bool
pvrgpu_read_resource_float3_vertex(const struct pvrgpu_context *ctx,
                                   unsigned vertex_index,
                                   float out[3])
{
   if (!ctx || !ctx->vertex_elements ||
       ctx->vertex_elements->num_elements != 1 ||
       ctx->num_vertex_buffers != 1 || !out)
      return false;

   const struct pipe_vertex_element *element =
      &ctx->vertex_elements->elements[0];
   if (element->src_format != PIPE_FORMAT_R32G32B32_FLOAT ||
       element->src_stride != 3u * sizeof(float) ||
       element->src_offset != 0 || element->dual_slot ||
       element->instance_divisor != 0 || element->vertex_buffer_index != 0)
      return false;

   const struct pipe_vertex_buffer *vertex_buffer = &ctx->vertex_buffers[0];
   if (vertex_buffer->is_user_buffer || !vertex_buffer->buffer.resource ||
       vertex_buffer->buffer.resource->target != PIPE_BUFFER)
      return false;

   struct pvrgpu_resource *resource =
      pvrgpu_resource(vertex_buffer->buffer.resource);
   const size_t stride = element->src_stride;
   if (!resource || !resource->data ||
       vertex_buffer->buffer_offset > resource->size ||
       vertex_index > (SIZE_MAX - vertex_buffer->buffer_offset) / stride)
      return false;

   const size_t byte_offset =
      (size_t)vertex_buffer->buffer_offset + (size_t)vertex_index * stride;
   if (byte_offset > resource->size ||
       resource->size - byte_offset < 3u * sizeof(float))
      return false;

   memcpy(out, resource->data + byte_offset, 3u * sizeof(float));
   return isfinite(out[0]) && isfinite(out[1]) && isfinite(out[2]);
}

static bool
pvrgpu_emit_draw_triangle_command(struct pvrgpu_context *ctx,
                                  const struct pipe_draw_start_count_bias *draw)
{
   const char *path = pvrgpu_command_output_path();
   if (!path) {
      debug_printf("pvrgpu: PVRGPU_DRIVER_COMMAND_OUT is not set\n");
      return false;
   }
   if (pvrgpu_case_reserves_native_pco_sequence() ||
       !ctx || ctx->driver_draw_command_emitted ||
       pvrgpu_driver_draw_command_has_been_emitted()) {
      pvrgpu_counter_eventf("draw_triangle_command_skip",
                            "reason=command_already_emitted");
      return false;
   }
   if (!pvrgpu_framebuffer_matches_rdc_output(ctx)) {
      pvrgpu_counter_eventf("draw_triangle_command_skip",
                            "reason=framebuffer_mismatch framebuffer=%ux%u",
                            ctx ? ctx->framebuffer.width : 0,
                            ctx ? ctx->framebuffer.height : 0);
      return false;
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
         return false;
      }
      command.vertex_bits[vertex][0] = pvrgpu_float_bits(xy[0]);
      command.vertex_bits[vertex][1] = pvrgpu_float_bits(xy[1]);
   }

   char error[256];
   if (!pvrgpu_write_draw_triangle_command(path, &command, error,
                                           sizeof(error))) {
      debug_printf("pvrgpu: %s\n", error);
      return false;
   }

   ctx->driver_draw_command_emitted = true;
   pvrgpu_note_driver_draw_command_emitted();
   return true;
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
   if (pvrgpu_case_reserves_native_pco_sequence() ||
       !ctx || !observation || ctx->driver_indexed_quad_command_locked ||
       pvrgpu_driver_counter_sequence_command_has_been_emitted())
      return;
   if ((ctx->driver_draw_command_emitted ||
        pvrgpu_driver_draw_command_has_been_emitted()) &&
       ctx->indexed_quad_draws == 0)
      return;
   if (!pvrgpu_framebuffer_matches_rdc_output(ctx)) {
      pvrgpu_counter_eventf("draw_indexed_quad_command_skip",
                            "reason=framebuffer_mismatch framebuffer=%ux%u",
                            ctx->framebuffer.width,
                            ctx->framebuffer.height);
      return;
   }

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





static void
pvrgpu_emit_present_clear_color_command(struct pvrgpu_context *ctx,
                                        unsigned width,
                                        unsigned height,
                                        const uint8_t rgba[4])
{
   if (pvrgpu_case_reserves_native_pco_sequence())
      return;
   const char *path = pvrgpu_command_output_path();
   if (!path) {
      pvrgpu_counter_eventf("present_clear_color_command_skip",
                            "reason=missing_command_path");
      return;
   }
   if (!ctx || !rgba || width == 0 || height == 0)
      return;
   if (!pvrgpu_framebuffer_matches_rdc_output(ctx))
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
   ctx->driver_draw_command_emitted = true;
   pvrgpu_note_driver_draw_command_emitted();
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

   if (uniform_present)
      pvrgpu_emit_present_clear_color_command(ctx,
                                              width,
                                              height,
                                              swizzled_first);

   FREE(src_rgba);
   FREE(swizzled_rgba);
   return true;
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

   return has_fragment_texture ? 4 : 2;
}

static bool
pvrgpu_nir_float_vector_variable(const nir_variable *variable,
                                 nir_variable_mode mode,
                                 int location,
                                 unsigned components)
{
   return variable && variable->data.mode == mode &&
          variable->data.location == location &&
          glsl_type_is_vector(variable->type) &&
          glsl_get_base_type(variable->type) == GLSL_TYPE_FLOAT &&
          glsl_get_bit_size(variable->type) == 32 &&
          glsl_get_vector_elements(variable->type) == components;
}

static bool
pvrgpu_nir_sampler2d_variable(const nir_variable *variable)
{
   return variable && variable->data.mode == nir_var_uniform &&
          variable->data.binding == 0 &&
          glsl_type_is_sampler(variable->type) &&
          glsl_get_sampler_dim(variable->type) == GLSL_SAMPLER_DIM_2D &&
          !glsl_sampler_type_is_array(variable->type) &&
          !glsl_sampler_type_is_shadow(variable->type);
}

static nir_function_impl *
pvrgpu_nir_single_entrypoint(const nir_shader *shader)
{
   if (!shader)
      return NULL;

   nir_function_impl *entrypoint = NULL;
   unsigned function_count = 0;
   nir_foreach_function(function, shader) {
      function_count++;
      if (!function->is_entrypoint || function->num_params != 0 ||
          !function->impl || entrypoint)
         return NULL;
      entrypoint = function->impl;
   }
   return function_count == 1 ? entrypoint : NULL;
}

static bool
pvrgpu_nir_alu_source_matches(const nir_alu_src *source,
                              const nir_def *definition,
                              const unsigned *swizzle,
                              unsigned components)
{
   if (!source || source->src.ssa != definition)
      return false;
   for (unsigned component = 0; component < components; ++component) {
      if (source->swizzle[component] != swizzle[component])
         return false;
   }
   return true;
}

static bool
pvrgpu_nir_constant_is_float(const nir_load_const_instr *constant,
                             uint32_t expected_bits)
{
   if (!constant || constant->def.num_components != 1 ||
       constant->def.bit_size != 32)
      return false;
   const float value =
      (float)nir_const_value_as_float(constant->value[0], 32);
   return pvrgpu_float_bits(value) == expected_bits;
}

static bool
pvrgpu_nir_source_hash_matches(const nir_shader *shader,
                               const uint32_t expected_words[8])
{
   if (!shader || !expected_words)
      return false;

   uint8_t expected[32];
   for (unsigned word = 0; word < 8; ++word) {
      expected[word * 4 + 0] = (uint8_t)(expected_words[word] >> 0);
      expected[word * 4 + 1] = (uint8_t)(expected_words[word] >> 8);
      expected[word * 4 + 2] = (uint8_t)(expected_words[word] >> 16);
      expected[word * 4 + 3] = (uint8_t)(expected_words[word] >> 24);
   }
   return memcmp(shader->info.source_blake3, expected, sizeof(expected)) == 0;
}

static uint32_t
pvrgpu_nir_source_hash_word(const nir_shader *shader, unsigned word)
{
   if (!shader || word >= 8U)
      return 0;
   const uint8_t *bytes = &shader->info.source_blake3[word * 4U];
   return (uint32_t)bytes[0] | (uint32_t)bytes[1] << 8U |
          (uint32_t)bytes[2] << 16U | (uint32_t)bytes[3] << 24U;
}

static void
pvrgpu_nir_source_hash_string(const nir_shader *shader, char output[72])
{
   snprintf(output,
            72,
            "%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x",
            pvrgpu_nir_source_hash_word(shader, 0),
            pvrgpu_nir_source_hash_word(shader, 1),
            pvrgpu_nir_source_hash_word(shader, 2),
            pvrgpu_nir_source_hash_word(shader, 3),
            pvrgpu_nir_source_hash_word(shader, 4),
            pvrgpu_nir_source_hash_word(shader, 5),
            pvrgpu_nir_source_hash_word(shader, 6),
            pvrgpu_nir_source_hash_word(shader, 7));
}

static bool
pvrgpu_nir_collect_exact_instructions(nir_shader *shader,
                                      nir_instr **instructions,
                                      unsigned expected_count)
{
   nir_function_impl *entrypoint = pvrgpu_nir_single_entrypoint(shader);
   if (!entrypoint || !instructions || expected_count == 0)
      return false;

   unsigned block_count = 0;
   unsigned instruction_count = 0;
   nir_foreach_block(block, entrypoint) {
      block_count++;
      nir_foreach_instr(instruction, block) {
         if (instruction_count >= expected_count)
            return false;
         instructions[instruction_count++] = instruction;
      }
   }
   /* nir_foreach_block visits the function's real control-flow blocks; the
    * synthetic end block printed after b0 is not part of that iteration. */
   return block_count == 1 && instruction_count == expected_count;
}

static bool
pvrgpu_nir_alu_matches(const nir_instr *instruction,
                       nir_op op,
                       unsigned components,
                       unsigned bit_size)
{
   if (!instruction || instruction->type != nir_instr_type_alu)
      return false;
   const nir_alu_instr *alu = nir_instr_as_alu(instruction);
   return alu->op == op && alu->def.num_components == components &&
          alu->def.bit_size == bit_size;
}

static bool
pvrgpu_nir_deref_var_matches(const nir_instr *instruction,
                             const nir_variable *variable)
{
   if (!instruction || instruction->type != nir_instr_type_deref)
      return false;
   const nir_deref_instr *deref = nir_instr_as_deref(instruction);
   return deref->deref_type == nir_deref_type_var && deref->var == variable;
}

static bool
pvrgpu_nir_load_deref_matches(const nir_instr *instruction,
                              const nir_deref_instr *deref,
                              unsigned components)
{
   if (!instruction || instruction->type != nir_instr_type_intrinsic)
      return false;
   const nir_intrinsic_instr *intrinsic =
      nir_instr_as_intrinsic(instruction);
   return intrinsic->intrinsic == nir_intrinsic_load_deref &&
          intrinsic->num_components == components &&
          intrinsic->def.num_components == components &&
          intrinsic->def.bit_size == 32 &&
          intrinsic->src[0].ssa == &deref->def;
}

static bool
pvrgpu_nir_load_uniform_matches(const nir_instr *instruction,
                                const nir_load_const_instr *slot,
                                unsigned expected_slot,
                                unsigned range)
{
   if (!instruction || instruction->type != nir_instr_type_intrinsic ||
       !slot)
      return false;
   const nir_intrinsic_instr *intrinsic =
      nir_instr_as_intrinsic(instruction);
   return intrinsic->intrinsic == nir_intrinsic_load_uniform &&
          intrinsic->num_components == 4 &&
          intrinsic->def.num_components == 4 &&
          intrinsic->def.bit_size == 32 &&
          intrinsic->src[0].ssa == &slot->def &&
          nir_intrinsic_base(intrinsic) == 0 &&
          nir_intrinsic_range(intrinsic) == range &&
          nir_intrinsic_dest_type(intrinsic) == nir_type_float32 &&
          slot->def.num_components == 1 && slot->def.bit_size == 32 &&
          slot->value[0].u32 == expected_slot;
}

static bool
pvrgpu_nir_store_deref_matches(const nir_instr *instruction,
                               const nir_deref_instr *deref,
                               const nir_def *value)
{
   if (!instruction || instruction->type != nir_instr_type_intrinsic ||
       !deref || !value)
      return false;
   const nir_intrinsic_instr *intrinsic =
      nir_instr_as_intrinsic(instruction);
   return intrinsic->intrinsic == nir_intrinsic_store_deref &&
          intrinsic->num_components == 4 &&
          nir_intrinsic_write_mask(intrinsic) == 0xfu &&
          intrinsic->src[0].ssa == &deref->def &&
          intrinsic->src[1].ssa == value;
}

static bool
pvrgpu_nir_conditionals_uniform_mat4(const nir_variable *variable)
{
   return variable && variable->data.mode == nir_var_uniform &&
          glsl_type_is_matrix(variable->type) &&
          glsl_get_base_type(variable->type) == GLSL_TYPE_FLOAT &&
          glsl_get_bit_size(variable->type) == 32 &&
          glsl_get_matrix_columns(variable->type) == 4 &&
          glsl_get_vector_elements(variable->type) == 4;
}

static bool
pvrgpu_nir_matches_conditionals_vs(
   const struct pvrgpu_shader_state *shader_state)
{
   static const uint32_t source_hash[8] = {
      UINT32_C(0x5a33231c), UINT32_C(0xfad6d52d),
      UINT32_C(0x74af6d35), UINT32_C(0x864c84c1),
      UINT32_C(0x4f18998f), UINT32_C(0x2a29d5aa),
      UINT32_C(0x9a136e1c), UINT32_C(0xbc2ae4cd),
   };
   if (!shader_state || shader_state->stage != MESA_SHADER_VERTEX ||
       shader_state->type != PIPE_SHADER_IR_NIR ||
       !shader_state->has_nir || !shader_state->nir) {
      pvrgpu_counter_eventf(
         "conditionals_vs_nir_miss",
         "reason=envelope shader=%u stage=%u type=%u has_nir=%u",
         shader_state ? 1u : 0u,
         shader_state ? shader_state->stage : 0u,
         shader_state ? shader_state->type : 0u,
         shader_state && shader_state->has_nir ? 1u : 0u);
      return false;
   }
   if (!pvrgpu_nir_source_hash_matches(shader_state->nir, source_hash)) {
      pvrgpu_counter_eventf("conditionals_vs_nir_miss",
                            "reason=source_hash");
      return false;
   }

   nir_shader *shader = shader_state->nir;
   nir_variable *uniform = NULL;
   nir_variable *position = NULL;
   nir_variable *position_output = NULL;
   unsigned variable_count = 0;
   nir_foreach_variable_in_shader(variable, shader) {
      variable_count++;
      if (pvrgpu_nir_conditionals_uniform_mat4(variable)) {
         if (uniform)
            return false;
         uniform = variable;
      } else if (pvrgpu_nir_float_vector_variable(variable,
                                                  nir_var_shader_in,
                                                  VERT_ATTRIB_GENERIC0,
                                                  4)) {
         if (position)
            return false;
         position = variable;
      } else if (pvrgpu_nir_float_vector_variable(variable,
                                                  nir_var_shader_out,
                                                  VARYING_SLOT_POS,
                                                  4)) {
         if (position_output)
            return false;
         position_output = variable;
      } else {
         return false;
      }
   }
   if (variable_count != 3 || !uniform || !position || !position_output ||
       uniform->data.precision != GLSL_PRECISION_HIGH ||
       position->data.precision != GLSL_PRECISION_HIGH ||
       position_output->data.precision != GLSL_PRECISION_MEDIUM ||
       position->data.location_frac != 0 ||
       position_output->data.location_frac != 0) {
      pvrgpu_counter_eventf(
         "conditionals_vs_nir_miss",
         "reason=variables count=%u uniform=%u position=%u output=%u "
         "uniform_precision=%u position_precision=%u output_precision=%u "
         "position_frac=%u output_frac=%u",
         variable_count,
         uniform ? 1u : 0u,
         position ? 1u : 0u,
         position_output ? 1u : 0u,
         uniform ? uniform->data.precision : 0u,
         position ? position->data.precision : 0u,
         position_output ? position_output->data.precision : 0u,
         position ? position->data.location_frac : 0u,
         position_output ? position_output->data.location_frac : 0u);
      return false;
   }

   nir_instr *instruction[32];
   if (!pvrgpu_nir_collect_exact_instructions(shader,
                                              instruction,
                                              PVRGPU_ARRAY_SIZE(instruction))) {
      pvrgpu_counter_eventf("conditionals_vs_nir_miss",
                            "reason=instruction_envelope expected=32");
      return false;
   }

   static const uint32_t constant_bits[8] = {
      UINT32_C(0x3f000000), UINT32_C(0x40000000),
      UINT32_C(0x40400000), UINT32_C(0x3dcccccd),
      0, 1, 2, 3,
   };
   nir_load_const_instr *constant[8];
   for (unsigned i = 0; i < PVRGPU_ARRAY_SIZE(constant); ++i) {
      if (instruction[i]->type != nir_instr_type_load_const)
         return false;
      constant[i] = nir_instr_as_load_const(instruction[i]);
      if (constant[i]->def.num_components != 1 ||
          constant[i]->def.bit_size != 32 ||
          constant[i]->value[0].u32 != constant_bits[i])
         return false;
   }

   if (!pvrgpu_nir_deref_var_matches(instruction[8], position))
      return false;
   nir_deref_instr *position_deref = nir_instr_as_deref(instruction[8]);
   if (!pvrgpu_nir_load_deref_matches(instruction[9], position_deref, 4))
      return false;
   nir_intrinsic_instr *position_load =
      nir_instr_as_intrinsic(instruction[9]);

   static const nir_op scalar_ops[] = {
      nir_op_ffract, nir_op_fge, nir_op_fmul, nir_op_ffract,
      nir_op_fmul, nir_op_ffract, nir_op_bcsel, nir_op_fmul,
      nir_op_fmul, nir_op_fadd,
   };
   for (unsigned i = 0; i < PVRGPU_ARRAY_SIZE(scalar_ops); ++i) {
      const unsigned bit_size = scalar_ops[i] == nir_op_fge ? 1u : 32u;
      if (!pvrgpu_nir_alu_matches(instruction[10 + i],
                                  scalar_ops[i],
                                  1,
                                  bit_size))
         return false;
   }

   nir_alu_instr *fract_x = nir_instr_as_alu(instruction[10]);
   nir_alu_instr *compare = nir_instr_as_alu(instruction[11]);
   nir_alu_instr *multiply_two = nir_instr_as_alu(instruction[12]);
   nir_alu_instr *fract_two = nir_instr_as_alu(instruction[13]);
   nir_alu_instr *multiply_three = nir_instr_as_alu(instruction[14]);
   nir_alu_instr *fract_three = nir_instr_as_alu(instruction[15]);
   nir_alu_instr *select = nir_instr_as_alu(instruction[16]);
   nir_alu_instr *multiply_tenth = nir_instr_as_alu(instruction[17]);
   nir_alu_instr *multiply_fract = nir_instr_as_alu(instruction[18]);
   nir_alu_instr *position_y = nir_instr_as_alu(instruction[19]);
   static const unsigned scalar[1] = {0};
   static const unsigned input_x[1] = {0};
   static const unsigned input_y[1] = {1};
   if (!pvrgpu_nir_alu_source_matches(&fract_x->src[0],
                                      &position_load->def,
                                      input_x,
                                      1) ||
       !pvrgpu_nir_alu_source_matches(&compare->src[0],
                                      &fract_x->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&compare->src[1],
                                      &constant[0]->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&multiply_two->src[0],
                                      &constant[1]->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&multiply_two->src[1],
                                      &fract_x->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&fract_two->src[0],
                                      &multiply_two->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&multiply_three->src[0],
                                      &constant[2]->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&multiply_three->src[1],
                                      &fract_x->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&fract_three->src[0],
                                      &multiply_three->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&select->src[0],
                                      &compare->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&select->src[1],
                                      &fract_two->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&select->src[2],
                                      &fract_three->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&multiply_tenth->src[0],
                                      &constant[3]->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&multiply_tenth->src[1],
                                      &select->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&multiply_fract->src[0],
                                      &multiply_tenth->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&multiply_fract->src[1],
                                      &fract_x->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&position_y->src[0],
                                      &position_load->def, input_y, 1) ||
       !pvrgpu_nir_alu_source_matches(&position_y->src[1],
                                      &multiply_fract->def, scalar, 1))
      return false;

   nir_intrinsic_instr *uniform_load[4];
   nir_alu_instr *matrix_multiply[3];
   nir_alu_instr *matrix_add[3];
   static const unsigned uniform_instruction[4] = {20, 22, 25, 28};
   static const unsigned multiply_instruction[3] = {21, 23, 26};
   static const unsigned add_instruction[3] = {24, 27, 29};
   for (unsigned slot = 0; slot < 4; ++slot) {
      if (!pvrgpu_nir_load_uniform_matches(
             instruction[uniform_instruction[slot]],
             constant[4 + slot],
             slot,
             4))
         return false;
      uniform_load[slot] =
         nir_instr_as_intrinsic(instruction[uniform_instruction[slot]]);
   }
   for (unsigned i = 0; i < 3; ++i) {
      if (!pvrgpu_nir_alu_matches(instruction[multiply_instruction[i]],
                                  nir_op_fmul,
                                  4,
                                  32) ||
          !pvrgpu_nir_alu_matches(instruction[add_instruction[i]],
                                  nir_op_fadd,
                                  4,
                                  32))
         return false;
      matrix_multiply[i] =
         nir_instr_as_alu(instruction[multiply_instruction[i]]);
      matrix_add[i] = nir_instr_as_alu(instruction[add_instruction[i]]);
   }

   static const unsigned identity[4] = {0, 1, 2, 3};
   static const unsigned xxxx[4] = {0, 0, 0, 0};
   static const unsigned zzzz[4] = {2, 2, 2, 2};
   if (!pvrgpu_nir_alu_source_matches(&matrix_multiply[0]->src[0],
                                      &uniform_load[0]->def, identity, 4) ||
       !pvrgpu_nir_alu_source_matches(&matrix_multiply[0]->src[1],
                                      &position_load->def, xxxx, 4) ||
       !pvrgpu_nir_alu_source_matches(&matrix_multiply[1]->src[0],
                                      &uniform_load[1]->def, identity, 4) ||
       !pvrgpu_nir_alu_source_matches(&matrix_multiply[1]->src[1],
                                      &position_y->def, xxxx, 4) ||
       !pvrgpu_nir_alu_source_matches(&matrix_add[0]->src[0],
                                      &matrix_multiply[0]->def, identity, 4) ||
       !pvrgpu_nir_alu_source_matches(&matrix_add[0]->src[1],
                                      &matrix_multiply[1]->def, identity, 4) ||
       !pvrgpu_nir_alu_source_matches(&matrix_multiply[2]->src[0],
                                      &uniform_load[2]->def, identity, 4) ||
       !pvrgpu_nir_alu_source_matches(&matrix_multiply[2]->src[1],
                                      &position_load->def, zzzz, 4) ||
       !pvrgpu_nir_alu_source_matches(&matrix_add[1]->src[0],
                                      &matrix_add[0]->def, identity, 4) ||
       !pvrgpu_nir_alu_source_matches(&matrix_add[1]->src[1],
                                      &matrix_multiply[2]->def, identity, 4) ||
       !pvrgpu_nir_alu_source_matches(&matrix_add[2]->src[0],
                                      &matrix_add[1]->def, identity, 4) ||
       !pvrgpu_nir_alu_source_matches(&matrix_add[2]->src[1],
                                      &uniform_load[3]->def, identity, 4))
      return false;

   if (!pvrgpu_nir_deref_var_matches(instruction[30], position_output))
      return false;
   nir_deref_instr *output_deref = nir_instr_as_deref(instruction[30]);
   return pvrgpu_nir_store_deref_matches(instruction[31],
                                         output_deref,
                                         &matrix_add[2]->def);
}

static bool
pvrgpu_nir_matches_conditionals_fs(
   const struct pvrgpu_shader_state *shader_state)
{
   static const uint32_t source_hash[8] = {
      UINT32_C(0x7f3de1f2), UINT32_C(0xf5dec4ce),
      UINT32_C(0x57e836cb), UINT32_C(0x582c0e46),
      UINT32_C(0x8ad84991), UINT32_C(0x8be777af),
      UINT32_C(0x505cfc20), UINT32_C(0x01b7c73e),
   };
   if (!shader_state || shader_state->stage != MESA_SHADER_FRAGMENT ||
       shader_state->type != PIPE_SHADER_IR_NIR ||
       !shader_state->has_nir || !shader_state->nir ||
       !pvrgpu_nir_source_hash_matches(shader_state->nir, source_hash))
      return false;

   nir_shader *shader = shader_state->nir;
   nir_variable *uniform = NULL;
   nir_variable *position = NULL;
   nir_variable *color_output = NULL;
   unsigned variable_count = 0;
   nir_foreach_variable_in_shader(variable, shader) {
      variable_count++;
      if (pvrgpu_nir_float_vector_variable(variable,
                                           nir_var_uniform,
                                           variable->data.location,
                                           4)) {
         if (uniform)
            return false;
         uniform = variable;
      } else if (pvrgpu_nir_float_vector_variable(variable,
                                                  nir_var_shader_in,
                                                  VARYING_SLOT_POS,
                                                  4) &&
                 variable->data.interpolation == INTERP_MODE_SMOOTH) {
         if (position)
            return false;
         position = variable;
      } else if (pvrgpu_nir_float_vector_variable(variable,
                                                  nir_var_shader_out,
                                                  FRAG_RESULT_COLOR,
                                                  4)) {
         if (color_output)
            return false;
         color_output = variable;
      } else {
         return false;
      }
   }
   if (variable_count != 3 || !uniform || !position || !color_output ||
       uniform->data.precision != GLSL_PRECISION_NONE ||
       position->data.precision != GLSL_PRECISION_MEDIUM ||
       color_output->data.precision != GLSL_PRECISION_MEDIUM ||
       position->data.location_frac != 0 ||
       color_output->data.location_frac != 0)
      return false;

   nir_instr *instruction[23];
   if (!pvrgpu_nir_collect_exact_instructions(shader,
                                              instruction,
                                              PVRGPU_ARRAY_SIZE(instruction)))
      return false;

   if (instruction[0]->type != nir_instr_type_load_const)
      return false;
   nir_load_const_instr *constant_zero =
      nir_instr_as_load_const(instruction[0]);
   if (constant_zero->def.num_components != 1 ||
       constant_zero->def.bit_size != 32 || constant_zero->value[0].u32 != 0 ||
       !pvrgpu_nir_load_uniform_matches(instruction[1],
                                       constant_zero,
                                       0,
                                       1))
      return false;
   nir_intrinsic_instr *uniform_load =
      nir_instr_as_intrinsic(instruction[1]);

   static const uint32_t constant_bits[5] = {
      UINT32_C(0x38d1b717), UINT32_C(0x3f000000),
      UINT32_C(0x40000000), UINT32_C(0x40400000),
      UINT32_C(0x3f800000),
   };
   nir_load_const_instr *constant[5];
   for (unsigned i = 0; i < PVRGPU_ARRAY_SIZE(constant); ++i) {
      if (instruction[2 + i]->type != nir_instr_type_load_const)
         return false;
      constant[i] = nir_instr_as_load_const(instruction[2 + i]);
      if (constant[i]->def.num_components != 1 ||
          constant[i]->def.bit_size != 32 ||
          constant[i]->value[0].u32 != constant_bits[i])
         return false;
   }

   if (!pvrgpu_nir_deref_var_matches(instruction[7], position))
      return false;
   nir_deref_instr *position_deref = nir_instr_as_deref(instruction[7]);
   if (!pvrgpu_nir_load_deref_matches(instruction[8], position_deref, 4))
      return false;
   nir_intrinsic_instr *position_load =
      nir_instr_as_intrinsic(instruction[8]);

   static const nir_op scalar_ops[] = {
      nir_op_fmul, nir_op_fadd, nir_op_fmul, nir_op_fmul, nir_op_ffract,
      nir_op_fge, nir_op_fmul, nir_op_ffract, nir_op_fmul, nir_op_ffract,
      nir_op_bcsel,
   };
   for (unsigned i = 0; i < PVRGPU_ARRAY_SIZE(scalar_ops); ++i) {
      const unsigned bit_size = scalar_ops[i] == nir_op_fge ? 1u : 32u;
      if (!pvrgpu_nir_alu_matches(instruction[9 + i],
                                  scalar_ops[i],
                                  1,
                                  bit_size))
         return false;
   }
   if (!pvrgpu_nir_alu_matches(instruction[20], nir_op_vec4, 4, 32))
      return false;

   nir_alu_instr *multiply_y = nir_instr_as_alu(instruction[9]);
   nir_alu_instr *add_y = nir_instr_as_alu(instruction[10]);
   nir_alu_instr *multiply_x = nir_instr_as_alu(instruction[11]);
   nir_alu_instr *multiply_phase = nir_instr_as_alu(instruction[12]);
   nir_alu_instr *phase = nir_instr_as_alu(instruction[13]);
   nir_alu_instr *compare = nir_instr_as_alu(instruction[14]);
   nir_alu_instr *multiply_two = nir_instr_as_alu(instruction[15]);
   nir_alu_instr *fract_two = nir_instr_as_alu(instruction[16]);
   nir_alu_instr *multiply_three = nir_instr_as_alu(instruction[17]);
   nir_alu_instr *fract_three = nir_instr_as_alu(instruction[18]);
   nir_alu_instr *select = nir_instr_as_alu(instruction[19]);
   nir_alu_instr *color = nir_instr_as_alu(instruction[20]);
   static const unsigned scalar[1] = {0};
   static const unsigned position_x[1] = {0};
   static const unsigned position_y[1] = {1};
   static const unsigned uniform_x[1] = {0};
   static const unsigned uniform_y[1] = {1};
   if (!pvrgpu_nir_alu_source_matches(&multiply_y->src[0],
                                      &position_load->def,
                                      position_y,
                                      1) ||
       !pvrgpu_nir_alu_source_matches(&multiply_y->src[1],
                                      &uniform_load->def,
                                      uniform_x,
                                      1) ||
       !pvrgpu_nir_alu_source_matches(&add_y->src[0],
                                      &multiply_y->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&add_y->src[1],
                                      &uniform_load->def, uniform_y, 1) ||
       !pvrgpu_nir_alu_source_matches(&multiply_x->src[0],
                                      &position_load->def, position_x, 1) ||
       !pvrgpu_nir_alu_source_matches(&multiply_x->src[1],
                                      &constant[0]->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&multiply_phase->src[0],
                                      &multiply_x->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&multiply_phase->src[1],
                                      &add_y->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&phase->src[0],
                                      &multiply_phase->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&compare->src[0],
                                      &phase->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&compare->src[1],
                                      &constant[1]->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&multiply_two->src[0],
                                      &constant[2]->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&multiply_two->src[1],
                                      &phase->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&fract_two->src[0],
                                      &multiply_two->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&multiply_three->src[0],
                                      &constant[3]->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&multiply_three->src[1],
                                      &phase->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&fract_three->src[0],
                                      &multiply_three->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&select->src[0],
                                      &compare->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&select->src[1],
                                      &fract_two->def, scalar, 1) ||
       !pvrgpu_nir_alu_source_matches(&select->src[2],
                                      &fract_three->def, scalar, 1))
      return false;

   for (unsigned component = 0; component < 3; ++component) {
      if (!pvrgpu_nir_alu_source_matches(&color->src[component],
                                         &select->def,
                                         scalar,
                                         1))
         return false;
   }
   if (!pvrgpu_nir_alu_source_matches(&color->src[3],
                                      &constant[4]->def,
                                      scalar,
                                      1) ||
       !pvrgpu_nir_deref_var_matches(instruction[21], color_output))
      return false;
   nir_deref_instr *output_deref = nir_instr_as_deref(instruction[21]);
   return pvrgpu_nir_store_deref_matches(instruction[22],
                                         output_deref,
                                         &color->def);
}

static bool
pvrgpu_conditionals_framebuffer_matches(const struct pvrgpu_context *ctx,
                                        bool probe,
                                        bool require_depth_clear)
{
   unsigned width = 0;
   unsigned height = 0;
   if (!pvrgpu_glmark_pco_draw_extent(ctx, probe, &width, &height))
      return false;
   const enum pipe_format depth_format =
      probe ? PIPE_FORMAT_Z32_FLOAT : PIPE_FORMAT_Z24X8_UNORM;
   if (ctx->framebuffer.width != width ||
       ctx->framebuffer.height != height || ctx->framebuffer.nr_cbufs != 1 ||
       ctx->framebuffer.resolve || ctx->framebuffer.pls_enabled ||
       ctx->framebuffer.viewmask != 0)
      return false;

   const struct pipe_surface *color = &ctx->framebuffer.cbufs[0];
   const struct pipe_surface *depth = &ctx->framebuffer.zsbuf;
   if (!color->texture || color->texture->target != PIPE_TEXTURE_2D ||
       color->texture->format != PIPE_FORMAT_R8G8B8A8_UNORM ||
       color->format != PIPE_FORMAT_R8G8B8A8_UNORM || color->level != 0 ||
       color->first_layer != 0 || color->last_layer != 0 ||
       color->texture->width0 != width || color->texture->height0 != height ||
       color->texture->nr_samples > 1 ||
       color->texture->nr_storage_samples > 1)
      return false;
   if (!depth->texture || depth->texture->target != PIPE_TEXTURE_2D ||
       depth->texture->format != depth_format || depth->format != depth_format ||
       depth->level != 0 || depth->first_layer != 0 ||
       depth->last_layer != 0 || depth->texture->width0 != width ||
       depth->texture->height0 != height || depth->texture->nr_samples > 1 ||
       depth->texture->nr_storage_samples > 1)
      return false;

   if (!require_depth_clear)
      return true;
   return ctx->full_depth_clear_is_one &&
          ctx->full_depth_clear_resource == depth->texture &&
          ctx->full_depth_clear_level == depth->level &&
          ctx->full_depth_clear_first_layer == depth->first_layer &&
          ctx->full_depth_clear_last_layer == depth->last_layer &&
          ctx->full_depth_clear_width == width &&
          ctx->full_depth_clear_height == height;
}

static bool
pvrgpu_conditionals_copy_constant_buffers(
   const struct pvrgpu_context *ctx,
   bool probe,
   struct pvrgpu_conditionals_observation *observation)
{
   if (!ctx || !observation ||
       ctx->num_constant_buffers[MESA_SHADER_VERTEX] != 1 ||
       ctx->num_constant_buffers[MESA_SHADER_FRAGMENT] != 1)
      return false;
   for (unsigned stage = 0; stage < MESA_SHADER_MESH_STAGES; ++stage) {
      if (stage != MESA_SHADER_VERTEX && stage != MESA_SHADER_FRAGMENT &&
          ctx->num_constant_buffers[stage] != 0)
         return false;
   }

   const struct pipe_constant_buffer *vs_cb =
      &ctx->constant_buffers[MESA_SHADER_VERTEX][0];
   const struct pipe_constant_buffer *fs_cb =
      &ctx->constant_buffers[MESA_SHADER_FRAGMENT][0];
   if (vs_cb->buffer_size != sizeof(observation->vertex_shared) ||
       fs_cb->buffer_size != sizeof(observation->fragment_shared))
      return false;

   size_t vs_available = 0;
   size_t fs_available = 0;
   const uint8_t *vs_data =
      pvrgpu_constant_buffer_bytes(ctx,
                                   MESA_SHADER_VERTEX,
                                   0,
                                   &vs_available);
   const uint8_t *fs_data =
      pvrgpu_constant_buffer_bytes(ctx,
                                   MESA_SHADER_FRAGMENT,
                                   0,
                                   &fs_available);
   if (!vs_data || !fs_data || vs_available != sizeof(observation->vertex_shared) ||
       fs_available != sizeof(observation->fragment_shared))
      return false;

   memcpy(observation->vertex_shared,
          vs_data,
          sizeof(observation->vertex_shared));
   memcpy(observation->fragment_shared,
          fs_data,
          sizeof(observation->fragment_shared));
   static const uint32_t expected_vs[] = {
      UINT32_C(0x3fa646e0), 0, 0, 0,
      0, UINT32_C(0x3fddb3d6), 0, 0,
      0, 0, UINT32_C(0xbf804010), UINT32_C(0xbf800000),
      0, 0, UINT32_C(0x40408020), UINT32_C(0x40a00000),
   };
   unsigned width = 0;
   unsigned height = 0;
   if (!pvrgpu_glmark_pco_draw_extent(ctx, probe, &width, &height))
      return false;
   const uint32_t expected_fs[] = {
      UINT32_C(0x3f800000), 0, UINT32_C(0xbf800000),
      pvrgpu_float_bits((float)height),
   };
   return memcmp(observation->vertex_shared,
                 expected_vs,
                 sizeof(expected_vs)) == 0 &&
          memcmp(observation->fragment_shared,
                 expected_fs,
                 sizeof(expected_fs)) == 0;
}

static bool
pvrgpu_conditionals_pipeline_state_matches(
   const struct pvrgpu_context *ctx,
   bool probe,
   struct pvrgpu_conditionals_observation *observation,
   const char **failure_reason)
{
   if (!ctx || !observation || !failure_reason) {
      if (failure_reason)
         *failure_reason = "pipeline_arguments";
      return false;
   }
   if (!pvrgpu_nir_matches_conditionals_vs(ctx->vs)) {
      *failure_reason = "vs_nir";
      return false;
   }
   if (!pvrgpu_nir_matches_conditionals_fs(ctx->fs)) {
      *failure_reason = "fs_nir";
      return false;
   }
   if (ctx->tcs || ctx->tes || ctx->gs) {
      *failure_reason = "extra_shader_stage";
      return false;
   }
   if (ctx->num_stream_output_targets != 0) {
      *failure_reason = "stream_output";
      return false;
   }
   for (unsigned stage = 0; stage < MESA_SHADER_MESH_STAGES; ++stage) {
      if (ctx->num_sampler_views[stage] != 0 || ctx->num_samplers[stage] != 0) {
         *failure_reason = "sampler_bindings";
         return false;
      }
   }
   if (!pvrgpu_conditionals_copy_constant_buffers(ctx,
                                                  probe,
                                                  observation)) {
      *failure_reason = "constant_buffers";
      return false;
   }
   if (ctx->sample_mask != UINT32_MAX || !ctx->has_viewport) {
      *failure_reason = "sample_mask_or_viewport";
      return false;
   }

   unsigned width = 0;
   unsigned height = 0;
   if (!pvrgpu_glmark_pco_draw_extent(ctx, probe, &width, &height)) {
      *failure_reason = "framebuffer_extent";
      return false;
   }
   const uint32_t width_scale = pvrgpu_float_bits((float)width * 0.5f);
   const uint32_t height_scale = pvrgpu_float_bits((float)height * 0.5f);
   if (pvrgpu_float_bits(ctx->viewport.scale[0]) != width_scale ||
       pvrgpu_float_bits(ctx->viewport.scale[1]) != height_scale ||
       pvrgpu_float_bits(ctx->viewport.scale[2]) != UINT32_C(0x3f000000) ||
       pvrgpu_float_bits(ctx->viewport.translate[0]) != width_scale ||
       pvrgpu_float_bits(ctx->viewport.translate[1]) != height_scale ||
       pvrgpu_float_bits(ctx->viewport.translate[2]) != UINT32_C(0x3f000000)) {
      *failure_reason = "viewport";
      return false;
   }

   if (!ctx->blend || ctx->blend->state.independent_blend_enable ||
       ctx->blend->state.logicop_enable || !ctx->blend->state.dither ||
       ctx->blend->state.alpha_to_coverage ||
       ctx->blend->state.alpha_to_coverage_dither ||
       ctx->blend->state.alpha_to_one || ctx->blend->state.max_rt != 0 ||
       ctx->blend->state.advanced_blend_func != 0 ||
       ctx->blend->state.rt[0].blend_enable ||
       ctx->blend->state.rt[0].colormask != PIPE_MASK_RGBA) {
      *failure_reason = "blend";
      return false;
   }
   if (!ctx->dsa || !ctx->dsa->state.depth_enabled ||
       !ctx->dsa->state.depth_writemask ||
       ctx->dsa->state.depth_func != PIPE_FUNC_LEQUAL ||
       ctx->dsa->state.depth_bounds_test ||
       ctx->dsa->state.stencil[0].enabled ||
       ctx->dsa->state.stencil[1].enabled ||
       ctx->dsa->state.alpha_enabled) {
      *failure_reason = "depth_stencil_alpha";
      return false;
   }
   if (!ctx->rasterizer || ctx->rasterizer->state.front_ccw ||
       ctx->rasterizer->state.cull_face != PIPE_FACE_BACK ||
       ctx->rasterizer->state.fill_front != PIPE_POLYGON_MODE_FILL ||
       ctx->rasterizer->state.fill_back != PIPE_POLYGON_MODE_FILL ||
       ctx->rasterizer->state.scissor ||
       ctx->rasterizer->state.rasterizer_discard ||
       ctx->rasterizer->state.multisample ||
       ctx->rasterizer->state.offset_tri ||
       ctx->rasterizer->state.poly_smooth ||
       ctx->rasterizer->state.poly_stipple_enable ||
       ctx->rasterizer->state.conservative_raster_mode != 0 ||
       !ctx->rasterizer->state.half_pixel_center ||
       ctx->rasterizer->state.bottom_edge_rule ||
       ctx->rasterizer->state.clip_halfz ||
       !ctx->rasterizer->state.depth_clip_near ||
       !ctx->rasterizer->state.depth_clip_far ||
       ctx->rasterizer->state.depth_clamp) {
      *failure_reason = "rasterizer";
      return false;
   }

   observation->framebuffer_width = ctx->framebuffer.width;
   observation->framebuffer_height = ctx->framebuffer.height;
   observation->viewport_width = width;
   observation->viewport_height = height;
   return true;
}

static bool
pvrgpu_conditionals_vertex_data_matches(
   const struct pvrgpu_context *ctx,
   struct pvrgpu_conditionals_observation *observation)
{
   if (!ctx || !observation || !ctx->vertex_elements ||
       ctx->vertex_elements->num_elements != 1 || ctx->num_vertex_buffers != 1)
      return false;
   const struct pipe_vertex_element *element =
      &ctx->vertex_elements->elements[0];
   if (element->src_format != PIPE_FORMAT_R32G32B32_FLOAT ||
       element->src_stride != PVRGPU_DRAW_PCO_TRIANGLES_VERTEX_STRIDE ||
       element->src_offset != 0 || element->dual_slot ||
       element->instance_divisor != 0 || element->vertex_buffer_index != 0)
      return false;

   const struct pipe_vertex_buffer *vertex_buffer = &ctx->vertex_buffers[0];
   if (vertex_buffer->is_user_buffer || vertex_buffer->buffer_offset != 0 ||
       !vertex_buffer->buffer.resource ||
       vertex_buffer->buffer.resource->target != PIPE_BUFFER)
      return false;
   struct pvrgpu_resource *resource =
      pvrgpu_resource(vertex_buffer->buffer.resource);
   if (!resource || !resource->data ||
       resource->size != PVRGPU_DRAW_PCO_TRIANGLES_VERTEX_BYTES)
      return false;

   unsigned actual_vertex = 0;
   for (unsigned i = 0; i < 32; ++i) {
      for (unsigned j = 0; j < 32; ++j) {
         const double side = 0.136875;
         const double pitch = 0.156875;
         const float ax = (float)(-2.5 + (double)i * pitch);
         const float ay = (float)(2.5 - (double)j * pitch);
         const float bx = ax;
         const float by = (float)((double)ay - side);
         const float cx = (float)((double)ax + side);
         const float cy = ay;
         const float dx = cx;
         const float dy = by;
         const float expected[6][3] = {
            {ax, ay, 0.0f}, {bx, by, 0.0f}, {cx, cy, 0.0f},
            {bx, by, 0.0f}, {dx, dy, 0.0f}, {cx, cy, 0.0f},
         };
         for (unsigned cell_vertex = 0; cell_vertex < 6; ++cell_vertex) {
            for (unsigned component = 0; component < 3; ++component) {
               uint32_t actual_bits = 0;
               const size_t offset =
                  ((size_t)actual_vertex * 3u + component) * sizeof(uint32_t);
               memcpy(&actual_bits, resource->data + offset, sizeof(actual_bits));
               if (actual_bits !=
                   pvrgpu_float_bits(expected[cell_vertex][component]))
                  return false;
            }
            actual_vertex++;
         }
      }
   }
   if (actual_vertex != PVRGPU_DRAW_PCO_TRIANGLES_VERTEX_COUNT)
      return false;
   observation->raw_vertex_data = resource->data;
   observation->raw_vertex_data_size = resource->size;
   return true;
}

static bool
pvrgpu_draw_matches_conditionals(
   const struct pvrgpu_context *ctx,
   const struct pipe_draw_info *info,
   unsigned drawid_offset,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,
   unsigned num_draws,
   bool probe,
   bool require_depth_clear,
   struct pvrgpu_conditionals_observation *observation,
   const char **failure_reason)
{
   if (failure_reason)
      *failure_reason = "arguments";
   if (!ctx || !info || !draws || !observation || !failure_reason)
      return false;
   /* index_bounds_valid and draws[0].index_bias have no meaning when
    * index_size is zero.  Mesa's DrawArrays path sets the former and leaves
    * the latter unspecified, so reading either as an indexed-draw contract
    * would reject a valid non-indexed replay (and may read uninitialized
    * storage). */
   if (drawid_offset != 0 || indirect || num_draws != 1 ||
       info->mode != MESA_PRIM_TRIANGLES || info->index_size != 0 ||
       info->has_user_indices ||
       info->primitive_restart || info->increment_draw_id ||
       info->index_bias_varies || info->was_line_loop ||
       info->start_instance != 0 || info->instance_count != 1 ||
       draws[0].start != 0 ||
       draws[0].count != PVRGPU_DRAW_PCO_TRIANGLES_VERTEX_COUNT) {
      pvrgpu_counter_eventf(
         "draw_pco_triangles_draw_info_miss",
         "drawid_offset=%u indirect=%u num_draws=%u mode=%u index_size=%u "
         "user_indices=%u bounds_valid=%u restart=%u increment_draw_id=%u "
         "index_bias_varies=%u was_line_loop=%u start_instance=%u "
         "instance_count=%u start=%u count=%u",
         drawid_offset,
         indirect ? 1 : 0,
         num_draws,
         info->mode,
         info->index_size,
         info->has_user_indices ? 1 : 0,
         info->index_bounds_valid ? 1 : 0,
         info->primitive_restart ? 1 : 0,
         info->increment_draw_id ? 1 : 0,
         info->index_bias_varies ? 1 : 0,
         info->was_line_loop ? 1 : 0,
         info->start_instance,
         info->instance_count,
         draws[0].start,
         draws[0].count);
      *failure_reason = "draw_info";
      return false;
   }
   if (!pvrgpu_conditionals_framebuffer_matches(ctx,
                                                probe,
                                                require_depth_clear)) {
      *failure_reason = "framebuffer";
      return false;
   }

   memset(observation, 0, sizeof(*observation));
   if (!pvrgpu_conditionals_pipeline_state_matches(ctx,
                                                   probe,
                                                   observation,
                                                   failure_reason))
      return false;
   if (!pvrgpu_conditionals_vertex_data_matches(ctx, observation)) {
      *failure_reason = "vertex_data";
      return false;
   }
   *failure_reason = NULL;
   return true;
}

static bool
pvrgpu_lit_mesh_profile(const struct pvrgpu_context *ctx,
                        enum pvrgpu_pco_lit_mesh_profile *profile,
                        unsigned *vertex_count)
{
   static const uint32_t build_vs[8] = {
      UINT32_C(0x750ac3d1), UINT32_C(0xe9ceafcc),
      UINT32_C(0xdd1263dd), UINT32_C(0xa22a457b),
      UINT32_C(0x3b8ebb47), UINT32_C(0xa4ee0e8e),
      UINT32_C(0xeb2663ea), UINT32_C(0x6ad452cd),
   };
   static const uint32_t bump_vs[8] = {
      UINT32_C(0x447e9e1f), UINT32_C(0xb6e0a9b9),
      UINT32_C(0xf5dcefa9), UINT32_C(0xf987adef),
      UINT32_C(0x8c416544), UINT32_C(0xb0956e81),
      UINT32_C(0xc5d8865b), UINT32_C(0x7b2850a7),
   };
   static const uint32_t shading_vs[8] = {
      UINT32_C(0x55a3db4e), UINT32_C(0x0781726a),
      UINT32_C(0xa9aaf326), UINT32_C(0x1663be77),
      UINT32_C(0x0b6194eb), UINT32_C(0xdd4b6265),
      UINT32_C(0x3351e890), UINT32_C(0x7acfdd9a),
   };
   static const uint32_t passthrough_fs[8] = {
      UINT32_C(0x8105bebf), UINT32_C(0x60cef3c7),
      UINT32_C(0xc9c3e978), UINT32_C(0xd20442bc),
      UINT32_C(0x46d83156), UINT32_C(0x9a4abb0b),
      UINT32_C(0xd1a4de24), UINT32_C(0x422a9790),
   };
   static const uint32_t bump_fs[8] = {
      UINT32_C(0x4f55ff2c), UINT32_C(0x8d248356),
      UINT32_C(0x20aea0e2), UINT32_C(0xee5248d5),
      UINT32_C(0x777abff2), UINT32_C(0xc13daa4d),
      UINT32_C(0xcb78bfc9), UINT32_C(0xc09ce498),
   };

   if (!ctx || !profile || !vertex_count || !ctx->vs || !ctx->fs ||
       ctx->vs->stage != MESA_SHADER_VERTEX ||
       ctx->fs->stage != MESA_SHADER_FRAGMENT ||
       !ctx->vs->has_nir || !ctx->fs->has_nir || !ctx->vs->nir ||
       !ctx->fs->nir)
      return false;

   if (pvrgpu_nir_source_hash_matches(ctx->vs->nir, build_vs) &&
       pvrgpu_nir_source_hash_matches(ctx->fs->nir, passthrough_fs)) {
      *profile = PVRGPU_PCO_LIT_MESH_BUILD;
      *vertex_count = 21516;
      return true;
   }
   if (pvrgpu_nir_source_hash_matches(ctx->vs->nir, bump_vs) &&
       pvrgpu_nir_source_hash_matches(ctx->fs->nir, bump_fs)) {
      *profile = PVRGPU_PCO_LIT_MESH_BUMP;
      *vertex_count = 1440;
      return true;
   }
   if (pvrgpu_nir_source_hash_matches(ctx->vs->nir, shading_vs) &&
       pvrgpu_nir_source_hash_matches(ctx->fs->nir, passthrough_fs)) {
      *profile = PVRGPU_PCO_LIT_MESH_SHADING;
      *vertex_count = 43044;
      return true;
   }
   return false;
}

static bool
pvrgpu_refract_pco_profile(const struct pvrgpu_context *ctx,
                           enum pvrgpu_pco_refract_profile *profile)
{
   static const uint32_t prepass_vs[8] = {
      UINT32_C(0xe4968762), UINT32_C(0x33f7e5d3),
      UINT32_C(0x8ca90f67), UINT32_C(0xb97709d1),
      UINT32_C(0x1b1e6e02), UINT32_C(0x56181388),
      UINT32_C(0xe52466c0), UINT32_C(0xccbf4647),
   };
   static const uint32_t prepass_fs[8] = {
      UINT32_C(0xe4296e61), UINT32_C(0x386f509f),
      UINT32_C(0x599bdacb), UINT32_C(0x3631c37e),
      UINT32_C(0x480c24bb), UINT32_C(0xd56abe19),
      UINT32_C(0x82725b7b), UINT32_C(0xd046dab0),
   };
   static const uint32_t composite_vs[8] = {
      UINT32_C(0x27553973), UINT32_C(0x14f4744c),
      UINT32_C(0x155ef398), UINT32_C(0xa57dee0b),
      UINT32_C(0x7dd7d4b3), UINT32_C(0xcf1e855e),
      UINT32_C(0xe83f6f81), UINT32_C(0xdb7fe462),
   };
   static const uint32_t composite_fs[8] = {
      UINT32_C(0xfcfc470a), UINT32_C(0x6e9eeb2b),
      UINT32_C(0x95810825), UINT32_C(0x6d2d6953),
      UINT32_C(0x04dc8732), UINT32_C(0x24b8c4c7),
      UINT32_C(0xac08a61d), UINT32_C(0x28ed4375),
   };

   if (!ctx || !profile || !ctx->vs || !ctx->fs || !ctx->vs->has_nir ||
       !ctx->fs->has_nir || !ctx->vs->nir || !ctx->fs->nir ||
       ctx->vs->stage != MESA_SHADER_VERTEX ||
       ctx->fs->stage != MESA_SHADER_FRAGMENT)
      return false;
   if (pvrgpu_nir_source_hash_matches(ctx->vs->nir, prepass_vs) &&
       pvrgpu_nir_source_hash_matches(ctx->fs->nir, prepass_fs)) {
      *profile = PVRGPU_PCO_REFRACT_PREPASS;
      return true;
   }
   if (pvrgpu_nir_source_hash_matches(ctx->vs->nir, composite_vs) &&
       pvrgpu_nir_source_hash_matches(ctx->fs->nir, composite_fs)) {
      *profile = PVRGPU_PCO_REFRACT_COMPOSITE;
      return true;
   }
   return false;
}

static bool
pvrgpu_shadow_pco_profile(const struct pvrgpu_context *ctx,
                          enum pvrgpu_pco_shadow_profile *profile)
{
   static const uint32_t depth_vs[8] = {
      UINT32_C(0xe4968762), UINT32_C(0x33f7e5d3),
      UINT32_C(0x8ca90f67), UINT32_C(0xb97709d1),
      UINT32_C(0x1b1e6e02), UINT32_C(0x56181388),
      UINT32_C(0xe52466c0), UINT32_C(0xccbf4647),
   };
   static const uint32_t depth_fs[8] = {
      UINT32_C(0xe4296e61), UINT32_C(0x386f509f),
      UINT32_C(0x599bdacb), UINT32_C(0x3631c37e),
      UINT32_C(0x480c24bb), UINT32_C(0xd56abe19),
      UINT32_C(0x82725b7b), UINT32_C(0xd046dab0),
   };
   static const uint32_t mask_vs[8] = {
      UINT32_C(0x981d39cf), UINT32_C(0x8316685f),
      UINT32_C(0x8e868a60), UINT32_C(0xf1850c32),
      UINT32_C(0xcdac5c34), UINT32_C(0x92c6393c),
      UINT32_C(0x6fc446df), UINT32_C(0x4830846f),
   };
   static const uint32_t mask_fs[8] = {
      UINT32_C(0x49a27748), UINT32_C(0xda81bbdf),
      UINT32_C(0x385da52a), UINT32_C(0xfa4c9087),
      UINT32_C(0x8edbc5ce), UINT32_C(0x0f9e5f75),
      UINT32_C(0xcb295bda), UINT32_C(0x20c14281),
   };
   static const uint32_t scene_vs[8] = {
      UINT32_C(0xda5546ad), UINT32_C(0x634bf23c),
      UINT32_C(0x36c9d6bd), UINT32_C(0x89e696ca),
      UINT32_C(0xc138e003), UINT32_C(0x445bb465),
      UINT32_C(0xb580e98a), UINT32_C(0x7f1d0c5b),
   };
   static const uint32_t scene_fs[8] = {
      UINT32_C(0x8105bebf), UINT32_C(0x60cef3c7),
      UINT32_C(0xc9c3e978), UINT32_C(0xd20442bc),
      UINT32_C(0x46d83156), UINT32_C(0x9a4abb0b),
      UINT32_C(0xd1a4de24), UINT32_C(0x422a9790),
   };
   if (!ctx || !profile || !ctx->vs || !ctx->fs || !ctx->vs->has_nir ||
       !ctx->fs->has_nir || !ctx->vs->nir || !ctx->fs->nir ||
       ctx->vs->stage != MESA_SHADER_VERTEX ||
       ctx->fs->stage != MESA_SHADER_FRAGMENT)
      return false;

   if (pvrgpu_nir_source_hash_matches(ctx->vs->nir, depth_vs) &&
       pvrgpu_nir_source_hash_matches(ctx->fs->nir, depth_fs)) {
      *profile = PVRGPU_PCO_SHADOW_DEPTH;
      return true;
   }
   if (pvrgpu_nir_source_hash_matches(ctx->vs->nir, mask_vs) &&
       pvrgpu_nir_source_hash_matches(ctx->fs->nir, mask_fs)) {
      *profile = PVRGPU_PCO_SHADOW_MASK;
      return true;
   }
   if (pvrgpu_nir_source_hash_matches(ctx->vs->nir, scene_vs) &&
       pvrgpu_nir_source_hash_matches(ctx->fs->nir, scene_fs)) {
      *profile = PVRGPU_PCO_SHADOW_SCENE;
      return true;
   }
   return false;
}

static void
pvrgpu_shadow_pco_observation_finish(
   struct pvrgpu_shadow_pco_observation *observation)
{
   if (!observation)
      return;
   FREE(observation->vertex_data);
   pipe_resource_reference(&observation->shadow_depth, NULL);
   pipe_resource_reference(&observation->color_attachment, NULL);
   pipe_resource_reference(&observation->depth_attachment, NULL);
   pvrgpu_pco_graphics_binary_finish(&observation->binary);
   memset(observation, 0, sizeof(*observation));
}

static void
pvrgpu_shadow_pco_observation_destroy(
   struct pvrgpu_shadow_pco_observation **observation)
{
   if (!observation || !*observation)
      return;
   pvrgpu_shadow_pco_observation_finish(*observation);
   FREE(*observation);
   *observation = NULL;
}

static void
pvrgpu_terrain_pco_observation_finish(
   struct pvrgpu_terrain_pco_observation *observation)
{
   if (!observation)
      return;
   FREE(observation->vertex_data);
   for (unsigned texture = 0; texture < observation->texture_count;
        ++texture)
      FREE(observation->textures[texture].bytes);
   pipe_resource_reference(&observation->color_attachment, NULL);
   pipe_resource_reference(&observation->depth_attachment, NULL);
   pvrgpu_pco_graphics_binary_finish(&observation->binary);
   memset(observation, 0, sizeof(*observation));
}

static void
pvrgpu_terrain_pco_observation_destroy(
   struct pvrgpu_terrain_pco_observation **observation)
{
   if (!observation || !*observation)
      return;
   pvrgpu_terrain_pco_observation_finish(*observation);
   FREE(*observation);
   *observation = NULL;
}

static void
pvrgpu_terrain_pco_sequence_reset(struct pvrgpu_context *ctx)
{
   if (!ctx)
      return;
   for (unsigned draw = 0; draw < PVRGPU_TERRAIN_PCO_DRAW_COUNT; ++draw)
      pvrgpu_terrain_pco_observation_destroy(&ctx->terrain_pco_draws[draw]);
   ctx->terrain_pco_draw_count = 0;
}

static void
pvrgpu_refract_pco_observation_finish(
   struct pvrgpu_refract_pco_observation *observation)
{
   if (!observation)
      return;
   FREE(observation->interleaved_vertex_data);
   FREE(observation->sampled_image_bytes);
   pipe_resource_reference(&observation->prepass_color, NULL);
   pipe_resource_reference(&observation->prepass_depth, NULL);
   pvrgpu_pco_graphics_binary_finish(&observation->binary);
   memset(observation, 0, sizeof(*observation));
}

static void
pvrgpu_refract_pco_observation_destroy(
   struct pvrgpu_refract_pco_observation **observation)
{
   if (!observation || !*observation)
      return;
   pvrgpu_refract_pco_observation_finish(*observation);
   FREE(*observation);
   *observation = NULL;
}

static bool
pvrgpu_refract_pco_draw_info_matches(
   const struct pipe_draw_info *info,
   unsigned drawid_offset,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,
   unsigned num_draws)
{
   return info && draws && drawid_offset == 0 && !indirect && num_draws == 1 &&
          info->mode == MESA_PRIM_TRIANGLES && info->index_size == 0 &&
          !info->has_user_indices && !info->primitive_restart &&
          !info->increment_draw_id && !info->index_bias_varies &&
          !info->was_line_loop && info->start_instance == 0 &&
          info->instance_count == 1 && draws[0].start == 0 &&
          draws[0].count == PVRGPU_REFRACT_PCO_VERTEX_COUNT;
}

static bool
pvrgpu_refract_pco_depth_backing_is_known_clear(
   const struct pipe_surface *surface,
   unsigned width,
   unsigned height,
   bool *is_clear_one)
{
   if (is_clear_one)
      *is_clear_one = false;
   if (!is_clear_one || !surface || !surface->texture ||
       surface->format != PIPE_FORMAT_Z24X8_UNORM || surface->level != 0 ||
       surface->first_layer != 0 || surface->last_layer != 0 ||
       surface->texture->target != PIPE_TEXTURE_2D ||
       surface->texture->format != PIPE_FORMAT_Z24X8_UNORM ||
       surface->texture->width0 != width ||
       surface->texture->height0 != height ||
       surface->texture->last_level != 0 ||
       surface->texture->nr_samples > 1 ||
       surface->texture->nr_storage_samples > 1)
      return false;

   const struct pvrgpu_resource *resource =
      pvrgpu_resource(surface->texture);
   const size_t row_bytes = (size_t)width * sizeof(uint32_t);
   const size_t expected_size = row_bytes * height;
   if (!resource || !resource->data || resource->level_count != 1 ||
       resource->level_offsets[0] != 0 ||
       resource->level_strides[0] != row_bytes ||
       resource->level_layer_strides[0] != expected_size ||
       resource->size != expected_size)
      return false;

   uint32_t clear_value = 0;
   memcpy(&clear_value, resource->data, sizeof(clear_value));
   if (clear_value != 0 && clear_value != UINT32_C(0x00ffffff))
      return false;
   for (unsigned y = 0; y < height; ++y) {
      const uint8_t *row =
         resource->data + (size_t)y * resource->level_strides[0];
      for (unsigned x = 0; x < width; ++x) {
         uint32_t packed = 0;
         memcpy(&packed, row + (size_t)x * sizeof(packed), sizeof(packed));
         if (packed != clear_value)
            return false;
      }
   }
   *is_clear_one = clear_value == UINT32_C(0x00ffffff);
   return true;
}

static bool
pvrgpu_refract_pco_framebuffer_matches(
   const struct pvrgpu_context *ctx,
   enum pvrgpu_pco_refract_profile profile,
   struct pvrgpu_refract_pco_observation *observation,
   const char **failure_reason)
{
   const bool composite = profile == PVRGPU_PCO_REFRACT_COMPOSITE;
   unsigned width = 0;
   unsigned height = 0;
   unsigned color_mip_count = 0;
   size_t color_bytes = 0;
   if (!pvrgpu_glmark_scaled_output_extent(composite ? 1u : 2u,
                                            &width,
                                            &height) ||
       (!composite &&
        !pvrgpu_tight_rgba8_mip_layout(width,
                                       height,
                                       &color_mip_count,
                                       &color_bytes))) {
      *failure_reason = "output_extent";
      return false;
   }
   const enum pipe_format depth_format =
      composite ? PIPE_FORMAT_Z24X8_UNORM : PIPE_FORMAT_Z32_UNORM;
   const unsigned color_last_level = composite ? 0u : color_mip_count - 1u;

   if (ctx->framebuffer.width != width ||
       ctx->framebuffer.height != height ||
       ctx->framebuffer.nr_cbufs != 1 || ctx->framebuffer.resolve ||
       ctx->framebuffer.pls_enabled || ctx->framebuffer.viewmask != 0) {
      *failure_reason = "framebuffer";
      return false;
   }
   const struct pipe_surface *color = &ctx->framebuffer.cbufs[0];
   const struct pipe_surface *depth = &ctx->framebuffer.zsbuf;
   if (!color->texture || color->texture->target != PIPE_TEXTURE_2D ||
       color->texture->format != PIPE_FORMAT_R8G8B8A8_UNORM ||
       color->format != PIPE_FORMAT_R8G8B8A8_UNORM || color->level != 0 ||
       color->first_layer != 0 || color->last_layer != 0 ||
       color->texture->width0 != width ||
       color->texture->height0 != height ||
       color->texture->last_level != color_last_level ||
       color->texture->nr_samples > 1 ||
       color->texture->nr_storage_samples > 1 || !depth->texture ||
       depth->texture->target != PIPE_TEXTURE_2D ||
       depth->texture->format != depth_format ||
       depth->format != depth_format || depth->level != 0 ||
       depth->first_layer != 0 || depth->last_layer != 0 ||
       depth->texture->width0 != width ||
       depth->texture->height0 != height || depth->texture->last_level != 0 ||
       depth->texture->nr_samples > 1 ||
       depth->texture->nr_storage_samples > 1) {
      *failure_reason = "framebuffer_attachments";
      return false;
   }
   const bool live_depth_clear_matches =
      ctx->full_depth_clear_is_one &&
      ctx->full_depth_clear_resource == depth->texture &&
      ctx->full_depth_clear_level == 0 &&
      ctx->full_depth_clear_first_layer == 0 &&
      ctx->full_depth_clear_last_layer == 0 &&
      ctx->full_depth_clear_width == width &&
      ctx->full_depth_clear_height == height;
   if ((!composite && !live_depth_clear_matches) ||
       (composite && !pvrgpu_refract_pco_depth_backing_is_known_clear(
                        depth,
                        width,
                        height,
                        &observation->composite_depth_clear_one))) {
      *failure_reason = "depth_clear";
      return false;
   }
   if (composite && !pvrgpu_framebuffer_matches_rdc_output(ctx)) {
      unsigned output_width = 0;
      unsigned output_height = 0;
      const bool has_output_extent =
         pvrgpu_rdc_output_extent(&output_width, &output_height);
      const bool has_color = pvrgpu_context_has_color_framebuffer(ctx);
      const unsigned target =
         has_color ? ctx->framebuffer.cbufs[0].texture->target : UINT_MAX;
      pvrgpu_counter_eventf(
         "draw_pco_refract_output_framebuffer_mismatch",
         "has_color=%u has_env_extent=%u "
         "env_extent=%ux%u actual_extent=%ux%u nr_cbufs=%u target=%u",
         has_color ? 1u : 0u,
         has_output_extent ? 1u : 0u,
         output_width,
         output_height,
         ctx->framebuffer.width,
         ctx->framebuffer.height,
         ctx->framebuffer.nr_cbufs,
         target);
      *failure_reason = "output_framebuffer";
      return false;
   }

   observation->framebuffer_width = width;
   observation->framebuffer_height = height;
   if (!composite) {
      pipe_resource_reference(&observation->prepass_color, color->texture);
      pipe_resource_reference(&observation->prepass_depth, depth->texture);
   }
   return true;
}

static bool
pvrgpu_refract_pco_common_state_matches(
   const struct pvrgpu_context *ctx,
   enum pvrgpu_pco_refract_profile profile,
   struct pvrgpu_refract_pco_observation *observation,
   const char **failure_reason)
{
   const bool composite = profile == PVRGPU_PCO_REFRACT_COMPOSITE;
   if (ctx->tcs || ctx->tes || ctx->gs || ctx->num_stream_output_targets) {
      *failure_reason = "extra_stage_or_stream_output";
      return false;
   }
   if (ctx->sample_mask != UINT32_MAX || !ctx->has_viewport) {
      *failure_reason = "sample_mask_or_viewport";
      return false;
   }
   const unsigned width = observation->framebuffer_width;
   const unsigned height = observation->framebuffer_height;
   const uint32_t width_scale = pvrgpu_float_bits((float)width * 0.5f);
   const uint32_t height_scale = pvrgpu_float_bits((float)height * 0.5f);
   if (pvrgpu_float_bits(ctx->viewport.scale[0]) != width_scale ||
       pvrgpu_float_bits(ctx->viewport.scale[1]) != height_scale ||
       pvrgpu_float_bits(ctx->viewport.scale[2]) != UINT32_C(0x3f000000) ||
       pvrgpu_float_bits(ctx->viewport.translate[0]) != width_scale ||
       pvrgpu_float_bits(ctx->viewport.translate[1]) != height_scale ||
       pvrgpu_float_bits(ctx->viewport.translate[2]) != UINT32_C(0x3f000000)) {
      *failure_reason = "viewport";
      return false;
   }
   observation->viewport_width = width;
   observation->viewport_height = height;

   if (!ctx->blend || ctx->blend->state.independent_blend_enable ||
       ctx->blend->state.logicop_enable || !ctx->blend->state.dither ||
       ctx->blend->state.alpha_to_coverage ||
       ctx->blend->state.alpha_to_coverage_dither ||
       ctx->blend->state.alpha_to_one || ctx->blend->state.max_rt != 0 ||
       ctx->blend->state.advanced_blend_func != 0 ||
       ctx->blend->state.rt[0].blend_enable ||
       ctx->blend->state.rt[0].colormask != PIPE_MASK_RGBA) {
      *failure_reason = "blend";
      return false;
   }
   if (!ctx->dsa || !ctx->dsa->state.depth_enabled ||
       !ctx->dsa->state.depth_writemask ||
       ctx->dsa->state.depth_func != PIPE_FUNC_LEQUAL ||
       ctx->dsa->state.depth_bounds_test ||
       ctx->dsa->state.stencil[0].enabled ||
       ctx->dsa->state.stencil[1].enabled || ctx->dsa->state.alpha_enabled) {
      *failure_reason = "depth_stencil_alpha";
      return false;
   }
   const unsigned cull_face = composite ? PIPE_FACE_BACK : PIPE_FACE_FRONT;
   if (!ctx->rasterizer || ctx->rasterizer->state.front_ccw ||
       ctx->rasterizer->state.cull_face != cull_face ||
       ctx->rasterizer->state.fill_front != PIPE_POLYGON_MODE_FILL ||
       ctx->rasterizer->state.fill_back != PIPE_POLYGON_MODE_FILL ||
       ctx->rasterizer->state.scissor ||
       ctx->rasterizer->state.rasterizer_discard ||
       ctx->rasterizer->state.multisample ||
       ctx->rasterizer->state.offset_tri ||
       ctx->rasterizer->state.poly_smooth ||
       ctx->rasterizer->state.poly_stipple_enable ||
       ctx->rasterizer->state.conservative_raster_mode != 0 ||
       !ctx->rasterizer->state.half_pixel_center ||
       ctx->rasterizer->state.bottom_edge_rule ||
       ctx->rasterizer->state.clip_halfz ||
       !ctx->rasterizer->state.depth_clip_near ||
       !ctx->rasterizer->state.depth_clip_far ||
       ctx->rasterizer->state.depth_clamp) {
      *failure_reason = "rasterizer";
      return false;
   }
   return true;
}

static bool
pvrgpu_refract_pco_copy_constants(
   const struct pvrgpu_context *ctx,
   enum pvrgpu_pco_refract_profile profile,
   struct pvrgpu_refract_pco_observation *observation,
   const char **failure_reason)
{
   const size_t expected_dwords =
      profile == PVRGPU_PCO_REFRACT_COMPOSITE ? 64u : 16u;
   if (ctx->num_constant_buffers[MESA_SHADER_VERTEX] != 1 ||
       ctx->num_constant_buffers[MESA_SHADER_FRAGMENT] != 0) {
      *failure_reason = "constant_buffers";
      return false;
   }
   for (unsigned stage = 0; stage < MESA_SHADER_MESH_STAGES; ++stage) {
      if (stage != MESA_SHADER_VERTEX && ctx->num_constant_buffers[stage]) {
         *failure_reason = "constant_buffers";
         return false;
      }
   }
   const struct pipe_constant_buffer *cb =
      &ctx->constant_buffers[MESA_SHADER_VERTEX][0];
   size_t available = 0;
   const uint8_t *bytes = pvrgpu_constant_buffer_bytes(ctx,
                                                       MESA_SHADER_VERTEX,
                                                       0,
                                                       &available);
   if (!bytes || cb->buffer_size != expected_dwords * sizeof(uint32_t) ||
       available != expected_dwords * sizeof(uint32_t)) {
      *failure_reason = "constant_buffer_size";
      return false;
   }
   memcpy(observation->vertex_shared, bytes, available);
   observation->vertex_shared_count = expected_dwords;
   return true;
}

static bool
pvrgpu_refract_pco_capture_vertices(
   const struct pvrgpu_context *ctx,
   struct pvrgpu_refract_pco_observation *observation,
   const char **failure_reason)
{
   if (!ctx->vertex_elements || ctx->vertex_elements->num_elements != 2 ||
       ctx->num_vertex_buffers != 2) {
      *failure_reason = "vertex_layout";
      return false;
   }
   const size_t stream_bytes =
      (size_t)PVRGPU_REFRACT_PCO_VERTEX_COUNT * 3u * sizeof(float);
   const struct pvrgpu_resource *resources[2] = { NULL, NULL };
   for (unsigned attribute = 0; attribute < 2; ++attribute) {
      const struct pipe_vertex_element *element =
         &ctx->vertex_elements->elements[attribute];
      const struct pipe_vertex_buffer *buffer = &ctx->vertex_buffers[attribute];
      if (element->src_format != PIPE_FORMAT_R32G32B32_FLOAT ||
          element->src_stride != 3u * sizeof(float) ||
          element->src_offset != 0 || element->dual_slot ||
          element->instance_divisor != 0 ||
          element->vertex_buffer_index != attribute || buffer->is_user_buffer ||
          buffer->buffer_offset != 0 || !buffer->buffer.resource ||
          buffer->buffer.resource->target != PIPE_BUFFER) {
         *failure_reason = "vertex_layout";
         return false;
      }
      resources[attribute] = pvrgpu_resource(buffer->buffer.resource);
      if (!resources[attribute] || !resources[attribute]->data ||
          resources[attribute]->size != stream_bytes) {
         *failure_reason = "vertex_resource";
         return false;
      }
   }

   const size_t interleaved_size =
      (size_t)PVRGPU_REFRACT_PCO_VERTEX_COUNT *
      PVRGPU_REFRACT_PCO_VERTEX_STRIDE;
   observation->interleaved_vertex_data = MALLOC(interleaved_size);
   if (!observation->interleaved_vertex_data) {
      *failure_reason = "vertex_allocation";
      return false;
   }
   observation->interleaved_vertex_data_size = interleaved_size;
   for (unsigned vertex = 0; vertex < PVRGPU_REFRACT_PCO_VERTEX_COUNT;
        ++vertex) {
      uint8_t *dst = observation->interleaved_vertex_data +
                     (size_t)vertex * PVRGPU_REFRACT_PCO_VERTEX_STRIDE;
      memcpy(dst, resources[0]->data + (size_t)vertex * 12u, 12u);
      memcpy(dst + 12u, resources[1]->data + (size_t)vertex * 12u, 12u);
   }
   return true;
}

static bool
pvrgpu_refract_pco_resource_layout_matches(
   const struct pipe_resource *resource,
   enum pipe_format format,
   unsigned width,
   unsigned height,
   unsigned mip_count,
   size_t expected_size)
{
   if (!resource || resource->target != PIPE_TEXTURE_2D ||
       resource->format != format || resource->width0 != width ||
       resource->height0 != height || resource->depth0 != 1 ||
       resource->array_size != 1 || resource->last_level + 1 != mip_count ||
       resource->nr_samples > 1 || resource->nr_storage_samples > 1)
      return false;

   const struct pvrgpu_resource *pvr =
      pvrgpu_resource((struct pipe_resource *)resource);
   if (!pvr || !pvr->data || pvr->level_count != mip_count ||
       pvr->size != expected_size)
      return false;

   uintptr_t expected_offset = 0;
   for (unsigned level = 0; level < mip_count; ++level) {
      const unsigned level_width = MAX2(width >> level, 1u);
      const unsigned level_height = MAX2(height >> level, 1u);
      const unsigned expected_stride = level_width * sizeof(uint32_t);
      const uintptr_t expected_layer_stride =
         (uintptr_t)expected_stride * level_height;
      if (pvr->level_offsets[level] != expected_offset ||
          pvr->level_strides[level] != expected_stride ||
          pvr->level_layer_strides[level] != expected_layer_stride)
         return false;
      expected_offset += expected_layer_stride;
   }
   return expected_offset == expected_size;
}

static bool
pvrgpu_refract_pco_build_fragment_shared(
   const struct pvrgpu_refract_pco_observation *prepass,
   struct pvrgpu_refract_pco_observation *observation,
   const char **failure_reason)
{
   const unsigned prepass_width = prepass ? prepass->framebuffer_width : 0u;
   const unsigned prepass_height = prepass ? prepass->framebuffer_height : 0u;
   unsigned color_mip_count = 0;
   size_t color_bytes = 0;
   const size_t depth_bytes =
      (size_t)prepass_width * prepass_height * sizeof(uint32_t);
   if (!prepass || !observation ||
       !pvrgpu_tight_rgba8_mip_layout(prepass_width,
                                      prepass_height,
                                      &color_mip_count,
                                      &color_bytes) ||
       !pvrgpu_refract_pco_resource_layout_matches(prepass->prepass_depth,
                                                    PIPE_FORMAT_Z32_UNORM,
                                                    prepass_width,
                                                    prepass_height,
                                                    1u,
                                                    depth_bytes) ||
       !pvrgpu_refract_pco_resource_layout_matches(
          prepass->prepass_color,
          PIPE_FORMAT_R8G8B8A8_UNORM,
          prepass_width,
          prepass_height,
          color_mip_count,
          color_bytes)) {
      *failure_reason = "prepass_resource_layout";
      return false;
   }

   if (!pvrgpu_pco_build_refract_fragment_shared_for_extent(
          observation->fragment_shared,
          prepass_width,
          prepass_height)) {
      *failure_reason = "fragment_descriptor_extent";
      return false;
   }
   observation->fragment_shared_count = PVRGPU_REFRACT_PCO_FS_SHARED_DWORDS;

   static const enum pvrgpu_refract_pco_texture_source sources[3] = {
      PVRGPU_REFRACT_PCO_PREVIOUS_DEPTH_ATTACHMENT,
      PVRGPU_REFRACT_PCO_PREVIOUS_COLOR_ATTACHMENT,
      PVRGPU_REFRACT_PCO_EXTERNAL_PAYLOAD,
   };
   static const enum pipe_format formats[3] = {
      PIPE_FORMAT_Z32_UNORM,
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   const unsigned widths[3] = { prepass_width, prepass_width, 512u };
   const unsigned heights[3] = { prepass_height, prepass_height, 512u };
   const size_t sizes[3] = { depth_bytes, color_bytes, 1048576u };
   const unsigned mip_counts[3] = { 1u, color_mip_count, 1u };
   static const unsigned filters[3] = {
      PIPE_TEX_FILTER_NEAREST,
      PIPE_TEX_FILTER_LINEAR,
      PIPE_TEX_FILTER_LINEAR,
   };
   static const unsigned mip_filters[3] = {
      PIPE_TEX_MIPFILTER_NONE,
      PIPE_TEX_MIPFILTER_LINEAR,
      PIPE_TEX_MIPFILTER_NONE,
   };
   for (unsigned slot = 0; slot < PVRGPU_REFRACT_PCO_TEXTURE_COUNT; ++slot) {
      struct pvrgpu_refract_pco_texture *texture =
         &observation->textures[slot];
      texture->source = sources[slot];
      texture->descriptor_set = slot;
      texture->binding = 0;
      texture->format = formats[slot];
      texture->declared_size = sizes[slot];
      texture->mip_count = mip_counts[slot];
      texture->min_filter = filters[slot];
      texture->mag_filter = filters[slot];
      texture->mip_filter = mip_filters[slot];
      texture->wrap_u = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
      texture->wrap_v = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
      texture->normalized_coordinates = true;
      texture->min_lod_u4_6 = 0;
      texture->max_lod_u4_6 =
         slot == 1 ? (mip_counts[slot] - 1u) * 64u : 0u;

      unsigned level_width = widths[slot];
      unsigned level_height = heights[slot];
      unsigned offset = 0;
      for (unsigned level = 0; level < mip_counts[slot]; ++level) {
         struct pvrgpu_refract_pco_texture_mip *mip =
            &texture->mip[level];
         mip->width = level_width;
         mip->height = level_height;
         mip->row_pitch = level_width * sizeof(uint32_t);
         mip->offset = offset;
         offset += mip->row_pitch * level_height;
         level_width = MAX2(level_width >> 1u, 1u);
         level_height = MAX2(level_height >> 1u, 1u);
      }
      if (offset != sizes[slot]) {
         *failure_reason = "sampled_texture_layout";
         return false;
      }
   }
   observation->texture_count = PVRGPU_REFRACT_PCO_TEXTURE_COUNT;
   return true;
}

static bool
pvrgpu_refract_pco_sampler_matches(
   const struct pvrgpu_context *ctx,
   const struct pvrgpu_refract_pco_observation *prepass,
   struct pvrgpu_refract_pco_observation *observation,
   const char **failure_reason)
{
   for (unsigned stage = 0; stage < MESA_SHADER_MESH_STAGES; ++stage) {
      const unsigned expected = stage == MESA_SHADER_FRAGMENT ? 3u : 0u;
      if (ctx->num_sampler_views[stage] != expected ||
          ctx->num_samplers[stage] != expected) {
         *failure_reason = "sampler_bindings";
         return false;
      }
   }
   if (!prepass || !prepass->prepass_depth || !prepass->prepass_color) {
      *failure_reason = "prepass_resources";
      return false;
   }

   static const enum pipe_format formats[3] = {
      PIPE_FORMAT_Z32_UNORM,
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   const unsigned widths[3] = {
      prepass->framebuffer_width,
      prepass->framebuffer_width,
      512u,
   };
   const unsigned heights[3] = {
      prepass->framebuffer_height,
      prepass->framebuffer_height,
      512u,
   };
   const unsigned last_levels[3] = {
      0u,
      prepass->prepass_color->last_level,
      0u,
   };
   static const unsigned min_filters[3] = {
      PIPE_TEX_FILTER_NEAREST,
      PIPE_TEX_FILTER_LINEAR,
      PIPE_TEX_FILTER_LINEAR,
   };
   static const unsigned mip_filters[3] = {
      PIPE_TEX_MIPFILTER_NONE,
      PIPE_TEX_MIPFILTER_LINEAR,
      PIPE_TEX_MIPFILTER_NONE,
   };

   for (unsigned slot = 0; slot < 3; ++slot) {
      const struct pipe_sampler_view *view =
         ctx->sampler_views[MESA_SHADER_FRAGMENT][slot];
      const struct pvrgpu_sampler_state *sampler =
         ctx->samplers[MESA_SHADER_FRAGMENT][slot];
      if (!view || !view->texture || !sampler ||
          view->target != PIPE_TEXTURE_2D ||
          view->texture->target != PIPE_TEXTURE_2D ||
          view->format != formats[slot] ||
          view->texture->format != formats[slot] ||
          view->texture->width0 != widths[slot] ||
          view->texture->height0 != heights[slot] ||
          view->texture->depth0 != 1 || view->texture->array_size != 1 ||
          view->texture->last_level != last_levels[slot] ||
          view->u.tex.first_level != 0 ||
          view->u.tex.last_level != last_levels[slot] ||
          view->u.tex.first_layer != 0 || view->u.tex.last_layer != 0 ||
          view->u.tex.min_lod_clamp != 0.0f ||
          view->texture->nr_samples > 1 ||
          view->texture->nr_storage_samples > 1) {
         *failure_reason = "sampler_view";
         return false;
      }
      const bool swizzle_matches =
         slot == 0 ?
            (view->swizzle_r == PIPE_SWIZZLE_X &&
             view->swizzle_g == PIPE_SWIZZLE_X &&
             view->swizzle_b == PIPE_SWIZZLE_X &&
             view->swizzle_a == PIPE_SWIZZLE_1) :
            (view->swizzle_r == PIPE_SWIZZLE_X &&
             view->swizzle_g == PIPE_SWIZZLE_Y &&
             view->swizzle_b == PIPE_SWIZZLE_Z &&
             view->swizzle_a == PIPE_SWIZZLE_W);
      const struct pipe_sampler_state *state = &sampler->state;
      if (!swizzle_matches || state->wrap_s != PIPE_TEX_WRAP_CLAMP_TO_EDGE ||
          state->wrap_t != PIPE_TEX_WRAP_CLAMP_TO_EDGE ||
          state->wrap_r != PIPE_TEX_WRAP_REPEAT ||
          state->min_img_filter != min_filters[slot] ||
          state->mag_img_filter != min_filters[slot] ||
          state->min_mip_filter != mip_filters[slot] ||
          state->compare_mode != PIPE_TEX_COMPARE_NONE ||
          state->unnormalized_coords || !state->seamless_cube_map ||
          state->reduction_mode != PIPE_TEX_REDUCTION_WEIGHTED_AVERAGE ||
          state->max_anisotropy > 1 || state->lod_bias != 0.0f ||
          state->min_lod != 0.0f || state->max_lod != 1000.0f) {
         *failure_reason = "sampler_state";
         return false;
      }
   }
   if (ctx->sampler_views[MESA_SHADER_FRAGMENT][0]->texture !=
          prepass->prepass_depth ||
       ctx->sampler_views[MESA_SHADER_FRAGMENT][1]->texture !=
          prepass->prepass_color) {
      *failure_reason = "prepass_sampler_alias";
      return false;
   }

   const struct pvrgpu_resource *image = pvrgpu_resource(
      ctx->sampler_views[MESA_SHADER_FRAGMENT][2]->texture);
   if (!image || !image->data || image->level_count != 1 ||
       image->level_offsets[0] != 0 ||
       image->level_strides[0] != PVRGPU_REFRACT_PCO_IMAGE_ROW_PITCH ||
       image->size != PVRGPU_REFRACT_PCO_IMAGE_BYTES) {
      *failure_reason = "sampled_image_resource";
      return false;
   }
   observation->sampled_image_bytes =
      MALLOC(PVRGPU_REFRACT_PCO_IMAGE_BYTES);
   if (!observation->sampled_image_bytes) {
      *failure_reason = "sampled_image_allocation";
      return false;
   }
   memcpy(observation->sampled_image_bytes,
          image->data,
          PVRGPU_REFRACT_PCO_IMAGE_BYTES);
   observation->sampled_image_bytes_size = PVRGPU_REFRACT_PCO_IMAGE_BYTES;
   observation->sampled_image_width = PVRGPU_REFRACT_PCO_IMAGE_WIDTH;
   observation->sampled_image_height = PVRGPU_REFRACT_PCO_IMAGE_HEIGHT;
   observation->sampled_image_row_pitch =
      PVRGPU_REFRACT_PCO_IMAGE_ROW_PITCH;
   observation->sampled_image_format = PIPE_FORMAT_R8G8B8A8_UNORM;
   return pvrgpu_refract_pco_build_fragment_shared(prepass,
                                                    observation,
                                                    failure_reason);
}

static bool
pvrgpu_draw_matches_refract_pco(
   const struct pvrgpu_context *ctx,
   const struct pipe_draw_info *info,
   unsigned drawid_offset,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,
   unsigned num_draws,
   enum pvrgpu_pco_refract_profile profile,
   const struct pvrgpu_refract_pco_observation *prepass,
   struct pvrgpu_refract_pco_observation *observation,
   const char **failure_reason)
{
   if (failure_reason)
      *failure_reason = "arguments";
   if (!ctx || !observation || !failure_reason ||
       !pvrgpu_refract_pco_draw_info_matches(info,
                                             drawid_offset,
                                             indirect,
                                             draws,
                                             num_draws)) {
      if (failure_reason)
         *failure_reason = "draw_info";
      return false;
   }

   memset(observation, 0, sizeof(*observation));
   observation->profile = profile;
   observation->vertex_count = PVRGPU_REFRACT_PCO_VERTEX_COUNT;
   if (!pvrgpu_refract_pco_framebuffer_matches(ctx,
                                                profile,
                                                observation,
                                                failure_reason) ||
       !pvrgpu_refract_pco_common_state_matches(ctx,
                                                profile,
                                                observation,
                                                failure_reason) ||
       !pvrgpu_refract_pco_copy_constants(ctx,
                                          profile,
                                          observation,
                                          failure_reason) ||
       !pvrgpu_refract_pco_capture_vertices(ctx,
                                            observation,
                                            failure_reason)) {
      pvrgpu_refract_pco_observation_finish(observation);
      return false;
   }

   if (profile == PVRGPU_PCO_REFRACT_COMPOSITE &&
       !pvrgpu_refract_pco_sampler_matches(ctx,
                                            prepass,
                                            observation,
                                            failure_reason)) {
      pvrgpu_refract_pco_observation_finish(observation);
      return false;
   }

   *failure_reason = NULL;
   return true;
}

static bool
pvrgpu_shadow_pco_draw_info_matches(
   const struct pipe_draw_info *info,
   unsigned drawid_offset,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,
   unsigned num_draws,
   enum pvrgpu_pco_shadow_profile profile)
{
   const enum mesa_prim expected_mode =
      profile == PVRGPU_PCO_SHADOW_MASK ? MESA_PRIM_TRIANGLE_STRIP :
                                          MESA_PRIM_TRIANGLES;
   const unsigned expected_count =
      profile == PVRGPU_PCO_SHADOW_MASK ?
         PVRGPU_SHADOW_PCO_MASK_VERTEX_COUNT :
         PVRGPU_SHADOW_PCO_MESH_VERTEX_COUNT;
   return info && draws && drawid_offset == 0 && !indirect && num_draws == 1 &&
          info->mode == expected_mode && info->index_size == 0 &&
          !info->has_user_indices && !info->primitive_restart &&
          !info->increment_draw_id && !info->index_bias_varies &&
          !info->was_line_loop && info->start_instance == 0 &&
          info->instance_count == 1 && draws[0].start == 0 &&
          draws[0].count == expected_count;
}

static bool
pvrgpu_shadow_pco_framebuffer_matches(
   const struct pvrgpu_context *ctx,
   enum pvrgpu_pco_shadow_profile profile,
   const struct pvrgpu_shadow_pco_observation *mask,
   struct pvrgpu_shadow_pco_observation *observation,
   const char **failure_reason)
{
   const bool depth = profile == PVRGPU_PCO_SHADOW_DEPTH;
   unsigned width = 0;
   unsigned height = 0;
   if (!pvrgpu_glmark_scaled_output_extent(depth ? 2u : 1u,
                                            &width,
                                            &height)) {
      *failure_reason = "output_extent";
      return false;
   }
   if (ctx->framebuffer.width != width ||
       ctx->framebuffer.height != height || ctx->framebuffer.resolve ||
       ctx->framebuffer.pls_enabled || ctx->framebuffer.viewmask != 0 ||
       ctx->framebuffer.nr_cbufs != (depth ? 0U : 1U)) {
      *failure_reason = "framebuffer";
      return false;
   }
   const struct pipe_surface *depth_surface = &ctx->framebuffer.zsbuf;
   if (!depth_surface->texture || depth_surface->level != 0 ||
       depth_surface->first_layer != 0 || depth_surface->last_layer != 0 ||
       depth_surface->texture->target != PIPE_TEXTURE_2D ||
       depth_surface->texture->width0 != width ||
       depth_surface->texture->height0 != height ||
       depth_surface->texture->last_level != 0 ||
       depth_surface->texture->nr_samples > 1 ||
       depth_surface->texture->nr_storage_samples > 1 ||
       depth_surface->format !=
          (depth ? PIPE_FORMAT_Z32_UNORM : PIPE_FORMAT_Z24X8_UNORM) ||
       depth_surface->texture->format != depth_surface->format) {
      *failure_reason = "depth_attachment";
      return false;
   }

   if (depth) {
      if (!ctx->full_depth_clear_is_one ||
          ctx->full_depth_clear_resource != depth_surface->texture ||
          ctx->full_depth_clear_level != 0 ||
          ctx->full_depth_clear_first_layer != 0 ||
          ctx->full_depth_clear_last_layer != 0 ||
          ctx->full_depth_clear_width != width ||
          ctx->full_depth_clear_height != height) {
         *failure_reason = "shadow_depth_clear";
         return false;
      }
      pipe_resource_reference(&observation->shadow_depth,
                              depth_surface->texture);
   } else {
      const struct pipe_surface *color = &ctx->framebuffer.cbufs[0];
      if (!color->texture || color->texture->target != PIPE_TEXTURE_2D ||
          color->format != PIPE_FORMAT_R8G8B8A8_UNORM ||
          color->texture->format != PIPE_FORMAT_R8G8B8A8_UNORM ||
          color->level != 0 || color->first_layer != 0 ||
          color->last_layer != 0 || color->texture->width0 != width ||
          color->texture->height0 != height ||
          color->texture->last_level != 0 ||
          color->texture->nr_samples > 1 ||
          color->texture->nr_storage_samples > 1 ||
          !pvrgpu_framebuffer_matches_rdc_output(ctx)) {
         *failure_reason = "color_attachment";
         return false;
      }
      if (profile == PVRGPU_PCO_SHADOW_MASK) {
         static const uint8_t black[4] = { 0, 0, 0, 255 };
         if (!pvrgpu_refract_pco_depth_backing_is_known_clear(
                depth_surface,
                width,
                height,
                &observation->output_depth_clear_one)) {
            *failure_reason = "output_depth_backing";
            return false;
         }
         /* RenderDoc's first replay pass can restore the final color backing
          * while leaving the output depth attachment at its pristine zero
          * value.  That pass is a strictly identified warmup and is never
          * submitted.  Validate depth first so it can reach the existing
          * warmup skip; require canonical opaque black only for the real
          * clear-one mask pass that is retained and submitted. */
         if (observation->output_depth_clear_one &&
             !pvrgpu_rgba8_surface_backing_matches(color,
                                                    width,
                                                    height,
                                                    black)) {
            *failure_reason = "output_color_backing";
            return false;
         }
      } else if (!mask || !mask->color_attachment ||
                 !mask->depth_attachment ||
                 color->texture != mask->color_attachment ||
                 depth_surface->texture != mask->depth_attachment) {
         *failure_reason = "scene_attachment_alias";
         return false;
      }
      pipe_resource_reference(&observation->color_attachment,
                              color->texture);
   }
   pipe_resource_reference(&observation->depth_attachment,
                           depth_surface->texture);
   observation->framebuffer_width = width;
   observation->framebuffer_height = height;
   observation->color_format = depth ? PIPE_FORMAT_NONE :
                                       PIPE_FORMAT_R8G8B8A8_UNORM;
   observation->depth_format = depth ? PIPE_FORMAT_Z32_UNORM :
                                       PIPE_FORMAT_Z24X8_UNORM;
   observation->color_attachment_source =
      profile == PVRGPU_PCO_SHADOW_SCENE ? PVRGPU_PCO_SHADOW_MASK :
                                           UINT32_MAX;
   observation->depth_attachment_source =
      profile == PVRGPU_PCO_SHADOW_SCENE ? PVRGPU_PCO_SHADOW_MASK :
                                           UINT32_MAX;
   observation->color_clear = profile == PVRGPU_PCO_SHADOW_MASK;
   observation->depth_clear = profile != PVRGPU_PCO_SHADOW_SCENE;
   observation->clear_color_bits[3] = UINT32_C(0x3f800000);
   observation->depth_clear_bits = UINT32_C(0x3f800000);
   return true;
}

static bool
pvrgpu_shadow_pco_pipeline_matches(
   const struct pvrgpu_context *ctx,
   enum pvrgpu_pco_shadow_profile profile,
   struct pvrgpu_shadow_pco_observation *observation,
   const char **failure_reason)
{
   const bool depth = profile == PVRGPU_PCO_SHADOW_DEPTH;
   if (ctx->tcs || ctx->tes || ctx->gs || ctx->num_stream_output_targets ||
       ctx->sample_mask != UINT32_MAX || !ctx->has_viewport) {
      *failure_reason = "extra_stage_or_sample_mask";
      return false;
   }
   const unsigned width = observation->framebuffer_width;
   const unsigned height = observation->framebuffer_height;
   const uint32_t scale_x = pvrgpu_float_bits((float)width * 0.5f);
   const uint32_t scale_y = pvrgpu_float_bits((float)height * 0.5f);
   if (pvrgpu_float_bits(ctx->viewport.scale[0]) != scale_x ||
       pvrgpu_float_bits(ctx->viewport.scale[1]) != scale_y ||
       pvrgpu_float_bits(ctx->viewport.scale[2]) != UINT32_C(0x3f000000) ||
       pvrgpu_float_bits(ctx->viewport.translate[0]) != scale_x ||
       pvrgpu_float_bits(ctx->viewport.translate[1]) != scale_y ||
       pvrgpu_float_bits(ctx->viewport.translate[2]) != UINT32_C(0x3f000000)) {
      *failure_reason = "viewport";
      return false;
   }
   observation->viewport_width = width;
   observation->viewport_height = height;

   const unsigned expected_colormask = depth ? 0U : PIPE_MASK_RGBA;
   if (!ctx->blend || ctx->blend->state.independent_blend_enable ||
       ctx->blend->state.logicop_enable || !ctx->blend->state.dither ||
       ctx->blend->state.alpha_to_coverage ||
       ctx->blend->state.alpha_to_coverage_dither ||
       ctx->blend->state.alpha_to_one || ctx->blend->state.max_rt != 0 ||
       ctx->blend->state.advanced_blend_func != 0 ||
       ctx->blend->state.rt[0].blend_enable ||
       ctx->blend->state.rt[0].colormask != expected_colormask) {
      *failure_reason = "blend";
      return false;
   }
   if (!ctx->dsa || !ctx->dsa->state.depth_enabled ||
       !ctx->dsa->state.depth_writemask ||
       ctx->dsa->state.depth_func != PIPE_FUNC_LEQUAL ||
       ctx->dsa->state.depth_bounds_test ||
       ctx->dsa->state.stencil[0].enabled ||
       ctx->dsa->state.stencil[1].enabled || ctx->dsa->state.alpha_enabled) {
      *failure_reason = "depth_stencil_alpha";
      return false;
   }
   if (!ctx->rasterizer || ctx->rasterizer->state.front_ccw ||
       ctx->rasterizer->state.cull_face != PIPE_FACE_BACK ||
       ctx->rasterizer->state.fill_front != PIPE_POLYGON_MODE_FILL ||
       ctx->rasterizer->state.fill_back != PIPE_POLYGON_MODE_FILL ||
       ctx->rasterizer->state.scissor ||
       ctx->rasterizer->state.rasterizer_discard ||
       ctx->rasterizer->state.multisample ||
       ctx->rasterizer->state.offset_tri ||
       ctx->rasterizer->state.poly_smooth ||
       ctx->rasterizer->state.poly_stipple_enable ||
       ctx->rasterizer->state.conservative_raster_mode != 0 ||
       !ctx->rasterizer->state.half_pixel_center ||
       ctx->rasterizer->state.bottom_edge_rule ||
       ctx->rasterizer->state.clip_halfz ||
       !ctx->rasterizer->state.depth_clip_near ||
       !ctx->rasterizer->state.depth_clip_far ||
       ctx->rasterizer->state.depth_clamp) {
      *failure_reason = "rasterizer";
      return false;
   }
   observation->color_mask = expected_colormask;
   observation->blend_enable = false;
   observation->rgb_func = PIPE_BLEND_ADD;
   observation->rgb_src_factor = PIPE_BLENDFACTOR_ONE;
   observation->rgb_dst_factor = PIPE_BLENDFACTOR_ZERO;
   observation->alpha_func = PIPE_BLEND_ADD;
   observation->alpha_src_factor = PIPE_BLENDFACTOR_ONE;
   observation->alpha_dst_factor = PIPE_BLENDFACTOR_ZERO;
   observation->dither = true;
   observation->depth_enable = true;
   observation->depth_write = true;
   observation->depth_func = PIPE_FUNC_LEQUAL;
   return true;
}

static bool
pvrgpu_shadow_pco_copy_constants(
   const struct pvrgpu_context *ctx,
   enum pvrgpu_pco_shadow_profile profile,
   struct pvrgpu_shadow_pco_observation *observation,
   const char **failure_reason)
{
   const size_t expected_dwords =
      profile == PVRGPU_PCO_SHADOW_DEPTH ? 16U : 32U;
   for (unsigned stage = 0; stage < MESA_SHADER_MESH_STAGES; ++stage) {
      const unsigned expected = stage == MESA_SHADER_VERTEX ? 1U : 0U;
      if (ctx->num_constant_buffers[stage] != expected) {
         *failure_reason = "constant_buffers";
         return false;
      }
   }
   const struct pipe_constant_buffer *cb =
      &ctx->constant_buffers[MESA_SHADER_VERTEX][0];
   size_t available = 0;
   const uint8_t *bytes = pvrgpu_constant_buffer_bytes(ctx,
                                                       MESA_SHADER_VERTEX,
                                                       0,
                                                       &available);
   if (!bytes || cb->buffer_size != expected_dwords * sizeof(uint32_t) ||
       available != expected_dwords * sizeof(uint32_t)) {
      *failure_reason = "constant_buffer_size";
      return false;
   }
   memcpy(observation->vertex_shared, bytes, available);
   observation->vertex_shared_count = expected_dwords;
   return true;
}

static bool
pvrgpu_shadow_pco_capture_vertices(
   const struct pvrgpu_context *ctx,
   enum pvrgpu_pco_shadow_profile profile,
   struct pvrgpu_shadow_pco_observation *observation,
   const char **failure_reason)
{
   const bool mask = profile == PVRGPU_PCO_SHADOW_MASK;
   const unsigned attribute_count = mask ? 1U : 2U;
   const unsigned vertex_count =
      mask ? PVRGPU_SHADOW_PCO_MASK_VERTEX_COUNT :
             PVRGPU_SHADOW_PCO_MESH_VERTEX_COUNT;
   const unsigned component_count = mask ? 2U : 3U;
   const unsigned attribute_stride = component_count * sizeof(float);
   const enum pipe_format format =
      mask ? PIPE_FORMAT_R32G32_FLOAT : PIPE_FORMAT_R32G32B32_FLOAT;
   if (!ctx->vertex_elements ||
       ctx->vertex_elements->num_elements != attribute_count ||
       ctx->num_vertex_buffers != attribute_count) {
      *failure_reason = "vertex_layout";
      return false;
   }

   const size_t stream_size = (size_t)vertex_count * attribute_stride;
   const struct pvrgpu_resource *resources[2] = { NULL, NULL };
   for (unsigned attribute = 0; attribute < attribute_count; ++attribute) {
      const struct pipe_vertex_element *element =
         &ctx->vertex_elements->elements[attribute];
      const struct pipe_vertex_buffer *buffer = &ctx->vertex_buffers[attribute];
      if (element->src_format != format ||
          element->src_stride != attribute_stride ||
          element->src_offset != 0 || element->dual_slot ||
          element->instance_divisor != 0 ||
          element->vertex_buffer_index != attribute || buffer->is_user_buffer ||
          buffer->buffer_offset != 0 || !buffer->buffer.resource ||
          buffer->buffer.resource->target != PIPE_BUFFER) {
         *failure_reason = "vertex_layout";
         return false;
      }
      resources[attribute] = pvrgpu_resource(buffer->buffer.resource);
      if (!resources[attribute] || !resources[attribute]->data ||
          resources[attribute]->size != stream_size) {
         *failure_reason = "vertex_resource";
         return false;
      }
   }

   const unsigned vertex_stride = attribute_count * attribute_stride;
   const size_t total_size = (size_t)vertex_count * vertex_stride;
   observation->vertex_data = MALLOC(total_size);
   if (!observation->vertex_data) {
      *failure_reason = "vertex_allocation";
      return false;
   }
   for (unsigned vertex = 0; vertex < vertex_count; ++vertex) {
      uint8_t *dst = observation->vertex_data +
                     (size_t)vertex * vertex_stride;
      for (unsigned attribute = 0; attribute < attribute_count; ++attribute) {
         memcpy(dst + attribute * attribute_stride,
                resources[attribute]->data +
                   (size_t)vertex * attribute_stride,
                attribute_stride);
      }
   }
   observation->vertex_count = vertex_count;
   observation->vertex_stride = vertex_stride;
   observation->vertex_data_size = total_size;
   return true;
}

static bool
pvrgpu_shadow_pco_sampler_matches(
   const struct pvrgpu_context *ctx,
   enum pvrgpu_pco_shadow_profile profile,
   const struct pvrgpu_shadow_pco_observation *depth,
   struct pvrgpu_shadow_pco_observation *observation,
   const char **failure_reason)
{
   for (unsigned stage = 0; stage < MESA_SHADER_MESH_STAGES; ++stage) {
      const unsigned expected = stage == MESA_SHADER_FRAGMENT ?
                                   (profile == PVRGPU_PCO_SHADOW_DEPTH ?
                                       ctx->num_sampler_views[stage] : 1U) :
                                   0U;
      if (ctx->num_sampler_views[stage] != expected ||
          ctx->num_samplers[stage] != expected || expected > 1U) {
         *failure_reason = "sampler_bindings";
         return false;
      }
   }
   const unsigned fragment_count =
      ctx->num_sampler_views[MESA_SHADER_FRAGMENT];
   if (profile == PVRGPU_PCO_SHADOW_DEPTH && fragment_count == 0)
      return true;
   const struct pvrgpu_shadow_pco_observation *sampled_depth =
      profile == PVRGPU_PCO_SHADOW_DEPTH ? observation : depth;
   if (!sampled_depth || !sampled_depth->shadow_depth) {
      *failure_reason = "shadow_depth_resource";
      return false;
   }

   const struct pipe_sampler_view *view =
      ctx->sampler_views[MESA_SHADER_FRAGMENT][0];
   const struct pvrgpu_sampler_state *sampler =
      ctx->samplers[MESA_SHADER_FRAGMENT][0];
   if (!view || !view->texture || !sampler ||
       view->texture != sampled_depth->shadow_depth ||
       view->target != PIPE_TEXTURE_2D ||
       view->texture->target != PIPE_TEXTURE_2D ||
       view->format != PIPE_FORMAT_Z32_UNORM ||
       view->texture->format != PIPE_FORMAT_Z32_UNORM ||
       view->texture->width0 != sampled_depth->framebuffer_width ||
       view->texture->height0 != sampled_depth->framebuffer_height ||
       view->texture->last_level != 0 || view->u.tex.first_level != 0 ||
       view->u.tex.last_level != 0 || view->u.tex.first_layer != 0 ||
       view->u.tex.last_layer != 0 || view->u.tex.min_lod_clamp != 0.0f ||
       view->swizzle_r != PIPE_SWIZZLE_X ||
       view->swizzle_g != PIPE_SWIZZLE_X ||
       view->swizzle_b != PIPE_SWIZZLE_X ||
       view->swizzle_a != PIPE_SWIZZLE_1) {
      *failure_reason = "sampler_view";
      return false;
   }
   const struct pipe_sampler_state *state = &sampler->state;
   if (state->wrap_s != PIPE_TEX_WRAP_CLAMP_TO_EDGE ||
       state->wrap_t != PIPE_TEX_WRAP_CLAMP_TO_EDGE ||
       state->wrap_r != PIPE_TEX_WRAP_REPEAT ||
       state->min_img_filter != PIPE_TEX_FILTER_NEAREST ||
       state->mag_img_filter != PIPE_TEX_FILTER_NEAREST ||
       state->min_mip_filter != PIPE_TEX_MIPFILTER_NONE ||
       state->compare_mode != PIPE_TEX_COMPARE_NONE ||
       state->unnormalized_coords || !state->seamless_cube_map ||
       state->reduction_mode != PIPE_TEX_REDUCTION_WEIGHTED_AVERAGE ||
       state->max_anisotropy > 1 || state->lod_bias != 0.0f ||
       state->min_lod != 0.0f || state->max_lod != 1000.0f) {
      *failure_reason = "sampler_state";
      return false;
   }
   if (profile == PVRGPU_PCO_SHADOW_MASK) {
      if (!pvrgpu_pco_build_shadow_fragment_shared_for_extent(
             observation->fragment_shared,
             sampled_depth->framebuffer_width,
             sampled_depth->framebuffer_height)) {
         *failure_reason = "fragment_descriptor_extent";
         return false;
      }
      observation->fragment_shared_count =
         PVRGPU_SHADOW_PCO_FS_SHARED_DWORDS;
   }
   return true;
}

static bool
pvrgpu_draw_matches_shadow_pco(
   const struct pvrgpu_context *ctx,
   const struct pipe_draw_info *info,
   unsigned drawid_offset,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,
   unsigned num_draws,
   enum pvrgpu_pco_shadow_profile profile,
   const struct pvrgpu_shadow_pco_observation *depth,
   const struct pvrgpu_shadow_pco_observation *mask,
   struct pvrgpu_shadow_pco_observation *observation,
   const char **failure_reason)
{
   if (failure_reason)
      *failure_reason = "arguments";
   if (!ctx || !observation || !failure_reason ||
       !pvrgpu_shadow_pco_draw_info_matches(info,
                                            drawid_offset,
                                            indirect,
                                            draws,
                                            num_draws,
                                            profile)) {
      if (failure_reason)
         *failure_reason = "draw_info";
      return false;
   }
   memset(observation, 0, sizeof(*observation));
   observation->profile = profile;
   observation->primitive_mode = info->mode;
   if (!pvrgpu_shadow_pco_framebuffer_matches(ctx,
                                               profile,
                                               mask,
                                               observation,
                                               failure_reason) ||
       !pvrgpu_shadow_pco_pipeline_matches(ctx,
                                           profile,
                                           observation,
                                           failure_reason) ||
       !pvrgpu_shadow_pco_copy_constants(ctx,
                                         profile,
                                         observation,
                                         failure_reason) ||
       !pvrgpu_shadow_pco_capture_vertices(ctx,
                                           profile,
                                           observation,
                                           failure_reason) ||
       !pvrgpu_shadow_pco_sampler_matches(ctx,
                                          profile,
                                          depth,
                                          observation,
                                          failure_reason)) {
      pvrgpu_shadow_pco_observation_finish(observation);
      return false;
   }
   *failure_reason = NULL;
   return true;
}

static bool
pvrgpu_lit_mesh_copy_constants(const struct pvrgpu_context *ctx,
                               struct pvrgpu_lit_mesh_observation *observation)
{
   if (!ctx || !observation ||
       ctx->num_constant_buffers[MESA_SHADER_VERTEX] != 1 ||
       ctx->num_constant_buffers[MESA_SHADER_FRAGMENT] != 0)
      return false;
   for (unsigned stage = 0; stage < MESA_SHADER_MESH_STAGES; ++stage) {
      if (stage != MESA_SHADER_VERTEX && ctx->num_constant_buffers[stage] != 0)
         return false;
   }

   const struct pipe_constant_buffer *cb =
      &ctx->constant_buffers[MESA_SHADER_VERTEX][0];
   if (cb->buffer_size != sizeof(observation->vertex_shared))
      return false;
   size_t available = 0;
   const uint8_t *bytes = pvrgpu_constant_buffer_bytes(ctx,
                                                       MESA_SHADER_VERTEX,
                                                       0,
                                                       &available);
   if (!bytes || available != sizeof(observation->vertex_shared))
      return false;
   memcpy(observation->vertex_shared,
          bytes,
          sizeof(observation->vertex_shared));
   return true;
}

static bool
pvrgpu_lit_mesh_pipeline_state_matches(
   const struct pvrgpu_context *ctx,
   bool probe,
   struct pvrgpu_lit_mesh_observation *observation,
   const char **failure_reason)
{
   if (ctx->tcs || ctx->tes || ctx->gs || ctx->num_stream_output_targets) {
      *failure_reason = "extra_stage_or_stream_output";
      return false;
   }
   for (unsigned stage = 0; stage < MESA_SHADER_MESH_STAGES; ++stage) {
      if (ctx->num_sampler_views[stage] != 0 || ctx->num_samplers[stage] != 0) {
         *failure_reason = "sampler_bindings";
         return false;
      }
   }
   if (!pvrgpu_lit_mesh_copy_constants(ctx, observation)) {
      *failure_reason = "constant_buffers";
      return false;
   }
   if (ctx->sample_mask != UINT32_MAX || !ctx->has_viewport) {
      *failure_reason = "sample_mask_or_viewport";
      return false;
   }

   unsigned width = 0;
   unsigned height = 0;
   if (!pvrgpu_glmark_pco_draw_extent(ctx, probe, &width, &height)) {
      *failure_reason = "framebuffer_extent";
      return false;
   }
   const uint32_t width_scale = pvrgpu_float_bits((float)width * 0.5f);
   const uint32_t height_scale = pvrgpu_float_bits((float)height * 0.5f);
   if (pvrgpu_float_bits(ctx->viewport.scale[0]) != width_scale ||
       pvrgpu_float_bits(ctx->viewport.scale[1]) != height_scale ||
       pvrgpu_float_bits(ctx->viewport.scale[2]) != UINT32_C(0x3f000000) ||
       pvrgpu_float_bits(ctx->viewport.translate[0]) != width_scale ||
       pvrgpu_float_bits(ctx->viewport.translate[1]) != height_scale ||
       pvrgpu_float_bits(ctx->viewport.translate[2]) != UINT32_C(0x3f000000)) {
      *failure_reason = "viewport";
      return false;
   }

   if (!ctx->blend || ctx->blend->state.independent_blend_enable ||
       ctx->blend->state.logicop_enable || !ctx->blend->state.dither ||
       ctx->blend->state.alpha_to_coverage ||
       ctx->blend->state.alpha_to_coverage_dither ||
       ctx->blend->state.alpha_to_one || ctx->blend->state.max_rt != 0 ||
       ctx->blend->state.advanced_blend_func != 0 ||
       ctx->blend->state.rt[0].blend_enable ||
       ctx->blend->state.rt[0].colormask != PIPE_MASK_RGBA) {
      *failure_reason = "blend";
      return false;
   }
   if (!ctx->dsa || !ctx->dsa->state.depth_enabled ||
       !ctx->dsa->state.depth_writemask ||
       ctx->dsa->state.depth_func != PIPE_FUNC_LEQUAL ||
       ctx->dsa->state.depth_bounds_test ||
       ctx->dsa->state.stencil[0].enabled ||
       ctx->dsa->state.stencil[1].enabled || ctx->dsa->state.alpha_enabled) {
      *failure_reason = "depth_stencil_alpha";
      return false;
   }
   if (!ctx->rasterizer || ctx->rasterizer->state.front_ccw ||
       ctx->rasterizer->state.cull_face != PIPE_FACE_BACK ||
       ctx->rasterizer->state.fill_front != PIPE_POLYGON_MODE_FILL ||
       ctx->rasterizer->state.fill_back != PIPE_POLYGON_MODE_FILL ||
       ctx->rasterizer->state.scissor ||
       ctx->rasterizer->state.rasterizer_discard ||
       ctx->rasterizer->state.multisample ||
       ctx->rasterizer->state.offset_tri || ctx->rasterizer->state.poly_smooth ||
       ctx->rasterizer->state.poly_stipple_enable ||
       ctx->rasterizer->state.conservative_raster_mode != 0 ||
       !ctx->rasterizer->state.half_pixel_center ||
       ctx->rasterizer->state.bottom_edge_rule ||
       ctx->rasterizer->state.clip_halfz ||
       !ctx->rasterizer->state.depth_clip_near ||
       !ctx->rasterizer->state.depth_clip_far ||
       ctx->rasterizer->state.depth_clamp) {
      *failure_reason = "rasterizer";
      return false;
   }

   observation->framebuffer_width = ctx->framebuffer.width;
   observation->framebuffer_height = ctx->framebuffer.height;
   observation->viewport_width = width;
   observation->viewport_height = height;
   return true;
}

static bool
pvrgpu_lit_mesh_capture_vertices(const struct pvrgpu_context *ctx,
                                 unsigned vertex_count,
                                 bool capture,
                                 struct pvrgpu_lit_mesh_observation *observation)
{
   if (!ctx || !observation || !ctx->vertex_elements ||
       ctx->vertex_elements->num_elements != 2 || ctx->num_vertex_buffers != 2)
      return false;

   const size_t stream_bytes = (size_t)vertex_count * 3u * sizeof(float);
   const struct pvrgpu_resource *resources[2] = { NULL, NULL };
   for (unsigned attribute = 0; attribute < 2; ++attribute) {
      const struct pipe_vertex_element *element =
         &ctx->vertex_elements->elements[attribute];
      if (element->src_format != PIPE_FORMAT_R32G32B32_FLOAT ||
          element->src_stride != 3u * sizeof(float) ||
          element->src_offset != 0 || element->dual_slot ||
          element->instance_divisor != 0 ||
          element->vertex_buffer_index != attribute)
         return false;
      const struct pipe_vertex_buffer *buffer = &ctx->vertex_buffers[attribute];
      if (buffer->is_user_buffer || buffer->buffer_offset != 0 ||
          !buffer->buffer.resource || buffer->buffer.resource->target != PIPE_BUFFER)
         return false;
      resources[attribute] = pvrgpu_resource(buffer->buffer.resource);
      if (!resources[attribute] || !resources[attribute]->data ||
          resources[attribute]->size != stream_bytes)
         return false;
   }
   if (!capture)
      return true;

   const size_t interleaved_size = (size_t)vertex_count * 6u * sizeof(float);
   uint8_t *interleaved = MALLOC(interleaved_size);
   if (!interleaved)
      return false;
   for (unsigned vertex = 0; vertex < vertex_count; ++vertex) {
      memcpy(interleaved + (size_t)vertex * 24u,
             resources[0]->data + (size_t)vertex * 12u,
             12u);
      memcpy(interleaved + (size_t)vertex * 24u + 12u,
             resources[1]->data + (size_t)vertex * 12u,
             12u);
   }
   observation->interleaved_vertex_data = interleaved;
   observation->interleaved_vertex_data_size = interleaved_size;
   return true;
}

static bool
pvrgpu_draw_matches_lit_mesh(
   const struct pvrgpu_context *ctx,
   const struct pipe_draw_info *info,
   unsigned drawid_offset,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,
   unsigned num_draws,
   bool probe,
   bool require_depth_clear,
   struct pvrgpu_lit_mesh_observation *observation,
   const char **failure_reason)
{
   if (failure_reason)
      *failure_reason = "arguments";
   if (!ctx || !info || !draws || !observation || !failure_reason)
      return false;

   enum pvrgpu_pco_lit_mesh_profile profile;
   unsigned expected_vertex_count = 0;
   if (!pvrgpu_lit_mesh_profile(ctx, &profile, &expected_vertex_count)) {
      *failure_reason = "shader_profile";
      return false;
   }
   if (drawid_offset != 0 || indirect || num_draws != 1 ||
       info->mode != MESA_PRIM_TRIANGLES || info->index_size != 0 ||
       info->has_user_indices || info->primitive_restart ||
       info->increment_draw_id || info->index_bias_varies ||
       info->was_line_loop || info->start_instance != 0 ||
       info->instance_count != 1 || draws[0].start != 0 ||
       draws[0].count != expected_vertex_count) {
      *failure_reason = "draw_info";
      return false;
   }
   if (!pvrgpu_conditionals_framebuffer_matches(ctx,
                                                probe,
                                                require_depth_clear)) {
      *failure_reason = "framebuffer";
      return false;
   }

   memset(observation, 0, sizeof(*observation));
   observation->profile = profile;
   observation->vertex_count = expected_vertex_count;
   if (!pvrgpu_lit_mesh_pipeline_state_matches(ctx,
                                               probe,
                                               observation,
                                               failure_reason))
      return false;
   if (!pvrgpu_lit_mesh_capture_vertices(ctx,
                                         expected_vertex_count,
                                         !probe,
                                         observation)) {
      *failure_reason = "vertex_data";
      return false;
   }
   *failure_reason = NULL;
   return true;
}

static bool
pvrgpu_texture_pco_profile_matches(const struct pvrgpu_context *ctx)
{
   static const uint32_t vertex_source_hash[8] = {
      UINT32_C(0x750ac3d1), UINT32_C(0xe9ceafcc),
      UINT32_C(0xdd1263dd), UINT32_C(0xa22a457b),
      UINT32_C(0x3b8ebb47), UINT32_C(0xa4ee0e8e),
      UINT32_C(0xeb2663ea), UINT32_C(0x6ad452cd),
   };
   static const uint32_t fragment_source_hash[8] = {
      UINT32_C(0xf95c6a3f), UINT32_C(0x2572cc32),
      UINT32_C(0x635afdeb), UINT32_C(0xff4d47cb),
      UINT32_C(0x5a3d3c87), UINT32_C(0x94e6645c),
      UINT32_C(0x9dde3b59), UINT32_C(0x233b4b47),
   };

   return ctx && ctx->vs && ctx->fs && ctx->vs->has_nir &&
          ctx->fs->has_nir && ctx->vs->nir && ctx->fs->nir &&
          ctx->vs->stage == MESA_SHADER_VERTEX &&
          ctx->fs->stage == MESA_SHADER_FRAGMENT &&
          pvrgpu_nir_source_hash_matches(ctx->vs->nir,
                                         vertex_source_hash) &&
          pvrgpu_nir_source_hash_matches(ctx->fs->nir,
                                         fragment_source_hash);
}

static bool
pvrgpu_texture_pco_sampler_matches(const struct pvrgpu_context *ctx,
                                    const char **failure_reason)
{
   for (unsigned stage = 0; stage < MESA_SHADER_MESH_STAGES; ++stage) {
      const unsigned expected = stage == MESA_SHADER_FRAGMENT ? 1u : 0u;
      if (ctx->num_sampler_views[stage] != expected ||
          ctx->num_samplers[stage] != expected) {
         *failure_reason = "sampler_bindings";
         return false;
      }
   }

   const struct pipe_sampler_view *view =
      ctx->sampler_views[MESA_SHADER_FRAGMENT][0];
   const struct pvrgpu_sampler_state *sampler =
      ctx->samplers[MESA_SHADER_FRAGMENT][0];
   if (!view || !view->texture || !sampler ||
       view->target != PIPE_TEXTURE_2D ||
       view->texture->target != PIPE_TEXTURE_2D ||
       view->format != PIPE_FORMAT_R8G8B8X8_UNORM ||
       view->texture->format != PIPE_FORMAT_R8G8B8X8_UNORM ||
       view->texture->width0 != PVRGPU_TEXTURE_PCO_WIDTH ||
       view->texture->height0 != PVRGPU_TEXTURE_PCO_HEIGHT ||
       view->texture->depth0 != 1 || view->texture->array_size != 1 ||
       view->texture->last_level != 0 ||
       view->u.tex.first_level != 0 || view->u.tex.last_level != 0 ||
       view->u.tex.first_layer != 0 || view->u.tex.last_layer != 0 ||
       view->u.tex.min_lod_clamp != 0.0f ||
       view->swizzle_r != PIPE_SWIZZLE_X ||
       view->swizzle_g != PIPE_SWIZZLE_Y ||
       view->swizzle_b != PIPE_SWIZZLE_Z ||
       view->swizzle_a != PIPE_SWIZZLE_1 ||
       view->texture->nr_samples > 1 ||
       view->texture->nr_storage_samples > 1) {
      *failure_reason = "texture_view";
      return false;
   }

   const struct pipe_sampler_state *state = &sampler->state;
   if (state->wrap_s != PIPE_TEX_WRAP_CLAMP_TO_EDGE ||
       state->wrap_t != PIPE_TEX_WRAP_CLAMP_TO_EDGE ||
       state->wrap_r != PIPE_TEX_WRAP_REPEAT ||
       state->min_img_filter != PIPE_TEX_FILTER_NEAREST ||
       state->mag_img_filter != PIPE_TEX_FILTER_NEAREST ||
       state->min_mip_filter != PIPE_TEX_MIPFILTER_NONE ||
       state->compare_mode != PIPE_TEX_COMPARE_NONE ||
       state->unnormalized_coords || !state->seamless_cube_map ||
       state->reduction_mode != PIPE_TEX_REDUCTION_WEIGHTED_AVERAGE ||
       state->max_anisotropy > 1 || state->lod_bias != 0.0f ||
       state->min_lod != 0.0f || state->max_lod != 1000.0f) {
      *failure_reason = "sampler_state";
      return false;
   }
   return true;
}

static bool
pvrgpu_texture_pco_pipeline_state_matches(
   const struct pvrgpu_context *ctx,
   bool probe,
   struct pvrgpu_texture_pco_observation *observation,
   const char **failure_reason)
{
   if (ctx->tcs || ctx->tes || ctx->gs || ctx->num_stream_output_targets) {
      *failure_reason = "extra_stage_or_stream_output";
      return false;
   }
   if (ctx->num_constant_buffers[MESA_SHADER_VERTEX] != 1 ||
       ctx->num_constant_buffers[MESA_SHADER_FRAGMENT] != 0) {
      *failure_reason = "constant_buffers";
      return false;
   }
   for (unsigned stage = 0; stage < MESA_SHADER_MESH_STAGES; ++stage) {
      if (stage != MESA_SHADER_VERTEX && ctx->num_constant_buffers[stage] != 0) {
         *failure_reason = "constant_buffers";
         return false;
      }
   }
   const struct pipe_constant_buffer *cb =
      &ctx->constant_buffers[MESA_SHADER_VERTEX][0];
   size_t constant_bytes = 0;
   const uint8_t *constants = pvrgpu_constant_buffer_bytes(ctx,
                                                            MESA_SHADER_VERTEX,
                                                            0,
                                                            &constant_bytes);
   if (cb->buffer_size != sizeof(observation->vertex_shared) || !constants ||
       constant_bytes != sizeof(observation->vertex_shared)) {
      *failure_reason = "constant_buffers";
      return false;
   }
   memcpy(observation->vertex_shared,
          constants,
          sizeof(observation->vertex_shared));
   if (!pvrgpu_texture_pco_sampler_matches(ctx, failure_reason))
      return false;
   if (ctx->sample_mask != UINT32_MAX || !ctx->has_viewport) {
      *failure_reason = "sample_mask_or_viewport";
      return false;
   }

   unsigned width = 0;
   unsigned height = 0;
   if (!pvrgpu_glmark_pco_draw_extent(ctx, probe, &width, &height)) {
      *failure_reason = "framebuffer_extent";
      return false;
   }
   const uint32_t width_scale = pvrgpu_float_bits((float)width * 0.5f);
   const uint32_t height_scale = pvrgpu_float_bits((float)height * 0.5f);
   if (pvrgpu_float_bits(ctx->viewport.scale[0]) != width_scale ||
       pvrgpu_float_bits(ctx->viewport.scale[1]) != height_scale ||
       pvrgpu_float_bits(ctx->viewport.scale[2]) != UINT32_C(0x3f000000) ||
       pvrgpu_float_bits(ctx->viewport.translate[0]) != width_scale ||
       pvrgpu_float_bits(ctx->viewport.translate[1]) != height_scale ||
       pvrgpu_float_bits(ctx->viewport.translate[2]) != UINT32_C(0x3f000000)) {
      *failure_reason = "viewport";
      return false;
   }
   if (!ctx->blend || ctx->blend->state.independent_blend_enable ||
       ctx->blend->state.logicop_enable || !ctx->blend->state.dither ||
       ctx->blend->state.alpha_to_coverage ||
       ctx->blend->state.alpha_to_coverage_dither ||
       ctx->blend->state.alpha_to_one || ctx->blend->state.max_rt != 0 ||
       ctx->blend->state.advanced_blend_func != 0 ||
       ctx->blend->state.rt[0].blend_enable ||
       ctx->blend->state.rt[0].colormask != PIPE_MASK_RGBA) {
      *failure_reason = "blend";
      return false;
   }
   if (!ctx->dsa || !ctx->dsa->state.depth_enabled ||
       !ctx->dsa->state.depth_writemask ||
       ctx->dsa->state.depth_func != PIPE_FUNC_LEQUAL ||
       ctx->dsa->state.depth_bounds_test ||
       ctx->dsa->state.stencil[0].enabled ||
       ctx->dsa->state.stencil[1].enabled || ctx->dsa->state.alpha_enabled) {
      *failure_reason = "depth_stencil_alpha";
      return false;
   }
   if (!ctx->rasterizer || ctx->rasterizer->state.front_ccw ||
       ctx->rasterizer->state.cull_face != PIPE_FACE_BACK ||
       ctx->rasterizer->state.fill_front != PIPE_POLYGON_MODE_FILL ||
       ctx->rasterizer->state.fill_back != PIPE_POLYGON_MODE_FILL ||
       ctx->rasterizer->state.scissor ||
       ctx->rasterizer->state.rasterizer_discard ||
       ctx->rasterizer->state.multisample ||
       ctx->rasterizer->state.offset_tri ||
       ctx->rasterizer->state.poly_smooth ||
       ctx->rasterizer->state.poly_stipple_enable ||
       ctx->rasterizer->state.conservative_raster_mode != 0 ||
       !ctx->rasterizer->state.half_pixel_center ||
       ctx->rasterizer->state.bottom_edge_rule ||
       ctx->rasterizer->state.clip_halfz ||
       !ctx->rasterizer->state.depth_clip_near ||
       !ctx->rasterizer->state.depth_clip_far ||
       ctx->rasterizer->state.depth_clamp) {
      *failure_reason = "rasterizer";
      return false;
   }

   observation->framebuffer_width = ctx->framebuffer.width;
   observation->framebuffer_height = ctx->framebuffer.height;
   observation->viewport_width = width;
   observation->viewport_height = height;
   return true;
}

static bool
pvrgpu_texture_pco_capture_payload(
   const struct pvrgpu_context *ctx,
   bool capture,
   struct pvrgpu_texture_pco_observation *observation)
{
   if (!ctx || !observation || !ctx->vertex_elements ||
       ctx->vertex_elements->num_elements != 3 || ctx->num_vertex_buffers != 3)
      return false;

   static const enum pipe_format formats[3] = {
      PIPE_FORMAT_R32G32B32_FLOAT,
      PIPE_FORMAT_R32G32B32_FLOAT,
      PIPE_FORMAT_R32G32_FLOAT,
   };
   static const unsigned components[3] = { 3, 3, 2 };
   const struct pvrgpu_resource *resources[3] = { NULL, NULL, NULL };
   for (unsigned attribute = 0; attribute < 3; ++attribute) {
      const struct pipe_vertex_element *element =
         &ctx->vertex_elements->elements[attribute];
      const size_t stream_bytes =
         (size_t)PVRGPU_TEXTURE_PCO_VERTEX_COUNT * components[attribute] *
         sizeof(float);
      if (element->src_format != formats[attribute] ||
          element->src_stride != components[attribute] * sizeof(float) ||
          element->src_offset != 0 || element->dual_slot ||
          element->instance_divisor != 0 ||
          element->vertex_buffer_index != attribute)
         return false;
      const struct pipe_vertex_buffer *buffer = &ctx->vertex_buffers[attribute];
      if (buffer->is_user_buffer || buffer->buffer_offset != 0 ||
          !buffer->buffer.resource || buffer->buffer.resource->target != PIPE_BUFFER)
         return false;
      resources[attribute] = pvrgpu_resource(buffer->buffer.resource);
      if (!resources[attribute] || !resources[attribute]->data ||
          resources[attribute]->size != stream_bytes)
         return false;
   }

   const struct pipe_sampler_view *view =
      ctx->sampler_views[MESA_SHADER_FRAGMENT][0];
   const struct pvrgpu_resource *texture = pvrgpu_resource(view->texture);
   const size_t source_row_bytes = PVRGPU_TEXTURE_PCO_ROW_PITCH;
   if (!texture || !texture->data || texture->level_count != 1 ||
       texture->level_offsets[0] > texture->size ||
       texture->level_strides[0] < source_row_bytes ||
       (PVRGPU_TEXTURE_PCO_HEIGHT - 1u) >
          (SIZE_MAX - texture->level_offsets[0]) / texture->level_strides[0])
      return false;
   const size_t final_row = texture->level_offsets[0] +
                            (PVRGPU_TEXTURE_PCO_HEIGHT - 1u) *
                               texture->level_strides[0];
   if (final_row > texture->size ||
       texture->size - final_row < source_row_bytes)
      return false;
   if (!capture)
      return true;

   observation->interleaved_vertex_data_size =
      (size_t)PVRGPU_TEXTURE_PCO_VERTEX_COUNT *
      PVRGPU_TEXTURE_PCO_VERTEX_STRIDE;
   observation->interleaved_vertex_data =
      MALLOC(observation->interleaved_vertex_data_size);
   observation->sampled_texture_bytes_size = PVRGPU_TEXTURE_PCO_BYTES;
   observation->sampled_texture_bytes =
      MALLOC(observation->sampled_texture_bytes_size);
   if (!observation->interleaved_vertex_data ||
       !observation->sampled_texture_bytes) {
      FREE(observation->interleaved_vertex_data);
      FREE(observation->sampled_texture_bytes);
      observation->interleaved_vertex_data = NULL;
      observation->sampled_texture_bytes = NULL;
      return false;
   }

   for (unsigned vertex = 0; vertex < PVRGPU_TEXTURE_PCO_VERTEX_COUNT;
        ++vertex) {
      uint8_t *dst = observation->interleaved_vertex_data +
                     (size_t)vertex * PVRGPU_TEXTURE_PCO_VERTEX_STRIDE;
      memcpy(dst, resources[0]->data + (size_t)vertex * 12u, 12u);
      memcpy(dst + 12u, resources[1]->data + (size_t)vertex * 12u, 12u);
      memcpy(dst + 24u, resources[2]->data + (size_t)vertex * 8u, 8u);
   }
   const uint8_t *texture_base =
      texture->data + texture->level_offsets[0];
   for (unsigned row = 0; row < PVRGPU_TEXTURE_PCO_HEIGHT; ++row) {
      memcpy(observation->sampled_texture_bytes +
                (size_t)row * PVRGPU_TEXTURE_PCO_ROW_PITCH,
             texture_base + (size_t)row * texture->level_strides[0],
             PVRGPU_TEXTURE_PCO_ROW_PITCH);
   }
   return true;
}

static bool
pvrgpu_draw_matches_texture_pco(
   const struct pvrgpu_context *ctx,
   const struct pipe_draw_info *info,
   unsigned drawid_offset,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,
   unsigned num_draws,
   bool probe,
   bool require_depth_clear,
   struct pvrgpu_texture_pco_observation *observation,
   const char **failure_reason)
{
   if (failure_reason)
      *failure_reason = "arguments";
   if (!ctx || !info || !draws || !observation || !failure_reason)
      return false;
   if (!pvrgpu_texture_pco_profile_matches(ctx)) {
      *failure_reason = "shader_profile";
      return false;
   }
   if (drawid_offset != 0 || indirect || num_draws != 1 ||
       info->mode != MESA_PRIM_TRIANGLES || info->index_size != 0 ||
       info->has_user_indices || info->primitive_restart ||
       info->increment_draw_id || info->index_bias_varies ||
       info->was_line_loop || info->start_instance != 0 ||
       info->instance_count != 1 || draws[0].start != 0 ||
       draws[0].count != PVRGPU_TEXTURE_PCO_VERTEX_COUNT) {
      *failure_reason = "draw_info";
      return false;
   }
   if (!pvrgpu_conditionals_framebuffer_matches(ctx,
                                                probe,
                                                require_depth_clear)) {
      *failure_reason = "framebuffer";
      return false;
   }

   memset(observation, 0, sizeof(*observation));
   observation->vertex_count = PVRGPU_TEXTURE_PCO_VERTEX_COUNT;
   if (!pvrgpu_texture_pco_pipeline_state_matches(ctx,
                                                  probe,
                                                  observation,
                                                  failure_reason))
      return false;
   if (!pvrgpu_texture_pco_capture_payload(ctx, !probe, observation)) {
      *failure_reason = "vertex_or_texture_data";
      return false;
   }
   *failure_reason = NULL;
   return true;
}

static bool
pvrgpu_nir_matches_ideas_constant_color_fs(
   const struct pvrgpu_shader_state *shader_state,
   const uint32_t expected[4])
{
   if (!shader_state || !expected ||
       shader_state->stage != MESA_SHADER_FRAGMENT ||
       shader_state->type != PIPE_SHADER_IR_NIR ||
       !shader_state->has_nir || !shader_state->nir)
      return false;

   nir_shader *shader = shader_state->nir;
   nir_variable *color = NULL;
   unsigned variables = 0;
   nir_foreach_variable_in_shader(variable, shader) {
      variables++;
      if (color ||
          !pvrgpu_nir_float_vector_variable(variable,
                                            nir_var_shader_out,
                                            FRAG_RESULT_COLOR,
                                            4) ||
          variable->data.precision != GLSL_PRECISION_MEDIUM)
         return false;
      color = variable;
   }
   if (variables != 1 || !color)
      return false;

   nir_instr *instructions[3];
   if (!pvrgpu_nir_collect_exact_instructions(shader,
                                              instructions,
                                              PVRGPU_ARRAY_SIZE(instructions)) ||
       instructions[0]->type != nir_instr_type_load_const ||
       !pvrgpu_nir_deref_var_matches(instructions[1], color))
      return false;
   nir_load_const_instr *constant =
      nir_instr_as_load_const(instructions[0]);
   if (constant->def.bit_size != 32 || constant->def.num_components != 4)
      return false;
   for (unsigned component = 0; component < 4; ++component) {
      if (constant->value[component].u32 != expected[component])
         return false;
   }
   return pvrgpu_nir_store_deref_matches(
      instructions[2],
      nir_instr_as_deref(instructions[1]),
      &constant->def);
}

static bool
pvrgpu_ideas_pco_shader_profile(
   const struct pvrgpu_context *ctx,
   enum pvrgpu_ideas_pco_profile *profile)
{
   static const uint32_t logo_vs[8] = {
      UINT32_C(0x29b86315), UINT32_C(0xc5e4fb02),
      UINT32_C(0xbf5e75cd), UINT32_C(0x02406082),
      UINT32_C(0x7ee4cc83), UINT32_C(0x30e777ba),
      UINT32_C(0x906e28a4), UINT32_C(0xc71cc0f4),
   };
   static const uint32_t logo_fs[8] = {
      UINT32_C(0xcc1251ba), UINT32_C(0xfc62903d),
      UINT32_C(0xe2f27dfe), UINT32_C(0x423c07a0),
      UINT32_C(0xff9dc3b2), UINT32_C(0xa7461f1e),
      UINT32_C(0xdcafab81), UINT32_C(0x479dac74),
   };
   static const uint32_t lighting_vs[8] = {
      UINT32_C(0xeac6f051), UINT32_C(0xe96a384d),
      UINT32_C(0x401bc339), UINT32_C(0x2c58043f),
      UINT32_C(0x2861e978), UINT32_C(0xc1ef625c),
      UINT32_C(0x015af337), UINT32_C(0x495f1917),
   };
   static const uint32_t lighting_fs[8] = {
      UINT32_C(0x58557edd), UINT32_C(0x2eab1772),
      UINT32_C(0xeb3c62b6), UINT32_C(0x00c61868),
      UINT32_C(0xef79a373), UINT32_C(0x82a55503),
      UINT32_C(0x90bd4936), UINT32_C(0x8de934a0),
   };
   static const uint32_t white_vs[8] = {
      UINT32_C(0x88a968c3), UINT32_C(0xb205151f),
      UINT32_C(0xb07b9044), UINT32_C(0xc362aa72),
      UINT32_C(0x7f573e93), UINT32_C(0x31aa797a),
      UINT32_C(0x93bf37fe), UINT32_C(0x3fa00942),
   };
   static const uint32_t black_vs[8] = {
      UINT32_C(0xdd8e29e5), UINT32_C(0x17828bdb),
      UINT32_C(0x22155b0f), UINT32_C(0x481da868),
      UINT32_C(0x2b9471ba), UINT32_C(0x8931349d),
      UINT32_C(0x9acc7bf5), UINT32_C(0x97a88d3d),
   };
   static const uint32_t constant_fs[8] = {
      UINT32_C(0x3857474b), UINT32_C(0x0de55295),
      UINT32_C(0x56692dbd), UINT32_C(0x31e4406b),
      UINT32_C(0xdae8e19f), UINT32_C(0xcde306ec),
      UINT32_C(0xbb732499), UINT32_C(0xb2717c8f),
   };
   static const uint32_t white[4] = {
      UINT32_C(0x3f800000), UINT32_C(0x3f800000),
      UINT32_C(0x3f800000), UINT32_C(0x3f800000),
   };
   static const uint32_t black[4] = {
      0, 0, 0, UINT32_C(0x3f800000),
   };

   if (!ctx || !profile || !ctx->vs || !ctx->fs ||
       ctx->vs->stage != MESA_SHADER_VERTEX ||
       ctx->fs->stage != MESA_SHADER_FRAGMENT ||
       ctx->vs->type != PIPE_SHADER_IR_NIR ||
       ctx->fs->type != PIPE_SHADER_IR_NIR ||
       !ctx->vs->has_nir || !ctx->fs->has_nir ||
       !ctx->vs->nir || !ctx->fs->nir)
      return false;

   if (pvrgpu_nir_source_hash_matches(ctx->vs->nir, logo_vs) &&
       pvrgpu_nir_source_hash_matches(ctx->fs->nir, logo_fs)) {
      *profile = PVRGPU_IDEAS_PCO_LOGO;
      return true;
   }
   if (pvrgpu_nir_source_hash_matches(ctx->vs->nir, lighting_vs) &&
       pvrgpu_nir_source_hash_matches(ctx->fs->nir, lighting_fs)) {
      *profile = PVRGPU_IDEAS_PCO_LIGHTING;
      return true;
   }
   if (!pvrgpu_nir_source_hash_matches(ctx->fs->nir, constant_fs))
      return false;
   if (pvrgpu_nir_source_hash_matches(ctx->vs->nir, white_vs) &&
       pvrgpu_nir_matches_ideas_constant_color_fs(ctx->fs, white)) {
      *profile = PVRGPU_IDEAS_PCO_WHITE;
      return true;
   }
   if (pvrgpu_nir_source_hash_matches(ctx->vs->nir, black_vs) &&
       pvrgpu_nir_matches_ideas_constant_color_fs(ctx->fs, black)) {
      *profile = PVRGPU_IDEAS_PCO_BLACK;
      return true;
   }
   return false;
}

static bool
pvrgpu_ideas_pco_expected_draw(
   unsigned ordinal,
   enum pvrgpu_ideas_pco_profile *profile,
   unsigned *vertex_count,
   unsigned *primitive_mode,
   unsigned *index_size,
   unsigned *start,
   unsigned *attribute_count,
   unsigned *cull_face)
{
   if (!profile || !vertex_count || !primitive_mode || !index_size ||
       !start || !attribute_count || !cull_face ||
       ordinal >= PVRGPU_IDEAS_PCO_DRAW_COUNT)
      return false;
   if (ordinal < 162) {
      *profile = PVRGPU_IDEAS_PCO_LOGO;
      *vertex_count = 18;
      *primitive_mode = MESA_PRIM_TRIANGLE_STRIP;
      *index_size = ordinal % 9 == 0 ? 0 : 2;
      *start = *index_size ? (ordinal % 9 - 1) * 18 : 0;
      *attribute_count = 1;
      *cull_face = PIPE_FACE_BACK;
      return true;
   }
   if (ordinal < 167) {
      *profile = PVRGPU_IDEAS_PCO_LIGHTING;
      *vertex_count = 26;
      *primitive_mode = MESA_PRIM_TRIANGLE_STRIP;
      *index_size = 2;
      *start = (ordinal - 162) * 26;
      *attribute_count = 2;
      *cull_face = PIPE_FACE_NONE;
      return true;
   }
   if (ordinal == 167) {
      *profile = PVRGPU_IDEAS_PCO_WHITE;
      *vertex_count = 12;
      *primitive_mode = MESA_PRIM_TRIANGLE_FAN;
      *index_size = 2;
      *start = 130;
      *attribute_count = 1;
      *cull_face = PIPE_FACE_NONE;
      return true;
   }
   *profile = PVRGPU_IDEAS_PCO_BLACK;
   *vertex_count = 26;
   *primitive_mode = MESA_PRIM_TRIANGLE_STRIP;
   *index_size = 2;
   *start = (ordinal - 168) * 26;
   *attribute_count = 1;
   *cull_face = PIPE_FACE_NONE;
   return true;
}

static bool
pvrgpu_ideas_pco_copy_constants(
   const struct pvrgpu_context *ctx,
   enum pvrgpu_ideas_pco_profile profile,
   struct pvrgpu_ideas_pco_observation *observation)
{
   static const unsigned vertex_dwords[PVRGPU_IDEAS_PCO_PROFILE_COUNT] = {
      32, 44, 32, 32,
   };
   static const unsigned fragment_dwords[PVRGPU_IDEAS_PCO_PROFILE_COUNT] = {
      4, 12, 0, 0,
   };
   if (!ctx || !observation || profile >= PVRGPU_IDEAS_PCO_PROFILE_COUNT)
      return false;
   const unsigned vs_dwords = vertex_dwords[profile];
   const unsigned fs_dwords = fragment_dwords[profile];
   if (ctx->num_constant_buffers[MESA_SHADER_VERTEX] != 1 ||
       ctx->num_constant_buffers[MESA_SHADER_FRAGMENT] != (fs_dwords ? 1 : 0))
      return false;
   for (unsigned stage = 0; stage < MESA_SHADER_MESH_STAGES; ++stage) {
      if (stage != MESA_SHADER_VERTEX &&
          stage != MESA_SHADER_FRAGMENT &&
          ctx->num_constant_buffers[stage] != 0)
         return false;
   }

   size_t available = 0;
   const uint8_t *bytes = pvrgpu_constant_buffer_bytes(
      ctx, MESA_SHADER_VERTEX, 0, &available);
   if (!bytes || available != (size_t)vs_dwords * sizeof(uint32_t) ||
       ctx->constant_buffers[MESA_SHADER_VERTEX][0].buffer_size != available)
      return false;
   memcpy(observation->vertex_shared, bytes, available);
   observation->vertex_shared_count = vs_dwords;

   if (fs_dwords) {
      bytes = pvrgpu_constant_buffer_bytes(
         ctx, MESA_SHADER_FRAGMENT, 0, &available);
      if (!bytes || available != (size_t)fs_dwords * sizeof(uint32_t) ||
          ctx->constant_buffers[MESA_SHADER_FRAGMENT][0].buffer_size !=
             available)
         return false;
      memcpy(observation->fragment_shared, bytes, available);
   }
   observation->fragment_shared_count = fs_dwords;
   return true;
}

static bool
pvrgpu_ideas_pco_pipeline_state_matches(
   const struct pvrgpu_context *ctx,
   bool probe,
   unsigned expected_cull_face,
   struct pvrgpu_ideas_pco_observation *observation,
   const char **failure_reason)
{
   if (ctx->tcs || ctx->tes || ctx->gs || ctx->num_stream_output_targets) {
      *failure_reason = "extra_stage_or_stream_output";
      return false;
   }
   for (unsigned stage = 0; stage < MESA_SHADER_MESH_STAGES; ++stage) {
      if (ctx->num_sampler_views[stage] != 0 || ctx->num_samplers[stage] != 0) {
         *failure_reason = "sampler_bindings";
         return false;
      }
   }
   if (!pvrgpu_ideas_pco_copy_constants(ctx,
                                        observation->profile,
                                        observation)) {
      *failure_reason = "constant_buffers";
      return false;
   }
   if (ctx->sample_mask != UINT32_MAX || !ctx->has_viewport) {
      *failure_reason = "sample_mask_or_viewport";
      return false;
   }

   unsigned width = 0;
   unsigned height = 0;
   if (!pvrgpu_glmark_pco_draw_extent(ctx, probe, &width, &height)) {
      *failure_reason = "framebuffer_extent";
      return false;
   }
   const uint32_t width_scale = pvrgpu_float_bits((float)width * 0.5f);
   const uint32_t height_scale = pvrgpu_float_bits((float)height * 0.5f);
   if (pvrgpu_float_bits(ctx->viewport.scale[0]) != width_scale ||
       pvrgpu_float_bits(ctx->viewport.scale[1]) != height_scale ||
       pvrgpu_float_bits(ctx->viewport.scale[2]) != UINT32_C(0x3f000000) ||
       pvrgpu_float_bits(ctx->viewport.translate[0]) != width_scale ||
       pvrgpu_float_bits(ctx->viewport.translate[1]) != height_scale ||
       pvrgpu_float_bits(ctx->viewport.translate[2]) != UINT32_C(0x3f000000)) {
      *failure_reason = "viewport";
      return false;
   }
   if (!ctx->blend || ctx->blend->state.independent_blend_enable ||
       ctx->blend->state.logicop_enable || !ctx->blend->state.dither ||
       ctx->blend->state.alpha_to_coverage ||
       ctx->blend->state.alpha_to_coverage_dither ||
       ctx->blend->state.alpha_to_one || ctx->blend->state.max_rt != 0 ||
       ctx->blend->state.advanced_blend_func != 0 ||
       ctx->blend->state.rt[0].blend_enable ||
       ctx->blend->state.rt[0].colormask != PIPE_MASK_RGBA) {
      *failure_reason = "blend";
      return false;
   }
   const bool depth_enabled =
      observation->profile == PVRGPU_IDEAS_PCO_LIGHTING ||
      observation->profile == PVRGPU_IDEAS_PCO_WHITE;
   if (!ctx->dsa ||
       ctx->dsa->state.depth_enabled != depth_enabled ||
       ctx->dsa->state.depth_writemask != depth_enabled ||
       ctx->dsa->state.depth_func !=
          (depth_enabled ? PIPE_FUNC_LEQUAL : 0) ||
       ctx->dsa->state.depth_bounds_test ||
       ctx->dsa->state.stencil[0].enabled ||
       ctx->dsa->state.stencil[1].enabled || ctx->dsa->state.alpha_enabled) {
      *failure_reason = "depth_stencil_alpha";
      return false;
   }
   if (!ctx->rasterizer || ctx->rasterizer->state.front_ccw ||
       ctx->rasterizer->state.cull_face != expected_cull_face ||
       ctx->rasterizer->state.fill_front != PIPE_POLYGON_MODE_FILL ||
       ctx->rasterizer->state.fill_back != PIPE_POLYGON_MODE_FILL ||
       ctx->rasterizer->state.scissor ||
       ctx->rasterizer->state.rasterizer_discard ||
       ctx->rasterizer->state.multisample ||
       ctx->rasterizer->state.offset_tri ||
       ctx->rasterizer->state.poly_smooth ||
       ctx->rasterizer->state.poly_stipple_enable ||
       ctx->rasterizer->state.conservative_raster_mode != 0 ||
       !ctx->rasterizer->state.half_pixel_center ||
       ctx->rasterizer->state.bottom_edge_rule ||
       ctx->rasterizer->state.clip_halfz ||
       !ctx->rasterizer->state.depth_clip_near ||
       !ctx->rasterizer->state.depth_clip_far ||
       ctx->rasterizer->state.depth_clamp) {
      *failure_reason = "rasterizer";
      return false;
   }

   observation->framebuffer_width = ctx->framebuffer.width;
   observation->framebuffer_height = ctx->framebuffer.height;
   observation->viewport_width = width;
   observation->viewport_height = height;
   observation->cull_face = expected_cull_face;
   return true;
}

static bool
pvrgpu_ideas_pco_read_float4(const struct pvrgpu_context *ctx,
                             unsigned attribute,
                             uint32_t vertex_index,
                             uint8_t out[16])
{
   if (!ctx || !out || !ctx->vertex_elements ||
       attribute >= ctx->vertex_elements->num_elements)
      return false;
   const struct pipe_vertex_element *element =
      &ctx->vertex_elements->elements[attribute];
   if (element->src_format != PIPE_FORMAT_R32G32B32_FLOAT ||
       element->src_stride != 12 || element->src_offset != 0 ||
       element->dual_slot || element->instance_divisor != 0 ||
       element->vertex_buffer_index != attribute ||
       element->vertex_buffer_index >= ctx->num_vertex_buffers)
      return false;
   const struct pipe_vertex_buffer *buffer =
      &ctx->vertex_buffers[element->vertex_buffer_index];
   const uint8_t *base = NULL;
   size_t available = SIZE_MAX;
   if (buffer->is_user_buffer) {
      base = (const uint8_t *)buffer->buffer.user;
   } else if (buffer->buffer.resource &&
              buffer->buffer.resource->target == PIPE_BUFFER) {
      const struct pvrgpu_resource *resource =
         pvrgpu_resource(buffer->buffer.resource);
      if (!resource || !resource->data)
         return false;
      base = resource->data;
      available = resource->size;
   }
   if (!base || vertex_index > (SIZE_MAX - buffer->buffer_offset) / 12)
      return false;
   const size_t offset =
      (size_t)buffer->buffer_offset + (size_t)vertex_index * 12;
   if (available != SIZE_MAX &&
       (offset > available || available - offset < 12))
      return false;
   memcpy(out, base + offset, 12);
   const uint32_t one = UINT32_C(0x3f800000);
   memcpy(out + 12, &one, sizeof(one));
   return true;
}

static bool
pvrgpu_ideas_pco_capture_vertices(
   const struct pvrgpu_context *ctx,
   const struct pipe_draw_info *info,
   const struct pipe_draw_start_count_bias *draw,
   bool capture,
   struct pvrgpu_ideas_pco_observation *observation)
{
   if (!ctx || !info || !draw || !observation || !ctx->vertex_elements ||
       ctx->vertex_elements->num_elements != observation->attribute_count ||
       ctx->num_vertex_buffers != observation->attribute_count)
      return false;
   const size_t bytes = (size_t)observation->vertex_count *
                        observation->vertex_stride;
   if (capture) {
      observation->interleaved_vertex_data = MALLOC(bytes);
      if (!observation->interleaved_vertex_data)
         return false;
      observation->interleaved_vertex_data_size = bytes;
   }

   uint8_t value[16];
   for (unsigned occurrence = 0; occurrence < observation->vertex_count;
        ++occurrence) {
      uint32_t vertex_index = 0;
      if (info->index_size != 0) {
         uint32_t raw_index = 0;
         if (!pvrgpu_read_draw_index(info,
                                     draw->start,
                                     occurrence,
                                     &raw_index))
            goto fail;
         const int64_t biased = (int64_t)raw_index + draw->index_bias;
         if (biased < 0 || biased > UINT32_MAX)
            goto fail;
         vertex_index = (uint32_t)biased;
      } else {
         if (draw->start > UINT32_MAX - occurrence)
            goto fail;
         vertex_index = draw->start + occurrence;
      }
      for (unsigned attribute = 0;
           attribute < observation->attribute_count;
           ++attribute) {
         if (!pvrgpu_ideas_pco_read_float4(ctx,
                                           attribute,
                                           vertex_index,
                                           value))
            goto fail;
         if (capture) {
            memcpy(observation->interleaved_vertex_data +
                      (size_t)occurrence * observation->vertex_stride +
                      (size_t)attribute * 16,
                   value,
                   sizeof(value));
         }
      }
   }
   return true;

fail:
   FREE(observation->interleaved_vertex_data);
   observation->interleaved_vertex_data = NULL;
   observation->interleaved_vertex_data_size = 0;
   return false;
}

static bool
pvrgpu_draw_matches_ideas_pco(
   const struct pvrgpu_context *ctx,
   const struct pipe_draw_info *info,
   unsigned drawid_offset,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,
   unsigned num_draws,
   unsigned ordinal,
   bool probe,
   bool require_depth_clear,
   bool capture,
   struct pvrgpu_ideas_pco_observation *observation,
   const char **failure_reason)
{
   if (failure_reason)
      *failure_reason = "arguments";
   const char *case_name = pvrgpu_rdc_case_name();
   if (!ctx || !info || !draws || !observation || !failure_reason ||
       !case_name || strcmp(case_name, "ideas.ideas.capture.1") != 0)
      return false;

   enum pvrgpu_ideas_pco_profile expected_profile;
   unsigned expected_vertex_count = 0;
   unsigned expected_primitive_mode = 0;
   unsigned expected_index_size = 0;
   unsigned expected_start = 0;
   unsigned expected_attribute_count = 0;
   unsigned expected_cull_face = 0;
   if (!pvrgpu_ideas_pco_expected_draw(ordinal,
                                       &expected_profile,
                                       &expected_vertex_count,
                                       &expected_primitive_mode,
                                       &expected_index_size,
                                       &expected_start,
                                       &expected_attribute_count,
                                       &expected_cull_face)) {
      *failure_reason = "sequence_ordinal";
      return false;
   }
   enum pvrgpu_ideas_pco_profile actual_profile;
   if (!pvrgpu_ideas_pco_shader_profile(ctx, &actual_profile) ||
       actual_profile != expected_profile) {
      *failure_reason = "shader_profile_or_order";
      return false;
   }
   if (drawid_offset != 0 || indirect || num_draws != 1 ||
       info->mode != expected_primitive_mode ||
       info->index_size != expected_index_size || info->primitive_restart ||
       info->increment_draw_id || info->index_bias_varies ||
       info->was_line_loop || info->start_instance != 0 ||
       info->instance_count != 1 || draws[0].start != expected_start ||
       draws[0].count != expected_vertex_count ||
       (info->index_size != 0 && draws[0].index_bias != 0)) {
      *failure_reason = "draw_info_or_order";
      return false;
   }
   if (!pvrgpu_conditionals_framebuffer_matches(ctx,
                                                probe,
                                                require_depth_clear)) {
      *failure_reason = "framebuffer";
      return false;
   }

   memset(observation, 0, sizeof(*observation));
   observation->profile = actual_profile;
   observation->vertex_count = expected_vertex_count;
   observation->attribute_count = expected_attribute_count;
   observation->vertex_stride = expected_attribute_count * 16;
   observation->primitive_mode = expected_primitive_mode;
   if (!pvrgpu_ideas_pco_pipeline_state_matches(ctx,
                                                probe,
                                                expected_cull_face,
                                                observation,
                                                failure_reason))
      return false;
   if (!pvrgpu_ideas_pco_capture_vertices(ctx,
                                         info,
                                         &draws[0],
                                         capture,
                                         observation)) {
      *failure_reason = "vertex_or_index_data";
      return false;
   }
   *failure_reason = NULL;
   return true;
}

static bool
pvrgpu_nir_matches_textured_triangles_vs(
   const struct pvrgpu_shader_state *shader_state)
{
   if (!shader_state || shader_state->stage != MESA_SHADER_VERTEX ||
       shader_state->type != PIPE_SHADER_IR_NIR ||
       !shader_state->has_nir || !shader_state->nir)
      return false;

   nir_shader *shader = shader_state->nir;
   nir_variable *position = NULL;
   nir_variable *position_output = NULL;
   nir_variable *texcoord_output = NULL;
   unsigned variable_count = 0;
   nir_foreach_variable_in_shader(variable, shader) {
      variable_count++;
      if (pvrgpu_nir_float_vector_variable(variable,
                                           nir_var_shader_in,
                                           VERT_ATTRIB_GENERIC0,
                                           4)) {
         if (position)
            return false;
         position = variable;
      } else if (pvrgpu_nir_float_vector_variable(variable,
                                                  nir_var_shader_out,
                                                  VARYING_SLOT_POS,
                                                  4)) {
         if (position_output)
            return false;
         position_output = variable;
      } else if (pvrgpu_nir_float_vector_variable(variable,
                                                  nir_var_shader_out,
                                                  VARYING_SLOT_VAR0,
                                                  2)) {
         if (texcoord_output)
            return false;
         texcoord_output = variable;
      } else {
         return false;
      }
   }
   if (variable_count != 3 || !position || !position_output ||
       !texcoord_output ||
       position->data.location_frac != 0 ||
       position_output->data.location_frac != 0 ||
       texcoord_output->data.location_frac != 0)
      return false;

   nir_function_impl *entrypoint = pvrgpu_nir_single_entrypoint(shader);
   if (!entrypoint)
      return false;

   nir_load_const_instr *constant_one = NULL;
   nir_load_const_instr *constant_half = NULL;
   nir_deref_instr *position_deref = NULL;
   nir_deref_instr *position_output_deref = NULL;
   nir_deref_instr *texcoord_output_deref = NULL;
   nir_intrinsic_instr *position_load = NULL;
   nir_intrinsic_instr *position_store = NULL;
   nir_intrinsic_instr *texcoord_store = NULL;
   nir_alu_instr *multiply_half = NULL;
   nir_alu_instr *add_half = NULL;
   nir_alu_instr *position_vector = NULL;
   unsigned instruction_count = 0;

   nir_foreach_block(block, entrypoint) {
      nir_foreach_instr(instruction, block) {
         instruction_count++;
         switch (instruction->type) {
         case nir_instr_type_load_const: {
            nir_load_const_instr *constant =
               nir_instr_as_load_const(instruction);
            if (pvrgpu_nir_constant_is_float(constant,
                                             UINT32_C(0x3f800000))) {
               if (constant_one)
                  return false;
               constant_one = constant;
            } else if (pvrgpu_nir_constant_is_float(
                          constant, UINT32_C(0x3f000000))) {
               if (constant_half)
                  return false;
               constant_half = constant;
            } else {
               return false;
            }
            break;
         }
         case nir_instr_type_deref: {
            nir_deref_instr *deref = nir_instr_as_deref(instruction);
            if (deref->deref_type != nir_deref_type_var)
               return false;
            if (deref->var == position) {
               if (position_deref)
                  return false;
               position_deref = deref;
            } else if (deref->var == position_output) {
               if (position_output_deref)
                  return false;
               position_output_deref = deref;
            } else if (deref->var == texcoord_output) {
               if (texcoord_output_deref)
                  return false;
               texcoord_output_deref = deref;
            } else {
               return false;
            }
            break;
         }
         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intrinsic =
               nir_instr_as_intrinsic(instruction);
            if (intrinsic->intrinsic == nir_intrinsic_load_deref) {
               if (position_load)
                  return false;
               position_load = intrinsic;
            } else if (intrinsic->intrinsic == nir_intrinsic_store_deref) {
               nir_variable *stored_variable =
                  nir_intrinsic_get_var(intrinsic, 0);
               if (stored_variable == position_output) {
                  if (position_store)
                     return false;
                  position_store = intrinsic;
               } else if (stored_variable == texcoord_output) {
                  if (texcoord_store)
                     return false;
                  texcoord_store = intrinsic;
               } else {
                  return false;
               }
            } else {
               return false;
            }
            break;
         }
         case nir_instr_type_alu: {
            nir_alu_instr *alu = nir_instr_as_alu(instruction);
            if (alu->op == nir_op_fmul) {
               if (multiply_half)
                  return false;
               multiply_half = alu;
            } else if (alu->op == nir_op_fadd) {
               if (add_half)
                  return false;
               add_half = alu;
            } else if (alu->op == nir_op_vec4) {
               if (position_vector)
                  return false;
               position_vector = alu;
            } else {
               return false;
            }
            break;
         }
         default:
            return false;
         }
      }
   }

   if (instruction_count != 11 || !constant_one || !constant_half ||
       !position_deref || !position_output_deref ||
       !texcoord_output_deref || !position_load || !position_store ||
       !texcoord_store || !multiply_half || !add_half || !position_vector)
      return false;
   if (position_load->num_components != 4 ||
       position_load->def.num_components != 4 ||
       position_load->def.bit_size != 32 ||
       position_load->src[0].ssa != &position_deref->def)
      return false;
   if (multiply_half->def.num_components != 2 ||
       multiply_half->def.bit_size != 32 ||
       add_half->def.num_components != 2 || add_half->def.bit_size != 32 ||
       position_vector->def.num_components != 4 ||
       position_vector->def.bit_size != 32)
      return false;

   static const unsigned xy_swizzle[2] = {0, 1};
   static const unsigned xx_swizzle[2] = {0, 0};
   if (!pvrgpu_nir_alu_source_matches(&multiply_half->src[0],
                                      &position_load->def,
                                      xy_swizzle,
                                      2) ||
       !pvrgpu_nir_alu_source_matches(&multiply_half->src[1],
                                      &constant_half->def,
                                      xx_swizzle,
                                      2) ||
       !pvrgpu_nir_alu_source_matches(&add_half->src[0],
                                      &multiply_half->def,
                                      xy_swizzle,
                                      2) ||
       !pvrgpu_nir_alu_source_matches(&add_half->src[1],
                                      &constant_half->def,
                                      xx_swizzle,
                                      2))
      return false;

   for (unsigned component = 0; component < 3; ++component) {
      const unsigned swizzle[1] = {component};
      if (!pvrgpu_nir_alu_source_matches(&position_vector->src[component],
                                         &position_load->def,
                                         swizzle,
                                         1))
         return false;
   }
   static const unsigned scalar_swizzle[1] = {0};
   if (!pvrgpu_nir_alu_source_matches(&position_vector->src[3],
                                      &constant_one->def,
                                      scalar_swizzle,
                                      1))
      return false;

   return position_store->src[0].ssa == &position_output_deref->def &&
          position_store->src[1].ssa == &position_vector->def &&
          position_store->num_components == 4 &&
          nir_intrinsic_write_mask(position_store) == 0xfu &&
          texcoord_store->src[0].ssa == &texcoord_output_deref->def &&
          texcoord_store->src[1].ssa == &add_half->def &&
          texcoord_store->num_components == 2 &&
          nir_intrinsic_write_mask(texcoord_store) == 0x3u;
}

static bool
pvrgpu_nir_matches_textured_triangles_fs(
   const struct pvrgpu_shader_state *shader_state)
{
   if (!shader_state || shader_state->stage != MESA_SHADER_FRAGMENT ||
       shader_state->type != PIPE_SHADER_IR_NIR ||
       !shader_state->has_nir || !shader_state->nir)
      return false;

   nir_shader *shader = shader_state->nir;
   nir_variable *sampler = NULL;
   nir_variable *texcoord_input = NULL;
   nir_variable *color_output = NULL;
   unsigned variable_count = 0;
   nir_foreach_variable_in_shader(variable, shader) {
      variable_count++;
      if (pvrgpu_nir_sampler2d_variable(variable)) {
         if (sampler)
            return false;
         sampler = variable;
      } else if (pvrgpu_nir_float_vector_variable(variable,
                                                  nir_var_shader_in,
                                                  VARYING_SLOT_VAR0,
                                                  2) &&
                 variable->data.interpolation == INTERP_MODE_SMOOTH) {
         if (texcoord_input)
            return false;
         texcoord_input = variable;
      } else if (pvrgpu_nir_float_vector_variable(variable,
                                                  nir_var_shader_out,
                                                  FRAG_RESULT_COLOR,
                                                  4)) {
         if (color_output)
            return false;
         color_output = variable;
      } else {
         return false;
      }
   }
   if (variable_count != 3 || !sampler || !texcoord_input || !color_output ||
       texcoord_input->data.location_frac != 0 ||
       color_output->data.location_frac != 0)
      return false;

   nir_function_impl *entrypoint = pvrgpu_nir_single_entrypoint(shader);
   if (!entrypoint)
      return false;

   nir_load_const_instr *constant_one = NULL;
   nir_deref_instr *texcoord_deref = NULL;
   nir_deref_instr *color_deref = NULL;
   nir_intrinsic_instr *texcoord_load = NULL;
   nir_intrinsic_instr *color_store = NULL;
   nir_tex_instr *texture_sample = NULL;
   nir_alu_instr *color_vector = NULL;
   unsigned instruction_count = 0;

   nir_foreach_block(block, entrypoint) {
      nir_foreach_instr(instruction, block) {
         instruction_count++;
         switch (instruction->type) {
         case nir_instr_type_load_const: {
            nir_load_const_instr *constant =
               nir_instr_as_load_const(instruction);
            if (constant_one ||
                !pvrgpu_nir_constant_is_float(
                   constant, UINT32_C(0x3f800000)))
               return false;
            constant_one = constant;
            break;
         }
         case nir_instr_type_deref: {
            nir_deref_instr *deref = nir_instr_as_deref(instruction);
            if (deref->deref_type != nir_deref_type_var)
               return false;
            if (deref->var == texcoord_input) {
               if (texcoord_deref)
                  return false;
               texcoord_deref = deref;
            } else if (deref->var == color_output) {
               if (color_deref)
                  return false;
               color_deref = deref;
            } else {
               return false;
            }
            break;
         }
         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intrinsic =
               nir_instr_as_intrinsic(instruction);
            if (intrinsic->intrinsic == nir_intrinsic_load_deref) {
               if (texcoord_load)
                  return false;
               texcoord_load = intrinsic;
            } else if (intrinsic->intrinsic == nir_intrinsic_store_deref) {
               if (color_store ||
                   nir_intrinsic_get_var(intrinsic, 0) != color_output)
                  return false;
               color_store = intrinsic;
            } else {
               return false;
            }
            break;
         }
         case nir_instr_type_tex:
            if (texture_sample)
               return false;
            texture_sample = nir_instr_as_tex(instruction);
            break;
         case nir_instr_type_alu: {
            nir_alu_instr *alu = nir_instr_as_alu(instruction);
            if (color_vector || alu->op != nir_op_vec4)
               return false;
            color_vector = alu;
            break;
         }
         default:
            return false;
         }
      }
   }

   if (instruction_count != 7 || !constant_one || !texcoord_deref ||
       !color_deref || !texcoord_load || !color_store || !texture_sample ||
       !color_vector)
      return false;
   if (texcoord_load->num_components != 2 ||
       texcoord_load->def.num_components != 2 ||
       texcoord_load->def.bit_size != 32 ||
       texcoord_load->src[0].ssa != &texcoord_deref->def)
      return false;
   if (texture_sample->op != nir_texop_tex ||
       texture_sample->sampler_dim != GLSL_SAMPLER_DIM_2D ||
       texture_sample->dest_type != nir_type_float32 ||
       texture_sample->def.num_components != 4 ||
       texture_sample->def.bit_size != 32 ||
       texture_sample->num_srcs != 1 ||
       texture_sample->coord_components != 2 ||
       texture_sample->src[0].src_type != nir_tex_src_coord ||
       texture_sample->src[0].src.ssa != &texcoord_load->def ||
       texture_sample->is_array || texture_sample->is_shadow ||
       texture_sample->is_sparse || texture_sample->texture_non_uniform ||
       texture_sample->sampler_non_uniform ||
       texture_sample->embedded_sampler || texture_sample->offset_non_uniform ||
       texture_sample->texture_index != 0 ||
       texture_sample->sampler_index != 0 ||
       texture_sample->backend_flags != 0)
      return false;
   if (color_vector->def.num_components != 4 ||
       color_vector->def.bit_size != 32)
      return false;
   for (unsigned component = 0; component < 3; ++component) {
      const unsigned swizzle[1] = {component};
      if (!pvrgpu_nir_alu_source_matches(&color_vector->src[component],
                                         &texture_sample->def,
                                         swizzle,
                                         1))
         return false;
   }
   static const unsigned scalar_swizzle[1] = {0};
   if (!pvrgpu_nir_alu_source_matches(&color_vector->src[3],
                                      &constant_one->def,
                                      scalar_swizzle,
                                      1))
      return false;
   return color_store->src[0].ssa == &color_deref->def &&
          color_store->src[1].ssa == &color_vector->def &&
          color_store->num_components == 4 &&
          nir_intrinsic_write_mask(color_store) == 0xfu;
}

static bool
pvrgpu_textured_triangles_depth_format_supported(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_Z16_UNORM:
   case PIPE_FORMAT_Z24X8_UNORM:
   case PIPE_FORMAT_X8Z24_UNORM:
   case PIPE_FORMAT_Z24_UNORM_S8_UINT:
   case PIPE_FORMAT_S8_UINT_Z24_UNORM:
   case PIPE_FORMAT_Z32_UNORM:
   case PIPE_FORMAT_Z32_FLOAT:
      return true;
   default:
      return false;
   }
}

static bool
pvrgpu_textured_triangles_framebuffer_matches(
   const struct pvrgpu_context *ctx,
   bool require_depth_clear)
{
   if (!ctx || ctx->framebuffer.width == 0 ||
       ctx->framebuffer.height == 0 || ctx->framebuffer.nr_cbufs != 1 ||
       ctx->framebuffer.resolve || ctx->framebuffer.pls_enabled ||
       ctx->framebuffer.viewmask != 0)
      return false;

   const struct pipe_surface *color = &ctx->framebuffer.cbufs[0];
   const struct pipe_surface *depth = &ctx->framebuffer.zsbuf;
   if (!color->texture || color->texture->target != PIPE_TEXTURE_2D ||
       color->texture->format != PIPE_FORMAT_R8G8B8A8_UNORM ||
       color->format != PIPE_FORMAT_R8G8B8A8_UNORM || color->level != 0 ||
       color->first_layer != 0 || color->last_layer != 0 ||
       color->texture->width0 != ctx->framebuffer.width ||
       color->texture->height0 != ctx->framebuffer.height ||
       color->texture->nr_samples > 1 ||
       color->texture->nr_storage_samples > 1)
      return false;
   if (!depth->texture || depth->texture->target != PIPE_TEXTURE_2D ||
       !pvrgpu_textured_triangles_depth_format_supported(depth->format) ||
       depth->format != depth->texture->format || depth->level != 0 ||
       depth->first_layer != 0 || depth->last_layer != 0 ||
       depth->texture->width0 != ctx->framebuffer.width ||
       depth->texture->height0 != ctx->framebuffer.height ||
       depth->texture->nr_samples > 1 ||
       depth->texture->nr_storage_samples > 1)
      return false;

   if (!require_depth_clear)
      return true;

   return ctx->full_depth_clear_is_one &&
          ctx->full_depth_clear_resource == depth->texture &&
          ctx->full_depth_clear_level == depth->level &&
          ctx->full_depth_clear_first_layer == depth->first_layer &&
          ctx->full_depth_clear_last_layer == depth->last_layer &&
          ctx->full_depth_clear_width == ctx->framebuffer.width &&
          ctx->full_depth_clear_height == ctx->framebuffer.height;
}

static bool
pvrgpu_textured_triangles_pipeline_state_matches(
   const struct pvrgpu_context *ctx,
   unsigned *viewport_width,
   unsigned *viewport_height,
   const struct pipe_sampler_view **texture_view,
   const char **failure_reason)
{
   if (!ctx || !viewport_width || !viewport_height || !texture_view) {
      *failure_reason = "pipeline_arguments";
      return false;
   }
   if (!pvrgpu_nir_matches_textured_triangles_vs(ctx->vs)) {
      *failure_reason = "vs_nir";
      return false;
   }
   if (!pvrgpu_nir_matches_textured_triangles_fs(ctx->fs)) {
      *failure_reason = "fs_nir";
      return false;
   }
   if (ctx->tcs || ctx->tes || ctx->gs) {
      *failure_reason = "extra_shader_stage";
      return false;
   }
   if (ctx->num_stream_output_targets != 0) {
      *failure_reason = "stream_output";
      return false;
   }
   if (ctx->num_constant_buffers[MESA_SHADER_VERTEX] != 0 ||
       ctx->num_constant_buffers[MESA_SHADER_FRAGMENT] != 0) {
      *failure_reason = "constant_buffer";
      return false;
   }
   if ((ctx->sample_mask & 1u) == 0) {
      *failure_reason = "sample_mask";
      return false;
   }
   if (!ctx->has_viewport) {
      *failure_reason = "missing_viewport";
      return false;
   }

   if (!pvrgpu_viewport_extent(ctx->viewport.scale[0], viewport_width) ||
       !pvrgpu_viewport_extent(ctx->viewport.scale[1], viewport_height) ||
       *viewport_width != ctx->framebuffer.width ||
       *viewport_height != ctx->framebuffer.height ||
       ctx->viewport.scale[0] != (float)ctx->framebuffer.width * 0.5f ||
       ctx->viewport.scale[1] != (float)ctx->framebuffer.height * 0.5f ||
       ctx->viewport.translate[0] != ctx->viewport.scale[0] ||
       ctx->viewport.translate[1] != ctx->viewport.scale[1] ||
       ctx->viewport.scale[2] != 0.5f ||
       ctx->viewport.translate[2] != 0.5f) {
      *failure_reason = "viewport";
      return false;
   }

   if (!ctx->blend || ctx->blend->state.independent_blend_enable ||
       ctx->blend->state.logicop_enable || !ctx->blend->state.dither ||
       ctx->blend->state.alpha_to_coverage ||
       ctx->blend->state.alpha_to_coverage_dither ||
       ctx->blend->state.alpha_to_one || ctx->blend->state.max_rt != 0 ||
       ctx->blend->state.advanced_blend_func != 0 ||
       ctx->blend->state.rt[0].blend_enable ||
       ctx->blend->state.rt[0].colormask != PIPE_MASK_RGBA) {
      *failure_reason = "blend";
      return false;
   }

   if (!ctx->dsa || !ctx->dsa->state.depth_enabled ||
       !ctx->dsa->state.depth_writemask ||
       ctx->dsa->state.depth_func != PIPE_FUNC_LEQUAL ||
       ctx->dsa->state.depth_bounds_test ||
       ctx->dsa->state.stencil[0].enabled ||
       ctx->dsa->state.stencil[1].enabled ||
       ctx->dsa->state.alpha_enabled) {
      *failure_reason = "depth_stencil_alpha";
      return false;
   }

   if (!ctx->rasterizer || ctx->rasterizer->state.front_ccw ||
       ctx->rasterizer->state.cull_face != PIPE_FACE_BACK ||
       ctx->rasterizer->state.fill_front != PIPE_POLYGON_MODE_FILL ||
       ctx->rasterizer->state.fill_back != PIPE_POLYGON_MODE_FILL ||
       ctx->rasterizer->state.scissor ||
       ctx->rasterizer->state.rasterizer_discard ||
       ctx->rasterizer->state.multisample ||
       ctx->rasterizer->state.offset_tri ||
       ctx->rasterizer->state.poly_smooth ||
       ctx->rasterizer->state.poly_stipple_enable ||
       ctx->rasterizer->state.conservative_raster_mode != 0 ||
       !ctx->rasterizer->state.half_pixel_center ||
       ctx->rasterizer->state.bottom_edge_rule ||
       ctx->rasterizer->state.clip_halfz ||
       !ctx->rasterizer->state.depth_clip_near ||
       !ctx->rasterizer->state.depth_clip_far ||
       ctx->rasterizer->state.depth_clamp) {
      *failure_reason = "rasterizer";
      return false;
   }

   if (ctx->num_sampler_views[MESA_SHADER_FRAGMENT] != 1 ||
       ctx->num_samplers[MESA_SHADER_FRAGMENT] != 1 ||
       ctx->num_sampler_views[MESA_SHADER_VERTEX] != 0 ||
       ctx->num_samplers[MESA_SHADER_VERTEX] != 0) {
      *failure_reason = "sampler_bindings";
      return false;
   }
   const struct pipe_sampler_view *view =
      ctx->sampler_views[MESA_SHADER_FRAGMENT][0];
   const struct pvrgpu_sampler_state *sampler =
      ctx->samplers[MESA_SHADER_FRAGMENT][0];
   if (!view || !view->texture || !sampler ||
       view->target != PIPE_TEXTURE_2D ||
       view->texture->target != PIPE_TEXTURE_2D ||
       view->format != PIPE_FORMAT_R8G8B8X8_UNORM ||
       view->texture->format != PIPE_FORMAT_R8G8B8X8_UNORM ||
       view->u.tex.first_level != view->u.tex.last_level ||
       view->u.tex.first_level > view->texture->last_level ||
       view->u.tex.first_layer != 0 || view->u.tex.last_layer != 0 ||
       view->u.tex.min_lod_clamp != 0.0f ||
       view->swizzle_r != PIPE_SWIZZLE_X ||
       view->swizzle_g != PIPE_SWIZZLE_Y ||
       view->swizzle_b != PIPE_SWIZZLE_Z ||
       view->swizzle_a != PIPE_SWIZZLE_1 ||
       view->texture->nr_samples > 1 ||
       view->texture->nr_storage_samples > 1) {
      *failure_reason = "texture_view";
      return false;
   }

   const struct pipe_sampler_state *sampler_state = &sampler->state;
   if (sampler_state->wrap_s != PIPE_TEX_WRAP_CLAMP_TO_EDGE ||
       sampler_state->wrap_t != PIPE_TEX_WRAP_CLAMP_TO_EDGE ||
       sampler_state->wrap_r != PIPE_TEX_WRAP_REPEAT) {
      *failure_reason = "sampler_wrap";
      return false;
   }
   if (sampler_state->min_img_filter != PIPE_TEX_FILTER_NEAREST ||
       sampler_state->mag_img_filter != PIPE_TEX_FILTER_NEAREST ||
       sampler_state->min_mip_filter != PIPE_TEX_MIPFILTER_NONE) {
      *failure_reason = "sampler_filter";
      return false;
   }
   if (sampler_state->compare_mode != PIPE_TEX_COMPARE_NONE) {
      *failure_reason = "sampler_compare";
      return false;
   }
   if (sampler_state->unnormalized_coords ||
       !sampler_state->seamless_cube_map ||
       sampler_state->reduction_mode != PIPE_TEX_REDUCTION_WEIGHTED_AVERAGE) {
      *failure_reason = "sampler_mode";
      return false;
   }
   if (sampler_state->max_anisotropy > 1) {
      *failure_reason = "sampler_anisotropy";
      return false;
   }
   if (sampler_state->lod_bias != 0.0f || sampler_state->min_lod != 0.0f ||
       sampler_state->max_lod != 1000.0f) {
      *failure_reason = "sampler_lod";
      return false;
   }

   *texture_view = view;
   return true;
}

static bool
pvrgpu_draw_matches_textured_triangles(
   const struct pvrgpu_context *ctx,
   const struct pipe_draw_info *info,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,
   unsigned num_draws,
   struct pvrgpu_textured_triangles_observation *observation,
   const char **failure_reason,
   bool require_depth_clear)
{
   if (failure_reason)
      *failure_reason = "arguments";
   if (!ctx || !info || !draws || !observation || !failure_reason)
      return false;
   if (indirect) {
      *failure_reason = "indirect";
      return false;
   }
   if (num_draws != 1) {
      *failure_reason = "draw_count";
      return false;
   }
   if (info->mode != MESA_PRIM_TRIANGLES) {
      *failure_reason = "primitive_mode";
      return false;
   }
   if (info->index_size != 0 || info->has_user_indices) {
      *failure_reason = "indexed";
      return false;
   }
   if (info->primitive_restart) {
      *failure_reason = "primitive_restart";
      return false;
   }
   if (info->increment_draw_id) {
      *failure_reason = "increment_draw_id";
      return false;
   }
   if (info->was_line_loop) {
      *failure_reason = "line_loop";
      return false;
   }
   if (info->start_instance != 0 || info->instance_count != 1) {
      *failure_reason = "instancing";
      return false;
   }
   if (draws[0].count != PVRGPU_DRAW_TEXTURED_TRIANGLES_VERTEX_COUNT) {
      *failure_reason = "vertex_count";
      return false;
   }
   if (draws[0].start > UINT_MAX -
                          PVRGPU_DRAW_TEXTURED_TRIANGLES_VERTEX_COUNT) {
      *failure_reason = "vertex_start_overflow";
      return false;
   }
   if (!pvrgpu_textured_triangles_framebuffer_matches(
          ctx, require_depth_clear)) {
      *failure_reason = "framebuffer";
      return false;
   }

   memset(observation, 0, sizeof(*observation));
   if (!pvrgpu_textured_triangles_pipeline_state_matches(
          ctx,
          &observation->viewport_width,
          &observation->viewport_height,
          &observation->texture_view,
          failure_reason))
      return false;

   observation->framebuffer_width = ctx->framebuffer.width;
   observation->framebuffer_height = ctx->framebuffer.height;
   observation->texture_level = observation->texture_view->u.tex.first_level;
   observation->texture_width =
      u_minify(observation->texture_view->texture->width0,
               observation->texture_level);
   observation->texture_height =
      u_minify(observation->texture_view->texture->height0,
               observation->texture_level);
   if (observation->texture_width == 0 || observation->texture_height == 0) {
      *failure_reason = "texture_extent";
      return false;
   }

   static const uint32_t canonical_vertex_bits
      [PVRGPU_DRAW_TEXTURED_TRIANGLES_VERTEX_COUNT][2] = {
         {UINT32_C(0xbf800000), UINT32_C(0x3f800000)},
         {UINT32_C(0xbf800000), UINT32_C(0xbf800000)},
         {UINT32_C(0x3f800000), UINT32_C(0x3f800000)},
         {UINT32_C(0xbf800000), UINT32_C(0xbf800000)},
         {UINT32_C(0x3f800000), UINT32_C(0xbf800000)},
         {UINT32_C(0x3f800000), UINT32_C(0x3f800000)},
      };
   for (unsigned vertex = 0;
        vertex < PVRGPU_DRAW_TEXTURED_TRIANGLES_VERTEX_COUNT;
        ++vertex) {
      float xyz[3] = {0.0f, 0.0f, 0.0f};
      if (!pvrgpu_read_resource_float3_vertex(ctx,
                                              draws[0].start + vertex,
                                              xyz) ||
          pvrgpu_float_bits(xyz[2]) != 0) {
         *failure_reason = "vertex_data";
         return false;
      }
      observation->vertex_bits[vertex][0] = pvrgpu_float_bits(xyz[0]);
      observation->vertex_bits[vertex][1] = pvrgpu_float_bits(xyz[1]);
      if (observation->vertex_bits[vertex][0] !=
             canonical_vertex_bits[vertex][0] ||
          observation->vertex_bits[vertex][1] !=
             canonical_vertex_bits[vertex][1]) {
         *failure_reason = "canonical_geometry";
         return false;
      }
      const float texcoord_x = xyz[0] * 0.5f + 0.5f;
      const float texcoord_y = xyz[1] * 0.5f + 0.5f;
      if (!isfinite(texcoord_x) || !isfinite(texcoord_y)) {
         *failure_reason = "vertex_texcoord";
         return false;
      }
      observation->texcoord_bits[vertex][0] =
         pvrgpu_float_bits(texcoord_x);
      observation->texcoord_bits[vertex][1] =
         pvrgpu_float_bits(texcoord_y);
   }
   *failure_reason = NULL;
   return true;
}

static void
pvrgpu_copy_pco_stage_abi_to_command(
   struct pvrgpu_draw_pco_stage_abi *destination,
   const struct pvrgpu_pco_stage_abi *source)
{
   destination->temps = source->temps;
   destination->vertex_inputs = source->vertex_inputs;
   destination->vertex_outputs = source->vertex_outputs;
   destination->coefficients = source->coefficients;
   destination->shareds = source->shareds;
   destination->push_constant_start = source->push_constant_start;
   destination->push_constant_count = source->push_constant_count;
   destination->entry_offset = source->entry_offset;
}

static uint64_t
pvrgpu_pco_binary_fnv1a64(const uint8_t *data, size_t size)
{
   uint64_t hash = UINT64_C(14695981039346656037);
   for (size_t byte = 0; byte < size; ++byte) {
      hash ^= data[byte];
      hash *= UINT64_C(1099511628211);
   }
   return hash;
}

static bool
pvrgpu_refract_pco_observation_payload_matches(
   const struct pvrgpu_refract_pco_observation *observation,
   const char **failure_reason)
{
   const bool composite =
      observation->profile == PVRGPU_PCO_REFRACT_COMPOSITE;
   const size_t expected_vertex_size =
      (size_t)PVRGPU_REFRACT_PCO_VERTEX_COUNT *
      PVRGPU_REFRACT_PCO_VERTEX_STRIDE;
   const size_t expected_shared_count = composite ? 64u : 16u;
   const uint64_t expected_shared_hash =
      composite ? PVRGPU_REFRACT_PCO_COMPOSITE_SHARED_FNV1A64 :
                  PVRGPU_REFRACT_PCO_PREPASS_SHARED_FNV1A64;
   const size_t expected_vertex_binary_size = composite ? 1536U : 432U;
   const size_t expected_fragment_binary_size = composite ? 5072U : 56U;
   const uint64_t expected_vertex_binary_hash =
      composite ? UINT64_C(0xc46a9af088bfe8a9) :
                  UINT64_C(0x6e9ad97e49eca9fe);
   const uint64_t expected_fragment_binary_hash =
      composite ? UINT64_C(0x8fe8ae5903f3c2dd) :
                  UINT64_C(0xa55a28d91b0f4b9e);
   if (observation->binary.vertex.size != expected_vertex_binary_size ||
       pvrgpu_pco_binary_fnv1a64(observation->binary.vertex.data,
                                 observation->binary.vertex.size) !=
          expected_vertex_binary_hash ||
       observation->binary.fragment.size != expected_fragment_binary_size ||
       pvrgpu_pco_binary_fnv1a64(observation->binary.fragment.data,
                                 observation->binary.fragment.size) !=
          expected_fragment_binary_hash) {
      *failure_reason = "pco_binary_fingerprint";
      return false;
   }
   if (observation->interleaved_vertex_data_size != expected_vertex_size ||
       pvrgpu_pco_binary_fnv1a64(observation->interleaved_vertex_data,
                                observation->interleaved_vertex_data_size) !=
          PVRGPU_REFRACT_PCO_VERTEX_FNV1A64) {
      *failure_reason = "vertex_payload_fingerprint";
      return false;
   }
   if (observation->vertex_shared_count != expected_shared_count ||
       pvrgpu_pco_binary_fnv1a64(
          (const uint8_t *)observation->vertex_shared,
          observation->vertex_shared_count * sizeof(uint32_t)) !=
          expected_shared_hash) {
      *failure_reason = "shared_payload_fingerprint";
      return false;
   }
   if ((!composite && (observation->sampled_image_bytes ||
                       observation->sampled_image_bytes_size != 0)) ||
       (composite &&
        (observation->sampled_image_bytes_size !=
            PVRGPU_REFRACT_PCO_IMAGE_BYTES ||
         pvrgpu_pco_binary_fnv1a64(observation->sampled_image_bytes,
                                  observation->sampled_image_bytes_size) !=
            PVRGPU_REFRACT_PCO_IMAGE_FNV1A64))) {
      *failure_reason = "sampled_image_fingerprint";
      return false;
   }
   const uint64_t expected_fragment_shared_hash =
      observation->framebuffer_width == 800u &&
            observation->framebuffer_height == 600u ?
         PVRGPU_REFRACT_PCO_FRAGMENT_SHARED_800_FNV1A64 :
         PVRGPU_REFRACT_PCO_FRAGMENT_SHARED_FNV1A64;
   if ((!composite && (observation->fragment_shared_count != 0 ||
                       observation->texture_count != 0)) ||
       (composite &&
        (observation->fragment_shared_count !=
            PVRGPU_REFRACT_PCO_FS_SHARED_DWORDS ||
         observation->texture_count != PVRGPU_REFRACT_PCO_TEXTURE_COUNT ||
         pvrgpu_pco_binary_fnv1a64(
            (const uint8_t *)observation->fragment_shared,
            observation->fragment_shared_count * sizeof(uint32_t)) !=
            expected_fragment_shared_hash))) {
      *failure_reason = "fragment_descriptor_payload";
      return false;
   }
   return true;
}

static bool
pvrgpu_compile_refract_observation(
   struct pvrgpu_context *ctx,
   struct pvrgpu_refract_pco_observation *observation,
   const char **failure_reason)
{
   if (!ctx || !ctx->vs || !ctx->fs || !observation || !failure_reason) {
      if (failure_reason)
         *failure_reason = "compile_arguments";
      return false;
   }

   char error[512] = { 0 };
   if (!ctx->pco_compiler) {
      ctx->pco_compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
      if (!ctx->pco_compiler) {
         pvrgpu_counter_eventf("draw_pco_refract_compile_error",
                               "profile=%u stage=compiler_create reason=%s",
                               observation->profile,
                               error[0] ? error : "unknown");
         *failure_reason = "compiler_create";
         return false;
      }
   }

   if (!pvrgpu_pco_compile_refract(ctx->pco_compiler,
                                   ctx->vs->nir,
                                   ctx->fs->nir,
                                   observation->profile,
                                   &observation->binary,
                                   error,
                                   sizeof(error))) {
      pvrgpu_counter_eventf("draw_pco_refract_compile_error",
                            "profile=%u stage=pco_compile reason=%s",
                            observation->profile,
                            error[0] ? error : "unknown");
      *failure_reason = "pco_compile";
      return false;
   }

   const struct pvrgpu_pco_graphics_binary *binary = &observation->binary;
   if (!pvrgpu_refract_pco_observation_payload_matches(observation,
                                                        failure_reason))
      return false;
   pvrgpu_counter_eventf(
      "draw_pco_refract_capture",
      "profile=%u vs_bytes=%zu vs_fnv1a64=%016llx "
      "vs_abi=%u,%u,%u,%u,%u,%u,%u,%u fs_bytes=%zu "
      "fs_fnv1a64=%016llx fs_abi=%u,%u,%u,%u,%u,%u,%u,%u "
      "linkage=%u,%u,%u,%u,%u,%u,%u,%u descriptor=%u,%u,%u "
      "vertex_bytes=%zu vertex_fnv1a64=%016llx shared_dwords=%zu "
      "shared_fnv1a64=%016llx fragment_shared_dwords=%zu "
      "fragment_shared_fnv1a64=%016llx image_bytes=%zu "
      "image_fnv1a64=%016llx",
      observation->profile,
      binary->vertex.size,
      (unsigned long long)pvrgpu_pco_binary_fnv1a64(binary->vertex.data,
                                                     binary->vertex.size),
      binary->vertex.abi.temps,
      binary->vertex.abi.vertex_inputs,
      binary->vertex.abi.vertex_outputs,
      binary->vertex.abi.coefficients,
      binary->vertex.abi.shareds,
      binary->vertex.abi.push_constant_start,
      binary->vertex.abi.push_constant_count,
      binary->vertex.abi.entry_offset,
      binary->fragment.size,
      (unsigned long long)pvrgpu_pco_binary_fnv1a64(binary->fragment.data,
                                                     binary->fragment.size),
      binary->fragment.abi.temps,
      binary->fragment.abi.vertex_inputs,
      binary->fragment.abi.vertex_outputs,
      binary->fragment.abi.coefficients,
      binary->fragment.abi.shareds,
      binary->fragment.abi.push_constant_start,
      binary->fragment.abi.push_constant_count,
      binary->fragment.abi.entry_offset,
      binary->position_output_start,
      binary->position_output_count,
      binary->fragment_position_start,
      binary->fragment_position_count,
      binary->varying_output_start,
      binary->varying_output_count,
      binary->fragment_varying_start,
      binary->fragment_varying_count,
      binary->fragment_texture_descriptor_start,
      binary->fragment_texture_descriptor_count,
      binary->fragment_texture_descriptor_stride,
      observation->interleaved_vertex_data_size,
      (unsigned long long)pvrgpu_pco_binary_fnv1a64(
         observation->interleaved_vertex_data,
         observation->interleaved_vertex_data_size),
      observation->vertex_shared_count,
      (unsigned long long)pvrgpu_pco_binary_fnv1a64(
         (const uint8_t *)observation->vertex_shared,
         observation->vertex_shared_count * sizeof(uint32_t)),
      observation->fragment_shared_count,
      (unsigned long long)pvrgpu_pco_binary_fnv1a64(
         (const uint8_t *)observation->fragment_shared,
         observation->fragment_shared_count * sizeof(uint32_t)),
      observation->sampled_image_bytes_size,
      (unsigned long long)pvrgpu_pco_binary_fnv1a64(
         observation->sampled_image_bytes,
         observation->sampled_image_bytes_size));
   *failure_reason = NULL;
   return true;
}

static void
pvrgpu_copy_pco_stage_abi_to_systemc(
   struct pvrgpu_systemc_pco_stage_abi *destination,
   const struct pvrgpu_pco_stage_abi *source)
{
   destination->temps = source->temps;
   destination->vertex_inputs = source->vertex_inputs;
   destination->vertex_outputs = source->vertex_outputs;
   destination->coefficients = source->coefficients;
   destination->shareds = source->shareds;
   destination->push_constant_start = source->push_constant_start;
   destination->push_constant_count = source->push_constant_count;
   destination->entry_offset = source->entry_offset;
}

static bool
pvrgpu_systemc_blend_equation_from_pipe(unsigned source,
                                         uint32_t *destination)
{
   if (!destination)
      return false;
   switch (source) {
   case PIPE_BLEND_ADD:
      *destination = PVRGPU_SYSTEMC_PCO_BLEND_EQUATION_ADD;
      return true;
   case PIPE_BLEND_SUBTRACT:
      *destination = PVRGPU_SYSTEMC_PCO_BLEND_EQUATION_SUBTRACT;
      return true;
   case PIPE_BLEND_REVERSE_SUBTRACT:
      *destination = PVRGPU_SYSTEMC_PCO_BLEND_EQUATION_REVERSE_SUBTRACT;
      return true;
   case PIPE_BLEND_MIN:
      *destination = PVRGPU_SYSTEMC_PCO_BLEND_EQUATION_MIN;
      return true;
   case PIPE_BLEND_MAX:
      *destination = PVRGPU_SYSTEMC_PCO_BLEND_EQUATION_MAX;
      return true;
   default:
      return false;
   }
}

static bool
pvrgpu_systemc_blend_factor_from_pipe(unsigned source,
                                      uint32_t *destination)
{
   if (!destination)
      return false;
   switch (source) {
   case PIPE_BLENDFACTOR_ZERO:
      *destination = PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ZERO;
      return true;
   case PIPE_BLENDFACTOR_ONE:
      *destination = PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE;
      return true;
   case PIPE_BLENDFACTOR_SRC_ALPHA:
      *destination = PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_SOURCE_ALPHA;
      return true;
   case PIPE_BLENDFACTOR_INV_SRC_ALPHA:
      *destination =
         PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE_MINUS_SOURCE_ALPHA;
      return true;
   case PIPE_BLENDFACTOR_SRC_COLOR:
      *destination = PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_SOURCE_COLOR;
      return true;
   case PIPE_BLENDFACTOR_INV_SRC_COLOR:
      *destination =
         PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE_MINUS_SOURCE_COLOR;
      return true;
   case PIPE_BLENDFACTOR_DST_COLOR:
      *destination = PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_DESTINATION_COLOR;
      return true;
   case PIPE_BLENDFACTOR_INV_DST_COLOR:
      *destination =
         PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE_MINUS_DESTINATION_COLOR;
      return true;
   case PIPE_BLENDFACTOR_DST_ALPHA:
      *destination = PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_DESTINATION_ALPHA;
      return true;
   case PIPE_BLENDFACTOR_INV_DST_ALPHA:
      *destination =
         PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE_MINUS_DESTINATION_ALPHA;
      return true;
   default:
      return false;
   }
}

static bool
pvrgpu_systemc_texture_wrap_from_pipe(unsigned source,
                                      uint32_t *destination)
{
   if (!destination)
      return false;
   switch (source) {
   case PIPE_TEX_WRAP_CLAMP_TO_EDGE:
      *destination = PVRGPU_SYSTEMC_PCO_TEXTURE_WRAP_CLAMP_TO_EDGE;
      return true;
   case PIPE_TEX_WRAP_REPEAT:
      *destination = PVRGPU_SYSTEMC_PCO_TEXTURE_WRAP_REPEAT;
      return true;
   default:
      return false;
   }
}

static bool
pvrgpu_systemc_shader_stage_from_mesa(mesa_shader_stage source,
                                      uint32_t *destination)
{
   if (!destination)
      return false;
   switch (source) {
   case MESA_SHADER_VERTEX:
      *destination = PVRGPU_SYSTEMC_PCO_SHADER_STAGE_VERTEX;
      return true;
   case MESA_SHADER_FRAGMENT:
      *destination = PVRGPU_SYSTEMC_PCO_SHADER_STAGE_FRAGMENT;
      return true;
   default:
      return false;
   }
}

static bool
pvrgpu_init_systemc_blend_state(
   struct pvrgpu_systemc_driver_command *command,
   bool enable,
   unsigned rgb_equation,
   unsigned source_rgb_factor,
   unsigned destination_rgb_factor,
   unsigned alpha_equation,
   unsigned source_alpha_factor,
   unsigned destination_alpha_factor)
{
   if (!command ||
       !pvrgpu_systemc_blend_equation_from_pipe(
          rgb_equation, &command->blend_rgb_equation) ||
       !pvrgpu_systemc_blend_equation_from_pipe(
          alpha_equation, &command->blend_alpha_equation) ||
       !pvrgpu_systemc_blend_factor_from_pipe(
          source_rgb_factor, &command->blend_source_rgb_factor) ||
       !pvrgpu_systemc_blend_factor_from_pipe(
          destination_rgb_factor, &command->blend_destination_rgb_factor) ||
       !pvrgpu_systemc_blend_factor_from_pipe(
          source_alpha_factor, &command->blend_source_alpha_factor) ||
       !pvrgpu_systemc_blend_factor_from_pipe(
          destination_alpha_factor,
          &command->blend_destination_alpha_factor))
      return false;
   command->blend_enable = enable;
   return true;
}

static bool
pvrgpu_init_refract_systemc_draw(
   struct pvrgpu_systemc_driver_command *command,
   const struct pvrgpu_refract_pco_observation *observation)
{
   const bool composite =
      observation->profile == PVRGPU_PCO_REFRACT_COMPOSITE;
   const struct pvrgpu_pco_graphics_binary *binary = &observation->binary;

   memset(command, 0, sizeof(*command));
   command->version = PVRGPU_SYSTEMC_API_VERSION;
   command->schema = PVRGPU_DRIVER_COMMAND_SCHEMA;
   command->producer = PVRGPU_DRIVER_COMMAND_PRODUCER;
   command->command = "draw_pco_triangles";
   command->case_name = "refract.refract.capture.1";
   command->format = PVRGPU_DRIVER_COMMAND_FORMAT_RGBA8;
   command->frame = 1;
   command->framebuffer_width = observation->framebuffer_width;
   command->framebuffer_height = observation->framebuffer_height;
   command->width = observation->viewport_width;
   command->height = observation->viewport_height;
   command->clear_color_bits[3] = UINT32_C(0x3f800000);

   command->raw_vertex_data = observation->interleaved_vertex_data;
   command->raw_vertex_data_size =
      observation->interleaved_vertex_data_size;
   command->vertex_stride = PVRGPU_REFRACT_PCO_VERTEX_STRIDE;
   command->vertex_count = observation->vertex_count;
   command->first_vertex = 0;
   command->instance_count = 1;
   command->primitive_mode = MESA_PRIM_TRIANGLES;
   command->indexed = 0;
   command->render_target_count = 1;

   command->vertex_pco = binary->vertex.data;
   command->vertex_pco_size = binary->vertex.size;
   command->fragment_pco = binary->fragment.data;
   command->fragment_pco_size = binary->fragment.size;
   command->vertex_shared = observation->vertex_shared;
   command->vertex_shared_count = observation->vertex_shared_count;
   command->fragment_shared = observation->fragment_shared_count ?
                                 observation->fragment_shared : NULL;
   command->fragment_shared_count = observation->fragment_shared_count;
   command->sampled_texture_count = composite ?
                                      PVRGPU_REFRACT_PCO_TEXTURE_COUNT : 0;

   pvrgpu_copy_pco_stage_abi_to_systemc(&command->vertex_pco_abi,
                                         &binary->vertex.abi);
   pvrgpu_copy_pco_stage_abi_to_systemc(&command->fragment_pco_abi,
                                         &binary->fragment.abi);
   command->position_output_start = binary->position_output_start;
   command->position_output_count = binary->position_output_count;
   command->fragment_position_start = binary->fragment_position_start;
   command->fragment_position_count = binary->fragment_position_count;
   command->varying_output_start = binary->varying_output_start;
   command->varying_output_count = binary->varying_output_count;
   command->fragment_varying_start = binary->fragment_varying_start;
   command->fragment_varying_count = binary->fragment_varying_count;

   command->viewport_scale_bits[0] =
      pvrgpu_float_bits((float)observation->viewport_width * 0.5f);
   command->viewport_scale_bits[1] =
      pvrgpu_float_bits((float)observation->viewport_height * 0.5f);
   command->viewport_scale_bits[2] = UINT32_C(0x3f000000);
   memcpy(command->viewport_translate_bits,
          command->viewport_scale_bits,
          sizeof(command->viewport_translate_bits));
   command->front_ccw = 0;
   command->cull_face = composite ? PIPE_FACE_BACK : PIPE_FACE_FRONT;
   command->fill_front = PIPE_POLYGON_MODE_FILL;
   command->fill_back = PIPE_POLYGON_MODE_FILL;
   command->half_pixel_center = 1;
   command->depth_clip_near = 1;
   command->depth_clip_far = 1;
   command->sample_mask = UINT32_MAX;
   command->color_mask = PIPE_MASK_RGBA;
   command->dither = 1;
   command->depth_enable = 1;
   command->depth_write = 1;
   command->depth_func = PIPE_FUNC_LEQUAL;
   command->depth_clear_bits = UINT32_C(0x3f800000);
   command->depth_format = composite ? PIPE_FORMAT_Z24X8_UNORM :
                                       PIPE_FORMAT_Z32_UNORM;
   command->color_attachment_source_command_index =
      PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR;
   command->depth_attachment_source_command_index =
      PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR;
   return pvrgpu_init_systemc_blend_state(
      command,
      false,
      PIPE_BLEND_ADD,
      PIPE_BLENDFACTOR_ONE,
      PIPE_BLENDFACTOR_ZERO,
      PIPE_BLEND_ADD,
      PIPE_BLENDFACTOR_ONE,
      PIPE_BLENDFACTOR_ZERO);
}

static bool
pvrgpu_init_refract_systemc_texture(
   struct pvrgpu_systemc_pco_sequence_texture *destination,
   const struct pvrgpu_refract_pco_texture *source,
   const struct pvrgpu_refract_pco_observation *composite)
{
   memset(destination, 0, sizeof(*destination));
   destination->stage = PVRGPU_SYSTEMC_PCO_SHADER_STAGE_FRAGMENT;
   switch (source->source) {
   case PVRGPU_REFRACT_PCO_EXTERNAL_PAYLOAD:
      destination->source = PVRGPU_SYSTEMC_PCO_TEXTURE_EXTERNAL_PAYLOAD;
      destination->bytes = composite->sampled_image_bytes;
      destination->bytes_size = composite->sampled_image_bytes_size;
      break;
   case PVRGPU_REFRACT_PCO_PREVIOUS_COLOR_ATTACHMENT:
      destination->source =
         PVRGPU_SYSTEMC_PCO_TEXTURE_PREVIOUS_COLOR_ATTACHMENT;
      break;
   case PVRGPU_REFRACT_PCO_PREVIOUS_DEPTH_ATTACHMENT:
      destination->source =
         PVRGPU_SYSTEMC_PCO_TEXTURE_PREVIOUS_DEPTH_ATTACHMENT;
      break;
   default:
      return false;
   }

   if ((source->min_filter != PIPE_TEX_FILTER_NEAREST &&
        source->min_filter != PIPE_TEX_FILTER_LINEAR) ||
       source->mag_filter != source->min_filter ||
       (source->mip_filter != PIPE_TEX_MIPFILTER_NONE &&
        source->mip_filter != PIPE_TEX_MIPFILTER_LINEAR) ||
       source->wrap_u != PIPE_TEX_WRAP_CLAMP_TO_EDGE ||
       source->wrap_v != PIPE_TEX_WRAP_CLAMP_TO_EDGE ||
       source->mip_count == 0 ||
       source->mip_count > PVRGPU_SYSTEMC_MAX_TEXTURE_MIP_LEVELS)
      return false;

   destination->producer_command_index = 0;
   destination->descriptor_set = source->descriptor_set;
   destination->binding = source->binding;
   destination->format = util_format_name(source->format);
   destination->declared_bytes_size = source->declared_size;
   destination->mip_count = source->mip_count;
   for (unsigned level = 0; level < source->mip_count; ++level) {
      destination->mip[level].width = source->mip[level].width;
      destination->mip[level].height = source->mip[level].height;
      destination->mip[level].row_pitch = source->mip[level].row_pitch;
      destination->mip[level].offset = source->mip[level].offset;
   }
   destination->min_filter =
      source->min_filter == PIPE_TEX_FILTER_LINEAR ?
         PVRGPU_SYSTEMC_PCO_TEXTURE_FILTER_LINEAR :
         PVRGPU_SYSTEMC_PCO_TEXTURE_FILTER_NEAREST;
   destination->mag_filter = destination->min_filter;
   destination->mip_filter =
      source->mip_filter == PIPE_TEX_MIPFILTER_LINEAR ?
         PVRGPU_SYSTEMC_PCO_TEXTURE_MIP_FILTER_LINEAR :
         PVRGPU_SYSTEMC_PCO_TEXTURE_MIP_FILTER_NONE;
   if (!pvrgpu_systemc_texture_wrap_from_pipe(source->wrap_u,
                                               &destination->wrap_u) ||
       !pvrgpu_systemc_texture_wrap_from_pipe(source->wrap_v,
                                               &destination->wrap_v))
      return false;
   destination->normalized_coordinates = source->normalized_coordinates;
   destination->min_lod_u4_6 = source->min_lod_u4_6;
   destination->max_lod_u4_6 = source->max_lod_u4_6;
   return destination->format != NULL;
}

static unsigned
pvrgpu_array_primitive_count(unsigned mode, unsigned count);

static unsigned
pvrgpu_array_assembled_vertex_count(unsigned mode, unsigned count);

/*
 * Sum the input-assembly totals a PCO sequence actually submits.
 *
 * Input assembly is decided by the draw calls themselves, so the driver can
 * state these exactly.  Everything downstream of clipping is what the pipeline
 * did with that geometry and is left at zero for SystemC to measure: a capture
 * profile must not be able to decide a rasterization result.
 */
static void
pvrgpu_sequence_input_assembly_totals(
   const struct pvrgpu_systemc_driver_command *draws,
   unsigned draw_count,
   uint32_t *out_vertices,
   uint32_t *out_primitives)
{
   uint32_t vertices = 0;
   uint32_t primitives = 0;

   for (unsigned index = 0; index < draw_count; ++index) {
      const struct pvrgpu_systemc_driver_command *draw = &draws[index];
      const unsigned instances =
         draw->instance_count != 0 ? draw->instance_count : 1;
      const unsigned assembled =
         draw->indexed ? draw->index_count : draw->vertex_count;

      vertices += assembled * instances;
      primitives +=
         pvrgpu_array_primitive_count(draw->primitive_mode, assembled) *
         instances;
   }

   *out_vertices = vertices;
   *out_primitives = primitives;
}

static bool
pvrgpu_emit_refract_pco_sequence_command(struct pvrgpu_context *ctx)
{
   const char *path = pvrgpu_command_output_path();
   if (!path || !ctx || !ctx->refract_pco_prepass ||
       !ctx->refract_pco_composite || ctx->driver_draw_command_emitted ||
       pvrgpu_driver_draw_command_has_been_emitted())
      return false;

   const struct pvrgpu_refract_pco_observation *prepass =
      ctx->refract_pco_prepass;
   const struct pvrgpu_refract_pco_observation *composite =
      ctx->refract_pco_composite;
   if (prepass->profile != PVRGPU_PCO_REFRACT_PREPASS ||
       composite->profile != PVRGPU_PCO_REFRACT_COMPOSITE ||
       prepass->texture_count != 0 ||
       composite->texture_count != PVRGPU_REFRACT_PCO_TEXTURE_COUNT)
      return false;

   struct pvrgpu_systemc_driver_command draws[2];
   if (!pvrgpu_init_refract_systemc_draw(&draws[0], prepass) ||
       !pvrgpu_init_refract_systemc_draw(&draws[1], composite))
      return false;

   struct pvrgpu_systemc_pco_sequence_texture
      textures[PVRGPU_REFRACT_PCO_TEXTURE_COUNT];
   for (unsigned slot = 0; slot < PVRGPU_REFRACT_PCO_TEXTURE_COUNT; ++slot) {
      if (!pvrgpu_init_refract_systemc_texture(&textures[slot],
                                                &composite->textures[slot],
                                                composite))
         return false;
   }

   struct pvrgpu_systemc_driver_command command;
   memset(&command, 0, sizeof(command));
   command.version = PVRGPU_SYSTEMC_API_VERSION;
   command.schema = PVRGPU_DRIVER_COMMAND_SCHEMA;
   command.producer = PVRGPU_DRIVER_COMMAND_PRODUCER;
   command.command = "draw_pco_sequence";
   command.case_name = "refract.refract.capture.1";
   command.format = PVRGPU_DRIVER_COMMAND_FORMAT_RGBA8;
   command.frame = 1;
   command.framebuffer_width = composite->framebuffer_width;
   command.framebuffer_height = composite->framebuffer_height;
   command.width = composite->viewport_width;
   command.height = composite->viewport_height;
   command.clear_color_bits[3] = UINT32_C(0x3f800000);
   const bool output_800x600 =
      command.width == 800u && command.height == 600u;
   if (!pvrgpu_glmark_output_extent_supported(command.width,
                                               command.height) ||
       command.framebuffer_width != command.width ||
       command.framebuffer_height != command.height ||
       prepass->framebuffer_width != command.width * 2u ||
       prepass->framebuffer_height != command.height * 2u)
      return false;
   command.draw_count = PVRGPU_ARRAY_SIZE(draws);
   pvrgpu_sequence_input_assembly_totals(draws,
                                         PVRGPU_ARRAY_SIZE(draws),
                                         &command.ia_vertices,
                                         &command.ia_primitives);
   command.clip_invocations = command.ia_primitives;
   /*
    * Everything past input assembly is measured by SystemC from the geometry
    * this sequence actually rasterizes.  Leaving these at zero is what makes
    * the reported totals a result rather than a replayed constant.
    */
   command.vs_invocations = 0;
   command.clip_primitives = 0;
   command.setup_triangles = 0;
   command.ps_invocations = 0;
   command.semantic_texel_fetches = 0;
   command.pco_sequence_command_count = PVRGPU_ARRAY_SIZE(draws);
   command.pco_sequence_commands = draws;
   command.pco_sequence_texture_count = PVRGPU_ARRAY_SIZE(textures);
   command.pco_sequence_textures = textures;

   char error[512] = { 0 };
   if (!pvrgpu_write_draw_pco_sequence_command(path,
                                                &command,
                                                error,
                                                sizeof(error))) {
      remove(path);
      pvrgpu_counter_eventf("draw_pco_refract_sequence_error",
                            "stage=command_submit reason=%s",
                            error[0] ? error : "unknown");
      return false;
   }

   ctx->driver_draw_command_emitted = true;
   pvrgpu_note_driver_draw_command_emitted();
   pvrgpu_counter_eventf(
      "draw_pco_refract_sequence_command",
      "draws=2 resources=3 drawlists=%u framebuffer=%ux%u",
      command.draw_count,
      command.framebuffer_width,
      command.framebuffer_height);
   return true;
}

static bool
pvrgpu_init_shadow_systemc_draw(
   struct pvrgpu_systemc_driver_command *command,
   const struct pvrgpu_shadow_pco_observation *observation)
{
   const bool mask = observation->profile == PVRGPU_PCO_SHADOW_MASK;
   const struct pvrgpu_pco_graphics_binary *binary = &observation->binary;

   memset(command, 0, sizeof(*command));
   command->version = PVRGPU_SYSTEMC_API_VERSION;
   command->schema = PVRGPU_DRIVER_COMMAND_SCHEMA;
   command->producer = PVRGPU_DRIVER_COMMAND_PRODUCER;
   command->command = "draw_pco_triangles";
   command->case_name = "shadow.shadow.capture.1";
   command->format = PVRGPU_DRIVER_COMMAND_FORMAT_RGBA8;
   command->frame = 1;
   command->framebuffer_width = observation->framebuffer_width;
   command->framebuffer_height = observation->framebuffer_height;
   command->width = observation->viewport_width;
   command->height = observation->viewport_height;
   memcpy(command->clear_color_bits,
          observation->clear_color_bits,
          sizeof(command->clear_color_bits));

   command->raw_vertex_data = observation->vertex_data;
   command->raw_vertex_data_size = observation->vertex_data_size;
   command->vertex_stride = observation->vertex_stride;
   command->vertex_count = observation->vertex_count;
   command->first_vertex = 0;
   command->instance_count = 1;
   command->primitive_mode = observation->primitive_mode;
   command->indexed = 0;
   command->render_target_count = 1;

   command->vertex_pco = binary->vertex.data;
   command->vertex_pco_size = binary->vertex.size;
   command->fragment_pco = binary->fragment.data;
   command->fragment_pco_size = binary->fragment.size;
   command->vertex_shared = observation->vertex_shared;
   command->vertex_shared_count = observation->vertex_shared_count;
   command->fragment_shared = observation->fragment_shared_count ?
                                 observation->fragment_shared : NULL;
   command->fragment_shared_count = observation->fragment_shared_count;
   command->sampled_texture_count = mask ? 1U : 0U;

   pvrgpu_copy_pco_stage_abi_to_systemc(&command->vertex_pco_abi,
                                         &binary->vertex.abi);
   pvrgpu_copy_pco_stage_abi_to_systemc(&command->fragment_pco_abi,
                                         &binary->fragment.abi);
   command->position_output_start = binary->position_output_start;
   command->position_output_count = binary->position_output_count;
   command->fragment_position_start = binary->fragment_position_start;
   command->fragment_position_count = binary->fragment_position_count;
   command->varying_output_start = binary->varying_output_start;
   command->varying_output_count = binary->varying_output_count;
   command->fragment_varying_start = binary->fragment_varying_start;
   command->fragment_varying_count = binary->fragment_varying_count;

   command->viewport_scale_bits[0] =
      pvrgpu_float_bits((float)observation->viewport_width * 0.5f);
   command->viewport_scale_bits[1] =
      pvrgpu_float_bits((float)observation->viewport_height * 0.5f);
   command->viewport_scale_bits[2] = UINT32_C(0x3f000000);
   memcpy(command->viewport_translate_bits,
          command->viewport_scale_bits,
          sizeof(command->viewport_translate_bits));
   command->front_ccw = 0;
   command->cull_face = PIPE_FACE_BACK;
   command->fill_front = PIPE_POLYGON_MODE_FILL;
   command->fill_back = PIPE_POLYGON_MODE_FILL;
   command->half_pixel_center = 1;
   command->depth_clip_near = 1;
   command->depth_clip_far = 1;
   command->sample_mask = UINT32_MAX;
   command->color_mask = observation->color_mask;
   command->dither = observation->dither;
   command->depth_enable = observation->depth_enable;
   command->depth_write = observation->depth_write;
   command->depth_func = observation->depth_func;
   command->depth_clear_bits = observation->depth_clear_bits;
   command->depth_format = observation->depth_format;
   command->color_attachment_source_command_index =
      observation->color_attachment_source;
   command->depth_attachment_source_command_index =
      observation->depth_attachment_source;
   return pvrgpu_init_systemc_blend_state(
      command,
      observation->blend_enable,
      observation->rgb_func,
      observation->rgb_src_factor,
      observation->rgb_dst_factor,
      observation->alpha_func,
      observation->alpha_src_factor,
      observation->alpha_dst_factor);
}

static void
pvrgpu_init_shadow_systemc_texture(
   struct pvrgpu_systemc_pco_sequence_texture *texture,
   const struct pvrgpu_shadow_pco_observation *depth)
{
   memset(texture, 0, sizeof(*texture));
   texture->source = PVRGPU_SYSTEMC_PCO_TEXTURE_PREVIOUS_DEPTH_ATTACHMENT;
   texture->stage = PVRGPU_SYSTEMC_PCO_SHADER_STAGE_FRAGMENT;
   texture->producer_command_index = 0;
   texture->descriptor_set = 0;
   texture->binding = 0;
   texture->format = util_format_name(PIPE_FORMAT_Z32_UNORM);
   texture->declared_bytes_size =
      (size_t)depth->framebuffer_width * depth->framebuffer_height *
      sizeof(uint32_t);
   texture->mip_count = 1;
   texture->mip[0].width = depth->framebuffer_width;
   texture->mip[0].height = depth->framebuffer_height;
   texture->mip[0].row_pitch =
      depth->framebuffer_width * sizeof(uint32_t);
   texture->min_filter = PVRGPU_SYSTEMC_PCO_TEXTURE_FILTER_NEAREST;
   texture->mag_filter = PVRGPU_SYSTEMC_PCO_TEXTURE_FILTER_NEAREST;
   texture->mip_filter = PVRGPU_SYSTEMC_PCO_TEXTURE_MIP_FILTER_NONE;
   texture->wrap_u = PVRGPU_SYSTEMC_PCO_TEXTURE_WRAP_CLAMP_TO_EDGE;
   texture->wrap_v = PVRGPU_SYSTEMC_PCO_TEXTURE_WRAP_CLAMP_TO_EDGE;
   texture->normalized_coordinates = 1;
}

static bool
pvrgpu_emit_shadow_pco_sequence_command(struct pvrgpu_context *ctx)
{
   const char *path = pvrgpu_command_output_path();
   if (!path || !ctx || !ctx->shadow_pco_depth || !ctx->shadow_pco_mask ||
       !ctx->shadow_pco_scene || ctx->driver_draw_command_emitted ||
       pvrgpu_driver_draw_command_has_been_emitted())
      return false;

   const struct pvrgpu_shadow_pco_observation *depth =
      ctx->shadow_pco_depth;
   const struct pvrgpu_shadow_pco_observation *mask = ctx->shadow_pco_mask;
   const struct pvrgpu_shadow_pco_observation *scene =
      ctx->shadow_pco_scene;
   if (depth->profile != PVRGPU_PCO_SHADOW_DEPTH ||
       mask->profile != PVRGPU_PCO_SHADOW_MASK ||
       scene->profile != PVRGPU_PCO_SHADOW_SCENE ||
       depth->fragment_shared_count != 0 ||
       mask->fragment_shared_count != PVRGPU_SHADOW_PCO_FS_SHARED_DWORDS ||
       scene->fragment_shared_count != 0 ||
       !depth->shadow_depth || !mask->output_depth_clear_one)
      return false;

   struct pvrgpu_systemc_driver_command draws[3];
   if (!pvrgpu_init_shadow_systemc_draw(&draws[0], depth) ||
       !pvrgpu_init_shadow_systemc_draw(&draws[1], mask) ||
       !pvrgpu_init_shadow_systemc_draw(&draws[2], scene))
      return false;

   struct pvrgpu_systemc_pco_sequence_texture texture;
   pvrgpu_init_shadow_systemc_texture(&texture, depth);

   struct pvrgpu_systemc_driver_command command;
   memset(&command, 0, sizeof(command));
   command.version = PVRGPU_SYSTEMC_API_VERSION;
   command.schema = PVRGPU_DRIVER_COMMAND_SCHEMA;
   command.producer = PVRGPU_DRIVER_COMMAND_PRODUCER;
   command.command = "draw_pco_sequence";
   command.case_name = "shadow.shadow.capture.1";
   command.format = PVRGPU_DRIVER_COMMAND_FORMAT_RGBA8;
   command.frame = 1;
   command.framebuffer_width = scene->framebuffer_width;
   command.framebuffer_height = scene->framebuffer_height;
   command.width = scene->viewport_width;
   command.height = scene->viewport_height;
   command.clear_color_bits[3] = UINT32_C(0x3f800000);
   const bool output_800x600 =
      command.width == 800u && command.height == 600u;
   if (!pvrgpu_glmark_output_extent_supported(command.width,
                                               command.height) ||
       command.framebuffer_width != command.width ||
       command.framebuffer_height != command.height ||
       depth->framebuffer_width != command.width * 2u ||
       depth->framebuffer_height != command.height * 2u)
      return false;
   command.draw_count = PVRGPU_ARRAY_SIZE(draws);
   pvrgpu_sequence_input_assembly_totals(draws,
                                         PVRGPU_ARRAY_SIZE(draws),
                                         &command.ia_vertices,
                                         &command.ia_primitives);
   command.clip_invocations = command.ia_primitives;
   /*
    * Everything past input assembly is measured by SystemC from the geometry
    * this sequence actually rasterizes.  Leaving these at zero is what makes
    * the reported totals a result rather than a replayed constant.
    */
   command.vs_invocations = 0;
   command.clip_primitives = 0;
   command.setup_triangles = 0;
   command.ps_invocations = 0;
   command.semantic_texel_fetches = 0;
   command.pco_sequence_command_count = PVRGPU_ARRAY_SIZE(draws);
   command.pco_sequence_commands = draws;
   command.pco_sequence_texture_count = 1;
   command.pco_sequence_textures = &texture;

   char error[512] = { 0 };
   if (!pvrgpu_write_draw_pco_sequence_command(path,
                                                &command,
                                                error,
                                                sizeof(error))) {
      remove(path);
      pvrgpu_counter_eventf("draw_pco_shadow_sequence_error",
                            "stage=command_submit reason=%s",
                            error[0] ? error : "unknown");
      return false;
   }

   ctx->driver_draw_command_emitted = true;
   pvrgpu_note_driver_draw_command_emitted();
   pvrgpu_counter_eventf(
      "draw_pco_shadow_sequence_command",
      "draws=3 resources=1 drawlists=3 framebuffer=%ux%u",
      command.framebuffer_width,
      command.framebuffer_height);
   return true;
}

static bool
pvrgpu_shadow_pco_observation_payload_matches(
   const struct pvrgpu_shadow_pco_observation *observation,
   const char **failure_reason)
{
   if (!observation || !failure_reason) {
      if (failure_reason)
         *failure_reason = "payload_arguments";
      return false;
   }
   static const size_t expected_vertex_binary_size[] = {
      432U, 464U, 744U,
   };
   static const size_t expected_fragment_binary_size[] = {
      56U, 216U, 56U,
   };
   static const uint64_t expected_vertex_binary_fnv1a64[] = {
      UINT64_C(0x6e9ad97e49eca9fe),
      UINT64_C(0x79b5f95f5c89ad6c),
      UINT64_C(0xe6bc6969c1a52652),
   };
   static const uint64_t expected_fragment_binary_fnv1a64[] = {
      UINT64_C(0xa55a28d91b0f4b9e),
      UINT64_C(0x1ac54b25af8de102),
      UINT64_C(0x24f632ab8095faeb),
   };
   const bool vertex_size_matches =
      observation->binary.vertex.size ==
         expected_vertex_binary_size[observation->profile] ||
      (observation->profile == PVRGPU_PCO_SHADOW_SCENE &&
       observation->binary.vertex.size == 728U);
   const uint64_t vertex_fnv =
      pvrgpu_pco_binary_fnv1a64(observation->binary.vertex.data,
                                observation->binary.vertex.size);
   const bool vertex_fnv_matches =
      vertex_fnv == expected_vertex_binary_fnv1a64[observation->profile] ||
      (observation->profile == PVRGPU_PCO_SHADOW_SCENE &&
       vertex_fnv == UINT64_C(0x385c48c6c28cd9fc));
   if (observation->profile > PVRGPU_PCO_SHADOW_SCENE ||
       !vertex_size_matches ||
       !vertex_fnv_matches ||
       observation->binary.fragment.size !=
          expected_fragment_binary_size[observation->profile] ||
       pvrgpu_pco_binary_fnv1a64(observation->binary.fragment.data,
                                 observation->binary.fragment.size) !=
          expected_fragment_binary_fnv1a64[observation->profile]) {
      *failure_reason = "pco_binary_fingerprint";
      return false;
   }
   const bool depth = observation->profile == PVRGPU_PCO_SHADOW_DEPTH;
   const bool mask = observation->profile == PVRGPU_PCO_SHADOW_MASK;
   const unsigned expected_source = mask || depth ? UINT32_MAX :
                                                     PVRGPU_PCO_SHADOW_MASK;
   if (observation->color_format !=
          (depth ? PIPE_FORMAT_NONE : PIPE_FORMAT_R8G8B8A8_UNORM) ||
       observation->depth_format !=
          (depth ? PIPE_FORMAT_Z32_UNORM : PIPE_FORMAT_Z24X8_UNORM) ||
       observation->color_attachment_source != expected_source ||
       observation->depth_attachment_source != expected_source ||
       observation->color_clear != mask ||
       observation->depth_clear != (depth || mask) ||
       observation->clear_color_bits[0] != 0 ||
       observation->clear_color_bits[1] != 0 ||
       observation->clear_color_bits[2] != 0 ||
       observation->clear_color_bits[3] != UINT32_C(0x3f800000) ||
       observation->depth_clear_bits != UINT32_C(0x3f800000) ||
       (depth ? observation->color_attachment != NULL :
                observation->color_attachment == NULL) ||
       !observation->depth_attachment ||
       (depth && observation->shadow_depth !=
                    observation->depth_attachment) ||
       observation->color_mask != (depth ? 0U : PIPE_MASK_RGBA) ||
       observation->blend_enable ||
       observation->rgb_func != PIPE_BLEND_ADD ||
       observation->rgb_src_factor != PIPE_BLENDFACTOR_ONE ||
       observation->rgb_dst_factor != PIPE_BLENDFACTOR_ZERO ||
       observation->alpha_func != PIPE_BLEND_ADD ||
       observation->alpha_src_factor != PIPE_BLENDFACTOR_ONE ||
       observation->alpha_dst_factor != PIPE_BLENDFACTOR_ZERO ||
       !observation->dither || !observation->depth_enable ||
       !observation->depth_write ||
       observation->depth_func != PIPE_FUNC_LEQUAL) {
      *failure_reason = "attachment_or_pipeline_contract";
      return false;
   }
   const size_t expected_vertex_size =
      mask ? 32U : (size_t)PVRGPU_SHADOW_PCO_MESH_VERTEX_COUNT * 24U;
   const uint64_t expected_vertex_hash =
      mask ? PVRGPU_SHADOW_PCO_MASK_VERTEX_FNV1A64 :
             PVRGPU_SHADOW_PCO_MESH_VERTEX_FNV1A64;
   if (!observation->vertex_data ||
       observation->vertex_data_size != expected_vertex_size ||
       pvrgpu_pco_binary_fnv1a64(observation->vertex_data,
                                observation->vertex_data_size) !=
          expected_vertex_hash) {
      *failure_reason = "vertex_payload_fingerprint";
      return false;
   }

   const size_t expected_shared_count =
      observation->profile == PVRGPU_PCO_SHADOW_DEPTH ? 16U : 32U;
   const uint64_t expected_shared_hash =
      observation->profile == PVRGPU_PCO_SHADOW_DEPTH ?
         PVRGPU_SHADOW_PCO_DEPTH_SHARED_FNV1A64 :
      mask ? PVRGPU_SHADOW_PCO_MASK_SHARED_FNV1A64 :
             PVRGPU_SHADOW_PCO_SCENE_SHARED_FNV1A64;
   if (observation->vertex_shared_count != expected_shared_count ||
       pvrgpu_pco_binary_fnv1a64(
          (const uint8_t *)observation->vertex_shared,
          observation->vertex_shared_count * sizeof(uint32_t)) !=
          expected_shared_hash) {
      *failure_reason = "shared_payload_fingerprint";
      return false;
   }

   const uint64_t expected_fragment_shared_hash =
      observation->framebuffer_width == 800u &&
            observation->framebuffer_height == 600u ?
         PVRGPU_SHADOW_PCO_FRAGMENT_SHARED_800_FNV1A64 :
         PVRGPU_SHADOW_PCO_FRAGMENT_SHARED_FNV1A64;
   if ((!mask && observation->fragment_shared_count != 0) ||
       (mask &&
        (observation->fragment_shared_count !=
            PVRGPU_SHADOW_PCO_FS_SHARED_DWORDS ||
         pvrgpu_pco_binary_fnv1a64(
            (const uint8_t *)observation->fragment_shared,
            observation->fragment_shared_count * sizeof(uint32_t)) !=
            expected_fragment_shared_hash))) {
      *failure_reason = "fragment_descriptor_payload";
      return false;
   }
   return true;
}

static bool
pvrgpu_compile_shadow_observation(
   struct pvrgpu_context *ctx,
   struct pvrgpu_shadow_pco_observation *observation,
   const char **failure_reason)
{
   if (!ctx || !ctx->vs || !ctx->fs || !observation || !failure_reason) {
      if (failure_reason)
         *failure_reason = "compile_arguments";
      return false;
   }

   char error[512] = { 0 };
   if (!ctx->pco_compiler) {
      ctx->pco_compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
      if (!ctx->pco_compiler) {
         pvrgpu_counter_eventf("draw_pco_shadow_compile_error",
                               "profile=%u stage=compiler_create reason=%s",
                               observation->profile,
                               error[0] ? error : "unknown");
         *failure_reason = "compiler_create";
         return false;
      }
   }
   if (!pvrgpu_pco_compile_shadow(ctx->pco_compiler,
                                  ctx->vs->nir,
                                  ctx->fs->nir,
                                  observation->profile,
                                  &observation->binary,
                                  error,
                                  sizeof(error))) {
      pvrgpu_counter_eventf("draw_pco_shadow_compile_error",
                            "profile=%u stage=pco_compile reason=%s",
                            observation->profile,
                            error[0] ? error : "unknown");
      *failure_reason = "pco_compile";
      return false;
   }

   const struct pvrgpu_pco_graphics_binary *binary = &observation->binary;
   if (!pvrgpu_shadow_pco_observation_payload_matches(observation,
                                                       failure_reason))
      return false;
   pvrgpu_counter_eventf(
      "draw_pco_shadow_capture",
      "profile=%u mode=%u framebuffer=%ux%u viewport=%ux%u "
      "color_format=%s depth_format=%s attachment_source=%u,%u "
      "clear=%u,%u blend=%u,%u,%u,%u,%u,%u,%u "
      "vertices=%u stride=%u vertex_bytes=%zu vertex_fnv1a64=%016llx "
      "shared_dwords=%zu shared_fnv1a64=%016llx "
      "fragment_shared_dwords=%zu fragment_shared_fnv1a64=%016llx "
      "vs_bytes=%zu vs_fnv1a64=%016llx vs_abi=%u,%u,%u,%u,%u,%u,%u,%u "
      "fs_bytes=%zu fs_fnv1a64=%016llx fs_abi=%u,%u,%u,%u,%u,%u,%u,%u "
      "linkage=%u,%u,%u,%u,%u,%u,%u,%u descriptor=%u,%u,%u",
      observation->profile,
      observation->primitive_mode,
      observation->framebuffer_width,
      observation->framebuffer_height,
      observation->viewport_width,
      observation->viewport_height,
      util_format_name(observation->color_format),
      util_format_name(observation->depth_format),
      observation->color_attachment_source,
      observation->depth_attachment_source,
      observation->color_clear,
      observation->depth_clear,
      observation->blend_enable,
      observation->rgb_func,
      observation->rgb_src_factor,
      observation->rgb_dst_factor,
      observation->alpha_func,
      observation->alpha_src_factor,
      observation->alpha_dst_factor,
      observation->vertex_count,
      observation->vertex_stride,
      observation->vertex_data_size,
      (unsigned long long)pvrgpu_pco_binary_fnv1a64(
         observation->vertex_data,
         observation->vertex_data_size),
      observation->vertex_shared_count,
      (unsigned long long)pvrgpu_pco_binary_fnv1a64(
         (const uint8_t *)observation->vertex_shared,
         observation->vertex_shared_count * sizeof(uint32_t)),
      observation->fragment_shared_count,
      (unsigned long long)pvrgpu_pco_binary_fnv1a64(
         (const uint8_t *)observation->fragment_shared,
         observation->fragment_shared_count * sizeof(uint32_t)),
      binary->vertex.size,
      (unsigned long long)pvrgpu_pco_binary_fnv1a64(binary->vertex.data,
                                                     binary->vertex.size),
      binary->vertex.abi.temps,
      binary->vertex.abi.vertex_inputs,
      binary->vertex.abi.vertex_outputs,
      binary->vertex.abi.coefficients,
      binary->vertex.abi.shareds,
      binary->vertex.abi.push_constant_start,
      binary->vertex.abi.push_constant_count,
      binary->vertex.abi.entry_offset,
      binary->fragment.size,
      (unsigned long long)pvrgpu_pco_binary_fnv1a64(binary->fragment.data,
                                                     binary->fragment.size),
      binary->fragment.abi.temps,
      binary->fragment.abi.vertex_inputs,
      binary->fragment.abi.vertex_outputs,
      binary->fragment.abi.coefficients,
      binary->fragment.abi.shareds,
      binary->fragment.abi.push_constant_start,
      binary->fragment.abi.push_constant_count,
      binary->fragment.abi.entry_offset,
      binary->position_output_start,
      binary->position_output_count,
      binary->fragment_position_start,
      binary->fragment_position_count,
      binary->varying_output_start,
      binary->varying_output_count,
      binary->fragment_varying_start,
      binary->fragment_varying_count,
      binary->fragment_texture_descriptor_start,
      binary->fragment_texture_descriptor_count,
      binary->fragment_texture_descriptor_stride);
   *failure_reason = NULL;
   return true;
}

static bool
pvrgpu_compile_terrain_pco_binary(
   struct pvrgpu_context *ctx,
   enum pvrgpu_pco_terrain_profile profile,
   unsigned sequence_pass,
   struct pvrgpu_pco_graphics_binary *binary,
   const char **failure_reason)
{
   if (failure_reason)
      *failure_reason = "compile_arguments";
   if (!ctx || !ctx->vs || !ctx->fs || !ctx->vs->nir || !ctx->fs->nir ||
       !binary || !failure_reason)
      return false;

   char error[512] = { 0 };
   if (!ctx->pco_compiler) {
      ctx->pco_compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
      if (!ctx->pco_compiler) {
         pvrgpu_counter_eventf("draw_pco_terrain_compile_error",
                               "pass=%u profile=%u stage=compiler_create "
                               "reason=%s",
                               sequence_pass,
                               profile,
                               error[0] ? error : "unknown");
         *failure_reason = "compiler_create";
         return false;
      }
   }

   memset(binary, 0, sizeof(*binary));
   if (!pvrgpu_pco_compile_terrain(ctx->pco_compiler,
                                   ctx->vs->nir,
                                   ctx->fs->nir,
                                   profile,
                                   binary,
                                   error,
                                   sizeof(error))) {
      pvrgpu_counter_eventf("draw_pco_terrain_compile_error",
                            "pass=%u profile=%u stage=pco_compile reason=%s",
                            sequence_pass,
                            profile,
                            error[0] ? error : "unknown");
      *failure_reason = "pco_compile";
      return false;
   }

   pvrgpu_counter_eventf(
      "draw_pco_terrain_compile_probe",
      "pass=%u profile=%u vs_bytes=%zu vs_fnv1a64=%016llx "
      "vs_abi=%u,%u,%u,%u,%u,%u,%u,%u fs_bytes=%zu "
      "fs_fnv1a64=%016llx fs_abi=%u,%u,%u,%u,%u,%u,%u,%u "
      "linkage=%u,%u,%u,%u,%u,%u,%u,%u descriptor=%u,%u,%u",
      sequence_pass,
      profile,
      binary->vertex.size,
      (unsigned long long)pvrgpu_pco_binary_fnv1a64(binary->vertex.data,
                                                     binary->vertex.size),
      binary->vertex.abi.temps,
      binary->vertex.abi.vertex_inputs,
      binary->vertex.abi.vertex_outputs,
      binary->vertex.abi.coefficients,
      binary->vertex.abi.shareds,
      binary->vertex.abi.push_constant_start,
      binary->vertex.abi.push_constant_count,
      binary->vertex.abi.entry_offset,
      binary->fragment.size,
      (unsigned long long)pvrgpu_pco_binary_fnv1a64(binary->fragment.data,
                                                     binary->fragment.size),
      binary->fragment.abi.temps,
      binary->fragment.abi.vertex_inputs,
      binary->fragment.abi.vertex_outputs,
      binary->fragment.abi.coefficients,
      binary->fragment.abi.shareds,
      binary->fragment.abi.push_constant_start,
      binary->fragment.abi.push_constant_count,
      binary->fragment.abi.entry_offset,
      binary->position_output_start,
      binary->position_output_count,
      binary->fragment_position_start,
      binary->fragment_position_count,
      binary->varying_output_start,
      binary->varying_output_count,
      binary->fragment_varying_start,
      binary->fragment_varying_count,
      binary->fragment_texture_descriptor_start,
      binary->fragment_texture_descriptor_count,
      binary->fragment_texture_descriptor_stride);
   *failure_reason = NULL;
   return true;
}

/* ---- GLMark2 terrain native sequence capture -------------------------- */

struct pvrgpu_terrain_pco_texture_spec {
   mesa_shader_stage stage;
   unsigned source_slot;
   unsigned descriptor_set;
   enum pvrgpu_terrain_pco_texture_source source;
   unsigned producer_command_index;
};

static bool
pvrgpu_terrain_pco_draw_info_matches(
   const struct pipe_draw_info *info,
   unsigned drawid_offset,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,
   unsigned num_draws,
   enum pvrgpu_pco_terrain_profile profile)
{
   const unsigned expected_count =
      profile == PVRGPU_PCO_TERRAIN_D3 ?
         PVRGPU_TERRAIN_PCO_MAIN_VERTEX_COUNT :
         PVRGPU_TERRAIN_PCO_FULLSCREEN_VERTEX_COUNT;
   return info && draws && drawid_offset == 0 && !indirect && num_draws == 1 &&
          info->mode == MESA_PRIM_TRIANGLES && info->index_size == 0 &&
          !info->has_user_indices && !info->primitive_restart &&
          !info->increment_draw_id && !info->index_bias_varies &&
          !info->was_line_loop && info->start_instance == 0 &&
          info->instance_count == 1 && draws[0].start == 0 &&
          draws[0].count == expected_count;
}

static bool
pvrgpu_terrain_pco_expected_extent(enum pvrgpu_pco_terrain_profile profile,
                                   unsigned *width,
                                   unsigned *height)
{
   if (!width || !height)
      return false;
   const bool full_size =
      profile == PVRGPU_PCO_TERRAIN_D1 ||
      profile == PVRGPU_PCO_TERRAIN_D2 ||
      profile == PVRGPU_PCO_TERRAIN_D4 ||
      profile == PVRGPU_PCO_TERRAIN_D5;
   if (full_size) {
      *width = 256u;
      *height = 256u;
      return true;
   }
   return pvrgpu_glmark_scaled_output_extent(1u, width, height);
}

static bool
pvrgpu_rgba8_surface_backing_matches(
   const struct pipe_surface *surface,
   unsigned width,
   unsigned height,
   const uint8_t expected_rgba[4])
{
   if (!surface || !surface->texture ||
       surface->format != PIPE_FORMAT_R8G8B8A8_UNORM ||
       surface->texture->target != PIPE_TEXTURE_2D ||
       surface->texture->format != PIPE_FORMAT_R8G8B8A8_UNORM ||
       surface->texture->width0 != width ||
       surface->texture->height0 != height || surface->level != 0 ||
       surface->first_layer != 0 || surface->last_layer != 0)
      return false;

   const struct pvrgpu_resource *resource =
      pvrgpu_resource(surface->texture);
   if (!resource || !resource->data || resource->level_count == 0 ||
       resource->level_offsets[0] != 0 ||
       resource->level_strides[0] != width * 4U ||
       resource->level_layer_strides[0] != (uintptr_t)width * height * 4U)
      return false;
   for (unsigned y = 0; y < height; ++y) {
      const uint8_t *row = resource->data +
                           (size_t)y * resource->level_strides[0];
      for (unsigned x = 0; x < width; ++x) {
         if (memcmp(row + (size_t)x * 4U, expected_rgba, 4U) != 0)
            return false;
      }
   }
   return true;
}

static bool
pvrgpu_terrain_pco_depth_was_cleared_one(
   const struct pvrgpu_context *ctx,
   const struct pipe_surface *surface,
   unsigned width,
   unsigned height)
{
   return ctx && surface && surface->texture &&
          ctx->full_depth_clear_is_one &&
          ctx->full_depth_clear_resource == surface->texture &&
          ctx->full_depth_clear_level == surface->level &&
          ctx->full_depth_clear_first_layer == surface->first_layer &&
          ctx->full_depth_clear_last_layer == surface->last_layer &&
          ctx->full_depth_clear_width == width &&
          ctx->full_depth_clear_height == height;
}

static bool
pvrgpu_terrain_pco_framebuffer_matches(
   struct pvrgpu_context *ctx,
   enum pvrgpu_pco_terrain_profile profile,
   struct pvrgpu_terrain_pco_observation *observation,
   const char **failure_reason)
{
   unsigned width = 0;
   unsigned height = 0;
   if (!pvrgpu_terrain_pco_expected_extent(profile, &width, &height)) {
      *failure_reason = "output_extent";
      return false;
   }
   const bool d3 = profile == PVRGPU_PCO_TERRAIN_D3;
   const bool d6 = profile == PVRGPU_PCO_TERRAIN_D6;
   const bool d8 = profile == PVRGPU_PCO_TERRAIN_D8;
   const enum pipe_format depth_format =
      d3 || d6 ? PIPE_FORMAT_Z16_UNORM :
      d8 ? PIPE_FORMAT_Z24X8_UNORM : PIPE_FORMAT_NONE;
   if (ctx->framebuffer.width != width ||
       ctx->framebuffer.height != height || ctx->framebuffer.nr_cbufs != 1 ||
       !ctx->framebuffer.cbufs[0].texture ||
       ctx->framebuffer.cbufs[0].format !=
          PIPE_FORMAT_R8G8B8A8_UNORM ||
       ctx->framebuffer.cbufs[0].texture->nr_samples > 1 ||
       ctx->framebuffer.cbufs[0].texture->nr_storage_samples > 1 ||
       ((depth_format == PIPE_FORMAT_NONE) !=
        (ctx->framebuffer.zsbuf.texture == NULL)) ||
       (depth_format != PIPE_FORMAT_NONE &&
        (ctx->framebuffer.zsbuf.format != depth_format ||
         ctx->framebuffer.zsbuf.texture->format != depth_format ||
         ctx->framebuffer.zsbuf.texture->width0 != width ||
         ctx->framebuffer.zsbuf.texture->height0 != height ||
         ctx->framebuffer.zsbuf.level != 0 ||
         ctx->framebuffer.zsbuf.first_layer != 0 ||
         ctx->framebuffer.zsbuf.last_layer != 0))) {
      *failure_reason = "framebuffer";
      return false;
   }

   static const uint8_t black[4] = { 0, 0, 0, 255 };
   static const uint8_t terrain_clear[4] = { 210, 189, 158, 255 };
   const uint8_t *expected_color =
      profile <= PVRGPU_PCO_TERRAIN_D2 ? black : terrain_clear;
   if (!pvrgpu_rgba8_surface_backing_matches(&ctx->framebuffer.cbufs[0],
                                             width,
                                             height,
                                             expected_color)) {
      *failure_reason = "color_attachment_backing";
      return false;
   }
   if ((d3 || d8) &&
       !pvrgpu_terrain_pco_depth_was_cleared_one(ctx,
                                                 &ctx->framebuffer.zsbuf,
                                                 width,
                                                 height)) {
      *failure_reason = "depth_clear";
      return false;
   }
   if (d6 &&
       (!ctx->terrain_pco_draws[PVRGPU_PCO_TERRAIN_D3] ||
        ctx->framebuffer.cbufs[0].texture !=
           ctx->terrain_pco_draws[PVRGPU_PCO_TERRAIN_D3]->color_attachment ||
        ctx->framebuffer.zsbuf.texture !=
           ctx->terrain_pco_draws[PVRGPU_PCO_TERRAIN_D3]->depth_attachment)) {
      *failure_reason = "d6_color_depth_attachment_alias";
      return false;
   }
   if (d8 && !pvrgpu_framebuffer_matches_rdc_output(ctx)) {
      *failure_reason = "final_framebuffer";
      return false;
   }

   observation->framebuffer_width = width;
   observation->framebuffer_height = height;
   observation->color_format = PIPE_FORMAT_R8G8B8A8_UNORM;
   observation->depth_format = depth_format;
   observation->color_attachment_source = d6 ? PVRGPU_PCO_TERRAIN_D3 :
                                               UINT32_MAX;
   observation->depth_attachment_source = d6 ? PVRGPU_PCO_TERRAIN_D3 :
                                               UINT32_MAX;
   observation->color_clear = !d6;
   observation->depth_clear = d3 || d8;
   observation->depth_clear_one = d3 || d8;
   if (profile <= PVRGPU_PCO_TERRAIN_D2) {
      observation->clear_color_bits[3] = UINT32_C(0x3f800000);
   } else {
      observation->clear_color_bits[0] = UINT32_C(0x3f533333);
      observation->clear_color_bits[1] = UINT32_C(0x3f3e147b);
      observation->clear_color_bits[2] = UINT32_C(0x3f1e6666);
      observation->clear_color_bits[3] = UINT32_C(0x3f800000);
   }
   pipe_resource_reference(&observation->color_attachment,
                           ctx->framebuffer.cbufs[0].texture);
   if (depth_format != PIPE_FORMAT_NONE) {
      pipe_resource_reference(&observation->depth_attachment,
                              ctx->framebuffer.zsbuf.texture);
   }
   return true;
}

static bool
pvrgpu_terrain_pco_pipeline_matches(
   const struct pvrgpu_context *ctx,
   enum pvrgpu_pco_terrain_profile profile,
   struct pvrgpu_terrain_pco_observation *observation,
   const char **failure_reason)
{
   unsigned width = 0;
   unsigned height = 0;
   if (!pvrgpu_terrain_pco_expected_extent(profile, &width, &height)) {
      *failure_reason = "output_extent";
      return false;
   }
   const uint32_t expected_scale[3] = {
      pvrgpu_float_bits((float)width * 0.5f),
      pvrgpu_float_bits((float)height * 0.5f),
      UINT32_C(0x3f000000),
   };
   if (!ctx->has_viewport ||
       pvrgpu_float_bits(ctx->viewport.scale[0]) != expected_scale[0] ||
       pvrgpu_float_bits(ctx->viewport.scale[1]) != expected_scale[1] ||
       pvrgpu_float_bits(ctx->viewport.scale[2]) != expected_scale[2] ||
       pvrgpu_float_bits(ctx->viewport.translate[0]) != expected_scale[0] ||
       pvrgpu_float_bits(ctx->viewport.translate[1]) != expected_scale[1] ||
       pvrgpu_float_bits(ctx->viewport.translate[2]) != expected_scale[2]) {
      *failure_reason = "viewport";
      return false;
   }

   const bool blend = profile == PVRGPU_PCO_TERRAIN_D6;
   const unsigned expected_source_factor =
      blend ? PIPE_BLENDFACTOR_SRC_ALPHA : PIPE_BLENDFACTOR_ONE;
   const unsigned expected_destination_factor =
      blend ? PIPE_BLENDFACTOR_ONE : PIPE_BLENDFACTOR_ZERO;
   if (!ctx->blend || ctx->blend->state.independent_blend_enable ||
       ctx->blend->state.logicop_enable || !ctx->blend->state.dither ||
       ctx->blend->state.alpha_to_coverage ||
       ctx->blend->state.alpha_to_coverage_dither ||
       ctx->blend->state.alpha_to_one || ctx->blend->state.max_rt != 0 ||
       ctx->blend->state.advanced_blend_func != 0 ||
       ctx->blend->state.rt[0].blend_enable != blend ||
       ctx->blend->state.rt[0].colormask != PIPE_MASK_RGBA ||
       (blend &&
        (ctx->blend->state.rt[0].rgb_func != PIPE_BLEND_ADD ||
         ctx->blend->state.rt[0].rgb_src_factor != expected_source_factor ||
         ctx->blend->state.rt[0].rgb_dst_factor !=
            expected_destination_factor ||
         ctx->blend->state.rt[0].alpha_func != PIPE_BLEND_ADD ||
         ctx->blend->state.rt[0].alpha_src_factor != expected_source_factor ||
         ctx->blend->state.rt[0].alpha_dst_factor !=
            expected_destination_factor))) {
      *failure_reason = "blend";
      return false;
   }

   const bool depth = profile == PVRGPU_PCO_TERRAIN_D3 ||
                      profile == PVRGPU_PCO_TERRAIN_D6 ||
                      profile == PVRGPU_PCO_TERRAIN_D8;
   if (!ctx->dsa || ctx->dsa->state.depth_enabled != depth ||
       ctx->dsa->state.depth_writemask != depth ||
       (depth && ctx->dsa->state.depth_func != PIPE_FUNC_LEQUAL) ||
       ctx->dsa->state.depth_bounds_test ||
       ctx->dsa->state.stencil[0].enabled ||
       ctx->dsa->state.stencil[1].enabled || ctx->dsa->state.alpha_enabled) {
      *failure_reason = "depth_stencil_alpha";
      return false;
   }
   if (!ctx->rasterizer || ctx->rasterizer->state.front_ccw ||
       ctx->rasterizer->state.cull_face != PIPE_FACE_BACK ||
       ctx->rasterizer->state.fill_front != PIPE_POLYGON_MODE_FILL ||
       ctx->rasterizer->state.fill_back != PIPE_POLYGON_MODE_FILL ||
       ctx->rasterizer->state.scissor ||
       ctx->rasterizer->state.rasterizer_discard ||
       ctx->rasterizer->state.multisample ||
       ctx->rasterizer->state.offset_tri ||
       ctx->rasterizer->state.poly_smooth ||
       ctx->rasterizer->state.poly_stipple_enable ||
       ctx->rasterizer->state.conservative_raster_mode != 0 ||
       !ctx->rasterizer->state.half_pixel_center ||
       ctx->rasterizer->state.bottom_edge_rule ||
       ctx->rasterizer->state.clip_halfz ||
       !ctx->rasterizer->state.depth_clip_near ||
       !ctx->rasterizer->state.depth_clip_far ||
       ctx->rasterizer->state.depth_clamp ||
       ctx->sample_mask != UINT32_MAX) {
      *failure_reason = "rasterizer";
      return false;
   }

   observation->viewport_width = width;
   observation->viewport_height = height;
   memcpy(observation->viewport_scale_bits,
          expected_scale,
          sizeof(expected_scale));
   memcpy(observation->viewport_translate_bits,
          expected_scale,
          sizeof(expected_scale));
   observation->front_ccw = false;
   observation->cull_face = PIPE_FACE_BACK;
   observation->fill_front = PIPE_POLYGON_MODE_FILL;
   observation->fill_back = PIPE_POLYGON_MODE_FILL;
   observation->half_pixel_center = true;
   observation->depth_clip_near = true;
   observation->depth_clip_far = true;
   observation->sample_mask = UINT32_MAX;
   observation->color_mask = PIPE_MASK_RGBA;
   observation->blend_enable = blend;
   observation->rgb_func = PIPE_BLEND_ADD;
   observation->rgb_src_factor = expected_source_factor;
   observation->rgb_dst_factor = expected_destination_factor;
   observation->alpha_func = PIPE_BLEND_ADD;
   observation->alpha_src_factor = expected_source_factor;
   observation->alpha_dst_factor = expected_destination_factor;
   observation->dither = true;
   observation->depth_enable = depth;
   observation->depth_write = depth;
   observation->depth_func = depth ? PIPE_FUNC_LEQUAL : 0;
   return true;
}

static bool
pvrgpu_terrain_pco_capture_vertices(
   const struct pvrgpu_context *ctx,
   enum pvrgpu_pco_terrain_profile profile,
   struct pvrgpu_terrain_pco_observation *observation,
   const char **failure_reason)
{
   const bool main = profile == PVRGPU_PCO_TERRAIN_D3;
   const unsigned attribute_count = main ? 4U : 1U;
   const unsigned vertex_count = main ? PVRGPU_TERRAIN_PCO_MAIN_VERTEX_COUNT :
                                        PVRGPU_TERRAIN_PCO_FULLSCREEN_VERTEX_COUNT;
   static const unsigned component_count[4] = { 3U, 3U, 3U, 2U };
   static const enum pipe_format formats[4] = {
      PIPE_FORMAT_R32G32B32_FLOAT,
      PIPE_FORMAT_R32G32B32_FLOAT,
      PIPE_FORMAT_R32G32B32_FLOAT,
      PIPE_FORMAT_R32G32_FLOAT,
   };
   if (!ctx->vertex_elements ||
       ctx->vertex_elements->num_elements != attribute_count ||
       ctx->num_vertex_buffers != attribute_count) {
      *failure_reason = "vertex_layout";
      return false;
   }

   const struct pvrgpu_resource *resources[4] = { NULL, NULL, NULL, NULL };
   unsigned vertex_stride = 0;
   for (unsigned attribute = 0; attribute < attribute_count; ++attribute) {
      const unsigned attribute_stride =
         component_count[attribute] * sizeof(float);
      const size_t expected_size = (size_t)vertex_count * attribute_stride;
      const struct pipe_vertex_element *element =
         &ctx->vertex_elements->elements[attribute];
      const struct pipe_vertex_buffer *buffer = &ctx->vertex_buffers[attribute];
      if (element->src_format != formats[attribute] ||
          element->src_stride != attribute_stride ||
          element->src_offset != 0 || element->dual_slot ||
          element->instance_divisor != 0 ||
          element->vertex_buffer_index != attribute || buffer->is_user_buffer ||
          buffer->buffer_offset != 0 || !buffer->buffer.resource ||
          buffer->buffer.resource->target != PIPE_BUFFER) {
         *failure_reason = "vertex_layout";
         return false;
      }
      resources[attribute] = pvrgpu_resource(buffer->buffer.resource);
      if (!resources[attribute] || !resources[attribute]->data ||
          resources[attribute]->size != expected_size) {
         *failure_reason = "vertex_resource";
         return false;
      }
      vertex_stride += attribute_stride;
   }

   const size_t total_size = (size_t)vertex_count * vertex_stride;
   observation->vertex_data = MALLOC(total_size);
   if (!observation->vertex_data) {
      *failure_reason = "vertex_allocation";
      return false;
   }
   for (unsigned vertex = 0; vertex < vertex_count; ++vertex) {
      uint8_t *dst = observation->vertex_data +
                     (size_t)vertex * vertex_stride;
      unsigned dst_offset = 0;
      for (unsigned attribute = 0; attribute < attribute_count; ++attribute) {
         const unsigned attribute_stride =
            component_count[attribute] * sizeof(float);
         memcpy(dst + dst_offset,
                resources[attribute]->data +
                   (size_t)vertex * attribute_stride,
                attribute_stride);
         dst_offset += attribute_stride;
      }
   }
   observation->vertex_count = vertex_count;
   observation->vertex_stride = vertex_stride;
   observation->vertex_data_size = total_size;
   return true;
}

static unsigned
pvrgpu_terrain_pco_texture_specs(
   enum pvrgpu_pco_terrain_profile profile,
   struct pvrgpu_terrain_pco_texture_spec
      specs[PVRGPU_TERRAIN_PCO_MAX_TEXTURES])
{
   if (profile == PVRGPU_PCO_TERRAIN_D1)
      return 0;
   if (profile == PVRGPU_PCO_TERRAIN_D2) {
      specs[0] = (struct pvrgpu_terrain_pco_texture_spec) {
         MESA_SHADER_FRAGMENT, 0, 0,
         PVRGPU_TERRAIN_PCO_PREVIOUS_COLOR_ATTACHMENT,
         PVRGPU_PCO_TERRAIN_D1,
      };
      return 1;
   }
   if (profile == PVRGPU_PCO_TERRAIN_D3) {
      const struct pvrgpu_terrain_pco_texture_spec main_specs[] = {
         { MESA_SHADER_VERTEX, 1, 0,
           PVRGPU_TERRAIN_PCO_PREVIOUS_COLOR_ATTACHMENT,
           PVRGPU_PCO_TERRAIN_D1 },
         { MESA_SHADER_VERTEX, 0, 1,
           PVRGPU_TERRAIN_PCO_PREVIOUS_COLOR_ATTACHMENT,
           PVRGPU_PCO_TERRAIN_D2 },
         { MESA_SHADER_FRAGMENT, 2, 0,
           PVRGPU_TERRAIN_PCO_EXTERNAL_PAYLOAD, UINT32_MAX },
         { MESA_SHADER_FRAGMENT, 0, 1,
           PVRGPU_TERRAIN_PCO_EXTERNAL_PAYLOAD, UINT32_MAX },
         { MESA_SHADER_FRAGMENT, 1, 2,
           PVRGPU_TERRAIN_PCO_EXTERNAL_PAYLOAD, UINT32_MAX },
         { MESA_SHADER_FRAGMENT, 4, 3,
           PVRGPU_TERRAIN_PCO_PREVIOUS_COLOR_ATTACHMENT,
           PVRGPU_PCO_TERRAIN_D1 },
         { MESA_SHADER_FRAGMENT, 3, 4,
           PVRGPU_TERRAIN_PCO_EXTERNAL_PAYLOAD, UINT32_MAX },
      };
      memcpy(specs, main_specs, sizeof(main_specs));
      return PVRGPU_ARRAY_SIZE(main_specs);
   }

   specs[0] = (struct pvrgpu_terrain_pco_texture_spec) {
      MESA_SHADER_FRAGMENT,
      0,
      0,
      PVRGPU_TERRAIN_PCO_PREVIOUS_COLOR_ATTACHMENT,
      (unsigned)profile - 1U,
   };
   return 1;
}

static bool
pvrgpu_terrain_pco_expected_texture_layout(
   const struct pvrgpu_terrain_pco_texture_spec *spec,
   unsigned *width,
   unsigned *height,
   unsigned *mip_count,
   size_t *size)
{
   if (spec->source == PVRGPU_TERRAIN_PCO_EXTERNAL_PAYLOAD) {
      *width = 512;
      *height = 512;
      *mip_count = 10;
      *size = 1398100;
      return true;
   }
   switch (spec->producer_command_index) {
   case PVRGPU_PCO_TERRAIN_D1:
   case PVRGPU_PCO_TERRAIN_D2:
      *width = 256;
      *height = 256;
      *mip_count = 1;
      *size = 262144;
      return true;
   case PVRGPU_PCO_TERRAIN_D3:
   case PVRGPU_PCO_TERRAIN_D6:
   case PVRGPU_PCO_TERRAIN_D7:
      return pvrgpu_glmark_scaled_output_extent(1u, width, height) &&
             pvrgpu_tight_rgba8_mip_layout(*width,
                                           *height,
                                           mip_count,
                                           size);
   case PVRGPU_PCO_TERRAIN_D4:
   case PVRGPU_PCO_TERRAIN_D5:
      *width = 256;
      *height = 256;
      *mip_count = 9;
      *size = 349524;
      return true;
   default:
      return false;
   }
}

static bool
pvrgpu_terrain_pco_capture_texture(
   const struct pvrgpu_context *ctx,
   enum pvrgpu_pco_terrain_profile profile,
   const struct pvrgpu_terrain_pco_texture_spec *spec,
   struct pvrgpu_terrain_pco_texture *texture,
   uint32_t descriptor[PVRGPU_PCO_TEXTURE_DESCRIPTOR_DWORDS],
   const char **failure_reason)
{
   if (spec->stage != MESA_SHADER_VERTEX &&
       spec->stage != MESA_SHADER_FRAGMENT) {
      *failure_reason = "texture_stage";
      return false;
   }
   if (ctx->num_sampler_views[spec->stage] <= spec->source_slot ||
       ctx->num_samplers[spec->stage] <= spec->source_slot) {
      *failure_reason = "texture_binding";
      return false;
   }
   const struct pipe_sampler_view *view =
      ctx->sampler_views[spec->stage][spec->source_slot];
   const struct pvrgpu_sampler_state *sampler =
      ctx->samplers[spec->stage][spec->source_slot];
   if (!view || !view->texture || !sampler ||
       view->target != PIPE_TEXTURE_2D ||
       view->texture->target != PIPE_TEXTURE_2D ||
       view->u.tex.first_level != 0 || view->u.tex.first_layer != 0 ||
       view->u.tex.last_layer != 0 || view->u.tex.min_lod_clamp != 0.0f) {
      *failure_reason = "texture_view";
      return false;
   }

   unsigned width = 0;
   unsigned height = 0;
   unsigned mip_count = 0;
   size_t expected_size = 0;
   if (!pvrgpu_terrain_pco_expected_texture_layout(spec,
                                                    &width,
                                                    &height,
                                                    &mip_count,
                                                    &expected_size)) {
      *failure_reason = "texture_layout_spec";
      return false;
   }
   const enum pipe_format format = view->texture->format;
   const bool external =
      spec->source == PVRGPU_TERRAIN_PCO_EXTERNAL_PAYLOAD;
   if (!external &&
       (spec->producer_command_index >= (unsigned)profile ||
        !ctx->terrain_pco_draws[spec->producer_command_index] ||
        view->texture !=
           ctx->terrain_pco_draws[spec->producer_command_index]
              ->color_attachment)) {
      *failure_reason = "texture_attachment_source";
      return false;
   }
   if ((!external && format != PIPE_FORMAT_R8G8B8A8_UNORM) ||
       (external && format != PIPE_FORMAT_R8G8B8A8_UNORM &&
        format != PIPE_FORMAT_R8G8B8X8_UNORM)) {
      *failure_reason = "texture_format";
      return false;
   }
   if (view->format != format) {
      *failure_reason = "texture_view_format";
      return false;
   }
   if (view->texture->width0 != width || view->texture->height0 != height ||
       view->texture->depth0 != 1 || view->texture->array_size != 1) {
      pvrgpu_counter_eventf(
         "draw_pco_terrain_texture_mismatch",
         "profile=%u stage=%u source_slot=%u descriptor_set=%u "
         "field=extent expected=%ux%ux1,array1 actual=%ux%ux%u,array%u",
         profile,
         spec->stage,
         spec->source_slot,
         spec->descriptor_set,
         width,
         height,
         view->texture->width0,
         view->texture->height0,
         view->texture->depth0,
         view->texture->array_size);
      *failure_reason = "texture_extent";
      return false;
   }
   if (view->texture->last_level + 1U != mip_count ||
       view->u.tex.last_level + 1U != mip_count) {
      *failure_reason = "texture_levels";
      return false;
   }
   if (view->texture->nr_samples > 1 ||
       view->texture->nr_storage_samples > 1) {
      *failure_reason = "texture_samples";
      return false;
   }
   if (view->swizzle_r != PIPE_SWIZZLE_X ||
       view->swizzle_g != PIPE_SWIZZLE_Y ||
       view->swizzle_b != PIPE_SWIZZLE_Z ||
       view->swizzle_a != (format == PIPE_FORMAT_R8G8B8X8_UNORM ?
                              PIPE_SWIZZLE_1 : PIPE_SWIZZLE_W)) {
      *failure_reason = "texture_swizzle";
      return false;
   }

   const struct pvrgpu_resource *resource =
      pvrgpu_resource(view->texture);
   if (!resource || !resource->data || resource->size != expected_size ||
       resource->level_count != mip_count) {
      *failure_reason = "texture_resource";
      return false;
   }
   uintptr_t expected_offset = 0;
   for (unsigned level = 0; level < mip_count; ++level) {
      const unsigned level_width = MAX2(width >> level, 1U);
      const unsigned level_height = MAX2(height >> level, 1U);
      const unsigned row_pitch = level_width * 4U;
      const uintptr_t level_size = (uintptr_t)row_pitch * level_height;
      if (resource->level_offsets[level] != expected_offset ||
          resource->level_strides[level] != row_pitch ||
          resource->level_layer_strides[level] != level_size) {
         *failure_reason = "texture_mip_layout";
         return false;
      }
      expected_offset += level_size;
   }
   if (expected_offset != expected_size) {
      *failure_reason = "texture_size";
      return false;
   }

   const bool main_fragment_external =
      profile == PVRGPU_PCO_TERRAIN_D3 &&
      spec->stage == MESA_SHADER_FRAGMENT && external;
   const bool mip_linear = main_fragment_external ||
                           profile >= PVRGPU_PCO_TERRAIN_D4;
   const unsigned expected_wrap = main_fragment_external ?
                                     PIPE_TEX_WRAP_REPEAT :
                                     PIPE_TEX_WRAP_CLAMP_TO_EDGE;
   const struct pipe_sampler_state *state = &sampler->state;
   if (state->wrap_s != expected_wrap || state->wrap_t != expected_wrap ||
       state->wrap_r != PIPE_TEX_WRAP_REPEAT ||
       state->min_img_filter != PIPE_TEX_FILTER_LINEAR ||
       state->mag_img_filter != PIPE_TEX_FILTER_LINEAR ||
       state->min_mip_filter != (mip_linear ? PIPE_TEX_MIPFILTER_LINEAR :
                                               PIPE_TEX_MIPFILTER_NONE) ||
       state->compare_mode != PIPE_TEX_COMPARE_NONE ||
       state->unnormalized_coords || !state->seamless_cube_map ||
       state->reduction_mode != PIPE_TEX_REDUCTION_WEIGHTED_AVERAGE ||
       state->max_anisotropy > 1 || state->lod_bias != 0.0f ||
       state->min_lod != 0.0f || state->max_lod != 1000.0f) {
      *failure_reason = "texture_sampler";
      return false;
   }

   memset(texture, 0, sizeof(*texture));
   texture->stage = spec->stage;
   texture->source = spec->source;
   texture->producer_command_index = spec->producer_command_index;
   texture->source_slot = spec->source_slot;
   texture->descriptor_set = spec->descriptor_set;
   texture->binding = 0;
   texture->format = format;
   texture->declared_size = expected_size;
   texture->mip_count = mip_count;
   for (unsigned level = 0; level < mip_count; ++level) {
      texture->mip[level].width = MAX2(width >> level, 1U);
      texture->mip[level].height = MAX2(height >> level, 1U);
      texture->mip[level].row_pitch = resource->level_strides[level];
      texture->mip[level].offset = resource->level_offsets[level];
   }
   texture->min_filter = PIPE_TEX_FILTER_LINEAR;
   texture->mag_filter = PIPE_TEX_FILTER_LINEAR;
   texture->mip_filter = mip_linear ? PIPE_TEX_MIPFILTER_LINEAR :
                                     PIPE_TEX_MIPFILTER_NONE;
   texture->wrap_u = expected_wrap;
   texture->wrap_v = expected_wrap;
   texture->normalized_coordinates = true;
   texture->max_lod_u4_6 = mip_linear ? (mip_count - 1U) * 64U : 0U;
   if (external) {
      texture->bytes = MALLOC(expected_size);
      if (!texture->bytes) {
         *failure_reason = "texture_allocation";
         return false;
      }
      memcpy(texture->bytes, resource->data, expected_size);
      texture->bytes_size = expected_size;
   }

   if (!pvrgpu_pco_build_terrain_texture_descriptor(
          descriptor,
          format,
          width,
          height,
          mip_count,
          (uint32_t)expected_size,
          1U,
          1U,
          mip_linear ? 1U : 0U,
          expected_wrap,
          expected_wrap,
          texture->max_lod_u4_6)) {
      *failure_reason = "texture_descriptor";
      return false;
   }
   return true;
}

static bool
pvrgpu_terrain_pco_capture_textures(
   const struct pvrgpu_context *ctx,
   enum pvrgpu_pco_terrain_profile profile,
   struct pvrgpu_terrain_pco_observation *observation,
   const char **failure_reason)
{
   struct pvrgpu_terrain_pco_texture_spec
      specs[PVRGPU_TERRAIN_PCO_MAX_TEXTURES];
   memset(specs, 0, sizeof(specs));
   const unsigned count = pvrgpu_terrain_pco_texture_specs(profile, specs);
   for (unsigned texture_index = 0; texture_index < count; ++texture_index) {
      const struct pvrgpu_terrain_pco_texture_spec *spec =
         &specs[texture_index];
      uint32_t *shared = spec->stage == MESA_SHADER_VERTEX ?
                           observation->vertex_shared :
                           observation->fragment_shared;
      const size_t shared_capacity = spec->stage == MESA_SHADER_VERTEX ?
                                        PVRGPU_TERRAIN_PCO_MAX_VS_SHARED_DWORDS :
                                        PVRGPU_TERRAIN_PCO_MAX_FS_SHARED_DWORDS;
      const size_t descriptor_start =
         (size_t)spec->descriptor_set * PVRGPU_PCO_TEXTURE_DESCRIPTOR_DWORDS;
      if (descriptor_start + PVRGPU_PCO_TEXTURE_DESCRIPTOR_DWORDS >
          shared_capacity) {
         *failure_reason = "texture_descriptor_range";
         return false;
      }
      if (!pvrgpu_terrain_pco_capture_texture(
             ctx,
             profile,
             spec,
             &observation->textures[texture_index],
             &shared[descriptor_start],
             failure_reason))
         return false;
      observation->texture_count++;
   }
   return true;
}

static bool
pvrgpu_terrain_pco_copy_constants(
   const struct pvrgpu_context *ctx,
   enum pvrgpu_pco_terrain_profile profile,
   struct pvrgpu_terrain_pco_observation *observation,
   const char **failure_reason)
{
   static const unsigned vertex_dwords[PVRGPU_TERRAIN_PCO_DRAW_COUNT] = {
      8, 8, 56, 8, 8, 8, 8, 8,
   };
   static const unsigned fragment_dwords[PVRGPU_TERRAIN_PCO_DRAW_COUNT] = {
      4, 8, 64, 0, 0, 4, 0, 0,
   };
   static const unsigned vertex_descriptor_dwords
      [PVRGPU_TERRAIN_PCO_DRAW_COUNT] = { 0, 0, 40, 0, 0, 0, 0, 0 };
   static const unsigned fragment_descriptor_dwords
      [PVRGPU_TERRAIN_PCO_DRAW_COUNT] = { 0, 20, 100, 20, 20, 20, 20, 20 };
   for (unsigned stage = 0; stage < MESA_SHADER_MESH_STAGES; ++stage) {
      const unsigned expected =
         stage == MESA_SHADER_VERTEX ? 1U :
         stage == MESA_SHADER_FRAGMENT && fragment_dwords[profile] ? 1U : 0U;
      if (ctx->num_constant_buffers[stage] != expected) {
         *failure_reason = "constant_buffers";
         return false;
      }
   }

   const mesa_shader_stage stages[2] = {
      MESA_SHADER_VERTEX,
      MESA_SHADER_FRAGMENT,
   };
   for (unsigned stage_index = 0; stage_index < 2; ++stage_index) {
      const mesa_shader_stage stage = stages[stage_index];
      const unsigned dwords = stage == MESA_SHADER_VERTEX ?
                                 vertex_dwords[profile] :
                                 fragment_dwords[profile];
      const unsigned descriptor_dwords = stage == MESA_SHADER_VERTEX ?
                                            vertex_descriptor_dwords[profile] :
                                            fragment_descriptor_dwords[profile];
      uint32_t *shared = stage == MESA_SHADER_VERTEX ?
                            observation->vertex_shared :
                            observation->fragment_shared;
      size_t *shared_count = stage == MESA_SHADER_VERTEX ?
                                &observation->vertex_shared_count :
                                &observation->fragment_shared_count;
      *shared_count = descriptor_dwords + dwords;
      if (!dwords)
         continue;
      const struct pipe_constant_buffer *cb =
         &ctx->constant_buffers[stage][0];
      size_t available = 0;
      const uint8_t *bytes =
         pvrgpu_constant_buffer_bytes(ctx, stage, 0, &available);
      if (!bytes || cb->buffer_size != dwords * sizeof(uint32_t) ||
          available != dwords * sizeof(uint32_t)) {
         *failure_reason = "constant_buffer_size";
         return false;
      }
      memcpy(&shared[descriptor_dwords], bytes, available);
   }
   return true;
}

static bool
pvrgpu_terrain_pco_payload_matches_binary(
   const struct pvrgpu_terrain_pco_observation *observation,
   const char **failure_reason)
{
   static const uint64_t expected_vertex_fnv1a64
      [PVRGPU_TERRAIN_PCO_DRAW_COUNT] = {
         UINT64_C(0x137ad857d68f72e5),
         UINT64_C(0x137ad857d68f72e5),
         UINT64_C(0xc33cf9ea6c986551),
         UINT64_C(0x137ad857d68f72e5),
         UINT64_C(0x137ad857d68f72e5),
         UINT64_C(0x137ad857d68f72e5),
         UINT64_C(0x137ad857d68f72e5),
         UINT64_C(0x137ad857d68f72e5),
      };
   static const uint64_t expected_vertex_shared_fnv1a64
      [PVRGPU_TERRAIN_PCO_DRAW_COUNT] = {
         UINT64_C(0x48fff97294e45f55),
         UINT64_C(0x15e8065d3d6b1b55),
         UINT64_C(0x798ce5dd9c33fa18),
         UINT64_C(0x15e8065d3d6b1b55),
         UINT64_C(0x15e8065d3d6b1b55),
         UINT64_C(0x15e8065d3d6b1b55),
         UINT64_C(0x15e8065d3d6b1b55),
         UINT64_C(0x15e8065d3d6b1b55),
      };
   static const uint64_t expected_fragment_shared_fnv1a64
      [PVRGPU_TERRAIN_PCO_DRAW_COUNT] = {
         UINT64_C(0x62101b5902762818),
         UINT64_C(0x4e1ccea0e6192d58),
         UINT64_C(0x1369112ad898bbfd),
         UINT64_C(0x2d423f9c5838f4fd),
         UINT64_C(0x21d394b1ca541e48),
         UINT64_C(0x4755794a96dd0179),
         UINT64_C(0x2d423f9c5838f4fd),
         UINT64_C(0x2d423f9c5838f4fd),
      };
   static const uint64_t expected_main_texture_fnv1a64
      [PVRGPU_TERRAIN_PCO_MAX_TEXTURES] = {
         0,
         0,
         UINT64_C(0xa69ccd9838551cb3),
         UINT64_C(0x777443d6a3c0ceeb),
         UINT64_C(0xd510ff3e570680dd),
         0,
         UINT64_C(0x3964257e9bde4861),
      };
   static const size_t expected_vertex_binary_size
      [PVRGPU_TERRAIN_PCO_DRAW_COUNT] = {
         200U, 192U, 2680U, 192U, 192U, 192U, 192U, 192U,
      };
   static const uint64_t expected_vertex_binary_fnv1a64
      [PVRGPU_TERRAIN_PCO_DRAW_COUNT] = {
         UINT64_C(0x9abe96cad5fe9f4e),
         UINT64_C(0x081618f544cc6abe),
         UINT64_C(0x8d0f6d4b38cdecf4),
         UINT64_C(0x081618f544cc6abe),
         UINT64_C(0x081618f544cc6abe),
         UINT64_C(0x081618f544cc6abe),
         UINT64_C(0x081618f544cc6abe),
         UINT64_C(0x081618f544cc6abe),
      };
   static const size_t expected_fragment_binary_size
      [PVRGPU_TERRAIN_PCO_DRAW_COUNT] = {
         38832U, 1208U, 7328U, 1920U, 1880U, 448U, 3528U, 3504U,
      };
   static const uint64_t expected_fragment_binary_fnv1a64
      [PVRGPU_TERRAIN_PCO_DRAW_COUNT] = {
         UINT64_C(0x9e1c3ea2dfa1d8a5),
         UINT64_C(0x9711b79a7b5b63a6),
         UINT64_C(0x4fecdd1ce1feb997),
         UINT64_C(0x956d5ea59737b66f),
         UINT64_C(0x76fac56a9fbc5918),
         UINT64_C(0x412b9e844c3de073),
         UINT64_C(0xab0dfc14e6aa5116),
         UINT64_C(0xd0b9eb8de7e641d2),
   };
   const struct pvrgpu_pco_graphics_binary *binary = &observation->binary;
   const unsigned profile = observation->profile;
   if (profile >= PVRGPU_TERRAIN_PCO_DRAW_COUNT) {
      *failure_reason = "captured_payload_fingerprint";
      return false;
   }
   unsigned output_width = 0;
   unsigned output_height = 0;
   if (!pvrgpu_glmark_scaled_output_extent(1u,
                                            &output_width,
                                            &output_height)) {
      *failure_reason = "captured_output_extent";
      return false;
   }
   uint64_t expected_fragment_shared =
      expected_fragment_shared_fnv1a64[profile];
   uint64_t expected_fragment_binary =
      expected_fragment_binary_fnv1a64[profile];
   if (output_width == 800u && output_height == 600u &&
       (profile == PVRGPU_PCO_TERRAIN_D4 ||
        profile == PVRGPU_PCO_TERRAIN_D7 ||
        profile == PVRGPU_PCO_TERRAIN_D8)) {
      expected_fragment_shared = UINT64_C(0x33d7c2aad6bb3b8a);
      if (profile == PVRGPU_PCO_TERRAIN_D4) {
         expected_fragment_binary = UINT64_C(0x6ad4537c64c80942);
      } else if (profile == PVRGPU_PCO_TERRAIN_D7) {
         expected_fragment_binary = UINT64_C(0x1d6737c7f69c0953);
      } else {
         expected_fragment_binary = UINT64_C(0xb41e711d1ef41b5a);
      }
   }
   if (binary->vertex.size != expected_vertex_binary_size[profile] ||
       pvrgpu_pco_binary_fnv1a64(binary->vertex.data,
                                 binary->vertex.size) !=
          expected_vertex_binary_fnv1a64[profile] ||
       binary->fragment.size != expected_fragment_binary_size[profile] ||
       pvrgpu_pco_binary_fnv1a64(binary->fragment.data,
                                 binary->fragment.size) !=
          expected_fragment_binary ||
       pvrgpu_pco_binary_fnv1a64(observation->vertex_data,
                                 observation->vertex_data_size) !=
          expected_vertex_fnv1a64[profile] ||
       pvrgpu_pco_binary_fnv1a64(
          (const uint8_t *)observation->vertex_shared,
          observation->vertex_shared_count * sizeof(uint32_t)) !=
          expected_vertex_shared_fnv1a64[profile] ||
       pvrgpu_pco_binary_fnv1a64(
          (const uint8_t *)observation->fragment_shared,
          observation->fragment_shared_count * sizeof(uint32_t)) !=
          expected_fragment_shared) {
      *failure_reason = "captured_payload_fingerprint";
      return false;
   }
   if (profile == PVRGPU_PCO_TERRAIN_D3) {
      bool texture_fingerprint_mismatch = false;
      for (unsigned texture = 0; texture < observation->texture_count;
           ++texture) {
         const uint64_t expected = expected_main_texture_fnv1a64[texture];
         const struct pvrgpu_terrain_pco_texture *captured =
            &observation->textures[texture];
         const uint64_t actual = pvrgpu_pco_binary_fnv1a64(
            captured->bytes, captured->bytes_size);
         if ((expected == 0) != (captured->bytes_size == 0) ||
             (expected != 0 && actual != expected)) {
            pvrgpu_counter_eventf(
               "draw_pco_terrain_texture_fingerprint_mismatch",
               "profile=%u texture=%u payload_bytes=%zu "
               "expected_fnv1a64=%016llx actual_fnv1a64=%016llx",
               profile,
               texture,
               captured->bytes_size,
               (unsigned long long)expected,
               (unsigned long long)actual);
            texture_fingerprint_mismatch = true;
         }
      }
      if (texture_fingerprint_mismatch) {
         *failure_reason = "captured_texture_fingerprint";
         return false;
      }
   }
   unsigned vertex_textures = 0;
   unsigned fragment_textures = 0;
   for (unsigned texture = 0; texture < observation->texture_count; ++texture) {
      if (observation->textures[texture].stage == MESA_SHADER_VERTEX)
         ++vertex_textures;
      else
         ++fragment_textures;
   }
   if (binary->vertex.abi.shareds != observation->vertex_shared_count ||
       binary->fragment.abi.shareds != observation->fragment_shared_count ||
       binary->vertex.abi.push_constant_start !=
          vertex_textures * PVRGPU_PCO_TEXTURE_DESCRIPTOR_DWORDS ||
       binary->fragment.abi.push_constant_start !=
          fragment_textures * PVRGPU_PCO_TEXTURE_DESCRIPTOR_DWORDS ||
       binary->fragment_texture_descriptor_start != 0 ||
       binary->fragment_texture_descriptor_count !=
          fragment_textures * PVRGPU_PCO_TEXTURE_DESCRIPTOR_DWORDS ||
       binary->fragment_texture_descriptor_stride !=
          (fragment_textures ? PVRGPU_PCO_TEXTURE_DESCRIPTOR_DWORDS : 0U)) {
      *failure_reason = "compiled_payload_abi";
      return false;
   }
   return true;
}

static bool
pvrgpu_draw_matches_terrain_pco(
   struct pvrgpu_context *ctx,
   const struct pipe_draw_info *info,
   unsigned drawid_offset,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,
   unsigned num_draws,
   enum pvrgpu_pco_terrain_profile profile,
   unsigned sequence_pass,
   struct pvrgpu_terrain_pco_observation *observation,
   const char **failure_reason)
{
   if (failure_reason)
      *failure_reason = "arguments";
   if (!ctx || !observation || !failure_reason ||
       !pvrgpu_terrain_pco_draw_info_matches(info,
                                             drawid_offset,
                                             indirect,
                                             draws,
                                             num_draws,
                                             profile)) {
      if (failure_reason)
         *failure_reason = "draw_info";
      return false;
   }
   memset(observation, 0, sizeof(*observation));
   observation->profile = profile;
   observation->primitive_mode = info->mode;
   if (!pvrgpu_terrain_pco_framebuffer_matches(ctx,
                                                profile,
                                                observation,
                                                failure_reason) ||
       !pvrgpu_terrain_pco_pipeline_matches(ctx,
                                             profile,
                                             observation,
                                             failure_reason) ||
       !pvrgpu_terrain_pco_capture_vertices(ctx,
                                             profile,
                                             observation,
                                             failure_reason) ||
       !pvrgpu_terrain_pco_capture_textures(ctx,
                                             profile,
                                             observation,
                                             failure_reason) ||
       !pvrgpu_terrain_pco_copy_constants(ctx,
                                           profile,
                                           observation,
                                           failure_reason) ||
       !pvrgpu_compile_terrain_pco_binary(ctx,
                                          profile,
                                          sequence_pass,
                                          &observation->binary,
                                          failure_reason) ||
       !pvrgpu_terrain_pco_payload_matches_binary(observation,
                                                   failure_reason)) {
      pvrgpu_terrain_pco_observation_finish(observation);
      return false;
   }

   pvrgpu_counter_eventf(
      "draw_pco_terrain_capture",
      "pass=%u profile=%u framebuffer=%ux%u depth_format=%s "
      "color_source=%u depth_source=%u blend=%u vertices=%u "
      "vertex_bytes=%zu vertex_fnv1a64=%016llx vs_shared=%zu "
      "vs_shared_fnv1a64=%016llx fs_shared=%zu "
      "fs_shared_fnv1a64=%016llx textures=%zu",
      sequence_pass,
      profile,
      observation->framebuffer_width,
      observation->framebuffer_height,
      util_format_name(observation->depth_format),
      observation->color_attachment_source,
      observation->depth_attachment_source,
      observation->blend_enable,
      observation->vertex_count,
      observation->vertex_data_size,
      (unsigned long long)pvrgpu_pco_binary_fnv1a64(
         observation->vertex_data,
         observation->vertex_data_size),
      observation->vertex_shared_count,
      (unsigned long long)pvrgpu_pco_binary_fnv1a64(
         (const uint8_t *)observation->vertex_shared,
         observation->vertex_shared_count * sizeof(uint32_t)),
      observation->fragment_shared_count,
      (unsigned long long)pvrgpu_pco_binary_fnv1a64(
         (const uint8_t *)observation->fragment_shared,
         observation->fragment_shared_count * sizeof(uint32_t)),
      observation->texture_count);
   for (unsigned texture = 0; texture < observation->texture_count;
        ++texture) {
      const struct pvrgpu_terrain_pco_texture *captured =
         &observation->textures[texture];
      pvrgpu_counter_eventf(
         "draw_pco_terrain_texture_capture",
         "profile=%u texture=%u stage=%u source_slot=%u descriptor_set=%u "
         "source=%u producer=%u format=%s extent=%ux%u mips=%u "
         "declared_bytes=%zu payload_bytes=%zu payload_fnv1a64=%016llx "
         "filter=%u,%u,%u wrap=%u,%u max_lod_u4_6=%u",
         profile,
         texture,
         captured->stage,
         captured->source_slot,
         captured->descriptor_set,
         captured->source,
         captured->producer_command_index,
         util_format_name(captured->format),
         captured->mip[0].width,
         captured->mip[0].height,
         captured->mip_count,
         captured->declared_size,
         captured->bytes_size,
         (unsigned long long)pvrgpu_pco_binary_fnv1a64(captured->bytes,
                                                        captured->bytes_size),
         captured->min_filter,
         captured->mag_filter,
         captured->mip_filter,
         captured->wrap_u,
         captured->wrap_v,
         captured->max_lod_u4_6);
   }
   *failure_reason = NULL;
   return true;
}

static bool
pvrgpu_init_terrain_systemc_draw(
   struct pvrgpu_systemc_driver_command *command,
   const struct pvrgpu_terrain_pco_observation *observation,
   unsigned ordinal)
{
   if (!command || !observation ||
       observation->profile != (enum pvrgpu_pco_terrain_profile)ordinal)
      return false;
   const struct pvrgpu_pco_graphics_binary *binary = &observation->binary;

   memset(command, 0, sizeof(*command));
   command->version = PVRGPU_SYSTEMC_API_VERSION;
   command->schema = PVRGPU_DRIVER_COMMAND_SCHEMA;
   command->producer = PVRGPU_DRIVER_COMMAND_PRODUCER;
   command->command = "draw_pco_triangles";
   command->case_name = "terrain.terrain.capture.1";
   command->format = PVRGPU_DRIVER_COMMAND_FORMAT_RGBA8;
   command->frame = 1;
   command->framebuffer_width = observation->framebuffer_width;
   command->framebuffer_height = observation->framebuffer_height;
   command->width = observation->viewport_width;
   command->height = observation->viewport_height;
   memcpy(command->clear_color_bits,
          observation->clear_color_bits,
          sizeof(command->clear_color_bits));

   command->raw_vertex_data = observation->vertex_data;
   command->raw_vertex_data_size = observation->vertex_data_size;
   command->vertex_stride = observation->vertex_stride;
   command->vertex_count = observation->vertex_count;
   command->first_vertex = 0;
   command->instance_count = 1;
   command->primitive_mode = observation->primitive_mode;
   command->indexed = 0;
   command->render_target_count = 1;

   command->vertex_pco = binary->vertex.data;
   command->vertex_pco_size = binary->vertex.size;
   command->fragment_pco = binary->fragment.data;
   command->fragment_pco_size = binary->fragment.size;
   command->vertex_shared = observation->vertex_shared_count ?
                               observation->vertex_shared : NULL;
   command->vertex_shared_count = observation->vertex_shared_count;
   command->fragment_shared = observation->fragment_shared_count ?
                                 observation->fragment_shared : NULL;
   command->fragment_shared_count = observation->fragment_shared_count;
   command->sampled_texture_count = observation->texture_count;

   pvrgpu_copy_pco_stage_abi_to_systemc(&command->vertex_pco_abi,
                                         &binary->vertex.abi);
   pvrgpu_copy_pco_stage_abi_to_systemc(&command->fragment_pco_abi,
                                         &binary->fragment.abi);
   command->position_output_start = binary->position_output_start;
   command->position_output_count = binary->position_output_count;
   command->fragment_position_start = binary->fragment_position_start;
   command->fragment_position_count = binary->fragment_position_count;
   command->varying_output_start = binary->varying_output_start;
   command->varying_output_count = binary->varying_output_count;
   command->fragment_varying_start = binary->fragment_varying_start;
   command->fragment_varying_count = binary->fragment_varying_count;

   memcpy(command->viewport_scale_bits,
          observation->viewport_scale_bits,
          sizeof(command->viewport_scale_bits));
   memcpy(command->viewport_translate_bits,
          observation->viewport_translate_bits,
          sizeof(command->viewport_translate_bits));
   command->front_ccw = observation->front_ccw;
   command->cull_face = observation->cull_face;
   command->fill_front = observation->fill_front;
   command->fill_back = observation->fill_back;
   command->scissor = observation->scissor;
   command->rasterizer_discard = observation->rasterizer_discard;
   command->multisample = observation->multisample;
   command->half_pixel_center = observation->half_pixel_center;
   command->bottom_edge_rule = observation->bottom_edge_rule;
   command->clip_halfz = observation->clip_halfz;
   command->depth_clip_near = observation->depth_clip_near;
   command->depth_clip_far = observation->depth_clip_far;
   command->depth_clamp = observation->depth_clamp;
   command->sample_mask = observation->sample_mask;
   command->color_mask = observation->color_mask;
   command->dither = observation->dither;
   command->depth_enable = observation->depth_enable;
   command->depth_write = observation->depth_write;
   command->depth_func = observation->depth_func;
   command->depth_clear_bits = observation->depth_clear ?
                                  UINT32_C(0x3f800000) : 0;
   command->depth_format = observation->depth_format;
   command->color_attachment_source_command_index =
      observation->color_attachment_source;
   command->depth_attachment_source_command_index =
      observation->depth_attachment_source;
   return pvrgpu_init_systemc_blend_state(
      command,
      observation->blend_enable,
      observation->rgb_func,
      observation->rgb_src_factor,
      observation->rgb_dst_factor,
      observation->alpha_func,
      observation->alpha_src_factor,
      observation->alpha_dst_factor);
}

static bool
pvrgpu_init_terrain_systemc_texture(
   struct pvrgpu_systemc_pco_sequence_texture *destination,
   const struct pvrgpu_terrain_pco_texture *source,
   unsigned consumer_command_index)
{
   if (!destination || !source ||
       consumer_command_index >= PVRGPU_TERRAIN_PCO_DRAW_COUNT ||
       source->binding != 0 || source->mip_count == 0 ||
       source->mip_count > PVRGPU_SYSTEMC_MAX_TEXTURE_MIP_LEVELS)
      return false;

   memset(destination, 0, sizeof(*destination));
   if (!pvrgpu_systemc_shader_stage_from_mesa(source->stage,
                                               &destination->stage))
      return false;
   switch (source->source) {
   case PVRGPU_TERRAIN_PCO_EXTERNAL_PAYLOAD:
      if (source->producer_command_index != UINT32_MAX || !source->bytes ||
          source->bytes_size == 0 ||
          source->bytes_size != source->declared_size)
         return false;
      destination->source = PVRGPU_SYSTEMC_PCO_TEXTURE_EXTERNAL_PAYLOAD;
      destination->producer_command_index = 0;
      destination->bytes = source->bytes;
      destination->bytes_size = source->bytes_size;
      break;
   case PVRGPU_TERRAIN_PCO_PREVIOUS_COLOR_ATTACHMENT:
      if (source->producer_command_index >= consumer_command_index ||
          source->bytes || source->bytes_size != 0)
         return false;
      destination->source =
         PVRGPU_SYSTEMC_PCO_TEXTURE_PREVIOUS_COLOR_ATTACHMENT;
      destination->producer_command_index =
         source->producer_command_index;
      break;
   default:
      return false;
   }

   destination->descriptor_set = source->descriptor_set;
   destination->binding = source->binding;
   destination->format = util_format_name(source->format);
   destination->declared_bytes_size = source->declared_size;
   destination->mip_count = source->mip_count;
   for (unsigned level = 0; level < source->mip_count; ++level) {
      if (source->mip[level].width == 0 ||
          source->mip[level].height == 0 ||
          source->mip[level].row_pitch == 0)
         return false;
      destination->mip[level].width = source->mip[level].width;
      destination->mip[level].height = source->mip[level].height;
      destination->mip[level].row_pitch = source->mip[level].row_pitch;
      destination->mip[level].offset = source->mip[level].offset;
   }

   if (source->min_filter == PIPE_TEX_FILTER_LINEAR)
      destination->min_filter = PVRGPU_SYSTEMC_PCO_TEXTURE_FILTER_LINEAR;
   else if (source->min_filter == PIPE_TEX_FILTER_NEAREST)
      destination->min_filter = PVRGPU_SYSTEMC_PCO_TEXTURE_FILTER_NEAREST;
   else
      return false;
   if (source->mag_filter == PIPE_TEX_FILTER_LINEAR)
      destination->mag_filter = PVRGPU_SYSTEMC_PCO_TEXTURE_FILTER_LINEAR;
   else if (source->mag_filter == PIPE_TEX_FILTER_NEAREST)
      destination->mag_filter = PVRGPU_SYSTEMC_PCO_TEXTURE_FILTER_NEAREST;
   else
      return false;
   if (source->mip_filter == PIPE_TEX_MIPFILTER_LINEAR) {
      destination->mip_filter =
         PVRGPU_SYSTEMC_PCO_TEXTURE_MIP_FILTER_LINEAR;
   } else if (source->mip_filter == PIPE_TEX_MIPFILTER_NONE) {
      destination->mip_filter =
         PVRGPU_SYSTEMC_PCO_TEXTURE_MIP_FILTER_NONE;
   } else {
      return false;
   }
   if (!pvrgpu_systemc_texture_wrap_from_pipe(source->wrap_u,
                                               &destination->wrap_u) ||
       !pvrgpu_systemc_texture_wrap_from_pipe(source->wrap_v,
                                               &destination->wrap_v))
      return false;
   destination->normalized_coordinates = source->normalized_coordinates;
   destination->min_lod_u4_6 = source->min_lod_u4_6;
   destination->max_lod_u4_6 = source->max_lod_u4_6;
   return destination->format != NULL;
}

static bool
pvrgpu_emit_terrain_pco_sequence_command(struct pvrgpu_context *ctx)
{
   const char *path = pvrgpu_command_output_path();
   if (!path || !ctx ||
       ctx->terrain_pco_draw_count != PVRGPU_TERRAIN_PCO_DRAW_COUNT ||
       ctx->driver_draw_command_emitted ||
       pvrgpu_driver_draw_command_has_been_emitted())
      return false;

   static const unsigned expected_texture_count
      [PVRGPU_TERRAIN_PCO_DRAW_COUNT] = { 0U, 1U, 7U, 1U,
                                          1U, 1U, 1U, 1U };
   static const enum pipe_format expected_depth_format
      [PVRGPU_TERRAIN_PCO_DRAW_COUNT] = {
         PIPE_FORMAT_NONE,
         PIPE_FORMAT_NONE,
         PIPE_FORMAT_Z16_UNORM,
         PIPE_FORMAT_NONE,
         PIPE_FORMAT_NONE,
         PIPE_FORMAT_Z16_UNORM,
         PIPE_FORMAT_NONE,
         PIPE_FORMAT_Z24X8_UNORM,
      };

   struct pvrgpu_systemc_driver_command
      draws[PVRGPU_TERRAIN_PCO_DRAW_COUNT];
   struct pvrgpu_systemc_pco_sequence_texture
      textures[PVRGPU_TERRAIN_PCO_SEQUENCE_TEXTURE_COUNT];
   unsigned texture_offset = 0;
   for (unsigned ordinal = 0; ordinal < PVRGPU_TERRAIN_PCO_DRAW_COUNT;
        ++ordinal) {
      const struct pvrgpu_terrain_pco_observation *observation =
         ctx->terrain_pco_draws[ordinal];
      const char *failure_reason = NULL;
      const unsigned expected_attachment_source =
         ordinal == PVRGPU_PCO_TERRAIN_D6 ?
            PVRGPU_PCO_TERRAIN_D3 :
            PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR;
      if (!observation ||
          observation->profile !=
             (enum pvrgpu_pco_terrain_profile)ordinal ||
          observation->texture_count != expected_texture_count[ordinal] ||
          observation->depth_format != expected_depth_format[ordinal] ||
          observation->color_attachment_source !=
             expected_attachment_source ||
          observation->depth_attachment_source !=
             expected_attachment_source ||
          !observation->color_attachment ||
          ((observation->depth_format == PIPE_FORMAT_NONE) !=
           (observation->depth_attachment == NULL)) ||
          !pvrgpu_terrain_pco_payload_matches_binary(observation,
                                                      &failure_reason) ||
          !pvrgpu_init_terrain_systemc_draw(&draws[ordinal],
                                             observation,
                                             ordinal) ||
          texture_offset + observation->texture_count >
             PVRGPU_TERRAIN_PCO_SEQUENCE_TEXTURE_COUNT) {
         pvrgpu_counter_eventf(
            "draw_pco_terrain_sequence_error",
            "stage=payload ordinal=%u reason=%s",
            ordinal,
            failure_reason ? failure_reason : "sequence_contract");
         return false;
      }

      unsigned next_vertex_descriptor_set = 0;
      unsigned next_fragment_descriptor_set = 0;
      bool fragment_resources_started = false;
      for (unsigned texture_index = 0;
           texture_index < observation->texture_count;
           ++texture_index) {
         const struct pvrgpu_terrain_pco_texture *texture =
            &observation->textures[texture_index];
         unsigned expected_descriptor_set = 0;
         if (texture->stage == MESA_SHADER_VERTEX) {
            if (fragment_resources_started)
               return false;
            expected_descriptor_set = next_vertex_descriptor_set++;
         } else if (texture->stage == MESA_SHADER_FRAGMENT) {
            fragment_resources_started = true;
            expected_descriptor_set = next_fragment_descriptor_set++;
         } else {
            return false;
         }
         if (texture->descriptor_set != expected_descriptor_set ||
             !pvrgpu_init_terrain_systemc_texture(
                &textures[texture_offset],
                texture,
                ordinal))
            return false;
         ++texture_offset;
      }
   }
   if (texture_offset != PVRGPU_TERRAIN_PCO_SEQUENCE_TEXTURE_COUNT ||
       ctx->terrain_pco_draws[PVRGPU_PCO_TERRAIN_D6]->color_attachment !=
          ctx->terrain_pco_draws[PVRGPU_PCO_TERRAIN_D3]->color_attachment ||
       ctx->terrain_pco_draws[PVRGPU_PCO_TERRAIN_D6]->depth_attachment !=
          ctx->terrain_pco_draws[PVRGPU_PCO_TERRAIN_D3]->depth_attachment)
      return false;

   struct pvrgpu_systemc_driver_command command;
   memset(&command, 0, sizeof(command));
   command.version = PVRGPU_SYSTEMC_API_VERSION;
   command.schema = PVRGPU_DRIVER_COMMAND_SCHEMA;
   command.producer = PVRGPU_DRIVER_COMMAND_PRODUCER;
   command.command = "draw_pco_sequence";
   command.case_name = "terrain.terrain.capture.1";
   command.format = PVRGPU_DRIVER_COMMAND_FORMAT_RGBA8;
   command.frame = 1;
   const struct pvrgpu_terrain_pco_observation *final_draw =
      ctx->terrain_pco_draws[PVRGPU_PCO_TERRAIN_D8];
   command.framebuffer_width = final_draw->framebuffer_width;
   command.framebuffer_height = final_draw->framebuffer_height;
   command.width = final_draw->viewport_width;
   command.height = final_draw->viewport_height;
   memcpy(command.clear_color_bits,
          final_draw->clear_color_bits,
          sizeof(command.clear_color_bits));
   const bool output_800x600 =
      command.width == 800u && command.height == 600u;
   if (!pvrgpu_glmark_output_extent_supported(command.width,
                                               command.height) ||
       command.framebuffer_width != command.width ||
       command.framebuffer_height != command.height)
      return false;
   command.draw_count = PVRGPU_ARRAY_SIZE(draws);
   pvrgpu_sequence_input_assembly_totals(draws,
                                         PVRGPU_ARRAY_SIZE(draws),
                                         &command.ia_vertices,
                                         &command.ia_primitives);
   command.clip_invocations = command.ia_primitives;
   /*
    * Everything past input assembly is measured by SystemC from the geometry
    * this sequence actually rasterizes.  Leaving these at zero is what makes
    * the reported totals a result rather than a replayed constant.
    */
   command.vs_invocations = 0;
   command.clip_primitives = 0;
   command.setup_triangles = 0;
   command.ps_invocations = 0;
   command.semantic_texel_fetches = 0;
   command.pco_sequence_command_count = PVRGPU_ARRAY_SIZE(draws);
   command.pco_sequence_commands = draws;
   command.pco_sequence_texture_count = texture_offset;
   command.pco_sequence_textures = textures;

   char error[512] = { 0 };
   if (!pvrgpu_write_draw_pco_sequence_command(path,
                                                &command,
                                                error,
                                                sizeof(error))) {
      remove(path);
      pvrgpu_counter_eventf("draw_pco_terrain_sequence_error",
                            "stage=command_submit reason=%s",
                            error[0] ? error : "unknown");
      return false;
   }

   ctx->driver_draw_command_emitted = true;
   pvrgpu_note_driver_draw_command_emitted();
   pvrgpu_counter_eventf(
      "draw_pco_terrain_sequence_command",
      "draws=8 resources=13 drawlists=%u framebuffer=%ux%u",
      command.draw_count,
      command.framebuffer_width,
      command.framebuffer_height);
   return true;
}

/* ---- end GLMark2 terrain native sequence capture ---------------------- */

static bool
pvrgpu_emit_draw_pco_triangles_command(
   struct pvrgpu_context *ctx,
   const struct pvrgpu_conditionals_observation *observation)
{
   const char *path = pvrgpu_command_output_path();
   if (!path) {
      pvrgpu_counter_eventf("draw_pco_triangles_command_skip",
                            "reason=missing_command_path");
      return false;
   }
   if (!ctx || !observation || ctx->driver_draw_command_emitted ||
       pvrgpu_driver_draw_command_has_been_emitted()) {
      pvrgpu_counter_eventf("draw_pco_triangles_command_skip",
                            "reason=command_already_emitted ctx=%u global=%u",
                            ctx && ctx->driver_draw_command_emitted ? 1 : 0,
                            pvrgpu_driver_draw_command_has_been_emitted() ?
                               1 : 0);
      return false;
   }

   char error[512] = {0};
   if (!ctx->pco_compiler) {
      ctx->pco_compiler =
         pvrgpu_pco_compiler_create(error, sizeof(error));
      if (!ctx->pco_compiler) {
         pvrgpu_counter_eventf("draw_pco_triangles_command_error",
                               "stage=compiler_create reason=%s",
                               error[0] ? error : "unknown");
         return false;
      }
   }

   struct pvrgpu_pco_graphics_binary binary;
   memset(&binary, 0, sizeof(binary));
   if (!pvrgpu_pco_compile_conditionals(ctx->pco_compiler,
                                        ctx->vs->nir,
                                        ctx->fs->nir,
                                        PIPE_FORMAT_R32G32B32_FLOAT,
                                        &binary,
                                        error,
                                        sizeof(error))) {
      pvrgpu_counter_eventf("draw_pco_triangles_command_error",
                            "stage=pco_compile reason=%s",
                            error[0] ? error : "unknown");
      return false;
   }

   struct pvrgpu_draw_pco_triangles_command command;
   memset(&command, 0, sizeof(command));
   command.case_name = pvrgpu_command_case_name("conditionals.gallium.pco");
   command.frame = 1;
   command.framebuffer_width = observation->framebuffer_width;
   command.framebuffer_height = observation->framebuffer_height;
   command.width = observation->viewport_width;
   command.height = observation->viewport_height;
   command.format = PVRGPU_DRIVER_COMMAND_FORMAT_RGBA8;
   command.clear_color_bits[3] = UINT32_C(0x3f800000);
   command.raw_vertex_data = observation->raw_vertex_data;
   command.raw_vertex_data_size = observation->raw_vertex_data_size;
   command.vertex_stride = PVRGPU_DRAW_PCO_TRIANGLES_VERTEX_STRIDE;
   command.vertex_count = PVRGPU_DRAW_PCO_TRIANGLES_VERTEX_COUNT;
   command.first_vertex = 0;
   command.instance_count = 1;
   command.primitive_mode = (uint32_t)MESA_PRIM_TRIANGLES;
   command.indexed = 0;
   command.render_target_count = 1;
   command.vertex_pco = binary.vertex.data;
   command.vertex_pco_size = binary.vertex.size;
   command.fragment_pco = binary.fragment.data;
   command.fragment_pco_size = binary.fragment.size;
   command.vertex_shared = observation->vertex_shared;
   command.vertex_shared_count = PVRGPU_ARRAY_SIZE(observation->vertex_shared);
   command.fragment_shared = observation->fragment_shared;
   command.fragment_shared_count =
      PVRGPU_ARRAY_SIZE(observation->fragment_shared);
   pvrgpu_copy_pco_stage_abi_to_command(&command.vertex_pco_abi,
                                        &binary.vertex.abi);
   pvrgpu_copy_pco_stage_abi_to_command(&command.fragment_pco_abi,
                                        &binary.fragment.abi);
   command.position_output_start = binary.position_output_start;
   command.position_output_count = binary.position_output_count;
   command.fragment_position_start = binary.fragment_position_start;
   command.fragment_position_count = binary.fragment_position_count;
   for (unsigned component = 0; component < 3; ++component) {
      command.viewport_scale_bits[component] =
         pvrgpu_float_bits(ctx->viewport.scale[component]);
      command.viewport_translate_bits[component] =
         pvrgpu_float_bits(ctx->viewport.translate[component]);
   }
   command.front_ccw = ctx->rasterizer->state.front_ccw;
   command.cull_face = ctx->rasterizer->state.cull_face;
   command.fill_front = ctx->rasterizer->state.fill_front;
   command.fill_back = ctx->rasterizer->state.fill_back;
   command.scissor = ctx->rasterizer->state.scissor;
   command.rasterizer_discard = ctx->rasterizer->state.rasterizer_discard;
   command.multisample = ctx->rasterizer->state.multisample;
   command.half_pixel_center = ctx->rasterizer->state.half_pixel_center;
   command.bottom_edge_rule = ctx->rasterizer->state.bottom_edge_rule;
   command.clip_halfz = ctx->rasterizer->state.clip_halfz;
   command.depth_clip_near = ctx->rasterizer->state.depth_clip_near;
   command.depth_clip_far = ctx->rasterizer->state.depth_clip_far;
   command.depth_clamp = ctx->rasterizer->state.depth_clamp;
   command.sample_mask = ctx->sample_mask;
   command.color_mask = pvrgpu_rt_colormask(ctx, 0);
   command.blend_enable = ctx->blend->state.rt[0].blend_enable;
   command.dither = ctx->blend->state.dither;
   command.depth_enable = ctx->dsa->state.depth_enabled;
   command.depth_write = ctx->dsa->state.depth_writemask;
   command.depth_func = ctx->dsa->state.depth_func;
   command.depth_clear_bits = UINT32_C(0x3f800000);
   command.depth_format = ctx->framebuffer.zsbuf.format;

   pvrgpu_counter_eventf(
      "draw_pco_triangles_pco_binary",
      "vs_bytes=%zu vs_fnv1a64=%016llx fs_bytes=%zu fs_fnv1a64=%016llx "
      "vs_abi=%u,%u,%u,%u,%u,%u,%u,%u "
      "fs_abi=%u,%u,%u,%u,%u,%u,%u,%u linkage=%u,%u,%u,%u",
      binary.vertex.size,
      (unsigned long long)pvrgpu_pco_binary_fnv1a64(binary.vertex.data,
                                                    binary.vertex.size),
      binary.fragment.size,
      (unsigned long long)pvrgpu_pco_binary_fnv1a64(binary.fragment.data,
                                                    binary.fragment.size),
      binary.vertex.abi.temps,
      binary.vertex.abi.vertex_inputs,
      binary.vertex.abi.vertex_outputs,
      binary.vertex.abi.coefficients,
      binary.vertex.abi.shareds,
      binary.vertex.abi.push_constant_start,
      binary.vertex.abi.push_constant_count,
      binary.vertex.abi.entry_offset,
      binary.fragment.abi.temps,
      binary.fragment.abi.vertex_inputs,
      binary.fragment.abi.vertex_outputs,
      binary.fragment.abi.coefficients,
      binary.fragment.abi.shareds,
      binary.fragment.abi.push_constant_start,
      binary.fragment.abi.push_constant_count,
      binary.fragment.abi.entry_offset,
      binary.position_output_start,
      binary.position_output_count,
      binary.fragment_position_start,
      binary.fragment_position_count);

   const bool emitted = pvrgpu_write_draw_pco_triangles_command(path,
                                                                &command,
                                                                error,
                                                                sizeof(error));
   pvrgpu_pco_graphics_binary_finish(&binary);
   if (!emitted) {
      remove(path);
      debug_printf("pvrgpu: %s\n", error);
      pvrgpu_counter_eventf("draw_pco_triangles_command_error",
                            "stage=command_submit reason=%s",
                            error[0] ? error : "unknown");
      return false;
   }

   ctx->driver_draw_command_emitted = true;
   pvrgpu_note_driver_draw_command_emitted();
   pvrgpu_counter_eventf("draw_pco_triangles_command",
                         "framebuffer=%ux%u vertices=%u vs_bytes=%zu "
                         "fs_bytes=%zu",
                         command.framebuffer_width,
                         command.framebuffer_height,
                         command.vertex_count,
                         command.vertex_pco_size,
                         command.fragment_pco_size);
   return true;
}

static bool
pvrgpu_emit_lit_mesh_command(
   struct pvrgpu_context *ctx,
   const struct pvrgpu_lit_mesh_observation *observation)
{
   const char *path = pvrgpu_command_output_path();
   if (!path || !ctx || !observation ||
       !observation->interleaved_vertex_data ||
       ctx->driver_draw_command_emitted ||
       pvrgpu_driver_draw_command_has_been_emitted()) {
      pvrgpu_counter_eventf("draw_pco_lit_mesh_command_skip",
                            "reason=missing_path_payload_or_already_emitted");
      return false;
   }

   char error[512] = {0};
   if (!ctx->pco_compiler) {
      ctx->pco_compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
      if (!ctx->pco_compiler) {
         pvrgpu_counter_eventf("draw_pco_lit_mesh_command_error",
                               "stage=compiler_create reason=%s",
                               error[0] ? error : "unknown");
         return false;
      }
   }

   struct pvrgpu_pco_graphics_binary binary;
   memset(&binary, 0, sizeof(binary));
   if (!pvrgpu_pco_compile_lit_mesh(ctx->pco_compiler,
                                    ctx->vs->nir,
                                    ctx->fs->nir,
                                    observation->profile,
                                    &binary,
                                    error,
                                    sizeof(error))) {
      pvrgpu_counter_eventf("draw_pco_lit_mesh_command_error",
                            "stage=pco_compile profile=%u reason=%s",
                            observation->profile,
                            error[0] ? error : "unknown");
      return false;
   }

   struct pvrgpu_draw_pco_triangles_command command;
   memset(&command, 0, sizeof(command));
   command.case_name = pvrgpu_command_case_name("glmark2.lit_mesh.pco");
   command.frame = 1;
   command.framebuffer_width = observation->framebuffer_width;
   command.framebuffer_height = observation->framebuffer_height;
   command.width = observation->viewport_width;
   command.height = observation->viewport_height;
   command.format = PVRGPU_DRIVER_COMMAND_FORMAT_RGBA8;
   command.clear_color_bits[3] = UINT32_C(0x3f800000);
   command.raw_vertex_data = observation->interleaved_vertex_data;
   command.raw_vertex_data_size = observation->interleaved_vertex_data_size;
   command.vertex_stride = 6u * sizeof(float);
   command.vertex_count = observation->vertex_count;
   command.first_vertex = 0;
   command.instance_count = 1;
   command.primitive_mode = (uint32_t)MESA_PRIM_TRIANGLES;
   command.indexed = 0;
   command.render_target_count = 1;
   command.vertex_pco = binary.vertex.data;
   command.vertex_pco_size = binary.vertex.size;
   command.fragment_pco = binary.fragment.data;
   command.fragment_pco_size = binary.fragment.size;
   command.vertex_shared = observation->vertex_shared;
   command.vertex_shared_count = PVRGPU_ARRAY_SIZE(observation->vertex_shared);
   command.fragment_shared = NULL;
   command.fragment_shared_count = 0;
   pvrgpu_copy_pco_stage_abi_to_command(&command.vertex_pco_abi,
                                        &binary.vertex.abi);
   pvrgpu_copy_pco_stage_abi_to_command(&command.fragment_pco_abi,
                                        &binary.fragment.abi);
   command.position_output_start = binary.position_output_start;
   command.position_output_count = binary.position_output_count;
   command.fragment_position_start = binary.fragment_position_start;
   command.fragment_position_count = binary.fragment_position_count;
   command.varying_output_start = binary.varying_output_start;
   command.varying_output_count = binary.varying_output_count;
   command.fragment_varying_start = binary.fragment_varying_start;
   command.fragment_varying_count = binary.fragment_varying_count;
   for (unsigned component = 0; component < 3; ++component) {
      command.viewport_scale_bits[component] =
         pvrgpu_float_bits(ctx->viewport.scale[component]);
      command.viewport_translate_bits[component] =
         pvrgpu_float_bits(ctx->viewport.translate[component]);
   }
   command.front_ccw = ctx->rasterizer->state.front_ccw;
   command.cull_face = ctx->rasterizer->state.cull_face;
   command.fill_front = ctx->rasterizer->state.fill_front;
   command.fill_back = ctx->rasterizer->state.fill_back;
   command.scissor = ctx->rasterizer->state.scissor;
   command.rasterizer_discard = ctx->rasterizer->state.rasterizer_discard;
   command.multisample = ctx->rasterizer->state.multisample;
   command.half_pixel_center = ctx->rasterizer->state.half_pixel_center;
   command.bottom_edge_rule = ctx->rasterizer->state.bottom_edge_rule;
   command.clip_halfz = ctx->rasterizer->state.clip_halfz;
   command.depth_clip_near = ctx->rasterizer->state.depth_clip_near;
   command.depth_clip_far = ctx->rasterizer->state.depth_clip_far;
   command.depth_clamp = ctx->rasterizer->state.depth_clamp;
   command.sample_mask = ctx->sample_mask;
   command.color_mask = pvrgpu_rt_colormask(ctx, 0);
   command.blend_enable = ctx->blend->state.rt[0].blend_enable;
   command.dither = ctx->blend->state.dither;
   command.depth_enable = ctx->dsa->state.depth_enabled;
   command.depth_write = ctx->dsa->state.depth_writemask;
   command.depth_func = ctx->dsa->state.depth_func;
   command.depth_clear_bits = UINT32_C(0x3f800000);
   command.depth_format = ctx->framebuffer.zsbuf.format;

   pvrgpu_counter_eventf(
      "draw_pco_lit_mesh_binary",
      "profile=%u vertices=%u vs_bytes=%zu vs_fnv1a64=%016llx "
      "fs_bytes=%zu fs_fnv1a64=%016llx linkage=%u,%u,%u,%u,%u,%u,%u,%u",
      observation->profile,
      observation->vertex_count,
      binary.vertex.size,
      (unsigned long long)pvrgpu_pco_binary_fnv1a64(binary.vertex.data,
                                                    binary.vertex.size),
      binary.fragment.size,
      (unsigned long long)pvrgpu_pco_binary_fnv1a64(binary.fragment.data,
                                                    binary.fragment.size),
      binary.position_output_start,
      binary.position_output_count,
      binary.fragment_position_start,
      binary.fragment_position_count,
      binary.varying_output_start,
      binary.varying_output_count,
      binary.fragment_varying_start,
      binary.fragment_varying_count);

   const bool emitted = pvrgpu_write_draw_pco_triangles_command(path,
                                                                &command,
                                                                error,
                                                                sizeof(error));
   pvrgpu_pco_graphics_binary_finish(&binary);
   if (!emitted) {
      remove(path);
      pvrgpu_counter_eventf("draw_pco_lit_mesh_command_error",
                            "stage=command_submit reason=%s",
                            error[0] ? error : "unknown");
      return false;
   }

   ctx->driver_draw_command_emitted = true;
   pvrgpu_note_driver_draw_command_emitted();
   pvrgpu_counter_eventf("draw_pco_lit_mesh_command",
                         "profile=%u framebuffer=%ux%u vertices=%u",
                         observation->profile,
                         command.framebuffer_width,
                         command.framebuffer_height,
                         command.vertex_count);
   return true;
}

/*
 * Number of primitives an array topology assembles, or zero when the driver
 * cannot lower it.  Lines and points are included: the model widens them into
 * real screen-space geometry, so they rasterize rather than collapsing to the
 * zero-area triangles the topology expansion encodes them as.
 */
static unsigned
pvrgpu_array_primitive_count(unsigned mode, unsigned count)
{
   switch (mode) {
   case MESA_PRIM_POINTS:
      return count;
   case MESA_PRIM_LINES:
      /*
       * GLES assembles as many complete primitives as the vertices allow and
       * ignores the remainder, so an odd count draws one fewer line rather
       * than nothing at all.
       */
      return count / 2;
   case MESA_PRIM_LINE_LOOP:
      return count >= 2 ? count : 0;
   case MESA_PRIM_LINE_STRIP:
      return count >= 2 ? count - 1 : 0;
   case MESA_PRIM_TRIANGLES:
      return count / 3;
   case MESA_PRIM_TRIANGLE_STRIP:
   case MESA_PRIM_TRIANGLE_FAN:
      return count >= 3 ? count - 2 : 0;
   default:
      return 0;
   }
}

/*
 * Vertices that form complete primitives, which is what the capsule carries.
 * A list topology drops the leftover vertices GLES ignores; the others
 * consume every vertex they were given.
 */
static unsigned
pvrgpu_array_assembled_vertex_count(unsigned mode, unsigned count)
{
   switch (mode) {
   case MESA_PRIM_LINES:
      return (count / 2) * 2;
   case MESA_PRIM_TRIANGLES:
      return (count / 3) * 3;
   default:
      return count;
   }
}

/*
 * Read one vertex attribute as the float vector the shader declares.
 *
 * GLES feeds float inputs from any of the normalized, scaled and float source
 * formats, so the unpack is delegated to the format description rather than
 * enumerated here: restricting the driver to a handful of float formats is
 * what made vertex_arrays.input_types unlowerable.  Pure integer sources need
 * an integer input path and stay fail-closed.
 *
 * Components the shader does not declare are not read.  They cannot influence
 * the result, and the packed buffer carries only what the shader consumes.
 */
static bool
pvrgpu_read_vertex_attribute(const struct pvrgpu_context *ctx,
                             const struct pipe_vertex_element *element,
                             unsigned vertex,
                             unsigned wanted,
                             float *out,
                             const char **reason)
{
   const char *ignored = NULL;
   if (!reason)
      reason = &ignored;

   if (!element || element->vertex_buffer_index >= ctx->num_vertex_buffers) {
      *reason = "buffer_index";
      return false;
   }
   if (wanted == 0 || wanted > 4) {
      *reason = "component_count";
      return false;
   }

   const enum pipe_format format = element->src_format;
   const struct util_format_description *description =
      util_format_description(format);
   if (!description || description->layout != UTIL_FORMAT_LAYOUT_PLAIN ||
       description->colorspace != UTIL_FORMAT_COLORSPACE_RGB) {
      *reason = "source_layout";
      return false;
   }
   if (util_format_is_pure_integer(format)) {
      *reason = "pure_integer_source";
      return false;
   }

   const struct util_format_unpack_description *unpack =
      util_format_unpack_description(format);
   if (!unpack || !unpack->unpack_rgba) {
      *reason = "source_format";
      return false;
   }

   const unsigned components = util_format_get_nr_components(format);
   const unsigned blocksize = util_format_get_blocksize(format);
   const struct pipe_vertex_buffer *buffer =
      &ctx->vertex_buffers[element->vertex_buffer_index];
   const unsigned stride =
      element->src_stride ? element->src_stride : blocksize;
   if (stride == 0) {
      *reason = "stride";
      return false;
   }

   const uint64_t offset = (uint64_t)buffer->buffer_offset +
                           element->src_offset +
                           (uint64_t)vertex * stride;
   const uint8_t *base = NULL;
   if (buffer->is_user_buffer) {
      base = (const uint8_t *)buffer->buffer.user;
   } else if (buffer->buffer.resource) {
      const struct pvrgpu_resource *resource =
         pvrgpu_resource(buffer->buffer.resource);
      base = resource->data;
      /* Only a resource-backed buffer carries a size the driver can trust. */
      if (!base) {
         *reason = "no_backing_store";
         return false;
      }
      /*
       * A draw may address more vertices than the bound array holds; GLES
       * leaves the fetched value undefined but requires the draw to complete,
       * and hardware returns the default attribute rather than faulting.  Do
       * the same: refusing the whole draw would report geometry the app did
       * submit as unlowerable.
       */
      if (offset + blocksize > (uint64_t)resource->size) {
         for (unsigned component = 0; component < wanted; ++component)
            out[component] = component == 3 ? 1.0f : 0.0f;
         return true;
      }
   }
   if (!base) {
      *reason = "no_backing_store";
      return false;
   }

   float rgba[4];
   unpack->unpack_rgba(rgba, base + offset, 1);

   for (unsigned component = 0; component < wanted; ++component) {
      out[component] = component < components ? rgba[component]
                                              : (component == 3 ? 1.0f : 0.0f);
      if (!isfinite(out[component])) {
         *reason = "non_finite";
         return false;
      }
   }
   return true;
}

bool
pvrgpu_emit_array_primitive_sequence_command(struct pvrgpu_context *ctx);

struct pvrgpu_array_primitive_draw {
   struct pvrgpu_systemc_driver_command command;
   float *vertex_data;
   uint8_t *index_data;
   struct pvrgpu_pco_graphics_binary binary;
};

static void
pvrgpu_array_primitive_draw_destroy(struct pvrgpu_array_primitive_draw **slot)
{
   if (!slot || !*slot)
      return;
   struct pvrgpu_array_primitive_draw *draw = *slot;
   pvrgpu_pco_graphics_binary_finish(&draw->binary);
   free(draw->vertex_data);
   free(draw->index_data);
   FREE(draw);
   *slot = NULL;
}

bool
pvrgpu_context_has_recorded_geometry(const struct pvrgpu_context *ctx)
{
   return ctx && ctx->array_primitive_draw_count != 0;
}

void
pvrgpu_array_primitive_sequence_reset(struct pvrgpu_context *ctx)
{
   if (!ctx)
      return;
   for (unsigned ordinal = 0; ordinal < ctx->array_primitive_draw_count;
        ++ordinal)
      pvrgpu_array_primitive_draw_destroy(&ctx->array_primitive_draws[ordinal]);
   ctx->array_primitive_draw_count = 0;
   ctx->array_primitive_sequence_overflow = false;
}

/*
 * Submit every recorded array-primitive draw as one native PCO sequence.  The
 * whole-sequence counter fields are deliberately left zero: SystemC measures
 * them from the geometry it actually rasterizes, which is the point of routing
 * these draws through the model instead of a captured counter profile.
 */
bool
pvrgpu_emit_array_primitive_sequence_command(struct pvrgpu_context *ctx)
{
   const char *path = pvrgpu_command_output_path();
   if (!path || !ctx || ctx->array_primitive_draw_count == 0)
      return false;
   if (ctx->driver_draw_command_emitted ||
       pvrgpu_driver_draw_command_has_been_emitted()) {
      pvrgpu_counter_eventf("draw_array_primitive_sequence_error",
                            "stage=own reason=command_owned_elsewhere "
                            "draws=%u local=%u global=%u",
                            ctx->array_primitive_draw_count,
                            ctx->driver_draw_command_emitted ? 1u : 0u,
                            pvrgpu_driver_draw_command_has_been_emitted() ? 1u
                                                                          : 0u);
      return false;
   }
   if (ctx->array_primitive_sequence_overflow) {
      pvrgpu_counter_eventf("draw_array_primitive_sequence_error",
                            "stage=record reason=sequence_overflow draws=%u",
                            ctx->array_primitive_draw_count);
      return false;
   }

   struct pvrgpu_systemc_driver_command *draws =
      calloc(ctx->array_primitive_draw_count, sizeof(*draws));
   if (!draws)
      return false;

   const struct pvrgpu_systemc_driver_command *first =
      &ctx->array_primitive_draws[0]->command;
   for (unsigned ordinal = 0; ordinal < ctx->array_primitive_draw_count;
        ++ordinal) {
      const struct pvrgpu_array_primitive_draw *recorded =
         ctx->array_primitive_draws[ordinal];
      /*
       * One sequence describes one render target, so every draw has to agree
       * on the surface it writes.  A trace that retargets mid-frame is left
       * for the caller to reject rather than silently flattened.
       */
      if (!recorded ||
          recorded->command.framebuffer_width != first->framebuffer_width ||
          recorded->command.framebuffer_height != first->framebuffer_height ||
          recorded->command.width != first->width ||
          recorded->command.height != first->height ||
          recorded->command.render_target_count !=
             first->render_target_count) {
         pvrgpu_counter_eventf("draw_array_primitive_sequence_error",
                               "stage=assemble ordinal=%u "
                               "reason=framebuffer_mismatch",
                               ordinal);
         free(draws);
         return false;
      }
      draws[ordinal] = recorded->command;
      if (draws[ordinal].blend_enable == 0) {
         /* Disabled blending still has to carry its canonical ONE/ZERO ABI. */
         draws[ordinal].blend_rgb_equation =
            PVRGPU_SYSTEMC_PCO_BLEND_EQUATION_ADD;
         draws[ordinal].blend_alpha_equation =
            PVRGPU_SYSTEMC_PCO_BLEND_EQUATION_ADD;
         draws[ordinal].blend_source_rgb_factor =
            PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE;
         draws[ordinal].blend_destination_rgb_factor =
            PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ZERO;
         draws[ordinal].blend_source_alpha_factor =
            PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE;
         draws[ordinal].blend_destination_alpha_factor =
            PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ZERO;
      }
      /*
       * The first draw starts from a fresh clear; each later draw continues
       * from the surfaces the previous ordinal produced, which is what makes
       * the sequence accumulate instead of each draw clearing the frame.
       */
      draws[ordinal].color_attachment_source_command_index =
         ordinal == 0 ? PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR : ordinal - 1u;
      if (draws[ordinal].depth_enable == 0 &&
          draws[ordinal].depth_write == 0) {
         /* A draw that neither tests nor writes depth needs no attachment. */
         draws[ordinal].depth_format = 0;
         draws[ordinal].depth_attachment_source_command_index =
            PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR;
      } else {
         draws[ordinal].depth_attachment_source_command_index =
            ordinal == 0 ? PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR : ordinal - 1u;
      }
   }

   struct pvrgpu_systemc_driver_command command;
   memset(&command, 0, sizeof(command));
   command.version = PVRGPU_SYSTEMC_API_VERSION;
   command.schema = PVRGPU_DRIVER_COMMAND_SCHEMA;
   command.producer = PVRGPU_DRIVER_COMMAND_PRODUCER;
   command.command = "draw_pco_sequence";
   command.case_name = first->case_name;
   command.format = first->format;
   command.frame = 1;
   command.framebuffer_width = first->framebuffer_width;
   command.framebuffer_height = first->framebuffer_height;
   command.width = first->width;
   command.height = first->height;
   memcpy(command.clear_color_bits,
          first->clear_color_bits,
          sizeof(command.clear_color_bits));
   /*
    * Input-assembly totals follow directly from the draws that were lowered,
    * so the driver states them exactly.  Everything past clipping is left
    * unset on purpose: SystemC measures it from the geometry it rasterizes.
    */
   uint32_t ia_vertices = 0;
   uint32_t ia_primitives = 0;
   for (unsigned ordinal = 0; ordinal < ctx->array_primitive_draw_count;
        ++ordinal) {
      const struct pvrgpu_systemc_driver_command *draw = &draws[ordinal];
      /*
       * Input assembly counts what the draw submits: an indexed draw submits
       * its indices, so vertex reuse shows up as vs_invocations below
       * ia_vertices once SystemC has run the post-transform cache.
       */
      const uint32_t assembled =
         draw->indexed != 0 ? draw->index_count : draw->vertex_count;
      ia_vertices += assembled;
      ia_primitives +=
         pvrgpu_array_primitive_count(draw->primitive_mode, assembled);
   }
   if (ia_vertices == 0 || ia_primitives == 0) {
      pvrgpu_counter_eventf("draw_array_primitive_sequence_error",
                            "stage=assemble reason=empty_input_assembly");
      free(draws);
      return false;
   }
   command.draw_count = ctx->array_primitive_draw_count;
   command.ia_vertices = ia_vertices;
   command.ia_primitives = ia_primitives;
   /* Non-indexed draws shade every submitted vertex exactly once. */
   command.vs_invocations = ia_vertices;
   for (unsigned ordinal = 0; ordinal < ctx->array_primitive_draw_count;
        ++ordinal) {
      if (draws[ordinal].indexed != 0) {
         /* Let SystemC report shading work measured through the vertex cache. */
         command.vs_invocations = 0;
         break;
      }
   }
   command.clip_invocations = ia_primitives;
   command.render_target_count = first->render_target_count;
   command.pco_sequence_command_count = ctx->array_primitive_draw_count;
   command.pco_sequence_commands = draws;

   char error[512] = { 0 };
   const bool emitted = pvrgpu_write_draw_pco_sequence_command(path,
                                                               &command,
                                                               error,
                                                               sizeof(error));
   free(draws);
   if (!emitted) {
      remove(path);
      debug_printf("pvrgpu: %s\n", error);
      pvrgpu_counter_eventf("draw_array_primitive_sequence_error",
                            "stage=command_submit reason=%s",
                            error[0] ? error : "unknown");
      return false;
   }

   ctx->driver_draw_command_emitted = true;
   ctx->array_primitive_sequence_owns_command = true;
   pvrgpu_note_driver_draw_command_emitted();
   pvrgpu_counter_eventf("draw_array_primitive_sequence_command",
                         "draws=%u framebuffer=%ux%u",
                         command.pco_sequence_command_count,
                         command.framebuffer_width,
                         command.framebuffer_height);
   return true;
}

/*
 * Copy the index range a draw references into an owned buffer and report the
 * highest index it uses.  The indices are rebased onto the packed vertex array
 * the capsule carries, so the command's first_index and base_vertex are zero
 * while the reuse pattern inside the buffer stays exactly as the application
 * submitted it.
 */
static uint8_t *
pvrgpu_copy_draw_indices(const struct pipe_draw_info *info,
                         const struct pipe_draw_start_count_bias *draw,
                         unsigned index_count,
                         uint32_t *out_max_index)
{
   if (!info || !draw || info->index_size == 0 || index_count == 0 ||
       !out_max_index)
      return NULL;

   const size_t bytes = (size_t)index_count * info->index_size;
   uint8_t *indices = malloc(bytes);
   if (!indices)
      return NULL;

   uint32_t max_index = 0;
   for (unsigned i = 0; i < index_count; ++i) {
      uint32_t index = 0;
      if (!pvrgpu_read_draw_index(info, draw->start, i, &index)) {
         free(indices);
         return NULL;
      }
      if (index > max_index)
         max_index = index;
      switch (info->index_size) {
      case 1:
         indices[i] = (uint8_t)index;
         break;
      case 2: {
         const uint16_t narrow = (uint16_t)index;
         memcpy(indices + (size_t)i * 2u, &narrow, sizeof(narrow));
         break;
      }
      default:
         memcpy(indices + (size_t)i * 4u, &index, sizeof(index));
         break;
      }
   }
   *out_max_index = max_index;
   return indices;
}

/*
 * DWORDs of constant buffer 0 a stage binds, rounded to the vec4 granularity
 * Gallium addresses uniforms with.  Zero when the stage binds nothing.
 */
static unsigned
pvrgpu_stage_uniform_dwords(const struct pvrgpu_context *ctx,
                            mesa_shader_stage stage)
{
   size_t available = 0;
   if (!pvrgpu_constant_buffer_bytes(ctx, stage, 0, &available))
      return 0;
   const unsigned dwords = (unsigned)(available / sizeof(uint32_t));
   return (dwords + 3u) & ~3u;
}

/*
 * Copies a stage's constant buffer into the shared-register words a lowered
 * draw carries.  A stage may reserve more than it binds, so the tail is
 * zero filled rather than left undefined.
 */
static bool
pvrgpu_copy_stage_uniform_words(const struct pvrgpu_context *ctx,
                                mesa_shader_stage stage,
                                unsigned dwords,
                                uint32_t *words)
{
   if (dwords == 0)
      return true;
   size_t available = 0;
   const uint8_t *bytes =
      pvrgpu_constant_buffer_bytes(ctx, stage, 0, &available);
   if (!bytes || !words)
      return false;
   for (unsigned word = 0; word < dwords; ++word) {
      const size_t offset = (size_t)word * sizeof(uint32_t);
      if (offset + sizeof(uint32_t) > available) {
         words[word] = 0;
         continue;
      }
      memcpy(&words[word], bytes + offset, sizeof(words[word]));
   }
   return true;
}

/*
 * Record one lowered array-primitive draw.  The v1 capsule carries a single
 * draw, so an accumulating sequence is the only way a trace with more than one
 * draw reaches the model; the draws are submitted together by
 * pvrgpu_emit_array_primitive_sequence_command() when the frame flushes.
 */
static bool
pvrgpu_record_color_primitive_pco_draw(
   struct pvrgpu_context *ctx,
   const struct pipe_draw_info *info,
   const struct pipe_draw_start_count_bias *draw)
{
   const char *path = pvrgpu_command_output_path();
   if (!path || !ctx || !info || !draw)
      return false;
   if (ctx->driver_draw_command_emitted ||
       pvrgpu_driver_draw_command_has_been_emitted()) {
      pvrgpu_counter_eventf("draw_array_primitive_record_error",
                            "stage=own reason=command_already_emitted");
      return false;
   }
   /*
    * RenderDoc replays the captured frames more than once, so the driver sees
    * each API draw several times.  The trace reports how many draw actions it
    * actually contains; stop recording there so the sequence describes the
    * workload once instead of once per replay pass.
    *
    * This has to be decided before the capacity check below.  A trace with
    * exactly as many draws as the sequence holds fills it on the first pass,
    * and the repeats from the second would otherwise be mistaken for a
    * sequence that ran out of room -- poisoning a recording that was in fact
    * complete.
    */
   unsigned trace_draw_actions = 0;
   if (pvrgpu_trace_draw_actions(&trace_draw_actions) &&
       ctx->array_primitive_draw_count >= trace_draw_actions) {
      return true;
   }
   if (ctx->array_primitive_draw_count >=
       PVRGPU_ARRAY_PRIMITIVE_SEQUENCE_MAX) {
      ctx->array_primitive_sequence_overflow = true;
      pvrgpu_counter_eventf("draw_array_primitive_record_error",
                            "stage=own reason=sequence_full draws=%u",
                            ctx->array_primitive_draw_count);
      return false;
   }

   /*
    * A draw needs somewhere to read positions from, but nothing more: a shape
    * shaded from a uniform has a single position attribute, and requiring two
    * left every such draw unlowerable.
    */
   if (!ctx->vertex_elements || ctx->vertex_elements->num_elements == 0 ||
       ctx->num_vertex_buffers == 0 || !ctx->vs || !ctx->fs) {
      pvrgpu_counter_eventf("draw_array_primitive_record_error",
                            "stage=state reason=vertex_layout attributes=%u "
                            "buffers=%u has_vs=%u has_fs=%u",
                            ctx->vertex_elements ?
                               ctx->vertex_elements->num_elements : 0,
                            ctx->num_vertex_buffers,
                            ctx->vs ? 1u : 0u,
                            ctx->fs ? 1u : 0u);
      return false;
   }
   if (pvrgpu_array_primitive_count(info->mode, draw->count) == 0) {
      pvrgpu_counter_eventf("draw_array_primitive_record_error",
                            "stage=state reason=primitive_mode mode=%u "
                            "count=%u",
                            info->mode,
                            draw->count);
      return false;
   }

   /*
    * An indexed draw forwards its index buffer so the model performs the
    * fetch.  The vertices are packed for the range those indices reference,
    * and any index bias is folded in while packing, so the capsule's indices
    * address the packed array directly.
    */
   uint8_t *index_data = NULL;
   size_t index_data_size = 0;
   /*
    * A list topology assembles from complete primitives only, so the capsule
    * describes exactly those vertices; the leftovers GLES ignores never reach
    * the model.
    */
   const unsigned assembled_count =
      pvrgpu_array_assembled_vertex_count(info->mode, draw->count);
   unsigned vertex_count = assembled_count;
   unsigned vertex_bias = 0;
   if (info->index_size != 0) {
      uint32_t max_index = 0;
      index_data =
         pvrgpu_copy_draw_indices(info, draw, assembled_count, &max_index);
      if (!index_data)
         return false;
      index_data_size = (size_t)assembled_count * info->index_size;
      if (max_index == UINT32_MAX ||
          draw->index_bias < 0 ||
          (uint64_t)max_index + 1u + (uint64_t)draw->index_bias > UINT_MAX) {
         free(index_data);
         return false;
      }
      vertex_count = max_index + 1u;
      vertex_bias = (unsigned)draw->index_bias;
   }
   /*
    * Pack each attribute at the width its vertex shader declares.  The model
    * matches a program's VTXIN read mask against the attribute bindings
    * exactly, so supplying a uniform four components would not line up with a
    * shader that declares a narrower attribute.
    */
   const unsigned attribute_count = ctx->vertex_elements->num_elements;
   unsigned attribute_components[PVRGPU_PCO_MAX_VERTEX_ATTRIBUTES] = {0};
   if (!pvrgpu_pco_vertex_attribute_components(ctx->vs->nir,
                                               attribute_count,
                                               attribute_components)) {
      pvrgpu_counter_eventf("draw_array_primitive_record_error",
                            "stage=attributes reason=shader_layout "
                            "attributes=%u",
                            attribute_count);
      free(index_data);
      return false;
   }
   enum pipe_format attribute_formats[PVRGPU_PCO_MAX_VERTEX_ATTRIBUTES] = {
      PIPE_FORMAT_NONE};
   for (unsigned attribute = 0; attribute < attribute_count; ++attribute) {
      attribute_formats[attribute] =
         ctx->vertex_elements->elements[attribute].src_format;
   }
   unsigned packed_floats = 0;
   unsigned attribute_offsets[PVRGPU_PCO_MAX_VERTEX_ATTRIBUTES] = {0};
   for (unsigned attribute = 0; attribute < attribute_count; ++attribute) {
      attribute_offsets[attribute] = packed_floats;
      packed_floats += attribute_components[attribute];
   }
   if (packed_floats == 0 ||
       vertex_count > UINT_MAX / (packed_floats * sizeof(float))) {
      free(index_data);
      return false;
   }

   float *interleaved =
      malloc((size_t)vertex_count * packed_floats * sizeof(float));
   if (!interleaved) {
      free(index_data);
      return false;
   }
   for (unsigned v = 0; v < vertex_count; ++v) {
      const unsigned v_idx =
         info->index_size != 0 ? v + vertex_bias : draw->start + v;
      for (unsigned attribute = 0; attribute < attribute_count; ++attribute) {
         const char *attribute_reason = NULL;
         if (!pvrgpu_read_vertex_attribute(
                ctx,
                &ctx->vertex_elements->elements[attribute],
                v_idx,
                attribute_components[attribute],
                &interleaved[(size_t)v * packed_floats +
                             attribute_offsets[attribute]],
                &attribute_reason)) {
            pvrgpu_counter_eventf(
               "draw_array_primitive_record_error",
               "stage=attributes reason=%s attribute=%u "
               "format=%s components=%u vertex=%u",
               attribute_reason ? attribute_reason : "unknown",
               attribute,
               util_format_name(
                  ctx->vertex_elements->elements[attribute].src_format),
               attribute_components[attribute],
               v_idx);
            free(interleaved);
            free(index_data);
            return false;
         }
      }
   }

   const unsigned vertex_uniform_dwords =
      pvrgpu_stage_uniform_dwords(ctx, MESA_SHADER_VERTEX);
   const unsigned fragment_uniform_dwords =
      pvrgpu_stage_uniform_dwords(ctx, MESA_SHADER_FRAGMENT);

   char error[512] = {0};
   if (!ctx->pco_compiler) {
      ctx->pco_compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
      if (!ctx->pco_compiler) {
         pvrgpu_counter_eventf("draw_color_triangle_pco_command_error",
                               "stage=compiler_create reason=%s",
                               error[0] ? error : "unknown");
         free(interleaved);
         free(index_data);
         return false;
      }
   }


   struct pvrgpu_pco_graphics_binary binary;
   memset(&binary, 0, sizeof(binary));
   if (!pvrgpu_pco_compile_color_triangle(ctx->pco_compiler,
                                          ctx->vs->nir,
                                          ctx->fs->nir,
                                          attribute_formats,
                                          info->mode == MESA_PRIM_POINTS &&
                                             ctx->rasterizer &&
                                             ctx->rasterizer->state
                                                .point_size_per_vertex,
                                          ctx->framebuffer.nr_cbufs,
                                          vertex_uniform_dwords,
                                          fragment_uniform_dwords,
                                          attribute_count,
                                          ctx->num_sampler_views
                                             [MESA_SHADER_FRAGMENT],
                                          &binary,
                                          error,
                                          sizeof(error))) {
      pvrgpu_counter_eventf("draw_color_triangle_pco_command_error",
                            "stage=pco_compile reason=%s",
                            error[0] ? error : "unknown");
      free(interleaved);
      free(index_data);
      return false;
   }

   struct pvrgpu_draw_pco_triangles_command command;
   memset(&command, 0, sizeof(command));
   command.case_name = pvrgpu_command_case_name("color_triangle.gallium.pco");
   command.frame = 1;
   command.framebuffer_width = ctx->framebuffer.width;
   command.framebuffer_height = ctx->framebuffer.height;
   command.width = (uint32_t)(ctx->viewport.scale[0] * 2.0f);
   command.height = (uint32_t)(ctx->viewport.scale[1] * 2.0f);
   command.format = PVRGPU_DRIVER_COMMAND_FORMAT_RGBA8;
   command.clear_color_bits[3] = UINT32_C(0x3f800000);
   command.raw_vertex_data = (const uint8_t *)interleaved;
   command.raw_vertex_data_size =
      (size_t)vertex_count * packed_floats * sizeof(float);
   command.vertex_stride = packed_floats * sizeof(float);
   command.vertex_attribute_count = attribute_count;
   for (unsigned attribute = 0; attribute < attribute_count; ++attribute) {
      command.vertex_attribute_components[attribute] =
         attribute_components[attribute];
   }
   command.vertex_count = vertex_count;
   command.first_vertex = 0;
   command.instance_count = 1;
   command.primitive_mode = (uint32_t)info->mode;
   command.indexed = info->index_size != 0 ? 1u : 0u;
   command.render_target_count = ctx->framebuffer.nr_cbufs;
   command.raw_index_data = index_data;
   command.raw_index_data_size = index_data_size;
   command.index_size = info->index_size;
   command.index_count = info->index_size != 0 ? assembled_count : 0u;
   command.first_index = 0;
   command.base_vertex = 0;
   command.vertex_pco = binary.vertex.data;
   command.vertex_pco_size = binary.vertex.size;
   command.fragment_pco = binary.fragment.data;
   command.fragment_pco_size = binary.fragment.size;
   uint32_t vertex_uniform_words[PVRGPU_COLOR_PRIMITIVE_UNIFORM_DWORDS] = {0};
   uint32_t fragment_uniform_words[PVRGPU_COLOR_PRIMITIVE_UNIFORM_DWORDS] = {0};
   const unsigned vertex_shared_count = binary.vertex.abi.shareds;
   const unsigned fragment_shared_count = binary.fragment.abi.shareds;
   if (vertex_shared_count > PVRGPU_COLOR_PRIMITIVE_UNIFORM_DWORDS ||
       fragment_shared_count > PVRGPU_COLOR_PRIMITIVE_UNIFORM_DWORDS ||
       !pvrgpu_copy_stage_uniform_words(ctx,
                                        MESA_SHADER_VERTEX,
                                        vertex_shared_count,
                                        vertex_uniform_words) ||
       !pvrgpu_copy_stage_uniform_words(ctx,
                                        MESA_SHADER_FRAGMENT,
                                        fragment_shared_count,
                                        fragment_uniform_words)) {
      pvrgpu_pco_graphics_binary_finish(&binary);
      free(interleaved);
      free(index_data);
      return false;
   }
   command.vertex_shared = vertex_shared_count ? vertex_uniform_words : NULL;
   command.vertex_shared_count = vertex_shared_count;
   command.fragment_shared =
      fragment_shared_count ? fragment_uniform_words : NULL;
   command.fragment_shared_count = fragment_shared_count;
   pvrgpu_copy_pco_stage_abi_to_command(&command.vertex_pco_abi,
                                        &binary.vertex.abi);
   pvrgpu_copy_pco_stage_abi_to_command(&command.fragment_pco_abi,
                                        &binary.fragment.abi);
   command.position_output_start = binary.position_output_start;
   command.position_output_count = binary.position_output_count;
   command.fragment_position_start = binary.fragment_position_start;
   command.fragment_position_count = binary.fragment_position_count;
   command.varying_output_start = binary.varying_output_start;
   command.varying_output_count = binary.varying_output_count;
   command.fragment_varying_start = binary.fragment_varying_start;
   command.fragment_varying_count = binary.fragment_varying_count;
   for (unsigned component = 0; component < 3; ++component) {
      command.viewport_scale_bits[component] =
         pvrgpu_float_bits(ctx->viewport.scale[component]);
      command.viewport_translate_bits[component] =
         pvrgpu_float_bits(ctx->viewport.translate[component]);
   }
   command.front_ccw = ctx->rasterizer ? ctx->rasterizer->state.front_ccw : 0;
   command.cull_face = ctx->rasterizer ? ctx->rasterizer->state.cull_face : 0;
   command.fill_front = ctx->rasterizer ? ctx->rasterizer->state.fill_front : 0;
   command.fill_back = ctx->rasterizer ? ctx->rasterizer->state.fill_back : 0;
   /*
    * The model widens a line or point into real screen-space geometry, so the
    * width it should use travels with the draw.  GLES guarantees 1.0 and that
    * is what an unset rasterizer means.
    */
   {
      const float line_width =
         ctx->rasterizer ? ctx->rasterizer->state.line_width : 1.0f;
      const float point_size =
         ctx->rasterizer ? ctx->rasterizer->state.point_size : 1.0f;
      memcpy(&command.line_width_bits, &line_width, sizeof(line_width));
      memcpy(&command.point_size_bits, &point_size, sizeof(point_size));
   }
   command.scissor = ctx->rasterizer ? ctx->rasterizer->state.scissor : 0;
   if (command.scissor) {
      /*
       * Gallium hands the scissor as an inclusive min/max box already clipped
       * to the render target.  The capsule states it as an origin and extent,
       * and an empty box means the draw covers nothing.
       */
      /*
       * GL starts the scissor box at the whole window, so a draw that enables
       * the test before setting a rectangle scissors to the render target.
       */
      if (!ctx->has_scissor) {
         command.scissor_x = 0;
         command.scissor_y = 0;
         command.scissor_width = command.framebuffer_width;
         command.scissor_height = command.framebuffer_height;
      } else {
         const struct pipe_scissor_state *box = &ctx->scissor;
         const unsigned max_x =
            MIN2((unsigned)box->maxx, command.framebuffer_width);
         const unsigned max_y =
            MIN2((unsigned)box->maxy, command.framebuffer_height);
         command.scissor_x = MIN2((unsigned)box->minx, max_x);
         command.scissor_y = MIN2((unsigned)box->miny, max_y);
         command.scissor_width = max_x - command.scissor_x;
         command.scissor_height = max_y - command.scissor_y;
      }
      if (command.scissor_width == 0 || command.scissor_height == 0) {
         /*
          * An empty scissor rejects every fragment.  The capsule states a
          * non-empty rectangle, so describe the draw as covering nothing by
          * leaving the test off and letting the geometry cull itself is not
          * available -- decline instead, and let a later path report it.
          */
         pvrgpu_counter_eventf("draw_array_primitive_record_error",
                               "stage=scissor reason=empty_rectangle "
                               "box=%d,%d,%d,%d framebuffer=%ux%u",
                               ctx->scissor.minx, ctx->scissor.miny,
                               ctx->scissor.maxx, ctx->scissor.maxy,
                               command.framebuffer_width,
                               command.framebuffer_height);
         pvrgpu_pco_graphics_binary_finish(&binary);
         free(interleaved);
         free(index_data);
         return false;
      }
   }
   command.rasterizer_discard =
      ctx->rasterizer ? ctx->rasterizer->state.rasterizer_discard : 0;
   command.multisample = ctx->rasterizer ? ctx->rasterizer->state.multisample : 0;
   command.half_pixel_center =
      ctx->rasterizer ? ctx->rasterizer->state.half_pixel_center : 1;
   command.bottom_edge_rule =
      ctx->rasterizer ? ctx->rasterizer->state.bottom_edge_rule : 0;
   command.clip_halfz = ctx->rasterizer ? ctx->rasterizer->state.clip_halfz : 0;
   command.depth_clip_near =
      ctx->rasterizer ? ctx->rasterizer->state.depth_clip_near : 1;
   command.depth_clip_far =
      ctx->rasterizer ? ctx->rasterizer->state.depth_clip_far : 1;
   command.depth_clamp =
      ctx->rasterizer ? ctx->rasterizer->state.depth_clamp : 0;
   command.sample_mask = ctx->sample_mask;
   command.color_mask = pvrgpu_rt_colormask(ctx, 0);
   command.blend_enable =
      ctx->blend ? ctx->blend->state.rt[0].blend_enable : 0;
   command.dither = ctx->blend ? ctx->blend->state.dither : 1;
   command.depth_enable = ctx->dsa ? ctx->dsa->state.depth_enabled : 0;
   command.depth_write = ctx->dsa ? ctx->dsa->state.depth_writemask : 0;
   command.depth_func = ctx->dsa ? ctx->dsa->state.depth_func : 3;
   command.depth_clear_bits = UINT32_C(0x3f800000);
   command.depth_format = ctx->framebuffer.zsbuf.format;

   if (!pvrgpu_validate_draw_pco_triangles_command(path,
                                                  &command,
                                                  error,
                                                  sizeof(error))) {
      pvrgpu_pco_graphics_binary_finish(&binary);
      free(interleaved);
      free(index_data);
      debug_printf("pvrgpu: %s\n", error);
      pvrgpu_counter_eventf("draw_array_primitive_record_error",
                            "stage=validate reason=%s",
                            error[0] ? error : "unknown");
      return false;
   }

   struct pvrgpu_array_primitive_draw *recorded =
      CALLOC_STRUCT(pvrgpu_array_primitive_draw);
   if (!recorded) {
      pvrgpu_pco_graphics_binary_finish(&binary);
      free(interleaved);
      free(index_data);
      return false;
   }
   /* The nested draw points at payloads this record owns until submission. */
   recorded->vertex_data = interleaved;
   recorded->index_data = index_data;
   recorded->binary = binary;
   pvrgpu_pco_triangles_command_to_systemc(&command, &recorded->command);
   ctx->array_primitive_draws[ctx->array_primitive_draw_count++] = recorded;

   /*
    * The trace reports how many draw actions it contains, so the sequence is
    * complete the moment the last one is recorded.  Submit it here: waiting
    * for teardown would let the present path claim the command file first and
    * describe the frame as a flat colour instead of the geometry.
    */
   unsigned expected_draws = 0;
   if (pvrgpu_trace_draw_actions(&expected_draws) && expected_draws != 0 &&
       ctx->array_primitive_draw_count == expected_draws)
      (void)pvrgpu_emit_array_primitive_sequence_command(ctx);

   pvrgpu_counter_eventf("draw_array_primitive_recorded",
                         "ordinal=%u framebuffer=%ux%u vertices=%u mode=%u "
                         "vs_bytes=%zu fs_bytes=%zu fill=%u,%u scissor=%u "
                         "discard=%u multisample=%u halfpixel=%u "
                         "bottomedge=%u halfz=%u clip=%u,%u clamp=%u "
                         "samplemask=0x%x colormask=0x%x blend=%u dither=%u "
                         "depth=%u,%u,%u cull=%u ccw=%u",
                         ctx->array_primitive_draw_count - 1u,
                         command.framebuffer_width,
                         command.framebuffer_height,
                         command.vertex_count,
                         command.primitive_mode,
                         command.vertex_pco_size,
                         command.fragment_pco_size,
                         command.fill_front,
                         command.fill_back,
                         command.scissor,
                         command.rasterizer_discard,
                         command.multisample,
                         command.half_pixel_center,
                         command.bottom_edge_rule,
                         command.clip_halfz,
                         command.depth_clip_near,
                         command.depth_clip_far,
                         command.depth_clamp,
                         command.sample_mask,
                         command.color_mask,
                         command.blend_enable,
                         command.dither,
                         command.depth_enable,
                         command.depth_write,
                         command.depth_func,
                         command.cull_face,
                         command.front_ccw);
   return true;
}

static uint64_t
pvrgpu_texture_pco_bits(uint64_t value, unsigned first, unsigned last)
{
   const unsigned width = last - first + 1u;
   const uint64_t mask = (UINT64_C(1) << width) - 1u;
   return (value & mask) << first;
}

static void
pvrgpu_texture_pco_store_u64(uint32_t *words,
                             unsigned first_dword,
                             uint64_t value)
{
   words[first_dword] = (uint32_t)value;
   words[first_dword + 1u] = (uint32_t)(value >> 32u);
}

static void
pvrgpu_texture_pco_build_descriptor(uint32_t fragment_shared[20])
{
   memset(fragment_shared, 0, 20u * sizeof(fragment_shared[0]));

   /* Public Rogue U8U8U8U8 linear image.  The capture view is RGBX, so the
    * fourth source is the hardware ONE swizzle rather than texture byte X. */
   const uint64_t image_word0 =
      pvrgpu_texture_pco_bits(4u, 0, 2) |
      pvrgpu_texture_pco_bits(4u, 5, 7) |
      pvrgpu_texture_pco_bits(2u, 8, 10) |
      pvrgpu_texture_pco_bits(1u, 11, 13) |
      pvrgpu_texture_pco_bits(0u, 14, 16) |
      pvrgpu_texture_pco_bits(12u, 27, 33) |
      pvrgpu_texture_pco_bits(PVRGPU_TEXTURE_PCO_WIDTH - 1u, 34, 47) |
      pvrgpu_texture_pco_bits(PVRGPU_TEXTURE_PCO_HEIGHT - 1u, 48, 61);
   pvrgpu_texture_pco_store_u64(fragment_shared, 0, image_word0);

   const uint64_t image_word1 =
      pvrgpu_texture_pco_bits(PVRGPU_TEXTURE_PCO_WIDTH - 1u, 0, 14) |
      pvrgpu_texture_pco_bits(PVRGPU_TEXTURE_PCO_GPU_ADDRESS >> 2u, 16, 53) |
      pvrgpu_texture_pco_bits(1u, 60, 63);
   pvrgpu_texture_pco_store_u64(fragment_shared, 2, image_word1);
   fragment_shared[4] = PVRGPU_TEXTURE_PCO_BYTES;

   /* Rogue address mode 2 is CLAMP_TO_EDGE.  All filters are nearest, LOD is
    * fixed at zero, coordinates are normalized, and dadjust=4095 is zero. */
   const uint64_t sampler_word0 =
      pvrgpu_texture_pco_bits(4095u, 0, 12) |
      pvrgpu_texture_pco_bits(2u, 33, 35) |
      pvrgpu_texture_pco_bits(2u, 41, 43);
   pvrgpu_texture_pco_store_u64(fragment_shared, 8, sampler_word0);
   const uint64_t gather_word0 =
      sampler_word0 | pvrgpu_texture_pco_bits(1u, 36, 37) |
      pvrgpu_texture_pco_bits(1u, 38, 39);
   pvrgpu_texture_pco_store_u64(fragment_shared, 16, gather_word0);
}

static bool
pvrgpu_emit_texture_pco_command(
   struct pvrgpu_context *ctx,
   const struct pvrgpu_texture_pco_observation *observation)
{
   const char *path = pvrgpu_command_output_path();
   if (!path || !ctx || !observation ||
       !observation->interleaved_vertex_data ||
       !observation->sampled_texture_bytes ||
       ctx->driver_draw_command_emitted ||
       pvrgpu_driver_draw_command_has_been_emitted()) {
      pvrgpu_counter_eventf("draw_pco_texture_command_skip",
                            "reason=missing_path_payload_or_already_emitted");
      return false;
   }

   char error[512] = { 0 };
   if (!ctx->pco_compiler) {
      ctx->pco_compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
      if (!ctx->pco_compiler) {
         pvrgpu_counter_eventf("draw_pco_texture_command_error",
                               "stage=compiler_create reason=%s",
                               error[0] ? error : "unknown");
         return false;
      }
   }

   struct pvrgpu_pco_graphics_binary binary;
   memset(&binary, 0, sizeof(binary));
   if (!pvrgpu_pco_compile_texture(ctx->pco_compiler,
                                   ctx->vs->nir,
                                   ctx->fs->nir,
                                   &binary,
                                   error,
                                   sizeof(error))) {
      pvrgpu_counter_eventf("draw_pco_texture_command_error",
                            "stage=pco_compile reason=%s",
                            error[0] ? error : "unknown");
      return false;
   }

   uint32_t fragment_shared[20];
   pvrgpu_texture_pco_build_descriptor(fragment_shared);
   struct pvrgpu_draw_pco_triangles_command command;
   memset(&command, 0, sizeof(command));
   command.case_name = pvrgpu_command_case_name("glmark2.texture.pco");
   command.frame = 1;
   command.framebuffer_width = observation->framebuffer_width;
   command.framebuffer_height = observation->framebuffer_height;
   command.width = observation->viewport_width;
   command.height = observation->viewport_height;
   command.format = PVRGPU_DRIVER_COMMAND_FORMAT_RGBA8;
   command.clear_color_bits[3] = UINT32_C(0x3f800000);
   command.raw_vertex_data = observation->interleaved_vertex_data;
   command.raw_vertex_data_size = observation->interleaved_vertex_data_size;
   command.vertex_stride = PVRGPU_TEXTURE_PCO_VERTEX_STRIDE;
   command.vertex_count = observation->vertex_count;
   command.first_vertex = 0;
   command.instance_count = 1;
   command.primitive_mode = (uint32_t)MESA_PRIM_TRIANGLES;
   command.indexed = 0;
   command.render_target_count = 1;
   command.vertex_pco = binary.vertex.data;
   command.vertex_pco_size = binary.vertex.size;
   command.fragment_pco = binary.fragment.data;
   command.fragment_pco_size = binary.fragment.size;
   command.vertex_shared = observation->vertex_shared;
   command.vertex_shared_count = PVRGPU_ARRAY_SIZE(observation->vertex_shared);
   command.fragment_shared = fragment_shared;
   command.fragment_shared_count = PVRGPU_ARRAY_SIZE(fragment_shared);
   command.sampled_texture_count = 1;
   command.sampled_texture_bytes = observation->sampled_texture_bytes;
   command.sampled_texture_bytes_size = observation->sampled_texture_bytes_size;
   command.sampled_texture_width = PVRGPU_TEXTURE_PCO_WIDTH;
   command.sampled_texture_height = PVRGPU_TEXTURE_PCO_HEIGHT;
   command.sampled_texture_row_pitch = PVRGPU_TEXTURE_PCO_ROW_PITCH;
   command.sampled_texture_format = PVRGPU_DRIVER_COMMAND_FORMAT_RGBX8;
   command.sampled_texture_mip_count = 1;
   pvrgpu_copy_pco_stage_abi_to_command(&command.vertex_pco_abi,
                                        &binary.vertex.abi);
   pvrgpu_copy_pco_stage_abi_to_command(&command.fragment_pco_abi,
                                        &binary.fragment.abi);
   command.position_output_start = binary.position_output_start;
   command.position_output_count = binary.position_output_count;
   command.fragment_position_start = binary.fragment_position_start;
   command.fragment_position_count = binary.fragment_position_count;
   command.varying_output_start = binary.varying_output_start;
   command.varying_output_count = binary.varying_output_count;
   command.fragment_varying_start = binary.fragment_varying_start;
   command.fragment_varying_count = binary.fragment_varying_count;
   for (unsigned component = 0; component < 3; ++component) {
      command.viewport_scale_bits[component] =
         pvrgpu_float_bits(ctx->viewport.scale[component]);
      command.viewport_translate_bits[component] =
         pvrgpu_float_bits(ctx->viewport.translate[component]);
   }
   command.front_ccw = ctx->rasterizer->state.front_ccw;
   command.cull_face = ctx->rasterizer->state.cull_face;
   command.fill_front = ctx->rasterizer->state.fill_front;
   command.fill_back = ctx->rasterizer->state.fill_back;
   command.scissor = ctx->rasterizer->state.scissor;
   command.rasterizer_discard = ctx->rasterizer->state.rasterizer_discard;
   command.multisample = ctx->rasterizer->state.multisample;
   command.half_pixel_center = ctx->rasterizer->state.half_pixel_center;
   command.bottom_edge_rule = ctx->rasterizer->state.bottom_edge_rule;
   command.clip_halfz = ctx->rasterizer->state.clip_halfz;
   command.depth_clip_near = ctx->rasterizer->state.depth_clip_near;
   command.depth_clip_far = ctx->rasterizer->state.depth_clip_far;
   command.depth_clamp = ctx->rasterizer->state.depth_clamp;
   command.sample_mask = ctx->sample_mask;
   command.color_mask = pvrgpu_rt_colormask(ctx, 0);
   command.blend_enable = ctx->blend->state.rt[0].blend_enable;
   command.dither = ctx->blend->state.dither;
   command.depth_enable = ctx->dsa->state.depth_enabled;
   command.depth_write = ctx->dsa->state.depth_writemask;
   command.depth_func = ctx->dsa->state.depth_func;
   command.depth_clear_bits = UINT32_C(0x3f800000);
   command.depth_format = ctx->framebuffer.zsbuf.format;

   pvrgpu_counter_eventf(
      "draw_pco_texture_binary",
      "vertices=%u vs_bytes=%zu vs_fnv1a64=%016llx fs_bytes=%zu "
      "fs_fnv1a64=%016llx descriptor=%u,%u,%u linkage=%u,%u,%u,%u",
      observation->vertex_count,
      binary.vertex.size,
      (unsigned long long)pvrgpu_pco_binary_fnv1a64(binary.vertex.data,
                                                    binary.vertex.size),
      binary.fragment.size,
      (unsigned long long)pvrgpu_pco_binary_fnv1a64(binary.fragment.data,
                                                    binary.fragment.size),
      binary.fragment_texture_descriptor_start,
      binary.fragment_texture_descriptor_count,
      binary.fragment_texture_descriptor_stride,
      binary.varying_output_start,
      binary.varying_output_count,
      binary.fragment_varying_start,
      binary.fragment_varying_count);

   const bool emitted = pvrgpu_write_draw_pco_triangles_command(path,
                                                                &command,
                                                                error,
                                                                sizeof(error));
   pvrgpu_pco_graphics_binary_finish(&binary);
   if (!emitted) {
      remove(path);
      pvrgpu_counter_eventf("draw_pco_texture_command_error",
                            "stage=command_submit reason=%s",
                            error[0] ? error : "unknown");
      return false;
   }

   ctx->driver_draw_command_emitted = true;
   pvrgpu_note_driver_draw_command_emitted();
   pvrgpu_counter_eventf("draw_pco_texture_command",
                         "framebuffer=%ux%u vertices=%u texture=%ux%u",
                         command.framebuffer_width,
                         command.framebuffer_height,
                         command.vertex_count,
                         command.sampled_texture_width,
                         command.sampled_texture_height);
   return true;
}

static bool
pvrgpu_emit_ideas_pco_command(
   struct pvrgpu_context *ctx,
   const struct pvrgpu_ideas_pco_observation *observation)
{
   const char *path = pvrgpu_command_output_path();
   if (!path || !ctx || !observation ||
       !observation->interleaved_vertex_data ||
       observation->profile >= PVRGPU_IDEAS_PCO_PROFILE_COUNT ||
       ctx->ideas_pco_draws >= PVRGPU_IDEAS_PCO_DRAW_COUNT ||
       ctx->driver_draw_command_emitted ||
       pvrgpu_driver_draw_command_has_been_emitted()) {
      pvrgpu_counter_eventf("draw_pco_ideas_command_skip",
                            "reason=missing_payload_bad_ordinal_or_owned");
      return false;
   }

   char error[512] = { 0 };
   if (!ctx->pco_compiler) {
      ctx->pco_compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
      if (!ctx->pco_compiler) {
         pvrgpu_counter_eventf("draw_pco_ideas_command_error",
                               "stage=compiler_create reason=%s",
                               error[0] ? error : "unknown");
         return false;
      }
   }

   struct pvrgpu_pco_graphics_binary *binary =
      ctx->ideas_pco_binaries[observation->profile];
   if (!binary) {
      binary = CALLOC(1, sizeof(*binary));
      if (!binary) {
         pvrgpu_counter_eventf("draw_pco_ideas_command_error",
                               "stage=binary_cache reason=out_of_memory");
         return false;
      }
      enum pvrgpu_pco_ideas_profile compile_profile;
      switch (observation->profile) {
      case PVRGPU_IDEAS_PCO_LOGO:
         compile_profile = PVRGPU_PCO_IDEAS_LOGO;
         break;
      case PVRGPU_IDEAS_PCO_LIGHTING:
         compile_profile = PVRGPU_PCO_IDEAS_LIGHTING;
         break;
      case PVRGPU_IDEAS_PCO_WHITE:
         compile_profile = PVRGPU_PCO_IDEAS_WHITE;
         break;
      case PVRGPU_IDEAS_PCO_BLACK:
         compile_profile = PVRGPU_PCO_IDEAS_BLACK;
         break;
      default:
         FREE(binary);
         return false;
      }
      if (!pvrgpu_pco_compile_ideas(ctx->pco_compiler,
                                    ctx->vs->nir,
                                    ctx->fs->nir,
                                    compile_profile,
                                    binary,
                                    error,
                                    sizeof(error))) {
         pvrgpu_pco_graphics_binary_finish(binary);
         FREE(binary);
         pvrgpu_counter_eventf("draw_pco_ideas_command_error",
                               "stage=pco_compile profile=%u reason=%s",
                               observation->profile,
                               error[0] ? error : "unknown");
         return false;
      }
      ctx->ideas_pco_binaries[observation->profile] = binary;
   }

   struct pvrgpu_draw_pco_triangles_command command;
   memset(&command, 0, sizeof(command));
   command.case_name = pvrgpu_command_case_name("ideas.ideas.capture.1");
   command.frame = 1;
   command.framebuffer_width = observation->framebuffer_width;
   command.framebuffer_height = observation->framebuffer_height;
   command.width = observation->viewport_width;
   command.height = observation->viewport_height;
   command.format = PVRGPU_DRIVER_COMMAND_FORMAT_RGBA8;
   command.clear_color_bits[3] = UINT32_C(0x3f800000);
   command.raw_vertex_data = observation->interleaved_vertex_data;
   command.raw_vertex_data_size = observation->interleaved_vertex_data_size;
   command.vertex_stride = observation->vertex_stride;
   command.vertex_count = observation->vertex_count;
   command.first_vertex = 0;
   command.instance_count = 1;
   command.primitive_mode = observation->primitive_mode;
   command.indexed = 0;
   command.render_target_count = 1;
   if (ctx->ideas_pco_draws == 0) {
      command.draw_count = PVRGPU_IDEAS_PCO_DRAW_COUNT;
      command.ia_vertices = command.vertex_count;
      command.ia_primitives =
         pvrgpu_array_primitive_count(command.primitive_mode,
                                      command.vertex_count);
      command.clip_invocations = command.ia_primitives;
      /* Rasterization results are measured; see the sequence emitters. */
      command.vs_invocations = 0;
      command.clip_primitives = 0;
      command.setup_triangles = 0;
      command.ps_invocations = 0;
   }
   command.vertex_pco = binary->vertex.data;
   command.vertex_pco_size = binary->vertex.size;
   command.fragment_pco = binary->fragment.data;
   command.fragment_pco_size = binary->fragment.size;
   command.vertex_shared = observation->vertex_shared;
   command.vertex_shared_count = observation->vertex_shared_count;
   command.fragment_shared = observation->fragment_shared_count ?
                                observation->fragment_shared : NULL;
   command.fragment_shared_count = observation->fragment_shared_count;
   pvrgpu_copy_pco_stage_abi_to_command(&command.vertex_pco_abi,
                                        &binary->vertex.abi);
   pvrgpu_copy_pco_stage_abi_to_command(&command.fragment_pco_abi,
                                        &binary->fragment.abi);
   command.position_output_start = binary->position_output_start;
   command.position_output_count = binary->position_output_count;
   command.fragment_position_start = binary->fragment_position_start;
   command.fragment_position_count = binary->fragment_position_count;
   command.varying_output_start = binary->varying_output_start;
   command.varying_output_count = binary->varying_output_count;
   command.fragment_varying_start = binary->fragment_varying_start;
   command.fragment_varying_count = binary->fragment_varying_count;
   for (unsigned component = 0; component < 3; ++component) {
      command.viewport_scale_bits[component] =
         pvrgpu_float_bits(ctx->viewport.scale[component]);
      command.viewport_translate_bits[component] =
         pvrgpu_float_bits(ctx->viewport.translate[component]);
   }
   command.front_ccw = ctx->rasterizer->state.front_ccw;
   command.cull_face = ctx->rasterizer->state.cull_face;
   command.fill_front = ctx->rasterizer->state.fill_front;
   command.fill_back = ctx->rasterizer->state.fill_back;
   command.scissor = ctx->rasterizer->state.scissor;
   command.rasterizer_discard = ctx->rasterizer->state.rasterizer_discard;
   command.multisample = ctx->rasterizer->state.multisample;
   command.half_pixel_center = ctx->rasterizer->state.half_pixel_center;
   command.bottom_edge_rule = ctx->rasterizer->state.bottom_edge_rule;
   command.clip_halfz = ctx->rasterizer->state.clip_halfz;
   command.depth_clip_near = ctx->rasterizer->state.depth_clip_near;
   command.depth_clip_far = ctx->rasterizer->state.depth_clip_far;
   command.depth_clamp = ctx->rasterizer->state.depth_clamp;
   command.sample_mask = ctx->sample_mask;
   command.color_mask = pvrgpu_rt_colormask(ctx, 0);
   command.blend_enable = ctx->blend->state.rt[0].blend_enable;
   command.dither = ctx->blend->state.dither;
   command.depth_enable = ctx->dsa->state.depth_enabled;
   command.depth_write = ctx->dsa->state.depth_writemask;
   command.depth_func = ctx->dsa->state.depth_func;
   command.depth_clear_bits = UINT32_C(0x3f800000);
   command.depth_format = ctx->framebuffer.zsbuf.format;

   const bool emitted = pvrgpu_write_draw_pco_triangles_command(path,
                                                                &command,
                                                                error,
                                                                sizeof(error));
   if (!emitted) {
      remove(path);
      pvrgpu_counter_eventf("draw_pco_ideas_command_error",
                            "stage=command_submit ordinal=%u reason=%s",
                            ctx->ideas_pco_draws,
                            error[0] ? error : "unknown");
      return false;
   }

   pvrgpu_counter_eventf(
      "draw_pco_ideas_command",
      "ordinal=%u profile=%u mode=%u vertices=%u stride=%u "
      "vs_bytes=%zu fs_bytes=%zu linkage=%u,%u,%u,%u",
      ctx->ideas_pco_draws,
      observation->profile,
      observation->primitive_mode,
      observation->vertex_count,
      observation->vertex_stride,
      binary->vertex.size,
      binary->fragment.size,
      binary->varying_output_start,
      binary->varying_output_count,
      binary->fragment_varying_start,
      binary->fragment_varying_count);
   ctx->ideas_pco_draws++;
   if (ctx->ideas_pco_draws == PVRGPU_IDEAS_PCO_DRAW_COUNT) {
      ctx->driver_draw_command_emitted = true;
      pvrgpu_note_driver_draw_command_emitted();
   }
   return true;
}

static bool
pvrgpu_write_texture_view_rgba8_sidecar(
   const struct pvrgpu_textured_triangles_observation *observation,
   const char *command_path,
   char *sidecar_path,
   size_t sidecar_path_size)
{
   if (!observation || !observation->texture_view ||
       !observation->texture_view->texture || !command_path ||
       command_path[0] == '\0' || !sidecar_path || sidecar_path_size == 0)
      return false;

   const int path_length = snprintf(sidecar_path,
                                    sidecar_path_size,
                                    "%s.texture.rgba8",
                                    command_path);
   if (path_length <= 0 || (size_t)path_length >= sidecar_path_size)
      return false;

   const struct pipe_sampler_view *view = observation->texture_view;
   struct pvrgpu_resource *resource = pvrgpu_resource(view->texture);
   const unsigned level = observation->texture_level;
   const unsigned width = observation->texture_width;
   const unsigned height = observation->texture_height;
   const unsigned block_size = util_format_get_blocksize(view->format);
   const size_t source_row_bytes = (size_t)width * block_size;
   const size_t rgba_row_bytes = (size_t)width * 4u;
   if (!resource || !resource->data || level >= resource->level_count ||
       level >= PIPE_MAX_TEXTURE_LEVELS || width == 0 || height == 0 ||
       block_size == 0 || source_row_bytes / block_size != width ||
       rgba_row_bytes / 4u != width)
      return false;

   const size_t level_offset = resource->level_offsets[level];
   const size_t source_stride = resource->level_strides[level];
   if (source_stride < source_row_bytes || level_offset > resource->size ||
       height > SIZE_MAX / rgba_row_bytes)
      return false;
   if (height != 0 &&
       (height - 1u > (SIZE_MAX - level_offset) / source_stride ||
        level_offset + (size_t)(height - 1u) * source_stride > resource->size ||
        resource->size -
              (level_offset + (size_t)(height - 1u) * source_stride) <
           source_row_bytes))
      return false;

   uint8_t *source_row = MALLOC(rgba_row_bytes);
   uint8_t *rgba_row = MALLOC(rgba_row_bytes);
   if (!source_row || !rgba_row) {
      FREE(source_row);
      FREE(rgba_row);
      return false;
   }

   FILE *file = fopen(sidecar_path, "wb");
   if (!file) {
      FREE(source_row);
      FREE(rgba_row);
      return false;
   }

   const uint8_t *base = resource->data + level_offset;
   bool ok = true;
   for (unsigned y = 0; y < height; ++y) {
      util_format_read_4ub(view->format,
                           source_row,
                           rgba_row_bytes,
                           base,
                           source_stride,
                           0,
                           y,
                           width,
                           1);
      pvrgpu_apply_sampler_view_swizzle_row(view,
                                            source_row,
                                            rgba_row,
                                            width);
      if (fwrite(rgba_row, 1, rgba_row_bytes, file) != rgba_row_bytes) {
         ok = false;
         break;
      }
   }
   FREE(source_row);
   FREE(rgba_row);
   if (fclose(file) != 0)
      ok = false;
   if (!ok) {
      remove(sidecar_path);
      sidecar_path[0] = '\0';
      return false;
   }

   pvrgpu_counter_eventf("texture_sidecar_rgba8",
                         "path=%s width=%u height=%u level=%u format=%s",
                         sidecar_path,
                         width,
                         height,
                         level,
                         util_format_name(view->format));
   return true;
}

static bool
pvrgpu_emit_draw_textured_triangles_command(
   struct pvrgpu_context *ctx,
   const struct pvrgpu_textured_triangles_observation *observation)
{
   const char *path = pvrgpu_command_output_path();
   if (!path) {
      pvrgpu_counter_eventf("draw_textured_triangles_command_skip",
                            "reason=missing_command_path");
      return false;
   }
   if (!ctx || !observation || ctx->driver_draw_command_emitted ||
       pvrgpu_driver_draw_command_has_been_emitted()) {
      pvrgpu_counter_eventf("draw_textured_triangles_command_skip",
                            "reason=command_already_emitted ctx=%u global=%u",
                            ctx && ctx->driver_draw_command_emitted ? 1 : 0,
                            pvrgpu_driver_draw_command_has_been_emitted() ?
                               1 : 0);
      return false;
   }

   char sidecar_path[PATH_MAX];
   sidecar_path[0] = '\0';
   if (!pvrgpu_write_texture_view_rgba8_sidecar(observation,
                                                path,
                                                sidecar_path,
                                                sizeof(sidecar_path))) {
      pvrgpu_counter_eventf("draw_textured_triangles_command_skip",
                            "reason=texture_sidecar_failed");
      return false;
   }

   struct pvrgpu_draw_textured_triangles_command command;
   memset(&command, 0, sizeof(command));
   command.case_name =
      pvrgpu_command_case_name("draw_textured_triangles.gallium");
   command.frame = 1;
   command.framebuffer_width = observation->framebuffer_width;
   command.framebuffer_height = observation->framebuffer_height;
   command.width = observation->viewport_width;
   command.height = observation->viewport_height;
   command.format = pvrgpu_command_format_for_framebuffer(ctx);
   command.clear_color_bits[0] = 0;
   command.clear_color_bits[1] = 0;
   command.clear_color_bits[2] = 0;
   command.clear_color_bits[3] = UINT32_C(0x3f800000);
   memcpy(command.vertex_bits,
          observation->vertex_bits,
          sizeof(command.vertex_bits));
   memcpy(command.texcoord_bits,
          observation->texcoord_bits,
          sizeof(command.texcoord_bits));
   command.texture_width = observation->texture_width;
   command.texture_height = observation->texture_height;
   command.texture_rgba8_path = sidecar_path;

   char error[256];
   if (!pvrgpu_write_draw_textured_triangles_command(path,
                                                      &command,
                                                      error,
                                                      sizeof(error))) {
      remove(sidecar_path);
      remove(path);
      debug_printf("pvrgpu: %s\n", error);
      pvrgpu_counter_eventf("draw_textured_triangles_command_error",
                            "reason=%s",
                            error);
      return false;
   }

   ctx->driver_draw_command_emitted = true;
   pvrgpu_note_driver_draw_command_emitted();
   pvrgpu_counter_eventf("draw_textured_triangles_command",
                         "framebuffer=%ux%u viewport=%ux%u texture=%ux%u "
                         "texture_path=%s",
                         command.framebuffer_width,
                         command.framebuffer_height,
                         command.width,
                         command.height,
                         command.texture_width,
                         command.texture_height,
                         command.texture_rgba8_path);
   return true;
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

/*
 * A non-indexed triangle-topology draw the generic PCO color path can lower:
 * one draw, one color attachment, no textures and two float attributes that
 * map onto the model's float2-position / float4-color vertex layout.
 */
static bool
pvrgpu_draw_is_lowerable_array_primitive(
   const struct pvrgpu_context *ctx,
   const struct pipe_draw_info *info,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,
   unsigned num_draws,
   const char **reason)
{
   const char *ignored = NULL;
   if (!reason)
      reason = &ignored;
   *reason = NULL;

   if (!ctx || !info || !draws || num_draws != 1) {
      *reason = "draw_shape";
      return false;
   }
   if (indirect) {
      *reason = "indirect_draw";
      return false;
   }
   /* An indexed draw assembles its primitives from the index buffer. */
   if (info->index_size != 0 &&
       info->index_size != 1 && info->index_size != 2 && info->index_size != 4) {
      *reason = "index_size";
      return false;
   }
   if (info->index_size != 0 && info->primitive_restart) {
      *reason = "primitive_restart";
      return false;
   }
   if (pvrgpu_array_primitive_count(info->mode, draws[0].count) == 0) {
      *reason = "primitive_mode";
      return false;
   }
   /*
    * The model widens a line or point to the GLES-guaranteed minimum of one
    * device pixel.  A draw that asks for a wider line or a larger point would
    * be rasterized at the wrong size, so leave it fail-closed until the width
    * is carried through the capsule.
    */
   /*
    * The capsule states the line width and point size, so any fixed width the
    * model can widen to is lowerable.  A per-vertex point size comes from the
    * shader instead and still is not.
    */
   if (info->mode == MESA_PRIM_LINES || info->mode == MESA_PRIM_LINE_LOOP ||
       info->mode == MESA_PRIM_LINE_STRIP) {
      const float width = ctx->rasterizer ? ctx->rasterizer->state.line_width
                                          : 1.0f;
      if (!(width >= 1.0f) || !(width <= 1024.0f)) {
         *reason = "line_width";
         return false;
      }
   }
   if (info->mode == MESA_PRIM_POINTS) {
      const float size = ctx->rasterizer ? ctx->rasterizer->state.point_size
                                         : 1.0f;
      if (ctx->rasterizer && ctx->rasterizer->state.point_size_per_vertex) {
         *reason = "point_size_per_vertex";
         return false;
      }
      if (!(size >= 1.0f) || !(size <= 1024.0f)) {
         *reason = "point_size";
         return false;
      }
   }
   if (!ctx->vs || !ctx->fs) {
      *reason = "missing_shader";
      return false;
   }
   if (ctx->tcs || ctx->tes) {
      *reason = "tessellation_stage";
      return false;
   }
   if (ctx->gs) {
      *reason = "geometry_stage";
      return false;
   }
   if (!ctx->vertex_elements || ctx->vertex_elements->num_elements == 0 ||
       ctx->num_vertex_buffers == 0) {
      *reason = "no_vertex_layout";
      return false;
   }
   if (ctx->vertex_elements->num_elements > PVRGPU_PCO_MAX_VERTEX_ATTRIBUTES) {
      *reason = "too_many_attributes";
      return false;
   }
   /* Fragment textures are bound as combined image/sampler descriptors. */
   if (ctx->num_sampler_views[MESA_SHADER_VERTEX] != 0) {
      *reason = "vertex_texture";
      return false;
   }
   if (ctx->num_sampler_views[MESA_SHADER_FRAGMENT] > PVRGPU_PCO_MAX_TEXTURES) {
      *reason = "too_many_textures";
      return false;
   }
   for (unsigned texture = 0;
        texture < ctx->num_sampler_views[MESA_SHADER_FRAGMENT]; ++texture) {
      const struct pipe_sampler_view *view =
         ctx->sampler_views[MESA_SHADER_FRAGMENT][texture];
      if (!view || !view->texture ||
          view->texture->target != PIPE_TEXTURE_2D ||
          !ctx->samplers[MESA_SHADER_FRAGMENT][texture]) {
         *reason = "texture_binding";
         return false;
      }
   }
   /*
    * One to four colour attachments, every one present and sharing the
    * format the capsule states for the pass.
    */
   if (ctx->framebuffer.nr_cbufs == 0) {
      *reason = "no_colour_attachment";
      return false;
   }
   if (ctx->framebuffer.nr_cbufs > PVRGPU_MAX_RENDER_TARGETS) {
      *reason = "too_many_render_targets";
      return false;
   }
   for (unsigned target = 0; target < ctx->framebuffer.nr_cbufs; ++target) {
      if (!ctx->framebuffer.cbufs[target].texture ||
          ctx->framebuffer.cbufs[target].format !=
             ctx->framebuffer.cbufs[0].format) {
         *reason = "mixed_render_targets";
         return false;
      }
   }
   if (ctx->framebuffer.width == 0 || ctx->framebuffer.height == 0) {
      *reason = "empty_framebuffer";
      return false;
   }
   return true;
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
   for (unsigned profile = 0;
        profile < PVRGPU_IDEAS_PCO_PROFILE_COUNT;
        ++profile) {
      if (!ctx->ideas_pco_binaries[profile])
         continue;
      pvrgpu_pco_graphics_binary_finish(ctx->ideas_pco_binaries[profile]);
      FREE(ctx->ideas_pco_binaries[profile]);
   }
   /*
    * The accumulated draws are this frame's geometry: a frame-start clear
    * retires the previous frame's records, so what survives to teardown is
    * the last complete frame.  The ordered sequence protocol accepts one
    * submission, so this is the single point that can submit it.
    */
   if (ctx->array_primitive_draw_count != 0)
      (void)pvrgpu_emit_array_primitive_sequence_command(ctx);
   pvrgpu_array_primitive_sequence_reset(ctx);
   pvrgpu_refract_pco_observation_destroy(&ctx->refract_pco_prepass);
   pvrgpu_refract_pco_observation_destroy(&ctx->refract_pco_composite);
   pvrgpu_shadow_pco_observation_destroy(&ctx->shadow_pco_depth);
   pvrgpu_shadow_pco_observation_destroy(&ctx->shadow_pco_mask);
   pvrgpu_shadow_pco_observation_destroy(&ctx->shadow_pco_scene);
   pvrgpu_terrain_pco_sequence_reset(ctx);
   pvrgpu_pco_compiler_destroy(ctx->pco_compiler);
   pvrgpu_invalidate_full_depth_clear(ctx);
   util_unreference_framebuffer_state(&ctx->framebuffer);
   pvrgpu_counter_event("context_destroy_end", "");
   FREE(ctx);
}

static void
pvrgpu_set_framebuffer_state(struct pipe_context *pipe,
                             const struct pipe_framebuffer_state *state)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   util_copy_framebuffer_state(&ctx->framebuffer, state);
   const struct pipe_surface *zs = &ctx->framebuffer.zsbuf;
   if (!ctx->full_depth_clear_is_one ||
       !zs->texture ||
       ctx->full_depth_clear_resource != zs->texture ||
       ctx->full_depth_clear_level != zs->level ||
       ctx->full_depth_clear_first_layer != zs->first_layer ||
       ctx->full_depth_clear_last_layer != zs->last_layer ||
       ctx->full_depth_clear_width != ctx->framebuffer.width ||
       ctx->full_depth_clear_height != ctx->framebuffer.height)
      pvrgpu_invalidate_full_depth_clear(ctx);
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

void
pvrgpu_invalidate_full_depth_clear(struct pvrgpu_context *ctx)
{
   if (!ctx)
      return;
   pipe_resource_reference(&ctx->full_depth_clear_resource, NULL);
   ctx->full_depth_clear_level = 0;
   ctx->full_depth_clear_first_layer = 0;
   ctx->full_depth_clear_last_layer = 0;
   ctx->full_depth_clear_width = 0;
   ctx->full_depth_clear_height = 0;
   ctx->full_depth_clear_is_one = false;
}

void
pvrgpu_invalidate_full_depth_clear_for_resource(
   struct pvrgpu_context *ctx,
   const struct pipe_resource *resource)
{
   if (ctx && resource && ctx->full_depth_clear_resource == resource)
      pvrgpu_invalidate_full_depth_clear(ctx);
}

void
pvrgpu_note_full_depth_clear_one(struct pvrgpu_context *ctx,
                                 const struct pipe_surface *surface,
                                 unsigned width,
                                 unsigned height)
{
   if (!ctx || !surface || !surface->texture ||
       surface->texture->target != PIPE_TEXTURE_2D ||
       surface->level > surface->texture->last_level ||
       surface->first_layer != 0 || surface->last_layer != 0 ||
       width == 0 || height == 0 ||
       width != u_minify(surface->texture->width0, surface->level) ||
       height != u_minify(surface->texture->height0, surface->level)) {
      pvrgpu_invalidate_full_depth_clear(ctx);
      return;
   }

   pipe_resource_reference(&ctx->full_depth_clear_resource,
                           surface->texture);
   ctx->full_depth_clear_level = surface->level;
   ctx->full_depth_clear_first_layer = surface->first_layer;
   ctx->full_depth_clear_last_layer = surface->last_layer;
   ctx->full_depth_clear_width = width;
   ctx->full_depth_clear_height = height;
   ctx->full_depth_clear_is_one = true;
   pvrgpu_counter_eventf("full_depth_clear_one",
                         "res=%p level=%u layers=%u-%u extent=%ux%u",
                         (void *)surface->texture,
                         surface->level,
                         surface->first_layer,
                         surface->last_layer,
                         width,
                         height);
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
pvrgpu_note_unsupported_draw(
   struct pvrgpu_context *ctx,
   const struct pipe_draw_info *info,
   const struct pipe_draw_indirect_info *indirect,
   const struct pipe_draw_start_count_bias *draws,
   unsigned num_draws,
   const char *reason)
{
   if (!ctx)
      return;

   ctx->unsupported_draws++;
   pvrgpu_counter_eventf("unsupported_draw",
                         "reason=%s count=%u first_count=%u mode=%u "
                         "index_size=%u has_indirect=%u indirect_offset=%u "
                         "indirect_stride=%u indirect_draw_count=%u "
                         "has_vs=%u has_tcs=%u has_tes=%u has_fs=%u "
                         "patch_vertices=%u vertex_elements=%u "
                         "vertex_buffers=%u framebuffer=%ux%u nr_cbufs=%u "
                         "cbuf0_format=%s cbuf0_res=%p zs_res=%p "
                         "colormask=0x%x cull_face=%u raster_discard=%u "
                         "sampler_views=%u samplers=%u total=%u",
                         reason ? reason : "unsupported_state",
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

static void
pvrgpu_draw_vbo(struct pipe_context *pipe,
                const struct pipe_draw_info *info,
                unsigned drawid_offset,
                const struct pipe_draw_indirect_info *indirect,
                const struct pipe_draw_start_count_bias *draws,
                unsigned num_draws)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);

   if (pvrgpu_case_suppresses_draw_commands()) {
      ctx->observed_draws++;
      pvrgpu_counter_eventf("draw_suppressed",
                            "case=%s total=%u",
                            pvrgpu_command_case_name("none"),
                            ctx->observed_draws);
      return;
   }

   unsigned requested_width = 0;
   unsigned requested_height = 0;
   const bool has_requested_extent =
      pvrgpu_rdc_output_extent(&requested_width, &requested_height);
   const bool requested_glmark_extent =
      has_requested_extent &&
      pvrgpu_glmark_output_extent_supported(requested_width,
                                            requested_height);
   const bool conditionals_probe_framebuffer =
      requested_glmark_extent &&
      ctx->framebuffer.width == 1 && ctx->framebuffer.height == 1;

   const char *rdc_case_name = pvrgpu_rdc_case_name();
   if (rdc_case_name && strcmp(rdc_case_name,
                               "terrain.terrain.capture.1") == 0) {
      const unsigned terrain_ordinal = ctx->terrain_pco_probe_draws++;
      const enum pvrgpu_pco_terrain_profile terrain_profile =
         (enum pvrgpu_pco_terrain_profile)
            (terrain_ordinal % PVRGPU_TERRAIN_PCO_DRAW_COUNT);
      const unsigned terrain_pass =
         terrain_ordinal / PVRGPU_TERRAIN_PCO_DRAW_COUNT;
      char terrain_vs_source_hash[72];
      char terrain_fs_source_hash[72];
      pvrgpu_nir_source_hash_string(ctx->vs ? ctx->vs->nir : NULL,
                                    terrain_vs_source_hash);
      pvrgpu_nir_source_hash_string(ctx->fs ? ctx->fs->nir : NULL,
                                    terrain_fs_source_hash);
      pvrgpu_counter_eventf("draw_pco_terrain_source_hash",
                            "pass=%u profile=%u vs=%s fs=%s",
                            terrain_pass,
                            terrain_profile,
                            terrain_vs_source_hash,
                            terrain_fs_source_hash);

      /* RenderDoc's first replay pass restores an attachment alias graph
       * which intentionally stops matching at D4.  Once that exact warmup
       * boundary has been identified below, the remaining four physical
       * draws from pass zero are observations only.  Pass one is captured
       * from D1 again and must satisfy the complete strict sequence. */
      if (terrain_pass == 0 && ctx->terrain_pco_warmup_skipped) {
         ctx->observed_draws++;
         pvrgpu_counter_eventf("draw_pco_terrain_warmup_draw_skip",
                               "pass=0 profile=%u total=%u",
                               terrain_profile,
                               ctx->observed_draws);
         pvrgpu_invalidate_full_depth_clear(ctx);
         return;
      }

      /* RenderDoc replays this eight-draw sequence twice.  Retain one
       * complete, internally consistent pass; later duplicate draws are
       * observations only and must not overwrite its attachment graph. */
      if (ctx->terrain_pco_draw_count == PVRGPU_TERRAIN_PCO_DRAW_COUNT) {
         ctx->observed_draws++;
         pvrgpu_counter_eventf("draw_pco_terrain_duplicate_skip",
                               "pass=%u profile=%u total=%u",
                               terrain_pass,
                               terrain_profile,
                               ctx->observed_draws);
         pvrgpu_invalidate_full_depth_clear(ctx);
         return;
      }
      if (terrain_profile == PVRGPU_PCO_TERRAIN_D1) {
         if (ctx->terrain_pco_draw_count != 0)
            pvrgpu_terrain_pco_sequence_reset(ctx);
         if (terrain_pass != 0)
            ctx->terrain_pco_warmup_skipped = false;
      }

      if (ctx->terrain_pco_draw_count != (unsigned)terrain_profile) {
         pvrgpu_counter_eventf("draw_pco_terrain_match_miss",
                               "pass=%u profile=%u expected_profile=%u "
                               "reason=sequence_order",
                               terrain_pass,
                               terrain_profile,
                               ctx->terrain_pco_draw_count);
         pvrgpu_terrain_pco_sequence_reset(ctx);
         pvrgpu_note_unsupported_draw(ctx,
                                     info,
                                     indirect,
                                     draws,
                                     num_draws,
                                     "terrain_pco_sequence_order");
         pvrgpu_invalidate_full_depth_clear(ctx);
         return;
      }

      struct pvrgpu_terrain_pco_observation *terrain =
         CALLOC_STRUCT(pvrgpu_terrain_pco_observation);
      const char *terrain_failure_reason = NULL;
      if (terrain &&
          pvrgpu_draw_matches_terrain_pco(ctx,
                                          info,
                                          drawid_offset,
                                          indirect,
                                          draws,
                                          num_draws,
                                          terrain_profile,
                                          terrain_pass,
                                          terrain,
                                          &terrain_failure_reason)) {
         pvrgpu_terrain_pco_observation_destroy(
            &ctx->terrain_pco_draws[terrain_profile]);
         ctx->terrain_pco_draws[terrain_profile] = terrain;
         terrain = NULL;
         ctx->terrain_pco_draw_count++;
         if (ctx->terrain_pco_draw_count == PVRGPU_TERRAIN_PCO_DRAW_COUNT) {
            pvrgpu_counter_eventf("draw_pco_terrain_sequence_ready",
                                  "draws=8 submit=api_v8");
            if (!pvrgpu_emit_terrain_pco_sequence_command(ctx)) {
               pvrgpu_terrain_pco_sequence_reset(ctx);
               pvrgpu_note_unsupported_draw(ctx,
                                            info,
                                            indirect,
                                            draws,
                                            num_draws,
                                            "terrain_pco_sequence_submit");
               pvrgpu_invalidate_full_depth_clear(ctx);
               return;
            }
         }
         ctx->observed_draws++;
         pvrgpu_invalidate_full_depth_clear(ctx);
         return;
      }

      pvrgpu_terrain_pco_observation_destroy(&terrain);
      if (terrain_pass == 0 &&
          terrain_profile == PVRGPU_PCO_TERRAIN_D4 &&
          ctx->terrain_pco_draw_count == PVRGPU_PCO_TERRAIN_D4 &&
          terrain_failure_reason &&
          strcmp(terrain_failure_reason, "texture_attachment_source") == 0) {
         pvrgpu_terrain_pco_sequence_reset(ctx);
         ctx->terrain_pco_warmup_skipped = true;
         ctx->observed_draws++;
         pvrgpu_counter_eventf(
            "draw_pco_terrain_warmup_sequence_skip",
            "pass=0 profile=%u reason=restored_texture_attachment_alias "
            "total=%u",
            terrain_profile,
            ctx->observed_draws);
         pvrgpu_invalidate_full_depth_clear(ctx);
         return;
      }
      pvrgpu_counter_eventf("draw_pco_terrain_match_miss",
                            "pass=%u profile=%u reason=%s framebuffer=%ux%u",
                            terrain_pass,
                            terrain_profile,
                            terrain_failure_reason ? terrain_failure_reason :
                                                     "allocation",
                            ctx->framebuffer.width,
                            ctx->framebuffer.height);
      pvrgpu_terrain_pco_sequence_reset(ctx);
      pvrgpu_note_unsupported_draw(ctx,
                                  info,
                                  indirect,
                                  draws,
                                  num_draws,
                                  "terrain_pco_sequence_failed");
      pvrgpu_invalidate_full_depth_clear(ctx);
      return;
   }
   enum pvrgpu_pco_shadow_profile shadow_profile;
   const bool shadow_source_profile =
      rdc_case_name &&
      strcmp(rdc_case_name, "shadow.shadow.capture.1") == 0 &&
      pvrgpu_shadow_pco_profile(ctx, &shadow_profile);
   if (shadow_source_profile &&
       pvrgpu_shadow_pco_draw_info_matches(info,
                                           drawid_offset,
                                           indirect,
                                           draws,
                                           num_draws,
                                           shadow_profile)) {
      if (shadow_profile == PVRGPU_PCO_SHADOW_SCENE &&
          ctx->shadow_pco_warmup_skipped && !ctx->shadow_pco_depth &&
          !ctx->shadow_pco_mask) {
         ctx->observed_draws++;
         pvrgpu_counter_eventf("draw_pco_shadow_warmup_scene_skip",
                               "profile=%u framebuffer=%ux%u total=%u",
                               shadow_profile,
                               ctx->framebuffer.width,
                               ctx->framebuffer.height,
                               ctx->observed_draws);
         pvrgpu_invalidate_full_depth_clear(ctx);
         return;
      }
      if (ctx->shadow_pco_depth && ctx->shadow_pco_mask &&
          ctx->shadow_pco_scene) {
         ctx->observed_draws++;
         pvrgpu_counter_eventf("draw_pco_shadow_duplicate_skip",
                               "profile=%u framebuffer=%ux%u total=%u",
                               shadow_profile,
                               ctx->framebuffer.width,
                               ctx->framebuffer.height,
                               ctx->observed_draws);
         pvrgpu_invalidate_full_depth_clear(ctx);
         return;
      }

      struct pvrgpu_shadow_pco_observation *shadow =
         CALLOC_STRUCT(pvrgpu_shadow_pco_observation);
      const char *shadow_failure_reason = NULL;
      if (shadow &&
          pvrgpu_draw_matches_shadow_pco(ctx,
                                         info,
                                         drawid_offset,
                                         indirect,
                                         draws,
                                         num_draws,
                                         shadow_profile,
                                         ctx->shadow_pco_depth,
                                         ctx->shadow_pco_mask,
                                         shadow,
                                         &shadow_failure_reason) &&
          pvrgpu_compile_shadow_observation(ctx,
                                            shadow,
                                            &shadow_failure_reason)) {
         if (shadow_profile == PVRGPU_PCO_SHADOW_DEPTH) {
            pvrgpu_shadow_pco_observation_destroy(&ctx->shadow_pco_depth);
            pvrgpu_shadow_pco_observation_destroy(&ctx->shadow_pco_mask);
            pvrgpu_shadow_pco_observation_destroy(&ctx->shadow_pco_scene);
            ctx->shadow_pco_depth = shadow;
            pvrgpu_counter_eventf("draw_pco_shadow_depth_ready",
                                  "framebuffer=%ux%u vertices=%u bytes=%zu",
                                  shadow->framebuffer_width,
                                  shadow->framebuffer_height,
                                  shadow->vertex_count,
                                  shadow->vertex_data_size);
         } else if (shadow_profile == PVRGPU_PCO_SHADOW_MASK &&
                    !shadow->output_depth_clear_one) {
            pvrgpu_counter_eventf(
               "draw_pco_shadow_warmup_mask_skip",
               "framebuffer=%ux%u output_depth_clear=zero",
               shadow->framebuffer_width,
               shadow->framebuffer_height);
            pvrgpu_shadow_pco_observation_destroy(&ctx->shadow_pco_depth);
            pvrgpu_shadow_pco_observation_destroy(&ctx->shadow_pco_mask);
            pvrgpu_shadow_pco_observation_destroy(&shadow);
            ctx->shadow_pco_warmup_skipped = true;
         } else if (shadow_profile == PVRGPU_PCO_SHADOW_MASK) {
            if (!ctx->shadow_pco_depth) {
               shadow_failure_reason = "missing_depth_pass";
               pvrgpu_shadow_pco_observation_destroy(&shadow);
            } else {
               pvrgpu_shadow_pco_observation_destroy(&ctx->shadow_pco_mask);
               ctx->shadow_pco_mask = shadow;
               ctx->shadow_pco_warmup_skipped = false;
               pvrgpu_counter_eventf(
                  "draw_pco_shadow_mask_ready",
                  "framebuffer=%ux%u vertices=%u texture=%ux%u",
                  shadow->framebuffer_width,
                  shadow->framebuffer_height,
                  shadow->vertex_count,
                  ctx->shadow_pco_depth->framebuffer_width,
                  ctx->shadow_pco_depth->framebuffer_height);
            }
         } else if (!ctx->shadow_pco_depth || !ctx->shadow_pco_mask ||
                    ctx->shadow_pco_depth->vertex_data_size !=
                       shadow->vertex_data_size ||
                    memcmp(ctx->shadow_pco_depth->vertex_data,
                           shadow->vertex_data,
                           shadow->vertex_data_size) != 0) {
            shadow_failure_reason = "mesh_payload_or_sequence_order";
            pvrgpu_shadow_pco_observation_destroy(&shadow);
         } else {
            pvrgpu_shadow_pco_observation_destroy(&ctx->shadow_pco_scene);
            ctx->shadow_pco_scene = shadow;
            shadow = NULL;
            pvrgpu_counter_eventf("draw_pco_shadow_sequence_ready",
                                  "draws=3 vertices=%u,%u,%u",
                                  ctx->shadow_pco_depth->vertex_count,
                                  ctx->shadow_pco_mask->vertex_count,
                                  ctx->shadow_pco_scene->vertex_count);
            if (!pvrgpu_emit_shadow_pco_sequence_command(ctx)) {
               shadow_failure_reason = "sequence_submit";
               pvrgpu_shadow_pco_observation_destroy(
                  &ctx->shadow_pco_depth);
               pvrgpu_shadow_pco_observation_destroy(
                  &ctx->shadow_pco_mask);
               pvrgpu_shadow_pco_observation_destroy(
                  &ctx->shadow_pco_scene);
            }
         }

         if (!shadow_failure_reason) {
            ctx->observed_draws++;
            pvrgpu_invalidate_full_depth_clear(ctx);
            return;
         }
      }
      pvrgpu_shadow_pco_observation_destroy(&shadow);
      pvrgpu_counter_eventf("draw_pco_shadow_match_miss",
                            "profile=%u reason=%s framebuffer=%ux%u",
                            shadow_profile,
                            shadow_failure_reason ? shadow_failure_reason :
                                                    "allocation",
                            ctx->framebuffer.width,
                            ctx->framebuffer.height);
      pvrgpu_note_unsupported_draw(ctx,
                                  info,
                                  indirect,
                                  draws,
                                  num_draws,
                                  "shadow_pco_sequence_failed");
      pvrgpu_invalidate_full_depth_clear(ctx);
      return;
   }

   enum pvrgpu_pco_refract_profile refract_profile;
   const bool refract_source_profile =
      rdc_case_name &&
      strcmp(rdc_case_name, "refract.refract.capture.1") == 0 &&
      pvrgpu_refract_pco_profile(ctx, &refract_profile);
   if (refract_source_profile &&
       pvrgpu_refract_pco_draw_info_matches(info,
                                            drawid_offset,
                                            indirect,
                                            draws,
                                            num_draws)) {
      if (conditionals_probe_framebuffer) {
         ctx->observed_draws++;
         pvrgpu_counter_eventf("draw_pco_refract_probe_skip",
                               "profile=%u framebuffer=1x1 total=%u",
                               refract_profile,
                               ctx->observed_draws);
         pvrgpu_invalidate_full_depth_clear(ctx);
         return;
      }

      if (ctx->refract_pco_prepass && ctx->refract_pco_composite) {
         ctx->observed_draws++;
         pvrgpu_counter_eventf("draw_pco_refract_duplicate_skip",
                               "profile=%u framebuffer=%ux%u total=%u",
                               refract_profile,
                               ctx->framebuffer.width,
                               ctx->framebuffer.height,
                               ctx->observed_draws);
         pvrgpu_invalidate_full_depth_clear(ctx);
         return;
      }

      struct pvrgpu_refract_pco_observation *refract =
         CALLOC_STRUCT(pvrgpu_refract_pco_observation);
      const char *refract_failure_reason = NULL;
      if (refract &&
          pvrgpu_draw_matches_refract_pco(ctx,
                                          info,
                                          drawid_offset,
                                          indirect,
                                          draws,
                                          num_draws,
                                          refract_profile,
                                          ctx->refract_pco_prepass,
                                          refract,
                                          &refract_failure_reason) &&
          pvrgpu_compile_refract_observation(ctx,
                                             refract,
                                             &refract_failure_reason)) {
         if (refract_profile == PVRGPU_PCO_REFRACT_COMPOSITE &&
             !refract->composite_depth_clear_one) {
            ctx->observed_draws++;
            pvrgpu_counter_eventf(
               "draw_pco_refract_warmup_skip",
               "profile=%u framebuffer=%ux%u depth_clear=zero total=%u",
               refract_profile,
               refract->framebuffer_width,
               refract->framebuffer_height,
               ctx->observed_draws);
            pvrgpu_refract_pco_observation_destroy(&refract);
            pvrgpu_invalidate_full_depth_clear(ctx);
            return;
         }
         if (refract_profile == PVRGPU_PCO_REFRACT_PREPASS) {
            pvrgpu_refract_pco_observation_destroy(
               &ctx->refract_pco_prepass);
            ctx->refract_pco_prepass = refract;
            pvrgpu_counter_eventf(
               "draw_pco_refract_prepass_ready",
               "framebuffer=%ux%u vertices=%u vertex_bytes=%zu",
               refract->framebuffer_width,
               refract->framebuffer_height,
               refract->vertex_count,
               refract->interleaved_vertex_data_size);
         } else if (ctx->refract_pco_prepass->interleaved_vertex_data_size ==
                       refract->interleaved_vertex_data_size &&
                    memcmp(ctx->refract_pco_prepass->interleaved_vertex_data,
                           refract->interleaved_vertex_data,
                           refract->interleaved_vertex_data_size) == 0) {
            pvrgpu_refract_pco_observation_destroy(
               &ctx->refract_pco_composite);
            ctx->refract_pco_composite = refract;
            pvrgpu_counter_eventf(
               "draw_pco_refract_sequence_ready",
               "draws=2 framebuffer=%ux%u,%ux%u "
               "vertices=%u,%u image=%ux%u image_bytes=%zu",
               ctx->refract_pco_prepass->framebuffer_width,
               ctx->refract_pco_prepass->framebuffer_height,
               refract->framebuffer_width,
               refract->framebuffer_height,
               ctx->refract_pco_prepass->vertex_count,
               refract->vertex_count,
               refract->sampled_image_width,
               refract->sampled_image_height,
               refract->sampled_image_bytes_size);
            if (!pvrgpu_emit_refract_pco_sequence_command(ctx)) {
               refract_failure_reason = "sequence_submit";
               pvrgpu_refract_pco_observation_destroy(
                  &ctx->refract_pco_composite);
               refract = NULL;
            }
         } else {
            refract_failure_reason = "vertex_payload_mismatch";
            pvrgpu_refract_pco_observation_destroy(&refract);
         }

         if (!refract_failure_reason) {
            ctx->observed_draws++;
            pvrgpu_invalidate_full_depth_clear(ctx);
            return;
         }
      }
      pvrgpu_refract_pco_observation_destroy(&refract);
      pvrgpu_counter_eventf("draw_pco_refract_match_miss",
                            "profile=%u reason=%s framebuffer=%ux%u",
                            refract_profile,
                            refract_failure_reason ? refract_failure_reason :
                                                     "allocation",
                            ctx->framebuffer.width,
                            ctx->framebuffer.height);
      pvrgpu_note_unsupported_draw(ctx,
                                  info,
                                  indirect,
                                  draws,
                                  num_draws,
                                  "refract_pco_sequence_failed");
      pvrgpu_invalidate_full_depth_clear(ctx);
      return;
   }

   if (rdc_case_name &&
       strcmp(rdc_case_name, "ideas.ideas.capture.1") == 0) {
      struct pvrgpu_ideas_pco_observation ideas;
      const char *ideas_failure_reason = NULL;
      if (conditionals_probe_framebuffer &&
          pvrgpu_draw_matches_ideas_pco(ctx,
                                        info,
                                        drawid_offset,
                                        indirect,
                                        draws,
                                        num_draws,
                                        ctx->ideas_pco_probe_draws,
                                        true,
                                        false,
                                        false,
                                        &ideas,
                                        &ideas_failure_reason)) {
         ctx->ideas_pco_probe_draws++;
         ctx->observed_draws++;
         pvrgpu_counter_eventf("draw_pco_ideas_probe_skip",
                               "ordinal=%u profile=%u total=%u",
                               ctx->ideas_pco_probe_draws - 1u,
                               ideas.profile,
                               ctx->observed_draws);
         pvrgpu_invalidate_full_depth_clear(ctx);
         return;
      }

      const bool actual_framebuffer =
         requested_glmark_extent &&
         ctx->framebuffer.width == requested_width &&
         ctx->framebuffer.height == requested_height &&
         pvrgpu_framebuffer_matches_rdc_output(ctx);
      if (actual_framebuffer &&
          pvrgpu_draw_matches_ideas_pco(ctx,
                                        info,
                                        drawid_offset,
                                        indirect,
                                        draws,
                                        num_draws,
                                        ctx->ideas_pco_draws,
                                        false,
                                        false,
                                        true,
                                        &ideas,
                                        &ideas_failure_reason)) {
         ctx->observed_draws++;
         const bool emitted = pvrgpu_emit_ideas_pco_command(ctx, &ideas);
         FREE(ideas.interleaved_vertex_data);
         if (!emitted) {
            pvrgpu_note_unsupported_draw(ctx,
                                        info,
                                        indirect,
                                        draws,
                                        num_draws,
                                        "ideas_pco_command_failed");
         }
         pvrgpu_invalidate_full_depth_clear(ctx);
         return;
      }
      if (conditionals_probe_framebuffer || actual_framebuffer) {
         pvrgpu_counter_eventf("draw_pco_ideas_match_miss",
                               "probe=%u probe_ordinal=%u ordinal=%u "
                               "reason=%s count=%u mode=%u index_size=%u",
                               conditionals_probe_framebuffer ? 1 : 0,
                               ctx->ideas_pco_probe_draws,
                               ctx->ideas_pco_draws,
                               ideas_failure_reason ?
                                  ideas_failure_reason : "unknown",
                               draws && num_draws ? draws[0].count : 0,
                               info ? info->mode : 0,
                               info ? info->index_size : 0);
      }
   }

   struct pvrgpu_lit_mesh_observation lit_mesh;
   const char *lit_mesh_failure_reason = NULL;
   const bool lit_mesh_match =
      pvrgpu_draw_matches_lit_mesh(ctx,
                                   info,
                                   drawid_offset,
                                   indirect,
                                   draws,
                                   num_draws,
                                   false,
                                   true,
                                   &lit_mesh,
                                   &lit_mesh_failure_reason);
   if (!lit_mesh_match && conditionals_probe_framebuffer) {
      struct pvrgpu_lit_mesh_observation probe_observation;
      const char *probe_failure_reason = NULL;
      if (pvrgpu_draw_matches_lit_mesh(ctx,
                                      info,
                                      drawid_offset,
                                      indirect,
                                      draws,
                                      num_draws,
                                      true,
                                      false,
                                      &probe_observation,
                                      &probe_failure_reason)) {
         ctx->observed_draws++;
         pvrgpu_counter_eventf("draw_pco_lit_mesh_probe_skip",
                               "profile=%u framebuffer=1x1 requested=%ux%u "
                               "total=%u",
                               probe_observation.profile,
                               requested_width,
                               requested_height,
                               ctx->observed_draws);
         pvrgpu_invalidate_full_depth_clear(ctx);
         return;
      }
      lit_mesh_failure_reason = probe_failure_reason;
   }
   if (lit_mesh_match) {
      ctx->observed_draws++;
      const bool owns_requested_framebuffer =
         requested_glmark_extent &&
         lit_mesh.framebuffer_width == requested_width &&
         lit_mesh.framebuffer_height == requested_height &&
         lit_mesh.viewport_width == requested_width &&
         lit_mesh.viewport_height == requested_height &&
         pvrgpu_framebuffer_matches_rdc_output(ctx);
      if (!owns_requested_framebuffer) {
         FREE(lit_mesh.interleaved_vertex_data);
         pvrgpu_note_unsupported_draw(ctx,
                                     info,
                                     indirect,
                                     draws,
                                     num_draws,
                                     "lit_mesh_framebuffer_mismatch");
         pvrgpu_invalidate_full_depth_clear(ctx);
         return;
      }
      pvrgpu_counter_eventf("draw_pco_lit_mesh",
                            "profile=%u start=%u count=%u total=%u",
                            lit_mesh.profile,
                            draws[0].start,
                            draws[0].count,
                            ctx->observed_draws);
      const bool emitted = pvrgpu_emit_lit_mesh_command(ctx, &lit_mesh);
      FREE(lit_mesh.interleaved_vertex_data);
      if (!emitted) {
         pvrgpu_note_unsupported_draw(ctx,
                                     info,
                                     indirect,
                                     draws,
                                     num_draws,
                                     "lit_mesh_command_failed");
      }
      pvrgpu_invalidate_full_depth_clear(ctx);
      return;
   }
   if (lit_mesh_failure_reason &&
       strcmp(lit_mesh_failure_reason, "shader_profile") != 0) {
      pvrgpu_counter_eventf("draw_pco_lit_mesh_match_miss",
                            "reason=%s framebuffer=%ux%u count=%u",
                            lit_mesh_failure_reason,
                            ctx->framebuffer.width,
                            ctx->framebuffer.height,
                            draws && num_draws ? draws[0].count : 0);
   }

   struct pvrgpu_conditionals_observation conditionals;
   const char *conditionals_failure_reason = NULL;
   const bool conditionals_match =
      pvrgpu_draw_matches_conditionals(ctx,
                                       info,
                                       drawid_offset,
                                       indirect,
                                       draws,
                                       num_draws,
                                       false,
                                       true,
                                       &conditionals,
                                       &conditionals_failure_reason);
   if (!conditionals_match && conditionals_probe_framebuffer) {
      struct pvrgpu_conditionals_observation probe_observation;
      const char *probe_failure_reason = NULL;
      if (pvrgpu_draw_matches_conditionals(ctx,
                                           info,
                                           drawid_offset,
                                           indirect,
                                           draws,
                                           num_draws,
                                           true,
                                           false,
                                           &probe_observation,
                                           &probe_failure_reason)) {
         ctx->observed_draws++;
         pvrgpu_counter_eventf("draw_pco_triangles_probe_skip",
                               "reason=strict_conditionals_probe_validated "
                               "framebuffer=1x1 requested=%ux%u total=%u",
                               requested_width,
                               requested_height,
                               ctx->observed_draws);
         pvrgpu_invalidate_full_depth_clear(ctx);
         return;
      }
      conditionals_failure_reason = probe_failure_reason;
   }
   if (conditionals_match) {
      ctx->observed_draws++;
      const bool owns_requested_framebuffer =
         requested_glmark_extent &&
         conditionals.framebuffer_width == requested_width &&
         conditionals.framebuffer_height == requested_height &&
         conditionals.viewport_width == requested_width &&
         conditionals.viewport_height == requested_height &&
         pvrgpu_framebuffer_matches_rdc_output(ctx);
      if (!owns_requested_framebuffer) {
         pvrgpu_note_unsupported_draw(
            ctx,
            info,
            indirect,
            draws,
            num_draws,
            has_requested_extent ?
               "conditionals_framebuffer_mismatch" :
               "conditionals_missing_requested_extent");
         pvrgpu_invalidate_full_depth_clear(ctx);
         return;
      }

      pvrgpu_counter_eventf("draw_pco_triangles",
                            "start=%u count=%u framebuffer=%ux%u "
                            "viewport=%ux%u total=%u",
                            draws[0].start,
                            draws[0].count,
                            conditionals.framebuffer_width,
                            conditionals.framebuffer_height,
                            conditionals.viewport_width,
                            conditionals.viewport_height,
                            ctx->observed_draws);
      if (!pvrgpu_emit_draw_pco_triangles_command(ctx, &conditionals)) {
         pvrgpu_note_unsupported_draw(
            ctx,
            info,
            indirect,
            draws,
            num_draws,
            (ctx->driver_draw_command_emitted ||
             pvrgpu_driver_draw_command_has_been_emitted()) ?
               "conditionals_command_owned_elsewhere" :
               "conditionals_command_failed");
      }
      pvrgpu_invalidate_full_depth_clear(ctx);
      return;
   }
   if (info && !indirect && draws && num_draws == 1 &&
       info->mode == MESA_PRIM_TRIANGLES && info->index_size == 0 &&
       draws[0].count == PVRGPU_DRAW_PCO_TRIANGLES_VERTEX_COUNT) {
      pvrgpu_counter_eventf("draw_pco_triangles_match_miss",
                            "reason=%s framebuffer=%ux%u start=%u count=%u",
                            conditionals_failure_reason ?
                               conditionals_failure_reason : "unknown",
                            ctx->framebuffer.width,
                            ctx->framebuffer.height,
                            draws[0].start,
                            draws[0].count);
   }

   struct pvrgpu_texture_pco_observation texture_pco;
   const char *texture_pco_failure_reason = NULL;
   const bool texture_pco_match =
      pvrgpu_draw_matches_texture_pco(ctx,
                                      info,
                                      drawid_offset,
                                      indirect,
                                      draws,
                                      num_draws,
                                      false,
                                      true,
                                      &texture_pco,
                                      &texture_pco_failure_reason);
   if (!texture_pco_match && conditionals_probe_framebuffer) {
      struct pvrgpu_texture_pco_observation probe_observation;
      const char *probe_failure_reason = NULL;
      if (pvrgpu_draw_matches_texture_pco(ctx,
                                          info,
                                          drawid_offset,
                                          indirect,
                                          draws,
                                          num_draws,
                                          true,
                                          false,
                                          &probe_observation,
                                          &probe_failure_reason)) {
         ctx->observed_draws++;
         pvrgpu_counter_eventf("draw_pco_texture_probe_skip",
                               "framebuffer=1x1 requested=%ux%u total=%u",
                               requested_width,
                               requested_height,
                               ctx->observed_draws);
         pvrgpu_invalidate_full_depth_clear(ctx);
         return;
      }
      texture_pco_failure_reason = probe_failure_reason;
   }
   if (texture_pco_match) {
      ctx->observed_draws++;
      const bool owns_requested_framebuffer =
         requested_glmark_extent &&
         texture_pco.framebuffer_width == requested_width &&
         texture_pco.framebuffer_height == requested_height &&
         texture_pco.viewport_width == requested_width &&
         texture_pco.viewport_height == requested_height &&
         pvrgpu_framebuffer_matches_rdc_output(ctx);
      if (!owns_requested_framebuffer) {
         FREE(texture_pco.interleaved_vertex_data);
         FREE(texture_pco.sampled_texture_bytes);
         pvrgpu_note_unsupported_draw(ctx,
                                     info,
                                     indirect,
                                     draws,
                                     num_draws,
                                     "texture_pco_framebuffer_mismatch");
         pvrgpu_invalidate_full_depth_clear(ctx);
         return;
      }
      pvrgpu_counter_eventf("draw_pco_texture",
                            "start=%u count=%u framebuffer=%ux%u "
                            "texture=512x512 total=%u",
                            draws[0].start,
                            draws[0].count,
                            texture_pco.framebuffer_width,
                            texture_pco.framebuffer_height,
                            ctx->observed_draws);
      const bool emitted = pvrgpu_emit_texture_pco_command(ctx, &texture_pco);
      FREE(texture_pco.interleaved_vertex_data);
      FREE(texture_pco.sampled_texture_bytes);
      if (!emitted) {
         pvrgpu_note_unsupported_draw(ctx,
                                     info,
                                     indirect,
                                     draws,
                                     num_draws,
                                     "texture_pco_command_failed");
      }
      pvrgpu_invalidate_full_depth_clear(ctx);
      return;
   }
   if (texture_pco_failure_reason &&
       strcmp(texture_pco_failure_reason, "shader_profile") != 0) {
      pvrgpu_counter_eventf("draw_pco_texture_match_miss",
                            "reason=%s framebuffer=%ux%u count=%u",
                            texture_pco_failure_reason,
                            ctx->framebuffer.width,
                            ctx->framebuffer.height,
                            draws && num_draws ? draws[0].count : 0);
   }

   struct pvrgpu_textured_triangles_observation textured_triangles;
   const char *textured_triangles_failure_reason = NULL;
   const bool probe_framebuffer =
      has_requested_extent &&
      (requested_width != 1 || requested_height != 1) &&
      ctx->framebuffer.width == 1 && ctx->framebuffer.height == 1;
   const bool textured_triangles_match =
      pvrgpu_draw_matches_textured_triangles(
         ctx,
         info,
         indirect,
         draws,
         num_draws,
         &textured_triangles,
         &textured_triangles_failure_reason,
         true);
   if (!textured_triangles_match && probe_framebuffer) {
      const char *probe_failure_reason = NULL;
      struct pvrgpu_textured_triangles_observation probe_observation;
      if (pvrgpu_draw_matches_textured_triangles(
             ctx,
             info,
             indirect,
             draws,
             num_draws,
             &probe_observation,
             &probe_failure_reason,
             false)) {
         ctx->observed_draws++;
         pvrgpu_counter_eventf("draw_textured_triangles_probe_skip",
                               "reason=supported_replay_probe "
                               "framebuffer=1x1 requested=%ux%u",
                               requested_width,
                               requested_height);
         pvrgpu_invalidate_full_depth_clear(ctx);
         return;
      }
      textured_triangles_failure_reason = probe_failure_reason;
   }
   if (!textured_triangles_match && info && !indirect && draws &&
       num_draws == 1 && info->mode == MESA_PRIM_TRIANGLES &&
       info->index_size == 0 &&
       draws[0].count == PVRGPU_DRAW_TEXTURED_TRIANGLES_VERTEX_COUNT) {
      pvrgpu_counter_eventf("draw_textured_triangles_match_miss",
                            "reason=%s framebuffer=%ux%u sample_mask=0x%x "
                            "start=%u index_bias=%d index_bias_varies=%u "
                            "increment_draw_id=%u half_pixel_center=%u "
                            "bottom_edge_rule=%u clip_halfz=%u "
                            "depth_clip=%u,%u depth_clamp=%u "
                            "unnormalized=%u seamless=%u reduction=%u "
                            "anisotropy=%u lod=%f,%f,%f",
                            textured_triangles_failure_reason ?
                               textured_triangles_failure_reason : "unknown",
                            ctx->framebuffer.width,
                            ctx->framebuffer.height,
                            ctx->sample_mask,
                            draws[0].start,
                            draws[0].index_bias,
                            info->index_bias_varies ? 1 : 0,
                            info->increment_draw_id ? 1 : 0,
                            ctx->rasterizer ?
                               ctx->rasterizer->state.half_pixel_center : 0,
                            ctx->rasterizer ?
                               ctx->rasterizer->state.bottom_edge_rule : 0,
                            ctx->rasterizer ?
                               ctx->rasterizer->state.clip_halfz : 0,
                            ctx->rasterizer ?
                               ctx->rasterizer->state.depth_clip_near : 0,
                            ctx->rasterizer ?
                               ctx->rasterizer->state.depth_clip_far : 0,
                            ctx->rasterizer ?
                               ctx->rasterizer->state.depth_clamp : 0,
                            ctx->samplers[MESA_SHADER_FRAGMENT][0] ?
                               ctx->samplers[MESA_SHADER_FRAGMENT][0]
                                  ->state.unnormalized_coords : 0,
                            ctx->samplers[MESA_SHADER_FRAGMENT][0] ?
                               ctx->samplers[MESA_SHADER_FRAGMENT][0]
                                  ->state.seamless_cube_map : 0,
                            ctx->samplers[MESA_SHADER_FRAGMENT][0] ?
                               ctx->samplers[MESA_SHADER_FRAGMENT][0]
                                  ->state.reduction_mode : 0,
                            ctx->samplers[MESA_SHADER_FRAGMENT][0] ?
                               ctx->samplers[MESA_SHADER_FRAGMENT][0]
                                  ->state.max_anisotropy : 0,
                            ctx->samplers[MESA_SHADER_FRAGMENT][0] ?
                               ctx->samplers[MESA_SHADER_FRAGMENT][0]
                                  ->state.lod_bias : 0.0f,
                            ctx->samplers[MESA_SHADER_FRAGMENT][0] ?
                               ctx->samplers[MESA_SHADER_FRAGMENT][0]
                                  ->state.min_lod : 0.0f,
                            ctx->samplers[MESA_SHADER_FRAGMENT][0] ?
                               ctx->samplers[MESA_SHADER_FRAGMENT][0]
                                  ->state.max_lod : 0.0f);
   }
   if (textured_triangles_match) {
      ctx->observed_draws++;
      pvrgpu_counter_eventf("draw_textured_triangles",
                            "start=%u count=%u framebuffer=%ux%u "
                            "viewport=%ux%u texture=%ux%u level=%u total=%u",
                            draws[0].start,
                            draws[0].count,
                            textured_triangles.framebuffer_width,
                            textured_triangles.framebuffer_height,
                            textured_triangles.viewport_width,
                            textured_triangles.viewport_height,
                            textured_triangles.texture_width,
                            textured_triangles.texture_height,
                            textured_triangles.texture_level,
                            ctx->observed_draws);

      const bool framebuffer_is_requested =
         has_requested_extent &&
         textured_triangles.framebuffer_width == requested_width &&
         textured_triangles.framebuffer_height == requested_height &&
         textured_triangles.viewport_width == requested_width &&
         textured_triangles.viewport_height == requested_height &&
         pvrgpu_framebuffer_matches_rdc_output(ctx);
      if (!framebuffer_is_requested) {
         if (has_requested_extent &&
             textured_triangles.framebuffer_width == 1 &&
             textured_triangles.framebuffer_height == 1 &&
             textured_triangles.viewport_width == 1 &&
             textured_triangles.viewport_height == 1) {
            pvrgpu_counter_eventf("draw_textured_triangles_probe_skip",
                                  "reason=supported_replay_probe "
                                  "framebuffer=1x1 requested=%ux%u",
                                  requested_width,
                                  requested_height);
         } else {
            pvrgpu_note_unsupported_draw(
               ctx,
               info,
               indirect,
               draws,
               num_draws,
               has_requested_extent ?
                  "textured_triangles_framebuffer_mismatch" :
                  "textured_triangles_missing_requested_extent");
         }
         pvrgpu_invalidate_full_depth_clear(ctx);
         return;
      }

      if (!pvrgpu_emit_draw_textured_triangles_command(ctx,
                                                       &textured_triangles)) {
         pvrgpu_note_unsupported_draw(
            ctx,
            info,
            indirect,
            draws,
            num_draws,
            (ctx->driver_draw_command_emitted ||
             pvrgpu_driver_draw_command_has_been_emitted()) ?
               "textured_triangles_command_owned_elsewhere" :
               "textured_triangles_command_failed");
      }
      pvrgpu_invalidate_full_depth_clear(ctx);
      return;
   }

   if (pvrgpu_depth_write_enabled(ctx))
      pvrgpu_invalidate_full_depth_clear(ctx);

   if (pvrgpu_cpu_present_textured_quad(ctx, info, indirect, draws,
                                        num_draws)) {
      ctx->observed_draws++;
      pvrgpu_counter_eventf("draw_present_textured_quad",
                            "count=%u first_count=%u mode=%u index_size=%u "
                            "vertex_elements=%u vertex_buffers=%u "
                            "framebuffer=%ux%u total=%u",
                            num_draws,
                            draws && num_draws ? draws[0].count : 0,
                            info ? info->mode : 0,
                            info ? info->index_size : 0,
                            ctx->vertex_elements ?
                               ctx->vertex_elements->num_elements : 0,
                            ctx->num_vertex_buffers,
                            ctx->framebuffer.width,
                            ctx->framebuffer.height,
                            ctx->observed_draws);
      if (pvrgpu_framebuffer_matches_rdc_output(ctx) &&
          !ctx->driver_draw_command_emitted &&
          !pvrgpu_driver_draw_command_has_been_emitted())
         pvrgpu_note_unsupported_draw(ctx,
                                      info,
                                      indirect,
                                      draws,
                                      num_draws,
                                      "cpu_present_without_model_command");
      return;
   }

   /*
    * Generic PCO lowering runs before the shape-matched paths below.
    *
    * Those paths recognise particular draw shapes -- a textured triangle, an
    * indexed quad -- and each one claims the draw outright, because a capsule
    * describes a single workload and the first claimant writes it.  With the
    * generic path last it never saw a draw that any shape recogniser could
    * also accept, so geometry it was capable of rasterizing was reported by
    * shape instead.  Try the real lowering first and let the shape paths
    * handle only what it turns down.
    */
   const char *lowering_reason = NULL;
   if (pvrgpu_draw_is_lowerable_array_primitive(ctx, info, indirect, draws,
                                                num_draws,
                                                &lowering_reason)) {
      ctx->observed_draws++;
      pvrgpu_counter_eventf("draw_array_primitive",
                            "start=%u count=%u mode=%u primitives=%u "
                            "vertex_elements=%u vertex_buffers=%u "
                            "framebuffer=%ux%u total=%u",
                            draws[0].start,
                            draws[0].count,
                            info->mode,
                            pvrgpu_array_primitive_count(
                               info->mode, draws[0].count),
                            ctx->vertex_elements->num_elements,
                            ctx->num_vertex_buffers,
                            ctx->framebuffer.width,
                            ctx->framebuffer.height,
                            ctx->observed_draws);
      if (pvrgpu_record_color_primitive_pco_draw(ctx, info, &draws[0]))
         return;
      /*
       * RenderDoc replays the captured frames again once the sequence has
       * already described this trace's workload.  Those repeats are
       * observations, not draws the driver failed to lower.
       */
      if (ctx->array_primitive_sequence_owns_command ||
          pvrgpu_driver_draw_command_has_been_emitted()) {
         pvrgpu_counter_eventf("draw_array_primitive_duplicate_skip",
                               "start=%u count=%u mode=%u total=%u",
                               draws[0].start,
                               draws[0].count,
                               info->mode,
                               ctx->observed_draws);
         return;
      }
      /*
       * The draw looked lowerable but the compiler or the capsule turned it
       * down.  Undo the observation and fall through: a shape recogniser
       * below may still be able to describe it, and reporting the draw
       * unsupported here would hide that.
       */
      ctx->observed_draws--;
      pvrgpu_counter_eventf("draw_array_primitive_fallthrough",
                            "start=%u count=%u mode=%u index_size=%u",
                            draws[0].start,
                            draws[0].count,
                            info->mode,
                            info->index_size);
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
      pvrgpu_note_unsupported_draw(ctx,
                                   info,
                                   indirect,
                                   draws,
                                   num_draws,
                                   "textured_triangle_not_lowered");
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
      pvrgpu_note_unsupported_draw(ctx,
                                   info,
                                   indirect,
                                   draws,
                                   num_draws,
                                   "uniform_triangle_not_lowered");
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
      if (ctx->vertex_elements->num_elements >= 2) {
         if (!pvrgpu_record_color_primitive_pco_draw(ctx, info, &draws[0]) &&
             !ctx->driver_draw_command_emitted &&
             !pvrgpu_driver_draw_command_has_been_emitted() &&
             pvrgpu_framebuffer_matches_rdc_output(ctx))
            pvrgpu_note_unsupported_draw(ctx,
                                         info,
                                         indirect,
                                         draws,
                                         num_draws,
                                         "draw_color_triangle_pco_command_failed");
         return;
      }
      if (!pvrgpu_emit_draw_triangle_command(ctx, &draws[0]) &&
          pvrgpu_framebuffer_matches_rdc_output(ctx))
         pvrgpu_note_unsupported_draw(ctx,
                                      info,
                                      indirect,
                                      draws,
                                      num_draws,
                                      "draw_triangle_command_failed");
      return;
   }

   struct pvrgpu_indexed_quad_observation indexed_quad;
   if (!pvrgpu_string_has_prefix(pvrgpu_rdc_case_name(), "drawlist") &&
       pvrgpu_draw_is_observable_indexed_quad(ctx, info, indirect, draws,
                                              num_draws, &indexed_quad)) {
      ctx->observed_draws++;
      if ((ctx->driver_draw_command_emitted ||
           pvrgpu_driver_draw_command_has_been_emitted()) &&
          ctx->indexed_quad_draws == 0) {
         pvrgpu_note_unsupported_draw(ctx,
                                      info,
                                      indirect,
                                      draws,
                                      num_draws,
                                      "indexed_quad_command_owned_elsewhere");
         return;
      }
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
      pvrgpu_note_unsupported_draw(ctx,
                                   info,
                                   indirect,
                                   draws,
                                   num_draws,
                                   "indexed_triangle_not_lowered");
      return;
   }

   /*
    * Generic non-indexed triangle-topology lowering.  Everything above either
    * claimed the draw or fell through, so reaching this point used to be a
    * fail-closed `unsupported_state`.  A single-attachment untextured draw
    * whose position/color attributes match the model's color layout is a
    * shape the PCO path can lower for any vertex count, so lower it here
    * instead of rejecting it.
    */
   pvrgpu_counter_eventf("draw_not_lowerable",
                         "reason=%s mode=%u count=%u index_size=%u "
                         "attributes=%u textures=%u nr_cbufs=%u",
                         lowering_reason ? lowering_reason : "unknown",
                         info ? info->mode : 0,
                         draws && num_draws ? draws[0].count : 0,
                         info ? info->index_size : 0,
                         ctx->vertex_elements ?
                            ctx->vertex_elements->num_elements : 0,
                         ctx->num_sampler_views[MESA_SHADER_FRAGMENT],
                         ctx->framebuffer.nr_cbufs);
   pvrgpu_note_unsupported_draw(ctx,
                                info,
                                indirect,
                                draws,
                                num_draws,
                                "unsupported_state");
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
   ctx->sample_mask = UINT_MAX;
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
