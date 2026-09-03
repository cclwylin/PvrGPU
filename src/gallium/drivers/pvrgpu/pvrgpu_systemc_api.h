/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_SYSTEMC_API_H
#define PVRGPU_SYSTEMC_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* API-v8 expands the by-value sequence-texture mip table from 10 to 15. */
#define PVRGPU_SYSTEMC_API_VERSION 8u
#define PVRGPU_SYSTEMC_MAX_PCO_SEQUENCE_COMMANDS 64u
#define PVRGPU_SYSTEMC_MAX_PCO_SEQUENCE_TEXTURES 16u
#define PVRGPU_SYSTEMC_MAX_TEXTURE_MIP_LEVELS 15u
#define PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR UINT32_MAX

struct pvrgpu_systemc_pco_stage_abi {
   uint32_t temps;
   uint32_t vertex_inputs;
   uint32_t vertex_outputs;
   uint32_t coefficients;
   uint32_t shareds;
   uint32_t push_constant_start;
   uint32_t push_constant_count;
   uint32_t entry_offset;
};

enum pvrgpu_systemc_pco_texture_source {
   PVRGPU_SYSTEMC_PCO_TEXTURE_EXTERNAL_PAYLOAD = 0,
   PVRGPU_SYSTEMC_PCO_TEXTURE_PREVIOUS_COLOR_ATTACHMENT = 1,
   PVRGPU_SYSTEMC_PCO_TEXTURE_PREVIOUS_DEPTH_ATTACHMENT = 2,
};

enum pvrgpu_systemc_pco_shader_stage {
   PVRGPU_SYSTEMC_PCO_SHADER_STAGE_VERTEX = 0,
   PVRGPU_SYSTEMC_PCO_SHADER_STAGE_FRAGMENT = 1,
};

enum pvrgpu_systemc_pco_blend_equation {
   PVRGPU_SYSTEMC_PCO_BLEND_EQUATION_ADD = 0,
   PVRGPU_SYSTEMC_PCO_BLEND_EQUATION_SUBTRACT = 1,
   PVRGPU_SYSTEMC_PCO_BLEND_EQUATION_REVERSE_SUBTRACT = 2,
   PVRGPU_SYSTEMC_PCO_BLEND_EQUATION_MIN = 3,
   PVRGPU_SYSTEMC_PCO_BLEND_EQUATION_MAX = 4,
};

enum pvrgpu_systemc_pco_blend_factor {
   PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ZERO = 0,
   PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE = 1,
   PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_SOURCE_ALPHA = 2,
   PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE_MINUS_SOURCE_ALPHA = 3,
   PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_SOURCE_COLOR = 4,
   PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE_MINUS_SOURCE_COLOR = 5,
   PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_DESTINATION_COLOR = 6,
   PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE_MINUS_DESTINATION_COLOR = 7,
   PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_DESTINATION_ALPHA = 8,
   PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE_MINUS_DESTINATION_ALPHA = 9,
};

enum pvrgpu_systemc_pco_texture_filter {
   PVRGPU_SYSTEMC_PCO_TEXTURE_FILTER_NEAREST = 0,
   PVRGPU_SYSTEMC_PCO_TEXTURE_FILTER_LINEAR = 1,
};

enum pvrgpu_systemc_pco_texture_mip_filter {
   PVRGPU_SYSTEMC_PCO_TEXTURE_MIP_FILTER_NONE = 0,
   PVRGPU_SYSTEMC_PCO_TEXTURE_MIP_FILTER_LINEAR = 1,
};

enum pvrgpu_systemc_pco_texture_wrap {
   PVRGPU_SYSTEMC_PCO_TEXTURE_WRAP_CLAMP_TO_EDGE = 0,
   PVRGPU_SYSTEMC_PCO_TEXTURE_WRAP_REPEAT = 1,
};

struct pvrgpu_systemc_pco_texture_mip {
   uint32_t width;
   uint32_t height;
   uint32_t row_pitch;
   uint32_t offset;
};

/*
 * Redundant structured metadata for one raw 20-dword Rogue combined
 * image/sampler descriptor.  Attachment sources have no byte payload at the
 * API boundary; the SystemC model binds the actual output of the preceding
 * physical draw.  External bytes retain the same transient lifetime as PCO
 * and VBO pointers and are deep-copied before submit returns.
 */
struct pvrgpu_systemc_pco_sequence_texture {
   uint32_t source;
   uint32_t stage;
   uint32_t producer_command_index;
   uint32_t descriptor_set;
   uint32_t binding;
   const char *format;
   const uint8_t *bytes;
   size_t bytes_size;
   size_t declared_bytes_size;
   uint32_t mip_count;
   struct pvrgpu_systemc_pco_texture_mip
      mip[PVRGPU_SYSTEMC_MAX_TEXTURE_MIP_LEVELS];
   uint32_t min_filter;
   uint32_t mag_filter;
   uint32_t mip_filter;
   uint32_t wrap_u;
   uint32_t wrap_v;
   uint32_t normalized_coordinates;
   uint32_t min_lod_u4_6;
   uint32_t max_lod_u4_6;
};

struct pvrgpu_systemc_driver_command {
   uint32_t version;
   const char *schema;
   const char *producer;
   const char *command;
   const char *case_name;
   const char *format;
   const char *framebuffer_rgba8_path;
   uint32_t frame;
   uint32_t framebuffer_width;
   uint32_t framebuffer_height;
   uint32_t width;
   uint32_t height;
   uint32_t clear_color_bits[4];
   uint32_t vertex_bits[6][2];
   uint32_t texcoord_bits[6][2];
   uint32_t fragment_color_bits[4];
   uint32_t texture_width;
   uint32_t texture_height;
   const char *texture_rgba8_path;
   uint32_t draw_count;
   uint32_t index_count;
   uint32_t unique_vertices;
   uint32_t primitive_count;
   uint32_t clip_primitives;
   uint32_t setup_triangles;
   uint32_t ia_vertices;
   uint32_t ia_primitives;
   uint32_t vs_invocations;
   uint32_t gs_invocations;
   uint32_t gs_primitives;
   uint32_t clip_invocations;
   uint32_t hs_invocations;
   uint32_t ds_invocations;
   uint32_t cs_invocations;
   uint64_t ps_invocations;
   uint64_t semantic_texel_fetches;

   /*
    * Legacy single-draw draw_pco_triangles payload. These pointers remain valid only for
    * the duration of pvrgpu_systemc_submit_driver_command(); a consumer that
    * queues work must deep-copy every pointed-to byte before returning.
    */
   const uint8_t *raw_vertex_data;
   size_t raw_vertex_data_size;
   uint32_t vertex_stride;
   uint32_t vertex_count;
   uint32_t first_vertex;
   uint32_t instance_count;
   uint32_t primitive_mode;
   uint32_t indexed;

   const uint8_t *vertex_pco;
   size_t vertex_pco_size;
   const uint8_t *fragment_pco;
   size_t fragment_pco_size;
   const uint32_t *vertex_shared;
   size_t vertex_shared_count;
   const uint32_t *fragment_shared;
   size_t fragment_shared_count;

   /*
    * Optional slot-zero sampled texture. The single-draw contract keeps the existing
    * draw_pco_triangles contract: it accepts either no
    * texture, with every field below zero/NULL, or one tightly packed 512x512
    * RGBX8 level.  The byte pointer has the same transient lifetime as the PCO
    * and VBO pointers above.
    */
   uint32_t sampled_texture_count;
   const uint8_t *sampled_texture_bytes;
   size_t sampled_texture_bytes_size;
   uint32_t sampled_texture_width;
   uint32_t sampled_texture_height;
   uint32_t sampled_texture_row_pitch;
   const char *sampled_texture_format;
   uint32_t sampled_texture_mip_count;

   struct pvrgpu_systemc_pco_stage_abi vertex_pco_abi;
   struct pvrgpu_systemc_pco_stage_abi fragment_pco_abi;
   uint32_t position_output_start;
   uint32_t position_output_count;
   uint32_t fragment_position_start;
   uint32_t fragment_position_count;
   uint32_t varying_output_start;
   uint32_t varying_output_count;
   uint32_t fragment_varying_start;
   uint32_t fragment_varying_count;

   uint32_t viewport_scale_bits[3];
   uint32_t viewport_translate_bits[3];
   uint32_t front_ccw;
   uint32_t cull_face;
   uint32_t fill_front;
   uint32_t fill_back;
   uint32_t scissor;
   uint32_t rasterizer_discard;
   uint32_t multisample;
   uint32_t half_pixel_center;
   uint32_t bottom_edge_rule;
   uint32_t clip_halfz;
   uint32_t depth_clip_near;
   uint32_t depth_clip_far;
   uint32_t depth_clamp;
   uint32_t sample_mask;
   uint32_t color_mask;
   uint32_t blend_enable;
   uint32_t dither;
   uint32_t depth_enable;
   uint32_t depth_write;
   uint32_t depth_func;
   uint32_t depth_clear_bits;
   uint32_t depth_format;

   /*
    * API-v8 render-pass continuity. UINT32_MAX creates and clears a new
    * attachment; any other value aliases and LOADs the exact attachment
    * produced by that earlier nested command ordinal.  Alias dimensions and
    * formats are validated before any SystemC work is queued.
    */
   uint32_t color_attachment_source_command_index;
   uint32_t depth_attachment_source_command_index;

   /* Explicit GLES blend state, even when blend_enable is zero. */
   uint32_t blend_rgb_equation;
   uint32_t blend_alpha_equation;
   uint32_t blend_source_rgb_factor;
   uint32_t blend_destination_rgb_factor;
   uint32_t blend_source_alpha_factor;
   uint32_t blend_destination_alpha_factor;

   /*
    * API-v8 native PCO sequence.  The outer logical command carries captured
    * counter metadata; each nested command uses the ordinary PCO draw fields
    * above and must leave its own sequence tail zero.  Resources may be
    * immutable external payloads or actual color/depth attachments produced
    * by an earlier command ordinal.  The bridge enforces the explicit public
    * count bounds and deep-copies every nested payload before returning.
    */
   uint32_t pco_sequence_command_count;
   const struct pvrgpu_systemc_driver_command *pco_sequence_commands;
   uint32_t pco_sequence_texture_count;
   const struct pvrgpu_systemc_pco_sequence_texture *pco_sequence_textures;
};

struct pvrgpu_systemc_submit_info {
   uint32_t version;
   const struct pvrgpu_systemc_driver_command *command;
   const char *jsonl_path;
   const char *stderr_path;
   const char *outdir;
   const char *memory_mode;
};

typedef int (*pvrgpu_systemc_submit_driver_command_fn)(
   const struct pvrgpu_systemc_submit_info *info,
   char *error,
   size_t error_size);

int
pvrgpu_systemc_submit_driver_command(
   const struct pvrgpu_systemc_submit_info *info,
   char *error,
   size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* PVRGPU_SYSTEMC_API_H */
