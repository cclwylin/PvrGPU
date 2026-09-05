/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_CMD_H
#define PVRGPU_CMD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "pvrgpu_systemc_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PVRGPU_DRIVER_COMMAND_SCHEMA "pvrgpu.driver-command.v1"
#define PVRGPU_DRIVER_COMMAND_PRODUCER "pvrgpu-gallium-driver"
#define PVRGPU_DRIVER_COMMAND_FORMAT_RGBA8 "PIPE_FORMAT_R8G8B8A8_UNORM"
#define PVRGPU_DRIVER_COMMAND_FORMAT_RGBX8 "PIPE_FORMAT_R8G8B8X8_UNORM"
#define PVRGPU_DRIVER_COMMAND_FORMAT_BGRX8 "PIPE_FORMAT_B8G8R8X8_UNORM"
#define PVRGPU_DRIVER_COMMAND_FORMAT_R5G6B5 "PIPE_FORMAT_R5G6B5_UNORM"
#define PVRGPU_DRIVER_COMMAND_FORMAT_B5G6R5 "PIPE_FORMAT_B5G6R5_UNORM"
#define PVRGPU_DRIVER_COMMAND_FORMAT_R10G10B10A2 "PIPE_FORMAT_R10G10B10A2_UNORM"
#define PVRGPU_DRIVER_COMMAND_FORMAT_B10G10R10A2 "PIPE_FORMAT_B10G10R10A2_UNORM"

struct pvrgpu_clear_color_command {
   const char *case_name;
   uint32_t frame;
   uint32_t width;
   uint32_t height;
   const char *format;
   uint32_t clear_color_bits[4];
};

struct pvrgpu_draw_triangle_command {
   const char *case_name;
   uint32_t frame;
   uint32_t width;
   uint32_t height;
   const char *format;
   uint32_t clear_color_bits[4];
   uint32_t vertex_bits[3][2];
   uint32_t fragment_color_bits[4];
};

struct pvrgpu_draw_indexed_quad_command {
   const char *case_name;
   uint32_t frame;
   uint32_t framebuffer_width;
   uint32_t framebuffer_height;
   uint32_t width;
   uint32_t height;
   const char *format;
   uint32_t clear_color_bits[4];
   uint32_t draw_count;
   uint32_t index_count;
   uint32_t unique_vertices;
   uint32_t primitive_count;
   uint32_t clip_primitives;
   uint32_t setup_triangles;
   uint64_t semantic_texel_fetches;
};

#define PVRGPU_DRAW_TEXTURED_TRIANGLES_VERTEX_COUNT 6u

struct pvrgpu_draw_textured_triangles_command {
   const char *case_name;
   uint32_t frame;
   uint32_t framebuffer_width;
   uint32_t framebuffer_height;
   uint32_t width;
   uint32_t height;
   const char *format;
   uint32_t clear_color_bits[4];
   uint32_t vertex_bits[PVRGPU_DRAW_TEXTURED_TRIANGLES_VERTEX_COUNT][2];
   uint32_t texcoord_bits[PVRGPU_DRAW_TEXTURED_TRIANGLES_VERTEX_COUNT][2];
   uint32_t texture_width;
   uint32_t texture_height;
   const char *texture_rgba8_path;
};

#define PVRGPU_DRAW_PCO_TRIANGLES_VERTEX_COUNT 6144u
#define PVRGPU_DRAW_PCO_TRIANGLES_VERTEX_STRIDE 12u
#define PVRGPU_DRAW_PCO_TRIANGLES_VERTEX_BYTES \
   (PVRGPU_DRAW_PCO_TRIANGLES_VERTEX_COUNT * \
    PVRGPU_DRAW_PCO_TRIANGLES_VERTEX_STRIDE)
#define PVRGPU_DRAW_PCO_TRIANGLES_VS_SHARED_DWORDS 16u
#define PVRGPU_DRAW_PCO_TRIANGLES_FS_SHARED_DWORDS 4u
#define PVRGPU_DRAW_PCO_TRIANGLES_VS_TEMPS 10u
#define PVRGPU_DRAW_PCO_TRIANGLES_VS_INPUTS 4u
#define PVRGPU_DRAW_PCO_TRIANGLES_VS_OUTPUTS 4u
#define PVRGPU_DRAW_PCO_TRIANGLES_FS_TEMPS 4u
#define PVRGPU_DRAW_PCO_TRIANGLES_VS_PCO_BYTES 520u
#define PVRGPU_DRAW_PCO_TRIANGLES_FS_PCO_BYTES 520u

struct pvrgpu_draw_pco_stage_abi {
   uint32_t temps;
   uint32_t vertex_inputs;
   uint32_t vertex_outputs;
   uint32_t coefficients;
   uint32_t shareds;
   uint32_t push_constant_start;
   uint32_t push_constant_count;
   uint32_t entry_offset;
};

struct pvrgpu_draw_pco_triangles_command {
   const char *case_name;
   uint32_t frame;
   uint32_t framebuffer_width;
   uint32_t framebuffer_height;
   uint32_t width;
   uint32_t height;
   const char *format;
   uint32_t clear_color_bits[4];

   const uint8_t *raw_vertex_data;
   size_t raw_vertex_data_size;
   uint32_t vertex_stride;
   uint32_t vertex_count;
   uint32_t first_vertex;
   uint32_t instance_count;
   uint32_t primitive_mode;
   uint32_t indexed;

   /*
    * Index payload for an indexed draw.  The model fetches through these
    * indices itself, so the driver forwards the buffer rather than
    * dereferencing it: that keeps index reuse visible to the vertex cache and
    * lets SystemC report vs_invocations below ia_vertices as real hardware
    * does.  All four fields are zero for a non-indexed draw.
    */
   /*
    * Colour attachments this draw writes.  One for an ordinary draw; a
    * deferred pass writing a G-buffer states how many its fragment shader
    * produces.  Every attachment shares the command's format and extent.
    */
   uint32_t render_target_count;

   /*
    * Packed vertex attribute layout.  Attribute N occupies
    * vertex_attribute_components[N] consecutive floats, laid out in order, and
    * lands in VTXIN register 4 * N.  The model matches a program's read mask
    * against these widths exactly.
    */
   uint32_t vertex_attribute_count;
   uint32_t vertex_attribute_components[16];
   uint32_t vertex_attribute_integer[16];

   const uint8_t *raw_index_data;
   size_t raw_index_data_size;
   uint32_t index_size;
   uint32_t index_count;
   uint32_t first_index;
   int32_t base_vertex;

   /* Optional whole-sequence API counter contract.  A multi-draw lowering
    * carries these totals on its first command only; single-draw profiles
    * leave them zero and use the counters measured by SystemC directly. */
   uint32_t draw_count;
   uint32_t ia_vertices;
   uint32_t ia_primitives;
   uint32_t vs_invocations;
   uint32_t gs_invocations;
   uint32_t gs_primitives;
   uint32_t clip_invocations;
   uint32_t clip_primitives;
   uint32_t hs_invocations;
   uint32_t ds_invocations;
   uint32_t cs_invocations;
   uint64_t ps_invocations;
   uint32_t setup_triangles;
   uint64_t semantic_texel_fetches;

   const uint8_t *vertex_pco;
   size_t vertex_pco_size;
   const uint8_t *fragment_pco;
   size_t fragment_pco_size;
   const uint32_t *vertex_shared;
   size_t vertex_shared_count;
   const uint32_t *fragment_shared;
   size_t fragment_shared_count;

   uint32_t sampled_texture_count;
   const uint8_t *sampled_texture_bytes;
   size_t sampled_texture_bytes_size;
   uint32_t sampled_texture_width;
   uint32_t sampled_texture_height;
   uint32_t sampled_texture_row_pitch;
   const char *sampled_texture_format;
   uint32_t sampled_texture_mip_count;

   struct pvrgpu_draw_pco_stage_abi vertex_pco_abi;
   struct pvrgpu_draw_pco_stage_abi fragment_pco_abi;
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
   uint32_t scissor_x;
   uint32_t scissor_y;
   uint32_t scissor_width;
   uint32_t scissor_height;
   uint32_t line_width_bits;
   uint32_t point_size_bits;
   uint32_t point_size_output_start;
   uint32_t point_size_output_count;
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
   /* Index 0 is the front face, index 1 the back. */
   uint32_t stencil_enable;
   uint32_t stencil_clear;
   uint32_t stencil_func[2];
   uint32_t stencil_fail_op[2];
   uint32_t stencil_depth_fail_op[2];
   uint32_t stencil_pass_op[2];
   uint32_t stencil_value_mask[2];
   uint32_t stencil_write_mask[2];
   uint32_t stencil_ref[2];
   /* Scissored depth/stencil clears issued since the previous draw. */
   const struct pvrgpu_systemc_attachment_clear *attachment_clears;
   uint32_t attachment_clear_count;
};

struct pvrgpu_systemc_driver_command;

bool
pvrgpu_write_clear_color_command(const char *path,
                                 const struct pvrgpu_clear_color_command *cmd,
                                 char *error,
                                 size_t error_size);

bool
pvrgpu_write_draw_triangle_command(const char *path,
                                   const struct pvrgpu_draw_triangle_command *cmd,
                                   char *error,
                                   size_t error_size);

bool
pvrgpu_write_draw_indexed_quad_command(
   const char *path,
   const struct pvrgpu_draw_indexed_quad_command *cmd,
   char *error,
   size_t error_size);

bool
pvrgpu_write_draw_textured_triangles_command(
   const char *path,
   const struct pvrgpu_draw_textured_triangles_command *cmd,
   char *error,
   size_t error_size);

bool
pvrgpu_write_draw_pco_triangles_command(
   const char *path,
   const struct pvrgpu_draw_pco_triangles_command *cmd,
   char *error,
   size_t error_size);

/* Validate one PCO triangles command without submitting it. */
bool
pvrgpu_validate_draw_pco_triangles_command(
   const char *path,
   const struct pvrgpu_draw_pco_triangles_command *cmd,
   char *error,
   size_t error_size);

/* Project one PCO triangles command onto the public SystemC command layout so
 * a caller can embed it as a nested sequence draw instead of submitting it. */
void
pvrgpu_pco_triangles_command_to_systemc(
   const struct pvrgpu_draw_pco_triangles_command *cmd,
   struct pvrgpu_systemc_driver_command *out);

/* Submit one API-v6 logical PCO sequence.  The public SystemC command is used
 * directly so nested draw/resource pointers have one authoritative layout;
 * the bridge deep-copies every transient payload before this call returns. */
bool
pvrgpu_write_draw_pco_sequence_command(
   const char *path,
   const struct pvrgpu_systemc_driver_command *cmd,
   char *error,
   size_t error_size);

/*
 * True when the model's ISS can decode this compiled binary.  A draw whose
 * shader it cannot execute must be declined before the driver claims it.
 */
bool
pvrgpu_pco_binary_is_executable(uint32_t stage,
                                const uint8_t *binary,
                                size_t binary_size,
                                char *error,
                                size_t error_size);

/*
 * Run everything submitted since the last flush and copy the RGBA8 result into
 * `pixels`.  `*out_written` says whether the model actually published pixels
 * for this surface; when it did not, `pixels` is untouched.  Returns false
 * only when the flush itself failed.
 */
bool
pvrgpu_systemc_flush_readback_rgba8(uint32_t width,
                                    uint32_t height,
                                    uint8_t *pixels,
                                    size_t pixels_size,
                                    bool *out_written,
                                    char *error,
                                    size_t error_size);

bool
pvrgpu_driver_draw_command_has_been_emitted(void);

void
pvrgpu_note_driver_draw_command_emitted(void);

/*
 * Reopen the once-per-frame draw-command gate.  A readback has run the model,
 * so the frame it described is finished and the draws that follow are the next
 * one's.
 */
void
pvrgpu_reset_driver_draw_command_emitted(void);

bool
pvrgpu_driver_counter_sequence_command_has_been_emitted(void);

void
pvrgpu_note_driver_counter_sequence_command_emitted(void);

/* These captures own one deferred native multi-pass submission.  Legacy
 * clear/blit/single-draw helpers must not claim the process-global command
 * slot while their physical passes are still being collected. */
bool
pvrgpu_case_reserves_native_pco_sequence(void);

#ifdef __cplusplus
}
#endif

#endif /* PVRGPU_CMD_H */
