/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_CMD_H
#define PVRGPU_CMD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "pvrgpu_systemc_api.h"
#include "pvrgpu_systemc_compute_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Synchronous, independent compute entry point. The model alone mutates the
 * dispatch snapshots; no graphics command or framebuffer supplies results. */
bool pvrgpu_submit_compute_command(
   const struct pvrgpu_systemc_compute_dispatch *dispatch,
   struct pvrgpu_systemc_compute_stats *stats,
   char *error, size_t error_size);

bool pvrgpu_read_graphics_stats(
   uint64_t submission_generation,
   struct pvrgpu_systemc_graphics_stats *stats,
   char *error, size_t error_size);

bool pvrgpu_read_stream_output(
   struct pvrgpu_systemc_stream_output_readback *readback,
   char *error, size_t error_size);
bool pvrgpu_read_shader_image(
   struct pvrgpu_systemc_shader_image_readback *readback,
   char *error, size_t error_size);

#define PVRGPU_DRIVER_COMMAND_SCHEMA "pvrgpu.driver-command.v1"
#define PVRGPU_DRIVER_COMMAND_PRODUCER "pvrgpu-gallium-driver"
#define PVRGPU_DRIVER_COMMAND_FORMAT_RGBA8 "PIPE_FORMAT_R8G8B8A8_UNORM"
/*
 * The sRGB-encoded eight-bit colour target.  Its bytes are laid out exactly
 * like RGBA8, but the stored value is the sRGB transfer of a linear colour:
 * the model encodes the shader's linear PIXOUT on write and, when blending,
 * decodes the destination to linear, blends there, and re-encodes -- which is
 * what GLES does for an sRGB framebuffer with no toggle.
 */
#define PVRGPU_DRIVER_COMMAND_FORMAT_RGBA8_SRGB "PIPE_FORMAT_R8G8B8A8_SRGB"
#define PVRGPU_DRIVER_COMMAND_FORMAT_BGRA8_SRGB "PIPE_FORMAT_B8G8R8A8_SRGB"
#define PVRGPU_DRIVER_COMMAND_FORMAT_RGBX8 "PIPE_FORMAT_R8G8B8X8_UNORM"
#define PVRGPU_DRIVER_COMMAND_FORMAT_BGRX8 "PIPE_FORMAT_B8G8R8X8_UNORM"
#define PVRGPU_DRIVER_COMMAND_FORMAT_R5G6B5 "PIPE_FORMAT_R5G6B5_UNORM"
#define PVRGPU_DRIVER_COMMAND_FORMAT_B5G6R5 "PIPE_FORMAT_B5G6R5_UNORM"
/* Packed normalized render targets retain their actual four-byte word in
 * LOAD/PBE/writeback. Shader outputs are floats, not raw integer exports. */
#define PVRGPU_DRIVER_COMMAND_FORMAT_RGB10_A2 "PIPE_FORMAT_R10G10B10A2_UNORM"
#define PVRGPU_DRIVER_COMMAND_FORMAT_BGRA10_A2 "PIPE_FORMAT_B10G10R10A2_UNORM"
#define PVRGPU_DRIVER_COMMAND_FORMAT_R10G10B10A2 PVRGPU_DRIVER_COMMAND_FORMAT_RGB10_A2
#define PVRGPU_DRIVER_COMMAND_FORMAT_B10G10R10A2 PVRGPU_DRIVER_COMMAND_FORMAT_BGRA10_A2
/*
 * The 32-bit integer attachments.  Their pixel is the fragment shader's PIXOUT
 * lanes verbatim -- one dword per channel -- rather than four UNORM8 channels,
 * which is what dEQP's shader tests render into: a scalar result goes to
 * R32_UINT, a vec2 to RG32UI, and a vec3 or vec4 to RGBA32UI, GLES having no
 * three-channel integer target.  A pixel is therefore 4, 8 or 16 bytes wide
 * and nothing downstream may assume four.
 *
 * A signed result -- an int, an ivec, or a bvec, which dEQP reads back as
 * integers -- goes to the matching SINT format.  The stored pixel is the same
 * raw dword per channel; only the value's interpretation on the host differs,
 * so every width and path below treats the two alike.
 */
#define PVRGPU_DRIVER_COMMAND_FORMAT_R32UI "PIPE_FORMAT_R32_UINT"
#define PVRGPU_DRIVER_COMMAND_FORMAT_RG32UI "PIPE_FORMAT_R32G32_UINT"
#define PVRGPU_DRIVER_COMMAND_FORMAT_RGBA32UI "PIPE_FORMAT_R32G32B32A32_UINT"
#define PVRGPU_DRIVER_COMMAND_FORMAT_R32I "PIPE_FORMAT_R32_SINT"
#define PVRGPU_DRIVER_COMMAND_FORMAT_RG32I "PIPE_FORMAT_R32G32_SINT"
#define PVRGPU_DRIVER_COMMAND_FORMAT_RGBA32I "PIPE_FORMAT_R32G32B32A32_SINT"
/* Linear floating-point transport; PBE blending retains its full range. */
#define PVRGPU_DRIVER_COMMAND_FORMAT_RGBA32F "PIPE_FORMAT_R32G32B32A32_FLOAT"

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
   uint32_t uniform_buffer_descriptor_start;
   uint32_t uniform_buffer_descriptor_count;
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
   const struct pvrgpu_systemc_pco_uniform_buffer *uniform_buffers;
   uint32_t uniform_buffer_count;

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
   const uint8_t *geometry_pco;
   size_t geometry_pco_size;
   const uint32_t *geometry_shared;
   uint32_t geometry_shared_count;
   struct pvrgpu_draw_pco_stage_abi geometry_pco_abi;
   uint32_t geometry_input_primitive_vertices;
   uint32_t geometry_output_primitive;
   uint32_t geometry_max_vertices;
   uint32_t geometry_invocations;
   uint32_t geometry_input_stride_dwords;
   uint32_t geometry_vertices_per_instance;
   uint32_t geometry_layer_output_start;
   uint32_t geometry_layer_output_count;
   uint32_t geometry_primitive_id_output_start;
   uint32_t geometry_primitive_id_output_count;
   const struct pvrgpu_systemc_tessellation *tessellation;
   uint32_t position_output_start;
   uint32_t position_output_count;
   uint32_t fragment_position_start;
   uint32_t fragment_position_count;
   uint32_t varying_output_start;
   uint32_t varying_output_count;
   uint32_t fragment_varying_start;
   uint32_t fragment_varying_count;
   uint32_t varying_flat_mask;
   uint32_t fragment_output_mask[8];

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
   uint32_t sample_frequency;
   uint32_t alpha_to_coverage;
   uint32_t alpha_to_coverage_dither;
   uint32_t alpha_to_one;
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
 * Run everything submitted since the last flush and copy the result into
 * `pixels`.  `bytes_per_pixel` is the attachment's stored pixel width -- four
 * for UNORM8, four per 32-bit channel for an integer attachment -- and a
 * readback that does not name the width the model rendered publishes nothing.
 * `*out_written` says whether the model actually published pixels for this
 * surface; when it did not, `pixels` is untouched.  Returns false only when
 * the flush itself failed.
 */
bool
pvrgpu_systemc_flush_readback_pixels(uint32_t width,
                                     uint32_t height,
                                     uint32_t bytes_per_pixel,
                                     uint32_t attachment,
                                     uint32_t sample_count,
                                     uint32_t depth_format,
                                     uint32_t layer_count,
                                     uint8_t *pixels,
                                     size_t pixels_size,
                                     bool *out_written,
                                     char *error,
                                     size_t error_size);

bool
pvrgpu_driver_draw_command_has_been_emitted(void);

void
pvrgpu_note_driver_draw_command_emitted(void);

/* Changes on every attempted bridge submission, including clears and errors. */
uint64_t
pvrgpu_systemc_submission_generation(void);

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
