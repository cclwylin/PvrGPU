/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_SYSTEMC_API_H
#define PVRGPU_SYSTEMC_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* API-v26 adds independent native transform-feedback transport/readback. */
#define PVRGPU_SYSTEMC_API_VERSION 27u
#define PVRGPU_SYSTEMC_MAX_UNIFORM_BUFFERS_PER_STAGE 15u
#define PVRGPU_SYSTEMC_MAX_UNIFORM_BUFFER_BYTES (64u * 1024u)
/*
 * Draws one sequence may describe; must match the model's own bound.  This is
 * independent of how many attachments the sequence creates, which the model's
 * address map bounds separately.
 */
#define PVRGPU_SYSTEMC_MAX_PCO_SEQUENCE_COMMANDS 256u
#define PVRGPU_SYSTEMC_MAX_PCO_SEQUENCE_TEXTURES 16u
#define PVRGPU_SYSTEMC_MAX_TEXTURE_MIP_LEVELS 15u
#define PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR UINT32_MAX

/* Planes an attachment clear touches. */
#define PVRGPU_SYSTEMC_CLEAR_ASPECT_DEPTH 0x1u
#define PVRGPU_SYSTEMC_CLEAR_ASPECT_STENCIL 0x2u

/*
 * One rectangle of the depth/stencil attachment set before a draw ran.
 *
 * A whole-surface clear is stated once, as the draw's `depth_clear_bits` or
 * `stencil_clear`.  A scissored one is not describable that way, and dropping
 * it left the model testing against planes the application had already
 * overwritten -- dEQP's fragment_ops.stencil.* paints a grid of 21x21
 * rectangles into both planes and then draws over it.  Each draw therefore
 * carries the clears issued since the previous one, in issue order.
 */
struct pvrgpu_systemc_attachment_clear {
   uint32_t x;
   uint32_t y;
   uint32_t width;
   uint32_t height;
   uint32_t aspects;
   uint32_t depth_bits;
   uint32_t stencil_value;
};

struct pvrgpu_systemc_pco_stage_abi {
   uint32_t temps;
   uint32_t vertex_inputs;
   uint32_t vertex_outputs;
   uint32_t coefficients;
   uint32_t shareds;
   uint32_t push_constant_start;
   uint32_t push_constant_count;
   uint32_t entry_offset;
   /* API-v21: four DWORDs per UBO, after textures and before CB0 push data. */
   uint32_t uniform_buffer_descriptor_start;
   uint32_t uniform_buffer_descriptor_count;
};

enum pvrgpu_systemc_pco_texture_source {
   PVRGPU_SYSTEMC_PCO_TEXTURE_EXTERNAL_PAYLOAD = 0,
   PVRGPU_SYSTEMC_PCO_TEXTURE_PREVIOUS_COLOR_ATTACHMENT = 1,
   PVRGPU_SYSTEMC_PCO_TEXTURE_PREVIOUS_DEPTH_ATTACHMENT = 2,
};

enum pvrgpu_systemc_pco_shader_stage {
   PVRGPU_SYSTEMC_PCO_SHADER_STAGE_VERTEX = 0,
   PVRGPU_SYSTEMC_PCO_SHADER_STAGE_FRAGMENT = 1,
   PVRGPU_SYSTEMC_PCO_SHADER_STAGE_GEOMETRY = 2,
   PVRGPU_SYSTEMC_PCO_SHADER_STAGE_TESS_CONTROL = 3,
   PVRGPU_SYSTEMC_PCO_SHADER_STAGE_TESS_EVALUATION = 4,
};

/* Immutable patch pipeline. Domain: triangles=0, quads=1, isolines=2;
 * spacing: equal=0, fractional_even=1, fractional_odd=2.
 * Native TCS SH0..3=input descriptor, SH4..7=output descriptor; TES SH0..3
 * reads that output. UBO descriptors follow; addresses are model-relocated.
 * Output patch storage begins with outer[4], inner[2] then patch varyings,
 * followed by output_vertices AoS records. No tessellation result is input. */
#define PVRGPU_SYSTEMC_MAX_TESSELLATION_PATCHES 4096u
#define PVRGPU_SYSTEMC_MAX_TESSELLATION_INPUT_VERTICES (4096u * 32u)
struct pvrgpu_systemc_tessellation {
   const uint8_t *control_pco;
   size_t control_pco_size;
   const uint32_t *control_shared;
   uint32_t control_shared_count;
   struct pvrgpu_systemc_pco_stage_abi control_abi;
   const uint8_t *evaluation_pco;
   size_t evaluation_pco_size;
   const uint32_t *evaluation_shared;
   uint32_t evaluation_shared_count;
   struct pvrgpu_systemc_pco_stage_abi evaluation_abi;
   uint32_t input_vertices;
   uint32_t output_vertices;
   uint32_t vertices_per_instance;
   uint32_t input_stride_dwords;
   uint32_t output_vertex_stride_dwords;
   uint32_t per_vertex_offset_dwords;
   uint32_t patch_stride_dwords;
   uint32_t control_barrier_count;
   uint32_t domain;
   uint32_t spacing;
   uint32_t clockwise;
   uint32_t point_mode;
};

/*
 * Native transform feedback consumes the final pre-raster shader's raw
 * VTXOUT DWORDs.  Bindings name those physical DWORDs after compiler layout;
 * targets retain the Gallium byte range and append cursor.  The complete
 * resource snapshot is input storage only: StreamOutput writes it through
 * modeled GPU memory and readback returns that modeled result.
 */
#define PVRGPU_SYSTEMC_MAX_STREAM_OUTPUT_BUFFERS 4u
#define PVRGPU_SYSTEMC_MAX_STREAM_OUTPUT_BINDINGS 64u
#define PVRGPU_SYSTEMC_MAX_STREAM_OUTPUT_RESOURCE_BYTES (256u * 1024u * 1024u)
struct pvrgpu_systemc_stream_output_binding {
   uint32_t output_dword;
   uint32_t num_components;
   uint32_t output_buffer;
   uint32_t dst_offset_dwords;
   uint32_t stream;
};

struct pvrgpu_systemc_stream_output_target {
   uint32_t output_buffer;
   uint64_t resource_token;
   uint64_t target_token;
   const uint8_t *bytes;
   size_t bytes_size;
   uint32_t buffer_offset;
   uint32_t buffer_size;
   uint32_t internal_offset;
   uint32_t stride_dwords;
};

struct pvrgpu_systemc_stream_output {
   const struct pvrgpu_systemc_stream_output_binding *bindings;
   uint32_t binding_count;
   const struct pvrgpu_systemc_stream_output_target *targets;
   uint32_t target_count;
};

#define PVRGPU_SYSTEMC_MAX_VARYING_BINDINGS 64u
struct pvrgpu_systemc_varying_binding {
   uint32_t output_dword;
   uint32_t num_components;
   uint32_t coefficient_dword;
   uint32_t flat;
};

/* Immutable snapshot of the bound range, not the whole Gallium buffer.
 * block_index is stage-local NIR UBO index (Gallium constant buffer index - 1).
 * Descriptor words are [0, 0, bytes_size, 0] before model relocation. */
struct pvrgpu_systemc_pco_uniform_buffer {
   uint32_t stage;
   uint32_t block_index;
   const uint8_t *bytes;
   size_t bytes_size;
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
   PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_SOURCE_ALPHA_SATURATE = 10,
   PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_CONSTANT_COLOR = 11,
   PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR = 12,
   PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_CONSTANT_ALPHA = 13,
   PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA = 14,
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
   /*
    * The texture unit has always implemented mirrored repeat -- both its
    * nearest and its linear address paths fold the coordinate over a period
    * of twice the extent -- but the capsule had no way to ask for it, so
    * every GL_MIRRORED_REPEAT sampler was declined at the driver.
    */
   PVRGPU_SYSTEMC_PCO_TEXTURE_WRAP_MIRRORED_REPEAT = 2,
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
   uint32_t wrap_w;
   uint32_t normalized_coordinates;
   uint32_t min_lod_u4_6;
   uint32_t max_lod_u4_6;
   /*
    * The sampled image's dimensionality: 0 = plain 2D, 1 = 2D array.  A 2D
    * array stores `layers` complete 2D images per mip level, layer-minor
    * inside each level, and the shader's third texture coordinate selects one.
    * Plain 2D leaves layers at one.
    */
   uint32_t texture_kind;
   uint32_t layers;
   /* 0 等同 1；支援 1/2/4/8。MS 資料採 pixel-interleaved samples，
    * 僅外部 payload、2D/2D-array、單一 mip；row_pitch 包含全部 samples。 */
   uint32_t sample_count;
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
   /* Colour attachments this draw writes; one for an ordinary draw. */
   uint32_t render_target_count;
   /* Packed vertex attribute widths; attribute N lands in VTXIN 4 * N. */
   uint32_t vertex_attribute_count;
   uint32_t vertex_attribute_components[16];
   /*
    * Whether attribute N carries integers rather than floats.  The width alone
    * does not say: a shader reading gl_InstanceID or an ivec attribute needs
    * the raw 32-bit value in its VTXIN register, and reading those bits as a
    * float and writing them back is only exact by accident.
    */
   uint32_t vertex_attribute_integer[16];
   /*
    * Index payload for an indexed draw.  The bridge deep-copies the buffer
    * before returning, exactly as it does for the vertex payload.
    */
   const uint8_t *raw_index_data;
   size_t raw_index_data_size;
   uint32_t index_size;
   uint32_t first_index;
   int32_t base_vertex;

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
   /*
    * Bit N set: varying slot N is flat-qualified.  Its coefficient set is the
    * provoking vertex's value, not an interpolation plane; without this the
    * model interpolated every varying and a flat integer read back as the
    * plane's first term.
    */
   uint32_t varying_flat_mask;
   /*
    * PIXOUT lanes colour attachment N expects the fragment shader to write.
    * A four-channel attachment wants 0xf; a single-channel one wants 0x1, and
    * requiring all four rejected every shader whose output is narrower than a
    * vec4.
    */
   uint32_t fragment_output_mask[8];

   uint32_t viewport_scale_bits[3];
   uint32_t viewport_translate_bits[3];
   uint32_t front_ccw;
   uint32_t cull_face;
   uint32_t fill_front;
   uint32_t fill_back;
   uint32_t scissor;
   /* Scissor rectangle in framebuffer pixels; only read when scissor is set. */
   uint32_t scissor_x;
   uint32_t scissor_y;
   uint32_t scissor_width;
   uint32_t scissor_height;
   /* Line width and point size as IEEE-754 bits; 1.0f when unset. */
   uint32_t line_width_bits;
   uint32_t point_size_bits;
   /*
    * Vertex output holding gl_PointSize when the shader sizes each point.
    * A count of zero means every point uses point_size_bits instead.
    */
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

   /*
    * Stencil state, per GLES 3.0 4.1.4.  Index 0 is the front face and index 1
    * the back; a draw that does not enable the test still states them, so the
    * model never has to infer a face's operation from a default.  Masks and the
    * reference are carried at the width the application set them, and the model
    * narrows them to the plane.
    */
   uint32_t stencil_enable;
   uint32_t stencil_clear;
   uint32_t stencil_func[2];
   uint32_t stencil_fail_op[2];
   uint32_t stencil_depth_fail_op[2];
   uint32_t stencil_pass_op[2];
   uint32_t stencil_value_mask[2];
   uint32_t stencil_write_mask[2];
   uint32_t stencil_ref[2];
   /* Clears applied to the depth/stencil attachment before this draw. */
   const struct pvrgpu_systemc_attachment_clear *attachment_clears;
   uint32_t attachment_clear_count;

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
    * GLES blend constant colour (glBlendColor), as four IEEE-754 float bit
    * patterns R,G,B,A.  Only read when a CONSTANT_* blend factor selects it.
    */
   uint32_t blend_constant_color_bits[4];

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

   /*
    * API-v19 optional initial contents for a nested PCO draw's single color
    * attachment.  The source index must be ATTACHMENT_NEW_CLEAR and the
    * effective render_target_count must be one.  Rows are tightly packed in
    * the command format's model transport: RGBA8 is four bytes per pixel;
    * R32, RG32 and RGBA32 integer targets are four, eight and sixteen bytes.
    * Supply the complete framebuffer extent, or leave both fields zero.
    * Submission deep-copies these bytes.  The model imports them into DRAM
    * and performs a PBE LOAD before rasterizing the draw.
    */
   const uint8_t *initial_color_attachment_bytes;
   size_t initial_color_attachment_bytes_size;
   /* API-v20: samples per pixel, stored pixel-interleaved in attachment
    * payloads. Zero retains the single-sample default of older producers. */
   uint32_t raster_samples;
   /* Optional complete native depth/stencil LOAD, using depth_format. */
   const uint8_t *initial_depth_attachment_bytes;
   size_t initial_depth_attachment_bytes_size;
   /* API-v21 per-physical-draw payloads; all bytes are copied before return. */
   const struct pvrgpu_systemc_pco_uniform_buffer *uniform_buffers;
   uint32_t uniform_buffer_count;
   /* API-v22: boolean Gallium blend state, snapshotted independently per draw.
    * Coverage uses original DATA0 alpha before alpha-to-one and blending. */
   uint32_t alpha_to_coverage;
   uint32_t alpha_to_coverage_dither;
   uint32_t alpha_to_one;
   /* API-v24: optional immutable GS executable and stage-local SHARED bank.
    * SHARED0..3 is the primitive-input descriptor, relocated by the GS module.
    * Output primitive uses Gallium topology values: points=0, line_strip=3,
    * triangle_strip=5. Inputs are not expanded into triangles before GS. */
   const uint8_t *geometry_pco;
   size_t geometry_pco_size;
   const uint32_t *geometry_shared;
   uint32_t geometry_shared_count;
   struct pvrgpu_systemc_pco_stage_abi geometry_pco_abi;
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
   /* API-v25: null means no patch pipeline; payload is deep-copied. */
   const struct pvrgpu_systemc_tessellation *tessellation;
   /* API-v26: null means no transform feedback; payload is deep-copied. */
   const struct pvrgpu_systemc_stream_output *stream_output;
   /* Non-null with count zero explicitly means no FS varying linkage. */
   const struct pvrgpu_systemc_varying_binding *varying_bindings;
   uint32_t varying_binding_count;
   /* API-v27: zero is a non-layered framebuffer. Otherwise every attachment
    * contains this many tightly packed layer-major images and gl_Layer
    * selects one. Pixel/sample layout within each layer is unchanged. */
   uint32_t framebuffer_layers;
};

struct pvrgpu_systemc_submit_info {
   uint32_t version;
   const struct pvrgpu_systemc_driver_command *command;
   const char *jsonl_path;
   const char *stderr_path;
   const char *outdir;
   const char *memory_mode;
   /* Driver ownership token, not a primitive count or expected result. */
   uint64_t submission_generation;
};

typedef int (*pvrgpu_systemc_submit_driver_command_fn)(
   const struct pvrgpu_systemc_submit_info *info,
   char *error,
   size_t error_size);

/*
 * A readback of whatever the submitted work left in the model's DRAM.
 *
 * `pixels` is the caller's RGBA8 destination and `pixels_size` its capacity in
 * bytes; the model fills it only when its own framebuffer is exactly
 * `width` x `height`.  `pixels_written` says whether it did, which is how a
 * caller tells "nothing was pending" from "the model drew something".
 */
struct pvrgpu_systemc_readback_info {
   uint32_t version;
   uint32_t width;
   uint32_t height;
   /*
    * The stored width of one pixel.  Four for a UNORM8 attachment; an integer
    * attachment stores one 32-bit channel per dword, so RG32UI is eight and
    * RGBA32UI sixteen.  The model refuses a readback whose pixel width is not
    * the one it rendered rather than reinterpreting the bytes.
    */
   uint32_t bytes_per_pixel;
   /*
    * Which colour attachment to read.  A fragment shader returning more than
    * one result writes one attachment per result, and the caller reads each
    * in turn; the flush runs once and the attachments it produced stay
    * readable until the next submission replaces them, so asking for a second
    * attachment does not need a second flush and does not lose the first.
    */
   uint32_t attachment;
   uint8_t *pixels;
   size_t pixels_size;
   uint32_t pixels_written;
   /* Zero means one. Multisample output retains every sample, not a resolve. */
   uint32_t sample_count;
   /* UINT32_MAX attachment selects depth/stencil and requires this exact
    * native format, so equal-byte-width layouts cannot be confused. */
   uint32_t depth_format;
   /* Zero means one; must match the complete rendered attachment extent. */
   uint32_t layer_count;
};

/*
 * Run everything submitted since the last flush and hand back the pixels.
 *
 * The model is elaborated once and stays alive between flushes, so this may be
 * called as often as the application reads back.  A flush with nothing pending
 * succeeds with `pixels_written` zero and leaves `pixels` untouched.  Returns
 * 0 on success and fills `error` otherwise.
 */
typedef int (*pvrgpu_systemc_flush_readback_fn)(
   struct pvrgpu_systemc_readback_info *readback,
   char *error,
   size_t error_size);

/* Flush only the requested submission and return its actual physical stage
 * counters. Repeated reads are idempotent; a different generation is an error,
 * so one context cannot consume another context's last completed draw. */
struct pvrgpu_systemc_graphics_stats {
   uint32_t version;
   uint64_t submission_generation;
   uint64_t physical_submissions;
   uint64_t primitives_generated;
   uint64_t ia_primitives;
   uint64_t gs_primitives;
   uint64_t gs_invocations;
   uint64_t stream_output_primitives_written;
   uint64_t stream_output_primitives_storage_needed;
};

typedef int (*pvrgpu_systemc_flush_graphics_stats_fn)(
   struct pvrgpu_systemc_graphics_stats *stats,
   char *error, size_t error_size);

int pvrgpu_systemc_flush_graphics_stats(
   struct pvrgpu_systemc_graphics_stats *stats,
   char *error, size_t error_size);

/* Generation-qualified readback of one transform-feedback resource. */
struct pvrgpu_systemc_stream_output_readback {
   uint32_t version;
   uint64_t submission_generation;
   uint64_t resource_token;
   uint64_t target_token;
   uint8_t *bytes;
   size_t bytes_size;
   uint32_t data_written;
   uint32_t internal_offset;
};

typedef int (*pvrgpu_systemc_flush_stream_output_fn)(
   struct pvrgpu_systemc_stream_output_readback *readback,
   char *error, size_t error_size);

int pvrgpu_systemc_flush_stream_output(
   struct pvrgpu_systemc_stream_output_readback *readback,
   char *error, size_t error_size);

/*
 * Ask the model whether it can execute a compiled PCO binary.
 *
 * The compiler emits the whole PowerVR instruction set; the model implements
 * the subset its ISS decodes.  A draw whose shader falls outside that subset
 * has to be declined before the driver claims it, because a sequence is
 * submitted only once every draw has been recorded -- by which point there is
 * no other path left to describe the frame.  Returns 0 when the binary is
 * executable, and fills `error` with the first instruction it could not
 * decode otherwise.
 */
typedef int (*pvrgpu_systemc_can_execute_pco_binary_fn)(
   uint32_t stage,
   const uint8_t *binary,
   size_t binary_size,
   char *error,
   size_t error_size);

int
pvrgpu_systemc_can_execute_pco_binary(
   uint32_t stage,
   const uint8_t *binary,
   size_t binary_size,
   char *error,
   size_t error_size);

int
pvrgpu_systemc_submit_driver_command(
   const struct pvrgpu_systemc_submit_info *info,
   char *error,
   size_t error_size);

int
pvrgpu_systemc_flush_readback(
   struct pvrgpu_systemc_readback_info *readback,
   char *error,
   size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* PVRGPU_SYSTEMC_API_H */
