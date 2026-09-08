/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_PCO_H
#define PVRGPU_PCO_H

#include "util/format/u_formats.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct nir_shader;
struct pvrgpu_pco_compiler;

/* Lower float builtins whose NIR semantics are wider than native PCO's
 * instruction sequence before public PCO instruction selection. */
bool pvrgpu_lower_float_builtins_nir(struct nir_shader *nir);
bool pvrgpu_lower_uniform_fragment_texture_selects(struct nir_shader *nir);

/* Run the common PCO preprocessing pipeline with the compiler-owned NIR
 * options, including the builtin lowerings above. */
void pvrgpu_pco_preprocess_nir(struct pvrgpu_pco_compiler *compiler,
                               struct nir_shader *nir);

#define PVRGPU_PCO_PUBLIC_TARGET "gx6250"
#define PVRGPU_PCO_TEXTURE_DESCRIPTOR_DWORDS 20U
#define PVRGPU_PCO_REFRACT_TEXTURE_COUNT 3U
#define PVRGPU_PCO_REFRACT_FRAGMENT_SHARED_DWORDS \
   (PVRGPU_PCO_REFRACT_TEXTURE_COUNT * \
    PVRGPU_PCO_TEXTURE_DESCRIPTOR_DWORDS)

/*
 * Stable driver/model ABI distilled from pco_data.  pco_data itself is not a
 * transport format: it contains compiler-private pointers and changes with
 * Mesa.  Counts and offsets below are all DWORD counts/indices after PCO's
 * byte-to-DWORD I/O lowering.
 */
struct pvrgpu_pco_stage_abi {
   uint32_t temps;
   uint32_t vertex_inputs;
   uint32_t vertex_outputs;
   uint32_t coefficients;
   uint32_t shareds;
   uint32_t push_constant_start;
   uint32_t push_constant_count;
   uint32_t entry_offset;
   /* Four DWORDs per block: base low/high, byte size, dynamic byte offset. */
   uint32_t uniform_buffer_descriptor_start;
   uint32_t uniform_buffer_descriptor_count;
};

struct pvrgpu_pco_owned_binary {
   uint8_t *data;
   size_t size;
   struct pvrgpu_pco_stage_abi abi;
};

/* Compute has its own transport contract; it is never a graphics stage.
 * LOCAL_INVOCATION_INDEX uses VTXIN. WORKGROUP_ID and NUM_WORKGROUPS use
 * COEFF, as in Mesa's public PDS compute ABI. All other IDs are shader ALU.
 * Shared registers contain UBO descriptors, SSBO descriptors, image descriptors,
 * optional private workgroup-memory descriptor, then CB0. Buffer descriptors
 * are baseLo/baseHi/byteSize/dynamicByteOffset; images append width/height/
 * rowStride/format to baseLo/baseHi/byteSize/zero. Zero images preserve old SH. */
struct pvrgpu_pco_compute_abi {
   struct pvrgpu_pco_stage_abi stage;
   uint32_t local_size[3];
   uint32_t local_invocation_index_start;
   uint32_t local_invocation_index_count;
   uint32_t workgroup_id_start;
   uint32_t workgroup_id_count;
   uint32_t num_workgroups_start;
   uint32_t num_workgroups_count;
   uint32_t storage_buffer_descriptor_start;
   uint32_t storage_buffer_descriptor_count;
   uint32_t uniform_buffer_used_mask;
   uint32_t storage_buffer_used_mask;
   uint32_t storage_buffer_read_mask;
   uint32_t storage_buffer_write_mask;
   uint32_t shared_memory_bytes;
   uint32_t scratch_bytes;
   uint32_t shared_memory_descriptor_start;
   uint32_t shared_memory_descriptor_count;
   uint32_t image_descriptor_start;
   uint32_t image_descriptor_count;
   uint32_t image_used_mask;
   uint32_t image_read_mask;
   uint32_t image_write_mask;
};

struct pvrgpu_pco_compute_binary {
   uint8_t *data;
   size_t size;
   struct pvrgpu_pco_compute_abi abi;
};

/* Input NIR is cloned. A successful caller owns data until finish(). The
 * initial native subset accepts static workgroups, IDs, CB0 and 32-bit
 * UBO/SSBO accesses with static bindings and runtime byte offsets. Unsupported
 * images and scratch fail before PCO lowering. Shared storage is bounded to
 * 32 KiB, and native MUTEX sleep/wakeup implements multi-task barriers. */
bool pvrgpu_pco_compile_compute(
   struct pvrgpu_pco_compiler *compiler,
   const struct nir_shader *compute_nir,
   unsigned uniform_dwords,
   struct pvrgpu_pco_compute_binary *out,
   char *error,
   size_t error_size);

void pvrgpu_pco_compute_binary_finish(struct pvrgpu_pco_compute_binary *binary);

struct pvrgpu_pco_varying_binding {
   uint32_t output_dword;
   uint32_t num_components;
   uint32_t coefficient_dword;
   uint32_t flat;
};

struct pvrgpu_pco_graphics_binary {
   struct pvrgpu_pco_owned_binary vertex;
   struct pvrgpu_pco_owned_binary fragment;
   /* Actual final pre-raster output placement (VS or TES), keyed by the
    * original NIR varying location of that stage.
    * Gallium stream-output register indices are ordinals in outputs_written,
    * not physical VTXOUT indices. */
   uint32_t vertex_output_start[64];
   uint32_t vertex_output_count[64];
   struct pvrgpu_pco_varying_binding varying_bindings[16];
   uint32_t varying_binding_count;
   bool explicit_varying_bindings;
   /* Vertex output holding gl_PointSize, when the shader writes one. */
   uint32_t point_size_output_start;
   uint32_t point_size_output_count;
   uint32_t position_output_start;
   uint32_t position_output_count;
   uint32_t fragment_position_start;
   uint32_t fragment_position_count;
   uint32_t varying_output_start;
   uint32_t varying_output_count;
   uint32_t fragment_varying_start;
   uint32_t fragment_varying_count;
   /*
    * Bit N set: varying slot N is flat-qualified, so its coefficient set is
    * the provoking vertex's value rather than an interpolation plane.  PCO
    * already knows -- it emits MBYP from the coefficient instead of FITRP --
    * and the model has to be told the same thing or it interpolates.
    */
   uint32_t varying_flat_mask;
   /* PIXOUT lanes each colour attachment expects the shader to write. */
   uint32_t fragment_output_mask[8];
   uint32_t fragment_texture_descriptor_start;
   uint32_t fragment_texture_descriptor_count;
   uint32_t fragment_texture_descriptor_stride;
};

/* Independent Geometry stage. These are compiler/driver contracts, not a
 * claim that Geometry shares the Compute data master or shader module.
 * Every location maps to a contiguous range of raw 32-bit components.
 * The input layout is the preceding VS's actual UVSW output layout; its
 * per-primitive AoS bytes are read by native LD instructions. */
#define PVRGPU_PCO_GEOMETRY_LOCATIONS 64u
#define PVRGPU_PCO_GEOMETRY_MAX_DWORDS 64u
#define PVRGPU_PCO_GEOMETRY_INPUT_DESCRIPTOR_START 0u
/* After the four primitive-input descriptor words, N combined samplers use
 * 20 DWORDs each. This is the zero-texture UBO origin; the actual UBO origin
 * is stage.uniform_buffer_descriptor_start = 4 + 20*N, then CB0 follows. */
#define PVRGPU_PCO_GEOMETRY_UBO_DESCRIPTOR_START 4u
#define PVRGPU_PCO_GEOMETRY_PRIMITIVE_ID_INPUT 0u
#define PVRGPU_PCO_GEOMETRY_INVOCATION_ID_INPUT 1u

struct pvrgpu_pco_geometry_layout {
   uint32_t start[PVRGPU_PCO_GEOMETRY_LOCATIONS];
   uint32_t count[PVRGPU_PCO_GEOMETRY_LOCATIONS];
   uint32_t stride_dwords;
};

struct pvrgpu_pco_geometry_abi {
   struct pvrgpu_pco_stage_abi stage;
   struct pvrgpu_pco_geometry_layout input;
   struct pvrgpu_pco_geometry_layout output;
   /* Mesa primitive enums, not pipe primitive enums. */
   uint32_t input_primitive;
   uint32_t output_primitive;
   uint32_t vertices_in;
   uint32_t vertices_out;
   uint32_t invocations;
   uint32_t primitive_id_input;
   uint32_t invocation_id_input;
};

struct pvrgpu_pco_geometry_binary {
   struct pvrgpu_pco_owned_binary shader;
   struct pvrgpu_pco_geometry_abi abi;
};

/* graphics.vertex is the original VS and writes geometry.abi.input;
 * graphics.fragment consumes geometry.abi.output. All rasterizer-facing
 * output ranges in graphics describe GS output, not VS output. */
struct pvrgpu_pco_geometry_pipeline_binary {
   struct pvrgpu_pco_graphics_binary graphics;
   struct pvrgpu_pco_geometry_binary geometry;
};

bool pvrgpu_pco_compile_geometry(
   struct pvrgpu_pco_compiler *compiler,
   const struct nir_shader *geometry_nir,
   const struct pvrgpu_pco_geometry_layout *input_layout,
   const struct pvrgpu_pco_geometry_layout *output_layout,
   unsigned uniform_dwords,
   struct pvrgpu_pco_geometry_binary *out,
   char *error, size_t error_size);

bool pvrgpu_pco_compile_geometry_pipeline(
   struct pvrgpu_pco_compiler *compiler,
   const struct nir_shader *vertex_nir,
   const struct nir_shader *geometry_nir,
   const struct nir_shader *fragment_nir,
   const enum pipe_format *attribute_formats,
   unsigned render_target_count,
   unsigned vertex_uniform_dwords,
   unsigned geometry_uniform_dwords,
   unsigned fragment_uniform_dwords,
   unsigned attribute_count,
   unsigned fragment_texture_count,
   struct pvrgpu_pco_geometry_pipeline_binary *out,
   char *error, size_t error_size);

void pvrgpu_pco_geometry_binary_finish(struct pvrgpu_pco_geometry_binary *binary);
void pvrgpu_pco_geometry_pipeline_binary_finish(
   struct pvrgpu_pco_geometry_pipeline_binary *binary);

/* Native TCS/TES ABI. The shader stages are preserved through PCO. TCS
 * accesses patch storage with native LD/ST; TES reads the same storage and
 * exports its evaluated vertex with UVSW. All quantities below are DWORDs.
 * Patch storage starts with outer[4], inner[2], then patch varyings and AoS
 * per-vertex outputs. Undefined/unwritten values are not fabricated exports.
 * TCS runs <=32 invocations in one instruction-group-lockstep task. */
#define PVRGPU_PCO_TESS_MAX_VERTICES 32u
#define PVRGPU_PCO_TESS_PATCH_LOCATIONS 32u
#define PVRGPU_PCO_TESS_MAX_PATCH_DWORDS 128u
struct pvrgpu_pco_tessellation_layout {
   struct pvrgpu_pco_geometry_layout vertex;
   uint32_t patch_start[PVRGPU_PCO_TESS_PATCH_LOCATIONS];
   uint32_t patch_count[PVRGPU_PCO_TESS_PATCH_LOCATIONS];
   uint32_t per_vertex_offset_dwords;
   uint32_t patch_stride_dwords;
};
struct pvrgpu_pco_tessellation_binary {
   struct pvrgpu_pco_owned_binary shader;
   uint32_t barrier_count;
};
struct pvrgpu_pco_tessellation_pipeline_binary {
   struct pvrgpu_pco_graphics_binary graphics;
   struct pvrgpu_pco_tessellation_binary control;
   struct pvrgpu_pco_tessellation_binary evaluation;
   struct pvrgpu_pco_geometry_layout input;
   struct pvrgpu_pco_tessellation_layout patch;
   struct pvrgpu_pco_geometry_layout output;
   uint32_t output_vertices;
   uint32_t primitive_mode; /* Mesa tess_primitive_mode. */
   uint32_t spacing;        /* Mesa gl_tess_spacing. */
   uint32_t ccw;
   uint32_t point_mode;
};
bool pvrgpu_pco_compile_tessellation_pipeline(
   struct pvrgpu_pco_compiler *compiler,
   const struct nir_shader *vertex_nir,
   const struct nir_shader *control_nir,
   const struct nir_shader *evaluation_nir,
   const struct nir_shader *fragment_nir,
   const enum pipe_format *attribute_formats,
   unsigned render_target_count,
   unsigned vertex_uniform_dwords,
   unsigned control_uniform_dwords,
   unsigned evaluation_uniform_dwords,
   unsigned fragment_uniform_dwords,
   unsigned attribute_count,
   unsigned fragment_texture_count,
   struct pvrgpu_pco_tessellation_pipeline_binary *out,
   char *error, size_t error_size);
void pvrgpu_pco_tessellation_pipeline_binary_finish(
   struct pvrgpu_pco_tessellation_pipeline_binary *binary);

enum pvrgpu_pco_lit_mesh_profile {
   PVRGPU_PCO_LIT_MESH_BUILD,
   PVRGPU_PCO_LIT_MESH_BUMP,
   PVRGPU_PCO_LIT_MESH_SHADING,
};

enum pvrgpu_pco_ideas_profile {
   PVRGPU_PCO_IDEAS_LOGO,
   PVRGPU_PCO_IDEAS_LIGHTING,
   PVRGPU_PCO_IDEAS_WHITE,
   PVRGPU_PCO_IDEAS_BLACK,
};

enum pvrgpu_pco_refract_profile {
   PVRGPU_PCO_REFRACT_PREPASS,
   PVRGPU_PCO_REFRACT_COMPOSITE,
};

enum pvrgpu_pco_shadow_profile {
   PVRGPU_PCO_SHADOW_DEPTH,
   PVRGPU_PCO_SHADOW_MASK,
   PVRGPU_PCO_SHADOW_SCENE,
};

enum pvrgpu_pco_terrain_profile {
   PVRGPU_PCO_TERRAIN_D1,
   PVRGPU_PCO_TERRAIN_D2,
   PVRGPU_PCO_TERRAIN_D3,
   PVRGPU_PCO_TERRAIN_D4,
   PVRGPU_PCO_TERRAIN_D5,
   PVRGPU_PCO_TERRAIN_D6,
   PVRGPU_PCO_TERRAIN_D7,
   PVRGPU_PCO_TERRAIN_D8,
};

/* One compiler owns one public gx6250 PCO context. */
struct pvrgpu_pco_compiler *pvrgpu_pco_compiler_create(char *error,
                                                       size_t error_size);

void pvrgpu_pco_compiler_destroy(struct pvrgpu_pco_compiler *compiler);

/*
 * Compile the strict GLBench conditionals profile.  The input shaders are
 * never modified: this function clones both NIR shaders before applying the
 * destructive PCO pipeline.  The first implementation accepts exactly one
 * GENERIC0 R32G32B32_FLOAT vertex attribute and an RGBA8 render target.
 *
 * On success, out owns both byte arrays and must be released with
 * pvrgpu_pco_graphics_binary_finish().  On failure, out is left empty and the
 * diagnostic explains the first fail-closed gate.
 */
bool pvrgpu_pco_compile_conditionals(struct pvrgpu_pco_compiler *compiler,
                                     const struct nir_shader *vertex_nir,
                                     const struct nir_shader *fragment_nir,
                                     enum pipe_format vertex_format,
                                     struct pvrgpu_pco_graphics_binary *out,
                                     char *error,
                                     size_t error_size);

/*
 * Compile a basic color triangle or color mesh profile (Position + Color varying).
 * Position attribute is GENERIC0, Color attribute is GENERIC1.
 * Exports position and smooth color varying (4 scalar components).
 */
/* Vertex attributes one generically lowered draw can bind. */
#define PVRGPU_PCO_MAX_VERTEX_ATTRIBUTES 16u

/* Varying slots one generically lowered draw can pass between stages. */
#define PVRGPU_PCO_MAX_VARYINGS 16u

/* Combined image/sampler descriptors one generically lowered draw can bind. */
#define PVRGPU_PCO_MAX_TEXTURES 8u

/*
 * Reports, for each bound vertex element, the component width the vertex
 * shader declares for the attribute it feeds and the generic location that
 * attribute occupies.  The driver packs each attribute at the width the
 * program reads, and the PCO vertex-input data has to be keyed by location
 * because a shader that reads a sparse set of locations still gets one dense
 * vertex element per location it reads.
 *
 * `reason` names the condition that rejected the layout so a declined draw
 * says which one it was rather than reporting a bare boolean.
 */
bool pvrgpu_pco_vertex_attribute_components(const struct nir_shader *vertex_nir,
                                            unsigned attribute_count,
                                            unsigned *components,
                                            unsigned *locations,
                                            const char **reason);

bool pvrgpu_pco_compile_color_triangle(
   struct pvrgpu_pco_compiler *compiler,
   const struct nir_shader *vertex_nir,
   const struct nir_shader *fragment_nir,
   const enum pipe_format *attribute_formats,
   bool topology_uses_point_size,
   unsigned render_target_count,
   unsigned vertex_uniform_dwords,
   unsigned fragment_uniform_dwords,
   unsigned attribute_count,
   unsigned texture_count,
   struct pvrgpu_pco_graphics_binary *out,
   char *error,
   size_t error_size);

/*
 * Compile one of the fail-closed GLMark2 lit-mesh profiles.  These profiles
 * share two R32G32B32_FLOAT attributes (position and normal), a 32-DWORD VS
 * constant-buffer ABI, and one smooth scalar/vec3 varying.  The profile enum
 * selects the exact NIR signature and precision contract: build and shading
 * legally retain fp32, while bump lowers its default-mediump fragment graph at
 * every binary16 operation boundary.  Arbitrary shaders are rejected.
 */
bool pvrgpu_pco_compile_lit_mesh(
   struct pvrgpu_pco_compiler *compiler,
   const struct nir_shader *vertex_nir,
   const struct nir_shader *fragment_nir,
   enum pvrgpu_pco_lit_mesh_profile profile,
   struct pvrgpu_pco_graphics_binary *out,
   char *error,
   size_t error_size);

/*
 * Compile the single-draw GLMark2 texture profile.  The accepted VS has
 * position/normal/UV attributes, one 32-DWORD CB0, and a smooth
 * (intensity, u, v) varying retained in fp32.  The accepted FS performs one
 * implicit
 * 2D sample from texture/sampler slot zero and modulates RGB by intensity.
 *
 * The returned fragment descriptor range is the gx6250 combined image /
 * sampler layout consumed by the real PCO SMP instruction.  It contains the
 * image state and metadata, normal sampler state and metadata, and gather
 * sampler state, in that order.
 */
bool pvrgpu_pco_compile_texture(
   struct pvrgpu_pco_compiler *compiler,
   const struct nir_shader *vertex_nir,
   const struct nir_shader *fragment_nir,
   struct pvrgpu_pco_graphics_binary *out,
   char *error,
   size_t error_size);

/* Compile one draw of the two-pass GLMark2 refract pipeline.  PREPASS writes
 * the mesh normal/depth target using one 16-DWORD matrix.  COMPOSITE consumes
 * four 4x4 matrices, exports eleven smooth scalar components, and addresses
 * three independent combined image/sampler descriptors.  Both profiles are
 * source-hash and NIR-signature locked; arbitrary shaders fail closed. */
bool pvrgpu_pco_compile_refract(
   struct pvrgpu_pco_compiler *compiler,
   const struct nir_shader *vertex_nir,
   const struct nir_shader *fragment_nir,
   enum pvrgpu_pco_refract_profile profile,
   struct pvrgpu_pco_graphics_binary *out,
   char *error,
   size_t error_size);

/* Build the canonical address-zero public Rogue descriptor block consumed by
 * the strict refract composite FS.  The bridge owns relocation: it validates
 * these non-address fields, allocates the three resources in unified DRAM,
 * then patches IMAGE_WORD1 address bits in its private copy. */
void pvrgpu_pco_build_refract_fragment_shared(
   uint32_t out[PVRGPU_PCO_REFRACT_FRAGMENT_SHARED_DWORDS]);

bool pvrgpu_pco_build_refract_fragment_shared_for_extent(
   uint32_t out[PVRGPU_PCO_REFRACT_FRAGMENT_SHARED_DWORDS],
   unsigned width,
   unsigned height);

/* Compile one draw of the strict three-draw GLMark2 shadow pipeline.  DEPTH
 * writes the native 2x-output Z32 shadow attachment, MASK samples that
 * attachment while drawing the four-vertex screen strip, and SCENE shades
 * the 21,516-vertex mesh into the final target.  Source hashes, NIR graphs,
 * precision, uniform slots and linkage are all fail-closed. */
bool pvrgpu_pco_compile_shadow(
   struct pvrgpu_pco_compiler *compiler,
   const struct nir_shader *vertex_nir,
   const struct nir_shader *fragment_nir,
   enum pvrgpu_pco_shadow_profile profile,
   struct pvrgpu_pco_graphics_binary *out,
   char *error,
   size_t error_size);

/* Canonical address-zero Rogue combined image/sampler descriptor for the
 * shadow MASK pass.  The native sequence submitter owns resource allocation
 * and relocation after validating every non-address field. */
void pvrgpu_pco_build_shadow_fragment_shared(
   uint32_t out[PVRGPU_PCO_TEXTURE_DESCRIPTOR_DWORDS]);

bool pvrgpu_pco_build_shadow_fragment_shared_for_extent(
   uint32_t out[PVRGPU_PCO_TEXTURE_DESCRIPTOR_DWORDS],
   unsigned width,
   unsigned height);

/* Compile one shader pair from the strict eight-profile GLMark2 terrain
 * sequence.  Texture indices are validated in their captured GL order, then
 * packed on private NIR clones as set 0 bindings so all five MAIN resources
 * fit the public PCO descriptor ABI.  Descriptor blocks precede Gallium CB0
 * push constants in shared registers. */
bool pvrgpu_pco_compile_terrain(
   struct pvrgpu_pco_compiler *compiler,
   const struct nir_shader *vertex_nir,
   const struct nir_shader *fragment_nir,
   enum pvrgpu_pco_terrain_profile profile,
   struct pvrgpu_pco_graphics_binary *out,
   char *error,
   size_t error_size);

/* Build one canonical address-zero descriptor used by the strict Terrain
 * profile.  RGBA8 preserves alpha; RGBX8 forces alpha to one.  The caller
 * supplies already validated Rogue filter/address mode fields (0 or 1 for
 * filters, 0 or 2 for repeat/clamp) and owns relocation after capture. */
bool pvrgpu_pco_build_terrain_texture_descriptor(
   uint32_t out[PVRGPU_PCO_TEXTURE_DESCRIPTOR_DWORDS],
   enum pipe_format format,
   unsigned width,
   unsigned height,
   unsigned mip_count,
   uint32_t byte_size,
   unsigned min_filter,
   unsigned mag_filter,
   unsigned mip_filter,
   unsigned wrap_u,
   unsigned wrap_v,
   unsigned max_lod_u4_6,
   unsigned layers,
   unsigned wrap_w);

/* Rogue IMAGE_WORD0 SMPCNT is log2(samples), exactly two bits. The image
 * extent and stride remain logical texels; storage is pixel-interleaved. */
bool pvrgpu_pco_set_texture_sample_count(
   uint32_t out[PVRGPU_PCO_TEXTURE_DESCRIPTOR_DWORDS], unsigned sample_count);

/* Compile one of the four shader pairs used by the 180-draw GLMark2 ideas
 * capture.  The simple profiles consume one float4 attribute and 32 VS
 * shared DWORDs.  LIGHTING consumes two float4 attributes, 44 VS shared
 * DWORDs, 12 FS shared DWORDs, and exports ten smooth scalar components.
 * Source signatures, NIR I/O, uniform ABI, and constant white/black payloads
 * are all fail-closed before destructive PCO lowering. */
bool pvrgpu_pco_compile_ideas(
   struct pvrgpu_pco_compiler *compiler,
   const struct nir_shader *vertex_nir,
   const struct nir_shader *fragment_nir,
   enum pvrgpu_pco_ideas_profile profile,
   struct pvrgpu_pco_graphics_binary *out,
   char *error,
   size_t error_size);

void pvrgpu_pco_graphics_binary_finish(
   struct pvrgpu_pco_graphics_binary *binary);

#endif /* PVRGPU_PCO_H */
