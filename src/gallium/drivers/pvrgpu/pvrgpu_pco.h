/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_PCO_H
#define PVRGPU_PCO_H

#include "util/format/u_formats.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct nir_shader;
struct pvrgpu_pco_compiler;

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
};

struct pvrgpu_pco_owned_binary {
   uint8_t *data;
   size_t size;
   struct pvrgpu_pco_stage_abi abi;
};

struct pvrgpu_pco_graphics_binary {
   struct pvrgpu_pco_owned_binary vertex;
   struct pvrgpu_pco_owned_binary fragment;
   uint32_t position_output_start;
   uint32_t position_output_count;
   uint32_t fragment_position_start;
   uint32_t fragment_position_count;
   uint32_t varying_output_start;
   uint32_t varying_output_count;
   uint32_t fragment_varying_start;
   uint32_t fragment_varying_count;
   uint32_t fragment_texture_descriptor_start;
   uint32_t fragment_texture_descriptor_count;
   uint32_t fragment_texture_descriptor_stride;
};

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
#define PVRGPU_PCO_MAX_VERTEX_ATTRIBUTES 8u

/* Varying slots one generically lowered draw can pass between stages. */
#define PVRGPU_PCO_MAX_VARYINGS 8u

/* Combined image/sampler descriptors one generically lowered draw can bind. */
#define PVRGPU_PCO_MAX_TEXTURES 5u

/* Reports the component width the vertex shader declares for each generic
 * attribute, so the driver can pack them at the width the program reads. */
bool pvrgpu_pco_vertex_attribute_components(const struct nir_shader *vertex_nir,
                                            unsigned attribute_count,
                                            unsigned *components);

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
   unsigned max_lod_u4_6);

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
