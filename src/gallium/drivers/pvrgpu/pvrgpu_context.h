/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_CONTEXT_H
#define PVRGPU_CONTEXT_H

#include "pvrgpu_state.h"

#include "pipe/p_context.h"
#include "pipe/p_state.h"

#include <stdbool.h>
#include <stdint.h>

/*
 * Upper bound on the draws one generic array-primitive sequence carries.  The
 * public SystemC sequence header enforces its own limit; this keeps the
 * driver-side accumulation bounded independently of it.
 */
#define PVRGPU_ARRAY_PRIMITIVE_SEQUENCE_MAX 64u

/* Colour attachments one lowered draw can write. */
#define PVRGPU_MAX_RENDER_TARGETS 4u

/* Shared-register words one lowered draw carries per stage. */
#define PVRGPU_COLOR_PRIMITIVE_UNIFORM_DWORDS 64u

struct pvrgpu_array_primitive_draw;
struct pvrgpu_deqp_primitive_sequence_profile;
struct pvrgpu_pco_compiler;
struct pvrgpu_pco_graphics_binary;
struct pvrgpu_refract_pco_observation;
struct pvrgpu_shadow_pco_observation;
struct pvrgpu_terrain_pco_observation;

struct pvrgpu_context {
   struct pipe_context base;
   struct pipe_framebuffer_state framebuffer;
   struct pvrgpu_blend_state *blend;
   struct pvrgpu_depth_stencil_alpha_state *dsa;
   struct pvrgpu_rasterizer_state *rasterizer;
   struct pvrgpu_sampler_state *samplers[MESA_SHADER_MESH_STAGES]
                                      [PIPE_MAX_SAMPLERS];
   unsigned num_samplers[MESA_SHADER_MESH_STAGES];
   struct pipe_sampler_view *sampler_views[MESA_SHADER_MESH_STAGES]
                                            [PIPE_MAX_SHADER_SAMPLER_VIEWS];
   unsigned num_sampler_views[MESA_SHADER_MESH_STAGES];
   struct pipe_constant_buffer constant_buffers[MESA_SHADER_MESH_STAGES]
                                                [PIPE_MAX_CONSTANT_BUFFERS];
   unsigned num_constant_buffers[MESA_SHADER_MESH_STAGES];
   struct pvrgpu_shader_state *vs;
   struct pvrgpu_shader_state *fs;
   struct pvrgpu_shader_state *gs;
   struct pvrgpu_shader_state *tcs;
   struct pvrgpu_shader_state *tes;
   struct pvrgpu_vertex_elements_state *vertex_elements;
   struct pipe_vertex_buffer vertex_buffers[PIPE_MAX_ATTRIBS];
   unsigned num_vertex_buffers;
   struct pipe_stream_output_target *stream_output_targets[PIPE_MAX_SO_BUFFERS];
   unsigned num_stream_output_targets;
   enum mesa_prim stream_output_prim;
   struct pipe_blend_color blend_color;
   struct pipe_stencil_ref stencil_ref;
   unsigned sample_mask;
   struct pipe_viewport_state viewport;
   bool has_viewport;
   struct pipe_scissor_state scissor;
   bool has_scissor;
   uint8_t patch_vertices;
   float tess_default_outer_level[4];
   float tess_default_inner_level[2];
   unsigned max_framebuffer_width;
   unsigned max_framebuffer_height;
   unsigned framebuffer_updates;
   unsigned flushes;
   unsigned unsupported_draws;
   unsigned observed_draws;
   unsigned unsupported_resource_ops;
   unsigned indexed_quad_draws;
   struct pipe_resource *full_depth_clear_resource;
   unsigned full_depth_clear_level;
   unsigned full_depth_clear_first_layer;
   unsigned full_depth_clear_last_layer;
   unsigned full_depth_clear_width;
   unsigned full_depth_clear_height;
   bool full_depth_clear_is_one;
   bool driver_draw_command_emitted;
   bool driver_indexed_quad_command_locked;
   bool driver_counter_sequence_command_emitted;
   const struct pvrgpu_deqp_primitive_sequence_profile
      *pending_primitive_sequence_profile;
   struct pvrgpu_pco_compiler *pco_compiler;
   unsigned ideas_pco_probe_draws;
   unsigned ideas_pco_draws;
   struct pvrgpu_pco_graphics_binary *ideas_pco_binaries[4];
   struct pvrgpu_refract_pco_observation *refract_pco_prepass;
   struct pvrgpu_refract_pco_observation *refract_pco_composite;
   struct pvrgpu_shadow_pco_observation *shadow_pco_depth;
   struct pvrgpu_shadow_pco_observation *shadow_pco_mask;
   struct pvrgpu_shadow_pco_observation *shadow_pco_scene;
   bool shadow_pco_warmup_skipped;
   struct pvrgpu_array_primitive_draw
      *array_primitive_draws[PVRGPU_ARRAY_PRIMITIVE_SEQUENCE_MAX];
   unsigned array_primitive_draw_count;
   bool array_primitive_sequence_overflow;
   bool array_primitive_sequence_owns_command;
   unsigned terrain_pco_probe_draws;
   unsigned terrain_pco_draw_count;
   bool terrain_pco_warmup_skipped;
   struct pvrgpu_terrain_pco_observation *terrain_pco_draws[8];
};

static inline struct pvrgpu_context *
pvrgpu_context(struct pipe_context *pipe)
{
   return (struct pvrgpu_context *)pipe;
}

struct pipe_context *
pvrgpu_create_context(struct pipe_screen *screen, void *priv,
                      unsigned flags);

bool
pvrgpu_case_prefers_draw_counter_sequence(void);

bool
pvrgpu_case_counter_sequence_allows_clear_emit(void);

bool
pvrgpu_emit_case_counter_sequence_command(struct pvrgpu_context *ctx);

bool
pvrgpu_emit_array_primitive_sequence_command(struct pvrgpu_context *ctx);

void
pvrgpu_array_primitive_sequence_reset(struct pvrgpu_context *ctx);

void
pvrgpu_note_full_depth_clear_one(struct pvrgpu_context *ctx,
                                 const struct pipe_surface *surface,
                                 unsigned width,
                                 unsigned height);

void
pvrgpu_invalidate_full_depth_clear(struct pvrgpu_context *ctx);

void
pvrgpu_invalidate_full_depth_clear_for_resource(
   struct pvrgpu_context *ctx,
   const struct pipe_resource *resource);

void
pvrgpu_clear(struct pipe_context *pipe,
             unsigned buffers,
             uint32_t color_clear_mask,
             uint8_t stencil_clear_mask,
             const struct pipe_scissor_state *scissor_state,
             const union pipe_color_union *color,
             double depth,
             unsigned stencil);

void
pvrgpu_clear_render_target(struct pipe_context *pipe,
                           struct pipe_surface *dst,
                           const union pipe_color_union *color,
                           unsigned dstx,
                           unsigned dsty,
                           unsigned width,
                           unsigned height,
                           bool render_condition_enabled);

void
pvrgpu_clear_depth_stencil(struct pipe_context *pipe,
                           struct pipe_surface *dst,
                           unsigned clear_flags,
                           double depth,
                           unsigned stencil,
                           unsigned dstx,
                           unsigned dsty,
                           unsigned width,
                           unsigned height,
                           bool render_condition_enabled);

#endif /* PVRGPU_CONTEXT_H */
