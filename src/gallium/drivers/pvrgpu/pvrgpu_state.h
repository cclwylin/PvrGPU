/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_STATE_H
#define PVRGPU_STATE_H

#include "pipe/p_context.h"
#include "pipe/p_state.h"

#include <stdbool.h>

struct nir_shader;

struct pvrgpu_shader_state {
   mesa_shader_stage stage;
   enum pipe_shader_ir type;
   bool has_nir;
   bool has_tgsi;
   struct nir_shader *nir;
   const struct tgsi_token *tgsi;
   struct pipe_stream_output_info stream_output;
};

struct pvrgpu_blend_state {
   struct pipe_blend_state state;
};

struct pvrgpu_sampler_state {
   struct pipe_sampler_state state;
};

struct pvrgpu_sampler_view {
   struct pipe_sampler_view base;
};

struct pvrgpu_depth_stencil_alpha_state {
   struct pipe_depth_stencil_alpha_state state;
};

struct pvrgpu_rasterizer_state {
   struct pipe_rasterizer_state state;
};

struct pvrgpu_vertex_elements_state {
   unsigned num_elements;
   struct pipe_vertex_element elements[PIPE_MAX_ATTRIBS];
};

void
pvrgpu_init_state_functions(struct pipe_context *pipe);

#endif /* PVRGPU_STATE_H */
