/* SPDX-License-Identifier: MIT */

#include "pvrgpu_state.h"

#include "pvrgpu_context.h"
#include "pvrgpu_counter.h"
#include "pvrgpu_resource.h"

#include "pipe/p_state.h"
#include "util/format/u_format.h"
#include "util/u_debug_cb.h"
#include "util/u_helpers.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"
#include "util/ralloc.h"

#include <string.h>

struct pvrgpu_state_object {
   unsigned placeholder;
};

static void *
pvrgpu_create_state_object(struct pipe_context *pipe, const void *state)
{
   (void)pipe;
   (void)state;
   return CALLOC_STRUCT(pvrgpu_state_object);
}

static void
pvrgpu_bind_state_object(struct pipe_context *pipe, void *state)
{
   (void)pipe;
   (void)state;
}

static void
pvrgpu_delete_state_object(struct pipe_context *pipe, void *state)
{
   (void)pipe;
   FREE(state);
}

static const char *
pvrgpu_shader_stage_name(mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:
      return "vertex";
   case MESA_SHADER_FRAGMENT:
      return "fragment";
   case MESA_SHADER_COMPUTE:
      return "compute";
   case MESA_SHADER_TASK:
      return "task";
   case MESA_SHADER_MESH:
      return "mesh";
   case MESA_SHADER_GEOMETRY:
      return "geometry";
   case MESA_SHADER_TESS_CTRL:
      return "tess_ctrl";
   case MESA_SHADER_TESS_EVAL:
      return "tess_eval";
   default:
      return "unknown";
   }
}

static bool
pvrgpu_is_valid_context_stage(mesa_shader_stage shader)
{
   return shader >= MESA_SHADER_VERTEX && shader < MESA_SHADER_MESH_STAGES;
}

static unsigned
pvrgpu_count_bound_samplers(const struct pvrgpu_context *ctx,
                            mesa_shader_stage shader)
{
   if (!pvrgpu_is_valid_context_stage(shader))
      return 0;

   unsigned count = PIPE_MAX_SAMPLERS;
   while (count > 0 && !ctx->samplers[shader][count - 1])
      count--;
   return count;
}

static unsigned
pvrgpu_count_bound_sampler_views(const struct pvrgpu_context *ctx,
                                 mesa_shader_stage shader)
{
   if (!pvrgpu_is_valid_context_stage(shader))
      return 0;

   unsigned count = PIPE_MAX_SHADER_SAMPLER_VIEWS;
   while (count > 0 && !ctx->sampler_views[shader][count - 1])
      count--;
   return count;
}

static bool
pvrgpu_constant_buffer_is_bound(const struct pipe_constant_buffer *buffer)
{
   return buffer &&
          buffer->buffer_size != 0 &&
          (buffer->buffer || buffer->user_buffer);
}

static unsigned
pvrgpu_count_bound_constant_buffers(const struct pvrgpu_context *ctx,
                                    mesa_shader_stage shader)
{
   if (!pvrgpu_is_valid_context_stage(shader))
      return 0;

   unsigned count = PIPE_MAX_CONSTANT_BUFFERS;
   while (count > 0 &&
          !pvrgpu_constant_buffer_is_bound(
             &ctx->constant_buffers[shader][count - 1]))
      count--;
   return count;
}

static bool
pvrgpu_constant_buffer_first_words(const struct pipe_constant_buffer *buffer,
                                   uint32_t words[4])
{
   memset(words, 0, 4 * sizeof(words[0]));
   if (!pvrgpu_constant_buffer_is_bound(buffer))
      return false;

   const uint8_t *data = NULL;
   size_t available = buffer->buffer_size;
   if (buffer->user_buffer) {
      data = (const uint8_t *)buffer->user_buffer;
   } else if (buffer->buffer) {
      struct pvrgpu_resource *resource = pvrgpu_resource(buffer->buffer);
      if (!resource || !resource->data ||
          buffer->buffer_offset >= resource->size)
         return false;
      data = resource->data + buffer->buffer_offset;
      const size_t resource_available = resource->size - buffer->buffer_offset;
      if (available > resource_available)
         available = resource_available;
   }

   if (!data || available < sizeof(uint32_t))
      return false;

   const size_t copy_bytes =
      available < 4 * sizeof(uint32_t) ? available : 4 * sizeof(uint32_t);
   memcpy(words, data, copy_bytes);
   return true;
}

static void *
pvrgpu_create_blend_state(struct pipe_context *pipe,
                          const struct pipe_blend_state *state)
{
   (void)pipe;
   struct pvrgpu_blend_state *blend = CALLOC_STRUCT(pvrgpu_blend_state);
   if (!blend)
      return NULL;
   if (state)
      blend->state = *state;

   pvrgpu_counter_eventf("create_blend_state",
                         "rt0_enable=%u rt0_colormask=0x%x "
                         "rt0_rgb_func=%u rt0_rgb_src=%u rt0_rgb_dst=%u "
                         "rt0_alpha_func=%u logicop=%u dither=%u",
                         blend->state.rt[0].blend_enable,
                         blend->state.rt[0].colormask,
                         blend->state.rt[0].rgb_func,
                         blend->state.rt[0].rgb_src_factor,
                         blend->state.rt[0].rgb_dst_factor,
                         blend->state.rt[0].alpha_func,
                         blend->state.logicop_enable,
                         blend->state.dither);
   return blend;
}

static void *
pvrgpu_create_sampler_state(struct pipe_context *pipe,
                            const struct pipe_sampler_state *state)
{
   (void)pipe;
   struct pvrgpu_sampler_state *sampler = CALLOC_STRUCT(pvrgpu_sampler_state);
   if (!sampler)
      return NULL;
   if (state)
      sampler->state = *state;

   pvrgpu_counter_eventf("create_sampler_state",
                         "wrap=%u,%u,%u min_img_filter=%u "
                         "min_mip_filter=%u mag_img_filter=%u "
                         "unnormalized=%u min_lod=%f max_lod=%f",
                         sampler->state.wrap_s,
                         sampler->state.wrap_t,
                         sampler->state.wrap_r,
                         sampler->state.min_img_filter,
                         sampler->state.min_mip_filter,
                         sampler->state.mag_img_filter,
                         sampler->state.unnormalized_coords,
                         sampler->state.min_lod,
                         sampler->state.max_lod);
   return sampler;
}

static void *
pvrgpu_create_rasterizer_state(struct pipe_context *pipe,
                               const struct pipe_rasterizer_state *state)
{
   (void)pipe;
   struct pvrgpu_rasterizer_state *rast =
      CALLOC_STRUCT(pvrgpu_rasterizer_state);
   if (!rast)
      return NULL;
   if (state)
      rast->state = *state;

   pvrgpu_counter_eventf("create_rasterizer_state",
                         "front_ccw=%u cull_face=%u scissor=%u "
                         "fill_front=%u fill_back=%u discard=%u "
                         "multisample=%u line_width=%f",
                         rast->state.front_ccw,
                         rast->state.cull_face,
                         rast->state.scissor,
                         rast->state.fill_front,
                         rast->state.fill_back,
                         rast->state.rasterizer_discard,
                         rast->state.multisample,
                         rast->state.line_width);
   return rast;
}

static void *
pvrgpu_create_depth_stencil_alpha_state(
   struct pipe_context *pipe,
   const struct pipe_depth_stencil_alpha_state *state)
{
   (void)pipe;
   struct pvrgpu_depth_stencil_alpha_state *dsa =
      CALLOC_STRUCT(pvrgpu_depth_stencil_alpha_state);
   if (!dsa)
      return NULL;
   if (state)
      dsa->state = *state;

   pvrgpu_counter_eventf("create_depth_stencil_alpha_state",
                         "depth_enable=%u depth_write=%u depth_func=%u "
                         "stencil0_enable=%u stencil0_func=%u "
                         "stencil0_writemask=0x%x alpha_enable=%u",
                         dsa->state.depth_enabled,
                         dsa->state.depth_writemask,
                         dsa->state.depth_func,
                         dsa->state.stencil[0].enabled,
                         dsa->state.stencil[0].func,
                         dsa->state.stencil[0].writemask,
                         dsa->state.alpha_enabled);
   return dsa;
}

static void
pvrgpu_bind_blend_state(struct pipe_context *pipe, void *state)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   ctx->blend = (struct pvrgpu_blend_state *)state;
   pvrgpu_counter_eventf("bind_blend_state",
                         "bound=%u rt0_enable=%u rt0_colormask=0x%x",
                         state ? 1 : 0,
                         ctx->blend ? ctx->blend->state.rt[0].blend_enable : 0,
                         ctx->blend ? ctx->blend->state.rt[0].colormask : 0);
}

static void
pvrgpu_bind_rasterizer_state(struct pipe_context *pipe, void *state)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   ctx->rasterizer = (struct pvrgpu_rasterizer_state *)state;
   pvrgpu_counter_eventf("bind_rasterizer_state",
                         "bound=%u front_ccw=%u cull_face=%u scissor=%u",
                         state ? 1 : 0,
                         ctx->rasterizer ? ctx->rasterizer->state.front_ccw : 0,
                         ctx->rasterizer ? ctx->rasterizer->state.cull_face : 0,
                         ctx->rasterizer ? ctx->rasterizer->state.scissor : 0);
}

static void
pvrgpu_bind_depth_stencil_alpha_state(struct pipe_context *pipe, void *state)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   ctx->dsa = (struct pvrgpu_depth_stencil_alpha_state *)state;
   pvrgpu_counter_eventf("bind_depth_stencil_alpha_state",
                         "bound=%u depth_enable=%u depth_write=%u "
                         "depth_func=%u stencil0_enable=%u alpha_enable=%u",
                         state ? 1 : 0,
                         ctx->dsa ? ctx->dsa->state.depth_enabled : 0,
                         ctx->dsa ? ctx->dsa->state.depth_writemask : 0,
                         ctx->dsa ? ctx->dsa->state.depth_func : 0,
                         ctx->dsa ? ctx->dsa->state.stencil[0].enabled : 0,
                         ctx->dsa ? ctx->dsa->state.alpha_enabled : 0);
}

static void *
pvrgpu_create_shader_state_for_stage(struct pipe_context *pipe,
                                     const struct pipe_shader_state *state,
                                     mesa_shader_stage stage)
{
   (void)pipe;
   struct pvrgpu_shader_state *shader = CALLOC_STRUCT(pvrgpu_shader_state);
   if (!shader)
      return NULL;

   shader->stage = stage;
   if (state) {
      shader->type = state->type;
      shader->has_tgsi = state->tokens != NULL;
      shader->has_nir = state->ir.nir != NULL;
      shader->tgsi = state->tokens;
      shader->nir = state->ir.nir;
      shader->stream_output = state->stream_output;
   }
   pvrgpu_counter_eventf("create_shader",
                         "stage=%s ir=%u has_nir=%u has_tgsi=%u",
                         pvrgpu_shader_stage_name(stage),
                         shader->type,
                         shader->has_nir ? 1 : 0,
                         shader->has_tgsi ? 1 : 0);
   return shader;
}

static void *
pvrgpu_create_vs_state(struct pipe_context *pipe,
                       const struct pipe_shader_state *state)
{
   return pvrgpu_create_shader_state_for_stage(pipe, state, MESA_SHADER_VERTEX);
}

static void *
pvrgpu_create_fs_state(struct pipe_context *pipe,
                       const struct pipe_shader_state *state)
{
   return pvrgpu_create_shader_state_for_stage(pipe, state, MESA_SHADER_FRAGMENT);
}

static void *
pvrgpu_create_unsupported_shader_state(struct pipe_context *pipe,
                                       const struct pipe_shader_state *state)
{
   return pvrgpu_create_shader_state_for_stage(pipe, state, MESA_SHADER_NONE);
}

static void
pvrgpu_bind_vs_state(struct pipe_context *pipe, void *state)
{
   pvrgpu_context(pipe)->vs = (struct pvrgpu_shader_state *)state;
   pvrgpu_counter_eventf("bind_shader",
                         "stage=vertex bound=%u",
                         state ? 1 : 0);
}

static void
pvrgpu_bind_fs_state(struct pipe_context *pipe, void *state)
{
   pvrgpu_context(pipe)->fs = (struct pvrgpu_shader_state *)state;
   pvrgpu_counter_eventf("bind_shader",
                         "stage=fragment bound=%u",
                         state ? 1 : 0);
}

static void
pvrgpu_delete_shader_state(struct pipe_context *pipe, void *state)
{
   (void)pipe;
   struct pvrgpu_shader_state *shader = (struct pvrgpu_shader_state *)state;
   if (shader)
      ralloc_free(shader->nir);
   FREE(shader);
}

static void *
pvrgpu_create_vertex_elements_state(struct pipe_context *pipe,
                                    unsigned num_elements,
                                    const struct pipe_vertex_element *elements)
{
   (void)pipe;
   if (num_elements > PIPE_MAX_ATTRIBS)
      return NULL;

   struct pvrgpu_vertex_elements_state *state =
      CALLOC_STRUCT(pvrgpu_vertex_elements_state);
   if (!state)
      return NULL;

   state->num_elements = num_elements;
   if (num_elements)
      memcpy(state->elements, elements,
             num_elements * sizeof(struct pipe_vertex_element));

   pvrgpu_counter_eventf("create_vertex_elements",
                         "count=%u first_format=%u first_stride=%u",
                         state->num_elements,
                         state->num_elements ? state->elements[0].src_format : 0,
                         state->num_elements ? state->elements[0].src_stride : 0);
   return state;
}

static void
pvrgpu_bind_vertex_elements_state(struct pipe_context *pipe, void *state)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   ctx->vertex_elements = (struct pvrgpu_vertex_elements_state *)state;
   pvrgpu_counter_eventf("bind_vertex_elements",
                         "count=%u",
                         ctx->vertex_elements ?
                            ctx->vertex_elements->num_elements : 0);
}

static void
pvrgpu_bind_sampler_states(struct pipe_context *pipe,
                           mesa_shader_stage shader,
                           unsigned start_slot,
                           unsigned num_samplers,
                           void **samplers)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   if (!pvrgpu_is_valid_context_stage(shader) ||
       start_slot >= PIPE_MAX_SAMPLERS) {
      pvrgpu_counter_eventf("bind_sampler_states_unsupported",
                            "stage=%s start=%u count=%u",
                            pvrgpu_shader_stage_name(shader),
                            start_slot,
                            num_samplers);
      return;
   }

   const unsigned max_slots = PIPE_MAX_SAMPLERS - start_slot;
   const unsigned bind_count =
      num_samplers < max_slots ? num_samplers : max_slots;
   for (unsigned i = 0; i < bind_count; ++i) {
      ctx->samplers[shader][start_slot + i] =
         samplers ? (struct pvrgpu_sampler_state *)samplers[i] : NULL;
   }

   ctx->num_samplers[shader] = pvrgpu_count_bound_samplers(ctx, shader);
   struct pvrgpu_sampler_state *first =
      bind_count ? ctx->samplers[shader][start_slot] : NULL;
   pvrgpu_counter_eventf("bind_sampler_states",
                         "stage=%s start=%u count=%u bound=%u "
                         "total=%u first_min_img_filter=%u "
                         "first_mag_img_filter=%u first_wrap=%u,%u,%u",
                         pvrgpu_shader_stage_name(shader),
                         start_slot,
                         bind_count,
                         first ? 1 : 0,
                         ctx->num_samplers[shader],
                         first ? first->state.min_img_filter : 0,
                         first ? first->state.mag_img_filter : 0,
                         first ? first->state.wrap_s : 0,
                         first ? first->state.wrap_t : 0,
                         first ? first->state.wrap_r : 0);
}

static struct pipe_sampler_view *
pvrgpu_create_sampler_view(struct pipe_context *pipe,
                           struct pipe_resource *texture,
                           const struct pipe_sampler_view *template)
{
   if (!texture || !template)
      return NULL;

   struct pvrgpu_sampler_view *pvrgpu_view =
      CALLOC_STRUCT(pvrgpu_sampler_view);
   if (!pvrgpu_view)
      return NULL;

   struct pipe_sampler_view *view = &pvrgpu_view->base;
   *view = *template;
   view->texture = NULL;
   pipe_resource_reference(&view->texture, texture);
   pipe_reference_init(&view->reference, 1);
   view->context = pipe;
   pvrgpu_counter_eventf("create_sampler_view",
                         "target=%u texture_target=%u format=%s "
                         "texture_format=%s width=%u height=%u levels=%u-%u "
                         "swizzle=%u,%u,%u,%u",
                         view->target,
                         texture->target,
                         util_format_name(view->format),
                         util_format_name(texture->format),
                         texture->width0,
                         texture->height0,
                         view->u.tex.first_level,
                         view->u.tex.last_level,
                         view->swizzle_r,
                         view->swizzle_g,
                         view->swizzle_b,
                         view->swizzle_a);
   return view;
}

static void
pvrgpu_sampler_view_destroy(struct pipe_context *pipe,
                            struct pipe_sampler_view *view)
{
   (void)pipe;
   if (!view)
      return;

   pvrgpu_counter_eventf("destroy_sampler_view",
                         "target=%u has_texture=%u",
                         view->target,
                         view->texture ? 1 : 0);
   pipe_resource_reference(&view->texture, NULL);
   FREE(view);
}

static void
pvrgpu_set_blend_color(struct pipe_context *pipe,
                       const struct pipe_blend_color *state)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   if (state)
      ctx->blend_color = *state;
   pvrgpu_counter_eventf("set_blend_color",
                         "rgba=%f,%f,%f,%f",
                         ctx->blend_color.color[0],
                         ctx->blend_color.color[1],
                         ctx->blend_color.color[2],
                         ctx->blend_color.color[3]);
}

static void
pvrgpu_set_stencil_ref(struct pipe_context *pipe,
                       const struct pipe_stencil_ref state)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   ctx->stencil_ref = state;
   pvrgpu_counter_eventf("set_stencil_ref",
                         "front=%u back=%u",
                         ctx->stencil_ref.ref_value[0],
                         ctx->stencil_ref.ref_value[1]);
}

static void
pvrgpu_set_sample_mask(struct pipe_context *pipe, unsigned sample_mask)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   ctx->sample_mask = sample_mask;
   pvrgpu_counter_eventf("set_sample_mask", "mask=0x%x", sample_mask);
}

static void
pvrgpu_set_min_samples(struct pipe_context *pipe, unsigned min_samples)
{
   (void)pipe;
   (void)min_samples;
}

static void
pvrgpu_set_clip_state(struct pipe_context *pipe,
                      const struct pipe_clip_state *state)
{
   (void)pipe;
   (void)state;
}

static void
pvrgpu_set_inlinable_constants(struct pipe_context *pipe,
                               mesa_shader_stage shader,
                               uint num_values,
                               uint32_t *values)
{
   (void)pipe;
   pvrgpu_counter_eventf("set_inlinable_constants",
                         "stage=%s values=%u has_values=%u first=0x%08x",
                         pvrgpu_shader_stage_name(shader),
                         num_values,
                         values && num_values ? 1 : 0,
                         values && num_values ? values[0] : 0);
}

static void
pvrgpu_set_sample_locations(struct pipe_context *pipe,
                            size_t size,
                            const uint8_t *locations)
{
   (void)pipe;
   (void)size;
   (void)locations;
}

static void
pvrgpu_set_polygon_stipple(struct pipe_context *pipe,
                           const struct pipe_poly_stipple *state)
{
   (void)pipe;
   (void)state;
}

static void
pvrgpu_set_window_rectangles(struct pipe_context *pipe,
                             bool include,
                             unsigned num_rectangles,
                             const struct pipe_scissor_state *rectangles)
{
   (void)pipe;
   (void)include;
   (void)num_rectangles;
   (void)rectangles;
}

static void
pvrgpu_set_tess_state(struct pipe_context *pipe,
                      const float default_outer_level[4],
                      const float default_inner_level[2])
{
   (void)pipe;
   (void)default_outer_level;
   (void)default_inner_level;
}

static void
pvrgpu_set_patch_vertices(struct pipe_context *pipe, uint8_t patch_vertices)
{
   (void)pipe;
   (void)patch_vertices;
}

static void
pvrgpu_set_constant_buffer(struct pipe_context *pipe,
                           mesa_shader_stage shader,
                           uint index,
                           const struct pipe_constant_buffer *buffer)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   if (!pvrgpu_is_valid_context_stage(shader) ||
       index >= PIPE_MAX_CONSTANT_BUFFERS) {
      pvrgpu_counter_eventf("set_constant_buffer_unsupported",
                            "stage=%s index=%u",
                            pvrgpu_shader_stage_name(shader),
                            index);
      return;
   }

   util_copy_constant_buffer(&ctx->constant_buffers[shader][index], buffer);
   ctx->num_constant_buffers[shader] =
      pvrgpu_count_bound_constant_buffers(ctx, shader);

   const struct pipe_constant_buffer *bound =
      &ctx->constant_buffers[shader][index];
   uint32_t words[4];
   const bool has_words = pvrgpu_constant_buffer_first_words(bound, words);
   pvrgpu_counter_eventf("set_constant_buffer",
                         "stage=%s index=%u has_buffer=%u has_resource=%u "
                         "has_user=%u offset=%u size=%u total=%u "
                         "has_words=%u first_words=0x%08x,0x%08x,0x%08x,0x%08x",
                         pvrgpu_shader_stage_name(shader),
                         index,
                         pvrgpu_constant_buffer_is_bound(bound) ? 1 : 0,
                         bound->buffer ? 1 : 0,
                         bound->user_buffer ? 1 : 0,
                         bound->buffer_offset,
                         bound->buffer_size,
                         ctx->num_constant_buffers[shader],
                         has_words ? 1 : 0,
                         words[0],
                         words[1],
                         words[2],
                         words[3]);
}

static void
pvrgpu_set_sampler_views(struct pipe_context *pipe,
                         mesa_shader_stage shader,
                         unsigned start_slot,
                         unsigned num_views,
                         unsigned unbind_num_trailing_slots,
                         struct pipe_sampler_view **views)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   if (!pvrgpu_is_valid_context_stage(shader) ||
       start_slot >= PIPE_MAX_SHADER_SAMPLER_VIEWS) {
      pvrgpu_counter_eventf("set_sampler_views_unsupported",
                            "stage=%s start=%u count=%u unbind=%u",
                            pvrgpu_shader_stage_name(shader),
                            start_slot,
                            num_views,
                            unbind_num_trailing_slots);
      return;
   }

   const unsigned max_slots = PIPE_MAX_SHADER_SAMPLER_VIEWS - start_slot;
   const unsigned bind_count = num_views < max_slots ? num_views : max_slots;
   for (unsigned i = 0; i < bind_count; ++i) {
      pipe_sampler_view_reference(&ctx->sampler_views[shader][start_slot + i],
                                  views ? views[i] : NULL);
   }

   unsigned unbound_count = 0;
   const unsigned first_unbind_slot = start_slot + bind_count;
   while (unbound_count < unbind_num_trailing_slots &&
          first_unbind_slot + unbound_count <
             PIPE_MAX_SHADER_SAMPLER_VIEWS) {
      pipe_sampler_view_reference(
         &ctx->sampler_views[shader][first_unbind_slot + unbound_count],
         NULL);
      unbound_count++;
   }

   ctx->num_sampler_views[shader] =
      pvrgpu_count_bound_sampler_views(ctx, shader);
   struct pipe_sampler_view *first =
      bind_count ? ctx->sampler_views[shader][start_slot] : NULL;
   struct pipe_resource *texture = first ? first->texture : NULL;
   pvrgpu_counter_eventf("set_sampler_views",
                         "stage=%s start=%u count=%u unbind=%u bound=%u "
                         "total=%u first_target=%u first_format=%s "
                         "texture=%ux%u texture_format=%s",
                         pvrgpu_shader_stage_name(shader),
                         start_slot,
                         bind_count,
                         unbound_count,
                         first ? 1 : 0,
                         ctx->num_sampler_views[shader],
                         first ? first->target : 0,
                         first ? util_format_name(first->format) : "none",
                         texture ? texture->width0 : 0,
                         texture ? texture->height0 : 0,
                         texture ? util_format_name(texture->format) : "none");
}

static void
pvrgpu_set_scissor_states(struct pipe_context *pipe,
                          unsigned start_slot,
                          unsigned num_scissors,
                          const struct pipe_scissor_state *states)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   if (start_slot == 0 && num_scissors != 0 && states) {
      ctx->scissor = states[0];
      ctx->has_scissor = true;
      pvrgpu_counter_eventf("set_scissor",
                            "minx=%u miny=%u maxx=%u maxy=%u",
                            ctx->scissor.minx,
                            ctx->scissor.miny,
                            ctx->scissor.maxx,
                            ctx->scissor.maxy);
   }
}

static void
pvrgpu_set_viewport_states(struct pipe_context *pipe,
                           unsigned start_slot,
                           unsigned num_viewports,
                           const struct pipe_viewport_state *states)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   if (start_slot == 0 && num_viewports != 0 && states) {
      ctx->viewport = states[0];
      ctx->has_viewport = true;
      pvrgpu_counter_eventf("set_viewport",
                            "scale=%f,%f,%f translate=%f,%f,%f",
                            ctx->viewport.scale[0],
                            ctx->viewport.scale[1],
                            ctx->viewport.scale[2],
                            ctx->viewport.translate[0],
                            ctx->viewport.translate[1],
                            ctx->viewport.translate[2]);
   }
}

static void
pvrgpu_set_vertex_buffers(struct pipe_context *pipe,
                          unsigned count,
                          const struct pipe_vertex_buffer *buffers)
{
   struct pvrgpu_context *ctx = pvrgpu_context(pipe);
   if (count > PIPE_MAX_ATTRIBS)
      count = PIPE_MAX_ATTRIBS;

   util_set_vertex_buffers_count(ctx->vertex_buffers,
                                 &ctx->num_vertex_buffers,
                                 buffers,
                                 count);
   pvrgpu_counter_eventf("set_vertex_buffers",
                         "count=%u first_user=%u first_offset=%u",
                         ctx->num_vertex_buffers,
                         ctx->num_vertex_buffers ?
                            ctx->vertex_buffers[0].is_user_buffer : 0,
                         ctx->num_vertex_buffers ?
                            ctx->vertex_buffers[0].buffer_offset : 0);
}

static enum pipe_reset_status
pvrgpu_get_device_reset_status(struct pipe_context *pipe)
{
   (void)pipe;
   return PIPE_NO_RESET;
}

void
pvrgpu_init_state_functions(struct pipe_context *pipe)
{
   pipe->create_blend_state = pvrgpu_create_blend_state;
   pipe->bind_blend_state = pvrgpu_bind_blend_state;
   pipe->delete_blend_state = pvrgpu_delete_state_object;

   pipe->create_sampler_state = pvrgpu_create_sampler_state;
   pipe->bind_sampler_states = pvrgpu_bind_sampler_states;
   pipe->delete_sampler_state = pvrgpu_delete_state_object;

   pipe->create_rasterizer_state = pvrgpu_create_rasterizer_state;
   pipe->bind_rasterizer_state = pvrgpu_bind_rasterizer_state;
   pipe->delete_rasterizer_state = pvrgpu_delete_state_object;

   pipe->create_depth_stencil_alpha_state =
      pvrgpu_create_depth_stencil_alpha_state;
   pipe->bind_depth_stencil_alpha_state =
      pvrgpu_bind_depth_stencil_alpha_state;
   pipe->delete_depth_stencil_alpha_state = pvrgpu_delete_state_object;

   pipe->create_fs_state = pvrgpu_create_fs_state;
   pipe->bind_fs_state = pvrgpu_bind_fs_state;
   pipe->delete_fs_state = pvrgpu_delete_shader_state;

   pipe->create_vs_state = pvrgpu_create_vs_state;
   pipe->bind_vs_state = pvrgpu_bind_vs_state;
   pipe->delete_vs_state = pvrgpu_delete_shader_state;

   pipe->create_gs_state = pvrgpu_create_unsupported_shader_state;
   pipe->bind_gs_state = pvrgpu_bind_state_object;
   pipe->delete_gs_state = pvrgpu_delete_shader_state;

   pipe->create_tcs_state = pvrgpu_create_unsupported_shader_state;
   pipe->bind_tcs_state = pvrgpu_bind_state_object;
   pipe->delete_tcs_state = pvrgpu_delete_shader_state;

   pipe->create_tes_state = pvrgpu_create_unsupported_shader_state;
   pipe->bind_tes_state = pvrgpu_bind_state_object;
   pipe->delete_tes_state = pvrgpu_delete_shader_state;

   pipe->create_vertex_elements_state = pvrgpu_create_vertex_elements_state;
   pipe->bind_vertex_elements_state = pvrgpu_bind_vertex_elements_state;
   pipe->delete_vertex_elements_state = pvrgpu_delete_state_object;

   pipe->create_sampler_view = pvrgpu_create_sampler_view;
   pipe->sampler_view_destroy = pvrgpu_sampler_view_destroy;
   pipe->sampler_view_release = u_default_sampler_view_release;

   pipe->set_blend_color = pvrgpu_set_blend_color;
   pipe->set_stencil_ref = pvrgpu_set_stencil_ref;
   pipe->set_sample_mask = pvrgpu_set_sample_mask;
   pipe->set_min_samples = pvrgpu_set_min_samples;
   pipe->set_clip_state = pvrgpu_set_clip_state;
   pipe->set_constant_buffer = pvrgpu_set_constant_buffer;
   pipe->set_inlinable_constants = pvrgpu_set_inlinable_constants;
   pipe->set_sampler_views = pvrgpu_set_sampler_views;
   pipe->set_scissor_states = pvrgpu_set_scissor_states;
   pipe->set_sample_locations = pvrgpu_set_sample_locations;
   pipe->set_polygon_stipple = pvrgpu_set_polygon_stipple;
   pipe->set_window_rectangles = pvrgpu_set_window_rectangles;
   pipe->set_viewport_states = pvrgpu_set_viewport_states;
   pipe->set_vertex_buffers = pvrgpu_set_vertex_buffers;
   pipe->set_tess_state = pvrgpu_set_tess_state;
   pipe->set_patch_vertices = pvrgpu_set_patch_vertices;
   pipe->set_debug_callback = u_default_set_debug_callback;

   pipe->get_device_reset_status = pvrgpu_get_device_reset_status;
}
