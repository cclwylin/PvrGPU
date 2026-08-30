/* SPDX-License-Identifier: MIT */

#include "pvrgpu_context.h"
#include "pvrgpu_counter.h"
#include "pvrgpu_resource.h"
#include "pvrgpu_screen.h"

#include "compiler/nir/nir.h"
#include "frontend/sw_winsys.h"
#include "pipe/p_defines.h"
#include "util/u_memory.h"
#include "util/u_screen.h"

static const nir_shader_compiler_options pvrgpu_nir_options = {
   .max_unroll_iterations = 32,
};

static void
pvrgpu_init_single_shader_caps(struct pipe_screen *screen,
                               mesa_shader_stage shader)
{
   struct pipe_shader_caps *caps =
      (struct pipe_shader_caps *)&screen->shader_caps[shader];

   caps->max_instructions =
   caps->max_alu_instructions =
   caps->max_tex_instructions =
   caps->max_tex_indirections = 16384;
   caps->max_control_flow_depth = 8;
   caps->max_inputs = shader == MESA_SHADER_FRAGMENT ? 8 : 16;
   caps->max_outputs = shader == MESA_SHADER_FRAGMENT ? 1 : 8;
   caps->max_const_buffer0_size = 16 * 1024;
   caps->max_const_buffers = 1;
   caps->max_temps = 256;
   caps->max_texture_samplers = 16;
   caps->max_sampler_views = 16;
   caps->supported_irs =
      (1 << PIPE_SHADER_IR_NIR) | (1 << PIPE_SHADER_IR_TGSI);
}

static void
pvrgpu_init_shader_caps(struct pipe_screen *screen)
{
   pvrgpu_init_single_shader_caps(screen, MESA_SHADER_VERTEX);
   pvrgpu_init_single_shader_caps(screen, MESA_SHADER_FRAGMENT);

   screen->nir_options[MESA_SHADER_VERTEX] = &pvrgpu_nir_options;
   screen->nir_options[MESA_SHADER_FRAGMENT] = &pvrgpu_nir_options;
}

static const char *
pvrgpu_get_vendor(struct pipe_screen *screen)
{
   (void)screen;
   return "PvrGPU";
}

static const char *
pvrgpu_get_name(struct pipe_screen *screen)
{
   (void)screen;
   return "PvrGPU SystemC Gallium bring-up";
}

static const char *
pvrgpu_get_device_vendor(struct pipe_screen *screen)
{
   (void)screen;
   return "PvrGPU";
}

static int
pvrgpu_get_screen_fd(struct pipe_screen *screen)
{
   struct pvrgpu_screen *pvrgpu = pvrgpu_screen(screen);
   if (pvrgpu->winsys && pvrgpu->winsys->get_fd)
      return pvrgpu->winsys->get_fd(pvrgpu->winsys);
   return -1;
}

static void
pvrgpu_fence_reference(struct pipe_screen *screen,
                       struct pipe_fence_handle **ptr,
                       struct pipe_fence_handle *fence)
{
   (void)screen;
   pvrgpu_counter_eventf("fence_reference",
                         "has_ptr=%u has_fence=%u",
                         ptr ? 1 : 0,
                         fence ? 1 : 0);
   if (ptr)
      *ptr = fence;
}

static bool
pvrgpu_fence_finish(struct pipe_screen *screen,
                    struct pipe_context *ctx,
                    struct pipe_fence_handle *fence,
                    uint64_t timeout)
{
   (void)screen;
   pvrgpu_counter_eventf("fence_finish",
                         "has_context=%u has_fence=%u timeout=%llu complete=1",
                         ctx ? 1 : 0,
                         fence ? 1 : 0,
                         (unsigned long long)timeout);
   return true;
}

static void
pvrgpu_init_screen_caps(struct pipe_screen *screen)
{
   struct pipe_caps *caps = (struct pipe_caps *)&screen->caps;

   u_init_pipe_screen_caps(screen, 0);
   caps->npot_textures = true;
   caps->blend_equation_separate = true;
   caps->texture_swizzle = true;
   caps->vertex_color_unclamped = true;
   caps->fragment_color_clamped = true;
   caps->fs_coord_origin_upper_left = true;
   caps->fs_coord_pixel_center_half_integer = true;
   caps->max_texture_2d_size = 4096;
   caps->max_texture_cube_levels = 13;
   caps->max_render_targets = 1;
   caps->max_constant_buffer_size = 16 * 1024;
   caps->max_vertex_attrib_stride = 2048;
   caps->max_vertex_buffers = PIPE_MAX_ATTRIBS;
   caps->max_stream_output_buffers = PIPE_MAX_SO_BUFFERS;
   caps->max_stream_output_separate_components = 16 * 4;
   caps->max_stream_output_interleaved_components = 16 * 4;
   caps->max_vertex_streams = 1;
   caps->stream_output_pause_resume = true;
   caps->stream_output_interleave_buffers = true;
   caps->user_vertex_buffers = true;
   caps->texture_transfer_modes = 0;
   caps->essl_feature_level = 100;
}

static bool
pvrgpu_is_supported_color_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8X8_UNORM:
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_B8G8R8X8_UNORM:
   case PIPE_FORMAT_R10G10B10A2_UNORM:
   case PIPE_FORMAT_B10G10R10A2_UNORM:
      return true;
   default:
      return false;
   }
}

static bool
pvrgpu_is_supported_depth_stencil_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_Z32_FLOAT:
   case PIPE_FORMAT_Z32_UNORM:
   case PIPE_FORMAT_Z32_FLOAT_S8X24_UINT:
      return true;
   default:
      return false;
   }
}

static bool
pvrgpu_is_supported_vertex_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R8_UINT:
   case PIPE_FORMAT_R16_UINT:
   case PIPE_FORMAT_R32_UINT:
   case PIPE_FORMAT_R32_FLOAT:
   case PIPE_FORMAT_R32G32_FLOAT:
   case PIPE_FORMAT_R32G32B32_FLOAT:
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
      return true;
   default:
      return false;
   }
}

static bool
pvrgpu_is_supported_texture_target(enum pipe_texture_target target)
{
   switch (target) {
   case PIPE_TEXTURE_2D:
   case PIPE_TEXTURE_2D_ARRAY:
   case PIPE_TEXTURE_3D:
   case PIPE_TEXTURE_CUBE:
      return true;
   default:
      return false;
   }
}

static bool
pvrgpu_is_format_supported(struct pipe_screen *screen,
                           enum pipe_format format,
                           enum pipe_texture_target target,
                           unsigned sample_count,
                           unsigned storage_sample_count,
                           unsigned bind)
{
   (void)screen;
   if (sample_count > 1 || storage_sample_count > 1)
      return false;

   if (target == PIPE_BUFFER) {
      const unsigned supported_buffer_binds =
         PIPE_BIND_VERTEX_BUFFER |
         PIPE_BIND_INDEX_BUFFER |
         PIPE_BIND_CONSTANT_BUFFER |
         PIPE_BIND_STREAM_OUTPUT |
         PIPE_BIND_QUERY_BUFFER;
      if (!(bind & supported_buffer_binds))
         return false;
      if (bind & ~supported_buffer_binds)
         return false;
      if (bind & (PIPE_BIND_STREAM_OUTPUT | PIPE_BIND_QUERY_BUFFER))
         return format == PIPE_FORMAT_NONE ||
                format == PIPE_FORMAT_R8_UNORM ||
                pvrgpu_is_supported_vertex_format(format);
      return pvrgpu_is_supported_vertex_format(format);
   }

   if (!pvrgpu_is_supported_texture_target(target))
      return false;
   if (pvrgpu_is_supported_color_format(format)) {
      unsigned supported_binds = PIPE_BIND_SAMPLER_VIEW;
      if (target == PIPE_TEXTURE_2D)
         supported_binds |= PIPE_BIND_RENDER_TARGET | PIPE_BIND_DISPLAY_TARGET;
      if (bind & ~supported_binds)
         return false;
      return true;
   }
   if (target != PIPE_TEXTURE_2D)
      return false;
   if (pvrgpu_is_supported_depth_stencil_format(format)) {
      if (bind & ~(PIPE_BIND_DEPTH_STENCIL | PIPE_BIND_SAMPLER_VIEW))
         return false;
      return true;
   }
   return false;
}

static void
pvrgpu_screen_destroy(struct pipe_screen *screen)
{
   FREE(pvrgpu_screen(screen));
}

struct pipe_screen *
pvrgpu_create_screen(struct sw_winsys *winsys,
                     const struct pipe_screen_config *config)
{
   (void)config;
   struct pvrgpu_screen *screen = CALLOC_STRUCT(pvrgpu_screen);
   if (!screen)
      return NULL;

   screen->winsys = winsys;
   screen->base.get_vendor = pvrgpu_get_vendor;
   screen->base.get_name = pvrgpu_get_name;
   screen->base.get_device_vendor = pvrgpu_get_device_vendor;
   screen->base.get_screen_fd = pvrgpu_get_screen_fd;
   screen->base.is_format_supported = pvrgpu_is_format_supported;
   screen->base.context_create = pvrgpu_create_context;
   screen->base.fence_reference = pvrgpu_fence_reference;
   screen->base.fence_finish = pvrgpu_fence_finish;
   screen->base.destroy = pvrgpu_screen_destroy;
   pvrgpu_init_resource_functions(&screen->base);
   pvrgpu_init_shader_caps(&screen->base);
   pvrgpu_init_screen_caps(&screen->base);
   return &screen->base;
}
