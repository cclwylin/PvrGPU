/* SPDX-License-Identifier: MIT */

#include "pvrgpu_context.h"
#include "pvrgpu_counter.h"
#include "pvrgpu_resource.h"
#include "pvrgpu_screen.h"

#include "compiler/nir/nir.h"
#include "frontend/sw_winsys.h"
#include "pipe/p_defines.h"
#include "util/format/u_format.h"
#include "util/u_memory.h"
#include "util/u_screen.h"

#include <stdlib.h>

static const nir_shader_compiler_options pvrgpu_nir_options = {
   .max_unroll_iterations = 32,
};

/*
 * OpenGL ES 3.0 / 3.1 advertisement.
 *
 * Mesa derives the reported ES version from screen caps and format support
 * (see _mesa_compute_version -> compute_version_es2 and st_init_extensions).
 * The caps below are what that computation needs; the driver still implements
 * only its bring-up slice, so exposing them makes ES3 contexts and shaders
 * reachable rather than making the whole feature set work.  Set
 * PVRGPU_DISABLE_ES3=1 to fall back to the earlier ES2-only surface, which is
 * the reference point when a regression has to be bisected.
 */
static bool
pvrgpu_es3_enabled(void)
{
   const char *disabled = getenv("PVRGPU_DISABLE_ES3");
   return !(disabled && disabled[0] != '\0' && disabled[0] != '0');
}

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
   caps->max_inputs = 32;
   caps->max_outputs = 32;
   caps->max_const_buffer0_size = 64 * 1024;
   caps->max_const_buffers = 16;
   caps->max_temps = 256;
   caps->max_texture_samplers = 16;
   caps->max_sampler_views = 16;
   if (pvrgpu_es3_enabled()) {
      /*
       * ES 3.1 needs shader storage, images and atomic counters in at least
       * the fragment and compute stages.  Leaving max_hw_atomic_counters at 0
       * makes Mesa carve the atomic counter buffers out of the SSBO budget
       * (st_init_limits), which is the arrangement drivers without dedicated
       * atomic hardware use.
       */
      caps->max_shader_buffers = 16;
      caps->max_shader_images = 8;
   }
   caps->supported_irs =
      (1 << PIPE_SHADER_IR_NIR) | (1 << PIPE_SHADER_IR_TGSI);
   caps->indirect_temp_addr = true;
   caps->indirect_const_addr = true;
   caps->integers = true;
}

static void
pvrgpu_init_shader_caps(struct pipe_screen *screen)
{
   pvrgpu_init_single_shader_caps(screen, MESA_SHADER_VERTEX);
   pvrgpu_init_single_shader_caps(screen, MESA_SHADER_FRAGMENT);
   pvrgpu_init_single_shader_caps(screen, MESA_SHADER_GEOMETRY);
   pvrgpu_init_single_shader_caps(screen, MESA_SHADER_TESS_CTRL);
   pvrgpu_init_single_shader_caps(screen, MESA_SHADER_TESS_EVAL);
   if (pvrgpu_es3_enabled())
      pvrgpu_init_single_shader_caps(screen, MESA_SHADER_COMPUTE);

   screen->nir_options[MESA_SHADER_VERTEX] = &pvrgpu_nir_options;
   screen->nir_options[MESA_SHADER_FRAGMENT] = &pvrgpu_nir_options;
   screen->nir_options[MESA_SHADER_GEOMETRY] = &pvrgpu_nir_options;
   screen->nir_options[MESA_SHADER_TESS_CTRL] = &pvrgpu_nir_options;
   screen->nir_options[MESA_SHADER_TESS_EVAL] = &pvrgpu_nir_options;
   if (pvrgpu_es3_enabled())
      screen->nir_options[MESA_SHADER_COMPUTE] = &pvrgpu_nir_options;
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
   caps->shareable_shaders = false;
   caps->npot_textures = true;
   caps->blend_equation_separate = true;
   caps->texture_swizzle = true;
   caps->vertex_color_unclamped = true;
   caps->fragment_color_clamped = true;
   caps->fs_coord_origin_upper_left = true;
   /*
    * Lower scissored and channel-masked clears in the driver.  Without these
    * the state tracker rewrites every such glClear into an internal
    * triangle-fan quad draw, which this bring-up driver cannot lower and has
    * to reject as an unsupported draw.  Gallium has no separate cap for the
    * stencil write mask, so a masked stencil clear now also reaches
    * pvrgpu_clear(); it stays fail-closed there exactly as the quad lowering
    * was.
    */
   caps->clear_scissored = true;
   caps->clear_masked = true;
   caps->fs_coord_pixel_center_half_integer = true;
   caps->max_texture_2d_size = 4096;
   caps->max_texture_3d_levels = 9;
   caps->max_texture_array_layers = 256;
   caps->max_texture_cube_levels = 13;
   caps->max_render_targets = 8;
   caps->max_constant_buffer_size = 64 * 1024;
   caps->constant_buffer_offset_alignment = 16;
   caps->max_vertex_attrib_stride = 2048;
   caps->max_vertex_buffers = PIPE_MAX_ATTRIBS;
   caps->max_stream_output_buffers = PIPE_MAX_SO_BUFFERS;
   caps->max_stream_output_separate_components = 16 * 4;
   caps->max_stream_output_interleaved_components = 16 * 4;
   caps->max_vertex_streams = 1;
   caps->stream_output_pause_resume = true;
   caps->stream_output_interleave_buffers = true;
   caps->user_vertex_buffers = true;
   caps->vs_instanceid = true;
   caps->vertex_element_instance_divisor = true;
   caps->texture_multisample = true;
   caps->cube_map_array = true;
   caps->sample_shading = true;
   caps->sampler_view_target = true;
   caps->texture_float_linear = true;
   caps->texture_half_float_linear = true;
   caps->texture_query_samples = true;
   caps->shader_samples_identical = true;
   caps->framebuffer_no_attachment = true;
   caps->shader_array_components = true;
   caps->draw_indirect = true;
   caps->fs_fine_derivative = true;
   caps->glsl_tess_levels_as_inputs = true;
   caps->copy_between_compressed_and_plain_formats = true;
   caps->texture_transfer_modes = 0;
   caps->max_viewports = PIPE_MAX_VIEWPORTS;
   caps->max_varyings = 32;
   caps->max_shader_patch_varyings = 32;
   caps->max_gs_invocations = 32;
   caps->max_geometry_output_vertices = 256;
   caps->max_geometry_total_output_components = 1024;
   caps->glsl_feature_level = 400;
   caps->glsl_feature_level_compatibility = 400;
   caps->essl_feature_level = 310;

   if (!pvrgpu_es3_enabled())
      return;

   /*
    * ES 3.0 gates, in the order compute_version_es2() checks them.
    * primitive restart backs both NV_primitive_restart and the fixed-index
    * form ES3 requires; mixed_framebuffer_sizes backs ARB_framebuffer_object;
    * fragment_shader_texture_lod backs ARB_shader_texture_lod.
    */
   caps->primitive_restart = true;
   caps->primitive_restart_fixed_index = true;
   caps->mixed_framebuffer_sizes = true;
   caps->fragment_shader_texture_lod = true;
   caps->seamless_cube_map = true;
   caps->occlusion_query = true;
   /*
    * indep_blend_enable is deliberately left off.  Mesa only needs it for
    * ES 3.2 (EXT_draw_buffers2 / ARB_draw_buffers_blend), which is out of
    * reach anyway, and the model has no per-render-target blend state to
    * back it.  Advertising it made st_atom_blend set independent_blend_enable
    * on a depth-only pass -- num_cb == 0 takes the promotion path and then
    * leaves rt[0] untouched -- which is state the model cannot describe.
    */

   /* ES 3.1 gates that are not compute. */
   caps->max_texture_gather_components = 4;
   caps->image_store_formatted = true;
   caps->shader_buffer_offset_alignment = 16;
   caps->max_combined_shader_buffers = 16;
   caps->max_combined_hw_atomic_counters = 0;
   caps->max_combined_hw_atomic_counter_buffers = 0;

   /*
    * ES 3.1 compute.  ARB_compute_shader additionally requires
    * max_threads_per_block >= 1024 (st_init_extensions), so the block limits
    * are the smallest ones that clear that bar.  The dispatch path itself is
    * still a stub -- see pvrgpu_launch_grid in pvrgpu_context.c.
    */
   caps->compute = true;

   struct pipe_compute_caps *compute =
      (struct pipe_compute_caps *)&screen->compute_caps;
   compute->address_bits = 64;
   compute->grid_dimension = 3;
   compute->max_grid_size[0] = 65535;
   compute->max_grid_size[1] = 65535;
   compute->max_grid_size[2] = 65535;
   compute->max_block_size[0] = 1024;
   compute->max_block_size[1] = 1024;
   compute->max_block_size[2] = 64;
   compute->max_threads_per_block = 1024;
   compute->max_variable_threads_per_block = 1024;
   compute->max_local_size = 32 * 1024;
   compute->max_compute_units = 1;
   compute->max_clock_frequency = 1;
   compute->max_mem_alloc_size = 1u << 30;
   compute->max_global_size = 1u << 30;
}

static bool
pvrgpu_is_supported_sample_count(unsigned sample_count)
{
   switch (sample_count) {
   case 0:
   case 1:
   case 2:
   case 3:
   case 4:
   case 5:
   case 6:
   case 7:
   case 8:
   case 9:
   case 10:
   case 11:
   case 12:
   case 13:
   case 14:
   case 15:
   case 16:
      return true;
   default:
      return false;
   }
}

static bool
pvrgpu_is_multisampled_request(unsigned sample_count,
                               unsigned storage_sample_count)
{
   return sample_count > 1 || storage_sample_count > 1;
}

static bool
pvrgpu_debug_format_support_enabled(void)
{
   const char *enabled = getenv("PVRGPU_DEBUG_FORMAT_SUPPORT");
   return enabled && enabled[0] != '\0' && enabled[0] != '0';
}

bool
pvrgpu_is_supported_color_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8X8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_SRGB:
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_B8G8R8X8_UNORM:
   case PIPE_FORMAT_A8R8G8B8_UNORM:
   case PIPE_FORMAT_A8B8G8R8_UNORM:
   case PIPE_FORMAT_R5G6B5_UNORM:
   case PIPE_FORMAT_B5G6R5_UNORM:
   case PIPE_FORMAT_R10G10B10A2_UNORM:
   case PIPE_FORMAT_B10G10R10A2_UNORM:
   case PIPE_FORMAT_R16_UNORM:
   case PIPE_FORMAT_R16G16_UNORM:
   case PIPE_FORMAT_R16G16B16A16_UNORM:
   case PIPE_FORMAT_R32G32B32A32_UNORM:
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
   case PIPE_FORMAT_R8G8B8A8_UINT:
   case PIPE_FORMAT_R8G8B8A8_SINT:
   case PIPE_FORMAT_R16G16B16A16_SINT:
   case PIPE_FORMAT_R32G32B32A32_SINT:
      return true;
   default:
      break;
   }

   if (!pvrgpu_es3_enabled())
      return false;

   /*
    * ES 3.0 colour-renderable and texturable formats.  ARB_texture_rg needs
    * R8/RG8, EXT_packed_float needs R11G11B10, EXT_texture_snorm and
    * EXT_render_snorm need the SNORM set, EXT_color_buffer_(half_)float needs
    * the R/RG float set, and ARB_ES3_compatibility additionally checks
    * R16/RG16 UNORM plus R16/RG16 SNORM as sampler formats.
    */
   switch (format) {
   /* ARB_texture_rg */
   case PIPE_FORMAT_R8_UNORM:
   case PIPE_FORMAT_R8G8_UNORM:
   /* EXT_texture_snorm / EXT_render_snorm */
   case PIPE_FORMAT_R8_SNORM:
   case PIPE_FORMAT_R8G8_SNORM:
   case PIPE_FORMAT_R8G8B8A8_SNORM:
   case PIPE_FORMAT_R16_SNORM:
   case PIPE_FORMAT_R16G16_SNORM:
   case PIPE_FORMAT_R16G16B16A16_SNORM:
   /* EXT_packed_float */
   case PIPE_FORMAT_R11G11B10_FLOAT:
   /* EXT_color_buffer_half_float / EXT_color_buffer_float */
   case PIPE_FORMAT_R16_FLOAT:
   case PIPE_FORMAT_R16G16_FLOAT:
   case PIPE_FORMAT_R32_FLOAT:
   case PIPE_FORMAT_R32G32_FLOAT:
   /* EXT_texture_integer and the ES3 integer texture set */
   case PIPE_FORMAT_R8_UINT:
   case PIPE_FORMAT_R8_SINT:
   case PIPE_FORMAT_R8G8_UINT:
   case PIPE_FORMAT_R8G8_SINT:
   case PIPE_FORMAT_R16_UINT:
   case PIPE_FORMAT_R16_SINT:
   case PIPE_FORMAT_R16G16_UINT:
   case PIPE_FORMAT_R16G16_SINT:
   case PIPE_FORMAT_R16G16B16A16_UINT:
   case PIPE_FORMAT_R32_UINT:
   case PIPE_FORMAT_R32_SINT:
   case PIPE_FORMAT_R32G32_UINT:
   case PIPE_FORMAT_R32G32_SINT:
   case PIPE_FORMAT_R32G32B32A32_UINT:
   /* ARB_texture_rgb10_a2ui */
   case PIPE_FORMAT_R10G10B10A2_UINT:
   case PIPE_FORMAT_B10G10R10A2_UINT:
   /* EXT_sRGB / EXT_texture_sRGB alternates */
   case PIPE_FORMAT_A8B8G8R8_SRGB:
   case PIPE_FORMAT_B8G8R8A8_SRGB:
      return true;
   default:
      return false;
   }
}

/*
 * Formats an ES3 context must be able to sample but never renders into.
 * Keeping them out of the colour list is what stops Mesa from advertising
 * them as colour-renderable.
 */
bool
pvrgpu_is_sampler_only_format(enum pipe_format format)
{
   if (!pvrgpu_es3_enabled())
      return false;

   switch (format) {
   /* EXT_texture_shared_exponent */
   case PIPE_FORMAT_R9G9B9E5_FLOAT:
   /* ARB_stencil_texturing (ES 3.1) */
   case PIPE_FORMAT_X24S8_UINT:
   case PIPE_FORMAT_S8X24_UINT:
      return true;
   default:
      return false;
   }
}

bool
pvrgpu_is_supported_depth_stencil_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_Z16_UNORM:
   case PIPE_FORMAT_Z24X8_UNORM:
   case PIPE_FORMAT_X8Z24_UNORM:
   case PIPE_FORMAT_Z24_UNORM_S8_UINT:
   case PIPE_FORMAT_S8_UINT_Z24_UNORM:
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
      break;
   }

   if (!pvrgpu_es3_enabled())
      return false;

   /*
    * ES 3.0 vertex attribute formats: the 8/16/32-bit integer and normalized
    * sets, half floats (ARB_half_float_vertex), and the packed 2_10_10_10_REV
    * and 10f_11f_11f_REV forms.
    */
   switch (format) {
   case PIPE_FORMAT_R8_SINT:
   case PIPE_FORMAT_R8_UNORM:
   case PIPE_FORMAT_R8_SNORM:
   case PIPE_FORMAT_R8G8_UINT:
   case PIPE_FORMAT_R8G8_SINT:
   case PIPE_FORMAT_R8G8_UNORM:
   case PIPE_FORMAT_R8G8_SNORM:
   case PIPE_FORMAT_R8G8B8_UINT:
   case PIPE_FORMAT_R8G8B8_SINT:
   case PIPE_FORMAT_R8G8B8_UNORM:
   case PIPE_FORMAT_R8G8B8_SNORM:
   case PIPE_FORMAT_R8G8B8A8_UINT:
   case PIPE_FORMAT_R8G8B8A8_SINT:
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_SNORM:
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_R16_SINT:
   case PIPE_FORMAT_R16_UNORM:
   case PIPE_FORMAT_R16_SNORM:
   case PIPE_FORMAT_R16_FLOAT:
   case PIPE_FORMAT_R16G16_UINT:
   case PIPE_FORMAT_R16G16_SINT:
   case PIPE_FORMAT_R16G16_UNORM:
   case PIPE_FORMAT_R16G16_SNORM:
   case PIPE_FORMAT_R16G16_FLOAT:
   case PIPE_FORMAT_R16G16B16_UINT:
   case PIPE_FORMAT_R16G16B16_SINT:
   case PIPE_FORMAT_R16G16B16_FLOAT:
   case PIPE_FORMAT_R16G16B16A16_UINT:
   case PIPE_FORMAT_R16G16B16A16_SINT:
   case PIPE_FORMAT_R16G16B16A16_UNORM:
   case PIPE_FORMAT_R16G16B16A16_SNORM:
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
   case PIPE_FORMAT_R32_SINT:
   case PIPE_FORMAT_R32G32_UINT:
   case PIPE_FORMAT_R32G32_SINT:
   case PIPE_FORMAT_R32G32B32_UINT:
   case PIPE_FORMAT_R32G32B32_SINT:
   case PIPE_FORMAT_R32G32B32A32_UINT:
   case PIPE_FORMAT_R32G32B32A32_SINT:
   case PIPE_FORMAT_R10G10B10A2_UNORM:
   case PIPE_FORMAT_R10G10B10A2_SNORM:
   case PIPE_FORMAT_R10G10B10A2_UINT:
   case PIPE_FORMAT_R10G10B10A2_USCALED:
   case PIPE_FORMAT_R10G10B10A2_SSCALED:
   case PIPE_FORMAT_B10G10R10A2_UNORM:
   case PIPE_FORMAT_B10G10R10A2_SNORM:
   case PIPE_FORMAT_B10G10R10A2_UINT:
   case PIPE_FORMAT_B10G10R10A2_USCALED:
   case PIPE_FORMAT_B10G10R10A2_SSCALED:
   case PIPE_FORMAT_R11G11B10_FLOAT:
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
   case PIPE_TEXTURE_CUBE_ARRAY:
      return true;
   default:
      return false;
   }
}

static bool
pvrgpu_texture_target_can_bind_surface(enum pipe_texture_target target)
{
   switch (target) {
   case PIPE_TEXTURE_2D:
   case PIPE_TEXTURE_2D_ARRAY:
   case PIPE_TEXTURE_3D:
   case PIPE_TEXTURE_CUBE:
   case PIPE_TEXTURE_CUBE_ARRAY:
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
   bool supported = false;

   if (!pvrgpu_is_supported_sample_count(sample_count) ||
       !pvrgpu_is_supported_sample_count(storage_sample_count))
      goto out;

   if (target == PIPE_BUFFER) {
      if (pvrgpu_is_multisampled_request(sample_count,
                                         storage_sample_count))
         goto out;
      const unsigned supported_buffer_binds =
         PIPE_BIND_VERTEX_BUFFER |
         PIPE_BIND_INDEX_BUFFER |
         PIPE_BIND_CONSTANT_BUFFER |
         PIPE_BIND_STREAM_OUTPUT |
         PIPE_BIND_QUERY_BUFFER |
         PIPE_BIND_COMMAND_ARGS_BUFFER;
      if (!(bind & supported_buffer_binds))
         goto out;
      if (bind & ~supported_buffer_binds)
         goto out;
      if (bind & (PIPE_BIND_STREAM_OUTPUT |
                  PIPE_BIND_QUERY_BUFFER |
                  PIPE_BIND_COMMAND_ARGS_BUFFER))
         supported = format == PIPE_FORMAT_NONE ||
                     format == PIPE_FORMAT_R8_UNORM ||
                     pvrgpu_is_supported_vertex_format(format);
      else
         supported = pvrgpu_is_supported_vertex_format(format);
      goto out;
   }

   if (!pvrgpu_is_supported_texture_target(target))
      goto out;
   if (pvrgpu_is_multisampled_request(sample_count, storage_sample_count) &&
       target != PIPE_TEXTURE_2D &&
       target != PIPE_TEXTURE_2D_ARRAY)
      goto out;
   if (format == PIPE_FORMAT_NONE) {
      if (bind & ~PIPE_BIND_RENDER_TARGET)
         goto out;
      supported = pvrgpu_texture_target_can_bind_surface(target);
      goto out;
   }
   if (pvrgpu_is_supported_color_format(format)) {
      unsigned supported_binds =
         PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_SHADER_IMAGE;
      if (pvrgpu_texture_target_can_bind_surface(target)) {
         supported_binds |=
            PIPE_BIND_RENDER_TARGET | PIPE_BIND_DISPLAY_TARGET |
            PIPE_BIND_BLENDABLE;
      }
      if (bind & ~supported_binds)
         goto out;
      supported = true;
      goto out;
   }
   if (pvrgpu_is_sampler_only_format(format)) {
      if (bind & ~(unsigned)PIPE_BIND_SAMPLER_VIEW)
         goto out;
      supported = true;
      goto out;
   }
   if (!pvrgpu_texture_target_can_bind_surface(target))
      goto out;
   if (pvrgpu_is_supported_depth_stencil_format(format)) {
      if (bind & ~(PIPE_BIND_DEPTH_STENCIL | PIPE_BIND_SAMPLER_VIEW))
         goto out;
      supported = true;
      goto out;
   }

out:
   if (pvrgpu_debug_format_support_enabled()) {
      pvrgpu_counter_eventf("format_supported",
                            "format=%s target=%u samples=%u storage=%u "
                            "bind=0x%x result=%u",
                            util_format_name(format),
                            target,
                            sample_count,
                            storage_sample_count,
                            bind,
                            supported ? 1 : 0);
   }
   return supported;
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
