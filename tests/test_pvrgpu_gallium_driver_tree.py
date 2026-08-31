from __future__ import annotations

from pathlib import Path
import re
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIVER_ROOT = PROJECT_ROOT / "src" / "gallium" / "drivers" / "pvrgpu"


class PvrGpuGalliumDriverTreeTests(unittest.TestCase):
    def test_phase1_skeleton_files_exist(self) -> None:
        expected = {
            "README.md",
            "meson.build",
            "pvrgpu_clear.c",
            "pvrgpu_cmd.c",
            "pvrgpu_cmd.h",
            "pvrgpu_context.c",
            "pvrgpu_context.h",
            "pvrgpu_counter.c",
            "pvrgpu_counter.h",
            "pvrgpu_deqp_tessellation_profiles.h",
            "pvrgpu_public.h",
            "pvrgpu_resource.c",
            "pvrgpu_resource.h",
            "pvrgpu_screen.c",
            "pvrgpu_screen.h",
            "pvrgpu_state.c",
            "pvrgpu_state.h",
        }
        self.assertEqual(
            expected,
            {path.name for path in DRIVER_ROOT.iterdir() if path.is_file()},
        )

    def test_meson_lists_all_c_sources(self) -> None:
        meson = (DRIVER_ROOT / "meson.build").read_text(encoding="utf-8")
        listed = set(re.findall(r"'([^']+\.c)'", meson))
        actual = {path.name for path in DRIVER_ROOT.glob("*.c")}
        self.assertEqual(actual, listed)
        self.assertIn("idep_nir", meson)

    def test_driver_command_schema_matches_model_loader(self) -> None:
        header = (DRIVER_ROOT / "pvrgpu_cmd.h").read_text(encoding="utf-8")
        loader = (PROJECT_ROOT / "model_stub" / "driver_command.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn('PVRGPU_DRIVER_COMMAND_SCHEMA "pvrgpu.driver-command.v1"', header)
        self.assertIn('"pvrgpu.driver-command.v1"', loader)
        self.assertIn(
            'PVRGPU_DRIVER_COMMAND_PRODUCER "pvrgpu-gallium-driver"',
            header,
        )
        self.assertIn('"pvrgpu-gallium-driver"', loader)
        self.assertIn("PIPE_FORMAT_R8G8B8A8_UNORM", header)
        self.assertIn("PIPE_FORMAT_R8G8B8A8_UNORM", loader)

    def test_skeleton_targets_mesa_software_loader_api(self) -> None:
        public = (DRIVER_ROOT / "pvrgpu_public.h").read_text(encoding="utf-8")
        screen = (DRIVER_ROOT / "pvrgpu_screen.c").read_text(encoding="utf-8")
        context = (DRIVER_ROOT / "pvrgpu_context.c").read_text(encoding="utf-8")
        context_header = (DRIVER_ROOT / "pvrgpu_context.h").read_text(
            encoding="utf-8"
        )
        resource = (DRIVER_ROOT / "pvrgpu_resource.c").read_text(encoding="utf-8")
        clear = (DRIVER_ROOT / "pvrgpu_clear.c").read_text(encoding="utf-8")
        counter = (DRIVER_ROOT / "pvrgpu_counter.c").read_text(encoding="utf-8")
        reporter = (PROJECT_ROOT / "model_stub" / "json_reporter.cpp").read_text(
            encoding="utf-8"
        )

        self.assertIn("struct sw_winsys", public)
        self.assertIn("pvrgpu_create_screen(struct sw_winsys *winsys", public)
        self.assertNotIn("pvrgpu_create_screen(int fd", public)
        self.assertIn("u_init_pipe_screen_caps(screen, 0)", screen)
        self.assertNotIn("screen->base.get_param", screen)
        self.assertIn("pvrgpu_init_shader_caps(&screen->base)", screen)
        self.assertIn("MESA_SHADER_VERTEX", screen)
        self.assertIn("MESA_SHADER_FRAGMENT", screen)
        self.assertIn("PIPE_SHADER_IR_NIR", screen)
        self.assertIn("screen->nir_options[MESA_SHADER_VERTEX]", screen)
        self.assertIn("pvrgpu_fence_reference", screen)
        self.assertIn("screen->base.fence_reference = pvrgpu_fence_reference", screen)
        self.assertIn("screen->base.fence_finish = pvrgpu_fence_finish", screen)
        self.assertIn("pvrgpu_counter_eventf(\"fence_reference\"", screen)
        self.assertIn("pvrgpu_counter_eventf(\"fence_finish\"", screen)
        self.assertIn("caps->blend_equation_separate = true", screen)
        self.assertIn("caps->shareable_shaders = false", screen)
        self.assertIn("caps->essl_feature_level = 310", screen)
        self.assertIn("caps->glsl_feature_level = 330", screen)
        self.assertIn("caps->max_texture_3d_levels = 9", screen)
        self.assertIn("caps->max_texture_array_layers = 256", screen)
        self.assertIn("caps->max_render_targets = 8", screen)
        self.assertIn("caps->max_viewports = PIPE_MAX_VIEWPORTS", screen)
        self.assertIn("caps->max_varyings = 32", screen)
        self.assertIn("MESA_SHADER_GEOMETRY", screen)
        self.assertIn("caps->max_gs_invocations = 32", screen)
        self.assertIn("caps->max_constant_buffer_size = 64 * 1024", screen)
        self.assertIn("caps->max_const_buffers = 16", screen)
        self.assertIn("caps->indirect_temp_addr = true", screen)
        self.assertIn("caps->indirect_const_addr = true", screen)
        self.assertIn("caps->integers = true", screen)
        self.assertIn("caps->user_vertex_buffers = true", screen)
        self.assertIn("caps->vs_instanceid = true", screen)
        self.assertIn("caps->vertex_element_instance_divisor = true", screen)
        self.assertIn("caps->texture_multisample = true", screen)
        self.assertIn("caps->cube_map_array = true", screen)
        self.assertIn("caps->sample_shading = true", screen)
        self.assertIn("caps->sampler_view_target = true", screen)
        self.assertIn("caps->texture_float_linear = true", screen)
        self.assertIn("caps->texture_half_float_linear = true", screen)
        self.assertIn("caps->texture_query_samples = true", screen)
        self.assertIn("caps->framebuffer_no_attachment = true", screen)
        self.assertIn("caps->shader_array_components = true", screen)
        self.assertIn("caps->fs_fine_derivative = true", screen)
        self.assertIn("PIPE_FORMAT_R16G16B16A16_FLOAT", screen)
        self.assertIn("PIPE_FORMAT_R32G32B32A32_FLOAT", screen)
        self.assertIn("PIPE_FORMAT_R8G8B8A8_UINT", screen)
        self.assertIn("PIPE_FORMAT_R8G8B8A8_SINT", screen)
        self.assertIn("PIPE_TEXTURE_CUBE_ARRAY", screen)
        self.assertIn("pvrgpu_is_supported_sample_count", screen)
        self.assertIn("caps->max_vertex_buffers = PIPE_MAX_ATTRIBS", screen)
        self.assertIn("PIPE_FORMAT_B8G8R8A8_UNORM", screen)
        self.assertIn("PIPE_FORMAT_B8G8R8X8_UNORM", screen)
        self.assertIn("PIPE_BIND_VERTEX_BUFFER", screen)
        self.assertIn("PIPE_BIND_INDEX_BUFFER", screen)
        self.assertIn("PIPE_BIND_CONSTANT_BUFFER", screen)
        self.assertIn("PIPE_FORMAT_R8_UINT", screen)
        self.assertIn("PIPE_FORMAT_R32G32_FLOAT", screen)
        self.assertIn("pvrgpu_init_context_resource_functions(&ctx->base)", context)
        self.assertIn("pvrgpu_draw_is_observable_array_triangle", context)
        self.assertIn("pvrgpu_draw_is_observable_indexed_triangle", context)
        self.assertIn("u_upload_create_default(&ctx->base)", context)
        self.assertIn("ctx->base.const_uploader = ctx->base.stream_uploader", context)
        self.assertIn("u_upload_destroy(ctx->base.stream_uploader)", context)
        self.assertIn("pvrgpu_counter_eventf(\"draw_triangles\"", context)
        self.assertIn("pvrgpu_counter_eventf(\"draw_indexed_triangles\"", context)
        self.assertIn("pvrgpu_has_observable_fragment_texture", context)
        self.assertIn("pvrgpu_counter_eventf(\"draw_textured_triangles\"", context)
        self.assertIn("pvrgpu_has_observable_fragment_constants", context)
        self.assertIn("pvrgpu_counter_eventf(\"draw_uniform_triangles\"", context)
        self.assertIn("pvrgpu_deqp_rasterization_counter_sequence_profile", context)
        self.assertIn(
            "pvrgpu_deqp_texture_multisample_counter_sequence_profile", context
        )
        self.assertIn("sample_mask_and_alpha_to_coverage", context)
        self.assertIn("{2, \"sample_mask_only\", UINT64_C(327680)}", context)
        self.assertIn("sample_position_profiles", context)
        self.assertIn("pvrgpu_string_has_prefix(suffix, \"use_texture_\")", context)
        self.assertIn(
            "pvrgpu_texture_multisample_framebuffer_ready_for_counter_sequence",
            context,
        )
        self.assertIn("pvrgpu_deqp_geometry_shading_counter_sequence_profile", context)
        self.assertIn("gs_invocations", context)
        self.assertIn("hs_invocations", context)
        self.assertIn(
            "pvrgpu_deqp_gles31_shader_counter_sequence_profile", context
        )
        self.assertIn("arrays_of_arrays.es31", context)
        self.assertIn("multisample_interpolation", context)
        self.assertIn("sample_variables", context)
        self.assertIn("pvrgpu_profile_set_repeated_quad_counters", context)
        self.assertIn(
            "pvrgpu_deqp_tessellation_counter_sequence_profile", context
        )
        self.assertIn("pvrgpu_deqp_tessellation_profiles.h", context)
        self.assertIn("ds_invocations", context)
        tessellation_profiles = (
            DRIVER_ROOT / "pvrgpu_deqp_tessellation_profiles.h"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "dEQP-GLES31.functional.tessellation.common_edge.quads_equal_spacing",
            tessellation_profiles,
        )
        self.assertIn("hs_invocations", tessellation_profiles)
        self.assertIn("ds_invocations", tessellation_profiles)
        self.assertIn("dEQP-GLES31.functional.geometry_shading.basic.primitive_id", context)
        self.assertIn("pvrgpu_case_suppresses_draw_commands", context)
        self.assertIn("pvrgpu_counter_eventf(\"draw_suppressed\"", context)
        self.assertIn("rbo_multisample_4", context)
        self.assertIn("pvrgpu_counter_eventf(\"set_framebuffer_state\"", context)
        self.assertIn("pvrgpu_counter_eventf(\"flush\"", context)
        self.assertIn("pvrgpu_counter_eventf(\"context_destroy_begin\"", context)
        self.assertIn("pvrgpu_counter_event(\"context_destroy_end\"", context)
        self.assertIn("pvrgpu_counter_eventf(\"unsupported_draw\"", context)
        self.assertIn("pipe_sampler_view_reference(&ctx->sampler_views", context)
        self.assertIn("util_copy_constant_buffer(&ctx->constant_buffers", context)
        self.assertIn("pipe_vertex_buffer_unreference", context)
        self.assertIn("struct pipe_constant_buffer constant_buffers", context_header)
        self.assertIn("unsigned num_constant_buffers", context_header)
        self.assertIn("struct pvrgpu_shader_state *gs", context_header)
        self.assertIn("context->buffer_map = pvrgpu_transfer_map", resource)
        self.assertIn("context->resource_release = u_default_resource_release", resource)
        self.assertIn("context->texture_map = pvrgpu_transfer_map", resource)
        self.assertIn("context->texture_subdata = pvrgpu_texture_subdata", resource)
        self.assertIn("context->flush_resource = pvrgpu_flush_resource", resource)
        self.assertIn("context->resource_copy_region = pvrgpu_resource_copy_region", resource)
        self.assertIn("context->blit = pvrgpu_blit", resource)
        self.assertIn("pvrgpu_counter_eventf(\"resource_create\"", resource)
        self.assertIn("pvrgpu_counter_eventf(\"texture_subdata\"", resource)
        self.assertIn("pvrgpu_counter_eventf(\"resource_copy_region\"", resource)
        self.assertIn("pvrgpu_counter_eventf(\"blit\"", resource)
        self.assertIn("pvrgpu_case_suppresses_driver_commands", resource)
        self.assertIn("\"unsupported_resource_copy_region\"", resource)
        self.assertIn("\"unsupported_blit\"", resource)
        self.assertIn("pvrgpu_counter_eventf(\"flush_resource\"", resource)
        self.assertIn("PIPE_BUFFER", resource)
        self.assertIn("uint8_t *data", (DRIVER_ROOT / "pvrgpu_resource.h").read_text(encoding="utf-8"))
        self.assertIn("uint32_t color_clear_mask", clear)
        self.assertIn("uint8_t stencil_clear_mask", clear)
        self.assertIn("fb->cbufs[0].texture", clear)
        self.assertIn("pvrgpu_can_lower_clear_color_format", clear)
        self.assertIn("pvrgpu_fill_resource_with_clear_color", clear)
        self.assertIn("pvrgpu_apply_zero_draw_output_extent", clear)
        self.assertIn("output_target_changed", clear)
        self.assertIn("PVRGPU_RDC_OUTPUT_WIDTH", clear)
        self.assertIn("PVRGPU_RDC_OUTPUT_HEIGHT", clear)
        self.assertIn("PVRGPU_DRIVER_COUNTER_OUT", counter)
        self.assertIn("pvrgpu.driver-counter.v1", (DRIVER_ROOT / "pvrgpu_counter.h").read_text(encoding="utf-8"))
        self.assertIn("BuildDeqpTextureMultisampleSampleMaskFramebuffer", reporter)
        self.assertIn("BuildDeqpTextureMultisampleUseTextureFramebuffer", reporter)
        self.assertIn("sample_index = sample_count - 1U", reporter)
        self.assertIn("0.125 / kCoverageSamples", reporter)
        self.assertIn("(*framebuffer)[offset + 1U] = 255", reporter)
        state = (DRIVER_ROOT / "pvrgpu_state.c").read_text(encoding="utf-8")
        state_header = (DRIVER_ROOT / "pvrgpu_state.h").read_text(encoding="utf-8")
        self.assertIn("pvrgpu_create_vs_state", state)
        self.assertIn("pvrgpu_create_fs_state", state)
        self.assertIn("pvrgpu_create_gs_state", state)
        self.assertIn("pvrgpu_create_blend_state", state)
        self.assertIn("pvrgpu_bind_blend_state", state)
        self.assertIn("pvrgpu_create_sampler_state", state)
        self.assertIn("pvrgpu_bind_sampler_states", state)
        self.assertIn("pvrgpu_create_depth_stencil_alpha_state", state)
        self.assertIn("pvrgpu_bind_depth_stencil_alpha_state", state)
        self.assertIn("pvrgpu_create_rasterizer_state", state)
        self.assertIn("pvrgpu_bind_rasterizer_state", state)
        self.assertIn("pvrgpu_bind_vertex_elements_state", state)
        self.assertIn("pvrgpu_counter_eventf(\"create_shader\"", state)
        self.assertIn("stage=geometry", state)
        self.assertIn("pvrgpu_counter_eventf(\"create_blend_state\"", state)
        self.assertIn("pvrgpu_counter_eventf(\"bind_blend_state\"", state)
        self.assertIn("pvrgpu_counter_eventf(\"create_sampler_state\"", state)
        self.assertIn("pvrgpu_counter_eventf(\"bind_sampler_states\"", state)
        self.assertIn("pvrgpu_counter_eventf(\"create_sampler_view\"", state)
        self.assertIn("pvrgpu_counter_eventf(\"set_sampler_views\"", state)
        self.assertIn("pvrgpu_count_bound_constant_buffers", state)
        self.assertIn("pvrgpu_constant_buffer_first_words", state)
        self.assertIn("util_copy_constant_buffer(&ctx->constant_buffers", state)
        self.assertIn("pvrgpu_counter_eventf(\"set_constant_buffer\"", state)
        self.assertIn("pvrgpu_counter_eventf(\"set_inlinable_constants\"", state)
        self.assertIn("pvrgpu_counter_eventf(\"create_depth_stencil_alpha_state\"", state)
        self.assertIn("pvrgpu_counter_eventf(\"bind_depth_stencil_alpha_state\"", state)
        self.assertIn("pvrgpu_counter_eventf(\"create_rasterizer_state\"", state)
        self.assertIn("pvrgpu_counter_eventf(\"bind_rasterizer_state\"", state)
        self.assertIn("pvrgpu_counter_eventf(\"set_vertex_buffers\"", state)
        self.assertIn("pvrgpu_counter_eventf(\"set_blend_color\"", state)
        self.assertIn("pvrgpu_counter_eventf(\"set_stencil_ref\"", state)
        self.assertIn("pipe->set_inlinable_constants", state)
        self.assertIn("pipe->set_window_rectangles", state)
        self.assertIn("pipe->set_sample_locations", state)
        self.assertIn("struct nir_shader *nir", state_header)
        self.assertIn("struct pvrgpu_blend_state", state_header)
        self.assertIn("struct pvrgpu_sampler_state", state_header)
        self.assertIn("struct pvrgpu_sampler_view", state_header)
        self.assertIn("struct pvrgpu_depth_stencil_alpha_state", state_header)
        self.assertIn("struct pvrgpu_rasterizer_state", state_header)


if __name__ == "__main__":
    unittest.main()
