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
            "pvrgpu_pco.c",
            "pvrgpu_pco.h",
            "pvrgpu_public.h",
            "pvrgpu_resource.c",
            "pvrgpu_resource.h",
            "pvrgpu_screen.c",
            "pvrgpu_screen.h",
            "pvrgpu_state.c",
            "pvrgpu_state.h",
            "pvrgpu_systemc_api.h",
        }
        self.assertEqual(
            expected,
            {path.name for path in DRIVER_ROOT.iterdir() if path.is_file()},
        )

    def test_meson_lists_all_c_sources(self) -> None:
        meson = (DRIVER_ROOT / "meson.build").read_text(encoding="utf-8")
        listed = {
            name
            for name in re.findall(r"'([^']+\.c)'", meson)
            if "/" not in name
        }
        actual = {path.name for path in DRIVER_ROOT.glob("*.c")}
        self.assertEqual(actual, listed)
        self.assertIn("idep_nir", meson)

    def test_driver_command_schema_matches_model_loader(self) -> None:
        header = (DRIVER_ROOT / "pvrgpu_cmd.h").read_text(encoding="utf-8")
        command = (DRIVER_ROOT / "pvrgpu_cmd.c").read_text(encoding="utf-8")
        bridge = (PROJECT_ROOT / "model_stub" / "pvrgpu_systemc_bridge.cpp").read_text(
            encoding="utf-8"
        )
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
        self.assertIn("pvrgpu_systemc_api.h", command)
        self.assertIn("pvrgpu_systemc_submit_driver_command", command)
        self.assertIn("PVRGPU_SYSTEMC_API_LIB", command)
        self.assertIn("RTLD_GLOBAL", command)
        self.assertIn("std::atexit(FlushPendingSubmitAtExit)", bridge)
        self.assertIn("g_pending_submit = std::move(pending)", bridge)

    def test_indexed_quad_command_accumulates_until_lock_count(self) -> None:
        context = (DRIVER_ROOT / "pvrgpu_context.c").read_text(encoding="utf-8")
        start = context.index("pvrgpu_emit_draw_indexed_quad_command")
        end = context.index("static bool\npvrgpu_context_has_color_framebuffer", start)
        body = context[start:end]

        self.assertNotIn("ctx->driver_counter_sequence_command_emitted ||", body)
        self.assertNotIn("ctx->driver_counter_sequence_command_emitted = true;", body)
        self.assertIn("pvrgpu_case_prefers_draw_counter_sequence()", body)
        self.assertIn("pvrgpu_indexed_quad_lock_draw_count", body)
        self.assertIn("ctx->driver_indexed_quad_command_locked = true;", body)

    def test_rdc_output_extent_is_not_polluted_by_historical_framebuffers(self) -> None:
        context = (DRIVER_ROOT / "pvrgpu_context.c").read_text(encoding="utf-8")
        resource = (DRIVER_ROOT / "pvrgpu_resource.c").read_text(encoding="utf-8")

        width_start = context.index("pvrgpu_effective_framebuffer_width")
        width_end = context.index("pvrgpu_effective_framebuffer_height", width_start)
        width_body = context[width_start:width_end]
        height_start = width_end
        height_end = context.index("pvrgpu_float_bits", height_start)
        height_body = context[height_start:height_end]
        self.assertIn("ctx->framebuffer.width", width_body)
        self.assertNotIn("ctx->max_framebuffer_width", width_body)
        self.assertNotIn("pvrgpu_rdc_output_extent", width_body)
        self.assertIn("ctx->framebuffer.height", height_body)
        self.assertNotIn("ctx->max_framebuffer_height", height_body)
        self.assertNotIn("pvrgpu_rdc_output_extent", height_body)

        blit_start = resource.index("pvrgpu_emit_framebuffer_blit_command")
        blit_end = resource.index("static void\npvrgpu_blit", blit_start)
        blit_body = resource[blit_start:blit_end]
        self.assertIn(
            "pvrgpu_resource_level_width(info->dst.resource, info->dst.level)",
            blit_body,
        )
        self.assertIn("pvrgpu_destination_box_matches_rdc_output", blit_body)
        self.assertIn('"reason=destination_extent_mismatch"', blit_body)
        self.assertIn("pvrgpu_driver_draw_command_has_been_emitted()", blit_body)
        self.assertIn("ctx->driver_draw_command_emitted = true;", blit_body)
        self.assertNotIn("framebuffer_width = output_width;", blit_body)
        self.assertNotIn("framebuffer_height = output_height;", blit_body)
        self.assertNotIn("ctx->max_framebuffer_width", blit_body)
        self.assertNotIn("ctx->max_framebuffer_height", blit_body)

        match_start = resource.index(
            "pvrgpu_destination_box_matches_rdc_output"
        )
        match_end = resource.index(
            "static unsigned\npvrgpu_resource_level_layer_count", match_start
        )
        match_body = resource[match_start:match_end]
        self.assertIn(
            "level < pvrgpu_resource_level_count(resource)", match_body
        )
        self.assertIn("origin_x == 0 && origin_y == 0", match_body)
        self.assertIn("origin_z == 0", match_body)
        self.assertIn("resource->target == PIPE_TEXTURE_2D", match_body)
        self.assertIn("width == (int64_t)output_width", match_body)
        self.assertIn("height == (int64_t)output_height", match_body)
        self.assertIn("depth == 1", match_body)
        self.assertIn(
            "pvrgpu_resource_level_width(resource, level) == output_width",
            match_body,
        )
        self.assertIn(
            "pvrgpu_resource_level_height(resource, level) == output_height",
            match_body,
        )

        copy_start = resource.rindex(
            "pvrgpu_emit_resource_copy_framebuffer_blit_command"
        )
        copy_end = resource.index(
            "static void\npvrgpu_emit_framebuffer_blit_command", copy_start
        )
        copy_body = resource[copy_start:copy_end]
        self.assertIn("pvrgpu_destination_box_matches_rdc_output", copy_body)
        self.assertIn('"reason=destination_extent_mismatch"', copy_body)
        self.assertNotIn("framebuffer_width = output_width;", copy_body)
        self.assertNotIn("framebuffer_height = output_height;", copy_body)

    def test_unlowered_draw_paths_are_reported_as_unsupported(self) -> None:
        context = (DRIVER_ROOT / "pvrgpu_context.c").read_text(encoding="utf-8")
        draw_start = context.index("static void\npvrgpu_draw_vbo")
        draw_end = context.index("struct pipe_context *\npvrgpu_create_context", draw_start)
        draw_body = context[draw_start:draw_end]

        for reason in (
            "cpu_present_without_model_command",
            "textured_triangle_not_lowered",
            "uniform_triangle_not_lowered",
            "draw_triangle_command_failed",
            "indexed_quad_command_owned_elsewhere",
            "indexed_triangle_not_lowered",
            "unsupported_state",
        ):
            self.assertIn(f'"{reason}"', draw_body)

        present_start = context.index("pvrgpu_emit_present_clear_color_command")
        present_end = context.index("static unsigned\npvrgpu_min_unsigned", present_start)
        present_body = context[present_start:present_end]
        self.assertIn("ctx->driver_draw_command_emitted = true;", present_body)
        self.assertIn("pvrgpu_note_driver_draw_command_emitted();", present_body)

    def test_effect2d_textured_triangle_lowering_is_strict(self) -> None:
        context = (DRIVER_ROOT / "pvrgpu_context.c").read_text(encoding="utf-8")
        context_header = (DRIVER_ROOT / "pvrgpu_context.h").read_text(
            encoding="utf-8"
        )
        command = (DRIVER_ROOT / "pvrgpu_cmd.c").read_text(encoding="utf-8")
        command_header = (DRIVER_ROOT / "pvrgpu_cmd.h").read_text(
            encoding="utf-8"
        )
        clear = (DRIVER_ROOT / "pvrgpu_clear.c").read_text(encoding="utf-8")
        resource = (DRIVER_ROOT / "pvrgpu_resource.c").read_text(encoding="utf-8")
        systemc_api = (DRIVER_ROOT / "pvrgpu_systemc_api.h").read_text(
            encoding="utf-8"
        )

        self.assertIn("command=draw_textured_triangles", command)
        self.assertIn("texture_rgba8_path=%s", command)
        self.assertIn("PVRGPU_DRAW_TEXTURED_TRIANGLES_VERTEX_COUNT 6u", command_header)
        self.assertIn("pvrgpu_nir_matches_textured_triangles_vs", context)
        self.assertIn("pvrgpu_nir_matches_textured_triangles_fs", context)
        self.assertIn("canonical_vertex_bits", context)
        self.assertIn("draw_textured_triangles_probe_skip", context)
        self.assertIn("pvrgpu_write_texture_view_rgba8_sidecar", context)
        self.assertIn("full_depth_clear_resource", context_header)
        self.assertIn("pvrgpu_note_full_depth_clear_one", clear)
        self.assertIn("pvrgpu_invalidate_full_depth_clear_for_resource", resource)
        self.assertIn("PVRGPU_SYSTEMC_API_VERSION 8u", systemc_api)
        for field in (
            "const uint8_t *raw_vertex_data;",
            "size_t raw_vertex_data_size;",
            "const uint8_t *vertex_pco;",
            "size_t vertex_pco_size;",
            "const uint8_t *fragment_pco;",
            "size_t fragment_pco_size;",
            "const uint32_t *vertex_shared;",
            "size_t vertex_shared_count;",
            "const uint32_t *fragment_shared;",
            "size_t fragment_shared_count;",
            "struct pvrgpu_systemc_pco_stage_abi vertex_pco_abi;",
            "struct pvrgpu_systemc_pco_stage_abi fragment_pco_abi;",
            "uint32_t viewport_scale_bits[3];",
            "uint32_t depth_format;",
        ):
            self.assertIn(field, systemc_api)

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
        model = (PROJECT_ROOT / "model_stub" / "pvrgpu_model_stub.cpp").read_text(
            encoding="utf-8"
        )
        reporter = (PROJECT_ROOT / "model_stub" / "json_reporter.cpp").read_text(
            encoding="utf-8"
        )
        submitter = (PROJECT_ROOT / "model_stub" / "submitter.cpp").read_text(
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
        self.assertIn("caps->glsl_feature_level = 400", screen)
        self.assertIn("caps->glsl_feature_level_compatibility = 400", screen)
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
        self.assertIn("PIPE_FORMAT_Z24_UNORM_S8_UINT", screen)
        self.assertIn("PIPE_FORMAT_S8_UINT_Z24_UNORM", screen)
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
        self.assertIn("pvrgpu_glbench_counter_sequence_profile", context)
        self.assertIn("fill_tex_trilinear_linear_05", context)
        self.assertIn("UINT64_C(1083136)", context)
        self.assertIn("model_has_builtin_framebuffer", context)
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
        self.assertIn("context->clear_buffer = pvrgpu_clear_buffer", resource)
        self.assertIn("context->clear_texture = pvrgpu_clear_texture", resource)
        self.assertIn("context->flush_resource = pvrgpu_flush_resource", resource)
        self.assertIn("context->resource_copy_region = pvrgpu_resource_copy_region", resource)
        self.assertIn("context->blit = pvrgpu_blit", resource)
        self.assertIn("util_format_is_pure_integer(format)", resource)
        self.assertIn("PIPE_FORMAT_Z24_UNORM_S8_UINT", resource)
        self.assertIn("PIPE_FORMAT_S8_UINT_Z24_UNORM", resource)
        self.assertIn("pvrgpu_counter_eventf(\"resource_create\"", resource)
        self.assertIn("pvrgpu_counter_eventf(\"texture_subdata\"", resource)
        self.assertIn("pvrgpu_counter_eventf(\"resource_copy_region\"", resource)
        self.assertIn("pvrgpu_emit_resource_copy_framebuffer_blit_command", resource)
        self.assertIn("resource_copy_framebuffer_blit_command", resource)
        self.assertIn("pvrgpu_deqp_fbo_default_framebuffer_blit_to_default_case", resource)
        self.assertIn("pvrgpu_counter_eventf(\"blit\"", resource)
        self.assertIn("pvrgpu_case_suppresses_driver_commands", resource)
        self.assertIn("\"unsupported_resource_copy_region\"", resource)
        self.assertIn("\"unsupported_blit\"", resource)
        self.assertIn("pvrgpu_counter_eventf(\"flush_resource\"", resource)
        self.assertIn("PIPE_BUFFER", resource)
        self.assertIn("uint8_t *data", (DRIVER_ROOT / "pvrgpu_resource.h").read_text(encoding="utf-8"))
        self.assertIn("uint32_t color_clear_mask", clear)
        self.assertIn("uint8_t stencil_clear_mask", clear)
        self.assertIn("pvrgpu_clear_depth_stencil", clear)
        self.assertIn("clear_depth_stencil", context)
        self.assertIn("fb->cbufs[0].texture", clear)
        self.assertIn("pvrgpu_can_lower_clear_color_format", clear)
        self.assertIn("pvrgpu_fill_surface_rect_with_clear_color", clear)
        self.assertIn("pvrgpu_apply_zero_draw_output_extent", clear)
        self.assertIn("output_target_changed", clear)
        self.assertIn("negative_coverage.callbacks.buffer.", clear)
        self.assertIn("UINT32_C(0x3f800000)", clear)
        self.assertIn("PVRGPU_SYSTEMC_JSONL_OUT", clear)
        self.assertIn("PVRGPU_SYSTEMC_JSONL_OUT", context)
        self.assertIn("PVRGPU_SYSTEMC_JSONL_OUT", resource)
        self.assertIn("present_clear_color_command", context)
        self.assertIn("PVRGPU_RDC_OUTPUT_WIDTH", clear)
        self.assertIn("PVRGPU_RDC_OUTPUT_HEIGHT", clear)
        self.assertIn("PVRGPU_DRIVER_COUNTER_OUT", counter)
        self.assertIn("pvrgpu.driver-counter.v1", (DRIVER_ROOT / "pvrgpu_counter.h").read_text(encoding="utf-8"))
        self.assertIn("BuildDeqpTextureMultisampleSampleMaskFramebuffer", reporter)
        self.assertIn("BuildDeqpTextureMultisampleUseTextureFramebuffer", reporter)
        self.assertIn("options->test_case = IsRasterFunctionalCase(command_case)", model)
        self.assertIn("driver_counter_only_primitive_sequence", submitter)
        self.assertIn(
            'options.driver_command.command == "draw_primitive_sequence";',
            reporter,
        )
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
        self.assertIn("pvrgpu_constant_buffer_words", state)
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

    def test_public_pco_lowering_is_wired_and_fail_closed(self) -> None:
        header = (DRIVER_ROOT / "pvrgpu_pco.h").read_text(encoding="utf-8")
        source = (DRIVER_ROOT / "pvrgpu_pco.c").read_text(encoding="utf-8")
        context = (DRIVER_ROOT / "pvrgpu_context.c").read_text(encoding="utf-8")
        meson = (DRIVER_ROOT / "meson.build").read_text(encoding="utf-8")
        native_test = (
            DRIVER_ROOT / "tests" / "pvrgpu_pco_lowering_test.c"
        ).read_text(encoding="utf-8")
        outer_patch = (
            PROJECT_ROOT / "third_party" / "mesa-gallium-pvrgpu-pco.patch"
        ).read_text(encoding="utf-8")

        self.assertIn("pvrgpu_pco_compile_conditionals", header)
        self.assertIn("pvrgpu_pco_compile_lit_mesh", header)
        self.assertIn("pvrgpu_pco_graphics_binary_finish", header)
        self.assertIn("nir_shader_clone", source)
        self.assertIn("vec4_slot * 16U", source)
        self.assertIn("pco_preprocess_nir", source)
        self.assertIn("pco_link_nir", source)
        self.assertIn("pco_rev_link_nir", source)
        self.assertIn("pco_lower_nir", source)
        self.assertIn("pco_postprocess_nir", source)
        self.assertIn("pco_trans_nir", source)
        self.assertIn("pco_process_ir", source)
        self.assertIn("pco_encode_ir", source)
        self.assertIn("PIPE_FORMAT_R32G32B32_FLOAT", source)
        self.assertIn("libpowervr_compiler", meson)
        self.assertIn("libpowervr_common", meson)
        self.assertIn("pvrgpu_pco_lowering_test.c", meson)
        self.assertIn("SH0..15", native_test)
        self.assertIn("SH0..3", native_test)
        self.assertIn("binary.fragment.size != 520", native_test)
        self.assertIn("0xe33aaff7bc4d515c", native_test)
        self.assertIn("pvrgpu_lower_texture_fragment_mediump", source)
        self.assertIn("nir_f2f16_rtz", source)
        self.assertIn("nir_f2f16_rtne", source)
        self.assertIn("binary.fragment.size != 304", native_test)
        self.assertIn("0x1e2d215432179c29", native_test)
        self.assertIn("texture compile modified caller-owned NIR", native_test)
        self.assertIn("texture source-hash mismatch did not fail closed", native_test)
        self.assertIn("texture precision mismatch did not fail closed", native_test)
        self.assertIn("pvrgpu_pco_compile_refract", header)
        self.assertIn("pvrgpu_pco_build_refract_fragment_shared", header)
        self.assertIn("PVRGPU_PCO_REFRACT_FRAGMENT_SHARED_DWORDS", header)
        self.assertIn("0x26536c76cbc158b5", native_test)
        self.assertIn("refract canonical descriptor fields changed", native_test)
        self.assertIn("0x6e9ad97e49eca9fe", native_test)
        self.assertIn("Gallivm preserves both Refract vertex shaders", source)
        self.assertIn("0xa55a28d91b0f4b9e", native_test)
        self.assertIn("0xc46a9af088bfe8a9", native_test)
        self.assertIn("0x8fe8ae5903f3c2dd", native_test)
        self.assertIn("refract compile modified caller-owned NIR", native_test)
        self.assertIn("refract source-hash mismatch did not fail closed", native_test)
        self.assertIn("refract precision mismatch did not fail closed", native_test)
        self.assertIn("pvrgpu_pco_compile_shadow", header)
        self.assertIn("pvrgpu_pco_build_shadow_fragment_shared", header)
        self.assertIn("PVRGPU_PCO_SHADOW_DEPTH", header)
        self.assertIn("PVRGPU_PCO_SHADOW_MASK", header)
        self.assertIn("PVRGPU_PCO_SHADOW_SCENE", header)
        self.assertIn("0x5d306f7625b3e88e", native_test)
        self.assertIn("0x79b5f95f5c89ad6c", native_test)
        self.assertIn("0x1ac54b25af8de102", native_test)
        self.assertIn("0x385c48c6c28cd9fc", native_test)
        self.assertIn("0x24f632ab8095faeb", native_test)
        self.assertIn("shadow compile modified caller-owned NIR", native_test)
        self.assertIn("shadow source-hash mismatch did not fail closed", native_test)
        self.assertIn("shadow precision mismatch did not fail closed", native_test)
        shadow_fb_start = context.index("pvrgpu_shadow_pco_framebuffer_matches")
        shadow_fb_end = context.index(
            "static bool\npvrgpu_shadow_pco_pipeline_matches", shadow_fb_start
        )
        shadow_fb = context[shadow_fb_start:shadow_fb_end]
        depth_backing = shadow_fb.index(
            "pvrgpu_refract_pco_depth_backing_is_known_clear"
        )
        color_backing = shadow_fb.index("pvrgpu_rgba8_surface_backing_matches")
        self.assertLess(depth_backing, color_backing)
        self.assertRegex(
            shadow_fb,
            r"observation->output_depth_clear_one\s*&&\s*"
            r"!pvrgpu_rgba8_surface_backing_matches",
        )
        self.assertIn("draw_pco_shadow_warmup_mask_skip", context)
        self.assertIn("draw_pco_shadow_warmup_scene_skip", context)
        self.assertIn("pvrgpu_pco_compile_terrain", header)
        for profile in range(1, 9):
            self.assertIn(f"PVRGPU_PCO_TERRAIN_D{profile}", header)
        self.assertIn("pvrgpu_validate_terrain_nir", source)
        self.assertIn("pvrgpu_pack_terrain_texture_bindings", source)
        self.assertIn("0x081618f544cc6abe", native_test)
        self.assertIn("0x9abe96cad5fe9f4e", native_test)
        self.assertIn("0x9e1c3ea2dfa1d8a5", native_test)
        self.assertIn("0x9711b79a7b5b63a6", native_test)
        self.assertIn("0x412b9e844c3de073", native_test)
        self.assertIn("0x8d0f6d4b38cdecf4", native_test)
        self.assertIn("0x4fecdd1ce1feb997", native_test)
        self.assertIn("0x76fac56a9fbc5918", native_test)
        self.assertIn("0x956d5ea59737b66f", native_test)
        self.assertIn("0xab0dfc14e6aa5116", native_test)
        self.assertIn("0xd0b9eb8de7e641d2", native_test)
        self.assertIn("0x1d6737c7f69c0953", native_test)
        self.assertIn("0xb41e711d1ef41b5a", native_test)
        self.assertIn("terrain compile modified caller-owned NIR", native_test)
        self.assertIn("terrain D1 compile modified caller-owned NIR", native_test)
        self.assertIn("terrain blur compile modified caller-owned NIR", native_test)
        self.assertIn("terrain D3 compile modified caller-owned NIR", native_test)
        self.assertIn("terrain source-hash mismatch did not fail closed", native_test)
        self.assertIn("terrain precision mismatch did not fail closed", native_test)
        self.assertIn("0xc33cf9ea6c986551", context)
        self.assertIn("0x798ce5dd9c33fa18", context)
        self.assertIn("0x1369112ad898bbfd", context)
        for fingerprint in (
            "0xa69ccd9838551cb3",
            "0x777443d6a3c0ceeb",
            "0xd510ff3e570680dd",
            "0x3964257e9bde4861",
        ):
            self.assertIn(fingerprint, context)
        self.assertNotIn("0x7292c85e49f17beb", context)
        self.assertIn("draw_pco_terrain_texture_fingerprint_mismatch", context)
        self.assertIn("draw_pco_terrain_sequence_ready", context)
        self.assertIn("pvrgpu_emit_terrain_pco_sequence_command(ctx)", context)
        self.assertIn("draw_pco_terrain_sequence_command", context)
        self.assertNotIn("blocked_api_v7", context)
        self.assertIn("draw_pco_terrain_warmup_sequence_skip", context)
        self.assertIn("draw_pco_terrain_warmup_draw_skip", context)
        self.assertIn(
            "command->depth_clear_bits = observation->depth_clear ?",
            context,
        )
        self.assertRegex(
            context,
            r"terrain_pass == 0\s*&&\s*"
            r"terrain_profile == PVRGPU_PCO_TERRAIN_D4\s*&&\s*"
            r"ctx->terrain_pco_draw_count == PVRGPU_PCO_TERRAIN_D4\s*&&\s*"
            r"terrain_failure_reason\s*&&\s*"
            r"strcmp\(terrain_failure_reason,\s*"
            r'"texture_attachment_source"\) == 0',
        )
        self.assertIn("PVRGPU_TERRAIN_PCO_PREVIOUS_COLOR_ATTACHMENT", context)
        terrain_main_specs = re.search(
            r"if \(profile == PVRGPU_PCO_TERRAIN_D3\) \{"
            r".*?const struct pvrgpu_terrain_pco_texture_spec main_specs\[\] = \{"
            r"(.*?)\n\s*\};",
            context,
            re.DOTALL,
        )
        self.assertIsNotNone(terrain_main_specs)
        main_specs = terrain_main_specs.group(1)
        self.assertRegex(
            main_specs,
            r"MESA_SHADER_VERTEX,\s*1,\s*0,\s*"
            r"PVRGPU_TERRAIN_PCO_PREVIOUS_COLOR_ATTACHMENT,\s*"
            r"PVRGPU_PCO_TERRAIN_D1",
        )
        self.assertRegex(
            main_specs,
            r"MESA_SHADER_VERTEX,\s*0,\s*1,\s*"
            r"PVRGPU_TERRAIN_PCO_PREVIOUS_COLOR_ATTACHMENT,\s*"
            r"PVRGPU_PCO_TERRAIN_D2",
        )
        self.assertRegex(
            main_specs,
            r"MESA_SHADER_FRAGMENT,\s*4,\s*3,\s*"
            r"PVRGPU_TERRAIN_PCO_PREVIOUS_COLOR_ATTACHMENT,\s*"
            r"PVRGPU_PCO_TERRAIN_D1",
        )
        self.assertIn("main ? 18U", source)
        self.assertEqual(outer_patch.count("with_gallium_pvrgpu"), 2)

    def test_lit_mesh_pco_profiles_use_exact_source_hashes(self) -> None:
        header = (DRIVER_ROOT / "pvrgpu_pco.h").read_text(encoding="utf-8")
        source = (DRIVER_ROOT / "pvrgpu_pco.c").read_text(encoding="utf-8")
        context = (DRIVER_ROOT / "pvrgpu_context.c").read_text(encoding="utf-8")

        def named_hash(name: str) -> tuple[str, ...]:
            match = re.search(
                rf"static const uint32_t {re.escape(name)}\[8\]\s*=\s*"
                r"\{(.*?)\};",
                context,
                re.DOTALL,
            )
            self.assertIsNotNone(match, f"missing context hash {name}")
            words = tuple(
                re.findall(r"UINT32_C\(0x([0-9a-fA-F]{8})\)", match.group(1))
            )
            self.assertEqual(8, len(words), f"incomplete context hash {name}")
            return words

        expected_hashes = {
            "build_vs": (
                "750ac3d1", "e9ceafcc", "dd1263dd", "a22a457b",
                "3b8ebb47", "a4ee0e8e", "eb2663ea", "6ad452cd",
            ),
            "bump_vs": (
                "447e9e1f", "b6e0a9b9", "f5dcefa9", "f987adef",
                "8c416544", "b0956e81", "c5d8865b", "7b2850a7",
            ),
            "shading_vs": (
                "55a3db4e", "0781726a", "a9aaf326", "1663be77",
                "0b6194eb", "dd4b6265", "3351e890", "7acfdd9a",
            ),
            "passthrough_fs": (
                "8105bebf", "60cef3c7", "c9c3e978", "d20442bc",
                "46d83156", "9a4abb0b", "d1a4de24", "422a9790",
            ),
            "bump_fs": (
                "4f55ff2c", "8d248356", "20aea0e2", "ee5248d5",
                "777abff2", "c13daa4d", "cb78bfc9", "c09ce498",
            ),
        }
        for hash_name, expected in expected_hashes.items():
            self.assertEqual(expected, named_hash(hash_name), hash_name)

        profiles = (
            (
                "PVRGPU_PCO_LIT_MESH_BUILD",
                "build",
                "build_vs",
                "passthrough_fs",
                1,
                21516,
            ),
            (
                "PVRGPU_PCO_LIT_MESH_BUMP",
                "bump",
                "bump_vs",
                "bump_fs",
                3,
                1440,
            ),
            (
                "PVRGPU_PCO_LIT_MESH_SHADING",
                "shading",
                "shading_vs",
                "passthrough_fs",
                3,
                43044,
            ),
        )
        profile_body_start = context.index("pvrgpu_lit_mesh_profile(")
        profile_body_end = context.index(
            "pvrgpu_lit_mesh_copy_constants", profile_body_start
        )
        profile_body = context[profile_body_start:profile_body_end]

        for index, (
            enum_name,
            profile_name,
            vertex_hash_name,
            fragment_hash_name,
            varying_components,
            vertex_count,
        ) in enumerate(profiles):
            with self.subTest(profile=profile_name):
                self.assertIn(enum_name, header)
                start = source.index(f"[{enum_name}] = {{")
                if index + 1 < len(profiles):
                    end = source.index(f"[{profiles[index + 1][0]}] = {{", start)
                else:
                    end = source.index("\n};", start)
                descriptor = source[start:end]
                self.assertIn(f'.name = "{profile_name}"', descriptor)
                self.assertIn(
                    f".varying_components = {varying_components}", descriptor
                )

                for field, expected in (
                    ("vertex_source_hash", named_hash(vertex_hash_name)),
                    ("fragment_source_hash", named_hash(fragment_hash_name)),
                ):
                    match = re.search(
                        rf"\.{field}\s*=\s*\{{(.*?)\}},",
                        descriptor,
                        re.DOTALL,
                    )
                    self.assertIsNotNone(match, f"missing {profile_name} {field}")
                    actual = tuple(
                        re.findall(
                            r"UINT32_C\(0x([0-9a-fA-F]{8})\)", match.group(1)
                        )
                    )
                    self.assertEqual(expected, actual)

                branch = re.search(
                    rf"pvrgpu_nir_source_hash_matches\(ctx->vs->nir, "
                    rf"{vertex_hash_name}\).*?"
                    rf"pvrgpu_nir_source_hash_matches\(ctx->fs->nir, "
                    rf"{fragment_hash_name}\).*?"
                    rf"\*profile = {enum_name};.*?"
                    rf"\*vertex_count = {vertex_count};",
                    profile_body,
                    re.DOTALL,
                )
                self.assertIsNotNone(branch, f"incomplete {profile_name} matcher")

        self.assertIn("shader->info.source_blake3", source)
        self.assertIn("shader->info.source_blake3", context)
        compile_start = source.index("bool pvrgpu_pco_compile_lit_mesh(")
        compile_body = source[compile_start:]
        self.assertLess(
            compile_body.index("pvrgpu_source_hash_matches"),
            compile_body.index("nir_shader_clone"),
        )
        self.assertLess(
            compile_body.index("pvrgpu_validate_lit_mesh_variables"),
            compile_body.index("nir_shader_clone"),
        )

    def test_lit_mesh_uses_generic_api_v5_command_without_cpu_shortcut(self) -> None:
        context = (DRIVER_ROOT / "pvrgpu_context.c").read_text(encoding="utf-8")
        command = (DRIVER_ROOT / "pvrgpu_cmd.c").read_text(encoding="utf-8")
        pco = (DRIVER_ROOT / "pvrgpu_pco.c").read_text(encoding="utf-8")
        systemc_api = (DRIVER_ROOT / "pvrgpu_systemc_api.h").read_text(
            encoding="utf-8"
        )

        self.assertIn("PVRGPU_SYSTEMC_API_VERSION 8u", systemc_api)
        for field in (
            "uint32_t vertex_stride;",
            "uint32_t position_output_start;",
            "uint32_t position_output_count;",
            "uint32_t fragment_position_start;",
            "uint32_t fragment_position_count;",
            "uint32_t varying_output_start;",
            "uint32_t varying_output_count;",
            "uint32_t fragment_varying_start;",
            "uint32_t fragment_varying_count;",
        ):
            self.assertIn(field, systemc_api)

        validator_start = command.index("pvrgpu_cmd_validate_draw_pco_triangles(")
        validator_end = command.index(
            "pvrgpu_cmd_validate_draw_primitive_sequence", validator_start
        )
        validator = command[validator_start:validator_end]
        self.assertIn(
            "cmd->vertex_stride == 24 && cmd->vertex_pco_abi.vertex_inputs == 8",
            validator,
        )
        self.assertIn(
            "cmd->raw_vertex_data_size != (size_t)expected_vertex_bytes",
            validator,
        )
        self.assertIn("cmd->varying_output_count > 4", validator)
        self.assertIn("cmd->fragment_position_count != 4", validator)
        self.assertIn(
            "cmd->fragment_varying_count != cmd->varying_output_count * 4",
            validator,
        )
        self.assertNotIn("PVRGPU_DRAW_PCO_TRIANGLES_VS_PCO_BYTES", validator)
        self.assertNotIn("PVRGPU_DRAW_PCO_TRIANGLES_FS_PCO_BYTES", validator)

        writer_start = command.index("pvrgpu_write_draw_pco_triangles_command(")
        writer_end = command.index(
            "pvrgpu_write_draw_triangle_command(", writer_start
        )
        writer = command[writer_start:writer_end]
        self.assertIn("command=draw_pco_triangles", writer)
        self.assertIn("varying_linkage=%u,%u,%u,%u", writer)
        for field in (
            "raw_vertex_data",
            "raw_vertex_data_size",
            "vertex_stride",
            "vertex_count",
            "first_vertex",
            "instance_count",
            "primitive_mode",
            "indexed",
            "vertex_pco",
            "vertex_pco_size",
            "fragment_pco",
            "fragment_pco_size",
            "vertex_shared",
            "vertex_shared_count",
            "fragment_shared",
            "fragment_shared_count",
            "position_output_start",
            "position_output_count",
            "fragment_position_start",
            "fragment_position_count",
            "varying_output_start",
            "varying_output_count",
            "fragment_varying_start",
            "fragment_varying_count",
        ):
            self.assertIn(f"api_command.{field} = cmd->{field};", writer)
        self.assertIn("pvrgpu_submit_systemc_api(&api_command", writer)

        emit_start = context.index("pvrgpu_emit_lit_mesh_command(")
        emit_end = context.index(
            "pvrgpu_write_texture_view_rgba8_sidecar", emit_start
        )
        emitter = context[emit_start:emit_end]
        self.assertIn("pvrgpu_pco_compile_lit_mesh", emitter)
        self.assertIn("command.vertex_stride = 6u * sizeof(float);", emitter)
        self.assertIn("pvrgpu_write_draw_pco_triangles_command", emitter)
        for field in (
            "position_output_start",
            "position_output_count",
            "fragment_position_start",
            "fragment_position_count",
            "varying_output_start",
            "varying_output_count",
            "fragment_varying_start",
            "fragment_varying_count",
        ):
            self.assertIn(f"command.{field} = binary.{field};", emitter)
            self.assertIn(f"out->{field}", pco)
        for shortcut in (
            "pvrgpu_cpu_",
            "framebuffer_rgba8_path",
            "pvrgpu_write_texture_view_rgba8_sidecar",
        ):
            self.assertNotIn(shortcut, emitter)

        route_start = context.index("struct pvrgpu_lit_mesh_observation lit_mesh;")
        route_end = context.index(
            "struct pvrgpu_conditionals_observation conditionals;", route_start
        )
        route = context[route_start:route_end]
        self.assertIn("draw_pco_lit_mesh_probe_skip", route)
        self.assertIn("pvrgpu_emit_lit_mesh_command(ctx, &lit_mesh)", route)
        self.assertIn('"lit_mesh_command_failed"', route)
        self.assertIn("requested_glmark_extent", route)
        self.assertIn("lit_mesh.framebuffer_width == requested_width", route)
        self.assertIn("lit_mesh.framebuffer_height == requested_height", route)
        self.assertIn("lit_mesh.viewport_width == requested_width", route)
        self.assertIn("lit_mesh.viewport_height == requested_height", route)
        self.assertNotIn("pvrgpu_cpu_", route)

    def test_conditionals_pco_command_is_strict_and_api_v5_backed(self) -> None:
        context = (DRIVER_ROOT / "pvrgpu_context.c").read_text(encoding="utf-8")
        command = (DRIVER_ROOT / "pvrgpu_cmd.c").read_text(encoding="utf-8")
        command_header = (DRIVER_ROOT / "pvrgpu_cmd.h").read_text(
            encoding="utf-8"
        )
        systemc_api = (DRIVER_ROOT / "pvrgpu_systemc_api.h").read_text(
            encoding="utf-8"
        )

        self.assertIn("PVRGPU_SYSTEMC_API_VERSION 8u", systemc_api)
        self.assertIn("command=draw_pco_triangles", command)
        self.assertIn("pvrgpu_write_draw_pco_triangles_command", command_header)
        self.assertIn("PVRGPU_DRAW_PCO_TRIANGLES_VERTEX_COUNT 6144u", command_header)
        self.assertIn("PVRGPU_DRAW_PCO_TRIANGLES_VS_SHARED_DWORDS 16u", command_header)
        self.assertIn("PVRGPU_DRAW_PCO_TRIANGLES_FS_SHARED_DWORDS 4u", command_header)
        self.assertIn("PVRGPU_DRAW_PCO_TRIANGLES_VS_TEMPS 10u", command_header)
        self.assertIn("PVRGPU_DRAW_PCO_TRIANGLES_VS_INPUTS 4u", command_header)
        self.assertIn("PVRGPU_DRAW_PCO_TRIANGLES_VS_OUTPUTS 4u", command_header)
        self.assertIn("PVRGPU_DRAW_PCO_TRIANGLES_FS_TEMPS 4u", command_header)
        self.assertIn("PVRGPU_DRAW_PCO_TRIANGLES_VS_PCO_BYTES 520u", command_header)
        self.assertIn("PVRGPU_DRAW_PCO_TRIANGLES_FS_PCO_BYTES 520u", command_header)
        self.assertIn("cmd->vertex_pco_abi.coefficients != 0", command)
        self.assertIn("cmd->fragment_pco_abi.coefficients !=", command)
        self.assertIn(
            "cmd->fragment_position_count + cmd->fragment_varying_count",
            command,
        )
        self.assertIn("cmd->varying_output_count != 0", command)
        self.assertIn("cmd->fragment_position_count != 0", command)
        self.assertIn("cmd->fragment_varying_count != 0", command)
        self.assertIn("cmd->vertex_pco_abi.entry_offset != 0", command)
        self.assertIn("cmd->fragment_pco_abi.entry_offset != 0", command)
        self.assertIn("pvrgpu_nir_matches_conditionals_vs", context)
        self.assertIn("pvrgpu_nir_matches_conditionals_fs", context)
        self.assertIn("0x5a33231c", context)
        self.assertIn("0x7f3de1f2", context)
        self.assertIn(
            "position_output->data.precision != GLSL_PRECISION_MEDIUM",
            context,
        )
        self.assertIn(
            "color_output->data.precision != GLSL_PRECISION_MEDIUM",
            context,
        )
        self.assertIn("pvrgpu_conditionals_vertex_data_matches", context)
        self.assertIn("resource->size != PVRGPU_DRAW_PCO_TRIANGLES_VERTEX_BYTES", context)
        self.assertIn("draw_pco_triangles_probe_skip", context)
        self.assertIn("pvrgpu_pco_compile_conditionals", context)
        self.assertIn("pvrgpu_pco_graphics_binary_finish", context)


if __name__ == "__main__":
    unittest.main()
