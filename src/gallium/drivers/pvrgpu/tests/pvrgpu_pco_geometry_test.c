/* SPDX-License-Identifier: MIT */
#include "pvrgpu_pco.h"
#include "nir/nir.h"
#include "nir/nir_builder.h"
#include "pco/pco.h"
#include "util/ralloc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks;
static void require(bool condition, const char *message)
{
   ++checks;
   if (!condition) {
      fprintf(stderr, "pvrgpu_pco_geometry_test: FAIL: %s\n", message);
      exit(1);
   }
}

static nir_variable *variable(nir_shader *s, nir_variable_mode mode,
                              const struct glsl_type *type, const char *name,
                              unsigned location)
{
   nir_variable *v = nir_variable_create(s, mode, type, name);
   v->data.location = location;
   return v;
}

static nir_shader *make_vertex(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, pco_nir_options(), "geometry_unit_vs");
   nir_variable *in = variable(b.shader, nir_var_shader_in, glsl_vec4_type(), "position", VERT_ATTRIB_GENERIC0);
   nir_variable *out = variable(b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position", VARYING_SLOT_POS);
   nir_store_var(&b, out, nir_load_var(&b, in), 15);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *make_geometry(unsigned kind)
{
   static const unsigned primitives[] = {MESA_PRIM_POINTS, MESA_PRIM_TRIANGLES,
      MESA_PRIM_LINES_ADJACENCY, MESA_PRIM_TRIANGLES_ADJACENCY};
   static const unsigned counts[] = {1,3,4,6};
   const unsigned type = kind < 4 ? kind : 1;
   const unsigned vertices = counts[type];
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_GEOMETRY, pco_nir_options(), "geometry_unit_gs");
   b.shader->info.internal = false;
   b.shader->info.gs.input_primitive = primitives[type];
   b.shader->info.gs.output_primitive = type == 0 ? MESA_PRIM_POINTS :
      type == 2 ? MESA_PRIM_LINE_STRIP : MESA_PRIM_TRIANGLE_STRIP;
   b.shader->info.gs.vertices_in = vertices;
   b.shader->info.gs.vertices_out = vertices;
   b.shader->info.gs.invocations = kind == 4 ? 4 : 1;
   nir_variable *in = variable(b.shader, nir_var_shader_in,
      glsl_array_type(glsl_vec4_type(), vertices, 0), "gl_in_position", VARYING_SLOT_POS);
   nir_variable *out = variable(b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position", VARYING_SLOT_POS);
   nir_variable *color = variable(b.shader, nir_var_shader_out, glsl_vec4_type(), "color", VARYING_SLOT_VAR0);
   color->data.interpolation = INTERP_MODE_SMOOTH;
   nir_variable *primitive = kind == 7 ? variable(b.shader, nir_var_shader_out,
      glsl_int_type(), "gl_PrimitiveID", VARYING_SLOT_PRIMITIVE_ID) : NULL;
   nir_variable *layer = kind == 7 ? variable(b.shader, nir_var_shader_out,
      glsl_int_type(), "gl_Layer", VARYING_SLOT_LAYER) : NULL;
   for (unsigned i = 0; i < vertices; ++i) {
      nir_def *index = kind == 4 ? nir_umod_imm(&b, nir_load_invocation_id(&b), vertices) : nir_imm_int(&b, i);
      nir_def *position = nir_load_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, in), index));
      nir_def *value = position;
      if (kind == 5) {
         nir_def *scale = nir_load_uniform(&b, 1, 32, nir_imm_int(&b, 0), .base = 0, .range = 1);
         value = nir_fmul(&b, position, scale);
      }
      if (kind == 6) {
         b.shader->info.num_ubos = 1;
         value = nir_load_ubo(&b, 4, 32, nir_imm_int(&b, 0),
            nir_imul_imm(&b, nir_iand_imm(&b, nir_load_primitive_id(&b), 3), 16),
            .align_mul = 16, .align_offset = 0, .range = 64);
      }
      nir_store_var(&b, out, position, 15);
      nir_store_var(&b, color, value, 15);
      if (primitive) nir_store_var(&b, primitive, nir_load_primitive_id(&b), 1);
      if (layer) nir_store_var(&b, layer, nir_iand_imm(&b, nir_load_primitive_id(&b), 1), 1);
      nir_emit_vertex(&b, .stream_id = 0);
      if (kind == 3 && i == 2) nir_end_primitive(&b, .stream_id = 0);
   }
   nir_end_primitive(&b, .stream_id = 0);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *make_fragment(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, pco_nir_options(), "geometry_unit_fs");
   nir_variable *color = variable(b.shader, nir_var_shader_in, glsl_vec4_type(), "color", VARYING_SLOT_VAR0);
   color->data.interpolation = INTERP_MODE_SMOOTH;
   nir_variable *out = variable(b.shader, nir_var_shader_out, glsl_vec4_type(), "fragmentColor", FRAG_RESULT_DATA0);
   nir_store_var(&b, out, nir_load_var(&b, color), 15);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *make_dynamic_uniform_geometry(unsigned base, unsigned range,
                                                  unsigned components)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_GEOMETRY,
      pco_nir_options(), "geometry_dynamic_uniform_gs");
   b.shader->info.internal = false;
   b.shader->info.gs.input_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.output_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.vertices_in = b.shader->info.gs.vertices_out = 1;
   b.shader->info.gs.invocations = 1;
   nir_variable *in = variable(b.shader, nir_var_shader_in,
      glsl_array_type(glsl_vec4_type(), 1, 0), "gl_in_position", VARYING_SLOT_POS);
   nir_variable *out = variable(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "gl_Position", VARYING_SLOT_POS);
   nir_variable *color = variable(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "color", VARYING_SLOT_VAR0);
   color->data.interpolation = INTERP_MODE_SMOOTH;
   nir_def *value = nir_load_uniform(&b, components, 32,
      nir_load_primitive_id(&b), .base = base, .range = range,
      .dest_type = nir_type_float32);
   if (components == 1) value = nir_vec4(&b, value, value, value, value);
   nir_store_var(&b, out, nir_load_deref(&b, nir_build_deref_array_imm(&b,
      nir_build_deref_var(&b, in), 0)), 15);
   nir_store_var(&b, color, value, 15);
   nir_emit_vertex(&b, .stream_id = 0);
   nir_end_primitive(&b, .stream_id = 0);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *make_dynamic_uniform_fragment(unsigned base, unsigned range)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      pco_nir_options(), "geometry_dynamic_uniform_fs");
   nir_variable *color = variable(b.shader, nir_var_shader_in,
      glsl_vec4_type(), "color", VARYING_SLOT_VAR0);
   color->data.interpolation = INTERP_MODE_SMOOTH;
   nir_variable *out = variable(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "fragmentColor", FRAG_RESULT_DATA0);
   nir_def *index = nir_f2u32(&b, nir_channel(&b, nir_load_var(&b, color), 0));
   nir_store_var(&b, out, nir_load_uniform(&b, 4, 32, index,
      .base = base, .range = range, .dest_type = nir_type_float32), 15);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *make_zero_geometry(unsigned declared_vertices)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_GEOMETRY,
      pco_nir_options(), "geometry_zero_emission_gs");
   b.shader->info.internal = false;
   b.shader->info.gs.input_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.output_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.vertices_in = 1;
   b.shader->info.gs.vertices_out = declared_vertices;
   b.shader->info.gs.invocations = 1;
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *make_constant_fragment(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      pco_nir_options(), "geometry_zero_emission_fs");
   nir_variable *out = variable(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "fragmentColor", FRAG_RESULT_DATA0);
   nir_store_var(&b, out, nir_imm_vec4(&b, 0.25f, 0.5f, 0.75f, 1.0f), 15);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *make_no_output_fragment(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      pco_nir_options(), "geometry_no_output_fs");
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *make_no_attribute_vertex(bool varying)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
      pco_nir_options(), "geometry_no_attribute_vs");
   if (varying) {
      nir_variable *out = variable(b.shader, nir_var_shader_out,
         glsl_vec4_type(), "color", VARYING_SLOT_VAR0);
      nir_store_var(&b, out, nir_imm_vec4(&b, 0.25f, 0.5f, 0.75f, 1.0f), 15);
   }
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *make_generated_position_geometry(bool varying)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_GEOMETRY,
      pco_nir_options(), "geometry_generated_position_gs");
   b.shader->info.internal = false;
   b.shader->info.gs.input_primitive = b.shader->info.gs.output_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.vertices_in = b.shader->info.gs.vertices_out = 1;
   b.shader->info.gs.invocations = 1;
   nir_variable *out = variable(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "gl_Position", VARYING_SLOT_POS);
   nir_store_var(&b, out, nir_imm_vec4(&b, 0.0f, 0.0f, 0.0f, 1.0f), 15);
   if (varying) {
      nir_variable *in = variable(b.shader, nir_var_shader_in,
         glsl_array_type(glsl_vec4_type(), 1, 0), "color_in", VARYING_SLOT_VAR0);
      nir_variable *color = variable(b.shader, nir_var_shader_out,
         glsl_vec4_type(), "color", VARYING_SLOT_VAR0);
      color->data.interpolation = INTERP_MODE_SMOOTH;
      nir_store_var(&b, color, nir_load_deref(&b, nir_build_deref_array_imm(&b,
         nir_build_deref_var(&b, in), 0)), 15);
   }
   nir_emit_vertex(&b, .stream_id = 0);
   nir_end_primitive(&b, .stream_id = 0);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static void write_fixture(unsigned kind, const struct pvrgpu_pco_geometry_binary *binary)
{
   const char *directory = getenv("PVRGPU_GS_FIXTURE_DIR");
   if (!directory || !directory[0]) return;
   char path[1024];
   require(snprintf(path, sizeof(path), "%s/geometry-%u.bin", directory, kind) < (int)sizeof(path), "fixture path too long");
   FILE *f = fopen(path, "wb");
   require(f != NULL, "opening fixture output");
   require(fwrite(binary->shader.data, 1, binary->shader.size, f) == binary->shader.size, "writing fixture");
   require(fclose(f) == 0, "closing fixture");
}

int main(void)
{
   glsl_type_singleton_init_or_ref();
   char error[512] = {0};
   struct pvrgpu_pco_compiler *compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
   require(compiler != NULL, error);
   struct pvrgpu_pco_geometry_layout input = {0}, output = {0};
   input.count[VARYING_SLOT_POS] = output.count[VARYING_SLOT_POS] = 4;
   input.stride_dwords = 4;
   output.start[VARYING_SLOT_VAR0] = 4;
   output.count[VARYING_SLOT_VAR0] = 4;
   output.stride_dwords = 8;
   for (unsigned kind = 0; kind < 8; ++kind) {
      nir_shader *gs = make_geometry(kind);
      struct pvrgpu_pco_geometry_layout target = output;
      if (kind == 7) {
         target.start[VARYING_SLOT_PRIMITIVE_ID] = 8;
         target.count[VARYING_SLOT_PRIMITIVE_ID] = 1;
         target.start[VARYING_SLOT_LAYER] = 9;
         target.count[VARYING_SLOT_LAYER] = 1;
         target.stride_dwords = 10;
      }
      struct pvrgpu_pco_geometry_binary binary = {0};
      bool ok = pvrgpu_pco_compile_geometry(compiler, gs, &input, &target,
         kind == 5 ? 4 : 0, &binary, error, sizeof(error));
      if (!ok) fprintf(stderr, "standalone kind=%u\n", kind);
      require(ok, error);
      require(binary.shader.data && binary.shader.size, "native GS is empty");
      require(binary.shader.abi.vertex_inputs == 2 && binary.shader.abi.coefficients == 0,
              "Geometry uses the wrong input register banks");
      require(binary.shader.abi.uniform_buffer_descriptor_start == 4,
              "Geometry UBO descriptor origin must remain four even when empty");
      require(gs->info.stage == MESA_SHADER_GEOMETRY && !gs->info.internal,
              "Geometry compilation modified caller stage metadata");
      write_fixture(kind, &binary);
      printf("geometry fixture %u: bytes=%zu temps=%u vi=%u vo=%u sh=%u push=%u/%u ubo=%u/%u\n",
         kind,binary.shader.size,binary.shader.abi.temps,binary.shader.abi.vertex_inputs,
         binary.shader.abi.vertex_outputs,binary.shader.abi.shareds,
         binary.shader.abi.push_constant_start,binary.shader.abi.push_constant_count,
         binary.shader.abi.uniform_buffer_descriptor_start,binary.shader.abi.uniform_buffer_descriptor_count);
      pvrgpu_pco_geometry_binary_finish(&binary);
      nir_shader *vs = make_vertex();
      nir_shader *fs = make_fragment();
      const enum pipe_format format = PIPE_FORMAT_R32G32B32A32_FLOAT;
      struct pvrgpu_pco_geometry_pipeline_binary pipeline = {0};
      ok = pvrgpu_pco_compile_geometry_pipeline(compiler, vs, gs, fs, &format, 1,
         0, kind == 5 ? 4 : 0, 0, 1, 0, &pipeline, error, sizeof(error));
      if (!ok) fprintf(stderr, "pipeline kind=%u\n", kind);
      require(ok, error);
      require(pipeline.graphics.vertex.data && pipeline.geometry.shader.data && pipeline.graphics.fragment.data,
              "Geometry pipeline did not compile all three real stages");
      require(pipeline.geometry.abi.input.stride_dwords == pipeline.graphics.vertex.abi.vertex_outputs,
              "VS output and GS primitive input disagree");
      require(pipeline.graphics.varying_output_start == 4 + pipeline.graphics.point_size_output_count,
              "Geometry raster varying prefix is not contiguous");
      require(pipeline.graphics.point_size_output_count == 0 &&
              pipeline.graphics.point_size_output_start == 0,
              "absent Geometry point size must use an empty canonical range");
      pvrgpu_pco_geometry_pipeline_binary_finish(&pipeline);
      ralloc_free(vs); ralloc_free(gs); ralloc_free(fs);
   }
   for (unsigned declared = 0; declared <= 4; declared += 4) {
      nir_shader *vs = make_vertex();
      nir_shader *gs = make_zero_geometry(declared);
      nir_shader *fs = make_constant_fragment();
      const enum pipe_format format = PIPE_FORMAT_R32G32B32A32_FLOAT;
      struct pvrgpu_pco_geometry_pipeline_binary pipeline = {0};
      require(pvrgpu_pco_compile_geometry_pipeline(compiler, vs, gs, fs,
         &format, 1, 0, 0, 0, 1, 0, &pipeline, error, sizeof(error)), error);
      require(pipeline.geometry.abi.vertices_out == declared,
              "zero-emission shader maximum output metadata changed");
      require(pipeline.geometry.abi.output.stride_dwords == 4 &&
              pipeline.geometry.shader.abi.vertex_outputs == 4,
              "zero-emission Geometry lost its position ABI");
      require(pipeline.geometry.shader.data && pipeline.geometry.shader.size,
              "zero-emission Geometry skipped its native executable");
      require(pipeline.graphics.point_size_output_count == 0 &&
              pipeline.graphics.point_size_output_start == 0,
              "zero-emission Geometry has a noncanonical point size range");
      write_fixture(declared ? 11 : 10, &pipeline.geometry);
      pvrgpu_pco_geometry_pipeline_binary_finish(&pipeline);
      ralloc_free(vs); ralloc_free(gs); ralloc_free(fs);
   }
   for (unsigned kind = 0; kind < 3; ++kind) {
      nir_shader *vs = make_no_attribute_vertex(kind == 2);
      nir_shader *gs = kind ? make_generated_position_geometry(kind == 2) : make_zero_geometry(0);
      nir_shader *fs = kind == 2 ? make_fragment() : make_constant_fragment();
      struct pvrgpu_pco_geometry_pipeline_binary pipeline = {0};
      require(pvrgpu_pco_compile_geometry_pipeline(compiler, vs, gs, fs, NULL,
         1, 0, 0, 0, 0, 0, &pipeline, error, sizeof(error)), error);
      require(pipeline.graphics.vertex.data && pipeline.graphics.vertex.size &&
              pipeline.graphics.vertex.abi.vertex_inputs == 0,
              "zero-attribute VS lost its native executable or reads VTXIN");
      require(pipeline.geometry.abi.input.count[VARYING_SLOT_POS] == 4 &&
              pipeline.geometry.abi.input.stride_dwords == (kind == 2 ? 8 : 4),
              "VS without position lost the primitive input ABI");
      require(!(vs->info.outputs_written & BITFIELD64_BIT(VARYING_SLOT_POS)),
              "GS pipeline fabricated a caller VS position output");
      pvrgpu_pco_geometry_pipeline_binary_finish(&pipeline);
      ralloc_free(vs); ralloc_free(gs); ralloc_free(fs);
   }
   for (unsigned emit = 0; emit < 2; ++emit) {
      nir_shader *vs = make_no_attribute_vertex(false);
      nir_shader *gs = emit ? make_generated_position_geometry(false) : make_zero_geometry(0);
      nir_shader *fs = make_no_output_fragment();
      struct pvrgpu_pco_geometry_pipeline_binary pipeline = {0};
      require(pvrgpu_pco_compile_geometry_pipeline(compiler, vs, gs, fs, NULL,
         1, 0, 0, 0, 0, 0, &pipeline, error, sizeof(error)), error);
      require(pipeline.graphics.fragment.data && pipeline.graphics.fragment.size &&
              pipeline.graphics.fragment.abi.temps == 0,
              "output-less FS lost its real no-op native executable");
      for (unsigned target = 0; target < 8; ++target)
         require(pipeline.graphics.fragment_output_mask[target] == 0,
                 "output-less FS fabricated a color write mask");
      require(fs->info.outputs_written == 0,
              "GS pipeline compilation fabricated a caller FS output");
      require(pipeline.geometry.abi.vertices_out == emit,
              "output-less FS changed genuine GS emission metadata");
      pvrgpu_pco_geometry_pipeline_binary_finish(&pipeline);
      ralloc_free(vs); ralloc_free(gs); ralloc_free(fs);
   }
   {
      nir_shader *vs = make_vertex(), *gs = make_geometry(0), *fs = make_fragment();
      struct pvrgpu_pco_geometry_pipeline_binary pipeline = {0};
      require(!pvrgpu_pco_compile_geometry_pipeline(compiler, vs, gs, fs, NULL,
         1, 0, 0, 0, 0, 0, &pipeline, error, sizeof(error)) && strstr(error, "zero-attribute"),
         "zero-attribute VS with live generic input was not rejected");
      ralloc_free(vs); ralloc_free(gs); ralloc_free(fs);
   }
   const unsigned dynamic_config[][4] = {{0,1,1,4}, {1,3,4,16}, {0,63,4,252}};
   for (unsigned i = 0; i < ARRAY_SIZE(dynamic_config); ++i) {
      const unsigned *config = dynamic_config[i];
      nir_shader *gs = make_dynamic_uniform_geometry(config[0], config[1], config[2]);
      struct pvrgpu_pco_geometry_binary dynamic = {0};
      require(pvrgpu_pco_compile_geometry(compiler, gs, &input, &output,
         config[3], &dynamic, error, sizeof(error)), error);
      require(dynamic.shader.abi.push_constant_start == 4 &&
              dynamic.shader.abi.push_constant_count == config[3],
              "dynamic Geometry CB0 lost its bounded push suffix");
      require(dynamic.shader.data && dynamic.shader.size,
              "dynamic Geometry CB0 skipped the native executable");
      if (i == 1) write_fixture(13, &dynamic);
      pvrgpu_pco_geometry_binary_finish(&dynamic);
      ralloc_free(gs);
   }
   for (unsigned slots = 3; slots <= 16; slots += 13) {
      nir_shader *vs = make_vertex(), *gs = make_geometry(0);
      nir_shader *fs = make_dynamic_uniform_fragment(slots == 3 ? 1 : 0, slots);
      const enum pipe_format format = PIPE_FORMAT_R32G32B32A32_FLOAT;
      struct pvrgpu_pco_geometry_pipeline_binary pipeline = {0};
      require(pvrgpu_pco_compile_geometry_pipeline(compiler, vs, gs, fs, &format,
         1, 0, 0, slots == 3 ? 16 : 64, 1, 0, &pipeline, error, sizeof(error)), error);
      require(pipeline.graphics.fragment.abi.push_constant_count == (slots == 3 ? 16 : 64),
              "dynamic GS pipeline FS CB0 lost its captured range");
      pvrgpu_pco_geometry_pipeline_binary_finish(&pipeline);
      ralloc_free(vs); ralloc_free(gs); ralloc_free(fs);
   }
   for (unsigned i = 0; i < 3; ++i) {
      nir_shader *gs = make_dynamic_uniform_geometry(i == 2 ? UINT32_MAX : 1,
         i == 1 ? 0 : 3, 4);
      struct pvrgpu_pco_geometry_binary dynamic = {0};
      const bool ok = pvrgpu_pco_compile_geometry(compiler, gs, &input, &output,
         i == 0 ? 15 : 16, &dynamic, error, sizeof(error));
      const bool range_error = strstr(error, "base/range") || strstr(error, "overflow");
      if (ok || !range_error)
         fprintf(stderr, "malformed dynamic CB0 case=%u ok=%u error=%s\n", i, ok, error);
      require(!ok && range_error,
         "dynamic Geometry CB0 malformed range did not fail closed");
      require(!dynamic.shader.data && !dynamic.shader.size,
              "failed dynamic CB0 compile retained native bytes");
      ralloc_free(gs);
   }
   nir_shader *bad = make_geometry(0);
   struct pvrgpu_pco_geometry_binary binary = {0};
   bad->info.gs.invocations = 33;
   require(!pvrgpu_pco_compile_geometry(compiler,bad,&input,&output,0,&binary,error,sizeof(error)) && strstr(error,"limits"), "Geometry invocation overflow did not fail closed");
   bad->info.gs.invocations = 1;
   bad->info.gs.active_stream_mask = 2;
   require(!pvrgpu_pco_compile_geometry(compiler,bad,&input,&output,0,&binary,error,sizeof(error)) && strstr(error,"stream"), "nonzero Geometry stream did not fail closed");
   bad->info.gs.active_stream_mask = 1;
   input.stride_dwords = 65;
   require(!pvrgpu_pco_compile_geometry(compiler,bad,&input,&output,0,&binary,error,sizeof(error)) && strstr(error,"64 DWORD"), "oversized Geometry input did not fail closed");
   input.stride_dwords = 4;
   output.start[VARYING_SLOT_VAR0] = 2;
   require(!pvrgpu_pco_compile_geometry(compiler,bad,&input,&output,0,&binary,error,sizeof(error)) && strstr(error,"overlaps"), "overlapping Geometry outputs did not fail closed");
   output.start[VARYING_SLOT_VAR0] = 4;
   bad->info.num_textures = 1;
   require(!pvrgpu_pco_compile_geometry(compiler,bad,&input,&output,0,&binary,error,sizeof(error)) && strstr(error,"resources"), "unsupported Geometry texture did not fail closed");
   require(!binary.shader.data && !binary.shader.size, "failed Geometry compile retained owned bytes");
   ralloc_free(bad);
   pvrgpu_pco_compiler_destroy(compiler);
   glsl_type_singleton_decref();
   printf("pvrgpu_pco_geometry_test: PASS (%u checks)\n",checks);
   return 0;
}
