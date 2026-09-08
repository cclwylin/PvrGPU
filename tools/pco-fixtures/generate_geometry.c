/* SPDX-License-Identifier: MIT
 * Native Geometry shader fixture generator for pinned Mesa 26.2.1 PCO,
 * commit da14d65e4499e66468094be52bff9ea0915a695e, public target gx6250.
 * Requires third_party/mesa-26.2.1-pco-geometry-stage.patch and links the
 * real pvrgpu_pco.c compiler helper plus Mesa PCO/NIR static libraries.
 * Arguments: KIND OUTPUT.bin. Kinds 0..7 share the compiler unit's real
 * shader builders; kinds 8/9 below are runtime-bound loops, 10/11 are
 * zero-emission shaders with declared maximum 0/4. Kind 12 writes a real
 * points pipeline to OUTPUT.bin.vs.bin/.gs.bin/.fs.bin and prints its ABI.
 * Kind 13 dynamically selects CB0 vec4 slots [1..3] with PrimitiveIDIn.
 * Kind 14 is a zero-emission pipeline; 15/16 are zero-attribute pipelines
 * with no-op VS / a VS writing only a generic varying (never gl_Position).
 * Kind 17 combines the no-op zero-attribute VS and zero-emission GS.
 * Kinds 18/19 use output-less native FS with GS emitting zero/one vertex.
 * No instruction
 * bytes or shader execution results are constructed by this generator.
 */
int original_geometry_unit_main(void);
#define main original_geometry_unit_main
#include "../../src/gallium/drivers/pvrgpu/tests/pvrgpu_pco_geometry_test.c"
#undef main

/* Input: three vec4 positions via descriptor SHARED0..3; primitive ID in
 * VTXIN0, invocation ID in VTXIN1; SHARED4 is a uint32 loop bound. Emit
 * min(bound,6) vertices, selecting position[i%3] and offsetting x by
 * float(primitiveID+invocationID)*0.125. Varying is vec4(i,primitiveID,
 * invocationID,1). CUT every third vertex; ENDTASK flushes a partial strip.
 * All indexing, arithmetic, branch/loop control and emit/cut execute PCO. */
static nir_shader *make_dynamic_loop(bool retain_backedge)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_GEOMETRY,
      pco_nir_options(), retain_backedge ? "geometry_native_backedge_probe" : "geometry_dynamic_loop_probe");
   b.shader->info.internal = false;
   b.shader->info.gs.input_primitive = MESA_PRIM_TRIANGLES;
   b.shader->info.gs.output_primitive = MESA_PRIM_TRIANGLE_STRIP;
   b.shader->info.gs.vertices_in = 3;
   b.shader->info.gs.vertices_out = 6;
   b.shader->info.gs.invocations = 4;
   nir_variable *input = variable(b.shader, nir_var_shader_in,
      glsl_array_type(glsl_vec4_type(), 3, 0), "gl_in_position", VARYING_SLOT_POS);
   nir_variable *output = variable(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "gl_Position", VARYING_SLOT_POS);
   nir_variable *color = variable(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "varying", VARYING_SLOT_VAR0);
   color->data.interpolation = INTERP_MODE_FLAT;
   nir_variable *counter = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_store_var(&b, counter, nir_imm_int(&b, 0), 1);
   nir_def *limit = nir_load_uniform(&b, 1, 32, nir_imm_int(&b, 0), .base = 0, .range = 1);
   if (!retain_backedge) limit = nir_umin(&b, limit, nir_imm_int(&b, 6));
   nir_def *primitive = nir_load_primitive_id(&b);
   nir_def *invocation = nir_load_invocation_id(&b);
   nir_loop *loop = nir_push_loop(&b);
   if (retain_backedge) loop->control = nir_loop_control_dont_unroll;
   nir_def *i = nir_load_var(&b, counter);
   nir_if *stop = nir_push_if(&b, nir_uge(&b, i, limit));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, stop);
   nir_def *index = nir_umod_imm(&b, i, 3);
   nir_def *position = nir_load_deref(&b, nir_build_deref_array(&b,
      nir_build_deref_var(&b, input), index));
   nir_def *offset = nir_fmul_imm(&b,
      nir_u2f32(&b, nir_iadd(&b, primitive, invocation)), 0.125f);
   nir_def *adjusted = nir_vec4(&b,
      nir_fadd(&b, nir_channel(&b, position, 0), offset),
      nir_channel(&b, position, 1), nir_channel(&b, position, 2), nir_channel(&b, position, 3));
   nir_store_var(&b, output, adjusted, 15);
   nir_store_var(&b, color, nir_vec4(&b, nir_u2f32(&b, i),
      nir_u2f32(&b, primitive), nir_u2f32(&b, invocation), nir_imm_float(&b, 1)), 15);
   nir_emit_vertex(&b, .stream_id = 0);
   nir_if *cut = nir_push_if(&b, nir_ieq_imm(&b, index, 2));
   nir_end_primitive(&b, .stream_id = 0);
   nir_pop_if(&b, cut);
   nir_store_var(&b, counter, nir_iadd_imm(&b, i, 1), 1);
   nir_pop_loop(&b, loop);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static void save_pipeline_stage(const char *prefix, const char *stage,
                               const struct pvrgpu_pco_owned_binary *binary)
{
   char path[1024];
   require(snprintf(path, sizeof(path), "%s.%s.bin", prefix, stage) <
           (int)sizeof(path), "pipeline fixture path too long");
   FILE *f = fopen(path, "wb");
   require(f != NULL, "opening pipeline stage output");
   require(fwrite(binary->data, 1, binary->size, f) == binary->size,
           "writing pipeline stage output");
   require(fclose(f) == 0, "closing pipeline stage output");
   const struct pvrgpu_pco_stage_abi *a = &binary->abi;
   fprintf(stderr, "%s bytes=%zu temps=%u vi=%u vo=%u coefficients=%u shareds=%u push=%u/%u ubo=%u/%u entry=%u\n",
      stage,binary->size,a->temps,a->vertex_inputs,a->vertex_outputs,a->coefficients,a->shareds,
      a->push_constant_start,a->push_constant_count,a->uniform_buffer_descriptor_start,
      a->uniform_buffer_descriptor_count,a->entry_offset);
}

int main(int argc, char **argv)
{
   if (argc != 3) return 2;
   char *end = NULL;
   unsigned long parsed = strtoul(argv[1], &end, 10);
   if (!end || *end || parsed > 19) return 2;
   unsigned kind = parsed;
   glsl_type_singleton_init_or_ref();
   char error[512] = {0};
   struct pvrgpu_pco_compiler *compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
   if (!compiler) { fprintf(stderr, "%s\n", error); return 1; }
   if (kind == 12 || kind >= 14) {
      nir_shader *vs = kind >= 15 ? make_no_attribute_vertex(kind == 16) : make_vertex();
      nir_shader *gs = kind == 14 || kind == 17 || kind == 18 ? make_zero_geometry(0) : kind >= 15 ?
         make_generated_position_geometry(kind == 16) : make_geometry(0);
      nir_shader *fs = kind >= 18 ? make_no_output_fragment() :
         kind == 14 || kind == 15 || kind == 17 ? make_constant_fragment() : make_fragment();
      const enum pipe_format format = PIPE_FORMAT_R32G32B32A32_FLOAT;
      struct pvrgpu_pco_geometry_pipeline_binary pipeline = {0};
      require(pvrgpu_pco_compile_geometry_pipeline(compiler, vs, gs, fs, &format,
         1, 0, 0, 0, kind >= 15 ? 0 : 1, 0, &pipeline, error, sizeof(error)), error);
      save_pipeline_stage(argv[2], "vs", &pipeline.graphics.vertex);
      save_pipeline_stage(argv[2], "gs", &pipeline.geometry.shader);
      save_pipeline_stage(argv[2], "fs", &pipeline.graphics.fragment);
      fprintf(stderr, "pipeline input_primitive=%u input_vertices=%u input_stride=%u output_primitive=%u max_vertices=%u invocations=%u position=%u/%u point_size=%u/%u varying=%u/%u fragment_position=%u/%u fragment_varying=%u/%u\n",
         pipeline.geometry.abi.input_primitive,pipeline.geometry.abi.vertices_in,
         pipeline.geometry.abi.input.stride_dwords,pipeline.geometry.abi.output_primitive,
         pipeline.geometry.abi.vertices_out,pipeline.geometry.abi.invocations,
         pipeline.graphics.position_output_start,pipeline.graphics.position_output_count,
         pipeline.graphics.point_size_output_start,pipeline.graphics.point_size_output_count,
         pipeline.graphics.varying_output_start,pipeline.graphics.varying_output_count,
         pipeline.graphics.fragment_position_start,pipeline.graphics.fragment_position_count,
         pipeline.graphics.fragment_varying_start,pipeline.graphics.fragment_varying_count);
      pvrgpu_pco_geometry_pipeline_binary_finish(&pipeline);
      ralloc_free(vs); ralloc_free(gs); ralloc_free(fs);
      pvrgpu_pco_compiler_destroy(compiler);
      glsl_type_singleton_decref();
      return 0;
   }
   nir_shader *shader = kind == 13 ? make_dynamic_uniform_geometry(1, 3, 4) :
      kind >= 10 ? make_zero_geometry(kind == 10 ? 0 : 4) :
      kind >= 8 ? make_dynamic_loop(kind == 9) : make_geometry(kind);
   struct pvrgpu_pco_geometry_layout input = {0}, output = {0};
   input.count[VARYING_SLOT_POS] = output.count[VARYING_SLOT_POS] = 4;
   input.stride_dwords = 4;
   output.start[VARYING_SLOT_VAR0] = 4;
   output.count[VARYING_SLOT_VAR0] = 4;
   output.stride_dwords = 8;
   if (kind == 10 || kind == 11) {
      output.count[VARYING_SLOT_VAR0] = 0;
      output.stride_dwords = 4;
   }
   if (kind == 7) {
      output.start[VARYING_SLOT_PRIMITIVE_ID] = 8;
      output.count[VARYING_SLOT_PRIMITIVE_ID] = 1;
      output.start[VARYING_SLOT_LAYER] = 9;
      output.count[VARYING_SLOT_LAYER] = 1;
      output.stride_dwords = 10;
   }
   struct pvrgpu_pco_geometry_binary binary = {0};
   if (!pvrgpu_pco_compile_geometry(compiler, shader, &input, &output,
      kind == 13 ? 16 : kind == 5 || kind == 8 || kind == 9 ? 4 : 0,
      &binary, error, sizeof(error))) {
      fprintf(stderr, "%s\n", error); return 1;
   }
   FILE *f = fopen(argv[2], "wb");
   if (!f || fwrite(binary.shader.data, 1, binary.shader.size, f) != binary.shader.size || fclose(f)) return 1;
   const struct pvrgpu_pco_stage_abi *a = &binary.shader.abi;
   fprintf(stderr, "kind=%u bytes=%zu temps=%u vi=%u vo=%u coefficients=%u shareds=%u push=%u/%u ubo=%u/%u entry=%u\n",
      kind,binary.shader.size,a->temps,a->vertex_inputs,a->vertex_outputs,a->coefficients,a->shareds,
      a->push_constant_start,a->push_constant_count,a->uniform_buffer_descriptor_start,
      a->uniform_buffer_descriptor_count,a->entry_offset);
   pvrgpu_pco_geometry_binary_finish(&binary);
   ralloc_free(shader);
   pvrgpu_pco_compiler_destroy(compiler);
   glsl_type_singleton_decref();
   return 0;
}
