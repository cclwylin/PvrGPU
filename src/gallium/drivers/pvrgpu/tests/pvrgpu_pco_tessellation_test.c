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
static void require(bool value, const char *message)
{
   ++checks;
   if (!value) { fprintf(stderr, "tessellation compiler FAIL: %s\n", message); exit(1); }
}
static nir_variable *variable(nir_shader *s, nir_variable_mode mode,
                              const struct glsl_type *type, const char *name, unsigned location)
{
   nir_variable *var = nir_variable_create(s, mode, type, name);
   var->data.location = location;
   return var;
}
static nir_shader *make_vertex(bool varying)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, pco_nir_options(), "tess_fixture_vs");
   if (varying) {
      nir_variable *in = variable(b.shader, nir_var_shader_in, glsl_vec4_type(), "position", VERT_ATTRIB_GENERIC0);
      nir_variable *out = variable(b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position", VARYING_SLOT_POS);
      nir_store_var(&b, out, nir_load_var(&b, in), 15);
   }
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}
static nir_def *sample_texture(nir_builder *b, nir_def *coord, nir_def *lod)
{
   nir_variable *sampler = nir_variable_create(b->shader, nir_var_uniform,
      glsl_sampler_type(GLSL_SAMPLER_DIM_2D, false, false, GLSL_TYPE_FLOAT), "image");
   sampler->data.descriptor_set = sampler->data.binding = 0;
   b->shader->info.num_textures = 1;
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, lod ? 2 : 1);
   tex->op = lod ? nir_texop_txl : nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->coord_components = 2;
   tex->dest_type = nir_type_float32;
   tex->texture_index = tex->sampler_index = 0;
   tex->src[0].src_type = nir_tex_src_coord;
   tex->src[0].src = nir_src_for_ssa(coord);
   if (lod) {
      tex->src[1].src_type = nir_tex_src_lod;
      tex->src[1].src = nir_src_for_ssa(lod);
   }
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(b, &tex->instr);
   return &tex->def;
}
static nir_shader *make_control(unsigned kind)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_TESS_CTRL, pco_nir_options(), "tess_fixture_tcs");
   b.shader->info.tess.tcs_vertices_out = kind ? 3 : 1;
   nir_variable *outer = variable(b.shader, nir_var_shader_out, glsl_array_type(glsl_float_type(), 4, 0),
                                  "gl_TessLevelOuter", VARYING_SLOT_TESS_LEVEL_OUTER);
   nir_variable *inner = variable(b.shader, nir_var_shader_out, glsl_array_type(glsl_float_type(), 2, 0),
                                  "gl_TessLevelInner", VARYING_SLOT_TESS_LEVEL_INNER);
   outer->data.patch = inner->data.patch = true;
   nir_def *level = nir_imm_float(&b, 5);
   if (kind == 3)
      level = nir_load_uniform(&b, 1, 32, nir_imm_int(&b, 0), .base = 0, .range = 1);
   if (kind == 4) {
      b.shader->info.num_ubos = 1;
      level = nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
         .align_mul = 4, .align_offset = 0, .range = 16);
   }
   if (kind >= 5) {
      nir_def *id = nir_u2f32(&b, nir_load_invocation_id(&b));
      nir_def *coord = nir_vec2(&b, nir_fadd(&b, nir_fmul_imm(&b, id, .25f),
         nir_imm_float(&b, .125f)), nir_imm_float(&b, .5f));
      level = nir_channel(&b, sample_texture(&b, coord, id), 0);
   }
   for (unsigned i = 0; i < 4; ++i)
      nir_store_deref(&b, nir_build_deref_array_imm(&b, nir_build_deref_var(&b, outer), i), level, 1);
   for (unsigned i = 0; i < 2; ++i)
      nir_store_deref(&b, nir_build_deref_array_imm(&b, nir_build_deref_var(&b, inner), i), level, 1);
   if (kind) {
      nir_variable *input = variable(b.shader, nir_var_shader_in, glsl_array_type(glsl_vec4_type(), 32, 0),
                                     "position_in", VARYING_SLOT_POS);
      nir_variable *output = variable(b.shader, nir_var_shader_out, glsl_array_type(glsl_vec4_type(), 3, 0),
                                      "position_out", VARYING_SLOT_POS);
      nir_def *id = nir_load_invocation_id(&b);
      nir_def *value = nir_load_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, input), id));
      nir_store_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, output), id), value, 15);
      if (kind >= 2) {
         nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
                     .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_shader_out);
         nir_variable *color = variable(b.shader, nir_var_shader_out,
            glsl_array_type(glsl_vec4_type(), 3, 0), "cross_vertex", VARYING_SLOT_VAR0);
         nir_def *next = nir_umod_imm(&b, nir_iadd_imm(&b, id, 1), 3);
         nir_def *other = nir_load_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, output), next));
         nir_store_deref(&b, nir_build_deref_array(&b, nir_build_deref_var(&b, color), id), other, 15);
         nir_variable *patch = variable(b.shader, nir_var_shader_out, glsl_vec4_type(), "patch_color", VARYING_SLOT_PATCH0);
         patch->data.patch = true;
         nir_if *first = nir_push_if(&b, nir_ieq_imm(&b, id, 0));
         nir_store_var(&b, patch, nir_imm_vec4(&b, .25f, .5f, .75f, 1), 15);
         nir_pop_if(&b, first);
      }
   }
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}
static nir_shader *make_evaluation(unsigned kind)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_TESS_EVAL, pco_nir_options(), "tess_fixture_tes");
   b.shader->info.tess._primitive_mode = TESS_PRIMITIVE_TRIANGLES;
   b.shader->info.tess.spacing = TESS_SPACING_EQUAL;
   b.shader->info.tess.ccw = true;
   nir_variable *position = variable(b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position", VARYING_SLOT_POS);
   nir_def *coord = nir_load_tess_coord(&b);
   nir_def *result;
   if (!kind) {
      result = nir_vec4(&b, nir_ffma(&b, nir_channel(&b, coord, 0), nir_imm_float(&b, 1.6f), nir_imm_float(&b, -.8f)),
         nir_ffma(&b, nir_channel(&b, coord, 1), nir_imm_float(&b, 1.6f), nir_imm_float(&b, -.8f)), nir_imm_float(&b, 0), nir_imm_float(&b, 1));
   } else {
      nir_variable *input = variable(b.shader, nir_var_shader_in, glsl_array_type(glsl_vec4_type(), 3, 0),
                                     "position_in", VARYING_SLOT_POS);
      result = nir_imm_vec4(&b, 0, 0, 0, 0);
      for (unsigned i = 0; i < 3; ++i) {
         nir_def *vertex = nir_load_deref(&b, nir_build_deref_array_imm(&b, nir_build_deref_var(&b, input), i));
         result = nir_ffma(&b, vertex, nir_channel(&b, coord, i), result);
      }
   }
   nir_store_var(&b, position, result, 15);
   nir_variable *color = variable(b.shader, nir_var_shader_out, glsl_vec4_type(), "color", VARYING_SLOT_VAR0);
   nir_def *color_value = nir_imm_vec4(&b, .25f, .5f, .75f, 1);
   if (kind >= 2) {
      nir_variable *patch = variable(b.shader, nir_var_shader_in, glsl_vec4_type(), "patch_color", VARYING_SLOT_PATCH0);
      patch->data.patch = true;
      nir_variable *cross = variable(b.shader, nir_var_shader_in, glsl_array_type(glsl_vec4_type(), 3, 0),
                                     "cross_vertex", VARYING_SLOT_VAR0);
      color_value = nir_fadd(&b, nir_load_var(&b, patch), nir_load_deref(&b,
         nir_build_deref_array_imm(&b, nir_build_deref_var(&b, cross), 0)));
   }
   if (kind == 3)
      color_value = nir_fmul(&b, color_value, nir_load_uniform(&b, 1, 32,
         nir_iand_imm(&b, nir_load_primitive_id(&b), 1), .base = 0, .range = 2));
   if (kind == 4) {
      b.shader->info.num_ubos = 1;
      color_value = nir_fmul(&b, color_value, nir_load_ubo(&b, 1, 32, nir_imm_int(&b, 0),
         nir_imul_imm(&b, nir_iand_imm(&b, nir_load_primitive_id(&b), 3), 4),
         .align_mul = 4, .align_offset = 0, .range = 16));
   }
   if (kind >= 5) {
      nir_def *lod = kind == 6 ? nir_u2f32(&b, nir_load_primitive_id(&b)) : NULL;
      color_value = nir_fmul(&b, color_value,
         sample_texture(&b, nir_channels(&b, coord, 3), lod));
   }
   nir_store_var(&b, color, color_value, 15);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}
static nir_shader *make_fragment(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, pco_nir_options(), "tess_fixture_fs");
   nir_variable *in = variable(b.shader, nir_var_shader_in, glsl_vec4_type(), "color", VARYING_SLOT_VAR0);
   nir_variable *out = variable(b.shader, nir_var_shader_out, glsl_vec4_type(), "fragmentColor", FRAG_RESULT_DATA0);
   nir_store_var(&b, out, nir_load_var(&b, in), 15);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}
static void save_stage(unsigned kind, const char *name, const struct pvrgpu_pco_owned_binary *s)
{
   printf("tess-%u.%s bytes=%zu temps=%u vi=%u vo=%u sh=%u cf=%u push=%u/%u ubo=%u/%u\n",
      kind, name, s->size, s->abi.temps, s->abi.vertex_inputs, s->abi.vertex_outputs,
      s->abi.shareds, s->abi.coefficients, s->abi.push_constant_start, s->abi.push_constant_count,
      s->abi.uniform_buffer_descriptor_start, s->abi.uniform_buffer_descriptor_count);
   const char *dir = getenv("PVRGPU_TESS_FIXTURE_DIR");
   if (!dir) return;
   char path[1024];
   require(snprintf(path, sizeof(path), "%s/tess-%u.%s.bin", dir, kind, name) < (int)sizeof(path), "fixture path");
   FILE *f = fopen(path, "wb");
   require(f != NULL, "fixture open");
   require(fwrite(s->data, 1, s->size, f) == s->size, "fixture write");
   require(fclose(f) == 0, "fixture close");
}

static void
test_stream_output_layout(struct pvrgpu_pco_compiler *compiler)
{
   for (unsigned raster = 0; raster < 2; ++raster) {
      nir_shader *vs = make_vertex(false), *tcs = make_control(0), *tes = make_evaluation(0);
      nir_builder b = nir_builder_create(nir_shader_get_entrypoint(tes));
      b.cursor = nir_before_impl(b.impl);
      nir_variable *size = variable(tes, nir_var_shader_out, glsl_float_type(), "gl_PointSize", VARYING_SLOT_PSIZ);
      size->data.always_active_io = true;
      nir_store_var(&b, size, nir_imm_float(&b, 2), 1);
      nir_variable *array = variable(tes, nir_var_shader_out,
         glsl_array_type(glsl_uvec2_type(), 2, 0), "tf_only_array", VARYING_SLOT_VAR0 + 3);
      array->data.always_active_io = true;
      for (unsigned i = 0; i < 2; ++i)
         nir_store_deref(&b, nir_build_deref_array_imm(&b, nir_build_deref_var(&b, array), i),
            nir_vec2(&b, nir_load_primitive_id(&b), nir_imm_int(&b, 0xffabcdefu + i)), 3);
      nir_variable *tag = variable(tes, nir_var_shader_out, glsl_uint_type(), "flat_tag", VARYING_SLOT_VAR0 + 8);
      tag->data.always_active_io = true;
      tag->data.interpolation = INTERP_MODE_FLAT;
      nir_store_var(&b, tag, nir_load_primitive_id(&b), 1);
      nir_shader_gather_info(tes, nir_shader_get_entrypoint(tes));
      const uint64_t original_outputs = tes->info.outputs_written;

      nir_builder f = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, pco_nir_options(), "tf_fixture_fs");
      nir_variable *color = variable(f.shader, nir_var_shader_out, glsl_vec4_type(), "color", FRAG_RESULT_DATA0);
      nir_def *value = nir_imm_vec4(&f, 1, 0, 0, 1);
      if (raster) {
         nir_variable *smooth = variable(f.shader, nir_var_shader_in, glsl_vec4_type(), "smooth", VARYING_SLOT_VAR0);
         nir_variable *flat = variable(f.shader, nir_var_shader_in, glsl_uint_type(), "flat", VARYING_SLOT_VAR0 + 8);
         flat->data.interpolation = INTERP_MODE_FLAT;
         value = nir_fmul(&f, nir_load_var(&f, smooth), nir_u2f32(&f, nir_load_var(&f, flat)));
      }
      nir_store_var(&f, color, value, 15);
      nir_jump(&f, nir_jump_return);
      nir_shader_gather_info(f.shader, f.impl);

      struct pvrgpu_pco_tessellation_pipeline_binary binary = {0};
      char error[512] = {0};
      require(pvrgpu_pco_compile_tessellation_pipeline(compiler, vs, tcs, tes, f.shader,
         NULL, 1, 0, 0, 0, 0, 0, 0, &binary, error, sizeof(error)), error);
      require(tes->info.outputs_written == original_outputs, "TF caller output ordinals unchanged");
      require(binary.output.count[VARYING_SLOT_PSIZ] == 1 && binary.output.start[VARYING_SLOT_PSIZ] == 4,
         "TF-only TES PointSize retained");
      require(binary.output.count[VARYING_SLOT_VAR0 + 3] == 2 &&
              binary.output.count[VARYING_SLOT_VAR0 + 4] == 2 &&
              binary.output.count[VARYING_SLOT_VAR0 + 8] == 1,
         "TF-only sparse TES scalar/array locations retained");
      for (unsigned location = 0; location < 64; ++location) {
         require(binary.graphics.vertex_output_start[location] == binary.output.start[location] &&
                 binary.graphics.vertex_output_count[location] == binary.output.count[location],
            "stream-output physical mapping describes TES, not feeding VS");
      }
      require(binary.evaluation.shader.abi.vertex_outputs == binary.output.stride_dwords,
         "TES export extent is its native ABI");
      require(binary.graphics.explicit_varying_bindings &&
              binary.graphics.varying_binding_count == raster * 2,
         "TF-only TES outputs create no raster bindings");
      require(binary.graphics.fragment.abi.coefficients == 4 + raster * 20 &&
              binary.graphics.fragment_varying_count == raster * 20,
         "only actual FS inputs allocate coefficients");
      if (raster) {
         const struct pvrgpu_pco_varying_binding *bindings = binary.graphics.varying_bindings;
         require(bindings[0].output_dword == binary.output.start[VARYING_SLOT_VAR0] &&
                 bindings[0].num_components == 4 && bindings[0].coefficient_dword == 4 && !bindings[0].flat,
            "smooth TES varying physical linkage");
         require(bindings[1].output_dword == binary.output.start[VARYING_SLOT_VAR0 + 8] &&
                 bindings[1].num_components == 1 && bindings[1].coefficient_dword == 20 && bindings[1].flat,
            "flat TES varying after TF-only gap physical linkage");
      }
      pvrgpu_pco_tessellation_pipeline_binary_finish(&binary);
      ralloc_free(vs); ralloc_free(tcs); ralloc_free(tes); ralloc_free(f.shader);
   }
}

static void test_texture_admission(struct pvrgpu_pco_compiler *compiler)
{
   /* Deliberately malformed/unsupported NIR must be rejected before native
    * PCO lowering, not reach assertions in the compiler or a fabricated SMP. */
   for (unsigned evaluation = 0; evaluation < 2; ++evaluation) {
      for (unsigned invalid = 0; invalid < 17; ++invalid) {
         nir_shader *vs = make_vertex(true), *tcs = make_control(5);
         nir_shader *tes = make_evaluation(6), *fs = make_fragment();
         nir_shader *selected = evaluation ? tes : tcs;
         nir_tex_instr *tex = NULL;
         nir_foreach_block(block, nir_shader_get_entrypoint(selected)) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_tex) { tex = nir_instr_as_tex(instr); break; }
            }
            if (tex) break;
         }
         require(tex != NULL, "negative texture fixture contains a real textureLod operation");
         switch (invalid) {
         case 0: tex->op = nir_texop_tg4; break;
         case 1: tex->op = nir_texop_txf; break;
         case 2: tex->sampler_dim = GLSL_SAMPLER_DIM_3D; break;
         case 3: tex->is_array = true; break;
         case 4: tex->is_shadow = true; break;
         case 5: tex->is_sparse = true; break;
         case 6: tex->texture_non_uniform = true; break;
         case 7: tex->sampler_non_uniform = true; break;
         case 8: tex->coord_components = 3; break;
         case 9: tex->def.num_components = 3; break;
         case 10: tex->def.bit_size = 16; break;
         case 11: tex->dest_type = nir_type_uint32; break;
         case 12: tex->sampler_index = 1; break;
         case 13: tex->texture_index = tex->sampler_index = 1; break;
         case 14: tex->src[1].src_type = nir_tex_src_bias; break;
         case 15: tex->src[0].src_type = nir_tex_src_ddx; break;
         case 16: tex->op = nir_texop_tex; break; /* illegal extra LOD source */
         }
         struct pvrgpu_pco_tessellation_pipeline_binary binary = {0};
         enum pipe_format format = PIPE_FORMAT_R32G32B32A32_FLOAT;
         char error[512] = {0};
         require(!pvrgpu_pco_compile_tessellation_pipeline(compiler, vs, tcs, tes, fs,
            &format, 1, 0, 0, 0, 0, 1, 0, &binary, error, sizeof(error)),
            "unsupported TCS/TES texture contract rejected");
         require(strstr(error, "tessellation texture") != NULL &&
                 !binary.control.shader.data && !binary.evaluation.shader.data,
            "texture admission refusal is named and publishes no native code");
         pvrgpu_pco_tessellation_pipeline_binary_finish(&binary);
         ralloc_free(vs); ralloc_free(tcs); ralloc_free(tes); ralloc_free(fs);
      }
   }
}

int main(void)
{
   glsl_type_singleton_init_or_ref();
   char error[512] = {0};
   struct pvrgpu_pco_compiler *compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
   require(compiler != NULL, error);
   test_stream_output_layout(compiler);
   for (unsigned kind = 0; kind < 7; ++kind) {
      nir_shader *vs = make_vertex(kind != 0), *tcs = make_control(kind), *tes = make_evaluation(kind), *fs = make_fragment();
      struct pvrgpu_pco_tessellation_pipeline_binary binary = {0};
      enum pipe_format format = PIPE_FORMAT_R32G32B32A32_FLOAT;
      bool ok = pvrgpu_pco_compile_tessellation_pipeline(compiler, vs, tcs, tes, fs,
         &format, 1, 0, kind == 3 ? 4 : 0, kind == 3 ? 8 : 0, 0,
         kind != 0, 0, &binary, error, sizeof(error));
      if (!ok) fprintf(stderr, "kind=%u TCS textures=%u TES textures=%u\n", kind,
                      tcs->info.num_textures, tes->info.num_textures);
      require(ok, error);
      require(tcs->info.stage == MESA_SHADER_TESS_CTRL && tes->info.stage == MESA_SHADER_TESS_EVAL, "caller stage unchanged");
      require(binary.control.shader.abi.vertex_inputs == 3 && binary.control.shader.abi.vertex_outputs == 0, "TCS VI/VO");
      require(binary.evaluation.shader.abi.vertex_inputs == 5 && binary.evaluation.shader.abi.vertex_outputs >= 4, "TES VI/VO");
      require(binary.control.shader.abi.uniform_buffer_descriptor_start == 8 + (kind >= 5 ? 20 : 0), "TCS descriptor prefix");
      require(binary.evaluation.shader.abi.uniform_buffer_descriptor_start == 4 + (kind >= 5 ? 20 : 0), "TES descriptor prefix");
      require(binary.patch.per_vertex_offset_dwords >= 6 && binary.patch.patch_stride_dwords ==
         binary.patch.per_vertex_offset_dwords + binary.output_vertices * binary.patch.vertex.stride_dwords, "patch extent");
      require(binary.control.barrier_count == (kind >= 2), "barrier metadata");
      require(binary.control.shader.abi.uniform_buffer_descriptor_count == (kind == 4) &&
              binary.evaluation.shader.abi.uniform_buffer_descriptor_count == (kind == 4), "UBO ranges");
      require(binary.control.shader.abi.push_constant_count == (kind == 3 ? 4 : 0) &&
              binary.evaluation.shader.abi.push_constant_count == (kind == 3 ? 8 : 0), "captured CB0 suffixes");
      save_stage(kind, "vs", &binary.graphics.vertex); save_stage(kind, "tcs", &binary.control.shader);
      save_stage(kind, "tes", &binary.evaluation.shader); save_stage(kind, "fs", &binary.graphics.fragment);
      printf("tess-%u patch vertices=%u input_stride=%u output_stride=%u per_vertex_offset=%u patch_stride=%u tes_stride=%u barrier=%u\n",
         kind,binary.output_vertices,binary.input.stride_dwords,binary.patch.vertex.stride_dwords,
         binary.patch.per_vertex_offset_dwords,binary.patch.patch_stride_dwords,binary.output.stride_dwords,binary.control.barrier_count);
      pvrgpu_pco_tessellation_pipeline_binary_finish(&binary);
      require(!binary.control.shader.data && !binary.evaluation.shader.data && !binary.graphics.vertex.data,
              "finish clears owned binaries");
      ralloc_free(vs); ralloc_free(tcs); ralloc_free(tes); ralloc_free(fs);
   }
   for (unsigned invalid = 0; invalid < 7; ++invalid) {
      nir_shader *vs = make_vertex(false), *tcs = make_control(0), *tes = make_evaluation(0), *fs = make_fragment();
      if (invalid == 0) tcs->info.stage = MESA_SHADER_COMPUTE;
      if (invalid == 1) tes->info.stage = MESA_SHADER_VERTEX;
      if (invalid == 2) tcs->info.tess.tcs_vertices_out = 33;
      if (invalid == 3) tes->info.tess.spacing = TESS_SPACING_UNSPECIFIED;
      if (invalid == 4) tes->info.num_textures = PVRGPU_PCO_MAX_TEXTURES + 1;
      if (invalid == 5) tcs->scratch_size = 16;
      if (invalid == 6) tes->info.tess._primitive_mode = TESS_PRIMITIVE_UNSPECIFIED;
      struct pvrgpu_pco_tessellation_pipeline_binary binary = {0};
      require(!pvrgpu_pco_compile_tessellation_pipeline(compiler, vs, tcs, tes, fs,
         NULL, 1, 0, 0, 0, 0, 0, 0, &binary, error, sizeof(error)), "unsupported input is rejected");
      require(error[0] && !binary.control.shader.data && !binary.evaluation.shader.data, "failure is named and has no output");
      pvrgpu_pco_tessellation_pipeline_binary_finish(&binary);
      ralloc_free(vs); ralloc_free(tcs); ralloc_free(tes); ralloc_free(fs);
   }
   for (unsigned domain = TESS_PRIMITIVE_TRIANGLES; domain <= TESS_PRIMITIVE_ISOLINES; ++domain) {
      for (unsigned spacing = TESS_SPACING_EQUAL; spacing <= TESS_SPACING_FRACTIONAL_EVEN; ++spacing) {
         nir_shader *vs = make_vertex(false), *tcs = make_control(0), *tes = make_evaluation(0), *fs = make_fragment();
         tes->info.tess._primitive_mode = domain;
         tes->info.tess.spacing = spacing;
         tes->info.tess.ccw = spacing & 1;
         tes->info.tess.point_mode = domain == TESS_PRIMITIVE_ISOLINES;
         tcs->info.tess.tcs_vertices_out = domain == TESS_PRIMITIVE_QUADS ? 32 : domain;
         struct pvrgpu_pco_tessellation_pipeline_binary binary = {0};
         require(pvrgpu_pco_compile_tessellation_pipeline(compiler, vs, tcs, tes, fs,
            NULL, 1, 0, 0, 0, 0, 0, 0, &binary, error, sizeof(error)), error);
         require(binary.primitive_mode == domain && binary.spacing == spacing &&
                 binary.ccw == (spacing & 1) && binary.point_mode == (domain == TESS_PRIMITIVE_ISOLINES),
                 "domain/spacing/winding/point metadata retains actual TES declaration");
         require(binary.output_vertices == (domain == TESS_PRIMITIVE_QUADS ? 32 : domain), "real TCS invocation count");
         pvrgpu_pco_tessellation_pipeline_binary_finish(&binary);
         ralloc_free(vs); ralloc_free(tcs); ralloc_free(tes); ralloc_free(fs);
      }
   }
   test_texture_admission(compiler);
   pvrgpu_pco_compiler_destroy(compiler);
   glsl_type_singleton_decref();
   printf("tessellation compiler: PASS %u checks\n", checks);
   return 0;
}
