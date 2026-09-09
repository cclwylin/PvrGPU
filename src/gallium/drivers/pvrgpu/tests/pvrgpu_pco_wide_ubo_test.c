/* SPDX-License-Identifier: MIT */
/* Compiler-only regression: neighboring dynamic UBO loads must not be
 * confused with a dynamic descriptor index after native preprocessing.
 * Direct helper mode additionally tests malformed shapes without handing
 * invalid NIR to an optimizer. It compiles the same production implementation,
 * not a copied validator. */
#ifdef PVRGPU_WIDE_UBO_DIRECT_HELPER
#include "../pvrgpu_pco.c"
#endif
int pvrgpu_existing_tessellation_fixture_main(void);
#define main pvrgpu_existing_tessellation_fixture_main
#include "pvrgpu_pco_tessellation_test.c"
#undef main

static nir_def *wide_ubo_value(nir_builder *b, nir_def *index)
{
   b->shader->info.num_ubos = 1;
   nir_def *offset = nir_ishl_imm(b, index, 7);
   nir_def *sum = nir_imm_float(b, 0);
   for (unsigned column = 0; column < 4; ++column) {
      nir_def *value = nir_load_ubo(b, 4, 32, nir_imm_int(b, 0),
         nir_iadd_imm(b, offset, column * 16),
         .align_mul = 128, .align_offset = column * 16,
         .range_base = column * 16, .range = 16272);
      /* Keep every loaded component observably live. */
      sum = nir_fadd(b, sum, nir_fdot4(b, value,
         nir_imm_vec4(b, 1 + 4 * column, 2 + 4 * column,
                         3 + 4 * column, 4 + 4 * column)));
   }
   return sum;
}

static nir_shader *make_wide_control(void)
{
   nir_shader *nir = make_control(1);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_create(impl);
   b.cursor = nir_before_impl(impl);
   nir_def *level = wide_ubo_value(&b, nir_load_invocation_id(&b));
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic) continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_deref) continue;
         nir_variable *var = nir_deref_instr_get_variable(nir_src_as_deref(intr->src[0]));
         if (var && (var->data.location == VARYING_SLOT_TESS_LEVEL_OUTER ||
                     var->data.location == VARYING_SLOT_TESS_LEVEL_INNER))
            nir_src_rewrite(&intr->src[1], level);
      }
   }
   nir_shader_gather_info(nir, impl);
   return nir;
}

static nir_shader *make_wide_evaluation(void)
{
   nir_shader *nir = make_evaluation(1);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_create(impl);
   b.cursor = nir_before_impl(impl);
   nir_def *value = wide_ubo_value(&b, nir_load_primitive_id(&b));
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic) continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_deref) continue;
         nir_variable *var = nir_deref_instr_get_variable(nir_src_as_deref(intr->src[0]));
         if (var && var->data.location == VARYING_SLOT_VAR0) {
            b.cursor = nir_before_instr(instr);
            nir_src_rewrite(&intr->src[1], nir_vec4(&b, value, value, value, value));
         }
      }
   }
   nir_shader_gather_info(nir, impl);
   return nir;
}

static nir_shader *make_wide_geometry(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_GEOMETRY,
      pco_nir_options(), "wide_ubo_geometry");
   b.shader->info.gs.input_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.output_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.vertices_in = b.shader->info.gs.vertices_out = 1;
   b.shader->info.gs.invocations = 1;
   nir_variable *input = variable(b.shader, nir_var_shader_in,
      glsl_array_type(glsl_vec4_type(), 1, 0), "position_in", VARYING_SLOT_POS);
   nir_variable *position = variable(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "gl_Position", VARYING_SLOT_POS);
   nir_variable *color = variable(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "color", VARYING_SLOT_VAR0);
   nir_def *value = wide_ubo_value(&b, nir_load_primitive_id(&b));
   nir_store_var(&b, position, nir_load_deref(&b,
      nir_build_deref_array_imm(&b, nir_build_deref_var(&b, input), 0)), 15);
   nir_store_var(&b, color, nir_vec4(&b, value, value, value, value), 15);
   nir_emit_vertex(&b, .stream_id = 0);
   nir_end_primitive(&b, .stream_id = 0);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

#ifdef PVRGPU_WIDE_UBO_DIRECT_HELPER
static void test_helper_shapes(void)
{
   static const unsigned widths[] = {0,1,2,3,4,5,6,7,8,9,15,16,17,32};
   for (unsigned size_query = 0; size_query < 2; ++size_query) {
      for (unsigned w = 0; w < sizeof(widths)/sizeof(widths[0]); ++w) {
         for (unsigned invalid = 0; invalid < 7; ++invalid) {
            nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_TESS_CTRL,
               pco_nir_options(), "ubo_metadata_negative");
            b.shader->info.num_ubos = invalid == 6 ? 16 : 1;
            nir_def *block = invalid == 3 ? nir_load_invocation_id(&b) :
               invalid == 4 ? nir_imm_ivec2(&b, 0, 0) :
               nir_imm_int(&b, invalid == 5 ? 1 : 0);
            nir_def *result = size_query ? nir_get_ubo_size(&b, 32, block) :
               nir_load_ubo(&b, 4, 32, block, nir_ishl_imm(&b, nir_load_invocation_id(&b), 7),
                  .align_mul = 128, .align_offset = 0, .range = 16384);
            const unsigned old_width = result->num_components, old_bits = result->bit_size;
            result->num_components = widths[w];
            result->bit_size = invalid == 1 ? 16 : invalid == 2 ? 64 : 32;
            pco_data data = {0};
            void *mem = ralloc_context(NULL);
            char error[512] = {0};
            const bool expected = !invalid && nir_num_components_valid(widths[w]) &&
               widths[w] <= 16 && (!size_query || widths[w] == 1);
            const bool ok = pvrgpu_lower_generic_uniform_buffers(b.shader, &data,
               8, mem, error, sizeof(error));
            require(ok == expected, "UBO exact width/bit/static-block helper admission");
            if (ok) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(nir_def_instr(result));
               require(intr->src[0].ssa->num_components == 2 &&
                       nir_src_comp_as_uint(intr->src[0], 0) == 0x10000 &&
                       data.common.shareds == 12,
                       "only descriptor identity is lowered, with stage prefix intact");
            } else {
               require(error[0] != '\0', "malformed UBO shape/block gives named refusal");
            }
            result->num_components = old_width;
            result->bit_size = old_bits;
            ralloc_free(mem);
            ralloc_free(b.shader);
         }
      }
   }
}
#endif

int main(int argc, char **argv)
{
   const bool old = argc == 2 && strcmp(argv[1], "--expect-old-refusal") == 0;
   require(argc == 1 || old || (argc == 2 && strcmp(argv[1], "--expect-success") == 0),
           "usage: wide-ubo [--expect-success|--expect-old-refusal]");
   glsl_type_singleton_init_or_ref();
   char error[1024] = {0};
   struct pvrgpu_pco_compiler *compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
   require(compiler != NULL, error);
   for (unsigned stage = 0; stage < 3; ++stage) {
   nir_shader *vs = make_vertex(true), *tcs = stage == 0 ? make_wide_control() : make_control(1);
   nir_shader *tes = stage == 1 ? make_wide_evaluation() : make_evaluation(1), *fs = make_fragment();
   nir_shader *gs = stage == 2 ? make_wide_geometry() : NULL;
   nir_shader *wide = stage == 2 ? gs : stage == 1 ? tes : tcs;
   enum pipe_format format = PIPE_FORMAT_R32G32B32A32_FLOAT;
   struct pvrgpu_pco_tessellation_pipeline_binary binary = {0};
   struct pvrgpu_pco_geometry_pipeline_binary geometry = {0};
   const bool ok = stage == 2 ? pvrgpu_pco_compile_geometry_pipeline(compiler, vs, gs, fs,
      &format, 1, 0, 0, 0, 1, 0, &geometry, error, sizeof(error)) :
      pvrgpu_pco_compile_tessellation_pipeline(compiler, vs, tcs, tes, fs,
      &format, 1, 0, 0, 0, 0, 1, 0, &binary, error, sizeof(error));
   printf("WIDE_UBO_COMPILE stage=%u ok=%u error=%s\n", stage, ok, error);
   if (!old) {
      require(ok, error);
      const struct pvrgpu_pco_owned_binary *output = stage == 2 ? &geometry.geometry.shader :
         stage == 1 ? &binary.evaluation.shader : &binary.control.shader;
      const unsigned start = stage == 0 ? 8 : stage == 1 ? 4 : PVRGPU_PCO_GEOMETRY_UBO_DESCRIPTOR_START;
      require(output->abi.uniform_buffer_descriptor_start == start &&
              output->abi.uniform_buffer_descriptor_count == 1,
              "wide UBO descriptor remains stage-local block0");
      save_stage(100 + stage, stage == 2 ? "gs" : stage == 1 ? "tes" : "tcs", output);
   } else {
      require(!ok && strstr(error, "UBO requires a bounded static block and 32-bit load") != NULL,
              "old compiler reproduces the exact wide-load refusal");
      require(!binary.control.shader.data && !binary.evaluation.shader.data,
              "rejected compile publishes no stage code");
   }
   unsigned original_ubo_loads = 0;
   nir_foreach_function_impl(impl, wide) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic) continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_ubo) continue;
            ++original_ubo_loads;
            require(intr->def.num_components == 4 && intr->def.bit_size == 32 &&
                    intr->src[0].ssa->num_components == 1 && nir_src_is_const(intr->src[0]) &&
                    nir_src_as_uint(intr->src[0]) == 0,
                    "original scalar block0 vec4 source remains unchanged");
         }
      }
   }
   require(original_ubo_loads == 4 && wide->info.num_ubos == 1,
           "original four adjacent UBO loads/inventory preserved");
   pvrgpu_pco_tessellation_pipeline_binary_finish(&binary);
   pvrgpu_pco_geometry_pipeline_binary_finish(&geometry);
   ralloc_free(vs); ralloc_free(tcs); ralloc_free(tes); ralloc_free(fs); ralloc_free(gs);
   }
#ifdef PVRGPU_WIDE_UBO_DIRECT_HELPER
   if (!old) test_helper_shapes();
#endif
   pvrgpu_pco_compiler_destroy(compiler);
   glsl_type_singleton_decref();
   printf("wide UBO compiler: PASS %u checks\n", checks);
   return 0;
}
