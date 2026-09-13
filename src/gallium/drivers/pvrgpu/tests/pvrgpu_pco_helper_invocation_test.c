/* SPDX-License-Identifier: MIT */

int pvrgpu_existing_tessellation_fixture_main(void);
#define main pvrgpu_existing_tessellation_fixture_main
#include "pvrgpu_pco_tessellation_test.c"
#undef main

static nir_shader *
make_helper_derivative_fragment(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      pco_nir_options(), "helper_invocation_derivative_fs");
   nir_variable *color = variable(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "color", FRAG_RESULT_DATA0);
   nir_def *helper = nir_load_helper_invocation(&b, 1);
   nir_def *value = nir_b2f32(&b, helper);
   nir_def *derivative = nir_fabs(&b, nir_ddx(&b, value));
   nir_store_var(&b, color,
      nir_vec4(&b, value, derivative, nir_imm_float(&b, 0),
               nir_imm_float(&b, 1)), 0xf);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   char error[512] = {0};
   struct pvrgpu_pco_compiler *compiler =
      pvrgpu_pco_compiler_create(error, sizeof(error));
   require(compiler != NULL, error);

   nir_shader *vs = make_vertex(true);
   nir_shader *fs = make_helper_derivative_fragment();
   char *original = nir_shader_as_str(fs, NULL);
   const enum pipe_format format = PIPE_FORMAT_R32G32B32A32_FLOAT;
   struct pvrgpu_pco_graphics_binary binary = {0};
   require(pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format,
      false, false, 1, 0, 0, 1, 0, &binary, error, sizeof(error)), error);

   require(binary.fragment.data && binary.fragment.size,
           "helper-invocation derivative produced no native FS");
   require(binary.fragment.abi.coefficients == 4 &&
           binary.fragment_position_start == 0 &&
           binary.fragment_position_count == 4 &&
           binary.fragment_position_uses_z == 0 &&
           binary.fragment_position_uses_w == 1 &&
           binary.fragment_varying_start == 4 &&
           binary.fragment_varying_count == 0 &&
           binary.varying_output_count == 0 &&
           binary.varying_binding_count == 0,
           "derivative-only FS lost its reciprocal-W coefficient ABI");
   require(binary.fragment_output_mask[0] == 0xf,
           "helper-invocation derivative lost its color output");

   char *after = nir_shader_as_str(fs, NULL);
   require(strcmp(original, after) == 0,
           "helper-invocation compilation modified caller-owned NIR");
   ralloc_free(original);
   ralloc_free(after);
   pvrgpu_pco_graphics_binary_finish(&binary);
   ralloc_free(vs);
   ralloc_free(fs);
   pvrgpu_pco_compiler_destroy(compiler);
   glsl_type_singleton_decref();
   printf("helper-invocation compiler: %u checks PASS\n", checks);
   return 0;
}
