// SPDX-License-Identifier: MIT
// Generate the true no-user-varying derivative/facing pipeline fixtures.
// Compile/link with the same pinned Mesa flags/libraries as the included
// pvrgpu_pco_tessellation_test.c. PVRGPU_TESS_FIXTURE_DIR selects output.
int existing_tess_fixture_main(void);
#define main existing_tess_fixture_main
#include "../../src/gallium/drivers/pvrgpu/tests/pvrgpu_pco_tessellation_test.c"
#undef main
int main(void)
{
   glsl_type_singleton_init_or_ref();
   char error[1024] = {0};
   struct pvrgpu_pco_compiler *compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
   require(compiler != NULL, error);
   nir_shader *vs = make_vertex(true);
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      pco_nir_options(), "private_derivative_position_front_face");
   nir_variable *pos = variable(b.shader, nir_var_shader_in, glsl_vec4_type(), "gl_FragCoord", VARYING_SLOT_POS);
   nir_variable *face = variable(b.shader, nir_var_shader_in, glsl_uint_type(), "gl_FrontFacing", VARYING_SLOT_FACE);
   face->data.interpolation = INTERP_MODE_FLAT;
   nir_variable *color = variable(b.shader, nir_var_shader_out, glsl_vec4_type(), "result", FRAG_RESULT_DATA0);
   nir_def *coord = nir_load_var(&b, pos);
   nir_def *dx = nir_ddx_fine(&b, nir_channel(&b, coord, 0));
   nir_def *dy = nir_ddy_fine(&b, nir_channel(&b, coord, 1));
   nir_def *front = nir_b2b1(&b, nir_load_var(&b, face));
   nir_store_var(&b, color, nir_vec4(&b, dx, dy, nir_b2f32(&b, front), nir_imm_float(&b, 1)), 15);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   nir_print_shader(b.shader, stderr);
   const enum pipe_format format = PIPE_FORMAT_R32G32B32A32_FLOAT;
   struct pvrgpu_pco_graphics_binary g = {0};
   require(pvrgpu_pco_compile_color_triangle(compiler, vs, b.shader, &format,
      false, false, 1, 0, 0, 1, 0, &g, error, sizeof(error)), error);
   save_stage(200, "vs", &g.vertex); save_stage(200, "fs", &g.fragment);
   printf("layout position=%u/%u fragment_position=%u/%u varying=%u/%u fragment_varying=%u/%u bindings=%u explicit=%u mask=%u\n",
      g.position_output_start, g.position_output_count, g.fragment_position_start, g.fragment_position_count,
      g.varying_output_start, g.varying_output_count, g.fragment_varying_start, g.fragment_varying_count,
      g.varying_binding_count, g.explicit_varying_bindings, g.fragment_output_mask[0]);
   pvrgpu_pco_graphics_binary_finish(&g);
   ralloc_free(vs); ralloc_free(b.shader);
   pvrgpu_pco_compiler_destroy(compiler); glsl_type_singleton_decref();
   return 0;
}
