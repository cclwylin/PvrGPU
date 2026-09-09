/* SPDX-License-Identifier: MIT
 * Genuine unsigned find-MSB / native FTB compiler fixture.
 * Link against the pinned PvrGPU/Mesa compiler archives and set
 * PVRGPU_TESS_FIXTURE_DIR to a fresh output directory. Outputs tess-400.vs.bin
 * and tess-400.fs.bin, with printed ABI. The FS consumes uint SH0 and exports
 * [findMSB(x), findMSB(x)+1, findMSB(x)^x, x]; findMSB(0) is 0xffffffff.
 * No encoded instruction bytes are authored or modified.
 */
int existing_tess_fixture_main(void);
#define main existing_tess_fixture_main
#include "../../src/gallium/drivers/pvrgpu/tests/pvrgpu_pco_tessellation_test.c"
#undef main
int main(void)
{
   glsl_type_singleton_init_or_ref(); char error[1024] = {0};
   struct pvrgpu_pco_compiler *compiler = pvrgpu_pco_compiler_create(error, sizeof(error)); require(compiler != NULL, error);
   nir_shader *vs = make_vertex(true);
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, pco_nir_options(), "genuine_find_msb");
   nir_variable *out = variable(b.shader, nir_var_shader_out, glsl_uvec4_type(), "result", FRAG_RESULT_DATA0);
   nir_def *x = nir_load_uniform(&b, 1, 32, nir_imm_int(&b, 0), .base=0, .range=1, .dest_type=nir_type_uint32);
   nir_def *msb = nir_ufind_msb(&b, x);
   nir_store_var(&b, out, nir_vec4(&b, msb, nir_iadd_imm(&b, msb, 1), nir_ixor(&b, msb, x), x), 15);
   nir_jump(&b, nir_jump_return); nir_shader_gather_info(b.shader, b.impl);
   const enum pipe_format format = PIPE_FORMAT_R32G32_FLOAT;
   struct pvrgpu_pco_graphics_binary g = {0};
   require(pvrgpu_pco_compile_color_triangle(compiler, vs, b.shader, &format,
      false, false, 1, 0, 1, 1, 0, &g, error, sizeof(error)), error);
   save_stage(400, "vs", &g.vertex); save_stage(400, "fs", &g.fragment);
   pvrgpu_pco_graphics_binary_finish(&g); ralloc_free(vs); ralloc_free(b.shader);
   pvrgpu_pco_compiler_destroy(compiler); glsl_type_singleton_decref(); return 0;
}
