/* SPDX-License-Identifier: MIT
 * Genuine Mesa PCO fixtures for TEMP shifts and vertex-input register reuse.
 * Link against the pinned PvrGPU/Mesa compiler archives. Set
 * PVRGPU_TESS_FIXTURE_DIR to a fresh output directory; the helper writes
 * tess-300/301/302 (LSL/SHR/ASR with dependent integer outputs) and tess-303
 * (packed-attribute, instance-addressed UBO LSL -> VTXIN) native VS/FS pairs.
 * PCO_DEBUG_PRINT=vs,internal,binary prints the unchanged compiler's own IR.
 * No encoded instruction bytes are constructed or modified here.
 */
int existing_tess_fixture_main(void);
#define main existing_tess_fixture_main
#include "../../src/gallium/drivers/pvrgpu/tests/pvrgpu_pco_tessellation_test.c"
#undef main

static nir_shader *shift_vertex(unsigned kind)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
      pco_nir_options(), "private_vertex_shift_reuse");
   nir_variable *input = variable(b.shader, nir_var_shader_in, glsl_uvec4_type(), "input", VERT_ATTRIB_GENERIC0);
   nir_variable *pos = variable(b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position", VARYING_SLOT_POS);
   nir_variable *out = variable(b.shader, nir_var_shader_out, glsl_uvec4_type(), "result", VARYING_SLOT_VAR0);
   out->data.interpolation = INTERP_MODE_FLAT;
   nir_def *v = nir_load_var(&b, input);
   nir_def *x = nir_channel(&b, v, 0), *count = nir_channel(&b, v, 1);
   nir_def *shift = kind == 0 ? nir_ishl(&b, x, count) :
      kind == 1 ? nir_ushr(&b, x, count) : nir_ishr(&b, x, count);
   nir_store_var(&b, pos, nir_imm_vec4(&b, 0, 0, 0, 1), 15);
   nir_store_var(&b, out, nir_vec4(&b, shift, nir_iadd_imm(&b, shift, 17),
      nir_ixor(&b, shift, nir_channel(&b, v, 2)), nir_channel(&b, v, 3)), 15);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *shift_fragment(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      pco_nir_options(), "private_shift_result");
   nir_variable *in = variable(b.shader, nir_var_shader_in, glsl_uvec4_type(), "result", VARYING_SLOT_VAR0);
   in->data.interpolation = INTERP_MODE_FLAT;
   nir_variable *out = variable(b.shader, nir_var_shader_out, glsl_uvec4_type(), "color", FRAG_RESULT_DATA0);
   nir_store_var(&b, out, nir_load_var(&b, in), 15);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

/* Independent reconstruction of a public native VS operation shape, not an
 * extracted capture binary. Packed SNORM attributes plus a long-lived instance
 * UBO offset make the unmodified PCO allocator reuse VTXIN for the LSL result. */
static nir_shader *reconstructed_vertex(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
      pco_nir_options(), "GLSL230_reconstructed");
   nir_variable *in[5], *out[6];
   for (unsigned i = 0; i < 5; ++i) in[i] = variable(b.shader, nir_var_shader_in,
      glsl_vec4_type(), "in", VERT_ATTRIB_GENERIC0 + i);
   for (unsigned i = 0; i < 6; ++i) out[i] = variable(b.shader, nir_var_shader_out,
      i == 5 ? glsl_vec2_type() : glsl_vec4_type(), "out",
      i ? VARYING_SLOT_VAR0 + i - 1 : VARYING_SLOT_POS);
   nir_def *normal = nir_load_var(&b, in[1]);
   nir_def *tangent = nir_load_var(&b, in[2]);
   nir_def *pos = nir_load_var(&b, in[0]);
   nir_def *id = nir_load_instance_id(&b);
   nir_def *uniform = nir_load_uniform(&b, 1, 32, nir_imm_int(&b, 0),
      .base = 0, .range = 1, .dest_type = nir_type_int32);
   nir_def *offset = nir_ishl_imm(&b, nir_iadd(&b, id, uniform), 7);
   b.shader->info.num_ubos = 2;
   nir_def *world = NULL;
   for (unsigned i = 0; i < 4; ++i) {
      nir_def *col = nir_load_ubo(&b, 4, 32, nir_imm_int(&b, 0),
         nir_iadd_imm(&b, offset, 16*i), .align_mul = 128,
         .align_offset = 16*i, .range_base = 16*i, .range = 16272);
      nir_def *term = i == 3 ? col : nir_fmul(&b, col, nir_channel(&b, pos, i));
      world = i ? nir_fadd(&b, world, term) : term;
   }
   nir_def *clip[2];
   for (unsigned matrix = 0; matrix < 2; ++matrix) {
      clip[matrix] = NULL;
      for (unsigned i = 0; i < 4; ++i) {
         nir_def *col = nir_load_uniform(&b, 4, 32, nir_imm_int(&b, i),
            .base = 1 + 4*matrix, .range = 4, .dest_type = nir_type_float32);
         nir_def *term = nir_fmul(&b, col, nir_channel(&b, world, i));
         clip[matrix] = i ? nir_fadd(&b, clip[matrix], term) : term;
      }
   }
   nir_def *row[3], *n[3], *t[3];
   for (unsigned i = 0; i < 3; ++i) {
      row[i] = nir_load_ubo(&b, 4, 32, nir_imm_int(&b, 0),
         nir_iadd_imm(&b, offset, 64+16*i), .align_mul = 128,
         .align_offset = 64+16*i, .range_base = 64+16*i, .range = 16272);
      n[i] = nir_fdot3(&b, nir_channels(&b, normal, 7), nir_channels(&b, row[i], 7));
   }
   for (unsigned i = 0; i < 3; ++i)
      t[i] = nir_fdot3(&b, nir_channels(&b, tangent, 7), nir_channels(&b, row[i], 7));
   nir_def *uv0 = nir_load_var(&b, in[3]), *uv1 = nir_load_var(&b, in[4]);
   nir_store_var(&b, out[0], clip[0], 15);
   nir_store_var(&b, out[1], clip[0], 15);
   nir_store_var(&b, out[2], clip[1], 15);
   nir_store_var(&b, out[3], nir_vec4(&b, nir_channel(&b, uv0, 0), nir_channel(&b, uv0, 1),
      nir_channel(&b, uv1, 0), nir_channel(&b, uv1, 1)), 15);
   nir_store_var(&b, out[4], nir_vec4(&b, n[0], n[1], n[2], t[0]), 15);
   nir_store_var(&b, out[5], nir_vec2(&b, t[1], t[2]), 3);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *reconstruction_fragment(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      pco_nir_options(), "reconstruction_keep_all_outputs");
   nir_def *sum = nir_imm_float(&b, 0);
   for (unsigned i = 0; i < 5; ++i) {
      nir_variable *in = variable(b.shader, nir_var_shader_in,
         i == 4 ? glsl_vec2_type() : glsl_vec4_type(), "in", VARYING_SLOT_VAR0+i);
      in->data.interpolation = INTERP_MODE_SMOOTH;
      nir_def *v = nir_load_var(&b, in);
      for (unsigned c = 0; c < (i == 4 ? 2 : 4); ++c)
         sum = nir_fadd(&b, sum, nir_channel(&b, v, c));
   }
   nir_variable *out = variable(b.shader, nir_var_shader_out, glsl_vec4_type(), "color", FRAG_RESULT_DATA0);
   nir_store_var(&b, out, nir_vec4(&b, sum, sum, sum, sum), 15);
   nir_jump(&b, nir_jump_return); nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

int main(void)
{
   glsl_type_singleton_init_or_ref();
   char error[1024] = {0};
   struct pvrgpu_pco_compiler *compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
   require(compiler != NULL, error);
   for (unsigned kind = 0; kind < 4; ++kind) {
      nir_shader *vs = kind == 3 ? reconstructed_vertex() : shift_vertex(kind);
      nir_shader *fs = kind == 3 ? reconstruction_fragment() : shift_fragment();
      enum pipe_format formats[5];
      for (unsigned i = 0; i < 5; ++i) formats[i] = kind == 3 ? PIPE_FORMAT_R32G32B32A32_FLOAT : PIPE_FORMAT_R32G32B32A32_UINT;
      if (kind == 3) {
         formats[0] = PIPE_FORMAT_R32G32B32_FLOAT;
         formats[1] = formats[2] = PIPE_FORMAT_R8G8B8A8_SNORM;
         formats[3] = formats[4] = PIPE_FORMAT_R32G32_FLOAT;
      }
      struct pvrgpu_pco_graphics_binary g = {0};
      printf("CASE %u\n", kind); fflush(stdout);
      require(pvrgpu_pco_compile_color_triangle(compiler, vs, fs, formats,
         false, false, 1, kind == 3 ? 36 : 0, 0, kind == 3 ? 5 : 1, 0,
         &g, error, sizeof(error)), error);
      save_stage(300+kind, "vs", &g.vertex); save_stage(300+kind, "fs", &g.fragment);
      pvrgpu_pco_graphics_binary_finish(&g); ralloc_free(vs); ralloc_free(fs);
   }
   pvrgpu_pco_compiler_destroy(compiler); glsl_type_singleton_decref();
   return 0;
}
