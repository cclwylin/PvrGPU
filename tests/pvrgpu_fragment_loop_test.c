/* SPDX-License-Identifier: MIT */
#include "pvrgpu_pco.h"
#include "nir/nir.h"
#include "nir/nir_builder.h"
#include "pco/pco.h"
#include "util/ralloc.h"
#include <stdio.h>
#include <stdlib.h>

static nir_shader *vertex(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, pco_nir_options(), "loop_vs");
   nir_variable *in = nir_variable_create(b.shader, nir_var_shader_in, glsl_vec4_type(), "position");
   in->data.location = VERT_ATTRIB_GENERIC0;
   nir_variable *out = nir_variable_create(b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position");
   out->data.location = VARYING_SLOT_POS;
   nir_store_var(&b, out, nir_load_var(&b, in), 15);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *fragment(unsigned kind)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, pco_nir_options(), "uniform_loop_derivative");
   nir_variable *out = nir_variable_create(b.shader, nir_var_shader_out, glsl_vec4_type(), "color");
   out->data.location = FRAG_RESULT_DATA0;
   nir_def *input = nir_load_uniform(&b, 4, 32, nir_imm_int(&b, 0),
      .base=0, .range=2, .dest_type=nir_type_float32);
   nir_def *count = nir_load_uniform(&b, 1, 32, nir_imm_int(&b, 1),
      .base=0, .range=2, .dest_type=nir_type_int32);
   nir_variable *index = nir_local_variable_create(b.impl, glsl_int_type(), "index");
   nir_variable *sum = nir_local_variable_create(b.impl, glsl_vec4_type(), "sum");
   nir_store_var(&b, index, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, sum, nir_imm_vec4(&b, 0,0,0,0), 15);
   nir_loop *loop = nir_push_loop(&b);
   nir_def *i = nir_load_var(&b, index);
   nir_if *stop = nir_push_if(&b, nir_ige(&b, i, count));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, stop);
   nir_def *operand = nir_fmul(&b, input, nir_i2f32(&b, i));
   nir_def *derivative = nir_ddx(&b, operand);
   if (kind)
      derivative = nir_fadd(&b, nir_fabs(&b, derivative), nir_fabs(&b, nir_ddy(&b, operand)));
   nir_store_var(&b, sum, nir_fadd(&b, nir_load_var(&b, sum), derivative), 15);
   nir_store_var(&b, index, nir_iadd_imm(&b, i, 1), 1);
   nir_pop_loop(&b, loop);
   nir_store_var(&b, out, nir_load_var(&b, sum), 15);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *sample_mask_fragment(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, pco_nir_options(), "sample_mask_bit_scan");
   nir_variable *out = nir_variable_create(b.shader, nir_var_shader_out, glsl_vec4_type(), "color");
   out->data.location = FRAG_RESULT_DATA0;
   nir_def *mask = nir_load_sample_mask_in(&b);
   nir_variable *index = nir_local_variable_create(b.impl, glsl_int_type(), "index");
   nir_variable *first = nir_local_variable_create(b.impl, glsl_int_type(), "first");
   nir_variable *duplicate = nir_local_variable_create(b.impl, glsl_bool_type(), "duplicate");
   nir_store_var(&b, first, nir_imm_int(&b, -1), 1);
   nir_store_var(&b, index, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, duplicate, nir_imm_false(&b), 1);
   nir_loop *loop = nir_push_loop(&b);
   nir_def *i = nir_load_var(&b, index);
   nir_if *stop = nir_push_if(&b, nir_ige_imm(&b, i, 32));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, stop);
   nir_if *found = nir_push_if(&b, nir_ine_imm(&b, nir_iand_imm(&b, nir_ishr(&b, mask, i), 1), 0));
   nir_store_var(&b, first, i, 1);
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, found);
   nir_store_var(&b, index, nir_iadd_imm(&b, i, 1), 1);
   nir_pop_loop(&b, loop);
   nir_store_var(&b, index, nir_iadd_imm(&b, nir_load_var(&b, first), 1), 1);
   loop = nir_push_loop(&b);
   i = nir_load_var(&b, index);
   stop = nir_push_if(&b, nir_ige_imm(&b, i, 32));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, stop);
   found = nir_push_if(&b, nir_ine_imm(&b, nir_iand_imm(&b, nir_ishr(&b, mask, i), 1), 0));
   nir_store_var(&b, duplicate, nir_imm_true(&b), 1);
   nir_pop_if(&b, found);
   nir_store_var(&b, index, nir_iadd_imm(&b, i, 1), 1);
   nir_pop_loop(&b, loop);
   nir_def *inverse_count = nir_load_uniform(&b, 1, 32, nir_imm_int(&b, 0),
      .base=0, .range=1, .dest_type=nir_type_float32);
   nir_def *error = nir_load_var(&b, duplicate);
   nir_def *green = nir_fmul(&b, nir_i2f32(&b, nir_load_var(&b, first)), inverse_count);
   nir_store_var(&b, out, nir_vec4(&b, nir_b2f32(&b, error),
      nir_bcsel(&b, error, nir_imm_float(&b, 0), green),
      nir_u2f32(&b, nir_load_sample_id(&b)), nir_imm_float(&b, 1)), 15);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

int main(int argc, char **argv)
{
   if (argc != 4) return 2;
   glsl_type_singleton_init_or_ref();
   char error[512] = {0};
   struct pvrgpu_pco_compiler *compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
   for (unsigned kind = 0; kind < 3; ++kind) {
      nir_shader *vs = vertex(), *fs = kind == 2 ? sample_mask_fragment() : fragment(kind);
      const enum pipe_format format = PIPE_FORMAT_R32G32B32A32_FLOAT;
      struct pvrgpu_pco_graphics_binary binary;
      if (!pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format, false, false,
            1, 0, 8, 1, 0, &binary, error, sizeof(error))) {
         fprintf(stderr, "%s\n", error); return 1;
      }
      FILE *file = fopen(argv[kind + 1], "wb");
      if (!file || fwrite(binary.fragment.data, 1, binary.fragment.size, file) != binary.fragment.size) return 1;
      fclose(file);
      fprintf(stderr, "kind=%u bytes=%zu CB0=SH0..7, %s\n", kind, binary.fragment.size,
         kind == 2 ? "inverse_sample_count=SH0; SAVMSK/SAMP_NUM=native state" :
                     "input=SH0..3 count=SH4");
      pvrgpu_pco_graphics_binary_finish(&binary);
      ralloc_free(vs); ralloc_free(fs);
   }
   pvrgpu_pco_compiler_destroy(compiler);
   glsl_type_singleton_decref();
   return 0;
}
