/* SPDX-License-Identifier: MIT */
#include "pvrgpu_pco.h"
#include "nir/nir.h"
#include "nir/nir_builder.h"
#include "pco/pco.h"
#include "util/ralloc.h"
#include <stdio.h>
#include <stdlib.h>

void check_generic_texture_program(const uint8_t *bytes, size_t size,
                                   const unsigned *units, unsigned count);
void test_generic_texture_bindings(void);

static nir_shader *vertex_shader(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
      pco_nir_options(), "generic_texture_binding_vs");
   nir_variable *position = nir_variable_create(b.shader, nir_var_shader_in,
      glsl_vec4_type(), "position");
   position->data.location = VERT_ATTRIB_GENERIC0;
   nir_variable *output = nir_variable_create(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "gl_Position");
   output->data.location = VARYING_SLOT_POS;
   nir_variable *varying = nir_variable_create(b.shader, nir_var_shader_out,
      glsl_vec2_type(), "uv");
   varying->data.location = VARYING_SLOT_VAR0;
   nir_def *value = nir_load_var(&b, position);
   nir_store_var(&b, output, value, 15);
   nir_store_var(&b, varying, nir_trim_vector(&b, value, 2), 3);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *fragment_shader(const unsigned *units, unsigned count)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      pco_nir_options(), "generic_texture_binding_fs");
   nir_variable *input = nir_variable_create(b.shader, nir_var_shader_in,
      glsl_vec2_type(), "uv");
   input->data.location = VARYING_SLOT_VAR0;
   nir_variable *output = nir_variable_create(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "color");
   output->data.location = FRAG_RESULT_DATA0;
   nir_def *uv = nir_load_var(&b, input);
   nir_def *sum = NULL;
   for (unsigned i = 0; i < count; ++i) {
      nir_tex_instr *sample = nir_tex_instr_create(b.shader, 1);
      sample->op = nir_texop_tex;
      sample->sampler_dim = GLSL_SAMPLER_DIM_2D;
      sample->coord_components = 2;
      sample->dest_type = nir_type_float32;
      sample->texture_index = units[i];
      sample->sampler_index = units[i];
      /* Different coordinates keep repeated sampling observable to CSE. */
      nir_def *coord = nir_fadd_imm(&b, uv, i < 2 ? 0.0f : 0.125f);
      sample->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
      nir_def_init(&sample->instr, &sample->def, 4, 32);
      nir_builder_instr_insert(&b, &sample->instr);
      nir_def *scale = nir_load_uniform(&b, 4, 32,
         nir_imm_int(&b, units[i] * 2), .base = 0, .range = 6,
         .dest_type = nir_type_float32);
      nir_def *bias = nir_load_uniform(&b, 4, 32,
         nir_imm_int(&b, units[i] * 2 + 1), .base = 0, .range = 6,
         .dest_type = nir_type_float32);
      nir_def *value = nir_ffma(&b, &sample->def, scale, bias);
      sum = sum ? nir_fadd(&b, sum, value) : value;
   }
   nir_def *scale = nir_load_uniform(&b, 4, 32,
      nir_imm_int(&b, 4), .base = 0, .range = 6,
      .dest_type = nir_type_float32);
   nir_def *bias = nir_load_uniform(&b, 4, 32,
      nir_imm_int(&b, 5), .base = 0, .range = 6,
      .dest_type = nir_type_float32);
   sum = nir_ffma(&b, sum, scale, bias);
   nir_store_var(&b, output, sum, 15);
   nir_shader_gather_info(b.shader, b.impl);
   b.shader->info.num_textures = 2;
   return b.shader;
}

void test_generic_texture_bindings(void)
{
   glsl_type_singleton_init_or_ref();
   char error[512] = {0};
   struct pvrgpu_pco_compiler *compiler =
      pvrgpu_pco_compiler_create(error, sizeof(error));
   if (!compiler) {
      fprintf(stderr, "%s\n", error);
      abort();
   }
   const unsigned sequences[][3] = {{0, 1}, {1, 0}, {1, 0, 1}};
   const unsigned counts[] = {2, 2, 3};
   const enum pipe_format format = PIPE_FORMAT_R32G32B32A32_FLOAT;
   for (unsigned test = 0; test < 3; ++test) {
      nir_shader *vs = vertex_shader();
      nir_shader *fs = fragment_shader(sequences[test], counts[test]);
      struct pvrgpu_pco_graphics_binary binary;
      if (!pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format,
            false, 1, 0, 24, 1, 2, &binary, error, sizeof(error))) {
         fprintf(stderr, "%s\n", error);
         abort();
      }
      check_generic_texture_program(binary.fragment.data, binary.fragment.size,
                                    sequences[test], counts[test]);
      pvrgpu_pco_graphics_binary_finish(&binary);
      ralloc_free(vs);
      ralloc_free(fs);
   }
   pvrgpu_pco_compiler_destroy(compiler);
   glsl_type_singleton_decref();
}
