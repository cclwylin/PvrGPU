/* SPDX-License-Identifier: MIT */
/* Private Mesa producer for pco_texture_sequence_test.cpp. This exercises
 * ordinary NIR texture and dynamic branch lowering, not a capture shader. */
#include "pvrgpu_pco.h"
#include "nir/nir.h"
#include "nir/nir_builder.h"
#include "pco/pco.h"
#include "util/ralloc.h"
#include <stdio.h>
#include <stdlib.h>

static nir_shader *vertex(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
      pco_nir_options(), "texture_sequence_vertex");
   nir_variable *in = nir_variable_create(b.shader, nir_var_shader_in,
      glsl_vec4_type(), "position");
   in->data.location = VERT_ATTRIB_GENERIC0;
   nir_variable *out = nir_variable_create(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "gl_Position");
   out->data.location = VARYING_SLOT_POS;
   nir_store_var(&b, out, nir_load_var(&b, in), 15);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_def *sample(nir_builder *b, nir_def *params, nir_def *index)
{
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 2);
   tex->op = nir_texop_txl;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->coord_components = 2;
   tex->dest_type = nir_type_float32;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
      nir_vec2(b, nir_fadd(b, nir_channel(b, params, 0), nir_i2f32(b, index)),
                    nir_channel(b, params, 1)));
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_float(b, 0));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(b, &tex->instr);
   return &tex->def;
}

static nir_shader *fragment(unsigned kind)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      pco_nir_options(), "texture_sequence_fragment");
   nir_variable *out = nir_variable_create(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "color");
   out->data.location = FRAG_RESULT_DATA0;
   nir_def *params = nir_load_uniform(&b, 4, 32, nir_imm_int(&b, 0),
      .range=1, .dest_type=nir_type_float32);
   nir_def *zero = nir_imm_vec4(&b, 0, 0, 0, 0);
   if (kind < 2) {
      nir_def *sum = zero;
      for (unsigned i = 0; i < (kind == 0 ? 10 : 16); ++i)
         sum = nir_fadd(&b, sum, sample(&b, params, nir_imm_int(&b, i)));
      nir_store_var(&b, out, sum, 15);
   } else {
      nir_variable *index = nir_local_variable_create(b.impl, glsl_int_type(), "index");
      nir_variable *sum = nir_local_variable_create(b.impl, glsl_vec4_type(), "sum");
      nir_store_var(&b, index, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, sum, zero, 15);
      nir_loop *loop = nir_push_loop(&b);
      nir_def *i = nir_load_var(&b, index);
      nir_if *stop = nir_push_if(&b, nir_ige(&b, i, nir_channel(&b, params, 2)));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, stop);
      nir_store_var(&b, sum, nir_fadd(&b, nir_load_var(&b, sum), sample(&b, params, i)), 15);
      nir_store_var(&b, index, nir_iadd_imm(&b, i, 1), 1);
      nir_pop_loop(&b, loop);
      nir_store_var(&b, out, nir_load_var(&b, sum), 15);
   }
   nir_shader_gather_info(b.shader, b.impl);
   b.shader->info.num_textures = 1;
   return b.shader;
}

int main(int argc, char **argv)
{
   if (argc != 2) return 2;
   glsl_type_singleton_init_or_ref();
   char error[512] = {0};
   struct pvrgpu_pco_compiler *compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
   if (!compiler) { fprintf(stderr, "%s\n", error); return 1; }
   for (unsigned kind = 0; kind < 3; ++kind) {
      nir_shader *vs = vertex(), *fs = fragment(kind);
      const enum pipe_format format = PIPE_FORMAT_R32G32B32A32_FLOAT;
      struct pvrgpu_pco_graphics_binary binary;
      if (!pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format,
            false, false, 1, 0, 4, 1, 1, &binary, error, sizeof(error))) {
         fprintf(stderr, "texture sequence %u: %s\n", kind, error); return 1;
      }
      char path[4096];
      snprintf(path, sizeof(path), "%s/sequence-%u.pco", argv[1], kind);
      FILE *file = fopen(path, "wb");
      if (!file || fwrite(binary.fragment.data, 1, binary.fragment.size, file) != binary.fragment.size)
         return 1;
      fclose(file);
      snprintf(path, sizeof(path), "%s/sequence-%u.abi", argv[1], kind);
      file = fopen(path, "w");
      const struct pvrgpu_pco_stage_abi *abi = &binary.fragment.abi;
      if (!file || fprintf(file, "%u %u %u %u\n", abi->shareds,
            abi->push_constant_start, abi->push_constant_count, abi->coefficients) < 0)
         return 1;
      fclose(file);
      printf("texture sequence kind=%u bytes=%zu shareds=%u push=%u+%u coefficients=%u\n",
         kind, binary.fragment.size, abi->shareds, abi->push_constant_start,
         abi->push_constant_count, abi->coefficients);
      pvrgpu_pco_graphics_binary_finish(&binary);
      ralloc_free(vs); ralloc_free(fs);
   }
   pvrgpu_pco_compiler_destroy(compiler);
   glsl_type_singleton_decref();
   return 0;
}
