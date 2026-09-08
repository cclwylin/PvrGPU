/* SPDX-License-Identifier: MIT */
/* Real Mesa NIR -> native PCO producer. No fabricated SMP encodings. */
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
      pco_nir_options(), "texture_bias_vertex");
   nir_variable *in = nir_variable_create(b.shader, nir_var_shader_in,
      glsl_vec4_type(), "position");
   in->data.location = VERT_ATTRIB_GENERIC0;
   nir_variable *pos = nir_variable_create(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "gl_Position");
   pos->data.location = VARYING_SLOT_POS;
   nir_variable *uv = nir_variable_create(b.shader, nir_var_shader_out,
      glsl_vector_type(GLSL_TYPE_FLOAT, 3), "coords");
   uv->data.location = VARYING_SLOT_VAR0;
   nir_def *p = nir_load_var(&b, in);
   nir_store_var(&b, pos, p, 15);
   nir_store_var(&b, uv, nir_channels(&b, p, 7), 7);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_def *sample(nir_builder *b, nir_def *coords, nir_def *bias,
                       enum glsl_sampler_dim dim, unsigned texture)
{
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, bias ? 2 : 1);
   tex->op = bias ? nir_texop_txb : nir_texop_tex;
   tex->sampler_dim = dim;
   tex->coord_components = dim == GLSL_SAMPLER_DIM_2D ? 2 : 3;
   tex->texture_index = tex->sampler_index = texture;
   tex->dest_type = nir_type_float32;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
      nir_channels(b, coords, (1u << tex->coord_components) - 1));
   if (bias) tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_bias, bias);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(b, &tex->instr);
   return &tex->def;
}

static nir_shader *fragment(unsigned kind)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      pco_nir_options(), "texture_bias_fragment");
   nir_variable *in = nir_variable_create(b.shader, nir_var_shader_in,
      glsl_vector_type(GLSL_TYPE_FLOAT, 3), "coords");
   in->data.location = VARYING_SLOT_VAR0;
   nir_variable *out = nir_variable_create(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "color");
   out->data.location = FRAG_RESULT_DATA0;
   nir_def *coords = nir_load_var(&b, in);
   nir_def *color;
   if (kind < 3 || kind == 4 || kind == 5) {
      const enum glsl_sampler_dim dims[] = {
         GLSL_SAMPLER_DIM_2D, GLSL_SAMPLER_DIM_3D, GLSL_SAMPLER_DIM_CUBE };
      nir_def *bias = kind == 5 ? NULL : kind == 4 ? nir_channel(&b, coords, 2) :
         nir_load_uniform(&b, 1, 32, nir_imm_int(&b, 0),
            .range=1, .dest_type=nir_type_float32);
      color = sample(&b, coords, bias, dims[kind == 5 ? 1 : kind == 4 ? 0 : kind], 0);
   } else {
      /* Generic five-sample postprocess: two AUTO plus BIAS 1/2/3. Each
       * distinct result contributes, so neither CSE nor DCE can hide txb. */
      color = nir_fadd(&b, sample(&b, coords, NULL, GLSL_SAMPLER_DIM_2D, 0),
                           sample(&b, coords, NULL, GLSL_SAMPLER_DIM_2D, 1));
      for (unsigned i = 1; i <= 3; ++i)
         color = nir_fadd(&b, color, sample(&b, coords, nir_imm_float(&b, i),
                                          GLSL_SAMPLER_DIM_2D, 1));
   }
   nir_store_var(&b, out, color, 15);
   nir_shader_gather_info(b.shader, b.impl);
   b.shader->info.num_textures = kind == 3 ? 2 : 1;
   return b.shader;
}

int main(int argc, char **argv)
{
   if (argc != 2) return 2;
   glsl_type_singleton_init_or_ref();
   char error[512] = {0};
   struct pvrgpu_pco_compiler *compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
   if (!compiler) { fprintf(stderr, "%s\n", error); return 1; }
   for (unsigned kind = 0; kind < 6; ++kind) {
      nir_shader *vs = vertex(), *fs = fragment(kind);
      const enum pipe_format format = PIPE_FORMAT_R32G32B32A32_FLOAT;
      struct pvrgpu_pco_graphics_binary binary;
      if (!pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format,
            false, false, 1, 0, 4, 1, kind == 3 ? 2 : 1, &binary, error, sizeof(error))) {
         fprintf(stderr, "texture bias %u: %s\n", kind, error); return 1;
      }
      char path[4096];
      snprintf(path, sizeof(path), "%s/bias-%u.pco", argv[1], kind);
      FILE *file = fopen(path, "wb");
      if (!file || fwrite(binary.fragment.data, 1, binary.fragment.size, file) != binary.fragment.size)
         return 1;
      fclose(file);
      snprintf(path, sizeof(path), "%s/bias-%u.abi", argv[1], kind);
      file = fopen(path, "w");
      const struct pvrgpu_pco_stage_abi *a = &binary.fragment.abi;
      if (!file || fprintf(file, "%u %u %u %u %u\n", a->temps, a->shareds,
             a->push_constant_start, a->push_constant_count, a->coefficients) < 0)
         return 1;
      fclose(file);
      printf("texture bias kind=%u bytes=%zu temps=%u shareds=%u push=%u+%u coeff=%u\n",
         kind, binary.fragment.size, a->temps, a->shareds, a->push_constant_start,
         a->push_constant_count, a->coefficients);
      pvrgpu_pco_graphics_binary_finish(&binary);
      ralloc_free(vs); ralloc_free(fs);
   }
   pvrgpu_pco_compiler_destroy(compiler);
   glsl_type_singleton_decref();
   return 0;
}
