/* SPDX-License-Identifier: MIT */
#include "pvrgpu_pco.h"
#include "nir/nir.h"
#include "nir/nir_builder.h"
#include "pco/pco.h"
#include "pco/pco_isa.h"
#include "util/ralloc.h"
#include <stdio.h>
#include <stdlib.h>

static unsigned checks;
static void check(bool value, const char *reason)
{
   ++checks;
   if (!value) { fprintf(stderr, "%s\n", reason); exit(1); }
}

static nir_def *branch(nir_builder *b, nir_deref_instr *sampler,
                       nir_def *selector, nir_def *position, unsigned depth, unsigned kind)
{
   if (depth == 6) return nir_imm_vec4(b, 1, 0, 1, 1);
   nir_push_if(b, nir_ieq_imm(b, selector, depth));
   const float directions[6][3] = {
      {1,.25f,-.5f}, {-1,.25f,.5f}, {.25f,1,-.5f},
      {.25f,-1,.5f}, {.25f,-.5f,1}, {-.25f,-.5f,-1}};
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, kind == 1 ? 3 : 4);
   tex->op = kind == 1 ? nir_texop_tex : nir_texop_txl;
   tex->sampler_dim = GLSL_SAMPLER_DIM_CUBE;
   tex->coord_components = 3; tex->dest_type = nir_type_float32;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_deref, &sampler->def);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_sampler_deref, &sampler->def);
   nir_def *dynamic = kind == 6 ? nir_load_uniform(b, 4, 32, nir_imm_int(b, 0),
      .base=depth+1, .range=1, .dest_type=nir_type_float32) : NULL;
   nir_def *coord = dynamic ? nir_trim_vector(b, dynamic, 3) :
      nir_imm_vec3(b, directions[depth][0], directions[depth][1], directions[depth][2]);
   if (position) coord = nir_fadd(b, coord, nir_fmul_imm(b, nir_trim_vector(b, position, 3), .03125));
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   if (kind != 1) tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_lod,
      dynamic ? nir_channel(b, dynamic, 3) : nir_imm_float(b, 0));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(b, &tex->instr);
   nir_def *value = &tex->def;
   if (kind == 3) nir_terminate(b);
   if (kind == 4 || kind == 7) value = nir_fadd(b, value,
      nir_load_ubo(b, 4, 32, nir_imm_int(b, 0),
         kind == 7 ? nir_imm_int(b, 4096) : nir_imul_imm(b, selector, 16),
         .align_mul=16, .align_offset=0, .range=256, .access=ACCESS_CAN_SPECULATE));
   if (kind == 5) value = nir_fadd(b, value, nir_ddx(b, nir_load_frag_coord(b)));
   nir_push_else(b, NULL);
   nir_def *other = branch(b, sampler, selector, position, depth + 1, kind);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, value, other);
}

static nir_shader *fragment(unsigned kind)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      pco_nir_options(), "uniform_explicit_texture_select");
   nir_variable *out = nir_variable_create(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "color");
   out->data.location = FRAG_RESULT_DATA0;
   nir_variable *sampler = nir_variable_create(b.shader, nir_var_uniform,
      glsl_sampler_type(GLSL_SAMPLER_DIM_CUBE, false, false, GLSL_TYPE_FLOAT), "image");
   sampler->data.binding = sampler->data.descriptor_set = 0;
   nir_def *selector = kind == 2 ? nir_f2i32(&b, nir_channel(&b, nir_load_frag_coord(&b), 0)) :
      nir_load_uniform(&b, 1, 32, nir_imm_int(&b, 0), .base=0, .range=1, .dest_type=nir_type_int32);
   nir_def *position = NULL;
   if (kind == 8) {
      nir_variable *pos = nir_variable_create(b.shader, nir_var_shader_in, glsl_vec4_type(), "gl_FragCoord");
      pos->data.location = VARYING_SLOT_POS;
      position = nir_load_var(&b, pos); // No ACCESS_CAN_SPECULATE; already outside every if.
   }
   nir_def *result = branch(&b, nir_build_deref_var(&b, sampler), selector, position, 0, kind);
   nir_store_var(&b, out, result, 15);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static unsigned if_count(nir_shader *nir)
{
   unsigned count = 0;
   nir_foreach_function_impl(impl, nir)
      nir_foreach_block(block, impl) count += nir_block_get_following_if(block) != NULL;
   return count;
}

static nir_shader *vertex(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
      pco_nir_options(), "texture_select_vertex");
   nir_variable *in = nir_variable_create(b.shader, nir_var_shader_in, glsl_vec4_type(), "position");
   nir_variable *out = nir_variable_create(b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position");
   in->data.location = VERT_ATTRIB_GENERIC0; out->data.location = VARYING_SLOT_POS;
   nir_store_var(&b, out, nir_load_var(&b, in), 15);
   nir_jump(&b, nir_jump_return); nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

int main(int argc, char **argv)
{
   check(argc == 4, "expected two native FS output paths and ISA upper-source records");
   char error[1024];
   struct pvrgpu_pco_compiler *compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
   check(compiler != NULL, error);
   for (unsigned kind = 0; kind != 9; ++kind) {
      nir_shader *nir = fragment(kind);
      check(if_count(nir) == 6, "fixture contains real six-level structured control flow");
      bool progress = pvrgpu_lower_uniform_fragment_texture_selects(nir);
      printf("kind=%u progress=%u ifs=%u\n", kind, progress, if_count(nir));
      check(progress == (kind == 0 || kind == 6 || kind == 8), "only uniform side-effect-free explicit LOD may flatten");
      check(if_count(nir) == (kind == 0 || kind == 6 || kind == 8 ? 0 : 6), "excluded control flow is preserved");
      if (kind == 0 || kind == 6 || kind == 8) check(!pvrgpu_lower_uniform_fragment_texture_selects(nir), "select lowering is idempotent");
      ralloc_free(nir);
   }
   for (unsigned variant = 0; variant != 2; ++variant) {
   nir_shader *vs = vertex(), *fs = fragment(variant ? 6 : 0);
   const enum pipe_format format = PIPE_FORMAT_R32G32B32A32_FLOAT;
   struct pvrgpu_pco_graphics_binary binary = {0};
   check(pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format,
      false, false, 1, 0, variant ? 28 : 4, 1, 1, &binary, error, sizeof(error)), error);
   FILE *file = fopen(argv[variant + 1], "wb"); check(file != NULL, "open native FS fixture");
   check(fwrite(binary.fragment.data, 1, binary.fragment.size, file) == binary.fragment.size,
      "write genuine native FS bytes");
   check(fclose(file) == 0, "close native FS fixture");
   printf("FS size=%zu temps=%u coefficients=%u shared=%u push=%u/%u\n",
      binary.fragment.size, binary.fragment.abi.temps, binary.fragment.abi.coefficients,
      binary.fragment.abi.shareds, binary.fragment.abi.push_constant_start, binary.fragment.abi.push_constant_count);
   pvrgpu_pco_graphics_binary_finish(&binary); ralloc_free(vs); ralloc_free(fs);
   }
   pvrgpu_pco_compiler_destroy(compiler);
   FILE *upper = fopen(argv[3], "wb"); check(upper != NULL, "open exact ISA-encoded response fixtures");
   const unsigned responses[] = {31,32,127,128,252,253,255};
   for (unsigned i=0;i<7;++i) {
      uint8_t record[6] = {responses[i],0};
      record[1] = responses[i] < 128 ? pco_src_2up_2b7i_2b7i_encode(record+2,
         .sb3=PCO_REGBANK_SPECIAL, .s3=0, .sb4=PCO_REGBANK_TEMP, .s4=responses[i]) :
         pco_src_2up_3b11i_2b8i_encode(record+2,
         .sb3=PCO_REGBANK_SPECIAL, .s3=0, .sb4=PCO_REGBANK_TEMP, .s4=responses[i]);
      check(fwrite(record,1,sizeof(record),upper)==sizeof(record), "write public Mesa ISA response fields");
   }
   check(fclose(upper)==0, "close response records");
   printf("fragment texture select compiler %u checks PASS\n", checks);
   return 0;
}
