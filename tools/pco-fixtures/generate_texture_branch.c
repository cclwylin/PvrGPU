/* SPDX-License-Identifier: MIT
 * Real Mesa PCO branch/SMP continuation fixture; no encoded bytes are made.
 * Dynamic UV in SH20..21, branch choice in SH22. Both branches sample.
 */
int original_fill_tex_main(int argc, char **argv);
#define main original_fill_tex_main
#include "generate_fill_tex_nearest.c"
#undef main

static nir_def *sample(nir_builder *b, nir_deref_instr *deref, nir_def *uv)
{
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 4);
   tex->op = nir_texop_txl;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->coord_components = 2;
   tex->dest_type = nir_type_float32;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_deref, &deref->def);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_sampler_deref, &deref->def);
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_coord, uv);
   tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_float(b, 0));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(b, &tex->instr);
   return &tex->def;
}

int main(int argc, char **argv)
{
   if (argc != 2) return 2;
   glsl_type_singleton_init_or_ref();
   void *mem = ralloc_context(NULL);
   struct pvr_device_info device;
   struct pvr_device_runtime_info runtime = {0};
   if (!pvr_device_info_init_public_name(&device, "gx6250")) return 2;
   pco_ctx *ctx = pco_ctx_create(&device, &runtime, mem);
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      pco_nir_options(), "native_branch_samples");
   b.shader->info.internal = true;
   nir_variable *sampler = nir_variable_create(b.shader, nir_var_uniform,
      glsl_sampler_type(GLSL_SAMPLER_DIM_2D, false, false, GLSL_TYPE_FLOAT), "image");
   sampler->data.descriptor_set = sampler->data.binding = 0;
   nir_deref_instr *deref = nir_build_deref_var(&b, sampler);
   nir_def *inputs = nir_load_push_constant(&b, 3, 32, nir_imm_int(&b, 0), .base=0, .range=12);
   nir_def *uv = nir_trim_vector(&b, inputs, 2);
   nir_push_if(&b, nir_ine_imm(&b, nir_channel(&b, inputs, 2), 0));
   nir_def *left = sample(&b, deref, nir_fmul_imm(&b, uv, 0.5));
   nir_push_else(&b, NULL);
   nir_def *right = sample(&b, deref, nir_fadd_imm(&b, uv, 0.25));
   nir_pop_if(&b, NULL);
   nir_def *merged = nir_if_phi(&b, left, right);
   for (unsigned c = 0; c < 4; ++c)
      nir_frag_store_pco(&b, nir_channel(&b, merged, c), .base=c);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   pco_data data = {0}; init_descriptor_binding(&data);
   nir_print_shader(b.shader, stdout);
   pco_preprocess_nir(ctx, b.shader); pco_lower_nir(ctx, b.shader, &data);
   pco_postprocess_nir(ctx, b.shader, &data);
   if (!allocate_fragment_shared_data(&data)) return 2;
   data.common.push_consts.range = (pco_range){.start=20, .count=3}; data.common.shareds += 3;
   pco_shader *shader = translate_shader(ctx, b.shader, &data);
   pco_print_shader(shader, stdout, "native branch samples");
   pco_print_binary(shader, stdout, "native branch samples");
   int status = write_binary(argv[1], shader);
   ralloc_free(shader); ralloc_free(data.common.desc_sets[0].bindings);
   ralloc_free(b.shader); ralloc_free(mem); glsl_type_singleton_decref();
   return status;
}
