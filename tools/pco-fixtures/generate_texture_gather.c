/* SPDX-License-Identifier: MIT
 * Genuine gx6250 PCO component gather fixtures. No instruction bytes are
 * constructed here. Kind 0/1 selects gathered component 0/1; kind 2 is an
 * ordinary textureLod(0) control. Dynamic normalized UV: SHARED20..21.
 */
int original_fill_tex_main(int argc, char **argv);
#define main original_fill_tex_main
#include "generate_fill_tex_nearest.c"
#undef main
#include <stdlib.h>

int main(int argc, char **argv)
{
   if (argc != 3) return 2;
   unsigned kind = strtoul(argv[1], NULL, 0);
   if (kind > 2) return 2;
   glsl_type_singleton_init_or_ref();
   void *mem = ralloc_context(NULL);
   struct pvr_device_info device;
   struct pvr_device_runtime_info runtime = {0};
   if (!pvr_device_info_init_public_name(&device, "gx6250")) return 2;
   pco_ctx *ctx = pco_ctx_create(&device, &runtime, mem);
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      pco_nir_options(), "native_component_gather");
   b.shader->info.internal = true;
   nir_variable *sampler = nir_variable_create(b.shader, nir_var_uniform,
      glsl_sampler_type(GLSL_SAMPLER_DIM_2D, false, false, GLSL_TYPE_FLOAT), "image");
   sampler->data.descriptor_set = 0;
   sampler->data.binding = 0;
   nir_deref_instr *deref = nir_build_deref_var(&b, sampler);
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, kind == 2 ? 4 : 3);
   tex->op = kind == 2 ? nir_texop_txl : nir_texop_tg4;
   tex->component = kind == 1 ? 1 : 0;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->coord_components = 2;
   tex->dest_type = nir_type_float32;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_deref, &deref->def);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_sampler_deref, &deref->def);
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_coord,
      nir_load_push_constant(&b, 2, 32, nir_imm_int(&b, 0), .base=0, .range=8));
   if (kind == 2)
      tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_float(&b, 0));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   for (unsigned c = 0; c < 4; ++c)
      nir_frag_store_pco(&b, nir_channel(&b, &tex->def, c), .base=c);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   pco_data data = {0};
   init_descriptor_binding(&data);
   nir_print_shader(b.shader, stdout);
   pco_preprocess_nir(ctx, b.shader);
   pco_lower_nir(ctx, b.shader, &data);
   pco_postprocess_nir(ctx, b.shader, &data);
   if (!allocate_fragment_shared_data(&data)) return 2;
   data.common.push_consts.range = (pco_range){.start=20, .count=2};
   data.common.shareds += 2;
   pco_shader *shader = translate_shader(ctx, b.shader, &data);
   pco_print_shader(shader, stdout, "native component gather");
   pco_print_binary(shader, stdout, "native component gather");
   const pco_data *compiled = pco_shader_data(shader);
   fprintf(stderr, "kind=%u temps=%u shared=%u push_used=%u coeff=%u\n", kind,
      compiled->common.temps, compiled->common.shareds,
      compiled->common.push_consts.used, compiled->common.coeffs);
   int status = write_binary(argv[2], shader);
   ralloc_free(shader);
   ralloc_free(data.common.desc_sets[0].bindings);
   ralloc_free(b.shader);
   ralloc_free(mem);
   glsl_type_singleton_decref();
   return status;
}
