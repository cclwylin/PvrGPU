/* SPDX-License-Identifier: MIT
 * Standalone native MS texture fixture generator, pinned Mesa PCO gx6250.
 * Reuses the documented descriptor allocation and native binary writer from
 * generate_fill_tex_nearest.c; no encoded instruction bytes are constructed.
 * Kinds 0..2: FS float/uint/sint texelFetch2DMS; 3..5: corresponding array.
 * Kinds 6/7: FS textureSize2DMS/array; 8/9: textureSamples2DMS/array.
 * Kind 10 is a size-query zero-LOD equivalence probe. Kinds 11..16 repeat
 * texelFetch cases 0..5 using real VTXOUT writes for VS continuation tests.
 * Kind 17 exports native gl_FragCoord x/y and floor(19-y), for origin tests.
 * Kinds 18/19 export MS size x/y/layers and sample count, with no SMP.
 * Kind 20 exports ordinary sampler2D size with dynamic LOD in SHARED20.
 * Kinds 21..23 export ordinary texelFetch on 2D, 2D-array, 3D; kind 24
 * exports cube textureLod. SHARED20..23 are coords and explicit LOD.
 * Command arguments: KIND OUTPUT.bin. Descriptors SHARED0..19; dynamic
 * integer x,y,layer,sample DWORD inputs SHARED20..23, except query-only cases.
 */
int original_fill_tex_main(int argc, char **argv);
#define main original_fill_tex_main
#include "generate_fill_tex_nearest.c"
#undef main
#include <stdlib.h>

static nir_shader *multisample_fragment(unsigned kind)
{
   const bool vertex = kind >= 11 && kind <= 16;
   if (vertex) kind -= 11;
   nir_builder b = nir_builder_init_simple_shader(vertex ? MESA_SHADER_VERTEX : MESA_SHADER_FRAGMENT, pco_nir_options(), "native_multisample_texture");
   /* Match pvrgpu_pco_compile_color_triangle's public graphics profile:
    * PVRGPU owns raster/PBE, so PCO emits the internal PIXOUT ABI rather
    * than Vulkan's extra alpha/sample-mask epilogue. No instruction bytes
    * are constructed here; all operations pass through native Mesa PCO. */
   b.shader->info.internal = true;
   if (kind >= 21 && kind <= 24) {
      const bool array = kind == 22;
      const bool fetch = kind != 24;
      const enum glsl_sampler_dim dim = kind == 23 ? GLSL_SAMPLER_DIM_3D :
         kind == 24 ? GLSL_SAMPLER_DIM_CUBE : GLSL_SAMPLER_DIM_2D;
      const unsigned dimensions = kind == 21 ? 2 : 3;
      nir_variable *sampler = nir_variable_create(b.shader, nir_var_uniform,
         glsl_sampler_type(dim, false, array, GLSL_TYPE_FLOAT), "image");
      sampler->data.descriptor_set = 0;
      sampler->data.binding = 0;
      nir_deref_instr *deref = nir_build_deref_var(&b, sampler);
      nir_def *inputs = nir_load_push_constant(&b, 4, 32, nir_imm_int(&b, 0), .base=0, .range=16);
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 4);
      tex->op = fetch ? nir_texop_txf : nir_texop_txl;
      tex->sampler_dim = dim;
      tex->is_array = array;
      tex->coord_components = dimensions;
      tex->dest_type = nir_type_float32;
      tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_deref, &deref->def);
      tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_sampler_deref, &deref->def);
      tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_coord, nir_trim_vector(&b, inputs, dimensions));
      tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_channel(&b, inputs, 3));
      nir_def_init(&tex->instr, &tex->def, 4, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      for (unsigned c=0; c<4; ++c)
         nir_frag_store_pco(&b, nir_channel(&b, &tex->def, c), .base=c);
      nir_jump(&b, nir_jump_return);
      nir_shader_gather_info(b.shader, b.impl);
      return b.shader;
   }
   if (kind == 20) {
      nir_variable *sampler = nir_variable_create(b.shader, nir_var_uniform,
         glsl_sampler_type(GLSL_SAMPLER_DIM_2D, false, false, GLSL_TYPE_UINT), "image");
      sampler->data.descriptor_set = 0;
      sampler->data.binding = 0;
      nir_deref_instr *deref = nir_build_deref_var(&b, sampler);
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);
      tex->op = nir_texop_txs;
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
      tex->dest_type = nir_type_int32;
      tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_deref, &deref->def);
      tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod,
         nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .base=0, .range=4));
      nir_def_init(&tex->instr, &tex->def, 2, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      for (unsigned c=0; c<4; ++c)
         nir_frag_store_pco(&b, c < 2 ? nir_channel(&b, &tex->def, c) :
            nir_imm_int(&b, c == 2 ? 1 : 81), .base=c);
      nir_jump(&b, nir_jump_return);
      nir_shader_gather_info(b.shader, b.impl);
      return b.shader;
   }
   if (kind == 18 || kind == 19) {
      const bool array = kind == 19;
      nir_variable *sampler = nir_variable_create(b.shader, nir_var_uniform,
         glsl_sampler_type(GLSL_SAMPLER_DIM_MS, false, array, GLSL_TYPE_UINT), "ms_image");
      sampler->data.descriptor_set = 0;
      sampler->data.binding = 0;
      nir_deref_instr *deref = nir_build_deref_var(&b, sampler);
      nir_def *results[2];
      for (unsigned query=0; query<2; ++query) {
         nir_tex_instr *tex = nir_tex_instr_create(b.shader, 1);
         tex->op = query == 0 ? nir_texop_txs : nir_texop_texture_samples;
         tex->sampler_dim = GLSL_SAMPLER_DIM_MS;
         tex->is_array = array;
         tex->dest_type = nir_type_uint32;
         tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_deref, &deref->def);
         nir_def_init(&tex->instr, &tex->def, query == 0 ? 2 + array : 1, 32);
         nir_builder_instr_insert(&b, &tex->instr);
         results[query] = &tex->def;
      }
      for (unsigned c=0; c<4; ++c)
         nir_frag_store_pco(&b, c == 3 ? results[1] : c == 2 && !array ?
            nir_imm_int(&b, 1) : nir_channel(&b, results[0], c), .base=c);
      nir_jump(&b, nir_jump_return);
      nir_shader_gather_info(b.shader, b.impl);
      return b.shader;
   }
   if (kind == 17) {
      nir_variable *position = nir_variable_create(b.shader, nir_var_shader_in,
         glsl_vec4_type(), "gl_FragCoord");
      position->data.location = VARYING_SLOT_POS;
      position->data.interpolation = INTERP_MODE_NOPERSPECTIVE;
      nir_def *xy = nir_load_input(&b, 2, 32, nir_imm_int(&b, 0),
         .base=0, .io_semantics={.location=VARYING_SLOT_POS, .num_slots=1});
      nir_def *values[] = {nir_channel(&b, xy, 0), nir_channel(&b, xy, 1),
         nir_ffloor(&b, nir_fsub(&b, nir_imm_float(&b, 19),
                               nir_channel(&b, xy, 1))), nir_imm_float(&b, 1)};
      for (unsigned c=0; c<4; ++c)
         nir_frag_store_pco(&b, values[c], .base=c);
      nir_jump(&b, nir_jump_return);
      nir_shader_gather_info(b.shader, b.impl);
      return b.shader;
   }
   const bool array = kind < 6 ? kind >= 3 : (kind == 7 || kind == 9);
   const bool query = kind >= 6;
   const bool size_query = kind == 6 || kind == 7 || kind == 10;
   const enum glsl_base_type type = kind < 6 ?
      (kind % 3 == 0 ? GLSL_TYPE_FLOAT : kind % 3 == 1 ? GLSL_TYPE_UINT : GLSL_TYPE_INT) : GLSL_TYPE_UINT;
   nir_variable *sampler = nir_variable_create(b.shader, nir_var_uniform,
      glsl_sampler_type(GLSL_SAMPLER_DIM_MS, false, array, type), "ms_image");
   sampler->data.descriptor_set = 0;
   sampler->data.binding = 0;
   nir_deref_instr *deref = nir_build_deref_var(&b, sampler);
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, query ? (kind == 10 ? 2 : 1) : 4);
   tex->op = !query ? nir_texop_txf_ms : size_query ? nir_texop_txs : nir_texop_texture_samples;
   tex->sampler_dim = GLSL_SAMPLER_DIM_MS;
   tex->is_array = array;
   tex->coord_components = array ? 3 : 2;
   tex->texture_index = 0;
   tex->sampler_index = 0;
   tex->dest_type = type == GLSL_TYPE_FLOAT ? nir_type_float32 : type == GLSL_TYPE_UINT ? nir_type_uint32 : nir_type_int32;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_deref, &deref->def);
   if (!query) {
      nir_def *data = nir_load_push_constant(&b, 4, 32, nir_imm_int(&b, 0), .base=0, .range=16);
      tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_sampler_deref, &deref->def);
      tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_coord, nir_trim_vector(&b, data, tex->coord_components));
      tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_ms_index, nir_channel(&b, data, 3));
   } else if (kind == 10) {
      tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(&b, 0));
   }
   const unsigned width = !query ? 4 : size_query ? (array ? 3 : 2) : 1;
   nir_def_init(&tex->instr, &tex->def, width, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   for (unsigned c=0; c<width; ++c) {
      if (vertex)
         nir_uvsw_write_pco(&b, nir_imm_int(&b, c), nir_channel(&b, &tex->def, c));
      else
         nir_frag_store_pco(&b, nir_channel(&b, &tex->def, c), .base=c);
   }
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

int main(int argc, char **argv)
{
   if (argc != 3) return 2;
   unsigned kind = strtoul(argv[1], NULL, 0);
   if (kind > 24) return 2;
   glsl_type_singleton_init_or_ref();
   void *mem = ralloc_context(NULL);
   struct pvr_device_info device;
   struct pvr_device_runtime_info runtime = {0};
   if (!pvr_device_info_init_public_name(&device, "gx6250")) return 2;
   pco_ctx *ctx = pco_ctx_create(&device, &runtime, mem);
   nir_shader *nir = multisample_fragment(kind);
   pco_data data = {0};
   init_descriptor_binding(&data);
   if (kind == 17) {
      data.fs.varyings[VARYING_SLOT_POS] = (pco_range){.start=0, .count=4};
      data.fs.uses.w = true;
      data.common.coeffs = 4;
   }
   nir_print_shader(nir, stdout);
   pco_preprocess_nir(ctx, nir);
   pco_lower_nir(ctx, nir, &data);
   pco_postprocess_nir(ctx, nir, &data);
   if (kind != 17 && !allocate_fragment_shared_data(&data)) return 2;
   data.common.push_consts.range = (pco_range){.start=kind == 17 ? 0 : 20,
      .count=(kind<6 || (kind>=11 && kind<=16) || kind>=21) ? 4 : kind == 20 ? 1 : 0};
   data.common.shareds += data.common.push_consts.range.count;
   pco_shader *shader = translate_shader(ctx, nir, &data);
   pco_print_shader(shader, stdout, "native multisample texture");
   pco_print_binary(shader, stdout, "native multisample texture");
   const pco_data *compiled = pco_shader_data(shader);
   fprintf(stderr, "kind=%u temps=%u shared=%u push_used=%u coeff=%u\n", kind, compiled->common.temps, compiled->common.shareds, compiled->common.push_consts.used, compiled->common.coeffs);
   int status = write_binary(argv[2], shader);
   ralloc_free(shader);
   ralloc_free(data.common.desc_sets[0].bindings);
   ralloc_free(nir);
   ralloc_free(mem);
   glsl_type_singleton_decref();
   return status;
}
