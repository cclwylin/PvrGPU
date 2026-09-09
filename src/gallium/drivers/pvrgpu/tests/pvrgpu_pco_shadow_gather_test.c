/* SPDX-License-Identifier: MIT */
/* Real NIR -> pinned PCO compilation and read-only sampler-use proof. */
int pvrgpu_existing_tessellation_fixture_main(void);
#define main pvrgpu_existing_tessellation_fixture_main
#include "pvrgpu_pco_tessellation_test.c"
#undef main

static nir_tex_instr *
shadow_gather_instruction(nir_builder *b, bool array, bool shadow, unsigned slot)
{
   nir_variable *sampler = nir_variable_create(b->shader, nir_var_uniform,
      glsl_sampler_type(GLSL_SAMPLER_DIM_2D, shadow, array, GLSL_TYPE_FLOAT), "image");
   sampler->data.descriptor_set = 0;
   sampler->data.binding = slot;
   nir_def *inputs = nir_load_uniform(b, 4, 32, nir_imm_int(b, 0),
      .base = 0, .range = 1, .dest_type = nir_type_float32);
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 1 + shadow);
   tex->op = nir_texop_tg4;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->is_array = array;
   tex->is_shadow = shadow;
   tex->coord_components = array ? 3 : 2;
   tex->texture_index = tex->sampler_index = slot;
   tex->dest_type = nir_type_float32;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
      nir_channels(b, inputs, array ? 7 : 3));
   if (shadow)
      tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_comparator,
         nir_channel(b, inputs, 3));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(b, &tex->instr);
   return tex;
}

static nir_shader *
shadow_gather_fragment(bool array, bool shadow, unsigned slot)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      pco_nir_options(), "private_array_shadow_gather");
   nir_variable *out = variable(b.shader, nir_var_shader_out, glsl_vec4_type(),
      "result", FRAG_RESULT_DATA0);
   nir_tex_instr *tex = shadow_gather_instruction(&b, array, shadow, slot);
   nir_store_var(&b, out, &tex->def, 15);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_tex_instr *
first_shadow_texture(nir_shader *s)
{
   nir_foreach_function_impl(impl, s)
      nir_foreach_block(block, impl)
         nir_foreach_instr(instr, block)
            if (instr->type == nir_instr_type_tex)
               return nir_instr_as_tex(instr);
   return NULL;
}

static void
shadow_gather_classifier_tests(void)
{
   require(!pvrgpu_pco_fragment_shadow_gather_only(NULL, 0, false), "null shader refused");
   require(!pvrgpu_pco_fragment_has_shadow_gather(NULL, 0), "no shader has no gather");
   for (unsigned array = 0; array < 2; ++array) {
      nir_shader *s = shadow_gather_fragment(array, true, 0);
      char *before = nir_shader_as_str(s, NULL);
      const struct shader_info info = s->info;
      require(pvrgpu_pco_fragment_shadow_gather_only(s, 0, array), "matching shadow gather proven");
      require(pvrgpu_pco_fragment_has_shadow_gather(s, 0), "shadow gather routes to strict state gate");
      require(!pvrgpu_pco_fragment_has_shadow_gather(s, 1), "unrelated slot has no shadow gather");
      require(!pvrgpu_pco_fragment_shadow_gather_only(s, 1, array), "unused slot not proven");
      require(!pvrgpu_pco_fragment_shadow_gather_only(s, 0, !array), "view dimension must match");
      require(!pvrgpu_pco_fragment_shadow_gather_only(s, PVRGPU_PCO_MAX_TEXTURES, array), "slot bound");
      char *after = nir_shader_as_str(s, NULL);
      require(strcmp(before, after) == 0 && memcmp(&info, &s->info, sizeof(info)) == 0,
              "proof is exactly read-only");
      ralloc_free(before); ralloc_free(after); ralloc_free(s);
   }
   /* Malformed shapes are checked only by the read-only classifier, never
    * passed to NIR optimizers or the instruction selector. */
   for (unsigned kind = 0; kind < 33; ++kind) {
      nir_shader *s = shadow_gather_fragment(true, true, 0);
      nir_tex_instr *tex = first_shadow_texture(s);
      require(tex != NULL, "negative fixture has a texture operation");
      nir_builder b = nir_builder_at(nir_before_instr(&tex->instr));
      switch (kind) {
      case 0: tex->component = 1; break;
      case 1: tex->is_sparse = true; break;
      case 2: tex->dest_type = nir_type_uint32; break;
      case 3: tex->def.bit_size = 16; break;
      case 4: tex->def.num_components = 1; break;
      case 5: tex->coord_components = 2; break;
      case 6: tex->is_gather_implicit_lod = true; break;
      case 7: tex->array_is_lowered_cube = true; break;
      case 8: tex->texture_non_uniform = true; break;
      case 9: tex->sampler_non_uniform = true; break;
      case 10: tex->embedded_sampler = true; break;
      case 11: tex->sampler_dim = GLSL_SAMPLER_DIM_CUBE; break;
      case 12: tex->is_shadow = false; break;
      case 13: tex->texture_index = 1; break;
      case 14: tex->sampler_index = 1; break;
      case 15: nir_tex_instr_remove_src(tex, 1); break;
      case 16: nir_tex_instr_add_src(tex, nir_tex_src_comparator, tex->src[1].src.ssa); break;
      case 17: nir_tex_instr_add_src(tex, nir_tex_src_coord, tex->src[0].src.ssa); break;
      case 18: nir_tex_instr_add_src(tex, nir_tex_src_offset, nir_imm_ivec2(&b, 1, 0)); break;
      case 19: nir_tex_instr_add_src(tex, nir_tex_src_lod, nir_imm_float(&b, 0)); break;
      case 20: nir_tex_instr_add_src(tex, nir_tex_src_texture_offset, nir_imm_int(&b, 0)); break;
      case 21: nir_tex_instr_add_src(tex, nir_tex_src_sampler_offset, nir_imm_int(&b, 0)); break;
      case 22: nir_tex_instr_add_src(tex, nir_tex_src_texture_handle, nir_imm_int64(&b, 0)); break;
      case 23: nir_tex_instr_add_src(tex, nir_tex_src_sampler_handle, nir_imm_int64(&b, 0)); break;
      case 24: nir_tex_instr_add_src(tex, nir_tex_src_backend1, nir_imm_int(&b, 0)); break;
      case 25: s->info.stage = MESA_SHADER_VERTEX; break;
      case 26: s->info.stage = MESA_SHADER_COMPUTE; break;
      case 27: s->info.stage = MESA_SHADER_GEOMETRY; break;
      case 28: s->info.stage = MESA_SHADER_TESS_CTRL; break;
      case 29: s->info.stage = MESA_SHADER_TESS_EVAL; break;
      case 30: tex->src[1].src.ssa->num_components = 2; break;
      case 31: tex->src[1].src.ssa->bit_size = 64; break;
      case 32: tex->tg4_offsets[0][0] = 2; break;
      }
      require(!pvrgpu_pco_fragment_shadow_gather_only(s, 0, true), "unsupported shape refused");
      require(pvrgpu_pco_fragment_has_shadow_gather(s, 0) ==
              (kind != 12 && !(kind >= 25 && kind <= 29)),
              "unsupported or aliased shadow gather cannot fall back to nearest");
      ralloc_free(s);
   }
   for (unsigned kind = 0; kind < 16; ++kind) {
      nir_shader *s = shadow_gather_fragment(true, true, 0);
      nir_tex_instr *first = first_shadow_texture(s);
      nir_builder b = nir_builder_at(nir_before_instr(&first->instr));
      nir_tex_instr *other = shadow_gather_instruction(&b, true, true, kind < 3 ? 0 : 1);
      bool expected = false;
      switch (kind) {
      case 0: other->op = nir_texop_tex; break; /* Same-slot ordinary shadow sample. */
      case 1: other->op = nir_texop_txs; expected = true; break;
      case 2: other->op = nir_texop_query_levels; expected = true; break;
      case 3: other->op = nir_texop_tex; expected = true; break; /* Resolved unrelated sample. */
      case 4: other->sampler_index = 0; break;
      case 5: other->texture_index = 0; break;
      case 6: nir_tex_instr_add_src(other, nir_tex_src_texture_offset, nir_imm_int(&b, 0)); break;
      case 7: nir_tex_instr_add_src(other, nir_tex_src_sampler_handle, nir_imm_int64(&b, 0)); break;
      case 8: other->texture_non_uniform = true; break;
      case 9: other->sampler_non_uniform = true; break;
      case 10: other->embedded_sampler = true; break;
      case 11: other->op = nir_texop_texture_samples; expected = true; break;
      case 12: nir_tex_instr_add_src(other, nir_tex_src_backend2, nir_imm_int(&b, 0)); break;
      case 13: nir_tex_instr_add_src(other, nir_tex_src_texture_heap_offset, nir_imm_int(&b, 0)); break;
      case 14: nir_tex_instr_add_src(other, nir_tex_src_sampler_heap_offset, nir_imm_int(&b, 0)); break;
      case 15: nir_tex_instr_add_src(other, nir_tex_src_texture_2_handle, nir_imm_int64(&b, 0)); break;
      }
      require(pvrgpu_pco_fragment_shadow_gather_only(s, 0, true) == expected,
              "mixed/unrelated slot use is classified conservatively");
      require(pvrgpu_pco_fragment_has_shadow_gather(s, 0),
              "mixed use still routes to strict gather-only proof");
      ralloc_free(s);
   }
   for (unsigned kind = 0; kind < 10; ++kind) {
      nir_shader *s = shadow_gather_fragment(true, true, 0);
      nir_tex_instr *tex = first_shadow_texture(s);
      nir_builder b = nir_builder_at(nir_before_instr(&tex->instr));
      nir_variable *var = nir_variable_create(s, nir_var_uniform,
         glsl_sampler_type(GLSL_SAMPLER_DIM_2D, true, true, GLSL_TYPE_FLOAT), "static_binding3");
      var->data.binding = 3;
      nir_deref_instr *deref = nir_build_deref_var(&b, var);
      nir_def *address = &deref->def;
      bool expected = kind < 3;
      if (kind == 1) { tex->sampler_index = 3; }
      if (kind == 2) { tex->texture_index = 3; }
      if (kind == 3) var->data.descriptor_set = 1;
      if (kind == 4) var->data.mode = nir_var_function_temp;
      if (kind == 5) var->type = glsl_float_type();
      if (kind == 6) {
         var->type = glsl_array_type(var->type, 2, 0);
         deref->type = var->type;
         address = &nir_build_deref_array_imm(&b, deref, 0)->def;
      }
      if (kind == 7) address = nir_imm_int64(&b, 3);
      if (kind != 2) nir_tex_instr_add_src(tex, nir_tex_src_texture_deref, address);
      if (kind != 1) nir_tex_instr_add_src(tex, nir_tex_src_sampler_deref, address);
      if (kind == 8) nir_tex_instr_add_src(tex, nir_tex_src_texture_deref, address);
      if (kind == 9) nir_tex_instr_add_src(tex, nir_tex_src_sampler_deref, address);
      require(pvrgpu_pco_fragment_shadow_gather_only(s, 3, true) == expected, "plain static deref proof");
      require(!pvrgpu_pco_fragment_shadow_gather_only(s, 0, true), "deref binding supersedes default zero indices");
      ralloc_free(s);
   }
   nir_shader *query = shadow_gather_fragment(true, true, 0);
   first_shadow_texture(query)->op = nir_texop_txs;
   require(!pvrgpu_pco_fragment_shadow_gather_only(query, 0, true), "query-only slot not proven gather");
   require(!pvrgpu_pco_fragment_has_shadow_gather(query, 0), "query-only is not a shadow gather");
   ralloc_free(query);
   for (unsigned kind = 0; kind < 5; ++kind) {
      nir_shader *s = shadow_gather_fragment(true, true, 1);
      nir_tex_instr *tex = first_shadow_texture(s);
      nir_builder b = nir_builder_at(nir_before_instr(&tex->instr));
      if (kind == 1) nir_tex_instr_add_src(tex, nir_tex_src_texture_offset, nir_imm_int(&b, 0));
      if (kind == 2) nir_tex_instr_add_src(tex, nir_tex_src_sampler_handle, nir_imm_int64(&b, 0));
      if (kind == 3) tex->sampler_index = 0;
      if (kind == 4) tex->texture_index = 0;
      require(pvrgpu_pco_fragment_has_shadow_gather(s, 0) == (kind != 0),
              "unresolved or cross-slot shadow gather conservatively aliases target");
      require(pvrgpu_pco_fragment_has_shadow_gather(s, 1), "actual slot still has gather");
      require(!pvrgpu_pco_fragment_has_shadow_gather(s, PVRGPU_PCO_MAX_TEXTURES), "routing slot bound");
      ralloc_free(s);
   }
   nir_shader *ordinary = shadow_gather_fragment(false, true, 0);
   first_shadow_texture(ordinary)->op = nir_texop_tex;
   require(!pvrgpu_pco_fragment_has_shadow_gather(ordinary, 0), "ordinary nearest-shadow route unchanged");
   ralloc_free(ordinary);
}

static void
compile_shadow_gather_fixture(struct pvrgpu_pco_compiler *compiler, unsigned kind)
{
   nir_shader *vs = make_vertex(true);
   nir_shader *fs = shadow_gather_fragment(kind == 1, kind != 0, 0);
   const enum pipe_format format = PIPE_FORMAT_R32G32_FLOAT;
   char error[2048] = {0};
   char *original = nir_shader_as_str(fs, NULL);
   const struct shader_info info = fs->info;
   struct pvrgpu_pco_graphics_binary binary = {0};
   require(pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format,
      false, false, 1, 0, 4, 1, 1, &binary, error, sizeof(error)), error);
   require(binary.fragment.abi.shareds == 24 &&
           binary.fragment.abi.push_constant_start == 20 &&
           binary.fragment.abi.push_constant_count == 4 &&
           binary.fragment.abi.uniform_buffer_descriptor_count == 0,
           "unchanged combined descriptor/CB0 ABI");
   save_stage(500 + kind, "vs", &binary.vertex);
   save_stage(500 + kind, "fs", &binary.fragment);
   char *after = nir_shader_as_str(fs, NULL);
   require(strcmp(original, after) == 0 && memcmp(&info, &fs->info, sizeof(info)) == 0,
           "compiler owns its clone, original remains reusable");
   ralloc_free(original); ralloc_free(after);
   pvrgpu_pco_graphics_binary_finish(&binary);
   ralloc_free(vs); ralloc_free(fs);
}

#ifndef PVRGPU_SHADOW_GATHER_NO_MAIN
int main(void)
{
   glsl_type_singleton_init_or_ref();
   shadow_gather_classifier_tests();
   char error[2048] = {0};
   struct pvrgpu_pco_compiler *compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
   require(compiler != NULL, error);
   for (unsigned kind = 0; kind < 3; ++kind)
      compile_shadow_gather_fixture(compiler, kind);
   pvrgpu_pco_compiler_destroy(compiler);
   glsl_type_singleton_decref();
   printf("shadow gather compiler PASS: %u checks\n", checks);
   return 0;
}
#endif
