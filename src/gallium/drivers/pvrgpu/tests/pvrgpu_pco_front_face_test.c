/* SPDX-License-Identifier: MIT */
/* Genuine compiler coverage; optional direct mode checks malformed NIR only
 * in the helper, never by passing invalid definitions to an optimizer. */
#ifdef PVRGPU_FRONT_FACE_DIRECT_HELPER
#include "../pvrgpu_pco.c"
#endif
int pvrgpu_existing_tessellation_fixture_main(void);
#define main pvrgpu_existing_tessellation_fixture_main
#include "pvrgpu_pco_tessellation_test.c"
#undef main

static nir_def *face_value(nir_builder *b, unsigned kind)
{
   if (kind == 2 || kind == 3)
      return nir_b2b1(b, nir_load_front_face(b, kind == 2 ? 1 : 32));
   nir_variable *face = variable(b->shader, nir_var_shader_in,
      glsl_uint_type(), "gl_FrontFacing", VARYING_SLOT_FACE);
   face->data.interpolation = INTERP_MODE_FLAT;
   return nir_b2b1(b, nir_load_var(b, face));
}

static nir_shader *face_fragment(unsigned kind)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      pco_nir_options(), "native_front_face_fixture");
   nir_variable *out = variable(b.shader, nir_var_shader_out, glsl_vec4_type(),
      "result", FRAG_RESULT_DATA0);
   nir_def *red = nir_imm_vec4(&b, 1, 0, 0, 1);
   nir_def *green = nir_imm_vec4(&b, 0, 1, 0, 1);
   nir_def *color = red;
   if (kind == 5)
      red = sample_texture(&b, nir_imm_vec2(&b, .25f, .5f), nir_imm_float(&b, 0));
   if (kind) {
      nir_def *front = face_value(&b, kind);
      if (kind == 4) {
         nir_if *branch = nir_push_if(&b, front);
         nir_def *a = sample_texture(&b, nir_imm_vec2(&b, .25f, .5f), nir_imm_float(&b, 0));
         nir_push_else(&b, branch);
         nir_def *c = sample_texture(&b, nir_imm_vec2(&b, .75f, .5f), nir_imm_float(&b, 0));
         nir_pop_if(&b, branch);
         color = nir_if_phi(&b, a, c);
      } else {
         color = nir_bcsel(&b, front, red, green);
      }
   }
   nir_store_var(&b, out, color, 15);
   nir_jump(&b, nir_jump_return);
   /* Both control-flow arms address one existing binding. The shared fixture
    * sampler builder creates a declaration per call, so retain exactly one. */
   nir_variable *sampler = NULL;
   nir_foreach_variable_with_modes_safe(var, b.shader, nir_var_uniform) {
      if (!glsl_type_is_sampler(var->type)) continue;
      if (sampler) exec_node_remove(&var->node);
      else sampler = var;
   }
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *face_geometry(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_GEOMETRY,
      pco_nir_options(), "front_face_geometry");
   b.shader->info.gs.input_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.output_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.vertices_in = b.shader->info.gs.vertices_out = 1;
   b.shader->info.gs.invocations = 1;
   nir_variable *in = variable(b.shader, nir_var_shader_in,
      glsl_array_type(glsl_vec4_type(), 1, 0), "position_in", VARYING_SLOT_POS);
   nir_variable *out = variable(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "gl_Position", VARYING_SLOT_POS);
   nir_store_var(&b, out, nir_load_deref(&b,
      nir_build_deref_array_imm(&b, nir_build_deref_var(&b, in), 0)), 15);
   nir_emit_vertex(&b, .stream_id = 0);
   nir_end_primitive(&b, .stream_id = 0);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

#ifdef PVRGPU_FRONT_FACE_DIRECT_HELPER
static void face_helper_tests(void)
{
   char error[512];
   /* The fast no-FACE path preserves every byte of shader_info and NIR. */
   nir_shader *plain = face_fragment(0);
   const struct shader_info info = plain->info;
   char *before = nir_shader_as_str(plain, NULL);
   require(pvrgpu_lower_fragment_front_face(plain, error, sizeof(error)), "no-FACE helper succeeds");
   char *after = nir_shader_as_str(plain, NULL);
   require(strcmp(before, after) == 0 && memcmp(&info, &plain->info, sizeof(info)) == 0,
           "no-FACE helper is an exact NIR/metadata no-op");
   ralloc_free(before); ralloc_free(after); ralloc_free(plain);

   for (unsigned kind = 0; kind < 23; ++kind) {
      nir_shader *fs = face_fragment(1);
      nir_variable *face = NULL;
      nir_foreach_shader_in_variable(var, fs)
         if (var->data.location == VARYING_SLOT_FACE) face = var;
      require(face != NULL, "negative fixture has actual scalar FACE");
      nir_function_impl *impl = nir_shader_get_entrypoint(fs);
      nir_intrinsic_instr *load = NULL;
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_deref)
               load = nir_instr_as_intrinsic(instr);
         }
      }
      require(load != NULL, "negative fixture has a direct load");
      nir_builder b = nir_builder_at(nir_before_instr(&load->instr));
      nir_deref_instr *deref = nir_src_as_deref(load->src[0]);
      switch (kind) {
      case 0: face->type = glsl_float_type(); break;
      case 1: face->type = glsl_uvec2_type(); break;
      case 2: face->type = glsl_array_type(glsl_uint_type(), 2, 0); break;
      case 3: face->data.location_frac = 1; break;
      case 4: face->data.interpolation = INTERP_MODE_SMOOTH; break;
      case 5: {
         nir_variable *other = variable(fs, nir_var_shader_in, glsl_uint_type(), "duplicate_face", VARYING_SLOT_FACE);
         other->data.interpolation = INTERP_MODE_FLAT;
         break;
      }
      case 6: load->def.bit_size = 16; break;
      case 7: load->def.bit_size = 64; break;
      case 8: load->def.num_components = 2; break;
      case 9: nir_store_deref(&b, deref, nir_imm_int(&b, 0), 1); break;
      case 10: (void)nir_iadd_imm(&b, &deref->def, 4); break;
      case 11:
         (void)nir_load_input(&b, 1, 32, nir_imm_int(&b, 0), .base = 0, .range = 1,
            .component = 0, .dest_type = nir_type_uint32,
            .io_semantics = {.location = VARYING_SLOT_FACE, .num_slots = 1});
         break;
      case 12: exec_node_remove(&face->node); break;
      case 13: face->data.mode = nir_var_system_value; face->data.location = SYSTEM_VALUE_FRONT_FACE; break;
      case 14: case 15: case 16: {
         nir_def *intr = nir_load_front_face(&b, 1);
         intr->bit_size = kind == 14 ? 16 : kind == 15 ? 64 : 1;
         if (kind == 16) intr->num_components = 2;
         break;
      }
      case 17: fs->info.stage = MESA_SHADER_VERTEX; break;
      case 18: fs->info.stage = MESA_SHADER_GEOMETRY; break;
      case 19: fs->info.stage = MESA_SHADER_TESS_CTRL; break;
      case 20: fs->info.stage = MESA_SHADER_TESS_EVAL; break;
      case 21: fs->info.stage = MESA_SHADER_COMPUTE; break;
      case 22: face->data.interpolation = INTERP_MODE_NOPERSPECTIVE; break;
      }
      /* These deliberately malformed shapes only enter the preflight helper.
       * They are never submitted to NIR validation, optimization or PCO. */
      before = nir_shader_as_str(fs, NULL);
      error[0] = '\0';
      require(!pvrgpu_lower_fragment_front_face(fs, error, sizeof(error)), "malformed/nonfragment FACE is rejected");
      require(strstr(error, "front-face") != NULL, "malformed FACE has a named refusal");
      after = nir_shader_as_str(fs, NULL);
      require(strcmp(before, after) == 0, "FACE preflight refuses without partial NIR changes");
      ralloc_free(before); ralloc_free(after); ralloc_free(fs);
   }

   for (unsigned kind = 0; kind < 3; ++kind) {
      nir_shader *fs = face_fragment(kind == 0 ? 4 : 1);
      if (kind) {
         nir_foreach_shader_in_variable(var, fs) var->type = glsl_bool_type();
         nir_foreach_function_impl(impl, fs) {
            nir_foreach_block(block, impl) {
               nir_foreach_instr(instr, block) {
                  if (instr->type == nir_instr_type_deref &&
                      nir_instr_as_deref(instr)->modes == nir_var_shader_in)
                     nir_instr_as_deref(instr)->type = glsl_bool_type();
                  if (kind == 2 && instr->type == nir_instr_type_intrinsic &&
                      nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_deref)
                     nir_instr_as_intrinsic(instr)->def.bit_size = 1;
               }
            }
         }
      }
      nir_variable_create(fs, nir_var_image,
         glsl_array_type(glsl_image_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_FLOAT), 3, 0),
         "retained_image_inventory");
      fs->info.num_ubos = 2; fs->info.num_ssbos = 1; fs->info.num_images = 3;
      require(pvrgpu_lower_fragment_front_face(fs, error, sizeof(error)), error);
      require(!(fs->info.inputs_read & VARYING_BIT_FACE) && fs->info.num_ubos == 2 &&
              fs->info.num_ssbos == 1 && fs->info.num_images == 3 &&
              fs->info.num_textures == (kind == 0 ? 1u : 0u),
              "only FACE IO disappears; every resource inventory remains unchanged");
      unsigned special = 0, bool32 = 0, op_uniform = 0;
      nir_foreach_function_impl(impl, fs) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_intrinsic) {
                  special += nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_face_ccw_pco;
                  op_uniform += nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_front_face_op_pco;
               } else if (instr->type == nir_instr_type_alu) {
                  bool32 += nir_instr_as_alu(instr)->op == nir_op_b2b32;
               }
            }
         }
      }
      require(special == 1 && op_uniform == 0 && bool32 == (kind == 2 ? 0u : 1u),
              "real special-register read and canonical bool32, never a winding-op SH load");
      ralloc_free(fs);
   }
}
#endif

int main(int argc, char **argv)
{
   const bool old = argc == 2 && strcmp(argv[1], "--expect-old-refusal") == 0;
   require(argc == 1 || old, "usage: front-face [--expect-old-refusal]");
   glsl_type_singleton_init_or_ref();
   char error[1024] = {0};
   struct pvrgpu_pco_compiler *compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
   require(compiler != NULL, error);
   const enum pipe_format format = PIPE_FORMAT_R32G32B32A32_FLOAT;
   for (unsigned pipeline = 0; pipeline < 3; ++pipeline) {
      for (unsigned kind = 0; kind < 6; ++kind) {
         /* Old intrinsic-only shaders get further than the varying guard and
          * cannot safely be passed into its unallocated front-face SH path. */
         if (old && (kind == 2 || kind == 3)) continue;
         nir_shader *vs = make_vertex(true), *fs = face_fragment(kind);
         nir_shader *gs = pipeline == 1 ? face_geometry() : NULL;
         nir_shader *tcs = pipeline == 2 ? make_control(1) : NULL;
         nir_shader *tes = pipeline == 2 ? make_evaluation(1) : NULL;
         char *original = nir_shader_as_str(fs, NULL);
         struct pvrgpu_pco_graphics_binary graphics = {0};
         struct pvrgpu_pco_geometry_pipeline_binary geometry = {0};
         struct pvrgpu_pco_tessellation_pipeline_binary tess = {0};
         const unsigned textures = kind >= 4 ? 1 : 0;
         const bool ok = pipeline == 0 ? pvrgpu_pco_compile_color_triangle(compiler,
            vs, fs, &format, false, false, 1, 0, 0, 1, textures, &graphics, error, sizeof(error)) :
            pipeline == 1 ? pvrgpu_pco_compile_geometry_pipeline(compiler, vs, gs, fs,
            &format, 1, 0, 0, 0, 1, textures, &geometry, error, sizeof(error)) :
            pvrgpu_pco_compile_tessellation_pipeline(compiler, vs, tcs, tes, fs,
            &format, 1, 0, 0, 0, 0, 1, textures, &tess, error, sizeof(error));
         printf("FRONT_FACE pipeline=%u kind=%u ok=%u error=%s\n", pipeline, kind, ok, error);
         char *after = nir_shader_as_str(fs, NULL);
         require(strcmp(original, after) == 0, "original fragment NIR is byte-identical after compilation");
         ralloc_free(original); ralloc_free(after);
         if (old && kind) {
            require(!ok && (strstr(error, "varyings are unsupported") ||
                            strstr(error, "consumer reads an unwritten interface")),
                    "old front-face admission refusal");
         } else {
            require(ok, error);
            struct pvrgpu_pco_graphics_binary *g = pipeline == 0 ? &graphics :
               pipeline == 1 ? &geometry.graphics : &tess.graphics;
            require(g->fragment.abi.shareds == textures * PVRGPU_PCO_TEXTURE_DESCRIPTOR_DWORDS,
                    "face does not allocate a shared uniform or shift descriptors");
            require(g->fragment_output_mask[0] == 15, "all four color channels remain live");
            save_stage(pipeline * 10 + kind, "fs", &g->fragment);
         }
         pvrgpu_pco_graphics_binary_finish(&graphics);
         pvrgpu_pco_geometry_pipeline_binary_finish(&geometry);
         pvrgpu_pco_tessellation_pipeline_binary_finish(&tess);
         ralloc_free(vs); ralloc_free(fs); ralloc_free(gs); ralloc_free(tcs); ralloc_free(tes);
      }
   }
#ifdef PVRGPU_FRONT_FACE_DIRECT_HELPER
   face_helper_tests();
#endif
   pvrgpu_pco_compiler_destroy(compiler);
   glsl_type_singleton_decref();
   printf("front-face compiler: %u checks PASS\n", checks);
   return 0;
}
