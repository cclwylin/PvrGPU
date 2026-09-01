/* SPDX-License-Identifier: MIT */

#include "pvrgpu_pco.h"

#include "nir/nir.h"
#include "nir/nir_builder.h"
#include "pco/pco.h"
#include "util/ralloc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *message)
{
   fprintf(stderr, "pvrgpu_pco_lowering_test: %s\n", message);
   exit(1);
}

static nir_shader *build_vertex_shader(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
                                                  pco_nir_options(),
                                                  "conditionals_gallium_vs");
   b.shader->info.internal = false;

   nir_variable *position_in = nir_variable_create(b.shader,
                                                   nir_var_shader_in,
                                                   glsl_vec4_type(),
                                                   "position");
   position_in->data.location = VERT_ATTRIB_GENERIC0;
   nir_variable *position_out = nir_variable_create(b.shader,
                                                    nir_var_shader_out,
                                                    glsl_vec4_type(),
                                                    "gl_Position");
   position_out->data.location = VARYING_SLOT_POS;
   position_out->data.precision = GLSL_PRECISION_MEDIUM;

   nir_def *position = nir_load_var(&b, position_in);
   nir_def *fract_x = nir_ffract(&b, nir_channel(&b, position, 0));
   nir_def *condition = nir_fge(&b, fract_x, nir_imm_float(&b, 0.5f));
   nir_def *twice = nir_ffract(
      &b, nir_fmul(&b, nir_imm_float(&b, 2.0f), fract_x));
   nir_def *thrice = nir_ffract(
      &b, nir_fmul(&b, nir_imm_float(&b, 3.0f), fract_x));
   nir_def *selected = nir_bcsel(&b, condition, twice, thrice);
   nir_def *adjusted_y =
      nir_fadd(&b,
               nir_channel(&b, position, 1),
               nir_fmul(&b,
                        nir_fmul(&b,
                                 nir_imm_float(&b, 0.1f),
                                 selected),
                        fract_x));

   nir_def *position_components[4] = {
      nir_channel(&b, position, 0),
      adjusted_y,
      nir_channel(&b, position, 2),
      nir_imm_float(&b, 1.0f),
   };
   nir_def *result = NULL;
   for (unsigned row = 0; row < 4; ++row) {
      /* Gallium load_uniform addresses CB0 in vec4 slots, not bytes. */
      nir_def *matrix_row = nir_load_uniform(&b,
                                             4,
                                             32,
                                             nir_imm_int(&b, row),
                                             .base = 0,
                                             .range = 4,
                                             .dest_type = nir_type_float32);
      nir_def *term = nir_fmul(&b, matrix_row, position_components[row]);
      result = result ? nir_fadd(&b, result, term) : term;
   }

   nir_store_var(&b, position_out, result, 0x0f);
   nir_jump(&b, nir_jump_return);
   return b.shader;
}

static nir_shader *build_fragment_shader(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                  pco_nir_options(),
                                                  "conditionals_gallium_fs");
   b.shader->info.internal = false;

   nir_variable *position_in = nir_variable_create(b.shader,
                                                   nir_var_shader_in,
                                                   glsl_vec4_type(),
                                                   "gl_FragCoord");
   position_in->data.location = VARYING_SLOT_POS;
   position_in->data.interpolation = INTERP_MODE_SMOOTH;
   nir_variable *color_out = nir_variable_create(b.shader,
                                                 nir_var_shader_out,
                                                 glsl_vec4_type(),
                                                 "gl_FragColor");
   color_out->data.location = FRAG_RESULT_COLOR;

   nir_def *frag_coord = nir_load_var(&b, position_in);
   nir_def *y_transform = nir_load_uniform(&b,
                                           4,
                                           32,
                                           nir_imm_int(&b, 0),
                                           .base = 0,
                                           .range = 1,
                                           .dest_type = nir_type_float32);
   nir_def *y = nir_fadd(&b,
                         nir_fmul(&b,
                                  nir_channel(&b, frag_coord, 1),
                                  nir_channel(&b, y_transform, 0)),
                         nir_channel(&b, y_transform, 1));
   /* Match the raw capture's non-associative float32 data flow exactly. */
   nir_def *scaled_x =
      nir_fmul_imm(&b, nir_channel(&b, frag_coord, 0), 0.0001f);
   nir_def *phase = nir_ffract(&b, nir_fmul(&b, scaled_x, y));
   nir_def *condition = nir_fge(&b, phase, nir_imm_float(&b, 0.5f));
   nir_def *twice = nir_ffract(&b, nir_fmul_imm(&b, phase, 2.0f));
   nir_def *thrice = nir_ffract(&b, nir_fmul_imm(&b, phase, 3.0f));
   nir_def *color = nir_bcsel(&b, condition, twice, thrice);

   nir_store_var(&b,
                 color_out,
                 nir_vec4(&b, color, color, color, nir_imm_float(&b, 1.0f)),
                 0x0f);
   nir_jump(&b, nir_jump_return);
   return b.shader;
}

static void set_source_hash(nir_shader *shader, const uint32_t words[8])
{
   for (unsigned word = 0; word < 8; ++word) {
      shader->info.source_blake3[word * 4 + 0] = words[word] >> 0;
      shader->info.source_blake3[word * 4 + 1] = words[word] >> 8;
      shader->info.source_blake3[word * 4 + 2] = words[word] >> 16;
      shader->info.source_blake3[word * 4 + 3] = words[word] >> 24;
   }
}

static nir_shader *build_texture_vertex_shader(void)
{
   static const uint32_t source_hash[8] = {
      UINT32_C(0x750ac3d1), UINT32_C(0xe9ceafcc),
      UINT32_C(0xdd1263dd), UINT32_C(0xa22a457b),
      UINT32_C(0x3b8ebb47), UINT32_C(0xa4ee0e8e),
      UINT32_C(0xeb2663ea), UINT32_C(0x6ad452cd),
   };
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
                                                  pco_nir_options(),
                                                  "GLSL6");
   b.shader->info.internal = false;
   b.shader->info.next_stage = MESA_SHADER_FRAGMENT;
   set_source_hash(b.shader, source_hash);

   nir_variable *mvp = nir_variable_create(b.shader,
                                            nir_var_uniform,
                                            glsl_matrix_type(GLSL_TYPE_FLOAT,
                                                             4,
                                                             4),
                                            "ModelViewProjectionMatrix");
   mvp->data.location = 0;
   mvp->data.precision = GLSL_PRECISION_HIGH;
   nir_variable *normal_matrix =
      nir_variable_create(b.shader,
                          nir_var_uniform,
                          glsl_matrix_type(GLSL_TYPE_FLOAT, 4, 4),
                          "NormalMatrix");
   normal_matrix->data.location = 1;
   normal_matrix->data.precision = GLSL_PRECISION_HIGH;

   nir_variable *position = nir_variable_create(b.shader,
                                                 nir_var_shader_in,
                                                 glsl_vec4_type(),
                                                 "position");
   position->data.location = VERT_ATTRIB_GENERIC0;
   position->data.precision = GLSL_PRECISION_HIGH;
   nir_variable *normal = nir_variable_create(b.shader,
                                               nir_var_shader_in,
                                               glsl_vec4_type(),
                                               "normal");
   normal->data.location = VERT_ATTRIB_GENERIC1;
   normal->data.precision = GLSL_PRECISION_HIGH;
   nir_variable *texcoord = nir_variable_create(b.shader,
                                                 nir_var_shader_in,
                                                 glsl_vec4_type(),
                                                 "texcoord");
   texcoord->data.location = VERT_ATTRIB_GENERIC2;
   texcoord->data.precision = GLSL_PRECISION_HIGH;

   nir_variable *position_out = nir_variable_create(b.shader,
                                                     nir_var_shader_out,
                                                     glsl_vec4_type(),
                                                     "gl_Position");
   position_out->data.location = VARYING_SLOT_POS;
   position_out->data.precision = GLSL_PRECISION_HIGH;
   nir_variable *varying_out =
      nir_variable_create(b.shader,
                          nir_var_shader_out,
                          glsl_vector_type(GLSL_TYPE_FLOAT, 3),
                          "TextureCoord");
   varying_out->data.location = VARYING_SLOT_VAR0;
   varying_out->data.precision = GLSL_PRECISION_MEDIUM;

   nir_def *row[4] = {
      nir_imm_int(&b, 0),
      nir_imm_int(&b, 1),
      nir_imm_int(&b, 2),
      nir_imm_int(&b, 3),
   };
   nir_def *light_direction =
      nir_imm_vec3(&b, 2.0f / 3.0f, 2.0f / 3.0f, 1.0f / 3.0f);

   nir_def *normal_value = nir_load_var(&b, normal);
   nir_def *normal_result = NULL;
   for (unsigned index = 0; index < 4; ++index) {
      nir_def *matrix_row = nir_load_uniform(&b,
                                             4,
                                             32,
                                             row[index],
                                             .base = 4,
                                             .range = 4,
                                             .dest_type = nir_type_float32);
      nir_def *term = index < 3
                         ? nir_fmul(&b,
                                    matrix_row,
                                    nir_channel(&b, normal_value, index))
                         : matrix_row;
      normal_result =
         normal_result ? nir_fadd(&b, normal_result, term) : term;
   }
   nir_def *normal_length2 = nir_fdot3(&b, normal_result, normal_result);
   nir_def *normal_scale = nir_frsq(&b, normal_length2);
   nir_def *normalized =
      nir_fmul(&b, nir_trim_vector(&b, normal_result, 3), normal_scale);
   nir_def *intensity = nir_fmax(
      &b,
      nir_fdot3(&b, normalized, light_direction),
      row[0]);

   nir_def *texcoord_value = nir_load_var(&b, texcoord);
   nir_def *position_value = nir_load_var(&b, position);
   nir_def *position_result = NULL;
   for (unsigned index = 0; index < 4; ++index) {
      nir_def *matrix_row = nir_load_uniform(&b,
                                             4,
                                             32,
                                             row[index],
                                             .base = 0,
                                             .range = 4,
                                             .dest_type = nir_type_float32);
      nir_def *term = index < 3
                         ? nir_fmul(&b,
                                    matrix_row,
                                    nir_channel(&b, position_value, index))
                         : matrix_row;
      position_result =
         position_result ? nir_fadd(&b, position_result, term) : term;
   }
   nir_store_var(&b, position_out, position_result, 0x0f);
   nir_store_var(&b,
                 varying_out,
                 nir_vec3(&b,
                          intensity,
                          nir_channel(&b, texcoord_value, 0),
                          nir_channel(&b, texcoord_value, 1)),
                 0x07);
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *build_texture_fragment_shader(
   unsigned varying_precision)
{
   static const uint32_t source_hash[8] = {
      UINT32_C(0xf95c6a3f), UINT32_C(0x2572cc32),
      UINT32_C(0x635afdeb), UINT32_C(0xff4d47cb),
      UINT32_C(0x5a3d3c87), UINT32_C(0x94e6645c),
      UINT32_C(0x9dde3b59), UINT32_C(0x233b4b47),
   };
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                  pco_nir_options(),
                                                  "GLSL6");
   b.shader->info.internal = false;
   b.shader->info.prev_stage = MESA_SHADER_VERTEX;
   set_source_hash(b.shader, source_hash);

   nir_variable *sampler = nir_variable_create(
      b.shader,
      nir_var_uniform,
      glsl_sampler_type(GLSL_SAMPLER_DIM_2D,
                        false,
                        false,
                        GLSL_TYPE_FLOAT),
      "MaterialTexture0");
   sampler->data.location = 2;
   sampler->data.binding = 0;
   sampler->data.precision = GLSL_PRECISION_LOW;
   nir_variable *varying =
      nir_variable_create(b.shader,
                          nir_var_shader_in,
                          glsl_vector_type(GLSL_TYPE_FLOAT, 3),
                          "TextureCoord");
   varying->data.location = VARYING_SLOT_VAR0;
   varying->data.interpolation = INTERP_MODE_SMOOTH;
   varying->data.precision = varying_precision;
   nir_variable *color_out = nir_variable_create(b.shader,
                                                  nir_var_shader_out,
                                                  glsl_vec4_type(),
                                                  "gl_FragColor");
   color_out->data.location = FRAG_RESULT_COLOR;
   color_out->data.precision = GLSL_PRECISION_MEDIUM;

   nir_def *varying_value = nir_load_var(&b, varying);
   nir_def *coord = nir_vec2(&b,
                             nir_channel(&b, varying_value, 1),
                             nir_channel(&b, varying_value, 2));
   nir_tex_instr *sample = nir_tex_instr_create(b.shader, 1);
   sample->op = nir_texop_tex;
   sample->sampler_dim = GLSL_SAMPLER_DIM_2D;
   sample->coord_components = 2;
   sample->dest_type = nir_type_float32;
   sample->texture_index = 0;
   sample->sampler_index = 0;
   sample->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   nir_def_init(&sample->instr, &sample->def, 4, 32);
   nir_builder_instr_insert(&b, &sample->instr);

   nir_def *one = nir_imm_float(&b, 1.0f);
   nir_def *light = nir_channel(&b, varying_value, 0);
   nir_def *intensity = nir_vec4(&b, light, light, light, one);
   nir_store_var(&b,
                 color_out,
                 nir_fmul(&b, &sample->def, intensity),
                 0x0f);
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_variable *
create_refract_matrix(nir_shader *nir, const char *name, unsigned location)
{
   nir_variable *matrix =
      nir_variable_create(nir,
                          nir_var_uniform,
                          glsl_matrix_type(GLSL_TYPE_FLOAT, 4, 4),
                          name);
   matrix->data.location = location;
   matrix->data.precision = GLSL_PRECISION_HIGH;
   return matrix;
}

static nir_variable *
create_refract_vector(nir_shader *nir,
                      nir_variable_mode mode,
                      const struct glsl_type *type,
                      const char *name,
                      int location,
                      unsigned precision)
{
   nir_variable *variable =
      nir_variable_create(nir, mode, type, name);
   variable->data.location = location;
   variable->data.precision = precision;
   return variable;
}

static nir_def *
build_refract_matrix_product(nir_builder *b,
                             nir_def *vector,
                             nir_def *row[4],
                             unsigned base)
{
   nir_def *result = NULL;
   for (unsigned index = 0; index < 4; ++index) {
      nir_def *matrix_row =
         nir_load_uniform(b,
                          4,
                          32,
                          row[index],
                          .base = base,
                          .range = 4,
                          .dest_type = nir_type_float32);
      nir_def *term =
         index < 3 ? nir_fmul(b,
                              matrix_row,
                              nir_channel(b, vector, index)) :
                     matrix_row;
      result = result ? nir_fadd(b, result, term) : term;
   }
   return result;
}

static nir_shader *
build_refract_prepass_vertex_shader(void)
{
   static const uint32_t source_hash[8] = {
      UINT32_C(0xe4968762), UINT32_C(0x33f7e5d3),
      UINT32_C(0x8ca90f67), UINT32_C(0xb97709d1),
      UINT32_C(0x1b1e6e02), UINT32_C(0x56181388),
      UINT32_C(0xe52466c0), UINT32_C(0xccbf4647),
   };
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
                                                  pco_nir_options(),
                                                  "GLSL29");
   b.shader->info.internal = false;
   b.shader->info.next_stage = MESA_SHADER_FRAGMENT;
   set_source_hash(b.shader, source_hash);

   create_refract_matrix(b.shader, "ModelViewProjectionMatrix", 0);
   nir_variable *position =
      create_refract_vector(b.shader,
                            nir_var_shader_in,
                            glsl_vec4_type(),
                            "position",
                            VERT_ATTRIB_GENERIC0,
                            GLSL_PRECISION_HIGH);
   nir_variable *normal =
      create_refract_vector(b.shader,
                            nir_var_shader_in,
                            glsl_vec4_type(),
                            "normal",
                            VERT_ATTRIB_GENERIC1,
                            GLSL_PRECISION_HIGH);
   nir_variable *position_out =
      create_refract_vector(b.shader,
                            nir_var_shader_out,
                            glsl_vec4_type(),
                            "gl_Position",
                            VARYING_SLOT_POS,
                            GLSL_PRECISION_HIGH);
   nir_variable *normal_out =
      create_refract_vector(b.shader,
                            nir_var_shader_out,
                            glsl_vector_type(GLSL_TYPE_FLOAT, 3),
                            "Normal",
                            VARYING_SLOT_VAR0,
                            GLSL_PRECISION_MEDIUM);

   nir_def *row[4] = {
      nir_imm_int(&b, 0),
      nir_imm_int(&b, 1),
      nir_imm_int(&b, 2),
      nir_imm_int(&b, 3),
   };
   nir_def *normal_value = nir_load_var(&b, normal);
   nir_def *position_value = nir_load_var(&b, position);
   nir_store_var(&b,
                 position_out,
                 build_refract_matrix_product(&b,
                                               position_value,
                                               row,
                                               0),
                 0x0f);
   nir_store_var(&b,
                 normal_out,
                 nir_vec3(&b,
                          nir_channel(&b, normal_value, 0),
                          nir_channel(&b, normal_value, 1),
                          nir_channel(&b, normal_value, 2)),
                 0x07);
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *
build_refract_prepass_fragment_shader(unsigned varying_precision)
{
   static const uint32_t source_hash[8] = {
      UINT32_C(0xe4296e61), UINT32_C(0x386f509f),
      UINT32_C(0x599bdacb), UINT32_C(0x3631c37e),
      UINT32_C(0x480c24bb), UINT32_C(0xd56abe19),
      UINT32_C(0x82725b7b), UINT32_C(0xd046dab0),
   };
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                  pco_nir_options(),
                                                  "GLSL29");
   b.shader->info.internal = false;
   b.shader->info.prev_stage = MESA_SHADER_VERTEX;
   set_source_hash(b.shader, source_hash);

   nir_variable *normal =
      create_refract_vector(b.shader,
                            nir_var_shader_in,
                            glsl_vector_type(GLSL_TYPE_FLOAT, 3),
                            "VARYING_SLOT_VAR0",
                            VARYING_SLOT_VAR0,
                            varying_precision);
   normal->data.interpolation = INTERP_MODE_SMOOTH;
   nir_variable *color_out =
      create_refract_vector(b.shader,
                            nir_var_shader_out,
                            glsl_vec4_type(),
                            "gl_FragColor",
                            FRAG_RESULT_COLOR,
                            GLSL_PRECISION_MEDIUM);
   nir_def *one = nir_imm_float(&b, 1.0f);
   nir_def *normal_value = nir_load_var(&b, normal);
   nir_store_var(&b,
                 color_out,
                 nir_vec4(&b,
                          nir_channel(&b, normal_value, 0),
                          nir_channel(&b, normal_value, 1),
                          nir_channel(&b, normal_value, 2),
                          one),
                 0x0f);
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *
build_refract_composite_vertex_shader(void)
{
   static const uint32_t source_hash[8] = {
      UINT32_C(0x27553973), UINT32_C(0x14f4744c),
      UINT32_C(0x155ef398), UINT32_C(0xa57dee0b),
      UINT32_C(0x7dd7d4b3), UINT32_C(0xcf1e855e),
      UINT32_C(0xe83f6f81), UINT32_C(0xdb7fe462),
   };
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
                                                  pco_nir_options(),
                                                  "GLSL6");
   b.shader->info.internal = false;
   b.shader->info.next_stage = MESA_SHADER_FRAGMENT;
   set_source_hash(b.shader, source_hash);

   create_refract_matrix(b.shader, "ModelViewProjectionMatrix", 0);
   create_refract_matrix(b.shader, "NormalMatrix", 1);
   create_refract_matrix(b.shader, "ModelViewMatrix", 2);
   create_refract_matrix(b.shader, "LightMatrix", 3);
   nir_variable *position =
      create_refract_vector(b.shader,
                            nir_var_shader_in,
                            glsl_vec4_type(),
                            "position",
                            VERT_ATTRIB_GENERIC0,
                            GLSL_PRECISION_HIGH);
   nir_variable *normal =
      create_refract_vector(b.shader,
                            nir_var_shader_in,
                            glsl_vec4_type(),
                            "normal",
                            VERT_ATTRIB_GENERIC1,
                            GLSL_PRECISION_HIGH);
   nir_variable *position_out =
      create_refract_vector(b.shader,
                            nir_var_shader_out,
                            glsl_vec4_type(),
                            "gl_Position",
                            VARYING_SLOT_POS,
                            GLSL_PRECISION_HIGH);
   nir_variable *vertex_position =
      create_refract_vector(b.shader,
                            nir_var_shader_out,
                            glsl_vec4_type(),
                            "vertex_position",
                            VARYING_SLOT_VAR0,
                            GLSL_PRECISION_MEDIUM);
   nir_variable *map_coord =
      create_refract_vector(b.shader,
                            nir_var_shader_out,
                            glsl_vec4_type(),
                            "MapCoord",
                            VARYING_SLOT_VAR1,
                            GLSL_PRECISION_MEDIUM);
   nir_variable *vertex_normal =
      create_refract_vector(b.shader,
                            nir_var_shader_out,
                            glsl_vector_type(GLSL_TYPE_FLOAT, 3),
                            "vertex_normal",
                            VARYING_SLOT_VAR2,
                            GLSL_PRECISION_MEDIUM);

   nir_def *row[4] = {
      nir_imm_int(&b, 0),
      nir_imm_int(&b, 1),
      nir_imm_int(&b, 2),
      nir_imm_int(&b, 3),
   };
   nir_def *position_value = nir_load_var(&b, position);
   nir_def *normal_value = nir_load_var(&b, normal);
   nir_def *normal_result =
      build_refract_matrix_product(&b, normal_value, row, 4);
   nir_def *normal_scale =
      nir_frsq(&b, nir_fdot3(&b, normal_result, normal_result));
   nir_def *normalized_normal =
      nir_fmul(&b, nir_trim_vector(&b, normal_result, 3), normal_scale);
   nir_def *view_position =
      build_refract_matrix_product(&b, position_value, row, 8);
   nir_def *light_position =
      build_refract_matrix_product(&b, position_value, row, 12);
   nir_def *clip_position =
      build_refract_matrix_product(&b, position_value, row, 0);

   nir_store_var(&b, position_out, clip_position, 0x0f);
   nir_store_var(&b, vertex_position, view_position, 0x0f);
   nir_store_var(&b, map_coord, light_position, 0x0f);
   nir_store_var(&b, vertex_normal, normalized_normal, 0x07);
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_def *
build_refract_texture_sample(nir_builder *b, unsigned slot, nir_def *coord)
{
   nir_tex_instr *sample = nir_tex_instr_create(b->shader, 1);
   sample->op = nir_texop_tex;
   sample->sampler_dim = GLSL_SAMPLER_DIM_2D;
   sample->coord_components = 2;
   sample->dest_type = nir_type_float32;
   sample->texture_index = slot;
   sample->sampler_index = slot;
   sample->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   nir_def_init(&sample->instr, &sample->def, 4, 32);
   nir_builder_instr_insert(b, &sample->instr);
   return &sample->def;
}

static void
create_refract_sampler(nir_shader *nir, const char *name, unsigned slot)
{
   nir_variable *sampler = nir_variable_create(
      nir,
      nir_var_uniform,
      glsl_sampler_type(GLSL_SAMPLER_DIM_2D, false, false, GLSL_TYPE_FLOAT),
      name);
   sampler->data.location = 4 + slot;
   sampler->data.binding = slot;
   sampler->data.precision = GLSL_PRECISION_LOW;
}

static nir_shader *
build_refract_composite_fragment_shader(unsigned varying_precision)
{
   static const uint32_t source_hash[8] = {
      UINT32_C(0xfcfc470a), UINT32_C(0x6e9eeb2b),
      UINT32_C(0x95810825), UINT32_C(0x6d2d6953),
      UINT32_C(0x04dc8732), UINT32_C(0x24b8c4c7),
      UINT32_C(0xac08a61d), UINT32_C(0x28ed4375),
   };
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                  pco_nir_options(),
                                                  "GLSL6");
   b.shader->info.internal = false;
   b.shader->info.prev_stage = MESA_SHADER_VERTEX;
   set_source_hash(b.shader, source_hash);

   create_refract_sampler(b.shader, "DistanceMap", 0);
   create_refract_sampler(b.shader, "NormalMap", 1);
   create_refract_sampler(b.shader, "ImageMap", 2);
   nir_variable *vertex_position =
      create_refract_vector(b.shader,
                            nir_var_shader_in,
                            glsl_vec4_type(),
                            "VARYING_SLOT_VAR0",
                            VARYING_SLOT_VAR0,
                            varying_precision);
   nir_variable *map_coord =
      create_refract_vector(b.shader,
                            nir_var_shader_in,
                            glsl_vec4_type(),
                            "VARYING_SLOT_VAR1",
                            VARYING_SLOT_VAR1,
                            varying_precision);
   nir_variable *vertex_normal =
      create_refract_vector(b.shader,
                            nir_var_shader_in,
                            glsl_vector_type(GLSL_TYPE_FLOAT, 3),
                            "VARYING_SLOT_VAR2",
                            VARYING_SLOT_VAR2,
                            varying_precision);
   vertex_position->data.interpolation = INTERP_MODE_SMOOTH;
   map_coord->data.interpolation = INTERP_MODE_SMOOTH;
   vertex_normal->data.interpolation = INTERP_MODE_SMOOTH;
   nir_variable *color_out =
      create_refract_vector(b.shader,
                            nir_var_shader_out,
                            glsl_vec4_type(),
                            "gl_FragColor",
                            FRAG_RESULT_COLOR,
                            GLSL_PRECISION_MEDIUM);

   nir_def *half = nir_imm_vec2(&b, 0.5f, 0.5f);
   nir_def *one = nir_imm_float(&b, 1.0f);
   nir_def *ior = nir_imm_float(&b, 1.2f);
   nir_def *zero = nir_imm_float(&b, 0.0f);
   nir_def *zero3 = nir_imm_vec3(&b, 0.0f, 0.0f, 0.0f);
   nir_def *inverse_ior = nir_imm_float(&b, 1.0f / 1.2f);
   nir_def *two = nir_imm_float(&b, 2.0f);
   nir_def *shininess = nir_imm_float(&b, 100.0f);

   nir_def *position = nir_load_var(&b, vertex_position);
   nir_def *negative_position =
      nir_fneg(&b, nir_trim_vector(&b, position, 3));
   nir_def *position_scale =
      nir_frsq(&b, nir_fdot3(&b, negative_position, negative_position));
   nir_def *incident =
      nir_fmul(&b, nir_trim_vector(&b, position, 3), position_scale);
   nir_def *negative_incident = nir_fneg(&b, incident);

   nir_def *normal = nir_load_var(&b, vertex_normal);
   nir_def *normal_scale = nir_frsq(&b, nir_fdot3(&b, normal, normal));
   nir_def *normalized_normal = nir_fmul(&b, normal, normal_scale);
   nir_def *dot_normal_incident =
      nir_fdot3(&b, normalized_normal, negative_incident);
   nir_def *dot_squared =
      nir_fmul(&b, dot_normal_incident, dot_normal_incident);
   nir_def *one_minus_dot_squared =
      nir_fadd(&b, one, nir_fneg(&b, dot_squared));
   nir_def *ior_squared = nir_imm_float(&b, 1.44f);
   nir_def *refract_root_argument =
      nir_fadd(&b,
               one,
               nir_fneg(&b,
                        nir_fmul(&b,
                                 one_minus_dot_squared,
                                 ior_squared)));
   nir_def *total_internal_reflection =
      nir_flt(&b, refract_root_argument, zero);
   nir_def *scaled_incident = nir_fmul(&b, incident, ior);
   nir_def *negative_scaled_incident = nir_fneg(&b, scaled_incident);
   nir_def *normal_weight =
      nir_fadd(&b,
               nir_fmul(&b, ior, dot_normal_incident),
               nir_fsqrt(&b, refract_root_argument));
   nir_def *weighted_normal =
      nir_fmul(&b, normal_weight, normalized_normal);
   nir_def *first_refraction =
      nir_fadd(&b, negative_scaled_incident, nir_fneg(&b, weighted_normal));
   first_refraction =
      nir_bcsel(&b, total_internal_reflection, zero3, first_refraction);

   nir_def *map = nir_load_var(&b, map_coord);
   nir_def *projected_map =
      nir_fdiv(&b,
               nir_trim_vector(&b, map, 3),
               nir_channel(&b, map, 3));
   nir_def *distance_vector = nir_fadd(&b, projected_map, first_refraction);
   nir_def *distance_coord =
      nir_fadd(&b,
               nir_fmul(&b, nir_trim_vector(&b, distance_vector, 2), half),
               half);
   nir_def *distance = build_refract_texture_sample(&b, 0, distance_coord);
   nir_def *displaced_position =
      nir_fadd(&b,
               nir_trim_vector(&b, position, 3),
               nir_fmul(&b,
                        first_refraction,
                        nir_channel(&b, distance, 0)));
   nir_def *normal_coord =
      nir_fadd(&b,
               nir_fmul(&b, nir_trim_vector(&b, displaced_position, 2), half),
               half);
   nir_def *normal_sample =
      build_refract_texture_sample(&b, 1, normal_coord);

   nir_def *exit_dot = nir_fdot3(&b, normal_sample, displaced_position);
   nir_def *exit_dot_squared = nir_fmul(&b, exit_dot, exit_dot);
   nir_def *exit_one_minus_dot_squared =
      nir_fadd(&b, one, nir_fneg(&b, exit_dot_squared));
   nir_def *inverse_ior_squared = nir_imm_float(&b, 1.0f / 1.44f);
   nir_def *exit_root_argument =
      nir_fadd(&b,
               one,
               nir_fneg(&b,
                        nir_fmul(&b,
                                 exit_one_minus_dot_squared,
                                 inverse_ior_squared)));
   nir_def *inverse_scaled_position =
      nir_fmul(&b, inverse_ior, displaced_position);
   nir_def *exit_normal_weight =
      nir_fadd(&b,
               nir_fmul(&b, inverse_ior, exit_dot),
               nir_fsqrt(&b, exit_root_argument));
   nir_def *exit_weighted_normal =
      nir_fmul(&b,
               exit_normal_weight,
               nir_trim_vector(&b, normal_sample, 3));
   nir_def *second_refraction =
      nir_fadd(&b,
               inverse_scaled_position,
               nir_fneg(&b, exit_weighted_normal));

   nir_def *view_offset =
      nir_imm_vec3(&b, -1.0f, -1.0f, -2.0f);
   nir_def *view =
      nir_fadd(&b,
               nir_fdiv(&b,
                        nir_trim_vector(&b, position, 3),
                        nir_channel(&b, position, 3)),
               view_offset);
   nir_def *view_scale = nir_frsq(&b, nir_fdot3(&b, view, view));
   nir_def *normalized_view = nir_fmul(&b, view, view_scale);
   nir_def *material = nir_imm_vec4(&b, 0.8f, 0.8f, 0.8f, 1.0f);
   nir_def *reflection_dot =
      nir_fdot3(&b, normalized_normal, normalized_view);
   nir_def *reflection_weight =
      nir_fmul(&b,
               nir_replicate(&b, reflection_dot, 3),
               nir_replicate(&b, two, 3));
   nir_def *reflection =
      nir_fadd(&b,
               normalized_view,
               nir_fneg(&b,
                        nir_fmul(&b,
                                 reflection_weight,
                                 normalized_normal)));
   nir_def *specular =
      nir_fmul(&b,
               material,
               nir_fpow(&b,
                         nir_fmax(&b,
                                  zero,
                                  nir_fdot3(&b,
                                            reflection,
                                            negative_incident)),
                         shininess));
   nir_def *image_coord =
      nir_fadd(&b,
               nir_fmul(&b, nir_trim_vector(&b, second_refraction, 2), half),
               half);
   nir_def *image = build_refract_texture_sample(&b, 2, image_coord);
   nir_store_var(&b, color_out, nir_fadd(&b, specular, image), 0x0f);

   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *
build_shadow_mask_vertex_shader(unsigned varying_precision)
{
   static const uint32_t source_hash[8] = {
      UINT32_C(0x981d39cf), UINT32_C(0x8316685f),
      UINT32_C(0x8e868a60), UINT32_C(0xf1850c32),
      UINT32_C(0xcdac5c34), UINT32_C(0x92c6393c),
      UINT32_C(0x6fc446df), UINT32_C(0x4830846f),
   };
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
                                                  pco_nir_options(),
                                                  "GLSL32");
   b.shader->info.internal = false;
   b.shader->info.next_stage = MESA_SHADER_FRAGMENT;
   set_source_hash(b.shader, source_hash);

   create_refract_matrix(b.shader, "LightMatrix", 0);
   create_refract_matrix(b.shader, "ModelViewProjectionMatrix", 1);
   nir_variable *position =
      create_refract_vector(b.shader,
                            nir_var_shader_in,
                            glsl_vec4_type(),
                            "position",
                            VERT_ATTRIB_GENERIC0,
                            GLSL_PRECISION_HIGH);
   nir_variable *position_out =
      create_refract_vector(b.shader,
                            nir_var_shader_out,
                            glsl_vec4_type(),
                            "gl_Position",
                            VARYING_SLOT_POS,
                            GLSL_PRECISION_HIGH);
   nir_variable *shadow_coord =
      create_refract_vector(b.shader,
                            nir_var_shader_out,
                            glsl_vec4_type(),
                            "ShadowCoord",
                            VARYING_SLOT_VAR0,
                            varying_precision);

   nir_def *row0 = nir_imm_int(&b, 0);
   nir_def *row1 = nir_imm_int(&b, 1);
   nir_def *row3 = nir_imm_int(&b, 3);
   nir_def *position_value = nir_load_var(&b, position);

   nir_def *light0 = nir_load_uniform(&b,
                                      4,
                                      32,
                                      row0,
                                      .base = 0,
                                      .range = 4,
                                      .dest_type = nir_type_float32);
   nir_def *light_x =
      nir_fmul(&b, light0, nir_channel(&b, position_value, 0));
   nir_def *light1 = nir_load_uniform(&b,
                                      4,
                                      32,
                                      row1,
                                      .base = 0,
                                      .range = 4,
                                      .dest_type = nir_type_float32);
   nir_def *light_y =
      nir_fmul(&b, light1, nir_channel(&b, position_value, 1));
   nir_def *light3 = nir_load_uniform(&b,
                                      4,
                                      32,
                                      row3,
                                      .base = 0,
                                      .range = 4,
                                      .dest_type = nir_type_float32);
   nir_def *light_result =
      nir_fadd(&b, nir_fadd(&b, light_x, light3), light_y);

   nir_def *mvp0 = nir_load_uniform(&b,
                                    4,
                                    32,
                                    row0,
                                    .base = 4,
                                    .range = 4,
                                    .dest_type = nir_type_float32);
   nir_def *mvp_x =
      nir_fmul(&b, mvp0, nir_channel(&b, position_value, 0));
   nir_def *mvp1 = nir_load_uniform(&b,
                                    4,
                                    32,
                                    row1,
                                    .base = 4,
                                    .range = 4,
                                    .dest_type = nir_type_float32);
   nir_def *mvp_y =
      nir_fmul(&b, mvp1, nir_channel(&b, position_value, 1));
   nir_def *mvp3 = nir_load_uniform(&b,
                                    4,
                                    32,
                                    row3,
                                    .base = 4,
                                    .range = 4,
                                    .dest_type = nir_type_float32);
   nir_def *mvp_result = nir_fadd(&b, nir_fadd(&b, mvp_x, mvp3), mvp_y);

   nir_store_var(&b, position_out, mvp_result, 0x0f);
   nir_store_var(&b, shadow_coord, light_result, 0x0f);
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *
build_shadow_mask_fragment_shader(unsigned varying_precision)
{
   static const uint32_t source_hash[8] = {
      UINT32_C(0x49a27748), UINT32_C(0xda81bbdf),
      UINT32_C(0x385da52a), UINT32_C(0xfa4c9087),
      UINT32_C(0x8edbc5ce), UINT32_C(0x0f9e5f75),
      UINT32_C(0xcb295bda), UINT32_C(0x20c14281),
   };
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                  pco_nir_options(),
                                                  "GLSL32");
   b.shader->info.internal = false;
   b.shader->info.prev_stage = MESA_SHADER_VERTEX;
   set_source_hash(b.shader, source_hash);

   nir_variable *sampler = nir_variable_create(
      b.shader,
      nir_var_uniform,
      glsl_sampler_type(GLSL_SAMPLER_DIM_2D, false, false, GLSL_TYPE_FLOAT),
      "ShadowMap");
   sampler->data.location = 2;
   sampler->data.binding = 0;
   sampler->data.precision = GLSL_PRECISION_LOW;
   nir_variable *shadow_coord =
      create_refract_vector(b.shader,
                            nir_var_shader_in,
                            glsl_vec4_type(),
                            "VARYING_SLOT_VAR0",
                            VARYING_SLOT_VAR0,
                            varying_precision);
   shadow_coord->data.interpolation = INTERP_MODE_SMOOTH;
   nir_variable *color_out =
      create_refract_vector(b.shader,
                            nir_var_shader_out,
                            glsl_vec4_type(),
                            "gl_FragColor",
                            FRAG_RESULT_COLOR,
                            GLSL_PRECISION_MEDIUM);

   nir_def *bias = nir_imm_float(&b, 0.1505f);
   nir_def *one = nir_imm_float(&b, 1.0f);
   nir_def *zero = nir_imm_float(&b, 0.0f);
   nir_def *coord = nir_load_var(&b, shadow_coord);
   nir_def *projected =
      nir_fdiv(&b, coord, nir_channel(&b, coord, 3));
   nir_def *compare_depth =
      nir_fadd(&b, nir_channel(&b, projected, 2), bias);
   nir_def *sample_coord = nir_trim_vector(&b, projected, 2);
   nir_def *sample = build_refract_texture_sample(&b, 0, sample_coord);
   nir_def *in_front = nir_flt(&b, zero, nir_channel(&b, coord, 3));
   nir_def *occluded =
      nir_flt(&b, nir_channel(&b, sample, 0), compare_depth);
   nir_def *shadowed = nir_iand(&b, in_front, occluded);
   nir_def *dark = nir_imm_vec3(&b, 0.15f, 0.15f, 0.15f);
   nir_def *light = nir_imm_vec3(&b, 0.3f, 0.3f, 0.3f);
   nir_def *rgb = nir_bcsel(&b, shadowed, dark, light);
   nir_store_var(&b,
                 color_out,
                 nir_vec4(&b,
                          nir_channel(&b, rgb, 0),
                          nir_channel(&b, rgb, 1),
                          nir_channel(&b, rgb, 2),
                          one),
                 0x0f);
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *
build_shadow_scene_vertex_shader(unsigned varying_precision)
{
   static const uint32_t source_hash[8] = {
      UINT32_C(0xda5546ad), UINT32_C(0x634bf23c),
      UINT32_C(0x36c9d6bd), UINT32_C(0x89e696ca),
      UINT32_C(0xc138e003), UINT32_C(0x445bb465),
      UINT32_C(0xb580e98a), UINT32_C(0x7f1d0c5b),
   };
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
                                                  pco_nir_options(),
                                                  "GLSL6");
   b.shader->info.internal = false;
   b.shader->info.next_stage = MESA_SHADER_FRAGMENT;
   set_source_hash(b.shader, source_hash);

   create_refract_matrix(b.shader, "ModelViewProjectionMatrix", 0);
   create_refract_matrix(b.shader, "NormalMatrix", 1);
   nir_variable *position =
      create_refract_vector(b.shader,
                            nir_var_shader_in,
                            glsl_vec4_type(),
                            "position",
                            VERT_ATTRIB_GENERIC0,
                            GLSL_PRECISION_HIGH);
   nir_variable *normal =
      create_refract_vector(b.shader,
                            nir_var_shader_in,
                            glsl_vec4_type(),
                            "normal",
                            VERT_ATTRIB_GENERIC1,
                            GLSL_PRECISION_HIGH);
   nir_variable *position_out =
      create_refract_vector(b.shader,
                            nir_var_shader_out,
                            glsl_vec4_type(),
                            "gl_Position",
                            VARYING_SLOT_POS,
                            GLSL_PRECISION_HIGH);
   nir_variable *color_out =
      create_refract_vector(b.shader,
                            nir_var_shader_out,
                            glsl_float_type(),
                            "Color",
                            VARYING_SLOT_VAR0,
                            varying_precision);

   nir_def *row[4] = {
      nir_imm_int(&b, 0),
      nir_imm_int(&b, 1),
      nir_imm_int(&b, 2),
      nir_imm_int(&b, 3),
   };
   nir_def *light_direction =
      nir_imm_vec3(&b, 0.0f, 0.8320503f, 0.5547002f);
   nir_def *normal_value = nir_load_var(&b, normal);
   nir_def *normal_result = NULL;
   for (unsigned index = 0; index < 4; ++index) {
      nir_def *matrix_row = nir_load_uniform(&b,
                                             4,
                                             32,
                                             row[index],
                                             .base = 4,
                                             .range = 4,
                                             .dest_type = nir_type_float32);
      nir_def *term = index < 3
                         ? nir_fmul(&b,
                                    matrix_row,
                                    nir_channel(&b, normal_value, index))
                         : matrix_row;
      normal_result =
         normal_result ? nir_fadd(&b, normal_result, term) : term;
   }
   nir_def *normal_scale =
      nir_frsq(&b, nir_fdot3(&b, normal_result, normal_result));
   nir_def *normalized =
      nir_fmul(&b, nir_trim_vector(&b, normal_result, 3), normal_scale);
   nir_def *intensity =
      nir_fmax(&b,
               nir_fdot2(&b,
                         nir_channels(&b, light_direction, 0x6),
                         nir_channels(&b, normalized, 0x6)),
               row[0]);

   nir_def *position_value = nir_load_var(&b, position);
   nir_def *position_result = NULL;
   for (unsigned index = 0; index < 4; ++index) {
      nir_def *matrix_row = nir_load_uniform(&b,
                                             4,
                                             32,
                                             row[index],
                                             .base = 0,
                                             .range = 4,
                                             .dest_type = nir_type_float32);
      nir_def *term = index < 3
                         ? nir_fmul(&b,
                                    matrix_row,
                                    nir_channel(&b, position_value, index))
                         : matrix_row;
      position_result =
         position_result ? nir_fadd(&b, position_result, term) : term;
   }
   nir_store_var(&b, position_out, position_result, 0x0f);
   nir_store_var(&b, color_out, intensity, 0x01);
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *
build_shadow_scene_fragment_shader(unsigned varying_precision)
{
   static const uint32_t source_hash[8] = {
      UINT32_C(0x8105bebf), UINT32_C(0x60cef3c7),
      UINT32_C(0xc9c3e978), UINT32_C(0xd20442bc),
      UINT32_C(0x46d83156), UINT32_C(0x9a4abb0b),
      UINT32_C(0xd1a4de24), UINT32_C(0x422a9790),
   };
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                  pco_nir_options(),
                                                  "GLSL6");
   b.shader->info.internal = false;
   b.shader->info.prev_stage = MESA_SHADER_VERTEX;
   set_source_hash(b.shader, source_hash);

   nir_variable *color =
      create_refract_vector(b.shader,
                            nir_var_shader_in,
                            glsl_float_type(),
                            "Color",
                            VARYING_SLOT_VAR0,
                            varying_precision);
   color->data.interpolation = INTERP_MODE_SMOOTH;
   nir_variable *color_out =
      create_refract_vector(b.shader,
                            nir_var_shader_out,
                            glsl_vec4_type(),
                            "gl_FragColor",
                            FRAG_RESULT_COLOR,
                            GLSL_PRECISION_MEDIUM);
   nir_def *value = nir_load_var(&b, color);
   nir_def *one = nir_imm_float(&b, 1.0f);
   nir_store_var(&b,
                 color_out,
                 nir_vec4(&b, value, value, value, one),
                 0x0f);
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *
build_terrain_fullscreen_vertex_shader(unsigned varying_precision)
{
   static const uint32_t source_hash[8] = {
      UINT32_C(0x64295326), UINT32_C(0xb7892f55),
      UINT32_C(0x40f4c2f4), UINT32_C(0x2524db37),
      UINT32_C(0x4a6849f7), UINT32_C(0x4212d7bb),
      UINT32_C(0x51ad07d2), UINT32_C(0x41c1b894),
   };
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
                                                  pco_nir_options(),
                                                  "GLSL41");
   b.shader->info.internal = false;
   b.shader->info.next_stage = MESA_SHADER_FRAGMENT;
   set_source_hash(b.shader, source_hash);

   nir_variable *uv_offset =
      create_refract_vector(b.shader,
                            nir_var_uniform,
                            glsl_vec2_type(),
                            "uvOffset",
                            0,
                            GLSL_PRECISION_HIGH);
   nir_variable *uv_scale =
      create_refract_vector(b.shader,
                            nir_var_uniform,
                            glsl_vec2_type(),
                            "uvScale",
                            1,
                            GLSL_PRECISION_MEDIUM);
   (void)uv_offset;
   (void)uv_scale;
   nir_variable *position =
      create_refract_vector(b.shader,
                            nir_var_shader_in,
                            glsl_vec4_type(),
                            "position",
                            VERT_ATTRIB_GENERIC0,
                            GLSL_PRECISION_HIGH);
   nir_variable *position_out =
      create_refract_vector(b.shader,
                            nir_var_shader_out,
                            glsl_vec4_type(),
                            "gl_Position",
                            VARYING_SLOT_POS,
                            GLSL_PRECISION_HIGH);
   nir_variable *uv_out =
      create_refract_vector(b.shader,
                            nir_var_shader_out,
                            glsl_vec2_type(),
                            "vUv",
                            VARYING_SLOT_VAR0,
                            varying_precision);

   nir_def *half = nir_imm_float(&b, 0.5f);
   nir_def *one = nir_imm_float(&b, 1.0f);
   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *scale = nir_load_uniform(&b,
                                     2,
                                     32,
                                     zero,
                                     .base = 1,
                                     .range = 1,
                                     .dest_type = nir_type_float32);
   nir_def *position_value = nir_load_var(&b, position);
   nir_def *centered =
      nir_fadd(&b,
               nir_fmul(&b, nir_trim_vector(&b, position_value, 2), half),
               half);
   nir_def *scaled = nir_fmul(&b, scale, centered);
   nir_def *offset = nir_load_uniform(&b,
                                      2,
                                      32,
                                      zero,
                                      .base = 0,
                                      .range = 1,
                                      .dest_type = nir_type_float32);
   nir_def *uv = nir_fadd(&b, scaled, offset);
   nir_store_var(&b,
                 position_out,
                 nir_vec4(&b,
                          nir_channel(&b, position_value, 0),
                          nir_channel(&b, position_value, 1),
                          nir_channel(&b, position_value, 2),
                          one),
                 0x0f);
   nir_store_var(&b, uv_out, uv, 0x03);
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *
build_terrain_d6_fragment_shader(unsigned varying_precision)
{
   static const uint32_t source_hash[8] = {
      UINT32_C(0x0e6f1ac2), UINT32_C(0xd8813568),
      UINT32_C(0x336936c8), UINT32_C(0xc6840a8c),
      UINT32_C(0xdd9ba6ee), UINT32_C(0x627d4cf7),
      UINT32_C(0x88cb830e), UINT32_C(0xd538aa03),
   };
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                  pco_nir_options(),
                                                  "GLSL41");
   b.shader->info.internal = false;
   b.shader->info.prev_stage = MESA_SHADER_VERTEX;
   set_source_hash(b.shader, source_hash);

   create_refract_vector(b.shader,
                         nir_var_uniform,
                         glsl_float_type(),
                         "opacity",
                         2,
                         GLSL_PRECISION_MEDIUM);
   nir_variable *sampler = nir_variable_create(
      b.shader,
      nir_var_uniform,
      glsl_sampler_type(GLSL_SAMPLER_DIM_2D, false, false, GLSL_TYPE_FLOAT),
      "tDiffuse");
   sampler->data.location = 3;
   sampler->data.binding = 0;
   sampler->data.precision = GLSL_PRECISION_LOW;
   nir_variable *uv =
      create_refract_vector(b.shader,
                            nir_var_shader_in,
                            glsl_vec2_type(),
                            "VARYING_SLOT_VAR0",
                            VARYING_SLOT_VAR0,
                            varying_precision);
   uv->data.interpolation = INTERP_MODE_SMOOTH;
   nir_variable *color_out =
      create_refract_vector(b.shader,
                            nir_var_shader_out,
                            glsl_vec4_type(),
                            "gl_FragColor",
                            FRAG_RESULT_COLOR,
                            GLSL_PRECISION_MEDIUM);

   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *opacity = nir_load_uniform(&b,
                                       1,
                                       32,
                                       zero,
                                       .base = 0,
                                       .range = 1,
                                       .dest_type = nir_type_float32);
   nir_def *coord = nir_load_var(&b, uv);
   nir_def *sample = build_refract_texture_sample(&b, 0, coord);
   nir_store_var(&b, color_out, nir_fmul(&b, opacity, sample), 0x0f);
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *
build_terrain_d2_fragment_shader(unsigned varying_precision)
{
   static const uint32_t source_hash[8] = {
      UINT32_C(0x401f2183), UINT32_C(0xbf28b09d),
      UINT32_C(0x8f3006aa), UINT32_C(0x9770f1a5),
      UINT32_C(0x183eb0a6), UINT32_C(0x21c3909c),
      UINT32_C(0x7ebceac5), UINT32_C(0x63eeefc1),
   };
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                  pco_nir_options(),
                                                  "GLSL29");
   b.shader->info.internal = false;
   b.shader->info.prev_stage = MESA_SHADER_VERTEX;
   set_source_hash(b.shader, source_hash);

   create_refract_vector(b.shader,
                         nir_var_uniform,
                         glsl_float_type(),
                         "height",
                         2,
                         GLSL_PRECISION_MEDIUM);
   create_refract_vector(b.shader,
                         nir_var_uniform,
                         glsl_vec2_type(),
                         "resolution",
                         3,
                         GLSL_PRECISION_MEDIUM);
   nir_variable *sampler = nir_variable_create(
      b.shader,
      nir_var_uniform,
      glsl_sampler_type(GLSL_SAMPLER_DIM_2D, false, false, GLSL_TYPE_FLOAT),
      "heightMap");
   sampler->data.location = 4;
   sampler->data.binding = 0;
   sampler->data.precision = GLSL_PRECISION_LOW;
   nir_variable *uv =
      create_refract_vector(b.shader,
                            nir_var_shader_in,
                            glsl_vec2_type(),
                            "VARYING_SLOT_VAR0",
                            VARYING_SLOT_VAR0,
                            varying_precision);
   uv->data.interpolation = INTERP_MODE_SMOOTH;
   nir_variable *color_out =
      create_refract_vector(b.shader,
                            nir_var_shader_out,
                            glsl_vec4_type(),
                            "gl_FragColor",
                            FRAG_RESULT_COLOR,
                            GLSL_PRECISION_MEDIUM);

   nir_def *zero = nir_imm_float(&b, 0.0f);
   nir_def *one = nir_imm_float(&b, 1.0f);
   nir_def *half = nir_imm_float(&b, 0.5f);
   nir_def *coord = nir_load_var(&b, uv);
   nir_def *center = build_refract_texture_sample(&b, 0, coord);
   nir_def *resolution = nir_load_uniform(&b,
                                          2,
                                          32,
                                          zero,
                                          .base = 1,
                                          .range = 1,
                                          .dest_type = nir_type_float32);
   nir_def *du = nir_frcp(&b, nir_channel(&b, resolution, 0));
   nir_def *dv = nir_frcp(&b, nir_channel(&b, resolution, 1));
   nir_def *coord_u = nir_fadd(&b, coord, nir_vec2(&b, du, zero));
   nir_def *sample_u = build_refract_texture_sample(&b, 0, coord_u);
   nir_def *normal_x =
      nir_fadd(&b,
               nir_channel(&b, center, 0),
               nir_fneg(&b, nir_channel(&b, sample_u, 0)));
   nir_def *coord_v = nir_fadd(&b, coord, nir_vec2(&b, zero, dv));
   nir_def *sample_v = build_refract_texture_sample(&b, 0, coord_v);
   nir_def *normal_y =
      nir_fadd(&b,
               nir_channel(&b, center, 0),
               nir_fneg(&b, nir_channel(&b, sample_v, 0)));
   nir_def *height = nir_load_uniform(&b,
                                      1,
                                      32,
                                      zero,
                                      .base = 0,
                                      .range = 1,
                                      .dest_type = nir_type_float32);
   nir_def *normal = nir_vec3(&b, normal_x, normal_y, height);
   nir_def *scale = nir_frsq(&b, nir_fdot3(&b, normal, normal));
   nir_def *color = nir_fadd(&b,
                             nir_fmul(&b,
                                      nir_fmul(&b, normal, half),
                                      scale),
                             half);
   nir_store_var(&b,
                 color_out,
                 nir_vec4(&b,
                          nir_channel(&b, color, 0),
                          nir_channel(&b, color, 1),
                          nir_channel(&b, color, 2),
                          one),
                 0x0f);
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_variable *
create_terrain_sampler(nir_shader *nir,
                       const char *name,
                       unsigned location,
                       unsigned binding)
{
   nir_variable *sampler = nir_variable_create(
      nir,
      nir_var_uniform,
      glsl_sampler_type(GLSL_SAMPLER_DIM_2D, false, false, GLSL_TYPE_FLOAT),
      name);
   sampler->data.location = location;
   sampler->data.binding = binding;
   sampler->data.driver_location = binding;
   sampler->data.precision = GLSL_PRECISION_LOW;
   return sampler;
}

static nir_variable *
create_terrain_uniform(nir_shader *nir,
                       const struct glsl_type *type,
                       const char *name,
                       unsigned location,
                       unsigned driver_location,
                       unsigned precision)
{
   nir_variable *uniform = create_refract_vector(nir,
                                                  nir_var_uniform,
                                                  type,
                                                  name,
                                                  location,
                                                  precision);
   uniform->data.driver_location = driver_location;
   return uniform;
}

static nir_shader *
build_terrain_d3_vertex_shader(unsigned varying_precision)
{
   static const uint32_t source_hash[8] = {
      UINT32_C(0xfdbd1d43), UINT32_C(0x307b26eb),
      UINT32_C(0x2982014c), UINT32_C(0x02f9dd89),
      UINT32_C(0x6199f3cb), UINT32_C(0x8ba2d70d),
      UINT32_C(0xbbb04d5f), UINT32_C(0x6621cac2),
   };
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
                                                  pco_nir_options(),
                                                  "GLSL32");
   b.shader->info.internal = false;
   b.shader->info.next_stage = MESA_SHADER_FRAGMENT;
   set_source_hash(b.shader, source_hash);

   nir_variable *model_view_matrix =
      create_refract_matrix(b.shader, "modelViewMatrix", 0);
   model_view_matrix->data.driver_location = 0;
   nir_variable *normal_matrix =
      create_refract_matrix(b.shader, "normalMatrix", 1);
   normal_matrix->data.driver_location = 4;
   nir_variable *projection_matrix =
      create_refract_matrix(b.shader, "projectionMatrix", 2);
   projection_matrix->data.driver_location = 8;
   create_terrain_sampler(b.shader, "tNormal", 3, 0);
   create_terrain_sampler(b.shader, "tDisplacement", 4, 1);
   create_terrain_uniform(b.shader,
                          glsl_float_type(),
                          "uDisplacementScale",
                          5,
                          12,
                          GLSL_PRECISION_HIGH);
   create_terrain_uniform(b.shader,
                          glsl_float_type(),
                          "uDisplacementBias",
                          6,
                          13,
                          GLSL_PRECISION_HIGH);

   nir_variable *position =
      create_refract_vector(b.shader,
                            nir_var_shader_in,
                            glsl_vec4_type(),
                            "position",
                            VERT_ATTRIB_GENERIC0,
                            GLSL_PRECISION_HIGH);
   position->data.driver_location = 0;
   nir_variable *normal =
      create_refract_vector(b.shader,
                            nir_var_shader_in,
                            glsl_vec4_type(),
                            "normal",
                            VERT_ATTRIB_GENERIC1,
                            GLSL_PRECISION_HIGH);
   normal->data.driver_location = 1;
   nir_variable *tangent =
      create_refract_vector(b.shader,
                            nir_var_shader_in,
                            glsl_vec4_type(),
                            "tangent",
                            VERT_ATTRIB_GENERIC2,
                            GLSL_PRECISION_HIGH);
   tangent->data.driver_location = 2;
   nir_variable *uv =
      create_refract_vector(b.shader,
                            nir_var_shader_in,
                            glsl_vec4_type(),
                            "uv",
                            VERT_ATTRIB_GENERIC3,
                            GLSL_PRECISION_HIGH);
   uv->data.driver_location = 3;
   nir_variable *position_out =
      create_refract_vector(b.shader,
                            nir_var_shader_out,
                            glsl_vec4_type(),
                            "gl_Position",
                            VARYING_SLOT_POS,
                            GLSL_PRECISION_HIGH);
   position_out->data.driver_location = 0;
   position_out->data.interpolation = INTERP_MODE_NONE;
   nir_variable *varying0 =
      create_refract_vector(b.shader,
                            nir_var_shader_out,
                            glsl_vec4_type(),
                            "packed:vTangent.xy,vUv",
                            VARYING_SLOT_VAR0,
                            varying_precision);
   varying0->data.driver_location = 1;
   varying0->data.interpolation = INTERP_MODE_NONE;
   nir_variable *varying1 =
      create_refract_vector(b.shader,
                            nir_var_shader_out,
                            glsl_vec4_type(),
                            "packed:vTangent.z,vBinormal",
                            VARYING_SLOT_VAR1,
                            varying_precision);
   varying1->data.driver_location = 2;
   varying1->data.interpolation = INTERP_MODE_NONE;
   nir_variable *varying2 =
      create_refract_vector(b.shader,
                            nir_var_shader_out,
                            glsl_vec4_type(),
                            "packed:vNormal,vViewPosition.x",
                            VARYING_SLOT_VAR2,
                            varying_precision);
   varying2->data.driver_location = 3;
   varying2->data.interpolation = INTERP_MODE_NONE;
   nir_variable *varying3 =
      create_refract_vector(b.shader,
                            nir_var_shader_out,
                            glsl_vec2_type(),
                            "packed:vViewPosition.yz",
                            VARYING_SLOT_VAR3,
                            varying_precision);
   varying3->data.driver_location = 4;
   varying3->data.interpolation = INTERP_MODE_NONE;

   nir_def *row[4] = {
      nir_imm_int(&b, 0),
      nir_imm_int(&b, 1),
      nir_imm_int(&b, 2),
      nir_imm_int(&b, 3),
   };
   nir_def *two = nir_imm_float(&b, 2.0f);
   nir_def *position_value = nir_load_var(&b, position);
   nir_def *model_view_row[4];
   nir_def *view_position = NULL;
   for (unsigned index = 0; index < 4; ++index) {
      model_view_row[index] = nir_load_uniform(&b,
                                               4,
                                               32,
                                               row[index],
                                               .base = 0,
                                               .range = 4,
                                               .dest_type = nir_type_float32);
      nir_def *term = index < 3U ?
         nir_fmul(&b,
                  model_view_row[index],
                  nir_channel(&b, position_value, index)) :
         model_view_row[index];
      view_position = view_position ? nir_fadd(&b, view_position, term) : term;
   }
   b.fp_math_ctrl = nir_fp_preserve_inf;
   nir_def *negative_view_position =
      nir_fneg(&b, nir_trim_vector(&b, view_position, 3));
   b.fp_math_ctrl = nir_fp_fast_math;

   nir_def *normal_value = nir_load_var(&b, normal);
   nir_def *normal_matrix_row[4];
   nir_def *transformed_normal = NULL;
   for (unsigned index = 0; index < 4; ++index) {
      normal_matrix_row[index] = nir_load_uniform(
         &b,
         4,
         32,
         row[index],
         .base = 4,
         .range = 4,
         .dest_type = nir_type_float32);
      nir_def *term = index < 3U ?
         nir_fmul(&b,
                  normal_matrix_row[index],
                  nir_channel(&b, normal_value, index)) :
         normal_matrix_row[index];
      transformed_normal = transformed_normal ?
         nir_fadd(&b, transformed_normal, term) : term;
   }
   nir_def *normal_scale =
      nir_frsq(&b, nir_fdot3(&b, transformed_normal, transformed_normal));
   nir_def *normalized_normal =
      nir_fmul(&b,
               nir_trim_vector(&b, transformed_normal, 3),
               normal_scale);

   nir_def *tangent_value = nir_load_var(&b, tangent);
   nir_def *transformed_tangent = NULL;
   for (unsigned index = 0; index < 4; ++index) {
      nir_def *term = index < 3U ?
         nir_fmul(&b,
                  normal_matrix_row[index],
                  nir_channel(&b, tangent_value, index)) :
         normal_matrix_row[index];
      transformed_tangent = transformed_tangent ?
         nir_fadd(&b, transformed_tangent, term) : term;
   }
   nir_def *tangent_scale =
      nir_frsq(&b, nir_fdot3(&b, transformed_tangent, transformed_tangent));
   nir_def *normalized_tangent =
      nir_fmul(&b,
               nir_trim_vector(&b, transformed_tangent, 3),
               tangent_scale);
   nir_def *cross_a = nir_fmul(&b,
                               nir_swizzle(&b,
                                           normalized_normal,
                                           (unsigned[]){ 1, 2, 0 },
                                           3),
                               nir_swizzle(&b,
                                           normalized_tangent,
                                           (unsigned[]){ 2, 0, 1 },
                                           3));
   nir_def *cross_b = nir_fmul(&b,
                               nir_swizzle(&b,
                                           normalized_normal,
                                           (unsigned[]){ 2, 0, 1 },
                                           3),
                               nir_swizzle(&b,
                                           normalized_tangent,
                                           (unsigned[]){ 1, 2, 0 },
                                           3));
   nir_def *binormal = nir_fadd(&b, cross_a, nir_fneg(&b, cross_b));
   nir_def *binormal_scale = nir_frsq(&b, nir_fdot3(&b, binormal, binormal));
   nir_def *normalized_binormal = nir_fmul(&b, binormal, binormal_scale);

   nir_def *uv_value = nir_load_var(&b, uv);
   nir_def *coord = nir_mov(&b, nir_trim_vector(&b, uv_value, 2));
   nir_def *displacement_scale = nir_load_uniform(
      &b,
      1,
      32,
      row[0],
      .base = 12,
      .range = 1,
      .dest_type = nir_type_float32);
   nir_def *displacement_sample = build_refract_texture_sample(&b, 1, coord);
   nir_def *displacement =
      nir_fmul(&b, displacement_scale, nir_channel(&b, displacement_sample, 0));
   nir_def *displacement_bias = nir_load_uniform(
      &b,
      1,
      32,
      row[0],
      .base = 13,
      .range = 1,
      .dest_type = nir_type_float32);
   displacement = nir_fadd(&b, displacement, displacement_bias);
   nir_def *displacement_vector =
      nir_fmul(&b, normalized_normal, displacement);
   nir_def *displaced_position =
      nir_fadd(&b,
               nir_vec4(&b,
                        nir_channel(&b, displacement_vector, 0),
                        nir_channel(&b, displacement_vector, 1),
                        nir_channel(&b, displacement_vector, 2),
                        row[0]),
               view_position);

   nir_def *clip_position = NULL;
   for (unsigned index = 0; index < 4; ++index) {
      nir_def *projection_row = nir_load_uniform(
         &b,
         4,
         32,
         row[index],
         .base = 8,
         .range = 4,
         .dest_type = nir_type_float32);
      nir_def *term =
         nir_fmul(&b,
                  projection_row,
                  nir_channel(&b, displaced_position, index));
      clip_position = clip_position ? nir_fadd(&b, clip_position, term) : term;
   }

   nir_def *normal_sample = build_refract_texture_sample(&b, 0, coord);
   nir_def *object_normal =
      nir_fadd(&b,
               nir_fmul(&b, nir_trim_vector(&b, normal_sample, 3), two),
               nir_imm_vec3(&b, -1.0f, -1.0f, -1.0f));
   nir_def *mapped_normal = NULL;
   for (unsigned index = 0; index < 4; ++index) {
      nir_def *term = index < 3U ?
         nir_fmul(&b,
                  normal_matrix_row[index],
                  nir_channel(&b, object_normal, index)) :
         normal_matrix_row[index];
      mapped_normal = mapped_normal ? nir_fadd(&b, mapped_normal, term) : term;
   }

   nir_store_var(&b, position_out, clip_position, 0x0f);
   nir_store_var(&b,
                 varying0,
                 nir_vec4(&b,
                          nir_channel(&b, uv_value, 0),
                          nir_channel(&b, uv_value, 1),
                          nir_channel(&b, normalized_tangent, 0),
                          nir_channel(&b, normalized_tangent, 1)),
                 0x0f);
   nir_store_var(&b,
                 varying1,
                 nir_vec4(&b,
                          nir_channel(&b, normalized_tangent, 2),
                          nir_channel(&b, normalized_binormal, 0),
                          nir_channel(&b, normalized_binormal, 1),
                          nir_channel(&b, normalized_binormal, 2)),
                 0x0f);
   nir_store_var(&b,
                 varying2,
                 nir_vec4(&b,
                          nir_channel(&b, mapped_normal, 0),
                          nir_channel(&b, mapped_normal, 1),
                          nir_channel(&b, mapped_normal, 2),
                          nir_channel(&b, negative_view_position, 0)),
                 0x0f);
   nir_store_var(&b,
                 varying3,
                 nir_vec2(&b,
                          nir_channel(&b, negative_view_position, 1),
                          nir_channel(&b, negative_view_position, 2)),
                 0x03);
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *
build_terrain_d3_fragment_shader(unsigned varying_precision)
{
   static const uint32_t source_hash[8] = {
      UINT32_C(0x1a5275b3), UINT32_C(0xa5aed1dd),
      UINT32_C(0x0d1c076a), UINT32_C(0x1839fb95),
      UINT32_C(0x31c35861), UINT32_C(0x717b45c9),
      UINT32_C(0xb9f92d1f), UINT32_C(0x9c4f0d76),
   };
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                  pco_nir_options(),
                                                  "GLSL32");
   b.shader->info.internal = false;
   b.shader->info.prev_stage = MESA_SHADER_VERTEX;
   set_source_hash(b.shader, source_hash);

#define TERRAIN_D3_UNIFORM(type, name, location, driver_location)        \
   create_terrain_uniform(b.shader,                                     \
                          type,                                         \
                          name,                                         \
                          location,                                     \
                          driver_location,                              \
                          GLSL_PRECISION_MEDIUM)
   TERRAIN_D3_UNIFORM(glsl_vector_type(GLSL_TYPE_FLOAT, 3),
                      "uAmbientColor",
                      7,
                      0);
   TERRAIN_D3_UNIFORM(glsl_vector_type(GLSL_TYPE_FLOAT, 3),
                      "uDiffuseColor",
                      8,
                      1);
   TERRAIN_D3_UNIFORM(glsl_vector_type(GLSL_TYPE_FLOAT, 3),
                      "uSpecularColor",
                      9,
                      2);
   TERRAIN_D3_UNIFORM(glsl_float_type(), "uShininess", 10, 3);
   TERRAIN_D3_UNIFORM(glsl_float_type(), "uOpacity", 11, 4);
   create_terrain_sampler(b.shader, "tDiffuse1", 12, 0);
   create_terrain_sampler(b.shader, "tDiffuse2", 13, 1);
   create_terrain_sampler(b.shader, "tDetail", 14, 2);
   create_terrain_sampler(b.shader, "tSpecular", 15, 3);
   create_terrain_sampler(b.shader, "tDisplacement", 4, 4);
   TERRAIN_D3_UNIFORM(glsl_float_type(), "uNormalScale", 16, 5);
   TERRAIN_D3_UNIFORM(glsl_vec2_type(), "uRepeatOverlay", 17, 6);
   TERRAIN_D3_UNIFORM(glsl_vec2_type(), "uOffset", 18, 7);
   TERRAIN_D3_UNIFORM(glsl_vector_type(GLSL_TYPE_FLOAT, 3),
                      "ambientLightColor",
                      19,
                      8);
   nir_variable *view_matrix = nir_variable_create(
      b.shader,
      nir_var_uniform,
      glsl_matrix_type(GLSL_TYPE_FLOAT, 4, 4),
      "viewMatrix");
   view_matrix->data.location = 20;
   view_matrix->data.driver_location = 9;
   view_matrix->data.precision = GLSL_PRECISION_MEDIUM;
   TERRAIN_D3_UNIFORM(glsl_array_type(
                         glsl_vector_type(GLSL_TYPE_FLOAT, 3), 1, 0),
                      "pointLightColor",
                      21,
                      13);
   TERRAIN_D3_UNIFORM(glsl_array_type(
                         glsl_vector_type(GLSL_TYPE_FLOAT, 3), 1, 0),
                      "pointLightPosition",
                      22,
                      14);
   TERRAIN_D3_UNIFORM(glsl_array_type(glsl_float_type(), 1, 0),
                      "pointLightDistance",
                      23,
                      15);
#undef TERRAIN_D3_UNIFORM

   nir_variable *input[4];
   for (unsigned index = 0; index < 4; ++index) {
      input[index] = create_refract_vector(
         b.shader,
         nir_var_shader_in,
         index == 3U ? glsl_vec2_type() : glsl_vec4_type(),
         "VARYING_SLOT_VAR",
         VARYING_SLOT_VAR0 + index,
         varying_precision);
      input[index]->data.interpolation = INTERP_MODE_SMOOTH;
      input[index]->data.driver_location = index;
   }
   nir_variable *color_out =
      create_refract_vector(b.shader,
                            nir_var_shader_out,
                            glsl_vec4_type(),
                            "gl_FragColor",
                            FRAG_RESULT_COLOR,
                            GLSL_PRECISION_MEDIUM);

   nir_def *zero = nir_imm_float(&b, 0.0f);
   nir_def *in0 = nir_load_var(&b, input[0]);
   nir_def *in1 = nir_load_var(&b, input[1]);
   nir_def *in2 = nir_load_var(&b, input[2]);
   nir_def *in3 = nir_load_var(&b, input[3]);
   nir_def *ones = nir_imm_vec3(&b, 1.0f, 1.0f, 1.0f);
   nir_def *two = nir_imm_float(&b, 2.0f);
   nir_def *one = nir_imm_float(&b, 1.0f);
   nir_def *row[4] = {
      zero,
      nir_imm_int(&b, 1),
      nir_imm_int(&b, 2),
      nir_imm_int(&b, 3),
   };
   nir_def *opacity = nir_load_uniform(&b,
                                       1,
                                       32,
                                       zero,
                                       .base = 4,
                                       .range = 1,
                                       .dest_type = nir_type_float32);
   nir_def *material = nir_vec4(&b,
                                nir_channel(&b, ones, 0),
                                nir_channel(&b, ones, 1),
                                nir_channel(&b, ones, 2),
                                opacity);
   nir_def *repeat = nir_load_uniform(&b,
                                      2,
                                      32,
                                      zero,
                                      .base = 6,
                                      .range = 1,
                                      .dest_type = nir_type_float32);
   nir_def *base_coord =
      nir_vec2(&b, nir_channel(&b, in0, 0), nir_channel(&b, in0, 1));
   nir_def *detail_coord = nir_fmul(&b, repeat, nir_trim_vector(&b, in0, 2));
   nir_def *offset = nir_load_uniform(&b,
                                      2,
                                      32,
                                      zero,
                                      .base = 7,
                                      .range = 1,
                                      .dest_type = nir_type_float32);
   detail_coord = nir_fadd(&b, detail_coord, offset);
   nir_def *detail_sample = build_refract_texture_sample(&b, 2, detail_coord);
   nir_def *normal_map = nir_fadd(
      &b,
      nir_fmul(&b, nir_trim_vector(&b, detail_sample, 3), two),
      nir_imm_vec3(&b, -1.0f, -1.0f, -1.0f));
   nir_def *normal_scale = nir_load_uniform(&b,
                                            1,
                                            32,
                                            zero,
                                            .base = 5,
                                            .range = 1,
                                            .dest_type = nir_type_float32);
   nir_def *scaled_normal_xy =
      nir_fmul(&b, nir_trim_vector(&b, normal_map, 2), normal_scale);
   normal_map = nir_vec3(&b,
                         nir_channel(&b, scaled_normal_xy, 0),
                         nir_channel(&b, scaled_normal_xy, 1),
                         nir_channel(&b, normal_map, 2));
   nir_def *mapped_scale =
      nir_frsq(&b, nir_fdot3(&b, normal_map, normal_map));
   normal_map = nir_fmul(&b, normal_map, mapped_scale);

   nir_def *diffuse1 = build_refract_texture_sample(&b, 0, detail_coord);
   nir_def *diffuse2 = build_refract_texture_sample(&b, 1, detail_coord);
   nir_def *displacement = build_refract_texture_sample(&b, 4, base_coord);
   nir_def *mix_factor = nir_fadd(&b, one, nir_fneg(&b, displacement));
   nir_def *surface = nir_flrp(&b, diffuse1, diffuse2, mix_factor);
   surface = nir_fmul(&b, material, surface);
   nir_def *specular_map = build_refract_texture_sample(&b, 3, detail_coord);

   nir_def *tangent = nir_vec3(&b,
                               nir_channel(&b, in0, 2),
                               nir_channel(&b, in0, 3),
                               nir_channel(&b, in1, 0));
   nir_def *view_normal = nir_fadd(
      &b,
      nir_fadd(&b,
               nir_fmul(&b, tangent, nir_channel(&b, normal_map, 0)),
               nir_fmul(&b,
                        nir_trim_vector(&b, nir_channels(&b, in1, 0x0e), 3),
                        nir_channel(&b, normal_map, 1))),
      nir_fmul(&b,
               nir_trim_vector(&b, in2, 3),
               nir_channel(&b, normal_map, 2)));
   view_normal = nir_fmul(
      &b,
      view_normal,
      nir_frsq(&b, nir_fdot3(&b, view_normal, view_normal)));
   nir_def *view_position = nir_vec3(&b,
                                     nir_channel(&b, in2, 3),
                                     nir_channel(&b, in3, 0),
                                     nir_channel(&b, in3, 1));
   nir_def *view_direction = nir_fmul(
      &b,
      view_position,
      nir_frsq(&b, nir_fdot3(&b, view_position, view_position)));

   nir_def *point_position = nir_load_uniform(&b,
                                              3,
                                              32,
                                              zero,
                                              .base = 14,
                                              .range = 1,
                                              .dest_type = nir_type_float32);
   nir_def *transformed_point = NULL;
   for (unsigned index = 0; index < 4; ++index) {
      nir_def *matrix_row = nir_load_uniform(&b,
                                             4,
                                             32,
                                             row[index],
                                             .base = 9,
                                             .range = 4,
                                             .dest_type = nir_type_float32);
      nir_def *term = index < 3U ?
         nir_fmul(&b, matrix_row, nir_channel(&b, point_position, index)) :
         matrix_row;
      transformed_point = transformed_point ?
         nir_fadd(&b, transformed_point, term) : term;
   }
   nir_def *light_vector = nir_fadd(
      &b, nir_trim_vector(&b, transformed_point, 3), view_position);
   nir_def *light_distance = nir_load_uniform(&b,
                                              1,
                                              32,
                                              zero,
                                              .base = 15,
                                              .range = 1,
                                              .dest_type = nir_type_float32);
   b.fp_math_ctrl = nir_fp_preserve_inf;
   nir_def *distance_enabled = nir_flt(&b, zero, light_distance);
   b.fp_math_ctrl = nir_fp_fast_math;
   nir_def *distance_squared = nir_fdot3(&b, light_vector, light_vector);
   nir_def *distance = nir_fsqrt(&b, distance_squared);
   nir_def *distance_ratio = nir_fdiv(&b, distance, light_distance);
   b.fp_math_ctrl = nir_fp_preserve_inf | nir_fp_preserve_nan;
   nir_def *clamped_ratio = nir_fmin(&b, distance_ratio, one);
   b.fp_math_ctrl = nir_fp_fast_math;
   nir_def *attenuation = nir_fadd(&b, one, nir_fneg(&b, clamped_ratio));
   attenuation = nir_bcsel(&b, distance_enabled, attenuation, one);
   nir_def *light_direction = nir_fmul(
      &b, light_vector, nir_frsq(&b, distance_squared));
   nir_def *normal_dot_light =
      nir_fdot3(&b, view_normal, light_direction);
   b.fp_math_ctrl = nir_fp_preserve_inf | nir_fp_preserve_nan;
   normal_dot_light = nir_fmax(&b, normal_dot_light, zero);
   b.fp_math_ctrl = nir_fp_fast_math;

   nir_def *point_color = nir_load_uniform(&b,
                                           3,
                                           32,
                                           zero,
                                           .base = 13,
                                           .range = 1,
                                           .dest_type = nir_type_float32);
   nir_def *attenuated_color = nir_fmul(&b, attenuation, point_color);
   nir_def *diffuse_color = nir_load_uniform(&b,
                                             3,
                                             32,
                                             zero,
                                             .base = 1,
                                             .range = 1,
                                             .dest_type = nir_type_float32);
   nir_def *diffuse_light =
      nir_fmul(&b,
               nir_fmul(&b, attenuated_color, diffuse_color),
               normal_dot_light);
   nir_def *specular_color = nir_load_uniform(&b,
                                              3,
                                              32,
                                              zero,
                                              .base = 2,
                                              .range = 1,
                                              .dest_type = nir_type_float32);
   nir_def *specular_light = nir_fmul(&b, attenuated_color, specular_color);
   nir_def *halfway = nir_fadd(&b, light_direction, view_direction);
   halfway = nir_fmul(&b,
                      halfway,
                      nir_frsq(&b, nir_fdot3(&b, halfway, halfway)));
   nir_def *normal_dot_half = nir_fdot3(&b, view_normal, halfway);
   b.fp_math_ctrl = nir_fp_preserve_inf | nir_fp_preserve_nan;
   normal_dot_half = nir_fmax(&b, normal_dot_half, zero);
   b.fp_math_ctrl = nir_fp_fast_math;
   nir_def *shininess = nir_load_uniform(&b,
                                         1,
                                         32,
                                         zero,
                                         .base = 3,
                                         .range = 1,
                                         .dest_type = nir_type_float32);
   nir_def *specular_power = nir_fpow(&b, normal_dot_half, shininess);
   specular_power =
      nir_fmul(&b, nir_channel(&b, specular_map, 0), specular_power);
   specular_light = nir_fmul(&b,
                             nir_fmul(&b, specular_light, specular_power),
                             normal_dot_light);

   nir_def *ambient_light = nir_load_uniform(&b,
                                             3,
                                             32,
                                             zero,
                                             .base = 8,
                                             .range = 1,
                                             .dest_type = nir_type_float32);
   nir_def *ambient_color = nir_load_uniform(&b,
                                             3,
                                             32,
                                             zero,
                                             .base = 0,
                                             .range = 1,
                                             .dest_type = nir_type_float32);
   nir_def *lighting = nir_fadd(
      &b,
      nir_fadd(&b, diffuse_light, nir_fmul(&b, ambient_light, ambient_color)),
      specular_light);
   nir_def *lit_surface =
      nir_fmul(&b, nir_trim_vector(&b, surface, 3), lighting);
   nir_store_var(&b,
                 color_out,
                 nir_vec4(&b,
                          nir_channel(&b, lit_surface, 0),
                          nir_channel(&b, lit_surface, 1),
                          nir_channel(&b, lit_surface, 2),
                          nir_channel(&b, surface, 3)),
                 0x0f);
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static float
float_from_bits(uint32_t bits)
{
   union {
      uint32_t u;
      float f;
   } value = { .u = bits };
   return value.f;
}

static nir_def *
build_terrain_fge_inf(nir_builder *b, nir_def *left, nir_def *right)
{
   b->fp_math_ctrl = nir_fp_preserve_inf;
   nir_def *result = nir_fge(b, left, right);
   b->fp_math_ctrl = nir_fp_fast_math;
   return result;
}

static nir_def *
build_terrain_fabs_inf(nir_builder *b, nir_def *value)
{
   b->fp_math_ctrl = nir_fp_preserve_inf;
   nir_def *result = nir_fabs(b, value);
   b->fp_math_ctrl = nir_fp_fast_math;
   return result;
}

static nir_def *
build_terrain_fmin_inf_nan(nir_builder *b, nir_def *left, nir_def *right)
{
   b->fp_math_ctrl = nir_fp_preserve_inf | nir_fp_preserve_nan;
   nir_def *result = nir_fmin(b, left, right);
   b->fp_math_ctrl = nir_fp_fast_math;
   return result;
}

static nir_def *
build_terrain_fmax_inf_nan(nir_builder *b, nir_def *left, nir_def *right)
{
   b->fp_math_ctrl = nir_fp_preserve_inf | nir_fp_preserve_nan;
   nir_def *result = nir_fmax(b, left, right);
   b->fp_math_ctrl = nir_fp_fast_math;
   return result;
}

struct terrain_simplex_constants {
   nir_def *one;
   nir_def *zero;
   nir_def *two;
   nir_def *one_third;
   nir_def *one_sixth;
   nir_def *modulus;
   nir_def *zero_one;
   nir_def *seven;
   nir_def *forty_nine;
   nir_def *zeros;
   nir_def *six_tenths;
   nir_def *forty_two;
   nir_def *negative_half;
   nir_def *thirty_four;
   nir_def *hash_scale;
   nir_def *one_over_forty_nine;
   nir_def *normalization_bias;
   nir_def *normalization_scale;
};

static nir_def *
build_terrain_fmul_swizzled4_inf(nir_builder *b,
                                 nir_def *left,
                                 const unsigned left_swizzle[4],
                                 nir_def *right,
                                 const unsigned right_swizzle[4])
{
   nir_alu_instr *mul = nir_alu_instr_create(b->shader, nir_op_fmul);
   mul->src[0].src = nir_src_for_ssa(left);
   mul->src[1].src = nir_src_for_ssa(right);
   for (unsigned component = 0; component < 4; ++component) {
      mul->src[0].swizzle[component] = left_swizzle[component];
      mul->src[1].swizzle[component] = right_swizzle[component];
   }
   b->fp_math_ctrl = nir_fp_preserve_inf;
   nir_def *result = nir_builder_alu_instr_finish_and_insert(b, mul);
   b->fp_math_ctrl = nir_fp_fast_math;
   return result;
}

static nir_def *
build_terrain_simplex3(nir_builder *b,
                       nir_def *position,
                       struct terrain_simplex_constants *constants)
{
   nir_def *skew = nir_fdot3(b, position, constants->one_third);
   nir_def *cell = nir_ffloor(b, nir_fadd(b, position, skew));
   nir_def *cell_offset = nir_fadd(b, position, nir_fneg(b, cell));
   nir_def *unskew = nir_fdot3(b, cell, constants->one_sixth);
   cell_offset = nir_fadd(b, cell_offset, unskew);

   nir_def *rank_x = nir_b2f32(
      b,
      build_terrain_fge_inf(b,
                            nir_channel(b, cell_offset, 0),
                            nir_channel(b, cell_offset, 1)));
   nir_def *rank_y = nir_b2f32(
      b,
      build_terrain_fge_inf(b,
                            nir_channel(b, cell_offset, 1),
                            nir_channel(b, cell_offset, 2)));
   nir_def *rank_z = nir_b2f32(
      b,
      build_terrain_fge_inf(b,
                            nir_channel(b, cell_offset, 2),
                            nir_channel(b, cell_offset, 0)));
   nir_def *rank = nir_vec3(b, rank_x, rank_y, rank_z);
   nir_def *inverse_rank = nir_fadd(b, constants->one, nir_fneg(b, rank));
   nir_def *inverse_rank_zxy =
      nir_swizzle(b, inverse_rank, (unsigned[]){ 2, 0, 1 }, 3);
   nir_def *corner1 =
      build_terrain_fmin_inf_nan(b, rank, inverse_rank_zxy);
   nir_def *corner2 =
      build_terrain_fmax_inf_nan(b, rank, inverse_rank_zxy);
   nir_def *negative_corner1 = nir_fneg(b, corner1);
   nir_def *offset1 = nir_fadd(b, cell_offset, constants->one_sixth);
   offset1 = nir_fadd(b, offset1, negative_corner1);
   nir_def *negative_corner2 = nir_fneg(b, corner2);
   nir_def *offset2 = nir_fadd(b, cell_offset, constants->one_third);
   offset2 = nir_fadd(b, offset2, negative_corner2);
   if (!constants->negative_half) {
      constants->negative_half = nir_imm_vec3(b, -0.5f, -0.5f, -0.5f);
   }
   nir_def *offset3 =
      nir_fadd(b, constants->negative_half, cell_offset);

   nir_def *permuted = nir_fmod(b, cell, constants->modulus);
   nir_def *offset_z = nir_vec4(b,
                                nir_channel(b, constants->zero_one, 0),
                                nir_channel(b, corner1, 2),
                                nir_channel(b, corner2, 2),
                                nir_channel(b, constants->zero_one, 1));
   nir_def *hash = nir_fadd(b,
                            nir_swizzle(b,
                                        permuted,
                                        (unsigned[]){ 2, 2, 2, 2 },
                                        4),
                            offset_z);
   if (!constants->thirty_four)
      constants->thirty_four = nir_imm_float(b, 34.0f);
   nir_def *hash_factor = nir_fadd(
      b, nir_fmul(b, hash, constants->thirty_four), constants->one);
   hash = nir_fmod(b,
                   nir_fmul(b, hash_factor, hash),
                   constants->modulus);

   hash = nir_fadd(b,
                   hash,
                   nir_swizzle(b,
                               permuted,
                               (unsigned[]){ 1, 1, 1, 1 },
                               4));
   nir_def *offset_y = nir_vec4(b,
                                nir_channel(b, constants->zero_one, 0),
                                nir_channel(b, corner1, 1),
                                nir_channel(b, corner2, 1),
                                nir_channel(b, constants->zero_one, 1));
   hash = nir_fadd(b, hash, offset_y);
   hash_factor = nir_fadd(
      b, nir_fmul(b, hash, constants->thirty_four), constants->one);
   hash = nir_fmod(b,
                   nir_fmul(b, hash_factor, hash),
                   constants->modulus);

   hash = nir_fadd(b,
                   hash,
                   nir_swizzle(b,
                               permuted,
                               (unsigned[]){ 0, 0, 0, 0 },
                               4));
   nir_def *offset_x = nir_vec4(b,
                                nir_channel(b, constants->zero_one, 0),
                                nir_channel(b, corner1, 0),
                                nir_channel(b, corner2, 0),
                                nir_channel(b, constants->zero_one, 1));
   hash = nir_fadd(b, hash, offset_x);
   hash_factor = nir_fadd(
      b, nir_fmul(b, hash, constants->thirty_four), constants->one);
   hash = nir_fmod(b,
                   nir_fmul(b, hash_factor, hash),
                   constants->modulus);

   if (!constants->hash_scale) {
      constants->hash_scale = nir_imm_vec3(
         b,
         float_from_bits(UINT32_C(0x3e924925)),
         float_from_bits(UINT32_C(0xbf6db6db)),
         float_from_bits(UINT32_C(0x3e124925)));
   }
   if (!constants->one_over_forty_nine) {
      const float value = float_from_bits(UINT32_C(0x3ca72f06));
      constants->one_over_forty_nine =
         nir_imm_vec4(b, value, value, value, value);
   }
   nir_def *hash_cell =
      nir_ffloor(b, nir_fmul(b, constants->one_over_forty_nine, hash));
   nir_def *hash_mod = nir_fadd(
      b,
      hash,
      nir_fneg(b, nir_fmul(b, constants->forty_nine, hash_cell)));
   nir_def *gradient_cell = nir_ffloor(
      b,
      nir_fmul(b,
               hash_mod,
               nir_channel(b, constants->hash_scale, 2)));
   nir_def *gradient_x = nir_fadd(
      b,
      nir_fmul(b,
               gradient_cell,
               nir_channel(b, constants->hash_scale, 0)),
      nir_channel(b, constants->hash_scale, 1));
   nir_def *gradient_y_cell = nir_ffloor(
      b,
      nir_fadd(b,
               hash_mod,
               nir_fneg(b,
                        nir_fmul(b, constants->seven, gradient_cell))));
   nir_def *gradient_y = nir_fadd(
      b,
      nir_fmul(b,
               gradient_y_cell,
               nir_channel(b, constants->hash_scale, 0)),
      nir_channel(b, constants->hash_scale, 1));
   nir_def *gradient_h = nir_fadd(
      b,
      nir_fadd(b,
               constants->one,
               nir_fneg(b, build_terrain_fabs_inf(b, gradient_x))),
      nir_fneg(b, build_terrain_fabs_inf(b, gradient_y)));

   nir_def *gradient_select[4];
   for (unsigned component = 0; component < 4; ++component) {
      gradient_select[component] = nir_b2f32(
         b,
         build_terrain_fge_inf(b,
                               nir_channel(b, constants->zeros, component),
                               nir_channel(b, gradient_h, component)));
   }
   nir_def *select = nir_vec4(b,
                              gradient_select[0],
                              gradient_select[1],
                              gradient_select[2],
                              gradient_select[3]);
   nir_def *gradient_xy0 = nir_vec4(b,
                                    nir_channel(b, gradient_x, 0),
                                    nir_channel(b, gradient_y, 0),
                                    nir_channel(b, gradient_x, 1),
                                    nir_channel(b, gradient_y, 1));
   nir_def *gradient_sign0 = nir_vec4(b,
                                      nir_channel(b, gradient_x, 0),
                                      nir_channel(b, gradient_x, 1),
                                      nir_channel(b, gradient_y, 0),
                                      nir_channel(b, gradient_y, 1));
   gradient_sign0 = nir_fadd(
      b,
      nir_fmul(b, nir_ffloor(b, gradient_sign0), constants->two),
      constants->one);
   nir_def *adjust0 = build_terrain_fmul_swizzled4_inf(
      b,
      select,
      (unsigned[]){ 0, 0, 1, 1 },
      gradient_sign0,
      (unsigned[]){ 0, 2, 1, 3 });
   b->fp_math_ctrl = nir_fp_preserve_inf;
   adjust0 = nir_fneg(b, adjust0);
   b->fp_math_ctrl = nir_fp_fast_math;
   gradient_xy0 = nir_fadd(b, gradient_xy0, adjust0);

   nir_def *gradient_xy1 = nir_vec4(b,
                                    nir_channel(b, gradient_x, 2),
                                    nir_channel(b, gradient_y, 2),
                                    nir_channel(b, gradient_x, 3),
                                    nir_channel(b, gradient_y, 3));
   nir_def *gradient_sign1 = nir_vec4(b,
                                      nir_channel(b, gradient_x, 2),
                                      nir_channel(b, gradient_x, 3),
                                      nir_channel(b, gradient_y, 2),
                                      nir_channel(b, gradient_y, 3));
   gradient_sign1 = nir_fadd(
      b,
      nir_fmul(b, nir_ffloor(b, gradient_sign1), constants->two),
      constants->one);
   nir_def *adjust1 = build_terrain_fmul_swizzled4_inf(
      b,
      select,
      (unsigned[]){ 2, 2, 3, 3 },
      gradient_sign1,
      (unsigned[]){ 0, 2, 1, 3 });
   b->fp_math_ctrl = nir_fp_preserve_inf;
   adjust1 = nir_fneg(b, adjust1);
   b->fp_math_ctrl = nir_fp_fast_math;
   gradient_xy1 = nir_fadd(b, gradient_xy1, adjust1);

   nir_def *gradient[4];
   nir_def *gradient_norm[4];
   for (unsigned component = 0; component < 4; ++component) {
      const bool upper = component >= 2;
      nir_def *xy = upper ? gradient_xy1 : gradient_xy0;
      const unsigned pair = (component & 1U) * 2U;
      gradient[component] = nir_vec3(b,
                                     nir_channel(b, xy, pair),
                                     nir_channel(b, xy, pair + 1U),
                                     nir_channel(b, gradient_h, component));
      gradient_norm[component] =
         nir_fdot3(b, gradient[component], gradient[component]);
   }
   if (!constants->normalization_bias) {
      constants->normalization_bias =
         nir_imm_float(b, float_from_bits(UINT32_C(0x3fe57be0)));
   }
   if (!constants->normalization_scale) {
      constants->normalization_scale =
         nir_imm_float(b, float_from_bits(UINT32_C(0x3f5a8e5c)));
   }
   nir_def *normalization = nir_fadd(
      b,
      constants->normalization_bias,
      nir_fneg(b,
               nir_fmul(b,
                        constants->normalization_scale,
                        nir_vec4(b,
                                 gradient_norm[0],
                                 gradient_norm[1],
                                 gradient_norm[2],
                                 gradient_norm[3]))));
   for (unsigned component = 0; component < 4; ++component) {
      gradient[component] =
         nir_fmul(b, gradient[component], nir_channel(b, normalization, component));
   }

   nir_def *corner_position[4] = {
      cell_offset,
      offset1,
      offset2,
      offset3,
   };
   nir_def *falloff_component[4];
   for (unsigned component = 0; component < 4; ++component) {
      falloff_component[component] =
         nir_fdot3(b, corner_position[component], corner_position[component]);
   }
   nir_def *falloff = nir_fadd(
      b,
      constants->six_tenths,
      nir_fneg(b,
               nir_vec4(b,
                        falloff_component[0],
                        falloff_component[1],
                        falloff_component[2],
                        falloff_component[3])));
   falloff = build_terrain_fmax_inf_nan(b, falloff, constants->zero);
   falloff = nir_fmul(b, falloff, falloff);
   nir_def *gradient_dot[4];
   for (unsigned component = 0; component < 4; ++component) {
      gradient_dot[component] =
         nir_fdot3(b, gradient[component], corner_position[component]);
   }
   falloff = nir_fmul(b, falloff, falloff);
   nir_def *noise = nir_fdot4(
      b,
      falloff,
      nir_vec4(b,
               gradient_dot[0],
               gradient_dot[1],
               gradient_dot[2],
               gradient_dot[3]));
   noise = nir_fmul(b, constants->forty_two, noise);
   return build_terrain_fabs_inf(b, noise);
}

static nir_shader *
build_terrain_d1_vertex_shader(unsigned varying_precision)
{
   static const uint32_t source_hash[8] = {
      UINT32_C(0x64295326), UINT32_C(0xb7892f55),
      UINT32_C(0x40f4c2f4), UINT32_C(0x2524db37),
      UINT32_C(0x4a6849f7), UINT32_C(0x4212d7bb),
      UINT32_C(0x51ad07d2), UINT32_C(0x41c1b894),
   };
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
                                                  pco_nir_options(),
                                                  "GLSL6");
   b.shader->info.internal = false;
   b.shader->info.next_stage = MESA_SHADER_FRAGMENT;
   set_source_hash(b.shader, source_hash);
   nir_variable *uv_offset = create_terrain_uniform(b.shader,
                                                     glsl_vec2_type(),
                                                     "uvOffset",
                                                     0,
                                                     0,
                                                     GLSL_PRECISION_HIGH);
   nir_variable *uv_scale = create_terrain_uniform(b.shader,
                                                    glsl_vec2_type(),
                                                    "uvScale",
                                                    1,
                                                    1,
                                                    GLSL_PRECISION_MEDIUM);
   (void)uv_offset;
   (void)uv_scale;
   nir_variable *position = create_refract_vector(b.shader,
                                                   nir_var_shader_in,
                                                   glsl_vec4_type(),
                                                   "position",
                                                   VERT_ATTRIB_GENERIC0,
                                                   GLSL_PRECISION_HIGH);
   position->data.driver_location = 0;
   nir_variable *position_out = create_refract_vector(b.shader,
                                                       nir_var_shader_out,
                                                       glsl_vec4_type(),
                                                       "gl_Position",
                                                       VARYING_SLOT_POS,
                                                       GLSL_PRECISION_HIGH);
   position_out->data.driver_location = 0;
   position_out->data.interpolation = INTERP_MODE_NONE;
   nir_variable *uv_out = create_refract_vector(b.shader,
                                                 nir_var_shader_out,
                                                 glsl_vec2_type(),
                                                 "vUv",
                                                 VARYING_SLOT_VAR0,
                                                 varying_precision);
   uv_out->data.driver_location = 1;
   uv_out->data.interpolation = INTERP_MODE_NONE;

   nir_def *half = nir_imm_float(&b, 0.5f);
   nir_def *one = nir_imm_float(&b, 1.0f);
   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *scale = nir_load_uniform(&b,
                                     2,
                                     32,
                                     zero,
                                     .base = 1,
                                     .range = 1,
                                     .dest_type = nir_type_float32);
   nir_def *position_value = nir_load_var(&b, position);
   nir_def *uv = nir_fmul(&b,
                          nir_trim_vector(&b, position_value, 2),
                          half);
   uv = nir_fadd(&b, uv, half);
   uv = nir_fmul(&b, scale, uv);
   nir_def *offset = nir_load_uniform(&b,
                                      2,
                                      32,
                                      zero,
                                      .base = 0,
                                      .range = 1,
                                      .dest_type = nir_type_float32);
   uv = nir_fadd(&b, uv, offset);
   nir_store_var(&b,
                 position_out,
                 nir_vec4(&b,
                          nir_channel(&b, position_value, 0),
                          nir_channel(&b, position_value, 1),
                          nir_channel(&b, position_value, 2),
                          one),
                 0x0f);
   nir_def *flipped_y = nir_fadd(&b,
                                 nir_channel(&b, scale, 1),
                                 nir_fneg(&b, nir_channel(&b, uv, 1)));
   nir_store_var(&b,
                 uv_out,
                 nir_vec2(&b, nir_channel(&b, uv, 0), flipped_y),
                 0x03);
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *
build_terrain_d1_fragment_shader(unsigned varying_precision)
{
   static const uint32_t source_hash[8] = {
      UINT32_C(0x72c70927), UINT32_C(0x1eee611e),
      UINT32_C(0xa4dd2fdc), UINT32_C(0xcd490824),
      UINT32_C(0xd08e0f18), UINT32_C(0xbd42bba9),
      UINT32_C(0x3508625b), UINT32_C(0xfa3c6660),
   };
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                  pco_nir_options(),
                                                  "GLSL6");
   b.shader->info.internal = false;
   b.shader->info.prev_stage = MESA_SHADER_VERTEX;
   set_source_hash(b.shader, source_hash);
   create_terrain_uniform(b.shader,
                          glsl_float_type(),
                          "time",
                          2,
                          0,
                          GLSL_PRECISION_MEDIUM);
   nir_variable *uv = create_refract_vector(b.shader,
                                             nir_var_shader_in,
                                             glsl_vec2_type(),
                                             "VARYING_SLOT_VAR0",
                                             VARYING_SLOT_VAR0,
                                             varying_precision);
   uv->data.driver_location = 0;
   uv->data.interpolation = INTERP_MODE_SMOOTH;
   nir_variable *color_out = create_refract_vector(b.shader,
                                                    nir_var_shader_out,
                                                    glsl_vec4_type(),
                                                    "gl_FragColor",
                                                    FRAG_RESULT_COLOR,
                                                    GLSL_PRECISION_MEDIUM);
   color_out->data.driver_location = 0;

   struct terrain_simplex_constants constants = { 0 };
   constants.one = nir_imm_float(&b, 1.0f);
   constants.zero = nir_imm_float(&b, 0.0f);
   nir_def *uv_value = nir_load_var(&b, uv);
   nir_def *time = nir_load_uniform(&b,
                                    1,
                                    32,
                                    constants.zero,
                                    .base = 0,
                                    .range = 1,
                                    .dest_type = nir_type_float32);
   b.fp_math_ctrl = nir_fp_preserve_inf;
   time = nir_fneg(&b, time);
   b.fp_math_ctrl = nir_fp_fast_math;
   constants.two = nir_imm_float(&b, 2.0f);
   nir_def *half = nir_imm_float(&b, 0.5f);
   nir_def *four = nir_imm_float(&b, 4.0f);
   nir_def *quarter = nir_imm_float(&b, 0.25f);
   nir_def *eight = nir_imm_float(&b, 8.0f);
   nir_def *eighth = nir_imm_float(&b, 0.125f);
   const float third = float_from_bits(UINT32_C(0x3eaaaaab));
   constants.one_third = nir_imm_vec3(&b, third, third, third);
   const float sixth = float_from_bits(UINT32_C(0x3e2aaaab));
   constants.one_sixth = nir_imm_vec3(&b, sixth, sixth, sixth);
   constants.modulus = nir_imm_float(&b, 289.0f);
   constants.zero_one = nir_imm_vec2(&b, 0.0f, 1.0f);
   constants.seven = nir_imm_float(&b, 7.0f);
   constants.forty_nine = nir_imm_float(&b, 49.0f);
   constants.zeros = nir_imm_vec4(&b, 0.0f, 0.0f, 0.0f, 0.0f);
   constants.six_tenths =
      nir_imm_float(&b, float_from_bits(UINT32_C(0x3f19999a)));
   constants.forty_two = nir_imm_float(&b, 42.0f);
   nir_def *position = nir_vec3(&b,
                                nir_channel(&b, uv_value, 0),
                                nir_channel(&b, uv_value, 1),
                                time);

   nir_def *noise = build_terrain_simplex3(&b, position, &constants);
   nir_def *noise2 = build_terrain_simplex3(
      &b, nir_fmul(&b, position, constants.two), &constants);
   noise = nir_fadd(&b, noise, nir_fmul(&b, half, noise2));
   nir_def *noise4 = build_terrain_simplex3(
      &b, nir_fmul(&b, position, four), &constants);
   noise = nir_fadd(&b, noise, nir_fmul(&b, quarter, noise4));
   nir_def *noise8 = build_terrain_simplex3(
      &b, nir_fmul(&b, position, eight), &constants);
   noise = nir_fadd(&b, noise, nir_fmul(&b, eighth, noise8));
   nir_store_var(&b,
                 color_out,
                 nir_vec4(&b,
                          noise,
                          noise,
                          noise,
                          constants.one),
                 0x0f);
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *
build_terrain_blur_fragment_shader(enum pvrgpu_pco_terrain_profile profile,
                                   unsigned varying_precision)
{
   static const uint32_t source_hash[4][8] = {
      {
         UINT32_C(0xc73e207c), UINT32_C(0x4ef26404),
         UINT32_C(0x604847d2), UINT32_C(0xfeebd5e6),
         UINT32_C(0xbea9c632), UINT32_C(0x79454818),
         UINT32_C(0xa5c408bc), UINT32_C(0x586cbda5),
      },
      {
         UINT32_C(0xa49328b3), UINT32_C(0xebd09eb6),
         UINT32_C(0xb68bf6c9), UINT32_C(0x98f24299),
         UINT32_C(0xa30fa19d), UINT32_C(0xc6c7159a),
         UINT32_C(0xba654c0a), UINT32_C(0x6d39f49c),
      },
      {
         UINT32_C(0xbbe35ba4), UINT32_C(0x0f0884d6),
         UINT32_C(0x054a182f), UINT32_C(0xb0b5363f),
         UINT32_C(0x69b62005), UINT32_C(0x3c8b1b0f),
         UINT32_C(0xef2f4a55), UINT32_C(0x48f9a4f9),
      },
      {
         UINT32_C(0x8bf6b63e), UINT32_C(0x59d22899),
         UINT32_C(0x88554142), UINT32_C(0x383e51c8),
         UINT32_C(0x90641bf4), UINT32_C(0xc10a2fc3),
         UINT32_C(0x3207cac2), UINT32_C(0x56282ca8),
      },
   };
   const unsigned hash_index =
      profile == PVRGPU_PCO_TERRAIN_D4 ? 1U :
      profile == PVRGPU_PCO_TERRAIN_D5 ? 0U :
      profile == PVRGPU_PCO_TERRAIN_D7 ? 3U : 2U;
   const bool nine_taps = profile == PVRGPU_PCO_TERRAIN_D7 ||
                          profile == PVRGPU_PCO_TERRAIN_D8;
   const bool horizontal = profile == PVRGPU_PCO_TERRAIN_D4 ||
                           profile == PVRGPU_PCO_TERRAIN_D7;
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                  pco_nir_options(),
                                                  nine_taps ? "GLSL47" :
                                                              "GLSL38");
   b.shader->info.internal = false;
   b.shader->info.prev_stage = MESA_SHADER_VERTEX;
   set_source_hash(b.shader, source_hash[hash_index]);

   nir_variable *sampler = nir_variable_create(
      b.shader,
      nir_var_uniform,
      glsl_sampler_type(GLSL_SAMPLER_DIM_2D, false, false, GLSL_TYPE_FLOAT),
      "Texture0");
   sampler->data.location = 2;
   sampler->data.binding = 0;
   sampler->data.precision = GLSL_PRECISION_LOW;
   nir_variable *uv =
      create_refract_vector(b.shader,
                            nir_var_shader_in,
                            glsl_vec2_type(),
                            "VARYING_SLOT_VAR0",
                            VARYING_SLOT_VAR0,
                            varying_precision);
   uv->data.interpolation = INTERP_MODE_SMOOTH;
   nir_variable *color_out =
      create_refract_vector(b.shader,
                            nir_var_shader_out,
                            glsl_vec4_type(),
                            "gl_FragColor",
                            FRAG_RESULT_COLOR,
                            GLSL_PRECISION_MEDIUM);

   nir_def *coord = NULL;
   nir_def *result = NULL;
   nir_def *one = NULL;
   if (!nine_taps) {
      one = nir_imm_float(&b, 1.0f);
      nir_def *zero = nir_imm_float(&b, 0.0f);
      nir_def *two = nir_imm_float(&b, 2.0f);
      nir_def *step = nir_imm_float(
         &b,
         float_from_bits(horizontal ? UINT32_C(0x3c4ccccd) :
                                      UINT32_C(0x3b7ffbce)));
      nir_def *weight0 =
         nir_imm_float(&b, float_from_bits(UINT32_C(0x3e40214b)));
      nir_def *weight1 =
         nir_imm_float(&b, float_from_bits(UINT32_C(0x3e53037d)));
      nir_def *weight2 =
         nir_imm_float(&b, float_from_bits(UINT32_C(0x3e59b62c)));
      coord = nir_load_var(&b, uv);
      nir_def *scaled =
         nir_fdiv(&b,
                  nir_fmul(&b,
                           step,
                           nir_fabs(&b, nir_channel(&b, coord, 1))),
                  one);
      nir_def *twice = nir_fmul(&b, two, scaled);
      nir_def *offset[5] = {
         nir_fneg(&b, twice),
         nir_fneg(&b, scaled),
         NULL,
         scaled,
         twice,
      };
      nir_def *weight[5] = {
         weight0,
         weight1,
         weight2,
         weight1,
         weight0,
      };
      for (unsigned tap = 0; tap < 5; ++tap) {
         nir_def *tap_coord = coord;
         if (offset[tap]) {
            nir_def *delta = horizontal ? nir_vec2(&b, offset[tap], zero) :
                                          nir_vec2(&b, zero, offset[tap]);
            tap_coord = nir_fadd(&b, coord, delta);
         }
         nir_def *term =
            nir_fmul(&b,
                     build_refract_texture_sample(&b, 0, tap_coord),
                     weight[tap]);
         result = result ? nir_fadd(&b, result, term) : term;
      }
   } else {
      nir_def *half = nir_imm_float(&b, 0.5f);
      nir_def *zero = nir_imm_float(&b, 0.0f);
      nir_def *four = nir_imm_float(&b, 4.0f);
      nir_def *three = nir_imm_float(&b, 3.0f);
      nir_def *two = nir_imm_float(&b, 2.0f);
      one = nir_imm_float(&b, 1.0f);
      nir_def *step = nir_imm_float(
         &b,
         float_from_bits(horizontal ? UINT32_C(0x3c4ccccd) :
                                      UINT32_C(0x3c88893b)));
      nir_def *weight0 =
         nir_imm_float(&b, float_from_bits(UINT32_C(0x3d5edbf9)));
      nir_def *weight1 =
         nir_imm_float(&b, float_from_bits(UINT32_C(0x3db4195d)));
      nir_def *weight2 =
         nir_imm_float(&b, float_from_bits(UINT32_C(0x3dfdc619)));
      nir_def *weight3 =
         nir_imm_float(&b, float_from_bits(UINT32_C(0x3e1be059)));
      nir_def *weight4 =
         nir_imm_float(&b, float_from_bits(UINT32_C(0x3e26f156)));
      coord = nir_load_var(&b, uv);
      nir_def *scaled = nir_fdiv(
         &b,
         nir_fmul(&b,
                  step,
                  nir_fabs(&b,
                            nir_fadd(&b,
                                     half,
                                     nir_fneg(&b,
                                              nir_channel(&b, coord, 1))))),
         half);
      nir_def *four_step = nir_fmul(&b, four, scaled);
      nir_def *three_step = nir_fmul(&b, three, scaled);
      nir_def *two_step = nir_fmul(&b, two, scaled);
      nir_def *offset[9] = {
         nir_fneg(&b, four_step),
         nir_fneg(&b, three_step),
         nir_fneg(&b, two_step),
         nir_fneg(&b, scaled),
         NULL,
         scaled,
         two_step,
         three_step,
         four_step,
      };
      nir_def *weight[9] = {
         weight0,
         weight1,
         weight2,
         weight3,
         weight4,
         weight3,
         weight2,
         weight1,
         weight0,
      };
      for (unsigned tap = 0; tap < 9; ++tap) {
         nir_def *tap_coord = coord;
         if (offset[tap]) {
            nir_def *delta = horizontal ? nir_vec2(&b, offset[tap], zero) :
                                          nir_vec2(&b, zero, offset[tap]);
            tap_coord = nir_fadd(&b, coord, delta);
         }
         nir_def *term =
            nir_fmul(&b,
                     build_refract_texture_sample(&b, 0, tap_coord),
                     weight[tap]);
         result = result ? nir_fadd(&b, result, term) : term;
      }
   }

   nir_store_var(&b,
                 color_out,
                 nir_vec4(&b,
                          nir_channel(&b, result, 0),
                          nir_channel(&b, result, 1),
                          nir_channel(&b, result, 2),
                          one),
                 0x0f);
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static unsigned count_intrinsic(const nir_shader *nir, nir_intrinsic_op op)
{
   unsigned count = 0;
   nir_foreach_function (function, nir) {
      if (!function->impl)
         continue;
      nir_foreach_block (block, function->impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == op)
               ++count;
         }
      }
   }
   return count;
}

static unsigned count_alu(const nir_shader *nir, nir_op op)
{
   unsigned count = 0;
   nir_foreach_function (function, nir) {
      if (!function->impl)
         continue;
      nir_foreach_block (block, function->impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type == nir_instr_type_alu &&
                nir_instr_as_alu(instr)->op == op)
               ++count;
         }
      }
   }
   return count;
}

static uint64_t fnv1a64(const void *data, size_t size)
{
   const uint8_t *bytes = data;
   uint64_t hash = UINT64_C(14695981039346656037);

   for (size_t i = 0; i < size; ++i) {
      hash ^= bytes[i];
      hash *= UINT64_C(1099511628211);
   }

   return hash;
}

static uint64_t read_u64(const uint32_t *words, unsigned first_dword)
{
   return (uint64_t)words[first_dword] |
          ((uint64_t)words[first_dword + 1] << 32U);
}

static void test_refract_fragment_descriptors(void)
{
   uint32_t shared[PVRGPU_PCO_REFRACT_FRAGMENT_SHARED_DWORDS];
   memset(shared, 0xcd, sizeof(shared));
   pvrgpu_pco_build_refract_fragment_shared(shared);
   if (fnv1a64(shared, sizeof(shared)) !=
       UINT64_C(0x26536c76cbc158b5)) {
      fail("refract canonical descriptor fingerprint changed");
   }

   static const unsigned formats[3] = { 24U, 12U, 12U };
   static const unsigned widths[3] = { 160U, 160U, 512U };
   static const unsigned heights[3] = { 120U, 120U, 512U };
   static const unsigned mip_counts[3] = { 1U, 8U, 1U };
   static const unsigned byte_sizes[3] = { 76800U, 102352U, 1048576U };
   static const unsigned sampler_max_lod[3] = { 0U, 448U, 0U };
   for (unsigned set = 0; set < 3; ++set) {
      const unsigned base = set * PVRGPU_PCO_TEXTURE_DESCRIPTOR_DWORDS;
      const uint64_t image_word0 = read_u64(shared, base);
      const uint64_t image_word1 = read_u64(shared, base + 2U);
      const uint64_t sampler_word0 = read_u64(shared, base + 8U);
      if (((image_word0 >> 0U) & 7U) != 4U ||
          ((image_word0 >> 27U) & 127U) != formats[set] ||
          ((image_word0 >> 34U) & 16383U) + 1U != widths[set] ||
          ((image_word0 >> 48U) & 16383U) + 1U != heights[set] ||
          ((image_word1 >> 0U) & 32767U) + 1U != widths[set] ||
          ((image_word1 >> 16U) & UINT64_C(0x3fffffffff)) != 0 ||
          ((image_word1 >> 60U) & 15U) != mip_counts[set] ||
          shared[base + 4U] != byte_sizes[set] ||
          ((sampler_word0 >> 23U) & 1023U) != sampler_max_lod[set] ||
          ((sampler_word0 >> 33U) & 7U) != 2U ||
          ((sampler_word0 >> 41U) & 7U) != 2U) {
         fail("refract canonical descriptor fields changed");
      }
   }
   const uint64_t depth_image = read_u64(shared, 0);
   if (((depth_image >> 5U) & 7U) != 4U ||
       ((depth_image >> 8U) & 7U) != 0U ||
       ((depth_image >> 11U) & 7U) != 0U ||
       ((depth_image >> 14U) & 7U) != 0U) {
      fail("refract Z32 descriptor is not XXX1 swizzled");
   }
}

static void test_terrain_texture_descriptors(void)
{
   uint32_t rgba[PVRGPU_PCO_TEXTURE_DESCRIPTOR_DWORDS];
   uint32_t rgbx[PVRGPU_PCO_TEXTURE_DESCRIPTOR_DWORDS];
   if (!pvrgpu_pco_build_terrain_texture_descriptor(
          rgba,
          PIPE_FORMAT_R8G8B8A8_UNORM,
          80,
          60,
          7,
          25552,
          1,
          1,
          1,
          2,
          2,
          384) ||
       !pvrgpu_pco_build_terrain_texture_descriptor(
          rgbx,
          PIPE_FORMAT_R8G8B8X8_UNORM,
          512,
          512,
          10,
          1398100,
          1,
          1,
          1,
          0,
          0,
          576)) {
      fail("terrain descriptor builder rejected valid layouts");
   }
   const uint64_t rgba_image = read_u64(rgba, 0);
   const uint64_t rgbx_image = read_u64(rgbx, 0);
   const uint64_t rgba_sampler = read_u64(rgba, 8);
   const uint64_t rgbx_sampler = read_u64(rgbx, 8);
   if (((rgba_image >> 5U) & 7U) != 3U ||
       ((rgbx_image >> 5U) & 7U) != 4U ||
       ((rgba_sampler >> 33U) & 7U) != 2U ||
       ((rgba_sampler >> 41U) & 7U) != 2U ||
       ((rgbx_sampler >> 33U) & 7U) != 0U ||
       ((rgbx_sampler >> 41U) & 7U) != 0U ||
       ((rgbx_sampler >> 23U) & 1023U) != 576U) {
      fail("terrain descriptor swizzle/sampler fields changed");
   }
   if (pvrgpu_pco_build_terrain_texture_descriptor(
          rgbx,
          PIPE_FORMAT_Z16_UNORM,
          512,
          512,
          10,
          1398100,
          1,
          1,
          1,
          0,
          0,
          576)) {
      fail("terrain descriptor builder accepted unsupported format");
   }
}

static void
test_refract_compile_profile(struct pvrgpu_pco_compiler *compiler,
                             enum pvrgpu_pco_refract_profile profile)
{
   const bool composite = profile == PVRGPU_PCO_REFRACT_COMPOSITE;
   nir_shader *vs = composite ? build_refract_composite_vertex_shader() :
                                build_refract_prepass_vertex_shader();
   nir_shader *fs =
      composite ? build_refract_composite_fragment_shader(
                     GLSL_PRECISION_MEDIUM) :
                  build_refract_prepass_fragment_shader(
                     GLSL_PRECISION_MEDIUM);
   uint8_t source_vs_hash[sizeof(vs->info.source_blake3)];
   uint8_t source_fs_hash[sizeof(fs->info.source_blake3)];
   memcpy(source_vs_hash, vs->info.source_blake3, sizeof(source_vs_hash));
   memcpy(source_fs_hash, fs->info.source_blake3, sizeof(source_fs_hash));

   const unsigned expected_uniform_loads = composite ? 16U : 4U;
   if (count_intrinsic(vs, nir_intrinsic_load_uniform) !=
          expected_uniform_loads ||
       count_intrinsic(vs, nir_intrinsic_load_push_constant) != 0 ||
       count_intrinsic(fs, nir_intrinsic_load_push_constant) != 0) {
      fail("refract test NIR did not start in Gallium load_uniform form");
   }

   char error[512] = { 0 };
   struct pvrgpu_pco_graphics_binary binary;
   if (!pvrgpu_pco_compile_refract(compiler,
                                   vs,
                                   fs,
                                   profile,
                                   &binary,
                                   error,
                                   sizeof(error))) {
      fail(error[0] ? error : "refract compile failed");
   }

   const size_t expected_vs_size = composite ? 1536U : 432U;
   const size_t expected_fs_size = composite ? 5072U : 56U;
   const uint64_t expected_vs_hash =
      composite ? UINT64_C(0xc46a9af088bfe8a9) :
                  UINT64_C(0x6e9ad97e49eca9fe);
   const uint64_t expected_fs_hash =
      composite ? UINT64_C(0x8fe8ae5903f3c2dd) :
                  UINT64_C(0xa55a28d91b0f4b9e);
   const uint64_t vertex_hash =
      fnv1a64(binary.vertex.data, binary.vertex.size);
   const uint64_t fragment_hash =
      fnv1a64(binary.fragment.data, binary.fragment.size);
   if (binary.vertex.size != expected_vs_size ||
       binary.fragment.size != expected_fs_size ||
       vertex_hash != expected_vs_hash || fragment_hash != expected_fs_hash) {
      fprintf(stderr,
              "refract profile=%u PCO got VS=%zu/%016llx FS=%zu/%016llx\n",
              profile,
              binary.vertex.size,
              (unsigned long long)vertex_hash,
              binary.fragment.size,
              (unsigned long long)fragment_hash);
      fail("refract PCO binary differs from the strict capture fixture");
   }

   const struct pvrgpu_pco_stage_abi expected_vs_abi = composite ?
      (struct pvrgpu_pco_stage_abi){ 20, 8, 15, 0, 64, 0, 64, 0 } :
      (struct pvrgpu_pco_stage_abi){ 10, 8, 7, 0, 16, 0, 16, 0 };
   const struct pvrgpu_pco_stage_abi expected_fs_abi = composite ?
      (struct pvrgpu_pco_stage_abi){ 21, 0, 0, 48, 60, 0, 0, 0 } :
      (struct pvrgpu_pco_stage_abi){ 3, 0, 0, 16, 0, 0, 0, 0 };
   if (memcmp(&binary.vertex.abi,
              &expected_vs_abi,
              sizeof(expected_vs_abi)) != 0 ||
       memcmp(&binary.fragment.abi,
              &expected_fs_abi,
              sizeof(expected_fs_abi)) != 0) {
      fprintf(stderr,
              "refract profile=%u ABI got VS=(%u,%u,%u,%u,%u,%u,%u,%u) "
              "FS=(%u,%u,%u,%u,%u,%u,%u,%u)\n",
              profile,
              binary.vertex.abi.temps,
              binary.vertex.abi.vertex_inputs,
              binary.vertex.abi.vertex_outputs,
              binary.vertex.abi.coefficients,
              binary.vertex.abi.shareds,
              binary.vertex.abi.push_constant_start,
              binary.vertex.abi.push_constant_count,
              binary.vertex.abi.entry_offset,
              binary.fragment.abi.temps,
              binary.fragment.abi.vertex_inputs,
              binary.fragment.abi.vertex_outputs,
              binary.fragment.abi.coefficients,
              binary.fragment.abi.shareds,
              binary.fragment.abi.push_constant_start,
              binary.fragment.abi.push_constant_count,
              binary.fragment.abi.entry_offset);
      fail("refract PCO stage ABI changed");
   }

   const unsigned varying_components = composite ? 11U : 3U;
   if (binary.position_output_start != 0 ||
       binary.position_output_count != 4 ||
       binary.fragment_position_start != 0 ||
       binary.fragment_position_count != 4 ||
       binary.varying_output_start != 4 ||
       binary.varying_output_count != varying_components ||
       binary.fragment_varying_start != 4 ||
       binary.fragment_varying_count != varying_components * 4 ||
       binary.fragment_texture_descriptor_start != 0 ||
       binary.fragment_texture_descriptor_count != (composite ? 20U : 0U) ||
       binary.fragment_texture_descriptor_stride != (composite ? 20U : 0U)) {
      fail("refract PCO linkage or descriptor ABI changed");
   }

   if (count_intrinsic(vs, nir_intrinsic_load_uniform) !=
          expected_uniform_loads ||
       count_intrinsic(vs, nir_intrinsic_load_push_constant) != 0 ||
       count_intrinsic(fs, nir_intrinsic_load_push_constant) != 0 ||
       memcmp(source_vs_hash,
              vs->info.source_blake3,
              sizeof(source_vs_hash)) != 0 ||
       memcmp(source_fs_hash,
              fs->info.source_blake3,
              sizeof(source_fs_hash)) != 0) {
      fail("refract compile modified caller-owned NIR");
   }
   pvrgpu_pco_graphics_binary_finish(&binary);

   fs->info.source_blake3[0] ^= 1;
   memset(error, 0, sizeof(error));
   if (pvrgpu_pco_compile_refract(compiler,
                                  vs,
                                  fs,
                                  profile,
                                  &binary,
                                  error,
                                  sizeof(error)) ||
       !strstr(error, "source signature mismatch")) {
      fail("refract source-hash mismatch did not fail closed");
   }

   ralloc_free(vs);
   ralloc_free(fs);
}

static void
test_refract_precision_gate(struct pvrgpu_pco_compiler *compiler)
{
   nir_shader *vs = build_refract_prepass_vertex_shader();
   nir_shader *fs =
      build_refract_prepass_fragment_shader(GLSL_PRECISION_HIGH);
   char error[512] = { 0 };
   struct pvrgpu_pco_graphics_binary binary;
   if (pvrgpu_pco_compile_refract(compiler,
                                  vs,
                                  fs,
                                  PVRGPU_PCO_REFRACT_PREPASS,
                                  &binary,
                                  error,
                                  sizeof(error)) ||
       !strstr(error, "varying ABI mismatch")) {
      fail("refract precision mismatch did not fail closed");
   }
   ralloc_free(vs);
   ralloc_free(fs);
}

static void
test_shadow_fragment_descriptor(void)
{
   uint32_t shadow[PVRGPU_PCO_TEXTURE_DESCRIPTOR_DWORDS];
   uint32_t refract[PVRGPU_PCO_REFRACT_FRAGMENT_SHARED_DWORDS];
   memset(shadow, 0xcd, sizeof(shadow));
   memset(refract, 0, sizeof(refract));
   pvrgpu_pco_build_shadow_fragment_shared(shadow);
   pvrgpu_pco_build_refract_fragment_shared(refract);
   if (fnv1a64(shadow, sizeof(shadow)) !=
          UINT64_C(0x5d306f7625b3e88e) ||
       memcmp(shadow, refract, sizeof(shadow)) != 0) {
      fail("shadow canonical depth descriptor changed");
   }
}

static void
test_shadow_compile_profile(struct pvrgpu_pco_compiler *compiler,
                            enum pvrgpu_pco_shadow_profile profile)
{
   nir_shader *vs = NULL;
   nir_shader *fs = NULL;
   switch (profile) {
   case PVRGPU_PCO_SHADOW_DEPTH:
      vs = build_refract_prepass_vertex_shader();
      fs = build_refract_prepass_fragment_shader(GLSL_PRECISION_MEDIUM);
      break;
   case PVRGPU_PCO_SHADOW_MASK:
      vs = build_shadow_mask_vertex_shader(GLSL_PRECISION_MEDIUM);
      fs = build_shadow_mask_fragment_shader(GLSL_PRECISION_MEDIUM);
      break;
   case PVRGPU_PCO_SHADOW_SCENE:
      vs = build_shadow_scene_vertex_shader(GLSL_PRECISION_MEDIUM);
      fs = build_shadow_scene_fragment_shader(GLSL_PRECISION_MEDIUM);
      break;
   default:
      fail("invalid shadow test profile");
   }

   uint8_t source_vs_hash[sizeof(vs->info.source_blake3)];
   uint8_t source_fs_hash[sizeof(fs->info.source_blake3)];
   memcpy(source_vs_hash, vs->info.source_blake3, sizeof(source_vs_hash));
   memcpy(source_fs_hash, fs->info.source_blake3, sizeof(source_fs_hash));
   const unsigned expected_uniform_loads =
      profile == PVRGPU_PCO_SHADOW_DEPTH ? 4U :
      profile == PVRGPU_PCO_SHADOW_MASK ? 6U : 8U;
   if (count_intrinsic(vs, nir_intrinsic_load_uniform) !=
          expected_uniform_loads ||
       count_intrinsic(vs, nir_intrinsic_load_push_constant) != 0 ||
       count_intrinsic(fs, nir_intrinsic_load_push_constant) != 0) {
      fail("shadow test NIR did not start in Gallium load_uniform form");
   }

   char error[512] = { 0 };
   struct pvrgpu_pco_graphics_binary binary;
   if (!pvrgpu_pco_compile_shadow(compiler,
                                  vs,
                                  fs,
                                  profile,
                                  &binary,
                                  error,
                                  sizeof(error))) {
      fail(error[0] ? error : "shadow compile failed");
   }

   static const size_t expected_vs_size[] = { 432U, 464U, 728U };
   static const size_t expected_fs_size[] = { 56U, 216U, 56U };
   static const uint64_t expected_vs_hash[] = {
      UINT64_C(0x6e9ad97e49eca9fe),
      UINT64_C(0x79b5f95f5c89ad6c),
      UINT64_C(0x385c48c6c28cd9fc),
   };
   static const uint64_t expected_fs_hash[] = {
      UINT64_C(0xa55a28d91b0f4b9e),
      UINT64_C(0x1ac54b25af8de102),
      UINT64_C(0x24f632ab8095faeb),
   };
   const uint64_t vertex_hash =
      fnv1a64(binary.vertex.data, binary.vertex.size);
   const uint64_t fragment_hash =
      fnv1a64(binary.fragment.data, binary.fragment.size);
   if (binary.vertex.size != expected_vs_size[profile] ||
       binary.fragment.size != expected_fs_size[profile] ||
       vertex_hash != expected_vs_hash[profile] ||
       fragment_hash != expected_fs_hash[profile]) {
      fprintf(stderr,
              "shadow profile=%u PCO got VS=%zu/%016llx FS=%zu/%016llx\n",
              profile,
              binary.vertex.size,
              (unsigned long long)vertex_hash,
              binary.fragment.size,
              (unsigned long long)fragment_hash);
      fail("shadow PCO binary differs from the strict capture fixture");
   }

   static const struct pvrgpu_pco_stage_abi expected_vs_abi[] = {
      { 10, 8, 7, 0, 16, 0, 16, 0 },
      { 16, 4, 8, 0, 32, 0, 32, 0 },
      { 11, 8, 5, 0, 32, 0, 32, 0 },
   };
   static const struct pvrgpu_pco_stage_abi expected_fs_abi[] = {
      { 3, 0, 0, 16, 0, 0, 0, 0 },
      { 11, 0, 0, 20, 20, 0, 0, 0 },
      { 1, 0, 0, 8, 0, 0, 0, 0 },
   };
   if (memcmp(&binary.vertex.abi,
              &expected_vs_abi[profile],
              sizeof(binary.vertex.abi)) != 0 ||
       memcmp(&binary.fragment.abi,
              &expected_fs_abi[profile],
              sizeof(binary.fragment.abi)) != 0) {
      fail("shadow PCO stage ABI changed");
   }

   static const unsigned varying_components[] = { 3U, 4U, 1U };
   const bool mask = profile == PVRGPU_PCO_SHADOW_MASK;
   const unsigned varyings = varying_components[profile];
   if (binary.position_output_start != 0 ||
       binary.position_output_count != 4 ||
       binary.fragment_position_start != 0 ||
       binary.fragment_position_count != 4 ||
       binary.varying_output_start != 4 ||
       binary.varying_output_count != varyings ||
       binary.fragment_varying_start != 4 ||
       binary.fragment_varying_count != varyings * 4 ||
       binary.fragment_texture_descriptor_start != 0 ||
       binary.fragment_texture_descriptor_count != (mask ? 20U : 0U) ||
       binary.fragment_texture_descriptor_stride != (mask ? 20U : 0U)) {
      fail("shadow PCO linkage or descriptor ABI changed");
   }

   if (count_intrinsic(vs, nir_intrinsic_load_uniform) !=
          expected_uniform_loads ||
       count_intrinsic(vs, nir_intrinsic_load_push_constant) != 0 ||
       count_intrinsic(fs, nir_intrinsic_load_push_constant) != 0 ||
       memcmp(source_vs_hash,
              vs->info.source_blake3,
              sizeof(source_vs_hash)) != 0 ||
       memcmp(source_fs_hash,
              fs->info.source_blake3,
              sizeof(source_fs_hash)) != 0) {
      fail("shadow compile modified caller-owned NIR");
   }
   pvrgpu_pco_graphics_binary_finish(&binary);

   fs->info.source_blake3[0] ^= 1;
   memset(error, 0, sizeof(error));
   if (pvrgpu_pco_compile_shadow(compiler,
                                 vs,
                                 fs,
                                 profile,
                                 &binary,
                                 error,
                                 sizeof(error)) ||
       !strstr(error, "source signature mismatch")) {
      fail("shadow source-hash mismatch did not fail closed");
   }
   ralloc_free(vs);
   ralloc_free(fs);
}

static void
test_shadow_precision_gate(struct pvrgpu_pco_compiler *compiler)
{
   nir_shader *vs = build_shadow_mask_vertex_shader(GLSL_PRECISION_MEDIUM);
   nir_shader *fs = build_shadow_mask_fragment_shader(GLSL_PRECISION_HIGH);
   char error[512] = { 0 };
   struct pvrgpu_pco_graphics_binary binary;
   if (pvrgpu_pco_compile_shadow(compiler,
                                 vs,
                                 fs,
                                 PVRGPU_PCO_SHADOW_MASK,
                                 &binary,
                                 error,
                                 sizeof(error)) ||
       !strstr(error, "varying ABI mismatch")) {
      fail("shadow precision mismatch did not fail closed");
   }
   ralloc_free(vs);
   ralloc_free(fs);
}

static void
test_terrain_d6_compile(struct pvrgpu_pco_compiler *compiler)
{
   nir_shader *vs =
      build_terrain_fullscreen_vertex_shader(GLSL_PRECISION_MEDIUM);
   nir_shader *fs = build_terrain_d6_fragment_shader(GLSL_PRECISION_MEDIUM);
   uint8_t source_vs_hash[sizeof(vs->info.source_blake3)];
   uint8_t source_fs_hash[sizeof(fs->info.source_blake3)];
   memcpy(source_vs_hash, vs->info.source_blake3, sizeof(source_vs_hash));
   memcpy(source_fs_hash, fs->info.source_blake3, sizeof(source_fs_hash));

   char error[512] = { 0 };
   struct pvrgpu_pco_graphics_binary binary;
   if (!pvrgpu_pco_compile_terrain(compiler,
                                   vs,
                                   fs,
                                   PVRGPU_PCO_TERRAIN_D6,
                                   &binary,
                                   error,
                                   sizeof(error))) {
      fail(error[0] ? error : "terrain D6 compile failed");
   }
   const uint64_t vertex_hash =
      fnv1a64(binary.vertex.data, binary.vertex.size);
   const uint64_t fragment_hash =
      fnv1a64(binary.fragment.data, binary.fragment.size);
   if (binary.vertex.size != 192 ||
       vertex_hash != UINT64_C(0x081618f544cc6abe) ||
       binary.fragment.size != 232 ||
       fragment_hash != UINT64_C(0x834f6b41c2c7bdd8)) {
      fprintf(stderr,
              "terrain D6 PCO got VS=%zu/%016llx FS=%zu/%016llx\n",
              binary.vertex.size,
              (unsigned long long)vertex_hash,
              binary.fragment.size,
              (unsigned long long)fragment_hash);
      fail("terrain D6 PCO binary differs from strict fixture");
   }
   const struct pvrgpu_pco_stage_abi expected_vs_abi =
      { 6, 4, 6, 0, 8, 0, 8, 0 };
   const struct pvrgpu_pco_stage_abi expected_fs_abi =
      { 17, 0, 0, 12, 24, 20, 4, 0 };
   if (memcmp(&binary.vertex.abi,
              &expected_vs_abi,
              sizeof(expected_vs_abi)) != 0 ||
       memcmp(&binary.fragment.abi,
              &expected_fs_abi,
              sizeof(expected_fs_abi)) != 0 ||
       binary.position_output_start != 0 ||
       binary.position_output_count != 4 ||
       binary.varying_output_start != 4 ||
       binary.varying_output_count != 2 ||
       binary.fragment_position_start != 0 ||
       binary.fragment_position_count != 4 ||
       binary.fragment_varying_start != 4 ||
       binary.fragment_varying_count != 8 ||
       binary.fragment_texture_descriptor_start != 0 ||
       binary.fragment_texture_descriptor_count != 20 ||
       binary.fragment_texture_descriptor_stride != 20) {
      fail("terrain D6 PCO ABI/linkage changed");
   }
   if (count_intrinsic(vs, nir_intrinsic_load_uniform) != 2 ||
       count_intrinsic(vs, nir_intrinsic_load_push_constant) != 0 ||
       count_intrinsic(fs, nir_intrinsic_load_uniform) != 1 ||
       count_intrinsic(fs, nir_intrinsic_load_push_constant) != 0 ||
       memcmp(source_vs_hash,
              vs->info.source_blake3,
              sizeof(source_vs_hash)) != 0 ||
       memcmp(source_fs_hash,
              fs->info.source_blake3,
              sizeof(source_fs_hash)) != 0) {
      fail("terrain compile modified caller-owned NIR");
   }
   pvrgpu_pco_graphics_binary_finish(&binary);

   fs->info.source_blake3[0] ^= 1;
   memset(error, 0, sizeof(error));
   if (pvrgpu_pco_compile_terrain(compiler,
                                  vs,
                                  fs,
                                  PVRGPU_PCO_TERRAIN_D6,
                                  &binary,
                                  error,
                                  sizeof(error)) ||
       !strstr(error, "source signature mismatch")) {
      fail("terrain source-hash mismatch did not fail closed");
   }
   fs->info.source_blake3[0] ^= 1;

   nir_shader *bad_precision_fs =
      build_terrain_d6_fragment_shader(GLSL_PRECISION_HIGH);
   memset(error, 0, sizeof(error));
   if (pvrgpu_pco_compile_terrain(compiler,
                                  vs,
                                  bad_precision_fs,
                                  PVRGPU_PCO_TERRAIN_D6,
                                  &binary,
                                  error,
                                  sizeof(error)) ||
       !strstr(error, "input ABI mismatch")) {
      fail("terrain precision mismatch did not fail closed");
   }
   ralloc_free(vs);
   ralloc_free(fs);
   ralloc_free(bad_precision_fs);
}

static void
test_terrain_d1_compile(struct pvrgpu_pco_compiler *compiler)
{
   nir_shader *vs = build_terrain_d1_vertex_shader(GLSL_PRECISION_MEDIUM);
   nir_shader *fs = build_terrain_d1_fragment_shader(GLSL_PRECISION_MEDIUM);
   uint8_t source_vs_hash[sizeof(vs->info.source_blake3)];
   uint8_t source_fs_hash[sizeof(fs->info.source_blake3)];
   memcpy(source_vs_hash, vs->info.source_blake3, sizeof(source_vs_hash));
   memcpy(source_fs_hash, fs->info.source_blake3, sizeof(source_fs_hash));
   char error[512] = { 0 };
   struct pvrgpu_pco_graphics_binary binary;
   if (!pvrgpu_pco_compile_terrain(compiler,
                                   vs,
                                   fs,
                                   PVRGPU_PCO_TERRAIN_D1,
                                   &binary,
                                   error,
                                   sizeof(error))) {
      fail(error[0] ? error : "terrain D1 compile failed");
   }
   const uint64_t vertex_hash =
      fnv1a64(binary.vertex.data, binary.vertex.size);
   const uint64_t fragment_hash =
      fnv1a64(binary.fragment.data, binary.fragment.size);
   if (binary.vertex.size != 200U ||
       vertex_hash != UINT64_C(0x9abe96cad5fe9f4e) ||
       binary.fragment.size != 39560U ||
       fragment_hash != UINT64_C(0xe5ede07e1317d604)) {
      fprintf(stderr,
              "terrain D1 PCO got VS=%zu/%016llx FS=%zu/%016llx\n",
              binary.vertex.size,
              (unsigned long long)vertex_hash,
              binary.fragment.size,
              (unsigned long long)fragment_hash);
      fail("terrain D1 PCO binary differs from actual replay fixture");
   }
   const struct pvrgpu_pco_stage_abi expected_vs_abi =
      { 7, 4, 6, 0, 8, 0, 8, 0 };
   const struct pvrgpu_pco_stage_abi expected_fs_abi =
      { 49, 0, 0, 12, 4, 0, 4, 0 };
   if (memcmp(&binary.vertex.abi,
              &expected_vs_abi,
              sizeof(expected_vs_abi)) != 0 ||
       memcmp(&binary.fragment.abi,
              &expected_fs_abi,
              sizeof(expected_fs_abi)) != 0 ||
       binary.position_output_start != 0 ||
       binary.position_output_count != 4 ||
       binary.varying_output_start != 4 ||
       binary.varying_output_count != 2 ||
       binary.fragment_position_start != 0 ||
       binary.fragment_position_count != 4 ||
       binary.fragment_varying_start != 4 ||
       binary.fragment_varying_count != 8 ||
       binary.fragment_texture_descriptor_start != 0 ||
       binary.fragment_texture_descriptor_count != 0 ||
       binary.fragment_texture_descriptor_stride != 0) {
      fprintf(stderr,
              "terrain D1 ABI got VS=(%u,%u,%u,%u,%u,%u,%u,%u) "
              "FS=(%u,%u,%u,%u,%u,%u,%u,%u)\n",
              binary.vertex.abi.temps,
              binary.vertex.abi.vertex_inputs,
              binary.vertex.abi.vertex_outputs,
              binary.vertex.abi.coefficients,
              binary.vertex.abi.shareds,
              binary.vertex.abi.push_constant_start,
              binary.vertex.abi.push_constant_count,
              binary.vertex.abi.entry_offset,
              binary.fragment.abi.temps,
              binary.fragment.abi.vertex_inputs,
              binary.fragment.abi.vertex_outputs,
              binary.fragment.abi.coefficients,
              binary.fragment.abi.shareds,
              binary.fragment.abi.push_constant_start,
              binary.fragment.abi.push_constant_count,
              binary.fragment.abi.entry_offset);
      fail("terrain D1 PCO ABI/linkage changed");
   }
   if (count_intrinsic(vs, nir_intrinsic_load_uniform) != 2 ||
       count_intrinsic(vs, nir_intrinsic_load_push_constant) != 0 ||
       count_intrinsic(fs, nir_intrinsic_load_uniform) != 1 ||
       count_intrinsic(fs, nir_intrinsic_load_push_constant) != 0 ||
       memcmp(source_vs_hash,
              vs->info.source_blake3,
              sizeof(source_vs_hash)) != 0 ||
       memcmp(source_fs_hash,
              fs->info.source_blake3,
              sizeof(source_fs_hash)) != 0) {
      fail("terrain D1 compile modified caller-owned NIR");
   }
   pvrgpu_pco_graphics_binary_finish(&binary);
   ralloc_free(vs);
   ralloc_free(fs);
}

static void
test_terrain_d2_compile(struct pvrgpu_pco_compiler *compiler)
{
   nir_shader *vs =
      build_terrain_fullscreen_vertex_shader(GLSL_PRECISION_MEDIUM);
   nir_shader *fs = build_terrain_d2_fragment_shader(GLSL_PRECISION_MEDIUM);
   char error[512] = { 0 };
   struct pvrgpu_pco_graphics_binary binary;
   if (!pvrgpu_pco_compile_terrain(compiler,
                                   vs,
                                   fs,
                                   PVRGPU_PCO_TERRAIN_D2,
                                   &binary,
                                   error,
                                   sizeof(error))) {
      fail(error[0] ? error : "terrain D2 compile failed");
   }
   const uint64_t vertex_hash =
      fnv1a64(binary.vertex.data, binary.vertex.size);
   const uint64_t fragment_hash =
      fnv1a64(binary.fragment.data, binary.fragment.size);
   if (binary.vertex.size != 192 ||
       vertex_hash != UINT64_C(0x081618f544cc6abe) ||
       binary.fragment.size != 472 ||
       fragment_hash != UINT64_C(0xf37c4b6785de2e44)) {
      fprintf(stderr,
              "terrain D2 PCO got VS=%zu/%016llx FS=%zu/%016llx\n",
              binary.vertex.size,
              (unsigned long long)vertex_hash,
              binary.fragment.size,
              (unsigned long long)fragment_hash);
      fail("terrain D2 PCO binary differs from strict fixture");
   }
   const struct pvrgpu_pco_stage_abi expected_vs_abi =
      { 6, 4, 6, 0, 8, 0, 8, 0 };
   const struct pvrgpu_pco_stage_abi expected_fs_abi =
      { 34, 0, 0, 12, 28, 20, 8, 0 };
   if (memcmp(&binary.vertex.abi,
              &expected_vs_abi,
              sizeof(expected_vs_abi)) != 0 ||
       memcmp(&binary.fragment.abi,
              &expected_fs_abi,
              sizeof(expected_fs_abi)) != 0 ||
       binary.fragment_texture_descriptor_start != 0 ||
       binary.fragment_texture_descriptor_count != 20 ||
       binary.fragment_texture_descriptor_stride != 20) {
      fail("terrain D2 PCO ABI/linkage changed");
   }
   pvrgpu_pco_graphics_binary_finish(&binary);
   ralloc_free(vs);
   ralloc_free(fs);
}

static void
test_terrain_d3_compile(struct pvrgpu_pco_compiler *compiler)
{
   nir_shader *vs = build_terrain_d3_vertex_shader(GLSL_PRECISION_MEDIUM);
   nir_shader *fs = build_terrain_d3_fragment_shader(GLSL_PRECISION_MEDIUM);
   uint8_t source_vs_hash[sizeof(vs->info.source_blake3)];
   uint8_t source_fs_hash[sizeof(fs->info.source_blake3)];
   memcpy(source_vs_hash, vs->info.source_blake3, sizeof(source_vs_hash));
   memcpy(source_fs_hash, fs->info.source_blake3, sizeof(source_fs_hash));
   char error[512] = { 0 };
   struct pvrgpu_pco_graphics_binary binary;
   if (!pvrgpu_pco_compile_terrain(compiler,
                                   vs,
                                   fs,
                                   PVRGPU_PCO_TERRAIN_D3,
                                   &binary,
                                   error,
                                   sizeof(error))) {
      fail(error[0] ? error : "terrain D3 compile failed");
   }
   const uint64_t vertex_hash =
      fnv1a64(binary.vertex.data, binary.vertex.size);
   const uint64_t fragment_hash =
      fnv1a64(binary.fragment.data, binary.fragment.size);
   if (binary.vertex.size != 2440U ||
       vertex_hash != UINT64_C(0xde47363e398a2bcc) ||
       binary.fragment.size != 2312U ||
       fragment_hash != UINT64_C(0x02a543327636e98f)) {
      fprintf(stderr,
              "terrain D3 PCO got VS=%zu/%016llx FS=%zu/%016llx\n",
              binary.vertex.size,
              (unsigned long long)vertex_hash,
              binary.fragment.size,
              (unsigned long long)fragment_hash);
      fail("terrain D3 PCO binary differs from strict fixture");
   }
   const struct pvrgpu_pco_stage_abi expected_vs_abi =
      { 44, 16, 18, 0, 96, 40, 56, 0 };
   const struct pvrgpu_pco_stage_abi expected_fs_abi =
      { 36, 0, 0, 60, 164, 100, 64, 0 };
   if (memcmp(&binary.vertex.abi,
              &expected_vs_abi,
              sizeof(expected_vs_abi)) != 0 ||
       memcmp(&binary.fragment.abi,
              &expected_fs_abi,
              sizeof(expected_fs_abi)) != 0 ||
       binary.position_output_start != 0 ||
       binary.position_output_count != 4 ||
       binary.varying_output_start != 4 ||
       binary.varying_output_count != 14 ||
       binary.fragment_position_start != 0 ||
       binary.fragment_position_count != 4 ||
       binary.fragment_varying_start != 4 ||
       binary.fragment_varying_count != 56 ||
       binary.fragment_texture_descriptor_start != 0 ||
       binary.fragment_texture_descriptor_count != 100 ||
       binary.fragment_texture_descriptor_stride != 20) {
      fail("terrain D3 PCO ABI/linkage changed");
   }
   if (count_intrinsic(vs, nir_intrinsic_load_uniform) != 14 ||
       count_intrinsic(vs, nir_intrinsic_load_push_constant) != 0 ||
       count_intrinsic(fs, nir_intrinsic_load_uniform) != 16 ||
       count_intrinsic(fs, nir_intrinsic_load_push_constant) != 0 ||
       memcmp(source_vs_hash,
              vs->info.source_blake3,
              sizeof(source_vs_hash)) != 0 ||
       memcmp(source_fs_hash,
              fs->info.source_blake3,
              sizeof(source_fs_hash)) != 0) {
      fail("terrain D3 compile modified caller-owned NIR");
   }
   pvrgpu_pco_graphics_binary_finish(&binary);
   ralloc_free(vs);
   ralloc_free(fs);
}

static void
test_terrain_blur_compile(struct pvrgpu_pco_compiler *compiler)
{
   static const enum pvrgpu_pco_terrain_profile profiles[] = {
      PVRGPU_PCO_TERRAIN_D4,
      PVRGPU_PCO_TERRAIN_D5,
      PVRGPU_PCO_TERRAIN_D7,
      PVRGPU_PCO_TERRAIN_D8,
   };
   static const unsigned profile_numbers[] = { 4U, 5U, 7U, 8U };
   static const size_t expected_fragment_sizes[] = { 680U, 680U, 1144U,
                                                     1144U };
   static const uint64_t expected_fragment_hashes[] = {
      UINT64_C(0xa6e167a6cd068362),
      UINT64_C(0x6b171a8aaec8fa96),
      UINT64_C(0x0196ab8e374c830d),
      UINT64_C(0x1fd6e9c5b3efd7bd),
   };
   static const unsigned expected_fragment_temps[] = { 31U, 31U, 33U,
                                                       33U };

   for (unsigned i = 0; i < ARRAY_SIZE(profiles); ++i) {
      nir_shader *vs =
         build_terrain_fullscreen_vertex_shader(GLSL_PRECISION_MEDIUM);
      nir_shader *fs = build_terrain_blur_fragment_shader(
         profiles[i], GLSL_PRECISION_MEDIUM);
      uint8_t source_vs_hash[sizeof(vs->info.source_blake3)];
      uint8_t source_fs_hash[sizeof(fs->info.source_blake3)];
      memcpy(source_vs_hash, vs->info.source_blake3, sizeof(source_vs_hash));
      memcpy(source_fs_hash, fs->info.source_blake3, sizeof(source_fs_hash));
      char error[512] = { 0 };
      struct pvrgpu_pco_graphics_binary binary;
      if (!pvrgpu_pco_compile_terrain(compiler,
                                      vs,
                                      fs,
                                      profiles[i],
                                      &binary,
                                      error,
                                      sizeof(error))) {
         fail(error[0] ? error : "terrain blur compile failed");
      }

      const uint64_t vertex_hash =
         fnv1a64(binary.vertex.data, binary.vertex.size);
      const uint64_t fragment_hash =
         fnv1a64(binary.fragment.data, binary.fragment.size);
      if (binary.vertex.size != 192U ||
          vertex_hash != UINT64_C(0x081618f544cc6abe) ||
          binary.fragment.size != expected_fragment_sizes[i] ||
          fragment_hash != expected_fragment_hashes[i]) {
         fprintf(stderr,
                 "terrain D%u PCO got VS=%zu/%016llx FS=%zu/%016llx\n",
                 profile_numbers[i],
                 binary.vertex.size,
                 (unsigned long long)vertex_hash,
                 binary.fragment.size,
                 (unsigned long long)fragment_hash);
         fail("terrain blur PCO binary differs from strict fixture");
      }

      const struct pvrgpu_pco_stage_abi expected_vs_abi =
         { 6, 4, 6, 0, 8, 0, 8, 0 };
      const struct pvrgpu_pco_stage_abi expected_fs_abi = {
         expected_fragment_temps[i], 0, 0, 12, 20, 20, 0, 0
      };
      if (memcmp(&binary.vertex.abi,
                 &expected_vs_abi,
                 sizeof(expected_vs_abi)) != 0 ||
          memcmp(&binary.fragment.abi,
                 &expected_fs_abi,
                 sizeof(expected_fs_abi)) != 0 ||
          binary.position_output_start != 0 ||
          binary.position_output_count != 4 ||
          binary.varying_output_start != 4 ||
          binary.varying_output_count != 2 ||
          binary.fragment_position_start != 0 ||
          binary.fragment_position_count != 4 ||
          binary.fragment_varying_start != 4 ||
          binary.fragment_varying_count != 8 ||
          binary.fragment_texture_descriptor_start != 0 ||
          binary.fragment_texture_descriptor_count != 20 ||
          binary.fragment_texture_descriptor_stride != 20) {
         fail("terrain blur PCO ABI/linkage changed");
      }
      if (count_intrinsic(vs, nir_intrinsic_load_uniform) != 2 ||
          count_intrinsic(vs, nir_intrinsic_load_push_constant) != 0 ||
          count_intrinsic(fs, nir_intrinsic_load_uniform) != 0 ||
          count_intrinsic(fs, nir_intrinsic_load_push_constant) != 0 ||
          memcmp(source_vs_hash,
                 vs->info.source_blake3,
                 sizeof(source_vs_hash)) != 0 ||
          memcmp(source_fs_hash,
                 fs->info.source_blake3,
                 sizeof(source_fs_hash)) != 0) {
         fail("terrain blur compile modified caller-owned NIR");
      }
      pvrgpu_pco_graphics_binary_finish(&binary);
      ralloc_free(vs);
      ralloc_free(fs);
   }
}

int main(void)
{
   test_refract_fragment_descriptors();
   test_terrain_texture_descriptors();
   test_shadow_fragment_descriptor();
   glsl_type_singleton_init_or_ref();
   nir_shader *vs = build_vertex_shader();
   nir_shader *fs = build_fragment_shader();
   if (count_intrinsic(vs, nir_intrinsic_load_uniform) != 4 ||
       count_intrinsic(fs, nir_intrinsic_load_uniform) != 1 ||
       count_intrinsic(vs, nir_intrinsic_load_push_constant) != 0 ||
       count_intrinsic(fs, nir_intrinsic_load_push_constant) != 0) {
      fail("test NIR did not start in Gallium load_uniform form");
   }

   char error[512] = { 0 };
   struct pvrgpu_pco_compiler *compiler =
      pvrgpu_pco_compiler_create(error, sizeof(error));
   if (!compiler)
      fail(error[0] ? error : "failed to create compiler");

   struct pvrgpu_pco_graphics_binary binary;
   if (!pvrgpu_pco_compile_conditionals(compiler,
                                        vs,
                                        fs,
                                        PIPE_FORMAT_R32G32B32_FLOAT,
                                        &binary,
                                        error,
                                        sizeof(error))) {
      fail(error[0] ? error : "conditionals compile failed");
   }

   if (!binary.vertex.data || !binary.vertex.size || !binary.fragment.data ||
       !binary.fragment.size)
      fail("PCO returned an empty owned binary");
   const uint64_t vertex_hash = fnv1a64(binary.vertex.data, binary.vertex.size);
   if (binary.vertex.size != 520 ||
       vertex_hash != UINT64_C(0x88ef7e84a69a0db7)) {
      fprintf(stderr,
              "VS PCO got size=%zu fnv1a64=%016llx\n",
              binary.vertex.size,
              (unsigned long long)vertex_hash);
      fail("VS PCO binary differs from the pinned gx6250 decoder fixture");
   }
   const uint64_t fragment_hash =
      fnv1a64(binary.fragment.data, binary.fragment.size);
   if (binary.fragment.size != 520 ||
       fragment_hash != UINT64_C(0xe33aaff7bc4d515c)) {
      fprintf(stderr,
              "FS PCO got size=%zu fnv1a64=%016llx\n",
              binary.fragment.size,
              (unsigned long long)fragment_hash);
      fail("FS PCO binary differs from the pinned gx6250 decoder fixture");
   }
   if (binary.vertex.abi.temps != 10 ||
       binary.vertex.abi.vertex_inputs != 4 ||
       binary.vertex.abi.vertex_outputs != 4 ||
       binary.vertex.abi.coefficients != 0 ||
       binary.vertex.abi.shareds != 16 ||
       binary.vertex.abi.push_constant_start != 0 ||
       binary.vertex.abi.push_constant_count != 16 ||
       binary.vertex.abi.entry_offset != 0) {
      fail("VS PCO ABI is not TEMP10/VTXIN4/VTXOUT4/COEFF0/SH0..15/ENTRY0");
   }
   if (binary.fragment.abi.temps != 4 ||
       binary.fragment.abi.vertex_inputs != 0 ||
       binary.fragment.abi.vertex_outputs != 0 ||
       binary.fragment.abi.coefficients != 0 ||
       binary.fragment.abi.shareds != 4 ||
       binary.fragment.abi.push_constant_start != 0 ||
       binary.fragment.abi.push_constant_count != 4 ||
       binary.fragment.abi.entry_offset != 0) {
      fail("FS PCO ABI is not TEMP4/VTXIN0/VTXOUT0/COEFF0/SH0..3/ENTRY0");
   }
   if (binary.position_output_start != 0 || binary.position_output_count != 4)
      fail("position linkage is not VTXOUT0..3");
   if (binary.fragment_position_start != 0 ||
       binary.fragment_position_count != 0)
      fail("fragment gl_FragCoord must not consume a varying linkage");

   /* The compile API promises clone-before-lower ownership semantics. */
   if (count_intrinsic(vs, nir_intrinsic_load_uniform) != 4 ||
       count_intrinsic(fs, nir_intrinsic_load_uniform) != 1 ||
       count_intrinsic(vs, nir_intrinsic_load_push_constant) != 0 ||
       count_intrinsic(fs, nir_intrinsic_load_push_constant) != 0) {
      fail("compile modified caller-owned NIR");
   }

   pvrgpu_pco_graphics_binary_finish(&binary);
   if (binary.vertex.data || binary.vertex.size || binary.fragment.data ||
       binary.fragment.size)
      fail("binary finish did not clear ownership fields");

   memset(error, 0, sizeof(error));
   if (pvrgpu_pco_compile_conditionals(compiler,
                                       vs,
                                       fs,
                                       PIPE_FORMAT_R32G32_FLOAT,
                                       &binary,
                                       error,
                                       sizeof(error)) ||
       !strstr(error, "R32G32B32_FLOAT")) {
      fail("unsupported vertex format did not fail closed");
   }

   test_refract_compile_profile(compiler, PVRGPU_PCO_REFRACT_PREPASS);
   test_refract_compile_profile(compiler, PVRGPU_PCO_REFRACT_COMPOSITE);
   test_refract_precision_gate(compiler);
   test_shadow_compile_profile(compiler, PVRGPU_PCO_SHADOW_DEPTH);
   test_shadow_compile_profile(compiler, PVRGPU_PCO_SHADOW_MASK);
   test_shadow_compile_profile(compiler, PVRGPU_PCO_SHADOW_SCENE);
   test_shadow_precision_gate(compiler);
   test_terrain_d1_compile(compiler);
   test_terrain_d2_compile(compiler);
   test_terrain_d3_compile(compiler);
   test_terrain_blur_compile(compiler);
   test_terrain_d6_compile(compiler);

   nir_shader *texture_vs = build_texture_vertex_shader();
   nir_shader *texture_fs =
      build_texture_fragment_shader(GLSL_PRECISION_MEDIUM);
   uint8_t texture_vs_hash[sizeof(texture_vs->info.source_blake3)];
   uint8_t texture_fs_hash[sizeof(texture_fs->info.source_blake3)];
   memcpy(texture_vs_hash,
          texture_vs->info.source_blake3,
          sizeof(texture_vs_hash));
   memcpy(texture_fs_hash,
          texture_fs->info.source_blake3,
          sizeof(texture_fs_hash));
   if (count_intrinsic(texture_vs, nir_intrinsic_load_uniform) != 8 ||
       count_intrinsic(texture_vs, nir_intrinsic_load_push_constant) != 0 ||
       count_alu(texture_fs, nir_op_fmul) != 1 ||
       count_alu(texture_fs, nir_op_f2f16_rtz) != 0 ||
       count_alu(texture_fs, nir_op_f2f16_rtne) != 0 ||
       count_alu(texture_fs, nir_op_f2f32) != 0) {
      fail("texture test NIR did not start in canonical fp32 Gallium form");
   }

   memset(error, 0, sizeof(error));
   if (!pvrgpu_pco_compile_texture(compiler,
                                   texture_vs,
                                   texture_fs,
                                   &binary,
                                   error,
                                   sizeof(error))) {
      fail(error[0] ? error : "texture compile failed");
   }
   const uint64_t texture_vertex_hash =
      fnv1a64(binary.vertex.data, binary.vertex.size);
   const uint64_t texture_fragment_hash =
      fnv1a64(binary.fragment.data, binary.fragment.size);
   if (binary.vertex.size != 752 ||
       texture_vertex_hash != UINT64_C(0x4a2faca1e62f6595) ||
       binary.fragment.size != 304 ||
       texture_fragment_hash != UINT64_C(0x1e2d215432179c29)) {
      fprintf(stderr,
              "texture PCO got VS=%zu/%016llx FS=%zu/%016llx\n",
              binary.vertex.size,
              (unsigned long long)texture_vertex_hash,
              binary.fragment.size,
              (unsigned long long)texture_fragment_hash);
      fail("texture PCO binary differs from the strict mediump fixture");
   }
   if (binary.vertex.abi.temps != 11 ||
       binary.vertex.abi.vertex_inputs != 12 ||
       binary.vertex.abi.vertex_outputs != 7 ||
       binary.vertex.abi.coefficients != 0 ||
       binary.vertex.abi.shareds != 32 ||
       binary.vertex.abi.push_constant_start != 0 ||
       binary.vertex.abi.push_constant_count != 32 ||
       binary.vertex.abi.entry_offset != 0 ||
       binary.fragment.abi.temps != 8 ||
       binary.fragment.abi.vertex_inputs != 0 ||
       binary.fragment.abi.vertex_outputs != 0 ||
       binary.fragment.abi.coefficients != 16 ||
       binary.fragment.abi.shareds != 20 ||
       binary.fragment.abi.push_constant_start != 0 ||
       binary.fragment.abi.push_constant_count != 0 ||
       binary.fragment.abi.entry_offset != 0) {
      fail("texture PCO ABI changed");
   }
   if (binary.position_output_start != 0 ||
       binary.position_output_count != 4 ||
       binary.varying_output_start != 4 ||
       binary.varying_output_count != 3 ||
       binary.fragment_position_start != 0 ||
       binary.fragment_position_count != 4 ||
       binary.fragment_varying_start != 4 ||
       binary.fragment_varying_count != 12 ||
       binary.fragment_texture_descriptor_start != 0 ||
       binary.fragment_texture_descriptor_count != 20 ||
       binary.fragment_texture_descriptor_stride != 20) {
      fail("texture PCO linkage or descriptor ABI changed");
   }

   /* Compile must lower only its private clones.  The source graph retains
    * the vector multiply and contains no precision conversions afterward. */
   if (count_intrinsic(texture_vs, nir_intrinsic_load_uniform) != 8 ||
       count_intrinsic(texture_vs, nir_intrinsic_load_push_constant) != 0 ||
       count_alu(texture_fs, nir_op_fmul) != 1 ||
       count_alu(texture_fs, nir_op_f2f16_rtz) != 0 ||
       count_alu(texture_fs, nir_op_f2f16_rtne) != 0 ||
       count_alu(texture_fs, nir_op_f2f32) != 0 ||
       memcmp(texture_vs_hash,
              texture_vs->info.source_blake3,
              sizeof(texture_vs_hash)) != 0 ||
       memcmp(texture_fs_hash,
              texture_fs->info.source_blake3,
              sizeof(texture_fs_hash)) != 0) {
      fail("texture compile modified caller-owned NIR");
   }
   pvrgpu_pco_graphics_binary_finish(&binary);

   texture_fs->info.source_blake3[0] ^= 1;
   memset(error, 0, sizeof(error));
   if (pvrgpu_pco_compile_texture(compiler,
                                  texture_vs,
                                  texture_fs,
                                  &binary,
                                  error,
                                  sizeof(error)) ||
       !strstr(error, "source signature mismatch")) {
      fail("texture source-hash mismatch did not fail closed");
   }
   texture_fs->info.source_blake3[0] ^= 1;

   nir_shader *bad_precision_fs =
      build_texture_fragment_shader(GLSL_PRECISION_HIGH);
   memset(error, 0, sizeof(error));
   if (pvrgpu_pco_compile_texture(compiler,
                                  texture_vs,
                                  bad_precision_fs,
                                  &binary,
                                  error,
                                  sizeof(error)) ||
       !strstr(error, "varying ABI mismatch")) {
      fail("texture precision mismatch did not fail closed");
   }

   pvrgpu_pco_compiler_destroy(compiler);
   ralloc_free(vs);
   ralloc_free(fs);
   ralloc_free(texture_vs);
   ralloc_free(texture_fs);
   ralloc_free(bad_precision_fs);
   glsl_type_singleton_decref();
   puts("pvrgpu_pco_lowering_test: PASS");
   return 0;
}
