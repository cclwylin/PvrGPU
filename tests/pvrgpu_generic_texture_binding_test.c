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
void check_generic_explicit_texture_program(const uint8_t *bytes, size_t size,
                                            unsigned kind);
void check_generic_vertex_texture_program(const uint8_t *bytes, size_t size,
                                          const uint32_t *shared, unsigned shared_count,
                                          unsigned varying_start, unsigned kind);
void check_generic_derivative_program(const uint8_t *bytes, size_t size,
                                     unsigned kind);
void check_generic_sample_program(const uint8_t *bytes, size_t size, unsigned kind,
                                  bool multisample_target);
void dump_generic_sample_program(const uint8_t *bytes, size_t size, unsigned kind);
void check_generic_shadow_program(const uint8_t *bytes, size_t size,
                                  const uint32_t *descriptor, unsigned kind,
                                  bool vertex, unsigned varying_start);
void check_generic_interpolation_program(const uint8_t *bytes, size_t size,
                                         unsigned kind);
void check_generic_dynamic_uniform_program(const uint8_t *bytes, size_t size,
                                           bool vertex, unsigned base,
                                           unsigned range, unsigned push_start,
                                           unsigned shared_count,
                                           unsigned varying_start, bool branch);

static nir_shader *native_dynamic_uniform_shader(bool vertex, unsigned base,
                                                  unsigned range, bool branch)
{
   nir_builder b = nir_builder_init_simple_shader(vertex ? MESA_SHADER_VERTEX : MESA_SHADER_FRAGMENT,
      pco_nir_options(), "native_dynamic_uniform");
   nir_variable *output = nir_variable_create(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "value");
   output->data.location = vertex ? VARYING_SLOT_VAR0 : FRAG_RESULT_DATA0;
   if (vertex) {
      nir_variable *input = nir_variable_create(b.shader, nir_var_shader_in,
         glsl_vec4_type(), "position");
      input->data.location = VERT_ATTRIB_GENERIC0;
      nir_variable *position = nir_variable_create(b.shader, nir_var_shader_out,
         glsl_vec4_type(), "gl_Position");
      position->data.location = VARYING_SLOT_POS;
      nir_store_var(&b, position, nir_load_var(&b, input), 15);
   }
   nir_def *selectors = nir_load_uniform(&b, 2, 32, nir_imm_int(&b, 0),
      .base = 0, .range = 1, .dest_type = nir_type_uint32);
   nir_def *index = nir_channel(&b, selectors, 0);
   if (branch) {
      nir_push_if(&b, nir_ine_imm(&b, nir_channel(&b, selectors, 1), 0));
      nir_def *incremented = nir_iadd_imm(&b, index, 1);
      nir_push_else(&b, NULL);
      nir_def *unchanged = nir_mov(&b, index);
      nir_pop_if(&b, NULL);
      index = nir_if_phi(&b, incremented, unchanged);
   }
   nir_def *value = nir_load_uniform(&b, 4, 32, index,
      .base = base, .range = range, .dest_type = nir_type_float32);
   if (!vertex) {
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 1);
      tex->op = nir_texop_tex;
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
      tex->coord_components = 2;
      tex->dest_type = nir_type_float32;
      tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, nir_imm_vec2(&b, .25f, .75f));
      nir_def_init(&tex->instr, &tex->def, 4, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      value = nir_fadd(&b, value, &tex->def);
   }
   nir_store_var(&b, output, value, 15);
   nir_shader_gather_info(b.shader, b.impl);
   b.shader->info.num_textures = vertex ? 0 : 1;
   return b.shader;
}

static nir_shader *native_interpolation_shader(unsigned kind, bool vertex)
{
   nir_builder b = nir_builder_init_simple_shader(vertex ? MESA_SHADER_VERTEX : MESA_SHADER_FRAGMENT,
      pco_nir_options(), "native_explicit_interpolation");
   const bool array = kind == 5 || kind == 7;
   const struct glsl_type *type = array ? glsl_array_type(glsl_vec4_type(), 2, 0) : glsl_vec4_type();
   nir_variable *varying = nir_variable_create(b.shader,
      vertex ? nir_var_shader_out : nir_var_shader_in, type, "interpolated");
   varying->data.location = VARYING_SLOT_VAR0;
   varying->data.interpolation = kind & 1 ? INTERP_MODE_NOPERSPECTIVE : INTERP_MODE_SMOOTH;
   if (vertex) {
      nir_variable *input = nir_variable_create(b.shader, nir_var_shader_in,
         glsl_vec4_type(), "position");
      input->data.location = VERT_ATTRIB_GENERIC0;
      nir_variable *clip = nir_variable_create(b.shader, nir_var_shader_out,
         glsl_vec4_type(), "gl_Position");
      clip->data.location = VARYING_SLOT_POS;
      nir_def *value = nir_load_var(&b, input);
      nir_store_var(&b, clip, value, 15);
      if (array) {
         for (unsigned element = 0; element < 2; ++element)
            nir_store_deref(&b, nir_build_deref_array_imm(&b,
               nir_build_deref_var(&b, varying), element), nir_fadd_imm(&b, value, element), 15);
      } else {
         nir_store_var(&b, varying, value, 15);
      }
   } else {
      nir_variable *output = nir_variable_create(b.shader, nir_var_shader_out,
         glsl_vec4_type(), "color");
      output->data.location = FRAG_RESULT_DATA0;
      nir_def *parameter = nir_load_uniform(&b, 4, 32, nir_imm_int(&b, 0),
         .base=0, .range=1, .dest_type=nir_type_uint32);
      nir_def *value;
      if (kind == 4) {
         nir_def *bary = nir_load_barycentric_at_offset(&b, 32,
            nir_trim_vector(&b, parameter, 2), .interp_mode=INTERP_MODE_SMOOTH);
         value = nir_load_interpolated_input(&b, 1, 32, bary, nir_imm_int(&b, 0),
            .component=2, .io_semantics={.location=VARYING_SLOT_VAR0, .num_slots=1});
         value = nir_vec4(&b, value, value, value, nir_imm_float(&b, 1));
      } else {
         nir_deref_instr *input = nir_build_deref_var(&b, varying);
         if (array)
            input = nir_build_deref_array_imm(&b, input, 1);
         value = kind == 6 ? nir_interp_deref_at_centroid(&b, 4, 32, &input->def) :
            kind == 2 || kind == 3 ?
            nir_interp_deref_at_sample(&b, 4, 32, &input->def, nir_channel(&b, parameter, 0)) :
            nir_interp_deref_at_offset(&b, 4, 32, &input->def, nir_trim_vector(&b, parameter, 2));
         if (array) {
            nir_deref_instr *first_input = nir_build_deref_array_imm(&b,
               nir_build_deref_var(&b, varying), 0);
            nir_def *first = kind == 7 ? nir_load_deref(&b, first_input) :
               nir_interp_deref_at_offset(&b, 4, 32, &first_input->def, nir_imm_vec2(&b, .0625f, 0));
            value = nir_vec4(&b, nir_channel(&b, value, 0), nir_channel(&b, value, 1),
               nir_channel(&b, value, 2), nir_channel(&b, first, 0));
         }
      }
      nir_store_var(&b, output, value, 15);
   }
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *native_shadow_shader(unsigned kind, bool vertex)
{
   nir_builder b = nir_builder_init_simple_shader(vertex ? MESA_SHADER_VERTEX : MESA_SHADER_FRAGMENT,
      pco_nir_options(), "native_nearest_shadow_fragment");
   nir_variable *output = nir_variable_create(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "color");
   output->data.location = vertex ? VARYING_SLOT_VAR0 : FRAG_RESULT_DATA0;
   if (vertex) {
      nir_variable *position = nir_variable_create(b.shader, nir_var_shader_in,
         glsl_vec4_type(), "position");
      position->data.location = VERT_ATTRIB_GENERIC0;
      nir_variable *clip = nir_variable_create(b.shader, nir_var_shader_out,
         glsl_vec4_type(), "gl_Position");
      clip->data.location = VARYING_SLOT_POS;
      nir_store_var(&b, clip, nir_load_var(&b, position), 15);
   }
   nir_def *input = nir_load_uniform(&b, 4, 32, nir_imm_int(&b, 0),
      .base=0, .range=1, .dest_type=nir_type_float32);
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, kind == 0 ? 2 : kind == 1 ? 3 : 4);
   tex->op = kind == 0 ? nir_texop_tex : kind == 1 ? nir_texop_txl : nir_texop_txd;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->is_shadow = true;
   tex->is_new_style_shadow = true;
   tex->coord_components = 2;
   tex->dest_type = nir_type_float32;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, nir_trim_vector(&b, input, 2));
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_comparator, nir_channel(&b, input, 2));
   if (kind == 1)
      tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_channel(&b, input, 3));
   if (kind == 2) {
      tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_ddx, nir_imm_vec2(&b, .125f, 0));
      tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_ddy, nir_imm_vec2(&b, 0, .25f));
   }
   nir_def_init(&tex->instr, &tex->def, 1, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_store_var(&b, output, nir_vec4(&b, &tex->def, &tex->def,
      &tex->def, nir_imm_float(&b, 1)), 15);
   nir_shader_gather_info(b.shader, b.impl);
   b.shader->info.num_textures = 1;
   return b.shader;
}

/* Shader-generated projected coordinates, gradients and offsets must execute
 * in USC, while size queries use the descriptor rather than a baked-in size. */
static nir_shader *native_vertex_texture_shader(unsigned kind)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
      pco_nir_options(), "native_vertex_texture");
   nir_variable *position = nir_variable_create(b.shader, nir_var_shader_in,
      glsl_vec4_type(), "position");
   position->data.location = VERT_ATTRIB_GENERIC0;
   nir_variable *output = nir_variable_create(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "gl_Position");
   output->data.location = VARYING_SLOT_POS;
   nir_variable *varying = nir_variable_create(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "sampled");
   varying->data.location = VARYING_SLOT_VAR0;
   nir_store_var(&b, output, nir_load_var(&b, position), 15);
   nir_def *values = nir_load_uniform(&b, 4, 32, nir_imm_int(&b, 0),
      .base=0, .range=1, .dest_type=nir_type_uint32);
   const bool query = kind >= 4 && kind <= 8;
   const unsigned query_components = kind == 5 || kind >= 7 ? 3 : 2;
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, query ? 1 : kind == 0 ? 2 : kind == 9 ? 4 : 3);
   tex->op = query ? nir_texop_txs : kind == 3 ? nir_texop_txf :
      kind == 2 || kind == 9 ? nir_texop_txd : nir_texop_txl;
   tex->sampler_dim = kind == 5 || kind == 9 ? GLSL_SAMPLER_DIM_3D :
      kind == 6 ? GLSL_SAMPLER_DIM_CUBE : GLSL_SAMPLER_DIM_2D;
   tex->is_array = kind == 7 || kind == 8;
   tex->is_shadow = kind == 8;
   tex->coord_components = query ? 0 : kind == 9 ? 3 : 2;
   tex->dest_type = query ? nir_type_int32 : nir_type_float32;
   if (query) {
      tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_channel(&b, values, 2));
   } else {
      tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, nir_trim_vector(&b, values, tex->coord_components));
      if (kind == 9) {
         tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_ddx, nir_imm_vec3(&b, 0, 0, 0));
         tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_ddy, nir_imm_vec3(&b, 0, 0, 0));
         tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_offset, nir_imm_ivec3(&b, 7, 3, -8));
      } else if (kind == 2) {
         tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_ddx,
            nir_vec2(&b, nir_imm_float(&b, 0.125f), nir_imm_float(&b, 0.0f)));
         tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_ddy,
            nir_vec2(&b, nir_imm_float(&b, 0.0f), nir_imm_float(&b, 0.25f)));
      } else {
         tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_channel(&b, values, 2));
         if (kind == 1)
            tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_projector, nir_channel(&b, values, 3));
         if (kind == 3)
            tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_offset, nir_imm_ivec2(&b, 1, -1));
      }
   }
   nir_def_init(&tex->instr, &tex->def, query ? query_components : 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_def *value = &tex->def;
   if (query) {
      value = nir_i2f32(&b, value);
      value = nir_vec4(&b, nir_channel(&b, value, 0), nir_channel(&b, value, 1),
                       query_components == 3 ? nir_channel(&b, value, 2) : nir_imm_float(&b, 0),
                       nir_imm_float(&b, 1));
   }
   nir_store_var(&b, varying, value, 15);
   nir_shader_gather_info(b.shader, b.impl);
   b.shader->info.num_textures = 1;
   return b.shader;
}

static nir_shader *native_vertex_texture_fragment(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      pco_nir_options(), "native_vertex_texture_fragment");
   nir_variable *input = nir_variable_create(b.shader, nir_var_shader_in,
      glsl_vec4_type(), "sampled");
   input->data.location = VARYING_SLOT_VAR0;
   nir_variable *output = nir_variable_create(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "color");
   output->data.location = FRAG_RESULT_DATA0;
   nir_store_var(&b, output, nir_load_var(&b, input), 15);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *native_derivative_fragment(unsigned kind)
{
   const unsigned components = kind < 24 ? kind / 6 + 1 : kind - 22;
   const unsigned operation = kind < 24 ? kind % 6 : 6;
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      pco_nir_options(), "native_derivative_fragment");
   nir_variable *output = nir_variable_create(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "color");
   output->data.location = FRAG_RESULT_DATA0;
   nir_def *input = nir_trim_vector(&b, nir_load_uniform(&b, 4, 32,
      nir_imm_int(&b, 0), .base=0, .range=1, .dest_type=nir_type_float32), components);
   static const nir_intrinsic_op operations[] = {
      nir_intrinsic_ddx, nir_intrinsic_ddy, nir_intrinsic_ddx_fine,
      nir_intrinsic_ddy_fine, nir_intrinsic_ddx_coarse, nir_intrinsic_ddy_coarse,
   };
   /* Build a genuine vector intrinsic, independently of the builder option
    * that scalarizes newly emitted operations, to reproduce linked GLSL NIR. */
   nir_intrinsic_instr *derivative = nir_intrinsic_instr_create(b.shader,
      operation == 6 ? nir_intrinsic_ddx : operations[operation]);
   derivative->num_components = components;
   derivative->src[0] = nir_src_for_ssa(input);
   nir_def_init(&derivative->instr, &derivative->def, components, 32);
   nir_builder_instr_insert(&b, &derivative->instr);
   nir_def *result = &derivative->def;
   if (operation == 6) {
      nir_intrinsic_instr *y = nir_intrinsic_instr_create(b.shader, nir_intrinsic_ddy);
      y->num_components = components;
      y->src[0] = nir_src_for_ssa(input);
      nir_def_init(&y->instr, &y->def, components, 32);
      nir_builder_instr_insert(&b, &y->instr);
      result = nir_fadd(&b, nir_fabs(&b, result), nir_fabs(&b, &y->def));
   }
   nir_def *values[4];
   for (unsigned c = 0; c < 4; ++c)
      values[c] = c < components ? nir_channel(&b, result, c) : nir_imm_float(&b, 0);
   nir_store_var(&b, output, nir_vec(&b, values, 4), 15);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *native_sample_fragment(unsigned kind)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      pco_nir_options(), "native_sample_fragment");
   nir_variable *output = nir_variable_create(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "color");
   output->data.location = FRAG_RESULT_DATA0;
   if (kind >= 3 && kind != 6) {
      nir_variable *mask = nir_variable_create(b.shader, nir_var_shader_out,
         glsl_array_type(glsl_int_type(), 1, 0), "gl_SampleMask");
      mask->data.location = FRAG_RESULT_SAMPLE_MASK;
      nir_deref_instr *element = nir_build_deref_array_imm(&b, nir_build_deref_var(&b, mask), 0);
      nir_store_deref(&b, element, nir_imm_int(&b, kind == 4 ? 0 : 0xaaaa), 1);
      if (kind == 4 || kind == 5)
         nir_store_deref(&b, element, nir_imm_int(&b, kind == 4 ? 0xffff : 0), 1);
      if (kind == 7) {
         nir_variable *depth = nir_variable_create(b.shader, nir_var_shader_out,
            glsl_float_type(), "gl_FragDepth");
         depth->data.location = FRAG_RESULT_DEPTH;
         nir_store_var(&b, depth, nir_imm_float(&b, 0.25f), 1);
      }
   }
   if (kind == 6)
      nir_terminate_if(&b, nir_ieq_imm(&b, nir_iand_imm(&b, nir_load_sample_id(&b), 1), 0));
   nir_def *value = kind == 0 ? nir_u2f32(&b, nir_load_sample_id(&b)) :
      kind == 1 ? nir_u2f32(&b, nir_load_sample_mask_in(&b)) :
      kind == 2 ? nir_load_sample_pos(&b) : nir_imm_float(&b, 0);
   nir_store_var(&b, output, nir_vec4(&b, nir_channel(&b, value, 0),
      kind == 2 ? nir_channel(&b, value, 1) : nir_imm_float(&b, 0),
      nir_imm_float(&b, 0), nir_imm_float(&b, 1)), 15);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *explicit_texture_shader(unsigned kind)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      pco_nir_options(), "generic_explicit_texture_fs");
   nir_variable *output = nir_variable_create(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "color");
   output->data.location = FRAG_RESULT_DATA0;
   nir_def *input = nir_load_uniform(&b, 4, 32, nir_imm_int(&b, 0),
      .base=0, .range=1, .dest_type=nir_type_uint32);
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);
   tex->op = kind == 3 ? nir_texop_txl : nir_texop_txf;
   tex->sampler_dim = kind == 2 || kind == 5 ? GLSL_SAMPLER_DIM_3D :
      kind == 3 || kind == 4 ? GLSL_SAMPLER_DIM_CUBE : GLSL_SAMPLER_DIM_2D;
   tex->is_array = kind == 1 || kind == 5;
   tex->coord_components = kind == 0 || kind == 6 ? 2 : 3;
   tex->texture_index = kind == 6 ? 1 : 0;
   tex->sampler_index = tex->texture_index;
   tex->dest_type = nir_type_float32;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
      nir_trim_vector(&b, input, tex->coord_components));
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_channel(&b, input, 3));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_store_var(&b, output, &tex->def, 15);
   nir_shader_gather_info(b.shader, b.impl);
   b.shader->info.num_textures = 1;
   return b.shader;
}

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

static nir_shader *fragment_shader(const unsigned *units, unsigned count,
                                   unsigned texture_count)
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
         nir_imm_int(&b, (units[i] % 2) * 2), .base = 0, .range = 6,
         .dest_type = nir_type_float32);
      nir_def *bias = nir_load_uniform(&b, 4, 32,
         nir_imm_int(&b, (units[i] % 2) * 2 + 1), .base = 0, .range = 6,
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
   b.shader->info.num_textures = texture_count;
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
   for (unsigned stage = 0; stage < 2; ++stage) {
      const bool vertex = stage == 0;
      const unsigned words = vertex ? 96 : 100;
      const unsigned range = words / 4 - 1;
      for (unsigned branch = 0; branch <= 1; ++branch) {
         nir_shader *vs = vertex ? native_dynamic_uniform_shader(true, 1, range, branch) : vertex_shader();
         nir_shader *fs = vertex ? native_vertex_texture_fragment() :
            native_dynamic_uniform_shader(false, 1, range, branch);
         struct pvrgpu_pco_graphics_binary binary;
         if (!pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format,
               false, false, 1, vertex ? words : 0, vertex ? 0 : words,
               1, vertex ? 0 : 1, &binary, error, sizeof(error))) {
            fprintf(stderr, "dynamic uniform stage=%u branch=%u: %s\n", stage, branch, error);
            abort();
         }
         const struct pvrgpu_pco_owned_binary *shader = vertex ? &binary.vertex : &binary.fragment;
         check_generic_dynamic_uniform_program(shader->data, shader->size,
            vertex, 1, range, shader->abi.push_constant_start, shader->abi.shareds,
            binary.varying_output_start, branch);
         pvrgpu_pco_graphics_binary_finish(&binary);
         ralloc_free(vs); ralloc_free(fs);
      }
   }
   for (unsigned bad = 0; bad < 3; ++bad) {
      nir_shader *vs = vertex_shader();
      nir_shader *fs = native_dynamic_uniform_shader(false,
         bad == 2 ? UINT32_MAX : 1, bad == 0 ? 0 : bad == 1 ? 25 : 1, false);
      struct pvrgpu_pco_graphics_binary binary;
      if (pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format,
            false, false, 1, 0, 100, 1, 1, &binary, error, sizeof(error)) ||
          binary.vertex.data || binary.fragment.data) {
         fprintf(stderr, "invalid dynamic uniform range %u did not fail closed\n", bad);
         abort();
      }
      ralloc_free(vs); ralloc_free(fs);
   }
   for (unsigned kind = 0; kind < 8; ++kind) {
      nir_shader *vs = native_interpolation_shader(kind, true);
      nir_shader *fs = native_interpolation_shader(kind, false);
      struct pvrgpu_pco_graphics_binary binary;
      if (!pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format,
            false, true, 1, 0, 4, 1, 0, &binary, error, sizeof(error))) {
         fprintf(stderr, "interpolation kind %u: %s\n", kind, error);
         abort();
      }
      dump_generic_sample_program(binary.fragment.data, binary.fragment.size, 100 + kind);
      check_generic_interpolation_program(binary.fragment.data, binary.fragment.size, kind);
      pvrgpu_pco_graphics_binary_finish(&binary);
      ralloc_free(vs); ralloc_free(fs);
   }
   for (unsigned shadow_test = 0; shadow_test < 6; ++shadow_test) {
      const bool vertex = shadow_test >= 3;
      const unsigned kind = shadow_test % 3;
      nir_shader *vs = vertex ? native_shadow_shader(kind, true) : vertex_shader();
      nir_shader *fs = vertex ? native_vertex_texture_fragment() : native_shadow_shader(kind, false);
      struct pvrgpu_pco_graphics_binary binary;
      if (!pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format,
            false, false, 1, vertex ? 4 : 0, vertex ? 0 : 4,
            1, vertex ? 0 : 1, &binary, error, sizeof(error))) {
         fprintf(stderr, "shadow kind %u: %s\n", kind, error);
         abort();
      }
      uint32_t descriptor[20] = {0};
      if (!pvrgpu_pco_build_terrain_texture_descriptor(descriptor, PIPE_FORMAT_Z24_UNORM_S8_UINT,
             8, 4, 1, 128, 0, 0, 0, 2, 2, 128, 1, 0))
         abort();
      check_generic_shadow_program(vertex ? binary.vertex.data : binary.fragment.data,
         vertex ? binary.vertex.size : binary.fragment.size, descriptor, kind,
         vertex, binary.varying_output_start);
      pvrgpu_pco_graphics_binary_finish(&binary);
      ralloc_free(vs); ralloc_free(fs);
   }
   for (unsigned test = 0; test < 3; ++test) {
      nir_shader *vs = vertex_shader();
      nir_shader *fs = fragment_shader(sequences[test], counts[test], 2);
      struct pvrgpu_pco_graphics_binary binary;
      if (!pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format,
            false, false, 1, 0, 24, 1, 2, &binary, error, sizeof(error))) {
         fprintf(stderr, "%s\n", error);
         abort();
      }
      check_generic_texture_program(binary.fragment.data, binary.fragment.size,
                                    sequences[test], counts[test]);
      pvrgpu_pco_graphics_binary_finish(&binary);
      ralloc_free(vs);
      ralloc_free(fs);
   }
   /* Decode actual native SMP operands above PCO's four-set boundary. The
    * compiler must read image SH[20*unit] and sampler SH[20*unit+8], even for
    * descending/repeated accesses in the packed last descriptor set. */
   for (unsigned textures = 4; textures <= 8; ++textures) {
      unsigned units[9];
      for (unsigned i = 0; i < textures; ++i)
         units[i] = textures - 1 - i;
      units[textures] = textures - 1;
      nir_shader *vs = vertex_shader();
      nir_shader *fs = fragment_shader(units, textures + 1, textures);
      struct pvrgpu_pco_graphics_binary binary;
      if (!pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format,
            false, false, 1, 0, 24, 1, textures,
            &binary, error, sizeof(error))) {
         fprintf(stderr, "packed texture descriptors (%u units): %s\n", textures, error);
         abort();
      }
      check_generic_texture_program(binary.fragment.data, binary.fragment.size,
                                    units, textures + 1);
      pvrgpu_pco_graphics_binary_finish(&binary);
      ralloc_free(vs);
      ralloc_free(fs);
   }
   for (unsigned kind = 0; kind < 7; ++kind) {
      nir_shader *vs = vertex_shader(), *fs = explicit_texture_shader(kind);
      struct pvrgpu_pco_graphics_binary binary;
      const bool accepted = pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format,
         false, false, 1, 0, 4, 1, 1, &binary, error, sizeof(error));
      if (accepted != (kind < 4)) {
         fprintf(stderr, "explicit texture kind %u acceptance mismatch: %s\n", kind, error);
         abort();
      }
      if (accepted) {
         check_generic_explicit_texture_program(binary.fragment.data, binary.fragment.size, kind);
         pvrgpu_pco_graphics_binary_finish(&binary);
      }
      ralloc_free(vs); ralloc_free(fs);
   }
   for (unsigned kind = 0; kind < 10; ++kind) {
      nir_shader *vs = native_vertex_texture_shader(kind);
      nir_shader *fs = native_vertex_texture_fragment();
      struct pvrgpu_pco_graphics_binary binary;
      if (!pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format,
            false, false, 1, 4, 0, 1, 0, &binary, error, sizeof(error))) {
         fprintf(stderr, "vertex texture kind %u: %s\n", kind, error);
         abort();
      }
      if (binary.vertex.abi.shareds != 24 ||
          binary.vertex.abi.push_constant_start != 20 ||
          binary.vertex.abi.push_constant_count != 4 ||
          binary.fragment.abi.shareds != 0) {
         fprintf(stderr, "vertex texture descriptor/CB0 ABI changed\n");
         abort();
      }
      uint32_t shared[24] = {0};
      if (!pvrgpu_pco_build_terrain_texture_descriptor(shared, PIPE_FORMAT_R8G8B8A8_UNORM,
             8, kind == 6 ? 8 : 4, 3, 1024, 0, 0, 0, 2, 2, 128,
             kind == 5 || kind >= 7 ? 4 : 1, 2))
         abort();
      if (kind == 7 || kind == 8) {
         shared[0] = (shared[0] & ~7U) | 1U; /* 2D array, not 3D. */
         shared[2] = 3U | (3U << 4) | (1U << 15);
         shared[3] = 0;
         shared[4] = 8 * 4 * 4;
      }
      shared[20] = kind == 3 ? 2 : 0x3e800000U; /* 0.25 */
      shared[21] = kind == 3 ? 3 : 0x3f400000U; /* 0.75 */
      shared[22] = kind >= 4 && kind <= 8 ? 1 : 0;
      shared[23] = 0x40000000U; /* projector 2 */
      check_generic_vertex_texture_program(binary.vertex.data, binary.vertex.size,
         shared, 24, binary.varying_output_start, kind);
      pvrgpu_pco_graphics_binary_finish(&binary);
      ralloc_free(vs); ralloc_free(fs);
   }
   for (unsigned kind = 0; kind < 27; ++kind) {
      nir_shader *vs = vertex_shader(), *fs = native_derivative_fragment(kind);
      struct pvrgpu_pco_graphics_binary binary;
      if (!pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format,
            false, false, 1, 0, 4, 1, 0, &binary, error, sizeof(error))) {
         fprintf(stderr, "derivative kind %u: %s\n", kind, error);
         abort();
      }
      check_generic_derivative_program(binary.fragment.data, binary.fragment.size, kind);
      pvrgpu_pco_graphics_binary_finish(&binary);
      ralloc_free(vs); ralloc_free(fs);
   }
   const unsigned sample_kinds[] = {0, 2, 1, 3, 4, 5, 6, 7};
   struct pvrgpu_pco_graphics_binary sample_binaries[16];
   for (unsigned sample_test = 0; sample_test < 16; ++sample_test) {
      const unsigned kind = sample_kinds[sample_test % 8];
      nir_shader *vs = vertex_shader(), *fs = native_sample_fragment(kind);
      struct pvrgpu_pco_graphics_binary *binary = &sample_binaries[sample_test];
      if (!pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format,
            false, sample_test < 8, 1, 0, 0, 1, 0, binary, error, sizeof(error))) {
         fprintf(stderr, "sample input kind %u: %s\n", kind, error);
         abort();
      }
      dump_generic_sample_program(binary->fragment.data, binary->fragment.size, kind + (sample_test / 8) * 8);
      ralloc_free(vs); ralloc_free(fs);
   }
   for (unsigned sample_test = 0; sample_test < 16; ++sample_test) {
      struct pvrgpu_pco_graphics_binary *binary = &sample_binaries[sample_test];
      check_generic_sample_program(binary->fragment.data, binary->fragment.size,
         sample_kinds[sample_test % 8], sample_test < 8);
      pvrgpu_pco_graphics_binary_finish(binary);
   }
   pvrgpu_pco_compiler_destroy(compiler);
   glsl_type_singleton_decref();
}
