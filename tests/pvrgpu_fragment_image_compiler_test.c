/* SPDX-License-Identifier: MIT */
#include "pvrgpu_pco.h"
#include "nir/nir.h"
#include "nir/nir_builder.h"
#include "pco/pco.h"
#include "util/ralloc.h"
#include <stdio.h>
#include <stdlib.h>

void check_native_fragment_image(const uint8_t *bytes, size_t size,
                                 unsigned shared_count, unsigned cb0_start,
                                 unsigned image_count, unsigned kind);
void test_native_fragment_images(void);

static nir_shader *image_vertex(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
      pco_nir_options(), "fragment_image_vertex");
   nir_variable *input = nir_variable_create(b.shader, nir_var_shader_in,
      glsl_vec4_type(), "position");
   input->data.location = VERT_ATTRIB_GENERIC0;
   nir_variable *output = nir_variable_create(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "gl_Position");
   output->data.location = VARYING_SLOT_POS;
   nir_store_var(&b, output, nir_load_var(&b, input), 15);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static nir_shader *image_fragment(unsigned kind)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      pco_nir_options(), "fragment_image_atomic");
   nir_variable *output = nir_variable_create(b.shader, nir_var_shader_out,
      glsl_vec4_type(), "color");
   output->data.location = FRAG_RESULT_DATA0;
   nir_def *params = nir_load_uniform(&b, 4, 32, nir_imm_int(&b, 0),
      .range=1, .dest_type=nir_type_uint32);
   nir_def *coords = nir_vec4(&b, nir_channel(&b, params, 0),
      nir_channel(&b, params, 1), nir_imm_int(&b, 0), nir_imm_int(&b, 0));
   nir_def *old = nir_image_atomic(&b, 32, nir_imm_int(&b, 0), coords,
      nir_undef(&b, 1, 32), nir_channel(&b, params, 2),
      .image_dim=GLSL_SAMPLER_DIM_2D, .format=PIPE_FORMAT_R32_UINT,
      .atomic_op=nir_atomic_op_iadd);
   if (kind == 3)
      old = nir_image_atomic(&b, 32, nir_imm_int(&b, 1), coords,
         nir_undef(&b, 1, 32), nir_channel(&b, params, 3),
         .image_dim=GLSL_SAMPLER_DIM_2D, .format=PIPE_FORMAT_R32_UINT,
         .atomic_op=nir_atomic_op_iadd);
   nir_def *red = kind == 0 ? nir_imm_float(&b, 1) : nir_u2f32(&b, old);
   if (kind == 4) {
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);
      tex->op = nir_texop_txl;
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
      tex->coord_components = 2;
      tex->dest_type = nir_type_float32;
      tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
         nir_vec2(&b, red, nir_imm_float(&b, 0.5f)));
      tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_float(&b, 0));
      nir_def_init(&tex->instr, &tex->def, 4, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      red = nir_fadd(&b, red, nir_channel(&b, &tex->def, 0));
   }
   nir_store_var(&b, output, nir_vec4(&b, red, nir_imm_float(&b, 0),
      nir_imm_float(&b, 0), nir_imm_float(&b, 1)), 15);
   if (kind == 2) {
      nir_variable *image = nir_variable_create(b.shader, nir_var_image,
         glsl_image_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_UINT), "bound_image");
      image->data.driver_location = 0;
      image->data.binding = 7; // Gallium slot, not the GLSL binding number.
      image->data.image.format = PIPE_FORMAT_R32_UINT;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic) continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_image_atomic) continue;
            intr->intrinsic = nir_intrinsic_image_deref_atomic;
            b.cursor = nir_before_instr(instr);
            nir_src_rewrite(&intr->src[0], &nir_build_deref_var(&b, image)->def);
            nir_intrinsic_set_format(intr, PIPE_FORMAT_NONE);
         }
      }
   }
   nir_shader_gather_info(b.shader, b.impl);
   b.shader->info.num_images = kind == 3 ? 2 : 1;
   b.shader->info.num_textures = kind == 4 ? 1 : 0;
   b.shader->info.fs.early_fragment_tests = kind == 2;
   return b.shader;
}

void test_native_fragment_images(void)
{
   glsl_type_singleton_init_or_ref();
   char error[512] = {0};
   struct pvrgpu_pco_compiler *compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
   if (!compiler) { fprintf(stderr, "%s\n", error); abort(); }
   const enum pipe_format format = PIPE_FORMAT_R32G32B32A32_FLOAT;
   for (unsigned kind = 0; kind < 5; ++kind) {
      nir_shader *vs = image_vertex(), *fs = image_fragment(kind);
      struct pvrgpu_pco_graphics_binary binary;
      if (!pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format,
            false, false, 1, 0, 4, 1, kind == 4 ? 1 : 0, &binary, error, sizeof(error))) {
         fprintf(stderr, "fragment image %u: %s\n", kind, error); abort();
      }
      const unsigned count = kind == 3 ? 2 : 1;
      if (binary.fragment_image_descriptor_start != (kind == 4 ? 20 : 0) ||
          binary.fragment_image_descriptor_count != count ||
          binary.fragment_image_read_mask != (1u << count) - 1 ||
          binary.fragment_image_write_mask != (1u << count) - 1 ||
          binary.fragment_early_tests != (kind == 2)) abort();
      const char *directory = getenv("MESA_TEST_OUT");
      if (directory) {
         const struct pvrgpu_pco_owned_binary *stages[] = {&binary.vertex, &binary.fragment};
         for (unsigned stage = 0; stage < 2; ++stage) {
            char path[4096];
            snprintf(path, sizeof(path), "%s/image-%u-%s.bin", directory, kind, stage ? "fs" : "vs");
            FILE *file = fopen(path, "wb");
            if (!file || fwrite(stages[stage]->data, 1, stages[stage]->size, file) != stages[stage]->size) abort();
            fclose(file);
            snprintf(path, sizeof(path), "%s/image-%u-%s.abi", directory, kind, stage ? "fs" : "vs");
            file = fopen(path, "w");
            const struct pvrgpu_pco_stage_abi *a = &stages[stage]->abi;
            if (!file || fprintf(file, "%u %u %u %u %u %u %u %u %u %u\n",
               a->temps, a->vertex_inputs, a->vertex_outputs, a->coefficients,
               a->shareds, a->push_constant_start, a->push_constant_count,
               a->entry_offset, a->uniform_buffer_descriptor_start,
               a->uniform_buffer_descriptor_count) < 0) abort();
            fclose(file);
         }
      }
      check_native_fragment_image(binary.fragment.data, binary.fragment.size,
         binary.fragment.abi.shareds, binary.fragment.abi.push_constant_start,
         count, kind);
      pvrgpu_pco_graphics_binary_finish(&binary);
      ralloc_free(vs); ralloc_free(fs);
   }
   for (unsigned invalid = 0; invalid < 6; ++invalid) {
      nir_shader *vs = image_vertex(), *fs = image_fragment(1);
      nir_function_impl *impl = nir_shader_get_entrypoint(fs);
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic) continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_image_atomic) continue;
            if (invalid == 0) nir_intrinsic_set_format(intr, PIPE_FORMAT_R32_FLOAT);
            if (invalid == 1) nir_intrinsic_set_image_dim(intr, GLSL_SAMPLER_DIM_3D);
            if (invalid == 2) nir_intrinsic_set_atomic_op(intr, nir_atomic_op_ixor);
            if (invalid == 3) nir_src_rewrite(&intr->src[0], intr->src[3].ssa);
         }
      }
      if (invalid == 4) fs->info.fs.uses_discard = true;
      if (invalid == 5) fs->info.num_images = 33;
      struct pvrgpu_pco_graphics_binary binary;
      if (pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format,
            false, false, 1, 0, 4, 1, 0, &binary, error, sizeof(error)) || !error[0]) {
         fprintf(stderr, "invalid fragment image accepted: %u\n", invalid); abort();
      }
      ralloc_free(vs); ralloc_free(fs);
   }
   pvrgpu_pco_compiler_destroy(compiler);
   glsl_type_singleton_decref();
}
