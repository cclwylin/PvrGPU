/* SPDX-License-Identifier: MIT */
/* Actual generic driver compiler ABI regression: no shader identities or
 * precomputed shader results. Verify empty/nonempty CB0 after UBO prefixes. */
#include "pvrgpu_pco.h"
#include "nir/nir.h"
#include "nir/nir_builder.h"
#include "pco/pco.h"
#include "util/ralloc.h"
#include <stdio.h>
#include <stdlib.h>

static void check(bool condition, const char *message)
{
   if (!condition) {
      fprintf(stderr, "%s\n", message);
      abort();
   }
}

static nir_shader *shader(bool vertex, unsigned blocks, bool push)
{
   nir_builder b = nir_builder_init_simple_shader(
      vertex ? MESA_SHADER_VERTEX : MESA_SHADER_FRAGMENT,
      pco_nir_options(), "generic_ubo_abi");
   nir_variable *input = nir_variable_create(b.shader, nir_var_shader_in,
                                              glsl_vec4_type(), "input");
   input->data.location = vertex ? VERT_ATTRIB_GENERIC0 : VARYING_SLOT_VAR0;
   nir_variable *output = nir_variable_create(b.shader, nir_var_shader_out,
                                               glsl_vec4_type(), "output");
   output->data.location = vertex ? VARYING_SLOT_VAR0 : FRAG_RESULT_DATA0;
   nir_def *value = nir_load_var(&b, input);
   if (vertex) {
      nir_variable *position = nir_variable_create(b.shader, nir_var_shader_out,
                                                    glsl_vec4_type(), "position");
      position->data.location = VARYING_SLOT_POS;
      nir_store_var(&b, position, value, 15);
   }
   if (blocks) {
      nir_def *buffer = nir_load_ubo(&b, 4, 32, nir_imm_int(&b, blocks - 1),
                                      nir_imm_int(&b, 16), .align_mul = 16,
                                      .align_offset = 0, .range_base = 16,
                                      .range = 16);
      value = nir_fadd(&b, value, buffer);
   }
   if (push) {
      nir_def *uniform = nir_load_uniform(&b, 4, 32, nir_imm_int(&b, 0),
                                            .base = 0, .range = 1,
                                            .dest_type = nir_type_float32);
      value = nir_fadd(&b, value, uniform);
   }
   nir_store_var(&b, output, value, 15);
   nir_shader_gather_info(b.shader, b.impl);
   b.shader->info.num_ubos = blocks;
   return b.shader;
}

int main(void)
{
   glsl_type_singleton_init_or_ref();
   char error[512] = {0};
   struct pvrgpu_pco_compiler *compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
   check(compiler != NULL, error);
   const enum pipe_format format = PIPE_FORMAT_R32G32B32A32_FLOAT;
   const unsigned block_counts[] = {1, 3, 15};
   for (unsigned kind = 1; kind <= 3; ++kind) {
      for (unsigned push = 0; push <= 1; ++push) {
         for (unsigned count = 0; count < 3; ++count) {
            unsigned vs_blocks = kind & 1 ? block_counts[count] : 0;
            unsigned fs_blocks = kind & 2 ? block_counts[count] : 0;
            nir_shader *vs = shader(true, vs_blocks, push);
            nir_shader *fs = shader(false, fs_blocks, push);
            struct pvrgpu_pco_graphics_binary binary = {0};
            check(pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format,
                      false, 1, push ? 4 : 0, push ? 4 : 0, 1, 0,
                      &binary, error, sizeof(error)), error);
            const struct pvrgpu_pco_stage_abi *abis[] = {
               &binary.vertex.abi, &binary.fragment.abi};
            const unsigned blocks[] = {vs_blocks, fs_blocks};
            for (unsigned stage = 0; stage < 2; ++stage) {
               const struct pvrgpu_pco_stage_abi *abi = abis[stage];
               check(abi->uniform_buffer_descriptor_start == 0 &&
                        abi->uniform_buffer_descriptor_count == blocks[stage],
                     "compiler UBO descriptor count/start differs from NIR blocks");
               check(abi->push_constant_start == blocks[stage] * 4 &&
                        abi->push_constant_count == (push ? 4u : 0u) &&
                        abi->shareds == blocks[stage] * 4 + (push ? 4u : 0u),
                     "compiler empty/nonempty push suffix overlaps its UBO descriptors");
            }
            pvrgpu_pco_graphics_binary_finish(&binary);
            ralloc_free(vs);
            ralloc_free(fs);
         }
      }
   }
   pvrgpu_pco_compiler_destroy(compiler);
   glsl_type_singleton_decref();
   puts("generic UBO compiler ABI tests: PASS (18 programs)");
   return 0;
}
