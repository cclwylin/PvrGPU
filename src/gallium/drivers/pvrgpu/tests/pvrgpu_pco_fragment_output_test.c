/* SPDX-License-Identifier: MIT */
/* Real generic-driver compiler matrix, including absent color exports.
 * Optional argv[1] writes actual PCO binaries for independent ISS checking.
 */
#include "pvrgpu_pco.h"
#include "nir/nir.h"
#include "nir/nir_builder.h"
#include "pco/pco.h"
#include "util/ralloc.h"
#include <stdio.h>
#include <stdlib.h>

static nir_variable *output(nir_builder *b, unsigned location,
                            unsigned components, const char *name)
{
   nir_variable *var = nir_variable_create(b->shader, nir_var_shader_out,
      glsl_vector_type(GLSL_TYPE_FLOAT, components), name);
   var->data.location = location;
   return var;
}

static nir_shader *vertex(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
      pco_nir_options(), "generic_output_matrix_vs");
   nir_variable *in = nir_variable_create(b.shader, nir_var_shader_in,
      glsl_vec4_type(), "position");
   in->data.location = VERT_ATTRIB_GENERIC0;
   nir_store_var(&b, output(&b, VARYING_SLOT_POS, 4, "gl_Position"),
      nir_load_var(&b, in), 15);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static void color(nir_builder *b, unsigned location, unsigned components)
{
   nir_store_var(b, output(b, location, components, "color"),
      nir_trim_vector(b, nir_imm_vec4(b, 0.125f, 0.25f, 0.5f, 0.75f), components),
      BITFIELD_MASK(components));
}

static nir_shader *fragment(unsigned kind)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      pco_nir_options(), "generic_output_matrix_fs");
   if (kind >= 2 && kind <= 5)
      color(&b, FRAG_RESULT_DATA0, kind - 1);
   if (kind == 6) {
      color(&b, FRAG_RESULT_DATA0, 2);
      color(&b, FRAG_RESULT_DATA2, 3);
   }
   if (kind == 7) color(&b, FRAG_RESULT_COLOR, 4);
   if (kind == 8) color(&b, FRAG_RESULT_DATA0, 3);
   if (kind == 1 || kind == 8)
      nir_store_var(&b, output(&b, FRAG_RESULT_DEPTH, 1, "gl_FragDepth"),
         nir_imm_float(&b, 0.25f), 1);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

int main(int argc, char **argv)
{
   if (argc > 2) return 2;
   char error[512] = {0};
   struct pvrgpu_pco_compiler *compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
   if (!compiler) { fprintf(stderr, "%s\n", error); return 1; }
   const enum pipe_format format = PIPE_FORMAT_R32G32B32A32_FLOAT;
   const unsigned expected[9][4] = {{0}, {0}, {1}, {3}, {7}, {15}, {3,0,7,0}, {15}, {7}};
   unsigned checks = 0, failed = 0;
   for (unsigned kind = 0; kind < 9; ++kind) {
      nir_shader *vs = vertex(), *fs = fragment(kind);
      struct pvrgpu_pco_graphics_binary binary = {0};
      const bool compiled = pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format,
            false, false, kind == 6 ? 4 : 1, 0, 0, 1, 0,
            &binary, error, sizeof(error));
      if (kind == 0) {
         ++checks;
         if (compiled) { fprintf(stderr, "empty FS unexpectedly admitted\n"); return 1; }
         printf("kind=0 empty-FS refusal retained: %s\n", error);
         ralloc_free(vs); ralloc_free(fs);
         continue;
      }
      if (!compiled) {
         fprintf(stderr, "kind=%u compile failure: %s\n", kind, error);
         return 1;
      }
      printf("kind=%u masks=%u,%u,%u,%u fs_bytes=%zu temps=%u shareds=%u\n", kind,
         binary.fragment_output_mask[0], binary.fragment_output_mask[1],
         binary.fragment_output_mask[2], binary.fragment_output_mask[3],
         binary.fragment.size, binary.fragment.abi.temps, binary.fragment.abi.shareds);
      for (unsigned target = 0; target < 8; ++target) {
         ++checks;
         if (binary.fragment_output_mask[target] != (target < 4 ? expected[kind][target] : 0)) {
            fprintf(stderr, "kind=%u target=%u declared=%u expected=%u\n", kind, target,
               binary.fragment_output_mask[target], target < 4 ? expected[kind][target] : 0);
            ++failed;
         }
      }
      if (argc == 2) {
         char path[4096];
         if (snprintf(path, sizeof(path), "%s/output-%u.bin", argv[1], kind) >= (int)sizeof(path)) return 2;
         FILE *file = fopen(path, "wb");
         if (!file || fwrite(binary.fragment.data, 1, binary.fragment.size, file) != binary.fragment.size ||
             fclose(file)) return 1;
      }
      pvrgpu_pco_graphics_binary_finish(&binary);
      ralloc_free(vs); ralloc_free(fs);
   }
   pvrgpu_pco_compiler_destroy(compiler);
   fprintf(stderr, "Generic fragment outputs: checks=%u failures=%u\n", checks, failed);
   return failed ? 1 : 0;
}
