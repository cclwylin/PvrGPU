/*
 * Development-time generator for the exact GLBench varyings_shader_1,
 * varyings_shader_2, varyings_shader_4 and varyings_shader_8 PCO vertex and
 * fragment fixtures.
 *
 * This utility builds real NIR and invokes Mesa's public PowerVR PCO backend;
 * it does not define a project-local instruction encoding.  The GLES vec4
 * input is backed by PIPE_FORMAT_R32G32_FLOAT, so PCO supplies z=0 and w=1.
 * Case 1 exports c as both gl_Position and one smooth varying. Case 2 follows
 * the upstream GLBench source exactly: it exports gl_Position=c,
 * v1=v2=c/2 and writes v1+v2. Case 4 exports
 * v1=v2=v3=v4=c/4 and writes the upstream left-associated
 * ((v1+v2)+v3)+v4 expression. Case 8 likewise exports eight c/8 values and
 * preserves the source's seven left-associated additions. All varyings are
 * perspective-interpolated vec4 values and no semantic byte stream is
 * constructed by this generator; every byte comes from the pinned public
 * Mesa PCO compiler.
 *
 * Generated files are immutable fixtures.  PvrGPU's runtime has no Mesa
 * compiler dependency.
 *
 * Provenance:
 *   GLBench commit e99bc684272bffd68b06c998e272531c9c84330f
 *   Mesa 26.2.1 commit da14d65e4499e66468094be52bff9ea0915a695e
 *   public target gx6250
 * Expected output:
 *   VS 72 bytes, SHA-256
 *   09636842506c3a05b4dfae96d232274bb2eeb59876591e9fe29fc27a2e0860df
 *   FS 48 bytes, SHA-256
 *   a9c070ea3feb5dc4f7666b1fc019aaa9e3c522f5a8a73605ea07481104efc71c
 *   varyings_shader_2 VS 120 bytes, SHA-256
 *   11a9256581cec718761818f8907337c86e458d2e44884ffe89a8d20c44647535
 *   varyings_shader_2 FS 104 bytes, SHA-256
 *   8c3c5427a0064009d8799a120f3e34645031f8c73b15a30ca224f0b007e21e99
 *   varyings_shader_4 VS 136 bytes, SHA-256
 *   d98cefa0385a774d1a7b0ddb0149cc6b5aca3023cccd287e3eeea1ca410f6538
 *   varyings_shader_4 FS 216 bytes, SHA-256
 *   f5c1fbac1b9281ce5093ba9c629c90ff5cd81e1807351f3bee2f1f5700f1a08a
 *   varyings_shader_8 VS 176 bytes, SHA-256
 *   f5314dcc5a24dca2c7d716b9d0c3bd1696df0038e826b34ce7f7e208945bb45a
 *   varyings_shader_8 FS 440 bytes, SHA-256
 *   aaebb7b4e027f846eecda4687dbc14fb10dc8b9bb3881ef0134cd0255449c385
 */

#include "common/pvr_device_info.h"
#include "nir/nir.h"
#include "nir/nir_builder.h"
#include "nir/nir_builder_opcodes.h"
#include "pco/pco.h"
#include "pco/pco_data.h"
#include "util/ralloc.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static const char *const varying_names[] = {
   "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
};

static nir_shader *build_vertex_shader(unsigned varying_count)
{
   const char *shader_name = varying_count == 1
                                ? "glbench_varyings_1_vs"
                                : varying_count == 2
                                     ? "glbench_varyings_2_vs"
                                     : varying_count == 4
                                          ? "glbench_varyings_4_vs"
                                          : "glbench_varyings_8_vs";
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
                                                   pco_nir_options(),
                                                   "%s",
                                                   shader_name);
   nir_variable *input = nir_variable_create(b.shader,
                                             nir_var_shader_in,
                                             glsl_vec4_type(),
                                             "c");
   input->data.location = VERT_ATTRIB_GENERIC0;

   nir_variable *position = nir_variable_create(b.shader,
                                                 nir_var_shader_out,
                                                 glsl_vec4_type(),
                                                 "gl_Position");
   position->data.location = VARYING_SLOT_POS;

   for (unsigned index = 0; index < varying_count; ++index) {
      nir_variable *varying = nir_variable_create(b.shader,
                                                   nir_var_shader_out,
                                                   glsl_vec4_type(),
                                                   varying_names[index]);
      varying->data.location = VARYING_SLOT_VAR0 + index;
      varying->data.interpolation = INTERP_MODE_NONE;
   }

   nir_def *value = nir_load_input(
      &b,
      4,
      32,
      nir_imm_int(&b, 0),
      .base = 0,
      .range = 1,
      .component = 0,
      .dest_type = nir_type_float32,
      .io_semantics = (nir_io_semantics){
         .location = VERT_ATTRIB_GENERIC0,
         .num_slots = 1,
      });

   nir_store_output(&b,
                    value,
                    nir_imm_int(&b, 0),
                    .base = 0,
                    .range = 1,
                    .write_mask = 0xf,
                    .component = 0,
                    .src_type = nir_type_float32,
                    .io_semantics = (nir_io_semantics){
                       .location = VARYING_SLOT_POS,
                       .num_slots = 1,
                    });
   nir_def *varying_value = varying_count == 1
                               ? value
                               : nir_fmul_imm(&b,
                                              value,
                                              1.0f / (float)varying_count);
   for (unsigned index = 0; index < varying_count; ++index) {
      nir_store_output(&b,
                       varying_value,
                       nir_imm_int(&b, 0),
                       .base = 0,
                       .range = 1,
                       .write_mask = 0xf,
                       .component = 0,
                       .src_type = nir_type_float32,
                       .io_semantics = (nir_io_semantics){
                          .location = VARYING_SLOT_VAR0 + index,
                          .num_slots = 1,
                       });
   }
   nir_jump(&b, nir_jump_return);
   return b.shader;
}

static nir_shader *build_fragment_shader(unsigned varying_count)
{
   const char *shader_name = varying_count == 1
                                ? "glbench_varyings_1_fs"
                                : varying_count == 2
                                     ? "glbench_varyings_2_fs"
                                     : varying_count == 4
                                          ? "glbench_varyings_4_fs"
                                          : "glbench_varyings_8_fs";
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                   pco_nir_options(),
                                                   "%s",
                                                   shader_name);
   for (unsigned index = 0; index < varying_count; ++index) {
      nir_variable *varying = nir_variable_create(b.shader,
                                                   nir_var_shader_in,
                                                   glsl_vec4_type(),
                                                   varying_names[index]);
      varying->data.location = VARYING_SLOT_VAR0 + index;
      varying->data.interpolation = INTERP_MODE_SMOOTH;
   }

   nir_def *barycentric = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *value = NULL;
   for (unsigned index = 0; index < varying_count; ++index) {
      nir_def *next = nir_load_interpolated_input(
         &b,
         4,
         32,
         barycentric,
         nir_imm_int(&b, 0),
         .base = 0,
         .component = 0,
         .dest_type = nir_type_float32,
         .io_semantics = (nir_io_semantics){
            .location = VARYING_SLOT_VAR0 + index,
            .num_slots = 1,
         });
      value = value ? nir_fadd(&b, value, next) : next;
   }
   for (unsigned component = 0; component < 4; ++component) {
      nir_frag_store_pco(&b,
                         nir_channel(&b, value, component),
                         .base = component);
   }
   nir_jump(&b, nir_jump_return);
   return b.shader;
}

static pco_shader *compile_shader(pco_ctx *ctx,
                                  nir_shader *nir,
                                  pco_data *data)
{
   /* The explicit intrinsic locations already express the linked VS/FS
    * contract.  This is the same independent stage route as the project's
    * pinned attribute generator; pco_link_nir expects driver-lowered vars. */
   pco_preprocess_nir(ctx, nir);
   pco_lower_nir(ctx, nir, data);
   pco_postprocess_nir(ctx, nir, data);

   pco_shader *shader = pco_trans_nir(ctx, nir, data, NULL);
   if (!shader)
      return NULL;
   pco_process_ir(ctx, shader);
   pco_encode_ir(ctx, shader);
   return shader;
}

static int write_binary(const char *path, pco_shader *shader)
{
   FILE *file = fopen(path, "wb");
   if (!file) {
      fprintf(stderr, "fopen(%s): %s\n", path, strerror(errno));
      return 1;
   }

   const unsigned size = pco_shader_binary_size(shader);
   const size_t written =
      fwrite(pco_shader_binary_data(shader), 1, size, file);
   const int close_result = fclose(file);
   if (written != size || close_result != 0) {
      fprintf(stderr, "failed to write complete PCO binary: %s\n", path);
      return 1;
   }
   fprintf(stderr, "wrote %u bytes of Mesa PCO binary to %s\n", size, path);
   return 0;
}

static int generate_case(pco_ctx *ctx,
                         unsigned varying_count,
                         const char *vertex_path,
                         const char *fragment_path)
{
   pco_data vertex_data = { 0 };
   vertex_data.vs.attrib_formats[VERT_ATTRIB_GENERIC0] =
      PIPE_FORMAT_R32G32_FLOAT;
   vertex_data.vs.attribs[VERT_ATTRIB_GENERIC0] = (pco_range){
      .start = 0,
      .count = 2,
   };
   vertex_data.common.vtxins = 2;
   vertex_data.vs.varyings[VARYING_SLOT_POS] = (pco_range){
      .start = 0,
      .count = 4,
   };
   for (unsigned index = 0; index < varying_count; ++index) {
      vertex_data.vs.varyings[VARYING_SLOT_VAR0 + index] = (pco_range){
         .start = 4 + index * 4,
         .count = 4,
      };
   }
   vertex_data.vs.f32_smooth = varying_count * 4;
   vertex_data.vs.vtxouts = 4 + varying_count * 4;

   pco_data fragment_data = { 0 };
   fragment_data.fs.uses.w = true;
   fragment_data.fs.varyings[VARYING_SLOT_POS] = (pco_range){
      .start = 0,
      .count = ROGUE_USC_COEFFICIENT_SET_SIZE,
   };
   for (unsigned index = 0; index < varying_count; ++index) {
      fragment_data.fs.varyings[VARYING_SLOT_VAR0 + index] = (pco_range){
         .start = (1 + index * 4) * ROGUE_USC_COEFFICIENT_SET_SIZE,
         .count = 4 * ROGUE_USC_COEFFICIENT_SET_SIZE,
      };
   }
   /* Each public Rogue coefficient set is A/B/C/PAD (four dwords): one
    * position-W set followed by four sets per smooth vec4 varying. */
   fragment_data.common.coeffs =
      (1 + varying_count * 4) * ROGUE_USC_COEFFICIENT_SET_SIZE;

   nir_shader *vertex_nir = build_vertex_shader(varying_count);
   nir_shader *fragment_nir = build_fragment_shader(varying_count);
   pco_shader *vertex =
      compile_shader(ctx, vertex_nir, &vertex_data);
   pco_shader *fragment =
      compile_shader(ctx, fragment_nir, &fragment_data);
   if (!vertex || !fragment) {
      fprintf(stderr, "PCO varying shader translation failed\n");
      ralloc_free(vertex);
      ralloc_free(fragment);
      ralloc_free(vertex_nir);
      ralloc_free(fragment_nir);
      return 1;
   }

   const char *case_name = varying_count == 1
                              ? "varyings_shader_1"
                              : varying_count == 2 ? "varyings_shader_2"
                                : varying_count == 4 ? "varyings_shader_4"
                                                     : "varyings_shader_8";
   pco_print_shader(vertex, stdout, case_name);
   pco_print_binary(vertex, stdout, case_name);
   pco_print_shader(fragment, stdout, case_name);
   pco_print_binary(fragment, stdout, case_name);
   const int result = write_binary(vertex_path, vertex) ||
                      write_binary(fragment_path, fragment);
   ralloc_free(vertex);
   ralloc_free(fragment);
   ralloc_free(vertex_nir);
   ralloc_free(fragment_nir);
   return result;
}

int main(int argc, char **argv)
{
   if (argc != 3 && argc != 5 && argc != 7 && argc != 9) {
      fprintf(stderr,
              "Usage: %s V1_VERTEX.bin V1_FRAGMENT.bin "
              "[V2_VERTEX.bin V2_FRAGMENT.bin "
              "[V4_VERTEX.bin V4_FRAGMENT.bin "
              "[V8_VERTEX.bin V8_FRAGMENT.bin]]]\n",
              argv[0]);
      return 2;
   }

   void *mem_ctx = ralloc_context(NULL);
   struct pvr_device_info dev_info;
   struct pvr_device_runtime_info runtime_info = { 0 };
   if (!pvr_device_info_init_public_name(&dev_info, "gx6250")) {
      fprintf(stderr, "Mesa does not recognize the public gx6250 target\n");
      ralloc_free(mem_ctx);
      return 1;
   }
   pco_ctx *ctx = pco_ctx_create(&dev_info, &runtime_info, mem_ctx);

   int result = generate_case(ctx, 1, argv[1], argv[2]);
   if (!result && argc >= 5)
      result = generate_case(ctx, 2, argv[3], argv[4]);
   if (!result && argc >= 7)
      result = generate_case(ctx, 4, argv[5], argv[6]);
   if (!result && argc == 9)
      result = generate_case(ctx, 8, argv[7], argv[8]);
   ralloc_free(mem_ctx);
   return result;
}
