/*
 * Development-time generator for GLBench attribute_fetch_shader cases
 * 1/2/4/8.
 *
 * This utility builds real NIR and invokes Mesa's public PowerVR PCO backend;
 * it does not define a project-local opcode format.  The vertex inputs are
 * GLES vec4 attributes backed by PIPE_FORMAT_R32G32_FLOAT, so Mesa's PVI
 * (PDS Vertex/Instance) lowering supplies x/y from consecutive vertex-input
 * registers and the GLES z=0,w=1 defaults. Cases 2/4/8 add the attributes with
 * public PCO FADD instructions. The fragment shader writes exact IEEE-754 0.5
 * to RGBA.
 * Generated files are immutable fixtures; PvrGPU has no runtime Mesa compiler
 * dependency.
 *
 * Provenance:
 *   Mesa 26.2.1 commit da14d65e4499e66468094be52bff9ea0915a695e
 *   public target gx6250
 * Expected output:
 *   VS 56 bytes, SHA-256
 *   01fb08add3c710fb9062ed0033fecc15e5cfbce56a38a49ed17db4e43f2bf026
 *   two-attribute VS 56 bytes, SHA-256
 *   a275bcd7b146f7243e995528c197a04ee24e17f11d313313c7a5bea78030b88f
 *   four-attribute VS 96 bytes, SHA-256
 *   81b4bf2b412eb2ba35adcd1076d965918336ffb0ffb860e66547695ef4a6ae28
 *   eight-attribute VS 176 bytes, SHA-256
 *   877802fe53fd258bb114aa2cf5713c317405c986b3a43b4612a58b6db9f7eccb
 *   FS 48 bytes, SHA-256
 *   45a123bc247c1b37570721ad7a18894be4d6802dfb459de191ebfc3a32ec5959
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

static pco_shader *compile_shader(pco_ctx *ctx, nir_shader *nir,
                                  pco_data *data)
{
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

static nir_shader *build_vertex_shader(unsigned attribute_count)
{
   const char *name = attribute_count == 1
                         ? "attribute_fetch_1_vs"
                         : attribute_count == 2
                              ? "attribute_fetch_2_vs"
                              : attribute_count == 4 ? "attribute_fetch_4_vs"
                                                     : "attribute_fetch_8_vs";
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
                                                   pco_nir_options(),
                                                   "%s",
                                                   name);
   nir_def *sum = NULL;
   for (unsigned index = 0; index < attribute_count; ++index) {
      gl_vert_attrib location = VERT_ATTRIB_GENERIC0 + index;
      nir_variable *input = nir_variable_create(b.shader,
                                                nir_var_shader_in,
                                                glsl_vec4_type(),
                                                "attribute");
      input->data.location = location;

      nir_def *attribute = nir_load_input(
         &b,
         4,
         32,
         nir_imm_int(&b, 0),
         .base = 0,
         .range = 1,
         .component = 0,
         .dest_type = nir_type_float32,
         .io_semantics = (nir_io_semantics){
            .location = location,
            .num_slots = 1,
         });
      sum = sum ? nir_fadd(&b, sum, attribute) : attribute;
   }

   nir_variable *position = nir_variable_create(b.shader,
                                                 nir_var_shader_out,
                                                 glsl_vec4_type(),
                                                 "gl_Position");
   position->data.location = VARYING_SLOT_POS;
   nir_store_output(&b,
                    sum,
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
   nir_jump(&b, nir_jump_return);
   return b.shader;
}

static nir_shader *build_fragment_shader(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                   pco_nir_options(),
                                                   "attribute_fetch_gray_fs");
   for (unsigned component = 0; component < 4; ++component) {
      nir_frag_store_pco(&b,
                         nir_imm_int(&b, (int32_t)UINT32_C(0x3f000000)),
                         .base = component);
   }
   nir_jump(&b, nir_jump_return);
   return b.shader;
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

static int generate_vertex(pco_ctx *ctx, unsigned attribute_count,
                           const char *path)
{
   pco_data data = { 0 };
   for (unsigned index = 0; index < attribute_count; ++index) {
      gl_vert_attrib location = VERT_ATTRIB_GENERIC0 + index;
      data.vs.attrib_formats[location] = PIPE_FORMAT_R32G32_FLOAT;
      data.vs.attribs[location] = (pco_range){
         .start = index * 2,
         .count = 2,
      };
   }
   data.common.vtxins = attribute_count * 2;
   data.vs.varyings[VARYING_SLOT_POS] = (pco_range){
      .start = 0,
      .count = 4,
   };
   data.vs.vtxouts = 4;

   pco_shader *shader =
      compile_shader(ctx, build_vertex_shader(attribute_count), &data);
   if (!shader) {
      fprintf(stderr, "PCO vertex translation failed\n");
      return 1;
   }
   pco_print_shader(shader, stdout, path);
   pco_print_binary(shader, stdout, path);
   const int result = write_binary(path, shader);
   ralloc_free(shader);
   return result;
}

static int generate_fragment(pco_ctx *ctx, const char *path)
{
   pco_data data = { 0 };
   pco_shader *shader = compile_shader(ctx, build_fragment_shader(), &data);
   if (!shader) {
      fprintf(stderr, "PCO fragment translation failed\n");
      return 1;
   }
   pco_print_shader(shader, stdout, path);
   pco_print_binary(shader, stdout, path);
   const int result = write_binary(path, shader);
   ralloc_free(shader);
   return result;
}

int main(int argc, char **argv)
{
   if (argc != 6) {
      fprintf(stderr,
              "Usage: %s ONE_ATTR_VS.bin TWO_ATTR_VS.bin FOUR_ATTR_VS.bin "
              "EIGHT_ATTR_VS.bin FRAGMENT_OUTPUT.bin\n",
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
   const int result = generate_vertex(ctx, 1, argv[1]) ||
                      generate_vertex(ctx, 2, argv[2]) ||
                      generate_vertex(ctx, 4, argv[3]) ||
                      generate_vertex(ctx, 8, argv[4]) ||
                      generate_fragment(ctx, argv[5]);
   ralloc_free(mem_ctx);
   return result;
}
