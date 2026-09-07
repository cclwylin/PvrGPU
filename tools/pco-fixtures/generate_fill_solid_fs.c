/*
 * Development-time generator for solid-color fragment-program fixtures.
 *
 * This utility deliberately uses Mesa's public PowerVR PCO compiler rather
 * than inventing a project-local instruction encoding.  It builds a minimal
 * fragment NIR shader which writes one of the named, exact IEEE-754 RGBA
 * fixtures to the four PowerVR pixel-output registers, then emits the exact
 * PCO binary bytes.  In particular, half-valued color/alpha components make
 * Mesa select public special constant sc75 for IEEE-754 0.5; they are used by
 * Fill.Solid blend probes and GLBench triangle-setup probes.
 * Generated binaries are checked into the model as immutable source data;
 * the SystemC runtime does not depend on a Mesa compiler installation.
 *
 * Mesa source provenance used by this project:
 *   Mesa 26.2.1, commit da14d65e4499e66468094be52bff9ea0915a695e
 */

#include "common/pvr_device_info.h"
#include "nir/nir.h"
#include "nir/nir_builder.h"
#include "nir/nir_builder_opcodes.h"
#include "pco/pco.h"
#include "pco/pco_data.h"
#include "util/ralloc.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct fixture {
   const char *name;
   const char *shader_name;
   uint32_t rgba[4];
};

static const struct fixture fixtures[] = {
   {
      "red",
      "fill_solid_fs_red",
      {
         UINT32_C(0x3f800000),
         UINT32_C(0x00000000),
         UINT32_C(0x00000000),
         UINT32_C(0x3f800000),
      },
   },
   {
      "red-half-alpha",
      "fill_solid_fs_red_half_alpha",
      {
         UINT32_C(0x3f800000),
         UINT32_C(0x00000000),
         UINT32_C(0x00000000),
         UINT32_C(0x3f000000),
      },
   },
   {
      "green-half-alpha",
      "fill_solid_fs_green_half_alpha",
      {
         UINT32_C(0x00000000),
         UINT32_C(0x3f800000),
         UINT32_C(0x00000000),
         UINT32_C(0x3f000000),
      },
   },
   {
      "triangle-setup-orange",
      "triangle_setup_fs_orange",
      {
         UINT32_C(0x3f800000),
         UINT32_C(0x3f000000),
         UINT32_C(0x00000000),
         UINT32_C(0x3f800000),
      },
   },
   {
      "triangle-setup-half-culled-cyan",
      "triangle_setup_half_culled_fs_cyan",
      {
         UINT32_C(0x00000000),
         UINT32_C(0x3f000000),
         UINT32_C(0x3f000000),
         UINT32_C(0x3f800000),
      },
   },
};

static const struct fixture *find_fixture(const char *name)
{
   for (unsigned i = 0; i < sizeof(fixtures) / sizeof(fixtures[0]); ++i) {
      if (strcmp(fixtures[i].name, name) == 0)
         return &fixtures[i];
   }

   return NULL;
}

static pco_shader *build_shader(pco_ctx *ctx, nir_shader *nir,
                                pco_data *data)
{
   pco_preprocess_nir(ctx, nir);
   pco_lower_nir(ctx, nir, data);
   pco_postprocess_nir(ctx, nir, data);
   if (getenv("PVRGPU_DEBUG_PCO_NIR"))
      nir_print_shader(nir, stderr);
   if (data->common.push_consts.used) {
      data->common.push_consts.range = (pco_range){
         .start = 0, .count = data->common.push_consts.used,
      };
      data->common.shareds = data->common.push_consts.used;
   }
   if (data->fs.uses.depth_feedback) {
      nir_builder b = nir_builder_create(nir_shader_get_entrypoint(nir));
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_isp_feedback_pco)
               continue;
            b.cursor = nir_before_instr(instr);
            nir_src_rewrite(&intr->src[0], nir_undef(&b, 1, 32));
         }
      }
   }

   pco_shader *shader = pco_trans_nir(ctx, nir, data, NULL);
   pco_process_ir(ctx, shader);
   if (getenv("PVRGPU_DEBUG_PCO_NIR"))
      pco_print_shader(shader, stderr, "fixture");
   pco_encode_ir(ctx, shader);
   return shader;
}

int main(int argc, char **argv)
{
   if (argc < 2 || argc > 4 ||
       (argc == 4 && strcmp(argv[3], "depth-feedback") != 0)) {
      fprintf(stderr,
              "Usage: %s OUTPUT.bin "
              "[red|red-half-alpha|green-half-alpha|"
              "triangle-setup-orange|triangle-setup-half-culled-cyan] "
              "[depth-feedback]\n",
              argv[0]);
      return 2;
   }

   const struct fixture *fixture = find_fixture(argc >= 3 ? argv[2] : "red");
   if (!fixture) {
      fprintf(stderr, "unknown fixture: %s\n", argv[2]);
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
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                   pco_nir_options(),
                                                   "%s",
                                                   fixture->shader_name);

   /* A genuine public DEPTHF, with the same fsat lowering used for
    * gl_FragDepth.  Shared register 0 permits independent clamp probes. */
   if (argc == 4) {
      b.shader->info.internal = true;
      nir_def *depth = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0),
                                              .base = 0, .range = 4);
      nir_isp_feedback_pco(&b, nir_undef(&b, 1, 32), nir_fsat(&b, depth));
   }

   /* Raw IEEE-754 float32 values written to pixout0..3. */
   for (unsigned component = 0; component < 4; ++component) {
      nir_frag_store_pco(&b,
                         nir_imm_int(&b, (int32_t)fixture->rgba[component]),
                         .base = component);
   }
   nir_jump(&b, nir_jump_return);

   pco_data data = { 0 };
   data.fs.uses.depth_feedback = argc == 4;
   pco_shader *shader = build_shader(ctx, b.shader, &data);
   const unsigned size = pco_shader_binary_size(shader);
   const void *bytes = pco_shader_binary_data(shader);

   FILE *output = fopen(argv[1], "wb");
   if (!output) {
      perror("fopen");
      ralloc_free(shader);
      ralloc_free(mem_ctx);
      return 1;
   }
   const size_t written = fwrite(bytes, 1, size, output);
   const int close_result = fclose(output);
   if (written != size || close_result != 0) {
      fprintf(stderr, "failed to write complete PCO binary\n");
      ralloc_free(shader);
      ralloc_free(mem_ctx);
      return 1;
   }

   fprintf(stderr, "wrote %u bytes of Mesa PCO binary to %s\n", size,
           argv[1]);
   ralloc_free(shader);
   ralloc_free(mem_ctx);
   return 0;
}
