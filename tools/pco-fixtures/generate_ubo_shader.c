/* Genuine Mesa 26.2.1 (da14d65e4499e66468094be52bff9ea0915a695e) UBO
 * fixtures. Descriptor SH0..3 is base64,size,dynamic_offset; SH4 supplies a
 * runtime byte offset. UBO values are emitted without numeric conversion. */
#include "common/pvr_device_info.h"
#include "nir/nir.h"
#include "nir/nir_builder.h"
#include "pco/pco.h"
#include "pco/pco_data.h"
#include "util/ralloc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
   if (argc != 4) {
      fprintf(stderr, "Usage: %s OUTPUT.bin vertex|fragment COUNT\n", argv[0]);
      return 2;
   }
   const bool vertex = strcmp(argv[2], "vertex") == 0;
   unsigned count = strtoul(argv[3], NULL, 10);
   if ((!vertex && strcmp(argv[2], "fragment") != 0) || count < 1 || count > 4)
      return 2;
   void *mem = ralloc_context(NULL);
   struct pvr_device_info device;
   struct pvr_device_runtime_info runtime = {0};
   if (!pvr_device_info_init_public_name(&device, "gx6250"))
      return 1;
   pco_ctx *ctx = pco_ctx_create(&device, &runtime, mem);
   nir_builder b = nir_builder_init_simple_shader(
      vertex ? MESA_SHADER_VERTEX : MESA_SHADER_FRAGMENT,
      pco_nir_options(), "ubo_raw_dwords");
   b.shader->info.internal = true;
   nir_def *offset = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0),
                                             .base = 0, .range = 4);
   nir_def *value = nir_load_ubo(&b, count, 32,
                                nir_imm_ivec2(&b, 1 << 16, 0), offset,
                                .align_mul = 4, .align_offset = 0,
                                .range_base = 0, .range = 65536);
   if (vertex) {
      for (unsigned channel = 0; channel < count; ++channel)
         nir_uvsw_write_pco(&b, nir_imm_int(&b, channel),
                            nir_channel(&b, value, channel));
   } else {
      for (unsigned channel = 0; channel < 4; ++channel)
         nir_frag_store_pco(&b, channel < count ? nir_channel(&b, value, channel)
                                               : nir_imm_int(&b, 0),
                            .base = channel);
   }
   nir_jump(&b, nir_jump_return);
   pco_binding_data bindings[2] = {0};
   bindings[1].used = true;
   bindings[1].range = (pco_range){.start = 0, .count = 4, .stride = 4};
   pco_data data = {0};
   data.common.desc_sets[0].used = true;
   data.common.desc_sets[0].binding_count = 2;
   data.common.desc_sets[0].bindings = bindings;
   data.common.desc_sets[0].range = (pco_range){.start = 0, .count = 4};
   data.common.shareds = 5;
   pco_preprocess_nir(ctx, b.shader);
   pco_lower_nir(ctx, b.shader, &data);
   pco_postprocess_nir(ctx, b.shader, &data);
   data.common.push_consts.range = (pco_range){.start = 4, .count = 1};
   data.common.shareds = 5;
   if (getenv("PVRGPU_DEBUG_PCO_NIR"))
      nir_print_shader(b.shader, stderr);
   pco_shader *shader = pco_trans_nir(ctx, b.shader, &data, mem);
   pco_process_ir(ctx, shader);
   if (getenv("PVRGPU_DEBUG_PCO_NIR"))
      pco_print_shader(shader, stderr, "ubo_fixture");
   pco_encode_ir(ctx, shader);
   unsigned size = pco_shader_binary_size(shader);
   FILE *file = fopen(argv[1], "wb");
   if (!file)
      return 1;
   bool ok = fwrite(pco_shader_binary_data(shader), 1, size, file) == size;
   ok = fclose(file) == 0 && ok;
   fprintf(stderr, "wrote %u bytes of Mesa PCO to %s\n", size, argv[1]);
   ralloc_free(mem);
   return ok ? 0 : 1;
}
