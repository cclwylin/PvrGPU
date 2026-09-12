/* SPDX-License-Identifier: MIT */
/* Genuine NIR -> PCO coverage for budget-pressure-only prefix allocation.
 * Optional arguments: artifact directory, then "old" for the frozen compiler
 * negative control. No captured values or hand-authored instruction bytes. */
#include "pvrgpu_pco.h"
#include "nir/nir.h"
#include "nir/nir_builder.h"
#include "pco/pco.h"
#include "util/ralloc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks;
static const char *current;
static void require(bool condition, const char *message)
{
   ++checks;
   if (!condition) {
      fprintf(stderr, "shared_budget: %s: FAIL: %s\n", current, message);
      exit(1);
   }
}

struct stage_case {
   unsigned bound, base, offset, range, components, bits, index_bits;
   unsigned ubos, block;
   bool indirect, load_ubo, size_ubo, indirect_ubo, no_uniform, deref_ubo, image;
};

static nir_variable *variable(nir_shader *s, nir_variable_mode mode,
                              const char *name, unsigned location)
{
   nir_variable *v = nir_variable_create(s, mode, glsl_vec4_type(), name);
   v->data.location = location;
   return v;
}

static nir_shader *make_shader(mesa_shader_stage stage, struct stage_case c)
{
   nir_builder b = nir_builder_init_simple_shader(stage, pco_nir_options(), "shared_budget_fixture");
   nir_variable *in = variable(b.shader, nir_var_shader_in, "position",
      stage == MESA_SHADER_VERTEX ? VERT_ATTRIB_GENERIC0 : VARYING_SLOT_POS);
   nir_variable *out = variable(b.shader, nir_var_shader_out, "result",
      stage == MESA_SHADER_VERTEX ? VARYING_SLOT_POS : FRAG_RESULT_DATA0);
   nir_def *position = nir_load_var(&b, in);
   nir_def *value = nir_imm_float(&b, 0.25);
   if (!c.no_uniform) {
      nir_def *index = c.indirect ? nir_f2u32(&b, nir_channel(&b, position, 0)) :
         nir_imm_intN_t(&b, c.offset, c.index_bits);
      if (c.indirect && c.index_bits != 32)
         index = nir_u2uN(&b, index, c.index_bits);
      nir_def *uniform = nir_load_uniform(&b, c.components, c.bits, index,
         .base = c.base, .range = c.range,
         .dest_type = nir_type_float | c.bits);
      for (unsigned i = 0; i < c.components; ++i) {
         nir_def *word = nir_channel(&b, uniform, i);
         if (c.bits != 32) word = nir_f2f32(&b, word);
         value = nir_fadd(&b, value, word);
      }
   }
   if (c.load_ubo || c.size_ubo) {
      nir_def *block = c.indirect_ubo ? nir_f2u32(&b, nir_channel(&b, position, 1)) :
         nir_imm_int(&b, c.block);
      nir_def *ubo = c.size_ubo ? nir_u2f32(&b, nir_get_ubo_size(&b, 32, block)) :
         nir_load_ubo(&b, 1, 32, block, nir_imm_int(&b, 0),
            .align_mul = 4, .align_offset = 0, .range = 4);
      value = nir_fadd(&b, value, ubo);
   }
   if (c.deref_ubo) {
      nir_variable *ubo = variable(b.shader, nir_var_mem_ubo, "unlowered_ubo", 0);
      value = nir_fadd(&b, value, nir_channel(&b, nir_load_var(&b, ubo), 0));
   }
   if (c.image) {
      nir_image_atomic(&b, 32, nir_imm_int(&b, 0), nir_imm_ivec4(&b, 0, 0, 0, 0),
         nir_imm_int(&b, 0), nir_imm_int(&b, 1),
         .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = false,
         .format = PIPE_FORMAT_R32_UINT, .atomic_op = nir_atomic_op_iadd);
   }
   nir_def *result = nir_vec4(&b, value, value, value, value);
   if (stage == MESA_SHADER_VERTEX) result = nir_fadd(&b, position, result);
   nir_store_var(&b, out, result, 15);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   b.shader->info.num_ubos = c.ubos;
   b.shader->info.num_images = c.image ? 1 : 0;
   return b.shader;
}

static void save(const char *directory, const char *suffix, const void *data, size_t size)
{
   if (!directory) return;
   char path[4096];
   require(snprintf(path, sizeof(path), "%s/%s.%s", directory, current, suffix) < sizeof(path), "artifact path");
   FILE *f = fopen(path, "wb");
   require(f != NULL, "open artifact");
   require(fwrite(data, 1, size, f) == size, "write artifact");
   require(fclose(f) == 0, "close artifact");
}

int main(int argc, char **argv)
{
   const char *directory = argc > 1 ? argv[1] : NULL;
   const bool old = argc > 2 && strcmp(argv[2], "old") == 0;
   char error[1024];
   current = "setup";
   struct pvrgpu_pco_compiler *compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
   require(compiler != NULL, error);
   const struct stage_case empty = {.no_uniform = true, .bits = 32, .index_bits = 32};
   const struct stage_case full = {.bound = 384, .base = 95, .range = 1,
      .components = 4, .bits = 32, .index_bits = 32, .ubos = 1};
   enum { CASE_COUNT = 38 };
   for (unsigned i = 0; i < CASE_COUNT; ++i) {
      struct stage_case v = full, f = empty;
      bool success = true, rescued = false;
      unsigned expected_ubos = 0, expected_count = 384, expected_start = 0;
      bool fragment = false, packed = false, dma = false;
      unsigned expected_cb0_slot = 0, expected_image_start = 4;
      switch (i) {
      case 0: current = "full384-dead-ubo"; rescued = true; break;
      case 1: current = "full384-live-ubo-packed"; v.load_ubo = true; packed = true;
         expected_ubos = 1; expected_count = 4; expected_start = 4; rescued = true; break;
      case 2: current = "static380-live-ubo"; v.base = 94; v.load_ubo = true;
         expected_ubos = 1; expected_count = 380; expected_start = 4; rescued = true; break;
      case 3: current = "indirect380-live-ubo"; v.base = 92; v.range = 3; v.indirect = true; v.load_ubo = true;
         expected_ubos = 1; expected_count = 380; expected_start = 4; rescued = true; break;
      case 4: current = "indirect384-live-ubo-packed"; v.base = 92; v.range = 4; v.indirect = true; v.load_ubo = true; packed = true;
         expected_ubos = 1; expected_count = 16; expected_start = 4; rescued = true; break;
      case 5: current = "high-binding-prefix-retained"; v.base = 93; v.ubos = 3; v.block = 1; v.load_ubo = true;
         expected_ubos = 2; expected_count = 376; expected_start = 8; rescued = true; break;
      case 6: current = "size-query-keeps-ubo-packed"; v.size_ubo = true; packed = true;
         expected_ubos = 1; expected_count = 4; expected_start = 4; rescued = true; break;
      case 7: current = "dynamic-binding-keeps-all"; v.load_ubo = true; v.indirect_ubo = true; success = false; break;
      case 8: current = "already-fitting-dead-ubo-unchanged"; v.bound = 380; v.base = 0;
         expected_ubos = 1; expected_count = 380; expected_start = 4; break;
      case 9: current = "already-fitting-live-ubo-unchanged"; v.bound = 380; v.base = 0; v.load_ubo = true;
         expected_ubos = 1; expected_count = 380; expected_start = 4; break;
      case 10: current = "no-load-canonical"; v.no_uniform = true; v.bound = UINT32_MAX;
         expected_ubos = 1; expected_count = 0; expected_start = 4; break;
      case 11: current = "bound-oob-refused"; v.bound = 383; success = false; break;
      case 12: current = "base-add-overflow-refused"; v.base = UINT32_MAX; v.offset = 1; success = false; break;
      case 13: current = "byte-address-overflow-refused"; v.bound = UINT32_MAX; v.base = UINT32_MAX / 16 + 1; success = false; break;
      case 14: current = "unknown-indirect-range-refused"; v.base = 0; v.range = 0; v.indirect = true; success = false; break;
      case 15: current = "wide-indirect-range-refused"; v.base = UINT32_MAX; v.range = UINT32_MAX; v.indirect = true; success = false; break;
      case 16: current = "uniform16-refused"; v.bits = 16; success = false; break;
      case 17: current = "uniform64-refused"; v.bits = 64; success = false; break;
      case 18: current = "index64-refused"; v.index_bits = 64; success = false; break;
      case 19: current = "components8-refused"; v.components = 8; success = false; break;
      case 20: current = "fragment380-live-ubo"; f = full; f.base = 94; f.load_ubo = true; v = empty;
         fragment = true; expected_ubos = 1; expected_count = 380; expected_start = 4; rescued = true; break;
      case 21: current = "fragment384-live-ubo-packed"; f = full; f.base = 95; f.load_ubo = true; v = empty;
         fragment = true; packed = true; expected_ubos = 1; expected_count = 4; expected_start = 4; rescued = true; break;
      case 22: current = "fragment384-dead-ubo"; f = full; v = empty;
         fragment = true; expected_count = 384; rescued = true; break;
      case 23: current = "static-nonzero-offset"; v.base = 92; v.offset = 2; v.load_ubo = true;
         expected_ubos = 1; expected_count = 380; expected_start = 4; rescued = true; break;
      case 24: current = "scalar-last-word-prefix"; v.base = 94; v.components = 1; v.load_ubo = true;
         expected_ubos = 1; expected_count = 377; expected_start = 4; rescued = true; break;
      case 25: current = "declared-ubo-count-limit-refused"; v.ubos = 16; success = false; break;
      case 26: current = "positive-base-offset-overflow-refused"; v.base = 1; v.offset = UINT32_MAX; success = false; break;
      case 27: current = "unlowered-ubo-deref-keeps-all"; v.deref_ubo = true; success = false; break;
      case 28: current = "size-query-preserves-binding-prefix"; v.base = 93; v.ubos = 3; v.block = 1; v.size_ubo = true;
         expected_ubos = 2; expected_count = 376; expected_start = 8; rescued = true; break;
      case 29: current = "fragment-image-pins-ubo-packed"; f = full; f.base = 93; f.image = true; v = empty;
         fragment = true; packed = true; expected_ubos = 1; expected_count = 4; expected_start = 12; rescued = true; break;
      case 30: current = "fragment-image-cb0-only-shrink"; f = full; f.base = 92; f.image = true; v = empty;
         fragment = true; expected_ubos = 1; expected_count = 372; expected_start = 12; rescued = true; break;
      case 31: current = "fragment-image-already-fitting-unchanged"; f = full; f.bound = 372; f.base = 92; f.image = true; v = empty;
         fragment = true; expected_ubos = 1; expected_count = 372; expected_start = 12; break;
      case 32: current = "d176-large-dynamic-vertex-cb0-dma";
         v.bound = 400; v.base = 4; v.range = 96; v.indirect = true; v.ubos = 0;
         expected_ubos = 1; expected_count = 0; expected_start = 4;
         expected_cb0_slot = 1; dma = true; rescued = true; break;
      case 33: current = "large-dynamic-vertex-after-live-ubo-prefix";
         v.bound = 400; v.base = 4; v.range = 96; v.indirect = true;
         v.ubos = 3; v.block = 1; v.load_ubo = true;
         expected_ubos = 3; expected_count = 0; expected_start = 12;
         expected_cb0_slot = 3; dma = true; rescued = true; break;
      case 34: current = "large-dynamic-fragment-cb0-dma";
         f = full; f.bound = 400; f.base = 4; f.range = 96;
         f.indirect = true; f.ubos = 0; v = empty; fragment = true;
         expected_ubos = 1; expected_count = 0; expected_start = 4;
         expected_cb0_slot = 1; dma = true; rescued = true; break;
      case 35: current = "fragment-image-after-large-cb0-dma";
         f = full; f.bound = 400; f.base = 4; f.range = 96;
         f.indirect = true; f.image = true; v = empty; fragment = true;
         expected_ubos = 2; expected_count = 0; expected_start = 16;
         expected_cb0_slot = 2; expected_image_start = 8;
         dma = true; rescued = true; break;
      case 36: current = "large-cb0-after-fifteen-ubos-refused";
         v.bound = 400; v.base = 4; v.range = 96; v.indirect = true;
         v.ubos = 15; v.block = 14; v.load_ubo = true; success = false; break;
      case 37: current = "large-cb0-over-64k-refused";
         v.bound = 16388; v.base = 0; v.range = 4097; v.indirect = true;
         v.ubos = 0; success = false; break;
      }
      nir_shader *vs = make_shader(MESA_SHADER_VERTEX, v);
      nir_shader *fs = make_shader(MESA_SHADER_FRAGMENT, f);
      const enum pipe_format format = PIPE_FORMAT_R32G32B32A32_FLOAT;
      struct pvrgpu_pco_graphics_binary result = {0};
      const bool ok = pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format,
         false, false, 1, v.bound, f.bound, 1, 0, &result, error, sizeof(error));
      require(ok == (success && !(old && rescued)), error);
      require(vs->info.num_ubos == v.ubos && fs->info.num_ubos == f.ubos, "caller NIR metadata untouched");
      if (ok) {
         const struct pvrgpu_pco_stage_abi *abi = fragment ? &result.fragment.abi : &result.vertex.abi;
         require(abi->uniform_buffer_descriptor_count == expected_ubos, "exact descriptor count");
         require(abi->push_constant_start == expected_start, "exact CB0 start");
         require(abi->push_constant_count == expected_count, "exact CB0 prefix length");
         require(abi->shareds == expected_start + expected_count, "physical shared allocation");
         const struct pvrgpu_pco_uniform_word_map *map = fragment ? &result.fragment.cb0_word_map : &result.vertex.cb0_word_map;
         const struct pvrgpu_pco_owned_binary *owned =
            fragment ? &result.fragment : &result.vertex;
         require(map->count == (packed ? expected_count : 0), "packing only formerly rejected cases");
         require(map->source_dwords == (packed ? (fragment ? f.bound : v.bound) : 0), "original source bound retained");
         require(owned->cb0_uniform_buffer_slot == expected_cb0_slot,
                 "large CB0 uses only its appended native UBO slot");
         require(dma == (expected_cb0_slot != 0), "DMA case expectation is coherent");
         require(result.vertex.data && result.vertex.size && result.fragment.data && result.fragment.size, "real compiler bytes");
         if (f.image) {
            require(result.fragment_image_descriptor_start == expected_image_start,
                    "image descriptor address follows the final UBO inventory");
            require(result.fragment_image_descriptor_count == 1 &&
               result.fragment_image_read_mask == 1 && result.fragment_image_write_mask == 1,
               "real image atomic descriptor identity");
         }
         save(directory, "vs.bin", result.vertex.data, result.vertex.size);
         save(directory, "fs.bin", result.fragment.data, result.fragment.size);
         save(directory, "vs.abi", &result.vertex.abi, sizeof(result.vertex.abi));
         save(directory, "fs.abi", &result.fragment.abi, sizeof(result.fragment.abi));
      } else {
         require(!result.vertex.data && !result.fragment.data && !result.vertex.size && !result.fragment.size, "failed compiler leaves no binary");
         require(error[0], "explicit rejection reason");
      }
      printf("%s: %s%s\n", current, ok ? "compiled" : "rejected", old && rescued ? " (frozen negative control)" : "");
      pvrgpu_pco_graphics_binary_finish(&result);
      ralloc_free(vs); ralloc_free(fs);
   }
   pvrgpu_pco_compiler_destroy(compiler);
   printf("shared_budget: PASS %u checks, %u cases%s\n", checks, CASE_COUNT, old ? " (old)" : "");
   return 0;
}
