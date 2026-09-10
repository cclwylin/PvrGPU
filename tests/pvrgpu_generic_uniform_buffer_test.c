/* SPDX-License-Identifier: MIT */
/* Actual generic driver compiler ABI regression: no shader identities or
 * precomputed shader results. Verify empty/nonempty CB0 after texture/UBO
 * prefixes, including more texture units than PCO descriptor sets. */
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

static void add_textures(nir_shader *fs, unsigned count, unsigned targets,
                         bool derefs)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(fs);
   nir_builder b = nir_builder_create(impl);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *store = nir_instr_as_intrinsic(instr);
         if (store->intrinsic != nir_intrinsic_store_deref)
            continue;
         b.cursor = nir_before_instr(instr);
         nir_def *value = store->src[1].ssa;
         /* Reverse unit order and revisit the highest unit with different
          * coordinates: descriptor placement must follow bindings, not the
          * order/count of texture instructions. */
         for (unsigned access = 0; access <= count; ++access) {
            const unsigned unit = count - 1 - access % count;
            nir_tex_instr *tex = nir_tex_instr_create(fs, derefs ? 3 : 1);
            tex->op = nir_texop_tex;
            tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
            tex->coord_components = 2;
            tex->dest_type = nir_type_float32;
            tex->texture_index = tex->sampler_index = unit;
            tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
               nir_imm_vec2(&b, .125f * access, .75f));
            if (derefs) {
               nir_variable *sampler = nir_variable_create(fs, nir_var_uniform,
                  glsl_sampler_type(GLSL_SAMPLER_DIM_2D, false, false, GLSL_TYPE_FLOAT),
                  "texture");
               sampler->data.binding = unit;
               nir_def *deref = &nir_build_deref_var(&b, sampler)->def;
               tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_deref, deref);
               tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_sampler_deref, deref);
            }
            nir_def_init(&tex->instr, &tex->def, 4, 32);
            nir_builder_instr_insert(&b, &tex->instr);
            value = nir_fadd(&b, value, &tex->def);
         }
         nir_src_rewrite(&store->src[1], value);
         for (unsigned target = 1; target < targets; ++target) {
            nir_variable *output = nir_variable_create(fs, nir_var_shader_out,
               glsl_vec4_type(), "other_color");
            output->data.location = FRAG_RESULT_DATA0 + target;
            nir_store_var(&b, output, value, 15);
         }
         nir_shader_gather_info(fs, impl);
         fs->info.num_textures = count;
         return;
      }
   }
   check(false, "texture fixture has no color output");
}

static unsigned test_texture_descriptors(struct pvrgpu_pco_compiler *compiler)
{
   const enum pipe_format format = PIPE_FORMAT_R32G32B32A32_FLOAT;
   unsigned programs = 0;
   /* Four is the previous in-bounds edge; five through eight overflowed
    * pco_common_data.desc_sets and corrupted push_consts/adjacent members. */
   for (unsigned textures = 4; textures <= 8; ++textures) {
      for (unsigned blocks = 0; blocks <= 3; blocks += 3) {
         for (unsigned push = 0; push <= 1; ++push) {
            for (unsigned targets = 1; targets <= 4; targets += 3) {
               for (unsigned derefs = 0; derefs <= 1; ++derefs) {
                  nir_shader *vs = shader(true, 4, push);
                  nir_shader *fs = shader(false, blocks, push);
                  add_textures(fs, textures, targets, derefs);
                  struct pvrgpu_pco_graphics_binary binary = {0};
                  char error[512] = {0};
                  check(pvrgpu_pco_compile_color_triangle(compiler, vs, fs,
                     &format, false, false, targets, push ? 4 : 0, push ? 4 : 0,
                     1, textures, &binary, error, sizeof(error)), error);
                  const unsigned texture_words = textures * PVRGPU_PCO_TEXTURE_DESCRIPTOR_DWORDS;
                  const struct pvrgpu_pco_stage_abi *abi = &binary.fragment.abi;
                  check(abi->uniform_buffer_descriptor_start == (blocks ? texture_words : 0) &&
                           abi->uniform_buffer_descriptor_count == blocks,
                        "texture descriptors changed the UBO binding prefix");
                  check(abi->push_constant_start == texture_words + blocks * 4 &&
                           abi->push_constant_count == (push ? 4u : 0u) &&
                           abi->shareds == texture_words + blocks * 4 + (push ? 4u : 0u),
                        "texture descriptor packing corrupted the CB0/shared ABI");
                  check(binary.vertex.abi.push_constant_start == 16 &&
                           binary.vertex.abi.push_constant_count == (push ? 4u : 0u),
                        "fragment descriptor packing corrupted the vertex ABI");
                  nir_foreach_block(block, nir_shader_get_entrypoint(fs)) {
                     nir_foreach_instr(instr, block) {
                        if (instr->type != nir_instr_type_tex) continue;
                        nir_tex_instr *tex = nir_instr_as_tex(instr);
                        check(tex->texture_index < textures && tex->sampler_index < textures,
                              "compiler changed caller-owned texture bindings");
                     }
                  }
                  pvrgpu_pco_graphics_binary_finish(&binary);
                  ralloc_free(vs);
                  ralloc_free(fs);
                  ++programs;
               }
            }
         }
      }
   }
   return programs;
}

static void test_shared_register_limits(struct pvrgpu_pco_compiler *compiler)
{
   const enum pipe_format format = PIPE_FORMAT_R32G32B32A32_FLOAT;
   const struct {
      unsigned vs_words, fs_words, vs_blocks, fs_blocks, textures;
      unsigned vs_shared, fs_shared, vs_map, fs_map;
   } cases[] = {
      {4, 100, 0, 0, 1, 4, 120, 0, 0},
      {80, 4, 4, 0, 1, 96, 24, 0, 0},
      {84, 4, 4, 0, 1, 20, 24, 0, 0},
      {4, 224, 0, 3, 1, 4, 256, 0, 0},
      {4, 228, 0, 3, 1, 4, 36, 0, 0},
      {4, 96, 0, 0, 8, 4, 256, 0, 0},
      {4, 100, 0, 0, 8, 4, 164, 0, 0},
   };
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      nir_shader *vs = shader(true, cases[i].vs_blocks, true);
      nir_shader *fs = shader(false, cases[i].fs_blocks, true);
      add_textures(fs, cases[i].textures, 1, false);
      struct pvrgpu_pco_graphics_binary binary = {0};
      char error[512] = {0};
      const bool accepted = pvrgpu_pco_compile_color_triangle(compiler, vs, fs,
         &format, false, false, 1, cases[i].vs_words, cases[i].fs_words,
         1, cases[i].textures, &binary, error, sizeof(error));
      check(accepted,
            "shared register pressure rejected a valid full or packed CB0 program");
      check(binary.vertex.abi.shareds == cases[i].vs_shared &&
            binary.fragment.abi.shareds == cases[i].fs_shared,
            "accepted stage has an incorrect full/packed shared span");
      check(binary.vertex.cb0_word_map.count == cases[i].vs_map &&
            binary.fragment.cb0_word_map.count == cases[i].fs_map,
            "shared pressure selected the wrong CB0 word map");
      pvrgpu_pco_graphics_binary_finish(&binary);
      ralloc_free(vs);
      ralloc_free(fs);
   }
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
                      false, false, 1, push ? 4 : 0, push ? 4 : 0, 1, 0,
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
   const unsigned texture_programs = test_texture_descriptors(compiler);
   test_shared_register_limits(compiler);
   pvrgpu_pco_compiler_destroy(compiler);
   glsl_type_singleton_decref();
   printf("generic texture/UBO compiler ABI tests: PASS (18 UBO + %u texture programs + 7 SH limits)\n",
          texture_programs);
   return 0;
}
