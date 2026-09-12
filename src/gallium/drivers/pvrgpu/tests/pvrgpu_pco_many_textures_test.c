/* SPDX-License-Identifier: MIT */
/* Genuine NIR tests for the shared descriptor-count and physical SH budgets.
 * Every texture result contributes to the export: no dead resource pruning. */
#ifdef PVRGPU_MANY_TEXTURES_DIRECT_HELPER
#include "../pvrgpu_pco.c"
#endif
int pvrgpu_many_textures_existing_fixture_main(void);
#define main pvrgpu_many_textures_existing_fixture_main
#include "pvrgpu_pco_tessellation_test.c"
#undef main
#include "util/format/u_format.h"
#include <limits.h>

/* Keep the independent coordinate-transport programs reproducible separately
 * from the register-pressure shader. SH20..23 = xyz/layer (uv/layer for 2D),
 * SH24 = explicit LOD; the complete eight-word CB0 is retained. */
static nir_shader *
array_coordinate_fragment(unsigned kind)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      pco_nir_options(), "independent_cube_array_coordinate_transport");
   const bool cube = kind != 2;
   const bool explicit_lod = kind != 0;
   const bool query = kind == 3;
   nir_variable *sampler = nir_variable_create(b.shader, nir_var_uniform,
      glsl_sampler_type(cube ? GLSL_SAMPLER_DIM_CUBE : GLSL_SAMPLER_DIM_2D,
                        false, true, GLSL_TYPE_FLOAT), "image");
   sampler->data.binding = 0;
   nir_variable *out = variable(b.shader, nir_var_shader_out, glsl_vec4_type(),
      "result", FRAG_RESULT_DATA0);
   nir_def *coord = nir_load_uniform(&b, 4, 32, nir_imm_int(&b, 0),
      .base = 0, .range = 1, .dest_type = nir_type_float32);
   nir_def *lod = nir_load_uniform(&b, 1, 32, nir_imm_int(&b, 0),
      .base = 1, .range = 1, .dest_type = nir_type_float32);
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, query ? 1 : explicit_lod ? 2 : 1);
   tex->op = query ? nir_texop_txs : explicit_lod ? nir_texop_txl : nir_texop_tex;
   tex->sampler_dim = cube ? GLSL_SAMPLER_DIM_CUBE : GLSL_SAMPLER_DIM_2D;
   tex->is_array = true;
   tex->coord_components = query ? 0 : cube ? 4 : 3;
   tex->texture_index = tex->sampler_index = 0;
   tex->dest_type = query ? nir_type_int32 : nir_type_float32;
   tex->src[0] = query ? nir_tex_src_for_ssa(nir_tex_src_lod, lod) :
      nir_tex_src_for_ssa(nir_tex_src_coord, cube ? coord : nir_channels(&b, coord, 7));
   if (explicit_lod && !query)
      tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, lod);
   nir_def_init(&tex->instr, &tex->def, query ? 3 : 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_def *value = &tex->def;
   if (query) {
      nir_def *size = nir_i2f32(&b, value);
      value = nir_vec4(&b, nir_channel(&b, size, 0), nir_channel(&b, size, 1),
         nir_channel(&b, size, 2), nir_imm_float(&b, 1));
   }
   nir_store_var(&b, out, value, 15);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   /* These synthetic texture instructions use fixed indices without deref
    * sources; mirror the resource bitmap carried by linked Mesa NIR. */
   BITSET_SET(b.shader->info.textures_used, 0);
   return b.shader;
}

static void
compile_array_coordinate_fixture(struct pvrgpu_pco_compiler *compiler, unsigned kind)
{
   nir_shader *vs = make_vertex(true), *fs = array_coordinate_fragment(kind);
   char *before = nir_shader_as_str(fs, NULL);
   const struct shader_info info = fs->info;
   const enum pipe_format format = PIPE_FORMAT_R32G32_FLOAT;
   struct pvrgpu_pco_graphics_binary binary = {0};
   char error[2048] = {0};
   require(pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format,
      false, false, 1, 0, 8, 1, 1, &binary, error, sizeof(error)), error);
   require(binary.fragment.abi.shareds == 28 &&
           binary.fragment.abi.push_constant_start == 20 &&
           binary.fragment.abi.push_constant_count == 8 &&
           binary.fragment.abi.uniform_buffer_descriptor_count == 0,
           "ordinary array coordinate fixture retains original ABI");
   save_stage(600 + kind, "vs", &binary.vertex);
   save_stage(600 + kind, "fs", &binary.fragment);
   char *after = nir_shader_as_str(fs, NULL);
   require(!strcmp(before, after) && !memcmp(&info, &fs->info, sizeof(info)),
           "array coordinate source unchanged");
   ralloc_free(before); ralloc_free(after);
   pvrgpu_pco_graphics_binary_finish(&binary); ralloc_free(vs); ralloc_free(fs);
}

static nir_shader *
many_textures_fragment(unsigned textures, unsigned last_uniform_slot)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
      pco_nir_options(), "many_distinct_textures_cube_array_and_2d_arrays");
   nir_variable *out = variable(b.shader, nir_var_shader_out, glsl_vec4_type(),
      "result", FRAG_RESULT_DATA0);
   nir_def *coord = nir_load_uniform(&b, 4, 32, nir_imm_int(&b, 0),
      .base = 0, .range = 1, .dest_type = nir_type_float32);
   nir_def *lod = nir_load_uniform(&b, 3, 32, nir_imm_int(&b, 0),
      .base = last_uniform_slot, .range = 1, .dest_type = nir_type_float32);
   nir_def *sum = nir_load_ubo(&b, 4, 32, nir_imm_int(&b, 0), nir_imm_int(&b, 16),
      .align_mul = 16, .range_base = 16, .range = 16);
   for (unsigned slot = 0; slot < textures; ++slot) {
      const bool cube = slot == 6, array = slot >= 6 && slot <= 8;
      nir_variable *sampler = nir_variable_create(b.shader, nir_var_uniform,
         glsl_sampler_type(cube ? GLSL_SAMPLER_DIM_CUBE : GLSL_SAMPLER_DIM_2D,
                           false, array, GLSL_TYPE_FLOAT), "image");
      sampler->data.binding = slot;
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, array ? 2 : 1);
      tex->op = array ? nir_texop_txl : nir_texop_tex;
      tex->sampler_dim = cube ? GLSL_SAMPLER_DIM_CUBE : GLSL_SAMPLER_DIM_2D;
      tex->is_array = array;
      tex->coord_components = cube ? 4 : array ? 3 : 2;
      tex->texture_index = tex->sampler_index = slot;
      tex->dest_type = nir_type_float32;
      tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
         nir_channels(&b, coord, (1u << tex->coord_components) - 1u));
      if (array)
         tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod,
                                          nir_channel(&b, lod, slot - 6));
      nir_def_init(&tex->instr, &tex->def, 4, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      sum = nir_fadd(&b, sum, nir_fmul_imm(&b, &tex->def, slot + 1));
   }
   nir_store_var(&b, out, sum, 15);
   nir_jump(&b, nir_jump_return);
   nir_shader_gather_info(b.shader, b.impl);
   for (unsigned slot = 0; slot < textures; ++slot)
      BITSET_SET(b.shader->info.textures_used, slot);
   b.shader->info.num_ubos = 1;
   require(b.shader->info.num_textures == textures, "every declared texture is used");
   return b.shader;
}

static void
compile_many_textures(struct pvrgpu_pco_compiler *compiler, unsigned textures,
                      unsigned uniform_words, unsigned last_uniform_slot,
                      unsigned expected_shareds, const char *expected_error,
                      unsigned fixture_number)
{
   nir_shader *vs = make_vertex(true);
   nir_shader *fs = many_textures_fragment(textures, last_uniform_slot);
   char *before = nir_shader_as_str(fs, NULL);
   const struct shader_info info = fs->info;
   const enum pipe_format format = PIPE_FORMAT_R32G32_FLOAT;
   struct pvrgpu_pco_graphics_binary binary = {0};
   char error[2048] = {0};
   const bool ok = pvrgpu_pco_compile_color_triangle(compiler, vs, fs, &format,
      false, false, 1, 0, uniform_words, 1, textures, &binary, error, sizeof(error));
   if (expected_error) {
      require(!ok && strstr(error, expected_error), "unsupported count/SH budget fails for exact cause");
      require(!binary.vertex.data && !binary.fragment.data, "refusal returns no executable bytes");
   } else {
      require(ok, error);
      const bool packed = expected_shareds != textures * 20 + 4 + uniform_words;
      const unsigned push_words = expected_shareds - textures * 20 - 4;
      require(binary.fragment.abi.shareds == expected_shareds &&
              binary.fragment.abi.push_constant_start == textures * 20 + 4 &&
              binary.fragment.abi.push_constant_count == push_words &&
              binary.fragment.abi.uniform_buffer_descriptor_start == textures * 20 &&
              binary.fragment.abi.uniform_buffer_descriptor_count == 1,
              "all texture descriptors, UBO and complete fitting CB0 retained");
      require(binary.fragment.cb0_word_map.count == (packed ? push_words : 0) &&
              binary.fragment.cb0_word_map.source_dwords == (packed ? uniform_words : 0),
              "only formerly overflowing CB0 uses driver-private word packing");
      require(binary.fragment.abi.shareds <=
                 PVRGPU_SYSTEMC_MAX_PCO_GRAPHICS_SHARED_DWORDS_PER_STAGE &&
              binary.vertex.abi.shareds <=
                 PVRGPU_SYSTEMC_MAX_PCO_GRAPHICS_SHARED_DWORDS_PER_STAGE,
              "graphics stages stay inside the public shared transport");
      save_stage(fixture_number, "vs", &binary.vertex);
      save_stage(fixture_number, "fs", &binary.fragment);
   }
   char *after = nir_shader_as_str(fs, NULL);
   require(!strcmp(before, after) && !memcmp(&info, &fs->info, sizeof(info)),
           "caller NIR and all resource metadata remain exact after success/refusal");
   ralloc_free(before); ralloc_free(after);
   pvrgpu_pco_graphics_binary_finish(&binary); ralloc_free(vs); ralloc_free(fs);
}

static void
cube_array_descriptor_tests(void)
{
   const enum pipe_format formats[] = {
      PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R8G8B8A8_SRGB,
      PIPE_FORMAT_B8G8R8A8_UNORM, PIPE_FORMAT_R5G6B5_UNORM,
      PIPE_FORMAT_R10G10B10A2_UNORM, PIPE_FORMAT_R16G16B16A16_FLOAT,
      PIPE_FORMAT_R32G32B32A32_FLOAT,
   };
   const unsigned faces[] = {6, 12, 2046};
   const unsigned sizes[] = {1, 3, 8};
   for (unsigned f = 0; f < ARRAY_SIZE(formats); ++f) {
      for (unsigned n = 0; n < ARRAY_SIZE(faces); ++n) {
         for (unsigned d = 0; d < ARRAY_SIZE(sizes); ++d) {
            const unsigned size = sizes[d], mips = size == 8 ? 4 : 1;
            const unsigned stride = size * size * util_format_get_blocksize(formats[f]);
            uint32_t descriptor[20], expected[20];
            require(pvrgpu_pco_build_terrain_texture_descriptor(descriptor, formats[f],
               size, size, mips, stride * faces[n] * 2, 1, 0, mips > 1,
               2, 2, (mips - 1) * 64, faces[n], 2), "valid physical-face descriptor");
            descriptor[5] = 0x12345678; descriptor[6] = 0xabcdef01;
            memcpy(expected, descriptor, sizeof(expected));
            expected[2] = (expected[2] & ~UINT32_C(0x7ff0)) | ((faces[n] / 6 - 1) << 4);
            expected[4] = stride;
            require(pvrgpu_pco_set_cube_array_texture_layout(descriptor, formats[f], faces[n]),
                    "whole square uncompressed single-sample CubeArray accepted");
            require(!memcmp(expected, descriptor, sizeof(expected)),
                    "only depth and per-face stride change; all other words stay exact");
         }
      }
   }
   require(!pvrgpu_pco_set_cube_array_texture_layout(NULL, PIPE_FORMAT_R8G8B8A8_UNORM, 6),
           "null CubeArray descriptor refused");
   for (unsigned kind = 0; kind < 21; ++kind) {
      uint32_t descriptor[20], before[20];
      enum pipe_format format = PIPE_FORMAT_R8G8B8A8_UNORM;
      unsigned count = 12;
      require(pvrgpu_pco_build_terrain_texture_descriptor(descriptor, format,
         8, 8, 4, 8 * 8 * 4 * 24, 1, 1, 1, 2, 2, 192, count, 2), "negative descriptor seed");
      switch (kind) {
      case 0: count = 0; break;
      case 1: count = 5; break;
      case 2: count = 7; break;
      case 3: count = 2049; break;
      case 4: count = 2052; break;
      case 5: count = UINT_MAX; break;
      case 6: count = 18; break; /* Fits allocation, but disagrees with raw depth. */
      case 7: descriptor[1] ^= 1u << 2; break; /* Nonsquare image. */
      case 8: require(pvrgpu_pco_set_texture_sample_count(descriptor, 2), "MSAA test encoding"); break;
      case 9: descriptor[4] = 8 * 8 * 4 * 12 - 1; break;
      case 10: descriptor[4] = 0; break;
      case 11: descriptor[2] &= ~15u; break;
      case 12: descriptor[0] &= ~7u; break;
      case 13: descriptor[2] |= 1u << 16; break; /* Premature address relocation. */
      case 14: descriptor[3] |= 1u; break;
      case 15: format = PIPE_FORMAT_NONE; break;
      case 16: format = PIPE_FORMAT_COUNT; break;
      case 17: format = (enum pipe_format)-1; break;
      case 18: format = PIPE_FORMAT_ASTC_5x5; break;
      case 19: format = PIPE_FORMAT_R16G16B16A16_FLOAT; break; /* Typed format disagrees. */
      case 20:
         format = PIPE_FORMAT_R32G32B32A32_FLOAT;
         require(pvrgpu_pco_build_terrain_texture_descriptor(descriptor, format,
            16384, 16384, 1, UINT32_MAX, 0, 0, 0, 2, 2, 0, count, 2), "overflow metadata seed");
         break;
      }
      memcpy(before, descriptor, sizeof(before));
      require(!pvrgpu_pco_set_cube_array_texture_layout(descriptor, format, count),
              "malformed, unsupported or overflowing CubeArray descriptor refused");
      require(!memcmp(before, descriptor, sizeof(before)), "failed conversion has no partial writes");
   }
}

#ifdef PVRGPU_MANY_TEXTURES_DIRECT_HELPER
static void
cube_array_admission_tests(void)
{
   for (unsigned kind = 0; kind < 23; ++kind) {
      nir_shader *s = array_coordinate_fragment(1);
      nir_tex_instr *tex = NULL;
      nir_foreach_function_impl(impl, s)
         nir_foreach_block(block, impl)
            nir_foreach_instr(instr, block)
               if (instr->type == nir_instr_type_tex) tex = nir_instr_as_tex(instr);
      require(tex != NULL, "CubeArray admission seed");
      nir_builder b = nir_builder_at(nir_before_instr(&tex->instr));
      mesa_shader_stage stage = MESA_SHADER_FRAGMENT;
      switch (kind) {
      case 0: tex->op = nir_texop_txd; break;
      case 1: tex->op = nir_texop_txb; break;
      case 2: tex->op = nir_texop_txf; break;
      case 3: tex->op = nir_texop_tg4; break;
      case 4: tex->is_shadow = true; break;
      case 5: tex->is_sparse = true; break;
      case 6: tex->array_is_lowered_cube = true; break;
      case 7: tex->dest_type = nir_type_uint32; break;
      case 8: tex->def.bit_size = 16; break;
      case 9: tex->def.num_components = 3; break;
      case 10: tex->coord_components = 3; break;
      case 11: tex->texture_non_uniform = true; break;
      case 12: tex->sampler_non_uniform = true; break;
      case 13: tex->embedded_sampler = true; break;
      case 14: tex->sampler_index = 1; break;
      case 15: tex->texture_index = tex->sampler_index = 1; break;
      case 16: nir_tex_instr_add_src(tex, nir_tex_src_offset, nir_imm_ivec3(&b, 0, 0, 0)); break;
      case 17: nir_tex_instr_add_src(tex, nir_tex_src_texture_offset, nir_imm_int(&b, 0)); break;
      case 18: nir_tex_instr_add_src(tex, nir_tex_src_sampler_handle, nir_imm_int64(&b, 0)); break;
      case 19: nir_tex_instr_add_src(tex, nir_tex_src_lod, tex->src[1].src.ssa); break;
      case 20: nir_tex_instr_remove_src(tex, 1); break;
      case 21: stage = MESA_SHADER_VERTEX; break;
      case 22: stage = MESA_SHADER_COMPUTE; break;
      }
      require(!pvrgpu_native_fragment_cube_array(tex, stage, 1), "CubeArray unproven shape refused directly");
      ralloc_free(s); /* Never pass malformed fixture definitions to an optimizer. */
   }
   nir_shader *s = array_coordinate_fragment(1);
   nir_tex_instr *tex = NULL;
   nir_foreach_function_impl(impl, s)
      nir_foreach_block(block, impl)
         nir_foreach_instr(instr, block)
            if (instr->type == nir_instr_type_tex) tex = nir_instr_as_tex(instr);
   require(pvrgpu_native_fragment_cube_array(tex, MESA_SHADER_FRAGMENT, 1), "explicit LOD shape supported");
   nir_tex_instr_remove_src(tex, 1); tex->op = nir_texop_tex;
   require(pvrgpu_native_fragment_cube_array(tex, MESA_SHADER_FRAGMENT, 1), "implicit LOD shape supported");
   nir_builder b = nir_builder_at(nir_before_instr(&tex->instr));
   nir_tex_instr_remove_src(tex, 0); tex->op = nir_texop_txs;
   tex->coord_components = 0; tex->dest_type = nir_type_int32; tex->def.num_components = 3;
   nir_tex_instr_add_src(tex, nir_tex_src_lod, nir_imm_int(&b, 0));
   require(pvrgpu_native_fragment_cube_array(tex, MESA_SHADER_FRAGMENT, 1), "existing CubeArray size query preserved");
   ralloc_free(s);
}
#endif

#ifndef PVRGPU_MANY_TEXTURES_NO_MAIN
int main(void)
{
   require(PVRGPU_PCO_MAX_TEXTURES == PVRGPU_SYSTEMC_MAX_PCO_TEXTURES_PER_STAGE,
           "compiler and transport use the same descriptor cap");
   require(PVRGPU_PCO_MAX_TEXTURES == 16 &&
              16 * 20 <=
                 PVRGPU_SYSTEMC_MAX_PCO_GRAPHICS_SHARED_DWORDS_PER_STAGE &&
              17 * 20 <=
                 PVRGPU_SYSTEMC_MAX_PCO_GRAPHICS_SHARED_DWORDS_PER_STAGE,
           "descriptor cap carries the GLES3 sixteen-unit minimum");
   glsl_type_singleton_init_or_ref();
   cube_array_descriptor_tests();
#ifdef PVRGPU_MANY_TEXTURES_DIRECT_HELPER
   cube_array_admission_tests();
#endif
   char error[2048] = {0};
   struct pvrgpu_pco_compiler *compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
   require(compiler != NULL, error);
   for (unsigned kind = 0; kind < 4; ++kind)
      compile_array_coordinate_fixture(compiler, kind);
   compile_many_textures(compiler, 9, 56, 13, 240, NULL, 609);
   compile_many_textures(compiler, 12, 8, 1, 252, NULL, 612);
   compile_many_textures(compiler, 12, 12, 2, 256, NULL, 613);
   compile_many_textures(compiler, 16, 8, 1, 332, NULL, 616);
   compile_many_textures(compiler, 17, 8, 1, 0, "texture count is unsupported", 0);
   /* These formerly overflowing 256-DWORD cases now fit the 384-DWORD
    * transport without packing. The current SH383/384/fallback edges are
    * covered by packed_uniforms. */
   compile_many_textures(compiler, 12, 16, 3, 260, NULL, 614);
   compile_many_textures(compiler, 12, 56, 13, 300, NULL, 615);
   pvrgpu_pco_compiler_destroy(compiler);
   glsl_type_singleton_decref();
   printf("many texture compiler PASS: %u checks\n", checks);
   return 0;
}
#endif
