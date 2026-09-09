/* SPDX-License-Identifier: MIT */
/* Exercise the actual context snapshot, not a second layout implementation. */
#include "../src/gallium/drivers/pvrgpu/pvrgpu_context.c"
#include "compiler/nir/nir_builder.h"

static unsigned snapshot_checks;
#define CHECK(test) do { ++snapshot_checks; if (!(test)) { \
   fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #test); exit(1); \
} } while (0)

static void
test_view(enum pipe_texture_target target, enum pipe_format format,
          unsigned width, unsigned height, unsigned depth,
          unsigned first, unsigned last, unsigned first_layer, unsigned layers)
{
   struct pvrgpu_resource resource = {0};
   resource.base.target = target;
   resource.base.format = format;
   resource.base.width0 = width;
   resource.base.height0 = height;
   resource.base.depth0 = target == PIPE_TEXTURE_3D ? depth : 1;
   resource.base.array_size = target == PIPE_TEXTURE_CUBE ? 6 :
      (target == PIPE_TEXTURE_2D_ARRAY || target == PIPE_TEXTURE_CUBE_ARRAY) ? depth : 1;
   resource.base.last_level = 6;
   resource.level_count = 7;
   for (unsigned level = 0; level < resource.level_count; ++level) {
      const unsigned slices = target == PIPE_TEXTURE_3D ? u_minify(depth, level) :
         resource.base.array_size;
      resource.level_offsets[level] = resource.size;
      resource.level_strides[level] = util_format_get_stride(format, u_minify(width, level));
      resource.level_layer_strides[level] = util_format_get_2d_size(format,
         resource.level_strides[level], u_minify(height, level));
      resource.size += slices * resource.level_layer_strides[level];
   }
   resource.data = malloc(resource.size);
   CHECK(resource.data != NULL);
   for (unsigned level = 0; level < resource.level_count; ++level) {
      const unsigned slices = target == PIPE_TEXTURE_3D ? u_minify(depth, level) :
         resource.base.array_size;
      for (unsigned layer = 0; layer < slices; ++layer) {
         uint8_t *at = resource.data + resource.level_offsets[level] +
            layer * resource.level_layer_strides[level];
         memset(at, 17 + 9 * level + layer, resource.level_layer_strides[level]);
      }
   }
   struct pipe_sampler_view view = {0};
   view.texture = &resource.base;
   view.target = target;
   view.format = format;
   view.u.tex.first_level = first;
   view.u.tex.last_level = last;
   view.u.tex.first_layer = first_layer;
   view.u.tex.last_layer = first_layer + layers - 1;
   view.swizzle_r = PIPE_SWIZZLE_X;
   view.swizzle_g = util_format_is_depth_or_stencil(format) ? PIPE_SWIZZLE_0 : PIPE_SWIZZLE_Y;
   view.swizzle_b = util_format_is_depth_or_stencil(format) ? PIPE_SWIZZLE_0 : PIPE_SWIZZLE_Z;
   view.swizzle_a = util_format_is_depth_or_stencil(format) ? PIPE_SWIZZLE_1 : PIPE_SWIZZLE_W;
   struct pvrgpu_sampler_state sampler = {0};
   sampler.state.min_mip_filter = PIPE_TEX_MIPFILTER_NEAREST;
   sampler.state.wrap_s = sampler.state.wrap_t = sampler.state.wrap_r = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
   struct pvrgpu_context ctx = {0};
   const mesa_shader_stage stage = target == PIPE_TEXTURE_CUBE_ARRAY ?
      MESA_SHADER_FRAGMENT : MESA_SHADER_VERTEX;
   ctx.sampler_views[stage][0] = &view;
   ctx.samplers[stage][0] = &sampler;
   struct pvrgpu_systemc_pco_sequence_texture captured = {0};
   uint8_t *bytes = NULL;
   const char *reason = NULL;
   CHECK(pvrgpu_capture_generic_sequence_texture(&ctx, stage, 0, 0,
                                                  &captured, &bytes, &reason));
   CHECK(bytes != resource.data && captured.bytes == bytes && captured.mip_count == last-first+1);
   CHECK(captured.layers == (target == PIPE_TEXTURE_3D ? u_minify(depth, first) :
         (target == PIPE_TEXTURE_2D_ARRAY || target == PIPE_TEXTURE_CUBE_ARRAY) ? layers :
         target == PIPE_TEXTURE_CUBE ? 6 : 1));
   if (target == PIPE_TEXTURE_CUBE_ARRAY)
      CHECK(captured.texture_kind == 4 && captured.stage == PVRGPU_SYSTEMC_PCO_SHADER_STAGE_FRAGMENT);
   size_t offset = 0;
   for (unsigned level = 0; level < captured.mip_count; ++level) {
      const unsigned backing_level = first + level;
      const unsigned slices = target == PIPE_TEXTURE_3D ? u_minify(captured.layers, level) : captured.layers;
      CHECK(captured.mip[level].width == u_minify(width, backing_level));
      CHECK(captured.mip[level].height == u_minify(height, backing_level));
      CHECK(captured.mip[level].offset == offset);
      for (unsigned layer = 0; layer < slices; ++layer) {
         const uint8_t *original = resource.data + resource.level_offsets[backing_level] +
            (first_layer + layer) * resource.level_layer_strides[backing_level];
         const uint8_t *copied = bytes + offset + layer *
            captured.mip[level].row_pitch * captured.mip[level].height;
         if (format == PIPE_FORMAT_Z16_UNORM) {
            float expected, actual;
            util_format_unpack_z_float(format, &expected, original, 1);
            memcpy(&actual, copied, sizeof(actual));
            CHECK(actual == expected);
            const uint32_t *channels = (const uint32_t *)copied;
            CHECK(channels[1] == 0 && channels[2] == 0 && channels[3] == 0x3f800000U);
         } else {
            CHECK(memcmp(original, copied, resource.level_layer_strides[backing_level]) == 0);
         }
      }
      offset += captured.mip[level].row_pitch * captured.mip[level].height * slices;
   }
   CHECK(offset == captured.bytes_size);
   if (first == 1 && width == 100 && height == 31)
      CHECK(u_minify(captured.mip[0].width, 3) == 6 && u_minify(captured.mip[0].height, 3) == 1);
   const uint8_t retained = bytes[0];
   memset(resource.data, 0, resource.size);
   CHECK(bytes[0] == retained);
   free(bytes);
   bytes = NULL;
   if (target == PIPE_TEXTURE_CUBE_ARRAY) {
      const unsigned final_slot = PVRGPU_PCO_MAX_TEXTURES - 1;
      ctx.sampler_views[stage][final_slot] = &view;
      ctx.samplers[stage][final_slot] = &sampler;
      CHECK(pvrgpu_capture_generic_sequence_texture(&ctx, stage, final_slot,
               final_slot, &captured, &bytes, &reason));
      CHECK(captured.descriptor_set == final_slot && captured.texture_kind == 4 &&
            captured.layers == layers && bytes != resource.data);
      free(bytes); bytes = NULL;
      ctx.samplers[stage][final_slot] = NULL;
      CHECK(!pvrgpu_capture_generic_sequence_texture(&ctx, stage, final_slot,
               final_slot, &captured, &bytes, &reason));
      CHECK(bytes == NULL && !strcmp(reason, "binding"));
      ctx.samplers[stage][final_slot] = &sampler;
      ctx.sampler_views[stage][final_slot] = NULL;
      CHECK(!pvrgpu_capture_generic_sequence_texture(&ctx, stage, final_slot,
               final_slot, &captured, &bytes, &reason));
      CHECK(bytes == NULL && !strcmp(reason, "binding"));
      CHECK(!pvrgpu_capture_generic_sequence_texture(&ctx, stage, final_slot + 1,
               final_slot + 1, &captured, &bytes, &reason));
      CHECK(bytes == NULL && !strcmp(reason, "arguments"));
   }
   view.u.tex.first_level = last + 1;
   CHECK(!pvrgpu_capture_generic_sequence_texture(&ctx, stage, 0, 0,
                                                   &captured, &bytes, &reason));
   CHECK(bytes == NULL && !strcmp(reason, "view_range"));
   view.u.tex.first_level = first;
   resource.level_layer_strides[first]++;
   CHECK(!pvrgpu_capture_generic_sequence_texture(&ctx, stage, 0, 0,
                                                   &captured, &bytes, &reason));
   CHECK(bytes == NULL && !strcmp(reason, "mip_layout"));
   resource.level_layer_strides[first]--;
   if (target == PIPE_TEXTURE_CUBE_ARRAY) {
#define REJECT_CUBE(expected) do { \
      CHECK(!pvrgpu_capture_generic_sequence_texture(&ctx, stage, 0, 0, \
                                                     &captured, &bytes, &reason)); \
      CHECK(bytes == NULL && !strcmp(reason, expected)); \
   } while (0)
      sampler.state.compare_mode = PIPE_TEX_COMPARE_R_TO_TEXTURE;
      REJECT_CUBE("cube_array_layout_or_sampler");
      sampler.state.compare_mode = PIPE_TEX_COMPARE_NONE;
      sampler.state.unnormalized_coords = true;
      REJECT_CUBE("sampler_state");
      sampler.state.unnormalized_coords = false;
      resource.base.nr_samples = 1;
      REJECT_CUBE("sample_layout");
      resource.base.nr_samples = 0;
      resource.base.nr_storage_samples = 2;
      REJECT_CUBE("cube_array_layout_or_sampler");
      resource.base.nr_storage_samples = 0;
      resource.base.width0++;
      REJECT_CUBE("cube_array_layout_or_sampler");
      resource.base.width0--;
      view.u.tex.first_layer++;
      REJECT_CUBE("cube_array_layout_or_sampler");
      view.u.tex.first_layer--;
      view.u.tex.last_layer--;
      REJECT_CUBE("cube_array_layout_or_sampler");
      view.u.tex.last_layer++;
      view.u.tex.last_layer = resource.base.array_size;
      REJECT_CUBE("view_range");
      view.u.tex.last_layer = first_layer + layers - 1;
      view.format = PIPE_FORMAT_Z16_UNORM;
      /* Depth is supported, but reinterpreting this colour allocation with
       * a different row/face stride must still fail before copying. */
      REJECT_CUBE("mip_layout");
      view.format = PIPE_FORMAT_ASTC_4x4;
      REJECT_CUBE("cube_array_layout_or_sampler");
      view.format = format;
      for (unsigned s = 0; s < MESA_SHADER_STAGES; ++s) {
         if (s == MESA_SHADER_FRAGMENT) continue;
         ctx.sampler_views[s][0] = &view;
         ctx.samplers[s][0] = &sampler;
         CHECK(!pvrgpu_capture_generic_sequence_texture(&ctx, s, 0, 0,
                                                        &captured, &bytes, &reason));
         CHECK(bytes == NULL);
      }
#undef REJECT_CUBE
   }
   free(resource.data);
}

static void test_cube_array_depth_views(void)
{
   const enum pipe_format formats[] = {
      PIPE_FORMAT_Z16_UNORM, PIPE_FORMAT_Z24X8_UNORM,
      PIPE_FORMAT_Z24_UNORM_S8_UINT, PIPE_FORMAT_Z32_UNORM,
      PIPE_FORMAT_Z32_FLOAT, PIPE_FORMAT_Z32_FLOAT_S8X24_UINT,
   };
   const unsigned swizzles[][4] = {
      {PIPE_SWIZZLE_X, PIPE_SWIZZLE_0, PIPE_SWIZZLE_0, PIPE_SWIZZLE_1},
      {PIPE_SWIZZLE_X, PIPE_SWIZZLE_X, PIPE_SWIZZLE_X, PIPE_SWIZZLE_1},
      {PIPE_SWIZZLE_1, PIPE_SWIZZLE_X, PIPE_SWIZZLE_0, PIPE_SWIZZLE_X},
      {PIPE_SWIZZLE_W, PIPE_SWIZZLE_Z, PIPE_SWIZZLE_X, PIPE_SWIZZLE_Y},
   };
   for (unsigned f = 0; f < ARRAY_SIZE(formats); ++f) {
      struct pvrgpu_resource resource = {0};
      resource.base.target = PIPE_TEXTURE_CUBE_ARRAY;
      resource.base.format = formats[f];
      resource.base.width0 = resource.base.height0 = 8;
      resource.base.depth0 = 1;
      resource.base.array_size = 18;
      resource.base.last_level = 3;
      resource.level_count = 4;
      const unsigned bpp = util_format_get_blocksize(formats[f]);
      for (unsigned level = 0; level < 4; ++level) {
         const unsigned size = 8U >> level;
         resource.level_offsets[level] = resource.size;
         resource.level_strides[level] = size * bpp;
         resource.level_layer_strides[level] = size * size * bpp;
         resource.size += 18U * size * size * bpp;
      }
      resource.data = malloc(resource.size);
      uint8_t *original = malloc(resource.size);
      CHECK(resource.data && original);
      memset(resource.data, 0xa5, resource.size); /* Includes stencil/padding. */
      for (unsigned level = 0; level < 4; ++level) {
         const unsigned size = 8U >> level;
         for (unsigned face = 0; face < 18; ++face)
            for (unsigned y = 0; y < size; ++y)
               for (unsigned x = 0; x < size; ++x) {
                  const float depth = ((level * 7 + face * 11 + y * 3 + x) % 32) / 31.0f;
                  uint8_t *at = resource.data + resource.level_offsets[level] +
                     face * resource.level_layer_strides[level] +
                     y * resource.level_strides[level] + x * bpp;
                  util_format_pack_z_float(formats[f], at, &depth, 1);
               }
      }
      memcpy(original, resource.data, resource.size);
      struct pipe_sampler_view view = {0};
      view.texture = &resource.base;
      view.target = PIPE_TEXTURE_CUBE_ARRAY;
      view.format = formats[f];
      view.u.tex.last_level = 3;
      struct pvrgpu_sampler_state sampler = {0};
      sampler.state.min_mip_filter = PIPE_TEX_MIPFILTER_NEAREST;
      sampler.state.wrap_s = sampler.state.wrap_t = sampler.state.wrap_r = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
      struct pvrgpu_context ctx = {0};
      ctx.sampler_views[MESA_SHADER_FRAGMENT][4] = &view;
      ctx.samplers[MESA_SHADER_FRAGMENT][4] = &sampler;
      for (unsigned first = 0; first <= 1; ++first)
         for (unsigned first_face = 0; first_face <= 6; first_face += 6)
            for (unsigned s = 0; s < ARRAY_SIZE(swizzles); ++s) {
               view.u.tex.first_level = first;
               view.u.tex.first_layer = first_face;
               view.u.tex.last_layer = first_face + 11;
               view.swizzle_r = swizzles[s][0]; view.swizzle_g = swizzles[s][1];
               view.swizzle_b = swizzles[s][2]; view.swizzle_a = swizzles[s][3];
               struct pvrgpu_systemc_pco_sequence_texture captured = {0};
               uint8_t *bytes = NULL;
               const char *reason = NULL;
               CHECK(pvrgpu_capture_generic_sequence_texture(&ctx, MESA_SHADER_FRAGMENT,
                  4, 4, &captured, &bytes, &reason));
               CHECK(bytes && bytes != resource.data && captured.bytes == bytes);
               CHECK(!strcmp(captured.format, "PIPE_FORMAT_R32G32B32A32_FLOAT"));
               CHECK(captured.texture_kind == 4 && captured.layers == 12 &&
                     captured.mip_count == 4 - first && captured.sample_count == 1);
               size_t offset = 0;
               for (unsigned level = first; level < 4; ++level) {
                  const unsigned size = 8U >> level;
                  const unsigned mip = level - first;
                  CHECK(captured.mip[mip].width == size && captured.mip[mip].height == size);
                  CHECK(captured.mip[mip].row_pitch == size * 16 && captured.mip[mip].offset == offset);
                  for (unsigned face = first_face; face < first_face + 12; ++face)
                     for (unsigned y = 0; y < size; ++y)
                        for (unsigned x = 0; x < size; ++x) {
                           float depth;
                           util_format_unpack_z_float(formats[f], &depth,
                              original + resource.level_offsets[level] +
                              face * resource.level_layer_strides[level] +
                              y * resource.level_strides[level] + x * bpp, 1);
                           uint32_t rgba[4] = {0, 0, 0, UINT32_C(0x3f800000)};
                           memcpy(&rgba[0], &depth, 4);
                           for (unsigned c = 0; c < 4; ++c) {
                              uint32_t actual;
                              memcpy(&actual, bytes + offset, 4);
                              const unsigned swizzle = swizzles[s][c];
                              const uint32_t expected = swizzle < 4 ? rgba[swizzle] :
                                 swizzle == PIPE_SWIZZLE_1 ? UINT32_C(0x3f800000) : 0;
                              CHECK(actual == expected);
                              offset += 4;
                           }
                        }
               }
               CHECK(offset == captured.bytes_size && offset == captured.declared_bytes_size);
               CHECK(memcmp(original, resource.data, resource.size) == 0);
               uint32_t descriptor[20];
               CHECK(pvrgpu_pco_build_terrain_texture_descriptor(descriptor,
                  PIPE_FORMAT_R32G32B32A32_FLOAT, 8U >> first, 8U >> first,
                  4 - first, offset, 0, 0, 0, 0, 0, 0, 12, 0));
               CHECK(pvrgpu_pco_set_cube_array_texture_layout(descriptor,
                  PIPE_FORMAT_R32G32B32A32_FLOAT, 12));
               CHECK(descriptor[4] == (8U >> first) * (8U >> first) * 16U);
               pvrgpu_set_generic_texture_compare_metadata(&view, &sampler.state, descriptor);
               CHECK(descriptor[7] == 0 && descriptor[12] == 0);
               free(bytes); bytes = NULL;
               sampler.state.compare_mode = PIPE_TEX_COMPARE_R_TO_TEXTURE;
               CHECK(!pvrgpu_capture_generic_sequence_texture(&ctx, MESA_SHADER_FRAGMENT,
                  4, 4, &captured, &bytes, &reason));
               CHECK(bytes == NULL && !strcmp(reason, "cube_array_layout_or_sampler"));
               sampler.state.compare_mode = PIPE_TEX_COMPARE_NONE;
               view.format = PIPE_FORMAT_S8_UINT;
               CHECK(!pvrgpu_capture_generic_sequence_texture(&ctx, MESA_SHADER_FRAGMENT,
                  4, 4, &captured, &bytes, &reason));
               CHECK(bytes == NULL && !strcmp(reason, "cube_array_layout_or_sampler"));
               view.format = formats[f];
            }
      free(original);
      free(resource.data);
   }
}

static void test_shadow_metadata(void)
{
   const enum pipe_format formats[] = {PIPE_FORMAT_Z16_UNORM,
      PIPE_FORMAT_Z24_UNORM_S8_UINT, PIPE_FORMAT_Z32_FLOAT};
   const float depth[4] = {0.f, 1.f, 0.25f, 0.75f};
   for (unsigned f = 0; f < ARRAY_SIZE(formats); ++f) {
      struct pvrgpu_resource resource = {0};
      resource.base.target = PIPE_TEXTURE_2D;
      resource.base.format = formats[f];
      resource.base.width0 = 4;
      resource.base.height0 = resource.base.depth0 = resource.base.array_size = 1;
      resource.level_count = 1;
      resource.level_strides[0] = util_format_get_stride(formats[f], 4);
      resource.level_layer_strides[0] = resource.size = resource.level_strides[0];
      resource.data = calloc(1, resource.size);
      CHECK(resource.data != NULL);
      util_format_pack_z_float(formats[f], resource.data, depth, 4);
      struct pipe_sampler_view view = {0};
      view.texture = &resource.base;
      view.format = formats[f];
      view.target = PIPE_TEXTURE_2D;
      view.swizzle_r = PIPE_SWIZZLE_X;
      view.swizzle_g = view.swizzle_b = PIPE_SWIZZLE_0;
      view.swizzle_a = PIPE_SWIZZLE_1;
      struct pvrgpu_sampler_state sampler = {0};
      sampler.state.compare_mode = PIPE_TEX_COMPARE_R_TO_TEXTURE;
      sampler.state.wrap_s = sampler.state.wrap_t = sampler.state.wrap_r = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
      struct pvrgpu_context ctx = {0};
      for (unsigned stage_index = 0; stage_index < 2; ++stage_index) {
         mesa_shader_stage stage = stage_index ? MESA_SHADER_FRAGMENT : MESA_SHADER_VERTEX;
         ctx.sampler_views[stage][0] = &view;
         ctx.samplers[stage][0] = &sampler;
         for (unsigned compare = PIPE_FUNC_NEVER; compare <= PIPE_FUNC_ALWAYS; ++compare) {
            sampler.state.compare_func = compare;
            for (unsigned mip_filter = PIPE_TEX_MIPFILTER_NEAREST;
                 mip_filter <= PIPE_TEX_MIPFILTER_NONE; mip_filter += 2) {
               sampler.state.min_mip_filter = mip_filter;
               struct pvrgpu_systemc_pco_sequence_texture captured = {0};
               uint8_t *bytes = NULL;
               const char *reason = NULL;
               CHECK(pvrgpu_capture_generic_sequence_texture(&ctx, stage, 0, 0,
                     &captured, &bytes, &reason));
               float first, second;
               if (formats[f] == PIPE_FORMAT_Z24_UNORM_S8_UINT) {
                  util_format_unpack_z_float(formats[f], &first, bytes, 1);
                  util_format_unpack_z_float(formats[f], &second, bytes + 4, 1);
               } else {
                  memcpy(&first, bytes, 4);
                  memcpy(&second, bytes + 16, 4);
               }
               CHECK(first == 0.f && second == 1.f);
               free(bytes);
               uint32_t descriptor[20] = {0};
               pvrgpu_set_generic_texture_compare_metadata(&view, &sampler.state, descriptor);
               CHECK(descriptor[12] == compare);
               CHECK(descriptor[7] == (formats[f] == PIPE_FORMAT_Z32_FLOAT ? 0 : 0x100));
            }
         }
         for (unsigned filter = 0; filter < 3; ++filter) {
            struct pvrgpu_systemc_pco_sequence_texture captured = {0};
            uint8_t *bytes = NULL;
            const char *reason = NULL;
            sampler.state.min_img_filter = filter == 0 ? PIPE_TEX_FILTER_LINEAR : PIPE_TEX_FILTER_NEAREST;
            sampler.state.mag_img_filter = filter == 1 ? PIPE_TEX_FILTER_LINEAR : PIPE_TEX_FILTER_NEAREST;
            sampler.state.min_mip_filter = filter == 2 ? PIPE_TEX_MIPFILTER_LINEAR : PIPE_TEX_MIPFILTER_NEAREST;
            CHECK(!pvrgpu_capture_generic_sequence_texture(&ctx, stage, 0, 0,
                  &captured, &bytes, &reason));
            CHECK(bytes == NULL && !strcmp(reason, "shadow_sampler_state_requires_nearest_2d"));
         }
         sampler.state.min_img_filter = sampler.state.mag_img_filter = PIPE_TEX_FILTER_NEAREST;
         sampler.state.min_mip_filter = PIPE_TEX_MIPFILTER_NEAREST;
      }
      free(resource.data);
   }
}

static nir_shader *shadow_gather_nir(bool array, bool mixed)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, NULL,
      "snapshot_shadow_gather");
   for (unsigned i = 0; i < (mixed ? 2U : 1U); ++i) {
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);
      tex->op = i ? nir_texop_tex : nir_texop_tg4;
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
      tex->is_array = array;
      tex->is_shadow = true;
      tex->is_new_style_shadow = true;
      tex->dest_type = nir_type_float32;
      tex->coord_components = array ? 3 : 2;
      tex->texture_index = tex->sampler_index = 2;
      tex->src[0].src_type = nir_tex_src_coord;
      tex->src[0].src = nir_src_for_ssa(array ? nir_imm_vec3(&b, .5f, .5f, 1.f) :
                                               nir_imm_vec2(&b, .5f, .5f));
      tex->src[1].src_type = nir_tex_src_comparator;
      tex->src[1].src = nir_src_for_ssa(nir_imm_float(&b, .5f));
      nir_def_init(&tex->instr, &tex->def, i ? 1 : 4, 32);
      nir_builder_instr_insert(&b, &tex->instr);
   }
   return b.shader;
}

static void test_shadow_gather_snapshot(void)
{
   const enum pipe_format formats[] = {PIPE_FORMAT_Z16_UNORM,
      PIPE_FORMAT_Z24_UNORM_S8_UINT, PIPE_FORMAT_Z32_FLOAT};
   for (unsigned array = 0; array < 2; ++array) {
      struct pvrgpu_shader_state fs = {.nir = shadow_gather_nir(array, false)};
      struct pvrgpu_shader_state mixed = {.nir = shadow_gather_nir(array, true)};
      for (unsigned f = 0; f < ARRAY_SIZE(formats); ++f) {
         struct pvrgpu_resource resource = {0};
         resource.base.target = array ? PIPE_TEXTURE_2D_ARRAY : PIPE_TEXTURE_2D;
         resource.base.format = formats[f];
         resource.base.width0 = 4;
         resource.base.height0 = resource.base.depth0 = 1;
         resource.base.array_size = array ? 3 : 1;
         resource.level_count = 1;
         resource.level_strides[0] = util_format_get_stride(formats[f], 4);
         resource.level_layer_strides[0] = resource.level_strides[0];
         resource.size = resource.level_strides[0] * resource.base.array_size;
         resource.data = calloc(1, resource.size);
         CHECK(resource.data != NULL);
         for (unsigned layer = 0; layer < resource.base.array_size; ++layer) {
            float depth[4];
            for (unsigned x = 0; x < 4; ++x)
               depth[x] = (float)((x + layer) % 4) / 3.f;
            util_format_pack_z_float(formats[f], resource.data +
               layer * resource.level_layer_strides[0], depth, 4);
         }
         struct pipe_sampler_view view = {0};
         view.texture = &resource.base;
         view.target = resource.base.target;
         view.format = formats[f];
         view.u.tex.last_layer = resource.base.array_size - 1;
         view.swizzle_r = PIPE_SWIZZLE_X;
         view.swizzle_g = view.swizzle_b = PIPE_SWIZZLE_0;
         view.swizzle_a = PIPE_SWIZZLE_1;
         struct pvrgpu_sampler_state sampler = {0};
         sampler.state.compare_mode = PIPE_TEX_COMPARE_R_TO_TEXTURE;
         sampler.state.wrap_s = sampler.state.wrap_t = sampler.state.wrap_r = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
         struct pvrgpu_context ctx = {.fs = &fs};
         ctx.sampler_views[MESA_SHADER_FRAGMENT][2] = &view;
         ctx.samplers[MESA_SHADER_FRAGMENT][2] = &sampler;
         for (unsigned filters = 0; filters < 4; ++filters) {
            sampler.state.min_img_filter = filters & 1;
            sampler.state.mag_img_filter = (filters >> 1) & 1;
            for (unsigned mip = PIPE_TEX_MIPFILTER_NEAREST;
                 mip <= PIPE_TEX_MIPFILTER_NONE; ++mip) {
               sampler.state.min_mip_filter = mip;
               for (unsigned compare = PIPE_FUNC_NEVER; compare <= PIPE_FUNC_ALWAYS; ++compare) {
                  sampler.state.compare_func = compare;
                  struct pvrgpu_systemc_pco_sequence_texture captured = {0};
                  uint8_t *bytes = NULL;
                  const char *reason = NULL;
                  CHECK(pvrgpu_shadow_gather_sampler_supported(&ctx,
                     MESA_SHADER_FRAGMENT, 2, &view, &sampler.state));
                  CHECK(pvrgpu_capture_generic_sequence_texture(&ctx,
                     MESA_SHADER_FRAGMENT, 2, 0, &captured, &bytes, &reason));
                  CHECK(bytes != resource.data && captured.bytes == bytes &&
                        captured.layers == resource.base.array_size && captured.mip_count == 1);
                  for (unsigned layer = 0; layer < captured.layers; ++layer) {
                     float original[4], copied[4];
                     util_format_unpack_z_float(formats[f], original, resource.data +
                        layer * resource.level_layer_strides[0], 4);
                     const uint8_t *at = bytes + layer * captured.mip[0].row_pitch;
                     if (formats[f] == PIPE_FORMAT_Z24_UNORM_S8_UINT)
                        util_format_unpack_z_float(formats[f], copied, at, 4);
                     else
                        for (unsigned x = 0; x < 4; ++x)
                           memcpy(&copied[x], at + x * 16, 4);
                     CHECK(!memcmp(original, copied, sizeof(original)));
                  }
                  uint32_t descriptor[20] = {0};
                  pvrgpu_set_generic_texture_compare_metadata(&view, &sampler.state, descriptor);
                  CHECK(descriptor[12] == compare);
                  CHECK(descriptor[7] == (formats[f] == PIPE_FORMAT_Z32_FLOAT ? 0 : 0x100));
                  const uint8_t retained = bytes[0];
                  resource.data[0] ^= 1;
                  CHECK(bytes[0] == retained);
                  resource.data[0] ^= 1;
                  free(bytes);
               }
            }
         }
         /* Linear state must remain refused unless the actual FS proves this
          * exact sampler has only supported gathers. No blanket promotion. */
         sampler.state.min_img_filter = sampler.state.mag_img_filter = PIPE_TEX_FILTER_LINEAR;
         for (unsigned negative = 0; negative < 15; ++negative) {
            sampler.state.min_img_filter = sampler.state.mag_img_filter =
               negative >= 8 ? PIPE_TEX_FILTER_NEAREST : PIPE_TEX_FILTER_LINEAR;
            ctx.fs = negative == 0 ? NULL : (negative == 1 || negative == 11) ? &mixed : &fs;
            sampler.state.wrap_s = (negative == 2 || negative == 10) ? PIPE_TEX_WRAP_REPEAT : PIPE_TEX_WRAP_CLAMP_TO_EDGE;
            sampler.state.wrap_t = negative == 3 ? PIPE_TEX_WRAP_REPEAT : PIPE_TEX_WRAP_CLAMP_TO_EDGE;
            sampler.state.unnormalized_coords = negative == 4;
            sampler.state.max_anisotropy = negative == 5 ? 2 : 0;
            sampler.state.compare_mode = negative == 13 ? PIPE_TEX_COMPARE_NONE :
                                                                        PIPE_TEX_COMPARE_R_TO_TEXTURE;
            resource.base.nr_samples = negative == 14 ? 1 : 0;
            view.swizzle_r = negative == 6 ? PIPE_SWIZZLE_Y :
               negative == 8 ? PIPE_SWIZZLE_0 : negative == 9 ? PIPE_SWIZZLE_1 :
               negative == 12 ? PIPE_SWIZZLE_Y : PIPE_SWIZZLE_X;
            /* Both objects are still bound; only the shader's slot differs. */
            const unsigned slot = negative == 7 ? 3 : 2;
            ctx.sampler_views[MESA_SHADER_FRAGMENT][slot] = &view;
            ctx.samplers[MESA_SHADER_FRAGMENT][slot] = &sampler;
            struct pvrgpu_systemc_pco_sequence_texture captured = {0};
            uint8_t *bytes = NULL;
            const char *reason = NULL;
            CHECK(!pvrgpu_shadow_gather_sampler_supported(&ctx,
               MESA_SHADER_FRAGMENT, slot, &view, &sampler.state));
            CHECK(!pvrgpu_capture_generic_sequence_texture(&ctx,
               MESA_SHADER_FRAGMENT, slot, 0, &captured, &bytes, &reason));
            CHECK(bytes == NULL && captured.bytes == NULL && reason != NULL);
         }
         ctx.fs = &fs;
         resource.base.nr_samples = 0;
         sampler.state.min_img_filter = sampler.state.mag_img_filter = PIPE_TEX_FILTER_LINEAR;
         view.swizzle_r = PIPE_SWIZZLE_X;
         CHECK(!pvrgpu_shadow_gather_sampler_supported(&ctx,
            MESA_SHADER_VERTEX, 2, &view, &sampler.state));
         view.u.tex.last_level = 1;
         CHECK(!pvrgpu_shadow_gather_sampler_supported(&ctx,
            MESA_SHADER_FRAGMENT, 2, &view, &sampler.state));
         view.u.tex.last_level = 0;
         resource.base.nr_samples = 1;
         CHECK(!pvrgpu_shadow_gather_sampler_supported(&ctx,
            MESA_SHADER_FRAGMENT, 2, &view, &sampler.state));
         resource.base.nr_samples = 0;
         /* A valid two-mip backing image distinguishes gather's mandatory
          * state check from an earlier malformed-layout rejection. */
         resource.base.last_level = 1;
         resource.level_count = 2;
         resource.level_offsets[1] = resource.size;
         resource.level_strides[1] = util_format_get_stride(formats[f], 2);
         resource.level_layer_strides[1] = resource.level_strides[1];
         resource.size += resource.level_strides[1] * resource.base.array_size;
         uint8_t *expanded = realloc(resource.data, resource.size);
         CHECK(expanded != NULL);
         resource.data = expanded;
         for (unsigned layer = 0; layer < resource.base.array_size; ++layer) {
            const float depth[2] = {layer * .25f, 1.f - layer * .25f};
            util_format_pack_z_float(formats[f], resource.data + resource.level_offsets[1] +
               layer * resource.level_layer_strides[1], depth, 2);
         }
         sampler.state.min_img_filter = sampler.state.mag_img_filter = PIPE_TEX_FILTER_NEAREST;
         view.u.tex.last_level = 1;
         struct pvrgpu_systemc_pco_sequence_texture rebased = {0};
         uint8_t *bytes = NULL;
         const char *reason = NULL;
         CHECK(!pvrgpu_capture_generic_sequence_texture(&ctx, MESA_SHADER_FRAGMENT,
            2, 0, &rebased, &bytes, &reason));
         CHECK(bytes == NULL && !strcmp(reason, "shadow_gather_sampler_state"));
         /* One exposed nonzero mip and a nonzero array view base remain
          * valid: snapshot coordinates/layers must be relative to the view. */
         view.u.tex.first_level = 1;
         view.u.tex.first_layer = array ? 1 : 0;
         CHECK(pvrgpu_capture_generic_sequence_texture(&ctx, MESA_SHADER_FRAGMENT,
            2, 0, &rebased, &bytes, &reason));
         CHECK(bytes != resource.data && rebased.layers == (array ? 2U : 1U) &&
               rebased.mip_count == 1 && rebased.mip[0].width == 2 &&
               rebased.mip[0].height == 1 && rebased.mip[0].offset == 0);
         for (unsigned layer = 0; layer < rebased.layers; ++layer) {
            float original[2], copied[2];
            util_format_unpack_z_float(formats[f], original, resource.data + resource.level_offsets[1] +
               (view.u.tex.first_layer + layer) * resource.level_layer_strides[1], 2);
            const uint8_t *at = bytes + layer * rebased.mip[0].row_pitch;
            if (formats[f] == PIPE_FORMAT_Z24_UNORM_S8_UINT)
               util_format_unpack_z_float(formats[f], copied, at, 2);
            else
               for (unsigned x = 0; x < 2; ++x)
                  memcpy(&copied[x], at + x * 16, 4);
            CHECK(!memcmp(original, copied, sizeof(original)));
         }
         free(bytes);
         bytes = NULL;
         if (array) {
            for (unsigned bad = 0; bad < 2; ++bad) {
               view.u.tex.first_layer = bad ? 1 : 3;
               view.u.tex.last_layer = bad ? 3 : 2;
               CHECK(!pvrgpu_capture_generic_sequence_texture(&ctx, MESA_SHADER_FRAGMENT,
                  2, 0, &rebased, &bytes, &reason));
               CHECK(bytes == NULL && !strcmp(reason, "view_range"));
            }
         }
         free(resource.data);
      }
      ralloc_free(fs.nir);
      ralloc_free(mixed.nir);
   }
}

static void
test_mrt_initial_snapshot(unsigned samples)
{
   struct pvrgpu_resource resources[4] = {0};
   struct pvrgpu_context ctx = {0};
   struct pvrgpu_array_primitive_draw recorded = {0};
   const unsigned width = 3, height = 2;
   const size_t row_bytes = width * samples * 4;
   const size_t target_bytes = row_bytes * height;
   ctx.framebuffer.width = width;
   ctx.framebuffer.height = height;
   ctx.framebuffer.nr_cbufs = 4;
   recorded.command.format = PVRGPU_DRIVER_COMMAND_FORMAT_RGBA8;
   recorded.command.render_target_count = 4;
   recorded.command.raster_samples = samples;
   for (unsigned target = 0; target < 4; ++target) {
      struct pvrgpu_resource *resource = &resources[target];
      resource->base.target = PIPE_TEXTURE_2D_ARRAY;
      resource->base.format = PIPE_FORMAT_R8G8B8A8_UNORM;
      resource->base.width0 = width;
      resource->base.height0 = height;
      resource->base.depth0 = 1;
      resource->base.array_size = 2;
      resource->base.nr_samples = resource->base.nr_storage_samples = samples;
      resource->level_count = 1;
      resource->level_offsets[0] = 16;
      resource->level_strides[0] = row_bytes + 16;
      resource->level_layer_strides[0] = resource->level_strides[0] * height;
      resource->size = 16 + 2 * resource->level_layer_strides[0];
      resource->data = malloc(resource->size);
      CHECK(resource->data != NULL);
      memset(resource->data, 0xee, resource->size);
      for (unsigned y = 0; y < height; ++y)
         for (unsigned x = 0; x < row_bytes; ++x)
            resource->data[16 + resource->level_layer_strides[0] +
                           y * resource->level_strides[0] + x] =
               (uint8_t)(target * 43 + y * 17 + x);
      ctx.framebuffer.cbufs[target] = (struct pipe_surface){
         .texture = &resource->base, .format = resource->base.format,
         .first_layer = 1, .last_layer = 1};
   }
   CHECK(pvrgpu_capture_initial_color_attachment(&ctx, &recorded));
   CHECK(recorded.command.initial_color_attachment_bytes_size == target_bytes * 4);
   CHECK(recorded.command.initial_color_attachment_bytes == recorded.initial_color_attachment_bytes);
   for (unsigned target = 0; target < 4; ++target) {
      memset(resources[target].data, 0xcc, resources[target].size);
      for (unsigned y = 0; y < height; ++y)
         for (unsigned x = 0; x < row_bytes; ++x)
            CHECK(recorded.initial_color_attachment_bytes[
                     target * target_bytes + y * row_bytes + x] ==
                  (uint8_t)(target * 43 + y * 17 + x));
   }
   free(recorded.initial_color_attachment_bytes);
   recorded.initial_color_attachment_bytes = NULL;
   recorded.command.initial_color_attachment_bytes = NULL;
   recorded.command.initial_color_attachment_bytes_size = 0;
   /* A missing final target must not publish an incomplete aggregate. */
   const size_t saved_size = resources[3].size;
   resources[3].size = 1;
   CHECK(!pvrgpu_capture_initial_color_attachment(&ctx, &recorded));
   CHECK(recorded.initial_color_attachment_bytes == NULL);
   CHECK(recorded.command.initial_color_attachment_bytes_size == 0);
   resources[3].size = saved_size;
   ctx.framebuffer.nr_cbufs = 3;
   CHECK(!pvrgpu_capture_initial_color_attachment(&ctx, &recorded));
   for (unsigned target = 0; target < 4; ++target)
      free(resources[target].data);
}

static void test_color_transport_bounds(void)
{
   struct pvrgpu_context ctx = {0};
   struct pipe_resource resource = {0};
   CHECK(!pvrgpu_framebuffer_color_transport_is_bounded(NULL));
   for (unsigned count = 0; count <= 8; ++count) {
      ctx.framebuffer.nr_cbufs = count;
      for (unsigned i = 0; i < count; ++i)
         ctx.framebuffer.cbufs[i] = (struct pipe_surface){
            .texture = &resource, .format = PIPE_FORMAT_R8G8B8A8_UNORM};
      CHECK(pvrgpu_framebuffer_color_transport_is_bounded(&ctx) == (count <= 4));
      if (count > 1) {
         ctx.framebuffer.cbufs[1].format = PIPE_FORMAT_R10G10B10A2_UNORM;
         CHECK(pvrgpu_framebuffer_color_transport_is_bounded(&ctx) == (count <= 4));
         ctx.framebuffer.cbufs[1].format = PIPE_FORMAT_B10G10R10A2_UNORM;
         CHECK(pvrgpu_framebuffer_color_transport_is_bounded(&ctx) == (count <= 4));
         ctx.framebuffer.cbufs[1].format = PIPE_FORMAT_R8_UNORM;
         CHECK(!pvrgpu_framebuffer_color_transport_is_bounded(&ctx));
      }
      if (count) {
         ctx.framebuffer.cbufs[count - 1].texture = NULL;
         CHECK(!pvrgpu_framebuffer_color_transport_is_bounded(&ctx));
      }
   }
}

int main(void)
{
   test_cube_array_depth_views();
   test_color_transport_bounds();
   test_view(PIPE_TEXTURE_2D, PIPE_FORMAT_R8G8B8A8_UNORM, 100, 31, 1, 1, 6, 0, 1);
   test_view(PIPE_TEXTURE_2D, PIPE_FORMAT_R8G8B8A8_UNORM, 64, 32, 1, 2, 4, 0, 1);
   test_view(PIPE_TEXTURE_3D, PIPE_FORMAT_R32G32B32A32_FLOAT, 32, 16, 7, 1, 5, 0, 1);
   test_view(PIPE_TEXTURE_CUBE, PIPE_FORMAT_R32G32B32A32_UINT, 64, 64, 6, 1, 6, 0, 6);
   test_view(PIPE_TEXTURE_CUBE_ARRAY, PIPE_FORMAT_R8G8B8A8_UNORM, 64, 64, 18, 1, 4, 6, 12);
   test_view(PIPE_TEXTURE_CUBE_ARRAY, PIPE_FORMAT_R32G32B32A32_FLOAT, 16, 16, 12, 0, 3, 0, 12);
   test_view(PIPE_TEXTURE_CUBE_ARRAY, PIPE_FORMAT_R32G32B32A32_UINT, 16, 16, 12, 2, 4, 6, 6);
   test_view(PIPE_TEXTURE_2D_ARRAY, PIPE_FORMAT_Z16_UNORM, 64, 64, 5, 1, 4, 1, 3);
   test_shadow_metadata();
   test_shadow_gather_snapshot();
   test_mrt_initial_snapshot(1);
   test_mrt_initial_snapshot(4);
   printf("texture view snapshot: PASS (%u checks)\n", snapshot_checks);
   return 0;
}
