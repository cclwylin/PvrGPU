/* SPDX-License-Identifier: MIT */
/* Exercise the actual context snapshot, not a second layout implementation. */
#include "../src/gallium/drivers/pvrgpu/pvrgpu_context.c"

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
      target == PIPE_TEXTURE_2D_ARRAY ? depth : 1;
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
   ctx.sampler_views[MESA_SHADER_VERTEX][0] = &view;
   ctx.samplers[MESA_SHADER_VERTEX][0] = &sampler;
   struct pvrgpu_systemc_pco_sequence_texture captured = {0};
   uint8_t *bytes = NULL;
   const char *reason = NULL;
   CHECK(pvrgpu_capture_generic_sequence_texture(&ctx, MESA_SHADER_VERTEX, 0, 0,
                                                  &captured, &bytes, &reason));
   CHECK(bytes != resource.data && captured.bytes == bytes && captured.mip_count == last-first+1);
   CHECK(captured.layers == (target == PIPE_TEXTURE_3D ? u_minify(depth, first) :
         target == PIPE_TEXTURE_2D_ARRAY ? layers : target == PIPE_TEXTURE_CUBE ? 6 : 1));
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
   view.u.tex.first_level = last + 1;
   CHECK(!pvrgpu_capture_generic_sequence_texture(&ctx, MESA_SHADER_VERTEX, 0, 0,
                                                   &captured, &bytes, &reason));
   CHECK(bytes == NULL && !strcmp(reason, "view_range"));
   view.u.tex.first_level = first;
   resource.level_layer_strides[first]++;
   CHECK(!pvrgpu_capture_generic_sequence_texture(&ctx, MESA_SHADER_VERTEX, 0, 0,
                                                   &captured, &bytes, &reason));
   CHECK(bytes == NULL && !strcmp(reason, "mip_layout"));
   free(resource.data);
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

int main(void)
{
   test_view(PIPE_TEXTURE_2D, PIPE_FORMAT_R8G8B8A8_UNORM, 100, 31, 1, 1, 6, 0, 1);
   test_view(PIPE_TEXTURE_2D, PIPE_FORMAT_R8G8B8A8_UNORM, 64, 32, 1, 2, 4, 0, 1);
   test_view(PIPE_TEXTURE_3D, PIPE_FORMAT_R32G32B32A32_FLOAT, 32, 16, 7, 1, 5, 0, 1);
   test_view(PIPE_TEXTURE_CUBE, PIPE_FORMAT_R32G32B32A32_UINT, 64, 64, 6, 1, 6, 0, 6);
   test_view(PIPE_TEXTURE_2D_ARRAY, PIPE_FORMAT_Z16_UNORM, 64, 64, 5, 1, 4, 1, 3);
   test_shadow_metadata();
   test_mrt_initial_snapshot(1);
   test_mrt_initial_snapshot(4);
   printf("texture view snapshot: PASS (%u checks)\n", snapshot_checks);
   return 0;
}
