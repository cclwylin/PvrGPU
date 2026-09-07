/* SPDX-License-Identifier: MIT */
/* Run with script/run_mesa_resource_unit.sh clear. */
#include "../src/gallium/drivers/pvrgpu/pvrgpu_clear.c"

static unsigned failures;
#define CHECK(condition) do { \
   if (!(condition)) { \
      fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #condition); \
      ++failures; \
   } \
} while (0)

static struct pvrgpu_resource
test_resource(enum pipe_format format, unsigned samples)
{
   struct pvrgpu_resource resource = {0};
   resource.base.target = PIPE_TEXTURE_2D;
   resource.base.format = format;
   resource.base.width0 = 3;
   resource.base.height0 = 2;
   resource.base.depth0 = resource.base.array_size = 1;
   resource.base.nr_samples = resource.base.nr_storage_samples = samples;
   resource.level_count = 1;
   resource.stride = resource.level_strides[0] =
      3 * samples * util_format_get_blocksize(format);
   resource.layer_stride = resource.level_layer_strides[0] = resource.stride * 2;
   resource.size = resource.layer_stride;
   resource.data = calloc(1, resource.size);
   CHECK(resource.data != NULL);
   return resource;
}

static struct pipe_surface
test_surface(struct pvrgpu_resource *resource)
{
   struct pipe_surface surface = {0};
   surface.texture = &resource->base;
   surface.format = resource->base.format;
   return surface;
}

static void
test_integer_color(enum pipe_format format)
{
   struct pvrgpu_resource resource = test_resource(format, 16);
   struct pipe_surface surface = test_surface(&resource);
   union pipe_color_union color;
   color.ui[0] = UINT32_C(0xffffffff);
   color.ui[1] = UINT32_C(0x80000000);
   color.ui[2] = UINT32_C(0x7fffffff);
   color.ui[3] = UINT32_C(0xdeadbeef);
   pvrgpu_fill_surface_rect_with_clear_color(&surface, 0, 0, 3, 2,
                                              PIPE_MASK_RGBA, &color);
   for (unsigned texel = 0; texel < 6 * 16; ++texel)
      CHECK(memcmp(resource.data + texel * 16, color.ui, 16) == 0);

   memset(resource.data, 0xa5, resource.size);
   pvrgpu_fill_surface_rect_with_clear_color(&surface, 1, 1, 1, 1,
                                              PIPE_MASK_R | PIPE_MASK_A, &color);
   for (unsigned texel = 0; texel < 6 * 16; ++texel) {
      const bool selected = texel >= 4 * 16 && texel < 5 * 16;
      const uint32_t *pixel = (const uint32_t *)resource.data + texel * 4;
      for (unsigned channel = 0; channel < 4; ++channel)
         CHECK(pixel[channel] ==
               (selected && (channel == 0 || channel == 3)
                   ? color.ui[channel] : UINT32_C(0xa5a5a5a5)));
   }
   free(resource.data);
}

static void
test_float_color(void)
{
   struct pvrgpu_resource resource = test_resource(PIPE_FORMAT_R32G32B32A32_FLOAT, 4);
   struct pipe_surface surface = test_surface(&resource);
   union pipe_color_union color = {.f = {-2.0F, 4.0F, 1234.5F, 0.125F}};
   pvrgpu_fill_surface_rect_with_clear_color(&surface, 0, 0, 3, 2,
                                              PIPE_MASK_RGBA, &color);
   for (unsigned texel = 0; texel < 6 * 4; ++texel)
      CHECK(memcmp(resource.data + texel * 16, color.f, 16) == 0);

   uint32_t *words = (uint32_t *)resource.data;
   for (unsigned word = 0; word < resource.size / 4; ++word)
      words[word] = UINT32_C(0x7fc12345);
   struct pipe_framebuffer_state fb = {.width = 3, .height = 2};
   struct pipe_scissor_state scissor = {.minx = 1, .miny = 1, .maxx = 2, .maxy = 2};
   unsigned x, y, width, height;
   CHECK(pvrgpu_clear_rect(&fb, &scissor, &x, &y, &width, &height));
   CHECK(x == 1 && y == 1 && width == 1 && height == 1);
   pvrgpu_fill_surface_rect_with_clear_color(&surface, x, y, width, height,
                                              PIPE_MASK_R, &color);
   for (unsigned texel = 0; texel < 6 * 4; ++texel)
      for (unsigned channel = 0; channel < 4; ++channel)
         CHECK(words[texel * 4 + channel] ==
               (texel >= 4 * 4 && texel < 5 * 4 && channel == 0
                  ? color.ui[0] : UINT32_C(0x7fc12345)));
   const uint32_t first = words[0];
   pvrgpu_fill_surface_rect_with_clear_color(&surface, 0, 0, 3, 2, 0, &color);
   CHECK(words[0] == first);
   free(resource.data);
}

static void
test_depth_stencil(void)
{
   struct pvrgpu_resource resource = test_resource(PIPE_FORMAT_Z24_UNORM_S8_UINT, 4);
   struct pipe_surface surface = test_surface(&resource);
   uint32_t *words = (uint32_t *)resource.data;
   for (unsigned word = 0; word < resource.size / 4; ++word)
      words[word] = UINT32_C(0xab123456);
   CHECK(pvrgpu_fill_surface_rect_with_clear_depth(&surface, 1, 1, 1, 1, 0.5));
   for (unsigned sample = 0; sample < 6 * 4; ++sample)
      CHECK(words[sample] == (sample >= 16 && sample < 20
                                 ? UINT32_C(0xab7fffff) : UINT32_C(0xab123456)));
   CHECK(pvrgpu_fill_surface_rect_with_clear_stencil(&surface, 1, 1, 1, 1,
                                                      0x31, 0x0f));
   for (unsigned sample = 0; sample < 6 * 4; ++sample)
      CHECK(words[sample] == (sample >= 16 && sample < 20
                                 ? UINT32_C(0xa17fffff) : UINT32_C(0xab123456)));
   free(resource.data);
}

static void
test_depth_layers(enum pipe_format format, unsigned samples,
                  enum pipe_texture_target target, unsigned layers,
                  unsigned level)
{
   const unsigned block_size = util_format_get_blocksize(format);
   const bool has_stencil =
      util_format_pack_description(format)->pack_s_8uint != NULL;
   struct pvrgpu_resource resource = test_resource(format, samples);
   free(resource.data);
   resource.base.target = target;
   resource.base.width0 = 3u << level;
   resource.base.height0 = 2u << level;
   resource.base.array_size = layers;
   resource.level_count = level + 1;
   /* Keep a prefix, row padding, layer padding and a suffix as guards. */
   resource.level_offsets[level] = 16;
   resource.level_strides[level] = 3 * samples * block_size + 16;
   resource.level_layer_strides[level] =
      resource.level_strides[level] * 2 + 16;
   resource.size = 32 + layers * resource.level_layer_strides[level];
   resource.data = malloc(resource.size);
   uint8_t *expected = malloc(resource.size);
   CHECK(resource.data && expected);
   if (!resource.data || !expected) {
      free(resource.data);
      free(expected);
      return;
   }
   memset(resource.data, 0xa5, resource.size);
   struct pipe_surface surface = test_surface(&resource);
   surface.level = level;
   const float initial_depth = 0.25f;
   const uint8_t initial_stencil = 0xab;
   for (unsigned layer = 0; layer < layers; ++layer) {
      for (unsigned y = 0; y < 2; ++y) {
         for (unsigned x = 0; x < 3 * samples; ++x) {
            uint8_t *pixel = resource.data + 16 +
               layer * resource.level_layer_strides[level] +
               y * resource.level_strides[level] + x * block_size;
            util_format_pack_z_float(format, pixel, &initial_depth, 1);
            if (has_stencil)
               util_format_pack_s_8uint(format, pixel, &initial_stencil, 1);
         }
      }
   }
   memcpy(expected, resource.data, resource.size);

   /* Match ordinary FBO attachment switches: a full clear of each single
    * layer must initialize every sample, including pixels never rasterized. */
   const float full_depth = 0.9375f;
   for (unsigned layer = 0; layer < layers; ++layer) {
      surface.first_layer = surface.last_layer = layer;
      CHECK(pvrgpu_fill_surface_rect_with_clear_depth(
         &surface, 0, 0, 3, 2, full_depth));
      for (unsigned y = 0; y < 2; ++y)
         for (unsigned x = 0; x < 3 * samples; ++x)
            util_format_pack_z_float(format,
               expected + 16 + layer * resource.level_layer_strides[level] +
               y * resource.level_strides[level] + x * block_size,
               &full_depth, 1);
      CHECK(memcmp(resource.data, expected, resource.size) == 0);
   }

   /* A scissored layer range preserves layer 0, all borders, all padding,
    * and the other aspect of packed depth/stencil formats. */
   surface.first_layer = 1;
   surface.last_layer = layers - 1;
   CHECK(pvrgpu_fill_surface_rect_with_clear_depth(&surface, 1, 1, 1, 1, 0.5));
   if (has_stencil) {
      CHECK(pvrgpu_fill_surface_rect_with_clear_stencil(
         &surface, 1, 1, 1, 1, 0x31, 0x0f));
      CHECK(pvrgpu_fill_surface_rect_with_clear_stencil(
         &surface, 0, 0, 3, 2, 0, 0));
   }
   const float partial_depth = 0.5f;
   const uint8_t partial_stencil = 0xa1;
   for (unsigned layer = 1; layer < layers; ++layer)
      for (unsigned sample = 0; sample < samples; ++sample) {
         uint8_t *pixel = expected + 16 +
            layer * resource.level_layer_strides[level] +
            resource.level_strides[level] + (samples + sample) * block_size;
         util_format_pack_z_float(format, pixel, &partial_depth, 1);
         if (has_stencil)
            util_format_pack_s_8uint(format, pixel, &partial_stencil, 1);
      }
   CHECK(memcmp(resource.data, expected, resource.size) == 0);

   /* Malformed metadata must fail before the first store, including when
    * the logical layer exists but the allocation cannot hold that layer. */
   const struct pvrgpu_resource saved = resource;
   const struct pipe_surface saved_surface = surface;
   for (unsigned invalid = 0; invalid < 8; ++invalid) {
      resource = saved;
      surface = saved_surface;
      switch (invalid) {
      case 0: surface.first_layer = 2; surface.last_layer = 1; break;
      case 1: surface.last_layer = layers; break;
      case 2: resource.level_strides[level] = 3 * samples * block_size - 1; break;
      case 3: resource.level_layer_strides[level] =
                 resource.level_strides[level] * 2 - 1; break;
      case 4: resource.size = resource.level_offsets[level] +
                 layers * resource.level_layer_strides[level] - 1; break;
      case 5: resource.level_offsets[level] = SIZE_MAX; break;
      case 6: surface.level = level + 1; break;
      case 7: resource.level_layer_strides[level] = SIZE_MAX; break;
      }
      CHECK(!pvrgpu_fill_surface_rect_with_clear_depth(
         &surface, 0, 0, 3, 2, 0.0));
      CHECK(!pvrgpu_fill_surface_rect_with_clear_stencil(
         &surface, 0, 0, 3, 2, 0, 0xff));
      CHECK(memcmp(saved.data, expected, saved.size) == 0);
   }
   resource = saved;
   surface = saved_surface;
   CHECK(!pvrgpu_fill_surface_rect_with_clear_depth(
      &surface, 2, 1, 2, 1, 0.0));
   CHECK(memcmp(resource.data, expected, resource.size) == 0);
   free(resource.data);
   free(expected);
}

static void
test_uniform_includes_last_sample(void)
{
   struct pvrgpu_resource resource = test_resource(PIPE_FORMAT_R8G8B8A8_UNORM, 4);
   struct pipe_surface surface = test_surface(&resource);
   memset(resource.data, 31, resource.size);
   float color[4];
   CHECK(pvrgpu_surface_uniform_color(&surface, 3, 2, color));
   resource.data[resource.size - 1] = 32;
   CHECK(!pvrgpu_surface_uniform_color(&surface, 3, 2, color));
   free(resource.data);
}

static void
test_pending_clear_lifecycle(void)
{
   struct pvrgpu_context ctx = {0};
   const unsigned masks[] = {0, 0x0f, 0xff};
   for (unsigned samples = 1; samples <= 16; samples *= 2) {
      for (unsigned mask_index = 0;
           mask_index < sizeof(masks) / sizeof(masks[0]); ++mask_index) {
         const unsigned mask = masks[mask_index];
         struct pvrgpu_resource resource =
            test_resource(PIPE_FORMAT_Z24_UNORM_S8_UINT, samples);
         struct pipe_surface surface = test_surface(&resource);
         uint32_t *words = (uint32_t *)resource.data;
         for (unsigned sample = 0; sample < 6 * samples; ++sample)
            words[sample] = UINT32_C(0xa5123456);
         CHECK(pvrgpu_fill_surface_rect_with_clear_stencil(
            &surface, 1, 1, 1, 1, 0x12, mask));

         /* Capture the actual native LOAD, then retire rectangles it already
          * contains. Replaying a rectangle with its implicit full-byte mask
          * would change A5/A2 to 12, despite the CPU clear being correct. */
         void *load = malloc(resource.size);
         CHECK(load != NULL);
         memcpy(load, resource.data, resource.size);
         ctx.array_primitive_draw_count = 0;
         ctx.pending_attachment_clear_count = 1;
         pvrgpu_retire_materialized_attachment_clears(&ctx, false, resource.size);
         CHECK(ctx.pending_attachment_clear_count == 0);
         resource.data = load;
         if (ctx.pending_attachment_clear_count)
            CHECK(pvrgpu_fill_surface_rect_with_clear_stencil(
               &surface, 1, 1, 1, 1, 0x12, 0xff));
         const uint32_t stencil = (0xa5 & ~mask) | (0x12 & mask);
         for (unsigned sample = 0; sample < 6 * samples; ++sample)
            CHECK(((uint32_t *)load)[sample] ==
                  (sample >= 4 * samples && sample < 5 * samples
                     ? (stencil << 24) | UINT32_C(0x123456)
                     : UINT32_C(0xa5123456)));
         free(words);
         free(load);
      }
   }

   /* No new LOAD: keep the ordered clear between two draws in an incomplete
    * RDC replay. Merely selecting the same framebuffer is not a boundary. */
   ctx.array_primitive_draw_count = 2;
   ctx.pending_attachment_clear_count = 3;
   pvrgpu_retire_materialized_attachment_clears(&ctx, false, 0);
   CHECK(ctx.pending_attachment_clear_count == 3);
   ctx.array_primitive_draw_count = 0;
   pvrgpu_retire_materialized_attachment_clears(&ctx, false, 0);
   CHECK(ctx.pending_attachment_clear_count == 3);

   /* An A -> B attachment-key change invalidates A's unconsumed rectangles,
    * even if A had no draw. B's own contents must not receive A's clear. */
   struct pvrgpu_resource other = test_resource(PIPE_FORMAT_Z24_UNORM_S8_UINT, 4);
   struct pipe_surface other_surface = test_surface(&other);
   memset(other.data, 0x3c, other.size);
   pvrgpu_retire_materialized_attachment_clears(&ctx, true, 0);
   CHECK(ctx.pending_attachment_clear_count == 0);
   if (ctx.pending_attachment_clear_count)
      CHECK(pvrgpu_fill_surface_rect_with_clear_stencil(
         &other_surface, 1, 1, 1, 1, 0x12, 0xff));
   for (unsigned byte = 0; byte < other.size; ++byte)
      CHECK(other.data[byte] == 0x3c);
   free(other.data);
}

int main(void)
{
   test_integer_color(PIPE_FORMAT_R32G32B32A32_UINT);
   test_integer_color(PIPE_FORMAT_R32G32B32A32_SINT);
   test_float_color();
   test_depth_stencil();
   const enum pipe_format depth_formats[] = {
      PIPE_FORMAT_Z32_FLOAT, PIPE_FORMAT_Z24_UNORM_S8_UINT,
      PIPE_FORMAT_Z32_FLOAT_S8X24_UINT
   };
   for (unsigned format = 0; format < 3; ++format) {
      for (unsigned samples = 1; samples <= 16; samples *= 2)
         test_depth_layers(depth_formats[format], samples,
                           PIPE_TEXTURE_2D_ARRAY, 3, 0);
      test_depth_layers(depth_formats[format], 1, PIPE_TEXTURE_2D_ARRAY, 3, 1);
      test_depth_layers(depth_formats[format], 1, PIPE_TEXTURE_CUBE, 6, 1);
   }
   test_uniform_includes_last_sample();
   test_pending_clear_lifecycle();
   if (!failures)
      puts("Mesa clear storage tests passed");
   return failures ? 1 : 0;
}
