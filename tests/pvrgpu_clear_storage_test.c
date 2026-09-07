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
   test_uniform_includes_last_sample();
   test_pending_clear_lifecycle();
   if (!failures)
      puts("Mesa clear storage tests passed");
   return failures ? 1 : 0;
}
