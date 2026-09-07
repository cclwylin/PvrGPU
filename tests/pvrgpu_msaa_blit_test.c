/* SPDX-License-Identifier: MIT */
/* Standalone Mesa integration test. Compile with the pvrgpu_resource.c Mesa
 * compilation flags and link libmesa_util.a with dead-code elimination. This
 * exercises the actual static format-converting blit implementation. */
#include "../src/gallium/drivers/pvrgpu/pvrgpu_resource.c"

#include <math.h>

static unsigned failures;

#define CHECK(condition) do { \
   if (!(condition)) { \
      fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #condition); \
      ++failures; \
   } \
} while (0)

static struct pvrgpu_resource
test_resource_2d(enum pipe_format format, unsigned width, unsigned height,
                 unsigned samples)
{
   struct pvrgpu_resource resource = {0};
   resource.base.target = PIPE_TEXTURE_2D;
   resource.base.format = format;
   resource.base.width0 = width;
   resource.base.height0 = height;
   resource.base.depth0 = 1;
   resource.base.array_size = 1;
   resource.base.nr_samples = samples;
   resource.base.nr_storage_samples = samples;
   resource.base.bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW;
   CHECK(pvrgpu_init_resource_storage(&resource));
   return resource;
}

static struct pvrgpu_resource
test_resource(enum pipe_format format, unsigned width, unsigned samples)
{
   return test_resource_2d(format, width, 1, samples);
}

static struct pipe_blit_info
test_blit(struct pvrgpu_resource *src, struct pvrgpu_resource *dst)
{
   struct pipe_blit_info info = {0};
   info.src.resource = &src->base;
   info.src.format = src->base.format;
   info.src.box.width = (int)src->base.width0;
   info.src.box.height = info.src.box.depth = 1;
   info.dst.resource = &dst->base;
   info.dst.format = dst->base.format;
   info.dst.box.width = (int)dst->base.width0;
   info.dst.box.height = info.dst.box.depth = 1;
   info.filter = PIPE_TEX_FILTER_NEAREST;
   info.mask = PIPE_MASK_RGBA;
   return info;
}

static void
test_float_resolve(void)
{
   struct pvrgpu_resource src = test_resource(PIPE_FORMAT_R32G32B32A32_FLOAT, 2, 4);
   struct pvrgpu_resource dst = test_resource(PIPE_FORMAT_R32G32B32A32_FLOAT, 2, 1);
   float *values = (float *)src.data;
   for (unsigned pixel = 0; pixel < 2; ++pixel)
      for (unsigned sample = 0; sample < 4; ++sample)
         for (unsigned channel = 0; channel < 4; ++channel)
            values[(pixel * 4 + sample) * 4 + channel] =
               (float)pixel * 100 + (float)sample * 10 - (float)channel;
   struct pipe_blit_info info = test_blit(&src, &dst);
   CHECK(pvrgpu_can_blit_as_texture_region(&info));
   CHECK(pvrgpu_blit_texture_region_unchecked(&info));
   for (unsigned pixel = 0; pixel < 2; ++pixel)
      for (unsigned channel = 0; channel < 4; ++channel)
         CHECK(((float *)dst.data)[pixel * 4 + channel] ==
                  (float)pixel * 100 + 15 - (float)channel);

   info.src.box.x = 2;
   info.src.box.width = -2;
   CHECK(pvrgpu_can_blit_as_texture_region(&info));
   CHECK(pvrgpu_blit_texture_region_unchecked(&info));
   CHECK(((float *)dst.data)[0] == 115.0f);
   CHECK(((float *)dst.data)[4] == 15.0f);
   FREE(src.data);
   FREE(dst.data);
}

static void
test_integer_resolve(void)
{
   struct pvrgpu_resource src = test_resource(PIPE_FORMAT_R32_UINT, 2, 4);
   struct pvrgpu_resource dst = test_resource(PIPE_FORMAT_R32_UINT, 2, 1);
   for (unsigned sample = 0; sample < 8; ++sample)
      ((uint32_t *)src.data)[sample] = UINT32_C(0xf1234560) + sample;
   struct pipe_blit_info info = test_blit(&src, &dst);
   CHECK(pvrgpu_can_blit_as_texture_region(&info));
   CHECK(pvrgpu_blit_texture_region_unchecked(&info));
   CHECK(((uint32_t *)dst.data)[0] == UINT32_C(0xf1234560));
   CHECK(((uint32_t *)dst.data)[1] == UINT32_C(0xf1234564));
   FREE(src.data);
   FREE(dst.data);
}

static void
test_unorm_resolve(void)
{
   struct pvrgpu_resource src = test_resource(PIPE_FORMAT_R8G8B8A8_UNORM, 1, 4);
   struct pvrgpu_resource dst = test_resource(PIPE_FORMAT_R8G8B8A8_UNORM, 1, 1);
   const uint8_t red[4] = {0, 64, 128, 255};
   for (unsigned sample = 0; sample < 4; ++sample) {
      src.data[sample * 4] = red[sample];
      src.data[sample * 4 + 1] = 0;
      src.data[sample * 4 + 2] = 255;
      src.data[sample * 4 + 3] = 255;
   }
   struct pipe_blit_info info = test_blit(&src, &dst);
   CHECK(pvrgpu_can_blit_as_texture_region(&info));
   CHECK(pvrgpu_blit_texture_region_unchecked(&info));
   CHECK(dst.data[0] == 112);
   CHECK(dst.data[1] == 0 && dst.data[2] == 255 && dst.data[3] == 255);
   info.sample0_only = true;
   CHECK(pvrgpu_can_blit_as_texture_region(&info));
   CHECK(pvrgpu_blit_texture_region_unchecked(&info));
   CHECK(dst.data[0] == 0);
   FREE(src.data);
   FREE(dst.data);
}

static void
test_sample_copy_and_mask(void)
{
   struct pvrgpu_resource src = test_resource(PIPE_FORMAT_R8G8B8A8_UNORM, 3, 4);
   struct pvrgpu_resource dst = test_resource(PIPE_FORMAT_R8G8B8A8_UNORM, 3, 4);
   for (unsigned byte = 0; byte < src.size; ++byte)
      src.data[byte] = (uint8_t)(byte + 11);
   memset(dst.data, 0xa5, dst.size);
   struct pipe_blit_info info = test_blit(&src, &dst);
   info.src.box.x = 1;
   info.dst.box.x = 2;
   info.src.box.width = info.dst.box.width = 1;
   CHECK(pvrgpu_can_blit_as_2d_copy(&info));
   pvrgpu_copy_texture_region_unchecked(&dst.base, 0, 2, 0, 0,
                                        &src.base, 0, &info.src.box);
   CHECK(memcmp(dst.data + 32, src.data + 16, 16) == 0);
   CHECK(dst.data[0] == 0xa5 && dst.data[31] == 0xa5);

   memset(dst.data, 0xa5, dst.size);
   info.mask = PIPE_MASK_R;
   info.dst_sample = 3;
   CHECK(pvrgpu_can_blit_as_texture_region(&info));
   CHECK(pvrgpu_blit_texture_region_unchecked(&info));
   for (unsigned byte = 0; byte < dst.size; ++byte)
      CHECK(dst.data[byte] == (byte == 40 ? src.data[24] : 0xa5));
   info.dst_sample = 5;
   CHECK(!pvrgpu_can_blit_as_texture_region(&info));
   info.dst_sample = 0;
   dst.base.nr_samples = dst.base.nr_storage_samples = 2;
   CHECK(!pvrgpu_can_blit_as_texture_region(&info));
   FREE(src.data);
   FREE(dst.data);
}

static void
test_depth_stencil_aspects(void)
{
   struct pvrgpu_resource src = test_resource(PIPE_FORMAT_Z24_UNORM_S8_UINT, 3, 1);
   struct pvrgpu_resource dst = test_resource(PIPE_FORMAT_Z24_UNORM_S8_UINT, 3, 1);
   const uint32_t original[] = {0x12123456, 0x349abcde, 0x56ffffff};
   memcpy(src.data, original, sizeof(original));
   for (unsigned x = 0; x < 3; ++x)
      ((uint32_t *)dst.data)[x] = 0xa5222222;
   struct pipe_blit_info info = test_blit(&src, &dst);
   info.mask = PIPE_MASK_Z;
   info.src.box.x = 3;
   info.src.box.width = -3;
   CHECK(pvrgpu_can_blit_depth_stencil(&info));
   CHECK(pvrgpu_blit_depth_stencil_unchecked(&info));
   for (unsigned x = 0; x < 3; ++x)
      CHECK(((uint32_t *)dst.data)[x] ==
               (0xa5000000 | (original[2 - x] & 0xffffff)));

   /* A depth-only view's ordinary packer clears its unused high byte.  The
    * physical target still contains stencil there, which must survive. */
   info.src.format = info.dst.format = PIPE_FORMAT_Z24X8_UNORM;
   CHECK(pvrgpu_can_blit_depth_stencil(&info));
   CHECK(pvrgpu_blit_depth_stencil_unchecked(&info));
   for (unsigned x = 0; x < 3; ++x)
      CHECK(((uint32_t *)dst.data)[x] ==
               (0xa5000000 | (original[2 - x] & 0xffffff)));
   info.src.format = info.dst.format = PIPE_FORMAT_Z24_UNORM_S8_UINT;
   info.mask = PIPE_MASK_S;
   info.src.box.x = 0;
   info.src.box.width = 3;
   CHECK(pvrgpu_can_blit_depth_stencil(&info));
   CHECK(pvrgpu_blit_depth_stencil_unchecked(&info));
   for (unsigned x = 0; x < 3; ++x)
      CHECK(((uint32_t *)dst.data)[x] ==
               ((original[x] & 0xff000000) | (original[2 - x] & 0xffffff)));
   info.dst.format = PIPE_FORMAT_Z24X8_UNORM;
   CHECK(!pvrgpu_can_blit_depth_stencil(&info));
   info.dst.format = PIPE_FORMAT_Z24_UNORM_S8_UINT;
   info.filter = PIPE_TEX_FILTER_LINEAR;
   CHECK(!pvrgpu_can_blit_depth_stencil(&info));
   info.filter = PIPE_TEX_FILTER_NEAREST;
   info.mask |= PIPE_MASK_R;
   CHECK(!pvrgpu_can_blit_depth_stencil(&info));
   FREE(src.data);
   FREE(dst.data);
}

static void
test_depth_float_scale_flip(void)
{
   struct pvrgpu_resource src = test_resource_2d(PIPE_FORMAT_Z32_FLOAT, 2, 2, 1);
   struct pvrgpu_resource dst = test_resource_2d(PIPE_FORMAT_Z32_FLOAT, 4, 2, 1);
   const float values[] = {0.125f, 0.375f, 0.625f, 0.875f};
   memcpy(src.data, values, sizeof(values));
   for (unsigned pixel = 0; pixel < 8; ++pixel)
      ((float *)dst.data)[pixel] = 0.25f;
   struct pipe_blit_info info = test_blit(&src, &dst);
   info.mask = PIPE_MASK_Z;
   info.src.box.x = info.src.box.y = 2;
   info.src.box.width = info.src.box.height = -2;
   info.dst.box.height = 2;
   info.scissor_enable = true;
   info.scissor.minx = 1;
   info.scissor.maxx = 3;
   info.scissor.maxy = 2;
   CHECK(pvrgpu_can_blit_depth_stencil(&info));
   CHECK(pvrgpu_blit_depth_stencil_unchecked(&info));
   const float expected[] = {0.25f, 0.875f, 0.625f, 0.25f,
                             0.25f, 0.375f, 0.125f, 0.25f};
   CHECK(memcmp(dst.data, expected, sizeof(expected)) == 0);
   FREE(src.data);
   FREE(dst.data);
}

static void
test_depth_stencil_samples(void)
{
   struct pvrgpu_resource src = test_resource(PIPE_FORMAT_Z24_UNORM_S8_UINT, 2, 4);
   struct pvrgpu_resource dst = test_resource(PIPE_FORMAT_Z24_UNORM_S8_UINT, 2, 4);
   struct pvrgpu_resource resolved = test_resource(PIPE_FORMAT_Z24_UNORM_S8_UINT, 2, 1);
   for (unsigned index = 0; index < 8; ++index) {
      ((uint32_t *)src.data)[index] = ((index + 1) << 24) | (0x123450 + index);
      ((uint32_t *)dst.data)[index] = 0xa5000000 | (0xabcdef - index);
   }
   struct pipe_blit_info info = test_blit(&src, &dst);
   info.mask = PIPE_MASK_S;
   info.dst_sample = 3;
   CHECK(pvrgpu_can_blit_depth_stencil(&info));
   CHECK(pvrgpu_blit_depth_stencil_unchecked(&info));
   for (unsigned index = 0; index < 8; ++index)
      CHECK(((uint32_t *)dst.data)[index] ==
               ((index % 4 == 2 ? (index + 1) << 24 : 0xa5000000) |
                (0xabcdef - index)));
   info = test_blit(&src, &resolved);
   info.mask = PIPE_MASK_ZS;
   CHECK(pvrgpu_can_blit_depth_stencil(&info));
   CHECK(pvrgpu_blit_depth_stencil_unchecked(&info));
   CHECK(((uint32_t *)resolved.data)[0] == ((uint32_t *)src.data)[0]);
   CHECK(((uint32_t *)resolved.data)[1] == ((uint32_t *)src.data)[4]);
   FREE(src.data);
   FREE(dst.data);
   FREE(resolved.data);
}

static void
test_depth_alias_snapshot(void)
{
   struct pvrgpu_resource resource = test_resource(PIPE_FORMAT_Z32_UNORM, 4, 1);
   const uint32_t original[] = {0x12345678, 0x9abcdef0, 0xfedcba98, 0x87654321};
   memcpy(resource.data, original, sizeof(original));
   struct pipe_blit_info info = test_blit(&resource, &resource);
   info.mask = PIPE_MASK_Z;
   info.src.box.width = info.dst.box.width = 3;
   info.dst.box.x = 1;
   CHECK(pvrgpu_can_blit_depth_stencil(&info));
   CHECK(pvrgpu_blit_depth_stencil_unchecked(&info));
   CHECK(((uint32_t *)resource.data)[0] == original[0]);
   for (unsigned x = 1; x < 4; ++x)
      CHECK(((uint32_t *)resource.data)[x] == original[x - 1]);
   FREE(resource.data);
}

static void
test_clipped_color_transform(void)
{
   struct pvrgpu_resource src = test_resource_2d(PIPE_FORMAT_R32_FLOAT, 4, 3, 1);
   struct pvrgpu_resource dst = test_resource_2d(PIPE_FORMAT_R32_FLOAT, 5, 4, 1);
   for (unsigned y = 0; y < 3; ++y)
      for (unsigned x = 0; x < 4; ++x)
         ((float *)src.data)[y * 4 + x] = 10.0f * y + x;
   for (unsigned linear = 0; linear < 2; ++linear) {
      for (unsigned flip = 0; flip < 2; ++flip) {
         for (unsigned pixel = 0; pixel < 20; ++pixel)
            ((float *)dst.data)[pixel] = -100.0f;
         struct pipe_blit_info info = test_blit(&src, &dst);
         info.src.box = (struct pipe_box){.x = flip ? 6 : -2, .y = -1, .z = 0,
                                          .width = flip ? -8 : 8,
                                          .height = 5, .depth = 1};
         info.dst.box = (struct pipe_box){.x = -2, .y = -1, .z = 0,
                                          .width = 9, .height = 6, .depth = 1};
         info.filter = linear ? PIPE_TEX_FILTER_LINEAR : PIPE_TEX_FILTER_NEAREST;
         info.scissor_enable = true;
         info.scissor = (struct pipe_scissor_state){1, 1, 5, 3};
         CHECK(pvrgpu_can_blit_as_texture_region(&info));
         CHECK(pvrgpu_blit_texture_region_unchecked(&info));
         for (unsigned y = 0; y < 4; ++y) {
            for (unsigned x = 0; x < 5; ++x) {
               float expected = -100.0f;
               if (x >= 1 && y >= 1 && y < 3) {
                  const double sx = (flip ? 6.0 : -2.0) +
                     (flip ? -8.0 : 8.0) * ((double)x + 2.5) / 9.0;
                  const double sy = -1.0 + 5.0 * ((double)y + 1.5) / 6.0;
                  const double tx = linear ? CLAMP(sx - 0.5, 0.0, 3.0) :
                                              CLAMP(floor(sx), 0.0, 3.0);
                  const double ty = linear ? CLAMP(sy - 0.5, 0.0, 2.0) :
                                              CLAMP(floor(sy), 0.0, 2.0);
                  expected = (float)(tx + 10.0 * ty);
               }
               CHECK(fabsf(((float *)dst.data)[y * 5 + x] - expected) < 0.00001f);
            }
         }
         /* Fully clipped operations must be successful no-ops, including when
          * the original destination rectangle is entirely outside storage. */
         float before[20];
         memcpy(before, dst.data, sizeof(before));
         info.dst.box.x = -100;
         CHECK(pvrgpu_can_blit_as_texture_region(&info));
         CHECK(pvrgpu_blit_texture_region_unchecked(&info));
         CHECK(memcmp(before, dst.data, sizeof(before)) == 0);
      }
   }
   FREE(src.data);
   FREE(dst.data);
}

int main(void)
{
   test_float_resolve();
   test_integer_resolve();
   test_unorm_resolve();
   test_sample_copy_and_mask();
   test_depth_stencil_aspects();
   test_depth_float_scale_flip();
   test_depth_stencil_samples();
   test_depth_alias_snapshot();
   test_clipped_color_transform();
   if (!failures)
      puts("MSAA Mesa format blit tests passed");
   return failures ? 1 : 0;
}
