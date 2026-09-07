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
   resource.base.bind = PIPE_BIND_RENDER_TARGET;
   /* 16 samples 僅供 render/resolve，不能宣告不可編碼的 sampler view。 */
   if (samples <= 8)
      resource.base.bind |= PIPE_BIND_SAMPLER_VIEW;
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

static double
test_srgb_decode(uint8_t value)
{
   const double encoded = (double)value / 255.0;
   return encoded <= 0.04045 ? encoded / 12.92 :
                              pow((encoded + 0.055) / 1.055, 2.4);
}

static void
test_srgb_resolve(void)
{
   /* RGB must be unpacked to linear light before the MSAA average, and only
    * encoded after it. Alpha remains linear in both source and destination.
    * Midtone inputs also distinguish decode-before-average from byte math. */
   const uint8_t pairs[2][2][4] = {
      {{0, 255, 0, 0}, {255, 0, 255, 255}},
      {{0, 64, 128, 0}, {128, 255, 255, 255}},
   };
   const uint8_t expected_encoded[2][4] = {
      {188, 188, 188, 128}, {92, 192, 205, 128},
   };
   for (unsigned samples = 2; samples <= 16; samples *= 2) {
      struct pvrgpu_resource src =
         test_resource(PIPE_FORMAT_R8G8B8A8_SRGB, 1, samples);
      struct pvrgpu_resource linear =
         test_resource(PIPE_FORMAT_R32G32B32A32_FLOAT, 1, 1);
      struct pvrgpu_resource encoded =
         test_resource(PIPE_FORMAT_R8G8B8A8_SRGB, 1, 1);
      for (unsigned pair = 0; pair < ARRAY_SIZE(pairs); ++pair) {
         for (unsigned sample = 0; sample < samples; ++sample)
            memcpy(src.data + sample * 4, pairs[pair][sample % 2], 4);

         struct pipe_blit_info info = test_blit(&src, &linear);
         CHECK(pvrgpu_can_blit_as_texture_region(&info));
         CHECK(pvrgpu_blit_texture_region_unchecked(&info));
         for (unsigned channel = 0; channel < 3; ++channel) {
            const double expected =
               (test_srgb_decode(pairs[pair][0][channel]) +
                test_srgb_decode(pairs[pair][1][channel])) * 0.5;
            CHECK(fabs((double)((float *)linear.data)[channel] - expected) <
                  0.000001);
         }
         CHECK(((float *)linear.data)[3] == 0.5f);

         info = test_blit(&src, &encoded);
         CHECK(pvrgpu_can_blit_as_texture_region(&info));
         CHECK(pvrgpu_blit_texture_region_unchecked(&info));
         CHECK(memcmp(encoded.data, expected_encoded[pair], 4) == 0);

         info.sample0_only = true;
         CHECK(pvrgpu_can_blit_as_texture_region(&info));
         CHECK(pvrgpu_blit_texture_region_unchecked(&info));
         CHECK(memcmp(encoded.data, pairs[pair][0], 4) == 0);
      }
      FREE(src.data);
      FREE(linear.data);
      FREE(encoded.data);
   }
}

static void
test_bilinear_resolve(void)
{
   /* Mesa's util_make_fs_msaa_resolve_bilinear averages every sample at each
    * of four texels before interpolation. Distinct, signed per-sample offsets
    * make sample-zero copying observably different from the true result. */
   for (unsigned samples = 2; samples <= 16; samples *= 2) {
      struct pvrgpu_resource src =
         test_resource_2d(PIPE_FORMAT_R32_FLOAT, 2, 2, samples);
      struct pvrgpu_resource dst =
         test_resource_2d(PIPE_FORMAT_R32_FLOAT, 3, 3, 1);
      for (unsigned y = 0; y < 2; ++y)
         for (unsigned x = 0; x < 2; ++x)
            for (unsigned sample = 0; sample < samples; ++sample)
               ((float *)src.data)[(y * 2 + x) * samples + sample] =
                  -8.0f + 16.0f * x + 32.0f * y +
                  (float)sample - (float)(samples - 1) * 0.5f;
      for (unsigned flip = 0; flip < 2; ++flip) {
         struct pipe_blit_info info = test_blit(&src, &dst);
         info.src.box.x = info.src.box.y = flip ? 2 : 0;
         info.src.box.width = info.src.box.height = flip ? -2 : 2;
         info.dst.box.height = 3;
         info.filter = PIPE_TEX_FILTER_LINEAR;
         CHECK(pvrgpu_can_blit_as_texture_region(&info));
         CHECK(pvrgpu_blit_texture_region_unchecked(&info));
         for (unsigned y = 0; y < 3; ++y) {
            for (unsigned x = 0; x < 3; ++x) {
               const float tx = (float)(flip ? 2 - x : x) * 0.5f;
               const float ty = (float)(flip ? 2 - y : y) * 0.5f;
               const float expected = -8.0f + 16.0f * tx + 32.0f * ty;
               CHECK(fabsf(((float *)dst.data)[y * 3 + x] - expected) <
                     0.00001f);
            }
         }
      }
      FREE(src.data);
      FREE(dst.data);
   }
}

static void
test_signed_integer_resolve_and_mask(void)
{
   const int32_t original[2][4] = {
      {INT32_MIN, INT32_MAX, -16777217, 16777217},
      {-1, -123456789, 123456789, 0},
   };
   const int32_t preserved[4] = {INT32_MAX - 1, INT32_MIN + 1, -42, 42};
   for (unsigned samples = 2; samples <= 16; samples *= 2) {
      struct pvrgpu_resource src =
         test_resource(PIPE_FORMAT_R32G32B32A32_SINT, 2, samples);
      struct pvrgpu_resource dst =
         test_resource(PIPE_FORMAT_R32G32B32A32_SINT, 2, 1);
      struct pvrgpu_resource copied =
         test_resource(PIPE_FORMAT_R32G32B32A32_SINT, 2, samples);
      for (unsigned pixel = 0; pixel < 2; ++pixel) {
         for (unsigned sample = 0; sample < samples; ++sample) {
            int32_t *value = (int32_t *)src.data +
                             (pixel * samples + sample) * 4;
            for (unsigned channel = 0; channel < 4; ++channel)
               value[channel] = sample == 0 ? original[pixel][channel] :
                  -(int32_t)(100 * pixel + 10 * sample + channel);
            memcpy((int32_t *)copied.data + (pixel * samples + sample) * 4,
                   preserved, sizeof(preserved));
         }
      }
      struct pipe_blit_info info = test_blit(&src, &dst);
      CHECK(pvrgpu_can_blit_as_texture_region(&info));
      CHECK(pvrgpu_blit_texture_region_unchecked(&info));
      CHECK(memcmp(dst.data, original, sizeof(original)) == 0);

      for (unsigned pixel = 0; pixel < 2; ++pixel)
         memcpy((int32_t *)dst.data + pixel * 4, preserved, sizeof(preserved));
      info.mask = PIPE_MASK_R | PIPE_MASK_B;
      CHECK(pvrgpu_can_blit_as_texture_region(&info));
      CHECK(pvrgpu_blit_texture_region_unchecked(&info));
      for (unsigned pixel = 0; pixel < 2; ++pixel)
         for (unsigned channel = 0; channel < 4; ++channel)
            CHECK(((int32_t *)dst.data)[pixel * 4 + channel] ==
                  (channel % 2 == 0 ? original[pixel][channel] :
                                     preserved[channel]));

      info = test_blit(&src, &copied);
      info.mask = PIPE_MASK_R | PIPE_MASK_B;
      info.dst_sample = samples; /* Gallium uses one-based selected samples. */
      CHECK(pvrgpu_can_blit_as_texture_region(&info));
      CHECK(pvrgpu_blit_texture_region_unchecked(&info));
      for (unsigned pixel = 0; pixel < 2; ++pixel)
         for (unsigned sample = 0; sample < samples; ++sample)
            for (unsigned channel = 0; channel < 4; ++channel) {
               const size_t index = (pixel * samples + sample) * 4 + channel;
               CHECK(((int32_t *)copied.data)[index] ==
                     (sample == samples - 1 && channel % 2 == 0 ?
                        ((int32_t *)src.data)[index] : preserved[channel]));
            }
      info.filter = PIPE_TEX_FILTER_LINEAR;
      CHECK(!pvrgpu_can_blit_as_texture_region(&info));
      FREE(src.data);
      FREE(dst.data);
      FREE(copied.data);
   }
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

static void
test_msaa_sample0_map(void)
{
   const enum pipe_format formats[] = {PIPE_FORMAT_R8_UNORM,
      PIPE_FORMAT_B8G8R8A8_UNORM, PIPE_FORMAT_R32_UINT,
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_Z24_UNORM_S8_UINT,
      PIPE_FORMAT_Z32_FLOAT_S8X24_UINT};
   const unsigned usages[] = {PIPE_MAP_READ, PIPE_MAP_WRITE,
      PIPE_MAP_READ | PIPE_MAP_WRITE, PIPE_MAP_WRITE | PIPE_MAP_DISCARD_RANGE,
      PIPE_MAP_WRITE | PIPE_MAP_DISCARD_WHOLE_RESOURCE,
      PIPE_MAP_WRITE | PIPE_MAP_FLUSH_EXPLICIT,
      PIPE_MAP_WRITE | PIPE_MAP_DISCARD_RANGE | PIPE_MAP_FLUSH_EXPLICIT};
   for (unsigned f = 0; f < ARRAY_SIZE(formats); ++f)
      for (unsigned samples = 2; samples <= 16; samples *= 2) {
         struct pvrgpu_resource resource = {0};
         resource.base.target = PIPE_TEXTURE_2D_ARRAY;
         resource.base.format = formats[f];
         resource.base.width0 = 16;
         resource.base.height0 = 8;
         resource.base.depth0 = 1;
         resource.base.array_size = 3;
         resource.base.last_level = 2;
         resource.base.nr_samples = resource.base.nr_storage_samples = samples;
         CHECK(pvrgpu_init_resource_storage(&resource));
         uint8_t *expected = malloc(resource.size);
         CHECK(expected != NULL);
         const unsigned bpp = util_format_get_blocksize(formats[f]);
         for (unsigned u = 0; u < ARRAY_SIZE(usages); ++u) {
            for (size_t i = 0; i < resource.size; ++i)
               resource.data[i] = (uint8_t)((i * 37U) ^ (i >> 8));
            memcpy(expected, resource.data, resource.size);
            struct pvrgpu_transfer transfer = {0};
            transfer.base.resource = &resource.base;
            transfer.base.level = 1;
            transfer.base.box = (struct pipe_box){.x = 3, .y = 1, .z = 1,
               .width = 4, .height = 2, .depth = 2};
            transfer.base.usage = usages[u];
            uint8_t *mapped = pvrgpu_map_msaa_sample0(&transfer);
            CHECK(mapped != NULL && mapped != resource.data);
            if (!mapped)
               exit(1);
            CHECK(transfer.base.stride == 4U * bpp &&
                  transfer.base.layer_stride == 8U * bpp &&
                  transfer.sample0_staging_size == 16U * bpp);
            for (unsigned z = 0; z < 2; ++z)
               for (unsigned y = 0; y < 2; ++y)
                  for (unsigned x = 0; x < 4; ++x) {
                     const size_t physical = resource.level_offsets[1] +
                        (z + 1U) * resource.level_layer_strides[1] +
                        (y + 1U) * resource.level_strides[1] +
                        (x + 3U) * bpp * samples;
                     const size_t packed = z * transfer.base.layer_stride +
                        y * transfer.base.stride + x * bpp;
                     CHECK(memcmp(mapped + packed, resource.data + physical, bpp) == 0);
                  }
            const size_t physical = resource.level_offsets[1] +
               2U * resource.level_layer_strides[1] +
               2U * resource.level_strides[1] + 5U * bpp * samples;
            const size_t packed = transfer.base.layer_stride + transfer.base.stride + 2U * bpp;
            memset(mapped + packed, 0x61, bpp);
            if (usages[u] & PIPE_MAP_WRITE)
               memset(expected + physical, 0x61, bpp);
            if (usages[u] & PIPE_MAP_FLUSH_EXPLICIT) {
               // Regions are relative to the map, not the resource origin.
               const struct pipe_box region = {.x = 2, .y = 1, .z = 1,
                  .width = 1, .height = 1, .depth = 1};
               pvrgpu_transfer_flush_region(NULL, &transfer.base, &region);
               CHECK(memcmp(resource.data, expected, resource.size) == 0);
               // Unflushed changes, including edits after an explicit flush,
               // must not be silently committed when the map is released.
               memset(mapped, 0xcc, transfer.sample0_staging_size);
               const struct pipe_box bad[] = {
                  {.x = -1, .width = 1, .height = 1, .depth = 1},
                  {.width = 5, .height = 1, .depth = 1},
                  {.y = 2, .width = 1, .height = 1, .depth = 1},
                  {.z = 2, .width = 1, .height = 1, .depth = 1},
                  {.width = 1, .height = 0, .depth = 1},
                  {.x = INT_MAX, .width = INT_MAX, .height = 1, .depth = 1}};
               for (unsigned i = 0; i < ARRAY_SIZE(bad); ++i)
                  CHECK(!pvrgpu_copy_msaa_sample0(&transfer, &bad[i], true));
            }
            pvrgpu_unmap_msaa_sample0(&transfer);
            CHECK(!transfer.sample0_staging && transfer.sample0_staging_size == 0);
            CHECK(memcmp(resource.data, expected, resource.size) == 0);
            // Double cleanup is harmless; map lifetimes own their staging.
            pvrgpu_unmap_msaa_sample0(&transfer);
         }
         for (unsigned flag = PIPE_MAP_DIRECTLY; flag <= PIPE_MAP_COHERENT; flag <<= 1) {
            if (flag != PIPE_MAP_DIRECTLY && flag != PIPE_MAP_PERSISTENT &&
                flag != PIPE_MAP_COHERENT)
               continue;
            struct pvrgpu_transfer transfer = {0};
            transfer.base.resource = &resource.base;
            transfer.base.box = (struct pipe_box){.width = 1, .height = 1, .depth = 1};
            transfer.base.usage = PIPE_MAP_WRITE | flag;
            CHECK(!pvrgpu_map_msaa_sample0(&transfer) && !transfer.sample0_staging);
         }
         struct pvrgpu_transfer transfer = {0};
         transfer.base.resource = &resource.base;
         transfer.base.level = 1;
         transfer.base.box = (struct pipe_box){.x = 3, .y = 1, .z = 1,
            .width = 4, .height = 2, .depth = 2};
         transfer.base.usage = PIPE_MAP_READ;
         const uintptr_t old_offset = resource.level_offsets[1];
         resource.level_offsets[1] = SIZE_MAX;
         CHECK(!pvrgpu_map_msaa_sample0(&transfer) && !transfer.sample0_staging);
         resource.level_offsets[1] = old_offset;
         const uintptr_t old_layer_stride = resource.level_layer_strides[1];
         resource.level_layer_strides[1] = SIZE_MAX;
         CHECK(!pvrgpu_map_msaa_sample0(&transfer) && !transfer.sample0_staging);
         resource.level_layer_strides[1] = old_layer_stride;
         const unsigned old_stride = resource.level_strides[1];
         resource.level_strides[1] = 1;
         CHECK(!pvrgpu_map_msaa_sample0(&transfer) && !transfer.sample0_staging);
         resource.level_strides[1] = old_stride;
         for (unsigned level = 0; level < 3; ++level) {
            const struct pipe_box box = {.x = 1,
               .y = level == 2 ? 0 : 1, .z = 1,
               .width = 3, .height = 2, .depth = 2};
            const unsigned source_stride = box.width * bpp + 5U;
            const size_t source_layer_stride = source_stride * box.height + 7U;
            const size_t source_size = source_layer_stride * box.depth;
            uint8_t *source = malloc(source_size);
            CHECK(source != NULL);
            for (unsigned mode = 0; mode < 3; ++mode) {
               for (size_t i = 0; i < resource.size; ++i)
                  resource.data[i] = (uint8_t)((i * 41U) ^ (i >> 9));
               for (size_t i = 0; i < source_size; ++i)
                  source[i] = (uint8_t)(0xa7U ^ (i * 59U));
               memcpy(expected, resource.data, resource.size);
               const unsigned flags = mode == 0 ? 0 : mode == 1 ?
                  PIPE_MAP_DISCARD_RANGE : PIPE_MAP_FLUSH_EXPLICIT;
               for (unsigned z = 0; z < (unsigned)box.depth; ++z)
                  for (unsigned y = 0; y < (unsigned)box.height; ++y)
                     for (unsigned x = 0; x < (unsigned)box.width; ++x) {
                        const size_t physical = resource.level_offsets[level] +
                           (box.z + z) * resource.level_layer_strides[level] +
                           (box.y + y) * resource.level_strides[level] +
                           (box.x + x) * bpp * samples;
                        memcpy(expected + physical,
                           source + z * source_layer_stride + y * source_stride + x * bpp,
                           bpp);
                     }
               CHECK(pvrgpu_texture_subdata_msaa_sample0(&resource.base, level,
                  flags, &box, source, source_stride, source_layer_stride));
               CHECK(memcmp(resource.data, expected, resource.size) == 0);
            }
            // Rejected source row/layer layouts and arithmetic overflow must
            // leave all storage intact, not partially copy the first row.
            CHECK(!pvrgpu_texture_subdata_msaa_sample0(&resource.base, level,
               0, &box, source, box.width * bpp - 1U, source_layer_stride));
            CHECK(!pvrgpu_texture_subdata_msaa_sample0(&resource.base, level,
               0, &box, source, source_stride, source_stride));
            CHECK(!pvrgpu_texture_subdata_msaa_sample0(&resource.base, level,
               0, &box, source, source_stride, UINTPTR_MAX));
            CHECK(!pvrgpu_texture_subdata_msaa_sample0(&resource.base, level,
               PIPE_MAP_READ, &box, source, source_stride, source_layer_stride));
            CHECK(!pvrgpu_texture_subdata_msaa_sample0(&resource.base, level,
               PIPE_MAP_DIRECTLY, &box, source, source_stride, source_layer_stride));
            CHECK(memcmp(resource.data, expected, resource.size) == 0);
            free(source);
         }
         // Single row/layer ignores the unused zero strides. Source aliases
         // destination storage, requiring snapshot-before-scatter semantics.
         const struct pipe_box alias_box = {.x = 1, .y = 1, .z = 1,
            .width = 3, .height = 1, .depth = 1};
         const size_t alias_offset = resource.level_offsets[1] +
            resource.level_layer_strides[1] + resource.level_strides[1] + bpp * samples;
         uint8_t snapshot[3 * 16];
         CHECK(3U * bpp <= sizeof(snapshot));
         memcpy(snapshot, resource.data + alias_offset, 3U * bpp);
         memcpy(expected, resource.data, resource.size);
         for (unsigned x = 0; x < 3; ++x)
            memcpy(expected + alias_offset + x * bpp * samples,
                   snapshot + x * bpp, bpp);
         CHECK(pvrgpu_texture_subdata_msaa_sample0(&resource.base, 1, 0,
            &alias_box, resource.data + alias_offset, 0, 0));
         CHECK(memcmp(resource.data, expected, resource.size) == 0);
         free(expected);
         FREE(resource.data);
      }
}

static void
test_sample_count_capabilities(void)
{
   struct pipe_resource resource = {0};
   resource.target = PIPE_TEXTURE_2D;
   resource.format = PIPE_FORMAT_R8G8B8A8_UNORM;
   resource.width0 = resource.height0 = 4;
   resource.depth0 = resource.array_size = 1;
   for (unsigned count = 0; count <= 17; ++count) {
      const bool storage_supported = count == 0 || count == 1 || count == 2 ||
                                     count == 4 || count == 8 || count == 16;
      resource.nr_samples = resource.nr_storage_samples = count;
      CHECK(pvrgpu_is_supported_resource_sample_count(count) == storage_supported);
      resource.bind = PIPE_BIND_RENDER_TARGET;
      CHECK(pvrgpu_can_create_texture_target(&resource) == storage_supported);
      resource.bind |= PIPE_BIND_SAMPLER_VIEW;
      CHECK(pvrgpu_can_create_texture_target(&resource) ==
            (storage_supported && count <= 8));
      resource.bind = PIPE_BIND_SHADER_IMAGE;
      CHECK(pvrgpu_can_create_texture_target(&resource) ==
            (storage_supported && count <= 8));
      resource.target = PIPE_TEXTURE_2D_ARRAY;
      resource.array_size = 3;
      CHECK(pvrgpu_can_create_texture_target(&resource) ==
            (storage_supported && count <= 8));
      resource.target = PIPE_TEXTURE_2D;
      resource.array_size = 1;
   }
   /* 兩個獨立欄位皆須有界，不能只檢查 API 的 requested samples。 */
   resource.bind = PIPE_BIND_SAMPLER_VIEW;
   resource.nr_samples = 4;
   resource.nr_storage_samples = 16;
   CHECK(!pvrgpu_can_create_texture_target(&resource));
   resource.nr_samples = 16;
   resource.nr_storage_samples = 4;
   CHECK(!pvrgpu_can_create_texture_target(&resource));
   resource.nr_samples = 3;
   CHECK(!pvrgpu_can_create_texture_target(&resource));

   /* Proxy 是 requested count 容量詢問，actual create 不可存入 3/5 samples。 */
   for (unsigned requested = 1; requested <= 8; ++requested) {
      struct pipe_resource probe;
      resource.bind = 0;
      resource.nr_samples = resource.nr_storage_samples = requested;
      CHECK(pvrgpu_normalize_proxy_texture_samples(&resource, &probe));
      unsigned physical = requested <= 2 ? requested : requested <= 4 ? 4 : 8;
      CHECK(probe.nr_samples == physical && probe.nr_storage_samples == physical);
      CHECK(probe.bind == PIPE_BIND_SAMPLER_VIEW);
      CHECK(pvrgpu_can_create_texture_target(&probe));
      CHECK(resource.nr_samples == requested && resource.nr_storage_samples == requested);
      CHECK(pvrgpu_can_create_texture_target(&resource) == (requested == physical));
   }
   for (unsigned requested = 9; requested <= 17; ++requested) {
      struct pipe_resource probe;
      resource.bind = 0;
      resource.nr_samples = resource.nr_storage_samples = requested;
      CHECK(!pvrgpu_normalize_proxy_texture_samples(&resource, &probe));
   }
   struct pipe_resource probe;
   resource.bind = PIPE_BIND_RENDER_TARGET;
   resource.nr_samples = resource.nr_storage_samples = 16;
   CHECK(pvrgpu_normalize_proxy_texture_samples(&resource, &probe));
   CHECK(probe.nr_samples == 16 && probe.bind == PIPE_BIND_RENDER_TARGET);
   CHECK(pvrgpu_can_create_texture_target(&probe));
   resource.nr_samples = resource.nr_storage_samples = 3;
   CHECK(pvrgpu_normalize_proxy_texture_samples(&resource, &probe));
   CHECK(!pvrgpu_can_create_texture_target(&probe));
}

int main(void)
{
   test_sample_count_capabilities();
   test_float_resolve();
   test_integer_resolve();
   test_unorm_resolve();
   test_sample_copy_and_mask();
   test_srgb_resolve();
   test_bilinear_resolve();
   test_signed_integer_resolve_and_mask();
   test_depth_stencil_aspects();
   test_depth_float_scale_flip();
   test_depth_stencil_samples();
   test_depth_alias_snapshot();
   test_clipped_color_transform();
   test_msaa_sample0_map();
   if (!failures)
      puts("MSAA Mesa format blit tests passed");
   return failures ? 1 : 0;
}
