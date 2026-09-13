/* SPDX-License-Identifier: MIT */
/* Exercise PIPE_BUFFER capture through the actual Gallium context code. */
#include "../src/gallium/drivers/pvrgpu/pvrgpu_context.c"

void
pvrgpu_counter_eventf(const char *event, const char *format, ...)
{
   (void)event;
   (void)format;
}

static unsigned checks;
#define CHECK(test) do { ++checks; if (!(test)) { \
   fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #test); exit(1); \
} } while (0)

static void
expect_texel(const struct pipe_sampler_view *view,
             const struct pvrgpu_resource *resource,
             const uint8_t *snapshot,
             unsigned element)
{
   uint32_t expected[4];
   util_format_unpack_rgba(view->format, expected,
      resource->data + view->u.buf.offset + (size_t)element * 4U, 1);
   CHECK(memcmp(snapshot + (size_t)element * 16U,
                expected, sizeof(expected)) == 0);
}

int
main(void)
{
   struct pvrgpu_resource resource = {0};
   struct pipe_sampler_view view = {0};
   struct pvrgpu_context ctx = {0};
   struct pvrgpu_systemc_pco_sequence_texture captured = {0};
   const mesa_shader_stage stage = MESA_SHADER_FRAGMENT;
   uint8_t *bytes = NULL;
   const char *reason = NULL;

   resource.base.target = PIPE_BUFFER;
   resource.base.format = PIPE_FORMAT_R8_UNORM;
   resource.size = 32U + 65537U;
   resource.base.width0 = resource.size;
   resource.data = malloc(resource.size);
   CHECK(resource.data != NULL);
   for (size_t i = 0; i < resource.size; ++i)
      resource.data[i] = (uint8_t)(i * 37U + 11U);

   view.texture = &resource.base;
   view.target = PIPE_BUFFER;
   view.format = PIPE_FORMAT_R8G8B8A8_UNORM;
   view.u.buf.offset = 16;
   view.u.buf.size = 513;
   view.swizzle_r = PIPE_SWIZZLE_X;
   view.swizzle_g = PIPE_SWIZZLE_Y;
   view.swizzle_b = PIPE_SWIZZLE_Z;
   view.swizzle_a = PIPE_SWIZZLE_W;
   ctx.sampler_views[stage][0] = &view;

   CHECK(pvrgpu_capture_generic_sequence_texture(&ctx, stage, 0, 7,
                                                  &captured, &bytes, &reason));
   CHECK(captured.texture_kind == 5 && captured.descriptor_set == 7);
   CHECK(captured.buffer_elements == 128 && captured.mip_count == 1);
   CHECK(captured.mip[0].width == 128 && captured.mip[0].height == 1);
   CHECK(captured.mip[0].row_pitch == 128 * 16 &&
         captured.bytes_size == 128 * 16);
   expect_texel(&view, &resource, bytes, 0);
   expect_texel(&view, &resource, bytes, captured.buffer_elements - 1);
   free(bytes);
   bytes = NULL;

   /* Rebinding a different aligned range must create a new immutable
    * snapshot. A final partial RGBA8 texel is ignored, per textureSize. */
   view.u.buf.offset = 32;
   view.u.buf.size = 65537;
   CHECK(pvrgpu_capture_generic_sequence_texture(&ctx, stage, 0, 0,
                                                  &captured, &bytes, &reason));
   CHECK(captured.buffer_elements == 16384);
   CHECK(captured.mip[0].width == 8192 && captured.mip[0].height == 2);
   CHECK(captured.mip[0].row_pitch == 8192 * 16 &&
         captured.bytes_size == 2 * 8192 * 16);
   expect_texel(&view, &resource, bytes, 0);
   expect_texel(&view, &resource, bytes, captured.buffer_elements - 1);
   free(bytes);
   bytes = NULL;

   view.u.buf.offset = 1;
   CHECK(!pvrgpu_capture_generic_sequence_texture(&ctx, stage, 0, 0,
                                                   &captured, &bytes, &reason));
   CHECK(bytes == NULL && !strcmp(reason, "buffer_view_range"));
   view.u.buf.offset = 32;
   view.u.buf.size = 3;
   CHECK(!pvrgpu_capture_generic_sequence_texture(&ctx, stage, 0, 0,
                                                   &captured, &bytes, &reason));
   CHECK(bytes == NULL && !strcmp(reason, "buffer_view_range"));
   view.u.buf.size = resource.size;
   CHECK(!pvrgpu_capture_generic_sequence_texture(&ctx, stage, 0, 0,
                                                   &captured, &bytes, &reason));
   CHECK(bytes == NULL && !strcmp(reason, "buffer_view_range"));

   free(resource.data);
   printf("texture buffer snapshot: PASS (%u checks)\n", checks);
   return 0;
}
