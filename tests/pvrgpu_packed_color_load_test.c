/* SPDX-License-Identifier: MIT */
/* The actual driver LOAD path, with unaligned/padded native 10-bit storage. */
#include "../src/gallium/drivers/pvrgpu/pvrgpu_context.c"

static unsigned checks;
#define CHECK(c) do { ++checks; if (!(c)) { \
   fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #c); exit(1); \
} } while (0)

static uint32_t word(unsigned target, unsigned layer, unsigned y, unsigned x,
                     bool bgr)
{
   const uint32_t r = (1 + x + 37 * y + 101 * layer + 211 * target) & 1023;
   const uint32_t g = (2 + 13 * x + 73 * y + 199 * layer + 83 * target) & 1023;
   const uint32_t b = (3 + 29 * x + 17 * y + 31 * layer + 157 * target) & 1023;
   const uint32_t a = (x + y + layer + target) & 3;
   return (bgr ? b : r) | (g << 10) | ((bgr ? r : b) << 20) | (a << 30);
}

static void run(enum pipe_format format, unsigned samples)
{
   const bool bgr = format == PIPE_FORMAT_B10G10R10A2_UNORM;
   const unsigned width = 5, height = 3, layers = 2;
   const size_t row_bytes = width * samples * 4;
   const size_t target_bytes = row_bytes * height * layers;
   struct pvrgpu_resource resources[4] = {0};
   struct pvrgpu_context ctx = {0};
   struct pvrgpu_array_primitive_draw recorded = {0};
   ctx.framebuffer.width = width;
   ctx.framebuffer.height = height;
   ctx.framebuffer.nr_cbufs = 4;
   recorded.command.render_target_count = 4;
   recorded.command.raster_samples = samples;
   recorded.command.framebuffer_layers = layers;
   for (unsigned target = 0; target < 4; ++target) {
      struct pvrgpu_resource *r = &resources[target];
      r->base.target = PIPE_TEXTURE_2D_ARRAY;
      r->base.format = format;
      r->base.width0 = width * 2;
      r->base.height0 = height * 2;
      r->base.depth0 = 1;
      r->base.array_size = 4;
      r->base.last_level = 1;
      r->base.nr_samples = r->base.nr_storage_samples = samples;
      r->level_count = 2;
      r->level_offsets[1] = 3;
      r->level_strides[1] = row_bytes + 7;
      r->level_layer_strides[1] = r->level_strides[1] * height + 5;
      r->size = 3 + 4 * r->level_layer_strides[1];
      r->data = malloc(r->size);
      CHECK(r->data != NULL);
      memset(r->data, 0xcd, r->size);
      for (unsigned layer = 0; layer < layers; ++layer)
      for (unsigned y = 0; y < height; ++y)
      for (unsigned x = 0; x < width * samples; ++x) {
         const uint32_t packed = word(target, layer, y, x, bgr);
         memcpy(r->data + 3 + (layer + 1) * r->level_layer_strides[1] +
                y * r->level_strides[1] + x * 4, &packed, sizeof(packed));
      }
      ctx.framebuffer.cbufs[target] = (struct pipe_surface){
         .texture = &r->base, .format = format, .level = 1,
         .first_layer = 1, .last_layer = 2};
   }
   const char *expected_format = bgr ? PVRGPU_DRIVER_COMMAND_FORMAT_BGRA10_A2 :
                                     PVRGPU_DRIVER_COMMAND_FORMAT_RGB10_A2;
   CHECK(!strcmp(pvrgpu_command_format_for_framebuffer(&ctx), expected_format));
   CHECK(pvrgpu_capture_initial_color_attachment(&ctx, &recorded));
   CHECK(recorded.command.initial_color_attachment_bytes_size == target_bytes * 4);
   CHECK(recorded.command.initial_color_attachment_bytes == recorded.initial_color_attachment_bytes);
   for (unsigned target = 0; target < 4; ++target) {
      /* Owned snapshots cannot refer to source pixels which may later change. */
      memset(resources[target].data, 0xa7, resources[target].size);
      for (unsigned layer = 0; layer < layers; ++layer)
      for (unsigned y = 0; y < height; ++y)
      for (unsigned x = 0; x < width * samples; ++x) {
         uint32_t actual;
         memcpy(&actual, recorded.initial_color_attachment_bytes + target * target_bytes +
                ((size_t)layer * height + y) * row_bytes + x * 4, sizeof(actual));
         CHECK(actual == word(target, layer, y, x, bgr));
      }
   }
   free(recorded.initial_color_attachment_bytes);
   recorded.initial_color_attachment_bytes = NULL;
   recorded.command.initial_color_attachment_bytes = NULL;
   recorded.command.initial_color_attachment_bytes_size = 0;
   /* Invalid final source must not publish a partial MRT snapshot. */
   resources[3].size = 1;
   CHECK(!pvrgpu_capture_initial_color_attachment(&ctx, &recorded));
   CHECK(!recorded.initial_color_attachment_bytes);
   CHECK(!recorded.command.initial_color_attachment_bytes_size);
   for (unsigned target = 0; target < 4; ++target) free(resources[target].data);
}

int main(void)
{
   for (unsigned samples = 1; samples <= 16; samples *= 2) {
      run(PIPE_FORMAT_R10G10B10A2_UNORM, samples);
      run(PIPE_FORMAT_B10G10R10A2_UNORM, samples);
   }
   printf("packed color LOAD: PASS (%u checks)\n", checks);
   return 0;
}
