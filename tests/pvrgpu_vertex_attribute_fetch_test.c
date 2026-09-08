/* SPDX-License-Identifier: MIT */
/* Exercise the actual Gallium reader with real Mesa format descriptors. */
#include "../src/gallium/drivers/pvrgpu/pvrgpu_context.c"

static unsigned checks;
#define CHECK(condition) do { ++checks; if (!(condition)) { \
   fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #condition); return 1; \
} } while (0)

static int test_index_reader(void)
{
   for (unsigned width = 1; width <= 4; width *= 2) {
      for (unsigned size = 0; size <= 8 * width; ++size) {
         /* Each advertised span ends at the ASan allocation boundary. The
          * deliberately unaligned pointer catches accidental typed loads;
          * partial trailing indices must not read past this exact span. */
         uint8_t *allocation = malloc((size_t)size + 1);
         CHECK(allocation != NULL);
         memset(allocation, 0xad, (size_t)size + 1);
         for (unsigned i = 0; i < size / width; ++i) {
            const uint32_t value = 129 + i * 17;
            if (width == 1) allocation[1 + i] = (uint8_t)value;
            else if (width == 2) {
               const uint16_t narrow = (uint16_t)value;
               memcpy(allocation + 1 + i * width, &narrow, sizeof(narrow));
            } else memcpy(allocation + 1 + i * width, &value, sizeof(value));
         }
         struct pvrgpu_resource resource = {0};
         resource.base.target = PIPE_BUFFER;
         resource.base.width0 = resource.size = size;
         resource.data = allocation + 1;
         struct pipe_draw_info info = {0};
         info.index_size = width;
         info.index.resource = &resource.base;
         for (unsigned start = 0; start < 10; ++start) {
            for (unsigned occurrence = 0; occurrence < 10; ++occurrence) {
               uint32_t result[] = {0x11223344, 0xdeadbeef, 0x55667788};
               CHECK(pvrgpu_read_draw_index(&info, start, occurrence, result + 1));
               CHECK(result[1] == (start + occurrence < size / width ?
                                      129 + (start + occurrence) * 17 : 0));
               CHECK(result[0] == 0x11223344 && result[2] == 0x55667788);
            }
         }
         const unsigned large[][2] = {
            {UINT32_MAX, 0}, {UINT32_MAX, 1}, {UINT32_MAX, UINT32_MAX}
         };
         for (unsigned i = 0; i < ARRAY_SIZE(large); ++i) {
            uint32_t result = 0xdeadbeef;
            CHECK(pvrgpu_read_draw_index(&info, large[i][0], large[i][1], &result));
            CHECK(result == 0); /* start + occurrence must not wrap to zero. */
         }
         CHECK(allocation[0] == 0xad);
         for (unsigned i = (size / width) * width; i < size; ++i)
            CHECK(allocation[i + 1] == 0xad);
         free(allocation);
      }
   }
   struct pipe_draw_info info = {0};
   info.index_size = 4;
   uint32_t output = 0xdeadbeef;
   CHECK(!pvrgpu_read_draw_index(NULL, 0, 0, &output));
   CHECK(!pvrgpu_read_draw_index(&info, 0, 0, &output));
   struct pvrgpu_resource resource = {0};
   resource.base.target = PIPE_BUFFER;
   resource.base.width0 = resource.size = 4;
   info.index.resource = &resource.base;
   CHECK(!pvrgpu_read_draw_index(&info, 0, 0, &output));
   CHECK(output == 0xdeadbeef); /* missing backing never reports raw-zero success */
   const uint32_t value = UINT32_MAX;
   resource.data = (uint8_t *)&value;
   CHECK(!pvrgpu_read_draw_index(&info, 0, 0, NULL));
   for (unsigned width = 0; width <= 7; ++width) {
      if (width == 1 || width == 2 || width == 4) continue;
      info.index_size = width;
      CHECK(!pvrgpu_read_draw_index(&info, 0, 0, &output));
      CHECK(output == 0xdeadbeef);
   }
   info.index_size = 4;
   CHECK(pvrgpu_read_draw_index(&info, 0, 0, &output) && output == UINT32_MAX);
   /* The pre-existing client-pointer path still reads valid source bytes;
    * no unknown client-memory extent is invented as a resource bound. */
   info.has_user_indices = true;
   info.index.user = &value;
   CHECK(pvrgpu_read_draw_index(&info, 0, 0, &output) && output == UINT32_MAX);
   info.index.user = NULL;
   output = 0xdeadbeef;
   CHECK(!pvrgpu_read_draw_index(&info, 0, 0, &output) && output == 0xdeadbeef);

   /* Exercise the thin caller above the reader too: truncated EBO bytes must
    * not turn an original six-element draw into an empty or shorter draw. */
   uint16_t source_indices[] = {3, 2, 1};
   resource.data = (uint8_t *)source_indices;
   resource.base.width0 = resource.size = sizeof(source_indices);
   info.has_user_indices = false;
   info.index.resource = &resource.base;
   info.index_size = 2;
   info.mode = MESA_PRIM_TRIANGLES;
   struct pipe_draw_start_count_bias draw = {0};
   draw.count = 6;
   draw.index_bias = 2;
   unsigned retained = 0;
   uint32_t maximum = 0;
   uint8_t *snapshot = pvrgpu_copy_draw_indices(&info, &draw, draw.count,
                                                &retained, &maximum);
   CHECK(snapshot != NULL && retained == 6 && maximum == 3 && draw.count == 6);
   const uint16_t expected[] = {3, 2, 1, 0, 0, 0};
   CHECK(memcmp(snapshot, expected, sizeof(expected)) == 0);
   uint32_t first = 0, vertices = 0;
   CHECK(pvrgpu_rebase_vertex_indices(snapshot, sizeof(expected), 2, retained,
                                       draw.index_bias, &first, &vertices));
   CHECK(first == 2 && vertices == 4); /* raw out-of-range 0 fetches source vertex 2 */
   CHECK(memcmp(snapshot, expected, sizeof(expected)) == 0);
   CHECK(memcmp(source_indices, expected, sizeof(source_indices)) == 0);
   free(snapshot);
   return 0;
}

int main(void)
{
   if (test_index_reader()) return 1;
   uint32_t backing[32];
   for (unsigned i = 0; i < ARRAY_SIZE(backing); ++i)
      backing[i] = 0x3f000000u + i * 997u;
   const enum pipe_format formats[] = {
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R32G32B32A32_SINT,
      PIPE_FORMAT_R32G32B32A32_UINT, PIPE_FORMAT_R32G32_FLOAT,
      PIPE_FORMAT_R8G8B8A8_UNORM
   };
   const unsigned vertices[] = {0, 1, 2, 11, 1024, UINT32_MAX};
   struct pvrgpu_resource resource = {0};
   resource.base.target = PIPE_BUFFER;
   resource.base.width0 = resource.size = sizeof(backing);
   resource.data = (uint8_t *)backing;
   struct pvrgpu_context ctx = {0};
   ctx.num_vertex_buffers = 1;
   ctx.vertex_buffers[0].buffer.resource = &resource.base;
   ctx.vertex_buffers[0].buffer_offset = 8;
   struct pipe_vertex_element element = {0};
   element.src_offset = 4;
   for (unsigned user = 0; user < 2; ++user) {
      ctx.vertex_buffers[0].is_user_buffer = user;
      if (user) ctx.vertex_buffers[0].buffer.user = backing;
      else ctx.vertex_buffers[0].buffer.resource = &resource.base;
      for (unsigned f = 0; f < ARRAY_SIZE(formats); ++f) {
         element.src_format = formats[f];
         const unsigned size = util_format_get_blocksize(formats[f]);
         const unsigned words = DIV_ROUND_UP(size, 4);
         element.src_stride = 0;
         for (unsigned v = 0; v < ARRAY_SIZE(vertices); ++v) {
            uint32_t output[6] = {0xaabbccdd, 0, 0, 0, 0, 0xeeff0011};
            const char *reason = NULL;
            CHECK(pvrgpu_read_vertex_attribute(&ctx, &element, vertices[v],
                                                words, output + 1, &reason));
            CHECK(memcmp(output + 1, (uint8_t *)backing + 12, size) == 0);
            CHECK(output[0] == 0xaabbccdd && output[5] == 0xeeff0011);
         }
         // Nonzero Gallium stride is already normalized, including the GL
         // tightly-packed array shorthand. It must still advance normally.
         element.src_stride = size;
         for (unsigned v = 0; v < 4; ++v) {
            uint32_t output[4] = {0};
            CHECK(pvrgpu_read_vertex_attribute(&ctx, &element, v,
                                                words, output, NULL));
            CHECK(memcmp(output, (uint8_t *)backing + 12 + v * size, size) == 0);
         }
      }
   }
   ctx.vertex_buffers[0].is_user_buffer = false;
   ctx.vertex_buffers[0].buffer.resource = &resource.base;
   element.src_format = PIPE_FORMAT_R32G32B32A32_FLOAT;
   element.src_stride = 16;
   uint32_t output[4] = {1, 2, 3, 4};
   CHECK(pvrgpu_read_vertex_attribute(&ctx, &element, UINT32_MAX, 4, output, NULL));
   CHECK(output[0] == 0 && output[1] == 0 && output[2] == 0 && output[3] == 0);
   printf("pvrgpu-vertex-attribute-fetch-test: %u checks PASS\n", checks);
   return 0;
}
