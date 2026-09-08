/* SPDX-License-Identifier: MIT */
/* Actual packed readback store: no second quantization or channel reorder. */
#include "../src/gallium/drivers/pvrgpu/pvrgpu_resource.c"

static unsigned checks;
#define CHECK(c) do { ++checks; if (!(c)) { \
   fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #c); exit(1); \
} } while (0)

int main(void)
{
   const enum pipe_format formats[] = {
      PIPE_FORMAT_R10G10B10A2_UNORM, PIPE_FORMAT_B10G10R10A2_UNORM};
   for (unsigned format_index = 0; format_index < ARRAY_SIZE(formats); ++format_index) {
      const enum pipe_format format = formats[format_index];
      CHECK(pvrgpu_resource_readback_bytes_per_pixel(format) == 4);
      for (unsigned alignment = 0; alignment < 4; ++alignment)
      for (unsigned a = 0; a < 4; ++a) {
         uint8_t input[1024 * 4 + 8], output[1024 * 4 + 8];
         memset(input, 0x8d, sizeof(input));
         memset(output, 0xcb, sizeof(output));
         for (unsigned r = 0; r < 1024; ++r) {
            const unsigned g = (r * 13 + 2) & 1023;
            const unsigned b = (r * 73 + 3) & 1023;
            const uint32_t packed = (format_index ? b : r) | (g << 10) |
               ((format_index ? r : b) << 20) | (a << 30);
            memcpy(input + alignment + r * 4, &packed, 4);
         }
         pvrgpu_resource_readback_store_row(format, output + alignment,
                                             input + alignment, 1024);
         CHECK(!memcmp(input + alignment, output + alignment, 1024 * 4));
         for (unsigned i = 0; i < alignment; ++i) CHECK(output[i] == 0xcb);
         for (unsigned i = alignment + 1024 * 4; i < sizeof(output); ++i)
            CHECK(output[i] == 0xcb);
         for (unsigned r = 0; r < 1024; ++r) {
            uint32_t packed;
            memcpy(&packed, output + alignment + r * 4, 4);
            CHECK((format_index ? (packed >> 20) & 1023 : packed & 1023) == r);
            CHECK(((packed >> 10) & 1023) == ((r * 13 + 2) & 1023));
            CHECK((format_index ? packed & 1023 : (packed >> 20) & 1023) == ((r * 73 + 3) & 1023));
            CHECK((packed >> 30) == a);
         }
      }
   }
   /* Legacy normalized transport is unchanged. */
   const uint8_t rgba[4] = {1, 2, 3, 255};
   uint8_t output[4] = {0};
   pvrgpu_resource_readback_store_row(PIPE_FORMAT_R8G8B8A8_UNORM, output, rgba, 1);
   CHECK(!memcmp(output, rgba, 4));
   printf("packed color store: PASS (%u checks)\n", checks);
   return 0;
}
