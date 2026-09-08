/* SPDX-License-Identifier: MIT */
/* Real Mesa-side descriptor words, independently decoded below. */
#include "../src/gallium/drivers/pvrgpu/pvrgpu_pco.c"

static unsigned checks;
#define CHECK(c) do { ++checks; if (!(c)) { \
   fprintf(stderr, "%s:%u: %s\n", __FILE__, __LINE__, #c); exit(1); \
} } while (0)

int main(void)
{
   for (unsigned layers = 1; layers <= 4; layers *= 2)
   for (unsigned mips = 1; mips <= 3; ++mips) {
      uint32_t rgba[20], bgra[20];
      CHECK(pvrgpu_pco_build_terrain_texture_descriptor(rgba,
         PIPE_FORMAT_R10G10B10A2_UNORM, 16, 8, mips, 4096,
         0, 0, 0, 2, 2, (mips - 1) * 64, layers, 2));
      CHECK(pvrgpu_pco_build_terrain_texture_descriptor(bgra,
         PIPE_FORMAT_B10G10R10A2_UNORM, 16, 8, mips, 4096,
         0, 0, 0, 2, 2, (mips - 1) * 64, layers, 2));
      const uint64_t rgb = rgba[0] | (uint64_t)rgba[1] << 32;
      const uint64_t bgr = bgra[0] | (uint64_t)bgra[1] << 32;
      CHECK(((rgb >> 27) & 127) == 14);
      CHECK(((bgr >> 27) & 127) == 14);
      CHECK(((rgb >> 14) & 7) == 0 && ((rgb >> 11) & 7) == 1 &&
            ((rgb >> 8) & 7) == 2 && ((rgb >> 5) & 7) == 3);
      CHECK(((bgr >> 14) & 7) == 2 && ((bgr >> 11) & 7) == 1 &&
            ((bgr >> 8) & 7) == 0 && ((bgr >> 5) & 7) == 3);
      CHECK((rgb ^ bgr) == ((UINT64_C(2) << 14) | (UINT64_C(2) << 8)));
      for (unsigned i = 1; i < 20; ++i) CHECK(rgba[i] == bgra[i]);
   }
   printf("packed color descriptors: PASS (%u checks)\n", checks);
   return 0;
}
