/* SPDX-License-Identifier: MIT */
/* Run only the native colour LOAD transport checks from the larger texture
 * snapshot fixture, so unrelated sampler admission changes cannot mask them. */
int pvrgpu_texture_view_snapshot_full_main(void);
#define main pvrgpu_texture_view_snapshot_full_main
#include "pvrgpu_texture_view_snapshot_test.c"
#undef main

int
main(void)
{
   const float snorm[4] = {-0.5f, 0.75f, 0.25f, 0.125f};
   const float unorm16[4] = {0.50001f, 0.75f, 0.25f, 0.125f};
   const float half[4] = {1.0006f, 7.0f, 8.0f, 9.0f};
   test_canonical_color_initial_snapshot(PIPE_FORMAT_R8_SNORM, snorm);
   test_canonical_color_initial_snapshot(PIPE_FORMAT_R8_UNORM, unorm16);
   test_canonical_color_initial_snapshot(PIPE_FORMAT_R8G8_UNORM, unorm16);
   test_canonical_color_initial_snapshot(PIPE_FORMAT_R5G6B5_UNORM, unorm16);
   test_canonical_color_initial_snapshot(PIPE_FORMAT_R16_UNORM, unorm16);
   test_canonical_color_initial_snapshot(PIPE_FORMAT_R16_FLOAT, half);
   test_unorm32_color_initial_snapshot();
   printf("color codec snapshot: PASS (%u checks)\n", snapshot_checks);
   return 0;
}
