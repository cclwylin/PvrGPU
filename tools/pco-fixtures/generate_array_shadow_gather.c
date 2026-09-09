/* SPDX-License-Identifier: MIT */
/* Genuine native 2D raw gather (500), array-shadow gather (501), and 2D
 * shadow gather (502). Inputs SH20=u, SH21=v, SH22=layer, SH23=reference.
 * Uses the same pinned compiler entry and NIR builder as the unit test.
 * Set PVRGPU_TESS_FIXTURE_DIR; no instruction bytes are authored here. */
#define PVRGPU_SHADOW_GATHER_NO_MAIN
#include "../../src/gallium/drivers/pvrgpu/tests/pvrgpu_pco_shadow_gather_test.c"
int main(void)
{
   glsl_type_singleton_init_or_ref();
   char error[2048] = {0};
   struct pvrgpu_pco_compiler *compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
   require(compiler != NULL, error);
   for (unsigned kind = 0; kind < 3; ++kind)
      compile_shadow_gather_fixture(compiler, kind);
   pvrgpu_pco_compiler_destroy(compiler);
   glsl_type_singleton_decref();
   return 0;
}
