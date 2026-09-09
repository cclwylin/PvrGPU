/* SPDX-License-Identifier: MIT */
/* Genuine independent CubeArray implicit/dynamic LOD and 2DArray programs.
 * Set PVRGPU_TESS_FIXTURE_DIR. This invokes the pinned native compiler and
 * never authors instruction bytes or substitutes the Car Chase shader. */
#define PVRGPU_MANY_TEXTURES_NO_MAIN
#include "../../src/gallium/drivers/pvrgpu/tests/pvrgpu_pco_many_textures_test.c"
int main(void)
{
   glsl_type_singleton_init_or_ref();
   char error[2048] = {0};
   struct pvrgpu_pco_compiler *compiler = pvrgpu_pco_compiler_create(error, sizeof(error));
   require(compiler != NULL, error);
   for (unsigned kind = 0; kind < 3; ++kind)
      compile_array_coordinate_fixture(compiler, kind);
   pvrgpu_pco_compiler_destroy(compiler);
   glsl_type_singleton_decref();
   return 0;
}
