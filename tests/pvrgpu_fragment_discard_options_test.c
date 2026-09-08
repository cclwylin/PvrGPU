/* SPDX-License-Identifier: MIT */
/* Read the actual screen initialization, not a copied "correct" option. */
#include "../src/gallium/drivers/pvrgpu/pvrgpu_screen.c"

const nir_shader_compiler_options *pvrgpu_discard_test_options(void);
const nir_shader_compiler_options *
pvrgpu_discard_test_options(void)
{
   struct pipe_screen screen = {0};
   pvrgpu_init_shader_caps(&screen);
   return screen.nir_options[MESA_SHADER_FRAGMENT];
}
