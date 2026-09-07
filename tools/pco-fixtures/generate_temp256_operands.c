/* Test-only register-encoding provenance. Compile against Mesa's generated
 * pco_isa.h (from public pco_isa.py), not the model's decoder. */
#include "pco/pco_isa.h"
#include <stdio.h>
#include <string.h>

int main(void)
{
   const unsigned registers[] = {0, 60, 63, 64, 124, 127, 128, 188,
                                 191, 192, 240, 252, 253, 254, 255, 256};
   for (unsigned i = 0; i < sizeof(registers) / sizeof(registers[0]); ++i) {
      uint8_t upper[3] = {0}, lower[3] = {0};
      _pco_src_1up_3b11i_encode(upper, (struct pco_src_1up_3b11i){
          .sb3 = PCO_REGBANK_TEMP, .s3 = registers[i]});
      _pco_src_1lo_3b11i_2m_encode(lower, (struct pco_src_1lo_3b11i_2m){
          .sb0 = PCO_REGBANK_TEMP, .s0 = registers[i], .is0 = PCO_IS0_SEL_S0});
      if (memcmp(upper, lower, sizeof(upper)) != 0)
         return 1;
      printf("{%u, {0x%02x, 0x%02x, 0x%02x}},\n", registers[i],
             upper[0], upper[1], upper[2]);
      if (registers[i] < 255) {
         uint8_t pair[3] = {0};
         _pco_dst_2_3b8i_3b8i_encode(pair, (struct pco_dst_2_3b8i_3b8i){
             .db0 = PCO_REGBANK_TEMP, .d0 = registers[i],
             .db1 = PCO_REGBANK_TEMP, .d1 = registers[i] + 1});
         printf("pair {%u, {0x%02x, 0x%02x, 0x%02x}},\n", registers[i],
                pair[0], pair[1], pair[2]);
      }
   }
   return 0;
}
