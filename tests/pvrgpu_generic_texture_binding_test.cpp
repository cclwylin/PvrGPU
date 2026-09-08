/* SPDX-License-Identifier: MIT */
#include "shader/pco_iss.h"
#include <cstdio>
#include <stdexcept>

extern "C" void test_generic_texture_bindings(void);
extern "C" void check_generic_explicit_texture_program(const std::uint8_t *bytes,
    std::size_t size, unsigned kind)
{
   using namespace pvrgpu::stub;
   const auto program = DecodePcoProgram(ShaderStage::kFragment,
      std::vector<std::uint8_t>(bytes, bytes + size));
   unsigned seen = 0;
   for (const auto &instruction : program.instructions) {
      if (instruction.opcode != PcoOpcode::kTextureSample) continue;
      if (!instruction.texture_lod_replace ||
          instruction.texture_non_normalized_coords != (kind != 3) ||
          instruction.source1.bank != PcoRegisterBank::kShared || instruction.source1.index != 0 ||
          instruction.source2.bank != PcoRegisterBank::kShared || instruction.source2.index != 8)
         throw std::runtime_error("explicit texture compiler lost native LOD/coordinate/descriptor contract");
      ++seen;
   }
   if (seen != 1) throw std::runtime_error("explicit texture compiler lost or invented SMP");
}
extern "C" void check_generic_texture_program(const std::uint8_t *bytes,
    std::size_t size, const unsigned *units, unsigned count)
{
   using namespace pvrgpu::stub;
   const auto program = DecodePcoProgram(ShaderStage::kFragment,
      std::vector<std::uint8_t>(bytes, bytes + size));
   unsigned seen = 0;
   for (const auto &instruction : program.instructions) {
      if (instruction.opcode != PcoOpcode::kTextureSample)
         continue;
      if (seen >= count || instruction.source1.bank != PcoRegisterBank::kShared ||
          instruction.source1.index != units[seen] * kPcoTextureDescriptorDwordCount ||
          instruction.source2.bank != PcoRegisterBank::kShared ||
          instruction.source2.index != units[seen] * kPcoTextureDescriptorDwordCount + 8)
         throw std::runtime_error("generic SMP does not address its Gallium texture unit");
      ++seen;
   }
   if (seen != count)
      throw std::runtime_error("generic compiler lost or invented a texture sample");
}

int main()
{
   try {
      test_generic_texture_bindings();
      std::puts("generic texture binding compiler/ISS tests passed");
      return 0;
   } catch (const std::exception &error) {
      std::fprintf(stderr, "%s\n", error.what());
      return 1;
   }
}
