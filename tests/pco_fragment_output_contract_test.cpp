// SPDX-License-Identifier: MIT
// Consume actual pvrgpu_pco_fragment_output_test compiler output, not encoded fixtures.
#include "shader/pco_iss.h"
#include <fstream>
#include <iterator>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  using namespace pvrgpu::stub;
  try {
    const unsigned masks[] = {0,0,1,3,7,15,0x703,15,7};
    unsigned checks = 0;
    for (unsigned kind = 1; kind < 9; ++kind) {
      std::ifstream stream(std::string(argv[1]) + "/output-" + std::to_string(kind) + ".bin", std::ios::binary);
      if (!stream) throw std::runtime_error("missing actual compiler binary");
      const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(stream), {}};
      const auto program = DecodePcoProgram(ShaderStage::kFragment, bytes);
      const bool depth = kind == 1 || kind == 8;
      if (program.summary.pixel_output_mask != masks[kind] || program.summary.writes_depth != depth)
        throw std::runtime_error("real compiler PIXOUT/depth summary mismatch");
      ++checks;
      const auto execution = ExecuteFragmentPco(program.summary, program.instructions);
      if (execution.suspended || execution.written_mask != masks[kind] ||
          execution.depth_written != depth || (depth && execution.depth != 0x3e800000U))
        throw std::runtime_error("real compiler execution fabricated or lost an output");
      ++checks;
      std::cout << "kind=" << kind << " mask=" << program.summary.pixel_output_mask
                << " depth=" << unsigned(program.summary.writes_depth) << " PASS\n";
    }
    std::cout << "Real generic fragment output contract: " << checks << " checks PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
