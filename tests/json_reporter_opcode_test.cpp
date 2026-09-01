// Stage-aware JsonReporter opcode evidence regression. Terrain's D3 vertex
// shader contains two ordinary SMP/WDF pairs; neighboring explicit-LOD and
// fragment interpolation operations remain fail-closed for vertex evidence.
#include "json_reporter.h"

#include <systemc>

#include <iostream>
#include <stdexcept>

namespace {

using pvrgpu::stub::ClassifyVertexPcoTextureEvidenceOpcode;
using pvrgpu::stub::PcoOpcode;
using pvrgpu::stub::VertexPcoTextureEvidenceClass;

void Check(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

} // namespace

int sc_main(int, char **) {
  try {
    Check(ClassifyVertexPcoTextureEvidenceOpcode(
              PcoOpcode::kTextureSample) ==
              VertexPcoTextureEvidenceClass::kTextureSample,
          "vertex SMP evidence was rejected");
    Check(ClassifyVertexPcoTextureEvidenceOpcode(
              PcoOpcode::kWaitDataFence) ==
              VertexPcoTextureEvidenceClass::kWaitDataFence,
          "vertex WDF evidence was rejected");
    Check(ClassifyVertexPcoTextureEvidenceOpcode(
              PcoOpcode::kTextureSampleLod) ==
              VertexPcoTextureEvidenceClass::kUnsupported,
          "unsupported vertex explicit-LOD sample was accepted");
    Check(ClassifyVertexPcoTextureEvidenceOpcode(
              PcoOpcode::kFloatInterpolatePerspective) ==
              VertexPcoTextureEvidenceClass::kUnsupported,
          "fragment interpolation was accepted as vertex evidence");
    std::cout << "json_reporter_opcode_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "json_reporter_opcode_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
