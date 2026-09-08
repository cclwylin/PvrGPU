// Stage-aware JsonReporter opcode evidence regression. Terrain's D3 vertex
// shader contains two ordinary SMP/WDF pairs; neighboring explicit-LOD and
// fragment interpolation operations remain fail-closed for vertex evidence.
#include "json_reporter.h"

#include <systemc>

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using pvrgpu::stub::ClassifyVertexPcoTextureEvidenceOpcode;
using pvrgpu::stub::ClassifyPcoShiftEvidenceOpcode;
using pvrgpu::stub::PcoShiftEvidenceClass;
using pvrgpu::stub::PcoOpcode;
using pvrgpu::stub::VertexPcoTextureEvidenceClass;
using pvrgpu::stub::CounterTxn;
using pvrgpu::stub::DrawListShaderStats;
using pvrgpu::stub::DrawListStats;
using pvrgpu::stub::ValidateDrawListShaderStatistics;

unsigned checks = 0;

void Check(bool condition, const char *message) {
  ++checks;
  if (!condition)
    throw std::runtime_error(message);
}

void RejectStatistics(const CounterTxn &counters,
                      const std::vector<DrawListStats> &drawlists,
                      const char *reason) {
  try {
    ValidateDrawListShaderStatistics(counters, drawlists);
  } catch (const std::exception &error) {
    Check(std::string(error.what()).find(reason) != std::string::npos,
          "statistics rejected for an unrelated reason");
    return;
  }
  Check(false, "invalid geometry statistics were accepted");
}

void GeometryTextureStatistics() {
  CounterTxn counters;
  counters.drawlists = 1;
  std::vector<DrawListStats> drawlists(1);
  auto &drawlist = drawlists[0];
  drawlist.vertex.program_recorded = drawlist.vertex.executions_recorded = 1;
  drawlist.fragment.program_recorded = drawlist.fragment.executions_recorded = 1;
  ValidateDrawListShaderStatistics(counters, drawlists);
  ++checks; // A draw with no GS and all GS fields zero is valid.

  // No stale static or dynamic evidence may survive a missing GS program.
  for (auto field : {&DrawListShaderStats::invocations,
                     &DrawListShaderStats::program_alu_instructions,
                     &DrawListShaderStats::program_tex_instructions,
                     &DrawListShaderStats::program_memory_instructions,
                     &DrawListShaderStats::executed_alu_instructions,
                     &DrawListShaderStats::executed_tex_instructions,
                     &DrawListShaderStats::executed_memory_instructions}) {
    drawlist.geometry.*field = 1;
    RejectStatistics(counters, drawlists, "incomplete geometry");
    drawlist.geometry.*field = 0;
  }
  for (auto field : {&DrawListShaderStats::program_groups,
                     &DrawListShaderStats::program_instructions}) {
    drawlist.geometry.*field = 1;
    RejectStatistics(counters, drawlists, "incomplete geometry");
    drawlist.geometry.*field = 0;
  }
  for (auto flags : {std::pair<unsigned,unsigned>{1,0}, {0,1}, {2,2}, {255,255}}) {
    drawlist.geometry.program_recorded = flags.first;
    drawlist.geometry.executions_recorded = flags.second;
    RejectStatistics(counters, drawlists, "incomplete geometry");
  }
  drawlist.geometry = {};
  counters.gs_tex_instructions = 1;
  RejectStatistics(counters, drawlists, "instruction totals mismatch");
  counters.gs_tex_instructions = 0;

  // Recorded zero-work / zero-emission draws are legal. Do not infer dynamic
  // work from static composition: conditional SMP may never execute.
  auto &gs = drawlist.geometry;
  gs.program_recorded = gs.executions_recorded = 1;
  ValidateDrawListShaderStatistics(counters, drawlists);
  ++checks;
  gs.program_groups = gs.program_instructions = 2;
  gs.program_tex_instructions = gs.program_memory_instructions = 1;
  ValidateDrawListShaderStatistics(counters, drawlists);
  ++checks;
  gs.invocations = counters.gs_invocations = 4;
  gs.executed_memory_instructions = counters.gs_memory_instructions = 4;
  ValidateDrawListShaderStatistics(counters, drawlists);
  ++checks;
  gs.executed_tex_instructions = counters.gs_tex_instructions = 4;
  counters.texture_requests = 4;
  ValidateDrawListShaderStatistics(counters, drawlists);
  Check(counters.gs_emitted_vertices == 0,
        "texture-only GS must not require vertex emission");

  for (auto field : {&CounterTxn::gs_invocations,
                     &CounterTxn::gs_alu_instructions,
                     &CounterTxn::gs_tex_instructions,
                     &CounterTxn::gs_memory_instructions}) {
    ++(counters.*field);
    RejectStatistics(counters, drawlists, "instruction totals mismatch");
    --(counters.*field);
  }
  ++gs.executed_tex_instructions;
  RejectStatistics(counters, drawlists, "instruction totals mismatch");
  --gs.executed_tex_instructions;

  // Ordered multiple physical draws must conserve all GS instruction classes.
  drawlists.push_back(drawlist);
  drawlists[1].drawlist_index = 1;
  counters.drawlists = 2;
  counters.gs_invocations *= 2;
  counters.gs_tex_instructions *= 2;
  counters.gs_memory_instructions *= 2;
  ValidateDrawListShaderStatistics(counters, drawlists);
  ++checks;
  drawlists[0].geometry.executed_tex_instructions =
      std::numeric_limits<std::uint64_t>::max();
  RejectStatistics(counters, drawlists, "counter overflow");
}

} // namespace

int sc_main(int, char **) {
  try {
    GeometryTextureStatistics();
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
    Check(ClassifyPcoShiftEvidenceOpcode(PcoOpcode::kShiftRight) ==
              PcoShiftEvidenceClass::kShiftRight,
          "SHR reporter evidence was rejected");
    Check(ClassifyPcoShiftEvidenceOpcode(PcoOpcode::kShiftLeft) ==
              PcoShiftEvidenceClass::kShiftLeft,
          "LSL reporter evidence was rejected");
    Check(ClassifyPcoShiftEvidenceOpcode(PcoOpcode::kBitwiseOr) ==
              PcoShiftEvidenceClass::kUnsupported,
          "non-shift opcode was accepted as shift reporter evidence");
    std::cout << "json_reporter_opcode_test: PASS " << checks << " checks\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "json_reporter_opcode_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
