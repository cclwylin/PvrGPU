// SPDX-License-Identifier: MIT
#include "shader/pco_iss.h"
#include "pco_texture_branch_fixture.h"
#include <array>
#include <fstream>
#include <iterator>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
using namespace pvrgpu::stub;
unsigned checks = 0;
void Check(bool value, const char *why) { ++checks; if (!value) throw std::runtime_error(why); }
template<class F> void Reject(F function, const char *why) {
  bool rejected = false; try { function(); } catch (const std::exception &) { rejected = true; }
  Check(rejected, why);
}
void RejectPredicatedCheckpoints(const PcoDecodedProgram &p,
                                 const PcoFragmentExecutionContext &context) {
  for (std::size_t pc = 0; pc < p.instructions.size(); ++pc) {
    const auto op = p.instructions[pc].opcode;
    if (op != PcoOpcode::kTextureSample && op != PcoOpcode::kDerivativeX &&
        op != PcoOpcode::kDerivativeY) continue;
    for (unsigned condition : {1U,2U,3U}) {
      auto instructions = p.instructions;
      instructions[pc].exec_cnd = condition;
      Reject([&] { ExecuteFragmentPco(p.summary,instructions,context); },
             "raw API admitted a predicated checkpoint");
      Reject([&] { PcoPreparedFragmentProgram invalid(p.summary,instructions); },
             "prepared API admitted a predicated checkpoint");
    }
  }
}
void DerivativeSummaryParity() {
  // Existing native derivative fixture from pco_prepared_fragment_test.cpp.
  // In particular the summary's derivative hint is not an admission proof.
  std::istringstream text(
      "35 82 00 87 80 08 00 00 00 40 "
      "34 8a 00 87 40 00 00 20 "
      "34 82 00 88 40 00 00 41 "
      "34 8a 00 87 41 00 00 21 "
      "34 82 00 89 40 00 00 42 "
      "34 8a 00 87 42 00 00 22 "
      "35 8a 80 87 80 01 00 00 00 23");
  std::vector<std::uint8_t> bytes;
  unsigned byte = 0;
  while (text >> std::hex >> byte) bytes.push_back(byte);
  auto p = DecodePcoProgram(ShaderStage::kFragment,bytes);
  p.summary.uses_derivatives = 0;
  const PcoPreparedFragmentProgram prepared(p.summary,p.instructions);
  PcoFragmentExecutionContext raw_context;
  raw_context.shared_count = 1;
  raw_context.shared_registers[0] = 0x3f800000;
  RejectPredicatedCheckpoints(p,raw_context);
  auto prepared_context = raw_context;
  for (unsigned step = 0; step < 3; ++step) {
    const auto raw = ExecuteFragmentPco(p.summary,p.instructions,raw_context);
    const auto owned = ExecuteFragmentPco(prepared,prepared_context);
    Check(raw.suspended == (step < 2) && owned.suspended == raw.suspended &&
          owned.pixel_outputs == raw.pixel_outputs && owned.written_mask == raw.written_mask &&
          owned.native_steps == raw.native_steps &&
          owned.executed_instructions.alu == raw.executed_instructions.alu &&
          owned.executed_instructions.texture == raw.executed_instructions.texture,
          "summary derivative hint changed raw/prepared execution");
    if (!raw.suspended) {
      Check(raw.pixel_outputs[0] == 0x3f800000 && raw.pixel_outputs[1] == 0x3e800000 &&
            raw.pixel_outputs[2] == 0x3f000000 && raw.written_mask == 15,
            "derivative response publication/output mismatch");
      break;
    }
    Check(raw.derivative_request_valid && owned.derivative_request_valid &&
          raw.continuation.temporary_written_mask == owned.continuation.temporary_written_mask,
          "derivative checkpoint raw/prepared mask mismatch");
    raw_context.continuation = raw.continuation;
    prepared_context.continuation = owned.continuation;
    raw_context.derivative_response_valid = prepared_context.derivative_response_valid = 1;
    raw_context.derivative_response = prepared_context.derivative_response =
        step == 0 ? 0x3e800000 : 0x3f000000;
    for (unsigned mutation = 0; mutation < 4; ++mutation) {
      auto bad = raw_context;
      switch (mutation) {
        case 0: bad.continuation.temporary_written_mask.words[0] &= ~UINT64_C(1); break;
        case 1: bad.continuation.temporary_written_mask.set(255); break;
        case 2: bad.continuation.temporary_written_mask.set(raw.continuation.pending_output_index); break;
        case 3: bad.continuation.execution_predicate = 0; break;
      }
      Reject([&] { ExecuteFragmentPco(p.summary,p.instructions,bad); },
             "raw derivative API accepted invalid saved state");
      Reject([&] { ExecuteFragmentPco(prepared,bad); },
             "prepared derivative API accepted invalid saved state");
    }
  }
}
void Exercise(std::uint32_t choice) {
  const auto p = DecodePcoProgram(ShaderStage::kFragment, test::TextureBranchFixture());
  const PcoPreparedFragmentProgram prepared(p.summary,p.instructions);
  PcoFragmentExecutionContext context; context.shared_count = 23;
  context.shared_registers[20] = 0x3e800000; // 0.25
  context.shared_registers[21] = 0x3f000000; // 0.5
  context.shared_registers[22] = choice;
  if (choice == 0) RejectPredicatedCheckpoints(p,context);
  const auto raw = ExecuteFragmentPco(p.summary,p.instructions,context);
  const auto fast = ExecuteFragmentPco(prepared,context);
  const unsigned checkpoint = choice ? 12 : 23;
  Check(raw.suspended && fast.suspended && raw.continuation.resume_instruction_index == checkpoint &&
        fast.continuation.resume_instruction_index == checkpoint, "real branch selected wrong checkpoint");
  Check(raw.continuation.temporary_written_mask.words[0] == 0x21f &&
        fast.continuation.temporary_written_mask == raw.continuation.temporary_written_mask,
        "skipped THEN writes must not be synthesized in ELSE continuation");
  Check(raw.texture_request.coordinates[0] == (choice ? 0x3e000000U : 0x3f000000U) &&
        raw.texture_request.coordinates[1] == (choice ? 0x3e800000U : 0x3f400000U),
        "native branch arithmetic did not select actual input-derived UV");
  Check(raw.executed_instructions.texture == 1 && fast.executed_instructions.texture == 1 &&
        raw.written_mask == 0, "branch suspension issued fabricated texture/output work");
  const std::array<std::uint32_t,4> response{0x3e000000,0x3e800000,0x3f000000,0x3f400000};
  const auto done = ResumeFragmentPco(p.summary,p.instructions,raw.continuation,response);
  const auto fast_done = ResumeFragmentPco(prepared,fast.continuation,response);
  Check(!done.suspended && !fast_done.suspended && done.written_mask == 15 &&
        done.pixel_outputs == fast_done.pixel_outputs &&
        std::equal(response.begin(),response.end(),done.pixel_outputs.begin()),
        "SMP/WDF branch must resume and export exactly the actual response");
  Check(done.executed_instructions.texture == 1 && fast_done.executed_instructions.texture == 1 &&
        done.executed_instructions.alu == fast_done.executed_instructions.alu &&
        done.native_steps == fast_done.native_steps, "branch resumption duplicated native work");
  // All six definitely written registers remain mandatory, including the
  // CND control register r9 that is only consumed after the checkpoint.
  for (unsigned reg : {0U,1U,2U,3U,4U,9U}) {
    auto bad = raw.continuation;
    bad.temporary_written_mask.words[0] &= ~(UINT64_C(1) << reg);
    Reject([&] { ResumeFragmentPco(prepared,bad,response); }, "missing dominating TEMP accepted");
    Reject([&] { ResumeFragmentPco(p.summary,p.instructions,bad,response); }, "raw API accepted missing dominating TEMP");
  }
  for (unsigned mutation = 0; mutation < 10; ++mutation) {
    auto bad = raw.continuation;
    switch (mutation) {
      case 0: bad.temporary_written_mask.set(255); break;
      case 1: bad.pending_output_index++; break;
      case 2: bad.pending_component_count = 3; break;
      case 3: bad.resume_instruction_index--; break;
      case 4: bad.program_signature++; break;
      case 5: bad.execution_predicate = 0; break;
      case 6: bad.execution_predicate = 2; break;
      case 7: bad.written_mask = 1; break;
      case 8: bad.depth_written = 1; break;
      case 9: bad.executed_instructions.texture = UINT64_MAX; break;
    }
    Reject([&] { ResumeFragmentPco(prepared,bad,response); }, "forged continuation state accepted");
    Reject([&] { ResumeFragmentPco(p.summary,p.instructions,bad,response); }, "raw API accepted forged continuation state");
  }
  // First THEN response includes unwritten r5; it must not be published at
  // issue time, before its matching WDF.
  if (choice) {
    auto bad = raw.continuation; bad.temporary_written_mask.set(5);
    Reject([&] { ResumeFragmentPco(prepared,bad,response); }, "current response became written before WDF");
  }
}
}
int main(int argc,char **argv) {
  try {
    if (argc > 2) return 2;
    if (argc == 2) {
      std::ifstream file(argv[1],std::ios::binary);
      const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(file),{}};
      Check(bytes == test::TextureBranchFixture(), "checked-in fixture differs from true PCO compilation");
    }
    for (std::uint32_t choice : {0U,1U,2U,UINT32_MAX}) Exercise(choice);
    DerivativeSummaryParity();
    std::cout << "Native branch continuation: " << checks << " checks PASS\n";
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
  return 0;
}
