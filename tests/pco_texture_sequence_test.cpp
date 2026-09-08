// SPDX-License-Identifier: MIT
// Real per-lane ISS texture handshakes, independent of USC's historical
// nine-static-SMP admission gate. Responses are an explicit ISS test peripheral,
// not a claim to exercise TextureUnit filtering or render a correct image.
#include "shader/pco_iss.h"
#include "pco_texture_sequence_fixtures.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

using namespace pvrgpu::stub;
namespace {
unsigned checks = 0;
void Check(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
  ++checks;
}
std::uint32_t Bits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
template<class F> std::string Refuse(F function) {
  try { function(); }
  catch (const std::exception &error) { ++checks; return error.what(); }
  throw std::runtime_error("malformed texture continuation was accepted");
}

// Compare semantic fields, never potentially uninitialized C++ padding.
std::vector<std::uint64_t> Words(const PcoFragmentExecution &execution) {
  std::vector<std::uint64_t> result;
  const auto add = [&](auto value) { result.push_back(std::uint64_t(value)); };
  const auto array = [&](const auto &values) { for (auto value : values) add(value); };
  const auto counts = [&](const auto &value) { add(value.alu); add(value.texture); add(value.memory); };
  array(execution.pixel_outputs);
  const auto &request = execution.texture_request;
  array(request.coordinates); array(request.spatial_offsets);
  array(request.texture_state); array(request.sampler_state);
#define REQUEST(field) add(request.field)
  REQUEST(texture_address_lo); REQUEST(texture_address_hi); REQUEST(coordinate_count);
  REQUEST(component_count); REQUEST(descriptor_set); REQUEST(binding); REQUEST(dimension);
  REQUEST(normalized); REQUEST(fcnorm); REQUEST(sample_index); REQUEST(sample_index_present);
  REQUEST(data_request); REQUEST(explicit_lod); REQUEST(explicit_lod_present);
#undef REQUEST
  const auto &continuation = execution.continuation;
  array(continuation.temporaries); array(continuation.temporary_written_mask.words);
  array(continuation.pixel_outputs);
  for (const auto &loop : continuation.loops) { add(loop.start_pc); add(loop.count); }
#define SAVED(field) add(continuation.field)
  SAVED(program_binary_size); SAVED(program_instruction_count); SAVED(program_signature);
  SAVED(resume_instruction_index); SAVED(pending_output_index); SAVED(pending_component_count);
  SAVED(data_request); SAVED(valid); SAVED(kind); SAVED(written_mask); SAVED(depth);
  SAVED(depth_written); SAVED(predicate); SAVED(predicate_valid); SAVED(discarded);
  SAVED(loop_depth); SAVED(execution_predicate); SAVED(native_steps);
#undef SAVED
  counts(continuation.executed_instructions);
#define EXEC(field) add(execution.field)
  EXEC(executed_instruction_count); EXEC(native_steps); EXEC(written_mask);
  EXEC(texture_request_valid); EXEC(derivative_source); EXEC(derivative_request_valid);
  EXEC(suspended); EXEC(discarded); EXEC(depth); EXEC(depth_written);
#undef EXEC
  counts(execution.executed_instructions);
  return result;
}

std::array<std::uint32_t, 4> Response(unsigned lane, unsigned round) {
  // Every value and the tested sums are exact integers in binary32. All four
  // components distinguish the lane and round, exposing replay/lost responses.
  std::array<std::uint32_t, 4> result{};
  for (unsigned component = 0; component < 4; ++component)
    result[component] = Bits(float(1 + lane * 16 + round + component));
  return result;
}

void Negative(const PcoDecodedProgram &program,
              const PcoPreparedFragmentProgram &prepared,
              const PcoFragmentExecutionContext &context,
              const PcoFragmentExecution &checkpoint) {
  for (unsigned mutation = 0; mutation < 8; ++mutation) {
    auto corrupt = context;
    corrupt.continuation = checkpoint.continuation;
    corrupt.texture_response_valid = 1;
    corrupt.texture_response = Response(0, 0);
    switch (mutation) {
      case 0: corrupt.continuation.program_signature ^= 1; break;
      case 1: ++corrupt.continuation.pending_output_index; break;
      case 2: corrupt.continuation.pending_component_count = 3; break;
      case 3: corrupt.continuation.resume_instruction_index = 0; break;
      case 4: corrupt.texture_response_valid = 0; break;
      case 5: corrupt.continuation.native_steps = UINT64_C(10000000); break;
      case 6: corrupt.continuation.executed_instructions.texture = UINT64_MAX; break;
      case 7: corrupt.continuation.execution_predicate = 2; break;
    }
    const auto raw = Refuse([&] { ExecuteFragmentPco(program.summary, program.instructions, corrupt); });
    const auto owned = Refuse([&] { ExecuteFragmentPco(prepared, corrupt); });
    Check(raw == owned, "raw/prepared continuation refusal differs");
  }
}

std::array<std::uint64_t, 4> Exercise(
    const PcoDecodedProgram &program, PcoFragmentExecutionContext context,
    unsigned lane, unsigned samples, bool accumulated, bool has_loop) {
  const PcoPreparedFragmentProgram prepared(program.summary, program.instructions);
  auto raw = ExecuteFragmentPco(program.summary, program.instructions, context);
  auto owned = ExecuteFragmentPco(prepared, context);
  std::uint64_t visits = raw.executed_instruction_count;
  unsigned issued = 0;
  std::array<std::uint32_t, 4> expected{};
  std::array<float, 4> sums{};
  std::uint32_t first_pc = 0;
  while (true) {
    Check(Words(raw) == Words(owned), "raw/prepared native state differs");
    Check(raw.native_steps == visits, "checkpoint restarted or replayed native visits");
    Check(raw.executed_instructions.texture == issued + (raw.suspended ? 1U : 0U),
          "dynamic TEX counter does not match actual SMP handshakes");
    if (!raw.suspended) break;
    Check(issued < samples && raw.texture_request_valid == 1 &&
          raw.derivative_request_valid == 0 && raw.continuation.valid == 1,
          "unexpected or excess native texture checkpoint");
    const auto &request = raw.texture_request;
    Check(request.descriptor_set == 0 && request.binding == 0 &&
          request.component_count == 4 && request.coordinate_count == 2,
          "SMP descriptor or response footprint changed");
    for (unsigned word = 0; word < 4; ++word) {
      Check(request.texture_state[word] == context.shared_registers[word] &&
            request.sampler_state[word] == context.shared_registers[8 + word],
            "SMP did not transport the real shared descriptor");
    }
    const auto pc = raw.continuation.resume_instruction_index;
    Check(pc > 0 && pc < program.instructions.size() &&
          program.instructions[pc - 1].opcode == PcoOpcode::kTextureSample &&
          program.instructions[pc].opcode == PcoOpcode::kWaitDataFence,
          "continuation does not point at the native SMP/WDF boundary");
    if (has_loop) {
      if (!issued) first_pc = pc;
      Check(pc == first_pc, "dynamic loop did not revisit its single static SMP");
    }
    if (accumulated) {
      Check(request.coordinates[0] == Bits(float(lane) + float(issued)) &&
            request.coordinates[1] == Bits(float(lane + 1)) &&
            request.explicit_lod_present == 1 && request.explicit_lod == Bits(0),
            "Mesa shader sampled wrong coordinates, order, or explicit LOD");
    }
    if (!issued && !lane) Negative(program, prepared, context, raw);
    expected = Response(lane, issued);
    for (unsigned component = 0; component < 4; ++component)
      sums[component] += float(1 + lane * 16 + issued + component);
    auto raw_context = context;
    raw_context.continuation = raw.continuation;
    raw_context.texture_response_valid = 1;
    raw_context.texture_response = expected;
    auto prepared_context = raw_context;
    prepared_context.continuation = owned.continuation;
    raw = ExecuteFragmentPco(program.summary, program.instructions, raw_context);
    owned = ExecuteFragmentPco(prepared, prepared_context);
    visits += raw.executed_instruction_count;
    ++issued;
  }
  Check(issued == samples && raw.written_mask == 15 && !raw.discarded &&
        !raw.texture_request_valid && !raw.continuation.valid,
        "texture shader did not complete exact requests and all outputs");
  if (accumulated)
    for (unsigned component = 0; component < 4; ++component)
      expected[component] = Bits(sums[component]);
  Check(std::equal(expected.begin(), expected.end(), raw.pixel_outputs.begin()),
        "texture response was lost, replayed, or committed to wrong output");
  if (!has_loop) {
    const auto counts = CountPcoInstructions(program.instructions, true);
    Check(raw.executed_instructions.alu == counts.alu &&
          raw.executed_instructions.texture == counts.texture &&
          raw.executed_instructions.memory == counts.memory &&
          visits == program.instructions.size(),
          "straight-line native counts differ from independent instruction enumeration");
  }
  return {{visits, raw.executed_instructions.alu,
           raw.executed_instructions.texture, raw.executed_instructions.memory}};
}

void PublicSequence(unsigned samples) {
  auto bytes = FillTexNearestFragmentPcoBinary();
  // Repeat an existing genuine SMP/WDF byte pair; no source-level program name
  // or synthetic opcode is sent to the decoder/executor.
  const std::vector<std::uint8_t> pair(bytes.begin() + 128, bytes.begin() + 146);
  for (unsigned sample = 1; sample < samples; ++sample)
    bytes.insert(bytes.begin() + 146, pair.begin(), pair.end());
  const auto program = DecodePcoProgram(ShaderStage::kFragment, bytes);
  Check(CountPcoInstructions(program.instructions, true).texture == samples,
        "native duplicated sequence did not decode all SMP instructions");
  for (unsigned lane = 0; lane < 4; ++lane) {
    PcoFragmentExecutionContext context;
    context.shared_count = 20;
    for (unsigned word = 0; word < 20; ++word) context.shared_registers[word] = 0x10000000U + word;
    context.coefficient_count = 12;
    context.coefficients[2] = Bits(1);
    context.coefficients[6] = Bits(float(lane));
    context.coefficients[10] = Bits(float(lane + 1));
    Exercise(program, context, lane, samples, false, false);
  }
}

void MesaSequence(const std::string &directory, unsigned kind) {
  const auto base = directory + "/sequence-" + std::to_string(kind);
  std::ifstream binary(base + ".pco", std::ios::binary), abi(base + ".abi");
  Check(binary && abi, "missing compiled Mesa sequence fixture");
  const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(binary)), {});
  unsigned shareds = 0, start = 0, count = 0, coefficients = 0;
  abi >> shareds >> start >> count >> coefficients;
  Check(bool(abi) && shareds <= kPcoMaximumFragmentSharedCount &&
        count == 4 && start >= 20 && start + count <= shareds && coefficients == 4,
        "compiled Mesa sequence ABI does not match its real uniforms and descriptor");
  const auto program = DecodePcoProgram(ShaderStage::kFragment, bytes);
  const auto static_samples = CountPcoInstructions(program.instructions, true).texture;
  Check(static_samples == (kind == 0 ? 10U : kind == 1 ? 16U : 1U),
        "Mesa texture instruction count changed the intended test scenario");
  const bool loop = kind == 2;
  Check(std::any_of(program.instructions.begin(), program.instructions.end(),
          [](const auto &instruction) { return instruction.opcode == PcoOpcode::kBranch; }) == loop,
        "Mesa uniform loop lacks a real native branch or straight shader unexpectedly branches");
  const std::vector<int> iterations = loop ? std::vector<int>{-3, 0, 1, 2, 10, 16, 257}
                                         : std::vector<int>{kind == 0 ? 10 : 16};
  std::array<std::array<std::uint64_t, 4>, 4> zero_counts{}, one_counts{};
  for (int iteration : iterations)
    for (unsigned lane = 0; lane < 4; ++lane) {
      PcoFragmentExecutionContext context;
      context.shared_count = static_cast<std::uint16_t>(shareds);
      context.coefficient_count = static_cast<std::uint16_t>(coefficients);
      for (unsigned word = 0; word < 20; ++word) context.shared_registers[word] = 0x10000000U + word;
      context.shared_registers[start] = Bits(float(lane));
      context.shared_registers[start + 1] = Bits(float(lane + 1));
      context.shared_registers[start + 2] = static_cast<std::uint32_t>(iteration);
      const auto actual = Exercise(program, context, lane,
          unsigned(std::max(iteration, 0)), true, loop);
      if (loop && iteration == 0) zero_counts[lane] = actual;
      if (loop && iteration == 1) one_counts[lane] = actual;
      if (loop && iteration > 1)
        for (unsigned field = 0; field < actual.size(); ++field) {
          // The source has exactly one uniform loop body, with no branch in
          // that body. Enumerate its dynamic contribution using actual zero-
          // and one-trip execution, not a run-observed fixed golden number.
          Check(one_counts[lane][field] >= zero_counts[lane][field] &&
                actual[field] == zero_counts[lane][field] +
                    (one_counts[lane][field] - zero_counts[lane][field]) *
                        unsigned(iteration),
                "native loop counters lost or duplicated a body execution");
        }
    }
  std::cout << "Mesa sequence kind=" << kind << " static_smp=" << static_samples << " PASS\n";
}

void CheckedInLoopSequence() {
  const auto program = DecodePcoProgram(ShaderStage::kFragment, TextureUniformLoopPcoBinary());
  Check(CountPcoInstructions(program.instructions, true).texture == 1,
        "checked-in loop must contain one actual static SMP");
  for (int trips : {-3, 0, 1, 10, 16, 257})
    for (unsigned lane = 0; lane < 4; ++lane) {
      PcoFragmentExecutionContext context;
      context.shared_count = 24;
      context.coefficient_count = 4;
      for (unsigned word = 0; word < 20; ++word) context.shared_registers[word] = 0x10000000U + word;
      context.shared_registers[20] = Bits(float(lane));
      context.shared_registers[21] = Bits(float(lane + 1));
      context.shared_registers[22] = static_cast<std::uint32_t>(trips);
      Exercise(program, context, lane, unsigned(std::max(trips, 0)), true, true);
    }
}
} // namespace

int main(int argc, char **argv) {
  if (argc > 2) return 2;
  try {
    PublicSequence(10); PublicSequence(16);
    CheckedInLoopSequence();
    if (argc == 2)
      for (unsigned kind = 0; kind < 3; ++kind) MesaSequence(argv[1], kind);
    std::cout << "PCO texture sequences PASS checks=" << checks
              << " mesa_producer=" << (argc == 2 ? "verified" : "not-requested") << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
  return 0;
}
