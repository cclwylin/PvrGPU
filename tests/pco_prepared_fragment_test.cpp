// Prepared-program ownership/validation regression. All functional programs
// are decoded from public native fixture bytes; no caller supplies a signature.
#include "shader/pco_iss.h"
#include "pco_temp256_fixtures.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace pvrgpu::stub;
namespace {
unsigned checks = 0;
void Check(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
  ++checks;
}
template <typename Function> std::string Refused(Function function) {
  try { function(); } catch (const std::exception &error) { ++checks; return error.what(); }
  throw std::runtime_error("expected fail-closed refusal");
}
std::vector<uint8_t> Hex(const char *text) {
  std::istringstream stream(text);
  std::vector<uint8_t> bytes;
  unsigned byte = 0;
  while (stream >> std::hex >> byte) bytes.push_back(uint8_t(byte));
  return bytes;
}
uint32_t FloatBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

// Compare every semantic field, not C++ padding bytes.
std::vector<uint64_t> Words(const PcoFragmentExecution &value) {
  std::vector<uint64_t> words;
  const auto add = [&](auto x) { words.push_back(uint64_t(x)); };
  const auto array = [&](const auto &values) { for (auto x : values) add(x); };
  const auto counts = [&](const auto &c) { add(c.alu); add(c.texture); add(c.memory); };
  array(value.pixel_outputs);
  const auto &request = value.texture_request;
  array(request.coordinates); array(request.spatial_offsets);
  array(request.texture_state); array(request.sampler_state);
#define REQUEST(field) add(request.field)
  REQUEST(texture_address_lo); REQUEST(texture_address_hi); REQUEST(coordinate_count);
  REQUEST(component_count); REQUEST(descriptor_set); REQUEST(binding); REQUEST(dimension);
  REQUEST(normalized); REQUEST(fcnorm); REQUEST(sample_index); REQUEST(sample_index_present);
  REQUEST(data_request); REQUEST(explicit_lod); REQUEST(explicit_lod_present);
#undef REQUEST
  const auto &continuation = value.continuation;
  array(continuation.temporaries); array(continuation.temporary_written_mask.words);
  array(continuation.pixel_outputs);
  for (const auto &loop : continuation.loops) { add(loop.start_pc); add(loop.count); }
#define CONTINUATION(field) add(continuation.field)
  CONTINUATION(program_binary_size); CONTINUATION(program_instruction_count);
  CONTINUATION(program_signature); CONTINUATION(resume_instruction_index);
  CONTINUATION(pending_output_index); CONTINUATION(pending_component_count);
  CONTINUATION(data_request); CONTINUATION(valid); CONTINUATION(kind);
  CONTINUATION(written_mask); CONTINUATION(depth); CONTINUATION(depth_written);
  CONTINUATION(predicate); CONTINUATION(predicate_valid); CONTINUATION(discarded);
  CONTINUATION(loop_depth); CONTINUATION(execution_predicate); CONTINUATION(native_steps);
#undef CONTINUATION
  counts(continuation.executed_instructions);
#define EXECUTION(field) add(value.field)
  EXECUTION(executed_instruction_count); EXECUTION(native_steps); EXECUTION(written_mask);
  EXECUTION(texture_request_valid); EXECUTION(derivative_source); EXECUTION(derivative_request_valid);
  EXECUTION(suspended); EXECUTION(discarded); EXECUTION(depth); EXECUTION(depth_written);
#undef EXECUTION
  counts(value.executed_instructions);
  return words;
}
PcoFragmentExecutionContext TextureContext(unsigned lane = 0) {
  PcoFragmentExecutionContext context;
  context.coefficient_count = 12;
  context.coefficients[2] = FloatBits(1.0F);
  context.coefficients[6] = FloatBits(float(lane + 1) / 128.0F);
  context.coefficients[10] = FloatBits(float(lane + 5) / 128.0F);
  context.shared_count = 20;
  for (unsigned i = 0; i < context.shared_count; ++i)
    context.shared_registers[i] = 0x10000000 + i;
  context.sample_x = FloatBits(float(lane));
  context.sample_y = FloatBits(float(lane % 7));
  return context;
}
PcoDecodedProgram TextureProgram(unsigned samples) {
  auto binary = FillTexNearestFragmentPcoBinary();
  // Duplicate the genuine public SMP/WDF pair before exports. The compiler's
  // initialized coordinate and descriptor registers remain valid for each SMP.
  const std::vector<uint8_t> sample(binary.begin() + 128, binary.begin() + 146);
  for (unsigned i = 1; i < samples; ++i)
    binary.insert(binary.begin() + 146, sample.begin(), sample.end());
  return Decode(ShaderStage::kFragment, binary);
}
std::array<uint32_t, 4> Response(unsigned lane, unsigned round) {
  // Register transport must preserve NaNs, infinities, signed zero and arbitrary
  // data exactly; this fixture does not stand in for the actual sampler unit.
  return {{UINT32_C(0x7fc00000) + lane, UINT32_C(0xff800000),
           UINT32_C(0x80000000), UINT32_C(0x11223344) + round}};
}

void Compare(const PcoDecodedProgram &program, PcoFragmentExecutionContext context,
             unsigned lane, unsigned expected_samples, unsigned expected_derivatives) {
  const PcoPreparedFragmentProgram prepared(program.summary, program.instructions);
  auto old = ExecuteFragmentPco(program.summary, program.instructions, context);
  auto owned = ExecuteFragmentPco(prepared, context);
  unsigned samples = 0, derivatives = 0;
  for (unsigned round = 0;; ++round) {
    Check(Words(old) == Words(owned), "raw/prepared semantic state differs");
    if (!old.suspended) break;
    Check(round < 64, "fixture did not complete");
    context.continuation = owned.continuation;
    context.texture_response_valid = owned.texture_request_valid;
    context.derivative_response_valid = owned.derivative_request_valid;
    if (owned.texture_request_valid) {
      context.texture_response = Response(lane, round);
      ++samples;
    } else {
      const auto &instruction = program.instructions[owned.continuation.resume_instruction_index - 1];
      const auto quad = EvaluatePcoDerivativeQuad(instruction,
          {{FloatBits(1.0F), FloatBits(3.0F), FloatBits(6.0F), FloatBits(12.0F)}});
      context.derivative_response = quad[lane % 4];
      ++derivatives;
    }
    auto raw_context = context;
    raw_context.continuation = old.continuation;
    old = ExecuteFragmentPco(program.summary, program.instructions, raw_context);
    owned = ExecuteFragmentPco(prepared, context);
  }
  Check(samples == expected_samples && derivatives == expected_derivatives,
        "differential test did not visit the intended native checkpoints");
  Check(owned.executed_instructions.texture == samples, "native sample counter differs from requests");
}

void TestDifferential() {
  for (const auto &binary : {FillSolidFragmentPcoBinary(), FillSolidBlackFragmentPcoBinary(),
                             FillSolidGreenHalfAlphaFragmentPcoBinary(), FillSolidRedHalfAlphaFragmentPcoBinary()})
    Compare(Decode(ShaderStage::kFragment, binary), {}, 0, 0, 0);
  for (unsigned samples : {1U, 2U, 8U})
    for (unsigned lane = 0; lane < 32; ++lane)
      Compare(TextureProgram(samples), TextureContext(lane), lane, samples, 0);

  const auto derivative = Decode(ShaderStage::kFragment, Hex(
      "35 82 00 87 80 08 00 00 00 40 "
      "34 8a 00 87 40 00 00 20 "
      "34 82 00 88 40 00 00 41 "
      "34 8a 00 87 41 00 00 21 "
      "34 82 00 89 40 00 00 42 "
      "34 8a 00 87 42 00 00 22 "
      "35 8a 80 87 80 01 00 00 00 23"));
  for (unsigned lane = 0; lane < 4; ++lane) {
    PcoFragmentExecutionContext context;
    context.shared_count = 1;
    context.shared_registers[0] = FloatBits(float(1 + lane * 3));
    Compare(derivative, context, lane, 0, 2);
  }
  // Exact native sample-mask/ALPHAF fixture. A derivative after feedback must
  // still resume for discarded helpers, with identical visibility and counters.
  auto feedback = Hex(
      "56 d2 40 00 02 80 81 80 d5 01 40 ff "
      "86 92 40 13 aa aa 00 00 00 00 41 ff "
      "56 b2 40 41 02 80 40 00 41 40 40 ff "
      "99 d2 00 d3 3c f8 c0 9c 1e 87 87 c0 cf 80 11 00 20 40 "
      "98 c0 00 d3 3c f1 a0 9c 1e 87 c0 cf 80 11 00 20 "
      "45 a9 00 82 80 40 00 07 00 00 "
      "02 80 6a ff "
      "34 8a 00 87 00 00 00 20 "
      "34 8a 00 87 00 00 00 21 "
      "36 8a 00 87 00 00 00 22 f2 ff ff ff "
      "38 8a 80 87 80 01 00 00 00 23 f3 ff ff ff ff ff");
  const auto checkpoint = Hex("34 82 00 88 40 00 00 41");
  feedback.insert(feedback.begin() + 84, checkpoint.begin(), checkpoint.end());
  const auto program = Decode(ShaderStage::kFragment, feedback);
  for (unsigned count : {1U, 2U, 4U, 8U, 16U})
    for (unsigned sample = 0; sample < count; ++sample) {
      PcoFragmentExecutionContext context;
      context.raster_sample_count = count;
      context.sample_id = sample;
      Compare(program, context, sample, 0, 1);
    }
}

void TestOwnershipAndValidation() {
  const auto original = TextureProgram(1);
  auto source = original;
  const PcoPreparedFragmentProgram owned(source.summary, source.instructions);
  const auto context = TextureContext();
  const auto first = ExecuteFragmentPco(owned, context);
  const auto baseline = ExecuteFragmentPco(original.summary, original.instructions, context);
  Check(Words(first) == Words(baseline), "owned initial snapshot differs");
  source.summary = {};
  source.instructions.clear(); source.instructions.shrink_to_fit();
  Check(Words(ExecuteFragmentPco(owned, context)) == Words(baseline),
        "caller mutation changed owned program");
  Refused([&] { PcoPreparedFragmentProgram invalid(source.summary, source.instructions); });

  auto replacement = original;
  replacement.instructions[16].source.index = 17; // Another valid initialized coordinate pair.
  const PcoPreparedFragmentProgram changed(replacement.summary, replacement.instructions);
  const auto changed_first = ExecuteFragmentPco(changed, context);
  Check(changed_first.texture_request.coordinates != first.texture_request.coordinates,
        "same-sized replacement did not change real native operands");
  Check(changed_first.continuation.program_signature != first.continuation.program_signature,
        "semantic signature did not bind native operands");
  Refused([&] { ResumeFragmentPco(changed, first.continuation, Response(0, 0)); });
  Refused([&] { ResumeFragmentPco(replacement.summary, replacement.instructions,
                                 first.continuation, Response(0, 0)); });
  Refused([&] { ResumeFragmentPco(owned, changed_first.continuation, Response(0, 0)); });
  replacement = original;
  replacement.summary.binary_size += 8; // Valid envelope, same group/instruction counts.
  const PcoPreparedFragmentProgram other_size(replacement.summary, replacement.instructions);
  Refused([&] { ResumeFragmentPco(other_size, first.continuation, Response(0, 0)); });

  for (unsigned mutation = 0; mutation < 13; ++mutation) {
    auto invalid = original;
    switch (mutation) {
      case 0: invalid.summary.stage = ShaderStage::kVertex; break;
      case 1: invalid.summary.instruction_count--; break;
      case 2: invalid.summary.group_count--; break;
      case 3: invalid.summary.vertex_input_mask = 1; break;
      case 4: invalid.summary.vertex_output_mask = 1; break;
      case 5: invalid.summary.ends_task = 1; break;
      case 6: invalid.summary.early_hsr_safe = 0; break;
      case 7: invalid.instructions[0].group_index++; break;
      case 8: invalid.instructions[0].repeat_count = 0; break;
      case 9: invalid.instructions[1].binary_offset = 0; break;
      case 10: invalid.instructions[16].integer_signed = 1; break;
      case 11: invalid.instructions[0].end_group = 1; break;
      case 12: invalid.instructions[16].source.index = 255; break;
    }
    const auto raw_error = Refused([&] { ExecuteFragmentPco(invalid.summary, invalid.instructions, context); });
    const auto owned_error = Refused([&] { PcoPreparedFragmentProgram bad(invalid.summary, invalid.instructions); });
    Check(raw_error == owned_error, "prepared construction changed strict validation");
  }
  for (unsigned mutation = 0; mutation < 22; ++mutation) {
    auto invalid = context;
    invalid.continuation = first.continuation;
    invalid.texture_response_valid = 1;
    invalid.texture_response = Response(1, 0);
    switch (mutation) {
      case 0: invalid.continuation.program_signature ^= 1; break;
      case 1: invalid.continuation.program_binary_size++; break;
      case 2: invalid.continuation.program_instruction_count++; break;
      case 3: invalid.continuation.resume_instruction_index = 0; break;
      case 4: invalid.continuation.pending_component_count = 3; break;
      case 5: invalid.continuation.pending_output_index++; break;
      case 6: invalid.continuation.temporary_written_mask.words[0] ^= 1; break;
      case 7: invalid.continuation.temporary_written_mask.words[3] ^= UINT64_C(1) << 63; break;
      case 8: invalid.continuation.written_mask = 15; break;
      case 9: invalid.continuation.predicate_valid = 1; break;
      case 10: invalid.continuation.discarded = 1; break;
      case 11: invalid.continuation.loop_depth = 33; break;
      case 12: invalid.continuation.execution_predicate = 2; break;
      case 13: invalid.continuation.native_steps = UINT64_C(10000001); break;
      case 14: invalid.continuation.executed_instructions.alu = UINT64_MAX; break;
      case 15: invalid.continuation.valid = 2; break;
      case 16: invalid.continuation.kind = 1; break;
      case 17: invalid.texture_response_valid = 0; break;
      case 18: invalid.derivative_response_valid = 1; break;
      case 19: invalid.memory_side_effects_enabled = 2; break;
      case 20: invalid.shared_count = kPcoMaximumFragmentSharedCount + 1; break;
      case 21: invalid.raster_sample_count = 3; break;
    }
    const auto raw_error = Refused([&] { ExecuteFragmentPco(original.summary, original.instructions, invalid); });
    const auto owned_error = Refused([&] { ExecuteFragmentPco(owned, invalid); });
    Check(raw_error == owned_error, "prepared execution changed continuation/context refusal");
  }
  const auto raw_next = ResumeFragmentPco(original.summary, original.instructions, first.continuation, Response(0, 0));
  const auto owned_next = ResumeFragmentPco(owned, first.continuation, Response(0, 0));
  Check(Words(raw_next) == Words(owned_next), "prepared Resume overload changed texture state");
}

void Benchmark() {
  const auto program = TextureProgram(8);
  const PcoPreparedFragmentProgram prepared(program.summary, program.instructions);
  const auto run = [&](bool owned) {
    const auto start = std::chrono::steady_clock::now();
    uint64_t checksum = 0;
    for (unsigned lane = 0; lane < 4096; ++lane) {
      auto context = TextureContext(lane % 32);
      auto result = owned ? ExecuteFragmentPco(prepared, context)
                          : ExecuteFragmentPco(program.summary, program.instructions, context);
      while (result.suspended) {
        context.continuation = result.continuation;
        context.texture_response_valid = 1;
        context.texture_response = Response(lane, 0);
        result = owned ? ExecuteFragmentPco(prepared, context)
                       : ExecuteFragmentPco(program.summary, program.instructions, context);
      }
      checksum += result.pixel_outputs[0] + result.native_steps + result.executed_instructions.texture;
    }
    const double milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return std::make_pair(milliseconds, checksum);
  };
  const auto raw = run(false), owned = run(true);
  Check(raw.second == owned.second, "benchmark changed native results");
  std::cout << "informational host benchmark raw_ms=" << raw.first << " prepared_ms=" << owned.first
            << " ratio=" << raw.first / owned.first << " (not a timing assertion)\n";
}
} // namespace
int main(int argc, char **argv) {
  try {
    TestDifferential();
    TestOwnershipAndValidation();
    if (argc == 2 && std::string(argv[1]) == "--benchmark") Benchmark();
    std::cout << "Prepared fragment program PASS checks=" << checks << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
