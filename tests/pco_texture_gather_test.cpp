// SPDX-License-Identifier: MIT
#include "shader/pco_iss.h"
#include "pco_texture_gather_fixtures.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace {
using namespace pvrgpu::stub;
unsigned checks = 0;
void Check(bool value, const char *why) {
  ++checks;
  if (!value) throw std::runtime_error(why);
}
template<class F> void Reject(F function, const char *why) {
  bool rejected = false;
  try { function(); } catch (const std::exception &) { rejected = true; }
  Check(rejected, why);
}
std::uint32_t Bits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
std::size_t Sample(const PcoDecodedProgram &program) {
  auto it = std::find_if(program.instructions.begin(), program.instructions.end(),
      [](const auto &i) { return i.opcode == PcoOpcode::kTextureSample; });
  Check(it != program.instructions.end(), "native fixture has no SMP");
  return static_cast<std::size_t>(it - program.instructions.begin());
}
PcoFragmentExecutionContext Context() {
  PcoFragmentExecutionContext context;
  context.shared_count = 22;
  for (unsigned i = 0; i < 20; ++i) context.shared_registers[i] = 100 + i;
  context.shared_registers[20] = Bits(0.375F);
  context.shared_registers[21] = Bits(0.625F);
  return context;
}
void TestNative() {
  const std::array<std::uint32_t, 4> taps{Bits(0.125F), Bits(0.25F), Bits(0.5F), Bits(0.75F)};
  for (unsigned kind : {0U, 2U}) {
    const auto binary = test::TextureGatherFixture(kind);
    const auto program = DecodePcoProgram(ShaderStage::kFragment, binary);
    const auto &i = program.instructions[Sample(program)];
    const bool gather = kind == 0;
    Check(i.texture_gather == gather && i.texture_dimension == 2 &&
          i.texture_fcnorm == 1 && i.texture_lod_replace == 1 &&
          i.component_count == 4 && i.source2.index == (gather ? 16 : 8),
          "native gather/raw ABI and ordinary sampler ABI must stay distinct");
    if (gather) {
      Check(binary[i.binary_offset] == 0xf4 && binary[i.binary_offset + 1] == 0x52 &&
            binary[i.binary_offset + 2] == 0x90,
            "genuine compiler bytes must prove CHAN1/RAWDATA/REPLACE/PPLOD");
      Check(!HasCanonicalTextureLodMode(i) && HasCanonicalTextureLodMode(i, true),
            "task-stage validation must refuse otherwise canonical fragment gather");
    }
    const auto context = Context();
    const PcoPreparedFragmentProgram prepared(program.summary, program.instructions);
    const auto first = ExecuteFragmentPco(program.summary, program.instructions, context);
    const auto fast = ExecuteFragmentPco(prepared, context);
    const auto &r = first.texture_request;
    Check(first.suspended && first.texture_request_valid && r.gather == gather &&
          r.coordinate_count == 2 && r.component_count == 4 && r.dimension == 2 &&
          r.normalized == 1 && r.fcnorm == 1 && r.explicit_lod_present == 1 &&
          r.explicit_lod == 0 && !r.lod_bias_present && !r.sample_index_present &&
          r.texture_address_lo == 0 && r.texture_address_hi == 0,
          "one raw gather request must retain its actual native contract");
    Check(r.coordinates == std::array<std::uint32_t, 3>{Bits(0.375F), Bits(0.625F), 0} &&
          r.spatial_offsets == std::array<std::int32_t, 3>{0, 0, 0},
          "LOD and descriptor metadata must not enter the UV payload");
    for (unsigned w = 0; w < 4; ++w)
      Check(r.texture_state[w] == 100 + w &&
            r.sampler_state[w] == 100 + (gather ? 16 : 8) + w,
            "sampler values must be read from the native selected descriptor block");
    Check(first.executed_instructions.texture == 1 && fast.executed_instructions.texture == 1 &&
          fast.texture_request.gather == r.gather &&
          fast.continuation.program_signature == first.continuation.program_signature,
          "prepared and vector APIs issue exactly one equivalent SMP");
    const auto done = ResumeFragmentPco(program.summary, program.instructions, first.continuation, taps);
    const auto fast_done = ResumeFragmentPco(prepared, fast.continuation, taps);
    const std::array<unsigned, 4> order = gather ? std::array<unsigned, 4>{2,3,1,0} :
                                                 std::array<unsigned, 4>{0,1,2,3};
    for (unsigned c = 0; c < 4; ++c)
      Check(done.pixel_outputs[c] == taps[order[c]] &&
            fast_done.pixel_outputs[c] == done.pixel_outputs[c],
            "real PCO post-SMP moves must provide GL gather order without CPU reordering");
    Check(!done.suspended && !done.texture_request_valid && done.written_mask == 15 &&
          done.executed_instructions.texture == 1 &&
          done.native_steps == program.instructions.size() &&
          fast_done.native_steps == done.native_steps,
          "WDF consumes the response once without another request or texture issue");
    auto bad = first.continuation;
    ++bad.program_signature;
    Reject([&] { ResumeFragmentPco(prepared, bad, taps); }, "changed program signature resumed");
    bad = first.continuation;
    ++bad.pending_component_count;
    Reject([&] { ResumeFragmentPco(prepared, bad, taps); }, "wrong gather response width resumed");
    auto vertex = program.summary;
    vertex.stage = ShaderStage::kVertex;
    if (gather)
      Reject([&] { ExecuteVertexPco(vertex, program.instructions, {}); },
             "fragment gather reached vertex sampling");
  }
}
void TestRefusal() {
  const auto binary = test::TextureGatherFixture(0);
  const auto program = DecodePcoProgram(ShaderStage::kFragment, binary);
  const auto index = Sample(program);
  const auto offset = program.instructions[index].binary_offset;
  Reject([&] { DecodePcoProgram(ShaderStage::kFragment, test::TextureGatherFixture(1)); },
         "real component1 uses eight response DWORDs and must remain unsupported");
  for (unsigned mask : {0x10U, 0x08U}) {
    auto bad = binary; bad[offset] ^= mask;
    Reject([&] { DecodePcoProgram(ShaderStage::kFragment, bad); }, "FCNORM/DRC mutation accepted");
  }
  for (unsigned mask : {0x80U, 0x20U, 0x04U, 0x08U, 0x0cU, 0x01U, 0x02U}) {
    auto bad = binary; bad[offset + 1] ^= mask;
    Reject([&] { DecodePcoProgram(ShaderStage::kFragment, bad); }, "EXTB/dimension/CHAN/LOD mutation accepted");
  }
  for (unsigned mask : {0x80U, 0x40U, 0x10U, 0x20U, 0x30U, 2U, 4U, 8U}) {
    auto bad = binary; bad[offset + 2] ^= mask;
    Reject([&] { DecodePcoProgram(ShaderStage::kFragment, bad); }, "PPLOD/PROJ/SBMODE/SOO/SNO/NN mutation accepted");
  }
  auto wrong_sampler = binary;
  Check(wrong_sampler[offset + 6] == 16, "fixture sampler encoding moved");
  wrong_sampler[offset + 6] = 8;
  Reject([&] { DecodePcoProgram(ShaderStage::kFragment, wrong_sampler); },
         "RAWDATA accepted the ordinary sampler block");
  const auto context = Context();
  for (unsigned mutation = 0; mutation < 14; ++mutation) {
    auto bad = program;
    auto &i = bad.instructions[index];
    switch (mutation) {
      case 0: i.texture_gather = 2; break;
      case 1: i.texture_gather = 0; break;
      case 2: i.texture_dimension = 3; break;
      case 3: i.texture_fcnorm = 0; break;
      case 4: i.texture_address_offset = 2; break;
      case 5: i.texture_non_normalized_coords = 1; break;
      case 6: i.texture_sample_index_present = 1; break;
      case 7: i.texture_spatial_offset_present = 1; break;
      case 8: i.texture_lod_replace = 0; break;
      case 9: i.texture_lod_bias = 1; break;
      case 10: i.source2.index = 8; break;
      case 11: i.component_count = 1; break;
      case 12: i.data_request = 1; break;
      case 13: i.source.index = kPcoTemporaryCount - 2; break;
    }
    Reject([&] { ExecuteFragmentPco(bad.summary, bad.instructions, context); },
           "invalid gather metadata executed");
    Reject([&] { PcoPreparedFragmentProgram p(bad.summary, bad.instructions); },
           "invalid gather metadata was prepared");
  }
  for (std::size_t k = 0; k < program.instructions.size(); ++k) {
    if (k == index) continue;
    for (unsigned flag : {1U, 2U, 255U}) {
      auto bad = program;
      bad.instructions[k].texture_gather = flag;
      Reject([&] { ExecuteFragmentPco(bad.summary, bad.instructions, context); },
             "gather flag on unrelated opcode bypassed validation");
    }
  }
  // Reuse the existing decoded load form, but source a nonzero shader LOD.
  auto nonzero = program;
  nonzero.instructions[2].source = nonzero.instructions[1].source;
  bool rejected_lod = false;
  try { ExecuteFragmentPco(nonzero.summary, nonzero.instructions, context); }
  catch (const std::exception &error) {
    rejected_lod = std::string(error.what()).find("zero LOD payload") != std::string::npos;
  }
  Check(rejected_lod, "gather must reject the actual nonzero runtime PPLOD, not an unrelated field");
  auto truncated = context; truncated.shared_count = 19;
  Reject([&] { ExecuteFragmentPco(program.summary, program.instructions, truncated); },
         "gather accepted a truncated descriptor block");
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc > 2) return 2;
    if (argc == 2) for (unsigned kind = 0; kind < 3; ++kind) {
      std::ifstream stream(std::string(argv[1]) + "/gather-" + std::to_string(kind) + ".bin", std::ios::binary);
      Check(bool(stream), "missing regenerated native fixture");
      const std::vector<std::uint8_t> actual{std::istreambuf_iterator<char>(stream), {}};
      Check(actual == test::TextureGatherFixture(kind), "fixture differs from actual PCO output");
    }
    TestNative(); TestRefusal();
    std::cout << "Native component gather ISS: " << checks << " checks PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Native component gather ISS FAIL: " << error.what() << '\n';
    return 1;
  }
}
