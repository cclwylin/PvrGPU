// SPDX-License-Identifier: MIT
#include "shader/pco_iss.h"
#include "pco_multisample_texture_fixtures.h"
#include "pco_explicit_texture_fixtures.h"
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
using pvrgpu::stub::test::MultisampleTextureFixture;
unsigned checks = 0;
void Check(bool condition, const char *reason) {
  ++checks;
  if (!condition) throw std::runtime_error(reason);
}
template <typename Function> void Reject(Function function, const char *reason) {
  bool failed = false;
  try { function(); } catch (const std::exception &) { failed = true; }
  Check(failed, reason);
}
std::uint32_t FloatBits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
constexpr std::uint64_t kBaseAddress = UINT64_C(0x1234000040);
std::array<std::uint32_t, kPcoMaximumSharedCount> Shared(unsigned samples,
                                                       unsigned sample,
                                                       bool array = true) {
  std::array<std::uint32_t, kPcoMaximumSharedCount> words{};
  unsigned log_samples = 0;
  for (unsigned n = samples; n > 1; n >>= 1) ++log_samples;
  // Public IMAGE_WORD0 width/height/smpcnt and IMAGE_WORD1 address/depth.
  const std::uint64_t word0 = (std::uint64_t(log_samples) << 62) |
      (UINT64_C(18) << 48) | (UINT64_C(12) << 34) | (array ? 1U : 4U);
  const std::uint64_t word1 = ((kBaseAddress >> 2) << 16) |
      (array ? (UINT64_C(4) << 4) | 1U : (UINT64_C(1) << 60) | 12U);
  words[0] = static_cast<std::uint32_t>(word0);
  words[1] = static_cast<std::uint32_t>(word0 >> 32);
  words[2] = static_cast<std::uint32_t>(word1);
  words[3] = static_cast<std::uint32_t>(word1 >> 32);
  // PCO_IMAGE_META_LAYER_SIZE is one layer, not the whole array allocation.
  words[4] = 13 * 19 * 16 * samples;
  words[20] = 4;
  words[21] = 7;
  words[22] = 2;
  words[23] = sample;
  return words;
}
std::size_t SampleInstruction(const PcoDecodedProgram &program) {
  const auto found = std::find_if(program.instructions.begin(), program.instructions.end(),
      [](const auto &instruction) { return instruction.opcode == PcoOpcode::kTextureSample; });
  Check(found != program.instructions.end(), "native fixture must contain SMP");
  return static_cast<std::size_t>(found - program.instructions.begin());
}
void CheckRequest(const PcoTextureRequest &request, unsigned samples,
                  unsigned sample, bool array, bool floating) {
  Check(request.normalized == 0 && request.sample_index_present == 1 &&
            request.sample_index == sample && request.fcnorm == unsigned(floating),
        "NNCOORDS/SNO/FCNORM must remain independent native fields");
  Check(request.coordinates[0] == FloatBits(4) && request.coordinates[1] == FloatBits(7) &&
            request.coordinates[2] == 0 && request.coordinate_count == 2 &&
            request.dimension == 2 && request.component_count == 4,
        "lookup/address words must not overflow or enter coordinates[3]");
  const std::uint64_t address = request.texture_address_lo |
      (std::uint64_t(request.texture_address_hi) << 32);
  Check(address == (array ? kBaseAddress + 2 * 13 * 19 * 16 * samples : 0),
        "array TAO must preserve native base plus true per-layer stride");
}
void TestNativeExplicitSampling() {
  const std::array<std::uint32_t, 4> response{FloatBits(0.25F), FloatBits(0.5F), FloatBits(0.75F), FloatBits(1)};
  for (unsigned kind = 21; kind <= 24; ++kind) {
    const auto binary = test::ExplicitTextureFixture(kind);
    const auto program = DecodePcoProgram(ShaderStage::kFragment, binary);
    const auto smp = SampleInstruction(program);
    const auto &instruction = program.instructions[smp];
    Check(instruction.texture_lod_replace == 1 && !instruction.texture_sample_index_present &&
          instruction.texture_non_normalized_coords == (kind != 24) &&
          instruction.texture_address_offset == (kind == 22) &&
          instruction.texture_dimension == (kind >= 23 ? 3 : 2),
          "native REPLACE/PPLod/NNCOORDS/TAO decoded independently");
    for (int level : {-1, 0, 1, 2, 15}) {
      PcoFragmentExecutionContext context;
      context.shared_count = 24;
      context.shared_registers = Shared(1, 0, kind == 22);
      context.shared_registers[23] = kind == 24 ? FloatBits(static_cast<float>(level)) : static_cast<std::uint32_t>(level);
      if (kind == 24) {
        context.shared_registers[20] = FloatBits(1);
        context.shared_registers[21] = FloatBits(-0.5F);
        context.shared_registers[22] = FloatBits(0.25F);
      }
      const auto first = ExecuteFragmentPco(program.summary, program.instructions, context);
      const auto &request = first.texture_request;
      Check(first.suspended && first.texture_request_valid && request.explicit_lod_present == 1 &&
            request.explicit_lod == FloatBits(static_cast<float>(level)) &&
            request.normalized == (kind == 24) && !request.sample_index_present,
            "native float LOD survives without substitution or implicit derivatives");
      Check(request.coordinates[0] == FloatBits(kind == 24 ? 1.0F : 4.0F) &&
            request.coordinates[1] == FloatBits(kind == 24 ? -0.5F : 7.0F) &&
            request.coordinates[2] == (kind >= 23 ? FloatBits(kind == 24 ? 0.25F : 2.0F) : 0),
            "native integer-to-float texel coordinates remain distinct from cube direction");
      const std::uint64_t address = request.texture_address_lo | (std::uint64_t(request.texture_address_hi) << 32);
      Check(address == (kind == 22 ? kBaseAddress + 2 * 13 * 19 * 16 : 0),
            "explicit LOD must not displace the array TAO pair");
      const auto done = ResumeFragmentPco(program.summary, program.instructions, first.continuation, response);
      Check(!done.suspended && std::equal(response.begin(), response.end(), done.pixel_outputs.begin()),
            "native explicit SMP/WDF returns the raw TPU response");
    }
    for (unsigned reserved : {4U, 16U, 32U, 64U}) {
      auto bad = binary;
      bad[instruction.binary_offset + 2] |= static_cast<std::uint8_t>(reserved);
      Reject([&] { (void)DecodePcoProgram(ShaderStage::kFragment, bad); },
             "unsupported REPLACE extension accepted");
    }
    for (unsigned mutation = 0; mutation < 3; ++mutation) {
      auto bad = program;
      if (mutation == 0) bad.instructions[smp].texture_lod_replace = 2;
      if (mutation == 1) bad.instructions[smp].texture_sample_index_present = 1;
      if (mutation == 2) bad.instructions[smp].source.index = kPcoTemporaryCount - 2;
      PcoFragmentExecutionContext context;
      context.shared_count = 24; context.shared_registers = Shared(1, 0, kind == 22);
      Reject([&] { (void)ExecuteFragmentPco(bad.summary, bad.instructions, context); },
             "malformed REPLACE semantic flags/span accepted");
    }
    for (std::size_t index = 0; index < program.instructions.size(); ++index) {
      if (index == smp) continue;
      for (unsigned flag : {1U,2U,255U}) {
        auto bad = program; bad.instructions[index].texture_lod_replace = flag;
        PcoFragmentExecutionContext context;
        context.shared_count = 24; context.shared_registers = Shared(1,0,kind==22);
        bool rejected = false;
        try { (void)ExecuteFragmentPco(bad.summary,bad.instructions,context); }
        catch (const std::exception &error) {
          rejected = std::string(error.what()).find("texture LOD replacement") != std::string::npos;
        }
        Check(rejected, "unrelated opcode must reject replacement flag before special branch");
      }
    }
  }
}

void TestNativeSpatialOffsetLookup() {
  for (unsigned kind : {21U, 23U}) {
    const auto original = test::ExplicitTextureFixture(kind);
    const auto original_program = DecodePcoProgram(ShaderStage::kFragment, original);
    const auto smp_index = SampleInstruction(original_program);
    const auto &smp = original_program.instructions[smp_index];
    const unsigned dimensions = smp.texture_dimension;
    const unsigned lookup_register = smp.source.index + dimensions + 1;
    for (int u : {-32, -1, 0, 31})
      for (int v : {-32, -1, 0, 31})
        for (int w : {-8, -1, 0, 7}) {
          const std::uint32_t lookup = (static_cast<std::uint32_t>(u) & 63U) |
              ((static_cast<std::uint32_t>(v) & 63U) << 6U) |
              (dimensions == 3 ? ((static_cast<std::uint32_t>(w) & 15U) << 12U) : 0U);
          auto binary = original;
          binary[smp.binary_offset + 2] |= 2U; // I_SMP.SOO, independent of LOD.
          std::vector<std::uint8_t> move{0x86, 0x92, 0x40, 0x13,
              0, 0, 0, 0, 0, 0, static_cast<std::uint8_t>(0x40U | lookup_register), 0xff};
          for (unsigned byte = 0; byte < 4; ++byte)
            move[4 + byte] = static_cast<std::uint8_t>(lookup >> (8 * byte));
          binary.insert(binary.begin() + smp.binary_offset - 3, move.begin(), move.end());
          const auto program = DecodePcoProgram(ShaderStage::kFragment, binary);
          PcoFragmentExecutionContext context;
          context.shared_count = 24;
          context.shared_registers = Shared(1, 0, false);
          const auto issued = ExecuteFragmentPco(program.summary, program.instructions, context);
          Check(issued.suspended && issued.texture_request_valid,
                "native SOO emits one ordinary texture continuation");
          Check(issued.texture_request.spatial_offsets == std::array<std::int32_t, 3>{u, v, dimensions == 3 ? w : 0},
                "native SOO sign extends U6/V6/W4 independently");
          Check(issued.texture_request.explicit_lod_present == 1 &&
                issued.texture_request.explicit_lod == FloatBits(0),
                "SOO lookup preserves the separate explicit LOD word");
          auto bad = program;
          bad.instructions[SampleInstruction(program)].texture_spatial_offset_present = 2;
          Reject([&] { ExecuteFragmentPco(bad.summary, bad.instructions, context); },
                 "SOO must be a canonical Boolean metadata flag");
        }
  }
}

void TestTextureDerivativeContinuation() {
  auto binary = test::ExplicitTextureFixture(21);
  const auto original = DecodePcoProgram(ShaderStage::kFragment, binary);
  const auto sample = SampleInstruction(original);
  // First export follows SMP/WDF. Keep the real texture result in r0..3,
  // then exchange r0 across the quad before the existing PIXOUT instructions.
  const auto output_offset = original.instructions[sample + 2].binary_offset - 3;
  const std::vector<std::uint8_t> derivative{0x34,0x82,0,0x8a,0x40,0,0,0x40};
  binary.insert(binary.begin() + output_offset, derivative.begin(), derivative.end());
  const auto program = DecodePcoProgram(ShaderStage::kFragment, binary);
  std::array<PcoFragmentExecutionContext,4> contexts;
  std::array<PcoFragmentExecution,4> lanes;
  std::array<std::uint32_t,4> sources;
  std::array<unsigned,4> steps{};
  for (unsigned lane=0; lane<4; ++lane) {
    auto &context=contexts[lane];
    context.shared_count=24; context.shared_registers=Shared(1,0,false);
    lanes[lane]=ExecuteFragmentPco(program.summary,program.instructions,context);
    steps[lane]+=lanes[lane].executed_instruction_count;
    Check(lanes[lane].texture_request_valid && lanes[lane].continuation.kind==0,
          "texture checkpoint remains distinct from a later derivative");
    context.continuation=lanes[lane].continuation;
    context.texture_response_valid=1;
    context.texture_response={FloatBits(float(lane*lane)),FloatBits(0.5F),FloatBits(0.25F),FloatBits(1)};
    lanes[lane]=ExecuteFragmentPco(program.summary,program.instructions,context);
    steps[lane]+=lanes[lane].executed_instruction_count;
    Check(lanes[lane].derivative_request_valid && !lanes[lane].texture_request_valid &&
          lanes[lane].continuation.kind==1, "SMP resume reaches the real derivative without replay");
    sources[lane]=lanes[lane].derivative_source;
  }
  const auto values=EvaluatePcoDerivativeQuad(program.instructions[sample+2],sources);
  for (unsigned lane=0; lane<4; ++lane) {
    auto &context=contexts[lane];
    context.continuation=lanes[lane].continuation;
    context.texture_response_valid=0;
    context.derivative_response_valid=1;
    context.derivative_response=values[lane];
    lanes[lane]=ExecuteFragmentPco(program.summary,program.instructions,context);
    steps[lane]+=lanes[lane].executed_instruction_count;
    Check(!lanes[lane].suspended && lanes[lane].pixel_outputs[0]==FloatBits(lane<2?1:5) &&
          lanes[lane].pixel_outputs[1]==FloatBits(0.5F) && steps[lane]==program.instructions.size(),
          "texture and derivative checkpoints preserve returned channels and exact instruction counts");
  }
}

void TestNativeSampling() {
  const std::array<std::uint32_t, 4> response = {
      UINT32_C(0xffffffff), UINT32_C(0x80000000), UINT32_C(0x01000001), UINT32_C(0x3f800000)};
  for (bool vertex : {false, true}) {
    for (unsigned kind = 0; kind < 6; ++kind) {
      const auto program = DecodePcoProgram(vertex ? ShaderStage::kVertex : ShaderStage::kFragment,
                                            MultisampleTextureFixture(kind + (vertex ? 11 : 0)));
      const auto smp = SampleInstruction(program);
      Check(program.instructions[smp].texture_non_normalized_coords == 1 &&
                program.instructions[smp].texture_sample_index_present == 1 &&
                program.instructions[smp].texture_address_offset == unsigned(kind >= 3),
            "decoder lost native multisample flags");
      for (unsigned samples : {1U, 2U, 4U, 8U}) {
        for (unsigned sample = 0; sample < samples; ++sample) {
          if (vertex) {
            PcoVertexExecutionContext context;
            context.shared_count = 24;
            context.shared_registers = Shared(samples, sample, kind >= 3);
            const auto first = ExecuteVertexPco(program.summary, program.instructions, {}, context);
            Check(first.suspended == 1 && first.texture_request_valid == 1, "VS native SMP must suspend");
            CheckRequest(first.texture_request, samples, sample, kind >= 3, kind % 3 == 0);
            const auto final = ResumeVertexPco(program.summary, program.instructions, first.continuation, response);
            Check(final.suspended == 0 && std::equal(response.begin(), response.end(), final.outputs.begin()),
                  "VS WDF must commit exactly the selected sample response DWORDs");
          } else {
            PcoFragmentExecutionContext context;
            context.shared_count = 24;
            context.shared_registers = Shared(samples, sample, kind >= 3);
            const auto first = ExecuteFragmentPco(program.summary, program.instructions, context);
            Check(first.suspended == 1 && first.texture_request_valid == 1, "FS native SMP must suspend");
            CheckRequest(first.texture_request, samples, sample, kind >= 3, kind % 3 == 0);
            const auto final = ResumeFragmentPco(program.summary, program.instructions, first.continuation, response);
            Check(final.suspended == 0 && std::equal(response.begin(), response.end(), final.pixel_outputs.begin()),
                  "FS WDF must preserve integer precision and response channel bits");
          }
        }
      }
    }
  }
}
void TestNativeQueries() {
  for (unsigned samples : {1U, 2U, 4U, 8U}) {
    for (unsigned kind = 6; kind <= 10; ++kind) {
      const auto program = DecodePcoProgram(ShaderStage::kFragment, MultisampleTextureFixture(kind));
      Check(std::none_of(program.instructions.begin(), program.instructions.end(),
          [](const auto &instruction) { return instruction.opcode == PcoOpcode::kTextureSample; }),
          "size/sample queries are real descriptor ALU, not TPU operations");
      PcoFragmentExecutionContext context;
      context.shared_count = 20;
      context.shared_registers = Shared(samples, 0, kind == 7 || kind == 9);
      const auto output = ExecuteFragmentPco(program.summary, program.instructions, context);
      if (kind == 8 || kind == 9)
        Check(output.pixel_outputs[0] == samples, "textureSamples must read descriptor SMPCNT");
      else
        Check(output.pixel_outputs[0] == 13 && output.pixel_outputs[1] == 19 &&
                  (kind != 7 || output.pixel_outputs[2] == 5), "textureSize must read native width/height/layers");
      Check(!output.suspended && !output.texture_request_valid, "query must not invent memory/SMP request");
    }
  }
}
void TestNativeStrideQueryLayout() {
  const auto ordinary = DecodePcoProgram(ShaderStage::kFragment,
                                         MultisampleTextureFixture(20));
  Check(std::none_of(ordinary.instructions.begin(), ordinary.instructions.end(),
      [](const auto &i) { return i.opcode == PcoOpcode::kTextureSample; }),
      "ordinary textureSize must be native descriptor ALU");
  for (unsigned mip_count : {1U, 3U, 5U}) {
    for (unsigned lod = 0; lod < mip_count; ++lod) {
      PcoFragmentExecutionContext context;
      context.shared_count = 21;
      context.shared_registers = Shared(1, 0, false);
      // STRIDE_WORD1[60:63] is the mip count, never a base LOD.
      context.shared_registers[3] =
          (context.shared_registers[3] & UINT32_C(0x0fffffff)) | (mip_count << 28);
      if (mip_count > 1) context.shared_registers[2] |= 1U << 15;
      context.shared_registers[20] = lod;
      const auto output = ExecuteFragmentPco(ordinary.summary, ordinary.instructions, context);
      Check(output.pixel_outputs[0] == std::max(13U >> lod, 1U) &&
                output.pixel_outputs[1] == std::max(19U >> lod, 1U) &&
                output.pixel_outputs[2] == 1 && output.pixel_outputs[3] == 81 &&
                output.written_mask == 0xf,
            "ordinary STRIDE query must use explicit LOD but not mip count");
      Check(!output.suspended && !output.texture_request_valid,
            "ordinary query must not manufacture a sample request");
    }
  }
  for (unsigned base_level : {0U, 1U, 2U}) {
    for (unsigned lod : {0U, 1U, 2U}) {
      PcoFragmentExecutionContext context;
      context.shared_count = 21;
      context.shared_registers = Shared(1, 0, true);
      context.shared_registers[2] =
          (context.shared_registers[2] & UINT32_C(0xffff0000)) | (1U << 15) | 5U;
      context.shared_registers[3] =
          (context.shared_registers[3] & UINT32_C(0x0fffffff)) | (base_level << 28);
      context.shared_registers[20] = lod;
      const auto output = ExecuteFragmentPco(ordinary.summary, ordinary.instructions, context);
      Check(output.pixel_outputs[0] == std::max(13U >> (lod + base_level), 1U) &&
                output.pixel_outputs[1] == std::max(19U >> (lod + base_level), 1U),
            "normal IMAGE query must retain base level plus explicit LOD");
    }
  }
}
void TestMalformedFields() {
  for (unsigned kind : {0U, 3U, 11U, 14U}) {
    const ShaderStage stage = kind >= 11 ? ShaderStage::kVertex : ShaderStage::kFragment;
    const auto binary = MultisampleTextureFixture(kind);
    const auto program = DecodePcoProgram(stage, binary);
    const auto smp = SampleInstruction(program);
    const std::size_t backend = program.instructions[smp].binary_offset;
    for (unsigned bit : {4U, 5U, 6U}) {
      auto bad = binary;
      bad[backend + 2] |= static_cast<std::uint8_t>(1U << bit);
      Reject([&] { (void)DecodePcoProgram(stage, bad); }, "unsupported SMP extension bit was ignored");
    }
    for (unsigned bit : {2U, 3U}) {
      auto bad = binary;
      bad[backend + 2] ^= static_cast<std::uint8_t>(1U << bit);
      Reject([&] { (void)DecodePcoProgram(stage, bad); }, "SNO/NNCOORDS mismatch must fail closed");
    }
    for (unsigned mutation = 0; mutation < 5; ++mutation) {
      auto bad = binary;
      if (mutation == 0) bad[backend] |= 8;          // DRC1
      if (mutation == 1) bad[backend + 1] |= 128;   // extb unsupported
      if (mutation == 2) bad[backend + 1] ^= 2;     // wrong LOD mode
      if (mutation == 3) bad[backend + 1] ^= 32;    // MS must be 2D
      if (mutation == 4) bad[backend + 2] ^= 1;    // TAO/pplod mismatch
      Reject([&] { (void)DecodePcoProgram(stage, bad); }, "unsupported native SMP form accepted");
    }
    for (unsigned mutation = 0; mutation < 3; ++mutation) {
      auto bad = program;
      if (mutation == 0) bad.instructions[smp].texture_sample_index_present = 2;
      if (mutation == 1) bad.instructions[smp].texture_non_normalized_coords = 0;
      if (mutation == 2) bad.instructions[smp].source.index = kPcoTemporaryCount - 2;
      Reject([&] {
        if (stage == ShaderStage::kVertex) {
          PcoVertexExecutionContext context;
          context.shared_count = 24; context.shared_registers = Shared(8, 7);
          (void)ExecuteVertexPco(bad.summary, bad.instructions, {}, context);
        } else {
          PcoFragmentExecutionContext context;
          context.shared_count = 24; context.shared_registers = Shared(8, 7);
          (void)ExecuteFragmentPco(bad.summary, bad.instructions, context);
        }
      }, "serialized SMP metadata/payload span must fail closed");
    }
  }
  // Pure ISS negative input: replace the existing lookup producer with a
  // decoded MBYP from SHARED23. This is explicitly a malformed semantic test,
  // not a claim that the native compiler emits reserved lookup bits.
  auto program = DecodePcoProgram(ShaderStage::kFragment, MultisampleTextureFixture(0));
  const auto smp = SampleInstruction(program);
  const auto lookup = program.instructions[smp].source.index + 2;
  std::size_t producer = smp;
  while (producer && program.instructions[--producer].output_index != lookup) {}
  Check(producer != smp, "lookup producer missing");
  PcoInstruction move = program.instructions[0];
  move.source.index = 23;
  move.output_index = lookup;
  move.group_index = program.instructions[producer].group_index;
  move.binary_offset = program.instructions[producer].binary_offset;
  program.instructions[producer] = move;
  PcoFragmentExecutionContext context;
  context.shared_count = 24;
  context.shared_registers = Shared(8, 0);
  for (std::uint32_t reserved : {UINT32_C(1), UINT32_C(0x80000), UINT32_C(0x80000000)}) {
    context.shared_registers[23] = reserved;
    Reject([&] { (void)ExecuteFragmentPco(program.summary, program.instructions, context); },
           "reserved lookup payload bit silently accepted");
  }
}

void TestNativeFragmentCoordinates() {
  const auto program = DecodePcoProgram(ShaderStage::kFragment,
                                        MultisampleTextureFixture(17));
  for (unsigned x : {0U, 4U, 12U}) {
    for (unsigned y : {0U, 7U, 18U}) {
      PcoFragmentExecutionContext context;
      context.sample_x = FloatBits(static_cast<float>(x));
      context.sample_y = FloatBits(static_cast<float>(y));
      // With the old conflated origin the flipped first row is exactly 19,
      // outside an image of height 19. This is a real native SR/ALU result.
      const auto old_origin = ExecuteFragmentPco(program.summary, program.instructions, context);
      Check(old_origin.pixel_outputs[2] == FloatBits(19.0F - y),
            "native flip probe must expose the integer-origin boundary");
      context.special_coordinate_offset = FloatBits(0.5F);
      const auto physical = ExecuteFragmentPco(program.summary, program.instructions, context);
      Check(physical.pixel_outputs[0] == FloatBits(x + 0.5F) &&
                physical.pixel_outputs[1] == FloatBits(y + 0.5F) &&
                physical.pixel_outputs[2] == FloatBits(18.0F - y),
            "native FragCoord SR must retain physical center across Y flip");
    }
  }
  const auto varying = DecodePcoProgram(ShaderStage::kFragment,
                                        VaryingsOneFragmentPcoBinary());
  PcoFragmentExecutionContext context;
  context.sample_x = FloatBits(2.0F);
  context.sample_y = FloatBits(3.0F);
  context.coefficient_count = 20;
  context.coefficients[2] = FloatBits(1.0F);
  for (unsigned set = 1; set <= 4; ++set) {
    context.coefficients[set * 4] = FloatBits(0.25F);
    context.coefficients[set * 4 + 1] = FloatBits(0.125F);
    context.coefficients[set * 4 + 2] = FloatBits(set * 0.25F);
  }
  const auto before = ExecuteFragmentPco(varying.summary, varying.instructions, context);
  context.special_coordinate_offset = FloatBits(0.5F);
  const auto after = ExecuteFragmentPco(varying.summary, varying.instructions, context);
  Check(before.pixel_outputs == after.pixel_outputs,
        "physical SR offset must never bias native FITRP interpolation");
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 2) {
      for (unsigned kind = 0; kind <= 24; ++kind) {
        std::ifstream file(std::string(argv[1]) + "/texture-" + std::to_string(kind) + ".bin", std::ios::binary);
        const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(file), {}};
        Check(bytes == (kind <= 20 ? MultisampleTextureFixture(kind) : test::ExplicitTextureFixture(kind)),
              "fixture differs from freshly compiled native binary");
      }
    }
    TestNativeSampling();
    TestNativeExplicitSampling();
    TestNativeSpatialOffsetLookup();
    TestTextureDerivativeContinuation();
    TestNativeQueries();
    TestNativeStrideQueryLayout();
    TestMalformedFields();
    TestNativeFragmentCoordinates();
    std::cout << "native multisample texture ISS tests: PASS (" << checks << " checks)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
