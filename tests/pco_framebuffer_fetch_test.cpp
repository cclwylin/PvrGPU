#include "shader/pco_iss.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using pvrgpu::stub::DecodePcoProgram;
using pvrgpu::stub::ExecuteFragmentPco;
using pvrgpu::stub::PcoFragmentExecution;
using pvrgpu::stub::PcoFragmentExecutionContext;
using pvrgpu::stub::PcoRegisterBank;
using pvrgpu::stub::PcoRegisterRef;
using pvrgpu::stub::ShaderStage;

void Check(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <typename Function>
void ExpectFailure(Function &&function, const char *message) {
  try {
    function();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error(message);
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::vector<std::uint8_t> BytesFromHex(const char *text) {
  std::istringstream stream(text);
  std::vector<std::uint8_t> bytes;
  unsigned value = 0;
  while (stream >> std::hex >> value) {
    Check(value <= 0xffU, "test byte exceeds eight bits");
    bytes.push_back(static_cast<std::uint8_t>(value));
  }
  return bytes;
}

const std::vector<std::uint8_t> &FramebufferFetchBinary() {
  // Mesa PCO for `gl_FragColor = u_color + gl_LastFragData[0]`.
  static const auto binary = BytesFromHex(R"hex(
34 8a 00 87 20 00 00 40 34 8a 00 87 21 00 00 41
34 8a 00 87 22 00 00 42 34 8a 00 87 23 00 00 43
35 82 00 87 80 08 00 00 00 44 35 82 00 87 81 08 00 00 00 45
35 82 00 87 82 08 00 00 00 46 35 82 00 87 83 08 00 00 00 47
35 82 00 00 c4 a0 00 00 40 ff 35 82 00 00 c5 a1 00 00 41 ff
35 82 00 00 c6 a2 00 00 42 ff 35 82 00 00 c7 a3 00 00 43 ff
34 8a 00 87 40 00 00 20 34 8a 00 87 41 00 00 21
34 8a 00 87 42 00 00 22 34 8a 80 87 43 00 00 23
)hex");
  return binary;
}

void TestAttachmentSeedAndMutablePixelOutputs() {
  auto program =
      DecodePcoProgram(ShaderStage::kFragment, FramebufferFetchBinary());
  Check(pvrgpu::stub::PcoFragmentProgramReadsPixelInput(program.instructions),
        "framebuffer-fetch lowering was not identified as a PIXOUT read");
  Check(program.summary.early_hsr_safe == 0,
        "framebuffer-fetch shader was incorrectly marked opaque-HSR safe");

  // PIXOUT is a fragment-only source.  Reusing the exact bytes under another
  // stage must fail during decode rather than reaching a later executor.
  ExpectFailure(
      [&] {
        (void)DecodePcoProgram(ShaderStage::kVertex, FramebufferFetchBinary());
      },
      "vertex decoder accepted a fragment PIXOUT source");

  PcoFragmentExecutionContext context;
  context.pixel_input_mask = UINT16_C(0x000f);
  context.pixel_inputs[0] = FloatBits(0.5F);
  context.pixel_inputs[3] = FloatBits(1.0F);
  context.shared_count = 4;
  constexpr std::uint32_t kUniform = UINT32_C(0x3dd0d0d1);
  context.shared_registers[0] = kUniform;
  context.shared_registers[1] = kUniform;
  context.shared_registers[2] = kUniform;
  context.shared_registers[3] = FloatBits(1.0F);

  const auto result =
      ExecuteFragmentPco(program.summary, program.instructions, context);
  Check(result.written_mask == UINT16_C(0x000f) &&
            result.pixel_outputs[0] == UINT32_C(0x3f1a1a1a) &&
            result.pixel_outputs[1] == kUniform &&
            result.pixel_outputs[2] == kUniform &&
            result.pixel_outputs[3] == FloatBits(2.0F),
        "fragment ALU did not add uniforms to the seeded attachment value");

  // Instruction 12 stores the computed red channel into PIXOUT0.  Make the
  // next export read PIXOUT0: it must observe that store, not the immutable
  // attachment value present at invocation entry.
  program.instructions[13].source =
      PcoRegisterRef{PcoRegisterBank::kSpecial, 32};
  const auto reread =
      ExecuteFragmentPco(program.summary, program.instructions, context);
  Check(reread.pixel_outputs[1] == UINT32_C(0x3f1a1a1a) &&
            reread.pixel_outputs[1] != context.pixel_inputs[0],
        "PIXOUT reread did not observe the current mutable register value");

  context.pixel_input_mask = 0;
  ExpectFailure(
      [&] {
        (void)ExecuteFragmentPco(program.summary, program.instructions,
                                 context);
      },
      "fragment PIXOUT read accepted an absent attachment input");
}

void TestDerivativeContinuationPreservesPixelOutputs() {
  const auto derivative_binary = BytesFromHex(R"hex(
35 82 00 87 80 08 00 00 00 40
34 8a 00 87 40 00 00 20
34 82 00 88 40 00 00 41
34 8a 00 87 41 00 00 21
34 82 00 89 40 00 00 42
34 8a 00 87 42 00 00 22
35 8a 80 87 80 01 00 00 00 23
)hex");
  auto program = DecodePcoProgram(ShaderStage::kFragment, derivative_binary);
  // Load the destination attachment before the first suspension, export it,
  // then reread PIXOUT0 immediately after the first derivative resumes.
  program.instructions[0].source =
      PcoRegisterRef{PcoRegisterBank::kSpecial, 32};
  program.instructions[3].source =
      PcoRegisterRef{PcoRegisterBank::kSpecial, 32};

  PcoFragmentExecutionContext context;
  context.pixel_input_mask = 1;
  context.pixel_inputs[0] = FloatBits(0.25F);
  PcoFragmentExecution execution =
      ExecuteFragmentPco(program.summary, program.instructions, context);
  Check(execution.suspended && execution.derivative_request_valid &&
            execution.continuation.pixel_outputs[0] == FloatBits(0.25F) &&
            execution.continuation.written_mask == 1,
        "first derivative checkpoint lost the fetched PIXOUT value");

  context.continuation = execution.continuation;
  context.derivative_response_valid = 1;
  context.derivative_response = FloatBits(2.0F);
  // A resumed invocation must restore the checkpoint before reading PIXOUT;
  // changing the invocation-entry seed makes that distinction observable.
  context.pixel_inputs[0] = FloatBits(0.75F);
  execution =
      ExecuteFragmentPco(program.summary, program.instructions, context);
  Check(execution.suspended && execution.derivative_request_valid &&
            execution.continuation.pixel_outputs[0] == FloatBits(0.25F) &&
            execution.continuation.pixel_outputs[1] == FloatBits(0.25F) &&
            execution.continuation.written_mask == 3,
        "resumed derivative lane did not restore and reread current PIXOUT");

  context.continuation = execution.continuation;
  context.derivative_response = FloatBits(3.0F);
  execution =
      ExecuteFragmentPco(program.summary, program.instructions, context);
  Check(!execution.suspended && execution.written_mask == 15 &&
            execution.pixel_outputs[0] == FloatBits(0.25F) &&
            execution.pixel_outputs[1] == FloatBits(0.25F),
        "final derivative resume lost framebuffer-fetch PIXOUT state");
}

} // namespace

int main() {
  try {
    TestAttachmentSeedAndMutablePixelOutputs();
    TestDerivativeContinuationPreservesPixelOutputs();
  } catch (const std::exception &error) {
    std::cerr << "pco framebuffer-fetch test failed: " << error.what() << '\n';
    return 1;
  }
  std::cout << "pco framebuffer-fetch tests passed\n";
  return 0;
}
