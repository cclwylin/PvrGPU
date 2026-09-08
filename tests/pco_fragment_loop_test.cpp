#include "shader/pco_iss.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

using namespace pvrgpu::stub;
static unsigned checks;
static void Require(bool value, const char *message) {
  ++checks; if (!value) throw std::runtime_error(message);
}
static std::uint32_t Bits(float value) {
  std::uint32_t result; std::memcpy(&result, &value, 4); return result;
}
template<class F> static void Reject(F test, const char *message) {
  bool rejected = false;
  try { test(); } catch (const std::exception &) { rejected = true; }
  Require(rejected, message);
}

static void Test(const char *path, unsigned kind) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("missing native uniform-loop fixture");
  const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(file)), {});
  const auto program = DecodePcoProgram(ShaderStage::kFragment, bytes);
  Require(program.summary.uses_derivatives, "loop fixture lacks native derivative instructions");
  std::array<std::uint64_t, 3> zero_counts{}, one_counts{};
  const auto branch = std::find_if(program.instructions.begin(), program.instructions.end(),
      [](const auto &instruction) { return instruction.opcode == PcoOpcode::kBranch; });
  Require(branch != program.instructions.end(), "loop fixture has no native branch");
  for (const std::uint32_t relative : {UINT32_C(1), UINT32_C(0x7ffffffe)}) {
    auto malformed = bytes;
    for (unsigned byte = 0; byte < 4; ++byte)
      malformed[branch->binary_offset + 4 + byte] = (relative >> (8 * byte)) & 255;
    Reject([&] { DecodePcoProgram(ShaderStage::kFragment, malformed); },
           "malformed native branch byte address was accepted");
  }
  for (int count : {-2, 0, 1, 2, 4, 7, 64}) {
    std::array<PcoFragmentExecutionContext, 4> contexts;
    std::array<PcoFragmentExecution, 4> executions;
    std::array<std::uint64_t, 4> visited{};
    for (unsigned lane = 0; lane < 4; ++lane) {
      auto &context = contexts[lane];
      context.shared_count = 8;
      context.memory_side_effects_enabled = lane < 2 ? 1 : 0;
      for (unsigned component = 0; component < 4; ++component)
        context.shared_registers[component] = Bits(float(component + 1) *
            (1.f + (lane & 1 ? 2.f : 0.f) + (lane & 2 ? 3.f : 0.f)));
      context.shared_registers[4] = static_cast<std::uint32_t>(count);
      executions[lane] = ExecuteFragmentPco(program.summary, program.instructions, context);
      visited[lane] = executions[lane].executed_instruction_count;
    }
    unsigned exchanges = 0;
    while (executions[0].suspended) {
      Require(++exchanges < 1000, "native uniform-loop derivative did not terminate");
      const auto pc = executions[0].continuation.resume_instruction_index;
      std::array<std::uint32_t, 4> sources;
      for (unsigned lane = 0; lane < 4; ++lane) {
        Require(executions[lane].derivative_request_valid &&
                executions[lane].continuation.resume_instruction_index == pc,
                "uniform-loop native PC diverged across its quad");
        sources[lane] = executions[lane].derivative_source;
        Require(executions[lane].native_steps == visited[lane] &&
                executions[lane].continuation.native_steps == visited[lane] &&
                executions[lane].continuation.executed_instructions.alu ==
                    executions[lane].executed_instructions.alu,
                "native checkpoint lost or replayed its instruction counters");
      }
      const auto response = EvaluatePcoDerivativeQuad(program.instructions[pc - 1], sources);
      if (exchanges == 1) {
        auto forged = contexts[0];
        forged.continuation = executions[0].continuation;
        forged.derivative_response_valid = 1;
        forged.derivative_response = response[0];
        forged.continuation.execution_predicate = 2;
        Reject([&] { ExecuteFragmentPco(program.summary, program.instructions, forged); },
               "noncanonical saved native execution mask was accepted");
        forged.continuation.execution_predicate = 1;
        forged.continuation.native_steps = UINT64_C(10000000);
        Reject([&] { ExecuteFragmentPco(program.summary, program.instructions, forged); },
               "native loop watchdog restarted at a derivative checkpoint");
        forged.continuation = executions[0].continuation;
        forged.continuation.executed_instructions.alu = UINT64_MAX;
        Reject([&] { ExecuteFragmentPco(program.summary, program.instructions, forged); },
               "overflowed continuation instruction count was accepted");
      }
      for (unsigned lane = 0; lane < 4; ++lane) {
        contexts[lane].continuation = executions[lane].continuation;
        contexts[lane].derivative_response_valid = 1;
        contexts[lane].derivative_response = response[lane];
        executions[lane] = ExecuteFragmentPco(program.summary, program.instructions, contexts[lane]);
        visited[lane] += executions[lane].executed_instruction_count;
      }
    }
    const unsigned iterations = count > 0 ? count : 0;
    Require(exchanges == iterations * (kind ? 8 : 4), "wrong number of native loop iterations");
    const float factor = float(iterations * (iterations ? iterations - 1 : 0) / 2);
    for (const auto &result : executions) {
      Require(!result.suspended && !result.discarded && result.written_mask == 15,
              "native loop did not finish all fragment outputs");
      for (unsigned component = 0; component < 4; ++component)
        Require(result.pixel_outputs[component] == Bits(factor * float(component + 1) * (kind ? 5.f : 2.f)),
                "native uniform-loop derivative accumulated the wrong result");
      Require(result.executed_instructions.texture == 0 &&
              result.executed_instructions.memory == 0,
              "native derivative loop invented texture or memory executions");
      const std::array<std::uint64_t, 3> measured{
          result.native_steps, result.executed_instructions.alu,
          result.executed_instruction_count};
      if (count == 0) zero_counts = measured;
      if (count == 1) one_counts = measured;
      if (count >= 2) {
        for (unsigned field = 0; field < 2; ++field)
          Require(measured[field] == zero_counts[field] +
                    (one_counts[field] - zero_counts[field]) * iterations,
                  "dynamic native counters do not include every loop iteration exactly once");
      }
    }
    for (unsigned lane = 0; lane < 4; ++lane)
      Require(executions[lane].native_steps == visited[lane],
              "final native group count replayed a checkpoint segment");
  }
}

static void TestSampleMask(const char *path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("missing native sample-mask loop fixture");
  const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(file)), {});
  const auto program = DecodePcoProgram(ShaderStage::kFragment, bytes);
  for (unsigned samples : {1U, 2U, 4U, 8U, 16U}) {
    for (unsigned id = 0; id < samples; ++id) {
      for (unsigned duplicate = 0; duplicate < 2; ++duplicate) {
        if (samples == 1 && duplicate) continue;
        PcoFragmentExecutionContext context;
        context.raster_sample_count = samples;
        context.sample_id = id;
        context.coverage_mask = (1U << id) | (duplicate ? 1U << ((id + 1) % samples) : 0);
        context.shared_count = 8;
        context.shared_registers[0] = Bits(1.0f / samples);
        const auto result = ExecuteFragmentPco(program.summary, program.instructions, context);
        Require(!result.suspended && !result.discarded && result.written_mask == 15,
                "sample-mask native nested loops failed to complete");
        Require(result.pixel_outputs == std::array<std::uint32_t, kPcoPixelOutputCount>{
          Bits(float(duplicate)), Bits(duplicate ? 0.0f : float(id) / samples), Bits(float(id)), Bits(1)},
          "sample-mask early break/dynamic loop returned a wrong unique bit");
      }
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 3 && argc != 4) return 2;
  try {
    PcoInstruction repeated;
    repeated.opcode = PcoOpcode::kFloatAdd;
    repeated.repeat_count = kPcoMaximumCountedGroupRepeat;
    Require(CountPcoInstructions({repeated}, true).alu == kPcoMaximumCountedGroupRepeat,
            "maximum native group repeat does not bound its issue count");
    ++repeated.repeat_count;
    Reject([&] { CountPcoInstructions({repeated}, true); },
           "instruction accounting accepted a repeat above its native gate");
    const auto solid = DecodePcoProgram(ShaderStage::kFragment, FillSolidFragmentPcoBinary());
    const auto execution = ExecuteFragmentPco(solid.summary, solid.instructions);
    const auto expected = CountPcoInstructions(solid.instructions, true);
    Require(execution.native_steps == solid.instructions.size() &&
            execution.executed_instructions.alu == expected.alu &&
            execution.executed_instructions.texture == expected.texture &&
            execution.executed_instructions.memory == expected.memory,
            "straight-line shader dynamic counters changed the native baseline");
    Test(argv[1], 0); Test(argv[2], 1); if (argc == 4) TestSampleMask(argv[3]);
  }
  catch (const std::exception &error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
  std::printf("native fragment uniform-loop derivative: PASS (%u checks)\n", checks);
}
