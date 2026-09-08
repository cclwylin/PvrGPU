// SPDX-License-Identifier: MIT
// Actual GLSL/PCO bytes; sampler results are explicit unit inputs, not a
// claimed full raster result. Every response services a real native request.
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
static void Check(bool ok, const char *why) {
   ++checks; if (!ok) throw std::runtime_error(why);
}
static std::uint32_t Bits(float f) {
   std::uint32_t u; std::memcpy(&u, &f, 4); return u;
}
static PcoDecodedProgram Read(const char *path) {
   std::ifstream file(path, std::ios::binary);
   Check(bool(file), "open genuine GLSL/PCO discard binary");
   std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(file), {}};
   return DecodePcoProgram(ShaderStage::kFragment, bytes);
}

static void Conditional(const PcoDecodedProgram &program) {
   Check(program.summary.uses_derivatives, "GLSL fixture lost its derivatives");
   Check(std::count_if(program.instructions.begin(), program.instructions.end(),
      [](const auto &i) { return i.opcode == PcoOpcode::kTextureSample; }) == 2,
      "genuine compiled shader must have mask and color SMP instructions");
   Check(std::any_of(program.instructions.begin(), program.instructions.end(),
      [](const auto &i) { return i.opcode == PcoOpcode::kAlphaFeedback; }),
      "GLSL discard must reach native ALPHAF feedback");
   Check(std::all_of(program.instructions.begin(), program.instructions.end(),
      [](const auto &i) { return (i.opcode != PcoOpcode::kDerivativeX &&
                                  i.opcode != PcoOpcode::kDerivativeY) || !i.derivative_fine; }),
      "public PCO lowers plain GLSL dFdx/dFdy to coarse quad derivatives");
   for (unsigned kill_mask = 0; kill_mask != 16; ++kill_mask) {
      // Exercise both ordinary fragments and geometric helpers. All lanes
      // still provide the source needed by surviving lanes' derivatives.
      for (unsigned helper_mask : {0U, 4U, 12U, 15U}) {
         std::array<PcoFragmentExecutionContext, 4> context;
         std::array<PcoFragmentExecution, 4> result;
         std::array<std::uint64_t, 4> steps{};
         std::array<unsigned, 4> samples{}, derivatives{};
         std::array<float, 4> mask{};
         for (unsigned lane = 0; lane < 4; ++lane) {
            mask[lane] = kill_mask & (1U << lane) ? 0.125f * float(lane + 1) :
                                                              .625f + .0625f * lane;
            // Make lane 3 strictly below 0.5 when killed as well.
            if (kill_mask & (1U << lane)) mask[lane] *= .5f;
            context[lane].shared_count = 40;
            context[lane].memory_side_effects_enabled = !(helper_mask & (1U << lane));
            result[lane] = ExecuteFragmentPco(program.summary, program.instructions, context[lane]);
            steps[lane] = result[lane].executed_instruction_count;
         }
         unsigned rounds = 0;
         while (result[0].suspended) {
            Check(++rounds <= 16, "discard quad continuation must terminate");
            const auto pc = result[0].continuation.resume_instruction_index;
            const bool texture = result[0].texture_request_valid;
            std::array<std::uint32_t, 4> sources{};
            for (unsigned lane = 0; lane < 4; ++lane) {
               Check(result[lane].suspended &&
                     result[lane].continuation.resume_instruction_index == pc &&
                     bool(result[lane].texture_request_valid) == texture,
                     "discarded/helper lane skipped a required quad operation");
               Check(result[lane].native_steps == steps[lane],
                     "native checkpoint repeated or lost execution counts");
               context[lane].continuation = result[lane].continuation;
               context[lane].texture_response_valid = 0;
               context[lane].derivative_response_valid = 0;
               if (texture) {
                  const auto &request = result[lane].texture_request;
                  Check(samples[lane] < 2 && request.descriptor_set == samples[lane],
                        "only real mask then color sample requests may be serviced");
                  Check(request.coordinates[0] == Bits(.25f) && request.coordinates[1] == Bits(.75f),
                        "discard continuation changed native texture coordinates");
                  if (samples[lane] == 1)
                     Check(result[lane].discarded == bool(kill_mask & (1U << lane)),
                           "ALPHAF feedback must precede the second sample and survive suspension");
                  context[lane].texture_response[0] = Bits(samples[lane] ? .25f * (lane + 1) : mask[lane]);
                  context[lane].texture_response[1] = Bits(0);
                  context[lane].texture_response[2] = Bits(0);
                  context[lane].texture_response[3] = Bits(1);
                  context[lane].texture_response_valid = 1;
                  ++samples[lane];
               } else {
                  Check(result[lane].derivative_request_valid, "unknown native suspension");
                  sources[lane] = result[lane].derivative_source;
                  ++derivatives[lane];
               }
            }
            if (!texture) {
               const auto values = EvaluatePcoDerivativeQuad(program.instructions[pc - 1], sources);
               for (unsigned lane = 0; lane < 4; ++lane) {
                  context[lane].derivative_response = values[lane];
                  context[lane].derivative_response_valid = 1;
               }
            }
            for (unsigned lane = 0; lane < 4; ++lane) {
               result[lane] = ExecuteFragmentPco(program.summary, program.instructions, context[lane]);
               steps[lane] += result[lane].executed_instruction_count;
            }
         }
         for (unsigned lane = 0; lane < 4; ++lane) {
            const float dx = mask[1] - mask[0];
            const float dy = mask[2] - mask[0];
            Check(samples[lane] == 2 && derivatives[lane] == 2 &&
                  !result[lane].suspended && result[lane].native_steps == steps[lane],
                  "each lane must finish both genuine SMP and derivative instructions exactly once");
            Check(result[lane].discarded == bool(kill_mask & (1U << lane)),
                  "final discard flag must suppress killed fragments at the pixel back end");
            Check(result[lane].executed_instructions.texture == 2,
                  "dynamic texture counters must reflect two actual native requests");
            const bool outputs_match = result[lane].written_mask == 15 &&
                  result[lane].pixel_outputs[0] == Bits(.25f * (lane + 1)) &&
                  result[lane].pixel_outputs[1] == Bits(dx) &&
                  result[lane].pixel_outputs[2] == Bits(dy) &&
                  result[lane].pixel_outputs[3] == Bits(1);
            if (!outputs_match) std::fprintf(stderr,
               "kill=%x helper=%x lane=%u written=%x output=%08x,%08x,%08x,%08x expected=%08x,%08x,%08x,%08x\n",
               kill_mask, helper_mask, lane, result[lane].written_mask,
               result[lane].pixel_outputs[0], result[lane].pixel_outputs[1],
               result[lane].pixel_outputs[2], result[lane].pixel_outputs[3],
               Bits(.25f * (lane + 1)), Bits(dx), Bits(dy), Bits(1));
            Check(outputs_match,
                  "surviving fragment derivatives must include discarded/helper values");
         }
      }
   }
}

int main(int argc, char **argv) {
   if (argc != 3) return 2;
   try {
      Conditional(Read(argv[1]));
      const auto program = Read(argv[2]);
      const auto result = ExecuteFragmentPco(program.summary, program.instructions);
      Check(result.discarded && !result.suspended,
            "unconditional GLSL discard must finish with native rejection feedback");
   } catch (const std::exception &error) {
      std::fprintf(stderr, "%s\n", error.what()); return 1;
   }
   std::printf("native GLSL discard/sample/helper-derivative: PASS (%u checks)\n", checks);
}
