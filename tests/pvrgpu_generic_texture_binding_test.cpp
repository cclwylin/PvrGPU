/* SPDX-License-Identifier: MIT */
#include "shader/pco_iss.h"
#include "common/msaa.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <limits>

extern "C" void test_generic_texture_bindings(void);
extern "C" void check_generic_dynamic_uniform_program(const std::uint8_t *bytes,
    std::size_t size, bool vertex, unsigned base, unsigned range,
    unsigned push_start, unsigned shared_count, unsigned varying_start, bool branch)
{
   using namespace pvrgpu::stub;
   const auto program = DecodePcoProgram(vertex ? ShaderStage::kVertex : ShaderStage::kFragment,
      std::vector<std::uint8_t>(bytes, bytes + size));
   const auto bits = [](float value) {
      std::uint32_t word; std::memcpy(&word, &value, sizeof(word)); return word;
   };
   if (push_start != (vertex ? 0U : 20U) || shared_count != (vertex ? 96U : 120U))
      throw std::runtime_error("dynamic uniform descriptor/CB0 suffix is incorrect");
   for (std::uint32_t selector : {0U, 1U, range - 1, range, 0xffffffffU}) {
      for (unsigned flag = 0; flag < 2; ++flag) {
         PcoFragmentExecutionContext context;
         context.shared_count = shared_count;
         for (unsigned slot = 0; slot < range; ++slot) {
            for (unsigned component = 0; component < 4; ++component)
               context.shared_registers[push_start + (base + slot) * 4 + component] =
                  bits(static_cast<float>(slot * 4 + component) + .25f);
         }
         context.shared_registers[push_start] = selector;
         context.shared_registers[push_start + 1] = flag;
         const std::uint32_t index = selector + (branch && flag ? 1U : 0U);
         std::array<std::uint32_t, 4> expected{};
         if (index < range)
            std::copy_n(context.shared_registers.begin() + push_start + (base + index) * 4,
                        4, expected.begin());
         if (vertex) {
            PcoVertexExecutionContext vertex_context;
            vertex_context.shared_count = shared_count;
            std::copy_n(context.shared_registers.begin(), shared_count,
                        vertex_context.shared_registers.begin());
            const auto result = ExecuteVertexPco(program.summary, program.instructions,
               {0, 0, 0, bits(1)}, vertex_context);
            if (result.suspended)
               throw std::runtime_error("dynamic vertex uniform unexpectedly suspended");
            for (unsigned component = 0; component < 4; ++component)
               if (result.outputs[varying_start + component] != expected[component])
                  throw std::runtime_error("dynamic vertex uniform in/out-of-range selection mismatch");
         } else {
            auto result = ExecuteFragmentPco(program.summary, program.instructions, context);
            if (!result.suspended || !result.texture_request_valid ||
                result.texture_request.binding != 0)
               throw std::runtime_error("dynamic fragment uniform lost its independent texture descriptor");
            context.continuation = result.continuation;
            context.texture_response_valid = 1;
            context.texture_response = {};
            result = ExecuteFragmentPco(program.summary, program.instructions, context);
            if (result.suspended || result.written_mask != 15)
               throw std::runtime_error("dynamic fragment uniform did not complete");
            for (unsigned component = 0; component < 4; ++component)
               if (result.pixel_outputs[component] != expected[component])
                  throw std::runtime_error("dynamic fragment uniform in/out-of-range selection mismatch");
         }
      }
   }
}

extern "C" void check_generic_interpolation_program(const std::uint8_t *bytes,
    std::size_t size, unsigned kind)
{
   using namespace pvrgpu::stub;
   const auto program = DecodePcoProgram(ShaderStage::kFragment,
      std::vector<std::uint8_t>(bytes, bytes + size));
   const auto bits = [](float value) {
      std::uint32_t word; std::memcpy(&word, &value, sizeof(word)); return word;
   };
   const auto as_float = [](std::uint32_t word) {
      float value; std::memcpy(&value, &word, sizeof(value)); return value;
   };
   const std::array<std::array<float, 2>, 4> offsets{{{0, 0}, {.375f, -.25f}, {-.5f, .4375f}, {.0625f, -.5f}}};
   for (unsigned samples : {1U, 2U, 4U, 8U, 16U}) {
      for (unsigned id = 0; id < samples; ++id) {
         for (unsigned origin = 0; origin < 2; ++origin) {
            PcoFragmentExecutionContext context;
            context.raster_sample_count = samples;
            context.sample_id = id;
            context.special_coordinate_offset = bits(.5f);
            const float pixel_x = origin ? -1 : 13;
            const float pixel_y = origin ? -2 : 7;
            context.sample_x = bits(pixel_x);
            context.sample_y = bits(pixel_y);
            const auto sample = RasterSamplePosition(samples, id);
            context.sample_position_valid = 1;
            context.sample_position_x = bits(pixel_x + sample[0] / 16.0f - .5f);
            context.sample_position_y = bits(pixel_y + sample[1] / 16.0f - .5f);
            context.centroid_position_valid = 1;
            context.centroid_x = bits(pixel_x + .25f);
            context.centroid_y = bits(pixel_y - .375f);
            context.shared_count = 4;
            float dx = offsets[id % offsets.size()][0], dy = offsets[id % offsets.size()][1];
            if (kind == 2 || kind == 3) {
               context.shared_registers[0] = id;
               dx = sample[0] / 16.0f - .5f;
               dy = sample[1] / 16.0f - .5f;
            } else {
               context.shared_registers[0] = bits(dx);
               context.shared_registers[1] = bits(dy);
            }
            if (kind == 6) { dx = .25f; dy = -.375f; }
            context.coefficient_count = kind == 5 || kind == 7 ? 36 : 20;
            const float w_a = .0078125f, w_b = -.00390625f, w_c = 2;
            context.coefficients[0] = bits(w_a);
            context.coefficients[1] = bits(w_b);
            context.coefficients[2] = bits(w_c);
            for (unsigned component = 0; component < (kind == 5 || kind == 7 ? 8 : 4); ++component) {
               const unsigned base = 4 + component * 4;
               context.coefficients[base] = bits((component + 1) * .03125f);
               context.coefficients[base + 1] = bits((static_cast<int>(component) - 2) * .0625f);
               context.coefficients[base + 2] = bits(component + .25f);
            }
            const auto execution = ExecuteFragmentPco(program.summary, program.instructions, context);
            if (execution.suspended || execution.written_mask != 15)
               throw std::runtime_error("native explicit interpolation did not complete");
            const float x = pixel_x + dx, y = pixel_y + dy;
            const float w = std::fma(w_b, y, std::fma(w_a, x, w_c));
            for (unsigned channel = 0; channel < 4; ++channel) {
               const unsigned component = kind == 4 ? 2 : kind == 5 || kind == 7 ? channel + 4 : channel;
               float expected = std::fma((static_cast<int>(component) - 2) * .0625f, y,
                  std::fma((component + 1) * .03125f, x, component + .25f));
               if (!(kind & 1)) expected /= w;
               if (kind == 4 && channel == 3) expected = 1;
               if (kind == 5 && channel == 3)
                  expected = std::fma(-.125f, pixel_y, std::fma(.03125f, pixel_x + .0625f, .25f));
               if (kind == 7 && channel == 3)
                  expected = std::fma(-.125f, pixel_y, std::fma(.03125f, pixel_x, .25f));
               const float actual = as_float(execution.pixel_outputs[channel]);
               if (!std::isfinite(actual) || std::fabs(actual - expected) > 0.000002f) {
                  std::fprintf(stderr, "interpolation kind=%u samples=%u id=%u origin=%u channel=%u got=%g expected=%g\n",
                     kind, samples, id, origin, channel, actual, expected);
                  throw std::runtime_error("native explicit interpolation coefficient/sample/component mismatch");
               }
            }
         }
      }
   }
   if (kind == 2 || kind == 3) {
      for (std::uint32_t id : {0U, 1U, 159U, 160U, 161U, 162U, 0xffffffffU}) {
         PcoFragmentExecutionContext context;
         context.raster_sample_count = 1;
         context.sample_x = bits(13);
         context.sample_y = bits(7);
         context.special_coordinate_offset = bits(.5f);
         context.shared_count = 4;
         context.shared_registers[0] = id;
         context.coefficient_count = 20;
         context.coefficients[2] = bits(1);
         for (unsigned channel = 0; channel < 4; ++channel) {
            context.coefficients[4 + channel * 4] = bits(1);
            context.coefficients[6 + channel * 4] = bits(.5f + channel);
         }
         const auto result = ExecuteFragmentPco(program.summary, program.instructions, context);
         for (unsigned channel = 0; channel < 4; ++channel)
            if (result.suspended || result.written_mask != 15 ||
                result.pixel_outputs[channel] != bits(13.5f + channel))
               throw std::runtime_error("single-sample interpolateAtSample must ignore arbitrary sample index");
      }
   }
}
extern "C" void check_generic_shadow_program(const std::uint8_t *bytes,
    std::size_t size, const std::uint32_t *descriptor, unsigned kind,
    bool vertex, unsigned varying_start)
{
   using namespace pvrgpu::stub;
   const auto program = DecodePcoProgram(vertex ? ShaderStage::kVertex : ShaderStage::kFragment,
      std::vector<std::uint8_t>(bytes, bytes + size));
   const auto bits = [](float value) {
      std::uint32_t word; std::memcpy(&word, &value, sizeof(word)); return word;
   };
   const float nan = std::numeric_limits<float>::quiet_NaN();
   const std::array<std::array<float, 2>, 7> pairs{{
      {.25f, .75f}, {.5f, .5f}, {.75f, .25f}, {-2, 0}, {2, 1}, {nan, .5f}, {.5f, nan}}};
   for (unsigned unorm = 0; unorm < 2; ++unorm) {
      for (unsigned op = 0; op < 8; ++op) {
         for (const auto &pair : pairs) {
            /* UNORM clamp of NaN is implementation-defined; float-depth
             * comparisons still explicitly cover ordered/unordered NaN. */
            if (unorm && (std::isnan(pair[0]) || std::isnan(pair[1]))) continue;
            PcoFragmentExecutionContext context;
            context.shared_count = 24;
            std::copy(descriptor, descriptor + 20, context.shared_registers.begin());
            context.shared_registers[7] = unorm << 8;
            context.shared_registers[12] = op;
            context.shared_registers[20] = bits(.25f);
            context.shared_registers[21] = bits(.75f);
            context.shared_registers[22] = bits(pair[0]);
            context.shared_registers[23] = 0;
            const float ref = unorm ? std::fmin(1.0f, std::fmax(0.0f, pair[0])) : pair[0];
            const bool comparisons[] = {false, ref < pair[1], ref == pair[1], ref <= pair[1],
               ref > pair[1], ref != pair[1], ref >= pair[1], true};
            const auto expected = bits(comparisons[op] ? 1.0f : 0.0f);
            if (vertex) {
               PcoVertexExecutionContext vertex_context;
               vertex_context.shared_count = 24;
               std::copy_n(context.shared_registers.begin(), 24, vertex_context.shared_registers.begin());
               auto execution = ExecuteVertexPco(program.summary, program.instructions,
                  {0, 0, 0, bits(1)}, vertex_context);
               if (!execution.suspended || !execution.texture_request_valid ||
                   execution.texture_request.coordinates[0] != bits(.25f) ||
                   execution.texture_request.coordinates[1] != bits(.75f))
                  throw std::runtime_error("nearest vertex shadow did not execute native SMP");
               execution = ResumeVertexPco(program.summary, program.instructions,
                  execution.continuation, {bits(pair[1]), bits(.8125f), bits(.1875f), bits(1)});
               if (execution.suspended || execution.outputs[varying_start] != expected ||
                   execution.outputs[varying_start + 1] != expected ||
                   execution.outputs[varying_start + 2] != expected ||
                   execution.outputs[varying_start + 3] != bits(1))
                  throw std::runtime_error("native nearest vertex shadow comparison mismatch");
               continue;
            }
            auto execution = ExecuteFragmentPco(program.summary, program.instructions, context);
            if (!execution.suspended || !execution.texture_request_valid ||
                execution.texture_request.binding != 0 ||
                execution.texture_request.coordinates[0] != bits(.25f) ||
                execution.texture_request.coordinates[1] != bits(.75f) ||
                execution.texture_request.explicit_lod_present != (kind != 0))
               throw std::runtime_error("nearest shadow did not execute its real native texture read");
            context.continuation = execution.continuation;
            context.texture_response_valid = 1;
            /* Non-depth components deliberately differ: scalar shadow must
             * compare channel zero, never the texture's alpha or green. */
            context.texture_response = {bits(pair[1]), bits(.8125f), bits(.1875f), bits(1)};
            execution = ExecuteFragmentPco(program.summary, program.instructions, context);
            if (execution.suspended || execution.discarded || execution.written_mask != 15 ||
                execution.pixel_outputs != std::array<std::uint32_t, kPcoPixelOutputCount>{
                   expected, expected, expected, bits(1)}) {
               std::fprintf(stderr, "shadow kind=%u unorm=%u op=%u reference=%g depth=%g got=%08x expected=%08x\n",
                  kind, unorm, op, pair[0], pair[1], execution.pixel_outputs[0], expected);
               throw std::runtime_error("native nearest shadow comparison or reference clamp mismatch");
            }
         }
      }
   }
}
extern "C" void dump_generic_sample_program(const std::uint8_t *bytes,
    std::size_t size, unsigned kind)
{
   if (const char *directory = std::getenv("PVRGPU_PCO_SAMPLE_FIXTURE_DIR")) {
      const std::string path = std::string(directory) + "/sample-input-" + std::to_string(kind) + ".pco";
      FILE *file = std::fopen(path.c_str(), "wb");
      if (!file || std::fwrite(bytes, 1, size, file) != size)
         throw std::runtime_error("cannot write generated native sample fixture");
      std::fclose(file);
   }
}
extern "C" void check_generic_sample_program(const std::uint8_t *bytes,
    std::size_t size, unsigned kind, bool multisample_target)
{
   using namespace pvrgpu::stub;
   const auto program = DecodePcoProgram(ShaderStage::kFragment,
      std::vector<std::uint8_t>(bytes, bytes + size));
   const auto bits = [](float value) {
      std::uint32_t word; std::memcpy(&word, &value, sizeof(word)); return word;
   };
   for (unsigned samples : {1U, 2U, 4U, 8U, 16U}) {
      if (!multisample_target && samples != 1) continue;
      for (unsigned id = 0; id < samples; ++id) {
         PcoFragmentExecutionContext context;
         context.raster_sample_count = samples;
         context.sample_id = id;
         context.coverage_mask = 1U << id;
         const auto execution = ExecuteFragmentPco(program.summary, program.instructions, context);
         if (kind >= 3) {
            const bool discarded = (multisample_target && (kind == 5 ||
               ((kind == 3 || kind == 7) && (id & 1U) == 0))) || (kind == 6 && !(id & 1U));
            if (execution.suspended || execution.discarded != discarded ||
                (!discarded && (execution.written_mask != 15 ||
                 execution.pixel_outputs != std::array<std::uint32_t, kPcoPixelOutputCount>{0, 0, 0, bits(1)})) ||
                (!discarded && kind == 7 && (!execution.depth_written || execution.depth != bits(0.25f)))) {
               std::fprintf(stderr, "sample feedback kind=%u samples=%u id=%u discard=%u expected=%u depth=%u\n",
                   kind, samples, id, execution.discarded, discarded, execution.depth_written);
               throw std::runtime_error("native sample-mask final assignment/discard/depth feedback mismatch");
            }
            continue;
         }
         const auto position = RasterSamplePosition(samples, id);
         const float x = kind == 0 ? static_cast<float>(id) :
            kind == 1 ? static_cast<float>(1U << id) : position[0] / 16.0f;
         const float y = kind == 2 ? position[1] / 16.0f : 0;
         if (execution.suspended || execution.written_mask != 15 ||
             execution.pixel_outputs[0] != bits(x) || execution.pixel_outputs[1] != bits(y) ||
             execution.pixel_outputs[2] != 0 || execution.pixel_outputs[3] != bits(1))
            throw std::runtime_error("native sample ID/mask/position does not match runtime sample state");
      }
   }
}
extern "C" void check_generic_vertex_texture_program(const std::uint8_t *bytes,
    std::size_t size, const std::uint32_t *shared, unsigned shared_count,
    unsigned varying_start, unsigned kind)
{
   using namespace pvrgpu::stub;
   const auto program = DecodePcoProgram(ShaderStage::kVertex,
      std::vector<std::uint8_t>(bytes, bytes + size));
   PcoVertexExecutionContext context;
   context.shared_count = shared_count;
   std::copy(shared, shared + shared_count, context.shared_registers.begin());
   auto execution = ExecuteVertexPco(program.summary, program.instructions,
      {0, 0, 0, 0x3f800000U}, context);
   const auto as_float = [](std::uint32_t word) {
      float value; std::memcpy(&value, &word, sizeof(value)); return value;
   };
   if (kind >= 4 && kind <= 8) {
      const std::uint32_t expected_height = kind == 6 ? 0x40800000U : 0x40000000U;
      const std::uint32_t expected_depth = kind == 5 ? 0x40000000U : kind >= 7 ? 0x40800000U : 0;
      if (execution.suspended || execution.texture_request_valid ||
          execution.outputs[varying_start] != 0x40800000U ||
          execution.outputs[varying_start + 1] != expected_height ||
          execution.outputs[varying_start + 2] != expected_depth ||
          execution.outputs[varying_start + 3] != 0x3f800000U)
         throw std::runtime_error("vertex textureSize must read descriptor mip dimensions without SMP");
      return;
   }
   if (!execution.suspended || !execution.texture_request_valid)
      throw std::runtime_error("vertex texture must suspend on native SMP");
   const auto &request = execution.texture_request;
   const float expected_x = kind == 1 ? 0.125f : kind == 3 ? 3.0f : 0.25f;
   const float expected_y = kind == 1 ? 0.375f : kind == 3 ? 2.0f : 0.75f;
   if (request.descriptor_set != 0 || request.binding != 0 ||
       request.coordinates[0] == 0 ||
       as_float(request.coordinates[0]) != expected_x ||
       as_float(request.coordinates[1]) != expected_y ||
       !request.explicit_lod_present ||
       (kind == 9 ? (!std::isinf(as_float(request.explicit_lod)) || as_float(request.explicit_lod) > 0) :
         std::fabs(as_float(request.explicit_lod)) > 0.0001f) ||
       request.normalized != (kind == 3 ? 0 : 1))
      throw std::runtime_error("native vertex texture lost projected/gradient/offset coordinate or LOD semantics");
   if (kind == 9 && (request.spatial_offsets != std::array<std::int32_t,3>{7, 3, -8} ||
                    request.dimension != 3)) {
      throw std::runtime_error("zero-gradient 3D native sample lost signed spatial offsets");
   }
   const std::array<std::uint32_t, 4> response = {0x3e000000U, 0x3e800000U, 0x3f000000U, 0x3f800000U};
   execution = ResumeVertexPco(program.summary, program.instructions,
                              execution.continuation, response);
   if (execution.suspended || execution.texture_request_valid)
      throw std::runtime_error("vertex texture resume must finish after one response");
   for (unsigned c = 0; c < 4; ++c)
      if (execution.outputs[varying_start + c] != response[c])
         throw std::runtime_error("vertex texture response did not reach linked varying");
}

extern "C" void check_generic_derivative_program(const std::uint8_t *bytes,
    std::size_t size, unsigned kind)
{
   using namespace pvrgpu::stub;
   const auto program = DecodePcoProgram(ShaderStage::kFragment,
      std::vector<std::uint8_t>(bytes, bytes + size));
   if (!program.summary.uses_derivatives)
      throw std::runtime_error("graphics NIR derivative did not lower to native quad exchange");
   const auto bits = [](float value) {
      std::uint32_t word; std::memcpy(&word, &value, sizeof(word)); return word;
   };
   std::array<PcoFragmentExecutionContext, 4> contexts;
   std::array<PcoFragmentExecution, 4> lanes;
   const unsigned components = kind < 24 ? kind / 6 + 1 : kind - 22;
   const unsigned operation = kind < 24 ? kind % 6 : 6;
   const std::array<std::array<float, 4>, 4> inputs{{
      {1, 2, 4, -1}, {3, 7, 16, 6}, {6, 20, 8, 23}, {12, 30, 19, 42}}};
   for (unsigned lane = 0; lane < 4; ++lane) {
      contexts[lane].shared_count = 4;
      for (unsigned c = 0; c < 4; ++c)
         contexts[lane].shared_registers[c] = bits(inputs[lane][c]);
      lanes[lane] = ExecuteFragmentPco(program.summary, program.instructions, contexts[lane]);
   }
   const bool fine = operation == 2 || operation == 3;
   for (unsigned round = 0; round < components * (operation == 6 ? 2 : 1); ++round) {
      std::array<std::uint32_t, 4> sources;
      for (unsigned lane = 0; lane < 4; ++lane) {
         if (!lanes[lane].suspended || !lanes[lane].derivative_request_valid ||
             lanes[lane].texture_request_valid ||
             lanes[lane].continuation.resume_instruction_index != lanes[0].continuation.resume_instruction_index)
            throw std::runtime_error("compiled vector derivative lost a component's native quad exchange");
         sources[lane] = lanes[lane].derivative_source;
      }
      const auto &instruction = program.instructions[lanes[0].continuation.resume_instruction_index - 1];
      if ((operation != 6 && (instruction.opcode == PcoOpcode::kDerivativeY) != ((operation & 1U) != 0)) ||
          (instruction.derivative_fine != 0) != fine || instruction.repeat_count != 1)
         throw std::runtime_error("compiled derivative lost scalar axis/fine mode");
      const auto results = EvaluatePcoDerivativeQuad(instruction, sources);
      for (unsigned lane = 0; lane < 4; ++lane) {
         contexts[lane].continuation = lanes[lane].continuation;
         contexts[lane].derivative_response_valid = 1;
         contexts[lane].derivative_response = results[lane];
         lanes[lane] = ExecuteFragmentPco(program.summary, program.instructions, contexts[lane]);
      }
   }
   for (unsigned lane = 0; lane < 4; ++lane) {
      const auto &resumed = lanes[lane];
      if (resumed.suspended || resumed.written_mask != 15)
         throw std::runtime_error("compiled derivative failed native quad continuation/output check");
      for (unsigned c = 0; c < 4; ++c) {
         unsigned first, second;
         if (operation & 1) {
            first = fine ? lane & 1U : 0; second = first + 2;
         } else {
            first = fine ? lane & 2U : 0; second = first + 1;
         }
         const float expected = c >= components ? 0 : operation == 6 ?
            std::fabs(inputs[1][c] - inputs[0][c]) + std::fabs(inputs[2][c] - inputs[0][c]) :
            inputs[second][c] - inputs[first][c];
         if (resumed.pixel_outputs[c] != bits(expected))
            throw std::runtime_error("compiled vector derivative read an unwritten or another component's native result");
      }
   }
}
extern "C" void check_generic_explicit_texture_program(const std::uint8_t *bytes,
    std::size_t size, unsigned kind)
{
   using namespace pvrgpu::stub;
   const auto program = DecodePcoProgram(ShaderStage::kFragment,
      std::vector<std::uint8_t>(bytes, bytes + size));
   unsigned seen = 0;
   for (const auto &instruction : program.instructions) {
      if (instruction.opcode != PcoOpcode::kTextureSample) continue;
      if (!instruction.texture_lod_replace ||
          instruction.texture_non_normalized_coords != (kind != 3) ||
          instruction.source1.bank != PcoRegisterBank::kShared || instruction.source1.index != 0 ||
          instruction.source2.bank != PcoRegisterBank::kShared || instruction.source2.index != 8)
         throw std::runtime_error("explicit texture compiler lost native LOD/coordinate/descriptor contract");
      ++seen;
   }
   if (seen != 1) throw std::runtime_error("explicit texture compiler lost or invented SMP");
}
extern "C" void check_generic_texture_program(const std::uint8_t *bytes,
    std::size_t size, const unsigned *units, unsigned count)
{
   using namespace pvrgpu::stub;
   const auto program = DecodePcoProgram(ShaderStage::kFragment,
      std::vector<std::uint8_t>(bytes, bytes + size));
   unsigned seen = 0;
   for (const auto &instruction : program.instructions) {
      if (instruction.opcode != PcoOpcode::kTextureSample)
         continue;
      if (seen >= count || instruction.source1.bank != PcoRegisterBank::kShared ||
          instruction.source1.index != units[seen] * kPcoTextureDescriptorDwordCount ||
          instruction.source2.bank != PcoRegisterBank::kShared ||
          instruction.source2.index != units[seen] * kPcoTextureDescriptorDwordCount + 8)
         throw std::runtime_error("generic SMP does not address its Gallium texture unit");
      ++seen;
   }
   if (seen != count)
      throw std::runtime_error("generic compiler lost or invented a texture sample");
}

int main()
{
   try {
      test_generic_texture_bindings();
      std::puts("generic texture binding compiler/ISS tests passed");
      return 0;
   } catch (const std::exception &error) {
      std::fprintf(stderr, "%s\n", error.what());
      return 1;
   }
}
