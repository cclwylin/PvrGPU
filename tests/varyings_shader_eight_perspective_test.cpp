/*
 * Gate-15 non-equal-W Parameter Buffer and fragment-USC proof.
 *
 * A W={1,2,4} triangle carries eight independently distinguishable smooth
 * vec4 values with dyadic scales summing to one. The production event-driven
 * ParameterBuffer builds one shared 1/W plane and 32 distinct varying/W
 * planes. The exact public varyings_shader_8 fragment PCO executes every
 * FITRP and left-associated FADD, ruling out affine and duplicated-varying
 * shortcuts. FIFO traffic contains only the PipelineState MemoryPool handle.
 */
#include "common/functional_types.h"
#include "common/pipeline_state.h"
#include "geometry/parameter_buffer.h"
#include "shader/pco_iss.h"

#include <systemc>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace pvrgpu::stub;

void Check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(
        "varyings_shader_eight perspective test failed: " + message);
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool IsZero(std::uint32_t bits) {
  return (bits & UINT32_C(0x7fffffff)) == 0;
}

void CheckPlane(const ParameterCoefficientSet &plane, std::uint32_t a,
                std::uint32_t b, std::uint32_t c,
                const char *description) {
  Check((IsZero(a) ? IsZero(plane.a) : plane.a == a) &&
            (IsZero(b) ? IsZero(plane.b) : plane.b == b) &&
            plane.c == c && plane.pad == 0,
        description);
}

} // namespace

int sc_main(int, char **) {
  try {
    MemoryPool pool;
    PipelineState state;
    state.width = 2;
    state.height = 2;
    state.sequence = 1;
    state.functional_case = FunctionalCase::kVaryingsShaderEight;
    state.stage = PipelineStage::kTiled;
    state.counters.c_primitives = 1;
    state.counters.setup_triangles = 1;

    std::vector<ShaderVaryingBinding> bindings;
    for (std::size_t varying = 0; varying < 8; ++varying) {
      ShaderVaryingBinding binding;
      binding.vertex_output_base =
          static_cast<std::uint16_t>(4 + varying * 4);
      binding.coefficient_set_base =
          static_cast<std::uint16_t>(1 + varying * 4);
      binding.w_coefficient_set = 0;
      binding.component_count = 4;
      binding.interpolation = InterpolationMode::kSmooth;
      Check(IsExactVaryingBinding(state.functional_case, binding, varying),
            "synthetic linkage follows the production ABI");
      bindings.push_back(binding);
    }
    state.shader_varying_bindings = StoreNewArray(pool, bindings);

    RasterTriangle triangle;
    triangle.x[0] = 0.0F;
    triangle.y[0] = 0.0F;
    triangle.x[1] = 2.0F;
    triangle.y[1] = 0.0F;
    triangle.x[2] = 0.0F;
    triangle.y[2] = 2.0F;
    triangle.reciprocal_w[0] = 1.0F;
    triangle.reciprocal_w[1] = 0.5F;
    triangle.reciprocal_w[2] = 0.25F;
    triangle.first_vertex_output_dword = 0;
    triangle.vertex_output_stride_dwords = 36;
    triangle.front_facing = 1;
    triangle.rasterizable = 1;
    state.raster_triangles =
        StoreNewArray(pool, std::vector<RasterTriangle>{triangle});

    const float c[3][4] = {
        {0.0F, 0.0F, 1.0F, 1.0F},
        {1.0F, 0.0F, 0.0F, 1.0F},
        {0.0F, 1.0F, 0.0F, 1.0F},
    };
    const float varying_scale[8] = {
        1.0F / 128.0F,  2.0F / 128.0F,  4.0F / 128.0F,
        8.0F / 128.0F,  16.0F / 128.0F, 24.0F / 128.0F,
        32.0F / 128.0F, 41.0F / 128.0F,
    };
    std::vector<std::uint32_t> raster_outputs(3 * 36, 0);
    for (std::size_t vertex = 0; vertex < 3; ++vertex) {
      for (std::size_t varying = 0; varying < 8; ++varying) {
        for (std::size_t component = 0; component < 4; ++component) {
          raster_outputs[vertex * 36 + 4 + varying * 4 + component] =
              FloatBits(c[vertex][component] * varying_scale[varying]);
        }
      }
    }
    state.raster_vertex_outputs = StoreNewArray(pool, raster_outputs);

    const PoolHandle state_handle = pool.Allocate(sizeof(PipelineState));
    StorePipelineState(pool, state_handle, state);
    sc_core::sc_fifo<PipelineTxn> input("input", 1);
    sc_core::sc_fifo<PipelineTxn> output("output", 1);
    ParameterBuffer parameter("parameter_buffer", pool);
    parameter.input(input);
    parameter.output(output);
    input.write({state_handle, 1, 1});
    sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_NS));

    PipelineTxn completed;
    Check(output.nb_read(completed), "ParameterBuffer event completes");
    const PipelineState result = LoadPipelineState(pool, state_handle);
    Check(result.stage == PipelineStage::kParameterBufferReady,
          "ParameterBuffer completion stage");
    const std::vector<ParameterTriangle> parameters =
        LoadArray<ParameterTriangle>(pool, result.parameter_triangles);
    const std::vector<ParameterCoefficientSet> coefficients =
        LoadArray<ParameterCoefficientSet>(pool,
                                           result.parameter_coefficients);
    Check(parameters.size() == 1 &&
              parameters[0].first_coefficient_set == 0 &&
              parameters[0].coefficient_set_count == 33 &&
              coefficients.size() == 33 &&
              result.counters.parameter_coefficient_sets == 33 &&
              result.counters.parameter_write_bytes == 528,
          "one triangle owns one 1/W plus 32 varying/W sets");

    CheckPlane(coefficients[0], FloatBits(-0.25F), FloatBits(-0.375F),
               FloatBits(1.0F), "shared non-constant reciprocal-W plane");
    for (std::size_t varying = 0; varying < 8; ++varying) {
      const std::size_t base = 1 + varying * 4;
      const float scale = varying_scale[varying];
      CheckPlane(coefficients[base], FloatBits(0.25F * scale),
                 FloatBits(0.0F), FloatBits(0.0F),
                 "scaled-R/W numerator plane");
      CheckPlane(coefficients[base + 1], FloatBits(0.0F),
                 FloatBits(0.125F * scale), FloatBits(0.0F),
                 "scaled-G/W numerator plane");
      CheckPlane(coefficients[base + 2], FloatBits(-0.5F * scale),
                 FloatBits(-0.5F * scale), FloatBits(scale),
                 "scaled-B/W numerator plane");
      CheckPlane(coefficients[base + 3], FloatBits(-0.25F * scale),
                 FloatBits(-0.375F * scale), FloatBits(scale),
                 "scaled-A/W numerator plane");
    }
    for (std::size_t varying = 1; varying < 8; ++varying) {
      Check(coefficients[1 + (varying - 1) * 4].a !=
                coefficients[1 + varying * 4].a,
            "every linkage range is independently observable");
    }

    PcoFragmentExecutionContext context;
    Check(context.coefficients.size() >= 132,
          "USC context holds eight-varying coefficient bank");
    context.coefficient_count = 132;
    context.sample_x = FloatBits(0.5F);
    context.sample_y = FloatBits(0.5F);
    for (std::size_t set = 0; set < coefficients.size(); ++set) {
      context.coefficients[set * 4] = coefficients[set].a;
      context.coefficients[set * 4 + 1] = coefficients[set].b;
      context.coefficients[set * 4 + 2] = coefficients[set].c;
      context.coefficients[set * 4 + 3] = coefficients[set].pad;
    }
    const PcoDecodedProgram fragment =
        Decode(ShaderStage::kFragment, VaryingsEightFragmentPcoBinary());
    const PcoFragmentExecution execution =
        ExecuteFragment(fragment.summary, fragment.instructions, context);
    Check(execution.written_mask == 0x0f,
          "eight FITRP values and left-associated FADD write RGBA");
    Check(execution.pixel_outputs[0] == UINT32_C(0x3e3a2e8c) &&
              execution.pixel_outputs[1] == UINT32_C(0x3dba2e8c) &&
              execution.pixel_outputs[2] == UINT32_C(0x3f3a2e8c) &&
              execution.pixel_outputs[3] == UINT32_C(0x3f800000),
          "eight distinct values reconstruct perspective 2/11,1/11,8/11,1");
    Check(execution.pixel_outputs[0] != FloatBits(0.25F) &&
              execution.pixel_outputs[1] != FloatBits(0.25F) &&
              execution.pixel_outputs[2] != FloatBits(0.5F),
          "result cannot be affine interpolation");

    ReleaseFunctionalPayloads(pool, result);
    pool.Release(state_handle);
    Check(pool.bytes_in_flight() == 0 &&
              pool.allocations() == pool.releases(),
          "MemoryPool ownership is balanced");
    std::cout << "varyings_shader_eight_perspective_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "varyings_shader_eight_perspective_test: FAIL: "
              << error.what() << '\n';
    return 1;
  }
}
