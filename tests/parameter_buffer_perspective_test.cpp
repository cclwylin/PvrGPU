/*
 * ParameterBuffer non-equal-W perspective interpolation test.
 *
 * A synthetic triangle with W={1,2,4} enters the real event-driven
 * ParameterBuffer module.  The test checks its five public A/B/C/PAD sets,
 * then feeds those raw coefficient dwords to the exact Mesa PCO
 * FITRP.PIXEL/WDF fragment program.  This proves the fixed-function path
 * constructs varying/W and 1/W planes instead of taking an affine-color
 * shortcut hidden by GLBench varyings_shader_1's normal W=1 fixture.
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
  if (!condition) {
    throw std::runtime_error(
        "parameter-buffer perspective test failed: " + message);
  }
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
                std::uint32_t b, std::uint32_t c, const char *description) {
  const bool a_matches = IsZero(a) ? IsZero(plane.a) : plane.a == a;
  const bool b_matches = IsZero(b) ? IsZero(plane.b) : plane.b == b;
  Check(a_matches && b_matches && plane.c == c && plane.pad == 0,
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
    state.functional_case = FunctionalCase::kVaryingsShaderOne;
    state.stage = PipelineStage::kTiled;
    state.counters.c_primitives = 1;
    state.counters.setup_triangles = 1;

    ShaderVaryingBinding binding;
    binding.vertex_output_base = 4;
    binding.coefficient_set_base = 1;
    binding.w_coefficient_set = 0;
    binding.component_count = 4;
    binding.interpolation = InterpolationMode::kSmooth;
    state.shader_varying_bindings = StoreNewArray(
        pool, std::vector<ShaderVaryingBinding>{binding});

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
    triangle.vertex_output_stride_dwords = 8;
    triangle.front_facing = 1;
    triangle.rasterizable = 1;
    state.raster_triangles =
        StoreNewArray(pool, std::vector<RasterTriangle>{triangle});

    // Position VTXOUT0..3 is unused at this already-rasterized boundary.
    // VTXOUT4..7 values are R/G/B/A at each triangle vertex.
    std::vector<std::uint32_t> raster_outputs(3 * 8, 0);
    const float varyings[3][4] = {
        {0.0F, 0.0F, 1.0F, 1.0F},
        {1.0F, 0.0F, 0.0F, 1.0F},
        {0.0F, 1.0F, 0.0F, 1.0F},
    };
    for (std::size_t vertex = 0; vertex < 3; ++vertex) {
      for (std::size_t component = 0; component < 4; ++component) {
        raster_outputs[vertex * 8 + 4 + component] =
            FloatBits(varyings[vertex][component]);
      }
    }
    state.raster_vertex_outputs = StoreNewArray(pool, raster_outputs);

    const PoolHandle state_handle = pool.Allocate(sizeof(PipelineState));
    StorePipelineState(pool, state_handle, state);
    sc_core::sc_fifo<PipelineTxn> input("input", 1);
    sc_core::sc_fifo<PipelineTxn> output("output", 1);
    ParameterBuffer parameter_buffer("parameter_buffer", pool);
    parameter_buffer.input(input);
    parameter_buffer.output(output);

    input.write({state_handle, 1, 1});
    sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_NS));
    PipelineTxn completed;
    Check(output.nb_read(completed), "ParameterBuffer FIFO completion");
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
              parameters[0].coefficient_set_count == 5,
          "one triangle owns five coefficient sets");
    Check(coefficients.size() == 5 &&
              result.counters.parameter_coefficient_sets == 5 &&
              result.counters.parameter_write_bytes == 80,
          "coefficient payload and traffic counters");

    // At (x,y), Wrecip=1-x/4-3y/8. Numerator planes are R=x/4,
    // G=y/8, B=1-x/2-y/2, and A=Wrecip.
    CheckPlane(coefficients[0], FloatBits(-0.25F), FloatBits(-0.375F),
               FloatBits(1.0F), "non-constant reciprocal-W plane");
    CheckPlane(coefficients[1], FloatBits(0.25F), FloatBits(0.0F),
               FloatBits(0.0F), "R/W numerator plane");
    CheckPlane(coefficients[2], FloatBits(0.0F), FloatBits(0.125F),
               FloatBits(0.0F), "G/W numerator plane");
    CheckPlane(coefficients[3], FloatBits(-0.5F), FloatBits(-0.5F),
               FloatBits(1.0F), "B/W numerator plane");
    CheckPlane(coefficients[4], FloatBits(-0.25F), FloatBits(-0.375F),
               FloatBits(1.0F), "A/W numerator plane");

    PcoFragmentExecutionContext context;
    context.coefficient_count = 20;
    context.sample_x = FloatBits(0.5F);
    context.sample_y = FloatBits(0.5F);
    for (std::size_t set = 0; set < coefficients.size(); ++set) {
      context.coefficients[set * 4] = coefficients[set].a;
      context.coefficients[set * 4 + 1] = coefficients[set].b;
      context.coefficients[set * 4 + 2] = coefficients[set].c;
      context.coefficients[set * 4 + 3] = coefficients[set].pad;
    }
    const PcoDecodedProgram fragment =
        Decode(ShaderStage::kFragment, VaryingsOneFragmentPcoBinary());
    const PcoFragmentExecution execution =
        ExecuteFragment(fragment.summary, fragment.instructions, context);
    Check(execution.written_mask == 0x0f,
          "FITRP/WDF writes all four pixel outputs");
    Check(execution.pixel_outputs[0] == UINT32_C(0x3e3a2e8c) &&
              execution.pixel_outputs[1] == UINT32_C(0x3dba2e8c) &&
              execution.pixel_outputs[2] == UINT32_C(0x3f3a2e8c) &&
              execution.pixel_outputs[3] == UINT32_C(0x3f800000),
          "perspective result is exact 2/11, 1/11, 8/11, 1");
    Check(execution.pixel_outputs[0] != FloatBits(0.25F) &&
              execution.pixel_outputs[1] != FloatBits(0.25F) &&
              execution.pixel_outputs[2] != FloatBits(0.5F),
          "result cannot be an affine interpolation shortcut");

    ReleaseFunctionalPayloads(pool, result);
    pool.Release(state_handle);
    Check(pool.bytes_in_flight() == 0 &&
              pool.allocations() == pool.releases(),
          "MemoryPool allocations and releases are balanced");
    std::cout << "parameter_buffer_perspective_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "parameter_buffer_perspective_test: FAIL: " << error.what()
              << '\n';
    return 1;
  }
}
