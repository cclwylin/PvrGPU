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
#include <algorithm>
#include <iostream>
#include <limits>
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
  if (!(a_matches && b_matches && plane.c == c && plane.pad == 0))
    std::fprintf(stderr,
                 "%s actual=%08x,%08x,%08x,%08x expected=%08x,%08x,%08x,00000000\n",
                 description, plane.a, plane.b, plane.c, plane.pad, a, b, c);
  Check(a_matches && b_matches && plane.c == c && plane.pad == 0,
        description);
}

float BitsFloat(std::uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// Each failure fixture is a separate process: a SystemC process which throws
// must not be resumed to test another transaction. These are native module
// tests, not replay shortcuts or capture-specific exception rules.
void RunNonFiniteFixture(const std::string &mode) {
  MemoryPool pool;
  PipelineState state;
  state.width = state.height = 128;
  state.functional_case = FunctionalCase::kDriverPcoTriangles;
  state.stage = PipelineStage::kTiled;
  state.position_output_count = state.varying_output_start = 4;
  state.varying_output_count = 4;
  state.fragment_position_count = state.fragment_varying_start = 4;
  state.fragment_varying_count = 16;
  state.vertex_pco_abi.vertex_outputs = 8;
  state.fragment_pco_abi.coefficients = 20;
  ShaderVaryingBinding binding;
  binding.vertex_output_base = 4;
  binding.coefficient_set_base = 1;
  binding.w_coefficient_set = 0;
  binding.component_count = 4;
  binding.interpolation = InterpolationMode::kSmooth;

  // Independently synthesized collinear-in-binary32 setup. Fixed area is
  // 6009; exactly one pixel center (27,32) is covered. Of the 16x positions,
  // only sample 13 at (25,30) is covered. This is a visible negative control
  // against treating every collapsed floating-point area as invisible.
  RasterTriangle triangle;
  triangle.x[0] = BitsFloat(0x40db440a);
  triangle.y[0] = BitsFloat(0x4030b4ba);
  triangle.x[1] = BitsFloat(0x41c0d6d7);
  triangle.y[1] = BitsFloat(0x41dce174);
  triangle.x[2] = BitsFloat(0x42453c2a);
  triangle.y[2] = BitsFloat(0x427fa4bc);
  bool varyings_fail = false;
  bool reciprocal_fails = false;
  bool regular_edges = false;
  bool rollback = false;
  std::string expected_error;
  const std::string plane_error = "non-finite llvmpipe driver plane";
  const auto scissor = [&](std::uint32_t x0, std::uint32_t y0, std::uint32_t x1,
                           std::uint32_t y1) {
    state.raster_state.scissor = {1, x0, y0, x1, y1};
  };
  if (mode == "empty-depth" || mode == "empty-varying" ||
      mode == "empty-reciprocal" || mode == "rollback" ||
      mode == "invalid-reciprocal" || mode == "invalid-layout" ||
      mode == "invalid-setup") {
    // Recorded screen coordinates uniquely identify their binary32 values
    // at this magnitude. Depth/varying values below are intentionally
    // synthetic: the diagnostic's six decimal depth digits were not exact.
    triangle.x[0] = 478.912903f;
    triangle.y[0] = 499.939392f;
    triangle.x[1] = 478.919250f;
    triangle.y[1] = 499.949097f;
    triangle.x[2] = 478.906555f;
    triangle.y[2] = 499.929688f;
    state.width = state.height = 960;
    varyings_fail = mode == "empty-varying" || mode == "rollback";
    reciprocal_fails = mode == "empty-reciprocal";
    rollback = mode == "rollback";
    if (mode == "invalid-reciprocal")
      expected_error = "invalid reciprocal W";
    if (mode == "invalid-layout")
      expected_error = "invalid RasterTriangle metadata";
    if (mode == "invalid-setup")
      expected_error = "invalid Mesa setup vertex order";
    if (!expected_error.empty())
      state.raster_state.sample_mask = 0;
  } else if (mode == "covered") {
    expected_error = plane_error;
  } else if (mode == "zero-mask") {
    state.raster_state.sample_mask = 0;
  } else if (mode == "scissor-empty") {
    scissor(0, 0, 1, 1);
  } else if (mode == "scissor-boundary") {
    scissor(28, 32, 29, 33);
  } else if (mode == "framebuffer-clipped") {
    state.width = 20;
  } else if (mode == "msaa-covered" || mode == "msaa-masked" ||
             mode == "msaa-disabled-covered" || mode == "msaa-disabled-empty" ||
             mode == "msaa-enabled-scissored") {
    state.raster_state.sample_count = 16;
    state.raster_state.sample_mask = 1U << 13;
    if (mode == "msaa-masked")
      state.raster_state.sample_mask = ~(1U << 13);
    if (mode == "msaa-disabled-covered" || mode == "msaa-disabled-empty")
      state.raster_state.multisample_enable = 0;
    if (mode == "msaa-disabled-empty" || mode == "msaa-enabled-scissored")
      scissor(25, 30, 26, 31);
    if (mode == "msaa-covered" || mode == "msaa-disabled-covered" ||
        mode == "msaa-enabled-scissored")
      expected_error = plane_error;
  } else if (mode == "msaa-2-empty" || mode == "msaa-4-empty" ||
             mode == "msaa-8-empty") {
    state.raster_state.sample_count = static_cast<unsigned>(mode[5] - '0');
  } else if (mode == "invalid-samples") {
    state.raster_state.sample_count = 3;
    state.raster_state.sample_mask = 0;
    expected_error = "unsupported raster sample_count";
  } else if (mode == "invalid-multisample") {
    state.raster_state.multisample_enable = 2;
    state.raster_state.sample_mask = 0;
    expected_error = "multisample flag is invalid";
  } else if (mode == "top-left-covered" || mode == "bottom-left-empty") {
    regular_edges = varyings_fail = true;
    triangle.x[0] = .5f;
    triangle.y[0] = .5f;
    triangle.x[1] = 2.5f;
    triangle.y[1] = .5f;
    triangle.x[2] = .5f;
    triangle.y[2] = 2.5f;
    scissor(1, 0, 2, 1); // The only sample lies exactly on the horizontal edge.
    state.raster_state.bottom_edge_rule = mode == "bottom-left-empty";
    if (mode == "top-left-covered")
      expected_error = plane_error + ":";
  } else {
    throw std::runtime_error("unknown non-finite fixture: " + mode);
  }

  if (!regular_edges) {
    const float dx01 = triangle.x[0] - triangle.x[1];
    const float dy01 = triangle.y[0] - triangle.y[1];
    const float dx20 = triangle.x[2] - triangle.x[0];
    const float dy20 = triangle.y[2] - triangle.y[0];
    const volatile float e = dx01 * dy20;
    const volatile float f = dy01 * dx20;
    Check(e - f == 0.f, "fixture has singular binary32 setup area");
  }
  const auto fixed_area = (QuantizeRasterSubpixel(triangle.x[1]) -
                           QuantizeRasterSubpixel(triangle.x[0])) *
                              (QuantizeRasterSubpixel(triangle.y[2]) -
                               QuantizeRasterSubpixel(triangle.y[0])) -
                          (QuantizeRasterSubpixel(triangle.y[1]) -
                           QuantizeRasterSubpixel(triangle.y[0])) *
                              (QuantizeRasterSubpixel(triangle.x[2]) -
                               QuantizeRasterSubpixel(triangle.x[0]));
  Check(fixed_area != 0, "fixture has nonzero fixed-point area");
  if (fixed_area < 0) {
    std::swap(triangle.x[1], triangle.x[2]);
    std::swap(triangle.y[1], triangle.y[2]);
  }
  triangle.rasterizable = triangle.front_facing = 1;
  triangle.vertex_output_stride_dwords = mode == "invalid-layout" ? 7 : 8;
  triangle.setup_vertex_order[0] = 0;
  triangle.setup_vertex_order[1] = 1;
  triangle.setup_vertex_order[2] = mode == "invalid-setup" ? 1 : 2;
  for (unsigned vertex = 0; vertex != 3; ++vertex) {
    triangle.window_z[vertex] =
        varyings_fail || reciprocal_fails ? .5f : .25f * (vertex + 1);
    triangle.reciprocal_w[vertex] = reciprocal_fails ? 1.f / (vertex + 1) : 1.f;
  }
  if (mode == "invalid-reciprocal")
    triangle.reciprocal_w[0] = 0.f;

  std::vector<RasterTriangle> triangles{triangle};
  if (rollback) {
    RasterTriangle finite = triangle;
    finite.x[0] = 0.f;
    finite.y[0] = 0.f;
    finite.x[1] = 2.f;
    finite.y[1] = 0.f;
    finite.x[2] = 0.f;
    finite.y[2] = 2.f;
    triangles = {finite, triangle, finite};
  }
  std::vector<std::uint32_t> outputs(triangles.size() * 24, FloatBits(1.f));
  for (std::size_t index = 0; index < triangles.size(); ++index) {
    triangles[index].first_vertex_output_dword =
        static_cast<std::uint32_t>(index * 24);
    triangles[index].key.submit_ordinal = 100 + index;
    triangles[index].key.draw_id = 31;
    triangles[index].key.api_primitive_id = 17 + index;
    triangles[index].key.instance_id = 9;
    triangles[index].key.clip_piece = 2;
    if (varyings_fail && (!rollback || index == 1)) {
      for (unsigned vertex = 0; vertex != 3; ++vertex) {
        // A later component fails after W and previous components were
        // written, exercising rollback of a genuinely partial allocation.
        const float value =
            regular_edges ? (vertex == 1 ? -std::numeric_limits<float>::max()
                                         : std::numeric_limits<float>::max())
                          : float(vertex + 1);
        outputs[index * 24 + vertex * 8 + 6] = FloatBits(value);
      }
    }
  }
  state.counters.c_primitives = state.counters.setup_triangles =
      triangles.size();
  state.raster_triangles = StoreNewArray(pool, triangles);
  state.raster_vertex_outputs = StoreNewArray(pool, outputs);
  state.shader_varying_bindings =
      StoreNewArray(pool, std::vector<ShaderVaryingBinding>{binding});
  const PoolHandle handle = pool.Allocate(sizeof(PipelineState));
  StorePipelineState(pool, handle, state);
  sc_core::sc_fifo<PipelineTxn> input("input", 1), output("output", 1);
  ParameterBuffer parameter_buffer("parameter_buffer", pool);
  parameter_buffer.input(input);
  parameter_buffer.output(output);
  input.write({handle, 1, 1});
  try {
    sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_NS));
  } catch (const sc_core::sc_report &error) {
    const std::string message = error.what();
    Check(!expected_error.empty() &&
              message.find(expected_error) != std::string::npos,
          "unexpected failure in " + mode + ": " + message);
    if (regular_edges)
      Check(message.find("varying_output=6") != std::string::npos,
            "typed numerical exception retains varying context");
    PipelineTxn unexpected;
    Check(!output.nb_read(unexpected) &&
              LoadPipelineState(pool, handle).stage == PipelineStage::kTiled,
          "failure cannot publish a completed parameter buffer");
    ReleaseFunctionalPayloads(pool, state);
    pool.Release(handle);
    Check(pool.bytes_in_flight() == 0, "negative fixture pool balance");
    return;
  }
  Check(expected_error.empty(), mode + " must retain its original failure");
  PipelineTxn completed;
  Check(output.nb_read(completed), "empty triangle FIFO completion");
  const PipelineState result = LoadPipelineState(pool, handle);
  const auto parameters =
      LoadArray<ParameterTriangle>(pool, result.parameter_triangles);
  const auto coefficients =
      LoadArray<ParameterCoefficientSet>(pool, result.parameter_coefficients);
  Check(result.stage == PipelineStage::kParameterBufferReady &&
            parameters.size() == triangles.size() &&
            result.counters.c_primitives == triangles.size() &&
            result.counters.setup_triangles == triangles.size(),
        "empty setup preserves parameter indices and setup counters");
  const std::size_t skipped_index = rollback ? 1 : 0;
  const ParameterTriangle &skipped = parameters[skipped_index];
  Check(skipped.key.submit_ordinal == 100 + skipped_index &&
            skipped.key.api_primitive_id == 17 + skipped_index &&
            skipped.key.draw_id == 31 && skipped.key.instance_id == 9 &&
            skipped.key.clip_piece == 2 && skipped.front_facing == 1 &&
            skipped.face_culled == 0,
        "inactive placeholder retains primitive identity");
  Check(skipped.rasterizable == 0 && skipped.signed_area == 0 &&
            skipped.coefficient_set_count == 0 &&
            skipped.depth_plane_valid == 0 &&
            HasCanonicalDepthPlaneMetadata(result.functional_case, skipped),
        "inactive placeholder has canonical zero plane metadata");
  Check(skipped.min_x == 0 && skipped.min_y == 0 && skipped.max_x == 0 &&
            skipped.max_y == 0,
        "stale tiler references see empty half-open ISP bounds");
  // ISP intersects every stale tile/scissor reference with these clamped
  // half-open bounds. Even a tile at the origin cannot enter either loop.
  Check(!(std::max(0, skipped.min_x) < std::max(0, skipped.max_x)) &&
            !(std::max(0, skipped.min_y) < std::max(0, skipped.max_y)),
        "ISP's clamped half-open loops cannot interpolate the placeholder");
  const unsigned expected_sets = rollback ? 10 : 0;
  Check(coefficients.size() == expected_sets &&
            result.counters.parameter_coefficient_sets == expected_sets &&
            result.counters.parameter_write_bytes ==
                expected_sets * sizeof(ParameterCoefficientSet),
        "partial coefficients are rolled back, with exact payload counters");
  if (rollback) {
    Check(parameters[0].first_coefficient_set == 0 &&
              parameters[0].coefficient_set_count == 5 &&
              skipped.first_coefficient_set == 5 &&
              parameters[2].first_coefficient_set == 5 &&
              parameters[2].coefficient_set_count == 5 &&
              parameters[0].rasterizable && parameters[2].rasterizable,
          "surrounding finite triangles keep contiguous coefficient ownership");
    for (const auto &coefficient : coefficients)
      CheckPlane(coefficient, 0, 0, FloatBits(1.f),
                 "surrounding finite planes unchanged");
  }
  ReleaseFunctionalPayloads(pool, result);
  pool.Release(handle);
  Check(pool.bytes_in_flight() == 0 && pool.allocations() == pool.releases(),
        "non-finite fixture MemoryPool balance");
}

} // namespace

int sc_main(int argc, char **argv) {
  try {
    if (argc == 2) {
      RunNonFiniteFixture(argv[1]);
      std::cout << "parameter_buffer_perspective_test: " << argv[1]
                << ": PASS\n";
      return 0;
    }
    Check(argc == 1, "expected zero or one fixture argument");
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
    state.shader_varying_bindings =
        StoreNewArray(pool, std::vector<ShaderVaryingBinding>{binding});

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

    // Exact screen positions from a fractional-odd TES edge: float setup
    // area is zero, but independent subpixel rounding produces a nonzero
    // fixed triangle. Its constant depth/W/varyings still have finite planes.
    PipelineState constant;
    constant.width = constant.height = 256;
    constant.functional_case = FunctionalCase::kDriverPcoTriangles;
    constant.stage = PipelineStage::kTiled;
    constant.counters.c_primitives = constant.counters.setup_triangles = 1;
    constant.position_output_count = constant.varying_output_start = 4;
    constant.varying_output_count = 4;
    constant.fragment_position_count = constant.fragment_varying_start = 4;
    constant.fragment_varying_count = 16;
    constant.vertex_pco_abi.vertex_outputs = 8;
    constant.fragment_pco_abi.coefficients = 20;
    constant.shader_varying_bindings = StoreNewArray(pool,
        std::vector<ShaderVaryingBinding>{binding});
    RasterTriangle edge;
    edge.x[0] = 77.206314f; edge.y[0] = 247.872986f;
    edge.x[1] = 78.560760f; edge.y[1] = 249.227432f;
    edge.x[2] = 75.851852f; edge.y[2] = 246.518524f;
    const float dx01 = edge.x[0] - edge.x[1], dy01 = edge.y[0] - edge.y[1];
    const float dx20 = edge.x[2] - edge.x[0], dy20 = edge.y[2] - edge.y[0];
    // Match setup's separately rounded float products. A contracted FMA in
    // this assertion alone would retain extra precision and miss the boundary.
    const volatile float product_e = dx01 * dy20;
    const volatile float product_f = dy01 * dx20;
    Check(product_e - product_f == 0.f, "fixture retains the zero float-area boundary");
    const auto fixed_area = (QuantizeRasterSubpixel(edge.x[1])-QuantizeRasterSubpixel(edge.x[0])) *
        (QuantizeRasterSubpixel(edge.y[2])-QuantizeRasterSubpixel(edge.y[0])) -
        (QuantizeRasterSubpixel(edge.y[1])-QuantizeRasterSubpixel(edge.y[0])) *
        (QuantizeRasterSubpixel(edge.x[2])-QuantizeRasterSubpixel(edge.x[0]));
    Check(fixed_area != 0, "fixture remains a nonzero fixed-point triangle");
    if (fixed_area < 0) {
      std::swap(edge.x[1],edge.x[2]); std::swap(edge.y[1],edge.y[2]);
    }
    edge.vertex_output_stride_dwords = 8; edge.rasterizable = edge.front_facing = 1;
    edge.setup_vertex_order[0] = 0; edge.setup_vertex_order[1] = 1; edge.setup_vertex_order[2] = 2;
    std::vector<std::uint32_t> constant_outputs(24,0);
    for (unsigned vertex = 0; vertex != 3; ++vertex) {
      edge.window_z[vertex] = .5f; edge.reciprocal_w[vertex] = 1.f;
      for (unsigned component = 0; component != 4; ++component)
        constant_outputs[vertex*8+4+component] = FloatBits(float(component+1));
    }
    constant.raster_triangles = StoreNewArray(pool,std::vector<RasterTriangle>{edge});
    constant.raster_vertex_outputs = StoreNewArray(pool,constant_outputs);
    const auto constant_handle = pool.Allocate(sizeof(PipelineState));
    StorePipelineState(pool,constant_handle,constant);
    input.write({constant_handle,2,1});
    sc_core::sc_start(sc_core::sc_time(100,sc_core::SC_NS));
    Check(output.nb_read(completed), "constant edge FIFO completion");
    const auto constant_result = LoadPipelineState(pool,constant_handle);
    const auto constant_parameters = LoadArray<ParameterTriangle>(pool,constant_result.parameter_triangles);
    const auto constant_coefficients = LoadArray<ParameterCoefficientSet>(pool,constant_result.parameter_coefficients);
    Check(constant_parameters.size() == 1 && constant_parameters[0].rasterizable &&
          constant_parameters[0].depth_plane_valid && constant_coefficients.size() == 5,
          "constant attributes preserve the real primitive and coefficient layout");
    const auto &depth = constant_parameters[0].depth_plane;
    Check(IsZero(depth[0]) && IsZero(depth[1]) && depth[2] == FloatBits(.5f) && !depth[3],
          "constant depth does not become zero times infinity");
    for (unsigned set = 0; set != 5; ++set)
      CheckPlane(constant_coefficients[set],0,0,FloatBits(float(set ? set : 1)),
                 "constant reciprocal-W/varying has exact finite coefficients");
    ReleaseFunctionalPayloads(pool,constant_result); pool.Release(constant_handle);
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
