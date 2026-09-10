// Independent SystemC polygon-offset regression.  A synthetic triangle enters
// the real ParameterBuffer and its output is then consumed by the real ISP.
// No replay fixture or capture-specific geometry participates in the oracle.

#include "common/functional_types.h"
#include "common/pipeline_state.h"
#include "fragment/isp.h"
#include "geometry/parameter_buffer.h"

#include <systemc>

#include <algorithm>
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
    throw std::runtime_error("polygon offset test failed: " + message);
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float StrictMultiply(float lhs, float rhs) {
  const volatile float result = lhs * rhs;
  return result;
}

float StrictAdd(float lhs, float rhs) {
  const volatile float result = lhs + rhs;
  return result;
}

float RoundToFloat(double value) {
  const volatile float result = static_cast<float>(value);
  return result;
}

struct OffsetCase {
  const char *name;
  std::uint32_t format;
  float factor;
  float units;
  float clamp;
  bool enabled = true;
  bool units_unscaled = false;
};

bool IsFloatDepth(std::uint32_t format) {
  return format == kDriverPcoDepthFormatZ32Float ||
         format == kDriverPcoDepthFormatZ32FloatS8X24Uint;
}

float ExpectedOffset(const OffsetCase &test) {
  if (!test.enabled)
    return 0.0F;
  float units = test.units;
  if (!test.units_unscaled && !IsFloatDepth(test.format) && units != 0.0F) {
    const std::uint64_t maximum =
        test.format == kDriverPcoDepthFormatZ16Unorm ? UINT64_C(0xffff)
        : (test.format == kDriverPcoDepthFormatZ24X8Unorm ||
           test.format == kDriverPcoDepthFormatZ24UnormS8Uint)
            ? UINT64_C(0xffffff)
        : test.format == kDriverPcoDepthFormatZ32Unorm
            ? UINT64_C(0xffffffff)
            : throw std::runtime_error(
                  "polygon offset oracle format is unsupported");
    const float adjustment = units > 0.0F ? 0.5F : -0.5F;
    units = RoundToFloat(static_cast<double>(StrictAdd(units, adjustment)) /
                         static_cast<double>(maximum));
  }
  if (IsFloatDepth(test.format) && !test.units_unscaled) {
    // The fixture's maximum vertex depth is 0.296875, whose exponent produces
    // 2^-25 after llvmpipe subtracts the 23 binary32 mantissa bits.
    units = StrictMultiply(units, 0x1p-25F);
  }
  // The independently chosen plane is z=x/64+y/32+1/8, so max slope is 1/32.
  float result = StrictAdd(units, StrictMultiply(0x1p-5F, test.factor));
  if (test.clamp > 0.0F && result > test.clamp)
    result = test.clamp;
  else if (test.clamp < 0.0F && result < test.clamp)
    result = test.clamp;
  return result;
}

struct Modules {
  MemoryPool pool;
  sc_core::sc_fifo<PipelineTxn> parameter_input{"parameter_input", 1};
  sc_core::sc_fifo<PipelineTxn> parameter_output{"parameter_output", 1};
  sc_core::sc_fifo<PipelineTxn> isp_input{"isp_input", 1};
  sc_core::sc_fifo<PipelineTxn> isp_output{"isp_output", 1};
  ParameterBuffer parameter_buffer{"parameter_buffer", pool};
  Isp isp{"isp", pool};

  Modules() {
    parameter_buffer.input(parameter_input);
    parameter_buffer.output(parameter_output);
    isp.input(isp_input);
    isp.output(isp_output);
  }
};

void RunCase(Modules &modules, const OffsetCase &test, std::uint64_t sequence) {
  PipelineState state;
  state.width = state.height = 8;
  state.sequence = sequence;
  state.functional_case = FunctionalCase::kDriverPcoTriangles;
  state.stage = PipelineStage::kTiled;
  state.counters.c_primitives = state.counters.setup_triangles = 1;
  state.position_output_count = 4;
  state.fragment_position_count = 4;
  state.vertex_pco_abi.vertex_outputs = 4;
  state.fragment_pco_abi.coefficients = 4;
  state.raster_state.depth.test_enable = 1;
  state.raster_state.depth.write_enable = 1;
  state.raster_state.depth.compare_op = DepthCompareOp::kAlways;
  state.raster_state.depth.clear_depth = 1.0F;
  state.raster_state.polygon_offset_enable = test.enabled ? 1 : 0;
  state.raster_state.polygon_offset_units_unscaled =
      test.units_unscaled ? 1 : 0;
  state.raster_state.polygon_offset_factor = test.factor;
  state.raster_state.polygon_offset_units = test.units;
  state.raster_state.polygon_offset_clamp = test.clamp;
  state.fragment_early_hsr_safe = 1;
  state.depth_attachment_format = test.format;
  state.capture_depth_attachment = 1;

  RasterTriangle triangle;
  triangle.key.submit_ordinal = sequence;
  triangle.key.api_primitive_id = static_cast<std::uint32_t>(sequence);
  triangle.x[0] = 1.5F;
  triangle.y[0] = 1.5F;
  triangle.x[1] = 5.5F;
  triangle.y[1] = 1.5F;
  triangle.x[2] = 1.5F;
  triangle.y[2] = 5.5F;
  triangle.setup_vertex_order[0] = 0;
  triangle.setup_vertex_order[1] = 1;
  triangle.setup_vertex_order[2] = 2;
  triangle.vertex_output_stride_dwords = 4;
  triangle.rasterizable = triangle.front_facing = 1;
  // Values of z=x/64+y/32+1/8 at each setup vertex's (x-.5,y-.5).
  triangle.window_z[0] = 0.171875F;
  triangle.window_z[1] = 0.234375F;
  triangle.window_z[2] = 0.296875F;
  state.raster_triangles =
      StoreNewArray(modules.pool, std::vector<RasterTriangle>{triangle});
  state.raster_vertex_outputs =
      StoreNewArray(modules.pool, std::vector<std::uint32_t>(12));

  const PoolHandle state_handle = modules.pool.Allocate(sizeof(PipelineState));
  StorePipelineState(modules.pool, state_handle, state);
  const PipelineTxn txn{state_handle, static_cast<std::uint32_t>(sequence),
                        sequence};
  modules.parameter_input.write(txn);
  sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_NS));
  PipelineTxn completed;
  Check(modules.parameter_output.nb_read(completed) &&
            completed.sequence == sequence,
        std::string(test.name) + " ParameterBuffer completion");

  PipelineState parameter_state = LoadPipelineState(modules.pool, state_handle);
  const std::vector<ParameterTriangle> parameters =
      LoadArray<ParameterTriangle>(modules.pool,
                                   parameter_state.parameter_triangles);
  Check(parameter_state.stage == PipelineStage::kParameterBufferReady &&
            parameters.size() == 1,
        std::string(test.name) + " parameter payload");
  const ParameterTriangle &parameter = parameters.front();
  Check(parameter.depth_plane[0] == FloatBits(0x1p-6F) &&
            parameter.depth_plane[1] == FloatBits(0x1p-5F) &&
            parameter.depth_plane[2] == FloatBits(0x1p-3F) &&
            parameter.depth_plane[3] == 0 && parameter.depth_plane_valid == 1,
        std::string(test.name) + " exact llvmpipe depth plane");
  const float expected_offset = ExpectedOffset(test);
  Check(parameter.depth_offset == FloatBits(expected_offset),
        std::string(test.name) + " exact per-primitive offset");
  Check(HasCanonicalDepthPlaneMetadata(parameter_state.functional_case,
                                       parameter),
        std::string(test.name) + " canonical depth metadata");

  parameter_state.tile_records =
      StoreNewArray(modules.pool, std::vector<TileRecord>{{0, 0, 8, 8, 0, 1}});
  parameter_state.tile_primitive_refs = StoreNewArray(
      modules.pool, std::vector<TilePrimitiveRef>{{0, 0, sequence}});
  parameter_state.scheduled_tiles = 1;
  parameter_state.stage = PipelineStage::kTilesScheduled;
  StorePipelineState(modules.pool, state_handle, parameter_state);
  modules.isp_input.write(txn);
  sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_NS));
  Check(modules.isp_output.nb_read(completed) && completed.sequence == sequence,
        std::string(test.name) + " ISP completion");

  const PipelineState result = LoadPipelineState(modules.pool, state_handle);
  const std::vector<FragmentCandidate> candidates =
      LoadArray<FragmentCandidate>(modules.pool, result.fragment_candidates);
  const std::vector<std::uint32_t> depth =
      LoadArray<std::uint32_t>(modules.pool, result.isp_depth_attachment);
  Check(result.stage == PipelineStage::kVisibilityReady &&
            !candidates.empty() && depth.size() == 64,
        std::string(test.name) + " ISP output payload");
  for (const FragmentCandidate &candidate : candidates) {
    const float base =
        std::fma(0x1p-5F, static_cast<float>(candidate.y),
                 std::fma(0x1p-6F, static_cast<float>(candidate.x), 0x1p-3F));
    const float expected_depth =
        std::clamp(StrictAdd(base, expected_offset), 0.0F, 1.0F);
    Check(FloatBits(candidate.depth) == FloatBits(expected_depth) &&
              FloatBits(candidate.sample_depth[0]) == FloatBits(expected_depth),
          std::string(test.name) + " ordered FMA+FADD interpolation");
    const std::size_t pixel =
        static_cast<std::size_t>(candidate.y) * result.width + candidate.x;
    Check(depth[pixel] ==
              EncodeDepthAttachmentUnorm(expected_depth, test.format),
          std::string(test.name) + " encoded depth write");
  }

  ReleaseFunctionalPayloads(modules.pool, result);
  modules.pool.Release(state_handle);
}

} // namespace

int sc_main(int, char **) {
  try {
    Modules modules;
    const std::vector<OffsetCase> cases = {
        {"z16", kDriverPcoDepthFormatZ16Unorm, 1.0F, 200.0F, 0.0F},
        {"z24x8", kDriverPcoDepthFormatZ24X8Unorm, 1.0F, 200.0F, 0.0F},
        {"z24s8", kDriverPcoDepthFormatZ24UnormS8Uint, 1.0F, 200.0F, 0.0F},
        {"z32", kDriverPcoDepthFormatZ32Unorm, 1.0F, 200.0F, 0.0F},
        {"z32f", kDriverPcoDepthFormatZ32Float, 1.0F, 200.0F, 0.0F},
        {"z32fs8", kDriverPcoDepthFormatZ32FloatS8X24Uint, 1.0F, 200.0F, 0.0F},
        {"positive-clamp", kDriverPcoDepthFormatZ24X8Unorm, 1.0F, 200.0F,
         0.01F},
        {"negative-clamp", kDriverPcoDepthFormatZ24X8Unorm, -1.0F, -200.0F,
         -0.01F},
        {"unscaled-float", kDriverPcoDepthFormatZ32Float, 0.0F, 0.125F, 0.0F,
         true, true},
        {"disabled", kDriverPcoDepthFormatZ24X8Unorm, 1.0F, 200.0F, 0.0F, false,
         false},
        {"upper-depth-clamp", kDriverPcoDepthFormatZ24X8Unorm, 100.0F, 0.0F,
         0.0F},
        {"lower-depth-clamp", kDriverPcoDepthFormatZ24X8Unorm, -100.0F, 0.0F,
         0.0F},
    };
    std::uint64_t sequence = 1;
    for (const OffsetCase &test : cases)
      RunCase(modules, test, sequence++);
    Check(modules.pool.bytes_in_flight() == 0 &&
              modules.pool.allocations() == modules.pool.releases(),
          "MemoryPool balance");
    std::cout << "polygon_offset_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "polygon_offset_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
