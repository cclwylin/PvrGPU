// Focused API-v7 depth-attachment lifecycle regression. ISP must preserve a
// complete LOAD surface (including untouched pixels) and quantize NEW_CLEAR
// through the declared Z16/Z24/Z32 UNORM format before later draws alias it.

#include "common/functional_types.h"
#include "common/pipeline_state.h"
#include "fragment/isp.h"

#include <systemc>

#include <array>
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
    throw std::runtime_error("ISP depth attachment test failed: " + message);
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

struct CasePayload {
  PoolHandle state;
  PipelineTxn txn;
};

CasePayload MakeCase(MemoryPool &pool, std::uint64_t sequence,
                     std::uint32_t format, float clear_depth,
                     const std::vector<std::uint32_t> &load) {
  PipelineState state;
  state.width = 3;
  state.height = 1;
  state.sequence = sequence;
  state.functional_case = FunctionalCase::kDriverPcoTriangles;
  state.stage = PipelineStage::kTilesScheduled;
  state.raster_state.sample_count = 1;
  state.raster_state.face_cull.enable = 1;
  state.raster_state.depth.test_enable = 1;
  state.raster_state.depth.write_enable = 1;
  state.raster_state.depth.compare_op = DepthCompareOp::kLessOrEqual;
  state.raster_state.depth.clear_depth = clear_depth;
  state.fragment_early_hsr_safe = 1;
  state.depth_attachment_format = format;
  state.capture_depth_attachment = 1;
  state.tile_records = StoreNewArray(pool, std::vector<TileRecord>{});
  state.tile_primitive_refs =
      StoreNewArray(pool, std::vector<TilePrimitiveRef>{});
  state.parameter_triangles =
      StoreNewArray(pool, std::vector<ParameterTriangle>{});
  if (!load.empty()) {
    Check(load.size() == 3, "LOAD pixel count");
    const std::vector<std::uint8_t> bytes =
        EncodeDepthAttachmentUnormBytes(load, format);
    state.depth_attachment_load = StoreNewArray(pool, bytes);
    state.depth_attachment_load_enable = 1;
    state.depth_attachment_load_bytes = bytes.size();
  }

  CasePayload payload;
  payload.state = pool.Allocate(sizeof(PipelineState));
  StorePipelineState(pool, payload.state, state);
  payload.txn.state = payload.state;
  payload.txn.frame = static_cast<std::uint32_t>(sequence);
  payload.txn.sequence = sequence;
  return payload;
}

CasePayload MakeDriverDepthPlaneCase(MemoryPool &pool,
                                     std::uint64_t sequence) {
  PipelineState state;
  state.width = 1;
  state.height = 1;
  state.sequence = sequence;
  state.functional_case = FunctionalCase::kDriverPcoTriangles;
  state.stage = PipelineStage::kTilesScheduled;
  state.raster_state.sample_count = 1;
  state.raster_state.depth.test_enable = 1;
  state.raster_state.depth.write_enable = 1;
  state.raster_state.depth.compare_op = DepthCompareOp::kLessOrEqual;
  state.raster_state.depth.clear_depth = 1.0F;
  state.fragment_early_hsr_safe = 1;
  state.depth_attachment_format = kDriverPcoDepthFormatZ32Unorm;
  state.capture_depth_attachment = 1;
  state.scheduled_tiles = 1;

  ParameterTriangle triangle;
  triangle.key.api_primitive_id = 7;
  triangle.rasterizable = 1;
  triangle.signed_area = 512LL * 512LL;
  triangle.min_x = 0;
  triangle.min_y = 0;
  triangle.max_x = 1;
  triangle.max_y = 1;
  // Coverage barycentrics would yield 0.75.  The driver position-Z plane is
  // deliberately 0.25, proving ISP depth uses the serialized llvmpipe plane
  // while retaining barycentrics only as fragment metadata.
  triangle.window_z[0] = 0.75F;
  triangle.window_z[1] = 0.75F;
  triangle.window_z[2] = 0.75F;
  triangle.edge[0] = {0, 512, 0, 1, {}};
  triangle.edge[1] = {-512, -512, 512LL * 512LL, 0, {}};
  triangle.edge[2] = {512, 0, 0, 0, {}};
  triangle.depth_plane[0] = FloatBits(0.0F);
  triangle.depth_plane[1] = FloatBits(0.0F);
  triangle.depth_plane[2] = FloatBits(0.25F);
  triangle.depth_plane_valid = 1;

  state.tile_records =
      StoreNewArray(pool, std::vector<TileRecord>{{0, 0, 1, 1, 0, 1}});
  state.tile_primitive_refs =
      StoreNewArray(pool, std::vector<TilePrimitiveRef>{{0, 0, 0}});
  state.parameter_triangles =
      StoreNewArray(pool, std::vector<ParameterTriangle>{triangle});

  CasePayload payload;
  payload.state = pool.Allocate(sizeof(PipelineState));
  StorePipelineState(pool, payload.state, state);
  payload.txn.state = payload.state;
  payload.txn.frame = static_cast<std::uint32_t>(sequence);
  payload.txn.sequence = sequence;
  return payload;
}

CasePayload MakeQuantizedDepthCompareCase(MemoryPool &pool,
                                          std::uint64_t sequence) {
  PipelineState state;
  state.width = 1;
  state.height = 1;
  state.sequence = sequence;
  state.functional_case = FunctionalCase::kDriverPcoTriangles;
  state.stage = PipelineStage::kTilesScheduled;
  state.raster_state.sample_count = 1;
  state.raster_state.depth.test_enable = 1;
  state.raster_state.depth.write_enable = 1;
  state.raster_state.depth.compare_op = DepthCompareOp::kLessOrEqual;
  state.raster_state.depth.clear_depth = 1.0F;
  state.fragment_early_hsr_safe = 1;
  state.depth_attachment_format = kDriverPcoDepthFormatZ16Unorm;
  state.capture_depth_attachment = 1;
  state.scheduled_tiles = 1;

  constexpr float first_depth = 0.9999373555F;
  constexpr float later_depth = 0.9999454021F;
  Check(first_depth < later_depth,
        "quantized compare fixture preserves raw depth order");
  Check(EncodeDepthAttachmentUnorm(first_depth,
                                   kDriverPcoDepthFormatZ16Unorm) ==
            EncodeDepthAttachmentUnorm(later_depth,
                                       kDriverPcoDepthFormatZ16Unorm),
        "quantized compare fixture aliases one Z16 value");

  auto make_triangle = [](std::uint64_t submit_ordinal,
                          std::uint32_t primitive_id,
                          float depth) {
    ParameterTriangle triangle;
    triangle.key.submit_ordinal = submit_ordinal;
    triangle.key.api_primitive_id = primitive_id;
    triangle.rasterizable = 1;
    triangle.signed_area = 512LL * 512LL;
    triangle.min_x = 0;
    triangle.min_y = 0;
    triangle.max_x = 1;
    triangle.max_y = 1;
    triangle.window_z[0] = depth;
    triangle.window_z[1] = depth;
    triangle.window_z[2] = depth;
    triangle.edge[0] = {0, 512, 0, 1, {}};
    triangle.edge[1] = {-512, -512, 512LL * 512LL, 0, {}};
    triangle.edge[2] = {512, 0, 0, 0, {}};
    triangle.depth_plane[0] = FloatBits(0.0F);
    triangle.depth_plane[1] = FloatBits(0.0F);
    triangle.depth_plane[2] = FloatBits(depth);
    triangle.depth_plane_valid = 1;
    return triangle;
  };
  const std::vector<ParameterTriangle> triangles = {
      make_triangle(10, 100, first_depth),
      make_triangle(11, 101, later_depth),
  };
  state.tile_records =
      StoreNewArray(pool, std::vector<TileRecord>{{0, 0, 1, 1, 0, 2}});
  state.tile_primitive_refs = StoreNewArray(
      pool, std::vector<TilePrimitiveRef>{{0, 0, 10}, {1, 0, 11}});
  state.parameter_triangles = StoreNewArray(pool, triangles);

  CasePayload payload;
  payload.state = pool.Allocate(sizeof(PipelineState));
  StorePipelineState(pool, payload.state, state);
  payload.txn.state = payload.state;
  payload.txn.frame = static_cast<std::uint32_t>(sequence);
  payload.txn.sequence = sequence;
  return payload;
}

void CheckCase(MemoryPool &pool, const CasePayload &payload,
               const std::vector<std::uint32_t> &expected) {
  const PipelineState state = LoadPipelineState(pool, payload.state);
  Check(state.stage == PipelineStage::kVisibilityReady, "completion stage");
  Check(state.active_fragment_invocations == 0,
        "face-culled setup emitted fragments");
  Check(HasPoolHandle(state.isp_depth_attachment),
        "missing complete depth surface");
  Check(LoadArray<std::uint32_t>(pool, state.isp_depth_attachment) == expected,
        "untouched/clear encoded depth values");
  ReleaseFunctionalPayloads(pool, state);
  pool.Release(payload.state);
}

void CheckDriverDepthPlaneCase(MemoryPool &pool,
                               const CasePayload &payload) {
  const PipelineState state = LoadPipelineState(pool, payload.state);
  Check(state.stage == PipelineStage::kVisibilityReady, "plane completion");
  Check(state.active_fragment_invocations == 1,
        "driver plane emitted one fragment");
  Check(LoadArray<std::uint32_t>(pool, state.isp_depth_attachment) ==
            std::vector<std::uint32_t>{UINT32_C(0x40000000)},
        "driver plane depth was FMA-evaluated and Z32-quantized");
  const std::vector<FragmentCandidate> candidates =
      LoadArray<FragmentCandidate>(pool, state.fragment_candidates);
  Check(candidates.size() == 1 && candidates[0].depth == 0.25F,
        "candidate carries the driver depth-plane result");
  ReleaseFunctionalPayloads(pool, state);
  pool.Release(payload.state);
}

void CheckQuantizedDepthCompareCase(MemoryPool &pool,
                                    const CasePayload &payload) {
  const PipelineState state = LoadPipelineState(pool, payload.state);
  Check(state.stage == PipelineStage::kVisibilityReady,
        "quantized compare completion");
  Check(state.active_fragment_invocations == 1,
        "quantized compare retained one owner");
  Check(LoadArray<std::uint32_t>(pool, state.isp_depth_attachment) ==
            std::vector<std::uint32_t>{UINT32_C(0xfffb)},
        "quantized compare stored aliased Z16 value");
  const std::vector<FragmentCandidate> candidates =
      LoadArray<FragmentCandidate>(pool, state.fragment_candidates);
  Check(candidates.size() == 2 &&
            candidates[0].visibility == FragmentVisibility::kRejected &&
            candidates[1].visibility == FragmentVisibility::kVisible &&
            candidates[1].primitive_id == 101,
        "LEQUAL compared encoded Z16 and selected the later owner");
  ReleaseFunctionalPayloads(pool, state);
  pool.Release(payload.state);
}

} // namespace

int sc_main(int, char **) {
  try {
    MemoryPool pool;
    sc_core::sc_fifo<PipelineTxn> input("input", 4);
    sc_core::sc_fifo<PipelineTxn> output("output", 4);
    Isp isp("isp", pool);
    isp.input(input);
    isp.output(output);

    const std::vector<std::uint32_t> z16_load = {0x1234U, 0xabcdU, 0xffffU};
    const CasePayload z16 = MakeCase(
        pool, 1, kDriverPcoDepthFormatZ16Unorm, 1.0F, z16_load);
    const CasePayload z24 = MakeCase(
        pool, 2, kDriverPcoDepthFormatZ24X8Unorm, 0.5F, {});
    const CasePayload depth_plane = MakeDriverDepthPlaneCase(pool, 3);
    const CasePayload quantized_compare =
        MakeQuantizedDepthCompareCase(pool, 4);
    input.write(z16.txn);
    input.write(z24.txn);
    input.write(depth_plane.txn);
    input.write(quantized_compare.txn);
    sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_NS));
    sc_core::sc_start(sc_core::SC_ZERO_TIME);

    PipelineTxn completed;
    Check(output.nb_read(completed) && completed.sequence == 1,
          "Z16 FIFO order");
    Check(output.nb_read(completed) && completed.sequence == 2,
          "Z24 FIFO order");
    Check(output.nb_read(completed) && completed.sequence == 3,
          "driver depth-plane FIFO order");
    Check(output.nb_read(completed) && completed.sequence == 4,
          "quantized depth-compare FIFO order");
    CheckCase(pool, z16, z16_load);
    CheckCase(pool, z24, {0x800000U, 0x800000U, 0x800000U});
    CheckDriverDepthPlaneCase(pool, depth_plane);
    CheckQuantizedDepthCompareCase(pool, quantized_compare);
    Check(pool.bytes_in_flight() == 0 &&
              pool.allocations() == pool.releases(),
          "MemoryPool balance");
    std::cout << "isp_depth_attachment_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "isp_depth_attachment_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
