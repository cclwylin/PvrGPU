// Focused API-v7 depth-attachment lifecycle regression. ISP must preserve a
// complete LOAD surface (including untouched pixels) and quantize NEW_CLEAR
// through the declared native format before later draws alias it. Float depth
// keeps its full binary32 precision and combined formats preserve stencil.

#include "common/functional_types.h"
#include "common/msaa.h"
#include "common/pipeline_state.h"
#include "fragment/isp.h"

#include <systemc>

#include <array>
#include <cmath>
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

float BitsFloat(std::uint32_t bits) {
  float value = 0.0F;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
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
                                          std::uint64_t sequence,
                                          bool float_depth = false) {
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
  state.depth_attachment_format = float_depth ? kDriverPcoDepthFormatZ32Float
                                              : kDriverPcoDepthFormatZ16Unorm;
  state.capture_depth_attachment = 1;
  state.scheduled_tiles = 1;

  const float first_depth = float_depth ? 0x1p-100F : 0.9999373555F;
  const float later_depth = float_depth ? 0x1p-99F : 0.9999454021F;
  Check(first_depth < later_depth,
        "quantized compare fixture preserves raw depth order");
  if (!float_depth) {
    Check(EncodeDepthAttachmentUnorm(first_depth,
                                     kDriverPcoDepthFormatZ16Unorm) ==
              EncodeDepthAttachmentUnorm(later_depth,
                                         kDriverPcoDepthFormatZ16Unorm),
          "quantized compare fixture aliases one Z16 value");
  }

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

CasePayload MakeMsaaVisibilityCase(MemoryPool &pool, std::uint64_t sequence,
                                  std::uint32_t samples,
                                  std::uint32_t sample_mask,
                                  bool multisample_enable) {
  PipelineState state;
  state.width = state.height = 1;
  state.sequence = sequence;
  state.functional_case = FunctionalCase::kDriverPcoTriangles;
  state.stage = PipelineStage::kTilesScheduled;
  state.raster_state.sample_count = samples;
  state.raster_state.sample_mask = sample_mask;
  state.raster_state.multisample_enable = multisample_enable ? 1 : 0;
  state.raster_state.depth.test_enable = 1;
  state.raster_state.depth.write_enable = 1;
  state.raster_state.depth.compare_op = DepthCompareOp::kLess;
  state.raster_state.depth.clear_depth = 1.0F;
  state.raster_state.stencil.test_enable = 1;
  state.raster_state.stencil.clear_stencil = 5;
  state.raster_state.stencil.front.pass_op = StencilOp::kIncrementClamp;
  state.raster_state.stencil.back.pass_op = StencilOp::kIncrementClamp;
  state.fragment_early_hsr_safe = 1;
  state.depth_attachment_format = kDriverPcoDepthFormatZ24UnormS8Uint;
  state.capture_depth_attachment = 1;
  state.scheduled_tiles = 1;

  auto make_triangle = [](std::uint64_t ordinal,
                          const std::array<std::array<float, 2>, 3> &positions,
                          float center_depth, float dzdx) {
    ParameterTriangle triangle;
    triangle.key.submit_ordinal = ordinal;
    triangle.key.api_primitive_id = static_cast<std::uint32_t>(ordinal);
    triangle.rasterizable = 1;
    triangle.max_x = triangle.max_y = 1;
    std::array<std::array<std::int64_t, 2>, 3> fixed{};
    for (std::size_t vertex = 0; vertex < 3; ++vertex)
      for (std::size_t axis = 0; axis < 2; ++axis)
        fixed[vertex][axis] = static_cast<std::int64_t>(
            positions[vertex][axis] * kSubpixelScale);
    for (std::size_t edge = 0; edge < 3; ++edge) {
      const auto &first = fixed[edge];
      const auto &last = fixed[(edge + 1) % 3];
      const std::int64_t dx = last[0] - first[0];
      const std::int64_t dy = last[1] - first[1];
      triangle.edge[edge] = {-dy, dx, dy * first[0] - dx * first[1],
                             static_cast<std::uint8_t>(
                                 dy < 0 || (dy == 0 && dx > 0)), {}};
    }
    triangle.signed_area =
        (fixed[1][0] - fixed[0][0]) * (fixed[2][1] - fixed[0][1]) -
        (fixed[1][1] - fixed[0][1]) * (fixed[2][0] - fixed[0][0]);
    Check(triangle.signed_area > 0, "MSAA fixture triangle winding");
    triangle.depth_plane[0] = FloatBits(dzdx);
    triangle.depth_plane[1] = FloatBits(0.0F);
    triangle.depth_plane[2] = FloatBits(center_depth);
    triangle.depth_plane_valid = 1;
    return triangle;
  };
  // Background covers the whole pixel. The later triangle covers only x<1/2;
  // its depth varies by sample position. Both must retain disjoint ownership.
  const std::vector<ParameterTriangle> triangles = {
      make_triangle(10, {{{-2, -2}, {4, -2}, {-2, 4}}}, 0.8F, 0.0F),
      make_triangle(11, {{{0.5F, -2}, {0.5F, 3}, {-4, 0.5F}}}, 0.2F, 0.4F),
  };
  state.tile_records =
      StoreNewArray(pool, std::vector<TileRecord>{{0, 0, 1, 1, 0, 2}});
  state.tile_primitive_refs = StoreNewArray(
      pool, std::vector<TilePrimitiveRef>{{0, 0, 10}, {1, 0, 11}});
  state.parameter_triangles = StoreNewArray(pool, triangles);
  CasePayload payload;
  payload.state = pool.Allocate(sizeof(PipelineState));
  StorePipelineState(pool, payload.state, state);
  payload.txn = {payload.state, static_cast<std::uint32_t>(sequence), sequence};
  return payload;
}

void CheckMsaaVisibilityCase(MemoryPool &pool, const CasePayload &payload,
                             std::uint32_t samples, std::uint32_t sample_mask,
                             bool multisample_enable) {
  const PipelineState state = LoadPipelineState(pool, payload.state);
  const auto depth = LoadArray<std::uint32_t>(pool, state.isp_depth_attachment);
  const auto stencil = LoadArray<std::uint8_t>(pool, state.isp_stencil_attachment);
  Check(depth.size() == samples && stencil.size() == samples,
        "MSAA depth/stencil storage lost sample planes");
  std::uint32_t foreground_mask = 0;
  const std::uint32_t enabled = RasterSampleMask(samples) & sample_mask;
  for (std::uint32_t sample = 0; sample < samples; ++sample) {
    const float sx = multisample_enable
        ? RasterSamplePosition(samples, sample)[0] / 16.0F : 0.5F;
    const bool active = (enabled & (1U << sample)) != 0;
    const bool foreground = active && sx < 0.5F;
    if (foreground)
      foreground_mask |= 1U << sample;
    const float expected_depth = !active ? 1.0F : foreground
        ? std::fma(0.4F, sx - 0.5F, 0.2F) : 0.8F;
    Check(depth[sample] == EncodeDepthAttachmentUnorm(
                              expected_depth, state.depth_attachment_format),
          "MSAA depth did not evaluate/test/write the actual sample");
    Check(stencil[sample] == (!active ? 5 : foreground ? 7 : 6),
          "MSAA stencil update did not preserve sample ownership");
  }
  const auto candidates = LoadArray<FragmentCandidate>(pool, state.fragment_candidates);
  std::uint32_t owned = 0;
  for (const auto &candidate : candidates) {
    if (candidate.visibility != FragmentVisibility::kVisible)
      continue;
    Check((owned & candidate.sample_mask) == 0,
          "MSAA opaque primitives overlap in sample ownership");
    const std::uint32_t expected = candidate.primitive_id == 11
        ? foreground_mask : enabled & ~foreground_mask;
    Check(candidate.sample_mask == expected,
          "MSAA HSR removed untouched samples of another primitive");
    owned |= candidate.sample_mask;
  }
  Check(owned == enabled, "MSAA HSR lost visible samples");
  ReleaseFunctionalPayloads(pool, state);
  pool.Release(payload.state);
}

void CheckFloatDepthCodecs() {
  Check(kDriverPcoDepthFormatZ32Float == 271 &&
            kDriverPcoDepthFormatZ32FloatS8X24Uint == 279,
        "floating depth formats preserve the Mesa pipe_format ABI");
  const std::vector<std::uint32_t> exact = {
      0U, 1U, FloatBits(0x1p-100F), FloatBits(0.5F), FloatBits(1.0F)};
  for (const auto format : {kDriverPcoDepthFormatZ32Float,
                            kDriverPcoDepthFormatZ32FloatS8X24Uint}) {
    for (const std::uint32_t bits : exact) {
      Check(EncodeDepthAttachmentUnorm(DecodeDepthAttachmentUnorm(bits, format),
                                       format) == bits,
            "float depth codec preserves normals and the smallest subnormal");
    }
    Check(EncodeDepthAttachmentUnorm(-0.0F, format) == 0,
          "signed zero uses one depth comparison key");
    Check(DecodeDepthAttachmentUnormBytes(
              EncodeDepthAttachmentUnormBytes(exact, format), format) == exact,
          "float depth native bytes preserve exact payloads");
    bool rejected = false;
    try {
      (void)DecodeDepthAttachmentUnorm(UINT32_C(0x7fc00000), format);
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    Check(rejected, "floating depth rejects non-finite stored values");
  }
  Check(DepthAttachmentBytesPerPixel(kDriverPcoDepthFormatZ32Float) == 4 &&
            DepthAttachmentBytesPerPixel(kDriverPcoDepthFormatZ32FloatS8X24Uint) == 8,
        "float-only and float-stencil use their native byte widths");
  const std::vector<std::uint8_t> stencil = {0, 1, 0x7f, 0x80, 0xff};
  auto packed = EncodeDepthAttachmentUnormBytes(
      exact, kDriverPcoDepthFormatZ32FloatS8X24Uint, &stencil);
  for (std::size_t pixel = 0; pixel < exact.size(); ++pixel) {
    Check(packed[pixel * 8 + 4] == stencil[pixel] &&
              packed[pixel * 8 + 5] == 0 && packed[pixel * 8 + 6] == 0 &&
              packed[pixel * 8 + 7] == 0,
          "float-stencil writes stencil after depth and initializes padding");
    packed[pixel * 8 + 5] = 0xa5;
    packed[pixel * 8 + 6] = 0x5a;
    packed[pixel * 8 + 7] = 0xff;
  }
  std::vector<std::uint8_t> restored_stencil;
  Check(DecodeDepthAttachmentUnormBytes(
            packed, kDriverPcoDepthFormatZ32FloatS8X24Uint, &restored_stencil) == exact &&
            restored_stencil == stencil,
        "float-stencil reads exact planes and ignores undefined X24 padding");
}

void CheckD24DepthCodecRounding() {
  // These values make the binary32 product land exactly on adjacent .5
  // boundaries. The first one distinguishes llvmpipe's rounded float
  // multiply from a higher-precision multiply followed by integer rounding;
  // the pair also verifies round-to-nearest-even in both directions.
  const float even_tie = BitsFloat(UINT32_C(0x3e800002));
  const float odd_tie = BitsFloat(UINT32_C(0x3e800004));
  for (const auto format : {kDriverPcoDepthFormatZ24X8Unorm,
                            kDriverPcoDepthFormatZ24UnormS8Uint}) {
    Check(EncodeDepthAttachmentUnorm(even_tie, format) == UINT32_C(0x400000),
          "D24 rounds an even binary32 half tie downward");
    Check(EncodeDepthAttachmentUnorm(odd_tie, format) == UINT32_C(0x400002),
          "D24 rounds an odd binary32 half tie upward");
  }
  Check(EncodeDepthAttachmentUnorm(even_tie,
                                   kDriverPcoDepthFormatZ16Unorm) ==
                UINT32_C(0x4000) &&
            EncodeDepthAttachmentUnorm(even_tie,
                                       kDriverPcoDepthFormatZ32Unorm) ==
                UINT32_C(0x40000100),
        "D24 binary32 scaling does not change Z16 or Z32 conversion");
}

void CheckFloatDepthCompareCase(MemoryPool &pool, const CasePayload &payload) {
  const PipelineState state = LoadPipelineState(pool, payload.state);
  Check(LoadArray<std::uint32_t>(pool, state.isp_depth_attachment) ==
            std::vector<std::uint32_t>{FloatBits(0x1p-100F)},
        "ISP retains float depth far below one UNORM32 unit");
  const auto candidates = LoadArray<FragmentCandidate>(pool, state.fragment_candidates);
  Check(candidates.size() == 2 &&
            candidates[0].visibility == FragmentVisibility::kVisible &&
            candidates[1].visibility == FragmentVisibility::kRejected,
        "float depth compares distinct small values rather than quantizing both to zero");
  ReleaseFunctionalPayloads(pool, state);
  pool.Release(payload.state);
}

} // namespace

int sc_main(int, char **) {
  try {
    CheckFloatDepthCodecs();
    CheckD24DepthCodecRounding();
    MemoryPool pool;
    sc_core::sc_fifo<PipelineTxn> input("input", 8);
    sc_core::sc_fifo<PipelineTxn> output("output", 8);
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
    const std::vector<std::uint32_t> float_load = {
        FloatBits(0.0F), FloatBits(0x1p-100F), FloatBits(1.0F)};
    const CasePayload z32f = MakeCase(
        pool, 5, kDriverPcoDepthFormatZ32Float, 1.0F, float_load);
    const CasePayload z32fs8 = MakeCase(
        pool, 6, kDriverPcoDepthFormatZ32FloatS8X24Uint, 0x1p-100F, {});
    const CasePayload float_compare = MakeQuantizedDepthCompareCase(pool, 7, true);
    input.write(z16.txn);
    input.write(z24.txn);
    input.write(depth_plane.txn);
    input.write(quantized_compare.txn);
    input.write(z32f.txn);
    input.write(z32fs8.txn);
    input.write(float_compare.txn);
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
    for (std::uint64_t sequence = 5; sequence <= 7; ++sequence)
      Check(output.nb_read(completed) && completed.sequence == sequence,
            "float depth FIFO order");
    CheckCase(pool, z16, z16_load);
    CheckCase(pool, z24, {0x800000U, 0x800000U, 0x800000U});
    CheckDriverDepthPlaneCase(pool, depth_plane);
    CheckQuantizedDepthCompareCase(pool, quantized_compare);
    CheckCase(pool, z32f, float_load);
    CheckCase(pool, z32fs8, {FloatBits(0x1p-100F), FloatBits(0x1p-100F), FloatBits(0x1p-100F)});
    CheckFloatDepthCompareCase(pool, float_compare);
    for (std::uint32_t samples : {1U, 2U, 4U, 8U, 16U}) {
      for (std::uint32_t mask : {UINT32_MAX, UINT32_C(0x5555)}) {
        for (const bool multisample_enable : {true, false}) {
          const std::uint64_t sequence = samples * 4 +
              (mask == UINT32_MAX ? 20 : 22) + multisample_enable;
          const CasePayload msaa = MakeMsaaVisibilityCase(
              pool, sequence, samples, mask, multisample_enable);
          input.write(msaa.txn);
          sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_NS));
          sc_core::sc_start(sc_core::SC_ZERO_TIME);
          Check(output.nb_read(completed) && completed.sequence == sequence,
                "MSAA visibility completion");
          CheckMsaaVisibilityCase(pool, msaa, samples, mask, multisample_enable);
        }
      }
    }
    // Center coverage does not collapse the depth/stencil storage: the same
    // background fragment must fail against even samples and pass against
    // odd samples. The foreground edge excludes the center entirely.
    for (std::uint32_t samples : {2U, 4U, 8U, 16U}) {
      const std::uint64_t sequence = 100 + samples;
      const CasePayload center = MakeMsaaVisibilityCase(
          pool, sequence, samples, UINT32_MAX, false);
      PipelineState state = LoadPipelineState(pool, center.state);
      std::vector<std::uint32_t> initial_depth(samples);
      for (std::uint32_t sample = 0; sample < samples; ++sample)
        initial_depth[sample] = EncodeDepthAttachmentUnorm(
            sample % 2 ? 1.0F : 0.1F, state.depth_attachment_format);
      const std::vector<std::uint8_t> initial_stencil(samples, 5);
      const auto bytes = EncodeDepthAttachmentUnormBytes(
          initial_depth, state.depth_attachment_format, &initial_stencil);
      state.depth_attachment_load = StoreNewArray(pool, bytes);
      state.depth_attachment_load_enable = 1;
      state.depth_attachment_load_bytes = bytes.size();
      StorePipelineState(pool, center.state, state);
      input.write(center.txn);
      sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_NS));
      Check(output.nb_read(completed) && completed.sequence == sequence,
            "center coverage independent depth completion");
      state = LoadPipelineState(pool, center.state);
      const auto depth = LoadArray<std::uint32_t>(pool, state.isp_depth_attachment);
      const auto stencil = LoadArray<std::uint8_t>(pool, state.isp_stencil_attachment);
      for (std::uint32_t sample = 0; sample < samples; ++sample) {
        Check(depth[sample] == EncodeDepthAttachmentUnorm(
                  sample % 2 ? 0.8F : 0.1F, state.depth_attachment_format),
              "center coverage must test each stored depth independently");
        Check(stencil[sample] == (sample % 2 ? 6 : 5),
              "center coverage must update only depth-passing sample stencils");
      }
      const auto candidates = LoadArray<FragmentCandidate>(
          pool, state.fragment_candidates);
      Check(candidates.size() == 1 && candidates[0].primitive_id == 10 &&
                candidates[0].sample_mask == (RasterSampleMask(samples) & 0xaaaaU),
            "center coverage HSR must preserve independent sample depth results");
      ReleaseFunctionalPayloads(pool, state);
      pool.Release(center.state);
    }
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
