// SPDX-License-Identifier: MIT
// Fixed-function native-driver boundary regression. The explicit VTXOUT
// records are unit inputs, not claimed VS/TES shader executions. Real modules
// classify them, build equations, bin coverage and update the S8 attachment.
// No dEQP case names or expected shader outputs enter the product pipeline.
#include "common/functional_types.h"
#include "common/geometry_emission.h"
#include "common/pipeline_state.h"
#include "common/tessellation_state.h"
#include "fragment/isp.h"
#include "fragment/tile_scheduler.h"
#include "geometry/clip_cull.h"
#include "geometry/parameter_buffer.h"
#include "geometry/tiler.h"
#include "shader/pco_decoder.h"
#include "pco_tessellation_compiler_fixtures.h"

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
std::uint64_t checks = 0;
std::string current_case;
constexpr std::uint32_t kExtent = 16;
constexpr std::uint8_t kClear = 0x19, kFront = 0xa5, kBack = 0x5a;

void Check(bool condition, const char *message) {
  ++checks;
  if (!condition)
    throw std::runtime_error(current_case + ": " + message);
}

std::uint32_t Bits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

PipelineState MakeState(MemoryPool &pool, std::uint64_t sequence,
                        bool tessellation, FrontFaceWinding winding,
                        bool ndc_ccw, bool negative_viewport_y,
                        unsigned cull, float depth_near = 0.0F,
                        float depth_far = 1.0F, float clip_depth = 0.0F) {
  PipelineState state;
  state.width = state.height = kExtent;
  state.sequence = sequence;
  state.stage = PipelineStage::kVertexShaded;
  state.functional_case = FunctionalCase::kDriverPcoTriangles;
  state.draw.topology = state.source_topology = PrimitiveTopology::kTriangleList;
  state.draw.vertex_count = 3;
  state.vertex_pco_abi.vertex_outputs = 4;
  state.position_output_count = 4;
  state.fragment_position_count = 4;
  state.fragment_varying_start = 4;
  state.fragment_pco_abi.temps = 1;
  state.fragment_pco_abi.coefficients = 4;
  state.fragment_output_mask[0] = 0xf;
  state.fragment_code = StoreNewArray(pool, Tess0FragmentPco());
  state.drawlist_stats = StoreNewArray(pool, std::vector<DrawListStats>{{}});
  auto &raster = state.raster_state;
  raster.face_cull.front_face = winding;
  raster.face_cull.enable = cull != 0;
  raster.face_cull.mode = cull == 1 ? CullFaceMode::kFront :
      cull == 2 ? CullFaceMode::kBack : CullFaceMode::kFrontAndBack;
  raster.viewport_scale[0] = kExtent * 0.5f;
  raster.viewport_scale[1] = (negative_viewport_y ? -0.5f : 0.5f) * kExtent;
  raster.viewport_scale[2] = (depth_far - depth_near) * 0.5F;
  raster.viewport_translate[0] = raster.viewport_translate[1] = kExtent * 0.5f;
  raster.viewport_translate[2] = (depth_far + depth_near) * 0.5F;
  raster.depth.test_enable = raster.depth.write_enable = 0;
  raster.stencil.test_enable = 1;
  raster.stencil.clear_stencil = kClear;
  raster.stencil.front.reference = kFront;
  raster.stencil.back.reference = kBack;
  raster.stencil.front.pass_op = raster.stencil.back.pass_op = StencilOp::kReplace;
  state.depth_attachment_format = kDriverPcoDepthFormatZ24UnormS8Uint;
  state.capture_depth_attachment = 1;

  // Swapping source occurrences reverses NDC area without changing coverage.
  const std::array<std::array<float, 2>, 3> xy{{
      {{-0.75f, -0.75f}}, {{0.75f, -0.75f}}, {{0.0f, 0.75f}}}};
  std::vector<VertexLane> lanes(3);
  for (unsigned i = 0; i < 3; ++i) {
    lanes[i].vertex_output[0] = Bits(xy[i][0]);
    lanes[i].vertex_output[1] = Bits(xy[i][1]);
    lanes[i].vertex_output[2] = Bits(clip_depth);
    lanes[i].vertex_output[3] = Bits(1.0f);
    lanes[i].emitted = lanes[i].ended = 1;
  }
  const std::array<std::uint32_t, 3> order = ndc_ccw ?
      std::array<std::uint32_t, 3>{0, 1, 2} :
      std::array<std::uint32_t, 3>{0, 2, 1};
  std::vector<VertexLaneRef> refs;
  for (unsigned i = 0; i < order.size(); ++i)
    refs.push_back({order[i], tessellation ? order[i] : i});
  state.vertex_lanes = StoreNewArray(pool, lanes);
  state.vertex_lane_refs = StoreNewArray(pool, refs);
  if (tessellation) {
    // Exercise the TES-output metadata boundary independently of native
    // shader execution, which tessellation-shader-test validates separately.
    TessellationState tess;
    tess.phase = TessellationPhase::kEvaluationComplete;
    tess.evaluation_abi.vertex_outputs = 4;
    state.tessellation_state = StoreNewArray(pool, std::vector<TessellationState>{tess});
    state.tessellation_output_dwords = 4;
    GeometryRasterPrimitive primitive;
    primitive.refs = {{order[0], order[1], order[2]}, 3, 2, {}};
    state.geometry_primitives = StoreNewArray(pool,
        std::vector<GeometryRasterPrimitive>{primitive});
  }
  return state;
}

void Verify(MemoryPool &pool, const PipelineState &state, bool front,
            bool culled) {
  Check(state.stage == PipelineStage::kVisibilityReady, "ISP completion stage");
  const auto raster = LoadArray<RasterTriangle>(pool, state.raster_triangles);
  const auto parameters = LoadArray<ParameterTriangle>(pool, state.parameter_triangles);
  Check(raster.size() == 1 && parameters.size() == 1, "one interior triangle");
  Check(raster[0].front_facing == front, "ClipCull original front_facing");
  Check(raster[0].face_culled == culled, "ClipCull face_culled");
  Check(raster[0].rasterizable == !culled, "ClipCull rasterizable");
  Check(parameters[0].front_facing == front, "ParameterBuffer preserves facing");
  Check(parameters[0].face_culled == culled, "ParameterBuffer preserves cull flag");
  Check(parameters[0].rasterizable == !culled, "ParameterBuffer rasterizable");
  Check(state.counters.c_invocations == 1 && state.counters.c_primitives == 1,
        "true fixed-function invocation/setup accounting");
  const auto refs = LoadArray<TilePrimitiveRef>(pool, state.tile_primitive_refs);
  Check(refs.empty() == culled, "culled triangle never enters a tile bin");
  const auto candidates = LoadArray<FragmentCandidate>(pool, state.fragment_candidates);
  Check(candidates.empty() == culled, "real coverage is present only without culling");
  const auto stencil = LoadArray<std::uint8_t>(pool, state.isp_stencil_attachment);
  Check(stencil.size() == kExtent * kExtent, "complete S8 surface is captured");
  const std::uint8_t expected = front ? kFront : kBack;
  std::vector<bool> covered(kExtent * kExtent, false);
  for (const auto &candidate : candidates) {
    Check(candidate.x < kExtent && candidate.y < kExtent,
          "candidate belongs to the attachment");
    Check(candidate.visibility == FragmentVisibility::kVisible &&
              candidate.sample_mask == 1, "stencil Always preserves real coverage");
    covered[candidate.y * kExtent + candidate.x] = true;
  }
  for (std::size_t pixel = 0; pixel < stencil.size(); ++pixel)
    Check(stencil[pixel] == (covered[pixel] ? expected : kClear),
          "ISP applies the correct face's Replace reference only to coverage");
  Check(stencil[(kExtent / 2) * kExtent + kExtent / 2] ==
            (culled ? kClear : expected),
        "interior pixel independently proves front/back stencil choice");
  Check(state.counters.stencil_tested_fragments == candidates.size() &&
            state.counters.stencil_written_fragments == candidates.size() &&
            state.counters.stencil_rejected_fragments == 0,
        "stencil counters match actual attachment work");
}
}  // namespace

int sc_main(int, char **) {
  try {
    MemoryPool pool;
    sc_core::sc_fifo<PipelineTxn> input("input", 1), clipped("clipped", 1),
        tiled("tiled", 1), parameterized("parameterized", 1),
        decoded("decoded", 1), scheduled("scheduled", 1), output("output", 1);
    ClipCull clip("clip", pool);
    Tiler tiler("tiler", pool);
    ParameterBuffer parameter("parameter", pool);
    PcoDecoder decoder("decoder", pool, ShaderStage::kFragment);
    TileScheduler scheduler("scheduler", pool);
    Isp isp("isp", pool);
    clip.input(input); clip.output(clipped);
    tiler.input(clipped); tiler.output(tiled);
    parameter.input(tiled); parameter.output(parameterized);
    decoder.input(parameterized); decoder.output(decoded);
    scheduler.input(decoded); scheduler.output(scheduled);
    isp.input(scheduled); isp.output(output);

    std::uint64_t sequence = 0;
    for (bool tessellation : {false, true})
      for (auto winding : {FrontFaceWinding::kClockwise,
                           FrontFaceWinding::kCounterClockwise})
        for (bool ndc_ccw : {false, true})
          for (bool negative_y : {false, true})
            for (unsigned cull = 0; cull < 4; ++cull) {
              current_case = std::string(tessellation ? "TES" : "VS") +
                  " front=" + std::to_string(static_cast<unsigned>(winding)) +
                  " ndc_ccw=" + std::to_string(ndc_ccw) +
                  " negative_viewport_y=" + std::to_string(negative_y) +
                  " cull=" + std::to_string(cull);
              auto state = MakeState(pool, ++sequence, tessellation, winding,
                  ndc_ccw, negative_y, cull);
              const auto handle = pool.Allocate(sizeof(PipelineState));
              StorePipelineState(pool, handle, state);
              input.write({handle, static_cast<std::uint32_t>(sequence), sequence});
              sc_core::sc_start(sc_core::sc_time(10, sc_core::SC_US));
              PipelineTxn completion;
              Check(output.nb_read(completion), "bounded FIFO completion");
              Check(completion.state.slot == handle.slot &&
                        completion.state.generation == handle.generation &&
                        completion.sequence == sequence, "transaction identity");
              state = LoadPipelineState(pool, handle);
              // The native driver ABI states winding after Gallium's
              // window convention; ClipCull classifies NDC before viewport
              // conversion. Both enums must therefore be reflected once.
              const bool front = (winding == FrontFaceWinding::kClockwise) == ndc_ccw;
              const bool culled = cull == 3 || (cull == 1 && front) ||
                  (cull == 2 && !front);
              Verify(pool, state, front, culled);
              ReleaseFunctionalPayloads(pool, state);
              pool.Release(handle);
            }
    Check(sequence == 64, "complete independent winding/cull/viewport/stage matrix");
    const std::array<std::array<float, 2>, 7> depth_ranges{{
        {{0, 1}}, {{1, 0}}, {{.25F, .75F}}, {{.75F, .25F}},
        {{0, 0}}, {{.375F, .375F}}, {{1, 1}},
    }};
    for (const auto &range : depth_ranges) {
      for (float clip_z : {-1.0F, -.5F, 0.0F, .5F, 1.0F}) {
        current_case = "depth range=" + std::to_string(range[0]) + "," +
            std::to_string(range[1]) + " clip_z=" + std::to_string(clip_z);
        auto state = MakeState(pool, ++sequence, false,
            FrontFaceWinding::kClockwise, true, false, 0, range[0], range[1], clip_z);
        state.raster_state.depth.test_enable = state.raster_state.depth.write_enable = 1;
        state.raster_state.depth.compare_op = DepthCompareOp::kAlways;
        const auto handle = pool.Allocate(sizeof(PipelineState));
        StorePipelineState(pool, handle, state);
        input.write({handle, static_cast<std::uint32_t>(sequence), sequence});
        sc_core::sc_start(sc_core::sc_time(10, sc_core::SC_US));
        PipelineTxn completion;
        Check(output.nb_read(completion), "depth viewport bounded FIFO completion");
        state = LoadPipelineState(pool, handle);
        Verify(pool, state, true, false);
        const float expected = clip_z * ((range[1] - range[0]) * .5F) +
            (range[1] + range[0]) * .5F;
        const auto raster = LoadArray<RasterTriangle>(pool, state.raster_triangles);
        const auto parameters = LoadArray<ParameterTriangle>(pool, state.parameter_triangles);
        for (unsigned vertex = 0; vertex < 3; ++vertex) {
          Check(raster[0].window_z[vertex] == expected, "ClipCull uses actual depth scale/translate");
          Check(parameters[0].window_z[vertex] == expected, "ParameterBuffer preserves window depth");
        }
        const auto candidates = LoadArray<FragmentCandidate>(pool, state.fragment_candidates);
        const auto depth = LoadArray<std::uint32_t>(pool, state.isp_depth_attachment);
        Check(!candidates.empty() && depth.size() == kExtent * kExtent,
              "depth range produces real covered attachment writes");
        for (const auto &candidate : candidates) {
          Check(candidate.depth == expected, "ISP interpolates transformed window depth");
          Check(depth[candidate.y * kExtent + candidate.x] ==
                    EncodeDepthAttachmentUnorm(expected, state.depth_attachment_format),
                "ISP stores the requested forward/reverse/constant depth");
        }
        ReleaseFunctionalPayloads(pool, state);
        pool.Release(handle);
      }
    }
    Check(sequence == 99, "complete depth range transform matrix");
    Check(pool.allocations() == pool.releases(), "all payload ownership retired");
    std::cout << "driver-front-face-test: " << checks << " checks PASS / "
              << sequence << " matrix cases\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "driver-front-face-test: " << error.what() << '\n';
    return 1;
  }
}
