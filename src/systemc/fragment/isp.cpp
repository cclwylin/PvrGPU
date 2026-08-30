// ISP (Image Synthesis Processor) functional module.
// It consumes each 32x32 tile's ordered Parameter Buffer primitive references,
// evaluates exact fixed-point top-left coverage at the reference uArch's one
// sample center, and performs opaque-safe HSR (Hidden Surface Removal). Opaque
// draws retain only the final owner; blending preserves every depth-passing
// candidate in API order for later PBE destination read/modify/write. FIFO
// traffic remains MemoryPool handles and completion is event-driven.
#include "fragment/isp.h"

#include "common/functional_types.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using pvrgpu::stub::DepthCompareOp;
using pvrgpu::stub::DepthState;
using pvrgpu::stub::EdgeEquation;
using pvrgpu::stub::FragmentCandidate;
using pvrgpu::stub::FragmentVisibility;
using pvrgpu::stub::ParameterTriangle;

bool CoversSample(const ParameterTriangle &triangle, std::int64_t sample_x,
                  std::int64_t sample_y, std::int64_t values[3]) {
  for (std::size_t edge = 0; edge < 3; ++edge) {
    const EdgeEquation &equation = triangle.edge[edge];
    values[edge] = equation.a * sample_x + equation.b * sample_y + equation.c;
    if (values[edge] < 0 || (values[edge] == 0 && equation.inclusive == 0)) {
      return false;
    }
  }
  return true;
}

bool DepthPass(DepthCompareOp compare_op, float incoming, float stored) {
  switch (compare_op) {
  case DepthCompareOp::kNever:
    return false;
  case DepthCompareOp::kLess:
    return incoming < stored;
  case DepthCompareOp::kEqual:
    return incoming == stored;
  case DepthCompareOp::kLessOrEqual:
    return incoming <= stored;
  case DepthCompareOp::kGreater:
    return incoming > stored;
  case DepthCompareOp::kNotEqual:
    return incoming != stored;
  case DepthCompareOp::kGreaterOrEqual:
    return incoming >= stored;
  case DepthCompareOp::kAlways:
    return true;
  }
  throw std::runtime_error("ISP received an invalid depth compare operation");
}

float InterpolateDepth(const ParameterTriangle &triangle,
                       const std::int64_t edge_values[3],
                       float barycentric[3]) {
  if (triangle.signed_area <= 0)
    throw std::runtime_error("ISP received a non-positive triangle area");
  const double reciprocal_area =
      1.0 / static_cast<double>(triangle.signed_area);
  // Edge 1 is opposite vertex 0, edge 2 is opposite vertex 1, and edge 0 is
  // opposite vertex 2 for the serialized v0->v1->v2 equations.
  barycentric[0] = static_cast<float>(edge_values[1] * reciprocal_area);
  barycentric[1] = static_cast<float>(edge_values[2] * reciprocal_area);
  barycentric[2] = static_cast<float>(edge_values[0] * reciprocal_area);
  return barycentric[0] * triangle.window_z[0] +
         barycentric[1] * triangle.window_z[1] +
         barycentric[2] * triangle.window_z[2];
}

} // namespace

namespace pvrgpu::stub {

Isp::Isp(sc_core::sc_module_name name, MemoryPool &pool)
    : sc_module(name), pool_(pool) {
  SC_THREAD(Run);
}

void Isp::Run() {
  while (true) {
    const PipelineTxn txn = input.read();
    PipelineState state = LoadPipelineState(pool_, txn.state);
    RequireStage(state.stage, PipelineStage::kTilesScheduled, name());
    if (!IsRasterFunctionalCase(state.functional_case))
      throw std::runtime_error("ISP received an unsupported case");
    bool opaque_early_hsr = state.raster_state.blend.enable == 0;
    // If the fragment shader may discard, we cannot perform opaque early HSR because
    // a front-most fragment might be discarded later, revealing fragments behind it.
    // Likewise, if early HSR is not safe, we disable early culling.
    if (state.raster_state.shader_may_discard || !state.fragment_early_hsr_safe) {
      opaque_early_hsr = false;
    }
    // If the shader writes custom depth, early depth writes are not allowed because
    // the final depth value is determined during shader execution.
    const bool early_depth_write = state.raster_state.depth.write_enable &&
                                   !state.raster_state.shader_writes_depth;
    if (state.raster_state.sample_count != 1)
      throw std::runtime_error("reference ISP currently requires one sample");

    const std::uint64_t pixel_count =
        static_cast<std::uint64_t>(state.width) * state.height;
    if (pixel_count > std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("ISP surface is too large");
    const std::vector<TileRecord> tiles =
        LoadArray<TileRecord>(pool_, state.tile_records);
    const std::vector<TilePrimitiveRef> primitive_refs =
        LoadArray<TilePrimitiveRef>(pool_, state.tile_primitive_refs);
    const std::vector<ParameterTriangle> parameters =
        LoadArray<ParameterTriangle>(pool_, state.parameter_triangles);
    const bool empty_face_culled_setup =
        parameters.empty() && primitive_refs.empty() &&
        state.raster_state.face_cull.enable &&
        state.counters.c_primitives == 0;
    if (tiles.size() != state.scheduled_tiles ||
        (parameters.empty() && !empty_face_culled_setup))
      throw std::runtime_error("ISP received invalid tile/parameter data");

    std::vector<FragmentCandidate> candidates;
    candidates.reserve(static_cast<std::size_t>(pixel_count));
    constexpr std::size_t kNoOwner = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> owner(static_cast<std::size_t>(pixel_count),
                                   kNoOwner);
    std::vector<std::uint8_t> covered(static_cast<std::size_t>(pixel_count), 0);
    std::vector<float> depth(static_cast<std::size_t>(pixel_count),
                             state.raster_state.depth.clear_depth);
    std::uint64_t covered_pixels = 0;
    std::uint64_t depth_tested = 0;
    std::uint64_t depth_rejected = 0;
    std::uint64_t depth_written = 0;

    for (const TileRecord &tile : tiles) {
      const std::uint64_t ref_end =
          static_cast<std::uint64_t>(tile.first_primitive_ref) +
          tile.primitive_ref_count;
      if (ref_end > primitive_refs.size())
        throw std::runtime_error("ISP tile primitive range is out of bounds");
      std::uint64_t previous_ordinal = 0;
      bool first_ref = true;
      for (std::uint32_t ref_offset = 0; ref_offset < tile.primitive_ref_count;
           ++ref_offset) {
        const TilePrimitiveRef &ref =
            primitive_refs[tile.first_primitive_ref + ref_offset];
        if (ref.parameter_index >= parameters.size())
          throw std::runtime_error("ISP parameter reference is out of bounds");
        if (!first_ref && ref.submit_ordinal < previous_ordinal)
          throw std::runtime_error("ISP primitive references lost API order");
        first_ref = false;
        previous_ordinal = ref.submit_ordinal;
        const ParameterTriangle &triangle = parameters[ref.parameter_index];
        if (triangle.key.submit_ordinal != ref.submit_ordinal)
          throw std::runtime_error("ISP primitive identity mismatch");

        const std::uint32_t y_begin = std::max<std::uint32_t>(
            tile.y0, static_cast<std::uint32_t>(std::max(0, triangle.min_y)));
        const std::uint32_t y_end = std::min<std::uint32_t>(
            tile.y1, static_cast<std::uint32_t>(std::max(0, triangle.max_y)));
        const std::uint32_t x_begin = std::max<std::uint32_t>(
            tile.x0, static_cast<std::uint32_t>(std::max(0, triangle.min_x)));
        const std::uint32_t x_end = std::min<std::uint32_t>(
            tile.x1, static_cast<std::uint32_t>(std::max(0, triangle.max_x)));
        for (std::uint32_t y = y_begin; y < y_end; ++y) {
          for (std::uint32_t x = x_begin; x < x_end; ++x) {
            const std::int64_t sample_x =
                static_cast<std::int64_t>(x) * kSubpixelScale +
                kSubpixelScale / 2;
            const std::int64_t sample_y =
                static_cast<std::int64_t>(y) * kSubpixelScale +
                kSubpixelScale / 2;
            std::int64_t edge_values[3]{};
            if (!CoversSample(triangle, sample_x, sample_y, edge_values))
              continue;

            FragmentCandidate candidate;
            candidate.x = x;
            candidate.y = y;
            candidate.primitive_id = triangle.key.api_primitive_id;
            candidate.parameter_index = ref.parameter_index;
            candidate.submit_ordinal = ref.submit_ordinal;
            candidate.sample_mask = 1;
            candidate.depth =
                InterpolateDepth(triangle, edge_values, candidate.barycentric);
            const std::size_t pixel_index =
                static_cast<std::size_t>(y) * state.width + x;
            if (covered[pixel_index] == 0) {
              covered[pixel_index] = 1;
              ++covered_pixels;
            }
            bool passes = true;
            if (state.raster_state.depth.test_enable) {
              ++depth_tested;
              passes = DepthPass(state.raster_state.depth.compare_op,
                                 candidate.depth, depth[pixel_index]);
              if (!passes)
                ++depth_rejected;
            }
            const std::size_t candidate_index = candidates.size();
            candidates.push_back(candidate);
            if (!passes)
              continue;

            candidates[candidate_index].visibility =
                FragmentVisibility::kVisible;
            if (opaque_early_hsr) {
              if (owner[pixel_index] != kNoOwner)
                candidates[owner[pixel_index]].visibility =
                    FragmentVisibility::kRejected;
              owner[pixel_index] = candidate_index;
            }
            if (state.raster_state.depth.test_enable && early_depth_write) {
              depth[pixel_index] = candidate.depth;
              ++depth_written;
            }
          }
        }
      }
    }

    const std::uint64_t visible = static_cast<std::uint64_t>(std::count_if(
        candidates.begin(), candidates.end(), [](const FragmentCandidate &c) {
          return c.visibility == FragmentVisibility::kVisible;
        }));
    if (visible > candidates.size())
      throw std::runtime_error("ISP visible count exceeds candidate count");
    if (visible > std::numeric_limits<std::uint32_t>::max())
      throw std::overflow_error("ISP visible pixel count exceeds uint32_t");

    state.fragment_candidates = StoreNewArray(pool_, candidates);
    state.active_fragment_invocations = static_cast<std::uint32_t>(visible);
    state.counters.fragment_candidates = candidates.size();
    state.counters.hsr_rejected_fragments = candidates.size() - visible;
    state.counters.covered_pixels = covered_pixels;
    state.counters.depth_tested_fragments = depth_tested;
    state.counters.depth_rejected_fragments = depth_rejected;
    state.counters.depth_written_fragments = depth_written;
    const std::uint64_t cycles =
        kReferenceUarch.isp_base_cycles +
        CeilDivide(candidates.size(), kReferenceUarch.isp_candidates_per_batch);
    state.counters.isp_cycles = cycles;
    state.counters.renderer_cycles += cycles;
    state.stage = PipelineStage::kVisibilityReady;

    WaitForCycles(cycles);
    StorePipelineState(pool_, txn.state, state);
    output.write(txn);
  }
}

} // namespace pvrgpu::stub
