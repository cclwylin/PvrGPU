// Shared late depth/stencil state transition.  PBE uses it for the real
// attachment and coherent framebuffer fetch uses it on an independent shadow
// copy, so only samples that PBE would commit become visible to later shaders.
#pragma once

#include "common/functional_types.h"
#include "common/pipeline_state.h"
#include "fragment/pbe_color_commit.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace pvrgpu::stub {

template <typename T>
inline bool PbeLateDepthPass(DepthCompareOp operation, T incoming, T stored) {
  switch (operation) {
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
  throw std::runtime_error("PBE late depth comparison is invalid");
}

inline bool CommitPbeLateDepthStencil(
    const PipelineState &state, const FragmentInvocation &invocation,
    float shader_depth, std::size_t sample_index,
    std::vector<std::uint32_t> &depth, std::vector<std::uint8_t> &stencil,
    CounterTxn *counters = nullptr) {
  if (sample_index >= depth.size())
    throw std::runtime_error("PBE late depth sample index is invalid");
  const StencilState &stencil_state = state.raster_state.stencil;
  const StencilFaceState &face = invocation.front_facing
                                     ? stencil_state.front
                                     : stencil_state.back;
  bool stencil_passes = true;
  if (stencil_state.test_enable && !stencil.empty()) {
    if (sample_index >= stencil.size())
      throw std::runtime_error("PBE late stencil sample index is invalid");
    if (counters)
      ++counters->stencil_tested_fragments;
    const auto mask = static_cast<std::uint8_t>(face.value_mask);
    stencil_passes = StencilPass(
        face.compare_op, static_cast<std::uint8_t>(face.reference & mask),
        static_cast<std::uint8_t>(stencil[sample_index] & mask));
    if (!stencil_passes && counters)
      ++counters->stencil_rejected_fragments;
  }

  bool passes = stencil_passes;
  std::uint32_t encoded = 0;
  if (stencil_passes && state.raster_state.depth.test_enable) {
    if (counters)
      ++counters->depth_tested_fragments;
    if (std::isnan(shader_depth))
      throw std::runtime_error("PBE shader depth is NaN");
    shader_depth = std::clamp(shader_depth, 0.0F, 1.0F);
    if (state.depth_attachment_format == 0) {
      std::memcpy(&encoded, &shader_depth, sizeof(encoded));
      passes = PbeLateDepthPass(state.raster_state.depth.compare_op,
                                shader_depth,
                                PbeFloatFromBits(depth[sample_index]));
    } else {
      encoded = EncodeDepthAttachmentUnorm(shader_depth,
                                           state.depth_attachment_format);
      passes = PbeLateDepthPass(state.raster_state.depth.compare_op, encoded,
                                depth[sample_index]);
    }
    if (!passes && counters)
      ++counters->depth_rejected_fragments;
  }

  if (stencil_state.test_enable && !stencil.empty()) {
    const StencilOp operation = !stencil_passes
                                    ? face.fail_op
                                    : passes ? face.pass_op
                                             : face.depth_fail_op;
    const auto write_mask = static_cast<std::uint8_t>(face.write_mask);
    const std::uint8_t old = stencil[sample_index];
    const std::uint8_t updated = ApplyStencilOp(
        operation, old, static_cast<std::uint8_t>(face.reference));
    stencil[sample_index] = static_cast<std::uint8_t>(
        (old & ~write_mask) | (updated & write_mask));
    if (stencil[sample_index] != old && counters)
      ++counters->stencil_written_fragments;
  }

  if (passes && state.raster_state.depth.test_enable &&
      state.raster_state.depth.write_enable) {
    depth[sample_index] = encoded;
    if (counters)
      ++counters->depth_written_fragments;
  }
  return passes;
}

} // namespace pvrgpu::stub
