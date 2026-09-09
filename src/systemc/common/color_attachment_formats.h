#pragma once

#include "common/pipeline_state.h"
#include <cstddef>
#include <stdexcept>

namespace pvrgpu::stub {

// Validate the complete metadata vector once at each consuming boundary,
// before loading or mutating attachment storage. Returns the effective count.
inline std::uint32_t ValidateColorAttachmentFormats(const PipelineState &state) {
  const auto targets = state.render_target_count ? state.render_target_count : 1U;
  if (targets > kMaxRenderTargets)
    throw std::runtime_error("per-target color attachment count is invalid");
  const auto count = state.color_attachment_format_count;
  if (count && (count != targets || state.color_attachment_raw_dwords ||
                state.color_attachment_float32 || state.color_is_srgb ||
                state.color_attachment_packed_unorms[0] != state.color_attachment_packed_unorm))
    throw std::runtime_error("per-target color attachment format contract is invalid");
  for (std::size_t target = 0; target < state.color_attachment_packed_unorms.size(); ++target) {
    const auto format = state.color_attachment_packed_unorms[target];
    if ((target >= count && format != PackedUnormFormat::kNone) ||
        (target < count && format != PackedUnormFormat::kNone &&
         format != PackedUnormFormat::kRgb10A2 && format != PackedUnormFormat::kBgr10A2))
      throw std::runtime_error("per-target color attachment format entry is invalid");
  }
  return targets;
}

// The caller validates the complete vector once with the helper above. The
// hot target loop then selects a codec without revalidating every attachment.
inline PackedUnormFormat ColorAttachmentPackedUnorm(const PipelineState &state,
                                                     std::uint32_t target) {
  const auto targets = state.render_target_count ? state.render_target_count : 1U;
  if (target >= targets || target >= kMaxRenderTargets ||
      (state.color_attachment_format_count && target >= state.color_attachment_format_count))
    throw std::runtime_error("per-target color attachment index is invalid");
  return state.color_attachment_format_count ? state.color_attachment_packed_unorms[target]
                                             : state.color_attachment_packed_unorm;
}
} // namespace pvrgpu::stub
