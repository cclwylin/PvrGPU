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
  if (!count) {
    const auto raw = state.color_attachment_raw_dwords;
    const auto float32 = state.color_attachment_float32;
    const auto srgb = state.color_is_srgb;
    const auto format = state.color_attachment_packed_unorm;
    if (format != PackedUnormFormat::kNone &&
        format != PackedUnormFormat::kRgb10A2 &&
        format != PackedUnormFormat::kBgr10A2)
      throw std::runtime_error("invalid packed UNORM format");
    if ((raw != 0 && raw != 1 && raw != 2 && raw != 4) || float32 > 1 ||
        srgb > 1 ||
        (unsigned(raw != 0) + unsigned(float32 != 0) +
             unsigned(srgb != 0) +
             unsigned(format != PackedUnormFormat::kNone) >
         1))
      throw std::runtime_error("packed UNORM attachment state is invalid");
  }
  if (count &&
      (count != targets ||
       state.color_attachment_raw_dwords_per_target[0] !=
           state.color_attachment_raw_dwords ||
       state.color_attachment_float32_per_target[0] !=
           state.color_attachment_float32 ||
       state.color_attachment_srgb_per_target[0] != state.color_is_srgb ||
       state.color_attachment_packed_unorms[0] !=
           state.color_attachment_packed_unorm))
    throw std::runtime_error("per-target color attachment format contract is invalid");
  for (std::size_t target = 0; target < state.color_attachment_packed_unorms.size(); ++target) {
    const auto format = state.color_attachment_packed_unorms[target];
    const auto raw = state.color_attachment_raw_dwords_per_target[target];
    const auto float32 = state.color_attachment_float32_per_target[target];
    const auto srgb = state.color_attachment_srgb_per_target[target];
    if ((target >= count &&
         (format != PackedUnormFormat::kNone || raw || float32 || srgb)) ||
        (target < count &&
         ((raw != 0 && raw != 1 && raw != 2 && raw != 4) || float32 > 1 ||
          srgb > 1 ||
          (format != PackedUnormFormat::kNone &&
           format != PackedUnormFormat::kRgb10A2 &&
           format != PackedUnormFormat::kBgr10A2) ||
          (unsigned(raw != 0) + unsigned(float32 != 0) +
               unsigned(srgb != 0) +
               unsigned(format != PackedUnormFormat::kNone) >
           1))))
      throw std::runtime_error("per-target color attachment format entry is invalid");
  }
  return targets;
}

inline std::uint8_t ColorAttachmentRawDwords(const PipelineState &state,
                                             std::uint32_t target) {
  return state.color_attachment_format_count
             ? state.color_attachment_raw_dwords_per_target.at(target)
             : state.color_attachment_raw_dwords;
}

inline std::uint8_t ColorAttachmentFloat32(const PipelineState &state,
                                           std::uint32_t target) {
  return state.color_attachment_format_count
             ? state.color_attachment_float32_per_target.at(target)
             : state.color_attachment_float32;
}

inline std::uint8_t ColorAttachmentSrgb(const PipelineState &state,
                                        std::uint32_t target) {
  return state.color_attachment_format_count
             ? state.color_attachment_srgb_per_target.at(target)
             : state.color_is_srgb;
}

inline std::size_t ColorAttachmentBytesPerPixel(const PipelineState &state,
                                                std::uint32_t target) {
  return ColorAttachmentBytesPerPixel(ColorAttachmentRawDwords(state, target),
                                      ColorAttachmentFloat32(state, target));
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
