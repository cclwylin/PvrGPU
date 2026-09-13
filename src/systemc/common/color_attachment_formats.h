#pragma once

#include "common/pipeline_state.h"
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace pvrgpu::stub {

inline bool ColorAttachmentCodecFromName(std::string_view name,
                                         ColorAttachmentCodec *out) {
  if (!out)
    return false;
  ColorAttachmentCodec codec{};
  const auto set = [&](ColorAttachmentNumberFormat number,
                       std::uint8_t mask,
                       std::array<std::uint8_t, 4> bits) {
    codec.number = number;
    codec.component_mask = mask;
    codec.component_bits = bits;
  };

  // Established exact transports deliberately remain legacy.
  if (name == "PIPE_FORMAT_R8G8B8A8_UNORM" ||
      name == "PIPE_FORMAT_R8G8B8A8_SRGB" ||
      name == "PIPE_FORMAT_B8G8R8A8_SRGB" ||
      name == "PIPE_FORMAT_R10G10B10A2_UNORM" ||
      name == "PIPE_FORMAT_B10G10R10A2_UNORM" ||
      name == "PIPE_FORMAT_R32G32B32A32_FLOAT") {
    *out = codec;
    return true;
  }

  if (name == "PIPE_FORMAT_R8_UNORM")
    set(ColorAttachmentNumberFormat::kUnorm, 0x1, {8, 0, 0, 0});
  else if (name == "PIPE_FORMAT_R8G8_UNORM")
    set(ColorAttachmentNumberFormat::kUnorm, 0x3, {8, 8, 0, 0});
  else if (name == "PIPE_FORMAT_R8G8B8X8_UNORM" ||
           name == "PIPE_FORMAT_B8G8R8X8_UNORM")
    set(ColorAttachmentNumberFormat::kUnorm, 0x7, {8, 8, 8, 0});
  else if (name == "PIPE_FORMAT_R5G6B5_UNORM" ||
           name == "PIPE_FORMAT_B5G6R5_UNORM")
    set(ColorAttachmentNumberFormat::kUnorm, 0x7, {5, 6, 5, 0});
  else if (name == "PIPE_FORMAT_R16_UNORM")
    set(ColorAttachmentNumberFormat::kUnorm, 0x1, {16, 0, 0, 0});
  else if (name == "PIPE_FORMAT_R16G16_UNORM")
    set(ColorAttachmentNumberFormat::kUnorm, 0x3, {16, 16, 0, 0});
  else if (name == "PIPE_FORMAT_R16G16B16A16_UNORM")
    set(ColorAttachmentNumberFormat::kUnorm, 0xf, {16, 16, 16, 16});
  else if (name == "PIPE_FORMAT_R32G32B32A32_UNORM")
    set(ColorAttachmentNumberFormat::kUnorm, 0xf, {32, 32, 32, 32});
  else if (name == "PIPE_FORMAT_R8_SNORM")
    set(ColorAttachmentNumberFormat::kSnorm, 0x1, {8, 0, 0, 0});
  else if (name == "PIPE_FORMAT_R8G8_SNORM")
    set(ColorAttachmentNumberFormat::kSnorm, 0x3, {8, 8, 0, 0});
  else if (name == "PIPE_FORMAT_R8G8B8A8_SNORM")
    set(ColorAttachmentNumberFormat::kSnorm, 0xf, {8, 8, 8, 8});
  else if (name == "PIPE_FORMAT_R16_SNORM")
    set(ColorAttachmentNumberFormat::kSnorm, 0x1, {16, 0, 0, 0});
  else if (name == "PIPE_FORMAT_R16G16_SNORM")
    set(ColorAttachmentNumberFormat::kSnorm, 0x3, {16, 16, 0, 0});
  else if (name == "PIPE_FORMAT_R16G16B16A16_SNORM")
    set(ColorAttachmentNumberFormat::kSnorm, 0xf, {16, 16, 16, 16});
  else if (name == "PIPE_FORMAT_R16_FLOAT")
    set(ColorAttachmentNumberFormat::kFloat16, 0x1, {16, 0, 0, 0});
  else if (name == "PIPE_FORMAT_R16G16_FLOAT")
    set(ColorAttachmentNumberFormat::kFloat16, 0x3, {16, 16, 0, 0});
  else if (name == "PIPE_FORMAT_R16G16B16A16_FLOAT")
    set(ColorAttachmentNumberFormat::kFloat16, 0xf, {16, 16, 16, 16});
  else if (name == "PIPE_FORMAT_R11G11B10_FLOAT")
    set(ColorAttachmentNumberFormat::kUnsignedFloat, 0x7, {11, 11, 10, 0});
  else if (name == "PIPE_FORMAT_R32_FLOAT")
    set(ColorAttachmentNumberFormat::kFloat32, 0x1, {32, 0, 0, 0});
  else if (name == "PIPE_FORMAT_R32G32_FLOAT")
    set(ColorAttachmentNumberFormat::kFloat32, 0x3, {32, 32, 0, 0});
  else if (name == "PIPE_FORMAT_R8_UINT")
    set(ColorAttachmentNumberFormat::kUint, 0x1, {8, 0, 0, 0});
  else if (name == "PIPE_FORMAT_R8G8_UINT")
    set(ColorAttachmentNumberFormat::kUint, 0x3, {8, 8, 0, 0});
  else if (name == "PIPE_FORMAT_R8G8B8A8_UINT")
    set(ColorAttachmentNumberFormat::kUint, 0xf, {8, 8, 8, 8});
  else if (name == "PIPE_FORMAT_R16_UINT")
    set(ColorAttachmentNumberFormat::kUint, 0x1, {16, 0, 0, 0});
  else if (name == "PIPE_FORMAT_R16G16_UINT")
    set(ColorAttachmentNumberFormat::kUint, 0x3, {16, 16, 0, 0});
  else if (name == "PIPE_FORMAT_R16G16B16A16_UINT")
    set(ColorAttachmentNumberFormat::kUint, 0xf, {16, 16, 16, 16});
  else if (name == "PIPE_FORMAT_R32_UINT")
    set(ColorAttachmentNumberFormat::kUint, 0x1, {32, 0, 0, 0});
  else if (name == "PIPE_FORMAT_R32G32_UINT")
    set(ColorAttachmentNumberFormat::kUint, 0x3, {32, 32, 0, 0});
  else if (name == "PIPE_FORMAT_R32G32B32A32_UINT")
    set(ColorAttachmentNumberFormat::kUint, 0xf, {32, 32, 32, 32});
  else if (name == "PIPE_FORMAT_R10G10B10A2_UINT" ||
           name == "PIPE_FORMAT_B10G10R10A2_UINT")
    set(ColorAttachmentNumberFormat::kUint, 0xf, {10, 10, 10, 2});
  else if (name == "PIPE_FORMAT_R8_SINT")
    set(ColorAttachmentNumberFormat::kSint, 0x1, {8, 0, 0, 0});
  else if (name == "PIPE_FORMAT_R8G8_SINT")
    set(ColorAttachmentNumberFormat::kSint, 0x3, {8, 8, 0, 0});
  else if (name == "PIPE_FORMAT_R8G8B8A8_SINT")
    set(ColorAttachmentNumberFormat::kSint, 0xf, {8, 8, 8, 8});
  else if (name == "PIPE_FORMAT_R16_SINT")
    set(ColorAttachmentNumberFormat::kSint, 0x1, {16, 0, 0, 0});
  else if (name == "PIPE_FORMAT_R16G16_SINT")
    set(ColorAttachmentNumberFormat::kSint, 0x3, {16, 16, 0, 0});
  else if (name == "PIPE_FORMAT_R16G16B16A16_SINT")
    set(ColorAttachmentNumberFormat::kSint, 0xf, {16, 16, 16, 16});
  else if (name == "PIPE_FORMAT_R32_SINT")
    set(ColorAttachmentNumberFormat::kSint, 0x1, {32, 0, 0, 0});
  else if (name == "PIPE_FORMAT_R32G32_SINT")
    set(ColorAttachmentNumberFormat::kSint, 0x3, {32, 32, 0, 0});
  else if (name == "PIPE_FORMAT_R32G32B32A32_SINT")
    set(ColorAttachmentNumberFormat::kSint, 0xf, {32, 32, 32, 32});
  else
    return false;

  *out = codec;
  return true;
}

inline bool ColorAttachmentCodecIsInteger(
    const ColorAttachmentCodec &codec) {
  return codec.number == ColorAttachmentNumberFormat::kUint ||
         codec.number == ColorAttachmentNumberFormat::kSint;
}

inline bool ColorAttachmentCodecIsSpecified(
    const ColorAttachmentCodec &codec) {
  return codec.number != ColorAttachmentNumberFormat::kLegacy;
}

inline bool ColorAttachmentCodecIsCanonical(
    const ColorAttachmentCodec &codec) {
  return ColorAttachmentCodecIsSpecified(codec) &&
         !ColorAttachmentCodecIsInteger(codec);
}

inline bool ColorAttachmentCodecUsesFloat64Storage(
    const ColorAttachmentCodec &codec) {
  return codec.number == ColorAttachmentNumberFormat::kUnorm &&
         codec.component_bits == std::array<std::uint8_t, 4>{32, 32, 32, 32};
}

// UNORM attachments of at most eight bits per channel store the four bytes
// Gallium's eight-bit blend reads back: each native code bit-replicated to
// eight bits, a missing colour channel zero and a missing alpha 255.  One
// 1080p RGB565 target then fits a sequence attachment slot, and a draw
// quantizes through the same RGBA8 datapath llvmpipe uses.
inline bool ColorAttachmentCodecUsesUnorm8Storage(
    const ColorAttachmentCodec &codec) {
  if (codec.number != ColorAttachmentNumberFormat::kUnorm)
    return false;
  for (std::size_t component = 0; component < 4; ++component)
    if (codec.component_bits[component] > 8)
      return false;
  return true;
}

inline bool ColorAttachmentCodecIsValid(const ColorAttachmentCodec &codec) {
  if (codec.number == ColorAttachmentNumberFormat::kLegacy)
    return codec.component_mask == 0 &&
           codec.component_bits == std::array<std::uint8_t, 4>{};
  if (codec.number < ColorAttachmentNumberFormat::kUnorm ||
      codec.number > ColorAttachmentNumberFormat::kSint)
    return false;
  if (!codec.component_mask || (codec.component_mask & ~0xfU))
    return false;
  for (std::size_t component = 0; component < 4; ++component) {
    const bool present = codec.component_mask & (1U << component);
    const auto bits = codec.component_bits[component];
    if (present != (bits != 0))
      return false;
    if (!present)
      continue;
    if (codec.number == ColorAttachmentNumberFormat::kUnorm &&
        bits != 5 && bits != 6 && bits != 8 && bits != 16 && bits != 32)
      return false;
    if (codec.number == ColorAttachmentNumberFormat::kSnorm &&
        bits != 8 && bits != 16)
      return false;
    if (codec.number == ColorAttachmentNumberFormat::kFloat16 && bits != 16)
      return false;
    if (codec.number == ColorAttachmentNumberFormat::kFloat32 && bits != 32)
      return false;
    if (codec.number == ColorAttachmentNumberFormat::kUnsignedFloat &&
        bits != 10 && bits != 11)
      return false;
    if (ColorAttachmentCodecIsInteger(codec) && bits != 2 && bits != 8 &&
        bits != 10 && bits != 16 && bits != 32)
      return false;
  }
  return true;
}

inline bool ColorAttachmentIntegerStorageIsValid(
    const ColorAttachmentCodec &codec, std::uint8_t raw_dwords) {
  if (!ColorAttachmentCodecIsInteger(codec))
    return true;
  const std::uint8_t logical_components =
      codec.component_mask & 0x8 ? 4 : codec.component_mask & 0x4 ? 3
                              : codec.component_mask & 0x2   ? 2
                                                             : 1;
  const std::uint8_t expected_dwords = logical_components == 3
                                           ? 4
                                           : logical_components;
  return raw_dwords == expected_dwords;
}

inline bool ColorAttachmentCodecEqual(const ColorAttachmentCodec &left,
                                      const ColorAttachmentCodec &right) {
  return left.number == right.number &&
         left.component_mask == right.component_mask &&
         left.component_bits == right.component_bits;
}

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
    const auto &codec = state.color_attachment_codec;
    if (format != PackedUnormFormat::kNone &&
        format != PackedUnormFormat::kRgb10A2 &&
        format != PackedUnormFormat::kBgr10A2)
      throw std::runtime_error("invalid packed UNORM format");
    if ((raw != 0 && raw != 1 && raw != 2 && raw != 4) || float32 > 1 ||
        srgb > 1 ||
        (unsigned(raw != 0) + unsigned(float32 != 0) +
             unsigned(srgb != 0) +
             unsigned(format != PackedUnormFormat::kNone) +
             unsigned(ColorAttachmentCodecIsCanonical(codec)) >
         1) || !ColorAttachmentCodecIsValid(codec) ||
        !ColorAttachmentIntegerStorageIsValid(codec, raw))
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
           state.color_attachment_packed_unorm ||
       !ColorAttachmentCodecEqual(state.color_attachment_codecs[0],
                                  state.color_attachment_codec)))
    throw std::runtime_error("per-target color attachment format contract is invalid");
  for (std::size_t target = 0; target < state.color_attachment_packed_unorms.size(); ++target) {
    const auto format = state.color_attachment_packed_unorms[target];
    const auto raw = state.color_attachment_raw_dwords_per_target[target];
    const auto float32 = state.color_attachment_float32_per_target[target];
    const auto srgb = state.color_attachment_srgb_per_target[target];
    const auto &codec = state.color_attachment_codecs[target];
    if ((target >= count &&
         (format != PackedUnormFormat::kNone || raw || float32 || srgb ||
          ColorAttachmentCodecIsSpecified(codec))) ||
        (target < count &&
         ((raw != 0 && raw != 1 && raw != 2 && raw != 4) || float32 > 1 ||
          srgb > 1 ||
          (format != PackedUnormFormat::kNone &&
           format != PackedUnormFormat::kRgb10A2 &&
           format != PackedUnormFormat::kBgr10A2) ||
          (unsigned(raw != 0) + unsigned(float32 != 0) +
               unsigned(srgb != 0) +
               unsigned(format != PackedUnormFormat::kNone) +
               unsigned(ColorAttachmentCodecIsCanonical(codec)) >
           1) || !ColorAttachmentCodecIsValid(codec) ||
          !ColorAttachmentIntegerStorageIsValid(codec, raw))))
      throw std::runtime_error("per-target color attachment format entry is invalid");
  }
  return targets;
}

inline const ColorAttachmentCodec &ColorAttachmentCodecForTarget(
    const PipelineState &state, std::uint32_t target) {
  return state.color_attachment_format_count
             ? state.color_attachment_codecs.at(target)
             : state.color_attachment_codec;
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
  const auto &codec = ColorAttachmentCodecForTarget(state, target);
  if (ColorAttachmentCodecUsesFloat64Storage(codec))
    return 4U * sizeof(double);
  if (ColorAttachmentCodecUsesUnorm8Storage(codec))
    return 4U;
  if (ColorAttachmentCodecIsCanonical(codec))
    return 4U * sizeof(float);
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
