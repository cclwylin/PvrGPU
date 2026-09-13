// Shared PBE colour commit used by the real pixel back end and by the
// coherent-framebuffer-fetch shadow in USC.  Keeping the format conversion,
// channel mask and fixed-function blend in one routine ensures a later
// fragment never fetches a raw PIXOUT value that PBE would not publish.
#pragma once

#include "common/color_attachment_formats.h"
#include "common/functional_types.h"
#include "common/pipeline_state.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace pvrgpu::stub {

inline float PbeFloatFromBits(std::uint32_t bits) {
  float value = 0.0F;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline double PbeRoundTiesToEven(double value) {
  const double lower = std::floor(value);
  const double fraction = value - lower;
  if (fraction < 0.5)
    return lower;
  if (fraction > 0.5)
    return lower + 1.0;
  return std::fmod(lower, 2.0) == 0.0 ? lower : lower + 1.0;
}

inline float PbeQuantizeHalf(float value) {
  if (!std::isfinite(value) || value == 0.0F)
    return value;
  const bool negative = std::signbit(value);
  const double magnitude = std::fabs(static_cast<double>(value));
  if (magnitude >= 65520.0)
    return negative ? -std::numeric_limits<float>::infinity()
                    : std::numeric_limits<float>::infinity();
  const int exponent = magnitude < std::ldexp(1.0, -14)
                           ? -14
                           : static_cast<int>(std::floor(std::log2(magnitude)));
  const double step = exponent == -14 && magnitude < std::ldexp(1.0, -14)
                          ? std::ldexp(1.0, -24)
                          : std::ldexp(1.0, exponent - 10);
  const double rounded = PbeRoundTiesToEven(magnitude / step) * step;
  const float result = static_cast<float>(rounded);
  return negative ? -result : result;
}

inline float PbeQuantizeUnsignedFloat(float value, unsigned bits) {
  if (std::isnan(value))
    return value;
  if (value <= 0.0F)
    return 0.0F;
  if (std::isinf(value))
    return value;
  const unsigned mantissa_bits = bits == 11 ? 6U : 5U;
  const double maximum = bits == 11 ? 65024.0 : 64512.0;
  const double magnitude = std::min<double>(value, maximum);
  const double minimum_normal = std::ldexp(1.0, -14);
  const int exponent = magnitude < minimum_normal
                           ? -14
                           : static_cast<int>(std::floor(std::log2(magnitude)));
  const double step = exponent == -14 && magnitude < minimum_normal
                          ? std::ldexp(1.0, -14 -
                                                static_cast<int>(mantissa_bits))
                          : std::ldexp(1.0,
                                       exponent -
                                           static_cast<int>(mantissa_bits));
  return static_cast<float>(
      std::min(PbeRoundTiesToEven(magnitude / step) * step, maximum));
}

inline float PbeCanonicalComponent(const ColorAttachmentCodec &codec,
                                   std::size_t component, float value) {
  if (!(codec.component_mask & (1U << component)))
    return component == 3 ? 1.0F : 0.0F;
  if (std::isnan(value) &&
      (codec.number == ColorAttachmentNumberFormat::kUnorm ||
       codec.number == ColorAttachmentNumberFormat::kSnorm))
    return 0.0F;
  const auto bits = codec.component_bits[component];
  switch (codec.number) {
  case ColorAttachmentNumberFormat::kUnorm: {
    const double maximum = bits == 32
                               ? 4294967295.0
                               : static_cast<double>((UINT64_C(1) << bits) - 1U);
    const float clamped = std::clamp(value, 0.0F, 1.0F);
    const double scaled = static_cast<double>(
        clamped * static_cast<float>(maximum));
    return static_cast<float>(
        std::min(PbeRoundTiesToEven(scaled), maximum) / maximum);
  }
  case ColorAttachmentNumberFormat::kSnorm: {
    const double maximum =
        static_cast<double>((UINT64_C(1) << (bits - 1U)) - 1U);
    const float clamped = std::clamp(value, -1.0F, 1.0F);
    const double scaled = static_cast<double>(
        clamped * static_cast<float>(maximum));
    const double magnitude = PbeRoundTiesToEven(std::fabs(scaled));
    return static_cast<float>(std::copysign(magnitude, scaled) / maximum);
  }
  case ColorAttachmentNumberFormat::kFloat16:
    return PbeQuantizeHalf(value);
  case ColorAttachmentNumberFormat::kUnsignedFloat:
    return PbeQuantizeUnsignedFloat(value, bits);
  case ColorAttachmentNumberFormat::kFloat32:
    return value;
  case ColorAttachmentNumberFormat::kUint:
  case ColorAttachmentNumberFormat::kSint:
  case ColorAttachmentNumberFormat::kLegacy:
    break;
  }
  throw std::runtime_error("PBE received an invalid canonical color codec");
}

inline std::uint32_t PbeCanonicalIntegerComponent(
    const ColorAttachmentCodec &codec, std::size_t component,
    std::uint32_t raw_value) {
  if (!ColorAttachmentCodecIsInteger(codec) ||
      !(codec.component_mask & (1U << component)))
    throw std::runtime_error("PBE received an invalid integer color codec");
  const unsigned bits = codec.component_bits[component];
  if (bits == 32)
    return raw_value;
  if (codec.number == ColorAttachmentNumberFormat::kUint) {
    const std::uint32_t maximum = (UINT32_C(1) << bits) - 1U;
    return std::min(raw_value, maximum);
  }
  std::int32_t signed_value = 0;
  std::memcpy(&signed_value, &raw_value, sizeof(signed_value));
  const std::int64_t minimum = -(INT64_C(1) << (bits - 1U));
  const std::int64_t maximum = (INT64_C(1) << (bits - 1U)) - 1;
  const std::int32_t canonical = static_cast<std::int32_t>(
      std::clamp<std::int64_t>(signed_value, minimum, maximum));
  std::uint32_t result = 0;
  std::memcpy(&result, &canonical, sizeof(result));
  return result;
}

inline std::array<std::uint32_t, 4> PbeReadRawIntegerColorForShader(
    std::uint8_t raw_dwords, const std::uint8_t *source = nullptr) {
  if (raw_dwords != 1 && raw_dwords != 2 && raw_dwords != 4)
    throw std::runtime_error("PBE received an invalid raw integer pixel width");
  std::array<std::uint32_t, 4> color = {0, 0, 0, 1};
  if (source)
    std::memcpy(color.data(), source,
                raw_dwords * sizeof(std::uint32_t));
  return color;
}

inline float PbeClampSourceForCodec(const ColorAttachmentCodec &codec,
                                    float value) {
  if (std::isnan(value) &&
      (codec.number == ColorAttachmentNumberFormat::kUnorm ||
       codec.number == ColorAttachmentNumberFormat::kSnorm))
    return 0.0F;
  if (codec.number == ColorAttachmentNumberFormat::kUnorm)
    return std::clamp(value, 0.0F, 1.0F);
  if (codec.number == ColorAttachmentNumberFormat::kSnorm)
    return std::clamp(value, -1.0F, 1.0F);
  return value;
}

inline std::array<float, 4> PbeCanonicalizeColor(
    const ColorAttachmentCodec &codec, const std::array<float, 4> &color) {
  std::array<float, 4> result{};
  for (std::size_t component = 0; component < result.size(); ++component)
    result[component] =
        PbeCanonicalComponent(codec, component, color[component]);
  return result;
}

// R32G32B32A32_UNORM needs more than float32 storage: an arbitrary native
// uint32 code is not generally recoverable after conversion to float.  Keep
// its canonical values in doubles so LOAD/readback and masked channels retain
// every native bit; shader inputs are narrowed to float only at USC ingress.
inline double PbeCanonicalComponentFloat64(
    const ColorAttachmentCodec &codec, std::size_t component, double value) {
  if (!ColorAttachmentCodecUsesFloat64Storage(codec))
    throw std::runtime_error("PBE received a non-float64 color codec");
  if (!(codec.component_mask & (1U << component)))
    return component == 3 ? 1.0 : 0.0;
  if (std::isnan(value))
    return 0.0;
  constexpr double maximum = 4294967295.0;
  const double clamped = std::clamp(value, 0.0, 1.0);
  return std::min(PbeRoundTiesToEven(clamped * maximum), maximum) / maximum;
}

inline void PbeWriteCanonicalColor(
    const ColorAttachmentCodec &codec, std::uint8_t *destination,
    const std::array<float, 4> &color) {
  if (ColorAttachmentCodecUsesFloat64Storage(codec)) {
    std::array<double, 4> canonical{};
    for (std::size_t component = 0; component < canonical.size(); ++component)
      canonical[component] = PbeCanonicalComponentFloat64(
          codec, component, static_cast<double>(color[component]));
    std::memcpy(destination, canonical.data(), sizeof(canonical));
    return;
  }
  const auto canonical = PbeCanonicalizeColor(codec, color);
  std::memcpy(destination, canonical.data(), sizeof(canonical));
}

inline std::array<float, 4> PbeReadCanonicalColorForShader(
    const ColorAttachmentCodec &codec, const std::uint8_t *source) {
  std::array<float, 4> color{};
  if (ColorAttachmentCodecUsesFloat64Storage(codec)) {
    std::array<double, 4> canonical{};
    std::memcpy(canonical.data(), source, sizeof(canonical));
    for (std::size_t component = 0; component < color.size(); ++component)
      color[component] = static_cast<float>(canonical[component]);
  } else {
    std::memcpy(color.data(), source, sizeof(color));
  }
  return color;
}

inline float PbeClampShaderUnorm(float value) {
  // GLES clamps shader colours at a normalized attachment conversion.  Mesa
  // maps NaN to zero; infinities saturate through std::clamp.
  return std::isnan(value) ? 0.0F : std::clamp(value, 0.0F, 1.0F);
}

inline std::uint8_t PbeFloatToUnorm8(float value) {
  const float scaled = PbeClampShaderUnorm(value) * 255.0F;
  // Gallivm's RGBA8 path uses floor(x * 255 + .5), not ties-to-even.
  const std::uint32_t rounded =
      static_cast<std::uint32_t>(std::floor(scaled + 0.5F));
  return static_cast<std::uint8_t>(std::min<std::uint32_t>(rounded, 255U));
}

inline std::uint8_t PbeFloatBitsToUnorm8(std::uint32_t bits) {
  return PbeFloatToUnorm8(PbeFloatFromBits(bits));
}

inline std::uint8_t PbeFiniteStateToUnorm8(float value) {
  if (!std::isfinite(value))
    throw std::runtime_error("PBE cannot convert non-finite UNORM state");
  return PbeFloatToUnorm8(value);
}

inline std::uint8_t PbeBlendFactorUnorm8(
    BlendFactor factor, const std::array<std::uint8_t, 4> &source,
    const std::array<std::uint8_t, 4> &destination,
    const std::array<std::uint8_t, 4> &constant, std::size_t component) {
  switch (factor) {
  case BlendFactor::kZero:
    return 0;
  case BlendFactor::kOne:
    return 255;
  case BlendFactor::kSourceAlpha:
    return source[3];
  case BlendFactor::kOneMinusSourceAlpha:
    return static_cast<std::uint8_t>(255U - source[3]);
  case BlendFactor::kSourceColor:
    return source[component];
  case BlendFactor::kOneMinusSourceColor:
    return static_cast<std::uint8_t>(255U - source[component]);
  case BlendFactor::kDestinationColor:
    return destination[component];
  case BlendFactor::kOneMinusDestinationColor:
    return static_cast<std::uint8_t>(255U - destination[component]);
  case BlendFactor::kDestinationAlpha:
    return destination[3];
  case BlendFactor::kOneMinusDestinationAlpha:
    return static_cast<std::uint8_t>(255U - destination[3]);
  case BlendFactor::kSourceAlphaSaturate:
    return component == 3
               ? static_cast<std::uint8_t>(255U)
               : std::min<std::uint8_t>(
                     source[3],
                     static_cast<std::uint8_t>(255U - destination[3]));
  case BlendFactor::kConstantColor:
    return constant[component];
  case BlendFactor::kOneMinusConstantColor:
    return static_cast<std::uint8_t>(255U - constant[component]);
  case BlendFactor::kConstantAlpha:
    return constant[3];
  case BlendFactor::kOneMinusConstantAlpha:
    return static_cast<std::uint8_t>(255U - constant[3]);
  }
  throw std::runtime_error("PBE received an unsupported blend factor");
}

inline std::uint8_t PbeBlendEquationUnorm8(
    BlendEquation equation, std::uint8_t source, std::uint8_t destination,
    std::uint8_t source_factor, std::uint8_t destination_factor) {
  if (equation == BlendEquation::kMin)
    return std::min(source, destination);
  if (equation == BlendEquation::kMax)
    return std::max(source, destination);

  const std::int32_t source_term =
      static_cast<std::int32_t>(source) * source_factor;
  const std::int32_t destination_term =
      static_cast<std::int32_t>(destination) * destination_factor;
  std::int32_t result = 0;
  if (equation == BlendEquation::kAdd)
    result = (source_term + destination_term + 127) / 255;
  else if (equation == BlendEquation::kSubtract)
    result = (source_term - destination_term + 127) / 255;
  else if (equation == BlendEquation::kReverseSubtract)
    result = (destination_term - source_term + 127) / 255;
  else
    throw std::runtime_error(
        "PBE received an unsupported blend equation in calculation");
  return static_cast<std::uint8_t>(std::clamp(result, 0, 255));
}

inline float PbeBlendFactorFloat(
    BlendFactor factor, const std::array<float, 4> &source,
    const std::array<float, 4> &destination,
    const std::array<float, 4> &constant, std::size_t component) {
  switch (factor) {
  case BlendFactor::kZero:
    return 0.0F;
  case BlendFactor::kOne:
    return 1.0F;
  case BlendFactor::kSourceAlpha:
    return source[3];
  case BlendFactor::kOneMinusSourceAlpha:
    return 1.0F - source[3];
  case BlendFactor::kSourceColor:
    return source[component];
  case BlendFactor::kOneMinusSourceColor:
    return 1.0F - source[component];
  case BlendFactor::kDestinationColor:
    return destination[component];
  case BlendFactor::kOneMinusDestinationColor:
    return 1.0F - destination[component];
  case BlendFactor::kDestinationAlpha:
    return destination[3];
  case BlendFactor::kOneMinusDestinationAlpha:
    return 1.0F - destination[3];
  case BlendFactor::kSourceAlphaSaturate:
    return component == 3 ? 1.0F
                          : std::min(source[3], 1.0F - destination[3]);
  case BlendFactor::kConstantColor:
    return constant[component];
  case BlendFactor::kOneMinusConstantColor:
    return 1.0F - constant[component];
  case BlendFactor::kConstantAlpha:
    return constant[3];
  case BlendFactor::kOneMinusConstantAlpha:
    return 1.0F - constant[3];
  }
  throw std::runtime_error("PBE received an unsupported blend factor");
}

inline float PbeBlendEquationFloat(BlendEquation equation, float source,
                                   float destination, float source_factor,
                                   float destination_factor) {
  if (equation == BlendEquation::kMin)
    return std::min(source, destination);
  if (equation == BlendEquation::kMax)
    return std::max(source, destination);
  const float source_term = source * source_factor;
  const float destination_term = destination * destination_factor;
  if (equation == BlendEquation::kAdd)
    return source_term + destination_term;
  if (equation == BlendEquation::kSubtract)
    return source_term - destination_term;
  if (equation == BlendEquation::kReverseSubtract)
    return destination_term - source_term;
  throw std::runtime_error(
      "PBE received an unsupported blend equation in calculation");
}

inline void ValidatePbeBlendState(const BlendState &blend) {
  if (!blend.enable)
    return;
  const auto valid_equation = [](BlendEquation equation) {
    return equation == BlendEquation::kAdd ||
           equation == BlendEquation::kSubtract ||
           equation == BlendEquation::kReverseSubtract ||
           equation == BlendEquation::kMin || equation == BlendEquation::kMax;
  };
  if (!valid_equation(blend.rgb_equation))
    throw std::runtime_error("PBE received an unsupported RGB blend equation");
  if (!valid_equation(blend.alpha_equation))
    throw std::runtime_error(
        "PBE received an unsupported alpha blend equation");
  const std::array<BlendFactor, 4> factors = {
      blend.source_rgb_factor, blend.destination_rgb_factor,
      blend.source_alpha_factor, blend.destination_alpha_factor};
  const std::array<std::uint8_t, 4> zero{};
  for (const BlendFactor factor : factors)
    (void)PbeBlendFactorUnorm8(factor, zero, zero, zero, 0);
}

// Commit one fragment/sample/attachment to already encoded attachment bytes.
// Coverage and late depth/stencil acceptance are deliberately caller-owned;
// this routine is the single source of truth after those tests have passed.
inline void CommitPbeColorSample(const PipelineState &state,
                                 const FragmentOutput &output,
                                 std::uint32_t target,
                                 std::uint8_t *destination) {
  if (!destination || target >= output.render_target_count)
    throw std::runtime_error("PBE color commit target is invalid");
  const BlendState &blend = BlendStateForTarget(state.raster_state, target);
  const std::uint8_t channel_mask = static_cast<std::uint8_t>(
      ColorMaskForTarget(state.raster_state, target) &
      output.written_mask[target]);
  if (channel_mask == 0)
    return;

  const PackedUnormFormat packed =
      ColorAttachmentPackedUnorm(state, target);
  const std::uint8_t raw_dwords = ColorAttachmentRawDwords(state, target);
  const bool float32 = ColorAttachmentFloat32(state, target) != 0;
  const bool srgb = ColorAttachmentSrgb(state, target) != 0;
  const ColorAttachmentCodec &codec =
      ColorAttachmentCodecForTarget(state, target);
  const std::uint32_t *raw_source = output.pixel_output + target * 4U;
  const bool alpha_to_one =
      state.raster_state.alpha_to_one &&
      (state.fragment_output_mask[target] & output.written_mask[target] & 8U);

  if (ColorAttachmentCodecIsCanonical(codec)) {
    std::array<float, 4> source{};
    std::array<float, 4> stored{};
    std::array<double, 4> stored_float64{};
    if (ColorAttachmentCodecUsesFloat64Storage(codec)) {
      std::memcpy(stored_float64.data(), destination, sizeof(stored_float64));
      for (std::size_t component = 0; component < stored.size(); ++component)
        stored[component] = static_cast<float>(stored_float64[component]);
    } else {
      std::memcpy(stored.data(), destination, sizeof(stored));
    }
    for (std::size_t component = 0; component < source.size(); ++component)
      source[component] = PbeClampSourceForCodec(
          codec, PbeFloatFromBits(raw_source[component]));
    if (alpha_to_one)
      source[3] = 1.0F;

    std::array<float, 4> candidate = source;
    if (blend.enable) {
      std::array<float, 4> constant{};
      for (std::size_t component = 0; component < constant.size(); ++component)
        constant[component] = std::clamp(
            PbeFloatFromBits(blend.constant_color_bits[component]), 0.0F,
            1.0F);
      for (std::size_t component = 0; component < candidate.size();
           ++component) {
        const BlendFactor source_factor =
            component == 3 ? blend.source_alpha_factor
                           : blend.source_rgb_factor;
        const BlendFactor destination_factor =
            component == 3 ? blend.destination_alpha_factor
                           : blend.destination_rgb_factor;
        const BlendEquation equation = component == 3 ? blend.alpha_equation
                                                        : blend.rgb_equation;
        candidate[component] = PbeBlendEquationFloat(
            equation, source[component], stored[component],
            PbeBlendFactorFloat(source_factor, source, stored, constant,
                                component),
            PbeBlendFactorFloat(destination_factor, source, stored, constant,
                                component));
      }
    }

    if (ColorAttachmentCodecUsesFloat64Storage(codec)) {
      std::array<double, 4> result = stored_float64;
      for (std::size_t component = 0; component < result.size(); ++component) {
        if (!(codec.component_mask & (1U << component))) {
          result[component] = component == 3 ? 1.0 : 0.0;
        } else if (channel_mask & (1U << component)) {
          result[component] = PbeCanonicalComponentFloat64(
              codec, component, static_cast<double>(candidate[component]));
        }
      }
      std::memcpy(destination, result.data(), sizeof(result));
      return;
    }

    std::array<float, 4> result = stored;
    for (std::size_t component = 0; component < result.size(); ++component) {
      if (!(codec.component_mask & (1U << component))) {
        result[component] = component == 3 ? 1.0F : 0.0F;
      } else if (channel_mask & (1U << component)) {
        result[component] =
            PbeCanonicalComponent(codec, component, candidate[component]);
      }
    }
    std::memcpy(destination, result.data(), sizeof(result));
    return;
  }

  if (float32 || packed != PackedUnormFormat::kNone) {
    std::array<float, 4> source{};
    std::array<float, 4> stored{};
    for (std::size_t component = 0; component < source.size(); ++component)
      source[component] = PbeFloatFromBits(raw_source[component]);
    if (alpha_to_one)
      source[3] = 1.0F;
    std::uint32_t packed_stored = 0;
    if (packed != PackedUnormFormat::kNone) {
      for (float &component : source)
        component = PbeClampShaderUnorm(component);
      std::memcpy(&packed_stored, destination, sizeof(packed_stored));
      stored = UnpackUnormColor(packed_stored, packed);
    } else {
      std::memcpy(stored.data(), destination, sizeof(stored));
    }
    std::array<float, 4> result = source;
    if (blend.enable) {
      std::array<float, 4> constant{};
      for (std::size_t component = 0; component < constant.size(); ++component)
        constant[component] = std::clamp(
            PbeFloatFromBits(blend.constant_color_bits[component]), 0.0F,
            1.0F);
      for (std::size_t component = 0; component < result.size(); ++component) {
        const BlendFactor source_factor =
            component == 3 ? blend.source_alpha_factor
                           : blend.source_rgb_factor;
        const BlendFactor destination_factor =
            component == 3 ? blend.destination_alpha_factor
                           : blend.destination_rgb_factor;
        const BlendEquation equation = component == 3 ? blend.alpha_equation
                                                        : blend.rgb_equation;
        result[component] = PbeBlendEquationFloat(
            equation, source[component], stored[component],
            PbeBlendFactorFloat(source_factor, source, stored, constant,
                                component),
            PbeBlendFactorFloat(destination_factor, source, stored, constant,
                                component));
      }
    }
    if (packed != PackedUnormFormat::kNone) {
      const std::uint32_t word =
          PackUnormColor(result, packed, packed_stored, channel_mask);
      std::memcpy(destination, &word, sizeof(word));
      return;
    }
    for (std::size_t component = 0; component < result.size(); ++component) {
      if (channel_mask & (1U << component))
        std::memcpy(destination + component * sizeof(float),
                    &result[component], sizeof(float));
    }
    return;
  }

  if (raw_dwords != 0) {
    // Unnormalized integer color buffers bypass blending even when GL_BLEND
    // is enabled for the draw; native range conversion still happens here.
    for (std::size_t component = 0; component < raw_dwords; ++component) {
      if (!(channel_mask & (1U << component)))
        continue;
      std::uint32_t value =
          component == 3 && alpha_to_one ? UINT32_C(0x3f800000)
                                         : raw_source[component];
      if (ColorAttachmentCodecIsInteger(codec))
        value = PbeCanonicalIntegerComponent(codec, component, value);
      std::memcpy(destination + component * sizeof(std::uint32_t), &value,
                  sizeof(value));
    }
    return;
  }

  if (srgb) {
    std::array<float, 4> source{};
    std::array<float, 4> result{};
    for (std::size_t component = 0; component < source.size(); ++component)
      source[component] =
          PbeClampShaderUnorm(PbeFloatFromBits(raw_source[component]));
    if (alpha_to_one)
      source[3] = 1.0F;
    result = source;
    if (blend.enable) {
      std::array<float, 4> stored{};
      for (std::size_t component = 0; component < 3; ++component)
        stored[component] = SrgbChannelToLinear(destination[component]);
      stored[3] = static_cast<float>(destination[3]) / 255.0F;
      std::array<float, 4> constant{};
      for (std::size_t component = 0; component < constant.size(); ++component)
        constant[component] = std::clamp(
            PbeFloatFromBits(blend.constant_color_bits[component]), 0.0F,
            1.0F);
      for (std::size_t component = 0; component < result.size(); ++component) {
        const BlendFactor source_factor =
            component == 3 ? blend.source_alpha_factor
                           : blend.source_rgb_factor;
        const BlendFactor destination_factor =
            component == 3 ? blend.destination_alpha_factor
                           : blend.destination_rgb_factor;
        const BlendEquation equation = component == 3 ? blend.alpha_equation
                                                        : blend.rgb_equation;
        result[component] = PbeBlendEquationFloat(
            equation, source[component], stored[component],
            PbeBlendFactorFloat(source_factor, source, stored, constant,
                                component),
            PbeBlendFactorFloat(destination_factor, source, stored, constant,
                                component));
      }
    }
    for (std::size_t component = 0; component < result.size(); ++component) {
      if (!(channel_mask & (1U << component)))
        continue;
      destination[component] = component == 3
                                   ? PbeFloatToUnorm8(result[3])
                                   : LinearChannelToSrgbUnorm8(
                                         PbeClampShaderUnorm(result[component]));
    }
    return;
  }

  std::array<std::uint8_t, 4> source{};
  for (std::size_t component = 0; component < source.size(); ++component)
    source[component] = PbeFloatBitsToUnorm8(raw_source[component]);
  if (alpha_to_one)
    source[3] = UINT8_C(255);
  if (!blend.enable) {
    for (std::size_t component = 0; component < source.size(); ++component)
      if (channel_mask & (1U << component))
        destination[component] = source[component];
    return;
  }

  std::array<std::uint8_t, 4> stored{};
  std::copy_n(destination, stored.size(), stored.begin());
  std::array<std::uint8_t, 4> constant{};
  for (std::size_t component = 0; component < constant.size(); ++component)
    constant[component] =
        PbeFiniteStateToUnorm8(PbeFloatFromBits(blend.constant_color_bits[component]));
  for (std::size_t component = 0; component < source.size(); ++component) {
    if (!(channel_mask & (1U << component)))
      continue;
    const BlendFactor source_factor =
        component == 3 ? blend.source_alpha_factor : blend.source_rgb_factor;
    const BlendFactor destination_factor =
        component == 3 ? blend.destination_alpha_factor
                       : blend.destination_rgb_factor;
    const BlendEquation equation =
        component == 3 ? blend.alpha_equation : blend.rgb_equation;
    destination[component] = PbeBlendEquationUnorm8(
        equation, source[component], stored[component],
        PbeBlendFactorUnorm8(source_factor, source, stored, constant,
                             component),
        PbeBlendFactorUnorm8(destination_factor, source, stored, constant,
                             component));
  }
}

} // namespace pvrgpu::stub
