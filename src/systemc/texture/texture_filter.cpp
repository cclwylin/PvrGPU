// Sampling algorithms of the reference TPU.  See texture_filter.h for the
// decision structure and for which llvmpipe routine each step follows.
#include "texture/texture_filter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pvrgpu::stub {

namespace {

inline constexpr float kLinearCoordinateRoundThreshold = 0.5F;

bool IsPowerOfTwo(std::uint32_t extent) {
  return extent != 0 && (extent & (extent - 1U)) == 0;
}

// lp_build_coord_mirror with posOnly: fold the coordinate into [0, 1] by
// reflecting every odd unit interval.  coord/2 minus its nearest integer is
// in [-0.5, 0.5]; doubled and made positive that is the mirrored
// coordinate.  The rounding is llvm.nearbyint, ties to even.
float MirrorCoordinate(float coordinate) {
  const float half = coordinate * 0.5F;
  const float fraction = half - std::nearbyint(half);
  const float mirrored = std::fabs(fraction + fraction);
  return std::max(mirrored, 0.0F);
}

// lp_build_fract_safe: the fractional part, kept strictly below one so a
// coordinate that rounds up to the next integer does not address one texel
// past the end.
float FractSafe(float coordinate) {
  const float fraction = coordinate - std::floor(coordinate);
  return std::min(fraction, std::nextafter(1.0F, 0.0F));
}

std::int64_t FloorToInt64(float value) {
  const float floored = std::floor(value);
  if (floored < static_cast<float>(std::numeric_limits<std::int64_t>::min()) ||
      floored > static_cast<float>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("TextureUnit normalized coordinate overflow");
  }
  return static_cast<std::int64_t>(floored);
}

std::int64_t TruncToInt64(float value) {
  if (!std::isfinite(value) ||
      value < static_cast<float>(std::numeric_limits<std::int64_t>::min()) ||
      value > static_cast<float>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("TextureUnit normalized coordinate overflow");
  }
  return static_cast<std::int64_t>(value);
}

std::uint32_t ClampIndex(std::int64_t index, std::uint32_t extent) {
  if (index < 0)
    return 0;
  if (index >= static_cast<std::int64_t>(extent))
    return extent - 1U;
  return static_cast<std::uint32_t>(index);
}

void RequireCoordinate(float coordinate, std::uint32_t extent) {
  if (!std::isfinite(coordinate) || extent == 0)
    throw std::runtime_error("TextureUnit coordinate/extent is invalid");
}

} // namespace

float TextureFastLog2(float x) {
  if (!(x > 0.0F) || !std::isfinite(x))
    throw std::runtime_error("TextureUnit fast log2 argument is invalid");
  // frexp gives x = m * 2^e with m in [0.5, 1); llvmpipe's mantissa is in
  // [1, 2) with exponent e - 1, so its ipart = (e - 1) - 1 and fpart = 2m.
  int binary_exponent = 0;
  const float half_open_mantissa = std::frexp(x, &binary_exponent);
  const float normalized_mantissa = half_open_mantissa * 2.0F;
  return static_cast<float>(binary_exponent - 2) + normalized_mantissa;
}

TextureLodSelection SelectTextureLod(
    float rho_squared, const RogueTextureSamplerDescriptor &sampler,
    std::uint32_t mip_count) {
  if (mip_count == 0)
    throw std::runtime_error("TextureUnit implicit LOD state is invalid");
  if (rho_squared < 0.0F || !std::isfinite(rho_squared))
    throw std::runtime_error("TextureUnit implicit derivative rho is invalid");
  const float min_lod = static_cast<float>(sampler.min_lod_u4_6) / 64.0F;
  const float max_lod = static_cast<float>(sampler.max_lod_u4_6) / 64.0F;
  if (max_lod < min_lod)
    throw std::runtime_error("TextureUnit LOD window runs backwards");

  TextureLodSelection result;
  result.rho_squared = rho_squared;
  // Public SMP LODM=NORMAL derives one isotropic LOD for a 2x2 quad from
  // rho^2 = max(|d/dx|^2, |d/dy|^2) (lp_build_rho without the rho_opt
  // shortcut).  log2(rho) is half of log2(rho^2), so the log2 is taken once
  // on the square and halved: exact at powers of two, no transcendental
  // unit.
  float lambda = min_lod;
  if (rho_squared > 0.0F)
    lambda = std::clamp(TextureFastLog2(rho_squared) * 0.5F, min_lod, max_lod);
  result.lambda = lambda;
  result.minified = lambda > 0.0F;
  const std::uint32_t last_level_u4_6 = (mip_count - 1U) * 64U;
  result.clamp_active =
      sampler.min_lod_u4_6 != 0 || sampler.max_lod_u4_6 < last_level_u4_6;
  return result;
}

TextureLevelSelection SelectTextureLevels(
    const TextureLodSelection &lod,
    const RogueTextureSamplerDescriptor &sampler, std::uint32_t mip_count) {
  if (mip_count == 0 || mip_count > kMaximumTextureMipLevels)
    throw std::runtime_error("TextureUnit level selection state is invalid");
  const std::uint32_t last_level = mip_count - 1U;
  TextureLevelSelection result;
  result.image_filter = lod.minified ? sampler.min_filter : sampler.mag_filter;

  if (sampler.mip_filter == TextureFilter::kLinear && mip_count > 1U) {
    // lp_build_linear_mip_levels: floor(lambda) and the level after it,
    // both clamped to the chain, the weight zeroed where the clamp bit.  A
    // magnified fragment lands on levels 0 and 1 with a zero weight, which
    // is its level-0 filter result.
    result.mip_mode = TextureMipMode::kLinear;
    const float window_lambda =
        std::clamp(lod.lambda, 0.0F, static_cast<float>(last_level));
    const float level0_float = std::floor(window_lambda);
    result.level0 = static_cast<std::uint8_t>(level0_float);
    result.level1 = static_cast<std::uint8_t>(
        std::min(level0_float + 1.0F, static_cast<float>(last_level)));
    if (result.level0 != result.level1) {
      const float fractional = window_lambda - level0_float;
      result.mip_weight = fractional;
      // Public Rogue exposes the filtering fraction as TFRAC_byte/256.  The
      // fixed-point datapath converts the fractional LOD with fptosi after
      // multiplying by 256: strict truncation, no near-integer snap.
      result.mip_weight_u8 = static_cast<std::uint8_t>(
          std::clamp(fractional * 256.0F, 0.0F, 255.0F));
    }
    return result;
  }

  if (!lod.minified || mip_count == 1U) {
    // Magnified, or nothing to select from: the base level.
    result.mip_mode = TextureMipMode::kNone;
    return result;
  }

  // Mip nearest.  Without an active LOD clamp llvmpipe never forms lambda
  // for this mode: lp_build_ilog2_sqrt takes floor(log2(rho^2)) + 1 from the
  // exponent field and halves it, which is floor(log2(rho) + 0.5) exactly.
  // With a clamp the clamped lambda is rounded with lp_build_iround, which
  // is llvm.nearbyint: ties to even.  The two differ only where rho^2 is an
  // exact odd power of two, and a texture unit can do either in a few
  // gates.
  result.mip_mode = TextureMipMode::kNearest;
  std::int64_t level = 0;
  if (lod.clamp_active) {
    level = static_cast<std::int64_t>(std::nearbyint(lod.lambda));
  } else {
    int binary_exponent = 0;
    (void)std::frexp(lod.rho_squared, &binary_exponent);
    // frexp's exponent is floor(log2 x) + 1, which is already the "+ 1".
    const int exponent_plus_one = binary_exponent;
    // Arithmetic shift right: floor division by two, negative included.
    level = exponent_plus_one >= 0 ? exponent_plus_one / 2
                                   : -((-exponent_plus_one + 1) / 2);
  }
  result.level0 = static_cast<std::uint8_t>(ClampIndex(level, mip_count));
  result.level1 = result.level0;
  return result;
}

std::uint32_t TextureLevelTaps(const TextureLevelSelection &levels) {
  const std::uint32_t image_taps =
      levels.image_filter == TextureFilter::kLinear ? 4U : 1U;
  return levels.mip_mode == TextureMipMode::kLinear ? image_taps * 2U
                                                    : image_taps;
}

TextureFilterDatapath SelectTextureFilterDatapath(
    TextureFormat format, const RogueTextureSamplerDescriptor &sampler) {
  // util_format_fits_8unorm: every channel an 8-bit unorm and the colour
  // space linear.  An ASTC LDR block decodes to 8-bit unorm texels, which is
  // what the state tracker hands llvmpipe for it as well.
  const bool fits_8unorm = format == TextureFormat::kRgba8Unorm ||
                           format == TextureFormat::kRgbx8Unorm ||
                           format == TextureFormat::kAstcLdr;
  const auto simple_wrap = [](TextureWrapMode wrap) {
    return wrap == TextureWrapMode::kRepeat ||
           wrap == TextureWrapMode::kClampToEdge;
  };
  return fits_8unorm && simple_wrap(sampler.wrap_u) &&
                 simple_wrap(sampler.wrap_v)
             ? TextureFilterDatapath::kUnorm8
             : TextureFilterDatapath::kFloat32;
}

std::uint32_t WrapTexelIndex(std::int64_t integer, std::uint32_t extent,
                             TextureWrapMode wrap) {
  if (extent == 0)
    throw std::runtime_error("TextureUnit texel extent is invalid");
  if (wrap == TextureWrapMode::kRepeat) {
    const std::int64_t modulus = extent;
    const std::int64_t wrapped = ((integer % modulus) + modulus) % modulus;
    return static_cast<std::uint32_t>(wrapped);
  }
  if (wrap == TextureWrapMode::kClampToEdge)
    return ClampIndex(integer, extent);
  if (wrap == TextureWrapMode::kMirroredRepeat) {
    const std::int64_t period = static_cast<std::int64_t>(extent) * 2;
    const std::int64_t wrapped = ((integer % period) + period) % period;
    if (wrapped >= static_cast<std::int64_t>(extent))
      return static_cast<std::uint32_t>(period - 1 - wrapped);
    return static_cast<std::uint32_t>(wrapped);
  }
  throw std::runtime_error("TextureUnit clamp-to-border sampling is unsupported");
}

std::uint32_t ComputeTextureNearestRepeat(float coordinate,
                                          std::uint32_t extent,
                                          TextureWrapMode wrap) {
  RequireCoordinate(coordinate, extent);
  const float extent_f = static_cast<float>(extent);
  switch (wrap) {
  case TextureWrapMode::kRepeat:
    if (IsPowerOfTwo(extent))
      return WrapTexelIndex(FloorToInt64(coordinate * extent_f), extent, wrap);
    // lp_build_sample_wrap_nearest_int, non-power-of-two: the fraction of
    // the coordinate scaled and truncated.
    return ClampIndex(TruncToInt64(FractSafe(coordinate) * extent_f), extent);
  case TextureWrapMode::kClampToEdge:
    return ClampIndex(FloorToInt64(coordinate * extent_f), extent);
  default:
    throw std::runtime_error(
        "TextureUnit fixed-point nearest addressing supports repeat and clamp");
  }
}

TextureLinearAxis ComputeTextureLinearRepeat(float coordinate,
                                             std::uint32_t extent,
                                             TextureWrapMode wrap,
                                             float round_threshold) {
  RequireCoordinate(coordinate, extent);
  if (!std::isfinite(round_threshold) || round_threshold < 0.0F ||
      round_threshold > 1.0F) {
    throw std::runtime_error("TextureUnit linear round threshold is invalid");
  }
  const bool npot_repeat =
      wrap == TextureWrapMode::kRepeat && !IsPowerOfTwo(extent);
  // The 8-bit UNORM filter datapath multiplies the live binary32 coordinate
  // by N*256 in binary32, rounds to nearest-even, then subtracts the
  // half-texel centre (128).  A non-power-of-two repeat scales the fractional
  // coordinate instead so the integer wrap below never sees a full period.
  // Keeping the multiply in binary32 is observable at half-LSB boundaries
  // and is part of the versioned reference-uArch assumption.
  const float source = npot_repeat ? (coordinate - std::floor(coordinate))
                                   : coordinate;
  const float scaled_extent = static_cast<float>(extent) * 256.0F;
  const float scaled = source * scaled_extent;
  const float scaled_floor = std::floor(scaled);
  const double scaled_floor_wide = static_cast<double>(scaled_floor);
  if (scaled_floor_wide <
          static_cast<double>(std::numeric_limits<std::int64_t>::min()) +
              128.0 ||
      scaled_floor_wide >
          static_cast<double>(std::numeric_limits<std::int64_t>::max()) -
              128.0) {
    throw std::overflow_error("TextureUnit normalized coordinate overflow");
  }
  std::int64_t rounded = static_cast<std::int64_t>(scaled_floor);
  const float remainder = scaled - scaled_floor;
  if (remainder > round_threshold ||
      (remainder == round_threshold &&
       (round_threshold != kLinearCoordinateRoundThreshold ||
        (rounded & INT64_C(1)) != 0))) {
    ++rounded;
  }
  const std::int64_t centered = rounded - 128;
  const std::int64_t lower_integer =
      centered >= 0 ? centered / 256 : -((-centered + 255) / 256);
  const std::int64_t weight = centered - lower_integer * 256;
  if (weight < 0 || weight > 255)
    throw std::runtime_error("TextureUnit linear weight is invalid");
  TextureLinearAxis result;
  if (npot_repeat) {
    // lp_build_coord_repeat_npot_linear_int: a lower tap before the first
    // texel comes from the last one, and the upper tap after the last texel
    // is the first.
    std::int64_t lower = lower_integer;
    if (lower < 0)
      lower += extent;
    result.lower = static_cast<std::uint32_t>(lower);
    result.upper =
        lower == static_cast<std::int64_t>(extent) - 1 ? 0U
                                                        : result.lower + 1U;
  } else {
    result.lower = WrapTexelIndex(lower_integer, extent, wrap);
    result.upper = WrapTexelIndex(lower_integer + 1, extent, wrap);
  }
  result.weight = static_cast<std::uint16_t>(weight);
  return result;
}

std::uint8_t LerpTextureUnorm8(std::uint8_t first, std::uint8_t second,
                               std::uint16_t weight) {
  if (weight > 255)
    throw std::runtime_error("TextureUnit linear weight exceeds U8 range");
  // The common U8-normalized filter datapath rounds the signed delta
  // contribution before adding the first endpoint:
  //   first + RNE(weight * (second - first) / 256).
  // This is observably different from rounding the final weighted sum when a
  // half tie changes parity after adding first.  Use an explicit magnitude so
  // the result does not depend on the host's signed-shift representation.
  const std::int32_t product =
      static_cast<std::int32_t>(weight) *
      (static_cast<std::int32_t>(second) -
       static_cast<std::int32_t>(first));
  const bool negative = product < 0;
  const std::uint32_t magnitude = static_cast<std::uint32_t>(
      negative ? -product : product);
  std::uint32_t quotient = magnitude >> 8U;
  const std::uint32_t remainder = magnitude & 0xffU;
  if (remainder > 128U || (remainder == 128U && (quotient & 1U) != 0))
    ++quotient;
  const std::int32_t result =
      static_cast<std::int32_t>(first) +
      (negative ? -static_cast<std::int32_t>(quotient)
                : static_cast<std::int32_t>(quotient));
  if (result < 0 || result > 255)
    throw std::runtime_error("TextureUnit linear result exceeds UNORM8");
  return static_cast<std::uint8_t>(result);
}

std::uint32_t ComputeTextureFloatNearest(float coordinate,
                                         std::uint32_t extent,
                                         TextureWrapMode wrap) {
  RequireCoordinate(coordinate, extent);
  const float extent_f = static_cast<float>(extent);
  switch (wrap) {
  case TextureWrapMode::kRepeat:
    if (IsPowerOfTwo(extent))
      return WrapTexelIndex(FloorToInt64(coordinate * extent_f), extent, wrap);
    return ClampIndex(TruncToInt64(FractSafe(coordinate) * extent_f), extent);
  case TextureWrapMode::kClampToEdge:
    return ClampIndex(TruncToInt64(coordinate * extent_f), extent);
  case TextureWrapMode::kMirroredRepeat:
    return ClampIndex(TruncToInt64(MirrorCoordinate(coordinate) * extent_f),
                      extent);
  default:
    throw std::runtime_error("TextureUnit clamp-to-border sampling is unsupported");
  }
}

TextureFloatAxis ComputeTextureFloatLinear(float coordinate,
                                           std::uint32_t extent,
                                           TextureWrapMode wrap) {
  RequireCoordinate(coordinate, extent);
  const float extent_f = static_cast<float>(extent);
  TextureFloatAxis result;
  switch (wrap) {
  case TextureWrapMode::kRepeat: {
    if (IsPowerOfTwo(extent)) {
      const float centred = coordinate * extent_f - 0.5F;
      const std::int64_t lower = FloorToInt64(centred);
      result.weight = centred - static_cast<float>(lower);
      result.lower = WrapTexelIndex(lower, extent, wrap);
      result.upper = WrapTexelIndex(lower + 1, extent, wrap);
      return result;
    }
    // lp_build_coord_repeat_npot_linear: scale the fraction, so the lower
    // tap is at most one texel before the first and wraps to the last.
    const float centred =
        (coordinate - std::floor(coordinate)) * extent_f - 0.5F;
    std::int64_t lower = FloorToInt64(centred);
    result.weight = centred - static_cast<float>(lower);
    if (lower < 0)
      lower += extent;
    result.lower = static_cast<std::uint32_t>(lower);
    result.upper =
        lower == static_cast<std::int64_t>(extent) - 1 ? 0U : result.lower + 1U;
    return result;
  }
  case TextureWrapMode::kClampToEdge: {
    // Clamp the scaled coordinate to the image, centre it, and never let the
    // lower tap go below zero or the upper past the last texel.
    const float scaled = std::min(coordinate * extent_f, extent_f);
    const float centred = std::max(scaled - 0.5F, 0.0F);
    const std::int64_t lower = FloorToInt64(centred);
    result.weight = centred - static_cast<float>(lower);
    result.lower = ClampIndex(lower, extent);
    result.upper = ClampIndex(lower + 1, extent);
    return result;
  }
  case TextureWrapMode::kMirroredRepeat: {
    const float centred = MirrorCoordinate(coordinate) * extent_f - 0.5F;
    const std::int64_t lower = FloorToInt64(centred);
    result.weight = centred - static_cast<float>(lower);
    result.lower = ClampIndex(lower, extent);
    result.upper = ClampIndex(lower + 1, extent);
    return result;
  }
  default:
    throw std::runtime_error("TextureUnit clamp-to-border sampling is unsupported");
  }
}

float LerpTextureFloat(float first, float second, float weight) {
  return first + weight * (second - first);
}

std::array<float, 4> DecodeTexelToFloat(TextureFormat format,
                                        const std::array<std::uint8_t, 4> &texel) {
  std::array<float, 4> result{};
  switch (format) {
  case TextureFormat::kRgba8Unorm:
  case TextureFormat::kAstcLdr:
    for (std::size_t component = 0; component < 4; ++component)
      result[component] = static_cast<float>(texel[component]) / 255.0F;
    return result;
  case TextureFormat::kRgbx8Unorm:
    for (std::size_t component = 0; component < 3; ++component)
      result[component] = static_cast<float>(texel[component]) / 255.0F;
    result[3] = 1.0F;
    return result;
  case TextureFormat::kRgba8Srgb:
  case TextureFormat::kAstcLdrSrgb:
    // Colour through the sRGB transfer function, alpha left linear.
    for (std::size_t component = 0; component < 3; ++component)
      result[component] = SrgbChannelToLinear(texel[component]);
    result[3] = static_cast<float>(texel[3]) / 255.0F;
    return result;
  default:
    throw std::runtime_error("TextureUnit cannot decode this format as colour");
  }
}

} // namespace pvrgpu::stub
