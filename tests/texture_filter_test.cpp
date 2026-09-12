// Unit checks for the reference TPU's sampling algorithms
// (src/systemc/texture/texture_filter.{h,cpp}).  Every expected value below
// is worked from the algorithm's definition -- an exponent, a floor, a
// rounding rule -- never from a captured image.
#include "texture/texture_filter.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

using pvrgpu::stub::ComputeTextureFloatLinear;
using pvrgpu::stub::ComputeTextureFloatNearest;
using pvrgpu::stub::ComputeTextureLinearRepeat;
using pvrgpu::stub::ComputeTextureNearestRepeat;
using pvrgpu::stub::DecodeTexelToFloat;
using pvrgpu::stub::LerpTextureFloat;
using pvrgpu::stub::LerpTextureUnorm8;
using pvrgpu::stub::RogueTextureSamplerDescriptor;
using pvrgpu::stub::SelectTextureFilterDatapath;
using pvrgpu::stub::SelectTextureLevels;
using pvrgpu::stub::SelectTextureBiasedLod;
using pvrgpu::stub::SelectTextureLod;
using pvrgpu::stub::TextureBytesPerTexel;
using pvrgpu::stub::TextureFastLog2;
using pvrgpu::stub::TextureFilter;
using pvrgpu::stub::TextureFilterDatapath;
using pvrgpu::stub::TextureFloatAxis;
using pvrgpu::stub::TextureFormat;
using pvrgpu::stub::TextureLevelSelection;
using pvrgpu::stub::TextureLevelTaps;
using pvrgpu::stub::TextureLinearAxis;
using pvrgpu::stub::TextureLodSelection;
using pvrgpu::stub::TextureMipMode;
using pvrgpu::stub::TextureWrapMode;

namespace {

int failures = 0;

void Check(bool condition, const std::string &message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
  }
}

bool Near(float actual, float expected) {
  return std::fabs(actual - expected) <= 1e-6F;
}

RogueTextureSamplerDescriptor Sampler(TextureFilter min_filter,
                                      TextureFilter mag_filter,
                                      TextureFilter mip_filter,
                                      std::uint16_t max_lod_u4_6,
                                      TextureWrapMode wrap = TextureWrapMode::kRepeat) {
  RogueTextureSamplerDescriptor sampler;
  sampler.min_filter = min_filter;
  sampler.mag_filter = mag_filter;
  sampler.mip_filter = mip_filter;
  sampler.wrap_u = wrap;
  sampler.wrap_v = wrap;
  sampler.min_lod_u4_6 = 0;
  sampler.max_lod_u4_6 = max_lod_u4_6;
  return sampler;
}

void CheckLog2AndLod() {
  // lp_build_fast_log2 is exact at powers of two and piece-wise linear
  // between them: 1.5 = 2^0 * 1.5 gives (0 - 1) + 1.5.
  Check(Near(TextureFastLog2(4.0F), 2.0F), "fast log2 of 4 is 2");
  Check(Near(TextureFastLog2(0.25F), -2.0F), "fast log2 of 1/4 is -2");
  Check(Near(TextureFastLog2(1.5F), 0.5F), "fast log2 of 1.5 is 0.5");
  Check(Near(TextureFastLog2(3.0F), 1.5F), "fast log2 of 3 is 1.5");

  // Ten levels, window 0..9: no active clamp.
  const RogueTextureSamplerDescriptor open =
      Sampler(TextureFilter::kLinear, TextureFilter::kLinear,
              TextureFilter::kLinear, 9 * 64);
  const TextureLodSelection four = SelectTextureLod(4.0F, open, 10);
  Check(Near(four.lambda, 1.0F) && four.minified && !four.clamp_active,
        "rho^2 of 4 is lambda 1, minified, unclamped");
  const TextureLodSelection one = SelectTextureLod(1.0F, open, 10);
  Check(Near(one.lambda, 0.0F) && !one.minified,
        "rho^2 of 1 is lambda 0: magnified (c = 0)");
  const TextureLodSelection quarter = SelectTextureLod(0.25F, open, 10);
  Check(Near(quarter.lambda, 0.0F) && !quarter.minified,
        "a negative lambda clamps to the window minimum");
  const TextureLodSelection zero = SelectTextureLod(0.0F, open, 10);
  Check(Near(zero.lambda, 0.0F) && !zero.minified,
        "the degenerate quad selects the window minimum");
  const TextureLodSelection huge = SelectTextureLod(1.0e12F, open, 10);
  Check(Near(huge.lambda, 9.0F) && huge.minified,
        "lambda clamps to the window maximum");
  const TextureLodSelection fast_mid = SelectTextureLod(12.0F, open, 10);
  const TextureLodSelection exact_mid = SelectTextureLod(12.0F, open, 10, true);
  Check(Near(fast_mid.lambda, 1.75F) &&
            Near(exact_mid.lambda, std::log2(12.0F) * 0.5F) &&
            exact_mid.lambda > fast_mid.lambda,
        "exact conformance LOD is distinct from Capture/Play fast-log2");
  const TextureLodSelection exact_biased =
      SelectTextureBiasedLod(12.0F, 0.25F, open, 10, true);
  Check(Near(exact_biased.lambda, std::log2(12.0F) * 0.5F + 0.25F),
        "exact LOD mode also applies before shader bias");

  // The mip-NONE window: 0..0.25 keeps the minification decision.
  const RogueTextureSamplerDescriptor base_only =
      Sampler(TextureFilter::kNearest, TextureFilter::kLinear,
              TextureFilter::kNearest, 16);
  const TextureLodSelection clamped = SelectTextureLod(64.0F, base_only, 10);
  Check(Near(clamped.lambda, 0.25F) && clamped.minified && clamped.clamp_active,
        "a 0..0.25 window clamps lambda to 0.25 and stays minified");
  const TextureLevelSelection base_levels =
      SelectTextureLevels(clamped, base_only, 10);
  Check(base_levels.level0 == 0 && base_levels.level1 == 0 &&
            base_levels.mip_mode == TextureMipMode::kNearest &&
            base_levels.image_filter == TextureFilter::kNearest,
        "0..0.25 window: base level with the minification filter");
  const TextureLevelSelection base_mag = SelectTextureLevels(
      SelectTextureLod(0.5F, base_only, 10), base_only, 10);
  Check(base_mag.level0 == 0 && base_mag.mip_mode == TextureMipMode::kNone &&
            base_mag.image_filter == TextureFilter::kLinear,
        "0..0.25 window magnified: base level with the magnification filter");
}

void CheckLevelSelection() {
  const RogueTextureSamplerDescriptor nearest_mip =
      Sampler(TextureFilter::kNearest, TextureFilter::kNearest,
              TextureFilter::kNearest, 9 * 64);
  // rho^2 = 2 is rho = sqrt 2, log2 rho = 0.5 exactly.  Without a clamp
  // lp_build_ilog2_sqrt takes (floor(log2 2) + 1) >> 1 = 1; with one,
  // nearbyint(0.5) = 0.  The two algorithms differ only on this tie.
  const TextureLevelSelection tie_open = SelectTextureLevels(
      SelectTextureLod(2.0F, nearest_mip, 10), nearest_mip, 10);
  Check(tie_open.level0 == 1 && tie_open.level1 == 1 &&
            tie_open.mip_mode == TextureMipMode::kNearest,
        "mip nearest, no clamp: rho^2 = 2 rounds up from the exponent");
  const RogueTextureSamplerDescriptor nearest_mip_clamped =
      Sampler(TextureFilter::kNearest, TextureFilter::kNearest,
              TextureFilter::kNearest, 5 * 64);
  const TextureLevelSelection tie_clamped = SelectTextureLevels(
      SelectTextureLod(2.0F, nearest_mip_clamped, 10), nearest_mip_clamped, 10);
  Check(tie_clamped.level0 == 0,
        "mip nearest, clamped: nearbyint(0.5) is 0 (ties to even)");
  const TextureLevelSelection eight = SelectTextureLevels(
      SelectTextureLod(8.0F, nearest_mip, 10), nearest_mip, 10);
  Check(eight.level0 == 2, "mip nearest: rho^2 = 8 is level 2");
  const TextureLevelSelection eight_clamped = SelectTextureLevels(
      SelectTextureLod(8.0F, nearest_mip_clamped, 10), nearest_mip_clamped, 10);
  Check(eight_clamped.level0 == 2,
        "mip nearest, clamped: nearbyint(1.5) is 2 (ties to even)");
  const TextureLevelSelection nearest_mag = SelectTextureLevels(
      SelectTextureLod(0.5F, nearest_mip, 10), nearest_mip, 10);
  Check(nearest_mag.level0 == 0 && nearest_mag.mip_mode == TextureMipMode::kNone &&
            TextureLevelTaps(nearest_mag) == 1,
        "mip nearest magnified: the base level, one tap");
  const TextureLevelSelection far = SelectTextureLevels(
      SelectTextureLod(1.0e12F, nearest_mip, 10), nearest_mip, 10);
  Check(far.level0 == 9, "mip nearest: the level clamps to the last one");

  const RogueTextureSamplerDescriptor trilinear =
      Sampler(TextureFilter::kLinear, TextureFilter::kLinear,
              TextureFilter::kLinear, 9 * 64);
  // rho^2 = 12: llvmpipe's fast_log2(12) = floor(log2 12) - 1 + 12/8 = 3.5
  // (an exact log2 would give 3.5849625), lambda 1.75 and TFRAC 192/256.
  const TextureLevelSelection blend = SelectTextureLevels(
      SelectTextureLod(12.0F, trilinear, 10), trilinear, 10);
  Check(blend.mip_mode == TextureMipMode::kLinear && blend.level0 == 1 &&
            blend.level1 == 2 && blend.mip_weight_u8 == 192 &&
            blend.mip_weight == 0.75F && TextureLevelTaps(blend) == 8,
        "mip linear: floor(lambda) and the next level, fraction 192/256");
  // rho^2 = 12.5: fast_log2 is 3 + 12.5/8 - 1 = 3.5625, lambda 1.78125 and
  // the fraction 200/256 exactly, so TFRAC also shows that the quantization
  // truncates the strict positive product rather than rounding it.
  const TextureLevelSelection blend_mid = SelectTextureLevels(
      SelectTextureLod(12.5F, trilinear, 10), trilinear, 10);
  Check(blend_mid.level0 == 1 && blend_mid.level1 == 2 &&
            blend_mid.mip_weight_u8 == 200 && blend_mid.mip_weight == 0.78125F,
        "mip linear: piece-wise linear fraction 200/256");
  const TextureLevelSelection blend_mag = SelectTextureLevels(
      SelectTextureLod(0.5F, trilinear, 10), trilinear, 10);
  Check(blend_mag.level0 == 0 && blend_mag.level1 == 1 &&
            blend_mag.mip_weight_u8 == 0 && blend_mag.mip_mode == TextureMipMode::kLinear,
        "mip linear magnified: levels 0 and 1 with a zero weight");
  const TextureLevelSelection blend_far = SelectTextureLevels(
      SelectTextureLod(1.0e12F, trilinear, 10), trilinear, 10);
  Check(blend_far.level0 == 9 && blend_far.level1 == 9 && blend_far.mip_weight_u8 == 0,
        "mip linear past the chain: the last level twice, zero weight");
  const TextureLevelSelection single = SelectTextureLevels(
      SelectTextureLod(12.0F, trilinear, 1), trilinear, 1);
  Check(single.mip_mode == TextureMipMode::kNone && single.level0 == 0,
        "a single-level image never blends levels");

  // min != mag: the decision is per fragment.
  const RogueTextureSamplerDescriptor mixed =
      Sampler(TextureFilter::kNearest, TextureFilter::kLinear,
              TextureFilter::kNearest, 9 * 64);
  Check(SelectTextureLevels(SelectTextureLod(4.0F, mixed, 10), mixed, 10)
                .image_filter == TextureFilter::kNearest,
        "minified: the minification filter");
  Check(SelectTextureLevels(SelectTextureLod(0.25F, mixed, 10), mixed, 10)
                .image_filter == TextureFilter::kLinear,
        "magnified: the magnification filter");
}

void CheckDatapathSelection() {
  const RogueTextureSamplerDescriptor repeat =
      Sampler(TextureFilter::kLinear, TextureFilter::kLinear,
              TextureFilter::kNearest, 0, TextureWrapMode::kRepeat);
  const RogueTextureSamplerDescriptor mirror =
      Sampler(TextureFilter::kLinear, TextureFilter::kLinear,
              TextureFilter::kNearest, 0, TextureWrapMode::kMirroredRepeat);
  Check(SelectTextureFilterDatapath(TextureFormat::kRgba8Unorm, repeat) ==
            TextureFilterDatapath::kUnorm8,
        "RGBA8 with repeat filters on the 8-bit datapath");
  Check(SelectTextureFilterDatapath(TextureFormat::kAstcLdr, repeat) ==
            TextureFilterDatapath::kUnorm8,
        "ASTC LDR decodes to 8-bit unorm and filters there");
  Check(SelectTextureFilterDatapath(TextureFormat::kRgba8Unorm, mirror) ==
            TextureFilterDatapath::kFloat32,
        "mirrored repeat is not a simple wrap: float datapath");
  Check(SelectTextureFilterDatapath(TextureFormat::kRgba8Srgb, repeat) ==
            TextureFilterDatapath::kFloat32,
        "sRGB does not fit 8-bit unorm: float datapath");
  Check(SelectTextureFilterDatapath(TextureFormat::kAstcLdrSrgb, repeat) ==
            TextureFilterDatapath::kFloat32,
        "sRGB ASTC: float datapath");
}

void CheckFixedPointAxes() {
  // Power-of-two repeat: floor(coord * 4) & 3.
  Check(ComputeTextureNearestRepeat(-0.125F, 4, TextureWrapMode::kRepeat) == 3,
        "fixed-point nearest repeat wraps a negative coordinate");
  // Non-power-of-two repeat: fract(1.0) is 0, so 1.0 wraps to the first
  // texel; just below 1.0 the safe fraction stays below one and truncates to
  // the last.
  Check(ComputeTextureNearestRepeat(1.0F, 3, TextureWrapMode::kRepeat) == 0,
        "fixed-point nearest NPOT repeat wraps 1.0 to the first texel");
  Check(ComputeTextureNearestRepeat(std::nextafter(1.0F, 0.0F), 3,
                                    TextureWrapMode::kRepeat) == 2,
        "fixed-point nearest NPOT repeat keeps just-below-1 on the last texel");
  Check(ComputeTextureNearestRepeat(1.2F, 4, TextureWrapMode::kClampToEdge) == 3,
        "fixed-point nearest clamp");
  // NPOT linear: fract(1.0) = 0 -> 0 * 768 - 128 -> lower -1 wraps to 2,
  // the upper tap after the last texel is the first, weight 128.
  const TextureLinearAxis npot =
      ComputeTextureLinearRepeat(1.0F, 3, TextureWrapMode::kRepeat);
  Check(npot.lower == 2 && npot.upper == 0 && npot.weight == 128,
        "fixed-point linear NPOT repeat wraps both taps");
  const TextureLinearAxis pot =
      ComputeTextureLinearRepeat(0.5F, 4, TextureWrapMode::kRepeat);
  Check(pot.lower == 1 && pot.upper == 2 && pot.weight == 128,
        "fixed-point linear: 0.5 on four texels is taps 1,2 at weight 128");
  // 0 + RNE(128/256) = 0 (quotient 0 is even); 0 + RNE(384/256) = 2
  // (quotient 1 is odd).
  Check(LerpTextureUnorm8(0, 1, 128) == 0 && LerpTextureUnorm8(0, 3, 128) == 2,
        "u8 lerp rounds a half to even");
  Check(LerpTextureUnorm8(200, 100, 64) == 175, "u8 lerp: 200 + (-100*64/256)");
}

void CheckFloatAxes() {
  const TextureFloatAxis pot =
      ComputeTextureFloatLinear(0.5F, 4, TextureWrapMode::kRepeat);
  Check(pot.lower == 1 && pot.upper == 2 && Near(pot.weight, 0.5F),
        "float linear repeat: 0.5 * 4 - 0.5 = 1.5");
  const TextureFloatAxis pot_edge =
      ComputeTextureFloatLinear(0.0F, 4, TextureWrapMode::kRepeat);
  Check(pot_edge.lower == 3 && pot_edge.upper == 0 && Near(pot_edge.weight, 0.5F),
        "float linear repeat: 0 is halfway between the last and first texel");
  const TextureFloatAxis npot =
      ComputeTextureFloatLinear(1.0F, 3, TextureWrapMode::kRepeat);
  Check(npot.lower == 2 && npot.upper == 0 && Near(npot.weight, 0.5F),
        "float linear NPOT repeat wraps both taps");
  const TextureFloatAxis clamp_low =
      ComputeTextureFloatLinear(0.0F, 4, TextureWrapMode::kClampToEdge);
  Check(clamp_low.lower == 0 && clamp_low.upper == 1 && Near(clamp_low.weight, 0.0F),
        "float linear clamp: the centred coordinate never goes below 0");
  const TextureFloatAxis clamp_high =
      ComputeTextureFloatLinear(1.0F, 4, TextureWrapMode::kClampToEdge);
  Check(clamp_high.lower == 3 && clamp_high.upper == 3 && Near(clamp_high.weight, 0.5F),
        "float linear clamp: the upper tap stops at the last texel");
  // mirror(1.25): 0.625 - nearbyint(0.625) = -0.375, doubled and absolute
  // 0.75; 0.75 * 4 - 0.5 = 2.5.
  const TextureFloatAxis mirror =
      ComputeTextureFloatLinear(1.25F, 4, TextureWrapMode::kMirroredRepeat);
  Check(mirror.lower == 2 && mirror.upper == 3 && Near(mirror.weight, 0.5F),
        "float linear mirror reflects the odd interval");
  Check(ComputeTextureFloatNearest(1.25F, 4, TextureWrapMode::kMirroredRepeat) == 3,
        "float nearest mirror: 0.75 * 4");
  Check(ComputeTextureFloatNearest(1.0F, 4, TextureWrapMode::kMirroredRepeat) == 3,
        "float nearest mirror: 1.0 reflects onto the last texel");
  Check(ComputeTextureFloatNearest(-0.125F, 4, TextureWrapMode::kRepeat) == 3,
        "float nearest repeat wraps a negative coordinate");
  Check(ComputeTextureFloatNearest(1.0F, 3, TextureWrapMode::kRepeat) == 0 &&
            ComputeTextureFloatNearest(std::nextafter(1.0F, 0.0F), 3,
                                       TextureWrapMode::kRepeat) == 2,
        "float nearest NPOT repeat: 1.0 wraps, just below stays on the last texel");
  Check(ComputeTextureFloatNearest(1.2F, 4, TextureWrapMode::kClampToEdge) == 3,
        "float nearest clamp");
  Check(Near(LerpTextureFloat(0.25F, 0.75F, 0.5F), 0.5F), "float lerp");

  const std::array<std::uint8_t, 8> white = {255, 255, 255, 255, 0, 0, 0, 0};
  const std::array<std::uint8_t, 8> mid = {128, 0, 0, 128, 0, 0, 0, 0};
  Check(Near(DecodeTexelToFloat(TextureFormat::kRgba8Srgb, white)[0], 1.0F),
        "sRGB 255 is linear 1");
  Check(DecodeTexelToFloat(TextureFormat::kRgba8Srgb, mid)[0] < 0.25F &&
            Near(DecodeTexelToFloat(TextureFormat::kRgba8Srgb, mid)[3], 128.0F / 255.0F),
        "sRGB 128 is below linear 0.25, alpha stays linear");
  Check(Near(DecodeTexelToFloat(TextureFormat::kRgbx8Unorm, mid)[3], 1.0F),
        "RGBX alpha is one");
  bool threw = false;
  try {
    (void)DecodeTexelToFloat(TextureFormat::kZ32Unorm, mid);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  Check(threw, "depth is not decoded as colour");
}

// The packed / wide colour formats, each expected value worked from the bit
// layout GL defines -- not from any captured texel.
void CheckPackedFormatDecode() {
  Check(TextureBytesPerTexel(TextureFormat::kRgb565Unorm) == 2 &&
            TextureBytesPerTexel(TextureFormat::kRgba16Float) == 8 &&
            TextureBytesPerTexel(TextureFormat::kRgb10A2Unorm) == 4 &&
            TextureBytesPerTexel(TextureFormat::kBgr10A2Unorm) == 4 &&
            TextureBytesPerTexel(TextureFormat::kR11fG11fB10f) == 4,
        "per-texel byte widths");

  // RGB565: 0xF800 = red 31/31, green 0, blue 0; little-endian bytes 0x00,0xF8.
  const std::array<std::uint8_t, 8> red565 = {0x00, 0xF8, 0, 0, 0, 0, 0, 0};
  const std::array<float, 4> r565 =
      DecodeTexelToFloat(TextureFormat::kRgb565Unorm, red565);
  Check(Near(r565[0], 1.0F) && Near(r565[1], 0.0F) && Near(r565[2], 0.0F) &&
            Near(r565[3], 1.0F),
        "RGB565 red, alpha implied one");
  // green 63/63: 0x07E0 -> bytes 0xE0,0x07.
  const std::array<std::uint8_t, 8> g565 = {0xE0, 0x07, 0, 0, 0, 0, 0, 0};
  Check(Near(DecodeTexelToFloat(TextureFormat::kRgb565Unorm, g565)[1], 1.0F),
        "RGB565 green full scale");

  // R10G10B10A2: R = 1023/1023, A = 3/3.  value = 0xC00003FF.
  const std::array<std::uint8_t, 8> ra1010102 = {0xFF, 0x03, 0x00, 0xC0,
                                                 0, 0, 0, 0};
  const std::array<float, 4> ra =
      DecodeTexelToFloat(TextureFormat::kRgb10A2Unorm, ra1010102);
  Check(Near(ra[0], 1.0F) && Near(ra[1], 0.0F) && Near(ra[2], 0.0F) &&
            Near(ra[3], 1.0F),
        "R10G10B10A2 red and alpha full scale");
  const auto ba = DecodeTexelToFloat(TextureFormat::kBgr10A2Unorm, ra1010102);
  Check(Near(ba[0], 0.0F) && Near(ba[1], 0.0F) && Near(ba[2], 1.0F) &&
            Near(ba[3], 1.0F), "B10G10R10A2 reverses only red/blue");
  // Distinct low bits must survive rendering-to-texture; no UNORM8 expansion.
  const std::uint32_t packed = 1U | (2U << 10U) | (3U << 20U) | (2U << 30U);
  const std::array<std::uint8_t, 8> low_bits = {
      static_cast<std::uint8_t>(packed), static_cast<std::uint8_t>(packed >> 8U),
      static_cast<std::uint8_t>(packed >> 16U), static_cast<std::uint8_t>(packed >> 24U),
      0, 0, 0, 0};
  const auto rgb = DecodeTexelToFloat(TextureFormat::kRgb10A2Unorm, low_bits);
  const auto bgr = DecodeTexelToFloat(TextureFormat::kBgr10A2Unorm, low_bits);
  Check(rgb[0] == 1.0F / 1023.0F && rgb[1] == 2.0F / 1023.0F &&
        rgb[2] == 3.0F / 1023.0F && rgb[3] == 2.0F / 3.0F &&
        bgr[0] == rgb[2] && bgr[1] == rgb[1] && bgr[2] == rgb[0] && bgr[3] == rgb[3],
        "packed 10-bit texture channels retain low bits and two-bit alpha");

  // RGBA8_SNORM: 127 -> +1, 129 (=-127) -> -1 (clamped), 0 -> 0.
  const std::array<std::uint8_t, 8> snorm = {127, 129, 0, 64, 0, 0, 0, 0};
  const std::array<float, 4> sn =
      DecodeTexelToFloat(TextureFormat::kRgba8Snorm, snorm);
  Check(Near(sn[0], 1.0F) && Near(sn[1], -1.0F) && Near(sn[2], 0.0F) &&
            Near(sn[3], 64.0F / 127.0F),
        "RGBA8_SNORM signed scale, clamped at -1");

  // RGBA16F: 0x3C00 = 1.0, 0x0000 = 0, 0x4000 = 2.0.  Little-endian halves.
  const std::array<std::uint8_t, 8> half = {0x00, 0x3C, 0x00, 0x00,
                                            0x00, 0x40, 0x00, 0x3C};
  const std::array<float, 4> hf =
      DecodeTexelToFloat(TextureFormat::kRgba16Float, half);
  Check(Near(hf[0], 1.0F) && Near(hf[1], 0.0F) && Near(hf[2], 2.0F) &&
            Near(hf[3], 1.0F),
        "RGBA16F half decode");

  // R11F_G11F_B10F: an all-zero word is (0,0,0), alpha one.  A red exponent
  // of 15 (bias) with zero mantissa is 1.0: bits 0x000003C0 in the low 11.
  const std::array<std::uint8_t, 8> one11 = {0xC0, 0x03, 0x00, 0x00,
                                             0, 0, 0, 0};
  const std::array<float, 4> f11 =
      DecodeTexelToFloat(TextureFormat::kR11fG11fB10f, one11);
  Check(Near(f11[0], 1.0F) && Near(f11[1], 0.0F) && Near(f11[2], 0.0F) &&
            Near(f11[3], 1.0F),
        "R11F_G11F_B10F red 1.0, alpha one");

  // RGB9E5: mantissa 256 with shared exponent giving scale 1/256 -> 1.0 in R.
  // exp field 15+9 = 24 gives 2^(24-15-9)=1; mantissa 1 -> 1.0.  value:
  // exponent<<27 | (0<<18)|(0<<9)|1, with exponent stored biased.
  const std::uint32_t e = 15U + 9U; // biased exponent for scale 1.0 per mantissa unit? build value
  const std::uint32_t rgb9e5_word = (e << 27) | 1U;
  const std::array<std::uint8_t, 8> e5 = {
      static_cast<std::uint8_t>(rgb9e5_word & 0xFF),
      static_cast<std::uint8_t>((rgb9e5_word >> 8) & 0xFF),
      static_cast<std::uint8_t>((rgb9e5_word >> 16) & 0xFF),
      static_cast<std::uint8_t>((rgb9e5_word >> 24) & 0xFF),
      0, 0, 0, 0};
  const std::array<float, 4> f9 =
      DecodeTexelToFloat(TextureFormat::kRgb9e5Float, e5);
  Check(Near(f9[0], 1.0F) && Near(f9[1], 0.0F) && Near(f9[2], 0.0F) &&
            Near(f9[3], 1.0F),
        "RGB9E5 shared-exponent red 1.0, alpha one");
}

void CheckIntegerFormatDecode() {
  const std::array<std::uint8_t, 16> texel = {
      0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x80,
      0x01, 0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00};
  const std::array<std::uint32_t, 4> expected = {
      UINT32_C(0xffffffff), UINT32_C(0x80000000),
      UINT32_C(0x01000001), UINT32_C(1)};
  for (const TextureFormat format : {TextureFormat::kRgba32Uint,
                                     TextureFormat::kRgba32Sint}) {
    Check(TextureBytesPerTexel(format) == 16 &&
              pvrgpu::stub::DecodeTexelToInteger(format, texel) == expected,
          "integer texels retain signed extrema and low bits beyond float precision");
  }
  bool threw = false;
  try {
    (void)pvrgpu::stub::DecodeTexelToInteger(TextureFormat::kRgba8Unorm, texel);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  Check(threw, "integer texel decode rejects a normalized color format");
}

void CheckSpatialOffsets() {
  Check(ComputeTextureNearestRepeat(-0.25F, 4, TextureWrapMode::kClampToEdge, 2) == 1,
        "spatial offset precedes clamp, not clamped index plus offset");
  Check(ComputeTextureNearestRepeat(0.0F, 4, TextureWrapMode::kRepeat, -1) == 3,
        "negative spatial offset wraps a POT nearest tap");
  Check(ComputeTextureNearestRepeat(0.25F, 8, TextureWrapMode::kRepeat, 1) == 3 &&
        ComputeTextureNearestRepeat(0.25F, 4, TextureWrapMode::kRepeat, 1) == 2,
        "one spatial offset is one texel at each selected mip, not base-level UV");
  Check(ComputeTextureFloatNearest(0.25F, 8, TextureWrapMode::kRepeat, 1) == 3 &&
        ComputeTextureFloatNearest(0.25F, 4, TextureWrapMode::kRepeat, 1) == 2,
        "float nearest applies offset separately for both mip extents");
  Check(ComputeTextureFloatNearest(1.125F, 4, TextureWrapMode::kMirroredRepeat, 1) == 2,
        "offset is applied before mirrored wrapping");
  const auto fixed = ComputeTextureLinearRepeat(0.0F, 4, TextureWrapMode::kRepeat, 0.5F, 1);
  Check(fixed.lower == 0 && fixed.upper == 1 && fixed.weight == 128,
        "fixed linear moves both taps and preserves half-texel weight");
  const auto fixed_npot = ComputeTextureLinearRepeat(0.25F, 3, TextureWrapMode::kRepeat, 0.5F, -1);
  Check(fixed_npot.lower == 2 && fixed_npot.upper == 0 && fixed_npot.weight == 64,
        "fixed NPOT repeat accepts signed texel offset");
  const auto floating = ComputeTextureFloatLinear(0.0F, 4, TextureWrapMode::kClampToEdge, 1);
  Check(floating.lower == 0 && floating.upper == 1 && Near(floating.weight, 0.5F),
        "float linear offset is applied before clamp");
  const auto float_npot = ComputeTextureFloatLinear(0.25F, 3, TextureWrapMode::kRepeat, -1);
  Check(float_npot.lower == 2 && float_npot.upper == 0 && Near(float_npot.weight, 0.25F),
        "float NPOT repeat signed offset preserves filter weight");
  for (unsigned extent : {1U, 3U, 4U, 8U}) {
    for (int offset = -7; offset <= 7; ++offset) {
      const auto axis = ComputeTextureFloatLinear(0.37F, extent, TextureWrapMode::kRepeat, offset);
      const auto fixed_axis = ComputeTextureLinearRepeat(0.37F, extent, TextureWrapMode::kRepeat, 0.5F, offset);
      Check(axis.lower < extent && axis.upper < extent && axis.weight >= 0 && axis.weight <= 1,
            "float offset taps remain in mip allocation");
      Check(fixed_axis.lower < extent && fixed_axis.upper < extent && fixed_axis.weight <= 255,
            "fixed offset taps remain in mip allocation");
    }
  }
}

} // namespace

int main() {
  CheckLog2AndLod();
  CheckLevelSelection();
  CheckDatapathSelection();
  CheckFixedPointAxes();
  CheckFloatAxes();
  CheckSpatialOffsets();
  CheckPackedFormatDecode();
  CheckIntegerFormatDecode();
  if (failures != 0) {
    std::cerr << failures << " texture filter check(s) failed\n";
    return 1;
  }
  std::cout << "texture filter checks passed\n";
  return 0;
}
