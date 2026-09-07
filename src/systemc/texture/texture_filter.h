// The sampling algorithms of the reference TPU, kept apart from the
// TextureUnit module so that they can be exercised without SystemC and read
// as one decision structure.
//
// That structure follows llvmpipe's software sampler
// (src/gallium/auxiliary/gallivm/lp_bld_sample*.c): compute one LOD per
// quad, decide from it whether the fragment is minified, pick the image
// filter and mip mode from that, select the level or level pair, then filter
// inside each level on one of two datapaths -- an 8-bit fixed-point one for
// 8-bit unorm images, a binary32 one for everything else.  Every step is
// something a texture unit does in hardware: an exponent-field log2, an
// integer shift, a fixed-point lerp.  Function comments name the llvmpipe
// routine each step implements; no output of that sampler is stored here --
// the texels, the LOD and the weights are all computed from the descriptor
// and the coordinates the shader supplied.
#pragma once

#include "common/functional_types.h"

#include <array>
#include <cstdint>

namespace pvrgpu::stub {

// Decoded subset of the public Rogue STRIDE image descriptor which the
// selected reference TPU supports.  Unsupported/reserved encodings fail in
// DecodeRogueTextureImageDescriptor rather than falling back to parallel
// software metadata.
struct RogueTextureImageDescriptor {
  std::uint64_t gpu_address = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t row_pitch_bytes = 0;
  std::uint8_t mip_count = 0;
  std::uint8_t sample_count = 1;
  TextureFormat format = TextureFormat::kRgba8Unorm;
  TextureLayout layout = TextureLayout::kLinear;
};

// Decoded subset of public Rogue SAMPLER_WORD0/1.  The raw fields select
// normalized repeat addressing, image filters, mip filtering, and U4.6 LOD
// clamps.  No parallel case metadata is allowed to override these hardware
// words.
struct RogueTextureSamplerDescriptor {
  TextureFilter min_filter = TextureFilter::kNearest;
  TextureFilter mag_filter = TextureFilter::kNearest;
  TextureFilter mip_filter = TextureFilter::kNearest;
  TextureWrapMode wrap_u = TextureWrapMode::kRepeat;
  TextureWrapMode wrap_v = TextureWrapMode::kRepeat;
  TextureWrapMode wrap_w = TextureWrapMode::kRepeat;
  std::uint16_t min_lod_u4_6 = 0;
  std::uint16_t max_lod_u4_6 = 0;
  std::uint8_t normalized_coordinates = 1;
};

// Which datapath filters a sample.  llvmpipe's `use_aos` rule
// (lp_build_sample_soa_code): an image whose texels are plain 8-bit unorm
// (`util_format_fits_8unorm`, which excludes sRGB) addressed with repeat or
// clamp-to-edge on every axis (`lp_is_simple_wrap_mode`) takes the 8-bit
// fixed-point filter; everything else -- sRGB, wider or float channels,
// mirrored repeat -- is filtered in binary32.
enum class TextureFilterDatapath : std::uint8_t {
  kUnorm8 = 0,
  kFloat32 = 1,
};

// How many levels a sample reads.  kNone is what a magnified fragment gets
// whatever the sampler's mip filter says (lp_build_sample_general samples it
// with the magnification filter and PIPE_TEX_MIPFILTER_NONE).
enum class TextureMipMode : std::uint8_t {
  kNone = 0,
  kNearest = 1,
  kLinear = 2,
};

// Two taps and their 8-bit weight on the fixed-point datapath.
struct TextureLinearAxis {
  std::uint32_t lower = 0;
  std::uint32_t upper = 0;
  std::uint16_t weight = 0;
};

// Two taps and their binary32 weight on the float datapath.
struct TextureFloatAxis {
  std::uint32_t lower = 0;
  std::uint32_t upper = 0;
  float weight = 0.0F;
};

// Result of lp_build_lod_selector: the level of detail one quad computed,
// already clamped to the sampler's LOD window.
struct TextureLodSelection {
  float lambda = 0.0F;
  float rho_squared = 0.0F;
  // lod_positive: lambda > 0.  GL's minification/magnification switch-over
  // constant is 0 (GL 3.1+, quoted in lp_build_lod_selector).
  bool minified = false;
  // The sampler's LOD window bites inside the image's levels.  Without an
  // active clamp llvmpipe rounds the mip-nearest level straight from the
  // exponent of rho^2 (lp_build_ilog2_sqrt); with one it rounds the clamped
  // lambda (lp_build_iround).
  bool clamp_active = false;
};

// Result of lp_build_sample_general plus lp_build_nearest_mip_level /
// lp_build_linear_mip_levels: which image filter, which level(s), what
// blend between them.
struct TextureLevelSelection {
  TextureFilter image_filter = TextureFilter::kNearest;
  TextureMipMode mip_mode = TextureMipMode::kNone;
  std::uint8_t level0 = 0;
  std::uint8_t level1 = 0;
  // The blend on each datapath: the fixed-point one converts the fractional
  // LOD with fptosi after multiplying by 256 (lp_build_sample_aos), the
  // float one uses it as it is (lp_build_sample_mipmap).
  std::uint8_t mip_weight_u8 = 0;
  float mip_weight = 0.0F;
};

// Result of the selected TPU's implicit-derivative LOD datapath for one
// quad, with the level selection that follows from it.  The four
// coordinates are ordered as the architectural 2x2 quad lanes 0,1,2,3.
struct TextureImplicitLod {
  float lambda = 0.0F;
  float mip_weight = 0.0F;
  // Retain exact f32 datapath intermediates for bounded diagnostics and unit
  // regressions; sampling still consumes only the selection below.
  float dsdx = 0.0F;
  float dtdx = 0.0F;
  float dsdy = 0.0F;
  float dtdy = 0.0F;
  float rho_squared = 0.0F;
  std::uint8_t level0 = 0;
  std::uint8_t level1 = 0;
  std::uint8_t mip_weight_u8 = 0;
  bool minified = false;
  TextureFilter image_filter = TextureFilter::kNearest;
  TextureMipMode mip_mode = TextureMipMode::kNone;
};

// lp_build_fast_log2: floor(log2 x) - 1 + x / 2^floor(log2 x), the
// piece-wise linear log2 that is exact at powers of two.  Requires x > 0.
float TextureFastLog2(float x);

// lp_build_lod_selector for LODM=NORMAL: lambda from rho^2 through
// TextureFastLog2, clamped to the sampler's U4.6 window, and the
// minification decision.  rho_squared of zero is the degenerate quad and
// selects the window's minimum.
TextureLodSelection SelectTextureLod(float rho_squared,
                                     const RogueTextureSamplerDescriptor &sampler,
                                     std::uint32_t mip_count);

// lp_build_sample_general: a minified fragment takes the minification filter
// and the sampler's mip filter, a magnified one the magnification filter on
// the base level.  Mip-nearest rounds per TextureLodSelection::clamp_active;
// mip-linear takes floor(lambda) and the next level with the fraction as
// weight.  A mip-linear sampler always names two levels: at either end of
// the chain both are the same level with a zero weight, which is
// lp_build_linear_mip_levels' clamped result by another route and keeps the
// unit's texel traffic independent of where lambda fell.
TextureLevelSelection SelectTextureLevels(
    const TextureLodSelection &lod,
    const RogueTextureSamplerDescriptor &sampler, std::uint32_t mip_count);

// Texel fetches one sample with this selection issues.
std::uint32_t TextureLevelTaps(const TextureLevelSelection &levels);

TextureFilterDatapath SelectTextureFilterDatapath(
    TextureFormat format, const RogueTextureSamplerDescriptor &sampler);

// Integer tap addressing shared by both datapaths.  Clamp-to-border is not
// supported and throws.
std::uint32_t WrapTexelIndex(std::int64_t integer, std::uint32_t extent,
                             TextureWrapMode wrap);

// --- 8-bit fixed-point datapath (lp_bld_sample_aos.c) ---

// lp_build_sample_image_nearest: floor(coord * extent), then the integer
// wrap of lp_build_sample_wrap_nearest_int.
std::uint32_t ComputeTextureNearestRepeat(float coordinate,
                                          std::uint32_t extent,
                                          TextureWrapMode wrap);

// lp_build_sample_image_linear: iround(coord * extent * 256) - 128, the tap
// is the quotient by 256 and the weight the remainder, then the integer wrap
// of lp_build_sample_wrap_linear_int.  A non-power-of-two repeat starts from
// the fractional coordinate (lp_build_coord_repeat_npot_linear_int).
TextureLinearAxis ComputeTextureLinearRepeat(
    float coordinate, std::uint32_t extent,
    TextureWrapMode wrap = TextureWrapMode::kRepeat,
    float round_threshold = 0.5F);

// lp_build_lerp on an 8-bit normalized type: first + RNE(weight * (second -
// first) / 256).
std::uint8_t LerpTextureUnorm8(std::uint8_t first, std::uint8_t second,
                               std::uint16_t weight);

// --- binary32 datapath (lp_bld_sample_soa.c) ---

// lp_build_sample_wrap_nearest.
std::uint32_t ComputeTextureFloatNearest(float coordinate,
                                         std::uint32_t extent,
                                         TextureWrapMode wrap);

// lp_build_sample_wrap_linear: coord * extent - 0.5, floor for the lower tap
// and the fraction for the weight, then the per-mode wrap of both taps.
TextureFloatAxis ComputeTextureFloatLinear(float coordinate,
                                           std::uint32_t extent,
                                           TextureWrapMode wrap);

// lp_build_lerp_simple on a float type: first + weight * (second - first).
float LerpTextureFloat(float first, float second, float weight);

// Bytes one texel of this format occupies in memory.  ASTC has no per-texel
// width (it stores 128-bit blocks) and throws; depth formats are four.
std::uint32_t TextureBytesPerTexel(TextureFormat format);

// The stored texel as the float datapath sees it, from up to eight raw
// little-endian bytes: unorm and snorm channels scaled to [0, 1] / [-1, 1],
// packed 565 / 10-10-10-2 fields unpacked, the two packed-float and the
// shared-exponent formats expanded, IEEE halves widened, sRGB colour channels
// through the transfer function with alpha left linear, RGBX and the
// three-channel formats' alpha forced to one.  Depth formats are not colour
// and throw.
std::array<float, 4> DecodeTexelToFloat(TextureFormat format,
                                        const std::array<std::uint8_t, 8> &texel);
std::array<float, 4> DecodeTexelToFloat(
    TextureFormat format, const std::array<std::uint8_t, 16> &texel);

// Integer nearest filtering returns all 128 stored bits unchanged. Signed
// and unsigned shader result types share the raw DWORD response contract.
std::array<std::uint32_t, 4> DecodeTexelToInteger(
    TextureFormat format, const std::array<std::uint8_t, 16> &texel);

} // namespace pvrgpu::stub
