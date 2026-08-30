// Exact pinned GLBench Fill.Texture nearest/bilinear/trilinear fixture
// implementation.
// The texel generator mirrors SetupTexture(9); descriptor packing follows
// public Rogue TEXSTATE_IMAGE/STRIDE_IMAGE/SAMPLER bit positions from Mesa
// csbgen. The shader is identical for all three gates; live scale and sampler
// state select point, four-tap linear, or mip-linear filtering.
#include "common/glbench_texture_fixture.h"

#include <limits>
#include <stdexcept>

namespace pvrgpu::stub {
namespace {

constexpr std::uint64_t Bits(std::uint64_t value, unsigned first,
                             unsigned last) {
  const unsigned width = last - first + 1U;
  const std::uint64_t mask =
      width == 64U ? ~UINT64_C(0) : ((UINT64_C(1) << width) - 1U);
  return (value & mask) << first;
}

void StoreU64(std::array<std::uint32_t, kFillTexNearestSharedDwordCount> &dst,
              std::size_t dword, std::uint64_t value) {
  dst.at(dword) = static_cast<std::uint32_t>(value);
  dst.at(dword + 1U) = static_cast<std::uint32_t>(value >> 32U);
}

// These are the exact public Rogue SAMPLER_WORD0 literals used by the three
// GLBench calls. Nearest and bilinear remain LOD0-clamped. The trilinear case
// encodes GL_LINEAR_MIPMAP_LINEAR: linear mag/min, mipfilter=1, minlod=0 and
// the public Rogue full U4.6 max-LOD clamp (959).
inline constexpr std::uint64_t kNearestSamplerWord0 =
    UINT64_C(0x0000000000000fff);
inline constexpr std::uint64_t kBilinearSamplerWord0 =
    UINT64_C(0x0000005000000fff);
inline constexpr std::uint64_t kTrilinearSamplerWord0 =
    UINT64_C(0x00000151df800fff);
static_assert(kNearestSamplerWord0 == Bits(4095U, 0, 12));
static_assert(kBilinearSamplerWord0 ==
              (Bits(4095U, 0, 12) | Bits(1U, 36, 37) |
               Bits(1U, 38, 39)));
static_assert(kTrilinearSamplerWord0 ==
              (Bits(4095U, 0, 12) | Bits(959U, 23, 32) |
               Bits(1U, 36, 37) | Bits(1U, 38, 39) |
               Bits(1U, 40, 40)));
inline constexpr std::uint32_t kScaleOneBits = UINT32_C(0x3f800000);
inline constexpr std::uint32_t kScaleTrilinearLinear01Bits =
    UINT32_C(0x3f6ed917); // IEEE-754 binary32 0.933f.
inline constexpr std::uint32_t kScaleTrilinearLinear04Bits =
    UINT32_C(0x3f420c4a); // IEEE-754 binary32 0.758f.
inline constexpr std::uint32_t kScaleTrilinearLinear05Bits =
    UINT32_C(0x3f350481); // IEEE-754 binary32 0.7071f.

struct TextureFixtureConfig {
  std::uint64_t sampler_word0 = kNearestSamplerWord0;
  std::uint32_t vertex_scale_bits = kScaleOneBits;
  TextureFilter min_filter = TextureFilter::kNearest;
  TextureFilter mag_filter = TextureFilter::kNearest;
  TextureFilter mip_filter = TextureFilter::kNearest;
};

TextureFixtureConfig ConfigForCase(FunctionalCase functional_case) {
  switch (functional_case) {
  case FunctionalCase::kFillTexNearest:
    return {};
  case FunctionalCase::kFillTexBilinear:
    return {kBilinearSamplerWord0, kScaleOneBits, TextureFilter::kLinear,
            TextureFilter::kLinear, TextureFilter::kNearest};
  case FunctionalCase::kFillTexTrilinearLinear01:
    return {kTrilinearSamplerWord0, kScaleTrilinearLinear01Bits,
            TextureFilter::kLinear, TextureFilter::kLinear,
            TextureFilter::kLinear};
  case FunctionalCase::kFillTexTrilinearLinear04:
    return {kTrilinearSamplerWord0, kScaleTrilinearLinear04Bits,
            TextureFilter::kLinear, TextureFilter::kLinear,
            TextureFilter::kLinear};
  case FunctionalCase::kFillTexTrilinearLinear05:
    return {kTrilinearSamplerWord0, kScaleTrilinearLinear05Bits,
            TextureFilter::kLinear, TextureFilter::kLinear,
            TextureFilter::kLinear};
  default:
    throw std::invalid_argument("unsupported GLBench texture case");
  }
}

} // namespace

GlbenchFillTextureFixture
MakeGlbenchFillTextureFixture(FunctionalCase functional_case) {
  const TextureFixtureConfig config = ConfigForCase(functional_case);
  GlbenchFillTextureFixture fixture;
  fixture.positions = {
      -1.0F, -1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F, 1.0F,
  };
  fixture.texture_coordinates = {
      0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F,
  };
  fixture.vertex_scale_bits = config.vertex_scale_bits;

  fixture.resource.gpu_address = kGlbenchTextureGpuAddress;
  fixture.resource.mip_count = kMaximumTextureMipLevels;
  fixture.resource.format = TextureFormat::kRgba8Unorm;
  fixture.resource.layout = TextureLayout::kLinear;
  std::uint32_t size = 512;
  for (std::uint32_t level = 0; level < kMaximumTextureMipLevels; ++level) {
    TextureMipLevel &mip = fixture.resource.mip[level];
    mip.width = size;
    mip.height = size;
    mip.row_pitch_bytes = size * 4U;
    if (fixture.texture_bytes.size() >
        std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("GLBench texture mip offset overflow");
    }
    mip.offset_bytes =
        static_cast<std::uint32_t>(fixture.texture_bytes.size());
    for (std::uint32_t row = 0; row < size; ++row) {
      for (std::uint32_t column = 0; column < size; ++column) {
        const std::uint8_t pattern = static_cast<std::uint8_t>(
            (row ^ column) << level);
        fixture.texture_bytes.push_back(level % 3U != 0U ? pattern : 0U);
        fixture.texture_bytes.push_back(level % 3U != 1U ? pattern : 0U);
        fixture.texture_bytes.push_back(level % 3U != 2U ? pattern : 0U);
        fixture.texture_bytes.push_back(255U);
      }
    }
    if (size == 1U) {
      const std::size_t last = fixture.texture_bytes.size() - 4U;
      fixture.texture_bytes[last + 0U] = 255U;
      fixture.texture_bytes[last + 1U] = 255U;
      fixture.texture_bytes[last + 2U] = 255U;
      fixture.texture_bytes[last + 3U] = 255U;
    }
    size >>= 1U;
  }
  if (fixture.texture_bytes.size() >
      std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("GLBench texture allocation overflow");
  }
  fixture.resource.byte_size =
      static_cast<std::uint32_t>(fixture.texture_bytes.size());

  // IMAGE_WORD0: U8U8U8U8, 512x512, identity swizzle, linear stride image.
  const std::uint64_t image_word0 =
      Bits(4U, 0, 2) | Bits(3U, 5, 7) | Bits(2U, 8, 10) |
      Bits(1U, 11, 13) | Bits(0U, 14, 16) | Bits(12U, 27, 33) |
      Bits(511U, 34, 47) | Bits(511U, 48, 61);
  StoreU64(fixture.fragment_shared, 0, image_word0);

  // STRIDE_IMAGE_WORD1: address bits [2..39] occupy [16..53], mipmapped,
  // 2048-byte row stride encoded as stride-1, ten uploaded levels.
  const std::uint64_t image_word1 =
      Bits((kGlbenchTextureGpuAddress >> 2U), 16, 53) |
      Bits(1U, 15, 15) | Bits(2047U, 0, 14) | Bits(10U, 60, 63);
  StoreU64(fixture.fragment_shared, 2, image_word1);

  // Dwords 4..7 are the public image meta area. The texture is non-arrayed,
  // non-buffer, z-slice zero. PCK info is not read by this FCNORM sample.
  fixture.fragment_shared[4] = fixture.resource.byte_size;

  // SAMPLER_WORD0 is selected from the exact raw public-Rogue literals above.
  // Nearest/bilinear retain their byte-identical LOD0 descriptors, while
  // trilinear exposes the complete uploaded mip range to implicit-LOD SMP.
  const std::uint64_t sampler_word0 = config.sampler_word0;
  StoreU64(fixture.fragment_shared, 8, sampler_word0);
  StoreU64(fixture.fragment_shared, 10, 0U);
  // Sampler meta dwords 12..15 remain zero (compare NEVER/reserved). Gather
  // state 16..19 uses the same normalized repeat state and linear filtering;
  // it is present because the public combined descriptor ABI is 20 dwords.
  const std::uint64_t gather_word0 =
      sampler_word0 | Bits(1U, 36, 37) | Bits(1U, 38, 39);
  StoreU64(fixture.fragment_shared, 16, gather_word0);
  StoreU64(fixture.fragment_shared, 18, 0U);
  fixture.sampler.min_filter = config.min_filter;
  fixture.sampler.mag_filter = config.mag_filter;
  fixture.sampler.mip_filter = config.mip_filter;
  return fixture;
}

GlbenchFillTextureFixture MakeGlbenchFillTextureFixture(TextureFilter filter) {
  if (filter == TextureFilter::kNearest)
    return MakeGlbenchFillTextureFixture(FunctionalCase::kFillTexNearest);
  if (filter == TextureFilter::kLinear)
    return MakeGlbenchFillTextureFixture(FunctionalCase::kFillTexBilinear);
  throw std::invalid_argument("unsupported GLBench texture filter");
}

} // namespace pvrgpu::stub
