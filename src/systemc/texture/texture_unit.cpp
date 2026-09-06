// TextureUnit：模擬 PowerVR TPU 的 texture sampling／filtering 階段。
// TPU（Texture Processing Unit，紋理處理單元）負責解析 SMP descriptor、
// normalized-coordinate repeat addressing、2x2 quad implicit derivatives、
// mip LOD selection，以及 nearest、四 tap bilinear 或八 tap trilinear
// filtering。它先把 texture allocation 預置於 DRAM，再為每個 USC shader
// lane 透過 TPU -> TCU -> SLC -> DRAM 的 request/response FIFO 取回每一個
// 真實 texel；回應僅供 USC continuation 在 WDF 後完成 PIXOUT。非紋理
// cases 無 request 地通過 Run。Bulk payload 全在 MemoryPool，延遲由
// event-driven wait 表示。
#include "texture/texture_unit.h"

#include "texture/astc_decoder.h"

#include "common/functional_types.h"
#include "common/pipeline_state.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pvrgpu::stub {

MemoryAccessStats MaterializeSequenceColorMipChain(
    GpuMemorySystem &memory, const DriverPcoSampledTexture &texture,
    std::uint64_t attachment_address) {
  if (texture.source != DriverPcoTextureSource::kPreviousColorAttachment ||
      texture.producer_command_index >=
          kDriverPcoMaximumNestedSequenceCommands ||
      texture.format != "PIPE_FORMAT_R8G8B8A8_UNORM" ||
      texture.mip_count == 0 ||
      texture.mip_count > kDriverPcoMaximumTextureMipLevels ||
      texture.declared_bytes_size == 0 ||
      texture.declared_bytes_size > kDriverPcoSequenceAttachmentStride ||
      !texture.bytes.empty()) {
    throw std::runtime_error(
        "TextureUnit sequence color mip metadata is invalid");
  }
  if (attachment_address == 0) {
    attachment_address =
        kDriverPcoSequenceColorAddressBase +
        static_cast<std::uint64_t>(texture.producer_command_index) *
            kDriverPcoSequenceAttachmentStride;
  }
  if (attachment_address < kDriverPcoSequenceColorAddressBase ||
      (attachment_address - kDriverPcoSequenceColorAddressBase) %
              kDriverPcoSequenceAttachmentStride !=
          0) {
    throw std::runtime_error(
        "TextureUnit sequence color attachment address is invalid");
  }
  std::uint64_t expected_offset = 0;
  std::uint32_t expected_width = texture.mip[0].width;
  std::uint32_t expected_height = texture.mip[0].height;
  if (expected_width == 0 || expected_height == 0)
    throw std::runtime_error("TextureUnit sequence color base mip is empty");
  for (std::uint32_t level = 0; level < texture.mip_count; ++level) {
    const DriverPcoTextureMipLayout &mip = texture.mip[level];
    const std::uint64_t row_pitch =
        static_cast<std::uint64_t>(expected_width) * 4U;
    const std::uint64_t level_bytes = row_pitch * expected_height;
    if (mip.width != expected_width || mip.height != expected_height ||
        mip.row_pitch_bytes != row_pitch ||
        mip.offset_bytes != expected_offset ||
        expected_offset > texture.declared_bytes_size ||
        level_bytes > texture.declared_bytes_size - expected_offset) {
      throw std::runtime_error(
          "TextureUnit sequence color mip layout is invalid");
    }
    expected_offset += level_bytes;
    expected_width = std::max(expected_width >> 1U, 1U);
    expected_height = std::max(expected_height >> 1U, 1U);
  }
  for (std::size_t level = texture.mip_count; level < texture.mip.size();
       ++level) {
    const DriverPcoTextureMipLayout &mip = texture.mip[level];
    if (mip.width != 0 || mip.height != 0 || mip.row_pitch_bytes != 0 ||
        mip.offset_bytes != 0) {
      throw std::runtime_error(
          "TextureUnit sequence color unused mip metadata is nonzero");
    }
  }
  if (expected_offset != texture.declared_bytes_size)
    throw std::runtime_error(
        "TextureUnit sequence color mip allocation size is invalid");

  const DriverPcoTextureMipLayout &base = texture.mip[0];
  const std::size_t base_bytes =
      static_cast<std::size_t>(base.row_pitch_bytes) * base.height;
  MemoryReadResult source = memory.Readback(
      attachment_address, base_bytes, MemoryClient::kFramebufferReadback);
  if (source.data.size() != base_bytes)
    throw std::runtime_error(
        "TextureUnit sequence color attachment readback is truncated");
  // A one-level previous-color view aliases the producer attachment exactly;
  // it needs validation and modeled readback residency, but no synthetic
  // write.  Multi-level views continue below and derive every lower level
  // from the real producer attachment in unified memory.
  if (texture.mip_count == 1)
    return source.stats;
  std::vector<std::uint8_t> chain(
      static_cast<std::size_t>(texture.declared_bytes_size), 0);
  std::copy(source.data.begin(), source.data.end(), chain.begin());

  for (std::uint32_t level = 1; level < texture.mip_count; ++level) {
    const DriverPcoTextureMipLayout &previous = texture.mip[level - 1U];
    const DriverPcoTextureMipLayout &current = texture.mip[level];
    for (std::uint32_t y = 0; y < current.height; ++y) {
      const float v =
          (static_cast<float>(y) + 0.5F) /
          static_cast<float>(current.height);
      const TextureLinearAxis y_axis = ComputeTextureLinearRepeat(
          v, previous.height, TextureWrapMode::kClampToEdge);
      for (std::uint32_t x = 0; x < current.width; ++x) {
        const float u =
            (static_cast<float>(x) + 0.5F) /
            static_cast<float>(current.width);
        const TextureLinearAxis x_axis = ComputeTextureLinearRepeat(
            u, previous.width, TextureWrapMode::kClampToEdge);
        const std::size_t destination =
            static_cast<std::size_t>(current.offset_bytes) +
            static_cast<std::size_t>(y) * current.row_pitch_bytes + x * 4U;
        for (std::size_t component = 0; component < 4; ++component) {
          const auto texel = [&](std::uint32_t sx, std::uint32_t sy) {
            return chain[static_cast<std::size_t>(previous.offset_bytes) +
                         static_cast<std::size_t>(sy) *
                             previous.row_pitch_bytes +
                         static_cast<std::size_t>(sx) * 4U + component];
          };
          // glGenerateMipmap is a normalized bilinear blit between each pair
          // of levels.  The selected U8 filter datapath quantizes the source
          // coordinate to eight fractional bits, rounds each horizontal
          // interpolation, then rounds the vertical interpolation.  This is
          // observably different both from a direct four-texel average at
          // half ties and from dropping the last row/column at odd extents.
          const std::uint8_t lower = LerpTextureUnorm8(
              texel(x_axis.lower, y_axis.lower),
              texel(x_axis.upper, y_axis.lower), x_axis.weight);
          const std::uint8_t upper = LerpTextureUnorm8(
              texel(x_axis.lower, y_axis.upper),
              texel(x_axis.upper, y_axis.upper), x_axis.weight);
          chain[destination + component] =
              LerpTextureUnorm8(lower, upper, y_axis.weight);
        }
      }
    }
  }

  MemoryAccessStats stats = source.stats;
  stats += memory.Write(attachment_address, chain.data(), chain.size(),
                        MemoryClient::kTextureMipmap);
  MemoryReadResult committed = memory.Readback(
      attachment_address, chain.size(), MemoryClient::kFramebufferReadback);
  stats += committed.stats;
  if (committed.data != chain)
    throw std::runtime_error(
        "TextureUnit sequence color mip DRAM commit mismatch");
  return stats;
}

namespace {

std::uint32_t DebugFragmentCoordinate(const char *name,
                                      std::uint32_t fallback) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0')
    return fallback;
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value || *end != '\0' ||
      parsed > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error(std::string("invalid debug coordinate in ") +
                             name);
  }
  return static_cast<std::uint32_t>(parsed);
}

float BitsFloat(std::uint32_t bits) {
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::uint64_t ReadU64(const std::array<std::uint32_t, 4> &words,
                      std::size_t first_dword) {
  return static_cast<std::uint64_t>(words.at(first_dword)) |
         (static_cast<std::uint64_t>(words.at(first_dword + 1U)) << 32U);
}

std::uint64_t ExtractBits(std::uint64_t value, unsigned first,
                          unsigned last) {
  const unsigned width = last - first + 1U;
  const std::uint64_t mask =
      width == 64U ? ~UINT64_C(0) : ((UINT64_C(1) << width) - 1U);
  return (value >> first) & mask;
}

TextureFilter DecodeFilter(std::uint64_t encoded, const char *field) {
  switch (encoded) {
  case 0:
    return TextureFilter::kNearest;
  case 1:
    return TextureFilter::kLinear;
  default:
    throw std::runtime_error(std::string("TextureUnit unsupported raw ") +
                             field + " filter");
  }
}

TextureWrapMode DecodeWrapMode(std::uint64_t encoded) {
  switch (encoded) {
  case 0:
    return TextureWrapMode::kRepeat;
  case 1:
    return TextureWrapMode::kMirroredRepeat;
  case 2:
    return TextureWrapMode::kClampToEdge;
  case 4:
    return TextureWrapMode::kClampToBorder;
  default:
    throw std::runtime_error("TextureUnit unsupported wrap mode encoding");
  }
}

} // namespace

RogueTextureImageDescriptor DecodeRogueTextureImageDescriptor(
    const std::array<std::uint32_t, 4> &words, bool compressed) {
  const std::uint64_t word0 = ReadU64(words, 0);
  const std::uint64_t word1 = ReadU64(words, 2);

  // Public Rogue IMAGE_WORD0.  The selected paths are linear-stride,
  // non-gamma U8U8U8U8 with identity RGBA/RGB1 or U32 with XXX1 swizzle for
  // Z32_UNORM FCNORM sampling.  Every other format/swizzle fails closed.
  const std::uint64_t alpha_swizzle = ExtractBits(word0, 5, 7);
  const std::uint64_t blue_swizzle = ExtractBits(word0, 8, 10);
  const std::uint64_t green_swizzle = ExtractBits(word0, 11, 13);
  const std::uint64_t red_swizzle = ExtractBits(word0, 14, 16);
  const std::uint64_t format = ExtractBits(word0, 27, 33);
  /*
   * Read through FORMAT_COMPRESSED when the command said the image is
   * compressed.  ASTC occupies values 0..13 there, one per footprint, and the
   * footprint itself travels with the resource rather than the descriptor.
   */
  const AstcBlockFootprint astc_footprint =
      compressed ? AstcFootprintForRogueFormat(
                       static_cast<std::uint32_t>(format))
                 : AstcBlockFootprint{};
  const bool astc = compressed && astc_footprint.valid();
  const bool rgba8 =
      !compressed && format == 12U && red_swizzle == 0U &&
      green_swizzle == 1U && blue_swizzle == 2U &&
      (alpha_swizzle == 3U || alpha_swizzle == 4U);
  // B8G8R8A8: the same U8U8U8U8 storage as RGBA8 with the descriptor's
  // (Z,Y,X,W) swizzle -- red reads channel 2 and blue channel 0.  The texel
  // fetch swaps the two bytes so the datapath treats it as plain RGBA8.
  const bool bgra8 =
      !compressed && format == 12U && red_swizzle == 2U &&
      green_swizzle == 1U && blue_swizzle == 0U && alpha_swizzle == 3U;
  const bool z32_unorm =
      format == 24U && red_swizzle == 0U && green_swizzle == 0U &&
      blue_swizzle == 0U && alpha_swizzle == 4U;
  // ST8U24 is the combined depth/stencil texel: stencil in bits 24..31,
  // depth in bits 0..23.  GL hands the depth to red and zeroes the rest, so
  // SRC_ZERO (5) selects green and blue and SRC_ONE (4) selects alpha.
  const bool z24_unorm_s8_uint =
      format == 22U && red_swizzle == 0U && green_swizzle == 5U &&
      blue_swizzle == 5U && alpha_swizzle == 4U;
  // Uncompressed non-RGBA8 colour formats.  Each names a Rogue FORMAT number
  // and an identity swizzle: the three-channel formats select SRC_ONE (4) for
  // alpha, the four-channel ones select the alpha channel (3).  The FORMAT
  // number is what distinguishes an RGB1-swizzled RGB565 from an RGB1-swizzled
  // RGBX8.
  const bool identity_rgb =
      !compressed && red_swizzle == 0U && green_swizzle == 1U &&
      blue_swizzle == 2U;
  const bool rgb565 = identity_rgb && format == 5U && alpha_swizzle == 4U;
  const bool rgba8_snorm = identity_rgb && format == 13U && alpha_swizzle == 3U;
  const bool rgb10_a2 = identity_rgb && format == 14U && alpha_swizzle == 3U;
  const bool rgb9e5 = identity_rgb && format == 26U && alpha_swizzle == 4U;
  const bool r11g11b10 = identity_rgb && format == 27U && alpha_swizzle == 4U;
  const bool rgba16f = identity_rgb && format == 28U && alpha_swizzle == 3U;
  const bool packed_colour =
      rgb565 || rgba8_snorm || rgb10_a2 || rgb9e5 || r11g11b10 || rgba16f;
  /*
   * Rogue IMAGE_WORD0 bit 3 is GAMMA and bit 4 is the second half of
   * TWOCOMP_GAMMA.  Gamma on a four-channel image is sRGB, which this unit
   * decodes; two-component gamma applies to formats it does not sample, so
   * bit 4 still has to be zero.
   */
  const bool gamma = ExtractBits(word0, 3, 3) != 0U;
  // Rogue TEXTYPE: a plain linear image is STRIDE (4) and carries its row
  // pitch in STRIDE_IMAGE_WORD1; a 2D-array image is 2D (1) and carries the
  // array depth in IMAGE_WORD1's depth field instead, because a sampled array
  // needs the layer count the shader clamps to and a strided word has no room
  // for it beside the pitch.
  const std::uint64_t textype = ExtractBits(word0, 0, 2);
  const bool array_image = textype == 1U;
  if ((textype != 4U && textype != 1U) ||
      ExtractBits(word0, 4, 4) != 0U ||
      (gamma && !rgba8 && !astc) ||
      ExtractBits(word0, 17, 26) != 0U ||
      (!rgba8 && !bgra8 && !astc && !z32_unorm && !z24_unorm_s8_uint &&
       !packed_colour) ||
      ExtractBits(word0, 62, 63) != 0U) {
    throw std::runtime_error(
        "TextureUnit unsupported raw Rogue image word0");
  }

  // Public Rogue STRIDE_IMAGE_WORD1. Compression/index/tile/alpha controls
  // are unsupported and must remain zero. A one-level driver image has no
  // mipmaps-present bit; the GLBench fixtures retain their complete ten-level
  // allocation. Other valid counts remain available for future command
  // lowering without weakening the structured-layout cross-check below.
  // IMAGE_WORD1 (array, non-stride) keeps num_mip_levels in [0:3] and the
  // array depth in [4:14]; STRIDE_IMAGE_WORD1 keeps num_mip_levels in [60:63]
  // and the texel stride in [0:14].  baselevel [60:63] and the compression
  // controls stay zero in the array form.
  const std::uint64_t raw_mip_count =
      array_image ? ExtractBits(word1, 0, 3) : ExtractBits(word1, 60, 63);
  const bool mipmaps_present = ExtractBits(word1, 15, 15) != 0U;
  const bool word1_reserved_set =
      array_image ? (ExtractBits(word1, 54, 63) != 0U)
                  : (ExtractBits(word1, 54, 59) != 0U);
  if (word1_reserved_set || raw_mip_count == 0U ||
      raw_mip_count > kMaximumTextureMipLevels ||
      mipmaps_present != (raw_mip_count > 1U)) {
    throw std::runtime_error(
        "TextureUnit unsupported raw Rogue stride image word1");
  }

  RogueTextureImageDescriptor descriptor;
  descriptor.width =
      static_cast<std::uint32_t>(ExtractBits(word0, 34, 47) + 1U);
  descriptor.height =
      static_cast<std::uint32_t>(ExtractBits(word0, 48, 61) + 1U);
  const std::uint32_t encoded_stride =
      array_image
          ? descriptor.width
          : static_cast<std::uint32_t>(ExtractBits(word1, 0, 14) + 1U);
  // The colour formats' byte width drives both the stride decode and the
  // minimum-pitch floor.  ASTC's stride is measured in blocks, not texels.
  const std::uint32_t bytes_per_texel =
      astc ? 0U : rgba16f ? 8U : rgb565 ? 2U : 4U;
  /* Public STRIDE_IMAGE_WORD1 expresses stride in texels. The pinned GLBench
   * literals predate that decoder contract and encode byte stride instead.
   * Their value is at least one complete RGBA8 byte row; a new tight public
   * descriptor is exactly one width in texels, so the two accepted encodings
   * remain unambiguous without weakening arbitrary mip-count validation. */
  if (astc) {
    /*
     * A compressed image's stride is bytes, because a row of blocks is not a
     * whole number of texels: ASTC 5x4 over 256 texels is 52 blocks, which is
     * 832 bytes and no texel count at all.
     */
    descriptor.row_pitch_bytes = encoded_stride;
  } else if (encoded_stride == descriptor.width) {
    descriptor.row_pitch_bytes = encoded_stride * bytes_per_texel;
  } else if (encoded_stride >= descriptor.width * bytes_per_texel) {
    descriptor.row_pitch_bytes = encoded_stride;
  } else {
    throw std::runtime_error(
        "TextureUnit ambiguous raw Rogue stride image pitch");
  }
  descriptor.gpu_address = ExtractBits(word1, 16, 53) << 2U;
  descriptor.mip_count = static_cast<std::uint8_t>(raw_mip_count);
  descriptor.format =
      astc ? (gamma ? TextureFormat::kAstcLdrSrgb : TextureFormat::kAstcLdr)
      : z24_unorm_s8_uint ? TextureFormat::kZ24UnormS8Uint
      : z32_unorm         ? TextureFormat::kZ32Unorm
      : rgb565            ? TextureFormat::kRgb565Unorm
      : rgba8_snorm       ? TextureFormat::kRgba8Snorm
      : rgb10_a2          ? TextureFormat::kRgb10A2Unorm
      : r11g11b10         ? TextureFormat::kR11fG11fB10f
      : rgb9e5            ? TextureFormat::kRgb9e5Float
      : rgba16f           ? TextureFormat::kRgba16Float
      : bgra8             ? TextureFormat::kBgra8Unorm
      : gamma             ? TextureFormat::kRgba8Srgb
      : alpha_swizzle == 4U ? TextureFormat::kRgbx8Unorm
                            : TextureFormat::kRgba8Unorm;
  const std::uint32_t minimum_row_pitch =
      astc ? ((descriptor.width + astc_footprint.width - 1U) /
              astc_footprint.width) * 16U
           : descriptor.width * bytes_per_texel;
  if (descriptor.gpu_address == 0 ||
      descriptor.row_pitch_bytes < minimum_row_pitch) {
    throw std::runtime_error("TextureUnit invalid raw Rogue image layout");
  }
  return descriptor;
}

RogueTextureSamplerDescriptor DecodeRogueTextureSamplerDescriptor(
    const std::array<std::uint32_t, 4> &words) {
  const std::uint64_t word0 = ReadU64(words, 0);
  const std::uint64_t word1 = ReadU64(words, 2);
  RogueTextureSamplerDescriptor descriptor;
  descriptor.min_filter = DecodeFilter(ExtractBits(word0, 38, 39), "min");
  descriptor.mag_filter = DecodeFilter(ExtractBits(word0, 36, 37), "mag");
  descriptor.mip_filter = ExtractBits(word0, 40, 40) == 0U
                              ? TextureFilter::kNearest
                              : TextureFilter::kLinear;
  descriptor.min_lod_u4_6 =
      static_cast<std::uint16_t>(ExtractBits(word0, 13, 22));
  descriptor.max_lod_u4_6 =
      static_cast<std::uint16_t>(ExtractBits(word0, 23, 32));
  descriptor.normalized_coordinates =
      ExtractBits(word0, 49, 49) == 0U ? 1U : 0U;

  descriptor.wrap_u = DecodeWrapMode(ExtractBits(word0, 33, 35));
  descriptor.wrap_v = DecodeWrapMode(ExtractBits(word0, 41, 43));

  /*
   * dadjust=4095 is zero bias.  What is checked here is that the encoding is
   * one this decoder understands -- the reserved fields, the address modes,
   * normalized coordinates, and a LOD window that runs forwards.
   *
   * Which filter combinations can actually be sampled is a question about the
   * image as well as the sampler, and DriverPcoTextureDescriptorClassSupported
   * answers it.  This used to enumerate three sampler shapes here as well,
   * which disagreed with that predicate as soon as either moved: a sampler the
   * capability check admitted could still be refused by the decode, and the
   * refusal arrived as a fatal exception mid-sample.
   */
  const bool lod_window_runs_forwards =
      descriptor.min_lod_u4_6 <= descriptor.max_lod_u4_6;
  const bool supported_wrap_u =
      descriptor.wrap_u == TextureWrapMode::kRepeat ||
      descriptor.wrap_u == TextureWrapMode::kMirroredRepeat ||
      descriptor.wrap_u == TextureWrapMode::kClampToEdge;
  const bool supported_wrap_v =
      descriptor.wrap_v == TextureWrapMode::kRepeat ||
      descriptor.wrap_v == TextureWrapMode::kMirroredRepeat ||
      descriptor.wrap_v == TextureWrapMode::kClampToEdge;
  if (ExtractBits(word0, 0, 12) != 4095U ||
      !supported_wrap_u || !supported_wrap_v ||
      ExtractBits(word0, 44, 46) != 0U ||
      ExtractBits(word0, 47, 48) != 0U ||
      descriptor.normalized_coordinates != 1U ||
      ExtractBits(word0, 50, 55) != 0U ||
      ExtractBits(word0, 56, 58) != 0U ||
      ExtractBits(word0, 59, 63) != 0U || word1 != 0U ||
      !lod_window_runs_forwards) {
    std::ostringstream detail;
    detail << "TextureUnit unsupported raw Rogue sampler descriptor"
           << " (word0=0x" << std::hex << word0 << " word1=0x" << word1
           << std::dec << " min=" << static_cast<unsigned>(
                  ExtractBits(word0, 38, 39))
           << " mag=" << static_cast<unsigned>(ExtractBits(word0, 36, 37))
           << " mip=" << static_cast<unsigned>(ExtractBits(word0, 40, 40))
           << " minlod=" << descriptor.min_lod_u4_6
           << " maxlod=" << descriptor.max_lod_u4_6
           << " wrapu=" << static_cast<unsigned>(ExtractBits(word0, 33, 35))
           << " wrapv=" << static_cast<unsigned>(ExtractBits(word0, 41, 43))
           << ')';
    throw std::runtime_error(detail.str());
  }
  return descriptor;
}

bool DriverPcoTextureDescriptorClassSupported(
    const RogueTextureImageDescriptor &image,
    const RogueTextureSamplerDescriptor &sampler,
    std::uint32_t descriptor_count) {
  (void)descriptor_count;
  /*
   * What this unit can sample, stated rather than enumerated.
   *
   * The format selects the decode and the sampler selects the taps, and the
   * two do not interact, so each is checked on its own.  Level selection is
   * not a constraint at all (texture_filter.h): every mip filter selects a
   * level, every image filter runs on one of the two datapaths, and the LOD
   * window may reach past the last level -- lp_build_nearest_mip_level clamps
   * the level, and a window of 0..0.25 is how a driver says "base level only"
   * while keeping the minification decision.  Earlier versions refused a
   * window past the last level and a mipped image sampled nearest; neither
   * absence meant the unit could not do it.
   */
  const bool decodable_format =
      image.format == TextureFormat::kRgba8Unorm ||
      image.format == TextureFormat::kRgbx8Unorm ||
      image.format == TextureFormat::kBgra8Unorm ||
      image.format == TextureFormat::kRgba8Srgb ||
      image.format == TextureFormat::kAstcLdr ||
      image.format == TextureFormat::kAstcLdrSrgb ||
      image.format == TextureFormat::kZ32Unorm ||
      image.format == TextureFormat::kZ24UnormS8Uint ||
      image.format == TextureFormat::kRgb565Unorm ||
      image.format == TextureFormat::kRgb10A2Unorm ||
      image.format == TextureFormat::kRgba8Snorm ||
      image.format == TextureFormat::kRgba16Float ||
      image.format == TextureFormat::kR11fG11fB10f ||
      image.format == TextureFormat::kRgb9e5Float;
  // Z32 is one uint32 per texel with no filter datapath: nearest taps only.
  const bool depth32_nearest_only =
      image.format != TextureFormat::kZ32Unorm ||
      (sampler.min_filter == TextureFilter::kNearest &&
       sampler.mag_filter == TextureFilter::kNearest &&
       sampler.mip_filter == TextureFilter::kNearest);
  const bool window_runs_forwards =
      sampler.min_lod_u4_6 <= sampler.max_lod_u4_6;
  // An address mode the wrap arithmetic implements.
  const bool supported_wrap =
      (sampler.wrap_u == TextureWrapMode::kClampToEdge ||
       sampler.wrap_u == TextureWrapMode::kRepeat ||
       sampler.wrap_u == TextureWrapMode::kMirroredRepeat) &&
      (sampler.wrap_v == TextureWrapMode::kClampToEdge ||
       sampler.wrap_v == TextureWrapMode::kRepeat ||
       sampler.wrap_v == TextureWrapMode::kMirroredRepeat);

  return decodable_format && window_runs_forwards && supported_wrap &&
         depth32_nearest_only;
}

// The sampled depth of a combined depth/stencil texel.  The driver's clear and
// depth-write paths pack the depth into bits 0..23 and leave the stencil in
// bits 24..31, so the stencil is masked off rather than normalized with it.
constexpr std::uint32_t kSampledDepth24Maximum = 0x00ffffffU;

std::uint32_t SampledDepth24FromTexel(const std::array<std::uint8_t, 8> &texel) {
  std::uint32_t encoded = 0;
  std::memcpy(&encoded, texel.data(), sizeof(encoded));
  return encoded & kSampledDepth24Maximum;
}

// The same fixed-point filter the U8 datapath uses, widened to the 24-bit
// depth channel: first + RNE(weight * (second - first) / 256).  Filtering the
// packed word byte by byte would blend the stencil into the depth's high byte
// and produce silently wrong samples.
std::uint32_t LerpSampledDepth24(std::uint32_t first, std::uint32_t second,
                                 std::uint16_t weight) {
  if (weight > 255)
    throw std::runtime_error("TextureUnit linear weight exceeds U8 range");
  if (first > kSampledDepth24Maximum || second > kSampledDepth24Maximum)
    throw std::runtime_error("TextureUnit depth endpoint exceeds UNORM24");
  const std::int64_t product =
      static_cast<std::int64_t>(weight) *
      (static_cast<std::int64_t>(second) - static_cast<std::int64_t>(first));
  const bool negative = product < 0;
  const std::uint64_t magnitude =
      static_cast<std::uint64_t>(negative ? -product : product);
  std::uint64_t quotient = magnitude >> 8U;
  const std::uint64_t remainder = magnitude & 0xffU;
  if (remainder > 128U || (remainder == 128U && (quotient & 1U) != 0))
    ++quotient;
  const std::int64_t result =
      static_cast<std::int64_t>(first) +
      (negative ? -static_cast<std::int64_t>(quotient)
                : static_cast<std::int64_t>(quotient));
  if (result < 0 || result > static_cast<std::int64_t>(kSampledDepth24Maximum))
    throw std::runtime_error("TextureUnit linear result exceeds UNORM24");
  return static_cast<std::uint32_t>(result);
}

float SampledDepth24ToFloat(std::uint32_t depth) {
  return static_cast<float>(static_cast<double>(depth) /
                            static_cast<double>(kSampledDepth24Maximum));
}

TextureImplicitLod ComputeTextureImplicitLod(
    const std::array<std::array<float, 2>, 4> &coordinates,
    const RogueTextureImageDescriptor &image,
    const RogueTextureSamplerDescriptor &sampler) {
  if (image.width == 0 || image.height == 0 || image.mip_count == 0)
    throw std::runtime_error("TextureUnit implicit LOD state is invalid");
  for (const auto &coordinate : coordinates) {
    if (!std::isfinite(coordinate[0]) || !std::isfinite(coordinate[1]))
      throw std::runtime_error("TextureUnit implicit LOD coordinate is invalid");
  }

  // Public SMP LODM=NORMAL derives one isotropic LOD for a 2x2 quad from the
  // exact Euclidean derivative norm (lp_build_rho); the log2 and everything
  // after it are in SelectTextureLod / SelectTextureLevels.
  const float dsdx =
      (coordinates[1][0] - coordinates[0][0]) * image.width;
  const float dtdx =
      (coordinates[1][1] - coordinates[0][1]) * image.height;
  const float dsdy =
      (coordinates[2][0] - coordinates[0][0]) * image.width;
  const float dtdy =
      (coordinates[2][1] - coordinates[0][1]) * image.height;
  const float rho_x_squared = dsdx * dsdx + dtdx * dtdx;
  const float rho_y_squared = dsdy * dsdy + dtdy * dtdy;
  const float rho_squared = std::max(rho_x_squared, rho_y_squared);
  if (rho_squared < 0.0F || !std::isfinite(rho_squared))
    throw std::runtime_error("TextureUnit implicit derivative rho is invalid");

  const TextureLodSelection lod =
      SelectTextureLod(rho_squared, sampler, image.mip_count);
  const TextureLevelSelection levels =
      SelectTextureLevels(lod, sampler, image.mip_count);

  TextureImplicitLod result;
  result.lambda = lod.lambda;
  result.dsdx = dsdx;
  result.dtdx = dtdx;
  result.dsdy = dsdy;
  result.dtdy = dtdy;
  result.rho_squared = rho_squared;
  result.minified = lod.minified;
  result.image_filter = levels.image_filter;
  result.mip_mode = levels.mip_mode;
  result.level0 = levels.level0;
  result.level1 = levels.level1;
  result.mip_weight_u8 = levels.mip_weight_u8;
  result.mip_weight = levels.mip_weight;
  return result;
}

TextureUnit::TextureUnit(sc_core::sc_module_name name, MemoryPool &pool,
                         GpuMemorySystem *memory)
    : sc_module(name), pool_(pool), memory_(memory) {
  SC_THREAD(Run);
  SC_THREAD(SampleRun);
  SC_THREAD(VertexSampleRun);
}

void TextureUnit::SampleRun() {
  SampleRunForStage(ShaderStage::kFragment, sample_input, sample_output);
}

void TextureUnit::VertexSampleRun() {
  SampleRunForStage(ShaderStage::kVertex, vertex_sample_input,
                    vertex_sample_output);
}

void TextureUnit::SampleRunForStage(
    ShaderStage shader_stage,
    sc_core::sc_port<sc_core::sc_fifo_in_if<PipelineTxn>, 0,
                     sc_core::SC_ZERO_OR_MORE_BOUND> &sample_input_port,
    sc_core::sc_port<sc_core::sc_fifo_out_if<PipelineTxn>, 0,
                     sc_core::SC_ZERO_OR_MORE_BOUND> &sample_output_port) {
  if (sample_input_port.size() == 0 || sample_output_port.size() == 0)
    return;
  if (!memory_ &&
      (cache_request.size() == 0 || cache_response.size() == 0 ||
       upload_request.size() == 0 || upload_response.size() == 0))
    return;
  while (true) {
    const PipelineTxn txn = sample_input_port->read();
    PipelineState state = LoadPipelineState(pool_, txn.state);
    const bool vertex_stage = shader_stage == ShaderStage::kVertex;
    const std::size_t stage_index = vertex_stage ? 0U : 1U;
    const PipelineStage pending_stage =
        vertex_stage ? PipelineStage::kVertexTexturePending
                     : PipelineStage::kFragmentTexturePending;
    const PipelineStage ready_stage =
        vertex_stage ? PipelineStage::kVertexTextureSamplesReady
                     : PipelineStage::kTextureSamplesReady;
    RequireStage(state.stage, pending_stage, name());
    if (memory_ && state.memory_mode != memory_->mode())
      throw std::runtime_error("TextureUnit memory mode mismatch");
    const PoolHandle resources_handle =
        vertex_stage ? state.vertex_texture_resources
                     : state.texture_resources;
    const PoolHandle samplers_handle =
        vertex_stage ? state.vertex_sampler_states : state.sampler_states;
    const PoolHandle shared_handle =
        vertex_stage ? state.vertex_shared_registers
                     : state.fragment_shared_registers;
    if (!UsesTextureSampling(state, shader_stage) ||
        !HasPoolHandle(state.texture_sample_requests) ||
        !HasPoolHandle(resources_handle) || !HasPoolHandle(samplers_handle) ||
        !HasPoolHandle(shared_handle)) {
      throw std::runtime_error(
          "TextureUnit received an invalid texture sample batch");
    }
    const std::vector<TextureSampleRequest> requests =
        LoadArray<TextureSampleRequest>(pool_, state.texture_sample_requests);
    const std::vector<TextureResource> resources =
        LoadArray<TextureResource>(pool_, resources_handle);
    const std::vector<SamplerState> samplers =
        LoadArray<SamplerState>(pool_, samplers_handle);
    std::vector<std::uint32_t> shared;
    if (vertex_stage) {
      const std::vector<ShaderSharedRegister> vertex_shared =
          LoadArray<ShaderSharedRegister>(pool_, shared_handle);
      shared.reserve(vertex_shared.size());
      for (const ShaderSharedRegister &word : vertex_shared)
        shared.push_back(word.value);
    } else {
      shared = LoadArray<std::uint32_t>(pool_, shared_handle);
    }
    const bool driver_pco =
        IsDriverPcoTrianglesCase(state.functional_case);
    const std::uint32_t descriptor_count =
        driver_pco
            ? (vertex_stage ? state.vertex_sampled_texture_count
                            : state.sampled_texture_count)
            : 1U;
    const std::uint32_t expected_shared_dwords =
        driver_pco
            ? (vertex_stage ? state.vertex_pco_abi.shareds
                            : state.fragment_pco_abi.shareds)
            : kFillTexNearestSharedDwordCount;
    const std::uint64_t expected_lane_count =
        vertex_stage ? state.counters.vs_invocations
                     : state.fragment_shader_lane_count;
    if ((vertex_stage && !driver_pco) || requests.empty() ||
        requests.size() != expected_lane_count ||
        descriptor_count == 0 ||
        descriptor_count > kPcoMaximumTextureDescriptorSets ||
        resources.size() != descriptor_count ||
        samplers.size() != descriptor_count ||
        expected_shared_dwords <
            descriptor_count * kFillTexNearestSharedDwordCount ||
        shared.size() != expected_shared_dwords) {
      throw std::runtime_error("TextureUnit resource/request count mismatch");
    }
    for (std::size_t set = 0; set < descriptor_count; ++set) {
      const TextureResource &candidate_resource = resources[set];
      const SamplerState &candidate_sampler = samplers[set];
      /*
       * The storage block is 1x1 for every format that stores one texel per
       * position, and the footprint for a compressed one.  A zero here would
       * make every later ceil(extent / block) divide by zero, so it is
       * checked with the rest of the metadata rather than at the divide.
       */
      const bool block_is_consistent =
          candidate_resource.block_width != 0 &&
          candidate_resource.block_height != 0 &&
          ((candidate_resource.format == TextureFormat::kAstcLdr ||
            candidate_resource.format == TextureFormat::kAstcLdrSrgb) ||
           (candidate_resource.block_width == 1 &&
            candidate_resource.block_height == 1));
      if (candidate_resource.descriptor_set != set ||
          candidate_resource.binding != 0 ||
          !block_is_consistent ||
          candidate_sampler.descriptor_set != set ||
          candidate_sampler.binding != 0 ||
          candidate_sampler.reserved[0] != 0 ||
          candidate_sampler.reserved[1] != 0) {
        throw std::runtime_error(
            "TextureUnit descriptor-set metadata is invalid");
      }
    }
    const std::uint32_t descriptor_set = requests.front().descriptor_set;
    if (descriptor_set >= descriptor_count)
      throw std::runtime_error("TextureUnit descriptor set is out of range");
    for (const TextureSampleRequest &request : requests) {
      if (request.shader_stage != shader_stage ||
          request.descriptor_set != descriptor_set || request.binding != 0) {
        throw std::runtime_error(
            "TextureUnit sample batch mixes shader stages, sets or bindings");
      }
    }

    const PoolHandle resident_state = residency_state_[stage_index];
    if (!HasPoolHandle(resident_state) || resident_state.slot != txn.state.slot ||
        resident_state.generation != txn.state.generation) {
      texture_preloaded_[stage_index].fill(false);
      preloaded_address_[stage_index].fill(0);
      preloaded_bytes_[stage_index].fill(0);
      residency_state_[stage_index] = txn.state;
    }
    const std::size_t descriptor_base =
        static_cast<std::size_t>(descriptor_set) *
        kFillTexNearestSharedDwordCount;
    const TextureResource &resource = resources[descriptor_set];
    const SamplerState &sampler = samplers[descriptor_set];
    std::array<std::uint32_t, 4> image_words{};
    std::array<std::uint32_t, 4> sampler_words{};
    std::copy_n(shared.begin() + descriptor_base, image_words.size(),
                image_words.begin());
    std::copy_n(shared.begin() + descriptor_base + 8U, sampler_words.size(),
                sampler_words.begin());
    const RogueTextureImageDescriptor image =
        DecodeRogueTextureImageDescriptor(
            image_words, resource.format == TextureFormat::kAstcLdr ||
                             resource.format == TextureFormat::kAstcLdrSrgb);
    const RogueTextureSamplerDescriptor decoded_sampler =
        DecodeRogueTextureSamplerDescriptor(sampler_words);

    // Raw public descriptor fields drive execution.  Structured resource and
    // sampler objects own the MemoryPool allocation and provide a redundant
    // command-side cross-check only; they may not silently override hardware
    // state.
    const bool resource_storage_valid =
        memory_ ? (!HasPoolHandle(resource.data) && resource.byte_size != 0 &&
                   memory_->backing().Contains(resource.gpu_address,
                                               resource.byte_size))
                : HasPoolHandle(resource.data);
    if (!resource_storage_valid || resource.byte_size == 0 ||
        resource.gpu_address != image.gpu_address ||
        resource.mip_count != image.mip_count ||
        resource.format != image.format || resource.layout != image.layout ||
        sampler.min_filter != decoded_sampler.min_filter ||
        sampler.mag_filter != decoded_sampler.mag_filter ||
        sampler.mip_filter != decoded_sampler.mip_filter ||
        sampler.wrap_u != decoded_sampler.wrap_u ||
        sampler.wrap_v != decoded_sampler.wrap_v ||
        sampler.min_lod_u4_6 != decoded_sampler.min_lod_u4_6 ||
        sampler.max_lod_u4_6 != decoded_sampler.max_lod_u4_6 ||
        sampler.normalized_coordinates !=
            decoded_sampler.normalized_coordinates ||
        sampler.base_mip_level != 0) {
      throw std::runtime_error(
          "TextureUnit structured state disagrees with raw descriptor");
    }
    if (driver_pco) {
      const std::uint64_t sampler_word0 = ReadU64(sampler_words, 0);
      const std::uint64_t expected_gather_word0 =
          sampler_word0 | (UINT64_C(1) << 36U) | (UINT64_C(1) << 38U);
      const std::uint64_t gather_word0 =
          static_cast<std::uint64_t>(shared[descriptor_base + 16U]) |
          (static_cast<std::uint64_t>(shared[descriptor_base + 17U])
           << 32U);
      const std::uint64_t gather_word1 =
          static_cast<std::uint64_t>(shared[descriptor_base + 18U]) |
          (static_cast<std::uint64_t>(shared[descriptor_base + 19U])
           << 32U);
      // Name the word that disagreed.  Ten conditions sharing one sentence
      // meant a descriptor that missed in one dword read exactly like one
      // built by a stage that had never been implemented.
      const char *descriptor_refusal =
          !DriverPcoTextureDescriptorClassSupported(image, decoded_sampler,
                                                    descriptor_count)
              ? "descriptor class is unsupported"
          : shared[descriptor_base + 4U] != resource.byte_size
              ? "word4 is not the resource byte size"
          : shared[descriptor_base + 5U] != 0   ? "word5 is not zero"
          : shared[descriptor_base + 6U] != 0   ? "word6 is not zero"
          : shared[descriptor_base + 7U] != 0   ? "word7 is not zero"
          : shared[descriptor_base + 12U] != 0  ? "word12 is not zero"
          : shared[descriptor_base + 13U] != 0  ? "word13 is not zero"
          : shared[descriptor_base + 14U] != 0  ? "word14 is not zero"
          : shared[descriptor_base + 15U] != 0  ? "word15 is not zero"
          : gather_word0 != expected_gather_word0
              ? "gather word0 does not match the sampler"
          : gather_word1 != 0 ? "gather word1 is not zero"
                              : nullptr;
      if (descriptor_refusal != nullptr) {
        std::ostringstream message;
        message << "TextureUnit driver PCO descriptor block mismatch: "
                << descriptor_refusal << " (descriptor " << descriptor_count
                << " at shared word " << descriptor_base
                << ", format=" << static_cast<unsigned>(image.format)
                << " mips=" << static_cast<unsigned>(image.mip_count)
                << " extent=" << image.width << 'x' << image.height
                << " filters=" << static_cast<unsigned>(decoded_sampler.min_filter)
                << '/' << static_cast<unsigned>(decoded_sampler.mag_filter)
                << '/' << static_cast<unsigned>(decoded_sampler.mip_filter)
                << " lod=" << decoded_sampler.min_lod_u4_6 << ".."
                << decoded_sampler.max_lod_u4_6
                << " wrap=" << static_cast<unsigned>(decoded_sampler.wrap_u)
                << ',' << static_cast<unsigned>(decoded_sampler.wrap_v)
                << ", word4=" << shared[descriptor_base + 4U]
                << " resource_bytes=" << resource.byte_size
                << ", gather0=0x" << std::hex << gather_word0
                << " expected=0x" << expected_gather_word0
                << ", gather1=0x" << gather_word1 << std::dec << ')';
        throw std::runtime_error(message.str());
      }
    }

    std::uint64_t expected_offset = 0;
    std::uint32_t expected_width = image.width;
    std::uint32_t expected_height = image.height;
    /*
     * A level occupies whole storage blocks: ceil(extent / block) of them in
     * each direction.  For every format that stores one texel per position
     * the block is 1x1 and this is the texel arithmetic it replaces.
     */
    const std::uint32_t storage_block_width =
        resource.block_width != 0 ? resource.block_width : 1U;
    const std::uint32_t storage_block_height =
        resource.block_height != 0 ? resource.block_height : 1U;
    const std::uint32_t storage_block_bytes =
        (image.format == TextureFormat::kAstcLdr ||
         image.format == TextureFormat::kAstcLdrSrgb)
            ? 16U
            : TextureBytesPerTexel(image.format);
    const auto storage_blocks = [](std::uint32_t extent,
                                   std::uint32_t block) {
      return (extent + block - 1U) / block;
    };
    for (std::uint32_t level = 0; level < image.mip_count; ++level) {
      const std::uint32_t tight_pitch =
          storage_blocks(expected_width, storage_block_width) *
          storage_block_bytes;
      const std::uint32_t expected_pitch =
          level == 0 ? image.row_pitch_bytes : tight_pitch;
      const TextureMipLevel &structured_mip = resource.mip[level];
      if (structured_mip.width != expected_width ||
          structured_mip.height != expected_height ||
          structured_mip.row_pitch_bytes != expected_pitch ||
          structured_mip.offset_bytes != expected_offset) {
        throw std::runtime_error(
            "TextureUnit structured mip layout disagrees with raw image");
      }
      // A 2D array stores layer_count images per level; a 3D image stores
      // `depth` slices whose count halves with each level.  Both are
      // slice-minor inside the level.
      const std::uint32_t level_slices =
          resource.dimension_type == TextureDimensionType::k3D
              ? std::max<std::uint32_t>(1U, resource.layer_count >> level)
              : resource.layer_count;
      expected_offset +=
          static_cast<std::uint64_t>(expected_pitch) *
          storage_blocks(expected_height, storage_block_height) *
          level_slices;
      if (expected_offset > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("TextureUnit mip allocation overflow");
      expected_width = std::max<std::uint32_t>(1U, expected_width >> 1U);
      expected_height = std::max<std::uint32_t>(1U, expected_height >> 1U);
    }
    if (expected_offset != resource.byte_size)
      throw std::runtime_error(
          "TextureUnit allocation size disagrees with raw mip state");

    /*
     * lp_build_sample_common computes a LOD only when something depends on
     * it: a mipped image whose level has to be selected, or a minification
     * filter that differs from the magnification one -- and then only if the
     * sampler's window lets lambda move at all.  A window of 0..0 pins
     * lambda to zero, which is the base level and the magnification filter
     * whatever the derivatives say.  Otherwise every lane samples the base
     * level with the one filter there is, and the batch need not be
     * quad-shaped.
     *
     * LODM=NORMAL is a quad operation, not four unrelated scalar requests.
     * Preserve the PDS/USC spatial identity and compute one derivative result
     * for each architectural lane quartet, including helper lanes.  A vertex
     * sample has no screen-space derivatives; llvmpipe samples it at LOD 0
     * and so does this unit.
     */
    const bool needs_lod =
        !vertex_stage && decoded_sampler.max_lod_u4_6 != 0 &&
        (image.mip_count > 1U ||
         decoded_sampler.min_filter != decoded_sampler.mag_filter);
    std::vector<TextureImplicitLod> implicit_lods(requests.size());
    if (needs_lod) {
      if (requests.size() % 4U != 0U)
        throw std::runtime_error(
            "TextureUnit LOD request batch is not quad aligned");
      for (std::size_t first = 0; first < requests.size(); first += 4U) {
        std::array<std::array<float, 2>, 4> coordinates{};
        const std::uint32_t quad_id = requests[first].quad_id;
        for (std::size_t lane = 0; lane < 4U; ++lane) {
          const TextureSampleRequest &request = requests[first + lane];
          if (request.quad_id != quad_id || request.quad_lane != lane)
            throw std::runtime_error(
                "TextureUnit LOD request lost 2x2 quad identity");
          coordinates[lane][0] = BitsFloat(request.coordinates[0]);
          coordinates[lane][1] = BitsFloat(request.coordinates[1]);
        }
        const TextureImplicitLod lod =
            ComputeTextureImplicitLod(coordinates, image, decoded_sampler);
        for (std::size_t lane = 0; lane < 4U; ++lane)
          implicit_lods[first + lane] = lod;
      }
    } else {
      // Zero derivatives: the window's minimum LOD, the base level.
      const std::array<std::array<float, 2>, 4> degenerate_quad{};
      const TextureImplicitLod base_level =
          ComputeTextureImplicitLod(degenerate_quad, image, decoded_sampler);
      std::fill(implicit_lods.begin(), implicit_lods.end(), base_level);
    }

    // Bounded, default-off evidence for diagnosing captured mip residency.
    // Report architectural request counts plus the exact selected mip pair and
    // TFRAC distributions; this deliberately observes the already-computed
    // datapath and cannot change sampling or descriptor semantics.
    if (!vertex_stage &&
        std::getenv("PVRGPU_SEQUENCE_DEBUG_LOD_HISTOGRAM") != nullptr) {
      std::array<std::array<std::uint64_t, kMaximumTextureMipLevels>,
                 kMaximumTextureMipLevels>
          level_pair_counts{};
      std::array<std::uint64_t, 256> tfrac_counts{};
      float minimum_lambda = 0.0F;
      float maximum_lambda = 0.0F;
      for (std::size_t index = 0; index < requests.size(); ++index) {
        const TextureImplicitLod &lod = implicit_lods[index];
        const std::uint8_t level0 = lod.level0;
        const std::uint8_t level1 = lod.level1;
        const std::uint8_t tfrac = lod.mip_weight_u8;
        if (level0 >= kMaximumTextureMipLevels ||
            level1 >= kMaximumTextureMipLevels) {
          throw std::runtime_error(
              "TextureUnit diagnostic observed an invalid mip level");
        }
        ++level_pair_counts[level0][level1];
        ++tfrac_counts[tfrac];
        if (needs_lod) {
          if (index == 0U) {
            minimum_lambda = lod.lambda;
            maximum_lambda = lod.lambda;
          } else {
            minimum_lambda = std::min(minimum_lambda, lod.lambda);
            maximum_lambda = std::max(maximum_lambda, lod.lambda);
          }
        }
      }
      std::cerr << "sequence-fragment-texture phase=lod-histogram"
                << " frame=" << txn.frame << " sequence=" << txn.sequence
                << " state=" << txn.state.slot << ':' << txn.state.generation
                << " set=" << static_cast<unsigned>(descriptor_set)
                << " requests=" << requests.size()
                << " quads=" << (requests.size() / 4U)
                << " needs_lod=" << static_cast<unsigned>(needs_lod)
                << " lambda_bits=0x" << std::hex << std::setw(8)
                << std::setfill('0') << FloatBits(minimum_lambda) << ",0x"
                << std::setw(8) << FloatBits(maximum_lambda) << std::dec
                << std::setfill(' ') << " level_pairs=";
      bool first_bin = true;
      for (std::size_t level0 = 0; level0 < level_pair_counts.size();
           ++level0) {
        for (std::size_t level1 = 0;
             level1 < level_pair_counts[level0].size(); ++level1) {
          const std::uint64_t count = level_pair_counts[level0][level1];
          if (count == 0)
            continue;
          if (!first_bin)
            std::cerr << ',';
          first_bin = false;
          std::cerr << level0 << '/' << level1 << ':' << count;
        }
      }
      std::cerr << " tfrac_bins=";
      first_bin = true;
      for (std::size_t tfrac = 0; tfrac < tfrac_counts.size(); ++tfrac) {
        const std::uint64_t count = tfrac_counts[tfrac];
        if (count == 0)
          continue;
        if (!first_bin)
          std::cerr << ',';
        first_bin = false;
        std::cerr << tfrac << ':' << count;
      }
      std::cerr << '\n';
    }

    MemoryAccessStats memory_stats;
    if (!texture_preloaded_[stage_index][descriptor_set]) {
      if (memory_) {
        if (!memory_->backing().Contains(image.gpu_address,
                                         resource.byte_size)) {
          throw std::runtime_error(
              "TextureUnit texture allocation is absent from DRAM backing");
        }
      } else {
        MemoryTxn upload;
        upload.pipeline = txn;
        upload.payload = resource.data;
        upload.address = image.gpu_address;
        upload.bytes = resource.byte_size;
        upload.operation = MemoryOperation::kWrite;
        upload.client = MemoryClient::kTextureUpload;
        upload.payload_format = MemoryPayloadFormat::kLinearBytes;
        upload_request->write(upload);
        const MemoryTxn upload_ack = upload_response->read();
        if (upload_ack.pipeline.state.slot != txn.state.slot ||
            upload_ack.pipeline.state.generation != txn.state.generation ||
            upload_ack.pipeline.frame != txn.frame ||
            upload_ack.pipeline.sequence != txn.sequence ||
            upload_ack.address != upload.address ||
            upload_ack.bytes != upload.bytes ||
            upload_ack.client != MemoryClient::kTextureUpload ||
            upload_ack.operation != MemoryOperation::kWrite ||
            upload_ack.payload_format != MemoryPayloadFormat::kLinearBytes ||
            HasPoolHandle(upload_ack.payload)) {
          throw std::runtime_error(
              "TextureUnit texture upload acknowledgement mismatch");
        }
      }
      texture_preloaded_[stage_index][descriptor_set] = true;
      preloaded_address_[stage_index][descriptor_set] = resource.gpu_address;
      preloaded_bytes_[stage_index][descriptor_set] = resource.byte_size;
    } else if (image.gpu_address !=
                   preloaded_address_[stage_index][descriptor_set] ||
               resource.byte_size !=
                   preloaded_bytes_[stage_index][descriptor_set]) {
      throw std::runtime_error(
          "TextureUnit pre-resident texture allocation changed without "
          "cache-coherent invalidation");
    }

    std::vector<TextureSampleResponse> responses;
    responses.reserve(requests.size());
    const bool debug_fragment =
        !vertex_stage &&
        std::getenv("PVRGPU_SEQUENCE_DEBUG_FRAGMENT") != nullptr;
    const std::uint32_t debug_x =
        debug_fragment
            ? DebugFragmentCoordinate("PVRGPU_SEQUENCE_DEBUG_X", 37U)
            : 37U;
    const std::uint32_t debug_y =
        debug_fragment
            ? DebugFragmentCoordinate("PVRGPU_SEQUENCE_DEBUG_Y", 46U)
            : 46U;
    const std::vector<FragmentShaderLane> debug_lanes =
        debug_fragment && HasPoolHandle(state.fragment_shader_lanes)
            ? LoadArray<FragmentShaderLane>(pool_,
                                            state.fragment_shader_lanes)
            : std::vector<FragmentShaderLane>{};
    const std::uint32_t debug_parameter =
        debug_fragment
            ? DebugFragmentCoordinate(
                  "PVRGPU_SEQUENCE_DEBUG_PARAMETER",
                  std::numeric_limits<std::uint32_t>::max())
            : std::numeric_limits<std::uint32_t>::max();
    const std::uint32_t debug_quad =
        debug_fragment
            ? DebugFragmentCoordinate(
                  "PVRGPU_SEQUENCE_DEBUG_QUAD_ID",
                  std::numeric_limits<std::uint32_t>::max())
            : std::numeric_limits<std::uint32_t>::max();
    bool debug_target_found = false;
    std::size_t debug_target_index = 0;
    std::uint32_t debug_target_parameter = 0;
    std::uint32_t debug_target_quad = 0;
    if (debug_lanes.size() == requests.size()) {
      for (std::size_t index = 0; index < requests.size(); ++index) {
        const FragmentShaderLane &lane = debug_lanes[index];
        if (lane.x != debug_x || lane.y != debug_y || lane.helper != 0 ||
            (debug_parameter != std::numeric_limits<std::uint32_t>::max() &&
             lane.parameter_index != debug_parameter) ||
            (debug_quad != std::numeric_limits<std::uint32_t>::max() &&
             lane.quad_id != debug_quad)) {
          continue;
        }
        if (debug_target_found &&
            (debug_target_parameter != lane.parameter_index ||
             debug_target_quad != lane.quad_id)) {
          throw std::runtime_error(
              "TextureUnit debug coordinate matches multiple fragment quads; "
              "set PVRGPU_SEQUENCE_DEBUG_PARAMETER/QUAD_ID");
        }
        debug_target_found = true;
        debug_target_index = index;
        debug_target_parameter = lane.parameter_index;
        debug_target_quad = lane.quad_id;
      }
    }

    if (debug_target_found && descriptor_set == 1U && needs_lod) {
      constexpr std::size_t kAbsent =
          std::numeric_limits<std::size_t>::max();
      std::array<std::size_t, 4> quad_indices = {
          kAbsent, kAbsent, kAbsent, kAbsent};
      for (std::size_t index = 0; index < requests.size(); ++index) {
        if (debug_lanes[index].parameter_index != debug_target_parameter ||
            requests[index].quad_id != debug_target_quad ||
            requests[index].quad_lane > 3U) {
          continue;
        }
        quad_indices[requests[index].quad_lane] = index;
      }
      if (std::any_of(quad_indices.begin(), quad_indices.end(),
                      [](std::size_t index) { return index == kAbsent; })) {
        throw std::runtime_error(
            "TextureUnit debug target lost an implicit-LOD quad lane");
      }
      const TextureImplicitLod &lod = implicit_lods[quad_indices[0]];
      const std::streamsize saved_precision = std::cerr.precision();
      std::cerr << std::setprecision(std::numeric_limits<float>::max_digits10)
                << "sequence-fragment-texture phase=implicit-lod-quad set="
                << static_cast<unsigned>(descriptor_set)
                << " target_lane=" << debug_target_index
                << " parameter=" << debug_target_parameter
                << " quad=" << debug_target_quad;
      for (std::size_t lane = 0; lane < quad_indices.size(); ++lane) {
        const std::size_t index = quad_indices[lane];
        std::cerr << " lane" << lane << '=' << debug_lanes[index].x << ','
                  << debug_lanes[index].y << ','
                  << static_cast<unsigned>(debug_lanes[index].helper)
                  << ",0x" << std::hex << std::setw(8) << std::setfill('0')
                  << requests[index].coordinates[0] << ",0x" << std::setw(8)
                  << requests[index].coordinates[1] << std::dec
                  << std::setfill(' ') << ','
                  << BitsFloat(requests[index].coordinates[0]) << ','
                  << BitsFloat(requests[index].coordinates[1]);
      }
      std::cerr << " derivatives_bits=0x" << std::hex << std::setw(8)
                << std::setfill('0') << FloatBits(lod.dsdx) << ",0x"
                << std::setw(8) << FloatBits(lod.dtdx) << ",0x"
                << std::setw(8) << FloatBits(lod.dsdy) << ",0x"
                << std::setw(8) << FloatBits(lod.dtdy)
                << " rho_squared_bits=0x" << std::setw(8)
                << FloatBits(lod.rho_squared) << " lambda_bits=0x"
                << std::setw(8) << FloatBits(lod.lambda) << std::dec
                << std::setfill(' ') << " derivatives=" << lod.dsdx << ','
                << lod.dtdx << ',' << lod.dsdy << ',' << lod.dtdy
                << " rho_squared=" << lod.rho_squared
                << " lambda=" << lod.lambda << " levels="
                << static_cast<unsigned>(lod.level0) << ','
                << static_cast<unsigned>(lod.level1) << " tfrac="
                << static_cast<unsigned>(lod.mip_weight_u8) << '\n';
      std::cerr.precision(saved_precision);
    }
    // A 2D array carries a third coordinate: the layer index.  Plain 2D
    // reads two.  The layer never affects LOD or the in-plane filter -- it
    // only selects which of the level's stacked images the taps read from.
    const bool array_texture =
        resource.dimension_type == TextureDimensionType::k2DArray;
    // A 3D image reads three coordinates: the third selects the depth slice
    // (or the two slices a linear filter blends).  Unlike an array layer it is
    // not folded into the address -- the shader passes it straight through and
    // the texture unit resolves the slice, whose stride is the same
    // slice-minor per-level image size the array path uses.  The slice count
    // halves with each mip level, so it is derived per level below.
    const bool volume_texture =
        resource.dimension_type == TextureDimensionType::k3D;
    // The SMP always carries two in-plane coordinates; the sample's
    // dimension names the texture (three for 3D, whose depth coordinate rides
    // in coordinates[2]).  A 2D array folds its layer into the address and
    // stays dimension two.
    const std::uint8_t expected_coordinate_count = 2U;
    const std::uint8_t expected_dimension = volume_texture ? 3U : 2U;
    std::uint64_t texel_fetch_count = 0;
    std::uint64_t expected_texel_fetches = 0;
    for (std::size_t index = 0; index < requests.size(); ++index) {
      const TextureSampleRequest &request = requests[index];
      const bool debug_request =
          debug_target_found &&
          debug_lanes[index].parameter_index == debug_target_parameter &&
          request.quad_id == debug_target_quad;
      const TextureImplicitLod &lod = implicit_lods[index];
      if (request.shader_lane_index != index || request.request_id != index ||
          request.shader_stage != shader_stage ||
          request.coordinate_count != expected_coordinate_count ||
          request.component_count != 4 ||
          request.descriptor_set != descriptor_set || request.binding != 0 ||
          request.dimension != expected_dimension ||
          request.normalized != 1 ||
          request.data_request != 0 ||
          (vertex_stage
               ? (request.quad_id != 0 || request.quad_lane != 0)
               : request.quad_lane > 3U) ||
          request.reserved[0] != 0 || request.reserved[1] != 0 ||
          request.reserved[2] != 0) {
        throw std::runtime_error("TextureUnit SMP request ABI mismatch");
      }
      for (std::size_t dword = 0; dword < 4; ++dword) {
        if (request.texture_state[dword] !=
                shared[descriptor_base + dword] ||
            request.sampler_state[dword] !=
                shared[descriptor_base + 8U + dword]) {
          throw std::runtime_error("TextureUnit SMP descriptor state mismatch");
        }
      }
      /*
       * An ASTC image stores one 128-bit block per footprint, so a texel is
       * fetched by fetching the block that contains it and decoding.  The
       * fetch goes through the same TCU path as any other, which is what
       * makes the cache and DRAM counters describe block traffic rather than
       * a texel read the hardware never issues.
       */
      // lp_build_layer_coord: the array layer is the third coordinate as a
      // signed integer (the shader applied f2i32_rtne before the sample),
      // clamped to the levels that exist.  It is 0 for a plain 2D image.
      // A 2D-array sample folds the layer into the texture address: the
      // shader computed base + layer * LAYER_SIZE, and the compiler's
      // LAYER_SIZE is the whole array allocation's byte size (its own array
      // stride), so the layer is (address - base) / byte_size.  The texel
      // within the layer is then addressed with the level-major per-level
      // layer stride below.
      std::uint32_t selected_layer = 0U;
      if (array_texture) {
        const std::uint64_t sample_address =
            (static_cast<std::uint64_t>(request.texture_address_hi) << 32U) |
            request.texture_address_lo;
        if (sample_address < image.gpu_address || resource.byte_size == 0)
          throw std::runtime_error(
              "TextureUnit array sample address is out of range");
        std::uint64_t layer =
            (sample_address - image.gpu_address) / resource.byte_size;
        if (layer >= resource.layer_count)
          layer = resource.layer_count - 1U;
        selected_layer = static_cast<std::uint32_t>(layer);
      }
      const bool astc_image = image.format == TextureFormat::kAstcLdr ||
                              image.format == TextureFormat::kAstcLdrSrgb;
      const AstcBlockFootprint astc_footprint{resource.block_width,
                                              resource.block_height};
      const std::uint32_t fetch_bytes =
          astc_image ? 16U : TextureBytesPerTexel(image.format);

      const auto read_texel = [&](const TextureMipLevel &mip,
                                  std::uint32_t x, std::uint32_t y,
                                  std::uint64_t memory_request_id) {
        const std::uint32_t fetch_x =
            astc_image ? x / astc_footprint.width : x;
        const std::uint32_t fetch_y =
            astc_image ? y / astc_footprint.height : y;
        const std::uint64_t layer_stride =
            static_cast<std::uint64_t>(mip.row_pitch_bytes) * mip.height;
        const std::uint64_t texel_offset =
            static_cast<std::uint64_t>(mip.offset_bytes) +
            static_cast<std::uint64_t>(selected_layer) * layer_stride +
            static_cast<std::uint64_t>(fetch_y) * mip.row_pitch_bytes +
            static_cast<std::uint64_t>(fetch_x) * fetch_bytes;
        if (texel_offset > resource.byte_size - fetch_bytes ||
            texel_offset > std::numeric_limits<std::uint64_t>::max() -
                               resource.gpu_address) {
          throw std::runtime_error(
              "TextureUnit texel address is out of range");
        }
        const std::uint64_t texel_address = image.gpu_address + texel_offset;
        std::vector<std::uint8_t> payload;
        if (memory_) {
          MemoryReadResult read = memory_->Read(
              texel_address, fetch_bytes, MemoryClient::kTextureCache);
          payload = std::move(read.data);
          memory_stats += read.stats;
        } else {
          MemoryTxn memory_request;
          memory_request.pipeline = txn;
          memory_request.address = texel_address;
          memory_request.bytes = fetch_bytes;
          memory_request.request_id = memory_request_id;
          memory_request.operation = MemoryOperation::kRead;
          memory_request.client = MemoryClient::kTextureCache;
          memory_request.payload_format = MemoryPayloadFormat::kLinearBytes;
          cache_request->write(memory_request);
          const MemoryTxn memory_response = cache_response->read();
          if (memory_response.pipeline.frame !=
                  memory_request.pipeline.frame ||
              memory_response.pipeline.sequence !=
                  memory_request.pipeline.sequence ||
              memory_response.pipeline.state.slot !=
                  memory_request.pipeline.state.slot ||
              memory_response.pipeline.state.generation !=
                  memory_request.pipeline.state.generation ||
              memory_response.request_id != memory_request.request_id ||
              memory_response.address != memory_request.address ||
              memory_response.bytes != fetch_bytes ||
              memory_response.client != MemoryClient::kTextureCache ||
              memory_response.operation != MemoryOperation::kRead ||
              memory_response.payload_format !=
                  MemoryPayloadFormat::kLinearBytes ||
              !HasPoolHandle(memory_response.payload)) {
            throw std::runtime_error("TextureUnit TCU response mismatch");
          }
          payload =
              LoadArray<std::uint8_t>(pool_, memory_response.payload);
          pool_.Release(memory_response.payload);
        }
        if (payload.size() != fetch_bytes)
          throw std::runtime_error("TextureUnit TCU texel size mismatch");
        if (texel_fetch_count == std::numeric_limits<std::uint64_t>::max())
          throw std::overflow_error("TextureUnit texel fetch overflow");
        ++texel_fetch_count;
        std::array<std::uint8_t, 8> texel{};
        if (astc_image) {
          AstcDecodedBlock block;
          const char *refusal = nullptr;
          if (!DecodeAstcBlock(payload.data(), astc_footprint,
                               /*srgb=*/false, &block, &refusal)) {
            throw std::runtime_error(
                std::string("TextureUnit cannot decode this ASTC block: ") +
                (refusal != nullptr ? refusal : "unstated"));
          }
          const std::uint32_t inside_x = x % astc_footprint.width;
          const std::uint32_t inside_y = y % astc_footprint.height;
          const std::array<std::uint8_t, 4> &decoded =
              block.texels[inside_y * astc_footprint.width + inside_x];
          std::copy(decoded.begin(), decoded.end(), texel.begin());
        } else {
          std::copy(payload.begin(), payload.end(), texel.begin());
        }
        // B8G8R8A8 storage arrives B,G,R,A; the descriptor's swizzle presents
        // it as RGBA, which the unit realises by swapping red and blue here so
        // every datapath downstream reads plain RGBA8 bytes.
        if (image.format == TextureFormat::kBgra8Unorm)
          std::swap(texel[0], texel[2]);
        return texel;  // valid bytes: fetch_bytes; upper bytes stay zero
      };

      const auto sample_bilinear_depth =
          [&](const TextureMipLevel &mip,
              std::uint64_t first_request_id) -> std::uint32_t {
        const TextureLinearAxis x = ComputeTextureLinearRepeat(
            BitsFloat(request.coordinates[0]), mip.width,
            decoded_sampler.wrap_u);
        const TextureLinearAxis y = ComputeTextureLinearRepeat(
            BitsFloat(request.coordinates[1]), mip.height,
            decoded_sampler.wrap_v);
        const std::uint32_t depth00 = SampledDepth24FromTexel(
            read_texel(mip, x.lower, y.lower, first_request_id + 0U));
        const std::uint32_t depth10 = SampledDepth24FromTexel(
            read_texel(mip, x.upper, y.lower, first_request_id + 1U));
        const std::uint32_t depth01 = SampledDepth24FromTexel(
            read_texel(mip, x.lower, y.upper, first_request_id + 2U));
        const std::uint32_t depth11 = SampledDepth24FromTexel(
            read_texel(mip, x.upper, y.upper, first_request_id + 3U));
        return LerpSampledDepth24(
            LerpSampledDepth24(depth00, depth10, x.weight),
            LerpSampledDepth24(depth01, depth11, x.weight), y.weight);
      };

      /*
       * One nearest tap from a chosen level.  The mip-linear path needs this
       * to serve GL_NEAREST_MIPMAP_LINEAR -- nearest inside each level, linear
       * between them -- which is GL's default minification filter and which
       * this unit used to decline because its only cross-level path filtered
       * bilinearly inside every level.
       */
      const auto sample_nearest =
          [&](const TextureMipLevel &mip, std::uint64_t request_id) {
        const std::uint32_t x =
            ComputeTextureNearestRepeat(BitsFloat(request.coordinates[0]),
                                        mip.width, decoded_sampler.wrap_u);
        const std::uint32_t y =
            ComputeTextureNearestRepeat(BitsFloat(request.coordinates[1]),
                                        mip.height, decoded_sampler.wrap_v);
        return read_texel(mip, x, y, request_id);
      };

      const auto sample_bilinear =
          [&](const TextureMipLevel &mip,
              std::uint64_t first_request_id) {
        const TextureLinearAxis x = ComputeTextureLinearRepeat(
            BitsFloat(request.coordinates[0]), mip.width,
            decoded_sampler.wrap_u);
        const TextureLinearAxis y = ComputeTextureLinearRepeat(
            BitsFloat(request.coordinates[1]), mip.height,
            decoded_sampler.wrap_v);
        const std::array<std::uint8_t, 8> texel00 =
            read_texel(mip, x.lower, y.lower, first_request_id + 0U);
        const std::array<std::uint8_t, 8> texel10 =
            read_texel(mip, x.upper, y.lower, first_request_id + 1U);
        const std::array<std::uint8_t, 8> texel01 =
            read_texel(mip, x.lower, y.upper, first_request_id + 2U);
        const std::array<std::uint8_t, 8> texel11 =
            read_texel(mip, x.upper, y.upper, first_request_id + 3U);
        std::array<std::uint8_t, 8> result{};
        for (std::size_t component = 0; component < 4U; ++component) {
          const std::uint8_t lower =
              LerpTextureUnorm8(texel00[component], texel10[component],
                                x.weight);
          const std::uint8_t upper =
              LerpTextureUnorm8(texel01[component], texel11[component],
                                x.weight);
          result[component] =
              LerpTextureUnorm8(lower, upper, y.weight);
        }
        if (debug_request) {
          std::cerr << "sequence-fragment-texture phase=bilinear set="
                    << static_cast<unsigned>(descriptor_set)
                    << " lane=" << index << " mip="
                    << static_cast<unsigned>(mip.width) << 'x'
                    << static_cast<unsigned>(mip.height) << " x="
                    << x.lower << ',' << x.upper << ','
                    << static_cast<unsigned>(x.weight) << " y=" << y.lower
                    << ',' << y.upper << ','
                    << static_cast<unsigned>(y.weight) << " texels=";
          const std::array<std::array<std::uint8_t, 8>, 4> texels = {
              texel00, texel10, texel01, texel11};
          for (const auto &texel : texels) {
            std::cerr << '[';
            for (std::size_t component = 0; component < 4; ++component) {
              if (component)
                std::cerr << ',';
              std::cerr << static_cast<unsigned>(texel[component]);
            }
            std::cerr << ']';
          }
          std::cerr << " result=";
          for (std::size_t component = 0; component < 4; ++component) {
            if (component)
              std::cerr << ',';
            std::cerr << static_cast<unsigned>(result[component]);
          }
          std::cerr << '\n';
        }
        return result;
      };

      // The float datapath's taps (lp_bld_sample_soa.c): binary32 weights,
      // texels decoded before the lerp so sRGB is blended in linear light.
      const auto sample_nearest_float =
          [&](const TextureMipLevel &mip, std::uint64_t request_id) {
        const std::uint32_t x = ComputeTextureFloatNearest(
            BitsFloat(request.coordinates[0]), mip.width,
            decoded_sampler.wrap_u);
        const std::uint32_t y = ComputeTextureFloatNearest(
            BitsFloat(request.coordinates[1]), mip.height,
            decoded_sampler.wrap_v);
        return DecodeTexelToFloat(image.format,
                                  read_texel(mip, x, y, request_id));
      };

      const auto sample_bilinear_float =
          [&](const TextureMipLevel &mip, std::uint64_t first_request_id) {
        const TextureFloatAxis x = ComputeTextureFloatLinear(
            BitsFloat(request.coordinates[0]), mip.width,
            decoded_sampler.wrap_u);
        const TextureFloatAxis y = ComputeTextureFloatLinear(
            BitsFloat(request.coordinates[1]), mip.height,
            decoded_sampler.wrap_v);
        const std::array<float, 4> texel00 = DecodeTexelToFloat(
            image.format,
            read_texel(mip, x.lower, y.lower, first_request_id + 0U));
        const std::array<float, 4> texel10 = DecodeTexelToFloat(
            image.format,
            read_texel(mip, x.upper, y.lower, first_request_id + 1U));
        const std::array<float, 4> texel01 = DecodeTexelToFloat(
            image.format,
            read_texel(mip, x.lower, y.upper, first_request_id + 2U));
        const std::array<float, 4> texel11 = DecodeTexelToFloat(
            image.format,
            read_texel(mip, x.upper, y.upper, first_request_id + 3U));
        std::array<float, 4> result{};
        for (std::size_t component = 0; component < result.size();
             ++component) {
          // lp_build_lerp_2d: along x first, then y.
          const float lower = LerpTextureFloat(
              texel00[component], texel10[component], x.weight);
          const float upper = LerpTextureFloat(
              texel01[component], texel11[component], x.weight);
          result[component] = LerpTextureFloat(lower, upper, y.weight);
        }
        if (debug_request) {
          std::cerr << "sequence-fragment-texture phase=bilinear-float set="
                    << static_cast<unsigned>(descriptor_set)
                    << " lane=" << index << " mip="
                    << static_cast<unsigned>(mip.width) << 'x'
                    << static_cast<unsigned>(mip.height) << " x="
                    << x.lower << ',' << x.upper << ',' << x.weight
                    << " y=" << y.lower << ',' << y.upper << ','
                    << y.weight << " result=" << result[0] << ','
                    << result[1] << ',' << result[2] << ',' << result[3]
                    << '\n';
        }
        return result;
      };

      std::array<float, 4> filtered{};
      if (debug_request) {
        std::cerr << "sequence-fragment-texture phase=request set="
                  << static_cast<unsigned>(descriptor_set) << " lane="
                  << index << " quad=" << request.quad_id << ':'
                  << static_cast<unsigned>(request.quad_lane)
                  << " coord_bits=0x" << std::hex << std::setw(8)
                  << std::setfill('0') << request.coordinates[0] << ",0x"
                  << std::setw(8) << request.coordinates[1] << std::dec
                  << std::setfill(' ') << " coord="
                  << BitsFloat(request.coordinates[0]) << ','
                  << BitsFloat(request.coordinates[1]);
        if (needs_lod) {
          std::cerr << " lod=" << lod.lambda << ','
                    << static_cast<unsigned>(lod.level0) << ','
                    << static_cast<unsigned>(lod.level1) << ','
                    << static_cast<unsigned>(lod.mip_weight_u8);
        }
        std::cerr << '\n';
      }
      /*
       * lp_build_sample_general, per lane: the selection made from the
       * quad's LOD names the image filter and the level pair, the format and
       * the address modes name the datapath.  A mip-linear selection blends
       * its two levels on the datapath that filtered them -- 8-bit weights on
       * the fixed-point one, the fractional LOD on the float one.  Depth
       * keeps its own 24-bit datapath.
       */
      const bool linear_filter = lod.image_filter == TextureFilter::kLinear;
      const bool two_levels = lod.mip_mode == TextureMipMode::kLinear;
      const TextureFilterDatapath datapath =
          SelectTextureFilterDatapath(image.format, decoded_sampler);
      if (request.request_id >
          (std::numeric_limits<std::uint64_t>::max() - 15U) / 16U) {
        throw std::overflow_error("TextureUnit sample request ID overflow");
      }
      // Up to sixteen taps per sample -- a 3D trilinear filter reads two mip
      // levels, two depth slices each and a 2x2 footprint per slice; memory
      // request IDs need only be distinct within the batch.
      const std::uint64_t tap_request_base = request.request_id * 16U;
      const TextureMipLevel &level0 = resource.mip[lod.level0];
      const TextureMipLevel &level1 = resource.mip[lod.level1];
      // A linearly filtered 3D image blends the two nearest depth slices, so
      // it doubles the taps a 2D image of the same filter would read.
      const bool volume_linear = volume_texture && linear_filter;
      expected_texel_fetches +=
          (linear_filter ? 4U : 1U) * (two_levels ? 2U : 1U) *
          (volume_linear ? 2U : 1U);
      // The third coordinate and the wrap on the depth axis.  The formats and
      // sizes subgroups wrap every axis alike, so the r axis reuses wrap_u
      // until a distinct depth wrap is plumbed for the wrap combinations.
      const float volume_r =
          volume_texture ? BitsFloat(request.coordinates[2]) : 0.0F;
      const TextureWrapMode wrap_r = decoded_sampler.wrap_u;
      const std::uint32_t base_depth =
          resource.layer_count == 0U ? 1U : resource.layer_count;
      const auto level_depth = [&](std::uint8_t level_index) -> std::uint32_t {
        const std::uint32_t d = base_depth >> level_index;
        return d == 0U ? 1U : d;
      };

      if (image.format == TextureFormat::kZ32Unorm) {
        if (linear_filter || two_levels)
          throw std::runtime_error("TextureUnit cannot filter Z32_UNORM");
        const std::uint32_t x = ComputeTextureNearestRepeat(
            BitsFloat(request.coordinates[0]), level0.width,
            decoded_sampler.wrap_u);
        const std::uint32_t y = ComputeTextureNearestRepeat(
            BitsFloat(request.coordinates[1]), level0.height,
            decoded_sampler.wrap_v);
        const std::array<std::uint8_t, 8> texel =
            read_texel(level0, x, y, tap_request_base);
        std::uint32_t encoded = 0;
        std::memcpy(&encoded, texel.data(), sizeof(encoded));
        const float depth = static_cast<float>(
            static_cast<double>(encoded) /
            static_cast<double>(std::numeric_limits<std::uint32_t>::max()));
        filtered = {depth, depth, depth, 1.0F};
        if (debug_request) {
          std::cerr << "sequence-fragment-texture phase=nearest-depth set="
                    << static_cast<unsigned>(descriptor_set) << " texel="
                    << x << ',' << y << " encoded=0x" << std::hex
                    << std::setw(8) << std::setfill('0') << encoded
                    << std::dec << std::setfill(' ') << " depth=" << depth
                    << '\n';
        }
      } else if (image.format == TextureFormat::kZ24UnormS8Uint) {
        const auto depth_level = [&](const TextureMipLevel &mip,
                                     std::uint64_t first_request_id) {
          if (linear_filter)
            return sample_bilinear_depth(mip, first_request_id);
          const std::uint32_t x = ComputeTextureNearestRepeat(
              BitsFloat(request.coordinates[0]), mip.width,
              decoded_sampler.wrap_u);
          const std::uint32_t y = ComputeTextureNearestRepeat(
              BitsFloat(request.coordinates[1]), mip.height,
              decoded_sampler.wrap_v);
          return SampledDepth24FromTexel(
              read_texel(mip, x, y, first_request_id));
        };
        std::uint32_t depth = depth_level(level0, tap_request_base);
        if (two_levels) {
          depth = LerpSampledDepth24(
              depth, depth_level(level1, tap_request_base + 4U),
              lod.mip_weight_u8);
        }
        filtered = {SampledDepth24ToFloat(depth), 0.0F, 0.0F, 1.0F};
      } else if (datapath == TextureFilterDatapath::kUnorm8) {
        const auto unorm8_plane = [&](const TextureMipLevel &mip,
                                      std::uint64_t rid) {
          return linear_filter ? sample_bilinear(mip, rid)
                               : sample_nearest(mip, rid);
        };
        // One mip level, filtered in the plane and then across depth: a
        // nearest depth filter takes the one slice the r axis rounds to, a
        // linear one lerps the two nearest slices with the axis weight.
        const auto unorm8_level = [&](const TextureMipLevel &mip,
                                      std::uint8_t level_index,
                                      std::uint64_t rid)
            -> std::array<std::uint8_t, 8> {
          if (!volume_texture)
            return unorm8_plane(mip, rid);
          const std::uint32_t d = level_depth(level_index);
          if (linear_filter) {
            const TextureLinearAxis z =
                ComputeTextureLinearRepeat(volume_r, d, wrap_r);
            selected_layer = z.lower;
            const std::array<std::uint8_t, 8> lo = unorm8_plane(mip, rid);
            selected_layer = z.upper;
            const std::array<std::uint8_t, 8> hi = unorm8_plane(mip, rid + 4U);
            std::array<std::uint8_t, 8> blended{};
            for (std::size_t component = 0; component < 4U; ++component)
              blended[component] =
                  LerpTextureUnorm8(lo[component], hi[component], z.weight);
            return blended;
          }
          selected_layer = ComputeTextureNearestRepeat(volume_r, d, wrap_r);
          return unorm8_plane(mip, rid);
        };
        std::array<std::uint8_t, 8> texel =
            unorm8_level(level0, lod.level0, tap_request_base);
        if (two_levels) {
          const std::array<std::uint8_t, 8> upper =
              unorm8_level(level1, lod.level1,
                           tap_request_base + (volume_texture ? 8U : 4U));
          for (std::size_t component = 0; component < 4; ++component) {
            texel[component] = LerpTextureUnorm8(
                texel[component], upper[component], lod.mip_weight_u8);
          }
        }
        for (std::size_t component = 0; component < 4; ++component)
          filtered[component] = static_cast<float>(texel[component]) / 255.0F;
      } else {
        const auto float_plane = [&](const TextureMipLevel &mip,
                                     std::uint64_t rid) {
          return linear_filter ? sample_bilinear_float(mip, rid)
                               : sample_nearest_float(mip, rid);
        };
        const auto float_level = [&](const TextureMipLevel &mip,
                                     std::uint8_t level_index,
                                     std::uint64_t rid)
            -> std::array<float, 4> {
          if (!volume_texture)
            return float_plane(mip, rid);
          const std::uint32_t d = level_depth(level_index);
          if (linear_filter) {
            const TextureFloatAxis z =
                ComputeTextureFloatLinear(volume_r, d, wrap_r);
            selected_layer = z.lower;
            const std::array<float, 4> lo = float_plane(mip, rid);
            selected_layer = z.upper;
            const std::array<float, 4> hi = float_plane(mip, rid + 4U);
            std::array<float, 4> blended{};
            for (std::size_t component = 0; component < 4U; ++component)
              blended[component] =
                  LerpTextureFloat(lo[component], hi[component], z.weight);
            return blended;
          }
          selected_layer = ComputeTextureFloatNearest(volume_r, d, wrap_r);
          return float_plane(mip, rid);
        };
        filtered = float_level(level0, lod.level0, tap_request_base);
        if (two_levels) {
          const std::array<float, 4> upper =
              float_level(level1, lod.level1,
                          tap_request_base + (volume_texture ? 8U : 4U));
          for (std::size_t component = 0; component < 4; ++component) {
            filtered[component] = LerpTextureFloat(
                filtered[component], upper[component], lod.mip_weight);
          }
        }
      }
      if (image.format == TextureFormat::kRgbx8Unorm)
        filtered[3] = 1.0F;
      TextureSampleResponse response;
      response.shader_lane_index = request.shader_lane_index;
      response.request_id = request.request_id;
      response.shader_stage = shader_stage;
      for (std::size_t component = 0; component < 4; ++component)
        response.rgba[component] = FloatBits(filtered[component]);
      if (debug_request) {
        std::cerr << "sequence-fragment-texture phase=response set="
                  << static_cast<unsigned>(descriptor_set) << " rgba_bits=";
        for (std::size_t component = 0; component < 4; ++component) {
          if (component)
            std::cerr << ',';
          std::cerr << "0x" << std::hex << std::setw(8)
                    << std::setfill('0') << response.rgba[component]
                    << std::dec << std::setfill(' ');
        }
        std::cerr << " rgba=" << filtered[0] << ',' << filtered[1] << ','
                  << filtered[2] << ',' << filtered[3] << '\n';
      }
      responses.push_back(response);
    }

    if (texel_fetch_count != expected_texel_fetches) {
      throw std::runtime_error(
          "TextureUnit sample batch has invalid texel traffic");
    }

    state = LoadPipelineState(pool_, txn.state);
    RequireStage(state.stage, pending_stage, name());
    if (HasPoolHandle(state.texture_sample_responses))
      throw std::runtime_error("TextureUnit response payload already exists");
    state.texture_sample_responses = StoreNewArray(pool_, responses);
    if (requests.size() >
        std::numeric_limits<std::uint64_t>::max() -
            state.counters.texture_requests)
      throw std::overflow_error("TextureUnit request counter overflow");
    state.counters.texture_requests += requests.size();
    if (texel_fetch_count >
        std::numeric_limits<std::uint64_t>::max() -
            state.counters.texel_fetches) {
      throw std::overflow_error("TextureUnit texel counter overflow");
    }
    state.counters.texel_fetches += texel_fetch_count;
    std::uint64_t &stage_requests =
        vertex_stage ? state.vertex_texture_request_count
                     : state.fragment_texture_request_count;
    std::uint64_t &stage_fetches =
        vertex_stage ? state.vertex_texel_fetch_count
                     : state.fragment_texel_fetch_count;
    if (requests.size() >
            std::numeric_limits<std::uint64_t>::max() - stage_requests ||
        texel_fetch_count >
            std::numeric_limits<std::uint64_t>::max() - stage_fetches) {
      throw std::overflow_error("TextureUnit stage texture counter overflow");
    }
    stage_requests += requests.size();
    stage_fetches += texel_fetch_count;
    if (requests.size() >
        std::numeric_limits<std::uint64_t>::max() /
            kReferenceUarch.texture_bypass_cycles) {
      throw std::overflow_error("TextureUnit cycle counter overflow");
    }
    const std::uint64_t functional_cycles =
        requests.size() * kReferenceUarch.texture_bypass_cycles;
    ApplyMemoryAccessStats(state.counters, memory_stats);
    const std::uint64_t cycles =
        functional_cycles + MemoryAccessDelayCycles(memory_stats);
    state.counters.texture_cycles += cycles;
    if (vertex_stage)
      state.counters.tiler_cycles += cycles;
    else
      state.counters.renderer_cycles += cycles;
    state.stage = ready_stage;
    WaitForCycles(cycles);
    StorePipelineState(pool_, txn.state, state);
    sample_output_port->write(txn);
  }
}

void TextureUnit::Run() {
  while (true) {
    const PipelineTxn txn = input.read();
    PipelineState state = LoadPipelineState(pool_, txn.state);

    RequireStage(state.stage, PipelineStage::kFragmentShaded, name());
    if (memory_ && state.memory_mode != memory_->mode())
      throw std::runtime_error("TextureUnit memory mode mismatch");
    if (!IsRasterFunctionalCase(state.functional_case))
      throw std::runtime_error("texture unit received an unsupported case");
    // The lanes the attachment expects, which is four only when it has four
    // channels.
    const std::uint32_t expected_pixel_output_mask =
        state.fragment_output_mask != 0 ? state.fragment_output_mask : 0x0fU;
    if (!HasPoolHandle(state.fragment_outputs) ||
        state.fragment_program_summary.pixel_output_mask !=
            expected_pixel_output_mask) {
      throw std::runtime_error(
          "texture bypass pixel output mask does not match the attachment: "
          "summary=" +
          std::to_string(state.fragment_program_summary.pixel_output_mask) +
          " expected=" + std::to_string(expected_pixel_output_mask));
    }
    const bool vertex_texture_case =
        UsesTextureSampling(state, ShaderStage::kVertex);
    const bool fragment_texture_case =
        UsesTextureSampling(state, ShaderStage::kFragment);
    const bool texture_case = vertex_texture_case || fragment_texture_case;
    if (!texture_case &&
        (state.counters.texture_requests != 0 ||
         state.counters.texel_fetches != 0 ||
         state.vertex_texture_request_count != 0 ||
         state.fragment_texture_request_count != 0 ||
         state.vertex_texel_fetch_count != 0 ||
         state.fragment_texel_fetch_count != 0)) {
      throw std::runtime_error(
          "solid-color raster case unexpectedly issued texture requests");
    }

    const std::uint64_t cycles = texture_case
            ? 0
            :
        state.active_fragment_invocations == 0
            ? 0
            : kReferenceUarch.texture_bypass_cycles;
    if (!texture_case) {
      state.counters.texture_requests = 0;
      state.counters.texel_fetches = 0;
    } else {
      const std::uint64_t stage_requests =
          state.vertex_texture_request_count +
          state.fragment_texture_request_count;
      const std::uint64_t stage_fetches =
          state.vertex_texel_fetch_count + state.fragment_texel_fetch_count;
      const std::uint64_t executed_texture_instructions =
          state.counters.vs_tex_instructions +
          state.counters.fs_tex_instructions;
      const bool request_sum_overflow =
          stage_requests < state.vertex_texture_request_count;
      const bool fetch_sum_overflow =
          stage_fetches < state.vertex_texel_fetch_count;
      const bool instruction_sum_overflow =
          executed_texture_instructions < state.counters.vs_tex_instructions;
      if (request_sum_overflow || fetch_sum_overflow ||
          instruction_sum_overflow ||
          state.counters.texture_requests != stage_requests ||
          state.counters.texel_fetches != stage_fetches ||
          state.counters.texture_requests != executed_texture_instructions ||
          state.vertex_texture_request_count !=
              state.counters.vs_tex_instructions ||
          state.fragment_texture_request_count !=
              state.counters.fs_tex_instructions ||
          (!vertex_texture_case &&
           (state.vertex_texture_request_count != 0 ||
            state.vertex_texel_fetch_count != 0)) ||
          (!fragment_texture_case &&
           (state.fragment_texture_request_count != 0 ||
            state.fragment_texel_fetch_count != 0))) {
        throw std::runtime_error(
            "TextureUnit completed with invalid texture traffic");
      }
    }

    state.counters.texture_cycles += cycles;
    state.counters.renderer_cycles += cycles;
    state.stage = PipelineStage::kTextureComplete;

    WaitForCycles(cycles);
    StorePipelineState(pool_, txn.state, state);
    output.write(txn);
  }
}

} // namespace pvrgpu::stub
