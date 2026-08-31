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

#include "common/functional_types.h"
#include "common/pipeline_state.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pvrgpu::stub {
namespace {

inline constexpr float kDefaultLinearCoordinateRoundThreshold = 0.5F;
inline constexpr float kGlbenchTrilinear04CoordinateRoundThreshold =
    0.5F - 1.0F / 256.0F;
inline constexpr float kMipWeightByteSnapEpsilon = 1.0F / 64.0F;

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
    return TextureWrapMode::kClampToEdge;
  case 2:
    return TextureWrapMode::kMirroredRepeat;
  case 3:
    return TextureWrapMode::kClampToBorder;
  default:
    throw std::runtime_error("TextureUnit unsupported wrap mode encoding");
  }
}

std::uint32_t RepeatIndex(std::int64_t integer, std::uint32_t extent) {
  const std::int64_t modulus = extent;
  const std::int64_t wrapped = ((integer % modulus) + modulus) % modulus;
  return static_cast<std::uint32_t>(wrapped);
}

std::uint32_t WrapTexelIndex(std::int64_t integer, std::uint32_t extent,
                             TextureWrapMode wrap) {
  if (extent == 0)
    throw std::runtime_error("TextureUnit texel extent is invalid");
  if (wrap == TextureWrapMode::kRepeat)
    return RepeatIndex(integer, extent);
  if (wrap == TextureWrapMode::kClampToEdge) {
    if (integer < 0)
      return 0;
    if (integer >= static_cast<std::int64_t>(extent))
      return extent - 1U;
    return static_cast<std::uint32_t>(integer);
  }
  if (wrap == TextureWrapMode::kMirroredRepeat) {
    const std::int64_t period = static_cast<std::int64_t>(extent) * 2;
    const std::int64_t wrapped = ((integer % period) + period) % period;
    if (wrapped >= static_cast<std::int64_t>(extent))
      return static_cast<std::uint32_t>(period - 1 - wrapped);
    return static_cast<std::uint32_t>(wrapped);
  }
  throw std::runtime_error("TextureUnit clamp-to-border sampling is unsupported");
}

std::uint32_t NearestRepeat(float coordinate, std::uint32_t extent, TextureWrapMode wrap) {
  if (!std::isfinite(coordinate) || extent == 0)
    throw std::runtime_error("TextureUnit coordinate/extent is invalid");

  if (wrap == TextureWrapMode::kClampToEdge) {
    float clamped = std::clamp(coordinate, 0.0f, 1.0f);
    double scaled = std::floor(static_cast<double>(clamped) * extent);
    std::int64_t index = static_cast<std::int64_t>(scaled);
    if (index >= static_cast<std::int64_t>(extent)) {
      index = extent - 1;
    }
    if (index < 0) {
      index = 0;
    }
    return static_cast<std::uint32_t>(index);
  } else if (wrap == TextureWrapMode::kMirroredRepeat) {
    float floored = std::floor(coordinate);
    float frac = coordinate - floored;
    bool is_odd = (static_cast<std::int64_t>(floored) % 2) != 0;
    float mapped = is_odd ? (1.0f - frac) : frac;
    double scaled = std::floor(static_cast<double>(mapped) * extent);
    std::int64_t index = static_cast<std::int64_t>(scaled);
    if (index >= static_cast<std::int64_t>(extent)) {
      index = extent - 1;
    }
    if (index < 0) {
      index = 0;
    }
    return static_cast<std::uint32_t>(index);
  } else {
    const double scaled = std::floor(static_cast<double>(coordinate) * extent);
    if (scaled < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
        scaled > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
      throw std::overflow_error("TextureUnit normalized coordinate overflow");
    }
    return RepeatIndex(static_cast<std::int64_t>(scaled), extent);
  }
}

} // namespace

RogueTextureImageDescriptor DecodeRogueTextureImageDescriptor(
    const std::array<std::uint32_t, 4> &words) {
  const std::uint64_t word0 = ReadU64(words, 0);
  const std::uint64_t word1 = ReadU64(words, 2);

  // Public Rogue IMAGE_WORD0.  The selected path is a linear-stride,
  // non-gamma U8U8U8U8 image with identity RGBA swizzle and no image min-LOD
  // or sample-count override.
  if (ExtractBits(word0, 0, 2) != 4U ||
      ExtractBits(word0, 3, 4) != 0U ||
      ExtractBits(word0, 5, 7) != 3U ||
      ExtractBits(word0, 8, 10) != 2U ||
      ExtractBits(word0, 11, 13) != 1U ||
      ExtractBits(word0, 14, 16) != 0U ||
      ExtractBits(word0, 17, 26) != 0U ||
      ExtractBits(word0, 27, 33) != 12U ||
      ExtractBits(word0, 62, 63) != 0U) {
    throw std::runtime_error(
        "TextureUnit unsupported raw Rogue image word0");
  }

  // Public Rogue STRIDE_IMAGE_WORD1.  Compression/index/tile/alpha controls
  // are unsupported and must remain zero.  Gate16/17 retain the complete ten
  // level GLBench allocation, even though the sampler clamps execution to L0.
  if (ExtractBits(word1, 54, 59) != 0U ||
      ExtractBits(word1, 15, 15) != 1U ||
      ExtractBits(word1, 60, 63) != kMaximumTextureMipLevels) {
    throw std::runtime_error(
        "TextureUnit unsupported raw Rogue stride image word1");
  }

  RogueTextureImageDescriptor descriptor;
  descriptor.width =
      static_cast<std::uint32_t>(ExtractBits(word0, 34, 47) + 1U);
  descriptor.height =
      static_cast<std::uint32_t>(ExtractBits(word0, 48, 61) + 1U);
  descriptor.row_pitch_bytes =
      static_cast<std::uint32_t>(ExtractBits(word1, 0, 14) + 1U);
  descriptor.gpu_address = ExtractBits(word1, 16, 53) << 2U;
  descriptor.mip_count =
      static_cast<std::uint8_t>(ExtractBits(word1, 60, 63));
  if (descriptor.gpu_address == 0 ||
      descriptor.row_pitch_bytes < descriptor.width * 4U) {
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

  // dadjust=4095 is zero bias.  The selected path implements repeat U/V/W,
  // no anisotropy/luma-key/border/compare/YUV state, and either an exact LOD0
  // non-mip sampler (Gates 16/17) or the public full-range mip-linear sampler
  // used by Gate 18.  Reject every other public or reserved encoding.
  const bool lod0_sampler =
      descriptor.mip_filter == TextureFilter::kNearest &&
      descriptor.min_lod_u4_6 == 0U && descriptor.max_lod_u4_6 == 0U &&
      descriptor.min_filter == descriptor.mag_filter;
  const bool trilinear_sampler =
      descriptor.mip_filter == TextureFilter::kLinear &&
      descriptor.min_lod_u4_6 == 0U && descriptor.max_lod_u4_6 == 959U &&
      descriptor.min_filter == TextureFilter::kLinear &&
      descriptor.mag_filter == TextureFilter::kLinear;
  if (ExtractBits(word0, 0, 12) != 4095U ||
      ExtractBits(word0, 33, 35) != 0U ||
      ExtractBits(word0, 41, 43) != 0U ||
      ExtractBits(word0, 44, 46) != 0U ||
      ExtractBits(word0, 47, 48) != 0U ||
      descriptor.normalized_coordinates != 1U ||
      ExtractBits(word0, 50, 55) != 0U ||
      ExtractBits(word0, 56, 58) != 0U ||
      ExtractBits(word0, 59, 63) != 0U || word1 != 0U ||
      (!lod0_sampler && !trilinear_sampler)) {
    throw std::runtime_error(
        "TextureUnit unsupported raw Rogue sampler descriptor");
  }
  return descriptor;
}

TextureLinearAxis ComputeTextureLinearRepeat(float coordinate,
                                             std::uint32_t extent,
                                             TextureWrapMode wrap,
                                             float round_threshold) {
  if (!std::isfinite(coordinate) || extent == 0)
    throw std::runtime_error("TextureUnit coordinate/extent is invalid");
  if (!std::isfinite(round_threshold) || round_threshold < 0.0F ||
      round_threshold > 1.0F) {
    throw std::runtime_error("TextureUnit linear round threshold is invalid");
  }
  // The selected reference TPU uses the common 8-bit UNORM filter datapath:
  // multiply the live binary32 coordinate by N*256 in binary32, round to
  // nearest-even, then subtract the half-texel centre (128).  Power-of-two
  // repeat is applied to the integer taps afterwards.  Keeping the multiply
  // in binary32 is observable at half-LSB boundaries and is part of the
  // versioned reference-uArch assumption.
  const float scaled_extent = static_cast<float>(extent) * 256.0F;
  const float scaled = coordinate * scaled_extent;
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
       (round_threshold != kDefaultLinearCoordinateRoundThreshold ||
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
  result.lower = WrapTexelIndex(lower_integer, extent, wrap);
  result.upper = WrapTexelIndex(lower_integer + 1, extent, wrap);
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

TextureImplicitLod ComputeTextureImplicitLod(
    const std::array<std::array<float, 2>, 4> &coordinates,
    const RogueTextureImageDescriptor &image,
    const RogueTextureSamplerDescriptor &sampler) {
  if (image.width == 0 || image.height == 0 || image.mip_count == 0 ||
      sampler.mip_filter != TextureFilter::kLinear) {
    throw std::runtime_error("TextureUnit implicit LOD state is invalid");
  }
  for (const auto &coordinate : coordinates) {
    if (!std::isfinite(coordinate[0]) || !std::isfinite(coordinate[1]))
      throw std::runtime_error("TextureUnit implicit LOD coordinate is invalid");
  }

  // Public SMP LODM=NORMAL derives one isotropic LOD for a 2x2 quad.  The
  // selected reference uArch first computes the exact Euclidean derivative
  // norm, then uses a hardware-style piecewise-linear log2 approximation on
  // rho^2.  It is exact at powers of two and avoids a transcendental unit.
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
  if (!(rho_squared > 0.0F) || !std::isfinite(rho_squared))
    throw std::runtime_error("TextureUnit implicit derivative rho is invalid");

  int binary_exponent = 0;
  const float half_open_mantissa =
      std::frexp(rho_squared, &binary_exponent); // [0.5, 1.0)
  const float normalized_mantissa = half_open_mantissa * 2.0F;
  const float approximate_log2_squared =
      static_cast<float>(binary_exponent - 2) + normalized_mantissa;

  const float min_lod = static_cast<float>(sampler.min_lod_u4_6) / 64.0F;
  const float max_lod = static_cast<float>(sampler.max_lod_u4_6) / 64.0F;
  const float last_level = static_cast<float>(image.mip_count - 1U);
  const float lambda = std::clamp(approximate_log2_squared * 0.5F, min_lod,
                                  std::min(max_lod, last_level));
  const float level0_float = std::floor(lambda);
  const float level1_float = std::min(level0_float + 1.0F, last_level);

  TextureImplicitLod result;
  result.lambda = lambda;
  result.level0 = static_cast<std::uint8_t>(level0_float);
  result.level1 = static_cast<std::uint8_t>(level1_float);
  if (result.level0 != result.level1) {
    // Public Rogue exposes the filtering fraction as TFRAC_byte/256.  The
    // selected uArch truncates the positive fractional LOD into that byte.
    // Live quad coordinates can jitter by one float32 ulp around an exact
    // TFRAC byte boundary; snap only when the scaled byte value is within a
    // tiny sub-LSB window so GLBench gate 20's half-mip boundary stays stable
    // without changing ordinary truncation points.
    const float fractional = lambda - level0_float;
    const float scaled_byte = fractional * 256.0F;
    float scaled = std::floor(scaled_byte);
    const float nearest_byte = std::round(scaled_byte);
    if (std::fabs(scaled_byte - nearest_byte) <= kMipWeightByteSnapEpsilon)
      scaled = nearest_byte;
    result.mip_weight_u8 = static_cast<std::uint8_t>(
        std::clamp(scaled, 0.0F, 255.0F));
    result.mip_weight =
        static_cast<float>(result.mip_weight_u8) / 256.0F;
  }
  return result;
}

TextureUnit::TextureUnit(sc_core::sc_module_name name, MemoryPool &pool,
                         GpuMemorySystem *memory)
    : sc_module(name), pool_(pool), memory_(memory) {
  SC_THREAD(Run);
  SC_THREAD(SampleRun);
}

void TextureUnit::SampleRun() {
  if (sample_input.size() == 0 || sample_output.size() == 0)
    return;
  if (!memory_ &&
      (cache_request.size() == 0 || cache_response.size() == 0 ||
       upload_request.size() == 0 || upload_response.size() == 0))
    return;
  while (true) {
    const PipelineTxn txn = sample_input->read();
    PipelineState state = LoadPipelineState(pool_, txn.state);
    RequireStage(state.stage, PipelineStage::kFragmentTexturePending, name());
    if (memory_ && state.memory_mode != memory_->mode())
      throw std::runtime_error("TextureUnit memory mode mismatch");
    if (!IsTextureFamily(state.functional_case) ||
        !HasPoolHandle(state.texture_sample_requests) ||
        !HasPoolHandle(state.texture_resources) ||
        !HasPoolHandle(state.sampler_states) ||
        !HasPoolHandle(state.fragment_shared_registers)) {
      throw std::runtime_error(
          "TextureUnit received an invalid texture sample batch");
    }
    const std::vector<TextureSampleRequest> requests =
        LoadArray<TextureSampleRequest>(pool_, state.texture_sample_requests);
    const std::vector<TextureResource> resources =
        LoadArray<TextureResource>(pool_, state.texture_resources);
    const std::vector<SamplerState> samplers =
        LoadArray<SamplerState>(pool_, state.sampler_states);
    const std::vector<std::uint32_t> shared =
        LoadArray<std::uint32_t>(pool_, state.fragment_shared_registers);
    if (requests.empty() || requests.size() != state.fragment_shader_lane_count ||
        resources.size() != 1 || samplers.size() != 1 ||
        shared.size() != kFillTexNearestSharedDwordCount) {
      throw std::runtime_error("TextureUnit resource/request count mismatch");
    }
    const TextureResource &resource = resources[0];
    const SamplerState &sampler = samplers[0];
    std::array<std::uint32_t, 4> image_words{};
    std::array<std::uint32_t, 4> sampler_words{};
    std::copy_n(shared.begin(), image_words.size(), image_words.begin());
    std::copy_n(shared.begin() + 8, sampler_words.size(),
                sampler_words.begin());
    const RogueTextureImageDescriptor image =
        DecodeRogueTextureImageDescriptor(image_words);
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
        sampler.normalized_coordinates !=
            decoded_sampler.normalized_coordinates ||
        sampler.base_mip_level != 0 || sampler.reserved != 0) {
      throw std::runtime_error(
          "TextureUnit structured state disagrees with raw descriptor");
    }

    std::uint64_t expected_offset = 0;
    std::uint32_t expected_width = image.width;
    std::uint32_t expected_height = image.height;
    for (std::uint32_t level = 0; level < image.mip_count; ++level) {
      const std::uint32_t expected_pitch =
          level == 0 ? image.row_pitch_bytes : expected_width * 4U;
      const TextureMipLevel &structured_mip = resource.mip[level];
      if (structured_mip.width != expected_width ||
          structured_mip.height != expected_height ||
          structured_mip.row_pitch_bytes != expected_pitch ||
          structured_mip.offset_bytes != expected_offset) {
        throw std::runtime_error(
            "TextureUnit structured mip layout disagrees with raw image");
      }
      expected_offset +=
          static_cast<std::uint64_t>(expected_pitch) * expected_height;
      if (expected_offset > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("TextureUnit mip allocation overflow");
      expected_width = std::max<std::uint32_t>(1U, expected_width >> 1U);
      expected_height = std::max<std::uint32_t>(1U, expected_height >> 1U);
    }
    if (expected_offset != resource.byte_size)
      throw std::runtime_error(
          "TextureUnit allocation size disagrees with raw mip state");

    const bool linear_filter =
        decoded_sampler.min_filter == TextureFilter::kLinear;
    const bool mip_linear =
        decoded_sampler.mip_filter == TextureFilter::kLinear;
    const float linear_coordinate_round_threshold =
        state.functional_case == FunctionalCase::kFillTexTrilinearLinear04
            ? kGlbenchTrilinear04CoordinateRoundThreshold
            : kDefaultLinearCoordinateRoundThreshold;

    // LODM=NORMAL is a quad operation, not four unrelated scalar requests.
    // Preserve the PDS/USC spatial identity and compute one derivative result
    // for each architectural lane quartet, including helper lanes.
    std::vector<TextureImplicitLod> implicit_lods(requests.size());
    if (mip_linear) {
      if (requests.size() % 4U != 0U)
        throw std::runtime_error(
            "TextureUnit mip-linear request batch is not quad aligned");
      for (std::size_t first = 0; first < requests.size(); first += 4U) {
        std::array<std::array<float, 2>, 4> coordinates{};
        const std::uint32_t quad_id = requests[first].quad_id;
        for (std::size_t lane = 0; lane < 4U; ++lane) {
          const TextureSampleRequest &request = requests[first + lane];
          if (request.quad_id != quad_id || request.quad_lane != lane)
            throw std::runtime_error(
                "TextureUnit mip-linear request lost 2x2 quad identity");
          coordinates[lane][0] = BitsFloat(request.coordinates[0]);
          coordinates[lane][1] = BitsFloat(request.coordinates[1]);
        }
        const TextureImplicitLod lod =
            ComputeTextureImplicitLod(coordinates, image, decoded_sampler);
        for (std::size_t lane = 0; lane < 4U; ++lane)
          implicit_lods[first + lane] = lod;
      }
    }

    MemoryAccessStats memory_stats;
    if (!texture_preloaded_) {
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
      texture_preloaded_ = true;
      preloaded_address_ = resource.gpu_address;
      preloaded_bytes_ = resource.byte_size;
    } else if (image.gpu_address != preloaded_address_ ||
               resource.byte_size != preloaded_bytes_) {
      throw std::runtime_error(
          "TextureUnit pre-resident texture allocation changed without "
          "cache-coherent invalidation");
    }

    std::vector<TextureSampleResponse> responses;
    responses.reserve(requests.size());
    std::uint64_t texel_fetch_count = 0;
    for (std::size_t index = 0; index < requests.size(); ++index) {
      const TextureSampleRequest &request = requests[index];
      if (request.shader_lane_index != index || request.request_id != index ||
          request.coordinate_count != 2 || request.component_count != 4 ||
          request.descriptor_set != 0 || request.binding != 0 ||
          request.dimension != 2 || request.normalized != 1 ||
          request.data_request != 0 || request.quad_lane > 3U ||
          request.reserved != 0) {
        throw std::runtime_error("TextureUnit SMP request ABI mismatch");
      }
      for (std::size_t dword = 0; dword < 4; ++dword) {
        if (request.texture_state[dword] != shared[dword] ||
            request.sampler_state[dword] != shared[8U + dword]) {
          throw std::runtime_error("TextureUnit SMP descriptor state mismatch");
        }
      }
      const auto read_texel = [&](const TextureMipLevel &mip,
                                  std::uint32_t x, std::uint32_t y,
                                  std::uint64_t memory_request_id) {
        const std::uint64_t texel_offset =
            static_cast<std::uint64_t>(mip.offset_bytes) +
            static_cast<std::uint64_t>(y) * mip.row_pitch_bytes +
            static_cast<std::uint64_t>(x) * 4U;
        if (texel_offset > resource.byte_size - 4U ||
            texel_offset > std::numeric_limits<std::uint64_t>::max() -
                               resource.gpu_address) {
          throw std::runtime_error(
              "TextureUnit texel address is out of range");
        }
        const std::uint64_t texel_address = image.gpu_address + texel_offset;
        std::vector<std::uint8_t> payload;
        if (memory_) {
          MemoryReadResult read = memory_->Read(
              texel_address, 4, MemoryClient::kTextureCache);
          payload = std::move(read.data);
          memory_stats += read.stats;
        } else {
          MemoryTxn memory_request;
          memory_request.pipeline = txn;
          memory_request.address = texel_address;
          memory_request.bytes = 4;
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
              memory_response.bytes != 4 ||
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
        if (payload.size() != 4)
          throw std::runtime_error("TextureUnit TCU texel size mismatch");
        if (texel_fetch_count == std::numeric_limits<std::uint64_t>::max())
          throw std::overflow_error("TextureUnit texel fetch overflow");
        ++texel_fetch_count;
        std::array<std::uint8_t, 4> texel{};
        std::copy(payload.begin(), payload.end(), texel.begin());
        return texel;
      };

      const auto sample_bilinear =
          [&](const TextureMipLevel &mip,
              std::uint64_t first_request_id) {
        const TextureLinearAxis x = ComputeTextureLinearRepeat(
            BitsFloat(request.coordinates[0]), mip.width,
            decoded_sampler.wrap_u, linear_coordinate_round_threshold);
        const TextureLinearAxis y = ComputeTextureLinearRepeat(
            BitsFloat(request.coordinates[1]), mip.height,
            decoded_sampler.wrap_v, linear_coordinate_round_threshold);
        const std::array<std::uint8_t, 4> texel00 =
            read_texel(mip, x.lower, y.lower, first_request_id + 0U);
        const std::array<std::uint8_t, 4> texel10 =
            read_texel(mip, x.upper, y.lower, first_request_id + 1U);
        const std::array<std::uint8_t, 4> texel01 =
            read_texel(mip, x.lower, y.upper, first_request_id + 2U);
        const std::array<std::uint8_t, 4> texel11 =
            read_texel(mip, x.upper, y.upper, first_request_id + 3U);
        std::array<std::uint8_t, 4> result{};
        for (std::size_t component = 0; component < result.size();
             ++component) {
          const std::uint8_t lower =
              LerpTextureUnorm8(texel00[component], texel10[component],
                                x.weight);
          const std::uint8_t upper =
              LerpTextureUnorm8(texel01[component], texel11[component],
                                x.weight);
          result[component] =
              LerpTextureUnorm8(lower, upper, y.weight);
        }
        return result;
      };

      std::array<float, 4> filtered{};
      if (!linear_filter) {
        const TextureMipLevel &mip = resource.mip[0];
        const std::uint32_t x =
            NearestRepeat(BitsFloat(request.coordinates[0]), mip.width,
                          decoded_sampler.wrap_u);
        const std::uint32_t y =
            NearestRepeat(BitsFloat(request.coordinates[1]), mip.height,
                          decoded_sampler.wrap_v);
        const std::array<std::uint8_t, 4> texel =
            read_texel(mip, x, y, request.request_id);
        for (std::size_t component = 0; component < 4; ++component)
          filtered[component] = static_cast<float>(texel[component]) / 255.0F;
      } else if (!mip_linear) {
        if (request.request_id >
            (std::numeric_limits<std::uint64_t>::max() - 3U) / 4U) {
          throw std::overflow_error("TextureUnit bilinear request ID overflow");
        }
        const std::uint64_t first_request_id = request.request_id * 4U;
        const std::array<std::uint8_t, 4> texel =
            sample_bilinear(resource.mip[0], first_request_id);
        for (std::size_t component = 0; component < 4; ++component) {
          filtered[component] = static_cast<float>(texel[component]) / 255.0F;
        }
      } else {
        if (request.request_id >
            (std::numeric_limits<std::uint64_t>::max() - 7U) / 8U) {
          throw std::overflow_error("TextureUnit trilinear request ID overflow");
        }
        const TextureImplicitLod &lod = implicit_lods[index];
        const std::uint64_t first_request_id = request.request_id * 8U;
        const std::array<std::uint8_t, 4> lower = sample_bilinear(
            resource.mip[lod.level0], first_request_id + 0U);
        const std::array<std::uint8_t, 4> upper = sample_bilinear(
            resource.mip[lod.level1], first_request_id + 4U);
        for (std::size_t component = 0; component < 4; ++component) {
          const std::uint8_t texel = LerpTextureUnorm8(
              lower[component], upper[component], lod.mip_weight_u8);
          filtered[component] = static_cast<float>(texel) / 255.0F;
        }
      }
      TextureSampleResponse response;
      response.shader_lane_index = request.shader_lane_index;
      response.request_id = request.request_id;
      for (std::size_t component = 0; component < 4; ++component)
        response.rgba[component] = FloatBits(filtered[component]);
      responses.push_back(response);
    }

    state = LoadPipelineState(pool_, txn.state);
    RequireStage(state.stage, PipelineStage::kFragmentTexturePending, name());
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
    state.counters.renderer_cycles += cycles;
    state.stage = PipelineStage::kTextureSamplesReady;
    WaitForCycles(cycles);
    StorePipelineState(pool_, txn.state, state);
    sample_output->write(txn);
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
    if (!HasPoolHandle(state.fragment_outputs) ||
        state.fragment_program_summary.pixel_output_mask != 0x0f) {
      throw std::runtime_error(
          "solid-color texture bypass received no complete USC pixel outputs");
    }
    const bool texture_case = IsTextureFamily(state.functional_case);
    if (!texture_case && (state.counters.texture_requests != 0 ||
        state.counters.texel_fetches != 0)) {
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
      if (!HasPoolHandle(state.fragment_shared_registers))
        throw std::runtime_error(
            "TextureUnit completed without live sampler descriptor state");
      const std::vector<std::uint32_t> shared = LoadArray<std::uint32_t>(
          pool_, state.fragment_shared_registers);
      if (shared.size() != kFillTexNearestSharedDwordCount)
        throw std::runtime_error(
            "TextureUnit completed with invalid sampler descriptor size");
      std::array<std::uint32_t, 4> sampler_words{};
      std::copy_n(shared.begin() + 8, sampler_words.size(),
                  sampler_words.begin());
      const RogueTextureSamplerDescriptor sampler =
          DecodeRogueTextureSamplerDescriptor(sampler_words);
      const std::uint64_t fetches_per_request =
          sampler.mip_filter == TextureFilter::kLinear
              ? 8U
              : sampler.min_filter == TextureFilter::kLinear ? 4U : 1U;
      if (state.counters.texture_requests == 0 ||
          state.counters.texture_requests >
              std::numeric_limits<std::uint64_t>::max() /
                  fetches_per_request ||
          state.counters.texel_fetches !=
              state.counters.texture_requests * fetches_per_request) {
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
