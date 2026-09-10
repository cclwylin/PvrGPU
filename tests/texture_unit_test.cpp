// Functional unit test for TextureUnit/TPU (Texture Processing Unit).
// It pins the public Rogue IMAGE/SAMPLER descriptor words, negative and edge
// repeat coordinates, the selected signed-delta 8-bit round-to-nearest-even
// filter, Gates 18-20 implicit LOD state, and actual event-driven
// four/eight-tap FIFO transactions.  The cache responders return individual
// texels so tap identity, mip selection, and interpolation order cannot be
// replaced by a fixed result.

#include "common/glbench_texture_fixture.h"
#include "common/pipeline_state.h"
#include "memory/gpu_memory_system.h"
#include "texture/texture_unit.h"

#include <systemc>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Rgba8 = std::array<std::uint8_t, 4>;

using pvrgpu::stub::DecodeRogueTextureImageDescriptor;
using pvrgpu::stub::DecodeRogueTextureSamplerDescriptor;
using pvrgpu::stub::DriverPcoTextureDescriptorClassSupported;
using pvrgpu::stub::ComputeTextureImplicitLod;
using pvrgpu::stub::ComputeTextureLinearRepeat;
using pvrgpu::stub::FunctionalCase;
using pvrgpu::stub::GlbenchFillTextureFixture;
using pvrgpu::stub::GpuMemorySystem;
using pvrgpu::stub::HasPoolHandle;
using pvrgpu::stub::LerpTextureUnorm8;
using pvrgpu::stub::LoadArray;
using pvrgpu::stub::LoadPipelineState;
using pvrgpu::stub::MakeGlbenchFillTextureFixture;
using pvrgpu::stub::MemoryClient;
using pvrgpu::stub::MemoryOperation;
using pvrgpu::stub::MemoryPayloadFormat;
using pvrgpu::stub::MemoryPool;
using pvrgpu::stub::MemoryTxn;
using pvrgpu::stub::PipelineStage;
using pvrgpu::stub::PipelineState;
using pvrgpu::stub::PipelineTxn;
using pvrgpu::stub::ReleaseFunctionalPayloads;
using pvrgpu::stub::RogueTextureImageDescriptor;
using pvrgpu::stub::RogueTextureSamplerDescriptor;
using pvrgpu::stub::StoreNewArray;
using pvrgpu::stub::StorePipelineState;
using pvrgpu::stub::TextureFilter;
using pvrgpu::stub::TextureFormat;
using pvrgpu::stub::TextureImplicitLod;
using pvrgpu::stub::TextureLinearAxis;
using pvrgpu::stub::TextureResource;
using pvrgpu::stub::TextureSampleRequest;
using pvrgpu::stub::TextureSampleResponse;
using pvrgpu::stub::TextureUnit;
using pvrgpu::stub::TextureWrapMode;
using pvrgpu::stub::UsesTextureSampling;

void Check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error("texture unit test failed: " + message);
}

template <typename Function>
void ExpectFailure(Function &&function, const std::string &description) {
  try {
    function();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error("expected descriptor failure: " + description);
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float BitsFloat(std::uint32_t bits) {
  float value = 0.0F;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint64_t DescriptorWord(
    const std::array<std::uint32_t,
                     pvrgpu::stub::kFillTexNearestSharedDwordCount> &shared,
    std::size_t first_dword) {
  return static_cast<std::uint64_t>(shared.at(first_dword)) |
         (static_cast<std::uint64_t>(shared.at(first_dword + 1U)) << 32U);
}

std::array<std::uint32_t, 4> DescriptorDwords(
    const GlbenchFillTextureFixture &fixture, std::size_t first_dword) {
  std::array<std::uint32_t, 4> words{};
  std::copy_n(fixture.fragment_shared.begin() + first_dword, words.size(),
              words.begin());
  return words;
}

void CheckDescriptorAndArithmetic() {
  const GlbenchFillTextureFixture nearest =
      MakeGlbenchFillTextureFixture(TextureFilter::kNearest);
  const GlbenchFillTextureFixture linear =
      MakeGlbenchFillTextureFixture(TextureFilter::kLinear);
  const GlbenchFillTextureFixture trilinear =
      MakeGlbenchFillTextureFixture(
          FunctionalCase::kFillTexTrilinearLinear01);
  const GlbenchFillTextureFixture trilinear04 =
      MakeGlbenchFillTextureFixture(
          FunctionalCase::kFillTexTrilinearLinear04);
  const GlbenchFillTextureFixture trilinear05 =
      MakeGlbenchFillTextureFixture(
          FunctionalCase::kFillTexTrilinearLinear05);

  Check(DescriptorWord(nearest.fragment_shared, 8) ==
            UINT64_C(0x0000000000000fff),
        "nearest literal sampler word with LOD0 clamp");
  Check(DescriptorWord(linear.fragment_shared, 8) ==
            UINT64_C(0x0000005000000fff),
        "linear literal sampler word with LOD0 clamp");
  Check(DescriptorWord(trilinear.fragment_shared, 8) ==
            UINT64_C(0x00000151df800fff),
        "Gate 18 trilinear literal sampler word");
  Check(trilinear04.vertex_scale_bits == UINT32_C(0x3f420c4a) &&
            DescriptorWord(trilinear04.fragment_shared, 8) ==
                UINT64_C(0x00000151df800fff),
        "Gate 19 live 0.758f SH0 and shared trilinear sampler word");
  Check(trilinear05.vertex_scale_bits == UINT32_C(0x3f350481) &&
            DescriptorWord(trilinear05.fragment_shared, 8) ==
                UINT64_C(0x00000151df800fff),
        "Gate 20 live 0.7071f SH0 and shared trilinear sampler word");

  const RogueTextureImageDescriptor image =
      DecodeRogueTextureImageDescriptor(DescriptorDwords(linear, 0));
  const RogueTextureSamplerDescriptor sampler =
      DecodeRogueTextureSamplerDescriptor(DescriptorDwords(linear, 8));
  Check(image.gpu_address == pvrgpu::stub::kGlbenchTextureGpuAddress &&
            image.width == 512 && image.height == 512 &&
            image.row_pitch_bytes == 2048 && image.mip_count == 10,
        "raw image address/extent/stride/mip decode");
  Check(sampler.min_filter == TextureFilter::kLinear &&
            sampler.mag_filter == TextureFilter::kLinear &&
            sampler.mip_filter == TextureFilter::kNearest &&
            sampler.min_lod_u4_6 == 0 && sampler.max_lod_u4_6 == 0 &&
            sampler.normalized_coordinates == 1,
        "raw sampler filter/normalized/LOD decode");

  auto single_mip_image_words = DescriptorDwords(nearest, 0);
  single_mip_image_words[2] &= ~UINT32_C(0x7fff);
  single_mip_image_words[2] |= UINT32_C(511); // 512 texels, public stride unit
  single_mip_image_words[2] &= ~(UINT32_C(1) << 15U);
  single_mip_image_words[3] &= UINT32_C(0x0fffffff);
  single_mip_image_words[3] |= UINT32_C(1) << (60U - 32U);
  const RogueTextureImageDescriptor single_mip_image =
      DecodeRogueTextureImageDescriptor(single_mip_image_words);
  Check(single_mip_image.mip_count == 1 &&
            single_mip_image.row_pitch_bytes == 2048,
        "single-mip public texel-stride image descriptor is accepted");

  auto rgbx_image_words = single_mip_image_words;
  // Rogue format14 is U10U10U10U2; the raw descriptor, not a host-side
  // channel guess, selects RGB or BGR from the same four-byte packed word.
  for (const bool bgr : {false, true}) {
    auto packed_words = single_mip_image_words;
    std::uint64_t word0 = static_cast<std::uint64_t>(packed_words[0]) |
        (static_cast<std::uint64_t>(packed_words[1]) << 32U);
    word0 &= ~((UINT64_C(127) << 27U) | (UINT64_C(0xfff) << 5U));
    word0 |= (UINT64_C(14) << 27U) | (UINT64_C(3) << 5U) |
        (static_cast<std::uint64_t>(bgr ? 0 : 2) << 8U) |
        (UINT64_C(1) << 11U) |
        (static_cast<std::uint64_t>(bgr ? 2 : 0) << 14U);
    packed_words[0] = static_cast<std::uint32_t>(word0);
    packed_words[1] = static_cast<std::uint32_t>(word0 >> 32U);
    const auto packed_image = DecodeRogueTextureImageDescriptor(packed_words);
    Check(packed_image.format == (bgr ? TextureFormat::kBgr10A2Unorm :
                                       TextureFormat::kRgb10A2Unorm) &&
          packed_image.row_pitch_bytes == 2048 && packed_image.mip_count == 1,
          "packed ten-bit raw descriptor swizzle and storage width");
  }
  rgbx_image_words[0] &= ~(UINT32_C(7) << 5U);
  rgbx_image_words[0] |= UINT32_C(4) << 5U; // SWIZ3=SRC_ONE.
  const RogueTextureImageDescriptor rgbx_image =
      DecodeRogueTextureImageDescriptor(rgbx_image_words);
  Check(rgbx_image.format == TextureFormat::kRgbx8Unorm &&
            rgbx_image.gpu_address ==
                pvrgpu::stub::kGlbenchTextureGpuAddress &&
            rgbx_image.mip_count == 1,
        "driver RGBX image descriptor selects hardware alpha ONE");

  auto maximum_image_words = single_mip_image_words;
  maximum_image_words[1] &= ~((UINT32_C(0x3fff) << 2U) |
                              (UINT32_C(0x3fff) << 16U));
  maximum_image_words[1] |= UINT32_C(0x3fff) << 2U;
  maximum_image_words[1] |= UINT32_C(0x3fff) << 16U;
  maximum_image_words[2] &= ~UINT32_C(0x7fff);
  maximum_image_words[2] |= UINT32_C(0x3fff);
  const RogueTextureImageDescriptor maximum_image =
      DecodeRogueTextureImageDescriptor(maximum_image_words);
  Check(maximum_image.width == 16384 && maximum_image.height == 16384 &&
            maximum_image.row_pitch_bytes == 65536 &&
            maximum_image.mip_count == 1,
        "maximum 16384x16384 single-mip descriptor is accepted");

  auto clamp_sampler_words = DescriptorDwords(nearest, 8);
  clamp_sampler_words[1] |= UINT32_C(2) << (33U - 32U);
  clamp_sampler_words[1] |= UINT32_C(2) << (41U - 32U);
  const RogueTextureSamplerDescriptor clamp_sampler =
      DecodeRogueTextureSamplerDescriptor(clamp_sampler_words);
  Check(clamp_sampler.wrap_u == TextureWrapMode::kClampToEdge &&
            clamp_sampler.wrap_v == TextureWrapMode::kClampToEdge,
        "normalized clamp-to-edge sampler is accepted");

  const RogueTextureImageDescriptor trilinear_image =
      DecodeRogueTextureImageDescriptor(DescriptorDwords(trilinear, 0));
  const RogueTextureSamplerDescriptor trilinear_sampler =
      DecodeRogueTextureSamplerDescriptor(DescriptorDwords(trilinear, 8));
  Check(trilinear_sampler.min_filter == TextureFilter::kLinear &&
            trilinear_sampler.mag_filter == TextureFilter::kLinear &&
            trilinear_sampler.mip_filter == TextureFilter::kLinear &&
            trilinear_sampler.min_lod_u4_6 == 0 &&
            trilinear_sampler.max_lod_u4_6 == 959 &&
            trilinear_sampler.normalized_coordinates == 1,
        "Gate 18 raw sampler enables implicit trilinear filtering");
  for (const std::uint16_t sequence_max_lod :
       {384U, 448U, 512U, 576U, 640U}) {
    auto terrain_sampler_words = DescriptorDwords(trilinear, 8);
    std::uint64_t terrain_word0 =
        static_cast<std::uint64_t>(terrain_sampler_words[0]) |
        (static_cast<std::uint64_t>(terrain_sampler_words[1]) << 32U);
    terrain_word0 &= ~(UINT64_C(0x3ff) << 23U);
    terrain_word0 |= static_cast<std::uint64_t>(sequence_max_lod) << 23U;
    terrain_sampler_words[0] = static_cast<std::uint32_t>(terrain_word0);
    terrain_sampler_words[1] =
        static_cast<std::uint32_t>(terrain_word0 >> 32U);
    const RogueTextureSamplerDescriptor terrain_sampler =
        DecodeRogueTextureSamplerDescriptor(terrain_sampler_words);
    Check(terrain_sampler.max_lod_u4_6 == sequence_max_lod &&
              terrain_sampler.wrap_u == TextureWrapMode::kRepeat &&
              terrain_sampler.wrap_v == TextureWrapMode::kRepeat,
          "bounded sequence mip LOD and repeat sampler are accepted");
  }
  /*
   * LOD is U4.6 fixed point, so 385 is 6.015625 -- a value GL_TEXTURE_MAX_LOD
   * can hold and the sampler can express.  Refusing it was an artifact of the
   * captures only ever containing whole levels, not a limit.  What the decode
   * still has to refuse is a window that runs backwards; whether a window
   * fits the image is the image-and-sampler class's question, checked above.
   */
  auto fractional_lod_words = DescriptorDwords(trilinear, 8);
  std::uint64_t fractional_word0 =
      static_cast<std::uint64_t>(fractional_lod_words[0]) |
      (static_cast<std::uint64_t>(fractional_lod_words[1]) << 32U);
  fractional_word0 &= ~(UINT64_C(0x3ff) << 23U);
  fractional_word0 |= UINT64_C(385) << 23U;
  fractional_lod_words[0] = static_cast<std::uint32_t>(fractional_word0);
  fractional_lod_words[1] = static_cast<std::uint32_t>(fractional_word0 >> 32U);
  Check(DecodeRogueTextureSamplerDescriptor(fractional_lod_words)
                .max_lod_u4_6 == 385,
        "a fractional mip LOD decodes");

  auto inverted_lod_words = DescriptorDwords(trilinear, 8);
  std::uint64_t inverted_word0 =
      static_cast<std::uint64_t>(inverted_lod_words[0]) |
      (static_cast<std::uint64_t>(inverted_lod_words[1]) << 32U);
  inverted_word0 &= ~(UINT64_C(0x3ff) << 23U);   // max lod -> 64
  inverted_word0 |= UINT64_C(64) << 23U;
  inverted_word0 &= ~(UINT64_C(0x3ff) << 13U);   // min lod -> 128
  inverted_word0 |= UINT64_C(128) << 13U;
  inverted_lod_words[0] = static_cast<std::uint32_t>(inverted_word0);
  inverted_lod_words[1] = static_cast<std::uint32_t>(inverted_word0 >> 32U);
  ExpectFailure(
      [&] { DecodeRogueTextureSamplerDescriptor(inverted_lod_words); },
      "an inverted mip LOD window");

  /*
   * 704 is eleven levels.  The old ceiling here was the largest LOD the
   * captured workloads used, which says nothing about what the field or the
   * sampler can express -- an image with twelve levels wants exactly this.
   * Whether the image has them is checked against the image, not here.
   */
  auto deep_lod_words = DescriptorDwords(trilinear, 8);
  std::uint64_t deep_lod_word0 =
      static_cast<std::uint64_t>(deep_lod_words[0]) |
      (static_cast<std::uint64_t>(deep_lod_words[1]) << 32U);
  deep_lod_word0 &= ~(UINT64_C(0x3ff) << 23U);
  deep_lod_word0 |= UINT64_C(704) << 23U;
  deep_lod_words[0] = static_cast<std::uint32_t>(deep_lod_word0);
  deep_lod_words[1] = static_cast<std::uint32_t>(deep_lod_word0 >> 32U);
  Check(DecodeRogueTextureSamplerDescriptor(deep_lod_words).max_lod_u4_6 ==
            704,
        "a deep mip LOD decodes");

  // Terrain D3's real external resources are ten-level RGBX8 chains.  Pin
  // the exact address-zero producer words, then relocate IMAGE_WORD1 exactly
  // as Submitter does before decoding the TPU-visible descriptor.
  std::array<std::uint32_t, 4> terrain_d3_image_words = {
      UINT32_C(0x60000a84), UINT32_C(0x01ff07fc),
      UINT32_C(0x000081ff), UINT32_C(0xa0000000)};
  const std::uint64_t terrain_address =
      pvrgpu::stub::kGlbenchTextureGpuAddress;
  std::uint64_t terrain_image_word1 =
      static_cast<std::uint64_t>(terrain_d3_image_words[2]) |
      (static_cast<std::uint64_t>(terrain_d3_image_words[3]) << 32U) |
      ((terrain_address >> 2U) << 16U);
  terrain_d3_image_words[2] =
      static_cast<std::uint32_t>(terrain_image_word1);
  terrain_d3_image_words[3] =
      static_cast<std::uint32_t>(terrain_image_word1 >> 32U);
  const std::array<std::uint32_t, 4> terrain_d3_sampler_words = {
      UINT32_C(0x20000fff), UINT32_C(0x00000151), 0, 0};
  const RogueTextureImageDescriptor terrain_d3_image =
      DecodeRogueTextureImageDescriptor(terrain_d3_image_words);
  const RogueTextureSamplerDescriptor terrain_d3_sampler =
      DecodeRogueTextureSamplerDescriptor(terrain_d3_sampler_words);
  Check(terrain_d3_image.format == TextureFormat::kRgbx8Unorm &&
            terrain_d3_image.width == 512 &&
            terrain_d3_image.height == 512 &&
            terrain_d3_image.mip_count == 10 &&
            terrain_d3_sampler.max_lod_u4_6 == 576 &&
            DriverPcoTextureDescriptorClassSupported(
                terrain_d3_image, terrain_d3_sampler, 5),
        "Terrain D3 real RGBX8 mip-linear descriptor class");
  /*
   * The LOD window is the sampler's, not the image's.  A sampler may
   * address fewer levels than the image carries -- GL_TEXTURE_MAX_LOD, or a
   * non-mipmapping filter on a mipmapped texture -- and it may reach past
   * the last level: the level clamp (lp_build_nearest_mip_level) answers
   * that, and a window of 0..0.25 is how the driver says "base level only"
   * for GL's mip filter NONE without losing the minification decision.
   */
  RogueTextureSamplerDescriptor clamped_terrain_lod = terrain_d3_sampler;
  clamped_terrain_lod.max_lod_u4_6 = 512;
  Check(DriverPcoTextureDescriptorClassSupported(
            terrain_d3_image, clamped_terrain_lod, 5),
        "Terrain D3 accepts a LOD window inside the image");
  RogueTextureSamplerDescriptor wide_terrain_lod = terrain_d3_sampler;
  wide_terrain_lod.max_lod_u4_6 = 640;
  Check(DriverPcoTextureDescriptorClassSupported(
            terrain_d3_image, wide_terrain_lod, 5),
        "Terrain D3 accepts a LOD window past the last level: the level clamps");
  RogueTextureSamplerDescriptor backwards_terrain_lod = terrain_d3_sampler;
  backwards_terrain_lod.min_lod_u4_6 = 640;
  Check(!DriverPcoTextureDescriptorClassSupported(
            terrain_d3_image, backwards_terrain_lod, 5),
        "Terrain D3 refuses a LOD window that runs backwards");
  RogueTextureImageDescriptor mipped_depth = terrain_d3_image;
  mipped_depth.format = TextureFormat::kZ32Unorm;
  Check(!DriverPcoTextureDescriptorClassSupported(
            mipped_depth, terrain_d3_sampler, 5),
        "Terrain D3 mipped-depth near mutation remains fail-closed");

  // Terrain 800 D3's first fragment SMP reaches a finite helper quad whose
  // four detail coordinates are bit-identical.  Its exact zero derivative is
  // a valid limiting case and selects the sampler's minimum LOD.
  constexpr std::uint32_t kTerrainD3DegenerateU = UINT32_C(0x3ff8a000);
  constexpr std::uint32_t kTerrainD3DegenerateV = UINT32_C(0x401c4000);
  const std::array<std::array<float, 2>, 4> terrain_d3_zero_rho = {{
      {{BitsFloat(kTerrainD3DegenerateU),
        BitsFloat(kTerrainD3DegenerateV)}},
      {{BitsFloat(kTerrainD3DegenerateU),
        BitsFloat(kTerrainD3DegenerateV)}},
      {{BitsFloat(kTerrainD3DegenerateU),
        BitsFloat(kTerrainD3DegenerateV)}},
      {{BitsFloat(kTerrainD3DegenerateU),
        BitsFloat(kTerrainD3DegenerateV)}},
  }};
  const TextureImplicitLod terrain_d3_zero_lod = ComputeTextureImplicitLod(
      terrain_d3_zero_rho, terrain_d3_image, terrain_d3_sampler);
  Check(FloatBits(terrain_d3_zero_lod.dsdx) == 0 &&
            FloatBits(terrain_d3_zero_lod.dtdx) == 0 &&
            FloatBits(terrain_d3_zero_lod.dsdy) == 0 &&
            FloatBits(terrain_d3_zero_lod.dtdy) == 0 &&
            FloatBits(terrain_d3_zero_lod.rho_squared) == 0 &&
            FloatBits(terrain_d3_zero_lod.lambda) == 0 &&
            terrain_d3_zero_lod.level0 == 0 &&
            terrain_d3_zero_lod.level1 == 1 &&
            terrain_d3_zero_lod.mip_weight_u8 == 0 &&
            FloatBits(terrain_d3_zero_lod.mip_weight) == 0,
        "Terrain D3 finite zero-rho quad selects minimum LOD");

  const float maximum_finite = std::numeric_limits<float>::max();
  const std::array<std::array<float, 2>, 4> overflowing_derivatives = {{
      {{0.0F, 0.0F}},
      {{maximum_finite, 0.0F}},
      {{0.0F, maximum_finite}},
      {{maximum_finite, maximum_finite}},
  }};
  ExpectFailure(
      [&] {
        ComputeTextureImplicitLod(overflowing_derivatives, terrain_d3_image,
                                  terrain_d3_sampler);
      },
      "Terrain D3 non-finite implicit derivative remains fail-closed");

  // One screen-pixel derivative across the 64x64 Gate 18 quad after its
  // float32 0.933 vertex scale.  The LODM=NORMAL datapath must expose mip
  // levels 3/4 and architectural TFRAC=25; the float datapath's weight is the
  // unquantized fraction TFRAC was truncated from.
  const float gate18_scale = BitsFloat(trilinear.vertex_scale_bits);
  const float gate18_derivative = 1.0F / (64.0F * gate18_scale);
  constexpr float kGate18U = 0.25F;
  constexpr float kGate18V = 0.375F;
  const std::array<std::array<float, 2>, 4> gate18_coordinates = {{
      {{kGate18U, kGate18V}},
      {{kGate18U + gate18_derivative, kGate18V}},
      {{kGate18U, kGate18V + gate18_derivative}},
      {{kGate18U + gate18_derivative, kGate18V + gate18_derivative}},
  }};
  const TextureImplicitLod gate18_lod = ComputeTextureImplicitLod(
      gate18_coordinates, trilinear_image, trilinear_sampler);
  Check(gate18_lod.level0 == 3 && gate18_lod.level1 == 4 &&
            gate18_lod.mip_weight_u8 == 25 &&
            gate18_lod.mip_weight >= 25.0F / 256.0F &&
            gate18_lod.mip_weight < 26.0F / 256.0F &&
            gate18_lod.lambda >= 3.0F && gate18_lod.lambda < 4.0F,
        "Gate 18 implicit LOD levels and TFRAC");

  // Gate 19 keeps the descriptor and shaders byte-identical while changing
  // only live SH0 to 0.758f.  Its larger implicit derivative remains between
  // mip levels 3/4 but quantizes to architectural TFRAC=102.
  const RogueTextureImageDescriptor trilinear04_image =
      DecodeRogueTextureImageDescriptor(DescriptorDwords(trilinear04, 0));
  const RogueTextureSamplerDescriptor trilinear04_sampler =
      DecodeRogueTextureSamplerDescriptor(DescriptorDwords(trilinear04, 8));
  const float gate19_scale = BitsFloat(trilinear04.vertex_scale_bits);
  const float gate19_derivative = 1.0F / (64.0F * gate19_scale);
  const std::array<std::array<float, 2>, 4> gate19_coordinates = {{
      {{kGate18U, kGate18V}},
      {{kGate18U + gate19_derivative, kGate18V}},
      {{kGate18U, kGate18V + gate19_derivative}},
      {{kGate18U + gate19_derivative, kGate18V + gate19_derivative}},
  }};
  const TextureImplicitLod gate19_lod = ComputeTextureImplicitLod(
      gate19_coordinates, trilinear04_image, trilinear04_sampler);
  Check(gate19_lod.level0 == 3 && gate19_lod.level1 == 4 &&
            gate19_lod.mip_weight_u8 == 102 &&
            gate19_lod.mip_weight >= 102.0F / 256.0F &&
            gate19_lod.mip_weight < 103.0F / 256.0F &&
            gate19_lod.lambda >= 3.0F && gate19_lod.lambda < 4.0F,
        "Gate 19 implicit LOD levels and TFRAC");

  // Gate 20's exact 0.7071f scale reaches the mip half-way boundary.  Pin the
  // ideal float32 derivative, the lambda after storing coordinates at the
  // chosen base point, and the architectural U8 TFRAC=0x80 -- the one gate
  // whose TFRAC an exact LOD leaves where the piecewise-linear one had it,
  // because half way is a value both resolve exactly.
  const RogueTextureImageDescriptor trilinear05_image =
      DecodeRogueTextureImageDescriptor(DescriptorDwords(trilinear05, 0));
  const RogueTextureSamplerDescriptor trilinear05_sampler =
      DecodeRogueTextureSamplerDescriptor(DescriptorDwords(trilinear05, 8));
  const float gate20_scale = BitsFloat(trilinear05.vertex_scale_bits);
  const float gate20_derivative = 1.0F / (64.0F * gate20_scale);
  const std::array<std::array<float, 2>, 4> gate20_coordinates = {{
      {{kGate18U, kGate18V}},
      {{kGate18U + gate20_derivative, kGate18V}},
      {{kGate18U, kGate18V + gate20_derivative}},
      {{kGate18U + gate20_derivative, kGate18V + gate20_derivative}},
  }};
  const TextureImplicitLod gate20_lod = ComputeTextureImplicitLod(
      gate20_coordinates, trilinear05_image, trilinear05_sampler);
  Check(FloatBits(gate20_derivative) == UINT32_C(0x3cb50565) &&
            gate20_lod.level0 == 3 && gate20_lod.level1 == 4 &&
            gate20_lod.mip_weight_u8 == 128 &&
            gate20_lod.mip_weight >= static_cast<float>(gate20_lod.mip_weight_u8) / 256.0F &&
            gate20_lod.mip_weight < static_cast<float>(gate20_lod.mip_weight_u8 + 1U) / 256.0F &&
            FloatBits(gate20_lod.lambda) == UINT32_C(0x40600038),
        "Gate 20 implicit LOD levels, approximate lambda, and TFRAC");
  const std::array<std::array<float, 2>, 4> gate20_edge_coordinates = {{
      {{0.5013811588287354F, 0.9764730930328369F}},
      {{0.5041432380676270F, 0.9764730930328369F}},
      {{0.5013811588287354F, 0.9792351722717285F}},
      {{0.5041432380676270F, 0.9792351722717285F}},
  }};
  const TextureImplicitLod gate20_edge_lod = ComputeTextureImplicitLod(
      gate20_edge_coordinates, trilinear05_image, trilinear05_sampler);
  Check(gate20_edge_lod.level0 == 0 && gate20_edge_lod.level1 == 1 &&
            gate20_edge_lod.mip_weight_u8 == 127 &&
            gate20_edge_lod.mip_weight >= static_cast<float>(gate20_edge_lod.mip_weight_u8) / 256.0F &&
            gate20_edge_lod.mip_weight < static_cast<float>(gate20_edge_lod.mip_weight_u8 + 1U) / 256.0F,
        "Gate 20 near-boundary TFRAC uses strict positive truncation");

  // Two exact f16->f32 Refract composite quads formerly rounded upward by a
  // near-integer epsilon. Gallivm uses fptosi(fpart * 256), so 10.99939 and
  // 230.98755 must remain 10 and 230 respectively.
  RogueTextureImageDescriptor refract_image = trilinear_image;
  refract_image.width = 160;
  refract_image.height = 120;
  refract_image.mip_count = 8;
  RogueTextureSamplerDescriptor refract_sampler = trilinear_sampler;
  refract_sampler.min_lod_u4_6 = 0;
  refract_sampler.max_lod_u4_6 = 448;
  const std::array<std::array<float, 2>, 4> refract_coordinates_a = {{
      {{0.01708984375F, 0.9638671875F}},
      {{0.43115234375F, 0.89990234375F}},
      {{-0.07470703125F, 0.9052734375F}},
      {{0.005859375F, 0.96630859375F}},
  }};
  const TextureImplicitLod refract_lod_a = ComputeTextureImplicitLod(
      refract_coordinates_a, refract_image, refract_sampler);
  const float refract_scaled_a =
      (refract_lod_a.lambda - std::floor(refract_lod_a.lambda)) * 256.0F;
  Check(FloatBits(refract_coordinates_a[0][0]) == UINT32_C(0x3c8c0000) &&
            FloatBits(refract_coordinates_a[1][0]) ==
                UINT32_C(0x3edcc000) &&
            FloatBits(refract_lod_a.lambda) == UINT32_C(0x40c1e728) &&
            refract_scaled_a > 15.22F && refract_scaled_a < 15.23F &&
            refract_lod_a.level0 == 6 && refract_lod_a.level1 == 7 &&
            refract_lod_a.mip_weight_u8 == 15 &&
            refract_lod_a.mip_weight >= static_cast<float>(refract_lod_a.mip_weight_u8) / 256.0F &&
            refract_lod_a.mip_weight < static_cast<float>(refract_lod_a.mip_weight_u8 + 1U) / 256.0F,
        "Refract lane 37,45 derives lambda 6.0594673 and TFRAC 15");

  const std::array<std::array<float, 2>, 4> refract_coordinates_b = {{
      {{0.466552734375F, 0.19873046875F}},
      {{0.4892578125F, 0.2080078125F}},
      {{0.467041015625F, 0.224853515625F}},
      {{0.4892578125F, 0.233154296875F}},
  }};
  const TextureImplicitLod refract_lod_b = ComputeTextureImplicitLod(
      refract_coordinates_b, refract_image, refract_sampler);
  const float refract_scaled_b =
      (refract_lod_b.lambda - std::floor(refract_lod_b.lambda)) * 256.0F;
  Check(FloatBits(refract_coordinates_b[0][0]) == UINT32_C(0x3eeee000) &&
            FloatBits(refract_coordinates_b[3][1]) ==
                UINT32_C(0x3e6ec000) &&
            FloatBits(refract_lod_b.lambda) == UINT32_C(0x3ff681c8) &&
            refract_scaled_b > 237.01F && refract_scaled_b < 237.02F &&
            refract_lod_b.level0 == 1 && refract_lod_b.level1 == 2 &&
            refract_lod_b.mip_weight_u8 == 237 &&
            refract_lod_b.mip_weight >= static_cast<float>(refract_lod_b.mip_weight_u8) / 256.0F &&
            refract_lod_b.mip_weight < static_cast<float>(refract_lod_b.mip_weight_u8 + 1U) / 256.0F,
        "Refract lane 38,17 derives lambda 1.9258356 and TFRAC 237");

  /* Golden Gallivm target primitive/quad for the final Refract mismatch:
   * internal lane (57,16), parameter 10301, quad 348:1.  Keep all four
   * same-primitive helper coordinates together; screen-neighbor fragments
   * can belong to overlapping primitives and are not derivative inputs. */
  const std::array<std::array<float, 2>, 4> refract_target_quad = {{
      {{0.86865234375F, -0.224609375F}},
      {{0.73828125F, 0.076904296875F}},
      {{0.9853515625F, -0.28564453125F}},
      {{0.8828125F, 0.07568359375F}},
  }};
  const TextureImplicitLod refract_target_lod = ComputeTextureImplicitLod(
      refract_target_quad, refract_image, refract_sampler);
  Check(FloatBits(refract_target_lod.dsdx) == FloatBits(-20.859375F) &&
            FloatBits(refract_target_lod.dtdx) == FloatBits(36.181640625F) &&
            FloatBits(refract_target_lod.dsdy) == FloatBits(18.671875F) &&
            FloatBits(refract_target_lod.dtdy) == FloatBits(-7.32421875F) &&
            FloatBits(refract_target_lod.rho_squared) ==
                FloatBits(1744.224609375F) &&
            FloatBits(refract_target_lod.lambda) == UINT32_C(0x40ac4b3e) &&
            refract_target_lod.level0 == 5 &&
            refract_target_lod.level1 == 6 &&
            refract_target_lod.mip_weight_u8 == 98,
        "Refract target primitive quad derives golden rho/lambda/TFRAC98");

  /*
   * A non-zero max LOD is a valid encoding; whether it can be sampled depends
   * on how many levels the image has, which is the class predicate's
   * question.  The decode no longer answers it, so this checks both halves:
   * the window decodes, and a window past a single-level image is served.
   */
  auto mutated = DescriptorDwords(linear, 8);
  mutated[0] |= UINT32_C(1) << 23U; // public maxlod[0]
  const RogueTextureSamplerDescriptor non_zero_lod =
      DecodeRogueTextureSamplerDescriptor(mutated);
  Check(non_zero_lod.max_lod_u4_6 == 1, "a non-zero max LOD decodes");
  RogueTextureImageDescriptor one_level_image;
  one_level_image.width = 64;
  one_level_image.height = 64;
  one_level_image.row_pitch_bytes = 64 * 4;
  one_level_image.mip_count = 1;
  one_level_image.format = TextureFormat::kRgba8Unorm;
  Check(DriverPcoTextureDescriptorClassSupported(one_level_image,
                                                non_zero_lod, 1),
        "a LOD window past a single-level image clamps to that level");
  mutated = DescriptorDwords(linear, 8);
  mutated[1] |= UINT32_C(4) << (41U - 32U); // addrmode_v=BORDER
  ExpectFailure([&] { DecodeRogueTextureSamplerDescriptor(mutated); },
                "unsupported clamp-to-border address mode");
  mutated = DescriptorDwords(linear, 8);
  mutated[1] |= UINT32_C(1) << (49U - 32U); // non-normalized coords
  ExpectFailure([&] { DecodeRogueTextureSamplerDescriptor(mutated); },
                "non-normalized coordinates");
  mutated = DescriptorDwords(linear, 8);
  mutated[2] = 1U; // SAMPLER_WORD1 plane/non-seamless state
  ExpectFailure([&] { DecodeRogueTextureSamplerDescriptor(mutated); },
                "unsupported sampler word1");

  /*
   * IMAGE_WORD0 bit 3 is GAMMA and bit 4 is the second half of
   * TWOCOMP_GAMMA.  Gamma on a four-channel image is sRGB, which the unit
   * decodes, so it now describes a format rather than failing; two-component
   * gamma still has no sampled format and must be refused.
   */
  auto mutated_image = DescriptorDwords(linear, 0);
  mutated_image[0] |= UINT32_C(1) << 3U; // gamma ON
  Check(DecodeRogueTextureImageDescriptor(mutated_image).format ==
            TextureFormat::kRgba8Srgb,
        "gamma image decodes as sRGB");
  mutated_image = DescriptorDwords(linear, 0);
  mutated_image[0] |= UINT32_C(1) << 4U; // two-component gamma
  ExpectFailure([&] { DecodeRogueTextureImageDescriptor(mutated_image); },
                "two-component gamma image mutation");
  mutated_image = DescriptorDwords(linear, 0);
  mutated_image[3] |= UINT32_C(1) << (54U - 32U); // compression state
  ExpectFailure([&] { DecodeRogueTextureImageDescriptor(mutated_image); },
                "compression image mutation");

  const TextureLinearAxis zero = ComputeTextureLinearRepeat(0.0F, 512);
  const TextureLinearAxis one = ComputeTextureLinearRepeat(1.0F, 512);
  const TextureLinearAxis negative =
      ComputeTextureLinearRepeat(-0.25F, 4);
  Check(zero.lower == 511 && zero.upper == 0 && zero.weight == 128,
        "u=0 repeat edge");
  Check(one.lower == 511 && one.upper == 0 && one.weight == 128,
        "u=1 repeat edge");
  Check(negative.lower == 2 && negative.upper == 3 &&
            negative.weight == 128,
        "u=-0.25 negative repeat");
  const TextureLinearAxis trilinear04_default =
      ComputeTextureLinearRepeat(0.921287477016449F, 256);
  const TextureLinearAxis trilinear04_snap = ComputeTextureLinearRepeat(
      0.921287477016449F, 256, TextureWrapMode::kRepeat,
      0.5F - 1.0F / 256.0F);
  Check(trilinear04_default.lower == 235 &&
            trilinear04_default.upper == 236 &&
            trilinear04_default.weight == 89 &&
            trilinear04_snap.lower == 235 &&
            trilinear04_snap.upper == 236 &&
            trilinear04_snap.weight == 90,
        "Gate 19 coordinate half-LSB snap preserves tap identity");
  Check(LerpTextureUnorm8(0, 1, 128) == 0,
        "positive +0.5 delta tie rounds to even zero");
  Check(LerpTextureUnorm8(1, 2, 128) == 1,
        "endpoint parity does not change positive delta-tie rounding");
  Check(LerpTextureUnorm8(1, 4, 128) == 3,
        "positive +1.5 delta tie rounds to even two");
  Check(LerpTextureUnorm8(1, 0, 128) == 1,
        "negative -0.5 delta tie rounds to even zero");
  Check(LerpTextureUnorm8(3, 0, 128) == 1,
        "negative -1.5 delta tie rounds to even minus two");
}

void CheckSequenceColorMipMaterialization() {
  using pvrgpu::stub::DriverPcoSampledTexture;
  using pvrgpu::stub::DriverPcoTextureSource;
  using pvrgpu::stub::MaterializeSequenceColorMipChain;
  using pvrgpu::stub::MemoryMode;
  using pvrgpu::stub::kDriverPcoSequenceColorAddressBase;

  GpuMemorySystem memory(MemoryMode::kDirect);
  std::vector<std::uint8_t> base(4U * 4U * 4U, 0);
  for (std::size_t texel = 0; texel < 16; ++texel) {
    base[texel * 4U + 0U] = static_cast<std::uint8_t>(texel * 4U);
    base[texel * 4U + 1U] = static_cast<std::uint8_t>(texel * 4U + 1U);
    base[texel * 4U + 2U] = static_cast<std::uint8_t>(texel * 4U + 2U);
    base[texel * 4U + 3U] = 255;
  }
  memory.HostWrite(kDriverPcoSequenceColorAddressBase, base.data(),
                   base.size());

  DriverPcoSampledTexture texture;
  texture.source = DriverPcoTextureSource::kPreviousColorAttachment;
  texture.producer_command_index = 0;
  texture.format = "PIPE_FORMAT_R8G8B8A8_UNORM";
  texture.declared_bytes_size = 84;
  texture.mip_count = 3;
  texture.mip[0] = {4, 4, 16, 0};
  texture.mip[1] = {2, 2, 8, 64};
  texture.mip[2] = {1, 1, 4, 80};
  const pvrgpu::stub::MemoryAccessStats stats =
      MaterializeSequenceColorMipChain(memory, texture);
  const pvrgpu::stub::MemoryReadResult chain = memory.Readback(
      kDriverPcoSequenceColorAddressBase, 84,
      pvrgpu::stub::MemoryClient::kFramebufferReadback);
  Check(chain.data.size() == 84 &&
            std::equal(base.begin(), base.end(), chain.data.begin()) &&
            chain.data[64] == 10 && chain.data[65] == 11 &&
            chain.data[66] == 12 && chain.data[67] == 255 &&
            chain.data[80] == 30 && chain.data[81] == 31 &&
            chain.data[82] == 32 && chain.data[83] == 255 &&
            stats.direct_read_bytes != 0 &&
            stats.direct_write_bytes == 84,
        "sequence mip chain is derived from and committed to producer DRAM");

  // Terrain's early passes sample a previous color attachment directly at
  // mip 0. This is a valid one-level texture view, not a request to fabricate
  // a mip chain or rewrite the producer attachment.
  memory.HostWrite(kDriverPcoSequenceColorAddressBase, base.data(),
                   base.size());
  DriverPcoSampledTexture single_mip = texture;
  single_mip.declared_bytes_size = base.size();
  single_mip.mip_count = 1;
  single_mip.mip[1] = {};
  single_mip.mip[2] = {};
  const pvrgpu::stub::MemoryAccessStats single_stats =
      MaterializeSequenceColorMipChain(memory, single_mip);
  const pvrgpu::stub::MemoryReadResult single_readback = memory.Readback(
      kDriverPcoSequenceColorAddressBase, base.size(),
      pvrgpu::stub::MemoryClient::kFramebufferReadback);
  Check(single_readback.data == base && single_stats.direct_read_bytes != 0 &&
            single_stats.direct_write_bytes == 0,
        "single-mip previous-color view aliases producer DRAM without write");

  DriverPcoSampledTexture trailing_mip = single_mip;
  trailing_mip.mip[1] = {1, 1, 4, 64};
  ExpectFailure(
      [&] { MaterializeSequenceColorMipChain(memory, trailing_mip); },
      "single-mip previous-color view rejects unused mip metadata");

  // A normalized blit maps 5x3 -> 2x1 through destination texel centres;
  // it does not average fixed 2x2 blocks or simply discard odd edges.  The
  // selected U8 filter also rounds horizontal and vertical lerps separately.
  std::vector<std::uint8_t> odd_base(5U * 3U * 4U, 0);
  static constexpr std::array<std::array<std::uint8_t, 4>, 5> kMiddleRow = {{
      {50, 7, 1, 255},
      {60, 8, 4, 255},
      {70, 9, 7, 255},
      {80, 10, 10, 255},
      {90, 11, 13, 255},
  }};
  for (std::size_t x = 0; x < kMiddleRow.size(); ++x) {
    std::copy(kMiddleRow[x].begin(), kMiddleRow[x].end(),
              odd_base.begin() + (5U + x) * 4U);
  }
  memory.HostWrite(kDriverPcoSequenceColorAddressBase, odd_base.data(),
                   odd_base.size());
  DriverPcoSampledTexture odd_texture = texture;
  odd_texture.declared_bytes_size = 72;
  odd_texture.mip[0] = {5, 3, 20, 0};
  odd_texture.mip[1] = {2, 1, 8, 60};
  odd_texture.mip[2] = {1, 1, 4, 68};
  MaterializeSequenceColorMipChain(memory, odd_texture);
  const pvrgpu::stub::MemoryReadResult odd_chain = memory.Readback(
      kDriverPcoSequenceColorAddressBase, 72,
      pvrgpu::stub::MemoryClient::kFramebufferReadback);
  Check(odd_chain.data.size() == 72 && odd_chain.data[60] == 58 &&
            odd_chain.data[61] == 8 && odd_chain.data[62] == 3 &&
            odd_chain.data[63] == 255 && odd_chain.data[64] == 82 &&
            odd_chain.data[65] == 10 && odd_chain.data[66] == 11 &&
            odd_chain.data[67] == 255 && odd_chain.data[68] == 70 &&
            odd_chain.data[69] == 9 && odd_chain.data[70] == 7 &&
            odd_chain.data[71] == 255,
        "sequence mip blit preserves odd-extent centre mapping and U8 RNE");

  DriverPcoSampledTexture malformed = texture;
  malformed.mip[1].offset_bytes = 60;
  ExpectFailure(
      [&] { MaterializeSequenceColorMipChain(memory, malformed); },
      "overlapping sequence mip layout");
}

void CheckIntegerImageDescriptors() {
  const auto fixture = MakeGlbenchFillTextureFixture(TextureFilter::kNearest);
  auto words = DescriptorDwords(fixture, 0);
  // A tight two-texel row, one mip, at the fixture's real address.
  for (const std::uint64_t format : {62U, 63U}) {
    const std::uint64_t word0 = UINT64_C(4) | (UINT64_C(3) << 5U) |
        (UINT64_C(2) << 8U) | (UINT64_C(1) << 11U) | (format << 27U) |
        (UINT64_C(1) << 34U);
    const std::uint64_t word1 =
        ((fixture.resource.gpu_address >> 2U) << 16U) |
        (UINT64_C(1) << 60U) | UINT64_C(1);
    words = {static_cast<std::uint32_t>(word0),
             static_cast<std::uint32_t>(word0 >> 32U),
             static_cast<std::uint32_t>(word1),
             static_cast<std::uint32_t>(word1 >> 32U)};
    const auto image = DecodeRogueTextureImageDescriptor(words);
    Check(image.format == (format == 62U ? TextureFormat::kRgba32Uint
                                        : TextureFormat::kRgba32Sint) &&
              image.width == 2 && image.height == 1 &&
              image.row_pitch_bytes == 32 && image.mip_count == 1,
          "Rogue integer format descriptor uses a sixteen-byte texel stride");
    auto sampler = DecodeRogueTextureSamplerDescriptor(DescriptorDwords(fixture, 8));
    Check(DriverPcoTextureDescriptorClassSupported(image, sampler, 1),
          "integer image admits nearest sampling");
    sampler.mag_filter = TextureFilter::kLinear;
    Check(!DriverPcoTextureDescriptorClassSupported(image, sampler, 1),
          "integer image rejects a linear filter");
    auto gamma = words;
    gamma[0] |= UINT32_C(1) << 3U;
    ExpectFailure([&] { (void)DecodeRogueTextureImageDescriptor(gamma); },
                  "integer image rejects gamma conversion");
  }
}

Rgba8 FixtureTexel(const GlbenchFillTextureFixture &fixture,
                   const pvrgpu::stub::TextureMipLevel &mip,
                   std::uint32_t x, std::uint32_t y) {
  const std::uint64_t offset =
      static_cast<std::uint64_t>(mip.offset_bytes) +
      static_cast<std::uint64_t>(y) * mip.row_pitch_bytes +
      static_cast<std::uint64_t>(x) * 4U;
  Check(offset + 4U <= fixture.texture_bytes.size(),
        "fixture texel address is in range");
  Rgba8 texel{};
  for (std::size_t component = 0; component < texel.size(); ++component)
    texel[component] = fixture.texture_bytes.at(offset + component);
  return texel;
}

void AppendBilinearReads(const GlbenchFillTextureFixture &fixture,
                         std::uint8_t mip_level, float u, float v,
                         std::vector<std::uint64_t> &addresses,
                         std::vector<Rgba8> &taps) {
  const pvrgpu::stub::TextureMipLevel &mip =
      fixture.resource.mip[mip_level];
  const TextureLinearAxis x = ComputeTextureLinearRepeat(u, mip.width);
  const TextureLinearAxis y = ComputeTextureLinearRepeat(v, mip.height);
  const std::array<std::array<std::uint32_t, 2>, 4> locations = {{
      {{x.lower, y.lower}},
      {{x.upper, y.lower}},
      {{x.lower, y.upper}},
      {{x.upper, y.upper}},
  }};
  for (const auto &location : locations) {
    const std::uint64_t offset =
        static_cast<std::uint64_t>(mip.offset_bytes) +
        static_cast<std::uint64_t>(location[1]) * mip.row_pitch_bytes +
        static_cast<std::uint64_t>(location[0]) * 4U;
    addresses.push_back(fixture.resource.gpu_address + offset);
    taps.push_back(FixtureTexel(fixture, mip, location[0], location[1]));
  }
}

class TextureMemoryResponder final : public sc_core::sc_module {
 public:
  sc_core::sc_fifo_in<MemoryTxn> upload_input{"upload_input"};
  sc_core::sc_fifo_out<MemoryTxn> upload_output{"upload_output"};
  sc_core::sc_fifo_in<MemoryTxn> cache_input{"cache_input"};
  sc_core::sc_fifo_out<MemoryTxn> cache_output{"cache_output"};

  TextureMemoryResponder(
      sc_core::sc_module_name name, MemoryPool &pool,
      std::uint64_t upload_address, std::uint64_t upload_bytes,
      std::vector<std::uint64_t> expected_addresses,
      std::vector<Rgba8> taps, std::size_t taps_per_sample = 0)
      : sc_core::sc_module(name), pool_(pool),
        upload_address_(upload_address), upload_bytes_(upload_bytes),
        expected_addresses_(expected_addresses), taps_(taps),
        taps_per_sample_(taps_per_sample != 0 ? taps_per_sample
                                              : expected_addresses.size()) {
    Check(expected_addresses_.size() == taps_.size() && !taps_.empty(),
          "cache responder address/tap vectors");
    SC_THREAD(UploadRun);
    SC_THREAD(CacheRun);
  }

 private:
  void UploadRun() {
    const MemoryTxn request = upload_input.read();
    Check(request.address == upload_address_ && request.bytes == upload_bytes_ &&
              request.operation == MemoryOperation::kWrite &&
              request.client == MemoryClient::kTextureUpload &&
              request.payload_format == MemoryPayloadFormat::kLinearBytes &&
              HasPoolHandle(request.payload),
          "raw descriptor address drives texture upload");
    MemoryTxn response = request;
    response.payload = {};
    upload_output.write(response);
  }

  void CacheRun() {
    for (std::size_t tap = 0; tap < expected_addresses_.size(); ++tap) {
      /*
       * A sample's taps are numbered from its own request ID times the tap
       * stride, so the IDs a batch of samples issues are distinct but not
       * dense -- the four-lane quad below reads taps 0..7 of lane 1 as IDs
       * 16..23.  Checking for a dense 0,1,2,... only held while the stride
       * happened to equal the taps a sample issued.
       */
      const std::uint64_t expected_request_id =
          (tap / taps_per_sample_) *
              pvrgpu::stub::kTextureSampleTapRequestStride +
          (tap % taps_per_sample_);
      const MemoryTxn request = cache_input.read();
      Check(request.address == expected_addresses_[tap] && request.bytes == 4 &&
                request.request_id == expected_request_id &&
                request.operation == MemoryOperation::kRead &&
                request.client == MemoryClient::kTextureCache &&
                request.payload_format == MemoryPayloadFormat::kLinearBytes &&
                !HasPoolHandle(request.payload),
            "FIFO tap address and request identity");
      MemoryTxn response = request;
      response.payload = StoreNewArray(
          pool_, std::vector<std::uint8_t>(taps_[tap].begin(),
                                           taps_[tap].end()));
      cache_output.write(response);
    }
  }

  MemoryPool &pool_;
  std::uint64_t upload_address_ = 0;
  std::uint64_t upload_bytes_ = 0;
  std::vector<std::uint64_t> expected_addresses_;
  std::vector<Rgba8> taps_;
  std::size_t taps_per_sample_;
};

void CheckCubeNonfiniteLod() {
  using pvrgpu::stub::ComputeTextureCubeImplicitLod;
  auto fixture = MakeGlbenchFillTextureFixture(TextureFilter::kLinear);
  std::array<std::uint32_t, 4> image_words{}, sampler_words{};
  std::copy_n(fixture.fragment_shared.begin(), 4, image_words.begin());
  std::copy_n(fixture.fragment_shared.begin() + 8, 4, sampler_words.begin());
  auto image = DecodeRogueTextureImageDescriptor(image_words);
  auto sampler = DecodeRogueTextureSamplerDescriptor(sampler_words);
  image.width = image.height = 64;
  image.mip_count = 7;
  sampler.min_lod_u4_6 = 0;
  sampler.max_lod_u4_6 = 6 * 64;
  sampler.mip_filter = TextureFilter::kLinear;
  const std::array<std::array<float, 3>, 4> finite = {{{1, 0, 0},
      {1, 0, -0.125F}, {1, -0.125F, 0}, {1, -0.125F, -0.125F}}};
  const auto control = ComputeTextureCubeImplicitLod(finite, image, sampler);
  Check(control.rho_squared == 16 && control.lambda == 2 &&
        control.dsdx == 4 && control.dtdy == 4,
        "finite cube projection retains the exact derivative and LOD");
  const std::array<float, 4> undefined = {{
      std::numeric_limits<float>::quiet_NaN(),
      -std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity()}};
  for (float value : undefined) {
    for (unsigned lane = 0; lane < 3; ++lane) {
      auto directions = finite;
      directions[lane] = {value, value, value};
      for (unsigned minimum : {0U, 96U}) {
        sampler.min_lod_u4_6 = minimum;
        const auto lod = ComputeTextureCubeImplicitLod(directions, image, sampler);
        Check(!std::isfinite(lod.rho_squared) &&
              lod.lambda == minimum / 64.0F &&
              lod.level0 == minimum / 64,
              "nonfinite cube footprint retains raw evidence and selects minimum LOD");
      }
    }
  }
  sampler.min_lod_u4_6 = 0;
  const std::array<std::array<float, 3>, 4> zero{};
  const auto zero_lod = ComputeTextureCubeImplicitLod(zero, image, sampler);
  Check(std::isnan(zero_lod.rho_squared) && zero_lod.lambda == 0,
        "zero cube directions use the undefined-footprint minimum LOD policy");
  auto unused_corner = finite;
  unused_corner[3].fill(undefined[0]);
  const auto unused_lod = ComputeTextureCubeImplicitLod(unused_corner, image, sampler);
  Check(unused_lod.rho_squared == control.rho_squared &&
        unused_lod.lambda == control.lambda,
        "unused coarse-derivative corner cannot erase a valid footprint");
  auto singular_projection = finite;
  singular_projection[1] = {0, 1, 0};
  ExpectFailure([&] {
    ComputeTextureCubeImplicitLod(singular_projection, image, sampler);
  }, "valid nonzero directions cannot hide an unmodeled common-face singularity");
  auto overflowing_projection = finite;
  overflowing_projection[1] = {std::numeric_limits<float>::min(), 1, 1};
  ExpectFailure([&] {
    ComputeTextureCubeImplicitLod(overflowing_projection, image, sampler);
  }, "finite nonzero direction cannot hide projection derivative overflow");
  image.width = 0;
  ExpectFailure([&] { ComputeTextureCubeImplicitLod(zero, image, sampler); },
                "nonfinite direction cannot bypass invalid image metadata");
}

void CheckEventPaths(pvrgpu::stub::MemoryMode memory_mode) {
  MemoryPool pool;
  GlbenchFillTextureFixture fixture =
      MakeGlbenchFillTextureFixture(TextureFilter::kLinear);

  // These exact binary fractions yield lower=(127,383), upper=(128,384),
  // and weights=(64,192) in the selected U8 coordinate datapath.
  constexpr float kU = 0.24951171875F;
  constexpr float kV = 0.75048828125F;
  const TextureLinearAxis x = ComputeTextureLinearRepeat(kU, 512);
  const TextureLinearAxis y = ComputeTextureLinearRepeat(kV, 512);
  Check(x.lower == 127 && x.upper == 128 && x.weight == 64 &&
            y.lower == 383 && y.upper == 384 && y.weight == 192,
        "asymmetric event-path weights");

  TextureSampleRequest request;
  request.shader_lane_index = 0;
  request.coordinates[0] = FloatBits(kU);
  request.coordinates[1] = FloatBits(kV);
  std::copy_n(fixture.fragment_shared.begin(), 4, request.texture_state);
  std::copy_n(fixture.fragment_shared.begin() + 8, 4,
              request.sampler_state);
  request.request_id = 0;
  request.coordinate_count = 2;
  request.component_count = 4;
  request.descriptor_set = 0;
  request.binding = 0;
  request.dimension = 2;
  request.normalized = 1;
  request.shader_stage = pvrgpu::stub::ShaderStage::kFragment;

  fixture.resource.data = StoreNewArray(pool, fixture.texture_bytes);
  PipelineState state;
  state.sequence = 1;
  state.functional_case = FunctionalCase::kFillTexBilinear;
  state.stage = PipelineStage::kFragmentTexturePending;
  state.fragment_shader_lane_count = 1;
  state.texture_sample_requests =
      StoreNewArray(pool, std::vector<TextureSampleRequest>{request});
  state.texture_resources =
      StoreNewArray(pool, std::vector<TextureResource>{fixture.resource});
  state.sampler_states = StoreNewArray(
      pool, std::vector<pvrgpu::stub::SamplerState>{fixture.sampler});
  state.fragment_shared_registers = StoreNewArray(
      pool, std::vector<std::uint32_t>(fixture.fragment_shared.begin(),
                                      fixture.fragment_shared.end()));
  const pvrgpu::stub::PoolHandle state_handle =
      pool.Allocate(sizeof(PipelineState));
  StorePipelineState(pool, state_handle, state);
  PipelineTxn transaction;
  transaction.state = state_handle;
  transaction.frame = 1;
  transaction.sequence = 1;

  const std::vector<Rgba8> taps = {
      {{0, 10, 20, 30}},
      {{40, 50, 60, 70}},
      {{80, 90, 100, 110}},
      {{120, 130, 140, 150}},
  };
  const std::uint64_t base = pvrgpu::stub::kGlbenchTextureGpuAddress;
  constexpr std::uint64_t pitch = 2048;
  const std::vector<std::uint64_t> expected_addresses = {
      base + 383U * pitch + 127U * 4U,
      base + 383U * pitch + 128U * 4U,
      base + 384U * pitch + 127U * 4U,
      base + 384U * pitch + 128U * 4U,
  };

  sc_core::sc_fifo<PipelineTxn> module_input("module_input", 1);
  sc_core::sc_fifo<PipelineTxn> module_output("module_output", 1);
  sc_core::sc_fifo<PipelineTxn> sample_input("sample_input", 1);
  sc_core::sc_fifo<PipelineTxn> sample_output("sample_output", 1);
  sc_core::sc_fifo<MemoryTxn> cache_request("cache_request", 1);
  sc_core::sc_fifo<MemoryTxn> cache_response("cache_response", 1);
  sc_core::sc_fifo<MemoryTxn> upload_request("upload_request", 1);
  sc_core::sc_fifo<MemoryTxn> upload_response("upload_response", 1);

  TextureUnit texture("texture", pool);
  texture.input(module_input);
  texture.output(module_output);
  texture.sample_input(sample_input);
  texture.sample_output(sample_output);
  texture.cache_request(cache_request);
  texture.cache_response(cache_response);
  texture.upload_request(upload_request);
  texture.upload_response(upload_response);

  TextureMemoryResponder responder(
      "responder", pool, base, fixture.resource.byte_size,
      expected_addresses, taps);
  responder.upload_input(upload_request);
  responder.upload_output(upload_response);
  responder.cache_input(cache_request);
  responder.cache_output(cache_response);

  // Gate 18 runs alongside the four-tap test so every SystemC object is
  // elaborated before the single sc_start().  Four adjacent lanes form a real
  // implicit-derivative quad; each lane emits mip3 00/10/01/11 followed by
  // mip4 00/10/01/11 through the cache FIFO.
  MemoryPool gate_pool;
  GlbenchFillTextureFixture gate_fixture = MakeGlbenchFillTextureFixture(
      FunctionalCase::kFillTexTrilinearLinear01);
  const float gate_scale = BitsFloat(gate_fixture.vertex_scale_bits);
  const float gate_derivative = 1.0F / (64.0F * gate_scale);
  constexpr float kGateU = 0.25F;
  constexpr float kGateV = 0.375F;
  const std::array<std::array<float, 2>, 4> gate_coordinates = {{
      {{kGateU, kGateV}},
      {{kGateU + gate_derivative, kGateV}},
      {{kGateU, kGateV + gate_derivative}},
      {{kGateU + gate_derivative, kGateV + gate_derivative}},
  }};

  std::vector<TextureSampleRequest> gate_requests(4);
  std::vector<std::uint64_t> gate_addresses;
  std::vector<Rgba8> gate_taps;
  gate_addresses.reserve(32);
  gate_taps.reserve(32);
  for (std::size_t lane = 0; lane < gate_requests.size(); ++lane) {
    TextureSampleRequest &gate_request = gate_requests[lane];
    gate_request.shader_lane_index = static_cast<std::uint32_t>(lane);
    gate_request.quad_id = 7;
    gate_request.coordinates[0] = FloatBits(gate_coordinates[lane][0]);
    gate_request.coordinates[1] = FloatBits(gate_coordinates[lane][1]);
    std::copy_n(gate_fixture.fragment_shared.begin(), 4,
                gate_request.texture_state);
    std::copy_n(gate_fixture.fragment_shared.begin() + 8, 4,
                gate_request.sampler_state);
    gate_request.request_id = lane;
    gate_request.coordinate_count = 2;
    gate_request.component_count = 4;
    gate_request.dimension = 2;
    gate_request.normalized = 1;
    gate_request.quad_lane = static_cast<std::uint8_t>(lane);
    gate_request.shader_stage = pvrgpu::stub::ShaderStage::kFragment;
    AppendBilinearReads(gate_fixture, 3, gate_coordinates[lane][0],
                        gate_coordinates[lane][1], gate_addresses, gate_taps);
    AppendBilinearReads(gate_fixture, 4, gate_coordinates[lane][0],
                        gate_coordinates[lane][1], gate_addresses, gate_taps);
  }
  Check(gate_addresses.size() == 32 && gate_taps.size() == 32,
        "Gate 18 quad expands to 32 physical reads");

  gate_fixture.resource.data =
      StoreNewArray(gate_pool, gate_fixture.texture_bytes);
  PipelineState gate_state;
  gate_state.sequence = 2;
  gate_state.functional_case = FunctionalCase::kFillTexTrilinearLinear01;
  gate_state.stage = PipelineStage::kFragmentTexturePending;
  gate_state.fragment_shader_lane_count = gate_requests.size();
  gate_state.texture_sample_requests =
      StoreNewArray(gate_pool, gate_requests);
  gate_state.texture_resources = StoreNewArray(
      gate_pool, std::vector<TextureResource>{gate_fixture.resource});
  gate_state.sampler_states = StoreNewArray(
      gate_pool,
      std::vector<pvrgpu::stub::SamplerState>{gate_fixture.sampler});
  gate_state.fragment_shared_registers = StoreNewArray(
      gate_pool,
      std::vector<std::uint32_t>(gate_fixture.fragment_shared.begin(),
                                 gate_fixture.fragment_shared.end()));
  const pvrgpu::stub::PoolHandle gate_state_handle =
      gate_pool.Allocate(sizeof(PipelineState));
  StorePipelineState(gate_pool, gate_state_handle, gate_state);
  PipelineTxn gate_transaction;
  gate_transaction.state = gate_state_handle;
  gate_transaction.frame = 2;
  gate_transaction.sequence = 2;

  sc_core::sc_fifo<PipelineTxn> gate_module_input("gate_module_input", 1);
  sc_core::sc_fifo<PipelineTxn> gate_module_output("gate_module_output", 1);
  sc_core::sc_fifo<PipelineTxn> gate_sample_input("gate_sample_input", 1);
  sc_core::sc_fifo<PipelineTxn> gate_sample_output("gate_sample_output", 1);
  sc_core::sc_fifo<MemoryTxn> gate_cache_request("gate_cache_request", 1);
  sc_core::sc_fifo<MemoryTxn> gate_cache_response("gate_cache_response", 1);
  sc_core::sc_fifo<MemoryTxn> gate_upload_request("gate_upload_request", 1);
  sc_core::sc_fifo<MemoryTxn> gate_upload_response("gate_upload_response", 1);

  TextureUnit gate_texture("gate_texture", gate_pool);
  gate_texture.input(gate_module_input);
  gate_texture.output(gate_module_output);
  gate_texture.sample_input(gate_sample_input);
  gate_texture.sample_output(gate_sample_output);
  gate_texture.cache_request(gate_cache_request);
  gate_texture.cache_response(gate_cache_response);
  gate_texture.upload_request(gate_upload_request);
  gate_texture.upload_response(gate_upload_response);

  /* Four lanes, and a trilinear sample reads two levels of four taps. */
  TextureMemoryResponder gate_responder(
      "gate_responder", gate_pool, base, gate_fixture.resource.byte_size,
      gate_addresses, gate_taps, gate_addresses.size() / gate_requests.size());
  gate_responder.upload_input(gate_upload_request);
  gate_responder.upload_output(gate_upload_response);
  gate_responder.cache_input(gate_cache_request);
  gate_responder.cache_output(gate_cache_response);

  // API-v6 driver PCO sampling uses a one-mip RGBX resource at the same fixed
  // GPU address. The raw image descriptor owns the alpha swizzle: even when
  // the stored X byte is non-one, the TPU response must contain 1.0F.
  MemoryPool driver_pool;
  GlbenchFillTextureFixture driver_fixture =
      MakeGlbenchFillTextureFixture(TextureFilter::kNearest);
  driver_fixture.texture_bytes.resize(
      static_cast<std::size_t>(pvrgpu::stub::kDriverPcoTextureBytes));
  const Rgba8 driver_texel = {{17, 34, 51, 68}};
  std::copy(driver_texel.begin(), driver_texel.end(),
            driver_fixture.texture_bytes.begin());
  driver_fixture.resource.byte_size = static_cast<std::uint32_t>(
      driver_fixture.texture_bytes.size());
  driver_fixture.resource.mip_count = 1;
  driver_fixture.resource.format = TextureFormat::kRgbx8Unorm;
  driver_fixture.fragment_shared[0] &= ~(UINT32_C(7) << 5U);
  driver_fixture.fragment_shared[0] |= UINT32_C(4) << 5U;
  driver_fixture.fragment_shared[2] &= ~UINT32_C(0xffff);
  driver_fixture.fragment_shared[2] |= UINT32_C(511);
  driver_fixture.fragment_shared[3] &= UINT32_C(0x0fffffff);
  driver_fixture.fragment_shared[3] |= UINT32_C(1) << 28U;
  driver_fixture.fragment_shared[4] =
      static_cast<std::uint32_t>(driver_fixture.texture_bytes.size());
  driver_fixture.fragment_shared[9] |=
      (UINT32_C(2) << (33U - 32U)) |
      (UINT32_C(2) << (41U - 32U));
  std::copy_n(driver_fixture.fragment_shared.begin() + 8, 4,
              driver_fixture.fragment_shared.begin() + 16);
  driver_fixture.fragment_shared[17] |= UINT32_C(1) << (36U - 32U);
  driver_fixture.fragment_shared[17] |= UINT32_C(1) << (38U - 32U);
  driver_fixture.sampler.wrap_u = TextureWrapMode::kClampToEdge;
  driver_fixture.sampler.wrap_v = TextureWrapMode::kClampToEdge;

  TextureSampleRequest driver_request;
  driver_request.shader_lane_index = 0;
  driver_request.coordinates[0] = FloatBits(0.0F);
  driver_request.coordinates[1] = FloatBits(0.0F);
  std::copy_n(driver_fixture.fragment_shared.begin(), 4,
              driver_request.texture_state);
  std::copy_n(driver_fixture.fragment_shared.begin() + 8, 4,
              driver_request.sampler_state);
  driver_request.coordinate_count = 2;
  driver_request.component_count = 4;
  driver_request.dimension = 2;
  driver_request.normalized = 1;
  driver_request.shader_stage = pvrgpu::stub::ShaderStage::kFragment;

  driver_fixture.resource.data =
      StoreNewArray(driver_pool, driver_fixture.texture_bytes);
  PipelineState driver_state;
  driver_state.sequence = 3;
  driver_state.functional_case = FunctionalCase::kDriverPcoTriangles;
  driver_state.stage = PipelineStage::kFragmentTexturePending;
  driver_state.sampled_texture_count = 1;
  driver_state.fragment_pco_abi.shareds =
      pvrgpu::stub::kFillTexNearestSharedDwordCount;
  driver_state.fragment_shader_lane_count = 1;
  Check(UsesTextureSampling(driver_state),
        "driver PCO sampled-image routing is state-derived");
  driver_state.texture_sample_requests = StoreNewArray(
      driver_pool, std::vector<TextureSampleRequest>{driver_request});
  driver_state.texture_resources = StoreNewArray(
      driver_pool, std::vector<TextureResource>{driver_fixture.resource});
  driver_state.sampler_states = StoreNewArray(
      driver_pool,
      std::vector<pvrgpu::stub::SamplerState>{driver_fixture.sampler});
  driver_state.fragment_shared_registers = StoreNewArray(
      driver_pool,
      std::vector<std::uint32_t>(driver_fixture.fragment_shared.begin(),
                                 driver_fixture.fragment_shared.end()));
  const pvrgpu::stub::PoolHandle driver_state_handle =
      driver_pool.Allocate(sizeof(PipelineState));
  StorePipelineState(driver_pool, driver_state_handle, driver_state);
  const PipelineTxn driver_transaction{driver_state_handle, 3, 3};

  sc_core::sc_fifo<PipelineTxn> driver_module_input("driver_module_input", 1);
  sc_core::sc_fifo<PipelineTxn> driver_module_output("driver_module_output", 1);
  sc_core::sc_fifo<PipelineTxn> driver_sample_input("driver_sample_input", 1);
  sc_core::sc_fifo<PipelineTxn> driver_sample_output("driver_sample_output", 1);
  sc_core::sc_fifo<MemoryTxn> driver_cache_request("driver_cache_request", 1);
  sc_core::sc_fifo<MemoryTxn> driver_cache_response("driver_cache_response", 1);
  sc_core::sc_fifo<MemoryTxn> driver_upload_request("driver_upload_request", 1);
  sc_core::sc_fifo<MemoryTxn> driver_upload_response("driver_upload_response", 1);

  TextureUnit driver_texture("driver_texture", driver_pool);
  driver_texture.input(driver_module_input);
  driver_texture.output(driver_module_output);
  driver_texture.sample_input(driver_sample_input);
  driver_texture.sample_output(driver_sample_output);
  driver_texture.cache_request(driver_cache_request);
  driver_texture.cache_response(driver_cache_response);
  driver_texture.upload_request(driver_upload_request);
  driver_texture.upload_response(driver_upload_response);

  TextureMemoryResponder driver_responder(
      "driver_responder", driver_pool, base,
      driver_fixture.resource.byte_size, std::vector<std::uint64_t>{base},
      std::vector<Rgba8>{driver_texel});
  driver_responder.upload_input(driver_upload_request);
  driver_responder.upload_output(driver_upload_response);
  driver_responder.cache_input(driver_cache_request);
  driver_responder.cache_output(driver_cache_response);

  // The same exact descriptor is also consumed through the independent VS
  // stage bank. Vertex requests carry no quad identity and are clamped to
  // descriptor LOD0; their cycles are charged to the tiler, not renderer.
  MemoryPool vertex_pool;
  GlbenchFillTextureFixture vertex_fixture = driver_fixture;
  vertex_fixture.resource.data =
      StoreNewArray(vertex_pool, vertex_fixture.texture_bytes);
  TextureSampleRequest vertex_request = driver_request;
  vertex_request.shader_stage = pvrgpu::stub::ShaderStage::kVertex;
  PipelineState vertex_state;
  vertex_state.sequence = 4;
  vertex_state.functional_case = FunctionalCase::kDriverPcoTriangles;
  vertex_state.stage = PipelineStage::kVertexTexturePending;
  vertex_state.vertex_sampled_texture_count = 1;
  vertex_state.vertex_pco_abi.shareds =
      pvrgpu::stub::kFillTexNearestSharedDwordCount;
  vertex_state.counters.vs_invocations = 1;
  vertex_state.texture_sample_requests = StoreNewArray(
      vertex_pool, std::vector<TextureSampleRequest>{vertex_request});
  vertex_state.vertex_texture_resources = StoreNewArray(
      vertex_pool, std::vector<TextureResource>{vertex_fixture.resource});
  vertex_state.vertex_sampler_states = StoreNewArray(
      vertex_pool,
      std::vector<pvrgpu::stub::SamplerState>{vertex_fixture.sampler});
  std::vector<pvrgpu::stub::ShaderSharedRegister> vertex_shared(
      pvrgpu::stub::kFillTexNearestSharedDwordCount);
  for (std::size_t index = 0; index < vertex_shared.size(); ++index)
    vertex_shared[index].value = vertex_fixture.fragment_shared[index];
  vertex_state.vertex_shared_registers =
      StoreNewArray(vertex_pool, vertex_shared);
  const pvrgpu::stub::PoolHandle vertex_state_handle =
      vertex_pool.Allocate(sizeof(PipelineState));
  StorePipelineState(vertex_pool, vertex_state_handle, vertex_state);
  const PipelineTxn vertex_transaction{vertex_state_handle, 4, 4};

  sc_core::sc_fifo<PipelineTxn> vertex_module_input("vertex_module_input", 1);
  sc_core::sc_fifo<PipelineTxn> vertex_module_output("vertex_module_output", 1);
  sc_core::sc_fifo<PipelineTxn> vertex_sample_input("vertex_sample_input", 1);
  sc_core::sc_fifo<PipelineTxn> vertex_sample_output("vertex_sample_output", 1);
  sc_core::sc_fifo<MemoryTxn> vertex_cache_request("vertex_cache_request", 1);
  sc_core::sc_fifo<MemoryTxn> vertex_cache_response("vertex_cache_response", 1);
  sc_core::sc_fifo<MemoryTxn> vertex_upload_request("vertex_upload_request", 1);
  sc_core::sc_fifo<MemoryTxn> vertex_upload_response("vertex_upload_response", 1);

  TextureUnit vertex_texture("vertex_texture", vertex_pool);
  vertex_texture.input(vertex_module_input);
  vertex_texture.output(vertex_module_output);
  vertex_texture.vertex_sample_input(vertex_sample_input);
  vertex_texture.vertex_sample_output(vertex_sample_output);
  vertex_texture.cache_request(vertex_cache_request);
  vertex_texture.cache_response(vertex_cache_response);
  vertex_texture.upload_request(vertex_upload_request);
  vertex_texture.upload_response(vertex_upload_response);

  TextureMemoryResponder vertex_responder(
      "vertex_responder", vertex_pool, base,
      vertex_fixture.resource.byte_size, std::vector<std::uint64_t>{base},
      std::vector<Rgba8>{driver_texel});
  vertex_responder.upload_input(vertex_upload_request);
  vertex_responder.upload_output(vertex_upload_response);
  vertex_responder.cache_input(vertex_cache_request);
  vertex_responder.cache_output(vertex_cache_response);

  // Two neighboring 128-bit texels travel through the modeled memory path.
  // Their values cannot be represented exactly as floats, so a floating
  // conversion or an eight-byte fetch is detected by the returned DWORDs.
  MemoryPool integer_pool;
  GpuMemorySystem integer_memory(memory_mode);
  const std::array<std::array<std::uint32_t, 4>, 2> integer_texels = {{
      {{UINT32_C(0xffffffff), UINT32_C(0x80000000), UINT32_C(0x01000001), 1}},
      {{UINT32_C(0x76543210), UINT32_C(0xfedcba98), UINT32_C(0x7fffffff), 0}},
  }};
  std::vector<std::uint8_t> integer_bytes(32);
  for (std::size_t texel = 0; texel < 2; ++texel) {
    for (std::size_t channel = 0; channel < 4; ++channel) {
      for (std::size_t byte = 0; byte < 4; ++byte) {
        integer_bytes[texel * 16 + channel * 4 + byte] =
            static_cast<std::uint8_t>(integer_texels[texel][channel] >> (byte * 8));
      }
    }
  }
  integer_memory.HostWrite(base, integer_bytes.data(), integer_bytes.size());
  TextureResource integer_resource;
  integer_resource.gpu_address = base;
  integer_resource.byte_size = 32;
  integer_resource.format = TextureFormat::kRgba32Uint;
  integer_resource.mip_count = 1;
  integer_resource.mip[0] = {2, 1, 32, 0};
  std::vector<std::uint32_t> integer_shared(
      driver_fixture.fragment_shared.begin(), driver_fixture.fragment_shared.end());
  const std::uint64_t integer_word0 = UINT64_C(4) | (UINT64_C(3) << 5U) |
      (UINT64_C(2) << 8U) | (UINT64_C(1) << 11U) | (UINT64_C(62) << 27U) |
      (UINT64_C(1) << 34U);
  const std::uint64_t integer_word1 =
      ((base >> 2U) << 16U) | (UINT64_C(1) << 60U) | UINT64_C(1);
  integer_shared[0] = static_cast<std::uint32_t>(integer_word0);
  integer_shared[1] = static_cast<std::uint32_t>(integer_word0 >> 32U);
  integer_shared[2] = static_cast<std::uint32_t>(integer_word1);
  integer_shared[3] = static_cast<std::uint32_t>(integer_word1 >> 32U);
  integer_shared[4] = 32;
  std::vector<TextureSampleRequest> integer_requests(2, driver_request);
  for (std::size_t lane = 0; lane < integer_requests.size(); ++lane) {
    auto &request = integer_requests[lane];
    request.shader_lane_index = static_cast<std::uint32_t>(lane);
    request.request_id = lane;
    request.coordinates[0] = FloatBits(lane == 0 ? 0.25F : 0.75F);
    request.fcnorm = 0;
    request.quad_lane = static_cast<std::uint8_t>(lane);
    std::copy_n(integer_shared.begin(), 4, request.texture_state);
  }
  PipelineState integer_state;
  integer_state.memory_mode = memory_mode;
  integer_state.sequence = 5;
  integer_state.functional_case = FunctionalCase::kDriverPcoTriangles;
  integer_state.stage = PipelineStage::kFragmentTexturePending;
  integer_state.sampled_texture_count = 1;
  integer_state.fragment_pco_abi.shareds = integer_shared.size();
  integer_state.fragment_shader_lane_count = integer_requests.size();
  integer_state.texture_sample_requests = StoreNewArray(integer_pool, integer_requests);
  integer_state.texture_resources = StoreNewArray(
      integer_pool, std::vector<TextureResource>{integer_resource});
  integer_state.sampler_states = StoreNewArray(
      integer_pool, std::vector<pvrgpu::stub::SamplerState>{driver_fixture.sampler});
  integer_state.fragment_shared_registers = StoreNewArray(integer_pool, integer_shared);
  const auto integer_state_handle = integer_pool.Allocate(sizeof(PipelineState));
  StorePipelineState(integer_pool, integer_state_handle, integer_state);
  const PipelineTxn integer_transaction{integer_state_handle, 5, 5};
  sc_core::sc_fifo<PipelineTxn> integer_module_input("integer_module_input", 1);
  sc_core::sc_fifo<PipelineTxn> integer_module_output("integer_module_output", 1);
  sc_core::sc_fifo<PipelineTxn> integer_sample_input("integer_sample_input", 1);
  sc_core::sc_fifo<PipelineTxn> integer_sample_output("integer_sample_output", 1);
  TextureUnit integer_texture("integer_texture", integer_pool, &integer_memory);
  integer_texture.input(integer_module_input);
  integer_texture.output(integer_module_output);
  integer_texture.sample_input(integer_sample_input);
  integer_texture.sample_output(integer_sample_output);

  // GS owns a third stage bank and reserves SH0..3 for primitive input.
  // A scalar request cannot accidentally enter the FS derivative/quad path.
  MemoryPool geometry_pool;
  GpuMemorySystem geometry_memory(memory_mode);
  geometry_memory.HostWrite(base, driver_fixture.texture_bytes.data(), driver_fixture.texture_bytes.size());
  geometry_memory.HostWrite(base, driver_texel.data(), driver_texel.size());
  auto geometry_resource = driver_fixture.resource;
  geometry_resource.data = {};
  TextureSampleRequest geometry_request = driver_request;
  geometry_request.shader_stage = pvrgpu::stub::ShaderStage::kGeometry;
  PipelineState geometry_state;
  geometry_state.memory_mode = memory_mode;
  geometry_state.functional_case = FunctionalCase::kDriverPcoTriangles;
  geometry_state.stage = PipelineStage::kGeometryTexturePending;
  geometry_state.geometry_sampled_texture_count = 1;
  geometry_state.geometry_pco_abi.shareds = 24;
  geometry_state.geometry_pco_abi.uniform_buffer_descriptor_start = 24;
  geometry_state.geometry_pco_abi.push_constant_start = 24;
  geometry_state.texture_sample_requests = StoreNewArray(
      geometry_pool, std::vector<TextureSampleRequest>{geometry_request});
  geometry_state.geometry_texture_resources = StoreNewArray(
      geometry_pool, std::vector<TextureResource>{geometry_resource});
  geometry_state.geometry_sampler_states = StoreNewArray(
      geometry_pool, std::vector<pvrgpu::stub::SamplerState>{driver_fixture.sampler});
  std::vector<std::uint32_t> geometry_shared(24, 0xa5a5a5a5U);
  std::copy(driver_fixture.fragment_shared.begin(), driver_fixture.fragment_shared.end(), geometry_shared.begin() + 4);
  geometry_state.geometry_shared_registers = StoreNewArray(geometry_pool, geometry_shared);
  const auto geometry_state_handle = geometry_pool.Allocate(sizeof(PipelineState));
  StorePipelineState(geometry_pool, geometry_state_handle, geometry_state);
  const PipelineTxn geometry_transaction{geometry_state_handle, 7, 7};
  sc_core::sc_fifo<PipelineTxn> geometry_module_input("geometry_module_input", 1);
  sc_core::sc_fifo<PipelineTxn> geometry_module_output("geometry_module_output", 1);
  sc_core::sc_fifo<PipelineTxn> geometry_sample_input("geometry_sample_input", 1);
  sc_core::sc_fifo<PipelineTxn> geometry_sample_output("geometry_sample_output", 1);
  TextureUnit geometry_texture("geometry_texture", geometry_pool, &geometry_memory);
  geometry_texture.input(geometry_module_input);
  geometry_texture.output(geometry_module_output);
  geometry_texture.geometry_sample_input(geometry_sample_input);
  geometry_texture.geometry_sample_output(geometry_sample_output);

  sample_input.write(transaction);
  gate_sample_input.write(gate_transaction);
  driver_sample_input.write(driver_transaction);
  vertex_sample_input.write(vertex_transaction);
  integer_sample_input.write(integer_transaction);
  geometry_sample_input.write(geometry_transaction);
  sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_NS));
  PipelineTxn completed;
  Check(sample_output.nb_read(completed) &&
            completed.state.slot == state_handle.slot &&
            completed.state.generation == state_handle.generation &&
            completed.frame == 1 && completed.sequence == 1,
        "event-driven sample completion identity");

  const PipelineState final_state = LoadPipelineState(pool, state_handle);
  Check(final_state.stage == PipelineStage::kTextureSamplesReady &&
            final_state.counters.texture_requests == 1 &&
            final_state.counters.texel_fetches == 4 &&
            final_state.fragment_texture_request_count == 1 &&
            final_state.fragment_texel_fetch_count == 4 &&
            final_state.counters.texture_cycles ==
                pvrgpu::stub::kReferenceUarch.texture_bypass_cycles,
        "one logical request preserves four physical tap counters");
  const std::vector<TextureSampleResponse> responses =
      LoadArray<TextureSampleResponse>(pool,
                                       final_state.texture_sample_responses);
  Check(responses.size() == 1 && responses[0].request_id == 0 &&
            responses[0].shader_lane_index == 0 &&
            responses[0].shader_stage ==
                pvrgpu::stub::ShaderStage::kFragment,
        "aggregated texture response identity");
  constexpr std::array<std::uint8_t, 4> kExpected = {{70, 80, 90, 100}};
  for (std::size_t component = 0; component < kExpected.size(); ++component) {
    Check(responses[0].rgba[component] ==
              FloatBits(static_cast<float>(kExpected[component]) / 255.0F),
          "asymmetric sequential U8 bilinear component");
  }

  PipelineTxn gate_completed;
  Check(gate_sample_output.nb_read(gate_completed) &&
            gate_completed.state.slot == gate_state_handle.slot &&
            gate_completed.state.generation == gate_state_handle.generation &&
            gate_completed.frame == 2 && gate_completed.sequence == 2,
        "Gate 18 event-driven sample completion identity");
  const PipelineState gate_final_state =
      LoadPipelineState(gate_pool, gate_state_handle);
  Check(gate_final_state.stage == PipelineStage::kTextureSamplesReady &&
            gate_final_state.counters.texture_requests == 4 &&
            gate_final_state.counters.texel_fetches == 32 &&
            gate_final_state.counters.texture_cycles ==
                4U * pvrgpu::stub::kReferenceUarch.texture_bypass_cycles,
        "Gate 18 quad preserves four requests and eight taps per lane");
  const std::vector<TextureSampleResponse> gate_responses =
      LoadArray<TextureSampleResponse>(
          gate_pool, gate_final_state.texture_sample_responses);
  constexpr std::array<Rgba8, 4> kGateExpected = {{
      {{12, 112, 124, 255}},
      {{5, 54, 60, 255}},
      {{12, 112, 124, 255}},
      {{6, 61, 68, 255}},
  }};
  Check(gate_responses.size() == kGateExpected.size(),
        "Gate 18 returns one response per quad lane");
  for (std::size_t lane = 0; lane < kGateExpected.size(); ++lane) {
    Check(gate_responses[lane].request_id == lane &&
              gate_responses[lane].shader_lane_index == lane &&
              gate_responses[lane].shader_stage ==
                  pvrgpu::stub::ShaderStage::kFragment,
          "Gate 18 response preserves lane identity");
    for (std::size_t component = 0;
         component < kGateExpected[lane].size(); ++component) {
      Check(gate_responses[lane].rgba[component] ==
                FloatBits(static_cast<float>(kGateExpected[lane][component]) /
                          255.0F),
            "Gate 18 mip3/mip4 U8 trilinear result");
    }
  }

  PipelineTxn driver_completed;
  Check(driver_sample_output.nb_read(driver_completed) &&
            driver_completed.state.slot == driver_state_handle.slot &&
            driver_completed.state.generation ==
                driver_state_handle.generation &&
            driver_completed.frame == 3 && driver_completed.sequence == 3,
        "driver RGBX sample completion identity");
  const PipelineState driver_final_state =
      LoadPipelineState(driver_pool, driver_state_handle);
  const std::vector<TextureSampleResponse> driver_responses =
      LoadArray<TextureSampleResponse>(
          driver_pool, driver_final_state.texture_sample_responses);
  Check(driver_final_state.stage == PipelineStage::kTextureSamplesReady &&
            driver_final_state.counters.texture_requests == 1 &&
            driver_final_state.counters.texel_fetches == 1 &&
            driver_final_state.fragment_texture_request_count == 1 &&
            driver_final_state.fragment_texel_fetch_count == 1 &&
            driver_responses.size() == 1 &&
            driver_responses[0].shader_stage ==
                pvrgpu::stub::ShaderStage::kFragment &&
            driver_responses[0].rgba[0] == FloatBits(17.0F / 255.0F) &&
            driver_responses[0].rgba[1] == FloatBits(34.0F / 255.0F) &&
            driver_responses[0].rgba[2] == FloatBits(51.0F / 255.0F) &&
            driver_responses[0].rgba[3] == FloatBits(1.0F),
        "raw RGBX descriptor forces sampled alpha to one");

  PipelineTxn vertex_completed;
  Check(vertex_sample_output.nb_read(vertex_completed) &&
            vertex_completed.state.slot == vertex_state_handle.slot &&
            vertex_completed.state.generation ==
                vertex_state_handle.generation &&
            vertex_completed.frame == 4 && vertex_completed.sequence == 4,
        "vertex sample completion identity");
  const PipelineState vertex_final_state =
      LoadPipelineState(vertex_pool, vertex_state_handle);
  const auto vertex_responses = LoadArray<TextureSampleResponse>(
      vertex_pool, vertex_final_state.texture_sample_responses);
  Check(vertex_final_state.stage ==
                PipelineStage::kVertexTextureSamplesReady &&
            vertex_final_state.counters.texture_requests == 1 &&
            vertex_final_state.counters.texel_fetches == 1 &&
            vertex_final_state.vertex_texture_request_count == 1 &&
            vertex_final_state.vertex_texel_fetch_count == 1 &&
            vertex_final_state.fragment_texture_request_count == 0 &&
            vertex_final_state.fragment_texel_fetch_count == 0 &&
            vertex_final_state.counters.tiler_cycles ==
                pvrgpu::stub::kReferenceUarch.texture_bypass_cycles &&
            vertex_final_state.counters.renderer_cycles == 0 &&
            vertex_responses.size() == 1 &&
            vertex_responses[0].shader_stage ==
                pvrgpu::stub::ShaderStage::kVertex &&
            vertex_responses[0].rgba[0] == FloatBits(17.0F / 255.0F) &&
            vertex_responses[0].rgba[3] == FloatBits(1.0F),
        "vertex LOD0 request uses its own bank, response stage, and counters");

  PipelineTxn integer_completed;
  Check(integer_sample_output.nb_read(integer_completed) &&
            integer_completed.sequence == 5,
        "integer sample completion identity");
  const auto integer_final_state = LoadPipelineState(integer_pool, integer_state_handle);
  const auto integer_responses = LoadArray<TextureSampleResponse>(
      integer_pool, integer_final_state.texture_sample_responses);
  Check(integer_final_state.counters.texel_fetches == 2 &&
            integer_responses.size() == 2,
        "two integer requests execute two physical texel reads");
  const auto check_tap_memory = [memory_mode](
      const pvrgpu::stub::CounterTxn &counters, std::uint64_t taps,
      std::uint64_t tap_bytes, std::uint64_t requests, bool tiler_stage) {
    const bool cached = memory_mode == pvrgpu::stub::MemoryMode::kCache;
    const bool direct = memory_mode == pvrgpu::stub::MemoryMode::kDirect;
    const bool bypass = memory_mode == pvrgpu::stub::MemoryMode::kBypass;
    // These taps occupy one cold 128-byte SLC line. The warm reads must
    // remain modeled accesses, not memoized texture answers; direct and
    // bypass retain their different traffic accounting.
    Check(counters.slc_line_accesses == (cached ? taps : 0) &&
              counters.slc_read_accesses == (cached ? taps : 0) &&
              counters.slc_misses == (cached ? 1U : 0U) &&
              counters.slc_hits == (cached ? taps - 1U : 0U) &&
              counters.slc_bypassed == (bypass ? taps : 0) &&
              counters.slc_cycles ==
                  (cached ? taps * pvrgpu::stub::kMemorySlcLookupCycles : 0) &&
              counters.dram_read_transactions ==
                  (cached ? 1U : bypass ? taps : 0) &&
              counters.dram_read_bytes ==
                  (cached ? 128U : bypass ? taps * tap_bytes : 0) &&
              counters.dram_cycles ==
                  (cached ? 1U : bypass ? taps : 0) *
                      pvrgpu::stub::kMemoryDramRequestCycles &&
              counters.memory_direct_read_bytes ==
                  (direct ? taps * tap_bytes : 0) &&
              counters.slc_write_accesses == 0 &&
              counters.slc_evictions == 0 && counters.slc_writebacks == 0 &&
              counters.dram_write_transactions == 0 &&
              counters.dram_write_bytes == 0,
          "inline TPU reads preserve cold/warm/direct/bypass tap accounting");
    const auto cycles = requests * pvrgpu::stub::kReferenceUarch.texture_bypass_cycles +
                        counters.slc_cycles + counters.dram_cycles;
    Check(counters.texture_cycles == cycles &&
              counters.tiler_cycles == (tiler_stage ? cycles : 0) &&
              counters.renderer_cycles == (tiler_stage ? 0 : cycles),
          "inline reads retain modeled delay and texture cycle stage routing");
  };
  check_tap_memory(integer_final_state.counters, 2, 16, 2, false);
  for (std::size_t lane = 0; lane < integer_responses.size(); ++lane) {
    Check(std::equal(integer_texels[lane].begin(), integer_texels[lane].end(),
                     std::begin(integer_responses[lane].rgba)),
          "integer memory samples preserve all four DWORDs at the selected address");
  }
  ReleaseFunctionalPayloads(integer_pool, integer_final_state);
  PipelineTxn geometry_completed;
  Check(geometry_sample_output.nb_read(geometry_completed) && geometry_completed.sequence == 7,
        "independent GS texture FIFO completion identity");
  const auto geometry_final_state = LoadPipelineState(geometry_pool, geometry_state_handle);
  const auto geometry_responses = LoadArray<TextureSampleResponse>(
      geometry_pool, geometry_final_state.texture_sample_responses);
  Check(geometry_final_state.stage == PipelineStage::kGeometryTextureSamplesReady &&
        geometry_final_state.counters.texture_requests == 1 && geometry_final_state.counters.texel_fetches == 1 &&
        geometry_final_state.geometry_texture_request_count == 1 && geometry_final_state.geometry_texel_fetch_count == 1 &&
        !geometry_final_state.vertex_texture_request_count && !geometry_final_state.fragment_texture_request_count &&
        geometry_final_state.counters.tiler_cycles > 0 && !geometry_final_state.counters.renderer_cycles &&
        geometry_responses.size() == 1 && geometry_responses[0].shader_stage == pvrgpu::stub::ShaderStage::kGeometry &&
        geometry_responses[0].rgba[0] == FloatBits(17.0F / 255.0F) &&
        geometry_responses[0].rgba[1] == FloatBits(34.0F / 255.0F) &&
        geometry_responses[0].rgba[2] == FloatBits(51.0F / 255.0F) &&
        geometry_responses[0].rgba[3] == FloatBits(1.0F),
        "GS raw SH4 descriptor samples modeled memory without FS derivatives or stage leakage");
  ReleaseFunctionalPayloads(geometry_pool, geometry_final_state);
  check_tap_memory(geometry_final_state.counters, 1, 4, 1, true);
  geometry_pool.Release(geometry_state_handle);
  Check(!geometry_pool.bytes_in_flight() && geometry_pool.allocations() == geometry_pool.releases(),
        "GS texture nested resource ownership is balanced");

  // Reuse the elaborated TPU for a different format and filter. The four
  // physical taps must decode full binary32 channels before interpolation,
  // preserving signed values and values outside the normalized color range.
  const std::array<std::array<float, 4>, 2> float_texels = {{
      {{-4.0F, 0.125F, 1024.5F, 0.25F}},
      {{8.0F, 0.875F, 2048.5F, 0.75F}},
  }};
  for (std::size_t texel = 0; texel < 2; ++texel) {
    for (std::size_t channel = 0; channel < 4; ++channel) {
      const std::uint32_t bits = FloatBits(float_texels[texel][channel]);
      for (std::size_t byte = 0; byte < 4; ++byte) {
        integer_bytes[texel * 16 + channel * 4 + byte] =
            static_cast<std::uint8_t>(bits >> (byte * 8));
      }
    }
  }
  integer_memory.HostWrite(base, integer_bytes.data(), integer_bytes.size());
  integer_resource.format = TextureFormat::kRgba32Float;
  const std::uint64_t float_word0 =
      (integer_word0 & ~(UINT64_C(127) << 27U)) | (UINT64_C(61) << 27U);
  integer_shared[0] = static_cast<std::uint32_t>(float_word0);
  integer_shared[1] = static_cast<std::uint32_t>(float_word0 >> 32U);
  integer_shared[9] |= (UINT32_C(1) << 4U) | (UINT32_C(1) << 6U);
  integer_shared[17] = integer_shared[9];
  auto float_sampler = driver_fixture.sampler;
  float_sampler.min_filter = TextureFilter::kLinear;
  float_sampler.mag_filter = TextureFilter::kLinear;
  auto float_request = integer_requests[0];
  float_request.fcnorm = 1;
  float_request.coordinates[0] = FloatBits(0.5F);
  float_request.coordinates[1] = FloatBits(0.5F);
  std::copy_n(integer_shared.begin(), 4, float_request.texture_state);
  std::copy_n(integer_shared.begin() + 8, 4, float_request.sampler_state);
  PipelineState float_state;
  float_state.memory_mode = memory_mode;
  float_state.sequence = 6;
  float_state.functional_case = FunctionalCase::kDriverPcoTriangles;
  float_state.stage = PipelineStage::kFragmentTexturePending;
  float_state.sampled_texture_count = 1;
  float_state.fragment_pco_abi.shareds = integer_shared.size();
  float_state.fragment_shader_lane_count = 1;
  float_state.texture_sample_requests = StoreNewArray(
      integer_pool, std::vector<TextureSampleRequest>{float_request});
  float_state.texture_resources = StoreNewArray(
      integer_pool, std::vector<TextureResource>{integer_resource});
  float_state.sampler_states = StoreNewArray(
      integer_pool, std::vector<pvrgpu::stub::SamplerState>{float_sampler});
  float_state.fragment_shared_registers = StoreNewArray(integer_pool, integer_shared);
  StorePipelineState(integer_pool, integer_state_handle, float_state);
  integer_sample_input.write(PipelineTxn{integer_state_handle, 6, 6});
  sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_NS));
  Check(integer_sample_output.nb_read(integer_completed) &&
            integer_completed.sequence == 6,
        "float32 sample completion identity");
  const auto float_final_state = LoadPipelineState(integer_pool, integer_state_handle);
  const auto float_responses = LoadArray<TextureSampleResponse>(
      integer_pool, float_final_state.texture_sample_responses);
  const std::array<float, 4> expected_float = {{2.0F, 0.5F, 1536.5F, 0.5F}};
  Check(float_final_state.counters.texel_fetches == 4 && float_responses.size() == 1,
        "float32 bilinear sample executes four physical sixteen-byte reads");
  // HostWrite above replaced the previous integer contents at the same
  // address. The cache must refill once and return the new float bits.
  check_tap_memory(float_final_state.counters, 4, 16, 1, false);
  for (std::size_t channel = 0; channel < 4; ++channel) {
    Check(float_responses[0].rgba[channel] == FloatBits(expected_float[channel]),
          "float32 bilinear filter preserves negative, high-range and alpha channels");
  }
  ReleaseFunctionalPayloads(integer_pool, float_final_state);
  integer_pool.Release(integer_state_handle);
  Check(integer_pool.bytes_in_flight() == 0,
        "integer texture MemoryPool ownership balanced");

  ReleaseFunctionalPayloads(pool, final_state);
  pool.Release(state_handle);
  Check(pool.bytes_in_flight() == 0 &&
            pool.allocations() == pool.releases(),
        "MemoryPool ownership balanced");
  ReleaseFunctionalPayloads(gate_pool, gate_final_state);
  gate_pool.Release(gate_state_handle);
  Check(gate_pool.bytes_in_flight() == 0 &&
            gate_pool.allocations() == gate_pool.releases(),
        "Gate 18 MemoryPool ownership balanced");
  ReleaseFunctionalPayloads(driver_pool, driver_final_state);
  driver_pool.Release(driver_state_handle);
  Check(driver_pool.bytes_in_flight() == 0 &&
            driver_pool.allocations() == driver_pool.releases(),
        "driver RGBX MemoryPool ownership balanced");
  ReleaseFunctionalPayloads(vertex_pool, vertex_final_state);
  vertex_pool.Release(vertex_state_handle);
  Check(vertex_pool.bytes_in_flight() == 0 &&
            vertex_pool.allocations() == vertex_pool.releases(),
        "vertex TextureUnit MemoryPool ownership balanced");
}

} // namespace

int sc_main(int argc, char **argv) {
  try {
    pvrgpu::stub::MemoryMode memory_mode = pvrgpu::stub::MemoryMode::kDirect;
    if (argc == 2 && std::string(argv[1]) == "cache")
      memory_mode = pvrgpu::stub::MemoryMode::kCache;
    else if (argc == 2 && std::string(argv[1]) == "bypass")
      memory_mode = pvrgpu::stub::MemoryMode::kBypass;
    else if (argc != 1)
      throw std::runtime_error("usage: texture-unit-test [cache|bypass]");
    CheckDescriptorAndArithmetic();
    CheckIntegerImageDescriptors();
    CheckSequenceColorMipMaterialization();
    CheckCubeNonfiniteLod();
    CheckEventPaths(memory_mode);
    std::cout << "texture_unit_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "texture_unit_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
