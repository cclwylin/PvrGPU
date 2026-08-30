// Exact pinned GLBench Fill.Texture nearest/bilinear/trilinear command and
// resource fixture.
// It builds the full 512x512 RGBA8 mip allocation and the public Rogue
// image/sampler descriptor dwords consumed by the PCO SMP instruction.
#pragma once

#include "common/functional_types.h"

#include <array>
#include <cstdint>
#include <vector>

namespace pvrgpu::stub {

inline constexpr std::uint64_t kGlbenchTextureGpuAddress =
    UINT64_C(0x40000000);

struct GlbenchFillTextureFixture {
  std::vector<float> positions;
  std::vector<float> texture_coordinates;
  std::vector<std::uint8_t> texture_bytes;
  std::uint32_t vertex_scale_bits = UINT32_C(0x3f800000);
  TextureResource resource;
  SamplerState sampler;
  std::array<std::uint32_t, kFillTexNearestSharedDwordCount>
      fragment_shared{};
};

GlbenchFillTextureFixture MakeGlbenchFillTextureFixture(TextureFilter filter);
GlbenchFillTextureFixture
MakeGlbenchFillTextureFixture(FunctionalCase functional_case);

} // namespace pvrgpu::stub
