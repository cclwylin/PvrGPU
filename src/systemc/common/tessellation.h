// SPDX-License-Identifier: MIT
// Pure bounded fixed-function tessellation; never executes shader code.
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace pvrgpu::stub {

constexpr std::uint32_t kTessellationMaxLevel = 64;
constexpr std::uint32_t kTessellationMaxPoints = 65 * 65;
constexpr std::uint32_t kTessellationMaxIndices = 64 * 64 * 2 * 3;

enum class TessellationDomain : std::uint8_t { kTriangles, kQuads, kIsolines };
enum class TessellationSpacing : std::uint8_t {
  kEqual, kFractionalEven, kFractionalOdd,
};
enum class TessellationStatus : std::uint8_t {
  kSuccess, kInvalidDomain, kInvalidSpacing, kInvalidFlags,
  kInvalidStorage, kInsufficientCapacity, kInternalRange,
};

// Float levels are the actual TCS outputs, before clamp/rounding. Unused outer
// and inner levels are ignored exactly as in llvmpipe's fixed-function path.
// clockwise is the GLSL TES declaration, not the D3D reference's orientation.
struct TessellationRequest {
  float outer[4]{};
  float inner[2]{};
  TessellationDomain domain = TessellationDomain::kTriangles;
  TessellationSpacing spacing = TessellationSpacing::kEqual;
  std::uint8_t clockwise = 0;
  std::uint8_t point_mode = 0;
};

// llvmpipe supplies U/V to TES and calculates triangle W as (1 - U) - V.
// PvrGPU preserves those exact fixed-point-derived U/V bits; shader input
// setup computes W by the same ordered operations (otherwise W is zero).
struct TessellationDomainPoint { float u = 0; float v = 0; };

struct TessellationStorage {
  TessellationDomainPoint *points = nullptr;
  std::size_t point_capacity = 0;
  std::uint32_t *indices = nullptr;
  std::size_t index_capacity = 0;
};

struct TessellationResult {
  std::uint32_t point_count = 0;
  std::uint32_t index_count = 0;
  std::uint32_t primitive_size = 0;
};

// Caller owns storage, normally an exclusive pool mapping, with at least
// kTessellationMaxPoints / kTessellationMaxIndices entries. Capacity and state
// validation occurs before any writes. No allocation, global mutable state,
// hidden queues or host shader execution. Cull is successful zero counts.
TessellationStatus TessellatePatch(const TessellationRequest &request,
                                 TessellationStorage storage,
                                 TessellationResult &result);
const char *TessellationStatusName(TessellationStatus status);
float TessellationCoordinateW(TessellationDomain domain,
                             TessellationDomainPoint point);

static_assert(std::is_trivially_copyable_v<TessellationRequest>);
static_assert(std::is_trivially_copyable_v<TessellationDomainPoint>);
static_assert(std::is_trivially_copyable_v<TessellationResult>);
static_assert(sizeof(TessellationRequest) == 28);
static_assert(sizeof(TessellationDomainPoint) == 8);

}  // namespace pvrgpu::stub
