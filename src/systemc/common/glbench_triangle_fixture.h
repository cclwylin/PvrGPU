// GLBench indexed-lattice functional fixture helpers.
// 中文：集中建立官方 GLBench CreateLattice/CreateMesh 資料。Triangle.Setup
// 使用 128x128 lattice；AttributeFetchShader 使用 64x64 tightly-packed float2
// lattice；VaryingsShader 使用 spacing=1/4 的 4x4 fullscreen lattice。
// 三者共用 j+=4 -> i -> j2 的 swath-order uint16 index stream。
// HalfCulled 使用 pinned macOS/Darwin srand(0) 的明確 Park-Miller sequence，
// 而非 process-global rand state，確保每幀及每次 simulation 都產生相同 winding。
// 此檔是資料演算法 helper，不是 SystemC module。
#pragma once

#include "common/functional_types.h"

#include <cstdint>
#include <vector>

namespace pvrgpu::stub {

inline constexpr std::uint32_t kGlbenchTriangleMeshWidth = 128;
inline constexpr std::uint32_t kGlbenchTriangleMeshHeight = 128;
inline constexpr std::uint32_t kGlbenchTriangleSwathHeight = 4;
inline constexpr std::uint32_t kGlbenchTriangleIndexCount = 98304;
inline constexpr std::uint32_t kGlbenchAttributeFetchMeshWidth = 64;
inline constexpr std::uint32_t kGlbenchAttributeFetchMeshHeight = 64;
inline constexpr std::uint32_t kGlbenchAttributeFetchIndexCount = 24576;
inline constexpr std::uint32_t kGlbenchVaryingsMeshWidth = 4;
inline constexpr std::uint32_t kGlbenchVaryingsMeshHeight = 4;
inline constexpr std::uint32_t kGlbenchVaryingsIndexCount = 96;

struct GlbenchTriangleMeshShape {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t swath_height = 0;
};

inline constexpr GlbenchTriangleMeshShape kGlbenchTriangleSetupMesh = {
    kGlbenchTriangleMeshWidth,
    kGlbenchTriangleMeshHeight,
    kGlbenchTriangleSwathHeight,
};

inline constexpr GlbenchTriangleMeshShape kGlbenchAttributeFetchMesh = {
    kGlbenchAttributeFetchMeshWidth,
    kGlbenchAttributeFetchMeshHeight,
    kGlbenchTriangleSwathHeight,
};

inline constexpr GlbenchTriangleMeshShape kGlbenchVaryingsMesh = {
    kGlbenchVaryingsMeshWidth,
    kGlbenchVaryingsMeshHeight,
    kGlbenchTriangleSwathHeight,
};

// GLBench calls srand(0), then rand() once per lattice cell. The pinned
// x86_64 macOS runner's libc implements rand/rand_r with this Park-Miller
// state transition, including its non-zero replacement for seed zero.
class GlbenchDarwinRand final {
public:
  explicit GlbenchDarwinRand(std::uint32_t seed = 0) : state_(seed) {}

  std::uint32_t Next();

private:
  std::uint32_t state_ = 0;
};

enum class GlbenchTriangleWindingPattern : std::uint8_t {
  kAllClockwise = 0,
  kSrandZeroHalfCulled,
};

std::vector<InputVertex>
MakeGlbenchTriangleVertices(std::uint32_t surface_width,
                            std::uint32_t surface_height);

std::vector<InputVertex>
MakeGlbenchTriangleVertices(std::uint32_t surface_width,
                            std::uint32_t surface_height,
                            const GlbenchTriangleMeshShape &mesh);

// AttributeFetchShader binds the same GLBench lattice as a tightly-packed
// GL_FLOAT size=2 VBO. Returning scalar floats preserves the exact 8-byte
// hardware stride rather than serializing the legacy InputVertex z component.
std::vector<float>
MakeGlbenchTriangleFloat2Vertices(std::uint32_t surface_width,
                                  std::uint32_t surface_height,
                                  const GlbenchTriangleMeshShape &mesh);

std::vector<std::uint16_t>
MakeGlbenchTriangleIndices(GlbenchTriangleWindingPattern pattern);

std::vector<std::uint16_t>
MakeGlbenchTriangleIndices(GlbenchTriangleWindingPattern pattern,
                           const GlbenchTriangleMeshShape &mesh);

} // namespace pvrgpu::stub
