// GLBench indexed-lattice functional fixture implementation.
// 中文：完全依官方 CreateLattice/CreateMesh loop nesting（j+=4 -> i -> j2）
// 建立 128x128 Triangle.Setup 或 64x64 AttributeFetchShader 資料。
// HalfCulled 每個 cell 只取一次 legacy random value，再同時決定該 cell 的
// 兩個 triangle winding；counter 仍由後續 VDM/clip/cull/raster 資料流計算，
// 不在此硬編。
#include "common/glbench_triangle_fixture.h"

#include <limits>
#include <stdexcept>

namespace pvrgpu::stub {
namespace {

inline constexpr std::uint32_t kDarwinRandMax = 2147483647U;
inline constexpr std::uint32_t kDarwinZeroSeedReplacement = 123459876U;
inline constexpr std::uint32_t kParkMillerModulus = 2147483647U;
inline constexpr std::uint32_t kParkMillerQuotient = 127773U;
inline constexpr std::uint32_t kParkMillerMultiplier = 16807U;
inline constexpr std::uint32_t kParkMillerRemainder = 2836U;

std::size_t VertexCount(const GlbenchTriangleMeshShape &mesh) {
  if (mesh.width == 0 || mesh.height == 0 || mesh.swath_height == 0 ||
      mesh.width % mesh.swath_height != 0 ||
      mesh.height % mesh.swath_height != 0) {
    throw std::runtime_error("GLBench lattice shape is invalid");
  }
  const std::uint64_t count =
      (static_cast<std::uint64_t>(mesh.width) + 1U) *
      (static_cast<std::uint64_t>(mesh.height) + 1U);
  if (count == 0 || count - 1U > std::numeric_limits<std::uint16_t>::max() ||
      count > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("GLBench lattice exceeds uint16 index range");
  }
  return static_cast<std::size_t>(count);
}

std::size_t IndexCount(const GlbenchTriangleMeshShape &mesh) {
  (void)VertexCount(mesh);
  const std::uint64_t count = static_cast<std::uint64_t>(mesh.width) *
                              mesh.height * 2U * 3U;
  if (count == 0 || count > std::numeric_limits<std::size_t>::max())
    throw std::runtime_error("GLBench index stream is too large");
  return static_cast<std::size_t>(count);
}

} // namespace

std::uint32_t GlbenchDarwinRand::Next() {
  std::int64_t value = state_;
  if (value == 0)
    value = kDarwinZeroSeedReplacement;
  const std::int64_t high = value / kParkMillerQuotient;
  const std::int64_t low = value % kParkMillerQuotient;
  value = static_cast<std::int64_t>(kParkMillerMultiplier) * low -
          static_cast<std::int64_t>(kParkMillerRemainder) * high;
  if (value < 0)
    value += kParkMillerModulus;
  if (value <= 0 || value >= kParkMillerModulus)
    throw std::runtime_error("GLBench legacy random state is out of range");
  state_ = static_cast<std::uint32_t>(value);
  return state_;
}

std::vector<InputVertex>
MakeGlbenchTriangleVertices(std::uint32_t surface_width,
                            std::uint32_t surface_height) {
  return MakeGlbenchTriangleVertices(surface_width, surface_height,
                                     kGlbenchTriangleSetupMesh);
}

std::vector<InputVertex>
MakeGlbenchTriangleVertices(std::uint32_t surface_width,
                            std::uint32_t surface_height,
                            const GlbenchTriangleMeshShape &mesh) {
  if (surface_width == 0 || surface_height == 0)
    throw std::runtime_error("GLBench lattice requires a non-empty surface");
  std::vector<InputVertex> vertices;
  vertices.reserve(VertexCount(mesh));
  const float size_x = 1.0F / static_cast<float>(surface_width);
  const float size_y = 1.0F / static_cast<float>(surface_height);
  const float shift_x = size_x * static_cast<float>(mesh.width);
  const float shift_y = size_y * static_cast<float>(mesh.height);
  for (std::uint32_t y = 0; y <= mesh.height; ++y) {
    for (std::uint32_t x = 0; x <= mesh.width; ++x) {
      vertices.push_back(
          {2.0F * static_cast<float>(x) * size_x - shift_x,
           2.0F * static_cast<float>(y) * size_y - shift_y, 0.0F});
    }
  }
  return vertices;
}

std::vector<float>
MakeGlbenchTriangleFloat2Vertices(std::uint32_t surface_width,
                                  std::uint32_t surface_height,
                                  const GlbenchTriangleMeshShape &mesh) {
  const std::vector<InputVertex> vertices =
      MakeGlbenchTriangleVertices(surface_width, surface_height, mesh);
  std::vector<float> packed;
  packed.reserve(vertices.size() * 2U);
  for (const InputVertex &vertex : vertices) {
    packed.push_back(vertex.x);
    packed.push_back(vertex.y);
  }
  return packed;
}

std::vector<std::uint16_t>
MakeGlbenchTriangleIndices(GlbenchTriangleWindingPattern pattern) {
  return MakeGlbenchTriangleIndices(pattern, kGlbenchTriangleSetupMesh);
}

std::vector<std::uint16_t>
MakeGlbenchTriangleIndices(GlbenchTriangleWindingPattern pattern,
                           const GlbenchTriangleMeshShape &mesh) {
  if (pattern != GlbenchTriangleWindingPattern::kAllClockwise &&
      pattern != GlbenchTriangleWindingPattern::kSrandZeroHalfCulled) {
    throw std::runtime_error("GLBench lattice winding pattern is invalid");
  }

  GlbenchDarwinRand random(0);
  std::vector<std::uint16_t> indices;
  const std::size_t expected_count = IndexCount(mesh);
  indices.reserve(expected_count);
  for (std::uint32_t y = 0; y < mesh.height; y += mesh.swath_height) {
    for (std::uint32_t x = 0; x < mesh.width; ++x) {
      for (std::uint32_t swath_y = 0;
           swath_y < mesh.swath_height; ++swath_y) {
        const std::uint32_t first =
            (y + swath_y) * (mesh.width + 1U) + x;
        const std::uint32_t second = first + 1;
        const std::uint32_t third = first + mesh.width + 1U;
        const std::uint32_t fourth = third + 1;
        const bool front_facing =
            pattern == GlbenchTriangleWindingPattern::kSrandZeroHalfCulled &&
            random.Next() < kDarwinRandMax / 2U;

        indices.push_back(static_cast<std::uint16_t>(first));
        indices.push_back(
            static_cast<std::uint16_t>(front_facing ? second : third));
        indices.push_back(
            static_cast<std::uint16_t>(front_facing ? third : second));

        indices.push_back(static_cast<std::uint16_t>(fourth));
        indices.push_back(
            static_cast<std::uint16_t>(front_facing ? third : second));
        indices.push_back(
            static_cast<std::uint16_t>(front_facing ? second : third));
      }
    }
  }
  if (indices.size() != expected_count)
    throw std::runtime_error("GLBench lattice index generation count mismatch");
  return indices;
}

} // namespace pvrgpu::stub
