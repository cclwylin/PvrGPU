// SPDX-License-Identifier: MIT
#include "common/tessellation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

// A private differential build links the unmodified pinned Mesa source.
// The normal repository test has no Mesa dependency and checks invariants.
#ifdef PVRGPU_MESA_TESSELLATION_REFERENCE
#include "tessellator.hpp"
#endif

namespace {
using namespace pvrgpu::stub;
using Domain = TessellationDomain;
using Spacing = TessellationSpacing;
using Status = TessellationStatus;
std::uint64_t checks = 0;
std::uint64_t patches = 0;
void Check(bool ok, const char *reason) {
  ++checks;
  if (!ok) throw std::runtime_error(reason);
}

struct Storage {
  std::array<TessellationDomainPoint, kTessellationMaxPoints> points{};
  std::array<std::uint32_t, kTessellationMaxIndices> indices{};
  TessellationStorage view() {
    return {points.data(), points.size(), indices.data(), indices.size()};
  }
};

TessellationRequest Request(Domain domain, Spacing spacing, float level) {
  TessellationRequest request;
  request.domain = domain;
  request.spacing = spacing;
  std::fill(std::begin(request.outer), std::end(request.outer), level);
  std::fill(std::begin(request.inner), std::end(request.inner), level);
  return request;
}

TessellationResult Run(TessellationRequest request, Storage &storage) {
  TessellationResult result;
  Check(TessellatePatch(request, storage.view(), result) == Status::kSuccess,
        "valid tessellation rejected");
  ++patches;
  Check(result.point_count <= kTessellationMaxPoints, "point count bound");
  Check(result.index_count <= kTessellationMaxIndices, "index count bound");
  Check(result.primitive_size == (request.point_mode ? 1u :
        request.domain == Domain::kIsolines ? 2u : 3u), "primitive size");
  Check(result.index_count % result.primitive_size == 0, "complete primitives");
  for (std::uint32_t i = 0; i < result.point_count; ++i) {
    const auto point = storage.points[i];
    Check(std::isfinite(point.u) && point.u >= 0 && point.u <= 1, "U domain");
    Check(std::isfinite(point.v) && point.v >= 0 && point.v <= 1, "V domain");
    const float w = TessellationCoordinateW(request.domain, point);
    Check(w >= 0 && w <= 1, "W domain");
    if (request.domain != Domain::kTriangles) Check(w == 0, "nontriangle W");
    else Check(point.u + point.v + w == 1.0f, "barycentric sum");
  }
  for (std::uint32_t i = 0; i < result.index_count; ++i)
    Check(storage.indices[i] < result.point_count, "index in point range");
  if (request.point_mode) {
    Check(result.index_count == result.point_count, "point mode one index per point");
    for (std::uint32_t i = 0; i < result.index_count; ++i)
      Check(storage.indices[i] == i, "point mode identity indices");
  } else if (request.domain != Domain::kIsolines) {
    for (std::uint32_t i = 0; i < result.index_count; i += 3) {
      const auto a = storage.points[storage.indices[i]];
      const auto b = storage.points[storage.indices[i + 1]];
      const auto c = storage.points[storage.indices[i + 2]];
      const double area = (double(b.u) - a.u) * (double(c.v) - a.v) -
                          (double(b.v) - a.v) * (double(c.u) - a.u);
      // Fractional transitions can coincide after the reference's 16-bit
      // fixed-point coordinate rounding; degenerate triangles are legal.
      Check(request.clockwise ? area <= 0 : area >= 0, "GL domain winding");
    }
  }
#ifdef PVRGPU_MESA_TESSELLATION_REFERENCE
  CHWTessellator reference;
  auto partition = request.spacing == Spacing::kEqual ? PIPE_TESSELLATOR_PARTITIONING_INTEGER :
    request.spacing == Spacing::kFractionalEven ? PIPE_TESSELLATOR_PARTITIONING_FRACTIONAL_EVEN :
                                               PIPE_TESSELLATOR_PARTITIONING_FRACTIONAL_ODD;
  auto output = request.point_mode ? PIPE_TESSELLATOR_OUTPUT_POINT :
    request.domain == Domain::kIsolines ? PIPE_TESSELLATOR_OUTPUT_LINE :
    request.clockwise ? PIPE_TESSELLATOR_OUTPUT_TRIANGLE_CCW : PIPE_TESSELLATOR_OUTPUT_TRIANGLE_CW;
  // Mesa's reference constructor leaves m_parity uninitialized; its integer
  // Init copies that enum before later deriving actual per-edge parity. Give
  // the reference object a valid preceding state without editing its source.
  reference.Init(PIPE_TESSELLATOR_PARTITIONING_FRACTIONAL_EVEN, output);
  reference.Init(partition, output);
  if (request.domain == Domain::kTriangles)
    reference.TessellateTriDomain(request.outer[0], request.outer[1], request.outer[2], request.inner[0]);
  else if (request.domain == Domain::kQuads)
    reference.TessellateQuadDomain(request.outer[0], request.outer[1], request.outer[2], request.outer[3], request.inner[0], request.inner[1]);
  else reference.TessellateIsoLineDomain(request.outer[0], request.outer[1]);
  Check(reference.GetPointCount() == static_cast<int>(result.point_count), "Mesa point count");
  Check(reference.GetIndexCount() == static_cast<int>(result.index_count), "Mesa index count");
  static_assert(sizeof(DOMAIN_POINT) == sizeof(TessellationDomainPoint));
  Check(std::memcmp(storage.points.data(), reference.GetPoints(), result.point_count * sizeof(DOMAIN_POINT)) == 0,
        "Mesa domain coordinate bits");
  Check(std::memcmp(storage.indices.data(), reference.GetIndices(), result.index_count * sizeof(std::uint32_t)) == 0,
        "Mesa exact connectivity/winding");
#endif
  return result;
}

void Validation() {
  Storage storage;
  auto request = Request(Domain::kTriangles, Spacing::kEqual, 1);
  const auto before = storage;
  auto expect = [&](TessellationRequest input, TessellationStorage view, Status status) {
    TessellationResult result{9, 9, 9};
    Check(TessellatePatch(input, view, result) == status, "invalid request status");
    Check(result.point_count == 0 && result.index_count == 0 && result.primitive_size == 0,
          "invalid request zero result");
    Check(std::memcmp(&before, &storage, sizeof(storage)) == 0, "invalid request no output writes");
  };
  auto bad = request; bad.domain = static_cast<Domain>(255);
  expect(bad, storage.view(), Status::kInvalidDomain);
  bad = request; bad.spacing = static_cast<Spacing>(255);
  expect(bad, storage.view(), Status::kInvalidSpacing);
  bad = request; bad.clockwise = 2;
  expect(bad, storage.view(), Status::kInvalidFlags);
  bad = request; bad.point_mode = 2;
  expect(bad, storage.view(), Status::kInvalidFlags);
  auto view = storage.view(); view.points = nullptr;
  expect(request, view, Status::kInvalidStorage);
  view = storage.view(); view.indices = nullptr;
  expect(request, view, Status::kInvalidStorage);
  view = storage.view(); --view.point_capacity;
  expect(request, view, Status::kInsufficientCapacity);
  view = storage.view(); --view.index_capacity;
  expect(request, view, Status::kInsufficientCapacity);
  view = storage.view(); view.indices = reinterpret_cast<std::uint32_t *>(storage.points.data());
  expect(request, view, Status::kInvalidStorage);
  view = storage.view(); view.points = reinterpret_cast<TessellationDomainPoint *>(reinterpret_cast<std::uintptr_t>(view.points) + 1);
  expect(request, view, Status::kInvalidStorage);
  view = storage.view(); view.indices = reinterpret_cast<std::uint32_t *>(reinterpret_cast<std::uintptr_t>(view.indices) + 1);
  expect(request, view, Status::kInvalidStorage);
}

void Levels() {
  Storage storage;
  constexpr Domain domains[]{Domain::kTriangles, Domain::kQuads, Domain::kIsolines};
  constexpr Spacing spacings[]{Spacing::kEqual, Spacing::kFractionalEven, Spacing::kFractionalOdd};
  const float specials[]{-std::numeric_limits<float>::infinity(), -4, -0.0f, 0,
    std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::min(),
    0.01f, 0.999999f, 1, 1.000001f, 1.01f, 1.5f, 1.999999f, 2, 2.000001f,
    2.5f, 3, 3.5f, 31.7f, 62.9f, 63, 63.1f, 64, 65, 1000000,
    std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
  for (auto domain : domains) for (auto spacing : spacings) {
    for (float level : specials) for (unsigned flags = 0; flags < 4; ++flags) {
      auto request = Request(domain, spacing, level);
      request.clockwise = flags & 1;
      request.point_mode = flags >> 1;
      auto result = Run(request, storage);
      Check((result.point_count == 0) == !(level > 0), "outer NaN/nonpositive culls only");
    }
    // Independently vary every level: unused outer/inner NaN must not cull.
    for (unsigned slot = 0; slot < 6; ++slot) for (float level : specials) {
      auto request = Request(domain, spacing, 3.5f);
      if (slot < 4) request.outer[slot] = level;
      else request.inner[slot - 4] = level;
      auto result = Run(request, storage);
      unsigned outer_count = domain == Domain::kQuads ? 4 : domain == Domain::kTriangles ? 3 : 2;
      Check((result.point_count == 0) == (slot < outer_count && !(level > 0)), "individual outer cull");
    }
    for (unsigned level = 1; level <= 64; ++level) {
      auto request = Request(domain, spacing, static_cast<float>(level));
      const auto result = Run(request, storage);
      if (spacing != Spacing::kEqual) continue;
      if (domain == Domain::kQuads) {
        Check(result.point_count == (level + 1) * (level + 1), "equal quad grid count");
        Check(result.index_count == level * level * 6, "equal quad triangle count");
      } else if (domain == Domain::kIsolines) {
        Check(result.point_count == level * (level + 1), "equal isoline grid count");
        Check(result.index_count == level * level * 2, "equal isoline segment count");
      }
    }
  }
}

void EdgesAndSymmetry() {
  Storage first, second;
  for (auto spacing : {Spacing::kEqual, Spacing::kFractionalEven, Spacing::kFractionalOdd}) {
    for (float level : {0.1f, 1.0f, 1.1f, 1.8f, 2.1f, 7.3f, 31.9f, 63.1f, 64.0f}) {
      auto request = Request(Domain::kQuads, spacing, 9.7f);
      request.outer[0] = level;
      const auto one = Run(request, first);
      request.domain = Domain::kTriangles;
      request.inner[0] = 13.3f;
      const auto two = Run(request, second);
      std::vector<float> quad_edge, tri_edge;
      for (unsigned i = 0; i < one.point_count; ++i)
        if (first.points[i].u == 0) quad_edge.push_back(first.points[i].v);
      for (unsigned i = 0; i < two.point_count; ++i)
        if (second.points[i].u == 0) tri_edge.push_back(second.points[i].v);
      std::sort(quad_edge.begin(), quad_edge.end());
      std::sort(tri_edge.begin(), tri_edge.end());
      Check(quad_edge == tri_edge, "matching outer factors give identical tri/quad boundary bits");
      for (unsigned i = 0; i < quad_edge.size(); ++i)
        Check(quad_edge[i] == 1.0f - quad_edge[quad_edge.size() - 1 - i], "edge partition reflection symmetry");
      request = Request(Domain::kIsolines, spacing, 1);
      request.outer[1] = level;
      const auto lines = Run(request, second);
      std::vector<float> line_edge;
      for (unsigned i = 0; i < lines.point_count; ++i)
        if (second.points[i].v == 0) line_edge.push_back(second.points[i].u);
      std::sort(line_edge.begin(), line_edge.end());
      Check(quad_edge == line_edge, "matching isoline detail has same edge partition");
    }
  }
  auto request = Request(Domain::kQuads, Spacing::kEqual, 64);
  const auto maximum = Run(request, first);
  Check(maximum.point_count == kTessellationMaxPoints && maximum.index_count == kTessellationMaxIndices,
        "maximum storage bounds attained");
  request.clockwise = 1;
  const auto reversed = Run(request, second);
  Check(maximum.point_count == reversed.point_count && maximum.index_count == reversed.index_count,
        "winding invariant counts");
  Check(std::memcmp(first.points.data(), second.points.data(), maximum.point_count * sizeof(TessellationDomainPoint)) == 0,
        "winding invariant domain bits");
  for (unsigned i = 0; i < maximum.index_count; i += 3) {
    Check(first.indices[i] == second.indices[i] && first.indices[i + 1] == second.indices[i + 2] &&
          first.indices[i + 2] == second.indices[i + 1], "winding exact reverse");
  }
}

void Randomized() {
  Storage storage;
  std::uint32_t random = 0x4bd769e3;
  auto next = [&]() { random ^= random << 13; random ^= random >> 17; random ^= random << 5; return random; };
  for (unsigned i = 0; i < 2400; ++i) {
    TessellationRequest request;
    request.domain = static_cast<Domain>(i % 3);
    request.spacing = static_cast<Spacing>((i / 3) % 3);
    request.clockwise = (i / 9) & 1;
    request.point_mode = (i / 18) & 1;
    for (float &level : request.outer) level = (next() % 70000 + 1) / 1000.0f;
    for (float &level : request.inner) level = (next() % 70000 + 1) / 1000.0f;
    Run(request, storage);
  }
}
}  // namespace

int main() {
  try {
    Validation();
    Levels();
    EdgesAndSymmetry();
    Randomized();
    std::printf("tessellation_test: %llu checks, %llu patches passed\n",
                static_cast<unsigned long long>(checks), static_cast<unsigned long long>(patches));
  } catch (const std::exception &error) {
    std::fprintf(stderr, "tessellation_test failed after %llu checks / %llu patches: %s\n",
                 static_cast<unsigned long long>(checks), static_cast<unsigned long long>(patches), error.what());
    return 1;
  }
  return 0;
}
