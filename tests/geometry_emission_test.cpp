// SPDX-License-Identifier: MIT
#include "common/geometry_emission.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using namespace pvrgpu::stub;
using Status = GeometryEmissionStatus;
using Output = GeometryOutputTopology;
using Input = GeometryInputTopology;
std::uint64_t checks = 0;

void Check(bool condition, const char *reason) {
  ++checks;
  if (!condition) throw std::runtime_error(reason);
}

bool Equal(const GeometryPrimitiveRef &a, const GeometryPrimitiveRef &b) {
  return std::equal(std::begin(a.vertex_indices), std::end(a.vertex_indices),
                    std::begin(b.vertex_indices)) &&
         a.vertex_count == b.vertex_count &&
         a.provoking_vertex == b.provoking_vertex;
}

struct Storage {
  std::array<std::uint32_t, 256 * 64> words{};
  std::array<std::uint64_t, 256> masks{};
  std::array<GeometryStripRange, 256> strips{};
  GeometryEmissionStorage view() {
    return {words.data(), words.size(), masks.data(), masks.size(),
            strips.data(), strips.size()};
  }
};

// Optional external differential build includes Mesa's actual source template.
// Ordinary CTest remains self-contained and checks the same spec identities.
#if defined(PVRGPU_MESA_GEOMETRY_REFERENCE)
namespace mesa_reference {
enum {
  MESA_PRIM_POINTS, MESA_PRIM_LINES, MESA_PRIM_LINE_LOOP,
  MESA_PRIM_LINE_STRIP, MESA_PRIM_TRIANGLES, MESA_PRIM_TRIANGLE_STRIP,
  MESA_PRIM_TRIANGLE_FAN, MESA_PRIM_QUADS, MESA_PRIM_QUAD_STRIP,
  MESA_PRIM_POLYGON, MESA_PRIM_LINES_ADJACENCY,
  MESA_PRIM_LINE_STRIP_ADJACENCY, MESA_PRIM_TRIANGLES_ADJACENCY,
  MESA_PRIM_TRIANGLE_STRIP_ADJACENCY
};
enum {
  DRAW_PIPE_RESET_STIPPLE = 1, DRAW_PIPE_EDGE_FLAG_0 = 2,
  DRAW_PIPE_EDGE_FLAG_1 = 4, DRAW_PIPE_EDGE_FLAG_2 = 8,
  DRAW_PIPE_EDGE_FLAG_ALL = 14, DRAW_SPLIT_BEFORE = 16, DRAW_SPLIT_AFTER = 32
};
void debug_printf(const char *, ...) {}
#define UNUSED [[maybe_unused]]
#define FUNC Decompose
#define FUNC_VARS unsigned prim, const std::uint32_t *indices, unsigned count, std::vector<GeometryInputPrimitive> &out
#define FUNC_ENTER const unsigned prim_flags = 0; const bool last_vertex_last = true; const bool quads_flatshade_last = false
#define GET_ELT(idx) (indices[idx])
#define POINT(a) out.push_back(GeometryInputPrimitive{{a}, 0, 0, 1, {}})
#define LINE(flags, a, b) out.push_back(GeometryInputPrimitive{{a, b}, 0, 0, 2, {}})
#define TRIANGLE(flags, a, b, c) out.push_back(GeometryInputPrimitive{{a, b, c}, 0, 0, 3, {}})
#define LINE_ADJ(flags, a, b, c, d) out.push_back(GeometryInputPrimitive{{a, b, c, d}, 0, 0, 4, {}})
#define TRIANGLE_ADJ(flags, a, b, c, d, e, f) out.push_back(GeometryInputPrimitive{{a, b, c, d, e, f}, 0, 0, 6, {}})
#include "draw_decompose_tmp.h"
#undef UNUSED
unsigned Mode(Input topology) {
  switch (topology) {
    case Input::kPoints: return MESA_PRIM_POINTS;
    case Input::kLines: return MESA_PRIM_LINES;
    case Input::kLineStrip: return MESA_PRIM_LINE_STRIP;
    case Input::kLineLoop: return MESA_PRIM_LINE_LOOP;
    case Input::kTriangles: return MESA_PRIM_TRIANGLES;
    case Input::kTriangleStrip: return MESA_PRIM_TRIANGLE_STRIP;
    case Input::kTriangleFan: return MESA_PRIM_TRIANGLE_FAN;
    case Input::kLinesAdjacency: return MESA_PRIM_LINES_ADJACENCY;
    case Input::kLineStripAdjacency: return MESA_PRIM_LINE_STRIP_ADJACENCY;
    case Input::kTrianglesAdjacency: return MESA_PRIM_TRIANGLES_ADJACENCY;
    case Input::kTriangleStripAdjacency: return MESA_PRIM_TRIANGLE_STRIP_ADJACENCY;
  }
  throw std::runtime_error("unknown reference topology");
}
}  // namespace mesa_reference
#endif

void TestEventTraces() {
  Storage storage;
  const std::array<std::uint32_t, 4> raw = {
      UINT32_C(0xffffffff), UINT32_C(0x80000000),
      UINT32_C(0x7fa01234), UINT32_C(0x00000001)};
  for (Output topology : {Output::kPoints, Output::kLineStrip,
                           Output::kTriangleStrip}) {
    for (unsigned limit : {0U, 1U, 2U, 3U, 7U, 256U}) {
      for (unsigned length = 0; length <= 12; ++length) {
        for (unsigned bits = 0; bits < (1U << length); ++bits) {
          GeometryEmissionBuffer emitter(storage.view(), limit, raw.size());
          std::vector<unsigned> lengths;
          unsigned total = 0, open = 0;
          for (unsigned i = 0; i < length; ++i) {
            if (bits & (1U << i)) {
              const auto result = emitter.Emit(raw.data(), raw.size(), 0x5);
              Check(result == (total < limit ? Status::kSuccess :
                                  Status::kEmitSuppressed), "emit max bound");
              if (total < limit) { ++total; ++open; }
            } else {
              Check(emitter.EndPrimitive() == Status::kSuccess, "explicit end");
              if (open) lengths.push_back(open);
              open = 0;
            }
          }
          if (open) lengths.push_back(open);
          Check(emitter.Finish() == Status::kSuccess, "implicit final end");
          Check(emitter.Finish() == Status::kSuccess, "repeat finish no-op");
          Check(emitter.finished(), "finish state");
          Check(emitter.Emit(nullptr, 0, UINT64_MAX) == Status::kFinished,
                "post finish emit rejected");
          Check(emitter.EndPrimitive() == Status::kFinished,
                "post finish cut rejected");
          Check(total == emitter.emitted_vertices() &&
                    lengths.size() == emitter.strip_count(), "trace counts");
          std::vector<GeometryPrimitiveRef> reference;
          unsigned first = 0;
          for (unsigned n : lengths) {
#if defined(PVRGPU_MESA_GEOMETRY_REFERENCE)
            std::array<std::uint32_t, 256> indices;
            for (unsigned i = 0; i < n; ++i) indices[i] = first + i;
            std::vector<GeometryInputPrimitive> prims;
            const Input input_topology = topology == Output::kPoints ? Input::kPoints :
                topology == Output::kLineStrip ? Input::kLineStrip : Input::kTriangleStrip;
            mesa_reference::Decompose(mesa_reference::Mode(input_topology),
                                        indices.data(), n, prims);
            for (const auto &p : prims) {
              const unsigned a = p.vertex_indices[0];
              const unsigned b = p.vertex_count > 1 ? p.vertex_indices[1] : a;
              const unsigned c = p.vertex_count > 2 ? p.vertex_indices[2] : b;
              reference.push_back({{a, b, c}, p.vertex_count,
                                    std::uint8_t(p.vertex_count - 1), {}});
            }
#else
            const unsigned width = topology == Output::kPoints ? 1 :
                topology == Output::kLineStrip ? 2 : 3;
            for (unsigned p = 0; p + width <= n; ++p) {
              std::array<unsigned, 3> v{first + p, first + p + width - 1,
                                        first + p + width - 1};
              if (width == 3) {
                v[0] = first + p + (p % 2);
                v[1] = first + p + 1 - (p % 2);
              }
              reference.push_back({{v[0], v[1], v[2]}, std::uint8_t(width),
                                    std::uint8_t(width - 1), {}});
            }
#endif
            first += n;
          }
          std::array<GeometryPrimitiveRef, 256> actual{};
          std::size_t written = 999;
          Check(ExpandGeometryPrimitives(topology, storage.strips.data(),
                    emitter.strip_count(), total, actual.data(), actual.size(),
                    written) == Status::kSuccess, "expand trace");
          Check(written == reference.size(), "trace primitive count");
          for (std::size_t i = 0; i < written; ++i)
            Check(Equal(actual[i], reference[i]), "trace primitive order");
          for (unsigned i = 0; i < total; ++i) {
            Check(std::equal(raw.begin(), raw.end(), storage.words.begin() + i * 4),
                  "raw DWORD snapshot exact");
            Check(storage.masks[i] == 5, "undefined output mask preserved");
          }
          if (written) {
            const auto before = actual;
            std::size_t unchanged = 123;
            Check(ExpandGeometryPrimitives(topology, storage.strips.data(),
                      emitter.strip_count(), total, actual.data(), written - 1,
                      unchanged) == Status::kInsufficientCapacity,
                  "expand rejects insufficient capacity");
            Check(unchanged == 123 && !std::memcmp(before.data(), actual.data(),
                    sizeof(actual)), "expand failure is transactional");
          }
        }
      }
    }
  }
}

void TestSnapshotsAndBounds() {
  Storage storage;
  GeometryEmissionBuffer emitter(storage.view(), 256, 64);
  std::array<std::uint32_t, 64> outputs;
  for (unsigned i = 0; i < 256; ++i) {
    for (unsigned c = 0; c < outputs.size(); ++c)
      outputs[c] = UINT32_C(0xff810001) ^ (i * 67 + c * 13);
    Check(emitter.Emit(outputs.data(), outputs.size(), UINT64_MAX) == Status::kSuccess,
          "64 DWORD native output snapshot");
  }
  Check(emitter.Emit(nullptr, 0, UINT64_MAX) == Status::kEmitSuppressed,
        "257th emit suppressed without reading source");
  Check(emitter.EndPrimitive() == Status::kSuccess &&
            emitter.Finish() == Status::kSuccess, "cut/end survive maximum");
  for (unsigned i = 0; i < 256; ++i) {
    for (unsigned c = 0; c < outputs.size(); ++c)
      Check(storage.words[i * 64 + c] ==
                (UINT32_C(0xff810001) ^ (i * 67 + c * 13)),
            "successive writes do not mutate earlier raw snapshots");
  }
  GeometryEmissionBuffer zero({}, 0, 4);
  Check(zero.Emit(nullptr, 0, UINT64_MAX) == Status::kEmitSuppressed,
        "legal zero-max shader emits nothing");
  Check(zero.EndPrimitive() == Status::kSuccess && zero.Finish() == Status::kSuccess,
        "zero-max native completion");
  for (unsigned width : {0U, 65U, UINT32_MAX}) {
    GeometryEmissionBuffer invalid(storage.view(), 1, width);
    Check(invalid.Emit(outputs.data(), 64, 0) == Status::kInvalidOutputDwordCount,
          "invalid output width");
  }
  GeometryEmissionBuffer bad_mask(storage.view(), 1, 4);
  Check(bad_mask.Emit(outputs.data(), 4, 0x10) == Status::kInvalidWrittenMask,
        "reject writes outside output register span");
  Check(bad_mask.emitted_vertices() == 0, "invalid emit leaves counters intact");
  auto view = storage.view();
  view.strip_capacity = 0;
  GeometryEmissionBuffer no_strips(view, 1, 4);
  const auto before = storage.words;
  Check(no_strips.Emit(outputs.data(), 4, 0xf) == Status::kInsufficientCapacity,
        "reserve a strip before accepting its first snapshot");
  Check(before == storage.words && no_strips.emitted_vertices() == 0,
        "rejected strip reservation changes no storage");
  view = storage.view();
  view.snapshot_dword_capacity = 3;
  GeometryEmissionBuffer no_words(view, 1, 4);
  Check(no_words.Emit(outputs.data(), 4, 0xf) == Status::kInsufficientCapacity,
        "snapshot allocation is bounded");
  view = storage.view();
  view.written_mask_capacity = 0;
  GeometryEmissionBuffer no_masks(view, 1, 4);
  Check(no_masks.Emit(outputs.data(), 4, 0xf) == Status::kInsufficientCapacity,
        "defined-mask allocation is bounded");
  std::array<GeometryPrimitiveRef, 4> refs{};
  std::size_t unchanged = 33;
  for (GeometryStripRange bad : {GeometryStripRange{1, 1}, {0, 0}, {0, 4},
                                  {UINT32_MAX, 2}}) {
    Check(ExpandGeometryPrimitives(Output::kPoints, &bad, 1, 3,
              refs.data(), refs.size(), unchanged) == Status::kInvalidRange,
          "malformed strips cannot escape snapshot allocation");
    Check(unchanged == 33, "invalid ranges do not publish a count");
  }
}

void TestInputAssembly() {
  constexpr std::uint32_t restart = UINT32_MAX;
  std::array<std::uint32_t, 36> indices{};
  for (unsigned i = 0; i < indices.size(); ++i) indices[i] = 101 + i * 7;
  for (unsigned mode = unsigned(Input::kPoints);
       mode <= unsigned(Input::kTriangleStripAdjacency); ++mode) {
    const Input topology = static_cast<Input>(mode);
    for (std::size_t count = 0; count <= indices.size(); ++count) {
      for (unsigned pattern = 0; pattern < 8; ++pattern) {
        auto source = indices;
        for (std::size_t i = 0; i < count; ++i) {
          if (((pattern & 1) && i % 5 == 0) ||
              ((pattern & 2) && i == count / 2) ||
              ((pattern & 4) && i + 1 == count)) source[i] = restart;
        }
        for (bool restart_enabled : {false, true}) {
          std::array<GeometryInputPrimitive, 36> actual{};
          std::size_t written = 999;
          Check(AssembleGeometryInputPrimitives(topology, source.data(), count,
                    restart_enabled, restart, 13, actual.data(), actual.size(),
                    written) == Status::kSuccess, "assemble all input topologies");
          for (std::size_t i = 0; i < written; ++i) {
            Check(actual[i].primitive_id == i && actual[i].instance_id == 13,
                  "restart preserves primitive ID and instance identity");
            Check(actual[i].vertex_count >= 1 && actual[i].vertex_count <= 6,
                  "input primitive lane span bound");
            for (unsigned c = 0; c < actual[i].vertex_count; ++c) {
              Check(std::find(source.begin(), source.begin() + count,
                                actual[i].vertex_indices[c]) != source.begin() + count,
                    "input assembly invents no vertex");
              Check(!restart_enabled || actual[i].vertex_indices[c] != restart,
                    "restart index never enters GS gl_in");
            }
          }
#if defined(PVRGPU_MESA_GEOMETRY_REFERENCE)
          std::vector<GeometryInputPrimitive> reference;
          std::size_t first = 0;
          for (std::size_t i = 0; i <= count; ++i) {
            if (i == count || (restart_enabled && source[i] == restart)) {
              if (i != first)
                mesa_reference::Decompose(mesa_reference::Mode(topology),
                    source.data() + first, unsigned(i - first), reference);
              first = i + 1;
            }
          }
          Check(written == reference.size(), "Mesa input count exact");
          for (std::size_t i = 0; i < written; ++i)
            Check(actual[i].vertex_count == reference[i].vertex_count &&
                      std::equal(std::begin(actual[i].vertex_indices),
                                  std::end(actual[i].vertex_indices),
                                  std::begin(reference[i].vertex_indices)),
                  "Mesa input vertex adjacency/parity exact");
#endif
          if (written) {
            const auto before = actual;
            std::size_t unchanged = 456;
            Check(AssembleGeometryInputPrimitives(topology, source.data(), count,
                      restart_enabled, restart, 13, actual.data(), written - 1,
                      unchanged) == Status::kInsufficientCapacity,
                  "input primitive capacity checked before writes");
            Check(unchanged == 456 && !std::memcmp(before.data(), actual.data(),
                      sizeof(actual)), "input failure is transactional");
          }
        }
      }
    }
  }
  // Explicit adjacency corner cases: first/last edge neighbours and odd strip
  // parity. Raw occurrence values are deliberately not contiguous lane IDs.
  std::array<GeometryInputPrimitive, 36> result{};
  std::size_t n = 0;
  Check(AssembleGeometryInputPrimitives(Input::kTriangleStripAdjacency,
            indices.data(), 10, false, 0, 0, result.data(), result.size(), n) ==
            Status::kSuccess && n == 3, "three adjacency triangles");
  const unsigned expected[3][6] = {{0, 1, 2, 6, 4, 3},
                                  {4, 0, 2, 5, 6, 8},
                                  {4, 2, 6, 9, 8, 7}};
  for (unsigned i = 0; i < 3; ++i)
    for (unsigned j = 0; j < 6; ++j)
      Check(result[i].vertex_indices[j] == indices[expected[i][j]],
            "adjacency vertex-order specification");
  std::size_t unchanged = 9;
  Check(AssembleGeometryInputPrimitives(Input::kPoints, nullptr, 1, false, 0, 0,
            result.data(), result.size(), unchanged) == Status::kInvalidStorage,
        "null input span rejected");
  Check(AssembleGeometryInputPrimitives(Input::kPoints, indices.data(),
            std::size_t(UINT32_MAX) + 1, false, 0, 0, result.data(), result.size(),
            unchanged) == Status::kInvalidRange,
        "input count cannot overflow primitive identities");
  Check(unchanged == 9, "malformed input does not publish");
}
}  // namespace

int main() {
  try {
    TestEventTraces();
    TestSnapshotsAndBounds();
    TestInputAssembly();
    std::printf("geometry-emission-test: %llu checks PASS%s\n",
                static_cast<unsigned long long>(checks),
#if defined(PVRGPU_MESA_GEOMETRY_REFERENCE)
                " (pinned Mesa decomposition differential)"
#else
                ""
#endif
    );
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "geometry-emission-test: %s after %llu checks\n",
                  error.what(), static_cast<unsigned long long>(checks));
    return 1;
  }
}
