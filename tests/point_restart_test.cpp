// SPDX-License-Identifier: MIT
#include "gallium/drivers/pvrgpu/pvrgpu_point_restart.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

static unsigned checks;
static void Check(bool condition, const char *reason) {
  ++checks;
  if (!condition) {
    std::fprintf(stderr, "point-restart-test: %s\n", reason);
    std::exit(1);
  }
}

template<class Index> void TestWidth() {
  constexpr Index sentinel = std::numeric_limits<Index>::max();
  for (const auto &source : std::vector<std::vector<Index>>{
      {}, {sentinel}, {sentinel, sentinel, sentinel}, {3, 2, sentinel, 1},
      {sentinel, 3, sentinel, 2, sentinel, 1, sentinel}, {0, 1, 2, 3},
      {sentinel, 0, sentinel, 0, sentinel}}) {
    const std::size_t bytes = source.size() * sizeof(Index);
    std::vector<uint8_t> buffer(bytes + 2 * sizeof(Index), 0xa5);
    if (bytes) std::memcpy(buffer.data() + sizeof(Index), source.data(), bytes);
    unsigned count = 123;
    uint32_t maximum = 456;
    Check(pvrgpu_compact_point_restart_indices(buffer.data() + sizeof(Index),
              bytes, sizeof(Index), source.size(), sentinel, &count, &maximum),
          "source-width restart compaction accepted");
    std::vector<Index> expected;
    uint32_t expected_maximum = 0;
    for (auto index : source) {
      if (index != sentinel) {
        expected.push_back(index);
        if (index > expected_maximum) expected_maximum = index;
      }
    }
    Check(count == expected.size() && maximum == expected_maximum,
          "sentinel excluded from effective count/max");
    if (count)
      Check(std::memcmp(buffer.data() + sizeof(Index), expected.data(),
                        count * sizeof(Index)) == 0,
            "retained index order and duplicates preserved");
    for (unsigned i = 0; i < sizeof(Index); ++i)
      Check(buffer[i] == 0xa5 && buffer[bytes + sizeof(Index) + i] == 0xa5,
            "owned snapshot compaction leaves both guards intact");
    // PrimitiveID is the compact occurrence number, not the source index
    // nor a counter reset by a marker. Instance offsets are applied after
    // filtering, and do not reinterpret shifted indices as restart markers.
    for (unsigned instance = 0; instance < 3; ++instance)
      for (unsigned primitive = 0; primitive < count; ++primitive) {
        Index index;
        std::memcpy(&index, buffer.data() + sizeof(Index) + primitive * sizeof(Index), sizeof(Index));
        const uint64_t shifted = uint64_t(index) + uint64_t(instance) * (maximum + 1);
        Check(shifted == uint64_t(expected[primitive]) + uint64_t(instance) * (expected_maximum + 1),
              "instance slice follows the retained source index");
      }
    Check(uint64_t(count) * 3 == expected.size() * 3,
          "three-instance point primitive count excludes all markers");
  }
  // A custom marker need not be the largest value. Its comparison is on
  // the raw unsigned source index, before a base-vertex offset of +1.
  std::array<Index, 5> custom{{1, 2, 3, 2, sentinel}};
  unsigned count = 0;
  uint32_t maximum = 0;
  Check(pvrgpu_compact_point_restart_indices(custom.data(), sizeof(custom),
            sizeof(Index), custom.size(), 2, &count, &maximum), "custom marker");
  Check(count == 3 && custom[0] == 1 && custom[1] == 3 && custom[2] == sentinel,
        "marker comparison happens before base vertex, preserving max index");
  Check(maximum == sentinel, "non-marker maximum remains visible for caller bounds checks");
  if (sizeof(Index) < 4) {
    std::array<Index, 1> largest{{sentinel}};
    Check(pvrgpu_compact_point_restart_indices(largest.data(), sizeof(largest),
              sizeof(Index), 1, UINT32_MAX, &count, &maximum), "out-of-width marker");
    Check(count == 1 && maximum == sentinel, "marker is not truncated to source width");
  }
}

int main() {
  TestWidth<uint8_t>();
  TestWidth<uint16_t>();
  TestWidth<uint32_t>();
  unsigned count = 7;
  uint32_t maximum = 9;
  std::array<uint8_t, 16> data{};
  const auto original = data;
  for (unsigned width : {0U, 3U, 8U})
    Check(!pvrgpu_compact_point_restart_indices(data.data(), data.size(), width,
              1, 0, &count, &maximum), "invalid index width rejected");
  Check(!pvrgpu_compact_point_restart_indices(data.data(), data.size(), 4,
            UINT32_MAX, 0, &count, &maximum), "count/capacity overflow rejected before reads");
  Check(!pvrgpu_compact_point_restart_indices(data.data(), 3, 4,
            1, 0, &count, &maximum), "partial source element rejected");
  Check(!pvrgpu_compact_point_restart_indices(nullptr, 4, 4,
            1, 0, &count, &maximum), "missing source rejected");
  Check(count == 7 && maximum == 9 && data == original,
        "rejected input leaves count/max and bytes untouched");
  Check(pvrgpu_compact_point_restart_indices(nullptr, 0, 4,
            0, 0, &count, &maximum) && count == 0 && maximum == 0,
        "empty draw requires no source access");
  std::printf("point-restart-test: %u checks PASS\n", checks);
}
