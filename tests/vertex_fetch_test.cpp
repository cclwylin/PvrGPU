// SPDX-License-Identifier: MIT
#include "gallium/drivers/pvrgpu/pvrgpu_vertex_fetch.h"
#include "gallium/drivers/pvrgpu/pvrgpu_point_restart.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static unsigned checks;
static void Check(bool value, const char *message) {
  ++checks;
  if (!value) {
    std::fprintf(stderr, "vertex-fetch-test: %s\n", message);
    std::exit(1);
  }
}

static void Store(uint8_t *bytes, unsigned width, uint32_t value) {
  if (width == 1) *bytes = static_cast<uint8_t>(value);
  else if (width == 2) {
    const auto narrow = static_cast<uint16_t>(value);
    std::memcpy(bytes, &narrow, sizeof(narrow));
  } else std::memcpy(bytes, &value, sizeof(value));
}

static void Exercise(unsigned width, std::vector<uint32_t> input, int32_t bias) {
  std::vector<uint8_t> bytes(input.size() * width + 2, 0xa7);
  for (std::size_t i = 0; i < input.size(); ++i)
    Store(bytes.data() + 1 + i * width, width, input[i]);
  const auto before = bytes;
  uint32_t minimum = UINT32_MAX, maximum = 0;
  for (auto index : input) {
    if (index < minimum) minimum = index;
    if (index > maximum) maximum = index;
  }
  const int64_t first = static_cast<int64_t>(minimum) + bias;
  const int64_t last = static_cast<int64_t>(maximum) + bias;
  const uint64_t extent = static_cast<uint64_t>(maximum) - minimum + 1;
  const bool valid = first >= 0 && last <= UINT32_MAX && extent <= UINT32_MAX;
  uint32_t source = 0x12345678, count = 0x87654321;
  Check(pvrgpu_rebase_vertex_indices(bytes.data() + 1, bytes.size() - 2,
        width, static_cast<uint32_t>(input.size()), bias, &source, &count) == valid,
        "signed range and full uint32 span validity");
  if (!valid) {
    Check(bytes == before && source == 0x12345678 && count == 0x87654321,
          "rejection leaves snapshot, guards and outputs untouched");
    return;
  }
  Check(source == first && count == extent, "only referenced min/max range is packed");
  Check(bytes.front() == 0xa7 && bytes.back() == 0xa7, "unaligned snapshot guards");
  for (std::size_t i = 0; i < input.size(); ++i) {
    const auto rebased = pvrgpu_snapshot_index(bytes.data() + 1 + i * width, width);
    Check(rebased < count, "every index remains within the packed range");
    Check(static_cast<int64_t>(rebased) + source == static_cast<int64_t>(input[i]) + bias,
          "index + source first exactly preserves signed baseVertex fetch");
    for (uint32_t instance : {0U, 1U, 11U}) {
      const uint64_t expanded = static_cast<uint64_t>(instance) * count + rebased;
      Check(expanded / count == instance && expanded % count == rebased,
            "instancing preserves draw-local index reuse and range");
    }
  }
}

int main() {
  for (unsigned width : {1U, 2U, 4U}) {
    for (int32_t bias : {INT32_MIN, -17, -2, -1, 0, 1, 2, 17, INT32_MAX}) {
      Exercise(width, {3, 7, 3, 4, 7, 5}, bias);
      Exercise(width, {0, 1, 0}, bias);
      Exercise(width, {17, 17, 17}, bias);
    }
    const uint32_t maximum = width == 1 ? UINT8_MAX : width == 2 ? UINT16_MAX : UINT32_MAX;
    Exercise(width, {maximum, maximum - 1, maximum}, -2);
    Exercise(width, {maximum, maximum}, 0);
    Exercise(width, {maximum, maximum}, 1);
    Exercise(width, {0, maximum}, 0);

    // Restart is filtered in original unsigned index space, before rebasing.
    std::array<uint8_t, 24> restart{};
    for (unsigned i = 0; i < 6; ++i)
      Store(restart.data() + i * width, width, i % 2 ? 4 : maximum);
    unsigned retained = 0;
    uint32_t max_index = 0, source = 0, count = 0;
    Check(pvrgpu_compact_point_restart_indices(restart.data(), restart.size(), width,
            6, maximum, &retained, &max_index) && retained == 3 && max_index == 4,
          "restart compaction removes raw markers before negative baseVertex");
    Check(pvrgpu_rebase_vertex_indices(restart.data(), restart.size(), width,
            retained, -2, &source, &count) && source == 2 && count == 1,
          "negative baseVertex after restart retains only actual vertices");
    for (unsigned i = 0; i < retained; ++i)
      Check(pvrgpu_snapshot_index(restart.data() + i * width, width) == 0,
            "all references preserve same vertex reuse");
  }
  Exercise(4, {0x80000000, 0x80000002, 0x80000001}, INT32_MIN);
  Exercise(4, {1000000000, 1000000007, 1000000000}, 1);
  std::array<uint8_t, 20> bytes{};
  uint32_t source = 123, count = 456;
  for (unsigned width : {0U, 3U, 8U, UINT32_MAX})
    Check(!pvrgpu_rebase_vertex_indices(bytes.data(), bytes.size(), width, 3,
            0, &source, &count), "invalid index widths rejected");
  Check(!pvrgpu_rebase_vertex_indices(nullptr, 20, 4, 3, 0, &source, &count), "null snapshot");
  Check(!pvrgpu_rebase_vertex_indices(bytes.data(), 11, 4, 3, 0, &source, &count), "truncated snapshot");
  Check(!pvrgpu_rebase_vertex_indices(bytes.data(), 20, 4, 0, 0, &source, &count), "empty work handled before rebase");
  Check(!pvrgpu_rebase_vertex_indices(bytes.data(), 20, 4, 3, 0, nullptr, &count), "null first output");
  Check(!pvrgpu_rebase_vertex_indices(bytes.data(), 20, 4, 3, 0, &source, nullptr), "null count output");
  Check(!pvrgpu_rebase_vertex_indices(bytes.data(), 20, 4, UINT32_MAX, 0, &source, &count), "huge count bounded before scan");
  Check(source == 123 && count == 456, "shape rejections preserve outputs");

  uint32_t random = 0x871add3b;
  for (unsigned iteration = 0; iteration < 12000; ++iteration) {
    random = random * 1664525U + 1013904223U;
    const unsigned width = std::array<unsigned, 3>{1, 2, 4}[iteration % 3];
    const uint32_t mask = width == 1 ? UINT8_MAX : width == 2 ? UINT16_MAX : UINT32_MAX;
    const uint32_t minimum = random & mask;
    const uint32_t span = (random >> 20) % 17;
    const uint32_t maximum = minimum > mask - span ? mask : minimum + span;
    const int32_t bias = static_cast<int32_t>((random >> 5) % 65) - 32;
    Exercise(width, {maximum, minimum, maximum, minimum + (maximum - minimum) / 2}, bias);
  }
  std::printf("vertex-fetch-test: %u checks PASS\n", checks);
}
