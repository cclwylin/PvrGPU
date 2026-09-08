// SPDX-License-Identifier: MIT
#include "gallium/drivers/pvrgpu/pvrgpu_index_fetch.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static unsigned checks;
static void Check(bool condition, const char *message) {
  ++checks;
  if (!condition) {
    std::fprintf(stderr, "index-fetch-test: %s\n", message);
    std::exit(1);
  }
}

int main() {
  for (unsigned width : {1U, 2U, 4U}) {
    std::array<uint8_t, 40> bytes;
    bytes.fill(0xcd);
    for (unsigned i = 0; i < 8; ++i) {
      const uint32_t value = 3 + i * 7;
      std::memcpy(bytes.data() + 1 + i * width, &value, width);
    }
    const auto original = bytes;
    // Deliberately unaligned backing and every complete/partial byte length.
    for (size_t size = 0; size <= 8 * width; ++size) {
      for (uint64_t position = 0; position < 11; ++position) {
        uint32_t result = 0xdeadbeef;
        Check(pvrgpu_fetch_bounded_index(bytes.data() + 1, size, width,
                                         position, &result), "valid bounded fetch");
        Check(result == (position < size / width ? 3 + position * 7 : 0),
              "only complete in-range elements read, out-of-range returns zero");
      }
      for (uint64_t position : {uint64_t(UINT32_MAX), uint64_t(UINT32_MAX) + 17,
                                 UINT64_MAX}) {
        uint32_t result = 123;
        Check(pvrgpu_fetch_bounded_index(bytes.data() + 1, size, width,
                                         position, &result) && result == 0,
              "large positions cannot wrap into the allocation");
      }
    }
    Check(bytes == original, "source and guards unchanged");
  }
  const uint32_t data = 7;
  for (unsigned width : {0U, 3U, 8U, UINT32_MAX}) {
    uint32_t result = 123;
    Check(!pvrgpu_fetch_bounded_index(&data, sizeof(data), width, 0, &result),
          "invalid index width rejected");
    Check(result == 123, "rejection does not publish output");
  }
  uint32_t result = 123;
  Check(!pvrgpu_fetch_bounded_index(nullptr, 0, 1, 0, &result) && result == 123,
        "missing backing is not an out-of-range success");
  Check(!pvrgpu_fetch_bounded_index(&data, sizeof(data), 4, 0, nullptr),
        "missing output rejected");
  std::printf("index-fetch-test: %u checks PASS\n", checks);
}
