#include "memory_pool.h"

#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {

bool Check(bool condition, const char* message) {
  if (!condition)
    std::cerr << "FAIL: " << message << '\n';
  return condition;
}

}  // namespace

int main() {
  using pvrgpu::stub::MemoryPool;

  MemoryPool pool;
  constexpr std::size_t kLargePayloadBytes = 2U * 1024U * 1024U;
  const auto large = pool.Allocate(kLargePayloadBytes);
  if (!Check(pool.bytes_in_flight() == kLargePayloadBytes,
             "large payload is not live") ||
      !Check(pool.capacity_bytes() >= kLargePayloadBytes,
             "large payload capacity is not accounted")) {
    return 1;
  }

  pool.Release(large);
  if (!Check(pool.bytes_in_flight() == 0,
             "released large payload remains in flight") ||
      !Check(pool.capacity_bytes() == 0,
             "released large payload retained its backing storage")) {
    return 1;
  }

  constexpr std::size_t kSmallPayloadBytes = 4096U;
  const auto small = pool.Allocate(kSmallPayloadBytes);
  pool.Release(small);
  if (!Check(pool.bytes_in_flight() == 0,
             "released small payload remains in flight") ||
      !Check(pool.capacity_bytes() >= kSmallPayloadBytes,
             "small payload was not retained for reuse") ||
      !Check(pool.allocations() == 2 && pool.releases() == 2,
             "allocation accounting changed")) {
    return 1;
  }

  const auto upgraded = pool.Allocate(kLargePayloadBytes);
  pool.Release(upgraded);
  if (!Check(pool.capacity_bytes() == 0,
             "upgraded slot retained a large backing allocation")) {
    return 1;
  }

  std::cout << "PASS memory-pool large-payload retention\n";
  return 0;
}
