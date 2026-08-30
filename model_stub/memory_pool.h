#pragma once

#include "model_types.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pvrgpu::stub {

class MemoryPool {
 public:
  PoolHandle Allocate(std::size_t bytes);
  const std::vector<std::uint8_t>& Read(PoolHandle handle) const;
  std::vector<std::uint8_t>& Write(PoolHandle handle);
  void Release(PoolHandle handle);

  std::uint64_t bytes_in_flight() const { return bytes_in_flight_; }
  std::uint64_t high_water_bytes() const { return high_water_bytes_; }
  std::uint64_t allocations() const { return allocations_; }
  std::uint64_t releases() const { return releases_; }

 private:
  struct Entry {
    bool live = false;
    std::uint32_t generation = 0;
    std::uint32_t ref_count = 0;
    std::vector<std::uint8_t> bytes;
  };

  Entry& Checked(PoolHandle handle);
  const Entry& Checked(PoolHandle handle) const;

  std::vector<Entry> entries_;
  std::uint64_t bytes_in_flight_ = 0;
  std::uint64_t high_water_bytes_ = 0;
  std::uint64_t allocations_ = 0;
  std::uint64_t releases_ = 0;
};

}  // namespace pvrgpu::stub
