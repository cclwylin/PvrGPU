// Unified GPU memory service.  All persistent GPU resources live in one
// DramAddressSpace.  Direct mode accesses that backing without modeled
// cache/DRAM time; bypass retains DRAM transactions; cache mode traverses the
// shared SLC data/tag array and writes dirty lines back to the same backing.
#pragma once

#include "cache_mmu/cache_array.h"
#include "memory/dram_address_space.h"
#include "model_types.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace pvrgpu::stub {

inline constexpr std::uint64_t kMemorySlcLookupCycles = 1;
inline constexpr std::uint64_t kMemoryDramRequestCycles = 1;

struct MemoryAccessStats {
  CacheStats slc;
  std::uint64_t slc_cycles = 0;
  std::uint64_t dram_read_transactions = 0;
  std::uint64_t dram_write_transactions = 0;
  std::uint64_t dram_read_bytes = 0;
  std::uint64_t dram_write_bytes = 0;
  std::uint64_t dram_cycles = 0;
  std::uint64_t direct_read_bytes = 0;
  std::uint64_t direct_write_bytes = 0;
};

MemoryAccessStats &operator+=(MemoryAccessStats &left,
                              const MemoryAccessStats &right);
void ApplyMemoryAccessStats(CounterTxn &counters,
                            const MemoryAccessStats &stats);
std::uint64_t MemoryAccessDelayCycles(const MemoryAccessStats &stats);

struct MemoryReadResult {
  std::vector<std::uint8_t> data;
  MemoryAccessStats stats;
};

class GpuMemorySystem final {
 public:
  explicit GpuMemorySystem(MemoryMode mode);

  MemoryMode mode() const noexcept { return mode_; }
  DramAddressSpace &backing() noexcept { return backing_; }
  const DramAddressSpace &backing() const noexcept { return backing_; }

  // CPU/loader initialization. This establishes DRAM contents but is outside
  // modeled GPU traffic and therefore returns no counters or latency.
  void HostWrite(std::uint64_t address, const void *source, std::size_t bytes);

  MemoryReadResult Read(std::uint64_t address, std::size_t bytes,
                        MemoryClient client);
  MemoryAccessStats Write(std::uint64_t address, const void *source,
                          std::size_t bytes, MemoryClient client);
  MemoryReadResult Readback(std::uint64_t address, std::size_t bytes,
                            MemoryClient client);
  MemoryAccessStats Flush();

 private:
  static void ValidateClient(MemoryClient client);
  static void AddDramRead(MemoryAccessStats &stats, std::size_t bytes);
  static void AddDramWrite(MemoryAccessStats &stats, std::size_t bytes);

  MemoryMode mode_;
  DramAddressSpace backing_;
  CacheArray slc_;
};

template <typename T>
void HostWriteArray(GpuMemorySystem &memory, std::uint64_t address,
                    const std::vector<T> &values) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (values.empty())
    return;
  if (values.size() > std::numeric_limits<std::size_t>::max() / sizeof(T))
    throw std::overflow_error("host GPU array byte size overflow");
  memory.HostWrite(address, values.data(), values.size() * sizeof(T));
}

template <typename T>
struct MemoryArrayReadResult {
  std::vector<T> values;
  MemoryAccessStats stats;
};

template <typename T>
MemoryArrayReadResult<T> ReadMemoryArray(GpuMemorySystem &memory,
                                         std::uint64_t address,
                                         std::uint64_t bytes,
                                         MemoryClient client) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (bytes == 0) {
    if (address != 0)
      throw std::runtime_error("empty GPU memory array has an address");
    return {};
  }
  if (address == 0 || bytes % sizeof(T) != 0 ||
      bytes > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("GPU memory array has an invalid byte size");
  }
  MemoryReadResult read =
      memory.Read(address, static_cast<std::size_t>(bytes), client);
  MemoryArrayReadResult<T> result;
  result.values.resize(read.data.size() / sizeof(T));
  std::memcpy(result.values.data(), read.data.data(), read.data.size());
  result.stats = read.stats;
  return result;
}

template <typename T>
MemoryAccessStats WriteMemoryArray(GpuMemorySystem &memory,
                                   std::uint64_t address,
                                   const std::vector<T> &values,
                                   MemoryClient client) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (values.empty())
    return {};
  if (address == 0 ||
      values.size() > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::runtime_error("GPU memory write array has an invalid size");
  }
  return memory.Write(address, values.data(), values.size() * sizeof(T),
                      client);
}

}  // namespace pvrgpu::stub
