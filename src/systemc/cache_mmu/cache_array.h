/*
 * CacheArray is the reusable data/tag-array core shared by the PvrGPU cache
 * controllers.  It is deliberately not a SystemC module: an event-driven
 * controller owns timing and FIFO traffic, while this class implements a
 * bank-interleaved set-associative cache with write-back, write-allocate and
 * true least-recently-used (LRU) replacement.
 *
 * MCU = Mixed Cache Unit, TCU = Texture Cache Unit, SLC = System Level Cache,
 * and USC = Unified Shading Cluster.  The profile constants below encode the
 * current DXTP/reference-uArch capacities without claiming a commercial BVNC.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string_view>
#include <vector>

namespace pvrgpu::stub {

struct CacheArrayConfig {
  std::string_view name;
  std::size_t capacity_bytes;
  std::size_t line_size_bytes;
  std::size_t ways;
  std::size_t banks;
};

// DXTP/reference-uArch cache profiles.  A bank contains an equal number of
// sets, and consecutive cache lines are interleaved across banks.
inline constexpr CacheArrayConfig kMcuCacheConfig{"MCU", 24U * 1024U, 64U, 4U,
                                                  4U};
inline constexpr CacheArrayConfig kTcuCacheConfig{"TCU", 24U * 1024U, 64U, 4U,
                                                  4U};
inline constexpr CacheArrayConfig kSlcCacheConfig{"SLC", 2U * 1024U * 1024U,
                                                  128U, 8U, 8U};
inline constexpr CacheArrayConfig kUscL2CacheConfig{"USC-L2", 8U * 1024U, 64U,
                                                    4U, 1U};

constexpr CacheArrayConfig McuCacheConfig() { return kMcuCacheConfig; }
constexpr CacheArrayConfig TcuCacheConfig() { return kTcuCacheConfig; }
constexpr CacheArrayConfig SlcCacheConfig() { return kSlcCacheConfig; }
constexpr CacheArrayConfig UscL2CacheConfig() { return kUscL2CacheConfig; }

struct CacheStats {
  std::uint64_t line_accesses = 0;
  std::uint64_t read_accesses = 0;
  std::uint64_t write_accesses = 0;
  std::uint64_t hits = 0;
  std::uint64_t misses = 0;
  std::uint64_t evictions = 0;
  std::uint64_t writebacks = 0;
  std::uint64_t bypassed = 0;
};

CacheStats operator-(const CacheStats &after, const CacheStats &before);
CacheStats &operator+=(CacheStats &left, const CacheStats &right);

using CacheLineData = std::vector<std::uint8_t>;

// A lower read must return exactly requested_bytes.  A missing callback means
// deterministic zero-filled backing storage, useful for performance-only runs.
using CacheLineRead = std::function<CacheLineData(std::uint64_t line_address,
                                                  std::size_t requested_bytes)>;
using CacheLineWrite =
    std::function<void(std::uint64_t line_address, const CacheLineData &data)>;

struct CacheLineAccess {
  static constexpr std::size_t kNoCacheIndex =
      std::numeric_limits<std::size_t>::max();

  std::uint64_t line_address = 0;
  std::size_t bank = kNoCacheIndex;
  std::size_t set = kNoCacheIndex;
  std::size_t way = kNoCacheIndex;
  bool hit = false;
  bool bypassed = false;
  CacheStats delta;
  CacheLineData data;
};

class CacheArray {
public:
  explicit CacheArray(CacheArrayConfig config, bool bypass = false);

  const CacheArrayConfig &config() const noexcept { return config_; }
  const CacheStats &stats() const noexcept { return stats_; }
  bool bypass() const noexcept { return bypass_; }
  std::size_t line_count() const noexcept { return line_count_; }
  std::size_t sets_per_bank() const noexcept { return sets_per_bank_; }

  // Line accesses require a line-aligned address.  On a read miss, lower_read
  // fills the allocated line.  A full-line write miss allocates directly from
  // incoming data and therefore needs no read-for-ownership.  Dirty victims
  // and Flush() use lower_write.
  CacheLineAccess ReadLine(std::uint64_t line_address,
                           const CacheLineRead &lower_read = {},
                           const CacheLineWrite &lower_write = {});
  CacheLineAccess WriteLine(std::uint64_t line_address,
                            const CacheLineData &data,
                            const CacheLineRead &lower_read = {},
                            const CacheLineWrite &lower_write = {});

  // Touch every line intersecting [address, address + bytes).  It is intended
  // for performance traffic where the actual bytes are held elsewhere.  The
  // returned counters are the delta caused by this call, not lifetime totals.
  CacheStats AccessRange(std::uint64_t address, std::size_t bytes,
                         bool is_write);

  // Write back all dirty lines and leave clean lines resident.  The return
  // value is the number of dirty lines written by this call.
  std::uint64_t Flush(const CacheLineWrite &lower_write = {});

  // Switching bypass on flushes dirty lines and invalidates every line.  This
  // prevents stale cache contents when bypass is later switched off.  Returns
  // the number of dirty lines written during the transition.
  std::uint64_t SetBypass(bool bypass, const CacheLineWrite &lower_write = {});

  void ResetStats() noexcept { stats_ = {}; }

private:
  struct Line {
    bool valid = false;
    bool dirty = false;
    std::uint64_t tag = 0;
    std::size_t lru_rank = 0;
    CacheLineData data;
  };

  struct AddressFields {
    std::uint64_t line_address = 0;
    std::uint64_t tag = 0;
    std::size_t bank = 0;
    std::size_t set = 0;
  };

  CacheLineAccess AccessLine(std::uint64_t line_address, bool is_write,
                             const CacheLineData *write_data,
                             const CacheLineRead &lower_read,
                             const CacheLineWrite &lower_write,
                             bool return_data);
  AddressFields DecodeAddress(std::uint64_t line_address) const;
  std::uint64_t ReconstructAddress(std::uint64_t tag, std::size_t bank,
                                   std::size_t set) const;
  std::size_t FindWay(const AddressFields &fields) const;
  std::size_t ChooseVictim(const AddressFields &fields) const;
  void Touch(std::size_t set_index, std::size_t way);
  void InvalidateAll() noexcept;
  CacheLineData ReadLower(std::uint64_t line_address,
                          const CacheLineRead &lower_read) const;
  void WriteLower(std::uint64_t line_address, const CacheLineData &data,
                  const CacheLineWrite &lower_write) const;

  CacheArrayConfig config_;
  bool bypass_ = false;
  std::size_t line_count_ = 0;
  std::size_t sets_per_bank_ = 0;
  std::vector<std::vector<Line>> sets_;
  CacheStats stats_;
};

} // namespace pvrgpu::stub
