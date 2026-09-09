// Authoritative sparse GPU DRAM backing shared by the fast direct path and
// the modeled SLC/DRAM path.  Host initialization writes this address space
// without claiming GPU traffic.  GPU clients must use GpuMemorySystem rather
// than retaining pointers into these pages.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace pvrgpu::stub {

class DramAddressSpace final {
 public:
  static constexpr std::size_t kPageBytes = 128;

  DramAddressSpace() = default;
  DramAddressSpace(const DramAddressSpace &other);
  DramAddressSpace &operator=(const DramAddressSpace &other);
  DramAddressSpace(DramAddressSpace &&other) noexcept;
  DramAddressSpace &operator=(DramAddressSpace &&other) noexcept;

  void Write(std::uint64_t address, const std::uint8_t *source,
             std::size_t bytes);
  std::vector<std::uint8_t> Read(std::uint64_t address,
                                 std::size_t bytes) const;
  bool Contains(std::uint64_t address, std::size_t bytes) const noexcept;
  std::size_t resident_pages() const noexcept { return pages_.size(); }

 private:
  void EnsureRange(std::uint64_t address, std::size_t bytes);
  void ClearValidatedRanges() noexcept;

  struct ValidatedRange {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
  };
  static constexpr std::size_t kValidatedRangeCount = 64;

  std::map<std::uint64_t, std::array<std::uint8_t, kPageBytes>> pages_;

  // Positive proofs only: Contains first checks every page, then remembers the
  // proved page span. Writes only add pages, so changing bytes cannot invalidate
  // a proof. Replacing/moving the backing clears its proofs; any future page
  // removal/remapping operation must do so too. Eviction only costs a recheck.
  // Fixed storage keeps Contains allocation-free and noexcept. Access, including
  // const queries, is serialized by the model like other shared memory state.
  mutable std::array<ValidatedRange, kValidatedRangeCount> validated_ranges_{};
  mutable std::size_t most_recent_range_ = 0;
  mutable std::size_t next_validated_range_ = 0;
};

}  // namespace pvrgpu::stub
