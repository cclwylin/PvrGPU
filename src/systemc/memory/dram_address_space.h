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

  void Write(std::uint64_t address, const std::uint8_t *source,
             std::size_t bytes);
  std::vector<std::uint8_t> Read(std::uint64_t address,
                                 std::size_t bytes) const;
  bool Contains(std::uint64_t address, std::size_t bytes) const noexcept;
  std::size_t resident_pages() const noexcept { return pages_.size(); }

 private:
  void EnsureRange(std::uint64_t address, std::size_t bytes);

  std::map<std::uint64_t, std::array<std::uint8_t, kPageBytes>> pages_;
};

}  // namespace pvrgpu::stub
