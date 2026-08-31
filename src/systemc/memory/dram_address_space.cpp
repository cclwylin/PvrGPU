#include "memory/dram_address_space.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace pvrgpu::stub {
namespace {

std::uint64_t CheckedEnd(std::uint64_t address, std::size_t bytes) {
  if (bytes == 0)
    throw std::invalid_argument("DRAM address-space range is empty");
  if (bytes > std::numeric_limits<std::uint64_t>::max() - address)
    throw std::overflow_error("DRAM address-space range wraps");
  return address + bytes;
}

}  // namespace

void DramAddressSpace::EnsureRange(std::uint64_t address, std::size_t bytes) {
  const std::uint64_t end = CheckedEnd(address, bytes);
  std::uint64_t page = address - address % kPageBytes;
  while (page < end) {
    pages_.try_emplace(page);
    if (kPageBytes > std::numeric_limits<std::uint64_t>::max() - page)
      throw std::overflow_error("DRAM address-space page advance wraps");
    page += kPageBytes;
  }
}

void DramAddressSpace::Write(std::uint64_t address,
                             const std::uint8_t *source,
                             std::size_t bytes) {
  if (!source && bytes != 0)
    throw std::invalid_argument("DRAM address-space write source is null");
  EnsureRange(address, bytes);
  std::size_t copied = 0;
  while (copied < bytes) {
    const std::uint64_t current = address + copied;
    const std::uint64_t page_address = current - current % kPageBytes;
    const std::size_t page_offset =
        static_cast<std::size_t>(current - page_address);
    const std::size_t chunk =
        std::min(bytes - copied, kPageBytes - page_offset);
    auto &page = pages_.at(page_address);
    std::copy_n(source + copied, chunk, page.begin() + page_offset);
    copied += chunk;
  }
}

std::vector<std::uint8_t> DramAddressSpace::Read(std::uint64_t address,
                                                 std::size_t bytes) const {
  (void)CheckedEnd(address, bytes);
  std::vector<std::uint8_t> result(bytes);
  std::size_t copied = 0;
  while (copied < bytes) {
    const std::uint64_t current = address + copied;
    const std::uint64_t page_address = current - current % kPageBytes;
    const auto page = pages_.find(page_address);
    if (page == pages_.end())
      throw std::runtime_error("DRAM read exceeds initialized backing");
    const std::size_t page_offset =
        static_cast<std::size_t>(current - page_address);
    const std::size_t chunk =
        std::min(bytes - copied, kPageBytes - page_offset);
    std::copy_n(page->second.begin() + page_offset, chunk,
                result.begin() + copied);
    copied += chunk;
  }
  return result;
}

bool DramAddressSpace::Contains(std::uint64_t address,
                                std::size_t bytes) const noexcept {
  if (bytes == 0 || bytes > std::numeric_limits<std::uint64_t>::max() - address)
    return false;
  const std::uint64_t end = address + bytes;
  std::uint64_t page = address - address % kPageBytes;
  while (page < end) {
    if (pages_.find(page) == pages_.end())
      return false;
    if (kPageBytes > std::numeric_limits<std::uint64_t>::max() - page)
      return false;
    page += kPageBytes;
  }
  return true;
}

}  // namespace pvrgpu::stub
