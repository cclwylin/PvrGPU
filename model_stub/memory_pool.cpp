#include "memory_pool.h"

#include <algorithm>
#include <stdexcept>

namespace pvrgpu::stub {

PoolHandle MemoryPool::Allocate(std::size_t bytes) {
  std::size_t slot = 0;
  for (; slot < entries_.size(); ++slot) {
    if (!entries_[slot].live)
      break;
  }
  if (slot == entries_.size())
    entries_.push_back({});

  Entry& entry = entries_[slot];
  entry.live = true;
  entry.generation++;
  entry.ref_count = 1;
  entry.bytes.assign(bytes, 0);
  bytes_in_flight_ += bytes;
  high_water_bytes_ = std::max(high_water_bytes_, bytes_in_flight_);
  allocations_++;
  return {static_cast<std::uint32_t>(slot), entry.generation};
}

const std::vector<std::uint8_t>& MemoryPool::Read(PoolHandle handle) const {
  return Checked(handle).bytes;
}

std::vector<std::uint8_t>& MemoryPool::Write(PoolHandle handle) {
  return Checked(handle).bytes;
}

void MemoryPool::Release(PoolHandle handle) {
  Entry& entry = Checked(handle);
  if (--entry.ref_count != 0)
    return;
  bytes_in_flight_ -= entry.bytes.size();
  entry.bytes.clear();
  entry.live = false;
  releases_++;
}

MemoryPool::Entry& MemoryPool::Checked(PoolHandle handle) {
  if (handle.slot >= entries_.size())
    throw std::runtime_error("MemoryPool handle slot is out of range");
  Entry& entry = entries_[handle.slot];
  if (!entry.live || entry.generation != handle.generation)
    throw std::runtime_error("MemoryPool stale handle");
  return entry;
}

const MemoryPool::Entry& MemoryPool::Checked(PoolHandle handle) const {
  if (handle.slot >= entries_.size())
    throw std::runtime_error("MemoryPool handle slot is out of range");
  const Entry& entry = entries_[handle.slot];
  if (!entry.live || entry.generation != handle.generation)
    throw std::runtime_error("MemoryPool stale handle");
  return entry;
}

}  // namespace pvrgpu::stub
