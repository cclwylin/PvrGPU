#include "memory_pool.h"

#include <algorithm>
#include <stdexcept>

namespace pvrgpu::stub {

namespace {

// Keep the allocator fast for the small control payloads that are recycled on
// almost every pipeline stage, but do not pin old full-frame/candidate buffers
// in every free slot.  Manhattan changes the allocation shape between native
// sequences, so clear() alone can otherwise retain several gigabytes even
// when bytes_in_flight() has returned to zero.
constexpr std::size_t kMaximumRetainedPayloadCapacity = 64U * 1024U;

}  // namespace

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
  if (entry.bytes.capacity() > kMaximumRetainedPayloadCapacity) {
    std::vector<std::uint8_t>().swap(entry.bytes);
  } else {
    entry.bytes.clear();
  }
  entry.live = false;
  releases_++;
}

std::uint64_t MemoryPool::capacity_bytes() const {
  std::uint64_t total = 0;
  for (const Entry& entry : entries_)
    total += entry.bytes.capacity();
  return total;
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
