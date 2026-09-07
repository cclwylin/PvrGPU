// Stage-local USC LD adapter. Buffer bytes remain in the shared GPU address
// space; only immutable exact bound ranges and access counters live here.
#pragma once

#include "common/functional_types.h"
#include "memory/gpu_memory_system.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pvrgpu::stub {

class UscUniformBufferMemory final {
public:
  UscUniformBufferMemory(GpuMemorySystem *memory, MemoryMode mode,
                         std::vector<UniformBufferResource> resources)
      : memory_(memory), resources_(std::move(resources)) {
    if (!resources_.empty() && (!memory_ || memory_->mode() != mode))
      throw std::runtime_error("USC UBO memory service/mode mismatch");
    if (resources_.size() > kMaximumUniformBuffersPerStage)
      throw std::runtime_error("USC UBO resource count exceeds stage limit");
    for (std::size_t index = 0; index < resources_.size(); ++index) {
      const auto &range = resources_[index];
      if (!range.gpu_address || range.gpu_address % sizeof(std::uint32_t) ||
          !range.bytes || range.bytes > kMaximumUniformBufferBytes ||
          range.block_index >= kMaximumUniformBuffersPerStage ||
          range.bytes > std::numeric_limits<std::uint64_t>::max() -
                            range.gpu_address)
        throw std::runtime_error("USC UBO resource range is invalid");
      for (std::size_t other = 0; other < index; ++other) {
        const auto &prior = resources_[other];
        if (range.block_index == prior.block_index ||
            (range.gpu_address < prior.gpu_address + prior.bytes &&
             prior.gpu_address < range.gpu_address + range.bytes))
          throw std::runtime_error("USC UBO resource ranges overlap/duplicate");
      }
    }
  }

  // This callback/user pointer belongs only to the active USC stack frame.
  // Continuation payloads must never serialize either host pointer.
  static void Read(void *user_data, std::uint64_t address,
                   std::uint32_t dword_count, std::uint32_t *destination) {
    if (!user_data)
      throw std::runtime_error("USC UBO load has no stage memory context");
    static_cast<UscUniformBufferMemory *>(user_data)->ReadDwords(
        address, dword_count, destination);
  }

  const MemoryAccessStats &stats() const noexcept { return stats_; }

private:
  void ReadDwords(std::uint64_t address, std::uint32_t dword_count,
                  std::uint32_t *destination) {
    // Native LD_IMMBL encodes a positive wrapped 4-bit burst: 1..16 DWORDs.
    // Mesa may coalesce adjacent NIR vector loads into the larger bursts.
    if (!memory_ || !destination || dword_count == 0 || dword_count > 16 ||
        address % sizeof(std::uint32_t))
      throw std::runtime_error("USC UBO load service/destination/alignment/count is invalid");
    const std::size_t bytes = dword_count * sizeof(std::uint32_t);
    // Check the exact API-bound range, not DRAM's coarser page allocation.
    // A read cannot bridge adjacent buffers or reach another shader stage.
    const bool in_range = std::any_of(
        resources_.begin(), resources_.end(), [&](const auto &range) {
          if (address < range.gpu_address)
            return false;
          const std::uint64_t offset = address - range.gpu_address;
          return offset <= range.bytes && bytes <= range.bytes - offset;
        });
    if (!in_range)
      throw std::runtime_error("USC UBO load exceeds its stage bound buffer range");
    const MemoryReadResult read =
        memory_->Read(address, bytes, MemoryClient::kUniformBuffer);
    if (read.data.size() != bytes)
      throw std::runtime_error("USC UBO load returned an incomplete memory response");
    std::memcpy(destination, read.data.data(), bytes);
    stats_ += read.stats;
  }

  GpuMemorySystem *memory_;
  std::vector<UniformBufferResource> resources_;
  MemoryAccessStats stats_;
};

} // namespace pvrgpu::stub
