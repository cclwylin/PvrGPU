#include "memory/gpu_memory_system.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace pvrgpu::stub {
namespace {

void AddChecked(std::uint64_t &left, std::uint64_t right,
                const char *field) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left)
    throw std::overflow_error(std::string("GPU memory counter overflow: ") +
                              field);
  left += right;
}

}  // namespace

MemoryAccessStats &operator+=(MemoryAccessStats &left,
                              const MemoryAccessStats &right) {
  left.slc += right.slc;
  AddChecked(left.slc_cycles, right.slc_cycles, "slc_cycles");
  AddChecked(left.dram_read_transactions, right.dram_read_transactions,
             "dram_read_transactions");
  AddChecked(left.dram_write_transactions, right.dram_write_transactions,
             "dram_write_transactions");
  AddChecked(left.dram_read_bytes, right.dram_read_bytes, "dram_read_bytes");
  AddChecked(left.dram_write_bytes, right.dram_write_bytes,
             "dram_write_bytes");
  AddChecked(left.dram_cycles, right.dram_cycles, "dram_cycles");
  AddChecked(left.direct_read_bytes, right.direct_read_bytes,
             "direct_read_bytes");
  AddChecked(left.direct_write_bytes, right.direct_write_bytes,
             "direct_write_bytes");
  return left;
}

void ApplyMemoryAccessStats(CounterTxn &counters,
                            const MemoryAccessStats &stats) {
  AddChecked(counters.slc_line_accesses, stats.slc.line_accesses,
             "slc_line_accesses");
  AddChecked(counters.slc_read_accesses, stats.slc.read_accesses,
             "slc_read_accesses");
  AddChecked(counters.slc_write_accesses, stats.slc.write_accesses,
             "slc_write_accesses");
  AddChecked(counters.slc_hits, stats.slc.hits, "slc_hits");
  AddChecked(counters.slc_misses, stats.slc.misses, "slc_misses");
  AddChecked(counters.slc_evictions, stats.slc.evictions, "slc_evictions");
  AddChecked(counters.slc_writebacks, stats.slc.writebacks,
             "slc_writebacks");
  AddChecked(counters.slc_bypassed, stats.slc.bypassed, "slc_bypassed");
  AddChecked(counters.slc_cycles, stats.slc_cycles, "slc_cycles");
  AddChecked(counters.dram_read_transactions, stats.dram_read_transactions,
             "dram_read_transactions");
  AddChecked(counters.dram_write_transactions, stats.dram_write_transactions,
             "dram_write_transactions");
  AddChecked(counters.dram_read_bytes, stats.dram_read_bytes,
             "dram_read_bytes");
  AddChecked(counters.dram_write_bytes, stats.dram_write_bytes,
             "dram_write_bytes");
  AddChecked(counters.dram_cycles, stats.dram_cycles, "dram_cycles");
  AddChecked(counters.memory_direct_read_bytes, stats.direct_read_bytes,
             "memory_direct_read_bytes");
  AddChecked(counters.memory_direct_write_bytes, stats.direct_write_bytes,
             "memory_direct_write_bytes");
}

std::uint64_t MemoryAccessDelayCycles(const MemoryAccessStats &stats) {
  if (stats.dram_cycles >
      std::numeric_limits<std::uint64_t>::max() - stats.slc_cycles) {
    throw std::overflow_error("GPU memory delay cycle overflow");
  }
  return stats.slc_cycles + stats.dram_cycles;
}

GpuMemorySystem::GpuMemorySystem(MemoryMode mode)
    : mode_(mode), slc_(SlcCacheConfig(), mode == MemoryMode::kBypass) {
  if (mode != MemoryMode::kDirect && mode != MemoryMode::kBypass &&
      mode != MemoryMode::kCache) {
    throw std::invalid_argument("GpuMemorySystem received invalid mode");
  }
}

void GpuMemorySystem::ValidateClient(MemoryClient client) {
  switch (client) {
  case MemoryClient::kFramebuffer:
  case MemoryClient::kTextureCache:
  case MemoryClient::kIndexFetch:
  case MemoryClient::kVertexFetch:
  case MemoryClient::kParameterWrite:
  case MemoryClient::kParameterRead:
  case MemoryClient::kFramebufferReadback:
    return;
  case MemoryClient::kMixedCache:
  case MemoryClient::kUscL2:
  case MemoryClient::kTextureUpload:
    break;
  }
  throw std::invalid_argument("unsupported unified GPU memory client");
}

void GpuMemorySystem::AddDramRead(MemoryAccessStats &stats,
                                  std::size_t bytes) {
  AddChecked(stats.dram_read_transactions, 1, "dram_read_transactions");
  AddChecked(stats.dram_read_bytes, bytes, "dram_read_bytes");
  AddChecked(stats.dram_cycles, kMemoryDramRequestCycles, "dram_cycles");
}

void GpuMemorySystem::AddDramWrite(MemoryAccessStats &stats,
                                   std::size_t bytes) {
  AddChecked(stats.dram_write_transactions, 1, "dram_write_transactions");
  AddChecked(stats.dram_write_bytes, bytes, "dram_write_bytes");
  AddChecked(stats.dram_cycles, kMemoryDramRequestCycles, "dram_cycles");
}

void GpuMemorySystem::HostWrite(std::uint64_t address, const void *source,
                                std::size_t bytes) {
  if (!source && bytes != 0)
    throw std::invalid_argument("GpuMemorySystem host source is null");
  backing_.Write(address, static_cast<const std::uint8_t *>(source), bytes);
}

MemoryReadResult GpuMemorySystem::Read(std::uint64_t address,
                                       std::size_t bytes,
                                       MemoryClient client) {
  ValidateClient(client);
  if (bytes == 0)
    throw std::invalid_argument("GpuMemorySystem read is empty");
  MemoryReadResult result;
  result.data.resize(bytes);
  if (mode_ == MemoryMode::kDirect) {
    result.data = backing_.Read(address, bytes);
    result.stats.direct_read_bytes = bytes;
    return result;
  }
  if (mode_ == MemoryMode::kBypass) {
    result.data = backing_.Read(address, bytes);
    result.stats.slc.bypassed = 1;
    AddDramRead(result.stats, bytes);
    return result;
  }

  const std::size_t line_bytes = slc_.config().line_size_bytes;
  std::size_t copied = 0;
  while (copied < bytes) {
    const std::uint64_t current = address + copied;
    const std::uint64_t line_address = current - current % line_bytes;
    const std::size_t line_offset =
        static_cast<std::size_t>(current - line_address);
    const std::size_t chunk = std::min(bytes - copied, line_bytes - line_offset);
    MemoryAccessStats lower;
    const CacheLineRead read_lower = [&](std::uint64_t lower_address,
                                         std::size_t lower_bytes) {
      AddDramRead(lower, lower_bytes);
      return backing_.Read(lower_address, lower_bytes);
    };
    const CacheLineWrite write_lower = [&](std::uint64_t lower_address,
                                           const CacheLineData &data) {
      backing_.Write(lower_address, data.data(), data.size());
      AddDramWrite(lower, data.size());
    };
    const CacheLineAccess access =
        slc_.ReadLine(line_address, read_lower, write_lower);
    result.stats.slc += access.delta;
    result.stats += lower;
    std::copy_n(access.data.begin() + line_offset, chunk,
                result.data.begin() + copied);
    copied += chunk;
  }
  result.stats.slc_cycles =
      result.stats.slc.line_accesses * kMemorySlcLookupCycles;
  return result;
}

MemoryAccessStats GpuMemorySystem::Write(std::uint64_t address,
                                         const void *source,
                                         std::size_t bytes,
                                         MemoryClient client) {
  ValidateClient(client);
  if (!source && bytes != 0)
    throw std::invalid_argument("GpuMemorySystem write source is null");
  if (bytes == 0)
    throw std::invalid_argument("GpuMemorySystem write is empty");
  const auto *input = static_cast<const std::uint8_t *>(source);
  MemoryAccessStats result;
  if (mode_ == MemoryMode::kDirect) {
    backing_.Write(address, input, bytes);
    result.direct_write_bytes = bytes;
    return result;
  }
  if (mode_ == MemoryMode::kBypass) {
    backing_.Write(address, input, bytes);
    result.slc.bypassed = 1;
    AddDramWrite(result, bytes);
    return result;
  }

  const std::size_t line_bytes = slc_.config().line_size_bytes;
  std::size_t copied = 0;
  while (copied < bytes) {
    const std::uint64_t current = address + copied;
    const std::uint64_t line_address = current - current % line_bytes;
    const std::size_t line_offset =
        static_cast<std::size_t>(current - line_address);
    const std::size_t chunk = std::min(bytes - copied, line_bytes - line_offset);
    MemoryAccessStats lower;
    const CacheLineRead read_lower = [&](std::uint64_t lower_address,
                                         std::size_t lower_bytes) {
      AddDramRead(lower, lower_bytes);
      return backing_.Read(lower_address, lower_bytes);
    };
    const CacheLineWrite write_lower = [&](std::uint64_t lower_address,
                                           const CacheLineData &data) {
      backing_.Write(lower_address, data.data(), data.size());
      AddDramWrite(lower, data.size());
    };

    CacheLineData line;
    if (line_offset == 0 && chunk == line_bytes) {
      line.assign(input + copied, input + copied + chunk);
    } else if (backing_.Contains(line_address, line_bytes)) {
      const CacheLineAccess read =
          slc_.ReadLine(line_address, read_lower, write_lower);
      result.slc += read.delta;
      result += lower;
      lower = {};
      line = read.data;
    } else {
      line.assign(line_bytes, 0);
    }
    std::copy_n(input + copied, chunk, line.begin() + line_offset);
    const CacheLineAccess write =
        slc_.WriteLine(line_address, line, read_lower, write_lower);
    result.slc += write.delta;
    result += lower;
    copied += chunk;
  }
  result.slc_cycles = result.slc.line_accesses * kMemorySlcLookupCycles;
  return result;
}

MemoryReadResult GpuMemorySystem::Readback(std::uint64_t address,
                                           std::size_t bytes,
                                           MemoryClient client) {
  ValidateClient(client);
  if (bytes == 0)
    throw std::invalid_argument("GpuMemorySystem readback is empty");

  MemoryReadResult result;
  result.stats += Flush();
  result.data = backing_.Read(address, bytes);
  if (mode_ == MemoryMode::kDirect) {
    result.stats.direct_read_bytes = bytes;
    return result;
  }
  if (mode_ == MemoryMode::kBypass)
    result.stats.slc.bypassed += 1;
  AddDramRead(result.stats, bytes);
  return result;
}

MemoryAccessStats GpuMemorySystem::Flush() {
  MemoryAccessStats result;
  if (mode_ != MemoryMode::kCache)
    return result;
  const CacheLineWrite write_lower = [&](std::uint64_t address,
                                         const CacheLineData &data) {
    backing_.Write(address, data.data(), data.size());
    AddDramWrite(result, data.size());
  };
  const CacheStats before = slc_.stats();
  (void)slc_.Flush(write_lower);
  result.slc = slc_.stats() - before;
  return result;
}

}  // namespace pvrgpu::stub
