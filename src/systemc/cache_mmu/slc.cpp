// SLC (System Level Cache) controller implementation. PBE framebuffer stores
// arrive through PixelDataMaster as a linear MemoryPool payload. With cache
// modeling enabled, each complete 128-byte line traverses the real tag/data
// array and dirty lines are serialized as DramLineWrite records. With bypass
// enabled the original linear payload is forwarded. Both paths complete at
// the DRAM module; this module never publishes a framebuffer or PNG directly.
#include "cache_mmu/slc.h"

#include "common/functional_types.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pvrgpu::stub {
namespace {

inline constexpr std::uint64_t kSlcLookupLatency = 1;

void AddChecked(std::uint64_t &target, std::uint64_t amount,
                const char *field) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - target)
    throw std::overflow_error(std::string("SLC counter overflow: ") + field);
  target += amount;
}

}  // namespace

Slc::Slc(sc_core::sc_module_name name, MemoryPool &pool, bool cache_bypass)
    : sc_module(name), pool_(pool), cache_bypass_(cache_bypass),
      cache_(SlcCacheConfig(), cache_bypass) {
  SC_THREAD(Run);
  SC_THREAD(TextureRun);
}

void Slc::Run() {
  while (true) {
    const MemoryTxn txn = input.read();
    PipelineState state = LoadPipelineState(pool_, txn.pipeline.state);
    RequireStage(state.stage, PipelineStage::kPixelDataMasterComplete, name());
    if (txn.operation != MemoryOperation::kWrite ||
        txn.client != MemoryClient::kFramebuffer ||
        txn.payload_format != MemoryPayloadFormat::kLinearBytes ||
        !HasPoolHandle(txn.payload) ||
        txn.payload.slot != state.pbe_framebuffer.slot ||
        txn.payload.generation != state.pbe_framebuffer.generation) {
      throw std::runtime_error("SLC received an invalid framebuffer store");
    }
    if (state.cache_bypass != static_cast<std::uint8_t>(cache_bypass_))
      throw std::runtime_error("SLC cache_bypass configuration mismatch");
    if (txn.address != state.framebuffer_gpu_address ||
        txn.bytes != state.framebuffer_bytes || txn.bytes == 0 ||
        txn.bytes > std::numeric_limits<std::size_t>::max()) {
      throw std::runtime_error("SLC framebuffer address or size mismatch");
    }

    MemoryTxn downstream = txn;
    std::uint64_t framebuffer_slc_cycles = 0;
    PoolHandle pbe_to_release{};
    if (cache_bypass_) {
      // The bulk payload is deliberately retained until DramModel has copied
      // it into its own backing store. SLC still remains a FIFO pipeline hop.
      AddChecked(state.counters.slc_bypassed, 1, "slc_bypassed");
    } else {
      // Copy before allocating another MemoryPool entry: MemoryPool vector
      // growth may invalidate references returned by Read().
      const std::vector<std::uint8_t> source =
          LoadArray<std::uint8_t>(pool_, state.pbe_framebuffer);
      if (source.size() != static_cast<std::size_t>(txn.bytes))
        throw std::runtime_error("SLC source framebuffer size mismatch");

      const std::size_t line_bytes = cache_.config().line_size_bytes;
      if (line_bytes != kDramLineWriteBytes || txn.address % line_bytes != 0)
        throw std::logic_error("SLC/DRAM cache-line contract mismatch");
      const std::uint64_t line_count = CeilDivide(txn.bytes, line_bytes);
      if (line_count > std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("SLC line count is too large");

      std::vector<DramLineWrite> writebacks;
      writebacks.reserve(static_cast<std::size_t>(line_count));
      const CacheLineWrite collect_writeback =
          [&writebacks, line_bytes](std::uint64_t address,
                                    const CacheLineData &data) {
            if (data.size() != line_bytes)
              throw std::logic_error("SLC emitted a malformed cache line");
            DramLineWrite record;
            record.address = address;
            record.bytes = static_cast<std::uint32_t>(line_bytes);
            std::memcpy(record.data, data.data(), line_bytes);
            writebacks.push_back(record);
          };

      cache_mutex_.lock();
      const CacheStats before = cache_.stats();
      for (std::uint64_t line = 0; line < line_count; ++line) {
        if (line > (std::numeric_limits<std::uint64_t>::max() - txn.address) /
                       line_bytes) {
          throw std::overflow_error("SLC framebuffer line address overflow");
        }
        const std::uint64_t line_address = txn.address + line * line_bytes;
        const std::size_t source_offset =
            static_cast<std::size_t>(line * line_bytes);
        const std::size_t copy_bytes =
            std::min(line_bytes, source.size() - source_offset);
        CacheLineData line_data(line_bytes, 0);
        std::copy_n(source.begin() + source_offset, copy_bytes,
                    line_data.begin());
        (void)cache_.WriteLine(line_address, line_data, {}, collect_writeback);
      }
      (void)cache_.Flush(collect_writeback);
      const CacheStats delta = cache_.stats() - before;
      if (delta.line_accesses != line_count ||
          delta.write_accesses != line_count ||
          delta.hits + delta.misses != delta.line_accesses ||
          delta.writebacks != writebacks.size()) {
        throw std::logic_error("SLC cache statistics/writeback mismatch");
      }
      cache_mutex_.unlock();

      AddChecked(state.counters.slc_line_accesses, delta.line_accesses,
                 "slc_line_accesses");
      AddChecked(state.counters.slc_read_accesses, delta.read_accesses,
                 "slc_read_accesses");
      AddChecked(state.counters.slc_write_accesses, delta.write_accesses,
                 "slc_write_accesses");
      AddChecked(state.counters.slc_hits, delta.hits, "slc_hits");
      AddChecked(state.counters.slc_misses, delta.misses, "slc_misses");
      AddChecked(state.counters.slc_evictions, delta.evictions,
                 "slc_evictions");
      AddChecked(state.counters.slc_writebacks, delta.writebacks,
                 "slc_writebacks");
      AddChecked(state.counters.slc_bypassed, delta.bypassed,
                 "slc_bypassed");
      AddChecked(state.counters.slc_cycles,
                 delta.line_accesses * kSlcLookupLatency, "slc_cycles");
      framebuffer_slc_cycles = delta.line_accesses * kSlcLookupLatency;
      state.slc_writeback_lines = StoreNewArray(pool_, writebacks);
      pbe_to_release = state.pbe_framebuffer;
      state.pbe_framebuffer = {};

      downstream.payload = state.slc_writeback_lines;
      downstream.payload_format = MemoryPayloadFormat::kCacheLineWrites;
    }

    AddChecked(state.counters.renderer_cycles, framebuffer_slc_cycles,
               "renderer_cycles");
    state.stage = PipelineStage::kSlcComplete;
    WaitForCycles(framebuffer_slc_cycles);
    StorePipelineState(pool_, txn.pipeline.state, state);
    if (HasPoolHandle(pbe_to_release))
      pool_.Release(pbe_to_release);
    output.write(downstream);
  }
}

void Slc::TextureRun() {
  if (texture_input.size() == 0 || texture_output.size() == 0 ||
      dram_request.size() == 0 || dram_response.size() == 0)
    return;
  const std::size_t line_bytes = cache_.config().line_size_bytes;
  while (true) {
    const MemoryTxn request = texture_input->read();
    if (request.client != MemoryClient::kTextureCache ||
        request.operation != MemoryOperation::kRead ||
        request.payload_format != MemoryPayloadFormat::kLinearBytes ||
        request.bytes == 0 || request.bytes > line_bytes ||
        HasPoolHandle(request.payload)) {
      throw std::runtime_error("SLC received an invalid texture-line read");
    }
    const std::uint64_t line_address =
        request.address - request.address % line_bytes;
    const CacheLineRead lower_read = [&](std::uint64_t address,
                                         std::size_t bytes) {
      MemoryTxn miss = request;
      miss.address = address;
      miss.bytes = bytes;
      dram_request->write(miss);
      const MemoryTxn response = dram_response->read();
      if (response.pipeline.frame != request.pipeline.frame ||
          response.pipeline.sequence != request.pipeline.sequence ||
          response.pipeline.state.slot != request.pipeline.state.slot ||
          response.pipeline.state.generation !=
              request.pipeline.state.generation ||
          response.request_id != request.request_id ||
          response.address != address || response.bytes != bytes ||
          response.client != MemoryClient::kTextureCache ||
          response.operation != MemoryOperation::kRead ||
          response.payload_format != MemoryPayloadFormat::kLinearBytes ||
          !HasPoolHandle(response.payload)) {
        throw std::runtime_error("SLC received an invalid DRAM fill response");
      }
      CacheLineData data =
          LoadArray<std::uint8_t>(pool_, response.payload);
      pool_.Release(response.payload);
      if (data.size() != bytes)
        throw std::runtime_error("SLC DRAM fill byte count mismatch");
      return data;
    };
    cache_mutex_.lock();
    const CacheLineAccess access =
        cache_.ReadLine(line_address, lower_read, {});
    cache_mutex_.unlock();
    if (access.data.size() != line_bytes ||
        request.address < line_address ||
        request.address - line_address > line_bytes - request.bytes) {
      throw std::runtime_error("SLC texture-line extraction is invalid");
    }
    const std::size_t offset =
        static_cast<std::size_t>(request.address - line_address);
    const std::size_t bytes = static_cast<std::size_t>(request.bytes);
    std::vector<std::uint8_t> result(access.data.begin() + offset,
                                     access.data.begin() + offset + bytes);
    PipelineState state = LoadPipelineState(pool_, request.pipeline.state);
    AddChecked(state.counters.slc_line_accesses, access.delta.line_accesses,
               "texture slc_line_accesses");
    AddChecked(state.counters.slc_read_accesses, access.delta.read_accesses,
               "texture slc_read_accesses");
    AddChecked(state.counters.slc_hits, access.delta.hits,
               "texture slc_hits");
    AddChecked(state.counters.slc_misses, access.delta.misses,
               "texture slc_misses");
    AddChecked(state.counters.slc_evictions, access.delta.evictions,
               "texture slc_evictions");
    AddChecked(state.counters.slc_writebacks, access.delta.writebacks,
               "texture slc_writebacks");
    AddChecked(state.counters.slc_bypassed, access.delta.bypassed,
               "texture slc_bypassed");
    AddChecked(state.counters.slc_cycles,
               access.delta.line_accesses * kSlcLookupLatency,
               "texture slc_cycles");
    AddChecked(state.counters.renderer_cycles,
               access.delta.line_accesses * kSlcLookupLatency,
               "texture renderer_cycles");
    StorePipelineState(pool_, request.pipeline.state, state);
    MemoryTxn response = request;
    response.payload = StoreNewArray(pool_, result);
    WaitForCycles(access.delta.line_accesses * kSlcLookupLatency);
    texture_output->write(response);
  }
}

}  // namespace pvrgpu::stub
