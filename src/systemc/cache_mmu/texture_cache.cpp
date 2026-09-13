// TCU（Texture Cache Unit）SystemC controller implementation. FIFO arrival
// wakes the SC_THREAD; one timed completion event represents controller service
// without a clock-driven loop. CacheArray owns cache-line data/tag/LRU state
// and MemoryPool owns bulk texture payload bytes.
#include "cache_mmu/texture_cache.h"

#include "common/functional_types.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pvrgpu::stub {
namespace {

inline constexpr std::uint64_t kTextureCacheAccessCycles = 1;

void AddChecked(std::uint64_t &target, std::uint64_t amount,
                const char *field) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - target)
    throw std::overflow_error(std::string("TCU counter overflow: ") + field);
  target += amount;
}

std::size_t ValidateTxn(const MemoryTxn &txn, const MemoryPool &pool) {
  if (txn.client != MemoryClient::kTextureCache)
    throw std::invalid_argument(
        "TextureCache received a transaction for another cache client");
  if (txn.bytes == 0)
    throw std::invalid_argument(
        "TextureCache received a zero-byte transaction");
  if (txn.bytes > std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("TextureCache transaction is too large");

  switch (txn.operation) {
  case MemoryOperation::kRead:
    break;
  case MemoryOperation::kWrite: {
    if (!HasPoolHandle(txn.payload))
      throw std::invalid_argument(
          "TextureCache write transaction has no MemoryPool payload");
    const auto &payload = pool.Read(txn.payload);
    if (txn.payload_format == MemoryPayloadFormat::kLinearBytes &&
        payload.size() < txn.bytes) {
      throw std::invalid_argument(
          "TextureCache linear write payload is shorter than transaction");
    }
    if (txn.payload_format == MemoryPayloadFormat::kCacheLineWrites &&
        (payload.empty() || payload.size() % sizeof(DramLineWrite) != 0)) {
      throw std::invalid_argument(
          "TextureCache cache-line payload is not a DramLineWrite array");
    }
    break;
  }
  default:
    throw std::invalid_argument("TextureCache received an invalid operation");
  }

  switch (txn.payload_format) {
  case MemoryPayloadFormat::kLinearBytes:
  case MemoryPayloadFormat::kCacheLineWrites:
    break;
  default:
    throw std::invalid_argument(
        "TextureCache received an invalid payload format");
  }
  return static_cast<std::size_t>(txn.bytes);
}

} // namespace

TextureCache::TextureCache(sc_core::sc_module_name name, MemoryPool &pool,
                           bool cache_bypass, GpuMemorySystem *memory)
    : sc_module(name), pool_(pool), memory_(memory),
      cache_(TcuCacheConfig(), cache_bypass) {
  if (memory_) {
    memory_->SetTextureCacheInvalidator(
        [this](std::uint64_t address, std::size_t bytes) {
          InvalidateRange(address, bytes);
        });
  }
  SC_THREAD(Run);
  SC_THREAD(SampleRun);
}

TextureCache::~TextureCache() {
  if (memory_)
    memory_->SetTextureCacheInvalidator({});
}

std::uint64_t TextureCache::SetCacheBypass(bool bypass) {
  return cache_.SetBypass(bypass);
}

void TextureCache::InvalidateRange(std::uint64_t address, std::size_t bytes) {
  (void)cache_.InvalidateRange(address, bytes);
}

void TextureCache::Run() {
  while (true) {
    const MemoryTxn txn = input.read();
    const std::size_t bytes = ValidateTxn(txn, pool_);
    last_delta_ = cache_.AccessRange(txn.address, bytes,
                                     txn.operation == MemoryOperation::kWrite);
    WaitForCycles(kTextureCacheAccessCycles);
    output.write(txn);
  }
}

void TextureCache::SampleRun() {
  if (sample_input.size() == 0 || sample_output.size() == 0)
    return;
  if (!memory_ && (lower_request.size() == 0 || lower_response.size() == 0))
    return;
  const std::size_t line_bytes = cache_.config().line_size_bytes;
  while (true) {
    const MemoryTxn request = sample_input->read();
    if (memory_ && memory_->mode() != MemoryMode::kCache) {
      throw std::runtime_error(
          "TCU cached texture path requires cache memory mode");
    }
    const bool supported_bytes =
        request.bytes == 1 || request.bytes == 2 || request.bytes == 4 ||
        request.bytes == 8 || request.bytes == 16;
    if (request.client != MemoryClient::kTextureCache ||
        request.operation != MemoryOperation::kRead ||
        request.payload_format != MemoryPayloadFormat::kLinearBytes ||
        !supported_bytes || HasPoolHandle(request.payload)) {
      throw std::runtime_error("TCU received an invalid texel read request");
    }
    if (request.bytes - 1 >
        std::numeric_limits<std::uint64_t>::max() - request.address) {
      throw std::overflow_error("TCU texel read address overflow");
    }
    MemoryAccessStats lower_stats;
    const CacheLineRead lower_read = [&](std::uint64_t address,
                                         std::size_t bytes) {
      if (memory_) {
        CacheLineData data(bytes);
        lower_stats += memory_->ReadInto(address, data.data(), data.size(),
                                         MemoryClient::kTextureCache);
        return data;
      }
      MemoryTxn miss = request;
      miss.address = address;
      miss.bytes = bytes;
      lower_request->write(miss);
      const MemoryTxn response = lower_response->read();
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
        throw std::runtime_error("TCU received an invalid SLC fill response");
      }
      CacheLineData data =
          LoadArray<std::uint8_t>(pool_, response.payload);
      pool_.Release(response.payload);
      if (data.size() != bytes)
        throw std::runtime_error("TCU SLC fill byte count mismatch");
      return data;
    };

    std::vector<std::uint8_t> texel(static_cast<std::size_t>(request.bytes));
    CacheStats tcu_delta;
    std::size_t copied = 0;
    while (copied < texel.size()) {
      const std::uint64_t current = request.address + copied;
      const std::uint64_t line_address = current - current % line_bytes;
      const std::size_t line_offset =
          static_cast<std::size_t>(current - line_address);
      const std::size_t chunk =
          std::min(texel.size() - copied, line_bytes - line_offset);
      const CacheLineAccess access = cache_.ReadLineInto(
          line_address, line_offset, texel.data() + copied, chunk, lower_read,
          {});
      tcu_delta += access.delta;
      copied += chunk;
    }

    PipelineState state = LoadPipelineState(pool_, request.pipeline.state);
    AddChecked(state.counters.tcu_line_accesses, tcu_delta.line_accesses,
               "line_accesses");
    AddChecked(state.counters.tcu_read_accesses, tcu_delta.read_accesses,
               "read_accesses");
    AddChecked(state.counters.tcu_hits, tcu_delta.hits, "hits");
    AddChecked(state.counters.tcu_misses, tcu_delta.misses, "misses");
    AddChecked(state.counters.tcu_evictions, tcu_delta.evictions,
               "evictions");
    AddChecked(state.counters.tcu_writebacks, tcu_delta.writebacks,
               "writebacks");
    AddChecked(state.counters.tcu_bypassed, tcu_delta.bypassed,
               "bypassed");
    const std::uint64_t tcu_cycles =
        tcu_delta.line_accesses * kTextureCacheAccessCycles;
    AddChecked(state.counters.tcu_cycles, tcu_cycles, "cycles");
    last_delta_ = tcu_delta;

    std::uint64_t wait_cycles = tcu_cycles;
    if (memory_) {
      ApplyMemoryAccessStats(state.counters, lower_stats);
      AddChecked(wait_cycles, MemoryAccessDelayCycles(lower_stats),
                 "wait_cycles");
      AddChecked(state.counters.texture_cycles, wait_cycles,
                 "texture_cycles");
      switch (state.stage) {
      case PipelineStage::kFragmentTexturePending:
        AddChecked(state.counters.renderer_cycles, wait_cycles,
                   "renderer_cycles");
        break;
      case PipelineStage::kVertexTexturePending:
      case PipelineStage::kGeometryTexturePending:
      case PipelineStage::kTessellationControlTexturePending:
      case PipelineStage::kTessellationEvaluationTexturePending:
        AddChecked(state.counters.tiler_cycles, wait_cycles, "tiler_cycles");
        break;
      case PipelineStage::kComputeTexturePending:
        break;
      default:
        throw std::runtime_error("TCU received texture work at an invalid stage");
      }
    } else {
      // The legacy fully event-driven lower chain accounts its own latency in
      // renderer_cycles. Preserve that contract for existing standalone tests.
      AddChecked(state.counters.renderer_cycles, tcu_cycles,
                 "renderer_cycles");
    }
    StorePipelineState(pool_, request.pipeline.state, state);
    MemoryTxn response = request;
    response.payload = StoreNewArray(pool_, texel);
    WaitForCycles(wait_cycles);
    sample_output->write(response);
  }
}

} // namespace pvrgpu::stub
