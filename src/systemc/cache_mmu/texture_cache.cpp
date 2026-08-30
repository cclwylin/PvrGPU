// TCU（Texture Cache Unit）SystemC controller implementation. FIFO arrival
// wakes the SC_THREAD; one timed completion event represents controller service
// without a clock-driven loop. CacheArray owns cache-line data/tag/LRU state
// and MemoryPool owns bulk texture payload bytes.
#include "cache_mmu/texture_cache.h"

#include "common/functional_types.h"

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
                           bool cache_bypass)
    : sc_module(name), pool_(pool), cache_(TcuCacheConfig(), cache_bypass) {
  SC_THREAD(Run);
  SC_THREAD(SampleRun);
}

std::uint64_t TextureCache::SetCacheBypass(bool bypass) {
  return cache_.SetBypass(bypass);
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
  if (sample_input.size() == 0 || sample_output.size() == 0 ||
      lower_request.size() == 0 || lower_response.size() == 0)
    return;
  constexpr std::size_t kTexelBytes = 4;
  const std::size_t line_bytes = cache_.config().line_size_bytes;
  while (true) {
    const MemoryTxn request = sample_input->read();
    if (request.client != MemoryClient::kTextureCache ||
        request.operation != MemoryOperation::kRead ||
        request.payload_format != MemoryPayloadFormat::kLinearBytes ||
        request.bytes != kTexelBytes || HasPoolHandle(request.payload)) {
      throw std::runtime_error("TCU received an invalid texel read request");
    }
    const std::uint64_t line_address =
        request.address - request.address % line_bytes;
    const CacheLineRead lower_read = [&](std::uint64_t address,
                                         std::size_t bytes) {
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
    const CacheLineAccess access =
        cache_.ReadLine(line_address, lower_read, {});
    if (access.data.size() != line_bytes ||
        request.address < line_address ||
        request.address - line_address > line_bytes - kTexelBytes) {
      throw std::runtime_error("TCU cache-line extraction is invalid");
    }
    const std::size_t offset =
        static_cast<std::size_t>(request.address - line_address);
    std::vector<std::uint8_t> texel(
        access.data.begin() + offset,
        access.data.begin() + offset + kTexelBytes);
    MemoryTxn response = request;
    response.payload = StoreNewArray(pool_, texel);
    PipelineState state = LoadPipelineState(pool_, request.pipeline.state);
    AddChecked(state.counters.tcu_line_accesses, access.delta.line_accesses,
               "line_accesses");
    AddChecked(state.counters.tcu_read_accesses, access.delta.read_accesses,
               "read_accesses");
    AddChecked(state.counters.tcu_hits, access.delta.hits, "hits");
    AddChecked(state.counters.tcu_misses, access.delta.misses, "misses");
    AddChecked(state.counters.tcu_evictions, access.delta.evictions,
               "evictions");
    AddChecked(state.counters.tcu_writebacks, access.delta.writebacks,
               "writebacks");
    AddChecked(state.counters.tcu_bypassed, access.delta.bypassed,
               "bypassed");
    AddChecked(state.counters.tcu_cycles, kTextureCacheAccessCycles,
               "cycles");
    AddChecked(state.counters.renderer_cycles, kTextureCacheAccessCycles,
               "renderer_cycles");
    StorePipelineState(pool_, request.pipeline.state, state);
    WaitForCycles(kTextureCacheAccessCycles);
    sample_output->write(response);
  }
}

} // namespace pvrgpu::stub
