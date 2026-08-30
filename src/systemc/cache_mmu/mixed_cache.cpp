// MCU（Mixed Cache Unit）SystemC controller implementation. MemoryTxn input
// wakes one SC_THREAD transaction at a time; completion uses a timed event,
// never a clock edge or per-cycle polling loop. Cache data remains owned by
// CacheArray and bulk request bytes remain in MemoryPool.
#include "cache_mmu/mixed_cache.h"

#include "common/functional_types.h"

#include <limits>
#include <stdexcept>

namespace pvrgpu::stub {
namespace {

inline constexpr std::uint64_t kMixedCacheAccessCycles = 1;

std::size_t ValidateTxn(const MemoryTxn &txn, const MemoryPool &pool) {
  if (txn.client != MemoryClient::kMixedCache)
    throw std::invalid_argument(
        "MixedCache received a transaction for another cache client");
  if (txn.bytes == 0)
    throw std::invalid_argument("MixedCache received a zero-byte transaction");
  if (txn.bytes > std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("MixedCache transaction is too large");

  switch (txn.operation) {
  case MemoryOperation::kRead:
    break;
  case MemoryOperation::kWrite: {
    if (!HasPoolHandle(txn.payload))
      throw std::invalid_argument(
          "MixedCache write transaction has no MemoryPool payload");
    const auto &payload = pool.Read(txn.payload);
    if (txn.payload_format == MemoryPayloadFormat::kLinearBytes &&
        payload.size() < txn.bytes) {
      throw std::invalid_argument(
          "MixedCache linear write payload is shorter than transaction");
    }
    if (txn.payload_format == MemoryPayloadFormat::kCacheLineWrites &&
        (payload.empty() || payload.size() % sizeof(DramLineWrite) != 0)) {
      throw std::invalid_argument(
          "MixedCache cache-line payload is not a DramLineWrite array");
    }
    break;
  }
  default:
    throw std::invalid_argument("MixedCache received an invalid operation");
  }

  switch (txn.payload_format) {
  case MemoryPayloadFormat::kLinearBytes:
  case MemoryPayloadFormat::kCacheLineWrites:
    break;
  default:
    throw std::invalid_argument(
        "MixedCache received an invalid payload format");
  }
  return static_cast<std::size_t>(txn.bytes);
}

} // namespace

MixedCache::MixedCache(sc_core::sc_module_name name, MemoryPool &pool,
                       bool cache_bypass)
    : sc_module(name), pool_(pool), cache_(McuCacheConfig(), cache_bypass) {
  SC_THREAD(Run);
}

std::uint64_t MixedCache::SetCacheBypass(bool bypass) {
  return cache_.SetBypass(bypass);
}

void MixedCache::Run() {
  while (true) {
    const MemoryTxn txn = input.read();
    const std::size_t bytes = ValidateTxn(txn, pool_);
    last_delta_ = cache_.AccessRange(txn.address, bytes,
                                     txn.operation == MemoryOperation::kWrite);
    WaitForCycles(kMixedCacheAccessCycles);
    output.write(txn);
  }
}

} // namespace pvrgpu::stub
