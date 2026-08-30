// USC L2（Unified Shading Cluster Level-2 cache）SystemC controller
// implementation. MemoryTxn FIFO arrival drives one timed completion event;
// no clock generator or cycle-polling process is used. CacheArray owns cache
// line/tag/LRU state and MemoryPool retains all bulk shader payload bytes.
#include "cache_mmu/usc_l2_cache.h"

#include "common/functional_types.h"

#include <limits>
#include <stdexcept>

namespace pvrgpu::stub {
namespace {

inline constexpr std::uint64_t kUscL2CacheAccessCycles = 1;

std::size_t ValidateTxn(const MemoryTxn &txn, const MemoryPool &pool) {
  if (txn.client != MemoryClient::kUscL2)
    throw std::invalid_argument(
        "UscL2Cache received a transaction for another cache client");
  if (txn.bytes == 0)
    throw std::invalid_argument("UscL2Cache received a zero-byte transaction");
  if (txn.bytes > std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("UscL2Cache transaction is too large");

  switch (txn.operation) {
  case MemoryOperation::kRead:
    break;
  case MemoryOperation::kWrite: {
    if (!HasPoolHandle(txn.payload))
      throw std::invalid_argument(
          "UscL2Cache write transaction has no MemoryPool payload");
    const auto &payload = pool.Read(txn.payload);
    if (txn.payload_format == MemoryPayloadFormat::kLinearBytes &&
        payload.size() < txn.bytes) {
      throw std::invalid_argument(
          "UscL2Cache linear write payload is shorter than transaction");
    }
    if (txn.payload_format == MemoryPayloadFormat::kCacheLineWrites &&
        (payload.empty() || payload.size() % sizeof(DramLineWrite) != 0)) {
      throw std::invalid_argument(
          "UscL2Cache cache-line payload is not a DramLineWrite array");
    }
    break;
  }
  default:
    throw std::invalid_argument("UscL2Cache received an invalid operation");
  }

  switch (txn.payload_format) {
  case MemoryPayloadFormat::kLinearBytes:
  case MemoryPayloadFormat::kCacheLineWrites:
    break;
  default:
    throw std::invalid_argument(
        "UscL2Cache received an invalid payload format");
  }
  return static_cast<std::size_t>(txn.bytes);
}

} // namespace

UscL2Cache::UscL2Cache(sc_core::sc_module_name name, MemoryPool &pool,
                       bool cache_bypass)
    : sc_module(name), pool_(pool), cache_(UscL2CacheConfig(), cache_bypass) {
  SC_THREAD(Run);
}

std::uint64_t UscL2Cache::SetCacheBypass(bool bypass) {
  return cache_.SetBypass(bypass);
}

void UscL2Cache::Run() {
  while (true) {
    const MemoryTxn txn = input.read();
    const std::size_t bytes = ValidateTxn(txn, pool_);
    last_delta_ = cache_.AccessRange(txn.address, bytes,
                                     txn.operation == MemoryOperation::kWrite);
    WaitForCycles(kUscL2CacheAccessCycles);
    output.write(txn);
  }
}

} // namespace pvrgpu::stub
