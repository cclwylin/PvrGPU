// MixedCache 是 MCU（Mixed Cache Unit，混合快取單元）的 event-driven
// SystemC controller。MCU 為 USC（Unified Shading Cluster，統一著色叢集）
// 的一般資料／程式碼 cache traffic envelope；其 24 KB、64-byte line、
// 4-bank、4-way data/tag array 採 write-back、write-allocate 與 true LRU。
// FIFO 只傳 MemoryTxn；bulk bytes 留在 MemoryPool。此版可處理 idle/synthetic
// traffic，但尚未宣稱具備 active workload 的 hit suppression/read response。
#pragma once

#include "cache_mmu/cache_array.h"
#include "common/pipeline_state.h"

#include <systemc>

namespace pvrgpu::stub {

class MixedCache final : public sc_core::sc_module {
public:
  sc_core::sc_fifo_in<MemoryTxn> input{"input"};
  sc_core::sc_fifo_out<MemoryTxn> output{"output"};

  MixedCache(sc_core::sc_module_name name, MemoryPool &pool,
             bool cache_bypass = false);

  const CacheStats &stats() const noexcept { return cache_.stats(); }
  const CacheStats &last_delta() const noexcept { return last_delta_; }
  bool cache_bypass() const noexcept { return cache_.bypass(); }
  std::uint64_t SetCacheBypass(bool bypass);

private:
  void Run();

  MemoryPool &pool_;
  CacheArray cache_;
  CacheStats last_delta_;
};

} // namespace pvrgpu::stub
