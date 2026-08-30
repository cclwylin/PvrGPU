// UscL2Cache 是 USC（Unified Shading Cluster，統一著色叢集）內部的 L2
// （Level 2，第二層）cache event-driven SystemC controller；它不同於 GPU
// clients 共用的 SLC（System Level Cache）。Reference profile 為 8 KB、
// 64-byte line、1 bank、4-way，採 write-back、write-allocate 與 true LRU。
// FIFO 只傳 MemoryTxn；bulk shader data 留在 MemoryPool。此版可接 idle/
// synthetic traffic，active USC load/store response protocol 留待後續接線。
#pragma once

#include "cache_mmu/cache_array.h"
#include "common/pipeline_state.h"

#include <systemc>

namespace pvrgpu::stub {

class UscL2Cache final : public sc_core::sc_module {
public:
  sc_core::sc_fifo_in<MemoryTxn> input{"input"};
  sc_core::sc_fifo_out<MemoryTxn> output{"output"};

  UscL2Cache(sc_core::sc_module_name name, MemoryPool &pool,
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
