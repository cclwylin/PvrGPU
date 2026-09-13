// TextureCache 是 TCU（Texture Cache Unit，紋理快取單元）的 event-driven
// SystemC controller。TCU 位於 TPU（Texture Processing Unit，紋理處理單元）
// 與 SLC（System Level Cache，系統層級快取）間；其 24 KB、64-byte line、
// 4-bank、4-way array 採 write-back、write-allocate 與 true LRU。
// FIFO 只傳 MemoryTxn；bulk texture data 留在 MemoryPool。SampleRun 處理
// active TPU line reads：TCU miss 透過 SLC/DRAM 取回 64-byte line，填入
// cache 後把結果寫入 TPU 借出的 16-byte batch scratch；batch end 才一次
// 結算 PipelineState 與總延遲。Legacy request 仍回傳 TCU 配置的 response
// handle；Run 保留給通用/cache unit 測試。
#pragma once

#include "cache_mmu/cache_array.h"
#include "common/pipeline_state.h"
#include "memory/gpu_memory_system.h"

#include <systemc>

#include <array>

namespace pvrgpu::stub {

class TextureCache final : public sc_core::sc_module {
public:
  sc_core::sc_fifo_in<MemoryTxn> input{"input"};
  sc_core::sc_fifo_out<MemoryTxn> output{"output"};
  sc_core::sc_port<sc_core::sc_fifo_in_if<MemoryTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      sample_input{"sample_input"};
  sc_core::sc_port<sc_core::sc_fifo_out_if<MemoryTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      sample_output{"sample_output"};
  sc_core::sc_port<sc_core::sc_fifo_out_if<MemoryTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      lower_request{"lower_request"};
  sc_core::sc_port<sc_core::sc_fifo_in_if<MemoryTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      lower_response{"lower_response"};

  TextureCache(sc_core::sc_module_name name, MemoryPool &pool,
               bool cache_bypass = false,
               GpuMemorySystem *memory = nullptr);
  ~TextureCache() override;

  const CacheStats &stats() const noexcept { return cache_.stats(); }
  const CacheStats &last_delta() const noexcept { return last_delta_; }
  bool cache_bypass() const noexcept { return cache_.bypass(); }
  std::uint64_t SetCacheBypass(bool bypass);
  void InvalidateRange(std::uint64_t address, std::size_t bytes);

private:
  void Run();
  void SampleRun();

  MemoryPool &pool_;
  GpuMemorySystem *memory_;
  CacheArray cache_;
  CacheStats last_delta_;

  struct PendingSampleBatch {
    bool active = false;
    PipelineTxn pipeline;
    PoolHandle scratch;
    std::uint64_t member_count = 0;
    std::uint64_t service_cycles = 0;
    CacheStats tcu;
    MemoryAccessStats lower;
  };
  std::array<PendingSampleBatch, 6> pending_sample_batches_{};
  sc_core::sc_time sample_service_ready_{sc_core::SC_ZERO_TIME};
};

} // namespace pvrgpu::stub
