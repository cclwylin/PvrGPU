// TextureCache 是 TCU（Texture Cache Unit，紋理快取單元）的 event-driven
// SystemC controller。TCU 位於 TPU（Texture Processing Unit，紋理處理單元）
// 與 SLC（System Level Cache，系統層級快取）間；其 24 KB、64-byte line、
// 4-bank、4-way array 採 write-back、write-allocate 與 true LRU。
// FIFO 只傳 MemoryTxn；bulk texture data 留在 MemoryPool。SampleRun 處理
// active TPU line reads：TCU miss 透過 SLC/DRAM 取回 64-byte line，填入
// cache 後把 response handle 回傳 TPU；Run 仍保留給通用/cache unit 測試。
#pragma once

#include "cache_mmu/cache_array.h"
#include "common/pipeline_state.h"

#include <systemc>

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
               bool cache_bypass = false);

  const CacheStats &stats() const noexcept { return cache_.stats(); }
  const CacheStats &last_delta() const noexcept { return last_delta_; }
  bool cache_bypass() const noexcept { return cache_.bypass(); }
  std::uint64_t SetCacheBypass(bool bypass);

private:
  void Run();
  void SampleRun();

  MemoryPool &pool_;
  CacheArray cache_;
  CacheStats last_delta_;
};

} // namespace pvrgpu::stub
