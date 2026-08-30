// SLC (System Level Cache) is the shared last-level GPU cache between data
// masters and external DRAM (Dynamic Random-Access Memory). This event-driven
// controller models the selected 2 MiB, 128-byte-line, 8-bank, 8-way
// write-back/write-allocate true-LRU cache. FIFO transactions carry only
// MemoryPool handles. cache_bypass=on skips the tag/data lookup for faster
// functional simulation, but it never skips the downstream DRAM model.
#pragma once

#include "cache_mmu/cache_array.h"
#include "common/pipeline_state.h"

#include <systemc>

namespace pvrgpu::stub {

class Slc final : public sc_core::sc_module {
 public:
  sc_core::sc_fifo_in<MemoryTxn> input{"input"};
  sc_core::sc_fifo_out<MemoryTxn> output{"output"};
  sc_core::sc_port<sc_core::sc_fifo_in_if<MemoryTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      texture_input{"texture_input"};
  sc_core::sc_port<sc_core::sc_fifo_out_if<MemoryTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      texture_output{"texture_output"};
  sc_core::sc_port<sc_core::sc_fifo_out_if<MemoryTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      dram_request{"dram_request"};
  sc_core::sc_port<sc_core::sc_fifo_in_if<MemoryTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      dram_response{"dram_response"};

  Slc(sc_core::sc_module_name name, MemoryPool &pool, bool cache_bypass);

 private:
  void Run();
  void TextureRun();

  MemoryPool &pool_;
  bool cache_bypass_ = false;
  CacheArray cache_;
  // Both framebuffer and texture ports share one physical SLC array. The
  // mutex is held across a texture miss roundtrip so another SC_THREAD cannot
  // interleave tag/LRU/stat mutation while ReadLine waits on DRAM.
  sc_core::sc_mutex cache_mutex_;
};

}  // namespace pvrgpu::stub
