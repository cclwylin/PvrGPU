// PdsEngine：PDS = Programmable Data Sequencer（可程式化資料定序器）。
// 它執行 fragment coefficient-loading work：每個 primitive/2x2 quad task
// 依 ParameterBuffer range 把 A/B/C/PAD raw dwords 複製到 USC coefficient
// bank，並發布獨立 task descriptor。FIFO 只傳 PipelineState handle；bulk
// coefficient/task data 留在 MemoryPool，採 event-driven SC_THREAD。
#pragma once

#include "common/pipeline_state.h"
#include "memory/gpu_memory_system.h"

#include <systemc>

namespace pvrgpu::stub {

class PdsEngine final : public sc_core::sc_module {
 public:
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};
  sc_core::sc_fifo_out<PipelineTxn> output{"output"};

  PdsEngine(sc_core::sc_module_name name, MemoryPool &pool,
            GpuMemorySystem *memory = nullptr);

 private:
  void Run();

  MemoryPool &pool_;
  GpuMemorySystem *memory_;
};

}  // namespace pvrgpu::stub
