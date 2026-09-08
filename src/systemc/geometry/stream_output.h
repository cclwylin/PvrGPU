// Independent pre-clipping transform-feedback fixed-function stage.
#pragma once

#include "common/pipeline_state.h"
#include <systemc>

namespace pvrgpu::stub {

class GpuMemorySystem;

class StreamOutput final : public sc_core::sc_module {
 public:
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};
  sc_core::sc_fifo_out<PipelineTxn> output{"output"};
  StreamOutput(sc_core::sc_module_name name, MemoryPool &pool,
               GpuMemorySystem *memory = nullptr);

 private:
  void Run();
  void Execute(PipelineState &state);
  MemoryPool &pool_;
  GpuMemorySystem *memory_;
};

}  // namespace pvrgpu::stub
