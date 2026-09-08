// Independent native tessellation stage; bounded FIFO transactions carry handles.
#pragma once
#include "common/pipeline_state.h"
#include <systemc>

namespace pvrgpu::stub {
class GpuMemorySystem;
class TessellationEvaluationShader final : public sc_core::sc_module {
 public:
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};
  sc_core::sc_fifo_out<PipelineTxn> output{"output"};
  TessellationEvaluationShader(sc_core::sc_module_name name, MemoryPool &pool,
      GpuMemorySystem *memory = nullptr);
 private:
  void Run();
  void Execute(PipelineState &state);
  MemoryPool &pool_;
  GpuMemorySystem *memory_;
};
} // namespace pvrgpu::stub
