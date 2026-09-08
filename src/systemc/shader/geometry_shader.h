// Independent native geometry stage. Bounded FIFOs carry PipelineTxn handles;
// native task state, exports and modeled memory belong to this stage.
#pragma once

#include "common/pipeline_state.h"
#include <systemc>

namespace pvrgpu::stub {

class GpuMemorySystem;

class GeometryShader final : public sc_core::sc_module {
 public:
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};
  sc_core::sc_fifo_out<PipelineTxn> output{"output"};
  sc_core::sc_port<sc_core::sc_fifo_out_if<PipelineTxn>, 0,
                  sc_core::SC_ZERO_OR_MORE_BOUND> texture_request_output{"texture_request_output"};
  sc_core::sc_port<sc_core::sc_fifo_in_if<PipelineTxn>, 0,
                  sc_core::SC_ZERO_OR_MORE_BOUND> texture_response_input{"texture_response_input"};
  GeometryShader(sc_core::sc_module_name name, MemoryPool &pool,
                  GpuMemorySystem *memory = nullptr);

 private:
  void Run();
  void Execute(PipelineState &state, const PipelineTxn &txn);
  void Sample(PipelineState &state, const PipelineTxn &txn,
              const PcoTextureRequest &request, std::uint32_t *response);
  MemoryPool &pool_;
  GpuMemorySystem *memory_;
};

}  // namespace pvrgpu::stub
