// UscCluster：USC = Unified Shading Cluster（統一著色叢集），是 PowerVR
// 的 programmable execution unit。本模組對每個 vertex/fragment lane 執行
// decoded PCO ISS，輸出 raw VTXOUT/PIXOUT register bits；不使用 shader enum
// 或固定顏色 branch，並把 PCO repeat 展開後乘上 invocation 數，記錄每
// DrawList 的 dynamic ALU/Tex/Memory instruction totals。FIFO 傳 handle，
// bulk lane/output 留在 MemoryPool。
#pragma once

#include "common/shader_stage.h"
#include "memory_pool.h"
#include "model_types.h"

#include <systemc>

namespace pvrgpu::stub {

// Validates the public shared-register layout used by a stage's combined
// image/sampler descriptor prefix and optional push-constant suffix.  Mesa
// may represent an empty suffix either as the legacy {start=0,count=0} pair or
// as the canonical empty range at the end of the descriptor prefix.
bool DriverPcoTextureSharedLayoutSupported(
    const DriverPcoStageAbi &abi, std::uint32_t descriptor_set_count);

class UscCluster final : public sc_core::sc_module {
public:
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};
  sc_core::sc_fifo_out<PipelineTxn> output{"output"};
  sc_core::sc_port<sc_core::sc_fifo_out_if<PipelineTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      texture_request_output{"texture_request_output"};
  sc_core::sc_port<sc_core::sc_fifo_in_if<PipelineTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      texture_response_input{"texture_response_input"};

  UscCluster(sc_core::sc_module_name name, MemoryPool &pool, ShaderStage stage);

private:
  void Run();

  MemoryPool &pool_;
  ShaderStage stage_;
};

} // namespace pvrgpu::stub
