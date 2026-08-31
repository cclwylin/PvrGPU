// Module：VertexFetch。
// 縮寫：非縮寫（頂點擷取）。
// 功能：把 IEEE-754 vertex attribute 原始位元放入 PCO VTXIN register bank；
// float2→vec4 的 GLES z=0/w=1 default 只做 register materialization，不算
// VBO traffic。目前所有 indexed raster case 依 reference uArch 分段，使用
// direct-mapped post-transform cache 做真實 reuse；每個 index occurrence 都
// 產生 VertexLaneRef，cache miss 才新增 USC（Unified Shading Cluster）lane。
// resource/binding 與 lane bulk data 留在 MemoryPool；bounded FIFO
// （First-In, First-Out）只傳 handle，完成採 event-driven wait。PCO 是 Mesa
// 公開 PowerVR compiler backend 的名稱，本專案不推定其縮寫展開。
#pragma once

#include "common/pipeline_state.h"

#include <systemc>

namespace pvrgpu::stub {

class GpuMemorySystem;

class VertexFetch final : public sc_core::sc_module {
public:
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};
  sc_core::sc_fifo_out<PipelineTxn> output{"output"};

  VertexFetch(sc_core::sc_module_name name, MemoryPool &pool,
              GpuMemorySystem *memory = nullptr);

private:
  void Run();

  MemoryPool &pool_;
  GpuMemorySystem *memory_ = nullptr;
};

} // namespace pvrgpu::stub
