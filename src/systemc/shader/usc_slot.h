// UscSlot：模擬 shader task 取得 USC execution slot 的 issue 階段。
// USC（Unified Shading Cluster，統一著色叢集）是 PowerVR 執行 vertex、fragment
// 與 compute shader 的可程式單元。目前支援的 solid-color raster cases 皆驗證
// vertex 4-lane issue group；有 fragment work 時則驗證 FragmentFrontend 產生的
// spatial 2×2 quad（含 partial quad），再把已 decode/可見的 task 推進至
// issued stage。模組以 FIFO（First-In, First-Out）傳遞 MemoryPool state
// handle，採單次事件延遲而非逐週期 clock。
#pragma once

#include "common/shader_stage.h"
#include "memory_pool.h"
#include "model_types.h"

#include <systemc>

namespace pvrgpu::stub {

class UscSlot final : public sc_core::sc_module {
public:
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};
  sc_core::sc_fifo_out<PipelineTxn> output{"output"};

  UscSlot(sc_core::sc_module_name name, MemoryPool &pool, ShaderStage stage);

private:
  void Run();

  MemoryPool &pool_;
  ShaderStage stage_;
};

} // namespace pvrgpu::stub
