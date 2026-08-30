// VertexPdsEngine：PDS（Programmable Data Sequencer）的頂點著色器分支。
// 依照 DXTP 架構圖，所有 Data Master（包含 VertexDM）的資料都先經由 PDS
// 調度後再送入 USC。本模組對應 VertexFetch → PDS → USC 頂點著色路徑。
// 目前頂點著色不需要 coefficient loading（係數載入屬 fragment-side），
// 因此本節點作為 pass-through 結構預留點，確保資料流拓樸與架構圖一致。
// 待 DXTP vertex PDS task descriptor 協議定義後實作完整功能。
#pragma once

#include "common/pipeline_state.h"

#include <systemc>

namespace pvrgpu::stub {

class VertexPdsEngine final : public sc_core::sc_module {
 public:
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};
  sc_core::sc_fifo_out<PipelineTxn> output{"output"};

  VertexPdsEngine(sc_core::sc_module_name name, MemoryPool &pool);

 private:
  void Run();

  MemoryPool &pool_;
};

}  // namespace pvrgpu::stub
