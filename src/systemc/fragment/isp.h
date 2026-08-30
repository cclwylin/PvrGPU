// Isp module：ISP = Image Synthesis Processor（影像合成處理器），HSR =
// Hidden Surface Removal。它依 32×32 tile 的 ordered primitive refs 執行
// fixed-point top-left coverage、depth/order owner resolution，保留每個
// candidate 的 identity；face-cull 產生空 parameter/ref stream 時仍完成清除
// 畫面。FIFO 只傳 MemoryPool handle，計時採 event-driven。
#pragma once

#include "common/pipeline_state.h"

#include <systemc>

namespace pvrgpu::stub {

class Isp final : public sc_core::sc_module {
public:
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};
  sc_core::sc_fifo_out<PipelineTxn> output{"output"};

  Isp(sc_core::sc_module_name name, MemoryPool &pool);

private:
  void Run();

  MemoryPool &pool_;
};

} // namespace pvrgpu::stub
