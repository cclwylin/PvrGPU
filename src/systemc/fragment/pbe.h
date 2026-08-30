// Pbe module：PBE = Pixel Back End（像素後端）。它驗證 USC PIXOUT0..3
// identity/order，做 F32→RGBA8 UNORM 格式轉換；opaque 路徑驗證每個 HSR
// owner 只寫一次，blend 路徑依 API order 做 destination read/modify/write。
// framebuffer 後續必須經 PbeWriteBack、SLC、DRAM readback；FIFO 只傳
// MemoryPool handle，fragment/framebuffer bulk data 留在 pool。
#pragma once

#include "common/pipeline_state.h"

#include <systemc>

namespace pvrgpu::stub {

class Pbe final : public sc_core::sc_module {
public:
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};
  sc_core::sc_fifo_out<PipelineTxn> output{"output"};

  Pbe(sc_core::sc_module_name name, MemoryPool &pool);

private:
  void Run();

  MemoryPool &pool_;
};

} // namespace pvrgpu::stub
