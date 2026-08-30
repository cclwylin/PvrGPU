// PixelDataMaster（PixelDM = Pixel Data Master）：pixel/framebuffer 資料搬移
// 的資料主控。
// 它與 PBE（Pixel Back End）不同：PBE 產生 RGBA payload，本 module 驗證
// framebuffer store、指定 GPU address，並以 MemoryTxn FIFO handle 送往 SLC。
// Bulk RGBA bytes 留在 MemoryPool；服務時間採 event-driven completion。
#pragma once

#include "common/pipeline_state.h"

#include <systemc>

namespace pvrgpu::stub {

class PixelDataMaster final : public sc_core::sc_module {
 public:
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};
  sc_core::sc_fifo_out<MemoryTxn> output{"output"};

  PixelDataMaster(sc_core::sc_module_name name, MemoryPool &pool);

 private:
  void Run();

  MemoryPool &pool_;
};

}  // namespace pvrgpu::stub
