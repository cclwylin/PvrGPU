// PbeWriteBack：PBE（Pixel Back End）完成後的 framebuffer write-back 模組。
// 它驗證 PBE 輸出的 framebuffer handle，指定 GPU 虛擬位址，並將
// MemoryTxn（kFramebuffer, kWrite）送往 SLC → DRAM。
// Bulk RGBA bytes 留在 MemoryPool；服務時間採 event-driven completion。
// 注意：本模組取代原先誤放在 data_master/ 的 PixelDataMaster write-back 邏輯。
// data_master/ 已矯正為 fetch-side（SLC/DRAM → Shader/PDS）的結構預留點。
#pragma once

#include "common/pipeline_state.h"

#include <systemc>

namespace pvrgpu::stub {

class GpuMemorySystem;

class PbeWriteBack final : public sc_core::sc_module {
 public:
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};
  sc_core::sc_port<sc_core::sc_fifo_out_if<MemoryTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      output{"output"};
  sc_core::sc_port<sc_core::sc_fifo_out_if<PipelineTxn>, 0,
                   sc_core::SC_ZERO_OR_MORE_BOUND>
      completion{"completion"};

  PbeWriteBack(sc_core::sc_module_name name, MemoryPool &pool,
               GpuMemorySystem *memory = nullptr);

 private:
  void Run();

  MemoryPool &pool_;
  GpuMemorySystem *memory_;
};

}  // namespace pvrgpu::stub
