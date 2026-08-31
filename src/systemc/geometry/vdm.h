// Module：Vdm。
// 縮寫：VDM = Vertex Data Master（公開 Imagination register 文件之用語）。
// 功能：fail-closed 驗證 Fill.Solid non-indexed triangle strip，或目前支援的
// Triangle.Setup／AttributeFetchShader uint16 indexed triangle list；VBO
// capacity 由 MemoryPool 內的 resource/binding table 推導，IA（Input
// Assembler，輸入組裝器）統計來自真實 draw/index occurrence 與 primitive。
// 以 bounded FIFO（First-In, First-Out）移交 MemoryPool handle，並以
// event-driven completion 推進。
#pragma once

#include "common/pipeline_state.h"

#include <systemc>

namespace pvrgpu::stub {

class GpuMemorySystem;

class Vdm final : public sc_core::sc_module {
 public:
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};
  sc_core::sc_fifo_out<PipelineTxn> output{"output"};

  Vdm(sc_core::sc_module_name name, MemoryPool &pool,
      GpuMemorySystem *memory = nullptr);

 private:
  void Run();

  MemoryPool &pool_;
  GpuMemorySystem *memory_ = nullptr;
};

} // namespace pvrgpu::stub
