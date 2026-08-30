// PcoDecoder：從 MemoryPool fetch 真實 PCO bytes，嚴格解碼成 USC ISS
// semantic instructions。PCO 是 Mesa 公開 PowerVR shader backend 名稱，
// 公開資料未提供可證實的縮寫；ISS = Instruction Set Simulator。
// 同時記錄每 DrawList、每 shader stage 的 static ALU/Tex/Memory program
// 組成。未知 encoding fail closed；FIFO 只傳 handle，採 event-driven
// completion。
#pragma once

#include "common/shader_stage.h"
#include "memory_pool.h"
#include "model_types.h"

#include <systemc>

namespace pvrgpu::stub {

class PcoDecoder final : public sc_core::sc_module {
public:
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};
  sc_core::sc_fifo_out<PipelineTxn> output{"output"};

  PcoDecoder(sc_core::sc_module_name name, MemoryPool &pool, ShaderStage stage);

private:
  void Run();

  MemoryPool &pool_;
  ShaderStage stage_;
};

} // namespace pvrgpu::stub
