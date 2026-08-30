// Module：ParameterBuffer。
// 縮寫：非縮寫（參數緩衝區）。
// 功能：為任意數量 RasterTriangle 建立 24.8-style fixed-point top-left
// edge equations、depth interpolation 與 bounding box（邊界框）。
// Zero-area 與 face-culled setup candidate 保留同 index placeholder 但跳過
// equation，確保
// TilePrimitiveRef.parameter_index identity 不被 compact。由 bounded FIFO
// 接收 MemoryPool handle，以 event-driven completion 發布參數。
#pragma once

#include "common/pipeline_state.h"

#include <systemc>

namespace pvrgpu::stub {

class ParameterBuffer final : public sc_core::sc_module {
public:
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};
  sc_core::sc_fifo_out<PipelineTxn> output{"output"};

  ParameterBuffer(sc_core::sc_module_name name, MemoryPool &pool);

private:
  void Run();

  MemoryPool &pool_;
};

} // namespace pvrgpu::stub
