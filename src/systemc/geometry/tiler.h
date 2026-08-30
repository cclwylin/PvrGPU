// Module：Tiler。
// 縮寫：非縮寫（分塊器）。
// 功能：將任意數量 RasterTriangle 分箱為 row-major 32×32 TileRecord 與
// ordered TilePrimitiveRef range；zero-area 與 face-culled setup candidates 不建立
// ref，但其原始 parameter_index/primitive identity 不 compact。
// 以 bounded FIFO 傳遞 MemoryPool handle，完成由 event-driven delay 通知。
#pragma once

#include "common/pipeline_state.h"

#include <systemc>

namespace pvrgpu::stub {

class Tiler final : public sc_core::sc_module {
public:
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};
  sc_core::sc_fifo_out<PipelineTxn> output{"output"};

  Tiler(sc_core::sc_module_name name, MemoryPool &pool);

private:
  void Run();

  MemoryPool &pool_;
};

} // namespace pvrgpu::stub
