// TileScheduler（tile 排程器）：把 parameter buffer 產生的 tile 工作送入
// fragment phase。 名稱不是縮寫；本模型以 scheduled tile 數代表 renderer
// 可執行的 tile 工作集合。Solid-color cases 在 fragment PCO decode 後
// 讀取 TileRecord 與 ordered TilePrimitiveRef；face-cull 明確允許空 ref
// stream，仍記錄 reference uArch 可執行的 tile 工作集合。
// FIFO transaction 只攜帶 MemoryPool handle；tile bulk data 留在
// pool，服務時間採事件驅動。
#pragma once

#include "common/pipeline_state.h"
#include "memory/gpu_memory_system.h"

#include <systemc>

namespace pvrgpu::stub {

class TileScheduler final : public sc_core::sc_module {
public:
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};
  sc_core::sc_fifo_out<PipelineTxn> output{"output"};

  TileScheduler(sc_core::sc_module_name name, MemoryPool &pool,
                GpuMemorySystem *memory = nullptr);

private:
  void Run();

  MemoryPool &pool_;
  GpuMemorySystem *memory_;
};

} // namespace pvrgpu::stub
