// FragmentFrontend（fragment 前端；名稱不是縮寫）：將 ISP 最終可見的
// primitive/sample owner 轉成帶 identity/depth/barycentric 的 2×2 USC quad。
// FIFO 只傳 MemoryPool handle；candidate/invocation bulk data 留在 pool。
#pragma once

#include "common/pipeline_state.h"

#include <systemc>

namespace pvrgpu::stub {

class FragmentFrontend final : public sc_core::sc_module {
public:
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};
  sc_core::sc_fifo_out<PipelineTxn> output{"output"};

  FragmentFrontend(sc_core::sc_module_name name, MemoryPool &pool);

private:
  void Run();

  MemoryPool &pool_;
};

} // namespace pvrgpu::stub
