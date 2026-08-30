// Module：JsonReporter；JSON 是 JavaScript Object Notation。
// 功能：接收完成的 PipelineTxn，驗證 framebuffer-ready stage 與 DRAM
// readback provenance，輸出
// pvrgpu.counter.v1 JSONL（JSON Lines）counter 與每 DrawList 的 VS/FS
// static/dynamic instruction 統計、實際 vertex PCO binary fingerprint 與
// decoded opcode histogram，並負責 PNG（Portable Network Graphics）發布後的
// MemoryPool payload/state 回收與 leak 回報。
#pragma once

#include "common/pipeline_state.h"

#include <systemc>

namespace pvrgpu::stub {

class JsonReporter final : public sc_core::sc_module {
public:
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};

  JsonReporter(sc_core::sc_module_name name, const Options &options,
               MemoryPool &pool);

  bool failed() const { return failed_; }

private:
  void Run();

  Options options_;
  MemoryPool &pool_;
  bool failed_ = false;
};

} // namespace pvrgpu::stub
