// Module：Submitter（非縮寫，意為工作提交器）。
// 功能：依執行選項建立每一幀的 PipelineState 與目前通過 gate 的內建
// GLBench Fill.Solid、Triangle.Setup、AttributeFetchShader 與通過 gate 的
// VaryingsShader 1/2/4/8 fixture。它配置
// VBO（Vertex Buffer Object，頂點緩衝）、EBO（Element Buffer Object，索引
// 緩衝）、attribute binding、cull state、真實 PCO shader binary 與 DrawList
// 統計 payload。大型資料留在 generation-checked MemoryPool；PipelineTxn
// 只攜帶 handle，透過有界 FIFO（First-In, First-Out）啟動 event-driven
// pipeline。PCO 是 Mesa 公開 PowerVR compiler backend 的名稱，本專案不推定
// 其縮寫展開。
#pragma once

#include "memory_pool.h"
#include "model_job.h"
#include "model_types.h"

#include <systemc>

#include <cstdint>

namespace pvrgpu::stub {

class GpuMemorySystem;

class Submitter final : public sc_core::sc_module {
public:
  sc_core::sc_fifo_out<PipelineTxn> output{"output"};

  // `job` is optional.  Without it the submitter queues the work `options`
  // describes and its thread ends, which is what a single-shot run wants.
  // With it the thread waits for each flush instead, adopts that flush's
  // options, drains them, and goes back to waiting -- so the pipeline is idle
  // rather than finished between draws, and a second draw can follow a
  // readback.
  Submitter(sc_core::sc_module_name name, MemoryPool &pool,
            const Options &options, GpuMemorySystem *memory = nullptr,
            sc_core::sc_event *sequence_completion = nullptr,
            ModelJob *job = nullptr);

  std::uint64_t fifo_stalls() const { return fifo_stalls_; }

private:
  void Run();
  void RunJob();

  MemoryPool &pool_;
  Options options_;
  GpuMemorySystem *memory_ = nullptr;
  sc_core::sc_event *sequence_completion_ = nullptr;
  ModelJob *job_ = nullptr;
  std::uint64_t fifo_stalls_ = 0;
};

} // namespace pvrgpu::stub
