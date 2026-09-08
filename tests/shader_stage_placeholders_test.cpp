// 三個真正 tessellation 模組的獨立邊界與 idle/backpressure 驗證。
#include "geometry/tessellator.h"
#include "shader/tessellation_control_shader.h"
#include "shader/tessellation_evaluation_shader.h"
#include "memory/gpu_memory_system.h"

#include <systemc>

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {

void Check(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

void CheckBoundary(const sc_core::sc_module &module, const char *name) {
  Check(std::strcmp(module.name(), name) == 0, "獨立模組名稱不正確");
  Check(std::strcmp(module.kind(), "sc_module") == 0,
        "不是獨立的 SystemC 模組");
  Check(module.get_parent_object() == nullptr, "不應附掛於其他著色器");
  unsigned inputs = 0, outputs = 0, threads = 0;
  for (const auto *child : module.get_child_objects()) {
    inputs += std::strcmp(child->kind(), "sc_fifo_in") == 0;
    outputs += std::strcmp(child->kind(), "sc_fifo_out") == 0;
    threads += std::strcmp(child->kind(), "sc_thread_process") == 0;
  }
  Check(inputs == 1 && outputs == 1 && threads == 1,
        "每個 stage 必須有自己的 FIFO input/output 與 event-driven process");
}

}  // namespace

int sc_main(int, char **) {
  try {
    using namespace pvrgpu::stub;
    MemoryPool pool;
    GpuMemorySystem memory(MemoryMode::kDirect);
    sc_core::sc_fifo<PipelineTxn> before_control("before_control", 1);
    sc_core::sc_fifo<PipelineTxn> before_fixed("before_fixed", 1);
    sc_core::sc_fifo<PipelineTxn> before_evaluation("before_evaluation", 1);
    sc_core::sc_fifo<PipelineTxn> after_evaluation("after_evaluation", 1);
    TessellationControlShader control_shader("tessellation_control_shader", pool, &memory);
    TessellationEvaluationShader evaluation_shader("tessellation_evaluation_shader", pool, &memory);
    Tessellator tessellator("tessellator", pool, &memory);
    control_shader.input(before_control); control_shader.output(before_fixed);
    tessellator.input(before_fixed); tessellator.output(before_evaluation);
    evaluation_shader.input(before_evaluation); evaluation_shader.output(after_evaluation);
    const std::array<const sc_core::sc_module *,3> modules{
        &control_shader, &evaluation_shader, &tessellator};
    const std::array<const char *,3> names{
        "tessellation_control_shader",
        "tessellation_evaluation_shader", "tessellator"};
    const auto verify = [&] {
      const auto &top = sc_core::sc_get_top_level_objects();
      Check(top.size() == modules.size() + 4, "應恰好建立三個獨立頂層模組及四個有界 FIFO");
      for (std::size_t index = 0; index < modules.size(); ++index) {
        CheckBoundary(*modules[index], names[index]);
        Check(std::count(top.begin(), top.end(), modules[index]) == 1,
              "模組未恰好出現於頂層一次");
      }
      for (const auto *fifo : {&before_control, &before_fixed, &before_evaluation, &after_evaluation})
        Check(fifo->num_available() == 0 && fifo->num_free() == 1, "idle FIFO 必須為空且深度為一");
    };
    verify();
    sc_core::sc_start(sc_core::SC_ZERO_TIME);
    verify();
    Check(sc_core::sc_time_stamp() == sc_core::SC_ZERO_TIME,
          "idle stage 不應推進模擬時間");
    Check(!sc_core::sc_pending_activity(), "idle stage 不應 polling 或留下待執行活動");
    Check(pool.allocations() == 0, "未提交時不配置 shader 工作或虛構結果");
    PipelineState original;
    const auto state = pool.Allocate(sizeof(PipelineState));
    StorePipelineState(pool, state, original);
    PipelineTxn first; first.state = state; first.sequence = 31;
    PipelineTxn second = first; second.sequence = 32;
    Check(before_control.nb_write(first), "depth-one FIFO 接收第一筆");
    Check(!before_control.nb_write(second), "depth-one FIFO 必須反壓第二筆");
    sc_core::sc_start(sc_core::sc_time(1, sc_core::SC_NS));
    PipelineTxn completed;
    Check(after_evaluation.nb_read(completed) && completed.sequence == 31,
          "非 tessellation 工作依序通過三個獨立 stage");
    const auto unchanged = LoadPipelineState(pool, completed.state);
    Check(std::memcmp(&original, &unchanged, sizeof(original)) == 0,
          "未提交 tessellation 不得執行 shader 或填入計數答案");
    pool.Release(state);
    Check(pool.bytes_in_flight() == 0 && pool.allocations() == pool.releases(), "結構測試 pool 平衡");
    std::cout << "三個獨立 tessellation event-driven 模組與 bounded FIFO 驗證通過\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
