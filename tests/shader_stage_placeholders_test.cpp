// 僅驗證四個獨立結構預留；不建立埠、假工作或假執行程序。
#include "geometry/tessellator.h"
#include "shader/geometry_shader.h"
#include "shader/tessellation_control_shader.h"
#include "shader/tessellation_evaluation_shader.h"

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
  Check(std::strcmp(module.name(), name) == 0, "結構預留的模組名稱不正確");
  Check(std::strcmp(module.kind(), "sc_module") == 0,
        "結構預留不是獨立的 SystemC 模組");
  Check(module.get_parent_object() == nullptr, "結構預留不應附掛於其他著色器");
  Check(module.get_child_objects().empty(), "結構預留不應包含埠或處理程序");
  Check(module.get_child_events().empty(), "結構預留不應包含時序事件");
}

}  // namespace

int sc_main(int, char **) {
  try {
    pvrgpu::stub::GeometryShader geometry_shader("geometry_shader");
    pvrgpu::stub::TessellationControlShader control_shader("tessellation_control_shader");
    pvrgpu::stub::TessellationEvaluationShader evaluation_shader("tessellation_evaluation_shader");
    pvrgpu::stub::Tessellator tessellator("tessellator");
    const std::array<const sc_core::sc_module *,4> modules{
        &geometry_shader, &control_shader, &evaluation_shader, &tessellator};
    const std::array<const char *,4> names{
        "geometry_shader", "tessellation_control_shader",
        "tessellation_evaluation_shader", "tessellator"};
    const auto verify = [&] {
      const auto &top = sc_core::sc_get_top_level_objects();
      Check(top.size() == modules.size(), "應恰好建立四個獨立頂層模組");
      for (std::size_t index = 0; index < modules.size(); ++index) {
        CheckBoundary(*modules[index], names[index]);
        Check(std::count(top.begin(), top.end(), modules[index]) == 1,
              "結構預留未恰好出現於頂層一次");
      }
      Check(sc_core::sc_get_top_level_events().empty(), "不應額外建立頂層時序事件");
    };
    verify();
    sc_core::sc_start(sc_core::SC_ZERO_TIME);
    verify();
    Check(sc_core::sc_time_stamp() == sc_core::SC_ZERO_TIME,
          "結構預留不應推進模擬時間");
    Check(!sc_core::sc_pending_activity(), "結構預留不應留下待執行活動");
    std::cout << "四個獨立著色階段／固定功能結構預留驗證通過\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
