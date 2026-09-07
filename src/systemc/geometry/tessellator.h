// 固定功能曲面細分器的獨立模組預留；此模組不是著色器。
// 目前尚未支援曲面細分，不建立埠、處理程序或時序，亦不接收工作。
#pragma once

#include <systemc>

namespace pvrgpu::stub {

class Tessellator final : public sc_core::sc_module {
 public:
  explicit Tessellator(sc_core::sc_module_name name);
};

}  // namespace pvrgpu::stub
