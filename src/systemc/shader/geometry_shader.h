// 幾何著色器的獨立模組預留；目前尚未支援幾何著色器執行。
// 不建立埠、處理程序或時序，也不接收、忽略或轉交任何工作。
#pragma once

#include <systemc>

namespace pvrgpu::stub {

class GeometryShader final : public sc_core::sc_module {
 public:
  explicit GeometryShader(sc_core::sc_module_name name);
};

}  // namespace pvrgpu::stub
