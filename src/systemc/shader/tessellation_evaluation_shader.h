// 曲面細分求值著色器的獨立模組預留；目前尚未支援此著色階段。
// 不建立埠、處理程序或時序，也不接收、忽略或轉交任何工作。
#pragma once

#include <systemc>

namespace pvrgpu::stub {

class TessellationEvaluationShader final : public sc_core::sc_module {
 public:
  explicit TessellationEvaluationShader(sc_core::sc_module_name name);
};

}  // namespace pvrgpu::stub
