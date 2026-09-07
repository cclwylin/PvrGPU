// 僅建立曲面細分控制著色器的結構邊界；不宣稱已支援曲面細分。
#include "shader/tessellation_control_shader.h"

namespace pvrgpu::stub {

TessellationControlShader::TessellationControlShader(sc_core::sc_module_name name)
    : sc_module(name) {}

}  // namespace pvrgpu::stub
