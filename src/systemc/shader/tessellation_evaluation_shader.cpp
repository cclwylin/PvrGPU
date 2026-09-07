// 僅建立曲面細分求值著色器的結構邊界；不宣稱已支援曲面細分。
#include "shader/tessellation_evaluation_shader.h"

namespace pvrgpu::stub {

TessellationEvaluationShader::TessellationEvaluationShader(
    sc_core::sc_module_name name)
    : sc_module(name) {}

}  // namespace pvrgpu::stub
