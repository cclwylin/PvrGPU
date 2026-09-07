// 僅建立幾何著色器的結構邊界；不代表已提供任何著色器功能。
#include "shader/geometry_shader.h"

namespace pvrgpu::stub {

GeometryShader::GeometryShader(sc_core::sc_module_name name)
    : sc_module(name) {}

}  // namespace pvrgpu::stub
