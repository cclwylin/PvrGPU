// 僅建立固定功能曲面細分器的結構邊界；不執行任何著色器或細分工作。
#include "geometry/tessellator.h"

namespace pvrgpu::stub {

Tessellator::Tessellator(sc_core::sc_module_name name)
    : sc_module(name) {}

}  // namespace pvrgpu::stub
