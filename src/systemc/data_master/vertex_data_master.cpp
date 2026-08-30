// VertexDataMaster：fetch-side 結構預留點。
// 對應架構圖中與 Pixel/Compute/Domain/2D DM 並列的 Vertex Data Master block。
// 待 DXTP vertex fetch protocol 定義後實作完整功能。
#include "data_master/vertex_data_master.h"

namespace pvrgpu::stub {

VertexDataMaster::VertexDataMaster(sc_core::sc_module_name name)
    : sc_module(name) {}

}  // namespace pvrgpu::stub
