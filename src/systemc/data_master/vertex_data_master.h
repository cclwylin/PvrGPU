// VertexDataMaster（VertexDM = Vertex Data Master）：DXTP 架構中的頂點資料
// 輸入側模組，負責從 SLC/DRAM 取得頂點緩衝區（Vertex Buffer）、索引緩衝區
// （Index Buffer）以及頂點屬性（Vertex Attribute）資料，送入 PDS → USC。
// 依照架構圖，VertexDM 與 Pixel/Compute/Domain/2D DM 並列，同屬 fetch-side。
// 現階段 VDM + VertexFetch 承擔部分對應職責；完整 VertexDM DXTP protocol
// 待定義後實作。目前為無 ports/process/timing、未連線的 elaboration placeholder。
#pragma once

#include <systemc>

namespace pvrgpu::stub {

class VertexDataMaster final : public sc_core::sc_module {
 public:
  explicit VertexDataMaster(sc_core::sc_module_name name);
};

}  // namespace pvrgpu::stub
