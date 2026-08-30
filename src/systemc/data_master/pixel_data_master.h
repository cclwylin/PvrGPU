// PixelDataMaster（PixelDM = Pixel Data Master）：DXTP 架構中的 pixel 資料
// 輸入側模組，負責從 SLC/DRAM 取得 pixel tile 資料（深度、模板、tile buffer）
// 並送入 ISP / Fragment 前端。
// 注意：framebuffer write-back（PBE → SLC → DRAM 寫出路徑）已遷移至
// fragment/pbe_write_back，本模組已矯正為 fetch-side 結構預留點。
// 現階段為無 ports/process/timing、未連線的 elaboration placeholder。
#pragma once

#include <systemc>

namespace pvrgpu::stub {

class PixelDataMaster final : public sc_core::sc_module {
 public:
  explicit PixelDataMaster(sc_core::sc_module_name name);
};

}  // namespace pvrgpu::stub
