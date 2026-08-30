// PixelDataMaster（PixelDM = Pixel Data Master）：fetch-side 結構預留點。
// write-back 邏輯已遷移至 fragment/pbe_write_back.cpp。
// 本模組預留 SLC/DRAM → ISP/Fragment 的 pixel tile fetch 路徑，
// 待 DXTP pixel fetch protocol 定義後實作。
#include "data_master/pixel_data_master.h"

namespace pvrgpu::stub {

PixelDataMaster::PixelDataMaster(sc_core::sc_module_name name)
    : sc_module(name) {}

}  // namespace pvrgpu::stub
