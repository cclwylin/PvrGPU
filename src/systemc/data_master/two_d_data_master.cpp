// TwoDDataMaster：2D（two-dimensional，二維）blit workload 的 Data Master
// module boundary。它未來承接 driver 組成的 2D render packets；目前 3D
// Fill.Solid slice 不經此 block，也沒有假造 blit、DMA 或 timing 行為。
// 現階段為無 ports/process/timing、未連線的 elaboration placeholder。
#include "data_master/two_d_data_master.h"

namespace pvrgpu::stub {

TwoDDataMaster::TwoDDataMaster(sc_core::sc_module_name name)
    : sc_module(name) {}

}  // namespace pvrgpu::stub
