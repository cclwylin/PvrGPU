// ComputeDataMaster：compute workload 的 Data Master（資料主控）module boundary。
// 它預留 compute dispatch/task 的資料移動與 memory-request ingress；目前 graphics
// Fill.Solid slice 不使用 compute，亦沒有假造 command 或 timing 行為。
// 現階段為無 ports/process/timing、未連線的 elaboration placeholder。
#include "data_master/compute_data_master.h"

namespace pvrgpu::stub {

ComputeDataMaster::ComputeDataMaster(sc_core::sc_module_name name)
    : sc_module(name) {}

}  // namespace pvrgpu::stub
