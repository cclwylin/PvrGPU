// XpuInterface 對應 DXTP 架構參考中的 XPU Interface block。
// 公開可驗證資料尚不足以展開 XPU 名稱或 protocol，因此此處只保留明確的
// module boundary，不臆造 request、coherency 或 timing 行為。
// 目前是無 ports/process/timing、未連線的 elaboration placeholder。
#include "host/xpu_interface.h"

namespace pvrgpu::stub {

XpuInterface::XpuInterface(sc_core::sc_module_name name) : sc_module(name) {}

}  // namespace pvrgpu::stub
