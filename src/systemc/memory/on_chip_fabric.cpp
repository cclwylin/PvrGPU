// OnChipFabric 代表片上互連 fabric，對應使用者提供之架構附圖的 OMNI Bus 路徑。
// OMNI 是該匯流排的公開名稱；此處不臆造未由公開資料定義的縮寫全名。
// 它未來負責 GPU clients、MMU／BIF、MCU／TCU 與 SLC 間的 routing/arbitration。
// 目前只是無 ports/process/timing 的 placeholder、尚未連入 functional pipeline。
#include "memory/on_chip_fabric.h"

namespace pvrgpu::stub {

OnChipFabric::OnChipFabric(sc_core::sc_module_name name) : sc_module(name) {}

}  // namespace pvrgpu::stub
