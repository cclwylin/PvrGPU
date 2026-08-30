// OnChipFabric 代表片上互連 fabric，對應使用者提供之架構附圖的 OMNI Bus 路徑。
// OMNI 是該匯流排的公開名稱；此處不臆造未由公開資料定義的縮寫全名。
// 它未來負責 GPU clients、MMU／BIF、MCU／TCU 與 SLC 間的 routing/arbitration。
// 目前只是無 ports/process/timing 的 placeholder、尚未連入 functional pipeline。
#pragma once

#include <systemc>

namespace pvrgpu::stub {

// TODO: Route memory transactions between GPU clients and the MMU/SLC path.
class OnChipFabric final : public sc_core::sc_module {
 public:
  explicit OnChipFabric(sc_core::sc_module_name name);
};

}  // namespace pvrgpu::stub
