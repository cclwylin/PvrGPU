// MMU（Memory Management Unit，記憶體管理單元）預留 virtual address（VA）轉譯與保護。
// BIF（Bus Interface，匯流排介面）預留把 GPU memory request 導向 SLC／system memory。
// MmuBif 是上述 address-management 與 bus-routing 責任的模型邊界。
// 目前只是無 ports/process/timing 的 placeholder、尚未連入 functional pipeline。
#pragma once

#include <systemc>

namespace pvrgpu::stub {

// TODO: Translate and route GPU virtual-memory requests to the shared cache.
class MmuBif final : public sc_core::sc_module {
 public:
  explicit MmuBif(sc_core::sc_module_name name);
};

}  // namespace pvrgpu::stub
