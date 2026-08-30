// SocBusInterface：SoC = System-on-Chip（系統單晶片）。此 module 保留
// DXTP 架構參考中的 host control-bus ingress 邊界；未來才會把外部 bus
// transaction 轉為 PvrGPU 內部 FIFO transaction。
// 目前是無 ports/process/timing 的 elaboration placeholder，不代表已實作 AXI。
#pragma once

#include <systemc>

namespace pvrgpu::stub {

class SocBusInterface final : public sc_core::sc_module {
 public:
  explicit SocBusInterface(sc_core::sc_module_name name);
};

}  // namespace pvrgpu::stub
