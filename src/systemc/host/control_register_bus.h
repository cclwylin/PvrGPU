// ControlRegisterBus：控制與暫存器匯流排 module boundary。
// 它將承接 MMIO（Memory-Mapped I/O）、kick、status、event 與 interrupt
// register transport；目前不宣稱已有 register map、ordering 或 IRQ 行為。
// 現階段為無 ports/process/timing、未連線的 elaboration placeholder。
#pragma once

#include <systemc>

namespace pvrgpu::stub {

class ControlRegisterBus final : public sc_core::sc_module {
 public:
  explicit ControlRegisterBus(sc_core::sc_module_name name);
};

}  // namespace pvrgpu::stub
