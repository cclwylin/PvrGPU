// ComputeDataMaster：compute workload 的 Data Master（資料主控）fetch-side 模組。
// 依照 DXTP 架構規範，它負責從 SLC/DRAM 取得 SSBO、Uniform Buffer、
// Atomic Counter 等資料，送入 USC Compute Shader 執行單元。
// 目前 graphics Fill.Solid slice 不使用 compute，亦沒有假造 command 或 timing 行為。
// 現階段為無 ports/process/timing、未連線的 elaboration placeholder。
#pragma once

#include <systemc>

namespace pvrgpu::stub {

class ComputeDataMaster final : public sc_core::sc_module {
 public:
  explicit ComputeDataMaster(sc_core::sc_module_name name);
};

}  // namespace pvrgpu::stub
