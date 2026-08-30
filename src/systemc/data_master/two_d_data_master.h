// TwoDDataMaster：2D（two-dimensional，二維）blit workload 的 Data Master
// fetch-side 模組。它負責從 SLC/DRAM 讀取 blit 來源 buffer（src surface），
// 送入 2D blit engine 執行 glBlitFramebuffer / glCopyTexImage 等操作。
// 目前 3D Fill.Solid slice 不經此 block，也沒有假造 blit、DMA 或 timing 行為。
// 現階段為無 ports/process/timing、未連線的 elaboration placeholder。
#pragma once

#include <systemc>

namespace pvrgpu::stub {

class TwoDDataMaster final : public sc_core::sc_module {
 public:
  explicit TwoDDataMaster(sc_core::sc_module_name name);
};

}  // namespace pvrgpu::stub
