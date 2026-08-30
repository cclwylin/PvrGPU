// DomainDataMaster：DXTP 架構中 Domain stage 的 Data Master fetch-side 模組。
// 它負責從 SLC/DRAM 取得 Tessellation Control/Evaluation Shader 所需的
// domain 輸入資料（patch constants、per-vertex data），送入 Domain 執行單元。
// 現有可驗證資料尚不足以定義其 domain protocol 或 task semantics，因此只保留
// 獨立 module boundary，未知行為不得以 silent no-op 假裝已支援。
// 目前是無 ports/process/timing、未連線的 elaboration placeholder。
#pragma once

#include <systemc>

namespace pvrgpu::stub {

class DomainDataMaster final : public sc_core::sc_module {
 public:
  explicit DomainDataMaster(sc_core::sc_module_name name);
};

}  // namespace pvrgpu::stub
