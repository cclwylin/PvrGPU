// DomainDataMaster 對應 DXTP 架構參考中的 Domain Data Master block。
// 現有可驗證資料尚不足以定義其 domain protocol 或 task semantics，因此只保留
// 獨立 module boundary，未知行為不得以 silent no-op 假裝已支援。
// 目前是無 ports/process/timing、未連線的 elaboration placeholder。
#include "data_master/domain_data_master.h"

namespace pvrgpu::stub {

DomainDataMaster::DomainDataMaster(sc_core::sc_module_name name)
    : sc_module(name) {}

}  // namespace pvrgpu::stub
