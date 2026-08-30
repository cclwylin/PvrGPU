// MemFabric 的 Mem 是 Memory，代表 SLC 之外通往 system memory 的外部記憶體互連。
// DRAM（Dynamic Random-Access Memory，動態隨機存取記憶體）是此路徑未來的服務端。
// 它將建模外部 latency/contention，與片上的 OnChipFabric／OMNI Bus 分工不同。
// 目前只是無 ports/process/timing 的 placeholder、尚未連入 functional pipeline。
#include "memory/mem_fabric.h"

namespace pvrgpu::stub {

MemFabric::MemFabric(sc_core::sc_module_name name) : sc_module(name) {}

}  // namespace pvrgpu::stub
