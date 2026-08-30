// FirmwareScheduler：以抽象 job/data-master scheduler 對應 DXTP 架構參考的
// dedicated firmware processor control boundary。它未來負責 queue、barrier、
// dispatch 與 completion ordering；不模擬或宣稱執行任何 RISC-V firmware ISA。
// 目前是無 ports/process/timing、未連線的 elaboration placeholder。
#include "firmware/firmware_scheduler.h"

namespace pvrgpu::stub {

FirmwareScheduler::FirmwareScheduler(sc_core::sc_module_name name)
    : sc_module(name) {}

}  // namespace pvrgpu::stub
