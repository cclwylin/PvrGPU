// PbeWriteBack：PBE framebuffer write-back 模組。
// 原始邏輯由 PixelDataMaster 承擔，但 PixelDataMaster 依照 DXTP 架構規範
// 屬於 fetch-side（從 SLC/DRAM 取資料送入 Shader/PDS），因此 write-back
// 路徑遷移至本模組（fragment/ 子目錄，與 PBE 同層）。
#include "fragment/pbe_write_back.h"

#include "common/functional_types.h"

#include <limits>
#include <stdexcept>

namespace pvrgpu::stub {
namespace {

inline constexpr std::uint64_t kFramebufferGpuAddress = 0x10000000ULL;
inline constexpr std::uint64_t kPbeWriteBackLatency = 1;

}  // namespace

PbeWriteBack::PbeWriteBack(sc_core::sc_module_name name, MemoryPool &pool)
    : sc_module(name), pool_(pool) {
  SC_THREAD(Run);
}

void PbeWriteBack::Run() {
  while (true) {
    const PipelineTxn txn = input.read();
    PipelineState state = LoadPipelineState(pool_, txn.state);
    RequireStage(state.stage, PipelineStage::kPbeComplete, name());
    if (!HasPoolHandle(state.pbe_framebuffer))
      throw std::runtime_error("PbeWriteBack received no PBE framebuffer");

    const std::uint64_t expected_bytes =
        static_cast<std::uint64_t>(state.width) * state.height * 4U;
    if (expected_bytes == 0 || state.framebuffer_bytes != expected_bytes ||
        pool_.Read(state.pbe_framebuffer).size() != expected_bytes) {
      throw std::runtime_error(
          "PbeWriteBack framebuffer byte count mismatch");
    }
    if (expected_bytes > std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("PbeWriteBack framebuffer is too large");

    state.framebuffer_gpu_address = kFramebufferGpuAddress;
    state.counters.pixel_data_master_transactions = 1;
    state.counters.pixel_data_master_bytes = expected_bytes;
    state.counters.pixel_data_master_cycles = kPbeWriteBackLatency;
    state.counters.renderer_cycles += kPbeWriteBackLatency;
    state.stage = PipelineStage::kPixelDataMasterComplete;
    WaitForCycles(kPbeWriteBackLatency);
    StorePipelineState(pool_, txn.state, state);

    MemoryTxn memory_txn;
    memory_txn.pipeline = txn;
    memory_txn.payload = state.pbe_framebuffer;
    memory_txn.address = state.framebuffer_gpu_address;
    memory_txn.bytes = expected_bytes;
    memory_txn.operation = MemoryOperation::kWrite;
    memory_txn.client = MemoryClient::kFramebuffer;
    memory_txn.payload_format = MemoryPayloadFormat::kLinearBytes;
    output.write(memory_txn);
  }
}

}  // namespace pvrgpu::stub
