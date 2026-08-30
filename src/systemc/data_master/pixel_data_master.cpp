// PixelDataMaster（PixelDM = Pixel Data Master）：pixel/framebuffer 資料搬移
// 的資料主控。
// 它與 PBE（Pixel Back End）不同：PBE 產生 RGBA payload，本 module 驗證
// framebuffer store、指定 GPU address，並以 MemoryTxn FIFO handle 送往 SLC。
// Bulk RGBA bytes 留在 MemoryPool；服務時間採 event-driven completion。
#include "data_master/pixel_data_master.h"

#include "common/functional_types.h"

#include <limits>
#include <stdexcept>

namespace pvrgpu::stub {
namespace {

inline constexpr std::uint64_t kFramebufferGpuAddress = 0x10000000ULL;
inline constexpr std::uint64_t kPixelDataMasterLatency = 1;

}  // namespace

PixelDataMaster::PixelDataMaster(sc_core::sc_module_name name,
                                 MemoryPool &pool)
    : sc_module(name), pool_(pool) {
  SC_THREAD(Run);
}

void PixelDataMaster::Run() {
  while (true) {
    const PipelineTxn txn = input.read();
    PipelineState state = LoadPipelineState(pool_, txn.state);
    RequireStage(state.stage, PipelineStage::kPbeComplete, name());
    if (!HasPoolHandle(state.pbe_framebuffer))
      throw std::runtime_error("PixelDataMaster received no PBE framebuffer");

    const std::uint64_t expected_bytes =
        static_cast<std::uint64_t>(state.width) * state.height * 4U;
    if (expected_bytes == 0 || state.framebuffer_bytes != expected_bytes ||
        pool_.Read(state.pbe_framebuffer).size() != expected_bytes) {
      throw std::runtime_error(
          "PixelDataMaster framebuffer byte count mismatch");
    }
    if (expected_bytes > std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("PixelDataMaster framebuffer is too large");

    state.framebuffer_gpu_address = kFramebufferGpuAddress;
    state.counters.pixel_data_master_transactions = 1;
    state.counters.pixel_data_master_bytes = expected_bytes;
    state.counters.pixel_data_master_cycles = kPixelDataMasterLatency;
    state.counters.renderer_cycles += kPixelDataMasterLatency;
    state.stage = PipelineStage::kPixelDataMasterComplete;
    WaitForCycles(kPixelDataMasterLatency);
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
