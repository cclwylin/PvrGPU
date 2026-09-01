// PbeWriteBack：PBE framebuffer write-back 模組。
// 原始邏輯由 PixelDataMaster 承擔，但 PixelDataMaster 依照 DXTP 架構規範
// 屬於 fetch-side（從 SLC/DRAM 取資料送入 Shader/PDS），因此 write-back
// 路徑遷移至本模組（fragment/ 子目錄，與 PBE 同層）。
#include "fragment/pbe_write_back.h"

#include "common/functional_types.h"
#include "memory/gpu_memory_system.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pvrgpu::stub {
namespace {

inline constexpr std::uint64_t kFramebufferGpuAddress = 0x10000000ULL;
inline constexpr std::uint64_t kPbeWriteBackLatency = 1;

bool SameHandle(PoolHandle left, PoolHandle right) {
  return left.slot == right.slot && left.generation == right.generation;
}

std::uint64_t CheckedAdd(std::uint64_t left, std::uint64_t right,
                         const char *description) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left)
    throw std::overflow_error(std::string("PbeWriteBack overflow: ") +
                              description);
  return left + right;
}

}  // namespace

PbeWriteBack::PbeWriteBack(sc_core::sc_module_name name, MemoryPool &pool,
                           GpuMemorySystem *memory)
    : sc_module(name), pool_(pool), memory_(memory) {
  SC_THREAD(Run);
}

void PbeWriteBack::Run() {
  while (true) {
    const PipelineTxn txn = input.read();
    PipelineState state = LoadPipelineState(pool_, txn.state);
    RequireStage(state.stage, PipelineStage::kPbeComplete, name());
    if (memory_ && state.memory_mode != memory_->mode())
      throw std::runtime_error("PbeWriteBack memory mode mismatch");
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

    if (state.framebuffer_gpu_address == 0) {
      state.framebuffer_gpu_address = kFramebufferGpuAddress;
    } else {
      const std::uint64_t sequence_offset =
          state.framebuffer_gpu_address - kDriverPcoSequenceColorAddressBase;
      if (state.framebuffer_gpu_address <
              kDriverPcoSequenceColorAddressBase ||
          sequence_offset % kDriverPcoSequenceAttachmentStride != 0 ||
          sequence_offset / kDriverPcoSequenceAttachmentStride >=
              kDriverPcoMaximumNestedSequenceCommands) {
        throw std::runtime_error(
            "PbeWriteBack sequence framebuffer address is invalid");
      }
    }
    state.counters.pixel_data_master_transactions = 1;
    state.counters.pixel_data_master_bytes = expected_bytes;
    state.counters.pixel_data_master_cycles = kPbeWriteBackLatency;
    state.counters.renderer_cycles += kPbeWriteBackLatency;
    state.stage = PipelineStage::kPixelDataMasterComplete;
    WaitForCycles(kPbeWriteBackLatency);

    if (memory_) {
      if (completion.size() == 0)
        throw std::runtime_error("PbeWriteBack completion port is unbound");
      if (HasPoolHandle(state.dram_framebuffer) ||
          HasPoolHandle(state.slc_writeback_lines) ||
          state.framebuffer_from_dram != 0) {
        throw std::runtime_error(
            "PbeWriteBack received existing framebuffer memory state");
      }

      const std::vector<std::uint8_t> source =
          LoadArray<std::uint8_t>(pool_, state.pbe_framebuffer);
      if (source.size() != static_cast<std::size_t>(expected_bytes))
        throw std::runtime_error("PbeWriteBack source framebuffer mismatch");

      MemoryAccessStats memory_stats =
          memory_->Write(state.framebuffer_gpu_address, source.data(),
                         source.size(), MemoryClient::kFramebuffer);
      MemoryReadResult readback = memory_->Readback(
          state.framebuffer_gpu_address, source.size(),
          MemoryClient::kFramebufferReadback);
      memory_stats += readback.stats;
      if (readback.data.size() != source.size())
        throw std::runtime_error("PbeWriteBack readback framebuffer mismatch");

      const PoolHandle readback_handle = StoreNewArray(pool_, readback.data);
      if (SameHandle(readback_handle, state.pbe_framebuffer)) {
        pool_.Release(readback_handle);
        throw std::logic_error("PbeWriteBack reused a live source handle");
      }

      const PoolHandle pbe_source = state.pbe_framebuffer;
      state.pbe_framebuffer = {};
      state.slc_writeback_lines = {};
      state.dram_framebuffer = readback_handle;
      state.framebuffer_from_dram = 1;
      ApplyMemoryAccessStats(state.counters, memory_stats);
      state.counters.framebuffer_dram_readback_bytes = expected_bytes;
      const std::uint64_t memory_cycles =
          MemoryAccessDelayCycles(memory_stats);
      state.counters.renderer_cycles = CheckedAdd(
          state.counters.renderer_cycles, memory_cycles, "renderer cycles");
      state.counters.virtual_gpu_cycles = CheckedAdd(
          CheckedAdd(state.counters.tiler_cycles, state.counters.renderer_cycles,
                     "virtual GPU pipeline cycles"),
          kReferenceUarch.fixed_submission_cycles,
          "virtual GPU submission cycles");
      state.counters.pool_high_water_bytes = pool_.high_water_bytes();
      state.stage = PipelineStage::kFramebufferReady;
      WaitForCycles(memory_cycles);

      try {
        StorePipelineState(pool_, txn.state, state);
      } catch (...) {
        pool_.Release(readback_handle);
        throw;
      }
      pool_.Release(pbe_source);
      completion->write(txn);
      continue;
    }

    if (output.size() == 0)
      throw std::runtime_error("PbeWriteBack output port is unbound");
    StorePipelineState(pool_, txn.state, state);

    MemoryTxn memory_txn;
    memory_txn.pipeline = txn;
    memory_txn.payload = state.pbe_framebuffer;
    memory_txn.address = state.framebuffer_gpu_address;
    memory_txn.bytes = expected_bytes;
    memory_txn.operation = MemoryOperation::kWrite;
    memory_txn.client = MemoryClient::kFramebuffer;
    memory_txn.payload_format = MemoryPayloadFormat::kLinearBytes;
    output->write(memory_txn);
  }
}

}  // namespace pvrgpu::stub
