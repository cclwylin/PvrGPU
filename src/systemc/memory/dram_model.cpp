// DramModel（Dynamic Random-Access Memory model）的 functional/timing實作。
// UploadRun 先把完整 mip allocation 預置到 sparse 128-byte page backing；
// TextureRun 處理 TPU -> TCU（Texture Cache Unit）-> SLC（System Level
// Cache）miss 所產生的 read/response。Frame-buffer Run 在 cache-bypass=off
// 時寫 DramLineWrite，on 時寫 linear payload，兩者均落入同一 backing，再執行
// 獨立 DRAM read 產生 PNG 的新 MemoryPool handle。WaitForCycles 只排程
// transaction-completion event，沒有 clock 或逐 cycle loop。
#include "memory/dram_model.h"

#include "common/functional_types.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pvrgpu::stub {
namespace {

bool SameHandle(PoolHandle left, PoolHandle right) {
  return left.slot == right.slot && left.generation == right.generation;
}

std::uint64_t CheckedAdd(std::uint64_t left, std::uint64_t right,
                         const char *description) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left)
    throw std::overflow_error(std::string("DramModel overflow: ") +
                              description);
  return left + right;
}

std::size_t CheckedSize(std::uint64_t value, const char *description) {
  if (value > std::numeric_limits<std::size_t>::max())
    throw std::overflow_error(std::string("DramModel size overflow: ") +
                              description);
  return static_cast<std::size_t>(value);
}

void AddCounter(std::uint64_t &counter, std::uint64_t amount,
                const char *description) {
  counter = CheckedAdd(counter, amount, description);
}

} // namespace

DramModel::DramModel(sc_core::sc_module_name name, MemoryPool &pool)
    : sc_module(name), pool_(pool) {
  SC_THREAD(Run);
  SC_THREAD(TextureRun);
  SC_THREAD(UploadRun);
}

void DramModel::EnsureBackingRange(std::uint64_t address, std::size_t bytes) {
  if (bytes == 0)
    throw std::invalid_argument("DramModel backing range is empty");
  const std::uint64_t requested_end =
      CheckedAdd(address, bytes, "backing range end");
  std::uint64_t page = address - address % kBackingPageBytes;
  while (page < requested_end) {
    backing_pages_.try_emplace(page);
    page = CheckedAdd(page, kBackingPageBytes, "backing page advance");
  }
}

void DramModel::WriteBacking(std::uint64_t address,
                             const std::uint8_t *source,
                             std::size_t bytes) {
  if (!source && bytes != 0)
    throw std::invalid_argument("DramModel write source is null");
  EnsureBackingRange(address, bytes);
  std::size_t copied = 0;
  while (copied < bytes) {
    const std::uint64_t current = CheckedAdd(address, copied, "write address");
    const std::uint64_t page_address =
        current - current % kBackingPageBytes;
    const std::size_t page_offset =
        static_cast<std::size_t>(current - page_address);
    const std::size_t chunk =
        std::min(bytes - copied, kBackingPageBytes - page_offset);
    auto &page = backing_pages_.at(page_address);
    std::copy_n(source + copied, chunk, page.begin() + page_offset);
    copied += chunk;
  }
}

std::vector<std::uint8_t>
DramModel::ReadBacking(std::uint64_t address, std::size_t bytes) const {
  if (bytes == 0)
    throw std::runtime_error("DramModel read is outside initialized backing");
  std::vector<std::uint8_t> result(bytes);
  std::size_t copied = 0;
  while (copied < bytes) {
    const std::uint64_t current = CheckedAdd(address, copied, "read address");
    const std::uint64_t page_address =
        current - current % kBackingPageBytes;
    const auto page = backing_pages_.find(page_address);
    if (page == backing_pages_.end())
      throw std::runtime_error("DramModel read exceeds initialized backing");
    const std::size_t page_offset =
        static_cast<std::size_t>(current - page_address);
    const std::size_t chunk =
        std::min(bytes - copied, kBackingPageBytes - page_offset);
    std::copy_n(page->second.begin() + page_offset, chunk,
                result.begin() + copied);
    copied += chunk;
  }
  return result;
}

void DramModel::Run() {
  while (true) {
    const MemoryTxn memory_txn = input.read();
    const PipelineTxn pipeline_txn = memory_txn.pipeline;
    PipelineState state = LoadPipelineState(pool_, pipeline_txn.state);
    RequireStage(state.stage, PipelineStage::kSlcComplete, name());

    if (pipeline_txn.frame != state.counters.frame ||
        pipeline_txn.sequence != state.sequence) {
      throw std::runtime_error("DramModel pipeline identity mismatch");
    }
    if (memory_txn.operation != MemoryOperation::kWrite ||
        memory_txn.client != MemoryClient::kFramebuffer) {
      throw std::runtime_error(
          "DramModel received a non-framebuffer write transaction");
    }
    if (memory_txn.address != state.framebuffer_gpu_address ||
        memory_txn.bytes == 0 ||
        memory_txn.bytes != state.framebuffer_bytes) {
      throw std::runtime_error("DramModel framebuffer transaction mismatch");
    }
    if (HasPoolHandle(state.dram_framebuffer) ||
        state.framebuffer_from_dram != 0) {
      throw std::runtime_error("DramModel received an existing readback");
    }

    const std::size_t framebuffer_bytes =
        CheckedSize(memory_txn.bytes, "framebuffer bytes");
    std::uint64_t write_transactions = 0;
    std::uint64_t write_bytes = 0;
    std::vector<std::uint8_t> linear_payload;
    std::vector<DramLineWrite> line_writes;

    if (state.cache_bypass != 0) {
      if (memory_txn.payload_format != MemoryPayloadFormat::kLinearBytes ||
          !HasPoolHandle(state.pbe_framebuffer) ||
          !SameHandle(memory_txn.payload, state.pbe_framebuffer) ||
          HasPoolHandle(state.slc_writeback_lines)) {
        throw std::runtime_error(
            "DramModel cache-bypass payload contract mismatch");
      }
      linear_payload =
          LoadArray<std::uint8_t>(pool_, memory_txn.payload);
      if (linear_payload.size() != framebuffer_bytes)
        throw std::runtime_error(
            "DramModel linear framebuffer byte count mismatch");
      write_transactions = 1;
      write_bytes = memory_txn.bytes;
    } else {
      if (memory_txn.payload_format !=
              MemoryPayloadFormat::kCacheLineWrites ||
          !HasPoolHandle(state.slc_writeback_lines) ||
          !SameHandle(memory_txn.payload, state.slc_writeback_lines)) {
        throw std::runtime_error(
            "DramModel SLC writeback payload contract mismatch");
      }
      line_writes = LoadArray<DramLineWrite>(pool_, memory_txn.payload);
      if (line_writes.empty())
        throw std::runtime_error("DramModel received no SLC writeback lines");

      if (memory_txn.address % kDramLineWriteBytes != 0)
        throw std::runtime_error("DramModel framebuffer base is not aligned");
      const std::uint64_t expected_line_count =
          CeilDivide(memory_txn.bytes, kDramLineWriteBytes);
      if (line_writes.size() != expected_line_count)
        throw std::runtime_error(
            "DramModel SLC writeback line count mismatch");
      std::vector<std::uint8_t> covered_lines(line_writes.size(), 0);
      for (const DramLineWrite &write : line_writes) {
        if (write.bytes != kDramLineWriteBytes ||
            write.address < memory_txn.address ||
            write.address % kDramLineWriteBytes != 0) {
          throw std::runtime_error("DramModel received an invalid line write");
        }
        const std::uint64_t offset = write.address - memory_txn.address;
        if (offset % kDramLineWriteBytes != 0)
          throw std::runtime_error("DramModel line offset is not aligned");
        const std::uint64_t line_index = offset / kDramLineWriteBytes;
        if (line_index >= expected_line_count ||
            covered_lines[static_cast<std::size_t>(line_index)] != 0) {
          throw std::runtime_error(
              "DramModel received an out-of-range or duplicate line write");
        }
        covered_lines[static_cast<std::size_t>(line_index)] = 1;
        write_bytes = CheckedAdd(write_bytes, write.bytes,
                                 "physical DRAM write bytes");
      }
      if (std::find(covered_lines.begin(), covered_lines.end(),
                    std::uint8_t{0}) != covered_lines.end()) {
        throw std::runtime_error(
            "DramModel SLC writebacks do not cover the framebuffer");
      }
      write_transactions = line_writes.size();
    }

    if (write_transactions >
        std::numeric_limits<std::uint64_t>::max() /
            kDramFixedLatencyCycles) {
      throw std::overflow_error("DramModel write latency overflow");
    }
    const std::uint64_t write_cycles =
        write_transactions * kDramFixedLatencyCycles;
    WaitForCycles(write_cycles);
    if (state.cache_bypass != 0) {
      WriteBacking(memory_txn.address, linear_payload.data(),
                   linear_payload.size());
    } else {
      for (const DramLineWrite &write : line_writes)
        WriteBacking(write.address, write.data, write.bytes);
    }

    // PNG/readback is a distinct DRAM request even in bypass mode. The wait
    // occurs before sampling backing, so the new payload can only contain data
    // committed by the DRAM model above.
    constexpr std::uint64_t kReadTransactions = 1;
    constexpr std::uint64_t kReadCycles =
        kReadTransactions * kDramFixedLatencyCycles;
    WaitForCycles(kReadCycles);
    std::vector<std::uint8_t> readback =
        ReadBacking(memory_txn.address, framebuffer_bytes);

    // Allocate while all source handles are live. This guarantees a distinct
    // generation-checked payload and lets a failed state commit roll it back
    // without making the persisted source ownership stale.
    const PoolHandle readback_handle = StoreNewArray(pool_, readback);
    if ((HasPoolHandle(state.pbe_framebuffer) &&
         SameHandle(readback_handle, state.pbe_framebuffer)) ||
        (HasPoolHandle(state.slc_writeback_lines) &&
         SameHandle(readback_handle, state.slc_writeback_lines))) {
      pool_.Release(readback_handle);
      throw std::logic_error("DramModel reused a live source handle");
    }
    if (HasPoolHandle(state.pbe_framebuffer) &&
        HasPoolHandle(state.slc_writeback_lines) &&
        SameHandle(state.pbe_framebuffer, state.slc_writeback_lines)) {
      pool_.Release(readback_handle);
      throw std::logic_error("DramModel source handles alias");
    }

    const PoolHandle pbe_source = state.pbe_framebuffer;
    const PoolHandle slc_source = state.slc_writeback_lines;
    state.pbe_framebuffer = {};
    state.slc_writeback_lines = {};
    state.dram_framebuffer = readback_handle;
    state.framebuffer_from_dram = 1;
    AddCounter(state.counters.dram_write_transactions, write_transactions,
               "DRAM write transactions");
    AddCounter(state.counters.dram_read_transactions, kReadTransactions,
               "DRAM read transactions");
    AddCounter(state.counters.dram_write_bytes, write_bytes,
               "DRAM write bytes");
    AddCounter(state.counters.dram_read_bytes, memory_txn.bytes,
               "DRAM read bytes");
    const std::uint64_t framebuffer_dram_cycles = CheckedAdd(
        write_cycles, kReadCycles, "framebuffer DRAM cycles");
    AddCounter(state.counters.dram_cycles, framebuffer_dram_cycles,
               "total DRAM cycles");
    state.counters.framebuffer_dram_readback_bytes = memory_txn.bytes;
    AddCounter(state.counters.renderer_cycles, framebuffer_dram_cycles,
               "renderer cycles");
    state.counters.virtual_gpu_cycles = CheckedAdd(
        CheckedAdd(state.counters.tiler_cycles,
                   state.counters.renderer_cycles,
                   "virtual GPU pipeline cycles"),
        kReferenceUarch.fixed_submission_cycles,
        "virtual GPU submission cycles");
    state.counters.pool_high_water_bytes = pool_.high_water_bytes();
    state.stage = PipelineStage::kFramebufferReady;

    try {
      StorePipelineState(pool_, pipeline_txn.state, state);
    } catch (...) {
      pool_.Release(readback_handle);
      throw;
    }

    // Persisted state no longer names the producer payloads, so ownership can
    // now be retired without exposing a stale handle to downstream cleanup.
    if (HasPoolHandle(pbe_source))
      pool_.Release(pbe_source);
    if (HasPoolHandle(slc_source))
      pool_.Release(slc_source);
    output.write(pipeline_txn);
  }
}

void DramModel::TextureRun() {
  if (texture_input.size() == 0 || texture_output.size() == 0)
    return;
  while (true) {
    const MemoryTxn request = texture_input->read();
    if (request.client != MemoryClient::kTextureCache ||
        request.operation != MemoryOperation::kRead ||
        request.payload_format != MemoryPayloadFormat::kLinearBytes ||
        request.bytes == 0 || HasPoolHandle(request.payload)) {
      throw std::runtime_error("DramModel received an invalid texture read");
    }
    const std::size_t bytes = CheckedSize(request.bytes, "texture read");
    WaitForCycles(kDramFixedLatencyCycles);
    const std::vector<std::uint8_t> data = ReadBacking(request.address, bytes);
    MemoryTxn response = request;
    response.payload = StoreNewArray(pool_, data);
    PipelineState state =
        LoadPipelineState(pool_, request.pipeline.state);
    AddCounter(state.counters.dram_read_transactions, 1,
               "texture read transactions");
    AddCounter(state.counters.dram_read_bytes, request.bytes,
               "texture read bytes");
    AddCounter(state.counters.dram_cycles, kDramFixedLatencyCycles,
               "texture DRAM cycles");
    AddCounter(state.counters.renderer_cycles, kDramFixedLatencyCycles,
               "texture renderer cycles");
    StorePipelineState(pool_, request.pipeline.state, state);
    texture_output->write(response);
  }
}

void DramModel::UploadRun() {
  if (upload_input.size() == 0 || upload_output.size() == 0)
    return;
  while (true) {
    const MemoryTxn request = upload_input->read();
    if (request.client != MemoryClient::kTextureUpload ||
        request.operation != MemoryOperation::kWrite ||
        request.payload_format != MemoryPayloadFormat::kLinearBytes ||
        request.bytes == 0 || !HasPoolHandle(request.payload)) {
      throw std::runtime_error("DramModel received an invalid texture upload");
    }
    const std::vector<std::uint8_t> data =
        LoadArray<std::uint8_t>(pool_, request.payload);
    if (data.size() != request.bytes)
      throw std::runtime_error("DramModel texture upload byte mismatch");
    WaitForCycles(kDramFixedLatencyCycles);
    WriteBacking(request.address, data.data(), data.size());
    MemoryTxn response = request;
    response.payload = {};
    upload_output->write(response);
  }
}

} // namespace pvrgpu::stub
