// Reproducible FIFO/event smoke test for the generic MCU (Mixed Cache Unit),
// TCU (Texture Cache Unit), and USC-L2 controllers, plus the TCU's active
// sampled-texture path into the shared SLC/DRAM hierarchy.

#include "cache_mmu/mixed_cache.h"
#include "cache_mmu/texture_cache.h"
#include "cache_mmu/usc_l2_cache.h"

#include <systemc>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using pvrgpu::stub::CacheStats;
using pvrgpu::stub::GpuMemorySystem;
using pvrgpu::stub::LoadPipelineState;
using pvrgpu::stub::MemoryClient;
using pvrgpu::stub::MemoryMode;
using pvrgpu::stub::MemoryOperation;
using pvrgpu::stub::MemoryPayloadFormat;
using pvrgpu::stub::MemoryPool;
using pvrgpu::stub::MemoryTxn;
using pvrgpu::stub::PipelineStage;
using pvrgpu::stub::PipelineState;
using pvrgpu::stub::StorePipelineState;

void Check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error("cache controller test failed: " + message);
}

bool SameTxn(const MemoryTxn &left, const MemoryTxn &right) {
  return left.pipeline.state.slot == right.pipeline.state.slot &&
         left.pipeline.state.generation == right.pipeline.state.generation &&
         left.pipeline.frame == right.pipeline.frame &&
         left.pipeline.sequence == right.pipeline.sequence &&
         left.payload.slot == right.payload.slot &&
         left.payload.generation == right.payload.generation &&
         left.address == right.address && left.bytes == right.bytes &&
         left.operation == right.operation && left.client == right.client &&
         left.payload_format == right.payload_format;
}

void CheckColdWrite(const CacheStats &stats, const char *name) {
  Check(stats.line_accesses == 1, std::string(name) + " line access");
  Check(stats.write_accesses == 1, std::string(name) + " write access");
  Check(stats.hits == 0 && stats.misses == 1,
        std::string(name) + " cold miss");
  Check(stats.bypassed == 0, std::string(name) + " default bypass off");
}

}  // namespace

int sc_main(int, char **) {
  try {
    MemoryPool pool;
    const auto payload = pool.Allocate(64);
    std::fill(pool.Write(payload).begin(), pool.Write(payload).end(), 0x5a);

    MemoryTxn mcu_request;
    mcu_request.payload = payload;
    mcu_request.address = 0x20000000ULL;
    mcu_request.bytes = 64;
    mcu_request.operation = MemoryOperation::kWrite;
    mcu_request.client = MemoryClient::kMixedCache;
    mcu_request.payload_format = MemoryPayloadFormat::kLinearBytes;
    MemoryTxn tcu_request = mcu_request;
    tcu_request.client = MemoryClient::kTextureCache;
    MemoryTxn usc_request = mcu_request;
    usc_request.client = MemoryClient::kUscL2;

    sc_core::sc_fifo<MemoryTxn> mcu_in("mcu_in", 1);
    sc_core::sc_fifo<MemoryTxn> mcu_out("mcu_out", 1);
    sc_core::sc_fifo<MemoryTxn> tcu_in("tcu_in", 1);
    sc_core::sc_fifo<MemoryTxn> tcu_out("tcu_out", 1);
    sc_core::sc_fifo<MemoryTxn> usc_in("usc_in", 1);
    sc_core::sc_fifo<MemoryTxn> usc_out("usc_out", 1);

    // Active texture-read path: two 64-byte TCU lines share one 128-byte SLC
    // line. This separately verifies the datapath used only when
    // TextureMemoryPath::kCached is selected by the TPU.
    constexpr std::uint64_t kTextureBase = UINT64_C(0x30000000);
    MemoryPool sample_pool;
    GpuMemorySystem sample_memory(MemoryMode::kCache);
    std::array<std::uint8_t, 256> texture_bytes{};
    for (std::size_t index = 0; index < texture_bytes.size(); ++index)
      texture_bytes[index] = static_cast<std::uint8_t>(index * 13U + 7U);
    sample_memory.HostWrite(kTextureBase, texture_bytes.data(),
                            texture_bytes.size());
    PipelineState sample_state;
    sample_state.memory_mode = MemoryMode::kCache;
    sample_state.stage = PipelineStage::kFragmentTexturePending;
    const auto sample_state_handle = sample_pool.Allocate(sizeof(sample_state));
    StorePipelineState(sample_pool, sample_state_handle, sample_state);

    sc_core::sc_fifo<MemoryTxn> active_tcu_generic_in(
        "active_tcu_generic_in", 1);
    sc_core::sc_fifo<MemoryTxn> active_tcu_generic_out(
        "active_tcu_generic_out", 1);
    sc_core::sc_fifo<MemoryTxn> active_tcu_in("active_tcu_in", 1);
    sc_core::sc_fifo<MemoryTxn> active_tcu_out("active_tcu_out", 1);

    pvrgpu::stub::MixedCache mcu("mcu", pool);
    pvrgpu::stub::TextureCache tcu("tcu", pool);
    pvrgpu::stub::UscL2Cache usc_l2("usc_l2", pool);
    pvrgpu::stub::TextureCache active_tcu("active_tcu", sample_pool, false,
                                          &sample_memory);
    mcu.input(mcu_in);
    mcu.output(mcu_out);
    tcu.input(tcu_in);
    tcu.output(tcu_out);
    usc_l2.input(usc_in);
    usc_l2.output(usc_out);
    active_tcu.input(active_tcu_generic_in);
    active_tcu.output(active_tcu_generic_out);
    active_tcu.sample_input(active_tcu_in);
    active_tcu.sample_output(active_tcu_out);

    mcu_in.write(mcu_request);
    tcu_in.write(tcu_request);
    usc_in.write(usc_request);
    sc_core::sc_start(sc_core::sc_time(1, sc_core::SC_NS));
    // A duration-limited sc_start may pause immediately before runnable
    // processes at the exact end-time boundary. Run only the pending delta
    // notifications; simulation time must remain 1 ns.
    sc_core::sc_start(sc_core::SC_ZERO_TIME);

    MemoryTxn forwarded;
    Check(mcu_out.nb_read(forwarded) && SameTxn(forwarded, mcu_request),
          "MCU forwarded MemoryTxn");
    Check(tcu_out.nb_read(forwarded) && SameTxn(forwarded, tcu_request),
          "TCU forwarded MemoryTxn");
    Check(usc_out.nb_read(forwarded) && SameTxn(forwarded, usc_request),
          "USC-L2 forwarded MemoryTxn");
    Check(sc_core::sc_time_stamp() == sc_core::sc_time(1, sc_core::SC_NS),
          "event-driven completion time");
    CheckColdWrite(mcu.last_delta(), "MCU");
    CheckColdWrite(tcu.last_delta(), "TCU");
    CheckColdWrite(usc_l2.last_delta(), "USC-L2");

    MemoryTxn sample_request;
    sample_request.pipeline.state = sample_state_handle;
    sample_request.pipeline.frame = 4;
    sample_request.pipeline.sequence = 9;
    sample_request.address = kTextureBase + 60;
    sample_request.bytes = 16;
    sample_request.request_id = 3;
    sample_request.operation = MemoryOperation::kRead;
    sample_request.client = MemoryClient::kTextureCache;
    sample_request.payload_format = MemoryPayloadFormat::kLinearBytes;

    const auto run_sample = [&](const std::array<std::uint8_t, 16> &expected,
                                const char *label) {
      active_tcu_in.write(sample_request);
      sc_core::sc_start(sc_core::sc_time(10, sc_core::SC_NS));
      MemoryTxn response;
      Check(active_tcu_out.nb_read(response),
            std::string(label) + " response available");
      Check(response.pipeline.state.slot == sample_state_handle.slot &&
                response.pipeline.state.generation ==
                    sample_state_handle.generation &&
                response.pipeline.frame == sample_request.pipeline.frame &&
                response.pipeline.sequence ==
                    sample_request.pipeline.sequence &&
                response.request_id == sample_request.request_id &&
                response.address == sample_request.address &&
                response.bytes == sample_request.bytes &&
                response.client == sample_request.client &&
                response.operation == sample_request.operation &&
                response.payload_format == sample_request.payload_format,
            std::string(label) + " response identity");
      const auto &payload = sample_pool.Read(response.payload);
      Check(payload.size() == expected.size() &&
                std::equal(expected.begin(), expected.end(), payload.begin()),
            std::string(label) + " response bytes");
      sample_pool.Release(response.payload);
    };

    std::array<std::uint8_t, 16> initial_cross_line{};
    std::copy_n(texture_bytes.begin() + 60, initial_cross_line.size(),
                initial_cross_line.begin());
    run_sample(initial_cross_line, "cold cross-line TCU read");
    run_sample(initial_cross_line, "warm cross-line TCU read");

    PipelineState sampled = LoadPipelineState(sample_pool, sample_state_handle);
    Check(sampled.counters.tcu_line_accesses == 4 &&
              sampled.counters.tcu_read_accesses == 4 &&
              sampled.counters.tcu_hits == 2 &&
              sampled.counters.tcu_misses == 2 &&
              sampled.counters.tcu_cycles == 4,
          "cross-line cold/warm reads account two TCU lines each");
    Check(sampled.counters.slc_line_accesses == 2 &&
              sampled.counters.slc_read_accesses == 2 &&
              sampled.counters.slc_hits == 1 &&
              sampled.counters.slc_misses == 1 &&
              sampled.counters.slc_cycles == 2 &&
              sampled.counters.dram_read_transactions == 1 &&
              sampled.counters.dram_read_bytes == 128 &&
              sampled.counters.dram_cycles == 1 &&
              sampled.counters.memory_direct_read_bytes == 0,
          "TCU misses enter the shared SLC and coalesce on its 128-byte line");
    Check(sampled.counters.texture_cycles == 7 &&
              sampled.counters.renderer_cycles == 7 &&
              sampled.counters.tiler_cycles == 0,
          "TCU, SLC, and DRAM latency is charged exactly once");

    std::array<std::uint8_t, 16> replacement{};
    for (std::size_t index = 0; index < replacement.size(); ++index)
      replacement[index] = static_cast<std::uint8_t>(0xe0U + index);
    sample_memory.HostWrite(kTextureBase + 64, replacement.data(),
                            replacement.size());
    std::array<std::uint8_t, 16> refreshed_cross_line{};
    std::copy_n(texture_bytes.begin() + 60, 4,
                refreshed_cross_line.begin());
    std::copy_n(replacement.begin(), 12, refreshed_cross_line.begin() + 4);
    run_sample(refreshed_cross_line, "host-write coherent TCU read");

    sampled = LoadPipelineState(sample_pool, sample_state_handle);
    Check(sampled.counters.tcu_line_accesses == 6 &&
              sampled.counters.tcu_read_accesses == 6 &&
              sampled.counters.tcu_hits == 3 &&
              sampled.counters.tcu_misses == 3 &&
              sampled.counters.tcu_cycles == 6,
          "host write invalidates only the intersecting TCU line");
    Check(sampled.counters.slc_line_accesses == 3 &&
              sampled.counters.slc_read_accesses == 3 &&
              sampled.counters.slc_hits == 1 &&
              sampled.counters.slc_misses == 2 &&
              sampled.counters.slc_cycles == 3 &&
              sampled.counters.dram_read_transactions == 2 &&
              sampled.counters.dram_read_bytes == 256 &&
              sampled.counters.dram_cycles == 2 &&
              sampled.counters.texture_cycles == 11 &&
              sampled.counters.renderer_cycles == 11,
          "host write drops stale TCU/SLC data before the next refill");

    std::array<std::uint8_t, 16> gpu_replacement{};
    for (std::size_t index = 0; index < gpu_replacement.size(); ++index)
      gpu_replacement[index] = static_cast<std::uint8_t>(0xa0U + index);
    const auto gpu_write = sample_memory.Write(
        kTextureBase + 60, gpu_replacement.data(), gpu_replacement.size(),
        MemoryClient::kFramebuffer);
    Check(gpu_write.slc.line_accesses == 2 &&
              gpu_write.slc.read_accesses == 1 &&
              gpu_write.slc.write_accesses == 1 &&
              gpu_write.slc.hits == 2 &&
              gpu_write.dram_write_transactions == 0,
          "partial GPU replacement reads and updates the resident SLC line");
    run_sample(gpu_replacement, "GPU-write coherent TCU read");

    sampled = LoadPipelineState(sample_pool, sample_state_handle);
    Check(sampled.counters.tcu_line_accesses == 8 &&
              sampled.counters.tcu_read_accesses == 8 &&
              sampled.counters.tcu_hits == 3 &&
              sampled.counters.tcu_misses == 5 &&
              sampled.counters.tcu_cycles == 8,
          "GPU write invalidates every intersecting TCU line");
    Check(sampled.counters.slc_line_accesses == 5 &&
              sampled.counters.slc_read_accesses == 5 &&
              sampled.counters.slc_hits == 3 &&
              sampled.counters.slc_misses == 2 &&
              sampled.counters.dram_read_transactions == 2 &&
              sampled.counters.texture_cycles == 15 &&
              sampled.counters.renderer_cycles == 15,
          "TCU refills observe the GPU-written SLC line without stale DRAM");

    PipelineState width_state;
    width_state.memory_mode = MemoryMode::kCache;
    width_state.stage = PipelineStage::kFragmentTexturePending;
    const auto width_state_handle =
        sample_pool.Allocate(sizeof(width_state));
    StorePipelineState(sample_pool, width_state_handle, width_state);
    constexpr std::array<std::size_t, 5> kSupportedReadBytes = {
        1, 2, 4, 8, 16};
    constexpr std::array<std::size_t, 5> kSupportedReadOffsets = {
        128, 130, 132, 136, 144};
    for (std::size_t index = 0; index < kSupportedReadBytes.size(); ++index) {
      MemoryTxn width_request = sample_request;
      width_request.pipeline.state = width_state_handle;
      width_request.address = kTextureBase + kSupportedReadOffsets[index];
      width_request.bytes = kSupportedReadBytes[index];
      width_request.request_id = 10 + index;
      active_tcu_in.write(width_request);
      sc_core::sc_start(sc_core::sc_time(10, sc_core::SC_NS));
      MemoryTxn response;
      Check(active_tcu_out.nb_read(response),
            "supported-width TCU response available");
      const auto &response_bytes = sample_pool.Read(response.payload);
      Check(response.bytes == kSupportedReadBytes[index] &&
                response_bytes.size() == kSupportedReadBytes[index] &&
                std::equal(response_bytes.begin(), response_bytes.end(),
                           texture_bytes.begin() +
                               kSupportedReadOffsets[index]),
            "TCU preserves a supported 1/2/4/8/16-byte read");
      sample_pool.Release(response.payload);
    }
    const PipelineState widths =
        LoadPipelineState(sample_pool, width_state_handle);
    Check(widths.counters.tcu_line_accesses == 5 &&
              widths.counters.tcu_read_accesses == 5 &&
              widths.counters.tcu_hits == 4 &&
              widths.counters.tcu_misses == 1 &&
              widths.counters.tcu_cycles == 5 &&
              widths.counters.slc_read_accesses == 1 &&
              widths.counters.slc_misses == 1 &&
              widths.counters.dram_read_transactions == 1 &&
              widths.counters.dram_read_bytes == 128 &&
              widths.counters.texture_cycles == 7 &&
              widths.counters.renderer_cycles == 7,
          "supported widths share one warm TCU line and one cold SLC fill");

    pool.Release(payload);
    Check(pool.bytes_in_flight() == 0 && pool.allocations() == pool.releases(),
          "MemoryPool balanced");
    sample_pool.Release(sample_state_handle);
    sample_pool.Release(width_state_handle);
    Check(sample_pool.bytes_in_flight() == 0 &&
              sample_pool.allocations() == sample_pool.releases(),
          "active TCU MemoryPool balanced");
    std::cout << "cache_controller_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "cache_controller_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
