// Reproducible FIFO/event smoke test for the implemented-idle MCU (Mixed
// Cache Unit), TCU (Texture Cache Unit), and USC-L2 (Unified Shading Cluster
// Level-2) SystemC cache controllers. Each controller must consume one
// MemoryTxn, schedule exactly one timed completion, update a cold-miss cache
// delta, and forward the unchanged transaction without a clock.

#include "cache_mmu/mixed_cache.h"
#include "cache_mmu/texture_cache.h"
#include "cache_mmu/usc_l2_cache.h"

#include <systemc>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using pvrgpu::stub::CacheStats;
using pvrgpu::stub::MemoryClient;
using pvrgpu::stub::MemoryOperation;
using pvrgpu::stub::MemoryPayloadFormat;
using pvrgpu::stub::MemoryPool;
using pvrgpu::stub::MemoryTxn;

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

    pvrgpu::stub::MixedCache mcu("mcu", pool);
    pvrgpu::stub::TextureCache tcu("tcu", pool);
    pvrgpu::stub::UscL2Cache usc_l2("usc_l2", pool);
    mcu.input(mcu_in);
    mcu.output(mcu_out);
    tcu.input(tcu_in);
    tcu.output(tcu_out);
    usc_l2.input(usc_in);
    usc_l2.output(usc_out);

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

    pool.Release(payload);
    Check(pool.bytes_in_flight() == 0 && pool.allocations() == pool.releases(),
          "MemoryPool balanced");
    std::cout << "cache_controller_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "cache_controller_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
