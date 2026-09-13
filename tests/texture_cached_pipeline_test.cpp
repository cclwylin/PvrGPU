// End-to-end timing contract for the explicit cached texture route:
// TPU -> TCU -> SLC -> DRAM. A cold nearest sample must pay each layer once;
// repeating the same sample must hit in TCU and pay only TPU + TCU latency.

#include "cache_mmu/texture_cache.h"
#include "common/functional_types.h"
#include "common/glbench_texture_fixture.h"
#include "common/pipeline_state.h"
#include "common/reference_uarch.h"
#include "memory/gpu_memory_system.h"
#include "texture/texture_unit.h"

#include <systemc>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using pvrgpu::stub::CounterTxn;
using pvrgpu::stub::FunctionalCase;
using pvrgpu::stub::GlbenchFillTextureFixture;
using pvrgpu::stub::GpuMemorySystem;
using pvrgpu::stub::LoadArray;
using pvrgpu::stub::LoadPipelineState;
using pvrgpu::stub::MemoryMode;
using pvrgpu::stub::MemoryPool;
using pvrgpu::stub::MemoryTxn;
using pvrgpu::stub::PipelineStage;
using pvrgpu::stub::PipelineState;
using pvrgpu::stub::PipelineTxn;
using pvrgpu::stub::PoolHandle;
using pvrgpu::stub::ReleaseFunctionalPayloads;
using pvrgpu::stub::StoreNewArray;
using pvrgpu::stub::StorePipelineState;
using pvrgpu::stub::TextureFilter;
using pvrgpu::stub::TextureMemoryPath;
using pvrgpu::stub::TextureResource;
using pvrgpu::stub::TextureSampleRequest;
using pvrgpu::stub::TextureSampleResponse;
using pvrgpu::stub::TextureUnit;

void Check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error("cached texture pipeline test failed: " + message);
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

struct PendingSample {
  PoolHandle state;
  PipelineTxn transaction;
};

PendingSample MakeNearestSample(MemoryPool &pool,
                                const GlbenchFillTextureFixture &fixture,
                                std::uint64_t sequence,
                                FunctionalCase functional_case =
                                    FunctionalCase::kFillTexNearest) {
  TextureSampleRequest request;
  request.shader_lane_index = 0;
  request.coordinates[0] = FloatBits(0.125F);
  request.coordinates[1] = FloatBits(0.125F);
  std::copy_n(fixture.fragment_shared.begin(), 4, request.texture_state);
  std::copy_n(fixture.fragment_shared.begin() + 8, 4, request.sampler_state);
  // Request IDs are dense within each batch; PipelineTxn.sequence owns the
  // cross-job identity.
  request.request_id = 0;
  request.coordinate_count = 2;
  request.component_count = 4;
  request.dimension = 2;
  request.normalized = 1;
  request.shader_stage = pvrgpu::stub::ShaderStage::kFragment;

  TextureResource resource = fixture.resource;
  // Persistent texture bytes live in the shared GpuMemorySystem. The resource
  // table owns metadata only when TextureUnit is attached to that memory.
  resource.data = {};

  PipelineState state;
  state.memory_mode = MemoryMode::kCache;
  state.sequence = sequence;
  state.functional_case = functional_case;
  state.stage = PipelineStage::kFragmentTexturePending;
  state.fragment_shader_lane_count = 1;
  state.texture_sample_requests =
      StoreNewArray(pool, std::vector<TextureSampleRequest>{request});
  state.texture_resources =
      StoreNewArray(pool, std::vector<TextureResource>{resource});
  state.sampler_states = StoreNewArray(
      pool, std::vector<pvrgpu::stub::SamplerState>{fixture.sampler});
  state.fragment_shared_registers = StoreNewArray(
      pool, std::vector<std::uint32_t>(fixture.fragment_shared.begin(),
                                       fixture.fragment_shared.end()));
  const PoolHandle state_handle = pool.Allocate(sizeof(PipelineState));
  StorePipelineState(pool, state_handle, state);
  return {state_handle,
          PipelineTxn{state_handle, static_cast<std::uint32_t>(sequence),
                      sequence}};
}

PipelineTxn RunUntilComplete(sc_core::sc_fifo<PipelineTxn> &input,
                             sc_core::sc_fifo<PipelineTxn> &output,
                             const PipelineTxn &request) {
  input.write(request);
  PipelineTxn response;
  for (;;) {
    if (output.nb_read(response))
      return response;
    Check(sc_core::sc_pending_activity(),
          "SystemC went idle before the texture sample completed");
    sc_core::sc_start(sc_core::sc_time_to_pending_activity());
  }
}

void CheckColdCounters(const CounterTxn &counters) {
  Check(counters.texture_requests == 1 && counters.texel_fetches == 1,
        "cold TPU request/fetch count");
  Check(counters.tcu_line_accesses == 1 && counters.tcu_read_accesses == 1 &&
            counters.tcu_hits == 0 && counters.tcu_misses == 1 &&
            counters.tcu_cycles == 1,
        "cold TCU miss counters");
  Check(counters.slc_line_accesses == 1 && counters.slc_read_accesses == 1 &&
            counters.slc_hits == 0 && counters.slc_misses == 1 &&
            counters.slc_cycles == 1,
        "cold SLC miss counters");
  Check(counters.dram_read_transactions == 1 &&
            counters.dram_read_bytes == 128 && counters.dram_cycles == 1 &&
            counters.memory_direct_read_bytes == 0,
        "cold DRAM line-fill counters");
  Check(counters.texture_cycles == 5 && counters.renderer_cycles == 5 &&
            counters.tiler_cycles == 0,
        "cold latency is charged once at TPU, TCU, SLC and DRAM");
}

void CheckWarmCounters(const CounterTxn &counters) {
  Check(counters.texture_requests == 1 && counters.texel_fetches == 1,
        "warm TPU request/fetch count");
  Check(counters.tcu_line_accesses == 1 && counters.tcu_read_accesses == 1 &&
            counters.tcu_hits == 1 && counters.tcu_misses == 0 &&
            counters.tcu_cycles == 1,
        "warm TCU hit counters");
  Check(counters.slc_line_accesses == 0 && counters.slc_read_accesses == 0 &&
            counters.slc_hits == 0 && counters.slc_misses == 0 &&
            counters.slc_cycles == 0 && counters.dram_read_transactions == 0 &&
            counters.dram_read_bytes == 0 && counters.dram_cycles == 0 &&
            counters.memory_direct_read_bytes == 0,
        "warm TCU hit does not access SLC or DRAM");
  Check(counters.texture_cycles == 3 && counters.renderer_cycles == 3 &&
            counters.tiler_cycles == 0,
        "warm latency is charged once at TPU and TCU");
}

} // namespace

int sc_main(int, char **) {
  try {
    MemoryPool pool;
    GpuMemorySystem memory(MemoryMode::kCache);
    const GlbenchFillTextureFixture fixture =
        pvrgpu::stub::MakeGlbenchFillTextureFixture(TextureFilter::kNearest);
    memory.HostWrite(fixture.resource.gpu_address, fixture.texture_bytes.data(),
                     fixture.texture_bytes.size());

    sc_core::sc_fifo<PipelineTxn> tpu_input("tpu_input", 1);
    sc_core::sc_fifo<PipelineTxn> tpu_output("tpu_output", 1);
    sc_core::sc_fifo<PipelineTxn> sample_input("sample_input", 1);
    sc_core::sc_fifo<PipelineTxn> sample_output("sample_output", 1);
    sc_core::sc_fifo<MemoryTxn> tpu_to_tcu("tpu_to_tcu", 1);
    sc_core::sc_fifo<MemoryTxn> tcu_to_tpu("tcu_to_tpu", 1);
    sc_core::sc_fifo<MemoryTxn> tcu_input("tcu_input", 1);
    sc_core::sc_fifo<MemoryTxn> tcu_output("tcu_output", 1);

    TextureUnit tpu("tpu", pool, &memory, false, TextureMemoryPath::kCached);
    pvrgpu::stub::TextureCache tcu("tcu", pool, false, &memory);
    tpu.input(tpu_input);
    tpu.output(tpu_output);
    tpu.sample_input(sample_input);
    tpu.sample_output(sample_output);
    tpu.cache_request(tpu_to_tcu);
    tpu.cache_response(tcu_to_tpu);
    tcu.input(tcu_input);
    tcu.output(tcu_output);
    tcu.sample_input(tpu_to_tcu);
    tcu.sample_output(tcu_to_tpu);

    sc_core::sc_start(sc_core::SC_ZERO_TIME);

    const PendingSample cold = MakeNearestSample(pool, fixture, 1);
    const sc_core::sc_time cold_start = sc_core::sc_time_stamp();
    const PipelineTxn cold_completed =
        RunUntilComplete(sample_input, sample_output, cold.transaction);
    const sc_core::sc_time cold_elapsed = sc_core::sc_time_stamp() - cold_start;
    Check(cold_completed.state.slot == cold.state.slot &&
              cold_completed.state.generation == cold.state.generation &&
              cold_completed.sequence == 1,
          "cold completion identity");
    const PipelineState cold_state = LoadPipelineState(pool, cold.state);
    Check(cold_state.stage == PipelineStage::kTextureSamplesReady,
          "cold sample reached the ready stage");
    const std::vector<TextureSampleResponse> cold_responses =
        LoadArray<TextureSampleResponse>(pool,
                                         cold_state.texture_sample_responses);
    Check(cold_responses.size() == 1, "cold sample produced one TPU response");
    CheckColdCounters(cold_state.counters);
    Check(cold_elapsed == sc_core::sc_time(5, sc_core::SC_NS),
          "cold TPU->TCU->SLC->DRAM elapsed time is exactly 5 ns");

    const PendingSample warm = MakeNearestSample(pool, fixture, 2);
    const sc_core::sc_time warm_start = sc_core::sc_time_stamp();
    const PipelineTxn warm_completed =
        RunUntilComplete(sample_input, sample_output, warm.transaction);
    const sc_core::sc_time warm_elapsed = sc_core::sc_time_stamp() - warm_start;
    Check(warm_completed.state.slot == warm.state.slot &&
              warm_completed.state.generation == warm.state.generation &&
              warm_completed.sequence == 2,
          "warm completion identity");
    const PipelineState warm_state = LoadPipelineState(pool, warm.state);
    const std::vector<TextureSampleResponse> warm_responses =
        LoadArray<TextureSampleResponse>(pool,
                                         warm_state.texture_sample_responses);
    Check(warm_responses.size() == 1 &&
              warm_responses[0].rgba[0] == cold_responses[0].rgba[0] &&
              warm_responses[0].rgba[1] == cold_responses[0].rgba[1] &&
              warm_responses[0].rgba[2] == cold_responses[0].rgba[2] &&
              warm_responses[0].rgba[3] == cold_responses[0].rgba[3],
          "warm TCU hit returns the same texel");
    CheckWarmCounters(warm_state.counters);
    Check(warm_elapsed == sc_core::sc_time(3, sc_core::SC_NS),
          "warm TPU->TCU elapsed time is exactly 3 ns");

    // Four bilinear taps still traverse the TCU one-by-one, but they reuse one
    // 16-byte borrowed result slot and commit one response array. There must
    // not be a MemoryPool allocation for every tap.
    const GlbenchFillTextureFixture bilinear_fixture =
        pvrgpu::stub::MakeGlbenchFillTextureFixture(TextureFilter::kLinear);
    memory.HostWrite(bilinear_fixture.resource.gpu_address,
                     bilinear_fixture.texture_bytes.data(),
                     bilinear_fixture.texture_bytes.size());
    const PendingSample bilinear = MakeNearestSample(
        pool, bilinear_fixture, 3, FunctionalCase::kFillTexBilinear);
    const std::uint64_t bilinear_allocations_before = pool.allocations();
    const PipelineTxn bilinear_completed =
        RunUntilComplete(sample_input, sample_output, bilinear.transaction);
    Check(bilinear_completed.sequence == 3,
          "batched bilinear completion identity");
    const PipelineState bilinear_state =
        LoadPipelineState(pool, bilinear.state);
    Check(bilinear_state.counters.texture_requests == 1 &&
              bilinear_state.counters.texel_fetches == 4,
          "bilinear sample preserved four ordered physical taps");
    Check(pool.allocations() - bilinear_allocations_before == 2,
          "cached batch allocates one scratch and one response, not per tap");

    ReleaseFunctionalPayloads(pool, cold_state);
    pool.Release(cold.state);
    ReleaseFunctionalPayloads(pool, warm_state);
    pool.Release(warm.state);
    ReleaseFunctionalPayloads(pool, bilinear_state);
    pool.Release(bilinear.state);
    Check(pool.bytes_in_flight() == 0 && pool.allocations() == pool.releases(),
          "MemoryPool ownership is balanced");

    std::cout << "texture_cached_pipeline_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "texture_cached_pipeline_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
