// SPDX-License-Identifier: MIT
#include "common/tessellation_state.h"
#include "geometry/tessellator.h"
#include "memory/gpu_memory_system.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace pvrgpu::stub;
unsigned checks = 0;
void Check(bool value, const char *reason) {
  ++checks;
  if (!value) throw std::runtime_error(reason);
}

TessellationRequest Request(unsigned sequence, unsigned patch) {
  TessellationRequest request;
  request.domain = static_cast<TessellationDomain>((sequence - 1) % 3);
  request.spacing = static_cast<TessellationSpacing>(((sequence - 1) / 3) % 3);
  request.clockwise = (sequence / 9) & 1;
  request.point_mode = (sequence / 18) & 1;
  for (unsigned i = 0; i < 4; ++i) request.outer[i] = 2.25f + ((sequence + patch + i) % 8);
  for (unsigned i = 0; i < 2; ++i) request.inner[i] = 3.75f + ((sequence + patch + i) % 4);
  if (patch == 1) request.outer[0] = 0;
  return request;
}

PipelineTxn Make(MemoryPool &pool, GpuMemorySystem &memory, unsigned sequence) {
  PipelineState state;
  state.stage = PipelineStage::kVertexShaded;
  state.memory_mode = memory.mode();
  state.counters.vs_invocations = 123;  // Must not be altered by fixed function.
  TessellationState tessellation;
  tessellation.phase = TessellationPhase::kControlComplete;
  const auto request = Request(sequence, 0);
  tessellation.domain = request.domain;
  tessellation.spacing = request.spacing;
  tessellation.clockwise = request.clockwise;
  tessellation.point_mode = request.point_mode;
  tessellation.output_address = UINT64_C(0x400000000) + sequence * kTessellationDrawAddressStride;
  tessellation.domain_address = tessellation.output_address + UINT64_C(0x5000000);
  std::vector<TessellationPatch> patches(3);
  for (unsigned i = 0; i < patches.size(); ++i) {
    patches[i].primitive_id = sequence * 3 + i;
    patches[i].instance_id = sequence / 2;
    patches[i].output_address = tessellation.output_address + i * kTessellationPatchAddressStride;
    const auto factors = Request(sequence, i);
    std::array<float, 6> levels;
    std::copy(std::begin(factors.outer), std::end(factors.outer), levels.begin());
    std::copy(std::begin(factors.inner), std::end(factors.inner), levels.begin() + 4);
    // Fixture emulates completed TCS stores through modeled memory (not
    // HostWrite). The fixed module cannot use a host-side copy of these levels.
    memory.Write(patches[i].output_address, levels.data(), sizeof(levels), MemoryClient::kTessellationControl);
  }
  tessellation.patches = StoreNewArray(pool, patches);
  state.tessellation_state = StoreNewArray(pool, std::vector<TessellationState>{tessellation});
  PipelineTxn transaction;
  transaction.sequence = sequence;
  transaction.state = pool.Allocate(sizeof(PipelineState));
  StorePipelineState(pool, transaction.state, state);
  return transaction;
}

void Verify(MemoryPool &pool, GpuMemorySystem &memory, PipelineTxn transaction) {
  const auto state = LoadPipelineState(pool, transaction.state);
  const auto records = LoadArray<TessellationState>(pool, state.tessellation_state);
  Check(records.size() == 1, "one pool-owned tessellation record");
  const auto &tessellation = records[0];
  Check(tessellation.phase == TessellationPhase::kDomainComplete, "domain phase completion");
  Check(state.stage == PipelineStage::kVertexShaded, "fixed module does not claim TES execution");
  const auto patches = LoadArray<TessellationPatch>(pool, tessellation.patches);
  const auto points = LoadArray<TessellationDomainPoint>(pool, tessellation.domain_points);
  const auto indices = LoadArray<std::uint32_t>(pool, tessellation.domain_indices);
  std::array<TessellationDomainPoint, kTessellationMaxPoints> expected_points;
  std::array<std::uint32_t, kTessellationMaxIndices> expected_indices;
  std::size_t point_start = 0, index_start = 0;
  unsigned primitives = 0;
  for (unsigned i = 0; i < patches.size(); ++i) {
    TessellationResult result;
    const auto status = TessellatePatch(Request(transaction.sequence, i),
        {expected_points.data(), expected_points.size(), expected_indices.data(), expected_indices.size()}, result);
    Check(status == TessellationStatus::kSuccess, "reference fixed function result");
    const auto &patch = patches[i];
    Check(patch.primitive_id == transaction.sequence * 3 + i && patch.instance_id == transaction.sequence / 2,
          "upstream patch and instance identity preserved");
    Check(patch.point_start == point_start && patch.index_start == index_start, "pool slice starts");
    Check(patch.point_count == result.point_count && patch.index_count == result.index_count &&
          patch.primitive_size == result.primitive_size, "per-patch exact generated counts");
    Check(patch.domain_address == tessellation.domain_address + point_start * sizeof(TessellationDomainPoint),
          "per-patch GPU coordinate address");
    if (patch.point_count) {
      Check(std::memcmp(points.data() + point_start, expected_points.data(), result.point_count * sizeof(TessellationDomainPoint)) == 0,
            "pool coordinate bits equal direct fixed algorithm");
      Check(std::memcmp(indices.data() + index_start, expected_indices.data(), result.index_count * sizeof(std::uint32_t)) == 0,
            "pool connectivity indices remain patch-local");
      // This is the same modeled client/address path TES will use, including
      // visibility of dirty domain writes through the cache hierarchy.
      const auto gpu = memory.Read(patch.domain_address, patch.point_count * sizeof(TessellationDomainPoint),
                                   MemoryClient::kTessellationEvaluation);
      Check(gpu.data.size() == patch.point_count * sizeof(TessellationDomainPoint), "TES modeled read size");
      if (std::memcmp(gpu.data.data(), expected_points.data(), gpu.data.size()) != 0) {
        std::cerr << "domain mismatch mode=" << unsigned(memory.mode()) << " sequence=" << transaction.sequence
                  << " patch=" << i << " address=" << patch.domain_address << '\n';
        const auto *expected = reinterpret_cast<const std::uint8_t *>(expected_points.data());
        for (unsigned byte = 0; byte < gpu.data.size(); ++byte)
          if (gpu.data[byte] != expected[byte]) {
            std::cerr << "byte=" << byte << " actual=" << unsigned(gpu.data[byte])
                      << " expected=" << unsigned(expected[byte]) << '\n'; break;
          }
        Check(false, "TES sees exact GPU domain bytes");
      }
      Check(true, "TES sees exact GPU domain bytes");
    } else Check(patch.index_count == 0, "culled patch has no primitive or domain output");
    point_start += result.point_count;
    index_start += result.index_count;
    primitives += result.index_count / result.primitive_size;
  }
  Check(points.size() == point_start && indices.size() == index_start, "bounded pool payload exact sizes");
  Check(state.counters.tessellation_patches == 3, "culled patch still processed once");
  Check(state.counters.tessellation_primitives == primitives, "actual generated primitive counter");
  Check(state.counters.tessellation_level_read_bytes == 72, "all levels really read from GPU");
  Check(state.counters.tessellation_domain_write_bytes == points.size() * sizeof(TessellationDomainPoint),
        "actual GPU domain write bytes");
  Check(state.counters.vs_invocations == 123 && !state.counters.hs_invocations &&
        !state.counters.ds_invocations && !state.counters.gs_invocations,
        "fixed tessellation never fabricates shader invocations");
  for (auto handle : {tessellation.patches, tessellation.domain_points, tessellation.domain_indices,
                      state.tessellation_state, transaction.state}) pool.Release(handle);
}
}

int sc_main(int argc, char **argv) {
  try {
    MemoryPool pool;
    GpuMemorySystem direct(MemoryMode::kDirect), bypass(MemoryMode::kBypass), cache(MemoryMode::kCache);
    sc_core::sc_fifo<PipelineTxn> di("di", 1), dout("dout", 1), bi("bi", 1), bout("bout", 1), ci("ci", 1), cout("cout", 1);
    Tessellator dm("tessellator_direct", pool, &direct), bm("tessellator_bypass", pool, &bypass), cm("tessellator_cache", pool, &cache);
    dm.input(di); dm.output(dout); bm.input(bi); bm.output(bout); cm.input(ci); cm.output(cout);
    const std::array<GpuMemorySystem *, 3> memories{&direct, &bypass, &cache};
    const std::array<sc_core::sc_fifo<PipelineTxn> *, 3> inputs{&di, &bi, &ci}, outputs{&dout, &bout, &cout};
    if (argc == 2) {
      const int invalid = std::atoi(argv[1]);
      Check(invalid >= 1 && invalid <= 7, "invalid test mode");
      auto transaction = Make(pool, direct, 1);
      auto state = LoadPipelineState(pool, transaction.state);
      auto records = LoadArray<TessellationState>(pool, state.tessellation_state);
      auto &tessellation = records[0];
      if (invalid == 1) tessellation.phase = TessellationPhase::kSubmitted;
      if (invalid == 2) tessellation.clockwise = 2;
      if (invalid == 3) tessellation.domain_address = tessellation.output_address;
      if (invalid == 4) {
        auto patches = LoadArray<TessellationPatch>(pool, tessellation.patches);
        patches[0].output_address += kTessellationPatchAddressStride;
        StoreArray(pool, tessellation.patches, patches);
      }
      if (invalid == 5) {
        pool.Release(tessellation.patches);
        tessellation.patches = StoreNewArray(pool, std::vector<TessellationPatch>(kTessellationMaxPatches + 1));
      }
      if (invalid == 6) state.memory_mode = MemoryMode::kBypass;
      if (invalid == 7) tessellation.domain_points = pool.Allocate(0);
      StoreArray(pool, state.tessellation_state, records);
      StorePipelineState(pool, transaction.state, state);
      Check(di.nb_write(transaction), "invalid transaction entered module");
      bool rejected = false;
      try { sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_US)); }
      catch (const std::exception &error) {
        rejected = std::string(error.what()).find("tessellator") != std::string::npos;
      }
      Check(rejected, "invalid tessellator contract rejected explicitly");
      PipelineTxn unexpected;
      Check(!dout.nb_read(unexpected), "rejected work has no successful completion");
      const auto after = LoadPipelineState(pool, transaction.state);
      const auto after_records = LoadArray<TessellationState>(pool, after.tessellation_state);
      Check(after_records[0].phase != TessellationPhase::kDomainComplete,
            "rejected work never claims domain completion");
      Check(!HasPoolHandle(after_records[0].domain_indices), "rejected work has no published indices");
      if (HasPoolHandle(tessellation.domain_points)) pool.Release(tessellation.domain_points);
      pool.Release(tessellation.patches);
      pool.Release(state.tessellation_state);
      pool.Release(transaction.state);
      Check(pool.bytes_in_flight() == 0 && pool.allocations() == pool.releases(), "rejected work scratch ownership balanced");
      std::cout << "Tessellator rejection " << invalid << ": " << checks << " checks passed\n";
      return 0;
    }
    for (unsigned mode = 0; mode < 3; ++mode) {
      for (unsigned sequence = 1; sequence <= 36; sequence += 2) {
        const auto first = Make(pool, *memories[mode], sequence);
        const auto second = Make(pool, *memories[mode], sequence + 1);
        Check(inputs[mode]->nb_write(first), "first depth-one input accepted");
        Check(!inputs[mode]->nb_write(second), "depth-one input applies backpressure");
        sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_US));
        Check(inputs[mode]->nb_write(second), "input resumes after first dequeue");
        sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_US));
        PipelineTxn done;
        Check(outputs[mode]->nb_read(done) && done.sequence == sequence, "full output preserves first completion");
        Verify(pool, *memories[mode], done);
        sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_US));
        Check(outputs[mode]->nb_read(done) && done.sequence == sequence + 1, "output event releases next completion");
        Verify(pool, *memories[mode], done);
      }
      PipelineState pass;
      pass.counters.gs_invocations = 71;
      PipelineTxn transaction;
      transaction.state = pool.Allocate(sizeof(PipelineState));
      StorePipelineState(pool, transaction.state, pass);
      Check(inputs[mode]->nb_write(transaction), "non-tessellated transaction accepted");
      sc_core::sc_start(sc_core::sc_time(1, sc_core::SC_US));
      PipelineTxn done;
      Check(outputs[mode]->nb_read(done), "non-tessellated transaction passed through");
      const auto unchanged = LoadPipelineState(pool, done.state);
      Check(std::memcmp(&pass, &unchanged, sizeof(pass)) == 0, "non-tessellation state byte-identical");
      pool.Release(done.state);
    }
    Check(pool.bytes_in_flight() == 0 && pool.allocations() == pool.releases(), "all tessellator pool ownership balanced");
    std::cout << "Tessellator module: " << checks << " checks, 108 draws / 324 patches passed\n";
  } catch (const std::exception &error) {
    std::cerr << "Tessellator module: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
