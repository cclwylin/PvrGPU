// SPDX-License-Identifier: MIT
// USC task stream helpers (docs/USC_TASK_STREAM_PHASE1.md): issue grouping,
// stream timing, the shared SMP request conversion and partial-render state.
#include "shader/usc_task_stream.h"

#include "common/pipeline_state.h"
#include "memory_pool.h"

#include <cstdint>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace pvrgpu::stub;
std::uint64_t checks = 0;

void Check(bool condition, const std::string &reason) {
  ++checks;
  if (!condition) throw std::runtime_error(reason);
}

void ExpectThrow(const std::function<void()> &operation, const std::string &reason) {
  bool threw = false;
  try { operation(); } catch (const std::exception &) { threw = true; }
  Check(threw, reason);
}

void TestIssuePlanAndTiming() {
  Check(UscIssuePlan::ForLanes(0).groups == 0, "zero lanes issue nothing");
  Check(UscIssuePlan::ForLanes(1).groups == 1, "one lane is one group");
  Check(UscIssuePlan::ForLanes(4).groups == 1, "four lanes share a group");
  Check(UscIssuePlan::ForLanes(5).groups == 2, "a fifth lane opens a group");
  UscStreamWork work;
  work.accepted_inputs = 1; work.invocations = 2;
  work.emitted_vertices = 10; work.exported_primitives = 8;
  // accept 1T, invocation 2T, emit 1T, export 2T.
  Check(UscStreamCycles(work, kGeometryStreamRates) == 1 + 4 + 10 + 16,
        "geometry stream rates");
  Check(GeometryStreamCapacity() == kGeometryStreamCapacityPrimitives,
        "default geometry stream capacity");
}

void TestTextureRequest() {
  PcoTextureRequest issued;
  issued.coordinates = {1, 2, 3};
  issued.spatial_offsets = {-1, 0, 1};
  issued.texture_state = {4, 5, 6, 7};
  issued.sampler_state = {8, 9, 10, 11};
  issued.texture_address_lo = 12; issued.texture_address_hi = 13;
  issued.coordinate_count = 3; issued.component_count = 4;
  issued.descriptor_set = 2; issued.dimension = 1; issued.normalized = 1;
  issued.fcnorm = 0; issued.sample_index = 3; issued.sample_index_present = 1;
  issued.data_request = 1; issued.explicit_lod = 14; issued.explicit_lod_present = 1;
  issued.shadow_reference = 15; issued.shadow_compare = 1;
  const auto vertex = MakeUscTextureRequest(UscStage::kVertex, issued);
  Check(vertex.shader_stage == ShaderStage::kVertex && vertex.coordinates[2] == 3 &&
            vertex.spatial_offsets[0] == -1 && vertex.texture_state[3] == 7 &&
            vertex.sampler_state[0] == 8 && vertex.texture_address_hi == 13 &&
            vertex.descriptor_set == 2 && vertex.fcnorm == 0 &&
            vertex.sample_index == 3 && vertex.explicit_lod == 14 &&
            vertex.shadow_reference == 15 && vertex.shadow_compare == 1 &&
            vertex.data_request == 1 && vertex.request_id == 0 &&
            vertex.shader_lane_index == 0,
        "vertex request copies every PCO operand and leaves identity to the caller");
  PcoTextureRequest biased = issued;
  biased.lod_bias_present = 1; biased.lod_bias = 16;
  ExpectThrow([&] { (void)MakeUscTextureRequest(UscStage::kVertex, biased); },
              "vertex SMP rejects shader LOD bias");
  ExpectThrow([&] { (void)MakeUscTextureRequest(UscStage::kGeometry, biased); },
              "geometry SMP rejects shader LOD bias");
  PcoTextureRequest gather = issued;
  gather.gather = 1;
  ExpectThrow([&] { (void)MakeUscTextureRequest(UscStage::kGeometry, gather); },
              "geometry SMP rejects raw gather");
  const auto fragment = MakeUscTextureRequest(UscStage::kFragment, biased);
  Check(fragment.shader_stage == ShaderStage::kFragment && fragment.lod_bias == 16 &&
            fragment.lod_bias_present == 1,
        "fragment SMP keeps shader LOD bias");
  Check(MakeUscTextureRequest(UscStage::kFragment, gather).gather == 1,
        "fragment SMP keeps raw gather");
}

void TestCounterMerge() {
  CounterTxn total;
  total.frame = total.functional_frame = 1; total.drawlists = 1;
  CounterTxn batch = total;
  batch.c_primitives = 10; batch.pbe_pixels_written = 64;
  batch.pool_high_water_bytes = 100; batch.pixel_data_master_transactions = 1;
  batch.pixel_data_master_bytes = 256; batch.framebuffer_dram_readback_bytes = 256;
  batch.tiler_cycles = 3; batch.renderer_cycles = 4; batch.virtual_gpu_cycles = 99;
  AccumulatePartialRenderCounters(total, batch);
  batch.pool_high_water_bytes = 50;
  AccumulatePartialRenderCounters(total, batch);
  Check(total.c_primitives == 20 && total.pbe_pixels_written == 128, "counts add");
  Check(total.pool_high_water_bytes == 100, "pool gauge keeps its maximum");
  Check(total.pixel_data_master_transactions == 1 && total.pixel_data_master_bytes == 256 &&
            total.framebuffer_dram_readback_bytes == 256,
        "final attachment commit is the last render's");
  Check(total.drawlists == 1 && total.frame == 1, "identities are kept");
  FinalizePartialRenderCounters(total);
  Check(total.virtual_gpu_cycles ==
            6 + 8 + kReferenceUarch.fixed_submission_cycles,
        "virtual GPU cycles are recomputed once per submission");
  CounterTxn other = batch;
  other.frame = 2;
  ExpectThrow([&] { AccumulatePartialRenderCounters(total, other); },
              "renders of different frames never merge");
}

void TestPartialRenderChain() {
  MemoryPool pool;
  {
    PipelineState state;
    state.width = state.height = 2;
    state.render_target_count = 1;
    state.counters.frame = state.counters.functional_frame = 1;
    state.counters.drawlists = 1;
    state.counters.c_primitives = 7;
    state.vertex_texture_request_count = 3;
    state.drawlist_stats = StoreNewArray(pool, std::vector<DrawListStats>(1));
    auto drawlists = LoadArray<DrawListStats>(pool, state.drawlist_stats);
    drawlists[0].fragment.program_recorded = drawlists[0].fragment.executions_recorded = 1;
    drawlists[0].fragment.invocations = 4;
    drawlists[0].fragment.executed_alu_instructions = 40;
    StoreArray(pool, state.drawlist_stats, drawlists);
    const PoolHandle original_load = StoreNewArray(pool, std::vector<std::uint8_t>(16, 1));
    state.color_attachment_load = original_load;
    state.color_attachment_load_bytes = 16;
    state.color_attachment_load_enable = 1;
    state.attachment_clears = StoreNewArray(pool, std::vector<AttachmentClearRect>(1));
    state.capture_depth_attachment = 1;
    state.depth_attachment_format = 1;

    BeginPartialRender(pool, state);
    ExpectThrow([&] { BeginPartialRender(pool, state); }, "a draw begins partial rendering once");
    // Render 0 finished downstream.
    state.stage = PipelineStage::kFramebufferReady;
    state.framebuffer_from_dram = 1;
    state.dram_framebuffer = StoreNewArray(pool, std::vector<std::uint8_t>(16, 2));
    state.depth_attachment = StoreNewArray(pool, std::vector<std::uint8_t>(16, 3));
    state.depth_attachment_bytes = 16;
    state.depth_attachment_ready = 1;
    state.raster_triangles = StoreNewArray(pool, std::vector<std::uint8_t>(8, 0));
    state.fragment_instructions = StoreNewArray(pool, std::vector<std::uint8_t>(8, 0));
    ChainPartialRender(pool, state);
    Check(state.stage == PipelineStage::kVertexShaded && state.partial_render.batch_index == 1,
          "the chained render restarts the raster stages");
    Check(LoadArray<std::uint8_t>(pool, state.color_attachment_load) ==
              std::vector<std::uint8_t>(16, 2) &&
              state.color_attachment_load_enable == 1,
          "committed colour becomes the next render's LOAD");
    Check(LoadArray<std::uint8_t>(pool, state.depth_attachment_load) ==
              std::vector<std::uint8_t>(16, 3) &&
              state.depth_attachment_load_enable == 1 && state.depth_attachment_ready == 0 &&
              state.depth_attachment_bytes == 0 && !HasPoolHandle(state.depth_attachment),
          "committed depth becomes the next render's LOAD");
    Check(!HasPoolHandle(state.attachment_clears) && !HasPoolHandle(state.dram_framebuffer) &&
              !HasPoolHandle(state.raster_triangles) &&
              !HasPoolHandle(state.fragment_instructions),
          "clears and raster intermediates belong to the finished render");
    Check(state.counters.c_primitives == 0 && state.counters.drawlists == 1 &&
              state.vertex_texture_request_count == 0,
          "counters and texture traffic restart per render");
    Check(LoadArray<DrawListStats>(pool, state.drawlist_stats)[0].fragment.program_recorded == 0,
          "fragment program statistics are decoded again");

    // Render 1 is the draw's last.
    state.counters.c_primitives = 5;
    drawlists = LoadArray<DrawListStats>(pool, state.drawlist_stats);
    drawlists[0].fragment.program_recorded = drawlists[0].fragment.executions_recorded = 1;
    drawlists[0].fragment.invocations = 2;
    drawlists[0].fragment.executed_alu_instructions = 20;
    StoreArray(pool, state.drawlist_stats, drawlists);
    state.partial_render.last = 1;
    FinishPartialRender(pool, state);
    Check(state.counters.c_primitives == 12, "the last render folds every render in");
    drawlists = LoadArray<DrawListStats>(pool, state.drawlist_stats);
    Check(drawlists[0].fragment.invocations == 6 &&
              drawlists[0].fragment.executed_alu_instructions == 60,
          "fragment executions add across renders");
    Check(state.color_attachment_load.slot == original_load.slot &&
              state.color_attachment_load.generation == original_load.generation &&
              state.color_attachment_load_bytes == 16 && !HasPoolHandle(state.depth_attachment_load) &&
              state.depth_attachment_load_enable == 0 && !state.partial_render.active,
          "the draw's own LOAD evidence is restored");
    Check(HasPoolHandle(state.attachment_clears), "the draw's inherited clears are restored");
    pool.Release(state.color_attachment_load);
    pool.Release(state.attachment_clears);
    pool.Release(state.drawlist_stats);
  }
  Check(pool.bytes_in_flight() == 0, "partial render payloads are balanced");
}

}  // namespace

int main() {
  try {
    TestIssuePlanAndTiming();
    TestTextureRequest();
    TestCounterMerge();
    TestPartialRenderChain();
  } catch (const std::exception &error) {
    std::fprintf(stderr, "usc_task_stream_test: FAIL: %s\n", error.what());
    return 1;
  }
  std::printf("usc_task_stream_test: PASS (%llu checks)\n",
              static_cast<unsigned long long>(checks));
  return 0;
}
