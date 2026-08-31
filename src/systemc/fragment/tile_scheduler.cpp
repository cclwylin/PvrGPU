// TileScheduler（tile 排程器）：把 parameter buffer 產生的 tile 工作送入
// fragment phase。 名稱不是縮寫；本模型以 scheduled tile 數代表 renderer
// 可執行的 tile 工作集合。Solid-color cases 在 fragment PCO decode 後
// 讀取 TileRecord 與 ordered TilePrimitiveRef；face-culled case 允許非空 tile
// grid 配空 reference stream，仍記錄 reference uArch 可執行的 tile 工作集合，
// 並推進至 tiles-scheduled stage。FIFO（First-In, First-Out）transaction 只
// 攜帶 MemoryPool handle；tile bulk data 留在 pool，服務時間採事件驅動。
#include "fragment/tile_scheduler.h"

#include "common/functional_types.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace pvrgpu::stub {

TileScheduler::TileScheduler(sc_core::sc_module_name name, MemoryPool &pool,
                             GpuMemorySystem *memory)
    : sc_module(name), pool_(pool), memory_(memory) {
  SC_THREAD(Run);
}

void TileScheduler::Run() {
  while (true) {
    const PipelineTxn txn = input.read();
    PipelineState state = LoadPipelineState(pool_, txn.state);
    RequireStage(state.stage, PipelineStage::kFragmentDecoded, "TileScheduler");
    if (memory_ && state.memory_mode != memory_->mode())
      throw std::runtime_error("TileScheduler memory mode mismatch");
    if (!IsRasterFunctionalCase(state.functional_case))
      throw std::runtime_error("TileScheduler received unsupported case");

    const std::vector<TileRecord> tiles =
        LoadArray<TileRecord>(pool_, state.tile_records);
    const std::vector<TilePrimitiveRef> primitive_refs =
        LoadArray<TilePrimitiveRef>(pool_, state.tile_primitive_refs);
    if (tiles.empty())
      throw std::runtime_error("TileScheduler received an empty tile list");
    MemoryAccessStats memory_stats;
    if (primitive_refs.empty()) {
      std::vector<ParameterTriangle> parameters;
      if (memory_) {
        auto read = ReadMemoryArray<ParameterTriangle>(
            *memory_, state.parameter_triangles_gpu_address,
            state.parameter_triangles_bytes, MemoryClient::kParameterRead);
        parameters = std::move(read.values);
        memory_stats += read.stats;
      } else {
        parameters =
            LoadArray<ParameterTriangle>(pool_, state.parameter_triangles);
      }
      const bool all_non_rasterizable =
          std::all_of(parameters.begin(), parameters.end(),
                      [](const ParameterTriangle &triangle) {
                        return triangle.rasterizable == 0;
                      });
      if (!state.raster_state.face_cull.enable ||
          parameters.size() != state.counters.c_primitives ||
          !all_non_rasterizable) {
        throw std::runtime_error(
            "TileScheduler empty refs lack face-cull invariants");
      }
    }
    state.scheduled_tiles = static_cast<std::uint32_t>(tiles.size());
    state.counters.tiles_scheduled = tiles.size();
    const std::uint64_t tile_count = tiles.size();
    const std::uint64_t functional_cycles =
        kReferenceUarch.scheduler_base_cycles +
        CeilDivide(tile_count, kReferenceUarch.scheduler_tiles_per_batch);
    ApplyMemoryAccessStats(state.counters, memory_stats);
    const std::uint64_t cycles =
        functional_cycles + MemoryAccessDelayCycles(memory_stats);
    state.counters.tile_scheduler_cycles = cycles;
    state.counters.renderer_cycles += cycles;
    state.stage = PipelineStage::kTilesScheduled;

    WaitForCycles(cycles);
    StorePipelineState(pool_, txn.state, state);
    output.write(txn);
  }
}

} // namespace pvrgpu::stub
