// Module：Tiler。
// 縮寫：非縮寫（分塊器）。
// 功能：將任意數量 RasterTriangle 分箱為 row-major 32×32 TileRecord，並
// 以 ordered TilePrimitiveRef 保留 primitive identity/API submission order。
// ClipCull 保留的 zero-area 及 fast-path face-culled setup candidates 仍
// 計入 setup_triangles，但不產生 tile ref；parameter_index 保持原陣列
// 位置。FIFO 只傳 handle，完成採
// event-driven wait。
#include "geometry/tiler.h"

#include "common/functional_types.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

bool BoundingBoxIntersectsTile(const pvrgpu::stub::RasterTriangle &triangle,
                               const pvrgpu::stub::TileRecord &tile) {
  const auto x_bounds =
      std::minmax({triangle.x[0], triangle.x[1], triangle.x[2]});
  const auto y_bounds =
      std::minmax({triangle.y[0], triangle.y[1], triangle.y[2]});
  return x_bounds.second > static_cast<float>(tile.x0) &&
         x_bounds.first < static_cast<float>(tile.x1) &&
         y_bounds.second > static_cast<float>(tile.y0) &&
         y_bounds.first < static_cast<float>(tile.y1);
}

} // namespace

namespace pvrgpu::stub {

Tiler::Tiler(sc_core::sc_module_name name, MemoryPool &pool)
    : sc_module(name), pool_(pool) {
  SC_THREAD(Run);
}

void Tiler::Run() {
  while (true) {
    const PipelineTxn txn = input.read();
    PipelineState state = LoadPipelineState(pool_, txn.state);

    RequireStage(state.stage, PipelineStage::kClipCullComplete, name());
    if (!HasPoolHandle(state.raster_triangles))
      throw std::runtime_error("Tiler received no raster triangles");
    const std::vector<RasterTriangle> triangles =
        LoadArray<RasterTriangle>(pool_, state.raster_triangles);
    if (triangles.size() != state.counters.c_primitives)
      throw std::runtime_error("Tiler clip-primitive count mismatch");
    if (triangles.empty() && !state.raster_state.face_cull.enable)
      throw std::runtime_error("Tiler received an empty setup stream");
    if (triangles.size() > std::numeric_limits<std::uint32_t>::max())
      throw std::overflow_error("Tiler parameter index exceeds uint32_t");

    std::vector<TileRecord> tiles;
    std::vector<TilePrimitiveRef> primitive_refs;
    const std::uint64_t tiles_x = CeilDivide(state.width, kTileWidth);
    const std::uint64_t tiles_y = CeilDivide(state.height, kTileHeight);
    if (tiles_y != 0 &&
        tiles_x > std::numeric_limits<std::size_t>::max() / tiles_y) {
      throw std::overflow_error("Tiler tile-grid size overflow");
    }
    tiles.reserve(static_cast<std::size_t>(tiles_x * tiles_y));
    for (std::uint32_t y0 = 0; y0 < state.height; y0 += kTileHeight) {
      for (std::uint32_t x0 = 0; x0 < state.width; x0 += kTileWidth) {
        TileRecord tile;
        tile.x0 = x0;
        tile.y0 = y0;
        tile.x1 = std::min<std::uint32_t>(x0 + kTileWidth, state.width);
        tile.y1 = std::min<std::uint32_t>(y0 + kTileHeight, state.height);
        tile.first_primitive_ref =
            static_cast<std::uint32_t>(primitive_refs.size());
        for (std::size_t primitive = 0; primitive < triangles.size();
             ++primitive) {
          if (!triangles[primitive].rasterizable)
            continue;
          if (BoundingBoxIntersectsTile(triangles[primitive], tile)) {
            if (primitive_refs.size() >=
                std::numeric_limits<std::uint32_t>::max()) {
              throw std::overflow_error(
                  "Tiler primitive reference count exceeds uint32_t");
            }
            primitive_refs.push_back({static_cast<std::uint32_t>(primitive), 0,
                                      triangles[primitive].key.submit_ordinal});
          }
        }
        tile.primitive_ref_count = static_cast<std::uint32_t>(
            primitive_refs.size() - tile.first_primitive_ref);
        tiles.push_back(tile);
      }
    }

    state.tile_records = StoreNewArray(pool_, tiles);
    state.tile_primitive_refs = StoreNewArray(pool_, primitive_refs);
    state.counters.setup_triangles = triangles.size();
    state.counters.tiles_binned = tiles.size();
    state.stage = PipelineStage::kTiled;

    const std::uint64_t cycles =
        kReferenceUarch.tiler_base_cycles +
        CeilDivide(state.counters.setup_triangles,
                   kReferenceUarch.tiler_triangles_per_batch);
    state.counters.tiler_bin_cycles = cycles;
    state.counters.tiler_cycles += cycles;
    WaitForCycles(cycles);
    StorePipelineState(pool_, txn.state, state);
    output.write(txn);
  }
}

} // namespace pvrgpu::stub
