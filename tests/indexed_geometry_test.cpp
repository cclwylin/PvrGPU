/*
 * Indexed Triangle.Setup geometry-path test.
 * 中文：以 pinned GLBench 128x128 lattice/uint16 index stream 驗證
 * VDM→VertexFetch→vertex PCO/USC→ClipCull→Tiler→ParameterBuffer 的完整
 * event-driven hardware route。Counter 必須由 cache/clipping 資料流產生，
 * 不使用 fixture-name shortcut 或硬編結果。
 */
#include "common/functional_types.h"
#include "common/pipeline_state.h"
#include "geometry/clip_cull.h"
#include "geometry/parameter_buffer.h"
#include "geometry/tiler.h"
#include "geometry/vdm.h"
#include "geometry/vertex_fetch.h"
#include "shader/pco_decoder.h"
#include "shader/pco_iss.h"
#include "shader/usc_cluster.h"
#include "shader/usc_slot.h"

#include <systemc>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace pvrgpu::stub;

constexpr std::uint32_t kMeshWidth = 128;
constexpr std::uint32_t kMeshHeight = 128;
constexpr std::uint32_t kSwathHeight = 4;
constexpr std::uint32_t kIndexCount = 98304;
constexpr std::uint32_t kPrimitiveCount = 32768;
constexpr std::uint32_t kExpectedVertexInvocations = 21144;
constexpr std::uint32_t kExpectedVertexFetchCycles = 87;
constexpr std::uint32_t kExpectedSetupTriangles = 8970;
constexpr std::uint32_t kExpectedRasterizableTriangles = 8192;

void Check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error("indexed geometry test failed: " + message);
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::vector<float> MakeVertices(std::uint32_t surface_width,
                                std::uint32_t surface_height) {
  std::vector<float> vertices;
  vertices.reserve(2 * (kMeshWidth + 1) * (kMeshHeight + 1));
  const float size_x = 1.0F / static_cast<float>(surface_width);
  const float size_y = 1.0F / static_cast<float>(surface_height);
  const float shift_x = size_x * kMeshWidth;
  const float shift_y = size_y * kMeshHeight;
  for (std::uint32_t y = 0; y <= kMeshHeight; ++y) {
    for (std::uint32_t x = 0; x <= kMeshWidth; ++x) {
      vertices.push_back(2.0F * static_cast<float>(x) * size_x - shift_x);
      vertices.push_back(2.0F * static_cast<float>(y) * size_y - shift_y);
    }
  }
  return vertices;
}

std::vector<std::uint16_t> MakeIndices() {
  std::vector<std::uint16_t> indices;
  indices.reserve(kIndexCount);
  for (std::uint32_t y = 0; y < kMeshHeight; y += kSwathHeight) {
    for (std::uint32_t x = 0; x < kMeshWidth; ++x) {
      for (std::uint32_t swath_y = 0; swath_y < kSwathHeight; ++swath_y) {
        const std::uint32_t first =
            (y + swath_y) * (kMeshWidth + 1) + x;
        const std::uint32_t second = first + 1;
        const std::uint32_t third = first + kMeshWidth + 1;
        const std::uint32_t fourth = third + 1;
        indices.push_back(static_cast<std::uint16_t>(first));
        indices.push_back(static_cast<std::uint16_t>(third));
        indices.push_back(static_cast<std::uint16_t>(second));
        indices.push_back(static_cast<std::uint16_t>(fourth));
        indices.push_back(static_cast<std::uint16_t>(second));
        indices.push_back(static_cast<std::uint16_t>(third));
      }
    }
  }
  return indices;
}

} // namespace

int sc_main(int, char **) {
  try {
    constexpr std::uint32_t kWidth = 64;
    constexpr std::uint32_t kHeight = 64;
    MemoryPool pool;
    const std::vector<float> vertices = MakeVertices(kWidth, kHeight);
    const std::vector<std::uint16_t> indices = MakeIndices();
    Check(vertices.size() == 2 * 16641, "float2 lattice vertex count");
    Check(indices.size() == kIndexCount, "fixture index count");

    PipelineState state;
    state.width = kWidth;
    state.height = kHeight;
    state.sequence = 1;
    state.functional_case = FunctionalCase::kTriangleSetup;
    state.stage = PipelineStage::kSubmitted;
    state.draw.topology = PrimitiveTopology::kTriangleList;
    state.draw.first_index = 0;
    state.draw.index_count = indices.size();
    state.draw.base_vertex = 0;
    state.draw.index_format = IndexFormat::kUint16;
    VertexBufferResource vertex_resource;
    vertex_resource.data = StoreNewArray(pool, vertices);
    vertex_resource.byte_size =
        static_cast<std::uint32_t>(vertices.size() * sizeof(float));
    state.vertex_buffer_resources = StoreNewArray(
        pool, std::vector<VertexBufferResource>{vertex_resource});
    VertexAttributeBinding position_binding;
    position_binding.stride_bytes = 2 * sizeof(float);
    position_binding.destination_register = 0;
    position_binding.component_type = VertexComponentType::kFloat32;
    position_binding.source_components = 2;
    position_binding.destination_components = 4;
    state.vertex_attribute_bindings = StoreNewArray(
        pool, std::vector<VertexAttributeBinding>{position_binding});
    state.vertex_indices = StoreNewArray(pool, indices);
    state.vertex_code = StoreNewArray(pool, FillSolidVertexPcoBinary());
    state.drawlist_stats =
        StoreNewArray(pool, std::vector<DrawListStats>{{}});

    const PoolHandle state_handle = pool.Allocate(sizeof(PipelineState));
    StorePipelineState(pool, state_handle, state);
    const PipelineTxn txn{state_handle, 1, 1};

    sc_core::sc_fifo<PipelineTxn> input("input", 1);
    sc_core::sc_fifo<PipelineTxn> vdm_to_fetch("vdm_to_fetch", 1);
    sc_core::sc_fifo<PipelineTxn> fetch_to_decoder("fetch_to_decoder", 1);
    sc_core::sc_fifo<PipelineTxn> decoder_to_slot("decoder_to_slot", 1);
    sc_core::sc_fifo<PipelineTxn> slot_to_cluster("slot_to_cluster", 1);
    sc_core::sc_fifo<PipelineTxn> cluster_to_clip("cluster_to_clip", 1);
    sc_core::sc_fifo<PipelineTxn> clip_to_tiler("clip_to_tiler", 1);
    sc_core::sc_fifo<PipelineTxn> tiler_to_parameter("tiler_to_parameter", 1);
    sc_core::sc_fifo<PipelineTxn> output("output", 1);

    Vdm vdm("vdm", pool);
    VertexFetch fetch("vertex_fetch", pool);
    PcoDecoder decoder("vertex_pco_decoder", pool, ShaderStage::kVertex);
    UscSlot slot("vertex_usc_slot", pool, ShaderStage::kVertex);
    UscCluster cluster("vertex_usc_cluster", pool, ShaderStage::kVertex);
    ClipCull clip("clip_cull", pool);
    Tiler tiler("tiler", pool);
    ParameterBuffer parameter("parameter_buffer", pool);
    vdm.input(input);
    vdm.output(vdm_to_fetch);
    fetch.input(vdm_to_fetch);
    fetch.output(fetch_to_decoder);
    decoder.input(fetch_to_decoder);
    decoder.output(decoder_to_slot);
    slot.input(decoder_to_slot);
    slot.output(slot_to_cluster);
    cluster.input(slot_to_cluster);
    cluster.output(cluster_to_clip);
    clip.input(cluster_to_clip);
    clip.output(clip_to_tiler);
    tiler.input(clip_to_tiler);
    tiler.output(tiler_to_parameter);
    parameter.input(tiler_to_parameter);
    parameter.output(output);

    input.write(txn);
    sc_core::sc_start(sc_core::sc_time(20, sc_core::SC_US));

    PipelineTxn completed;
    Check(output.nb_read(completed) && completed.sequence == 1,
          "geometry FIFO completion");
    const PipelineState result = LoadPipelineState(pool, state_handle);
    Check(result.stage == PipelineStage::kParameterBufferReady,
          "parameter-buffer completion stage");
    Check(result.counters.ia_vertices == kIndexCount &&
              result.counters.ia_primitives == kPrimitiveCount,
          "VDM indexed IA counters");
    Check(result.counters.vs_invocations == kExpectedVertexInvocations,
          "post-transform cache miss-derived VS invocation count");
    Check(result.counters.vertex_attribute_fetches ==
                  kExpectedVertexInvocations &&
              result.counters.vertex_attribute_bytes ==
                  kExpectedVertexInvocations * 2 * sizeof(float),
          "float2 attribute traffic follows cache misses");
    Check(result.counters.vertex_fetch_cycles == kExpectedVertexFetchCycles,
          "2048-byte vertex-fetch batch preserves reference timing");

    const std::vector<VertexLane> lanes =
        LoadArray<VertexLane>(pool, result.vertex_lanes);
    const std::vector<VertexLaneRef> lane_refs =
        LoadArray<VertexLaneRef>(pool, result.vertex_lane_refs);
    Check(lanes.size() == kExpectedVertexInvocations,
          "one lane per post-transform cache miss");
    Check(lane_refs.size() == indices.size(),
          "one lane ref per index occurrence");
    for (std::size_t occurrence = 0; occurrence < lane_refs.size();
         ++occurrence) {
      const VertexLaneRef &ref = lane_refs[occurrence];
      Check(ref.vertex_index == indices[occurrence],
            "lane ref preserves resolved index");
      Check(ref.lane_index < lanes.size(), "lane ref range");
      const std::size_t source = 2 * ref.vertex_index;
      Check(lanes[ref.lane_index].vertex_input[0] ==
                    FloatBits(vertices[source]) &&
                lanes[ref.lane_index].vertex_input[1] ==
                    FloatBits(vertices[source + 1]) &&
                lanes[ref.lane_index].vertex_input[2] == FloatBits(0.0F) &&
                lanes[ref.lane_index].vertex_input[3] == FloatBits(1.0F),
            "lane input data matches indexed vertex");
    }

    const std::vector<RasterTriangle> triangles =
        LoadArray<RasterTriangle>(pool, result.raster_triangles);
    Check(result.counters.c_invocations == kPrimitiveCount,
          "clip invocations");
    Check(result.counters.c_primitives == kExpectedSetupTriangles &&
              result.counters.setup_triangles == kExpectedSetupTriangles &&
              triangles.size() == kExpectedSetupTriangles,
          "Mesa-equivalent clipping fan emission count");
    const std::size_t rasterizable =
        std::count_if(triangles.begin(), triangles.end(),
                      [](const RasterTriangle &triangle) {
                        return triangle.rasterizable != 0;
                      });
    Check(rasterizable == kExpectedRasterizableTriangles,
          "24.8 fixed-point rasterizable triangle count");
    for (std::size_t index = 0; index < triangles.size(); ++index) {
      Check(triangles[index].key.submit_ordinal == index,
            "monotonic post-clip submit identity");
      Check(triangles[index].front_facing == 0,
            "original clockwise/front-facing identity preserved");
    }

    const std::vector<ParameterTriangle> parameters =
        LoadArray<ParameterTriangle>(pool, result.parameter_triangles);
    Check(parameters.size() == triangles.size(),
          "parameter index slots are not compacted");
    for (std::size_t index = 0; index < parameters.size(); ++index) {
      Check(parameters[index].key.submit_ordinal ==
                triangles[index].key.submit_ordinal,
            "parameter primitive identity");
      Check(parameters[index].rasterizable == triangles[index].rasterizable,
            "parameter rasterizable marker");
      Check(parameters[index].rasterizable
                ? parameters[index].signed_area > 0
                : parameters[index].signed_area == 0,
            "parameter area/marker agreement");
    }

    const std::vector<TileRecord> tiles =
        LoadArray<TileRecord>(pool, result.tile_records);
    const std::vector<TilePrimitiveRef> refs =
        LoadArray<TilePrimitiveRef>(pool, result.tile_primitive_refs);
    Check(tiles.size() == 4 && result.counters.tiles_binned == 4,
          "32x32 tile grid");
    Check(refs.size() == kExpectedRasterizableTriangles,
          "degenerate triangles do not enter tile refs");
    for (const TilePrimitiveRef &ref : refs) {
      Check(ref.parameter_index < parameters.size(), "parameter ref range");
      Check(parameters[ref.parameter_index].rasterizable != 0,
            "tile ref never targets a degenerate slot");
      Check(parameters[ref.parameter_index].key.submit_ordinal ==
                ref.submit_ordinal,
            "tile ref preserves parameter identity/order");
    }

    ReleaseFunctionalPayloads(pool, result);
    pool.Release(state_handle);
    Check(pool.bytes_in_flight() == 0 &&
              pool.allocations() == pool.releases(),
          "MemoryPool balanced");
    std::cout << "indexed_geometry_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "indexed_geometry_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
