/*
 * Triangle.Setup.AllCulled segmented clip/cull regression.
 * 中文：以六種 viewport 尺寸驅動完整 VDM（Vertex Data Master）→
 * VertexFetch→PCO decoder→USC（Unified Shading Cluster）ISS（Instruction
 * Set Simulator）→ClipCull event-driven FIFO 路徑。測試 1023-index segment
 * 的 aggregate clipmask fast/slow 選擇與 GLES GL_BACK/GL_CCW face cull；
 * expected setup counts 跨多尺寸變化，避免 case-name counter shortcut。
 */
#include "common/functional_types.h"
#include "common/pipeline_state.h"
#include "geometry/clip_cull.h"
#include "geometry/vdm.h"
#include "geometry/vertex_fetch.h"
#include "shader/pco_decoder.h"
#include "shader/pco_iss.h"
#include "shader/usc_cluster.h"
#include "shader/usc_slot.h"

#include <systemc>

#include <array>
#include <cstdint>
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
constexpr std::uint32_t kVertexInvocations = 21144;

struct ViewportExpectation {
  std::uint32_t size;
  std::uint32_t setup_triangles;
};

constexpr std::array<ViewportExpectation, 6> kExpectations = {{
    {32, 0},
    {48, 4092},
    {64, 5456},
    {80, 6479},
    {96, 7843},
    {128, 32768},
}};

void Check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error("all-culled geometry test failed: " + message);
}

std::vector<float> MakeVertices(std::uint32_t surface_size) {
  std::vector<float> vertices;
  vertices.reserve(2 * (kMeshWidth + 1) * (kMeshHeight + 1));
  const float size = 1.0F / static_cast<float>(surface_size);
  const float shift_x = size * kMeshWidth;
  const float shift_y = size * kMeshHeight;
  for (std::uint32_t y = 0; y <= kMeshHeight; ++y) {
    for (std::uint32_t x = 0; x <= kMeshWidth; ++x) {
      vertices.push_back(2.0F * static_cast<float>(x) * size - shift_x);
      vertices.push_back(2.0F * static_cast<float>(y) * size - shift_y);
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

PipelineState MakeState(MemoryPool &pool, std::uint32_t surface_size,
                        std::uint64_t sequence) {
  const std::vector<float> vertices = MakeVertices(surface_size);
  const std::vector<std::uint16_t> indices = MakeIndices();
  Check(vertices.size() == 2 * 16641, "float2 lattice vertex count");
  Check(indices.size() == kIndexCount, "fixture index count");

  PipelineState state;
  state.width = surface_size;
  state.height = surface_size;
  state.sequence = sequence;
  state.functional_case = FunctionalCase::kTriangleSetupAllCulled;
  state.stage = PipelineStage::kSubmitted;
  state.draw.topology = PrimitiveTopology::kTriangleList;
  state.draw.first_index = 0;
  state.draw.index_count = indices.size();
  state.draw.base_vertex = 0;
  state.draw.index_format = IndexFormat::kUint16;
  state.raster_state.face_cull.enable = 1;
  state.raster_state.face_cull.mode = CullFaceMode::kBack;
  state.raster_state.face_cull.front_face =
      FrontFaceWinding::kCounterClockwise;
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
  state.drawlist_stats = StoreNewArray(pool, std::vector<DrawListStats>{{}});
  return state;
}

} // namespace

int sc_main(int, char **) {
  try {
    Check(FunctionalCaseFromName("triangle_setup_all_culled") ==
              FunctionalCase::kTriangleSetupAllCulled,
          "functional case mapping");
    Check(std::string(FunctionalCaseName(
              FunctionalCase::kTriangleSetupAllCulled)) ==
              "triangle_setup_all_culled",
          "functional case reverse mapping");
    const RasterState default_raster;
    Check(default_raster.face_cull.enable == 0 &&
              default_raster.face_cull.mode == CullFaceMode::kBack &&
              default_raster.face_cull.front_face ==
                  FrontFaceWinding::kCounterClockwise,
          "disabled GLES BACK/CCW defaults");

    MemoryPool pool;
    sc_core::sc_fifo<PipelineTxn> input("input", 1);
    sc_core::sc_fifo<PipelineTxn> vdm_to_fetch("vdm_to_fetch", 1);
    sc_core::sc_fifo<PipelineTxn> fetch_to_decoder("fetch_to_decoder", 1);
    sc_core::sc_fifo<PipelineTxn> decoder_to_slot("decoder_to_slot", 1);
    sc_core::sc_fifo<PipelineTxn> slot_to_cluster("slot_to_cluster", 1);
    sc_core::sc_fifo<PipelineTxn> cluster_to_clip("cluster_to_clip", 1);
    sc_core::sc_fifo<PipelineTxn> output("output", 1);

    Vdm vdm("vdm", pool);
    VertexFetch fetch("vertex_fetch", pool);
    PcoDecoder decoder("vertex_pco_decoder", pool, ShaderStage::kVertex);
    UscSlot slot("vertex_usc_slot", pool, ShaderStage::kVertex);
    UscCluster cluster("vertex_usc_cluster", pool, ShaderStage::kVertex);
    ClipCull clip("clip_cull", pool);
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
    clip.output(output);

    for (std::size_t test_index = 0; test_index < kExpectations.size();
         ++test_index) {
      const ViewportExpectation expectation = kExpectations[test_index];
      const std::uint64_t sequence = test_index + 1;
      const PipelineState submitted =
          MakeState(pool, expectation.size, sequence);
      const PoolHandle state_handle = pool.Allocate(sizeof(PipelineState));
      StorePipelineState(pool, state_handle, submitted);
      input.write({state_handle, static_cast<std::uint32_t>(sequence),
                   sequence});
      sc_core::sc_start(sc_core::sc_time(20, sc_core::SC_US));

      PipelineTxn completed;
      Check(output.nb_read(completed) && completed.sequence == sequence,
            "geometry FIFO completion at viewport " +
                std::to_string(expectation.size));
      const PipelineState result = LoadPipelineState(pool, state_handle);
      Check(result.stage == PipelineStage::kClipCullComplete,
            "clip/cull completion stage");
      Check(result.counters.ia_vertices == kIndexCount &&
                result.counters.ia_primitives == kPrimitiveCount &&
                result.counters.vs_invocations == kVertexInvocations &&
                result.counters.vertex_attribute_fetches ==
                    kVertexInvocations &&
                result.counters.vertex_attribute_bytes ==
                    kVertexInvocations * 2 * sizeof(float) &&
                result.counters.c_invocations == kPrimitiveCount,
            "upstream indexed counters");
      Check(result.counters.c_primitives == expectation.setup_triangles,
            "segment-derived setup count at viewport " +
                std::to_string(expectation.size));

      const std::vector<RasterTriangle> triangles =
          LoadArray<RasterTriangle>(pool, result.raster_triangles);
      Check(triangles.size() == expectation.setup_triangles,
            "setup payload count at viewport " +
                std::to_string(expectation.size));
      std::uint64_t previous_submit_ordinal = 0;
      bool first = true;
      for (const RasterTriangle &triangle : triangles) {
        Check(triangle.front_facing == 0 && triangle.face_culled == 1 &&
                  triangle.rasterizable == 0,
              "CW/BACK triangle rejected before binning");
        if (!first)
          Check(triangle.key.submit_ordinal > previous_submit_ordinal,
                "surviving setup slots preserve API order");
        first = false;
        previous_submit_ordinal = triangle.key.submit_ordinal;
      }

      ReleaseFunctionalPayloads(pool, result);
      pool.Release(state_handle);
      Check(pool.bytes_in_flight() == 0 &&
                pool.allocations() == pool.releases(),
            "MemoryPool balanced after viewport " +
                std::to_string(expectation.size));
    }

    std::cout << "all_culled_geometry_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "all_culled_geometry_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
