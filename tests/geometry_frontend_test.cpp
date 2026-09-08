// SPDX-License-Identifier: MIT
// Fixed-function GS boundary test. Inputs are explicit unit transactions, not
// claimed shader executions. Native instruction execution has its own tests.
#include "common/functional_types.h"
#include "common/geometry_emission.h"
#include "common/pipeline_state.h"
#include "geometry/clip_cull.h"
#include "geometry/parameter_buffer.h"
#include "geometry/tiler.h"
#include "geometry/vdm.h"
#include "geometry/vertex_fetch.h"

#include <systemc>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace pvrgpu::stub;
unsigned checks = 0;
void Check(bool condition, const char *reason) {
  ++checks;
  if (!condition) throw std::runtime_error(reason);
}
std::uint32_t Bits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
PipelineState Base(MemoryPool &pool) {
  PipelineState state;
  state.width = state.height = 64;
  state.functional_case = FunctionalCase::kDriverPcoTriangles;
  state.raster_state.face_cull.enable = 0;
  state.position_output_start = 0;
  state.position_output_count = 4;
  // Explicit stage-enable token for this fixed-function boundary test. No
  // shader module receives or attempts to execute it.
  state.geometry_code = StoreNewArray(pool, std::vector<std::uint8_t>{1});
  state.drawlist_stats = StoreNewArray(pool, std::vector<DrawListStats>{{}});
  return state;
}
PoolHandle Publish(MemoryPool &pool, const PipelineState &state) {
  const auto handle = pool.Allocate(sizeof(PipelineState));
  StorePipelineState(pool, handle, state);
  return handle;
}
void Retire(MemoryPool &pool, PoolHandle handle) {
  ReleaseFunctionalPayloads(pool, LoadPipelineState(pool, handle));
  pool.Release(handle);
}
PipelineState SourceState(MemoryPool &pool, bool indexed) {
  auto state = Base(pool);
  state.stage = PipelineStage::kSubmitted;
  state.draw.topology = indexed ? PrimitiveTopology::kTriangleStripAdjacency :
                                  PrimitiveTopology::kTriangleStrip;
  state.geometry_input_primitive_vertices = indexed ? 6 : 3;
  state.geometry_vertices_per_instance = indexed ? 8 : 5;
  state.draw.index_format = indexed ? IndexFormat::kUint16 : IndexFormat::kNone;
  state.draw.vertex_count = indexed ? 0 : 10;
  state.draw.first_vertex = indexed ? 0 : 2;
  state.draw.index_count = indexed ? 16 : 0;
  std::vector<float> vertices;
  for (unsigned i = 0; i < 16; ++i) {
    vertices.push_back(float(i) * 0.03125f);
    vertices.push_back(float(i) * -0.015625f);
    vertices.push_back(0);
    vertices.push_back(1);
  }
  VertexBufferResource resource;
  resource.data = StoreNewArray(pool, vertices);
  resource.byte_size = vertices.size() * sizeof(float);
  state.vertex_buffer_resources = StoreNewArray(pool,
      std::vector<VertexBufferResource>{resource});
  VertexAttributeBinding binding;
  binding.source_components = binding.destination_components = 4;
  binding.stride_bytes = 4 * sizeof(float);
  state.vertex_attribute_bindings = StoreNewArray(pool,
      std::vector<VertexAttributeBinding>{binding});
  if (indexed)
    state.vertex_indices = StoreNewArray(pool, std::vector<std::uint16_t>{
        0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 8, 9, 10, 11});
  std::vector<GeometryInputPrimitive> primitives;
  for (unsigned instance = 0; instance < 2; ++instance) {
    std::vector<std::uint32_t> occurrences(state.geometry_vertices_per_instance);
    for (unsigned i = 0; i < occurrences.size(); ++i)
      occurrences[i] = instance * occurrences.size() + i;
    std::vector<GeometryInputPrimitive> assembled(occurrences.size());
    std::size_t n = 0;
    Check(AssembleGeometryInputPrimitives(indexed ?
              GeometryInputTopology::kTriangleStripAdjacency :
              GeometryInputTopology::kTriangleStrip,
              occurrences.data(), occurrences.size(), false, 0, instance,
              assembled.data(), assembled.size(), n) ==
              GeometryEmissionStatus::kSuccess, "input metadata assembly");
    primitives.insert(primitives.end(), assembled.begin(), assembled.begin() + n);
  }
  state.geometry_input_primitives = StoreNewArray(pool, primitives);
  return state;
}

PipelineState RasterStateFor(MemoryPool &pool, unsigned shape, bool clipped) {
  auto state = Base(pool);
  state.stage = PipelineStage::kVertexShaded;
  state.draw.topology = PrimitiveTopology::kTriangleList;
  state.source_topology = shape == 1 ? PrimitiveTopology::kPoints :
      shape == 2 ? PrimitiveTopology::kLineStrip : PrimitiveTopology::kTriangleStrip;
  state.geometry_output_topology = state.source_topology;
  state.draw.vertex_count = shape ? 3 : 0;
  state.geometry_pco_abi.vertex_outputs = 7;
  // Deliberately different VS and GS output spans: raster linkage belongs to
  // the final programmable stage, not to the upstream vertex shader.
  state.vertex_pco_abi.vertex_outputs = 4;
  state.varying_output_start = 4;
  state.varying_output_count = 1;
  state.fragment_position_start = 0;
  state.fragment_position_count = 4;
  state.fragment_varying_start = 4;
  state.fragment_varying_count = 4;
  state.fragment_pco_abi.coefficients = 8;
  state.geometry_primitive_id_output_start = 5;
  state.geometry_primitive_id_output_count = 1;
  state.geometry_layer_output_start = 6;
  state.geometry_layer_output_count = 1;
  ShaderVaryingBinding binding;
  binding.vertex_output_base = 4;
  binding.coefficient_set_base = 1;
  binding.w_coefficient_set = 0;
  binding.component_count = 1;
  binding.interpolation = InterpolationMode::kFlat;
  state.shader_varying_bindings = StoreNewArray(pool,
      std::vector<ShaderVaryingBinding>{binding});
  std::vector<VertexLane> lanes(shape);
  const float xy[3][2] = {{-0.5f, -0.5f}, {0.5f, -0.5f}, {0.0f, 0.5f}};
  for (unsigned i = 0; i < shape; ++i) {
    lanes[i].vertex_output[0] = Bits(xy[i][0]);
    lanes[i].vertex_output[1] = Bits(clipped && i == 2 ? 2.0f : xy[i][1]);
    lanes[i].vertex_output[2] = Bits(0);
    lanes[i].vertex_output[3] = Bits(1);
    lanes[i].vertex_output[4] = UINT32_C(0xffa01230) + i;
    lanes[i].vertex_output[5] = UINT32_C(0xffffff80) + i;
    lanes[i].vertex_output[6] = 0;
    lanes[i].emitted = lanes[i].ended = 1;
  }
  std::vector<VertexLaneRef> refs;
  std::vector<GeometryRasterPrimitive> primitives;
  if (shape) {
    GeometryRasterPrimitive p;
    p.refs = {{0, shape > 1 ? 1U : 0U, shape - 1},
               static_cast<std::uint8_t>(shape),
               static_cast<std::uint8_t>(shape - 1), {}};
    p.input_primitive_id = 19;
    p.instance_id = 3;
    p.invocation_id = 2;
    primitives.push_back(p);
    for (const auto index : p.refs.vertex_indices) refs.push_back({index, index});
  }
  state.vertex_lanes = StoreNewArray(pool, lanes);
  state.vertex_lane_refs = StoreNewArray(pool, refs);
  state.geometry_primitives = StoreNewArray(pool, primitives);
  return state;
}
}  // namespace

int sc_main(int, char **) {
  try {
    MemoryPool pool;
    sc_core::sc_fifo<PipelineTxn> source_in("source_in", 1);
    sc_core::sc_fifo<PipelineTxn> source_mid("source_mid", 1);
    sc_core::sc_fifo<PipelineTxn> source_out("source_out", 1);
    Vdm vdm("vdm", pool);
    VertexFetch fetch("fetch", pool);
    vdm.input(source_in); vdm.output(source_mid);
    fetch.input(source_mid); fetch.output(source_out);
    sc_core::sc_fifo<PipelineTxn> raster_in("raster_in", 1);
    sc_core::sc_fifo<PipelineTxn> clip_out("clip_out", 1);
    sc_core::sc_fifo<PipelineTxn> tile_out("tile_out", 1);
    sc_core::sc_fifo<PipelineTxn> raster_out("raster_out", 1);
    ClipCull clip("clip", pool);
    Tiler tiler("tiler", pool);
    ParameterBuffer parameter("parameter", pool);
    clip.input(raster_in); clip.output(clip_out);
    tiler.input(clip_out); tiler.output(tile_out);
    parameter.input(tile_out); parameter.output(raster_out);

    for (bool indexed : {false, true}) {
      const auto initial = SourceState(pool, indexed);
      const auto state_handle = Publish(pool, initial);
      source_in.write({state_handle, 1, 1});
      sc_core::sc_start(sc_core::sc_time(10, sc_core::SC_US));
      PipelineTxn completion;
      Check(source_out.nb_read(completion), "source FIFO completion");
      const auto state = LoadPipelineState(pool, completion.state);
      Check(state.stage == PipelineStage::kVertexFetched, "source stage");
      const auto refs = LoadArray<VertexLaneRef>(pool, state.vertex_lane_refs);
      const auto lanes = LoadArray<VertexLane>(pool, state.vertex_lanes);
      const unsigned count = indexed ? 16 : 10;
      Check(refs.size() == count, "GS source occurrences are never triangulated");
      Check(lanes.size() == count, "source lanes/cache reset per instance");
      Check(state.draw.topology == initial.draw.topology, "source topology retained");
      Check(state.counters.ia_vertices == count &&
                state.counters.ia_primitives == (indexed ? 4 : 6),
            "primitive count respects instance boundaries");
      for (unsigned i = 0; i < count; ++i) {
        const auto index = indexed ? (i < 8 ? i : i - 4) : i + 2;
        Check(refs[i].vertex_index == index, "original resolved index identity");
        Check(lanes[refs[i].lane_index].vertex_input[0] == Bits(float(index) * 0.03125f),
              "actual source attribute DWORD fetched");
      }
      Retire(pool, state_handle);
    }
    for (unsigned shape : {0U, 1U, 2U, 3U}) {
      for (bool clipped : {false, true}) {
        if (clipped && shape != 3) continue;
        const auto state_handle = Publish(pool, RasterStateFor(pool, shape, clipped));
        raster_in.write({state_handle, 1, 1});
        sc_core::sc_start(sc_core::sc_time(10, sc_core::SC_US));
        PipelineTxn completion;
        Check(raster_out.nb_read(completion), "raster FIFO completion including zero emit");
        const auto state = LoadPipelineState(pool, completion.state);
        Check(state.stage == PipelineStage::kParameterBufferReady, "raster stage");
        const auto triangles = LoadArray<RasterTriangle>(pool, state.raster_triangles);
        const auto words = LoadArray<std::uint32_t>(pool, state.raster_vertex_outputs);
        const auto coefficients = LoadArray<ParameterCoefficientSet>(pool,
            state.parameter_coefficients);
        Check(state.counters.c_invocations == (shape ? 1 : 0), "true emitted primitive count");
        Check(shape ? !triangles.empty() : triangles.empty(), "zero output is not synthetic geometry");
        for (const auto &triangle : triangles) {
          Check(triangle.key.api_primitive_id == UINT32_C(0xffffff80) + shape - 1 &&
                    triangle.key.instance_id == 3 && triangle.key.layer == 0,
                "native shader PrimitiveID and input instance reach raster");
          for (unsigned i = 0; i < 3; ++i)
            Check(words[triangle.first_vertex_output_dword +
                        i * triangle.vertex_output_stride_dwords + 4] ==
                      UINT32_C(0xffa01230) + shape - 1,
                  "flat raw DWORD survives provoking-vertex clipping and winding");
        }
        for (std::size_t i = 1; i < coefficients.size(); i += 2)
          Check(coefficients[i].a == 0 && coefficients[i].b == 0 &&
                    coefficients[i].c == UINT32_C(0xffa01230) + shape - 1,
                "flat coefficient preserves exact raw signed/NaN payload");
        Retire(pool, state_handle);
      }
    }
    Check(pool.allocations() == pool.releases(), "all GS boundary pool handles retired");
    std::cout << "geometry-frontend-test: " << checks << " checks PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "geometry-frontend-test: " << error.what() << '\n';
    return 1;
  }
}
