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
#include <cmath>
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

PipelineState RasterStateFor(MemoryPool &pool, unsigned shape, bool clipped,
                             bool geometry = true, bool smooth_nan = false,
                             bool reverse_winding = false) {
  auto state = Base(pool);
  if (!geometry) {
    pool.Release(state.geometry_code);
    state.geometry_code = {};
  }
  state.stage = PipelineStage::kVertexShaded;
  state.draw.topology = PrimitiveTopology::kTriangleList;
  state.source_topology = shape == 1 ? PrimitiveTopology::kPoints :
      shape == 2 ? PrimitiveTopology::kLineStrip : PrimitiveTopology::kTriangleStrip;
  state.geometry_output_topology = state.source_topology;
  state.draw.vertex_count = shape ? 3 : 0;
  state.geometry_pco_abi.vertex_outputs = geometry ? 7 : 0;
  // Deliberately different VS and GS output spans: raster linkage belongs to
  // the final programmable stage, not to the upstream vertex shader.
  state.vertex_pco_abi.vertex_outputs = geometry ? 4 : 5;
  state.varying_output_start = 4;
  state.varying_output_count = 1;
  state.fragment_position_start = 0;
  state.fragment_position_count = 4;
  state.fragment_varying_start = 4;
  state.fragment_varying_count = 4;
  state.fragment_pco_abi.coefficients = 8;
  state.geometry_primitive_id_output_start = 5;
  state.geometry_primitive_id_output_count = geometry ? 1 : 0;
  state.geometry_layer_output_start = 6;
  state.geometry_layer_output_count = geometry ? 1 : 0;
  ShaderVaryingBinding binding;
  binding.vertex_output_base = 4;
  binding.coefficient_set_base = 1;
  binding.w_coefficient_set = 0;
  binding.component_count = 1;
  binding.interpolation = smooth_nan ? InterpolationMode::kSmooth :
                                      InterpolationMode::kFlat;
  state.shader_varying_bindings = StoreNewArray(pool,
      std::vector<ShaderVaryingBinding>{binding});
  std::vector<VertexLane> lanes(shape);
  const float xy[3][2] = {{-0.5f, -0.5f}, {0.5f, -0.5f}, {0.0f, 0.5f}};
  for (unsigned i = 0; i < shape; ++i) {
    lanes[i].vertex_output[0] = Bits(reverse_winding ? -xy[i][0] : xy[i][0]);
    lanes[i].vertex_output[1] = Bits(clipped && i == shape - 1 ? 2.0f : xy[i][1]);
    lanes[i].vertex_output[2] = Bits(0);
    lanes[i].vertex_output[3] = Bits(1);
    lanes[i].vertex_output[4] = smooth_nan ? UINT32_C(0x7fc12345) :
                                          UINT32_C(0xffa01230) + i;
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
    for (unsigned occurrence = 0; occurrence < 3; ++occurrence) {
      const auto index = p.refs.vertex_indices[occurrence];
      refs.push_back({index, geometry ? index : occurrence});
    }
  }
  state.vertex_lanes = StoreNewArray(pool, lanes);
  state.vertex_lane_refs = StoreNewArray(pool, refs);
  if (geometry)
    state.geometry_primitives = StoreNewArray(pool, primitives);
  return state;
}

void VerifyExplicitGeometryBindings(MemoryPool &pool) {
  auto state = Base(pool);
  state.vertex_pco_abi.vertex_outputs = 4;
  state.geometry_pco_abi.vertex_outputs = 6;
  state.varying_output_start = 4; state.varying_output_count = 2;
  state.fragment_pco_abi.coefficients = 8;
  state.driver_varying_bindings_explicit = 1;
  state.driver_varying_binding_count = 1;
  state.geometry_primitive_id_output_start = 4; state.geometry_primitive_id_output_count = 1;
  state.geometry_layer_output_start = 5; state.geometry_layer_output_count = 1;
  ShaderVaryingBinding binding;
  binding.vertex_output_base = 5; binding.coefficient_set_base = 1;
  binding.component_count = 1; binding.interpolation = InterpolationMode::kFlat;
  Check(IsExactVaryingBinding(state, binding, 0), "explicit raw Layer follows final GS ABI, not smaller VS ABI");
  binding.vertex_output_base = 6;
  Check(!IsExactVaryingBinding(state, binding, 0), "explicit GS binding rejects final output overflow");
  binding.vertex_output_base = 5; binding.interpolation = InterpolationMode::kSmooth;
  const char *reason = nullptr;
  Check(!IsExactVaryingBinding(state, binding, 0, &reason) && reason &&
        std::strcmp(reason, "geometry_integer_varying_not_flat") == 0,
        "Layer cannot reach smooth float plane interpolation");
  binding.vertex_output_base = 4;
  Check(!IsExactVaryingBinding(state, binding, 0), "PrimitiveID also remains raw integer flat data");
  state.geometry_primitive_id_output_count = 0;
  Check(IsExactVaryingBinding(state, binding, 0), "ordinary float varying before Layer may stay smooth");
  binding.component_count = 2; state.fragment_pco_abi.coefficients = 12;
  Check(!IsExactVaryingBinding(state, binding, 0), "mixed smooth range cannot overlap raw Layer");
  binding.interpolation = InterpolationMode::kFlat;
  Check(IsExactVaryingBinding(state, binding, 0), "flat range may transport both raw DWORDs");
  Check(!IsExactVaryingBinding(state, binding, 1), "explicit GS binding count is bounded");
  binding.vertex_output_base = UINT16_MAX;
  Check(!IsExactVaryingBinding(state, binding, 0), "explicit GS output arithmetic cannot wrap");
  ReleaseFunctionalPayloads(pool, state);
}
}  // namespace

int sc_main(int, char **) {
  try {
    MemoryPool pool;
    VerifyExplicitGeometryBindings(pool);
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
    sc_core::sc_fifo<PipelineTxn> clip_only_in("clip_only_in", 1);
    sc_core::sc_fifo<PipelineTxn> clip_only_out("clip_only_out", 1);
    ClipCull clip_only("clip_only", pool);
    clip_only.input(clip_only_in); clip_only.output(clip_only_out);

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
    // GLES3 VS outputs use the same flat/raw contract as GS outputs. The
    // original last vertex must survive clipping it away, widening lines or
    // points, fan triangulation and either input winding.
    for (unsigned shape : {1U, 2U, 3U}) {
      for (bool clipped : {false, true}) {
        if (clipped && shape == 1) continue;
        for (bool reverse_winding : {false, true}) {
          const auto state_handle = Publish(pool, RasterStateFor(
              pool, shape, clipped, false, false, reverse_winding));
          raster_in.write({state_handle, 1, 1});
          sc_core::sc_start(sc_core::sc_time(10, sc_core::SC_US));
          PipelineTxn completion;
          Check(raster_out.nb_read(completion), "VS flat raster FIFO completion");
          const auto state = LoadPipelineState(pool, completion.state);
          const auto triangles = LoadArray<RasterTriangle>(pool, state.raster_triangles);
          const auto words = LoadArray<std::uint32_t>(pool, state.raster_vertex_outputs);
          const auto coefficients = LoadArray<ParameterCoefficientSet>(pool,
              state.parameter_coefficients);
          Check(!triangles.empty(), "VS flat primitive retains actual coverage");
          for (const auto &triangle : triangles)
            for (unsigned i = 0; i < 3; ++i)
              Check(words[triangle.first_vertex_output_dword +
                          i * triangle.vertex_output_stride_dwords + 4] ==
                        UINT32_C(0xffa01230) + shape - 1,
                    "VS original provoking raw DWORD survives clipping and winding");
          for (std::size_t i = 1; i < coefficients.size(); i += 2)
            Check(coefficients[i].a == 0 && coefficients[i].b == 0 &&
                      coefficients[i].c == UINT32_C(0xffa01230) + shape - 1,
                  "VS flat coefficient retains exact raw signed/NaN payload");
          Retire(pool, state_handle);
        }
      }
    }
    // Smooth floating outputs may also be NaN. Their propagation is distinct
    // from flat raw transport; the positions in this fixture remain finite.
    // Check the clipping boundary directly.
    for (bool geometry : {false, true}) {
      const auto state_handle = Publish(pool, RasterStateFor(
          pool, 3, true, geometry, true));
      clip_only_in.write({state_handle, 1, 1});
      sc_core::sc_start(sc_core::sc_time(10, sc_core::SC_US));
      PipelineTxn completion;
      Check(clip_only_out.nb_read(completion), "smooth NaN clip FIFO completion");
      const auto state = LoadPipelineState(pool, completion.state);
      const auto triangles = LoadArray<RasterTriangle>(pool, state.raster_triangles);
      const auto words = LoadArray<std::uint32_t>(pool, state.raster_vertex_outputs);
      Check(triangles.size() > 1, "smooth NaN test generates clip intersections");
      for (const auto &triangle : triangles)
        for (unsigned i = 0; i < 3; ++i) {
          const auto offset = triangle.first_vertex_output_dword +
                              i * triangle.vertex_output_stride_dwords;
          for (unsigned component = 0; component < 4; ++component) {
            float value;
            std::memcpy(&value, &words[offset + component], sizeof(value));
            Check(std::isfinite(value), "clip positions remain finite");
          }
          float varying;
          std::memcpy(&varying, &words[offset + 4], sizeof(varying));
          Check(std::isnan(varying), "legal smooth NaN varying survives clipping");
        }
      Retire(pool, state_handle);
    }
    // Mesa's clip test maps unordered comparisons to clip bits, then the
    // generic clipper drops only the primitive whose selected plane distance
    // is NaN or infinity.  Preserve the original shader DWORDs while proving
    // that neither a non-finite position nor a user clip distance aborts the
    // surrounding submission.  The finite raster cases above are controls.
    struct NonFiniteClipCase {
      unsigned component;
      std::uint32_t bits;
    };
    const std::array<NonFiniteClipCase, 6> nonfinite_positions = {{
        {0, UINT32_C(0xffc00000)},
        {1, UINT32_C(0x7fc12345)},
        {2, UINT32_C(0x7f800000)},
        {2, UINT32_C(0xff800000)},
        {3, UINT32_C(0xffc00000)},
        {3, UINT32_C(0xff800000)},
    }};
    for (bool geometry : {false, true}) {
      for (const auto &fixture : nonfinite_positions) {
        auto initial = RasterStateFor(pool, 3, false, geometry);
        auto original_lanes = LoadArray<VertexLane>(pool, initial.vertex_lanes);
        original_lanes[2].vertex_output[fixture.component] = fixture.bits;
        pool.Release(initial.vertex_lanes);
        initial.vertex_lanes = StoreNewArray(pool, original_lanes);
        const auto state_handle = Publish(pool, initial);
        clip_only_in.write({state_handle, 1, 1});
        sc_core::sc_start(sc_core::sc_time(10, sc_core::SC_US));
        PipelineTxn completion;
        Check(clip_only_out.nb_read(completion),
              "non-finite clip position completes");
        const auto state = LoadPipelineState(pool, completion.state);
        const auto triangles = LoadArray<RasterTriangle>(
            pool, state.raster_triangles);
        const auto output_lanes = LoadArray<VertexLane>(pool, state.vertex_lanes);
        Check(state.stage == PipelineStage::kClipCullComplete,
              "non-finite position reaches clip completion");
        Check(state.counters.c_invocations == 1 &&
                  state.counters.c_primitives == 0 && triangles.empty(),
              "non-finite position discards only its primitive");
        Check(output_lanes.size() == original_lanes.size() &&
                  std::memcmp(output_lanes.data(), original_lanes.data(),
                              original_lanes.size() * sizeof(VertexLane)) == 0,
              "non-finite position preserves original VTXOUT bits");
        Retire(pool, state_handle);
      }

      // A clean +Inf W is inside every hard plane in Mesa and reaches the
      // viewport with reciprocal W = +0.  If another vertex selects a hard
      // plane, however, Mesa's four-term dot sees that infinity and drops the
      // primitive from the generic clipper.
      for (const bool select_hard_plane : {false, true}) {
        auto initial = RasterStateFor(pool, 3, false, geometry);
        auto original_lanes = LoadArray<VertexLane>(pool, initial.vertex_lanes);
        original_lanes[2].vertex_output[3] = UINT32_C(0x7f800000);
        if (select_hard_plane)
          original_lanes[1].vertex_output[0] = Bits(2.0F);
        pool.Release(initial.vertex_lanes);
        initial.vertex_lanes = StoreNewArray(pool, original_lanes);
        const auto state_handle = Publish(pool, initial);
        clip_only_in.write({state_handle, 1, 1});
        sc_core::sc_start(sc_core::sc_time(10, sc_core::SC_US));
        PipelineTxn completion;
        Check(clip_only_out.nb_read(completion),
              "+Inf W clip transaction completes");
        const auto state = LoadPipelineState(pool, completion.state);
        const auto triangles = LoadArray<RasterTriangle>(
            pool, state.raster_triangles);
        Check(state.counters.c_invocations == 1,
              "+Inf W retains its clip invocation");
        if (select_hard_plane) {
          Check(state.counters.c_primitives == 0 && triangles.empty(),
                "+Inf W on a selected hard plane discards its primitive");
        } else {
          Check(state.counters.c_primitives == 1 && triangles.size() == 1,
                "clean +Inf W reaches fixed setup");
          bool found_zero_reciprocal = false;
          for (const auto &triangle : triangles) {
            for (unsigned i = 0; i < 3; ++i) {
              found_zero_reciprocal |= triangle.reciprocal_w[i] == 0.0F;
              Check(std::isfinite(triangle.x[i]) &&
                        std::isfinite(triangle.y[i]) &&
                        std::isfinite(triangle.window_z[i]) &&
                        std::isfinite(triangle.reciprocal_w[i]),
                    "clean +Inf W has a finite viewport result");
            }
          }
          Check(found_zero_reciprocal,
                "clean +Inf W produces zero reciprocal W");
        }
        Retire(pool, state_handle);
      }

      for (const std::uint32_t bits : {UINT32_C(0x7fc45678),
                                       UINT32_C(0x7f800000),
                                       UINT32_C(0xff800000)}) {
        auto initial = RasterStateFor(pool, 3, false, geometry, true);
        auto original_lanes = LoadArray<VertexLane>(pool, initial.vertex_lanes);
        for (auto &lane : original_lanes)
          lane.vertex_output[4] = Bits(1.0F);
        original_lanes[1].vertex_output[4] = bits;
        pool.Release(initial.vertex_lanes);
        initial.vertex_lanes = StoreNewArray(pool, original_lanes);
        initial.clip_distance_mask = 1;
        initial.clip_distance_register = 4;
        const auto state_handle = Publish(pool, initial);
        clip_only_in.write({state_handle, 1, 1});
        sc_core::sc_start(sc_core::sc_time(10, sc_core::SC_US));
        PipelineTxn completion;
        Check(clip_only_out.nb_read(completion),
              "non-finite user clip distance completes");
        const auto state = LoadPipelineState(pool, completion.state);
        const auto triangles = LoadArray<RasterTriangle>(
            pool, state.raster_triangles);
        const auto output_lanes =
            LoadArray<VertexLane>(pool, state.vertex_lanes);
        Check(state.stage == PipelineStage::kClipCullComplete,
              "non-finite user clip distance reaches clip completion");
        Check(state.counters.c_invocations == 1,
              "non-finite user clip distance retains its invocation");
        Check(state.counters.c_primitives == 0 && triangles.empty(),
              "non-finite user clip distance discards only its primitive");
        Check(output_lanes.size() == original_lanes.size() &&
                  std::memcmp(output_lanes.data(), original_lanes.data(),
                              original_lanes.size() * sizeof(VertexLane)) == 0,
              "non-finite user clip distance preserves original VTXOUT bits");
        Retire(pool, state_handle);
      }

      auto initial = RasterStateFor(pool, 3, false, geometry, true);
      auto original_lanes = LoadArray<VertexLane>(pool, initial.vertex_lanes);
      original_lanes[1].vertex_output[4] = UINT32_C(0x7f800000);
      pool.Release(initial.vertex_lanes);
      initial.vertex_lanes = StoreNewArray(pool, original_lanes);
      initial.clip_distance_mask = 0;
      initial.clip_distance_register = 4;
      const auto state_handle = Publish(pool, initial);
      clip_only_in.write({state_handle, 1, 1});
      sc_core::sc_start(sc_core::sc_time(10, sc_core::SC_US));
      PipelineTxn completion;
      Check(clip_only_out.nb_read(completion),
            "disabled non-finite user clip distance completes");
      const auto state = LoadPipelineState(pool, completion.state);
      const auto triangles = LoadArray<RasterTriangle>(
          pool, state.raster_triangles);
      Check(state.counters.c_invocations == 1 &&
                state.counters.c_primitives == 1 && triangles.size() == 1,
            "disabled non-finite user clip distance does not clip");
      Retire(pool, state_handle);
    }

    // The captured failure has invalid and ordinary triangles in one draw.
    // Keep both in one occurrence segment so this distinguishes per-primitive
    // discard from either aborting or dropping the complete submission.
    {
      auto initial = RasterStateFor(pool, 3, false, false);
      auto original_lanes = LoadArray<VertexLane>(pool, initial.vertex_lanes);
      const auto finite_lanes = original_lanes;
      original_lanes.insert(original_lanes.end(), finite_lanes.begin(),
                            finite_lanes.end());
      original_lanes[2].vertex_output[0] = UINT32_C(0xffc00000);
      pool.Release(initial.vertex_lanes);
      initial.vertex_lanes = StoreNewArray(pool, original_lanes);
      pool.Release(initial.vertex_lane_refs);
      std::vector<VertexLaneRef> refs;
      for (std::uint32_t i = 0; i < original_lanes.size(); ++i)
        refs.push_back({i, i});
      initial.vertex_lane_refs = StoreNewArray(pool, refs);
      initial.draw.vertex_count = 6;
      const auto state_handle = Publish(pool, initial);
      clip_only_in.write({state_handle, 1, 1});
      sc_core::sc_start(sc_core::sc_time(10, sc_core::SC_US));
      PipelineTxn completion;
      Check(clip_only_out.nb_read(completion),
            "mixed finite/non-finite submission completes");
      const auto state = LoadPipelineState(pool, completion.state);
      const auto triangles = LoadArray<RasterTriangle>(
          pool, state.raster_triangles);
      const auto output_lanes = LoadArray<VertexLane>(pool, state.vertex_lanes);
      Check(state.counters.c_invocations == 2 &&
                state.counters.c_primitives == 1 && triangles.size() == 1,
            "mixed submission drops only the non-finite primitive");
      Check(output_lanes.size() == original_lanes.size() &&
                std::memcmp(output_lanes.data(), original_lanes.data(),
                            original_lanes.size() * sizeof(VertexLane)) == 0,
            "mixed submission preserves every original VTXOUT DWORD");
      Retire(pool, state_handle);
    }
    // A finite homogeneous position is legal even at/behind the eye. The
    // clip cone's apex has no projective area, but clipping a w=0 non-apex
    // vertex or a negative-w vertex can still produce visible geometry.
    // Test both final programmable stages, independent of TF or case names.
    const std::array<std::array<std::array<float, 4>, 3>, 7> eye_positions = {{
        {{{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}},
        {{{0, 0, 0, 0}, {1, 0, 0, 0}, {2, 0, 0, 0}}},
        {{{-0.5f, -0.5f, 0, -1}, {0.5f, -0.5f, 0, -1}, {0, 0.5f, 0, -1}}},
        {{{-0.5f, -0.5f, 0, 1}, {0.5f, -0.5f, 0, 1}, {0, 0, 0, 0}}},
        {{{-0.5f, -0.5f, 0, 1}, {0.5f, -0.5f, 0, 1}, {0, 0, 0, -0.5f}}},
        {{{-0.5f, -0.5f, 0, 1}, {0.5f, -0.5f, 0, 1}, {0, 0.5f, 0, 0}}},
        // Clipping the first edge synthesizes the apex rather than reading
        // it from the shader. All finite projections lie on x+y=1.
        {{{-1, 0, 0, -1}, {1, 0, 0, 1}, {0, 1, 0, 1}}},
    }};
    for (bool geometry : {false, true}) {
      for (bool depth_clamp : {false, true}) {
        for (unsigned fixture = 0; fixture < eye_positions.size(); ++fixture) {
          for (unsigned shape : {1U, 2U, 3U}) {
            // Mixed-eye visible coverage is a triangle-specific assertion;
            // the degenerate origin/all-zero/all-behind fixtures cover all
            // three topologies (including widened point/line fallbacks).
            if (fixture >= 3 && shape != 3) continue;
            auto initial = RasterStateFor(pool, shape, false, geometry);
            auto original_lanes = LoadArray<VertexLane>(pool, initial.vertex_lanes);
            for (unsigned i = 0; i < shape; ++i)
              for (unsigned c = 0; c < 4; ++c)
                original_lanes[i].vertex_output[c] = Bits(eye_positions[fixture][i][c]);
            // Depth-clamped apex keeps a finite, nonzero clip Z and still
            // has no XY projection. It must not create synthetic fragments.
            if (depth_clamp && fixture == 0)
              for (auto &lane : original_lanes)
                lane.vertex_output[2] = Bits(3.0f);
            pool.Release(initial.vertex_lanes);
            initial.vertex_lanes = StoreNewArray(pool, original_lanes);
            initial.raster_state.depth_clamp_enable = depth_clamp;
            const auto state_handle = Publish(pool, initial);
            raster_in.write({state_handle, 1, 1});
            sc_core::sc_start(sc_core::sc_time(10, sc_core::SC_US));
            PipelineTxn completion;
            Check(raster_out.nb_read(completion), "eye-plane raster FIFO completes");
            const auto state = LoadPipelineState(pool, completion.state);
            const auto triangles = LoadArray<RasterTriangle>(pool, state.raster_triangles);
            const auto output_lanes = LoadArray<VertexLane>(pool, state.vertex_lanes);
            Check(state.stage == PipelineStage::kParameterBufferReady,
                  "eye-plane completion reaches parameter buffer");
            Check(state.counters.c_invocations == 1,
                  "eye-plane discard retains clipping invocation");
            Check(output_lanes.size() == original_lanes.size(),
                  "clipping never removes original stream-output lanes");
            for (unsigned i = 0; i < output_lanes.size(); ++i)
              Check(std::memcmp(output_lanes[i].vertex_output,
                                original_lanes[i].vertex_output,
                                sizeof(output_lanes[i].vertex_output)) == 0,
                    "eye-plane clipping preserves every original VTXOUT DWORD");
            if (fixture < 4) {
              Check(triangles.empty(), "nonprojectable primitive has no raster output");
              Check(state.counters.c_primitives == 0,
                    "no synthetic setup primitive for clip apex");
            } else if (fixture == 6) {
              for (const auto &triangle : triangles)
                Check(!triangle.rasterizable,
                      "generated apex does not turn a projective line into coverage");
            } else {
              bool covered = false;
              for (const auto &triangle : triangles) {
                covered |= triangle.rasterizable != 0;
                for (unsigned i = 0; i < 3; ++i)
                  Check(std::isfinite(triangle.x[i]) && std::isfinite(triangle.y[i]) &&
                            std::isfinite(triangle.reciprocal_w[i]) &&
                            triangle.reciprocal_w[i] > 0,
                        "eye-crossing survivors have finite positive projection");
              }
              Check(covered, "eye-crossing input retains genuinely visible coverage");
            }
            Retire(pool, state_handle);
          }
        }
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
