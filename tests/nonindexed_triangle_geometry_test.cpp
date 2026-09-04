/*
 * Driver PCO non-indexed triangle-list geometry regression.
 *
 * Uses nine vertices starting at vertex one so the route cannot accidentally
 * depend on the historical fullscreen 4/6-vertex fixtures. VDM, VertexFetch,
 * PCO/USC, ClipCull, Tiler and ParameterBuffer must preserve direct-draw
 * occurrence identity without manufacturing an index buffer.
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

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace pvrgpu::stub;

constexpr std::uint32_t kWidth = 64;
constexpr std::uint32_t kHeight = 64;
constexpr std::uint32_t kFirstVertex = 1;
constexpr std::uint32_t kVertexCount = 9;
constexpr std::uint32_t kPrimitiveCount = kVertexCount / 3;

void Check(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(
        "non-indexed triangle geometry test failed: " + message);
  }
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void CheckPlane(const ParameterCoefficientSet &plane, std::uint32_t a,
                std::uint32_t b, std::uint32_t c,
                const std::string &description) {
  Check(plane.a == a && plane.b == b && plane.c == c && plane.pad == 0,
        description);
}

float BitsFloat(std::uint32_t bits) {
  float value = 0.0F;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::vector<float> MakeFloat3Vertices() {
  return {
      // Prefix vertex proves first_vertex participates in fetch identity.
      8.0F, 8.0F, 8.0F,
      -0.9F, -0.9F, 0.0F,
      -0.5F, -0.9F, 0.0F,
      -0.7F, -0.5F, 0.0F,
      -0.2F, -0.2F, 0.0F,
      0.2F, -0.2F, 0.0F,
      0.0F, 0.2F, 0.0F,
      0.5F, 0.5F, 0.0F,
      0.9F, 0.5F, 0.0F,
      0.7F, 0.9F, 0.0F,
  };
}

void CheckTextureAttributeFetch() {
  MemoryPool pool;
  const std::vector<float> vertices = {
      -0.8F, -0.8F, 0.0F, 0.0F, 0.0F, 1.0F, 0.1F, 0.2F,
       0.8F, -0.8F, 0.0F, 0.0F, 0.0F, 1.0F, 0.9F, 0.2F,
       0.0F,  0.8F, 0.0F, 0.0F, 0.0F, 1.0F, 0.5F, 0.9F,
  };

  PipelineState state;
  state.width = kWidth;
  state.height = kHeight;
  state.sequence = 1;
  state.functional_case = FunctionalCase::kDriverPcoTriangles;
  state.stage = PipelineStage::kSubmitted;
  state.sampled_texture_count = 1;
  state.draw.topology = PrimitiveTopology::kTriangleList;
  state.draw.vertex_count = 3;
  state.draw.index_format = IndexFormat::kNone;
  Check(UsesTextureSampling(state),
        "driver texture route is selected from command state");

  VertexBufferResource resource;
  resource.data = StoreNewArray(pool, vertices);
  resource.byte_size =
      static_cast<std::uint32_t>(vertices.size() * sizeof(float));
  state.vertex_buffer_resources = StoreNewArray(
      pool, std::vector<VertexBufferResource>{resource});

  VertexAttributeBinding position;
  position.stride_bytes = 8U * sizeof(float);
  position.destination_register = 0;
  position.component_type = VertexComponentType::kFloat32;
  position.source_components = 3;
  position.destination_components = 4;
  VertexAttributeBinding normal = position;
  normal.offset_bytes = 3U * sizeof(float);
  normal.destination_register = 4;
  VertexAttributeBinding texcoord = position;
  texcoord.offset_bytes = 6U * sizeof(float);
  texcoord.destination_register = 8;
  texcoord.source_components = 2;
  state.vertex_attribute_bindings = StoreNewArray(
      pool, std::vector<VertexAttributeBinding>{position, normal, texcoord});
  state.drawlist_stats =
      StoreNewArray(pool, std::vector<DrawListStats>{{}});

  const PoolHandle handle = pool.Allocate(sizeof(PipelineState));
  StorePipelineState(pool, handle, state);
  const PipelineTxn txn{handle, 1, 1};
  sc_core::sc_fifo<PipelineTxn> input("texture_fetch_input", 1);
  sc_core::sc_fifo<PipelineTxn> vdm_to_fetch("texture_vdm_to_fetch", 1);
  sc_core::sc_fifo<PipelineTxn> output("texture_fetch_output", 1);
  Vdm vdm("texture_vdm", pool);
  VertexFetch fetch("texture_vertex_fetch", pool);
  vdm.input(input);
  vdm.output(vdm_to_fetch);
  fetch.input(vdm_to_fetch);
  fetch.output(output);

  input.write(txn);
  sc_core::sc_start(sc_core::sc_time(20, sc_core::SC_US));
  PipelineTxn completed;
  Check(output.nb_read(completed) && completed.state.slot == handle.slot &&
            completed.state.generation == handle.generation,
        "texture attribute fetch completion identity");
  const PipelineState result = LoadPipelineState(pool, handle);
  Check(result.stage == PipelineStage::kVertexFetched &&
            result.counters.ia_vertices == 3 &&
            result.counters.vertex_attribute_fetches == 9 &&
            result.counters.vertex_attribute_bytes ==
                3U * 8U * sizeof(float),
        "stride-32 position/normal/UV fetch counters");
  const std::vector<VertexLane> lanes =
      LoadArray<VertexLane>(pool, result.vertex_lanes);
  Check(lanes.size() == 3, "one texture lane per non-indexed vertex");
  for (std::size_t lane = 0; lane < lanes.size(); ++lane) {
    const std::size_t source = lane * 8U;
    Check(lanes[lane].vertex_input[0] == FloatBits(vertices[source]) &&
              lanes[lane].vertex_input[1] == FloatBits(vertices[source + 1]) &&
              lanes[lane].vertex_input[2] == FloatBits(vertices[source + 2]) &&
              lanes[lane].vertex_input[3] == FloatBits(1.0F) &&
              lanes[lane].vertex_input[4] == FloatBits(vertices[source + 3]) &&
              lanes[lane].vertex_input[5] == FloatBits(vertices[source + 4]) &&
              lanes[lane].vertex_input[6] == FloatBits(vertices[source + 5]) &&
              lanes[lane].vertex_input[7] == FloatBits(1.0F) &&
              lanes[lane].vertex_input[8] == FloatBits(vertices[source + 6]) &&
              lanes[lane].vertex_input[9] == FloatBits(vertices[source + 7]) &&
              lanes[lane].vertex_input[10] == FloatBits(0.0F) &&
              lanes[lane].vertex_input[11] == FloatBits(1.0F),
          "UV float2 at byte 24 reaches VTXIN8..11");
  }

  ReleaseFunctionalPayloads(pool, result);
  pool.Release(handle);
  Check(pool.bytes_in_flight() == 0 &&
            pool.allocations() == pool.releases(),
        "texture fetch payload ownership cleanup");
}

VertexLane MakeClipLane(float x, float y, float z, float w) {
  VertexLane lane;
  lane.vertex_output[0] = FloatBits(x);
  lane.vertex_output[1] = FloatBits(y);
  lane.vertex_output[2] = FloatBits(z);
  lane.vertex_output[3] = FloatBits(w);
  lane.emitted = 1;
  lane.ended = 1;
  return lane;
}

VertexLane MakeClipLane(const std::array<std::uint32_t, 6> &outputs) {
  VertexLane lane;
  for (std::size_t component = 0; component < outputs.size(); ++component)
    lane.vertex_output[component] = outputs[component];
  lane.emitted = 1;
  lane.ended = 1;
  return lane;
}

void CheckHomogeneousWClipping() {
  MemoryPool pool;
  PipelineState state;
  state.width = kWidth;
  state.height = kHeight;
  state.sequence = 1;
  state.functional_case = FunctionalCase::kDriverPcoTriangles;
  state.stage = PipelineStage::kVertexShaded;
  state.draw.topology = PrimitiveTopology::kTriangleList;
  state.draw.vertex_count = 3;
  state.draw.index_format = IndexFormat::kNone;
  state.position_output_start = 0;
  state.position_output_count = 4;
  state.raster_state.face_cull.enable = 0;
  state.raster_state.face_cull.mode = CullFaceMode::kBack;
  state.raster_state.face_cull.front_face = FrontFaceWinding::kClockwise;

  // The apex is behind the homogeneous eye plane. It is invalid to divide it
  // by W for pre-clip face classification; the six-plane clipper must first
  // replace the two crossing edges with positive-W intersections.
  state.vertex_lanes = StoreNewArray(
      pool, std::vector<VertexLane>{
                MakeClipLane(0.0F, 0.75F, 0.0F, -0.25F),
                MakeClipLane(-0.75F, -0.75F, 0.0F, 1.0F),
                MakeClipLane(0.75F, -0.75F, 0.0F, 1.0F)});
  state.vertex_lane_refs = StoreNewArray(
      pool, std::vector<VertexLaneRef>{{0, 0}, {1, 1}, {2, 2}});

  const PoolHandle handle = pool.Allocate(sizeof(PipelineState));
  StorePipelineState(pool, handle, state);
  sc_core::sc_fifo<PipelineTxn> input("clip_w_input", 1);
  sc_core::sc_fifo<PipelineTxn> output("clip_w_output", 1);
  ClipCull clip("clip_w", pool);
  clip.input(input);
  clip.output(output);
  input.write(PipelineTxn{handle, 1, 1});
  sc_core::sc_start(sc_core::sc_time(20, sc_core::SC_US));

  PipelineTxn completed;
  Check(output.nb_read(completed) && completed.state.slot == handle.slot &&
            completed.state.generation == handle.generation,
        "homogeneous-W clip completion identity");
  const PipelineState result = LoadPipelineState(pool, handle);
  const std::vector<RasterTriangle> triangles =
      LoadArray<RasterTriangle>(pool, result.raster_triangles);
  Check(result.stage == PipelineStage::kClipCullComplete &&
            result.counters.c_invocations == 1 && !triangles.empty(),
        "negative-W triangle reaches post-clip raster candidates");
  for (const RasterTriangle &triangle : triangles) {
    for (float reciprocal_w : triangle.reciprocal_w)
      Check(reciprocal_w > 0.0F,
            "post-clip raster vertex has positive homogeneous W");
  }

  ReleaseFunctionalPayloads(pool, result);
  pool.Release(handle);
  Check(pool.bytes_in_flight() == 0 &&
            pool.allocations() == pool.releases(),
        "homogeneous-W clip payload ownership cleanup");
}

void CheckDriverViewportFma() {
  MemoryPool pool;
  PipelineState state;
  state.width = 80;
  state.height = 60;
  state.sequence = 1;
  state.functional_case = FunctionalCase::kDriverPcoTriangles;
  state.stage = PipelineStage::kVertexShaded;
  state.draw.topology = PrimitiveTopology::kTriangleList;
  state.draw.vertex_count = 3;
  state.draw.index_format = IndexFormat::kNone;
  state.position_output_start = 0;
  state.position_output_count = 4;
  state.raster_state.face_cull.enable = 0;
  state.raster_state.face_cull.mode = CullFaceMode::kBack;
  state.raster_state.face_cull.front_face = FrontFaceWinding::kClockwise;

  // Captured Refract composite triangle 10301.  The post-VS clip values are
  // exact; only Gallivm's contracted viewport multiply/add produces the
  // public window-space result for vertex zero.  A split multiply/add gives
  // 0x4265e59c/0x41827d64 and must not be reintroduced.
  state.vertex_lanes = StoreNewArray(
      pool, std::vector<VertexLane>{
                MakeClipLane(BitsFloat(0x3fb219f6U),
                             BitsFloat(0xbfba068bU),
                             BitsFloat(0x3f28e456U),
                             BitsFloat(0x404bd83aU)),
                MakeClipLane(BitsFloat(0x3fb4e20aU),
                             BitsFloat(0xbfba22eeU),
                             BitsFloat(0x3f2afed2U),
                             BitsFloat(0x404c1435U)),
                MakeClipLane(BitsFloat(0x3fb2166eU),
                             BitsFloat(0xbfb4f588U),
                             BitsFloat(0x3f2ca067U),
                             BitsFloat(0x404c42b9U))});
  state.vertex_lane_refs = StoreNewArray(
      pool, std::vector<VertexLaneRef>{{0, 0}, {1, 1}, {2, 2}});

  const PoolHandle handle = pool.Allocate(sizeof(PipelineState));
  StorePipelineState(pool, handle, state);
  sc_core::sc_fifo<PipelineTxn> input("viewport_fma_input", 1);
  sc_core::sc_fifo<PipelineTxn> output("viewport_fma_output", 1);
  ClipCull clip("viewport_fma", pool);
  clip.input(input);
  clip.output(output);
  input.write(PipelineTxn{handle, 1, 1});
  sc_core::sc_start(sc_core::sc_time(20, sc_core::SC_US));

  PipelineTxn completed;
  Check(output.nb_read(completed) && completed.state.slot == handle.slot &&
            completed.state.generation == handle.generation,
        "driver viewport-FMA completion identity");
  const PipelineState result = LoadPipelineState(pool, handle);
  const std::vector<RasterTriangle> triangles =
      LoadArray<RasterTriangle>(pool, result.raster_triangles);
  Check(result.stage == PipelineStage::kClipCullComplete &&
            result.counters.c_invocations == 1 && triangles.size() == 1,
        "captured triangle reaches one raster candidate");
  Check(FloatBits(triangles[0].x[0]) == 0x4265e59dU &&
            FloatBits(triangles[0].y[0]) == 0x41827d63U,
        "driver viewport uses exact contracted multiply/add bits");

  ReleaseFunctionalPayloads(pool, result);
  pool.Release(handle);
  Check(pool.bytes_in_flight() == 0 &&
            pool.allocations() == pool.releases(),
        "driver viewport-FMA payload ownership cleanup");
}

void CheckDriverClippedSetupPrecision() {
  MemoryPool pool;
  PipelineState state;
  state.width = 800;
  state.height = 600;
  state.sequence = 1;
  state.functional_case = FunctionalCase::kDriverPcoTriangles;
  state.stage = PipelineStage::kVertexShaded;
  state.draw.topology = PrimitiveTopology::kTriangleList;
  state.draw.vertex_count = 6;
  state.draw.index_format = IndexFormat::kNone;
  state.position_output_start = 0;
  state.position_output_count = 4;
  state.varying_output_start = 4;
  state.varying_output_count = 2;
  state.fragment_position_start = 0;
  state.fragment_position_count = 4;
  state.fragment_varying_start = 4;
  state.fragment_varying_count = 8;
  state.vertex_pco_abi.vertex_outputs = 6;
  state.fragment_pco_abi.coefficients = 12;
  state.raster_state.face_cull.enable = 1;
  state.raster_state.face_cull.mode = CullFaceMode::kBack;
  state.raster_state.face_cull.front_face = FrontFaceWinding::kClockwise;

  ShaderVaryingBinding binding;
  binding.vertex_output_base = 4;
  binding.coefficient_set_base = 1;
  binding.w_coefficient_set = 0;
  binding.component_count = 2;
  binding.interpolation = InterpolationMode::kSmooth;
  state.shader_varying_bindings = StoreNewArray(
      pool, std::vector<ShaderVaryingBinding>{binding});

  // GLMark2 Terrain primitive 56151.  Two vertices cross the right clip
  // plane.  Mesa fan piece one contains two CPU-generated intersections and
  // one original Gallivm vertex; their distinct viewport arithmetic and the
  // clipper's cyclic fan order are both visible in the U/V setup planes.  A
  // second, fully-inside primitive shares the dirty middle-end segment and
  // proves Mesa's per-primitive trivial accept does not cyclically fan-emit
  // an otherwise clean triangle.
  state.vertex_lanes = StoreNewArray(
      pool, std::vector<VertexLane>{
                MakeClipLane({UINT32_C(0x44557020), UINT32_C(0xc4226a58),
                              UINT32_C(0x444f8bbc), UINT32_C(0x44505680),
                              UINT32_C(0x3eda0000), UINT32_C(0x3ea80000)}),
                MakeClipLane({UINT32_C(0x44557020), UINT32_C(0xc41d4219),
                              UINT32_C(0x4457d488), UINT32_C(0x44589d2e),
                              UINT32_C(0x3edc0000), UINT32_C(0x3eaa0000)}),
                MakeClipLane({UINT32_C(0x444ce685), UINT32_C(0xc41ca6c4),
                              UINT32_C(0x44532415), UINT32_C(0x4453edee),
                              UINT32_C(0x3eda0000), UINT32_C(0x3eaa0000)}),
                MakeClipLane({FloatBits(-0.5F), FloatBits(-0.5F),
                              FloatBits(0.0F), FloatBits(1.0F),
                              FloatBits(0.1F), FloatBits(0.4F)}),
                MakeClipLane({FloatBits(0.5F), FloatBits(-0.5F),
                              FloatBits(0.0F), FloatBits(1.0F),
                              FloatBits(0.2F), FloatBits(0.5F)}),
                MakeClipLane({FloatBits(0.0F), FloatBits(0.5F),
                              FloatBits(0.0F), FloatBits(1.0F),
                              FloatBits(0.3F), FloatBits(0.6F)})});
  state.vertex_lane_refs = StoreNewArray(
      pool, std::vector<VertexLaneRef>{{0, 0}, {1, 1}, {2, 2},
                                      {3, 3}, {4, 4}, {5, 5}});

  const PoolHandle handle = pool.Allocate(sizeof(PipelineState));
  StorePipelineState(pool, handle, state);
  sc_core::sc_fifo<PipelineTxn> input("clipped_setup_input", 1);
  sc_core::sc_fifo<PipelineTxn> clip_to_tiler("clipped_setup_clip_tiler", 1);
  sc_core::sc_fifo<PipelineTxn> tiler_to_parameter(
      "clipped_setup_tiler_parameter", 1);
  sc_core::sc_fifo<PipelineTxn> output("clipped_setup_output", 1);
  ClipCull clip("clipped_setup_clip", pool);
  Tiler tiler("clipped_setup_tiler", pool);
  ParameterBuffer parameter("clipped_setup_parameter", pool);
  clip.input(input);
  clip.output(clip_to_tiler);
  tiler.input(clip_to_tiler);
  tiler.output(tiler_to_parameter);
  parameter.input(tiler_to_parameter);
  parameter.output(output);
  input.write(PipelineTxn{handle, 1, 1});
  sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_US));

  PipelineTxn completed;
  Check(output.nb_read(completed) && completed.state.slot == handle.slot &&
            completed.state.generation == handle.generation,
        "clipped Terrain setup completion identity");
  const PipelineState result = LoadPipelineState(pool, handle);
  const std::vector<RasterTriangle> triangles =
      LoadArray<RasterTriangle>(pool, result.raster_triangles);
  const std::vector<ParameterTriangle> parameters =
      LoadArray<ParameterTriangle>(pool, result.parameter_triangles);
  const std::vector<ParameterCoefficientSet> coefficients =
      LoadArray<ParameterCoefficientSet>(pool, result.parameter_coefficients);
  const std::vector<std::uint32_t> raster_outputs =
      LoadArray<std::uint32_t>(pool, result.raster_vertex_outputs);
  Check(result.stage == PipelineStage::kParameterBufferReady &&
            triangles.size() == parameters.size(),
        "clipped Terrain reaches parameter setup");

  std::size_t piece = triangles.size();
  for (std::size_t index = 0; index < triangles.size(); ++index) {
    if (triangles[index].key.api_primitive_id == 0 &&
        triangles[index].key.clip_piece == 1) {
      piece = index;
      break;
    }
  }
  Check(piece < triangles.size() && triangles[piece].rasterizable == 1,
        "Terrain right-clipped fan piece one is rasterizable");
  const RasterTriangle &triangle = triangles[piece];
  Check(FloatBits(triangle.x[0]) == UINT32_C(0x44480000) &&
            FloatBits(triangle.y[0]) == UINT32_C(0x42985d5c) &&
            FloatBits(triangle.x[1]) == UINT32_C(0x4444aef1) &&
            FloatBits(triangle.y[1]) == UINT32_C(0x429c7fcc) &&
            FloatBits(triangle.x[2]) == UINT32_C(0x44480000) &&
            FloatBits(triangle.y[2]) == UINT32_C(0x428e8cc2),
        "generated intersections use CPU viewport while original uses FMA");
  Check(triangle.setup_vertex_order[0] == 2 &&
            triangle.setup_vertex_order[1] == 1 &&
            triangle.setup_vertex_order[2] == 0,
        "Mesa clipped fan setup order survives raster normalization");

  const std::size_t coefficient = parameters[piece].first_coefficient_set;
  Check(coefficient + 2 < coefficients.size(),
        "Terrain fan piece owns W/U/V coefficient sets");
  CheckPlane(coefficients[coefficient], UINT32_C(0xb5a9bfb0),
             UINT32_C(0xb68901ca), UINT32_C(0x3b2347ef),
             "Terrain reciprocal-W plane matches llvmpipe setup");
  CheckPlane(coefficients[coefficient + 1], UINT32_C(0xb47de000),
             UINT32_C(0xb59c39ae), UINT32_C(0x3a4c1f2a),
             "Terrain U/W plane matches llvmpipe setup");
  CheckPlane(coefficients[coefficient + 2], UINT32_C(0xb50c3af1),
             UINT32_C(0xb59c39ae), UINT32_C(0x3a6a11a2),
             "Terrain V/W plane matches llvmpipe setup");

  std::size_t clean = triangles.size();
  for (std::size_t index = 0; index < triangles.size(); ++index) {
    if (triangles[index].key.api_primitive_id == 1) {
      clean = index;
      break;
    }
  }
  Check(clean < triangles.size() && triangles[clean].key.clip_piece == 0 &&
            triangles[clean].rasterizable == 1,
        "clean primitive in dirty segment remains a trivial accept");
  const RasterTriangle &clean_triangle = triangles[clean];
  const auto setup_u = [&](std::size_t setup_vertex) {
    const std::size_t serialized =
        clean_triangle.setup_vertex_order[setup_vertex];
    return raster_outputs.at(
        clean_triangle.first_vertex_output_dword +
        serialized * clean_triangle.vertex_output_stride_dwords + 4);
  };
  Check(setup_u(0) == FloatBits(0.2F) &&
            setup_u(1) == FloatBits(0.1F) &&
            setup_u(2) == FloatBits(0.3F),
        "dirty-segment trivial accept preserves clean Mesa setup order");

  ReleaseFunctionalPayloads(pool, result);
  pool.Release(handle);
  Check(pool.bytes_in_flight() == 0 &&
            pool.allocations() == pool.releases(),
        "clipped Terrain setup payload ownership cleanup");
}

void CheckTerrainSubpixelRoundToEven() {
  // Terrain's shared vertex is exactly 32336.5 in 24.8 space. llvmpipe's
  // _mm_cvtps_epi32 keeps the even endpoint; half-away rounding moves the
  // edge by one subpixel and gives the pixel to the adjacent primitive.
  const float terrain_tie = BitsFloat(UINT32_C(0x42fca100));
  Check(QuantizeRasterSubpixel(terrain_tie) == 32336,
        "Terrain positive even tie rounds down to even");
  Check(QuantizeRasterSubpixel(32337.5F / 256.0F) == 32338,
        "positive odd tie rounds up to even");
  Check(QuantizeRasterSubpixel(-1.5F / 256.0F) == -2 &&
            QuantizeRasterSubpixel(-2.5F / 256.0F) == -2 &&
            QuantizeRasterSubpixel(0.5F / 256.0F) == 0 &&
            QuantizeRasterSubpixel(-0.5F / 256.0F) == 0,
        "negative and signed-zero-adjacent ties round to even");
  Check(QuantizeRasterSubpixel(std::nextafter(
            terrain_tie, -std::numeric_limits<float>::infinity())) == 32336 &&
            QuantizeRasterSubpixel(std::nextafter(
                terrain_tie, std::numeric_limits<float>::infinity())) ==
                32337,
        "values on either side of Terrain tie keep nearest endpoints");
  Check(QuantizeRasterSubpixel(-0x1p55F) ==
            std::numeric_limits<std::int64_t>::min(),
        "negative int64 endpoint remains representable");

  const auto check_overflow = [](float value) {
    try {
      (void)QuantizeRasterSubpixel(value);
    } catch (const std::overflow_error &) {
      return true;
    }
    return false;
  };
  Check(check_overflow(0x1p55F) &&
            check_overflow(std::numeric_limits<float>::max()) &&
            check_overflow(-std::numeric_limits<float>::max()) &&
            check_overflow(std::numeric_limits<float>::infinity()) &&
            check_overflow(-std::numeric_limits<float>::infinity()) &&
            check_overflow(std::numeric_limits<float>::quiet_NaN()),
        "non-finite and out-of-range subpixels fail closed");

  MemoryPool pool;
  PipelineState state;
  state.width = 800;
  state.height = 600;
  state.sequence = 1;
  state.functional_case = FunctionalCase::kDriverPcoTriangles;
  state.stage = PipelineStage::kTiled;
  state.counters.c_primitives = 2;

  const auto make_triangle = [](std::uint32_t primitive,
                                std::array<std::uint32_t, 3> x,
                                std::array<std::uint32_t, 3> y,
                                std::uint32_t output_offset) {
    RasterTriangle triangle;
    triangle.key.submit_ordinal = primitive;
    triangle.key.api_primitive_id = primitive;
    for (std::size_t vertex = 0; vertex < 3; ++vertex) {
      triangle.x[vertex] = BitsFloat(x[vertex]);
      triangle.y[vertex] = BitsFloat(y[vertex]);
      triangle.window_z[vertex] = 0.5F;
      triangle.reciprocal_w[vertex] = 1.0F;
    }
    triangle.first_vertex_output_dword = output_offset;
    triangle.vertex_output_stride_dwords = 4;
    triangle.front_facing = 1;
    triangle.rasterizable = 1;
    triangle.setup_vertex_order[0] = 1;
    triangle.setup_vertex_order[1] = 0;
    triangle.setup_vertex_order[2] = 2;
    return triangle;
  };

  // GLMark2 Terrain primitives 55622 and 55623 share A-C. Native RNE puts
  // sample (628.5,113.5) on primitive 55622's side by 104 fixed units.
  const RasterTriangle neighbor = make_triangle(
      55622,
      {UINT32_C(0x441d59e1), UINT32_C(0x441fea0d),
       UINT32_C(0x441b7edc)},
      {UINT32_C(0x42df7002), UINT32_C(0x42ed4340),
       UINT32_C(0x42fca100)},
      0);
  const RasterTriangle target = make_triangle(
      55623,
      {UINT32_C(0x441d59e1), UINT32_C(0x441b7edc),
       UINT32_C(0x4418ef0b)},
      {UINT32_C(0x42df7002), UINT32_C(0x42fca100),
       UINT32_C(0x42f14eff)},
      12);
  state.raster_triangles = StoreNewArray(
      pool, std::vector<RasterTriangle>{neighbor, target});
  state.raster_vertex_outputs =
      StoreNewArray(pool, std::vector<std::uint32_t>(24));

  const PoolHandle handle = pool.Allocate(sizeof(PipelineState));
  StorePipelineState(pool, handle, state);
  sc_core::sc_fifo<PipelineTxn> input("subpixel_rne_input", 1);
  sc_core::sc_fifo<PipelineTxn> output("subpixel_rne_output", 1);
  ParameterBuffer parameter("subpixel_rne_parameter", pool);
  parameter.input(input);
  parameter.output(output);
  input.write(PipelineTxn{handle, 1, 1});
  sc_core::sc_start(sc_core::sc_time(20, sc_core::SC_US));

  PipelineTxn completed;
  Check(output.nb_read(completed) && completed.state.slot == handle.slot &&
            completed.state.generation == handle.generation,
        "Terrain RNE parameter completion identity");
  const PipelineState result = LoadPipelineState(pool, handle);
  const std::vector<ParameterTriangle> parameters =
      LoadArray<ParameterTriangle>(pool, result.parameter_triangles);
  Check(parameters.size() == 2 && parameters[0].rasterizable == 1 &&
            parameters[1].rasterizable == 1,
        "Terrain shared-edge primitives remain rasterizable");

  constexpr std::int64_t sample_x = 628 * kSubpixelScale +
                                    kSubpixelScale / 2;
  constexpr std::int64_t sample_y = 113 * kSubpixelScale +
                                    kSubpixelScale / 2;
  const auto edge_value = [](const EdgeEquation &edge) {
    return edge.a * sample_x + edge.b * sample_y + edge.c;
  };
  const auto covers = [&](const ParameterTriangle &triangle) {
    for (const EdgeEquation &edge : triangle.edge) {
      const std::int64_t value = edge_value(edge);
      if (value < 0 || (value == 0 && edge.inclusive == 0))
        return false;
    }
    return true;
  };
  Check(edge_value(parameters[0].edge[2]) == 104 &&
            edge_value(parameters[1].edge[0]) == -104 &&
            covers(parameters[0]) && !covers(parameters[1]),
        "RNE shared edge assigns pixel to native primitive 55622");

  ReleaseFunctionalPayloads(pool, result);
  pool.Release(handle);
  Check(pool.bytes_in_flight() == 0 &&
            pool.allocations() == pool.releases(),
        "Terrain RNE payload ownership cleanup");
}

} // namespace

int sc_main(int argc, char **argv) {
  try {
    const bool invalid_count =
        argc == 2 && std::string(argv[1]) == "invalid-count";
    const bool texture_fetch =
        argc == 2 && std::string(argv[1]) == "texture-fetch";
    const bool front_cull =
        argc == 2 && std::string(argv[1]) == "front-cull";
    const bool clip_w = argc == 2 && std::string(argv[1]) == "clip-w";
    const bool viewport_fma =
        argc == 2 && std::string(argv[1]) == "viewport-fma";
    const bool clipped_setup =
        argc == 2 && std::string(argv[1]) == "clipped-setup";
    const bool subpixel_rne =
        argc == 2 && std::string(argv[1]) == "subpixel-rne";
    if (argc > 2 ||
        (argc == 2 && !invalid_count && !texture_fetch && !front_cull &&
         !clip_w && !viewport_fma && !clipped_setup && !subpixel_rne)) {
      throw std::runtime_error(
          "usage: nonindexed-triangle-geometry-test "
          "[invalid-count|texture-fetch|front-cull|clip-w|viewport-fma|"
          "clipped-setup|subpixel-rne]");
    }

    if (texture_fetch) {
      CheckTextureAttributeFetch();
      std::cout << "nonindexed_triangle_geometry_test: PASS "
                   "(texture-fetch)\n";
      return 0;
    }
    if (clip_w) {
      CheckHomogeneousWClipping();
      std::cout << "nonindexed_triangle_geometry_test: PASS (clip-w)\n";
      return 0;
    }
    if (viewport_fma) {
      CheckDriverViewportFma();
      std::cout <<
          "nonindexed_triangle_geometry_test: PASS (viewport-fma)\n";
      return 0;
    }
    if (clipped_setup) {
      CheckDriverClippedSetupPrecision();
      std::cout <<
          "nonindexed_triangle_geometry_test: PASS (clipped-setup)\n";
      return 0;
    }
    if (subpixel_rne) {
      CheckTerrainSubpixelRoundToEven();
      std::cout <<
          "nonindexed_triangle_geometry_test: PASS (subpixel-rne)\n";
      return 0;
    }

    Check(FunctionalCaseFromName("driver_pco_triangles") ==
              FunctionalCase::kDriverPcoTriangles,
          "functional case mapping");
    Check(std::string(FunctionalCaseName(
              FunctionalCase::kDriverPcoTriangles)) ==
              "driver_pco_triangles" &&
              IsDriverPcoTrianglesCase(
                  FunctionalCase::kDriverPcoTriangles) &&
              IsSolidColorRasterCase(
                  FunctionalCase::kDriverPcoTriangles) &&
              IsRasterFunctionalCase(
                  FunctionalCase::kDriverPcoTriangles) &&
              !IsIndexedTriangleRasterCase(
                  FunctionalCase::kDriverPcoTriangles) &&
              !UsesShaderVaryings(
                  FunctionalCase::kDriverPcoTriangles),
          "functional family classification");

    MemoryPool pool;
    const std::vector<float> vertices = MakeFloat3Vertices();
    Check(vertices.size() == 3U * (kFirstVertex + kVertexCount),
          "float3 fixture size");

    PipelineState state;
    state.width = kWidth;
    state.height = kHeight;
    state.sequence = 1;
    state.functional_case = FunctionalCase::kDriverPcoTriangles;
    state.stage = PipelineStage::kSubmitted;
    state.draw.topology = PrimitiveTopology::kTriangleList;
    state.draw.first_vertex = kFirstVertex;
    state.draw.vertex_count = invalid_count ? kVertexCount - 1 : kVertexCount;
    state.draw.index_format = IndexFormat::kNone;
    state.position_output_start = 0;
    state.position_output_count = 4;
    // Gallium reports the captured GL draw as BACK/CW after its window-Y
    // convention.  The model keeps NDC +Y upward, so the driver PCO path must
    // account for that reflection without culling these CCW model triangles.
    state.raster_state.face_cull.enable = 1;
    state.raster_state.face_cull.mode =
        front_cull ? CullFaceMode::kFront : CullFaceMode::kBack;
    state.raster_state.face_cull.front_face = FrontFaceWinding::kClockwise;

    VertexBufferResource vertex_resource;
    vertex_resource.data = StoreNewArray(pool, vertices);
    vertex_resource.byte_size =
        static_cast<std::uint32_t>(vertices.size() * sizeof(float));
    state.vertex_buffer_resources = StoreNewArray(
        pool, std::vector<VertexBufferResource>{vertex_resource});
    VertexAttributeBinding position_binding;
    position_binding.stride_bytes = 3U * sizeof(float);
    position_binding.destination_register = 0;
    position_binding.component_type = VertexComponentType::kFloat32;
    position_binding.source_components = 3;
    position_binding.destination_components = 4;
    state.vertex_attribute_bindings = StoreNewArray(
        pool, std::vector<VertexAttributeBinding>{position_binding});
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
    sc_core::sc_fifo<PipelineTxn> tiler_to_parameter(
        "tiler_to_parameter", 1);
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
    if (invalid_count) {
      try {
        sc_core::sc_start(sc_core::sc_time(20, sc_core::SC_US));
      } catch (const std::exception &error) {
        Check(std::string(error.what()).find(
                  "vertex_count_not_a_triangle_multiple") !=
                  std::string::npos,
              std::string("unexpected fail-closed diagnostic: ") +
                  error.what());
        const PipelineState failed =
            LoadPipelineState(pool, state_handle);
        ReleaseFunctionalPayloads(pool, failed);
        pool.Release(state_handle);
        Check(pool.bytes_in_flight() == 0 &&
                  pool.allocations() == pool.releases(),
              "invalid-count cleanup balances payload ownership");
        std::cout << "nonindexed_triangle_geometry_test: PASS "
                     "(invalid-count)\n";
        return 0;
      }
      throw std::runtime_error(
          "non-multiple-of-three draw unexpectedly reached completion");
    }

    sc_core::sc_start(sc_core::sc_time(20, sc_core::SC_US));

    PipelineTxn completed;
    Check(output.nb_read(completed) && completed.sequence == 1,
          "event-driven geometry FIFO completion");
    const PipelineState result = LoadPipelineState(pool, state_handle);
    Check(result.stage == PipelineStage::kParameterBufferReady,
          "parameter-buffer completion stage");
    Check(result.draw.index_format == IndexFormat::kNone &&
              result.draw.index_count == 0 &&
              !HasPoolHandle(result.vertex_indices) &&
              result.index_buffer_gpu_address == 0 &&
              result.index_buffer_bytes == 0,
          "non-indexed draw remains free of synthetic index state");
    Check(result.counters.ia_vertices == kVertexCount &&
              result.counters.ia_primitives == kPrimitiveCount &&
              result.counters.vs_invocations == kVertexCount &&
              result.counters.vertex_attribute_fetches == kVertexCount &&
              result.counters.vertex_attribute_bytes ==
                  kVertexCount * 3U * sizeof(float),
          "data-derived IA/VS/float3 fetch counters");

    const std::vector<VertexLane> lanes =
        LoadArray<VertexLane>(pool, result.vertex_lanes);
    const std::vector<VertexLaneRef> lane_refs =
        LoadArray<VertexLaneRef>(pool, result.vertex_lane_refs);
    Check(lanes.size() == kVertexCount &&
              lane_refs.size() == kVertexCount,
          "one direct lane/reference per vertex occurrence");
    for (std::size_t occurrence = 0; occurrence < lane_refs.size();
         ++occurrence) {
      const std::uint32_t source_vertex =
          kFirstVertex + static_cast<std::uint32_t>(occurrence);
      const VertexLaneRef &ref = lane_refs[occurrence];
      Check(ref.lane_index == occurrence &&
                ref.vertex_index == source_vertex,
            "sequential lane reference identity");
      const std::size_t source = 3U * source_vertex;
      Check(lanes[occurrence].vertex_input[0] ==
                    FloatBits(vertices[source]) &&
                lanes[occurrence].vertex_input[1] ==
                    FloatBits(vertices[source + 1]) &&
                lanes[occurrence].vertex_input[2] ==
                    FloatBits(vertices[source + 2]) &&
                lanes[occurrence].vertex_input[3] == FloatBits(1.0F),
            "float3 fetch and default W");
    }

    const std::vector<RasterTriangle> triangles =
        LoadArray<RasterTriangle>(pool, result.raster_triangles);
    const std::vector<ParameterTriangle> parameters =
        LoadArray<ParameterTriangle>(pool, result.parameter_triangles);
    Check(result.counters.c_invocations == kPrimitiveCount &&
              result.counters.c_primitives == kPrimitiveCount &&
              result.counters.setup_triangles == kPrimitiveCount &&
              triangles.size() == kPrimitiveCount &&
              parameters.size() == kPrimitiveCount,
          "three non-indexed triangles traverse clip/setup");
    for (std::size_t primitive = 0; primitive < triangles.size();
         ++primitive) {
      Check(triangles[primitive].key.api_primitive_id == primitive &&
                triangles[primitive].key.submit_ordinal == primitive &&
                triangles[primitive].rasterizable ==
                    (front_cull ? 0 : 1) &&
                triangles[primitive].face_culled ==
                    (front_cull ? 1 : 0),
            "window-reflected primitive identity and cull result");
      Check(HasCanonicalDepthPlaneMetadata(state.functional_case,
                                           parameters[primitive]),
            "driver depth-plane metadata is canonical");
      if (!front_cull) {
        Check(parameters[primitive].depth_plane_valid == 1 &&
                  parameters[primitive].depth_plane[0] == FloatBits(0.0F) &&
                  parameters[primitive].depth_plane[1] == FloatBits(0.0F) &&
                  parameters[primitive].depth_plane[2] == FloatBits(0.5F) &&
                  parameters[primitive].depth_plane[3] == 0,
              "constant position-Z uses exact llvmpipe A/B/C/PAD plane");
        ParameterTriangle malformed = parameters[primitive];
        malformed.depth_plane_valid = 0;
        Check(!HasCanonicalDepthPlaneMetadata(state.functional_case,
                                              malformed),
              "missing driver depth plane fails closed");
        malformed = parameters[primitive];
        malformed.depth_plane[3] = 1;
        Check(!HasCanonicalDepthPlaneMetadata(state.functional_case,
                                              malformed),
              "nonzero driver depth-plane PAD fails closed");
      }
    }

    ReleaseFunctionalPayloads(pool, result);
    pool.Release(state_handle);
    Check(pool.bytes_in_flight() == 0 &&
              pool.allocations() == pool.releases(),
          "payload ownership cleanup");
    std::cout << "nonindexed_triangle_geometry_test: PASS"
              << (front_cull ? " (front-cull)" : "") << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
