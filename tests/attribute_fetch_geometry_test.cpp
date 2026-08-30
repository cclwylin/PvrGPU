/*
 * GLBench AttributeFetchShader case-1/case-2/case-4/case-8
 * geometry/vertex-input regression.
 *
 * 中文：鎖定官方 64x64 CreateLattice/CreateMesh fixture 的 tightly-packed
 * float2 VBO 與 swath-order uint16 EBO，再沿 event-driven
 * VDM（Vertex Data Master）→VertexFetch→PCO（PowerVR Compiler Output）
 * ISS（Instruction Set Simulator）→ClipCull→Tiler→ParameterBuffer
 * 硬體路徑執行。預期 counter 由 fixture、segment 與 direct-mapped
 * post-transform cache 演算法推導；不由 case 名稱或 runtime counter 反推。
 */
#include "common/functional_types.h"
#include "common/glbench_triangle_fixture.h"
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
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace pvrgpu::stub;

inline constexpr std::uint32_t kWidth = 64;
inline constexpr std::uint32_t kHeight = 64;
inline constexpr std::uint64_t kVertexBufferGpuAddress =
    UINT64_C(0x10000000);
inline constexpr std::uint64_t kFnv1aOffsetBasis =
    UINT64_C(14695981039346656037);
inline constexpr std::uint64_t kFnv1aPrime = UINT64_C(1099511628211);

void Check(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(
        "attribute-fetch geometry test failed: " + message);
  }
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void AppendLittleEndian(std::vector<std::uint8_t> &bytes,
                        std::uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void AppendLittleEndian(std::vector<std::uint8_t> &bytes,
                        std::uint16_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8U));
}

std::uint64_t Fnv1a64(const std::vector<std::uint8_t> &bytes) {
  std::uint64_t hash = kFnv1aOffsetBasis;
  for (const std::uint8_t byte : bytes) {
    hash ^= byte;
    hash *= kFnv1aPrime;
  }
  return hash;
}

std::vector<std::uint8_t>
LittleEndianVertexBytes(const std::vector<float> &vertices) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(vertices.size() * sizeof(float));
  for (const float component : vertices)
    AppendLittleEndian(bytes, FloatBits(component));
  return bytes;
}

std::vector<std::uint8_t>
LittleEndianIndexBytes(const std::vector<std::uint16_t> &indices) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(indices.size() * sizeof(std::uint16_t));
  for (const std::uint16_t index : indices)
    AppendLittleEndian(bytes, index);
  return bytes;
}

std::uint64_t
DeriveVertexShaderInvocations(const std::vector<std::uint16_t> &indices) {
  Check(kReferenceUarch.index_segment_max_indices != 0 &&
            kReferenceUarch.index_segment_max_indices % 3 == 0 &&
            kReferenceUarch.post_transform_cache_slots != 0,
        "valid reference-uArch index segmentation/cache geometry");

  struct CacheEntry {
    std::uint32_t vertex_index = 0;
    bool valid = false;
  };
  std::uint64_t invocations = 0;
  std::size_t occurrence = 0;
  while (occurrence < indices.size()) {
    const std::size_t segment_count = std::min<std::size_t>(
        kReferenceUarch.index_segment_max_indices,
        indices.size() - occurrence);
    Check(segment_count % 3 == 0,
          "reference-uArch segment preserves complete triangles");
    std::vector<CacheEntry> cache(
        kReferenceUarch.post_transform_cache_slots);
    const std::size_t segment_end = occurrence + segment_count;
    for (; occurrence < segment_end; ++occurrence) {
      const std::uint32_t vertex_index = indices[occurrence];
      CacheEntry &entry = cache[vertex_index % cache.size()];
      if (!entry.valid || entry.vertex_index != vertex_index) {
        entry.vertex_index = vertex_index;
        entry.valid = true;
        ++invocations;
      }
    }
  }
  return invocations;
}

enum class TestMode {
  kOneAttribute,
  kTwoAttribute,
  kFourAttribute,
  kEightAttribute,
  kInvalidOverlap,
  kInvalidStride,
  kInvalidOffset,
};

TestMode ParseMode(int argc, char **argv) {
  if (argc == 1 || (argc == 2 && std::string(argv[1]) == "one"))
    return TestMode::kOneAttribute;
  if (argc == 2 && std::string(argv[1]) == "two")
    return TestMode::kTwoAttribute;
  if (argc == 2 && std::string(argv[1]) == "four")
    return TestMode::kFourAttribute;
  if (argc == 2 && std::string(argv[1]) == "eight")
    return TestMode::kEightAttribute;
  if (argc == 2 && std::string(argv[1]) == "invalid-overlap")
    return TestMode::kInvalidOverlap;
  if (argc == 2 && std::string(argv[1]) == "invalid-stride")
    return TestMode::kInvalidStride;
  if (argc == 2 && std::string(argv[1]) == "invalid-offset")
    return TestMode::kInvalidOffset;
  throw std::runtime_error(
      "usage: attribute-fetch-geometry-test "
      "[one|two|four|eight|invalid-overlap|invalid-stride|invalid-offset]");
}

bool IsInvalidMode(TestMode mode) {
  return mode == TestMode::kInvalidOverlap ||
         mode == TestMode::kInvalidStride ||
         mode == TestMode::kInvalidOffset;
}

std::size_t AttributeCount(TestMode mode) {
  if (mode == TestMode::kOneAttribute)
    return 1;
  if (mode == TestMode::kFourAttribute)
    return 4;
  if (mode == TestMode::kEightAttribute)
    return 8;
  // The fail-closed layout modes deliberately mutate case-2's second
  // binding, so they remain two-attribute tests.
  return 2;
}

const char *AttributeCountName(std::size_t attribute_count) {
  if (attribute_count == 1)
    return "one";
  if (attribute_count == 2)
    return "two";
  if (attribute_count == 4)
    return "four";
  if (attribute_count == 8)
    return "eight";
  throw std::runtime_error("unsupported attribute count");
}

VertexAttributeBinding MakePositionBinding(
    std::uint16_t destination_register,
    std::uint8_t destination_components) {
  VertexAttributeBinding binding;
  binding.buffer_index = 0;
  binding.offset_bytes = 0;
  binding.stride_bytes = 2U * sizeof(float);
  binding.destination_register = destination_register;
  binding.component_type = VertexComponentType::kFloat32;
  binding.source_components = 2;
  binding.destination_components = destination_components;
  binding.normalized = 0;
  binding.integer = 0;
  binding.instance_divisor = 0;
  return binding;
}

void CheckFixture(const std::vector<float> &vertices,
                  const std::vector<std::uint16_t> &indices) {
  const std::vector<std::uint8_t> vertex_bytes =
      LittleEndianVertexBytes(vertices);
  const std::vector<std::uint8_t> index_bytes =
      LittleEndianIndexBytes(indices);
  Check(vertices.size() == 2U * 65U * 65U &&
            vertex_bytes.size() == 33800,
        "exact tightly-packed float2 VBO size");
  Check(indices.size() == kGlbenchAttributeFetchIndexCount &&
            index_bytes.size() == 49152,
        "exact uint16 EBO size");
  Check(Fnv1a64(vertex_bytes) == UINT64_C(0x9298f09d4e5cf1c5),
        "pinned little-endian VBO bytes");
  Check(Fnv1a64(index_bytes) == UINT64_C(0xbad4b27e865b8735),
        "pinned little-endian EBO bytes");

  Check(FloatBits(vertices.front()) == UINT32_C(0xbf800000) &&
            FloatBits(vertices[1]) == UINT32_C(0xbf800000) &&
            FloatBits(vertices[vertices.size() - 2]) ==
                UINT32_C(0x3f800000) &&
            FloatBits(vertices.back()) == UINT32_C(0x3f800000),
        "lattice spans exact clip-space corners");
  Check(indices.size() % 6 == 0, "two triangles per lattice cell");
  for (std::size_t offset = 0; offset < indices.size(); offset += 6) {
    const std::uint16_t first = indices[offset];
    const std::uint16_t third = indices[offset + 1];
    const std::uint16_t second = indices[offset + 2];
    const std::uint16_t fourth = indices[offset + 3];
    Check(second == first + 1 &&
              third == first + kGlbenchAttributeFetchMeshWidth + 1 &&
              fourth == third + 1 && indices[offset + 4] == second &&
              indices[offset + 5] == third,
          "all-clockwise GLBench cell topology");
  }
}

} // namespace

int sc_main(int argc, char **argv) {
  try {
    const TestMode mode = ParseMode(argc, argv);
    const std::size_t attribute_count = AttributeCount(mode);
    const FunctionalCase functional_case =
        attribute_count == 1
            ? FunctionalCase::kAttributeFetchShader
            : attribute_count == 2
                  ? FunctionalCase::kAttributeFetchShaderTwoAttribute
                  : attribute_count == 4
                        ? FunctionalCase::kAttributeFetchShaderFourAttribute
                        : FunctionalCase::kAttributeFetchShaderEightAttribute;
    Check(FunctionalCaseFromName("attribute_fetch_shader") ==
              FunctionalCase::kAttributeFetchShader,
          "functional case mapping");
    Check(std::string(
              FunctionalCaseName(FunctionalCase::kAttributeFetchShader)) ==
              "attribute_fetch_shader",
          "functional case reverse mapping");
    Check(IsAttributeFetchFamily(FunctionalCase::kAttributeFetchShader) &&
              IsIndexedTriangleRasterCase(
                  FunctionalCase::kAttributeFetchShader) &&
              RequiresBackCcwFaceCull(
                  FunctionalCase::kAttributeFetchShader),
          "attribute-fetch case routing");
    Check(FunctionalCaseFromName("attribute_fetch_shader_2_attr") ==
              FunctionalCase::kAttributeFetchShaderTwoAttribute &&
              std::string(FunctionalCaseName(
                  FunctionalCase::kAttributeFetchShaderTwoAttribute)) ==
                  "attribute_fetch_shader_2_attr" &&
              IsAttributeFetchFamily(
                  FunctionalCase::kAttributeFetchShaderTwoAttribute) &&
              IsIndexedTriangleRasterCase(
                  FunctionalCase::kAttributeFetchShaderTwoAttribute) &&
              RequiresBackCcwFaceCull(
                  FunctionalCase::kAttributeFetchShaderTwoAttribute),
          "two-attribute case mapping/routing");
    Check(FunctionalCaseFromName("attribute_fetch_shader_4_attr") ==
                  FunctionalCase::kAttributeFetchShaderFourAttribute &&
              std::string(FunctionalCaseName(
                  FunctionalCase::kAttributeFetchShaderFourAttribute)) ==
                  "attribute_fetch_shader_4_attr" &&
              IsAttributeFetchFamily(
                  FunctionalCase::kAttributeFetchShaderFourAttribute) &&
              IsIndexedTriangleRasterCase(
                  FunctionalCase::kAttributeFetchShaderFourAttribute) &&
              RequiresBackCcwFaceCull(
                  FunctionalCase::kAttributeFetchShaderFourAttribute),
          "four-attribute case mapping/routing");
    Check(FunctionalCaseFromName("attribute_fetch_shader_8_attr") ==
                  FunctionalCase::kAttributeFetchShaderEightAttribute &&
              std::string(FunctionalCaseName(
                  FunctionalCase::kAttributeFetchShaderEightAttribute)) ==
                  "attribute_fetch_shader_8_attr" &&
              IsAttributeFetchFamily(
                  FunctionalCase::kAttributeFetchShaderEightAttribute) &&
              IsIndexedTriangleRasterCase(
                  FunctionalCase::kAttributeFetchShaderEightAttribute) &&
              RequiresBackCcwFaceCull(
                  FunctionalCase::kAttributeFetchShaderEightAttribute),
          "eight-attribute case mapping/routing");

    MemoryPool pool;
    const std::vector<float> vertices = MakeGlbenchTriangleFloat2Vertices(
        kWidth, kHeight, kGlbenchAttributeFetchMesh);
    const std::vector<std::uint16_t> indices = MakeGlbenchTriangleIndices(
        GlbenchTriangleWindingPattern::kAllClockwise,
        kGlbenchAttributeFetchMesh);
    CheckFixture(vertices, indices);

    const std::uint64_t expected_ia_vertices = indices.size();
    const std::uint64_t expected_ia_primitives = indices.size() / 3U;
    const std::uint64_t expected_vs_invocations =
        DeriveVertexShaderInvocations(indices);
    std::vector<VertexAttributeBinding> bindings;
    if (attribute_count == 1) {
      bindings = {MakePositionBinding(0, 4)};
    } else {
      bindings.reserve(attribute_count);
      for (std::size_t index = 0; index < attribute_count; ++index) {
        bindings.push_back(MakePositionBinding(
            static_cast<std::uint16_t>(index * 2U), 2));
      }
    }
    if (mode == TestMode::kInvalidOverlap)
      bindings[1].destination_register = 1;
    else if (mode == TestMode::kInvalidStride)
      bindings[1].stride_bytes = sizeof(float);
    else if (mode == TestMode::kInvalidOffset)
      bindings[1].offset_bytes = 33800U - sizeof(float);
    const std::uint64_t expected_attribute_fetches =
        expected_vs_invocations * bindings.size();
    std::uint64_t source_components_per_vertex = 0;
    for (const VertexAttributeBinding &binding : bindings)
      source_components_per_vertex += binding.source_components;
    const std::uint64_t expected_attribute_bytes =
        expected_vs_invocations * source_components_per_vertex * sizeof(float);
    const std::uint64_t pinned_attribute_fetches =
        attribute_count == 1
            ? UINT64_C(5317)
            : attribute_count == 2
                  ? UINT64_C(10634)
                  : attribute_count == 4 ? UINT64_C(21268)
                                         : UINT64_C(42536);
    const std::uint64_t pinned_attribute_bytes =
        attribute_count == 1
            ? UINT64_C(42536)
            : attribute_count == 2
                  ? UINT64_C(85072)
                  : attribute_count == 4 ? UINT64_C(170144)
                                         : UINT64_C(340288);
    const std::uint64_t pinned_vertex_fetch_cycles =
        attribute_count == 1
            ? UINT64_C(25)
            : attribute_count == 2
                  ? UINT64_C(46)
                  : attribute_count == 4 ? UINT64_C(88) : UINT64_C(171);
    Check(expected_ia_vertices == 24576 &&
              expected_ia_primitives == 8192 &&
              expected_vs_invocations == 5317 &&
              expected_attribute_fetches == pinned_attribute_fetches &&
              expected_attribute_bytes == pinned_attribute_bytes,
          "fixture/uArch derivation matches pinned llvmpipe Golden counters");

    PipelineState state;
    state.width = kWidth;
    state.height = kHeight;
    state.sequence = 1;
    state.functional_case = functional_case;
    state.stage = PipelineStage::kSubmitted;
    state.draw.topology = PrimitiveTopology::kTriangleList;
    state.draw.first_index = 0;
    state.draw.index_count = static_cast<std::uint32_t>(indices.size());
    state.draw.base_vertex = 0;
    state.draw.index_format = IndexFormat::kUint16;
    state.raster_state.clear_color[0] = 0.0F;
    state.raster_state.clear_color[1] = 1.0F;
    state.raster_state.clear_color[2] = 0.0F;
    state.raster_state.clear_color[3] = 1.0F;
    state.raster_state.face_cull.enable = 1;
    state.raster_state.face_cull.mode = CullFaceMode::kBack;
    state.raster_state.face_cull.front_face =
        FrontFaceWinding::kCounterClockwise;

    VertexBufferResource vertex_resource;
    vertex_resource.data = StoreNewArray(pool, vertices);
    vertex_resource.gpu_address = kVertexBufferGpuAddress;
    vertex_resource.byte_size =
        static_cast<std::uint32_t>(vertices.size() * sizeof(float));
    state.vertex_buffer_resources = StoreNewArray(
        pool, std::vector<VertexBufferResource>{vertex_resource});
    state.vertex_attribute_bindings = StoreNewArray(pool, bindings);
    state.vertex_indices = StoreNewArray(pool, indices);
    const std::vector<std::uint8_t> &vertex_binary =
        attribute_count == 1
            ? AttributeFetchVertexPcoBinary()
            : attribute_count == 2
                  ? AttributeFetchTwoAttributeVertexPcoBinary()
                  : attribute_count == 4
                        ? AttributeFetchFourAttributeVertexPcoBinary()
                        : AttributeFetchEightAttributeVertexPcoBinary();
    state.vertex_code = StoreNewArray(pool, vertex_binary);
    state.fragment_code =
        StoreNewArray(pool, AttributeFetchGrayFragmentPcoBinary());
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
    if (IsInvalidMode(mode)) {
      const char *expected_error =
          mode == TestMode::kInvalidOverlap
              ? "destination registers overlap"
              : mode == TestMode::kInvalidStride
              ? "stride is too small"
              : "starts outside its VBO";
      try {
        sc_core::sc_start(sc_core::sc_time(20, sc_core::SC_US));
      } catch (const std::exception &error) {
        Check(std::string(error.what()).find(expected_error) !=
                  std::string::npos,
              std::string("unexpected fail-closed diagnostic: ") +
                  error.what());
        const PipelineState failed =
            LoadPipelineState(pool, state_handle);
        ReleaseFunctionalPayloads(pool, failed);
        pool.Release(state_handle);
        Check(pool.bytes_in_flight() == 0 &&
                  pool.allocations() == pool.releases(),
              "invalid layout cleanup balances shared VBO ownership");
        std::cout << "attribute_fetch_geometry_test: PASS (fail-closed "
                  << argv[1] << ")\n";
        return 0;
      }
      throw std::runtime_error(
          "invalid vertex input layout unexpectedly reached completion");
    }
    sc_core::sc_start(sc_core::sc_time(20, sc_core::SC_US));

    PipelineTxn completed;
    Check(output.nb_read(completed) && completed.sequence == 1,
          "event-driven geometry FIFO completion");
    const PipelineState result = LoadPipelineState(pool, state_handle);
    Check(result.stage == PipelineStage::kParameterBufferReady,
          "parameter-buffer completion stage");
    Check(result.counters.ia_vertices == expected_ia_vertices &&
              result.counters.ia_primitives == expected_ia_primitives &&
              result.counters.vs_invocations == expected_vs_invocations &&
              result.counters.c_invocations == expected_ia_primitives,
          "derived IA/VS/clip counters");
    Check(result.counters.vertex_attribute_fetches ==
                  expected_attribute_fetches &&
              result.counters.vertex_attribute_bytes ==
                  expected_attribute_bytes &&
              result.counters.vertex_fetch_cycles ==
                  pinned_vertex_fetch_cycles,
          "cache-miss-derived float2 attribute traffic");
    Check(result.counters.c_primitives == expected_ia_primitives &&
              result.counters.setup_triangles == expected_ia_primitives,
          "all clean triangles reach fixed setup before face cull");

    const std::vector<VertexBufferResource> resources =
        LoadArray<VertexBufferResource>(pool,
                                        result.vertex_buffer_resources);
    const std::vector<VertexAttributeBinding> result_bindings =
        LoadArray<VertexAttributeBinding>(pool,
                                          result.vertex_attribute_bindings);
    Check(resources.size() == 1 && resources[0].byte_size == 33800 &&
              result_bindings.size() == bindings.size(),
          "one shared 33,800-byte VBO resource owns every attribute");
    for (std::size_t index = 0; index < result_bindings.size(); ++index) {
      Check(result_bindings[index].buffer_index == 0 &&
                result_bindings[index].destination_register ==
                    (attribute_count == 1 ? 0U : index * 2U) &&
                result_bindings[index].source_components == 2 &&
                result_bindings[index].destination_components ==
                    (attribute_count == 1 ? 4 : 2),
            "attribute bindings alias one VBO with exact VTXIN ranges");
    }

    const std::vector<VertexLane> lanes =
        LoadArray<VertexLane>(pool, result.vertex_lanes);
    const std::vector<VertexLaneRef> lane_refs =
        LoadArray<VertexLaneRef>(pool, result.vertex_lane_refs);
    Check(lanes.size() == expected_vs_invocations &&
              lane_refs.size() == expected_ia_vertices,
          "post-transform lane/reuse payload sizes");
    for (std::size_t occurrence = 0; occurrence < lane_refs.size();
         ++occurrence) {
      const VertexLaneRef &ref = lane_refs[occurrence];
      Check(ref.vertex_index == indices[occurrence] &&
                ref.lane_index < lanes.size(),
            "lane ref preserves EBO identity");
      const std::size_t source = 2U * ref.vertex_index;
      const VertexLane &lane = lanes[ref.lane_index];
      if (attribute_count > 1) {
        for (std::size_t attribute = 0; attribute < attribute_count;
             ++attribute) {
          Check(lane.vertex_input[attribute * 2U] ==
                        FloatBits(vertices[source]) &&
                    lane.vertex_input[attribute * 2U + 1U] ==
                        FloatBits(vertices[source + 1]),
                "aliased attributes fetch one float2 VBO into exact VTXIN "
                "ranges");
        }
        // Match the public PCO TEMP read/modify/write FADD chain one
        // instruction at a time.  This intentionally is not x*count: each
        // IEEE-754 single-precision addition rounds before the next one.
        float expected_x = vertices[source];
        float expected_y = vertices[source + 1];
        for (std::size_t attribute = 1; attribute < attribute_count;
             ++attribute) {
          expected_x += vertices[source];
          expected_y += vertices[source + 1];
        }
        Check(lane.vertex_output[0] == FloatBits(expected_x) &&
                  lane.vertex_output[1] == FloatBits(expected_y) &&
                  lane.vertex_output[2] == FloatBits(0.0F) &&
                  lane.vertex_output[3] ==
                      FloatBits(static_cast<float>(attribute_count)) &&
                  lane.emitted != 0 && lane.ended != 0,
              "aliased-attribute PCO FADD exports exact summed position");
      } else {
        Check(lane.vertex_input[0] == FloatBits(vertices[source]) &&
                  lane.vertex_input[1] == FloatBits(vertices[source + 1]) &&
                  lane.vertex_input[2] == FloatBits(0.0F) &&
                  lane.vertex_input[3] == FloatBits(1.0F),
              "float2 fetch plus GLES z/w defaults");
        Check(lane.vertex_output[0] == lane.vertex_input[0] &&
                  lane.vertex_output[1] == lane.vertex_input[1] &&
                  lane.vertex_output[2] == FloatBits(0.0F) &&
                  lane.vertex_output[3] == FloatBits(1.0F) &&
                  lane.emitted != 0 && lane.ended != 0,
              "attribute PCO ISS exports exact clip position");
      }
    }

    const std::vector<RasterTriangle> triangles =
        LoadArray<RasterTriangle>(pool, result.raster_triangles);
    Check(triangles.size() == expected_ia_primitives,
          "one fixed-setup slot per API triangle");
    for (std::size_t primitive = 0; primitive < triangles.size();
         ++primitive) {
      const RasterTriangle &triangle = triangles[primitive];
      Check(triangle.key.api_primitive_id == primitive &&
                triangle.key.submit_ordinal == primitive &&
                triangle.front_facing == 0 &&
                triangle.face_culled != 0 &&
                triangle.rasterizable == 0,
            "BACK/CCW rejects every clockwise setup candidate");
    }

    const std::vector<TileRecord> tiles =
        LoadArray<TileRecord>(pool, result.tile_records);
    const std::vector<TilePrimitiveRef> refs =
        LoadArray<TilePrimitiveRef>(pool, result.tile_primitive_refs);
    const std::vector<ParameterTriangle> parameters =
        LoadArray<ParameterTriangle>(pool, result.parameter_triangles);
    Check(tiles.size() == 4 && refs.empty(),
          "32x32 tile grid receives no culled primitive refs");
    Check(parameters.size() == expected_ia_primitives,
          "parameter slots preserve all setup identities");
    for (std::size_t primitive = 0; primitive < parameters.size();
         ++primitive) {
      const ParameterTriangle &parameter_slot = parameters[primitive];
      Check(parameter_slot.key.api_primitive_id == primitive &&
                parameter_slot.face_culled != 0 &&
                parameter_slot.rasterizable == 0 &&
                parameter_slot.signed_area == 0,
            "face-culled parameter placeholder");
    }

    const std::vector<DrawListStats> drawlists =
        LoadArray<DrawListStats>(pool, result.drawlist_stats);
    const std::uint64_t expected_program_groups =
        attribute_count == 8
            ? UINT64_C(18)
            : attribute_count == 4 ? UINT64_C(10) : UINT64_C(6);
    const std::uint64_t expected_program_alu =
        attribute_count == 8
            ? UINT64_C(16)
            : attribute_count == 4 ? UINT64_C(8) : UINT64_C(4);
    Check(drawlists.size() == 1 &&
              drawlists[0].vertex.program_recorded == 1 &&
              drawlists[0].vertex.executions_recorded == 1 &&
              drawlists[0].vertex.invocations == expected_vs_invocations &&
              drawlists[0].vertex.program_groups == expected_program_groups &&
              drawlists[0].vertex.program_instructions ==
                  expected_program_groups &&
              drawlists[0].vertex.program_alu_instructions ==
                  expected_program_alu &&
              drawlists[0].vertex.program_tex_instructions == 0 &&
              drawlists[0].vertex.program_memory_instructions == 2 &&
              drawlists[0].vertex.executed_alu_instructions ==
                  expected_vs_invocations * expected_program_alu &&
              drawlists[0].vertex.executed_tex_instructions == 0 &&
              drawlists[0].vertex.executed_memory_instructions ==
                  expected_vs_invocations * 5,
          "DrawList vertex PCO accounting follows executed lanes");

    ReleaseFunctionalPayloads(pool, result);
    pool.Release(state_handle);
    Check(pool.bytes_in_flight() == 0 &&
              pool.allocations() == pool.releases(),
          "MemoryPool resources and nested VBO ownership balanced");
    std::cout << "attribute_fetch_geometry_test: PASS ("
              << AttributeCountName(attribute_count) << ")\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "attribute_fetch_geometry_test: FAIL: "
              << error.what() << '\n';
    return 1;
  }
}
