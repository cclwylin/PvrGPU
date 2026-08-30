/*
 * Triangle.Setup.HalfCulled deterministic geometry regression.
 * 中文：先驗證 pinned Darwin srand(0) sequence、GLBench j+=4→i→j2
 * mixed-winding uint16 index bytes，再走完整 event-driven
 * VDM（Vertex Data Master）→VertexFetch→PCO（PowerVR Compiler Output）
 * ISS（Instruction Set Simulator）→ClipCull→Tiler→ParameterBuffer 路徑。
 * 測試不執行 fragment shader；6797 setup 與 2044 visible cell expectation
 * 都由 index/winding、segment clip/cull 及 24.8 setup 資料計算。
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
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace pvrgpu::stub;

constexpr std::uint32_t kWidth = 64;
constexpr std::uint32_t kHeight = 64;
constexpr std::uint32_t kExpectedPrimitiveCount = 32768;
constexpr std::uint32_t kExpectedVertexInvocations = 21144;
constexpr std::uint32_t kExpectedSetupTriangles = 6797;
constexpr std::uint32_t kExpectedVisibleCells = 2044;
constexpr std::uint32_t kExpectedRasterizableTriangles =
    2 * kExpectedVisibleCells;

void Check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error("half-culled geometry test failed: " + message);
}

std::uint32_t RotateRight(std::uint32_t value, unsigned amount) {
  return (value >> amount) | (value << (32U - amount));
}

// Small test-local SHA-256 keeps the byte-level fixture pin independent of
// platform crypto libraries and explicitly hashes little-endian uint16 data.
std::string Sha256(const std::vector<std::uint8_t> &input) {
  static constexpr std::array<std::uint32_t, 64> kRound = {
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
      0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
      0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
      0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
      0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
      0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
      0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
      0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
      0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
      0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
      0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
      0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
      0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
      0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
      0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
  };
  std::vector<std::uint8_t> padded = input;
  const std::uint64_t bit_length =
      static_cast<std::uint64_t>(padded.size()) * 8U;
  padded.push_back(0x80U);
  while (padded.size() % 64 != 56)
    padded.push_back(0);
  for (int shift = 56; shift >= 0; shift -= 8)
    padded.push_back(static_cast<std::uint8_t>(bit_length >> shift));

  std::array<std::uint32_t, 8> hash = {
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
  };
  for (std::size_t block = 0; block < padded.size(); block += 64) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16; ++index) {
      const std::size_t offset = block + index * 4;
      words[index] = static_cast<std::uint32_t>(padded[offset]) << 24U |
                     static_cast<std::uint32_t>(padded[offset + 1]) << 16U |
                     static_cast<std::uint32_t>(padded[offset + 2]) << 8U |
                     static_cast<std::uint32_t>(padded[offset + 3]);
    }
    for (std::size_t index = 16; index < words.size(); ++index) {
      const std::uint32_t s0 =
          RotateRight(words[index - 15], 7) ^
          RotateRight(words[index - 15], 18) ^ (words[index - 15] >> 3U);
      const std::uint32_t s1 =
          RotateRight(words[index - 2], 17) ^
          RotateRight(words[index - 2], 19) ^ (words[index - 2] >> 10U);
      words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }

    std::uint32_t a = hash[0];
    std::uint32_t b = hash[1];
    std::uint32_t c = hash[2];
    std::uint32_t d = hash[3];
    std::uint32_t e = hash[4];
    std::uint32_t f = hash[5];
    std::uint32_t g = hash[6];
    std::uint32_t h = hash[7];
    for (std::size_t index = 0; index < words.size(); ++index) {
      const std::uint32_t upper_sigma1 =
          RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
      const std::uint32_t choice = (e & f) ^ (~e & g);
      const std::uint32_t temp1 =
          h + upper_sigma1 + choice + kRound[index] + words[index];
      const std::uint32_t upper_sigma0 =
          RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = upper_sigma0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    hash[0] += a;
    hash[1] += b;
    hash[2] += c;
    hash[3] += d;
    hash[4] += e;
    hash[5] += f;
    hash[6] += g;
    hash[7] += h;
  }

  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const std::uint32_t word : hash)
    output << std::setw(8) << word;
  return output.str();
}

void CheckFixture(const std::vector<std::uint16_t> &indices) {
  GlbenchDarwinRand random(0);
  constexpr std::array<std::uint32_t, 10> kFirstValues = {
      520932930U,  28925691U,   822784415U, 890459872U, 145532761U,
      2132723841U, 1040043610U, 1643550337U, 68362598U,  66433441U,
  };
  for (const std::uint32_t expected : kFirstValues)
    Check(random.Next() == expected, "pinned srand(0) sequence");

  Check(indices.size() == kGlbenchTriangleIndexCount,
        "mixed-winding index count");
  std::vector<std::uint8_t> bytes;
  bytes.reserve(indices.size() * 2);
  for (const std::uint16_t index : indices) {
    bytes.push_back(static_cast<std::uint8_t>(index));
    bytes.push_back(static_cast<std::uint8_t>(index >> 8U));
  }
  Check(Sha256(bytes) ==
            "d8735669c79e4e5e99baf336272932bb9eab947e9a5bf96bf56a3f6ba81cd603",
        "pinned little-endian uint16 index SHA-256");

  std::size_t cell = 0;
  std::uint32_t front_cells = 0;
  std::uint32_t visible_front_cells = 0;
  for (std::uint32_t y = 0; y < kGlbenchTriangleMeshHeight;
       y += kGlbenchTriangleSwathHeight) {
    for (std::uint32_t x = 0; x < kGlbenchTriangleMeshWidth; ++x) {
      for (std::uint32_t swath_y = 0;
           swath_y < kGlbenchTriangleSwathHeight; ++swath_y, ++cell) {
        const std::size_t offset = cell * 6;
        const std::uint16_t first = indices[offset];
        const bool front_facing = indices[offset + 1] == first + 1;
        Check(indices[offset + 2] ==
                  (front_facing ? first + kGlbenchTriangleMeshWidth + 1
                                : first + 1),
              "first triangle winding pair");
        if (!front_facing)
          continue;
        ++front_cells;
        const std::uint32_t row = y + swath_y;
        if (x >= 32 && x < 96 && row >= 32 && row < 96)
          ++visible_front_cells;
      }
    }
  }
  Check(cell == 16384 && front_cells == 8166,
        "one swath-order random choice per lattice cell");
  Check(visible_front_cells == kExpectedVisibleCells,
        "64x64 viewport front-facing cell count");
}

} // namespace

int sc_main(int, char **) {
  try {
    Check(FunctionalCaseFromName("triangle_setup_half_culled") ==
              FunctionalCase::kTriangleSetupHalfCulled,
          "functional case mapping");
    Check(std::string(FunctionalCaseName(
              FunctionalCase::kTriangleSetupHalfCulled)) ==
              "triangle_setup_half_culled",
          "functional case reverse mapping");

    MemoryPool pool;
    const std::vector<InputVertex> fixture_vertices =
        MakeGlbenchTriangleVertices(kWidth, kHeight);
    std::vector<float> vertices;
    vertices.reserve(2 * fixture_vertices.size());
    for (const InputVertex &vertex : fixture_vertices) {
      vertices.push_back(vertex.x);
      vertices.push_back(vertex.y);
    }
    Check(fixture_vertices.size() == 16641 &&
              vertices.size() == 2 * fixture_vertices.size(),
          "float2 lattice vertex payload");
    const std::vector<std::uint16_t> indices = MakeGlbenchTriangleIndices(
        GlbenchTriangleWindingPattern::kSrandZeroHalfCulled);
    CheckFixture(indices);

    PipelineState state;
    state.width = kWidth;
    state.height = kHeight;
    state.sequence = 1;
    state.functional_case = FunctionalCase::kTriangleSetupHalfCulled;
    state.stage = PipelineStage::kSubmitted;
    state.draw.topology = PrimitiveTopology::kTriangleList;
    state.draw.index_count = indices.size();
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
    Check(result.counters.ia_vertices == kGlbenchTriangleIndexCount &&
              result.counters.ia_primitives == kExpectedPrimitiveCount &&
              result.counters.vs_invocations == kExpectedVertexInvocations &&
              result.counters.vertex_attribute_fetches ==
                  kExpectedVertexInvocations &&
              result.counters.vertex_attribute_bytes ==
                  kExpectedVertexInvocations * 2 * sizeof(float) &&
              result.counters.c_invocations == kExpectedPrimitiveCount,
          "indexed upstream counters");
    Check(result.counters.c_primitives == kExpectedSetupTriangles &&
              result.counters.setup_triangles == kExpectedSetupTriangles,
          "segment-derived mixed-winding setup count");

    const std::vector<RasterTriangle> triangles =
        LoadArray<RasterTriangle>(pool, result.raster_triangles);
    Check(triangles.size() == kExpectedSetupTriangles,
          "setup payload count");
    const std::size_t rasterizable =
        std::count_if(triangles.begin(), triangles.end(),
                      [](const RasterTriangle &triangle) {
                        return triangle.rasterizable != 0;
                      });
    Check(rasterizable == kExpectedRasterizableTriangles,
          "front-facing central cells emit two raster triangles each");
    for (const RasterTriangle &triangle : triangles) {
      Check(!triangle.rasterizable ||
                (triangle.front_facing != 0 && triangle.face_culled == 0),
            "only front-facing triangles reach tile bins");
    }

    const std::vector<ParameterTriangle> parameters =
        LoadArray<ParameterTriangle>(pool, result.parameter_triangles);
    const std::vector<TilePrimitiveRef> refs =
        LoadArray<TilePrimitiveRef>(pool, result.tile_primitive_refs);
    Check(parameters.size() == triangles.size(),
          "parameter slots preserve setup identity");
    Check(refs.size() == kExpectedRasterizableTriangles,
          "one tile ref per in-viewport raster triangle");

    ReleaseFunctionalPayloads(pool, result);
    pool.Release(state_handle);
    Check(pool.bytes_in_flight() == 0 &&
              pool.allocations() == pool.releases(),
          "MemoryPool balanced");
    std::cout << "half_culled_geometry_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "half_culled_geometry_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
