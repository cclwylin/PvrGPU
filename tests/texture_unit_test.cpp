// Functional unit test for TextureUnit/TPU (Texture Processing Unit).
// It pins the public Rogue IMAGE/SAMPLER descriptor words, negative and edge
// repeat coordinates, the selected signed-delta 8-bit round-to-nearest-even
// filter, Gates 18-20 implicit LOD state, and actual event-driven
// four/eight-tap FIFO transactions.  The cache responders return individual
// texels so tap identity, mip selection, and interpolation order cannot be
// replaced by a fixed result.

#include "common/glbench_texture_fixture.h"
#include "common/pipeline_state.h"
#include "texture/texture_unit.h"

#include <systemc>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Rgba8 = std::array<std::uint8_t, 4>;

using pvrgpu::stub::DecodeRogueTextureImageDescriptor;
using pvrgpu::stub::DecodeRogueTextureSamplerDescriptor;
using pvrgpu::stub::ComputeTextureImplicitLod;
using pvrgpu::stub::ComputeTextureLinearRepeat;
using pvrgpu::stub::FunctionalCase;
using pvrgpu::stub::GlbenchFillTextureFixture;
using pvrgpu::stub::HasPoolHandle;
using pvrgpu::stub::LerpTextureUnorm8;
using pvrgpu::stub::LoadArray;
using pvrgpu::stub::LoadPipelineState;
using pvrgpu::stub::MakeGlbenchFillTextureFixture;
using pvrgpu::stub::MemoryClient;
using pvrgpu::stub::MemoryOperation;
using pvrgpu::stub::MemoryPayloadFormat;
using pvrgpu::stub::MemoryPool;
using pvrgpu::stub::MemoryTxn;
using pvrgpu::stub::PipelineStage;
using pvrgpu::stub::PipelineState;
using pvrgpu::stub::PipelineTxn;
using pvrgpu::stub::ReleaseFunctionalPayloads;
using pvrgpu::stub::RogueTextureImageDescriptor;
using pvrgpu::stub::RogueTextureSamplerDescriptor;
using pvrgpu::stub::StoreNewArray;
using pvrgpu::stub::StorePipelineState;
using pvrgpu::stub::TextureFilter;
using pvrgpu::stub::TextureImplicitLod;
using pvrgpu::stub::TextureLinearAxis;
using pvrgpu::stub::TextureResource;
using pvrgpu::stub::TextureSampleRequest;
using pvrgpu::stub::TextureSampleResponse;
using pvrgpu::stub::TextureUnit;

void Check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error("texture unit test failed: " + message);
}

template <typename Function>
void ExpectFailure(Function &&function, const std::string &description) {
  try {
    function();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error("expected descriptor failure: " + description);
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float BitsFloat(std::uint32_t bits) {
  float value = 0.0F;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint64_t DescriptorWord(
    const std::array<std::uint32_t,
                     pvrgpu::stub::kFillTexNearestSharedDwordCount> &shared,
    std::size_t first_dword) {
  return static_cast<std::uint64_t>(shared.at(first_dword)) |
         (static_cast<std::uint64_t>(shared.at(first_dword + 1U)) << 32U);
}

std::array<std::uint32_t, 4> DescriptorDwords(
    const GlbenchFillTextureFixture &fixture, std::size_t first_dword) {
  std::array<std::uint32_t, 4> words{};
  std::copy_n(fixture.fragment_shared.begin() + first_dword, words.size(),
              words.begin());
  return words;
}

void CheckDescriptorAndArithmetic() {
  const GlbenchFillTextureFixture nearest =
      MakeGlbenchFillTextureFixture(TextureFilter::kNearest);
  const GlbenchFillTextureFixture linear =
      MakeGlbenchFillTextureFixture(TextureFilter::kLinear);
  const GlbenchFillTextureFixture trilinear =
      MakeGlbenchFillTextureFixture(
          FunctionalCase::kFillTexTrilinearLinear01);
  const GlbenchFillTextureFixture trilinear04 =
      MakeGlbenchFillTextureFixture(
          FunctionalCase::kFillTexTrilinearLinear04);
  const GlbenchFillTextureFixture trilinear05 =
      MakeGlbenchFillTextureFixture(
          FunctionalCase::kFillTexTrilinearLinear05);

  Check(DescriptorWord(nearest.fragment_shared, 8) ==
            UINT64_C(0x0000000000000fff),
        "nearest literal sampler word with LOD0 clamp");
  Check(DescriptorWord(linear.fragment_shared, 8) ==
            UINT64_C(0x0000005000000fff),
        "linear literal sampler word with LOD0 clamp");
  Check(DescriptorWord(trilinear.fragment_shared, 8) ==
            UINT64_C(0x00000151df800fff),
        "Gate 18 trilinear literal sampler word");
  Check(trilinear04.vertex_scale_bits == UINT32_C(0x3f420c4a) &&
            DescriptorWord(trilinear04.fragment_shared, 8) ==
                UINT64_C(0x00000151df800fff),
        "Gate 19 live 0.758f SH0 and shared trilinear sampler word");
  Check(trilinear05.vertex_scale_bits == UINT32_C(0x3f350481) &&
            DescriptorWord(trilinear05.fragment_shared, 8) ==
                UINT64_C(0x00000151df800fff),
        "Gate 20 live 0.7071f SH0 and shared trilinear sampler word");

  const RogueTextureImageDescriptor image =
      DecodeRogueTextureImageDescriptor(DescriptorDwords(linear, 0));
  const RogueTextureSamplerDescriptor sampler =
      DecodeRogueTextureSamplerDescriptor(DescriptorDwords(linear, 8));
  Check(image.gpu_address == pvrgpu::stub::kGlbenchTextureGpuAddress &&
            image.width == 512 && image.height == 512 &&
            image.row_pitch_bytes == 2048 && image.mip_count == 10,
        "raw image address/extent/stride/mip decode");
  Check(sampler.min_filter == TextureFilter::kLinear &&
            sampler.mag_filter == TextureFilter::kLinear &&
            sampler.mip_filter == TextureFilter::kNearest &&
            sampler.min_lod_u4_6 == 0 && sampler.max_lod_u4_6 == 0 &&
            sampler.normalized_coordinates == 1,
        "raw sampler filter/normalized/LOD decode");

  const RogueTextureImageDescriptor trilinear_image =
      DecodeRogueTextureImageDescriptor(DescriptorDwords(trilinear, 0));
  const RogueTextureSamplerDescriptor trilinear_sampler =
      DecodeRogueTextureSamplerDescriptor(DescriptorDwords(trilinear, 8));
  Check(trilinear_sampler.min_filter == TextureFilter::kLinear &&
            trilinear_sampler.mag_filter == TextureFilter::kLinear &&
            trilinear_sampler.mip_filter == TextureFilter::kLinear &&
            trilinear_sampler.min_lod_u4_6 == 0 &&
            trilinear_sampler.max_lod_u4_6 == 959 &&
            trilinear_sampler.normalized_coordinates == 1,
        "Gate 18 raw sampler enables implicit trilinear filtering");

  // One screen-pixel derivative across the 64x64 Gate 18 quad after its
  // float32 0.933 vertex scale.  The selected LODM=NORMAL datapath must expose
  // mip levels 3/4 and architectural TFRAC=19, not a host-float mip weight.
  const float gate18_scale = BitsFloat(trilinear.vertex_scale_bits);
  const float gate18_derivative = 1.0F / (64.0F * gate18_scale);
  constexpr float kGate18U = 0.25F;
  constexpr float kGate18V = 0.375F;
  const std::array<std::array<float, 2>, 4> gate18_coordinates = {{
      {{kGate18U, kGate18V}},
      {{kGate18U + gate18_derivative, kGate18V}},
      {{kGate18U, kGate18V + gate18_derivative}},
      {{kGate18U + gate18_derivative, kGate18V + gate18_derivative}},
  }};
  const TextureImplicitLod gate18_lod = ComputeTextureImplicitLod(
      gate18_coordinates, trilinear_image, trilinear_sampler);
  Check(gate18_lod.level0 == 3 && gate18_lod.level1 == 4 &&
            gate18_lod.mip_weight_u8 == 19 &&
            FloatBits(gate18_lod.mip_weight) == FloatBits(19.0F / 256.0F) &&
            gate18_lod.lambda >= 3.0F && gate18_lod.lambda < 4.0F,
        "Gate 18 implicit LOD levels and TFRAC");

  // Gate 19 keeps the descriptor and shaders byte-identical while changing
  // only live SH0 to 0.758f.  Its larger implicit derivative remains between
  // mip levels 3/4 but quantizes to architectural TFRAC=94.
  const RogueTextureImageDescriptor trilinear04_image =
      DecodeRogueTextureImageDescriptor(DescriptorDwords(trilinear04, 0));
  const RogueTextureSamplerDescriptor trilinear04_sampler =
      DecodeRogueTextureSamplerDescriptor(DescriptorDwords(trilinear04, 8));
  const float gate19_scale = BitsFloat(trilinear04.vertex_scale_bits);
  const float gate19_derivative = 1.0F / (64.0F * gate19_scale);
  const std::array<std::array<float, 2>, 4> gate19_coordinates = {{
      {{kGate18U, kGate18V}},
      {{kGate18U + gate19_derivative, kGate18V}},
      {{kGate18U, kGate18V + gate19_derivative}},
      {{kGate18U + gate19_derivative, kGate18V + gate19_derivative}},
  }};
  const TextureImplicitLod gate19_lod = ComputeTextureImplicitLod(
      gate19_coordinates, trilinear04_image, trilinear04_sampler);
  Check(gate19_lod.level0 == 3 && gate19_lod.level1 == 4 &&
            gate19_lod.mip_weight_u8 == 94 &&
            FloatBits(gate19_lod.mip_weight) == FloatBits(94.0F / 256.0F) &&
            gate19_lod.lambda >= 3.0F && gate19_lod.lambda < 4.0F,
        "Gate 19 implicit LOD levels and TFRAC");

  // Gate 20's exact 0.7071f scale reaches the mip half-way boundary in the
  // selected fast-log LODM=NORMAL datapath.  Pin the ideal float32 derivative,
  // the lambda after storing coordinates at the chosen base point, and the
  // architectural U8 TFRAC=0x80.
  const RogueTextureImageDescriptor trilinear05_image =
      DecodeRogueTextureImageDescriptor(DescriptorDwords(trilinear05, 0));
  const RogueTextureSamplerDescriptor trilinear05_sampler =
      DecodeRogueTextureSamplerDescriptor(DescriptorDwords(trilinear05, 8));
  const float gate20_scale = BitsFloat(trilinear05.vertex_scale_bits);
  const float gate20_derivative = 1.0F / (64.0F * gate20_scale);
  const std::array<std::array<float, 2>, 4> gate20_coordinates = {{
      {{kGate18U, kGate18V}},
      {{kGate18U + gate20_derivative, kGate18V}},
      {{kGate18U, kGate18V + gate20_derivative}},
      {{kGate18U + gate20_derivative, kGate18V + gate20_derivative}},
  }};
  const TextureImplicitLod gate20_lod = ComputeTextureImplicitLod(
      gate20_coordinates, trilinear05_image, trilinear05_sampler);
  Check(FloatBits(gate20_derivative) == UINT32_C(0x3cb50565) &&
            gate20_lod.level0 == 3 && gate20_lod.level1 == 4 &&
            gate20_lod.mip_weight_u8 == 128 &&
            FloatBits(gate20_lod.mip_weight) == FloatBits(0.5F) &&
            FloatBits(gate20_lod.lambda) == UINT32_C(0x40600026),
        "Gate 20 implicit LOD levels, approximate lambda, and TFRAC");

  auto mutated = DescriptorDwords(linear, 8);
  mutated[0] |= UINT32_C(1) << 23U; // public maxlod[0]
  ExpectFailure([&] { DecodeRogueTextureSamplerDescriptor(mutated); },
                "non-zero max LOD");
  mutated = DescriptorDwords(linear, 8);
  mutated[1] |= UINT32_C(1) << (41U - 32U); // addrmode_u=FLIP
  ExpectFailure([&] { DecodeRogueTextureSamplerDescriptor(mutated); },
                "unsupported U address mode");
  mutated = DescriptorDwords(linear, 8);
  mutated[1] |= UINT32_C(1) << (49U - 32U); // non-normalized coords
  ExpectFailure([&] { DecodeRogueTextureSamplerDescriptor(mutated); },
                "non-normalized coordinates");
  mutated = DescriptorDwords(linear, 8);
  mutated[2] = 1U; // SAMPLER_WORD1 plane/non-seamless state
  ExpectFailure([&] { DecodeRogueTextureSamplerDescriptor(mutated); },
                "unsupported sampler word1");

  auto mutated_image = DescriptorDwords(linear, 0);
  mutated_image[0] ^= UINT32_C(1) << 3U; // gamma/two-component state
  ExpectFailure([&] { DecodeRogueTextureImageDescriptor(mutated_image); },
                "gamma image mutation");
  mutated_image = DescriptorDwords(linear, 0);
  mutated_image[3] |= UINT32_C(1) << (54U - 32U); // compression state
  ExpectFailure([&] { DecodeRogueTextureImageDescriptor(mutated_image); },
                "compression image mutation");

  const TextureLinearAxis zero = ComputeTextureLinearRepeat(0.0F, 512);
  const TextureLinearAxis one = ComputeTextureLinearRepeat(1.0F, 512);
  const TextureLinearAxis negative =
      ComputeTextureLinearRepeat(-0.25F, 4);
  Check(zero.lower == 511 && zero.upper == 0 && zero.weight == 128,
        "u=0 repeat edge");
  Check(one.lower == 511 && one.upper == 0 && one.weight == 128,
        "u=1 repeat edge");
  Check(negative.lower == 2 && negative.upper == 3 &&
            negative.weight == 128,
        "u=-0.25 negative repeat");
  Check(LerpTextureUnorm8(0, 1, 128) == 0,
        "positive +0.5 delta tie rounds to even zero");
  Check(LerpTextureUnorm8(1, 2, 128) == 1,
        "endpoint parity does not change positive delta-tie rounding");
  Check(LerpTextureUnorm8(1, 4, 128) == 3,
        "positive +1.5 delta tie rounds to even two");
  Check(LerpTextureUnorm8(1, 0, 128) == 1,
        "negative -0.5 delta tie rounds to even zero");
  Check(LerpTextureUnorm8(3, 0, 128) == 1,
        "negative -1.5 delta tie rounds to even minus two");
}

Rgba8 FixtureTexel(const GlbenchFillTextureFixture &fixture,
                   const pvrgpu::stub::TextureMipLevel &mip,
                   std::uint32_t x, std::uint32_t y) {
  const std::uint64_t offset =
      static_cast<std::uint64_t>(mip.offset_bytes) +
      static_cast<std::uint64_t>(y) * mip.row_pitch_bytes +
      static_cast<std::uint64_t>(x) * 4U;
  Check(offset + 4U <= fixture.texture_bytes.size(),
        "fixture texel address is in range");
  Rgba8 texel{};
  for (std::size_t component = 0; component < texel.size(); ++component)
    texel[component] = fixture.texture_bytes.at(offset + component);
  return texel;
}

void AppendBilinearReads(const GlbenchFillTextureFixture &fixture,
                         std::uint8_t mip_level, float u, float v,
                         std::vector<std::uint64_t> &addresses,
                         std::vector<Rgba8> &taps) {
  const pvrgpu::stub::TextureMipLevel &mip =
      fixture.resource.mip[mip_level];
  const TextureLinearAxis x = ComputeTextureLinearRepeat(u, mip.width);
  const TextureLinearAxis y = ComputeTextureLinearRepeat(v, mip.height);
  const std::array<std::array<std::uint32_t, 2>, 4> locations = {{
      {{x.lower, y.lower}},
      {{x.upper, y.lower}},
      {{x.lower, y.upper}},
      {{x.upper, y.upper}},
  }};
  for (const auto &location : locations) {
    const std::uint64_t offset =
        static_cast<std::uint64_t>(mip.offset_bytes) +
        static_cast<std::uint64_t>(location[1]) * mip.row_pitch_bytes +
        static_cast<std::uint64_t>(location[0]) * 4U;
    addresses.push_back(fixture.resource.gpu_address + offset);
    taps.push_back(FixtureTexel(fixture, mip, location[0], location[1]));
  }
}

class TextureMemoryResponder final : public sc_core::sc_module {
 public:
  sc_core::sc_fifo_in<MemoryTxn> upload_input{"upload_input"};
  sc_core::sc_fifo_out<MemoryTxn> upload_output{"upload_output"};
  sc_core::sc_fifo_in<MemoryTxn> cache_input{"cache_input"};
  sc_core::sc_fifo_out<MemoryTxn> cache_output{"cache_output"};

  TextureMemoryResponder(
      sc_core::sc_module_name name, MemoryPool &pool,
      std::uint64_t upload_address, std::uint64_t upload_bytes,
      std::vector<std::uint64_t> expected_addresses,
      std::vector<Rgba8> taps)
      : sc_core::sc_module(name), pool_(pool),
        upload_address_(upload_address), upload_bytes_(upload_bytes),
        expected_addresses_(expected_addresses), taps_(taps) {
    Check(expected_addresses_.size() == taps_.size() && !taps_.empty(),
          "cache responder address/tap vectors");
    SC_THREAD(UploadRun);
    SC_THREAD(CacheRun);
  }

 private:
  void UploadRun() {
    const MemoryTxn request = upload_input.read();
    Check(request.address == upload_address_ && request.bytes == upload_bytes_ &&
              request.operation == MemoryOperation::kWrite &&
              request.client == MemoryClient::kTextureUpload &&
              request.payload_format == MemoryPayloadFormat::kLinearBytes &&
              HasPoolHandle(request.payload),
          "raw descriptor address drives texture upload");
    MemoryTxn response = request;
    response.payload = {};
    upload_output.write(response);
  }

  void CacheRun() {
    for (std::size_t tap = 0; tap < expected_addresses_.size(); ++tap) {
      const MemoryTxn request = cache_input.read();
      Check(request.address == expected_addresses_[tap] && request.bytes == 4 &&
                request.request_id == tap &&
                request.operation == MemoryOperation::kRead &&
                request.client == MemoryClient::kTextureCache &&
                request.payload_format == MemoryPayloadFormat::kLinearBytes &&
                !HasPoolHandle(request.payload),
            "FIFO tap address and request identity");
      MemoryTxn response = request;
      response.payload = StoreNewArray(
          pool_, std::vector<std::uint8_t>(taps_[tap].begin(),
                                           taps_[tap].end()));
      cache_output.write(response);
    }
  }

  MemoryPool &pool_;
  std::uint64_t upload_address_ = 0;
  std::uint64_t upload_bytes_ = 0;
  std::vector<std::uint64_t> expected_addresses_;
  std::vector<Rgba8> taps_;
};

void CheckEventPaths() {
  MemoryPool pool;
  GlbenchFillTextureFixture fixture =
      MakeGlbenchFillTextureFixture(TextureFilter::kLinear);

  // These exact binary fractions yield lower=(127,383), upper=(128,384),
  // and weights=(64,192) in the selected U8 coordinate datapath.
  constexpr float kU = 0.24951171875F;
  constexpr float kV = 0.75048828125F;
  const TextureLinearAxis x = ComputeTextureLinearRepeat(kU, 512);
  const TextureLinearAxis y = ComputeTextureLinearRepeat(kV, 512);
  Check(x.lower == 127 && x.upper == 128 && x.weight == 64 &&
            y.lower == 383 && y.upper == 384 && y.weight == 192,
        "asymmetric event-path weights");

  TextureSampleRequest request;
  request.shader_lane_index = 0;
  request.coordinates[0] = FloatBits(kU);
  request.coordinates[1] = FloatBits(kV);
  std::copy_n(fixture.fragment_shared.begin(), 4, request.texture_state);
  std::copy_n(fixture.fragment_shared.begin() + 8, 4,
              request.sampler_state);
  request.request_id = 0;
  request.coordinate_count = 2;
  request.component_count = 4;
  request.descriptor_set = 0;
  request.binding = 0;
  request.dimension = 2;
  request.normalized = 1;

  fixture.resource.data = StoreNewArray(pool, fixture.texture_bytes);
  PipelineState state;
  state.sequence = 1;
  state.functional_case = FunctionalCase::kFillTexBilinear;
  state.stage = PipelineStage::kFragmentTexturePending;
  state.fragment_shader_lane_count = 1;
  state.texture_sample_requests =
      StoreNewArray(pool, std::vector<TextureSampleRequest>{request});
  state.texture_resources =
      StoreNewArray(pool, std::vector<TextureResource>{fixture.resource});
  state.sampler_states = StoreNewArray(
      pool, std::vector<pvrgpu::stub::SamplerState>{fixture.sampler});
  state.fragment_shared_registers = StoreNewArray(
      pool, std::vector<std::uint32_t>(fixture.fragment_shared.begin(),
                                      fixture.fragment_shared.end()));
  const pvrgpu::stub::PoolHandle state_handle =
      pool.Allocate(sizeof(PipelineState));
  StorePipelineState(pool, state_handle, state);
  PipelineTxn transaction;
  transaction.state = state_handle;
  transaction.frame = 1;
  transaction.sequence = 1;

  const std::vector<Rgba8> taps = {
      {{0, 10, 20, 30}},
      {{40, 50, 60, 70}},
      {{80, 90, 100, 110}},
      {{120, 130, 140, 150}},
  };
  const std::uint64_t base = pvrgpu::stub::kGlbenchTextureGpuAddress;
  constexpr std::uint64_t pitch = 2048;
  const std::vector<std::uint64_t> expected_addresses = {
      base + 383U * pitch + 127U * 4U,
      base + 383U * pitch + 128U * 4U,
      base + 384U * pitch + 127U * 4U,
      base + 384U * pitch + 128U * 4U,
  };

  sc_core::sc_fifo<PipelineTxn> module_input("module_input", 1);
  sc_core::sc_fifo<PipelineTxn> module_output("module_output", 1);
  sc_core::sc_fifo<PipelineTxn> sample_input("sample_input", 1);
  sc_core::sc_fifo<PipelineTxn> sample_output("sample_output", 1);
  sc_core::sc_fifo<MemoryTxn> cache_request("cache_request", 1);
  sc_core::sc_fifo<MemoryTxn> cache_response("cache_response", 1);
  sc_core::sc_fifo<MemoryTxn> upload_request("upload_request", 1);
  sc_core::sc_fifo<MemoryTxn> upload_response("upload_response", 1);

  TextureUnit texture("texture", pool);
  texture.input(module_input);
  texture.output(module_output);
  texture.sample_input(sample_input);
  texture.sample_output(sample_output);
  texture.cache_request(cache_request);
  texture.cache_response(cache_response);
  texture.upload_request(upload_request);
  texture.upload_response(upload_response);

  TextureMemoryResponder responder(
      "responder", pool, base, fixture.resource.byte_size,
      expected_addresses, taps);
  responder.upload_input(upload_request);
  responder.upload_output(upload_response);
  responder.cache_input(cache_request);
  responder.cache_output(cache_response);

  // Gate 18 runs alongside the four-tap test so every SystemC object is
  // elaborated before the single sc_start().  Four adjacent lanes form a real
  // implicit-derivative quad; each lane emits mip3 00/10/01/11 followed by
  // mip4 00/10/01/11 through the cache FIFO.
  MemoryPool gate_pool;
  GlbenchFillTextureFixture gate_fixture = MakeGlbenchFillTextureFixture(
      FunctionalCase::kFillTexTrilinearLinear01);
  const float gate_scale = BitsFloat(gate_fixture.vertex_scale_bits);
  const float gate_derivative = 1.0F / (64.0F * gate_scale);
  constexpr float kGateU = 0.25F;
  constexpr float kGateV = 0.375F;
  const std::array<std::array<float, 2>, 4> gate_coordinates = {{
      {{kGateU, kGateV}},
      {{kGateU + gate_derivative, kGateV}},
      {{kGateU, kGateV + gate_derivative}},
      {{kGateU + gate_derivative, kGateV + gate_derivative}},
  }};

  std::vector<TextureSampleRequest> gate_requests(4);
  std::vector<std::uint64_t> gate_addresses;
  std::vector<Rgba8> gate_taps;
  gate_addresses.reserve(32);
  gate_taps.reserve(32);
  for (std::size_t lane = 0; lane < gate_requests.size(); ++lane) {
    TextureSampleRequest &gate_request = gate_requests[lane];
    gate_request.shader_lane_index = static_cast<std::uint32_t>(lane);
    gate_request.quad_id = 7;
    gate_request.coordinates[0] = FloatBits(gate_coordinates[lane][0]);
    gate_request.coordinates[1] = FloatBits(gate_coordinates[lane][1]);
    std::copy_n(gate_fixture.fragment_shared.begin(), 4,
                gate_request.texture_state);
    std::copy_n(gate_fixture.fragment_shared.begin() + 8, 4,
                gate_request.sampler_state);
    gate_request.request_id = lane;
    gate_request.coordinate_count = 2;
    gate_request.component_count = 4;
    gate_request.dimension = 2;
    gate_request.normalized = 1;
    gate_request.quad_lane = static_cast<std::uint8_t>(lane);
    AppendBilinearReads(gate_fixture, 3, gate_coordinates[lane][0],
                        gate_coordinates[lane][1], gate_addresses, gate_taps);
    AppendBilinearReads(gate_fixture, 4, gate_coordinates[lane][0],
                        gate_coordinates[lane][1], gate_addresses, gate_taps);
  }
  Check(gate_addresses.size() == 32 && gate_taps.size() == 32,
        "Gate 18 quad expands to 32 physical reads");

  gate_fixture.resource.data =
      StoreNewArray(gate_pool, gate_fixture.texture_bytes);
  PipelineState gate_state;
  gate_state.sequence = 2;
  gate_state.functional_case = FunctionalCase::kFillTexTrilinearLinear01;
  gate_state.stage = PipelineStage::kFragmentTexturePending;
  gate_state.fragment_shader_lane_count = gate_requests.size();
  gate_state.texture_sample_requests =
      StoreNewArray(gate_pool, gate_requests);
  gate_state.texture_resources = StoreNewArray(
      gate_pool, std::vector<TextureResource>{gate_fixture.resource});
  gate_state.sampler_states = StoreNewArray(
      gate_pool,
      std::vector<pvrgpu::stub::SamplerState>{gate_fixture.sampler});
  gate_state.fragment_shared_registers = StoreNewArray(
      gate_pool,
      std::vector<std::uint32_t>(gate_fixture.fragment_shared.begin(),
                                 gate_fixture.fragment_shared.end()));
  const pvrgpu::stub::PoolHandle gate_state_handle =
      gate_pool.Allocate(sizeof(PipelineState));
  StorePipelineState(gate_pool, gate_state_handle, gate_state);
  PipelineTxn gate_transaction;
  gate_transaction.state = gate_state_handle;
  gate_transaction.frame = 2;
  gate_transaction.sequence = 2;

  sc_core::sc_fifo<PipelineTxn> gate_module_input("gate_module_input", 1);
  sc_core::sc_fifo<PipelineTxn> gate_module_output("gate_module_output", 1);
  sc_core::sc_fifo<PipelineTxn> gate_sample_input("gate_sample_input", 1);
  sc_core::sc_fifo<PipelineTxn> gate_sample_output("gate_sample_output", 1);
  sc_core::sc_fifo<MemoryTxn> gate_cache_request("gate_cache_request", 1);
  sc_core::sc_fifo<MemoryTxn> gate_cache_response("gate_cache_response", 1);
  sc_core::sc_fifo<MemoryTxn> gate_upload_request("gate_upload_request", 1);
  sc_core::sc_fifo<MemoryTxn> gate_upload_response("gate_upload_response", 1);

  TextureUnit gate_texture("gate_texture", gate_pool);
  gate_texture.input(gate_module_input);
  gate_texture.output(gate_module_output);
  gate_texture.sample_input(gate_sample_input);
  gate_texture.sample_output(gate_sample_output);
  gate_texture.cache_request(gate_cache_request);
  gate_texture.cache_response(gate_cache_response);
  gate_texture.upload_request(gate_upload_request);
  gate_texture.upload_response(gate_upload_response);

  TextureMemoryResponder gate_responder(
      "gate_responder", gate_pool, base, gate_fixture.resource.byte_size,
      gate_addresses, gate_taps);
  gate_responder.upload_input(gate_upload_request);
  gate_responder.upload_output(gate_upload_response);
  gate_responder.cache_input(gate_cache_request);
  gate_responder.cache_output(gate_cache_response);

  sample_input.write(transaction);
  gate_sample_input.write(gate_transaction);
  sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_NS));
  PipelineTxn completed;
  Check(sample_output.nb_read(completed) &&
            completed.state.slot == state_handle.slot &&
            completed.state.generation == state_handle.generation &&
            completed.frame == 1 && completed.sequence == 1,
        "event-driven sample completion identity");

  const PipelineState final_state = LoadPipelineState(pool, state_handle);
  Check(final_state.stage == PipelineStage::kTextureSamplesReady &&
            final_state.counters.texture_requests == 1 &&
            final_state.counters.texel_fetches == 4 &&
            final_state.counters.texture_cycles ==
                pvrgpu::stub::kReferenceUarch.texture_bypass_cycles,
        "one logical request preserves four physical tap counters");
  const std::vector<TextureSampleResponse> responses =
      LoadArray<TextureSampleResponse>(pool,
                                       final_state.texture_sample_responses);
  Check(responses.size() == 1 && responses[0].request_id == 0 &&
            responses[0].shader_lane_index == 0,
        "aggregated texture response identity");
  constexpr std::array<std::uint8_t, 4> kExpected = {{70, 80, 90, 100}};
  for (std::size_t component = 0; component < kExpected.size(); ++component) {
    Check(responses[0].rgba[component] ==
              FloatBits(static_cast<float>(kExpected[component]) / 255.0F),
          "asymmetric sequential U8 bilinear component");
  }

  PipelineTxn gate_completed;
  Check(gate_sample_output.nb_read(gate_completed) &&
            gate_completed.state.slot == gate_state_handle.slot &&
            gate_completed.state.generation == gate_state_handle.generation &&
            gate_completed.frame == 2 && gate_completed.sequence == 2,
        "Gate 18 event-driven sample completion identity");
  const PipelineState gate_final_state =
      LoadPipelineState(gate_pool, gate_state_handle);
  Check(gate_final_state.stage == PipelineStage::kTextureSamplesReady &&
            gate_final_state.counters.texture_requests == 4 &&
            gate_final_state.counters.texel_fetches == 32 &&
            gate_final_state.counters.texture_cycles ==
                4U * pvrgpu::stub::kReferenceUarch.texture_bypass_cycles,
        "Gate 18 quad preserves four requests and eight taps per lane");
  const std::vector<TextureSampleResponse> gate_responses =
      LoadArray<TextureSampleResponse>(
          gate_pool, gate_final_state.texture_sample_responses);
  constexpr std::array<Rgba8, 4> kGateExpected = {{
      {{9, 115, 124, 255}},
      {{4, 56, 60, 255}},
      {{9, 115, 124, 255}},
      {{5, 63, 68, 255}},
  }};
  Check(gate_responses.size() == kGateExpected.size(),
        "Gate 18 returns one response per quad lane");
  for (std::size_t lane = 0; lane < kGateExpected.size(); ++lane) {
    Check(gate_responses[lane].request_id == lane &&
              gate_responses[lane].shader_lane_index == lane,
          "Gate 18 response preserves lane identity");
    for (std::size_t component = 0;
         component < kGateExpected[lane].size(); ++component) {
      Check(gate_responses[lane].rgba[component] ==
                FloatBits(static_cast<float>(kGateExpected[lane][component]) /
                          255.0F),
            "Gate 18 mip3/mip4 U8 trilinear result");
    }
  }

  ReleaseFunctionalPayloads(pool, final_state);
  pool.Release(state_handle);
  Check(pool.bytes_in_flight() == 0 &&
            pool.allocations() == pool.releases(),
        "MemoryPool ownership balanced");
  ReleaseFunctionalPayloads(gate_pool, gate_final_state);
  gate_pool.Release(gate_state_handle);
  Check(gate_pool.bytes_in_flight() == 0 &&
            gate_pool.allocations() == gate_pool.releases(),
        "Gate 18 MemoryPool ownership balanced");
}

} // namespace

int sc_main(int, char **) {
  try {
    CheckDescriptorAndArithmetic();
    CheckEventPaths();
    std::cout << "texture_unit_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "texture_unit_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
