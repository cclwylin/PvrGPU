// SPDX-License-Identifier: MIT
// Actual TextureUnit sample-index reads, not resolves or prepared shader
// results. Independent raw Rogue descriptors, byte-addressed backing storage,
// VS/FS FIFOs, typed extrema, layer/sample rebinding, OOB guards and all three
// modeled memory modes are checked. Existing texture_unit_test retains the
// normalized/filtering regressions.
#include "common/pipeline_state.h"
#include "memory/gpu_memory_system.h"
#include "texture/texture_unit.h"

#include <systemc>

#include <algorithm>
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
// External textures reside above all per-draw attachment reservations. Exercise
// the actual 40-bit Rogue base encoding, not only its low 32-bit half.
constexpr std::uint64_t kBase = UINT64_C(0x8000000000);
unsigned checks = 0;
unsigned batches = 0;

void Check(bool condition, const char *message) {
  ++checks;
  if (!condition)
    throw std::runtime_error(message);
}

template <typename F> void Reject(F &&function, const char *message) {
  try {
    function();
  } catch (const std::exception &) {
    ++checks;
    return;
  }
  throw std::runtime_error(message);
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::vector<std::uint32_t> Descriptor(const TextureResource &resource) {
  const bool array = resource.dimension_type == TextureDimensionType::k2DArray;
  const std::uint64_t format = resource.format == TextureFormat::kRgba8Unorm ? 12U :
      resource.format == TextureFormat::kRgba32Uint ? 62U :
      resource.format == TextureFormat::kRgba32Sint ? 63U : 61U;
  unsigned log_samples = 0;
  while ((1U << log_samples) != resource.sample_count)
    ++log_samples;
  const auto &mip = resource.mip[0];
  const std::uint64_t word0 = (array ? 1U : 4U) | (UINT64_C(3) << 5U) |
      (UINT64_C(2) << 8U) | (UINT64_C(1) << 11U) | (format << 27U) |
      (static_cast<std::uint64_t>(mip.width - 1U) << 34U) |
      (static_cast<std::uint64_t>(mip.height - 1U) << 48U) |
      (static_cast<std::uint64_t>(log_samples) << 62U);
  const std::uint64_t word1 = ((resource.gpu_address >> 2U) << 16U) |
      (array ? resource.mip_count | (static_cast<std::uint64_t>(resource.layer_count - 1U) << 4U)
             : (static_cast<std::uint64_t>(resource.mip_count) << 60U) | (mip.width - 1U)) |
      (resource.mip_count > 1 ? UINT64_C(1) << 15U : 0U);
  std::vector<std::uint32_t> words(20, 0);
  words[0] = static_cast<std::uint32_t>(word0);
  words[1] = static_cast<std::uint32_t>(word0 >> 32U);
  words[2] = static_cast<std::uint32_t>(word1);
  words[3] = static_cast<std::uint32_t>(word1 >> 32U);
  words[4] = array ? mip.row_pitch_bytes * mip.height : resource.byte_size;
  // Normalized raw sampler state remains set (NNCOORDS overrides per SMP).
  // Nearest/repeat/LOD0; gather's linear bits are still cross-checked.
  words[8] = 0xfff;
  words[16] = words[8];
  words[17] = (1U << 4U) | (1U << 6U);
  return words;
}

void CheckAddresses() {
  TextureResource resource;
  resource.gpu_address = kBase;
  resource.format = TextureFormat::kRgba32Uint;
  resource.dimension_type = TextureDimensionType::k2DArray;
  resource.layer_count = 3;
  resource.sample_count = 4;
  resource.mip_count = 1;
  resource.mip[0] = {3, 2, 3 * 4 * 16, 0};
  resource.byte_size = 3 * 2 * 3 * 4 * 16;
  TextureSampleRequest request;
  request.normalized = 0;
  request.sample_index_present = 1;
  request.sample_index = 3;
  request.coordinates[0] = FloatBits(2);
  request.coordinates[1] = FloatBits(1);
  std::uint64_t offset = 0;
  Check(ComputeTextureMultisampleTexelOffset(resource, request, 2, &offset) &&
        offset == resource.byte_size - 16, "last layer/pixel/sample address");
  for (const float coordinate : {-1.0F, 3.0F, 0.5F,
       std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
    auto bad = request;
    bad.coordinates[0] = FloatBits(coordinate);
    Check(!ComputeTextureMultisampleTexelOffset(resource, bad, 0, &offset),
          "OOB/nonintegral coordinate must not address backing memory");
  }
  auto bad_request = request;
  bad_request.sample_index = 4;
  Check(!ComputeTextureMultisampleTexelOffset(resource, bad_request, 0, &offset),
        "transport OOB sample returns zero (not a claim about native SNO truncation)");
  Check(!ComputeTextureMultisampleTexelOffset(resource, request, 3, &offset),
        "OOB layer must not address backing memory");
  for (unsigned mutation = 0; mutation < 8; ++mutation) {
    auto bad_resource = resource;
    bad_request = request;
    switch (mutation) {
    case 0: bad_resource.sample_count = 16; break;
    case 1: bad_resource.mip_count = 2; break;
    case 2: bad_resource.dimension_type = TextureDimensionType::k3D; break;
    case 3: bad_resource.mip[0].row_pitch_bytes = 3 * 16; break;
    case 4: bad_resource.byte_size -= 16; break;
    case 5: bad_resource.mip[0].height = UINT32_MAX; break;
    case 6: bad_request.normalized = 1; break;
    case 7: bad_request.sample_index_present = 0; break;
    }
    Reject([&] { ComputeTextureMultisampleTexelOffset(bad_resource, bad_request, 2, &offset); },
           "malformed MS metadata must fail closed");
  }
  const auto words = Descriptor(resource);
  std::array<std::uint32_t, 4> image_words{};
  std::copy_n(words.begin(), 4, image_words.begin());
  const auto image = DecodeRogueTextureImageDescriptor(image_words);
  Check(image.sample_count == 4 && image.row_pitch_bytes == 192 &&
        image.mip_count == 1 && image.gpu_address == kBase,
        "raw SMPCNT, high 40-bit address and logical stride decode");
  ValidateTextureSingleLevelDimensions(image_words, resource);
  auto mismatched_words = image_words;
  mismatched_words[0] = (mismatched_words[0] & ~7U) | 4U;
  Reject([&] { ValidateTextureSingleLevelDimensions(mismatched_words, resource); },
         "array cannot be replaced by raw STRIDE TEXTYPE");
  mismatched_words = image_words;
  mismatched_words[2] ^= 1U << 4U;
  Reject([&] { ValidateTextureSingleLevelDimensions(mismatched_words, resource); },
         "raw array depth cannot be overridden by structured layers");
  auto plain_resource = resource;
  plain_resource.dimension_type = TextureDimensionType::k2D;
  plain_resource.layer_count = 1;
  Reject([&] { ValidateTextureSingleLevelDimensions(image_words, plain_resource); },
         "plain 2D cannot silently accept raw array TEXTYPE");
  image_words[2] = (image_words[2] & ~15U) | 2U | (1U << 15U);
  Reject([&] { DecodeRogueTextureImageDescriptor(image_words); },
         "raw multisample mipmaps must fail closed");
  GpuMemorySystem memory(MemoryMode::kDirect);
  DriverPcoSampledTexture previous;
  previous.source = DriverPcoTextureSource::kPreviousColorAttachment;
  previous.format = "PIPE_FORMAT_R8G8B8A8_UNORM";
  previous.sample_count = 4;
  previous.mip_count = 1;
  previous.mip[0] = {1, 1, 4, 0};
  previous.declared_bytes_size = 4;
  Reject([&] { MaterializeSequenceColorMipChain(memory, previous); },
         "legacy previous-color mip path cannot resolve a multisample resource");
}

void CheckExplicitAddresses() {
  TextureResource resource;
  resource.gpu_address = kBase;
  resource.format = TextureFormat::kRgba8Unorm;
  resource.dimension_type = TextureDimensionType::k2DArray;
  resource.layer_count = 4; resource.sample_count = 1; resource.mip_count = 3;
  resource.mip[0] = {7,5,32,64};
  resource.mip[1] = {3,2,16,768};
  resource.mip[2] = {1,1,8,960};
  resource.byte_size = 1000;
  TextureSampleRequest request;
  request.normalized = 0; request.explicit_lod_present = 1;
  request.explicit_lod = FloatBits(1);
  request.coordinates[0] = FloatBits(2); request.coordinates[1] = FloatBits(1);
  std::uint64_t offset;
  Check(ComputeTextureTexelOffset(resource, request, 3, &offset) && offset == 888,
        "array mip1 must apply its own padded slice stride and prefix");
  request.explicit_lod = FloatBits(2);
  request.coordinates[0] = request.coordinates[1] = FloatBits(0);
  Check(ComputeTextureTexelOffset(resource, request, 3, &offset) && offset == 984,
        "array mip2 layer remains 3 while pixel extent shrinks");
  resource.dimension_type = TextureDimensionType::k3D;
  request.coordinates[2] = FloatBits(0);
  Check(ComputeTextureTexelOffset(resource, request, 0, &offset) && offset == 960,
        "3D mip2 depth shrinks to one slice");
  request.coordinates[2] = FloatBits(1);
  Check(!ComputeTextureTexelOffset(resource, request, 0, &offset), "3D mip2 slice1 is OOB");
  request.coordinates[2] = 0;
  for (float invalid : {-1.0F,3.0F,0.5F,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
    auto bad = request; bad.explicit_lod = FloatBits(invalid);
    Check(!ComputeTextureTexelOffset(resource, bad, 0, &offset), "LOD OOB must not wrap or clamp");
    bad = request; bad.coordinates[0] = FloatBits(invalid);
    Check(!ComputeTextureTexelOffset(resource, bad, 0, &offset), "texel OOB must not wrap or filter");
  }
  for (unsigned bad = 0; bad < 8; ++bad) {
    auto r = resource; auto q = request;
    if (bad == 0) r.sample_count = 2;
    if (bad == 1) r.mip_count = 0;
    if (bad == 2) r.mip[2].row_pitch_bytes = 3;
    if (bad == 3) r.byte_size = 963;
    if (bad == 4) r.dimension_type = TextureDimensionType::kCube;
    if (bad == 5) q.sample_index_present = 1;
    if (bad == 6) q.explicit_lod_present = 2;
    if (bad == 7) q.normalized = 1;
    Reject([&]{ ComputeTextureTexelOffset(r,q,0,&offset); }, "malformed explicit texel metadata accepted");
  }
  RogueTextureImageDescriptor image; image.mip_count = 4;
  RogueTextureSamplerDescriptor sampler; sampler.max_lod_u4_6 = 192;
  const auto nearest = ComputeTextureExplicitLod(1.5F,image,sampler);
  Check(nearest.level0 == 2 && nearest.level1 == 2, "explicit mip nearest uses float LOD rounding");
  sampler.mip_filter = TextureFilter::kLinear;
  const auto linear = ComputeTextureExplicitLod(1.75F,image,sampler);
  Check(linear.level0 == 1 && linear.level1 == 2 && linear.mip_weight == 0.75F && linear.mip_weight_u8 == 192,
        "textureLod retains trilinear LOD fraction instead of becoming texelFetch");
  sampler.max_lod_u4_6 = sampler.min_lod_u4_6 = 64;
  const auto clamped = ComputeTextureExplicitLod(3,image,sampler);
  Check(clamped.level0 == 1 && clamped.mip_weight == 0, "textureLod honors sampler LOD clamp");
}

struct Harness {
  MemoryPool pool;
  GpuMemorySystem memory;
  sc_core::sc_fifo<PipelineTxn> input, output, fs_input, fs_output, vs_input, vs_output;
  TextureUnit texture;

  Harness(const char *name, MemoryMode mode)
      : memory(mode), input(sc_core::sc_gen_unique_name("input"), 1),
        output(sc_core::sc_gen_unique_name("output"), 1),
        fs_input(sc_core::sc_gen_unique_name("fs_input"), 1),
        fs_output(sc_core::sc_gen_unique_name("fs_output"), 1),
        vs_input(sc_core::sc_gen_unique_name("vs_input"), 1),
        vs_output(sc_core::sc_gen_unique_name("vs_output"), 1), texture(name, pool, &memory) {
    texture.input(input);
    texture.output(output);
    texture.sample_input(fs_input);
    texture.sample_output(fs_output);
    texture.vertex_sample_input(vs_input);
    texture.vertex_sample_output(vs_output);
  }

  void Run(TextureFormat format, unsigned samples, unsigned layers, bool array, bool vertex,
           unsigned epoch) {
    const bool integer = format == TextureFormat::kRgba32Uint || format == TextureFormat::kRgba32Sint;
    const unsigned bpp = TextureBytesPerTexel(format);
    TextureResource resource;
    // A changed base plus changed bytes checks descriptor relocation and
    // cache-coherent new resource state, not only replay of one frozen image.
    resource.gpu_address = kBase + epoch * 0x10000U;
    resource.format = format;
    resource.dimension_type = array ? TextureDimensionType::k2DArray : TextureDimensionType::k2D;
    resource.layer_count = static_cast<std::uint16_t>(layers);
    resource.sample_count = static_cast<std::uint8_t>(samples);
    resource.mip_count = 1;
    resource.mip[0] = {3, 2, 3 * samples * bpp, 0};
    resource.byte_size = resource.mip[0].row_pitch_bytes * 2 * layers;
    std::vector<std::uint8_t> bytes(resource.byte_size + 64, 0xa7);
    std::vector<std::array<std::uint32_t, 4>> expected;
    std::vector<TextureSampleRequest> requests;
    const auto shared = Descriptor(resource);
    for (unsigned layer = 0; layer < layers; ++layer) {
      for (unsigned y = 0; y < 2; ++y) {
        for (unsigned x = 0; x < 3; ++x) {
          for (unsigned sample = 0; sample < samples; ++sample) {
            const unsigned index = ((layer * 2 + y) * 3 + x) * samples + sample;
            std::array<std::uint32_t, 4> value{};
            if (integer) {
              value = {{UINT32_MAX - index - epoch, UINT32_C(0x80000000) + index + epoch,
                        UINT32_C(0x01000001) + index * 3 + epoch, index & 1U}};
              std::memcpy(bytes.data() + index * bpp, value.data(), bpp);
            } else if (format == TextureFormat::kRgba32Float) {
              value = {{FloatBits(static_cast<float>(index + epoch) / 256.0F),
                        FloatBits(-static_cast<float>(index + 1)), FloatBits(2048.5F + epoch), FloatBits(1)}};
              std::memcpy(bytes.data() + index * bpp, value.data(), bpp);
            } else {
              for (unsigned channel = 0; channel < 4; ++channel) {
                const auto byte = static_cast<std::uint8_t>(index * 13 + channel * 17 + epoch);
                bytes[index * bpp + channel] = byte;
                value[channel] = FloatBits(static_cast<float>(byte) / 255.0F);
              }
            }
            expected.push_back(value);
            TextureSampleRequest request;
            request.shader_lane_index = index;
            request.request_id = index;
            request.coordinates[0] = FloatBits(static_cast<float>(x));
            request.coordinates[1] = FloatBits(static_cast<float>(y));
            request.coordinate_count = request.dimension = 2;
            request.component_count = 4;
            request.normalized = 0;
            request.sample_index_present = 1;
            request.sample_index = static_cast<std::uint8_t>(sample);
            request.fcnorm = integer ? 0 : 1;
            request.shader_stage = vertex ? ShaderStage::kVertex : ShaderStage::kFragment;
            std::copy_n(shared.begin(), 4, request.texture_state);
            std::copy_n(shared.begin() + 8, 4, request.sampler_state);
            if (array) {
              const std::uint64_t address = resource.gpu_address +
                  layer * resource.mip[0].row_pitch_bytes * resource.mip[0].height;
              request.texture_address_lo = static_cast<std::uint32_t>(address);
              request.texture_address_hi = static_cast<std::uint32_t>(address >> 32U);
            }
            requests.push_back(request);
          }
        }
      }
    }
    const auto valid_reads = requests.size();
    // These six lanes return all-zero, without reading an adjacent resource
    // or a different sample. All raw request identities remain legal.
    for (unsigned guard = 0; guard < 6; ++guard) {
      auto request = requests.front();
      request.shader_lane_index = static_cast<std::uint32_t>(requests.size());
      request.request_id = requests.size();
      switch (guard) {
      case 0: request.coordinates[0] = FloatBits(-1); break;
      case 1: request.coordinates[0] = FloatBits(3); break;
      case 2: request.coordinates[1] = FloatBits(2); break;
      case 3: request.sample_index = static_cast<std::uint8_t>(samples); break;
      case 4: request.coordinates[0] = FloatBits(std::numeric_limits<float>::infinity()); break;
      case 5: request.coordinates[0] = FloatBits(0.5F); break;
      }
      requests.push_back(request);
      expected.push_back({});
    }
    Submit(resource, shared, bytes, requests, expected, valid_reads, vertex);
  }

  void RunExplicit(TextureDimensionType dimension, bool vertex, unsigned epoch, bool normalized = false) {
    TextureResource resource;
    resource.gpu_address = kBase + 0x40000U + epoch * 0x10000U;
    resource.format = TextureFormat::kRgba8Unorm;
    resource.dimension_type = dimension;
    resource.layer_count = dimension == TextureDimensionType::k2D ? 1 : dimension == TextureDimensionType::kCube ? 6 : 4;
    resource.sample_count = 1; resource.mip_count = 3;
    unsigned total = 0;
    for (unsigned level = 0; level < 3; ++level) {
      const unsigned extent = 4U >> level;
      resource.mip[level] = {extent,extent,extent*4,total};
      total += extent*extent*4*(dimension == TextureDimensionType::k3D ? 4U >> level : resource.layer_count);
    }
    resource.byte_size = total;
    std::vector<std::uint8_t> bytes(total+64,0xa7);
    std::vector<std::array<std::uint32_t,4>> expected;
    std::vector<TextureSampleRequest> requests;
    auto shared = Descriptor(resource);
    if (normalized) {
      shared[8] |= 128U << 23U;
      shared[16] = shared[8];
    }
    for (unsigned level=0; level<3; ++level) {
      const auto &mip=resource.mip[level];
      const unsigned slices=dimension==TextureDimensionType::k3D ? 4U>>level : resource.layer_count;
      for (unsigned layer=0; layer<slices; ++layer)
      for (unsigned y=0; y<mip.height; ++y)
      for (unsigned x=0; x<mip.width; ++x) {
        const unsigned offset=mip.offset_bytes+(layer*mip.height+y)*mip.row_pitch_bytes+x*4;
        std::array<std::uint32_t,4> value{};
        for (unsigned c=0;c<4;++c) {
          const auto b=static_cast<std::uint8_t>(level*71+layer*31+x*13+y*17+c*29+epoch);
          bytes[offset+c]=b; value[c]=FloatBits(b/255.0F);
        }
        TextureSampleRequest q;
        q.shader_lane_index=q.request_id=requests.size();
        q.coordinates[0]=FloatBits(static_cast<float>(x)); q.coordinates[1]=FloatBits(static_cast<float>(y));
        q.coordinates[2]=dimension==TextureDimensionType::k3D ? FloatBits(static_cast<float>(layer)) : 0;
        q.dimension=(dimension==TextureDimensionType::k3D || dimension==TextureDimensionType::kCube) ? 3 : 2; q.coordinate_count=2;
        q.component_count=4; q.normalized=normalized; q.explicit_lod_present=1; q.explicit_lod=FloatBits(static_cast<float>(level));
        if (normalized) {
          const float u=(x+0.5F)/mip.width, v=(y+0.5F)/mip.height;
          q.coordinates[0]=FloatBits(u); q.coordinates[1]=FloatBits(v);
          if (dimension==TextureDimensionType::k3D) q.coordinates[2]=FloatBits((layer+0.5F)/slices);
          if (dimension==TextureDimensionType::kCube) {
            const float s=2*u-1, t=2*v-1;
            const std::array<std::array<float,3>,6> direction{{{1,-t,-s},{-1,-t,s},{s,1,t},{s,-1,-t},{s,-t,1},{-s,-t,-1}}};
            for (unsigned c=0;c<3;++c) q.coordinates[c]=FloatBits(direction[layer][c]);
          }
        }
        q.shader_stage=vertex?ShaderStage::kVertex:ShaderStage::kFragment;
        std::copy_n(shared.begin(),4,q.texture_state); std::copy_n(shared.begin()+8,4,q.sampler_state);
        if (dimension==TextureDimensionType::k2DArray) {
          const std::uint64_t address=resource.gpu_address+layer*resource.mip[0].row_pitch_bytes*resource.mip[0].height;
          q.texture_address_lo=address; q.texture_address_hi=address>>32;
        }
        requests.push_back(q); expected.push_back(value);
      }
    }
    const auto valid=requests.size();
    for (unsigned bad=0;!normalized && bad<5;++bad) {
      auto q=requests.back(); q.shader_lane_index=q.request_id=requests.size();
      if (bad==0) q.coordinates[0]=FloatBits(-1);
      if (bad==1) q.coordinates[1]=FloatBits(1); // last mip is 1x1
      if (bad==2) q.explicit_lod=FloatBits(-1);
      if (bad==3) q.explicit_lod=FloatBits(3);
      if (bad==4) q.coordinates[0]=FloatBits(0.5F);
      requests.push_back(q); expected.push_back({});
    }
    Submit(resource,shared,bytes,requests,expected,valid,vertex,normalized ? 128 : 0);
  }

  void Submit(const TextureResource &resource, const std::vector<std::uint32_t> &shared,
              const std::vector<std::uint8_t> &bytes, const std::vector<TextureSampleRequest> &requests,
              const std::vector<std::array<std::uint32_t,4>> &expected, std::size_t valid_reads, bool vertex,
              std::uint16_t maximum_lod = 0) {
    const auto bpp=TextureBytesPerTexel(resource.format);
    memory.HostWrite(resource.gpu_address, bytes.data(), bytes.size());
    PipelineState state;
    state.memory_mode = memory.mode();
    state.sequence = ++batches;
    state.functional_case = FunctionalCase::kDriverPcoTriangles;
    state.stage = vertex ? PipelineStage::kVertexTexturePending : PipelineStage::kFragmentTexturePending;
    state.texture_sample_requests = StoreNewArray(pool, requests);
    const auto resources = StoreNewArray(pool, std::vector<TextureResource>{resource});
    SamplerState sampler; sampler.max_lod_u4_6=maximum_lod;
    const auto samplers = StoreNewArray(pool, std::vector<SamplerState>{sampler});
    if (vertex) {
      state.vertex_sampled_texture_count = 1;
      state.vertex_pco_abi.shareds = shared.size();
      state.counters.vs_invocations = requests.size();
      state.vertex_texture_resources = resources;
      state.vertex_sampler_states = samplers;
      std::vector<ShaderSharedRegister> registers(shared.size());
      for (std::size_t i = 0; i < shared.size(); ++i)
        registers[i].value = shared[i];
      state.vertex_shared_registers = StoreNewArray(pool, registers);
    } else {
      state.sampled_texture_count = 1;
      state.fragment_pco_abi.shareds = shared.size();
      state.fragment_shader_lane_count = requests.size();
      state.texture_resources = resources;
      state.sampler_states = samplers;
      state.fragment_shared_registers = StoreNewArray(pool, shared);
    }
    const auto handle = pool.Allocate(sizeof(PipelineState));
    StorePipelineState(pool, handle, state);
    const PipelineTxn transaction{handle, batches, batches};
    (vertex ? vs_input : fs_input).write(transaction);
    sc_core::sc_start(sc_core::sc_time(100000, sc_core::SC_NS));
    PipelineTxn completed;
    Check((vertex ? vs_output : fs_output).nb_read(completed) &&
          completed.state.slot == handle.slot && completed.state.generation == handle.generation,
          "sample FIFO completion identity");
    const auto final = LoadPipelineState(pool, handle);
    const auto actual = LoadArray<TextureSampleResponse>(pool, final.texture_sample_responses);
    Check(actual.size() == expected.size() && final.counters.texture_requests == requests.size() &&
          final.counters.texel_fetches == valid_reads, "logical requests versus actual sample reads");
    for (std::size_t index = 0; index < actual.size(); ++index) {
      Check(actual[index].shader_lane_index == index && actual[index].request_id == index &&
            actual[index].shader_stage == (vertex ? ShaderStage::kVertex : ShaderStage::kFragment),
            "response stage/lane identity");
      Check(std::equal(expected[index].begin(), expected[index].end(), actual[index].rgba),
            "exact sample/layer/channel values (not sample zero or resolve)");
    }
    if (memory.mode() == MemoryMode::kDirect)
      Check(final.counters.memory_direct_read_bytes == valid_reads * bpp &&
            final.counters.dram_read_bytes == 0, "direct actual read byte count");
    else if (memory.mode() == MemoryMode::kBypass)
      Check(final.counters.dram_read_bytes == valid_reads * bpp &&
            final.counters.memory_direct_read_bytes == 0, "bypass actual read byte count");
    else
      Check(final.counters.dram_read_bytes > 0 && final.counters.memory_direct_read_bytes == 0,
            "cache mode actual backing reads");
    Check(memory.Readback(resource.gpu_address, bytes.size(), MemoryClient::kFramebufferReadback).data == bytes,
          "read-only sampling preserves all storage and backing sentinels");
    ReleaseFunctionalPayloads(pool, final);
    pool.Release(handle);
    Check(pool.bytes_in_flight() == 0 && pool.allocations() == pool.releases(),
          "sample and state payload ownership balanced");
  }
};
} // namespace

int sc_main(int, char **) {
  try {
    CheckAddresses();
    CheckExplicitAddresses();
    Harness direct("direct", MemoryMode::kDirect);
    Harness bypass("bypass", MemoryMode::kBypass);
    Harness cache("cache", MemoryMode::kCache);
    for (Harness *harness : {&direct, &bypass, &cache})
      for (TextureFormat format : {TextureFormat::kRgba8Unorm, TextureFormat::kRgba32Uint,
                                   TextureFormat::kRgba32Sint, TextureFormat::kRgba32Float})
        for (unsigned samples : {1U, 2U, 4U, 8U})
          for (unsigned dimension = 0; dimension < 3; ++dimension)
            for (bool vertex : {false, true})
              for (unsigned epoch : {0U, 1U})
                harness->Run(format, samples, dimension == 2 ? 3U : 1U,
                             dimension != 0, vertex, epoch);
    for (Harness *harness : {&direct,&bypass,&cache})
      for (auto dimension : {TextureDimensionType::k2D,TextureDimensionType::k2DArray,TextureDimensionType::k3D})
        for (bool vertex : {false,true})
          for (unsigned epoch : {0U,1U}) harness->RunExplicit(dimension,vertex,epoch);
    for (Harness *harness : {&direct,&bypass,&cache})
      for (auto dimension : {TextureDimensionType::k2D,TextureDimensionType::k2DArray,TextureDimensionType::k3D,TextureDimensionType::kCube})
        for (bool vertex : {false,true}) harness->RunExplicit(dimension,vertex,0,true);
    std::cout << "texture_multisample_test: PASS batches=" << batches << " checks=" << checks << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "texture_multisample_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
