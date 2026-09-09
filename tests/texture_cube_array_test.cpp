// SPDX-License-Identifier: MIT
// Actual TextureUnit FIFO / unified-memory CubeArray tests. Distinct cubes,
// faces and mips form an input-derived oracle, never a saved output image.
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
using Direction = std::array<float, 3>;
using Weights = std::array<float, 6>;
unsigned checks = 0, batches = 0;
std::string negative;
void Check(bool ok, const char *message) {
  ++checks;
  if (!ok) throw std::runtime_error(message);
}
std::uint32_t Bits(float value) {
  std::uint32_t bits; std::memcpy(&bits, &value, 4); return bits;
}
float Float(std::uint32_t bits) {
  float value; std::memcpy(&value, &bits, 4); return value;
}
Weights Face(unsigned face) { Weights w{}; w[face] = 1; return w; }
Weights Edge(unsigned a, unsigned b) {
  Weights w{}; w[a] = w[b] = 0.5F; return w;
}
Weights Corner(unsigned a, unsigned b, unsigned c) {
  Weights w{}; w[a] = w[b] = w[c] = 1.0F / 3.0F; return w;
}
struct Case {
  const char *name;
  Direction direction;
  unsigned nearest_face;
  Weights seamless_weights;
};
enum class Lod { kImplicit, kImplicitMinOne, kPinnedZero, kExplicitZero, kExplicitOne };

std::uint8_t Color(unsigned mip, unsigned face, unsigned channel, bool monochrome) {
  // Every mip and face is independently distinguishable, with nonzero alpha.
  return static_cast<std::uint8_t>(21U + mip * 31U + (face / 6U) * 67U +
      (monochrome ? 0U : (face % 6U) * 7U) + channel * 3U);
}

struct Harness {
  MemoryPool pool;
  GpuMemorySystem memory;
  sc_core::sc_fifo<PipelineTxn> input, output, sample_input, sample_output;
  TextureUnit unit;
  Harness(const char *name, MemoryMode mode)
      : memory(mode), input(sc_core::sc_gen_unique_name("input"), 1),
        output(sc_core::sc_gen_unique_name("output"), 1),
        sample_input(sc_core::sc_gen_unique_name("sample_input"), 1),
        sample_output(sc_core::sc_gen_unique_name("sample_output"), 1),
        unit(name, pool, &memory) {
    unit.input(input); unit.output(output);
    unit.sample_input(sample_input); unit.sample_output(sample_output);
  }

  void Run(const std::vector<Case> &cases, bool linear, Lod lod, bool monochrome,
           unsigned cube, unsigned expected_level = std::numeric_limits<unsigned>::max()) {
    Check(!cases.empty(), "empty cube test batch");
    const unsigned level = expected_level != std::numeric_limits<unsigned>::max() ? expected_level :
        lod == Lod::kImplicitMinOne || lod == Lod::kExplicitOne ? 1 : 0;
    const bool explicit_lod = lod == Lod::kExplicitZero || lod == Lod::kExplicitOne;
    TextureResource resource;
    resource.gpu_address = UINT64_C(0x8000000000) + (++batches) * UINT64_C(0x10000);
    resource.format = TextureFormat::kRgba8Unorm;
    resource.dimension_type = TextureDimensionType::kCubeArray;
    resource.layer_count = 18; resource.mip_count = 2;
    resource.mip[0] = {8, 8, 32, 0};
    resource.mip[1] = {4, 4, 16, 8 * 8 * 4 * 18};
    resource.byte_size = (8 * 8 + 4 * 4) * 4 * 18;
    std::vector<std::uint8_t> bytes(resource.byte_size + 64, 0xa7);
    for (unsigned mip = 0; mip < 2; ++mip)
      for (unsigned face = 0; face < 18; ++face)
        for (unsigned y = 0; y < resource.mip[mip].height; ++y)
          for (unsigned x = 0; x < resource.mip[mip].width; ++x)
            for (unsigned c = 0; c < 4; ++c)
              bytes[resource.mip[mip].offset_bytes +
                    (face * resource.mip[mip].height + y) * resource.mip[mip].row_pitch_bytes + x * 4 + c] =
                  Color(mip, face, c, monochrome);
    memory.HostWrite(resource.gpu_address, bytes.data(), bytes.size());
    SamplerState sampler;
    sampler.min_filter = sampler.mag_filter = linear ? TextureFilter::kLinear : TextureFilter::kNearest;
    sampler.min_lod_u4_6 = lod == Lod::kImplicitMinOne ? 64 : 0;
    sampler.max_lod_u4_6 = lod == Lod::kPinnedZero ? 0 : 64;
    // Independent raw Rogue STRIDE image / sampler descriptors. Cube direction
    // and layer identity travel in the request/resource ABI, as on this path.
    const std::uint64_t image0 = UINT64_C(1) | (UINT64_C(3) << 5) |
        (UINT64_C(2) << 8) | (UINT64_C(1) << 11) | (UINT64_C(12) << 27) |
        (UINT64_C(7) << 34) | (UINT64_C(7) << 48);
    const std::uint64_t image1 = ((resource.gpu_address >> 2) << 16) |
        UINT64_C(2) | (UINT64_C(2) << 4) | (UINT64_C(1) << 15);
    const std::uint64_t sampler0 = UINT64_C(0xfff) |
        (static_cast<std::uint64_t>(sampler.min_lod_u4_6) << 13) |
        (static_cast<std::uint64_t>(sampler.max_lod_u4_6) << 23) |
        (linear ? (UINT64_C(1) << 36) | (UINT64_C(1) << 38) : 0);
    std::vector<std::uint32_t> shared(20, 0);
    shared[0] = static_cast<std::uint32_t>(image0); shared[1] = static_cast<std::uint32_t>(image0 >> 32);
    shared[2] = static_cast<std::uint32_t>(image1); shared[3] = static_cast<std::uint32_t>(image1 >> 32);
    shared[4] = 8 * 8 * 4;
    shared[8] = static_cast<std::uint32_t>(sampler0); shared[9] = static_cast<std::uint32_t>(sampler0 >> 32);
    const std::uint64_t gather = sampler0 | (UINT64_C(1) << 36) | (UINT64_C(1) << 38);
    shared[16] = static_cast<std::uint32_t>(gather); shared[17] = static_cast<std::uint32_t>(gather >> 32);
    std::vector<TextureSampleRequest> requests(cases.size());
    std::vector<FragmentShaderLane> lanes(cases.size());
    for (std::size_t i = 0; i < cases.size(); ++i) {
      auto &q = requests[i];
      q.shader_lane_index = static_cast<std::uint32_t>(i); q.request_id = i;
      q.quad_id = static_cast<std::uint32_t>(i / 4); q.quad_lane = static_cast<std::uint8_t>(i % 4);
      q.coordinate_count = 2; q.dimension = 3; q.component_count = 4;
      q.normalized = 1; q.fcnorm = 1; q.shader_stage = ShaderStage::kFragment;
      const auto address = resource.gpu_address + cube * 6U * 8U * 8U * 4U;
      q.texture_address_lo = static_cast<std::uint32_t>(address);
      q.texture_address_hi = static_cast<std::uint32_t>(address >> 32U);
      q.explicit_lod_present = explicit_lod ? 1 : 0;
      q.explicit_lod = explicit_lod ? Bits(static_cast<float>(level)) : 0;
      for (unsigned c = 0; c < 3; ++c) q.coordinates[c] = Bits(cases[i].direction[c]);
      std::copy_n(shared.begin(), 4, q.texture_state);
      std::copy_n(shared.begin() + 8, 4, q.sampler_state);
      // Keep true helpers in each derivative quartet. They must still sample.
      lanes[i].x = static_cast<std::int32_t>(i % 2);
      lanes[i].y = static_cast<std::int32_t>((i % 4) / 2);
      lanes[i].helper = i % 4 == 1 ? 0 : 1;
    }
    if (!negative.empty()) {
      if (negative == "offset") requests[0].spatial_offsets[0] = 1;
      if (negative == "gather") for (auto &q : requests) q.gather = 1;
      if (negative == "bias") { requests[0].lod_bias_present = 1; requests[0].lod_bias = Bits(1); }
      if (negative == "face-address") requests[0].texture_address_lo += 8*8*4;
      if (negative == "low-address") { requests[0].texture_address_lo = 0; requests[0].texture_address_hi = 0; }
      if (negative == "end-address") { const auto a=resource.gpu_address+18U*8U*8U*4U; requests[0].texture_address_lo=static_cast<std::uint32_t>(a);requests[0].texture_address_hi=a>>32; }
      if (negative == "word4") shared[4] *= 6;
      if (negative == "raw-depth") shared[2] += 1U << 4;
      if (negative == "shadow") shared[12] = 3;
    }
    PipelineState state;
    state.sequence = batches; state.memory_mode = memory.mode();
    state.functional_case = FunctionalCase::kDriverPcoTriangles;
    state.stage = PipelineStage::kFragmentTexturePending;
    state.sampled_texture_count = 1; state.fragment_pco_abi.shareds = shared.size();
    state.fragment_shader_lane_count = requests.size();
    state.texture_resources = StoreNewArray(pool, std::vector<TextureResource>{resource});
    state.sampler_states = StoreNewArray(pool, std::vector<SamplerState>{sampler});
    state.fragment_shared_registers = StoreNewArray(pool, shared);
    state.texture_sample_requests = StoreNewArray(pool, requests);
    state.fragment_shader_lanes = StoreNewArray(pool, lanes);
    const auto handle = pool.Allocate(sizeof(PipelineState));
    StorePipelineState(pool, handle, state);
    sample_input.write(PipelineTxn{handle, batches, batches});
    sc_core::sc_start(sc_core::sc_time(100000, sc_core::SC_NS));
    PipelineTxn done;
    Check(sample_output.nb_read(done) && done.state.slot == handle.slot &&
          done.state.generation == handle.generation, "cube FIFO completion identity");
    const auto final = LoadPipelineState(pool, handle);
    const auto responses = LoadArray<TextureSampleResponse>(pool, final.texture_sample_responses);
    const auto retained = LoadArray<TextureSampleRequest>(pool, final.texture_sample_requests);
    const auto reads = requests.size() * (linear ? 4U : 1U);
    Check(responses.size() == requests.size() && final.counters.texture_requests == requests.size() &&
          final.counters.texel_fetches == reads, "nonfinite cube uses actual texel reads, including helpers");
    Check(retained.size() == requests.size(), "raw request payload retained");
    for (std::size_t i = 0; i < responses.size(); ++i) {
      Check(responses[i].request_id == i && responses[i].shader_lane_index == i &&
            responses[i].shader_stage == ShaderStage::kFragment, "cube response lane identity");
      Check(std::equal(requests[i].coordinates, requests[i].coordinates + 3, retained[i].coordinates),
            "address sanitization must not replace raw NaN/Inf request values");
      const Weights weights = linear ? cases[i].seamless_weights : Face(cases[i].nearest_face);
      for (unsigned c = 0; c < 4; ++c) {
        float expected = 0;
        for (unsigned face = 0; face < 6; ++face)
          expected += weights[face] * (static_cast<float>(Color(level, cube * 6U + face, c, monochrome)) / 255.0F);
        const float actual = Float(responses[i].rgba[c]);
        // Nearest is exact; seamless float sums may differ only in operation
        // ordering from this independently weighted constant-face oracle.
        const float tolerance = linear ? 8 * std::numeric_limits<float>::epsilon() : 0;
        if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
          std::cerr << "case=" << cases[i].name << " linear=" << linear << " lod=" << static_cast<unsigned>(lod)
                    << " channel=" << c << " expected=" << expected << " actual=" << actual << '\n';
          throw std::runtime_error("synthetic cube face/mip/filter color mismatch");
        }
        ++checks;
      }
    }
    if (memory.mode() == MemoryMode::kDirect)
      Check(final.counters.memory_direct_read_bytes == reads * 4 && final.counters.dram_read_bytes == 0,
            "cube direct memory fetch bytes");
    else if (memory.mode() == MemoryMode::kBypass)
      Check(final.counters.dram_read_bytes == reads * 4 && final.counters.memory_direct_read_bytes == 0,
            "cube bypass memory fetch bytes");
    else
      Check(final.counters.dram_read_bytes > 0 && final.counters.memory_direct_read_bytes == 0,
            "cube cache performs backing memory reads");
    Check(memory.Readback(resource.gpu_address, bytes.size(), MemoryClient::kFramebufferReadback).data == bytes,
          "cube fetch leaves storage and guard bytes intact");
    ReleaseFunctionalPayloads(pool, final); pool.Release(handle);
    Check(pool.bytes_in_flight() == 0 && pool.allocations() == pool.releases(), "cube pool ownership balances");
  }
};
} // namespace

int sc_main(int argc, char **argv) {
  try {
    if (argc == 2) negative=argv[1];
    else if (argc != 1) return 2;
    TextureResource r;
    r.dimension_type=TextureDimensionType::kCubeArray;r.layer_count=18;
    r.mip_count=2;r.mip[0]={8,8,32,0};r.mip[1]={4,4,16,8*8*4*18};
    r.byte_size=(8*8+4*4)*4*18;
    ValidateTextureCubeArrayLayout(r);
    for (unsigned cube=0;cube<3;++cube)
      Check(TextureCubeArrayBaseFace(r,0x1ffffff00ULL,0x1ffffff00ULL+cube*6*256)==cube*6,
            "independent exact cube base with 64-bit carry");
    const auto refuse=[&](auto action,const char *why){bool bad=false;try{action();}catch(const std::exception&){bad=true;}Check(bad,why);};
    for(unsigned mutation=0;mutation<18;++mutation){auto b=r;
      switch(mutation){
        case 0:b.layer_count=0;break;case 1:b.layer_count=17;break;
        case 2:b.layer_count=12294;break;case 3:b.sample_count=4;break;
        case 4:b.block_width=4;break;case 5:b.block_height=4;break;
        case 6:b.mip_count=0;break;case 7:b.mip_count=16;break;
        case 8:b.mip[0].height=7;break;case 9:b.mip[0].row_pitch_bytes=36;break;
        case 10:b.mip[1].width=3;break;case 11:b.mip[1].offset_bytes++;break;
        case 12:b.byte_size--;break;case 13:b.byte_size++;break;
        case 14:b.mip[2].width=1;break;case 15:b.format=TextureFormat::kAstcLdr;break;
        case 16:b.dimension_type=TextureDimensionType::kCube;break;
        case 17:b.mip[0].width=b.mip[0].height=UINT32_MAX;break;
      }
      refuse([&]{ValidateTextureCubeArrayLayout(b);},"invalid cube array layout refused");
    }
    for(uint64_t address:{UINT64_C(0x1000)-1,UINT64_C(0x1000)+1,UINT64_C(0x1000)+256,
                          UINT64_C(0x1000)+18*256,UINT64_MAX})
      refuse([&]{TextureCubeArrayBaseFace(r,0x1000,address);},"invalid cube-aligned TAO refused");
    const std::vector<Case> cases{
      {"positive_x",{1,0,0},0,Face(0)}, {"negative_x",{-1,0,0},1,Face(1)},
      {"positive_y",{0,1,0},2,Face(2)}, {"negative_y",{0,-1,0},3,Face(3)},
      {"positive_z",{0,0,1},4,Face(4)}, {"negative_z",{0,0,-1},5,Face(5)},
      {"edge_xy",{1,1,0},0,Edge(0,2)}, {"edge_xz",{1,0,1},0,Edge(0,4)},
      {"corner_xyz",{1,1,1},0,Corner(0,2,4)},
    };
    Harness direct("cube_array_direct",MemoryMode::kDirect);
    Harness bypass("cube_array_bypass",MemoryMode::kBypass);
    Harness cache("cube_array_cache",MemoryMode::kCache);
    if(!negative.empty()) {
      direct.Run(std::vector<Case>(4,cases[0]),false,Lod::kExplicitZero,false,1);
      throw std::runtime_error("negative test unexpectedly completed");
    }
    for(Harness *h:{&direct,&bypass,&cache})for(unsigned cube=0;cube<3;++cube)
      for(bool linear:{false,true})for(Lod lod:{Lod::kImplicit,Lod::kImplicitMinOne,Lod::kExplicitZero,Lod::kExplicitOne})
        for(const auto &c:cases)h->Run(std::vector<Case>(4,c),linear,lod,false,cube);
    std::cout<<"CubeArray TPU: "<<checks<<" checks, "<<batches<<" batches PASS\n";
    return 0;
  }catch(const std::exception &e){
    const std::string message=e.what();
    if(!negative.empty()&&message.find("unexpectedly completed")==std::string::npos&&
       (message.find("TextureUnit unsupported cube array sample mode")!=std::string::npos||
        message.find("TextureUnit cube array TAO")!=std::string::npos||
        message.find("word4 is not the image layer size")!=std::string::npos||
        message.find("raw cube count")!=std::string::npos||
        message.find("cube array requires ordinary")!=std::string::npos)){
      std::cout<<"CubeArray TPU refused "<<negative<<": "<<message<<" PASS\n";return 0;
    }
    std::cerr<<"CubeArray TPU after "<<checks<<" checks: "<<message<<'\n';return 1;
  }
}
