// SPDX-License-Identifier: MIT
// Genuine Mesa PCO -> USC -> TextureUnit FIFO -> modeled memory -> WDF/PIXOUT.
// Constant-color mip levels give an independent analytic oracle; no RDC data.
#include "common/pipeline_state.h"
#include "memory/gpu_memory_system.h"
#include "shader/pco_iss.h"
#include "shader/usc_cluster.h"
#include "texture/texture_unit.h"
#include "pco_texture_bias_fixtures.h"
#include <systemc>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using namespace pvrgpu::stub;
// Host pool expectations in check_model_pipeline.py use these explicit ABI
// sizes and the independently tested 256-quad residency cap, not run counters.
static_assert(sizeof(TextureSampleRequest)==128);
static_assert(sizeof(TextureSampleResponse)==40);
static_assert(sizeof(PcoFragmentContinuation)==1448);
static_assert(sizeof(PcoInstruction)==144);
unsigned checks = 0, batches = 0;
void Check(bool value, const char *why) { ++checks; if (!value) throw std::runtime_error(why); }
uint32_t Bits(float x) { uint32_t bits; std::memcpy(&bits,&x,4); return bits; }
float Float(uint32_t x) { float f; std::memcpy(&f,&x,4); return f; }
template<class F> void Refuse(F fn) {
  try { fn(); } catch (const std::exception &) { ++checks; return; }
  throw std::runtime_error("invalid bias metadata was accepted");
}
struct Program { PcoDecodedProgram decoded; DriverPcoStageAbi abi; };
Program Load(const std::string &root, unsigned kind) {
  if(root.empty()) {
    const auto &a=TextureBiasPcoAbi(kind);
    Program p{DecodePcoProgram(ShaderStage::kFragment,TextureBiasPcoBinary(kind)),{}};
    p.abi.temps=a[0];p.abi.shareds=a[1];p.abi.push_constant_start=a[2];
    p.abi.push_constant_count=a[3];p.abi.coefficients=a[4];
    return p;
  }
  std::ifstream in(root + "/bias-" + std::to_string(kind) + ".pco",std::ios::binary);
  Check(bool(in), "missing real Mesa PCO");
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)),{});
  Program p{DecodePcoProgram(ShaderStage::kFragment,bytes),{}};
  std::ifstream abi(root + "/bias-" + std::to_string(kind) + ".abi");
  abi >> p.abi.temps >> p.abi.shareds >> p.abi.push_constant_start
      >> p.abi.push_constant_count >> p.abi.coefficients;
  Check(bool(abi) && p.abi.coefficients == 16,"real compiler ABI changed fixture shape");
  unsigned samples=0,biases=0;
  for (const auto &i:p.decoded.instructions) if(i.opcode==PcoOpcode::kTextureSample) {
    ++samples; biases += i.texture_lod_bias;
    Check(!i.texture_lod_replace && !i.texture_address_offset,"producer did not emit implicit BIAS");
  }
  Check(samples==(kind==3?5U:1U) && biases==(kind==3?3U:kind==5?0U:1U),
        "real compiler lost live texture bias operations/AUTO control");
  return p;
}
float Lambda(float raw,float bias,float low,float high) {
  if(std::isnan(bias)) bias=0;
  if(std::isinf(bias)) return bias>0?high:low;
  return std::clamp(raw+bias,low,high);
}
unsigned Color(unsigned level,unsigned channel) { return 20+level*64+channel*7; }
float Expected(float raw,float bias,float low,float high,bool miplinear,unsigned channel) {
  const float lambda=std::clamp(Lambda(raw,bias,low,high),0.0F,3.0F);
  if(!miplinear) return float(Color(unsigned(std::nearbyint(lambda)),channel))/255.0F;
  const unsigned lo=unsigned(std::floor(lambda)), hi=std::min(3U,lo+1);
  const unsigned weight=unsigned((lambda-std::floor(lambda))*256);
  // Chosen quarter/integer biases make the independent fixed-point blend exact.
  return (Color(lo,channel)+(Color(hi,channel)-Color(lo,channel))*weight/256.0F)/255.0F;
}
std::vector<uint32_t> Planes(unsigned kind,float raw,float bias) {
  std::vector<uint32_t> result(16,0); result[2]=Bits(1);
  const float step=std::exp2(raw)/8;
  auto plane=[&](unsigned c,float dx,float dy,float base) {
    result[4+c*4]=Bits(dx); result[5+c*4]=Bits(dy); result[6+c*4]=Bits(base);
  };
  if(kind==2) { plane(0,0,0,1); plane(1,0,-2*step,0); plane(2,-2*step,0,0); }
  else if(kind==1||kind==5) { plane(0,0,0,0.25F); plane(1,0,0,0.25F); plane(2,step,0,0.25F); }
  else { plane(0,step,0,0.25F); plane(1,0,step,0.25F); plane(2,kind==4?1:0,kind==4?2:0,bias); }
  return result;
}

void RawPrepared(const Program &p,unsigned kind,float bias) {
  if(kind==3) return;
  PcoPreparedFragmentProgram prepared(p.decoded.summary,p.decoded.instructions);
  for(unsigned lane=0;lane<4;++lane) {
    PcoFragmentExecutionContext c;
    c.shared_count=uint16_t(p.abi.shareds); c.coefficient_count=16;
    c.sample_x=Bits(float(lane&1)); c.sample_y=Bits(float(lane>>1));
    auto planes=Planes(kind,-1,bias); std::copy(planes.begin(),planes.end(),c.coefficients.begin());
    if(p.abi.push_constant_count) c.shared_registers[p.abi.push_constant_start]=Bits(bias);
    const auto raw=ExecuteFragmentPco(p.decoded.summary,p.decoded.instructions,c);
    const auto owned=ExecuteFragmentPco(prepared,c);
    const auto expected=kind==5?0:kind==4?Bits(bias+float(lane)):Bits(bias);
    const auto present=kind==5?0:1;
    Check(raw.suspended && owned.suspended && raw.texture_request_valid && owned.texture_request_valid,
          "native BIAS did not suspend");
    Check(raw.texture_request.lod_bias_present==present && raw.texture_request.lod_bias==expected &&
          owned.texture_request.lod_bias==expected && owned.texture_request.lod_bias_present==present,
          "native per-pixel bias bits were dropped or rewritten");
    Check(raw.texture_request.coordinates==owned.texture_request.coordinates &&
          raw.continuation.temporaries==owned.continuation.temporaries &&
          raw.continuation.program_signature==owned.continuation.program_signature,
          "raw/prepared BIAS execution differs");
    c.texture_response_valid=1; c.texture_response={Bits(1),Bits(2),Bits(3),Bits(4)};
    c.continuation=raw.continuation;
    const auto done=ExecuteFragmentPco(p.decoded.summary,p.decoded.instructions,c);
    const auto done_owned=ExecuteFragmentPco(prepared,c);
    Check(!done.suspended && done.written_mask==15 && done.executed_instructions.texture==1 &&
          done.pixel_outputs==done_owned.pixel_outputs &&
          std::equal(c.texture_response.begin(),c.texture_response.end(),done.pixel_outputs.begin()),
          "BIAS response did not resume WDF/PIXOUT exactly once");
    auto mutated=p.decoded.instructions;
    const auto smp=std::find_if(mutated.begin(),mutated.end(),[](const auto&i){return i.opcode==PcoOpcode::kTextureSample;});
    smp->texture_lod_bias=kind==5?1:0;
    Refuse([&]{ExecuteFragmentPco(p.decoded.summary,mutated,c);});
    smp->texture_lod_bias=2;
    Refuse([&]{PcoPreparedFragmentProgram invalid(p.decoded.summary,mutated);});
  }
}

void RejectUnsupportedMetadata(const Program &p) {
  auto instructions=p.decoded.instructions;
  const auto index=std::find_if(instructions.begin(),instructions.end(),
    [](const auto &i){return i.opcode==PcoOpcode::kTextureSample;})-instructions.begin();
  for(unsigned malformed=0;malformed<6;++malformed) {
    auto changed=instructions;
    auto &i=changed[index];
    if(malformed==0) i.texture_lod_bias=2;
    if(malformed==1) i.texture_lod_replace=1;
    if(malformed==2) i.texture_address_offset=1;
    if(malformed==3) i.texture_non_normalized_coords=1;
    if(malformed==4) i.texture_sample_index_present=1;
    if(malformed==5) i.opcode=PcoOpcode::kNop;
    Check(!HasCanonicalTextureLodMode(i),"noncanonical BIAS instruction accepted");
    Refuse([&]{PcoPreparedFragmentProgram invalid(p.decoded.summary,changed);});
  }
  // Shader bias is a distinct operand. Do not silently accept nonzero sampler
  // DADJUST while only the shader-side feature is implemented.
  for(uint32_t dadjust:{0U,1U,4094U,4096U,8191U}) {
    std::array<uint32_t,4> words{dadjust,0,0,0};
    Refuse([&]{DecodeRogueTextureSamplerDescriptor(words);});
  }
  std::array<uint32_t,4> zero_sampler{4095,0,0,0};
  (void)DecodeRogueTextureSamplerDescriptor(zero_sampler);
  ++checks;
}

struct Harness {
  MemoryPool pool; GpuMemorySystem memory;
  sc_core::sc_fifo<PipelineTxn> input,output,to_texture,from_texture,unused_in,unused_out;
  UscCluster usc; TextureUnit texture;
  Harness(const char *name,MemoryMode mode):memory(mode),
    input(sc_core::sc_gen_unique_name("bias_in"),1),output(sc_core::sc_gen_unique_name("bias_out"),1),
    to_texture(sc_core::sc_gen_unique_name("bias_req"),1),from_texture(sc_core::sc_gen_unique_name("bias_rsp"),1),
    unused_in(sc_core::sc_gen_unique_name("unused_in"),1),unused_out(sc_core::sc_gen_unique_name("unused_out"),1),
    usc((std::string(name)+"_usc").c_str(),pool,ShaderStage::kFragment,&memory),
    texture((std::string(name)+"_texture").c_str(),pool,&memory) {
      usc.input(input); usc.output(output); usc.texture_request_output(to_texture); usc.texture_response_input(from_texture);
      texture.input(unused_in); texture.output(unused_out); texture.sample_input(to_texture); texture.sample_output(from_texture);
    }
  void Run(const Program&p,unsigned kind,float raw,float bias,bool miplinear,float low,float high,
           bool image_linear=false) {
    ++batches;
    const bool volume=kind==1||kind==5;
    TextureResource r; r.gpu_address=UINT64_C(0x8000000000)+batches*UINT64_C(0x10000);
    r.format=TextureFormat::kRgba8Unorm; r.mip_count=4;
    r.dimension_type=volume?TextureDimensionType::k3D:kind==2?TextureDimensionType::kCube:TextureDimensionType::k2D;
    r.layer_count=volume?8:kind==2?6:1;
    for(unsigned level=0,size=8;level<4;++level,size>>=1) {
      r.mip[level]={size,size,size*4,r.byte_size};
      r.byte_size+=size*size*4*(volume?size:r.layer_count);
    }
    std::vector<uint8_t> bytes(r.byte_size+128,0xA7);
    for(unsigned level=0;level<4;++level) {
      const auto &m=r.mip[level]; const unsigned layers=volume?std::max(1U,8U>>level):r.layer_count;
      for(unsigned texel=0;texel<m.width*m.height*layers;++texel)
        for(unsigned c=0;c<4;++c) bytes[m.offset_bytes+texel*4+c]=uint8_t(Color(level,c));
    }
    memory.HostWrite(r.gpu_address,bytes.data(),bytes.size());
    SamplerState sampler; sampler.min_lod_u4_6=uint16_t(low*64); sampler.max_lod_u4_6=uint16_t(high*64);
    sampler.mip_filter=miplinear?TextureFilter::kLinear:TextureFilter::kNearest;
    sampler.min_filter=sampler.mag_filter=image_linear?TextureFilter::kLinear:TextureFilter::kNearest;
    const uint64_t image0=4ULL|(3ULL<<5)|(2ULL<<8)|(1ULL<<11)|(12ULL<<27)|(7ULL<<34)|(7ULL<<48);
    const uint64_t image1=((r.gpu_address>>2)<<16)|(4ULL<<60)|(1ULL<<15)|7;
    const uint64_t sampler0=0xfffULL|(uint64_t(sampler.min_lod_u4_6)<<13)|
      (uint64_t(sampler.max_lod_u4_6)<<23)|(miplinear?(1ULL<<40):0)|
      (image_linear?((1ULL<<36)|(1ULL<<38)):0);
    std::vector<uint32_t> shared(p.abi.shareds,0);
    shared[0]=uint32_t(image0); shared[1]=uint32_t(image0>>32); shared[2]=uint32_t(image1); shared[3]=uint32_t(image1>>32);
    shared[4]=uint32_t(r.byte_size); shared[8]=uint32_t(sampler0); shared[9]=uint32_t(sampler0>>32);
    const uint64_t gather=sampler0|(1ULL<<36)|(1ULL<<38);
    shared[16]=uint32_t(gather); shared[17]=uint32_t(gather>>32);
    if(kind==3) std::copy_n(shared.begin(),20,shared.begin()+20);
    if(p.abi.push_constant_count) shared[p.abi.push_constant_start]=Bits(bias);
    const bool all_live=kind==4;
    std::vector<FragmentInvocation> invocations(all_live?4:1);
    std::vector<FragmentShaderLane> lanes(4);
    FragmentQuad quad; quad.quad_id=batches; quad.submit_ordinal=batches;
    quad.coverage_mask=all_live?15:1; quad.helper_mask=all_live?0:14; quad.write_mask=quad.coverage_mask;
    for(unsigned lane=0;lane<4;++lane) {
      auto &l=lanes[lane]; l.x=lane&1; l.y=lane>>1; l.quad_id=batches; l.quad_lane=uint8_t(lane);
      l.submit_ordinal=batches; l.helper=(!all_live&&lane)?1:0; l.visible_invocation_index=l.helper?UINT32_MAX:lane;
      quad.invocation_indices[lane]=lane;
      if(!l.helper) { auto&i=invocations[lane]; i.x=l.x;i.y=l.y;i.quad_id=l.quad_id;i.quad_lane=l.quad_lane;i.submit_ordinal=batches; }
    }
    UscFragmentTask task; task.coefficient_dword_count=16;
    DrawListStats stats; stats.fragment.program_groups=p.decoded.summary.group_count;
    stats.fragment.program_instructions=p.decoded.summary.instruction_count;
    const auto counts=CountPcoInstructions(p.decoded.instructions,false);
    stats.fragment.program_alu_instructions=counts.alu; stats.fragment.program_tex_instructions=counts.texture;
    stats.fragment.program_memory_instructions=counts.memory; stats.fragment.program_recorded=1;
    PipelineState s; s.width=s.height=8; s.sequence=batches; s.memory_mode=memory.mode();
    s.functional_case=FunctionalCase::kDriverPcoTriangles; s.stage=PipelineStage::kFragmentIssued;
    s.fragment_program_summary=p.decoded.summary; s.fragment_pco_abi=p.abi;
    s.fragment_position_count=4; s.fragment_varying_start=4;s.fragment_varying_count=12;
    s.position_output_count=4;s.varying_output_start=4;s.varying_output_count=3;
    s.sampled_texture_count=kind==3?2:1;s.active_fragment_invocations=invocations.size();s.fragment_shader_lane_count=4;
    s.fragment_groups=1;s.counters.drawlists=1;
    s.drawlist_stats=StoreNewArray(pool,std::vector<DrawListStats>{stats});
    s.fragment_instructions=StoreNewArray(pool,p.decoded.instructions);
    s.fragment_invocations=StoreNewArray(pool,invocations);s.fragment_shader_lanes=StoreNewArray(pool,lanes);
    s.fragment_quads=StoreNewArray(pool,std::vector<FragmentQuad>{quad});
    s.usc_fragment_tasks=StoreNewArray(pool,std::vector<UscFragmentTask>{task});
    s.usc_coefficient_banks=StoreNewArray(pool,Planes(kind,raw,bias));
    s.fragment_shared_registers=StoreNewArray(pool,shared);
    std::vector<TextureResource> resources{r};std::vector<SamplerState> samplers{sampler};
    if(kind==3) { resources.push_back(r);resources.back().descriptor_set=1;
      samplers.push_back(sampler);samplers.back().descriptor_set=1; }
    s.texture_resources=StoreNewArray(pool,resources);
    s.sampler_states=StoreNewArray(pool,samplers);
    auto handle=pool.Allocate(sizeof(PipelineState));StorePipelineState(pool,handle,s);
    input.write(PipelineTxn{handle,batches,batches});sc_core::sc_start(sc_core::sc_time(100000,sc_core::SC_NS));
    PipelineTxn done;Check(output.nb_read(done)&&done.state.slot==handle.slot&&
      done.state.generation==handle.generation,"native BIAS FIFO did not complete");
    s=LoadPipelineState(pool,handle);const auto pixels=LoadArray<FragmentOutput>(pool,s.fragment_outputs);
    const uint64_t samples=kind==3?20:4,
      reads=samples*(miplinear?2:1)*(image_linear?4:1)*(image_linear&&volume?2:1);
    Check(pixels.size()==invocations.size()&&s.counters.texture_requests==samples&&s.counters.texel_fetches==reads,
          "real BIAS request/texel/helper count mismatch");
    for(unsigned lane=0;lane<pixels.size();++lane)
      for(unsigned c=0;c<4;++c) {
        const float expected=kind==3?
          Expected(raw,0,low,high,miplinear,c)*2+Expected(raw,1,low,high,miplinear,c)+
          Expected(raw,2,low,high,miplinear,c)+Expected(raw,3,low,high,miplinear,c):
          Expected(raw,bias+(kind==4?float(lane):0),low,high,miplinear,c);
        const float actual=Float(pixels[lane].pixel_output[c]);
        if(!std::isfinite(actual)||std::fabs(expected-actual)>2e-6F) {
          std::cerr<<"kind="<<kind<<" raw="<<raw<<" bias="<<bias<<" miplinear="<<miplinear
            <<" lane="<<lane<<" c="<<c<<" expected="<<expected<<" actual="<<actual<<'\n';
          throw std::runtime_error("actual mip color differs from independent bias oracle");
        } ++checks;
      }
    if(memory.mode()==MemoryMode::kDirect)
      Check(s.counters.memory_direct_read_bytes==reads*4&&s.counters.dram_read_bytes==0,
            "BIAS direct mode must count real texel bytes");
    else if(memory.mode()==MemoryMode::kBypass)
      Check(s.counters.dram_read_bytes==reads*4&&s.counters.memory_direct_read_bytes==0,
            "BIAS bypass mode must count real texel bytes");
    else Check(s.counters.dram_read_bytes>0&&s.counters.memory_direct_read_bytes==0,
               "BIAS cache must issue real backing reads");
    Check(memory.Readback(r.gpu_address,bytes.size(),MemoryClient::kFramebufferReadback).data==bytes,
          "BIAS sampling changed texture/guards");
    ReleaseFunctionalPayloads(pool,s);pool.Release(handle);
    Check(pool.bytes_in_flight()==0,"BIAS FIFO ownership leaked");
  }
};
}

int sc_main(int argc,char **argv) {
  try {
    Check(argc<=2,"usage: texture-bias-test [MESA_PRODUCER_DIRECTORY]");
    std::vector<Program> programs;for(unsigned k=0;k<6;++k)programs.push_back(Load(argc==2?argv[1]:"",k));
    const float inf=std::numeric_limits<float>::infinity(), nan=std::numeric_limits<float>::quiet_NaN();
    RejectUnsupportedMetadata(programs[0]);
    RawPrepared(programs[5],5,0);
    for(unsigned kind:{0U,1U,2U,4U}) for(float bias:{-2.0F,-0.25F,0.0F,0.25F,1.0F,2.0F,3.0F})
      RawPrepared(programs[kind],kind,bias);
    for(unsigned kind:{0U,1U,2U}) for(float bias:{nan,inf,-inf}) RawPrepared(programs[kind],kind,bias);
    Harness direct("bias_direct",MemoryMode::kDirect),bypass("bias_bypass",MemoryMode::kBypass),cache("bias_cache",MemoryMode::kCache);
    for(auto *h:{&direct,&bypass,&cache})
      for(unsigned kind:{0U,1U,2U,4U})
        for(bool linear:{false,true})
          for(float raw:{-1.0F,0.0F,1.0F})
            for(float bias:{-2.0F,-0.25F,0.0F,0.25F,1.0F,2.0F,3.0F})
              h->Run(programs[kind],kind,raw,bias,linear,0,3);
    for(unsigned kind:{0U,1U,2U})
      for(float bias:{nan,inf,-inf}) direct.Run(programs[kind],kind,-1,bias,true,1,2);
    for(unsigned kind:{0U,1U,2U,4U}) for(bool linear:{false,true})
      for(float bias:{-2.0F,0.25F,2.0F,3.0F}) {
        direct.Run(programs[kind],kind,-1,bias,linear,1,2);
        direct.Run(programs[kind],kind,1,bias,linear,2,2);
      }
    for(auto *h:{&direct,&bypass,&cache}) for(bool linear:{false,true})
      h->Run(programs[3],3,-1,0,linear,0,3);
    for(auto *h:{&direct,&bypass,&cache}) for(unsigned kind:{0U,1U,2U,3U,4U})
      for(bool miplinear:{false,true}) for(float bias:{-0.25F,0.25F,2.0F})
        h->Run(programs[kind],kind,-1,bias,miplinear,0,3,true);
    // A genuine AUTO/no-bias 3D program, not BIAS with a zero payload. s/t
    // remain constant; only r varies across the quad, so losing dr/dx would
    // select the wrong mip. Helpers still execute to provide that derivative.
    for(auto *h:{&direct,&bypass,&cache}) for(bool image_linear:{false,true})
      for(bool miplinear:{false,true}) for(float raw:{-1.0F,0.0F,1.0F,2.0F}) {
        h->Run(programs[5],5,raw,0,miplinear,0,3,image_linear);
        h->Run(programs[5],5,raw,0,miplinear,1,2,image_linear);
      }
    std::cout<<"PASS true Mesa BIAS batches="<<batches<<" checks="<<checks
      <<" sizeof_request="<<sizeof(TextureSampleRequest)
      <<" sizeof_response="<<sizeof(TextureSampleResponse)
      <<" sizeof_continuation="<<sizeof(PcoFragmentContinuation)
      <<" sizeof_instruction="<<sizeof(PcoInstruction)<<'\n';return 0;
  } catch(const std::exception&e) {std::cerr<<e.what()<<'\n';return 1;}
}
