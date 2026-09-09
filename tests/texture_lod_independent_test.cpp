// SPDX-License-Identifier: MIT
// Actual FIFO/cache/DRAM sampling with incomplete derivative quads. The sampler
// proves base-level selection; no missing coordinates or texture values exist.
#include "common/pipeline_state.h"
#include "memory/gpu_memory_system.h"
#include "texture/texture_unit.h"
#include <systemc>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace pvrgpu::stub;
unsigned checks=0,batches=0;
void Check(bool b,const char*w){++checks;if(!b)throw std::runtime_error(w);}
uint32_t Bits(float f){uint32_t b;std::memcpy(&b,&f,4);return b;}
uint8_t Texel(unsigned level,unsigned x,unsigned y,unsigned c){return uint8_t((level==0?20:level==1?100:200)+x*16+y*32+c*4);}
struct Harness {
  MemoryPool pool;GpuMemorySystem memory;
  sc_core::sc_fifo<PipelineTxn> input,output,sample_input,sample_output;
  TextureUnit unit;
  Harness(const char*name,MemoryMode mode):memory(mode),input(sc_core::sc_gen_unique_name("input"),1),output(sc_core::sc_gen_unique_name("output"),1),sample_input(sc_core::sc_gen_unique_name("samples"),1),sample_output(sc_core::sc_gen_unique_name("responses"),1),unit(name,pool,&memory){unit.input(input);unit.output(output);unit.sample_input(sample_input);unit.sample_output(sample_output);}
  void Run(unsigned min,unsigned max,bool linear,unsigned partial,unsigned mutation=0,bool offcenter=false){
    TextureResource resource;resource.gpu_address=UINT64_C(0x30000000)+(++batches)*0x10000;
    resource.format=TextureFormat::kRgba8Unorm;resource.dimension_type=TextureDimensionType::k2D;
    resource.mip_count=3;resource.mip[0]={4,4,16,0};resource.mip[1]={2,2,8,64};resource.mip[2]={1,1,4,80};resource.byte_size=84;
    std::vector<uint8_t>bytes(84+64,0xa7);
    for(unsigned level=0;level<3;++level)for(unsigned y=0;y<resource.mip[level].height;++y)for(unsigned x=0;x<resource.mip[level].width;++x)for(unsigned c=0;c<4;++c)bytes[resource.mip[level].offset_bytes+y*resource.mip[level].row_pitch_bytes+x*4+c]=Texel(level,x,y,c);
    memory.HostWrite(resource.gpu_address,bytes.data(),bytes.size());
    SamplerState sampler;sampler.min_filter=sampler.mag_filter=linear?TextureFilter::kLinear:TextureFilter::kNearest;
    sampler.min_lod_u4_6=min;sampler.max_lod_u4_6=max;
    if(mutation==1)sampler.mag_filter=linear?TextureFilter::kNearest:TextureFilter::kLinear;
    if(mutation==2)sampler.mip_filter=TextureFilter::kLinear;
    const uint64_t image0=4ULL|(3ULL<<5)|(2ULL<<8)|(1ULL<<11)|(12ULL<<27)|(3ULL<<34)|(3ULL<<48);
    const uint64_t image1=((resource.gpu_address>>2)<<16)|(3ULL<<60)|(1ULL<<15)|3ULL;
    const uint64_t sampler0=0xfffULL|(uint64_t(min)<<13)|(uint64_t(max)<<23)|
        (uint64_t(sampler.mag_filter==TextureFilter::kLinear)<<36)|
        (uint64_t(sampler.min_filter==TextureFilter::kLinear)<<38)|
        (uint64_t(sampler.mip_filter==TextureFilter::kLinear)<<40);
    const uint64_t gather=sampler0|(1ULL<<36)|(1ULL<<38);
    std::vector<uint32_t>shared(20);shared[0]=uint32_t(image0);shared[1]=uint32_t(image0>>32);shared[2]=uint32_t(image1);shared[3]=uint32_t(image1>>32);
    shared[4]=resource.byte_size;shared[8]=sampler0;shared[9]=sampler0>>32;shared[16]=gather;shared[17]=gather>>32;
    // Four real requests can come from two distinct partial quads. A separate
    // two-request variant also proves the count need not be divisible by four.
    const unsigned count=partial==2?2:4;
    std::vector<TextureSampleRequest>requests(count);
    std::vector<FragmentShaderLane>lanes(8);
    const std::array<unsigned,4>xs{0,1,2,3},ys{1,2,3,0};
    for(unsigned i=0;i<count;++i){auto&q=requests[i];q.request_id=i;
      q.shader_lane_index=partial?i*2:i;q.quad_id=q.shader_lane_index/4;q.quad_lane=q.shader_lane_index%4;
      q.shader_stage=ShaderStage::kFragment;q.coordinate_count=2;q.component_count=4;q.dimension=2;q.normalized=q.fcnorm=1;
      q.coordinates[0]=Bits((xs[i]+(offcenter?1.F:.5F))/4);q.coordinates[1]=Bits((ys[i]+(offcenter?1.F:.5F))/4);
      std::copy_n(shared.begin(),4,q.texture_state);std::copy_n(shared.begin()+8,4,q.sampler_state);
      auto&l=lanes[q.shader_lane_index];l.quad_id=q.quad_id;l.quad_lane=q.quad_lane;l.x=xs[i];l.y=ys[i];l.helper=i%2;
    }
    PipelineState state;state.sequence=batches;state.memory_mode=memory.mode();state.functional_case=FunctionalCase::kDriverPcoTriangles;
    state.stage=PipelineStage::kFragmentTexturePending;state.sampled_texture_count=1;state.fragment_pco_abi.shareds=20;state.fragment_shader_lane_count=8;
    state.texture_resources=StoreNewArray(pool,std::vector<TextureResource>{resource});state.sampler_states=StoreNewArray(pool,std::vector<SamplerState>{sampler});
    state.fragment_shared_registers=StoreNewArray(pool,shared);state.fragment_shader_lanes=StoreNewArray(pool,lanes);state.texture_sample_requests=StoreNewArray(pool,requests);
    auto handle=pool.Allocate(sizeof(PipelineState));StorePipelineState(pool,handle,state);sample_input.write(PipelineTxn{handle,batches,batches});
    sc_core::sc_start(sc_core::sc_time(10000,sc_core::SC_NS));PipelineTxn done;
    Check(sample_output.nb_read(done)&&done.state.slot==handle.slot&&done.state.generation==handle.generation,"real TPU completion identity");
    const auto final=LoadPipelineState(pool,handle);const auto responses=LoadArray<TextureSampleResponse>(pool,final.texture_sample_responses);
    const auto retained=LoadArray<TextureSampleRequest>(pool,final.texture_sample_requests);
    Check(responses.size()==count&&retained.size()==count,"only real requests produce responses");
    for(unsigned i=0;i<count;++i){Check(responses[i].request_id==i&&responses[i].shader_lane_index==requests[i].shader_lane_index,"response keeps original sparse lane identity");
      Check(std::equal(requests[i].coordinates,requests[i].coordinates+3,retained[i].coordinates)&&
            std::equal(requests[i].sampler_state,requests[i].sampler_state+4,retained[i].sampler_state),"coordinates and original nonzero LOD window retained");
      for(unsigned c=0;c<4;++c){
        const unsigned expected=offcenter?
          (unsigned(Texel(0,xs[i],ys[i],c))+Texel(0,(xs[i]+1)%4,ys[i],c)+Texel(0,xs[i],(ys[i]+1)%4,c)+Texel(0,(xs[i]+1)%4,(ys[i]+1)%4,c))/4:
          Texel(0,xs[i],ys[i],c);
        Check(responses[i].rgba[c]==Bits(float(expected)/255.F),"actual nonconstant base mip sample, never other-level data");
      }
    }
    const unsigned taps=count*(linear?4:1);
    Check(final.counters.texture_requests==count&&final.counters.texel_fetches==taps,"all original nearest/bilinear taps counted once");
    if(memory.mode()==MemoryMode::kDirect)Check(final.counters.memory_direct_read_bytes==taps*4&&final.counters.dram_read_bytes==0,"direct actual texture read bytes");
    else if(memory.mode()==MemoryMode::kBypass)Check(final.counters.dram_read_bytes==taps*4&&final.counters.memory_direct_read_bytes==0,"bypass actual texture read bytes");
    else Check(final.counters.dram_read_bytes>0&&final.counters.slc_read_accesses==taps,"unified cache and backing remain on native path");
    Check(memory.Readback(resource.gpu_address,bytes.size(),MemoryClient::kFramebufferReadback).data==bytes,"mips and guard bytes unchanged");
    ReleaseFunctionalPayloads(pool,final);pool.Release(handle);Check(pool.bytes_in_flight()==0&&pool.allocations()==pool.releases(),"pool resources balanced");
  }
};
}
int sc_main(int argc,char**argv){std::string mode=argc==2?argv[1]:"";try{
  if(argc>2)return 2;Harness direct("lod_direct",MemoryMode::kDirect),bypass("lod_bypass",MemoryMode::kBypass),cache("lod_cache",MemoryMode::kCache);
  if(!mode.empty()){
    unsigned max=mode=="half"?32:mode=="next"?33:mode=="wide"?128:16;
    unsigned mutation=mode=="filters"?1:mode=="miplinear"?2:0;
    if(mode!="half"&&mode!="next"&&mode!="wide"&&mode!="filters"&&mode!="miplinear")return 2;
    direct.Run(0,max,true,1,mutation);throw std::runtime_error("negative unexpectedly completed");
  }
  for(auto*h:{&direct,&bypass,&cache})for(bool linear:{false,true})for(unsigned max:{0U,1U,16U,31U})for(unsigned partial:{0U,1U,2U})h->Run(0,max,linear,partial);
  for(auto*h:{&direct,&bypass,&cache})for(bool linear:{false,true})for(unsigned min:{8U,16U})for(unsigned partial:{1U,2U})h->Run(min,16,linear,partial);
  for(auto*h:{&direct,&bypass,&cache})for(unsigned partial:{0U,1U,2U})h->Run(0,16,true,partial,0,true);
  std::cout<<"LOD-independent TPU: "<<checks<<" checks / "<<batches<<" batches PASS\n";return 0;
}catch(const std::exception&e){const std::string error=e.what();if(!mode.empty()&&error.find("TextureUnit LOD request lost 2x2 quad identity")!=std::string::npos){std::cout<<"LOD-dependent "<<mode<<" refused: "<<error<<" PASS\n";return 0;}std::cerr<<"LOD-independent after "<<checks<<" checks: "<<error<<'\n';return 1;}}
