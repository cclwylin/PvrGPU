// SPDX-License-Identifier: MIT
// Genuine native CubeArray compiler bytes; synthetic ISS peripheral responses
// test payload/control flow, not the TPU's independent filtering/memory tests.
#include "shader/pco_iss.h"
#include "pco_cube_array_fixtures.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
using namespace pvrgpu::stub;
namespace {
unsigned checks=0;
void Check(bool x,const char *why){++checks;if(!x)throw std::runtime_error(why);}
template<class F>void Reject(F f,const char *why){bool bad=false;try{f();}catch(const std::exception&){bad=true;}Check(bad,why);}
uint32_t Bits(float f){uint32_t x;std::memcpy(&x,&f,4);return x;}
constexpr uint64_t base=UINT64_C(0x1ffffff00),face_stride=8*8*4;
PcoFragmentExecutionContext Context(float layer,float lod){
  PcoFragmentExecutionContext c;c.shared_count=28;c.coefficient_count=4;
  const uint64_t w0=1ULL|(3ULL<<5)|(2ULL<<8)|(1ULL<<11)|(12ULL<<27)|(7ULL<<34)|(7ULL<<48);
  const uint64_t w1=((base>>2)<<16)|2ULL|(2ULL<<4)|(1ULL<<15);
  c.shared_registers[0]=uint32_t(w0);c.shared_registers[1]=w0>>32;
  c.shared_registers[2]=uint32_t(w1);c.shared_registers[3]=w1>>32;
  c.shared_registers[4]=face_stride;
  for(unsigned i=8;i<12;++i)c.shared_registers[i]=0x440000+i;
  c.shared_registers[20]=Bits(.125F);c.shared_registers[21]=Bits(-.25F);
  c.shared_registers[22]=Bits(1.F);c.shared_registers[23]=Bits(layer);
  c.shared_registers[24]=Bits(lod);return c;
}
int Layer(float f){double lo=std::floor(double(f)),r=double(f)-lo;int i=int(lo)+(r>.5||(r==.5&&(int(lo)&1)));return std::max(0,std::min(2,i));}
void Run(bool explicit_lod){
  auto bytes=explicit_lod?test::CubeArray601Fixture():test::CubeArray600Fixture();
  const auto p=DecodePcoProgram(ShaderStage::kFragment,bytes);
  Check(bytes.size()==536&&p.instructions.size()==50&&p.summary.pixel_output_mask==15,"complete genuine CubeArray program");
  const auto it=std::find_if(p.instructions.begin(),p.instructions.end(),[](const auto&i){return i.opcode==PcoOpcode::kTextureSample;});
  Check(it!=p.instructions.end(),"genuine native SMP exists");const size_t index=it-p.instructions.begin();
  const auto &s=*it;
  Check(s.texture_dimension==3&&s.texture_address_offset==1&&s.texture_lod_replace==explicit_lod&&s.texture_gather==0&&s.source2.index==8,"genuine dimension3 TAO and sampler payload");
  const PcoPreparedFragmentProgram prepared(p.summary,p.instructions);
  const std::array<uint32_t,4> response{Bits(.125F),Bits(.25F),Bits(.5F),Bits(1.F)};
  for(float layer:{-100.F,-.5F,0.F,.49F,.5F,1.5F,2.5F,2.F,100.F})
    for(float lod:{-1.F,0.F,.5F,1.F,3.F})for(bool fast:{false,true}){
      auto c=Context(layer,lod);
      auto first=fast?ExecuteFragmentPco(prepared,c):ExecuteFragmentPco(p.summary,p.instructions,c);
      const auto &q=first.texture_request;
      Check(first.suspended&&first.texture_request_valid&&q.dimension==3&&q.coordinate_count==2&&q.component_count==4,"one real six-word payload checkpoint");
      Check(q.coordinates==std::array<uint32_t,3>{Bits(.125F),Bits(-.25F),Bits(1.F)},"xyz preserved independently of layer/LOD/address");
      Check((uint64_t(q.texture_address_hi)<<32|q.texture_address_lo)==base+uint64_t(Layer(layer))*6*face_stride,"RTNE clamped cube times6 face stride, 64-bit carry");
      Check(q.explicit_lod_present==explicit_lod&&q.explicit_lod==(explicit_lod?Bits(lod):0)&&!q.lod_bias_present&&!q.lod_bias&&!q.gather,"explicit LOD or genuine zero implicit TAO bias");
      for(unsigned i=0;i<4;++i)Check(q.texture_state[i]==c.shared_registers[i]&&q.sampler_state[i]==c.shared_registers[i+8],"actual descriptor words preserved");
      c.continuation=first.continuation;c.texture_response_valid=1;c.texture_response=response;
      auto done=fast?ExecuteFragmentPco(prepared,c):ExecuteFragmentPco(p.summary,p.instructions,c);
      Check(!done.suspended&&done.written_mask==15&&std::equal(response.begin(),response.end(),done.pixel_outputs.begin()),"WDF publishes four actual response channels");
      Check(done.executed_instructions.texture==1&&done.native_steps==p.instructions.size(),"one sample across WDF resume, all groups execute");
    }
  for(unsigned mutation=0;mutation<5;++mutation){auto b=p;auto &i=b.instructions[index];
    if(mutation==0)i.texture_dimension=4;
    if(mutation==1)i.texture_address_offset=2;
    if(mutation==2)i.source.index=252; // six-word source must not wrap TEMP.
    if(mutation==3)i.texture_gather=1;
    if(mutation==4){i.source1.index=240;i.source2.index=248;} // descriptor13.
    Reject([&]{PcoPreparedFragmentProgram invalid(b.summary,b.instructions);},"malformed CubeArray semantic contract refused");
  }
  auto c=Context(1,0);auto first=ExecuteFragmentPco(prepared,c);
  c.continuation=first.continuation;c.texture_response_valid=1;c.texture_response=response;
  c.continuation.pending_output_index++;
  Reject([&]{ExecuteFragmentPco(prepared,c);},"forged pending result refused");
  c.continuation=first.continuation;c.continuation.program_signature^=1;
  Reject([&]{ExecuteFragmentPco(prepared,c);},"forged program identity refused");
}
void TwelveTextures(){
  const auto bytes=test::TwelveTextureFixture();const auto p=DecodePcoProgram(ShaderStage::kFragment,bytes);
  Check(bytes.size()==2440&&CountPcoInstructions(p.instructions,false).texture==12,"genuine twelve-SMP full SH256 program");
  const PcoPreparedFragmentProgram prepared(p.summary,p.instructions);
  for(bool fast:{false,true}) {
    PcoFragmentExecutionContext c;c.shared_count=256;c.coefficient_count=4;
    for(unsigned slot=0;slot<12;++slot){auto image=Context(1,0);
      std::copy_n(image.shared_registers.begin(),20,c.shared_registers.begin()+slot*20);
    }
    c.shared_registers[240]=0x1000;c.shared_registers[241]=1;c.shared_registers[242]=32;
    c.shared_registers[244]=Bits(.25F);c.shared_registers[245]=Bits(.75F);
    c.shared_registers[246]=Bits(.5F);c.shared_registers[247]=Bits(1.F);
    c.shared_registers[252]=Bits(0);c.shared_registers[253]=Bits(1);c.shared_registers[254]=Bits(2);
    unsigned reads=0;
    c.memory_user_data=&reads;
    c.memory_read=+[](void *u,uint64_t address,uint32_t count,uint32_t *out){
      Check(address==UINT64_C(0x100001010)&&count==4,"actual UBO read exact base+16 and four DWORDs");
      ++*static_cast<unsigned*>(u);for(unsigned i=0;i<4;++i)out[i]=Bits(float(i+1));
    };
    std::array<float,4> expected{1,2,3,4};unsigned samples=0;
    auto result=fast?ExecuteFragmentPco(prepared,c):ExecuteFragmentPco(p.summary,p.instructions,c);
    while(result.suspended){
      Check(result.texture_request_valid&&samples<12,"exactly twelve native requests");
      const auto &q=result.texture_request;const unsigned slot=samples++;
      Check(q.descriptor_set==slot&&q.binding==0,"all twelve distinct descriptor slots in original order");
      Check(q.dimension==(slot==6?3:2)&&q.explicit_lod_present==(slot>=6&&slot<=8),"genuine CubeArray/2DArray/plain2D modes retained");
      if(slot>=6&&slot<=8){
        Check(q.explicit_lod==Bits(float(slot-6)),"slot-specific actual LOD source");
        const auto address=(uint64_t(q.texture_address_hi)<<32)|q.texture_address_lo;
        Check(address==base+(slot==6?6*face_stride:0),"cube layer1, 2DArray RTNE(.5)=0 address");
      }
      c.continuation=result.continuation;c.texture_response_valid=1;
      for(unsigned i=0;i<4;++i){const float response=float(slot+i+1);c.texture_response[i]=Bits(response);expected[i]+=response*float(slot+1);}
      result=fast?ExecuteFragmentPco(prepared,c):ExecuteFragmentPco(p.summary,p.instructions,c);
    }
    Check(samples==12&&reads==1&&result.executed_instructions.texture==12&&result.written_mask==15,"once-only UBO and every SMP/WDF completion");
    for(unsigned i=0;i<4;++i)Check(result.pixel_outputs[i]==Bits(expected[i]),"exact independent integer-float weighted sum from all descriptors");
  }
}
void SizeQueries(){
  const auto p=DecodePcoProgram(ShaderStage::kFragment,test::CubeArraySizeFixture());
  Check(p.summary.binary_size==416&&CountPcoInstructions(p.instructions,false).texture==0,"genuine CubeArray query has no sample");
  const PcoPreparedFragmentProgram prepared(p.summary,p.instructions);
  for(unsigned size:{1U,8U,32U})for(unsigned cubes:{1U,3U,17U})
    for(unsigned lod=0;lod<4;++lod)for(bool fast:{false,true}){
      auto c=Context(0,0);
      const uint64_t w0=1ULL|(3ULL<<5)|(2ULL<<8)|(1ULL<<11)|(12ULL<<27)|
          (uint64_t(size-1)<<34)|(uint64_t(size-1)<<48);
      const uint64_t w1=((base>>2)<<16)|4ULL|(uint64_t(cubes-1)<<4)|(1ULL<<15);
      c.shared_registers[0]=w0;c.shared_registers[1]=w0>>32;
      c.shared_registers[2]=w1;c.shared_registers[3]=w1>>32;c.shared_registers[4]=size*size*4;
      c.shared_registers[24]=lod;
      const auto done=fast?ExecuteFragmentPco(prepared,c):ExecuteFragmentPco(p.summary,p.instructions,c);
      const unsigned extent=std::max(1U,size>>lod);
      const std::array<uint32_t,4> expected{Bits(float(extent)),Bits(float(extent)),Bits(float(cubes)),Bits(1)};
      Check(!done.suspended&&done.written_mask==15&&done.executed_instructions.texture==0,"actual query finishes without texture work");
      Check(std::equal(expected.begin(),expected.end(),done.pixel_outputs.begin()),"actual raw descriptor query returns square mip size and cube count, not face count");
    }
}
}
int main(int argc,char **argv){try{
  static_assert(kPcoMaximumTextureDescriptorSets==16&&
                kPcoMaximumVertexSharedCount==384&&
                kPcoMaximumFragmentSharedCount==384);
  if(argc!=1&&argc!=3)return 2;
  if(argc==3)for(unsigned i=0;i<2;++i){std::ifstream f(argv[i+1],std::ios::binary);Check(bool(f),"fixture file exists");std::vector<uint8_t>b((std::istreambuf_iterator<char>(f)),{});Check(b==(i?test::CubeArray601Fixture():test::CubeArray600Fixture()),"pinned genuine bytes exactly match");}
  Run(false);Run(true);TwelveTextures();SizeQueries();std::cout<<"CubeArray ISS: "<<checks<<" checks PASS\n";return 0;
}catch(const std::exception&e){std::cerr<<"CubeArray ISS after "<<checks<<" checks: "<<e.what()<<'\n';return 1;}}
