// SPDX-License-Identifier: MIT
// Execute genuine compiler bytes. The four supplied taps are an ISS test
// peripheral; actual TPU/cache/DRAM traffic is tested separately.
#include "shader/pco_iss.h"
#include "pco_array_shadow_gather_fixture.h"
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
void Check(bool v,const char*w){++checks;if(!v)throw std::runtime_error(w);}
template<class F>void Reject(F f,const char*w){bool bad=false;try{f();}catch(const std::exception&){bad=true;}Check(bad,w);}
uint32_t Bits(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}
constexpr uint64_t base=UINT64_C(0x1ffffff00),stride=3*5*16;
PcoFragmentExecutionContext Context(float layer,float ref,unsigned compare,bool unorm){
  PcoFragmentExecutionContext c;c.shared_count=24;c.coefficient_count=4;
  const uint64_t image0=1ULL|(3ULL<<5)|(2ULL<<8)|(1ULL<<11)|(61ULL<<27)|(2ULL<<34)|(4ULL<<48);
  const uint64_t image1=((base>>2)<<16)|1ULL|(3ULL<<4);
  c.shared_registers[0]=static_cast<uint32_t>(image0);c.shared_registers[1]=image0>>32;
  c.shared_registers[2]=static_cast<uint32_t>(image1);c.shared_registers[3]=image1>>32;
  c.shared_registers[4]=stride;c.shared_registers[7]=unorm?0x100:0;c.shared_registers[12]=compare;
  for(unsigned n=16;n<20;++n)c.shared_registers[n]=0x550000+n;
  c.shared_registers[20]=Bits(.375F);c.shared_registers[21]=Bits(.625F);
  c.shared_registers[22]=Bits(layer);c.shared_registers[23]=Bits(ref);return c;
}
int Layer(float f){double lo=std::floor(double(f)),frac=double(f)-lo;int rounded=int(lo)+(frac>.5||(frac==.5&&(int(lo)&1)));return std::max(0,std::min(3,rounded));}
bool Compare(unsigned op,float ref,float depth){switch(op){case 0:return false;case 1:return ref<depth;case 2:return ref==depth;case 3:return ref<=depth;case 4:return ref>depth;case 5:return ref!=depth;case 6:return ref>=depth;case 7:return true;}throw std::runtime_error("oracle op");}
void Run(){
  const auto bytes=test::ArrayShadowGatherFixture();const auto p=DecodePcoProgram(ShaderStage::kFragment,bytes);
  Check(bytes.size()==1608&&p.instructions.size()==114&&p.summary.pixel_output_mask==15,"genuine complete shadow gather summary");
  const auto it=std::find_if(p.instructions.begin(),p.instructions.end(),[](const auto&i){return i.opcode==PcoOpcode::kTextureSample;});
  Check(it!=p.instructions.end(),"genuine sample present");const size_t index=it-p.instructions.begin();const auto&i=*it;
  Check(i.texture_gather==1&&i.texture_address_offset==1&&i.texture_dimension==2&&i.texture_lod_replace==1&&i.source2.index==16&&i.component_count==4,"genuine array gather semantic contract");
  Check(i.binary_offset==481&&bytes[481]==0xf4&&bytes[482]==0x52&&bytes[483]==0x91,"genuine TAO RAWDATA native encoding");
  const PcoPreparedFragmentProgram prepared(p.summary,p.instructions);
  const std::array<float,4> depths{.125F,.25F,.5F,.75F};const std::array<unsigned,4> order{2,3,1,0};
  std::array<uint32_t,4> taps{};for(unsigned n=0;n<4;++n)taps[n]=Bits(depths[n]);
  for(float layer:{-100.F,0.F,.49F,.5F,1.5F,2.5F,3.F,100.F})for(float ref:{-1.F,0.F,.125F,.5F,.75F,1.F,2.F})for(unsigned op=0;op<8;++op)for(bool unorm:{false,true})for(bool fast:{false,true}){
    auto c=Context(layer,ref,op,unorm);auto first=fast?ExecuteFragmentPco(prepared,c):ExecuteFragmentPco(p.summary,p.instructions,c);
    const auto&q=first.texture_request;
    Check(first.suspended&&first.texture_request_valid&&q.gather==1&&q.explicit_lod_present==1&&q.explicit_lod==0&&q.coordinate_count==2,"one true array-gather checkpoint");
    Check(q.coordinates==std::array<uint32_t,3>{Bits(.375F),Bits(.625F),0}&&q.spatial_offsets==std::array<int32_t,3>{0,0,0},"UV distinct from LOD/address payload");
    Check((uint64_t(q.texture_address_hi)<<32|q.texture_address_lo)==base+uint64_t(Layer(layer))*stride,"native layer RTNE/clamp/64-bit address carry");
    for(unsigned w=0;w<4;++w)Check(q.texture_state[w]==c.shared_registers[w]&&q.sampler_state[w]==c.shared_registers[16+w],"actual source descriptors retained");
    c.continuation=first.continuation;c.texture_response_valid=1;c.texture_response=taps;
    const auto done=fast?ExecuteFragmentPco(prepared,c):ExecuteFragmentPco(p.summary,p.instructions,c);
    float clamped=unorm?std::max(0.F,std::min(1.F,ref)):ref;
    for(unsigned n=0;n<4;++n)Check(done.pixel_outputs[n]==Bits(Compare(op,clamped,depths[order[n]])?1.F:0.F),"actual shader compare/clamp and GL tap order");
    Check(!done.suspended&&done.written_mask==15&&done.executed_instructions.texture==1&&done.native_steps==p.instructions.size(),"one request across original WDF, complete program");
  }
  for(unsigned bit:{2U,4U,8U,0x20U,0x40U,0x80U}){auto bad=bytes;bad[483]^=bit;Reject([&]{DecodePcoProgram(ShaderStage::kFragment,bad);},"unsupported native gather extension");}
  for(unsigned mutation=0;mutation<7;++mutation){auto bad=p;auto&b=bad.instructions[index];switch(mutation){case 0:b.texture_address_offset=2;break;case 1:b.texture_lod_replace=0;break;case 2:b.source.index=253;break;case 3:b.source2.index=8;break;case 4:b.component_count=8;break;case 5:b.texture_dimension=3;break;case 6:b.texture_lod_bias=1;break;}Reject([&]{PcoPreparedFragmentProgram x(bad.summary,bad.instructions);},"malformed array-gather metadata");}
  auto c=Context(1,.5,3,true);auto first=ExecuteFragmentPco(prepared,c);
  c.continuation=first.continuation;c.texture_response_valid=1;c.texture_response=taps;c.continuation.program_signature^=1;
  Reject([&]{ExecuteFragmentPco(prepared,c);},"forged array-gather continuation program identity");
  c.continuation=first.continuation;c.continuation.pending_component_count=3;
  Reject([&]{ExecuteFragmentPco(prepared,c);},"forged raw response width");
  c.continuation=first.continuation;c.continuation.pending_output_index++;
  Reject([&]{ExecuteFragmentPco(prepared,c);},"forged pending result destination");
}
}
int main(int argc,char**argv){try{if(argc>2)return 2;if(argc==2){std::ifstream f(argv[1],std::ios::binary);Check(bool(f),"fixture path");std::vector<uint8_t>b((std::istreambuf_iterator<char>(f)),{});Check(b==test::ArrayShadowGatherFixture(),"fixture bytes equal actual compiler output");}Run();std::cout<<"array shadow gather ISS: "<<checks<<" checks PASS\n";return 0;}catch(const std::exception&e){std::cerr<<"array shadow gather ISS after "<<checks<<" checks: "<<e.what()<<'\n';return 1;}}
