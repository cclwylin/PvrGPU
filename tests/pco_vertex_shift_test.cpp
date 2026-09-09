// SPDX-License-Identifier: MIT
// Genuine compiler bytes, independent input-derived oracles; explicit negative
// field mutations below are not presented as additional compiler output.
#include "shader/pco_iss.h"
#include "pco_vertex_shift_fixtures.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
namespace {
using namespace pvrgpu::stub;
unsigned checks=0;
void Check(bool value,const char *reason){++checks;if(!value)throw std::runtime_error(reason);}
template<class F> void Reject(F fn,const char *part){bool rejected=false;try{fn();}catch(const std::exception&e){rejected=std::string(e.what()).find(part)!=std::string::npos;if(!rejected)std::cerr<<"unexpected diagnostic: "<<e.what()<<'\n';}Check(rejected,part);}
uint32_t Bits(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}
uint32_t Shift(uint32_t x,uint32_t count,unsigned kind){
  count&=31;if(kind==300)return x<<count;
  uint32_t v=x>>count;
  if(kind==302&&count&&(x&0x80000000U))v|=~uint32_t(0)<<(32-count);
  return v;
}
void TestTemporaryControls(){
  for(unsigned kind:{300U,301U,302U}){
    const auto p=DecodePcoProgram(ShaderStage::kVertex,test::VertexShiftFixture(kind));
    Check(p.instructions.front().target==PcoWriteTarget::kTemporary,"ordinary shift target remains TEMP");
    Check(p.summary.vertex_input_mask==15&&p.summary.vertex_output_mask==255,"ordinary compiler register ABI");
    Check(p.instructions.front().integer_signed==(kind==302),"native shift signedness");
    const auto shift_count=CountPcoInstructions({p.instructions.front()},true);
    Check(shift_count.alu==1&&shift_count.texture==0&&shift_count.memory==0,"TEMP shift remains one ALU issue");
    for(uint32_t x:{0U,1U,0x7fffffffU,0x80000000U,0x80000001U,0xffffffffU})
      for(uint32_t count:{0U,1U,7U,31U,32U,33U,0xffffffffU}){
        const std::vector<uint32_t> input{x,count,0x10203040U,0x76543210U};
        const auto saved=input;
        const auto r=ExecuteVertexPco(p.summary,p.instructions,input);
        const uint32_t s=Shift(x,count,kind);
        const std::array<uint32_t,8> expected{0,0,0,Bits(1),s,s+17,s^input[2],input[3]};
        Check(std::equal(expected.begin(),expected.end(),r.outputs.begin()),"genuine TEMP shift result/dependent reads");
        Check(input==saved,"ISS must not overwrite caller input");
        Check(r.written_mask==255&&r.emitted&&r.ended_task&&!r.suspended&&r.executed_instruction_count==p.instructions.size(),"ordinary native shift exact execution count");
      }
  }
}
struct Memory {
  static constexpr uint64_t base=0x123450000ULL;
  std::array<uint32_t,128*32> words{};
  std::vector<std::pair<uint64_t,uint32_t>> reads;
  static void Read(void *opaque,uint64_t address,uint32_t count,uint32_t *out){
    auto &m=*static_cast<Memory*>(opaque);
    if(address<base||(address-base)%4||count>m.words.size()||(address-base)/4>m.words.size()-count)
      throw std::runtime_error("fixture UBO read out of range");
    m.reads.emplace_back(address,count);
    std::copy_n(m.words.data()+(address-base)/4,count,out);
  }
};
void TestReconstructedProgram(){
  const auto bytes=test::VertexShiftFixture(303);
  const auto p=DecodePcoProgram(ShaderStage::kVertex,bytes);
  Check(bytes.size()==1544&&p.summary.group_count==140&&p.instructions.size()==140,"entire genuine reconstructed shader decoded");
  const auto it=std::find_if(p.instructions.begin(),p.instructions.end(),[](const auto&i){return i.binary_offset==49;});
  Check(it!=p.instructions.end()&&it->opcode==PcoOpcode::kShiftLeft&&it->target==PcoWriteTarget::kVertexInput&&it->output_index==6&&!it->integer_signed,"actual failing group is canonical VS LSL to VTXIN6");
  const auto shift_count=CountPcoInstructions({*it},true);
  Check(shift_count.alu==1&&shift_count.texture==0&&shift_count.memory==0,"VTXIN reuse shift remains one ALU issue");
  Check(p.summary.vertex_input_mask==1257751&&!(p.summary.vertex_input_mask&(1ULL<<6)),"reused VTXIN6 is not an original input dependency");
  Memory m;
  for(unsigned instance=0;instance<128;++instance){
    for(unsigned i=0;i<4;++i)m.words[32*instance+5*i]=Bits(1);
    m.words[32*instance+12]=Bits(float(instance));
    m.words[32*instance+13]=Bits(float(instance*2));
    m.words[32*instance+14]=Bits(float(instance*3));
    for(unsigned i=0;i<3;++i)m.words[32*instance+16+5*i]=Bits(1);
  }
  PcoVertexExecutionContext c;c.shared_count=44;c.memory_read=Memory::Read;c.memory_user_data=&m;
  c.shared_registers[0]=uint32_t(Memory::base);c.shared_registers[1]=uint32_t(Memory::base>>32);c.shared_registers[2]=sizeof(m.words);
  for(unsigned matrix=0;matrix<2;++matrix)for(unsigned i=0;i<4;++i)c.shared_registers[12+matrix*16+5*i]=Bits(1);
  for(unsigned id:{0U,1U,2U,17U,63U})for(unsigned offset:{0U,1U,7U}){
    const unsigned selected=id+offset;
    std::vector<uint32_t> input(24,0);input[0]=Bits(2);input[1]=Bits(3);input[2]=Bits(4);
    input[4]=0x7f00007fU;input[8]=0x7f007f00U;
    input[12]=Bits(.125F);input[13]=Bits(.25F);input[16]=Bits(.5F);input[17]=Bits(.75F);input[20]=id;
    const auto saved=input;c.shared_registers[8]=offset;m.reads.clear();
    const auto r=ExecuteVertexPco(p.summary,p.instructions,input,c);
    const std::array<uint32_t,4> clip{Bits(float(2+selected)),Bits(float(3+2*selected)),Bits(float(4+3*selected)),Bits(1)};
    for(unsigned copy=0;copy<3;++copy)Check(std::equal(clip.begin(),clip.end(),r.outputs.begin()+copy*4),"reconstructed shifted instance address yields independent identity-matrix clip");
    const std::array<uint32_t,10> tail{Bits(.125F),Bits(.25F),Bits(.5F),Bits(.75F),Bits(1),0,0,0,Bits(1),0};
    Check(std::equal(tail.begin(),tail.end(),r.outputs.begin()+12),"packed SNORM plus all dependent varying outputs");
    const uint64_t base=Memory::base+128ULL*selected;
    // Only xyz of the final three normal/tangent rows are live in source.
    const std::vector<std::pair<uint64_t,uint32_t>> expected{{base,16},{base+64,3},{base+80,3},{base+96,3}};
    Check(m.reads==expected,"shift result must feed exact once-only UBO addresses and widths");
    Check(input==saved,"reused VTXIN must not mutate caller input");
    Check(r.written_mask==((1ULL<<22)-1)&&r.emitted&&r.ended_task&&!r.suspended&&r.executed_instruction_count==140,"reconstructed program completes exactly once");
  }
}
void TestDecoderGuards(){
  const auto original=test::VertexShiftFixture(303);
  // The entire genuine group at capture byte46; cut after this complete group
  // so unrelated later stage-specific exports cannot mask the destination gate.
  const std::vector<uint8_t> group(original.begin()+46,original.begin()+58);
  for(auto stage:{ShaderStage::kFragment,ShaderStage::kCompute,ShaderStage::kGeometry,ShaderStage::kTessellationControl,ShaderStage::kTessellationEvaluation}){
    Reject([&]{if(stage==ShaderStage::kCompute)DecodeComputePcoProgram(group);else if(stage==ShaderStage::kGeometry)DecodeGeometryPcoProgram(group);else if(stage==ShaderStage::kTessellationControl||stage==ShaderStage::kTessellationEvaluation)DecodeTessellationPcoProgram(stage,group);else DecodePcoProgram(stage,group);},"shift destination must be temporary");
  }
  for(unsigned bank:{0U,3U,4U,5U,6U,7U}){
    auto b=original;const unsigned index=bank==0?32:6;
    b[55]=0x80|((bank&1)<<6)|(index&63);b[56]=(bank>>1)<<2;
    Reject([&]{DecodePcoProgram(ShaderStage::kVertex,b);},bank==0?"shift destination must be temporary":"ALU destination is neither TEMP nor PIXOUT");
  }
  for(unsigned index:{64U,255U,256U,1023U,2047U}){
    auto b=original;b[55]=0x80|(index&63);b[56]=4|((index>>6)&3)|((index>>8)<<4);
    Reject([&]{DecodePcoProgram(ShaderStage::kVertex,b);},"vertex-input destination exceeds");
  }
  {auto b=original;b[56]|=0x80;Reject([&]{DecodePcoProgram(ShaderStage::kVertex,b);},"reserved bit");}
  {auto b=original;b[57]=0;Reject([&]{DecodePcoProgram(ShaderStage::kVertex,b);},"padding");}
  {auto b=original;b[55]=0xbf;const auto p=DecodePcoProgram(ShaderStage::kVertex,b);
    PcoVertexExecutionContext c;c.shared_count=44;
    Reject([&]{ExecuteVertexPco(p.summary,p.instructions,std::vector<uint32_t>(24),c);},"generic vertex ALU destination is out of range");}
  for(unsigned op:{2U,3U,5U,6U,7U}){auto b=original;b[49]=op;Reject([&]{DecodePcoProgram(ShaderStage::kVertex,b);},"phase-2");}
}
}
int main(){try{TestTemporaryControls();TestReconstructedProgram();TestDecoderGuards();std::cout<<"vertex shift: "<<checks<<" checks PASS\n";return 0;}catch(const std::exception&e){std::cerr<<"vertex shift after "<<checks<<" checks: "<<e.what()<<'\n';return 1;}}
