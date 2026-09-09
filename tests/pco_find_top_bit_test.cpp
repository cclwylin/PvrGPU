// SPDX-License-Identifier: MIT
#include "shader/pco_iss.h"
#include "pco_find_top_bit_fixtures.h"
#include "pco_texture_gather_fixtures.h"
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
template<class F> void Reject(F fn,const char *reason){bool rejected=false;try{fn();}catch(const std::exception&){rejected=true;}Check(rejected,reason);}
uint32_t Bits(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}
// Independent oracle scans ascending bit positions; the implementation scans
// by shifting the original word down. Both treat zero as the all-ones sentinel.
uint32_t Oracle(uint32_t x){uint32_t out=UINT32_MAX;for(unsigned b=0;b<32;++b)if(x&(1U<<b))out=b;return out;}
std::vector<uint32_t> Inputs(){
  std::vector<uint32_t> out{0,UINT32_MAX,0x80000000U,0x7fffffffU,0x80000001U};
  for(unsigned b=0;b<32;++b){const uint32_t x=1U<<b;out.push_back(x);out.push_back(x-1);out.push_back(x+1);}
  uint32_t x=0x17a5826dU;
  for(unsigned n=0;n<1024;++n){x^=x<<13;x^=x>>17;x^=x<<5;out.push_back(x);}
  return out;
}
void TestGenuine(){
  const auto values=Inputs();
  for(unsigned kind:{0U,1U}){
    const auto p=DecodePcoProgram(ShaderStage::kFragment,test::FindTopBitFixture(kind));
    const PcoPreparedFragmentProgram prepared(p.summary,p.instructions);
    Check(p.summary.pixel_output_mask==15&&p.summary.instruction_count==(kind?12U:8U),"genuine FTB compiler summary");
    unsigned ftb=0;
    for(const auto&i:p.instructions)if(i.opcode==PcoOpcode::kFindTopBit){
      ++ftb;Check(i.source_count==1&&i.repeat_count==1&&i.target==PcoWriteTarget::kTemporary&&!i.integer_signed,"genuine FTB canonical semantic shape");
      const auto counts=CountPcoInstructions({i},true);
      Check(counts.alu==1&&!counts.texture&&!counts.memory,"FTB is one ALU issue");
    }
    Check(ftb==(kind?4U:1U),"genuine FTB opcode census");
    for(uint32_t x:values){
      PcoFragmentExecutionContext c;c.shared_count=kind?4:1;c.coefficient_count=4;
      c.shared_registers[0]=x;c.shared_registers[1]=~x;c.shared_registers[2]=x^0x80000000U;c.shared_registers[3]=0;
      const uint32_t msb=Oracle(x);
      const std::array<uint32_t,4> expected=kind?std::array<uint32_t,4>{msb,Oracle(~x),Oracle(x^0x80000000U),UINT32_MAX}:
        std::array<uint32_t,4>{msb,msb+1,msb^x,x};
      const auto raw=ExecuteFragmentPco(p.summary,p.instructions,c),fast=ExecuteFragmentPco(prepared,c);
      Check(std::equal(expected.begin(),expected.end(),raw.pixel_outputs.begin())&&raw.pixel_outputs==fast.pixel_outputs,"genuine raw/prepared FTB outputs and dependent reads");
      Check(raw.written_mask==15&&fast.written_mask==15&&!raw.suspended&&!fast.suspended,"genuine FTB completes all outputs");
      Check(raw.executed_instruction_count==p.instructions.size()&&fast.executed_instruction_count==raw.executed_instruction_count&&raw.executed_instructions.alu==p.instructions.size()&&!raw.executed_instructions.memory&&!raw.executed_instructions.texture,"genuine FTB once-only ALU counters");
      PcoInstruction ftb_op;ftb_op.opcode=PcoOpcode::kFindTopBit;
      Check(EvaluatePcoAluInstruction(ftb_op,{x,0,0,0})==msb,"pure FTB semantics exact bits");
    }
  }
}
void Reindex(PcoDecodedProgram &p){
  for(size_t pc=0;pc<p.instructions.size();++pc){p.instructions[pc].binary_offset=pc*8+3;p.instructions[pc].group_index=pc;}
  p.summary.binary_size=p.instructions.size()*8;p.summary.group_count=p.summary.instruction_count=p.instructions.size();
}
void TestContinuation(){
  // Explicit semantic composition for continuation guards, not a claim that
  // this combined byte program was compiler-generated. Individual FTB and SMP
  // operations come from the actual decoded compiler fixtures.
  auto p=DecodePcoProgram(ShaderStage::kFragment,test::TextureGatherFixture(2));
  auto ftb=DecodePcoProgram(ShaderStage::kFragment,test::FindTopBitFixture(0)).instructions[1];
  ftb.output_index=250;ftb.source={PcoRegisterBank::kShared,22};
  p.instructions.insert(p.instructions.begin(),ftb);
  auto pixel=std::find_if(p.instructions.begin(),p.instructions.end(),[](const auto&i){return i.target==PcoWriteTarget::kPixelOutput&&i.output_index==0;});
  Check(pixel!=p.instructions.end(),"fixture has PIXOUT0");pixel->source={PcoRegisterBank::kTemporary,250};
  Reindex(p);const PcoPreparedFragmentProgram prepared(p.summary,p.instructions);
  for(uint32_t x:{0U,1U,0x80000000U,0xffffffffU})for(bool fast:{false,true}){
    PcoFragmentExecutionContext c;c.shared_count=23;c.shared_registers[20]=Bits(.25F);c.shared_registers[21]=Bits(.75F);c.shared_registers[22]=x;
    const auto first=fast?ExecuteFragmentPco(prepared,c):ExecuteFragmentPco(p.summary,p.instructions,c);
    Check(first.suspended&&first.texture_request_valid&&first.executed_instructions.texture==1,"FTB precedes one true SMP checkpoint");
    Check(first.continuation.temporary_written_mask.test(250)&&first.continuation.temporaries[250]==Oracle(x),"checkpoint owns FTB result including zero sentinel");
    c.continuation=first.continuation;c.texture_response_valid=1;c.texture_response={1,2,3,4};
    const auto done=fast?ExecuteFragmentPco(prepared,c):ExecuteFragmentPco(p.summary,p.instructions,c);
    // executed_instruction_count is local to this call; the suspended SMP is
    // committed on resume. Sum the call-local issue counts across the boundary.
    Check(!done.suspended&&done.pixel_outputs[0]==Oracle(x)&&done.executed_instructions.texture==1&&first.executed_instruction_count+done.executed_instruction_count==p.instructions.size(),"WDF resume preserves FTB without reissue");
    c.continuation.temporary_written_mask.words[250/64]&=~(1ULL<<(250%64));
    Reject([&]{if(fast)ExecuteFragmentPco(prepared,c);else ExecuteFragmentPco(p.summary,p.instructions,c);},"missing required FTB result must fail closed");
  }
}
void TestGuards(){
  const auto original=test::FindTopBitFixture(0);
  const std::vector<uint8_t> group(original.begin()+10,original.begin()+20);
  for(auto stage:{ShaderStage::kVertex,ShaderStage::kCompute,ShaderStage::kGeometry,ShaderStage::kTessellationControl,ShaderStage::kTessellationEvaluation})
    Reject([&]{if(stage==ShaderStage::kVertex)DecodePcoProgram(stage,group);else if(stage==ShaderStage::kCompute)DecodeComputePcoProgram(group);else if(stage==ShaderStage::kGeometry)DecodeGeometryPcoProgram(group);else DecodeTessellationPcoProgram(stage,group);},"unproven stage FTB admission");
  const std::array<std::pair<size_t,uint8_t>,17> changes{{
    {10,0x35},{11,0x92},{11,0x96},{11,0x9c},{12,0x42},{12,0x44},{12,0x46},
    {13,0x00},{13,0x60},{13,0x21},{13,0x22},{13,0x24},{14,0x81},{15,0x41},
    {16,0x20},{18,1},{19,0x20}}};
  for(auto change:changes){auto b=original;b[change.first]=change.second;Reject([&]{DecodePcoProgram(ShaderStage::kFragment,b);},"noncanonical native FTB encoding");}
  for(size_t n=10;n<20;++n){auto b=original;b.resize(n);Reject([&]{DecodePcoProgram(ShaderStage::kFragment,b);},"truncated FTB binary");}
  const auto valid=DecodePcoProgram(ShaderStage::kFragment,original);
  for(unsigned mutation=0;mutation<14;++mutation){
    auto p=valid;auto &i=p.instructions[1];
    switch(mutation){
      case 0:i.output_index=256;break;case 1:i.target=PcoWriteTarget::kPixelOutput;break;
      case 2:i.target=PcoWriteTarget::kVertexInput;break;case 3:i.source.index=256;break;
      case 4:i.source.bank=PcoRegisterBank::kVertexInput;break;case 5:i.source_count=0;break;
      case 6:i.source_count=2;break;case 7:i.repeat_count=2;break;case 8:i.integer_signed=1;break;
      case 9:i.immediate=1;break;case 10:i.source0_floor=1;break;case 11:i.source0_absolute=1;break;
      case 12:i.source0_negate=1;break;case 13:i.saturate=1;break;
    }
    PcoFragmentExecutionContext c;c.shared_count=1;c.shared_registers[0]=1;
    Reject([&]{ExecuteFragmentPco(p.summary,p.instructions,c);},"raw malformed FTB metadata");
    Reject([&]{PcoPreparedFragmentProgram prepared(p.summary,p.instructions);ExecuteFragmentPco(prepared,c);},"prepared malformed FTB metadata");
  }
}
}
int main(){try{TestGenuine();TestContinuation();TestGuards();std::cout<<"find top bit: "<<checks<<" checks PASS\n";return 0;}catch(const std::exception&e){std::cerr<<"find top bit after "<<checks<<" checks: "<<e.what()<<'\n';return 1;}}
