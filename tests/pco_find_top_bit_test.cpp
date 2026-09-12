// SPDX-License-Identifier: MIT
#include "shader/pco_iss.h"
#include "pco_find_top_bit_fixtures.h"
#include "pco_geometry_fixtures.h"
#include "pco_tessellation_patch_fixtures.h"
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
uint32_t CountOracle(uint32_t x){uint32_t out=0;for(unsigned b=0;b<32;++b)out+=(x>>b)&1U;return out;}
uint32_t ReverseOracle(uint32_t x){uint32_t out=0;for(unsigned b=0;b<32;++b)out|=((x>>b)&1U)<<(31U-b);return out;}
struct UnaryOperation { uint8_t phase; PcoOpcode opcode; };
constexpr std::array<UnaryOperation,3> kUnaryOperations{{
  {0x00U,PcoOpcode::kCountBitsSet},
  {0x20U,PcoOpcode::kFindTopBit},
  {0x4aU,PcoOpcode::kReverseBits},
}};
// Exact scalar W1 phase-0 group shape captured from the real compiler output.
// CBS, FTB and REV differ only in the observed phase byte at offset three.
std::vector<uint8_t> UnaryGroup(uint8_t phase){return {
  0x47,0x94,0x40,phase,0x80,0x40,0x00,0x84,0x00,0x40,
  0xf2,0xff,0xff,0xff};}
std::vector<uint8_t> Prefix(std::vector<uint8_t> head,const std::vector<uint8_t>&tail){
  head.insert(head.end(),tail.begin(),tail.end());return head;
}
void CheckNativeUnary(const PcoDecodedProgram&p,PcoOpcode opcode){
  Check(!p.instructions.empty(),"unary program retained its compiler group");
  const auto&i=p.instructions.front();
  Check(i.opcode==opcode&&i.source_count==1&&i.repeat_count==1&&
        i.source.bank==PcoRegisterBank::kVertexInput&&i.source.index==4&&
        i.target==PcoWriteTarget::kTemporary&&!i.output_index,
        "native unary p0 group semantic shape");
  const auto counts=CountPcoInstructions({i},true);
  Check(counts.alu==1&&!counts.texture&&!counts.memory,
        "native unary p0 group is one ALU issue");
}
void TestUnaryFamilyStages(){
  const std::vector<uint8_t> nop_end{
      0x04,0x80,0xee,0x00,0xf2,0xff,0xff,0xff};
  for(const auto operation:kUnaryOperations){
    // Fragment uses the independently pinned real FTB program; replacing its
    // one phase byte exercises the two other compiler-confirmed p0 encodings.
    auto fragment_bytes=test::FindTopBitFixture(0);
    fragment_bytes[13]=operation.phase;
    const auto fragment=DecodePcoProgram(ShaderStage::kFragment,fragment_bytes);
    Check(fragment.instructions[1].opcode==operation.opcode&&
          fragment.instructions[1].source_count==1&&
          fragment.instructions[1].target==PcoWriteTarget::kTemporary,
          "fragment unary p0 phase decode");

    CheckNativeUnary(DecodePcoProgram(
        ShaderStage::kVertex,
        Prefix(UnaryGroup(operation.phase),FillSolidVertexPcoBinary())),
        operation.opcode);
    CheckNativeUnary(DecodeComputePcoProgram(
        Prefix(UnaryGroup(operation.phase),nop_end)),operation.opcode);
    CheckNativeUnary(DecodeGeometryPcoProgram(
        Prefix(UnaryGroup(operation.phase),GeometryNativeLoadFixture())),
        operation.opcode);
    CheckNativeUnary(DecodeTessellationPcoProgram(
        ShaderStage::kTessellationControl,
        Prefix(UnaryGroup(operation.phase),kTessPatchFiveToTenTcs)),
        operation.opcode);
    CheckNativeUnary(DecodeTessellationPcoProgram(
        ShaderStage::kTessellationEvaluation,
        Prefix(UnaryGroup(operation.phase),kTessPatchFiveToTenTes)),
        operation.opcode);
  }
  for(uint32_t x:Inputs())for(const auto operation:kUnaryOperations){
    PcoInstruction i;i.opcode=operation.opcode;
    const auto expected=operation.opcode==PcoOpcode::kCountBitsSet?CountOracle(x):
        operation.opcode==PcoOpcode::kFindTopBit?Oracle(x):ReverseOracle(x);
    Check(EvaluatePcoAluInstruction(i,{x,0,0,0})==expected,
          "unary p0 evaluator matches independent bit oracle");
  }
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
void TestSignedFindMsbXnorPrecursor(){
  // Exact first group emitted for signed findMSB.  THREE_LO is four bytes:
  // s0/s1=sc0 and the visible s2 is VTXIN4; s3 is the following sc0 byte.
  const std::vector<uint8_t> group{
      0x56,0xb2,0x40,0x46,0x02,0x80,0x40,0x00,0x84,0x00,0x40,0xff};
  const auto p=DecodePcoProgram(
      ShaderStage::kVertex,Prefix(group,FillSolidVertexPcoBinary()));
  const auto&i=p.instructions.front();
  Check(i.opcode==PcoOpcode::kBitwiseXnor&&i.source_count==2&&
        i.source.bank==PcoRegisterBank::kVertexInput&&i.source.index==4&&
        i.source1.bank==PcoRegisterBank::kSpecial&&!i.source1.index&&
        i.target==PcoWriteTarget::kTemporary&&!i.output_index,
        "signed findMSB XNOR precursor consumes full THREE_LO source block");
}
void TestRealBitCountVertexProgram(){
  // Exact linked VS binary from the dEQP int_highp_fragment bitCount case.
  // Reverse-linking hoists the builtin to this stage; its group at byte 58
  // is the public CBS s2 encoding with phase byte 0x00.
  const std::vector<uint8_t> bytes{
      0x35,0x82,0x00,0x87,0x80,0x04,0x00,0x00,0x00,0x40,
      0x35,0x82,0x00,0x87,0x81,0x04,0x00,0x00,0x00,0x41,
      0x34,0x82,0x00,0x87,0x00,0x00,0x00,0x42,
      0x35,0x82,0x00,0x87,0x80,0x01,0x00,0x00,0x00,0x43,
      0x55,0xa0,0x06,0x08,0x00,0xc0,0x00,0x00,0x00,0x30,
      0x55,0xa0,0x00,0x08,0x04,0x80,0x01,0x00,0x00,0x30,
      0x47,0x94,0x40,0x00,0x80,0x40,0x00,0x84,0x00,0x40,
      0xf2,0xff,0xff,0xff,
      0x58,0xa0,0x80,0x0e,0x05,0xc0,0x00,0x00,0x00,0x30,
      0xf3,0xff,0xff,0xff,0xff,0xff};
  Check(bytes.size()==88,"real bitCount vertex binary size changed");
  const auto p=DecodePcoProgram(ShaderStage::kVertex,bytes);
  const auto cbs=std::find_if(p.instructions.begin(),p.instructions.end(),
      [](const auto&i){return i.opcode==PcoOpcode::kCountBitsSet;});
  Check(cbs!=p.instructions.end()&&cbs->binary_offset==61&&
        cbs->source.bank==PcoRegisterBank::kVertexInput&&cbs->source.index==4&&
        cbs->target==PcoWriteTarget::kTemporary&&!cbs->output_index,
        "real bitCount vertex program decodes exact CBS group");

  // Its paired FS contains no CBS: reverse-linking exported the result as
  // varying coefficient 2, which this final group writes to pixel output 0.
  const std::vector<uint8_t> fragment{
      0x38,0x8a,0x80,0x87,0xc2,0x04,0x00,0x00,
      0x00,0x20,0xf3,0xff,0xff,0xff,0xff,0xff};
  const auto fp=DecodePcoProgram(ShaderStage::kFragment,fragment);
  Check(fp.instructions.size()==1&&
        fp.instructions.front().opcode==PcoOpcode::kMoveBypass&&
        fp.instructions.front().source.bank==PcoRegisterBank::kCoefficient&&
        fp.instructions.front().source.index==2&&
        fp.instructions.front().target==PcoWriteTarget::kPixelOutput&&
        !fp.instructions.front().output_index,
        "real bitCount fragment program consumes the reverse-linked CBS result");
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
  const std::array<std::pair<size_t,uint8_t>,17> changes{{
    {10,0x35},{11,0x92},{11,0x96},{11,0x9c},{12,0x42},{12,0x44},{12,0x46},
    {13,0x06},{13,0x60},{13,0x21},{13,0x22},{13,0x24},{14,0x81},{15,0x41},
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
int main(){try{TestUnaryFamilyStages();TestGenuine();TestSignedFindMsbXnorPrecursor();TestRealBitCountVertexProgram();TestContinuation();TestGuards();std::cout<<"unary bitwise p0: "<<checks<<" checks PASS\n";return 0;}catch(const std::exception&e){std::cerr<<"unary bitwise p0 after "<<checks<<" checks: "<<e.what()<<'\n';return 1;}}
