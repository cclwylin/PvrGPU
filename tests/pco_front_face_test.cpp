// SPDX-License-Identifier: MIT
// Raster-facing ISS/context guards. Synthetic semantic mutations below are
// explicit tests, not claimed compiler output or captured shader execution.
#include "shader/pco_iss.h"
#include "pco_texture_gather_fixtures.h"
#include "pco_front_face_fixtures.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace pvrgpu::stub;
unsigned checks=0;
void Check(bool value,const char *message){++checks;if(!value)throw std::runtime_error(message);}
template<class F> void Reject(F fn,const char *message){bool caught=false;try{fn();}catch(const std::exception&){caught=true;}Check(caught,message);}
uint32_t Bits(float v){uint32_t u;std::memcpy(&u,&v,4);return u;}
PcoDecodedProgram Semantic(std::vector<PcoInstruction> instructions,uint16_t mask){
  for(size_t pc=0;pc<instructions.size();++pc){instructions[pc].binary_offset=1+pc*8;instructions[pc].group_index=pc;}
  instructions.back().end_group=1;
  PcoDecodedProgram p;p.instructions=std::move(instructions);p.summary.stage=ShaderStage::kFragment;
  p.summary.binary_size=p.instructions.size()*8;p.summary.instruction_count=p.summary.group_count=p.instructions.size();
  p.summary.pixel_output_mask=mask;p.summary.early_hsr_safe=1;return p;
}
PcoInstruction ReadFace(unsigned output,PcoWriteTarget target){
  PcoInstruction i;i.opcode=PcoOpcode::kMoveBypass;i.target=target;i.output_index=output;
  i.source={PcoRegisterBank::kSpecial,kPcoSpecialFragmentBackFace};return i;
}
PcoFragmentExecutionContext Context(unsigned facing){
  PcoFragmentExecutionContext c;c.front_facing=facing;c.front_facing_valid=1;return c;
}
void TestDirect(){
  auto p=Semantic({ReadFace(0,PcoWriteTarget::kPixelOutput)},1);
  const PcoPreparedFragmentProgram prepared(p.summary,p.instructions);
  uint32_t ignored=0;Check(!PcoSpecialConstantBits(44,&ignored),"FACE must not become a compile-time special constant");
  for(unsigned facing:{0U,1U}){
    auto c=Context(facing);
    const auto raw=ExecuteFragmentPco(p.summary,p.instructions,c),fast=ExecuteFragmentPco(prepared,c);
    Check(raw.pixel_outputs[0]==1-facing&&fast.pixel_outputs==raw.pixel_outputs,"special44 is normalized back-face0/1");
    Check(raw.written_mask==1&&!raw.suspended&&raw.executed_instructions.alu==1,"FACE read is one ordinary ALU issue");
  }
  for(auto pair:std::array<std::array<unsigned,2>,6>{{{{0,0}},{{1,0}},{{2,1}},{{255,1}},{{0,2}},{{1,255}}}}){
    auto c=Context(pair[0]);c.front_facing_valid=pair[1];
    Reject([&]{ExecuteFragmentPco(p.summary,p.instructions,c);},"raw accepted absent/noncanonical raster facing");
    Reject([&]{ExecuteFragmentPco(prepared,c);},"prepared accepted absent/noncanonical raster facing");
  }
  auto plain=p;plain.instructions[0].source={PcoRegisterBank::kSpecial,0};
  const PcoPreparedFragmentProgram old(plain.summary,plain.instructions);
  Check(ExecuteFragmentPco(old).pixel_outputs[0]==0,"legacy program withoutFACE keeps default-context behavior");
  auto changed=p.instructions;changed[0].source.index=45;
  Reject([&]{ExecuteFragmentPco(p.summary,changed,Context(1));},"neighbor unsupported special register accepted");
}
void TestTexture(){
  // The ordinary texture program is genuine compiled bytes. Replacing one
  // post-WDF PIXOUT source is an explicit semantic continuation guard test.
  auto p=DecodePcoProgram(ShaderStage::kFragment,test::TextureGatherFixture(2));
  auto output=std::find_if(p.instructions.begin(),p.instructions.end(),[](const auto&i){return i.target==PcoWriteTarget::kPixelOutput&&i.output_index==0;});
  Check(output!=p.instructions.end(),"ordinary compiled fixture missingPIXOUT0");
  output->source={PcoRegisterBank::kSpecial,kPcoSpecialFragmentBackFace};
  const PcoPreparedFragmentProgram prepared(p.summary,p.instructions);
  const std::array<uint32_t,4> response{Bits(.125F),Bits(.25F),Bits(.5F),Bits(.75F)};
  for(unsigned facing:{0U,1U})for(bool fast:{false,true}){
    auto c=Context(facing);c.shared_count=22;c.shared_registers[20]=Bits(.25F);c.shared_registers[21]=Bits(.75F);
    const auto first=fast?ExecuteFragmentPco(prepared,c):ExecuteFragmentPco(p.summary,p.instructions,c);
    Check(first.suspended&&first.texture_request_valid&&first.executed_instructions.texture==1,"single original texture request");
    Check(first.continuation.front_facing==facing&&first.continuation.front_facing_valid==1,"WDF checkpoint pins raster facing before laterFACE read");
    c.continuation=first.continuation;c.texture_response=response;c.texture_response_valid=1;
    const auto done=fast?ExecuteFragmentPco(prepared,c):ExecuteFragmentPco(p.summary,p.instructions,c);
    Check(done.pixel_outputs[0]==1-facing&&done.pixel_outputs[1]==response[1]&&done.pixel_outputs[2]==response[2]&&done.pixel_outputs[3]==response[3],"postWDF FACE uses same actual orientation");
    Check(!done.suspended&&done.executed_instructions.texture==1,"resume must not reissue sampler");
    const auto convenient=fast?ResumeFragmentPco(prepared,first.continuation,response):ResumeFragmentPco(p.summary,p.instructions,first.continuation,response);
    Check(convenient.pixel_outputs==done.pixel_outputs,"convenience resume retains saved facing");
    for(unsigned mutation=0;mutation<5;++mutation){
      auto bad=c;
      if(mutation==0)bad.front_facing^=1;
      if(mutation==1)bad.continuation.front_facing^=1;
      if(mutation==2)bad.front_facing_valid=0;
      if(mutation==3)bad.continuation.front_facing_valid=0;
      if(mutation==4)bad.front_facing=bad.continuation.front_facing=2;
      Reject([&]{if(fast)ExecuteFragmentPco(prepared,bad);else ExecuteFragmentPco(p.summary,p.instructions,bad);},"changed raster facing resumed a saved texture checkpoint");
    }
    auto absent=c;absent.continuation={};absent.texture_response_valid=0;absent.front_facing=absent.front_facing_valid=0;
    Reject([&]{if(fast)ExecuteFragmentPco(prepared,absent);else ExecuteFragmentPco(p.summary,p.instructions,absent);},"deferred FACE use must reject absent input before issuingSMP");
  }
}
void TestDerivative(){
  auto face=ReadFace(0,PcoWriteTarget::kTemporary);
  PcoInstruction derivative;derivative.opcode=PcoOpcode::kDerivativeX;derivative.target=PcoWriteTarget::kTemporary;
  derivative.source={PcoRegisterBank::kTemporary,0};derivative.output_index=1;
  auto pixel=ReadFace(0,PcoWriteTarget::kPixelOutput);
  auto p=Semantic({face,derivative,pixel},1);
  const PcoPreparedFragmentProgram prepared(p.summary,p.instructions);
  for(unsigned facing:{0U,1U})for(bool fast:{false,true}){
    auto c=Context(facing);
    const auto first=fast?ExecuteFragmentPco(prepared,c):ExecuteFragmentPco(p.summary,p.instructions,c);
    Check(first.derivative_request_valid&&first.continuation.kind==1,"derivative checkpoint reached");
    c.continuation=first.continuation;c.derivative_response_valid=1;c.derivative_response=0;
    const auto done=fast?ExecuteFragmentPco(prepared,c):ExecuteFragmentPco(p.summary,p.instructions,c);
    Check(done.pixel_outputs[0]==1-facing&&!done.suspended,"derivative preserves raster-facing value");
    c.front_facing^=1;
    Reject([&]{if(fast)ExecuteFragmentPco(prepared,c);else ExecuteFragmentPco(p.summary,p.instructions,c);},"derivative resumed with changedfacing");
  }
}
void TestNativeCompiler(){
  const std::array<uint32_t,4> red{Bits(1),Bits(0),Bits(0),Bits(1)};
  const std::array<uint32_t,4> green{Bits(0),Bits(1),Bits(0),Bits(1)};
  const std::array<uint32_t,4> response{Bits(.125F),Bits(.25F),Bits(.5F),Bits(.75F)};
  for(unsigned kind=0;kind<6;++kind){
    const auto p=DecodePcoProgram(ShaderStage::kFragment,test::FrontFaceFixture(kind));
    const PcoPreparedFragmentProgram prepared(p.summary,p.instructions);
    Check(p.summary.pixel_output_mask==15,"genuine compiler FACE output mask");
    for(unsigned facing:{0U,1U})for(bool fast:{false,true}){
      auto c=Context(facing);c.coefficient_count=4;c.shared_count=kind>=4?20:0;
      const auto first=fast?ExecuteFragmentPco(prepared,c):ExecuteFragmentPco(p.summary,p.instructions,c);
      auto done=first;
      if(kind>=4){
        Check(first.suspended&&first.texture_request_valid&&first.executed_instructions.texture==1,
              "genuine FACE program issues exactly one selected SMP");
        const auto &request=first.texture_request;
        Check(request.coordinates[0]==Bits(kind==4&&!facing?.75F:.25F)&&
                  request.coordinates[1]==Bits(.5F)&&request.descriptor_set==0&&
                  request.explicit_lod_present==1&&request.explicit_lod==Bits(0)&&
                  request.gather==0,
              "genuine FACE branch selects actual UV without changing sampling ABI");
        c.continuation=first.continuation;c.texture_response=response;c.texture_response_valid=1;
        done=fast?ExecuteFragmentPco(prepared,c):ExecuteFragmentPco(p.summary,p.instructions,c);
        auto changed=c;changed.front_facing^=1;
        Reject([&]{if(fast)ExecuteFragmentPco(prepared,changed);else ExecuteFragmentPco(p.summary,p.instructions,changed);},
               "genuine FACE SMP resumed with different raster input");
      }
      const auto &expected=kind==4||(kind==5&&facing)?response:kind==0||facing?red:green;
      Check(std::equal(expected.begin(),expected.end(),done.pixel_outputs.begin())&&
                done.written_mask==15&&!done.suspended&&done.executed_instructions.texture==(kind>=4?1U:0U),
            "genuine compiled FACE front/back and WDF result mismatch");
      if(kind){
        auto absent=c;absent.front_facing=absent.front_facing_valid=0;
        absent.continuation={};absent.texture_response_valid=0;
        Reject([&]{if(fast)ExecuteFragmentPco(prepared,absent);else ExecuteFragmentPco(p.summary,p.instructions,absent);},
               "genuine FACE program accepted missing raster input");
      }
    }
  }
}
}
int main(){try{TestDirect();TestTexture();TestDerivative();TestNativeCompiler();std::cout<<"front-facing ISS: "<<checks<<" checks PASS\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
