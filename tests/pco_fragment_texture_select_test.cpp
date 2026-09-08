// SPDX-License-Identifier: MIT
// Native instruction/TextureUnit continuation boundary. Responses below are
// explicit unit inputs, not a claimed full raster or dEQP shader result.
#include "shader/pco_iss.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace {
using namespace pvrgpu::stub;
unsigned checks;
void Check(bool good, const char *reason) {
  ++checks; if (!good) throw std::runtime_error(reason);
}
std::uint32_t Bits(float f) { std::uint32_t u; std::memcpy(&u,&f,4); return u; }
float Float(std::uint32_t u) { float f; std::memcpy(&f,&u,4); return f; }
template<typename F> void Reject(F function, const char *reason) {
  bool rejected=false; try{function();}catch(const std::exception&){rejected=true;}
  Check(rejected,reason);
}
constexpr float directions[6][3] = {
  {1,.25f,-.5f}, {-1,.25f,.5f}, {.25f,1,-.5f},
  {.25f,-1,.5f}, {.25f,-.5f,1}, {-.25f,-.5f,-1}};
void Test(const char *path, bool dynamic) {
  std::ifstream file(path,std::ios::binary);
  Check(bool(file),"read actual compiler binary");
  const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(file),{}};
  const auto program=DecodePcoProgram(ShaderStage::kFragment,bytes);
  const auto samples=std::count_if(program.instructions.begin(),program.instructions.end(),
      [](const auto &i){return i.opcode==PcoOpcode::kTextureSample;});
  Check(samples==6,"six real native SMP instructions, no host branch dispatch");
  for(int selector : {-13,-1,0,1,2,3,4,5,6,23,INT32_MAX}) {
    for (unsigned poison=0; poison!=(dynamic?3U:1U); ++poison) {
      PcoFragmentExecutionContext context;
      context.shared_count=dynamic?48:24;
      context.shared_registers[20]=static_cast<std::uint32_t>(selector);
      if(dynamic) for(unsigned branch=0;branch!=6;++branch) {
        for(unsigned c=0;c!=3;++c)
          context.shared_registers[24+branch*4+c]=
              selector==int(branch)?Bits(directions[branch][c]):
              poison==0?UINT32_C(0x7fc12345):poison==1?UINT32_C(0x7f800000):0;
        context.shared_registers[27+branch*4]=selector==int(branch)?Bits(2):UINT32_C(0x7fc23456);
      }
      auto result=ExecuteFragmentPco(program.summary,program.instructions,context);
      unsigned requests=0, active_requests=0;
      while(result.suspended) {
        Check(result.texture_request_valid,"native continuation must identify its SMP request");
        const auto &request=result.texture_request;
        Check(request.explicit_lod_present && request.dimension==3,
              "real explicit cube operands survive instruction decode");
        const float x=Float(request.coordinates[0]),y=Float(request.coordinates[1]),z=Float(request.coordinates[2]);
        Check(std::isfinite(x)&&std::isfinite(y)&&std::isfinite(z)&&
              std::max({std::fabs(x),std::fabs(y),std::fabs(z)})>0&&
              std::isfinite(Float(request.explicit_lod)),
              "unselected NaN/Inf/zero-cube/NaN-LOD never reaches TextureUnit");
        int face=std::fabs(x)>=std::fabs(y)&&std::fabs(x)>=std::fabs(z)?(x<0?1:0):
            std::fabs(y)>=std::fabs(z)?(y<0?3:2):(z<0?5:4);
        if(dynamic && Float(request.explicit_lod)==2) {
          ++active_requests;
          Check(selector>=0&&selector<6&&face==selector,"active branch retains original raw coordinate and LOD");
        } else if(dynamic) {
          Check(x==0&&y==0&&z==1&&Float(request.explicit_lod)==0,
                "inactive branch uses safe native-selected inputs, not invalid application values");
        }
        const std::array<std::uint32_t,4> response={Bits(float(face+1)),Bits(.25f),Bits(.75f),Bits(1)};
        context.continuation=result.continuation; context.texture_response=response;
        context.texture_response_valid=1;
        result=ExecuteFragmentPco(program.summary,program.instructions,context);
        Check(++requests<=6,"bounded native sampling continuation");
      }
      Check(requests==6,"all speculative texture instructions execute natively");
      if(dynamic) Check(active_requests==unsigned(selector>=0&&selector<6),"exactly original selected branch uses application inputs");
      const std::array<std::uint32_t,4> expected=selector>=0&&selector<6?
          std::array<std::uint32_t,4>{Bits(float(selector+1)),Bits(.25f),Bits(.75f),Bits(1)}:
          std::array<std::uint32_t,4>{Bits(1),Bits(0),Bits(1),Bits(1)};
      Check(std::equal(expected.begin(),expected.end(),result.pixel_outputs.begin()),
            "native ALU chooses original sampled value or arbitrary-selector default");
    }
  }
  if(dynamic) {
    PcoFragmentExecutionContext context; context.shared_count=48;
    context.shared_registers[20]=2;
    context.shared_registers[24+2*4]=UINT32_C(0x7fcabcde);
    context.shared_registers[25+2*4]=Bits(1);
    context.shared_registers[27+2*4]=UINT32_C(0x7f800000);
    auto result=ExecuteFragmentPco(program.summary,program.instructions,context);
    unsigned invalid_active=0;
    while(result.suspended) {
      const auto &r=result.texture_request;
      if(!std::isfinite(Float(r.coordinates[0]))||!std::isfinite(Float(r.explicit_lod))) {
        ++invalid_active;
        Check(r.coordinates[0]==UINT32_C(0x7fcabcde)&&r.explicit_lod==UINT32_C(0x7f800000),
              "selected invalid operands remain exact rather than silently sanitized");
      }
      context.continuation=result.continuation; context.texture_response={};
      context.texture_response_valid=1;
      result=ExecuteFragmentPco(program.summary,program.instructions,context);
    }
    Check(invalid_active==1,"only original active invalid texture operation retains its error path");
  }
  std::cout<<"native select fixture bytes="<<bytes.size()<<" instructions="<<program.instructions.size()<<" SMP="<<samples<<" PASS\n";
}

void TestResponseRegisters(const char *binary_path, const char *records_path) {
  std::ifstream file(binary_path,std::ios::binary), records(records_path,std::ios::binary);
  const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(file),{}};
  const std::vector<std::uint8_t> encodings{std::istreambuf_iterator<char>(records),{}};
  Check(encodings.size()==7*6,"exact upper-source bytes came from original Mesa ISA encoder");
  const auto original=DecodePcoProgram(ShaderStage::kFragment,bytes);
  const auto sample=std::find_if(original.instructions.begin(),original.instructions.end(),
      [](const auto &i){return i.opcode==PcoOpcode::kTextureSample;});
  Check(sample!=original.instructions.end(),"native fixture has response-register probe location");
  const auto group=sample->binary_offset-3;
  const auto old_size=(bytes[group]&15)*2;
  const auto upper=group+11; // Native 3-byte header, 3-byte SMP, 5-byte lower block.
  Check(old_size==14&&bytes[upper]==0x80&&(bytes[upper+1]&0xe0)==0xa0&&bytes[upper+2]==0,
        "genuine compiler probe starts with canonical brief response encoding");
  for(unsigned record=0;record!=7;++record) {
    const unsigned response=encodings[record*6], length=encodings[record*6+1];
    Check(length==(response<128?3U:4U),"Mesa selects appropriate extended upper-source form");
    std::vector<std::uint8_t> native(bytes.begin()+group,bytes.begin()+upper);
    native.insert(native.end(),encodings.begin()+record*6+2,encodings.begin()+record*6+2+length);
    native.push_back(0); // Existing canonical ISS selector, not an instruction.
    if(native.size()%2) native.push_back(0xff);
    native[0]=static_cast<std::uint8_t>((native[0]&0xf0)|(native.size()/2));
    auto patched=bytes;
    patched.erase(patched.begin()+group,patched.begin()+group+old_size);
    patched.insert(patched.begin()+group,native.begin(),native.end());
    if(response>252) {
      Reject([&]{(void)DecodePcoProgram(ShaderStage::kFragment,patched);},
             "SMP response r253/r255 cannot fit four raw DWORDs");
      continue;
    }
    const auto program=DecodePcoProgram(ShaderStage::kFragment,patched);
    const auto index=std::size_t(sample-original.instructions.begin());
    Check(program.instructions[index].output_index==response,"native upper response index is never truncated to five/seven bits");
    PcoFragmentExecutionContext context; context.shared_count=24; context.shared_registers[20]=0;
    const auto first=ExecuteFragmentPco(program.summary,program.instructions,context);
    Check(first.suspended&&first.continuation.pending_output_index==response,
          "native executor preserves bounded four-DWORD response destination through suspension");
    auto semantic=program;
    semantic.instructions[index].output_index=253;
    Reject([&]{(void)ExecuteFragmentPco(semantic.summary,semantic.instructions,context);},
           "decoded instruction boundary also rejects out-of-range SMP response");
    for(unsigned mutation=0;mutation!=3;++mutation) {
      auto bad=patched;
      if(mutation==0) bad[upper]|=1; // noncanonical unused s3=sc1
      if(mutation==1) bad[upper+2]|=0x08; // s4 coefficient rather than TEMP
      if(mutation==2) bad[upper+2]|=0x20; // reserved upper selector bit
      Reject([&]{(void)DecodePcoProgram(ShaderStage::kFragment,bad);},
             "extended response rejects dummy-source/bank/reserved corruption");
    }
    for(unsigned truncated=1;truncated<length;++truncated) {
      const std::vector<std::uint8_t> bad(patched.begin(),patched.begin()+upper+truncated);
      Reject([&]{(void)DecodePcoProgram(ShaderStage::kFragment,bad);},
             "truncated extended response is rejected before reading omitted bytes");
    }
  }
}
}
int main(int argc,char **argv) {
  try {
    Check(argc==4,"expected two compiler-produced native fixtures and ISA response records");
    Test(argv[1],false); Test(argv[2],true);
    TestResponseRegisters(argv[1],argv[3]);
    std::cout<<"native fragment texture select "<<checks<<" checks PASS\n";return 0;
  }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
