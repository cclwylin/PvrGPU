#include "shader/tessellation_iss.h"
#include "pco_tessellation_compiler_fixtures.h"
#include "pco_tessellation_patch_fixtures.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace pvrgpu::stub;
namespace {
unsigned checks = 0;
void Check(bool condition,const char *message) {
  ++checks; if (!condition) throw std::runtime_error(message);
}
template<class Fn> void Reject(Fn fn) {
  try { fn(); } catch (const std::exception &) { ++checks; return; }
  throw std::runtime_error("expected native tessellation rejection");
}
std::uint32_t Bits(float value) { std::uint32_t bits; std::memcpy(&bits,&value,4); return bits; }
float Float(std::uint32_t bits) { float value; std::memcpy(&value,&bits,4); return value; }
struct Memory {
  std::vector<std::uint32_t> input;
  std::vector<std::uint32_t> output;
  std::vector<std::uint32_t> control_uniform,evaluation_uniform;
  std::vector<bool> written;
  bool require_written_reads=true;
  unsigned reads=0,writes=0;
  static void Read(void *opaque,std::uint64_t address,std::uint32_t count,std::uint32_t *destination) {
    auto &m=*static_cast<Memory*>(opaque);
    const bool uniform=address>=UINT64_C(0x8100003000);
    const bool evaluation=address>=UINT64_C(0x8100004000);
    const bool out=!uniform&&address>=UINT64_C(0x8100002000);
    const auto base=uniform?(evaluation?UINT64_C(0x8100004000):UINT64_C(0x8100003000)):
        out?UINT64_C(0x8100002000):UINT64_C(0x8100001000);
    const auto &source=uniform?(evaluation?m.evaluation_uniform:m.control_uniform):out?m.output:m.input;
    Check(address>=base && (address-base)%4==0 && address-base<=source.size()*4 && count*4<=source.size()*4-(address-base),"native LD inside input/output extent");
    const auto offset=(address-base)/4;
    if(out&&m.require_written_reads) for(unsigned c=0;c<count;++c) Check(m.written[offset+c],"cross-invocation LD observes a prior native ST, never initialized expected data");
    std::copy_n(source.data()+offset,count,destination); ++m.reads;
  }
  static void Write(void *opaque,std::uint64_t address,std::uint32_t count,const std::uint32_t *source) {
    auto &m=*static_cast<Memory*>(opaque);
    const auto base=UINT64_C(0x8100002000);
    Check(address>=base && (address-base)%4==0 && address-base<=m.output.size()*4 && count*4<=m.output.size()*4-(address-base),"native ST inside output extent");
    const auto offset=(address-base)/4;
    std::copy_n(source,count,m.output.data()+offset);
    for(unsigned c=0;c<count;++c)m.written[offset+c]=true;
    ++m.writes;
  }
};
void SignedNativeAlu() {
  const auto program=DecodeTessellationPcoProgram(ShaderStage::kTessellationControl,
      kTessPatchFiveToTenTcs);
  const PcoInstruction *multiply=nullptr,*shift=nullptr;
  for(const auto &instruction:program.instructions) {
    if(instruction.opcode==PcoOpcode::kIntegerMultiplyAdd64High) multiply=&instruction;
    if(instruction.opcode==PcoOpcode::kShiftRight&&instruction.integer_signed) shift=&instruction;
  }
  Check(multiply&&multiply->integer_signed,"actual compiler IMADD64.s preserves its signed encoding");
  Check(shift&&shift->source_count==2,"actual compiler ASR_TWB is decoded independently");
  const std::array<std::uint32_t,10> values{{0,1,2,0x7fffffff,0x80000000,
      0x80000001,0xffffffff,0xffff,0x10000,0xdeadbeef}};
  const auto signed_value=[](std::uint32_t value)->std::int64_t {
    return value&UINT32_C(0x80000000)?std::int64_t{value}-INT64_C(0x100000000):value;
  };
  for(bool is_signed:{false,true}) {
    auto operation=*multiply;operation.integer_signed=is_signed;
    for(auto a:values)for(auto b:values)for(auto low:values)for(auto high:values) {
      // An independent 128-bit oracle catches both 64-bit carry and signed
      // edge products without sharing the executor's high-word correction.
      const __int128 product=is_signed?__int128{signed_value(a)}*signed_value(b):
          __int128{a}*b;
      const __int128 addend=(__int128{high}<<32)+low;
      const auto expected=static_cast<std::uint32_t>(
          static_cast<std::uint64_t>(product+addend)>>32);
      Check(EvaluatePcoAluInstruction(operation,{a,b,low,high})==expected,
            "native signed/unsigned multiply-add high retains modulo-64 carry");
    }
    auto operation_shift=*shift;operation_shift.integer_signed=is_signed;
    for(auto value:values)for(unsigned count=0;count<96;++count) {
      const unsigned n=count&31;
      std::uint32_t expected=0;
      for(unsigned bit=0;bit<32;++bit) {
        const unsigned source=bit+n;
        const bool one=source<32?((value>>source)&1)!=0:
            is_signed&&(value&UINT32_C(0x80000000));
        if(one)expected|=UINT32_C(1)<<bit;
      }
      Check(EvaluatePcoAluInstruction(operation_shift,{value,count,0,0})==expected,
            "ASR_TWB sign extension and modulo-32 count; SHR remains logical");
    }
  }
  for(auto invalid:{0xe7U,0xefU,0xf3U,0xfbU}) {
    auto bytes=kTessPatchFiveToTenTcs;bytes[multiply->binary_offset]=invalid;
    Reject([&]{DecodeTessellationPcoProgram(ShaderStage::kTessellationControl,bytes);});
  }
  for(auto invalid:{2U,3U,5U,6U,7U}) {
    auto bytes=kTessPatchFiveToTenTcs;bytes[shift->binary_offset]=invalid;
    Reject([&]{DecodeTessellationPcoProgram(ShaderStage::kTessellationControl,bytes);});
  }
}

void UnequalPatches() {
  const std::vector<std::uint8_t>*controls[]={&kTessPatchFiveToTenTcs,&kTessPatchTenToFiveTcs};
  const std::vector<std::uint8_t>*evaluations[]={&kTessPatchFiveToTenTes,&kTessPatchTenToFiveTes};
  for(unsigned kind=0;kind<2;++kind) {
    const unsigned inputs=kind?10:5,outputs=kind?5:10;
    auto control=DecodeTessellationPcoProgram(ShaderStage::kTessellationControl,*controls[kind]);
    auto evaluation=DecodeTessellationPcoProgram(ShaderStage::kTessellationEvaluation,*evaluations[kind]);
    DriverPcoStageAbi tcs,tes;
    tcs.temps=8;tcs.vertex_inputs=3;tcs.shareds=8;
    tcs.uniform_buffer_descriptor_start=tcs.push_constant_start=8;
    tes.temps=5;tes.vertex_inputs=5;tes.shareds=4;tes.vertex_outputs=4;
    tes.uniform_buffer_descriptor_start=tes.push_constant_start=4;
    ValidateTessellationProgram(control,tcs);ValidateTessellationProgram(evaluation,tes);
    for(unsigned epoch=0;epoch<17;++epoch) {
      Memory memory;
      memory.input.resize(inputs*4,0xc0decafe);
      for(unsigned vertex=0;vertex<inputs;++vertex)
        memory.input[vertex*4]=Bits(float(vertex+1+epoch)*.03125f);
      memory.output.resize(6+outputs*4,0xc0decafe);
      memory.written.resize(memory.output.size(),false);
      // Scalar varying occupies the first DWORD of a vec4-aligned record.
      // Native vector loads may fetch untouched padding, but only the actual
      // scalar ST may determine TES results; padding remains poisoned below.
      memory.require_written_reads=false;
      std::vector<std::uint32_t> shared={0x1000,0x81,inputs*16,0,
          0x2000,0x81,static_cast<unsigned>(memory.output.size()*4),0};
      auto task=MakeTessellationControlTask(tcs,shared,epoch,inputs,outputs);
      TessellationExecutionStats stats;
      const TessellationMemoryCallbacks callbacks{&memory,Memory::Read,Memory::Write};
      while(!task.ended)StepTessellationTask(control,tcs,task,callbacks,stats);
      Check(stats.store_instructions==memory.writes&&stats.load_instructions==memory.reads,
            "unequal-patch native request counters");
      for(unsigned c=0;c<6;++c)
        Check(memory.written[c]&&memory.output[c]==Bits(5),"unequal TCS actual six tessellation levels");
      for(unsigned vertex=0;vertex<outputs;++vertex) {
        Check(memory.written[6+vertex*4]&&
                  memory.output[6+vertex*4]==memory.input[(vertex*inputs/outputs)*4],
              "native signed division maps each real output invocation to the correct input");
        for(unsigned c=1;c<4;++c)
          Check(!memory.written[6+vertex*4+c]&&memory.output[6+vertex*4+c]==0xc0decafe,
                "scalar varying never invents stores to vec4 padding");
      }
      for(unsigned count:{1U,7U,32U}) {
        std::array<std::array<std::uint32_t,3>,32> coords{};
        for(unsigned lane=0;lane<count;++lane)
          coords[lane]={Bits(float(lane%17)/16),Bits(float(lane%7)/8),Bits(0)};
        std::vector<std::uint32_t> te_shared={0x2000,0x81,
            static_cast<unsigned>(memory.output.size()*4),0};
        auto task_te=MakeTessellationEvaluationTask(tes,te_shared,epoch,outputs,coords.data(),count);
        TessellationExecutionStats stats_te;
        while(!task_te.ended)StepTessellationTask(evaluation,tes,task_te,callbacks,stats_te);
        Check(stats_te.emit_instructions==count,"unequal-patch TES emits once per actual coordinate");
        for(unsigned lane=0;lane<count;++lane) {
          const float u=Float(coords[lane][0]),v=Float(coords[lane][1]);
          const float scaled=u*(outputs-1);
          unsigned output_index=static_cast<unsigned>(std::floor(scaled));
          if(scaled-output_index>.5f||(scaled-output_index==.5f&&(output_index&1)))++output_index;
          const unsigned source=output_index*inputs/outputs;
          Check(task_te.lanes[lane].outputs[0]==Bits(std::fma(u,2,-1))&&
                    task_te.lanes[lane].outputs[1]==Bits(v-Float(memory.input[source*4]))&&
                    task_te.lanes[lane].outputs[2]==Bits(0)&&task_te.lanes[lane].outputs[3]==Bits(1),
                "real TES dynamic rounded indexing consumes native unequal-patch TCS writes");
        }
      }
    }
  }
}
void Fixtures() {
  const std::vector<std::uint8_t>*controls[]={&kTess0tcs,&kTess1tcs,&kTess2tcs,&kTess3tcs,&kTess4tcs};
  const std::vector<std::uint8_t>*evaluations[]={&kTess0tes,&kTess1tes,&kTess2tes,&kTess3tes,&kTess4tes};
  const unsigned control_temps[]={6,13,12,13,13},evaluation_temps[]={2,11,20,20,20};
  for(unsigned kind=0;kind<5;++kind) {
    std::cout<<"fixture "<<kind<<std::endl;
    auto control=DecodeTessellationPcoProgram(ShaderStage::kTessellationControl,*controls[kind]);
    auto evaluation=DecodeTessellationPcoProgram(ShaderStage::kTessellationEvaluation,*evaluations[kind]);
    DriverPcoStageAbi tcs,tes;
    tcs.temps=control_temps[kind]; tcs.vertex_inputs=3; tcs.shareds=8;
    tcs.uniform_buffer_descriptor_start=tcs.push_constant_start=8;
    tes.temps=evaluation_temps[kind]; tes.vertex_inputs=5; tes.shareds=4;
    tes.vertex_outputs=kind>=2?8:4; tes.uniform_buffer_descriptor_start=tes.push_constant_start=4;
    if(kind==3){tcs.shareds=12;tcs.push_constant_count=4;tes.shareds=12;tes.push_constant_count=8;}
    if(kind==4){tcs.shareds=12;tcs.uniform_buffer_descriptor_count=1;tcs.push_constant_start=12;
               tes.shareds=8;tes.uniform_buffer_descriptor_count=1;tes.push_constant_start=8;}
    ValidateTessellationProgram(control,tcs); ValidateTessellationProgram(evaluation,tes);
    for(unsigned epoch=0;epoch<23;++epoch) {
      const unsigned vertices=kind?3:1,offset=kind>=2?10:6,stride=kind>=2?8:4;
      const float level=kind>=3?1+epoch*.25f:5;
      Memory memory;
      for(unsigned i=0;i<vertices*4;++i)memory.input.push_back(Bits(float(i+1+epoch)/32));
      memory.output.resize(offset+vertices*stride,0xc0decafe); memory.written.resize(memory.output.size(),false);
      std::vector<std::uint32_t> shared={0x1000,0x81,vertices*16,0,0x2000,0x81,static_cast<unsigned>(memory.output.size()*4),0};
      shared.resize(tcs.shareds);
      if(kind==3)shared[8]=Bits(level);
      if(kind==4){shared[8]=0x3000;shared[9]=0x81;shared[10]=16;memory.control_uniform={Bits(level),0,0,0};}
      auto task=MakeTessellationControlTask(tcs,shared,9+epoch,vertices,vertices);
      TessellationExecutionStats stats;
      const TessellationMemoryCallbacks callbacks{&memory,Memory::Read,Memory::Write};
      while(!task.ended)StepTessellationTask(control,tcs,task,callbacks,stats);
      Check(stats.store_instructions==memory.writes && stats.load_instructions==memory.reads,"TCS native request counts");
      for(unsigned i=0;i<6;++i)Check(memory.written[i]&&memory.output[i]==Bits(level),"native TCS wrote all actual levels");
      if(kind)for(unsigned vertex=0;vertex<vertices;++vertex)for(unsigned c=0;c<4;++c)
        Check(memory.output[offset+vertex*stride+c]==memory.input[vertex*4+c],"TCS invocation writes its actual input vertex");
      if(kind>=2) {
        for(unsigned c=0;c<4;++c)Check(memory.output[6+c]==Bits(float(c+1)*.25f),"TCS invocation zero writes patch varying");
        for(unsigned vertex=0;vertex<vertices;++vertex)for(unsigned c=0;c<4;++c)
          Check(memory.output[offset+vertex*stride+4+c]==memory.input[((vertex+1)%3)*4+c],"barrier cross-lane read follows all producer native stores");
      }
      std::array<std::array<std::uint32_t,3>,32> coords{};
      for(unsigned count: {1U,7U,32U}) {
        for(unsigned lane=0;lane<count;++lane) {
          const float u=float(lane%5)*.125f,v=float(lane%3)*.125f;
          coords[lane]={Bits(u),Bits(v),Bits(1-u-v)};
        }
        std::vector<std::uint32_t> te_shared={0x2000,0x81,static_cast<unsigned>(memory.output.size()*4),0};
        te_shared.resize(tes.shareds);
        float scale=1;
        if(kind==3){te_shared[4]=Bits(.5f);te_shared[8]=Bits(1.25f);scale=((9+epoch)&1)?1.25f:.5f;}
        if(kind==4){te_shared[4]=0x4000;te_shared[5]=0x81;te_shared[6]=16;
          memory.evaluation_uniform.clear();for(unsigned c=0;c<4;++c)memory.evaluation_uniform.push_back(Bits(float(c+1)*.125f+epoch*.03125f));
          scale=Float(memory.evaluation_uniform[(9+epoch)&3]);}
        auto te=MakeTessellationEvaluationTask(tes,te_shared,9+epoch,vertices,coords.data(),count);
        TessellationExecutionStats te_stats;
        while(!te.ended)StepTessellationTask(evaluation,tes,te,callbacks,te_stats);
        Check(te_stats.emit_instructions==count,"TES executes exactly one UVSW emission per domain point");
        for(unsigned lane=0;lane<count;++lane)for(unsigned c=0;c<4;++c) {
          float expected;
          if(!kind)expected=c<2?std::fma(Float(coords[lane][c]),1.6f,-.8f):c==2?0:1;
          else expected=std::fma(Float(memory.input[8+c]),Float(coords[lane][2]),
              std::fma(Float(memory.input[4+c]),Float(coords[lane][1]),Float(memory.input[c])*Float(coords[lane][0])));
          Check(std::abs(Float(te.lanes[lane].outputs[c])-expected)<1e-6f,"TES executes real coord interpolation");
          if(kind>=2)Check(te.lanes[lane].outputs[4+c]==Bits((float(c+1)*.25f+Float(memory.input[4+c]))*scale),"TES consumes native patch/crossvertex outputs and stage-local CB0/UBO");
        }
      }
    }
    auto bad=tcs;bad.vertex_inputs=2;Reject([&]{ValidateTessellationProgram(control,bad);});
    bad=tcs;bad.uniform_buffer_descriptor_start=4;Reject([&]{ValidateTessellationProgram(control,bad);});
    auto bytes=*controls[kind];bytes.push_back(0);Reject([&]{DecodeTessellationPcoProgram(ShaderStage::kTessellationControl,bytes);});
    Reject([&]{DecodeTessellationPcoProgram(ShaderStage::kCompute,*controls[kind]);});
    Reject([&]{DecodeTessellationPcoProgram(ShaderStage::kTessellationControl,*evaluations[kind]);});
  }
}
} // namespace
int main() {
  try { SignedNativeAlu();UnequalPatches();Fixtures();std::cout<<"native tessellation ISS: PASS "<<checks<<" checks\n";return 0; }
  catch(const std::exception&e){std::cerr<<e.what()<<" after "<<checks<<" checks\n";return 1;}
}
