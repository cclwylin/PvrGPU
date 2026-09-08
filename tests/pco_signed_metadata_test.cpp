// SPDX-License-Identifier: MIT
// Native decoded-POD boundary checks, using real PCO programs as the valid
// baseline. Mutations must fail specifically on integer signedness before
// any opcode-specific memory/control/export work, never an unrelated error.
#include "shader/compute_iss.h"
#include "shader/geometry_iss.h"
#include "shader/tessellation_iss.h"
#include "pco_compute_fixtures.h"
#include "pco_geometry_fixtures.h"
#include "pco_tessellation_compiler_fixtures.h"
#include "pco_tessellation_patch_fixtures.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

namespace {
using namespace pvrgpu::stub;
unsigned checks=0;
std::set<PcoOpcode> rejected_opcodes;
void Check(bool condition,const std::string &reason) {
  ++checks;
  if(!condition)throw std::runtime_error(reason);
}
template<class Fn> void RejectSigned(Fn fn) {
  try { fn(); }
  catch(const std::exception &error) {
    Check(std::string(error.what()).find("integer signedness")!=std::string::npos,
          std::string("mutation failed for an unrelated reason: ")+error.what());
    return;
  }
  throw std::runtime_error("noncanonical integer signedness was accepted");
}
template<class Validate,class Step> void Mutate(
    const PcoDecodedProgram &program,Validate validate,Step step) {
  validate(program);
  for(std::size_t index=0;index<program.instructions.size();++index)
    for(unsigned value:{0U,1U,2U,255U}) {
      auto mutated=program;
      auto &instruction=mutated.instructions[index];
      instruction.integer_signed=static_cast<std::uint8_t>(value);
      const bool allowed=value==0||(value==1&&
          (instruction.opcode==PcoOpcode::kIntegerMultiplyAdd64High||
           instruction.opcode==PcoOpcode::kShiftRight));
      Check(HasCanonicalNativeIntegerSignedness(instruction)==allowed,
            "common signed predicate matches the declared opcode/Boolean contract");
      if(allowed) {
        validate(mutated);
      } else {
        RejectSigned([&]{validate(mutated);});
        RejectSigned([&]{step(mutated,index);});
        RejectSigned([&]{EvaluatePcoAluInstruction(instruction,{0,0,0,0});});
        rejected_opcodes.insert(instruction.opcode);
      }
    }
}

void Compute() {
  auto program=DecodeComputePcoProgram(ComputePcoFixture(1));
  ComputePcoAbi abi;
  abi.local_size={30,1,1};
  abi.stage.temps=6;
  abi.stage.vertex_inputs=abi.local_invocation_index_count=1;
  abi.stage.shareds=abi.stage.push_constant_start=8;
  abi.storage_buffer_descriptor_count=2;
  abi.storage_buffer_used_mask=3;
  abi.storage_buffer_read_mask=1;
  abi.storage_buffer_write_mask=2;
  Mutate(program,[&](const auto &p){ValidateComputeProgram(p,abi);},
      [&](const auto &p,std::size_t index){
        auto task=MakeComputeTask(abi,std::vector<std::uint32_t>(8),{1,1,1},{0,0,0},0,30);
        task.instruction_index=index;
        ComputeWorkgroupResult stats;
        try {StepComputeTask(p,abi,task,{},stats);}
        catch(...) {
          Check(task.steps==0&&task.instruction_index==index&&
                    stats.instructions_executed==0,"compute rejects before native work/counters");
          throw;
        }
      });
}

void Geometry() {
  const auto program=DecodeGeometryPcoProgram(GeometryNativeLoadFixture());
  const auto abi=GeometryNativeLoadAbi();
  Mutate(program,[&](const auto &p){ValidateGeometryProgram(p,abi);},
      [&](const auto &p,std::size_t index){
        auto task=MakeGeometryTask(abi,std::vector<std::uint32_t>(4),0,0);
        task.instruction_index=index;
        GeometryExecutionStats stats;
        try {StepGeometryTask(p,abi,task,{},stats);}
        catch(...) {
          Check(task.steps==0&&task.instruction_index==index&&stats.instructions==0,
                "geometry rejects before native work/counters");
          throw;
        }
      });
}

void Tessellation() {
  for(bool control:{false,true}) {
    const auto stage=control?ShaderStage::kTessellationControl:ShaderStage::kTessellationEvaluation;
    const auto program=DecodeTessellationPcoProgram(stage,
        control?kTessPatchFiveToTenTcs:kTessPatchFiveToTenTes);
    DriverPcoStageAbi abi;
    abi.temps=control?8:5;abi.vertex_inputs=control?3:5;
    abi.vertex_outputs=control?0:4;
    abi.shareds=abi.uniform_buffer_descriptor_start=abi.push_constant_start=control?8:4;
    Mutate(program,[&](const auto &p){ValidateTessellationProgram(p,abi);},
        [&](const auto &p,std::size_t index){
          const std::array<std::uint32_t,3> coord{};
          auto task=control?MakeTessellationControlTask(abi,std::vector<std::uint32_t>(8),0,5,10):
              MakeTessellationEvaluationTask(abi,std::vector<std::uint32_t>(4),0,10,&coord,1);
          task.instruction_index=index;
          TessellationExecutionStats stats;
          try {StepTessellationTask(p,abi,task,{},stats);}
          catch(...) {
            Check(task.steps==0&&task.instruction_index==index&&stats.instructions==0&&stats.groups==0,
                  "tessellation rejects before native work/counters");
            throw;
          }
        });
  }
}

void DecodeOlchk() {
  auto original=DecodeTessellationPcoProgram(ShaderStage::kTessellationControl,kTessPatchFiveToTenTcs);
  unsigned found=0;
  for(const auto &instruction:original.instructions) {
    if(instruction.opcode!=PcoOpcode::kIntegerMultiplyAdd64High)continue;
    ++found;
    for(bool is_signed:{false,true}) {
      auto bytes=kTessPatchFiveToTenTcs;
      bytes[instruction.binary_offset]=is_signed?0xeb:0xe3;
      // A legal signedness-only mutation still decodes; OLCHK must not be
      // silently dropped by the single-high TEMP destination decoder.
      DecodeTessellationPcoProgram(ShaderStage::kTessellationControl,bytes);
      bytes[instruction.binary_offset-2]|=0x08;
      try {DecodeTessellationPcoProgram(ShaderStage::kTessellationControl,bytes);}
      catch(const std::exception &error) {
        Check(std::string(error.what()).find("IMADD64 high requires")!=std::string::npos,
              "OLCHK rejected by its exact native header gate");
        continue;
      }
      throw std::runtime_error("IMADD64 OLCHK was silently discarded");
    }
  }
  Check(found==1,"true signed multiply-high fixture contains one target group");
}
} // namespace

int main() {
  try {
    Compute();Geometry();Tessellation();DecodeOlchk();
    for(auto opcode:{PcoOpcode::kMoveBypass,PcoOpcode::kMoveImmediate,
        PcoOpcode::kNop,PcoOpcode::kBufferLoad,PcoOpcode::kBufferStore,
        PcoOpcode::kWaitDataFence,PcoOpcode::kBranch,PcoOpcode::kConditionalMask,
        PcoOpcode::kUvsWrite,PcoOpcode::kUvsEmit,PcoOpcode::kUvsCut,
        PcoOpcode::kUvsEndTask,PcoOpcode::kUvsWriteEmitEndTask})
      Check(rejected_opcodes.count(opcode),"negative corpus covers native special opcode path");
    std::cout<<"native signed-metadata: "<<checks<<" checks PASS\n";
    return 0;
  } catch(const std::exception &error) {
    std::cerr<<"native signed-metadata: "<<error.what()<<" after "<<checks<<" checks\n";
    return 1;
  }
}
