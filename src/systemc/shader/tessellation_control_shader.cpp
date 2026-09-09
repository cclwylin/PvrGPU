#include "shader/tessellation_control_shader.h"

#include "common/tessellation_state.h"
#include "memory/gpu_memory_system.h"
#include "shader/tessellation_iss.h"
#include "shader/usc_uniform_buffer_memory.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pvrgpu::stub {
namespace {
bool Inside(std::uint64_t address,std::uint64_t bytes,std::uint64_t base,std::uint64_t size) {
  return address>=base && address-base<=size && bytes<=size-(address-base);
}
struct ControlMemory {
  GpuMemorySystem &memory;
  UscUniformBufferMemory &uniforms;
  CounterTxn &counters;
  std::uint64_t input_address,input_bytes,output_address,output_bytes;
  std::uint32_t level_written_mask=0;
  std::function<void(const PcoTextureRequest &, std::uint32_t *)> sample{};
  static void Sample(void *opaque, const PcoTextureRequest &request, std::uint32_t *response) {
    auto &self = *static_cast<ControlMemory *>(opaque);
    if (!self.sample) throw std::runtime_error("TCS has no texture route");
    self.sample(request, response);
  }
  static void Read(void *opaque,std::uint64_t address,std::uint32_t count,std::uint32_t *destination) {
    auto &self=*static_cast<ControlMemory*>(opaque);
    if(!destination || !count || count>16 || address%4)
      throw std::runtime_error("TCS LD alignment/count/destination is invalid");
    const auto bytes=count*sizeof(std::uint32_t);
    const bool input=Inside(address,bytes,self.input_address,self.input_bytes);
    const bool output=Inside(address,bytes,self.output_address,self.output_bytes);
    if(!input && !output) {
      UscUniformBufferMemory::Read(&self.uniforms,address,count,destination); return;
    }
    const auto read=self.memory.Read(address,bytes,MemoryClient::kTessellationControl);
    if(read.data.size()!=bytes)throw std::runtime_error("TCS LD completion size mismatch");
    std::memcpy(destination,read.data.data(),bytes);
    ApplyMemoryAccessStats(self.counters,read.stats);
    if(input)self.counters.tcs_input_read_bytes+=bytes;
    else self.counters.tcs_output_read_bytes+=bytes;
    WaitForCycles(MemoryAccessDelayCycles(read.stats));
  }
  static void Write(void *opaque,std::uint64_t address,std::uint32_t count,const std::uint32_t *source) {
    auto &self=*static_cast<ControlMemory*>(opaque);
    const auto bytes=count*sizeof(std::uint32_t);
    if(!source || !count || count>16 || address%4 ||
       !Inside(address,bytes,self.output_address,self.output_bytes))
      throw std::runtime_error("TCS ST exceeds its exact output patch range");
    const auto write=self.memory.Write(address,source,bytes,MemoryClient::kTessellationControl);
    ApplyMemoryAccessStats(self.counters,write);
    self.counters.tcs_output_write_bytes+=bytes;
    const auto first=(address-self.output_address)/4;
    for(unsigned c=0;c<count && first+c<6;++c)self.level_written_mask|=1U<<(first+c);
    WaitForCycles(MemoryAccessDelayCycles(write));
  }
};
}

TessellationControlShader::TessellationControlShader(sc_core::sc_module_name name,
    MemoryPool &pool,GpuMemorySystem *memory)
    : sc_module(name),pool_(pool),memory_(memory) { SC_THREAD(Run); }

void TessellationControlShader::Execute(PipelineState &state, const PipelineTxn &txn) {
  auto records=LoadArray<TessellationState>(pool_,state.tessellation_state);
  if(records.size()!=1 || state.stage!=PipelineStage::kVertexShaded || !memory_ ||
     memory_->mode()!=state.memory_mode)
    throw std::runtime_error("TCS pipeline/state/memory contract is invalid");
  auto &t=records[0];
  if (state.tessellation_control_sampled_texture_count > kPcoMaximumTextureDescriptorSets ||
      t.control_abi.uniform_buffer_descriptor_start != 8U + 20U * state.tessellation_control_sampled_texture_count)
    throw std::runtime_error("TCS texture count/descriptor prefix mismatch");
  if(t.phase!=TessellationPhase::kSubmitted || !t.input_address || t.input_address%4 ||
     !t.output_address || t.output_address%4 || !t.input_vertices || t.input_vertices>32 ||
     !t.output_vertices || t.output_vertices>32 || !t.input_stride_dwords ||
     t.input_stride_dwords>64 || t.output_vertex_stride_dwords>64 ||
     t.per_vertex_offset_dwords<6 || t.per_vertex_offset_dwords>t.patch_stride_dwords ||
     t.patch_stride_dwords!=t.per_vertex_offset_dwords+t.output_vertices*t.output_vertex_stride_dwords ||
     t.patch_stride_dwords>kTessellationPatchAddressStride/4 ||
     t.input_address>UINT64_MAX-8192 ||
     t.output_address>UINT64_MAX-std::uint64_t{kTessellationMaxPatches}*kTessellationPatchAddressStride ||
     HasPoolHandle(t.control_instructions))
    throw std::runtime_error("TCS phase/patch/address/layout contract is invalid");
  const auto program=DecodeTessellationPcoProgram(ShaderStage::kTessellationControl,
      LoadArray<std::uint8_t>(pool_,t.control_code));
  ValidateTessellationProgram(program,t.control_abi);
  t.control_summary=program.summary;
  t.control_instructions=StoreNewArray(pool_,program.instructions);
  StoreArray(pool_,state.tessellation_state,records);
  auto shared=LoadArray<std::uint32_t>(pool_,t.control_shared);
  if(shared.size()!=t.control_abi.shareds || shared.size()<8)
    throw std::runtime_error("TCS shared register snapshot size mismatch");
  const auto patches=LoadArray<TessellationPatch>(pool_,t.patches);
  const auto lanes=LoadArray<VertexLane>(pool_,state.vertex_lanes);
  const auto refs=LoadArray<VertexLaneRef>(pool_,state.vertex_lane_refs);
  if(patches.size()>kTessellationMaxPatches)
    throw std::runtime_error("TCS patch count exceeds bounded draw storage");
  UscUniformBufferMemory uniforms(memory_,state.memory_mode,
      HasPoolHandle(t.control_uniform_buffers)?LoadArray<UniformBufferResource>(pool_,t.control_uniform_buffers):
                                              std::vector<UniformBufferResource>{});
  TessellationExecutionStats execution;
  const auto task_handle=pool_.Allocate(sizeof(TessellationTaskState));
  try {
    for(const auto &patch:patches) {
      if(patch.input_vertices!=t.input_vertices || patch.first_occurrence>refs.size() ||
         patch.input_vertices>refs.size()-patch.first_occurrence || patch.output_address<t.output_address ||
         patch.output_address-t.output_address>=std::uint64_t{kTessellationMaxPatches}*kTessellationPatchAddressStride ||
         (patch.output_address-t.output_address)%kTessellationPatchAddressStride ||
         patch.point_count || patch.index_count || patch.domain_address)
        throw std::runtime_error("TCS patch occurrence/output address contract is invalid");
      const auto input_words=patch.input_vertices*t.input_stride_dwords;
      std::array<std::uint32_t,32*64> input{};
      for(unsigned vertex=0;vertex<patch.input_vertices;++vertex) {
        const auto lane_index=refs[patch.first_occurrence+vertex].lane_index;
        if(lane_index>=lanes.size() || !lanes[lane_index].emitted || !lanes[lane_index].ended)
          throw std::runtime_error("TCS patch references an unfinished VS invocation");
        std::copy_n(lanes[lane_index].vertex_output,t.input_stride_dwords,
                    input.data()+vertex*t.input_stride_dwords);
      }
      const auto write=memory_->Write(t.input_address,input.data(),input_words*4,
                                      MemoryClient::kTessellationControl);
      ApplyMemoryAccessStats(state.counters,write);
      state.counters.tcs_input_write_bytes+=input_words*4;
      WaitForCycles(MemoryAccessDelayCycles(write));
      shared[0]=static_cast<std::uint32_t>(t.input_address);
      shared[1]=static_cast<std::uint32_t>(t.input_address>>32U);shared[2]=input_words*4;shared[3]=0;
      shared[4]=static_cast<std::uint32_t>(patch.output_address);
      shared[5]=static_cast<std::uint32_t>(patch.output_address>>32U);shared[6]=t.patch_stride_dwords*4;shared[7]=0;
      auto &task=*reinterpret_cast<TessellationTaskState*>(pool_.Write(task_handle).data());
      task=MakeTessellationControlTask(t.control_abi,shared,patch.primitive_id,patch.input_vertices,t.output_vertices);
      ControlMemory context{*memory_,uniforms,state.counters,t.input_address,input_words*4,
                             patch.output_address,t.patch_stride_dwords*4};
      context.sample = [this, &state, &txn](const PcoTextureRequest &request, std::uint32_t *response) {
        SampleTessellationTexture(pool_, state, txn, ShaderStage::kTessellationControl, request,
                                  response, texture_request_output, texture_response_input);
      };
      const TessellationMemoryCallbacks callbacks{&context,ControlMemory::Read,ControlMemory::Write,ControlMemory::Sample};
      while(!task.ended)StepTessellationTask(program,t.control_abi,task,callbacks,execution);
      const auto required=t.domain==TessellationDomain::kQuads?63U:
                          t.domain==TessellationDomain::kTriangles?23U:3U;
      if((context.level_written_mask&required)!=required)
        throw std::runtime_error("TCS ended without writing all used tessellation levels");
      ++state.counters.hs_invocations;
      state.counters.tcs_invocations+=task.lane_count;
    }
  } catch(...) { pool_.Release(task_handle);throw; }
  pool_.Release(task_handle);
  ApplyMemoryAccessStats(state.counters,uniforms.stats());
  WaitForCycles(MemoryAccessDelayCycles(uniforms.stats()));
  state.counters.pco_decode_cycles+=program.summary.group_count;
  state.counters.pco_instructions+=program.summary.instruction_count;
  state.counters.usc_groups+=execution.groups;
  state.counters.usc_cluster_cycles+=execution.groups;
  state.counters.tcs_alu_instructions+=execution.alu_instructions;
  state.counters.tcs_tex_instructions+=execution.texture_instructions;
  state.counters.tcs_memory_instructions+=execution.memory_instructions;
  state.counters.tcs_load_instructions+=execution.load_instructions;
  state.counters.tcs_store_instructions+=execution.store_instructions;
  if(HasPoolHandle(state.drawlist_stats)) {
    auto stats=LoadArray<DrawListStats>(pool_,state.drawlist_stats);
    if(stats.size()!=1)throw std::runtime_error("TCS requires one DrawList statistics record");
    auto &s=stats[0].tessellation_control;
    const auto composition=CountPcoInstructions(program.instructions,false);
    s.invocations=state.counters.tcs_invocations;
    s.program_groups=program.summary.group_count;s.program_instructions=program.summary.instruction_count;
    s.program_alu_instructions=composition.alu;s.program_memory_instructions=composition.memory;
    s.program_tex_instructions=composition.texture;s.executed_tex_instructions=execution.texture_instructions;
    s.executed_alu_instructions=execution.alu_instructions;s.executed_memory_instructions=execution.memory_instructions;
    s.program_recorded=s.executions_recorded=1;
    StoreArray(pool_,state.drawlist_stats,stats);
  }
  t.phase=TessellationPhase::kControlComplete;
  StoreArray(pool_,state.tessellation_state,records);
  WaitForCycles(program.summary.group_count+execution.groups);
}

void TessellationControlShader::Run() {
  for(;;) {
    PipelineTxn transaction;
    while(!input.nb_read(transaction))wait(input.data_written_event());
    auto state=LoadPipelineState(pool_,transaction.state);
    if(HasPoolHandle(state.tessellation_state)) {
      try { Execute(state, transaction); }
      catch(...) { StorePipelineState(pool_,transaction.state,state);throw; }
      StorePipelineState(pool_,transaction.state,state);
    }
    while(!output.nb_write(transaction))wait(output.data_read_event());
  }
}

}  // namespace pvrgpu::stub
