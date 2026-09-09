#include "shader/tessellation_evaluation_shader.h"

#include "common/geometry_emission.h"
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
class OwnedPayload {
 public:
  OwnedPayload(MemoryPool &pool,std::size_t bytes):pool_(pool),handle_(pool.Allocate(bytes)) {}
  ~OwnedPayload(){if(HasPoolHandle(handle_))pool_.Release(handle_);}
  OwnedPayload(const OwnedPayload&)=delete;
  OwnedPayload &operator=(const OwnedPayload&)=delete;
  template<class T>T *data(){return reinterpret_cast<T*>(pool_.Write(handle_).data());}
  PoolHandle Publish(){auto h=handle_;handle_={};return h;}
 private:
  MemoryPool &pool_;
  PoolHandle handle_;
};
struct EvaluationMemory {
  GpuMemorySystem &memory;
  UscUniformBufferMemory &uniforms;
  CounterTxn &counters;
  std::uint64_t patch_address,patch_bytes;
  std::function<void(const PcoTextureRequest &, std::uint32_t *)> sample{};
  static void Sample(void *opaque, const PcoTextureRequest &request, std::uint32_t *response) {
    auto &self = *static_cast<EvaluationMemory *>(opaque);
    if (!self.sample) throw std::runtime_error("TES has no texture route");
    self.sample(request, response);
  }
  static void Read(void *opaque,std::uint64_t address,std::uint32_t count,std::uint32_t *destination) {
    auto &self=*static_cast<EvaluationMemory*>(opaque);
    if(!destination || !count || count>16 || address%4)
      throw std::runtime_error("TES LD destination/count/alignment is invalid");
    const auto bytes=count*sizeof(std::uint32_t);
    if(address<self.patch_address || address-self.patch_address>self.patch_bytes ||
       bytes>self.patch_bytes-(address-self.patch_address)) {
      UscUniformBufferMemory::Read(&self.uniforms,address,count,destination);return;
    }
    const auto read=self.memory.Read(address,bytes,MemoryClient::kTessellationEvaluation);
    if(read.data.size()!=bytes)throw std::runtime_error("TES patch LD completion size mismatch");
    std::memcpy(destination,read.data.data(),bytes);
    ApplyMemoryAccessStats(self.counters,read.stats);
    self.counters.tes_patch_read_bytes+=bytes;
    WaitForCycles(MemoryAccessDelayCycles(read.stats));
  }
};
std::uint32_t Bits(float value){std::uint32_t bits;std::memcpy(&bits,&value,4);return bits;}
}

TessellationEvaluationShader::TessellationEvaluationShader(sc_core::sc_module_name name,
    MemoryPool &pool,GpuMemorySystem *memory)
    : sc_module(name),pool_(pool),memory_(memory){SC_THREAD(Run);}

void TessellationEvaluationShader::Execute(PipelineState &state, const PipelineTxn &txn) {
  auto records=LoadArray<TessellationState>(pool_,state.tessellation_state);
  if(records.size()!=1 || state.stage!=PipelineStage::kVertexShaded || !memory_ ||
     memory_->mode()!=state.memory_mode)
    throw std::runtime_error("TES pipeline/state/memory contract is invalid");
  auto &t=records[0];
  if (state.tessellation_evaluation_sampled_texture_count > kPcoMaximumTextureDescriptorSets ||
      t.evaluation_abi.uniform_buffer_descriptor_start != 4U + 20U * state.tessellation_evaluation_sampled_texture_count)
    throw std::runtime_error("TES texture count/descriptor prefix mismatch");
  if(t.phase!=TessellationPhase::kDomainComplete || !t.output_address || t.output_address%4 ||
     !t.domain_address || t.domain_address%8 || !t.output_vertices || t.output_vertices>32 ||
     t.patch_stride_dwords<6 || t.patch_stride_dwords>kTessellationPatchAddressStride/4 ||
     state.position_output_count!=4 || state.position_output_start>60 ||
     t.domain_address>UINT64_MAX-kTessellationMaxDrawPoints*sizeof(TessellationDomainPoint) ||
     t.output_address>UINT64_MAX-std::uint64_t{kTessellationMaxPatches}*kTessellationPatchAddressStride ||
     HasPoolHandle(t.evaluation_instructions) || HasPoolHandle(state.geometry_code) ||
     HasPoolHandle(state.geometry_primitives))
    throw std::runtime_error("TES phase/address/register/ownership contract is invalid");
  const auto program=DecodeTessellationPcoProgram(ShaderStage::kTessellationEvaluation,
      LoadArray<std::uint8_t>(pool_,t.evaluation_code));
  ValidateTessellationProgram(program,t.evaluation_abi);
  t.evaluation_summary=program.summary;
  t.evaluation_instructions=StoreNewArray(pool_,program.instructions);
  StoreArray(pool_,state.tessellation_state,records);
  auto shared=LoadArray<std::uint32_t>(pool_,t.evaluation_shared);
  if(shared.size()!=t.evaluation_abi.shareds || shared.size()<4)
    throw std::runtime_error("TES shared register snapshot size mismatch");
  const auto patches=LoadArray<TessellationPatch>(pool_,t.patches);
  const auto indices=LoadArray<std::uint32_t>(pool_,t.domain_indices);
  const auto point_bytes=pool_.Read(t.domain_points).size();
  if(point_bytes%sizeof(TessellationDomainPoint) || patches.size()>kTessellationMaxPatches ||
     point_bytes/sizeof(TessellationDomainPoint)>kTessellationMaxDrawPoints ||
     indices.size()>kTessellationMaxDrawIndices)
    throw std::runtime_error("TES domain storage exceeds draw bound");
  const auto point_count=point_bytes/sizeof(TessellationDomainPoint);
  const auto primitive_size=t.point_mode?1U:t.domain==TessellationDomain::kIsolines?2U:3U;
  const auto required_mask=UINT64_C(15)<<state.position_output_start;
  UscUniformBufferMemory uniforms(memory_,state.memory_mode,
      HasPoolHandle(t.evaluation_uniform_buffers)?LoadArray<UniformBufferResource>(pool_,t.evaluation_uniform_buffers):
                                                 std::vector<UniformBufferResource>{});
  OwnedPayload task_payload(pool_,sizeof(TessellationTaskState));
  std::vector<VertexLane> lanes;
  std::vector<VertexLaneRef> refs;
  std::vector<GeometryRasterPrimitive> primitives;
  lanes.reserve(point_count);refs.reserve(indices.size()/primitive_size*3);
  primitives.reserve(indices.size()/primitive_size);
  TessellationExecutionStats execution;
  std::size_t consumed_indices=0;
  for(const auto &patch:patches) {
    if(patch.point_start!=lanes.size() || patch.point_start>point_count ||
       patch.point_count>point_count-patch.point_start || patch.index_start!=consumed_indices ||
       patch.index_start>indices.size() || patch.index_count>indices.size()-patch.index_start ||
       patch.index_count%primitive_size || patch.primitive_size!=primitive_size ||
       patch.output_address<t.output_address ||
       patch.output_address-t.output_address>=std::uint64_t{kTessellationMaxPatches}*kTessellationPatchAddressStride ||
       (patch.output_address-t.output_address)%kTessellationPatchAddressStride ||
       patch.domain_address!=t.domain_address+std::uint64_t{patch.point_start}*sizeof(TessellationDomainPoint))
      throw std::runtime_error("TES patch domain/primitive/address spans are invalid");
    shared[0]=static_cast<std::uint32_t>(patch.output_address);
    shared[1]=static_cast<std::uint32_t>(patch.output_address>>32U);
    shared[2]=t.patch_stride_dwords*4;shared[3]=0;
    EvaluationMemory context{*memory_,uniforms,state.counters,patch.output_address,t.patch_stride_dwords*4};
    context.sample = [this, &state, &txn](const PcoTextureRequest &request, std::uint32_t *response) {
      SampleTessellationTexture(pool_, state, txn, ShaderStage::kTessellationEvaluation, request,
                                response, texture_request_output, texture_response_input);
    };
    const TessellationMemoryCallbacks callbacks{&context,EvaluationMemory::Read,nullptr,EvaluationMemory::Sample};
    for(unsigned first=0;first<patch.point_count;first+=kTessellationTaskWidth) {
      const auto count=std::min(kTessellationTaskWidth,patch.point_count-first);
      const auto bytes=count*sizeof(TessellationDomainPoint);
      const auto read=memory_->Read(patch.domain_address+first*sizeof(TessellationDomainPoint),bytes,
                                    MemoryClient::kTessellationEvaluation);
      if(read.data.size()!=bytes)throw std::runtime_error("TES domain coordinate read completion size mismatch");
      ApplyMemoryAccessStats(state.counters,read.stats);
      state.counters.tessellation_domain_read_bytes+=bytes;
      WaitForCycles(MemoryAccessDelayCycles(read.stats));
      std::array<std::array<std::uint32_t,3>,32> coordinates{};
      for(unsigned lane=0;lane<count;++lane) {
        TessellationDomainPoint point;
        std::memcpy(&point,read.data.data()+lane*sizeof(point),sizeof(point));
        coordinates[lane]={Bits(point.u),Bits(point.v),Bits(TessellationCoordinateW(t.domain,point))};
      }
      auto &task=*task_payload.data<TessellationTaskState>();
      task=MakeTessellationEvaluationTask(t.evaluation_abi,shared,patch.primitive_id,t.output_vertices,
                                          coordinates.data(),count);
      while(!task.ended)StepTessellationTask(program,t.evaluation_abi,task,callbacks,execution);
      for(unsigned index=0;index<count;++index) {
        const auto &result=task.lanes[index];
        if(!result.emitted || (result.outputs_written&required_mask)!=required_mask)
          throw std::runtime_error("TES vertex has no complete native position emission");
        VertexLane lane;
        std::copy_n(result.outputs.data(),t.evaluation_abi.vertex_outputs,lane.vertex_output);
        lane.emitted=lane.ended=1;lanes.push_back(lane);
      }
      state.counters.ds_invocations+=count;
    }
    for(unsigned index=0;index<patch.index_count;index+=primitive_size) {
      GeometryRasterPrimitive primitive;
      primitive.refs.vertex_count=static_cast<std::uint8_t>(primitive_size);
      primitive.refs.provoking_vertex=static_cast<std::uint8_t>(primitive_size-1);
      primitive.input_primitive_id=patch.primitive_id;primitive.instance_id=patch.instance_id;
      for(unsigned c=0;c<3;++c) {
        const auto local=indices[patch.index_start+index+std::min(c,primitive_size-1)];
        if(local>=patch.point_count)throw std::runtime_error("TES generated index exceeds its patch point range");
        const auto vertex=patch.point_start+local;
        primitive.refs.vertex_indices[c]=vertex;refs.push_back({vertex,vertex});
      }
      primitives.push_back(primitive);
    }
    consumed_indices+=patch.index_count;
  }
  if(lanes.size()!=point_count || consumed_indices!=indices.size())
    throw std::runtime_error("TES patch descriptors do not own the entire domain storage");
  ApplyMemoryAccessStats(state.counters,uniforms.stats());
  WaitForCycles(MemoryAccessDelayCycles(uniforms.stats()));
  state.counters.pco_decode_cycles+=program.summary.group_count;
  state.counters.pco_instructions+=program.summary.instruction_count;
  state.counters.usc_groups+=execution.groups;state.counters.usc_cluster_cycles+=execution.groups;
  state.counters.tes_alu_instructions+=execution.alu_instructions;
  state.counters.tes_tex_instructions+=execution.texture_instructions;
  state.counters.tes_memory_instructions+=execution.memory_instructions;
  state.counters.tes_load_instructions+=execution.load_instructions;
  if(HasPoolHandle(state.drawlist_stats)) {
    auto stats=LoadArray<DrawListStats>(pool_,state.drawlist_stats);
    if(stats.size()!=1)throw std::runtime_error("TES requires one DrawList statistics record");
    auto &s=stats[0].tessellation_evaluation;
    const auto composition=CountPcoInstructions(program.instructions,false);
    s.invocations=state.counters.ds_invocations;
    s.program_groups=program.summary.group_count;s.program_instructions=program.summary.instruction_count;
    s.program_alu_instructions=composition.alu;s.program_memory_instructions=composition.memory;
    s.program_tex_instructions=composition.texture;s.executed_tex_instructions=execution.texture_instructions;
    s.executed_alu_instructions=execution.alu_instructions;s.executed_memory_instructions=execution.memory_instructions;
    s.program_recorded=s.executions_recorded=1;StoreArray(pool_,state.drawlist_stats,stats);
  }
  OwnedPayload new_lanes(pool_,lanes.size()*sizeof(VertexLane));
  OwnedPayload new_refs(pool_,refs.size()*sizeof(VertexLaneRef));
  OwnedPayload new_primitives(pool_,primitives.size()*sizeof(GeometryRasterPrimitive));
  if(!lanes.empty())std::memcpy(new_lanes.data<VertexLane>(),lanes.data(),lanes.size()*sizeof(VertexLane));
  if(!refs.empty())std::memcpy(new_refs.data<VertexLaneRef>(),refs.data(),refs.size()*sizeof(VertexLaneRef));
  if(!primitives.empty())std::memcpy(new_primitives.data<GeometryRasterPrimitive>(),primitives.data(),primitives.size()*sizeof(GeometryRasterPrimitive));
  pool_.Release(state.vertex_lanes);pool_.Release(state.vertex_lane_refs);
  state.vertex_lanes=new_lanes.Publish();state.vertex_lane_refs=new_refs.Publish();
  state.geometry_primitives=new_primitives.Publish();
  for(auto *handle:{&state.vertex_indices,&state.expanded_source_vertices}) {
    if(HasPoolHandle(*handle))pool_.Release(*handle);*handle={};
  }
  state.source_topology=t.point_mode?PrimitiveTopology::kPoints:
      t.domain==TessellationDomain::kIsolines?PrimitiveTopology::kLines:PrimitiveTopology::kTriangleList;
  state.draw.topology=PrimitiveTopology::kTriangleList;state.draw.vertex_count=static_cast<std::uint32_t>(refs.size());
  state.draw.first_vertex=state.draw.first_index=state.draw.index_count=0;
  state.draw.base_vertex=0;state.draw.index_format=IndexFormat::kNone;
  state.index_buffer_gpu_address=state.index_buffer_bytes=0;
  t.phase=TessellationPhase::kEvaluationComplete;StoreArray(pool_,state.tessellation_state,records);
  WaitForCycles(program.summary.group_count+execution.groups);
}

void TessellationEvaluationShader::Run() {
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
