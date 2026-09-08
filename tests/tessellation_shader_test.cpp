// Native TCS -> fixed tessellation -> native TES, bounded depth-one channels.
#include "common/geometry_emission.h"
#include "common/tessellation_state.h"
#include "common/stream_output_types.h"
#include "geometry/stream_output.h"
#include "geometry/tessellator.h"
#include "memory/gpu_memory_system.h"
#include "pco_tessellation_compiler_fixtures.h"
#include "shader/tessellation_control_shader.h"
#include "shader/tessellation_evaluation_shader.h"
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>

using namespace pvrgpu::stub;
namespace {
unsigned checks=0;
void Check(bool value,const char *message){++checks;if(!value)throw std::runtime_error(message);}
std::uint32_t Bits(float value){std::uint32_t bits;std::memcpy(&bits,&value,4);return bits;}
float Float(std::uint32_t bits){float value;std::memcpy(&value,&bits,4);return value;}
float Input(unsigned sequence,unsigned patch,unsigned vertex,unsigned component) {
  return float(sequence+patch*5+vertex*4+component+1)/64;
}
PipelineTxn Make(MemoryPool &pool,GpuMemorySystem &memory,unsigned sequence) {
  const unsigned kind=(sequence-1)%3,vertices=kind?3:1;
  PipelineState state;
  state.stage=PipelineStage::kVertexShaded;state.memory_mode=memory.mode();
  state.position_output_count=4;state.position_output_start=0;
  state.counters.vs_invocations=2*vertices;
  TessellationState t;
  const std::vector<std::uint8_t>*controls[]={&kTess0tcs,&kTess1tcs,&kTess2tcs};
  const std::vector<std::uint8_t>*evaluations[]={&kTess0tes,&kTess1tes,&kTess2tes};
  const unsigned tcs_temps[]={6,13,12},tes_temps[]={2,11,20};
  t.control_code=StoreNewArray(pool,*controls[kind]);t.evaluation_code=StoreNewArray(pool,*evaluations[kind]);
  t.control_abi.temps=tcs_temps[kind];t.control_abi.vertex_inputs=3;t.control_abi.shareds=8;
  t.control_abi.uniform_buffer_descriptor_start=t.control_abi.push_constant_start=8;
  t.evaluation_abi.temps=tes_temps[kind];t.evaluation_abi.vertex_inputs=5;t.evaluation_abi.shareds=4;
  t.evaluation_abi.vertex_outputs=kind==2?8:4;
  t.evaluation_abi.uniform_buffer_descriptor_start=t.evaluation_abi.push_constant_start=4;
  t.control_shared=StoreNewArray(pool,std::vector<std::uint32_t>(8));
  t.evaluation_shared=StoreNewArray(pool,std::vector<std::uint32_t>(4));
  t.input_vertices=t.output_vertices=vertices;t.vertices_per_instance=vertices;
  t.input_stride_dwords=4;t.output_vertex_stride_dwords=kind==2?8:4;
  t.per_vertex_offset_dwords=kind==2?10:6;
  t.patch_stride_dwords=t.per_vertex_offset_dwords+vertices*t.output_vertex_stride_dwords;
  t.control_barrier_count=kind==2;
  t.input_address=UINT64_C(0x400000000)+sequence*kTessellationDrawAddressStride;
  t.output_address=t.input_address+kTessellationPatchAddressStride;
  t.domain_address=t.output_address+UINT64_C(0x4000000);
  std::vector<TessellationPatch> patches(2);
  std::vector<VertexLane> lanes(vertices*2);
  std::vector<VertexLaneRef> refs(vertices*2);
  for(unsigned patch=0;patch<2;++patch) {
    patches[patch].input_vertices=vertices;patches[patch].first_occurrence=patch*vertices;
    patches[patch].primitive_id=17+patch;patches[patch].instance_id=patch;
    patches[patch].output_address=t.output_address+patch*kTessellationPatchAddressStride;
    for(unsigned vertex=0;vertex<vertices;++vertex) {
      // Deliberately permute physical lanes: the patch consumes occurrence
      // references, not an assumed contiguous vertex-lane array.
      const auto lane_index=vertices*2-1-(patch*vertices+vertex);
      for(unsigned c=0;c<4;++c)lanes[lane_index].vertex_output[c]=Bits(Input(sequence,patch,vertex,c));
      lanes[lane_index].emitted=lanes[lane_index].ended=1;
      refs[patch*vertices+vertex]={lane_index,100+patch*vertices+vertex};
    }
  }
  t.patches=StoreNewArray(pool,patches);
  state.vertex_lanes=StoreNewArray(pool,lanes);state.vertex_lane_refs=StoreNewArray(pool,refs);
  state.drawlist_stats=StoreNewArray(pool,std::vector<DrawListStats>(1));
  state.tessellation_output_dwords=t.evaluation_abi.vertex_outputs;
  state.tessellation_state=StoreNewArray(pool,std::vector<TessellationState>{t});
  // Alternate unchanged raster-only draws with complete TES -> SO draws.
  // The last fixture captures the TES-only color export at DWORD4..7; the
  // preceding VS exports only position, so using its ABI would reject this.
  state.vertex_pco_abi.vertex_outputs=4;
  if(sequence%2==0) {
    StreamOutputTarget target;
    target.output_buffer=0;target.resource_token=sequence;target.target_token=sequence+100;
    target.gpu_address=UINT64_C(0x700000000)+sequence*0x10000;
    target.bytes_size=16384;target.buffer_offset=16;
    target.buffer_size=kind==0?16352:160;target.internal_offset=kind==0?0:16;
    target.stride_dwords=kind==2?6:4;
    const std::vector<std::uint8_t> initial(target.bytes_size,0xa5);
    memory.HostWrite(target.gpu_address,initial.data(),initial.size());
    state.stream_output_bindings=StoreNewArray(pool,std::vector<StreamOutputBinding>{
        {kind==2?4U:0U,4,0,kind==2?1U:0U,0}});
    state.stream_output_targets=StoreNewArray(pool,std::vector<StreamOutputTarget>{target});
  }
  PipelineTxn txn;txn.sequence=sequence;txn.state=pool.Allocate(sizeof(PipelineState));
  StorePipelineState(pool,txn.state,state);return txn;
}
void Release(MemoryPool &pool,PipelineTxn txn) {
  const auto state=LoadPipelineState(pool,txn.state);
  ReleaseFunctionalPayloads(pool,state);pool.Release(txn.state);
}
void Verify(MemoryPool &pool,GpuMemorySystem &memory,PipelineTxn txn) {
  const auto state=LoadPipelineState(pool,txn.state);
  const auto t=LoadArray<TessellationState>(pool,state.tessellation_state).at(0);
  const auto patches=LoadArray<TessellationPatch>(pool,t.patches);
  const auto points=LoadArray<TessellationDomainPoint>(pool,t.domain_points);
  const auto indices=LoadArray<std::uint32_t>(pool,t.domain_indices);
  const auto lanes=LoadArray<VertexLane>(pool,state.vertex_lanes);
  const auto refs=LoadArray<VertexLaneRef>(pool,state.vertex_lane_refs);
  const auto primitives=LoadArray<GeometryRasterPrimitive>(pool,state.geometry_primitives);
  const auto stats=LoadArray<DrawListStats>(pool,state.drawlist_stats).at(0);
  const auto kind=(txn.sequence-1)%3;
  Check(t.phase==TessellationPhase::kEvaluationComplete && state.stage==PipelineStage::kVertexShaded,"three native stages completed without aliasing GS/VS");
  Check(!state.counters.gs_invocations && !state.counters.gs_primitives,"TCS/TES never run GS");
  Check(state.counters.hs_invocations==2 && state.counters.tcs_invocations==2*t.output_vertices,"patch vs TCS-lane invocation accounting");
  Check(state.counters.ds_invocations==points.size() && lanes.size()==points.size(),"TES executes each actual generated domain point");
  Check(state.counters.tessellation_patches==2 && state.counters.tessellation_primitives==primitives.size(),"fixed primitive count agrees with emitted connectivity");
  Check(state.counters.tcs_input_write_bytes==2*t.input_vertices*t.input_stride_dwords*4,"actual modeled VS patch staging bytes");
  Check(state.counters.tcs_output_write_bytes>=48 && state.counters.tcs_store_instructions>=12,"actual native TCS level stores");
  Check(state.counters.tessellation_level_read_bytes==48,"fixed module reads actual six-level patch bytes");
  Check(state.counters.tessellation_domain_write_bytes==points.size()*8 &&
        state.counters.tessellation_domain_read_bytes==points.size()*8,"domain values traverse modeled write/read, not pool bypass");
  Check(stats.tessellation_control.invocations==state.counters.tcs_invocations &&
        stats.tessellation_evaluation.invocations==state.counters.ds_invocations,"DrawList stage-local invocation counts");
  Check(stats.tessellation_control.executed_memory_instructions==state.counters.tcs_memory_instructions &&
        stats.tessellation_evaluation.executed_alu_instructions==state.counters.tes_alu_instructions,"DrawList native instruction totals");
  Check(stats.tessellation_control.program_recorded && stats.tessellation_evaluation.program_recorded,"native program provenance recorded");
  if(kind==0)Check(!state.counters.tcs_input_read_bytes && !state.counters.tes_patch_read_bytes,"constant TCS/TES do not invent input reads");
  else Check(state.counters.tcs_input_read_bytes && state.counters.tes_patch_read_bytes,"real per-vertex LD path executed");
  if(kind==2)Check(state.counters.tcs_output_read_bytes,"cross-invocation barrier reads actual completed output stores");
  for(unsigned p=0;p<patches.size();++p) {
    const auto &patch=patches[p];
    const auto read=memory.Read(patch.output_address,24,MemoryClient::kTessellationEvaluation);
    for(unsigned c=0;c<6;++c){std::uint32_t word;std::memcpy(&word,read.data.data()+c*4,4);Check(word==Bits(5),"six actual native TCS levels visible through memory");}
    for(unsigned point=0;point<patch.point_count;++point) {
      const auto index=patch.point_start+point;
      const auto coord=points[index];
      for(unsigned c=0;c<4;++c) {
        const float expected=kind?std::fma(Input(txn.sequence,p,2,c),TessellationCoordinateW(t.domain,coord),
             std::fma(Input(txn.sequence,p,1,c),coord.v,Input(txn.sequence,p,0,c)*coord.u)):
             c<2?std::fma(c==0?coord.u:coord.v,1.6f,-.8f):c==2?0:1;
        Check(lanes[index].ended&&lanes[index].emitted&&std::abs(Float(lanes[index].vertex_output[c])-expected)<1e-6f,"native TES interpolates actual input patch and domain");
        if(kind==2)Check(lanes[index].vertex_output[4+c]==Bits(float(c+1)*.25f+Input(txn.sequence,p,1,c)),"TES native perpatch/crossvertex color transport");
      }
    }
    for(unsigned i=0;i<patch.index_count/3;++i) {
      const auto primitive_index=patch.index_start/3+i;
      const auto &primitive=primitives[primitive_index];
      Check(primitive.input_primitive_id==patch.primitive_id && primitive.instance_id==patch.instance_id &&
            primitive.refs.vertex_count==3 && primitive.refs.provoking_vertex==2,"patch identity and last provoking-vertex connectivity preserved");
      for(unsigned c=0;c<3;++c)Check(primitive.refs.vertex_indices[c]==patch.point_start+indices[patch.index_start+i*3+c] &&
          refs[primitive_index*3+c].lane_index==primitive.refs.vertex_indices[c],"generated raster references map exact patch-local domain indices");
    }
  }
  if(txn.sequence%2==0) {
    const auto target=LoadArray<StreamOutputTarget>(pool,state.stream_output_targets).at(0);
    const auto binding=LoadArray<StreamOutputBinding>(pool,state.stream_output_bindings).at(0);
    const auto initial_cursor=kind==0?0U:16U;
    const auto admitted=std::min<std::size_t>(primitives.size(),
        (target.buffer_size-initial_cursor)/(3*target.stride_dwords*4));
    Check(state.stream_output_complete && state.stream_output_primitives_written==admitted &&
          state.stream_output_primitives_storage_needed==primitives.size(),
          "TES feedback queries count completed primitives, not distinct domain points");
    Check(target.internal_offset==initial_cursor+admitted*3*target.stride_dwords*4,
          "TES feedback append cursor advances only whole admitted primitives");
    std::vector<std::uint8_t> expected(target.bytes_size,0xa5);
    for(unsigned primitive=0;primitive<admitted;++primitive)for(unsigned c=0;c<3;++c) {
      const auto vertex=primitives[primitive].refs.vertex_indices[c];
      const auto offset=target.buffer_offset+initial_cursor+
          (primitive*3+c)*target.stride_dwords*4+binding.dst_offset_dwords*4;
      std::memcpy(expected.data()+offset,lanes[vertex].vertex_output+binding.output_dword,16);
    }
    const auto actual=LoadArray<std::uint8_t>(pool,target.readback);
    Check(actual==expected && memory.backing().Read(target.gpu_address,target.bytes_size)==expected,
          "real TES raw export memory preserves primitive order, holes, append prefix and overflow guards");
  } else Check(!state.stream_output_complete,"draws without feedback pass through unchanged");
  Release(pool,txn);
}
} // namespace
int sc_main(int argc,char **argv) {
  try {
    MemoryPool pool;
    std::array<std::unique_ptr<GpuMemorySystem>,3> memories;
    std::array<std::unique_ptr<sc_core::sc_fifo<PipelineTxn>>,3> input,tc,te,so,output;
    std::array<std::unique_ptr<TessellationControlShader>,3> controls;
    std::array<std::unique_ptr<Tessellator>,3> fixed;
    std::array<std::unique_ptr<TessellationEvaluationShader>,3> evaluations;
    std::array<std::unique_ptr<StreamOutput>,3> feedback;
    for(unsigned mode=0;mode<3;++mode) {
      memories[mode]=std::make_unique<GpuMemorySystem>(static_cast<MemoryMode>(mode));
      input[mode]=std::make_unique<sc_core::sc_fifo<PipelineTxn>>(sc_core::sc_gen_unique_name("input"),1);
      tc[mode]=std::make_unique<sc_core::sc_fifo<PipelineTxn>>(sc_core::sc_gen_unique_name("tc"),1);
      te[mode]=std::make_unique<sc_core::sc_fifo<PipelineTxn>>(sc_core::sc_gen_unique_name("te"),1);
      so[mode]=std::make_unique<sc_core::sc_fifo<PipelineTxn>>(sc_core::sc_gen_unique_name("so"),1);
      output[mode]=std::make_unique<sc_core::sc_fifo<PipelineTxn>>(sc_core::sc_gen_unique_name("output"),1);
      controls[mode]=std::make_unique<TessellationControlShader>(sc_core::sc_gen_unique_name("tcs"),pool,memories[mode].get());
      fixed[mode]=std::make_unique<Tessellator>(sc_core::sc_gen_unique_name("fixed"),pool,memories[mode].get());
      evaluations[mode]=std::make_unique<TessellationEvaluationShader>(sc_core::sc_gen_unique_name("tes"),pool,memories[mode].get());
      feedback[mode]=std::make_unique<StreamOutput>(sc_core::sc_gen_unique_name("stream_output"),pool,memories[mode].get());
      controls[mode]->input(*input[mode]);controls[mode]->output(*tc[mode]);
      fixed[mode]->input(*tc[mode]);fixed[mode]->output(*te[mode]);
      evaluations[mode]->input(*te[mode]);evaluations[mode]->output(*so[mode]);
      feedback[mode]->input(*so[mode]);feedback[mode]->output(*output[mode]);
    }
    if(argc==2) {
      const auto invalid=static_cast<unsigned>(std::atoi(argv[1]));
      Check(invalid>=1&&invalid<=8,"invalid rejection-test selector");
      auto txn=Make(pool,*memories[0],1);
      auto state=LoadPipelineState(pool,txn.state);
      auto records=LoadArray<TessellationState>(pool,state.tessellation_state);
      auto &t=records[0];
      if(invalid==1)t.phase=TessellationPhase::kControlComplete;
      if(invalid==2)t.input_vertices=33;
      if(invalid==3){auto patches=LoadArray<TessellationPatch>(pool,t.patches);patches[0].first_occurrence=UINT32_MAX;StoreArray(pool,t.patches,patches);}
      if(invalid==4)t.control_abi.vertex_inputs=2;
      if(invalid==5)t.input_address+=1;
      if(invalid==6){pool.Release(t.control_code);t.control_code=StoreNewArray(pool,std::vector<std::uint8_t>(kTess0tcs.end()-8,kTess0tcs.end()));}
      if(invalid==7){pool.Release(t.evaluation_shared);t.evaluation_shared=StoreNewArray(pool,std::vector<std::uint32_t>());}
      if(invalid==8){pool.Release(t.evaluation_code);t.evaluation_code=StoreNewArray(pool,kTess0tcs);}
      StoreArray(pool,state.tessellation_state,records);StorePipelineState(pool,txn.state,state);
      Check(input[0]->nb_write(txn),"invalid payload entered native stage chain");
      bool rejected=false;
      try{sc_core::sc_start(sc_core::sc_time(1,sc_core::SC_MS));}
      catch(const std::exception &){rejected=true;}
      Check(rejected,"invalid native tessellation payload rejected");
      PipelineTxn unexpected;Check(!output[0]->nb_read(unexpected),"invalid pipeline never publishes successful completion");
      const auto after=LoadPipelineState(pool,txn.state);
      const auto after_t=LoadArray<TessellationState>(pool,after.tessellation_state).at(0);
      Check(after_t.phase!=TessellationPhase::kEvaluationComplete,"invalid payload never claims TES completion");
      Release(pool,txn);
      Check(pool.bytes_in_flight()==0&&pool.allocations()==pool.releases(),"rejected native tasks/code/export ownership balanced");
      std::cout<<"native tessellation shader rejection "<<invalid<<": PASS "<<checks<<" checks\n";
      return 0;
    }
    for(unsigned mode=0;mode<3;++mode)for(unsigned sequence=1;sequence<=6;sequence+=2) {
      const auto first=Make(pool,*memories[mode],sequence),second=Make(pool,*memories[mode],sequence+1);
      Check(input[mode]->nb_write(first),"depth-one input accepts first job");
      Check(!input[mode]->nb_write(second),"depth-one input applies backpressure");
      sc_core::sc_start(sc_core::sc_time(1,sc_core::SC_MS));
      Check(input[mode]->nb_write(second),"dequeue event releases next job");
      sc_core::sc_start(sc_core::sc_time(1,sc_core::SC_MS));
      PipelineTxn done;Check(output[mode]->nb_read(done)&&done.sequence==sequence,"full output preserves completion order");
      Verify(pool,*memories[mode],done);
      sc_core::sc_start(sc_core::sc_time(1,sc_core::SC_MS));
      Check(output[mode]->nb_read(done)&&done.sequence==sequence+1,"output event resumes producer");Verify(pool,*memories[mode],done);
    }
    for(unsigned mode=0;mode<3;++mode) {
      PipelineState state;state.counters.gs_invocations=91;
      PipelineTxn txn;txn.state=pool.Allocate(sizeof(state));StorePipelineState(pool,txn.state,state);
      Check(input[mode]->nb_write(txn),"non-tess job accepted");sc_core::sc_start(sc_core::sc_time(1,sc_core::SC_US));
      PipelineTxn done;Check(output[mode]->nb_read(done),"all three stages pass non-tess job through");
      const auto after=LoadPipelineState(pool,done.state);Check(std::memcmp(&state,&after,sizeof(state))==0,"non-tess state remains byte identical");
      pool.Release(done.state);
    }
    Check(pool.bytes_in_flight()==0 && pool.allocations()==pool.releases(),"all TCS/TES pool tasks/exports released");
    std::cout<<"native tessellation shader modules: PASS "<<checks<<" checks / 18 draws / 36 patches\n";
  } catch(const std::exception &e){std::cerr<<e.what()<<" after "<<checks<<" checks\n";return 1;}
  return 0;
}
