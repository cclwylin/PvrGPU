// SPDX-License-Identifier: MIT
// Genuine compiled derivative-only/no-varying FS through the complete original
// Submitter -> VDM/VS -> ClipCull -> ParameterBuffer -> FS decoder -> ISP/PDS
// -> USC route. No shader result or coefficient payload is supplied by tests.
#include "common/functional_types.h"
#include "common/pipeline_state.h"
#include "fragment/fragment_frontend.h"
#include "fragment/isp.h"
#include "fragment/tile_scheduler.h"
#include "geometry/clip_cull.h"
#include "geometry/parameter_buffer.h"
#include "geometry/tiler.h"
#include "geometry/vdm.h"
#include "geometry/vertex_fetch.h"
#include "pds/pds_engine.h"
#include "pds/vertex_pds_engine.h"
#include "shader/pco_decoder.h"
#include "shader/usc_cluster.h"
#include "shader/usc_slot.h"
#include "submitter.h"
#include "memory/gpu_memory_system.h"
#include "pco_derivative_fixtures.h"
#include <systemc>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
namespace {
using namespace pvrgpu::stub;
unsigned checks=0;
void Check(bool ok,const char *why){++checks;if(!ok)throw std::runtime_error(why);}
uint32_t Bits(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}
class Audit final: public sc_core::sc_module {
public:
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};
  sc_core::sc_fifo_out<PipelineTxn> output{"output"};
  Audit(sc_core::sc_module_name n,MemoryPool&p,bool after_parameter,bool mutate)
    :sc_module(n),pool(p),after(after_parameter),corrupt(mutate){SC_THREAD(Run);}
  PipelineTxn seen{};
private:
  MemoryPool&pool;bool after,corrupt;
  void Run(){
    seen=input.read();auto s=LoadPipelineState(pool,seen.state);
    Check(s.fragment_program_summary.uses_derivatives==1,"derivative classification missing before geometry/decoder");
    Check(s.fragment_program_summary.instruction_count==0&&!HasPoolHandle(s.fragment_instructions),"planning fabricated modeled FS decode");
    Check(UsesShaderVaryings(s)&&VaryingCoefficientDwordCount(s)==4&&VaryingVectorCount(s)==0,"position-only layout changed");
    Check(HasPoolHandle(s.shader_varying_bindings)&&LoadArray<ShaderVaryingBinding>(pool,s.shader_varying_bindings).empty(),"empty exact linkage must exist before geometry");
    if(after){
      Check(s.stage==PipelineStage::kParameterBufferReady,"unexpected parameter audit stage");
      Check(s.counters.parameter_coefficient_sets==1&&s.parameter_coefficients_bytes==sizeof(ParameterCoefficientSet)&&s.parameter_coefficients_gpu_address!=0,"real ParameterBuffer must emit position coefficients into DRAM");
    }else Check(s.counters.pco_instructions==0&&s.counters.pco_decode_cycles==0,"planning charged modeled decode work");
    if(corrupt){s.fragment_program_summary.uses_derivatives=0;StorePipelineState(pool,seen.state,s);}
    output.write(seen);
  }
};
}
int sc_main(int argc,char**argv){
  using namespace pvrgpu::stub;
  try{
    const std::string mode=argc>1?argv[1]:"front";
    Check(mode=="front"||mode=="back"||mode=="changed-classification","unknown pipeline mode");
    const bool back=mode=="back",negative=mode=="changed-classification";
    MemoryPool pool;GpuMemorySystem memory(MemoryMode::kCache);Options options;options.frames=1;options.width=options.height=8;
    options.test_case="driver_pco_triangles";
    auto &c=options.driver_command;c.enabled=true;c.command="draw_pco_triangles";
    c.framebuffer_width=c.framebuffer_height=c.width=c.height=8;c.format="PIPE_FORMAT_R8G8B8A8_UNORM";
    c.vertex_pco=test::DerivativeVertexFixture();c.fragment_pco=test::DerivativeFragmentFixture();
    c.vertex_pco_abi=test::DerivativeVertexAbi();c.fragment_pco_abi=test::DerivativeFragmentAbi();
    c.vertex_stride=16;c.vertex_count=3;c.instance_count=1;c.primitive_mode=4;
    c.vertex_attribute_count=1;c.vertex_attribute_components[0]=4;
    const std::array<float,12> vertices{-.75F,-.625F,0,1,.6875F,-.625F,0,1,-.75F,.6875F,0,1};
    c.raw_vertex_data.resize(sizeof(vertices));std::memcpy(c.raw_vertex_data.data(),vertices.data(),sizeof(vertices));
    c.position_output_count=c.fragment_position_count=c.varying_output_start=c.fragment_varying_start=4;
    c.explicit_varying_bindings=true;c.fragment_output_mask[0]=15;
    c.viewport_scale_bits=c.viewport_translate_bits={Bits(4),Bits(4),Bits(.5F)};
    // Gallium front_ccw is inverted once in the model's +Y-up window space.
    c.front_ccw=back?1:0;c.half_pixel_center=1;c.depth_clip_near=c.depth_clip_far=1;
    c.sample_mask=UINT32_MAX;c.color_mask=15;
    const auto vs=DecodePcoProgram(ShaderStage::kVertex,c.vertex_pco);
    const auto fs=DecodePcoProgram(ShaderStage::kFragment,c.fragment_pco);
    Check(fs.summary.uses_derivatives==1&&CountPcoInstructions(fs.instructions,false).texture==0,"fixture must really use derivatives without SMP");
    options.driver_commands.push_back(c);
    c=DriverCommand{};c.enabled=true;c.command="draw_pco_sequence";
    c.schema="pvrgpu.driver-command.v1";c.producer="pvrgpu-gallium-driver";
    c.framebuffer_width=c.framebuffer_height=c.width=c.height=8;
    c.format="PIPE_FORMAT_R8G8B8A8_UNORM";
    sc_core::sc_event completion;
    sc_core::sc_fifo<PipelineTxn> q0("q0",1),q1("q1",1),q2("q2",1),q3("q3",1),q4("q4",1),q5("q5",1),q6("q6",1),q7("q7",1),q8("q8",1),q9("q9",1),q10("q10",1),q11("q11",1),q12("q12",1),q13("q13",1),q14("q14",1),q15("q15",1),q16("q16",1),q17("q17",1),done("done",1);
    sc_core::sc_fifo<PipelineTxn> sample_request("sample_request",1),sample_response("sample_response",1);
    Submitter submit("submit",pool,options,&memory,&completion);Audit start("start_audit",pool,false,false),post("parameter_audit",pool,true,negative);
    Vdm vdm("vdm",pool,&memory);VertexFetch fetch("fetch",pool,&memory);VertexPdsEngine vpds("vpds",pool);
    PcoDecoder vd("vd",pool,ShaderStage::kVertex),fd("fd",pool,ShaderStage::kFragment);
    UscSlot vslot("vslot",pool,ShaderStage::kVertex),fslot("fslot",pool,ShaderStage::kFragment);
    UscCluster vu("vu",pool,ShaderStage::kVertex,&memory),fu("fu",pool,ShaderStage::kFragment,&memory);
    ClipCull clip("clip",pool);Tiler tiler("tiler",pool);ParameterBuffer pb("pb",pool,&memory);
    TileScheduler scheduler("scheduler",pool,&memory);Isp isp("isp",pool,&memory);FragmentFrontend frontend("frontend",pool,&memory);PdsEngine pds("pds",pool,&memory);
    submit.output(q0);start.input(q0);start.output(q1);vdm.input(q1);vdm.output(q2);fetch.input(q2);fetch.output(q3);vpds.input(q3);vpds.output(q4);vd.input(q4);vd.output(q5);vslot.input(q5);vslot.output(q6);vu.input(q6);vu.output(q7);clip.input(q7);clip.output(q8);tiler.input(q8);tiler.output(q9);pb.input(q9);pb.output(q10);post.input(q10);post.output(q11);fd.input(q11);fd.output(q12);scheduler.input(q12);scheduler.output(q13);isp.input(q13);isp.output(q14);frontend.input(q14);frontend.output(q15);pds.input(q15);pds.output(q16);fslot.input(q16);fslot.output(q17);fu.input(q17);fu.output(done);fu.texture_request_output(sample_request);fu.texture_response_input(sample_response);
    try{sc_core::sc_start(sc_core::sc_time(1,sc_core::SC_MS));}
    catch(const std::exception&e){
      if(!negative||std::string(e.what()).find("fragment PCO derivative classification changed after submit")==std::string::npos)throw;
      const auto state=LoadPipelineState(pool,post.seen.state);ReleaseFunctionalPayloads(pool,state);pool.Release(post.seen.state);
      Check(pool.bytes_in_flight()==0&&pool.allocations()==pool.releases(),"negative pipeline leaked payloads");
      std::cout<<"derivative pipeline: "<<checks<<" checks PASS (changed classification refused)\n";return 0;
    }
    Check(!negative,"changed derivative classification was accepted");
    PipelineTxn completed;Check(done.nb_read(completed),"complete pipeline did not finish");
    const auto state=LoadPipelineState(pool,completed.state);
    Check(state.stage==PipelineStage::kFragmentShaded&&state.counters.ia_primitives==1&&state.counters.vs_invocations==3,"actual geometry/USC work missing");
    Check(state.fragment_shader_lane_count>state.active_fragment_invocations&&state.active_fragment_invocations>0,"small triangle must execute uncovered quad helpers");
    Check(state.counters.texture_requests==0&&state.counters.texel_fetches==0&&sample_request.num_available()==0,"derivatives generated texture traffic");
    Check(state.counters.pco_instructions==vs.instructions.size()+fs.instructions.size(),"predecode was charged as extra modeled decode");
    const auto tasks=LoadArray<UscFragmentTask>(pool,state.usc_fragment_tasks);
    for(const auto&t:tasks)Check(t.coefficient_dword_count==4,"PDS did not copy exact position CF4");
    const auto lanes=LoadArray<FragmentShaderLane>(pool,state.fragment_shader_lanes);
    for(const auto&l:lanes)Check(l.front_facing==!back,"helper/visible raster facing changed");
    const auto outputs=LoadArray<FragmentOutput>(pool,state.fragment_outputs);
    for(const auto&o:outputs){
      Check(o.written_mask[0]==15&&o.pixel_output[0]==Bits(1)&&o.pixel_output[1]==Bits(1)&&o.pixel_output[2]==Bits(back?0:1)&&o.pixel_output[3]==Bits(1),"genuine derivative/facing output mismatch");
    }
    ReleaseFunctionalPayloads(pool,state);pool.Release(completed.state);
    Check(pool.bytes_in_flight()==0&&pool.allocations()==pool.releases(),"pipeline leaked payloads");
    std::cout<<"derivative pipeline: "<<checks<<" checks PASS ("<<mode<<", visible="<<outputs.size()<<", lanes="<<lanes.size()<<")\n";return 0;
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
