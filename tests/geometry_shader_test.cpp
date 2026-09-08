// Native GS decoder/executor plus depth-one FIFO, modeled staging and ownership.
#include "common/geometry_emission.h"
#include "memory/gpu_memory_system.h"
#include "pco_geometry_fixtures.h"
#include "pco_geometry_compiler_fixtures.h"
#include "pco_geometry_texture_fixtures.h"
#include "common/glbench_texture_fixture.h"
#include "shader/geometry_iss.h"
#include "shader/geometry_shader.h"
#include "texture/texture_unit.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace pvrgpu::stub;
namespace {
unsigned checks=0;
void Check(bool yes,const char *reason) { ++checks; if(!yes) throw std::runtime_error(reason); }
template<class Fn> void Reject(Fn fn) {
  try { fn(); } catch(const std::exception &) { ++checks; return; }
  throw std::runtime_error("expected fail-closed native GS rejection");
}
std::uint32_t FloatBits(float value) { std::uint32_t bits; std::memcpy(&bits,&value,4); return bits; }
void CompilerNative() {
  for(unsigned kind=0;kind<8;++kind) {
    const auto program=DecodeGeometryPcoProgram(GeometryCompilerFixture(kind));
    auto abi=GeometryNativeLoadAbi();
    const unsigned temps[8]={4,6,6,6,8,9,6,6};
    abi.temps=temps[kind]; abi.vertex_outputs=kind==7?10:8;
    if(kind==5) {abi.shareds=8;abi.push_constant_count=4;}
    if(kind==6) {abi.shareds=8;abi.uniform_buffer_descriptor_count=1;abi.push_constant_start=8;}
    ValidateGeometryProgram(program,abi);
    const unsigned vertex_count=kind==0?1:kind==2?4:kind==3?6:3;
    for(unsigned primitive=0;primitive<8;++primitive) for(unsigned invocation=0;invocation<4;++invocation) {
      struct Record {
        std::vector<std::uint32_t> input,ubo;
        std::vector<std::array<std::uint32_t,10>> snapshots;
        unsigned cuts=0,ends=0,reads=0;
      } record;
      for(unsigned i=0;i<vertex_count*4;++i) record.input.push_back(FloatBits(float(i+1)));
      for(unsigned i=0;i<16;++i) record.ubo.push_back(0x80000000U+i*0x01020304U);
      std::vector<std::uint32_t> shared(abi.shareds);
      shared[0]=0x1000;shared[1]=0x80;shared[2]=vertex_count*16;
      if(kind==5) shared[4]=FloatBits(0.5f);
      if(kind==6) {shared[4]=0x2000;shared[5]=0x80;shared[6]=64;}
      auto task=MakeGeometryTask(abi,shared,primitive,invocation);
      GeometryExecutionCallbacks cb;
      cb.user_data=&record;
      cb.read=[](void *opaque,std::uint64_t address,std::uint32_t count,std::uint32_t *dst) {
        auto &r=*static_cast<Record*>(opaque);
        const bool uniform=address>=UINT64_C(0x8000002000);
        const auto base=uniform?UINT64_C(0x8000002000):UINT64_C(0x8000001000);
        const auto &source=uniform?r.ubo:r.input;
        Check(address>=base&&(address-base)%4==0&&address-base<=source.size()*4&&count*4<=source.size()*4-(address-base),
            "compiler GS LD is bounded within the primitive/UBO");
        std::copy_n(source.data()+(address-base)/4,count,dst); ++r.reads;
      };
      cb.emit=[](void *opaque,const std::uint32_t *words,std::uint32_t count,std::uint64_t mask) {
        auto &r=*static_cast<Record*>(opaque);
        Check((count==8||count==10)&&mask==((UINT64_C(1)<<count)-1),"every GS emit supplies its own actual writes");
        std::array<std::uint32_t,10> snapshot{};std::copy_n(words,count,snapshot.begin());r.snapshots.push_back(snapshot);
      };
      cb.cut=[](void *opaque){++static_cast<Record*>(opaque)->cuts;};
      cb.finish=[](void *opaque){++static_cast<Record*>(opaque)->ends;};
      GeometryExecutionStats stats;
      while(!task.ended) StepGeometryTask(program,abi,task,cb,stats);
      Check(record.snapshots.size()==vertex_count&&record.cuts==(kind==3?2:1)&&record.ends==1,"compiler GS emits/cuts preserve all primitive boundaries");
      for(unsigned vertex=0;vertex<vertex_count;++vertex) for(unsigned component=0;component<4;++component) {
        const auto index=(kind==4?invocation%3:vertex)*4+component;
        Check(record.snapshots[vertex][component]==record.input[index],"compiler GS position reads real primitive data");
        const auto expected=kind==5?FloatBits(float(index+1)*0.5f):kind==6?record.ubo[(primitive&3)*4+component]:record.input[index];
        Check(record.snapshots[vertex][4+component]==expected,"compiler GS varying math/dynamic UBO exact raw result");
      }
      if(kind==7) for(const auto &snapshot:record.snapshots)
        Check(snapshot[8]==primitive&&snapshot[9]==(primitive&1),"compiler native PrimitiveID/Layer output is not host substituted");
    }
    std::cout<<"geometry native compiler fixture "<<kind<<" PASS\n";
  }
}
void LoopNative() {
  for (unsigned kind : {8U,9U}) {
  const auto program=DecodeGeometryPcoProgram(GeometryCompilerFixture(kind));
  auto abi=GeometryNativeLoadAbi();
  abi.temps=kind==8?14:11;abi.vertex_outputs=8;abi.shareds=8;abi.push_constant_count=4;
  ValidateGeometryProgram(program,abi);
  unsigned branches=0,masks=0,backedges=0;
  for(const auto &i:program.instructions) {
    branches+=i.opcode==PcoOpcode::kBranch;
    masks+=i.opcode==PcoOpcode::kConditionalMask;
    backedges+=i.opcode==PcoOpcode::kBranch&&i.branch_target_index<i.group_index;
  }
  // Mesa legitimately unrolled this statically bounded source loop into six
  // dynamically guarded emission blocks; do not pretend it has a backedge.
  Check(branches&&masks,"native dynamic GS emission/control masks remain in the binary");
  Check(kind==8?!backedges:backedges!=0,"native GS loop lowering is accurately identified");
  for(unsigned limit=0;limit<(kind==8?8U:7U);++limit) for(unsigned primitive=0;primitive<5;++primitive) for(unsigned invocation=0;invocation<4;++invocation) {
    struct Record {
      std::array<std::uint32_t,12> input{};
      std::vector<std::array<std::uint32_t,8>> snapshots;
      unsigned cuts=0,ends=0,reads=0;
    } record;
    for(unsigned i=0;i<12;++i) record.input[i]=FloatBits(float(i+1));
    auto task=MakeGeometryTask(abi,{0x1000,0x80,48,0,limit,0,0,0},primitive,invocation);
    GeometryExecutionCallbacks cb;
    cb.user_data=&record;
    cb.read=[](void *opaque,std::uint64_t address,std::uint32_t count,std::uint32_t *dst){
      auto &r=*static_cast<Record*>(opaque);
      Check(address>=UINT64_C(0x8000001000)&&address-UINT64_C(0x8000001000)<=48&&
          count*4<=48-(address-UINT64_C(0x8000001000)),"dynamic loop native LD stays within primitive");
      std::copy_n(r.input.data()+(address-UINT64_C(0x8000001000))/4,count,dst);++r.reads;
    };
    cb.emit=[](void *opaque,const std::uint32_t *words,std::uint32_t count,std::uint64_t mask){
      auto &r=*static_cast<Record*>(opaque);
      Check(count==8&&mask==255,"dynamic GS Emit snapshot contains this vertex's actual writes");
      std::array<std::uint32_t,8> values{};std::copy_n(words,8,values.begin());r.snapshots.push_back(values);
    };
    cb.cut=[](void *opaque){++static_cast<Record*>(opaque)->cuts;};
    cb.finish=[](void *opaque){++static_cast<Record*>(opaque)->ends;};
    GeometryExecutionStats stats;
    while(!task.ended) StepGeometryTask(program,abi,task,cb,stats);
    const auto count=std::min(limit,6U);
    Check(record.snapshots.size()==count&&record.cuts==count/3&&record.ends==1,"native dynamic loop obeys uniform limit/CUT/ENDTASK");
    for(unsigned vertex=0;vertex<count;++vertex) {
      const auto &out=record.snapshots[vertex];
      Check(out[0]==FloatBits(float((vertex%3)*4+1)+float(primitive+invocation)*0.125f),"native GS primitive/invocation arithmetic exact");
      for(unsigned component=1;component<4;++component)
        Check(out[component]==record.input[(vertex%3)*4+component],"native GS dynamic gl_in preserves raw yz/w");
      Check(out[4]==FloatBits(float(vertex))&&out[5]==FloatBits(float(primitive))&&out[6]==FloatBits(float(invocation))&&out[7]==FloatBits(1),
          "native GS per-emission dynamic varying snapshot");
    }
  }
  std::cout<<"geometry native "<<(kind==8?"dynamically guarded emission":"backedge loop")<<" fixture PASS\n";
  }
}
void DynamicPushNative() {
  const auto program=DecodeGeometryPcoProgram(GeometryCompilerFixture(13));
  auto abi=GeometryNativeLoadAbi();
  abi.temps=9;abi.vertex_outputs=8;abi.shareds=20;abi.push_constant_count=16;
  ValidateGeometryProgram(program,abi);
  for (unsigned epoch=0;epoch<31;++epoch)
    for (std::uint32_t index : {0U,1U,2U,3U,UINT32_MAX}) {
      struct Record {
        std::array<std::uint32_t,4> input{};
        std::array<std::uint32_t,8> output{};
        unsigned reads=0,emits=0,cuts=0,ends=0;
      } record;
      std::vector<std::uint32_t> shared(20);
      shared[0]=0x1000;shared[1]=0x80;shared[2]=16;
      for(unsigned i=0;i<4;++i) {
        record.input[i]=0x80000000U ^ (epoch*0x01020304U+i*0x00112233U);
        shared[4+i]=0xdeadbeefU+i; // unused CB0 slot 0 must not leak into selection
      }
      for(unsigned i=8;i<20;++i) shared[i]=0x7fc12345U ^ (epoch*0x00432101U+i*0x08000000U);
      auto task=MakeGeometryTask(abi,shared,index,0);
      GeometryExecutionCallbacks cb;cb.user_data=&record;
      cb.read=[](void *opaque,std::uint64_t address,std::uint32_t count,std::uint32_t *dst) {
        auto &r=*static_cast<Record*>(opaque);
        Check(address==UINT64_C(0x8000001000)&&count==4,"dynamic CB0 GS only loads its real primitive input");
        std::copy(r.input.begin(),r.input.end(),dst);++r.reads;
      };
      cb.emit=[](void *opaque,const std::uint32_t *words,std::uint32_t count,std::uint64_t mask) {
        auto &r=*static_cast<Record*>(opaque);Check(count==8&&mask==255,"dynamic CB0 native emit has all real output writes");
        std::copy_n(words,8,r.output.begin());++r.emits;
      };
      cb.cut=[](void *opaque){++static_cast<Record*>(opaque)->cuts;};
      cb.finish=[](void *opaque){++static_cast<Record*>(opaque)->ends;};
      GeometryExecutionStats stats;
      while(!task.ended) StepGeometryTask(program,abi,task,cb,stats);
      Check(record.reads==1&&record.emits==1&&record.cuts==1&&record.ends==1,
            "dynamic CB0 shader executes one native invocation without host selection");
      for(unsigned component=0;component<4;++component) {
        Check(record.output[component]==record.input[component],"dynamic CB0 preserves raw primitive position");
        Check(record.output[4+component]==(index<3?shared[8+4*index+component]:0),
              "native dynamic CB0 selection preserves raw NaN/sign bits and returns zero out of range");
      }
    }
  std::cout<<"geometry native dynamic CB0 fixture PASS\n";
}
void EmptyFragmentNative() {
  const auto binary=GeometryEmptyFragmentFixture();
  const auto program=DecodePcoProgram(ShaderStage::kFragment,binary);
  Check(program.summary.stage==ShaderStage::kFragment&&program.summary.group_count==1&&
        program.summary.instruction_count==1&&program.summary.pixel_output_mask==0&&
        program.summary.vertex_output_mask==0&&!program.summary.ends_task,
        "compiler empty FS is a genuine fragment program with no output writes");
  Check(program.instructions[0].opcode==PcoOpcode::kNop&&program.instructions[0].end_group==1&&
        program.instructions[0].source_count==0&&program.instructions[0].target==PcoWriteTarget::kNone,
        "native empty FS is operand-free NOP.end, not a fake color export");
  const auto counts=CountPcoInstructions(program.instructions,false);
  Check(counts.alu==0&&counts.memory==0&&counts.texture==0,"NOP is control, not ALU or memory traffic");
  for(unsigned epoch=0;epoch<31;++epoch) {
    PcoFragmentExecutionContext context;
    context.shared_count=epoch%2?0:4;context.coefficient_count=epoch%2?0:4;
    context.shared_registers.fill(0xdeadbeefU+epoch);context.coefficients.fill(0x7fc12345U+epoch);
    const auto result=ExecuteFragmentPco(program.summary,program.instructions,context);
    Check(result.executed_instruction_count==1&&result.written_mask==0&&!result.discarded&&
          !result.depth_written&&!result.texture_request_valid&&!result.suspended,
          "NOP.end executes once without color/depth/discard/resource side effects");
  }
  Reject([&]{DecodePcoProgram(ShaderStage::kVertex,binary);});
  for(unsigned mutation=0;mutation<8;++mutation) {
    auto bad=binary;
    if(mutation==0) bad[0]|=0x10; // no destination address region
    if(mutation==1) bad[1]|=1; // no predicate control
    if(mutation==2) bad[1]|=2; // no output target
    if(mutation==3) bad[2]&=0x7f; // only native final NOP.end is supported
    if(mutation==4) bad[3]=1; // no source/operand byte
    if(mutation==5) bad[4]=0xf1; // alignment count must match
    if(mutation==6) bad.back()=0; // alignment reserved bytes
    if(mutation==7) bad.push_back(0); // no trailing group after END
    Reject([&]{DecodePcoProgram(ShaderStage::kFragment,bad);});
  }
  const auto reject_instruction=[&](const auto &mutate) {
    auto bad=program.instructions;mutate(bad[0]);
    Reject([&]{ExecuteFragmentPco(program.summary,bad);});
  };
  reject_instruction([](auto &i){i.source_count=1;});
  reject_instruction([](auto &i){i.source.bank=PcoRegisterBank::kTemporary;});
  reject_instruction([](auto &i){i.target=PcoWriteTarget::kPixelOutput;});
  reject_instruction([](auto &i){i.output_index=1;});
  reject_instruction([](auto &i){i.output_index1=1;});
  reject_instruction([](auto &i){i.repeat_count=2;});
  reject_instruction([](auto &i){i.end_group=0;});
  reject_instruction([](auto &i){i.immediate=1;});
  reject_instruction([](auto &i){i.data_request=1;});
  reject_instruction([](auto &i){i.branch_target_index=1;});
  reject_instruction([](auto &i){i.loop_count=1;});
  reject_instruction([](auto &i){i.exec_cnd=1;});
  reject_instruction([](auto &i){i.control_operation=1;});
  reject_instruction([](auto &i){i.memory_cache_mode=1;});
  reject_instruction([](auto &i){i.phase_composed=1;});
  std::cout<<"geometry pipeline native empty fragment NOP.end PASS\n";
}
void PureNative() {
  auto binary=GeometryNativeLoadFixture();
  const auto program=DecodeGeometryPcoProgram(binary);
  const auto abi=GeometryNativeLoadAbi();
  Check(program.summary.stage==ShaderStage::kGeometry && program.summary.group_count==9 &&
      program.summary.ends_task && program.summary.vertex_output_mask==15,"genuine geometry decode summary");
  ValidateGeometryProgram(program,abi);
  Reject([&]{DecodePcoProgram(ShaderStage::kVertex,binary);});
  Reject([&]{DecodePcoProgram(ShaderStage::kFragment,binary);});
  Reject([&]{DecodePcoProgram(ShaderStage::kCompute,binary);});
  for(unsigned field: {0U,1U,2U,3U,4U}) {
    auto bad=binary;
    if(field==0) bad[59]=7; // reserved UVSW operation
    if(field==1) bad[59]|=0x10; // forbidden DSEL
    if(field==2) bad[60]=1; // reserved/unused source
    if(field==3) bad[74]&=0x7f; // ENDTASK requires native END
    if(field==4) bad.push_back(0);
    Reject([&]{DecodeGeometryPcoProgram(bad);});
  }
  struct Record {
    std::array<std::uint32_t,4> input{}, output{};
    unsigned reads=0,emits=0,cuts=0,ends=0;
  } record;
  GeometryExecutionCallbacks cb;
  cb.user_data=&record;
  cb.read=[](void *opaque,std::uint64_t address,std::uint32_t count,std::uint32_t *dst){
    auto &r=*static_cast<Record*>(opaque);
    Check(address==UINT64_C(0x8000001000)&&count==4,"GS native LD preserves 64-bit range/width");
    std::copy(r.input.begin(),r.input.end(),dst); ++r.reads;
  };
  cb.emit=[](void *opaque,const std::uint32_t *words,std::uint32_t count,std::uint64_t mask){
    auto &r=*static_cast<Record*>(opaque); Check(count==4&&mask==15,"GS emit written mask");
    std::copy_n(words,4,r.output.begin()); ++r.emits;
  };
  cb.cut=[](void *opaque){++static_cast<Record*>(opaque)->cuts;};
  cb.finish=[](void *opaque){++static_cast<Record*>(opaque)->ends;};
  for(unsigned epoch=0;epoch<31;++epoch) {
    record={}; for(unsigned i=0;i<4;++i) record.input[i]=(epoch*0x1234567U+i*0x1020304U)^0x80000000U;
    auto task=MakeGeometryTask(abi,{0x1000,0x80,16,0},19,7);
    GeometryExecutionStats stats;
    Check(task.inputs[0]==19&&task.inputs[1]==7,"GS true primitive/invocation system values");
    StepGeometryTask(program,abi,task,cb,stats);
    Check(task.pending_count==4&&!task.temporary_written.test(0)&&record.reads==1,"LD result remains pending before WDF");
    StepGeometryTask(program,abi,task,cb,stats);
    Check(!task.pending_count&&task.temporary_written.contains_range(0,4),"WDF commits true LD response");
    while(!task.ended) StepGeometryTask(program,abi,task,cb,stats);
    Check(record.output==record.input&&record.reads==1&&record.emits==1&&record.cuts==1&&record.ends==1,
        "native GS exact raw export/event conservation");
    Check(stats.load_instructions==1&&stats.emit_instructions==1&&stats.cut_instructions==1&&stats.instructions==9,
        "GS native instruction counters");
    Check(!task.outputs_written,"EmitVertex invalidates prior output-written metadata");
    Reject([&]{StepGeometryTask(program,abi,task,cb,stats);});
  }
  auto bad=program; bad.instructions[1].opcode=PcoOpcode::kNop;
  Reject([&]{ValidateGeometryProgram(bad,abi);});
  auto badabi=abi; badabi.uniform_buffer_descriptor_start=0;
  Reject([&]{ValidateGeometryProgram(program,badabi);});
  for (unsigned inputs : {0U,1U,3U,64U}) {
    badabi=abi; badabi.vertex_inputs=inputs;
    Reject([&]{ValidateGeometryProgram(program,badabi);});
    Reject([&]{MakeGeometryTask(badabi,{0x1000,0x80,16,0},0,0);});
  }
  for (unsigned outputs : {0U,3U,65U}) {
    badabi=abi; badabi.vertex_outputs=outputs;
    Reject([&]{ValidateGeometryProgram(program,badabi);});
  }
  badabi=abi; badabi.shareds=5;
  Reject([&]{ValidateGeometryProgram(program,badabi);});
  Reject([&]{MakeGeometryTask(badabi,{0x1000,0x80,16,0,0},0,0);});
}

void SampleExecution() {
  const auto native = DecodeGeometryPcoProgram(GeometryTextureNativeFixture());
  ValidateGeometryProgram(native, GeometryTextureNativeAbi());
  unsigned native_samples = 0;
  for (const auto &i : native.instructions) if (i.opcode == PcoOpcode::kTextureSample) {
    Check(i.source1.index == 4 + native_samples * 20 && i.source2.index == 12 + native_samples * 20,
          "genuine GS binary decodes SH4/SH24 texture sets independently");
    auto bad = GeometryTextureNativeFixture();
    bad[i.binary_offset + 1] |= 3U;
    Reject([&] { DecodeGeometryPcoProgram(bad); });
    ++native_samples;
  }
  Check(native_samples == 2, "native GS compiler fixture retains both true SMP instructions");
  auto program = DecodeGeometryPcoProgram(GeometryNativeLoadFixture());
  auto abi = GeometryNativeLoadAbi();
  abi.temps = 8; abi.shareds = 44;
  abi.uniform_buffer_descriptor_start = abi.push_constant_start = 44;
  auto &sample = program.instructions[0];
  sample = {};
  sample.opcode = PcoOpcode::kTextureSample;
  sample.source = {PcoRegisterBank::kTemporary, 4};
  sample.source1 = {PcoRegisterBank::kShared, 4};
  sample.source2 = {PcoRegisterBank::kShared, 12};
  sample.source_count = 3; sample.repeat_count = 1;
  sample.component_count = 4; sample.texture_dimension = 2;
  sample.texture_fcnorm = 1; sample.target = PcoWriteTarget::kTemporary;
  ValidateGeometryProgram(program, abi);
  struct Record {
    PcoTextureRequest request;
    unsigned calls = 0;
    std::array<std::uint32_t, 4> words{0x80000000U, 0x7fc0abcdU, 0x12345678U, 0xffffffffU};
  } record;
  GeometryExecutionCallbacks cb;
  cb.user_data = &record;
  cb.sample = [](void *opaque, const PcoTextureRequest &request, std::uint32_t *response) {
    auto &r = *static_cast<Record *>(opaque);
    r.request = request; ++r.calls; std::copy(r.words.begin(), r.words.end(), response);
  };
  for (unsigned set : {0U, 1U}) for (unsigned dimension : {2U, 3U}) {
    std::vector<std::uint32_t> shared(44);
    for (unsigned word = 4; word < shared.size(); ++word) shared[word] = 0x01020300U + word;
    auto task = MakeGeometryTask(abi, shared, 5, 7);
    for (unsigned c = 4; c < 8; ++c) {
      task.temporaries[c] = FloatBits(0.125F * (c - 3)); task.temporary_written.set(c);
    }
    sample.source1.index = 4 + set * 20; sample.source2.index = 12 + set * 20;
    sample.texture_dimension = dimension;
    ValidateGeometryProgram(program, abi);
    GeometryExecutionStats stats;
    record.calls = 0;
    StepGeometryTask(program, abi, task, cb, stats);
    Check(record.calls == 1 && task.pending_count == 4 && !task.temporary_written.test(0),
          "GS SMP real response is pending until WDF");
    Check(record.request.descriptor_set == set && record.request.dimension == dimension &&
          record.request.coordinates[0] == task.temporaries[4] &&
          record.request.coordinates[1] == task.temporaries[5] &&
          record.request.coordinates[2] == (dimension == 3 ? task.temporaries[6] : 0U),
          "GS SMP preserves raw coordinates and independent descriptor namespace");
    for (unsigned word = 0; word < 4; ++word)
      Check(record.request.texture_state[word] == shared[4 + set * 20 + word] &&
            record.request.sampler_state[word] == shared[12 + set * 20 + word],
            "GS SMP raw descriptors begin after primitive input SH0..3");
    StepGeometryTask(program, abi, task, cb, stats);
    Check(!task.pending_count && task.temporary_written.contains_range(0, 4) &&
          std::equal(record.words.begin(), record.words.end(), task.temporaries.begin()),
          "GS WDF commits exact raw float/integer response DWORDs");
    Check(stats.texture_instructions == 1 && !stats.memory_instructions && !stats.load_instructions,
          "GS SMP instruction accounting is separate from primitive LD");
    task = MakeGeometryTask(abi, shared, 0, 0);
    Reject([&] { StepGeometryTask(program, abi, task, cb, stats); });
    sample.exec_cnd = 1; task.predicate = 0; record.calls = 0;
    StepGeometryTask(program, abi, task, cb, stats);
    StepGeometryTask(program, abi, task, cb, stats);
    Check(!record.calls && !task.pending_count && !task.temporary_written.test(0),
          "predicated-away GS SMP neither reads unwritten coords nor requests memory");
    sample.exec_cnd = 0;
  }
  for (unsigned mutation = 0; mutation < 13; ++mutation) {
    auto bad = program;
    auto &s = bad.instructions[0];
    switch (mutation) {
      case 0: s.source1.index = 0; break;
      case 1: s.source1.index = 5; break;
      case 2: s.source1.index = 44; s.source2.index = 52; break;
      case 3: s.source2.index = s.source1.index + 4; break;
      case 4: s.source.bank = PcoRegisterBank::kShared; break;
      case 5: s.source.index = 7; break;
      case 6: s.output_index = 6; break;
      case 7: s.component_count = 3; break;
      case 8: s.texture_dimension = 1; break;
      case 9: s.texture_non_normalized_coords = 1; break;
      case 10: s.data_request = 1; break;
      case 11: s.repeat_count = 2; break;
      case 12: bad.instructions[1].opcode = PcoOpcode::kNop; break;
    }
    Reject([&] { ValidateGeometryProgram(bad, abi); });
  }
  for (unsigned start : {0U, 3U, 5U, 20U, 184U}) {
    auto bad = abi; bad.uniform_buffer_descriptor_start = bad.push_constant_start = bad.shareds = start;
    Reject([&] { ValidateGeometryProgram(program, bad); });
  }
}

PipelineTxn MakePipeline(MemoryPool &pool, MemoryMode mode, unsigned epoch, unsigned max_vertices);
PipelineTxn MakeTexturePipeline(MemoryPool &pool, GpuMemorySystem &memory, unsigned epoch) {
  auto txn = MakePipeline(pool, memory.mode(), epoch, 1);
  auto s = LoadPipelineState(pool, txn.state);
  pool.Release(s.geometry_code); pool.Release(s.geometry_shared_registers);
  s.functional_case = FunctionalCase::kDriverPcoTriangles;
  s.geometry_code = StoreNewArray(pool, GeometryTextureNativeFixture());
  s.geometry_pco_abi = GeometryTextureNativeAbi();
  s.geometry_sampled_texture_count = 2;
  std::vector<std::uint32_t> shared(52);
  std::vector<TextureResource> resources;
  std::vector<SamplerState> samplers;
  for (unsigned set = 0; set < 2; ++set) {
    auto fixture = MakeGlbenchFillTextureFixture(TextureFilter::kNearest);
    const std::array<std::uint8_t, 4> color = set ? std::array<std::uint8_t,4>{16, 32, 64, 255}
                                               : std::array<std::uint8_t,4>{128, 64, 32, 255};
    for (std::size_t word = 0; word < fixture.texture_bytes.size(); ++word)
      fixture.texture_bytes[word] = color[word % 4];
    fixture.resource.gpu_address += set * UINT64_C(0x200000);
    fixture.resource.descriptor_set = set; fixture.sampler.descriptor_set = set;
    const auto address_mask = ((UINT64_C(1) << 38) - 1) << 16;
    auto image_word1 = std::uint64_t(fixture.fragment_shared[2]) |
                       (std::uint64_t(fixture.fragment_shared[3]) << 32);
    image_word1 = (image_word1 & ~address_mask) | ((fixture.resource.gpu_address >> 2) << 16);
    fixture.fragment_shared[2] = image_word1; fixture.fragment_shared[3] = image_word1 >> 32;
    std::copy(fixture.fragment_shared.begin(), fixture.fragment_shared.end(), shared.begin() + 4 + set * 20);
    memory.HostWrite(fixture.resource.gpu_address, fixture.texture_bytes.data(), fixture.texture_bytes.size());
    resources.push_back(fixture.resource); samplers.push_back(fixture.sampler);
  }
  s.geometry_texture_resources = StoreNewArray(pool, resources);
  s.geometry_sampler_states = StoreNewArray(pool, samplers);
  const std::uint64_t uniform_address = UINT64_C(0x8000020000);
  const std::array<std::uint32_t,4> uniform{FloatBits(-1), FloatBits(0.5F), FloatBits(4), FloatBits(2)};
  memory.HostWrite(uniform_address, reinterpret_cast<const std::uint8_t *>(uniform.data()), sizeof(uniform));
  shared[44] = static_cast<std::uint32_t>(uniform_address);
  shared[45] = static_cast<std::uint32_t>(uniform_address >> 32); shared[46] = sizeof(uniform);
  shared[48] = FloatBits(0.5F);
  s.geometry_uniform_buffer_resources = StoreNewArray(pool,
      std::vector<UniformBufferResource>{{uniform_address, sizeof(uniform), 0, 44}});
  s.geometry_shared_registers = StoreNewArray(pool, shared);
  auto lanes = LoadArray<VertexLane>(pool, s.vertex_lanes);
  for (auto &lane : lanes) {
    const std::array<std::uint32_t,4> position{FloatBits(0.125F), FloatBits(0.375F), 0, FloatBits(1)};
    std::copy(position.begin(), position.end(), lane.vertex_output);
  }
  StoreArray(pool, s.vertex_lanes, lanes);
  StorePipelineState(pool, txn.state, s);
  return txn;
}

void VerifyTextureAndRelease(MemoryPool &pool, PipelineTxn txn) {
  const auto s = LoadPipelineState(pool, txn.state);
  const auto lanes = LoadArray<VertexLane>(pool, s.vertex_lanes);
  std::cout << "GS texture memorymode=" << static_cast<unsigned>(s.memory_mode)
      << " lanes=" << lanes.size() << " invocations=" << s.counters.gs_invocations
      << " load=" << s.counters.gs_load_instructions << " memory=" << s.counters.gs_memory_instructions
      << " smp=" << s.geometry_texture_instruction_count << " request=" << s.geometry_texture_request_count
      << " fetch=" << s.geometry_texel_fetch_count << '\n';
  Check(s.stage == PipelineStage::kVertexShaded && lanes.size() == 6 &&
        s.counters.gs_invocations == 6 && s.counters.gs_load_instructions == 12 &&
        s.counters.gs_memory_instructions == 78 && s.counters.gs_tex_instructions == 12 &&
        s.geometry_texture_instruction_count == 12 &&
        s.geometry_texture_request_count == 12 && s.geometry_texel_fetch_count == 12 &&
        s.counters.texture_requests == 12 && s.counters.texel_fetches == 12 &&
        !s.vertex_texture_request_count && !s.fragment_texture_request_count,
        "native GS+UBO+two SMP FIFO rounds conserve per-invocation traffic across all memory modes");
  const auto drawlists = LoadArray<DrawListStats>(pool, s.drawlist_stats);
  Check(drawlists.size() == 1 && drawlists[0].geometry.program_tex_instructions == 2 &&
        drawlists[0].geometry.executed_tex_instructions == 12 &&
        drawlists[0].geometry.program_memory_instructions == 13 &&
        drawlists[0].geometry.executed_memory_instructions == 78,
        "GS DrawList distinguishes native texture from other memory instructions");
  const std::array<float,4> ubo{-1, 0.5F, 4, 2}, tex0{128,64,32,255}, tex1{16,32,64,255};
  for (const auto &lane : lanes) for (unsigned component = 0; component < 4; ++component) {
    const float expected = ((ubo[component] + tex0[component] / 255.0F) + tex1[component] / 255.0F) * 0.5F;
    Check(lane.vertex_output[4 + component] == FloatBits(expected),
          "native GS varying comes from real TextureUnit texels, UBO and CB0 arithmetic");
  }
  Check(!HasPoolHandle(s.texture_sample_requests) && !HasPoolHandle(s.texture_sample_responses),
        "native GS retires every FIFO response before publishing completed vertices");
  ReleaseFunctionalPayloads(pool, s); pool.Release(txn.state);
}

PipelineTxn MakePipeline(MemoryPool &pool,MemoryMode mode,unsigned epoch,unsigned max_vertices) {
  PipelineState s;
  s.stage=PipelineStage::kVertexShaded; s.memory_mode=mode;
  s.geometry_code=StoreNewArray(pool,GeometryNativeLoadFixture());
  s.geometry_pco_abi=GeometryNativeLoadAbi();
  s.geometry_input_buffer_gpu_address=UINT64_C(0x8000001000);
  s.geometry_invocations=3; s.geometry_max_vertices=max_vertices;
  s.geometry_input_stride_dwords=4; s.geometry_input_primitive_vertices=1;
  s.geometry_output_topology=PrimitiveTopology::kPoints;
  s.position_output_start=0; s.position_output_count=4;
  s.geometry_shared_registers=StoreNewArray(pool,std::vector<std::uint32_t>(4));
  std::vector<VertexLane> lanes(3);
  for(unsigned vertex=0;vertex<3;++vertex) {
    lanes[vertex].emitted=lanes[vertex].ended=1;
    for(unsigned c=0;c<4;++c) lanes[vertex].vertex_output[c]=epoch*10000+vertex*100+c;
  }
  s.vertex_lanes=StoreNewArray(pool,lanes);
  s.vertex_lane_refs=StoreNewArray(pool,std::vector<VertexLaneRef>{{2,32},{0,11},{1,21}});
  GeometryInputPrimitive a,b;
  a.vertex_indices[0]=0; a.vertex_count=1; a.primitive_id=5; a.instance_id=2;
  b.vertex_indices[0]=2; b.vertex_count=1; b.primitive_id=6; b.instance_id=2;
  s.geometry_input_primitives=StoreNewArray(pool,std::vector<GeometryInputPrimitive>{a,b});
  s.drawlist_stats=StoreNewArray(pool,std::vector<DrawListStats>(1));
  const auto handle=pool.Allocate(sizeof(s)); StorePipelineState(pool,handle,s);
  return {handle,epoch,epoch};
}
void VerifyAndRelease(MemoryPool &pool,PipelineTxn txn,unsigned epoch,unsigned max_vertices) {
  const auto s=LoadPipelineState(pool,txn.state);
  const auto lanes=LoadArray<VertexLane>(pool,s.vertex_lanes);
  const auto refs=LoadArray<VertexLaneRef>(pool,s.vertex_lane_refs);
  const auto primitives=LoadArray<GeometryRasterPrimitive>(pool,s.geometry_primitives);
  const auto count=max_vertices?6U:0U;
  Check(lanes.size()==count&&refs.size()==count*3&&primitives.size()==count,"true GS emitted/assembled output count");
  for(unsigned i=0;i<count;++i) {
    for(unsigned c=0;c<4;++c) Check(lanes[i].vertex_output[c]==epoch*10000+(i<3?2:1)*100+c,"GS input occurrence mapping/raw output");
    for(unsigned r=0;r<3;++r) Check(refs[3*i+r].lane_index==i&&refs[3*i+r].vertex_index==i,"GS point expansion immutable snapshot identity");
    Check(primitives[i].input_primitive_id==(i<3?5:6)&&primitives[i].instance_id==2&&primitives[i].invocation_id==i%3,
        "GS primitive/instance/invocation identities survive module boundary");
  }
  Check(s.counters.gs_invocations==6&&s.counters.gs_primitives==count&&s.counters.gs_emitted_vertices==count&&
      s.counters.gs_input_write_bytes==32&&s.counters.gs_input_read_bytes==96&&s.counters.gs_load_instructions==6,
      "GS executes native input memory and zero-max invocation work");
  Check(!s.counters.vs_invocations&&!s.counters.fs_alu_instructions&&!s.counters.vs_memory_instructions,
      "GS work cannot be charged to another shader stage");
  const auto stats=LoadArray<DrawListStats>(pool,s.drawlist_stats);
  Check(stats[0].geometry.invocations==6&&stats[0].geometry.executed_memory_instructions!=0,"GS DrawList native evidence");
  for(auto handle:{s.geometry_code,s.geometry_instructions,s.geometry_shared_registers,s.geometry_input_primitives,
      s.vertex_lanes,s.vertex_lane_refs,s.geometry_primitives,s.drawlist_stats}) pool.Release(handle);
  pool.Release(txn.state);
}
} // namespace

int sc_main(int,char**) {
  try {
    PureNative();
    SampleExecution();
    CompilerNative();
    LoopNative();
    DynamicPushNative();
    EmptyFragmentNative();
    MemoryPool pool;
    GpuMemorySystem direct(MemoryMode::kDirect),bypass(MemoryMode::kBypass),cache(MemoryMode::kCache);
    sc_core::sc_fifo<PipelineTxn> di("di",1),do_("do",1),bi("bi",1),bo("bo",1),ci("ci",1),co("co",1);
    GeometryShader d("geometry_direct",pool,&direct),b("geometry_bypass",pool,&bypass),c("geometry_cache",pool,&cache);
    d.input(di);d.output(do_);b.input(bi);b.output(bo);c.input(ci);c.output(co);
    sc_core::sc_fifo<PipelineTxn> drq("drq",1),drs("drs",1),brq("brq",1),brs("brs",1),crq("crq",1),crs("crs",1);
    sc_core::sc_fifo<PipelineTxn> dti("dti",1),dto("dto",1),bti("bti",1),bto("bto",1),cti("cti",1),cto("cto",1);
    TextureUnit dt("texture_direct",pool,&direct),bt("texture_bypass",pool,&bypass),ct("texture_cache",pool,&cache);
    dt.input(dti);dt.output(dto);bt.input(bti);bt.output(bto);ct.input(cti);ct.output(cto);
    d.texture_request_output(drq);d.texture_response_input(drs);dt.geometry_sample_input(drq);dt.geometry_sample_output(drs);
    b.texture_request_output(brq);b.texture_response_input(brs);bt.geometry_sample_input(brq);bt.geometry_sample_output(brs);
    c.texture_request_output(crq);c.texture_response_input(crs);ct.geometry_sample_input(crq);ct.geometry_sample_output(crs);
    const std::array<MemoryMode,3> modes{MemoryMode::kDirect,MemoryMode::kBypass,MemoryMode::kCache};
    const std::array<sc_core::sc_fifo<PipelineTxn>*,3> inputs{&di,&bi,&ci},outputs{&do_,&bo,&co};
    for(unsigned mode=0;mode<3;++mode) {
      auto first=MakePipeline(pool,modes[mode],1,1);
      auto second=MakePipeline(pool,modes[mode],2,0);
      Check(inputs[mode]->nb_write(first),"depth-one input accepts one GS transaction");
      Check(!inputs[mode]->nb_write(second),"depth-one input applies backpressure");
      sc_core::sc_start(sc_core::sc_time(1,sc_core::SC_US));
      Check(inputs[mode]->nb_write(second),"GS input resumes after prior dequeue");
      sc_core::sc_start(sc_core::sc_time(1,sc_core::SC_US));
      PipelineTxn done;
      Check(outputs[mode]->nb_read(done)&&done.sequence==1,"first completion preserved while output was full");
      VerifyAndRelease(pool,done,1,1);
      sc_core::sc_start(sc_core::sc_time(1,sc_core::SC_US));
      Check(outputs[mode]->nb_read(done)&&done.sequence==2,"output data_read event releases pending GS completion");
      VerifyAndRelease(pool,done,2,0);
    }
    const std::array<GpuMemorySystem *,3> memories{&direct,&bypass,&cache};
    for (unsigned mode = 0; mode < 3; ++mode) {
      const auto textured = MakeTexturePipeline(pool, *memories[mode], 3 + mode);
      Check(inputs[mode]->nb_write(textured), "native textured GS task enters bounded FIFO");
      sc_core::sc_start(sc_core::sc_time(100,sc_core::SC_US));
      PipelineTxn done;
      Check(outputs[mode]->nb_read(done) && done.sequence == 3 + mode,
            "native GS TextureUnit WDF continuation completes in each memory mode");
      VerifyTextureAndRelease(pool, done);
    }
    Check(pool.bytes_in_flight()==0&&pool.allocations()==pool.releases(),"GS pool ownership is balanced");
    std::cout<<"native GeometryShader "<<checks<<" checks PASS\n";
    return 0;
  } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
