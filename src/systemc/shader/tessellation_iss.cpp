#include "shader/tessellation_iss.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace pvrgpu::stub {
namespace {
[[noreturn]] void Fail(const char *message) {
  throw std::runtime_error(std::string("native tessellation: ") + message);
}
bool Fits(std::uint32_t start, std::uint32_t count, std::uint32_t limit) {
  return start <= limit && count <= limit - start;
}
void ValidateAbi(ShaderStage stage, const DriverPcoStageAbi &abi) {
  const bool control = stage == ShaderStage::kTessellationControl;
  if (!control && stage != ShaderStage::kTessellationEvaluation)
    Fail("stage is neither TCS nor TES");
  const auto descriptors = control ? 8U : 4U;
  // Reserved system-input prefix plus PCO's aligned writable VTXIN scratch.
  // MakeTask initializes only the prefix, so spare words still require a
  // native write before any read and every access stays within this ABI.
  if (abi.temps > kPcoTemporaryCount || abi.vertex_inputs < (control ? 3U : 5U) ||
      abi.vertex_inputs > kPcoVertexInputCount ||
      (control ? abi.vertex_outputs != 0 : abi.vertex_outputs < 4 || abi.vertex_outputs > 64) ||
      abi.coefficients || abi.entry_offset || abi.shareds < descriptors ||
      abi.shareds > kPcoMaximumSharedCount || abi.uniform_buffer_descriptor_count > 15 ||
      abi.uniform_buffer_descriptor_start != descriptors ||
      abi.push_constant_start != descriptors + abi.uniform_buffer_descriptor_count * 4 ||
      !Fits(abi.push_constant_start, abi.push_constant_count, abi.shareds) ||
      std::uint64_t{abi.push_constant_start} + abi.push_constant_count != abi.shareds)
    Fail("register/descriptor/push ABI is invalid");
}
bool IsAlu(PcoOpcode op) {
  switch (op) {
  case PcoOpcode::kMoveImmediate: case PcoOpcode::kMoveBypass:
  case PcoOpcode::kFloatNegate: case PcoOpcode::kFloatAbs:
  case PcoOpcode::kIntegerAdd: case PcoOpcode::kIntegerMultiplyAdd32:
  case PcoOpcode::kIntegerMultiplyAdd64High: case PcoOpcode::kIntegerAdd64_32:
  case PcoOpcode::kBitwiseAnd: case PcoOpcode::kBitwiseOr:
  case PcoOpcode::kBitwiseXor: case PcoOpcode::kBitwiseXnor:
  case PcoOpcode::kShiftLeft: case PcoOpcode::kShiftRight:
  case PcoOpcode::kBitfieldExtractUnsigned: case PcoOpcode::kBitfieldExtractSigned:
  case PcoOpcode::kBitfieldInsert: case PcoOpcode::kBooleanCompare:
  case PcoOpcode::kFloatEqual: case PcoOpcode::kFloatLess:
  case PcoOpcode::kFloatGreaterEqual: case PcoOpcode::kTestZero:
  case PcoOpcode::kConditionalSelect: case PcoOpcode::kConditionalSelectNegateTrue:
  case PcoOpcode::kConditionalSelectGreaterZero: case PcoOpcode::kTestConditionalSelect:
  case PcoOpcode::kFloatFloor: case PcoOpcode::kFloatAdd:
  case PcoOpcode::kFloatAddNegateSource0: case PcoOpcode::kFloatMultiply:
  case PcoOpcode::kFloatMad: case PcoOpcode::kFloatMadNegateSource0:
  case PcoOpcode::kFloatMadNegateSource2: case PcoOpcode::kFloatMadNegateSource0Source2:
  case PcoOpcode::kFloatMin: case PcoOpcode::kFloatMax:
  case PcoOpcode::kIntegerMaxSigned: case PcoOpcode::kIntegerMinSigned:
  case PcoOpcode::kReciprocal: case PcoOpcode::kReciprocalSquareRoot:
  case PcoOpcode::kFloatLog2: case PcoOpcode::kFloatExp2:
  case PcoOpcode::kUnpackUnsignedToFloat: case PcoOpcode::kUnpackSignedToFloat:
  case PcoOpcode::kFloatToUint32Rtne: case PcoOpcode::kFloatToUint32Rtz:
  case PcoOpcode::kFloatToInt32Rtne: case PcoOpcode::kFloatToInt32Rtz:
  case PcoOpcode::kFloatPackHalfRtne: case PcoOpcode::kFloatPackHalfRtz:
  case PcoOpcode::kFloatUnpackHalf: case PcoOpcode::kUnpackVector:
    return true;
  default: return false;
  }
}
bool IsWrite(PcoOpcode op) {
  return op == PcoOpcode::kUvsWrite || op == PcoOpcode::kUvsWriteEmitEndTask;
}
bool IsEmit(PcoOpcode op) {
  return op == PcoOpcode::kUvsEmitEndTask || op == PcoOpcode::kUvsWriteEmitEndTask;
}
bool Packed(PcoOpcode op) {
  return op == PcoOpcode::kFloatUnpackHalf || op == PcoOpcode::kUnpackVector;
}
void ValidateSource(PcoRegisterRef ref, unsigned repeat, const DriverPcoStageAbi &abi) {
  const auto index = ref.index + repeat;
  switch (ref.bank) {
  case PcoRegisterBank::kTemporary: if (index < abi.temps) return; break;
  case PcoRegisterBank::kVertexInput: if (index < abi.vertex_inputs) return; break;
  case PcoRegisterBank::kShared: if (index < abi.shareds) return; break;
  case PcoRegisterBank::kSpecial:
    if (!repeat && (ref.index == kPcoSpecialInstanceNumber ||
                    PcoSpecialConstantBits(ref.index, nullptr))) return;
    break;
  default: break;
  }
  Fail("source register exceeds stage ABI");
}
std::uint32_t Read(PcoRegisterRef ref, unsigned repeat, const DriverPcoStageAbi &abi,
                   const TessellationTaskState &task, unsigned lane_index) {
  ValidateSource(ref, repeat, abi);
  const auto index = ref.index + repeat;
  const auto &lane = task.lanes[lane_index];
  switch (ref.bank) {
  case PcoRegisterBank::kTemporary:
    if (!lane.temporary_written.test(index)) Fail("TEMP read before write");
    return lane.temporaries[index];
  case PcoRegisterBank::kVertexInput:
    if (!(lane.inputs_written & (UINT64_C(1) << index))) Fail("VTXIN read before write");
    return lane.inputs[index];
  case PcoRegisterBank::kShared: return task.shared[index];
  case PcoRegisterBank::kSpecial: {
    if (ref.index == kPcoSpecialInstanceNumber) return lane_index;
    std::uint32_t bits = 0;
    if (!PcoSpecialConstantBits(ref.index, &bits)) Fail("unknown special register");
    return bits;
  }
  default: Fail("unsupported source bank");
  }
}
void Write(PcoWriteTarget target, unsigned index, std::uint32_t value,
           const DriverPcoStageAbi &abi, TessellationLaneState &lane) {
  if (target == PcoWriteTarget::kNone) return;
  if (target == PcoWriteTarget::kTemporary && index < abi.temps) {
    lane.temporaries[index] = value; lane.temporary_written.set(index);
  } else if (target == PcoWriteTarget::kVertexInput && index < abi.vertex_inputs) {
    lane.inputs[index] = value; lane.inputs_written |= UINT64_C(1) << index;
  } else if (target == PcoWriteTarget::kVertexOutput && index < abi.vertex_outputs) {
    lane.outputs[index] = value; lane.outputs_written |= UINT64_C(1) << index;
  } else Fail("destination register exceeds stage ABI");
}
bool Selected(const PcoInstruction &i, const TessellationLaneState &lane) {
  switch (i.exec_cnd) {
  case 0: return lane.execution_predicate;
  case 1: return lane.execution_predicate && lane.predicate;
  case 2: return true;
  case 3: return lane.execution_predicate && !lane.predicate;
  default: Fail("invalid execution condition");
  }
}
bool Condition(unsigned condition, const TessellationLaneState &lane) {
  switch (condition) {
  case 0: return true; case 1: return lane.predicate;
  case 2: return false; case 3: return !lane.predicate;
  default: Fail("invalid execution-mask condition");
  }
}
TessellationTaskState MakeTask(ShaderStage stage, const DriverPcoStageAbi &abi,
    const std::vector<std::uint32_t> &shared, std::uint32_t count) {
  ValidateAbi(stage, abi);
  if (shared.size() != abi.shareds || shared[3] ||
      (stage == ShaderStage::kTessellationControl && shared[7]) ||
      !count || count > kTessellationTaskWidth)
    Fail("task width or immutable shared payload is invalid");
  TessellationTaskState task;
  task.stage = stage; task.lane_count = count;
  std::copy(shared.begin(), shared.end(), task.shared.begin());
  return task;
}
} // namespace

void ValidateTessellationProgram(const PcoDecodedProgram &program,
                                 const DriverPcoStageAbi &abi) {
  ValidateAbi(program.summary.stage, abi);
  const bool control = program.summary.stage == ShaderStage::kTessellationControl;
  if (program.instructions.empty() || !program.summary.ends_task ||
      program.summary.pixel_output_mask || program.summary.writes_depth ||
      program.summary.instruction_count != program.instructions.size() ||
      (control && program.summary.vertex_output_mask))
    Fail("program metadata is not complete native TCS/TES");
  bool pending = false, ended = false;
  for (const auto &i : program.instructions) {
    if (!HasCanonicalDerivativeMode(i)) Fail("derivative mode is not canonical for opcode");
    if (!HasCanonicalTextureLodMode(i)) Fail("texture LOD replacement flag is not canonical for opcode");
    if (!HasCanonicalNativeIntegerSignedness(i))
      Fail("integer signedness flag is not canonical for the native opcode");
    if (ended || !i.repeat_count || i.repeat_count > 16 || i.source_count > 4 ||
        i.exec_cnd > 3 || i.writes_predicate > 1)
      Fail("invalid native instruction metadata or bytes after END");
    const bool load = i.opcode == PcoOpcode::kBufferLoad;
    const bool store = i.opcode == PcoOpcode::kBufferStore;
    const bool wdf = i.opcode == PcoOpcode::kWaitDataFence;
    const bool mask = i.opcode == PcoOpcode::kConditionalMask;
    const bool branch = i.opcode == PcoOpcode::kBranch;
    const bool export_vertex = IsWrite(i.opcode) || IsEmit(i.opcode);
    if (!IsAlu(i.opcode) && !load && !store && !wdf && !mask && !branch &&
        !export_vertex && i.opcode != PcoOpcode::kNop)
      Fail("opcode requires unimplemented tessellation functionality");
    if ((control && export_vertex) || (!control && store))
      Fail("TCS UVSW or TES store is outside this stage contract");
    if (pending && !wdf) Fail("native LD/ST is not followed by WDF");
    if (load || store) {
      if (i.repeat_count != 1 || i.source_count != (load ? 2 : 3) ||
          i.data_request || !i.component_count || i.component_count > 16 ||
          i.memory_cache_mode > (load ? 1U : 2U) || i.end_group ||
          i.target != (load ? PcoWriteTarget::kTemporary : PcoWriteTarget::kNone))
        Fail("invalid native LD/ST metadata");
      pending = true;
    } else if (wdf) {
      if (!pending || i.exec_cnd || i.data_request || i.end_group)
        Fail("unmatched or invalid WDF");
      pending = false;
    }
    if (mask) {
      const bool set = i.control_operation == 2, loop = i.control_operation == 3;
      if (i.repeat_count != 1 || i.source_count != (set ? 2 : 1) ||
          i.target != PcoWriteTarget::kTemporary || i.control_operation > 4 ||
          i.control_condition > 3 || (set ? i.immediate != 0 : i.immediate == 0) ||
          i.writes_predicate != (loop ? 1 : 0) ||
          ((loop || i.control_operation == 1) && (i.control_condition || i.exec_cnd != 2)))
        Fail("invalid native execution-mask transition");
    }
    if (branch && (i.branch_condition > 2 || (i.branch_condition && i.exec_cnd) ||
                   i.branch_target_index >= program.instructions.size()))
      Fail("invalid native branch target or condition");
    if (IsWrite(i.opcode) && (i.source_count != 1 || i.target != PcoWriteTarget::kVertexOutput))
      Fail("invalid TES UVSW write");
    if (i.opcode == PcoOpcode::kUvsEmitEndTask &&
        (i.source_count || i.target != PcoWriteTarget::kNone || i.repeat_count != 1))
      Fail("invalid TES UVSW emit/endtask");
    if (i.opcode == PcoOpcode::kNop &&
        (i.source_count || i.target != PcoWriteTarget::kNone || i.repeat_count != 1))
      Fail("NOP carries operands");
    if (i.target == PcoWriteTarget::kPixelOutput ||
        (i.target == PcoWriteTarget::kVertexOutput && !IsWrite(i.opcode)))
      Fail("tessellation has an invalid export destination");
    const auto count = load ? i.component_count : i.repeat_count;
    if ((i.target == PcoWriteTarget::kTemporary && !Fits(i.output_index,count,abi.temps)) ||
        (i.target == PcoWriteTarget::kVertexInput && !Fits(i.output_index,count,abi.vertex_inputs)) ||
        (i.target == PcoWriteTarget::kVertexOutput && !Fits(i.output_index,count,abi.vertex_outputs)))
      Fail("native destination exceeds its declared span");
    if (i.opcode == PcoOpcode::kIntegerAdd64_32 &&
        (i.repeat_count != 1 || i.source_count != 3 ||
         (i.output_target1 == PcoWriteTarget::kTemporary ? !Fits(i.output_index1,1,abi.temps) :
          i.output_target1 == PcoWriteTarget::kVertexInput ? !Fits(i.output_index1,1,abi.vertex_inputs) : true)))
      Fail("ADD64 secondary destination is invalid");
    const std::array<PcoRegisterRef,4> refs{i.source,i.source1,i.source2,i.source3};
    for (unsigned repeat = 0; repeat < i.repeat_count; ++repeat) {
      const auto sr = Packed(i.opcode) ? 0 : repeat;
      for (unsigned source = 0; source < i.source_count; ++source) ValidateSource(refs[source],sr,abi);
      if (i.phase_composed) for (const auto &phase : {i.phase0,i.phase1}) {
        if (phase.source_count > 3) Fail("invalid composed phase source count");
        const std::array<PcoRegisterRef,3> pr{phase.source,phase.source1,phase.source2};
        for (unsigned source = 0; source < phase.source_count; ++source) ValidateSource(pr[source],sr,abi);
      }
    }
    if (store) ValidateSource(i.source2,i.component_count-1,abi);
    if (i.end_group && (pending || (control ? i.opcode != PcoOpcode::kNop : !IsEmit(i.opcode))))
      Fail("native stage END has wrong opcode or unfinished memory");
    if (IsEmit(i.opcode) && !i.end_group) Fail("TES emits without terminating");
    ended = i.end_group;
  }
  if (pending || !ended) Fail("missing resolved native stage END");
}

TessellationTaskState MakeTessellationControlTask(
    const DriverPcoStageAbi &abi, const std::vector<std::uint32_t> &shared,
    std::uint32_t primitive_id, std::uint32_t patch_vertices, std::uint32_t output_vertices) {
  if (!patch_vertices || patch_vertices > 32) Fail("TCS input patch size is invalid");
  auto task = MakeTask(ShaderStage::kTessellationControl,abi,shared,output_vertices);
  for (unsigned lane = 0; lane < task.lane_count; ++lane) {
    task.lanes[lane].inputs[0] = primitive_id;
    task.lanes[lane].inputs[1] = lane;
    task.lanes[lane].inputs[2] = patch_vertices;
    task.lanes[lane].inputs_written = 7;
  }
  return task;
}
TessellationTaskState MakeTessellationEvaluationTask(
    const DriverPcoStageAbi &abi, const std::vector<std::uint32_t> &shared,
    std::uint32_t primitive_id, std::uint32_t patch_vertices,
    const std::array<std::uint32_t,3> *coordinates, std::uint32_t count) {
  if (!coordinates || !patch_vertices || patch_vertices > 32) Fail("TES input patch or coordinates are invalid");
  auto task = MakeTask(ShaderStage::kTessellationEvaluation,abi,shared,count);
  for (unsigned lane = 0; lane < count; ++lane) {
    std::copy(coordinates[lane].begin(),coordinates[lane].end(),task.lanes[lane].inputs.begin());
    task.lanes[lane].inputs[3] = primitive_id;
    task.lanes[lane].inputs[4] = patch_vertices;
    task.lanes[lane].inputs_written = 31;
  }
  return task;
}

void StepTessellationTask(const PcoDecodedProgram &program,
    const DriverPcoStageAbi &abi, TessellationTaskState &task,
    const TessellationMemoryCallbacks &memory, TessellationExecutionStats &stats) {
  if (task.ended || task.stage != program.summary.stage || !task.lane_count ||
      task.lane_count > 32 || task.instruction_index >= program.instructions.size())
    Fail("task stepped outside its native stage program");
  const auto &i = program.instructions[task.instruction_index];
  if (!HasCanonicalDerivativeMode(i)) Fail("derivative mode is not canonical for opcode");
  if (!HasCanonicalTextureLodMode(i)) Fail("texture LOD replacement flag is not canonical for opcode");
  if (!HasCanonicalNativeIntegerSignedness(i))
    Fail("integer signedness flag is not canonical for the native opcode");
  if (++task.steps > UINT64_C(10000000)) Fail("native instruction watchdog");
  ++stats.groups;
  auto next = task.instruction_index + 1;
  if (i.opcode == PcoOpcode::kWaitDataFence) {
    for (unsigned index = 0; index < task.lane_count; ++index) {
      auto &lane = task.lanes[index];
      if (lane.pending_operation || Selected(i,lane)) ++stats.instructions;
      if (lane.pending_operation == 1)
        for (unsigned c = 0; c < lane.pending_count; ++c)
          Write(PcoWriteTarget::kTemporary,lane.pending_output+c,lane.pending_words[c],abi,lane);
      lane.pending_operation = lane.pending_count = 0;
    }
  } else if (i.opcode == PcoOpcode::kBranch) {
    bool any = false, enabled = false;
    for (unsigned index = 0; index < task.lane_count; ++index) {
      any |= Selected(i,task.lanes[index]);
      enabled |= task.lanes[index].execution_predicate != 0;
    }
    const bool take = i.branch_condition == 1 ? !enabled : i.branch_condition == 2 ? enabled :
                      i.exec_cnd == 0 || i.exec_cnd == 2 || any;
    if (take) next = i.branch_target_index;
    ++stats.instructions; ++stats.alu_instructions;
  } else if (i.opcode == PcoOpcode::kConditionalMask && i.control_operation == 3) {
    // CNDLT is a task-wide loop rendezvous: a broken lane remains masked
    // until every lane stops continuing, then outer EMC levels are restored.
    std::array<std::uint32_t,32> old{};
    bool continues = false;
    for (unsigned index = 0; index < task.lane_count; ++index) {
      if (!Selected(i,task.lanes[index]) || task.lanes[index].pending_operation)
        Fail("CNDLT requires all lanes without pending memory");
      old[index] = Read(i.source,0,abi,task,index);
      continues |= old[index] < i.immediate;
    }
    for (unsigned index = 0; index < task.lane_count; ++index) {
      auto &lane = task.lanes[index];
      const bool lane_continues = old[index] < i.immediate;
      const auto value = continues ? (lane_continues ? 0U : old[index]) :
          (old[index] > i.immediate ? old[index]-i.immediate : 0U);
      Write(i.target,i.output_index,value,abi,lane);
      lane.execution_predicate = value == 0;
      lane.predicate = continues && lane_continues;
      ++stats.instructions; ++stats.alu_instructions;
    }
  } else for (unsigned index = 0; index < task.lane_count; ++index) {
    auto &lane = task.lanes[index];
    if (!Selected(i,lane)) continue;
    if (lane.pending_operation) Fail("lane executes before native memory WDF");
    ++stats.instructions;
    if (i.opcode == PcoOpcode::kNop) continue;
    if (i.opcode == PcoOpcode::kConditionalMask) {
      const auto old = Read(i.source,0,abi,task,index);
      auto value = old;
      switch (i.control_operation) {
      case 0:
        if (old || !Condition(i.control_condition,lane)) {
          if (old > UINT32_MAX-i.immediate) Fail("execution mask overflow");
          value += i.immediate;
        }
        break;
      case 1: value = old == 0 ? i.immediate : old == i.immediate ? 0 : old; break;
      case 2:
        if (!old && Condition(i.control_condition,lane)) value = Read(i.source1,0,abi,task,index);
        break;
      case 4: value = old > i.immediate ? old-i.immediate : 0; break;
      default: Fail("unsupported execution mask operation");
      }
      Write(i.target,i.output_index,value,abi,lane);
      lane.execution_predicate = value == 0;
      ++stats.alu_instructions;
      continue;
    }
    if (i.opcode == PcoOpcode::kBufferLoad || i.opcode == PcoOpcode::kBufferStore) {
      const auto address = Read(i.source,0,abi,task,index) |
          (std::uint64_t{Read(i.source1,0,abi,task,index)} << 32U);
      if (i.opcode == PcoOpcode::kBufferLoad) {
        if (!memory.read) Fail("native LD has no modeled memory callback");
        memory.read(memory.user_data,address,i.component_count,lane.pending_words.data());
        lane.pending_output = i.output_index; lane.pending_count = i.component_count;
        lane.pending_operation = 1; ++stats.load_instructions;
      } else {
        if (task.stage != ShaderStage::kTessellationControl || !memory.write)
          Fail("native ST has no TCS patch memory callback");
        std::array<std::uint32_t,16> words{};
        for (unsigned c = 0; c < i.component_count; ++c) words[c] = Read(i.source2,c,abi,task,index);
        memory.write(memory.user_data,address,i.component_count,words.data());
        lane.pending_operation = 2; ++stats.store_instructions;
      }
      ++stats.memory_instructions;
      continue;
    }
    if (IsWrite(i.opcode) || IsEmit(i.opcode)) {
      if (task.stage != ShaderStage::kTessellationEvaluation) Fail("TCS cannot UVSW");
      if (IsWrite(i.opcode)) for (unsigned r = 0; r < i.repeat_count; ++r)
        Write(PcoWriteTarget::kVertexOutput,i.output_index+r,Read(i.source,r,abi,task,index),abi,lane);
      if (IsEmit(i.opcode)) {
        if (lane.emitted) Fail("TES emitted more than one vertex");
        lane.emitted = 1; ++stats.emit_instructions;
      }
      stats.memory_instructions += i.repeat_count;
      continue;
    }
    if (!IsAlu(i.opcode)) Fail("unsupported tessellation ALU opcode");
    for (unsigned repeat = 0; repeat < i.repeat_count; ++repeat) {
      const auto sr = Packed(i.opcode) ? 0 : repeat;
      const std::array<PcoRegisterRef,4> refs{i.source,i.source1,i.source2,i.source3};
      std::array<std::uint32_t,4> raw{};
      for (unsigned s = 0; s < i.source_count; ++s) raw[s] = Read(refs[s],sr,abi,task,index);
      std::array<std::uint32_t,3> p0{},p1{};
      if (i.phase_composed) {
        const auto phase = [&](const PcoPhaseOperation &p, auto &values) {
          const std::array<PcoRegisterRef,3> pr{p.source,p.source1,p.source2};
          for (unsigned s = 0; s < p.source_count; ++s) values[s] = Read(pr[s],sr,abi,task,index);
        };
        phase(i.phase0,p0); phase(i.phase1,p1);
      }
      if (i.opcode == PcoOpcode::kIntegerAdd64_32) {
        const auto base = raw[0] | (std::uint64_t{raw[1]} << 32U);
        const std::uint64_t offset = i.address_offset_signed ?
            static_cast<std::uint64_t>(static_cast<std::int64_t>(static_cast<std::int32_t>(raw[2]))) : raw[2];
        const auto value = base+offset;
        Write(i.target,i.output_index,static_cast<std::uint32_t>(value),abi,lane);
        Write(i.output_target1,i.output_index1,static_cast<std::uint32_t>(value >> 32U),abi,lane);
      } else Write(i.target,i.output_index+repeat,EvaluatePcoAluInstruction(i,raw,repeat,p0,p1),abi,lane);
      if (i.writes_predicate) lane.predicate = EvaluatePcoPredicate(i,raw,p0,p1);
      ++stats.alu_instructions;
    }
  }
  if (i.end_group) {
    for (unsigned index = 0; index < task.lane_count; ++index)
      if (task.lanes[index].pending_operation ||
          (task.stage == ShaderStage::kTessellationEvaluation && !task.lanes[index].emitted))
        Fail("native END has unfinished memory or missing TES emission");
    task.ended = 1;
  }
  task.instruction_index = next;
}
} // namespace pvrgpu::stub
