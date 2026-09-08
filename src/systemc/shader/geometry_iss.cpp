#include "shader/geometry_iss.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace pvrgpu::stub {
namespace {
[[noreturn]] void Fail(const char *reason) {
  throw std::runtime_error(std::string("native geometry: ") + reason);
}
bool Fits(std::uint32_t first, std::uint32_t count, std::uint32_t limit) {
  return first <= limit && count <= limit - first;
}
void ValidateAbi(const DriverPcoStageAbi &abi) {
  if (abi.temps > kPcoTemporaryCount || abi.vertex_inputs != 2 ||
      abi.vertex_outputs < 4 || abi.vertex_outputs > kPcoVertexOutputCount || abi.coefficients ||
      abi.shareds < 4 || abi.shareds > kPcoMaximumSharedCount || abi.entry_offset ||
      abi.uniform_buffer_descriptor_count > 15 ||
      abi.uniform_buffer_descriptor_start != 4 ||
      abi.push_constant_start != 4 + 4 * abi.uniform_buffer_descriptor_count ||
      !Fits(abi.push_constant_start, abi.push_constant_count, abi.shareds) ||
      std::uint64_t{abi.push_constant_start} + abi.push_constant_count != abi.shareds)
    Fail("register or primitive/UBO/push ABI is invalid");
}
bool IsAlu(PcoOpcode opcode) {
  switch (opcode) {
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
  return op == PcoOpcode::kUvsEmit || op == PcoOpcode::kUvsEmitCut ||
         op == PcoOpcode::kUvsEmitEndTask || op == PcoOpcode::kUvsWriteEmitEndTask;
}
bool IsCut(PcoOpcode op) {
  return op == PcoOpcode::kUvsCut || op == PcoOpcode::kUvsEmitCut;
}
bool IsEnd(PcoOpcode op) {
  return op == PcoOpcode::kUvsEndTask || op == PcoOpcode::kUvsEmitEndTask ||
         op == PcoOpcode::kUvsWriteEmitEndTask;
}
bool Packed(PcoOpcode op) {
  return op == PcoOpcode::kFloatUnpackHalf || op == PcoOpcode::kUnpackVector;
}
void ValidateSource(PcoRegisterRef source, unsigned repeat,
                    const DriverPcoStageAbi &abi) {
  const unsigned index = source.index + repeat;
  switch (source.bank) {
  case PcoRegisterBank::kTemporary: if (index < abi.temps) return; break;
  case PcoRegisterBank::kVertexInput: if (index < abi.vertex_inputs) return; break;
  case PcoRegisterBank::kShared: if (index < abi.shareds) return; break;
  case PcoRegisterBank::kSpecial:
    if (!repeat && PcoSpecialConstantBits(source.index, nullptr)) return;
    break;
  default: break;
  }
  Fail("source is outside the declared geometry register file");
}
std::uint32_t Read(PcoRegisterRef source, unsigned repeat,
                  const DriverPcoStageAbi &abi, const GeometryTaskState &task) {
  ValidateSource(source, repeat, abi);
  const unsigned index = source.index + repeat;
  switch (source.bank) {
  case PcoRegisterBank::kTemporary:
    if (!task.temporary_written.test(index)) Fail("TEMP read before write");
    return task.temporaries[index];
  case PcoRegisterBank::kVertexInput:
    if (!(task.inputs_written & (UINT64_C(1) << index))) Fail("VTXIN read before write");
    return task.inputs[index];
  case PcoRegisterBank::kShared: return task.shared[index];
  case PcoRegisterBank::kSpecial: {
    std::uint32_t value = 0;
    if (!PcoSpecialConstantBits(source.index, &value)) Fail("unknown special");
    return value;
  }
  default: Fail("invalid register bank");
  }
}
void Write(PcoWriteTarget target, unsigned index, std::uint32_t value,
           const DriverPcoStageAbi &abi, GeometryTaskState &task) {
  if (target == PcoWriteTarget::kNone) return;
  if (target == PcoWriteTarget::kTemporary && index < abi.temps) {
    task.temporaries[index] = value; task.temporary_written.set(index);
  } else if (target == PcoWriteTarget::kVertexInput && index < abi.vertex_inputs) {
    task.inputs[index] = value; task.inputs_written |= UINT64_C(1) << index;
  } else if (target == PcoWriteTarget::kVertexOutput && index < abi.vertex_outputs) {
    task.outputs[index] = value; task.outputs_written |= UINT64_C(1) << index;
  } else Fail("destination is outside the declared geometry file");
}
bool Selected(const PcoInstruction &i, const GeometryTaskState &task) {
  switch (i.exec_cnd) {
  case 0: return task.execution_predicate;
  case 1: return task.execution_predicate && task.predicate;
  case 2: return true;
  case 3: return task.execution_predicate && !task.predicate;
  default: Fail("invalid execution condition");
  }
}
bool Condition(unsigned condition, const GeometryTaskState &task) {
  switch (condition) {
  case 0: return true;
  case 1: return task.predicate;
  case 2: return false;
  case 3: return !task.predicate;
  default: Fail("invalid execution-mask condition");
  }
}
} // namespace

void ValidateGeometryProgram(const PcoDecodedProgram &program,
                             const DriverPcoStageAbi &abi) {
  ValidateAbi(abi);
  if (program.summary.stage != ShaderStage::kGeometry || program.instructions.empty() ||
      !program.summary.ends_task || program.summary.pixel_output_mask ||
      program.summary.writes_depth || program.summary.instruction_count != program.instructions.size())
    Fail("program is not a complete native geometry program");
  bool pending = false, ended = false;
  for (const auto &i : program.instructions) {
    if (!HasCanonicalNativeIntegerSignedness(i))
      Fail("integer signedness flag is not canonical for the native opcode");
    if (ended || !i.repeat_count || i.repeat_count > 16 || i.source_count > 4 ||
        i.exec_cnd > 3 || i.writes_predicate > 1)
      Fail("invalid instruction metadata or bytes following ENDTASK");
    const bool load = i.opcode == PcoOpcode::kBufferLoad;
    const bool wdf = i.opcode == PcoOpcode::kWaitDataFence;
    const bool mask = i.opcode == PcoOpcode::kConditionalMask;
    const bool branch = i.opcode == PcoOpcode::kBranch;
    if (!IsAlu(i.opcode) && !IsWrite(i.opcode) && !IsEmit(i.opcode) && !IsCut(i.opcode) &&
        !IsEnd(i.opcode) && !load && !wdf && !mask && !branch && i.opcode != PcoOpcode::kNop)
      Fail("opcode requires unimplemented geometry functionality");
    if (pending && !wdf) Fail("native LD is not followed by WDF");
    if (load) {
      if (i.repeat_count != 1 || i.source_count != 2 || i.data_request ||
          !i.component_count || i.component_count > 16 || i.memory_cache_mode > 1 ||
          i.end_group || i.target != PcoWriteTarget::kTemporary)
        Fail("invalid native LD metadata");
      pending = true;
    } else if (wdf) {
      if (!pending || i.exec_cnd || i.data_request) Fail("unmatched WDF");
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
    if (branch && (i.branch_condition > 2 ||
        (i.branch_condition && i.exec_cnd) || i.branch_target_index >= program.instructions.size()))
      Fail("invalid native branch");
    if ((IsEmit(i.opcode) || IsCut(i.opcode) || IsEnd(i.opcode)) && !IsWrite(i.opcode) &&
        (i.source_count || i.repeat_count != 1 || i.target != PcoWriteTarget::kNone))
      Fail("invalid native UVSW control metadata");
    if (IsWrite(i.opcode) && (i.source_count != 1 || i.target != PcoWriteTarget::kVertexOutput))
      Fail("invalid UVSW write");
    if (i.target == PcoWriteTarget::kPixelOutput ||
        (i.target == PcoWriteTarget::kVertexOutput && !IsWrite(i.opcode)))
      Fail("non-UVSW graphics export in geometry program");
    const auto count = load ? i.component_count : i.repeat_count;
    if ((i.target == PcoWriteTarget::kTemporary && !Fits(i.output_index, count, abi.temps)) ||
        (i.target == PcoWriteTarget::kVertexInput && !Fits(i.output_index, count, abi.vertex_inputs)) ||
        (i.target == PcoWriteTarget::kVertexOutput && !Fits(i.output_index, count, abi.vertex_outputs)))
      Fail("native destination exceeds its ABI span");
    if (i.opcode == PcoOpcode::kIntegerAdd64_32 &&
        (i.repeat_count != 1 || i.source_count != 3 ||
         (i.output_target1 == PcoWriteTarget::kTemporary ? !Fits(i.output_index1, 1, abi.temps) :
          i.output_target1 == PcoWriteTarget::kVertexInput ? !Fits(i.output_index1, 1, abi.vertex_inputs) : true)))
      Fail("ADD64 secondary destination is invalid");
    const std::array<PcoRegisterRef,4> refs{i.source, i.source1, i.source2, i.source3};
    for (unsigned repeat = 0; repeat < i.repeat_count; ++repeat) {
      const auto sr = Packed(i.opcode) ? 0 : repeat;
      for (unsigned source = 0; source < i.source_count; ++source) ValidateSource(refs[source], sr, abi);
      if (i.phase_composed) for (const auto &phase : {i.phase0, i.phase1}) {
        if (phase.source_count > 3) Fail("invalid composed phase source count");
        const std::array<PcoRegisterRef,3> pr{phase.source, phase.source1, phase.source2};
        for (unsigned source = 0; source < phase.source_count; ++source) ValidateSource(pr[source], sr, abi);
      }
    }
    if (i.end_group != IsEnd(i.opcode) || (i.end_group && pending))
      Fail("END bit disagrees with native UVSW ENDTASK");
    ended = i.end_group;
  }
  if (pending || !ended) Fail("no resolved native ENDTASK");
}

GeometryTaskState MakeGeometryTask(const DriverPcoStageAbi &abi,
    const std::vector<std::uint32_t> &shared, std::uint32_t primitive_id,
    std::uint32_t invocation_id) {
  ValidateAbi(abi);
  if (shared.size() != abi.shareds || shared[3]) Fail("shared payload disagrees with geometry ABI");
  GeometryTaskState task;
  std::copy(shared.begin(), shared.end(), task.shared.begin());
  if (abi.vertex_inputs) { task.inputs[0] = primitive_id; task.inputs_written = 1; }
  if (abi.vertex_inputs > 1) { task.inputs[1] = invocation_id; task.inputs_written |= 2; }
  return task;
}

void StepGeometryTask(const PcoDecodedProgram &program,
    const DriverPcoStageAbi &abi, GeometryTaskState &task,
    const GeometryExecutionCallbacks &cb, GeometryExecutionStats &stats) {
  if (task.ended || task.instruction_index >= program.instructions.size())
    Fail("task stepped outside its native program");
  const auto &i = program.instructions[task.instruction_index];
  if (!HasCanonicalNativeIntegerSignedness(i))
    Fail("integer signedness flag is not canonical for the native opcode");
  if (++task.steps > UINT64_C(10000000)) Fail("native task instruction watchdog");
  std::uint32_t next = task.instruction_index + 1;
  if (i.opcode == PcoOpcode::kWaitDataFence) {
    if (task.pending_count || Selected(i, task)) ++stats.instructions;
    for (unsigned component = 0; component < task.pending_count; ++component)
      Write(PcoWriteTarget::kTemporary, task.pending_output + component,
            task.pending_words[component], abi, task);
    task.pending_count = 0;
  } else if (i.opcode == PcoOpcode::kBranch) {
    const bool take = i.branch_condition == 1 ? !task.execution_predicate :
                      i.branch_condition == 2 ? task.execution_predicate != 0 :
                      i.exec_cnd == 0 || i.exec_cnd == 2 || Selected(i, task);
    if (take) next = i.branch_target_index;
    ++stats.instructions; ++stats.alu_instructions;
  } else if (Selected(i, task)) {
    if (task.pending_count) Fail("instruction before its LD completion WDF");
    ++stats.instructions;
    if (i.opcode == PcoOpcode::kConditionalMask) {
      const auto old = Read(i.source, 0, abi, task);
      auto value = old;
      switch (i.control_operation) {
      case 0:
        if (old || !Condition(i.control_condition, task)) {
          if (old > UINT32_MAX - i.immediate) Fail("execution-mask counter overflow");
          value += i.immediate;
        }
        break;
      case 1: value = old == 0 ? i.immediate : old == i.immediate ? 0 : old; break;
      case 2:
        if (!old && Condition(i.control_condition, task)) value = Read(i.source1, 0, abi, task);
        break;
      case 3:
        task.predicate = old < i.immediate;
        value = old < i.immediate ? 0 : old > i.immediate ? old - i.immediate : 0;
        break;
      case 4: value = old > i.immediate ? old - i.immediate : 0; break;
      default: Fail("invalid execution-mask operation");
      }
      Write(i.target, i.output_index, value, abi, task);
      task.execution_predicate = value == 0;
      ++stats.alu_instructions;
    } else if (i.opcode == PcoOpcode::kBufferLoad) {
      if (!cb.read) Fail("native LD has no modeled memory callback");
      const auto address = Read(i.source, 0, abi, task) |
          (static_cast<std::uint64_t>(Read(i.source1, 0, abi, task)) << 32U);
      cb.read(cb.user_data, address, i.component_count, task.pending_words.data());
      task.pending_output = i.output_index; task.pending_count = i.component_count;
      ++stats.load_instructions; ++stats.memory_instructions;
    } else if (IsWrite(i.opcode) || IsEmit(i.opcode) || IsCut(i.opcode) || IsEnd(i.opcode)) {
      if (IsWrite(i.opcode)) for (unsigned r = 0; r < i.repeat_count; ++r)
        Write(PcoWriteTarget::kVertexOutput, i.output_index + r, Read(i.source, r, abi, task), abi, task);
      if (IsEmit(i.opcode)) {
        if (!cb.emit) Fail("native EMIT has no owned output storage");
        cb.emit(cb.user_data, task.outputs.data(), abi.vertex_outputs, task.outputs_written);
        // GLSL outputs become undefined after EmitVertex. Preserve raw USC
        // storage, but do not label an old vertex's write as a new write.
        task.outputs_written = 0;
        ++stats.emit_instructions;
      }
      if (IsCut(i.opcode)) {
        if (!cb.cut) Fail("native CUT has no primitive boundary callback");
        cb.cut(cb.user_data); ++stats.cut_instructions;
      }
      stats.memory_instructions += i.repeat_count;
    } else if (i.opcode != PcoOpcode::kNop) {
      if (!IsAlu(i.opcode)) Fail("unsupported geometry ALU instruction");
      for (unsigned r = 0; r < i.repeat_count; ++r) {
        const auto sr = Packed(i.opcode) ? 0 : r;
        const std::array<PcoRegisterRef,4> refs{i.source, i.source1, i.source2, i.source3};
        std::array<std::uint32_t,4> raw{};
        for (unsigned s = 0; s < i.source_count; ++s) raw[s] = Read(refs[s], sr, abi, task);
        std::array<std::uint32_t,3> p0{}, p1{};
        const auto phase = [&](const PcoPhaseOperation &p, auto &values) {
          const std::array<PcoRegisterRef,3> pr{p.source, p.source1, p.source2};
          for (unsigned s = 0; s < p.source_count; ++s) values[s] = Read(pr[s], sr, abi, task);
        };
        if (i.phase_composed) { phase(i.phase0, p0); phase(i.phase1, p1); }
        if (i.opcode == PcoOpcode::kIntegerAdd64_32) {
          const std::uint64_t base = raw[0] | (static_cast<std::uint64_t>(raw[1]) << 32U);
          const std::uint64_t offset = i.address_offset_signed ?
              static_cast<std::uint64_t>(static_cast<std::int64_t>(static_cast<std::int32_t>(raw[2]))) : raw[2];
          const auto value = base + offset;
          Write(i.target, i.output_index, static_cast<std::uint32_t>(value), abi, task);
          Write(i.output_target1, i.output_index1, static_cast<std::uint32_t>(value >> 32U), abi, task);
        } else Write(i.target, i.output_index + r, EvaluatePcoAluInstruction(i, raw, r, p0, p1), abi, task);
        if (i.writes_predicate) task.predicate = EvaluatePcoPredicate(i, raw, p0, p1);
        ++stats.alu_instructions;
      }
    }
  }
  if (i.end_group) {
    if (!IsEnd(i.opcode) || task.pending_count || !cb.finish)
      Fail("ENDTASK has incomplete memory or no output completion callback");
    cb.finish(cb.user_data);
    task.ended = 1;
  }
  task.instruction_index = next;
}
} // namespace pvrgpu::stub
