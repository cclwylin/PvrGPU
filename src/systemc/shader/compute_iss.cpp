#include "shader/compute_iss.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace pvrgpu::stub {
namespace {

[[noreturn]] void Fail(const char *reason) {
  throw std::runtime_error(std::string("native compute: ") + reason);
}

bool Fits(std::uint32_t first, std::uint32_t count, std::uint32_t limit) {
  return first <= limit && count <= limit - first;
}

bool InRange(std::uint32_t index, std::uint32_t first, std::uint32_t count) {
  return index >= first && index - first < count;
}

void ValidateAbi(const ComputePcoAbi &abi) {
  const auto &stage = abi.stage;
  if (stage.temps > kPcoTemporaryCount ||
      stage.vertex_inputs > kPcoVertexInputCount || stage.vertex_outputs != 0 ||
      stage.coefficients > kPcoMaximumVaryingCoefficientCount ||
      stage.shareds > kPcoMaximumSharedCount || stage.entry_offset != 0 ||
      abi.shared_memory_bytes > kComputeMaximumSharedBytes ||
      (abi.shared_memory_bytes & 3U) || abi.scratch_bytes != 0)
    Fail("unsupported register, entry point, shared-memory or scratch ABI");
  if (abi.local_invocation_index_count > 1 ||
      !Fits(abi.local_invocation_index_start, abi.local_invocation_index_count,
            stage.vertex_inputs) ||
      (abi.workgroup_id_count != 0 && abi.workgroup_id_count != 3) ||
      (abi.num_workgroups_count != 0 && abi.num_workgroups_count != 3) ||
      !Fits(abi.workgroup_id_start, abi.workgroup_id_count, stage.coefficients) ||
      !Fits(abi.num_workgroups_start, abi.num_workgroups_count, stage.coefficients))
    Fail("compute system-value register span is invalid");
  if (abi.workgroup_id_count && abi.num_workgroups_count &&
      abi.workgroup_id_start < abi.num_workgroups_start + abi.num_workgroups_count &&
      abi.num_workgroups_start < abi.workgroup_id_start + abi.workgroup_id_count)
    Fail("compute system-value coefficient spans overlap");
  const std::uint32_t private_count = abi.shared_memory_bytes ? 4U : 0U;
  const std::uint32_t user_prefix = 4U *
      (stage.uniform_buffer_descriptor_count + abi.storage_buffer_descriptor_count);
  const std::uint32_t image_end = user_prefix + 8U * abi.image_descriptor_count;
  if (abi.image_descriptor_count > 32 ||
      abi.image_descriptor_start != (abi.image_descriptor_count ? user_prefix : 0U))
    Fail("compute image descriptor layout is invalid");
  if (abi.shared_memory_descriptor_count != private_count ||
      abi.shared_memory_descriptor_start != (private_count ? image_end : 0U))
    Fail("compute private workgroup descriptor layout is invalid");
  if (stage.uniform_buffer_descriptor_count > 15 ||
      abi.storage_buffer_descriptor_count > 32 ||
      stage.uniform_buffer_descriptor_start != 0 ||
      abi.storage_buffer_descriptor_start !=
          stage.uniform_buffer_descriptor_count * 4U ||
      stage.push_constant_start != image_end + private_count ||
      !Fits(stage.push_constant_start, stage.push_constant_count, stage.shareds))
    Fail("compute shared-register descriptor/push layout is invalid");
  const auto mask_fits = [](std::uint32_t mask, std::uint32_t count) {
    return count == 32 || (mask >> count) == 0;
  };
  if (!mask_fits(abi.uniform_buffer_used_mask,
                 stage.uniform_buffer_descriptor_count) ||
      !mask_fits(abi.storage_buffer_used_mask, abi.storage_buffer_descriptor_count) ||
      !mask_fits(abi.image_used_mask, abi.image_descriptor_count) ||
      ((abi.image_read_mask | abi.image_write_mask) & ~abi.image_used_mask) ||
      ((abi.storage_buffer_read_mask | abi.storage_buffer_write_mask) &
       ~abi.storage_buffer_used_mask) != 0)
    Fail("compute resource mask exceeds its declared descriptor span");
  std::uint32_t invocations = 1;
  for (auto extent : abi.local_size) {
    if (!extent || extent > 1024U / invocations)
      Fail("compute local size exceeds the modeled bound");
    invocations *= extent;
  }
}

bool IsAlu(PcoOpcode opcode) {
  switch (opcode) {
  case PcoOpcode::kMoveBypass:
  case PcoOpcode::kFloatNegate:
  case PcoOpcode::kFloatAbs:
  case PcoOpcode::kMoveImmediate:
  case PcoOpcode::kFloatFloor:
  case PcoOpcode::kFloatSubtract:
  case PcoOpcode::kFloatGreaterEqual:
  case PcoOpcode::kFloatEqual:
  case PcoOpcode::kFloatLess:
  case PcoOpcode::kBooleanCompare:
  case PcoOpcode::kConditionalSelect:
  case PcoOpcode::kConditionalSelectNegateTrue:
  case PcoOpcode::kTestConditionalSelect:
  case PcoOpcode::kFloatAdd:
  case PcoOpcode::kFloatAddNegateSource0:
  case PcoOpcode::kFloatMultiply:
  case PcoOpcode::kFloatMad:
  case PcoOpcode::kFloatMadNegateSource2:
  case PcoOpcode::kFloatMadNegateSource0:
  case PcoOpcode::kFloatMadNegateSource0Source2:
  case PcoOpcode::kFloatMin:
  case PcoOpcode::kFloatMax:
  case PcoOpcode::kIntegerMaxSigned:
  case PcoOpcode::kIntegerMinSigned:
  case PcoOpcode::kReciprocal:
  case PcoOpcode::kReciprocalSquareRoot:
  case PcoOpcode::kFloatLog2:
  case PcoOpcode::kFloatExp2:
  case PcoOpcode::kIntegerAdd:
  case PcoOpcode::kIntegerMultiplyAdd32:
  case PcoOpcode::kIntegerMultiplyAdd64High:
  case PcoOpcode::kBitfieldInsert:
  case PcoOpcode::kBitfieldExtractUnsigned:
  case PcoOpcode::kBitfieldExtractSigned:
  case PcoOpcode::kIntegerAdd64_32:
  case PcoOpcode::kBitwiseAnd:
  case PcoOpcode::kBitwiseOr:
  case PcoOpcode::kBitwiseXor:
  case PcoOpcode::kBitwiseXnor:
  case PcoOpcode::kShiftRight:
  case PcoOpcode::kShiftLeft:
  case PcoOpcode::kTestZero:
  case PcoOpcode::kFloatSine:
  case PcoOpcode::kFloatCosine:
  case PcoOpcode::kPackHalf2x16:
  case PcoOpcode::kUnpackHalf2x16:
  case PcoOpcode::kFloatPackHalfRtne:
  case PcoOpcode::kFloatPackHalfRtz:
  case PcoOpcode::kFloatToUint32Rtne:
  case PcoOpcode::kFloatToUint32Rtz:
  case PcoOpcode::kFloatToInt32Rtne:
  case PcoOpcode::kFloatToInt32Rtz:
  case PcoOpcode::kFloatUnpackHalf:
  case PcoOpcode::kUnpackUnsignedToFloat:
  case PcoOpcode::kUnpackSignedToFloat:
  case PcoOpcode::kUnpackVector:
  case PcoOpcode::kConditionalSelectGreaterZero:
    return true;
  default:
    return false;
  }
}

void ValidateSource(PcoRegisterRef source, std::uint32_t repeat,
                    const ComputePcoAbi &abi) {
  const std::uint32_t index = source.index + repeat;
  switch (source.bank) {
  case PcoRegisterBank::kTemporary:
    if (index < abi.stage.temps) return;
    break;
  case PcoRegisterBank::kVertexInput:
    if (index < abi.stage.vertex_inputs) return;
    break;
  case PcoRegisterBank::kShared:
    if (index < abi.stage.shareds) return;
    break;
  case PcoRegisterBank::kCoefficient:
    if (index < abi.stage.coefficients &&
        (InRange(index, abi.workgroup_id_start, abi.workgroup_id_count) ||
         InRange(index, abi.num_workgroups_start, abi.num_workgroups_count)))
      return;
    break;
  case PcoRegisterBank::kSpecial: {
    std::uint32_t ignored = 0;
    if (!repeat && source.index == kPcoSpecialInstanceNumber) return;
    if (!repeat && PcoSpecialConstantBits(source.index, &ignored)) return;
    break;
  }
  default:
    break;
  }
  Fail("source register is outside the declared compute file/system values");
}

std::uint32_t Read(PcoRegisterRef source, std::uint32_t repeat,
                    const ComputePcoAbi &abi, const ComputeTaskState &task,
                    const ComputeLaneState &lane) {
  ValidateSource(source, repeat, abi);
  const std::uint32_t index = source.index + repeat;
  switch (source.bank) {
  case PcoRegisterBank::kTemporary:
    if (!lane.temporary_written.test(index))
      Fail("TEMP was read before being written");
    return lane.temporaries[index];
  case PcoRegisterBank::kVertexInput:
    if ((lane.inputs_written & (UINT64_C(1) << index)) == 0)
      Fail("VTXIN was read before a primitive system value or ALU write");
    return lane.inputs[index];
  case PcoRegisterBank::kShared: return task.shared[index];
  case PcoRegisterBank::kCoefficient: return task.coefficients[index];
  case PcoRegisterBank::kSpecial: {
    if (source.index == kPcoSpecialInstanceNumber) {
      if (lane.instance_number >= kComputeTaskWidth) Fail("compute instance number exceeds physical task width");
      return lane.instance_number;
    }
    std::uint32_t bits = 0;
    if (!PcoSpecialConstantBits(source.index, &bits)) Fail("unsupported special");
    return bits;
  }
  default: Fail("unsupported compute register bank");
  }
}

void Write(PcoWriteTarget target, std::uint32_t index, std::uint32_t value,
           const ComputePcoAbi &abi, ComputeLaneState &lane) {
  if (target == PcoWriteTarget::kNone) return;
  if (target == PcoWriteTarget::kTemporary && index < abi.stage.temps) {
    lane.temporaries[index] = value;
    lane.temporary_written.set(index);
  } else if (target == PcoWriteTarget::kVertexInput &&
             index < abi.stage.vertex_inputs) {
    lane.inputs[index] = value;
    lane.inputs_written |= UINT64_C(1) << index;
  } else {
    Fail("compute attempted an invalid or graphics register write");
  }
}

bool Selected(const PcoInstruction &instruction, const ComputeLaneState &lane) {
  switch (instruction.exec_cnd) {
  case 0: return lane.execution_predicate != 0;
  case 1: return lane.execution_predicate != 0 && lane.predicate != 0;
  case 2: return true;
  case 3: return lane.execution_predicate != 0 && lane.predicate == 0;
  default: Fail("invalid execution condition");
  }
}

bool Condition(std::uint32_t condition, const ComputeLaneState &lane) {
  switch (condition) {
  case 0: return true;
  case 1: return lane.predicate != 0;
  case 2: return false;
  case 3: return lane.predicate == 0;
  default: Fail("invalid conditional-mask condition");
  }
}

bool RepeatsPackedWord(PcoOpcode opcode) {
  return opcode == PcoOpcode::kFloatUnpackHalf || opcode == PcoOpcode::kUnpackVector;
}

ComputeMemoryOperation AtomicMemoryOperation(PcoOpcode opcode) {
  switch (opcode) {
  case PcoOpcode::kAtomicAdd32: return ComputeMemoryOperation::kAtomicAdd32;
  case PcoOpcode::kAtomicSub32: return ComputeMemoryOperation::kAtomicSub32;
  case PcoOpcode::kAtomicExchange32: return ComputeMemoryOperation::kAtomicExchange32;
  case PcoOpcode::kAtomicUnsignedMin32: return ComputeMemoryOperation::kAtomicUnsignedMin32;
  case PcoOpcode::kAtomicSignedMin32: return ComputeMemoryOperation::kAtomicSignedMin32;
  case PcoOpcode::kAtomicUnsignedMax32: return ComputeMemoryOperation::kAtomicUnsignedMax32;
  case PcoOpcode::kAtomicSignedMax32: return ComputeMemoryOperation::kAtomicSignedMax32;
  case PcoOpcode::kAtomicAnd32: return ComputeMemoryOperation::kAtomicAnd32;
  case PcoOpcode::kAtomicOr32: return ComputeMemoryOperation::kAtomicOr32;
  case PcoOpcode::kAtomicXor32: return ComputeMemoryOperation::kAtomicXor32;
  default: Fail("native atomic operation is unsupported");
  }
}

} // namespace

void ValidateComputeProgram(const PcoDecodedProgram &program,
                            const ComputePcoAbi &abi) {
  ValidateAbi(abi);
  if (program.summary.stage != ShaderStage::kCompute ||
      program.instructions.empty() || !program.summary.ends_task ||
      program.summary.vertex_output_mask || program.summary.pixel_output_mask ||
      program.summary.writes_depth ||
      program.summary.instruction_count != program.instructions.size())
    Fail("program is not a complete native compute program");
  bool pending = false;
  bool end = false;
  for (const auto &instruction : program.instructions) {
    if (!HasCanonicalTextureLodMode(instruction)) Fail("texture LOD replacement flag is not canonical for opcode");
    if (!HasCanonicalNativeIntegerSignedness(instruction))
      Fail("integer signedness flag is not canonical for the native opcode");
    if (!instruction.repeat_count || instruction.repeat_count > 16 ||
        instruction.source_count > 4 || instruction.exec_cnd > 3 ||
        instruction.writes_predicate > 1)
      Fail("invalid instruction operand or repeat metadata");
    if (pending && instruction.opcode != PcoOpcode::kWaitDataFence)
      Fail("native memory request is not followed by its WDF");
    const bool load = instruction.opcode == PcoOpcode::kBufferLoad;
    const bool store = instruction.opcode == PcoOpcode::kBufferStore;
    const bool atomic = IsPcoAtomic32(instruction.opcode);
    const bool mask = instruction.opcode == PcoOpcode::kConditionalMask;
    const bool branch = instruction.opcode == PcoOpcode::kBranch;
    const bool wdf = instruction.opcode == PcoOpcode::kWaitDataFence;
    const bool mutex = instruction.opcode == PcoOpcode::kMutex;
    if (!IsAlu(instruction.opcode) && !load && !store && !atomic && !mask && !branch &&
        !wdf && !mutex && instruction.opcode != PcoOpcode::kNop)
      Fail("native opcode requires unimplemented compute functionality");
    if (mutex && (instruction.source_count || instruction.repeat_count != 1 ||
                  instruction.exec_cnd || instruction.end_group || instruction.writes_predicate ||
                  instruction.target != PcoWriteTarget::kNone || instruction.immediate >= 16 ||
                  instruction.control_operation > 3))
      Fail("invalid native MUTEX metadata");
    if (load || store || atomic) {
      if (instruction.repeat_count != 1 || instruction.data_request != 0 ||
          instruction.component_count == 0 || instruction.component_count > 16 ||
          instruction.source_count != (load ? 2U : 3U) ||
          instruction.memory_cache_mode > (load ? 1U : 2U) ||
          instruction.end_group)
        Fail("invalid native LD/ST metadata");
      if (atomic && (instruction.component_count != 1 ||
                     instruction.memory_cache_mode != 0 ||
                     instruction.target != PcoWriteTarget::kTemporary))
        Fail("invalid native atomic32 metadata");
      pending = true;
    } else if (wdf) {
      if (!pending || instruction.data_request != 0 || instruction.exec_cnd != 0)
        Fail("unmatched native WDF");
      pending = false;
    }
    if (mask) {
      const bool set = instruction.control_operation == 2;
      const bool loop = instruction.control_operation == 3;
      if (instruction.repeat_count != 1 ||
          instruction.source_count != (set ? 2U : 1U) ||
          (instruction.target != PcoWriteTarget::kTemporary &&
           instruction.target != PcoWriteTarget::kVertexInput) ||
          instruction.control_operation > 4 || instruction.control_condition > 3 ||
          (set ? instruction.immediate != 0 : instruction.immediate == 0) ||
          instruction.writes_predicate != (loop ? 1U : 0U) ||
          ((loop || instruction.control_operation == 1) &&
           (instruction.control_condition != 0 || instruction.exec_cnd != 2)))
        Fail("unsupported native execution-mask transition");
    }
    if (branch && (instruction.branch_condition > 2 ||
                   (instruction.branch_condition != 0 && instruction.exec_cnd != 0) ||
                   instruction.branch_target_index >= program.instructions.size()))
      Fail("unsupported native branch target or condition");
    if (instruction.target != PcoWriteTarget::kNone &&
        instruction.target != PcoWriteTarget::kTemporary &&
        instruction.target != PcoWriteTarget::kVertexInput)
      Fail("compute program contains a graphics export");
    const std::uint32_t output_count = load ? instruction.component_count :
                                              instruction.repeat_count;
    if ((instruction.target == PcoWriteTarget::kTemporary &&
         !Fits(instruction.output_index, output_count, abi.stage.temps)) ||
        (instruction.target == PcoWriteTarget::kVertexInput &&
         !Fits(instruction.output_index, output_count, abi.stage.vertex_inputs)))
      Fail("native destination exceeds its declared register span");
    if (instruction.opcode == PcoOpcode::kIntegerAdd64_32 &&
        (instruction.repeat_count != 1 || instruction.source_count != 3 ||
         (instruction.target != PcoWriteTarget::kTemporary &&
          instruction.target != PcoWriteTarget::kVertexInput) ||
         (instruction.output_target1 == PcoWriteTarget::kTemporary ?
              !Fits(instruction.output_index1, 1, abi.stage.temps) :
          instruction.output_target1 == PcoWriteTarget::kVertexInput ?
              !Fits(instruction.output_index1, 1, abi.stage.vertex_inputs) : true)))
      Fail("ADD64_32 destination pair exceeds its declared register spans");
    const std::array<PcoRegisterRef,4> sources{
        instruction.source, instruction.source1, instruction.source2,
        instruction.source3};
    for (std::uint32_t repeat = 0; repeat < instruction.repeat_count; ++repeat) {
      const auto source_repeat = RepeatsPackedWord(instruction.opcode) ? 0 : repeat;
      for (unsigned i = 0; i < instruction.source_count; ++i)
        ValidateSource(sources[i], source_repeat, abi);
      if (instruction.phase_composed) {
        for (const auto &phase : {instruction.phase0, instruction.phase1}) {
          if (phase.source_count > 3) Fail("invalid composed phase source count");
          const std::array<PcoRegisterRef,3> refs{phase.source, phase.source1,
                                                 phase.source2};
          for (unsigned i = 0; i < phase.source_count; ++i)
            ValidateSource(refs[i], source_repeat, abi);
        }
      }
    }
    if (store)
      ValidateSource(instruction.source2, instruction.component_count - 1, abi);
    end |= instruction.end_group != 0;
    if (instruction.end_group && pending) Fail("END has an unfinished request");
  }
  if (pending || !end) Fail("compute program has no resolved native END");
}

ComputeTaskState MakeComputeTask(
    const ComputePcoAbi &abi, const std::vector<std::uint32_t> &shared,
    const std::array<std::uint32_t, 3> &grid,
    const std::array<std::uint32_t, 3> &group,
    std::uint32_t first_local_index, std::uint32_t lane_count) {
  ValidateAbi(abi);
  const auto local_count = abi.local_size[0] * abi.local_size[1] * abi.local_size[2];
  if (!lane_count || lane_count > kComputeTaskWidth ||
      first_local_index % kComputeTaskWidth != 0 || first_local_index >= local_count ||
      lane_count != std::min(kComputeTaskWidth, local_count - first_local_index) ||
      shared.size() != abi.stage.shareds)
    Fail("task lane/shared payload disagrees with its dispatch ABI");
  ComputeTaskState task;
  // Physical tasks have 32 lanes, including the tail. The compiler's native
  // instance guard masks excess local indices; do not replace it on the host.
  task.lane_count = kComputeTaskWidth;
  std::copy(shared.begin(), shared.end(), task.shared.begin());
  for (unsigned axis = 0; axis < 3; ++axis) {
    if (grid[axis] == 0 || group[axis] >= grid[axis])
      Fail("workgroup ID is outside its dispatch grid");
    if (abi.workgroup_id_count)
      task.coefficients[abi.workgroup_id_start + axis] = group[axis];
    if (abi.num_workgroups_count)
      task.coefficients[abi.num_workgroups_start + axis] = grid[axis];
  }
  for (unsigned lane = 0; lane < task.lane_count; ++lane) {
    task.lanes[lane].instance_number = lane;
    if (abi.local_invocation_index_count) {
      task.lanes[lane].inputs[abi.local_invocation_index_start] = first_local_index + lane;
      task.lanes[lane].inputs_written = UINT64_C(1) << abi.local_invocation_index_start;
    }
  }
  return task;
}

void StepComputeTask(const PcoDecodedProgram &program, const ComputePcoAbi &abi,
                     ComputeTaskState &task, const ComputeMemoryCallbacks &memory,
                     ComputeWorkgroupResult &result) {
  if (task.ended || !task.lane_count || task.lane_count > kComputeTaskWidth ||
      task.instruction_index >= program.instructions.size())
    Fail("task stepped outside its native program");
  const auto &instruction = program.instructions[task.instruction_index];
  if (!HasCanonicalTextureLodMode(instruction)) Fail("texture LOD replacement flag is not canonical for opcode");
  if (!HasCanonicalNativeIntegerSignedness(instruction))
    Fail("integer signedness flag is not canonical for the native opcode");
  // Finite watchdog bounds runaway native control flow; it never substitutes
  // a shader result or silently declares an unfinished invocation complete.
  if (++task.steps > UINT64_C(10000000)) Fail("native task instruction limit");
  std::uint32_t next = task.instruction_index + 1;
  if (instruction.opcode == PcoOpcode::kWaitDataFence) {
    // pco_cf.c can predicate an LD/ST but deliberately never its WDF. DRC
    // completion belongs to the task: commit each issued response exactly
    // once, and a lane that issued no request simply has nothing to wait for.
    for (unsigned i = 0; i < task.lane_count; ++i) {
      auto &lane = task.lanes[i];
      if (lane.pending_operation || Selected(instruction, lane)) {
        ++result.instructions_executed;
        ++result.stats.memory_instructions;
      }
      if (lane.pending_operation == 1 || lane.pending_operation == 3) {
        for (unsigned component = 0; component < lane.pending_count; ++component)
          Write(PcoWriteTarget::kTemporary, lane.pending_output + component,
                lane.pending_words[component], abi, lane);
      }
      lane.pending_operation = lane.pending_count = 0;
    }
  } else if (instruction.opcode == PcoOpcode::kMutex) {
    // MUTEX is task control, not 32 separate lock operations. Real ownership
    // lives across the CDM FIFO; this mask also rejects an unterminated END.
    if (!memory.mutex && !memory.try_mutex)
      Fail("native MUTEX has no modeled ownership callback");
    if (instruction.immediate >= 16) Fail("native MUTEX ID exceeds its field");
    for (const auto &lane : task.lanes)
      if (lane.pending_operation) Fail("MUTEX precedes a pending request WDF");
    const auto bit = UINT32_C(1) << instruction.immediate;
    task.mutex_blocked = 0;
    task.mutex_wakeup_mask = 0;
    const auto exchange = [&](unsigned operation) {
      if (memory.try_mutex)
        return memory.try_mutex(memory.user_data, instruction.immediate, operation);
      memory.mutex(memory.user_data, instruction.immediate, operation);
      return true;
    };
    if (instruction.control_operation == 3) {
      if (task.mutex_held_mask & bit) Fail("native MUTEX is not reentrant");
      if (!exchange(3)) {
        task.mutex_blocked = 1;
        return; // Contention neither advances PC nor fabricates an instruction.
      }
      task.mutex_held_mask |= bit;
    } else if (instruction.control_operation <= 2) {
      if (!(task.mutex_held_mask & bit)) Fail("native MUTEX release has no matching lock");
      if (!exchange(instruction.control_operation))
        Fail("native MUTEX release was blocked");
      task.mutex_held_mask &= ~bit;
      if (instruction.control_operation == 1) task.mutex_sleep_mask |= bit;
      if (instruction.control_operation == 2) task.mutex_wakeup_mask |= bit;
    } else {
      Fail("native MUTEX operation is invalid");
    }
    ++result.instructions_executed;
  } else if (instruction.opcode == PcoOpcode::kBranch) {
    bool any = false, any_enabled = false;
    for (unsigned i = 0; i < task.lane_count; ++i) {
      const bool selected = Selected(instruction, task.lanes[i]);
      any |= selected;
      any_enabled |= task.lanes[i].execution_predicate != 0;
    }
    // pco_legalize.c lower_vote explicitly defines ALLINST as no enabled
    // instances and ANYINST as at least one. Its following unqualified CC
    // branch must still execute when all instances are masked.
    bool take = false;
    if (instruction.branch_condition == 1) {
      take = !any_enabled;
    } else if (instruction.branch_condition == 2) {
      take = any_enabled;
    } else if (instruction.branch_condition == 0) {
      take = instruction.exec_cnd == 0 || instruction.exec_cnd == 2 || any;
    } else {
      Fail("unsupported native branch condition");
    }
    if (take) next = instruction.branch_target_index;
    ++result.instructions_executed;
    ++result.stats.alu_instructions;
  } else if (instruction.opcode == PcoOpcode::kConditionalMask &&
             instruction.control_operation == 3) {
    // pco_cf.c lower_loop reserves two EMC levels (continue/break). A loop
    // cannot release its broken lanes while any task instance still loops:
    // they must retain their EMC until the task-wide rendezvous at CNDLT.
    // Keep this reduction before every lane write, not inside a lane loop.
    std::array<std::uint32_t,kComputeTaskWidth> old{};
    bool continues = false;
    for (unsigned i = 0; i < task.lane_count; ++i) {
      auto &lane = task.lanes[i];
      if (!Selected(instruction, lane) || lane.pending_operation)
        Fail("CNDLT requires all task lanes with no pending memory request");
      old[i] = Read(instruction.source, 0, abi, task, lane);
      continues |= old[i] < instruction.immediate;
    }
    for (unsigned i = 0; i < task.lane_count; ++i) {
      auto &lane = task.lanes[i];
      const bool lane_continues = old[i] < instruction.immediate;
      const auto value = continues ? (lane_continues ? 0U : old[i]) :
          (old[i] > instruction.immediate ? old[i] - instruction.immediate : 0U);
      Write(instruction.target, instruction.output_index, value, abi, lane);
      lane.execution_predicate = value == 0;
      lane.predicate = continues && lane_continues;
      ++result.instructions_executed;
      ++result.stats.alu_instructions;
    }
  } else {
    for (unsigned lane_index = 0; lane_index < task.lane_count; ++lane_index) {
      auto &lane = task.lanes[lane_index];
      if (!Selected(instruction, lane)) continue;
      ++result.instructions_executed;
      const bool load = instruction.opcode == PcoOpcode::kBufferLoad;
      const bool store = instruction.opcode == PcoOpcode::kBufferStore;
      const bool atomic = IsPcoAtomic32(instruction.opcode);
      if (load || store || atomic)
        ++result.stats.memory_instructions;
      else
        result.stats.alu_instructions += instruction.repeat_count;
      if (lane.pending_operation)
        Fail("lane executes before its native memory request WDF");
      if (instruction.opcode == PcoOpcode::kNop) continue;
      if (instruction.opcode == PcoOpcode::kConditionalMask) {
        const auto old = Read(instruction.source, 0, abi, task, lane);
        std::uint32_t value = old;
        if (instruction.control_operation == 0) {
          if (old != 0 || !Condition(instruction.control_condition, lane)) {
            if (old > std::numeric_limits<std::uint32_t>::max() - instruction.immediate)
              Fail("execution mask counter overflow");
            value += instruction.immediate;
          }
        } else if (instruction.control_operation == 1) {
          // Flip only the current if/else level. Outer masks and a loop
          // break/continue counter must survive this inner else transition.
          value = old == 0 ? instruction.immediate :
                  old == instruction.immediate ? 0U : old;
        } else if (instruction.control_operation == 2) {
          // The set-value may be an active-lane-only TEMP. Do not read it
          // from a masked lane whose preceding MOVI did not execute.
          if (old == 0 && Condition(instruction.control_condition, lane))
            value = Read(instruction.source1, 0, abi, task, lane);
        } else if (instruction.control_operation == 4) {
          value = old > instruction.immediate ? old - instruction.immediate : 0;
        } else {
          Fail("unsupported execution mask operation");
        }
        Write(instruction.target, instruction.output_index, value, abi, lane);
        lane.execution_predicate = value == 0;
        continue;
      }
      if (load || store || atomic) {
        const std::uint64_t address = Read(instruction.source, 0, abi, task, lane) |
            (static_cast<std::uint64_t>(Read(instruction.source1, 0, abi, task, lane)) << 32U);
        if (load) {
          if (!memory.read) Fail("native LD has no modeled memory callback");
          memory.read(memory.user_data, address, instruction.component_count,
                      lane.pending_words.data());
          lane.pending_operation = 1;
          lane.pending_output = instruction.output_index;
          lane.pending_count = instruction.component_count;
          ++result.stats.load_instructions;
        } else if (store) {
          if (!memory.write) Fail("native ST has no modeled memory callback");
          std::array<std::uint32_t,16> values{};
          for (unsigned i = 0; i < instruction.component_count; ++i)
            values[i] = Read(instruction.source2, i, abi, task, lane);
          memory.write(memory.user_data, address, instruction.component_count, values.data());
          lane.pending_operation = 2;
          ++result.stats.store_instructions;
        } else {
          if (!memory.atomic32) Fail("native atomic has no modeled RMW callback");
          const auto operand = Read(instruction.source2, 0, abi, task, lane);
          lane.pending_words[0] = memory.atomic32(
              memory.user_data, AtomicMemoryOperation(instruction.opcode), address, operand);
          lane.pending_operation = 3;
          lane.pending_output = instruction.output_index;
          lane.pending_count = 1;
          ++result.stats.atomic_instructions;
        }
        continue;
      }
      if (!IsAlu(instruction.opcode)) Fail("unsupported compute ALU instruction");
      for (unsigned repeat = 0; repeat < instruction.repeat_count; ++repeat) {
        const auto source_repeat = RepeatsPackedWord(instruction.opcode) ? 0 : repeat;
        const std::array<PcoRegisterRef,4> refs{instruction.source, instruction.source1,
                                                instruction.source2, instruction.source3};
        std::array<std::uint32_t,4> raw{};
        for (unsigned i = 0; i < instruction.source_count; ++i)
          raw[i] = Read(refs[i], source_repeat, abi, task, lane);
        std::array<std::uint32_t,3> phase0{}, phase1{};
        if (instruction.phase_composed) {
          const auto read_phase = [&](const PcoPhaseOperation &phase,
                                       std::array<std::uint32_t,3> &values) {
            const std::array<PcoRegisterRef,3> phase_refs{phase.source, phase.source1, phase.source2};
            for (unsigned i = 0; i < phase.source_count; ++i)
              values[i] = Read(phase_refs[i], source_repeat, abi, task, lane);
          };
          read_phase(instruction.phase0, phase0);
          read_phase(instruction.phase1, phase1);
        }
        if (instruction.opcode == PcoOpcode::kIntegerAdd64_32) {
          const std::uint64_t base = raw[0] | (static_cast<std::uint64_t>(raw[1]) << 32U);
          const std::uint64_t offset = instruction.address_offset_signed ?
              static_cast<std::uint64_t>(static_cast<std::int64_t>(static_cast<std::int32_t>(raw[2]))) : raw[2];
          const auto value = base + offset;
          Write(instruction.target, instruction.output_index,
                static_cast<std::uint32_t>(value), abi, lane);
          Write(instruction.output_target1, instruction.output_index1,
                static_cast<std::uint32_t>(value >> 32U), abi, lane);
        } else {
          // The shared helper is a pure ALU operation, not a VS/FS executor.
          const auto value = EvaluatePcoAluInstruction(instruction, raw,
              static_cast<std::uint8_t>(repeat), phase0, phase1);
          Write(instruction.target, instruction.output_index + repeat, value, abi, lane);
        }
        if (instruction.writes_predicate)
          lane.predicate = EvaluatePcoPredicate(instruction, raw, phase0, phase1);
      }
    }
  }
  if (instruction.end_group) {
    if (task.mutex_held_mask) Fail("native END leaves a MUTEX owned by the task");
    for (unsigned i = 0; i < task.lane_count; ++i)
      if (task.lanes[i].pending_operation) Fail("native END precedes WDF");
    task.ended = 1;
  }
  task.instruction_index = next;
}

} // namespace pvrgpu::stub
