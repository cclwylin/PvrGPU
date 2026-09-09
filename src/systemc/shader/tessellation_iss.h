// Native TCS/TES tasks. Stages share pure ALU semantics and the public PCO
// encoding, never VS/GS/compute task state or execution entry points.
#pragma once

#include "common/tessellation_state.h"
#include "shader/pco_iss.h"

namespace pvrgpu::stub {

struct TessellationLaneState {
  std::array<std::uint32_t, kPcoTemporaryCount> temporaries{};
  PcoTemporaryMask temporary_written;
  std::array<std::uint32_t, kPcoVertexInputCount> inputs{};
  std::array<std::uint32_t, kPcoVertexOutputCount> outputs{};
  std::array<std::uint32_t, kPcoMaximumBufferLoadDwords> pending_words{};
  std::uint64_t inputs_written = 0;
  std::uint64_t outputs_written = 0;
  std::uint32_t pending_output = 0;
  std::uint32_t pending_count = 0;
  std::uint32_t pending_operation = 0; // 0=none, 1=LD, 2=ST, 3=SMP
  std::uint32_t predicate = 0;
  std::uint32_t execution_predicate = 1;
  std::uint32_t emitted = 0;
};
struct TessellationTaskState {
  std::array<TessellationLaneState, kTessellationTaskWidth> lanes;
  std::array<std::uint32_t, kPcoMaximumSharedCount> shared{};
  ShaderStage stage = ShaderStage::kTessellationControl;
  std::uint32_t lane_count = 0;
  std::uint32_t instruction_index = 0;
  std::uint32_t ended = 0;
  std::uint64_t steps = 0;
};
struct TessellationExecutionStats {
  std::uint64_t instructions = 0;
  std::uint64_t groups = 0;
  std::uint64_t alu_instructions = 0;
  std::uint64_t memory_instructions = 0;
  std::uint64_t load_instructions = 0;
  std::uint64_t store_instructions = 0;
  std::uint64_t emit_instructions = 0;
  std::uint64_t texture_instructions = 0;
};
struct TessellationMemoryCallbacks {
  void *user_data = nullptr;
  PcoMemoryReadCallback read = nullptr;
  void (*write)(void *, std::uint64_t, std::uint32_t, const std::uint32_t *) = nullptr;
  void (*sample)(void *, const PcoTextureRequest &, std::uint32_t *) = nullptr;
};

void ValidateTessellationProgram(const PcoDecodedProgram &program,
                                 const DriverPcoStageAbi &abi);
TessellationTaskState MakeTessellationControlTask(
    const DriverPcoStageAbi &abi, const std::vector<std::uint32_t> &shared,
    std::uint32_t primitive_id, std::uint32_t patch_vertices,
    std::uint32_t output_vertices);
TessellationTaskState MakeTessellationEvaluationTask(
    const DriverPcoStageAbi &abi, const std::vector<std::uint32_t> &shared,
    std::uint32_t primitive_id, std::uint32_t patch_vertices,
    const std::array<std::uint32_t, 3> *coordinates, std::uint32_t count);
void StepTessellationTask(const PcoDecodedProgram &program,
    const DriverPcoStageAbi &abi, TessellationTaskState &task,
    const TessellationMemoryCallbacks &memory, TessellationExecutionStats &stats);

static_assert(std::is_trivially_copyable_v<TessellationLaneState>);
static_assert(std::is_trivially_copyable_v<TessellationTaskState>);
static_assert(std::is_trivially_copyable_v<TessellationExecutionStats>);
} // namespace pvrgpu::stub
