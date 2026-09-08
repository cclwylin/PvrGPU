// Native geometry task state. Pure instruction semantics are shared, but
// geometry task/export lifetime is independent of VS, FS and compute.
#pragma once

#include "model_types.h"
#include "shader/pco_iss.h"

namespace pvrgpu::stub {

struct GeometryTaskState {
  std::array<std::uint32_t, kPcoTemporaryCount> temporaries{};
  PcoTemporaryMask temporary_written;
  std::array<std::uint32_t, kPcoVertexInputCount> inputs{};
  std::array<std::uint32_t, kPcoVertexOutputCount> outputs{};
  std::array<std::uint32_t, kPcoMaximumSharedCount> shared{};
  std::array<std::uint32_t, kPcoMaximumBufferLoadDwords> pending_words{};
  std::uint64_t inputs_written = 0;
  std::uint64_t outputs_written = 0;
  std::uint64_t steps = 0;
  std::uint32_t instruction_index = 0;
  std::uint32_t pending_output = 0;
  std::uint32_t pending_count = 0;
  std::uint32_t predicate = 0;
  std::uint32_t execution_predicate = 1;
  std::uint32_t ended = 0;
};

struct GeometryExecutionStats {
  std::uint64_t instructions = 0;
  std::uint64_t alu_instructions = 0;
  std::uint64_t memory_instructions = 0;
  std::uint64_t load_instructions = 0;
  std::uint64_t texture_instructions = 0;
  std::uint64_t emit_instructions = 0;
  std::uint64_t cut_instructions = 0;
};

// Stack-only callback pointers: no callback/host pointer is serialized in a
// task or passed through a FIFO. The module owns modeled memory and exports.
struct GeometryExecutionCallbacks {
  void *user_data = nullptr;
  PcoMemoryReadCallback read = nullptr;
  void (*emit)(void *, const std::uint32_t *, std::uint32_t, std::uint64_t) = nullptr;
  void (*cut)(void *) = nullptr;
  void (*finish)(void *) = nullptr;
  void (*sample)(void *, const PcoTextureRequest &, std::uint32_t *) = nullptr;
};

void ValidateGeometryProgram(const PcoDecodedProgram &program,
                              const DriverPcoStageAbi &abi);
GeometryTaskState MakeGeometryTask(const DriverPcoStageAbi &abi,
    const std::vector<std::uint32_t> &shared, std::uint32_t primitive_id,
    std::uint32_t invocation_id);
void StepGeometryTask(const PcoDecodedProgram &program,
    const DriverPcoStageAbi &abi, GeometryTaskState &task,
    const GeometryExecutionCallbacks &callbacks, GeometryExecutionStats &stats);

static_assert(std::is_trivially_copyable_v<GeometryTaskState>);
static_assert(std::is_trivially_copyable_v<GeometryExecutionStats>);

} // namespace pvrgpu::stub
