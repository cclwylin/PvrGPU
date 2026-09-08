// Native compute-only PCO task state and stepping. This is a non-module pure
// interpreter; ComputeShader owns scheduling, FIFO memory and pool lifetime.
#pragma once

#include "compute_types.h"
#include "shader/pco_iss.h"

#include <array>
#include <cstdint>
#include <vector>

namespace pvrgpu::stub {

struct ComputeLaneState {
  std::array<std::uint32_t, kPcoTemporaryCount> temporaries{};
  PcoTemporaryMask temporary_written;
  std::array<std::uint32_t, kPcoVertexInputCount> inputs{};
  std::uint64_t inputs_written = 0;
  std::array<std::uint32_t, kPcoMaximumBufferLoadDwords> pending_words{};
  std::uint32_t pending_output = 0;
  std::uint32_t pending_count = 0;
  std::uint32_t pending_operation = 0; // 0=none, 1=LD, 2=ST, 3=atomic old value
  std::uint32_t predicate = 0;
  std::uint32_t execution_predicate = 1;
  std::uint32_t instance_number = 0;
};

// A task advances one native instruction group for all selected lanes before
// any lane reaches the next group. It never runs a VS or FS executor.
struct ComputeTaskState {
  std::array<ComputeLaneState, kComputeTaskWidth> lanes;
  std::array<std::uint32_t, kPcoMaximumSharedCount> shared{};
  std::array<std::uint32_t, kPcoMaximumVaryingCoefficientCount> coefficients{};
  std::uint32_t lane_count = 0;
  std::uint32_t instruction_index = 0;
  std::uint32_t ended = 0;
  std::uint64_t steps = 0;
  std::uint32_t mutex_held_mask = 0;
  std::uint32_t mutex_sleep_mask = 0;
  std::uint32_t mutex_wakeup_mask = 0;
  std::uint32_t mutex_blocked = 0;
};

using ComputeMemoryRead = void (*)(void *, std::uint64_t,
                                   std::uint32_t, std::uint32_t *);
using ComputeMemoryWrite = void (*)(void *, std::uint64_t,
                                    std::uint32_t, const std::uint32_t *);
using ComputeMemoryAtomic32 = std::uint32_t (*)(void *, ComputeMemoryOperation,
                                               std::uint64_t, std::uint32_t);
using ComputeMutex = void (*)(void *, std::uint32_t, std::uint32_t);
using ComputeTryMutex = bool (*)(void *, std::uint32_t, std::uint32_t);
struct ComputeMemoryCallbacks {
  void *user_data = nullptr;
  ComputeMemoryRead read = nullptr;
  ComputeMemoryWrite write = nullptr;
  ComputeMemoryAtomic32 atomic32 = nullptr;
  ComputeMutex mutex = nullptr;
  ComputeTryMutex try_mutex = nullptr;
};

void ValidateComputeProgram(const PcoDecodedProgram &program,
                            const ComputePcoAbi &abi);
ComputeTaskState MakeComputeTask(
    const ComputePcoAbi &abi, const std::vector<std::uint32_t> &shared,
    const std::array<std::uint32_t, 3> &grid,
    const std::array<std::uint32_t, 3> &group,
    std::uint32_t first_local_index, std::uint32_t lane_count);
void StepComputeTask(const PcoDecodedProgram &program, const ComputePcoAbi &abi,
                     ComputeTaskState &task,
                     const ComputeMemoryCallbacks &memory,
                     ComputeWorkgroupResult &result);

static_assert(std::is_trivially_copyable_v<ComputeLaneState>);
static_assert(std::is_trivially_copyable_v<ComputeTaskState>);

} // namespace pvrgpu::stub
