// Compute-only host and FIFO contracts. No graphics shader executor or
// framebuffer carries compute work. Bulk state remains in MemoryPool.
#pragma once

#include "model_types.h"

#include <array>
#include <cstdint>
#include <ostream>
#include <type_traits>
#include <vector>

namespace pvrgpu::stub {

struct ComputePcoAbi {
  DriverPcoStageAbi stage;
  std::array<std::uint32_t, 3> local_size{};
  std::uint32_t local_invocation_index_start = 0;
  std::uint32_t local_invocation_index_count = 0;
  std::uint32_t workgroup_id_start = 0;
  std::uint32_t workgroup_id_count = 0;
  std::uint32_t num_workgroups_start = 0;
  std::uint32_t num_workgroups_count = 0;
  std::uint32_t storage_buffer_descriptor_start = 0;
  std::uint32_t storage_buffer_descriptor_count = 0;
  std::uint32_t uniform_buffer_used_mask = 0;
  std::uint32_t storage_buffer_used_mask = 0;
  std::uint32_t storage_buffer_read_mask = 0;
  std::uint32_t storage_buffer_write_mask = 0;
  std::uint32_t shared_memory_bytes = 0;
  std::uint32_t scratch_bytes = 0;
};

struct ModelComputeResource {
  std::vector<std::uint8_t> bytes;
  bool writable = false;
};

struct ModelComputeBinding {
  std::uint32_t kind = 0;
  std::uint32_t slot = 0;
  std::uint32_t resource_index = 0;
  std::uint32_t access = 0;
  std::uint64_t offset = 0;
  std::uint64_t bytes_size = 0;
};

struct ModelComputeDispatch {
  ComputePcoAbi abi;
  std::vector<std::uint8_t> binary;
  std::array<std::uint32_t, 3> grid{};
  std::array<std::uint32_t, 3> block{};
  std::vector<std::uint32_t> push_words;
  std::vector<ModelComputeResource> resources;
  std::vector<ModelComputeBinding> bindings;
  MemoryMode memory_mode = MemoryMode::kDirect;
};

struct ModelComputeStats {
  std::uint64_t workgroups = 0;
  std::uint64_t invocations = 0;
  std::uint64_t alu_instructions = 0;
  std::uint64_t memory_instructions = 0;
  std::uint64_t atomic_instructions = 0;
  std::uint64_t load_instructions = 0;
  std::uint64_t store_instructions = 0;
  std::uint64_t dram_read_bytes = 0;
  std::uint64_t dram_write_bytes = 0;
  std::uint64_t direct_read_bytes = 0;
  std::uint64_t direct_write_bytes = 0;
  std::uint64_t readback_bytes = 0;
  std::uint64_t pool_allocations = 0;
  std::uint64_t pool_releases = 0;
};

inline constexpr std::uint32_t kComputeAccessRead = 1;
inline constexpr std::uint32_t kComputeAccessWrite = 2;
inline constexpr std::uint32_t kComputeTaskWidth = 32;

// A binding view, not a backing BO. Overlapping views intentionally preserve
// their shared GPU address; each complete access must fit a permitted view.
struct ComputeBufferRange {
  std::uint64_t gpu_address = 0;
  std::uint64_t bytes = 0;
  std::uint32_t access = 0;
  std::uint32_t binding = 0;
  std::uint32_t kind = 0;
};

// Host/session owns this state and its three input handles through completion.
// CDM is its sole writer. ComputeShader reads it but publishes workgroup
// counters separately, so memory-service updates cannot be overwritten.
struct ComputeDispatchState {
  ComputePcoAbi abi;
  std::array<std::uint32_t, 3> grid{};
  std::uint64_t sequence = 0;
  PoolHandle code;
  PoolHandle shared_registers;
  PoolHandle buffer_ranges;
  CounterTxn counters;
  ModelComputeStats stats;
  std::uint64_t instructions_executed = 0;
  std::uint32_t failed = 0;
  std::array<char, 256> error{};
};

struct ComputeDispatchTxn {
  PoolHandle state;
  std::uint64_t sequence = 0;
};

// CDM owns result, allocates one workgroup at a time and releases it after the
// matching completion. Shader-owned lane state never crosses a FIFO by value.
struct ComputeWorkgroupResult {
  ModelComputeStats stats;
  std::uint64_t instructions_executed = 0;
  std::uint32_t failed = 0;
  std::array<char, 256> error{};
};

struct ComputeWorkgroupTxn {
  PoolHandle state;
  PoolHandle result;
  std::array<std::uint32_t, 3> group{};
  std::uint64_t ordinal = 0;
};

enum class ComputeMemoryOperation : std::uint32_t {
  kRead = 0,
  kWrite = 1,
  kAtomicAdd32 = 2,
  kAtomicSub32,
  kAtomicExchange32,
  kAtomicUnsignedMin32,
  kAtomicSignedMin32,
  kAtomicUnsignedMax32,
  kAtomicSignedMax32,
  kAtomicAnd32,
  kAtomicOr32,
  kAtomicXor32,
  kMutexLock,
  kMutexRelease,
  // An internal task-failure cleanup, not an invented shader instruction.
  kMutexCleanup,
};

inline bool IsComputeMutexOperation(ComputeMemoryOperation operation) {
  return operation == ComputeMemoryOperation::kMutexLock ||
         operation == ComputeMemoryOperation::kMutexRelease ||
         operation == ComputeMemoryOperation::kMutexCleanup;
}

// Keep the permission/size rules identical for every genuine DMA RMW.
// An explicit whitelist also rejects unknown values crossing a FIFO.
inline bool IsComputeAtomic32(ComputeMemoryOperation operation) {
  switch (operation) {
  case ComputeMemoryOperation::kAtomicAdd32:
  case ComputeMemoryOperation::kAtomicSub32:
  case ComputeMemoryOperation::kAtomicExchange32:
  case ComputeMemoryOperation::kAtomicUnsignedMin32:
  case ComputeMemoryOperation::kAtomicSignedMin32:
  case ComputeMemoryOperation::kAtomicUnsignedMax32:
  case ComputeMemoryOperation::kAtomicSignedMax32:
  case ComputeMemoryOperation::kAtomicAnd32:
  case ComputeMemoryOperation::kAtomicOr32:
  case ComputeMemoryOperation::kAtomicXor32:
    return true;
  default:
    return false;
  }
}

// Request payload ownership transfers to CDM (ST/atomic operand); response
// payload ownership transfers to ComputeShader (LD bytes, atomic old DWORD,
// or a failed=1 error string).
// Every accepted request produces exactly one response with the same ID.
struct ComputeMemoryTxn {
  PoolHandle state;
  PoolHandle payload;
  // Pool generation identifies the resident task owning a native MUTEX.
  PoolHandle task;
  std::uint64_t request_id = 0;
  std::uint64_t address = 0;
  std::uint32_t bytes = 0;
  ComputeMemoryOperation operation = ComputeMemoryOperation::kRead;
  std::uint32_t failed = 0;
};

inline std::ostream &operator<<(std::ostream &out,
                                const ComputeDispatchTxn &txn) {
  return out << "compute dispatch " << txn.sequence << " state=" << txn.state;
}
inline std::ostream &operator<<(std::ostream &out,
                                const ComputeWorkgroupTxn &txn) {
  return out << "compute workgroup " << txn.ordinal << " state=" << txn.state;
}
inline std::ostream &operator<<(std::ostream &out,
                                const ComputeMemoryTxn &txn) {
  return out << "compute memory " << txn.request_id << " bytes=" << txn.bytes;
}

static_assert(std::is_trivially_copyable_v<ComputePcoAbi>);
static_assert(std::is_trivially_copyable_v<ComputeBufferRange>);
static_assert(std::is_trivially_copyable_v<ComputeDispatchState>);
static_assert(std::is_trivially_copyable_v<ComputeWorkgroupResult>);
static_assert(std::is_trivially_copyable_v<ComputeDispatchTxn>);
static_assert(std::is_trivially_copyable_v<ComputeWorkgroupTxn>);
static_assert(std::is_trivially_copyable_v<ComputeMemoryTxn>);
static_assert(sizeof(ComputeDispatchTxn) <= 32);
static_assert(sizeof(ComputeWorkgroupTxn) <= 48);
static_assert(sizeof(ComputeMemoryTxn) <= 64);

} // namespace pvrgpu::stub
