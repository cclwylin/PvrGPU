// CDM owns dispatch accounting and services compute LD/ST through the same
// modeled memory system as graphics. No host-side shader substitute exists.
#include "data_master/compute_data_master.h"

#include "common/functional_types.h"
#include "common/pipeline_state.h"
#include "memory/gpu_memory_system.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace pvrgpu::stub {
namespace {

template <typename T>
T ReadPod(const MemoryPool &pool, PoolHandle handle) {
  const auto &bytes = pool.Read(handle);
  if (bytes.size() != sizeof(T))
    throw std::runtime_error("compute POD payload size mismatch");
  T value;
  std::memcpy(&value, bytes.data(), sizeof(value));
  return value;
}

template <typename T>
void WritePod(MemoryPool &pool, PoolHandle handle, const T &value) {
  auto &bytes = pool.Write(handle);
  if (bytes.size() != sizeof(T))
    throw std::runtime_error("compute POD payload size mismatch");
  std::memcpy(bytes.data(), &value, sizeof(value));
}

void SetError(ComputeDispatchState &state, const char *message) {
  state.failed = 1;
  const std::size_t bytes = std::min(std::strlen(message), state.error.size()-1);
  std::memcpy(state.error.data(), message, bytes);
  state.error[bytes] = 0;
}

bool SameHandle(PoolHandle a, PoolHandle b) {
  return a.slot == b.slot && a.generation == b.generation;
}

struct ValidatedMemoryAccess {
  std::size_t valid_bytes = 0;
};

ValidatedMemoryAccess ValidateMemoryAccess(
    const ComputeMemoryTxn &txn,
    const std::vector<ComputeBufferRange> &ranges) {
  if (txn.bytes == 0 || txn.bytes > 16U * sizeof(std::uint32_t) ||
      txn.bytes % sizeof(std::uint32_t) != 0 || txn.address % 4U != 0)
    throw std::runtime_error("compute memory access has invalid size/alignment");
  std::uint32_t required = 0;
  if (txn.operation == ComputeMemoryOperation::kRead)
    required = kComputeAccessRead;
  else if (txn.operation == ComputeMemoryOperation::kWrite)
    required = kComputeAccessWrite;
  else if (IsComputeAtomic32(txn.operation)) {
    if (txn.bytes != sizeof(std::uint32_t))
      throw std::runtime_error("compute atomic32 requires exactly one DWORD");
    // The complete atomic access must be authorized by one read/write view.
    // A read-only alias plus a separate write-only alias is not such a view.
    required = kComputeAccessRead | kComputeAccessWrite;
  }
  else
    throw std::runtime_error("compute memory operation is unsupported");
  bool starts_in_a_view = false;
  std::uint64_t maximum_remaining = 0;
  for (const auto &range : ranges) {
    if (txn.address < range.gpu_address)
      continue;
    const std::uint64_t offset = txn.address - range.gpu_address;
    if (offset >= range.bytes)
      continue;
    starts_in_a_view = true;
    if ((range.access & required) == required)
      maximum_remaining = std::max(maximum_remaining, range.bytes - offset);
  }
  // A request against a bound view with the wrong access mask is a malformed
  // shader/ABI contract, not dynamic OOB.  Keep rejecting it (including an
  // atomic whose read/write permissions are split across aliases).
  if (maximum_remaining == 0 && starts_in_a_view)
    throw std::runtime_error("compute memory access is not permitted by its bound view");

  // Dynamic addresses outside every view are robust: reads are zero-filled,
  // stores are discarded and atomics return zero.  A vector beginning inside
  // a view retains its in-range DWORD prefix and suppresses only the tail.
  const std::uint64_t aligned_remaining = maximum_remaining & ~UINT64_C(3);
  return {static_cast<std::size_t>(
      std::min<std::uint64_t>(txn.bytes, aligned_remaining))};
}

std::uint32_t AtomicValue(ComputeMemoryOperation operation, std::uint32_t old,
                          std::uint32_t operand) {
  // Compare signed DWORDs by biasing the sign bit: no signed overflow or
  // implementation-defined conversion of values above INT32_MAX is needed.
  const bool signed_less = (old ^ UINT32_C(0x80000000)) <
                           (operand ^ UINT32_C(0x80000000));
  switch (operation) {
  case ComputeMemoryOperation::kAtomicAdd32: return old + operand;
  case ComputeMemoryOperation::kAtomicSub32: return old - operand;
  case ComputeMemoryOperation::kAtomicExchange32: return operand;
  case ComputeMemoryOperation::kAtomicUnsignedMin32: return std::min(old, operand);
  case ComputeMemoryOperation::kAtomicSignedMin32: return signed_less ? old : operand;
  case ComputeMemoryOperation::kAtomicUnsignedMax32: return std::max(old, operand);
  case ComputeMemoryOperation::kAtomicSignedMax32: return signed_less ? operand : old;
  case ComputeMemoryOperation::kAtomicAnd32: return old & operand;
  case ComputeMemoryOperation::kAtomicOr32: return old | operand;
  case ComputeMemoryOperation::kAtomicXor32: return old ^ operand;
  default: throw std::runtime_error("compute atomic operation is unsupported");
  }
}

} // namespace

ComputeDataMaster::ComputeDataMaster(sc_core::sc_module_name name,
                                     MemoryPool &pool, GpuMemorySystem &memory)
    : sc_module(name), pool_(pool), memory_(memory) {
  SC_THREAD(DispatchRun);
  SC_THREAD(MemoryRun);
}

void ComputeDataMaster::DispatchRun() {
  for (;;) {
    ComputeDispatchTxn txn;
    while (!input.nb_read(txn))
      wait(input.data_written_event());
    PoolHandle result_handle{};
    try {
      auto state = ReadPod<ComputeDispatchState>(pool_, txn.state);
      if (txn.sequence != state.sequence)
        throw std::runtime_error("compute dispatch sequence mismatch");
      std::uint64_t groups = 1;
      std::uint64_t lanes = 1;
      for (unsigned axis = 0; axis < 3; ++axis) {
        if (state.abi.local_size[axis] == 0 ||
            lanes > 1024U / state.abi.local_size[axis])
          throw std::runtime_error("compute local size exceeds modeled bound");
        lanes *= state.abi.local_size[axis];
        if (state.grid[axis] > 65535U ||
            (state.grid[axis] != 0 &&
             groups > std::numeric_limits<std::uint64_t>::max() / state.grid[axis]))
          throw std::runtime_error("compute grid extent is invalid");
        groups *= state.grid[axis];
      }
      if (groups > std::numeric_limits<std::uint64_t>::max() / lanes)
        throw std::runtime_error("compute invocation count overflows");
      for (std::uint64_t ordinal = 0; ordinal < groups; ++ordinal) {
        if (state.abi.shared_memory_bytes) {
          if (state.abi.shared_memory_bytes > kComputeMaximumSharedBytes ||
              (state.abi.shared_memory_bytes & 3U))
            throw std::runtime_error("compute shared backing exceeds 32 KiB");
          // Workgroup allocation, not shader execution. Zero is a permitted
          // initial value for undefined GLSL shared storage and initializes
          // the compiler's barrier counters. HostWrite invalidates stale cache.
          const std::vector<std::uint8_t> zero(state.abi.shared_memory_bytes, 0);
          memory_.HostWrite(kComputeSharedAddress, zero.data(), zero.size());
        }
        ComputeWorkgroupResult result;
        result_handle = pool_.Allocate(sizeof(result));
        WritePod(pool_, result_handle, result);
        ComputeWorkgroupTxn work;
        work.state = txn.state;
        work.result = result_handle;
        work.ordinal = ordinal;
        work.group[0] = static_cast<std::uint32_t>(ordinal % state.grid[0]);
        work.group[1] = static_cast<std::uint32_t>(
            (ordinal / state.grid[0]) % state.grid[1]);
        work.group[2] = static_cast<std::uint32_t>(
            ordinal / state.grid[0] / state.grid[1]);
        while (!workgroup_output.nb_write(work))
          wait(workgroup_output.data_read_event());
        ComputeWorkgroupTxn done;
        while (!workgroup_completion.nb_read(done))
          wait(workgroup_completion.data_written_event());
        if (!SameHandle(done.state, work.state) ||
            !SameHandle(done.result, work.result) || done.ordinal != ordinal ||
            done.group != work.group)
          throw std::runtime_error("compute workgroup completion mismatch");
        result = ReadPod<ComputeWorkgroupResult>(pool_, result_handle);
        pool_.Release(result_handle);
        result_handle = {};
        // The memory thread updated this state while the shader was running.
        state = ReadPod<ComputeDispatchState>(pool_, txn.state);
        if (result.failed)
          throw std::runtime_error(result.error.data());
        state.stats.workgroups += result.stats.workgroups;
        state.stats.invocations += result.stats.invocations;
        state.stats.alu_instructions += result.stats.alu_instructions;
        state.stats.memory_instructions += result.stats.memory_instructions;
        state.stats.atomic_instructions += result.stats.atomic_instructions;
        state.stats.load_instructions += result.stats.load_instructions;
        state.stats.store_instructions += result.stats.store_instructions;
        state.instructions_executed += result.instructions_executed;
        WritePod(pool_, txn.state, state);
      }
    } catch (const std::exception &error) {
      if (HasPoolHandle(result_handle))
        pool_.Release(result_handle);
      auto state = ReadPod<ComputeDispatchState>(pool_, txn.state);
      SetError(state, error.what());
      WritePod(pool_, txn.state, state);
    }
    while (!completion.nb_write(txn))
      wait(completion.data_read_event());
  }
}

void ComputeDataMaster::MemoryRun() {
  for (;;) {
    ComputeMemoryTxn request;
    while (!memory_input.nb_read(request))
      wait(memory_input.data_written_event());
    ComputeMemoryTxn response = request;
    response.payload = {};
    response.failed = 0;
    MemoryAccessStats stats;
    try {
      const auto state = ReadPod<ComputeDispatchState>(pool_, request.state);
      if (IsComputeMutexOperation(request.operation)) {
        response.blocked = !ApplyMutex(request);
      } else {
        auto ranges = LoadArray<ComputeBufferRange>(pool_, state.buffer_ranges);
        if (state.abi.shared_memory_bytes)
          ranges.push_back({kComputeSharedAddress, state.abi.shared_memory_bytes,
                            kComputeAccessRead | kComputeAccessWrite, 0, 2});
        const auto access = ValidateMemoryAccess(request, ranges);
        if (request.operation == ComputeMemoryOperation::kRead) {
          if (HasPoolHandle(request.payload))
            throw std::runtime_error("compute read request unexpectedly owns bytes");
          std::vector<std::uint8_t> result(request.bytes, 0);
          if (access.valid_bytes != 0) {
            const auto read = memory_.Read(request.address, access.valid_bytes,
                                            MemoryClient::kComputeShader);
            if (read.data.size() != access.valid_bytes)
              throw std::runtime_error("compute read returned incomplete data");
            std::copy(read.data.begin(), read.data.end(), result.begin());
            stats = read.stats;
          }
          response.payload = StoreNewArray(pool_, result);
        } else if (request.operation == ComputeMemoryOperation::kWrite) {
          const auto &bytes = pool_.Read(request.payload);
          if (bytes.size() != request.bytes)
            throw std::runtime_error("compute store payload size mismatch");
          if (access.valid_bytes != 0)
            stats = memory_.Write(request.address, bytes.data(),
                                  access.valid_bytes,
                                  MemoryClient::kComputeShader);
        } else {
          const auto &operand_bytes = pool_.Read(request.payload);
          if (operand_bytes.size() != sizeof(std::uint32_t))
            throw std::runtime_error("compute atomic operand payload size mismatch");
          std::uint32_t operand = 0;
          std::memcpy(&operand, operand_bytes.data(), sizeof(operand));
          if (access.valid_bytes == 0) {
            response.payload =
                StoreNewArray(pool_, std::vector<std::uint32_t>{0});
          } else {
          // This service thread owns the whole RMW. Neither memory operation
          // yields; FIFO requests and other SystemC processes cannot interleave
          // between this read and write. Both are real modeled memory accesses.
          const auto read = memory_.Read(request.address, sizeof(std::uint32_t),
                                          MemoryClient::kComputeShader);
          std::uint32_t old = 0;
          std::memcpy(&old, read.data.data(), sizeof(old));
          const std::uint32_t value = AtomicValue(request.operation, old, operand);
          stats = read.stats;
          stats += memory_.Write(request.address, &value, sizeof(value),
                                   MemoryClient::kComputeShader);
          response.payload = StoreNewArray(pool_, std::vector<std::uint32_t>{old});
          }
        }
      }
    } catch (const std::exception &error) {
      response.failed = 1;
      const auto *begin = reinterpret_cast<const std::uint8_t *>(error.what());
      response.payload = StoreNewArray(
          pool_, std::vector<std::uint8_t>(begin, begin + std::strlen(error.what()) + 1));
    }
    if (HasPoolHandle(request.payload))
      pool_.Release(request.payload);
    auto state = ReadPod<ComputeDispatchState>(pool_, request.state);
    ApplyMemoryAccessStats(state.counters, stats);
    state.stats.dram_read_bytes += stats.dram_read_bytes;
    state.stats.dram_write_bytes += stats.dram_write_bytes;
    state.stats.direct_read_bytes += stats.direct_read_bytes;
    state.stats.direct_write_bytes += stats.direct_write_bytes;
    WritePod(pool_, request.state, state);
    const std::uint64_t cycles = MemoryAccessDelayCycles(stats);
    if (cycles != 0)
      WaitForCycles(cycles);
    while (!memory_output.nb_write(response))
      wait(memory_output.data_read_event());
  }
}

bool ComputeDataMaster::ApplyMutex(const ComputeMemoryTxn &request) {
  if (request.bytes || HasPoolHandle(request.payload) || !HasPoolHandle(request.task) ||
      (request.operation == ComputeMemoryOperation::kMutexCleanup ? request.address != 0 : request.address >= 16))
    throw std::runtime_error("compute MUTEX has invalid ID, task or payload");
  // Generation-checked task liveness prevents a recycled pool slot from
  // releasing a different task's lock. Dispatch identity is checked as well.
  (void)pool_.Read(request.task);
  const auto owns = [&](const MutexOwner &owner) {
    return SameHandle(owner.task, request.task) && SameHandle(owner.dispatch, request.state);
  };
  if (request.operation == ComputeMemoryOperation::kMutexCleanup) {
    for (auto &owner : mutex_owners_)
      if (owns(owner)) owner = {};
    return true;
  }
  auto &owner = mutex_owners_[request.address];
  if (request.operation == ComputeMemoryOperation::kMutexLock) {
    if (HasPoolHandle(owner.task)) {
      if (owns(owner)) throw std::runtime_error("compute MUTEX lock is not reentrant");
      return false;
    }
    owner = {request.state,request.task};
  } else if (request.operation == ComputeMemoryOperation::kMutexRelease) {
    if (!owns(owner)) throw std::runtime_error("compute MUTEX release is not owned by this task");
    owner = {};
  } else {
    throw std::runtime_error("compute MUTEX operation is unsupported");
  }
  return true;
}

}  // namespace pvrgpu::stub
