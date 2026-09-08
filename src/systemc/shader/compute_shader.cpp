#include "shader/compute_shader.h"

#include "common/functional_types.h"
#include "common/pipeline_state.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace pvrgpu::stub {
namespace {

bool SameHandle(PoolHandle a, PoolHandle b) {
  return a.slot == b.slot && a.generation == b.generation;
}

template <typename T>
T ReadPod(const MemoryPool &pool, PoolHandle handle) {
  const auto &bytes = pool.Read(handle);
  if (bytes.size() != sizeof(T))
    throw std::runtime_error("compute shader POD payload size mismatch");
  T value;
  std::memcpy(&value, bytes.data(), sizeof(value));
  return value;
}

template <typename T>
void WritePod(MemoryPool &pool, PoolHandle handle, const T &value) {
  auto &bytes = pool.Write(handle);
  if (bytes.size() != sizeof(T))
    throw std::runtime_error("compute shader POD payload size mismatch");
  std::memcpy(bytes.data(), &value, sizeof(value));
}

} // namespace

ComputeShader::ComputeShader(sc_core::sc_module_name name, MemoryPool &pool)
    : sc_module(name), pool_(pool) {
  SC_THREAD(Run);
}

void ComputeShader::SampleTexture(void *context, const PcoTextureRequest &issued,
                                  std::uint32_t *response) {
  auto &self = *static_cast<ComputeShader *>(context);
  const auto dispatch = ReadPod<ComputeDispatchState>(self.pool_, self.current_dispatch_);
  if (!response || !self.texture_request_output.size() || !self.texture_response_input.size() ||
      !HasPoolHandle(dispatch.texture_state))
    throw std::runtime_error("compute SMP FIFO/state is unbound");
  auto state = LoadPipelineState(self.pool_, dispatch.texture_state);
  if (!UsesTextureSampling(state, ShaderStage::kCompute) ||
      HasPoolHandle(state.texture_sample_requests) || HasPoolHandle(state.texture_sample_responses))
    throw std::runtime_error("compute SMP rendezvous payload is already occupied");
  TextureSampleRequest request;
  request.shader_stage = ShaderStage::kCompute;
  std::copy_n(issued.coordinates.begin(), 3, request.coordinates);
  std::copy_n(issued.texture_state.begin(), 4, request.texture_state);
  std::copy_n(issued.sampler_state.begin(), 4, request.sampler_state);
  request.texture_address_lo = issued.texture_address_lo;
  request.texture_address_hi = issued.texture_address_hi;
  request.coordinate_count = issued.coordinate_count;
  request.component_count = issued.component_count;
  request.descriptor_set = issued.descriptor_set;
  request.binding = issued.binding;
  request.dimension = issued.dimension;
  request.normalized = issued.normalized;
  request.fcnorm = issued.fcnorm;
  request.data_request = issued.data_request;
  request.explicit_lod = issued.explicit_lod;
  request.explicit_lod_present = issued.explicit_lod_present;
  state.texture_sample_requests = StoreNewArray(self.pool_, std::vector<TextureSampleRequest>{request});
  state.stage = PipelineStage::kComputeTexturePending;
  StorePipelineState(self.pool_, dispatch.texture_state, state);
  PipelineTxn txn;
  txn.state = dispatch.texture_state;
  txn.sequence = dispatch.sequence;
  self.texture_request_output->write(txn);
  const auto completion = self.texture_response_input->read();
  if (!SameHandle(completion.state, txn.state) || completion.sequence != txn.sequence || completion.frame != txn.frame)
    throw std::runtime_error("compute SMP response identity mismatch");
  state = LoadPipelineState(self.pool_, txn.state);
  RequireStage(state.stage, PipelineStage::kComputeTextureSamplesReady, self.name());
  const auto requests = LoadArray<TextureSampleRequest>(self.pool_, state.texture_sample_requests);
  const auto responses = LoadArray<TextureSampleResponse>(self.pool_, state.texture_sample_responses);
  if (requests.size() != 1 || responses.size() != 1 ||
      !std::equal(std::begin(requests[0].coordinates), std::end(requests[0].coordinates), std::begin(request.coordinates)) ||
      !std::equal(std::begin(requests[0].texture_state), std::end(requests[0].texture_state), std::begin(request.texture_state)) ||
      !std::equal(std::begin(requests[0].sampler_state), std::end(requests[0].sampler_state), std::begin(request.sampler_state)) ||
      requests[0].shader_stage != request.shader_stage ||
      requests[0].descriptor_set != request.descriptor_set || requests[0].binding != request.binding ||
      requests[0].texture_address_lo != request.texture_address_lo ||
      requests[0].texture_address_hi != request.texture_address_hi ||
      requests[0].explicit_lod != request.explicit_lod ||
      requests[0].explicit_lod_present != request.explicit_lod_present ||
      requests[0].dimension != request.dimension || requests[0].normalized != request.normalized ||
      requests[0].fcnorm != request.fcnorm || requests[0].coordinate_count != request.coordinate_count ||
      requests[0].component_count != request.component_count || requests[0].data_request != request.data_request ||
      responses[0].shader_stage != ShaderStage::kCompute ||
      responses[0].shader_lane_index || responses[0].request_id)
    throw std::runtime_error("compute SMP completion ordering/payload mismatch");
  std::copy_n(responses[0].rgba, 4, response);
  self.pool_.Release(state.texture_sample_requests); state.texture_sample_requests = {};
  self.pool_.Release(state.texture_sample_responses); state.texture_sample_responses = {};
  StorePipelineState(self.pool_, txn.state, state);
}

ComputeMemoryTxn ComputeShader::ExchangeMemory(
    std::uint64_t address, std::uint32_t count,
    ComputeMemoryOperation operation, const std::uint32_t *words) {
  const bool mutex = IsComputeMutexOperation(operation);
  if (mutex ? (count != 0 || words || !HasPoolHandle(current_task_)) :
      (count == 0 || count > 16 ||
      (operation != ComputeMemoryOperation::kRead && !words) ||
      (IsComputeAtomic32(operation) && count != 1) ||
      (operation != ComputeMemoryOperation::kRead &&
       operation != ComputeMemoryOperation::kWrite &&
       !IsComputeAtomic32(operation))))
    throw std::runtime_error("compute shader issued invalid memory count");
  ComputeMemoryTxn request;
  request.state = current_dispatch_;
  request.task = current_task_;
  request.address = address;
  request.bytes = count * sizeof(std::uint32_t);
  request.operation = operation;
  request.request_id = ++next_request_id_;
  if (!mutex && operation != ComputeMemoryOperation::kRead) {
    request.payload = pool_.Allocate(request.bytes);
    auto &bytes = pool_.Write(request.payload);
    std::memcpy(bytes.data(), words, request.bytes);
  }
  while (!memory_request_output.nb_write(request))
    wait(memory_request_output.data_read_event());
  // CDM now owns request.payload, including on a failed store.
  ComputeMemoryTxn response;
  while (!memory_response_input.nb_read(response))
    wait(memory_response_input.data_written_event());
  if (!SameHandle(response.state, request.state) ||
      !SameHandle(response.task, request.task) ||
      response.request_id != request.request_id ||
      response.address != request.address || response.bytes != request.bytes ||
      response.operation != request.operation) {
    if (HasPoolHandle(response.payload))
      pool_.Release(response.payload);
    throw std::runtime_error("compute memory response identity mismatch");
  }
  if (response.failed) {
    std::string message = "compute memory request failed";
    if (HasPoolHandle(response.payload)) {
      const auto &bytes = pool_.Read(response.payload);
      if (!bytes.empty() && bytes.back() == 0)
        message.assign(reinterpret_cast<const char *>(bytes.data()),
                       bytes.size() - 1);
      pool_.Release(response.payload);
    }
    throw std::runtime_error(message);
  }
  return response;
}

void ComputeShader::ReadMemory(void *context, std::uint64_t address,
                               std::uint32_t count, std::uint32_t *words) {
  auto &self = *static_cast<ComputeShader *>(context);
  const auto response = self.ExchangeMemory(address, count,
                                             ComputeMemoryOperation::kRead,
                                             nullptr);
  const auto &bytes = self.pool_.Read(response.payload);
  if (bytes.size() != count * sizeof(std::uint32_t)) {
    self.pool_.Release(response.payload);
    throw std::runtime_error("compute LD response has an invalid byte size");
  }
  std::memcpy(words, bytes.data(), bytes.size());
  self.pool_.Release(response.payload);
}

void ComputeShader::WriteMemory(void *context, std::uint64_t address,
                                std::uint32_t count,
                                const std::uint32_t *words) {
  auto &self = *static_cast<ComputeShader *>(context);
  const auto response = self.ExchangeMemory(address, count,
                                             ComputeMemoryOperation::kWrite,
                                             words);
  if (HasPoolHandle(response.payload)) {
    self.pool_.Release(response.payload);
    throw std::runtime_error("compute ST completion unexpectedly owns bytes");
  }
}

std::uint32_t ComputeShader::Atomic32Memory(void *context,
                                            ComputeMemoryOperation operation,
                                            std::uint64_t address,
                                            std::uint32_t operand) {
  if (!IsComputeAtomic32(operation))
    throw std::runtime_error("compute atomic callback received an invalid operation");
  auto &self = *static_cast<ComputeShader *>(context);
  const auto response = self.ExchangeMemory(address, 1, operation, &operand);
  const auto &bytes = self.pool_.Read(response.payload);
  if (bytes.size() != sizeof(std::uint32_t)) {
    self.pool_.Release(response.payload);
    throw std::runtime_error("compute atomic response is not one old DWORD");
  }
  std::uint32_t old = 0;
  std::memcpy(&old, bytes.data(), sizeof(old));
  self.pool_.Release(response.payload);
  return old;
}

bool ComputeShader::MutexMemory(void *context, std::uint32_t id,
                                std::uint32_t operation) {
  auto &self = *static_cast<ComputeShader *>(context);
  if (id >= 16 || operation > 3)
    throw std::runtime_error("compute issued an invalid native MUTEX");
  const auto response = self.ExchangeMemory(id, 0, operation == 3 ?
      ComputeMemoryOperation::kMutexLock : ComputeMemoryOperation::kMutexRelease, nullptr);
  if (HasPoolHandle(response.payload)) {
    self.pool_.Release(response.payload);
    throw std::runtime_error("compute MUTEX response unexpectedly owns bytes");
  }
  if (response.blocked) return false;
  const auto bit = UINT32_C(1) << id;
  if (operation == 3) self.current_mutex_mask_ |= bit;
  else self.current_mutex_mask_ &= ~bit;
  return true;
}

void ComputeShader::ReleaseTaskMutexes() {
  if (!current_mutex_mask_) return;
  const auto response = ExchangeMemory(0, 0, ComputeMemoryOperation::kMutexCleanup, nullptr);
  if (HasPoolHandle(response.payload)) {
    pool_.Release(response.payload);
    throw std::runtime_error("compute mutex cleanup unexpectedly owns bytes");
  }
  current_mutex_mask_ = 0;
}

void ComputeShader::Run() {
  for (;;) {
    ComputeWorkgroupTxn work;
    while (!input.nb_read(work))
      wait(input.data_written_event());
    ComputeWorkgroupResult result;
    std::vector<PoolHandle> resident;
    try {
      current_dispatch_ = work.state;
      const auto state = ReadPod<ComputeDispatchState>(pool_, work.state);
      if (!SameHandle(cached_dispatch_, work.state)) {
        if (HasPoolHandle(cached_instructions_))
          throw std::runtime_error("compute dispatch replaced an unfinished task");
        const auto program = DecodePcoProgram(
            ShaderStage::kCompute, LoadArray<std::uint8_t>(pool_, state.code));
        ValidateComputeProgram(program, state.abi);
        cached_summary_ = program.summary;
        cached_instructions_ = StoreNewArray(pool_, program.instructions);
        cached_dispatch_ = work.state;
      }
      const PcoDecodedProgram program{
          cached_summary_, LoadArray<PcoInstruction>(pool_, cached_instructions_)};
      auto shared =
          LoadArray<std::uint32_t>(pool_, state.shared_registers);
      if (state.abi.shared_memory_bytes) {
        const auto first = state.abi.shared_memory_descriptor_start;
        if (first > shared.size() || 4U > shared.size() - first)
          throw std::runtime_error("compute shared descriptor exceeds register payload");
        shared[first] = static_cast<std::uint32_t>(kComputeSharedAddress);
        shared[first + 1] = static_cast<std::uint32_t>(kComputeSharedAddress >> 32);
        shared[first + 2] = state.abi.shared_memory_bytes;
        shared[first + 3] = 0;
      }
      const std::uint32_t local_count = state.abi.local_size[0] *
          state.abi.local_size[1] * state.abi.local_size[2];
      const ComputeMemoryCallbacks memory{this, ReadMemory, WriteMemory,
                                            Atomic32Memory, nullptr, MutexMemory, SampleTexture};
      for (std::uint32_t first = 0; first < local_count;
           first += kComputeTaskWidth) {
        const std::uint32_t count = std::min(kComputeTaskWidth, local_count-first);
        auto task = MakeComputeTask(state.abi, shared, state.grid, work.group,
                                    first, count);
        const auto handle = pool_.Allocate(sizeof(task));
        resident.push_back(handle);
        WritePod(pool_, handle, task);
        result.stats.invocations += count;
      }
      // All tasks in a workgroup are resident. One native instruction group
      // per runnable task gives FIFO-completed memory a deterministic fair
      // ordering without running a task through another task's barrier.
      std::size_t remaining = resident.size();
      while (remaining) {
        bool progress = false;
        for (const auto handle : resident) {
          auto task = ReadPod<ComputeTaskState>(pool_, handle);
          if (task.ended || task.mutex_sleep_mask) continue;
          current_task_ = handle;
          current_mutex_mask_ = task.mutex_held_mask;
          task.mutex_wakeup_mask = 0;
          task.mutex_blocked = 0;
          StepComputeTask(program, state.abi, task, memory, result);
          WritePod(pool_, handle, task);
          progress |= !task.mutex_blocked;
          if (task.ended) --remaining;
          if (task.mutex_wakeup_mask) {
            for (const auto sleeper_handle : resident) {
              auto sleeper = ReadPod<ComputeTaskState>(pool_, sleeper_handle);
              sleeper.mutex_sleep_mask &= ~task.mutex_wakeup_mask;
              WritePod(pool_, sleeper_handle, sleeper);
            }
          }
        }
        if (!progress)
          throw std::runtime_error("compute resident tasks deadlocked on native MUTEX wait");
      }
      for (const auto handle : resident) pool_.Release(handle);
      resident.clear();
      current_task_ = {};
      current_mutex_mask_ = 0;
      result.stats.workgroups = 1;
      const std::uint64_t total_groups =
          static_cast<std::uint64_t>(state.grid[0]) * state.grid[1] * state.grid[2];
      if (work.ordinal + 1 == total_groups) {
        pool_.Release(cached_instructions_);
        cached_instructions_ = {};
        cached_dispatch_ = {};
      }
    } catch (const std::exception &error) {
      std::string message = error.what();
      for (const auto handle : resident) {
        current_task_ = handle;
        // The failed step may have acquired a lock before publishing its
        // task snapshot. Cleanup all task-owned locks through the CDM FIFO.
        current_mutex_mask_ = UINT32_MAX;
        try { ReleaseTaskMutexes(); }
        catch (const std::exception &cleanup) {
          message += "; MUTEX cleanup failed: "; message += cleanup.what();
        }
        pool_.Release(handle);
      }
      resident.clear();
      current_task_ = {};
      if (HasPoolHandle(cached_instructions_))
        pool_.Release(cached_instructions_);
      cached_instructions_ = {};
      cached_dispatch_ = {};
      result.failed = 1;
      const std::size_t length =
          std::min(message.size(), result.error.size() - 1);
      std::memcpy(result.error.data(), message.data(), length);
      result.error[length] = 0;
    }
    WritePod(pool_, work.result, result);
    while (!output.nb_write(work))
      wait(output.data_read_event());
  }
}

} // namespace pvrgpu::stub
