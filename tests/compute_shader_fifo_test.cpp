// Native compute task lockstep and ownership across depth-one SystemC FIFOs.
// The overlapping copy is intentionally not equivalent to lane-major shader
// execution: every LD in a task must precede that task's first ST.
#include "common/functional_types.h"
#include "data_master/compute_data_master.h"
#include "memory/gpu_memory_system.h"
#include "pco_compute_fixtures.h"
#include "compute_atomic_reference.h"
#include "shader/compute_shader.h"

#include <systemc>

#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace pvrgpu::stub;

void Check(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename T> void Store(MemoryPool &pool, PoolHandle handle, const T &value) {
  auto &bytes = pool.Write(handle);
  Check(bytes.size() == sizeof(value), "compute test POD size");
  std::memcpy(bytes.data(), &value, sizeof(value));
}

template <typename T> T Load(MemoryPool &pool, PoolHandle handle) {
  T value;
  const auto &bytes = pool.Read(handle);
  Check(bytes.size() == sizeof(value), "compute test POD size");
  std::memcpy(&value, bytes.data(), sizeof(value));
  return value;
}

ComputePcoAbi CopyAbi(unsigned local_count) {
  ComputePcoAbi abi;
  abi.local_size = local_count == 30 ? std::array<std::uint32_t,3>{3,2,5} :
                                      std::array<std::uint32_t,3>{local_count,1,1};
  abi.stage.temps = 6;
  abi.stage.vertex_inputs = abi.local_invocation_index_count = 1;
  abi.stage.shareds = abi.stage.push_constant_start = 8;
  abi.storage_buffer_descriptor_count = 2;
  abi.storage_buffer_used_mask = 3;
  abi.storage_buffer_read_mask = 1;
  abi.storage_buffer_write_mask = 2;
  return abi;
}

const std::array<ComputeMemoryOperation,10> kAtomicMemoryOps{
    ComputeMemoryOperation::kAtomicAdd32, ComputeMemoryOperation::kAtomicSub32,
    ComputeMemoryOperation::kAtomicExchange32, ComputeMemoryOperation::kAtomicUnsignedMin32,
    ComputeMemoryOperation::kAtomicSignedMin32, ComputeMemoryOperation::kAtomicUnsignedMax32,
    ComputeMemoryOperation::kAtomicSignedMax32, ComputeMemoryOperation::kAtomicAnd32,
    ComputeMemoryOperation::kAtomicOr32, ComputeMemoryOperation::kAtomicXor32};
const std::array<PcoOpcode,10> kAtomicPcoOps{
    PcoOpcode::kAtomicAdd32, PcoOpcode::kAtomicSub32, PcoOpcode::kAtomicExchange32,
    PcoOpcode::kAtomicUnsignedMin32, PcoOpcode::kAtomicSignedMin32,
    PcoOpcode::kAtomicUnsignedMax32, PcoOpcode::kAtomicSignedMax32,
    PcoOpcode::kAtomicAnd32, PcoOpcode::kAtomicOr32, PcoOpcode::kAtomicXor32};

void AtomicMaskAndFence() {
  // Decoded-POD invariant test: real per-lane dispatch/WDF with recording
  // callbacks. CDM arithmetic is independently exercised below through FIFO.
  for (unsigned op = 0; op < kAtomicPcoOps.size(); ++op) {
    for (unsigned condition = 0; condition < 4; ++condition) {
      ComputePcoAbi abi;
      abi.local_size = {32,1,1}; abi.stage.temps = 4;
      PcoInstruction atomic;
      atomic.opcode = kAtomicPcoOps[op]; atomic.exec_cnd = condition;
      atomic.source = {PcoRegisterBank::kTemporary,0};
      atomic.source1 = {PcoRegisterBank::kTemporary,1};
      atomic.source2 = {PcoRegisterBank::kTemporary,2};
      atomic.source_count = 3; atomic.target = PcoWriteTarget::kTemporary;
      atomic.output_index = 3;
      PcoInstruction fence;
      fence.opcode = PcoOpcode::kWaitDataFence; fence.source_count = 0;
      PcoInstruction end;
      end.opcode = PcoOpcode::kNop; end.source_count = 0; end.end_group = 1;
      PcoDecodedProgram program;
      program.summary.stage = ShaderStage::kCompute;
      program.summary.ends_task = 1;
      program.summary.instruction_count = program.summary.group_count = 3;
      program.instructions = {atomic,fence,end};
      ValidateComputeProgram(program, abi);
      auto task = MakeComputeTask(abi, {}, {1,1,1}, {0,0,0}, 0, 32);
      struct Record {
        ComputeMemoryOperation operation;
        unsigned count = 0;
      } record{kAtomicMemoryOps[op]};
      ComputeMemoryCallbacks callbacks;
      callbacks.user_data = &record;
      callbacks.atomic32 = [](void *opaque, ComputeMemoryOperation operation,
                               std::uint64_t address, std::uint32_t operand) {
        auto &r = *static_cast<Record *>(opaque);
        Check(operation == r.operation && address == UINT64_C(0x1000000000000),
              "native atomic callback lost its operation/address or executed a masked lane");
        ++r.count;
        return operand ^ UINT32_C(0xa5a5a5a5);
      };
      std::array<bool,32> selected{};
      unsigned count = 0;
      for (unsigned lane = 0; lane < 32; ++lane) {
        auto &state = task.lanes[lane];
        state.execution_predicate = lane % 3 != 0;
        state.predicate = lane % 2 != 0;
        selected[lane] = condition == 2 || (state.execution_predicate &&
            (condition == 0 || (condition == 1 ? state.predicate : !state.predicate)));
        count += selected[lane];
        // Inactive lanes deliberately have unwritten address/data sources.
        if (selected[lane]) {
          state.temporaries[0] = 0; state.temporaries[1] = 0x10000;
          state.temporaries[2] = lane;
          for (unsigned reg = 0; reg < 3; ++reg) state.temporary_written.set(reg);
        }
        state.temporaries[3] = 0xdeadbeef;
        state.temporary_written.set(3);
      }
      ComputeWorkgroupResult result;
      StepComputeTask(program, abi, task, callbacks, result);
      Check(record.count == count && result.stats.atomic_instructions == count,
            "atomic execution mask changed the number of RMWs");
      for (unsigned lane = 0; lane < 32; ++lane)
        Check(task.lanes[lane].temporaries[3] == 0xdeadbeef &&
                  task.lanes[lane].pending_operation == (selected[lane] ? 3U : 0U),
              "atomic old value became visible before WDF");
      StepComputeTask(program, abi, task, callbacks, result);
      for (unsigned lane = 0; lane < 32; ++lane)
        Check(task.lanes[lane].temporaries[3] == (selected[lane] ?
                  lane ^ UINT32_C(0xa5a5a5a5) : UINT32_C(0xdeadbeef)) &&
                  !task.lanes[lane].pending_operation,
              "WDF lost an atomic old value or wrote an inactive destination");
      StepComputeTask(program, abi, task, callbacks, result);
      Check(task.ended && record.count == count, "WDF duplicated an atomic RMW");
      for (unsigned malformed = 0; malformed < 4; ++malformed) {
        auto bad = program;
        if (malformed == 0) bad.instructions[0].component_count = 2; // no 64-bit AMO
        if (malformed == 1) bad.instructions[0].data_request = 1;
        if (malformed == 2) bad.instructions[0].output_index = 4;
        if (malformed == 3) bad.instructions[1].opcode = PcoOpcode::kNop;
        bool rejected = false;
        try { ValidateComputeProgram(bad, abi); }
        catch (const std::exception &) { rejected = true; }
        Check(rejected, "malformed atomic metadata or missing WDF was accepted");
      }
    }
  }
}

void AtomicServiceAllOps(MemoryPool &pool, GpuMemorySystem &memory,
                         sc_core::sc_fifo<ComputeMemoryTxn> &requests,
                         sc_core::sc_fifo<ComputeMemoryTxn> &responses) {
  constexpr std::uint64_t address = UINT64_C(0x1000000000000);
  ComputeDispatchState state;
  state.buffer_ranges = StoreNewArray(pool, std::vector<ComputeBufferRange>{
      {address,4,kComputeAccessRead | kComputeAccessWrite,0,1}});
  const auto handle = pool.Allocate(sizeof(state)); Store(pool, handle, state);
  std::uint64_t id = 1000;
  for (unsigned op = 0; op < kAtomicMemoryOps.size(); ++op) {
    for (const auto old : kComputeAtomicWords) {
      for (const auto operand : kComputeAtomicWords) {
        memory.HostWrite(address, &old, sizeof(old));
        ComputeMemoryTxn request;
        request.state = handle; request.request_id = ++id; request.address = address;
        request.operation = kAtomicMemoryOps[op]; request.bytes = 4;
        request.payload = StoreNewArray(pool, std::vector<std::uint32_t>{operand});
        Check(requests.nb_write(request), "atomic arithmetic FIFO was not empty");
        sc_core::sc_start(sc_core::sc_time(1,sc_core::SC_US));
        ComputeMemoryTxn response;
        Check(responses.nb_read(response) && !response.failed &&
                  response.request_id == id && response.operation == request.operation,
              "atomic arithmetic response failed or lost operation identity");
        const auto &bytes = pool.Read(response.payload);
        std::uint32_t actual_old = 0;
        Check(bytes.size() == sizeof(actual_old), "atomic old value is not exactly one DWORD");
        std::memcpy(&actual_old, bytes.data(), sizeof(actual_old));
        pool.Release(response.payload);
        const auto stored = memory.Readback(address,4,MemoryClient::kComputeReadback);
        std::uint32_t actual = 0; std::memcpy(&actual, stored.data.data(), sizeof(actual));
        Check(actual_old == old && actual == ComputeAtomicReference(kComputeAtomicNibbles[op],old,operand),
              "atomic op lost old bits, signed extrema, logical semantics or modulo wrapping");
      }
    }
    // Every op shares exactly the same extent/operand validation and error
    // ownership, not just the pre-existing ADD service-error tests.
    for (unsigned fault = 0; fault < 6; ++fault) {
      const std::uint32_t initial = 0x12345678;
      memory.HostWrite(address, &initial, sizeof(initial));
      ComputeMemoryTxn request;
      request.state = handle; request.request_id = ++id;
      request.address = fault == 0 ? address + 1 : fault == 1 ? address + 4 : address;
      request.operation = fault == 5 ? static_cast<ComputeMemoryOperation>(0xffffffffU) : kAtomicMemoryOps[op];
      request.bytes = fault == 2 ? 8 : 4;
      const std::vector<std::uint32_t> words = fault == 3 ? std::vector<std::uint32_t>{} :
          fault == 4 ? std::vector<std::uint32_t>{1,2} : std::vector<std::uint32_t>{1};
      request.payload = StoreNewArray(pool, words);
      Check(requests.nb_write(request), "invalid atomic FIFO was not empty");
      sc_core::sc_start(sc_core::sc_time(1,sc_core::SC_US));
      ComputeMemoryTxn response;
      Check(responses.nb_read(response) && response.failed && response.request_id == id,
            "invalid atomic request did not fail closed");
      Check(!pool.Read(response.payload).empty() && !pool.Read(response.payload).back(),
            "atomic error did not own a terminated message");
      pool.Release(response.payload);
      const auto stored = memory.Readback(address,4,MemoryClient::kComputeReadback);
      std::uint32_t actual = 0; std::memcpy(&actual,stored.data.data(),sizeof(actual));
      Check(actual == initial, "invalid atomic request performed a partial RMW");
    }
  }
  pool.Release(state.buffer_ranges); pool.Release(handle);
  Check(pool.allocations() == pool.releases() && pool.bytes_in_flight() == 0,
        "atomic operations or exceptions leaked FIFO payloads");
}

void MutexServiceOwnership(MemoryPool &pool,
                            sc_core::sc_fifo<ComputeMemoryTxn> &requests,
                            sc_core::sc_fifo<ComputeMemoryTxn> &responses) {
  ComputeDispatchState state;
  const auto handle = pool.Allocate(sizeof(state)); Store(pool,handle,state);
  const auto first = pool.Allocate(sizeof(ComputeTaskState));
  const auto second = pool.Allocate(sizeof(ComputeTaskState));
  std::uint64_t id = 9000;
  const auto exchange = [&](ComputeMemoryOperation operation, PoolHandle task,
                            unsigned mutex, bool fail, unsigned bad = 0,
                            bool blocked = false) {
    ComputeMemoryTxn request;
    request.state = handle; request.task = task;
    request.request_id = ++id; request.address = mutex; request.operation = operation;
    if (bad == 1) request.bytes = 4;
    if (bad == 2) request.payload = StoreNewArray(pool,std::vector<std::uint32_t>{0});
    Check(requests.nb_write(request), "MUTEX FIFO was not empty");
    sc_core::sc_start(sc_core::sc_time(1,sc_core::SC_US));
    ComputeMemoryTxn response;
    const bool received = responses.nb_read(response);
    if (!received || (response.failed != 0) != fail) {
      std::cerr << "mutex test id=" << id << " expected_failed=" << fail
                << " received=" << received << " failed=" << response.failed;
      if (received && response.failed && HasPoolHandle(response.payload))
        std::cerr << " error=" << reinterpret_cast<const char *>(pool.Read(response.payload).data());
      std::cerr << '\n';
    }
    Check(received && response.request_id == id &&
              (response.failed != 0) == fail && response.task.slot == task.slot &&
              response.task.generation == task.generation &&
              (response.blocked != 0) == blocked,
          "MUTEX ownership reply was incorrect");
    if (fail) {
      const auto &bytes = pool.Read(response.payload);
      Check(!bytes.empty() && !bytes.back(), "MUTEX error has no owned terminated message");
      pool.Release(response.payload);
    } else Check(!HasPoolHandle(response.payload), "MUTEX success invented result bytes");
  };
  for (unsigned mutex = 0; mutex < 16; ++mutex) {
    exchange(ComputeMemoryOperation::kMutexLock,first,mutex,false);
    exchange(ComputeMemoryOperation::kMutexLock,first,mutex,true); // no reentry
    exchange(ComputeMemoryOperation::kMutexLock,second,mutex,false,0,true); // busy owner
    exchange(ComputeMemoryOperation::kMutexRelease,second,mutex,true);
    exchange(ComputeMemoryOperation::kMutexCleanup,second,0,false);
    exchange(ComputeMemoryOperation::kMutexLock,second,mutex,false,0,true); // cleanup cannot steal
    exchange(ComputeMemoryOperation::kMutexRelease,first,mutex,false);
    exchange(ComputeMemoryOperation::kMutexLock,second,mutex,false);
    exchange(ComputeMemoryOperation::kMutexCleanup,second,0,false);
    exchange(ComputeMemoryOperation::kMutexLock,first,mutex,false);
    exchange(ComputeMemoryOperation::kMutexCleanup,first,0,false);
  }
  exchange(ComputeMemoryOperation::kMutexLock,first,16,true);
  exchange(ComputeMemoryOperation::kMutexLock,first,0,true,1);
  exchange(ComputeMemoryOperation::kMutexLock,first,0,true,2);
  exchange(ComputeMemoryOperation::kMutexLock,{},0,true);
  pool.Release(first);
  exchange(ComputeMemoryOperation::kMutexLock,first,0,true); // stale generation
  pool.Release(second); pool.Release(handle);
  Check(pool.allocations() == pool.releases() && !pool.bytes_in_flight(),
        "MUTEX validation/cleanup leaked a payload or owner");
}

void RunNativeCas(MemoryPool &pool, GpuMemorySystem &memory,
                   sc_core::sc_fifo<ComputeDispatchTxn> &input,
                   sc_core::sc_fifo<ComputeDispatchTxn> &completion,
                   unsigned fault, std::uint64_t sequence) {
  constexpr std::uint64_t address = UINT64_C(0x1000000000000);
  constexpr auto output = address + 512;
  constexpr std::uint32_t initial = UINT32_MAX - 17;
  std::array<std::uint32_t,30> operands{};
  for (unsigned lane = 0; lane < operands.size(); ++lane)
    operands[lane] = initial + 13U * (lane / 2);
  memory.HostWrite(address,&initial,sizeof(initial));
  memory.HostWrite(output,operands.data(),sizeof(operands));
  ComputeDispatchState state;
  state.abi = CopyAbi(30); state.abi.stage.temps = 10;
  state.abi.storage_buffer_read_mask = state.abi.storage_buffer_write_mask = 3;
  state.grid = {1,1,1}; state.sequence = sequence;
  auto binary = ComputePcoFixture(21);
  if (fault == 1) {
    const auto decoded = DecodeComputePcoProgram(binary);
    for (const auto &instruction : decoded.instructions) {
      if (instruction.opcode == PcoOpcode::kMutex && instruction.control_operation == 0) {
        // Public 4-byte NOP replaces RELEASE in this malformed encoding test.
        // The real END must fail and driver-owned task cleanup release its lock.
        binary[instruction.binary_offset + 2] = 0x6e;
        binary[instruction.binary_offset + 3] = 0;
      }
    }
  }
  state.code = StoreNewArray(pool,binary);
  state.shared_registers = StoreNewArray(pool,std::vector<std::uint32_t>{
      static_cast<std::uint32_t>(address),static_cast<std::uint32_t>(address >> 32),4,0,
      static_cast<std::uint32_t>(output),static_cast<std::uint32_t>(output >> 32),sizeof(operands),0});
  state.buffer_ranges = StoreNewArray(pool,std::vector<ComputeBufferRange>{
      {address,4,fault == 2 ? kComputeAccessRead : kComputeAccessRead | kComputeAccessWrite,0,1},
      {output,sizeof(operands),kComputeAccessRead | kComputeAccessWrite,1,1}});
  const auto handle = pool.Allocate(sizeof(state)); Store(pool,handle,state);
  Check(input.nb_write({handle,sequence}), "CAS dispatch FIFO was not empty");
  sc_core::sc_start(sc_core::sc_time(1,sc_core::SC_MS));
  ComputeDispatchTxn done;
  Check(completion.nb_read(done) && done.sequence == sequence,"CAS dispatch did not complete");
  state = Load<ComputeDispatchState>(pool,handle);
  if (fault) {
    Check(state.failed && std::string(state.error.data()).find(fault == 1 ?
              "END leaves a MUTEX" : "outside permitted views") != std::string::npos,
          "malformed CAS failed for an unrelated reason or silently succeeded");
  } else {
    Check(!state.failed,state.error.data());
    const auto values = memory.Readback(output,sizeof(operands),MemoryClient::kComputeReadback);
    std::array<std::uint32_t,30> old{};
    std::memcpy(old.data(),values.data.data(),sizeof(old));
    const auto counter_bytes = memory.Readback(address,4,MemoryClient::kComputeReadback);
    std::uint32_t counter = 0; std::memcpy(&counter,counter_bytes.data.data(),4);
    // The actual usclib loops over INST_NUM 0..31 under a task mutex; these
    // pairs deliberately alternate comparison success/mismatch and wrap.
    std::uint32_t expected = initial;
    unsigned success = 0, mismatch = 0;
    for (unsigned lane = 0; lane < old.size(); ++lane) {
      Check(old[lane] == expected,"native CAS did not return the old value for its real instance");
      if (operands[lane] == expected) { expected += 13; ++success; }
      else ++mismatch;
    }
    Check(counter == expected && success == 15 && mismatch == 15 &&
              state.stats.atomic_instructions == 0 && state.stats.load_instructions == 60 &&
              state.stats.store_instructions == 60 && state.stats.invocations == 30,
          "CAS mutex/LD/ST path lost updates, counted tail lanes, or invented DMA AMO statistics");
  }
  pool.Release(state.code); pool.Release(state.shared_registers);
  pool.Release(state.buffer_ranges); pool.Release(handle);
  Check(pool.allocations() == pool.releases() && !pool.bytes_in_flight(),
        "CAS completion/fault leaked a task or memory payload");
}

void RejectMetadata() {
  const auto program = DecodePcoProgram(ShaderStage::kCompute, ComputePcoFixture(1));
  const auto reject = [&](ComputePcoAbi abi) {
    bool rejected = false;
    try { ValidateComputeProgram(program, abi); }
    catch (const std::exception &) { rejected = true; }
    Check(rejected, "invalid compute metadata was accepted");
  };
  auto abi = CopyAbi(30);
  ValidateComputeProgram(program, abi);
  abi.stage.temps = 5;
  reject(abi);
  abi = CopyAbi(30);
  abi.stage.entry_offset = 8;
  reject(abi);
  abi = CopyAbi(30);
  abi.stage.coefficients = 260;
  abi.workgroup_id_start = 258;
  abi.workgroup_id_count = 3;
  reject(abi);
  abi = CopyAbi(30);
  abi.shared_memory_bytes = 4;
  reject(abi);
  auto graphics = program;
  graphics.summary.stage = ShaderStage::kFragment;
  bool rejected = false;
  try { ValidateComputeProgram(graphics, CopyAbi(30)); }
  catch (const std::exception &) { rejected = true; }
  Check(rejected, "compute accepted graphics-stage metadata");
}

void RepeatedAluCounters() {
  // A decoded POD invariant test, not a synthetic binary or shader-result
  // shortcut: the real interpreter executes the two register transfers.
  ComputePcoAbi abi;
  abi.local_size = {32,1,1};
  abi.stage.temps = abi.stage.shareds = abi.stage.push_constant_count = 2;
  PcoInstruction move;
  move.opcode = PcoOpcode::kMoveBypass;
  move.source = {PcoRegisterBank::kShared, 0};
  move.target = PcoWriteTarget::kTemporary;
  move.repeat_count = 2;
  move.end_group = 1;
  PcoDecodedProgram program;
  program.summary.stage = ShaderStage::kCompute;
  program.summary.instruction_count = program.summary.group_count = 1;
  program.summary.ends_task = 1;
  program.instructions = {move};
  ValidateComputeProgram(program, abi);
  auto task = MakeComputeTask(abi, {0xdeadbeefU,0x01234567U}, {1,1,1}, {0,0,0}, 0, 32);
  ComputeWorkgroupResult result;
  StepComputeTask(program, abi, task, {}, result);
  Check(task.ended && result.instructions_executed == 32 &&
            result.stats.alu_instructions == 64 && result.stats.memory_instructions == 0,
        "native group repeats were not expanded in compute ALU counters");
  for (const auto &lane : task.lanes)
    Check(lane.temporaries[0] == 0xdeadbeefU && lane.temporaries[1] == 0x01234567U &&
              lane.temporary_written.contains_range(0,2),
          "repeated native ALU did not transfer both registers");
}

void TaskLoopReconvergence(bool input_counter) {
  // The decoded CNDLT invariant includes running, continuing, broken and
  // outer-if-masked lanes in one task. A break must not reactivate merely
  // because another lane reaches the loop epilogue earlier.
  ComputePcoAbi abi;
  abi.local_size = {32,1,1};
  abi.stage.temps = 1;
  abi.stage.vertex_inputs = input_counter ? 1 : 0;
  PcoInstruction loop;
  loop.opcode = PcoOpcode::kConditionalMask;
  loop.control_operation = 3;
  loop.exec_cnd = 2;
  loop.writes_predicate = 1;
  loop.source = {input_counter ? PcoRegisterBank::kVertexInput : PcoRegisterBank::kTemporary,0};
  loop.target = input_counter ? PcoWriteTarget::kVertexInput : PcoWriteTarget::kTemporary;
  loop.immediate = 2;
  PcoInstruction branch;
  branch.opcode = PcoOpcode::kBranch;
  branch.source_count = 0;
  branch.exec_cnd = 1;
  branch.branch_target_index = 0;
  PcoInstruction end;
  end.opcode = PcoOpcode::kNop;
  end.source_count = 0;
  end.end_group = 1;
  PcoDecodedProgram program;
  program.summary.stage = ShaderStage::kCompute;
  program.summary.instruction_count = program.summary.group_count = 3;
  program.summary.ends_task = 1;
  program.instructions = {loop,branch,end};
  ValidateComputeProgram(program, abi);
  auto task = MakeComputeTask(abi, {}, {1,1,1}, {0,0,0}, 0, 32);
  const auto counter = [&](ComputeLaneState &lane) -> std::uint32_t & {
    return input_counter ? lane.inputs[0] : lane.temporaries[0];
  };
  for (unsigned i = 0; i < task.lane_count; ++i) {
    counter(task.lanes[i]) = i / 8;
    task.lanes[i].temporary_written.set(0);
    if (input_counter) task.lanes[i].inputs_written = 1;
    task.lanes[i].execution_predicate = i < 8;
  }
  ComputeWorkgroupResult result;
  StepComputeTask(program, abi, task, {}, result);
  for (unsigned i = 0; i < task.lane_count; ++i)
    Check(counter(task.lanes[i]) == (i < 16 ? 0U : i / 8) &&
              task.lanes[i].execution_predicate == (i < 16) &&
              task.lanes[i].predicate == (i < 16),
          "CNDLT released a broken/outer-masked lane before task reconvergence");
  StepComputeTask(program, abi, task, {}, result);
  Check(task.instruction_index == 0, "partial-task native loop branch was not taken");
  for (unsigned i = 0; i < 16; ++i) {
    counter(task.lanes[i]) = 2;
    task.lanes[i].execution_predicate = 0;
  }
  StepComputeTask(program, abi, task, {}, result);
  for (unsigned i = 0; i < task.lane_count; ++i)
    Check(counter(task.lanes[i]) == (i < 24 ? 0U : 1U) &&
              task.lanes[i].execution_predicate == (i < 24) && !task.lanes[i].predicate,
          "finished CNDLT failed to restore the enclosing execution mask");
  StepComputeTask(program, abi, task, {}, result);
  Check(task.instruction_index == 2, "finished native loop incorrectly branched back");
  StepComputeTask(program, abi, task, {}, result);
  Check(task.ended, "reconverged native loop did not reach END");
}

void UnpredicatedFence() {
  // Mid-task POD invariant: pco_cf.c explicitly permits predicated LD/ST
  // followed by an unpredicated WDF. No-request lanes must simply NOP.
  const auto abi = CopyAbi(30);
  const auto program = DecodePcoProgram(ShaderStage::kCompute, ComputePcoFixture(1));
  ValidateComputeProgram(program, abi);
  auto task = MakeComputeTask(abi, std::vector<std::uint32_t>(8), {1,1,1}, {0,0,0}, 0, 30);
  const auto fence = std::find_if(program.instructions.begin(), program.instructions.end(),
      [](const PcoInstruction &instruction) { return instruction.opcode == PcoOpcode::kWaitDataFence; });
  Check(fence != program.instructions.end(), "native copy has no WDF");
  task.instruction_index = static_cast<std::uint32_t>(fence - program.instructions.begin());
  auto &pending = task.lanes[0];
  pending.pending_operation = pending.pending_count = 1;
  pending.pending_output = 4;
  pending.pending_words[0] = 0x89abcdefU;
  pending.execution_predicate = 0;
  ComputeWorkgroupResult result;
  StepComputeTask(program, abi, task, {}, result);
  Check(pending.temporary_written.test(4) && pending.temporaries[4] == 0x89abcdefU,
        "unpredicated WDF dropped an issued native LD response");
  for (const auto &lane : task.lanes)
    Check(!lane.pending_operation && !lane.pending_count,
          "unpredicated WDF left an outstanding native request");
  Check(!task.lanes[1].temporary_written.test(4),
        "WDF invented a response for a lane without a request");
}

void RunCopy(MemoryPool &pool, GpuMemorySystem &memory,
              sc_core::sc_fifo<ComputeDispatchTxn> &input,
              sc_core::sc_fifo<ComputeDispatchTxn> &completion,
              unsigned local_count, unsigned fixture, bool fault,
              std::uint64_t sequence) {
  constexpr std::uint64_t address = UINT64_C(0x1000000000000);
  std::array<std::uint32_t, 40> original{};
  for (unsigned i = 0; i < original.size(); ++i)
    original[i] = 0xfeed1234U ^ (i * 0x01020305U);
  memory.HostWrite(address, original.data(), sizeof(original));
  ComputeDispatchState state;
  state.abi = CopyAbi(local_count);
  state.grid = {1,1,1};
  state.sequence = sequence;
  state.code = StoreNewArray(pool, ComputePcoFixture(fixture));
  const std::uint32_t bytes = local_count * 4;
  const std::uint64_t destination = address + 4;
  const std::vector<std::uint32_t> shared{
      static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32), bytes, 0,
      static_cast<std::uint32_t>(destination), static_cast<std::uint32_t>(destination >> 32), bytes, 0};
  state.shared_registers = StoreNewArray(pool, shared);
  state.buffer_ranges = StoreNewArray(pool, std::vector<ComputeBufferRange>{
      {address, bytes, kComputeAccessRead, 0, 1},
      {destination, fault ? bytes - 4U : bytes, kComputeAccessWrite, 1, 1}});
  const auto handle = pool.Allocate(sizeof(state));
  Store(pool, handle, state);
  const ComputeDispatchTxn submitted{handle, sequence};
  Check(input.nb_write(submitted), "depth-one compute input rejected idle dispatch");
  sc_core::sc_start(sc_core::sc_time(1, sc_core::SC_MS));
  ComputeDispatchTxn done;
  Check(completion.nb_read(done), "compute FIFO pipeline failed to complete before timeout");
  Check(done.sequence == sequence && done.state.slot == handle.slot &&
            done.state.generation == handle.generation,
        "compute completion did not preserve dispatch ownership");
  state = Load<ComputeDispatchState>(pool, handle);
  Check((state.failed != 0) == fault,
        state.failed ? state.error.data() : "out-of-range ST unexpectedly succeeded");
  if (fault) {
    Check(std::string(state.error.data()).find("outside permitted views") != std::string::npos,
          "fault did not originate at the modeled memory permission boundary");
  } else {
    const auto readback = memory.Readback(address, sizeof(original), MemoryClient::kComputeReadback);
    std::array<std::uint32_t,40> actual{};
    std::memcpy(actual.data(), readback.data.data(), sizeof(actual));
    for (unsigned i = 0; i < actual.size(); ++i)
      Check(actual[i] == (i >= 1 && i <= local_count ? original[i-1] : original[i]),
            "overlapping copy violated task lockstep or native tail-lane masking");
    Check(state.stats.workgroups == 1 && state.stats.invocations == local_count &&
              state.stats.load_instructions == local_count &&
              state.stats.store_instructions == local_count && state.instructions_executed != 0,
          "native compute invocation or LD/ST counters are incorrect");
    Check(state.counters.vs_invocations == 0 && state.counters.ps_invocations == 0,
          "compute dispatch was incorrectly counted as graphics execution");
    if (memory.mode() == MemoryMode::kDirect)
      Check(state.stats.direct_read_bytes == bytes && state.stats.direct_write_bytes == bytes &&
                state.stats.dram_read_bytes == 0 && state.stats.dram_write_bytes == 0,
            "direct compute memory counters are not actual LD/ST bytes");
    else
      Check(state.stats.direct_read_bytes == 0 && state.stats.direct_write_bytes == 0 &&
                state.stats.dram_read_bytes != 0,
            "compute bypass/cache dispatch did not use modeled DRAM");
  }
  pool.Release(state.code);
  pool.Release(state.shared_registers);
  pool.Release(state.buffer_ranges);
  pool.Release(handle);
  Check(pool.allocations() == pool.releases() && pool.bytes_in_flight() == 0,
        "compute request, response, lane or decode payload leaked");
  Check(input.num_available() == 0 && completion.num_available() == 0,
        "compute completion left an extra FIFO token");
}

void RunNativeLoop(MemoryPool &pool, GpuMemorySystem &memory,
                    sc_core::sc_fifo<ComputeDispatchTxn> &input,
                    sc_core::sc_fifo<ComputeDispatchTxn> &completion,
                    bool divergent, std::uint64_t sequence) {
  constexpr std::uint64_t source = UINT64_C(0x1000000000000);
  constexpr std::uint64_t destination = source + 512;
  constexpr std::uint32_t sentinel = 0x9badfeedU;
  std::array<std::uint32_t,90> original{}, initial{};
  initial.fill(sentinel);
  for (unsigned i = 0; i < original.size(); ++i) original[i] = 0x01234567U ^ (i * 71U);
  memory.HostWrite(source, original.data(), sizeof(original));
  memory.HostWrite(destination, initial.data(), sizeof(initial));
  ComputeDispatchState state;
  state.abi = CopyAbi(30);
  state.abi.stage.temps = divergent ? 9 : 8;
  if (!divergent) {
    state.abi.stage.coefficients = 3;
    state.abi.num_workgroups_count = 3;
  }
  state.grid = {divergent ? 1U : 3U,1,1};
  state.sequence = sequence;
  state.code = StoreNewArray(pool, ComputePcoFixture(divergent ? 9 : 8));
  state.shared_registers = StoreNewArray(pool, std::vector<std::uint32_t>{
      static_cast<std::uint32_t>(source), static_cast<std::uint32_t>(source >> 32), sizeof(original), 0,
      static_cast<std::uint32_t>(destination), static_cast<std::uint32_t>(destination >> 32), sizeof(initial), 0});
  state.buffer_ranges = StoreNewArray(pool, std::vector<ComputeBufferRange>{
      {source,sizeof(original),kComputeAccessRead,0,1},
      {destination,sizeof(initial),kComputeAccessWrite,1,1}});
  const auto handle = pool.Allocate(sizeof(state));
  Store(pool, handle, state);
  Check(input.nb_write({handle,sequence}), "native loop input FIFO was not empty");
  sc_core::sc_start(sc_core::sc_time(1, sc_core::SC_MS));
  ComputeDispatchTxn done;
  Check(completion.nb_read(done) && done.sequence == sequence,
        "native loop did not complete across depth-one FIFOs");
  state = Load<ComputeDispatchState>(pool, handle);
  Check(!state.failed, state.error.data());
  const auto readback = memory.Readback(destination, sizeof(initial), MemoryClient::kComputeReadback);
  std::array<std::uint32_t,90> actual{};
  std::memcpy(actual.data(), readback.data.data(), sizeof(actual));
  for (unsigned i = 0; i < actual.size(); ++i) {
    const bool written = !divergent || i / 30U < (i % 30U) % 3U + 1U;
    Check(actual[i] == (written ? original[i] : sentinel),
          "native loop lost a lane or resumed an invocation after its break");
  }
  const std::uint64_t expected_accesses = divergent ? 60 : 270;
  Check(state.stats.workgroups == (divergent ? 1U : 3U) &&
            state.stats.invocations == (divergent ? 30U : 90U) &&
            state.stats.load_instructions == expected_accesses &&
            state.stats.store_instructions == expected_accesses,
        "native loop dynamic LD/ST counters do not match actual iterations");
  pool.Release(state.code);
  pool.Release(state.shared_registers);
  pool.Release(state.buffer_ranges);
  pool.Release(handle);
  Check(pool.allocations() == pool.releases() && pool.bytes_in_flight() == 0,
        "native loop left a task or memory payload alive");
}

void RunNativeAtomic(MemoryPool &pool, GpuMemorySystem &memory,
                      sc_core::sc_fifo<ComputeDispatchTxn> &input,
                      sc_core::sc_fifo<ComputeDispatchTxn> &completion,
                      bool variable_operand, std::uint64_t sequence) {
  constexpr std::uint64_t address = UINT64_C(0x1000000000000);
  constexpr std::uint64_t output_address = address + 512;
  constexpr std::uint32_t initial = UINT32_MAX - 5U;
  std::array<std::uint32_t,30> empty{};
  memory.HostWrite(address, &initial, sizeof(initial));
  memory.HostWrite(output_address, empty.data(), sizeof(empty));
  ComputeDispatchState state;
  state.abi = CopyAbi(30);
  state.abi.storage_buffer_write_mask = 3;
  state.grid = {2,1,1};
  state.sequence = sequence;
  std::vector<std::uint32_t> shared{
      static_cast<std::uint32_t>(address), static_cast<std::uint32_t>(address >> 32), 4, 0,
      static_cast<std::uint32_t>(output_address), static_cast<std::uint32_t>(output_address >> 32), sizeof(empty), 0};
  if (variable_operand) {
    state.abi.stage.shareds = 16;
    state.abi.stage.push_constant_count = 8;
    shared.resize(16);
    shared[12] = 1; // CB0 DWORD4; native ALU adds the local index.
  }
  state.code = StoreNewArray(pool, ComputePcoFixture(variable_operand ? 11 : 10));
  state.shared_registers = StoreNewArray(pool, shared);
  state.buffer_ranges = StoreNewArray(pool, std::vector<ComputeBufferRange>{
      {address,4,kComputeAccessRead | kComputeAccessWrite,0,1},
      {output_address,sizeof(empty),kComputeAccessWrite,1,1}});
  const auto handle = pool.Allocate(sizeof(state));
  Store(pool, handle, state);
  Check(input.nb_write({handle,sequence}), "atomic input FIFO was not empty");
  sc_core::sc_start(sc_core::sc_time(1, sc_core::SC_MS));
  ComputeDispatchTxn done;
  Check(completion.nb_read(done) && done.sequence == sequence,
        "native atomic dispatch did not complete");
  state = Load<ComputeDispatchState>(pool, handle);
  Check(!state.failed, state.error.data());
  const auto counter_bytes = memory.Readback(address, 4, MemoryClient::kComputeReadback);
  std::uint32_t counter = 0;
  std::memcpy(&counter, counter_bytes.data.data(), sizeof(counter));
  const std::uint32_t group_sum = variable_operand ? 465 : 30;
  Check(counter == initial + group_sum * 2U,
        "native atomic additions were lost, split or failed modulo-32 wrap");
  const auto old_bytes = memory.Readback(output_address, sizeof(empty), MemoryClient::kComputeReadback);
  std::array<std::uint32_t,30> old{};
  std::memcpy(old.data(), old_bytes.data.data(), sizeof(old));
  // Invocation ordering is not a GLSL guarantee. Validate a complete legal
  // linearization using each native returned-old value and its real addend.
  std::array<std::pair<std::uint32_t,std::uint32_t>,30> transitions{};
  for (unsigned lane = 0; lane < old.size(); ++lane)
    transitions[lane] = {old[lane] - (initial + group_sum), variable_operand ? lane + 1U : 1U};
  std::sort(transitions.begin(), transitions.end());
  std::uint32_t total = 0;
  for (const auto &transition : transitions) {
    Check(transition.first == total,
          "native atomic old values do not form one serialized RMW history");
    total += transition.second;
  }
  Check(total == group_sum && state.stats.atomic_instructions == 60 &&
            state.stats.load_instructions == 0 && state.stats.store_instructions == 60 &&
            state.stats.invocations == 60 && state.stats.workgroups == 2,
        "native atomic counters confuse RMWs with LD/ST or physical tail lanes");
  if (memory.mode() == MemoryMode::kDirect)
    Check(state.stats.direct_read_bytes == 240 && state.stats.direct_write_bytes == 480,
          "atomic traffic omits either the modeled read or write");
  pool.Release(state.code); pool.Release(state.shared_registers);
  pool.Release(state.buffer_ranges); pool.Release(handle);
  Check(pool.allocations() == pool.releases() && pool.bytes_in_flight() == 0,
        "native atomic dispatch leaked request/response or task ownership");
}

void AtomicServiceErrors(MemoryPool &pool, GpuMemorySystem &memory,
                          sc_core::sc_fifo<ComputeMemoryTxn> &requests,
                          sc_core::sc_fifo<ComputeMemoryTxn> &responses) {
  constexpr std::uint64_t address = UINT64_C(0x1000000000000);
  std::uint32_t initial = UINT32_MAX - 1U;
  memory.HostWrite(address, &initial, sizeof(initial));
  ComputeDispatchState state;
  std::vector<ComputeBufferRange> ranges{
      {address,4,kComputeAccessRead,0,1}, {address,4,kComputeAccessWrite,1,1}};
  state.buffer_ranges = StoreNewArray(pool, ranges);
  const auto handle = pool.Allocate(sizeof(state));
  Store(pool, handle, state);
  const auto make = [&](std::uint64_t id, std::uint64_t target, unsigned bytes,
                         std::uint32_t operand) {
    ComputeMemoryTxn txn;
    txn.state = handle; txn.request_id = id; txn.address = target; txn.bytes = bytes;
    txn.operation = ComputeMemoryOperation::kAtomicAdd32;
    txn.payload = StoreNewArray(pool, std::vector<std::uint32_t>{operand});
    return txn;
  };
  const auto receive = [&](std::uint64_t id, bool failed, std::uint32_t expected) {
    ComputeMemoryTxn response;
    Check(responses.nb_read(response) && response.request_id == id &&
              (response.failed != 0) == failed,
          "atomic service response identity/status is incorrect");
    const auto &bytes = pool.Read(response.payload);
    if (!failed) {
      std::uint32_t value = 0;
      Check(bytes.size() == sizeof(value), "atomic service response is not one DWORD");
      std::memcpy(&value, bytes.data(), sizeof(value));
      Check(value == expected, "atomic FIFO response does not contain the original old DWORD");
    } else {
      Check(!bytes.empty() && !bytes.back(), "atomic failure did not own a terminated error");
    }
    pool.Release(response.payload);
  };
  Check(requests.nb_write(make(100,address,4,1)), "atomic request input was not empty");
  sc_core::sc_start(sc_core::sc_time(1,sc_core::SC_US));
  receive(100,true,0); // Read and write permissions cannot be combined across views.
  ranges[0].access = kComputeAccessRead | kComputeAccessWrite;
  StoreArray(pool, state.buffer_ranges, ranges);
  for (unsigned invalid = 0; invalid < 3; ++invalid) {
    const auto target = invalid == 0 ? address + 1 : invalid == 1 ? UINT64_MAX - 3U : address;
    Check(requests.nb_write(make(101+invalid,target,invalid == 2 ? 8 : 4,1)),
          "atomic invalid-request FIFO was not empty");
    sc_core::sc_start(sc_core::sc_time(1,sc_core::SC_US));
    receive(101+invalid,true,0);
  }
  // Keep response 1 in its depth-one FIFO while request 2 completes its RMW
  // and blocks on response backpressure. Neither response may be duplicated.
  Check(requests.nb_write(make(104,address,4,1)), "atomic recovery enqueue failed");
  sc_core::sc_start(sc_core::sc_time(1,sc_core::SC_US));
  Check(requests.nb_write(make(105,address,4,2)), "atomic second enqueue failed");
  sc_core::sc_start(sc_core::sc_time(1,sc_core::SC_US));
  receive(104,false,UINT32_MAX-1U);
  sc_core::sc_start(sc_core::sc_time(1,sc_core::SC_US));
  receive(105,false,UINT32_MAX);
  const auto final_bytes = memory.Readback(address,4,MemoryClient::kComputeReadback);
  std::uint32_t final = 0;
  std::memcpy(&final, final_bytes.data.data(), sizeof(final));
  Check(final == 1, "atomic service did not recover with exact modulo-32 RMW");
  pool.Release(state.buffer_ranges); pool.Release(handle);
  Check(pool.allocations() == pool.releases() && pool.bytes_in_flight() == 0 &&
            requests.num_available() == 0 && responses.num_available() == 0,
        "atomic error/backpressure/recovery left a payload or FIFO token");
}

} // namespace

int sc_main(int argc, char **argv) {
  try {
    const unsigned mode = argc > 1 ? static_cast<unsigned>(std::stoul(argv[1])) : 0;
    Check(mode <= 2, "invalid test memory mode");
    RejectMetadata();
    RepeatedAluCounters();
    TaskLoopReconvergence(false);
    TaskLoopReconvergence(true);
    UnpredicatedFence();
    AtomicMaskAndFence();
    MemoryPool pool;
    GpuMemorySystem memory(static_cast<MemoryMode>(mode));
    sc_core::sc_fifo<ComputeDispatchTxn> input("dispatch", 1), completion("completion", 1);
    sc_core::sc_fifo<ComputeWorkgroupTxn> groups("groups", 1), group_done("group_done", 1);
    sc_core::sc_fifo<ComputeMemoryTxn> requests("requests", 1), responses("responses", 1);
    ComputeDataMaster cdm("compute_data_master", pool, memory);
    ComputeShader shader("compute_shader", pool);
    cdm.input(input); cdm.completion(completion);
    cdm.workgroup_output(groups); cdm.workgroup_completion(group_done);
    cdm.memory_input(requests); cdm.memory_output(responses);
    shader.input(groups); shader.output(group_done);
    shader.memory_request_output(requests); shader.memory_response_input(responses);
    RunCopy(pool, memory, input, completion, 30, 1, false, 1);
    RunCopy(pool, memory, input, completion, 30, 5, false, 2);
    RunCopy(pool, memory, input, completion, 30, 1, true, 3);
    RunCopy(pool, memory, input, completion, 30, 1, false, 4);
    RunCopy(pool, memory, input, completion, 32, 7, false, 5);
    RunNativeLoop(pool, memory, input, completion, false, 6);
    RunNativeLoop(pool, memory, input, completion, true, 7);
    RunNativeAtomic(pool, memory, input, completion, false, 8);
    RunNativeAtomic(pool, memory, input, completion, true, 9);
    AtomicServiceErrors(pool, memory, requests, responses);
    AtomicServiceAllOps(pool, memory, requests, responses);
    MutexServiceOwnership(pool, requests, responses);
    RunNativeCas(pool, memory, input, completion, 0, 20);
    RunNativeCas(pool, memory, input, completion, 1, 21);
    RunNativeCas(pool, memory, input, completion, 0, 22); // END-error lock recovery
    RunNativeCas(pool, memory, input, completion, 2, 23);
    RunNativeCas(pool, memory, input, completion, 0, 24); // in-lock memory-fault recovery
    Check(groups.num_available() == 0 && group_done.num_available() == 0 &&
              requests.num_available() == 0 && responses.num_available() == 0,
          "compute workgroup or memory FIFO failed to drain");
    std::cout << "compute FIFO-depth-one lockstep/ownership test passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
