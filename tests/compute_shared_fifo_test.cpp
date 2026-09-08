// Real native PCO workgroup shared/barrier programs across depth-one FIFOs.
#include "common/functional_types.h"
#include "data_master/compute_data_master.h"
#include "memory/gpu_memory_system.h"
#include "pco_compute_shared_fixtures.h"
#include "shader/compute_shader.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace pvrgpu::stub;
unsigned checks = 0;
void Check(bool ok, const char *message) {
  ++checks;
  if (!ok) throw std::runtime_error(message);
}
template <class T> T Load(const MemoryPool &pool, PoolHandle handle) {
  const auto &bytes = pool.Read(handle);
  Check(bytes.size() == sizeof(T), "shared test POD extent");
  T value;
  std::memcpy(&value, bytes.data(), sizeof(value));
  return value;
}
template <class T> PoolHandle Store(MemoryPool &pool, const T &value) {
  auto handle = pool.Allocate(sizeof(value));
  std::memcpy(pool.Write(handle).data(), &value, sizeof(value));
  return handle;
}

void Run(MemoryPool &pool, GpuMemorySystem &memory,
         sc_core::sc_fifo<ComputeDispatchTxn> &input,
         sc_core::sc_fifo<ComputeDispatchTxn> &completion,
         unsigned kind, unsigned groups, unsigned fault, unsigned sequence) {
  auto abi = ComputeSharedPcoAbi(kind);
  auto binary = ComputeSharedPcoFixture(kind);
  const auto program = DecodePcoProgram(ShaderStage::kCompute, binary);
  ValidateComputeProgram(program, abi);
  unsigned sleep = 0, wake = 0;
  for (const auto &instruction : program.instructions) {
    if (instruction.opcode != PcoOpcode::kMutex) continue;
    sleep += instruction.control_operation == 1;
    wake += instruction.control_operation == 2;
    if (fault == 2 && instruction.control_operation == 2)
      binary[instruction.binary_offset + 3] =
         (binary[instruction.binary_offset + 3] & 0x3fU) | 0x40U;
  }
  const unsigned lanes = abi.local_size[0];
  Check((lanes <= 32) == (!sleep && !wake), "native cross-task sleep/wakeup absent");
  if (fault == 1) abi.shared_memory_bytes -= 4; // Fault while barrier owns MUTEX.
  constexpr std::uint64_t address = UINT64_C(0x1000000000000);
  std::vector<std::uint32_t> poison(groups * lanes, 0xa5c3e17b);
  memory.HostWrite(address, poison.data(), poison.size() * 4);
  std::vector<std::uint32_t> registers(abi.stage.shareds, 0);
  registers[0] = static_cast<std::uint32_t>(address);
  registers[1] = static_cast<std::uint32_t>(address >> 32);
  registers[2] = static_cast<std::uint32_t>(poison.size() * 4);
  ComputeDispatchState state;
  state.abi = abi; state.grid = {groups, 1, 1}; state.sequence = sequence;
  state.code = StoreNewArray(pool, binary);
  state.shared_registers = StoreNewArray(pool, registers);
  state.buffer_ranges = StoreNewArray(pool, std::vector<ComputeBufferRange>{
    {address, poison.size() * 4, kComputeAccessWrite, 0, 1}});
  const auto handle = Store(pool, state);
  Check(input.nb_write({handle, sequence}), "shared dispatch FIFO not empty");
  ComputeDispatchTxn done;
  bool received = false;
  for (unsigned timeout = 0; timeout < 100 && !received; ++timeout) {
    sc_core::sc_start(sc_core::sc_time(1, sc_core::SC_MS));
    received = completion.nb_read(done);
  }
  Check(received && done.state.slot == handle.slot && done.sequence == sequence,
        "shared dispatch did not complete or lost ownership");
  const auto final = Load<ComputeDispatchState>(pool, handle);
  if (fault) {
    Check(final.failed != 0, "shared fault/deadlock was silently accepted");
    Check(std::string(final.error.data()).find(fault == 1 ? "permitted views" : "deadlocked") !=
             std::string::npos, "shared fault failed for unrelated reason");
  } else {
    if (final.failed) throw std::runtime_error(final.error.data());
    Check(final.stats.workgroups == groups && final.stats.invocations == groups * lanes,
          "shared workgroup/invocation counters");
    Check(final.stats.load_instructions >= groups * lanes &&
          final.stats.store_instructions >= 2 * groups * lanes &&
          final.stats.memory_instructions >= final.stats.load_instructions +
                                             final.stats.store_instructions,
          "shared native accesses were omitted from memory accounting");
    Check(memory.mode() == MemoryMode::kCache ?
          (final.counters.slc_read_accesses > 0 && final.counters.slc_write_accesses > 0) :
          (final.stats.direct_read_bytes + final.stats.dram_read_bytes > 0 &&
           final.stats.direct_write_bytes + final.stats.dram_write_bytes > 0),
          "shared shader bypassed modeled memory traffic");
    const auto readback = memory.Readback(address, poison.size() * 4,
                                          MemoryClient::kComputeReadback);
    std::vector<std::uint32_t> actual(poison.size());
    std::memcpy(actual.data(), readback.data.data(), readback.data.size());
    for (unsigned group = 0; group < groups; ++group) {
      if (kind == 30)
        std::sort(actual.begin() + group * lanes, actual.begin() + (group + 1) * lanes);
      for (unsigned lane = 0; lane < lanes; ++lane) {
        const auto base = (lanes - 1) * (lanes - 1) + group * lanes;
        const auto expected = kind == 30 ? 2 * base + lanes + lane :
          (lanes - lane - 1) * (lanes - lane - 1) + group * lanes;
        Check(actual[group * lanes + lane] == expected,
              "shared reverse/atomic/barrier or workgroup isolation mismatch");
      }
    }
    if (kind == 30)
      Check(final.stats.atomic_instructions == groups * lanes,
            "shared atomic old values or RMW count changed");
  }
  pool.Release(state.code); pool.Release(state.shared_registers);
  pool.Release(state.buffer_ranges); pool.Release(handle);
  Check(pool.allocations() == pool.releases() && pool.bytes_in_flight() == 0,
        "shared resident/wait/error state leaked a pool payload");
}
} // namespace

int sc_main(int argc, char **argv) {
  try {
    const unsigned mode = argc > 1 ? static_cast<unsigned>(std::stoul(argv[1])) : 0;
    Check(mode <= 2, "shared memory mode");
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
    unsigned sequence = 0;
    for (unsigned epoch = 0; epoch < 3; ++epoch)
      for (unsigned kind = 24; kind <= 30; ++kind)
        Run(pool, memory, input, completion, kind, epoch + 1, 0, ++sequence);
    Run(pool, memory, input, completion, 27, 1, 1, ++sequence);
    Run(pool, memory, input, completion, 27, 2, 0, ++sequence);
    Run(pool, memory, input, completion, 27, 1, 2, ++sequence);
    Run(pool, memory, input, completion, 27, 2, 0, ++sequence);
    Check(groups.num_available() == 0 && group_done.num_available() == 0 &&
          requests.num_available() == 0 && responses.num_available() == 0,
          "shared FIFO did not drain");
    std::cout << "compute shared native FIFO: PASS " << checks << " checks\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
