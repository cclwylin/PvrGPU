// Independent compute shader SystemC module. Native tasks progress in
// instruction-group lockstep; GPU loads and stores use bounded CDM FIFOs.
#pragma once

#include "compute_types.h"
#include "memory_pool.h"
#include "shader/compute_iss.h"

#include <systemc>

namespace pvrgpu::stub {

class ComputeShader final : public sc_core::sc_module {
 public:
  sc_core::sc_fifo_in<ComputeWorkgroupTxn> input{"input"};
  sc_core::sc_fifo_out<ComputeWorkgroupTxn> output{"output"};
  sc_core::sc_fifo_out<ComputeMemoryTxn> memory_request_output{
      "memory_request_output"};
  sc_core::sc_fifo_in<ComputeMemoryTxn> memory_response_input{
      "memory_response_input"};

  ComputeShader(sc_core::sc_module_name name, MemoryPool &pool);

 private:
  void Run();
  static void ReadMemory(void *context, std::uint64_t address,
                          std::uint32_t count, std::uint32_t *words);
  static void WriteMemory(void *context, std::uint64_t address,
                           std::uint32_t count, const std::uint32_t *words);
  static std::uint32_t Atomic32Memory(void *context, ComputeMemoryOperation operation,
                                      std::uint64_t address, std::uint32_t operand);
  static void MutexMemory(void *context, std::uint32_t id, std::uint32_t operation);
  void ReleaseTaskMutexes();
  ComputeMemoryTxn ExchangeMemory(std::uint64_t address, std::uint32_t count,
                                  ComputeMemoryOperation operation,
                                  const std::uint32_t *words);

  MemoryPool &pool_;
  PoolHandle current_dispatch_;
  PoolHandle current_task_;
  std::uint32_t current_mutex_mask_ = 0;
  PoolHandle cached_dispatch_;
  PoolHandle cached_instructions_;
  PcoProgramSummary cached_summary_;
  std::uint64_t next_request_id_ = 0;
};

} // namespace pvrgpu::stub
