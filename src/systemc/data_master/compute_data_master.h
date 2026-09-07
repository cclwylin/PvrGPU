// Compute dispatch and memory-service boundary. Workgroups and native shader
// memory requests use independent bounded FIFOs so dispatch backpressure
// cannot prevent a resident task from receiving its memory completion.
#pragma once

#include "compute_types.h"
#include "memory_pool.h"

#include <systemc>

namespace pvrgpu::stub {

class GpuMemorySystem;

class ComputeDataMaster final : public sc_core::sc_module {
 public:
  sc_core::sc_fifo_in<ComputeDispatchTxn> input{"input"};
  sc_core::sc_fifo_out<ComputeDispatchTxn> completion{"completion"};
  sc_core::sc_fifo_out<ComputeWorkgroupTxn> workgroup_output{"workgroup_output"};
  sc_core::sc_fifo_in<ComputeWorkgroupTxn> workgroup_completion{
      "workgroup_completion"};
  sc_core::sc_fifo_in<ComputeMemoryTxn> memory_input{"memory_input"};
  sc_core::sc_fifo_out<ComputeMemoryTxn> memory_output{"memory_output"};

  ComputeDataMaster(sc_core::sc_module_name name, MemoryPool &pool,
                    GpuMemorySystem &memory);

 private:
  void DispatchRun();
  void MemoryRun();
  void ApplyMutex(const ComputeMemoryTxn &request);
  struct MutexOwner {
    PoolHandle dispatch;
    PoolHandle task;
  };
  std::array<MutexOwner,16> mutex_owners_{};
  MemoryPool &pool_;
  GpuMemorySystem &memory_;
};

}  // namespace pvrgpu::stub
