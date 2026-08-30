// PipelineState MemoryPool serialization and event-driven timing helpers.
// 中文：只做固定大小 state 搬移與單次 completion wait，不產生 clock edge。
#include "common/pipeline_state.h"

#include <systemc>

#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace pvrgpu::stub {

static_assert(std::is_trivially_copyable_v<PipelineState>);

PipelineState LoadPipelineState(const MemoryPool& pool, PoolHandle handle) {
  const auto& bytes = pool.Read(handle);
  if (bytes.size() != sizeof(PipelineState))
    throw std::runtime_error("MemoryPool payload is not a PipelineState");
  PipelineState state;
  std::memcpy(&state, bytes.data(), sizeof(state));
  return state;
}

void StorePipelineState(MemoryPool& pool, PoolHandle handle,
                        const PipelineState& state) {
  auto& bytes = pool.Write(handle);
  if (bytes.size() != sizeof(PipelineState))
    throw std::runtime_error("MemoryPool payload is not a PipelineState");
  std::memcpy(bytes.data(), &state, sizeof(state));
}

std::uint64_t CeilDivide(std::uint64_t value, std::uint64_t divisor) {
  if (divisor == 0)
    throw std::invalid_argument("CeilDivide divisor must be non-zero");
  return value / divisor + (value % divisor != 0);
}

void WaitForCycles(std::uint64_t cycles) {
  if (cycles == 0)
    return;
  sc_core::wait(
      sc_core::sc_time(static_cast<double>(cycles), sc_core::SC_NS));
}

}  // namespace pvrgpu::stub
