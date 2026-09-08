// StreamOutput's FIFO-visible tables own no host pointers. Native exports
// are written to modeled GPU resources; completed snapshots own pool handles.
#pragma once

#include "model_types.h"

#include <cstdint>
#include <vector>

namespace pvrgpu::stub {

inline constexpr std::uint32_t kMaximumStreamOutputBuffers = 4;
inline constexpr std::uint32_t kMaximumStreamOutputBindings = 64;
inline constexpr std::uint64_t kMaximumStreamOutputResourceBytes = UINT64_C(256) * 1024 * 1024;

struct StreamOutputBinding {
  std::uint32_t output_dword = 0;
  std::uint32_t num_components = 0;
  std::uint32_t output_buffer = 0;
  std::uint32_t dst_offset_dwords = 0;
  std::uint32_t stream = 0;
};

struct StreamOutputTarget {
  std::uint32_t output_buffer = 0;
  std::uint64_t resource_token = 0;
  std::uint64_t target_token = 0;
  std::uint64_t gpu_address = 0;
  std::uint64_t bytes_size = 0;
  std::uint32_t buffer_offset = 0;
  std::uint32_t buffer_size = 0;
  std::uint32_t internal_offset = 0;
  std::uint32_t stride_dwords = 0;
  PoolHandle readback;
};

// Pure, fail-closed transport validation. A missing bound target is legal:
// as in draw_pt_so_emit.c, it makes each complete primitive overflow.
void ValidateStreamOutputLayout(const std::vector<StreamOutputBinding> &bindings,
                                const std::vector<StreamOutputTarget> &targets,
                                std::uint32_t vertex_output_dwords);

}  // namespace pvrgpu::stub
