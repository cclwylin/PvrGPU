#pragma once

#include "memory_pool.h"
#include <cstdint>
#include <type_traits>

namespace pvrgpu::stub {

// FS stage-local image view and backing resource. Readback is a distinct pool
// allocation per view; aliases share GPU storage, not ownership of this handle.
struct ShaderImageResource {
  std::uint64_t resource_token = 0;
  std::uint64_t gpu_address = 0;
  std::uint64_t bytes = 0;
  std::uint64_t offset = 0;
  std::uint32_t image_slot = 0;
  std::uint32_t format = 0;
  std::uint32_t access = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t depth = 0;
  std::uint32_t row_stride = 0;
  std::uint32_t layer_stride = 0;
  std::uint32_t texel_bytes = 0;
  PoolHandle readback;
};
static_assert(std::is_trivially_copyable_v<ShaderImageResource>);

} // namespace pvrgpu::stub
