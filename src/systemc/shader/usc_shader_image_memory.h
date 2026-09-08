// Native fragment DMA service. This is an USC memory adapter, not a shader
// interpreter or a Compute dispatch. All values originate in GPU storage.
#pragma once

#include "common/shader_image_types.h"
#include "memory/gpu_memory_system.h"
#include "shader/pco_iss.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pvrgpu::stub {

class UscShaderImageMemory final {
public:
  UscShaderImageMemory(GpuMemorySystem *memory, MemoryMode mode,
                       std::vector<ShaderImageResource> resources)
      : memory_(memory), resources_(std::move(resources)) {
    if ((!resources_.empty() && (!memory_ || memory_->mode() != mode)) ||
        resources_.size() > 32)
      throw std::runtime_error("USC image memory service/mode/count is invalid");
    std::uint32_t slots = 0;
    for (const auto &image : resources_) {
      if (!image.resource_token || !image.gpu_address || image.gpu_address % 4 ||
          !image.bytes || image.bytes > UINT64_C(256) * 1024 * 1024 ||
          image.gpu_address > UINT64_MAX - image.bytes || image.image_slot >= 32 ||
          (slots & (UINT32_C(1) << image.image_slot)) || image.format != 1 ||
          image.access != 3 || image.depth != 1 || image.texel_bytes != 4 ||
          !image.width || !image.height || image.offset % 4 || image.row_stride % 4 ||
          std::uint64_t{image.width} * 4 > image.row_stride ||
          image.layer_stride < std::uint64_t{image.row_stride} * image.height ||
          image.offset > image.bytes ||
          std::uint64_t{image.height - 1} * image.row_stride +
              std::uint64_t{image.width} * 4 > image.bytes - image.offset)
        throw std::runtime_error("USC image view is outside its R32UI resource");
      slots |= UINT32_C(1) << image.image_slot;
    }
  }

  static std::uint32_t Atomic32(void *user_data, PcoOpcode operation,
                                std::uint64_t address, std::uint32_t operand) {
    if (!user_data)
      throw std::runtime_error("USC atomic has no image memory context");
    return static_cast<UscShaderImageMemory *>(user_data)->Atomic(
        operation, address, operand);
  }

  std::vector<std::uint8_t> Readback(const ShaderImageResource &image) {
    if (!memory_ || !(image.access & 2))
      throw std::runtime_error("USC image readback has no writable resource");
    const auto result = memory_->Read(image.gpu_address, image.bytes,
                                      MemoryClient::kShaderImageReadback);
    if (result.data.size() != image.bytes)
      throw std::runtime_error("USC image readback is incomplete");
    stats_ += result.stats;
    return result.data;
  }
  const MemoryAccessStats &stats() const noexcept { return stats_; }
  std::uint64_t atomics() const noexcept { return atomics_; }

private:
  std::uint32_t Atomic(PcoOpcode operation, std::uint64_t address,
                       std::uint32_t operand) {
    if (!memory_ || !IsPcoAtomic32(operation) || address % 4)
      throw std::runtime_error("USC image atomic operation/alignment is invalid");
    const bool in_view = std::any_of(resources_.begin(), resources_.end(),
        [&](const auto &image) {
          const std::uint64_t begin = image.gpu_address + image.offset;
          if (address < begin) return false;
          const std::uint64_t offset = address - begin;
          return offset / image.row_stride < image.height &&
                 offset % image.row_stride < std::uint64_t{image.width} * 4;
        });
    if (!in_view)
      throw std::runtime_error("USC atomic address exceeds its bound image view");
    // No SystemC wait occurs between the read and write. The USC process
    // serializes each complete RMW, matching the modeled Compute DMA service.
    const auto read = memory_->Read(address, 4, MemoryClient::kFragmentImage);
    if (read.data.size() != 4)
      throw std::runtime_error("USC atomic read returned incomplete data");
    std::uint32_t old = 0;
    std::memcpy(&old, read.data.data(), 4);
    std::uint32_t next = 0;
    const auto signed_bits = [](std::uint32_t bits) {
      std::int32_t value = 0; std::memcpy(&value, &bits, 4); return value;
    };
    switch (operation) {
    case PcoOpcode::kAtomicAdd32: next = old + operand; break;
    case PcoOpcode::kAtomicSub32: next = old - operand; break;
    case PcoOpcode::kAtomicExchange32: next = operand; break;
    case PcoOpcode::kAtomicUnsignedMin32: next = std::min(old, operand); break;
    case PcoOpcode::kAtomicUnsignedMax32: next = std::max(old, operand); break;
    case PcoOpcode::kAtomicSignedMin32: next = signed_bits(old) < signed_bits(operand) ? old : operand; break;
    case PcoOpcode::kAtomicSignedMax32: next = signed_bits(old) > signed_bits(operand) ? old : operand; break;
    case PcoOpcode::kAtomicAnd32: next = old & operand; break;
    case PcoOpcode::kAtomicOr32: next = old | operand; break;
    case PcoOpcode::kAtomicXor32: next = old ^ operand; break;
    default: throw std::runtime_error("USC atomic operation is unsupported");
    }
    std::vector<std::uint8_t> bytes(4);
    std::memcpy(bytes.data(), &next, 4);
    stats_ += read.stats;
    stats_ += memory_->Write(address, bytes.data(), bytes.size(),
                             MemoryClient::kFragmentImage);
    ++atomics_;
    return old;
  }

  GpuMemorySystem *memory_;
  std::vector<ShaderImageResource> resources_;
  MemoryAccessStats stats_;
  std::uint64_t atomics_ = 0;
};

} // namespace pvrgpu::stub
