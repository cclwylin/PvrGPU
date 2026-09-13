// Stage-neutral bounded graphics SSBO service. Whole resources share one GPU
// allocation; stage-local ranges authorize each LD/ST/atomic operation.
#pragma once

#include "common/tessellation_state.h"
#include "memory/gpu_memory_system.h"
#include "shader/pco_iss.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pvrgpu::stub {

class UscShaderBufferMemory final {
public:
  UscShaderBufferMemory(GpuMemorySystem *memory, MemoryMode mode,
                        std::vector<ShaderBufferResource> resources,
                        std::vector<ShaderBufferRange> ranges,
                        MemoryClient client)
      : memory_(memory), resources_(std::move(resources)),
        ranges_(std::move(ranges)), client_(client) {
    if ((!resources_.empty() || !ranges_.empty()) &&
        (!memory_ || memory_->mode() != mode))
      throw std::runtime_error("USC shader-buffer memory service/mode mismatch");
    if (resources_.size() > 64 || ranges_.size() > 64)
      throw std::runtime_error("USC shader-buffer resource count exceeds ABI");
    for (std::size_t index = 0; index < resources_.size(); ++index) {
      const auto &resource = resources_[index];
      if (!resource.resource_token || !resource.gpu_address ||
          (resource.gpu_address & 3U) || !resource.bytes ||
          resource.gpu_address > UINT64_MAX - resource.bytes ||
          resource.readback.generation == 0)
        throw std::runtime_error("USC shader-buffer backing resource is invalid");
      for (std::size_t prior = 0; prior < index; ++prior)
        if (resources_[prior].resource_token == resource.resource_token ||
            (resource.gpu_address < resources_[prior].gpu_address + resources_[prior].bytes &&
             resources_[prior].gpu_address < resource.gpu_address + resource.bytes))
          throw std::runtime_error("USC shader-buffer resources overlap or duplicate");
    }
    std::uint32_t slots = 0;
    for (const auto &range : ranges_) {
      if (!range.gpu_address || (range.gpu_address & 3U) || !range.bytes ||
          range.gpu_address > UINT64_MAX - range.bytes || range.slot >= 32 ||
          (range.access & ~3U) || (slots & (UINT32_C(1) << range.slot)))
        throw std::runtime_error("USC shader-buffer stage range is invalid");
      const bool in_resource = std::any_of(resources_.begin(), resources_.end(),
          [&](const auto &resource) {
            return range.gpu_address >= resource.gpu_address &&
                   range.gpu_address - resource.gpu_address <= resource.bytes &&
                   range.bytes <= resource.bytes -
                                      (range.gpu_address - resource.gpu_address);
          });
      if (!in_resource)
        throw std::runtime_error("USC shader-buffer view has no backing resource");
      slots |= UINT32_C(1) << range.slot;
    }
  }

  bool Contains(std::uint64_t address, std::size_t bytes,
                std::uint32_t access) const {
    return bytes != 0 && std::any_of(ranges_.begin(), ranges_.end(),
        [&](const auto &range) {
          if ((range.access & access) != access || address < range.gpu_address)
            return false;
          const auto offset = address - range.gpu_address;
          return offset <= range.bytes && bytes <= range.bytes - offset;
        });
  }

  // The native memory callback carries an absolute address, not a descriptor
  // slot.  Starting in a range is therefore the best available discriminator
  // for routing a vector whose trailing DWORDs may be robustly out of bounds.
  bool StartsIn(std::uint64_t address, std::uint32_t access) const {
    return SelectRange(address, access) != ranges_.end();
  }

  bool OwnsAddress(std::uint64_t address) const {
    return std::any_of(ranges_.begin(), ranges_.end(), [&](const auto &range) {
      return address >= range.gpu_address &&
             address - range.gpu_address < range.bytes;
    });
  }

  static void Read(void *user_data, std::uint64_t address,
                   std::uint32_t dword_count, std::uint32_t *destination) {
    if (!user_data)
      throw std::runtime_error("USC shader-buffer load has no context");
    static_cast<UscShaderBufferMemory *>(user_data)->ReadDwords(
        address, dword_count, destination);
  }

  static void Write(void *user_data, std::uint64_t address,
                    std::uint32_t dword_count,
                    const std::uint32_t *source) {
    if (!user_data)
      throw std::runtime_error("USC shader-buffer store has no context");
    static_cast<UscShaderBufferMemory *>(user_data)->WriteDwords(
        address, dword_count, source);
  }

  static std::uint32_t Atomic32(void *user_data, PcoOpcode operation,
                                std::uint64_t address,
                                std::uint32_t operand) {
    if (!user_data)
      throw std::runtime_error("USC shader-buffer atomic has no context");
    return static_cast<UscShaderBufferMemory *>(user_data)->Atomic(
        operation, address, operand);
  }

  std::vector<std::uint8_t> Readback(const ShaderBufferResource &resource) {
    if (!memory_ || !(resource.access & 2U))
      throw std::runtime_error("USC shader-buffer readback is not writable");
    const auto result = memory_->Read(resource.gpu_address, resource.bytes,
                                      MemoryClient::kShaderImageReadback);
    if (result.data.size() != resource.bytes)
      throw std::runtime_error("USC shader-buffer readback is incomplete");
    stats_ += result.stats;
    return result.data;
  }

  const MemoryAccessStats &stats() const noexcept { return stats_; }
  std::uint64_t atomics() const noexcept { return atomics_; }

private:
  using RangeIterator = std::vector<ShaderBufferRange>::const_iterator;

  RangeIterator SelectRange(std::uint64_t address, std::uint32_t access) const {
    RangeIterator selected = ranges_.end();
    std::uint64_t selected_remaining = 0;
    for (auto range = ranges_.begin(); range != ranges_.end(); ++range) {
      if ((range->access & access) != access || address < range->gpu_address)
        continue;
      const std::uint64_t offset = address - range->gpu_address;
      if (offset >= range->bytes)
        continue;
      const std::uint64_t remaining = range->bytes - offset;
      if (selected == ranges_.end() || remaining > selected_remaining) {
        selected = range;
        selected_remaining = remaining;
      }
    }
    return selected;
  }

  std::size_t ValidPrefixBytes(std::uint64_t address, std::size_t bytes,
                               std::uint32_t access) const {
    const auto range = SelectRange(address, access);
    if (range == ranges_.end())
      return 0;
    const std::uint64_t remaining =
        range->bytes - (address - range->gpu_address);
    return static_cast<std::size_t>(
        std::min<std::uint64_t>(bytes, remaining - remaining % 4U));
  }

  void RejectPermissionMismatch(std::uint64_t address,
                                std::uint32_t access,
                                const char *operation) const {
    if (OwnsAddress(address) && !StartsIn(address, access))
      throw std::runtime_error(std::string("USC shader-buffer ") + operation +
                               " is not permitted by its stage view");
  }

  void ReadDwords(std::uint64_t address, std::uint32_t dword_count,
                  std::uint32_t *destination) {
    if (!destination || !dword_count || dword_count > 16 || (address & 3U))
      throw std::runtime_error("USC shader-buffer load shape is invalid");
    const auto bytes = static_cast<std::size_t>(dword_count) * 4U;
    std::fill_n(destination, dword_count, 0U);
    RejectPermissionMismatch(address, 1U, "load");
    const auto valid_bytes = ValidPrefixBytes(address, bytes, 1U);
    if (valid_bytes == 0)
      return;
    const auto read = memory_->Read(address, valid_bytes, client_);
    if (read.data.size() != valid_bytes)
      throw std::runtime_error("USC shader-buffer load returned incomplete data");
    std::memcpy(destination, read.data.data(), valid_bytes);
    stats_ += read.stats;
  }

  void WriteDwords(std::uint64_t address, std::uint32_t dword_count,
                   const std::uint32_t *source) {
    if (!source || !dword_count || dword_count > 16 || (address & 3U))
      throw std::runtime_error("USC shader-buffer store shape is invalid");
    const auto bytes = static_cast<std::size_t>(dword_count) * 4U;
    RejectPermissionMismatch(address, 2U, "store");
    const auto valid_bytes = ValidPrefixBytes(address, bytes, 2U);
    if (valid_bytes == 0)
      return;
    stats_ += memory_->Write(address,
                            reinterpret_cast<const std::uint8_t *>(source),
                            valid_bytes, client_);
  }

  std::uint32_t Atomic(PcoOpcode operation, std::uint64_t address,
                       std::uint32_t operand) {
    if (!IsPcoAtomic32(operation))
      throw std::runtime_error("USC shader-buffer atomic opcode is unsupported");
    if (address & 3U)
      throw std::runtime_error("USC shader-buffer atomic address is unaligned");
    RejectPermissionMismatch(address, 3U, "atomic");
    if (!Contains(address, 4, 3U))
      return 0;
    const auto read = memory_->Read(address, 4, client_);
    if (read.data.size() != 4)
      throw std::runtime_error("USC shader-buffer atomic read is incomplete");
    std::uint32_t old = 0;
    std::memcpy(&old, read.data.data(), 4);
    const auto signed_bits = [](std::uint32_t bits) {
      std::int32_t value = 0;
      std::memcpy(&value, &bits, 4);
      return value;
    };
    std::uint32_t next = 0;
    switch (operation) {
    case PcoOpcode::kAtomicAdd32: next = old + operand; break;
    case PcoOpcode::kAtomicSub32: next = old - operand; break;
    case PcoOpcode::kAtomicExchange32: next = operand; break;
    case PcoOpcode::kAtomicUnsignedMin32: next = std::min(old, operand); break;
    case PcoOpcode::kAtomicUnsignedMax32: next = std::max(old, operand); break;
    case PcoOpcode::kAtomicSignedMin32:
      next = signed_bits(old) < signed_bits(operand) ? old : operand; break;
    case PcoOpcode::kAtomicSignedMax32:
      next = signed_bits(old) > signed_bits(operand) ? old : operand; break;
    case PcoOpcode::kAtomicAnd32: next = old & operand; break;
    case PcoOpcode::kAtomicOr32: next = old | operand; break;
    case PcoOpcode::kAtomicXor32: next = old ^ operand; break;
    default: throw std::runtime_error("USC shader-buffer atomic opcode is unsupported");
    }
    std::array<std::uint8_t, 4> bytes{};
    std::memcpy(bytes.data(), &next, 4);
    stats_ += read.stats;
    stats_ += memory_->Write(address, bytes.data(), bytes.size(), client_);
    ++atomics_;
    return old;
  }

  GpuMemorySystem *memory_ = nullptr;
  std::vector<ShaderBufferResource> resources_;
  std::vector<ShaderBufferRange> ranges_;
  MemoryClient client_ = MemoryClient::kVertexShader;
  MemoryAccessStats stats_;
  std::uint64_t atomics_ = 0;
};

} // namespace pvrgpu::stub
