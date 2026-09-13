#pragma once

#include "model_types.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace pvrgpu::stub {

inline constexpr std::size_t kMaximumGraphicsBufferResources = 64;
inline constexpr std::size_t kMaximumGraphicsBufferBindings = 64;
inline constexpr std::uint64_t kMaximumGraphicsBufferBytes =
    kDriverPcoMaximumSequencePayloadBytes;
inline constexpr std::size_t kGraphicsShaderStageCount = 5;

struct DriverGraphicsDescriptorLayout {
  std::uint32_t texture_end = 0;
  std::uint32_t uniform_end = 0;
  std::uint32_t image_start = 0;
  std::uint32_t image_end = 0;
  std::uint32_t storage_end = 0;
};

inline std::uint32_t GraphicsLowMask(std::uint32_t count) {
  return count >= 32U ? UINT32_MAX : count ? (UINT32_C(1) << count) - 1U : 0U;
}

inline const std::vector<std::uint32_t> &GraphicsStageShared(
    const DriverCommand &command, std::size_t stage) {
  switch (stage) {
  case 0: return command.vertex_shared;
  case 1: return command.fragment_shared;
  case 2: return command.geometry_shared;
  case 3: return command.tessellation.control_shared;
  case 4: return command.tessellation.evaluation_shared;
  default: throw std::runtime_error("graphics shader stage index is invalid");
  }
}

inline const DriverPcoStageAbi &GraphicsStageAbi(
    const DriverCommand &command, std::size_t stage) {
  switch (stage) {
  case 0: return command.vertex_pco_abi;
  case 1: return command.fragment_pco_abi;
  case 2: return command.geometry_pco_abi;
  case 3: return command.tessellation.control_abi;
  case 4: return command.tessellation.evaluation_abi;
  default: throw std::runtime_error("graphics shader stage index is invalid");
  }
}

inline std::uint32_t GraphicsStageTextureCount(const DriverCommand &command,
                                               std::size_t stage) {
  switch (stage) {
  case 0: return command.vertex_sampled_texture_count;
  case 1: return command.fragment_sampled_texture_count;
  case 2: return command.geometry_sampled_texture_count;
  case 3: return command.tessellation_control_sampled_texture_count;
  case 4: return command.tessellation_evaluation_sampled_texture_count;
  default: throw std::runtime_error("graphics shader stage index is invalid");
  }
}

inline bool ResolveDriverGraphicsDescriptorLayout(
    const DriverPcoStageAbi &abi, std::uint32_t system_dwords,
    std::uint32_t texture_count, std::uint32_t image_count,
    const DriverStorageBufferAbi &storage,
    DriverGraphicsDescriptorLayout *resolved = nullptr) {
  if (texture_count > PVRGPU_SYSTEMC_MAX_PCO_TEXTURES_PER_STAGE ||
      image_count > 32U ||
      abi.uniform_buffer_descriptor_count > kMaximumUniformBuffersPerStage ||
      storage.descriptor_count > 32U)
    return false;
  const std::uint64_t texture_end =
      std::uint64_t{system_dwords} + 20U * texture_count;
  const std::uint64_t uniform_end =
      texture_end + 4U * abi.uniform_buffer_descriptor_count;
  const std::uint64_t image_end = uniform_end + 8U * image_count;
  const std::uint32_t storage_mask = GraphicsLowMask(storage.descriptor_count);
  const std::uint64_t storage_end =
      image_end + 4U * storage.descriptor_count;
  /* A zero-length UBO window is encoded as start zero by ordinary VS/FS, while
   * native GS/tess stages retain their system/texture prefix.  Following
   * image/storage windows always begin after the real prefix computed above. */
  const std::uint32_t expected_uniform_start =
      abi.uniform_buffer_descriptor_count || system_dwords
          ? static_cast<std::uint32_t>(texture_end)
          : 0U;
  if (storage_end > PVRGPU_SYSTEMC_MAX_PCO_GRAPHICS_SHARED_DWORDS_PER_STAGE ||
      abi.shareds > PVRGPU_SYSTEMC_MAX_PCO_GRAPHICS_SHARED_DWORDS_PER_STAGE ||
      abi.uniform_buffer_descriptor_start != expected_uniform_start ||
      (storage.descriptor_count
           ? storage.descriptor_start != image_end
           : storage.descriptor_start || storage.used_mask ||
                 storage.read_mask || storage.write_mask) ||
      (storage.used_mask & ~storage_mask) ||
      ((storage.read_mask | storage.write_mask) & ~storage.used_mask) ||
      abi.push_constant_start != storage_end ||
      abi.push_constant_start > abi.shareds ||
      abi.push_constant_count != abi.shareds - abi.push_constant_start)
    return false;
  if (resolved) {
    resolved->texture_end = static_cast<std::uint32_t>(texture_end);
    resolved->uniform_end = static_cast<std::uint32_t>(uniform_end);
    resolved->image_start = static_cast<std::uint32_t>(uniform_end);
    resolved->image_end = static_cast<std::uint32_t>(image_end);
    resolved->storage_end = static_cast<std::uint32_t>(storage_end);
  }
  return true;
}

inline bool ResolveDriverGraphicsDescriptorLayout(
    const DriverCommand &command, std::size_t stage,
    DriverGraphicsDescriptorLayout *resolved = nullptr) {
  const auto &abi = GraphicsStageAbi(command, stage);
  const auto &shared = GraphicsStageShared(command, stage);
  const auto &storage = command.graphics_storage[stage];
  const std::uint32_t textures = GraphicsStageTextureCount(command, stage);
  const std::uint32_t images =
      stage == 1 ? command.fragment_image_descriptor_count : 0U;
  const bool empty = shared.empty() && !textures && !images &&
                     !storage.descriptor_count && !storage.descriptor_start &&
                     !storage.used_mask && !storage.read_mask &&
                     !storage.write_mask && !abi.temps && !abi.vertex_inputs &&
                     !abi.vertex_outputs && !abi.coefficients && !abi.shareds &&
                     !abi.push_constant_start && !abi.push_constant_count &&
                     !abi.entry_offset && !abi.uniform_buffer_descriptor_start &&
                     !abi.uniform_buffer_descriptor_count;
  if (empty) {
    if (resolved)
      *resolved = {};
    return stage != 1 || (!command.fragment_image_descriptor_start &&
                           !command.fragment_image_read_mask &&
                           !command.fragment_image_write_mask);
  }
  const std::uint32_t system_dwords =
      stage == 3 ? 8U : (stage == 2 || stage == 4 ? 4U : 0U);
  DriverGraphicsDescriptorLayout layout;
  if (!ResolveDriverGraphicsDescriptorLayout(
          abi, system_dwords, textures, images, storage, &layout) ||
      shared.size() != abi.shareds ||
      (stage == 1 &&
       (images ? command.fragment_image_descriptor_start != layout.image_start
               : command.fragment_image_descriptor_start ||
                     command.fragment_image_read_mask ||
                     command.fragment_image_write_mask)))
    return false;
  if (resolved)
    *resolved = layout;
  return true;
}

inline bool ValidateDriverGraphicsShaderBuffers(const DriverCommand &command,
                                                std::string *error) {
  const auto reject = [&](const char *message) {
    if (error)
      *error = std::string("PCO graphics storage buffer ") + message;
    return false;
  };
  if (command.graphics_buffer_resources.size() > kMaximumGraphicsBufferResources ||
      command.graphics_buffer_bindings.size() > kMaximumGraphicsBufferBindings)
    return reject("resource or binding count exceeds the transport bound");

  for (std::size_t stage = 0; stage < kGraphicsShaderStageCount; ++stage) {
    const auto &storage = command.graphics_storage[stage];
    const auto &abi = GraphicsStageAbi(command, stage);
    const auto &shared = GraphicsStageShared(command, stage);
    const auto mask = GraphicsLowMask(storage.descriptor_count);
    if (!ResolveDriverGraphicsDescriptorLayout(command, stage) ||
        storage.descriptor_count > 32U ||
        (storage.used_mask & ~mask) ||
        ((storage.read_mask | storage.write_mask) & ~storage.used_mask) ||
        storage.descriptor_start > shared.size() ||
        static_cast<std::uint64_t>(storage.descriptor_count) * 4U >
            shared.size() - storage.descriptor_start ||
        (storage.descriptor_count &&
         abi.push_constant_start !=
             storage.descriptor_start + 4U * storage.descriptor_count))
      return reject("descriptor layout or use masks are invalid");
    if (!storage.descriptor_count &&
        (storage.descriptor_start || storage.used_mask || storage.read_mask ||
         storage.write_mask))
      return reject("empty stage has nonzero descriptor metadata");
  }

  std::uint64_t total = 0;
  for (std::size_t index = 0;
       index < command.graphics_buffer_resources.size(); ++index) {
    const auto &resource = command.graphics_buffer_resources[index];
    if (!resource.resource_token || resource.bytes.empty() ||
        resource.bytes.size() > kMaximumGraphicsBufferBytes ||
        resource.bytes.size() > kMaximumGraphicsBufferBytes - total)
      return reject("backing resource token or extent is invalid");
    total += resource.bytes.size();
    for (std::size_t prior = 0; prior < index; ++prior)
      if (command.graphics_buffer_resources[prior].resource_token ==
          resource.resource_token)
        return reject("backing resource token is duplicated");
  }

  std::array<std::uint32_t, kGraphicsShaderStageCount> present{};
  for (const auto &binding : command.graphics_buffer_bindings) {
    const auto stage = static_cast<std::size_t>(binding.stage);
    if (stage >= kGraphicsShaderStageCount ||
        binding.resource_index >= command.graphics_buffer_resources.size() ||
        binding.slot >= command.graphics_storage[stage].descriptor_count ||
        !(command.graphics_storage[stage].used_mask &
          (UINT32_C(1) << binding.slot)) ||
        (binding.access & ~3U) || (binding.offset & 3U) ||
        !binding.bytes_size || binding.bytes_size > UINT32_MAX)
      return reject("binding stage/slot/access/range is invalid");
    const auto &resource =
        command.graphics_buffer_resources[binding.resource_index];
    if (binding.offset > resource.bytes.size() ||
        binding.bytes_size > resource.bytes.size() - binding.offset)
      return reject("binding view exceeds its backing resource");
    const auto bit = UINT32_C(1) << binding.slot;
    if (present[stage] & bit)
      return reject("stage-local binding slot is duplicated");
    present[stage] |= bit;
    const auto &storage = command.graphics_storage[stage];
    const unsigned required = ((storage.read_mask & bit) ? 1U : 0U) |
                              ((storage.write_mask & bit) ? 2U : 0U);
    if (binding.access != required)
      return reject("binding permissions do not exactly match shader accesses");
    const auto word = storage.descriptor_start + 4U * binding.slot;
    const auto &shared = GraphicsStageShared(command, stage);
    const std::array<std::uint32_t, 4> expected{
        0U, 0U, static_cast<std::uint32_t>(binding.bytes_size), 0U};
    if (!std::equal(expected.begin(), expected.end(), shared.begin() + word))
      return reject("descriptor is not canonical before relocation");
  }
  for (std::size_t stage = 0; stage < kGraphicsShaderStageCount; ++stage) {
    const auto &storage = command.graphics_storage[stage];
    const auto &shared = GraphicsStageShared(command, stage);
    for (unsigned slot = 0; slot < storage.descriptor_count; ++slot) {
      if (present[stage] & (UINT32_C(1) << slot))
        continue;
      const auto word = storage.descriptor_start + 4U * slot;
      if (shared[word] || shared[word + 1U] || shared[word + 2U] ||
          shared[word + 3U])
        return reject("unused descriptor is not zero");
    }
  }
  return true;
}

} // namespace pvrgpu::stub
