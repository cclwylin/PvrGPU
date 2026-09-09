#pragma once

#include "model_types.h"

#include <array>
#include <string>

namespace pvrgpu::stub {

// Check structured payloads against native descriptor words before relocation.
// A hole carries four zero words and no payload; LD from it has no valid range.
inline bool ValidateDriverUniformBuffers(const DriverCommand &command,
                                          std::string *error) {
  const auto reject = [&](const char *message) {
    if (error)
      *error = std::string("PCO uniform buffer ") + message;
    return false;
  };
  if (command.uniform_buffers.size() > 5U * kMaximumUniformBuffersPerStage)
    return reject("payload count exceeds the stage limits");
  std::array<std::array<const DriverPcoUniformBuffer *,
                        kMaximumUniformBuffersPerStage>, 5> payloads{};
  for (const auto &buffer : command.uniform_buffers) {
    const auto stage = static_cast<unsigned>(buffer.stage);
    if (stage > 4 || buffer.block_index >= kMaximumUniformBuffersPerStage ||
        buffer.bytes.empty() || buffer.bytes.size() > kMaximumUniformBufferBytes)
      return reject("payload stage/index/size is invalid");
    if (payloads[stage][buffer.block_index])
      return reject("stage/block index is duplicated");
    payloads[stage][buffer.block_index] = &buffer;
  }
  for (unsigned stage = 0; stage < 5; ++stage) {
    const auto &abi = stage == 4 ? command.tessellation.evaluation_abi
                     : stage == 3 ? command.tessellation.control_abi
                     : stage == 2 ? command.geometry_pco_abi
                     : stage ? command.fragment_pco_abi : command.vertex_pco_abi;
    const auto &shared = stage == 4 ? command.tessellation.evaluation_shared
                        : stage == 3 ? command.tessellation.control_shared
                        : stage == 2 ? command.geometry_shared
                        : stage ? command.fragment_shared : command.vertex_shared;
    const std::size_t count = abi.uniform_buffer_descriptor_count;
    const std::size_t start = abi.uniform_buffer_descriptor_start;
    const std::size_t native_base = stage >= 3 && !command.tessellation.evaluation_pco.empty()
        ? (stage == 3 ? 8U + 20U * command.tessellation_control_sampled_texture_count
                      : 4U + 20U * command.tessellation_evaluation_sampled_texture_count)
        : stage == 2 && !command.geometry_pco.empty()
            ? 4U + 20U * command.geometry_sampled_texture_count : 0U;
    if (count > kMaximumUniformBuffersPerStage ||
        (count == 0 && start != native_base))
      return reject("descriptor range is invalid");
    if (count != 0) {
      const std::size_t textures = stage ? command.fragment_sampled_texture_count
                                        : command.vertex_sampled_texture_count;
      if (start != (stage >= 2 ? native_base : textures * 20U) || start > shared.size() ||
          count * kUniformBufferDescriptorDwordCount > shared.size() - start ||
          shared.size() != abi.shareds ||
          abi.push_constant_start != start + count * kUniformBufferDescriptorDwordCount +
              (stage == 1 ? command.fragment_image_descriptor_count * 8U : 0U) ||
          abi.push_constant_count > shared.size() - abi.push_constant_start ||
          abi.push_constant_start + abi.push_constant_count != shared.size())
        return reject("descriptor/texture/push-constant layout is inconsistent");
    }
    for (std::size_t block = 0; block < kMaximumUniformBuffersPerStage; ++block) {
      const auto *payload = payloads[stage][block];
      if (block >= count) {
        if (payload)
          return reject("payload block exceeds its stage descriptor range");
        continue;
      }
      const std::size_t word = start + block * kUniformBufferDescriptorDwordCount;
      if (shared[word] != 0 || shared[word + 1] != 0 || shared[word + 3] != 0 ||
          shared[word + 2] != (payload ? payload->bytes.size() : 0U))
        return reject("descriptor address/size/dynamic offset is not canonical");
    }
  }
  return true;
}

}  // namespace pvrgpu::stub
