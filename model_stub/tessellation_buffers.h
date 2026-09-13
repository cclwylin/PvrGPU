#pragma once

#include "model_types.h"

#include <array>
#include <limits>
#include <string>

namespace pvrgpu::stub {

inline constexpr std::size_t kMaximumTessellationBufferResources = 64;
inline constexpr std::size_t kMaximumTessellationBufferBindings = 64;
inline constexpr std::uint64_t kMaximumTessellationBufferBytes =
    kDriverPcoMaximumSequencePayloadBytes;

inline std::uint32_t TessellationLowMask(std::uint32_t count) {
  return count >= 32U ? UINT32_MAX : count ? (UINT32_C(1) << count) - 1U : 0U;
}

inline bool ValidateDriverTessellationBuffers(const DriverCommand &command,
                                               std::string *error) {
  const auto reject = [&](const char *message) {
    if (error) *error = std::string("PCO tessellation storage buffer ") + message;
    return false;
  };
  const auto &tess = command.tessellation;
  if (tess.control_pco.empty()) {
    if (!tess.buffer_resources.empty() || !tess.buffer_bindings.empty())
      return reject("payload exists without a tessellation pipeline");
    return true;
  }
  if (tess.buffer_resources.size() > kMaximumTessellationBufferResources ||
      tess.buffer_bindings.size() > kMaximumTessellationBufferBindings)
    return reject("resource or binding count exceeds the transport bound");

  const bool generic = !command.graphics_buffer_resources.empty() ||
      !command.graphics_buffer_bindings.empty() ||
      std::any_of(command.graphics_storage.begin(),
                  command.graphics_storage.end(), [](const auto &storage) {
                    return storage.descriptor_start || storage.descriptor_count ||
                           storage.used_mask || storage.read_mask ||
                           storage.write_mask;
                  });
  if (generic &&
      (!tess.buffer_resources.empty() || !tess.buffer_bindings.empty() ||
       tess.control_storage.descriptor_start ||
       tess.control_storage.descriptor_count || tess.control_storage.used_mask ||
       tess.control_storage.read_mask || tess.control_storage.write_mask ||
       tess.evaluation_storage.descriptor_start ||
       tess.evaluation_storage.descriptor_count ||
       tess.evaluation_storage.used_mask || tess.evaluation_storage.read_mask ||
       tess.evaluation_storage.write_mask))
    return reject("legacy and outer graphics capsules overlap");

  const std::array<const DriverStorageBufferAbi *, 2> storage{
      generic ? &command.graphics_storage[3] : &tess.control_storage,
      generic ? &command.graphics_storage[4] : &tess.evaluation_storage};
  const std::array<const DriverPcoStageAbi *, 2> abi{
      &tess.control_abi, &tess.evaluation_abi};
  const std::array<const std::vector<std::uint32_t> *, 2> shared{
      &tess.control_shared, &tess.evaluation_shared};
  for (unsigned stage = 0; stage < 2; ++stage) {
    const auto &s = *storage[stage];
    const auto &a = *abi[stage];
    const auto &words = *shared[stage];
    const auto mask = TessellationLowMask(s.descriptor_count);
    const std::uint64_t expected = static_cast<std::uint64_t>(
        a.uniform_buffer_descriptor_start) + 4U * a.uniform_buffer_descriptor_count;
    const std::uint64_t storage_start = s.descriptor_count ? s.descriptor_start : expected;
    if (s.descriptor_count > 32U ||
        (s.descriptor_start && s.descriptor_start != expected) ||
        (s.descriptor_count && s.descriptor_start != expected) ||
        (s.used_mask & ~mask) || ((s.read_mask | s.write_mask) & ~s.used_mask) ||
        storage_start > words.size() ||
        static_cast<std::uint64_t>(s.descriptor_count) * 4U >
            words.size() - storage_start ||
        a.push_constant_start != storage_start + 4U * s.descriptor_count)
      return reject("descriptor layout or use masks are invalid");
  }
  if (generic)
    return true;

  std::uint64_t total = 0;
  for (std::size_t index = 0; index < tess.buffer_resources.size(); ++index) {
    const auto &resource = tess.buffer_resources[index];
    if (!resource.resource_token || resource.bytes.empty() ||
        resource.bytes.size() > kMaximumTessellationBufferBytes ||
        resource.bytes.size() > kMaximumTessellationBufferBytes - total)
      return reject("backing resource token or extent is invalid");
    total += resource.bytes.size();
    for (std::size_t prior = 0; prior < index; ++prior)
      if (tess.buffer_resources[prior].resource_token == resource.resource_token)
        return reject("backing resource token is duplicated");
  }

  std::array<std::uint32_t, 2> present{};
  for (const auto &binding : tess.buffer_bindings) {
    const unsigned stage = binding.stage == DriverPcoShaderStage::kTessellationControl ? 0U
                         : binding.stage == DriverPcoShaderStage::kTessellationEvaluation ? 1U
                         : 2U;
    if (stage >= 2 || binding.resource_index >= tess.buffer_resources.size() ||
        binding.slot >= storage[stage]->descriptor_count || (binding.access & ~3U) ||
        (binding.offset & 3U) || !binding.bytes_size || binding.bytes_size > UINT32_MAX)
      return reject("binding stage/slot/access/range is invalid");
    const auto &resource = tess.buffer_resources[binding.resource_index];
    if (binding.offset > resource.bytes.size() ||
        binding.bytes_size > resource.bytes.size() - binding.offset)
      return reject("binding view exceeds its backing resource");
    const auto bit = UINT32_C(1) << binding.slot;
    if (present[stage] & bit)
      return reject("stage-local binding slot is duplicated");
    present[stage] |= bit;
    const unsigned required = ((storage[stage]->read_mask & bit) ? 1U : 0U) |
                              ((storage[stage]->write_mask & bit) ? 2U : 0U);
    if ((binding.access & required) != required)
      return reject("binding permissions do not cover shader accesses");
    const auto word = storage[stage]->descriptor_start + 4U * binding.slot;
    const std::array<std::uint32_t, 4> expected{
        0U, 0U, static_cast<std::uint32_t>(binding.bytes_size), 0U};
    if (!std::equal(expected.begin(), expected.end(), shared[stage]->begin() + word))
      return reject("descriptor is not canonical before relocation");
  }
  for (unsigned stage = 0; stage < 2; ++stage) {
    for (unsigned slot = 0; slot < storage[stage]->descriptor_count; ++slot) {
      if (present[stage] & (UINT32_C(1) << slot)) continue;
      const auto word = storage[stage]->descriptor_start + 4U * slot;
      if ((*shared[stage])[word] || (*shared[stage])[word + 1U] ||
          (*shared[stage])[word + 2U] || (*shared[stage])[word + 3U])
        return reject("unused descriptor is not zero");
    }
  }
  return true;
}

}  // namespace pvrgpu::stub
