#pragma once

#include "model_types.h"
#include <stdexcept>

namespace pvrgpu::stub {

// The public texture stage numbers already name five graphics stages. Keep
// their namespaces distinct; Compute has its separate command/resource ABI.
inline unsigned DriverTextureStageIndex(DriverPcoShaderStage stage) {
  const auto index = static_cast<unsigned>(stage);
  if (index >= 5) throw std::runtime_error("invalid graphics texture stage");
  return index;
}

inline std::uint32_t DriverTextureDescriptorStart(DriverPcoShaderStage stage) {
  switch (stage) {
    case DriverPcoShaderStage::kVertex:
    case DriverPcoShaderStage::kFragment: return 0;
    case DriverPcoShaderStage::kGeometry:
    case DriverPcoShaderStage::kTessellationEvaluation: return 4;
    case DriverPcoShaderStage::kTessellationControl: return 8;
    default: throw std::runtime_error("invalid graphics texture stage");
  }
}

inline std::uint32_t &DriverStageTextureCount(DriverCommand &command,
                                             DriverPcoShaderStage stage) {
  switch (stage) {
    case DriverPcoShaderStage::kVertex: return command.vertex_sampled_texture_count;
    case DriverPcoShaderStage::kFragment: return command.fragment_sampled_texture_count;
    case DriverPcoShaderStage::kGeometry: return command.geometry_sampled_texture_count;
    case DriverPcoShaderStage::kTessellationControl: return command.tessellation_control_sampled_texture_count;
    case DriverPcoShaderStage::kTessellationEvaluation: return command.tessellation_evaluation_sampled_texture_count;
    default: throw std::runtime_error("invalid graphics texture stage");
  }
}

inline const std::vector<std::uint32_t> &DriverTextureShared(
    const DriverCommand &command, DriverPcoShaderStage stage) {
  switch (stage) {
    case DriverPcoShaderStage::kVertex: return command.vertex_shared;
    case DriverPcoShaderStage::kFragment: return command.fragment_shared;
    case DriverPcoShaderStage::kGeometry: return command.geometry_shared;
    case DriverPcoShaderStage::kTessellationControl: return command.tessellation.control_shared;
    case DriverPcoShaderStage::kTessellationEvaluation: return command.tessellation.evaluation_shared;
    default: throw std::runtime_error("invalid graphics texture stage");
  }
}
} // namespace pvrgpu::stub
