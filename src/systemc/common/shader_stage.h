#pragma once

namespace pvrgpu::stub {

enum class ShaderStage {
  kVertex,
  kFragment,
  kCompute,
  kGeometry,
  kTessellationControl,
  kTessellationEvaluation,
};

}  // namespace pvrgpu::stub
