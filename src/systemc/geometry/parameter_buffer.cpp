// Module：ParameterBuffer。
// 縮寫：非縮寫（參數緩衝區）。
// 功能：把 screen coordinates 量化為 reference uArch 的 24.8-style
// fixed-point，建立 exact top-left edge equations 與 bbox；native driver-PCO
// depth 及 smooth varying 則建立 llvmpipe-compatible A/B/C/PAD planes。
// Zero-area 與 face-culled candidate 只保留 identity placeholder，
// 不產生 raster equation。FIFO 只傳 MemoryPool handle，完成採
// event-driven wait。
#include "geometry/parameter_buffer.h"

#include "common/functional_types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

std::int64_t Quantize(float value) {
  const double scaled =
      static_cast<double>(value) * pvrgpu::stub::kSubpixelScale;
  if (!std::isfinite(scaled) ||
      scaled < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      scaled > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error(
        "ParameterBuffer fixed-point coordinate overflow");
  }
  return static_cast<std::int64_t>(std::llround(scaled));
}

std::int64_t CheckedSub(std::int64_t lhs, std::int64_t rhs,
                        const char *description) {
  std::int64_t result = 0;
  if (__builtin_sub_overflow(lhs, rhs, &result))
    throw std::overflow_error(description);
  return result;
}

std::int64_t CheckedMul(std::int64_t lhs, std::int64_t rhs,
                        const char *description) {
  std::int64_t result = 0;
  if (__builtin_mul_overflow(lhs, rhs, &result))
    throw std::overflow_error(description);
  return result;
}

std::int32_t ClampFloor(float value, std::uint32_t limit) {
  const double clamped = std::clamp(std::floor(static_cast<double>(value)), 0.0,
                                    static_cast<double>(limit));
  return static_cast<std::int32_t>(clamped);
}

std::int32_t ClampCeil(float value, std::uint32_t limit) {
  const double clamped = std::clamp(std::ceil(static_cast<double>(value)), 0.0,
                                    static_cast<double>(limit));
  return static_cast<std::int32_t>(clamped);
}

float BitsFloat(std::uint32_t bits) {
  float value = 0.0F;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

pvrgpu::stub::ParameterCoefficientSet
BuildPlane(const pvrgpu::stub::RasterTriangle &triangle,
           const float value[3]) {
  const double x0 = triangle.x[0];
  const double y0 = triangle.y[0];
  const double x1 = triangle.x[1];
  const double y1 = triangle.y[1];
  const double x2 = triangle.x[2];
  const double y2 = triangle.y[2];
  const double determinant =
      (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
  if (!(determinant > 0.0) || !std::isfinite(determinant)) {
    throw std::runtime_error(
        "ParameterBuffer cannot build a plane for a degenerate triangle");
  }
  const double a =
      ((static_cast<double>(value[1]) - value[0]) * (y2 - y0) -
       (static_cast<double>(value[2]) - value[0]) * (y1 - y0)) /
      determinant;
  const double b =
      ((x1 - x0) * (static_cast<double>(value[2]) - value[0]) -
       (x2 - x0) * (static_cast<double>(value[1]) - value[0])) /
      determinant;
  const double c = static_cast<double>(value[0]) - a * x0 - b * y0;
  const float af = static_cast<float>(a);
  const float bf = static_cast<float>(b);
  const float cf = static_cast<float>(c);
  if (!std::isfinite(af) || !std::isfinite(bf) || !std::isfinite(cf))
    throw std::runtime_error("ParameterBuffer produced a non-finite plane");
  pvrgpu::stub::ParameterCoefficientSet coefficient;
  coefficient.a = FloatBits(af);
  coefficient.b = FloatBits(bf);
  coefficient.c = FloatBits(cf);
  coefficient.pad = 0;
  return coefficient;
}

/* The validated Gallium driver command uses BACK/CW culling.  llvmpipe
 * normalizes those clockwise triangles by exchanging v0/v1 before its JIT
 * setup function computes interpolation coefficients.  Its coefficient
 * arithmetic is binary32 and is anchored at (v0 - 0.5), while fragment
 * interpolation is evaluated at integer pixel offsets.  This ordering is
 * observable when a nearest-filtered coordinate lands exactly on a texel
 * boundary, so preserve it for that narrow command instead of weakening the
 * reference-uArch plane construction used by every other functional case. */
pvrgpu::stub::ParameterCoefficientSet
BuildLlvmPipeDriverPlane(const pvrgpu::stub::RasterTriangle &triangle,
                         const float value[3]) {
  constexpr std::array<std::size_t, 3> order{1, 0, 2};
  const std::size_t i0 = order[0];
  const std::size_t i1 = order[1];
  const std::size_t i2 = order[2];

  const float x0_center = triangle.x[i0] - 0.5F;
  const float y0_center = triangle.y[i0] - 0.5F;
  const float dx01 = triangle.x[i0] - triangle.x[i1];
  const float dy01 = triangle.y[i0] - triangle.y[i1];
  const float dx20 = triangle.x[i2] - triangle.x[i0];
  const float dy20 = triangle.y[i2] - triangle.y[i0];
  const float e = dx01 * dy20;
  const float f = dy01 * dx20;
  const float reciprocal_area = 1.0F / (e - f);
  const float dx01_ooa = dx01 * reciprocal_area;
  const float dy01_ooa = dy01 * reciprocal_area;
  const float dx20_ooa = dx20 * reciprocal_area;
  const float dy20_ooa = dy20 * reciprocal_area;

  const float da01 = value[i0] - value[i1];
  const float da20 = value[i2] - value[i0];
  const float dadx_left = da01 * dy20_ooa;
  const float dadx_right = da20 * dy01_ooa;
  const float dady_left = da20 * dx01_ooa;
  const float dady_right = da01 * dx20_ooa;
  const float dadx = dadx_left - dadx_right;
  const float dady = dady_left - dady_right;
  const float origin_x = dadx * x0_center;
  const float origin_y = dady * y0_center;
  const float origin = origin_x + origin_y;
  const float attr0 = value[i0] - origin;
  if (!std::isfinite(dadx) || !std::isfinite(dady) ||
      !std::isfinite(attr0)) {
    throw std::runtime_error(
        "ParameterBuffer produced a non-finite llvmpipe driver plane");
  }

  pvrgpu::stub::ParameterCoefficientSet coefficient;
  coefficient.a = FloatBits(dadx);
  coefficient.b = FloatBits(dady);
  coefficient.c = FloatBits(attr0);
  coefficient.pad = 0;
  return coefficient;
}

} // namespace

namespace pvrgpu::stub {

namespace {

inline constexpr std::uint64_t kParameterTrianglesGpuAddress =
    UINT64_C(0x30000000);
inline constexpr std::uint64_t kParameterCoefficientsGpuAddress =
    UINT64_C(0x34000000);

} // namespace

ParameterBuffer::ParameterBuffer(sc_core::sc_module_name name, MemoryPool &pool,
                                 GpuMemorySystem *memory)
    : sc_module(name), pool_(pool), memory_(memory) {
  SC_THREAD(Run);
}

void ParameterBuffer::Run() {
  while (true) {
    const PipelineTxn txn = input.read();
    PipelineState state = LoadPipelineState(pool_, txn.state);

    RequireStage(state.stage, PipelineStage::kTiled, name());
    if (memory_ && state.memory_mode != memory_->mode())
      throw std::runtime_error("ParameterBuffer memory mode mismatch");
    if (!HasPoolHandle(state.raster_triangles))
      throw std::runtime_error("ParameterBuffer received no raster triangles");
    const std::vector<RasterTriangle> triangles =
        LoadArray<RasterTriangle>(pool_, state.raster_triangles);
    if (triangles.size() != state.counters.c_primitives)
      throw std::runtime_error(
          "ParameterBuffer clip-primitive count mismatch");
    if (triangles.empty() && !state.raster_state.face_cull.enable)
      throw std::runtime_error("ParameterBuffer received no setup triangles");

    if (!HasPoolHandle(state.raster_vertex_outputs)) {
      throw std::runtime_error(
          "ParameterBuffer received no flattened raster VTXOUT payload");
    }
    const std::vector<std::uint32_t> raster_vertex_outputs =
        LoadArray<std::uint32_t>(pool_, state.raster_vertex_outputs);
    std::vector<ShaderVaryingBinding> varying_bindings;
    if (UsesShaderVaryings(state)) {
      if (!HasPoolHandle(state.shader_varying_bindings)) {
        throw std::runtime_error(
            "ParameterBuffer varying case has no linkage payload");
      }
      varying_bindings = LoadArray<ShaderVaryingBinding>(
          pool_, state.shader_varying_bindings);
      const std::uint32_t varying_count =
          VaryingVectorCount(state);
      if (varying_count == 0 || varying_bindings.size() != varying_count) {
        throw std::runtime_error(
            "ParameterBuffer varying linkage count is invalid");
      }
      for (std::size_t index = 0; index < varying_bindings.size(); ++index) {
        if (!IsExactVaryingBinding(state, varying_bindings[index], index)) {
          throw std::runtime_error(
              "ParameterBuffer varying linkage is not exact");
        }
      }
    } else if (HasPoolHandle(state.shader_varying_bindings) ||
               HasPoolHandle(state.parameter_coefficients)) {
      throw std::runtime_error(
          "ParameterBuffer solid-color case has varying payload state");
    }

    std::vector<ParameterTriangle> parameters;
    std::vector<ParameterCoefficientSet> coefficients;
    parameters.reserve(triangles.size());
    for (const RasterTriangle &triangle : triangles) {
      const std::uint16_t expected_stride =
          UsesShaderVaryings(state)
              ? static_cast<std::uint16_t>(
                    VaryingVertexOutputDwordCount(state))
              : 4;
      if (triangle.vertex_output_stride_dwords != expected_stride ||
          triangle.front_facing > 1 || triangle.rasterizable > 1 ||
          triangle.face_culled > 1 ||
          (triangle.face_culled != 0 && triangle.rasterizable != 0) ||
          triangle.reserved[0] != 0 || triangle.reserved[1] != 0 ||
          triangle.reserved[2] != 0) {
        throw std::runtime_error(
            "ParameterBuffer received invalid RasterTriangle metadata");
      }
      const std::uint64_t vertex_output_end =
          static_cast<std::uint64_t>(triangle.first_vertex_output_dword) +
          static_cast<std::uint64_t>(3) *
              triangle.vertex_output_stride_dwords;
      if (vertex_output_end > raster_vertex_outputs.size()) {
        throw std::runtime_error(
            "ParameterBuffer raster VTXOUT range is out of bounds");
      }
      ParameterTriangle parameter;
      parameter.key = triangle.key;
      parameter.front_facing = triangle.front_facing;
      parameter.rasterizable = triangle.rasterizable;
      parameter.face_culled = triangle.face_culled;
      if (coefficients.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error(
            "ParameterBuffer coefficient-set offset overflow");
      parameter.first_coefficient_set =
          static_cast<std::uint32_t>(coefficients.size());
      for (std::size_t vertex = 0; vertex < 3; ++vertex)
        parameter.window_z[vertex] = triangle.window_z[vertex];
      if (!triangle.rasterizable) {
        parameters.push_back(parameter);
        continue;
      }
      for (std::size_t edge = 0; edge < 3; ++edge) {
        const std::size_t next = (edge + 1) % 3;
        const std::int64_t x0 = Quantize(triangle.x[edge]);
        const std::int64_t y0 = Quantize(triangle.y[edge]);
        const std::int64_t x1 = Quantize(triangle.x[next]);
        const std::int64_t y1 = Quantize(triangle.y[next]);
        EdgeEquation &equation = parameter.edge[edge];
        equation.a = CheckedSub(
            y0, y1, "ParameterBuffer edge A overflow");
        equation.b = CheckedSub(
            x1, x0, "ParameterBuffer edge B overflow");
        const std::int64_t positive = CheckedMul(
            x0, y1, "ParameterBuffer edge C product overflow");
        const std::int64_t negative = CheckedMul(
            x1, y0, "ParameterBuffer edge C product overflow");
        equation.c = CheckedSub(
            positive, negative, "ParameterBuffer edge C overflow");
        const std::int64_t dx = CheckedSub(
            x1, x0, "ParameterBuffer edge dx overflow");
        const std::int64_t dy = CheckedSub(
            y1, y0, "ParameterBuffer edge dy overflow");
        equation.inclusive = (dy < 0 || (dy == 0 && dx > 0)) ? 1 : 0;
      }
      const std::int64_t x0 = Quantize(triangle.x[0]);
      const std::int64_t y0 = Quantize(triangle.y[0]);
      const std::int64_t x1 = Quantize(triangle.x[1]);
      const std::int64_t y1 = Quantize(triangle.y[1]);
      const std::int64_t x2 = Quantize(triangle.x[2]);
      const std::int64_t y2 = Quantize(triangle.y[2]);
      const std::int64_t dx10 = CheckedSub(
          x1, x0, "ParameterBuffer x delta overflow");
      const std::int64_t dy20 = CheckedSub(
          y2, y0, "ParameterBuffer y delta overflow");
      const std::int64_t dy10 = CheckedSub(
          y1, y0, "ParameterBuffer y delta overflow");
      const std::int64_t dx20 = CheckedSub(
          x2, x0, "ParameterBuffer x delta overflow");
      const std::int64_t positive = CheckedMul(
          dx10, dy20, "ParameterBuffer area product overflow");
      const std::int64_t negative = CheckedMul(
          dy10, dx20, "ParameterBuffer area product overflow");
      parameter.signed_area = CheckedSub(
          positive, negative, "ParameterBuffer area overflow");
      if (parameter.signed_area <= 0)
        throw std::runtime_error(
            "ParameterBuffer rasterizable marker/area mismatch");

      const auto x_bounds =
          std::minmax({triangle.x[0], triangle.x[1], triangle.x[2]});
      const auto y_bounds =
          std::minmax({triangle.y[0], triangle.y[1], triangle.y[2]});
      parameter.min_x = ClampFloor(x_bounds.first, state.width);
      parameter.min_y = ClampFloor(y_bounds.first, state.height);
      parameter.max_x = ClampCeil(x_bounds.second, state.width);
      parameter.max_y = ClampCeil(y_bounds.second, state.height);

      if (IsDriverPcoTrianglesCase(state.functional_case)) {
        const ParameterCoefficientSet depth_plane =
            BuildLlvmPipeDriverPlane(triangle, triangle.window_z);
        parameter.depth_plane[0] = depth_plane.a;
        parameter.depth_plane[1] = depth_plane.b;
        parameter.depth_plane[2] = depth_plane.c;
        parameter.depth_plane[3] = depth_plane.pad;
        parameter.depth_plane_valid = 1;
      }

      if (UsesShaderVaryings(state)) {
        parameter.coefficient_set_count = static_cast<std::uint16_t>(
            VaryingCoefficientSetCount(state));
        const std::size_t coefficient_base = coefficients.size();
        if (coefficient_base >
            std::numeric_limits<std::uint32_t>::max() -
                parameter.coefficient_set_count) {
          throw std::overflow_error(
              "ParameterBuffer coefficient-set range overflow");
        }
        coefficients.resize(coefficient_base + parameter.coefficient_set_count);

        float reciprocal_w[3]{};
        for (std::size_t vertex = 0; vertex < 3; ++vertex) {
          reciprocal_w[vertex] = triangle.reciprocal_w[vertex];
          if (!(reciprocal_w[vertex] > 0.0F) ||
              !std::isfinite(reciprocal_w[vertex])) {
            throw std::runtime_error(
                "ParameterBuffer received invalid reciprocal W");
          }
        }
        const bool llvmpipe_driver_plane =
            state.functional_case == FunctionalCase::kDriverTexturedTriangles ||
            IsDriverPcoTrianglesCase(state.functional_case);
        coefficients[coefficient_base] =
            llvmpipe_driver_plane
                ? BuildLlvmPipeDriverPlane(triangle, reciprocal_w)
                : BuildPlane(triangle, reciprocal_w);

        for (const ShaderVaryingBinding &binding : varying_bindings) {
          for (std::uint8_t component = 0;
               component < binding.component_count; ++component) {
            float numerator[3]{};
            for (std::size_t vertex = 0; vertex < 3; ++vertex) {
              const std::size_t output_index =
                  triangle.first_vertex_output_dword +
                  vertex * triangle.vertex_output_stride_dwords +
                  binding.vertex_output_base + component;
              const float varying =
                  BitsFloat(raster_vertex_outputs[output_index]);
              if (!std::isfinite(varying)) {
                throw std::runtime_error(
                    "ParameterBuffer received a non-finite varying");
              }
              if (binding.interpolation == InterpolationMode::kFlat) {
                numerator[vertex] = varying;
              } else if (binding.interpolation == InterpolationMode::kNoPerspective) {
                numerator[vertex] = varying;
              } else {
                numerator[vertex] = varying * reciprocal_w[vertex];
              }
              if (!std::isfinite(numerator[vertex])) {
                throw std::runtime_error(
                    "ParameterBuffer varying/W product is non-finite");
              }
            }
            if (binding.interpolation == InterpolationMode::kFlat) {
              pvrgpu::stub::ParameterCoefficientSet coefficient;
              coefficient.a = 0;
              coefficient.b = 0;
              coefficient.c = FloatBits(numerator[2]); // Provoking vertex (default is vertex 2)
              coefficient.pad = 0;
              coefficients[coefficient_base + binding.coefficient_set_base +
                           component] = coefficient;
            } else {
              coefficients[coefficient_base + binding.coefficient_set_base +
                           component] =
                  llvmpipe_driver_plane
                      ? BuildLlvmPipeDriverPlane(triangle, numerator)
                      : BuildPlane(triangle, numerator);
            }
          }
        }
      }
      parameters.push_back(parameter);
    }

    MemoryAccessStats memory_stats;
    if (memory_) {
      state.parameter_triangles = {};
      state.parameter_triangles_gpu_address =
          parameters.empty() ? 0 : kParameterTrianglesGpuAddress;
      state.parameter_triangles_bytes =
          static_cast<std::uint64_t>(parameters.size()) *
          sizeof(ParameterTriangle);
      memory_stats += WriteMemoryArray(*memory_,
                                      state.parameter_triangles_gpu_address,
                                      parameters,
                                      MemoryClient::kParameterWrite);
    } else {
      state.parameter_triangles = StoreNewArray(pool_, parameters);
    }
    if (UsesShaderVaryings(state)) {
      if (memory_) {
        state.parameter_coefficients = {};
        state.parameter_coefficients_gpu_address =
            coefficients.empty() ? 0 : kParameterCoefficientsGpuAddress;
        state.parameter_coefficients_bytes =
            static_cast<std::uint64_t>(coefficients.size()) *
            sizeof(ParameterCoefficientSet);
        memory_stats += WriteMemoryArray(
            *memory_, state.parameter_coefficients_gpu_address, coefficients,
            MemoryClient::kParameterWrite);
      } else {
        state.parameter_coefficients = StoreNewArray(pool_, coefficients);
      }
      state.counters.parameter_coefficient_sets = coefficients.size();
      state.counters.parameter_write_bytes =
          static_cast<std::uint64_t>(coefficients.size()) *
          sizeof(ParameterCoefficientSet);
    } else {
      if (!coefficients.empty())
        throw std::runtime_error(
            "ParameterBuffer generated coefficients for a solid case");
      state.counters.parameter_coefficient_sets = 0;
      state.counters.parameter_write_bytes = 0;
      state.parameter_coefficients_gpu_address = 0;
      state.parameter_coefficients_bytes = 0;
    }
    state.stage = PipelineStage::kParameterBufferReady;

    const std::uint64_t functional_cycles =
        kReferenceUarch.parameter_base_cycles +
        CeilDivide(state.counters.setup_triangles,
                   kReferenceUarch.parameter_triangles_per_batch);
    ApplyMemoryAccessStats(state.counters, memory_stats);
    const std::uint64_t cycles =
        functional_cycles + MemoryAccessDelayCycles(memory_stats);
    state.counters.parameter_buffer_cycles = cycles;
    state.counters.tiler_cycles += cycles;
    WaitForCycles(cycles);
    StorePipelineState(pool_, txn.state, state);
    output.write(txn);
  }
}

} // namespace pvrgpu::stub
