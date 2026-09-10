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
#include "common/msaa.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class NonFiniteDriverPlane : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

std::int64_t CheckedAdd(std::int64_t lhs, std::int64_t rhs,
                        const char *description) {
  std::int64_t result = 0;
  if (__builtin_add_overflow(lhs, rhs, &result))
    throw std::overflow_error(description);
  return result;
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

// This is deliberately a conservative proof using the ISP's exact coverage
// domain, not a tolerance on the floating-point plane determinant. No depth,
// alpha, line or shader rejection is used to declare a primitive invisible.
// It runs only after numerical plane setup has failed. Arithmetic overflow or
// invalid sample state must remain errors rather than evidence of no coverage.
bool HasEnabledRasterSample(const pvrgpu::stub::PipelineState &state,
                            const pvrgpu::stub::ParameterTriangle &triangle) {
  using namespace pvrgpu::stub;
  const RasterState &raster = state.raster_state;
  if (raster.multisample_enable > 1)
    throw std::runtime_error("ParameterBuffer multisample flag is invalid");
  const std::uint32_t enabled =
      RasterSampleMask(raster.sample_count) & raster.sample_mask;
  if (enabled == 0)
    return false;
  const bool multisample =
      raster.sample_count > 1 && raster.multisample_enable != 0;
  const ScissorState &scissor = raster.scissor;
  const std::uint32_t x_begin = std::max(
      scissor.enable ? scissor.x0 : 0U,
      static_cast<std::uint32_t>(std::max(0, triangle.min_x)));
  const std::uint32_t y_begin = std::max(
      scissor.enable ? scissor.y0 : 0U,
      static_cast<std::uint32_t>(std::max(0, triangle.min_y)));
  const std::uint32_t x_end = std::min({
      state.width, scissor.enable ? scissor.x1 : state.width,
      static_cast<std::uint32_t>(std::max(0, triangle.max_x))});
  const std::uint32_t y_end = std::min({
      state.height, scissor.enable ? scissor.y1 : state.height,
      static_cast<std::uint32_t>(std::max(0, triangle.max_y))});
  for (std::uint32_t y = y_begin; y < y_end; ++y) {
    for (std::uint32_t x = x_begin; x < x_end; ++x) {
      for (std::uint32_t sample = 0; sample < raster.sample_count; ++sample) {
        if ((enabled & (1U << sample)) == 0)
          continue;
        const auto position = multisample
            ? RasterSamplePosition(raster.sample_count, sample)
            : RasterSamplePosition(1, 0);
        const std::int64_t sample_x =
            static_cast<std::int64_t>(x) * kSubpixelScale +
            position[0] * (kSubpixelScale / 16);
        const std::int64_t sample_y =
            static_cast<std::int64_t>(y) * kSubpixelScale +
            position[1] * (kSubpixelScale / 16);
        bool covered = true;
        for (const EdgeEquation &edge : triangle.edge) {
          const std::int64_t ax = CheckedMul(
              edge.a, sample_x, "ParameterBuffer sample edge product overflow");
          const std::int64_t by = CheckedMul(
              edge.b, sample_y, "ParameterBuffer sample edge product overflow");
          const std::int64_t value = CheckedAdd(
              CheckedAdd(ax, by, "ParameterBuffer sample edge sum overflow"),
              edge.c, "ParameterBuffer sample edge sum overflow");
          if (value < 0 || (value == 0 && edge.inclusive == 0)) {
            covered = false;
            break;
          }
        }
        if (covered)
          return true;
      }
    }
  }
  return false;
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

// Keep llvmpipe setup's individual LLVM FMul/FSub/FAdd/FDiv instructions
// individually rounded.  In particular, the offset slope is formed from a
// cross product before multiplying by reciprocal area; deriving it from the
// already-serialized depth plane would change the rounding order.
float StrictMultiply(float lhs, float rhs) {
  const volatile float result = lhs * rhs;
  return result;
}

float StrictSubtract(float lhs, float rhs) {
  const volatile float result = lhs - rhs;
  return result;
}

float StrictAdd(float lhs, float rhs) {
  const volatile float result = lhs + rhs;
  return result;
}

float StrictDivide(float lhs, float rhs) {
  const volatile float result = lhs / rhs;
  return result;
}

float RoundToFloat(double value) {
  const volatile float result = static_cast<float>(value);
  return result;
}

bool IsFloatingPointDepthFormat(std::uint32_t format) {
  using namespace pvrgpu::stub;
  return format == kDriverPcoDepthFormatZ32Float ||
         format == kDriverPcoDepthFormatZ32FloatS8X24Uint;
}

double UnormDepthMrd(std::uint32_t format) {
  using namespace pvrgpu::stub;
  std::uint64_t maximum = UINT64_C(0xffffff);
  if (format == 0 || format == kDriverPcoDepthFormatZ24X8Unorm ||
      format == kDriverPcoDepthFormatZ24UnormS8Uint) {
    maximum = UINT64_C(0xffffff);
  } else if (format == kDriverPcoDepthFormatZ16Unorm) {
    maximum = UINT64_C(0xffff);
  } else if (format == kDriverPcoDepthFormatZ32Unorm) {
    maximum = UINT64_C(0xffffffff);
  } else {
    throw std::runtime_error(
        "ParameterBuffer polygon offset has an unsupported depth format");
  }
  return 1.0 / static_cast<double>(maximum);
}

float BuildLlvmPipePolygonOffset(
    const pvrgpu::stub::RasterTriangle &triangle,
    const pvrgpu::stub::RasterState &raster, std::uint32_t depth_format) {
  using namespace pvrgpu::stub;
  if (raster.polygon_offset_enable == 0)
    return 0.0F;

  const bool floating_depth = IsFloatingPointDepthFormat(depth_format);
  float units = raster.polygon_offset_units;
  if (!raster.polygon_offset_units_unscaled && !floating_depth &&
      units != 0.0F) {
    // lp_make_setup_variant_key performs this calculation on the host before
    // emitting the setup JIT.  The half-unit adjustment is deliberately made
    // in binary32, then the product with its double-precision MRD is rounded
    // back to the binary32 value embedded in the setup key.
    const float adjustment = units > 0.0F ? 0.5F : -0.5F;
    const float adjusted_units = StrictAdd(units, adjustment);
    units = RoundToFloat(static_cast<double>(adjusted_units) *
                         UnormDepthMrd(depth_format));
  }

  // llvmpipe skips the whole calculation when both setup-key inputs compare
  // equal to zero.  A nonzero clamp alone therefore cannot create an offset.
  if (raster.polygon_offset_factor == 0.0F && units == 0.0F)
    return 0.0F;

  const std::size_t i0 = triangle.setup_vertex_order[0];
  const std::size_t i1 = triangle.setup_vertex_order[1];
  const std::size_t i2 = triangle.setup_vertex_order[2];
  const float dx01 = StrictSubtract(triangle.x[i0], triangle.x[i1]);
  const float dy01 = StrictSubtract(triangle.y[i0], triangle.y[i1]);
  const float dx20 = StrictSubtract(triangle.x[i2], triangle.x[i0]);
  const float dy20 = StrictSubtract(triangle.y[i2], triangle.y[i0]);
  const float dz01 =
      StrictSubtract(triangle.window_z[i0], triangle.window_z[i1]);
  const float dz20 =
      StrictSubtract(triangle.window_z[i2], triangle.window_z[i0]);
  const float determinant = StrictSubtract(StrictMultiply(dx01, dy20),
                                           StrictMultiply(dy01, dx20));
  const float reciprocal_area = StrictDivide(1.0F, determinant);

  // lp_do_offset_tri computes cross(e,f).xy, then multiplies each component
  // by inv_det and takes its absolute value.  These products/subtractions are
  // intentionally not replaced with fabs(depth_plane.{a,b}).
  const float slope_x = std::fabs(StrictMultiply(
      StrictSubtract(StrictMultiply(dz20, dy01),
                     StrictMultiply(dy20, dz01)),
      reciprocal_area));
  const float slope_y = std::fabs(StrictMultiply(
      StrictSubtract(StrictMultiply(dx20, dz01),
                     StrictMultiply(dz20, dx01)),
      reciprocal_area));
  const float maximum_slope = slope_x > slope_y ? slope_x : slope_y;
  const float slope_offset =
      StrictMultiply(maximum_slope, raster.polygon_offset_factor);

  float unit_offset = units;
  if (floating_depth && !raster.polygon_offset_units_unscaled) {
    const float z01_max =
        std::fabs(triangle.window_z[i0]) > std::fabs(triangle.window_z[i1])
            ? std::fabs(triangle.window_z[i0])
            : std::fabs(triangle.window_z[i1]);
    const float maximum_z = std::fabs(triangle.window_z[i2]) > z01_max
                                ? std::fabs(triangle.window_z[i2])
                                : z01_max;
    const std::uint32_t exponent =
        FloatBits(maximum_z) & UINT32_C(0x7f800000);
    constexpr std::uint32_t kMantissaShift = 23U << 23U;
    const std::uint32_t mrd_bits =
        exponent > kMantissaShift ? exponent - kMantissaShift : 0U;
    unit_offset = StrictMultiply(BitsFloat(mrd_bits), units);
  }

  float offset = StrictAdd(unit_offset, slope_offset);
  if (raster.polygon_offset_clamp > 0.0F &&
      offset > raster.polygon_offset_clamp) {
    offset = raster.polygon_offset_clamp;
  } else if (raster.polygon_offset_clamp < 0.0F &&
             offset < raster.polygon_offset_clamp) {
    offset = raster.polygon_offset_clamp;
  }
  if (!std::isfinite(offset)) {
    throw NonFiniteDriverPlane(
        "ParameterBuffer produced a non-finite llvmpipe polygon offset");
  }
  return offset;
}

std::size_t DebugParameterIndex() {
  const char *text = std::getenv("PVRGPU_PARAMETER_DEBUG_INDEX");
  if (!text)
    return std::numeric_limits<std::size_t>::max();
  errno = 0;
  char *end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' ||
      value > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error(
        "PVRGPU_PARAMETER_DEBUG_INDEX must be an unsigned integer");
  }
  return static_cast<std::size_t>(value);
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

/* llvmpipe's coefficient arithmetic is binary32 and is anchored at
 * (v0 - 0.5), while fragment interpolation is evaluated at integer pixel
 * offsets.  RasterTriangle keeps the actual Mesa setup-JIT vertex order
 * separately from the model's edge-walker normalization: clean primitives
 * and clipper fan pieces have observably different subtraction sequences. */
pvrgpu::stub::ParameterCoefficientSet
BuildLlvmPipeDriverPlane(const pvrgpu::stub::RasterTriangle &triangle,
                         const float value[3]) {
  // A constant attribute has no derivatives, even when the floating-point
  // area collapses while the fixed-point raster triangle remains nonzero.
  // In particular, tessellation can generate collinear edge vertices whose
  // independently rounded subpixel coordinates form a tiny setup triangle.
  // Do not turn its constant depth/reciprocal-W into 0 * infinity / NaN.
  if (std::isfinite(value[0]) && value[0] == value[1] && value[0] == value[2]) {
    pvrgpu::stub::ParameterCoefficientSet coefficient;
    coefficient.a = FloatBits(0.f);
    coefficient.b = FloatBits(0.f);
    coefficient.c = FloatBits(value[0]);
    coefficient.pad = 0;
    return coefficient;
  }
  const std::size_t i0 = triangle.setup_vertex_order[0];
  const std::size_t i1 = triangle.setup_vertex_order[1];
  const std::size_t i2 = triangle.setup_vertex_order[2];

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
    throw NonFiniteDriverPlane(
        "ParameterBuffer produced a non-finite llvmpipe driver plane: "
        "xy=(" + std::to_string(triangle.x[i0]) + "," + std::to_string(triangle.y[i0]) +
        "),(" + std::to_string(triangle.x[i1]) + "," + std::to_string(triangle.y[i1]) +
        "),(" + std::to_string(triangle.x[i2]) + "," + std::to_string(triangle.y[i2]) +
        ") values=" + std::to_string(value[i0]) + "," + std::to_string(value[i1]) +
        "," + std::to_string(value[i2]) + " area=" + std::to_string(e-f));
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
    const RasterState &raster = state.raster_state;
    if (raster.polygon_offset_enable > 1 ||
        raster.polygon_offset_units_unscaled > 1) {
      throw std::runtime_error(
          "ParameterBuffer polygon offset control flag is invalid");
    }
    if (raster.polygon_offset_enable != 0) {
      if (!IsDriverPcoTrianglesCase(state.functional_case)) {
        throw std::runtime_error(
            "ParameterBuffer polygon offset requires a driver depth plane");
      }
      if (!std::isfinite(raster.polygon_offset_factor) ||
          !std::isfinite(raster.polygon_offset_units) ||
          !std::isfinite(raster.polygon_offset_clamp)) {
        throw std::runtime_error(
            "ParameterBuffer polygon offset state is non-finite");
      }
    }
    if (!HasPoolHandle(state.raster_triangles))
      throw std::runtime_error("ParameterBuffer received no raster triangles");
    const std::vector<RasterTriangle> triangles =
        LoadArray<RasterTriangle>(pool_, state.raster_triangles);
    if (triangles.size() != state.counters.c_primitives)
      throw std::runtime_error(
          "ParameterBuffer clip-primitive count mismatch");
    // A draw may legitimately contribute no primitives -- every one culled, or
    // (in a sequence) every one clipped away by the view volume, as dEQP's
    // fragment_ops.depth_stencil depth-visualize quads at z = -1.05 are.  The
    // parameter buffer then holds no triangles, coefficients or parameters
    // (the empty handles below already account for that) and the draw shades
    // nothing, leaving the accumulated frame untouched.

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
      if ((varying_count == 0 &&
           VaryingCoefficientDwordCount(state) != kCoefficientSetDwordCount) ||
          varying_bindings.size() != varying_count) {
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

    const std::size_t debug_parameter_index = DebugParameterIndex();
    std::vector<ParameterTriangle> parameters;
    std::vector<ParameterCoefficientSet> coefficients;
    parameters.reserve(triangles.size());
    for (std::size_t triangle_index = 0; triangle_index < triangles.size();
         ++triangle_index) {
      const RasterTriangle &triangle = triangles[triangle_index];
      const std::uint16_t expected_stride =
          static_cast<std::uint16_t>(ActiveVertexOutputDwordCount(state));
      if (triangle.vertex_output_stride_dwords != expected_stride ||
          triangle.front_facing > 1 || triangle.rasterizable > 1 ||
          triangle.face_culled > 1 ||
          (triangle.face_culled != 0 && triangle.rasterizable != 0)) {
        throw std::runtime_error(
            "ParameterBuffer received invalid RasterTriangle metadata");
      }
      if (state.functional_case == FunctionalCase::kDriverTexturedTriangles ||
          IsDriverPcoTrianglesCase(state.functional_case)) {
        bool setup_vertex_seen[3]{};
        for (std::uint8_t vertex : triangle.setup_vertex_order) {
          if (vertex >= 3 || setup_vertex_seen[vertex]) {
            throw std::runtime_error(
                "ParameterBuffer received invalid Mesa setup vertex order");
          }
          setup_vertex_seen[vertex] = true;
        }
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
      parameter.line = triangle.line;
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
        const std::int64_t x0 = QuantizeRasterSubpixel(triangle.x[edge]);
        const std::int64_t y0 = QuantizeRasterSubpixel(triangle.y[edge]);
        const std::int64_t x1 = QuantizeRasterSubpixel(triangle.x[next]);
        const std::int64_t y1 = QuantizeRasterSubpixel(triangle.y[next]);
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
        // Left edges are inclusive under both fill conventions; the
        // horizontal tie is the top edge for the top-left rule and the
        // bottom edge for the bottom-left rule (llvmpipe lp_setup_tri.c
        // applies the same swap on dcdy for bottom_edge_rule).
        const bool horizontal_inclusive =
            state.raster_state.bottom_edge_rule ? dx < 0 : dx > 0;
        equation.inclusive =
            (dy < 0 || (dy == 0 && horizontal_inclusive)) ? 1 : 0;
      }
      const std::int64_t x0 = QuantizeRasterSubpixel(triangle.x[0]);
      const std::int64_t y0 = QuantizeRasterSubpixel(triangle.y[0]);
      const std::int64_t x1 = QuantizeRasterSubpixel(triangle.x[1]);
      const std::int64_t y1 = QuantizeRasterSubpixel(triangle.y[1]);
      const std::int64_t x2 = QuantizeRasterSubpixel(triangle.x[2]);
      const std::int64_t y2 = QuantizeRasterSubpixel(triangle.y[2]);
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

      // Validate all non-numerical varying inputs before depth setup can
      // fail. An invisible primitive must not hide an invalid payload.
      float reciprocal_w[3]{};
      const std::size_t coefficient_base = coefficients.size();
      if (UsesShaderVaryings(state)) {
        parameter.coefficient_set_count = static_cast<std::uint16_t>(
            VaryingCoefficientSetCount(state));
        if (coefficient_base >
            std::numeric_limits<std::uint32_t>::max() -
                parameter.coefficient_set_count) {
          throw std::overflow_error(
              "ParameterBuffer coefficient-set range overflow");
        }
        for (std::size_t vertex = 0; vertex < 3; ++vertex) {
          reciprocal_w[vertex] = triangle.reciprocal_w[vertex];
          if (!(reciprocal_w[vertex] > 0.0F) ||
              !std::isfinite(reciprocal_w[vertex])) {
            throw std::runtime_error(
                "ParameterBuffer received invalid reciprocal W");
          }
        }
      }

      try {
        if (IsDriverPcoTrianglesCase(state.functional_case)) {
          parameter.depth_offset = FloatBits(BuildLlvmPipePolygonOffset(
              triangle, state.raster_state, state.depth_attachment_format));
          const ParameterCoefficientSet depth_plane =
              BuildLlvmPipeDriverPlane(triangle, triangle.window_z);
          parameter.depth_plane[0] = depth_plane.a;
          parameter.depth_plane[1] = depth_plane.b;
          parameter.depth_plane[2] = depth_plane.c;
          parameter.depth_plane[3] = depth_plane.pad;
          parameter.depth_plane_valid = 1;
        }

        if (UsesShaderVaryings(state)) {
          coefficients.resize(coefficient_base +
                              parameter.coefficient_set_count);

          const bool llvmpipe_driver_plane =
              state.functional_case ==
                  FunctionalCase::kDriverTexturedTriangles ||
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
                /* A varying is a shader result, and GLSL lets a shader compute
                 * a NaN or an infinity and write one out.  This stage only fits
                 * a plane through the three values it is given, so a non-finite
                 * varying is carried into the coefficients and reaches the
                 * fragment shader as the non-finite value it is, exactly as the
                 * finite ones are interpolated. */
                const float varying =
                    BitsFloat(raster_vertex_outputs[output_index]);
                if (binding.interpolation == InterpolationMode::kFlat) {
                  numerator[vertex] = varying;
                } else if (binding.interpolation ==
                           InterpolationMode::kNoPerspective) {
                  numerator[vertex] = varying;
                } else {
                  numerator[vertex] = varying * reciprocal_w[vertex];
                }
              }
              if (binding.interpolation == InterpolationMode::kFlat) {
                pvrgpu::stub::ParameterCoefficientSet coefficient;
                coefficient.a = 0;
                coefficient.b = 0;
                coefficient.c = FloatBits(
                    numerator[2]); // Provoking vertex (default is vertex 2)
                coefficient.pad = 0;
                coefficients[coefficient_base + binding.coefficient_set_base +
                             component] = coefficient;
              } else {
                try {
                  coefficients[coefficient_base + binding.coefficient_set_base +
                               component] =
                      llvmpipe_driver_plane
                          ? BuildLlvmPipeDriverPlane(triangle, numerator)
                          : BuildPlane(triangle, numerator);
                } catch (const std::runtime_error &error) {
                  const std::string description =
                      std::string(error.what()) + " varying_output=" +
                      std::to_string(binding.vertex_output_base + component) +
                      " interpolation=" +
                      std::to_string(
                          static_cast<unsigned>(binding.interpolation)) +
                      " numerator_bits=" +
                      std::to_string(FloatBits(numerator[0])) + "," +
                      std::to_string(FloatBits(numerator[1])) + "," +
                      std::to_string(FloatBits(numerator[2]));
                  if (dynamic_cast<const NonFiniteDriverPlane *>(&error))
                    throw NonFiniteDriverPlane(description);
                  throw std::runtime_error(description);
                }
              }
            }
          }
        }
      } catch (const NonFiniteDriverPlane &) {
        if (HasEnabledRasterSample(state, parameter))
          throw;
        // Tiler references already name this identity, so keep its slot and
        // setup accounting. A canonical inactive payload and empty bounds
        // ensure those references cannot reach interpolation in the ISP.
        ParameterTriangle inactive;
        inactive.key = parameter.key;
        inactive.front_facing = parameter.front_facing;
        inactive.face_culled = parameter.face_culled;
        inactive.line = parameter.line;
        inactive.first_coefficient_set = parameter.first_coefficient_set;
        std::copy(std::begin(parameter.window_z), std::end(parameter.window_z),
                  std::begin(inactive.window_z));
        parameter = inactive;
        coefficients.resize(coefficient_base);
      }
      if (triangle_index == debug_parameter_index) {
        std::cerr << "parameter-debug index=" << triangle_index
                  << " submit=" << triangle.key.submit_ordinal
                  << " draw=" << triangle.key.draw_id
                  << " api_primitive=" << triangle.key.api_primitive_id
                  << " instance=" << triangle.key.instance_id
                  << " clip_piece=" << triangle.key.clip_piece
                  << " setup_order="
                  << static_cast<unsigned>(triangle.setup_vertex_order[0])
                  << ','
                  << static_cast<unsigned>(triangle.setup_vertex_order[1])
                  << ','
                  << static_cast<unsigned>(triangle.setup_vertex_order[2]);
        for (std::size_t vertex = 0; vertex < 3; ++vertex) {
          std::cerr << " v" << vertex << "=(0x" << std::hex
                    << std::setw(8) << std::setfill('0')
                    << FloatBits(triangle.x[vertex]) << ",0x" << std::setw(8)
                    << FloatBits(triangle.y[vertex]) << ",0x" << std::setw(8)
                    << FloatBits(triangle.window_z[vertex]) << ",0x"
                    << std::setw(8) << FloatBits(triangle.reciprocal_w[vertex])
                    << std::dec << std::setfill(' ') << ')';
          std::cerr << " out" << vertex << '=';
          const std::size_t output_base =
              triangle.first_vertex_output_dword +
              vertex * triangle.vertex_output_stride_dwords;
          for (std::size_t component = 0;
               component < triangle.vertex_output_stride_dwords; ++component) {
            if (component)
              std::cerr << ',';
            std::cerr << "0x" << std::hex << std::setw(8)
                      << std::setfill('0')
                      << raster_vertex_outputs[output_base + component]
                      << std::dec << std::setfill(' ');
          }
        }
        std::cerr << " coefficients=";
        for (std::size_t coefficient = parameter.first_coefficient_set;
             coefficient < coefficients.size(); ++coefficient) {
          if (coefficient != parameter.first_coefficient_set)
            std::cerr << ',';
          const ParameterCoefficientSet &set = coefficients[coefficient];
          std::cerr << "[0x" << std::hex << std::setw(8)
                    << std::setfill('0') << set.a << ",0x" << std::setw(8)
                    << set.b << ",0x" << std::setw(8) << set.c << "]"
                    << std::dec << std::setfill(' ');
        }
        std::cerr << '\n';
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
      if (state.parameter_triangles_bytes > kParameterRegionBytes) {
        throw std::runtime_error(
            "ParameterBuffer triangle payload is larger than its DRAM region: "
            "bytes=" + std::to_string(state.parameter_triangles_bytes) +
            " region=" + std::to_string(kParameterRegionBytes) +
            " triangles=" + std::to_string(parameters.size()));
      }
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
        if (state.parameter_coefficients_bytes > kParameterRegionBytes) {
          throw std::runtime_error(
              "ParameterBuffer coefficient payload is larger than its DRAM "
              "region: bytes=" +
              std::to_string(state.parameter_coefficients_bytes) +
              " region=" + std::to_string(kParameterRegionBytes) +
              " sets=" + std::to_string(coefficients.size()));
        }
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
