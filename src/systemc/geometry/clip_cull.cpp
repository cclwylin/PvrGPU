// Module：ClipCull。
// 縮寫：非縮寫（裁剪與剔除）。
// 功能：讀取 USC ISS 寫出的最多 64 個 VTXOUT；0..3 是 clip position，
// exact shader linkage 指定的其餘輸出會在 clipping intersection 同步插值，
// 並以 flat MemoryPool payload 交給 ParameterBuffer。Triangle.Setup 與
// AttributeFetchShader indexed workloads 按 Mesa 公開 clipper 的 plane-bit
// order 與 dp>=0 boundary 規則做 homogeneous 六平面 Sutherland-Hodgman
// clipping，再 fan-emit setup candidates。零面積或 clean-path face-culled
// candidate 仍保留 identity/counter，rasterizable marker 讓後續 tiler 跳過。
// FIFO 只傳 MemoryPool handle，完成採 event-driven wait。
#include "geometry/clip_cull.h"

#include "common/functional_types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using pvrgpu::stub::RasterTriangle;
using pvrgpu::stub::ShaderVaryingBinding;
using pvrgpu::stub::VertexLane;
using pvrgpu::stub::CullFaceMode;
using pvrgpu::stub::FaceCullState;
using pvrgpu::stub::FrontFaceWinding;

float BitsFloat(std::uint32_t bits) {
  float value = 0.0f;
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

struct ClipVertex {
  float output[pvrgpu::stub::kPcoVertexOutputRegisterCount]{};
  std::uint16_t output_count = 0;
  // Mesa's LLVM vertex path contracts the viewport multiply/add for original
  // vertices.  draw_pipe_clip.c recomputes window coordinates for generated
  // intersections with three separate CPU operations instead.
  bool generated_intersection = false;
};

float StrictMultiply(float left, float right) {
  volatile float result = left * right;
  return result;
}

float StrictAdd(float left, float right) {
  volatile float result = left + right;
  return result;
}

ClipVertex ReadClipVertex(const VertexLane &vertex,
                          std::uint16_t output_count) {
  if (output_count < 4 ||
      output_count > pvrgpu::stub::kPcoVertexOutputRegisterCount) {
    throw std::runtime_error("ClipCull received an invalid VTXOUT count");
  }
  ClipVertex result;
  result.output_count = output_count;
  for (std::size_t component = 0; component < output_count; ++component)
    result.output[component] = BitsFloat(vertex.vertex_output[component]);
  if (!vertex.emitted || !vertex.ended) {
    throw std::runtime_error(
        "ClipCull received a vertex without UVSW emit/end-task");
  }
  for (std::size_t component = 0; component < output_count; ++component) {
    if (!std::isfinite(result.output[component]))
      throw std::runtime_error("ClipCull received a non-finite clip vertex");
  }
  return result;
}

float PlaneDistance(const ClipVertex &vertex, std::uint32_t plane, std::uint16_t clip_dist_reg = 0) {
  if (plane >= 6) {
    return vertex.output[clip_dist_reg + (plane - 6)];
  }
  const float x = vertex.output[0];
  const float y = vertex.output[1];
  const float z = vertex.output[2];
  const float w = vertex.output[3];
  switch (plane) {
  case 0:
    return -x + w; // right: -x + w >= 0
  case 1:
    return x + w; // left:   x + w >= 0
  case 2:
    return -y + w; // top:   -y + w >= 0
  case 3:
    return y + w; // bottom:  y + w >= 0
  case 4:
    return z + w; // GL full-Z near
  case 5:
    return -z + w; // GL full-Z far
  default:
    throw std::runtime_error("ClipCull received an invalid clip plane");
  }
}

std::uint32_t ClipMask(const ClipVertex &vertex, bool depth_clamp, std::uint8_t clip_dist_mask = 0, std::uint16_t clip_dist_reg = 0) {
  std::uint32_t mask = 0;
  for (std::uint32_t plane = 0; plane < (depth_clamp ? 4U : 6U); ++plane) {
    // Mesa's clip test is expressed as !(dp >= 0), so boundary vertices are
    // inside and unordered NaNs fail closed as outside.
    if (!(PlaneDistance(vertex, plane, clip_dist_reg) >= 0.0F))
      mask |= 1U << plane;
  }
  if (clip_dist_mask != 0) {
    for (std::uint32_t i = 0; i < 8; ++i) {
      if ((clip_dist_mask & (1U << i)) != 0) {
        if (!(PlaneDistance(vertex, 6 + i, clip_dist_reg) >= 0.0F)) {
          mask |= 1U << (6 + i);
        }
      }
    }
  }
  return mask;
}

ClipVertex Interpolate(float t, const ClipVertex &outside,
                       const ClipVertex &inside) {
  if (outside.output_count != inside.output_count ||
      outside.output_count < 4 ||
      outside.output_count > pvrgpu::stub::kPcoVertexOutputRegisterCount) {
    throw std::runtime_error("ClipCull interpolation VTXOUT ranges disagree");
  }
  ClipVertex result;
  result.output_count = outside.output_count;
  result.generated_intersection = true;
  for (std::size_t component = 0; component < result.output_count;
       ++component) {
    result.output[component] =
        outside.output[component] +
        t * (inside.output[component] - outside.output[component]);
    if (!std::isfinite(result.output[component]))
      throw std::runtime_error("ClipCull intersection is non-finite");
  }
  return result;
}

std::vector<ClipVertex>
ClipTriangle(const std::array<ClipVertex, 3> &input, bool depth_clamp, std::uint8_t clip_dist_mask = 0, std::uint16_t clip_dist_reg = 0) {
  const std::uint32_t masks[3] = {
      ClipMask(input[0], depth_clamp, clip_dist_mask, clip_dist_reg),
      ClipMask(input[1], depth_clamp, clip_dist_mask, clip_dist_reg),
      ClipMask(input[2], depth_clamp, clip_dist_mask, clip_dist_reg)};
  const std::uint32_t union_mask = masks[0] | masks[1] | masks[2];
  if (union_mask == 0)
    return {input.begin(), input.end()};
  if ((masks[0] & masks[1] & masks[2]) != 0)
    return {};

  std::vector<ClipVertex> polygon(input.begin(), input.end());
  for (std::uint32_t plane = 0; plane < 14 && polygon.size() >= 3; ++plane) {
    if (depth_clamp && (plane == 4 || plane == 5))
      continue;
    if (plane >= 6 && (clip_dist_mask & (1U << (plane - 6))) == 0)
      continue;

    if ((union_mask & (1U << plane)) == 0)
      continue;
    if (polygon.size() > 15)
      throw std::runtime_error("ClipCull polygon exceeds six-plane bound");
    std::vector<ClipVertex> output;
    output.reserve(polygon.size() + 1);
    ClipVertex previous = polygon.front();
    float previous_distance = PlaneDistance(previous, plane, clip_dist_reg);
    for (std::size_t edge = 1; edge <= polygon.size(); ++edge) {
      const ClipVertex current = polygon[edge % polygon.size()];
      const float distance = PlaneDistance(current, plane, clip_dist_reg);
      bool different_sign = false;
      if (previous_distance >= 0.0F) {
        output.push_back(previous);
        different_sign = distance < 0.0F;
      } else {
        different_sign = !(distance < 0.0F);
      }

      if (different_sign) {
        const float denominator = distance - previous_distance;
        if (denominator == 0.0F || !std::isfinite(denominator))
          throw std::runtime_error("ClipCull invalid edge intersection");
        ClipVertex intersection;
        // The two algebraically equivalent branches mirror Mesa's public
        // clipper and choose the endpoint that minimizes interpolation error.
        if (distance < 0.0F) {
          if (-distance < previous_distance) {
            intersection = Interpolate(distance / denominator, current,
                                       previous);
          } else {
            intersection = Interpolate(-previous_distance / denominator,
                                       previous, current);
          }
        } else {
          if (-previous_distance < distance) {
            intersection = Interpolate(-previous_distance / denominator,
                                       previous, current);
          } else {
            intersection = Interpolate(distance / denominator, current,
                                       previous);
          }
        }
        output.push_back(intersection);
      }
      previous = current;
      previous_distance = distance;
    }
    if (output.size() > 15)
      throw std::runtime_error("ClipCull polygon output exceeds bound");
    polygon.swap(output);
  }
  return polygon;
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

std::int64_t QuantizedArea(const RasterTriangle &triangle) {
  const std::int64_t x0 =
      pvrgpu::stub::QuantizeRasterSubpixel(triangle.x[0]);
  const std::int64_t y0 =
      pvrgpu::stub::QuantizeRasterSubpixel(triangle.y[0]);
  const std::int64_t x1 =
      pvrgpu::stub::QuantizeRasterSubpixel(triangle.x[1]);
  const std::int64_t y1 =
      pvrgpu::stub::QuantizeRasterSubpixel(triangle.y[1]);
  const std::int64_t x2 =
      pvrgpu::stub::QuantizeRasterSubpixel(triangle.x[2]);
  const std::int64_t y2 =
      pvrgpu::stub::QuantizeRasterSubpixel(triangle.y[2]);
  const std::int64_t dx10 = CheckedSub(x1, x0, "ClipCull x delta overflow");
  const std::int64_t dy20 = CheckedSub(y2, y0, "ClipCull y delta overflow");
  const std::int64_t dy10 = CheckedSub(y1, y0, "ClipCull y delta overflow");
  const std::int64_t dx20 = CheckedSub(x2, x0, "ClipCull x delta overflow");
  const std::int64_t positive =
      CheckedMul(dx10, dy20, "ClipCull area product overflow");
  const std::int64_t negative =
      CheckedMul(dy10, dx20, "ClipCull area product overflow");
  return CheckedSub(positive, negative, "ClipCull area overflow");
}

float NdcSignedArea(const std::array<ClipVertex, 3> &vertices) {
  float ndc_x[3]{};
  float ndc_y[3]{};
  for (std::size_t index = 0; index < vertices.size(); ++index) {
    const float w = vertices[index].output[3];
    if (!(w > 0.0F))
      throw std::runtime_error(
          "ClipCull fixture requires positive homogeneous W");
    ndc_x[index] = vertices[index].output[0] / w;
    ndc_y[index] = vertices[index].output[1] / w;
  }
  const float area =
      (ndc_x[1] - ndc_x[0]) * (ndc_y[2] - ndc_y[0]) -
      (ndc_y[1] - ndc_y[0]) * (ndc_x[2] - ndc_x[0]);
  if (!std::isfinite(area))
    throw std::runtime_error("ClipCull received a non-finite triangle area");
  return area;
}

bool ClassifyFrontFacing(const std::array<ClipVertex, 3> &vertices,
                         FrontFaceWinding front_face,
                         bool reject_degenerate) {
  const float area = NdcSignedArea(vertices);
  if (area == 0.0F) {
    if (reject_degenerate)
      throw std::runtime_error(
          "ClipCull received a degenerate input triangle");
    // GLES defines a zero-area polygon as non-front-facing.
    return false;
  }
  switch (front_face) {
  case FrontFaceWinding::kClockwise:
    return area < 0.0F;
  case FrontFaceWinding::kCounterClockwise:
    return area > 0.0F;
  }
  throw std::runtime_error("ClipCull received an invalid front-face winding");
}

void ValidateFaceCullState(const FaceCullState &state) {
  if (state.enable > 1)
    throw std::runtime_error("ClipCull received an invalid cull enable bit");
  switch (state.mode) {
  case CullFaceMode::kFront:
  case CullFaceMode::kBack:
  case CullFaceMode::kFrontAndBack:
    break;
  default:
    throw std::runtime_error("ClipCull received an invalid cull-face mode");
  }
  switch (state.front_face) {
  case FrontFaceWinding::kClockwise:
  case FrontFaceWinding::kCounterClockwise:
    break;
  default:
    throw std::runtime_error("ClipCull received an invalid front-face winding");
  }
}

bool IsFaceCulled(const FaceCullState &state, bool front_facing) {
  if (!state.enable)
    return false;
  switch (state.mode) {
  case CullFaceMode::kFront:
    return front_facing;
  case CullFaceMode::kBack:
    return !front_facing;
  case CullFaceMode::kFrontAndBack:
    return true;
  }
  throw std::runtime_error("ClipCull received an invalid cull-face mode");
}

// glLineWidth and a fixed gl_PointSize travel with the draw and are widened
// into real screen-space geometry here.  A per-vertex point size comes from
// the shader and is still not lowered.

// Offsets a clip-space vertex by a screen-pixel delta, leaving every other
// VTXOUT component (z, w, varyings) unchanged. The delta is converted from
// pixels to a clip-space delta using that vertex's own w, so the offset
// still lands at the requested pixel distance after the perspective divide
// and viewport transform BuildRasterTriangle performs later.
ClipVertex OffsetClipVertexScreenPixels(const ClipVertex &vertex,
                                        float delta_screen_x,
                                        float delta_screen_y,
                                        std::uint32_t width,
                                        std::uint32_t height) {
  ClipVertex result = vertex;
  const float w = vertex.output[3];
  const float delta_ndc_x = delta_screen_x * 2.0F / static_cast<float>(width);
  const float delta_ndc_y = delta_screen_y * 2.0F / static_cast<float>(height);
  result.output[0] += delta_ndc_x * w;
  result.output[1] += delta_ndc_y * w;
  return result;
}

// Widens a line segment's two shaded endpoints into the four corners of a
// screen-space quad, offset perpendicular to the segment by half_width_px on
// each side. Both endpoints must have positive homogeneous W: the model does
// not clip-space-widen a line with an endpoint behind the eye point, and the
// caller falls back to the pre-existing degenerate-triangle path when this
// returns false. Also returns false when the segment is too short in screen
// space to have a well-defined perpendicular.
bool BuildLineQuadCorners(const ClipVertex &a, const ClipVertex &b,
                          float half_width_px, std::uint32_t width,
                          std::uint32_t height,
                          std::array<ClipVertex, 4> &corners) {
  const float wa = a.output[3];
  const float wb = b.output[3];
  if (!(wa > 0.0F) || !(wb > 0.0F))
    return false;
  const float screen_ax =
      (a.output[0] / wa * 0.5F + 0.5F) * static_cast<float>(width);
  const float screen_ay =
      (a.output[1] / wa * 0.5F + 0.5F) * static_cast<float>(height);
  const float screen_bx =
      (b.output[0] / wb * 0.5F + 0.5F) * static_cast<float>(width);
  const float screen_by =
      (b.output[1] / wb * 0.5F + 0.5F) * static_cast<float>(height);
  const float dx = screen_bx - screen_ax;
  const float dy = screen_by - screen_ay;
  const float length = std::sqrt(dx * dx + dy * dy);
  if (!(length > 1.0e-6F))
    return false;
  const float nx = -dy / length * half_width_px;
  const float ny = dx / length * half_width_px;
  corners[0] = OffsetClipVertexScreenPixels(a, nx, ny, width, height);
  corners[1] = OffsetClipVertexScreenPixels(a, -nx, -ny, width, height);
  corners[2] = OffsetClipVertexScreenPixels(b, -nx, -ny, width, height);
  corners[3] = OffsetClipVertexScreenPixels(b, nx, ny, width, height);
  return true;
}

// Widens one shaded point vertex into the four corners of an axis-aligned
// half_size_px x half_size_px screen-space square centered on it. Requires
// positive homogeneous W for the same reason as BuildLineQuadCorners.
bool BuildPointQuadCorners(const ClipVertex &p, float half_size_px,
                           std::uint32_t width, std::uint32_t height,
                           std::array<ClipVertex, 4> &corners) {
  if (!(p.output[3] > 0.0F))
    return false;
  corners[0] =
      OffsetClipVertexScreenPixels(p, -half_size_px, -half_size_px, width, height);
  corners[1] =
      OffsetClipVertexScreenPixels(p, half_size_px, -half_size_px, width, height);
  corners[2] =
      OffsetClipVertexScreenPixels(p, half_size_px, half_size_px, width, height);
  corners[3] =
      OffsetClipVertexScreenPixels(p, -half_size_px, half_size_px, width, height);
  return true;
}

RasterTriangle BuildRasterTriangle(const std::array<ClipVertex, 3> &vertices,
                                   std::uint32_t width, std::uint32_t height,
                                   bool front_facing,
                                   std::vector<std::uint32_t> &vertex_outputs,
                                   bool depth_clamp_enable,
                                   bool mesa_viewport_order,
                                   bool mesa_clipper_emit_order) {
  if (mesa_clipper_emit_order && !mesa_viewport_order) {
    throw std::runtime_error(
        "ClipCull Mesa clipper order requires Mesa viewport semantics");
  }
  RasterTriangle triangle;
  triangle.front_facing = front_facing ? 1U : 0U;
  const std::uint16_t output_count = vertices[0].output_count;
  if (output_count < 4 ||
      output_count > pvrgpu::stub::kPcoVertexOutputRegisterCount) {
    throw std::runtime_error("ClipCull cannot serialize invalid VTXOUT range");
  }
  std::array<std::size_t, 3> output_order = {0, 1, 2};
  for (std::size_t index = 0; index < vertices.size(); ++index) {
    if (vertices[index].output_count != output_count)
      throw std::runtime_error("ClipCull triangle VTXOUT strides disagree");
    const float clip_w = vertices[index].output[3];
    if (!(clip_w > 0.0F) || !std::isfinite(clip_w))
      throw std::runtime_error("ClipCull post-clip W is not positive");
    triangle.reciprocal_w[index] = 1.0F / clip_w;
    const float ndc_x =
        vertices[index].output[0] * triangle.reciprocal_w[index];
    const float ndc_y =
        vertices[index].output[1] * triangle.reciprocal_w[index];
    float ndc_z =
        vertices[index].output[2] * triangle.reciprocal_w[index];
    if (depth_clamp_enable) {
      ndc_z = std::max(std::min(ndc_z, 1.0F), -1.0F);
    }
    if (mesa_viewport_order) {
      const float scale_x = static_cast<float>(width) * 0.5F;
      const float scale_y = static_cast<float>(height) * 0.5F;
      if (vertices[index].generated_intersection) {
        // draw_pipe_clip.c's CPU intersection path is compiled as distinct
        // MUL/MUL/ADD operations.  A one-ULP window coordinate difference is
        // observable in setup coefficients and half-precision texture LOD.
        triangle.x[index] =
            StrictAdd(StrictMultiply(ndc_x, scale_x), scale_x);
        triangle.y[index] =
            StrictAdd(StrictMultiply(ndc_y, scale_y), scale_y);
        triangle.window_z[index] =
            StrictAdd(StrictMultiply(ndc_z, 0.5F), 0.5F);
      } else {
        // Gallivm's do_rhw_viewport emits data * reciprocal_w first, then
        // contracts the viewport multiply/add for original shader vertices.
        triangle.x[index] = std::fma(ndc_x, scale_x, scale_x);
        triangle.y[index] = std::fma(ndc_y, scale_y, scale_y);
        triangle.window_z[index] = std::fma(ndc_z, 0.5F, 0.5F);
      }
    } else {
      triangle.x[index] =
          (ndc_x * 0.5F + 0.5F) * static_cast<float>(width);
      triangle.y[index] =
          (ndc_y * 0.5F + 0.5F) * static_cast<float>(height);
      triangle.window_z[index] = ndc_z * 0.5F + 0.5F;
    }
    if (!std::isfinite(triangle.x[index]) ||
        !std::isfinite(triangle.y[index]) ||
        !std::isfinite(triangle.window_z[index]) ||
        !std::isfinite(triangle.reciprocal_w[index])) {
      throw std::runtime_error("ClipCull viewport transform is non-finite");
    }
  }

  std::int64_t area = QuantizedArea(triangle);
  if (area < 0) {
    std::swap(triangle.x[1], triangle.x[2]);
    std::swap(triangle.y[1], triangle.y[2]);
    std::swap(triangle.window_z[1], triangle.window_z[2]);
    std::swap(triangle.reciprocal_w[1], triangle.reciprocal_w[2]);
    std::swap(output_order[1], output_order[2]);
    area = QuantizedArea(triangle);
    if (area <= 0)
      throw std::runtime_error("ClipCull failed to normalize raster winding");
  }
  triangle.rasterizable = area > 0 ? 1U : 0U;

  // The GL driver command uses last-vertex provoking semantics.  Clean
  // primitives enter llvmpipe setup in application order; the generic Mesa
  // clip stage fan-emits them as {fan-1, fan, 0}.  llvmpipe then exchanges
  // its first two vertices while normalizing the validated CW command for
  // setup, yielding {1,0,2} and {2,1,0}, respectively.  Translate that input
  // order through this model's independent raster-winding normalization.
  const std::array<std::size_t, 3> setup_input_order =
      mesa_viewport_order
          ? (mesa_clipper_emit_order
                 ? std::array<std::size_t, 3>{2, 1, 0}
                 : std::array<std::size_t, 3>{1, 0, 2})
          : std::array<std::size_t, 3>{0, 1, 2};
  for (std::size_t setup_index = 0; setup_index < setup_input_order.size();
       ++setup_index) {
    const auto serialized = std::find(output_order.begin(), output_order.end(),
                                      setup_input_order[setup_index]);
    if (serialized == output_order.end())
      throw std::runtime_error("ClipCull lost Mesa setup vertex identity");
    triangle.setup_vertex_order[setup_index] = static_cast<std::uint8_t>(
        std::distance(output_order.begin(), serialized));
  }
  if (vertex_outputs.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("ClipCull raster VTXOUT offset overflow");
  triangle.first_vertex_output_dword =
      static_cast<std::uint32_t>(vertex_outputs.size());
  triangle.vertex_output_stride_dwords = output_count;
  const std::uint64_t serialized_end =
      static_cast<std::uint64_t>(vertex_outputs.size()) +
      static_cast<std::uint64_t>(3) * output_count;
  if (serialized_end > std::numeric_limits<std::uint32_t>::max())
    throw std::overflow_error("ClipCull raster VTXOUT payload is too large");
  for (const std::size_t vertex_index : output_order) {
    for (std::size_t component = 0; component < output_count; ++component)
      vertex_outputs.push_back(FloatBits(vertices[vertex_index].output[component]));
  }
  return triangle;
}

bool IsInsideHomogeneousClip(const VertexLane &vertex,
                             std::uint16_t output_count,
                             bool depth_clamp) {
  const ClipVertex clip = ReadClipVertex(vertex, output_count);
  const float clip_x = clip.output[0];
  const float clip_y = clip.output[1];
  const float clip_z = clip.output[2];
  const float clip_w = clip.output[3];
  if (clip_w <= 0.0F)
    return false;
  (void)clip;
  bool xy_inside = clip_x >= -clip_w && clip_x <= clip_w && clip_y >= -clip_w &&
                   clip_y <= clip_w;
  if (depth_clamp)
    return xy_inside;
  return xy_inside && clip_z >= -clip_w && clip_z <= clip_w;
}

} // namespace

namespace pvrgpu::stub {

ClipCull::ClipCull(sc_core::sc_module_name name, MemoryPool &pool)
    : sc_module(name), pool_(pool) {
  SC_THREAD(Run);
}

void ClipCull::Run() {
  while (true) {
    const PipelineTxn txn = input.read();
    PipelineState state = LoadPipelineState(pool_, txn.state);

    RequireStage(state.stage, PipelineStage::kVertexShaded, name());
    const bool depth_clamp = state.raster_state.depth_clamp_enable != 0;
    const std::uint8_t clip_dist_mask = state.clip_distance_mask;
    const std::uint16_t clip_dist_reg = state.clip_distance_register;
    if (!IsRasterFunctionalCase(state.functional_case))
      throw std::runtime_error("ClipCull received an unsupported case");
    if (!HasPoolHandle(state.vertex_lanes))
      throw std::runtime_error("ClipCull received no shaded vertices");
    const std::vector<VertexLane> lanes =
        LoadArray<VertexLane>(pool_, state.vertex_lanes);
    std::uint16_t active_vertex_output_dwords = 4;
    if (UsesShaderVaryings(state)) {
      if (!HasPoolHandle(state.shader_varying_bindings)) {
        throw std::runtime_error(
            "ClipCull varying case has no shader linkage payload");
      }
      const std::vector<ShaderVaryingBinding> bindings =
          LoadArray<ShaderVaryingBinding>(pool_,
                                          state.shader_varying_bindings);
      const std::uint32_t varying_count =
          VaryingVectorCount(state);
      if (varying_count == 0 || bindings.size() != varying_count) {
        throw std::runtime_error(
            "ClipCull varying linkage count is invalid");
      }
      for (std::size_t index = 0; index < bindings.size(); ++index) {
        if (!IsExactVaryingBinding(state, bindings[index], index)) {
          throw std::runtime_error(
              "ClipCull varying linkage is not exact");
        }
      }
      active_vertex_output_dwords = static_cast<std::uint16_t>(
          VaryingVertexOutputDwordCount(state));
      if (active_vertex_output_dwords > kPcoVertexOutputRegisterCount)
        throw std::runtime_error("ClipCull varying VTXOUT range is too large");
    } else if (HasPoolHandle(state.shader_varying_bindings)) {
      throw std::runtime_error(
          "ClipCull solid-color case has unexpected varying linkage");
    }
    std::vector<RasterTriangle> triangles;
    std::vector<std::uint32_t> raster_vertex_outputs;
    const bool driver_pco_triangles =
        IsDriverPcoTrianglesCase(state.functional_case);
    if (IsFillSolidFamily(state.functional_case) ||
        IsTextureFamily(state.functional_case)) {
      const bool driver_textured_triangles =
          state.functional_case == FunctionalCase::kDriverTexturedTriangles;
      const std::size_t expected_lane_count =
          driver_textured_triangles ? 6U : 4U;
      if (lanes.size() != expected_lane_count ||
          HasPoolHandle(state.vertex_lane_refs))
        throw std::runtime_error(
            "ClipCull direct raster lane count is invalid");
      for (const VertexLane &lane : lanes) {
        if (!IsInsideHomogeneousClip(lane, active_vertex_output_dwords, depth_clamp)) {
          throw std::runtime_error(
              "ClipCull direct raster vertex is outside homogeneous clip space");
        }
      }
      constexpr std::size_t kStripTriangleIndices[2][3] = {
          {0, 1, 2}, {2, 1, 3},
      };
      constexpr std::size_t kListTriangleIndices[2][3] = {
          {0, 1, 2}, {3, 4, 5},
      };
      ValidateFaceCullState(state.raster_state.face_cull);
      if (driver_textured_triangles) {
        if (state.raster_state.face_cull.enable != 1 ||
            state.raster_state.face_cull.mode != CullFaceMode::kBack ||
            state.raster_state.face_cull.front_face !=
                FrontFaceWinding::kClockwise) {
          throw std::runtime_error(
              "ClipCull driver textured-triangle cull state is invalid");
        }
      } else if (state.raster_state.face_cull.enable != 0) {
        throw std::runtime_error(
            "ClipCull fullscreen strip unexpectedly enables face culling");
      }
      triangles.reserve(2);
      for (std::size_t primitive = 0; primitive < 2; ++primitive) {
        std::array<ClipVertex, 3> vertices;
        for (std::size_t vertex = 0; vertex < 3; ++vertex) {
          const std::size_t lane =
              driver_textured_triangles
                  ? kListTriangleIndices[primitive][vertex]
                  : kStripTriangleIndices[primitive][vertex];
          vertices[vertex] = ReadClipVertex(
              lanes[lane], active_vertex_output_dwords);
        }
        // Gallium's clockwise front-face bit is defined after its window-Y
        // convention, while this model keeps NDC +Y upward through the
        // viewport transform. Account for that single axis reflection when
        // classifying the validated driver command; retain the public state
        // itself as BACK/CW for reporting and fixed-function validation.
        const FrontFaceWinding classification_winding =
            driver_textured_triangles
                ? FrontFaceWinding::kCounterClockwise
                : state.raster_state.face_cull.front_face;
        const bool front_facing =
            ClassifyFrontFacing(vertices, classification_winding, true);
        const bool face_culled =
            IsFaceCulled(state.raster_state.face_cull, front_facing);
        RasterTriangle triangle = BuildRasterTriangle(
            vertices, state.width, state.height, front_facing,
            raster_vertex_outputs, depth_clamp, driver_textured_triangles,
            false);
        if (!triangle.rasterizable)
          throw std::runtime_error(
              "ClipCull direct raster draw produced a degenerate triangle");
        triangle.key.submit_ordinal = primitive;
        triangle.key.api_primitive_id = primitive;
        triangle.face_culled = face_culled ? 1U : 0U;
        if (face_culled)
          triangle.rasterizable = 0;
        triangles.push_back(triangle);
      }
      state.counters.c_invocations = 2;
    } else {
      const bool indexed_triangle =
          IsIndexedTriangleRasterCase(state.functional_case);
      /*
       * A lowered draw may be indexed.  Vertex fetch has already resolved its
       * indices into lane references, so what distinguishes the two here is
       * only whether an index payload is expected to be present.
       */
      const bool driver_pco_indexed =
          driver_pco_triangles &&
          state.draw.index_format != IndexFormat::kNone;
      const bool direct_pco = driver_pco_triangles && !driver_pco_indexed;
      if ((!driver_pco_triangles && !indexed_triangle) ||
          state.draw.topology != PrimitiveTopology::kTriangleList ||
          !HasPoolHandle(state.vertex_lane_refs)) {
        throw std::runtime_error(
            "ClipCull triangle-list payload is incomplete");
      }
      std::vector<std::uint16_t> indices;
      if (direct_pco) {
        if (state.draw.vertex_count == 0 ||
            state.draw.vertex_count % 3 != 0 ||
            state.draw.first_index != 0 || state.draw.index_count != 0 ||
            state.draw.base_vertex != 0 ||
            state.draw.index_format != IndexFormat::kNone ||
            state.primitive_restart_enable != 0 ||
            HasPoolHandle(state.vertex_indices) ||
            state.index_buffer_gpu_address != 0 ||
            state.index_buffer_bytes != 0) {
          throw std::runtime_error(
              "ClipCull direct triangle-list state is invalid");
        }
      } else if (!HasPoolHandle(state.vertex_indices)) {
        throw std::runtime_error(
            "ClipCull indexed triangle-list has no index payload");
      } else if (!driver_pco_indexed) {
        // The pinned indexed cases are uint16; a lowered draw states its own
        // index width, and its occurrence count is checked against the lane
        // references rather than against a re-decoded index array.
        indices = LoadArray<std::uint16_t>(pool_, state.vertex_indices);
      }
      const std::vector<VertexLaneRef> lane_refs =
          LoadArray<VertexLaneRef>(pool_, state.vertex_lane_refs);
      const std::size_t occurrence_count =
          direct_pco ? state.draw.vertex_count : state.draw.index_count;
      if (lane_refs.size() != occurrence_count ||
          lane_refs.size() % 3 != 0 ||
          (direct_pco && lanes.size() != occurrence_count) ||
          state.draw.first_index != 0 ||
          (indexed_triangle && indices.size() != occurrence_count)) {
        throw std::runtime_error(
            "ClipCull triangle-list occurrence/lane-ref ranges disagree");
      }
      const std::size_t primitive_count = lane_refs.size() / 3;
      triangles.reserve(primitive_count);
      ValidateFaceCullState(state.raster_state.face_cull);
      if (indexed_triangle) {
        const bool expects_face_culling =
            RequiresBackCcwFaceCull(state.functional_case);
        if ((!expects_face_culling &&
             state.raster_state.face_cull.enable != 0) ||
            (expects_face_culling &&
             (state.raster_state.face_cull.enable == 0 ||
              state.raster_state.face_cull.mode != CullFaceMode::kBack ||
              state.raster_state.face_cull.front_face !=
                  FrontFaceWinding::kCounterClockwise))) {
          throw std::runtime_error(
              "ClipCull indexed-triangle case/cull state is inconsistent");
        }
      }
      // Gallium records the application's clockwise front-face after its
      // window-Y convention.  This model retains NDC +Y upward until viewport
      // conversion, which is one axis reflection.  That reflection changes
      // winding independently of whether the application culls front or back
      // faces, so classify every validated driver PCO CULL/CW draw as CCW.
      const bool driver_window_y_reflection =
          driver_pco_triangles && state.raster_state.face_cull.enable == 1 &&
          state.raster_state.face_cull.front_face ==
              FrontFaceWinding::kClockwise;
      const FrontFaceWinding classification_winding =
          driver_window_y_reflection ? FrontFaceWinding::kCounterClockwise
                                     : state.raster_state.face_cull.front_face;
      if (kReferenceUarch.index_segment_max_indices == 0 ||
          kReferenceUarch.index_segment_max_indices % 3 != 0) {
        throw std::runtime_error(
            "ClipCull triangle segment configuration is invalid");
      }

      std::uint64_t next_submit_ordinal = 0;
      std::size_t segment_begin = 0;
      while (segment_begin < lane_refs.size()) {
        const std::size_t segment_count = std::min<std::size_t>(
            kReferenceUarch.index_segment_max_indices,
            lane_refs.size() - segment_begin);
        if (segment_count == 0 || segment_count % 3 != 0)
          throw std::runtime_error(
              "ClipCull occurrence segment split a triangle");
        const std::size_t segment_end = segment_begin + segment_count;

        // Mesa's VS middle-end ORs every fetched lane's clipmask across one
        // complete-primitive segment. A clean segment bypasses the generic
        // clip pipeline and reaches fixed triangle setup; a dirty segment
        // runs homogeneous clipping and post-clip face culling first.
        bool generic_clip_path = false;
        for (std::size_t occurrence = segment_begin;
             occurrence < segment_end; ++occurrence) {
          const VertexLaneRef &ref = lane_refs[occurrence];
          if (ref.lane_index >= lanes.size())
            throw std::runtime_error(
                "ClipCull lane reference is outside shaded lanes");
          if (ClipMask(ReadClipVertex(lanes[ref.lane_index],
                                      active_vertex_output_dwords), depth_clamp, clip_dist_mask, clip_dist_reg) != 0)
            generic_clip_path = true;
        }

        for (std::size_t occurrence = segment_begin;
             occurrence < segment_end; occurrence += 3) {
          const std::size_t primitive = occurrence / 3;
          std::array<ClipVertex, 3> vertices;
          for (std::size_t vertex = 0; vertex < 3; ++vertex) {
            const std::size_t vertex_occurrence = occurrence + vertex;
            const VertexLaneRef &ref = lane_refs[vertex_occurrence];
            if (ref.lane_index >= lanes.size())
              throw std::runtime_error(
                  "ClipCull lane reference is outside shaded lanes");
            std::uint64_t resolved = 0;
            if (direct_pco) {
              resolved = static_cast<std::uint64_t>(state.draw.first_vertex) +
                         vertex_occurrence;
              if (ref.lane_index != vertex_occurrence ||
                  resolved > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(
                    "ClipCull direct lane reference is not sequential");
              }
            } else if (driver_pco_indexed) {
              /*
               * Vertex fetch resolved this occurrence through the index
               * buffer, so the lane reference already carries the vertex it
               * shaded; re-deriving it here would mean decoding the indices a
               * second time in a width this stage does not know.
               */
              resolved = ref.vertex_index;
            } else {
              const std::int64_t indexed_resolved =
                  static_cast<std::int64_t>(indices[vertex_occurrence]) +
                  state.draw.base_vertex;
              if (indexed_resolved < 0) {
                throw std::runtime_error(
                    "ClipCull indexed lane reference resolved negative");
              }
              resolved = static_cast<std::uint64_t>(indexed_resolved);
            }
            if (resolved != ref.vertex_index) {
              throw std::runtime_error(
                  "ClipCull lane reference lost vertex identity");
            }
            vertices[vertex] = ReadClipVertex(
                lanes[ref.lane_index], active_vertex_output_dwords);
          }
          const bool source_is_point =
              state.source_topology == PrimitiveTopology::kPoints;
          const bool source_is_line =
              state.source_topology == PrimitiveTopology::kLines ||
              state.source_topology == PrimitiveTopology::kLineStrip ||
              state.source_topology == PrimitiveTopology::kLineLoop;
          std::array<ClipVertex, 4> quad_corners{};
          const bool width_expanded =
              (source_is_point &&
               BuildPointQuadCorners(vertices[0],
                                     0.5F * state.raster_state.point_size,
                                     state.width, state.height,
                                     quad_corners)) ||
              (source_is_line &&
               BuildLineQuadCorners(vertices[0], vertices[1],
                                    0.5F * state.raster_state.line_width,
                                    state.width,
                                    state.height, quad_corners));

          // Emits one already-non-degenerate triangle through the same
          // homogeneous clip + fan-emission + fixed-setup path every
          // triangle in this model goes through, so a width-expanded
          // line/point quad is clipped, counted and serialized identically
          // to real geometry instead of taking a shortcut around it.
          const auto emit_triangle = [&](const std::array<ClipVertex, 3> &tri,
                                         bool allow_face_cull) {
            const bool primitive_clipped = std::any_of(
                tri.begin(), tri.end(),
                [depth_clamp, clip_dist_mask,
                 clip_dist_reg](const ClipVertex &vertex) {
                  return ClipMask(vertex, depth_clamp, clip_dist_mask,
                                  clip_dist_reg) != 0;
                });
            const std::vector<ClipVertex> polygon =
                ClipTriangle(tri, depth_clamp, clip_dist_mask, clip_dist_reg);
            for (std::size_t fan = 2; fan < polygon.size(); ++fan) {
              if (fan - 2 > std::numeric_limits<std::uint16_t>::max())
                throw std::overflow_error(
                    "ClipCull clip-piece index overflow");
              const std::array<ClipVertex, 3> fan_triangle = {
                  polygon[0], polygon[fan - 1], polygon[fan]};
              const bool front_facing = ClassifyFrontFacing(
                  fan_triangle, classification_winding, false);
              const bool face_culled =
                  allow_face_cull &&
                  IsFaceCulled(state.raster_state.face_cull, front_facing);
              const std::uint64_t submit_ordinal = next_submit_ordinal++;
              // Generic clip-path culling happens before setup. Clean
              // segments enter fixed setup first, so retain a
              // non-rasterizable slot for the setup/c_primitives counters
              // before rejecting the backface.
              if (generic_clip_path && face_culled)
                continue;
              RasterTriangle triangle = BuildRasterTriangle(
                  fan_triangle, state.width, state.height, front_facing,
                  raster_vertex_outputs, depth_clamp, driver_pco_triangles,
                  driver_pco_triangles && primitive_clipped);
              triangle.key.submit_ordinal = submit_ordinal;
              triangle.key.api_primitive_id =
                  static_cast<std::uint32_t>(primitive);
              triangle.key.clip_piece = static_cast<std::uint16_t>(fan - 2);
              triangle.face_culled = face_culled ? 1U : 0U;
              if (face_culled)
                triangle.rasterizable = 0;
              triangles.push_back(triangle);
            }
          };

          if (width_expanded) {
            // GLES culling applies only to polygons: lines and points are
            // never face-culled regardless of the current cull state.
            const std::array<ClipVertex, 3> quad_tri_a = {
                quad_corners[0], quad_corners[1], quad_corners[2]};
            const std::array<ClipVertex, 3> quad_tri_b = {
                quad_corners[0], quad_corners[2], quad_corners[3]};
            emit_triangle(quad_tri_a, /*allow_face_cull=*/false);
            emit_triangle(quad_tri_b, /*allow_face_cull=*/false);
          } else {
            /* Expanded Gallium strips legitimately contain zero-area
             * primitives.  They still enter the driver PCO clip stage and
             * contribute to its invocation accounting, but fixed setup marks
             * them non-rasterizable below.  Legacy fixture inputs remain
             * fail-closed on an accidental degenerate triangle.  A
             * line/point primitive whose endpoint(s) fail the positive-W
             * precondition BuildLineQuadCorners/BuildPointQuadCorners require
             * also lands here, still encoded as vertices[1]/[2] duplicating
             * the last endpoint the way ExpandTopology built it. */
            const bool preclip_ndc_defined = std::all_of(
                vertices.begin(), vertices.end(),
                [](const ClipVertex &vertex) {
                  return vertex.output[3] > 0.0F;
                });
            if (preclip_ndc_defined) {
              (void)ClassifyFrontFacing(
                  vertices, classification_winding, !driver_pco_triangles);
            } else if (!generic_clip_path) {
              throw std::runtime_error(
                  "ClipCull clean segment has non-positive homogeneous W");
            }
            emit_triangle(vertices, /*allow_face_cull=*/true);
          }
        }
        segment_begin = segment_end;
      }
      state.counters.c_invocations = primitive_count;
    }

    state.raster_triangles = StoreNewArray(pool_, triangles);
    state.raster_vertex_outputs =
        StoreNewArray(pool_, raster_vertex_outputs);
    state.counters.c_primitives = triangles.size();
    state.stage = PipelineStage::kClipCullComplete;

    const std::uint64_t cycles =
        kReferenceUarch.clip_base_cycles +
        CeilDivide(state.counters.c_invocations,
                   kReferenceUarch.clip_primitives_per_batch);
    state.counters.clip_cull_cycles = cycles;
    state.counters.tiler_cycles += cycles;
    WaitForCycles(cycles);
    StorePipelineState(pool_, txn.state, state);
    output.write(txn);
  }
}

} // namespace pvrgpu::stub
