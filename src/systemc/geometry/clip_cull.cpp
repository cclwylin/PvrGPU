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
};

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

float PlaneDistance(const ClipVertex &vertex, std::uint32_t plane) {
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

std::uint32_t ClipMask(const ClipVertex &vertex) {
  std::uint32_t mask = 0;
  for (std::uint32_t plane = 0; plane < 6; ++plane) {
    // Mesa's clip test is expressed as !(dp >= 0), so boundary vertices are
    // inside and unordered NaNs fail closed as outside.
    if (!(PlaneDistance(vertex, plane) >= 0.0F))
      mask |= 1U << plane;
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
ClipTriangle(const std::array<ClipVertex, 3> &input) {
  const std::uint32_t masks[3] = {
      ClipMask(input[0]), ClipMask(input[1]), ClipMask(input[2])};
  const std::uint32_t union_mask = masks[0] | masks[1] | masks[2];
  if (union_mask == 0)
    return {input.begin(), input.end()};
  if ((masks[0] & masks[1] & masks[2]) != 0)
    return {};

  std::vector<ClipVertex> polygon(input.begin(), input.end());
  for (std::uint32_t plane = 0; plane < 6 && polygon.size() >= 3; ++plane) {
    if ((union_mask & (1U << plane)) == 0)
      continue;
    if (polygon.size() > 15)
      throw std::runtime_error("ClipCull polygon exceeds six-plane bound");
    std::vector<ClipVertex> output;
    output.reserve(polygon.size() + 1);
    ClipVertex previous = polygon.front();
    float previous_distance = PlaneDistance(previous, plane);
    for (std::size_t edge = 1; edge <= polygon.size(); ++edge) {
      const ClipVertex current = polygon[edge % polygon.size()];
      const float distance = PlaneDistance(current, plane);
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

std::int64_t Quantize(float value) {
  const double scaled =
      static_cast<double>(value) * pvrgpu::stub::kSubpixelScale;
  if (!std::isfinite(scaled) ||
      scaled < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      scaled > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    throw std::overflow_error("ClipCull fixed-point coordinate overflow");
  }
  return static_cast<std::int64_t>(std::llround(scaled));
}

std::int64_t QuantizedArea(const RasterTriangle &triangle) {
  const std::int64_t x0 = Quantize(triangle.x[0]);
  const std::int64_t y0 = Quantize(triangle.y[0]);
  const std::int64_t x1 = Quantize(triangle.x[1]);
  const std::int64_t y1 = Quantize(triangle.y[1]);
  const std::int64_t x2 = Quantize(triangle.x[2]);
  const std::int64_t y2 = Quantize(triangle.y[2]);
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

RasterTriangle BuildRasterTriangle(const std::array<ClipVertex, 3> &vertices,
                                   std::uint32_t width, std::uint32_t height,
                                   bool front_facing,
                                   std::vector<std::uint32_t> &vertex_outputs) {
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
    const float ndc_z =
        vertices[index].output[2] * triangle.reciprocal_w[index];
    triangle.x[index] =
        (ndc_x * 0.5F + 0.5F) * static_cast<float>(width);
    triangle.y[index] =
        (ndc_y * 0.5F + 0.5F) * static_cast<float>(height);
    triangle.window_z[index] = ndc_z * 0.5F + 0.5F;
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
                             std::uint16_t output_count) {
  const ClipVertex clip = ReadClipVertex(vertex, output_count);
  const float clip_x = clip.output[0];
  const float clip_y = clip.output[1];
  const float clip_z = clip.output[2];
  const float clip_w = clip.output[3];
  if (clip_w <= 0.0F)
    return false;
  (void)clip;
  return clip_x >= -clip_w && clip_x <= clip_w && clip_y >= -clip_w &&
         clip_y <= clip_w && clip_z >= -clip_w && clip_z <= clip_w;
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
    if (!IsRasterFunctionalCase(state.functional_case))
      throw std::runtime_error("ClipCull received an unsupported case");
    if (!HasPoolHandle(state.vertex_lanes))
      throw std::runtime_error("ClipCull received no shaded vertices");
    const std::vector<VertexLane> lanes =
        LoadArray<VertexLane>(pool_, state.vertex_lanes);
    std::uint16_t active_vertex_output_dwords = 4;
    if (UsesShaderVaryings(state.functional_case)) {
      if (!HasPoolHandle(state.shader_varying_bindings)) {
        throw std::runtime_error(
            "ClipCull varying case has no shader linkage payload");
      }
      const std::vector<ShaderVaryingBinding> bindings =
          LoadArray<ShaderVaryingBinding>(pool_,
                                          state.shader_varying_bindings);
      const std::uint32_t varying_count =
          VaryingVectorCount(state.functional_case);
      if (varying_count == 0 || bindings.size() != varying_count) {
        throw std::runtime_error(
            "ClipCull varying linkage count is invalid");
      }
      for (std::size_t index = 0; index < bindings.size(); ++index) {
        if (!IsExactVaryingBinding(state.functional_case, bindings[index],
                                   index)) {
          throw std::runtime_error(
              "ClipCull varying linkage is not exact");
        }
      }
      active_vertex_output_dwords = static_cast<std::uint16_t>(
          VaryingVertexOutputDwordCount(state.functional_case));
      if (active_vertex_output_dwords > kPcoVertexOutputRegisterCount)
        throw std::runtime_error("ClipCull varying VTXOUT range is too large");
    } else if (HasPoolHandle(state.shader_varying_bindings)) {
      throw std::runtime_error(
          "ClipCull solid-color case has unexpected varying linkage");
    }
    std::vector<RasterTriangle> triangles;
    std::vector<std::uint32_t> raster_vertex_outputs;
    if (IsFillSolidFamily(state.functional_case) ||
        IsTextureFamily(state.functional_case)) {
      if (lanes.size() != 4 || HasPoolHandle(state.vertex_lane_refs))
        throw std::runtime_error(
          "ClipCull fullscreen strip requires four direct shaded vertices");
      for (const VertexLane &lane : lanes) {
        if (!IsInsideHomogeneousClip(lane, active_vertex_output_dwords)) {
          throw std::runtime_error(
              "ClipCull Fill.Solid vertex is outside homogeneous clip space");
        }
      }
      constexpr std::size_t kTriangleIndices[2][3] = {
          {0, 1, 2},
          {2, 1, 3},
      };
      triangles.reserve(2);
      for (std::size_t primitive = 0; primitive < 2; ++primitive) {
        std::array<ClipVertex, 3> vertices;
        for (std::size_t vertex = 0; vertex < 3; ++vertex)
          vertices[vertex] = ReadClipVertex(
              lanes[kTriangleIndices[primitive][vertex]],
              active_vertex_output_dwords);
        RasterTriangle triangle = BuildRasterTriangle(
            vertices, state.width, state.height, true,
            raster_vertex_outputs);
        if (!triangle.rasterizable)
          throw std::runtime_error(
              "ClipCull Fill.Solid produced a degenerate triangle");
        triangle.key.submit_ordinal = primitive;
        triangle.key.api_primitive_id = primitive;
        triangles.push_back(triangle);
      }
      state.counters.c_invocations = 2;
    } else {
      if (!IsIndexedTriangleRasterCase(state.functional_case) ||
          state.draw.topology != PrimitiveTopology::kTriangleList ||
          !HasPoolHandle(state.vertex_indices) ||
          !HasPoolHandle(state.vertex_lane_refs)) {
        throw std::runtime_error(
            "ClipCull indexed-triangle payload is incomplete");
      }
      const std::vector<std::uint16_t> indices =
          LoadArray<std::uint16_t>(pool_, state.vertex_indices);
      const std::vector<VertexLaneRef> lane_refs =
          LoadArray<VertexLaneRef>(pool_, state.vertex_lane_refs);
      if (state.draw.first_index != 0 ||
          lane_refs.size() != state.draw.index_count ||
          indices.size() != state.draw.index_count ||
          lane_refs.size() % 3 != 0) {
        throw std::runtime_error(
            "ClipCull indexed-triangle index/lane-ref ranges disagree");
      }
      const std::size_t primitive_count = lane_refs.size() / 3;
      triangles.reserve(primitive_count);
      ValidateFaceCullState(state.raster_state.face_cull);
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
      if (kReferenceUarch.index_segment_max_indices == 0 ||
          kReferenceUarch.index_segment_max_indices % 3 != 0) {
        throw std::runtime_error(
            "ClipCull indexed segment configuration is invalid");
      }

      std::uint64_t next_submit_ordinal = 0;
      std::size_t segment_begin = 0;
      while (segment_begin < lane_refs.size()) {
        const std::size_t segment_count = std::min<std::size_t>(
            kReferenceUarch.index_segment_max_indices,
            lane_refs.size() - segment_begin);
        if (segment_count == 0 || segment_count % 3 != 0)
          throw std::runtime_error("ClipCull index segment split a triangle");
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
                                      active_vertex_output_dwords)) != 0)
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
            const std::int64_t resolved =
                static_cast<std::int64_t>(indices[vertex_occurrence]) +
                state.draw.base_vertex;
            if (resolved < 0 ||
                static_cast<std::uint64_t>(resolved) != ref.vertex_index) {
              throw std::runtime_error(
                  "ClipCull lane reference lost index identity");
            }
            vertices[vertex] = ReadClipVertex(
                lanes[ref.lane_index], active_vertex_output_dwords);
          }
          (void)ClassifyFrontFacing(
              vertices, state.raster_state.face_cull.front_face, true);
          const std::vector<ClipVertex> polygon = ClipTriangle(vertices);
          for (std::size_t fan = 2; fan < polygon.size(); ++fan) {
            if (fan - 2 > std::numeric_limits<std::uint16_t>::max())
              throw std::overflow_error("ClipCull clip-piece index overflow");
            const std::array<ClipVertex, 3> fan_triangle = {
                polygon[0], polygon[fan - 1], polygon[fan]};
            const bool front_facing = ClassifyFrontFacing(
                fan_triangle, state.raster_state.face_cull.front_face, false);
            const bool face_culled =
                IsFaceCulled(state.raster_state.face_cull, front_facing);
            const std::uint64_t submit_ordinal = next_submit_ordinal++;
            // Generic clip-path culling happens before setup. Clean segments
            // enter fixed setup first, so retain a non-rasterizable slot for
            // the setup/c_primitives counters before rejecting the backface.
            if (generic_clip_path && face_culled)
              continue;
            RasterTriangle triangle = BuildRasterTriangle(
                fan_triangle, state.width, state.height, front_facing,
                raster_vertex_outputs);
            triangle.key.submit_ordinal = submit_ordinal;
            triangle.key.api_primitive_id =
                static_cast<std::uint32_t>(primitive);
            triangle.key.clip_piece = static_cast<std::uint16_t>(fan - 2);
            triangle.face_culled = face_culled ? 1U : 0U;
            if (face_culled)
              triangle.rasterizable = 0;
            triangles.push_back(triangle);
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
