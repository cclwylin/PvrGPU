// ISP (Image Synthesis Processor) functional module.
// It consumes each 32x32 tile's ordered Parameter Buffer primitive references,
// evaluates exact fixed-point top-left coverage at every declared sample
// position, and performs opaque-safe HSR (Hidden Surface Removal). Opaque
// draws retain only the final owner; blending preserves every depth-passing
// candidate in API order for later PBE destination read/modify/write. FIFO
// traffic remains MemoryPool handles and completion is event-driven.
#include "fragment/isp.h"

#include "common/functional_types.h"
#include "common/msaa.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using pvrgpu::stub::DepthCompareOp;
using pvrgpu::stub::DepthState;
using pvrgpu::stub::EdgeEquation;
using pvrgpu::stub::FragmentCandidate;
using pvrgpu::stub::FragmentVisibility;
using pvrgpu::stub::LineSegment;
using pvrgpu::stub::ParameterTriangle;

bool CoversSample(const ParameterTriangle &triangle, std::int64_t sample_x,
                  std::int64_t sample_y, std::int64_t values[3]) {
  for (std::size_t edge = 0; edge < 3; ++edge) {
    const EdgeEquation &equation = triangle.edge[edge];
    values[edge] = equation.a * sample_x + equation.b * sample_y + equation.c;
    if (values[edge] < 0 || (values[edge] == 0 && equation.inclusive == 0)) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool DepthPass(DepthCompareOp compare_op, T incoming, T stored) {
  switch (compare_op) {
  case DepthCompareOp::kNever:
    return false;
  case DepthCompareOp::kLess:
    return incoming < stored;
  case DepthCompareOp::kEqual:
    return incoming == stored;
  case DepthCompareOp::kLessOrEqual:
    return incoming <= stored;
  case DepthCompareOp::kGreater:
    return incoming > stored;
  case DepthCompareOp::kNotEqual:
    return incoming != stored;
  case DepthCompareOp::kGreaterOrEqual:
    return incoming >= stored;
  case DepthCompareOp::kAlways:
    return true;
  }
  throw std::runtime_error("ISP received an invalid depth compare operation");
}

/*
 * Whether a width-1 line puts a fragment in this pixel.
 *
 * One fragment per step of the major axis, at the row (or column) the segment
 * passes through at that step's centre, which is what the diamond-exit rule
 * reduces to for a line one pixel wide.
 */
bool LineCoversPixel(const LineSegment &line, std::uint32_t x,
                     std::uint32_t y) {
  const float dx = line.x1 - line.x0;
  const float dy = line.y1 - line.y0;
  if (!std::isfinite(dx) || !std::isfinite(dy))
    return false;
  if (std::fabs(dx) >= std::fabs(dy)) {
    if (dx == 0.0F)
      return true;  // A point-length segment; the quad already bounds it.
    const float centre = static_cast<float>(x) + 0.5F;
    const float at = line.y0 + (centre - line.x0) * (dy / dx);
    return static_cast<std::int64_t>(std::floor(at)) ==
           static_cast<std::int64_t>(y);
  }
  const float centre = static_cast<float>(y) + 0.5F;
  const float at = line.x0 + (centre - line.y0) * (dx / dy);
  return static_cast<std::int64_t>(std::floor(at)) ==
         static_cast<std::int64_t>(x);
}

float BitsFloat(std::uint32_t bits) {
  float value = 0.0F;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

float InterpolateDepth(const ParameterTriangle &triangle,
                       const std::int64_t edge_values[3], float x,
                       float y, bool llvmpipe_driver_plane,
                       float barycentric[3]) {
  if (triangle.signed_area <= 0)
    throw std::runtime_error("ISP received a non-positive triangle area");
  const double reciprocal_area =
      1.0 / static_cast<double>(triangle.signed_area);
  // Edge 1 is opposite vertex 0, edge 2 is opposite vertex 1, and edge 0 is
  // opposite vertex 2 for the serialized v0->v1->v2 equations.
  barycentric[0] = static_cast<float>(edge_values[1] * reciprocal_area);
  barycentric[1] = static_cast<float>(edge_values[2] * reciprocal_area);
  barycentric[2] = static_cast<float>(edge_values[0] * reciprocal_area);
  if (llvmpipe_driver_plane) {
    if (triangle.depth_plane_valid != 1) {
      throw std::runtime_error("ISP driver depth plane metadata is invalid");
    }
    const float a = BitsFloat(triangle.depth_plane[0]);
    const float b = BitsFloat(triangle.depth_plane[1]);
    const float c = BitsFloat(triangle.depth_plane[2]);
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c))
      throw std::runtime_error("ISP driver depth plane is non-finite");
    // llvmpipe's FS evaluates its internal position-Z slot with these two
    // ordered llvm.fmuladd operations.  The order is observable after Z32
    // quantization and later binary16 texture sampling.
    return std::fma(b, static_cast<float>(y),
                    std::fma(a, static_cast<float>(x), c));
  }
  return barycentric[0] * triangle.window_z[0] +
         barycentric[1] * triangle.window_z[1] +
         barycentric[2] * triangle.window_z[2];
}

} // namespace

namespace pvrgpu::stub {

Isp::Isp(sc_core::sc_module_name name, MemoryPool &pool,
         GpuMemorySystem *memory)
    : sc_module(name), pool_(pool), memory_(memory) {
  SC_THREAD(Run);
}

void Isp::Run() {
  while (true) {
    const PipelineTxn txn = input.read();
    PipelineState state = LoadPipelineState(pool_, txn.state);
    RequireStage(state.stage, PipelineStage::kTilesScheduled, name());
    if (memory_ && state.memory_mode != memory_->mode())
      throw std::runtime_error("ISP memory mode mismatch");
    if (!IsRasterFunctionalCase(state.functional_case))
      throw std::runtime_error("ISP received an unsupported case");
    bool opaque_early_hsr = state.raster_state.blend.enable == 0;
    // If the fragment shader may discard, we cannot perform opaque early HSR because
    // a front-most fragment might be discarded later, revealing fragments behind it.
    // Likewise, if early HSR is not safe, we disable early culling.
    const bool late_depth_stencil =
        RasterRequiresLateDepthStencil(state.raster_state);
    if (state.raster_state.shader_may_discard || late_depth_stencil ||
        !state.fragment_early_hsr_safe) {
      opaque_early_hsr = false;
    }
    // If the shader writes custom depth, early depth writes are not allowed because
    // the final depth value is determined during shader execution.
    const bool early_depth_write = state.raster_state.depth.write_enable &&
                                   !late_depth_stencil;
    const std::uint32_t sample_count = state.raster_state.sample_count;
    if (state.raster_state.multisample_enable > 1 ||
        state.raster_state.alpha_to_coverage > 1 ||
        state.raster_state.alpha_to_coverage_dither > 1 ||
        state.raster_state.alpha_to_one > 1)
      throw std::runtime_error("ISP multisample/alpha control flag is invalid");
    const bool multisample_rasterization =
        sample_count > 1 && state.raster_state.multisample_enable != 0;
    const std::uint32_t enabled_samples =
        RasterSampleMask(sample_count) & state.raster_state.sample_mask;

    const std::uint64_t pixel_count =
        static_cast<std::uint64_t>(state.width) * state.height * state.attachment_layers;
    if (pixel_count > std::numeric_limits<std::size_t>::max() / sample_count)
      throw std::overflow_error("ISP surface is too large");
    const std::size_t storage_count =
        static_cast<std::size_t>(pixel_count) * sample_count;
    const std::vector<TileRecord> tiles =
        LoadArray<TileRecord>(pool_, state.tile_records);
    const std::vector<TilePrimitiveRef> primitive_refs =
        LoadArray<TilePrimitiveRef>(pool_, state.tile_primitive_refs);
    MemoryAccessStats memory_stats;
    std::vector<ParameterTriangle> parameters;
    if (memory_) {
      auto read = ReadMemoryArray<ParameterTriangle>(
          *memory_, state.parameter_triangles_gpu_address,
          state.parameter_triangles_bytes, MemoryClient::kParameterRead);
      parameters = std::move(read.values);
      memory_stats += read.stats;
    } else {
      parameters =
          LoadArray<ParameterTriangle>(pool_, state.parameter_triangles);
    }
    // A draw can reach the ISP with no primitives: every triangle was culled,
    // or -- in a sequence -- every triangle was clipped away by the view
    // volume (dEQP's fragment_ops.depth_stencil depth-visualize quads at
    // z = -1.05).  Either way the setup is legitimately empty; the ISP shades
    // no fragment but still loads the attachment, applies the scissored
    // depth/stencil clears the draw inherited, and writes the planes back, so
    // the frame carries forward.  Only a parameter buffer that is empty
    // without the primitive count agreeing is malformed.
    const bool empty_setup =
        parameters.empty() && primitive_refs.empty() &&
        state.counters.c_primitives == 0;
    if (tiles.size() != state.scheduled_tiles ||
        (parameters.empty() && !empty_setup))
      throw std::runtime_error("ISP received invalid tile/parameter data");

    std::vector<FragmentCandidate> candidates;
    candidates.reserve(static_cast<std::size_t>(pixel_count));
    constexpr std::size_t kNoOwner = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> owner(storage_count, kNoOwner);
    std::vector<std::uint8_t> covered(static_cast<std::size_t>(pixel_count), 0);
    if (state.depth_attachment_load_enable > 1 ||
        (state.depth_attachment_load_enable != 0) !=
            HasPoolHandle(state.depth_attachment_load) ||
        (state.depth_attachment_load_enable == 0 &&
         state.depth_attachment_load_bytes != 0) ||
        (state.depth_attachment_load_enable != 0 &&
         state.depth_attachment_format == 0)) {
      throw std::runtime_error("ISP depth attachment LOAD state is invalid");
    }
    const std::size_t pixel_count_size = storage_count;
    std::vector<std::uint32_t> encoded_depth(pixel_count_size, 0);
    std::vector<float> depth(pixel_count_size, 0.0F);
    // The stencil plane of a combined attachment.  Formats without one keep an
    // all-zero plane that nothing reads and nothing writes back.
    const bool has_stencil =
        DepthAttachmentHasStencil(state.depth_attachment_format);
    std::vector<std::uint8_t> stencil(pixel_count_size, 0);
    if (state.depth_attachment_load_enable != 0) {
      const std::size_t bytes_per_pixel =
          DepthAttachmentBytesPerPixel(state.depth_attachment_format);
      const std::uint64_t expected_bytes = storage_count * bytes_per_pixel;
      const std::vector<std::uint8_t> encoded =
          LoadArray<std::uint8_t>(pool_, state.depth_attachment_load);
      if (state.depth_attachment_load_bytes != expected_bytes ||
          encoded.size() != expected_bytes) {
        throw std::runtime_error(
            "ISP depth attachment LOAD byte count mismatch");
      }
      encoded_depth = DecodeDepthAttachmentUnormBytes(
          encoded, state.depth_attachment_format,
          has_stencil ? &stencil : nullptr);
      for (std::size_t pixel = 0; pixel < pixel_count_size; ++pixel) {
        depth[pixel] = DecodeDepthAttachmentUnorm(
            encoded_depth[pixel], state.depth_attachment_format);
      }
    } else if (state.depth_attachment_format != 0) {
      const std::uint32_t encoded_clear = EncodeDepthAttachmentUnorm(
          state.raster_state.depth.clear_depth,
          state.depth_attachment_format);
      const float quantized_clear = DecodeDepthAttachmentUnorm(
          encoded_clear, state.depth_attachment_format);
      std::fill(encoded_depth.begin(), encoded_depth.end(), encoded_clear);
      std::fill(depth.begin(), depth.end(), quantized_clear);
      if (has_stencil) {
        std::fill(stencil.begin(), stencil.end(),
                  static_cast<std::uint8_t>(
                      state.raster_state.stencil.clear_stencil & 0xFFU));
      }
    } else {
      std::fill(depth.begin(), depth.end(),
                state.raster_state.depth.clear_depth);
    }
    /*
     * Scissored depth/stencil clears the draw inherits.  They happened before
     * it, so they land on the planes before any fragment is tested; folding
     * them into the whole-surface clear values is not possible, which is why
     * they travel as rectangles.
     */
    if (HasPoolHandle(state.attachment_clears)) {
      const std::vector<AttachmentClearRect> clears =
          LoadArray<AttachmentClearRect>(pool_, state.attachment_clears);
      for (const AttachmentClearRect &clear : clears) {
        if (clear.width == 0 || clear.height == 0)
          continue;
        const std::uint64_t x_end =
            static_cast<std::uint64_t>(clear.x) + clear.width;
        const std::uint64_t y_end =
            static_cast<std::uint64_t>(clear.y) + clear.height;
        if (x_end > state.width || y_end > state.height) {
          throw std::runtime_error(
              "ISP inherited attachment clear leaves the attachment: x=" +
              std::to_string(clear.x) + " y=" + std::to_string(clear.y) +
              " width=" + std::to_string(clear.width) +
              " height=" + std::to_string(clear.height) +
              " extent=" + std::to_string(state.width) + "x" +
              std::to_string(state.height));
        }
        const bool clears_depth =
            (clear.aspects & kClearAspectDepth) != 0 &&
            state.depth_attachment_format != 0;
        const bool clears_stencil =
            (clear.aspects & kClearAspectStencil) != 0 && has_stencil;
        const std::uint32_t encoded_clear =
            clears_depth ? EncodeDepthAttachmentUnorm(
                               BitsFloat(clear.depth_bits),
                               state.depth_attachment_format)
                         : 0;
        const float decoded_clear =
            clears_depth ? DecodeDepthAttachmentUnorm(
                               encoded_clear, state.depth_attachment_format)
                         : 0.0F;
        const std::uint8_t stencil_value =
            static_cast<std::uint8_t>(clear.stencil_value & 0xFFU);
        for (std::uint32_t layer = 0; layer < state.attachment_layers; ++layer)
        for (std::uint32_t y = clear.y; y < y_end; ++y) {
          const std::size_t row =
              (static_cast<std::size_t>(layer) * state.height + y) * state.width * sample_count;
          if (clears_stencil) {
            std::fill_n(stencil.begin() + row + clear.x * sample_count,
                        clear.width * sample_count,
                        stencil_value);
          }
          if (clears_depth) {
            std::fill_n(encoded_depth.begin() + row + clear.x * sample_count,
                        clear.width * sample_count,
                        encoded_clear);
            std::fill_n(depth.begin() + row + clear.x * sample_count,
                        clear.width * sample_count,
                        decoded_clear);
          }
        }
      }
    }

    std::uint64_t covered_pixels = 0;
    std::uint64_t depth_tested = 0;
    std::uint64_t depth_rejected = 0;
    std::uint64_t depth_written = 0;
    std::uint64_t stencil_tested = 0;
    std::uint64_t stencil_rejected = 0;
    std::uint64_t stencil_written = 0;
    const bool llvmpipe_driver_depth =
        IsDriverPcoTrianglesCase(state.functional_case);

    for (const TileRecord &tile : tiles) {
      const std::uint64_t ref_end =
          static_cast<std::uint64_t>(tile.first_primitive_ref) +
          tile.primitive_ref_count;
      if (ref_end > primitive_refs.size())
        throw std::runtime_error("ISP tile primitive range is out of bounds");
      std::uint64_t previous_ordinal = 0;
      bool first_ref = true;
      for (std::uint32_t ref_offset = 0; ref_offset < tile.primitive_ref_count;
           ++ref_offset) {
        const TilePrimitiveRef &ref =
            primitive_refs[tile.first_primitive_ref + ref_offset];
        if (ref.parameter_index >= parameters.size())
          throw std::runtime_error("ISP parameter reference is out of bounds");
        if (!first_ref && ref.submit_ordinal < previous_ordinal)
          throw std::runtime_error("ISP primitive references lost API order");
        first_ref = false;
        previous_ordinal = ref.submit_ordinal;
        const ParameterTriangle &triangle = parameters[ref.parameter_index];
        if (triangle.key.submit_ordinal != ref.submit_ordinal) {
          throw std::runtime_error(
              "ISP primitive identity mismatch: parameter_index=" +
              std::to_string(ref.parameter_index) + " of " +
              std::to_string(parameters.size()) + " ref_ordinal=" +
              std::to_string(ref.submit_ordinal) + " triangle_ordinal=" +
              std::to_string(triangle.key.submit_ordinal) +
              " api_primitive_id=" +
              std::to_string(triangle.key.api_primitive_id));
        }
        if (!HasCanonicalDepthPlaneMetadata(state.functional_case,
                                            triangle)) {
          throw std::runtime_error("ISP depth plane metadata is invalid");
        }

        // The scissor test rejects a fragment before any other per-fragment
        // work, so it narrows the span the ISP walks rather than being
        // evaluated per sample.
        const ScissorState &scissor = state.raster_state.scissor;
        const std::uint32_t scissor_x0 = scissor.enable ? scissor.x0 : 0U;
        const std::uint32_t scissor_y0 = scissor.enable ? scissor.y0 : 0U;
        const std::uint32_t scissor_x1 =
            scissor.enable ? scissor.x1 : state.width;
        const std::uint32_t scissor_y1 =
            scissor.enable ? scissor.y1 : state.height;

        const std::uint32_t y_begin = std::max({
            tile.y0, scissor_y0,
            static_cast<std::uint32_t>(std::max(0, triangle.min_y))});
        const std::uint32_t y_end = std::min({
            tile.y1, scissor_y1,
            static_cast<std::uint32_t>(std::max(0, triangle.max_y))});
        const std::uint32_t x_begin = std::max({
            tile.x0, scissor_x0,
            static_cast<std::uint32_t>(std::max(0, triangle.min_x))});
        const std::uint32_t x_end = std::min({
            tile.x1, scissor_x1,
            static_cast<std::uint32_t>(std::max(0, triangle.max_x))});
        for (std::uint32_t y = y_begin; y < y_end; ++y) {
          for (std::uint32_t x = x_begin; x < x_end; ++x) {
            const std::int64_t center_x =
                static_cast<std::int64_t>(x) * kSubpixelScale +
                kSubpixelScale / 2;
            const std::int64_t center_y =
                static_cast<std::int64_t>(y) * kSubpixelScale +
                kSubpixelScale / 2;
            std::int64_t edge_values[3]{};
            std::uint32_t coverage_mask = 0;
            std::array<float, 16> sample_depth{};
            for (std::uint32_t sample = 0; sample < sample_count; ++sample) {
              const std::uint32_t sample_bit = 1U << sample;
              if ((enabled_samples & sample_bit) == 0)
                continue;
              // With multisample rasterization disabled, one center test
              // supplies coverage to all selected samples. Depth/stencil
              // still reads and writes each sample's independent storage.
              const auto position = multisample_rasterization
                  ? RasterSamplePosition(sample_count, sample)
                  : RasterSamplePosition(1, 0);
              const std::int64_t sample_x =
                  static_cast<std::int64_t>(x) * kSubpixelScale +
                  position[0] * (kSubpixelScale / 16);
              const std::int64_t sample_y =
                  static_cast<std::int64_t>(y) * kSubpixelScale +
                  position[1] * (kSubpixelScale / 16);
              if (!CoversSample(triangle, sample_x, sample_y, edge_values))
                continue;
              float barycentric[3];
              sample_depth[sample] = InterpolateDepth(
                  triangle, edge_values,
                  static_cast<float>(x) + position[0] / 16.0F - 0.5F,
                  static_cast<float>(y) + position[1] / 16.0F - 0.5F,
                  llvmpipe_driver_depth, barycentric);
              coverage_mask |= sample_bit;
            }
            if (coverage_mask == 0)
              continue;
            /*
             * A width-1 line: the quad decided the region, the segment decides
             * the pixel.  GLES rasterises such a line one fragment per step of
             * its major axis, and the widened rectangle covers two wherever it
             * straddles a row -- 323 fragments against the diamond-exit rule's
             * 301, and fifty-three spans two pixels wide.  The quad's coverage
             * is a superset of the right answer, so intersecting the two is
             * exact; the fill rule already gives a shared edge to one of the
             * quad's two triangles, so no pixel is produced twice.
             */
            if (!multisample_rasterization && triangle.line.valid != 0 &&
                !LineCoversPixel(triangle.line, x, y))
              continue;

            // Pixel-frequency shading still interpolates at the pixel center;
            // sample positions govern coverage and depth/stencil individually.
            // Evaluate every center edge even when the center is uncovered.
            for (std::size_t edge = 0; edge < 3; ++edge) {
              const EdgeEquation &equation = triangle.edge[edge];
              edge_values[edge] = equation.a * center_x +
                                  equation.b * center_y + equation.c;
            }

            FragmentCandidate candidate;
            candidate.x = x;
            candidate.y = y;
            candidate.primitive_id = triangle.key.api_primitive_id;
            candidate.parameter_index = ref.parameter_index;
            candidate.submit_ordinal = ref.submit_ordinal;
            candidate.sample_mask = 0;
            candidate.depth = InterpolateDepth(
                triangle, edge_values, x, y, llvmpipe_driver_depth,
                candidate.barycentric);
            std::copy(sample_depth.begin(), sample_depth.end(),
                      candidate.sample_depth);
            const std::size_t coverage_index =
                (static_cast<std::size_t>(triangle.key.layer) * state.height + y) * state.width + x;
            if (triangle.key.layer >= state.attachment_layers)
              throw std::runtime_error("ISP primitive layer is outside its attachment");
            if (covered[coverage_index] == 0) {
              covered[coverage_index] = 1;
              ++covered_pixels;
            }
            const std::size_t candidate_index = candidates.size();
            candidates.push_back(candidate);
            for (std::uint32_t sample = 0; sample < sample_count; ++sample) {
              const std::uint32_t sample_bit = 1U << sample;
              if ((coverage_mask & sample_bit) == 0)
                continue;
              const std::size_t pixel_index =
                  coverage_index * sample_count + sample;
              if (late_depth_stencil) {
                // Shader depth or final alpha coverage is not known until
                // USC/PBE. Preserve geometry coverage, raster sample depths
                // and API order without changing either attachment.
                candidates[candidate_index].sample_mask |= sample_bit;
                candidates[candidate_index].visibility = FragmentVisibility::kVisible;
                continue;
              }
            /*
             * GLES 3.0 4.1.4: the stencil test runs before the depth test, and
             * the operation applied depends on which of the two failed.  A
             * fragment that fails the stencil test is discarded, but its
             * stencil-fail operation still updates the plane.
             */
            const StencilState &stencil_state = state.raster_state.stencil;
            const StencilFaceState &face =
                triangle.front_facing != 0 ? stencil_state.front
                                           : stencil_state.back;
            bool stencil_passes = true;
            if (stencil_state.test_enable && has_stencil) {
              ++stencil_tested;
              const std::uint8_t value_mask =
                  static_cast<std::uint8_t>(face.value_mask & 0xFFU);
              const std::uint8_t reference =
                  static_cast<std::uint8_t>(face.reference & 0xFFU);
              stencil_passes = StencilPass(
                  face.compare_op,
                  static_cast<std::uint8_t>(reference & value_mask),
                  static_cast<std::uint8_t>(stencil[pixel_index] & value_mask));
              if (!stencil_passes)
                ++stencil_rejected;
            }

            bool passes = stencil_passes;
            std::uint32_t incoming_encoded_depth = 0;
            if (stencil_passes && state.raster_state.depth.test_enable) {
              ++depth_tested;
              if (state.depth_attachment_format == 0) {
                passes = DepthPass(state.raster_state.depth.compare_op,
                                   sample_depth[sample], depth[pixel_index]);
              } else {
                // llvmpipe converts an incoming floating-point fragment Z to
                // the attachment's integer UNORM domain before testing it.
                // Comparing the raw float against a decoded stored value can
                // incorrectly reject two depths that quantize to the same
                // Z16/Z24/Z32 value under LEQUAL/EQUAL.
                incoming_encoded_depth = EncodeDepthAttachmentUnorm(
                    sample_depth[sample], state.depth_attachment_format);
                passes = DepthPass(state.raster_state.depth.compare_op,
                                   incoming_encoded_depth,
                                   encoded_depth[pixel_index]);
              }
              if (!passes)
                ++depth_rejected;
            }
            if (stencil_state.test_enable && has_stencil) {
              const StencilOp op = !stencil_passes  ? face.fail_op
                                   : passes         ? face.pass_op
                                                    : face.depth_fail_op;
              const std::uint8_t write_mask =
                  static_cast<std::uint8_t>(face.write_mask & 0xFFU);
              if (write_mask != 0) {
                const std::uint8_t updated = ApplyStencilOp(
                    op, stencil[pixel_index],
                    static_cast<std::uint8_t>(face.reference & 0xFFU));
                const std::uint8_t written = static_cast<std::uint8_t>(
                    (stencil[pixel_index] & ~write_mask) |
                    (updated & write_mask));
                if (written != stencil[pixel_index])
                  ++stencil_written;
                stencil[pixel_index] = written;
              }
            }
            if (!passes)
              continue;

            candidates[candidate_index].sample_mask |= sample_bit;
            candidates[candidate_index].visibility =
                FragmentVisibility::kVisible;
            if (opaque_early_hsr) {
              if (owner[pixel_index] != kNoOwner) {
                FragmentCandidate &previous = candidates[owner[pixel_index]];
                previous.sample_mask &= ~sample_bit;
                if (previous.sample_mask == 0)
                  previous.visibility = FragmentVisibility::kRejected;
              }
              owner[pixel_index] = candidate_index;
            }
            if (state.raster_state.depth.test_enable && early_depth_write) {
              if (state.depth_attachment_format == 0) {
                depth[pixel_index] = sample_depth[sample];
              } else {
                encoded_depth[pixel_index] = incoming_encoded_depth;
                depth[pixel_index] = DecodeDepthAttachmentUnorm(
                    encoded_depth[pixel_index],
                    state.depth_attachment_format);
              }
              ++depth_written;
            }
            }
          }
        }
      }
    }

    const std::uint64_t visible = static_cast<std::uint64_t>(std::count_if(
        candidates.begin(), candidates.end(), [](const FragmentCandidate &c) {
          return c.visibility == FragmentVisibility::kVisible;
        }));
    if (visible > candidates.size())
      throw std::runtime_error("ISP visible count exceeds candidate count");
    if (visible > std::numeric_limits<std::uint32_t>::max())
      throw std::overflow_error("ISP visible pixel count exceeds uint32_t");

    state.fragment_candidates = StoreNewArray(pool_, candidates);
    if (state.capture_depth_attachment != 0 || late_depth_stencil) {
      if (state.capture_depth_attachment > 1 ||
          (state.capture_depth_attachment && state.depth_attachment_format == 0) ||
          HasPoolHandle(state.isp_depth_attachment)) {
        throw std::runtime_error(
            "ISP depth attachment capture state is invalid");
      }
      if (state.depth_attachment_format == 0) {
        for (std::size_t sample = 0; sample < depth.size(); ++sample)
          std::memcpy(&encoded_depth[sample], &depth[sample], sizeof(float));
      }
      state.isp_depth_attachment = StoreNewArray(pool_, encoded_depth);
      if (has_stencil)
        state.isp_stencil_attachment = StoreNewArray(pool_, stencil);
    } else if (HasPoolHandle(state.isp_depth_attachment)) {
      throw std::runtime_error("ISP received an unexpected final depth payload");
    }
    state.active_fragment_invocations = static_cast<std::uint32_t>(visible);
    state.counters.fragment_candidates = candidates.size();
    state.counters.hsr_rejected_fragments = candidates.size() - visible;
    state.counters.covered_pixels = covered_pixels;
    state.counters.stencil_tested_fragments = stencil_tested;
    state.counters.stencil_rejected_fragments = stencil_rejected;
    state.counters.stencil_written_fragments = stencil_written;
    state.counters.depth_tested_fragments = depth_tested;
    state.counters.depth_rejected_fragments = depth_rejected;
    state.counters.depth_written_fragments = depth_written;
    const std::uint64_t functional_cycles =
        kReferenceUarch.isp_base_cycles +
        CeilDivide(candidates.size(), kReferenceUarch.isp_candidates_per_batch);
    ApplyMemoryAccessStats(state.counters, memory_stats);
    const std::uint64_t cycles =
        functional_cycles + MemoryAccessDelayCycles(memory_stats);
    state.counters.isp_cycles = cycles;
    state.counters.renderer_cycles += cycles;
    state.stage = PipelineStage::kVisibilityReady;

    WaitForCycles(cycles);
    StorePipelineState(pool_, txn.state, state);
    output.write(txn);
  }
}

} // namespace pvrgpu::stub
