// FragmentFrontend converts ISP (Image Synthesis Processor) visibility into
// spatial 2x2 fragment quads for USC (Unified Shading Cluster) issue. It
// preserves primitive identity, submit order, barycentrics, depth and sample
// mask in both flat invocations and explicit primitive/quad lane maps for PDS.
// Opaque state requires one HSR owner per pixel; blended state permits
// ordered multiple invocations per pixel. FIFO traffic carries only the
// MemoryPool state handle.
#include "fragment/fragment_frontend.h"

#include "common/functional_types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace pvrgpu::stub {

namespace {

void MaterializeDepthAttachment(MemoryPool &pool, GpuMemorySystem *memory,
                                PipelineState *state) {
  if (!state)
    throw std::runtime_error("FragmentFrontend has no depth state");
  if (state->capture_depth_attachment > 1 ||
      state->depth_attachment_ready > 1 ||
      HasPoolHandle(state->depth_attachment) ||
      state->depth_attachment_bytes != 0 ||
      state->depth_attachment_ready != 0) {
    throw std::runtime_error(
        "FragmentFrontend depth attachment control is invalid");
  }
  if (state->capture_depth_attachment == 0) {
    if (HasPoolHandle(state->isp_depth_attachment))
      throw std::runtime_error(
          "FragmentFrontend received unrequested final depth values");
    return;
  }
  const std::uint64_t depth_offset =
      state->depth_attachment_gpu_address -
      kDriverPcoSequenceDepthAddressBase;
  if (!memory ||
      state->depth_attachment_gpu_address <
          kDriverPcoSequenceDepthAddressBase ||
      depth_offset % kDriverPcoSequenceAttachmentStride != 0 ||
      depth_offset / kDriverPcoSequenceAttachmentStride >=
          kDriverPcoMaximumNestedSequenceCommands) {
    throw std::runtime_error(
        "FragmentFrontend sequence depth attachment address is invalid");
  }

  if (state->width == 0 || state->height == 0 ||
      state->raster_state.sample_count != 1 ||
      state->raster_state.shader_may_discard != 0 ||
      state->raster_state.shader_writes_depth != 0 ||
      state->raster_state.shader_writes_sample_mask != 0 ||
      state->depth_attachment_format == 0 ||
      !HasPoolHandle(state->isp_depth_attachment)) {
    throw std::runtime_error(
        "FragmentFrontend cannot materialize this final depth attachment");
  }
  const std::uint64_t pixel_count =
      static_cast<std::uint64_t>(state->width) * state->height;
  if (pixel_count == 0 ||
      pixel_count > std::numeric_limits<std::size_t>::max() ||
      pixel_count > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error(
        "FragmentFrontend depth attachment size is invalid");
  }

  const std::vector<std::uint32_t> final_depth =
      LoadArray<std::uint32_t>(pool, state->isp_depth_attachment);
  if (final_depth.size() != pixel_count)
    throw std::runtime_error(
        "FragmentFrontend final depth value count mismatch");
  const std::size_t bytes_per_pixel =
      DepthAttachmentBytesPerPixel(state->depth_attachment_format);
  if (pixel_count > std::numeric_limits<std::uint64_t>::max() /
                        bytes_per_pixel)
    throw std::overflow_error(
        "FragmentFrontend depth attachment byte size overflow");
  const std::uint64_t attachment_bytes = pixel_count * bytes_per_pixel;
  // A combined attachment carries the stencil plane the ISP left beside the
  // depth one; writing back without it would erase every stencil op the draw
  // performed.
  std::vector<std::uint8_t> final_stencil;
  const bool has_stencil =
      DepthAttachmentHasStencil(state->depth_attachment_format);
  if (has_stencil && HasPoolHandle(state->isp_stencil_attachment)) {
    final_stencil = LoadArray<std::uint8_t>(pool, state->isp_stencil_attachment);
    if (final_stencil.size() != pixel_count) {
      throw std::runtime_error(
          "FragmentFrontend final stencil value count mismatch");
    }
  }
  std::vector<std::uint8_t> attachment = EncodeDepthAttachmentUnormBytes(
      final_depth, state->depth_attachment_format,
      final_stencil.empty() ? nullptr : &final_stencil);
  if (attachment.size() != attachment_bytes)
    throw std::runtime_error(
        "FragmentFrontend encoded depth attachment size mismatch");
  MemoryAccessStats memory_stats = memory->Write(
      state->depth_attachment_gpu_address, attachment.data(),
      static_cast<std::size_t>(attachment_bytes), MemoryClient::kFramebuffer);
  MemoryReadResult readback = memory->Readback(
      state->depth_attachment_gpu_address,
      static_cast<std::size_t>(attachment_bytes),
      MemoryClient::kFramebufferReadback);
  memory_stats += readback.stats;
  if (readback.data != attachment) {
    throw std::runtime_error(
        "FragmentFrontend depth attachment DRAM readback mismatch");
  }
  ApplyMemoryAccessStats(state->counters, memory_stats);
  WaitForCycles(MemoryAccessDelayCycles(memory_stats));
  state->depth_attachment = StoreNewArray(pool, readback.data);
  state->depth_attachment_bytes = attachment_bytes;
  state->depth_attachment_ready = 1;
}

} // namespace

FragmentFrontend::FragmentFrontend(sc_core::sc_module_name name,
                                   MemoryPool &pool,
                                   GpuMemorySystem *memory)
    : sc_module(name), pool_(pool), memory_(memory) {
  SC_THREAD(Run);
}

void FragmentFrontend::Run() {
  while (true) {
    const PipelineTxn txn = input.read();
    PipelineState state = LoadPipelineState(pool_, txn.state);
    RequireStage(state.stage, PipelineStage::kVisibilityReady, name());
    if (memory_ && state.memory_mode != memory_->mode())
      throw std::runtime_error("FragmentFrontend memory mode mismatch");
    if (!IsRasterFunctionalCase(state.functional_case) ||
        !HasPoolHandle(state.fragment_candidates)) {
      throw std::runtime_error(
          "FragmentFrontend received no ISP visibility payload");
    }

    const std::vector<FragmentCandidate> candidates =
        LoadArray<FragmentCandidate>(pool_, state.fragment_candidates);
    MemoryAccessStats memory_stats;
    std::vector<ParameterTriangle> parameters;
    if (memory_) {
      auto read = ReadMemoryArray<ParameterTriangle>(
          *memory_, state.parameter_triangles_gpu_address,
          state.parameter_triangles_bytes, MemoryClient::kParameterRead);
      parameters = std::move(read.values);
      memory_stats += read.stats;
    } else {
      if (!HasPoolHandle(state.parameter_triangles))
        throw std::runtime_error("FragmentFrontend has no parameter payload");
      parameters =
          LoadArray<ParameterTriangle>(pool_, state.parameter_triangles);
    }
    const std::uint64_t pixel_count =
        static_cast<std::uint64_t>(state.width) * state.height;
    if (pixel_count > std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("FragmentFrontend surface is too large");
    const std::uint32_t quads_x = static_cast<std::uint32_t>(
        CeilDivide(state.width, kReferenceUarch.fragment_quad_width));
    const std::uint32_t quads_y = static_cast<std::uint32_t>(
        CeilDivide(state.height, kReferenceUarch.fragment_quad_height));
    const std::uint64_t possible_quads =
        static_cast<std::uint64_t>(quads_x) * quads_y;
    if (possible_quads > std::numeric_limits<std::size_t>::max())
      throw std::overflow_error("FragmentFrontend quad grid is too large");
    if (possible_quads > std::numeric_limits<std::uint32_t>::max())
      throw std::overflow_error("FragmentFrontend quad ID exceeds uint32_t");

    std::vector<std::uint32_t> pixel_seen(
        static_cast<std::size_t>(pixel_count), 0);
    std::vector<std::uint64_t> last_submit_ordinal(
        static_cast<std::size_t>(pixel_count), 0);
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::size_t>
        quad_indices;
    std::vector<FragmentQuad> fragment_quads;
    std::vector<FragmentInvocation> invocations;
    invocations.reserve(state.active_fragment_invocations);
    for (const FragmentCandidate &candidate : candidates) {
      if (candidate.parameter_index >= parameters.size()) {
        throw std::runtime_error(
            "FragmentFrontend candidate parameter index is out of bounds");
      }
      const ParameterTriangle &parameter =
          parameters[candidate.parameter_index];
      if (!parameter.rasterizable || parameter.face_culled ||
          parameter.front_facing > 1 || parameter.reserved[0] != 0 ||
          parameter.reserved[1] != 0 || parameter.reserved[2] != 0 ||
          !HasCanonicalDepthPlaneMetadata(state.functional_case, parameter) ||
          parameter.key.api_primitive_id != candidate.primitive_id ||
          parameter.key.submit_ordinal != candidate.submit_ordinal ||
          candidate.x >= state.width || candidate.y >= state.height ||
          candidate.sample_mask != 1 || candidate.reserved[0] != 0 ||
          candidate.reserved[1] != 0 || !std::isfinite(candidate.depth) ||
          !std::isfinite(candidate.barycentric[0]) ||
          !std::isfinite(candidate.barycentric[1]) ||
          !std::isfinite(candidate.barycentric[2])) {
        throw std::runtime_error(
            "FragmentFrontend candidate metadata is invalid");
      }
      if (candidate.visibility != FragmentVisibility::kVisible &&
          candidate.visibility != FragmentVisibility::kRejected) {
        throw std::runtime_error(
            "FragmentFrontend candidate visibility is invalid");
      }
      if (candidate.visibility != FragmentVisibility::kVisible)
        continue;
      const std::size_t pixel_index =
          static_cast<std::size_t>(candidate.y) * state.width + candidate.x;
      if (pixel_seen[pixel_index] != 0) {
        if (!state.raster_state.blend.enable) {
          throw std::runtime_error(
              "FragmentFrontend received multiple opaque HSR owners");
        }
        if (candidate.submit_ordinal < last_submit_ordinal[pixel_index]) {
          throw std::runtime_error(
              "FragmentFrontend blended fragments lost API order");
        }
      }
      ++pixel_seen[pixel_index];
      last_submit_ordinal[pixel_index] = candidate.submit_ordinal;
      const std::uint32_t quad_x =
          candidate.x / kReferenceUarch.fragment_quad_width;
      const std::uint32_t quad_y =
          candidate.y / kReferenceUarch.fragment_quad_height;
      const std::uint32_t quad_id = quad_y * quads_x + quad_x;

      FragmentInvocation invocation;
      invocation.x = candidate.x;
      invocation.y = candidate.y;
      invocation.primitive_id = candidate.primitive_id;
      invocation.parameter_index = candidate.parameter_index;
      invocation.submit_ordinal = candidate.submit_ordinal;
      invocation.quad_id = quad_id;
      invocation.quad_lane = static_cast<std::uint8_t>(
          (candidate.y % kReferenceUarch.fragment_quad_height) *
              kReferenceUarch.fragment_quad_width +
          (candidate.x % kReferenceUarch.fragment_quad_width));
      invocation.sample_mask = candidate.sample_mask;
      invocation.depth = candidate.depth;
      for (std::size_t component = 0; component < 3; ++component)
        invocation.barycentric[component] = candidate.barycentric[component];
      const std::uint32_t invocation_index =
          static_cast<std::uint32_t>(invocations.size());
      invocations.push_back(invocation);

      const std::pair<std::uint32_t, std::uint32_t> key = {
          candidate.parameter_index, quad_id};
      auto [quad_it, inserted] =
          quad_indices.emplace(key, fragment_quads.size());
      if (inserted) {
        FragmentQuad quad;
        quad.parameter_index = candidate.parameter_index;
        quad.quad_id = quad_id;
        quad.submit_ordinal = candidate.submit_ordinal;
        fragment_quads.push_back(quad);
      }
      FragmentQuad &quad = fragment_quads[quad_it->second];
      if (quad.parameter_index != candidate.parameter_index ||
          quad.quad_id != quad_id ||
          quad.submit_ordinal != candidate.submit_ordinal ||
          invocation.quad_lane >= 4) {
        throw std::runtime_error(
            "FragmentFrontend quad identity is inconsistent");
      }
      const std::uint8_t lane_bit =
          static_cast<std::uint8_t>(1U << invocation.quad_lane);
      if ((quad.coverage_mask & lane_bit) != 0 ||
          quad.invocation_indices[invocation.quad_lane] !=
              kInvalidFragmentInvocationIndex) {
        throw std::runtime_error(
            "FragmentFrontend received duplicate primitive/quad lane work");
      }
      quad.invocation_indices[invocation.quad_lane] = invocation_index;
      quad.coverage_mask |= lane_bit;
      quad.write_mask |= lane_bit;
    }
    if (invocations.size() != state.active_fragment_invocations) {
      throw std::runtime_error(
          "FragmentFrontend visible invocation count mismatch");
    }
    MaterializeDepthAttachment(pool_, memory_, &state);
    std::vector<FragmentShaderLane> shader_lanes;
    if (UsesTextureSampling(state)) {
      // The selected reference USC dispatches a touched 4x2 SIMD half-stamp as
      // two architectural 2x2 quads. Lanes outside coverage are helpers: they
      // execute interpolation/SMP for derivatives and texture issue but never
      // write. A half-stamp with no covered sample for that primitive is not
      // issued. This spatial rule derives only from ISP work; it does not use
      // an image size, case label, or expected texture counter.
      using PixelKey =
          std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>;
      std::map<PixelKey, std::uint32_t> visible_invocations;
      for (std::uint32_t index = 0; index < invocations.size(); ++index) {
        const FragmentInvocation &invocation = invocations[index];
        visible_invocations.emplace(
            PixelKey{invocation.parameter_index, invocation.x, invocation.y},
            index);
      }
      std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint8_t>
          touched_half_stamps;
      const std::uint32_t half_stamps_x = static_cast<std::uint32_t>(
          CeilDivide(state.width, 4U));
      for (const FragmentCandidate &candidate : candidates) {
        if (candidate.parameter_index >= parameters.size() ||
            candidate.x >= state.width || candidate.y >= state.height)
          throw std::runtime_error(
              "FragmentFrontend texture stamp candidate is invalid");
        const std::uint32_t half_stamp_id =
            (candidate.y / 2U) * half_stamps_x + candidate.x / 4U;
        touched_half_stamps.emplace(
            std::make_pair(candidate.parameter_index, half_stamp_id), 1U);
      }

      fragment_quads.clear();
      shader_lanes.reserve(touched_half_stamps.size() * 8U);
      fragment_quads.reserve(touched_half_stamps.size() * 2U);
      for (const auto &entry : touched_half_stamps) {
        const std::uint32_t parameter_index = entry.first.first;
        const std::uint32_t half_stamp_id = entry.first.second;
        const std::uint32_t stamp_x =
            (half_stamp_id % half_stamps_x) * 4U;
        const std::uint32_t stamp_y =
            (half_stamp_id / half_stamps_x) * 2U;
        const ParameterTriangle &parameter = parameters[parameter_index];
        for (std::uint32_t child_x = 0; child_x < 2U; ++child_x) {
            const std::uint32_t quad_x = stamp_x + child_x * 2U;
            const std::uint32_t quad_y = stamp_y;
            if (quad_x >= state.width || quad_y >= state.height)
              continue;
            FragmentQuad quad;
            quad.parameter_index = parameter_index;
            quad.quad_id = (quad_y / 2U) * quads_x + quad_x / 2U;
            quad.submit_ordinal = parameter.key.submit_ordinal;
            for (std::uint8_t lane = 0; lane < 4U; ++lane) {
              const std::uint32_t x = quad_x + lane % 2U;
              const std::uint32_t y = quad_y + lane / 2U;
              if (x >= state.width || y >= state.height)
                continue;
              FragmentShaderLane shader_lane;
              shader_lane.x = x;
              shader_lane.y = y;
              shader_lane.primitive_id = parameter.key.api_primitive_id;
              shader_lane.parameter_index = parameter_index;
              shader_lane.submit_ordinal = parameter.key.submit_ordinal;
              shader_lane.quad_id = quad.quad_id;
              shader_lane.quad_lane = lane;
              shader_lane.sample_mask = 1;
              const auto visible = visible_invocations.find(
                  PixelKey{parameter_index, x, y});
              const std::uint8_t lane_bit =
                  static_cast<std::uint8_t>(1U << lane);
              if (visible != visible_invocations.end()) {
                shader_lane.visible_invocation_index = visible->second;
                const FragmentInvocation &invocation =
                    invocations[visible->second];
                shader_lane.depth = invocation.depth;
                for (std::size_t component = 0; component < 3; ++component)
                  shader_lane.barycentric[component] =
                      invocation.barycentric[component];
                quad.coverage_mask |= lane_bit;
                quad.write_mask |= lane_bit;
              } else {
                shader_lane.helper = 1;
                quad.helper_mask |= lane_bit;
              }
              if (shader_lanes.size() >
                  std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error(
                    "FragmentFrontend texture shader-lane index overflow");
              }
              quad.invocation_indices[lane] =
                  static_cast<std::uint32_t>(shader_lanes.size());
              shader_lanes.push_back(shader_lane);
            }
            if ((quad.coverage_mask | quad.helper_mask) == 0 ||
                (quad.coverage_mask & quad.helper_mask) != 0 ||
                quad.write_mask != quad.coverage_mask) {
              throw std::runtime_error(
                  "FragmentFrontend generated an invalid texture quad");
            }
            fragment_quads.push_back(quad);
        }
      }
    }
    std::uint64_t grouped_invocations = 0;
    for (const FragmentQuad &quad : fragment_quads) {
      const bool texture_case = UsesTextureSampling(state);
      const std::uint8_t active_mask = static_cast<std::uint8_t>(
          quad.coverage_mask | quad.helper_mask);
      if ((!texture_case &&
           (quad.coverage_mask == 0 || quad.helper_mask != 0)) ||
          (texture_case &&
           (active_mask == 0 ||
            (quad.coverage_mask & quad.helper_mask) != 0)) ||
          quad.write_mask != quad.coverage_mask || quad.reserved != 0) {
        throw std::runtime_error(
            "FragmentFrontend generated invalid quad masks");
      }
      for (std::size_t lane = 0; lane < 4; ++lane) {
        const bool covered = (quad.coverage_mask & (1U << lane)) != 0;
        const bool active = (active_mask & (1U << lane)) != 0;
        const std::uint32_t invocation_index = quad.invocation_indices[lane];
        if (active ==
            (invocation_index == kInvalidFragmentInvocationIndex)) {
          throw std::runtime_error(
              "FragmentFrontend quad lane/index mapping is invalid");
        }
        if (active) {
          const std::size_t lane_limit = texture_case
                                             ? shader_lanes.size()
                                             : invocations.size();
          if (invocation_index >= lane_limit)
            throw std::runtime_error(
                "FragmentFrontend quad invocation index is out of bounds");
          if (covered)
            ++grouped_invocations;
        }
      }
    }
    if (grouped_invocations != invocations.size()) {
      throw std::runtime_error(
          "FragmentFrontend quad grouping lost an invocation");
    }
    const std::uint64_t active_quad_count = fragment_quads.size();
    if ((invocations.empty() && active_quad_count != 0) ||
        (!invocations.empty() && active_quad_count == 0)) {
      throw std::runtime_error(
          "FragmentFrontend active quad count is inconsistent");
    }

    state.fragment_invocations = StoreNewArray(pool_, invocations);
    if (UsesTextureSampling(state)) {
      state.fragment_shader_lanes = StoreNewArray(pool_, shader_lanes);
      state.fragment_shader_lane_count =
          static_cast<std::uint32_t>(shader_lanes.size());
    } else {
      state.fragment_shader_lane_count =
          static_cast<std::uint32_t>(invocations.size());
    }
    state.fragment_quads = StoreNewArray(pool_, fragment_quads);
    state.counters.ps_invocations = invocations.size();
    state.fragment_groups = active_quad_count;
    const std::uint64_t functional_cycles =
        kReferenceUarch.fragment_frontend_base_cycles +
        CeilDivide(state.fragment_shader_lane_count,
                   kReferenceUarch.fragment_lanes_per_batch);
    ApplyMemoryAccessStats(state.counters, memory_stats);
    const std::uint64_t cycles =
        functional_cycles + MemoryAccessDelayCycles(memory_stats);
    state.counters.fragment_frontend_cycles = cycles;
    state.counters.renderer_cycles += cycles;
    state.stage = PipelineStage::kFragmentsReady;

    WaitForCycles(cycles);
    StorePipelineState(pool_, txn.state, state);
    output.write(txn);
  }
}

} // namespace pvrgpu::stub
