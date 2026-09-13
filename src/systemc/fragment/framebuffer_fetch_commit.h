// Commit-order shadow used only to seed coherent framebuffer-fetch PIXOUT
// reads.  It owns private copies of encoded color and late depth/stencil state;
// the real PBE independently replays the same raw outputs from the same LOAD
// state and remains the only module that publishes attachment writes.
#pragma once

#include "common/color_attachment_formats.h"
#include "common/msaa.h"
#include "common/pipeline_state.h"
#include "fragment/pbe_color_commit.h"
#include "fragment/pbe_depth_stencil_commit.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace pvrgpu::stub {

struct FramebufferFetchCommitStorage {
  std::vector<std::uint8_t> bytes;
  std::array<std::size_t, kMaxRenderTargets + 1> target_offsets{};
  std::vector<std::uint32_t> late_depth;
  std::vector<std::uint8_t> late_stencil;
  std::vector<std::uint64_t> last_submit_ordinal;
  std::vector<std::uint8_t> committed;
  std::size_t stored_samples = 0;
  std::uint32_t render_target_count = 0;
  bool late_depth_stencil = false;
  bool enabled = false;
};

// Returns the number of samples whose color was committed. A discarded,
// A2C-masked or depth/stencil-rejected fragment therefore returns zero.
inline std::uint32_t CommitFramebufferFetchFragment(
    FramebufferFetchCommitStorage &storage, const PipelineState &state,
    const FragmentInvocation &invocation, const FragmentOutput &output) {
  if (!storage.enabled)
    return 0;
  const std::uint32_t targets =
      state.render_target_count ? state.render_target_count : 1U;
  if (storage.render_target_count != output.render_target_count ||
      storage.render_target_count != targets || output.x != invocation.x ||
      output.y != invocation.y ||
      output.primitive_id != invocation.primitive_id ||
      output.parameter_index != invocation.parameter_index ||
      output.submit_ordinal != invocation.submit_ordinal ||
      invocation.x >= state.width || invocation.y >= state.height ||
      invocation.layer >= state.attachment_layers ||
      storage.committed.size() != storage.stored_samples ||
      storage.last_submit_ordinal.size() != storage.stored_samples ||
      storage.target_offsets[targets] != storage.bytes.size()) {
    throw std::runtime_error(
        "fragment framebuffer feedback metadata is invalid");
  }
  if (output.discarded > 1)
    throw std::runtime_error(
        "fragment framebuffer feedback discard flag is invalid");
  if (output.discarded) {
    if (!state.raster_state.shader_may_discard ||
        (!storage.late_depth_stencil &&
         !state.raster_state.shader_early_tests)) {
      throw std::runtime_error(
          "fragment framebuffer feedback discard scheduling is invalid");
    }
    return 0;
  }

  const std::uint32_t sample_count = state.raster_state.sample_count;
  const std::uint32_t valid_samples = RasterSampleMask(sample_count);
  if (invocation.sample_mask == 0 ||
      (invocation.sample_mask & ~valid_samples) != 0) {
    throw std::runtime_error(
        "fragment framebuffer feedback sample mask is invalid");
  }
  if (storage.late_depth_stencil !=
          RasterRequiresLateDepthStencil(state.raster_state) ||
      (storage.late_depth_stencil &&
       storage.late_depth.size() != storage.stored_samples) ||
      (DepthAttachmentHasStencil(state.depth_attachment_format) &&
       storage.late_depth_stencil &&
       storage.late_stencil.size() != storage.stored_samples) ||
      (state.raster_state.shader_writes_depth &&
       !state.raster_state.shader_early_tests && output.depth_written != 1) ||
      (storage.late_depth_stencil && invocation.front_facing > 1)) {
    throw std::runtime_error(
        "fragment framebuffer feedback depth/stencil state is invalid");
  }

  std::uint32_t coverage = invocation.sample_mask;
  // A2C consumes the original shader alpha, before alpha-to-one and before
  // any depth/stencil state transition, exactly as the real PBE does.
  if (state.raster_state.alpha_to_coverage &&
      (state.fragment_output_mask[0] & output.written_mask[0] & 8U) != 0) {
    coverage &= RasterAlphaCoverageMask(
        sample_count, PbeFloatFromBits(output.pixel_output[3]), output.x,
        output.y, state.raster_state.alpha_to_coverage_dither != 0,
        state.raster_state.multisample_enable != 0);
  }

  const std::size_t pixel_index =
      (static_cast<std::size_t>(invocation.layer) * state.height +
       invocation.y) *
          state.width +
      invocation.x;
  std::uint32_t committed_samples = 0;
  for (std::uint32_t sample = 0; sample < sample_count; ++sample) {
    if (!(coverage & (1U << sample)))
      continue;
    const std::size_t stored_index = pixel_index * sample_count + sample;
    if (stored_index >= storage.stored_samples)
      throw std::runtime_error(
          "fragment framebuffer feedback index is out of bounds");
    if (storage.late_depth_stencil) {
      const float incoming_depth = state.raster_state.shader_writes_depth
                                       ? output.depth
                                       : invocation.sample_depth[sample];
      if (!CommitPbeLateDepthStencil(
              state, invocation, incoming_depth, stored_index,
              storage.late_depth, storage.late_stencil))
        continue;
    }
    if (storage.committed[stored_index] &&
        output.submit_ordinal < storage.last_submit_ordinal[stored_index]) {
      throw std::runtime_error(
          "fragment framebuffer feedback lost API commit order");
    }
    storage.committed[stored_index] = 1;
    storage.last_submit_ordinal[stored_index] = output.submit_ordinal;
    for (std::uint32_t target = 0; target < targets; ++target) {
      std::uint8_t *destination =
          storage.bytes.data() + storage.target_offsets[target] +
          stored_index * ColorAttachmentBytesPerPixel(state, target);
      CommitPbeColorSample(state, output, target, destination);
    }
    ++committed_samples;
  }
  return committed_samples;
}

} // namespace pvrgpu::stub
