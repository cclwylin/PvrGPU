// Commit the final native depth/stencil planes through modeled DRAM.
// Early tests commit at FragmentFrontend; shader-depth draws commit only after
// PBE's late tests, at PbeWriteBack. Both use the same attachment lifecycle.
#pragma once

#include "common/functional_types.h"
#include "common/msaa.h"
#include "common/pipeline_state.h"
#include "memory/gpu_memory_system.h"

#include <limits>
#include <stdexcept>
#include <vector>

namespace pvrgpu::stub {

inline void MaterializeDepthAttachment(MemoryPool &pool, GpuMemorySystem *memory,
                                PipelineState *state) {
  if (!state)
    throw std::runtime_error("Depth attachment commit: has no depth state");
  if (state->capture_depth_attachment > 1 ||
      state->depth_attachment_ready > 1 ||
      HasPoolHandle(state->depth_attachment) ||
      state->depth_attachment_bytes != 0 ||
      state->depth_attachment_ready != 0) {
    throw std::runtime_error(
        "Depth attachment commit: depth attachment control is invalid");
  }
  if (state->capture_depth_attachment == 0) {
    if (HasPoolHandle(state->isp_depth_attachment))
      throw std::runtime_error(
          "Depth attachment commit: received unrequested final depth values");
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
        "Depth attachment commit: sequence depth attachment address is invalid");
  }

  if (state->width == 0 || state->height == 0 ||
      !IsSupportedRasterSampleCount(state->raster_state.sample_count) ||
      state->depth_attachment_format == 0 ||
      !HasPoolHandle(state->isp_depth_attachment)) {
    throw std::runtime_error(
        "Depth attachment commit: cannot materialize this final depth attachment");
  }
  const std::uint64_t pixel_count =
      static_cast<std::uint64_t>(state->width) * state->height *
      state->raster_state.sample_count * state->attachment_layers;
  if (pixel_count == 0 ||
      pixel_count > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error(
        "Depth attachment commit: depth attachment size is invalid");
  }

  const std::vector<std::uint32_t> final_depth =
      LoadArray<std::uint32_t>(pool, state->isp_depth_attachment);
  if (final_depth.size() != pixel_count)
    throw std::runtime_error(
        "Depth attachment commit: final depth value count mismatch");
  const std::size_t bytes_per_pixel =
      DepthAttachmentBytesPerPixel(state->depth_attachment_format);
  if (pixel_count > std::numeric_limits<std::uint64_t>::max() /
                        bytes_per_pixel)
    throw std::overflow_error(
        "Depth attachment commit: depth attachment byte size overflow");
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
          "Depth attachment commit: final stencil value count mismatch");
    }
  }
  std::vector<std::uint8_t> attachment = EncodeDepthAttachmentUnormBytes(
      final_depth, state->depth_attachment_format,
      final_stencil.empty() ? nullptr : &final_stencil);
  if (attachment.size() != attachment_bytes)
    throw std::runtime_error(
        "Depth attachment commit: encoded depth attachment size mismatch");
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
        "Depth attachment commit: depth attachment DRAM readback mismatch");
  }
  ApplyMemoryAccessStats(state->counters, memory_stats);
  WaitForCycles(MemoryAccessDelayCycles(memory_stats));
  state->depth_attachment = StoreNewArray(pool, readback.data);
  state->depth_attachment_bytes = attachment_bytes;
  state->depth_attachment_ready = 1;
}

} // namespace pvrgpu::stub
