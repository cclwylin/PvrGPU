// PBE (Pixel Back End) consumes ordered USC PIXOUT records, converts raw
// float32 PIXOUT values to the declared attachment format, and performs GLES
// blend destination read/modify/write when enabled. Untouched pixels come from
// explicit render-target clear state. PixelDataMaster, SLC and DramModel then
// commit/read the result; JsonReporter never consumes this pre-memory handle.
// FIFO traffic carries only the state handle and timing is event-driven.
#include "fragment/pbe.h"

#include "common/functional_types.h"
#include "common/color_attachment_formats.h"
#include "common/msaa.h"
#include "fragment/pbe_color_commit.h"
#include "fragment/pbe_depth_stencil_commit.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pvrgpu::stub {

Pbe::Pbe(sc_core::sc_module_name name, MemoryPool &pool)
    : sc_module(name), pool_(pool) {
  SC_THREAD(Run);
}

void Pbe::Run() {
  while (true) {
    const PipelineTxn txn = input.read();
    PipelineState state = LoadPipelineState(pool_, txn.state);
    RequireStage(state.stage, PipelineStage::kTextureComplete, name());
    if (!IsRasterFunctionalCase(state.functional_case) ||
        !HasPoolHandle(state.fragment_outputs) ||
        !HasPoolHandle(state.fragment_invocations)) {
      throw std::runtime_error("PBE received no supported fragment results");
    }
    ValidatePbeBlendState(state.raster_state.blend);
    if (state.raster_state.render_target_state_count > kMaxRenderTargets)
      throw std::runtime_error("PBE render-target state count is invalid");
    for (std::size_t target = 0;
         target < state.raster_state.render_target_state_count; ++target) {
      ValidatePbeBlendState(state.raster_state.target_blend[target]);
      if (state.raster_state.target_color_mask[target] > 0x0f)
        throw std::runtime_error("PBE render-target color mask is invalid");
    }

    const std::uint64_t pixel_count =
        static_cast<std::uint64_t>(state.width) * state.height * state.attachment_layers;
    const std::uint32_t sample_count = state.raster_state.sample_count;
    if (!IsSupportedRasterSampleCount(sample_count) ||
        pixel_count > std::numeric_limits<std::size_t>::max() / sample_count)
      throw std::runtime_error("PBE sample count or framebuffer size is invalid");
    const std::size_t stored_samples =
        static_cast<std::size_t>(pixel_count) * sample_count;
    const std::uint32_t render_target_count = ValidateColorAttachmentFormats(state);
    if (state.raster_state.render_target_state_count != 0 &&
        state.raster_state.render_target_state_count != render_target_count)
      throw std::runtime_error("PBE render-target state/attachment count mismatch");
    std::array<std::size_t, kMaxRenderTargets> target_bytes{};
    std::array<std::size_t, kMaxRenderTargets + 1> target_offsets{};
    for (std::uint32_t target = 0; target < render_target_count; ++target) {
      const auto bytes_per_pixel = ColorAttachmentBytesPerPixel(state, target);
      if (stored_samples > std::numeric_limits<std::size_t>::max() / bytes_per_pixel)
        throw std::overflow_error("PBE framebuffer size overflow");
      target_bytes[target] = stored_samples * bytes_per_pixel;
      if (target_bytes[target] > std::numeric_limits<std::size_t>::max() -
                                     target_offsets[target])
        throw std::overflow_error("PBE aggregate framebuffer size overflow");
      target_offsets[target + 1] = target_offsets[target] + target_bytes[target];
    }
    const std::vector<FragmentInvocation> invocations =
        LoadArray<FragmentInvocation>(pool_, state.fragment_invocations);
    const std::vector<FragmentOutput> outputs =
        LoadArray<FragmentOutput>(pool_, state.fragment_outputs);
    if (outputs.size() != invocations.size() ||
        outputs.size() != state.active_fragment_invocations) {
      throw std::runtime_error("PBE fragment input/output count mismatch");
    }

    if (state.color_attachment_load_enable > 1 ||
        (state.color_attachment_load_enable != 0) !=
            HasPoolHandle(state.color_attachment_load) ||
        (state.color_attachment_load_enable == 0 &&
         state.color_attachment_load_bytes != 0)) {
      throw std::runtime_error("PBE color attachment LOAD state is invalid");
    }
    const bool explicit_output_masks = HasExplicitFragmentOutputMasks(state);
    // The explicit format vector is the per-target storage contract. Legacy
    // GS/TES MRT still lacks that proof and stays refused.
    if ((HasPoolHandle(state.geometry_code) || HasPoolHandle(state.tessellation_state)) &&
        render_target_count > 1 && state.color_attachment_format_count == 0)
      throw std::runtime_error(
          "PBE geometry MRT requires independent attachment LOAD");
    // API-v30 LOAD is target-major. Each target retains its independent
    // samples/layers, including targets or channels this draw never writes.
    std::vector<std::uint8_t> initial_colors;
    if (state.color_attachment_load_enable) {
      initial_colors = LoadArray<std::uint8_t>(pool_, state.color_attachment_load);
      if (state.color_attachment_load_bytes != target_offsets[render_target_count] ||
          initial_colors.size() != state.color_attachment_load_bytes)
        throw std::runtime_error("PBE color attachment LOAD byte count mismatch");
    }
    std::vector<std::vector<std::uint8_t>> framebuffers(render_target_count);
    for (std::uint32_t target = 0; target < render_target_count; ++target) {
      const auto packed_format = ColorAttachmentPackedUnorm(state, target);
      const auto raw_dwords = ColorAttachmentRawDwords(state, target);
      const bool float32 = ColorAttachmentFloat32(state, target) != 0;
      const auto &codec = ColorAttachmentCodecForTarget(state, target);
      const auto bytes_per_pixel = ColorAttachmentBytesPerPixel(state, target);
      std::vector<std::uint8_t> &attachment = framebuffers[target];
      if (state.color_attachment_load_enable != 0) {
        attachment.assign(initial_colors.begin() + target_offsets[target],
                          initial_colors.begin() + target_offsets[target + 1]);
        continue;
      }
      attachment.assign(target_bytes[target], 0);
      if (ColorAttachmentCodecIsCanonical(codec)) {
        std::array<float, 4> clear{};
        std::copy_n(state.raster_state.clear_color, 4, clear.begin());
        PbeWriteCanonicalColor(codec, attachment.data(), clear);
        for (std::size_t pixel = 1; pixel < stored_samples; ++pixel)
          std::memcpy(attachment.data() + pixel * bytes_per_pixel,
                      attachment.data(), bytes_per_pixel);
      } else if (packed_format != PackedUnormFormat::kNone) {
        std::array<float, 4> clear{};
        std::copy_n(state.raster_state.clear_color, 4, clear.begin());
        const std::uint32_t word =
            PackUnormColor(clear, packed_format);
        for (std::size_t pixel = 0; pixel < stored_samples; ++pixel)
          std::memcpy(attachment.data() + pixel * 4U, &word, sizeof(word));
      } else if (float32) {
        for (std::size_t pixel = 0; pixel < stored_samples; ++pixel) {
          std::memcpy(attachment.data() + pixel * bytes_per_pixel,
                      state.raster_state.clear_color, bytes_per_pixel);
        }
      } else if (raw_dwords != 0) {
        // An integer attachment clears to the raw value, not a colour, and to
        // one such value per channel it stores.
        const std::size_t channels = raw_dwords;
        std::array<std::uint32_t, 4> raw{};
        for (std::size_t channel = 0; channel < channels; ++channel) {
          std::memcpy(&raw[channel], &state.raster_state.clear_color[channel],
                      sizeof(raw[channel]));
          if (ColorAttachmentCodecIsInteger(codec))
            raw[channel] =
                PbeCanonicalIntegerComponent(codec, channel, raw[channel]);
        }
        for (std::size_t pixel = 0; pixel < stored_samples; ++pixel) {
          std::memcpy(attachment.data() + pixel * bytes_per_pixel, raw.data(),
                      channels * sizeof(std::uint32_t));
        }
      } else {
        const bool srgb = ColorAttachmentSrgb(state, target) != 0;
        for (std::size_t pixel = 0; pixel < stored_samples; ++pixel) {
          for (std::size_t component = 0; component < 4; ++component) {
            const float value = state.raster_state.clear_color[component];
            attachment[pixel * bytes_per_pixel + component] =
                srgb ? PbeFiniteStateToUnorm8(value)
                     : PbeClearStateToUnorm8(value);
          }
        }
      }
    }
    std::vector<std::uint32_t> written_map(stored_samples, 0);
    std::vector<std::uint64_t> last_submit_ordinal(stored_samples, 0);
    const bool late_depth_stencil =
        RasterRequiresLateDepthStencil(state.raster_state);
    std::vector<std::uint32_t> late_depth;
    std::vector<std::uint8_t> late_stencil;
    std::uint64_t late_tested_samples = 0;
    if (late_depth_stencil) {
      if (!HasPoolHandle(state.isp_depth_attachment))
        throw std::runtime_error("PBE has no late depth attachment state");
      late_depth = LoadArray<std::uint32_t>(pool_, state.isp_depth_attachment);
      if (late_depth.size() != stored_samples)
        throw std::runtime_error("PBE late depth sample count is invalid");
      if (DepthAttachmentHasStencil(state.depth_attachment_format)) {
        late_stencil = LoadArray<std::uint8_t>(pool_, state.isp_stencil_attachment);
        if (late_stencil.size() != stored_samples)
          throw std::runtime_error("PBE late stencil sample count is invalid");
      }
    }
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      const FragmentInvocation &invocation = invocations[index];
      FragmentOutput output = outputs[index];
      // Name the property that broke: fragment identity and PIXOUT lane
      // coverage are different failures with different causes.
      const char *identity_reason = nullptr;
      if (output.x != invocation.x || output.y != invocation.y)
        identity_reason = "coordinate";
      else if (output.primitive_id != invocation.primitive_id)
        identity_reason = "primitive_id";
      else if (output.parameter_index != invocation.parameter_index)
        identity_reason = "parameter_index";
      else if (output.submit_ordinal != invocation.submit_ordinal)
        identity_reason = "submit_ordinal";
      else if (output.render_target_count != render_target_count)
        identity_reason = "render_target_count";
      if (identity_reason) {
        throw std::runtime_error(std::string("PBE lost fragment identity: ") +
                                 identity_reason);
      }
      if (output.discarded > 1)
        throw std::runtime_error("PBE received noncanonical shader discard flag");
      if (output.discarded) {
        if (!state.raster_state.shader_may_discard ||
            (!late_depth_stencil && !state.raster_state.shader_early_tests))
          throw std::runtime_error("PBE shader discard lacks late depth/stencil scheduling");
        continue;
      }
      // Every lane the attachment expects, which is four only when it has
      // four channels, and which each target declares for itself.
      for (std::uint32_t target = 0; target < render_target_count; ++target) {
        const std::uint32_t declared =
            target < state.fragment_output_mask.size()
                ? state.fragment_output_mask[target]
                : 0U;
        const std::uint32_t expected_pixel_output_mask =
            declared != 0 || explicit_output_masks ? declared : 0x0fU;
        if (output.written_mask[target] != expected_pixel_output_mask) {
          throw std::runtime_error(
              "PBE fragment did not write every expected PIXOUT lane of "
              "target " +
              std::to_string(target) + ": mask=" +
              std::to_string(output.written_mask[target]) + " expected=" +
              std::to_string(expected_pixel_output_mask));
        }
      }
      if (output.x >= state.width || output.y >= state.height ||
          invocation.layer >= state.attachment_layers)
        throw std::runtime_error("PBE fragment coordinate is out of bounds");
      const std::size_t pixel_index =
          (static_cast<std::size_t>(invocation.layer) * state.height + output.y) * state.width + output.x;
      std::uint32_t coverage = invocation.sample_mask;
      if (coverage == 0 || (coverage & ~RasterSampleMask(sample_count)) != 0)
        throw std::runtime_error("PBE fragment sample coverage is invalid");
      if ((state.raster_state.shader_writes_depth &&
           !state.raster_state.shader_early_tests && output.depth_written != 1) ||
          (late_depth_stencil && invocation.front_facing > 1))
        throw std::runtime_error("PBE shader depth output or facing is invalid");
      // Coverage is derived only from a declared/written DATA0 alpha, before
      // alpha-to-one and before any depth/stencil mutation. An absent alpha
      // output is not an implicit zero (llvmpipe skips A2C in that case).
      if (state.raster_state.alpha_to_coverage &&
          (state.fragment_output_mask[0] & output.written_mask[0] & 8U) != 0) {
        coverage &= RasterAlphaCoverageMask(
            sample_count, PbeFloatFromBits(output.pixel_output[3]), output.x,
            output.y,
            state.raster_state.alpha_to_coverage_dither != 0,
            state.raster_state.multisample_enable != 0);
      }
      for (std::uint32_t sample = 0; sample < sample_count; ++sample) {
        if ((coverage & (1U << sample)) == 0)
          continue;
        const std::size_t stored_index = pixel_index * sample_count + sample;
        if (late_depth_stencil) {
          ++late_tested_samples;
          const float incoming_depth = state.raster_state.shader_writes_depth
                                           ? output.depth
                                           : invocation.sample_depth[sample];
          if (!CommitPbeLateDepthStencil(
                  state, invocation, incoming_depth, stored_index, late_depth,
                  late_stencil, &state.counters))
            continue;
        }
        if (written_map[stored_index] != 0) {
          if (!AnyBlendEnabled(state.raster_state) && !late_depth_stencil &&
              !state.raster_state.shader_writes_memory &&
              !state.raster_state.shader_may_discard &&
              state.fragment_early_hsr_safe) {
            throw std::runtime_error(
                "PBE attempted to shade one opaque owner twice");
          }
          if (output.submit_ordinal < last_submit_ordinal[stored_index])
            throw std::runtime_error("PBE blended fragments lost API order");
        }
        ++written_map[stored_index];
        last_submit_ordinal[stored_index] = output.submit_ordinal;
        for (std::uint32_t target = 0; target < render_target_count; ++target) {
          // A native FS may genuinely have no output for this attachment.
          // Preserve its pixels; no default color export or blend is fabricated.
          if (explicit_output_masks && state.fragment_output_mask[target] == 0)
            continue;
          const std::size_t byte_offset =
              stored_index * ColorAttachmentBytesPerPixel(state, target);
          CommitPbeColorSample(state, output, target,
                               framebuffers[target].data() + byte_offset);
        }
      }
    }
    const std::uint64_t pixels_touched = static_cast<std::uint64_t>(
        std::count_if(written_map.begin(), written_map.end(),
                      [](std::uint32_t writes) { return writes != 0; }));
    if ((!AnyBlendEnabled(state.raster_state) && !late_depth_stencil && sample_count == 1 &&
         !state.raster_state.shader_writes_memory && !state.raster_state.shader_may_discard &&
         state.fragment_early_hsr_safe &&
         pixels_touched != state.active_fragment_invocations) ||
        outputs.size() != state.active_fragment_invocations)
      throw std::runtime_error("PBE fragment write count mismatch");

    if (late_depth_stencil) {
      StoreArray(pool_, state.isp_depth_attachment, late_depth);
      if (!late_stencil.empty())
        StoreArray(pool_, state.isp_stencil_attachment, late_stencil);
    }

    state.pbe_framebuffer = StoreNewArray(pool_, framebuffers[0]);
    state.framebuffer_bytes = framebuffers[0].size();
    for (std::uint32_t target = 1; target < render_target_count; ++target) {
      state.extra_pbe_framebuffer[target - 1] =
          StoreNewArray(pool_, framebuffers[target]);
    }
    // Pixel-frequency shader outputs can own several samples, and each
    // attachment has an independent destination. Count the work performed
    // above, including overdraw, separately from full-surface serialization.
    std::uint64_t sample_owners = 0;
    for (const std::uint32_t writes : written_map)
      sample_owners += writes;
    state.counters.occlusion_samples_passed = sample_owners;
    std::uint32_t writable_targets = 0;
    std::uint32_t blendable_targets = 0;
    for (std::uint32_t target = 0; target < render_target_count; ++target) {
      if (explicit_output_masks && state.fragment_output_mask[target] == 0)
        continue;
      const auto raw_dwords = ColorAttachmentRawDwords(state, target);
      const std::uint32_t stored_channel_mask =
          raw_dwords != 0 ? (1U << raw_dwords) - 1U : 0x0fU;
      writable_targets +=
          (ColorMaskForTarget(state.raster_state, target) & stored_channel_mask) != 0;
      blendable_targets +=
          raw_dwords == 0 && BlendStateForTarget(state.raster_state, target).enable;
    }
    state.counters.pbe_pixels_written = stored_samples * render_target_count;
    state.counters.pbe_fragment_writes = sample_owners * writable_targets;
    // Integer attachments bypass the blend equation even if API blend state
    // is enabled. A masked floating/UNORM output still runs that equation in
    // this model, but does not produce a color write when all lanes are off.
    const std::uint64_t blended_colors = sample_owners * blendable_targets;
    state.counters.pbe_color_reads = blended_colors;
    state.counters.pbe_blended_fragments = blended_colors;
    const std::uint64_t blend_cycles =
        CeilDivide(blended_colors, kReferenceUarch.pbe_blend_fragments_per_batch);
    const std::uint64_t cycles =
        kReferenceUarch.pbe_base_cycles +
        CeilDivide(state.counters.pbe_pixels_written,
                   kReferenceUarch.pbe_pixels_per_batch) +
        CeilDivide(late_tested_samples, kReferenceUarch.isp_candidates_per_batch) +
        blend_cycles;
    state.counters.pbe_cycles = cycles;
    state.counters.renderer_cycles += cycles;
    WaitForCycles(cycles);

    state.stage = PipelineStage::kPbeComplete;
    StorePipelineState(pool_, txn.state, state);
    output.write(txn);
  }
}

} // namespace pvrgpu::stub
