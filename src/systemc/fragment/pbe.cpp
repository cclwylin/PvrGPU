// PBE (Pixel Back End) consumes ordered USC PIXOUT records, converts raw
// float32 PIXOUT0..3 values to RGBA8 UNORM, and performs fixed-function GLES
// blend destination read/modify/write when enabled. Untouched pixels come from
// explicit render-target clear state. PixelDataMaster, SLC and DramModel then
// commit/read the result; JsonReporter never consumes this pre-memory handle.
// FIFO traffic carries only the state handle and timing is event-driven.
#include "fragment/pbe.h"

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

float BitsFloat(std::uint32_t bits) {
  float value = 0.0f;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint8_t FloatValueToUnorm8(float value) {
  if (!std::isfinite(value))
    throw std::runtime_error("PBE cannot convert a non-finite PIXOUT value");
  const float clamped = std::clamp(value, 0.0f, 1.0f);
  const float scaled = clamped * 255.0F;
  // The Gallivm/Mesa RGBA8 store path uses the UNORM conversion
  // floor(value * 255 + 0.5), including exact half-way values.  This is not
  // IEEE round-to-nearest-even: for example the real Shadow PIXOUT value
  // 0.3f scales to exactly 76.5f and must serialize as 77, not 76.
  const std::uint32_t rounded =
      static_cast<std::uint32_t>(std::floor(scaled + 0.5F));
  return static_cast<std::uint8_t>(std::min<std::uint32_t>(rounded, 255U));
}

std::uint8_t FloatBitsToUnorm8(std::uint32_t raw_bits) {
  return FloatValueToUnorm8(BitsFloat(raw_bits));
}

std::uint8_t FactorToUnorm8(pvrgpu::stub::BlendFactor factor,
                            const std::array<std::uint8_t, 4> &source,
                            const std::array<std::uint8_t, 4> &destination,
                            std::size_t component) {
  using pvrgpu::stub::BlendFactor;
  switch (factor) {
  case BlendFactor::kZero:
    return 0;
  case BlendFactor::kOne:
    return 255;
  case BlendFactor::kSourceAlpha:
    return source[3];
  case BlendFactor::kOneMinusSourceAlpha:
    return static_cast<std::uint8_t>(255U - source[3]);
  case BlendFactor::kSourceColor:
    return source[component];
  case BlendFactor::kOneMinusSourceColor:
    return static_cast<std::uint8_t>(255U - source[component]);
  case BlendFactor::kDestinationColor:
    return destination[component];
  case BlendFactor::kOneMinusDestinationColor:
    return static_cast<std::uint8_t>(255U - destination[component]);
  case BlendFactor::kDestinationAlpha:
    return destination[3];
  case BlendFactor::kOneMinusDestinationAlpha:
    return static_cast<std::uint8_t>(255U - destination[3]);
  }
  throw std::runtime_error("PBE received an unsupported blend factor");
}

std::uint8_t BlendEquationUnorm8(pvrgpu::stub::BlendEquation equation,
                                 std::uint8_t source, std::uint8_t destination,
                                 std::uint8_t source_factor,
                                 std::uint8_t destination_factor) {
  using pvrgpu::stub::BlendEquation;
  if (equation == BlendEquation::kMin) {
    return std::min(source, destination);
  }
  if (equation == BlendEquation::kMax) {
    return std::max(source, destination);
  }

  const std::int32_t term1 = static_cast<std::int32_t>(source) * source_factor;
  const std::int32_t term2 = static_cast<std::int32_t>(destination) * destination_factor;
  std::int32_t result = 0;

  if (equation == BlendEquation::kAdd) {
    result = (term1 + term2 + 127) / 255;
  } else if (equation == BlendEquation::kSubtract) {
    result = (term1 - term2 + 127) / 255;
  } else if (equation == BlendEquation::kReverseSubtract) {
    result = (term2 - term1 + 127) / 255;
  } else {
    throw std::runtime_error("PBE received an unsupported blend equation in calculation");
  }

  return static_cast<std::uint8_t>(std::clamp(result, 0, 255));
}

void ValidateBlendState(const pvrgpu::stub::BlendState &blend) {
  using pvrgpu::stub::BlendEquation;
  if (!blend.enable)
    return;
  if (blend.rgb_equation != BlendEquation::kAdd &&
      blend.rgb_equation != BlendEquation::kSubtract &&
      blend.rgb_equation != BlendEquation::kReverseSubtract &&
      blend.rgb_equation != BlendEquation::kMin &&
      blend.rgb_equation != BlendEquation::kMax) {
    throw std::runtime_error("PBE received an unsupported RGB blend equation");
  }
  if (blend.alpha_equation != BlendEquation::kAdd &&
      blend.alpha_equation != BlendEquation::kSubtract &&
      blend.alpha_equation != BlendEquation::kReverseSubtract &&
      blend.alpha_equation != BlendEquation::kMin &&
      blend.alpha_equation != BlendEquation::kMax) {
    throw std::runtime_error("PBE received an unsupported alpha blend equation");
  }
  const std::array<pvrgpu::stub::BlendFactor, 4> factors = {
      blend.source_rgb_factor,
      blend.destination_rgb_factor,
      blend.source_alpha_factor,
      blend.destination_alpha_factor,
  };
  const std::array<std::uint8_t, 4> dummy_source{};
  const std::array<std::uint8_t, 4> dummy_dest{};
  for (const pvrgpu::stub::BlendFactor factor : factors)
    (void)FactorToUnorm8(factor, dummy_source, dummy_dest, 0);
}

} // namespace

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
    ValidateBlendState(state.raster_state.blend);

    const std::uint64_t pixel_count =
        static_cast<std::uint64_t>(state.width) * state.height;
    if (pixel_count > std::numeric_limits<std::size_t>::max() / 4)
      throw std::overflow_error("PBE framebuffer size overflow");
    const std::vector<FragmentInvocation> invocations =
        LoadArray<FragmentInvocation>(pool_, state.fragment_invocations);
    const std::vector<FragmentOutput> outputs =
        LoadArray<FragmentOutput>(pool_, state.fragment_outputs);
    if (outputs.size() != invocations.size() ||
        outputs.size() != state.active_fragment_invocations) {
      throw std::runtime_error("PBE fragment input/output count mismatch");
    }

    const std::uint64_t framebuffer_bytes = pixel_count * 4U;
    if (state.color_attachment_load_enable > 1 ||
        (state.color_attachment_load_enable != 0) !=
            HasPoolHandle(state.color_attachment_load) ||
        (state.color_attachment_load_enable == 0 &&
         state.color_attachment_load_bytes != 0)) {
      throw std::runtime_error("PBE color attachment LOAD state is invalid");
    }
    const std::uint32_t render_target_count =
        state.render_target_count == 0 ? 1U : state.render_target_count;
    if (render_target_count > kMaxRenderTargets)
      throw std::runtime_error("PBE render target count is unsupported");

    // Attachment 0 honours the API-v7 LOAD payload; the remaining attachments
    // of a multiple-render-target pass start from the clear colour.
    std::vector<std::vector<std::uint8_t>> framebuffers(render_target_count);
    for (std::uint32_t target = 0; target < render_target_count; ++target) {
      std::vector<std::uint8_t> &attachment = framebuffers[target];
      if (target == 0 && state.color_attachment_load_enable != 0) {
        attachment = LoadArray<std::uint8_t>(pool_, state.color_attachment_load);
        if (state.color_attachment_load_bytes != framebuffer_bytes ||
            attachment.size() != framebuffer_bytes) {
          throw std::runtime_error(
              "PBE color attachment LOAD byte count mismatch");
        }
        continue;
      }
      attachment.assign(static_cast<std::size_t>(framebuffer_bytes), 0);
      for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
        for (std::size_t component = 0; component < 4; ++component) {
          attachment[pixel * 4 + component] =
              FloatValueToUnorm8(state.raster_state.clear_color[component]);
        }
      }
    }
    std::vector<std::uint32_t> written_map(
        static_cast<std::size_t>(pixel_count), 0);
    std::vector<std::uint64_t> last_submit_ordinal(
        static_cast<std::size_t>(pixel_count), 0);
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      const FragmentInvocation &invocation = invocations[index];
      const FragmentOutput &output = outputs[index];
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
      for (std::uint32_t target = 0; target < render_target_count; ++target) {
        if (output.written_mask[target] != 0x0f) {
          throw std::runtime_error(
              "PBE fragment did not write every PIXOUT lane of target " +
              std::to_string(target) + ": mask 0x" +
              std::to_string(output.written_mask[target]));
        }
      }
      if (output.x >= state.width || output.y >= state.height)
        throw std::runtime_error("PBE fragment coordinate is out of bounds");
      const std::size_t pixel_index =
          static_cast<std::size_t>(output.y) * state.width + output.x;
      if (written_map[pixel_index] != 0) {
        if (!state.raster_state.blend.enable)
          throw std::runtime_error("PBE attempted to shade one opaque owner twice");
        if (output.submit_ordinal < last_submit_ordinal[pixel_index])
          throw std::runtime_error("PBE blended fragments lost API order");
      }
      ++written_map[pixel_index];
      last_submit_ordinal[pixel_index] = output.submit_ordinal;
      const std::size_t byte_offset = pixel_index * 4;
      for (std::uint32_t target = 0; target < render_target_count; ++target) {
      std::vector<std::uint8_t> &framebuffer = framebuffers[target];
      std::array<std::uint8_t, 4> source{};
      for (std::size_t component = 0; component < 4; ++component) {
        source[component] =
            FloatBitsToUnorm8(output.pixel_output[target * 4 + component]);
      }

      if (state.raster_state.blend.enable) {
        const BlendState &blend = state.raster_state.blend;
        std::array<std::uint8_t, 4> destination_color{};
        for (std::size_t component = 0; component < 4; ++component) {
          destination_color[component] = framebuffer[byte_offset + component];
        }
        for (std::size_t component = 0; component < 4; ++component) {
          const BlendFactor source_factor =
              component == 3 ? blend.source_alpha_factor
                             : blend.source_rgb_factor;
          const BlendFactor destination_factor =
              component == 3 ? blend.destination_alpha_factor
                             : blend.destination_rgb_factor;
          const BlendEquation equation =
              component == 3 ? blend.alpha_equation
                             : blend.rgb_equation;
          const std::uint8_t sf = FactorToUnorm8(source_factor, source, destination_color, component);
          const std::uint8_t df = FactorToUnorm8(destination_factor, source, destination_color, component);

          const std::uint8_t blended_val = BlendEquationUnorm8(
              equation, source[component], destination_color[component], sf, df);

          if ((state.raster_state.color_mask & (1U << component)) != 0) {
            framebuffer[byte_offset + component] = blended_val;
          }
        }
      } else {
        for (std::size_t component = 0; component < 4; ++component) {
          if ((state.raster_state.color_mask & (1U << component)) != 0) {
            framebuffer[byte_offset + component] = source[component];
          }
        }
      }
      }
    }
    const std::uint64_t pixels_touched = static_cast<std::uint64_t>(
        std::count_if(written_map.begin(), written_map.end(),
                      [](std::uint32_t writes) { return writes != 0; }));
    if ((!state.raster_state.blend.enable &&
         pixels_touched != state.active_fragment_invocations) ||
        outputs.size() != state.active_fragment_invocations)
      throw std::runtime_error("PBE fragment write count mismatch");

    state.pbe_framebuffer = StoreNewArray(pool_, framebuffers[0]);
    state.framebuffer_bytes = framebuffers[0].size();
    for (std::uint32_t target = 1; target < render_target_count; ++target) {
      state.extra_pbe_framebuffer[target - 1] =
          StoreNewArray(pool_, framebuffers[target]);
    }
    // Every attachment is written for each covered pixel.
    state.counters.pbe_pixels_written = pixel_count * render_target_count;
    state.counters.pbe_fragment_writes = outputs.size();
    if (state.raster_state.blend.enable) {
      state.counters.pbe_color_reads = outputs.size();
      state.counters.pbe_blended_fragments = outputs.size();
    }
    const std::uint64_t blend_cycles =
        state.raster_state.blend.enable
            ? CeilDivide(outputs.size(),
                         kReferenceUarch.pbe_blend_fragments_per_batch)
            : 0;
    const std::uint64_t cycles =
        kReferenceUarch.pbe_base_cycles +
        CeilDivide(pixel_count, kReferenceUarch.pbe_pixels_per_batch) +
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
