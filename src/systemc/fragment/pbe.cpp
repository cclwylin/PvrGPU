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

template <typename T>
bool LateDepthPass(pvrgpu::stub::DepthCompareOp operation, T incoming, T stored) {
  using pvrgpu::stub::DepthCompareOp;
  switch (operation) {
  case DepthCompareOp::kNever: return false;
  case DepthCompareOp::kLess: return incoming < stored;
  case DepthCompareOp::kEqual: return incoming == stored;
  case DepthCompareOp::kLessOrEqual: return incoming <= stored;
  case DepthCompareOp::kGreater: return incoming > stored;
  case DepthCompareOp::kNotEqual: return incoming != stored;
  case DepthCompareOp::kGreaterOrEqual: return incoming >= stored;
  case DepthCompareOp::kAlways: return true;
  }
  throw std::runtime_error("PBE late depth comparison is invalid");
}

bool TestLateDepthStencil(pvrgpu::stub::PipelineState &state,
                          const pvrgpu::stub::FragmentInvocation &invocation,
                          float shader_depth, std::size_t sample_index,
                          std::vector<std::uint32_t> &depth,
                          std::vector<std::uint8_t> &stencil) {
  using namespace pvrgpu::stub;
  const StencilState &stencil_state = state.raster_state.stencil;
  const StencilFaceState &face = invocation.front_facing
      ? stencil_state.front : stencil_state.back;
  bool stencil_passes = true;
  if (stencil_state.test_enable && !stencil.empty()) {
    ++state.counters.stencil_tested_fragments;
    const auto mask = static_cast<std::uint8_t>(face.value_mask);
    stencil_passes = StencilPass(face.compare_op,
        static_cast<std::uint8_t>(face.reference & mask),
        static_cast<std::uint8_t>(stencil[sample_index] & mask));
    if (!stencil_passes)
      ++state.counters.stencil_rejected_fragments;
  }
  bool passes = stencil_passes;
  std::uint32_t encoded = 0;
  if (stencil_passes && state.raster_state.depth.test_enable) {
    ++state.counters.depth_tested_fragments;
    // GLES restricts the shader's final depth to [0,1] before native-format
    // conversion, as llvmpipe's late-Z lp_build_depth_clamp path does.
    if (std::isnan(shader_depth))
      throw std::runtime_error("PBE shader depth is NaN");
    shader_depth = std::clamp(shader_depth, 0.0F, 1.0F);
    if (state.depth_attachment_format == 0) {
      std::memcpy(&encoded, &shader_depth, sizeof(encoded));
      passes = LateDepthPass(state.raster_state.depth.compare_op,
                             shader_depth, BitsFloat(depth[sample_index]));
    } else {
      encoded = EncodeDepthAttachmentUnorm(shader_depth, state.depth_attachment_format);
      passes = LateDepthPass(state.raster_state.depth.compare_op,
                             encoded, depth[sample_index]);
    }
    if (!passes)
      ++state.counters.depth_rejected_fragments;
  }
  if (stencil_state.test_enable && !stencil.empty()) {
    const StencilOp operation = !stencil_passes ? face.fail_op
        : passes ? face.pass_op : face.depth_fail_op;
    const auto write_mask = static_cast<std::uint8_t>(face.write_mask);
    const std::uint8_t old = stencil[sample_index];
    const std::uint8_t updated = ApplyStencilOp(
        operation, old, static_cast<std::uint8_t>(face.reference));
    stencil[sample_index] = static_cast<std::uint8_t>(
        (old & ~write_mask) | (updated & write_mask));
    if (stencil[sample_index] != old)
      ++state.counters.stencil_written_fragments;
  }
  if (passes && state.raster_state.depth.test_enable &&
      state.raster_state.depth.write_enable) {
    depth[sample_index] = encoded;
    ++state.counters.depth_written_fragments;
  }
  return passes;
}

float ClampShaderUnorm(float value) {
  // GLES 3.1 section 2.3.4.2 clamps floating-point colors before UNORM
  // conversion. Mesa util/u_math.h float_to_ubyte also maps NaN to zero.
  // This is a fixed-function conversion policy, not an ISS value rewrite:
  // +Inf saturates to one, -Inf to zero, and finite arithmetic is unchanged.
  return std::isnan(value) ? 0.0F : std::clamp(value, 0.0F, 1.0F);
}

std::uint8_t FloatValueToUnorm8(float value) {
  const float clamped = ClampShaderUnorm(value);
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

std::uint8_t FiniteStateToUnorm8(float value) {
  // Clear/blend-constant state retains its existing validation contract;
  // accepting nonfinite shader results must not relax command metadata.
  if (!std::isfinite(value))
    throw std::runtime_error("PBE cannot convert non-finite UNORM state");
  return FloatValueToUnorm8(value);
}

std::uint8_t FactorToUnorm8(pvrgpu::stub::BlendFactor factor,
                            const std::array<std::uint8_t, 4> &source,
                            const std::array<std::uint8_t, 4> &destination,
                            const std::array<std::uint8_t, 4> &constant,
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
  case BlendFactor::kSourceAlphaSaturate:
    // GLES: f = min(As, 1 - Ad) for the colour components, exactly 1 for the
    // alpha component.
    return component == 3
               ? static_cast<std::uint8_t>(255U)
               : std::min<std::uint8_t>(
                     source[3],
                     static_cast<std::uint8_t>(255U - destination[3]));
  case BlendFactor::kConstantColor:
    return constant[component];
  case BlendFactor::kOneMinusConstantColor:
    return static_cast<std::uint8_t>(255U - constant[component]);
  case BlendFactor::kConstantAlpha:
    return constant[3];
  case BlendFactor::kOneMinusConstantAlpha:
    return static_cast<std::uint8_t>(255U - constant[3]);
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

// The linear-domain blend factor, for an sRGB attachment whose blending GLES
// performs in linear space.  The values are already linear: colour channels
// decoded from the stored sRGB destination and taken straight from the shader's
// linear source, alpha and the (linear) blend constant unchanged.
float FactorToFloat(pvrgpu::stub::BlendFactor factor,
                    const std::array<float, 4> &source,
                    const std::array<float, 4> &destination,
                    const std::array<float, 4> &constant,
                    std::size_t component) {
  using pvrgpu::stub::BlendFactor;
  switch (factor) {
  case BlendFactor::kZero:
    return 0.0F;
  case BlendFactor::kOne:
    return 1.0F;
  case BlendFactor::kSourceAlpha:
    return source[3];
  case BlendFactor::kOneMinusSourceAlpha:
    return 1.0F - source[3];
  case BlendFactor::kSourceColor:
    return source[component];
  case BlendFactor::kOneMinusSourceColor:
    return 1.0F - source[component];
  case BlendFactor::kDestinationColor:
    return destination[component];
  case BlendFactor::kOneMinusDestinationColor:
    return 1.0F - destination[component];
  case BlendFactor::kDestinationAlpha:
    return destination[3];
  case BlendFactor::kOneMinusDestinationAlpha:
    return 1.0F - destination[3];
  case BlendFactor::kSourceAlphaSaturate:
    return component == 3 ? 1.0F
                          : std::min(source[3], 1.0F - destination[3]);
  case BlendFactor::kConstantColor:
    return constant[component];
  case BlendFactor::kOneMinusConstantColor:
    return 1.0F - constant[component];
  case BlendFactor::kConstantAlpha:
    return constant[3];
  case BlendFactor::kOneMinusConstantAlpha:
    return 1.0F - constant[3];
  }
  throw std::runtime_error("PBE received an unsupported blend factor");
}

float BlendEquationFloat(pvrgpu::stub::BlendEquation equation, float source,
                         float destination, float source_factor,
                         float destination_factor) {
  using pvrgpu::stub::BlendEquation;
  if (equation == BlendEquation::kMin)
    return std::min(source, destination);
  if (equation == BlendEquation::kMax)
    return std::max(source, destination);
  const float term1 = source * source_factor;
  const float term2 = destination * destination_factor;
  float result = 0.0F;
  if (equation == BlendEquation::kAdd)
    result = term1 + term2;
  else if (equation == BlendEquation::kSubtract)
    result = term1 - term2;
  else if (equation == BlendEquation::kReverseSubtract)
    result = term2 - term1;
  else
    throw std::runtime_error("PBE received an unsupported blend equation in calculation");
  // Floating-point attachments retain values outside [0,1].  A normalized
  // attachment performs its clamp when the result is encoded for storage.
  return result;
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
  const std::array<std::uint8_t, 4> dummy_constant{};
  for (const pvrgpu::stub::BlendFactor factor : factors)
    (void)FactorToUnorm8(factor, dummy_source, dummy_dest, dummy_constant, 0);
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
        static_cast<std::uint64_t>(state.width) * state.height * state.attachment_layers;
    const std::uint32_t sample_count = state.raster_state.sample_count;
    if (!IsSupportedRasterSampleCount(sample_count) ||
        pixel_count > std::numeric_limits<std::size_t>::max() / sample_count)
      throw std::runtime_error("PBE sample count or framebuffer size is invalid");
    const std::size_t stored_samples =
        static_cast<std::size_t>(pixel_count) * sample_count;
    // An integer attachment stores one dword per channel, so a pixel is not
    // always four bytes wide.  Everything below sizes and indexes through
    // this rather than assuming.
    const std::size_t bytes_per_pixel =
        ColorAttachmentBytesPerPixel(state.color_attachment_raw_dwords,
                                     state.color_attachment_float32);
    if (state.color_attachment_float32 &&
        (state.color_attachment_raw_dwords != 0 || state.color_is_srgb))
      throw std::runtime_error("PBE floating-point attachment state is invalid");
    if (state.color_attachment_packed_unorm != PackedUnormFormat::kNone) {
      (void)PackedUnormShift(state.color_attachment_packed_unorm, 0);
      if (state.color_attachment_raw_dwords || state.color_attachment_float32 ||
          state.color_is_srgb)
        throw std::runtime_error("PBE packed UNORM attachment state is invalid");
    }
    const std::uint32_t render_target_count = ValidateColorAttachmentFormats(state);
    if (stored_samples > std::numeric_limits<std::size_t>::max() / bytes_per_pixel)
      throw std::overflow_error("PBE framebuffer size overflow");
    const std::vector<FragmentInvocation> invocations =
        LoadArray<FragmentInvocation>(pool_, state.fragment_invocations);
    const std::vector<FragmentOutput> outputs =
        LoadArray<FragmentOutput>(pool_, state.fragment_outputs);
    if (outputs.size() != invocations.size() ||
        outputs.size() != state.active_fragment_invocations) {
      throw std::runtime_error("PBE fragment input/output count mismatch");
    }

    const std::uint64_t framebuffer_bytes = stored_samples * bytes_per_pixel;
    if (state.color_attachment_load_enable > 1 ||
        (state.color_attachment_load_enable != 0) !=
            HasPoolHandle(state.color_attachment_load) ||
        (state.color_attachment_load_enable == 0 &&
         state.color_attachment_load_bytes != 0)) {
      throw std::runtime_error("PBE color attachment LOAD state is invalid");
    }
    const bool explicit_output_masks = HasExplicitFragmentOutputMasks(state);
    // The explicit normalized4B vector is the new per-target storage
    // contract. Legacy GS/TES MRT still lacks that proof and stays refused.
    if ((HasPoolHandle(state.geometry_code) || HasPoolHandle(state.tessellation_state)) &&
        render_target_count > 1 && state.color_attachment_format_count == 0)
      throw std::runtime_error(
          "PBE geometry MRT requires independent attachment LOAD");
    // API-v30 LOAD is target-major. Each target retains its independent
    // samples/layers, including targets or channels this draw never writes.
    std::vector<std::uint8_t> initial_colors;
    if (state.color_attachment_load_enable) {
      initial_colors = LoadArray<std::uint8_t>(pool_, state.color_attachment_load);
      if (state.color_attachment_load_bytes != framebuffer_bytes * render_target_count ||
          initial_colors.size() != state.color_attachment_load_bytes)
        throw std::runtime_error("PBE color attachment LOAD byte count mismatch");
    }
    std::vector<std::vector<std::uint8_t>> framebuffers(render_target_count);
    for (std::uint32_t target = 0; target < render_target_count; ++target) {
      const auto packed_format = ColorAttachmentPackedUnorm(state, target);
      std::vector<std::uint8_t> &attachment = framebuffers[target];
      if (state.color_attachment_load_enable != 0) {
        attachment.assign(initial_colors.begin() + target * framebuffer_bytes,
                          initial_colors.begin() + (target + 1) * framebuffer_bytes);
        continue;
      }
      attachment.assign(static_cast<std::size_t>(framebuffer_bytes), 0);
      if (packed_format != PackedUnormFormat::kNone) {
        std::array<float, 4> clear{};
        std::copy_n(state.raster_state.clear_color, 4, clear.begin());
        const std::uint32_t word =
            PackUnormColor(clear, packed_format);
        for (std::size_t pixel = 0; pixel < stored_samples; ++pixel)
          std::memcpy(attachment.data() + pixel * 4U, &word, sizeof(word));
      } else if (state.color_attachment_float32) {
        for (std::size_t pixel = 0; pixel < stored_samples; ++pixel) {
          std::memcpy(attachment.data() + pixel * bytes_per_pixel,
                      state.raster_state.clear_color, bytes_per_pixel);
        }
      } else if (state.color_attachment_raw_dwords != 0) {
        // An integer attachment clears to the raw value, not a colour, and to
        // one such value per channel it stores.
        const std::size_t channels = state.color_attachment_raw_dwords;
        std::array<std::uint32_t, 4> raw{};
        for (std::size_t channel = 0; channel < channels; ++channel) {
          std::memcpy(&raw[channel], &state.raster_state.clear_color[channel],
                      sizeof(raw[channel]));
        }
        for (std::size_t pixel = 0; pixel < stored_samples; ++pixel) {
          std::memcpy(attachment.data() + pixel * bytes_per_pixel, raw.data(),
                      channels * sizeof(std::uint32_t));
        }
      } else {
        for (std::size_t pixel = 0; pixel < stored_samples; ++pixel) {
          for (std::size_t component = 0; component < 4; ++component) {
            attachment[pixel * bytes_per_pixel + component] =
                FiniteStateToUnorm8(state.raster_state.clear_color[component]);
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
            sample_count, BitsFloat(output.pixel_output[3]), output.x, output.y,
            state.raster_state.alpha_to_coverage_dither != 0,
            state.raster_state.multisample_enable != 0);
      }
      if (state.raster_state.alpha_to_one) {
        for (std::uint32_t target = 0; target < render_target_count; ++target)
          if ((state.fragment_output_mask[target] & output.written_mask[target] &
               8U) != 0)
            output.pixel_output[target * 4 + 3] = UINT32_C(0x3f800000);
      }
      for (std::uint32_t sample = 0; sample < sample_count; ++sample) {
      if ((coverage & (1U << sample)) == 0)
        continue;
      const std::size_t stored_index = pixel_index * sample_count + sample;
      if (late_depth_stencil) {
        ++late_tested_samples;
        const float incoming_depth = state.raster_state.shader_writes_depth
            ? output.depth : invocation.sample_depth[sample];
        if (!TestLateDepthStencil(state, invocation, incoming_depth, stored_index,
                                  late_depth, late_stencil))
          continue;
      }
      if (written_map[stored_index] != 0) {
        if (!state.raster_state.blend.enable && !late_depth_stencil &&
            !state.raster_state.shader_writes_memory &&
            !state.raster_state.shader_may_discard && state.fragment_early_hsr_safe)
          throw std::runtime_error("PBE attempted to shade one opaque owner twice");
        if (output.submit_ordinal < last_submit_ordinal[stored_index])
          throw std::runtime_error("PBE blended fragments lost API order");
      }
      ++written_map[stored_index];
      last_submit_ordinal[stored_index] = output.submit_ordinal;
      const std::size_t byte_offset = stored_index * bytes_per_pixel;
      for (std::uint32_t target = 0; target < render_target_count; ++target) {
      // A native FS may genuinely have no output for this attachment.
      // Preserve its pixels; no default color export or blend is fabricated.
      if (explicit_output_masks && state.fragment_output_mask[target] == 0)
        continue;
      const auto packed_format = ColorAttachmentPackedUnorm(state, target);
      const bool packed_unorm = packed_format != PackedUnormFormat::kNone;
      std::vector<std::uint8_t> &framebuffer = framebuffers[target];
      if (state.color_attachment_float32 || packed_unorm) {
        std::array<float, 4> source{};
        std::array<float, 4> destination{};
        std::memcpy(source.data(), &output.pixel_output[target * 4],
                    sizeof(source));
        std::uint32_t packed_destination = 0;
        if (packed_unorm) {
          for (float &component : source)
            component = ClampShaderUnorm(component);
          std::memcpy(&packed_destination, framebuffer.data() + byte_offset, 4);
          destination = UnpackUnormColor(packed_destination, packed_format);
        } else {
          std::memcpy(destination.data(), framebuffer.data() + byte_offset,
                      sizeof(destination));
        }
        std::array<float, 4> result = source;
        if (state.raster_state.blend.enable) {
          const BlendState &blend = state.raster_state.blend;
          std::array<float, 4> constant{};
          for (std::size_t component = 0; component < 4; ++component)
            constant[component] = std::clamp(
                BitsFloat(blend.constant_color_bits[component]), 0.0F, 1.0F);
          for (std::size_t component = 0; component < 4; ++component) {
            const BlendFactor source_factor = component == 3
                ? blend.source_alpha_factor : blend.source_rgb_factor;
            const BlendFactor destination_factor = component == 3
                ? blend.destination_alpha_factor : blend.destination_rgb_factor;
            const BlendEquation equation = component == 3
                ? blend.alpha_equation : blend.rgb_equation;
            result[component] = BlendEquationFloat(
                equation, source[component], destination[component],
                FactorToFloat(source_factor, source, destination, constant,
                              component),
                FactorToFloat(destination_factor, source, destination, constant,
                              component));
          }
        }
        if (packed_unorm) {
          // Quantize after every fragment, before any later destination LOAD.
          // Masked channels retain their original packed bits, not a recode.
          const std::uint32_t word = PackUnormColor(result,
              packed_format, packed_destination,
              state.raster_state.color_mask);
          std::memcpy(framebuffer.data() + byte_offset, &word, sizeof(word));
          continue;
        }
        for (std::size_t component = 0; component < 4; ++component) {
          if ((state.raster_state.color_mask & (1U << component)) != 0) {
            std::memcpy(framebuffer.data() + byte_offset + component * 4U,
                        &result[component], sizeof(float));
          }
        }
        continue;
      }
      if (state.color_attachment_raw_dwords != 0) {
        /*
         * A 32-bit integer attachment stores the shader's PIXOUT lanes
         * verbatim, one dword per channel it holds.  UNORM8 conversion would
         * quantise a value that was never a colour, and GLES forbids blending
         * on an integer format, so this path writes and returns.
         */
        const std::size_t channels = state.color_attachment_raw_dwords;
        for (std::size_t channel = 0; channel < channels; ++channel) {
          if ((state.raster_state.color_mask & (1U << channel)) == 0)
            continue;
          const std::uint32_t raw = output.pixel_output[target * 4 + channel];
          std::memcpy(framebuffer.data() + byte_offset +
                          channel * sizeof(raw),
                      &raw, sizeof(raw));
        }
        continue;
      }
      if (state.color_is_srgb) {
        /*
         * GLES blends an sRGB colour buffer in linear space with no toggle.
         * The shader source is already linear; decode the stored sRGB
         * destination to linear, blend (or pass the source through) there, and
         * re-encode on write.  Alpha never passes through the sRGB transfer.
         */
        std::array<float, 4> source_linear{};
        for (std::size_t component = 0; component < 4; ++component) {
          source_linear[component] = ClampShaderUnorm(
              BitsFloat(output.pixel_output[target * 4 + component]));
        }
        std::array<float, 4> result_linear = source_linear;
        if (state.raster_state.blend.enable) {
          const BlendState &blend = state.raster_state.blend;
          std::array<float, 4> dest_linear{};
          for (std::size_t component = 0; component < 3; ++component) {
            dest_linear[component] =
                SrgbChannelToLinear(framebuffer[byte_offset + component]);
          }
          dest_linear[3] =
              static_cast<float>(framebuffer[byte_offset + 3]) / 255.0F;
          std::array<float, 4> constant_linear{};
          for (std::size_t component = 0; component < 4; ++component) {
            constant_linear[component] = std::clamp(
                BitsFloat(blend.constant_color_bits[component]), 0.0F, 1.0F);
          }
          for (std::size_t component = 0; component < 4; ++component) {
            const BlendFactor source_factor =
                component == 3 ? blend.source_alpha_factor
                               : blend.source_rgb_factor;
            const BlendFactor destination_factor =
                component == 3 ? blend.destination_alpha_factor
                               : blend.destination_rgb_factor;
            const BlendEquation equation = component == 3
                                               ? blend.alpha_equation
                                               : blend.rgb_equation;
            const float sf = FactorToFloat(source_factor, source_linear,
                                           dest_linear, constant_linear,
                                           component);
            const float df = FactorToFloat(destination_factor, source_linear,
                                           dest_linear, constant_linear,
                                           component);
            result_linear[component] = BlendEquationFloat(
                equation, source_linear[component], dest_linear[component], sf,
                df);
          }
        }
        for (std::size_t component = 0; component < 4; ++component) {
          if ((state.raster_state.color_mask & (1U << component)) == 0)
            continue;
          framebuffer[byte_offset + component] =
              component == 3
                  ? FloatValueToUnorm8(result_linear[3])
                  : LinearChannelToSrgbUnorm8(
                        ClampShaderUnorm(result_linear[component]));
        }
        continue;
      }
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
        std::array<std::uint8_t, 4> constant_color{};
        for (std::size_t component = 0; component < 4; ++component) {
          constant_color[component] =
              FiniteStateToUnorm8(BitsFloat(blend.constant_color_bits[component]));
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
          const std::uint8_t sf = FactorToUnorm8(source_factor, source, destination_color, constant_color, component);
          const std::uint8_t df = FactorToUnorm8(destination_factor, source, destination_color, constant_color, component);

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
    }
    const std::uint64_t pixels_touched = static_cast<std::uint64_t>(
        std::count_if(written_map.begin(), written_map.end(),
                      [](std::uint32_t writes) { return writes != 0; }));
    if ((!state.raster_state.blend.enable && !late_depth_stencil && sample_count == 1 &&
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
    std::uint64_t sample_colors = 0;
    std::uint32_t color_output_targets = render_target_count;
    if (explicit_output_masks) {
      color_output_targets = 0;
      for (std::uint32_t target = 0; target < render_target_count; ++target)
        color_output_targets += state.fragment_output_mask[target] != 0;
    }
    for (const std::uint32_t writes : written_map)
      sample_colors += static_cast<std::uint64_t>(writes) * color_output_targets;
    const std::uint32_t stored_channel_mask =
        state.color_attachment_raw_dwords != 0
            ? (1U << state.color_attachment_raw_dwords) - 1U : 0x0fU;
    state.counters.pbe_pixels_written = stored_samples * render_target_count;
    state.counters.pbe_fragment_writes =
        (state.raster_state.color_mask & stored_channel_mask) != 0
            ? sample_colors : 0;
    // Integer attachments bypass the blend equation even if API blend state
    // is enabled. A masked floating/UNORM output still runs that equation in
    // this model, but does not produce a color write when all lanes are off.
    const std::uint64_t blended_colors =
        state.raster_state.blend.enable &&
                state.color_attachment_raw_dwords == 0
            ? sample_colors : 0;
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
