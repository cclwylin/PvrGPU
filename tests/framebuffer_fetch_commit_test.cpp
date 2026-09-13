// Coherent framebuffer-fetch publication tests.  The shadow must expose only
// color values that the real PBE would commit after discard, late depth and
// fixed-function blending -- never an earlier raw PIXOUT value.
#include "fragment/framebuffer_fetch_commit.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace pvrgpu::stub;

void Check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error("framebuffer-fetch commit test failed: " +
                             message);
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

PipelineState BaseState() {
  PipelineState state;
  state.width = 1;
  state.height = 1;
  state.attachment_layers = 1;
  state.render_target_count = 1;
  state.raster_state.sample_count = 1;
  state.fragment_output_mask[0] = 0x0f;
  return state;
}

FragmentInvocation Invocation(std::uint64_t ordinal, float depth = 0.5F) {
  FragmentInvocation invocation;
  invocation.x = 0;
  invocation.y = 0;
  invocation.primitive_id = static_cast<std::uint32_t>(ordinal);
  invocation.parameter_index = static_cast<std::uint32_t>(ordinal);
  invocation.submit_ordinal = ordinal;
  invocation.sample_mask = 1;
  invocation.sample_depth[0] = depth;
  invocation.front_facing = 1;
  return invocation;
}

FragmentOutput Output(const FragmentInvocation &invocation,
                      const std::array<float, 4> &color) {
  FragmentOutput output;
  output.x = invocation.x;
  output.y = invocation.y;
  output.primitive_id = invocation.primitive_id;
  output.parameter_index = invocation.parameter_index;
  output.submit_ordinal = invocation.submit_ordinal;
  output.render_target_count = 1;
  output.written_mask[0] = 0x0f;
  for (std::size_t component = 0; component < color.size(); ++component)
    output.pixel_output[component] = FloatBits(color[component]);
  return output;
}

FragmentOutput IntegerOutput(
    const std::array<std::uint32_t, 4> &color) {
  FragmentOutput output;
  output.render_target_count = 1;
  output.written_mask[0] = 0x0f;
  std::copy(color.begin(), color.end(), output.pixel_output);
  return output;
}

FramebufferFetchCommitStorage Storage(
    const PipelineState &state, std::array<std::uint8_t, 4> initial) {
  FramebufferFetchCommitStorage storage;
  storage.enabled = true;
  storage.render_target_count = 1;
  storage.stored_samples = state.raster_state.sample_count;
  storage.target_offsets[1] = initial.size() * storage.stored_samples;
  for (std::size_t sample = 0; sample < storage.stored_samples; ++sample)
    storage.bytes.insert(storage.bytes.end(), initial.begin(), initial.end());
  storage.committed.assign(storage.stored_samples, 0);
  storage.last_submit_ordinal.assign(storage.stored_samples, 0);
  storage.late_depth_stencil =
      RasterRequiresLateDepthStencil(state.raster_state);
  if (storage.late_depth_stencil)
    storage.late_depth.assign(storage.stored_samples, FloatBits(1.0F));
  return storage;
}

FramebufferFetchCommitStorage CanonicalStorage(
    const PipelineState &state, const std::array<float, 4> &initial) {
  FramebufferFetchCommitStorage storage;
  storage.enabled = true;
  storage.render_target_count = 1;
  storage.stored_samples = state.raster_state.sample_count;
  storage.target_offsets[1] = sizeof(initial) * storage.stored_samples;
  storage.bytes.resize(storage.target_offsets[1]);
  for (std::size_t sample = 0; sample < storage.stored_samples; ++sample)
    std::memcpy(storage.bytes.data() + sample * sizeof(initial),
                initial.data(), sizeof(initial));
  storage.committed.assign(storage.stored_samples, 0);
  storage.last_submit_ordinal.assign(storage.stored_samples, 0);
  return storage;
}

std::array<float, 4> CanonicalPixel(
    const FramebufferFetchCommitStorage &storage) {
  std::array<float, 4> color{};
  Check(storage.bytes.size() == sizeof(color),
        "canonical pixel did not retain RGBA32F storage");
  std::memcpy(color.data(), storage.bytes.data(), sizeof(color));
  return color;
}

void SetCodec(PipelineState &state, const char *format) {
  Check(ColorAttachmentCodecFromName(format,
                                     &state.color_attachment_codec),
        "test format has no canonical codec");
  Check(ColorAttachmentCodecIsCanonical(state.color_attachment_codec),
        "test format unexpectedly uses a legacy codec");
  const std::size_t expected_bytes =
      ColorAttachmentCodecUsesFloat64Storage(state.color_attachment_codec)
          ? 4U * sizeof(double)
          : 4U * sizeof(float);
  Check(ColorAttachmentBytesPerPixel(state, 0) == expected_bytes,
        "canonical codec selected the wrong transport width");
  (void)ValidateColorAttachmentFormats(state);
}

void SetIntegerCodec(PipelineState &state, const char *format) {
  Check(ColorAttachmentCodecFromName(format,
                                     &state.color_attachment_codec),
        "test integer format has no codec");
  Check(ColorAttachmentCodecIsInteger(state.color_attachment_codec),
        "test integer format did not select an integer codec");
  state.color_attachment_raw_dwords =
      state.color_attachment_codec.component_mask & 0x8
          ? 4
          : state.color_attachment_codec.component_mask & 0x2 ? 2 : 1;
  (void)ValidateColorAttachmentFormats(state);
}

std::uint32_t SignedBits(std::int32_t value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::uint32_t Unorm32Code(double canonical) {
  constexpr double maximum = 4294967295.0;
  return static_cast<std::uint32_t>(
      PbeRoundTiesToEven(std::clamp(canonical, 0.0, 1.0) * maximum));
}

FramebufferFetchCommitStorage Unorm32Storage(
    const PipelineState &state,
    const std::array<std::uint32_t, 4> &native) {
  std::array<double, 4> canonical{};
  for (std::size_t component = 0; component < canonical.size(); ++component)
    canonical[component] =
        static_cast<double>(native[component]) / 4294967295.0;
  FramebufferFetchCommitStorage storage;
  storage.enabled = true;
  storage.render_target_count = 1;
  storage.stored_samples = state.raster_state.sample_count;
  storage.target_offsets[1] = sizeof(canonical) * storage.stored_samples;
  storage.bytes.resize(storage.target_offsets[1]);
  for (std::size_t sample = 0; sample < storage.stored_samples; ++sample)
    std::memcpy(storage.bytes.data() + sample * sizeof(canonical),
                canonical.data(), sizeof(canonical));
  storage.committed.assign(storage.stored_samples, 0);
  storage.last_submit_ordinal.assign(storage.stored_samples, 0);
  return storage;
}

void TestFixedBlendIsPublishedInsteadOfRawPixout() {
  PipelineState state = BaseState();
  state.raster_state.blend.enable = 1;
  state.raster_state.blend.rgb_equation = BlendEquation::kAdd;
  state.raster_state.blend.alpha_equation = BlendEquation::kAdd;
  state.raster_state.blend.source_rgb_factor = BlendFactor::kSourceAlpha;
  state.raster_state.blend.destination_rgb_factor =
      BlendFactor::kOneMinusSourceAlpha;
  state.raster_state.blend.source_alpha_factor = BlendFactor::kSourceAlpha;
  state.raster_state.blend.destination_alpha_factor =
      BlendFactor::kOneMinusSourceAlpha;
  auto storage = Storage(state, {0, 0, 255, 255});
  const FragmentInvocation invocation = Invocation(1);
  const FragmentOutput output = Output(invocation, {1.0F, 0.0F, 0.0F, 0.5F});

  Check(CommitFramebufferFetchFragment(storage, state, invocation, output) ==
            1,
        "blended fragment did not commit");
  Check(storage.bytes == std::vector<std::uint8_t>({128, 0, 127, 191}),
        "feedback exposed raw PIXOUT instead of quantized blended color");
}

void TestDiscardDoesNotPublishColor() {
  PipelineState state = BaseState();
  state.raster_state.shader_may_discard = 1;
  auto storage = Storage(state, {3, 5, 7, 11});
  const FragmentInvocation invocation = Invocation(1);
  FragmentOutput output = Output(invocation, {1.0F, 0.0F, 0.0F, 1.0F});
  output.discarded = 1;

  Check(CommitFramebufferFetchFragment(storage, state, invocation, output) ==
            0,
        "discarded fragment reported a commit");
  Check(storage.bytes == std::vector<std::uint8_t>({3, 5, 7, 11}) &&
            storage.committed[0] == 0,
        "discarded fragment became visible to framebuffer fetch");
}

void TestFailedLateDepthDoesNotPublishColor() {
  PipelineState state = BaseState();
  state.raster_state.shader_writes_depth = 1;
  state.raster_state.depth.test_enable = 1;
  state.raster_state.depth.write_enable = 1;
  state.raster_state.depth.compare_op = DepthCompareOp::kLess;
  auto storage = Storage(state, {13, 17, 19, 23});
  storage.late_depth[0] = FloatBits(0.25F);

  const FragmentInvocation rejected_invocation = Invocation(1);
  FragmentOutput rejected =
      Output(rejected_invocation, {1.0F, 0.0F, 0.0F, 1.0F});
  rejected.depth_written = 1;
  rejected.depth = 0.75F;
  Check(CommitFramebufferFetchFragment(storage, state, rejected_invocation,
                                       rejected) == 0,
        "depth-rejected fragment reported a commit");
  Check(storage.bytes == std::vector<std::uint8_t>({13, 17, 19, 23}) &&
            storage.late_depth[0] == FloatBits(0.25F) &&
            storage.committed[0] == 0,
        "depth-rejected fragment changed fetch-visible state");

  const FragmentInvocation accepted_invocation = Invocation(2);
  FragmentOutput accepted =
      Output(accepted_invocation, {0.0F, 1.0F, 0.0F, 1.0F});
  accepted.depth_written = 1;
  accepted.depth = 0.125F;
  Check(CommitFramebufferFetchFragment(storage, state, accepted_invocation,
                                       accepted) == 1,
        "depth-passing fragment did not commit");
  Check(storage.bytes == std::vector<std::uint8_t>({0, 255, 0, 255}) &&
            storage.late_depth[0] == FloatBits(0.125F) &&
            storage.committed[0] == 1,
        "depth-passing fragment did not publish color and depth together");
}

void TestSnormAndWideUnormKeepNativePrecision() {
  {
    PipelineState state = BaseState();
    SetCodec(state, "PIPE_FORMAT_R8_SNORM");
    auto storage = CanonicalStorage(state, {0.0F, 0.0F, 0.0F, 1.0F});
    const auto invocation = Invocation(1);
    const auto output = Output(invocation, {-0.5F, 0.75F, 0.25F, 0.125F});
    Check(CommitFramebufferFetchFragment(storage, state, invocation, output) ==
              1,
          "R8_SNORM fragment did not commit");
    const auto pixel = CanonicalPixel(storage);
    Check(pixel[0] == -64.0F / 127.0F,
          "R8_SNORM negative value was clamped or rounded incorrectly");
    Check(pixel[1] == 0.0F && pixel[2] == 0.0F && pixel[3] == 1.0F,
          "R8_SNORM fetch did not expose native missing-channel defaults");
  }
  {
    PipelineState state = BaseState();
    SetCodec(state, "PIPE_FORMAT_R16_UNORM");
    auto storage = CanonicalStorage(state, {0.0F, 0.0F, 0.0F, 1.0F});
    const auto invocation = Invocation(1);
    constexpr float value = 0.50001F;
    const auto output = Output(invocation, {value, 0.0F, 0.0F, 1.0F});
    Check(CommitFramebufferFetchFragment(storage, state, invocation, output) ==
              1,
          "R16_UNORM fragment did not commit");
    const auto pixel = CanonicalPixel(storage);
    const float expected = static_cast<float>(
        std::floor(static_cast<double>(value) * 65535.0 + 0.5) / 65535.0);
    Check(pixel[0] == expected && pixel[0] != 128.0F / 255.0F,
          "R16_UNORM was reduced to legacy eight-bit precision");
  }
}

void TestCanonicalClearUsesTheSameNativeCodec() {
  PipelineState state = BaseState();
  SetCodec(state, "PIPE_FORMAT_R8_SNORM");
  const auto snorm = PbeCanonicalizeColor(
      state.color_attachment_codec, {-0.5F, 0.75F, 0.25F, 0.125F});
  Check(snorm[0] == -64.0F / 127.0F && snorm[1] == 0.0F &&
            snorm[2] == 0.0F && snorm[3] == 1.0F,
        "R8_SNORM clear did not use native quantization/defaults");

  state = BaseState();
  SetCodec(state, "PIPE_FORMAT_R16_FLOAT");
  const auto half = PbeCanonicalizeColor(
      state.color_attachment_codec, {1.0006F, 7.0F, 8.0F, 9.0F});
  Check(half[0] == 1.0009765625F && half[1] == 0.0F &&
            half[2] == 0.0F && half[3] == 1.0F,
        "R16_FLOAT clear did not use native quantization/defaults");
}

void TestUnorm32ExactTransportAndMaskedCommit() {
  PipelineState state = BaseState();
  SetCodec(state, "PIPE_FORMAT_R32G32B32A32_UNORM");
  Check(ColorAttachmentBytesPerPixel(state, 0) == 32,
        "RGBA32_UNORM did not select exact RGBA64F transport");

  const std::array<std::uint32_t, 4> native = {
      UINT32_C(0x01000001), UINT32_C(0x80000001),
      UINT32_C(0xfffffffe), UINT32_C(0x12345679)};
  auto storage = Unorm32Storage(state, native);
  state.raster_state.color_mask = 0x1;
  const auto invocation = Invocation(1);
  Check(CommitFramebufferFetchFragment(
            storage, state, invocation,
            Output(invocation, {0.25F, 0.75F, 0.5F, 1.0F})) == 1,
        "RGBA32_UNORM masked fragment did not commit");

  std::array<double, 4> stored{};
  Check(storage.bytes.size() == sizeof(stored),
        "RGBA32_UNORM storage was not 32 bytes");
  std::memcpy(stored.data(), storage.bytes.data(), sizeof(stored));
  Check(Unorm32Code(stored[0]) == UINT32_C(0x40000000),
        "RGBA32_UNORM written component quantized incorrectly");
  for (std::size_t component = 1; component < native.size(); ++component)
    Check(Unorm32Code(stored[component]) == native[component],
          "RGBA32_UNORM masked component lost a native code");

  const auto shader = PbeReadCanonicalColorForShader(
      state.color_attachment_codec, storage.bytes.data());
  for (std::size_t component = 0; component < shader.size(); ++component)
    Check(shader[component] == static_cast<float>(stored[component]),
          "RGBA32_UNORM USC input did not narrow canonical double to float");

  Check(Unorm32Code(PbeCanonicalComponentFloat64(
            state.color_attachment_codec, 0, 2.5 / 4294967295.0)) == 2,
        "RGBA32_UNORM even midpoint did not round down to even");
  Check(Unorm32Code(PbeCanonicalComponentFloat64(
            state.color_attachment_codec, 0, 3.5 / 4294967295.0)) == 4,
        "RGBA32_UNORM odd midpoint did not round up to even");
}

void TestIntegerNativeCommitAndFetchDefaults() {
  {
    PipelineState state = BaseState();
    SetIntegerCodec(state, "PIPE_FORMAT_R8_UINT");
    std::uint32_t stored = 17;
    CommitPbeColorSample(state, IntegerOutput({256, 0, 0, 0}), 0,
                         reinterpret_cast<std::uint8_t *>(&stored));
    Check(stored == 255, "R8_UINT did not clamp 256 to 255");
  }
  {
    PipelineState state = BaseState();
    SetIntegerCodec(state, "PIPE_FORMAT_R8_SINT");
    std::uint32_t stored = 0;
    CommitPbeColorSample(
        state, IntegerOutput({SignedBits(-129), 0, 0, 0}), 0,
        reinterpret_cast<std::uint8_t *>(&stored));
    Check(stored == SignedBits(-128), "R8_SINT negative overflow was not clamped");
    CommitPbeColorSample(
        state, IntegerOutput({SignedBits(128), 0, 0, 0}), 0,
        reinterpret_cast<std::uint8_t *>(&stored));
    Check(stored == SignedBits(127), "R8_SINT positive overflow was not clamped");
  }
  {
    PipelineState state = BaseState();
    SetIntegerCodec(state, "PIPE_FORMAT_R10G10B10A2_UINT");
    std::array<std::uint32_t, 4> stored{};
    CommitPbeColorSample(state, IntegerOutput({1024, 5000, 42, 7}), 0,
                         reinterpret_cast<std::uint8_t *>(stored.data()));
    Check(stored == std::array<std::uint32_t, 4>{1023, 1023, 42, 3},
          "RGB10_A2_UINT did not apply native component ranges");
    state.raster_state.alpha_to_one = 1;
    CommitPbeColorSample(state, IntegerOutput({1, 2, 3, 0}), 0,
                         reinterpret_cast<std::uint8_t *>(stored.data()));
    Check(stored[3] == 3,
          "A2_UINT alpha-to-one did not saturate to native maximum");
  }
  {
    PipelineState state = BaseState();
    SetIntegerCodec(state, "PIPE_FORMAT_R8G8B8A8_UINT");
    state.raster_state.color_mask = 0x5;
    std::array<std::uint32_t, 4> stored = {1, 77, 3, 99};
    CommitPbeColorSample(state, IntegerOutput({300, 200, 400, 250}), 0,
                         reinterpret_cast<std::uint8_t *>(stored.data()));
    Check(stored == std::array<std::uint32_t, 4>{255, 77, 255, 99},
          "integer color mask did not preserve untouched native channels");
    state.raster_state.color_mask = 0x0f;
    state.raster_state.blend.enable = 1;
    CommitPbeColorSample(state, IntegerOutput({300, 2, 3, 4}), 0,
                         reinterpret_cast<std::uint8_t *>(stored.data()));
    Check(stored == std::array<std::uint32_t, 4>{255, 2, 3, 4},
          "integer attachment did not bypass blending with native conversion");
  }

  const std::array<std::uint32_t, 2> rg = {9, 10};
  const auto r_default = PbeReadRawIntegerColorForShader(
      1, reinterpret_cast<const std::uint8_t *>(rg.data()));
  const auto rg_default = PbeReadRawIntegerColorForShader(
      2, reinterpret_cast<const std::uint8_t *>(rg.data()));
  Check(r_default == std::array<std::uint32_t, 4>{9, 0, 0, 1},
        "R integer framebuffer fetch defaults are not (R,0,0,1)");
  Check(rg_default == std::array<std::uint32_t, 4>{9, 10, 0, 1},
        "RG integer framebuffer fetch defaults are not (R,G,0,1)");
}

void TestNormalizedMidpointsRoundToEven() {
  ColorAttachmentCodec codec{};
  Check(ColorAttachmentCodecFromName("PIPE_FORMAT_R8_UNORM", &codec),
        "R8_UNORM codec missing");
  Check(PbeCanonicalComponent(codec, 0, 2.5F / 255.0F) == 2.0F / 255.0F,
        "R8_UNORM midpoint did not round to even");
  Check(ColorAttachmentCodecFromName("PIPE_FORMAT_R8_SNORM", &codec),
        "R8_SNORM codec missing");
  Check(PbeCanonicalComponent(codec, 0, 2.5F / 127.0F) == 2.0F / 127.0F,
        "R8_SNORM positive midpoint did not round to even");
  Check(PbeCanonicalComponent(codec, 0, -2.5F / 127.0F) ==
            -2.0F / 127.0F,
        "R8_SNORM negative midpoint did not round to even");
}

void TestMissingComponentsAndFloat16Quantization() {
  for (const char *format : {"PIPE_FORMAT_R8_UNORM",
                             "PIPE_FORMAT_R8G8_UNORM",
                             "PIPE_FORMAT_R5G6B5_UNORM"}) {
    PipelineState state = BaseState();
    SetCodec(state, format);
    auto storage = CanonicalStorage(state, {0.0F, 0.0F, 0.0F, 1.0F});
    const auto invocation = Invocation(1);
    const auto output = Output(invocation, {0.25F, 0.5F, 0.75F, 0.125F});
    Check(CommitFramebufferFetchFragment(storage, state, invocation, output) ==
              1,
          std::string(format) + " fragment did not commit");
    const auto pixel = CanonicalPixel(storage);
    const auto mask = state.color_attachment_codec.component_mask;
    if (!(mask & 0x2))
      Check(pixel[1] == 0.0F, std::string(format) + " default G is not zero");
    if (!(mask & 0x4))
      Check(pixel[2] == 0.0F, std::string(format) + " default B is not zero");
    Check(pixel[3] == 1.0F,
          std::string(format) + " default alpha is not one");
  }

  PipelineState state = BaseState();
  SetCodec(state, "PIPE_FORMAT_R16_FLOAT");
  auto storage = CanonicalStorage(state, {0.0F, 0.0F, 0.0F, 1.0F});
  const auto invocation = Invocation(1);
  const auto output = Output(invocation, {1.0006F, 7.0F, 8.0F, 9.0F});
  Check(CommitFramebufferFetchFragment(storage, state, invocation, output) ==
            1,
        "R16_FLOAT fragment did not commit");
  const auto pixel = CanonicalPixel(storage);
  Check(pixel[0] == 1.0009765625F,
        "R16_FLOAT fetch did not observe half-float quantization");
  Check(pixel[1] == 0.0F && pixel[2] == 0.0F && pixel[3] == 1.0F,
        "R16_FLOAT fetch did not expose native missing-channel defaults");

  state = BaseState();
  SetCodec(state, "PIPE_FORMAT_R11G11B10_FLOAT");
  storage = CanonicalStorage(state, {0.0F, 0.0F, 0.0F, 1.0F});
  const auto packed_float_invocation = Invocation(2);
  const auto packed_float_output =
      Output(packed_float_invocation, {1.01F, 1.01F, 1.01F, 0.0F});
  Check(CommitFramebufferFetchFragment(storage, state,
                                       packed_float_invocation,
                                       packed_float_output) == 1,
        "R11G11B10_FLOAT fragment did not commit");
  const auto packed_float_pixel = CanonicalPixel(storage);
  Check(packed_float_pixel[0] == 1.015625F &&
            packed_float_pixel[1] == 1.015625F &&
            packed_float_pixel[2] == 1.0F &&
            packed_float_pixel[3] == 1.0F,
        "R11G11B10_FLOAT native pack/unpack quantization is incorrect");

  state.raster_state.blend.enable = 1;
  state.raster_state.blend.rgb_equation = BlendEquation::kAdd;
  state.raster_state.blend.alpha_equation = BlendEquation::kAdd;
  state.raster_state.blend.source_rgb_factor = BlendFactor::kOne;
  state.raster_state.blend.destination_rgb_factor = BlendFactor::kOne;
  state.raster_state.blend.source_alpha_factor = BlendFactor::kOne;
  state.raster_state.blend.destination_alpha_factor = BlendFactor::kOne;
  storage = CanonicalStorage(state, {1.0F, 1.0F, 1.0F, 1.0F});
  const auto negative_invocation = Invocation(3);
  const auto negative_output =
      Output(negative_invocation, {-0.5F, -2.0F, -3.0F, 0.0F});
  Check(CommitFramebufferFetchFragment(storage, state, negative_invocation,
                                       negative_output) == 1,
        "R11G11B10_FLOAT negative-source blend did not commit");
  const auto blended = CanonicalPixel(storage);
  Check(blended[0] == 0.5F && blended[1] == 0.0F && blended[2] == 0.0F,
        "R11G11B10_FLOAT clamped source before blending");

  state.raster_state.blend.enable = 0;
  storage = CanonicalStorage(state, {1.0F, 1.0F, 1.0F, 1.0F});
  const auto direct_invocation = Invocation(4);
  Check(CommitFramebufferFetchFragment(
            storage, state, direct_invocation,
            Output(direct_invocation, {-0.5F, -2.0F, -3.0F, 0.0F})) == 1,
        "R11G11B10_FLOAT negative direct write did not commit");
  const auto direct = CanonicalPixel(storage);
  Check(direct[0] == 0.0F && direct[1] == 0.0F && direct[2] == 0.0F,
        "R11G11B10_FLOAT direct encode retained a negative value");
}

} // namespace

int main() {
  try {
    TestFixedBlendIsPublishedInsteadOfRawPixout();
    TestDiscardDoesNotPublishColor();
    TestFailedLateDepthDoesNotPublishColor();
    TestSnormAndWideUnormKeepNativePrecision();
    TestCanonicalClearUsesTheSameNativeCodec();
    TestUnorm32ExactTransportAndMaskedCommit();
    TestIntegerNativeCommitAndFetchDefaults();
    TestNormalizedMidpointsRoundToEven();
    TestMissingComponentsAndFloat16Quantization();
    std::cout << "framebuffer-fetch commit tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
