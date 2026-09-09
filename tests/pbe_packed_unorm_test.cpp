// Exercise the real event-driven PBE with four-byte RGB/BGR10_A2 storage.
// Expected words are assembled from independently chosen integer channel
// codes. No production pack/unpack helper is used as the test oracle.
#include "common/functional_types.h"
#include "common/pipeline_state.h"
#include "common/tessellation_state.h"
#include "fragment/pbe.h"

#include <systemc>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace pvrgpu::stub;
using Codes = std::array<std::uint32_t, 4>;
using Color = std::array<float, 4>;
unsigned checks = 0;
unsigned cases = 0;

void Check(bool value, const std::string &message) {
  ++checks;
  if (!value)
    throw std::runtime_error("PBE packed UNORM: " + message);
}

std::uint32_t Bits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

// Fixed storage definitions, independent of PackedUnormShift/PackUnormColor.
std::uint32_t Word(const Codes &codes, PackedUnormFormat format) {
  if (format == PackedUnormFormat::kNone) {
    Check(std::all_of(codes.begin(), codes.end(), [](auto code) { return code < 256; }),
          "RGBA8 oracle channel range");
    return codes[0] | (codes[1] << 8) | (codes[2] << 16) | (codes[3] << 24);
  }
  Check(codes[0] < 1024 && codes[1] < 1024 && codes[2] < 1024 && codes[3] < 4,
        "oracle channel range");
  const auto red = format == PackedUnormFormat::kBgr10A2 ? codes[2] : codes[0];
  const auto blue = format == PackedUnormFormat::kBgr10A2 ? codes[0] : codes[2];
  return red | (codes[1] << 10) | (blue << 20) | (codes[3] << 30);
}

Color Normalized(const Codes &codes, PackedUnormFormat format = PackedUnormFormat::kRgb10A2) {
  if (format == PackedUnormFormat::kNone)
    return {codes[0] / 255.0F, codes[1] / 255.0F, codes[2] / 255.0F, codes[3] / 255.0F};
  return {codes[0] / 1023.0F, codes[1] / 1023.0F,
          codes[2] / 1023.0F, codes[3] / 3.0F};
}

std::uint32_t ChannelMask(std::uint8_t logical, PackedUnormFormat format) {
  Codes mask{};
  for (unsigned c = 0; c < 4; ++c)
    mask[c] = (logical & (1U << c)) ? (format == PackedUnormFormat::kNone ? 255 : c == 3 ? 3 : 1023) : 0;
  return Word(mask, format);
}

struct Scenario {
  PipelineState state;
  std::vector<FragmentInvocation> invocations;
  std::vector<FragmentOutput> outputs;
  std::vector<std::uint32_t> initial;
  std::vector<std::uint32_t> expected;
  std::uint64_t covered_colors = 0;
  std::string name;

  std::size_t TargetWords() const {
    return state.width * state.height * state.attachment_layers *
           state.raster_state.sample_count;
  }
};

Scenario NewScenario(PackedUnormFormat format, const std::string &name,
                     unsigned width = 2, unsigned height = 2,
                     unsigned layers = 1, unsigned samples = 1,
                     unsigned targets = 1, bool load = true) {
  Scenario scenario;
  scenario.name = name;
  auto &state = scenario.state;
  state.width = width;
  state.height = height;
  state.attachment_layers = layers;
  state.raster_state.sample_count = samples;
  state.render_target_count = targets;
  state.color_attachment_packed_unorm = format;
  state.functional_case = FunctionalCase::kDriverPcoTriangles;
  state.stage = PipelineStage::kTextureComplete;
  state.raster_state.color_mask = 15;
  state.raster_state.blend.source_rgb_factor = BlendFactor::kOne;
  state.raster_state.blend.source_alpha_factor = BlendFactor::kOne;
  state.raster_state.blend.destination_rgb_factor = BlendFactor::kOne;
  state.raster_state.blend.destination_alpha_factor = BlendFactor::kOne;
  const Codes clear{1, 257, 769, 2};
  const auto clear_float = Normalized(clear);
  std::memcpy(state.raster_state.clear_color, clear_float.data(), sizeof(clear_float));
  for (std::size_t i = 0; i < scenario.TargetWords() * targets; ++i) {
    const Codes loaded{static_cast<std::uint32_t>((7 + i * 31) % 1024),
                       static_cast<std::uint32_t>((13 + i * 47) % 1024),
                       static_cast<std::uint32_t>((521 + i * 61) % 1024),
                       static_cast<std::uint32_t>(i % 4)};
    scenario.expected.push_back(Word(load ? loaded : clear, format));
  }
  if (load)
    scenario.initial = scenario.expected;
  return scenario;
}

Codes ReadCodes(std::uint32_t word, PackedUnormFormat format) {
  if (format == PackedUnormFormat::kNone)
    return {word & 255U, (word >> 8) & 255U, (word >> 16) & 255U, word >> 24};
  Codes codes{word & 1023U, (word >> 10) & 1023U, (word >> 20) & 1023U, word >> 30};
  if (format == PackedUnormFormat::kBgr10A2) std::swap(codes[0], codes[2]);
  return codes;
}

Scenario NewMixedScenario(unsigned rotation = 0, bool load = true,
                          unsigned samples = 4, unsigned layers = 2) {
  auto scenario = NewScenario(PackedUnormFormat::kRgb10A2, "mixed normalized4B", 2, 2, layers, samples, 4);
  auto &state = scenario.state;
  const std::array<PackedUnormFormat, 4> formats{PackedUnormFormat::kNone,
      PackedUnormFormat::kRgb10A2, PackedUnormFormat::kBgr10A2, PackedUnormFormat::kNone};
  state.color_attachment_format_count = 4;
  for (unsigned target = 0; target < 4; ++target)
    state.color_attachment_packed_unorms[target] = formats[(target + rotation) % 4];
  state.color_attachment_packed_unorm = state.color_attachment_packed_unorms[0];
  const Color clear{0, 0.25F, 0.5F, 1};
  std::memcpy(state.raster_state.clear_color, clear.data(), sizeof(clear));
  for (unsigned target = 0; target < 4; ++target) {
    const auto format = state.color_attachment_packed_unorms[target];
    for (std::size_t sample = 0; sample < scenario.TargetWords(); ++sample) {
      const auto i = static_cast<unsigned>(target * scenario.TargetWords() + sample);
      const Codes initial = format == PackedUnormFormat::kNone
          ? Codes{(1 + i * 3) % 256, (2 + i * 7) % 256, (3 + i * 11) % 256, i % 256}
          : Codes{(1 + i * 3) % 1024, (257 + i * 7) % 1024, (769 + i * 11) % 1024, i % 4};
      const Codes cleared = format == PackedUnormFormat::kNone ? Codes{0,64,128,255} : Codes{0,256,512,3};
      scenario.expected[i] = Word(load ? initial : cleared, format);
    }
  }
  scenario.initial = load ? scenario.expected : std::vector<std::uint32_t>{};
  return scenario;
}

void AddFragment(Scenario &scenario, const std::vector<Color> &colors,
                 const std::vector<Codes> &result_codes, unsigned x = 0,
                 unsigned y = 0, unsigned layer = 0, unsigned coverage = 1) {
  const auto &state = scenario.state;
  Check(colors.size() == state.render_target_count &&
            result_codes.size() == state.render_target_count,
        scenario.name + " target count");
  FragmentInvocation invocation;
  invocation.x = x;
  invocation.y = y;
  invocation.layer = layer;
  invocation.sample_mask = coverage;
  invocation.submit_ordinal = scenario.invocations.size() + 1;
  invocation.primitive_id = static_cast<std::uint32_t>(invocation.submit_ordinal);
  invocation.parameter_index = invocation.primitive_id;
  FragmentOutput output;
  output.x = x;
  output.y = y;
  output.primitive_id = invocation.primitive_id;
  output.parameter_index = invocation.parameter_index;
  output.submit_ordinal = invocation.submit_ordinal;
  output.render_target_count = static_cast<std::uint8_t>(state.render_target_count);
  const bool explicit_outputs = std::any_of(state.fragment_output_mask.begin(), state.fragment_output_mask.end(),
                                            [](auto mask) { return mask != 0; });
  for (unsigned target = 0; target < state.render_target_count; ++target) {
    if (explicit_outputs && state.fragment_output_mask[target] == 0)
      continue;
    const auto format = state.color_attachment_format_count ? state.color_attachment_packed_unorms[target]
                                                           : state.color_attachment_packed_unorm;
    const auto mask = ChannelMask(state.raster_state.color_mask, format);
    output.written_mask[target] = 15;
    for (unsigned c = 0; c < 4; ++c)
      output.pixel_output[target * 4 + c] = Bits(colors[target][c]);
    for (unsigned sample = 0; sample < state.raster_state.sample_count; ++sample) {
      if (!(coverage & (1U << sample)))
        continue;
      const auto index = target * scenario.TargetWords() +
          ((layer * state.height + y) * state.width + x) *
              state.raster_state.sample_count + sample;
      const auto word = Word(result_codes[target], format);
      scenario.expected[index] = (scenario.expected[index] & ~mask) | (word & mask);
      ++scenario.covered_colors;
    }
  }
  scenario.invocations.push_back(invocation);
  scenario.outputs.push_back(output);
}

void Run(MemoryPool &pool, sc_core::sc_fifo<PipelineTxn> &input,
         sc_core::sc_fifo<PipelineTxn> &output, Scenario scenario,
         const std::string &failure = {}) {
  ++cases;
  auto &state = scenario.state;
  state.sequence = cases;
  state.active_fragment_invocations = static_cast<std::uint32_t>(scenario.invocations.size());
  state.counters.ps_invocations = scenario.invocations.size();
  state.fragment_invocations = StoreNewArray(pool, scenario.invocations);
  state.fragment_outputs = StoreNewArray(pool, scenario.outputs);
  if (std::any_of(state.fragment_output_mask.begin(), state.fragment_output_mask.end(),
                  [](auto mask) { return mask != 0; })) {
    // PBE-only fixture: the handle marks the explicit PIXOUT contract; these
    // are synthetic fragment records, not a native shader-execution claim.
    state.fragment_code = StoreNewArray(pool, std::vector<std::uint8_t>{0});
  }
  if (!scenario.initial.empty()) {
    state.color_attachment_load = StoreNewArray(pool, scenario.initial);
    state.color_attachment_load_enable = 1;
    state.color_attachment_load_bytes = scenario.initial.size() * 4;
  }
  if (failure == "float") state.color_attachment_float32 = 1;
  if (failure == "integer") state.color_attachment_raw_dwords = 4;
  if (failure == "srgb") state.color_is_srgb = 1;
  if (failure == "format") state.color_attachment_packed_unorm = static_cast<PackedUnormFormat>(255);
  if (failure == "load-size") --state.color_attachment_load_bytes;
  if (failure == "mixed-count") state.color_attachment_format_count = 3;
  if (failure == "mixed-count-overflow") state.color_attachment_format_count = 5;
  if (failure == "mixed-target0") state.color_attachment_packed_unorms[0] = PackedUnormFormat::kBgr10A2;
  if (failure == "mixed-inactive") { state.render_target_count = state.color_attachment_format_count = 2; state.color_attachment_packed_unorms[3] = PackedUnormFormat::kRgb10A2; }
  if (failure == "mixed-enum") state.color_attachment_packed_unorms[1] = static_cast<PackedUnormFormat>(255);
  if (failure == "mixed-legacy") state.color_attachment_format_count = 0;
  if (failure == "mixed-float") state.color_attachment_float32 = 1;
  if (failure == "mixed-integer") state.color_attachment_raw_dwords = 1;
  if (failure == "mixed-srgb") state.color_is_srgb = 1;
  if (failure == "legacy-geometry" || failure == "legacy-tessellation") {
    state.color_attachment_format_count = 0;
    state.color_attachment_packed_unorms = {};
    if (failure == "legacy-geometry")
      state.geometry_code = StoreNewArray(pool, std::vector<std::uint8_t>{0});
    else
      state.tessellation_state = StoreNewArray(pool, std::vector<TessellationState>{TessellationState{}});
  }
  const auto handle = pool.Allocate(sizeof(PipelineState));
  StorePipelineState(pool, handle, state);
  input.write({handle, cases, cases});
  const auto allocations_before = pool.allocations();
  try {
    sc_core::sc_start(sc_core::sc_time(10000, sc_core::SC_NS));
  } catch (const sc_core::sc_report &error) {
    const std::string message = error.what();
    const std::string reason = failure.rfind("legacy-", 0) == 0 ? "geometry MRT requires independent attachment LOAD"
        : failure.rfind("mixed-", 0) == 0 ? "per-target color attachment"
        : failure == "format" ? "invalid packed UNORM format"
        : failure == "load-size" ? "LOAD byte count"
        : "packed UNORM attachment state is invalid";
    Check(!failure.empty() && message.find(reason) != std::string::npos,
          "unexpected failure: " + message);
    PipelineTxn unexpected;
    Check(!output.nb_read(unexpected) &&
              LoadPipelineState(pool, handle).stage == PipelineStage::kTextureComplete,
          "invalid state published completed output");
    Check(pool.allocations() == allocations_before, "invalid metadata allocated attachment output");
    ReleaseFunctionalPayloads(pool, state);
    pool.Release(handle);
    return;
  }
  Check(failure.empty(), "invalid state unexpectedly passed");
  PipelineTxn completed;
  Check(output.nb_read(completed) && completed.sequence == cases, scenario.name + " FIFO order");
  state = LoadPipelineState(pool, handle);
  Check(state.stage == PipelineStage::kPbeComplete, scenario.name + " completion");
  Check(state.counters.ps_invocations == scenario.invocations.size(), scenario.name + " real shader count");
  Check(state.framebuffer_bytes == scenario.TargetWords() * 4, scenario.name + " packed byte extent");
  for (unsigned target = 0; target < state.render_target_count; ++target) {
    const auto actual = LoadArray<std::uint32_t>(pool, target ?
        state.extra_pbe_framebuffer[target - 1] : state.pbe_framebuffer);
    Check(actual.size() == scenario.TargetWords(), scenario.name + " independent target extent");
    for (std::size_t i = 0; i < actual.size(); ++i)
      Check(actual[i] == scenario.expected[target * scenario.TargetWords() + i],
            scenario.name + " target=" + std::to_string(target) + " sample_word=" + std::to_string(i) +
            " got=" + std::to_string(actual[i]) +
            " expected=" + std::to_string(scenario.expected[target * scenario.TargetWords() + i]));
  }
  Check(state.counters.pbe_fragment_writes ==
            (state.raster_state.color_mask ? scenario.covered_colors : 0), scenario.name + " writes");
  Check(state.counters.pbe_pixels_written == scenario.expected.size(), scenario.name + " serialization");
  Check(state.counters.pbe_color_reads == (state.raster_state.blend.enable ? scenario.covered_colors : 0) &&
            state.counters.pbe_blended_fragments == state.counters.pbe_color_reads,
        scenario.name + " blend work");
  for (unsigned target = 1; target < state.render_target_count; ++target)
    pool.Release(state.extra_pbe_framebuffer[target - 1]);
  ReleaseFunctionalPayloads(pool, state);
  pool.Release(handle);
}

void RunMixed(MemoryPool &pool, sc_core::sc_fifo<PipelineTxn> &input,
              sc_core::sc_fifo<PipelineTxn> &output) {
  // The same shared mask/blend state is applied independently in each
  // attachment's integer code domain. Two ordered fragments prove that the
  // second blend sees the first format-specific quantized result.
  for (unsigned rotation = 0; rotation < 3; ++rotation)
    for (bool load : {false, true}) for (bool blend : {false, true})
      for (unsigned mask = 0; mask < 16; ++mask) {
        auto scenario = NewMixedScenario(rotation, load);
        auto &state = scenario.state;
        state.raster_state.color_mask = mask;
        state.raster_state.blend.enable = blend;
        for (unsigned fragment = 0; fragment < 2; ++fragment) {
          std::vector<Codes> result;
          std::vector<Color> colors;
          for (unsigned target = 0; target < 4; ++target) {
            const auto format = state.color_attachment_packed_unorms[target];
            const Codes source{1U + target + fragment, 2U + target + fragment,
                               3U + target + fragment, format == PackedUnormFormat::kNone ? 15U : 1U};
            colors.push_back(Normalized(source, format));
            Codes codes = source;
            if (blend) {
              // All covered samples are initialized independently. For this
              // one-pixel RMW check, cover one selected sample only.
              const auto index = target * scenario.TargetWords() +
                  ((state.attachment_layers - 1) * state.height * state.width + 3) * state.raster_state.sample_count + 2;
              const Codes destination = ReadCodes(scenario.expected[index], format);
              for (unsigned c = 0; c < 4; ++c)
                codes[c] = std::min(destination[c] + source[c], format == PackedUnormFormat::kNone ? 255U : c == 3 ? 3U : 1023U);
            }
            result.push_back(codes);
          }
          AddFragment(scenario, colors, result, 1, 1, state.attachment_layers - 1, 4);
        }
        Run(pool, input, output, scenario);
      }
  // Partial sample coverage and entirely absent attachment exports preserve
  // independently supplied LOAD bits (including packed RGB low bits).
  for (unsigned absent = 0; absent < 4; ++absent) {
    auto scenario = NewMixedScenario();
    scenario.state.fragment_output_mask = {15,15,15,15};
    scenario.state.fragment_output_mask[absent] = 0;
    std::vector<Color> colors;
    std::vector<Codes> codes;
    for (unsigned target = 0; target < 4; ++target) {
      const auto format = scenario.state.color_attachment_packed_unorms[target];
      codes.push_back(format == PackedUnormFormat::kNone ? Codes{1,2,3,255} : Codes{1,257,769,2});
      colors.push_back(Normalized(codes.back(), format));
    }
    AddFragment(scenario, colors, codes, 1, 0, 1, 5);
    Run(pool, input, output, scenario);
  }
  // One source value simultaneously exercises different codec tie rules:
  // RGBA8 rounds half upward; packed10/2 uses nearest-even.
  auto boundary = NewMixedScenario(0, true, 1, 1);
  const Color values{0.5F, 0.5F / 1023, 1.5F / 1023, 0.5F};
  AddFragment(boundary, std::vector<Color>(4, values),
      {{128,0,0,128}, {512,0,2,2}, {512,0,2,2}, {128,0,0,128}});
  Run(pool, input, output, boundary);
  for (unsigned targets = 1; targets < 4; ++targets) {
    auto bounded = NewMixedScenario(0, true, 1, 1);
    bounded.state.render_target_count = bounded.state.color_attachment_format_count = targets;
    for (unsigned target = targets; target < 4; ++target)
      bounded.state.color_attachment_packed_unorms[target] = PackedUnormFormat::kNone;
    bounded.initial.resize(targets * bounded.TargetWords());
    bounded.expected = bounded.initial;
    std::vector<Color> colors;
    std::vector<Codes> codes;
    for (unsigned target = 0; target < targets; ++target) {
      const auto format = bounded.state.color_attachment_packed_unorms[target];
      codes.push_back(format == PackedUnormFormat::kNone ? Codes{1,2,3,255} : Codes{1,257,769,2});
      colors.push_back(Normalized(codes.back(), format));
    }
    AddFragment(bounded, colors, codes);
    Run(pool, input, output, bounded);
  }
  Check(pool.bytes_in_flight() == 0 && pool.allocations() == pool.releases(),
        "mixed format output and LOAD ownership balanced");
}

using PreciseColor = std::array<long double, 4>;

template <typename Rgb>
PreciseColor RgbAndAlpha(Rgb rgb, long double alpha) {
  return {rgb(0), rgb(1), rgb(2), alpha};
}

// These are algebraic GL blend expressions, not calls to the PBE's factor,
// equation or packing helpers. Source/constant values are exact dyadic
// fractions; destinations are the exact rational values of selected codes.
// New cases deliberately avoid quantizer halfway boundaries (covered above)
// so long-double oracle arithmetic cannot hide a float rounding discrepancy.
Codes BlendOracleCodes(const PreciseColor &result, const std::string &name) {
  Codes codes{};
  for (unsigned c = 0; c < 4; ++c) {
    const long double scaled = std::clamp(result[c], 0.0L, 1.0L) *
                               (c == 3 ? 3.0L : 1023.0L);
    Check(std::fabs(scaled - (std::floor(scaled) + 0.5L)) > 1.0L / 1024,
          name + " independent oracle must stay away from halfway");
    codes[c] = static_cast<std::uint32_t>(std::floor(scaled + 0.5L));
  }
  return codes;
}

void RunBlendEquationMatrix(MemoryPool &pool,
                            sc_core::sc_fifo<PipelineTxn> &input,
                            sc_core::sc_fifo<PipelineTxn> &output,
                            PackedUnormFormat format) {
  using E = BlendEquation;
  using F = BlendFactor;
  using Oracle = std::function<PreciseColor(const PreciseColor &,
                                           const PreciseColor &,
                                           const PreciseColor &)>;
  struct BlendCase {
    const char *name;
    E rgb_equation, alpha_equation;
    F source_rgb, destination_rgb, source_alpha, destination_alpha;
    Oracle oracle;
  };
  const std::vector<BlendCase> blend_cases = {
    {"subtract RGB / reverse alpha", E::kSubtract, E::kReverseSubtract,
     F::kOne, F::kOne, F::kOne, F::kOne,
     [](const auto &s, const auto &d, const auto &) {
       return RgbAndAlpha([&](unsigned c) { return s[c] - d[c]; }, d[3] - s[3]);
     }},
    {"reverse RGB / subtract alpha", E::kReverseSubtract, E::kSubtract,
     F::kOne, F::kOne, F::kOne, F::kOne,
     [](const auto &s, const auto &d, const auto &) {
       return RgbAndAlpha([&](unsigned c) { return d[c] - s[c]; }, s[3] - d[3]);
     }},
    {"MIN RGB / MAX alpha ignore zero factors", E::kMin, E::kMax,
     F::kZero, F::kZero, F::kZero, F::kZero,
     [](const auto &s, const auto &d, const auto &) {
       return RgbAndAlpha([&](unsigned c) { return std::min(s[c], d[c]); },
                          std::max(s[3], d[3]));
     }},
    {"MAX RGB / MIN alpha ignore nonzero factors", E::kMax, E::kMin,
     F::kSourceColor, F::kDestinationAlpha, F::kConstantAlpha, F::kSourceAlpha,
     [](const auto &s, const auto &d, const auto &) {
       return RgbAndAlpha([&](unsigned c) { return std::max(s[c], d[c]); },
                          std::min(s[3], d[3]));
     }},
    {"source color and complement / separate constant alpha", E::kAdd, E::kSubtract,
     F::kSourceColor, F::kOneMinusSourceColor, F::kConstantAlpha, F::kOneMinusConstantAlpha,
     [](const auto &s, const auto &d, const auto &k) {
       return RgbAndAlpha([&](unsigned c) { return s[c] * s[c] + d[c] * (1 - s[c]); },
                          s[3] * k[3] - d[3] * (1 - k[3]));
     }},
    {"destination color and complement / separate alpha factors", E::kSubtract, E::kAdd,
     F::kDestinationColor, F::kOneMinusDestinationColor, F::kDestinationAlpha, F::kOneMinusSourceAlpha,
     [](const auto &s, const auto &d, const auto &) {
       return RgbAndAlpha([&](unsigned c) { return s[c] * d[c] - d[c] * (1 - d[c]); },
                          s[3] * d[3] + d[3] * (1 - s[3]));
     }},
    {"source alpha complement / reverse RGB", E::kReverseSubtract, E::kAdd,
     F::kSourceAlpha, F::kOneMinusSourceAlpha, F::kOneMinusDestinationAlpha, F::kDestinationAlpha,
     [](const auto &s, const auto &d, const auto &) {
       return RgbAndAlpha([&](unsigned c) { return d[c] * (1 - s[3]) - s[c] * s[3]; },
                          s[3] * (1 - d[3]) + d[3] * d[3]);
     }},
    {"destination alpha complement / reverse alpha", E::kAdd, E::kReverseSubtract,
     F::kDestinationAlpha, F::kOneMinusDestinationAlpha, F::kSourceAlpha, F::kConstantAlpha,
     [](const auto &s, const auto &d, const auto &k) {
       return RgbAndAlpha([&](unsigned c) { return s[c] * d[3] + d[c] * (1 - d[3]); },
                          d[3] * k[3] - s[3] * s[3]);
     }},
    {"per-channel constant and complement / MAX alpha", E::kAdd, E::kMax,
     F::kConstantColor, F::kOneMinusConstantColor, F::kZero, F::kZero,
     [](const auto &s, const auto &d, const auto &k) {
       return RgbAndAlpha([&](unsigned c) { return s[c] * k[c] + d[c] * (1 - k[c]); },
                          std::max(s[3], d[3]));
     }},
    {"constant alpha complement / MIN alpha", E::kSubtract, E::kMin,
     F::kConstantAlpha, F::kOneMinusConstantAlpha, F::kOneMinusConstantAlpha, F::kConstantColor,
     [](const auto &s, const auto &d, const auto &k) {
       return RgbAndAlpha([&](unsigned c) { return s[c] * k[3] - d[c] * (1 - k[3]); },
                          std::min(s[3], d[3]));
     }},
    {"source alpha saturate / alpha factor is one", E::kAdd, E::kSubtract,
     F::kSourceAlphaSaturate, F::kOne, F::kSourceAlphaSaturate, F::kOne,
     [](const auto &s, const auto &d, const auto &) {
       return RgbAndAlpha([&](unsigned c) {
         return s[c] * std::min(s[3], 1 - d[3]) + d[c];
       }, s[3] - d[3]);
     }},
    {"constant factors exchange source/destination", E::kAdd, E::kAdd,
     F::kOneMinusConstantColor, F::kConstantColor, F::kOneMinusConstantAlpha, F::kConstantAlpha,
     [](const auto &s, const auto &d, const auto &k) {
       return RgbAndAlpha([&](unsigned c) { return s[c] * (1 - k[c]) + d[c] * k[c]; },
                          s[3] * (1 - k[3]) + d[3] * k[3]);
     }},
    {"destination factors exchange / reverse RGB", E::kReverseSubtract, E::kSubtract,
     F::kOneMinusDestinationColor, F::kDestinationColor, F::kOneMinusSourceAlpha, F::kSourceAlpha,
     [](const auto &s, const auto &d, const auto &) {
       return RgbAndAlpha([&](unsigned c) { return d[c] * d[c] - s[c] * (1 - d[c]); },
                          s[3] * (1 - s[3]) - d[3] * s[3]);
     }},
    {"MIN ignores RGB factors / alpha uses color factors", E::kMin, E::kSubtract,
     F::kZero, F::kZero, F::kConstantColor, F::kDestinationColor,
     [](const auto &s, const auto &d, const auto &k) {
       return RgbAndAlpha([&](unsigned c) { return std::min(s[c], d[c]); },
                          s[3] * k[3] - d[3] * d[3]);
     }},
  };
  const PreciseColor source{0.3125L, 0.625L, 0.875L, 0.75L};
  const PreciseColor constant{0.125L, 0.375L, 0.875L, 0.625L};
  // All four representable alpha values exercise both branches of
  // min(source-alpha, 1-destination-alpha), including its zero bound.
  const std::array<Codes, 4> destinations = {{
      {137, 619, 851, 1}, {1001, 257, 31, 2},
      {19, 991, 503, 0}, {751, 3, 1021, 3}}};
  for (const auto &test : blend_cases)
  for (const std::uint8_t mask : {15U, 5U, 10U}) {
    auto scenario = NewScenario(format, test.name, 1, 1, 1, 1, 4);
    scenario.state.raster_state.color_mask = mask;
    auto &blend = scenario.state.raster_state.blend;
    blend.enable = 1;
    blend.rgb_equation = test.rgb_equation;
    blend.alpha_equation = test.alpha_equation;
    blend.source_rgb_factor = test.source_rgb;
    blend.destination_rgb_factor = test.destination_rgb;
    blend.source_alpha_factor = test.source_alpha;
    blend.destination_alpha_factor = test.destination_alpha;
    Color shader_color{};
    for (unsigned c = 0; c < 4; ++c) {
      shader_color[c] = static_cast<float>(source[c]);
      blend.constant_color_bits[c] = Bits(static_cast<float>(constant[c]));
    }
    std::vector<Codes> result_codes;
    for (unsigned target = 0; target < destinations.size(); ++target) {
      scenario.initial[target] = scenario.expected[target] = Word(destinations[target], format);
      const auto &d = destinations[target];
      const PreciseColor destination{d[0] / 1023.0L, d[1] / 1023.0L,
                                     d[2] / 1023.0L, d[3] / 3.0L};
      result_codes.push_back(BlendOracleCodes(test.oracle(source, destination, constant), test.name));
    }
    AddFragment(scenario, std::vector<Color>(4, shader_color), result_codes);
    Run(pool, input, output, scenario);
  }
}
} // namespace

int sc_main(int argc, char **argv) {
  try {
    MemoryPool pool;
    sc_core::sc_fifo<PipelineTxn> input("input", 2), output("output", 2);
    Pbe pbe("pbe", pool);
    pbe.input(input);
    pbe.output(output);
    if (argc == 2 && std::string(argv[1]) == "mixed") {
      RunMixed(pool, input, output);
      std::cout << "PBE mixed normalized4B PASS cases=" << cases << " checks=" << checks << '\n';
      return 0;
    }
    if (argc == 2) {
      const bool mixed = std::string(argv[1]).rfind("mixed-", 0) == 0 ||
                         std::string(argv[1]).rfind("legacy-", 0) == 0;
      auto scenario = mixed ? NewMixedScenario() : NewScenario(PackedUnormFormat::kRgb10A2, "reject");
      if (!mixed) AddFragment(scenario, {{0.5F, 0.25F, 0.75F, 1.0F}}, {{512, 256, 767, 3}});
      Run(pool, input, output, scenario, argv[1]);
      std::cout << "PBE packed UNORM reject " << argv[1] << " PASS checks=" << checks << '\n';
      return 0;
    }
    Check(argc == 1, "unexpected CLI arguments");
    for (const auto format : {PackedUnormFormat::kRgb10A2, PackedUnormFormat::kBgr10A2}) {
      for (unsigned samples : {1U, 2U, 4U, 8U, 16U})
        for (unsigned targets : {1U, 4U})
          for (unsigned layers : {1U, 2U})
            for (unsigned mask = 0; mask < 16; ++mask) {
              auto scenario = NewScenario(format, "LOAD mask/layer/sample/MRT", 2, 2, layers, samples, targets);
              scenario.state.raster_state.color_mask = static_cast<std::uint8_t>(mask);
              std::vector<Codes> codes;
              std::vector<Color> colors;
              for (unsigned target = 0; target < targets; ++target) {
                codes.push_back({1 + target * 3, 257 + target, 769 - target, target % 4});
                colors.push_back(Normalized(codes.back()));
              }
              AddFragment(scenario, colors, codes, 1, 1, layers - 1,
                          ((1U << samples) - 1) & 0x5555U);
              Run(pool, input, output, scenario);
            }
      auto clear = NewScenario(format, "packed clear", 2, 2, 2, 4, 4, false);
      clear.state.raster_state.color_mask = 0;
      AddFragment(clear, std::vector<Color>(4), std::vector<Codes>(4));
      Run(pool, input, output, clear);

      auto sweep = NewScenario(format, "all 1024 exact RGB codes", 1024, 1);
      for (unsigned value = 0; value < 1024; ++value) {
        const Codes codes{value, 1023 - value, (value * 37) % 1024, value % 4};
        AddFragment(sweep, {Normalized(codes)}, {codes}, value);
      }
      Run(pool, input, output, sweep);

      const float nan = std::numeric_limits<float>::quiet_NaN();
      const float inf = std::numeric_limits<float>::infinity();
      const std::vector<std::pair<Color, Codes>> boundaries = {
        {{0.5F, 0.5F / 1023, 1.5F / 1023, 0.5F}, {512, 0, 2, 2}},
        {{2.5F / 1023, 3.5F / 1023, 1022.5F / 1023, 1.0F / 6}, {2, 4, 1022, 0}},
        {{0.4999F, 0.5001F, -1, 0.4999F}, {511, 512, 0, 1}},
        {{nan, inf, -inf, nan}, {0, 1023, 0, 0}},
        {{0.0003F, 0.0006F, 0.9999F, 1.1F}, {0, 1, 1023, 3}},
      };
      for (const auto &[color, codes] : boundaries) {
        auto scenario = NewScenario(format, "finite/nonfinite precision boundary");
        AddFragment(scenario, {color}, {codes});
        Run(pool, input, output, scenario);
      }
      for (float increment : {0.49F, 0.51F, 1.0F}) {
        auto scenario = NewScenario(format, "per-fragment additive quantization", 1, 1);
        scenario.state.raster_state.blend.enable = 1;
        const Codes initial{1, 5, 513, 0};
        scenario.initial[0] = scenario.expected[0] = Word(initial, format);
        const Color color{increment / 1023, increment / 1023, increment / 1023, increment / 3};
        for (unsigned fragment = 1; fragment <= 2; ++fragment) {
          const unsigned add = increment < 0.5F ? 0 : fragment;
          AddFragment(scenario, {color}, {{initial[0] + add, initial[1] + add, initial[2] + add, add}});
        }
        Run(pool, input, output, scenario);
      }
      auto blend = NewScenario(format, "float source before blend and clamp", 1, 1);
      blend.state.raster_state.blend.enable = 1;
      blend.state.raster_state.blend.source_rgb_factor = BlendFactor::kSourceAlpha;
      blend.state.raster_state.blend.destination_rgb_factor = BlendFactor::kZero;
      blend.state.raster_state.blend.destination_alpha_factor = BlendFactor::kZero;
      AddFragment(blend, {{1.4F / 1023, 2.6F / 1023, 3.4F / 1023, 0.5F}}, {{1, 1, 2, 2}});
      Run(pool, input, output, blend);
      auto clamp = NewScenario(format, "source clamp before blend", 1, 1);
      clamp.state.raster_state.blend = blend.state.raster_state.blend;
      AddFragment(clamp, {{2.0F, -2.0F, inf, 0.5F}}, {{512, 0, 512, 2}});
      Run(pool, input, output, clamp);
      RunBlendEquationMatrix(pool, input, output, format);
    }
    Check(output.num_available() == 0, "no extra completion");
    std::cout << "PBE packed UNORM PASS cases=" << cases << " checks=" << checks << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
