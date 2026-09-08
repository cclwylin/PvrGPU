// Execute public native MBYP exports, then pass their untouched PIXOUT bits
// through the real PBE. Nonfinite shader colors are data, not malformed state.
#include "common/functional_types.h"
#include "common/msaa.h"
#include "common/pipeline_state.h"
#include "fragment/pbe.h"
#include "shader/pco_iss.h"

#include <systemc>

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
unsigned checks = 0;

void Check(bool condition, const std::string &message) {
  ++checks;
  if (!condition)
    throw std::runtime_error("PBE nonfinite test: " + message);
}

float Float(std::uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::uint32_t Bits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::vector<std::uint8_t> ExportProgram(unsigned targets, bool sparse) {
  std::vector<std::uint8_t> binary;
  for (unsigned lane = 0; lane < targets * 4; ++lane) {
    // SH -> TEMP is the public form in Mesa's UBO fixture prologue.
    const std::array<std::uint8_t, 10> load = {
      0x35, 0x82, 0, 0x87, static_cast<std::uint8_t>(0x80 | lane),
      0x08, 0, 0, 0, static_cast<std::uint8_t>(0x40 | lane)};
    binary.insert(binary.end(), load.begin(), load.end());
  }
  for (unsigned lane = 0; lane < targets * 4; ++lane) {
    if (sparse && lane / 4 == 2)
      continue;
    // TEMP -> PIXOUT with the extended destination form preserves all four
    // MRT identities. The final 0xff is the canonical odd-byte word pad.
    const unsigned special = lane < 4 ? 32 + lane : 160 + lane;
    const std::array<std::uint8_t, 10> move = {
      0x35, 0x8a, static_cast<std::uint8_t>(lane + 1 == targets * 4 ? 0x80 : 0),
      0x87, static_cast<std::uint8_t>(0x40 | lane), 0, 0,
      static_cast<std::uint8_t>(0x80 | (special & 0x3f)),
      static_cast<std::uint8_t>(special >> 6), 0xff};
    binary.insert(binary.end(), move.begin(), move.end());
  }
  return binary;
}

enum class Format { kUnorm, kSrgb, kFloat, kUint };

std::uint8_t ExpectedByte(std::uint32_t bits, bool srgb) {
  const float raw = Float(bits);
  const float clamped = std::isnan(raw) ? 0.0F : std::clamp(raw, 0.0F, 1.0F);
  if (!srgb) {
    // Preserve the existing binary32 half-up quantization, including 0.3f.
    return static_cast<std::uint8_t>(std::floor(clamped * 255.0F + 0.5F));
  }
  const double x = clamped;
  const double encoded = x <= 0.0031308 ? 12.92 * x
      : 1.055 * std::pow(x, 1.0 / 2.4) - 0.055;
  return static_cast<std::uint8_t>(std::floor(std::clamp(encoded, 0.0, 1.0) * 255 + 0.5));
}

void RunCase(MemoryPool &pool, sc_core::sc_fifo<PipelineTxn> &input,
             sc_core::sc_fifo<PipelineTxn> &output, std::uint32_t sequence,
             const std::array<std::uint32_t, 4> &words, Format format,
             unsigned samples, unsigned targets, std::uint8_t mask,
             bool sparse, bool blend, const std::string &failure = {}) {
  const auto binary = ExportProgram(targets, sparse);
  const auto decoded = Decode(ShaderStage::kFragment, binary);
  PcoFragmentExecutionContext context;
  context.shared_count = targets * 4;
  for (unsigned target = 0; target < targets; ++target)
    for (unsigned component = 0; component < 4; ++component)
      context.shared_registers[target * 4 + component] = words[(target + component) % 4];
  const auto execution = ExecuteFragment(decoded.summary, decoded.instructions, context);
  Check(!execution.suspended && !execution.discarded && !execution.texture_request_valid &&
            execution.executed_instruction_count == targets * 8 - (sparse ? 4 : 0),
        "native exports must really execute without synthetic sampler responses");
  for (unsigned lane = 0; lane < targets * 4; ++lane)
    if (!(sparse && lane / 4 == 2))
      Check(execution.pixel_outputs[lane] == context.shared_registers[lane],
            "native ISS must preserve every NaN payload/Inf/finite PIXOUT bit");

  const bool raw = format == Format::kFloat || format == Format::kUint;
  const unsigned bpp = raw ? 16 : 4;
  const unsigned target_bytes = 2 * samples * bpp;
  std::vector<std::uint8_t> initial(targets * target_bytes);
  for (unsigned byte = 0; byte < initial.size(); ++byte)
    initial[byte] = static_cast<std::uint8_t>(17 + byte * 37);
  auto expected = initial;
  FragmentInvocation invocation;
  invocation.sample_mask = RasterSampleMask(samples) & 0x5555U;
  invocation.submit_ordinal = 1;
  FragmentOutput result;
  result.submit_ordinal = 1;
  result.render_target_count = targets;
  for (unsigned target = 0; target < targets; ++target) {
    result.written_mask[target] = (execution.written_mask >> (target * 4)) & 0xf;
    for (unsigned component = 0; component < 4; ++component)
      result.pixel_output[target * 4 + component] = execution.pixel_outputs[target * 4 + component];
    if (sparse && target == 2)
      continue;
    for (unsigned sample = 0; sample < samples; ++sample) {
      if (!(invocation.sample_mask & (1U << sample)))
        continue;
      for (unsigned component = 0; component < 4; ++component) {
        if (!(mask & (1U << component)))
          continue;
        const auto value = context.shared_registers[target * 4 + component];
        const unsigned offset = target * target_bytes + sample * bpp + component * (raw ? 4 : 1);
        if (raw)
          std::memcpy(expected.data() + offset, &value, sizeof(value));
        else
          expected[offset] = ExpectedByte(value, format == Format::kSrgb && component != 3);
      }
    }
  }
  PipelineState state;
  state.width = 2;
  state.height = 1;
  state.sequence = sequence;
  state.functional_case = FunctionalCase::kDriverPcoTriangles;
  state.stage = PipelineStage::kTextureComplete;
  state.active_fragment_invocations = 1;
  state.counters.ps_invocations = 1;
  state.raster_state.sample_count = samples;
  state.raster_state.color_mask = mask;
  state.raster_state.blend.enable = blend;
  state.raster_state.blend.source_rgb_factor = BlendFactor::kOne;
  state.raster_state.blend.source_alpha_factor = BlendFactor::kOne;
  state.raster_state.blend.destination_rgb_factor = BlendFactor::kZero;
  state.raster_state.blend.destination_alpha_factor = BlendFactor::kZero;
  state.color_attachment_float32 = format == Format::kFloat;
  state.color_attachment_raw_dwords = format == Format::kUint ? 4 : 0;
  state.color_is_srgb = format == Format::kSrgb;
  state.render_target_count = targets;
  for (unsigned target = 0; target < targets; ++target)
    state.fragment_output_mask[target] = sparse && target == 2 ? 0 : 15;
  state.fragment_code = StoreNewArray(pool, binary);
  state.color_attachment_load = StoreNewArray(pool, initial);
  state.color_attachment_load_enable = 1;
  state.color_attachment_load_bytes = initial.size();
  if (failure == "output-mask")
    state.fragment_output_mask[0] = 0;
  else if (failure == "load-size")
    --state.color_attachment_load_bytes;
  else if (failure == "identity")
    result.primitive_id = 1;
  else if (failure == "clear-nan") {
    pool.Release(state.color_attachment_load);
    state.color_attachment_load = {};
    state.color_attachment_load_enable = 0;
    state.color_attachment_load_bytes = 0;
    state.raster_state.clear_color[0] = Float(0x7fc00000);
  } else if (failure == "blend-constant-nan") {
    state.raster_state.blend.enable = 1;
    state.raster_state.blend.constant_color_bits[0] = 0x7fc00000;
  }
  state.fragment_invocations = StoreNewArray(pool, std::vector<FragmentInvocation>{invocation});
  state.fragment_outputs = StoreNewArray(pool, std::vector<FragmentOutput>{result});
  const PoolHandle handle = pool.Allocate(sizeof(PipelineState));
  StorePipelineState(pool, handle, state);
  input.write({handle, sequence, sequence});
  try {
    sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_NS));
  } catch (const sc_core::sc_report &error) {
    const std::string message = error.what();
    const std::string expected_reason = failure == "output-mask" ? "expected PIXOUT"
        : failure == "load-size" ? "LOAD byte count"
        : failure == "identity" ? "lost fragment identity"
        : failure == "clear-nan" || failure == "blend-constant-nan" ? "non-finite UNORM state" : "";
    Check(!expected_reason.empty() && message.find(expected_reason) != std::string::npos,
          "unexpected native failure: " + message);
    PipelineTxn unexpected;
    Check(!output.nb_read(unexpected) &&
              LoadPipelineState(pool, handle).stage == PipelineStage::kTextureComplete,
          "invalid metadata must not publish a completed framebuffer");
    ReleaseFunctionalPayloads(pool, state);
    pool.Release(handle);
    return;
  }
  Check(failure.empty(), "invalid metadata unexpectedly completed");
  PipelineTxn completed;
  Check(output.nb_read(completed) && completed.sequence == sequence, "PBE FIFO completion");
  state = LoadPipelineState(pool, handle);
  Check(state.stage == PipelineStage::kPbeComplete && state.counters.ps_invocations == 1,
        "PBE completion must not fabricate or drop shader invocations");
  for (unsigned target = 0; target < targets; ++target) {
    const auto bytes = LoadArray<std::uint8_t>(pool, target == 0 ? state.pbe_framebuffer
        : state.extra_pbe_framebuffer[target - 1]);
    Check(bytes.size() == target_bytes &&
              std::equal(bytes.begin(), bytes.end(), expected.begin() + target * target_bytes),
          "wrong normalized conversion/raw bits or changed masked, uncovered, LOAD-only texel");
  }
  const unsigned covered_samples = (samples + 1) / 2;
  Check(state.counters.pbe_fragment_writes == (mask ? covered_samples * (targets - (sparse ? 1 : 0)) : 0),
        "nonfinite conversion must retain actual sample/attachment write counts");
  for (unsigned target = 1; target < targets; ++target)
    pool.Release(state.extra_pbe_framebuffer[target - 1]);
  ReleaseFunctionalPayloads(pool, state);
  pool.Release(handle);
}
} // namespace

int sc_main(int argc, char **argv) {
  try {
    MemoryPool pool;
    sc_core::sc_fifo<PipelineTxn> input("input", 1), output("output", 1);
    Pbe pbe("pbe", pool);
    pbe.input(input);
    pbe.output(output);
    const std::vector<std::array<std::uint32_t, 4>> values = {
      {0x7fc00000, 0x7f800000, 0xff800000, 0x3f800000},
      {0x7f800001, 0xff812345, 0x7fffffff, 0xffc54321},
      {0, 0x80000000, 1, 0x80000001},
      {0x7f7fffff, 0xff7fffff, Bits(2.0F), Bits(-2.0F)},
      {Bits(0.3F), Bits(0.5F / 255), Bits(2.5F / 255), Bits(3.5F / 255)},
      {Bits(0.25F), Bits(0.5F), Bits(0.75F), Bits(1.0F)},
    };
    std::uint32_t sequence = 0;
    if (argc == 2) {
      RunCase(pool, input, output, ++sequence, values[0], Format::kUnorm,
              1, 1, 15, false, false, argv[1]);
    } else {
      for (const auto &words : values)
        for (Format format : {Format::kUnorm, Format::kSrgb, Format::kFloat, Format::kUint})
          for (unsigned samples : {1U, 4U, 16U})
            for (std::uint8_t mask : {0xf, 0x5, 0x0})
              for (unsigned layout = 0; layout < 3; ++layout)
                RunCase(pool, input, output, ++sequence, words, format, samples,
                        layout ? 4 : 1, mask, layout == 2, false);
      for (const auto &words : values)
        for (Format format : {Format::kUnorm, Format::kSrgb})
          RunCase(pool, input, output, ++sequence, words, format, 4, 4, 15, true, true);
    }
    Check(pool.bytes_in_flight() == 0 && pool.allocations() == pool.releases(), "MemoryPool balance");
    std::cout << "pbe_nonfinite_output_test: PASS (" << sequence << " cases, " << checks << " checks)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "pbe_nonfinite_output_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
