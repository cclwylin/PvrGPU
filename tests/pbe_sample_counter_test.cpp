// PBE counters count actual sample/attachment work, not pixel-frequency
// shader invocations. Exercise coverage, overdraw, MRT, masks and integer LOAD.
#include "common/functional_types.h"
#include "common/msaa.h"
#include "common/pipeline_state.h"
#include "fragment/pbe.h"

#include <systemc>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace pvrgpu::stub;

void Check(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void RunCase(MemoryPool &pool, sc_core::sc_fifo<PipelineTxn> &input,
             sc_core::sc_fifo<PipelineTxn> &output, std::uint64_t sequence,
             std::uint32_t samples, std::uint32_t targets, bool blend,
             std::uint8_t color_mask, bool integer, bool empty) {
  const std::uint32_t valid_mask = RasterSampleMask(samples);
  std::vector<std::uint32_t> coverage;
  if (!empty) {
    coverage.push_back(valid_mask & 0x5555U);
    if ((valid_mask & 0xaaaaU) != 0)
      coverage.push_back(valid_mask & 0xaaaaU);
    if (blend)
      coverage.push_back(valid_mask);
  }
  std::vector<FragmentInvocation> invocations;
  std::vector<FragmentOutput> outputs;
  std::uint64_t covered_samples = 0;
  for (std::size_t index = 0; index < coverage.size(); ++index) {
    FragmentInvocation invocation;
    invocation.sample_mask = coverage[index];
    invocation.submit_ordinal = index + 1;
    invocation.primitive_id = static_cast<std::uint32_t>(index);
    invocations.push_back(invocation);
    FragmentOutput result;
    result.submit_ordinal = invocation.submit_ordinal;
    result.primitive_id = invocation.primitive_id;
    result.render_target_count = static_cast<std::uint8_t>(targets);
    for (std::uint32_t target = 0; target < targets; ++target) {
      result.written_mask[target] = 0xf;
      for (std::uint32_t channel = 0; channel < 4; ++channel)
        result.pixel_output[target * 4 + channel] = integer
            ? 0xf1234560U + channel : FloatBits(0.25F);
    }
    outputs.push_back(result);
    for (std::uint32_t sample = 0; sample < samples; ++sample)
      covered_samples += (coverage[index] >> sample) & 1U;
  }

  PipelineState state;
  state.width = state.height = 1;
  state.sequence = sequence;
  state.functional_case = FunctionalCase::kDriverPcoTriangles;
  state.stage = PipelineStage::kTextureComplete;
  state.active_fragment_invocations = invocations.size();
  state.counters.ps_invocations = invocations.size();
  state.counters.renderer_cycles = 7;
  state.render_target_count = targets;
  state.raster_state.sample_count = samples;
  state.raster_state.color_mask = color_mask;
  state.raster_state.blend.enable = blend ? 1 : 0;
  state.raster_state.blend.source_rgb_factor = BlendFactor::kOne;
  state.raster_state.blend.source_alpha_factor = BlendFactor::kOne;
  state.raster_state.blend.destination_rgb_factor = BlendFactor::kOne;
  state.raster_state.blend.destination_alpha_factor = BlendFactor::kOne;
  state.color_attachment_raw_dwords = integer ? 4 : 0;
  if (integer) {
    std::vector<std::uint32_t> initial(samples * 4);
    for (std::size_t index = 0; index < initial.size(); ++index)
      initial[index] = 0x80000000U + static_cast<std::uint32_t>(index);
    std::vector<std::uint8_t> bytes(initial.size() * sizeof(initial[0]));
    std::memcpy(bytes.data(), initial.data(), bytes.size());
    state.color_attachment_load = StoreNewArray(pool, bytes);
    state.color_attachment_load_bytes = bytes.size();
    state.color_attachment_load_enable = 1;
  }
  state.fragment_invocations = StoreNewArray(pool, invocations);
  state.fragment_outputs = StoreNewArray(pool, outputs);
  const PoolHandle handle = pool.Allocate(sizeof(PipelineState));
  StorePipelineState(pool, handle, state);
  input.write({handle, static_cast<std::uint32_t>(sequence), sequence});
  sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_NS));
  PipelineTxn completed;
  Check(output.nb_read(completed) && completed.sequence == sequence,
        "PBE counter FIFO completion");
  state = LoadPipelineState(pool, handle);
  const std::uint64_t sample_colors = covered_samples * targets;
  const std::uint64_t blend_colors = blend && !integer ? sample_colors : 0;
  const std::uint64_t serialized = samples * targets;
  const std::uint64_t expected_cycles = kReferenceUarch.pbe_base_cycles +
      CeilDivide(serialized, kReferenceUarch.pbe_pixels_per_batch) +
      CeilDivide(blend_colors, kReferenceUarch.pbe_blend_fragments_per_batch);
  Check(state.counters.pbe_fragment_writes == (color_mask ? sample_colors : 0),
        "PBE writes must count covered sample colors and enabled channel masks");
  Check(state.counters.pbe_color_reads == blend_colors &&
            state.counters.pbe_blended_fragments == blend_colors,
        "PBE blend counters must count sample/attachment work and integer bypass");
  Check(state.counters.pbe_pixels_written == serialized,
        "PBE serialization includes untouched samples and every attachment");
  Check(state.counters.pbe_cycles == expected_cycles &&
            state.counters.renderer_cycles == expected_cycles + 7,
        "PBE cycles must use actual blend and serialization sample colors");
  Check(state.counters.ps_invocations == invocations.size(),
        "PBE must preserve pixel-frequency shader invocation count");
  if (integer) {
    const auto bytes = LoadArray<std::uint8_t>(pool, state.pbe_framebuffer);
    for (std::uint32_t sample = 0; sample < samples; ++sample) {
      for (std::uint32_t channel = 0; channel < 4; ++channel) {
        const std::size_t index = sample * 4 + channel;
        std::uint32_t actual;
        std::memcpy(&actual, bytes.data() + index * sizeof(actual), sizeof(actual));
        const std::uint32_t expected = !empty && (color_mask & (1U << channel))
            ? 0xf1234560U + channel : 0x80000000U + index;
        Check(actual == expected, "integer LOAD must retain each masked sample lane");
      }
    }
  }
  // This focused test stops before PbeWriteBack, which normally retires the
  // extra pre-memory attachment handles.
  for (std::uint32_t target = 1; target < targets; ++target)
    pool.Release(state.extra_pbe_framebuffer[target - 1]);
  ReleaseFunctionalPayloads(pool, state);
  pool.Release(handle);
}
} // namespace

int sc_main(int, char **) {
  try {
    MemoryPool pool;
    sc_core::sc_fifo<PipelineTxn> input("input", 1);
    sc_core::sc_fifo<PipelineTxn> output("output", 1);
    Pbe pbe("pbe", pool);
    pbe.input(input);
    pbe.output(output);
    std::uint64_t sequence = 0;
    for (std::uint32_t samples : {1U, 2U, 4U, 8U, 16U})
      for (std::uint32_t targets : {1U, 2U, 4U})
        for (bool blend : {false, true})
          for (std::uint8_t mask : {0xf, 0x5, 0x0})
            for (bool integer : {false, true})
              for (bool empty : {false, true})
                RunCase(pool, input, output, ++sequence, samples, targets,
                        blend, mask, integer, empty);
    Check(pool.bytes_in_flight() == 0 && pool.allocations() == pool.releases(),
          "PBE counter MemoryPool balance");
    std::cout << "pbe_sample_counter_test: PASS (" << sequence << " cases)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "pbe_sample_counter_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
