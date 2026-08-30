/*
 * SystemC PBE blend regression.
 *
 * PBE means Pixel Back End and ISS means Instruction Set Simulator. The test
 * first decodes and executes exact Mesa-generated public PowerVR PCO fragment
 * binaries, then passes those PIXOUT values through the event-driven PBE FIFO.
 * It verifies RGBA8 destination read/modify/write, per-fragment quantization,
 * and API ordering without a shader-name or expected-image shortcut.
 */
#include "common/functional_types.h"
#include "common/pipeline_state.h"
#include "fragment/pbe.h"
#include "shader/pco_iss.h"

#include <systemc>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using pvrgpu::stub::BlendEquation;
using pvrgpu::stub::BlendFactor;
using pvrgpu::stub::FragmentInvocation;
using pvrgpu::stub::FragmentOutput;
using pvrgpu::stub::FunctionalCase;
using pvrgpu::stub::LoadArray;
using pvrgpu::stub::LoadPipelineState;
using pvrgpu::stub::MemoryPool;
using pvrgpu::stub::Pbe;
using pvrgpu::stub::PcoFragmentExecution;
using pvrgpu::stub::PipelineStage;
using pvrgpu::stub::PipelineState;
using pvrgpu::stub::PipelineTxn;
using pvrgpu::stub::PoolHandle;
using pvrgpu::stub::ShaderStage;
using pvrgpu::stub::StoreNewArray;
using pvrgpu::stub::StorePipelineState;

void Check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error("PBE blend test failed: " + message);
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

PcoFragmentExecution ExecuteFixture(const std::vector<std::uint8_t> &binary) {
  const auto decoded = pvrgpu::stub::Decode(ShaderStage::kFragment, binary);
  return pvrgpu::stub::ExecuteFragment(decoded.summary, decoded.instructions);
}

FragmentInvocation MakeInvocation(std::uint64_t ordinal) {
  FragmentInvocation invocation;
  invocation.x = 0;
  invocation.y = 0;
  invocation.primitive_id = static_cast<std::uint32_t>(ordinal);
  invocation.parameter_index = static_cast<std::uint32_t>(ordinal);
  invocation.submit_ordinal = ordinal;
  invocation.quad_id = 0;
  invocation.quad_lane = 0;
  invocation.sample_mask = 1;
  return invocation;
}

FragmentOutput MakeOutput(const FragmentInvocation &invocation,
                          const PcoFragmentExecution &execution) {
  FragmentOutput output;
  output.x = invocation.x;
  output.y = invocation.y;
  output.primitive_id = invocation.primitive_id;
  output.parameter_index = invocation.parameter_index;
  output.submit_ordinal = invocation.submit_ordinal;
  output.written_mask = execution.written_mask;
  for (std::size_t component = 0; component < 4; ++component)
    output.pixel_output[component] = execution.pixel_outputs[component];
  return output;
}

struct TestStateHandles {
  PoolHandle state;
  PoolHandle invocations;
  PoolHandle outputs;
};

TestStateHandles MakeState(MemoryPool &pool, std::uint32_t sequence,
                           const std::vector<PcoFragmentExecution> &executions,
                           bool blend_enabled = true) {
  std::vector<FragmentInvocation> invocations;
  std::vector<FragmentOutput> outputs;
  for (std::size_t index = 0; index < executions.size(); ++index) {
    const FragmentInvocation invocation = MakeInvocation(index + 1);
    invocations.push_back(invocation);
    outputs.push_back(MakeOutput(invocation, executions[index]));
  }

  PipelineState state;
  state.width = 1;
  state.height = 1;
  state.sequence = sequence;
  state.functional_case = blend_enabled ? FunctionalCase::kFillSolidBlended
                                        : FunctionalCase::kFillSolid;
  state.stage = PipelineStage::kTextureComplete;
  state.active_fragment_invocations =
      static_cast<std::uint32_t>(executions.size());
  state.raster_state.clear_color[0] = 0.0F;
  state.raster_state.clear_color[1] = 0.0F;
  state.raster_state.clear_color[2] = 1.0F;
  state.raster_state.clear_color[3] = 1.0F;
  state.raster_state.blend.enable = blend_enabled ? 1 : 0;
  state.raster_state.blend.rgb_equation = BlendEquation::kAdd;
  state.raster_state.blend.alpha_equation = BlendEquation::kAdd;
  state.raster_state.blend.source_rgb_factor = BlendFactor::kSourceAlpha;
  state.raster_state.blend.destination_rgb_factor =
      BlendFactor::kOneMinusSourceAlpha;
  state.raster_state.blend.source_alpha_factor = BlendFactor::kSourceAlpha;
  state.raster_state.blend.destination_alpha_factor =
      BlendFactor::kOneMinusSourceAlpha;
  state.fragment_invocations = StoreNewArray(pool, invocations);
  state.fragment_outputs = StoreNewArray(pool, outputs);

  const PoolHandle state_handle = pool.Allocate(sizeof(PipelineState));
  StorePipelineState(pool, state_handle, state);
  return {state_handle, state.fragment_invocations, state.fragment_outputs};
}

void CheckResult(MemoryPool &pool, const TestStateHandles &handles,
                 const std::array<std::uint8_t, 4> &expected,
                 std::uint64_t expected_fragments,
                 bool blend_enabled = true) {
  const PipelineState state = LoadPipelineState(pool, handles.state);
  Check(state.stage == PipelineStage::kPbeComplete, "PBE completion stage");
  Check(state.counters.pbe_color_reads ==
            (blend_enabled ? expected_fragments : 0),
        "destination read counter");
  Check(state.counters.pbe_blended_fragments ==
            (blend_enabled ? expected_fragments : 0),
        "blend operation counter");
  Check(state.counters.pbe_fragment_writes == expected_fragments,
        "fragment write counter");
  Check(state.counters.pbe_pixels_written == 1,
        "one-pixel surface serialization counter");
  const std::vector<std::uint8_t> framebuffer =
      LoadArray<std::uint8_t>(pool, state.pbe_framebuffer);
  Check(framebuffer == std::vector<std::uint8_t>(expected.begin(), expected.end()),
        "RGBA8 result");

  pool.Release(state.pbe_framebuffer);
  pool.Release(handles.invocations);
  pool.Release(handles.outputs);
  pool.Release(handles.state);
}

} // namespace

int sc_main(int, char **) {
  try {
    MemoryPool pool;
    sc_core::sc_fifo<PipelineTxn> input("input", 4);
    sc_core::sc_fifo<PipelineTxn> output("output", 4);
    Pbe pbe("pbe", pool);
    pbe.input(input);
    pbe.output(output);

    const PcoFragmentExecution red_half = ExecuteFixture(
        pvrgpu::stub::FillSolidRedHalfAlphaFragmentPcoBinary());
    const PcoFragmentExecution green_half = ExecuteFixture(
        pvrgpu::stub::FillSolidGreenHalfAlphaFragmentPcoBinary());

    const TestStateHandles single = MakeState(pool, 1, {red_half});
    const TestStateHandles red_green =
        MakeState(pool, 2, {red_half, green_half});
    const TestStateHandles green_red =
        MakeState(pool, 3, {green_half, red_half});
    PcoFragmentExecution tie_quantization;
    tie_quantization.written_mask = 0x0f;
    tie_quantization.pixel_outputs = {
        FloatBits(0.5F / 255.0F), FloatBits(1.5F / 255.0F),
        FloatBits(2.5F / 255.0F), FloatBits(3.5F / 255.0F)};
    const TestStateHandles ties =
        MakeState(pool, 4, {tie_quantization}, false);
    input.write({single.state, 1, 1});
    input.write({red_green.state, 2, 2});
    input.write({green_red.state, 3, 3});
    input.write({ties.state, 4, 4});

    sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_NS));
    sc_core::sc_start(sc_core::SC_ZERO_TIME);

    PipelineTxn completed;
    Check(output.nb_read(completed) && completed.sequence == 1,
          "single-fragment FIFO completion order");
    Check(output.nb_read(completed) && completed.sequence == 2,
          "red-green FIFO completion order");
    Check(output.nb_read(completed) && completed.sequence == 3,
          "green-red FIFO completion order");
    Check(output.nb_read(completed) && completed.sequence == 4,
          "tie-quantization FIFO completion order");
    CheckResult(pool, single, {128, 0, 127, 191}, 1);
    CheckResult(pool, red_green, {64, 128, 63, 159}, 2);
    CheckResult(pool, green_red, {128, 64, 63, 159}, 2);
    CheckResult(pool, ties, {0, 2, 2, 4}, 1, false);
    Check(pool.bytes_in_flight() == 0 && pool.allocations() == pool.releases(),
          "MemoryPool balanced");
    std::cout << "pbe_blend_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "pbe_blend_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
