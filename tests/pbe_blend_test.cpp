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
                           bool blend_enabled = true,
                           BlendEquation rgb_eq = BlendEquation::kAdd,
                           BlendEquation alpha_eq = BlendEquation::kAdd,
                           BlendFactor src_rgb_f = BlendFactor::kSourceAlpha,
                           BlendFactor dst_rgb_f = BlendFactor::kOneMinusSourceAlpha,
                           std::uint8_t color_mask = 0x0f,
                           BlendFactor src_alpha_f = BlendFactor::kSourceAlpha,
                           BlendFactor dst_alpha_f =
                               BlendFactor::kOneMinusSourceAlpha) {
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
  state.raster_state.blend.rgb_equation = rgb_eq;
  state.raster_state.blend.alpha_equation = alpha_eq;
  state.raster_state.blend.source_rgb_factor = src_rgb_f;
  state.raster_state.blend.destination_rgb_factor = dst_rgb_f;
  state.raster_state.blend.source_alpha_factor = src_alpha_f;
  state.raster_state.blend.destination_alpha_factor = dst_alpha_f;
  state.raster_state.color_mask = color_mask;
  state.fragment_invocations = StoreNewArray(pool, invocations);
  state.fragment_outputs = StoreNewArray(pool, outputs);

  const PoolHandle state_handle = pool.Allocate(sizeof(PipelineState));
  StorePipelineState(pool, state_handle, state);
  return {state_handle, state.fragment_invocations, state.fragment_outputs};
}

void SetColorAttachmentLoad(MemoryPool &pool,
                            const TestStateHandles &handles,
                            const std::array<std::uint8_t, 4> &rgba) {
  PipelineState state = LoadPipelineState(pool, handles.state);
  Check(!pvrgpu::stub::HasPoolHandle(state.color_attachment_load),
        "duplicate color attachment LOAD");
  state.color_attachment_load = StoreNewArray(
      pool, std::vector<std::uint8_t>(rgba.begin(), rgba.end()));
  state.color_attachment_load_enable = 1;
  state.color_attachment_load_bytes = rgba.size();
  StorePipelineState(pool, handles.state, state);
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
  if (pvrgpu::stub::HasPoolHandle(state.color_attachment_load))
    pool.Release(state.color_attachment_load);
  pool.Release(handles.invocations);
  pool.Release(handles.outputs);
  pool.Release(handles.state);
}

} // namespace

int sc_main(int, char **) {
  try {
    MemoryPool pool;
    sc_core::sc_fifo<PipelineTxn> input("input", 8);
    sc_core::sc_fifo<PipelineTxn> output("output", 8);
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
        FloatBits(0.3F), FloatBits(0.5F / 255.0F),
        FloatBits(2.5F / 255.0F), FloatBits(3.5F / 255.0F)};
    const TestStateHandles ties =
        MakeState(pool, 4, {tie_quantization}, false);

    // GLES 3.x Subtract Blending Equation test
    const TestStateHandles subtract_test =
        MakeState(pool, 5, {red_half}, true,
                  BlendEquation::kSubtract, BlendEquation::kAdd,
                  BlendFactor::kOne, BlendFactor::kOne);

    // GLES 3.x Color Write Mask test (write red and blue only, keep green and alpha)
    const TestStateHandles colormask_test =
        MakeState(pool, 6, {red_half}, false,
                  BlendEquation::kAdd, BlendEquation::kAdd,
                  BlendFactor::kOne, BlendFactor::kZero,
                  0x05); // write mask Red(0b0001) | Blue(0b0100) = 0x05

    // GLES 3.x Max Blending Equation test
    const TestStateHandles max_test =
        MakeState(pool, 7, {red_half}, true,
                  BlendEquation::kMax, BlendEquation::kAdd);

    // API-v7 attachment aliasing must initialize the blend destination from
    // the prior DRAM color attachment. Terrain's continuation pass uses
    // ADD/SRC_ALPHA/ONE independently for RGB and alpha.
    const TestStateHandles loaded_additive =
        MakeState(pool, 8, {red_half}, true, BlendEquation::kAdd,
                  BlendEquation::kAdd, BlendFactor::kSourceAlpha,
                  BlendFactor::kOne, 0x0f, BlendFactor::kSourceAlpha,
                  BlendFactor::kOne);
    SetColorAttachmentLoad(pool, loaded_additive, {10, 20, 30, 40});

    input.write({single.state, 1, 1});
    input.write({red_green.state, 2, 2});
    input.write({green_red.state, 3, 3});
    input.write({ties.state, 4, 4});
    input.write({subtract_test.state, 5, 5});
    input.write({colormask_test.state, 6, 6});
    input.write({max_test.state, 7, 7});
    input.write({loaded_additive.state, 8, 8});

    sc_core::sc_start(sc_core::sc_time(200, sc_core::SC_NS));
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
    Check(output.nb_read(completed) && completed.sequence == 5,
          "subtract-test FIFO completion order");
    Check(output.nb_read(completed) && completed.sequence == 6,
          "colormask-test FIFO completion order");
    Check(output.nb_read(completed) && completed.sequence == 7,
          "max-test FIFO completion order");
    Check(output.nb_read(completed) && completed.sequence == 8,
          "attachment-LOAD FIFO completion order");

    CheckResult(pool, single, {128, 0, 127, 191}, 1);
    CheckResult(pool, red_green, {64, 128, 63, 159}, 2);
    CheckResult(pool, green_red, {128, 64, 63, 159}, 2);
    CheckResult(pool, ties, {77, 1, 3, 4}, 1, false);
    CheckResult(pool, subtract_test, {255, 0, 0, 191}, 1);
    CheckResult(pool, colormask_test, {255, 0, 0, 255}, 1, false);
    CheckResult(pool, max_test, {255, 0, 255, 191}, 1);
    CheckResult(pool, loaded_additive, {138, 20, 30, 104}, 1);
    Check(pool.bytes_in_flight() == 0 && pool.allocations() == pool.releases(),
          "MemoryPool balanced");
    std::cout << "pbe_blend_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "pbe_blend_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
