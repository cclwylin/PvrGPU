// SPDX-License-Identifier: MIT
// Real USC -> TextureUnit FIFO regression for bounded host register residency.
// The synthetic shader runs every helper, two native SMP continuations and
// derivatives on both sides of a texture fence. No captured output is an input.
#include "common/pipeline_state.h"
#include "memory/gpu_memory_system.h"
#include "shader/pco_iss.h"
#include "shader/usc_cluster.h"
#include "texture/texture_unit.h"

#include <systemc>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace pvrgpu::stub;
unsigned checks = 0;
void Check(bool ok, const char *message) {
  ++checks;
  if (!ok) throw std::runtime_error(message);
}
std::uint32_t Bits(float value) {
  std::uint32_t bits; std::memcpy(&bits, &value, 4); return bits;
}
constexpr std::uint64_t kImageAddress = UINT64_C(0x7100000000);
constexpr std::size_t kQuadCap = 256;
constexpr std::array<const char *, 13> kProgramMutations{
  "program-stage", "program-bytes", "program-groups", "program-instructions",
  "program-vertex-input", "program-vertex-output", "program-pixel-output",
  "program-early-hsr", "program-depth", "program-derivatives", "program-ends-task",
  "program-handle-slot", "program-handle-generation"
};

const char *ExpectedFailure(const std::string &mode) {
  if (mode == "duplicate-task" || mode == "task-out-of-range")
    return "fragment residency repeats or loses a quad";
  if (mode == "duplicate-lane" || mode == "lane-out-of-range")
    return "fragment residency repeats or loses a shader lane";
  if (mode == "missing-lane")
    return "fragment residency did not execute every quad/lane";
  if (mode == "missing-quad")
    return "texture fragment USC task/shared count mismatch";
  if (mode == "overlapping-mask" || mode == "high-coverage-mask" ||
      mode == "high-helper-mask" || mode == "high-write-mask" || mode == "empty-mask")
    return "fragment residency received an invalid quad mask";
  if (mode == "partial-derivative-quad")
    return "derivative quad is incomplete or exceeds instruction budget";
  if (mode == "coefficient-offset" || mode == "coefficient-span")
    return "texture fragment USC coefficient task is out of range";
  if (mode == "output-collision")
    return "texture fragment USC visible-lane mapping is invalid";
  if (std::find(kProgramMutations.begin(), kProgramMutations.end(), mode) != kProgramMutations.end())
    return "texture fragment USC response changed its immutable program";
  throw std::invalid_argument("unknown residency rejection mode: " + mode);
}

std::vector<PcoInstruction> Program(bool atomics) {
  std::vector<PcoInstruction> code;
  const auto emit = [&](PcoInstruction instruction) { code.push_back(instruction); };
  const auto wait = [&] {
    PcoInstruction op; op.opcode = PcoOpcode::kWaitDataFence;
    op.source_count = 0; emit(op);
  };
  const auto immediate = [&](unsigned reg, std::uint32_t value) {
    PcoInstruction op; op.opcode = PcoOpcode::kMoveImmediate;
    op.target = PcoWriteTarget::kTemporary; op.source_count = 0;
    op.output_index = reg; op.immediate = value; emit(op);
  };
  const auto derivative = [&](bool y, unsigned source, unsigned dest) {
    PcoInstruction op;
    op.opcode = y ? PcoOpcode::kDerivativeY : PcoOpcode::kDerivativeX;
    op.derivative_fine = 1; op.target = PcoWriteTarget::kTemporary;
    op.source = {PcoRegisterBank::kTemporary, static_cast<std::uint16_t>(source)};
    op.output_index = dest; emit(op);
  };
  const auto atomic = [&](unsigned dest) {
    PcoInstruction op; op.opcode = PcoOpcode::kAtomicAdd32;
    op.target = PcoWriteTarget::kTemporary; op.source_count = 3;
    op.source = {PcoRegisterBank::kTemporary, 20};
    op.source1 = {PcoRegisterBank::kTemporary, 21};
    op.source2 = {PcoRegisterBank::kTemporary, 22};
    op.output_index = dest; emit(op); wait();
  };
  if (atomics) {
    // Native ALPHAF rejects right-hand lanes without removing their helper
    // ALU/derivative/texture work. Only live lane 0 may update the image.
    immediate(18, Bits(-.5F));
    PcoInstruction add; add.opcode = PcoOpcode::kFloatAdd;
    add.target = PcoWriteTarget::kTemporary; add.output_index = 19;
    add.source = {PcoRegisterBank::kSpecial, 97};
    add.source1 = {PcoRegisterBank::kTemporary, 18}; add.source_count = 2; emit(add);
    PcoInstruction compare; compare.opcode = PcoOpcode::kBooleanCompare;
    compare.source = {PcoRegisterBank::kTemporary, 19}; compare.writes_predicate = 1;
    emit(compare);
    PcoInstruction feedback; feedback.opcode = PcoOpcode::kAlphaFeedback;
    feedback.source_count = 0; feedback.exec_cnd = 1; emit(feedback); wait();
    immediate(20, static_cast<std::uint32_t>(kImageAddress));
    immediate(21, static_cast<std::uint32_t>(kImageAddress >> 32));
    immediate(22, 1); atomic(23);
  }
  PcoInstruction interpolate; interpolate.opcode = PcoOpcode::kFloatInterpolate;
  interpolate.target = PcoWriteTarget::kTemporary;
  interpolate.source = {PcoRegisterBank::kCoefficient, 0};
  interpolate.component_count = 2; emit(interpolate); wait();
  derivative(false, 0, 2);
  for (unsigned set = 0; set < 2; ++set) {
    PcoInstruction sample; sample.opcode = PcoOpcode::kTextureSample;
    sample.target = PcoWriteTarget::kTemporary; sample.source_count = 3;
    sample.source = {PcoRegisterBank::kTemporary, 0};
    sample.source1 = {PcoRegisterBank::kShared, static_cast<std::uint16_t>(set * 20)};
    sample.source2 = {PcoRegisterBank::kShared, static_cast<std::uint16_t>(set * 20 + 8)};
    sample.component_count = 4; sample.output_index = 4 + set * 4;
    emit(sample); wait();
    derivative(set != 0, set == 0 ? 4 : 9, 12 + set);
  }
  if (atomics) atomic(24);
  const std::array<unsigned, 4> sources = atomics
      ? std::array<unsigned, 4>{23, 24, 8, 2}
      : std::array<unsigned, 4>{8, 12, 13, 2};
  for (unsigned channel = 0; channel < 4; ++channel) {
    PcoInstruction op; op.opcode = PcoOpcode::kMoveBypass;
    op.target = PcoWriteTarget::kPixelOutput; op.output_index = channel;
    op.source = {PcoRegisterBank::kTemporary, static_cast<std::uint16_t>(sources[channel])};
    emit(op);
  }
  for (std::size_t i = 0; i < code.size(); ++i) {
    code[i].binary_offset = static_cast<std::uint32_t>(1 + i * 8);
    code[i].group_index = static_cast<std::uint16_t>(i);
  }
  code.back().end_group = 1;
  return code;
}

std::uint8_t Color(unsigned set, unsigned mip, unsigned x, unsigned y, unsigned channel) {
  return static_cast<std::uint8_t>(17 + set * 39 + mip * 61 + x * 5 + y * 9 + channel * 11);
}
float Sample(unsigned set, unsigned quad, unsigned lane, unsigned channel) {
  quad ^= (quad >> 4U) ^ (quad >> 8U);
  const unsigned x = (quad % 4) / 2 + (lane & 1U);
  const unsigned y = ((quad / 4) % 4) / 2 + (lane >> 1U);
  return static_cast<float>(Color(set, 1, x, y, channel)) / 255.F;
}

class Harness final : public sc_core::sc_module {
public:
  Harness(sc_core::sc_module_name name, MemoryMode mode)
      : sc_module(name), memory_(mode), cluster_("cluster", pool_, ShaderStage::kFragment, &memory_),
        texture_("texture", pool_, &memory_) {
    cluster_.input(input_); cluster_.output(output_);
    cluster_.texture_request_output(cluster_requests_);
    cluster_.texture_response_input(cluster_responses_);
    texture_.input(texture_input_); texture_.output(texture_output_);
    texture_.sample_input(texture_requests_); texture_.sample_output(texture_responses_);
    SC_THREAD(Relay);
  }

  void Run(unsigned quad_count, bool atomics, bool reverse_tasks,
           const std::string &failure_mode = {}) {
    Check(quad_count != 0, "residency fixture is not empty");
    failure_mode_ = failure_mode;
    fifo_batches_ = largest_batch_ = 0;
    const auto program = Program(atomics);
    PipelineState state;
    state.sequence = ++sequence_; state.width = state.height = 2;
    state.memory_mode = memory_.mode(); state.functional_case = FunctionalCase::kDriverPcoTriangles;
    state.stage = PipelineStage::kFragmentIssued;
    state.fragment_program_summary.stage = ShaderStage::kFragment;
    state.fragment_program_summary.binary_size = static_cast<std::uint32_t>(program.size() * 8);
    state.fragment_program_summary.group_count = static_cast<std::uint32_t>(program.size());
    state.fragment_program_summary.instruction_count = static_cast<std::uint32_t>(program.size());
    state.fragment_program_summary.pixel_output_mask = 15;
    state.fragment_program_summary.early_hsr_safe = atomics ? 0 : 1;
    state.fragment_program_summary.uses_derivatives = 1;
    state.fragment_pco_abi.temps = 25; state.fragment_pco_abi.shareds = 40;
    state.fragment_pco_abi.coefficients = 8;
    state.position_output_count = state.fragment_position_count = 4;
    state.varying_output_start = state.fragment_varying_start = 4;
    state.varying_output_count = 1; state.fragment_varying_count = 4;
    state.sampled_texture_count = 2;
    state.active_fragment_invocations = quad_count * 2;
    state.fragment_shader_lane_count = quad_count * 4;
    state.fragment_groups = quad_count; state.counters.drawlists = 1;
    state.counters.ps_invocations = quad_count * 2;
    state.fragment_instructions = StoreNewArray(pool_, program);
    DrawListStats stats; stats.fragment.program_recorded = 1;
    const auto counts = CountPcoInstructions(program, false);
    stats.fragment.program_groups = stats.fragment.program_instructions = program.size();
    stats.fragment.program_alu_instructions = counts.alu;
    stats.fragment.program_tex_instructions = counts.texture;
    stats.fragment.program_memory_instructions = counts.memory;
    state.drawlist_stats = StoreNewArray(pool_, std::vector<DrawListStats>{stats});

    std::vector<FragmentInvocation> invocations(quad_count * 2);
    std::vector<FragmentShaderLane> lanes(quad_count * 4);
    std::vector<FragmentQuad> quads(quad_count);
    std::vector<UscFragmentTask> tasks(quad_count);
    std::vector<std::uint32_t> coefficients(quad_count * 12, UINT32_C(0x7fc00000));
    for (unsigned q = 0; q < quad_count; ++q) {
      auto &quad = quads[q]; quad.quad_id = 1000 + q; quad.parameter_index = q;
      quad.submit_ordinal = 7 + q; quad.coverage_mask = quad.write_mask = 9; quad.helper_mask = 6;
      tasks[q].fragment_quad_index = q; tasks[q].first_coefficient_dword = q * 12 + 4;
      tasks[q].coefficient_dword_count = 8;
      const auto c = tasks[q].first_coefficient_dword;
      const unsigned tag = q ^ (q >> 4U) ^ (q >> 8U);
      coefficients[c] = Bits(.25F); coefficients[c + 1] = 0;
      coefficients[c + 2] = Bits((static_cast<float>(tag % 4) + .5F) / 8.F); coefficients[c + 3] = 0;
      coefficients[c + 4] = 0; coefficients[c + 5] = Bits(.25F);
      coefficients[c + 6] = Bits((static_cast<float>((tag / 4) % 4) + .5F) / 8.F); coefficients[c + 7] = 0;
      for (unsigned lane = 0; lane < 4; ++lane) {
        const unsigned global = q * 4 + lane;
        auto &shader = lanes[global]; quad.invocation_indices[lane] = global;
        shader.x = lane & 1U; shader.y = lane >> 1U; shader.primitive_id = 500 + q;
        shader.parameter_index = q; shader.submit_ordinal = quad.submit_ordinal;
        shader.quad_id = quad.quad_id; shader.quad_lane = lane;
        shader.helper = lane == 1 || lane == 2; shader.sample_mask = shader.helper ? 0 : 1;
        if (!shader.helper) {
          const auto visible = 2 * (quad_count - 1 - q) + (lane == 3 ? 1 : 0);
          shader.visible_invocation_index = visible;
          auto &inv = invocations[visible]; inv.x = shader.x; inv.y = shader.y;
          inv.primitive_id = shader.primitive_id; inv.parameter_index = shader.parameter_index;
          inv.submit_ordinal = shader.submit_ordinal; inv.quad_id = shader.quad_id;
          inv.quad_lane = shader.quad_lane; inv.sample_mask = 1;
        }
      }
    }
    if (reverse_tasks) std::reverse(tasks.begin(), tasks.end());
    if (!failure_mode.empty()) {
      (void)ExpectedFailure(failure_mode);
      Check(quad_count == kQuadCap + 1 && !atomics && !reverse_tasks,
            "negative fixture must cross one complete resident batch");
      // Corrupt the first item in the second resident batch where possible.
      // Completed earlier work must never turn this malformed transaction into
      // a successful final output. The orphan tests cover global completeness.
      const auto last = quad_count - 1;
      if (failure_mode == "duplicate-task") tasks[last].fragment_quad_index = 0;
      if (failure_mode == "duplicate-lane") quads[last].invocation_indices[0] = 0;
      if (failure_mode == "missing-lane") lanes.push_back(lanes.back());
      if (failure_mode == "missing-quad") tasks.pop_back();
      if (failure_mode == "lane-out-of-range")
        quads[last].invocation_indices[0] = static_cast<std::uint32_t>(lanes.size());
      if (failure_mode == "task-out-of-range")
        tasks[last].fragment_quad_index = static_cast<std::uint32_t>(quads.size());
      if (failure_mode == "overlapping-mask") quads[last].helper_mask |= 1;
      if (failure_mode == "high-coverage-mask") {
        quads[last].coverage_mask |= 0x10; quads[last].write_mask |= 0x10;
      }
      if (failure_mode == "high-helper-mask") quads[last].helper_mask |= 0x80;
      if (failure_mode == "high-write-mask") quads[last].write_mask |= 0x40;
      if (failure_mode == "empty-mask") {
        quads[last].coverage_mask = quads[last].helper_mask = quads[last].write_mask = 0;
      }
      if (failure_mode == "partial-derivative-quad") {
        quads[last].helper_mask &= ~4U;
        quads[last].invocation_indices[2] = kInvalidFragmentInvocationIndex;
      }
      if (failure_mode == "coefficient-offset")
        tasks[last].first_coefficient_dword = static_cast<std::uint32_t>(coefficients.size() + 1);
      if (failure_mode == "coefficient-span")
        tasks[last].first_coefficient_dword = static_cast<std::uint32_t>(coefficients.size() - 4);
      if (failure_mode == "output-collision")
        lanes[last * 4].visible_invocation_index = lanes[0].visible_invocation_index;
      state.fragment_groups = static_cast<std::uint32_t>(tasks.size());
      state.fragment_shader_lane_count = static_cast<std::uint32_t>(lanes.size());
    }
    expected_lane_count_ = static_cast<unsigned>(lanes.size());
    requests_per_lane_.assign(lanes.size(), 0);
    state.fragment_invocations = StoreNewArray(pool_, invocations);
    state.fragment_shader_lanes = StoreNewArray(pool_, lanes);
    state.fragment_quads = StoreNewArray(pool_, quads);
    state.usc_fragment_tasks = StoreNewArray(pool_, tasks);
    state.usc_coefficient_banks = StoreNewArray(pool_, coefficients);

    std::vector<TextureResource> resources(2);
    std::vector<SamplerState> samplers(2);
    std::vector<std::uint32_t> shared(40, 0);
    std::array<std::vector<std::uint8_t>, 2> original;
    for (unsigned set = 0; set < 2; ++set) {
      auto &resource = resources[set];
      resource.descriptor_set = set; samplers[set].descriptor_set = set;
      resource.gpu_address = UINT64_C(0x7000000000) + sequence_ * 0x10000 + set * 0x1000;
      resource.format = TextureFormat::kRgba8Unorm; resource.mip_count = 2;
      resource.mip[0] = {8, 8, 32, 0}; resource.mip[1] = {4, 4, 16, 256}; resource.byte_size = 320;
      original[set].resize(384, 0xad);
      for (unsigned mip = 0; mip < 2; ++mip)
        for (unsigned y = 0; y < resource.mip[mip].height; ++y)
          for (unsigned x = 0; x < resource.mip[mip].width; ++x)
            for (unsigned channel = 0; channel < 4; ++channel)
              original[set][resource.mip[mip].offset_bytes + y * resource.mip[mip].row_pitch_bytes + x * 4 + channel] =
                  Color(set, mip, x, y, channel);
      memory_.HostWrite(resource.gpu_address, original[set].data(), original[set].size());
      samplers[set].max_lod_u4_6 = 64;
      const std::uint64_t image0 = UINT64_C(4) | (UINT64_C(3) << 5) | (UINT64_C(2) << 8) |
          (UINT64_C(1) << 11) | (UINT64_C(12) << 27) | (UINT64_C(7) << 34) | (UINT64_C(7) << 48);
      const std::uint64_t image1 = ((resource.gpu_address >> 2) << 16) |
          (UINT64_C(2) << 60) | (UINT64_C(1) << 15) | 7;
      const std::uint64_t sampler = UINT64_C(0xfff) | (UINT64_C(64) << 23);
      shared[set * 20] = static_cast<std::uint32_t>(image0);
      shared[set * 20 + 1] = static_cast<std::uint32_t>(image0 >> 32);
      shared[set * 20 + 2] = static_cast<std::uint32_t>(image1);
      shared[set * 20 + 3] = static_cast<std::uint32_t>(image1 >> 32);
      shared[set * 20 + 4] = resource.byte_size;
      shared[set * 20 + 8] = static_cast<std::uint32_t>(sampler);
      shared[set * 20 + 9] = static_cast<std::uint32_t>(sampler >> 32);
      const auto gather = sampler | (UINT64_C(1) << 36) | (UINT64_C(1) << 38);
      shared[set * 20 + 16] = static_cast<std::uint32_t>(gather);
      shared[set * 20 + 17] = static_cast<std::uint32_t>(gather >> 32);
    }
    state.texture_resources = StoreNewArray(pool_, resources);
    state.sampler_states = StoreNewArray(pool_, samplers);
    state.fragment_shared_registers = StoreNewArray(pool_, shared);
    if (atomics) {
      ShaderImageResource image;
      image.resource_token = 5; image.gpu_address = kImageAddress; image.bytes = 4;
      image.format = 1; image.access = 3; image.width = image.height = image.depth = 1;
      image.row_stride = image.layer_stride = image.texel_bytes = 4;
      const std::array<std::uint8_t, 4> zero{};
      memory_.HostWrite(kImageAddress, zero.data(), zero.size());
      state.fragment_image_resources = StoreNewArray(pool_, std::vector<ShaderImageResource>{image});
    }
    const auto handle = pool_.Allocate(sizeof(PipelineState));
    StorePipelineState(pool_, handle, state);
    PipelineTxn transaction; transaction.state = handle;
    transaction.sequence = sequence_; transaction.frame = static_cast<std::uint32_t>(sequence_);
    input_.write(transaction);
    try {
      sc_core::sc_start(sc_core::sc_time(1, sc_core::SC_MS));
    } catch (const std::exception &error) {
      if (failure_mode.empty()) throw;
      const std::string message = error.what();
      Check(message.find(ExpectedFailure(failure_mode)) != std::string::npos,
            "malformed input must be rejected at its exact production validation gate");
      const auto failed = LoadPipelineState(pool_, handle);
      Check(output_.num_available() == 0 && failed.stage != PipelineStage::kFragmentShaded &&
            !HasPoolHandle(failed.fragment_outputs),
            "rejected transaction must publish neither success nor fabricated fragment outputs");
      if (failure_mode != "missing-quad")
        Check(fifo_batches_ >= 2, "late corruption is checked after real work in an earlier resident batch");
      ReleaseFunctionalPayloads(pool_, failed); pool_.Release(handle);
      if (HasPoolHandle(retained_program_)) {
        pool_.Release(retained_program_); retained_program_ = {};
      }
      Check(pool_.bytes_in_flight() == 0 && pool_.allocations() == pool_.releases(),
            "failed transaction retains recoverable pool ownership");
      std::cout << "usc_fragment_residency_test: reject " << failure_mode
                << " PASS diagnostic=" << ExpectedFailure(failure_mode) << '\n';
      return;
    }
    Check(failure_mode.empty(), "malformed residency input was silently accepted");
    PipelineTxn completed;
    Check(output_.nb_read(completed) && completed.state.slot == handle.slot &&
          completed.state.generation == handle.generation, "bounded USC completes original transaction");
    const auto final = LoadPipelineState(pool_, handle);
    Check(final.stage == PipelineStage::kFragmentShaded && !HasPoolHandle(final.fragment_continuations) &&
          !HasPoolHandle(final.texture_sample_requests) && !HasPoolHandle(final.texture_sample_responses),
          "resident payloads retire before final publication");
    Check(std::all_of(requests_per_lane_.begin(), requests_per_lane_.end(), [](unsigned n) { return n == 2; }),
          "every original global shader lane samples each native site exactly once");
    Check(fifo_batches_ == 2 * ((quad_count + kQuadCap - 1) / kQuadCap) && largest_batch_ <= 4 * kQuadCap,
          "FIFO continuations are bounded independently of total frame lanes");
    Check(final.counters.texture_requests == quad_count * 8 && final.counters.texel_fetches == quad_count * 8 &&
          final.counters.fs_tex_instructions == quad_count * 8 && final.counters.ps_invocations == quad_count * 2,
          "native texel/instruction counts include all helpers without duplicating visible invocations");
    Check(final.counters.fs_alu_instructions == counts.alu * quad_count * 4,
          "native ALU work counts once across derivative and texture suspensions");
    Check(final.counters.fs_memory_instructions == (atomics ?
          (counts.memory - 2) * quad_count * 4 + quad_count * 2 : counts.memory * quad_count * 4),
          "helper and demoted image writes are suppressed without suppressing reads");
    const auto outputs = LoadArray<FragmentOutput>(pool_, final.fragment_outputs);
    Check(outputs.size() == invocations.size(), "helpers never allocate visible output slots");
    std::vector<std::uint32_t> tickets;
    for (unsigned index = 0; index < outputs.size(); ++index) {
      const auto &actual = outputs[index]; const auto &inv = invocations[index];
      const unsigned q = inv.parameter_index, lane = inv.quad_lane;
      Check(actual.x == inv.x && actual.y == inv.y && actual.parameter_index == q &&
            actual.primitive_id == inv.primitive_id && actual.submit_ordinal == inv.submit_ordinal &&
            actual.written_mask[0] == 15 && actual.discarded == (atomics && lane == 3),
            "visible output index and draw/primitive identity remain global");
      if (atomics) {
        if (lane == 0) {
          Check(actual.pixel_output[0] < actual.pixel_output[1], "per-invocation atomic ISA order is retained");
          tickets.push_back(actual.pixel_output[0]); tickets.push_back(actual.pixel_output[1]);
        } else Check(actual.pixel_output[0] == 0 && actual.pixel_output[1] == 0,
                     "demoted fragment produces no atomic transaction");
        Check(actual.pixel_output[2] == Bits(Sample(1, q, lane, 0)) && actual.pixel_output[3] == Bits(.25F),
              "atomics do not corrupt texture/derivative continuation values");
      } else {
        const float dx = Sample(0, q, lane | 1U, 0) - Sample(0, q, lane & ~1U, 0);
        const float dy = Sample(1, q, lane | 2U, 1) - Sample(1, q, lane & ~2U, 1);
        Check(actual.pixel_output[0] == Bits(Sample(1, q, lane, 0)) && actual.pixel_output[1] == Bits(dx) &&
              actual.pixel_output[2] == Bits(dy) && actual.pixel_output[3] == Bits(.25F),
              "real sampled pixel and fine derivatives match the independent storage oracle");
      }
    }
    if (atomics) {
      std::sort(tickets.begin(), tickets.end());
      for (unsigned i = 0; i < tickets.size(); ++i)
        Check(tickets[i] == i, "atomic returned tickets form a complete exactly-once permutation");
      const auto images = LoadArray<ShaderImageResource>(pool_, final.fragment_image_resources);
      const auto bytes = LoadArray<std::uint8_t>(pool_, images[0].readback);
      std::uint32_t value = 0; std::memcpy(&value, bytes.data(), 4);
      Check(value == quad_count * 2 && final.fragment_image_atomics == quad_count * 2 && final.fragment_images_complete,
            "image readback includes every live atomic and no helper side effect");
    }
    for (unsigned set = 0; set < 2; ++set)
      Check(memory_.Readback(resources[set].gpu_address, original[set].size(), MemoryClient::kFramebufferReadback).data == original[set],
            "sampled storage and guard bytes remain immutable");
    Check(LoadArray<FragmentShaderLane>(pool_, final.fragment_shader_lanes).size() == quad_count * 4,
          "external shader-lane payload retains global size");
    ReleaseFunctionalPayloads(pool_, final); pool_.Release(handle);
    Check(pool_.bytes_in_flight() == 0 && pool_.allocations() == pool_.releases(), "resident pool ownership balances");
  }

private:
  void Relay() {
    for (;;) {
      const auto txn = cluster_requests_.read();
      const auto state = LoadPipelineState(pool_, txn.state);
      const auto requests = LoadArray<TextureSampleRequest>(pool_, state.texture_sample_requests);
      const auto continuations = LoadArray<PcoFragmentContinuation>(pool_, state.fragment_continuations);
      Check(!requests.empty() && requests.size() <= kQuadCap * 4 && requests.size() % 4 == 0 &&
            requests.size() == continuations.size() && state.fragment_shader_lane_count == expected_lane_count_,
            "bounded FIFO carries complete derivative quads against global state");
      ++fifo_batches_; largest_batch_ = std::max(largest_batch_, requests.size());
      for (std::size_t i = 0; i < requests.size(); ++i) {
        const auto &request = requests[i];
        const auto lane = request.shader_lane_index;
        Check(lane < requests_per_lane_.size() && request.request_id == i && request.quad_id == 1000 + lane / 4 &&
              request.quad_lane == lane % 4 && (i == 0 || lane > requests[i - 1].shader_lane_index) &&
              request.descriptor_set == requests_per_lane_[lane] &&
              continuations[i].resume_instruction_index == continuations[0].resume_instruction_index,
              "request IDs are dense but shader/quad IDs remain original across resident batches");
        ++requests_per_lane_[lane];
      }
      texture_requests_.write(txn);
      const auto done = texture_responses_.read();
      Check(done.state.slot == txn.state.slot && done.state.generation == txn.state.generation,
            "actual TextureUnit response retains transaction ownership");
      if (fifo_batches_ == 3 &&
          std::find(kProgramMutations.begin(), kProgramMutations.end(), failure_mode_) != kProgramMutations.end()) {
        auto changed = LoadPipelineState(pool_, txn.state);
        auto &summary = changed.fragment_program_summary;
        if (failure_mode_ == "program-stage") summary.stage = ShaderStage::kVertex;
        if (failure_mode_ == "program-bytes") ++summary.binary_size;
        if (failure_mode_ == "program-groups") ++summary.group_count;
        if (failure_mode_ == "program-instructions") ++summary.instruction_count;
        if (failure_mode_ == "program-vertex-input") summary.vertex_input_mask ^= 1;
        if (failure_mode_ == "program-vertex-output") summary.vertex_output_mask ^= 1;
        if (failure_mode_ == "program-pixel-output") summary.pixel_output_mask ^= 1;
        if (failure_mode_ == "program-early-hsr") summary.early_hsr_safe ^= 1;
        if (failure_mode_ == "program-depth") summary.writes_depth ^= 1;
        if (failure_mode_ == "program-derivatives") summary.uses_derivatives ^= 1;
        if (failure_mode_ == "program-ends-task") summary.ends_task ^= 1;
        if (failure_mode_ == "program-handle-slot" || failure_mode_ == "program-handle-generation") {
          const auto old = changed.fragment_instructions;
          const auto code = LoadArray<PcoInstruction>(pool_, old);
          if (failure_mode_ == "program-handle-generation") pool_.Release(old);
          else retained_program_ = old;
          changed.fragment_instructions = StoreNewArray(pool_, code);
          Check(failure_mode_ == "program-handle-slot" ? changed.fragment_instructions.slot != old.slot :
                changed.fragment_instructions.slot == old.slot && changed.fragment_instructions.generation != old.generation,
                "negative program handle mutation isolates slot from generation identity");
        }
        StorePipelineState(pool_, txn.state, changed);
      }
      cluster_responses_.write(done);
    }
  }
  MemoryPool pool_;
  GpuMemorySystem memory_;
  sc_core::sc_fifo<PipelineTxn> input_{"input", 1}, output_{"output", 1},
      cluster_requests_{"cluster_requests", 1}, cluster_responses_{"cluster_responses", 1},
      texture_requests_{"texture_requests", 1}, texture_responses_{"texture_responses", 1},
      texture_input_{"texture_input", 1}, texture_output_{"texture_output", 1};
  UscCluster cluster_;
  TextureUnit texture_;
  std::uint64_t sequence_ = 0;
  unsigned expected_lane_count_ = 0, fifo_batches_ = 0;
  std::size_t largest_batch_ = 0;
  std::vector<unsigned> requests_per_lane_;
  std::string failure_mode_;
  PoolHandle retained_program_;
};
} // namespace

int sc_main(int argc, char **argv) {
  try {
    if (argc == 2) {
      (void)ExpectedFailure(argv[1]);
      Harness rejection("rejection", MemoryMode::kDirect);
      rejection.Run(kQuadCap + 1, false, false, argv[1]);
      return 0;
    }
    Check(argc == 1, "residency fixture expects at most one rejection mode");
    std::cout << "residency FIFO bytes: request=" << sizeof(TextureSampleRequest)
              << " response=" << sizeof(TextureSampleResponse)
              << " continuation=" << sizeof(PcoFragmentContinuation)
              << " total_per_lane=" << sizeof(TextureSampleRequest) +
                  sizeof(TextureSampleResponse) + sizeof(PcoFragmentContinuation) << '\n';
    Harness direct("direct", MemoryMode::kDirect);
    Harness bypass("bypass", MemoryMode::kBypass);
    Harness cache("cache", MemoryMode::kCache);
    for (auto *h : {&direct, &bypass, &cache}) {
      for (unsigned quads : {1U, 255U, 256U, 257U, 513U})
        h->Run(quads, false, quads % 2 != 0);
      h->Run(513, true, true);
    }
    std::cout << "usc_fragment_residency_test: PASS checks=" << checks << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
