// Focused event-path regression for fragment USC texture continuation.
// One lane is enough to prove that a response resumes at WDF, retains the
// coefficient/shared context needed by the next SMP, and emits PIXOUT only
// after the final response.  The responder deliberately sits on the other
// side of the real SystemC FIFO and MemoryPool boundary.

#include "common/pipeline_state.h"
#include "shader/pco_iss.h"
#include "shader/usc_cluster.h"
#include "memory/gpu_memory_system.h"
#include "pco_uniform_buffer_fixtures.h"
#include "pco_multisample_texture_fixtures.h"
#include "pds/pds_engine.h"
#include "shader/usc_slot.h"

#include <systemc>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using pvrgpu::stub::CountPcoInstructions;
using pvrgpu::stub::AttributeFetchVertexPcoBinary;
using pvrgpu::stub::Decode;
using pvrgpu::stub::DrawListStats;
using pvrgpu::stub::DriverPcoStageAbi;
using pvrgpu::stub::DriverPcoTextureSharedLayoutSupported;
using pvrgpu::stub::FragmentInvocation;
using pvrgpu::stub::FragmentOutput;
using pvrgpu::stub::FragmentQuad;
using pvrgpu::stub::FragmentShaderLane;
using pvrgpu::stub::FunctionalCase;
using pvrgpu::stub::HasPoolHandle;
using pvrgpu::stub::LoadArray;
using pvrgpu::stub::LoadPipelineState;
using pvrgpu::stub::MemoryPool;
using pvrgpu::stub::PcoInstruction;
using pvrgpu::stub::PcoOpcode;
using pvrgpu::stub::PcoProgramSummary;
using pvrgpu::stub::PcoRegisterBank;
using pvrgpu::stub::PcoWriteTarget;
using pvrgpu::stub::PipelineStage;
using pvrgpu::stub::PipelineState;
using pvrgpu::stub::PipelineTxn;
using pvrgpu::stub::ReleaseFunctionalPayloads;
using pvrgpu::stub::ShaderStage;
using pvrgpu::stub::StoreNewArray;
using pvrgpu::stub::StorePipelineState;
using pvrgpu::stub::TextureSampleRequest;
using pvrgpu::stub::TextureSampleResponse;
using pvrgpu::stub::UscCluster;
using pvrgpu::stub::UscFragmentTask;
using pvrgpu::stub::VertexLane;

void Check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error("USC continuation test failed: " + message);
}

std::vector<PcoInstruction> MakeTextureProgram(std::size_t sample_count,
                                               std::size_t descriptor_count) {
  Check(sample_count >= 1 &&
            sample_count <=
                pvrgpu::stub::kPcoMaximumTextureSampleInstructions + 1U &&
            descriptor_count >= 1 &&
            descriptor_count <=
                pvrgpu::stub::kPcoMaximumTextureDescriptorSets,
        "test sample count");
  std::vector<PcoInstruction> instructions(2U + sample_count * 2U + 4U);
  for (std::size_t index = 0; index < instructions.size(); ++index) {
    instructions[index].binary_offset =
        static_cast<std::uint32_t>(1U + index * 8U);
    instructions[index].group_index = static_cast<std::uint16_t>(index);
  }

  for (std::size_t coordinate = 0; coordinate < 2; ++coordinate) {
    PcoInstruction &move = instructions[coordinate];
    move.opcode = PcoOpcode::kMoveImmediate;
    move.target = PcoWriteTarget::kTemporary;
    move.source_count = 0;
    move.output_index = static_cast<std::uint16_t>(coordinate);
    move.immediate = coordinate == 0 ? UINT32_C(0x3e800000)
                                     : UINT32_C(0x3f400000);
  }
  for (std::size_t set = 0; set < sample_count; ++set) {
    const std::size_t descriptor_set = set % descriptor_count;
    PcoInstruction &sample = instructions[2U + set * 2U];
    sample.opcode = PcoOpcode::kTextureSample;
    sample.target = PcoWriteTarget::kTemporary;
    sample.source = {PcoRegisterBank::kTemporary, 0};
    sample.source1 = {
        PcoRegisterBank::kShared,
        static_cast<std::uint16_t>(
            descriptor_set * pvrgpu::stub::kPcoTextureDescriptorDwordCount)};
    sample.source2 = {
        PcoRegisterBank::kShared,
        static_cast<std::uint16_t>(
            descriptor_set * pvrgpu::stub::kPcoTextureDescriptorDwordCount +
            8U)};
    sample.source_count = 3;
    sample.component_count = 4;
    sample.output_index = static_cast<std::uint16_t>(4U + set * 4U);

    PcoInstruction &wait = instructions[3U + set * 2U];
    wait.opcode = PcoOpcode::kWaitDataFence;
    wait.target = PcoWriteTarget::kNone;
    wait.source_count = 0;
  }

  const std::uint16_t final_sample_base =
      static_cast<std::uint16_t>(4U + (sample_count - 1U) * 4U);
  const std::size_t output_base = 2U + sample_count * 2U;
  for (std::size_t component = 0; component < 4; ++component) {
    PcoInstruction &move = instructions[output_base + component];
    move.opcode = PcoOpcode::kMoveBypass;
    move.target = PcoWriteTarget::kPixelOutput;
    move.source = {PcoRegisterBank::kTemporary,
                   static_cast<std::uint16_t>(final_sample_base + component)};
    move.output_index = static_cast<std::uint16_t>(component);
  }
  instructions.back().end_group = 1;
  return instructions;
}

std::array<std::uint32_t, 4> ResponseForRound(std::size_t round) {
  return {
      static_cast<std::uint32_t>(UINT32_C(0x3e800000) + round),
      static_cast<std::uint32_t>(UINT32_C(0x3f000000) + round),
      static_cast<std::uint32_t>(UINT32_C(0x3f400000) + round),
      UINT32_C(0x3f800000),
  };
}

struct CasePayload {
  pvrgpu::stub::PoolHandle state;
  PipelineTxn txn;
};

CasePayload MakeCase(MemoryPool &pool, std::size_t sample_count,
                     std::uint64_t sequence,
                     std::size_t descriptor_count = 0,
                     std::size_t push_constant_count = 0,
                     bool empty_push_at_descriptor_end = false,
                     std::size_t coefficient_dword_count = 8) {
  if (descriptor_count == 0)
    descriptor_count = sample_count;
  Check(coefficient_dword_count >= 4 &&
            coefficient_dword_count % 4 == 0 &&
            coefficient_dword_count <=
                pvrgpu::stub::kPcoMaximumVaryingCoefficientCount,
        "fragment coefficient fixture count");
  const std::vector<PcoInstruction> instructions =
      MakeTextureProgram(sample_count, descriptor_count);
  PcoProgramSummary summary;
  summary.stage = ShaderStage::kFragment;
  summary.binary_size =
      static_cast<std::uint32_t>(instructions.size() * 8U);
  summary.group_count = static_cast<std::uint32_t>(instructions.size());
  summary.instruction_count = summary.group_count;
  summary.pixel_output_mask = 0x0f;
  summary.early_hsr_safe = 1;

  DrawListStats stats;
  stats.drawlist_index = 0;
  const auto instruction_counts = CountPcoInstructions(instructions, false);
  stats.fragment.program_groups = summary.group_count;
  stats.fragment.program_instructions = summary.instruction_count;
  stats.fragment.program_alu_instructions = instruction_counts.alu;
  stats.fragment.program_tex_instructions = instruction_counts.texture;
  stats.fragment.program_memory_instructions = instruction_counts.memory;
  stats.fragment.program_recorded = 1;

  FragmentInvocation invocation;
  invocation.x = 7;
  invocation.y = 11;
  invocation.parameter_index = 3;
  invocation.submit_ordinal = sequence;
  invocation.quad_id = 9;
  invocation.quad_lane = 0;
  invocation.depth = 0.5F;

  FragmentShaderLane lane;
  lane.x = invocation.x;
  lane.y = invocation.y;
  lane.parameter_index = invocation.parameter_index;
  lane.submit_ordinal = invocation.submit_ordinal;
  lane.quad_id = invocation.quad_id;
  lane.quad_lane = invocation.quad_lane;
  lane.visible_invocation_index = 0;
  lane.depth = invocation.depth;

  FragmentQuad quad;
  quad.parameter_index = invocation.parameter_index;
  quad.quad_id = invocation.quad_id;
  quad.submit_ordinal = invocation.submit_ordinal;
  quad.invocation_indices[0] = 0;
  quad.coverage_mask = 1;
  quad.write_mask = 1;

  UscFragmentTask task;
  task.fragment_quad_index = 0;
  task.first_coefficient_dword = 0;
  task.coefficient_dword_count =
      static_cast<std::uint16_t>(coefficient_dword_count);

  std::vector<std::uint32_t> shared(
      descriptor_count * pvrgpu::stub::kPcoTextureDescriptorDwordCount +
      push_constant_count);
  for (std::size_t index = 0; index < shared.size(); ++index)
    shared[index] = static_cast<std::uint32_t>(UINT32_C(0x5000) + index);

  PipelineState state;
  state.width = 16;
  state.height = 16;
  state.sequence = sequence;
  state.functional_case = FunctionalCase::kDriverPcoTriangles;
  state.stage = PipelineStage::kFragmentIssued;
  state.drawlist_stats = StoreNewArray(pool, std::vector<DrawListStats>{stats});
  state.fragment_instructions = StoreNewArray(pool, instructions);
  state.fragment_invocations =
      StoreNewArray(pool, std::vector<FragmentInvocation>{invocation});
  state.fragment_shader_lanes =
      StoreNewArray(pool, std::vector<FragmentShaderLane>{lane});
  state.fragment_quads =
      StoreNewArray(pool, std::vector<FragmentQuad>{quad});
  state.usc_fragment_tasks =
      StoreNewArray(pool, std::vector<UscFragmentTask>{task});
  state.usc_coefficient_banks =
      StoreNewArray(pool,
                    std::vector<std::uint32_t>(coefficient_dword_count, 0));
  state.fragment_shared_registers = StoreNewArray(pool, shared);
  state.fragment_program_summary = summary;
  state.fragment_pco_abi.temps =
      static_cast<std::uint32_t>(4U + sample_count * 4U);
  state.fragment_pco_abi.coefficients =
      static_cast<std::uint32_t>(coefficient_dword_count);
  state.fragment_pco_abi.shareds =
      static_cast<std::uint32_t>(shared.size());
  if (push_constant_count != 0 || empty_push_at_descriptor_end) {
    state.fragment_pco_abi.push_constant_start =
        static_cast<std::uint32_t>(
            descriptor_count *
            pvrgpu::stub::kPcoTextureDescriptorDwordCount);
    state.fragment_pco_abi.push_constant_count =
        static_cast<std::uint32_t>(push_constant_count);
  }
  state.position_output_start = 0;
  state.position_output_count = 4;
  state.varying_output_start = 4;
  state.varying_output_count = coefficient_dword_count > 4 ? 1 : 0;
  state.fragment_position_start = 0;
  state.fragment_position_count = 4;
  state.fragment_varying_start = 4;
  state.fragment_varying_count =
      static_cast<std::uint32_t>(coefficient_dword_count - 4U);
  state.sampled_texture_count = static_cast<std::uint32_t>(descriptor_count);
  state.active_fragment_invocations = 1;
  state.fragment_shader_lane_count = 1;
  state.fragment_groups = 1;
  state.counters.drawlists = 1;

  CasePayload payload;
  payload.state = pool.Allocate(sizeof(PipelineState));
  StorePipelineState(pool, payload.state, state);
  payload.txn.state = payload.state;
  payload.txn.frame = 1;
  payload.txn.sequence = sequence;
  return payload;
}

class TextureResponder final : public sc_core::sc_module {
public:
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};
  sc_core::sc_fifo_out<PipelineTxn> output{"output"};

  TextureResponder(sc_core::sc_module_name name, MemoryPool &pool,
                   std::size_t expected_rounds,
                   std::size_t descriptor_count,
                   bool corrupt_response_order = false,
                   bool native_multisample = false,
                   std::string corrupt_continuation = {})
      : sc_module(name), pool_(pool), expected_rounds_(expected_rounds),
        descriptor_count_(descriptor_count),
        corrupt_response_order_(corrupt_response_order),
        native_multisample_(native_multisample),
        corrupt_continuation_(std::move(corrupt_continuation)) {
    SC_THREAD(Run);
  }

  const std::vector<std::uint8_t> &descriptor_sets() const {
    return descriptor_sets_;
  }

  std::size_t descriptor_count() const { return descriptor_count_; }

private:
  void Run() {
    for (std::size_t round = 0; round < expected_rounds_; ++round) {
      const PipelineTxn txn = input.read();
      PipelineState state = LoadPipelineState(pool_, txn.state);
      Check(state.stage == PipelineStage::kFragmentTexturePending,
            "request stage");
      const std::vector<TextureSampleRequest> requests =
          LoadArray<TextureSampleRequest>(pool_, state.texture_sample_requests);
      Check(requests.size() == 1 && requests[0].shader_lane_index == 0 &&
                requests[0].request_id == 0 && requests[0].binding == 0 &&
                requests[0].data_request == 0 &&
                requests[0].shader_stage == ShaderStage::kFragment &&
                requests[0].descriptor_set == round % descriptor_count_,
            "one ordered request per texture round");
      descriptor_sets_.push_back(requests[0].descriptor_set);
      if (native_multisample_) {
        Check(requests[0].sample_index_present == 1 &&
                  requests[0].sample_index == 3 && requests[0].normalized == 0 &&
                  requests[0].coordinates[0] == UINT32_C(0x40800000) &&
                  requests[0].coordinates[1] == UINT32_C(0x40e00000),
              "native MS request preserves integer texels and selected sample");
      }

      TextureSampleResponse response;
      response.shader_lane_index = requests[0].shader_lane_index;
      response.shader_stage = ShaderStage::kFragment;
      response.request_id =
          corrupt_response_order_ ? requests[0].request_id + 1U
                                  : requests[0].request_id;
      const auto rgba = ResponseForRound(round);
      std::copy(rgba.begin(), rgba.end(), response.rgba);
      Check(!HasPoolHandle(state.texture_sample_responses),
            "response handle starts empty");
      state.texture_sample_responses =
          StoreNewArray(pool_, std::vector<TextureSampleResponse>{response});
      if (!corrupt_continuation_.empty()) {
        auto saved = LoadArray<pvrgpu::stub::PcoFragmentContinuation>(pool_, state.fragment_continuations);
        Check(saved.size() == 1, "malformed continuation target must be one native lane");
        if (corrupt_continuation_ == "count") ++saved[0].executed_instructions.alu;
        else if (corrupt_continuation_ == "steps") ++saved[0].native_steps;
        else if (corrupt_continuation_ == "mask") saved[0].execution_predicate ^= 1;
        else throw std::runtime_error("unknown continuation mutation");
        StoreArray(pool_, state.fragment_continuations, saved);
      }
      state.stage = PipelineStage::kTextureSamplesReady;
      StorePipelineState(pool_, txn.state, state);
      output.write(txn);
    }
  }

  MemoryPool &pool_;
  std::size_t expected_rounds_ = 0;
  std::size_t descriptor_count_ = 0;
  bool corrupt_response_order_ = false;
  bool native_multisample_ = false;
  std::string corrupt_continuation_;
  std::vector<std::uint8_t> descriptor_sets_;
};

CasePayload MakeVertexCase(MemoryPool &pool, std::uint64_t sequence) {
  /* Two byte-identical Terrain D3 VS SMP/WDF pairs.  The real shader samples
   * descriptor set 1 first and set 0 second. */
  const std::vector<std::uint8_t> terrain_groups = {
      0x57, 0xa0, 0x00, 0xf4, 0x4c, 0x94, 0x60, 0x80, 0x1c,
      0x88, 0x80, 0xa0, 0x00, 0xff, 0x02, 0x80, 0x6a, 0xff,
      0x57, 0xa0, 0x00, 0xf4, 0x4c, 0x80, 0x60, 0x80, 0x08,
      0x88, 0x80, 0xa0, 0x00, 0xff, 0x02, 0x80, 0x6a, 0xff,
  };
  std::vector<std::uint8_t> binary = AttributeFetchVertexPcoBinary();
  const auto base = Decode(ShaderStage::kVertex, binary);
  const std::size_t insertion_offset = base.instructions[2].binary_offset - 3U;
  binary.insert(binary.begin() + insertion_offset, terrain_groups.begin(),
                terrain_groups.end());
  const auto decoded = Decode(ShaderStage::kVertex, binary);
  Check(decoded.instructions.size() == 10 &&
            decoded.instructions[2].opcode == PcoOpcode::kTextureSample &&
            decoded.instructions[2].source1.index == 20 &&
            decoded.instructions[4].opcode == PcoOpcode::kTextureSample &&
            decoded.instructions[4].source1.index == 0,
        "canonical Terrain vertex set order");

  DrawListStats stats;
  stats.drawlist_index = 0;
  const auto instruction_counts =
      CountPcoInstructions(decoded.instructions, false);
  stats.vertex.program_groups = decoded.summary.group_count;
  stats.vertex.program_instructions = decoded.summary.instruction_count;
  stats.vertex.program_alu_instructions = instruction_counts.alu;
  stats.vertex.program_tex_instructions = instruction_counts.texture;
  stats.vertex.program_memory_instructions = instruction_counts.memory;
  stats.vertex.program_recorded = 1;

  VertexLane lane;
  lane.vertex_input[0] = UINT32_C(0x3e800000);
  lane.vertex_input[1] = UINT32_C(0x3f400000);
  std::vector<pvrgpu::stub::ShaderSharedRegister> shared(40);
  for (std::size_t index = 0; index < shared.size(); ++index)
    shared[index].value = static_cast<std::uint32_t>(UINT32_C(0x7000) + index);

  PipelineState state;
  state.width = 16;
  state.height = 16;
  state.sequence = sequence;
  state.functional_case = FunctionalCase::kDriverPcoTriangles;
  state.stage = PipelineStage::kVertexIssued;
  state.drawlist_stats = StoreNewArray(pool, std::vector<DrawListStats>{stats});
  state.vertex_instructions = StoreNewArray(pool, decoded.instructions);
  state.vertex_lanes = StoreNewArray(pool, std::vector<VertexLane>{lane});
  state.vertex_shared_registers = StoreNewArray(pool, shared);
  state.vertex_program_summary = decoded.summary;
  state.vertex_pco_abi.temps = 8;
  state.vertex_pco_abi.vertex_inputs = 2;
  state.vertex_pco_abi.vertex_outputs = 4;
  state.vertex_pco_abi.shareds = 40;
  state.vertex_sampled_texture_count = 2;
  state.vertex_groups = 1;
  state.counters.drawlists = 1;
  state.counters.vs_invocations = 1;

  CasePayload payload;
  payload.state = pool.Allocate(sizeof(PipelineState));
  StorePipelineState(pool, payload.state, state);
  payload.txn.state = payload.state;
  payload.txn.frame = 1;
  payload.txn.sequence = sequence;
  return payload;
}

class VertexTextureResponder final : public sc_core::sc_module {
public:
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};
  sc_core::sc_fifo_out<PipelineTxn> output{"output"};

  VertexTextureResponder(sc_core::sc_module_name name, MemoryPool &pool)
      : sc_module(name), pool_(pool) {
    SC_THREAD(Run);
  }

private:
  void Run() {
    constexpr std::uint8_t kSets[] = {1, 0};
    for (std::size_t round = 0; round < 2; ++round) {
      const PipelineTxn txn = input.read();
      PipelineState state = LoadPipelineState(pool_, txn.state);
      Check(state.stage == PipelineStage::kVertexTexturePending,
            "vertex request stage");
      const auto requests = LoadArray<TextureSampleRequest>(
          pool_, state.texture_sample_requests);
      Check(requests.size() == 1 && requests[0].shader_lane_index == 0 &&
                requests[0].request_id == 0 && requests[0].quad_id == 0 &&
                requests[0].quad_lane == 0 &&
                requests[0].shader_stage == ShaderStage::kVertex &&
                requests[0].descriptor_set == kSets[round],
            "vertex request keeps stage/local id and Terrain set order");
      TextureSampleResponse response;
      response.shader_lane_index = 0;
      response.request_id = 0;
      response.shader_stage = ShaderStage::kVertex;
      const auto rgba = ResponseForRound(round);
      std::copy(rgba.begin(), rgba.end(), response.rgba);
      state.texture_sample_responses =
          StoreNewArray(pool_, std::vector<TextureSampleResponse>{response});
      state.stage = PipelineStage::kVertexTextureSamplesReady;
      StorePipelineState(pool_, txn.state, state);
      output.write(txn);
    }
  }

  MemoryPool &pool_;
};

void CheckCompletedVertexCase(MemoryPool &pool, const CasePayload &payload) {
  const PipelineState state = LoadPipelineState(pool, payload.state);
  Check(state.stage == PipelineStage::kVertexShaded,
        "vertex completion stage");
  Check(!HasPoolHandle(state.texture_sample_requests) &&
            !HasPoolHandle(state.texture_sample_responses) &&
            !HasPoolHandle(state.vertex_continuations) &&
            !HasPoolHandle(state.fragment_continuations),
        "vertex continuation payloads are retired");
  const auto lanes = LoadArray<VertexLane>(pool, state.vertex_lanes);
  const auto final_response = ResponseForRound(1);
  Check(lanes.size() == 1 && lanes[0].emitted == 1 && lanes[0].ended == 1 &&
            lanes[0].vertex_output[0] == final_response[0] &&
            lanes[0].vertex_output[1] == final_response[1] &&
            lanes[0].vertex_output[2] == 0 &&
            lanes[0].vertex_output[3] == UINT32_C(0x3f800000),
        "vertex WDF resumes twice and commits exact VTXOUT");
  Check(state.counters.vs_tex_instructions == 2,
        "vertex SMP instructions count exactly once per lane");
  ReleaseFunctionalPayloads(pool, state);
  pool.Release(payload.state);
  Check(pool.bytes_in_flight() == 0 &&
            pool.allocations() == pool.releases(),
        "vertex MemoryPool allocations are balanced");
}

void CheckCompletedCase(MemoryPool &pool, const CasePayload &payload,
                        const TextureResponder &responder,
                        std::size_t sample_count) {
  const PipelineState state = LoadPipelineState(pool, payload.state);
  Check(state.stage == PipelineStage::kFragmentShaded,
        "fragment completion stage");
  Check(!HasPoolHandle(state.texture_sample_requests) &&
            !HasPoolHandle(state.texture_sample_responses) &&
            !HasPoolHandle(state.fragment_continuations),
        "round payloads are retired");
  Check(HasPoolHandle(state.fragment_outputs), "fragment output exists");
  const auto outputs = LoadArray<FragmentOutput>(pool, state.fragment_outputs);
  const auto expected = ResponseForRound(sample_count - 1U);
  Check(outputs.size() == 1 && outputs[0].written_mask[0] == 0x0f &&
            outputs[0].render_target_count == 1 &&
            std::equal(expected.begin(), expected.end(),
                       outputs[0].pixel_output),
        "final response alone reaches PIXOUT");
  Check(responder.descriptor_sets().size() == sample_count,
        "exact request round count");
  for (std::size_t round = 0; round < sample_count; ++round) {
    Check(responder.descriptor_sets()[round] ==
              round % responder.descriptor_count(),
          "descriptor sets advance in program order");
  }
  Check(state.counters.fs_tex_instructions == sample_count,
        "native SMP issues are counted once across all continuation rounds");
  const auto golden = pvrgpu::stub::CountPcoInstructions(
      LoadArray<pvrgpu::stub::PcoInstruction>(pool, state.fragment_instructions), true);
  Check(state.counters.fs_alu_instructions == golden.alu &&
            state.counters.fs_tex_instructions == golden.texture &&
            state.counters.fs_memory_instructions == golden.memory,
        "straight-line native USC counters retain the original ALU/TEX/MEM golden");

  ReleaseFunctionalPayloads(pool, state);
  pool.Release(payload.state);
  Check(pool.bytes_in_flight() == 0 &&
            pool.allocations() == pool.releases(),
        "MemoryPool allocations are balanced");
}

constexpr std::uint64_t kUniformAddress = UINT64_C(0x12345678000);
constexpr std::uint64_t kTextureReadAddress = kUniformAddress + 4096;
constexpr std::array<std::uint32_t, 4> kFirstUniformWords = {
    0x3e800000, 0x3f400000, 0x80000000, 0x7fa12345};
constexpr std::array<std::uint32_t, 4> kSecondUniformWords = {
    0xdeadbeef, 0x01000001, 0x40000000, 0x3f800000};

CasePayload MakeUniformTextureCase(MemoryPool &pool,
                                    pvrgpu::stub::GpuMemorySystem &memory,
                                    ShaderStage stage) {
  using namespace pvrgpu::stub;
  const bool vertex = stage == ShaderStage::kVertex;
  auto payload = vertex ? MakeVertexCase(pool, 81) : MakeCase(pool, 1, 82, 2);
  auto state = LoadPipelineState(pool, payload.state);
  state.memory_mode = memory.mode();
  const auto original = test::UniformBufferFixture(vertex, 4);
  // Genuine Mesa groups: LD/WDF, Terrain set1 SMP/WDF, LD/WDF + output.
  const std::array<std::uint8_t, 18> sample = {
      0x57, 0xa0, 0x00, 0xf4, 0x4c, 0x94, 0x60, 0x80, 0x1c,
      0x88, 0x80, 0xa0, 0x00, 0xff, 0x02, 0x80, 0x6a, 0xff};
  std::vector<std::uint8_t> binary(original.begin(), original.begin() + 62);
  binary.insert(binary.end(), sample.begin(), sample.end());
  binary.insert(binary.end(), original.begin(), original.end());
  auto program = Decode(stage, binary);
  Check(program.instructions[6].opcode == PcoOpcode::kTextureSample &&
            program.instructions[7].opcode == PcoOpcode::kWaitDataFence,
        "UBO/texture fixture group boundary");
  // Relocate the decoded UBO shared operands behind two texture descriptors.
  // Give the second LD its own runtime push offset (SH45 instead of SH44),
  // so returning/repeating the first read cannot accidentally pass.
  for (std::size_t index = 0; index < program.instructions.size(); ++index) {
    auto &instruction = program.instructions[index];
    if (instruction.opcode == PcoOpcode::kTextureSample)
      continue;
    for (auto *source : {&instruction.source, &instruction.source1,
                         &instruction.source2, &instruction.source3}) {
      if (source->bank == PcoRegisterBank::kShared) {
        source->index += 40;
        if (index >= 8 && source->index == 44)
          source->index = 45;
      }
    }
  }
  memory.HostWrite(kUniformAddress, kFirstUniformWords.data(), 16);
  memory.HostWrite(kUniformAddress + 16, kSecondUniformWords.data(), 16);
  const auto texture_words = ResponseForRound(0);
  memory.HostWrite(kTextureReadAddress, texture_words.data(), 16);
  std::vector<std::uint32_t> shared(46, 0);
  shared[40] = static_cast<std::uint32_t>(kUniformAddress);
  shared[41] = static_cast<std::uint32_t>(kUniformAddress >> 32U);
  shared[42] = 32;
  shared[45] = 16;
  auto &abi = vertex ? state.vertex_pco_abi : state.fragment_pco_abi;
  abi.temps = 8;
  abi.shareds = shared.size();
  abi.uniform_buffer_descriptor_start = 40;
  abi.uniform_buffer_descriptor_count = 1;
  abi.push_constant_start = 44;
  abi.push_constant_count = 2;
  auto &instructions = vertex ? state.vertex_instructions : state.fragment_instructions;
  pool.Release(instructions);
  instructions = StoreNewArray(pool, program.instructions);
  auto &registers = vertex ? state.vertex_shared_registers : state.fragment_shared_registers;
  pool.Release(registers);
  if (vertex) {
    std::vector<ShaderSharedRegister> words(shared.size());
    for (std::size_t index = 0; index < shared.size(); ++index)
      words[index].value = shared[index];
    registers = StoreNewArray(pool, words);
    state.vertex_program_summary = program.summary;
  } else {
    registers = StoreNewArray(pool, shared);
    state.fragment_program_summary = program.summary;
  }
  const auto resources = StoreNewArray(pool,
      std::vector<UniformBufferResource>{{kUniformAddress, 32, 0, 40}});
  (vertex ? state.vertex_uniform_buffer_resources
          : state.fragment_uniform_buffer_resources) = resources;
  auto drawlists = LoadArray<DrawListStats>(pool, state.drawlist_stats);
  auto &stats = vertex ? drawlists[0].vertex : drawlists[0].fragment;
  stats = {};
  const auto counts = CountPcoInstructions(program.instructions, false);
  stats.program_groups = program.summary.group_count;
  stats.program_instructions = program.summary.instruction_count;
  stats.program_alu_instructions = counts.alu;
  stats.program_tex_instructions = counts.texture;
  stats.program_memory_instructions = counts.memory;
  stats.program_recorded = 1;
  StoreArray(pool, state.drawlist_stats, drawlists);
  StorePipelineState(pool, payload.state, state);
  return payload;
}

class UniformTextureResponder final : public sc_core::sc_module {
public:
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};
  sc_core::sc_fifo_out<PipelineTxn> output{"output"};
  UniformTextureResponder(sc_core::sc_module_name name, MemoryPool &pool,
                          pvrgpu::stub::GpuMemorySystem &memory, ShaderStage stage)
      : sc_module(name), pool_(pool), memory_(memory), stage_(stage) {
    SC_THREAD(Run);
  }
  unsigned requests_seen = 0;
private:
  void Run() {
    using namespace pvrgpu::stub;
    const auto txn = input.read();
    auto state = LoadPipelineState(pool_, txn.state);
    const bool vertex = stage_ == ShaderStage::kVertex;
    Check(state.stage == (vertex ? PipelineStage::kVertexTexturePending
                                 : PipelineStage::kFragmentTexturePending),
          "UBO sequence reaches the real texture FIFO");
    const auto requests = LoadArray<TextureSampleRequest>(pool_, state.texture_sample_requests);
    Check(requests.size() == 1 && requests[0].shader_stage == stage_ &&
              requests[0].descriptor_set == 1 && requests[0].request_id == 0,
          "UBO sequence has one stage-local texture request");
    ++requests_seen;
    const auto check_saved = [&](const auto &saved) {
      Check(saved.size() == 1 && saved[0].valid == 1 &&
                std::equal(kFirstUniformWords.begin(), kFirstUniformWords.end(),
                           saved[0].temporaries.begin()),
            "first LD's four exact words survive into the FIFO continuation");
    };
    if (vertex)
      check_saved(LoadArray<PcoVertexContinuation>(pool_, state.vertex_continuations));
    else
      check_saved(LoadArray<PcoFragmentContinuation>(pool_, state.fragment_continuations));
    // Give USC a real suspended interval, then add genuine texture-read stats
    // to the carried state. The final result must retain these plus both LDs.
    wait(sc_core::sc_time(7, sc_core::SC_NS));
    const auto read = memory_.Read(kTextureReadAddress, 16, MemoryClient::kTextureCache);
    ApplyMemoryAccessStats(state.counters, read.stats);
    TextureSampleResponse response;
    response.shader_stage = stage_;
    response.shader_lane_index = requests[0].shader_lane_index;
    response.request_id = requests[0].request_id;
    std::memcpy(response.rgba, read.data.data(), 16);
    state.texture_sample_responses = StoreNewArray(pool_,
        std::vector<TextureSampleResponse>{response});
    state.stage = vertex ? PipelineStage::kVertexTextureSamplesReady
                         : PipelineStage::kTextureSamplesReady;
    StorePipelineState(pool_, txn.state, state);
    output.write(txn);
  }
  MemoryPool &pool_;
  pvrgpu::stub::GpuMemorySystem &memory_;
  ShaderStage stage_;
};

class UniformTextureHarness final : public sc_core::sc_module {
public:
  UniformTextureHarness(sc_core::sc_module_name name, ShaderStage stage,
                        pvrgpu::stub::MemoryMode mode)
      : sc_module(name), memory_(mode), stage_(stage),
        payload_(MakeUniformTextureCase(pool_, memory_, stage)),
        cluster_("cluster", pool_, stage, &memory_),
        responder_("responder", pool_, memory_, stage) {
    cluster_.input(input_);
    cluster_.output(output_);
    cluster_.texture_request_output(requests_);
    cluster_.texture_response_input(responses_);
    responder_.input(requests_);
    responder_.output(responses_);
    input_.write(payload_.txn);
  }
  void Verify() {
    using namespace pvrgpu::stub;
    PipelineTxn completed;
    Check(output_.nb_read(completed) && completed.sequence == payload_.txn.sequence &&
              responder_.requests_seen == 1 && !output_.nb_read(completed),
          "LD/SMP/LD completes exactly once");
    const auto state = LoadPipelineState(pool_, payload_.state);
    const bool vertex = stage_ == ShaderStage::kVertex;
    Check(state.stage == (vertex ? PipelineStage::kVertexShaded
                                 : PipelineStage::kFragmentShaded), "UBO completion stage");
    if (vertex) {
      const auto lanes = LoadArray<VertexLane>(pool_, state.vertex_lanes);
      Check(lanes.size() == 1 && lanes[0].emitted == 1 && lanes[0].ended == 1 &&
                std::equal(kSecondUniformWords.begin(), kSecondUniformWords.end(),
                           std::begin(lanes[0].vertex_output)),
            "vertex second LD reads its distinct offset after texture resume");
    } else {
      const auto pixels = LoadArray<FragmentOutput>(pool_, state.fragment_outputs);
      Check(pixels.size() == 1 && pixels[0].written_mask[0] == 15 &&
                std::equal(kSecondUniformWords.begin(), kSecondUniformWords.end(),
                           std::begin(pixels[0].pixel_output)),
            "fragment second LD reads its distinct offset after texture resume");
    }
    const auto &c = state.counters;
    if (memory_.mode() == MemoryMode::kDirect) {
      Check(c.memory_direct_read_bytes == 48 && c.dram_read_transactions == 0 &&
                c.slc_read_accesses == 0, "direct mode counts two LDs plus one texture read");
    } else if (memory_.mode() == MemoryMode::kBypass) {
      Check(c.memory_direct_read_bytes == 0 && c.dram_read_transactions == 3 &&
                c.dram_read_bytes == 48 && c.slc_read_accesses == 0,
            "bypass mode preserves all three reads through state reload");
    } else {
      Check(c.slc_read_accesses == 3 && c.slc_misses == 2 && c.slc_hits == 1 &&
                c.dram_read_transactions == 2 && c.dram_read_bytes == 256,
            "cache mode retains UBO cold/warm reads and the texture cold read");
    }
    Check((vertex ? c.vs_tex_instructions : c.fs_tex_instructions) == 1,
          "texture suspension does not repeat the SMP instruction");
    if (!vertex) {
      const auto golden = CountPcoInstructions(
          LoadArray<PcoInstruction>(pool_, state.fragment_instructions), true);
      Check(c.fs_alu_instructions == golden.alu && c.fs_tex_instructions == golden.texture &&
                c.fs_memory_instructions == golden.memory && c.fs_memory_instructions == 2,
            "native LD/SMP/LD counter totals exclude checkpoint replay in every memory mode");
    }
    Check(!HasPoolHandle(state.vertex_continuations) &&
              !HasPoolHandle(state.fragment_continuations) &&
              !HasPoolHandle(state.texture_sample_requests) &&
              !HasPoolHandle(state.texture_sample_responses),
          "UBO texture continuation payloads retire");
    ReleaseFunctionalPayloads(pool_, state);
    pool_.Release(payload_.state);
    Check(pool_.bytes_in_flight() == 0 && pool_.allocations() == pool_.releases(),
          "UBO range and continuation handles are released exactly once");
  }
private:
  MemoryPool pool_;
  pvrgpu::stub::GpuMemorySystem memory_;
  ShaderStage stage_;
  CasePayload payload_;
  sc_core::sc_fifo<PipelineTxn> input_{"input", 1};
  sc_core::sc_fifo<PipelineTxn> requests_{"requests", 1};
  sc_core::sc_fifo<PipelineTxn> responses_{"responses", 1};
  sc_core::sc_fifo<PipelineTxn> output_{"output", 1};
  UscCluster cluster_;
  UniformTextureResponder responder_;
};

// Genuine PCO with no VS-to-FS user varying. PDS must transport the actual
// position-W plane even though its shader-varying linkage vector is empty.
class NativeMultisampleHarness final : public sc_core::sc_module {
public:
  NativeMultisampleHarness(sc_core::sc_module_name name, bool array,
                          bool query = false)
      : sc_module(name), array_(array), query_(query), pds_("pds", pool_),
        slot_("slot", pool_, ShaderStage::kFragment),
        cluster_("cluster", pool_, ShaderStage::kFragment),
        responder_("responder", pool_, query ? 0 : 1, 1, false, !query) {
    payload_ = MakeCase(pool_, 1, array ? 202 : 201, 1, query ? 0 : 4, true, 4);
    auto state = LoadPipelineState(pool_, payload_.state);
    const auto program = Decode(ShaderStage::kFragment,
        pvrgpu::stub::test::MultisampleTextureFixture(query ? (array ? 19 : 18) :
                                                            (array ? 3 : 0)));
    pool_.Release(state.fragment_instructions);
    state.fragment_instructions = StoreNewArray(pool_, program.instructions);
    state.fragment_program_summary = program.summary;
    auto stats = LoadArray<DrawListStats>(pool_, state.drawlist_stats);
    const auto counts = CountPcoInstructions(program.instructions, false);
    stats[0].fragment.program_groups = program.summary.group_count;
    stats[0].fragment.program_instructions = program.summary.instruction_count;
    stats[0].fragment.program_alu_instructions = counts.alu;
    stats[0].fragment.program_tex_instructions = counts.texture;
    stats[0].fragment.program_memory_instructions = counts.memory;
    pvrgpu::stub::StoreArray(pool_, state.drawlist_stats, stats);
    std::vector<std::uint32_t> shared(query ? 20 : 24, 0);
    const std::uint64_t image0 = (UINT64_C(2) << 62) |
        (UINT64_C(18) << 48) | (UINT64_C(12) << 34) | (array ? 1U : 4U);
    const std::uint64_t image1 = ((UINT64_C(0x1234000040) >> 2) << 16) |
        (array ? (UINT64_C(4) << 4) | 1U : (UINT64_C(1) << 60) | 12U);
    shared[0] = static_cast<std::uint32_t>(image0);
    shared[1] = static_cast<std::uint32_t>(image0 >> 32);
    shared[2] = static_cast<std::uint32_t>(image1);
    shared[3] = static_cast<std::uint32_t>(image1 >> 32);
    shared[4] = 13 * 19 * 16 * 4;
    if (!query) {
      shared[20] = 4; shared[21] = 7; shared[22] = 2; shared[23] = 3;
    }
    pvrgpu::stub::StoreArray(pool_, state.fragment_shared_registers, shared);
    Check(pvrgpu::stub::UsesShaderVaryings(state) &&
              pvrgpu::stub::VaryingVectorCount(state) == 0 &&
              pvrgpu::stub::VaryingCoefficientDwordCount(state) == 4 &&
              pvrgpu::stub::ActiveVertexOutputDwordCount(state) == 4,
          "zero varying native texture retains exact position-only ABI");
    pool_.Release(state.usc_fragment_tasks);
    pool_.Release(state.usc_coefficient_banks);
    state.usc_fragment_tasks = {};
    state.usc_coefficient_banks = {};
    state.stage = PipelineStage::kFragmentsReady;
    state.shader_varying_bindings = StoreNewArray(pool_,
        std::vector<pvrgpu::stub::ShaderVaryingBinding>{});
    auto invocations = LoadArray<FragmentInvocation>(pool_, state.fragment_invocations);
    auto lanes = LoadArray<FragmentShaderLane>(pool_, state.fragment_shader_lanes);
    auto quads = LoadArray<FragmentQuad>(pool_, state.fragment_quads);
    invocations[0].parameter_index = lanes[0].parameter_index = quads[0].parameter_index = 0;
    pvrgpu::stub::StoreArray(pool_, state.fragment_invocations, invocations);
    pvrgpu::stub::StoreArray(pool_, state.fragment_shader_lanes, lanes);
    pvrgpu::stub::StoreArray(pool_, state.fragment_quads, quads);
    pvrgpu::stub::ParameterTriangle parameter;
    parameter.rasterizable = 1;
    parameter.front_facing = 1;
    parameter.key.submit_ordinal = payload_.txn.sequence;
    parameter.coefficient_set_count = 1;
    parameter.depth_plane_valid = 1;
    parameter.depth_plane[2] = UINT32_C(0x3f000000);
    state.parameter_triangles = StoreNewArray(pool_,
        std::vector<pvrgpu::stub::ParameterTriangle>{parameter});
    pvrgpu::stub::ParameterCoefficientSet plane;
    plane.a = UINT32_C(0x3e000000);
    plane.b = UINT32_C(0x3e800000);
    plane.c = UINT32_C(0x3f800000);
    state.parameter_coefficients = StoreNewArray(pool_,
        std::vector<pvrgpu::stub::ParameterCoefficientSet>{plane});
    state.counters.parameter_coefficient_sets = 1;
    state.counters.parameter_write_bytes = sizeof(plane);
    StorePipelineState(pool_, payload_.state, state);
    pds_.input(input_); pds_.output(pds_output_);
    slot_.input(pds_output_); slot_.output(issued_);
    cluster_.input(issued_); cluster_.output(output_);
    cluster_.texture_request_output(requests_);
    cluster_.texture_response_input(responses_);
    responder_.input(requests_); responder_.output(responses_);
    input_.write(payload_.txn);
  }
  void Verify() {
    PipelineTxn completed;
    Check(output_.nb_read(completed), "native position-only PDS/USC FIFO completion");
    const auto state = LoadPipelineState(pool_, payload_.state);
    const auto bank = LoadArray<std::uint32_t>(pool_, state.usc_coefficient_banks);
    Check(bank == std::vector<std::uint32_t>{UINT32_C(0x3e000000),
               UINT32_C(0x3e800000), UINT32_C(0x3f800000), 0} &&
              state.counters.pds_coefficient_tasks == 1 &&
              state.counters.usc_coefficient_load_bytes == 16,
          "PDS transports actual A/B/C/PAD rather than a fabricated varying");
    if (!query_) {
      CheckCompletedCase(pool_, payload_, responder_, 1);
    } else {
      const auto outputs = LoadArray<FragmentOutput>(pool_, state.fragment_outputs);
      Check(outputs.size() == 1 && outputs[0].pixel_output[0] == 13 &&
                outputs[0].pixel_output[1] == 19 &&
                outputs[0].pixel_output[2] == (array_ ? 5U : 1U) &&
                outputs[0].pixel_output[3] == 4 &&
                state.counters.fs_tex_instructions == 0 &&
                responder_.descriptor_sets().empty() &&
                !HasPoolHandle(state.texture_sample_requests) &&
                !HasPoolHandle(state.texture_sample_responses) &&
                !HasPoolHandle(state.fragment_continuations) &&
                requests_.num_available() == 0 && responses_.num_available() == 0,
            "native descriptor-only query executes ALU without a texture FIFO request");
      ReleaseFunctionalPayloads(pool_, state);
      pool_.Release(payload_.state);
      Check(pool_.bytes_in_flight() == 0 && pool_.allocations() == pool_.releases(),
            "query-only payload ownership balances without fabricated continuations");
    }
  }
private:
  MemoryPool pool_;
  CasePayload payload_;
  bool array_;
  bool query_;
  sc_core::sc_fifo<PipelineTxn> input_{"input", 1}, pds_output_{"pds_output", 1},
      issued_{"issued", 1}, requests_{"requests", 1}, responses_{"responses", 1},
      output_{"output", 1};
  pvrgpu::stub::PdsEngine pds_;
  pvrgpu::stub::UscSlot slot_;
  UscCluster cluster_;
  TextureResponder responder_;
};

enum class TextureControlFlowCase {
  kMixedSkipRemaining,
  kAllSkipRemaining,
  kAllSkipAll,
  kMixedSkipAll,
  kDiscardThenMixed,
  kDivergentSites,
};

// Execute real native TST/P0, BR, ALPHAF, SMP and WDF semantics through the
// USC FIFOs. The responder services only issued requests, including helper
// work; a skipped instruction never gets a synthetic texture response.
class TextureControlFlowHarness final : public sc_core::sc_module {
public:
  TextureControlFlowHarness(sc_core::sc_module_name name,
                           TextureControlFlowCase mode)
      : sc_module(name), mode_(mode),
        cluster_("cluster", pool_, ShaderStage::kFragment) {
    payload_ = MakeCase(pool_, 2, 300, 2, 0, false, 4);
    auto state = LoadPipelineState(pool_, payload_.state);
    const auto straight = MakeTextureProgram(2, 2);
    std::vector<PcoInstruction> program{straight[0], straight[1]};
    for (unsigned component = 0; component < 4; ++component) {
      auto move = straight[0];
      move.output_index = 4 + component;
      move.immediate = Fallback()[component];
      program.push_back(move);
    }
    auto half = straight[0];
    half.output_index = 2;
    half.immediate = UINT32_C(0xbf000000);
    program.push_back(half);
    PcoInstruction subtract;
    subtract.opcode = PcoOpcode::kFloatAdd;
    subtract.target = PcoWriteTarget::kTemporary;
    subtract.output_index = 3;
    subtract.source = {PcoRegisterBank::kSpecial, 97};
    subtract.source1 = {PcoRegisterBank::kTemporary, 2};
    subtract.source_count = 2;
    program.push_back(subtract);
    PcoInstruction predicate;
    predicate.opcode = PcoOpcode::kBooleanCompare;
    predicate.source = {PcoRegisterBank::kTemporary, 3};
    predicate.writes_predicate = 1;
    program.push_back(predicate); // P0 = (pixel center X - 0.5 == 0).
    if (mode_ == TextureControlFlowCase::kDiscardThenMixed) {
      PcoInstruction feedback;
      feedback.opcode = PcoOpcode::kAlphaFeedback;
      feedback.source_count = 0;
      feedback.exec_cnd = 1;
      program.push_back(feedback);
      program.push_back(straight[3]);
      state.fragment_program_summary.early_hsr_safe = 0;
    }
    const bool pre_branch = mode_ == TextureControlFlowCase::kAllSkipAll ||
        mode_ == TextureControlFlowCase::kMixedSkipAll ||
        mode_ == TextureControlFlowCase::kDivergentSites;
    const bool post_branch = mode_ == TextureControlFlowCase::kMixedSkipRemaining ||
        mode_ == TextureControlFlowCase::kAllSkipRemaining ||
        mode_ == TextureControlFlowCase::kDiscardThenMixed ||
        mode_ == TextureControlFlowCase::kDivergentSites;
    PcoInstruction branch;
    branch.opcode = PcoOpcode::kBranch;
    branch.source_count = 0;
    branch.exec_cnd = mode_ == TextureControlFlowCase::kAllSkipAll ? 0 : 1;
    const auto pre_index = program.size();
    if (pre_branch)
      program.push_back(branch);
    program.push_back(straight[2]);
    program.push_back(straight[3]);
    const auto post_index = program.size();
    branch.exec_cnd = mode_ == TextureControlFlowCase::kAllSkipRemaining ||
        mode_ == TextureControlFlowCase::kDivergentSites ? 0 : 1;
    if (post_branch)
      program.push_back(branch);
    const auto second_sample = program.size();
    auto sample = straight[4];
    sample.output_index = 4; // Both paths write the same final color registers.
    program.push_back(sample);
    program.push_back(straight[5]);
    const auto output_index = program.size();
    for (unsigned component = 0; component < 4; ++component) {
      auto output = straight[6 + component];
      output.source.index = 4 + component;
      program.push_back(output);
    }
    if (pre_branch)
      program[pre_index].branch_target_index =
          mode_ == TextureControlFlowCase::kDivergentSites ? second_sample : output_index;
    if (post_branch)
      program[post_index].branch_target_index = output_index;
    for (std::size_t pc = 0; pc < program.size(); ++pc) {
      program[pc].binary_offset = 1 + pc * 8;
      program[pc].group_index = pc;
    }
    pool_.Release(state.fragment_instructions);
    state.fragment_instructions = StoreNewArray(pool_, program);
    auto &summary = state.fragment_program_summary;
    summary.binary_size = program.size() * 8;
    summary.group_count = summary.instruction_count = program.size();
    const auto counts = CountPcoInstructions(program, false);
    auto stats = LoadArray<DrawListStats>(pool_, state.drawlist_stats);
    stats[0].fragment.program_groups = summary.group_count;
    stats[0].fragment.program_instructions = summary.instruction_count;
    stats[0].fragment.program_alu_instructions = counts.alu;
    stats[0].fragment.program_tex_instructions = counts.texture;
    stats[0].fragment.program_memory_instructions = counts.memory;
    pvrgpu::stub::StoreArray(pool_, state.drawlist_stats, stats);

    auto invocations = LoadArray<FragmentInvocation>(pool_, state.fragment_invocations);
    invocations.resize(2, invocations[0]);
    auto lanes = LoadArray<FragmentShaderLane>(pool_, state.fragment_shader_lanes);
    lanes.resize(4, lanes[0]);
    for (unsigned lane = 0; lane < 4; ++lane) {
      lanes[lane].x = lane % 2;
      lanes[lane].y = lane / 2;
      lanes[lane].quad_lane = lane;
      lanes[lane].sample_mask = lane < 2 ? 1 : 0;
      lanes[lane].helper = lane >= 2;
      lanes[lane].visible_invocation_index = lane;
      if (lane < 2) {
        invocations[lane].x = lanes[lane].x;
        invocations[lane].y = lanes[lane].y;
        invocations[lane].quad_lane = lane;
        invocations[lane].sample_mask = 1;
      }
    }
    pool_.Release(state.fragment_invocations);
    pool_.Release(state.fragment_shader_lanes);
    state.fragment_invocations = StoreNewArray(pool_, invocations);
    state.fragment_shader_lanes = StoreNewArray(pool_, lanes);
    auto quads = LoadArray<FragmentQuad>(pool_, state.fragment_quads);
    quads[0].coverage_mask = quads[0].write_mask = 0x03;
    quads[0].helper_mask = 0x0c;
    for (unsigned lane = 0; lane < 4; ++lane)
      quads[0].invocation_indices[lane] = lane;
    pvrgpu::stub::StoreArray(pool_, state.fragment_quads, quads);
    state.active_fragment_invocations = 2;
    state.fragment_shader_lane_count = 4;
    state.counters.ps_invocations = 2;
    StorePipelineState(pool_, payload_.state, state);
    cluster_.input(input_);
    cluster_.output(output_);
    cluster_.texture_request_output(requests_);
    cluster_.texture_response_input(responses_);
    SC_THREAD(Respond);
    input_.write(payload_.txn);
  }

  void Verify() {
    PipelineTxn completed;
    Check(output_.nb_read(completed), std::string(name()) + " must complete");
    const auto state = LoadPipelineState(pool_, payload_.state);
    Check(state.stage == PipelineStage::kFragmentShaded &&
              !HasPoolHandle(state.fragment_continuations) &&
              !HasPoolHandle(state.texture_sample_requests) &&
              !HasPoolHandle(state.texture_sample_responses) &&
              requests_.num_available() == 0 && responses_.num_available() == 0,
          "native branch completion retires every sparse response payload");
    unsigned total = 0;
    for (unsigned lane = 0; lane < 4; ++lane) {
      const auto expected = ExpectedSets(lane);
      Check(issued_[lane] == expected, "only the lane's reached SMP sites request samples");
      total += expected.size();
      const bool discarded = mode_ == TextureControlFlowCase::kDiscardThenMixed && (lane & 1);
      Check(discarded_requests_[lane] == (discarded ? expected.size() : 0),
            "ALPHAF rejection survives every helper SMP checkpoint");
    }
    Check(state.counters.fs_tex_instructions == total &&
              state.counters.fs_alu_instructions ==
                  (mode_ == TextureControlFlowCase::kDivergentSites ? 58U : 56U) &&
              state.counters.fs_memory_instructions ==
                  (mode_ == TextureControlFlowCase::kDiscardThenMixed ? 4U : 0U),
          "actual native branches/helpers count once; skipped SMPs count zero");
    const auto outputs = LoadArray<FragmentOutput>(pool_, state.fragment_outputs);
    Check(outputs.size() == 2, "geometric helpers publish no fragment outputs");
    for (unsigned lane = 0; lane < outputs.size(); ++lane) {
      const auto sets = ExpectedSets(lane);
      const auto expected = sets.empty() ? Fallback() : ResponseForRound(lane * 2 + sets.back());
      Check(outputs[lane].written_mask[0] == 0x0f &&
                std::equal(expected.begin(), expected.end(), outputs[lane].pixel_output) &&
                outputs[lane].discarded ==
                    (mode_ == TextureControlFlowCase::kDiscardThenMixed && (lane & 1)),
            "native branch selects color and feedback rejection remains set for PBE");
    }
    ReleaseFunctionalPayloads(pool_, state);
    pool_.Release(payload_.state);
    Check(pool_.bytes_in_flight() == 0 && pool_.allocations() == pool_.releases(),
          "sparse native continuation ownership balances");
  }

private:
  static std::array<std::uint32_t, 4> Fallback() {
    return {UINT32_C(0x3e000000), UINT32_C(0x3e800000),
            UINT32_C(0x3f000000), UINT32_C(0x3f800000)};
  }

  std::vector<std::uint8_t> ExpectedSets(unsigned lane) const {
    const bool left = (lane & 1) == 0;
    switch (mode_) {
    case TextureControlFlowCase::kAllSkipAll: return {};
    case TextureControlFlowCase::kAllSkipRemaining: return {0};
    case TextureControlFlowCase::kMixedSkipAll:
      return left ? std::vector<std::uint8_t>{} : std::vector<std::uint8_t>{0, 1};
    case TextureControlFlowCase::kDivergentSites:
      return {static_cast<std::uint8_t>(left ? 1 : 0)};
    default:
      return left ? std::vector<std::uint8_t>{0} : std::vector<std::uint8_t>{0, 1};
    }
  }

  void Respond() {
    for (;;) {
      const auto txn = requests_.read();
      auto state = LoadPipelineState(pool_, txn.state);
      const auto requests = LoadArray<TextureSampleRequest>(pool_, state.texture_sample_requests);
      const auto saved = LoadArray<pvrgpu::stub::PcoFragmentContinuation>(pool_, state.fragment_continuations);
      Check(state.stage == PipelineStage::kFragmentTexturePending &&
                !requests.empty() && requests.size() <= 4 && saved.size() == requests.size(),
            "only nonempty native SMP batches cross the FIFO");
      std::vector<TextureSampleResponse> replies;
      for (std::size_t index = 0; index < requests.size(); ++index) {
        const auto &request = requests[index];
        const auto lane = request.shader_lane_index;
        Check(lane < 4 && request.request_id == index &&
                  request.quad_lane == lane && request.shader_stage == ShaderStage::kFragment &&
                  request.descriptor_set == requests[0].descriptor_set &&
                  saved[index].resume_instruction_index == saved[0].resume_instruction_index,
              "sparse batches preserve shader lane identity and group actual SMP sites");
        const auto expected = ExpectedSets(lane);
        Check(issued_[lane].size() < expected.size() &&
                  request.descriptor_set == expected[issued_[lane].size()],
              "no sampler call may be invented for a skipped or completed lane");
        issued_[lane].push_back(request.descriptor_set);
        discarded_requests_[lane] += saved[index].discarded != 0;
        TextureSampleResponse response;
        response.shader_lane_index = lane;
        response.shader_stage = ShaderStage::kFragment;
        response.request_id = request.request_id;
        const auto rgba = ResponseForRound(lane * 2 + request.descriptor_set);
        std::copy(rgba.begin(), rgba.end(), response.rgba);
        replies.push_back(response);
      }
      Check(!HasPoolHandle(state.texture_sample_responses), "response handle starts empty");
      state.texture_sample_responses = StoreNewArray(pool_, replies);
      state.stage = PipelineStage::kTextureSamplesReady;
      StorePipelineState(pool_, txn.state, state);
      responses_.write(txn);
    }
  }

  TextureControlFlowCase mode_;
  MemoryPool pool_;
  CasePayload payload_;
  std::array<std::vector<std::uint8_t>, 4> issued_;
  std::array<unsigned, 4> discarded_requests_{};
  sc_core::sc_fifo<PipelineTxn> input_{"input", 1}, output_{"output", 1},
      requests_{"requests", 1}, responses_{"responses", 1};
  UscCluster cluster_;
};

class DerivativeQuadHarness final : public sc_core::sc_module {
public:
  explicit DerivativeQuadHarness(sc_core::sc_module_name name)
      : sc_module(name), slot_("slot", pool_, ShaderStage::kFragment),
        cluster_("cluster", pool_, ShaderStage::kFragment) {
    payload_ = MakeCase(pool_, 1, 250, 1, 0, false, 4);
    auto state = LoadPipelineState(pool_, payload_.state);
    std::vector<PcoInstruction> instructions(7);
    for (std::size_t pc = 0; pc < instructions.size(); ++pc) {
      auto &op = instructions[pc];
      op.group_index = pc;
      op.binary_offset = 1 + pc * 8;
      op.opcode = PcoOpcode::kMoveBypass;
      op.target = PcoWriteTarget::kTemporary;
      op.output_index = pc;
    }
    instructions[0].source = {PcoRegisterBank::kSpecial, 97};
    for (unsigned axis = 0; axis < 2; ++axis) {
      auto &op = instructions[axis + 1];
      op.opcode = axis ? PcoOpcode::kDerivativeY : PcoOpcode::kDerivativeX;
      op.source = {PcoRegisterBank::kTemporary, 0};
      op.derivative_fine = 1;
    }
    for (unsigned channel = 0; channel < 4; ++channel) {
      auto &op = instructions[channel + 3];
      op.target = PcoWriteTarget::kPixelOutput;
      op.output_index = channel;
      op.source = {PcoRegisterBank::kTemporary,
                   static_cast<std::uint16_t>(channel == 1 ? 2 : 1)};
    }
    instructions.back().end_group = 1;
    pool_.Release(state.fragment_instructions);
    state.fragment_instructions = StoreNewArray(pool_, instructions);
    auto &summary = state.fragment_program_summary;
    summary.binary_size = instructions.size() * 8;
    summary.group_count = summary.instruction_count = instructions.size();
    summary.uses_derivatives = 1;
    auto stats = LoadArray<DrawListStats>(pool_, state.drawlist_stats);
    const auto counts = CountPcoInstructions(instructions, false);
    stats[0].fragment.program_groups = summary.group_count;
    stats[0].fragment.program_instructions = summary.instruction_count;
    stats[0].fragment.program_alu_instructions = counts.alu;
    stats[0].fragment.program_tex_instructions = counts.texture;
    stats[0].fragment.program_memory_instructions = counts.memory;
    pvrgpu::stub::StoreArray(pool_, state.drawlist_stats, stats);
    auto invocations = LoadArray<FragmentInvocation>(pool_, state.fragment_invocations);
    invocations[0].x = invocations[0].y = 0;
    invocations[0].sample_mask = 1;
    pvrgpu::stub::StoreArray(pool_, state.fragment_invocations, invocations);
    auto lanes = LoadArray<FragmentShaderLane>(pool_, state.fragment_shader_lanes);
    lanes.resize(4, lanes[0]);
    for (unsigned lane = 0; lane < 4; ++lane) {
      lanes[lane].x = lane % 2;
      lanes[lane].y = lane / 2;
      lanes[lane].quad_lane = lane;
      lanes[lane].sample_mask = 1;
      lanes[lane].helper = lane != 0;
    }
    pool_.Release(state.fragment_shader_lanes);
    state.fragment_shader_lanes = StoreNewArray(pool_, lanes);
    auto quads = LoadArray<FragmentQuad>(pool_, state.fragment_quads);
    quads[0].helper_mask = 0x0e;
    for (unsigned lane = 0; lane < 4; ++lane)
      quads[0].invocation_indices[lane] = lane;
    pvrgpu::stub::StoreArray(pool_, state.fragment_quads, quads);
    pool_.Release(state.fragment_shared_registers);
    state.fragment_shared_registers = {};
    state.fragment_pco_abi.shareds = 0;
    state.fragment_pco_abi.coefficients = 0;
    state.fragment_position_count = 0;
    state.fragment_varying_start = 0;
    auto tasks = LoadArray<UscFragmentTask>(pool_, state.usc_fragment_tasks);
    tasks[0].coefficient_dword_count = 0;
    pvrgpu::stub::StoreArray(pool_, state.usc_fragment_tasks, tasks);
    pool_.Release(state.usc_coefficient_banks);
    state.usc_coefficient_banks = StoreNewArray(pool_, std::vector<std::uint32_t>{});
    state.sampled_texture_count = 0;
    state.fragment_shader_lane_count = 4;
    state.stage = PipelineStage::kPdsReady;
    state.counters.ps_invocations = 1;
    StorePipelineState(pool_, payload_.state, state);
    slot_.input(input_); slot_.output(issued_);
    cluster_.input(issued_); cluster_.output(output_);
    cluster_.texture_request_output(requests_);
    cluster_.texture_response_input(responses_);
    input_.write(payload_.txn);
  }
  void Verify() {
    PipelineTxn completed;
    Check(output_.nb_read(completed), "derivative quad must complete through USC FIFO");
    const auto state = LoadPipelineState(pool_, payload_.state);
    const auto outputs = LoadArray<FragmentOutput>(pool_, state.fragment_outputs);
    Check(outputs.size() == 1 && outputs[0].pixel_output[0] == 0x3f800000U &&
              outputs[0].pixel_output[1] == 0 && outputs[0].pixel_output[2] == 0x3f800000U &&
              outputs[0].pixel_output[3] == 0x3f800000U &&
              state.counters.fs_tex_instructions == 0 &&
              requests_.num_available() == 0 && responses_.num_available() == 0,
          "helper lanes supply native fine derivatives without texture traffic or helper writes");
    ReleaseFunctionalPayloads(pool_, state);
    pool_.Release(payload_.state);
    Check(pool_.bytes_in_flight() == 0 && pool_.allocations() == pool_.releases(),
          "derivative quad payload ownership balances");
  }
private:
  MemoryPool pool_;
  CasePayload payload_;
  sc_core::sc_fifo<PipelineTxn> input_{"input", 1}, output_{"output", 1},
      issued_{"issued", 1}, requests_{"requests", 1}, responses_{"responses", 1};
  pvrgpu::stub::UscSlot slot_;
  UscCluster cluster_;
};

int RunExpectedFailure(bool too_many_requests, std::string corrupt_continuation = {}) {
  MemoryPool pool;
  const std::size_t sample_count =
      too_many_requests
          ? pvrgpu::stub::kPcoMaximumTextureSampleInstructions + 1U
          : 1U;
  const std::size_t descriptor_count = 1U;
  const CasePayload payload =
      MakeCase(pool, sample_count, 99, descriptor_count);
  sc_core::sc_fifo<PipelineTxn> input("failure_input", 1);
  sc_core::sc_fifo<PipelineTxn> requests("failure_requests", 1);
  sc_core::sc_fifo<PipelineTxn> responses("failure_responses", 1);
  sc_core::sc_fifo<PipelineTxn> output("failure_output", 1);
  UscCluster cluster("failure_cluster", pool, ShaderStage::kFragment);
  TextureResponder responder("failure_responder", pool, sample_count,
                             descriptor_count,
                             !too_many_requests && corrupt_continuation.empty(),
                             false, corrupt_continuation);
  cluster.input(input);
  cluster.texture_request_output(requests);
  cluster.texture_response_input(responses);
  cluster.output(output);
  responder.input(requests);
  responder.output(responses);
  input.write(payload.txn);
  try {
    sc_core::sc_start(sc_core::sc_time(1, sc_core::SC_US));
  } catch (const std::exception &error) {
    const std::string message = error.what();
    const std::string expected =
        too_many_requests ? "task/shared count mismatch"
                          : "response ordering is invalid";
    Check(message.find(expected) != std::string::npos,
          "unexpected fail-closed diagnostic: " + message);
    std::cout << "usc_cluster_texture_continuation_test: expected failure "
              << (too_many_requests ? "request-limit" : "response-order")
              << " PASS\n";
    return 0;
  }
  throw std::runtime_error(
      too_many_requests
          ? "USC accepted a tenth sequential SMP request"
          : "USC accepted a response with the wrong batch-local request id");
}

} // namespace

int sc_main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "request-limit")
      return RunExpectedFailure(true);
    if (argc == 2 && std::string(argv[1]) == "response-order")
      return RunExpectedFailure(false);
    if (argc == 2 && std::string(argv[1]) == "continuation-count")
      return RunExpectedFailure(false, "count");
    if (argc == 2 && std::string(argv[1]) == "continuation-steps")
      return RunExpectedFailure(false, "steps");
    if (argc == 2 && std::string(argv[1]) == "continuation-mask")
      return RunExpectedFailure(false, "mask");
    Check(argc == 1, "unknown test mode");

    DriverPcoStageAbi terrain_d4_fragment_abi;
    terrain_d4_fragment_abi.temps = 31;
    terrain_d4_fragment_abi.coefficients = 12;
    terrain_d4_fragment_abi.shareds = 20;
    terrain_d4_fragment_abi.push_constant_start = 20;
    terrain_d4_fragment_abi.push_constant_count = 0;
    Check(DriverPcoTextureSharedLayoutSupported(terrain_d4_fragment_abi, 1),
          "Terrain D4 canonical empty push range follows descriptor prefix");
    terrain_d4_fragment_abi.push_constant_start = 19;
    Check(!DriverPcoTextureSharedLayoutSupported(terrain_d4_fragment_abi, 1),
          "Terrain D4 empty push range rejects start before prefix end");
    terrain_d4_fragment_abi.push_constant_start = 21;
    Check(!DriverPcoTextureSharedLayoutSupported(terrain_d4_fragment_abi, 1),
          "Terrain D4 empty push range rejects start after prefix end");
    terrain_d4_fragment_abi.push_constant_start = 0;
    Check(DriverPcoTextureSharedLayoutSupported(terrain_d4_fragment_abi, 1),
          "legacy descriptor-only empty push range remains accepted");

    DriverPcoStageAbi mixed_abi;
    mixed_abi.shareds = 32;
    mixed_abi.uniform_buffer_descriptor_start = 20;
    mixed_abi.uniform_buffer_descriptor_count = 2;
    mixed_abi.push_constant_start = 28;
    mixed_abi.push_constant_count = 4;
    Check(DriverPcoTextureSharedLayoutSupported(mixed_abi, 1),
          "texture plus UBO prefix precedes push constants");
    mixed_abi.push_constant_count = 0;
    mixed_abi.shareds = 28;
    Check(DriverPcoTextureSharedLayoutSupported(mixed_abi, 1),
          "UBO empty push suffix uses canonical prefix end");
    mixed_abi.push_constant_start = 0;
    Check(!DriverPcoTextureSharedLayoutSupported(mixed_abi, 1),
          "UBO layouts reject noncanonical legacy empty push start");
    mixed_abi.push_constant_start = 28;
    mixed_abi.push_constant_count = 4;
    mixed_abi.shareds = 32;
    mixed_abi.uniform_buffer_descriptor_start = 19;
    Check(!DriverPcoTextureSharedLayoutSupported(mixed_abi, 1),
          "UBO descriptor cannot overlap texture descriptor");
    mixed_abi.uniform_buffer_descriptor_start = 20;
    mixed_abi.push_constant_start = 20;
    Check(!DriverPcoTextureSharedLayoutSupported(mixed_abi, 1),
          "push constants cannot overlap UBO descriptors");
    mixed_abi.push_constant_start = 28;
    mixed_abi.push_constant_count = 228;
    mixed_abi.shareds = 256;
    Check(DriverPcoTextureSharedLayoutSupported(mixed_abi, 1),
          "256 DWORD shared-register endpoint remains representable");
    mixed_abi.push_constant_count = 229;
    mixed_abi.shareds = 257;
    Check(!DriverPcoTextureSharedLayoutSupported(mixed_abi, 1),
          "shared-register file overflow is rejected");

    MemoryPool one_pool;
    MemoryPool three_pool;
    MemoryPool five_pool;
    MemoryPool nine_pool;
    MemoryPool vertex_pool;
    const CasePayload one = MakeCase(one_pool, 1, 1, 1, 0, true, 12);
    const CasePayload three = MakeCase(three_pool, 3, 2);
    const CasePayload five = MakeCase(five_pool, 5, 3, 5, 64);
    const CasePayload nine = MakeCase(nine_pool, 9, 4, 1);
    const CasePayload vertex = MakeVertexCase(vertex_pool, 5);
    std::vector<std::unique_ptr<UniformTextureHarness>> uniform_cases;
    NativeMultisampleHarness native_ms("native_ms", false);
    NativeMultisampleHarness native_ms_array("native_ms_array", true);
    NativeMultisampleHarness native_ms_query("native_ms_query", false, true);
    NativeMultisampleHarness native_ms_array_query("native_ms_array_query", true, true);
    DerivativeQuadHarness derivative_quad("derivative_quad");
    std::vector<std::unique_ptr<TextureControlFlowHarness>> control_flow_cases;
    for (auto mode : {TextureControlFlowCase::kMixedSkipRemaining,
                      TextureControlFlowCase::kAllSkipRemaining,
                      TextureControlFlowCase::kAllSkipAll,
                      TextureControlFlowCase::kMixedSkipAll,
                      TextureControlFlowCase::kDiscardThenMixed,
                      TextureControlFlowCase::kDivergentSites}) {
      control_flow_cases.emplace_back(new TextureControlFlowHarness(
          sc_core::sc_gen_unique_name("texture_control_flow"), mode));
    }
    for (auto mode : {pvrgpu::stub::MemoryMode::kDirect,
                      pvrgpu::stub::MemoryMode::kBypass,
                      pvrgpu::stub::MemoryMode::kCache}) {
      for (auto stage : {ShaderStage::kVertex, ShaderStage::kFragment})
        uniform_cases.emplace_back(new UniformTextureHarness(
            sc_core::sc_gen_unique_name("uniform_texture"), stage, mode));
    }

    sc_core::sc_fifo<PipelineTxn> one_input("one_input", 1);
    sc_core::sc_fifo<PipelineTxn> one_requests("one_requests", 1);
    sc_core::sc_fifo<PipelineTxn> one_responses("one_responses", 1);
    sc_core::sc_fifo<PipelineTxn> one_output("one_output", 1);
    sc_core::sc_fifo<PipelineTxn> three_input("three_input", 1);
    sc_core::sc_fifo<PipelineTxn> three_requests("three_requests", 1);
    sc_core::sc_fifo<PipelineTxn> three_responses("three_responses", 1);
    sc_core::sc_fifo<PipelineTxn> three_output("three_output", 1);
    sc_core::sc_fifo<PipelineTxn> five_input("five_input", 1);
    sc_core::sc_fifo<PipelineTxn> five_requests("five_requests", 1);
    sc_core::sc_fifo<PipelineTxn> five_responses("five_responses", 1);
    sc_core::sc_fifo<PipelineTxn> five_output("five_output", 1);
    sc_core::sc_fifo<PipelineTxn> nine_input("nine_input", 1);
    sc_core::sc_fifo<PipelineTxn> nine_requests("nine_requests", 1);
    sc_core::sc_fifo<PipelineTxn> nine_responses("nine_responses", 1);
    sc_core::sc_fifo<PipelineTxn> nine_output("nine_output", 1);
    sc_core::sc_fifo<PipelineTxn> vertex_input("vertex_input", 1);
    sc_core::sc_fifo<PipelineTxn> vertex_requests("vertex_requests", 1);
    sc_core::sc_fifo<PipelineTxn> vertex_responses("vertex_responses", 1);
    sc_core::sc_fifo<PipelineTxn> vertex_output("vertex_output", 1);

    UscCluster one_cluster("one_cluster", one_pool, ShaderStage::kFragment);
    TextureResponder one_responder("one_responder", one_pool, 1, 1);
    one_cluster.input(one_input);
    one_cluster.texture_request_output(one_requests);
    one_cluster.texture_response_input(one_responses);
    one_cluster.output(one_output);
    one_responder.input(one_requests);
    one_responder.output(one_responses);

    UscCluster three_cluster("three_cluster", three_pool,
                             ShaderStage::kFragment);
    TextureResponder three_responder("three_responder", three_pool, 3, 3);
    three_cluster.input(three_input);
    three_cluster.texture_request_output(three_requests);
    three_cluster.texture_response_input(three_responses);
    three_cluster.output(three_output);
    three_responder.input(three_requests);
    three_responder.output(three_responses);

    UscCluster five_cluster("five_cluster", five_pool,
                            ShaderStage::kFragment);
    TextureResponder five_responder("five_responder", five_pool, 5, 5);
    five_cluster.input(five_input);
    five_cluster.texture_request_output(five_requests);
    five_cluster.texture_response_input(five_responses);
    five_cluster.output(five_output);
    five_responder.input(five_requests);
    five_responder.output(five_responses);

    UscCluster nine_cluster("nine_cluster", nine_pool,
                            ShaderStage::kFragment);
    TextureResponder nine_responder("nine_responder", nine_pool, 9, 1);
    nine_cluster.input(nine_input);
    nine_cluster.texture_request_output(nine_requests);
    nine_cluster.texture_response_input(nine_responses);
    nine_cluster.output(nine_output);
    nine_responder.input(nine_requests);
    nine_responder.output(nine_responses);

    UscCluster vertex_cluster("vertex_cluster", vertex_pool,
                              ShaderStage::kVertex);
    VertexTextureResponder vertex_responder("vertex_responder", vertex_pool);
    vertex_cluster.input(vertex_input);
    vertex_cluster.texture_request_output(vertex_requests);
    vertex_cluster.texture_response_input(vertex_responses);
    vertex_cluster.output(vertex_output);
    vertex_responder.input(vertex_requests);
    vertex_responder.output(vertex_responses);

    one_input.write(one.txn);
    three_input.write(three.txn);
    five_input.write(five.txn);
    nine_input.write(nine.txn);
    vertex_input.write(vertex.txn);
    sc_core::sc_start(sc_core::sc_time(1, sc_core::SC_US));
    sc_core::sc_start(sc_core::SC_ZERO_TIME);

    PipelineTxn completed;
    Check(one_output.nb_read(completed) && completed.sequence == 1,
          "one-request case completes");
    Check(three_output.nb_read(completed) && completed.sequence == 2,
          "three-request case completes");
    Check(five_output.nb_read(completed) && completed.sequence == 3,
          "five-request five-descriptor push-constant case completes");
    Check(nine_output.nb_read(completed) && completed.sequence == 4,
          "nine-request one-descriptor case completes");
    Check(vertex_output.nb_read(completed) && completed.sequence == 5,
          "two-request Terrain vertex case completes");
    CheckCompletedCase(one_pool, one, one_responder, 1);
    CheckCompletedCase(three_pool, three, three_responder, 3);
    CheckCompletedCase(five_pool, five, five_responder, 5);
    CheckCompletedCase(nine_pool, nine, nine_responder, 9);
    CheckCompletedVertexCase(vertex_pool, vertex);
    for (const auto &test : uniform_cases)
      test->Verify();
    native_ms.Verify();
    native_ms_array.Verify();
    native_ms_query.Verify();
    native_ms_array_query.Verify();
    derivative_quad.Verify();
    for (const auto &test : control_flow_cases)
      test->Verify();

    std::cout << "usc_cluster_texture_continuation_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "usc_cluster_texture_continuation_test: FAIL: "
              << error.what() << '\n';
    return 1;
  }
}
