// Focused event-path regression for fragment USC texture continuation.
// One lane is enough to prove that a response resumes at WDF, retains the
// coefficient/shared context needed by the next SMP, and emits PIXOUT only
// after the final response.  The responder deliberately sits on the other
// side of the real SystemC FIFO and MemoryPool boundary.

#include "common/pipeline_state.h"
#include "shader/pco_iss.h"
#include "shader/usc_cluster.h"

#include <systemc>

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
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
  state.varying_output_count = 1;
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
                   bool corrupt_response_order = false)
      : sc_module(name), pool_(pool), expected_rounds_(expected_rounds),
        descriptor_count_(descriptor_count),
        corrupt_response_order_(corrupt_response_order) {
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
      state.stage = PipelineStage::kTextureSamplesReady;
      StorePipelineState(pool_, txn.state, state);
      output.write(txn);
    }
  }

  MemoryPool &pool_;
  std::size_t expected_rounds_ = 0;
  std::size_t descriptor_count_ = 0;
  bool corrupt_response_order_ = false;
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
  Check(outputs.size() == 1 && outputs[0].written_mask == 0x0f &&
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
        "static SMP count is expanded once per lane");

  ReleaseFunctionalPayloads(pool, state);
  pool.Release(payload.state);
  Check(pool.bytes_in_flight() == 0 &&
            pool.allocations() == pool.releases(),
        "MemoryPool allocations are balanced");
}

int RunExpectedFailure(bool too_many_requests) {
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
                             descriptor_count, !too_many_requests);
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

    std::cout << "usc_cluster_texture_continuation_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "usc_cluster_texture_continuation_test: FAIL: "
              << error.what() << '\n';
    return 1;
  }
}
