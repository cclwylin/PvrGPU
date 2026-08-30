// UscCluster executes decoded PowerVR PCO instructions with the USC ISS.
// USC means Unified Shading Cluster and ISS means Instruction Set Simulator.
// Vertex lanes expose raw VTXIN/VTXOUT register bits; fragment invocations
// receive independent raw PIXOUT0..3 results. No shader enum, global color, or
// name-specific branch supplies functional results. Per-DrawList dynamic
// ALU/Tex/Memory totals expand PCO repeat and multiply by actual lane
// invocations. FIFO traffic carries only the MemoryPool PipelineState handle
// and completion is event-driven.
#include "shader/usc_cluster.h"

#include "common/functional_types.h"
#include "common/pipeline_state.h"
#include "shader/pco_iss.h"

#include <iterator>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pvrgpu::stub {
namespace {

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::uint64_t CheckedInstructionTotal(std::uint64_t per_invocation,
                                      std::uint64_t invocations) {
  if (per_invocation != 0 &&
      invocations >
          std::numeric_limits<std::uint64_t>::max() / per_invocation) {
    throw std::overflow_error("USC instruction counter overflow");
  }
  return per_invocation * invocations;
}

void AddInstructionCounter(std::uint64_t &counter, std::uint64_t amount) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - counter)
    throw std::overflow_error("USC aggregate instruction counter overflow");
  counter += amount;
}

void RecordInstructionExecutions(
    CounterTxn &counters, DrawListShaderStats &stats, ShaderStage stage,
    const std::vector<PcoInstruction> &instructions,
    std::uint64_t logical_invocations, std::uint64_t execution_lanes) {
  const PcoInstructionCounts static_counts =
      CountPcoInstructions(instructions, false);
  if (stats.program_recorded != 1 ||
      stats.program_instructions != instructions.size() ||
      stats.program_alu_instructions != static_counts.alu ||
      stats.program_tex_instructions != static_counts.texture ||
      stats.program_memory_instructions != static_counts.memory) {
    throw std::runtime_error("USC DrawList program statistics mismatch");
  }
  if (stats.executions_recorded != 0) {
    throw std::runtime_error("USC DrawList executions were counted twice");
  }

  const PcoInstructionCounts per_invocation =
      CountPcoInstructions(instructions, true);
  stats.invocations = logical_invocations;
  stats.executed_alu_instructions =
      CheckedInstructionTotal(per_invocation.alu, execution_lanes);
  stats.executed_tex_instructions =
      CheckedInstructionTotal(per_invocation.texture, execution_lanes);
  stats.executed_memory_instructions =
      CheckedInstructionTotal(per_invocation.memory, execution_lanes);
  stats.executions_recorded = 1;

  if (stage == ShaderStage::kVertex) {
    AddInstructionCounter(counters.vs_alu_instructions,
                          stats.executed_alu_instructions);
    AddInstructionCounter(counters.vs_tex_instructions,
                          stats.executed_tex_instructions);
    AddInstructionCounter(counters.vs_memory_instructions,
                          stats.executed_memory_instructions);
  } else {
    AddInstructionCounter(counters.fs_alu_instructions,
                          stats.executed_alu_instructions);
    AddInstructionCounter(counters.fs_tex_instructions,
                          stats.executed_tex_instructions);
    AddInstructionCounter(counters.fs_memory_instructions,
                          stats.executed_memory_instructions);
  }
}

} // namespace

UscCluster::UscCluster(sc_core::sc_module_name name, MemoryPool &pool,
                       ShaderStage stage)
    : sc_module(name), pool_(pool), stage_(stage) {
  SC_THREAD(Run);
}

void UscCluster::Run() {
  while (true) {
    const PipelineTxn txn = input.read();
    PipelineState state = LoadPipelineState(pool_, txn.state);
    if (!IsRasterFunctionalCase(state.functional_case))
      throw std::runtime_error("USC cluster received an unsupported case");

    const std::uint64_t groups = stage_ == ShaderStage::kVertex
                                     ? state.vertex_groups
                                     : state.fragment_groups;
    if (!HasPoolHandle(state.drawlist_stats))
      throw std::runtime_error("USC cluster received no DrawList statistics");
    std::vector<DrawListStats> drawlists =
        LoadArray<DrawListStats>(pool_, state.drawlist_stats);
    if (drawlists.size() != 1 || drawlists[0].drawlist_index != 0)
      throw std::runtime_error("USC cluster requires DrawList 0");
    if (stage_ == ShaderStage::kVertex) {
      RequireStage(state.stage, PipelineStage::kVertexIssued, name());
      if (!HasPoolHandle(state.vertex_lanes) ||
          !HasPoolHandle(state.vertex_instructions)) {
        throw std::runtime_error("vertex USC has no lane/program payload");
      }
      const std::vector<PcoInstruction> instructions =
          LoadArray<PcoInstruction>(pool_, state.vertex_instructions);
      std::vector<VertexLane> lanes =
          LoadArray<VertexLane>(pool_, state.vertex_lanes);
      if (lanes.size() != state.counters.vs_invocations)
        throw std::runtime_error("vertex USC lane count mismatch");
      PcoVertexExecutionContext vertex_context;
      const bool texture_case = IsTextureFamily(state.functional_case);
      if (texture_case) {
        if (!HasPoolHandle(state.vertex_shared_registers))
          throw std::runtime_error("vertex USC has no shared-register payload");
        const std::vector<ShaderSharedRegister> shared =
            LoadArray<ShaderSharedRegister>(pool_,
                                            state.vertex_shared_registers);
        if (shared.size() != kPcoFillTexNearestVertexSharedCount)
          throw std::runtime_error("vertex USC shared-register count mismatch");
        vertex_context.shared_count = static_cast<std::uint8_t>(shared.size());
        for (std::size_t index = 0; index < shared.size(); ++index)
          vertex_context.shared_registers[index] = shared[index].value;
      } else if (HasPoolHandle(state.vertex_shared_registers)) {
        throw std::runtime_error(
            "vertex USC non-texture case has shared registers");
      }
      for (VertexLane &lane : lanes) {
        const std::vector<std::uint32_t> inputs(std::begin(lane.vertex_input),
                                                std::end(lane.vertex_input));
        const PcoVertexExecution execution =
            texture_case
                ? ExecuteVertexPco(state.vertex_program_summary, instructions,
                                   inputs, vertex_context)
                : ExecuteVertexPco(state.vertex_program_summary, instructions,
                                   inputs);
        for (std::size_t index = 0; index < kPcoVertexOutputCount; ++index)
          lane.vertex_output[index] = execution.outputs[index];
        lane.emitted = execution.emitted;
        lane.ended = execution.ended_task;
      }
      RecordInstructionExecutions(state.counters, drawlists[0].vertex, stage_,
                                  instructions, lanes.size(), lanes.size());
      StoreArray(pool_, state.vertex_lanes, lanes);
      state.stage = PipelineStage::kVertexShaded;
    } else {
      RequireStage(state.stage, PipelineStage::kFragmentIssued, name());
      if (!HasPoolHandle(state.fragment_invocations) ||
          !HasPoolHandle(state.fragment_instructions)) {
        throw std::runtime_error("fragment USC has no work/program payload");
      }
      const std::vector<FragmentInvocation> invocations =
          LoadArray<FragmentInvocation>(pool_, state.fragment_invocations);
      const std::vector<PcoInstruction> instructions =
          LoadArray<PcoInstruction>(pool_, state.fragment_instructions);
      if (invocations.size() != state.active_fragment_invocations)
        throw std::runtime_error("fragment USC invocation count mismatch");

      std::vector<FragmentOutput> outputs(invocations.size());
      std::vector<std::uint8_t> output_written(invocations.size(), 0);
      const auto execute_invocation =
          [&](std::size_t invocation_index,
              const PcoFragmentExecutionContext *context) {
        if (invocation_index >= invocations.size() ||
            output_written[invocation_index] != 0) {
          throw std::runtime_error(
              "fragment USC invocation task is invalid or duplicated");
        }
        const FragmentInvocation &invocation = invocations[invocation_index];
        const PcoFragmentExecution execution =
            context ? ExecuteFragmentPco(state.fragment_program_summary,
                                         instructions, *context)
                    : ExecuteFragmentPco(state.fragment_program_summary,
                                         instructions);
        FragmentOutput fragment_output;
        fragment_output.x = invocation.x;
        fragment_output.y = invocation.y;
        fragment_output.primitive_id = invocation.primitive_id;
        fragment_output.parameter_index = invocation.parameter_index;
        fragment_output.submit_ordinal = invocation.submit_ordinal;
        fragment_output.depth = invocation.depth;
        for (std::size_t component = 0; component < 4; ++component) {
          fragment_output.pixel_output[component] =
              execution.pixel_outputs[component];
        }
        fragment_output.written_mask = execution.written_mask;
        outputs[invocation_index] = fragment_output;
        output_written[invocation_index] = 1;
      };

      std::uint64_t fragment_execution_lanes = invocations.size();
      if (IsTextureFamily(state.functional_case)) {
        if (texture_request_output.size() == 0 ||
            texture_response_input.size() == 0 ||
            !HasPoolHandle(state.fragment_quads) ||
            !HasPoolHandle(state.fragment_shader_lanes) ||
            !HasPoolHandle(state.usc_fragment_tasks) ||
            !HasPoolHandle(state.usc_coefficient_banks) ||
            !HasPoolHandle(state.fragment_shared_registers)) {
          throw std::runtime_error(
              "texture fragment USC has incomplete request plumbing");
        }
        const std::vector<FragmentQuad> quads =
            LoadArray<FragmentQuad>(pool_, state.fragment_quads);
        const std::vector<FragmentShaderLane> shader_lanes =
            LoadArray<FragmentShaderLane>(pool_, state.fragment_shader_lanes);
        const std::vector<UscFragmentTask> tasks =
            LoadArray<UscFragmentTask>(pool_, state.usc_fragment_tasks);
        const std::vector<std::uint32_t> coefficient_bank =
            LoadArray<std::uint32_t>(pool_, state.usc_coefficient_banks);
        const std::vector<std::uint32_t> shared_registers =
            LoadArray<std::uint32_t>(pool_,
                                     state.fragment_shared_registers);
        if (shader_lanes.size() != state.fragment_shader_lane_count ||
            tasks.size() != state.fragment_groups ||
            quads.size() != tasks.size() ||
            shared_registers.size() !=
                kPcoFillTexNearestFragmentSharedCount) {
          throw std::runtime_error(
              "texture fragment USC task/shared count mismatch");
        }
        fragment_execution_lanes = shader_lanes.size();
        std::vector<PcoFragmentContinuation> continuations(
            shader_lanes.size());
        std::vector<TextureSampleRequest> requests;
        requests.reserve(shader_lanes.size());
        for (const UscFragmentTask &task : tasks) {
          if (task.fragment_quad_index >= quads.size() ||
              task.coefficient_dword_count !=
                  kFillTexNearestCoefficientDwordCount ||
              task.first_coefficient_dword > coefficient_bank.size() ||
              task.coefficient_dword_count >
                  coefficient_bank.size() - task.first_coefficient_dword) {
            throw std::runtime_error(
                "texture fragment USC coefficient task is out of range");
          }
          const FragmentQuad &quad = quads[task.fragment_quad_index];
          const std::uint8_t active_mask = static_cast<std::uint8_t>(
              quad.coverage_mask | quad.helper_mask);
          if (active_mask == 0 ||
              (quad.coverage_mask & quad.helper_mask) != 0 ||
              quad.write_mask != quad.coverage_mask) {
            throw std::runtime_error(
                "texture fragment USC received an invalid quad mask");
          }
          PcoFragmentExecutionContext context;
          context.coefficient_count = static_cast<std::uint8_t>(
              task.coefficient_dword_count);
          context.shared_count = static_cast<std::uint8_t>(
              shared_registers.size());
          for (std::size_t dword = 0;
               dword < task.coefficient_dword_count; ++dword) {
            context.coefficients[dword] =
                coefficient_bank[task.first_coefficient_dword + dword];
          }
          for (std::size_t dword = 0; dword < shared_registers.size();
               ++dword)
            context.shared_registers[dword] = shared_registers[dword];
          for (std::uint8_t lane = 0; lane < 4U; ++lane) {
            if ((active_mask & (1U << lane)) == 0)
              continue;
            const std::uint32_t shader_lane_index =
                quad.invocation_indices[lane];
            if (shader_lane_index >= shader_lanes.size())
              throw std::runtime_error(
                  "texture fragment USC shader lane is out of range");
            const FragmentShaderLane &shader_lane =
                shader_lanes[shader_lane_index];
            if (shader_lane.quad_id != quad.quad_id ||
                shader_lane.quad_lane != lane ||
                shader_lane.parameter_index != quad.parameter_index ||
                shader_lane.submit_ordinal != quad.submit_ordinal) {
              throw std::runtime_error(
                  "texture fragment USC lost shader-lane identity");
            }
            context.sample_x =
                FloatBits(static_cast<float>(shader_lane.x) + 0.5F);
            context.sample_y =
                FloatBits(static_cast<float>(shader_lane.y) + 0.5F);
            const PcoFragmentExecution execution = ExecuteFragmentPco(
                state.fragment_program_summary, instructions, context);
            if (execution.suspended != 1 ||
                execution.texture_request_valid != 1 ||
                execution.continuation.valid != 1 ||
                execution.written_mask != 0 || execution.discarded) {
              throw std::runtime_error(
                  "texture fragment USC did not suspend at SMP");
            }
            continuations[shader_lane_index] = execution.continuation;
            TextureSampleRequest request;
            request.shader_lane_index = shader_lane_index;
            request.quad_id = shader_lane.quad_id;
            request.quad_lane = shader_lane.quad_lane;
            request.request_id = requests.size();
            for (std::size_t component = 0; component < 2; ++component)
              request.coordinates[component] =
                  execution.texture_request.coordinates[component];
            for (std::size_t dword = 0; dword < 4; ++dword) {
              request.texture_state[dword] =
                  execution.texture_request.texture_state[dword];
              request.sampler_state[dword] =
                  execution.texture_request.sampler_state[dword];
            }
            request.coordinate_count =
                execution.texture_request.coordinate_count;
            request.component_count =
                execution.texture_request.component_count;
            request.descriptor_set =
                execution.texture_request.descriptor_set;
            request.binding = execution.texture_request.binding;
            request.dimension = execution.texture_request.dimension;
            request.normalized = execution.texture_request.normalized;
            request.data_request = execution.texture_request.data_request;
            requests.push_back(request);
          }
        }
        if (requests.size() != shader_lanes.size())
          throw std::runtime_error(
              "texture fragment USC did not issue one SMP per shader lane");
        state.texture_sample_requests = StoreNewArray(pool_, requests);
        state.fragment_continuations = StoreNewArray(pool_, continuations);
        state.stage = PipelineStage::kFragmentTexturePending;
        StorePipelineState(pool_, txn.state, state);
        texture_request_output->write(txn);

        const PipelineTxn response_txn = texture_response_input->read();
        if (response_txn.state.slot != txn.state.slot ||
            response_txn.state.generation != txn.state.generation ||
            response_txn.sequence != txn.sequence ||
            response_txn.frame != txn.frame) {
          throw std::runtime_error(
              "texture fragment USC response identity mismatch");
        }
        state = LoadPipelineState(pool_, txn.state);
        RequireStage(state.stage, PipelineStage::kTextureSamplesReady, name());
        if (!HasPoolHandle(state.texture_sample_responses) ||
            !HasPoolHandle(state.fragment_continuations)) {
          throw std::runtime_error(
              "texture fragment USC received no response/continuation");
        }
        const std::vector<TextureSampleResponse> responses =
            LoadArray<TextureSampleResponse>(pool_,
                                             state.texture_sample_responses);
        const std::vector<PcoFragmentContinuation> saved_continuations =
            LoadArray<PcoFragmentContinuation>(pool_,
                                                state.fragment_continuations);
        if (responses.size() != shader_lanes.size() ||
            saved_continuations.size() != shader_lanes.size()) {
          throw std::runtime_error(
              "texture fragment USC response lane count mismatch");
        }
        for (std::size_t lane_index = 0; lane_index < shader_lanes.size();
             ++lane_index) {
          const TextureSampleResponse &response = responses[lane_index];
          if (response.shader_lane_index != lane_index ||
              response.request_id != lane_index ||
              saved_continuations[lane_index].valid != 1) {
            throw std::runtime_error(
                "texture fragment USC response ordering is invalid");
          }
          std::array<std::uint32_t, kPcoPixelOutputCount> rgba{};
          for (std::size_t component = 0; component < 4; ++component)
            rgba[component] = response.rgba[component];
          const PcoFragmentExecution execution = ResumeFragmentPco(
              state.fragment_program_summary, instructions,
              saved_continuations[lane_index], rgba);
          if (execution.suspended != 0 ||
              execution.texture_request_valid != 0 ||
              execution.written_mask != 0x0f || execution.discarded) {
            throw std::runtime_error(
                "texture fragment USC continuation did not complete PIXOUT");
          }
          const FragmentShaderLane &shader_lane = shader_lanes[lane_index];
          if (shader_lane.helper)
            continue;
          const std::uint32_t invocation_index =
              shader_lane.visible_invocation_index;
          if (invocation_index >= invocations.size() ||
              output_written[invocation_index] != 0)
            throw std::runtime_error(
                "texture fragment USC visible-lane mapping is invalid");
          const FragmentInvocation &invocation = invocations[invocation_index];
          FragmentOutput fragment_output;
          fragment_output.x = invocation.x;
          fragment_output.y = invocation.y;
          fragment_output.primitive_id = invocation.primitive_id;
          fragment_output.parameter_index = invocation.parameter_index;
          fragment_output.submit_ordinal = invocation.submit_ordinal;
          fragment_output.depth = invocation.depth;
          for (std::size_t component = 0; component < 4; ++component)
            fragment_output.pixel_output[component] =
                execution.pixel_outputs[component];
          fragment_output.written_mask = execution.written_mask;
          outputs[invocation_index] = fragment_output;
          output_written[invocation_index] = 1;
        }
        const PoolHandle request_payload = state.texture_sample_requests;
        const PoolHandle response_payload = state.texture_sample_responses;
        const PoolHandle continuation_payload = state.fragment_continuations;
        state.texture_sample_requests = {};
        state.texture_sample_responses = {};
        state.fragment_continuations = {};
        pool_.Release(request_payload);
        pool_.Release(response_payload);
        pool_.Release(continuation_payload);
      } else if (IsVaryingsFamily(state.functional_case)) {
        if (!HasPoolHandle(state.fragment_quads) ||
            !HasPoolHandle(state.usc_fragment_tasks) ||
            !HasPoolHandle(state.usc_coefficient_banks)) {
          throw std::runtime_error(
              "varying fragment USC received no PDS coefficient tasks");
        }
        const std::vector<FragmentQuad> quads =
            LoadArray<FragmentQuad>(pool_, state.fragment_quads);
        const std::vector<UscFragmentTask> tasks =
            LoadArray<UscFragmentTask>(pool_, state.usc_fragment_tasks);
        const std::vector<std::uint32_t> coefficient_bank =
            LoadArray<std::uint32_t>(pool_, state.usc_coefficient_banks);
        if (tasks.size() != state.fragment_groups ||
            quads.size() != tasks.size()) {
          throw std::runtime_error(
              "varying fragment USC task/group count mismatch");
        }
        const std::uint32_t expected_coefficient_dwords =
            VaryingCoefficientDwordCount(state.functional_case);
        for (const UscFragmentTask &task : tasks) {
          if (task.fragment_quad_index >= quads.size() ||
              task.coefficient_dword_count !=
                  expected_coefficient_dwords ||
              task.first_coefficient_dword > coefficient_bank.size() ||
              task.coefficient_dword_count >
                  coefficient_bank.size() - task.first_coefficient_dword) {
            throw std::runtime_error(
                "varying fragment USC coefficient task is out of range");
          }
          const FragmentQuad &quad = quads[task.fragment_quad_index];
          if (quad.helper_mask != 0 || quad.coverage_mask == 0 ||
              quad.write_mask != quad.coverage_mask) {
            throw std::runtime_error(
                "varying fragment USC received an invalid quad lane mask");
          }
          PcoFragmentExecutionContext context;
          if (task.coefficient_dword_count > context.coefficients.size() ||
              task.coefficient_dword_count >
                  std::numeric_limits<std::uint8_t>::max()) {
            throw std::runtime_error(
                "varying fragment USC coefficient context is too small");
          }
          context.coefficient_count = static_cast<std::uint8_t>(
              task.coefficient_dword_count);
          for (std::size_t dword = 0;
               dword < task.coefficient_dword_count; ++dword) {
            context.coefficients[dword] =
                coefficient_bank[task.first_coefficient_dword + dword];
          }
          for (std::uint8_t lane = 0;
               lane < kReferenceUarch.fragment_quad_width *
                          kReferenceUarch.fragment_quad_height;
               ++lane) {
            const bool covered = (quad.write_mask & (1U << lane)) != 0;
            const std::uint32_t invocation_index =
                quad.invocation_indices[lane];
            if (!covered) {
              if (invocation_index != kInvalidFragmentInvocationIndex) {
                throw std::runtime_error(
                    "varying fragment USC uncovered lane has an invocation");
              }
              continue;
            }
            if (invocation_index >= invocations.size()) {
              throw std::runtime_error(
                  "varying fragment USC quad invocation is out of range");
            }
            const FragmentInvocation &invocation =
                invocations[invocation_index];
            if (invocation.parameter_index != quad.parameter_index ||
                invocation.quad_id != quad.quad_id ||
                invocation.quad_lane != lane ||
                invocation.submit_ordinal != quad.submit_ordinal) {
              throw std::runtime_error(
                  "varying fragment USC lost PDS quad identity");
            }
            context.sample_x =
                FloatBits(static_cast<float>(invocation.x) + 0.5F);
            context.sample_y =
                FloatBits(static_cast<float>(invocation.y) + 0.5F);
            execute_invocation(invocation_index, &context);
          }
        }
      } else {
        for (std::size_t index = 0; index < invocations.size(); ++index)
          execute_invocation(index, nullptr);
      }
      for (const std::uint8_t written : output_written) {
        if (written != 1) {
          throw std::runtime_error(
              "fragment USC did not execute every visible invocation");
        }
      }
      RecordInstructionExecutions(state.counters, drawlists[0].fragment, stage_,
                                  instructions, invocations.size(),
                                  fragment_execution_lanes);
      state.fragment_outputs = StoreNewArray(pool_, outputs);
      state.stage = PipelineStage::kFragmentShaded;
    }

    const std::uint64_t cycles =
        groups == 0
            ? 0
            : kReferenceUarch.usc_cluster_base_cycles +
                  CeilDivide(groups,
                             kReferenceUarch.usc_groups_per_cluster_batch);
    state.counters.usc_groups += groups;
    state.counters.usc_cluster_cycles += cycles;
    if (stage_ == ShaderStage::kVertex)
      state.counters.tiler_cycles += cycles;
    else
      state.counters.renderer_cycles += cycles;
    StoreArray(pool_, state.drawlist_stats, drawlists);

    WaitForCycles(cycles);
    StorePipelineState(pool_, txn.state, state);
    output.write(txn);
  }
}

} // namespace pvrgpu::stub
