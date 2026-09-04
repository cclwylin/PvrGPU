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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pvrgpu::stub {

bool DriverPcoTextureSharedLayoutSupported(
    const DriverPcoStageAbi &abi, std::uint32_t descriptor_set_count) {
  if (descriptor_set_count == 0 ||
      descriptor_set_count > kPcoMaximumTextureDescriptorSets) {
    return false;
  }
  const std::uint64_t descriptor_shared_dwords =
      static_cast<std::uint64_t>(descriptor_set_count) *
      kPcoTextureDescriptorDwordCount;
  if (abi.push_constant_count == 0) {
    return abi.shareds == descriptor_shared_dwords &&
           (abi.push_constant_start == 0 ||
            abi.push_constant_start == descriptor_shared_dwords);
  }
  const std::uint64_t push_end =
      static_cast<std::uint64_t>(abi.push_constant_start) +
      abi.push_constant_count;
  return abi.push_constant_start == descriptor_shared_dwords &&
         push_end == abi.shareds;
}

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

std::uint32_t DebugFragmentCoordinate(const char *name,
                                      std::uint32_t fallback) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0')
    return fallback;
  char *end = nullptr;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (end == value || *end != '\0' ||
      parsed > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error(std::string("invalid debug coordinate in ") +
                             name);
  }
  return static_cast<std::uint32_t>(parsed);
}

bool SameTextureSampleRequest(const TextureSampleRequest &left,
                              const TextureSampleRequest &right) {
  return left.shader_lane_index == right.shader_lane_index &&
         left.quad_id == right.quad_id &&
         std::equal(std::begin(left.coordinates), std::end(left.coordinates),
                    std::begin(right.coordinates)) &&
         std::equal(std::begin(left.texture_state),
                    std::end(left.texture_state),
                    std::begin(right.texture_state)) &&
         std::equal(std::begin(left.sampler_state),
                    std::end(left.sampler_state),
                    std::begin(right.sampler_state)) &&
         left.request_id == right.request_id &&
         left.coordinate_count == right.coordinate_count &&
         left.component_count == right.component_count &&
         left.descriptor_set == right.descriptor_set &&
         left.binding == right.binding && left.dimension == right.dimension &&
         left.normalized == right.normalized &&
         left.data_request == right.data_request &&
         left.quad_lane == right.quad_lane &&
         left.shader_stage == right.shader_stage && left.reserved[0] == 0 &&
         left.reserved[1] == 0 && left.reserved[2] == 0 &&
         right.reserved[0] == 0 && right.reserved[1] == 0 &&
         right.reserved[2] == 0;
}

bool SameVertexContinuation(const PcoVertexContinuation &left,
                            const PcoVertexContinuation &right) {
  return left.vertex_inputs == right.vertex_inputs &&
         left.shared_registers == right.shared_registers &&
         left.temporaries == right.temporaries &&
         left.outputs == right.outputs &&
         left.temporary_written_mask == right.temporary_written_mask &&
         left.output_written_mask == right.output_written_mask &&
         left.program_binary_size == right.program_binary_size &&
         left.program_instruction_count == right.program_instruction_count &&
         left.resume_instruction_index == right.resume_instruction_index &&
         left.pending_output_index == right.pending_output_index &&
         left.pending_component_count == right.pending_component_count &&
         left.data_request == right.data_request &&
         left.vertex_input_count == right.vertex_input_count &&
         left.shared_count == right.shared_count &&
         left.emitted == right.emitted &&
         left.ended_task == right.ended_task && left.valid == right.valid;
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
      const bool vertex_texture_case =
          UsesTextureSampling(state, ShaderStage::kVertex);
      const bool driver_pco =
          IsDriverPcoTrianglesCase(state.functional_case);
      const bool vertex_requires_context =
          UsesTextureSampling(state.functional_case) || driver_pco;
      if (vertex_requires_context) {
        const std::size_t expected_shared_count =
            driver_pco ? state.vertex_pco_abi.shareds
                       : kPcoFillTexNearestVertexSharedCount;
        if (expected_shared_count > kPcoMaximumVertexSharedCount ||
            (expected_shared_count != 0 &&
             !HasPoolHandle(state.vertex_shared_registers)) ||
            (expected_shared_count == 0 &&
             HasPoolHandle(state.vertex_shared_registers))) {
          throw std::runtime_error(
              "vertex USC shared-register handle/count mismatch");
        }
        const std::vector<ShaderSharedRegister> shared =
            expected_shared_count == 0
                ? std::vector<ShaderSharedRegister>{}
                : LoadArray<ShaderSharedRegister>(
                      pool_, state.vertex_shared_registers);
        if (shared.size() != expected_shared_count) {
          throw std::runtime_error("vertex USC shared-register count mismatch");
        }
        vertex_context.shared_count = static_cast<std::uint8_t>(shared.size());
        for (std::size_t index = 0; index < shared.size(); ++index)
          vertex_context.shared_registers[index] = shared[index].value;
      } else if (HasPoolHandle(state.vertex_shared_registers)) {
        throw std::runtime_error(
              "vertex USC non-texture case has shared registers");
      }
      if (vertex_texture_case) {
        const std::uint32_t descriptor_set_count =
            state.vertex_sampled_texture_count;
        const bool shared_layout_valid =
            DriverPcoTextureSharedLayoutSupported(
                state.vertex_pco_abi, descriptor_set_count);
        const std::size_t sample_instruction_count =
            static_cast<std::size_t>(std::count_if(
                instructions.begin(), instructions.end(),
                [](const PcoInstruction &instruction) {
                  return instruction.opcode == PcoOpcode::kTextureSample;
                }));
        if (!driver_pco || texture_request_output.size() == 0 ||
            texture_response_input.size() == 0 || lanes.empty() ||
            descriptor_set_count == 0 ||
            descriptor_set_count > kPcoMaximumTextureDescriptorSets ||
            !shared_layout_valid ||
            state.vertex_pco_abi.shareds > kPcoMaximumVertexSharedCount ||
            sample_instruction_count == 0 ||
            sample_instruction_count > kPcoMaximumTextureSampleInstructions ||
            !HasPoolHandle(state.vertex_shared_registers)) {
          throw std::runtime_error(
              "texture vertex USC task/shared count mismatch");
        }
        if (HasPoolHandle(state.texture_sample_requests) ||
            HasPoolHandle(state.texture_sample_responses) ||
            HasPoolHandle(state.vertex_continuations) ||
            HasPoolHandle(state.fragment_continuations)) {
          throw std::runtime_error(
              "texture vertex USC received stale continuation payloads");
        }

        std::vector<std::uint8_t> lane_request_count(lanes.size(), 0);
        std::vector<std::uint8_t> lane_completed(lanes.size(), 0);
        const auto commit_output =
            [&](std::size_t lane_index, const PcoVertexExecution &execution) {
              if (lane_index >= lanes.size() ||
                  lane_completed[lane_index] != 0 ||
                  execution.suspended != 0 ||
                  execution.texture_request_valid != 0 ||
                  execution.continuation.valid != 0 ||
                  execution.written_mask !=
                      state.vertex_program_summary.vertex_output_mask ||
                  execution.emitted != 1 ||
                  execution.ended_task !=
                      state.vertex_program_summary.ends_task ||
                  lane_request_count[lane_index] !=
                      sample_instruction_count) {
                throw std::runtime_error(
                    "texture vertex USC lane did not complete exact VTXOUT");
              }
              VertexLane &lane = lanes[lane_index];
              std::copy(execution.outputs.begin(), execution.outputs.end(),
                        std::begin(lane.vertex_output));
              lane.emitted = execution.emitted;
              lane.ended = execution.ended_task;
              lane_completed[lane_index] = 1;
            };

        const auto queue_suspension =
            [&](std::size_t lane_index,
                const PcoVertexExecution &execution,
                std::vector<TextureSampleRequest> &requests,
                std::vector<PcoVertexContinuation> &continuations,
                std::vector<std::uint8_t> &queued) {
              if (lane_index >= lanes.size() ||
                  requests.size() != lanes.size() ||
                  continuations.size() != lanes.size() ||
                  queued.size() != lanes.size() || queued[lane_index] != 0 ||
                  lane_completed[lane_index] != 0) {
                throw std::runtime_error(
                    "texture vertex USC suspension lane is invalid");
              }
              if (execution.suspended == 0) {
                commit_output(lane_index, execution);
                return;
              }
              const PcoTextureRequest &issued = execution.texture_request;
              if (execution.suspended != 1 ||
                  execution.texture_request_valid != 1 ||
                  execution.continuation.valid != 1 ||
                  execution.emitted != 0 || execution.ended_task != 0 ||
                  lane_request_count[lane_index] >=
                      sample_instruction_count ||
                  lane_request_count[lane_index] >=
                      kPcoMaximumTextureSampleInstructions ||
                  issued.descriptor_set >= descriptor_set_count ||
                  issued.binding != 0 ||
                  issued.data_request != execution.continuation.data_request) {
                throw std::runtime_error(
                    "texture vertex USC received an invalid SMP suspension");
              }
              const std::size_t descriptor_base =
                  static_cast<std::size_t>(issued.descriptor_set) *
                  kPcoTextureDescriptorDwordCount;
              for (std::size_t dword = 0; dword < 4; ++dword) {
                if (issued.texture_state[dword] !=
                        vertex_context.shared_registers[descriptor_base +
                                                        dword] ||
                    issued.sampler_state[dword] !=
                        vertex_context.shared_registers[descriptor_base + 8U +
                                                        dword]) {
                  throw std::runtime_error(
                      "texture vertex USC SMP descriptor state mismatch");
                }
              }

              TextureSampleRequest request;
              request.shader_lane_index = static_cast<std::uint32_t>(lane_index);
              request.request_id = lane_index;
              request.shader_stage = ShaderStage::kVertex;
              for (std::size_t component = 0; component < 2; ++component)
                request.coordinates[component] = issued.coordinates[component];
              for (std::size_t dword = 0; dword < 4; ++dword) {
                request.texture_state[dword] = issued.texture_state[dword];
                request.sampler_state[dword] = issued.sampler_state[dword];
              }
              request.coordinate_count = issued.coordinate_count;
              request.component_count = issued.component_count;
              request.descriptor_set = issued.descriptor_set;
              request.binding = issued.binding;
              request.dimension = issued.dimension;
              request.normalized = issued.normalized;
              request.data_request = issued.data_request;
              requests[lane_index] = request;
              continuations[lane_index] = execution.continuation;
              queued[lane_index] = 1;
              ++lane_request_count[lane_index];
            };

        std::vector<TextureSampleRequest> pending_requests(lanes.size());
        std::vector<PcoVertexContinuation> pending_continuations(lanes.size());
        std::vector<std::uint8_t> pending_queued(lanes.size(), 0);
        for (std::size_t lane_index = 0; lane_index < lanes.size();
             ++lane_index) {
          const VertexLane &lane = lanes[lane_index];
          const std::vector<std::uint32_t> inputs(
              std::begin(lane.vertex_input), std::end(lane.vertex_input));
          const PcoVertexExecution execution = ExecuteVertexPco(
              state.vertex_program_summary, instructions, inputs,
              vertex_context);
          queue_suspension(lane_index, execution, pending_requests,
                           pending_continuations, pending_queued);
        }
        if (std::any_of(pending_queued.begin(), pending_queued.end(),
                        [](std::uint8_t value) { return value != 1; })) {
          throw std::runtime_error(
              "texture vertex USC did not issue one SMP per shader lane");
        }

        while (!pending_requests.empty()) {
          const std::uint8_t descriptor_set =
              pending_requests.front().descriptor_set;
          for (std::size_t lane_index = 0; lane_index < lanes.size();
               ++lane_index) {
            const TextureSampleRequest &request =
                pending_requests[lane_index];
            if (pending_queued[lane_index] != 1 ||
                request.shader_stage != ShaderStage::kVertex ||
                request.shader_lane_index != lane_index ||
                request.request_id != lane_index || request.quad_id != 0 ||
                request.quad_lane != 0 ||
                request.descriptor_set != descriptor_set ||
                request.data_request !=
                    pending_continuations[lane_index].data_request ||
                pending_continuations[lane_index].valid != 1) {
              throw std::runtime_error(
                  "texture vertex USC request batch ordering is invalid");
            }
          }

          state.texture_sample_requests =
              StoreNewArray(pool_, pending_requests);
          state.vertex_continuations =
              StoreNewArray(pool_, pending_continuations);
          state.stage = PipelineStage::kVertexTexturePending;
          StorePipelineState(pool_, txn.state, state);
          texture_request_output->write(txn);

          const PipelineTxn response_txn = texture_response_input->read();
          if (response_txn.state.slot != txn.state.slot ||
              response_txn.state.generation != txn.state.generation ||
              response_txn.sequence != txn.sequence ||
              response_txn.frame != txn.frame) {
            throw std::runtime_error(
                "texture vertex USC response identity mismatch");
          }
          state = LoadPipelineState(pool_, txn.state);
          RequireStage(state.stage,
                       PipelineStage::kVertexTextureSamplesReady, name());
          if (!HasPoolHandle(state.texture_sample_requests) ||
              !HasPoolHandle(state.texture_sample_responses) ||
              !HasPoolHandle(state.vertex_continuations) ||
              HasPoolHandle(state.fragment_continuations)) {
            throw std::runtime_error(
                "texture vertex USC received no response/continuation");
          }
          const std::vector<TextureSampleRequest> carried_requests =
              LoadArray<TextureSampleRequest>(
                  pool_, state.texture_sample_requests);
          const std::vector<TextureSampleResponse> responses =
              LoadArray<TextureSampleResponse>(
                  pool_, state.texture_sample_responses);
          const std::vector<PcoVertexContinuation> saved_continuations =
              LoadArray<PcoVertexContinuation>(pool_,
                                               state.vertex_continuations);
          if (carried_requests.size() != lanes.size() ||
              responses.size() != lanes.size() ||
              saved_continuations.size() != lanes.size()) {
            throw std::runtime_error(
                "texture vertex USC response lane count mismatch");
          }

          std::vector<TextureSampleRequest> next_requests(lanes.size());
          std::vector<PcoVertexContinuation> next_continuations(lanes.size());
          std::vector<std::uint8_t> next_queued(lanes.size(), 0);
          for (std::size_t lane_index = 0; lane_index < lanes.size();
               ++lane_index) {
            const TextureSampleRequest &issued =
                carried_requests[lane_index];
            const TextureSampleRequest &expected =
                pending_requests[lane_index];
            const TextureSampleResponse &response = responses[lane_index];
            const PcoVertexContinuation &saved =
                saved_continuations[lane_index];
            if (!SameTextureSampleRequest(issued, expected) ||
                response.shader_stage != ShaderStage::kVertex ||
                response.shader_lane_index != issued.shader_lane_index ||
                response.request_id != issued.request_id || saved.valid != 1 ||
                saved.data_request != issued.data_request ||
                !SameVertexContinuation(
                    saved, pending_continuations[lane_index])) {
              throw std::runtime_error(
                  "texture vertex USC response ordering is invalid");
            }
            std::array<std::uint32_t, kPcoPixelOutputCount> texture_response{};
            std::copy(std::begin(response.rgba), std::end(response.rgba),
                      texture_response.begin());
            const PcoVertexExecution execution = ResumeVertexPco(
                state.vertex_program_summary, instructions, saved,
                texture_response);
            queue_suspension(lane_index, execution, next_requests,
                             next_continuations, next_queued);
          }

          const PoolHandle request_payload = state.texture_sample_requests;
          const PoolHandle response_payload = state.texture_sample_responses;
          const PoolHandle continuation_payload = state.vertex_continuations;
          state.texture_sample_requests = {};
          state.texture_sample_responses = {};
          state.vertex_continuations = {};
          pool_.Release(request_payload);
          pool_.Release(response_payload);
          pool_.Release(continuation_payload);

          const bool another_round = std::any_of(
              next_queued.begin(), next_queued.end(),
              [](std::uint8_t value) { return value != 0; });
          if (another_round &&
              std::any_of(next_queued.begin(), next_queued.end(),
                          [](std::uint8_t value) { return value != 1; })) {
            throw std::runtime_error(
                "texture vertex USC lanes diverged across SMP rounds");
          }
          if (another_round) {
            pending_requests = std::move(next_requests);
            pending_continuations = std::move(next_continuations);
            pending_queued = std::move(next_queued);
          } else {
            pending_requests.clear();
            pending_continuations.clear();
            pending_queued.clear();
          }
        }
        if (std::any_of(lane_completed.begin(), lane_completed.end(),
                        [](std::uint8_t value) { return value != 1; })) {
          throw std::runtime_error(
              "texture vertex USC did not complete every shader lane");
        }
      } else {
        for (VertexLane &lane : lanes) {
          const std::vector<std::uint32_t> inputs(
              std::begin(lane.vertex_input), std::end(lane.vertex_input));
          const PcoVertexExecution execution =
              vertex_requires_context
                  ? ExecuteVertexPco(state.vertex_program_summary,
                                     instructions, inputs, vertex_context)
                  : ExecuteVertexPco(state.vertex_program_summary,
                                     instructions, inputs);
          if (execution.suspended != 0 ||
              execution.texture_request_valid != 0 ||
              execution.continuation.valid != 0) {
            throw std::runtime_error(
                "non-texture vertex USC unexpectedly suspended");
          }
          for (std::size_t index = 0; index < kPcoVertexOutputCount; ++index)
            lane.vertex_output[index] = execution.outputs[index];
          lane.emitted = execution.emitted;
          lane.ended = execution.ended_task;
        }
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
      const bool debug_fragment =
          std::getenv("PVRGPU_SEQUENCE_DEBUG_FRAGMENT") != nullptr;
      const std::uint32_t debug_x =
          debug_fragment
              ? DebugFragmentCoordinate("PVRGPU_SEQUENCE_DEBUG_X", 37U)
              : 37U;
      const std::uint32_t debug_y =
          debug_fragment
              ? DebugFragmentCoordinate("PVRGPU_SEQUENCE_DEBUG_Y", 46U)
              : 46U;
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
        if (debug_fragment && invocation.x == debug_x &&
            invocation.y == debug_y) {
          std::cerr << "sequence-fragment-usc phase=execution invocation="
                    << invocation_index
                    << " primitive=" << invocation.primitive_id
                    << " parameter=" << invocation.parameter_index
                    << " submit=" << invocation.submit_ordinal
                    << " quad=" << invocation.quad_id;
          if (context) {
            std::cerr << " sample=0x" << std::hex << std::setw(8)
                      << std::setfill('0') << context->sample_x << ",0x"
                      << std::setw(8) << context->sample_y << std::dec
                      << std::setfill(' ') << " coefficients=";
            for (std::size_t coefficient = 0;
                 coefficient < context->coefficient_count; ++coefficient) {
              if (coefficient)
                std::cerr << ',';
              std::cerr << "0x" << std::hex << std::setw(8)
                        << std::setfill('0')
                        << context->coefficients[coefficient] << std::dec
                        << std::setfill(' ');
            }
          }
          std::cerr << " pixout=";
          for (std::size_t component = 0; component < 4; ++component) {
            if (component)
              std::cerr << ',';
            std::cerr << "0x" << std::hex << std::setw(8)
                      << std::setfill('0')
                      << execution.pixel_outputs[component] << std::dec
                      << std::setfill(' ');
          }
          std::cerr << '\n';
        }
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
        fragment_output.written_mask[0] = execution.written_mask;
        fragment_output.render_target_count = 1;
        outputs[invocation_index] = fragment_output;
        output_written[invocation_index] = 1;
      };

      std::uint64_t fragment_execution_lanes = invocations.size();
      if (UsesTextureSampling(state, ShaderStage::kFragment)) {
        const bool driver_pco_texture =
            IsDriverPcoTrianglesCase(state.functional_case);
        const std::uint32_t descriptor_set_count =
            driver_pco_texture ? state.sampled_texture_count : 1U;
        const std::uint32_t expected_shared_dwords =
            driver_pco_texture ? state.fragment_pco_abi.shareds
                               : kPcoFillTexNearestFragmentSharedCount;
        const std::size_t descriptor_shared_dwords =
            static_cast<std::size_t>(descriptor_set_count) *
            kPcoTextureDescriptorDwordCount;
        const bool shared_layout_valid =
            !driver_pco_texture
                ? expected_shared_dwords == descriptor_shared_dwords
                : DriverPcoTextureSharedLayoutSupported(
                      state.fragment_pco_abi, descriptor_set_count);
        const std::uint32_t expected_coefficient_dwords =
            VaryingCoefficientDwordCount(state);
        const std::size_t sample_instruction_count =
            static_cast<std::size_t>(std::count_if(
                instructions.begin(), instructions.end(),
                [](const PcoInstruction &instruction) {
                  return instruction.opcode == PcoOpcode::kTextureSample;
                }));
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
        if (HasPoolHandle(state.texture_sample_requests) ||
            HasPoolHandle(state.texture_sample_responses) ||
            HasPoolHandle(state.fragment_continuations) ||
            HasPoolHandle(state.vertex_continuations)) {
          throw std::runtime_error(
              "texture fragment USC received stale continuation payloads");
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
            descriptor_set_count == 0 ||
            descriptor_set_count > kPcoMaximumTextureDescriptorSets ||
            !shared_layout_valid ||
            expected_shared_dwords > kPcoMaximumFragmentSharedCount ||
            shared_registers.size() != expected_shared_dwords ||
            expected_coefficient_dwords == 0 ||
            expected_coefficient_dwords >
                kPcoMaximumVaryingCoefficientCount ||
            sample_instruction_count == 0 ||
            sample_instruction_count >
                kPcoMaximumTextureSampleInstructions ||
            (driver_pco_texture &&
             state.fragment_pco_abi.coefficients !=
                 expected_coefficient_dwords)) {
          throw std::runtime_error(
              "texture fragment USC task/shared count mismatch");
        }
        fragment_execution_lanes = shader_lanes.size();
        std::vector<PcoFragmentExecutionContext> lane_contexts(
            shader_lanes.size());
        std::vector<std::uint8_t> lane_context_initialized(
            shader_lanes.size(), 0);
        std::vector<std::uint8_t> lane_request_count(shader_lanes.size(), 0);
        std::vector<std::uint8_t> lane_completed(shader_lanes.size(), 0);

        const auto commit_output =
            [&](std::size_t shader_lane_index,
                const PcoFragmentExecution &execution) {
          if (shader_lane_index >= shader_lanes.size() ||
              lane_completed[shader_lane_index] != 0 ||
              execution.suspended != 0 ||
              execution.texture_request_valid != 0 ||
              execution.continuation.valid != 0 ||
              execution.written_mask != 0x0f || execution.discarded ||
              lane_request_count[shader_lane_index] !=
                  sample_instruction_count) {
            throw std::runtime_error(
                "texture fragment USC lane did not complete exact PIXOUT");
          }
          lane_completed[shader_lane_index] = 1;
          const FragmentShaderLane &shader_lane =
              shader_lanes[shader_lane_index];
          if (debug_fragment && shader_lane.x == debug_x &&
              shader_lane.y == debug_y &&
              shader_lane.helper == 0) {
            std::cerr << "sequence-fragment-usc phase=final lane="
                      << shader_lane_index
                      << " primitive=" << shader_lane.primitive_id
                      << " parameter=" << shader_lane.parameter_index
                      << " submit=" << shader_lane.submit_ordinal
                      << " quad=" << shader_lane.quad_id << " pixout=";
            for (std::size_t component = 0; component < 4; ++component) {
              if (component)
                std::cerr << ',';
              std::cerr << "0x" << std::hex << std::setw(8)
                        << std::setfill('0')
                        << execution.pixel_outputs[component] << std::dec
                        << std::setfill(' ');
            }
            std::cerr << '\n';
          }
          if (shader_lane.helper)
            return;
          const std::uint32_t invocation_index =
              shader_lane.visible_invocation_index;
          if (invocation_index >= invocations.size() ||
              output_written[invocation_index] != 0) {
            throw std::runtime_error(
                "texture fragment USC visible-lane mapping is invalid");
          }
          const FragmentInvocation &invocation = invocations[invocation_index];
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
          fragment_output.written_mask[0] = execution.written_mask;
        fragment_output.render_target_count = 1;
          outputs[invocation_index] = fragment_output;
          output_written[invocation_index] = 1;
        };

        const auto queue_suspension =
            [&](std::size_t shader_lane_index,
                const PcoFragmentExecution &execution,
                std::vector<TextureSampleRequest> &requests,
                std::vector<PcoFragmentContinuation> &continuations,
                std::vector<std::uint8_t> &queued) {
          if (shader_lane_index >= shader_lanes.size() ||
              requests.size() != shader_lanes.size() ||
              continuations.size() != shader_lanes.size() ||
              queued.size() != shader_lanes.size() ||
              queued[shader_lane_index] != 0 ||
              lane_completed[shader_lane_index] != 0) {
            throw std::runtime_error(
                "texture fragment USC suspension lane is invalid");
          }
          if (execution.suspended == 0) {
            commit_output(shader_lane_index, execution);
            return;
          }
          if (execution.suspended != 1 ||
              execution.texture_request_valid != 1 ||
              execution.continuation.valid != 1 ||
              execution.written_mask != 0 || execution.discarded ||
              lane_request_count[shader_lane_index] >=
                  sample_instruction_count ||
              lane_request_count[shader_lane_index] >=
                  kPcoMaximumTextureSampleInstructions ||
              execution.texture_request.descriptor_set >=
                  descriptor_set_count ||
              execution.texture_request.binding != 0 ||
              execution.texture_request.data_request !=
                  execution.continuation.data_request) {
            throw std::runtime_error(
                "texture fragment USC received an invalid SMP suspension");
          }

          const FragmentShaderLane &shader_lane =
              shader_lanes[shader_lane_index];
          TextureSampleRequest request;
          request.shader_lane_index =
              static_cast<std::uint32_t>(shader_lane_index);
          request.quad_id = shader_lane.quad_id;
          request.quad_lane = shader_lane.quad_lane;
          // TextureUnit's public batch ABI numbers requests locally in every
          // round.  Lane identity remains stable across all continuations.
          request.request_id = shader_lane_index;
          request.shader_stage = ShaderStage::kFragment;
          for (std::size_t component = 0; component < 2; ++component) {
            request.coordinates[component] =
                execution.texture_request.coordinates[component];
          }
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
          if (debug_fragment && shader_lane.x == debug_x &&
              shader_lane.y == debug_y &&
              shader_lane.helper == 0) {
            std::cerr << "sequence-fragment-usc phase=suspend lane="
                      << shader_lane_index << " round="
                      << static_cast<unsigned>(
                             lane_request_count[shader_lane_index])
                      << " set="
                      << static_cast<unsigned>(request.descriptor_set)
                      << " resume="
                      << execution.continuation.resume_instruction_index
                      << " pending="
                      << execution.continuation.pending_output_index
                      << " temp_mask=0x" << std::hex
                      << execution.continuation.temporary_written_mask
                      << std::dec << " temps=";
            for (std::size_t temporary = 0;
                 temporary < execution.continuation.temporaries.size();
                 ++temporary) {
              if ((execution.continuation.temporary_written_mask &
                   (UINT64_C(1) << temporary)) == 0)
                continue;
              std::cerr << temporary << ":0x" << std::hex << std::setw(8)
                        << std::setfill('0')
                        << execution.continuation.temporaries[temporary]
                        << std::dec << std::setfill(' ') << ';';
            }
            std::cerr << '\n';
          }
          requests[shader_lane_index] = request;
          continuations[shader_lane_index] = execution.continuation;
          queued[shader_lane_index] = 1;
          ++lane_request_count[shader_lane_index];
        };

        std::vector<TextureSampleRequest> pending_requests(
            shader_lanes.size());
        std::vector<PcoFragmentContinuation> pending_continuations(
            shader_lanes.size());
        std::vector<std::uint8_t> pending_queued(shader_lanes.size(), 0);
        for (const UscFragmentTask &task : tasks) {
          if (task.fragment_quad_index >= quads.size() ||
              task.coefficient_dword_count !=
                  expected_coefficient_dwords ||
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
                shader_lane.submit_ordinal != quad.submit_ordinal ||
                lane_context_initialized[shader_lane_index] != 0) {
              throw std::runtime_error(
                  "texture fragment USC lost shader-lane identity");
            }
            // The strict driver profile carries llvmpipe's coefficients,
            // whose origin already includes its half-pixel setup offset.
            const float interpolation_offset =
                driver_pco_texture ||
                        state.functional_case ==
                            FunctionalCase::kDriverTexturedTriangles
                    ? 0.0F
                    : 0.5F;
            context.sample_x = FloatBits(
                static_cast<float>(shader_lane.x) + interpolation_offset);
            context.sample_y = FloatBits(
                static_cast<float>(shader_lane.y) + interpolation_offset);
            lane_contexts[shader_lane_index] = context;
            lane_context_initialized[shader_lane_index] = 1;
            if (debug_fragment && shader_lane.x == debug_x &&
                shader_lane.y == debug_y &&
                shader_lane.helper == 0) {
              std::cerr << "sequence-fragment-usc phase=context lane="
                        << shader_lane_index << " parameter="
                        << shader_lane.parameter_index << " coefficients=";
              for (std::size_t dword = 0;
                   dword < task.coefficient_dword_count; ++dword) {
                if (dword)
                  std::cerr << ',';
                std::cerr << "0x" << std::hex << std::setw(8)
                          << std::setfill('0') << context.coefficients[dword]
                          << std::dec << std::setfill(' ');
              }
              std::cerr << '\n';
            }
            const PcoFragmentExecution execution = ExecuteFragmentPco(
                state.fragment_program_summary, instructions, context);
            queue_suspension(shader_lane_index, execution, pending_requests,
                             pending_continuations, pending_queued);
          }
        }
        if (std::any_of(lane_context_initialized.begin(),
                        lane_context_initialized.end(),
                        [](std::uint8_t value) { return value != 1; }) ||
            std::any_of(pending_queued.begin(), pending_queued.end(),
                        [](std::uint8_t value) { return value != 1; })) {
          throw std::runtime_error(
              "texture fragment USC did not issue one SMP per shader lane");
        }

        while (!pending_requests.empty()) {
          const std::uint8_t descriptor_set =
              pending_requests.front().descriptor_set;
          for (std::size_t lane_index = 0;
               lane_index < pending_requests.size(); ++lane_index) {
            const TextureSampleRequest &request =
                pending_requests[lane_index];
            if (pending_queued[lane_index] != 1 ||
                request.shader_stage != ShaderStage::kFragment ||
                request.shader_lane_index != lane_index ||
                request.request_id != lane_index ||
                request.reserved[0] != 0 || request.reserved[1] != 0 ||
                request.reserved[2] != 0 ||
                request.descriptor_set != descriptor_set ||
                request.data_request !=
                    pending_continuations[lane_index].data_request ||
                pending_continuations[lane_index].valid != 1) {
              throw std::runtime_error(
                  "texture fragment USC request batch ordering is invalid");
            }
          }

          state.texture_sample_requests =
              StoreNewArray(pool_, pending_requests);
          state.fragment_continuations =
              StoreNewArray(pool_, pending_continuations);
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
          RequireStage(state.stage, PipelineStage::kTextureSamplesReady,
                       name());
          if (!HasPoolHandle(state.texture_sample_requests) ||
              !HasPoolHandle(state.texture_sample_responses) ||
              !HasPoolHandle(state.fragment_continuations) ||
              HasPoolHandle(state.vertex_continuations)) {
            throw std::runtime_error(
                "texture fragment USC received no response/continuation");
          }
          const std::vector<TextureSampleRequest> carried_requests =
              LoadArray<TextureSampleRequest>(pool_,
                                              state.texture_sample_requests);
          const std::vector<TextureSampleResponse> responses =
              LoadArray<TextureSampleResponse>(pool_,
                                               state.texture_sample_responses);
          const std::vector<PcoFragmentContinuation> saved_continuations =
              LoadArray<PcoFragmentContinuation>(
                  pool_, state.fragment_continuations);
          if (carried_requests.size() != shader_lanes.size() ||
              responses.size() != shader_lanes.size() ||
              saved_continuations.size() != shader_lanes.size()) {
            throw std::runtime_error(
                "texture fragment USC response lane count mismatch");
          }

          std::vector<TextureSampleRequest> next_requests(
              shader_lanes.size());
          std::vector<PcoFragmentContinuation> next_continuations(
              shader_lanes.size());
          std::vector<std::uint8_t> next_queued(shader_lanes.size(), 0);
          for (std::size_t lane_index = 0;
               lane_index < shader_lanes.size(); ++lane_index) {
            const TextureSampleRequest &issued = carried_requests[lane_index];
            const TextureSampleRequest &expected =
                pending_requests[lane_index];
            const TextureSampleResponse &response = responses[lane_index];
            const PcoFragmentContinuation &saved =
                saved_continuations[lane_index];
            const PcoFragmentContinuation &expected_continuation =
                pending_continuations[lane_index];
            if (!SameTextureSampleRequest(issued, expected) ||
                response.shader_stage != ShaderStage::kFragment ||
                response.shader_lane_index != issued.shader_lane_index ||
                response.request_id != issued.request_id || saved.valid != 1 ||
                saved.data_request != issued.data_request ||
                saved.program_binary_size !=
                    expected_continuation.program_binary_size ||
                saved.program_instruction_count !=
                    expected_continuation.program_instruction_count ||
                saved.resume_instruction_index !=
                    expected_continuation.resume_instruction_index ||
                saved.pending_output_index !=
                    expected_continuation.pending_output_index ||
                saved.pending_component_count !=
                    expected_continuation.pending_component_count ||
                saved.temporary_written_mask !=
                    expected_continuation.temporary_written_mask ||
                saved.temporaries != expected_continuation.temporaries) {
              throw std::runtime_error(
                  "texture fragment USC response ordering is invalid");
            }

            PcoFragmentExecutionContext resume_context =
                lane_contexts[lane_index];
            if (debug_fragment &&
                shader_lanes[lane_index].x == debug_x &&
                shader_lanes[lane_index].y == debug_y &&
                shader_lanes[lane_index].helper == 0) {
              std::cerr << "sequence-fragment-usc phase=resume lane="
                        << lane_index << " round="
                        << static_cast<unsigned>(
                               lane_request_count[lane_index] - 1U)
                        << " set="
                        << static_cast<unsigned>(issued.descriptor_set)
                        << " rgba=";
              for (std::size_t component = 0; component < 4; ++component) {
                if (component)
                  std::cerr << ',';
                std::cerr << "0x" << std::hex << std::setw(8)
                          << std::setfill('0') << response.rgba[component]
                          << std::dec << std::setfill(' ');
              }
              std::cerr << '\n';
            }
            resume_context.continuation = saved;
            for (std::size_t component = 0; component < 4; ++component) {
              resume_context.texture_response[component] =
                  response.rgba[component];
            }
            resume_context.texture_response_valid = 1;
            const PcoFragmentExecution execution = ExecuteFragmentPco(
                state.fragment_program_summary, instructions, resume_context);
            queue_suspension(lane_index, execution, next_requests,
                             next_continuations, next_queued);
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

          const bool another_round = std::any_of(
              next_queued.begin(), next_queued.end(),
              [](std::uint8_t value) { return value != 0; });
          if (another_round &&
              std::any_of(next_queued.begin(), next_queued.end(),
                          [](std::uint8_t value) { return value != 1; })) {
            throw std::runtime_error(
                "texture fragment USC lanes diverged across SMP rounds");
          }
          if (another_round) {
            pending_requests = std::move(next_requests);
            pending_continuations = std::move(next_continuations);
            pending_queued = std::move(next_queued);
          } else {
            pending_requests.clear();
            pending_continuations.clear();
            pending_queued.clear();
          }
        }
        if (std::any_of(lane_completed.begin(), lane_completed.end(),
                        [](std::uint8_t value) { return value != 1; })) {
          throw std::runtime_error(
              "texture fragment USC did not complete every shader lane");
        }
      } else if (IsDriverPcoTrianglesCase(state.functional_case) &&
                 !UsesShaderVaryings(state)) {
        const std::size_t expected_shared_count =
            state.fragment_pco_abi.shareds;
        if (expected_shared_count > kPcoMaximumFragmentSharedCount ||
            (expected_shared_count != 0 &&
             !HasPoolHandle(state.fragment_shared_registers)) ||
            (expected_shared_count == 0 &&
             HasPoolHandle(state.fragment_shared_registers))) {
          throw std::runtime_error(
              "driver PCO fragment USC shared handle/count mismatch");
        }
        const std::vector<std::uint32_t> shared_registers =
            expected_shared_count == 0
                ? std::vector<std::uint32_t>{}
                : LoadArray<std::uint32_t>(
                      pool_, state.fragment_shared_registers);
        if (shared_registers.size() != expected_shared_count) {
          throw std::runtime_error(
              "driver PCO fragment USC shared-register count mismatch");
        }
        for (std::size_t index = 0; index < invocations.size(); ++index) {
          const FragmentInvocation &invocation = invocations[index];
          PcoFragmentExecutionContext context;
          context.sample_x =
              FloatBits(static_cast<float>(invocation.x) + 0.5F);
          context.sample_y =
              FloatBits(static_cast<float>(invocation.y) + 0.5F);
          context.shared_count = static_cast<std::uint8_t>(
              shared_registers.size());
          for (std::size_t shared = 0; shared < shared_registers.size();
               ++shared) {
            context.shared_registers[shared] = shared_registers[shared];
          }
          execute_invocation(index, &context);
        }
      } else if (UsesShaderVaryings(state)) {
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
        const bool driver_pco =
            IsDriverPcoTrianglesCase(state.functional_case);
        const std::size_t expected_shared_count =
            driver_pco ? state.fragment_pco_abi.shareds : 0U;
        if (expected_shared_count > kPcoMaximumFragmentSharedCount ||
            (expected_shared_count != 0 &&
             !HasPoolHandle(state.fragment_shared_registers)) ||
            (expected_shared_count == 0 &&
             HasPoolHandle(state.fragment_shared_registers))) {
          throw std::runtime_error(
              "varying fragment USC shared handle/count mismatch");
        }
        const std::vector<std::uint32_t> shared_registers =
            expected_shared_count == 0
                ? std::vector<std::uint32_t>{}
                : LoadArray<std::uint32_t>(
                      pool_, state.fragment_shared_registers);
        if (shared_registers.size() != expected_shared_count) {
          throw std::runtime_error(
              "varying fragment USC shared-register count mismatch");
        }
        if (tasks.size() != state.fragment_groups ||
            quads.size() != tasks.size()) {
          throw std::runtime_error(
              "varying fragment USC task/group count mismatch");
        }
        const std::uint32_t expected_coefficient_dwords =
            VaryingCoefficientDwordCount(state);
        if (driver_pco && state.fragment_pco_abi.coefficients !=
                              expected_coefficient_dwords) {
          throw std::runtime_error(
              "driver PCO fragment coefficient ABI mismatch");
        }
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
          context.shared_count =
              static_cast<std::uint8_t>(shared_registers.size());
          for (std::size_t dword = 0;
               dword < task.coefficient_dword_count; ++dword) {
            context.coefficients[dword] =
                coefficient_bank[task.first_coefficient_dword + dword];
          }
          for (std::size_t shared = 0; shared < shared_registers.size();
               ++shared) {
            context.shared_registers[shared] = shared_registers[shared];
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
            const float interpolation_offset = driver_pco ? 0.0F : 0.5F;
            context.sample_x = FloatBits(
                static_cast<float>(invocation.x) + interpolation_offset);
            context.sample_y = FloatBits(
                static_cast<float>(invocation.y) + interpolation_offset);
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
