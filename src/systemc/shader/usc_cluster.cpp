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
#include "common/diagnostics.h"
#include "common/centroid.h"
#include "common/pipeline_state.h"
#include "common/msaa.h"
#include "shader/pco_iss.h"
#include "shader/usc_uniform_buffer_memory.h"
#include "shader/usc_shader_image_memory.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace pvrgpu::stub {

bool DriverPcoTextureSharedLayoutSupported(
    const DriverPcoStageAbi &abi, std::uint32_t descriptor_set_count,
    std::uint32_t image_descriptor_count) {
  if (descriptor_set_count == 0 ||
      descriptor_set_count > kPcoMaximumTextureDescriptorSets) {
    return false;
  }
  std::uint64_t descriptor_shared_dwords =
      static_cast<std::uint64_t>(descriptor_set_count) *
      kPcoTextureDescriptorDwordCount;
  if (abi.uniform_buffer_descriptor_count > kMaximumUniformBuffersPerStage ||
      (abi.uniform_buffer_descriptor_count == 0 &&
       abi.uniform_buffer_descriptor_start != 0) ||
      (abi.uniform_buffer_descriptor_count != 0 &&
       abi.uniform_buffer_descriptor_start != descriptor_shared_dwords))
    return false;
  descriptor_shared_dwords +=
      static_cast<std::uint64_t>(abi.uniform_buffer_descriptor_count) *
      kUniformBufferDescriptorDwordCount;
  if (image_descriptor_count > 32) return false;
  descriptor_shared_dwords += 8U * image_descriptor_count;
  if (abi.shareds > kPcoMaximumFragmentSharedCount)
    return false;
  if (abi.push_constant_count == 0) {
    return abi.shareds == descriptor_shared_dwords &&
           ((abi.uniform_buffer_descriptor_count == 0 &&
             abi.push_constant_start == 0) ||
            abi.push_constant_start == descriptor_shared_dwords);
  }
  const std::uint64_t push_end =
      static_cast<std::uint64_t>(abi.push_constant_start) +
      abi.push_constant_count;
  return abi.push_constant_start == descriptor_shared_dwords &&
         push_end == abi.shareds;
}

namespace {

// Same owned snapshot and validation contract as LoadArray, but retain host
// capacity between texture rounds. No reference into MemoryPool survives this
// function, a FIFO wait, or a later pool allocation/release. resize reuses
// existing elements when the batch size is unchanged; memcpy still replaces
// every payload byte before any response/continuation validation reads it.
template <typename T>
void LoadFragmentScratchArray(const MemoryPool &pool, PoolHandle handle,
                              std::vector<T> &values) {
  static_assert(std::is_trivially_copyable_v<T>);
  const auto &source = pool.Read(handle);
  if (source.size() % sizeof(T) != 0)
    throw std::runtime_error("MemoryPool array has an invalid byte size");
  values.resize(source.size() / sizeof(T));
  if (!source.empty())
    std::memcpy(values.data(), source.data(), source.size());
}

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
  const char *value = DiagnosticEnvironment(name);
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

void SetFragmentCentroidContext(PcoFragmentExecutionContext &context,
                                std::uint32_t coverage,
                                std::uint32_t state_mask,
                                const RasterTriangle *primitive = nullptr) {
  const auto position = RasterCentroidPosition(context.raster_sample_count,
                                               coverage, state_mask);
  float x, y;
  std::memcpy(&x, &context.sample_x, sizeof(x));
  std::memcpy(&y, &context.sample_y, sizeof(y));
  std::array<float, 2> physical{x + static_cast<float>(position[0]) / 16.0F,
                               y + static_cast<float>(position[1]) / 16.0F};
  if (primitive && coverage)
    physical = CentroidInsidePrimitive(primitive->x, primitive->y,
        static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y), physical);
  context.centroid_x = FloatBits(physical[0] - 0.5F);
  context.centroid_y = FloatBits(physical[1] - 0.5F);
  context.centroid_position_valid = 1;
}

void SetFragmentSampleContext(PcoFragmentExecutionContext &context,
                              std::uint32_t sample_id,
                              std::uint32_t coverage,
                              bool sample_frequency) {
  const auto position = RasterSamplePosition(context.raster_sample_count, sample_id);
  float x, y;
  std::memcpy(&x, &context.sample_x, sizeof(x));
  std::memcpy(&y, &context.sample_y, sizeof(y));
  context.sample_id = sample_id;
  context.coverage_mask = coverage;
  context.sample_position_x = sample_frequency
      ? FloatBits(x + static_cast<float>(position[0]) / 16.0F - 0.5F) : context.sample_x;
  context.sample_position_y = sample_frequency
      ? FloatBits(y + static_cast<float>(position[1]) / 16.0F - 0.5F) : context.sample_y;
  context.sample_position_valid = 1;
}

void SetFragmentFacingContext(PcoFragmentExecutionContext &context,
                              std::uint8_t front_facing) {
  if (front_facing > 1)
    throw std::runtime_error("fragment USC raster facing is not canonical");
  context.front_facing = front_facing;
  context.front_facing_valid = 1;
}

bool SameTextureSampleRequest(const TextureSampleRequest &left,
                              const TextureSampleRequest &right) {
  return left.shader_lane_index == right.shader_lane_index &&
         left.sample_id == right.sample_id &&
         std::equal(std::begin(left.spatial_offsets), std::end(left.spatial_offsets),
                    std::begin(right.spatial_offsets)) &&
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
         left.texture_address_lo == right.texture_address_lo &&
         left.texture_address_hi == right.texture_address_hi &&
         left.coordinate_count == right.coordinate_count &&
         left.component_count == right.component_count &&
         left.descriptor_set == right.descriptor_set &&
         left.binding == right.binding && left.dimension == right.dimension &&
         left.normalized == right.normalized &&
         left.fcnorm == right.fcnorm &&
         left.sample_index == right.sample_index &&
         left.sample_index_present == right.sample_index_present &&
         left.explicit_lod == right.explicit_lod &&
         left.explicit_lod_present == right.explicit_lod_present &&
         left.lod_bias == right.lod_bias &&
         left.lod_bias_present == right.lod_bias_present &&
         left.gather == right.gather &&
         left.data_request == right.data_request &&
         left.quad_lane == right.quad_lane &&
         left.shader_stage == right.shader_stage && left.gather <= 1 &&
         right.gather <= 1;
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
    std::uint64_t logical_invocations, std::uint64_t execution_lanes,
    const PcoInstructionCounts *fragment_dynamic = nullptr) {
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
  if ((stage == ShaderStage::kFragment) != (fragment_dynamic != nullptr))
    throw std::runtime_error("USC fragment instruction accounting is not native");

  const PcoInstructionCounts per_invocation =
      CountPcoInstructions(instructions, true);
  stats.invocations = logical_invocations;
  stats.executed_alu_instructions =
      fragment_dynamic ? fragment_dynamic->alu :
          CheckedInstructionTotal(per_invocation.alu, execution_lanes);
  stats.executed_tex_instructions =
      fragment_dynamic ? fragment_dynamic->texture :
          CheckedInstructionTotal(per_invocation.texture, execution_lanes);
  stats.executed_memory_instructions =
      fragment_dynamic ? fragment_dynamic->memory :
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
                       ShaderStage stage, GpuMemorySystem *memory)
    : sc_module(name), pool_(pool), stage_(stage), memory_(memory) {
  if (stage != ShaderStage::kVertex && stage != ShaderStage::kFragment)
    throw std::invalid_argument("graphics UscCluster cannot execute compute");
  SC_THREAD(Run);
}

/*
 * Spread a fragment program's pixel outputs over the attachments it wrote.
 * The pixel-output file is render-target major -- pixout0..3 are the first
 * attachment's channels and pixout4..7 the second's -- and FragmentOutput
 * stores them the same way, so this is a copy plus the per-target slice of
 * the written mask.  A shader returning two results, as dEQP's modf does,
 * writes two attachments in one pass.
 */
void StoreFragmentPixelOutputs(FragmentOutput &fragment_output,
                               const std::array<std::uint32_t,
                                                kPcoPixelOutputCount> &outputs,
                               std::uint16_t written_mask,
                               std::uint32_t render_target_count) {
  const std::uint32_t targets =
      render_target_count == 0 ? 1U : render_target_count;
  if (targets > kMaxRenderTargets || targets * 4 > outputs.size())
    throw std::runtime_error("fragment output render-target count exceeds ABI");
  for (std::uint32_t target = 0; target < targets; ++target) {
    for (std::size_t component = 0; component < 4; ++component) {
      fragment_output.pixel_output[target * 4 + component] =
          outputs[target * 4 + component];
    }
    fragment_output.written_mask[target] =
        static_cast<std::uint8_t>((written_mask >> (4U * target)) & 0x0fU);
  }
  fragment_output.render_target_count = static_cast<std::uint8_t>(targets);
}

void UscCluster::Run() {
  while (true) {
    const PipelineTxn txn = input.read();
    PipelineState state = LoadPipelineState(pool_, txn.state);
    if (!IsRasterFunctionalCase(state.functional_case))
      throw std::runtime_error("USC cluster received an unsupported case");

    const PoolHandle uniform_resources = stage_ == ShaderStage::kVertex
        ? state.vertex_uniform_buffer_resources
        : state.fragment_uniform_buffer_resources;
    UscUniformBufferMemory uniform_memory(
        memory_, state.memory_mode,
        HasPoolHandle(uniform_resources)
            ? LoadArray<UniformBufferResource>(pool_, uniform_resources)
            : std::vector<UniformBufferResource>{});
    std::vector<ShaderImageResource> image_resources =
        stage_ == ShaderStage::kFragment && HasPoolHandle(state.fragment_image_resources)
            ? LoadArray<ShaderImageResource>(pool_, state.fragment_image_resources)
            : std::vector<ShaderImageResource>{};
    UscShaderImageMemory image_memory(memory_, state.memory_mode, image_resources);

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
      vertex_context.memory_read = UscUniformBufferMemory::Read;
      vertex_context.memory_user_data = &uniform_memory;
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
        vertex_context.shared_count = static_cast<std::uint16_t>(shared.size());
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
            sample_instruction_count > kPcoMaximumVertexTextureSampleInstructions ||
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
                      kPcoMaximumVertexTextureSampleInstructions ||
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
              // Every coordinate the request holds: a 2D sample leaves the
              // third zero; cube/3D use it, arrays use the TAO address.
              for (std::size_t component = 0;
                   component < std::size(request.coordinates); ++component)
                request.coordinates[component] = issued.coordinates[component];
              for (std::size_t dword = 0; dword < 4; ++dword) {
                request.texture_state[dword] = issued.texture_state[dword];
                request.sampler_state[dword] = issued.sampler_state[dword];
              }
              std::copy(issued.spatial_offsets.begin(), issued.spatial_offsets.end(),
                        std::begin(request.spatial_offsets));
              request.coordinate_count = issued.coordinate_count;
              request.component_count = issued.component_count;
              request.descriptor_set = issued.descriptor_set;
              request.binding = issued.binding;
              request.dimension = issued.dimension;
              request.normalized = issued.normalized;
              request.fcnorm = issued.fcnorm;
              request.sample_index = issued.sample_index;
              request.sample_index_present = issued.sample_index_present;
              request.explicit_lod = issued.explicit_lod;
              request.explicit_lod_present = issued.explicit_lod_present;
              if (issued.lod_bias_present || issued.lod_bias)
                throw std::runtime_error("vertex SMP shader LOD bias is unsupported");
              if (issued.gather)
                throw std::runtime_error("vertex SMP raw gather is unsupported");
              request.data_request = issued.data_request;
              request.texture_address_lo = issued.texture_address_lo;
              request.texture_address_hi = issued.texture_address_hi;
              requests[lane_index] = request;
              continuations[lane_index] = execution.continuation;
              queued[lane_index] = 1;
              ++lane_request_count[lane_index];
            };

        const auto pending_lanes = sample_instruction_count == 0 ? 0 : lanes.size();
        std::vector<TextureSampleRequest> pending_requests(pending_lanes);
        std::vector<PcoVertexContinuation> pending_continuations(pending_lanes);
        std::vector<std::uint8_t> pending_queued(pending_lanes, 0);
        for (std::size_t lane_index = 0; lane_index < lanes.size();
             ++lane_index) {
          const VertexLane &lane = lanes[lane_index];
          const std::vector<std::uint32_t> inputs(
              std::begin(lane.vertex_input), std::end(lane.vertex_input));
          const PcoVertexExecution execution = ExecuteVertexPco(
              state.vertex_program_summary, instructions, inputs,
              vertex_context);
          if (sample_instruction_count == 0)
            commit_output(lane_index, execution);
          else
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
            std::array<std::uint32_t, kPcoTextureResponseCount> texture_response{};
            std::copy(std::begin(response.rgba), std::end(response.rgba),
                      texture_response.begin());
            const PcoVertexExecution execution = ResumeVertexPco(
                state.vertex_program_summary, instructions, saved,
                texture_response, UscUniformBufferMemory::Read, &uniform_memory);
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
      const PcoProgramSummary fragment_summary = state.fragment_program_summary;
      const PoolHandle fragment_program_handle = state.fragment_instructions;
      // One immutable, fully checked program owns this draw's execution. Only
      // static validation/signature work is shared; each lane still interprets
      // its ISA and validates its own continuation and memory responses.
      const PcoPreparedFragmentProgram fragment_program(fragment_summary,
                                                       instructions);
      const auto fragment_program_identity_valid = [&] {
        const auto &summary = state.fragment_program_summary;
        return state.fragment_instructions.slot == fragment_program_handle.slot &&
            state.fragment_instructions.generation == fragment_program_handle.generation &&
            summary.stage == fragment_summary.stage &&
            summary.binary_size == fragment_summary.binary_size &&
            summary.group_count == fragment_summary.group_count &&
            summary.instruction_count == fragment_summary.instruction_count &&
            summary.vertex_input_mask == fragment_summary.vertex_input_mask &&
            summary.vertex_output_mask == fragment_summary.vertex_output_mask &&
            summary.pixel_output_mask == fragment_summary.pixel_output_mask &&
            summary.early_hsr_safe == fragment_summary.early_hsr_safe &&
            summary.writes_depth == fragment_summary.writes_depth &&
            summary.uses_derivatives == fragment_summary.uses_derivatives &&
            summary.ends_task == fragment_summary.ends_task;
      };
      const bool geometry_centroid = IsDriverPcoTrianglesCase(state.functional_case) &&
          std::any_of(instructions.begin(), instructions.end(), [](const auto &instruction) {
            return instruction.iteration_mode == PcoIterationMode::kCentroid;
          });
      const std::vector<RasterTriangle> centroid_primitives = geometry_centroid
          ? LoadArray<RasterTriangle>(pool_, state.raster_triangles)
          : std::vector<RasterTriangle>{};
      const auto centroid_primitive = [&](std::uint32_t index) -> const RasterTriangle * {
        if (!geometry_centroid)
          return nullptr;
        if (index >= centroid_primitives.size())
          throw std::runtime_error("fragment centroid primitive identity is out of range");
        return &centroid_primitives[index];
      };
      if (invocations.size() != state.active_fragment_invocations)
        throw std::runtime_error("fragment USC invocation count mismatch");

      std::vector<FragmentOutput> outputs(invocations.size());
      std::vector<std::uint8_t> output_written(invocations.size(), 0);
      PcoInstructionCounts fragment_dynamic{};
      const auto record_fragment_execution = [&](const PcoFragmentExecution &execution) {
        if (execution.suspended || execution.continuation.valid ||
            execution.texture_request_valid || execution.derivative_request_valid)
          throw std::runtime_error("USC cannot commit partial fragment counters");
        AddInstructionCounter(fragment_dynamic.alu, execution.executed_instructions.alu);
        AddInstructionCounter(fragment_dynamic.texture, execution.executed_instructions.texture);
        AddInstructionCounter(fragment_dynamic.memory, execution.executed_instructions.memory);
      };
      const bool debug_fragment =
          DiagnosticEnvironment("PVRGPU_SEQUENCE_DEBUG_FRAGMENT") != nullptr;
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
        PcoFragmentExecutionContext raster_context;
        if (context)
          raster_context = *context;
        SetFragmentFacingContext(raster_context, invocation.front_facing);
        context = &raster_context;
        if (IsDriverPcoTrianglesCase(state.functional_case)) {
          raster_context.raster_sample_count = state.raster_state.sample_count;
          raster_context.memory_atomic32 = UscShaderImageMemory::Atomic32;
          raster_context.image_memory_user_data = &image_memory;
          raster_context.memory_side_effects_enabled = 1;
          raster_context.sample_x = FloatBits(static_cast<float>(invocation.x));
          raster_context.sample_y = FloatBits(static_cast<float>(invocation.y));
          // Driver commands require half_pixel_center=1. Keep the integer
          // llvmpipe interpolation origin and the physical SR origin apart.
          raster_context.special_coordinate_offset = FloatBits(0.5F);
          SetFragmentCentroidContext(raster_context, invocation.sample_mask,
                                      state.raster_state.sample_mask,
                                      centroid_primitive(invocation.parameter_index));
          SetFragmentSampleContext(raster_context, invocation.sample_id,
                                    invocation.sample_mask,
                                    state.raster_state.sample_frequency != 0);
        }
        const PcoFragmentExecution execution =
            context ? ExecuteFragmentPco(fragment_program, *context)
                    : ExecuteFragmentPco(fragment_program);
        if (debug_fragment && invocation.x == debug_x &&
            invocation.y == debug_y) {
          std::cerr << "sequence-fragment-usc phase=execution invocation="
                    << invocation_index
                    << " primitive=" << invocation.primitive_id
                    << " parameter=" << invocation.parameter_index
                    << " submit=" << invocation.submit_ordinal
                    << " quad=" << invocation.quad_id
                    << " sample_id=" << static_cast<unsigned>(invocation.sample_id)
                    << " coverage=0x" << std::hex << invocation.sample_mask
                    << std::dec;
          if (context) {
            std::cerr << " sample=0x" << std::hex << std::setw(8)
                      << std::setfill('0') << context->sample_x << ",0x"
                      << std::setw(8) << context->sample_y
                      << " centroid=0x" << std::setw(8) << context->centroid_x
                      << ",0x" << std::setw(8) << context->centroid_y << std::dec
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
            if (const RasterTriangle *primitive =
                    centroid_primitive(invocation.parameter_index)) {
              std::cerr << " triangle=";
              for (std::size_t vertex = 0; vertex < 3; ++vertex) {
                if (vertex)
                  std::cerr << ';';
                std::cerr << primitive->x[vertex] << ',' << primitive->y[vertex];
              }
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
        fragment_output.depth_written = execution.depth_written;
        fragment_output.discarded = execution.discarded;
        if (execution.depth_written)
          std::memcpy(&fragment_output.depth, &execution.depth, sizeof(float));
        StoreFragmentPixelOutputs(fragment_output, execution.pixel_outputs,
                                  execution.written_mask,
                                  state.render_target_count);
        outputs[invocation_index] = fragment_output;
        output_written[invocation_index] = 1;
        record_fragment_execution(execution);
      };

      std::uint64_t fragment_execution_lanes = invocations.size();
      if (UsesFragmentQuadLanes(state)) {
        const bool has_fragment_texture =
            UsesTextureSampling(state, ShaderStage::kFragment);
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
                : (!has_fragment_texture ||
                   DriverPcoTextureSharedLayoutSupported(
                       state.fragment_pco_abi, descriptor_set_count,
                       state.fragment_image_descriptor_count));
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
            (expected_shared_dwords != 0 &&
             !HasPoolHandle(state.fragment_shared_registers))) {
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
        const std::vector<FragmentQuad> all_quads =
            LoadArray<FragmentQuad>(pool_, state.fragment_quads);
        const std::vector<FragmentShaderLane> all_shader_lanes =
            LoadArray<FragmentShaderLane>(pool_, state.fragment_shader_lanes);
        const std::vector<UscFragmentTask> all_tasks =
            LoadArray<UscFragmentTask>(pool_, state.usc_fragment_tasks);
        const std::vector<std::uint32_t> coefficient_bank =
            LoadArray<std::uint32_t>(pool_, state.usc_coefficient_banks);
        const std::vector<std::uint32_t> shared_registers =
            expected_shared_dwords == 0 ? std::vector<std::uint32_t>{}
                : LoadArray<std::uint32_t>(pool_,
                                          state.fragment_shared_registers);
        if (all_shader_lanes.size() != state.fragment_shader_lane_count ||
            all_tasks.size() != state.fragment_groups ||
            all_quads.size() != all_tasks.size() ||
            (has_fragment_texture && descriptor_set_count == 0) ||
            descriptor_set_count > kPcoMaximumTextureDescriptorSets ||
            !shared_layout_valid ||
            expected_shared_dwords > kPcoMaximumFragmentSharedCount ||
            shared_registers.size() != expected_shared_dwords ||
            (UsesShaderVaryings(state) && expected_coefficient_dwords == 0) ||
            expected_coefficient_dwords >
                kPcoMaximumVaryingCoefficientCount ||
            (!driver_pco_texture && sample_instruction_count == 0) ||
            (driver_pco_texture &&
             state.fragment_pco_abi.coefficients !=
                 expected_coefficient_dwords)) {
          throw std::runtime_error(
              "texture fragment USC task/shared count mismatch");
        }
        fragment_execution_lanes = all_shader_lanes.size();
        // Fragment SMP instructions may occur in arbitrary validated control
        // flow. A lane owns only its current suspension, not a stack sized by
        // the static instruction count. Bound actual dynamic requests below;
        // do not impose the vertex path's straight-line sample-count gate.
        // Bound host-side live register/continuation storage independently of
        // frame size and overdraw. Every complete quad still runs to
        // completion; this is not a shader-work limit or a claim about physical
        // Rogue occupancy. Output indices and external FIFO lane IDs stay
        // global.
        constexpr std::size_t kResidentQuadLimit = 256;
        std::vector<std::uint8_t> visited_quads(all_quads.size(), 0);
        std::vector<std::uint8_t> visited_lanes(all_shader_lanes.size(), 0);
        for (std::size_t task_begin = 0; task_begin < all_tasks.size();
             task_begin += kResidentQuadLimit) {
          const std::size_t task_end =
              std::min(task_begin + kResidentQuadLimit, all_tasks.size());
          std::vector<std::uint32_t> global_lane_indices;
          std::vector<FragmentQuad> quads;
          std::vector<UscFragmentTask> tasks;
          global_lane_indices.reserve((task_end - task_begin) * 4);
          quads.reserve(task_end - task_begin);
          tasks.reserve(task_end - task_begin);
          for (std::size_t index = task_begin; index < task_end; ++index) {
            UscFragmentTask task = all_tasks[index];
            if (task.fragment_quad_index >= all_quads.size() ||
                visited_quads[task.fragment_quad_index]++)
              throw std::runtime_error(
                  "fragment residency repeats or loses a quad");
            quads.push_back(all_quads[task.fragment_quad_index]);
            task.fragment_quad_index =
                static_cast<std::uint32_t>(quads.size() - 1);
            tasks.push_back(task);
            const auto &quad = quads.back();
            const auto active = quad.coverage_mask | quad.helper_mask;
            if (active == 0 || (active & ~0x0fU) != 0 ||
                (quad.coverage_mask & quad.helper_mask) != 0 ||
                quad.write_mask != quad.coverage_mask)
              throw std::runtime_error(
                  "fragment residency received an invalid quad mask");
            for (std::size_t lane = 0; lane < 4; ++lane) {
              if (!(active & (1U << lane)))
                continue;
              const auto global = quad.invocation_indices[lane];
              if (global >= all_shader_lanes.size() || visited_lanes[global]++)
                throw std::runtime_error(
                    "fragment residency repeats or loses a shader lane");
              global_lane_indices.push_back(global);
            }
          }
          // TextureUnit's compact batch contract retains ascending global lane
          // identity, even if PDS task order differs from the lane allocation.
          std::sort(global_lane_indices.begin(), global_lane_indices.end());
          std::vector<FragmentShaderLane> shader_lanes;
          shader_lanes.reserve(global_lane_indices.size());
          for (const auto global : global_lane_indices)
            shader_lanes.push_back(all_shader_lanes[global]);
          for (auto &quad : quads) {
            const auto active = quad.coverage_mask | quad.helper_mask;
            for (std::size_t lane = 0; lane < 4; ++lane) {
              if (!(active & (1U << lane)))
                continue;
              const auto global = quad.invocation_indices[lane];
              quad.invocation_indices[lane] = static_cast<std::uint32_t>(
                  std::lower_bound(global_lane_indices.begin(),
                                   global_lane_indices.end(), global) -
                  global_lane_indices.begin());
            }
          }
          std::vector<PcoFragmentExecutionContext> lane_contexts(
              shader_lanes.size());
          std::vector<std::uint8_t> lane_context_initialized(
              shader_lanes.size(), 0);
          std::vector<std::uint32_t> lane_request_count(shader_lanes.size(), 0);
          std::vector<std::uint8_t> lane_completed(shader_lanes.size(), 0);
          std::vector<PcoFragmentExecution> lane_executions(
              shader_lanes.size());

          // Native derivative instructions are quad rendezvous points. Each
          // lane resumes its saved ISA state; interpolation, ALU and preceding
          // texture instructions are never replayed to obtain a neighbour.
          const auto resolve_quad_derivatives = [&](const FragmentQuad &quad) {
            const std::uint8_t active = quad.coverage_mask | quad.helper_mask;
            std::uint32_t rounds = 0;
            for (;;) {
              bool pending = false;
              for (std::uint8_t lane = 0; lane < 4; ++lane) {
                if ((active & (1U << lane)) != 0) {
                  const auto index = quad.invocation_indices[lane];
                  if (index >= lane_executions.size())
                    throw std::runtime_error(
                        "derivative quad lane out of range");
                  pending |=
                      lane_executions[index].derivative_request_valid != 0;
                }
              }
              if (!pending)
                return;
              if (active != 0x0f || ++rounds > 65536)
                throw std::runtime_error("derivative quad is incomplete or "
                                         "exceeds instruction budget");
              std::array<std::uint32_t, 4> sources{};
              const auto &first = lane_executions[quad.invocation_indices[0]];
              const auto resume_pc =
                  first.continuation.resume_instruction_index;
              if (resume_pc == 0 || resume_pc > instructions.size())
                throw std::runtime_error(
                    "derivative quad continuation PC is invalid");
              for (std::uint8_t lane = 0; lane < 4; ++lane) {
                const auto &execution =
                    lane_executions[quad.invocation_indices[lane]];
                if (execution.suspended != 1 ||
                    execution.derivative_request_valid != 1 ||
                    execution.texture_request_valid != 0 ||
                    execution.continuation.valid != 1 ||
                    execution.continuation.kind != 1 ||
                    execution.continuation.resume_instruction_index !=
                        resume_pc)
                  throw std::runtime_error("fragment lanes diverged at native "
                                           "derivative rendezvous");
                sources[lane] = execution.derivative_source;
              }
              const auto values = EvaluatePcoDerivativeQuad(
                  instructions[resume_pc - 1], sources);
              for (std::uint8_t lane = 0; lane < 4; ++lane) {
                const auto index = quad.invocation_indices[lane];
                auto context = lane_contexts[index];
                context.continuation = lane_executions[index].continuation;
                context.derivative_response = values[lane];
                context.derivative_response_valid = 1;
                lane_executions[index] = ExecuteFragmentPco(
                    fragment_program, context);
              }
            }
          };

          const auto commit_output = [&](std::size_t shader_lane_index,
                                         const PcoFragmentExecution
                                             &execution) {
            if (shader_lane_index >= shader_lanes.size() ||
                lane_completed[shader_lane_index] != 0 ||
                execution.suspended != 0 ||
                execution.texture_request_valid != 0 ||
                execution.derivative_request_valid != 0 ||
                execution.continuation.valid != 0 ||
                (!execution.discarded &&
                 execution.written_mask !=
                     state.fragment_program_summary.pixel_output_mask)) {
              throw std::runtime_error(
                  "texture fragment USC lane did not complete exact PIXOUT: "
                  "lane=" +
                  std::to_string(shader_lane_index) + " written=" +
                  std::to_string(execution.written_mask) + " expected=" +
                  std::to_string(
                      state.fragment_program_summary.pixel_output_mask) +
                  " discarded=" + std::to_string(execution.discarded) +
                  " suspended=" + std::to_string(execution.suspended) +
                  " continuation=" +
                  std::to_string(execution.continuation.valid) + " samples=" +
                  (shader_lane_index < lane_request_count.size()
                       ? std::to_string(lane_request_count[shader_lane_index])
                       : "out-of-range") +
                  " static_samples=" +
                  std::to_string(sample_instruction_count));
            }
            lane_completed[shader_lane_index] = 1;
            // Include real helper ALU/texture work exactly once, before
            // skipping its pixel output. Each final result already includes all
            // resumes.
            record_fragment_execution(execution);
            const FragmentShaderLane &shader_lane =
                shader_lanes[shader_lane_index];
            if (debug_fragment && shader_lane.x == debug_x &&
                shader_lane.y == debug_y && shader_lane.helper == 0) {
              std::cerr << "sequence-fragment-usc phase=final lane="
                        << global_lane_indices[shader_lane_index]
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
            const FragmentInvocation &invocation =
                invocations[invocation_index];
            if (invocation.front_facing != shader_lane.front_facing)
              throw std::runtime_error("texture fragment USC visible facing identity mismatch");
            FragmentOutput fragment_output;
            fragment_output.x = invocation.x;
            fragment_output.y = invocation.y;
            fragment_output.primitive_id = invocation.primitive_id;
            fragment_output.parameter_index = invocation.parameter_index;
            fragment_output.submit_ordinal = invocation.submit_ordinal;
            fragment_output.depth = invocation.depth;
            fragment_output.depth_written = execution.depth_written;
            fragment_output.discarded = execution.discarded;
            if (execution.depth_written)
              std::memcpy(&fragment_output.depth, &execution.depth,
                          sizeof(float));
            StoreFragmentPixelOutputs(fragment_output, execution.pixel_outputs,
                                      execution.written_mask,
                                      state.render_target_count);
            outputs[invocation_index] = fragment_output;
            output_written[invocation_index] = 1;
          };

          const auto queue_suspension = [&](std::size_t shader_lane_index,
                                            const PcoFragmentExecution
                                                &execution,
                                            std::vector<TextureSampleRequest>
                                                &requests,
                                            std::vector<PcoFragmentContinuation>
                                                &continuations,
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
            if (lane_request_count[shader_lane_index] >= 65536U) {
              throw std::runtime_error(
                  "texture fragment USC dynamic SMP request limit exceeded");
            }
            if (execution.suspended != 1 ||
                execution.texture_request_valid != 1 ||
                execution.derivative_request_valid != 0 ||
                execution.continuation.valid != 1 ||
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
            request.shader_lane_index = global_lane_indices[shader_lane_index];
            request.quad_id = shader_lane.quad_id;
            request.quad_lane = shader_lane.quad_lane;
            request.sample_id = shader_lane.sample_id;
            // TextureUnit's public batch ABI numbers requests locally in every
            // round.  Lane identity remains stable across all continuations.
            request.request_id = shader_lane_index;
            request.shader_stage = ShaderStage::kFragment;
            for (std::size_t component = 0;
                 component < std::size(request.coordinates); ++component) {
              request.coordinates[component] =
                  execution.texture_request.coordinates[component];
            }
            for (std::size_t dword = 0; dword < 4; ++dword) {
              request.texture_state[dword] =
                  execution.texture_request.texture_state[dword];
              request.sampler_state[dword] =
                  execution.texture_request.sampler_state[dword];
            }
            std::copy(execution.texture_request.spatial_offsets.begin(),
                      execution.texture_request.spatial_offsets.end(),
                      std::begin(request.spatial_offsets));
            request.coordinate_count =
                execution.texture_request.coordinate_count;
            request.component_count = execution.texture_request.component_count;
            request.descriptor_set = execution.texture_request.descriptor_set;
            request.binding = execution.texture_request.binding;
            request.dimension = execution.texture_request.dimension;
            request.normalized = execution.texture_request.normalized;
            request.fcnorm = execution.texture_request.fcnorm;
            request.sample_index = execution.texture_request.sample_index;
            request.sample_index_present =
                execution.texture_request.sample_index_present;
            request.explicit_lod = execution.texture_request.explicit_lod;
            request.explicit_lod_present =
                execution.texture_request.explicit_lod_present;
            request.lod_bias = execution.texture_request.lod_bias;
            request.lod_bias_present = execution.texture_request.lod_bias_present;
            request.gather = execution.texture_request.gather;
            request.data_request = execution.texture_request.data_request;
            request.texture_address_lo =
                execution.texture_request.texture_address_lo;
            request.texture_address_hi =
                execution.texture_request.texture_address_hi;
            if (debug_fragment && shader_lane.x == debug_x &&
                shader_lane.y == debug_y && shader_lane.helper == 0) {
              std::cerr << "sequence-fragment-usc phase=suspend lane="
                        << global_lane_indices[shader_lane_index] << " round="
                        << static_cast<unsigned>(
                               lane_request_count[shader_lane_index])
                        << " set="
                        << static_cast<unsigned>(request.descriptor_set)
                        << " resume="
                        << execution.continuation.resume_instruction_index
                        << " pending="
                        << execution.continuation.pending_output_index
                        << " temp_mask=0x" << std::hex << std::setfill('0');
              for (auto word = execution.continuation.temporary_written_mask
                                   .words.rbegin();
                   word !=
                   execution.continuation.temporary_written_mask.words.rend();
                   ++word)
                std::cerr << std::setw(16) << *word;
              std::cerr << std::dec << std::setfill(' ') << " temps=";
              for (std::size_t temporary = 0;
                   temporary < execution.continuation.temporaries.size();
                   ++temporary) {
                if (!execution.continuation.temporary_written_mask.test(
                        temporary))
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

          // Descriptor-only queries are ordinary native ALU. The bound image
          // still supplies SHARED words and position inputs, but only an actual
          // decoded SMP may allocate/send a texture continuation round.
          const std::size_t pending_lane_count =
              sample_instruction_count == 0 ? 0 : shader_lanes.size();
          std::vector<TextureSampleRequest> pending_requests(
              pending_lane_count);
          std::vector<PcoFragmentContinuation> pending_continuations(
              pending_lane_count);
          std::vector<std::uint8_t> pending_queued(pending_lane_count, 0);
          for (const UscFragmentTask &task : tasks) {
            if (task.fragment_quad_index >= quads.size() ||
                task.coefficient_dword_count != expected_coefficient_dwords ||
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
            context.raster_sample_count = state.raster_state.sample_count;
            context.memory_read = UscUniformBufferMemory::Read;
            context.memory_user_data = &uniform_memory;
            if (driver_pco_texture ||
                state.functional_case ==
                    FunctionalCase::kDriverTexturedTriangles)
              context.special_coordinate_offset = FloatBits(0.5F);
            context.coefficient_count =
                static_cast<std::uint8_t>(task.coefficient_dword_count);
            context.shared_count =
                static_cast<std::uint16_t>(shared_registers.size());
            for (std::size_t dword = 0; dword < task.coefficient_dword_count;
                 ++dword) {
              context.coefficients[dword] =
                  coefficient_bank[task.first_coefficient_dword + dword];
            }
            for (std::size_t dword = 0; dword < shared_registers.size();
                 ++dword)
              context.shared_registers[dword] = shared_registers[dword];
            bool quad_facing_set = false;
            std::uint8_t quad_front_facing = 0;
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
                  shader_lane.sample_id != quad.sample_id ||
                  shader_lane.quad_lane != lane ||
                  shader_lane.parameter_index != quad.parameter_index ||
                  shader_lane.submit_ordinal != quad.submit_ordinal ||
                  shader_lane.front_facing > 1 ||
                  shader_lane.helper !=
                      static_cast<std::uint8_t>((quad.helper_mask >> lane) & 1U) ||
                  (quad_facing_set && shader_lane.front_facing != quad_front_facing) ||
                  lane_context_initialized[shader_lane_index] != 0) {
                throw std::runtime_error(
                    "texture fragment USC lost shader-lane identity");
              }
              quad_facing_set = true;
              quad_front_facing = shader_lane.front_facing;
              if (!shader_lane.helper &&
                  (shader_lane.visible_invocation_index >= invocations.size() ||
                   invocations[shader_lane.visible_invocation_index].front_facing !=
                       shader_lane.front_facing))
                throw std::runtime_error("texture fragment USC visible facing identity mismatch");
              // The strict driver profile carries llvmpipe's coefficients,
              // whose origin already includes its half-pixel setup offset.
              const float interpolation_offset =
                  driver_pco_texture ||
                          state.functional_case ==
                              FunctionalCase::kDriverTexturedTriangles
                      ? 0.0F
                      : 0.5F;
              context.sample_x = FloatBits(static_cast<float>(shader_lane.x) +
                                           interpolation_offset);
              context.sample_y = FloatBits(static_cast<float>(shader_lane.y) +
                                           interpolation_offset);
              SetFragmentCentroidContext(
                  context, shader_lane.sample_mask,
                  state.raster_state.sample_mask,
                  centroid_primitive(shader_lane.parameter_index));
              SetFragmentSampleContext(
                  context, shader_lane.sample_id, shader_lane.sample_mask,
                  state.raster_state.sample_frequency != 0);
              context.memory_atomic32 = UscShaderImageMemory::Atomic32;
              context.image_memory_user_data = &image_memory;
              context.memory_side_effects_enabled = shader_lane.helper ? 0 : 1;
              SetFragmentFacingContext(context, shader_lane.front_facing);
              lane_contexts[shader_lane_index] = context;
              lane_context_initialized[shader_lane_index] = 1;
              if (debug_fragment && shader_lane.x == debug_x &&
                  shader_lane.y == debug_y && shader_lane.helper == 0) {
                std::cerr << "sequence-fragment-usc phase=context lane="
                          << global_lane_indices[shader_lane_index]
                          << " parameter=" << shader_lane.parameter_index
                          << " coefficients=";
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
              lane_executions[shader_lane_index] = ExecuteFragmentPco(
                  fragment_program, context);
            }
            resolve_quad_derivatives(quad);
            for (std::uint8_t lane = 0; lane < 4; ++lane) {
              if ((active_mask & (1U << lane)) == 0)
                continue;
              const auto shader_lane_index = quad.invocation_indices[lane];
              const auto &execution = lane_executions[shader_lane_index];
              if (sample_instruction_count == 0)
                commit_output(shader_lane_index, execution);
              else
                queue_suspension(shader_lane_index, execution, pending_requests,
                                 pending_continuations, pending_queued);
            }
          }
          if (std::any_of(lane_context_initialized.begin(),
                          lane_context_initialized.end(),
                          [](std::uint8_t value) { return value != 1; })) {
            throw std::runtime_error(
                "texture fragment USC did not initialize every shader lane");
          }

          // A resident chunk owns these buffers across all of its texture
          // FIFO rounds. clear/resize preserve host capacity, while the pool
          // payloads remain separate owned copies with the original lifetime.
          // The next-pending buffers are still full transactional snapshots:
          // do not mutate current pending state while validating a response.
          std::vector<std::size_t> batch_lanes;
          std::vector<TextureSampleRequest> batch_requests;
          std::vector<PcoFragmentContinuation> batch_continuations;
          std::vector<TextureSampleRequest> carried_requests;
          std::vector<TextureSampleResponse> responses;
          std::vector<PcoFragmentContinuation> saved_continuations;
          std::vector<TextureSampleRequest> next_requests;
          std::vector<PcoFragmentContinuation> next_continuations;
          std::vector<std::uint8_t> next_queued;
          while (std::any_of(pending_queued.begin(), pending_queued.end(),
                             [](std::uint8_t value) { return value != 0; })) {
            // Native branches may finish lanes or reach different SMPs. Send
            // only real suspended requests, retaining stable shader-lane IDs
            // and assigning dense request IDs within each texture FIFO batch.
            const auto first = static_cast<std::size_t>(
                std::find(pending_queued.begin(), pending_queued.end(), 1) -
                pending_queued.begin());
            const std::uint8_t descriptor_set =
                pending_requests[first].descriptor_set;
            const auto resume_pc =
                pending_continuations[first].resume_instruction_index;
            batch_lanes.clear();
            batch_requests.clear();
            batch_continuations.clear();
            // Reserve only after a real suspension exists. The resident lane
            // bound covers every possible compact batch, including divergent
            // rounds whose sizes shrink and later grow again.
            batch_lanes.reserve(pending_lane_count);
            batch_requests.reserve(pending_lane_count);
            batch_continuations.reserve(pending_lane_count);
            for (std::size_t lane_index = 0;
                 lane_index < pending_requests.size(); ++lane_index) {
              if (!pending_queued[lane_index] ||
                  pending_requests[lane_index].descriptor_set !=
                      descriptor_set ||
                  pending_continuations[lane_index].resume_instruction_index !=
                      resume_pc)
                continue;
              const TextureSampleRequest &request =
                  pending_requests[lane_index];
              if (pending_queued[lane_index] != 1 ||
                  request.shader_stage != ShaderStage::kFragment ||
                  request.shader_lane_index !=
                      global_lane_indices[lane_index] ||
                  request.request_id != lane_index ||
                  request.gather > 1 ||
                  request.gather != pending_requests[first].gather ||
                  request.descriptor_set != descriptor_set ||
                  request.data_request !=
                      pending_continuations[lane_index].data_request ||
                  pending_continuations[lane_index].valid != 1) {
                throw std::runtime_error(
                    "texture fragment USC request batch ordering is invalid");
              }
              batch_lanes.push_back(lane_index);
              batch_requests.push_back(request);
              batch_requests.back().request_id = batch_requests.size() - 1;
              batch_continuations.push_back(pending_continuations[lane_index]);
            }

            state.texture_sample_requests =
                StoreNewArray(pool_, batch_requests);
            state.fragment_continuations =
                StoreNewArray(pool_, batch_continuations);
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
            if (!fragment_program_identity_valid())
              throw std::runtime_error(
                  "texture fragment USC response changed its immutable program");
            if (!HasPoolHandle(state.texture_sample_requests) ||
                !HasPoolHandle(state.texture_sample_responses) ||
                !HasPoolHandle(state.fragment_continuations) ||
                HasPoolHandle(state.vertex_continuations)) {
              throw std::runtime_error(
                  "texture fragment USC received no response/continuation");
            }
            LoadFragmentScratchArray(pool_, state.texture_sample_requests,
                                     carried_requests);
            LoadFragmentScratchArray(pool_, state.texture_sample_responses,
                                     responses);
            LoadFragmentScratchArray(pool_, state.fragment_continuations,
                                     saved_continuations);
            if (carried_requests.size() != batch_lanes.size() ||
                responses.size() != batch_lanes.size() ||
                saved_continuations.size() != batch_lanes.size()) {
              throw std::runtime_error(
                  "texture fragment USC response lane count mismatch");
            }

            next_requests = pending_requests;
            next_continuations = pending_continuations;
            next_queued = pending_queued;
            for (std::size_t batch_index = 0; batch_index < batch_lanes.size();
                 ++batch_index) {
              const auto lane_index = batch_lanes[batch_index];
              next_queued[lane_index] = 0;
              const TextureSampleRequest &issued =
                  carried_requests[batch_index];
              const TextureSampleRequest &expected =
                  batch_requests[batch_index];
              const TextureSampleResponse &response = responses[batch_index];
              const PcoFragmentContinuation &saved =
                  saved_continuations[batch_index];
              const PcoFragmentContinuation &expected_continuation =
                  pending_continuations[lane_index];
              if (!SameTextureSampleRequest(issued, expected) ||
                  response.shader_stage != ShaderStage::kFragment ||
                  response.shader_lane_index != issued.shader_lane_index ||
                  response.request_id != issued.request_id ||
                  saved.valid != 1 ||
                  saved.data_request != issued.data_request ||
                  saved.program_binary_size !=
                      expected_continuation.program_binary_size ||
                  saved.program_instruction_count !=
                      expected_continuation.program_instruction_count ||
                  saved.program_signature !=
                      expected_continuation.program_signature ||
                  saved.resume_instruction_index !=
                      expected_continuation.resume_instruction_index ||
                  saved.pending_output_index !=
                      expected_continuation.pending_output_index ||
                  saved.pending_component_count !=
                      expected_continuation.pending_component_count ||
                  saved.temporary_written_mask !=
                      expected_continuation.temporary_written_mask ||
                  saved.temporaries != expected_continuation.temporaries ||
                  saved.kind != expected_continuation.kind ||
                  saved.pixel_outputs != expected_continuation.pixel_outputs ||
                  saved.written_mask != expected_continuation.written_mask ||
                  saved.depth != expected_continuation.depth ||
                  saved.depth_written != expected_continuation.depth_written ||
                  saved.predicate != expected_continuation.predicate ||
                  saved.predicate_valid !=
                      expected_continuation.predicate_valid ||
                  saved.discarded != expected_continuation.discarded ||
                  saved.execution_predicate !=
                      expected_continuation.execution_predicate ||
                  saved.front_facing != expected_continuation.front_facing ||
                  saved.front_facing_valid != expected_continuation.front_facing_valid ||
                  saved.native_steps != expected_continuation.native_steps ||
                  saved.executed_instructions.alu !=
                      expected_continuation.executed_instructions.alu ||
                  saved.executed_instructions.texture !=
                      expected_continuation.executed_instructions.texture ||
                  saved.executed_instructions.memory !=
                      expected_continuation.executed_instructions.memory ||
                  saved.loop_depth != expected_continuation.loop_depth ||
                  !std::equal(saved.loops.begin(), saved.loops.end(),
                              expected_continuation.loops.begin(),
                              [](const auto &left, const auto &right) {
                                return left.start_pc == right.start_pc &&
                                       left.count == right.count;
                              })) {
                throw std::runtime_error(
                    "texture fragment USC response ordering is invalid");
              }

              PcoFragmentExecutionContext resume_context =
                  lane_contexts[lane_index];
              if (debug_fragment && shader_lanes[lane_index].x == debug_x &&
                  shader_lanes[lane_index].y == debug_y &&
                  shader_lanes[lane_index].helper == 0) {
                std::cerr << "sequence-fragment-usc phase=resume lane="
                          << global_lane_indices[lane_index] << " round="
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
              lane_executions[lane_index] = ExecuteFragmentPco(
                  fragment_program, resume_context);
            }
            for (const FragmentQuad &quad : quads)
              resolve_quad_derivatives(quad);
            for (const auto index : batch_lanes)
              queue_suspension(index, lane_executions[index], next_requests,
                               next_continuations, next_queued);

            const PoolHandle request_payload = state.texture_sample_requests;
            const PoolHandle response_payload = state.texture_sample_responses;
            const PoolHandle continuation_payload =
                state.fragment_continuations;
            state.texture_sample_requests = {};
            state.texture_sample_responses = {};
            state.fragment_continuations = {};
            pool_.Release(request_payload);
            pool_.Release(response_payload);
            pool_.Release(continuation_payload);
            // A later residency batch may fail its input checks before the
            // next FIFO write. Keep the pool-owned state recoverable instead
            // of leaving already-released handles in its last response copy.
            StorePipelineState(pool_, txn.state, state);

            // Both sides retain capacity for the next round. Publication and
            // pool cleanup above must succeed before replacing pending state.
            pending_requests.swap(next_requests);
            pending_continuations.swap(next_continuations);
            pending_queued.swap(next_queued);
          }
          if (std::any_of(lane_completed.begin(), lane_completed.end(),
                          [](std::uint8_t value) { return value != 1; })) {
            throw std::runtime_error(
                "texture fragment USC did not complete every shader lane");
          }
        }
        if (std::any_of(visited_quads.begin(), visited_quads.end(),
                        [](std::uint8_t visited) { return visited != 1; }) ||
            std::any_of(visited_lanes.begin(), visited_lanes.end(),
                        [](std::uint8_t visited) { return visited != 1; }))
          throw std::runtime_error(
              "fragment residency did not execute every quad/lane");
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
                : LoadArray<std::uint32_t>(pool_,
                                           state.fragment_shared_registers);
        if (shared_registers.size() != expected_shared_count) {
          throw std::runtime_error(
              "driver PCO fragment USC shared-register count mismatch");
        }
        for (std::size_t index = 0; index < invocations.size(); ++index) {
          const FragmentInvocation &invocation = invocations[index];
          PcoFragmentExecutionContext context;
          context.raster_sample_count = state.raster_state.sample_count;
          context.memory_read = UscUniformBufferMemory::Read;
          context.memory_user_data = &uniform_memory;
          context.sample_x = FloatBits(static_cast<float>(invocation.x) + 0.5F);
          context.sample_y =
              FloatBits(static_cast<float>(invocation.y) + 0.5F);
          context.shared_count = static_cast<std::uint16_t>(
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
          context.raster_sample_count = state.raster_state.sample_count;
          context.memory_read = UscUniformBufferMemory::Read;
          context.memory_user_data = &uniform_memory;
          if (task.coefficient_dword_count > context.coefficients.size() ||
              task.coefficient_dword_count >
                  std::numeric_limits<std::uint8_t>::max()) {
            throw std::runtime_error(
                "varying fragment USC coefficient context is too small");
          }
          context.coefficient_count = static_cast<std::uint8_t>(
              task.coefficient_dword_count);
          context.shared_count =
              static_cast<std::uint16_t>(shared_registers.size());
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
                invocation.sample_id != quad.sample_id ||
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
                                  fragment_execution_lanes, &fragment_dynamic);
      state.fragment_outputs = StoreNewArray(pool_, outputs);
      for (auto &image : image_resources) {
        if (!(image.access & 2)) continue;
        const auto bytes = image_memory.Readback(image);
        if (HasPoolHandle(image.readback))
          StoreArray(pool_, image.readback, bytes);
        else
          image.readback = StoreNewArray(pool_, bytes);
      }
      if (!image_resources.empty()) {
        StoreArray(pool_, state.fragment_image_resources, image_resources);
        state.fragment_images_complete = 1;
        state.fragment_image_atomics += image_memory.atomics();
      }
      state.stage = PipelineStage::kFragmentShaded;
    }

    std::uint64_t cycles =
        groups == 0
            ? 0
            : kReferenceUarch.usc_cluster_base_cycles +
                  CeilDivide(groups,
                             kReferenceUarch.usc_groups_per_cluster_batch);
    // Texture suspension can reload PipelineState; apply the independent LD
    // accounting once to the latest state, after all shader lanes complete.
    ApplyMemoryAccessStats(state.counters, uniform_memory.stats());
    ApplyMemoryAccessStats(state.counters, image_memory.stats());
    AddInstructionCounter(cycles, MemoryAccessDelayCycles(image_memory.stats()));
    AddInstructionCounter(cycles, MemoryAccessDelayCycles(uniform_memory.stats()));
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
