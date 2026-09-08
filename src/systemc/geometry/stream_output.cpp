#include "geometry/stream_output.h"

#include "common/stream_output_types.h"
#include "common/tessellation_state.h"
#include "memory/gpu_memory_system.h"

#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace pvrgpu::stub {
namespace {

std::uint32_t PrimitiveVertices(PrimitiveTopology topology) {
  switch (topology) {
  case PrimitiveTopology::kPoints: return 1;
  case PrimitiveTopology::kLines:
  case PrimitiveTopology::kLineStrip:
  case PrimitiveTopology::kLineLoop: return 2;
  case PrimitiveTopology::kTriangleList:
  case PrimitiveTopology::kTriangleStrip:
  case PrimitiveTopology::kTriangleFan: return 3;
  default: throw std::runtime_error("stream output topology is unsupported");
  }
}

}  // namespace

void ValidateStreamOutputLayout(const std::vector<StreamOutputBinding> &bindings,
                                const std::vector<StreamOutputTarget> &targets,
                                std::uint32_t vertex_output_dwords) {
  if (bindings.empty() || bindings.size() > kMaximumStreamOutputBindings ||
      targets.size() > kMaximumStreamOutputBuffers ||
      vertex_output_dwords > kPcoVertexOutputCount)
    throw std::runtime_error("stream output table/register extent is invalid");
  std::array<const StreamOutputTarget *, kMaximumStreamOutputBuffers> by_buffer{};
  for (const auto &target : targets) {
    if (target.output_buffer >= by_buffer.size() || by_buffer[target.output_buffer] ||
        !target.resource_token || !target.target_token || !target.gpu_address ||
        target.gpu_address % 4 || !target.bytes_size ||
        target.bytes_size > kMaximumStreamOutputResourceBytes ||
        target.bytes_size > std::numeric_limits<std::uint64_t>::max() - target.gpu_address ||
        target.buffer_offset % 4 || target.internal_offset % 4 ||
        target.buffer_offset > target.bytes_size ||
        target.buffer_size > target.bytes_size - target.buffer_offset ||
        target.internal_offset > target.buffer_size || !target.stride_dwords ||
        target.stride_dwords > kPcoVertexOutputCount || HasPoolHandle(target.readback))
      throw std::runtime_error("stream output target address/range/cursor is invalid");
    for (const auto &other : targets) {
      if (&other == &target) break;
      if (other.target_token == target.target_token ||
          (other.resource_token == target.resource_token &&
           (other.gpu_address != target.gpu_address || other.bytes_size != target.bytes_size)) ||
          (other.resource_token != target.resource_token &&
           other.gpu_address < target.gpu_address + target.bytes_size &&
           target.gpu_address < other.gpu_address + other.bytes_size))
        throw std::runtime_error("stream output target/resource identity is inconsistent");
    }
    by_buffer[target.output_buffer] = &target;
  }
  for (const auto &binding : bindings) {
    if (binding.stream || binding.output_buffer >= by_buffer.size() ||
        !binding.num_components || binding.num_components > 4 ||
        binding.output_dword > vertex_output_dwords ||
        binding.num_components > vertex_output_dwords - binding.output_dword ||
        binding.dst_offset_dwords > kPcoVertexOutputCount ||
        binding.num_components > kPcoVertexOutputCount - binding.dst_offset_dwords)
      throw std::runtime_error("stream output binding native export range is invalid");
    const auto *target = by_buffer[binding.output_buffer];
    if (target && (binding.dst_offset_dwords > target->stride_dwords ||
                   binding.num_components > target->stride_dwords - binding.dst_offset_dwords))
      throw std::runtime_error("stream output binding exceeds target stride");
    for (const auto &prior : bindings) {
      if (&prior == &binding) break;
      if (prior.output_buffer == binding.output_buffer &&
          prior.dst_offset_dwords < binding.dst_offset_dwords + binding.num_components &&
          binding.dst_offset_dwords < prior.dst_offset_dwords + prior.num_components)
        throw std::runtime_error("stream output binding destination ranges overlap");
    }
  }
}

StreamOutput::StreamOutput(sc_core::sc_module_name name, MemoryPool &pool,
                           GpuMemorySystem *memory)
    : sc_module(name), pool_(pool), memory_(memory) { SC_THREAD(Run); }

void StreamOutput::Execute(PipelineState &state) {
  if (state.stage != PipelineStage::kVertexShaded || !memory_ ||
      memory_->mode() != state.memory_mode || HasPoolHandle(state.geometry_code) ||
      state.draw.topology != PrimitiveTopology::kTriangleList ||
      state.stream_output_complete || state.stream_output_primitives_written ||
      state.stream_output_primitives_storage_needed)
    throw std::runtime_error("stream output pipeline stage/memory contract is invalid");
  auto output_dwords = state.vertex_pco_abi.vertex_outputs;
  if (HasPoolHandle(state.tessellation_state)) {
    const auto tess = LoadArray<TessellationState>(pool_, state.tessellation_state);
    if (tess.size() != 1 || tess[0].phase != TessellationPhase::kEvaluationComplete ||
        state.tessellation_output_dwords != tess[0].evaluation_abi.vertex_outputs ||
        state.source_topology != (tess[0].point_mode ? PrimitiveTopology::kPoints :
            tess[0].domain == TessellationDomain::kIsolines ? PrimitiveTopology::kLines :
                                                           PrimitiveTopology::kTriangleList))
      throw std::runtime_error("stream output requires completed TES exports and topology");
    output_dwords = tess[0].evaluation_abi.vertex_outputs;
  }
  const auto bindings = HasPoolHandle(state.stream_output_bindings) ?
      LoadArray<StreamOutputBinding>(pool_, state.stream_output_bindings) : std::vector<StreamOutputBinding>{};
  auto targets = HasPoolHandle(state.stream_output_targets) ?
      LoadArray<StreamOutputTarget>(pool_, state.stream_output_targets) : std::vector<StreamOutputTarget>{};
  ValidateStreamOutputLayout(bindings, targets, output_dwords);
  const auto lanes = LoadArray<VertexLane>(pool_, state.vertex_lanes);
  const auto refs = LoadArray<VertexLaneRef>(pool_, state.vertex_lane_refs);
  const auto vertex_count = PrimitiveVertices(state.source_topology);
  if (refs.size() % 3)
    throw std::runtime_error("stream output connectivity is not complete expanded primitives");
  for (const auto &ref : refs) {
    if (ref.lane_index >= lanes.size() || !lanes[ref.lane_index].ended ||
        !lanes[ref.lane_index].emitted)
      throw std::runtime_error("stream output connectivity references unfinished native exports");
  }
  for (std::size_t first = 0; first < refs.size(); first += 3)
    for (std::uint32_t v = vertex_count; v < 3; ++v)
      if (refs[first + v].lane_index != refs[first + vertex_count - 1].lane_index)
        throw std::runtime_error("stream output point/line connectivity padding is invalid");
  std::array<int, kMaximumStreamOutputBuffers> by_buffer{{-1, -1, -1, -1}};
  std::array<bool, kMaximumStreamOutputBuffers> used{};
  for (std::size_t i = 0; i < targets.size(); ++i) {
    by_buffer[targets[i].output_buffer] = static_cast<int>(i);
    if (!memory_->backing().Contains(targets[i].gpu_address, targets[i].bytes_size))
      throw std::runtime_error("stream output resource has no complete modeled backing");
  }
  for (const auto &binding : bindings) used[binding.output_buffer] = true;

  // VertexFetch, or the completed TES stage, already produced ordered complete
  // primitive occurrences. Consume those references, not the deduplicated
  // shader lane array or padded point/line vertices. Whole-primitive preflight and
  // per-vertex cursor advancement follow Mesa draw_pt_so_emit.c semantics.
  for (std::size_t first = 0; first < refs.size() && !bindings.empty(); first += 3) {
    ++state.stream_output_primitives_storage_needed;
    bool overflow = false;
    for (std::size_t buffer = 0; buffer < used.size(); ++buffer) {
      if (!used[buffer]) continue;
      if (by_buffer[buffer] < 0) { overflow = true; continue; }
      const auto &target = targets[static_cast<std::size_t>(by_buffer[buffer])];
      const auto bytes = std::uint64_t(vertex_count) * target.stride_dwords * 4;
      if (bytes > target.buffer_size - target.internal_offset) overflow = true;
    }
    // Admission itself takes one fixed-function cycle, also for overflow.
    WaitForCycles(1);
    if (overflow) continue;
    for (std::uint32_t v = 0; v < vertex_count; ++v) {
      const auto &lane = lanes[refs[first + v].lane_index];
      for (const auto &binding : bindings) {
        auto &target = targets[static_cast<std::size_t>(by_buffer[binding.output_buffer])];
        const auto address = target.gpu_address + target.buffer_offset +
            target.internal_offset + std::uint64_t(binding.dst_offset_dwords) * 4;
        const auto stats = memory_->Write(address, lane.vertex_output + binding.output_dword,
            binding.num_components * sizeof(std::uint32_t), MemoryClient::kStreamOutput);
        ApplyMemoryAccessStats(state.counters, stats);
        WaitForCycles(MemoryAccessDelayCycles(stats));
      }
      for (auto &target : targets)
        if (used[target.output_buffer])
          target.internal_offset += target.stride_dwords * 4;
    }
    ++state.stream_output_primitives_written;
  }

  // Finish every target's writes before taking any snapshot: two target
  // objects may share one resource. Publish each nested handle immediately
  // so cancellation cleanup can reclaim partial readback completion safely.
  for (auto &target : targets) {
    const auto read = memory_->Readback(target.gpu_address, target.bytes_size,
                                       MemoryClient::kStreamOutput);
    ApplyMemoryAccessStats(state.counters, read.stats);
    WaitForCycles(MemoryAccessDelayCycles(read.stats));
    target.readback = StoreNewArray(pool_, read.data);
    auto &table = pool_.Write(state.stream_output_targets);
    std::memcpy(table.data(), targets.data(), table.size());
  }
  state.stream_output_complete = 1;
}

void StreamOutput::Run() {
  for (;;) {
    PipelineTxn txn;
    while (!input.nb_read(txn)) wait(input.data_written_event());
    auto state = LoadPipelineState(pool_, txn.state);
    if (HasPoolHandle(state.stream_output_bindings) || HasPoolHandle(state.stream_output_targets)) {
      try { Execute(state); }
      catch (...) { StorePipelineState(pool_, txn.state, state); throw; }
      StorePipelineState(pool_, txn.state, state);
    }
    while (!output.nb_write(txn)) wait(output.data_read_event());
  }
}

}  // namespace pvrgpu::stub
