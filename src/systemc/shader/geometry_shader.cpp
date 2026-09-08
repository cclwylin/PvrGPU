#include "shader/geometry_shader.h"

#include "common/geometry_emission.h"
#include "memory/gpu_memory_system.h"
#include "shader/geometry_iss.h"
#include "shader/usc_uniform_buffer_memory.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pvrgpu::stub {

namespace {
constexpr std::uint64_t kMaximumGeometrySnapshots = UINT64_C(1) << 20;
class OwnedPayload {
 public:
  OwnedPayload(MemoryPool &pool, std::size_t bytes) : pool_(pool), handle_(pool.Allocate(bytes)) {}
  ~OwnedPayload() { if (HasPoolHandle(handle_)) pool_.Release(handle_); }
  OwnedPayload(const OwnedPayload &) = delete;
  OwnedPayload &operator=(const OwnedPayload &) = delete;
  template <typename T> T *data() { return reinterpret_cast<T *>(pool_.Write(handle_).data()); }
  PoolHandle Publish() { const auto handle = handle_; handle_ = {}; return handle; }
 private:
  MemoryPool &pool_;
  PoolHandle handle_;
};
void Require(GeometryEmissionStatus status, const char *operation) {
  if (status != GeometryEmissionStatus::kSuccess)
    throw std::runtime_error(std::string("geometry ") + operation + ": " + GeometryEmissionStatusName(status));
}
GeometryOutputTopology OutputTopology(PrimitiveTopology topology) {
  switch (topology) {
  case PrimitiveTopology::kPoints: return GeometryOutputTopology::kPoints;
  case PrimitiveTopology::kLineStrip: return GeometryOutputTopology::kLineStrip;
  case PrimitiveTopology::kTriangleStrip: return GeometryOutputTopology::kTriangleStrip;
  default: throw std::runtime_error("geometry output topology is invalid");
  }
}
struct InvocationContext {
  GpuMemorySystem &memory;
  UscUniformBufferMemory &uniforms;
  GeometryEmissionBuffer &emission;
  CounterTxn &counters;
  std::uint64_t input_address, input_bytes, required_mask;
  std::uint32_t maximum_vertices;
  static void Read(void *opaque, std::uint64_t address, std::uint32_t count, std::uint32_t *destination) {
    auto &self = *static_cast<InvocationContext *>(opaque);
    if (!destination || !count || count > 16 || address % 4)
      throw std::runtime_error("geometry LD destination/count/alignment is invalid");
    const std::uint64_t bytes = count * sizeof(std::uint32_t);
    if (address >= self.input_address && address - self.input_address <= self.input_bytes &&
        bytes <= self.input_bytes - (address - self.input_address)) {
      const auto read = self.memory.Read(address, bytes, MemoryClient::kGeometryShader);
      if (read.data.size() != bytes) throw std::runtime_error("geometry LD completion size mismatch");
      std::memcpy(destination, read.data.data(), bytes);
      ApplyMemoryAccessStats(self.counters, read.stats);
      self.counters.gs_input_read_bytes += bytes;
      WaitForCycles(MemoryAccessDelayCycles(read.stats));
    } else UscUniformBufferMemory::Read(&self.uniforms, address, count, destination);
  }
  static void Emit(void *opaque, const std::uint32_t *words, std::uint32_t count, std::uint64_t mask) {
    auto &self = *static_cast<InvocationContext *>(opaque);
    if (self.emission.emitted_vertices() < self.maximum_vertices && (mask & self.required_mask) != self.required_mask)
      throw std::runtime_error("geometry EmitVertex has unwritten position outputs");
    const auto status = self.emission.Emit(words, count, mask);
    if (status != GeometryEmissionStatus::kEmitSuppressed) Require(status, "EMIT");
  }
  static void Cut(void *opaque) { Require(static_cast<InvocationContext *>(opaque)->emission.EndPrimitive(), "CUT"); }
  static void Finish(void *opaque) { Require(static_cast<InvocationContext *>(opaque)->emission.Finish(), "ENDTASK"); }
};
} // namespace

GeometryShader::GeometryShader(sc_core::sc_module_name name, MemoryPool &pool, GpuMemorySystem *memory)
    : sc_module(name), pool_(pool), memory_(memory) { SC_THREAD(Run); }

void GeometryShader::Execute(PipelineState &state) {
  if (state.stage != PipelineStage::kVertexShaded || !memory_ || memory_->mode() != state.memory_mode ||
      !state.geometry_input_buffer_gpu_address || !state.geometry_invocations || state.geometry_invocations > 32 ||
      state.geometry_max_vertices > 256 || !state.geometry_input_stride_dwords ||
      state.geometry_input_stride_dwords > kPcoVertexOutputCount || state.position_output_count != 4 ||
      state.position_output_start > kPcoVertexOutputCount - 4)
    throw std::runtime_error("geometry pipeline state/register/memory contract is invalid");
  const auto topology = OutputTopology(state.geometry_output_topology);
  const auto program = DecodeGeometryPcoProgram(LoadArray<std::uint8_t>(pool_, state.geometry_code));
  ValidateGeometryProgram(program, state.geometry_pco_abi);
  if (HasPoolHandle(state.geometry_instructions)) pool_.Release(state.geometry_instructions);
  state.geometry_instructions = {};
  state.geometry_program_summary = program.summary;
  state.geometry_instructions = StoreNewArray(pool_, program.instructions);
  const auto primitives = LoadArray<GeometryInputPrimitive>(pool_, state.geometry_input_primitives);
  const auto source_lanes = LoadArray<VertexLane>(pool_, state.vertex_lanes);
  const auto source_refs = LoadArray<VertexLaneRef>(pool_, state.vertex_lane_refs);
  const auto max_snapshots = static_cast<std::uint64_t>(primitives.size()) * state.geometry_invocations * state.geometry_max_vertices;
  if (max_snapshots > kMaximumGeometrySnapshots) throw std::runtime_error("geometry per-draw snapshot bound exceeded");
  auto shared = LoadArray<std::uint32_t>(pool_, state.geometry_shared_registers);
  if (shared.size() != state.geometry_pco_abi.shareds || shared.size() < 4)
    throw std::runtime_error("geometry shared register transport size mismatch");
  UscUniformBufferMemory uniforms(memory_, state.memory_mode,
      HasPoolHandle(state.geometry_uniform_buffer_resources) ?
          LoadArray<UniformBufferResource>(pool_, state.geometry_uniform_buffer_resources) : std::vector<UniformBufferResource>{});
  const auto stride = state.geometry_pco_abi.vertex_outputs;
  OwnedPayload snapshots(pool_, static_cast<std::size_t>(state.geometry_max_vertices) * stride * 4);
  OwnedPayload masks(pool_, static_cast<std::size_t>(state.geometry_max_vertices) * sizeof(std::uint64_t));
  OwnedPayload strips(pool_, static_cast<std::size_t>(state.geometry_max_vertices) * sizeof(GeometryStripRange));
  OwnedPayload task_payload(pool_, sizeof(GeometryTaskState));
  std::vector<VertexLane> lanes;
  std::vector<VertexLaneRef> refs;
  std::vector<GeometryRasterPrimitive> raster;
  lanes.reserve(max_snapshots); refs.reserve(max_snapshots * 3); raster.reserve(max_snapshots);
  std::vector<GeometryPrimitiveRef> expanded(state.geometry_max_vertices);
  GeometryExecutionStats execution;
  const auto required_mask = UINT64_C(0xf) << state.position_output_start;
  for (const auto &primitive : primitives) {
    if (primitive.vertex_count != state.geometry_input_primitive_vertices ||
        !(primitive.vertex_count == 1 || primitive.vertex_count == 2 || primitive.vertex_count == 3 ||
          primitive.vertex_count == 4 || primitive.vertex_count == 6))
      throw std::runtime_error("geometry input count disagrees with linked layout");
    const auto input_words = primitive.vertex_count * state.geometry_input_stride_dwords;
    std::array<std::uint32_t, 6 * kPcoVertexOutputCount> primitive_words{};
    for (unsigned vertex = 0; vertex < primitive.vertex_count; ++vertex) {
      const auto occurrence = primitive.vertex_indices[vertex];
      if (occurrence >= source_refs.size() || source_refs[occurrence].lane_index >= source_lanes.size())
        throw std::runtime_error("geometry input occurrence does not name a shaded VS lane");
      const auto &lane = source_lanes[source_refs[occurrence].lane_index];
      if (!lane.emitted || !lane.ended) throw std::runtime_error("geometry input references unfinished VS invocation");
      std::copy_n(lane.vertex_output, state.geometry_input_stride_dwords,
                   primitive_words.data() + vertex * state.geometry_input_stride_dwords);
    }
    const auto write = memory_->Write(state.geometry_input_buffer_gpu_address,
        primitive_words.data(), input_words * sizeof(std::uint32_t), MemoryClient::kGeometryShader);
    ApplyMemoryAccessStats(state.counters, write);
    state.counters.gs_input_write_bytes += input_words * sizeof(std::uint32_t);
    WaitForCycles(MemoryAccessDelayCycles(write));
    shared[0] = static_cast<std::uint32_t>(state.geometry_input_buffer_gpu_address);
    shared[1] = static_cast<std::uint32_t>(state.geometry_input_buffer_gpu_address >> 32U);
    shared[2] = input_words * sizeof(std::uint32_t); shared[3] = 0;
    for (unsigned invocation = 0; invocation < state.geometry_invocations; ++invocation) {
      auto &task = *task_payload.data<GeometryTaskState>();
      task = MakeGeometryTask(state.geometry_pco_abi, shared, primitive.primitive_id, invocation);
      GeometryEmissionBuffer emission({snapshots.data<std::uint32_t>(),
          static_cast<std::size_t>(state.geometry_max_vertices) * stride, masks.data<std::uint64_t>(),
          state.geometry_max_vertices, strips.data<GeometryStripRange>(), state.geometry_max_vertices},
          state.geometry_max_vertices, stride);
      InvocationContext context{*memory_, uniforms, emission, state.counters, state.geometry_input_buffer_gpu_address,
          input_words * sizeof(std::uint32_t), required_mask, state.geometry_max_vertices};
      const GeometryExecutionCallbacks callbacks{&context, InvocationContext::Read,
          InvocationContext::Emit, InvocationContext::Cut, InvocationContext::Finish};
      while (!task.ended) StepGeometryTask(program, state.geometry_pco_abi, task, callbacks, execution);
      ++state.counters.gs_invocations;
      std::size_t complete = 0;
      Require(ExpandGeometryPrimitives(topology, strips.data<GeometryStripRange>(), emission.strip_count(),
          emission.emitted_vertices(), expanded.data(), expanded.size(), complete), "primitive assembly");
      const auto base = static_cast<std::uint32_t>(lanes.size());
      for (unsigned vertex = 0; vertex < emission.emitted_vertices(); ++vertex) {
        VertexLane lane;
        std::copy_n(snapshots.data<std::uint32_t>() + vertex * stride, stride, lane.vertex_output);
        lane.emitted = lane.ended = 1;
        lanes.push_back(lane);
      }
      for (std::size_t index = 0; index < complete; ++index) {
        GeometryRasterPrimitive metadata;
        metadata.refs = expanded[index]; metadata.input_primitive_id = primitive.primitive_id;
        metadata.instance_id = primitive.instance_id; metadata.invocation_id = invocation;
        for (auto &vertex : metadata.refs.vertex_indices) { vertex += base; refs.push_back({vertex, vertex}); }
        raster.push_back(metadata);
      }
      state.counters.gs_primitives += complete;
    }
  }
  ApplyMemoryAccessStats(state.counters, uniforms.stats());
  WaitForCycles(MemoryAccessDelayCycles(uniforms.stats()));
  state.counters.pco_decode_cycles += program.summary.group_count;
  state.counters.pco_instructions += program.summary.instruction_count;
  state.counters.usc_groups += execution.instructions;
  state.counters.usc_cluster_cycles += execution.instructions;
  state.counters.gs_alu_instructions += execution.alu_instructions;
  state.counters.gs_memory_instructions += execution.memory_instructions;
  state.counters.gs_load_instructions += execution.load_instructions;
  state.counters.gs_emitted_vertices += lanes.size();
  if (HasPoolHandle(state.drawlist_stats)) {
    auto stats = LoadArray<DrawListStats>(pool_, state.drawlist_stats);
    if (stats.size() != 1) throw std::runtime_error("geometry requires one DrawList statistics record");
    auto &gs = stats[0].geometry;
    const auto composition = CountPcoInstructions(program.instructions, false);
    gs.invocations = state.counters.gs_invocations;
    gs.program_groups = program.summary.group_count; gs.program_instructions = program.summary.instruction_count;
    gs.program_alu_instructions = composition.alu; gs.program_memory_instructions = composition.memory;
    gs.executed_alu_instructions = execution.alu_instructions; gs.executed_memory_instructions = execution.memory_instructions;
    gs.program_recorded = gs.executions_recorded = 1;
    StoreArray(pool_, state.drawlist_stats, stats);
  }
  WaitForCycles(program.summary.group_count + execution.instructions);
  // Publish only complete native output; no downstream stage sees an open strip.
  OwnedPayload new_lanes(pool_, lanes.size() * sizeof(VertexLane));
  OwnedPayload new_refs(pool_, refs.size() * sizeof(VertexLaneRef));
  OwnedPayload new_primitives(pool_, raster.size() * sizeof(GeometryRasterPrimitive));
  if (!lanes.empty()) std::memcpy(new_lanes.data<VertexLane>(), lanes.data(), lanes.size() * sizeof(VertexLane));
  if (!refs.empty()) std::memcpy(new_refs.data<VertexLaneRef>(), refs.data(), refs.size() * sizeof(VertexLaneRef));
  if (!raster.empty()) std::memcpy(new_primitives.data<GeometryRasterPrimitive>(), raster.data(), raster.size() * sizeof(GeometryRasterPrimitive));
  pool_.Release(state.vertex_lanes); pool_.Release(state.vertex_lane_refs);
  if (HasPoolHandle(state.geometry_primitives)) pool_.Release(state.geometry_primitives);
  state.vertex_lanes = new_lanes.Publish(); state.vertex_lane_refs = new_refs.Publish();
  state.geometry_primitives = new_primitives.Publish();
  for (auto *handle : {&state.vertex_indices, &state.expanded_source_vertices}) {
    if (HasPoolHandle(*handle)) pool_.Release(*handle);
    *handle = {};
  }
  state.source_topology = state.geometry_output_topology;
  state.draw.topology = PrimitiveTopology::kTriangleList;
  state.draw.vertex_count = static_cast<std::uint32_t>(refs.size());
  state.draw.first_vertex = state.draw.first_index = state.draw.index_count = 0;
  state.draw.base_vertex = 0; state.draw.index_format = IndexFormat::kNone;
  state.index_buffer_gpu_address = state.index_buffer_bytes = 0;
}

void GeometryShader::Run() {
  for (;;) {
    PipelineTxn txn;
    while (!input.nb_read(txn)) wait(input.data_written_event());
    auto state = LoadPipelineState(pool_, txn.state);
    if (HasPoolHandle(state.geometry_code)) {
      try { Execute(state); }
      catch (...) {
        // Root cancellation owns the current handles, including any decoder
        // payload allocated before a later task failed. Never leave stale
        // pre-execution handles in the serialized pipeline control block.
        StorePipelineState(pool_, txn.state, state);
        throw;
      }
      StorePipelineState(pool_, txn.state, state);
    }
    while (!output.nb_write(txn)) wait(output.data_read_event());
  }
}

}  // namespace pvrgpu::stub
