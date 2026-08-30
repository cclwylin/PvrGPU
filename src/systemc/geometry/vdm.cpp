// Module：Vdm。
// 縮寫：VDM = Vertex Data Master（公開 Imagination register 文件之用語）。
// 功能：fail-closed 驗證 non-indexed Fill.Solid 或 generic uint16 indexed
// triangle-list draw；VBO capacity 由 MemoryPool resource/binding tables 推導，
// IA counters 由實際 draw/index occurrence/primitive 數量產生。FIFO 僅移交
// MemoryPool state handle，完成採 event-driven wait。
#include "geometry/vdm.h"

#include "common/functional_types.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pvrgpu::stub {
namespace {

inline constexpr std::uint64_t kFloat32Bytes = sizeof(float);

void ValidateDrawList(const MemoryPool &pool, PoolHandle handle) {
  if (!HasPoolHandle(handle))
    throw std::runtime_error("VDM received no DrawList statistics payload");
  const std::vector<DrawListStats> drawlists =
      LoadArray<DrawListStats>(pool, handle);
  if (drawlists.size() != 1 || drawlists[0].drawlist_index != 0)
    throw std::runtime_error("VDM requires exactly DrawList 0");
}

std::uint64_t ValidateVertexInputState(const MemoryPool &pool,
                                       const PipelineState &state) {
  if (!HasPoolHandle(state.vertex_buffer_resources) ||
      !HasPoolHandle(state.vertex_attribute_bindings)) {
    throw std::runtime_error("VDM received no vertex buffer/binding tables");
  }
  const std::vector<VertexBufferResource> resources =
      LoadArray<VertexBufferResource>(pool, state.vertex_buffer_resources);
  const std::vector<VertexAttributeBinding> bindings =
      LoadArray<VertexAttributeBinding>(pool,
                                        state.vertex_attribute_bindings);
  if (resources.empty() || bindings.empty())
    throw std::runtime_error("VDM received an empty vertex input layout");

  for (std::size_t index = 0; index < resources.size(); ++index) {
    const VertexBufferResource &resource = resources[index];
    if (!HasPoolHandle(resource.data) ||
        pool.Read(resource.data).size() != resource.byte_size) {
      throw std::runtime_error("VDM vertex buffer resource size mismatch");
    }
    for (std::size_t prior = 0; prior < index; ++prior) {
      if (resources[prior].data.slot == resource.data.slot &&
          resources[prior].data.generation == resource.data.generation) {
        throw std::runtime_error(
            "VDM vertex buffer resources must own unique payloads");
      }
    }
  }

  std::uint64_t vertex_capacity = std::numeric_limits<std::uint64_t>::max();
  for (const VertexAttributeBinding &binding : bindings) {
    if (binding.buffer_index >= resources.size() ||
        binding.component_type != VertexComponentType::kFloat32 ||
        binding.source_components == 0 || binding.source_components > 4 ||
        binding.destination_components < binding.source_components ||
        binding.destination_components > 4 || binding.normalized != 0 ||
        binding.integer != 0 || binding.instance_divisor != 0) {
      throw std::runtime_error("VDM received an unsupported vertex binding");
    }
    const std::uint64_t element_bytes =
        static_cast<std::uint64_t>(binding.source_components) * kFloat32Bytes;
    if (binding.stride_bytes < element_bytes)
      throw std::runtime_error("VDM vertex binding stride is too small");
    const VertexBufferResource &resource = resources[binding.buffer_index];
    const std::uint64_t first_element_end =
        static_cast<std::uint64_t>(binding.offset_bytes) + element_bytes;
    if (first_element_end > resource.byte_size)
      throw std::runtime_error("VDM vertex binding starts outside its VBO");
    const std::uint64_t capacity =
        1U + (resource.byte_size - first_element_end) / binding.stride_bytes;
    vertex_capacity = std::min(vertex_capacity, capacity);
  }
  if (vertex_capacity == 0 ||
      vertex_capacity == std::numeric_limits<std::uint64_t>::max()) {
    throw std::runtime_error("VDM vertex input capacity is invalid");
  }
  return vertex_capacity;
}

} // namespace

Vdm::Vdm(sc_core::sc_module_name name, MemoryPool &pool)
    : sc_module(name), pool_(pool) {
  SC_THREAD(Run);
}

void Vdm::Run() {
  while (true) {
    const PipelineTxn txn = input.read();
    PipelineState state = LoadPipelineState(pool_, txn.state);

    RequireStage(state.stage, PipelineStage::kSubmitted, name());
    if (!IsRasterFunctionalCase(state.functional_case))
      throw std::runtime_error("VDM received an unsupported functional case");
    ValidateDrawList(pool_, state.drawlist_stats);
    const std::uint64_t vertex_capacity =
        ValidateVertexInputState(pool_, state);

    if (IsFillSolidFamily(state.functional_case) ||
        IsTextureFamily(state.functional_case)) {
      if (state.draw.topology != PrimitiveTopology::kTriangleStrip ||
          state.draw.first_vertex != 0 || state.draw.vertex_count != 4 ||
          state.draw.first_index != 0 || state.draw.index_count != 0 ||
          state.draw.base_vertex != 0 ||
          state.draw.index_format != IndexFormat::kNone ||
          HasPoolHandle(state.vertex_indices)) {
        throw std::runtime_error(
            "VDM Fill.Solid requires non-indexed TRIANGLE_STRIP first=0 "
            "count=4");
      }
      const std::uint64_t vertex_end =
          static_cast<std::uint64_t>(state.draw.first_vertex) +
          state.draw.vertex_count;
      if (vertex_end > vertex_capacity)
        throw std::runtime_error("VDM Fill.Solid vertex range exceeds its VBO");
      state.counters.ia_vertices = state.draw.vertex_count;
      state.counters.ia_primitives = 2;
    } else {
      if (!IsIndexedTriangleRasterCase(state.functional_case) ||
          state.draw.topology != PrimitiveTopology::kTriangleList ||
          state.draw.first_vertex != 0 ||
          state.draw.index_format != IndexFormat::kUint16 ||
          state.draw.index_count == 0 || state.draw.index_count % 3 != 0) {
        throw std::runtime_error(
            "VDM indexed raster cases require a uint16 TRIANGLE_LIST");
      }
      if (!HasPoolHandle(state.vertex_indices))
        throw std::runtime_error("VDM indexed draw received no index buffer");
      const std::vector<std::uint16_t> indices =
          LoadArray<std::uint16_t>(pool_, state.vertex_indices);
      const std::uint64_t index_end =
          static_cast<std::uint64_t>(state.draw.first_index) +
          state.draw.index_count;
      if (index_end > indices.size()) {
        throw std::runtime_error(
            "VDM indexed draw range exceeds its index buffer");
      }
      for (std::uint64_t occurrence = state.draw.first_index;
           occurrence < index_end; ++occurrence) {
        const std::int64_t resolved =
            static_cast<std::int64_t>(indices[occurrence]) +
            state.draw.base_vertex;
        if (resolved < 0 ||
            static_cast<std::uint64_t>(resolved) >= vertex_capacity) {
          throw std::runtime_error(
              "VDM resolved index is outside the vertex input capacity");
        }
      }
      state.counters.ia_vertices = state.draw.index_count;
      state.counters.ia_primitives = state.draw.index_count / 3;
    }

    state.counters.drawlists = 1;
    state.stage = PipelineStage::kVdmComplete;

    const std::uint64_t cycles =
        kReferenceUarch.vdm_base_cycles +
        CeilDivide(state.counters.ia_vertices,
                   kReferenceUarch.vdm_vertices_per_batch);
    state.counters.vdm_cycles = cycles;
    state.counters.tiler_cycles += cycles;
    WaitForCycles(cycles);
    StorePipelineState(pool_, txn.state, state);
    output.write(txn);
  }
}

} // namespace pvrgpu::stub
