// Module：Vdm。
// 縮寫：VDM = Vertex Data Master（公開 Imagination register 文件之用語）。
// 功能：fail-closed 驗證 non-indexed Fill.Solid 或 generic uint16 indexed
// triangle-list draw；VBO capacity 由 MemoryPool resource/binding tables 推導，
// IA counters 由實際 draw/index occurrence/primitive 數量產生。FIFO 僅移交
// MemoryPool state handle，完成採 event-driven wait。
#include "geometry/vdm.h"

#include "common/functional_types.h"
#include "common/geometry_emission.h"
#include "common/tessellation_state.h"
#include "memory/gpu_memory_system.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pvrgpu::stub {
namespace {

// inline constexpr std::uint64_t kFloat32Bytes = sizeof(float);

void ValidateDrawList(const MemoryPool &pool, PoolHandle handle) {
  if (!HasPoolHandle(handle))
    throw std::runtime_error("VDM received no DrawList statistics payload");
  const std::vector<DrawListStats> drawlists =
      LoadArray<DrawListStats>(pool, handle);
  if (drawlists.size() != 1 || drawlists[0].drawlist_index != 0)
    throw std::runtime_error("VDM requires exactly DrawList 0");
}

GeometryInputTopology GeometryTopology(PrimitiveTopology topology) {
  switch (topology) {
    case PrimitiveTopology::kPoints: return GeometryInputTopology::kPoints;
    case PrimitiveTopology::kLines: return GeometryInputTopology::kLines;
    case PrimitiveTopology::kLineStrip: return GeometryInputTopology::kLineStrip;
    case PrimitiveTopology::kLineLoop: return GeometryInputTopology::kLineLoop;
    case PrimitiveTopology::kTriangleList: return GeometryInputTopology::kTriangles;
    case PrimitiveTopology::kTriangleStrip: return GeometryInputTopology::kTriangleStrip;
    case PrimitiveTopology::kTriangleFan: return GeometryInputTopology::kTriangleFan;
    case PrimitiveTopology::kLinesAdjacency: return GeometryInputTopology::kLinesAdjacency;
    case PrimitiveTopology::kLineStripAdjacency: return GeometryInputTopology::kLineStripAdjacency;
    case PrimitiveTopology::kTrianglesAdjacency: return GeometryInputTopology::kTrianglesAdjacency;
    case PrimitiveTopology::kTriangleStripAdjacency: return GeometryInputTopology::kTriangleStripAdjacency;
    case PrimitiveTopology::kPatches: break;
  }
  throw std::runtime_error("VDM geometry input topology is invalid");
}

std::uint64_t ValidateGeometryInput(const MemoryPool &pool,
                                    const PipelineState &state,
                                    std::uint32_t occurrence_count) {
  if (!HasPoolHandle(state.geometry_input_primitives))
    throw std::runtime_error("VDM geometry_input_primitives is absent");
  if (!state.geometry_vertices_per_instance ||
      occurrence_count % state.geometry_vertices_per_instance)
    throw std::runtime_error("VDM geometry_vertices_per_instance is invalid");
  if (state.primitive_restart_enable)
    throw std::runtime_error("VDM geometry primitive_restart is not transported");
  const auto actual = LoadArray<GeometryInputPrimitive>(
      pool, state.geometry_input_primitives);
  if (actual.size() > occurrence_count)
    throw std::runtime_error("VDM geometry primitive count exceeds occurrences");
  const auto topology = GeometryTopology(state.draw.topology);
  std::vector<std::uint32_t> occurrences(state.geometry_vertices_per_instance);
  std::vector<GeometryInputPrimitive> expected(occurrences.size());
  std::size_t cursor = 0;
  for (std::uint32_t instance = 0;
       instance < occurrence_count / state.geometry_vertices_per_instance;
       ++instance) {
    for (std::size_t i = 0; i < occurrences.size(); ++i)
      occurrences[i] = instance * state.geometry_vertices_per_instance + i;
    std::size_t count = 0;
    const auto status = AssembleGeometryInputPrimitives(
        topology, occurrences.data(), occurrences.size(), false, 0, instance,
        expected.data(), expected.size(), count);
    if (status != GeometryEmissionStatus::kSuccess)
      throw std::runtime_error(std::string("VDM geometry input assembly: ") +
                                GeometryEmissionStatusName(status));
    if (count > actual.size() - cursor)
      throw std::runtime_error("VDM geometry primitive metadata is incomplete");
    for (std::size_t i = 0; i < count; ++i) {
      const auto &a = actual[cursor + i];
      const auto &e = expected[i];
      if (a.primitive_id != e.primitive_id || a.instance_id != e.instance_id ||
          a.vertex_count != e.vertex_count ||
          a.vertex_count != state.geometry_input_primitive_vertices ||
          !std::equal(std::begin(a.vertex_indices), std::end(a.vertex_indices),
                        std::begin(e.vertex_indices)) ||
          a.reserved[0] || a.reserved[1] || a.reserved[2])
        throw std::runtime_error("VDM geometry input occurrence metadata mismatch");
    }
    cursor += count;
  }
  if (cursor != actual.size())
    throw std::runtime_error("VDM geometry primitive metadata has extra entries");
  return cursor;
}

std::uint64_t ValidateVertexInputState(const MemoryPool &pool,
                                       const PipelineState &state,
                                       const GpuMemorySystem *memory) {
  if (!HasPoolHandle(state.vertex_buffer_resources) ||
      !HasPoolHandle(state.vertex_attribute_bindings)) {
    throw std::runtime_error("VDM received no vertex buffer/binding tables");
  }
  const std::vector<VertexBufferResource> resources =
      LoadArray<VertexBufferResource>(pool, state.vertex_buffer_resources);
  const std::vector<VertexAttributeBinding> bindings =
      LoadArray<VertexAttributeBinding>(pool,
                                        state.vertex_attribute_bindings);
  if ((HasPoolHandle(state.geometry_code) || HasPoolHandle(state.tessellation_state)) && state.driver_describes_attributes &&
      state.vertex_pco_abi.vertex_inputs == 0 && resources.empty() && bindings.empty())
    return state.draw.vertex_count;
  if (resources.empty() || bindings.empty())
    throw std::runtime_error("VDM received an empty vertex input layout");

  for (std::size_t index = 0; index < resources.size(); ++index) {
    const VertexBufferResource &resource = resources[index];
    if (memory) {
      if (HasPoolHandle(resource.data) || resource.gpu_address == 0 ||
          resource.byte_size == 0 ||
          !memory->backing().Contains(resource.gpu_address,
                                      resource.byte_size)) {
        throw std::runtime_error(
            "VDM DRAM-backed vertex resource is invalid");
      }
    } else if (!HasPoolHandle(resource.data) ||
               pool.Read(resource.data).size() != resource.byte_size) {
      throw std::runtime_error("VDM vertex buffer resource size mismatch");
    }
    for (std::size_t prior = 0; prior < index; ++prior) {
      const bool aliases = memory
                               ? resources[prior].gpu_address ==
                                     resource.gpu_address
                               : resources[prior].data.slot ==
                                         resource.data.slot &&
                                     resources[prior].data.generation ==
                                         resource.data.generation;
      if (aliases) {
        throw std::runtime_error(
            "VDM vertex buffer resources must own unique payloads");
      }
    }
  }

  std::uint64_t vertex_capacity = std::numeric_limits<std::uint64_t>::max();
  bool has_vertex_binding = false;
  for (const VertexAttributeBinding &binding : bindings) {
    if (binding.buffer_index >= resources.size() ||
        binding.source_components == 0 || binding.source_components > 4 ||
        binding.destination_components < binding.source_components ||
        binding.destination_components > 4) {
      throw std::runtime_error("VDM received an unsupported vertex binding");
    }
    const std::uint64_t element_bytes =
        static_cast<std::uint64_t>(binding.source_components) *
        GetComponentTypeBytes(binding.component_type);
    if (binding.stride_bytes < element_bytes)
      throw std::runtime_error("VDM vertex binding stride is too small");
    const VertexBufferResource &resource = resources[binding.buffer_index];
    const std::uint64_t first_element_end =
        static_cast<std::uint64_t>(binding.offset_bytes) + element_bytes;
    if (first_element_end > resource.byte_size)
      throw std::runtime_error("VDM vertex binding starts outside its VBO");

    // Instanced attributes scale with instance_count, not vertex_count
    if (binding.instance_divisor != 0) {
      continue;
    }
    has_vertex_binding = true;

    const std::uint64_t capacity =
        1U + (resource.byte_size - first_element_end) / binding.stride_bytes;
    vertex_capacity = std::min(vertex_capacity, capacity);
  }
  if (has_vertex_binding && (vertex_capacity == 0 ||
      vertex_capacity == std::numeric_limits<std::uint64_t>::max())) {
    throw std::runtime_error("VDM vertex input capacity is invalid");
  }
  return has_vertex_binding ? vertex_capacity : std::numeric_limits<std::uint64_t>::max();
}

} // namespace

Vdm::Vdm(sc_core::sc_module_name name, MemoryPool &pool,
         GpuMemorySystem *memory)
    : sc_module(name), pool_(pool), memory_(memory) {
  SC_THREAD(Run);
}

void Vdm::Run() {
  while (true) {
    const PipelineTxn txn = input.read();
    PipelineState state = LoadPipelineState(pool_, txn.state);

    RequireStage(state.stage, PipelineStage::kSubmitted, name());
    if (!IsRasterFunctionalCase(state.functional_case))
      throw std::runtime_error("VDM received an unsupported functional case");
    if (memory_ && state.memory_mode != memory_->mode())
      throw std::runtime_error("VDM memory mode mismatch");
    ValidateDrawList(pool_, state.drawlist_stats);
    const std::uint64_t vertex_capacity =
        ValidateVertexInputState(pool_, state, memory_);
    MemoryAccessStats memory_stats;

    const bool driver_pco_triangles =
        IsDriverPcoTrianglesCase(state.functional_case);
    const bool tessellation_enabled = HasPoolHandle(state.tessellation_state);
    const bool geometry_enabled = HasPoolHandle(state.geometry_code) || tessellation_enabled;
    if (geometry_enabled && !driver_pco_triangles)
      throw std::runtime_error("VDM geometry_code requires a native driver draw");
    /*
     * A lowered draw that carries an index buffer belongs on the indexed path
     * below, which fetches the indices and assembles every GLES topology from
     * them.  Routing it to the direct-raster branch instead was what made an
     * indexed draw unrepresentable: that branch reads vertices in order and
     * rejects an index buffer outright.
     */
    const bool driver_pco_indexed =
        driver_pco_triangles &&
        state.draw.index_format != IndexFormat::kNone;
    if (IsFillSolidFamily(state.functional_case) ||
        IsTextureFamily(state.functional_case) ||
        (driver_pco_triangles && !driver_pco_indexed)) {
      const bool driver_textured_triangles =
          state.functional_case == FunctionalCase::kDriverTexturedTriangles;
      const PrimitiveTopology expected_topology =
          driver_textured_triangles ? PrimitiveTopology::kTriangleList
                                    : PrimitiveTopology::kTriangleStrip;
      const std::uint32_t expected_vertex_count =
          driver_textured_triangles ? 6U : 4U;
      // Name the property that is wrong.  "unsupported topology or range"
      // covers eight conditions, and a capture that trips one of them gives
      // no indication which pipeline feature it actually needs.
      const char *direct_reason = nullptr;
      if (driver_pco_triangles) {
        if (!geometry_enabled &&
            state.draw.topology != PrimitiveTopology::kTriangleList)
          direct_reason = "topology_not_triangle_list";
        else if (state.draw.vertex_count == 0)
          direct_reason = "empty_draw";
        else if (!geometry_enabled && state.draw.vertex_count % 3 != 0)
          direct_reason = "vertex_count_not_a_triangle_multiple";
        else if (state.primitive_restart_enable != 0)
          direct_reason = "primitive_restart";
      } else if (state.draw.topology != expected_topology) {
        direct_reason = "topology";
      } else if (state.draw.first_vertex != 0) {
        direct_reason = "first_vertex";
      } else if (state.draw.vertex_count != expected_vertex_count) {
        direct_reason = "vertex_count";
      }
      if (!direct_reason) {
        if (state.draw.first_index != 0)
          direct_reason = "first_index";
        else if (state.draw.index_count != 0)
          direct_reason = "index_count";
        else if (state.draw.base_vertex != 0)
          direct_reason = "base_vertex";
        else if (state.draw.index_format != IndexFormat::kNone)
          direct_reason = "index_format";
        else if (HasPoolHandle(state.vertex_indices))
          direct_reason = "vertex_indices";
        else if (state.index_buffer_gpu_address != 0)
          direct_reason = "index_buffer_address";
        else if (state.index_buffer_bytes != 0)
          direct_reason = "index_buffer_bytes";
      }
      if (direct_reason) {
        throw std::runtime_error(
            std::string("VDM direct raster draw is unsupported: ") +
            direct_reason + " (topology=" +
            std::to_string(static_cast<int>(state.draw.topology)) +
            " vertices=" + std::to_string(state.draw.vertex_count) +
            " indices=" + std::to_string(state.draw.index_count) + ")");
      }
      const std::uint64_t vertex_end =
          static_cast<std::uint64_t>(state.draw.first_vertex) +
          state.draw.vertex_count;
      if (vertex_end > vertex_capacity)
        throw std::runtime_error("VDM direct vertex range exceeds its VBO");
      state.counters.ia_vertices = state.draw.vertex_count;
      state.counters.ia_primitives =
          tessellation_enabled ? 0U : geometry_enabled ? ValidateGeometryInput(pool_, state, state.draw.vertex_count)
          : driver_pco_triangles ? state.draw.vertex_count / 3U : 2U;
    } else {
      if ((!IsIndexedTriangleRasterCase(state.functional_case) &&
           !driver_pco_indexed) ||
          (state.draw.topology != PrimitiveTopology::kTriangleList &&
           state.draw.topology != PrimitiveTopology::kTriangleStrip &&
           state.draw.topology != PrimitiveTopology::kPoints &&
           state.draw.topology != PrimitiveTopology::kLines &&
           state.draw.topology != PrimitiveTopology::kLineStrip &&
           state.draw.topology != PrimitiveTopology::kLineLoop &&
           state.draw.topology != PrimitiveTopology::kTriangleFan &&
           !(tessellation_enabled && state.draw.topology == PrimitiveTopology::kPatches) &&
           !(geometry_enabled &&
             (state.draw.topology == PrimitiveTopology::kLinesAdjacency ||
              state.draw.topology == PrimitiveTopology::kLineStripAdjacency ||
              state.draw.topology == PrimitiveTopology::kTrianglesAdjacency ||
              state.draw.topology == PrimitiveTopology::kTriangleStripAdjacency))) ||
          state.draw.first_vertex != 0 ||
          state.draw.index_format == IndexFormat::kNone ||
          state.draw.index_count == 0) {
        throw std::runtime_error(
            "VDM indexed raster cases require a valid GLES indexed draw command");
      }
      if (!HasPoolHandle(state.vertex_indices)) {
        if (!memory_ || state.index_buffer_gpu_address == 0 ||
            state.index_buffer_bytes == 0 ||
            !memory_->backing().Contains(state.index_buffer_gpu_address,
                                         state.index_buffer_bytes)) {
          throw std::runtime_error("VDM indexed draw received no index buffer");
        }
        MemoryReadResult read = memory_->Read(
            state.index_buffer_gpu_address,
            static_cast<std::size_t>(state.index_buffer_bytes),
            MemoryClient::kIndexFetch);
        memory_stats += read.stats;
        state.vertex_indices = pool_.Allocate(read.data.size());
        pool_.Write(state.vertex_indices) = std::move(read.data);
      }

      std::vector<std::uint32_t> indices;
      if (state.draw.index_format == IndexFormat::kUint8) {
        const std::vector<std::uint8_t> raw = LoadArray<std::uint8_t>(pool_, state.vertex_indices);
        indices.assign(raw.begin(), raw.end());
      } else if (state.draw.index_format == IndexFormat::kUint16) {
        const std::vector<std::uint16_t> raw = LoadArray<std::uint16_t>(pool_, state.vertex_indices);
        indices.assign(raw.begin(), raw.end());
      } else if (state.draw.index_format == IndexFormat::kUint32) {
        indices = LoadArray<std::uint32_t>(pool_, state.vertex_indices);
      } else {
        throw std::runtime_error("VDM received an unsupported index format");
      }

      const std::uint64_t index_end =
          static_cast<std::uint64_t>(state.draw.first_index) +
          state.draw.index_count;
      if (index_end > indices.size()) {
        throw std::runtime_error(
            "VDM indexed draw range exceeds its index buffer");
      }

      std::uint64_t ia_primitives = 0;
      std::uint64_t ia_vertices = 0;

      std::vector<std::vector<std::uint32_t>> segments;
      std::vector<std::uint32_t> current_segment;

      for (std::uint64_t occurrence = state.draw.first_index;
           occurrence < index_end; ++occurrence) {
        std::uint32_t idx = indices[occurrence];
        if (state.primitive_restart_enable && idx == state.primitive_restart_index) {
          if (!current_segment.empty()) {
            segments.push_back(current_segment);
            current_segment.clear();
          }
        } else {
          const std::int64_t resolved =
              static_cast<std::int64_t>(idx) + state.draw.base_vertex;
          if (resolved < 0 ||
              static_cast<std::uint64_t>(resolved) >= vertex_capacity) {
            // Say which index, from where, against what -- the numbers are the
            // whole diagnosis, and without them this abort names no field.
            throw std::runtime_error(
                "VDM resolved index is outside the vertex input capacity: "
                "index=" + std::to_string(idx) +
                " base_vertex=" + std::to_string(state.draw.base_vertex) +
                " resolved=" + std::to_string(resolved) +
                " capacity=" + std::to_string(vertex_capacity) +
                " occurrence=" + std::to_string(occurrence) +
                " index_count=" + std::to_string(state.draw.index_count));
          }
          current_segment.push_back(idx);
          ia_vertices++;
        }
      }
      if (!current_segment.empty()) {
        segments.push_back(current_segment);
      }

      for (const auto& seg : segments) {
        std::uint64_t n = seg.size();
        if (state.draw.topology == PrimitiveTopology::kPoints) {
          ia_primitives += n;
        } else if (state.draw.topology == PrimitiveTopology::kLines) {
          ia_primitives += n / 2;
        } else if (state.draw.topology == PrimitiveTopology::kLineStrip) {
          if (n >= 2) ia_primitives += (n - 1);
        } else if (state.draw.topology == PrimitiveTopology::kLineLoop) {
          if (n >= 2) ia_primitives += n;
        } else if (state.draw.topology == PrimitiveTopology::kTriangleList) {
          ia_primitives += n / 3;
        } else if (state.draw.topology == PrimitiveTopology::kTriangleStrip ||
                   state.draw.topology == PrimitiveTopology::kTriangleFan) {
          if (n >= 3) ia_primitives += (n - 2);
        }
      }

      state.counters.ia_vertices = ia_vertices;
      state.counters.ia_primitives = ia_primitives;
      if (geometry_enabled && !tessellation_enabled)
        state.counters.ia_primitives =
            ValidateGeometryInput(pool_, state, state.draw.index_count);
    }

    if (tessellation_enabled) {
      const auto tess = LoadArray<TessellationState>(pool_, state.tessellation_state);
      if (tess.size() != 1 || state.draw.topology != PrimitiveTopology::kPatches ||
          state.primitive_restart_enable)
        throw std::runtime_error("VDM tessellation patch contract is invalid");
      state.counters.ia_primitives = LoadArray<TessellationPatch>(pool_, tess[0].patches).size();
    }
    state.counters.drawlists = 1;
    state.stage = PipelineStage::kVdmComplete;

    ApplyMemoryAccessStats(state.counters, memory_stats);
    const std::uint64_t memory_cycles = MemoryAccessDelayCycles(memory_stats);
    const std::uint64_t service_cycles =
        kReferenceUarch.vdm_base_cycles +
        CeilDivide(state.counters.ia_vertices,
                   kReferenceUarch.vdm_vertices_per_batch);
    if (memory_cycles > std::numeric_limits<std::uint64_t>::max() -
                            service_cycles)
      throw std::overflow_error("VDM total cycle overflow");
    const std::uint64_t cycles = service_cycles + memory_cycles;
    state.counters.vdm_cycles = cycles;
    state.counters.tiler_cycles += cycles;
    WaitForCycles(cycles);
    StorePipelineState(pool_, txn.state, state);
    output.write(txn);
  }
}

} // namespace pvrgpu::stub
