// Module：VertexFetch。
// 縮寫：非縮寫（頂點擷取）。
// 功能：將 IEEE-754 vertex attribute 原始位元放入 PCO VTXIN register
// bank。所有 indexed raster cases 依 reference uArch 分段並用真實
// direct-mapped post-transform reuse；buffer/binding bulk data 留在
// MemoryPool，每個 index occurrence 都建立 lane ref，miss 才新增 USC lane。
// FIFO 只傳 handle，完成採 event-driven wait。
#include "geometry/vertex_fetch.h"

#include "common/functional_types.h"

#include <algorithm>
#include <array>
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

inline constexpr std::uint64_t kFloat32Bytes = sizeof(float);

struct VertexInputState {
  std::vector<VertexBufferResource> resources;
  std::vector<VertexAttributeBinding> bindings;
  std::vector<std::vector<std::uint8_t>> buffers;
  std::uint64_t bytes_per_vertex = 0;
};

VertexInputState LoadVertexInputState(const MemoryPool &pool,
                                      const PipelineState &state) {
  if (!HasPoolHandle(state.vertex_buffer_resources) ||
      !HasPoolHandle(state.vertex_attribute_bindings)) {
    throw std::runtime_error(
        "VertexFetch received no vertex buffer/binding tables");
  }
  VertexInputState input;
  input.resources = LoadArray<VertexBufferResource>(
      pool, state.vertex_buffer_resources);
  input.bindings = LoadArray<VertexAttributeBinding>(
      pool, state.vertex_attribute_bindings);
  if (input.resources.empty() || input.bindings.empty())
    throw std::runtime_error("VertexFetch received an empty input layout");

  input.buffers.reserve(input.resources.size());
  for (std::size_t index = 0; index < input.resources.size(); ++index) {
    const VertexBufferResource &resource = input.resources[index];
    if (!HasPoolHandle(resource.data))
      throw std::runtime_error("VertexFetch resource has no VBO payload");
    for (std::size_t prior = 0; prior < index; ++prior) {
      if (input.resources[prior].data.slot == resource.data.slot &&
          input.resources[prior].data.generation == resource.data.generation) {
        throw std::runtime_error(
            "VertexFetch resources must own unique VBO payloads");
      }
    }
    input.buffers.push_back(LoadArray<std::uint8_t>(pool, resource.data));
    if (input.buffers.back().size() != resource.byte_size)
      throw std::runtime_error("VertexFetch VBO byte size mismatch");
  }

  std::array<std::uint8_t, kPcoVertexInputRegisterCount> occupied{};
  for (const VertexAttributeBinding &binding : input.bindings) {
    if (binding.buffer_index >= input.resources.size() ||
        binding.component_type != VertexComponentType::kFloat32 ||
        binding.source_components == 0 || binding.source_components > 4 ||
        binding.destination_components < binding.source_components ||
        binding.destination_components > 4 || binding.normalized != 0 ||
        binding.integer != 0 || binding.instance_divisor != 0) {
      throw std::runtime_error(
          "VertexFetch received an unsupported attribute binding");
    }
    const std::uint64_t destination_end =
        static_cast<std::uint64_t>(binding.destination_register) +
        binding.destination_components;
    if (destination_end > occupied.size())
      throw std::runtime_error(
          "VertexFetch attribute exceeds the PCO VTXIN file");
    for (std::uint32_t component = 0;
         component < binding.destination_components; ++component) {
      std::uint8_t &slot = occupied[binding.destination_register + component];
      if (slot != 0)
        throw std::runtime_error(
            "VertexFetch attribute destination registers overlap");
      slot = 1;
    }
    const std::uint64_t element_bytes =
        static_cast<std::uint64_t>(binding.source_components) * kFloat32Bytes;
    if (binding.stride_bytes < element_bytes)
      throw std::runtime_error("VertexFetch attribute stride is too small");
    const VertexBufferResource &resource =
        input.resources[binding.buffer_index];
    if (static_cast<std::uint64_t>(binding.offset_bytes) + element_bytes >
        resource.byte_size) {
      throw std::runtime_error(
          "VertexFetch first attribute element exceeds its VBO");
    }
    if (element_bytes >
        std::numeric_limits<std::uint64_t>::max() - input.bytes_per_vertex) {
      throw std::overflow_error("VertexFetch byte counter overflow");
    }
    input.bytes_per_vertex += element_bytes;
  }
  return input;
}

VertexLane MakeLane(std::uint32_t vertex_index,
                    const VertexInputState &input) {
  VertexLane lane;
  for (const VertexAttributeBinding &binding : input.bindings) {
    const std::uint64_t stride_offset =
        static_cast<std::uint64_t>(vertex_index) * binding.stride_bytes;
    const std::uint64_t source_offset =
        stride_offset + binding.offset_bytes;
    const std::uint64_t source_bytes =
        static_cast<std::uint64_t>(binding.source_components) * kFloat32Bytes;
    const std::vector<std::uint8_t> &buffer =
        input.buffers[binding.buffer_index];
    if (source_offset > buffer.size() ||
        source_bytes > buffer.size() - source_offset) {
      throw std::runtime_error(
          "VertexFetch resolved attribute exceeds its VBO");
    }
    for (std::uint32_t component = 0;
         component < binding.destination_components; ++component) {
      std::uint32_t value = component == 3 ? FloatBits(1.0F) : 0U;
      if (component < binding.source_components) {
        const std::size_t component_offset = static_cast<std::size_t>(
            source_offset + component * kFloat32Bytes);
        std::memcpy(&value, buffer.data() + component_offset, sizeof(value));
      }
      lane.vertex_input[binding.destination_register + component] = value;
    }
  }
  return lane;
}

struct CacheEntry {
  std::uint32_t vertex_index = 0;
  std::uint32_t lane_index = 0;
  bool valid = false;
};

} // namespace

VertexFetch::VertexFetch(sc_core::sc_module_name name, MemoryPool &pool)
    : sc_module(name), pool_(pool) {
  SC_THREAD(Run);
}

void VertexFetch::Run() {
  while (true) {
    const PipelineTxn txn = input.read();
    PipelineState state = LoadPipelineState(pool_, txn.state);

    RequireStage(state.stage, PipelineStage::kVdmComplete, name());
    if (HasPoolHandle(state.vertex_lanes) ||
        HasPoolHandle(state.vertex_lane_refs)) {
      throw std::runtime_error("VertexFetch received pre-existing output");
    }
    const VertexInputState vertex_input =
        LoadVertexInputState(pool_, state);

    std::vector<VertexLane> lanes;
    if (IsFillSolidFamily(state.functional_case) ||
        IsTextureFamily(state.functional_case)) {
      if (state.draw.vertex_count != 4 ||
          state.draw.index_format != IndexFormat::kNone ||
          HasPoolHandle(state.vertex_indices)) {
        throw std::runtime_error(
            "VertexFetch Fill.Solid requires four non-indexed vertices");
      }
      lanes.reserve(state.draw.vertex_count);
      for (std::uint32_t vertex = 0; vertex < state.draw.vertex_count;
           ++vertex) {
        const std::uint64_t resolved =
            static_cast<std::uint64_t>(state.draw.first_vertex) + vertex;
        if (resolved > std::numeric_limits<std::uint32_t>::max())
          throw std::overflow_error("VertexFetch vertex index exceeds uint32");
        lanes.push_back(
            MakeLane(static_cast<std::uint32_t>(resolved), vertex_input));
      }
    } else if (IsIndexedTriangleRasterCase(state.functional_case)) {
      if (state.draw.topology != PrimitiveTopology::kTriangleList ||
          state.draw.index_format != IndexFormat::kUint16 ||
          !HasPoolHandle(state.vertex_indices)) {
        throw std::runtime_error(
            "VertexFetch indexed raster state is invalid");
      }
      const std::vector<std::uint16_t> indices =
          LoadArray<std::uint16_t>(pool_, state.vertex_indices);
      const std::uint64_t index_end =
          static_cast<std::uint64_t>(state.draw.first_index) +
          state.draw.index_count;
      if (index_end > indices.size() || state.draw.index_count == 0 ||
          state.draw.index_count % 3 != 0) {
        throw std::runtime_error(
            "VertexFetch indexed draw range is invalid");
      }
      if (kReferenceUarch.index_segment_max_indices == 0 ||
          kReferenceUarch.index_segment_max_indices % 3 != 0 ||
          kReferenceUarch.post_transform_cache_slots == 0) {
        throw std::runtime_error(
            "VertexFetch indexed reference-uArch configuration is invalid");
      }

      std::vector<VertexLaneRef> lane_refs;
      lane_refs.reserve(state.draw.index_count);
      std::size_t occurrence = state.draw.first_index;
      while (occurrence < index_end) {
        const std::size_t segment_count = std::min<std::size_t>(
            kReferenceUarch.index_segment_max_indices, index_end - occurrence);
        if (segment_count == 0 || segment_count % 3 != 0)
          throw std::runtime_error(
              "VertexFetch index segment split a triangle");
        std::vector<CacheEntry> cache(
            kReferenceUarch.post_transform_cache_slots);
        const std::size_t segment_end = occurrence + segment_count;
        for (; occurrence < segment_end; ++occurrence) {
          const std::int64_t resolved =
              static_cast<std::int64_t>(indices[occurrence]) +
              state.draw.base_vertex;
          if (resolved < 0 ||
              static_cast<std::uint64_t>(resolved) >
                  std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error(
                "VertexFetch resolved index exceeds the modeled range");
          }
          const std::uint32_t vertex_index =
              static_cast<std::uint32_t>(resolved);
          CacheEntry &entry = cache[vertex_index % cache.size()];
          if (!entry.valid || entry.vertex_index != vertex_index) {
            if (lanes.size() >= std::numeric_limits<std::uint32_t>::max())
              throw std::overflow_error(
                  "VertexFetch lane index exceeds uint32_t");
            entry.vertex_index = vertex_index;
            entry.lane_index = static_cast<std::uint32_t>(lanes.size());
            entry.valid = true;
            lanes.push_back(MakeLane(vertex_index, vertex_input));
          }
          lane_refs.push_back({entry.lane_index, vertex_index});
        }
      }
      if (lane_refs.size() != state.draw.index_count)
        throw std::runtime_error(
            "VertexFetch did not emit one lane ref per index occurrence");
      state.vertex_lane_refs = StoreNewArray(pool_, lane_refs);
    } else {
      throw std::runtime_error("VertexFetch received an unsupported case");
    }
    if (lanes.empty())
      throw std::runtime_error("VertexFetch produced no shader lanes");
    state.vertex_lanes = StoreNewArray(pool_, lanes);
    state.counters.vs_invocations = lanes.size();
    if (vertex_input.bindings.size() != 0 &&
        lanes.size() > std::numeric_limits<std::uint64_t>::max() /
                           vertex_input.bindings.size()) {
      throw std::overflow_error("VertexFetch attribute counter overflow");
    }
    state.counters.vertex_attribute_fetches =
        lanes.size() * vertex_input.bindings.size();
    if (vertex_input.bytes_per_vertex != 0 &&
        lanes.size() > std::numeric_limits<std::uint64_t>::max() /
                           vertex_input.bytes_per_vertex) {
      throw std::overflow_error("VertexFetch byte counter overflow");
    }
    state.counters.vertex_attribute_bytes =
        lanes.size() * vertex_input.bytes_per_vertex;
    state.stage = PipelineStage::kVertexFetched;

    const std::uint64_t cycles =
        kReferenceUarch.vertex_fetch_base_cycles +
        CeilDivide(state.counters.vertex_attribute_bytes,
                   kReferenceUarch.vertex_fetch_bytes_per_batch);
    state.counters.vertex_fetch_cycles = cycles;
    state.counters.tiler_cycles += cycles;
    WaitForCycles(cycles);
    StorePipelineState(pool_, txn.state, state);
    output.write(txn);
  }
}

} // namespace pvrgpu::stub
