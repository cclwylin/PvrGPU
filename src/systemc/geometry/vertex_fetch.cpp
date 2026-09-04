// Module：VertexFetch。
// 縮寫：非縮寫（頂點擷取）。
// 功能：將 IEEE-754 vertex attribute 原始位元放入 PCO VTXIN register
// bank。所有 indexed raster cases 依 reference uArch 分段並用真實
// direct-mapped post-transform reuse；buffer/binding bulk data 留在
// MemoryPool，每個 index occurrence 都建立 lane ref，miss 才新增 USC lane。
// FIFO 只傳 handle，完成採 event-driven wait。
#include "geometry/vertex_fetch.h"

#include "memory/gpu_memory_system.h"

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

// inline constexpr std::uint64_t kFloat32Bytes = sizeof(float);

struct VertexInputState {
  std::vector<VertexBufferResource> resources;
  std::vector<VertexAttributeBinding> bindings;
  std::vector<std::vector<std::uint8_t>> buffers;
  std::uint64_t bytes_per_vertex = 0;
};

VertexInputState LoadVertexInputState(const MemoryPool &pool,
                                      const PipelineState &state,
                                      GpuMemorySystem *memory,
                                      MemoryAccessStats *memory_stats) {
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
    if (memory) {
      if (HasPoolHandle(resource.data) || resource.gpu_address == 0 ||
          resource.byte_size == 0 ||
          !memory->backing().Contains(resource.gpu_address,
                                      resource.byte_size)) {
        throw std::runtime_error(
            "VertexFetch resource has no DRAM-backed VBO");
      }
    } else if (!HasPoolHandle(resource.data)) {
      throw std::runtime_error("VertexFetch resource has no VBO payload");
    }
    for (std::size_t prior = 0; prior < index; ++prior) {
      const bool aliases = memory
                               ? input.resources[prior].gpu_address ==
                                     resource.gpu_address
                               : input.resources[prior].data.slot ==
                                         resource.data.slot &&
                                     input.resources[prior].data.generation ==
                                         resource.data.generation;
      if (aliases) {
        throw std::runtime_error(
            "VertexFetch resources must own unique VBO payloads");
      }
    }
    if (memory) {
      MemoryReadResult read = memory->Read(resource.gpu_address,
                                           resource.byte_size,
                                           MemoryClient::kVertexFetch);
      if (memory_stats)
        *memory_stats += read.stats;
      input.buffers.push_back(std::move(read.data));
    } else {
      input.buffers.push_back(LoadArray<std::uint8_t>(pool, resource.data));
    }
    if (input.buffers.back().size() != resource.byte_size)
      throw std::runtime_error("VertexFetch VBO byte size mismatch");
  }

  std::array<std::uint8_t, kPcoVertexInputRegisterCount> occupied{};
  for (const VertexAttributeBinding &binding : input.bindings) {
    if (binding.buffer_index >= input.resources.size() ||
        binding.source_components == 0 || binding.source_components > 4 ||
        binding.destination_components < binding.source_components ||
        binding.destination_components > 4) {
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
        static_cast<std::uint64_t>(binding.source_components) *
        GetComponentTypeBytes(binding.component_type);
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

float ReadComponentAsFloat(const std::vector<std::uint8_t>& buffer, std::size_t offset, VertexComponentType type, bool normalized);
std::uint32_t ReadComponentAsInteger(const std::vector<std::uint8_t>& buffer, std::size_t offset, VertexComponentType type);

VertexLane MakeLane(std::uint32_t vertex_index,
                    const VertexInputState &input) {
  VertexLane lane;
  for (const VertexAttributeBinding &binding : input.bindings) {
    std::uint32_t active_index = vertex_index;
    if (binding.instance_divisor != 0) {
      active_index = 0;
    }
    const std::uint64_t stride_offset =
        static_cast<std::uint64_t>(active_index) * binding.stride_bytes;
    const std::uint64_t source_offset =
        stride_offset + binding.offset_bytes;
    const std::size_t component_bytes = GetComponentTypeBytes(binding.component_type);
    const std::uint64_t source_bytes =
        static_cast<std::uint64_t>(binding.source_components) * component_bytes;
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
            source_offset + component * component_bytes);
        if (binding.integer) {
          value = ReadComponentAsInteger(buffer, component_offset, binding.component_type);
        } else {
          float fval = ReadComponentAsFloat(buffer, component_offset, binding.component_type, binding.normalized != 0);
          value = FloatBits(fval);
        }
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

std::vector<std::uint32_t> ExpandTopology(const std::vector<std::uint32_t>& indices,
                                          PrimitiveTopology topology,
                                          bool restart_enable,
                                          std::uint32_t restart_index) {
  std::vector<std::uint32_t> expanded;

  std::vector<std::vector<std::uint32_t>> segments;
  std::vector<std::uint32_t> current_segment;
  for (std::uint32_t idx : indices) {
    if (restart_enable && idx == restart_index) {
      if (!current_segment.empty()) {
        segments.push_back(current_segment);
        current_segment.clear();
      }
    } else {
      current_segment.push_back(idx);
    }
  }
  if (!current_segment.empty()) {
    segments.push_back(current_segment);
  }

  for (const auto& seg : segments) {
    std::size_t n = seg.size();
    if (topology == PrimitiveTopology::kTriangleList) {
      for (std::size_t i = 0; i + 2 < n; i += 3) {
        expanded.push_back(seg[i]);
        expanded.push_back(seg[i + 1]);
        expanded.push_back(seg[i + 2]);
      }
    } else if (topology == PrimitiveTopology::kTriangleStrip) {
      for (std::size_t i = 0; i + 2 < n; ++i) {
        if (i % 2 == 0) {
          expanded.push_back(seg[i]);
          expanded.push_back(seg[i + 1]);
          expanded.push_back(seg[i + 2]);
        } else {
          expanded.push_back(seg[i + 1]);
          expanded.push_back(seg[i]);
          expanded.push_back(seg[i + 2]);
        }
      }
    } else if (topology == PrimitiveTopology::kTriangleFan) {
      for (std::size_t i = 1; i + 1 < n; ++i) {
        expanded.push_back(seg[0]);
        expanded.push_back(seg[i]);
        expanded.push_back(seg[i + 1]);
      }
    } else if (topology == PrimitiveTopology::kLines) {
      for (std::size_t i = 0; i + 1 < n; i += 2) {
        expanded.push_back(seg[i]);
        expanded.push_back(seg[i + 1]);
        expanded.push_back(seg[i + 1]);
      }
    } else if (topology == PrimitiveTopology::kLineStrip) {
      for (std::size_t i = 0; i + 1 < n; ++i) {
        expanded.push_back(seg[i]);
        expanded.push_back(seg[i + 1]);
        expanded.push_back(seg[i + 1]);
      }
    } else if (topology == PrimitiveTopology::kLineLoop) {
      if (n >= 2) {
        for (std::size_t i = 0; i + 1 < n; ++i) {
          expanded.push_back(seg[i]);
          expanded.push_back(seg[i + 1]);
          expanded.push_back(seg[i + 1]);
        }
        expanded.push_back(seg[n - 1]);
        expanded.push_back(seg[0]);
        expanded.push_back(seg[0]);
      }
    } else if (topology == PrimitiveTopology::kPoints) {
      for (std::size_t i = 0; i < n; ++i) {
        expanded.push_back(seg[i]);
        expanded.push_back(seg[i]);
        expanded.push_back(seg[i]);
      }
    }
  }
  return expanded;
}

float ReadComponentAsFloat(const std::vector<std::uint8_t>& buffer, std::size_t offset, VertexComponentType type, bool normalized) {
  if (type == VertexComponentType::kFloat32) {
    float val;
    std::memcpy(&val, buffer.data() + offset, sizeof(val));
    return val;
  }

  if (type == VertexComponentType::kUint8) {
    std::uint8_t val = buffer[offset];
    if (normalized) return static_cast<float>(val) / 255.0f;
    return static_cast<float>(val);
  }
  if (type == VertexComponentType::kInt8) {
    std::int8_t val;
    std::memcpy(&val, buffer.data() + offset, sizeof(val));
    if (normalized) {
      return std::max(static_cast<float>(val) / 127.0f, -1.0f);
    }
    return static_cast<float>(val);
  }
  if (type == VertexComponentType::kUint16) {
    std::uint16_t val;
    std::memcpy(&val, buffer.data() + offset, sizeof(val));
    if (normalized) return static_cast<float>(val) / 65535.0f;
    return static_cast<float>(val);
  }
  if (type == VertexComponentType::kInt16) {
    std::int16_t val;
    std::memcpy(&val, buffer.data() + offset, sizeof(val));
    if (normalized) {
      return std::max(static_cast<float>(val) / 32767.0f, -1.0f);
    }
    return static_cast<float>(val);
  }
  if (type == VertexComponentType::kUint32) {
    std::uint32_t val;
    std::memcpy(&val, buffer.data() + offset, sizeof(val));
    if (normalized) return static_cast<float>(val) / 4294967295.0f;
    return static_cast<float>(val);
  }
  if (type == VertexComponentType::kInt32) {
    std::int32_t val;
    std::memcpy(&val, buffer.data() + offset, sizeof(val));
    if (normalized) {
      return std::max(static_cast<float>(val) / 2147483647.0f, -1.0f);
    }
    return static_cast<float>(val);
  }
  if (type == VertexComponentType::kHalfFloat) {
    std::uint16_t raw;
    std::memcpy(&raw, buffer.data() + offset, sizeof(raw));
    std::uint32_t sign = (raw & 0x8000) >> 15;
    std::uint32_t exponent = (raw & 0x7C00) >> 10;
    std::uint32_t fraction = raw & 0x03FF;

    std::uint32_t result;
    if (exponent == 0) {
      if (fraction == 0) {
        result = sign << 31;
      } else {
        while ((fraction & 0x0400) == 0) {
          fraction <<= 1;
          exponent--;
        }
        exponent++;
        fraction &= ~0x0400;
        result = (sign << 31) | ((exponent - 15 + 127) << 23) | (fraction << 13);
      }
    } else if (exponent == 31) {
      result = (sign << 31) | (0xFF << 23) | (fraction << 13);
    } else {
      result = (sign << 31) | ((exponent - 15 + 127) << 23) | (fraction << 13);
    }
    float val;
    std::memcpy(&val, &result, sizeof(val));
    return val;
  }
  return 0.0f;
}

std::uint32_t ReadComponentAsInteger(const std::vector<std::uint8_t>& buffer, std::size_t offset, VertexComponentType type) {
  if (type == VertexComponentType::kUint8) {
    return static_cast<std::uint32_t>(buffer[offset]);
  }
  if (type == VertexComponentType::kInt8) {
    std::int8_t val;
    std::memcpy(&val, buffer.data() + offset, sizeof(val));
    return static_cast<std::uint32_t>(static_cast<std::int32_t>(val));
  }
  if (type == VertexComponentType::kUint16) {
    std::uint16_t val;
    std::memcpy(&val, buffer.data() + offset, sizeof(val));
    return static_cast<std::uint32_t>(val);
  }
  if (type == VertexComponentType::kInt16) {
    std::int16_t val;
    std::memcpy(&val, buffer.data() + offset, sizeof(val));
    return static_cast<std::uint32_t>(static_cast<std::int32_t>(val));
  }
  if (type == VertexComponentType::kUint32 || type == VertexComponentType::kInt32) {
    std::uint32_t val;
    std::memcpy(&val, buffer.data() + offset, sizeof(val));
    return val;
  }
  if (type == VertexComponentType::kFloat32) {
    float val;
    std::memcpy(&val, buffer.data() + offset, sizeof(val));
    return static_cast<std::uint32_t>(static_cast<std::int32_t>(val));
  }
  return 0;
}

} // namespace

VertexFetch::VertexFetch(sc_core::sc_module_name name, MemoryPool &pool,
                         GpuMemorySystem *memory)
    : sc_module(name), pool_(pool), memory_(memory) {
  SC_THREAD(Run);
}

void VertexFetch::Run() {
  while (true) {
    const PipelineTxn txn = input.read();
    PipelineState state = LoadPipelineState(pool_, txn.state);

    RequireStage(state.stage, PipelineStage::kVdmComplete, name());
    if (memory_ && state.memory_mode != memory_->mode())
      throw std::runtime_error("VertexFetch memory mode mismatch");
    if (HasPoolHandle(state.vertex_lanes) ||
        HasPoolHandle(state.vertex_lane_refs)) {
      throw std::runtime_error("VertexFetch received pre-existing output");
    }
    MemoryAccessStats memory_stats;
    const VertexInputState vertex_input =
        LoadVertexInputState(pool_, state, memory_, &memory_stats);

    std::vector<VertexLane> lanes;
    const bool driver_pco_triangles =
        IsDriverPcoTrianglesCase(state.functional_case);
    if (IsFillSolidFamily(state.functional_case) ||
        IsTextureFamily(state.functional_case) || driver_pco_triangles) {
      const std::uint32_t expected_vertex_count =
          state.functional_case == FunctionalCase::kDriverTexturedTriangles
              ? 6U
              : 4U;
      const bool direct_geometry_invalid =
          driver_pco_triangles
              ? state.draw.topology != PrimitiveTopology::kTriangleList ||
                    state.draw.vertex_count == 0 ||
                    state.draw.vertex_count % 3 != 0 ||
                    state.primitive_restart_enable != 0
              : state.draw.vertex_count != expected_vertex_count;
      if (direct_geometry_invalid || state.draw.first_index != 0 ||
          state.draw.index_count != 0 || state.draw.base_vertex != 0 ||
          state.draw.index_format != IndexFormat::kNone ||
          HasPoolHandle(state.vertex_indices) ||
          state.index_buffer_gpu_address != 0 ||
          state.index_buffer_bytes != 0) {
        throw std::runtime_error(
            "VertexFetch direct raster draw has an invalid vertex range");
      }
      std::vector<VertexLaneRef> lane_refs;
      if (driver_pco_triangles)
        lane_refs.reserve(state.draw.vertex_count);
      lanes.reserve(state.draw.vertex_count);
      for (std::uint32_t vertex = 0; vertex < state.draw.vertex_count;
           ++vertex) {
        const std::uint64_t resolved =
            static_cast<std::uint64_t>(state.draw.first_vertex) + vertex;
        if (resolved > std::numeric_limits<std::uint32_t>::max())
          throw std::overflow_error("VertexFetch vertex index exceeds uint32");
        const std::uint32_t vertex_index =
            static_cast<std::uint32_t>(resolved);
        const std::uint32_t lane_index =
            static_cast<std::uint32_t>(lanes.size());
        lanes.push_back(MakeLane(vertex_index, vertex_input));
        if (driver_pco_triangles)
          lane_refs.push_back({lane_index, vertex_index});
      }
      if (driver_pco_triangles)
        state.vertex_lane_refs = StoreNewArray(pool_, lane_refs);
    } else if (IsIndexedTriangleRasterCase(state.functional_case)) {
      if ((state.draw.topology != PrimitiveTopology::kTriangleList &&
           state.draw.topology != PrimitiveTopology::kTriangleStrip &&
           state.draw.topology != PrimitiveTopology::kPoints &&
           state.draw.topology != PrimitiveTopology::kLines &&
           state.draw.topology != PrimitiveTopology::kLineStrip &&
           state.draw.topology != PrimitiveTopology::kLineLoop &&
           state.draw.topology != PrimitiveTopology::kTriangleFan) ||
          state.draw.index_format == IndexFormat::kNone ||
          !HasPoolHandle(state.vertex_indices)) {
        throw std::runtime_error(
            "VertexFetch indexed raster state is invalid");
      }
      // Snapshot the topology as submitted before any expansion below
      // rewrites state.draw.topology to kTriangleList. ClipCull uses
      // this to widen lines/points into real geometry instead of the
      // degenerate zero-area triangle ExpandTopology encodes them as.
      state.source_topology = state.draw.topology;

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
        throw std::runtime_error("VertexFetch received an unsupported index format");
      }

      std::vector<std::uint16_t> expanded_indices_16;
      bool needs_expansion = (state.draw.topology != PrimitiveTopology::kTriangleList ||
                              state.primitive_restart_enable != 0 ||
                              state.draw.index_format != IndexFormat::kUint16);

      if (needs_expansion) {
        // Expand topologies and restarts into a standard TriangleList
        std::vector<std::uint32_t> expanded_indices = ExpandTopology(
            indices, state.draw.topology, state.primitive_restart_enable != 0,
            state.primitive_restart_index);

        // Reallocate and update state.vertex_indices
        expanded_indices_16.reserve(expanded_indices.size());
        for (std::uint32_t val : expanded_indices) {
          expanded_indices_16.push_back(static_cast<std::uint16_t>(val));
        }
        pool_.Release(state.vertex_indices);
        state.vertex_indices = StoreNewArray(pool_, expanded_indices_16);

        state.draw.topology = PrimitiveTopology::kTriangleList;
        state.draw.index_count = expanded_indices_16.size();
        state.draw.first_index = 0;
      } else {
        // Direct conversion without pool allocation/release to maintain exact telemetry
        expanded_indices_16.reserve(indices.size());
        for (std::uint32_t val : indices) {
          expanded_indices_16.push_back(static_cast<std::uint16_t>(val));
        }
      }

      const std::uint64_t index_end = state.draw.index_count;
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
              static_cast<std::int64_t>(expanded_indices_16[occurrence]) +
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

    ApplyMemoryAccessStats(state.counters, memory_stats);
    const std::uint64_t memory_cycles = MemoryAccessDelayCycles(memory_stats);
    const std::uint64_t service_cycles =
        kReferenceUarch.vertex_fetch_base_cycles +
        CeilDivide(state.counters.vertex_attribute_bytes,
                   kReferenceUarch.vertex_fetch_bytes_per_batch);
    if (memory_cycles > std::numeric_limits<std::uint64_t>::max() -
                            service_cycles)
      throw std::overflow_error("VertexFetch total cycle overflow");
    const std::uint64_t cycles = service_cycles + memory_cycles;
    state.counters.vertex_fetch_cycles = cycles;
    state.counters.tiler_cycles += cycles;
    WaitForCycles(cycles);
    StorePipelineState(pool_, txn.state, state);
    output.write(txn);
  }
}

} // namespace pvrgpu::stub
