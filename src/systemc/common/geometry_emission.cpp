// SPDX-License-Identifier: MIT
/*
 * Input primitive decomposition is adapted from Mesa draw_decompose_tmp.h.
 * Copyright 2008 VMware, Inc.
 * Copyright (C) 2010 LunarG Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include "common/geometry_emission.h"

#include <array>
#include <cstring>
#include <limits>

namespace pvrgpu::stub {
namespace {

bool ValidTopology(GeometryOutputTopology topology) {
  return topology == GeometryOutputTopology::kPoints ||
         topology == GeometryOutputTopology::kLineStrip ||
         topology == GeometryOutputTopology::kTriangleStrip;
}

bool ValidTopology(GeometryInputTopology topology) {
  return topology >= GeometryInputTopology::kPoints &&
         topology <= GeometryInputTopology::kTriangleStripAdjacency;
}

// Mesa draw_decompose_tmp.h (pin da14d65e...) is the algorithmic reference.
// GLES uses last-vertex provoking semantics, so strip parity swaps the first
// two principal vertices, never the latest/provoking vertex. Adjacency retains
// the matching edge-neighbour permutation; it is not discarded as raster data.
template <typename Sink>
void VisitInputSegment(GeometryInputTopology topology,
                       const std::uint32_t *indices, std::size_t count,
                       Sink sink) {
  const auto point = [&](std::size_t a) {
    sink(std::array<std::uint32_t, 6>{indices[a]}, std::uint8_t{1});
  };
  const auto line = [&](std::size_t a, std::size_t b) {
    sink(std::array<std::uint32_t, 6>{indices[a], indices[b]}, std::uint8_t{2});
  };
  const auto triangle = [&](std::size_t a, std::size_t b, std::size_t c) {
    sink(std::array<std::uint32_t, 6>{indices[a], indices[b], indices[c]},
         std::uint8_t{3});
  };
  switch (topology) {
    case GeometryInputTopology::kPoints:
      for (std::size_t i = 0; i < count; ++i) point(i);
      break;
    case GeometryInputTopology::kLines:
      for (std::size_t i = 0; i + 1 < count; i += 2) line(i, i + 1);
      break;
    case GeometryInputTopology::kLineStrip:
    case GeometryInputTopology::kLineLoop:
      for (std::size_t i = 0; i + 1 < count; ++i) line(i, i + 1);
      if (topology == GeometryInputTopology::kLineLoop && count >= 2)
        line(count - 1, 0);
      break;
    case GeometryInputTopology::kTriangles:
      for (std::size_t i = 0; i + 2 < count; i += 3)
        triangle(i, i + 1, i + 2);
      break;
    case GeometryInputTopology::kTriangleStrip:
      for (std::size_t i = 0; i + 2 < count; ++i) {
        if (i & 1) triangle(i + 1, i, i + 2);
        else triangle(i, i + 1, i + 2);
      }
      break;
    case GeometryInputTopology::kTriangleFan:
      for (std::size_t i = 1; i + 1 < count; ++i)
        triangle(0, i, i + 1);
      break;
    case GeometryInputTopology::kLinesAdjacency:
    case GeometryInputTopology::kLineStripAdjacency: {
      const std::size_t step =
          topology == GeometryInputTopology::kLinesAdjacency ? 4 : 1;
      for (std::size_t i = 0; i + 3 < count; i += step)
        sink(std::array<std::uint32_t, 6>{indices[i], indices[i + 1],
                                         indices[i + 2], indices[i + 3]},
             std::uint8_t{4});
      break;
    }
    case GeometryInputTopology::kTrianglesAdjacency:
      for (std::size_t i = 0; i + 5 < count; i += 6)
        sink(std::array<std::uint32_t, 6>{
                 indices[i], indices[i + 1], indices[i + 2], indices[i + 3],
                 indices[i + 4], indices[i + 5]}, std::uint8_t{6});
      break;
    case GeometryInputTopology::kTriangleStripAdjacency:
      for (std::size_t i = 0; i + 5 < count; i += 2) {
        const std::size_t adjacent_ab = i ? i - 2 : 1;
        const std::size_t adjacent_bc = i + 7 < count ? i + 6 : i + 5;
        const std::size_t adjacent_ca = i + 3;
        if (i & 2) {
          sink(std::array<std::uint32_t, 6>{
                   indices[i + 2], indices[adjacent_ab], indices[i],
                   indices[adjacent_ca], indices[i + 4], indices[adjacent_bc]},
               std::uint8_t{6});
        } else {
          sink(std::array<std::uint32_t, 6>{
                   indices[i], indices[adjacent_ab], indices[i + 2],
                   indices[adjacent_bc], indices[i + 4], indices[adjacent_ca]},
               std::uint8_t{6});
        }
      }
      break;
  }
}

template <typename Sink>
void VisitInput(GeometryInputTopology topology, const std::uint32_t *indices,
                std::size_t count, bool restart, std::uint32_t restart_index,
                Sink sink) {
  std::size_t first = 0;
  for (std::size_t i = 0; i < count; ++i) {
    if (restart && indices[i] == restart_index) {
      if (i != first)
        VisitInputSegment(topology, indices + first, i - first, sink);
      first = i + 1;
    }
  }
  if (first != count)
    VisitInputSegment(topology, indices + first, count - first, sink);
}

}  // namespace

const char *GeometryEmissionStatusName(GeometryEmissionStatus status) {
  switch (status) {
    case GeometryEmissionStatus::kSuccess: return "success";
    case GeometryEmissionStatus::kEmitSuppressed: return "emit_suppressed";
    case GeometryEmissionStatus::kInvalidOutputDwordCount:
      return "invalid_output_dword_count";
    case GeometryEmissionStatus::kInvalidStorage: return "invalid_storage";
    case GeometryEmissionStatus::kInvalidWrittenMask: return "invalid_written_mask";
    case GeometryEmissionStatus::kInvalidTopology: return "invalid_topology";
    case GeometryEmissionStatus::kInvalidRange: return "invalid_range";
    case GeometryEmissionStatus::kInsufficientCapacity:
      return "insufficient_capacity";
    case GeometryEmissionStatus::kFinished: return "finished";
  }
  return "unknown_geometry_emission_status";
}

GeometryEmissionBuffer::GeometryEmissionBuffer(
    GeometryEmissionStorage storage, std::uint32_t max_vertices,
    std::uint32_t output_dword_count)
    : storage_(storage), max_vertices_(max_vertices),
      output_dword_count_(output_dword_count) {}

GeometryEmissionStatus GeometryEmissionBuffer::ValidateStorage() const {
  if (!output_dword_count_ || output_dword_count_ > 64)
    return GeometryEmissionStatus::kInvalidOutputDwordCount;
  if ((!storage_.snapshot_dwords && storage_.snapshot_dword_capacity) ||
      (!storage_.written_masks && storage_.written_mask_capacity) ||
      (!storage_.strips && storage_.strip_capacity))
    return GeometryEmissionStatus::kInvalidStorage;
  return GeometryEmissionStatus::kSuccess;
}

GeometryEmissionStatus GeometryEmissionBuffer::Emit(
    const std::uint32_t *output_dwords, std::size_t output_dword_count,
    std::uint64_t written_mask) {
  const auto validation = ValidateStorage();
  if (validation != GeometryEmissionStatus::kSuccess) return validation;
  if (finished_) return GeometryEmissionStatus::kFinished;
  if (emitted_vertices_ == max_vertices_)
    return GeometryEmissionStatus::kEmitSuppressed;
  if (output_dword_count != output_dword_count_)
    return GeometryEmissionStatus::kInvalidOutputDwordCount;
  if (!output_dwords) return GeometryEmissionStatus::kInvalidStorage;
  if (output_dword_count_ < 64 && (written_mask >> output_dword_count_))
    return GeometryEmissionStatus::kInvalidWrittenMask;

  const std::uint64_t first =
      std::uint64_t{emitted_vertices_} * output_dword_count_;
  const std::uint64_t end = first + output_dword_count_;
  if (end > storage_.snapshot_dword_capacity ||
      emitted_vertices_ >= storage_.written_mask_capacity ||
      (!open_strip_vertices_ && strip_count_ >= storage_.strip_capacity))
    return GeometryEmissionStatus::kInsufficientCapacity;

  // Reserve the strip slot before its first snapshot. EndPrimitive can never
  // fail after an accepted Emit due to a late range-storage allocation.
  std::memmove(storage_.snapshot_dwords + static_cast<std::size_t>(first),
               output_dwords, output_dword_count_ * sizeof(std::uint32_t));
  storage_.written_masks[emitted_vertices_] = written_mask;
  ++emitted_vertices_;
  ++open_strip_vertices_;
  return GeometryEmissionStatus::kSuccess;
}

GeometryEmissionStatus GeometryEmissionBuffer::EndPrimitive() {
  const auto validation = ValidateStorage();
  if (validation != GeometryEmissionStatus::kSuccess) return validation;
  if (finished_) return GeometryEmissionStatus::kFinished;
  if (!open_strip_vertices_) return GeometryEmissionStatus::kSuccess;
  if (strip_count_ >= storage_.strip_capacity)
    return GeometryEmissionStatus::kInsufficientCapacity;
  storage_.strips[strip_count_++] = {
      emitted_vertices_ - open_strip_vertices_, open_strip_vertices_};
  open_strip_vertices_ = 0;
  return GeometryEmissionStatus::kSuccess;
}

GeometryEmissionStatus GeometryEmissionBuffer::Finish() {
  if (finished_) return GeometryEmissionStatus::kSuccess;
  const auto status = EndPrimitive();
  if (status == GeometryEmissionStatus::kSuccess) finished_ = true;
  return status;
}

std::uint32_t GeometryEmissionBuffer::emitted_vertices() const {
  return emitted_vertices_;
}
std::size_t GeometryEmissionBuffer::strip_count() const { return strip_count_; }
bool GeometryEmissionBuffer::finished() const { return finished_; }

GeometryEmissionStatus ExpandGeometryPrimitives(
    GeometryOutputTopology topology, const GeometryStripRange *strips,
    std::size_t strip_count, std::uint32_t snapshot_count,
    GeometryPrimitiveRef *output, std::size_t output_capacity,
    std::size_t &written_count) {
  if (!ValidTopology(topology)) return GeometryEmissionStatus::kInvalidTopology;
  if ((!strips && strip_count) || (!output && output_capacity))
    return GeometryEmissionStatus::kInvalidStorage;
  if (strip_count > snapshot_count)
    return GeometryEmissionStatus::kInvalidRange;
  std::uint64_t next = 0;
  std::uint64_t required = 0;
  for (std::size_t j = 0; j < strip_count; ++j) {
    const auto &strip = strips[j];
    if (!strip.vertex_count || strip.first_vertex != next ||
        std::uint64_t{strip.first_vertex} + strip.vertex_count > snapshot_count)
      return GeometryEmissionStatus::kInvalidRange;
    next += strip.vertex_count;
    if (topology == GeometryOutputTopology::kPoints)
      required += strip.vertex_count;
    else if (topology == GeometryOutputTopology::kLineStrip)
      required += strip.vertex_count > 1 ? strip.vertex_count - 1 : 0;
    else
      required += strip.vertex_count > 2 ? strip.vertex_count - 2 : 0;
  }
  if (next != snapshot_count) return GeometryEmissionStatus::kInvalidRange;
  if (required > output_capacity)
    return GeometryEmissionStatus::kInsufficientCapacity;

  std::size_t cursor = 0;
  for (std::size_t j = 0; j < strip_count; ++j) {
    const auto &strip = strips[j];
    for (std::uint32_t i = 0; i < strip.vertex_count; ++i) {
      const std::uint32_t a = strip.first_vertex + i;
      if (topology == GeometryOutputTopology::kPoints) {
        output[cursor++] = {{a, a, a}, 1, 0, {}};
      } else if (topology == GeometryOutputTopology::kLineStrip && i >= 1) {
        output[cursor++] = {{a - 1, a, a}, 2, 1, {}};
      } else if (topology == GeometryOutputTopology::kTriangleStrip && i >= 2) {
        output[cursor++] = i & 1
            ? GeometryPrimitiveRef{{a - 1, a - 2, a}, 3, 2, {}}
            : GeometryPrimitiveRef{{a - 2, a - 1, a}, 3, 2, {}};
      }
    }
  }
  written_count = cursor;
  return GeometryEmissionStatus::kSuccess;
}

GeometryEmissionStatus AssembleGeometryInputPrimitives(
    GeometryInputTopology topology, const std::uint32_t *vertex_indices,
    std::size_t vertex_count, bool primitive_restart,
    std::uint32_t restart_index, std::uint32_t instance_id,
    GeometryInputPrimitive *output, std::size_t output_capacity,
    std::size_t &written_count) {
  if (!ValidTopology(topology)) return GeometryEmissionStatus::kInvalidTopology;
  if ((!vertex_indices && vertex_count) || (!output && output_capacity))
    return GeometryEmissionStatus::kInvalidStorage;
  if (vertex_count > std::numeric_limits<std::uint32_t>::max())
    return GeometryEmissionStatus::kInvalidRange;

  std::uint64_t required = 0;
  VisitInput(topology, vertex_indices, vertex_count, primitive_restart,
             restart_index,
             [&required](const auto &, std::uint8_t) { ++required; });
  if (required > output_capacity)
    return GeometryEmissionStatus::kInsufficientCapacity;
  std::uint32_t cursor = 0;
  VisitInput(topology, vertex_indices, vertex_count, primitive_restart,
             restart_index, [&](const auto &indices, std::uint8_t count) {
    GeometryInputPrimitive primitive{};
    std::memcpy(primitive.vertex_indices, indices.data(),
                 sizeof(primitive.vertex_indices));
    primitive.primitive_id = cursor;
    primitive.instance_id = instance_id;
    primitive.vertex_count = count;
    output[cursor++] = primitive;
  });
  written_count = cursor;
  return GeometryEmissionStatus::kSuccess;
}

}  // namespace pvrgpu::stub
