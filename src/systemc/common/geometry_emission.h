// SPDX-License-Identifier: MIT
// Fixed-function geometry assembly. This helper neither executes shaders nor
// invents emission events: the native geometry ISS supplies Emit/CUT/ENDTASK.
#pragma once

#include <cstddef>
#include <cstdint>

namespace pvrgpu::stub {

enum class GeometryOutputTopology : std::uint8_t {
  kPoints,
  kLineStrip,
  kTriangleStrip,
};

enum class GeometryInputTopology : std::uint8_t {
  kPoints,
  kLines,
  kLineStrip,
  kLineLoop,
  kTriangles,
  kTriangleStrip,
  kTriangleFan,
  kLinesAdjacency,
  kLineStripAdjacency,
  kTrianglesAdjacency,
  kTriangleStripAdjacency,
};

enum class GeometryEmissionStatus : std::uint8_t {
  kSuccess,
  kEmitSuppressed,
  kInvalidOutputDwordCount,
  kInvalidStorage,
  kInvalidWrittenMask,
  kInvalidTopology,
  kInvalidRange,
  kInsufficientCapacity,
  kFinished,
};

const char *GeometryEmissionStatusName(GeometryEmissionStatus status);

struct GeometryStripRange {
  std::uint32_t first_vertex = 0;
  std::uint32_t vertex_count = 0;
};

// Indices address immutable GS emission snapshots, not original VS indices.
// Repeated entries adapt points/lines to a three-reference downstream path;
// vertex_count still identifies the real primitive and provoking_vertex is a
// local index. Input identity and shader-written PrimitiveID remain separate.
struct GeometryPrimitiveRef {
  std::uint32_t vertex_indices[3]{};
  std::uint8_t vertex_count = 0;
  std::uint8_t provoking_vertex = 0;
  std::uint8_t reserved[2]{};
};

struct GeometryInputPrimitive {
  std::uint32_t vertex_indices[6]{};
  std::uint32_t primitive_id = 0;
  std::uint32_t instance_id = 0;
  std::uint8_t vertex_count = 0;
  std::uint8_t reserved[3]{};
};

// One record per complete GS-generated primitive (one three-entry raster
// lane-ref tuple). Shader-written Layer/PrimitiveID remain raw output fields;
// these are upstream provenance, not replacements for shader outputs.
struct GeometryRasterPrimitive {
  GeometryPrimitiveRef refs;
  std::uint32_t input_primitive_id = 0;
  std::uint32_t instance_id = 0;
  std::uint32_t invocation_id = 0;
};

// Caller owns all bounded storage. In SystemC the module maps its exclusively
// owned MemoryPool payloads while executing, then publishes only PoolHandles.
// No allocation, queue, global state or memory timing is hidden in this helper.
struct GeometryEmissionStorage {
  std::uint32_t *snapshot_dwords = nullptr;
  std::size_t snapshot_dword_capacity = 0;
  std::uint64_t *written_masks = nullptr;
  std::size_t written_mask_capacity = 0;
  GeometryStripRange *strips = nullptr;
  std::size_t strip_capacity = 0;
};

class GeometryEmissionBuffer {
 public:
  GeometryEmissionBuffer(GeometryEmissionStorage storage,
                         std::uint32_t max_vertices,
                         std::uint32_t output_dword_count);

  // Snapshot all raw output DWORDs and preserve which were actually written.
  // No missing position/varying is synthesized. The native executor validates
  // any required outputs and resets its written mask after successful Emit.
  // At max_vertices (including zero), Emit is suppressed without reading the
  // source, while later shader instructions, CUT and ENDTASK must still run.
  GeometryEmissionStatus Emit(const std::uint32_t *output_dwords,
                              std::size_t output_dword_count,
                              std::uint64_t written_mask);
  GeometryEmissionStatus EndPrimitive();
  GeometryEmissionStatus Finish();

  std::uint32_t emitted_vertices() const;
  std::size_t strip_count() const;
  bool finished() const;

 private:
  GeometryEmissionStatus ValidateStorage() const;

  GeometryEmissionStorage storage_;
  std::uint32_t max_vertices_ = 0;
  std::uint32_t output_dword_count_ = 0;
  std::uint32_t emitted_vertices_ = 0;
  std::uint32_t open_strip_vertices_ = 0;
  std::size_t strip_count_ = 0;
  bool finished_ = false;
};

// All ranges must form the contiguous snapshot stream of one completed GS
// invocation. Empty/partial strips generate no complete raster primitive.
// Capacity/range failure leaves both the output and written count unchanged.
GeometryEmissionStatus ExpandGeometryPrimitives(
    GeometryOutputTopology topology, const GeometryStripRange *strips,
    std::size_t strip_count, std::uint32_t snapshot_count,
    GeometryPrimitiveRef *output, std::size_t output_capacity,
    std::size_t &written_count);

// Assemble one draw instance before clipping, width expansion or face culling.
// Entries are caller-defined VS occurrence/lane reference indices, not VBO
// addresses. Restart markers split topology/parity but do not reset primitive
// IDs; each call starts primitive_id at zero for the supplied instance_id.
// All output storage is validated before any entry/count is published.
GeometryEmissionStatus AssembleGeometryInputPrimitives(
    GeometryInputTopology topology, const std::uint32_t *vertex_indices,
    std::size_t vertex_count, bool primitive_restart,
    std::uint32_t restart_index, std::uint32_t instance_id,
    GeometryInputPrimitive *output, std::size_t output_capacity,
    std::size_t &written_count);

}  // namespace pvrgpu::stub
