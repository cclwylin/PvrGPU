// SPDX-License-Identifier: MIT
// Pool-owned state between independent TCS, fixed tessellator, and TES.
#pragma once

#include "common/tessellation.h"
#include "memory_pool.h"
#include "model_types.h"
#include "shader/pco_iss.h"

namespace pvrgpu::stub {

constexpr std::uint32_t kTessellationTaskWidth = 32;
constexpr std::uint32_t kTessellationMaxPatches = 4096;
constexpr std::uint32_t kTessellationMaxDrawPoints = 1U << 20U;
constexpr std::uint32_t kTessellationMaxDrawIndices = 6U << 20U;
constexpr std::uint32_t kTessellationPatchAddressStride = 16384;
constexpr std::uint64_t kTessellationDrawAddressStride = UINT64_C(0x8000000);

enum class TessellationPhase : std::uint32_t {
  kSubmitted, kControlComplete, kDomainComplete, kEvaluationComplete,
};

struct TessellationPatch {
  std::uint32_t primitive_id = 0;
  std::uint32_t instance_id = 0;
  std::uint32_t first_occurrence = 0;
  std::uint32_t input_vertices = 0;
  std::uint64_t output_address = 0;
  std::uint64_t domain_address = 0;
  std::uint32_t point_start = 0;
  std::uint32_t point_count = 0;
  std::uint32_t index_start = 0;
  std::uint32_t index_count = 0;
  std::uint32_t primitive_size = 0;
  std::uint32_t reserved = 0;
};

struct TessellationState {
  PoolHandle control_code;
  PoolHandle control_instructions;
  PoolHandle control_shared;
  PoolHandle control_uniform_buffers;
  PoolHandle evaluation_code;
  PoolHandle evaluation_instructions;
  PoolHandle evaluation_shared;
  PoolHandle evaluation_uniform_buffers;
  PoolHandle patches;
  // Indices use patch-local point indices; point_start locates each patch's
  // domain values in the GPU address region. No bulk data crosses a FIFO.
  PoolHandle domain_points;
  PoolHandle domain_indices;
  DriverPcoStageAbi control_abi;
  DriverPcoStageAbi evaluation_abi;
  PcoProgramSummary control_summary;
  PcoProgramSummary evaluation_summary;
  TessellationPhase phase = TessellationPhase::kSubmitted;
  std::uint32_t input_vertices = 0;
  std::uint32_t output_vertices = 0;
  std::uint32_t vertices_per_instance = 0;
  std::uint32_t input_stride_dwords = 0;
  std::uint32_t output_vertex_stride_dwords = 0;
  std::uint32_t per_vertex_offset_dwords = 0;
  std::uint32_t patch_stride_dwords = 0;
  std::uint32_t control_barrier_count = 0;
  TessellationDomain domain = TessellationDomain::kTriangles;
  TessellationSpacing spacing = TessellationSpacing::kEqual;
  std::uint8_t clockwise = 0;
  std::uint8_t point_mode = 0;
  std::uint64_t input_address = 0;
  std::uint64_t output_address = 0;
  std::uint64_t domain_address = 0;
};

static_assert(std::is_trivially_copyable_v<TessellationPatch>);
static_assert(std::is_trivially_copyable_v<TessellationState>);

} // namespace pvrgpu::stub
