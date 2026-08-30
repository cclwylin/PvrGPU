// PipelineState is the small, trivially-copyable control block stored in the
// MemoryPool. 中文：FIFO 僅傳它的 handle；index、lane reference 與其他大型
// payload 各自使用獨立 handle，維持 event-driven module 邊界。
#pragma once

#include "common/functional_types.h"
#include "memory_pool.h"
#include "model_types.h"
#include "shader/pco_iss.h"

#include <cstdint>

namespace pvrgpu::stub {

static_assert(kPcoVertexOutputRegisterCount == kPcoVertexOutputCount,
              "common and PCO VTXOUT payload sizes must stay aligned");
static_assert(kVaryingsShaderOneCoefficientDwordCount ==
                  kPcoVaryingOneCoefficientCount,
              "PDS and PCO varying coefficient banks must stay aligned");
static_assert(kVaryingsShaderTwoCoefficientDwordCount ==
                  kPcoVaryingTwoCoefficientCount,
              "PDS and PCO two-varying coefficient banks must stay aligned");
static_assert(kVaryingsShaderFourCoefficientDwordCount ==
                  kPcoVaryingFourCoefficientCount,
              "PDS and PCO four-varying coefficient banks must stay aligned");
static_assert(kVaryingsShaderEightCoefficientDwordCount ==
                  kPcoVaryingEightCoefficientCount,
              "PDS and PCO eight-varying coefficient banks must stay aligned");
static_assert(kFillTexNearestCoefficientDwordCount ==
                  kPcoFillTexNearestCoefficientCount,
              "PDS and PCO texture coefficient banks must stay aligned");
static_assert(kFillTexNearestSharedDwordCount ==
                  kPcoFillTexNearestFragmentSharedCount,
              "PDS and PCO texture shared registers must stay aligned");

// Functional payload and accumulated counters live in MemoryPool. FIFO
// transactions carry only the generation-checked handle plus ordering fields.
// Raw PCO code, semantic instructions, primitive lists and per-fragment USC
// output remain separate handles so no large object crosses a module FIFO.
struct PipelineState {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t workload_class = 0;
  std::uint64_t sequence = 0;
  FunctionalCase functional_case = FunctionalCase::kNone;
  PipelineStage stage = PipelineStage::kSubmitted;
  DrawCommand draw;
  RasterState raster_state;
  PoolHandle drawlist_stats;
  PoolHandle vertex_buffer_resources;
  PoolHandle vertex_attribute_bindings;
  PoolHandle vertex_indices;
  PoolHandle vertex_lanes;
  PoolHandle vertex_lane_refs;
  PoolHandle vertex_shared_registers;
  PoolHandle shader_varying_bindings;
  PoolHandle texture_resources;
  PoolHandle sampler_states;
  PoolHandle fragment_shared_registers;
  PoolHandle vertex_code;
  PoolHandle vertex_instructions;
  PoolHandle fragment_code;
  PoolHandle fragment_instructions;
  PoolHandle raster_triangles;
  PoolHandle raster_vertex_outputs;
  PoolHandle tile_records;
  PoolHandle tile_primitive_refs;
  PoolHandle parameter_triangles;
  PoolHandle parameter_coefficients;
  PoolHandle fragment_candidates;
  PoolHandle fragment_invocations;
  PoolHandle fragment_shader_lanes;
  PoolHandle fragment_quads;
  PoolHandle usc_fragment_tasks;
  PoolHandle usc_coefficient_banks;
  PoolHandle texture_sample_requests;
  PoolHandle texture_sample_responses;
  PoolHandle fragment_continuations;
  PoolHandle fragment_outputs;
  PoolHandle pbe_framebuffer;
  PoolHandle slc_writeback_lines;
  PoolHandle dram_framebuffer;
  PcoProgramSummary vertex_program_summary;
  PcoProgramSummary fragment_program_summary;
  std::uint64_t framebuffer_gpu_address = 0;
  std::uint64_t framebuffer_bytes = 0;
  // Number of fragment invocations. With ordered blending this may exceed the
  // number of unique covered pixels.
  std::uint32_t active_fragment_invocations = 0;
  std::uint32_t fragment_shader_lane_count = 0;
  std::uint32_t scheduled_tiles = 0;
  std::uint8_t fragment_early_hsr_safe = 0;
  std::uint8_t cache_bypass = 0;
  std::uint8_t framebuffer_from_dram = 0;
  std::uint8_t primitive_restart_enable = 0;
  std::uint32_t primitive_restart_index = 0xFFFFFFFF;
  std::uint8_t clip_distance_mask = 0;
  std::uint16_t clip_distance_register = 0;
  std::uint64_t vertex_groups = 0;
  std::uint64_t fragment_groups = 0;
  CounterTxn counters;
};

PipelineState LoadPipelineState(const MemoryPool &pool, PoolHandle handle);
void StorePipelineState(MemoryPool &pool, PoolHandle handle,
                        const PipelineState &state);

std::uint64_t CeilDivide(std::uint64_t value, std::uint64_t divisor);

// The stub uses one nanosecond per configured cycle.  Each call schedules one
// timed completion event; it never produces a clock edge or polls per cycle.
void WaitForCycles(std::uint64_t cycles);

} // namespace pvrgpu::stub
