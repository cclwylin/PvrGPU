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
  // Topology as originally submitted, before VertexFetch expands lines
  // and points into a TriangleList index buffer for the fixed
  // triangle-only downstream stages. ClipCull reads this to know which
  // "triangles" are a line/point that still needs real width expansion
  // rather than a genuine shaded triangle. Defaults to kTriangleStrip,
  // matching DrawCommand::topology's own default; every non-indexed
  // path never reads this field.
  PrimitiveTopology source_topology = PrimitiveTopology::kTriangleStrip;
  // Set when the draw command stated its own vertex attribute widths rather
  // than matching one of the pinned capture layouts.
  std::uint32_t driver_describes_attributes = 0;
  RasterState raster_state;
  PoolHandle drawlist_stats;
  PoolHandle vertex_buffer_resources;
  PoolHandle vertex_attribute_bindings;
  PoolHandle vertex_indices;
  PoolHandle vertex_lanes;
  PoolHandle vertex_lane_refs;
  PoolHandle vertex_shared_registers;
  PoolHandle shader_varying_bindings;
  PoolHandle vertex_texture_resources;
  PoolHandle vertex_sampler_states;
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
  // Initial attachment payloads for API-v7 alias+LOAD render passes. The
  // Submitter obtains these from the authoritative DRAM allocation at the
  // ordered sequence barrier; ISP/PBE consume them as initial destination
  // state rather than re-clearing the aliased target.
  PoolHandle color_attachment_load;
  PoolHandle depth_attachment_load;
  // ISP owns the complete post-depth-test surface (including untouched LOAD
  // pixels). Entries retain the exact encoded UNORM integer so an untouched
  // Z32 LOAD pixel is not rounded through float before the DRAM commit.
  PoolHandle isp_depth_attachment;
  // Optional Z32_UNORM attachment materialized from the same ISP-visible work
  // that
  // feeds FragmentFrontend.  Native multi-pass commands use this handle at a
  // completion barrier before the next draw binds it as a sampled resource.
  // It is never populated for ordinary single-draw submissions.
  PoolHandle depth_attachment;
  PoolHandle fragment_invocations;
  PoolHandle fragment_shader_lanes;
  PoolHandle fragment_quads;
  PoolHandle usc_fragment_tasks;
  PoolHandle usc_coefficient_banks;
  PoolHandle texture_sample_requests;
  PoolHandle texture_sample_responses;
  PoolHandle vertex_continuations;
  PoolHandle fragment_continuations;
  PoolHandle fragment_outputs;
  PoolHandle pbe_framebuffer;
  PoolHandle slc_writeback_lines;
  PoolHandle dram_framebuffer;
  PcoProgramSummary vertex_program_summary;
  PcoProgramSummary fragment_program_summary;
  // Producer-declared PCO resource ABI and exact VS-to-FS linkage. Keeping
  // these in the per-draw control block lets every downstream module validate
  // dynamic driver payloads without recognizing a shader name or binary.
  DriverPcoStageAbi vertex_pco_abi;
  DriverPcoStageAbi fragment_pco_abi;
  std::uint32_t position_output_start = 0;
  std::uint32_t position_output_count = 0;
  std::uint32_t fragment_position_start = 0;
  std::uint32_t fragment_position_count = 0;
  std::uint32_t varying_output_start = 0;
  std::uint32_t varying_output_count = 0;
  std::uint32_t fragment_varying_start = 0;
  std::uint32_t fragment_varying_count = 0;
  // Number of exact descriptor-set resources bound by this physical draw.
  // TextureUnit cross-checks each raw 20-dword combined descriptor against
  // the correspondingly numbered owned resource and sampler.
  std::uint32_t vertex_sampled_texture_count = 0;
  std::uint32_t sampled_texture_count = 0;
  // Internal stage-bank accounting.  Public counters remain aggregate, while
  // these totals prove that VS and FS FIFO traffic cannot be silently charged
  // to the other stage and that multi-round SMP continuations are not lost.
  std::uint64_t vertex_texture_request_count = 0;
  std::uint64_t fragment_texture_request_count = 0;
  std::uint64_t vertex_texel_fetch_count = 0;
  std::uint64_t fragment_texel_fetch_count = 0;
  // Colour attachment 0.  A multiple-render-target pass keeps attachment 0
  // on these fields so every single-target path stays byte identical, and
  // describes attachments 1..render_target_count-1 alongside them.
  std::uint64_t framebuffer_gpu_address = 0;
  std::uint64_t framebuffer_bytes = 0;
  std::uint32_t render_target_count = 1;
  // Resolved attachments 1..render_target_count-1; attachment 0 stays in
  // pbe_framebuffer so every single-target consumer is unchanged.
  PoolHandle extra_pbe_framebuffer[kMaxRenderTargets - 1]{};
  std::uint64_t extra_framebuffer_gpu_address[kMaxRenderTargets - 1]{};
  std::uint64_t extra_framebuffer_bytes[kMaxRenderTargets - 1]{};
  std::uint64_t depth_attachment_gpu_address = 0;
  std::uint64_t depth_attachment_bytes = 0;
  std::uint64_t color_attachment_load_bytes = 0;
  std::uint64_t depth_attachment_load_bytes = 0;
  std::uint32_t depth_attachment_format = 0;
  std::uint64_t index_buffer_gpu_address = 0;
  std::uint64_t index_buffer_bytes = 0;
  std::uint64_t parameter_triangles_gpu_address = 0;
  std::uint64_t parameter_triangles_bytes = 0;
  std::uint64_t parameter_coefficients_gpu_address = 0;
  std::uint64_t parameter_coefficients_bytes = 0;
  // Number of fragment invocations. With ordered blending this may exceed the
  // number of unique covered pixels.
  std::uint32_t active_fragment_invocations = 0;
  std::uint32_t fragment_shader_lane_count = 0;
  std::uint32_t scheduled_tiles = 0;
  std::uint8_t fragment_early_hsr_safe = 0;
  std::uint8_t color_attachment_load_enable = 0;
  std::uint8_t depth_attachment_load_enable = 0;
  std::uint8_t capture_depth_attachment = 0;
  std::uint8_t depth_attachment_ready = 0;
  MemoryMode memory_mode = MemoryMode::kCache;
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
