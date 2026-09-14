// USC task stream helpers; see usc_task_stream.h.
#include "shader/usc_task_stream.h"

#include "common/pipeline_state.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pvrgpu::stub {
namespace {

void AddCounter(std::uint64_t &total, std::uint64_t value, const char *field) {
  if (value > std::numeric_limits<std::uint64_t>::max() - total)
    throw std::overflow_error(std::string("partial render counter overflow: ") + field);
  total += value;
}

std::uint64_t CheckedProduct(std::uint64_t count, std::uint32_t rate) {
  if (rate != 0 && count > std::numeric_limits<std::uint64_t>::max() / rate)
    throw std::overflow_error("USC stream cycle overflow");
  return count * rate;
}

void ReleaseIfOwned(MemoryPool &pool, PoolHandle &handle, PoolHandle keep) {
  if (!HasPoolHandle(handle)) return;
  if (!(handle.slot == keep.slot && handle.generation == keep.generation))
    pool.Release(handle);
  handle = {};
}

void ReleaseHandle(MemoryPool &pool, PoolHandle &handle) {
  if (HasPoolHandle(handle)) pool.Release(handle);
  handle = {};
}

}  // namespace

std::uint64_t GeometryStreamCapacity() {
  // Validation knob: a small capacity forces many partial renders so their
  // equivalence with a single render can be checked on ordinary draws.
  static const std::uint64_t capacity = [] {
    const char *text = std::getenv("PVRGPU_GEOMETRY_STREAM_CAPACITY");
    if (!text || !*text) return kGeometryStreamCapacityPrimitives;
    char *end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (!end || *end || value == 0)
      throw std::runtime_error("PVRGPU_GEOMETRY_STREAM_CAPACITY must be a positive integer");
    return static_cast<std::uint64_t>(value);
  }();
  return capacity;
}

UscIssuePlan UscIssuePlan::ForLanes(std::uint64_t lanes) {
  const std::uint64_t width = kReferenceUarch.usc_issue_lanes;
  return {lanes, lanes / width + (lanes % width != 0 ? 1U : 0U)};
}

std::uint64_t UscStreamCycles(const UscStreamWork &work,
                              const UscStreamRates &rates) {
  std::uint64_t total = 0;
  for (const auto term : {CheckedProduct(work.accepted_inputs, rates.per_accepted_input),
                          CheckedProduct(work.invocations, rates.per_invocation),
                          CheckedProduct(work.emitted_vertices, rates.per_emitted_vertex),
                          CheckedProduct(work.exported_primitives, rates.per_exported_primitive)})
    AddCounter(total, term, "stream cycles");
  return CheckedProduct(total, kUscStreamPeriodCycles);
}

TextureSampleRequest MakeUscTextureRequest(UscStage stage,
                                           const PcoTextureRequest &issued) {
  const char *const name = stage == UscStage::kVertex ? "vertex"
                         : stage == UscStage::kGeometry ? "geometry" : "fragment";
  if (stage != UscStage::kFragment) {
    if (issued.lod_bias_present || issued.lod_bias)
      throw std::runtime_error(std::string(name) + " SMP shader LOD bias is unsupported");
    if (issued.gather)
      throw std::runtime_error(std::string(name) + " SMP raw gather is unsupported");
  }
  TextureSampleRequest request;
  request.shader_stage = stage == UscStage::kVertex ? ShaderStage::kVertex
                       : stage == UscStage::kGeometry ? ShaderStage::kGeometry
                                                      : ShaderStage::kFragment;
  std::copy(issued.coordinates.begin(), issued.coordinates.end(), std::begin(request.coordinates));
  std::copy(issued.texture_state.begin(), issued.texture_state.end(), std::begin(request.texture_state));
  std::copy(issued.sampler_state.begin(), issued.sampler_state.end(), std::begin(request.sampler_state));
  std::copy(issued.spatial_offsets.begin(), issued.spatial_offsets.end(), std::begin(request.spatial_offsets));
  request.coordinate_count = issued.coordinate_count;
  request.component_count = issued.component_count;
  request.descriptor_set = issued.descriptor_set;
  request.binding = issued.binding;
  request.dimension = issued.dimension;
  request.normalized = issued.normalized;
  request.fcnorm = issued.fcnorm;
  request.sample_index = issued.sample_index;
  request.sample_index_present = issued.sample_index_present;
  request.explicit_lod = issued.explicit_lod;
  request.explicit_lod_present = issued.explicit_lod_present;
  request.lod_bias = issued.lod_bias;
  request.lod_bias_present = issued.lod_bias_present;
  request.gather = issued.gather;
  request.shadow_reference = issued.shadow_reference;
  request.shadow_compare = issued.shadow_compare;
  request.data_request = issued.data_request;
  request.texture_address_lo = issued.texture_address_lo;
  request.texture_address_hi = issued.texture_address_hi;
  return request;
}

// Adding a counter must decide its partial-render merge rule here.
static_assert(sizeof(CounterTxn) == 968, "update AccumulatePartialRenderCounters for new CounterTxn fields");

void AccumulatePartialRenderCounters(CounterTxn &total, const CounterTxn &batch) {
  if (total.frame != batch.frame || total.functional_frame != batch.functional_frame)
    throw std::runtime_error("partial render counters belong to different frames");
  AddCounter(total.ia_vertices, batch.ia_vertices, "ia_vertices");
  AddCounter(total.ia_primitives, batch.ia_primitives, "ia_primitives");
  AddCounter(total.vs_invocations, batch.vs_invocations, "vs_invocations");
  AddCounter(total.gs_invocations, batch.gs_invocations, "gs_invocations");
  AddCounter(total.gs_primitives, batch.gs_primitives, "gs_primitives");
  AddCounter(total.c_invocations, batch.c_invocations, "c_invocations");
  AddCounter(total.c_primitives, batch.c_primitives, "c_primitives");
  AddCounter(total.ps_invocations, batch.ps_invocations, "ps_invocations");
  AddCounter(total.hs_invocations, batch.hs_invocations, "hs_invocations");
  AddCounter(total.ds_invocations, batch.ds_invocations, "ds_invocations");
  AddCounter(total.tcs_invocations, batch.tcs_invocations, "tcs_invocations");
  AddCounter(total.tcs_alu_instructions, batch.tcs_alu_instructions, "tcs_alu_instructions");
  AddCounter(total.tcs_tex_instructions, batch.tcs_tex_instructions, "tcs_tex_instructions");
  AddCounter(total.tcs_memory_instructions, batch.tcs_memory_instructions, "tcs_memory_instructions");
  AddCounter(total.tcs_load_instructions, batch.tcs_load_instructions, "tcs_load_instructions");
  AddCounter(total.tcs_store_instructions, batch.tcs_store_instructions, "tcs_store_instructions");
  AddCounter(total.tcs_input_write_bytes, batch.tcs_input_write_bytes, "tcs_input_write_bytes");
  AddCounter(total.tcs_input_read_bytes, batch.tcs_input_read_bytes, "tcs_input_read_bytes");
  AddCounter(total.tcs_output_write_bytes, batch.tcs_output_write_bytes, "tcs_output_write_bytes");
  AddCounter(total.tcs_output_read_bytes, batch.tcs_output_read_bytes, "tcs_output_read_bytes");
  AddCounter(total.tes_alu_instructions, batch.tes_alu_instructions, "tes_alu_instructions");
  AddCounter(total.tes_tex_instructions, batch.tes_tex_instructions, "tes_tex_instructions");
  AddCounter(total.tes_memory_instructions, batch.tes_memory_instructions, "tes_memory_instructions");
  AddCounter(total.tes_load_instructions, batch.tes_load_instructions, "tes_load_instructions");
  AddCounter(total.tes_patch_read_bytes, batch.tes_patch_read_bytes, "tes_patch_read_bytes");
  AddCounter(total.tessellation_patches, batch.tessellation_patches, "tessellation_patches");
  AddCounter(total.tessellation_primitives, batch.tessellation_primitives, "tessellation_primitives");
  AddCounter(total.tessellation_domain_write_bytes, batch.tessellation_domain_write_bytes, "tessellation_domain_write_bytes");
  AddCounter(total.tessellation_domain_read_bytes, batch.tessellation_domain_read_bytes, "tessellation_domain_read_bytes");
  AddCounter(total.tessellation_level_read_bytes, batch.tessellation_level_read_bytes, "tessellation_level_read_bytes");
  total.drawlists = batch.drawlists;  // keep
  AddCounter(total.setup_triangles, batch.setup_triangles, "setup_triangles");
  AddCounter(total.texel_fetches, batch.texel_fetches, "texel_fetches");
  total.virtual_gpu_cycles = batch.virtual_gpu_cycles;  // recompute
  AddCounter(total.tiler_cycles, batch.tiler_cycles, "tiler_cycles");
  AddCounter(total.renderer_cycles, batch.renderer_cycles, "renderer_cycles");
  AddCounter(total.usc_groups, batch.usc_groups, "usc_groups");
  AddCounter(total.texture_requests, batch.texture_requests, "texture_requests");
  AddCounter(total.fifo_stall_events, batch.fifo_stall_events, "fifo_stall_events");
  total.pool_bytes_in_flight = std::max(total.pool_bytes_in_flight, batch.pool_bytes_in_flight);
  total.pool_high_water_bytes = std::max(total.pool_high_water_bytes, batch.pool_high_water_bytes);
  AddCounter(total.vdm_cycles, batch.vdm_cycles, "vdm_cycles");
  AddCounter(total.vertex_fetch_cycles, batch.vertex_fetch_cycles, "vertex_fetch_cycles");
  AddCounter(total.vertex_attribute_fetches, batch.vertex_attribute_fetches, "vertex_attribute_fetches");
  AddCounter(total.vertex_attribute_bytes, batch.vertex_attribute_bytes, "vertex_attribute_bytes");
  AddCounter(total.pco_decode_cycles, batch.pco_decode_cycles, "pco_decode_cycles");
  AddCounter(total.pco_instructions, batch.pco_instructions, "pco_instructions");
  AddCounter(total.vs_alu_instructions, batch.vs_alu_instructions, "vs_alu_instructions");
  AddCounter(total.vs_tex_instructions, batch.vs_tex_instructions, "vs_tex_instructions");
  AddCounter(total.vs_memory_instructions, batch.vs_memory_instructions, "vs_memory_instructions");
  AddCounter(total.fs_alu_instructions, batch.fs_alu_instructions, "fs_alu_instructions");
  AddCounter(total.fs_tex_instructions, batch.fs_tex_instructions, "fs_tex_instructions");
  AddCounter(total.fs_memory_instructions, batch.fs_memory_instructions, "fs_memory_instructions");
  AddCounter(total.usc_slot_cycles, batch.usc_slot_cycles, "usc_slot_cycles");
  AddCounter(total.usc_cluster_cycles, batch.usc_cluster_cycles, "usc_cluster_cycles");
  AddCounter(total.clip_cull_cycles, batch.clip_cull_cycles, "clip_cull_cycles");
  AddCounter(total.tiler_bin_cycles, batch.tiler_bin_cycles, "tiler_bin_cycles");
  AddCounter(total.parameter_buffer_cycles, batch.parameter_buffer_cycles, "parameter_buffer_cycles");
  AddCounter(total.parameter_coefficient_sets, batch.parameter_coefficient_sets, "parameter_coefficient_sets");
  AddCounter(total.parameter_write_bytes, batch.parameter_write_bytes, "parameter_write_bytes");
  AddCounter(total.pds_coefficient_tasks, batch.pds_coefficient_tasks, "pds_coefficient_tasks");
  AddCounter(total.pds_douti_issues, batch.pds_douti_issues, "pds_douti_issues");
  AddCounter(total.usc_coefficient_load_bytes, batch.usc_coefficient_load_bytes, "usc_coefficient_load_bytes");
  AddCounter(total.tile_scheduler_cycles, batch.tile_scheduler_cycles, "tile_scheduler_cycles");
  AddCounter(total.isp_cycles, batch.isp_cycles, "isp_cycles");
  AddCounter(total.fragment_frontend_cycles, batch.fragment_frontend_cycles, "fragment_frontend_cycles");
  AddCounter(total.texture_cycles, batch.texture_cycles, "texture_cycles");
  AddCounter(total.pbe_cycles, batch.pbe_cycles, "pbe_cycles");
  total.pixel_data_master_transactions = batch.pixel_data_master_transactions;  // last: the draw's final attachment commit
  total.pixel_data_master_bytes = batch.pixel_data_master_bytes;  // last: the draw's final attachment commit
  AddCounter(total.pixel_data_master_cycles, batch.pixel_data_master_cycles, "pixel_data_master_cycles");
  AddCounter(total.tcu_line_accesses, batch.tcu_line_accesses, "tcu_line_accesses");
  AddCounter(total.tcu_read_accesses, batch.tcu_read_accesses, "tcu_read_accesses");
  AddCounter(total.tcu_hits, batch.tcu_hits, "tcu_hits");
  AddCounter(total.tcu_misses, batch.tcu_misses, "tcu_misses");
  AddCounter(total.tcu_evictions, batch.tcu_evictions, "tcu_evictions");
  AddCounter(total.tcu_writebacks, batch.tcu_writebacks, "tcu_writebacks");
  AddCounter(total.tcu_bypassed, batch.tcu_bypassed, "tcu_bypassed");
  AddCounter(total.tcu_cycles, batch.tcu_cycles, "tcu_cycles");
  AddCounter(total.slc_line_accesses, batch.slc_line_accesses, "slc_line_accesses");
  AddCounter(total.slc_read_accesses, batch.slc_read_accesses, "slc_read_accesses");
  AddCounter(total.slc_write_accesses, batch.slc_write_accesses, "slc_write_accesses");
  AddCounter(total.slc_hits, batch.slc_hits, "slc_hits");
  AddCounter(total.slc_misses, batch.slc_misses, "slc_misses");
  AddCounter(total.slc_evictions, batch.slc_evictions, "slc_evictions");
  AddCounter(total.slc_writebacks, batch.slc_writebacks, "slc_writebacks");
  AddCounter(total.slc_bypassed, batch.slc_bypassed, "slc_bypassed");
  AddCounter(total.slc_cycles, batch.slc_cycles, "slc_cycles");
  AddCounter(total.dram_read_transactions, batch.dram_read_transactions, "dram_read_transactions");
  AddCounter(total.dram_write_transactions, batch.dram_write_transactions, "dram_write_transactions");
  AddCounter(total.dram_read_bytes, batch.dram_read_bytes, "dram_read_bytes");
  AddCounter(total.dram_write_bytes, batch.dram_write_bytes, "dram_write_bytes");
  AddCounter(total.dram_cycles, batch.dram_cycles, "dram_cycles");
  AddCounter(total.memory_direct_read_bytes, batch.memory_direct_read_bytes, "memory_direct_read_bytes");
  AddCounter(total.memory_direct_write_bytes, batch.memory_direct_write_bytes, "memory_direct_write_bytes");
  total.framebuffer_dram_readback_bytes = batch.framebuffer_dram_readback_bytes;  // last
  AddCounter(total.tiles_binned, batch.tiles_binned, "tiles_binned");
  AddCounter(total.tiles_scheduled, batch.tiles_scheduled, "tiles_scheduled");
  AddCounter(total.covered_pixels, batch.covered_pixels, "covered_pixels");
  AddCounter(total.fragment_candidates, batch.fragment_candidates, "fragment_candidates");
  AddCounter(total.hsr_rejected_fragments, batch.hsr_rejected_fragments, "hsr_rejected_fragments");
  AddCounter(total.stencil_tested_fragments, batch.stencil_tested_fragments, "stencil_tested_fragments");
  AddCounter(total.stencil_rejected_fragments, batch.stencil_rejected_fragments, "stencil_rejected_fragments");
  AddCounter(total.stencil_written_fragments, batch.stencil_written_fragments, "stencil_written_fragments");
  AddCounter(total.depth_tested_fragments, batch.depth_tested_fragments, "depth_tested_fragments");
  AddCounter(total.depth_rejected_fragments, batch.depth_rejected_fragments, "depth_rejected_fragments");
  AddCounter(total.depth_written_fragments, batch.depth_written_fragments, "depth_written_fragments");
  AddCounter(total.pbe_color_reads, batch.pbe_color_reads, "pbe_color_reads");
  AddCounter(total.pbe_blended_fragments, batch.pbe_blended_fragments, "pbe_blended_fragments");
  AddCounter(total.pbe_fragment_writes, batch.pbe_fragment_writes, "pbe_fragment_writes");
  AddCounter(total.occlusion_samples_passed, batch.occlusion_samples_passed, "occlusion_samples_passed");
  AddCounter(total.pbe_pixels_written, batch.pbe_pixels_written, "pbe_pixels_written");
  AddCounter(total.gs_alu_instructions, batch.gs_alu_instructions, "gs_alu_instructions");
  AddCounter(total.gs_tex_instructions, batch.gs_tex_instructions, "gs_tex_instructions");
  AddCounter(total.gs_memory_instructions, batch.gs_memory_instructions, "gs_memory_instructions");
  AddCounter(total.gs_load_instructions, batch.gs_load_instructions, "gs_load_instructions");
  AddCounter(total.gs_emitted_vertices, batch.gs_emitted_vertices, "gs_emitted_vertices");
  AddCounter(total.gs_input_write_bytes, batch.gs_input_write_bytes, "gs_input_write_bytes");
  AddCounter(total.gs_input_read_bytes, batch.gs_input_read_bytes, "gs_input_read_bytes");
}

void FinalizePartialRenderCounters(CounterTxn &counters) {
  std::uint64_t cycles = counters.tiler_cycles;
  AddCounter(cycles, counters.renderer_cycles, "virtual GPU cycles");
  AddCounter(cycles, kReferenceUarch.fixed_submission_cycles, "virtual GPU cycles");
  counters.virtual_gpu_cycles = cycles;
}

CounterTxn PartialRenderCounterIdentity(const CounterTxn &counters) {
  CounterTxn identity;
  identity.frame = counters.frame;
  identity.functional_frame = counters.functional_frame;
  identity.drawlists = counters.drawlists;
  return identity;
}

void AccumulateShaderExecutions(DrawListShaderStats &total,
                                const DrawListShaderStats &batch) {
  AddCounter(total.invocations, batch.invocations, "shader invocations");
  AddCounter(total.executed_alu_instructions, batch.executed_alu_instructions, "shader ALU");
  AddCounter(total.executed_tex_instructions, batch.executed_tex_instructions, "shader TEX");
  AddCounter(total.executed_memory_instructions, batch.executed_memory_instructions, "shader MEM");
}

void BeginPartialRender(MemoryPool &pool, PipelineState &state) {
  auto &partial = state.partial_render;
  if (partial.active || HasPoolHandle(partial.counters) || HasPoolHandle(partial.fragment_executions))
    throw std::runtime_error("partial render began twice");
  partial = {};
  partial.active = 1;
  partial.counters = StoreNewArray(pool, std::vector<CounterTxn>{PartialRenderCounterIdentity(state.counters)});
  partial.fragment_executions = StoreNewArray(pool, std::vector<DrawListShaderStats>(1));
  partial.original_color_load = state.color_attachment_load;
  partial.original_depth_load = state.depth_attachment_load;
  partial.original_attachment_clears = state.attachment_clears;
  partial.original_color_load_bytes = state.color_attachment_load_bytes;
  partial.original_depth_load_bytes = state.depth_attachment_load_bytes;
  partial.original_color_load_enable = state.color_attachment_load_enable;
  partial.original_depth_load_enable = state.depth_attachment_load_enable;
}

void ChainPartialRender(MemoryPool &pool, PipelineState &state) {
  auto &partial = state.partial_render;
  if (!partial.active || partial.last || state.stage != PipelineStage::kFramebufferReady ||
      state.framebuffer_from_dram != 1 || !HasPoolHandle(state.dram_framebuffer))
    throw std::runtime_error("partial render completion is not a finished render");

  // Fold this render's work into the running totals.
  auto totals = LoadArray<CounterTxn>(pool, partial.counters);
  auto fragment_totals = LoadArray<DrawListShaderStats>(pool, partial.fragment_executions);
  if (totals.size() != 1 || fragment_totals.size() != 1)
    throw std::runtime_error("partial render accumulator is malformed");
  AccumulatePartialRenderCounters(totals[0], state.counters);
  StoreArray(pool, partial.counters, totals);
  auto drawlists = LoadArray<DrawListStats>(pool, state.drawlist_stats);
  if (drawlists.size() != 1)
    throw std::runtime_error("partial render requires one DrawList record");
  AccumulateShaderExecutions(fragment_totals[0], drawlists[0].fragment);
  StoreArray(pool, partial.fragment_executions, fragment_totals);
  drawlists[0].fragment = DrawListShaderStats{};   // decoded again next render
  StoreArray(pool, state.drawlist_stats, drawlists);
  state.counters = PartialRenderCounterIdentity(state.counters);
  // TextureUnit's per-render traffic reconciliation restarts with the counters.
  for (std::uint64_t *traffic : {&state.vertex_texture_request_count, &state.fragment_texture_request_count,
                                 &state.geometry_texture_request_count,
                                 &state.tessellation_control_texture_request_count,
                                 &state.tessellation_evaluation_texture_request_count,
                                 &state.vertex_texel_fetch_count, &state.fragment_texel_fetch_count,
                                 &state.geometry_texel_fetch_count,
                                 &state.tessellation_control_texel_fetch_count,
                                 &state.tessellation_evaluation_texel_fetch_count,
                                 &state.geometry_texture_instruction_count})
    *traffic = 0;

  // Committed attachments become the next render's LOAD.
  const std::uint32_t targets = state.render_target_count ? state.render_target_count : 1U;
  std::vector<std::uint8_t> color = LoadArray<std::uint8_t>(pool, state.dram_framebuffer);
  for (std::uint32_t target = 1; target < targets; ++target) {
    const auto extra = LoadArray<std::uint8_t>(pool, state.extra_dram_framebuffer[target - 1]);
    color.insert(color.end(), extra.begin(), extra.end());
  }
  ReleaseIfOwned(pool, state.color_attachment_load, partial.original_color_load);
  state.color_attachment_load = StoreNewArray(pool, color);
  state.color_attachment_load_bytes = color.size();
  state.color_attachment_load_enable = 1;
  ReleaseHandle(pool, state.dram_framebuffer);
  for (auto &extra : state.extra_dram_framebuffer) ReleaseHandle(pool, extra);
  state.framebuffer_from_dram = 0;
  if (state.capture_depth_attachment) {
    if (state.depth_attachment_ready != 1 || !HasPoolHandle(state.depth_attachment))
      throw std::runtime_error("partial render has no committed depth attachment");
    const auto depth = LoadArray<std::uint8_t>(pool, state.depth_attachment);
    ReleaseIfOwned(pool, state.depth_attachment_load, partial.original_depth_load);
    state.depth_attachment_load = StoreNewArray(pool, depth);
    state.depth_attachment_load_bytes = depth.size();
    state.depth_attachment_load_enable = 1;
    ReleaseHandle(pool, state.depth_attachment);
    state.depth_attachment_bytes = 0;
    state.depth_attachment_ready = 0;
  }
  // Inherited clears already landed in render 0.
  ReleaseIfOwned(pool, state.attachment_clears, partial.original_attachment_clears);

  // Every raster intermediate belongs to the finished render.
  for (PoolHandle *handle : {&state.raster_triangles, &state.raster_vertex_outputs,
                             &state.tile_records, &state.tile_primitive_refs,
                             &state.parameter_triangles, &state.parameter_coefficients,
                             &state.fragment_candidates, &state.isp_depth_attachment,
                             &state.isp_stencil_attachment, &state.fragment_invocations,
                             &state.fragment_shader_lanes, &state.fragment_quads,
                             &state.usc_fragment_tasks, &state.usc_coefficient_banks,
                             &state.fragment_outputs, &state.fragment_instructions,
                             &state.pbe_framebuffer, &state.slc_writeback_lines})
    ReleaseHandle(pool, *handle);
  for (auto &extra : state.extra_pbe_framebuffer) ReleaseHandle(pool, extra);
  state.active_fragment_invocations = 0;
  state.fragment_shader_lane_count = 0;
  state.scheduled_tiles = 0;
  state.fragment_groups = 0;
  state.fragment_images_complete = 0;
  state.stage = PipelineStage::kVertexShaded;
  ++partial.batch_index;
}

void FinishPartialRender(MemoryPool &pool, PipelineState &state) {
  auto &partial = state.partial_render;
  if (!partial.active || !partial.last)
    throw std::runtime_error("partial render finished before its last render");
  auto totals = LoadArray<CounterTxn>(pool, partial.counters);
  const auto fragment_totals = LoadArray<DrawListShaderStats>(pool, partial.fragment_executions);
  if (totals.size() != 1 || fragment_totals.size() != 1)
    throw std::runtime_error("partial render accumulator is malformed");
  AccumulatePartialRenderCounters(totals[0], state.counters);
  FinalizePartialRenderCounters(totals[0]);
  state.counters = totals[0];
  auto drawlists = LoadArray<DrawListStats>(pool, state.drawlist_stats);
  if (drawlists.size() != 1)
    throw std::runtime_error("partial render requires one DrawList record");
  AccumulateShaderExecutions(drawlists[0].fragment, fragment_totals[0]);
  StoreArray(pool, state.drawlist_stats, drawlists);

  // Restore the draw's own LOAD evidence.
  ReleaseIfOwned(pool, state.color_attachment_load, partial.original_color_load);
  ReleaseIfOwned(pool, state.depth_attachment_load, partial.original_depth_load);
  state.color_attachment_load = partial.original_color_load;
  state.depth_attachment_load = partial.original_depth_load;
  state.color_attachment_load_bytes = partial.original_color_load_bytes;
  state.depth_attachment_load_bytes = partial.original_depth_load_bytes;
  state.color_attachment_load_enable = partial.original_color_load_enable;
  state.depth_attachment_load_enable = partial.original_depth_load_enable;
  if (HasPoolHandle(state.attachment_clears))
    throw std::runtime_error("partial render left a later attachment clear");
  state.attachment_clears = partial.original_attachment_clears;
  pool.Release(partial.counters);
  pool.Release(partial.fragment_executions);
  partial = {};
}

}  // namespace pvrgpu::stub
