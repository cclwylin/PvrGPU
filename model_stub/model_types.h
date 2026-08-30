#pragma once

#include <array>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace pvrgpu::stub {

inline constexpr const char *kSchema = "pvrgpu.counter.v1";

struct MesaPocCommand {
  bool enabled = false;
  std::string schema;
  std::string mesa_driver;
  std::uint32_t manifest_index = 0;
  std::string test_case;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::string rdc_sha256;
  std::string api_trace_sha256;
  std::string gallium_trace_sha256;
  std::string vertex_shader_sha256;
  std::string fragment_shader_sha256;
  std::uint64_t api_calls = 0;
  std::uint64_t api_draw_calls = 0;
  std::string primitive;
  bool indexed = false;
  std::uint32_t draw_count = 0;
  std::uint32_t index_size = 0;
  std::vector<std::uint16_t> indices;
  std::uint64_t gallium_draw_calls = 0;
  std::uint64_t gallium_target_draws = 0;
  std::uint64_t gallium_framebuffer_matches = 0;
  std::uint64_t gallium_resource_creates = 0;
  std::uint64_t gallium_state_calls = 0;
  std::uint64_t gallium_constant_bytes = 0;
  std::string framebuffer_format;
  std::array<std::uint32_t, 4> clear_color_bits{};
  bool depth_enabled = false;
  bool depth_write = false;
  std::uint32_t depth_func = 0;
  bool blend_enabled = false;
  std::string blend_rgb_func;
  std::string blend_rgb_src_factor;
  std::string blend_rgb_dst_factor;
  std::string blend_alpha_func;
  std::string blend_alpha_src_factor;
  std::string blend_alpha_dst_factor;
  bool front_ccw = false;
  std::uint32_t cull_face = 0;
  std::uint32_t vertex_elements = 0;
  std::string vertex_format;
  std::uint32_t vertex_stride = 0;
  std::uint32_t vertex_offset = 0;
  std::vector<std::uint32_t> vertex_buffer_map;
  std::vector<float> vertex_float2;
  std::vector<float> vertex_float2_1;
  std::vector<std::uint8_t> vertex_constants;
  std::vector<std::uint8_t> fragment_constants;
  std::string texture_format;
  std::uint32_t texture_width = 0;
  std::uint32_t texture_height = 0;
  std::uint32_t texture_levels = 0;
  std::vector<std::uint32_t> texture_mip_sizes;
  std::vector<std::uint8_t> texture_bytes;
  std::uint32_t sampler_min_img_filter = 0;
  std::uint32_t sampler_min_mip_filter = 0;
  std::uint32_t sampler_mag_img_filter = 0;
};

struct Options {
  unsigned frames = 5;
  unsigned width = 512;
  unsigned height = 512;
  std::string test_case = "fill_solid";
  std::string output_dir;
  // false: run the complete cache hierarchy; true: bypass caches for a
  // faster functional simulation while preserving the DRAM access path.
  bool cache_bypass = false;
  std::string mesa_command_path;
  MesaPocCommand mesa_command;
};

bool ParseOptions(int argc, char **argv, Options *options);
std::string JsonEscape(const std::string &value);
std::uint32_t WorkloadClass(const std::string &test_case);

struct PoolHandle {
  std::uint32_t slot = 0;
  std::uint32_t generation = 0;
};

std::ostream &operator<<(std::ostream &stream, const PoolHandle &handle);

struct PipelineTxn {
  PoolHandle state;
  std::uint32_t frame = 0;
  std::uint64_t sequence = 0;
};

std::ostream &operator<<(std::ostream &stream, const PipelineTxn &txn);

enum class MemoryOperation : std::uint8_t {
  kRead = 0,
  kWrite = 1,
};

enum class MemoryClient : std::uint8_t {
  kFramebuffer = 0,
  kMixedCache = 1,
  kTextureCache = 2,
  kUscL2 = 3,
  kTextureUpload = 4,
};

enum class MemoryPayloadFormat : std::uint8_t {
  kLinearBytes = 0,
  kCacheLineWrites = 1,
};

// FIFO-visible memory transaction. Bulk data remains in MemoryPool and only
// its generation-checked handle crosses a SystemC module boundary.
struct MemoryTxn {
  PipelineTxn pipeline;
  PoolHandle payload;
  std::uint64_t address = 0;
  std::uint64_t bytes = 0;
  // Monotonic lane/request identity is preserved on every cache/memory
  // response FIFO. Bulk request/response data remains in `payload`.
  std::uint64_t request_id = 0;
  MemoryOperation operation = MemoryOperation::kRead;
  MemoryClient client = MemoryClient::kFramebuffer;
  MemoryPayloadFormat payload_format = MemoryPayloadFormat::kLinearBytes;
  std::uint8_t reserved[5]{};
};

std::ostream &operator<<(std::ostream &stream, const MemoryTxn &txn);

struct CounterTxn {
  std::uint32_t frame = 0;
  std::uint64_t ia_vertices = 0;
  std::uint64_t ia_primitives = 0;
  std::uint64_t vs_invocations = 0;
  std::uint64_t c_invocations = 0;
  std::uint64_t c_primitives = 0;
  std::uint64_t ps_invocations = 0;
  std::uint64_t drawlists = 0;
  std::uint64_t setup_triangles = 0;
  std::uint64_t texel_fetches = 0;
  std::uint64_t virtual_gpu_cycles = 0;
  std::uint64_t tiler_cycles = 0;
  std::uint64_t renderer_cycles = 0;
  std::uint64_t usc_groups = 0;
  std::uint64_t texture_requests = 0;
  std::uint64_t fifo_stall_events = 0;
  std::uint64_t pool_bytes_in_flight = 0;
  std::uint64_t pool_high_water_bytes = 0;
  std::uint64_t vdm_cycles = 0;
  std::uint64_t vertex_fetch_cycles = 0;
  std::uint64_t vertex_attribute_fetches = 0;
  std::uint64_t vertex_attribute_bytes = 0;
  std::uint64_t pco_decode_cycles = 0;
  std::uint64_t pco_instructions = 0;
  std::uint64_t vs_alu_instructions = 0;
  std::uint64_t vs_tex_instructions = 0;
  std::uint64_t vs_memory_instructions = 0;
  std::uint64_t fs_alu_instructions = 0;
  std::uint64_t fs_tex_instructions = 0;
  std::uint64_t fs_memory_instructions = 0;
  std::uint64_t usc_slot_cycles = 0;
  std::uint64_t usc_cluster_cycles = 0;
  std::uint64_t clip_cull_cycles = 0;
  std::uint64_t tiler_bin_cycles = 0;
  std::uint64_t parameter_buffer_cycles = 0;
  std::uint64_t parameter_coefficient_sets = 0;
  std::uint64_t parameter_write_bytes = 0;
  std::uint64_t pds_coefficient_tasks = 0;
  std::uint64_t pds_douti_issues = 0;
  std::uint64_t usc_coefficient_load_bytes = 0;
  std::uint64_t tile_scheduler_cycles = 0;
  std::uint64_t isp_cycles = 0;
  std::uint64_t fragment_frontend_cycles = 0;
  std::uint64_t texture_cycles = 0;
  std::uint64_t pbe_cycles = 0;
  std::uint64_t pixel_data_master_transactions = 0;
  std::uint64_t pixel_data_master_bytes = 0;
  std::uint64_t pixel_data_master_cycles = 0;
  std::uint64_t tcu_line_accesses = 0;
  std::uint64_t tcu_read_accesses = 0;
  std::uint64_t tcu_hits = 0;
  std::uint64_t tcu_misses = 0;
  std::uint64_t tcu_evictions = 0;
  std::uint64_t tcu_writebacks = 0;
  std::uint64_t tcu_bypassed = 0;
  std::uint64_t tcu_cycles = 0;
  std::uint64_t slc_line_accesses = 0;
  std::uint64_t slc_read_accesses = 0;
  std::uint64_t slc_write_accesses = 0;
  std::uint64_t slc_hits = 0;
  std::uint64_t slc_misses = 0;
  std::uint64_t slc_evictions = 0;
  std::uint64_t slc_writebacks = 0;
  std::uint64_t slc_bypassed = 0;
  std::uint64_t slc_cycles = 0;
  std::uint64_t dram_read_transactions = 0;
  std::uint64_t dram_write_transactions = 0;
  std::uint64_t dram_read_bytes = 0;
  std::uint64_t dram_write_bytes = 0;
  std::uint64_t dram_cycles = 0;
  std::uint64_t framebuffer_dram_readback_bytes = 0;
  std::uint64_t tiles_binned = 0;
  std::uint64_t tiles_scheduled = 0;
  std::uint64_t covered_pixels = 0;
  std::uint64_t fragment_candidates = 0;
  std::uint64_t hsr_rejected_fragments = 0;
  std::uint64_t depth_tested_fragments = 0;
  std::uint64_t depth_rejected_fragments = 0;
  std::uint64_t depth_written_fragments = 0;
  std::uint64_t pbe_color_reads = 0;
  std::uint64_t pbe_blended_fragments = 0;
  std::uint64_t pbe_fragment_writes = 0;
  std::uint64_t pbe_pixels_written = 0;
  std::uint32_t functional_frame = 0;
};

std::ostream &operator<<(std::ostream &stream, const CounterTxn &txn);

} // namespace pvrgpu::stub
