#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace pvrgpu::stub {

inline constexpr const char *kSchema = "pvrgpu.counter.v1";

enum class MemoryMode : std::uint8_t {
  // Fast functional mode. GPU clients access the authoritative DRAM backing
  // directly and do not claim cache or DRAM timing/counter provenance.
  kDirect = 0,
  // Cache controllers are bypassed, but accesses still reach DRAM and retain
  // transaction/latency accounting.
  kBypass = 1,
  // Full SLC tag/data, write-back/write-allocate and DRAM miss/writeback path.
  kCache = 2,
};

const char *MemoryModeName(MemoryMode mode);

struct DriverPcoStageAbi {
  std::uint32_t temps = 0;
  std::uint32_t vertex_inputs = 0;
  std::uint32_t vertex_outputs = 0;
  std::uint32_t coefficients = 0;
  std::uint32_t shareds = 0;
  std::uint32_t push_constant_start = 0;
  std::uint32_t push_constant_count = 0;
  std::uint32_t entry_offset = 0;
};

struct DriverPcoTopologyExpansion {
  std::vector<std::uint8_t> vertices;
  std::uint32_t input_primitives = 0;
  std::uint32_t emitted_primitives = 0;
  std::uint32_t duplicate_position_primitives = 0;
};

// A sequence texture is either an immutable byte payload captured from the
// replay or an attachment produced by an earlier physical draw.  Refract uses
// the latter two sources for its prepass color/depth images, so those images
// cannot be replaced by CPU/golden sidecars at the API boundary.
enum class DriverPcoTextureSource : std::uint8_t {
  kExternalPayload = 0,
  kPreviousColorAttachment = 1,
  kPreviousDepthAttachment = 2,
};

enum class DriverPcoShaderStage : std::uint8_t {
  kVertex = 0,
  kFragment = 1,
};

struct DriverPcoTextureMipLayout {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t row_pitch_bytes = 0;
  std::uint32_t offset_bytes = 0;
};

// Owned descriptor-backed resource for a native PCO sequence.  The raw Rogue
// image/sampler descriptor dwords remain in fragment_shared; this redundant
// structured metadata bounds every byte copy and lets Submitter/TextureUnit
// fail closed if descriptor-set routing, mip layout, or sampler state drifts.
struct DriverPcoSampledTexture {
  DriverPcoTextureSource source = DriverPcoTextureSource::kExternalPayload;
  DriverPcoShaderStage stage = DriverPcoShaderStage::kVertex;
  std::uint32_t producer_command_index = 0;
  std::uint32_t descriptor_set = 0;
  std::uint32_t binding = 0;
  std::string format;
  std::vector<std::uint8_t> bytes;
  std::uint64_t declared_bytes_size = 0;
  std::uint32_t mip_count = 0;
  std::array<DriverPcoTextureMipLayout, 10> mip{};
  std::uint32_t min_filter = 0;
  std::uint32_t mag_filter = 0;
  std::uint32_t mip_filter = 0;
  std::uint32_t wrap_u = 0;
  std::uint32_t wrap_v = 0;
  std::uint32_t normalized_coordinates = 0;
  std::uint32_t min_lod_u4_6 = 0;
  std::uint32_t max_lod_u4_6 = 0;
};

inline constexpr DriverPcoStageAbi kConditionalsVertexPcoAbi = {
    10, 4, 4, 0, 16, 0, 16, 0};
inline constexpr DriverPcoStageAbi kConditionalsFragmentPcoAbi = {
    4, 0, 0, 0, 4, 0, 4, 0};
inline constexpr std::uint32_t kDriverPcoPositionVertexStride = 12;
inline constexpr std::uint32_t kDriverPcoPositionNormalVertexStride = 24;
inline constexpr std::uint32_t kDriverPcoPositionNormalTexcoordVertexStride =
    32;
// One public PCO command may link several vec4 coefficient slots.  Sixteen is
// deliberately below both the 64-dword VTXOUT bank (after four position
// dwords) and the public coefficient-bank bound, while covering Ideas' 10
// scalar smooth-varying components.
inline constexpr std::uint32_t kDriverPcoMaximumVaryingComponents = 16;
inline constexpr std::size_t kDriverPcoIdeasSequenceCommands = 180;
inline constexpr std::size_t kDriverPcoIdeasDepthEnabledFirstCommand = 162;
inline constexpr std::size_t kDriverPcoIdeasDepthEnabledEndCommand = 168;
inline constexpr std::uint32_t kDriverPcoMaximumBinaryBytes =
    16U * 1024U * 1024U;
inline constexpr std::size_t kDriverPcoMaximumSequenceCommands = 4096;
inline constexpr std::size_t kDriverPcoMaximumNestedSequenceCommands = 64;
inline constexpr std::size_t kDriverPcoMaximumSequenceTextures = 16;
inline constexpr std::uint64_t kDriverPcoMaximumSequencePayloadBytes =
    UINT64_C(512) * 1024U * 1024U;
inline constexpr std::uint32_t kDriverPcoTextureWidth = 512;
inline constexpr std::uint32_t kDriverPcoTextureHeight = 512;
inline constexpr std::uint32_t kDriverPcoTextureRowPitch = 2048;
inline constexpr std::uint64_t kDriverPcoTextureBytes =
    UINT64_C(1048576);
// Mesa 26.2.1 pipe_format values transported verbatim by the v7 API.
inline constexpr std::uint32_t kDriverPcoDepthFormatZ16Unorm = 268;
inline constexpr std::uint32_t kDriverPcoDepthFormatZ32Unorm = 270;
inline constexpr std::uint32_t kDriverPcoDepthFormatZ24X8Unorm = 276;
inline constexpr std::uint32_t kDriverPcoNewAttachment = UINT32_MAX;
inline constexpr std::size_t kDriverPcoRefractSequenceCommands = 2;
inline constexpr std::size_t kDriverPcoRefractSampledTextures = 3;
inline constexpr std::size_t kDriverPcoMaximumTextureMipLevels = 10;
inline constexpr std::uint64_t kDriverPcoSequenceAttachmentStride =
    UINT64_C(0x01000000);
inline constexpr std::uint64_t kDriverPcoSequenceColorAddressBase =
    UINT64_C(0x50000000);
inline constexpr std::uint64_t kDriverPcoSequenceDepthAddressBase =
    UINT64_C(0x60000000);
inline constexpr std::uint64_t kDriverPcoSequenceExternalAddressBase =
    UINT64_C(0x70000000);
inline constexpr std::uint64_t kDriverPcoRefractVertexFnv1a64 =
    UINT64_C(0x83920b2733098afa);
inline constexpr std::uint64_t kDriverPcoRefractPrepassSharedFnv1a64 =
    UINT64_C(0x1b26b797d92ed099);
inline constexpr std::uint64_t kDriverPcoRefractCompositeSharedFnv1a64 =
    UINT64_C(0xd5db135e7933d92d);
inline constexpr std::uint64_t kDriverPcoRefractFragmentSharedFnv1a64 =
    UINT64_C(0x26536c76cbc158b5);
inline constexpr std::uint64_t kDriverPcoRefractExternalFnv1a64 =
    UINT64_C(0xecfb10435885d2ce);
inline constexpr std::uint64_t kDriverPcoRefractPrepassVertexPcoFnv1a64 =
    UINT64_C(0x6e9ad97e49eca9fe);
inline constexpr std::uint64_t kDriverPcoRefractPrepassFragmentPcoFnv1a64 =
    UINT64_C(0xa55a28d91b0f4b9e);
inline constexpr std::uint64_t kDriverPcoRefractCompositeVertexPcoFnv1a64 =
    UINT64_C(0xc46a9af088bfe8a9);
inline constexpr std::uint64_t kDriverPcoRefractCompositeFragmentPcoFnv1a64 =
    UINT64_C(0x8fe8ae5903f3c2dd);

constexpr bool DriverPcoStageAbiMatches(const DriverPcoStageAbi &actual,
                                        const DriverPcoStageAbi &expected) {
  return actual.temps == expected.temps &&
         actual.vertex_inputs == expected.vertex_inputs &&
         actual.vertex_outputs == expected.vertex_outputs &&
         actual.coefficients == expected.coefficients &&
         actual.shareds == expected.shareds &&
         actual.push_constant_start == expected.push_constant_start &&
         actual.push_constant_count == expected.push_constant_count &&
         actual.entry_offset == expected.entry_offset;
}

struct DriverCommand {
  bool enabled = false;
  std::string schema;
  std::string producer;
  std::string command;
  std::string test_case;
  std::uint32_t frame = 1;
  std::uint32_t framebuffer_width = 0;
  std::uint32_t framebuffer_height = 0;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::string format;
  std::array<std::uint32_t, 4> clear_color_bits{};
  std::array<std::array<std::uint32_t, 2>, 6> vertex_bits{};
  std::array<std::array<std::uint32_t, 2>, 6> texcoord_bits{};
  std::array<std::uint32_t, 4> fragment_color_bits{};
  std::uint32_t texture_width = 0;
  std::uint32_t texture_height = 0;
  std::string texture_rgba8_path;
  std::vector<std::uint8_t> texture_rgba8_bytes;
  // `draw_pco_triangles` owns all producer memory before the API call
  // returns.  These vectors therefore remain valid until the deferred
  // SystemC run at process exit, unlike Gallium's transient pointers.
  std::vector<std::uint8_t> raw_vertex_data;
  std::vector<std::uint8_t> vertex_pco;
  std::vector<std::uint8_t> fragment_pco;
  std::vector<std::uint32_t> vertex_shared;
  std::vector<std::uint32_t> fragment_shared;
  std::uint32_t sampled_texture_count = 0;
  std::vector<std::uint8_t> sampled_texture_bytes;
  std::uint64_t declared_sampled_texture_bytes_size = 0;
  std::uint32_t sampled_texture_width = 0;
  std::uint32_t sampled_texture_height = 0;
  std::uint32_t sampled_texture_row_pitch = 0;
  std::string sampled_texture_format;
  std::uint32_t sampled_texture_mip_count = 0;
  // Multi-resource state is reserved for an explicitly validated native PCO
  // sequence.  Existing draw_pco_triangles profiles continue to use the
  // singular fields above, preserving their public ABI and strict fixtures.
  std::vector<DriverPcoSampledTexture> sampled_textures;
  std::uint32_t vertex_sampled_texture_count = 0;
  std::uint32_t fragment_sampled_texture_count = 0;
  std::uint64_t declared_raw_vertex_data_size = 0;
  std::uint64_t declared_vertex_pco_size = 0;
  std::uint64_t declared_fragment_pco_size = 0;
  std::uint32_t vertex_stride = 0;
  std::uint32_t vertex_count = 0;
  std::uint32_t first_vertex = 0;
  std::uint32_t instance_count = 0;
  std::uint32_t primitive_mode = 0;
  std::uint32_t indexed = 0;
  DriverPcoStageAbi vertex_pco_abi;
  DriverPcoStageAbi fragment_pco_abi;
  std::uint32_t position_output_start = 0;
  std::uint32_t position_output_count = 0;
  std::uint32_t fragment_position_start = 0;
  std::uint32_t fragment_position_count = 0;
  // Generic smooth-varying linkage. The vertex range is expressed in VTXOUT
  // dwords; the fragment range is expressed in coefficient dwords. A zero
  // count on both sides is the legacy conditionals profile.
  std::uint32_t varying_output_start = 0;
  std::uint32_t varying_output_count = 0;
  std::uint32_t fragment_varying_start = 0;
  std::uint32_t fragment_varying_count = 0;
  std::array<std::uint32_t, 3> viewport_scale_bits{};
  std::array<std::uint32_t, 3> viewport_translate_bits{};
  std::uint32_t front_ccw = 0;
  std::uint32_t cull_face = 0;
  std::uint32_t fill_front = 0;
  std::uint32_t fill_back = 0;
  std::uint32_t scissor = 0;
  std::uint32_t rasterizer_discard = 0;
  std::uint32_t multisample = 0;
  std::uint32_t half_pixel_center = 0;
  std::uint32_t bottom_edge_rule = 0;
  std::uint32_t clip_halfz = 0;
  std::uint32_t depth_clip_near = 0;
  std::uint32_t depth_clip_far = 0;
  std::uint32_t depth_clamp = 0;
  std::uint32_t sample_mask = 0;
  std::uint32_t color_mask = 0;
  std::uint32_t blend_enable = 0;
  std::uint32_t blend_rgb_equation = 0;
  std::uint32_t blend_alpha_equation = 0;
  std::uint32_t blend_source_rgb_factor = 1;
  std::uint32_t blend_destination_rgb_factor = 0;
  std::uint32_t blend_source_alpha_factor = 1;
  std::uint32_t blend_destination_alpha_factor = 0;
  std::uint32_t dither = 0;
  std::uint32_t depth_enable = 0;
  std::uint32_t depth_write = 0;
  std::uint32_t depth_func = 0;
  std::uint32_t depth_clear_bits = 0;
  std::uint32_t depth_format = 0;
  std::uint32_t color_attachment_source_command_index =
      kDriverPcoNewAttachment;
  std::uint32_t depth_attachment_source_command_index =
      kDriverPcoNewAttachment;
  std::uint32_t draw_count = 0;
  std::uint32_t index_count = 0;
  std::uint32_t unique_vertices = 0;
  std::uint32_t primitive_count = 0;
  std::uint32_t clip_primitives = 0;
  std::uint32_t setup_triangles = 0;
  std::uint64_t semantic_texel_fetches = 0;
  std::uint32_t ia_vertices = 0;
  std::uint32_t ia_primitives = 0;
  std::uint32_t vs_invocations = 0;
  std::uint32_t gs_invocations = 0;
  std::uint32_t gs_primitives = 0;
  std::uint32_t clip_invocations = 0;
  std::uint64_t ps_invocations = 0;
  std::uint32_t hs_invocations = 0;
  std::uint32_t ds_invocations = 0;
  std::uint32_t cs_invocations = 0;
  std::string framebuffer_rgba8_path;
};

// Converts a validated non-indexed triangle strip/fan to the model's direct
// triangle-list occurrence stream. Legal duplicate-position degenerates are
// retained so fixed-function clipping/setup can account for them; all records
// and strip winding are preserved.
DriverPcoTopologyExpansion
ExpandDriverPcoTopology(const DriverCommand &command);

struct Options {
  unsigned frames = 5;
  unsigned width = 512;
  unsigned height = 512;
  std::string test_case = "fill_solid";
  std::string output_dir;
  MemoryMode memory_mode = MemoryMode::kCache;
  // Legacy compatibility mirror for --cache-bypass and existing reports.
  // It is true only for kBypass; kDirect is identified by memory_mode.
  bool cache_bypass = false;
  std::string driver_command_path;
  DriverCommand driver_command;
  // The bridge fills this only when one replay submits more than one native
  // draw_pco_triangles command.  `driver_command` remains the logical report
  // command/counter-metadata carrier, while every entry below is an owned
  // physical draw that must traverse the SystemC pipeline in order.
  std::vector<DriverCommand> driver_commands;
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
  kIndexFetch = 5,
  kVertexFetch = 6,
  kParameterWrite = 7,
  kParameterRead = 8,
  kFramebufferReadback = 9,
  kTextureMipmap = 10,
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
  std::uint64_t gs_invocations = 0;
  std::uint64_t gs_primitives = 0;
  std::uint64_t c_invocations = 0;
  std::uint64_t c_primitives = 0;
  std::uint64_t ps_invocations = 0;
  std::uint64_t hs_invocations = 0;
  std::uint64_t ds_invocations = 0;
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
  std::uint64_t memory_direct_read_bytes = 0;
  std::uint64_t memory_direct_write_bytes = 0;
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
