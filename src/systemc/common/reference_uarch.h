// PvrGPU Reference uArch v1.
//
// uArch means microarchitecture. Public API/PCO semantics remain normative;
// internal widths, queues, service batches and latencies below are one explicit
// engineering assumption set selected for continued performance-model work.
// They are not claims about a particular commercial PowerVR BVNC. Keeping the
// assumptions in one versioned object lets later measurements recalibrate
// timing without changing functional payloads or SystemC module connections.
#pragma once

#include <cstdint>

namespace pvrgpu::stub {

struct ReferenceUarchConfig {
  std::uint32_t version;
  std::uint32_t fifo_depth;
  std::uint32_t tile_width;
  std::uint32_t tile_height;
  std::uint32_t subpixel_bits;
  std::uint32_t sample_count;
  std::uint32_t usc_issue_lanes;
  std::uint32_t fragment_quad_width;
  std::uint32_t fragment_quad_height;
  // Assumed indexed-front-end segmentation/cache geometry.  Segments end on
  // complete triangles and reset the direct-mapped post-transform cache.
  std::uint32_t index_segment_max_indices;
  std::uint32_t post_transform_cache_slots;

  std::uint32_t vdm_base_cycles;
  std::uint32_t vdm_vertices_per_batch;
  std::uint32_t vertex_fetch_base_cycles;
  std::uint32_t vertex_fetch_bytes_per_batch;
  std::uint32_t pco_decode_base_cycles;
  std::uint32_t pco_groups_per_decode_batch;
  std::uint32_t usc_slot_base_cycles;
  std::uint32_t usc_groups_per_slot_batch;
  std::uint32_t usc_cluster_base_cycles;
  std::uint32_t usc_groups_per_cluster_batch;
  std::uint32_t clip_base_cycles;
  std::uint32_t clip_primitives_per_batch;
  std::uint32_t tiler_base_cycles;
  std::uint32_t tiler_triangles_per_batch;
  std::uint32_t parameter_base_cycles;
  std::uint32_t parameter_triangles_per_batch;
  std::uint32_t scheduler_base_cycles;
  std::uint32_t scheduler_tiles_per_batch;
  std::uint32_t isp_base_cycles;
  std::uint32_t isp_candidates_per_batch;
  std::uint32_t fragment_frontend_base_cycles;
  std::uint32_t fragment_lanes_per_batch;
  std::uint32_t texture_bypass_cycles;
  std::uint32_t pbe_base_cycles;
  std::uint32_t pbe_pixels_per_batch;
  std::uint32_t pbe_blend_fragments_per_batch;
  std::uint32_t fixed_submission_cycles;
};

inline constexpr ReferenceUarchConfig kReferenceUarch = {
    1,     // version
    4,     // FIFO depth
    32,    // tile width
    32,    // tile height
    8,     // subpixel fractional bits
    1,     // samples per pixel
    4,     // USC issue lanes
    2,  2, // fragment quad width/height
    1023,  // max uint16 index occurrences per complete-triangle segment
    256,   // direct-mapped post-transform vertex-cache slots

    8,  512, // VDM
    4,  2048, // vertex fetch bytes (256 tightly packed float2 elements)
    3,  4,   // PCO decode
    2,  2,   // USC slot issue
    4,  4,   // USC cluster
    6,  128, // clip/cull
    10, 64,  // tiler
    5,  128, // parameter buffer
    8,  4,   // tile scheduler
    10, 64,  // ISP
    5,  128, // fragment frontend
    2,       // texture bypass
    12, 128, // PBE surface store
    128,     // PBE blend destination read/modify/write
    25,      // fixed submission overhead
};

static_assert(kReferenceUarch.tile_width == 32 &&
              kReferenceUarch.tile_height == 32);
static_assert(kReferenceUarch.usc_issue_lanes == 4);
static_assert(kReferenceUarch.fragment_quad_width == 2 &&
              kReferenceUarch.fragment_quad_height == 2);
static_assert(kReferenceUarch.index_segment_max_indices % 3 == 0);
static_assert(kReferenceUarch.post_transform_cache_slots != 0);
static_assert(kReferenceUarch.vertex_fetch_bytes_per_batch != 0);

} // namespace pvrgpu::stub
