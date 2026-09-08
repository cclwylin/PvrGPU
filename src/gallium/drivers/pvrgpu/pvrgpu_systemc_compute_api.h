/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_SYSTEMC_COMPUTE_API_H
#define PVRGPU_SYSTEMC_COMPUTE_API_H

#include "pvrgpu_systemc_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Independent, synchronous compute ABI. No graphics command or framebuffer
 * carries compute results. Check version before reading any later field. */
#define PVRGPU_SYSTEMC_COMPUTE_API_VERSION 4u
#define PVRGPU_SYSTEMC_COMPUTE_MAX_SHARED_BYTES (32u * 1024u)
#define PVRGPU_SYSTEMC_COMPUTE_MAX_IMAGES 32u
#define PVRGPU_SYSTEMC_COMPUTE_IMAGE_DESCRIPTOR_DWORDS 8u
#define PVRGPU_SYSTEMC_COMPUTE_IMAGE_R32UI 1u
#define PVRGPU_SYSTEMC_COMPUTE_MAX_RESOURCES 64u
#define PVRGPU_SYSTEMC_COMPUTE_MAX_BINDINGS 47u
#define PVRGPU_SYSTEMC_COMPUTE_MAX_BINARY_BYTES (1024u * 1024u)
#define PVRGPU_SYSTEMC_COMPUTE_MAX_RESOURCE_BYTES (1024u * 1024u * 1024u)

struct pvrgpu_systemc_compute_abi {
   struct pvrgpu_systemc_pco_stage_abi stage;
   uint32_t local_size[3];
   uint32_t local_invocation_index_start;
   uint32_t local_invocation_index_count;
   uint32_t workgroup_id_start;
   uint32_t workgroup_id_count;
   uint32_t num_workgroups_start;
   uint32_t num_workgroups_count;
   uint32_t storage_buffer_descriptor_start;
   uint32_t storage_buffer_descriptor_count;
   uint32_t uniform_buffer_used_mask;
   uint32_t storage_buffer_used_mask;
   uint32_t storage_buffer_read_mask;
   uint32_t storage_buffer_write_mask;
   uint32_t shared_memory_bytes;
   uint32_t scratch_bytes;
   /* Canonical SH layout: UBO4, SSBO4, image8 per slot, optional private
    * workgroup4, then CB0. Private count is four iff shared bytes is nonzero. */
   uint32_t shared_memory_descriptor_start;
   uint32_t shared_memory_descriptor_count;
   uint32_t image_descriptor_start;
   uint32_t image_descriptor_count;
   uint32_t image_used_mask;
   uint32_t image_read_mask;
   uint32_t image_write_mask;
};

enum pvrgpu_systemc_compute_binding_kind {
   PVRGPU_SYSTEMC_COMPUTE_UNIFORM_BUFFER = 0,
   PVRGPU_SYSTEMC_COMPUTE_STORAGE_BUFFER = 1,
};

#define PVRGPU_SYSTEMC_COMPUTE_ACCESS_READ 1u
#define PVRGPU_SYSTEMC_COMPUTE_ACCESS_WRITE 2u

/* Exactly one entry per backing BO, even when several binding views alias it.
 * All pointers are borrowed only for the call. The bridge snapshots every
 * resource before execution and copies writable results back only on success.
 * Bytes outside written views must be preserved. */
struct pvrgpu_systemc_compute_resource {
   uint8_t *bytes;
   size_t bytes_size;
};

struct pvrgpu_systemc_compute_binding {
   uint32_t kind;
   uint32_t slot;
   uint32_t resource_index;
   uint32_t access;
   uint64_t offset;
   uint64_t bytes_size;
};

/* Separate image binding namespace, sharing resources[] backing ownership.
 * One linear R32UI image2D view. Byte offsets include the selected mip/layer;
 * dimensions exclude row padding. The compiler computes actual texel addresses.
 * Image descriptor: baseLo/baseHi/extent/0, width/height/row_stride/format. */
struct pvrgpu_systemc_compute_image_binding {
   uint32_t slot;
   uint32_t resource_index;
   uint32_t access;
   uint32_t format;
   uint64_t offset;
   uint64_t bytes_size;
   uint32_t width;
   uint32_t height;
   uint32_t row_stride_bytes;
   uint32_t reserved;
};

struct pvrgpu_systemc_compute_dispatch {
   uint32_t version;
   struct pvrgpu_systemc_compute_abi abi;
   const uint8_t *binary;
   size_t binary_size;
   uint32_t grid[3];
   uint32_t block[3];
   const uint32_t *push_words;
   size_t push_word_count;
   struct pvrgpu_systemc_compute_resource *resources;
   size_t resource_count;
   const struct pvrgpu_systemc_compute_binding *bindings;
   size_t binding_count;
   /* 0=direct, 1=DRAM bypass, 2=SLC cache; fixed after session elaboration. */
   uint32_t memory_mode;
   const struct pvrgpu_systemc_compute_image_binding *images;
   size_t image_count;
};

struct pvrgpu_systemc_compute_stats {
   uint64_t workgroups;
   uint64_t invocations;
   uint64_t alu_instructions;
   uint64_t memory_instructions;
   uint64_t atomic_instructions;
   uint64_t load_instructions;
   uint64_t store_instructions;
   uint64_t dram_read_bytes;
   uint64_t dram_write_bytes;
   uint64_t direct_read_bytes;
   uint64_t direct_write_bytes;
   uint64_t readback_bytes;
   uint64_t pool_allocations;
   uint64_t pool_releases;
};

typedef int (*pvrgpu_systemc_submit_compute_fn)(
   const struct pvrgpu_systemc_compute_dispatch *dispatch,
   struct pvrgpu_systemc_compute_stats *stats,
   char *error, size_t error_size);

int pvrgpu_systemc_submit_compute(
   const struct pvrgpu_systemc_compute_dispatch *dispatch,
   struct pvrgpu_systemc_compute_stats *stats,
   char *error, size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
