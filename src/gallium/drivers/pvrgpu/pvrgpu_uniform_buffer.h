/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_UNIFORM_BUFFER_H
#define PVRGPU_UNIFORM_BUFFER_H

#include "pvrgpu_resource.h"
#include "pvrgpu_systemc_api.h"
#include "pipe/p_state.h"

#include <stdlib.h>
#include <string.h>

/* Generic VS/FS compilation may prove an unused UBO suffix under register
 * pressure. Its ABI retains the original indices of every remaining block.
 * Other compilation paths still require the complete declared inventory. */
static inline bool
pvrgpu_uniform_buffer_prefix_count_valid(unsigned declared_blocks,
                                          unsigned compiled_blocks,
                                          bool allow_unused_suffix)
{
   return declared_blocks <= PVRGPU_SYSTEMC_MAX_UNIFORM_BUFFERS_PER_STAGE &&
      compiled_blocks <= declared_blocks &&
      (allow_unused_suffix || compiled_blocks == declared_blocks);
}

/* Snapshot one already-selected Gallium constant-buffer range. The stage's
 * NIR block i is bound at CB[i+1]; CB0 remains the default uniform block.
 * This copies inputs, never evaluates shader instructions or repacks layout.
 * An unbound slot is a descriptor hole, not a fabricated zero-valued buffer.
 * The caller owns entry->bytes until its draw record is retired. */
static inline bool
pvrgpu_snapshot_uniform_buffer(const struct pipe_constant_buffer *binding,
                                uint32_t stage, uint32_t block_index,
                                struct pvrgpu_systemc_pco_uniform_buffer *entry,
                                uint32_t descriptor[4])
{
   if (!binding || !entry || !descriptor ||
       stage > PVRGPU_SYSTEMC_PCO_SHADER_STAGE_TESS_EVALUATION ||
       block_index >= PVRGPU_SYSTEMC_MAX_UNIFORM_BUFFERS_PER_STAGE)
      return false;
   memset(entry, 0, sizeof(*entry));
   memset(descriptor, 0, 4 * sizeof(*descriptor));
   if (binding->buffer_size == 0 ||
       (!binding->buffer && !binding->user_buffer))
      return true;

   const uint8_t *source;
   size_t available = binding->buffer_size;
   if (binding->user_buffer) {
      /* As in Mesa pipe_upload_constant_buffer0 and this driver's CB0 path,
       * user_buffer already names the upload range start. buffer_offset is
       * used only for resource-backed storage, not applied a second time. */
      source = binding->user_buffer;
   } else {
      const struct pvrgpu_resource *resource =
         (const struct pvrgpu_resource *)binding->buffer;
      if (binding->buffer->target != PIPE_BUFFER || !resource->data ||
          binding->buffer_offset >= resource->size)
         return false;
      source = resource->data + binding->buffer_offset;
      const size_t remaining = resource->size - binding->buffer_offset;
      if (available > remaining)
         available = remaining;
   }
   if (available == 0 || available > PVRGPU_SYSTEMC_MAX_UNIFORM_BUFFER_BYTES)
      return false;
   uint8_t *snapshot = malloc(available);
   if (!snapshot)
      return false;
   memcpy(snapshot, source, available);
   entry->stage = stage;
   entry->block_index = block_index;
   entry->bytes = snapshot;
   entry->bytes_size = available;
   /* Model submission relocates the base address into its own DRAM. The
    * snapshot already starts at the bound range, so dynamic offset is zero. */
   descriptor[2] = (uint32_t)available;
   return true;
}

/* CB0's push-constant contract discards an incomplete trailing DWORD and
 * zero-pads complete words to a vec4.  Preserve those exact bytes when a
 * large CB0 is represented by a UBO descriptor; ordinary UBO snapshots above
 * continue to expose their exact bound range without padding. */
static inline bool
pvrgpu_snapshot_cb0_uniform_buffer(
   const struct pipe_constant_buffer *binding, uint32_t stage,
   uint32_t block_index, uint32_t uniform_dwords,
   struct pvrgpu_systemc_pco_uniform_buffer *entry, uint32_t descriptor[4])
{
   struct pvrgpu_systemc_pco_uniform_buffer source;
   uint32_t source_descriptor[4];
   if (!uniform_dwords ||
       uniform_dwords > PVRGPU_SYSTEMC_MAX_UNIFORM_BUFFER_BYTES / 4u ||
       !pvrgpu_snapshot_uniform_buffer(binding, stage, block_index,
                                       &source, source_descriptor) ||
       !source.bytes)
      return false;
   const size_t bytes_size = (size_t)uniform_dwords * sizeof(uint32_t);
   uint8_t *bytes = calloc(1, bytes_size);
   if (!bytes) {
      free((void *)source.bytes);
      return false;
   }
   const size_t complete_source_bytes =
      source.bytes_size / sizeof(uint32_t) * sizeof(uint32_t);
   const size_t copied = complete_source_bytes < bytes_size
      ? complete_source_bytes : bytes_size;
   memcpy(bytes, source.bytes, copied);
   free((void *)source.bytes);
   *entry = (struct pvrgpu_systemc_pco_uniform_buffer){
      .stage = stage,
      .block_index = block_index,
      .bytes = bytes,
      .bytes_size = bytes_size,
   };
   memset(descriptor, 0, 4 * sizeof(*descriptor));
   descriptor[2] = (uint32_t)bytes_size;
   return true;
}

/* The active native extent normally selects CB[1..active_blocks], not the
 * context's highest ever bound slot. Preserve holes so block indices remain
 * stable. A nonzero mapping replaces only the final native slot with CB0. */
static inline bool
pvrgpu_snapshot_stage_uniform_buffers_mapped(
   const struct pipe_constant_buffer *bindings, uint32_t stage,
   uint32_t active_blocks, uint32_t descriptor_start,
   uint32_t cb0_uniform_buffer_slot,
   uint32_t cb0_uniform_dwords,
   uint32_t *shared, size_t shared_count,
   struct pvrgpu_systemc_pco_uniform_buffer *entries,
   unsigned *entry_count, unsigned entry_capacity)
{
   if (!bindings || !entries || !entry_count ||
       stage > PVRGPU_SYSTEMC_PCO_SHADER_STAGE_TESS_EVALUATION ||
       active_blocks > PVRGPU_SYSTEMC_MAX_UNIFORM_BUFFERS_PER_STAGE ||
       (cb0_uniform_buffer_slot &&
        cb0_uniform_buffer_slot != active_blocks) ||
       (!!cb0_uniform_buffer_slot != !!cb0_uniform_dwords) ||
       (uint64_t)descriptor_start + 4u * active_blocks > shared_count ||
       (shared_count && !shared) || *entry_count > entry_capacity ||
       active_blocks > entry_capacity - *entry_count)
      return false;
   for (unsigned block = 0; block < active_blocks; ++block) {
      struct pvrgpu_systemc_pco_uniform_buffer snapshot;
      /* Native UBO slots normally mirror Gallium CB[1..].  A large CB0 is
       * appended as the final native descriptor, preserving every public UBO
       * index while reusing the same owned per-draw byte snapshot contract. */
      const unsigned gallium_slot =
         cb0_uniform_buffer_slot == block + 1 ? 0 : block + 1;
      const bool captured = gallium_slot == 0
         ? pvrgpu_snapshot_cb0_uniform_buffer(&bindings[0], stage, block,
              cb0_uniform_dwords, &snapshot,
              shared + descriptor_start + 4u * block)
         : pvrgpu_snapshot_uniform_buffer(&bindings[gallium_slot], stage, block,
              &snapshot, shared + descriptor_start + 4u * block);
      if (!captured)
         return false;
      if (snapshot.bytes)
         entries[(*entry_count)++] = snapshot;
   }
   return true;
}

static inline bool
pvrgpu_snapshot_stage_uniform_buffers(
   const struct pipe_constant_buffer *bindings, uint32_t stage,
   uint32_t active_blocks, uint32_t descriptor_start,
   uint32_t *shared, size_t shared_count,
   struct pvrgpu_systemc_pco_uniform_buffer *entries,
   unsigned *entry_count, unsigned entry_capacity)
{
   return pvrgpu_snapshot_stage_uniform_buffers_mapped(
      bindings, stage, active_blocks, descriptor_start, 0, 0,
      shared, shared_count, entries, entry_count, entry_capacity);
}

static inline void
pvrgpu_finish_uniform_buffer_snapshots(
   struct pvrgpu_systemc_pco_uniform_buffer *entries, unsigned *entry_count)
{
   for (unsigned i = 0; i < *entry_count; ++i) {
      free((void *)entries[i].bytes);
      memset(&entries[i], 0, sizeof(entries[i]));
   }
   *entry_count = 0;
}

#endif /* PVRGPU_UNIFORM_BUFFER_H */
