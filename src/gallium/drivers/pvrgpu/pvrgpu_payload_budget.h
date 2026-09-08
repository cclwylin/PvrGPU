/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_PAYLOAD_BUDGET_H
#define PVRGPU_PAYLOAD_BUDGET_H

#include "pvrgpu_systemc_api.h"
#include <stdbool.h>

static inline bool
pvrgpu_payload_add(uint64_t *total, uint64_t bytes)
{
   if (bytes > UINT64_MAX - *total)
      return false;
   *total += bytes;
   return true;
}

static inline bool
pvrgpu_payload_add_dwords(uint64_t *total, uint64_t count)
{
   return count <= UINT64_MAX / sizeof(uint32_t) &&
          pvrgpu_payload_add(total, count * sizeof(uint32_t));
}

/* Mirror CommandOwnedPayloadBytes after CopyPcoSequenceDraw/Texture. Count
 * each owned copy, even when views or bindings alias the same resource.
 * The bridge currently excludes index bytes and metadata vectors (clears,
 * varying/SO bindings); this is its transport budget, not all allocated heap.
 * Nested sequence draws do not load the legacy texture sidecar path. */
static inline bool
pvrgpu_pco_draw_payload_bytes(
   const struct pvrgpu_systemc_driver_command *draw,
   const struct pvrgpu_systemc_pco_sequence_texture *textures,
   unsigned texture_count, uint64_t *bytes)
{
   if (!draw || !bytes ||
       texture_count > PVRGPU_SYSTEMC_MAX_PCO_SEQUENCE_TEXTURES ||
       texture_count != draw->sampled_texture_count ||
       (texture_count && !textures) ||
       draw->uniform_buffer_count > 5u * PVRGPU_SYSTEMC_MAX_UNIFORM_BUFFERS_PER_STAGE ||
       (draw->uniform_buffer_count && !draw->uniform_buffers) ||
       draw->fragment_image_count > PVRGPU_SYSTEMC_MAX_SHADER_IMAGES ||
       (draw->fragment_image_count && !draw->fragment_images) ||
       (draw->stream_output &&
        (draw->stream_output->target_count > PVRGPU_SYSTEMC_MAX_STREAM_OUTPUT_BUFFERS ||
         (draw->stream_output->target_count && !draw->stream_output->targets))))
      return false;
   uint64_t total = 0;
   if (!pvrgpu_payload_add(&total, draw->raw_vertex_data_size) ||
       !pvrgpu_payload_add(&total, draw->vertex_pco_size) ||
       !pvrgpu_payload_add(&total, draw->fragment_pco_size) ||
       !pvrgpu_payload_add(&total, draw->geometry_pco_size) ||
       !pvrgpu_payload_add(&total, draw->initial_color_attachment_bytes_size) ||
       !pvrgpu_payload_add(&total, draw->initial_depth_attachment_bytes_size) ||
       !pvrgpu_payload_add_dwords(&total, draw->vertex_shared_count) ||
       !pvrgpu_payload_add_dwords(&total, draw->fragment_shared_count) ||
       (draw->geometry_pco_size &&
        !pvrgpu_payload_add_dwords(&total, draw->geometry_shared_count)) ||
       (draw->sampled_texture_count == 1 && draw->sampled_texture_bytes &&
        !pvrgpu_payload_add(&total, draw->sampled_texture_bytes_size)))
      return false;
   if (draw->tessellation) {
      const struct pvrgpu_systemc_tessellation *t = draw->tessellation;
      if (!pvrgpu_payload_add(&total, t->control_pco_size) ||
          !pvrgpu_payload_add(&total, t->evaluation_pco_size) ||
          !pvrgpu_payload_add_dwords(&total, t->control_shared_count) ||
          !pvrgpu_payload_add_dwords(&total, t->evaluation_shared_count))
         return false;
   }
   for (unsigned i = 0; i < texture_count; ++i) {
      if (textures[i].source == PVRGPU_SYSTEMC_PCO_TEXTURE_EXTERNAL_PAYLOAD &&
          !pvrgpu_payload_add(&total, textures[i].bytes_size))
         return false;
   }
   for (unsigned i = 0; i < draw->uniform_buffer_count; ++i)
      if (!pvrgpu_payload_add(&total, draw->uniform_buffers[i].bytes_size))
         return false;
   for (unsigned i = 0; i < draw->fragment_image_count; ++i)
      if (!pvrgpu_payload_add(&total, draw->fragment_images[i].bytes_size))
         return false;
   if (draw->stream_output)
      for (unsigned i = 0; i < draw->stream_output->target_count; ++i)
         if (!pvrgpu_payload_add(&total, draw->stream_output->targets[i].bytes_size))
            return false;
   *bytes = total;
   return true;
}

enum pvrgpu_payload_decision {
   PVRGPU_PAYLOAD_FITS,
   PVRGPU_PAYLOAD_FLUSH,
   PVRGPU_PAYLOAD_REJECT,
};

/* Production always supplies the shared 512 MiB limit. An explicit argument
 * lets isolated fixtures test boundaries without allocating half a GiB. */
static inline enum pvrgpu_payload_decision
pvrgpu_payload_decide(uint64_t previous, uint64_t candidate,
                      unsigned previous_draws, bool may_retry, uint64_t limit)
{
   if (previous > limit || candidate > limit)
      return PVRGPU_PAYLOAD_REJECT;
   if (candidate <= limit - previous)
      return PVRGPU_PAYLOAD_FITS;
   return previous_draws && may_retry ? PVRGPU_PAYLOAD_FLUSH : PVRGPU_PAYLOAD_REJECT;
}

static inline bool
pvrgpu_payload_flush_completed(uint64_t failures_before, uint64_t failures_after,
                               unsigned draws, unsigned pending,
                               bool local_emitted, bool global_emitted)
{
   return failures_before == failures_after && !draws && !pending &&
          !local_emitted && !global_emitted;
}

#endif
