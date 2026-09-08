/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_INDIRECT_DRAW_H
#define PVRGPU_INDIRECT_DRAW_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Only command decoding happens on the CPU. The resulting ordinary draw must
 * still use the driver's native PCO and SystemC pipeline, including stream
 * output. Callers must complete preceding resource producers before reading
 * this buffer; mapping an arbitrary PIPE_BUFFER does not imply such a flush.
 */
struct pvrgpu_indirect_draw {
   uint32_t count;
   uint32_t instance_count;
   uint32_t first;
   int32_t base_vertex;
   uint32_t base_instance;
};

static inline bool
pvrgpu_decode_indirect_draw(const void *buffer, size_t buffer_size,
                            size_t offset, size_t stride, unsigned draw_count,
                            unsigned index_size, bool has_indirect_draw_count,
                            bool count_from_stream_output,
                            struct pvrgpu_indirect_draw *draw,
                            const char **reason)
{
   const char *ignored = NULL;
   if (!reason)
      reason = &ignored;
   *reason = NULL;
   if (!draw || !buffer) {
      *reason = "indirect_buffer_missing";
      return false;
   }
   if (draw_count != 1 || has_indirect_draw_count || count_from_stream_output) {
      *reason = "indirect_variant";
      return false;
   }
   if (index_size != 0 && index_size != 1 && index_size != 2 && index_size != 4) {
      *reason = "indirect_index_size";
      return false;
   }
   const size_t command_size = index_size ? 20u : 16u;
   if ((offset & 3u) || (stride & 3u) ||
       (stride != 0 && stride < command_size)) {
      *reason = "indirect_alignment_or_stride";
      return false;
   }
   /* Subtraction bounds avoid overflowing offset + command_size. A single
    * command consumes exactly its ABI size, never a padded stride's tail.
    * memcpy also permits an unaligned allocation despite an aligned offset.
    */
   if (offset > buffer_size || command_size > buffer_size - offset) {
      *reason = "indirect_buffer_bounds";
      return false;
   }
   uint32_t words[5] = {0};
   memcpy(words, (const uint8_t *)buffer + offset, command_size);
   struct pvrgpu_indirect_draw decoded;
   decoded.count = words[0];
   decoded.instance_count = words[1];
   decoded.first = words[2];
   decoded.base_vertex = 0;
   decoded.base_instance = words[index_size ? 4 : 3];
   /* Preserve signed baseVertex without implementation-defined uint32 casts.
    * Direct indexed lowering decides whether the referenced vertices fit its
    * supported range; the decoder must not reinterpret -1 as UINT32_MAX.
    */
   if (index_size)
      memcpy(&decoded.base_vertex, &words[3], sizeof(decoded.base_vertex));
   /* The current direct input packer uses uint32 first/instance indices.
    * Refuse wraparound before they reach its address arithmetic. Checking
    * every instance is conservative for divisors > 1, but never wraps a
    * legal large baseInstance onto the start of an attribute buffer.
    * Empty work must not inspect or require any vertex/instance range.
    */
   if (decoded.count && decoded.instance_count &&
       (decoded.count - 1 > UINT32_MAX - decoded.first ||
        decoded.instance_count - 1 > UINT32_MAX - decoded.base_instance)) {
      *reason = "indirect_vertex_or_instance_range";
      return false;
   }
   *draw = decoded;
   return true;
}

static inline bool
pvrgpu_indirect_draw_is_empty(const struct pvrgpu_indirect_draw *draw)
{
   return draw && (draw->count == 0 || draw->instance_count == 0);
}

#endif
