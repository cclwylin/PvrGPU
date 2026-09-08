/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_VERTEX_FETCH_H
#define PVRGPU_VERTEX_FETCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

static inline uint32_t
pvrgpu_snapshot_index(const uint8_t *bytes, unsigned width)
{
   uint32_t value = 0;
   if (width == 1)
      value = *bytes;
   else if (width == 2) {
      uint16_t narrow;
      memcpy(&narrow, bytes, sizeof(narrow));
      value = narrow;
   } else
      memcpy(&value, bytes, sizeof(value));
   return value;
}

/* Rebase an owned, restart-free index snapshot onto its referenced range.
 * Signed baseVertex affects the source fetch, not the index's unsigned ABI.
 * Validate the whole range before changing any bytes or publishing outputs.
 * Primitive order/reuse and the source index width are unchanged. A shader
 * exposing gl_VertexID needs a separate original-ID input; this helper must
 * not be used to substitute a rebased value for that system value.
 */
static inline bool
pvrgpu_rebase_vertex_indices(void *data, size_t capacity, unsigned width,
                             uint32_t count, int32_t base_vertex,
                             uint32_t *first_vertex, uint32_t *vertex_count)
{
   if (!data || !first_vertex || !vertex_count || !count ||
       (width != 1 && width != 2 && width != 4) ||
       count > SIZE_MAX / width || (size_t)count * width > capacity)
      return false;
   uint8_t *bytes = (uint8_t *)data;
   uint32_t minimum = UINT32_MAX, maximum = 0;
   for (uint32_t i = 0; i < count; ++i) {
      const uint32_t index = pvrgpu_snapshot_index(bytes + (size_t)i * width, width);
      if (index < minimum) minimum = index;
      if (index > maximum) maximum = index;
   }
   const int64_t first = (int64_t)minimum + base_vertex;
   const int64_t last = (int64_t)maximum + base_vertex;
   const uint64_t extent = (uint64_t)maximum - minimum + 1;
   if (first < 0 || last > UINT32_MAX || extent > UINT32_MAX)
      return false;
   for (uint32_t i = 0; i < count; ++i) {
      uint8_t *destination = bytes + (size_t)i * width;
      const uint32_t index = pvrgpu_snapshot_index(destination, width) - minimum;
      if (width == 1)
         *destination = (uint8_t)index;
      else if (width == 2) {
         const uint16_t narrow = (uint16_t)index;
         memcpy(destination, &narrow, sizeof(narrow));
      } else
         memcpy(destination, &index, sizeof(index));
   }
   *first_vertex = (uint32_t)first;
   *vertex_count = (uint32_t)extent;
   return true;
}

#endif
