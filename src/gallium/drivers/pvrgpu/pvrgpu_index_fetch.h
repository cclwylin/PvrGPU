/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_INDEX_FETCH_H
#define PVRGPU_INDEX_FETCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Resource-backed element fetch. Like Mesa draw's DRAW_GET_IDX, an element
 * outside the complete elements in the buffer supplies raw index zero. The
 * caller still applies baseVertex/restart and submits the original draw.
 * This is a bounds policy, not a shader result or a successful no-op.
 * An invalid type/missing backing remains an error; a partial trailing index
 * is never read. Division avoids overflow even for arbitrary 64-bit positions.
 */
static inline bool
pvrgpu_fetch_bounded_index(const void *data, size_t size_bytes,
                           unsigned index_size, uint64_t position,
                           uint32_t *out_index)
{
   if (!out_index || !data ||
       (index_size != 1 && index_size != 2 && index_size != 4))
      return false;
   uint32_t value = 0;
   if (position < size_bytes / index_size) {
      const uint8_t *source = (const uint8_t *)data + (size_t)position * index_size;
      if (index_size == 1)
         value = *source;
      else if (index_size == 2) {
         uint16_t narrow;
         memcpy(&narrow, source, sizeof(narrow));
         value = narrow;
      } else
         memcpy(&value, source, sizeof(value));
   }
   *out_index = value;
   return true;
}

#endif
