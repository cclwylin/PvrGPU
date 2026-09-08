/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_POINT_RESTART_H
#define PVRGPU_POINT_RESTART_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* POINTS has no connectivity to reset. Removing restart markers is therefore
 * exact input assembly, unlike deleting markers from strips/fans/lists.
 * Compare unsigned source indices before any base-vertex or instance bias.
 * This operates only on an owned snapshot; the application buffer is never
 * changed. Output count/max describe retained points, not restart markers.
 */
static inline bool
pvrgpu_compact_point_restart_indices(void *data, size_t capacity_bytes,
                                     unsigned index_size, unsigned input_count,
                                     uint32_t restart_index,
                                     unsigned *output_count,
                                     uint32_t *maximum_index)
{
   if (!output_count || !maximum_index ||
       (index_size != 1 && index_size != 2 && index_size != 4) ||
       input_count > SIZE_MAX / index_size ||
       (size_t)input_count * index_size > capacity_bytes ||
       (input_count && !data))
      return false;
   uint8_t *bytes = (uint8_t *)data;
   unsigned retained = 0;
   uint32_t maximum = 0;
   for (unsigned i = 0; i < input_count; ++i) {
      uint32_t index = 0;
      const uint8_t *source = bytes + (size_t)i * index_size;
      if (index_size == 1)
         index = *source;
      else if (index_size == 2) {
         uint16_t value;
         memcpy(&value, source, sizeof(value));
         index = value;
      } else
         memcpy(&index, source, sizeof(index));
      if (index == restart_index)
         continue;
      memmove(bytes + (size_t)retained * index_size, source, index_size);
      ++retained;
      if (index > maximum)
         maximum = index;
   }
   *output_count = retained;
   *maximum_index = maximum;
   return true;
}

#endif
