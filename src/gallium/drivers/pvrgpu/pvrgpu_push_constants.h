/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_PUSH_CONSTANTS_H
#define PVRGPU_PUSH_CONSTANTS_H

#include "pvrgpu_pco.h"
#include <string.h>

/* Compiler-private source-word maps are consumed before enqueue. The model
 * receives only the ordinary flat shared-register payload, not this map.
 * Preflight the complete map before modifying even one destination word. */
static inline bool
pvrgpu_copy_push_constant_words(
   const struct pvrgpu_pco_stage_abi *abi,
   const struct pvrgpu_pco_uniform_word_map *map,
   const uint8_t *source, size_t source_bytes,
   uint32_t *shared, size_t shared_capacity)
{
   if (!abi || !shared || abi->shareds > shared_capacity ||
       (uint64_t)abi->push_constant_start + abi->push_constant_count > abi->shareds)
      return false;

   if (map) {
      const size_t capacity = sizeof(map->source_words) / sizeof(map->source_words[0]);
      if (map->count > capacity)
         return false;
      if (map->count) {
         /* Keep the old CB0 rule: discard a partial DWORD, then round its
          * captured extent up to a vec4. Short complete-word tails remain
          * zero padded, just as on the uncompressed legacy path. */
         const size_t complete_words = source_bytes / sizeof(uint32_t);
         if (map->count != abi->push_constant_count ||
             !map->source_dwords || complete_words > UINT32_MAX - 3U ||
             map->source_dwords != ((complete_words + 3U) & ~(size_t)3U))
            return false;
         for (uint32_t word = 0; word < map->count; ++word) {
            if (map->source_words[word] >= map->source_dwords ||
                (word && map->source_words[word] <= map->source_words[word - 1]))
               return false;
         }
      } else if (map->source_dwords) {
         return false;
      }
      for (size_t word = map->count; word < capacity; ++word)
         if (map->source_words[word])
            return false;
   }
   if (!abi->push_constant_count)
      return true;
   if (!source)
      return false;

   for (uint32_t word = 0; word < abi->push_constant_count; ++word) {
      const uint32_t source_word = map && map->count ? map->source_words[word] : word;
      const uint64_t offset = (uint64_t)source_word * sizeof(uint32_t);
      uint32_t value = 0;
      if (offset <= source_bytes && source_bytes - (size_t)offset >= sizeof(value))
         memcpy(&value, source + (size_t)offset, sizeof(value));
      shared[abi->push_constant_start + word] = value;
   }
   return true;
}

#endif /* PVRGPU_PUSH_CONSTANTS_H */
