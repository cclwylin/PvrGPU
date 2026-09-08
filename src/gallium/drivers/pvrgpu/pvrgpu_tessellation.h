/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_TESSELLATION_H
#define PVRGPU_TESSELLATION_H
#include "pvrgpu_systemc_api.h"

/* Shared producer/consumer boundary validation, before copying owned bytes.
 * This checks declared resources, not a shader name or tessellated result. */
static inline const char *
pvrgpu_tessellation_payload_error(const struct pvrgpu_systemc_tessellation *t)
{
   if (!t)
      return "missing tessellation payload";
   if (!t->control_pco || !t->control_pco_size || t->control_pco_size > (1u << 24) ||
       !t->evaluation_pco || !t->evaluation_pco_size || t->evaluation_pco_size > (1u << 24))
      return "tessellation executable size/pointer";
   if (!t->input_vertices || t->input_vertices > 32 ||
       !t->output_vertices || t->output_vertices > 32 || !t->vertices_per_instance)
      return "tessellation patch vertex extent";
   if (t->input_stride_dwords < 4 || t->input_stride_dwords > 64 ||
       t->output_vertex_stride_dwords > 64 || t->per_vertex_offset_dwords < 6 ||
       t->per_vertex_offset_dwords > 134 ||
       t->patch_stride_dwords != t->per_vertex_offset_dwords +
          t->output_vertices * t->output_vertex_stride_dwords)
      return "tessellation patch storage layout";
   if (t->domain > 2 || t->spacing > 2 || t->clockwise > 1 || t->point_mode > 1)
      return "tessellation domain/spacing/winding/point mode";
   for (unsigned stage = 0; stage < 2; ++stage) {
      const struct pvrgpu_systemc_pco_stage_abi *a = stage ? &t->evaluation_abi : &t->control_abi;
      const uint32_t *shared = stage ? t->evaluation_shared : t->control_shared;
      const uint32_t count = stage ? t->evaluation_shared_count : t->control_shared_count;
      const uint32_t descriptors = stage ? 4u : 8u;
      /* PCO may use aligned spare VTXIN words as writable registers. Only
       * the fixed 3/5-word system-input prefix is initialized by the task. */
      if (a->temps > 256 || a->vertex_inputs < (stage ? 5u : 3u) ||
          a->vertex_inputs > 64 ||
          (stage ? (a->vertex_outputs < 4 || a->vertex_outputs > 64) : a->vertex_outputs != 0) ||
          a->coefficients || a->entry_offset || !shared || count != a->shareds ||
          count < descriptors || count > 256 ||
          a->uniform_buffer_descriptor_count > 15 ||
          a->uniform_buffer_descriptor_start != descriptors ||
          a->push_constant_start != descriptors + 4u * a->uniform_buffer_descriptor_count ||
          (uint64_t)a->push_constant_start + a->push_constant_count != count)
         return "tessellation native stage register/descriptor ABI";
      for (unsigned word = 0; word < descriptors; ++word)
         if (shared[word])
            return "tessellation patch descriptor must be unrelocated";
   }
   return NULL;
}

/* Call after the payload validator, before copying/allocating input storage. */
static inline const char *
pvrgpu_tessellation_draw_extent_error(const struct pvrgpu_systemc_tessellation *t,
                                      uint32_t occurrences)
{
   if (occurrences > PVRGPU_SYSTEMC_MAX_TESSELLATION_INPUT_VERTICES)
      return "tessellation input occurrence count exceeds bounded draw storage";
   if ((uint64_t)(occurrences / t->vertices_per_instance) *
          (t->vertices_per_instance / t->input_vertices) > PVRGPU_SYSTEMC_MAX_TESSELLATION_PATCHES)
      return "tessellation patch count exceeds bounded draw storage";
   return NULL;
}
#endif
