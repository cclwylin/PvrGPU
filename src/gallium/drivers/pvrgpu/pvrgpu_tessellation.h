/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_TESSELLATION_H
#define PVRGPU_TESSELLATION_H
#include "pvrgpu_systemc_api.h"

/* Shared producer/consumer boundary validation, before copying owned bytes.
 * This checks declared resources, not a shader name or tessellated result. */
static inline const char *
pvrgpu_tessellation_payload_error_with_graphics(
   const struct pvrgpu_systemc_tessellation *t,
   const struct pvrgpu_systemc_graphics_shader_buffers *graphics)
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
      const struct pvrgpu_systemc_storage_buffer_abi *legacy_storage =
         stage ? &t->evaluation_storage : &t->control_storage;
      const struct pvrgpu_systemc_storage_buffer_abi *storage = graphics ?
         &graphics->storage[stage ?
            PVRGPU_SYSTEMC_PCO_SHADER_STAGE_TESS_EVALUATION :
            PVRGPU_SYSTEMC_PCO_SHADER_STAGE_TESS_CONTROL] : legacy_storage;
      const uint32_t *shared = stage ? t->evaluation_shared : t->control_shared;
      const uint32_t count = stage ? t->evaluation_shared_count : t->control_shared_count;
      const uint32_t descriptors = stage ? 4u : 8u;
      const uint32_t ubo_start = a->uniform_buffer_descriptor_start;
      const uint32_t ubo_end = ubo_start + 4u * a->uniform_buffer_descriptor_count;
      const uint32_t storage_start = storage->descriptor_count ?
         storage->descriptor_start : ubo_end;
      /* PCO may use aligned spare VTXIN words as writable registers. Only
       * the fixed 3/5-word system-input prefix is initialized by the task. */
      if (a->temps > 256 || a->vertex_inputs < (stage ? 5u : 3u) ||
          a->vertex_inputs > 64 ||
          (stage ? (a->vertex_outputs < 4 || a->vertex_outputs > 64) : a->vertex_outputs != 0) ||
          a->coefficients || a->entry_offset || !shared || count != a->shareds ||
          count < descriptors ||
          count > PVRGPU_SYSTEMC_MAX_PCO_GRAPHICS_SHARED_DWORDS_PER_STAGE ||
          a->uniform_buffer_descriptor_count > 15 ||
          ubo_start < descriptors || ubo_start > descriptors + 8u * 20u ||
          (ubo_start - descriptors) % 20u ||
          storage->descriptor_count > 32 ||
          (storage->descriptor_start && storage->descriptor_start != ubo_end) ||
          (storage->descriptor_count && storage->descriptor_start != ubo_end) ||
          (storage->descriptor_count < 32 &&
           (storage->used_mask >> storage->descriptor_count)) ||
          ((storage->read_mask | storage->write_mask) & ~storage->used_mask) ||
          (uint64_t)storage_start + 4u * storage->descriptor_count > count ||
          a->push_constant_start != storage_start +
             4u * storage->descriptor_count ||
          (uint64_t)a->push_constant_start + a->push_constant_count != count)
         return "tessellation native stage register/descriptor ABI";
      for (unsigned word = 0; word < descriptors; ++word)
         if (shared[word])
            return "tessellation patch descriptor must be unrelocated";
   }
   if (graphics &&
       (t->control_storage.descriptor_start ||
        t->control_storage.descriptor_count || t->control_storage.used_mask ||
        t->control_storage.read_mask || t->control_storage.write_mask ||
        t->evaluation_storage.descriptor_start ||
        t->evaluation_storage.descriptor_count ||
        t->evaluation_storage.used_mask || t->evaluation_storage.read_mask ||
        t->evaluation_storage.write_mask || t->buffer_resource_count ||
        t->buffer_resources || t->buffer_binding_count || t->buffer_bindings))
      return "tessellation storage must use only the outer graphics capsule";
   if (graphics)
      return NULL;
   if (t->buffer_resource_count > PVRGPU_SYSTEMC_MAX_SHADER_BUFFER_RESOURCES ||
       t->buffer_binding_count > PVRGPU_SYSTEMC_MAX_SHADER_BUFFER_BINDINGS ||
       ((t->buffer_resource_count != 0) != (t->buffer_resources != NULL)) ||
       ((t->buffer_binding_count != 0) != (t->buffer_bindings != NULL)))
      return "tessellation storage buffer count/pointer";
   uint64_t total_bytes = 0;
   for (unsigned resource = 0; resource < t->buffer_resource_count; ++resource) {
      const struct pvrgpu_systemc_shader_buffer_resource *r =
         &t->buffer_resources[resource];
      if (!r->resource_token || !r->bytes || !r->bytes_size ||
          r->bytes_size > PVRGPU_SYSTEMC_MAX_SHADER_BUFFER_BYTES - total_bytes)
         return "tessellation storage buffer backing resource";
      total_bytes += r->bytes_size;
      for (unsigned prior = 0; prior < resource; ++prior)
         if (t->buffer_resources[prior].resource_token == r->resource_token)
            return "tessellation storage buffer resource token is duplicated";
   }
   uint32_t present[2] = {0, 0};
   for (unsigned index = 0; index < t->buffer_binding_count; ++index) {
      const struct pvrgpu_systemc_shader_buffer_binding *b =
         &t->buffer_bindings[index];
      const unsigned stage = b->stage == PVRGPU_SYSTEMC_PCO_SHADER_STAGE_TESS_CONTROL ? 0 :
         b->stage == PVRGPU_SYSTEMC_PCO_SHADER_STAGE_TESS_EVALUATION ? 1 : 2;
      if (stage >= 2 || b->resource_index >= t->buffer_resource_count)
         return "tessellation storage buffer binding stage/resource";
      const struct pvrgpu_systemc_storage_buffer_abi *storage =
         stage ? &t->evaluation_storage : &t->control_storage;
      const struct pvrgpu_systemc_shader_buffer_resource *resource =
         &t->buffer_resources[b->resource_index];
      if (b->slot >= storage->descriptor_count || (b->access & ~3u) ||
          (b->offset & 3u) || !b->bytes_size || b->bytes_size > UINT32_MAX ||
          b->offset > resource->bytes_size ||
          b->bytes_size > resource->bytes_size - b->offset)
         return "tessellation storage buffer binding view";
      const uint32_t bit = UINT32_C(1) << b->slot;
      if (present[stage] & bit)
         return "tessellation storage buffer binding slot is duplicated";
      present[stage] |= bit;
      const unsigned required = ((storage->read_mask & bit) ? 1u : 0u) |
                                ((storage->write_mask & bit) ? 2u : 0u);
      if ((b->access & required) != required)
         return "tessellation storage buffer binding access";
      const uint32_t *shared = stage ? t->evaluation_shared : t->control_shared;
      const unsigned word = storage->descriptor_start + 4u * b->slot;
      if (shared[word] || shared[word + 1] ||
          shared[word + 2] != b->bytes_size || shared[word + 3])
         return "tessellation storage buffer descriptor must be canonical";
   }
   for (unsigned stage = 0; stage < 2; ++stage) {
      const struct pvrgpu_systemc_storage_buffer_abi *storage =
         stage ? &t->evaluation_storage : &t->control_storage;
      const uint32_t *shared = stage ? t->evaluation_shared : t->control_shared;
      if (storage->used_mask & ~present[stage])
         return "tessellation shader uses an unbound storage buffer";
      for (unsigned slot = 0; slot < storage->descriptor_count; ++slot) {
         if (present[stage] & (UINT32_C(1) << slot))
            continue;
         const unsigned word = storage->descriptor_start + 4u * slot;
         if (shared[word] || shared[word + 1] || shared[word + 2] || shared[word + 3])
            return "unused tessellation storage buffer descriptor must be zero";
      }
   }
   return NULL;
}

static inline const char *
pvrgpu_tessellation_payload_error(const struct pvrgpu_systemc_tessellation *t)
{
   return pvrgpu_tessellation_payload_error_with_graphics(t, NULL);
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
