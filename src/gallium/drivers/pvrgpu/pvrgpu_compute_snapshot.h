/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_COMPUTE_SNAPSHOT_H
#define PVRGPU_COMPUTE_SNAPSHOT_H

#include "pvrgpu_resource.h"
#include "pvrgpu_systemc_compute_api.h"
#include "util/u_inlines.h"

#include <stdlib.h>
#include <string.h>

/* A dispatch owns one copy of each BO, not one copy of each binding. Views
 * into the same BO must observe one another's stores and atomic operations.
 * Host pointers here only transport bytes; shader work happens in the model. */
struct pvrgpu_compute_snapshot {
   struct pvrgpu_systemc_compute_resource
      resources[PVRGPU_SYSTEMC_COMPUTE_MAX_RESOURCES];
   struct pipe_resource *owners[PVRGPU_SYSTEMC_COMPUTE_MAX_RESOURCES];
   struct pvrgpu_systemc_compute_binding
      bindings[PVRGPU_SYSTEMC_COMPUTE_MAX_BINDINGS];
   size_t resource_count;
   size_t binding_count;
   struct pvrgpu_systemc_compute_image_binding
      images[PVRGPU_SYSTEMC_COMPUTE_MAX_IMAGES];
   size_t image_count;
};

static inline void
pvrgpu_compute_snapshot_finish(struct pvrgpu_compute_snapshot *snapshot)
{
   for (size_t i = 0; i < snapshot->resource_count; ++i) {
      free(snapshot->resources[i].bytes);
      pipe_resource_reference(&snapshot->owners[i], NULL);
   }
   memset(snapshot, 0, sizeof(*snapshot));
}

static inline bool
pvrgpu_compute_snapshot_add(struct pvrgpu_compute_snapshot *snapshot,
                             uint32_t kind, uint32_t slot, uint32_t access,
                             struct pipe_resource *buffer,
                             const void *user_buffer,
                             uint64_t offset, uint64_t bytes_size,
                             const char **reason)
{
   if (!snapshot || !reason)
      return false;
   *reason = "compute_binding_range";
   if (snapshot->binding_count >= PVRGPU_SYSTEMC_COMPUTE_MAX_BINDINGS ||
       (kind != PVRGPU_SYSTEMC_COMPUTE_UNIFORM_BUFFER &&
        kind != PVRGPU_SYSTEMC_COMPUTE_STORAGE_BUFFER) ||
       (access & ~(PVRGPU_SYSTEMC_COMPUTE_ACCESS_READ |
                   PVRGPU_SYSTEMC_COMPUTE_ACCESS_WRITE)) ||
       (kind == PVRGPU_SYSTEMC_COMPUTE_UNIFORM_BUFFER &&
        (access != PVRGPU_SYSTEMC_COMPUTE_ACCESS_READ ||
         slot >= PVRGPU_SYSTEMC_MAX_UNIFORM_BUFFERS_PER_STAGE ||
         bytes_size > PVRGPU_SYSTEMC_MAX_UNIFORM_BUFFER_BYTES)) ||
       (kind == PVRGPU_SYSTEMC_COMPUTE_STORAGE_BUFFER &&
        (user_buffer || slot >= PIPE_MAX_SHADER_BUFFERS)) ||
       bytes_size > PVRGPU_SYSTEMC_COMPUTE_MAX_RESOURCE_BYTES ||
       !bytes_size || (!buffer && !user_buffer))
      return false;

   size_t index = snapshot->resource_count;
   const uint8_t *source;
   size_t source_size;
   if (user_buffer) {
      /* Mesa user constant buffers already point at their bound range start.
       * As in the graphics CB0/UBO path, do not apply buffer_offset twice. */
      source = user_buffer;
      source_size = bytes_size;
      offset = 0;
   } else {
      const struct pvrgpu_resource *resource = pvrgpu_resource(buffer);
      if (buffer->target != PIPE_BUFFER || !resource->data ||
          resource->size < buffer->width0)
         return false;
      source = resource->data;
      source_size = buffer->width0;
      if (offset >= source_size)
         return false;
      if (bytes_size > source_size - offset)
         bytes_size = source_size - offset;
      for (size_t i = 0; i < snapshot->resource_count; ++i) {
         if (snapshot->owners[i] == buffer) {
            index = i;
            break;
         }
      }
   }
   if (!source_size ||
       source_size > PVRGPU_SYSTEMC_COMPUTE_MAX_RESOURCE_BYTES ||
       offset > source_size || bytes_size > source_size - offset)
      return false;
   if (index == snapshot->resource_count) {
      if (index >= PVRGPU_SYSTEMC_COMPUTE_MAX_RESOURCES) {
         *reason = "compute_resource_count";
         return false;
      }
      /* Match the API's aggregate global-memory limit before allocating a
       * new copy. Aliased views add no storage and are not counted twice. */
      size_t total = source_size;
      for (size_t i = 0; i < snapshot->resource_count; ++i) {
         if (snapshot->resources[i].bytes_size >
             PVRGPU_SYSTEMC_COMPUTE_MAX_RESOURCE_BYTES - total) {
            *reason = "compute_snapshot_total_bytes";
            return false;
         }
         total += snapshot->resources[i].bytes_size;
      }
      uint8_t *copy = malloc(source_size);
      if (!copy) {
         *reason = "compute_snapshot_allocation";
         return false;
      }
      memcpy(copy, source, source_size);
      snapshot->resources[index] =
         (struct pvrgpu_systemc_compute_resource){copy, source_size};
      if (!user_buffer)
         pipe_resource_reference(&snapshot->owners[index], buffer);
      ++snapshot->resource_count;
   }
   snapshot->bindings[snapshot->binding_count++] =
      (struct pvrgpu_systemc_compute_binding){
         .kind = kind, .slot = slot, .resource_index = index,
         .access = access, .offset = offset, .bytes_size = bytes_size,
      };
   *reason = NULL;
   return true;
}

static inline bool
pvrgpu_compute_snapshot_add_image(struct pvrgpu_compute_snapshot *snapshot,
                                  uint32_t slot, uint32_t access,
                                  const struct pipe_image_view *image,
                                  const char **reason)
{
   if (!snapshot || !reason) return false;
   *reason = "compute_image_view";
   if (!image || !image->resource || slot >= PVRGPU_SYSTEMC_COMPUTE_MAX_IMAGES ||
       snapshot->image_count >= PVRGPU_SYSTEMC_COMPUTE_MAX_IMAGES ||
       (access & ~3U) || (image->access & access) != access ||
       image->format != PIPE_FORMAT_R32_UINT ||
       image->resource->target != PIPE_TEXTURE_2D ||
       image->resource->format != PIPE_FORMAT_R32_UINT ||
       !image->resource->width0 || !image->resource->height0 ||
       image->resource->depth0 != 1 || image->resource->array_size != 1 ||
       image->resource->nr_samples > 1 || image->resource->nr_storage_samples > 1 ||
       image->u.tex.first_layer != 0 || image->u.tex.last_layer != 0)
      return false;
   struct pvrgpu_resource *resource = pvrgpu_resource(image->resource);
   const unsigned level = image->u.tex.level;
   if (!resource->data || !resource->size ||
       resource->size > PVRGPU_SYSTEMC_COMPUTE_MAX_RESOURCE_BYTES ||
       level >= resource->level_count || level > image->resource->last_level ||
       level >= PIPE_MAX_TEXTURE_LEVELS || level >= 32)
      return false;
   const unsigned width = MAX2(1U, image->resource->width0 >> level);
   const unsigned height = MAX2(1U, image->resource->height0 >> level);
   const uint64_t offset = resource->level_offsets[level];
   const uint32_t stride = resource->level_strides[level];
   if (width > UINT32_MAX / 4U || stride < width * 4U || (stride & 3U) ||
       (offset & 3U) || offset > resource->size)
      return false;
   const uint64_t extent = (uint64_t)(height - 1) * stride + (uint64_t)width * 4;
   if (extent > resource->size - offset || extent > UINT32_MAX ||
       extent > resource->level_layer_strides[level] ||
       resource->level_layer_strides[level] > resource->size - offset)
      return false;
   size_t index = snapshot->resource_count;
   for (size_t i = 0; i < snapshot->resource_count; ++i) {
      if (snapshot->owners[i] == image->resource) {
         index = i;
         if (snapshot->resources[i].bytes_size != resource->size) return false;
         break;
      }
   }
   if (index == snapshot->resource_count) {
      if (index >= PVRGPU_SYSTEMC_COMPUTE_MAX_RESOURCES) return false;
      size_t total = resource->size;
      for (size_t i = 0; i < snapshot->resource_count; ++i) {
         if (snapshot->resources[i].bytes_size >
             PVRGPU_SYSTEMC_COMPUTE_MAX_RESOURCE_BYTES - total) return false;
         total += snapshot->resources[i].bytes_size;
      }
      uint8_t *copy = malloc(resource->size);
      if (!copy) return false;
      memcpy(copy, resource->data, resource->size);
      snapshot->resources[index] =
         (struct pvrgpu_systemc_compute_resource){copy, resource->size};
      pipe_resource_reference(&snapshot->owners[index], image->resource);
      ++snapshot->resource_count;
   }
   snapshot->images[snapshot->image_count++] =
      (struct pvrgpu_systemc_compute_image_binding){
         .slot = slot, .resource_index = index, .access = access,
         .format = PVRGPU_SYSTEMC_COMPUTE_IMAGE_R32UI,
         .offset = offset, .bytes_size = extent,
         .width = width, .height = height, .row_stride_bytes = stride,
      };
   *reason = NULL;
   return true;
}

/* Call only after a successful synchronous dispatch. Even if a bridge alters
 * another byte in its private snapshot, a read-only view or an unbound range
 * is never copied over the original BO. Overlapping views share final bytes. */
static inline void
pvrgpu_compute_snapshot_writeback(struct pvrgpu_compute_snapshot *snapshot)
{
   for (size_t i = 0; i < snapshot->binding_count; ++i) {
      const struct pvrgpu_systemc_compute_binding *binding =
         &snapshot->bindings[i];
      if (!(binding->access & PVRGPU_SYSTEMC_COMPUTE_ACCESS_WRITE))
         continue;
      struct pvrgpu_resource *resource =
         pvrgpu_resource(snapshot->owners[binding->resource_index]);
      memcpy(resource->data + binding->offset,
             snapshot->resources[binding->resource_index].bytes + binding->offset,
             binding->bytes_size);
   }
   for (size_t i = 0; i < snapshot->image_count; ++i) {
      const struct pvrgpu_systemc_compute_image_binding *image = &snapshot->images[i];
      if (!(image->access & PVRGPU_SYSTEMC_COMPUTE_ACCESS_WRITE)) continue;
      struct pvrgpu_resource *resource = pvrgpu_resource(snapshot->owners[image->resource_index]);
      for (unsigned row = 0; row < image->height; ++row) {
         const size_t offset = image->offset + (size_t)row * image->row_stride_bytes;
         memcpy(resource->data + offset,
                snapshot->resources[image->resource_index].bytes + offset,
                (size_t)image->width * 4);
      }
      // The separate graphics framebuffer snapshot predates these genuine
      // model writes. Its cached readback must not overwrite compute results.
      resource->driver_writes_model_cannot_reproduce = true;
   }
}

#endif /* PVRGPU_COMPUTE_SNAPSHOT_H */
