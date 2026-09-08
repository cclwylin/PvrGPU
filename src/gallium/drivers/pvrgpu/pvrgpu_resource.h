/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_RESOURCE_H
#define PVRGPU_RESOURCE_H

#include "pipe/p_state.h"

#include <stddef.h>
#include <stdint.h>

struct sw_displaytarget;
struct pipe_context;
struct pvrgpu_context;

struct pvrgpu_resource {
   struct pipe_resource base;
   uint8_t *data;
   struct sw_displaytarget *displaytarget;
   unsigned displaytarget_stride;
   unsigned stride;
   uintptr_t layer_stride;
   unsigned level_strides[PIPE_MAX_TEXTURE_LEVELS];
   uintptr_t level_layer_strides[PIPE_MAX_TEXTURE_LEVELS];
   uintptr_t level_offsets[PIPE_MAX_TEXTURE_LEVELS];
   unsigned level_count;
   size_t size;
   /*
    * The driver has put content in `data` that the model cannot reproduce.
    *
    * A scissored or channel-masked clear is written here by the CPU and is not
    * describable by the v1 `clear_color` capsule, so the model never sees it.
    * Its framebuffer therefore no longer describes this surface, and a readback
    * that copied it back would erase the region the application just cleared.
    * A full-surface RGBA clear does not set this: the model starts a sequence
    * from the same uniform state, so its output still agrees.
    */
   bool driver_writes_model_cannot_reproduce;
};

/* Validate a complete row-padded, layer-major surface span before either
 * snapshot or readback touches backing memory. No partial write is allowed. */
static inline bool
pvrgpu_surface_span(const struct pvrgpu_resource *resource,
                    const struct pipe_surface *surface, size_t row_bytes,
                    unsigned height, unsigned layer_count, size_t *offset)
{
   if (!resource || !surface || !offset || !row_bytes || !height || !layer_count ||
       layer_count > 256 || surface->level >= resource->level_count ||
       surface->level >= PIPE_MAX_TEXTURE_LEVELS ||
       surface->first_layer > surface->last_layer ||
       layer_count > (unsigned)surface->last_layer - surface->first_layer + 1)
      return false;
   const size_t base = resource->level_offsets[surface->level];
   const size_t stride = resource->level_strides[surface->level];
   const size_t layer_stride = resource->level_layer_strides[surface->level];
   const size_t last_layer = (size_t)surface->first_layer + layer_count - 1;
   if (base > resource->size || !layer_stride || stride < row_bytes ||
       row_bytes > layer_stride || (size_t)(height - 1) > (layer_stride - row_bytes) / stride ||
       last_layer > (resource->size - base) / layer_stride)
      return false;
   const size_t last_offset = base + last_layer * layer_stride;
   if ((size_t)(height - 1) > (resource->size - last_offset) / stride ||
       row_bytes > resource->size - last_offset - (size_t)(height - 1) * stride)
      return false;
   *offset = base + (size_t)surface->first_layer * layer_stride;
   return true;
}

static inline struct pvrgpu_resource *
pvrgpu_resource(struct pipe_resource *resource)
{
   return (struct pvrgpu_resource *)resource;
}

void
pvrgpu_init_resource_functions(struct pipe_screen *screen);

void
pvrgpu_init_context_resource_functions(struct pipe_context *context);

/*
 * Materialize a pending generic draw sequence into the backing stores of the
 * framebuffer's currently bound colour surfaces.  Framebuffer changes and
 * CPU-side blits use this before they can make the current model framebuffer
 * refer to a different surface.
 */
void
pvrgpu_flush_current_color_attachments(struct pipe_context *context);

/* Call only after a draw command for the bound framebuffer was submitted. */
void
pvrgpu_note_current_color_readback_pending(struct pvrgpu_context *context);

#endif /* PVRGPU_RESOURCE_H */
