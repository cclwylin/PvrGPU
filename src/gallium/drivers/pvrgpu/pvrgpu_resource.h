/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_RESOURCE_H
#define PVRGPU_RESOURCE_H

#include "pipe/p_state.h"

#include <stddef.h>
#include <stdint.h>

struct sw_displaytarget;

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
};

static inline struct pvrgpu_resource *
pvrgpu_resource(struct pipe_resource *resource)
{
   return (struct pvrgpu_resource *)resource;
}

void
pvrgpu_init_resource_functions(struct pipe_screen *screen);

void
pvrgpu_init_context_resource_functions(struct pipe_context *context);

#endif /* PVRGPU_RESOURCE_H */
