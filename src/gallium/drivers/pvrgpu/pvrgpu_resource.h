/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_RESOURCE_H
#define PVRGPU_RESOURCE_H

#include "pipe/p_state.h"

#include <stddef.h>
#include <stdint.h>

struct pvrgpu_resource {
   struct pipe_resource base;
   uint8_t *data;
   unsigned stride;
   uintptr_t layer_stride;
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
