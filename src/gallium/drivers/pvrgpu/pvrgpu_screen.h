/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_SCREEN_H
#define PVRGPU_SCREEN_H

#include "pipe/p_screen.h"

struct sw_winsys;

struct pvrgpu_screen {
   struct pipe_screen base;
   struct sw_winsys *winsys;
};

static inline struct pvrgpu_screen *
pvrgpu_screen(struct pipe_screen *screen)
{
   return (struct pvrgpu_screen *)screen;
}

struct pipe_screen *
pvrgpu_create_screen(struct sw_winsys *winsys,
                     const struct pipe_screen_config *config);

/*
 * What this screen can hold, as one answer.
 *
 * These back `is_format_supported()`, and resource creation asks them too.
 * They used to be duplicated as a second, shorter list inside
 * `pvrgpu_resource.c`, so the screen would advertise a format that allocation
 * then refused -- `glTexImage2D(GL_RED)` answered `GL_OUT_OF_MEMORY` for a
 * format the screen had just called supported. One list cannot disagree with
 * itself.
 */
bool
pvrgpu_is_supported_color_format(enum pipe_format format);

bool
pvrgpu_is_sampler_only_format(enum pipe_format format);

bool
pvrgpu_is_supported_depth_stencil_format(enum pipe_format format);

#endif /* PVRGPU_SCREEN_H */
