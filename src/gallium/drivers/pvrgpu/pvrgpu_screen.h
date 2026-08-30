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

#endif /* PVRGPU_SCREEN_H */
