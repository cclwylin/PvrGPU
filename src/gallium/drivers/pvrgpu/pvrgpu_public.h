/* SPDX-License-Identifier: MIT */
#ifndef PVRGPU_PUBLIC_H
#define PVRGPU_PUBLIC_H

#include "pipe/p_screen.h"

#ifdef __cplusplus
extern "C" {
#endif

struct sw_winsys;

struct pipe_screen *
pvrgpu_create_screen(struct sw_winsys *winsys,
                     const struct pipe_screen_config *config);

#ifdef __cplusplus
}
#endif

#endif /* PVRGPU_PUBLIC_H */
