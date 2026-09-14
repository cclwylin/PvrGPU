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

/* Synchronous work that failed must not be represented by a signaled NULL
 * fence. This immutable token has process lifetime and never completes. */
struct pipe_fence_handle *
pvrgpu_failed_fence(void);

struct pipe_fence_handle *
pvrgpu_signaled_fence(void);

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

/* Formats whose typed image load/store lowering and snapshot transport agree.
 * Keep screen capability queries and the PCO compiler on this single list. */
static inline bool
pvrgpu_is_supported_shader_image_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R32_UINT:
   case PIPE_FORMAT_R32_SINT:
   case PIPE_FORMAT_R32_FLOAT:
   case PIPE_FORMAT_R32G32_UINT:
   case PIPE_FORMAT_R32G32_SINT:
   case PIPE_FORMAT_R32G32_FLOAT:
   case PIPE_FORMAT_R32G32B32A32_UINT:
   case PIPE_FORMAT_R32G32B32A32_SINT:
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
   case PIPE_FORMAT_R16G16B16A16_UINT:
   case PIPE_FORMAT_R16G16B16A16_SINT:
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
   case PIPE_FORMAT_R16G16B16A16_UNORM:
   case PIPE_FORMAT_R16G16B16A16_SNORM:
   case PIPE_FORMAT_R16G16_UINT:
   case PIPE_FORMAT_R16G16_SINT:
   case PIPE_FORMAT_R16G16_FLOAT:
   case PIPE_FORMAT_R16G16_UNORM:
   case PIPE_FORMAT_R16G16_SNORM:
   case PIPE_FORMAT_R8G8B8A8_UINT:
   case PIPE_FORMAT_R8G8B8A8_SINT:
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_SNORM:
   case PIPE_FORMAT_R10G10B10A2_UINT:
   case PIPE_FORMAT_R10G10B10A2_UNORM:
   case PIPE_FORMAT_R11G11B10_FLOAT:
      return true;
   default:
      return false;
   }
}

bool
pvrgpu_is_supported_depth_stencil_format(enum pipe_format format);

#endif /* PVRGPU_SCREEN_H */
