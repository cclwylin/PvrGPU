/* SPDX-License-Identifier: MIT */
/* Resolve arithmetic adapted from Mesa's
 * src/gallium/auxiliary/util/u_simple_shaders.c.
 *
 * Copyright 2008 VMware, Inc.
 * All Rights Reserved.
 * Copyright 2009 Marek Olšák <maraeo@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#ifndef PVRGPU_MSAA_H
#define PVRGPU_MSAA_H

#include <stddef.h>
#include <string.h>

/* CPU texture backing is pixel-interleaved: every pixel owns consecutive
 * sample texels, then the next pixel follows. Row/layer strides include all
 * samples. Keep the addressing shared by copies, clears and model readback. */
static inline size_t
pvrgpu_msaa_texel_index(unsigned x, unsigned sample, unsigned sample_count)
{
   return (size_t)x * sample_count + sample;
}

/* Like Mesa's util_blitter MSAA resolve used by llvmpipe, non-integer colors
 * are resolved by averaging all sample values after format unpacking. Integer
 * colors instead select one actual sample; callers copy sample zero directly
 * so integer bits never pass through floating-point arithmetic. Match
 * util_make_fs_msaa_resolve's +0, ascending-sample ADDs, then reciprocal MUL;
 * callers must supply a nonzero sample count. The supported sample counts
 * are powers of two, whose reciprocals are exactly representable. */
static inline void
pvrgpu_msaa_resolve_float(const void *samples,
                          unsigned sample_count,
                          size_t sample_stride,
                          float result[4])
{
   float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
   for (unsigned sample = 0; sample < sample_count; ++sample) {
      float value[4];
      memcpy(value, (const unsigned char *)samples + sample * sample_stride,
             sizeof(value));
      for (unsigned channel = 0; channel < 4; ++channel)
         sums[channel] += value[channel];
   }
   const float inverse_samples = 1.0f / (float)sample_count;
   for (unsigned channel = 0; channel < 4; ++channel)
      result[channel] = sums[channel] * inverse_samples;
}

#endif /* PVRGPU_MSAA_H */
