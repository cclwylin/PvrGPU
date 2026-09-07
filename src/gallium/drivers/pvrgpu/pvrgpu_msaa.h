/* SPDX-License-Identifier: MIT */
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
 * so integer bits never pass through floating-point arithmetic. */
static inline void
pvrgpu_msaa_resolve_float(const void *samples,
                          unsigned sample_count,
                          size_t sample_stride,
                          float result[4])
{
   float sums[4];
   memcpy(sums, samples, sizeof(sums));
   for (unsigned sample = 1; sample < sample_count; ++sample) {
      float value[4];
      memcpy(value, (const unsigned char *)samples + sample * sample_stride,
             sizeof(value));
      for (unsigned channel = 0; channel < 4; ++channel)
         sums[channel] += value[channel];
   }
   for (unsigned channel = 0; channel < 4; ++channel)
      result[channel] = sums[channel] / (float)sample_count;
}

#endif /* PVRGPU_MSAA_H */
