/*
 * Copyright 2023 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 *
 * Standard sample-position tables adapted from Mesa
 * src/gallium/auxiliary/util/u_sample_positions.c. The 4x and 8x positions
 * also match llvmpipe's lp_rast.c tables. Values below are exact sixteenths
 * of a pixel, so coverage never rounds a floating-point sample position.
 */
#ifndef PVRGPU_SYSTEMC_COMMON_MSAA_H
#define PVRGPU_SYSTEMC_COMMON_MSAA_H

#include <array>
#include <cstdint>
#include <stdexcept>

namespace pvrgpu::stub {

inline bool IsSupportedRasterSampleCount(std::uint32_t count) {
  return count == 1 || count == 2 || count == 4 || count == 8 || count == 16;
}

inline std::uint32_t RasterSampleMask(std::uint32_t count) {
  if (!IsSupportedRasterSampleCount(count))
    throw std::runtime_error("unsupported raster sample_count");
  return (1U << count) - 1U;
}

inline std::array<std::uint8_t, 2>
RasterSamplePosition(std::uint32_t count, std::uint32_t index) {
  if (!IsSupportedRasterSampleCount(count) || index >= count)
    throw std::runtime_error("invalid raster sample position index/count");
  constexpr std::array<std::array<std::uint8_t, 2>, 2> positions2 = {{{12, 12}, {4, 4}}};
  constexpr std::array<std::array<std::uint8_t, 2>, 4> positions4 =
      {{{6, 2}, {14, 6}, {2, 10}, {10, 14}}};
  constexpr std::array<std::array<std::uint8_t, 2>, 8> positions8 =
      {{{9, 5}, {7, 11}, {13, 9}, {5, 3}, {3, 13}, {1, 7}, {11, 15}, {15, 1}}};
  constexpr std::array<std::array<std::uint8_t, 2>, 16> positions16 =
      {{{9, 9}, {7, 5}, {5, 10}, {12, 7}, {3, 6}, {10, 13}, {13, 11}, {11, 3},
        {6, 14}, {8, 1}, {4, 2}, {2, 12}, {0, 8}, {15, 4}, {14, 15}, {1, 0}}};
  switch (count) {
  case 1: return {8, 8};
  case 2: return positions2[index];
  case 4: return positions4[index];
  case 8: return positions8[index];
  default: return positions16[index];
  }
}

} // namespace pvrgpu::stub

#endif
