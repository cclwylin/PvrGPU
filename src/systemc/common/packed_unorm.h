// Native four-byte normalized color storage. Keep the packed word intact
// across LOAD, blending, DRAM and readback; an RGBA8 intermediate loses bits.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace pvrgpu::stub {

enum class PackedUnormFormat : std::uint8_t { kNone, kRgb10A2, kBgr10A2 };

inline PackedUnormFormat PackedUnormFormatFromName(std::string_view name) {
  if (name == "PIPE_FORMAT_R10G10B10A2_UNORM")
    return PackedUnormFormat::kRgb10A2;
  if (name == "PIPE_FORMAT_B10G10R10A2_UNORM")
    return PackedUnormFormat::kBgr10A2;
  return PackedUnormFormat::kNone;
}

inline unsigned PackedUnormShift(PackedUnormFormat format, unsigned component) {
  if ((format != PackedUnormFormat::kRgb10A2 &&
       format != PackedUnormFormat::kBgr10A2) || component > 3)
    throw std::runtime_error("invalid packed UNORM format/channel");
  if (component == 3)
    return 30;
  return 10 * (format == PackedUnormFormat::kBgr10A2 ? 2 - component : component);
}

inline std::uint32_t QuantizePackedUnorm(float value, unsigned bits) {
  if (bits != 2 && bits != 10)
    throw std::runtime_error("invalid packed UNORM channel width");
  const std::uint32_t maximum = (1U << bits) - 1U;
  if (!(value > 0.0F)) // Includes NaN, like the existing UNORM8 store.
    return 0;
  if (value >= 1.0F)
    return maximum;
  const float scaled = value * static_cast<float>(maximum);
  const auto lower = static_cast<std::uint32_t>(scaled);
  const float fraction = scaled - static_cast<float>(lower);
  // Nearest-even, independent of the host process's floating-point mode.
  return lower + (fraction > 0.5F || (fraction == 0.5F && (lower & 1U)));
}

inline std::array<float, 4> UnpackUnormColor(std::uint32_t word,
                                           PackedUnormFormat format) {
  std::array<float, 4> color{};
  for (unsigned component = 0; component < 4; ++component) {
    const std::uint32_t maximum = component == 3 ? 3U : 1023U;
    color[component] = static_cast<float>(
        (word >> PackedUnormShift(format, component)) & maximum) /
        static_cast<float>(maximum);
  }
  return color;
}

inline std::uint32_t PackUnormColor(const std::array<float, 4> &color,
                                   PackedUnormFormat format,
                                   std::uint32_t previous = 0,
                                   std::uint8_t color_mask = 0x0f) {
  for (unsigned component = 0; component < 4; ++component) {
    const unsigned shift = PackedUnormShift(format, component);
    if (!(color_mask & (1U << component)))
      continue;
    const unsigned bits = component == 3 ? 2 : 10;
    const std::uint32_t mask = ((1U << bits) - 1U) << shift;
    previous = (previous & ~mask) |
        (QuantizePackedUnorm(color[component], bits) << shift);
  }
  return previous;
}

} // namespace pvrgpu::stub
