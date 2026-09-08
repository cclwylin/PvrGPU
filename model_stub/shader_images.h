#pragma once

#include "model_types.h"
#include <algorithm>
#include <array>
#include <string>

namespace pvrgpu::stub {
inline constexpr std::uint32_t kMaximumFragmentImages = 32;
inline constexpr std::uint64_t kMaximumFragmentImageBytes = UINT64_C(0x10000000);

inline bool ValidateDriverShaderImages(const DriverCommand &command, std::string *error) {
  const auto reject = [&](const char *message) {
    if (error) *error = std::string("PCO fragment image ") + message;
    return false;
  };
  const auto count = command.fragment_image_descriptor_count;
  const auto used = command.fragment_image_read_mask | command.fragment_image_write_mask;
  if (command.fragment_early_tests > 1 || count > kMaximumFragmentImages ||
      command.fragment_images.size() > kMaximumFragmentImages ||
      (count < 32 && (used >> count)))
    return reject("descriptor count/masks/early-test flag is invalid");
  if (!count)
    return command.fragment_images.empty() && !command.fragment_image_descriptor_start && !used
        ? true : reject("payload without a native descriptor range");
  const auto &abi = command.fragment_pco_abi;
  const auto &shared = command.fragment_shared;
  const std::size_t start = command.fragment_image_descriptor_start;
  if (start != command.fragment_sampled_texture_count * 20U +
                   abi.uniform_buffer_descriptor_count * 4U ||
      start > shared.size() || count * 8U > shared.size() - start ||
      abi.push_constant_start != start + count * 8U || shared.size() != abi.shareds ||
      abi.push_constant_count != shared.size() - abi.push_constant_start)
    return reject("texture/UBO/image/CB0 shared layout is inconsistent");
  std::array<const DriverShaderImage *, kMaximumFragmentImages> images{};
  for (const auto &image : command.fragment_images) {
    if (image.image_slot >= count || images[image.image_slot] ||
        image.format != 1 || image.access != 3 ||
        !image.resource_token || image.bytes.empty() || image.bytes.size() > kMaximumFragmentImageBytes ||
        !image.width || !image.height || image.depth != 1 || image.texel_bytes != 4 ||
        image.offset % 4 || image.row_stride % 4 || image.layer_stride % 4 ||
        std::uint64_t(image.width) * 4 > image.row_stride ||
        std::uint64_t(image.row_stride) * image.height > image.layer_stride ||
        image.offset > image.bytes.size())
      return reject("view slot/format/access/backing/layout is invalid");
    std::uint64_t remaining = image.bytes.size() - image.offset;
    for (const auto extent : {std::uint64_t(image.depth - 1) * image.layer_stride,
                              std::uint64_t(image.height - 1) * image.row_stride,
                              std::uint64_t(image.width) * 4}) {
      if (extent > remaining) return reject("view footprint exceeds its backing resource");
      remaining -= extent;
    }
    const auto bit = UINT32_C(1) << image.image_slot;
    const unsigned access = ((command.fragment_image_read_mask & bit) ? 1U : 0U) |
                            ((command.fragment_image_write_mask & bit) ? 2U : 0U);
    if (!access || (image.access & access) != access)
      return reject("native read/write mask disagrees with binding access");
    for (const auto *prior : images)
      if (prior && prior->resource_token == image.resource_token && prior->bytes != image.bytes)
        return reject("aliased views disagree on the immutable backing snapshot");
    images[image.image_slot] = &image;
  }
  for (unsigned slot = 0; slot < count; ++slot) {
    const auto *image = images[slot];
    if (((used >> slot) & 1U) != (image != nullptr))
      return reject("native image slot lacks exactly one bound view");
    const std::array<std::uint32_t, 8> expected = image
        ? std::array<std::uint32_t, 8>{0, 0, image->depth, image->layer_stride,
             image->width, image->height, image->row_stride, image->texel_bytes}
        : std::array<std::uint32_t, 8>{};
    if (!std::equal(expected.begin(), expected.end(), shared.begin() + start + slot * 8))
      return reject("native image descriptor is not canonical before relocation");
  }
  return true;
}
} // namespace pvrgpu::stub
