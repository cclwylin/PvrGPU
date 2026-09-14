#pragma once

#include "graphics_shader_buffers.h"
#include "model_types.h"
#include <algorithm>
#include <array>
#include <string>

namespace pvrgpu::stub {
inline constexpr std::uint32_t kMaximumFragmentImages = 32;
inline constexpr std::uint64_t kMaximumFragmentImageBytes = UINT64_C(0x10000000);

inline constexpr std::uint32_t kDriverShaderImageR32Uint = 1;
inline constexpr std::uint32_t kDriverShaderImageRaw = 2;

// One stage's image list against its native descriptor range. R32UI views keep
// the atomic read/write contract; raw views are read-only native bytes the
// shader unpacks itself.
inline bool ValidateDriverStageShaderImages(
    const DriverCommand &command, std::size_t stage,
    const std::vector<DriverShaderImage> &stage_images, std::uint32_t start_word,
    std::uint32_t count, std::uint32_t read_mask, std::uint32_t write_mask,
    std::string *error) {
  const char *stage_name = stage == 0 ? "vertex" : "fragment";
  const auto reject = [&](const char *message) {
    if (error) *error = std::string("PCO ") + stage_name + " image " + message;
    return false;
  };
  const auto used = read_mask | write_mask;
  DriverGraphicsDescriptorLayout layout;
  if (count > kMaximumFragmentImages ||
      stage_images.size() > kMaximumFragmentImages ||
      (count < 32 && (used >> count)) ||
      !ResolveDriverGraphicsDescriptorLayout(command, stage, &layout))
    return reject("descriptor count/masks/early-test flag is invalid");
  if (!count)
    return stage_images.empty() && !start_word && !used
        ? true : reject("payload without a native descriptor range");
  const auto &abi = stage == 0 ? command.vertex_pco_abi : command.fragment_pco_abi;
  const auto &shared = stage == 0 ? command.vertex_shared : command.fragment_shared;
  const std::size_t start = start_word;
  if (start != layout.image_start ||
      start > shared.size() || count * 8U > shared.size() - start ||
      shared.size() != abi.shareds)
    return reject("texture/UBO/image/CB0 shared layout is inconsistent");
  std::array<const DriverShaderImage *, kMaximumFragmentImages> images{};
  for (const auto &image : stage_images) {
    const bool r32 = image.format == kDriverShaderImageR32Uint;
    const bool raw = image.format == kDriverShaderImageRaw;
    if (image.image_slot >= count || images[image.image_slot] ||
        (!r32 && !raw) ||
        (r32 ? (image.access != 3 || image.texel_bytes != 4 || image.depth != 1)
             : (image.access != 1 || !image.texel_bytes || image.texel_bytes > 16 ||
                image.texel_bytes % 4 || !image.depth)) ||
        (stage == 0 && !raw) ||
        !image.resource_token || image.bytes.empty() || image.bytes.size() > kMaximumFragmentImageBytes ||
        !image.width || !image.height ||
        image.offset % 4 || image.row_stride % 4 || image.layer_stride % 4 ||
        std::uint64_t(image.width) * image.texel_bytes > image.row_stride ||
        std::uint64_t(image.row_stride) * image.height > image.layer_stride ||
        image.offset > image.bytes.size())
      return reject("view slot/format/access/backing/layout is invalid");
    std::uint64_t remaining = image.bytes.size() - image.offset;
    for (const auto extent : {std::uint64_t(image.depth - 1) * image.layer_stride,
                              std::uint64_t(image.height - 1) * image.row_stride,
                              std::uint64_t(image.width) * image.texel_bytes}) {
      if (extent > remaining) return reject("view footprint exceeds its backing resource");
      remaining -= extent;
    }
    const auto bit = UINT32_C(1) << image.image_slot;
    const unsigned access = ((read_mask & bit) ? 1U : 0U) |
                            ((write_mask & bit) ? 2U : 0U);
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

inline bool ValidateDriverShaderImages(const DriverCommand &command, std::string *error) {
  if (command.fragment_early_tests > 1) {
    if (error) *error = "PCO fragment image descriptor count/masks/early-test flag is invalid";
    return false;
  }
  return ValidateDriverStageShaderImages(
             command, 0, command.vertex_images, command.vertex_image_descriptor_start,
             command.vertex_image_descriptor_count, command.vertex_image_read_mask, 0,
             error) &&
         ValidateDriverStageShaderImages(
             command, 1, command.fragment_images, command.fragment_image_descriptor_start,
             command.fragment_image_descriptor_count, command.fragment_image_read_mask,
             command.fragment_image_write_mask, error);
}
} // namespace pvrgpu::stub
