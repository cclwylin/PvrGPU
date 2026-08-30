#include "png_writer.h"

#include <png.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>

namespace pvrgpu::stub {
namespace {

[[noreturn]] void Fail(const std::filesystem::path& part_path,
                       const std::string& message) {
  std::error_code ignored;
  std::filesystem::remove(part_path, ignored);
  throw std::runtime_error(message);
}

}  // namespace

void WriteRgbaPngAtomic(
    const std::filesystem::path& final_path,
    const std::vector<std::uint8_t>& rgba_bottom_up,
    std::uint32_t width,
    std::uint32_t height) {
  std::filesystem::path part_path = final_path;
  part_path += ".part";

  if (final_path.empty())
    Fail(part_path, "PNG output path is empty");
  if (width == 0 || height == 0)
    Fail(part_path, "PNG dimensions must be non-zero");

  constexpr std::size_t kChannels = 4;
  const std::size_t size_max = std::numeric_limits<std::size_t>::max();
  if (static_cast<std::size_t>(width) > size_max / kChannels) {
    Fail(part_path, "PNG row byte count overflows size_t");
  }
  const std::size_t row_bytes = static_cast<std::size_t>(width) * kChannels;
  if (static_cast<std::size_t>(height) > size_max / row_bytes) {
    Fail(part_path, "PNG framebuffer byte count overflows size_t");
  }
  const std::size_t expected_bytes =
      row_bytes * static_cast<std::size_t>(height);
  if (rgba_bottom_up.size() != expected_bytes) {
    Fail(part_path,
         "PNG RGBA buffer size mismatch: expected " +
             std::to_string(expected_bytes) + ", got " +
             std::to_string(rgba_bottom_up.size()));
  }
  if (row_bytes >
      static_cast<std::size_t>(std::numeric_limits<png_int_32>::max())) {
    Fail(part_path, "PNG row stride exceeds libpng's signed stride limit");
  }

  const std::filesystem::path parent = final_path.parent_path();
  if (!parent.empty()) {
    std::error_code directory_error;
    std::filesystem::create_directories(parent, directory_error);
    if (directory_error) {
      Fail(part_path,
           "Could not create PNG output directory '" + parent.string() +
               "': " + directory_error.message());
    }
  }

  png_image image{};
  image.version = PNG_IMAGE_VERSION;
  image.width = width;
  image.height = height;
  image.format = PNG_FORMAT_RGBA;

  const std::string part_name = part_path.string();
  const png_int_32 bottom_up_stride =
      -static_cast<png_int_32>(row_bytes);
  if (!png_image_write_to_file(&image, part_name.c_str(), 0,
                               rgba_bottom_up.data(), bottom_up_stride,
                               nullptr)) {
    const std::string libpng_message =
        image.message[0] ? image.message : "unknown libpng error";
    png_image_free(&image);
    Fail(part_path,
         "Could not write PNG '" + final_path.string() +
             "': " + libpng_message);
  }
  png_image_free(&image);

  std::error_code rename_error;
  std::filesystem::rename(part_path, final_path, rename_error);
  if (rename_error) {
    Fail(part_path,
         "Could not publish PNG '" + final_path.string() +
             "': " + rename_error.message());
  }
}

}  // namespace pvrgpu::stub
