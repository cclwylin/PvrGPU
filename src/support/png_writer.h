#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace pvrgpu::stub {

// Writes a bottom-up, tightly packed RGBA8 framebuffer. The completed PNG is
// published with a same-directory rename so artifact consumers never observe
// a partially written final file.
void WriteRgbaPngAtomic(
    const std::filesystem::path& final_path,
    const std::vector<std::uint8_t>& rgba_bottom_up,
    std::uint32_t width,
    std::uint32_t height);

}  // namespace pvrgpu::stub
