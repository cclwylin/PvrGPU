#pragma once

#include <filesystem>
#include <string>

namespace pvrgpu::rdc {

// ProcessRequest string fields and serialized runner artifacts use UTF-8.
// In C++17, path::u8string() and generic_u8string() both return std::string.
inline std::string PathToUtf8(const std::filesystem::path &path) {
  return path.u8string();
}

inline std::string PathToGenericUtf8(const std::filesystem::path &path) {
  return path.generic_u8string();
}

inline std::filesystem::path PathFromUtf8(const std::string &text) {
  return std::filesystem::u8path(text);
}

}  // namespace pvrgpu::rdc
