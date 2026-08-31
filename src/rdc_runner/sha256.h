#pragma once

#include <filesystem>
#include <string>

namespace pvrgpu::rdc {

bool Sha256File(const std::filesystem::path &path, std::string *digest,
                std::string *error);

}  // namespace pvrgpu::rdc
