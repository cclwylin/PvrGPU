#pragma once

#include <filesystem>
#include <map>
#include <string>

namespace pvrgpu::rdc {

class RuntimeConfig final {
 public:
  explicit RuntimeConfig(std::filesystem::path project_root);

  const std::filesystem::path &project_root() const { return project_root_; }
  std::string Get(const std::string &name,
                  const std::string &fallback = {}) const;
  std::filesystem::path Path(const std::string &name,
                             const std::filesystem::path &fallback = {}) const;

 private:
  void LoadLocalEnvironment();
  std::string Expand(std::string value) const;

  std::filesystem::path project_root_;
  std::map<std::string, std::string> values_;
};

std::filesystem::path DefaultProjectRoot();
std::filesystem::path DefaultWorkRoot();
std::filesystem::path ResolveExecutable(const std::filesystem::path &base);

}  // namespace pvrgpu::rdc
