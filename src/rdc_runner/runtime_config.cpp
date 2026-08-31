#include "rdc_runner/runtime_config.h"
#include "rdc_runner/path_utf8.h"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <regex>
#include <string>
#include <utility>

#ifndef PVRGPU_SOURCE_DIR
#define PVRGPU_SOURCE_DIR "."
#endif

namespace pvrgpu::rdc {
namespace {

std::string Trim(std::string value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.front())))
    value.erase(value.begin());
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back())))
    value.pop_back();
  return value;
}

bool IsName(const std::string &name) {
  if (name.empty() ||
      !(std::isalpha(static_cast<unsigned char>(name.front())) ||
        name.front() == '_'))
    return false;
  for (char character : name) {
    if (!(std::isalnum(static_cast<unsigned char>(character)) ||
          character == '_'))
      return false;
  }
  return true;
}

std::string EnvironmentValue(const std::string &name) {
  const char *value = std::getenv(name.c_str());
  return value ? std::string(value) : std::string();
}

}  // namespace

std::filesystem::path DefaultProjectRoot() {
  return PathFromUtf8(PVRGPU_SOURCE_DIR);
}

std::filesystem::path DefaultWorkRoot() {
#ifdef _WIN32
  std::string home = EnvironmentValue("USERPROFILE");
#else
  std::string home = EnvironmentValue("HOME");
#endif
  if (home.empty())
    return std::filesystem::current_path() / "PvrGPU-work";
  return PathFromUtf8(home) / "Downloads" / "_Codex" / "Working" /
         "PvrGPU";
}

RuntimeConfig::RuntimeConfig(std::filesystem::path project_root)
    : project_root_(std::move(project_root)) {
  std::error_code error;
  const std::filesystem::path canonical =
      std::filesystem::weakly_canonical(project_root_, error);
  if (!error)
    project_root_ = canonical;
  values_["PVRGPU_PROJECT_ROOT"] = PathToUtf8(project_root_);
  values_["PVRGPU_WORK_ROOT"] = PathToUtf8(DefaultWorkRoot());
  LoadLocalEnvironment();
}

std::string RuntimeConfig::Get(const std::string &name,
                               const std::string &fallback) const {
  const char *environment = std::getenv(name.c_str());
  if (environment && *environment)
    return environment;
  const auto found = values_.find(name);
  if (found != values_.end() && !found->second.empty())
    return found->second;
  return fallback;
}

std::filesystem::path RuntimeConfig::Path(
    const std::string &name, const std::filesystem::path &fallback) const {
  const std::string value = Get(name, PathToUtf8(fallback));
  if (value.empty())
    return {};
  std::filesystem::path path = PathFromUtf8(value);
  if (path.is_relative())
    path = project_root_ / path;
  return path.lexically_normal();
}

std::string RuntimeConfig::Expand(std::string value) const {
  if (value.find('`') != std::string::npos ||
      value.find("$(") != std::string::npos)
    return {};

  static const std::regex variable(R"(\$\{([A-Za-z_][A-Za-z0-9_]*)\})");
  for (unsigned pass = 0; pass < 16; ++pass) {
    std::smatch match;
    if (!std::regex_search(value, match, variable))
      break;
    const std::string name = match[1].str();
    std::string replacement = EnvironmentValue(name);
    if (replacement.empty()) {
      const auto found = values_.find(name);
      if (found != values_.end())
        replacement = found->second;
    }
    value.replace(static_cast<std::size_t>(match.position()),
                  static_cast<std::size_t>(match.length()), replacement);
  }
  return value;
}

void RuntimeConfig::LoadLocalEnvironment() {
  std::ifstream input(project_root_ / "config" / "local.env");
  if (!input)
    return;
  std::string line;
  while (std::getline(input, line)) {
    line = Trim(line);
    if (line.empty() || line.front() == '#')
      continue;
    if (line.rfind("export ", 0) == 0)
      line = Trim(line.substr(7));
    const std::size_t equals = line.find('=');
    if (equals == std::string::npos)
      continue;
    const std::string name = Trim(line.substr(0, equals));
    if (!IsName(name))
      continue;
    std::string value = Trim(line.substr(equals + 1));
    if (value.size() >= 2 &&
        ((value.front() == '\'' && value.back() == '\'') ||
         (value.front() == '"' && value.back() == '"')))
      value = value.substr(1, value.size() - 2);
    value = Expand(value);
    values_[name] = value;
  }
}

std::filesystem::path ResolveExecutable(const std::filesystem::path &base) {
  if (std::filesystem::is_regular_file(base))
    return base;
#ifdef _WIN32
  if (base.extension() != ".exe") {
    std::filesystem::path with_extension = base;
    with_extension += ".exe";
    if (std::filesystem::is_regular_file(with_extension))
      return with_extension;
  }
#else
  if (base.extension() == ".exe") {
    std::filesystem::path without_extension = base;
    without_extension.replace_extension();
    if (std::filesystem::is_regular_file(without_extension))
      return without_extension;
  } else {
    std::filesystem::path with_extension = base;
    with_extension += ".exe";
    if (std::filesystem::is_regular_file(with_extension))
      return with_extension;
  }
#endif
  return base;
}

}  // namespace pvrgpu::rdc
