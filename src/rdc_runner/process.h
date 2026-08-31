#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace pvrgpu::rdc {

struct ProcessRequest {
  std::filesystem::path executable;
  std::vector<std::string> arguments;
  std::map<std::string, std::string> environment;
  std::vector<std::string> unset_environment;
  std::filesystem::path stdout_path;
  std::filesystem::path stderr_path;
};

struct ProcessResult {
  bool started = false;
  int exit_code = -1;
  std::string error;
};

ProcessResult RunProcess(const ProcessRequest &request);

}  // namespace pvrgpu::rdc
