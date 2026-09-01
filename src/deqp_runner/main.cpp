/*-------------------------------------------------------------------------
 * PvrGPU live dEQP runner
 * ----------------------
 *
 * Copyright 2026 PvrGPU contributors
 * Copyright 2014 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * This entry point is based on VK-GL-CTS framework/platform/tcuMain.cpp and
 * adds the environment/artifact contract required by the PvrGPU driver.
 *------------------------------------------------------------------------*/

#include "deUniquePtr.hpp"
#include "pvrgpu_systemc_api.h"
#include "qpDebugOut.h"
#include "tcuApp.hpp"
#include "tcuCommandLine.hpp"
#include "tcuDefs.hpp"
#include "tcuPlatform.hpp"
#include "tcuResource.hpp"
#include "tcuTestLog.hpp"
#include "tcuTestSessionExecutor.hpp"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <process.h>
#else
#include <unistd.h>
#endif

#ifndef PVRGPU_DEQP_DEFAULT_ARCHIVE_DIR
#define PVRGPU_DEQP_DEFAULT_ARCHIVE_DIR "."
#endif

#ifndef PVRGPU_DEQP_DEFAULT_SYSTEMC_API_LIB
#define PVRGPU_DEQP_DEFAULT_SYSTEMC_API_LIB ""
#endif

// Implemented by the selected VK-GL-CTS platform target.
tcu::Platform *createPlatform(void);

using PvrGpuDeqpCaseCallback = void (*)(const char *case_path);
extern "C" void pvrgpuDeqpSetCaseCallback(
    PvrGpuDeqpCaseCallback callback);

namespace {

namespace fs = std::filesystem;

struct RunnerOptions {
  fs::path output_root;
  fs::path systemc_api_library;
  bool show_pvrgpu_help = false;
  std::vector<std::string> deqp_arguments;
};

struct RunnerState {
  fs::path output_root;
  fs::path framework_dir;
};

RunnerState *g_runner_state = nullptr;

bool DisableRawWrites(int, const char *) { return false; }

bool DisableFormattedWrites(int, const char *, va_list) { return false; }

void DisableStdout() {
  qpRedirectOut(DisableRawWrites, DisableFormattedWrites);
}

std::string PathToUtf8(const fs::path &path) {
#ifdef _WIN32
  return path.u8string();
#else
  return path.string();
#endif
}

bool SetEnvironment(const char *name, const std::string &value) {
#ifdef _WIN32
  return _putenv_s(name, value.c_str()) == 0;
#else
  return setenv(name, value.c_str(), 1) == 0;
#endif
}

bool UnsetEnvironment(const char *name) {
#ifdef _WIN32
  return _putenv_s(name, "") == 0;
#else
  return unsetenv(name) == 0;
#endif
}

std::string EnvironmentValue(const char *name) {
  const char *value = std::getenv(name);
  return value ? std::string(value) : std::string();
}

std::string SafeCaseName(const std::string &case_name) {
  std::string result;
  result.reserve(case_name.size());
  for (const unsigned char character : case_name) {
    const bool safe =
        (character >= 'a' && character <= 'z') ||
        (character >= 'A' && character <= 'Z') ||
        (character >= '0' && character <= '9') || character == '.' ||
        character == '_' || character == '-';
    result.push_back(safe ? static_cast<char>(character) : '_');
  }
  if (result.empty())
    result = "unnamed";
  return result;
}

bool SetArtifactEnvironment(const fs::path &directory,
                            const char *case_name) {
  std::error_code error;
  fs::create_directories(directory / "systemc", error);
  if (error) {
    std::cerr << "pvrgpu-deqp: cannot create artifact directory "
              << PathToUtf8(directory) << ": " << error.message() << '\n';
    return false;
  }

  bool ok = true;
  if (case_name && *case_name)
    ok = SetEnvironment("PVRGPU_RDC_CASE_NAME", case_name) && ok;
  else
    ok = UnsetEnvironment("PVRGPU_RDC_CASE_NAME") && ok;
  ok = SetEnvironment("PVRGPU_DRIVER_COMMAND_OUT",
                      PathToUtf8(directory / "driver-command.txt")) &&
       ok;
  ok = SetEnvironment("PVRGPU_DRIVER_COUNTER_OUT",
                      PathToUtf8(directory / "driver-counter.txt")) &&
       ok;
  ok = SetEnvironment("PVRGPU_SYSTEMC_JSONL_OUT",
                      PathToUtf8(directory / "systemc.jsonl")) &&
       ok;
  ok = SetEnvironment("PVRGPU_SYSTEMC_STDERR_OUT",
                      PathToUtf8(directory / "systemc.stderr.log")) &&
       ok;
  ok = SetEnvironment("PVRGPU_SYSTEMC_OUTDIR",
                      PathToUtf8(directory / "systemc")) &&
       ok;
  if (!ok)
    std::cerr << "pvrgpu-deqp: cannot update the test artifact environment\n";
  return ok;
}

void BeginCase(const char *case_path) {
  if (!g_runner_state)
    return;
  if (!case_path || !*case_path) {
    SetArtifactEnvironment(g_runner_state->framework_dir, nullptr);
    return;
  }
  const fs::path case_dir =
      g_runner_state->output_root / "cases" / SafeCaseName(case_path);
  SetArtifactEnvironment(case_dir, case_path);
}

int ProcessId() {
#ifdef _WIN32
  return _getpid();
#else
  return static_cast<int>(getpid());
#endif
}

fs::path DefaultOutputRoot() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const auto milliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
  std::error_code error;
  fs::path temporary = fs::temp_directory_path(error);
  if (error)
    temporary = fs::current_path();
  return temporary /
         ("pvrgpu-deqp-" + std::to_string(milliseconds) + "-" +
          std::to_string(ProcessId()));
}

bool HasOption(const std::vector<std::string> &arguments,
               const std::string &option) {
  for (const std::string &argument : arguments) {
    if (argument == option || argument.rfind(option + "=", 0) == 0)
      return true;
  }
  return false;
}

bool ReadRunnerValue(const std::string &argument, const std::string &name,
                     std::string *value) {
  const std::string prefix = name + "=";
  if (argument.rfind(prefix, 0) != 0)
    return false;
  *value = argument.substr(prefix.size());
  return true;
}

void PrintPvrGpuHelp(const char *program) {
  std::cout
      << "PvrGPU live dEQP runner\n\n"
      << "Usage:\n  " << program
      << " --deqp-case=<case-or-pattern> [dEQP options]\n\n"
      << "PvrGPU options:\n"
      << "  --pvrgpu-output-dir=<path>\n"
      << "      Artifact root. Defaults to a unique temporary directory.\n"
      << "  --pvrgpu-systemc-api-lib=<path>\n"
      << "      Override the linked PvrGPU SystemC bridge.\n"
      << "  --pvrgpu-help\n"
      << "      Show this help without starting dEQP.\n\n"
      << "The runner defaults to a surfaceless pbuffer and writes one artifact\n"
      << "directory per expanded dEQP case. Standard dEQP arguments pass through.\n";
}

bool ParseRunnerOptions(int argc, char **argv, RunnerOptions *options,
                        std::string *error) {
  options->deqp_arguments.clear();
  options->deqp_arguments.emplace_back(argc > 0 ? argv[0] : "pvrgpu-deqp");

  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    std::string value;
    if (argument == "--pvrgpu-help") {
      options->show_pvrgpu_help = true;
      continue;
    }
    if (ReadRunnerValue(argument, "--pvrgpu-output-dir", &value)) {
      if (value.empty()) {
        *error = "--pvrgpu-output-dir requires a non-empty value";
        return false;
      }
      options->output_root = fs::path(value);
      continue;
    }
    if (argument == "--pvrgpu-output-dir") {
      if (++index >= argc || !argv[index][0]) {
        *error = "--pvrgpu-output-dir requires a value";
        return false;
      }
      options->output_root = fs::path(argv[index]);
      continue;
    }
    if (ReadRunnerValue(argument, "--pvrgpu-systemc-api-lib", &value)) {
      if (value.empty()) {
        *error = "--pvrgpu-systemc-api-lib requires a non-empty value";
        return false;
      }
      options->systemc_api_library = fs::path(value);
      continue;
    }
    if (argument == "--pvrgpu-systemc-api-lib") {
      if (++index >= argc || !argv[index][0]) {
        *error = "--pvrgpu-systemc-api-lib requires a value";
        return false;
      }
      options->systemc_api_library = fs::path(argv[index]);
      continue;
    }
    if (argument.rfind("--pvrgpu-", 0) == 0) {
      *error = "unknown PvrGPU option: " + argument;
      return false;
    }
    options->deqp_arguments.push_back(argument);
  }

  if (options->output_root.empty())
    options->output_root = DefaultOutputRoot();
  std::error_code path_error;
  options->output_root = fs::absolute(options->output_root, path_error);
  if (path_error) {
    *error = "cannot resolve output directory: " + path_error.message();
    return false;
  }

  if (options->systemc_api_library.empty()) {
    const std::string configured = EnvironmentValue("PVRGPU_SYSTEMC_API_LIB");
    options->systemc_api_library =
        configured.empty() ? fs::path(PVRGPU_DEQP_DEFAULT_SYSTEMC_API_LIB)
                           : fs::path(configured);
  }
  if (options->systemc_api_library.empty()) {
    *error = "PvrGPU SystemC bridge path is empty";
    return false;
  }
  options->systemc_api_library =
      fs::absolute(options->systemc_api_library, path_error);
  if (path_error) {
    *error = "cannot resolve SystemC bridge: " + path_error.message();
    return false;
  }
  if (!fs::is_regular_file(options->systemc_api_library, path_error) ||
      path_error) {
    *error = "PvrGPU SystemC bridge does not exist: " +
             PathToUtf8(options->systemc_api_library);
    return false;
  }

  if (!HasOption(options->deqp_arguments, "--deqp-surface-type"))
    options->deqp_arguments.emplace_back("--deqp-surface-type=pbuffer");
  if (!HasOption(options->deqp_arguments, "--deqp-archive-dir")) {
    options->deqp_arguments.emplace_back(
        std::string("--deqp-archive-dir=") +
        PVRGPU_DEQP_DEFAULT_ARCHIVE_DIR);
  }
  if (!HasOption(options->deqp_arguments, "--deqp-log-filename")) {
    options->deqp_arguments.emplace_back(
        "--deqp-log-filename=" +
        PathToUtf8(options->output_root / "results.qpa"));
  }
  return true;
}

bool PrepareRuntime(const RunnerOptions &options, RunnerState *state,
                    std::string *error) {
  std::error_code filesystem_error;
  fs::create_directories(options.output_root / "cases", filesystem_error);
  if (filesystem_error) {
    *error = "cannot create output directory: " + filesystem_error.message();
    return false;
  }

  state->output_root = options.output_root;
  state->framework_dir = options.output_root / "framework";
  g_runner_state = state;

  bool ok = true;
  ok = SetEnvironment("EGL_PLATFORM", "surfaceless") && ok;
  ok = SetEnvironment("GALLIUM_DRIVER", "pvrgpu") && ok;
  ok = SetEnvironment("MESA_LOADER_DRIVER_OVERRIDE", "swrast") && ok;
  ok = SetEnvironment("LIBGL_ALWAYS_SOFTWARE", "true") && ok;
  ok = SetEnvironment("MESA_SHADER_CACHE_DISABLE", "true") && ok;
  ok = SetEnvironment("PVRGPU_DEQP_LIVE", "1") && ok;
  ok = SetEnvironment("PVRGPU_DEQP_OUTPUT_ROOT",
                      PathToUtf8(options.output_root)) &&
       ok;
  ok = SetEnvironment("PVRGPU_SYSTEMC_API_LIB",
                      PathToUtf8(options.systemc_api_library)) &&
       ok;
  ok = SetArtifactEnvironment(state->framework_dir, nullptr) && ok;
  if (!ok) {
    *error = "cannot configure the PvrGPU runtime environment";
    return false;
  }

  std::ofstream manifest(options.output_root / "run.txt",
                         std::ios::out | std::ios::trunc);
  if (!manifest) {
    *error = "cannot write run manifest";
    return false;
  }
  manifest << "schema=pvrgpu.deqp-live-run.v1\n"
           << "backend=pvrgpu\n"
           << "output_root=" << PathToUtf8(options.output_root) << '\n'
           << "archive_dir=" << PVRGPU_DEQP_DEFAULT_ARCHIVE_DIR << '\n'
           << "systemc_api_lib="
           << PathToUtf8(options.systemc_api_library) << '\n';
  if (!manifest) {
    *error = "cannot finish run manifest";
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  // Keep the bridge as a direct Mach-O/ELF dependency as well as the driver's
  // dlopen target.  The volatile store prevents dead stripping of the symbol.
  volatile pvrgpu_systemc_submit_driver_command_fn linked_systemc_bridge =
      &pvrgpu_systemc_submit_driver_command;
  if (!linked_systemc_bridge) {
    std::cerr << "pvrgpu-deqp: linked SystemC bridge symbol is unavailable\n";
    return EXIT_FAILURE;
  }

  RunnerOptions options;
  std::string error;
  if (!ParseRunnerOptions(argc, argv, &options, &error)) {
    std::cerr << "pvrgpu-deqp: " << error << '\n';
    return EXIT_FAILURE;
  }
  if (options.show_pvrgpu_help) {
    PrintPvrGpuHelp(argc > 0 ? argv[0] : "pvrgpu-deqp");
    return EXIT_SUCCESS;
  }

  RunnerState state;
  if (!PrepareRuntime(options, &state, &error)) {
    std::cerr << "pvrgpu-deqp: " << error << '\n';
    return EXIT_FAILURE;
  }
  pvrgpuDeqpSetCaseCallback(BeginCase);

  std::vector<char *> mutable_arguments;
  mutable_arguments.reserve(options.deqp_arguments.size() + 1);
  for (std::string &argument : options.deqp_arguments)
    mutable_arguments.push_back(argument.data());
  mutable_arguments.push_back(nullptr);

  std::cout << "PvrGPU artifacts: " << PathToUtf8(options.output_root) << '\n';

  int exit_status = EXIT_SUCCESS;
#if (DE_OS != DE_OS_WIN32)
  setvbuf(stdout, nullptr, _IOLBF, 4 * 1024);
#endif

  try {
    tcu::CommandLine command_line(
        static_cast<int>(options.deqp_arguments.size()),
        mutable_arguments.data());
    if (command_line.quietMode())
      DisableStdout();

    tcu::DirArchive archive(command_line.getArchiveDir());
    tcu::TestLog log(command_line.getLogFileName(), command_line.getLogFlags());
    de::UniquePtr<tcu::Platform> platform(createPlatform());
    de::UniquePtr<tcu::App> app(
        new tcu::App(*platform, archive, log, command_line));

    for (;;) {
      if (!app->iterate()) {
        if (command_line.getRunMode() == tcu::RUNMODE_EXECUTE &&
            (!app->getResult().isComplete || app->getResult().numFailed))
          exit_status = EXIT_FAILURE;
        break;
      }
    }
  } catch (const std::exception &exception) {
    tcu::die("%s", exception.what());
  }

  pvrgpuDeqpSetCaseCallback(nullptr);
  BeginCase(nullptr);
  return exit_status;
}
