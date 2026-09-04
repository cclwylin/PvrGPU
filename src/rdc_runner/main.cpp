#include "rdc_runner/process.h"
#include "rdc_runner/path_utf8.h"
#include "rdc_runner/runtime_config.h"
#include "rdc_runner/sha256.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#ifndef PVRGPU_RDC_BACKEND
#error "PVRGPU_RDC_BACKEND must be llvmpipe or pvrgpu"
#endif

namespace pvrgpu::rdc {
namespace {

constexpr std::string_view kBackend = PVRGPU_RDC_BACKEND;
constexpr std::string_view kResultSchema = "pvrgpu.backend-result.v1";
constexpr std::array<std::string_view, 17> kCounterFields = {
    "ia_vertices",     "ia_primitives",   "vs_invocations",
    "gs_invocations",  "gs_primitives",   "c_invocations",
    "c_primitives",    "ps_invocations",  "hs_invocations",
    "ds_invocations",  "cs_invocations",  "ts_invocations",
    "ms_invocations",  "ms_primitives",   "drawlists",
    "setup_triangles", "texel_fetches",
};

struct Options {
  std::filesystem::path rdc;
  std::filesystem::path outdir;
  std::filesystem::path project_root = DefaultProjectRoot();
  std::filesystem::path mesa_prefix;
  std::filesystem::path renderdoc_root;
  std::filesystem::path player;
  std::filesystem::path model;
  std::string case_name;
  unsigned width = 512;
  unsigned height = 512;
  unsigned trace_draw_actions = 0;
  bool case_explicit = false;
  bool width_explicit = false;
  bool height_explicit = false;
  bool trace_draw_actions_explicit = false;
  bool extent_from_manifest = false;
};

struct RunOutcome {
  bool success = false;
  std::string stage = "setup";
  std::string reason;
  int player_exit_code = -1;
  int model_exit_code = -1;
  std::filesystem::path report;
  std::filesystem::path counter;
  std::filesystem::path frame;
  std::filesystem::path stdout_log;
  std::filesystem::path stderr_log;
  unsigned frame_width = 0;
  unsigned frame_height = 0;
};

std::string Lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](char character) {
    return static_cast<char>(
        std::tolower(static_cast<unsigned char>(character)));
  });
  return value;
}

std::string Trim(std::string value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.front())))
    value.erase(value.begin());
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back())))
    value.pop_back();
  return value;
}

std::vector<std::string> Split(std::string line, char separator) {
  std::vector<std::string> parts;
  std::string part;
  std::istringstream input(line);
  while (std::getline(input, part, separator))
    parts.push_back(Trim(part));
  if (!line.empty() && line.back() == separator)
    parts.emplace_back();
  return parts;
}

bool ParsePositive(const std::string &text, unsigned *value) {
  if (!value || text.empty())
    return false;
  std::uint64_t parsed = 0;
  for (char character : text) {
    if (!std::isdigit(static_cast<unsigned char>(character)))
      return false;
    parsed = parsed * 10U + static_cast<unsigned>(character - '0');
    if (parsed > 0xffffffffULL)
      return false;
  }
  if (parsed == 0)
    return false;
  *value = static_cast<unsigned>(parsed);
  return true;
}

bool IsSafeCaseName(const std::string &value) {
  if (value.empty())
    return false;
  return std::all_of(value.begin(), value.end(), [](char character) {
    return std::isalnum(static_cast<unsigned char>(character)) ||
           character == '_' || character == '-' || character == '.' ||
           character == '*';
  });
}

std::string SafeCaseName(std::string value) {
  for (char &character : value) {
    if (!(std::isalnum(static_cast<unsigned char>(character)) ||
          character == '_' || character == '-' || character == '.'))
      character = '-';
  }
  while (!value.empty() && (value.front() == '.' || value.front() == '-'))
    value.erase(value.begin());
  while (!value.empty() && (value.back() == '.' || value.back() == '-'))
    value.pop_back();
  return value.empty() ? "rdc" : value;
}

std::string JsonEscape(const std::string &value) {
  std::ostringstream output;
  for (unsigned char character : value) {
    switch (character) {
      case '\\': output << "\\\\"; break;
      case '"': output << "\\\""; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (character < 0x20U) {
          output << "\\u00" << "0123456789abcdef"[(character >> 4U) & 0xfU]
                 << "0123456789abcdef"[character & 0xfU];
        } else {
          output << static_cast<char>(character);
        }
        break;
    }
  }
  return output.str();
}

bool ReadText(const std::filesystem::path &path, std::string *text) {
  if (!text)
    return false;
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return false;
  std::ostringstream buffer;
  buffer << input.rdbuf();
  if (!input.good() && !input.eof())
    return false;
  *text = buffer.str();
  return true;
}

bool AtomicWriteText(const std::filesystem::path &path, const std::string &text,
                     std::string *error) {
  std::error_code filesystem_error;
  std::filesystem::create_directories(path.parent_path(), filesystem_error);
  if (filesystem_error) {
    if (error)
      *error = "cannot create output directory: " + filesystem_error.message();
    return false;
  }
  std::filesystem::path temporary = path.parent_path() / ".";
  temporary += path.filename();
  temporary += ".tmp";
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
      if (error)
        *error = "cannot open temporary output: " + PathToUtf8(temporary);
      return false;
    }
    output << text;
    output.close();
    if (!output) {
      if (error)
        *error = "cannot write temporary output: " + PathToUtf8(temporary);
      return false;
    }
  }
  std::filesystem::remove(path, filesystem_error);
  filesystem_error.clear();
  std::filesystem::rename(temporary, path, filesystem_error);
  if (filesystem_error) {
    if (error)
      *error = "cannot publish output: " + filesystem_error.message();
    std::filesystem::remove(temporary, filesystem_error);
    return false;
  }
  return true;
}

bool NonEmptyFile(const std::filesystem::path &path) {
  std::error_code error;
  return std::filesystem::is_regular_file(path, error) &&
         std::filesystem::file_size(path, error) > 0 && !error;
}

std::string PrependPath(const std::filesystem::path &prefix,
                        const std::string &existing) {
#ifdef _WIN32
  constexpr char separator = ';';
#else
  constexpr char separator = ':';
#endif
  const std::string prefix_utf8 = PathToUtf8(prefix);
  return existing.empty() ? prefix_utf8
                          : prefix_utf8 + separator + existing;
}

std::string Environment(const char *name) {
  const char *value = std::getenv(name);
  return value ? std::string(value) : std::string();
}

std::string DynamicLibraryVariable() {
#ifdef _WIN32
  return "PATH";
#elif defined(__APPLE__)
  return "DYLD_LIBRARY_PATH";
#else
  return "LD_LIBRARY_PATH";
#endif
}

std::filesystem::path FindExisting(
    const std::vector<std::filesystem::path> &candidates) {
  for (const auto &candidate : candidates) {
    if (!candidate.empty() && std::filesystem::is_regular_file(candidate))
      return candidate;
  }
  return candidates.empty() ? std::filesystem::path{} : candidates.front();
}

std::filesystem::path RuntimeLibrary(const std::filesystem::path &prefix,
                                     const std::string &unix_name,
                                     const std::string &windows_name) {
  return FindExisting({prefix / "lib" / ("lib" + unix_name + ".dylib"),
                       prefix / "lib" / ("lib" + unix_name + ".so"),
                       prefix / "bin" / windows_name,
                       prefix / "lib" / windows_name});
}

bool ParseTraceDrawActions(const std::filesystem::path &log, unsigned *value) {
  if (!value)
    return false;
  std::ifstream input(log);
  std::string line;
  const std::regex pattern(R"(^Trace draw actions:\s*([0-9]+)\s*$)");
  std::smatch match;
  while (std::getline(input, line)) {
    if (!std::regex_match(line, match, pattern))
      continue;
    std::uint64_t parsed = 0;
    for (char character : match[1].str()) {
      parsed = parsed * 10U + static_cast<unsigned>(character - '0');
      if (parsed > 0xffffffffULL)
        return false;
    }
    *value = static_cast<unsigned>(parsed);
    return true;
  }
  return false;
}

std::filesystem::path PlayerPath(const Options &options,
                                 const RuntimeConfig &config) {
  if (!options.player.empty())
    return ResolveExecutable(options.player);
  const std::filesystem::path configured = config.Path("PVRGPU_RDC_PLAYER");
  if (!configured.empty())
    return ResolveExecutable(configured);
  return ResolveExecutable(options.renderdoc_root / "bin" /
                           "renderdoc-mesa-player");
}

std::map<std::string, std::string> BaseEnvironment(
    const std::filesystem::path &artifact_root,
    const std::filesystem::path &mesa_prefix) {
  const std::filesystem::path temporary = artifact_root / "tmp";
  const std::filesystem::path cache = artifact_root / "xdg-cache";
  std::filesystem::create_directories(temporary);
  std::filesystem::create_directories(cache);
  std::map<std::string, std::string> environment = {
      {"LC_ALL", "C"},
      {"LANG", "C"},
      {"TZ", "UTC"},
      {"TMPDIR", PathToUtf8(temporary)},
      {"XDG_CACHE_HOME", PathToUtf8(cache)},
      {"EGL_PLATFORM", "surfaceless"},
      {"LIBGL_ALWAYS_SOFTWARE", "1"},
      {"MESA_LOADER_DRIVER_OVERRIDE", "swrast"},
      {"MESA_SHADER_CACHE_DISABLE", "true"},
      {"RENDERDOC_OUTPUT_FROM_DRAW_FBO", "1"},
  };
  const std::string loader_variable = DynamicLibraryVariable();
  std::string loader_path =
      PrependPath(mesa_prefix / "lib", Environment(loader_variable.c_str()));
#ifdef _WIN32
  loader_path = PrependPath(mesa_prefix / "bin", loader_path);
#endif
  environment[loader_variable] = loader_path;
  return environment;
}

void PrintUsage(std::ostream &output) {
  output << "Usage: " << kBackend
         << "[.exe] FILE.rdc [--outdir DIR] [--case NAME]"
            " [--width W --height H]\n\n"
         << "Native single-RDC " << kBackend
         << " replay adapter. It launches the pinned RenderDoc/Mesa runtime"
            " directly without shell scripts.\n\n"
         << "Options:\n"
         << "  --rdc FILE             Input RDC (positional FILE is equivalent)\n"
         << "  --outdir, --out-dir D  Artifact directory\n"
         << "  --case NAME            Safe case name\n"
         << "  --width W --height H   Output extent; manifest metadata is used when available\n"
         << "  --project-root DIR     Repository root containing config/local.env\n"
         << "  --mesa-prefix DIR      Override backend Mesa install prefix\n"
         << "  --renderdoc-root DIR   Override RenderDoc Mesa runtime root\n"
         << "  --player FILE          Override renderdoc-mesa-player executable\n";
  if (kBackend == "pvrgpu") {
    output << "  --model FILE           Override pvrgpu-model-stub executable\n"
           << "  --trace-draw-actions N Golden draw action metadata\n";
  }
  output << "  -h, --help             Show this help\n";
}

bool ParseArguments(int argc, char **argv, Options *options,
                    std::string *error) {
  if (!options || !error)
    return false;
  auto require_value = [&](int *index, const std::string &name) -> std::string {
    if (*index + 1 >= argc) {
      *error = name + " requires a value";
      return {};
    }
    ++*index;
    return argv[*index];
  };

  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "-h" || argument == "--help") {
      PrintUsage(std::cout);
      std::exit(0);
    }
    if (argument == "--rdc") {
      const std::string value = require_value(&index, argument);
      if (value.empty())
        return false;
      options->rdc = value;
    } else if (argument == "--outdir" || argument == "--out-dir") {
      const std::string value = require_value(&index, argument);
      if (value.empty())
        return false;
      options->outdir = value;
    } else if (argument == "--case") {
      options->case_name = require_value(&index, argument);
      if (options->case_name.empty())
        return false;
      options->case_explicit = true;
    } else if (argument == "--width") {
      const std::string value = require_value(&index, argument);
      if (!ParsePositive(value, &options->width)) {
        *error = "--width must be a positive integer";
        return false;
      }
      options->width_explicit = true;
    } else if (argument == "--height") {
      const std::string value = require_value(&index, argument);
      if (!ParsePositive(value, &options->height)) {
        *error = "--height must be a positive integer";
        return false;
      }
      options->height_explicit = true;
    } else if (argument == "--trace-draw-actions") {
      const std::string value = require_value(&index, argument);
      if (!ParsePositive(value, &options->trace_draw_actions)) {
        if (value == "0") {
          options->trace_draw_actions = 0;
        } else {
          *error = "--trace-draw-actions must be a non-negative integer";
          return false;
        }
      }
      options->trace_draw_actions_explicit = true;
    } else if (argument == "--project-root") {
      options->project_root = require_value(&index, argument);
    } else if (argument == "--mesa-prefix") {
      options->mesa_prefix = require_value(&index, argument);
    } else if (argument == "--renderdoc-root") {
      options->renderdoc_root = require_value(&index, argument);
    } else if (argument == "--player") {
      options->player = require_value(&index, argument);
    } else if (argument == "--model") {
      options->model = require_value(&index, argument);
    } else if (!argument.empty() && argument.front() == '-') {
      *error = "unknown option: " + argument;
      return false;
    } else if (options->rdc.empty()) {
      options->rdc = argument;
    } else {
      *error = "only one RDC input may be specified";
      return false;
    }
  }

  if (options->rdc.empty()) {
    *error = "an input FILE.rdc is required";
    return false;
  }
  if (Lower(PathToUtf8(options->rdc.extension())) != ".rdc") {
    *error = "input must have the .rdc extension";
    return false;
  }
  if (!std::filesystem::is_regular_file(options->rdc)) {
    *error = "RDC is missing or is not a regular file: " +
             PathToUtf8(options->rdc);
    return false;
  }
  if (options->case_name.empty())
    options->case_name = SafeCaseName(PathToUtf8(options->rdc.stem()));
  if (!IsSafeCaseName(options->case_name)) {
    *error = "unsafe case name: " + options->case_name;
    return false;
  }
  if (options->outdir.empty()) {
    std::filesystem::path directory_name = options->rdc.stem();
    directory_name += "-";
    directory_name += std::string(kBackend);
    options->outdir = std::filesystem::current_path() / directory_name;
  }
  return true;
}

void ApplyManifestMetadata(const std::filesystem::path &manifest,
                           const std::string &digest, Options *options) {
  if (!options)
    return;
  std::ifstream input(manifest);
  std::string line;
  if (!std::getline(input, line))
    return;
  const std::vector<std::string> headers = Split(line, '\t');
  std::map<std::string, std::size_t> columns;
  for (std::size_t index = 0; index < headers.size(); ++index)
    columns[headers[index]] = index;
  if (!columns.count("rdc_sha256") || !columns.count("case") ||
      !columns.count("width") || !columns.count("height"))
    return;
  while (std::getline(input, line)) {
    const std::vector<std::string> values = Split(line, '\t');
    auto value = [&](const std::string &name) -> std::string {
      const std::size_t index = columns[name];
      return index < values.size() ? values[index] : std::string();
    };
    if (Lower(value("rdc_sha256")) != digest)
      continue;
    if (!options->case_explicit && IsSafeCaseName(value("case")))
      options->case_name = value("case");
    unsigned parsed = 0;
    if (!options->width_explicit && ParsePositive(value("width"), &parsed)) {
      options->width = parsed;
      options->extent_from_manifest = true;
    }
    if (!options->height_explicit && ParsePositive(value("height"), &parsed)) {
      options->height = parsed;
      options->extent_from_manifest = true;
    }
    return;
  }
}

void ApplyRecorderManifestMetadata(const std::filesystem::path &rdc_path,
                                   Options *options) {
  if (!options || options->case_explicit)
    return;
  const std::filesystem::path parent = rdc_path.parent_path();
  const std::vector<std::filesystem::path> candidates = {
      parent.parent_path() / "manifest.txt",
      parent / "manifest.txt",
  };
  const std::string filename = rdc_path.filename().string();
  for (const auto &manifest_path : candidates) {
    std::ifstream input(manifest_path);
    if (!input)
      continue;
    std::string line;
    while (std::getline(input, line)) {
      if (line.empty() || line[0] == '#')
        continue;
      const std::vector<std::string> parts = Split(line, '|');
      if (parts.size() >= 3) {
        const std::string case_name = Trim(parts[1]);
        const std::string rdc_name = Trim(parts[2]);
        if (rdc_name == filename && IsSafeCaseName(case_name)) {
          options->case_name = case_name;
          return;
        }
      }
    }
  }
}

std::string ZeroCounterReport(const Options &options,
                              const std::string &note) {
  std::ostringstream output;
  output << "# Mesa llvmpipe Frame Counter Report\n\n"
         << "- Mesa: `26.2.1`\n"
         << "- Renderer: `llvmpipe clear-only synthetic zero-counter fallback`\n"
         << "- Counter owner: `Mesa llvmpipe`\n"
         << "- Frame selection markers: `replay`\n"
         << "- Note: `" << note << "`\n\n"
         << "## Per-frame counters\n\n| Frame | Marker";
  for (std::string_view field : kCounterFields)
    output << " | " << field;
  output << " |\n| ---: | ---:";
  for ([[maybe_unused]] std::string_view field : kCounterFields)
    output << " | ---:";
  output << " |\n| 1 | " << options.case_name;
  for ([[maybe_unused]] std::string_view field : kCounterFields)
    output << " | 0";
  output << " |\n";
  return output.str();
}

bool ParseGoldenCounters(const std::string &report,
                         std::map<std::string, std::uint64_t> *counters,
                         std::string *error) {
  if (!counters || !error)
    return false;
  std::istringstream input(report);
  std::string line;
  std::vector<std::string> headers;
  while (std::getline(input, line)) {
    if (line.empty() || line.front() != '|')
      continue;
    std::vector<std::string> cells = Split(line, '|');
    if (!cells.empty() && cells.front().empty())
      cells.erase(cells.begin());
    if (!cells.empty() && cells.back().empty())
      cells.pop_back();
    if (headers.empty()) {
      const bool has_counter = std::any_of(
          kCounterFields.begin(), kCounterFields.end(), [&](std::string_view field) {
            return std::find(cells.begin(), cells.end(), field) != cells.end();
          });
      if (has_counter)
        headers = std::move(cells);
      continue;
    }
    if (line.find("---") != std::string::npos)
      continue;
    if (cells.size() != headers.size())
      continue;
    try {
      for (std::string_view field : kCounterFields) {
        const auto found = std::find(headers.begin(), headers.end(), field);
        if (found == headers.end()) {
          *error = "Golden report is missing counter column " +
                   std::string(field);
          return false;
        }
        const std::size_t index =
            static_cast<std::size_t>(std::distance(headers.begin(), found));
        const std::string value = cells[index];
        if (value.empty() ||
            !std::all_of(value.begin(), value.end(), [](char character) {
              return std::isdigit(static_cast<unsigned char>(character));
            })) {
          *error = "Golden counter " + std::string(field) +
                   " is not a non-negative integer";
          return false;
        }
        (*counters)[std::string(field)] = std::stoull(value);
      }
    } catch (const std::exception &exception) {
      *error = "cannot parse Golden counters: " + std::string(exception.what());
      return false;
    }
    return true;
  }
  *error = "Golden report has no per-frame counter row";
  return false;
}

bool ParsePvrgpuCounters(const std::string &jsonl,
                         std::map<std::string, std::uint64_t> *counters,
                         std::string *error) {
  if (!counters || !error)
    return false;
  std::istringstream input(jsonl);
  std::string line;
  while (std::getline(input, line)) {
    if (line.find("\"type\":\"counter\"") == std::string::npos)
      continue;
    for (std::string_view field : kCounterFields) {
      const std::regex pattern("\\\"" + std::string(field) +
                               "\\\"\\s*:\\s*([0-9]+)");
      std::smatch match;
      if (!std::regex_search(line, match, pattern)) {
        *error = "PvrGPU counter message is missing " + std::string(field);
        return false;
      }
      try {
        (*counters)[std::string(field)] = std::stoull(match[1].str());
      } catch (const std::exception &exception) {
        *error = "cannot parse PvrGPU counter " + std::string(field) + ": " +
                 exception.what();
        return false;
      }
    }
    return true;
  }
  *error = "PvrGPU output has no counter message";
  return false;
}

bool ValidatePvrgpuCompletion(const std::string &jsonl, std::string *error) {
  if (!error)
    return false;
  const std::regex type_pattern(R"json("type"\s*:\s*"([^"]+)")json");
  const std::regex leaks_pattern(
      R"json("pool_leaks"\s*:\s*([0-9]+))json");
  std::size_t hello_count = 0;
  std::size_t counter_count = 0;
  std::size_t done_count = 0;
  std::istringstream input(jsonl);
  std::string line;
  while (std::getline(input, line)) {
    std::smatch type_match;
    if (!std::regex_search(line, type_match, type_pattern))
      continue;
    const std::string type = type_match[1].str();
    if (type == "error") {
      *error = "PvrGPU model emitted an error message";
      return false;
    }
    if (type == "hello") {
      ++hello_count;
      continue;
    }
    if (type == "counter") {
      ++counter_count;
      continue;
    }
    if (type != "done")
      continue;
    ++done_count;
    std::smatch leaks_match;
    if (!std::regex_search(line, leaks_match, leaks_pattern)) {
      *error = "PvrGPU done message is missing pool_leaks";
      return false;
    }
    try {
      if (std::stoull(leaks_match[1].str()) != 0) {
        *error = "PvrGPU model reported a non-zero MemoryPool leak count";
        return false;
      }
    } catch (const std::exception &exception) {
      *error = "cannot parse PvrGPU pool_leaks: " +
               std::string(exception.what());
      return false;
    }
  }
  if (hello_count != 1 || counter_count != 1 || done_count != 1) {
    std::ostringstream reason;
    reason << "PvrGPU protocol requires exactly one hello, counter, and done "
              "message (got hello="
           << hello_count << ", counter=" << counter_count
           << ", done=" << done_count << ')';
    *error = reason.str();
    return false;
  }
  return true;
}

std::string FormatCounters(
    const std::map<std::string, std::uint64_t> &counters) {
  std::ostringstream output;
  for (std::string_view field : kCounterFields)
    output << field << '=' << counters.at(std::string(field)) << '\n';
  return output.str();
}

std::string InjectRdcDigest(const std::string &jsonl,
                            const std::string &digest) {
  std::istringstream input(jsonl);
  std::ostringstream output;
  std::string line;
  while (std::getline(input, line)) {
    if (line.find("\"type\":\"hello\"") != std::string::npos &&
        line.find("\"backend\":\"pvrgpu\"") != std::string::npos &&
        line.find("\"rdc_sha256\"") == std::string::npos) {
      const std::size_t end = line.rfind('}');
      if (end != std::string::npos)
        line.insert(end, ",\"rdc_sha256\":\"" + digest + "\"");
    }
    output << line << '\n';
  }
  return output.str();
}

std::filesystem::path CapturePng(const std::string &jsonl,
                                 const std::filesystem::path &directory) {
  std::istringstream input(jsonl);
  std::string line;
  const std::regex marker(R"(^@CAPTURE:.*\spng=([A-Za-z0-9_.-]+)\s*$)");
  std::smatch match;
  while (std::getline(input, line)) {
    if (!std::regex_match(line, match, marker))
      continue;
    const std::filesystem::path candidate = directory / match[1].str();
    if (Lower(PathToUtf8(candidate.extension())) == ".png" &&
        NonEmptyFile(candidate))
      return candidate;
  }
  return {};
}

std::size_t CountDriverEvents(const std::string &text,
                              std::string_view event) {
  const std::string marker = "event=" + std::string(event);
  std::istringstream input(text);
  std::size_t count = 0;
  std::string line;
  while (std::getline(input, line)) {
    const std::size_t position = line.find(marker);
    if (position == std::string::npos)
      continue;
    const std::size_t end = position + marker.size();
    if (end == line.size() ||
        std::isspace(static_cast<unsigned char>(line[end])))
      ++count;
  }
  return count;
}

bool ReadPngExtent(const std::filesystem::path &path, unsigned *width,
                   unsigned *height) {
  if (!width || !height)
    return false;
  std::ifstream input(path, std::ios::binary);
  std::array<unsigned char, 24> header{};
  if (!input.read(reinterpret_cast<char *>(header.data()),
                  static_cast<std::streamsize>(header.size())))
    return false;
  constexpr std::array<unsigned char, 8> kPngSignature = {
      0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};
  if (!std::equal(kPngSignature.begin(), kPngSignature.end(), header.begin()) ||
      header[12] != 'I' || header[13] != 'H' || header[14] != 'D' ||
      header[15] != 'R')
    return false;
  auto read_be32 = [&](std::size_t offset) {
    return (static_cast<std::uint32_t>(header[offset]) << 24U) |
           (static_cast<std::uint32_t>(header[offset + 1]) << 16U) |
           (static_cast<std::uint32_t>(header[offset + 2]) << 8U) |
           static_cast<std::uint32_t>(header[offset + 3]);
  };
  const std::uint32_t parsed_width = read_be32(16);
  const std::uint32_t parsed_height = read_be32(20);
  if (parsed_width == 0 || parsed_height == 0)
    return false;
  *width = parsed_width;
  *height = parsed_height;
  return true;
}

bool CopyArtifact(const std::filesystem::path &source,
                  const std::filesystem::path &destination,
                  std::string *error) {
  if (source.empty())
    return true;
  std::error_code filesystem_error;
  std::filesystem::copy_file(source, destination,
                             std::filesystem::copy_options::overwrite_existing,
                             filesystem_error);
  if (filesystem_error) {
    if (error)
      *error = "cannot copy framebuffer artifact: " +
               filesystem_error.message();
    return false;
  }
  return true;
}

std::filesystem::path FindGalliumDriver(
    const std::filesystem::path &mesa_prefix) {
  const std::filesystem::path dri = mesa_prefix / "lib" / "dri";
  const std::filesystem::path direct = FindExisting(
      {dri / "swrast_dri.dylib", dri / "swrast_dri.so", dri / "swrast_dri.dll"});
  if (std::filesystem::is_regular_file(direct))
    return direct;
  std::vector<std::filesystem::path> candidates;
  std::error_code error;
  const std::filesystem::path library_dir = mesa_prefix / "lib";
  for (const auto &entry : std::filesystem::directory_iterator(library_dir, error)) {
    if (error || !entry.is_regular_file())
      continue;
    const std::string filename = Lower(PathToUtf8(entry.path().filename()));
    if (filename.rfind("libgallium-", 0) == 0 ||
        filename.rfind("gallium-", 0) == 0)
      candidates.push_back(entry.path());
  }
  std::sort(candidates.begin(), candidates.end());
  return candidates.empty() ? std::filesystem::path{} : candidates.front();
}

std::filesystem::path PrepareDriverSearchPath(
    const std::filesystem::path &mesa_prefix,
    const std::filesystem::path &artifact_root, std::string *error) {
  const std::filesystem::path direct_dir = mesa_prefix / "lib" / "dri";
  if (std::filesystem::is_regular_file(direct_dir / "swrast_dri.dylib") ||
      std::filesystem::is_regular_file(direct_dir / "swrast_dri.so") ||
      std::filesystem::is_regular_file(direct_dir / "swrast_dri.dll"))
    return direct_dir;
  const std::filesystem::path gallium = FindGalliumDriver(mesa_prefix);
  if (gallium.empty()) {
    *error = "Mesa pvrgpu runtime has no swrast DRI or libgallium library under " +
             PathToUtf8(mesa_prefix / "lib");
    return {};
  }
  const std::filesystem::path driver_dir = artifact_root / "dri";
  std::filesystem::create_directories(driver_dir);
  std::filesystem::path link = driver_dir / "swrast_dri";
  link += gallium.extension();
  std::error_code filesystem_error;
  std::filesystem::remove(link, filesystem_error);
  filesystem_error.clear();
  std::filesystem::create_symlink(gallium, link, filesystem_error);
  if (filesystem_error) {
    filesystem_error.clear();
    std::filesystem::copy_file(
        gallium, link, std::filesystem::copy_options::overwrite_existing,
        filesystem_error);
  }
  if (filesystem_error) {
    *error = "cannot stage Mesa pvrgpu DRI library: " +
             filesystem_error.message();
    return {};
  }
  return driver_dir;
}

RunOutcome RunLlvmPipe(const Options &options, const RuntimeConfig &config,
                       const std::string &digest) {
  RunOutcome outcome;
  const std::filesystem::path player = PlayerPath(options, config);
  if (!std::filesystem::is_regular_file(player)) {
    outcome.reason = "RenderDoc player is missing: " + PathToUtf8(player);
    return outcome;
  }
  if (!std::filesystem::is_directory(options.mesa_prefix)) {
    outcome.reason = "llvmpipe Mesa prefix is missing: " +
                     PathToUtf8(options.mesa_prefix);
    return outcome;
  }

  const std::filesystem::path png_dir = options.outdir / "png";
  std::filesystem::create_directories(png_dir);
  std::filesystem::path png_name = options.rdc.stem();
  png_name += "_replay.png";
  const std::filesystem::path png = png_dir / png_name;
  const std::filesystem::path report = options.outdir / "Report.md";
  const std::filesystem::path player_stdout =
      options.outdir / "player.stdout.log";
  const std::filesystem::path player_stderr =
      options.outdir / "player.stderr.log";
  std::error_code filesystem_error;
  for (const auto &stale : {png, report, options.outdir / "counter.txt",
                            options.outdir / "frame.png"}) {
    std::filesystem::remove(stale, filesystem_error);
    filesystem_error.clear();
  }

  std::map<std::string, std::string> environment =
      BaseEnvironment(options.outdir, options.mesa_prefix);
  environment["GALLIUM_DRIVER"] = "llvmpipe";
  environment["PVRGPU_RDC_SHA256"] = digest;
  environment["LIBGL_DRIVERS_PATH"] =
      PathToUtf8(options.mesa_prefix / "lib" / "dri");
  environment["MESA_COUNTER_REPORT_PATH"] = PathToUtf8(report);
  environment["MESA_COUNTER_FRAME_SELECTION_MS"] = "replay";
  const std::filesystem::path proxy_egl = FindExisting(
      {options.renderdoc_root / "real" / "libMesaEGL.dylib",
       options.renderdoc_root / "real" / "libMesaEGL.so",
       options.mesa_prefix / "lib" / "libEGL.dylib",
       options.mesa_prefix / "lib" / "libEGL.so",
       options.mesa_prefix / "bin" / "libEGL.dll"});
  const std::filesystem::path proxy_gles = FindExisting(
      {options.renderdoc_root / "real" / "libMesaGLESv2.dylib",
       options.renderdoc_root / "real" / "libMesaGLESv2.so",
       options.mesa_prefix / "lib" / "libGLESv2.dylib",
       options.mesa_prefix / "lib" / "libGLESv2.so",
       options.mesa_prefix / "bin" / "libGLESv2.dll"});
  if (!std::filesystem::is_regular_file(proxy_egl) ||
      !std::filesystem::is_regular_file(proxy_gles)) {
    outcome.reason = "Mesa EGL/GLES replay libraries are missing";
    return outcome;
  }
  environment["RENDERDOC_MESA_EGL_PATH"] = PathToUtf8(proxy_egl);
  environment["RENDERDOC_MESA_GLES_PATH"] = PathToUtf8(proxy_gles);

  outcome.stage = "player";
  ProcessRequest request;
  request.executable = player;
  request.arguments = {PathToUtf8(options.rdc), PathToUtf8(png)};
  request.environment = std::move(environment);
  request.unset_environment = {"MESA_COUNTER_FRAME_TIME_MS",
                               "MESA_COUNTER_FORCE_END"};
  request.stdout_path = player_stdout;
  request.stderr_path = player_stderr;
  const ProcessResult result = RunProcess(request);
  outcome.player_exit_code = result.exit_code;
  outcome.stdout_log = player_stdout;
  outcome.stderr_log = player_stderr;
  if (!result.started) {
    outcome.reason = result.error;
    return outcome;
  }

  std::string stdout_text;
  ReadText(player_stdout, &stdout_text);
  std::string ignored;
  AtomicWriteText(options.outdir / "player-wrapper.stdout.log", stdout_text,
                  &ignored);
  std::cout << stdout_text;
  const bool zero_draws =
      stdout_text.find("Trace draw actions: 0") != std::string::npos;
  if (result.exit_code != 0) {
    outcome.reason = "RenderDoc llvmpipe replay exited with code " +
                     std::to_string(result.exit_code);
    return outcome;
  }
  if (zero_draws && NonEmptyFile(png)) {
    const std::string note =
        "RenderDoc reported zero draw actions; API pipeline counters are "
        "normalized to zero for clear-only replay.";
    if (!AtomicWriteText(report, ZeroCounterReport(options, note),
                         &outcome.reason))
      return outcome;
  }
  if (!NonEmptyFile(report)) {
    outcome.reason = "llvmpipe replay did not produce Report.md";
    return outcome;
  }

  std::string report_text;
  ReadText(report, &report_text);
  if (Lower(report_text).find("llvmpipe") == std::string::npos) {
    outcome.reason = "Golden Report.md does not identify llvmpipe";
    return outcome;
  }
  std::map<std::string, std::uint64_t> counters;
  if (!ParseGoldenCounters(report_text, &counters, &outcome.reason)) {
    outcome.stage = "counter";
    return outcome;
  }
  const std::filesystem::path counter = options.outdir / "counter.txt";
  if (!AtomicWriteText(counter, FormatCounters(counters), &outcome.reason)) {
    outcome.stage = "counter";
    return outcome;
  }
  std::filesystem::path frame;
  if (NonEmptyFile(png)) {
    frame = options.outdir / "frame.png";
    if (!CopyArtifact(png, frame, &outcome.reason)) {
      outcome.stage = "framebuffer";
      return outcome;
    }
  }
  outcome.success = true;
  outcome.stage = "complete";
  outcome.report = report;
  outcome.counter = counter;
  outcome.frame = frame;
  return outcome;
}

RunOutcome RunPvrgpu(const Options &options, const RuntimeConfig &config,
                     const std::string &digest) {
  RunOutcome outcome;
  const std::filesystem::path player = PlayerPath(options, config);
  if (!std::filesystem::is_regular_file(player)) {
    outcome.reason = "RenderDoc player is missing: " + PathToUtf8(player);
    return outcome;
  }
  if (!std::filesystem::is_directory(options.mesa_prefix)) {
    outcome.reason = "PvrGPU Mesa prefix is missing: " +
                     PathToUtf8(options.mesa_prefix);
    return outcome;
  }
  if (!std::filesystem::is_regular_file(options.model)) {
    outcome.reason = "PvrGPU model is missing: " + PathToUtf8(options.model);
    return outcome;
  }

  const std::filesystem::path artifact_root = options.outdir;
  const std::filesystem::path model_png_dir = artifact_root / "png";
  const std::filesystem::path player_png_dir = artifact_root / "player-png";
  std::filesystem::create_directories(model_png_dir);
  std::filesystem::create_directories(player_png_dir);

  const std::filesystem::path driver_search =
      PrepareDriverSearchPath(options.mesa_prefix, artifact_root,
                              &outcome.reason);
  if (driver_search.empty())
    return outcome;

  std::filesystem::path replay_png_name = options.rdc.stem();
  replay_png_name += "_replay.png";
  const std::filesystem::path player_png =
      player_png_dir / replay_png_name;
  const std::filesystem::path command = artifact_root / "driver-command.txt";
  const std::filesystem::path driver_counter =
      artifact_root / "driver-counter.txt";
  const std::filesystem::path player_stdout =
      artifact_root / "player.stdout.log";
  const std::filesystem::path player_stderr =
      artifact_root / "player.stderr.log";
  const std::filesystem::path model_stdout =
      artifact_root / "model.stdout.jsonl";
  const std::filesystem::path model_stderr =
      artifact_root / "model.stderr.log";
  /*
   * The in-process bridge runs the model from an atexit handler, so its
   * diagnosis lands in model.stderr.log after the replay has finished.  The
   * fallback stub must not write over it: when the bridge produced nothing,
   * the reason it produced nothing is the only thing worth reading.
   */
  const std::filesystem::path model_stub_stderr =
      artifact_root / "model-stub.stderr.log";
  std::filesystem::path probe_png_name = options.rdc.stem();
  probe_png_name += "_trace_probe.png";
  const std::filesystem::path probe_png =
      player_png_dir / probe_png_name;
  const std::filesystem::path probe_stdout =
      artifact_root / "player.trace-probe.stdout.log";
  const std::filesystem::path probe_stderr =
      artifact_root / "player.trace-probe.stderr.log";

  // RenderDoc's replay initial-state path queries every texture level before
  // copying captured mip chains.  glGetTexLevelParameteriv is core in GLES
  // 3.1, while forcing 3.0 sends the emulated DSA path through incomplete
  // replay metadata and silently truncates the restore to level zero.
  std::string gles = "3.1";
  if (options.case_name.rfind("dEQP-GLES32.", 0) == 0)
    gles = "3.2";
  else if (options.case_name.rfind("dEQP-GLES31.", 0) == 0)
    gles = "3.1";
  const std::string configured_gles =
      config.Get("PVRGPU_MESA_GLES_VERSION_OVERRIDE");
  if (!configured_gles.empty())
    gles = configured_gles;

  const std::filesystem::path egl =
      RuntimeLibrary(options.mesa_prefix, "EGL", "libEGL.dll");
  const std::filesystem::path gles_library =
      RuntimeLibrary(options.mesa_prefix, "GLESv2", "libGLESv2.dll");
  if (!std::filesystem::is_regular_file(egl) ||
      !std::filesystem::is_regular_file(gles_library)) {
    outcome.reason = "PvrGPU Mesa EGL/GLES runtime is incomplete under: " +
                     PathToUtf8(options.mesa_prefix);
    return outcome;
  }

  std::filesystem::path bridge = config.Path("PVRGPU_SYSTEMC_API_LIB");
  if (bridge.empty())
    bridge = config.Path("PVRGPU_SYSTEMC_BRIDGE");
  if (bridge.empty()) {
    const std::filesystem::path build_dir =
        config.Path("PVRGPU_BUILD_DIR", DefaultWorkRoot() / "build");
    const std::filesystem::path build_lib = build_dir / "lib";
    const std::filesystem::path build_bin = build_dir / "bin";
    bridge = FindExisting({build_lib / "libpvrgpu_systemc_bridge.dylib",
                           build_lib / "libpvrgpu_systemc_bridge.so",
                           build_bin / "pvrgpu_systemc_bridge.dll",
                           build_bin / "Debug" / "pvrgpu_systemc_bridge.dll",
                           build_bin / "Release" / "pvrgpu_systemc_bridge.dll",
                           build_bin / "RelWithDebInfo" /
                               "pvrgpu_systemc_bridge.dll",
                           build_bin / "MinSizeRel" /
                               "pvrgpu_systemc_bridge.dll",
                           build_lib / "pvrgpu_systemc_bridge.dll"});
  }

  auto make_environment = [&](bool enable_systemc, bool has_trace_actions,
                              unsigned trace_actions) {
    std::map<std::string, std::string> environment =
        BaseEnvironment(artifact_root, options.mesa_prefix);
    environment["GALLIUM_DRIVER"] = "pvrgpu";
    environment["LIBGL_DRIVERS_PATH"] = PathToUtf8(driver_search);
    environment["MESA_GLES_VERSION_OVERRIDE"] = gles;
    environment["RENDERDOC_MESA_EGL_PATH"] = PathToUtf8(egl);
    environment["RENDERDOC_MESA_GLES_PATH"] = PathToUtf8(gles_library);
    environment["PVRGPU_DRIVER_COMMAND_OUT"] = PathToUtf8(command);
    environment["PVRGPU_DRIVER_COUNTER_OUT"] = PathToUtf8(driver_counter);
    environment["PVRGPU_RDC_CASE_NAME"] = options.case_name;
    environment["PVRGPU_RDC_OUTPUT_WIDTH"] = std::to_string(options.width);
    environment["PVRGPU_RDC_OUTPUT_HEIGHT"] = std::to_string(options.height);
    if (has_trace_actions)
      environment["PVRGPU_RDC_TRACE_DRAW_ACTIONS"] =
          std::to_string(trace_actions);
    if (enable_systemc && std::filesystem::is_regular_file(bridge)) {
      environment["PVRGPU_SYSTEMC_API_LIB"] = PathToUtf8(bridge);
      environment["PVRGPU_SYSTEMC_JSONL_OUT"] = PathToUtf8(model_stdout);
      environment["PVRGPU_SYSTEMC_STDERR_OUT"] = PathToUtf8(model_stderr);
      environment["PVRGPU_SYSTEMC_OUTDIR"] = PathToUtf8(model_png_dir);
    }
    return environment;
  };

  std::error_code cleanup_error;
  for (const auto &stale : {player_png, probe_png, command, driver_counter,
                            player_stdout, player_stderr, probe_stdout,
                            probe_stderr, model_stdout, model_stderr,
                            model_stub_stderr,
                            artifact_root / "counter.txt",
                            artifact_root / "frame.png"}) {
    std::filesystem::remove(stale, cleanup_error);
    cleanup_error.clear();
  }

  bool has_trace_actions = options.trace_draw_actions_explicit;
  unsigned trace_actions = options.trace_draw_actions;
  if (!has_trace_actions) {
    outcome.stage = "trace-probe";
    ProcessRequest probe_request;
    probe_request.executable = player;
    probe_request.arguments = {PathToUtf8(options.rdc),
                               PathToUtf8(probe_png)};
    probe_request.environment = make_environment(false, false, 0);
    probe_request.unset_environment = {
        "PVRGPU_SYSTEMC_API_LIB", "PVRGPU_SYSTEMC_BRIDGE",
        "PVRGPU_SYSTEMC_JSONL_OUT", "PVRGPU_SYSTEMC_STDERR_OUT",
        "PVRGPU_SYSTEMC_OUTDIR"};
    probe_request.stdout_path = probe_stdout;
    probe_request.stderr_path = probe_stderr;
    const ProcessResult probe_result = RunProcess(probe_request);
    outcome.player_exit_code = probe_result.exit_code;
    outcome.stdout_log = probe_stdout;
    outcome.stderr_log = probe_stderr;
    if (!probe_result.started) {
      outcome.reason = probe_result.error;
      return outcome;
    }
    if (probe_result.exit_code != 0) {
      outcome.reason = "RenderDoc trace probe exited with code " +
                       std::to_string(probe_result.exit_code);
      return outcome;
    }
    if (!ParseTraceDrawActions(probe_stdout, &trace_actions)) {
      outcome.reason =
          "RenderDoc trace probe did not report 'Trace draw actions'";
      return outcome;
    }
    has_trace_actions = true;
    for (const auto &probe_artifact :
         {command, driver_counter, model_stdout, model_stderr}) {
      std::filesystem::remove(probe_artifact, cleanup_error);
      cleanup_error.clear();
    }
  }

  std::ostringstream runner_metadata;
  runner_metadata << "schema=pvrgpu.rdc-native-runner.v1\n"
                  << "backend=pvrgpu\n"
                  << "rdc=" << PathToUtf8(options.rdc) << '\n'
                  << "rdc_sha256=" << digest << '\n'
                  << "case=" << options.case_name << '\n'
                  << "mesa_prefix=" << PathToUtf8(options.mesa_prefix) << '\n'
                  << "renderdoc_root=" << PathToUtf8(options.renderdoc_root)
                  << '\n'
                  << "gles_version_override=" << gles << '\n'
                  << "trace_draw_actions="
                  << (has_trace_actions ? std::to_string(trace_actions)
                                        : std::string())
                  << '\n'
                  << "trace_draw_actions_source="
                  << (options.trace_draw_actions_explicit ? "argument-or-env"
                                                          : "player-probe")
                  << '\n'
                  << "systemc_api_lib="
                  << (std::filesystem::is_regular_file(bridge)
                          ? PathToUtf8(bridge)
                          : std::string())
                  << '\n';
  std::string ignored;
  AtomicWriteText(artifact_root / "runner.txt", runner_metadata.str(), &ignored);

  outcome.stage = "player";
  ProcessRequest player_request;
  player_request.executable = player;
  player_request.arguments = {PathToUtf8(options.rdc),
                              PathToUtf8(player_png)};
  player_request.environment =
      make_environment(true, has_trace_actions, trace_actions);
  player_request.unset_environment = {
      "PVRGPU_SYSTEMC_API_LIB", "PVRGPU_SYSTEMC_BRIDGE",
      "PVRGPU_SYSTEMC_JSONL_OUT", "PVRGPU_SYSTEMC_STDERR_OUT",
      "PVRGPU_SYSTEMC_OUTDIR"};
  player_request.stdout_path = player_stdout;
  player_request.stderr_path = player_stderr;
  const ProcessResult player_result = RunProcess(player_request);
  outcome.player_exit_code = player_result.exit_code;
  outcome.stdout_log = player_stdout;
  outcome.stderr_log = player_stderr;
  if (!player_result.started) {
    outcome.reason = player_result.error;
    return outcome;
  }
  if (player_result.exit_code != 0) {
    outcome.reason = "RenderDoc Mesa/Gallium pvrgpu replay exited with code " +
                     std::to_string(player_result.exit_code);
    return outcome;
  }
  std::string driver_counter_text;
  if (!ReadText(driver_counter, &driver_counter_text)) {
    outcome.stage = "driver-support";
    outcome.reason = "PvrGPU Gallium driver did not emit driver-counter.txt";
    return outcome;
  }
  const std::size_t unsupported_draws =
      CountDriverEvents(driver_counter_text, "unsupported_draw");
  if (unsupported_draws != 0) {
    outcome.stage = "driver-support";
    outcome.reason =
        "PvrGPU Gallium replay reported " +
        std::to_string(unsupported_draws) +
        " unsupported draw event(s) across replay passes; refusing to treat "
        "a later framebuffer blit as the workload result";
    return outcome;
  }
  if (!NonEmptyFile(command)) {
    outcome.stage = "driver-support";
    outcome.reason = "PvrGPU Gallium driver did not emit driver-command.txt";
    return outcome;
  }

  std::string native_bridge_output;
  const bool bridge_started =
      ReadText(model_stdout, &native_bridge_output) &&
      !native_bridge_output.empty();
  const bool bridge_completed =
      bridge_started &&
      native_bridge_output.find("\"type\":\"done\"") !=
          std::string::npos;
  if (!bridge_completed && !bridge_started) {
    /*
     * Only the in-process bridge can consume a draw_pco_sequence capsule; the
     * standalone stub reads the text form, which carries the outer command
     * without its nested draws.  Running it here would replace the bridge's
     * diagnosis with a parse error about a field it was never meant to see.
     */
    std::string capsule;
    if (ReadText(command, &capsule) &&
        capsule.find("command=draw_pco_sequence") != std::string::npos) {
      outcome.stage = "model";
      outcome.stdout_log = model_stdout;
      outcome.stderr_log = model_stderr;
      std::string bridge_error;
      ReadText(model_stderr, &bridge_error);
      const std::size_t last = bridge_error.find_last_not_of(" \t\r\n");
      if (last != std::string::npos)
        bridge_error.resize(last + 1);
      const std::size_t line = bridge_error.rfind('\n');
      outcome.reason =
          "PvrGPU SystemC model produced no output for the PCO sequence" +
          (bridge_error.empty()
               ? std::string()
               : ": " + bridge_error.substr(line == std::string::npos
                                                ? 0
                                                : line + 1));
      return outcome;
    }
    outcome.stage = "model";
    outcome.stdout_log = model_stdout;
    outcome.stderr_log = model_stub_stderr;
    ProcessRequest model_request;
    model_request.executable = ResolveExecutable(options.model);
    model_request.arguments = {"--driver-command", PathToUtf8(command),
                               "--outdir", PathToUtf8(model_png_dir)};
    model_request.stdout_path = model_stdout;
    model_request.stderr_path = model_stub_stderr;
    const ProcessResult model_result = RunProcess(model_request);
    outcome.model_exit_code = model_result.exit_code;
    if (!model_result.started) {
      outcome.reason = model_result.error;
      return outcome;
    }
    if (model_result.exit_code != 0) {
      outcome.reason = "PvrGPU SystemC model exited with code " +
                       std::to_string(model_result.exit_code);
      return outcome;
    }
  } else if (bridge_completed) {
    outcome.model_exit_code = 0;
  } else {
    // The in-process bridge already consumed the transient VBO, PCO binaries,
    // shared registers and sampled-image bytes.  A standalone retry can only
    // reload driver-command.txt metadata, so it would discard the real
    // payload and overwrite the first actionable SystemC error with a false
    // "empty PCO" failure.  Preserve the native JSONL and let protocol
    // validation below report its exact error record.
    outcome.model_exit_code = 1;
  }

  std::string model_text;
  if (!ReadText(model_stdout, &model_text)) {
    outcome.reason = "PvrGPU model stdout is missing";
    return outcome;
  }
  const std::string bound_jsonl = InjectRdcDigest(model_text, digest);
  if (!AtomicWriteText(model_stdout, bound_jsonl, &outcome.reason))
    return outcome;
  std::cout << bound_jsonl;

  if (!ValidatePvrgpuCompletion(bound_jsonl, &outcome.reason)) {
    outcome.stage = "model-protocol";
    outcome.stdout_log = model_stdout;
    outcome.stderr_log = model_stderr;
    return outcome;
  }

  std::map<std::string, std::uint64_t> counters;
  if (!ParsePvrgpuCounters(bound_jsonl, &counters, &outcome.reason)) {
    outcome.stage = "counter";
    return outcome;
  }
  const std::filesystem::path counter = artifact_root / "counter.txt";
  if (!AtomicWriteText(counter, FormatCounters(counters), &outcome.reason)) {
    outcome.stage = "counter";
    return outcome;
  }
  const std::filesystem::path source_frame =
      CapturePng(bound_jsonl, model_png_dir);
  std::filesystem::path frame;
  std::string player_output_text;
  const bool explicit_no_color_output =
      ReadText(player_stdout, &player_output_text) &&
      player_output_text.find("selected replay range has no color output") !=
          std::string::npos;
  if (source_frame.empty() && !explicit_no_color_output) {
    outcome.stage = "framebuffer";
    outcome.reason =
        "PvrGPU model did not emit a framebuffer PNG and the player did not "
        "report explicit no-color output evidence";
    return outcome;
  }
  if (!source_frame.empty()) {
    unsigned frame_width = 0;
    unsigned frame_height = 0;
    if (!ReadPngExtent(source_frame, &frame_width, &frame_height)) {
      outcome.stage = "framebuffer";
      outcome.reason = "PvrGPU model framebuffer is not a valid PNG";
      return outcome;
    }
    const bool extent_enforced =
        options.width_explicit || options.height_explicit ||
        options.extent_from_manifest;
    if (extent_enforced &&
        (frame_width != options.width || frame_height != options.height)) {
      outcome.stage = "framebuffer";
      outcome.reason =
          "PvrGPU model framebuffer extent mismatch: requested=" +
          std::to_string(options.width) + "x" +
          std::to_string(options.height) + " actual=" +
          std::to_string(frame_width) + "x" +
          std::to_string(frame_height);
      return outcome;
    }
    outcome.frame_width = frame_width;
    outcome.frame_height = frame_height;
    frame = artifact_root / "frame.png";
    if (!CopyArtifact(source_frame, frame, &outcome.reason)) {
      outcome.stage = "framebuffer";
      return outcome;
    }
  }
  outcome.success = true;
  outcome.stage = "complete";
  outcome.counter = counter;
  outcome.frame = frame;
  outcome.stdout_log = model_stdout;
  outcome.stderr_log = model_stderr;
  return outcome;
}

std::string RelativeOrAbsolute(const std::filesystem::path &root,
                               const std::filesystem::path &path) {
  if (path.empty())
    return {};
  std::error_code error;
  const std::filesystem::path relative =
      std::filesystem::relative(path, root, error);
  const std::string relative_text = PathToGenericUtf8(relative);
  if (!error && !relative.empty() && relative_text.rfind("..", 0) != 0)
    return relative_text;
  return PathToUtf8(path);
}

bool WriteBackendResult(const Options &options, const std::string &digest,
                        const RunOutcome &outcome,
                        std::chrono::milliseconds duration,
                        std::string *error) {
  const std::filesystem::path &root = options.outdir;
  std::ostringstream json;
  json << "{\n"
       << "  \"schema\": \"" << kResultSchema << "\",\n"
       << "  \"backend\": \"" << kBackend << "\",\n"
       << "  \"status\": \"" << (outcome.success ? "PASS" : "FAIL")
       << "\",\n"
       << "  \"stage\": \"" << JsonEscape(outcome.stage) << "\",\n"
       << "  \"reason\": \"" << JsonEscape(outcome.reason) << "\",\n"
       << "  \"input\": {\"path\": \"" << JsonEscape(PathToUtf8(options.rdc))
       << "\", \"sha256\": \"" << digest << "\"},\n"
       << "  \"case\": \"" << JsonEscape(options.case_name) << "\",\n"
       << "  \"width\": "
       << (outcome.frame_width ? outcome.frame_width : options.width) << ",\n"
       << "  \"height\": "
       << (outcome.frame_height ? outcome.frame_height : options.height) << ",\n"
       << "  \"duration_ms\": " << duration.count() << ",\n"
       << "  \"player_exit_code\": " << outcome.player_exit_code << ",\n"
       << "  \"model_exit_code\": " << outcome.model_exit_code << ",\n"
       << "  \"artifacts\": {\n"
       << "    \"report\": \""
       << JsonEscape(RelativeOrAbsolute(root, outcome.report)) << "\",\n"
       << "    \"counter\": \""
       << JsonEscape(RelativeOrAbsolute(root, outcome.counter)) << "\",\n"
       << "    \"frame\": \""
       << JsonEscape(RelativeOrAbsolute(root, outcome.frame)) << "\",\n"
       << "    \"stdout\": \""
       << JsonEscape(RelativeOrAbsolute(root, outcome.stdout_log)) << "\",\n"
       << "    \"stderr\": \""
       << JsonEscape(RelativeOrAbsolute(root, outcome.stderr_log)) << "\"\n"
       << "  }\n"
       << "}\n";
  return AtomicWriteText(root / "backend-result.json", json.str(), error);
}

}  // namespace
}  // namespace pvrgpu::rdc

int main(int argc, char **argv) {
  using namespace pvrgpu::rdc;
  Options options;
  std::string error;
  if (!ParseArguments(argc, argv, &options, &error)) {
    std::cerr << error << "\n\n";
    PrintUsage(std::cerr);
    return 2;
  }

  std::error_code filesystem_error;
  options.rdc = std::filesystem::absolute(options.rdc, filesystem_error);
  if (filesystem_error) {
    std::cerr << "cannot resolve RDC path: " << filesystem_error.message()
              << '\n';
    return 2;
  }
  options.outdir = std::filesystem::absolute(options.outdir, filesystem_error);
  if (filesystem_error) {
    std::cerr << "cannot resolve output path: " << filesystem_error.message()
              << '\n';
    return 2;
  }
  std::filesystem::create_directories(options.outdir, filesystem_error);
  if (filesystem_error) {
    std::cerr << "cannot create output directory: " << filesystem_error.message()
              << '\n';
    return 1;
  }

  std::string digest;
  if (!Sha256File(options.rdc, &digest, &error)) {
    std::cerr << error << '\n';
    return 1;
  }
  RuntimeConfig config(options.project_root);
  ApplyManifestMetadata(config.project_root() / "config" / "rdc-glbench-v1.tsv",
                        digest, &options);
  ApplyManifestMetadata(config.project_root() / "config" / "rdc-glmark2-80x60-v1.tsv",
                        digest, &options);
  ApplyManifestMetadata(config.project_root() / "config" / "rdc-glmark2-800x600-v1.tsv",
                        digest, &options);
  ApplyRecorderManifestMetadata(options.rdc, &options);
  const std::filesystem::path work_root =
      config.Path("PVRGPU_WORK_ROOT", DefaultWorkRoot());
  if (options.renderdoc_root.empty()) {
    options.renderdoc_root = config.Path(
        "PVRGPU_RENDERDOC_MESA_ROOT",
        work_root.parent_path() / "build" / "renderdoc-mesa");
  }
  if (options.mesa_prefix.empty()) {
    options.mesa_prefix =
        kBackend == "llvmpipe"
            ? config.Path("PVRGPU_LLVMPIPE_MESA_PREFIX",
                          work_root / "mesa-counter" / "install")
            : config.Path("PVRGPU_MESA_PVRGPU_PREFIX",
                          work_root / "mesa-pvrgpu" / "install");
  }
  if (kBackend == "pvrgpu" && options.model.empty()) {
    const std::filesystem::path build_dir =
        config.Path("PVRGPU_BUILD_DIR", work_root / "build");
    options.model = config.Path("PVRGPU_MODEL_STUB",
                                build_dir / "bin" / "pvrgpu-model-stub");
    options.model = ResolveExecutable(options.model);
  }
  if (!options.trace_draw_actions_explicit) {
    const std::string trace_actions = Environment("PVRGPU_RDC_TRACE_DRAW_ACTIONS");
    unsigned parsed = 0;
    if (ParsePositive(trace_actions, &parsed) || trace_actions == "0") {
      options.trace_draw_actions = parsed;
      options.trace_draw_actions_explicit = true;
    }
  }

  const auto started = std::chrono::steady_clock::now();
  RunOutcome outcome =
      kBackend == "llvmpipe" ? RunLlvmPipe(options, config, digest)
                             : RunPvrgpu(options, config, digest);
  const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - started);
  std::string result_error;
  if (!WriteBackendResult(options, digest, outcome, duration, &result_error)) {
    std::cerr << result_error << '\n';
    return 1;
  }
  if (!outcome.success) {
    std::cerr << kBackend << " failed at " << outcome.stage << ": "
              << outcome.reason << '\n';
    return 1;
  }
  return 0;
}
