#include "model_runner.h"
#include "pvrgpu_systemc_api.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <utility>

namespace {

std::mutex g_bridge_mutex;

struct PendingSubmit {
  pvrgpu::stub::Options options;
  std::string jsonl_path;
  std::string stderr_path;
  bool valid = false;
  bool executed = false;
};

PendingSubmit g_pending_submit;
bool g_atexit_registered = false;

void CopyError(char *error, std::size_t error_size, const std::string &message) {
  if (!error || error_size == 0)
    return;
  const std::size_t count = message.size() < error_size - 1 ? message.size()
                                                            : error_size - 1;
  for (std::size_t i = 0; i < count; ++i)
    error[i] = message[i];
  error[count] = '\0';
}

bool SetMemoryMode(const char *text, pvrgpu::stub::Options *options,
                   std::string *error) {
  if (!options || !error)
    return false;
  const std::string mode = text && text[0] ? text : "cache";
  if (mode == "direct") {
    options->memory_mode = pvrgpu::stub::MemoryMode::kDirect;
    options->cache_bypass = false;
    return true;
  }
  if (mode == "bypass") {
    options->memory_mode = pvrgpu::stub::MemoryMode::kBypass;
    options->cache_bypass = true;
    return true;
  }
  if (mode == "cache") {
    options->memory_mode = pvrgpu::stub::MemoryMode::kCache;
    options->cache_bypass = false;
    return true;
  }
  *error = "invalid memory mode for SystemC API: " + mode;
  return false;
}

bool CopyCommand(const pvrgpu_systemc_driver_command &source,
                 pvrgpu::stub::DriverCommand *destination,
                 std::string *error) {
  if (!destination || !error)
    return false;
  if (source.version != PVRGPU_SYSTEMC_API_VERSION) {
    *error = "unsupported SystemC API command version";
    return false;
  }
  if (!source.command || !source.command[0]) {
    *error = "missing SystemC API command";
    return false;
  }
  if (!source.case_name || !source.case_name[0]) {
    *error = "missing SystemC API case name";
    return false;
  }
  if (!source.format || !source.format[0]) {
    *error = "missing SystemC API format";
    return false;
  }

  pvrgpu::stub::DriverCommand command;
  command.enabled = true;
  command.schema = source.schema && source.schema[0]
                       ? source.schema
                       : "pvrgpu.driver-command.v1";
  command.producer = source.producer && source.producer[0]
                         ? source.producer
                         : "pvrgpu-gallium-driver";
  command.command = source.command;
  command.test_case = source.case_name;
  command.frame = source.frame;
  command.framebuffer_width = source.framebuffer_width != 0
                                  ? source.framebuffer_width
                                  : source.width;
  command.framebuffer_height = source.framebuffer_height != 0
                                   ? source.framebuffer_height
                                   : source.height;
  command.width = source.width;
  command.height = source.height;
  command.format = source.format;
  command.clear_color_bits = {source.clear_color_bits[0],
                              source.clear_color_bits[1],
                              source.clear_color_bits[2],
                              source.clear_color_bits[3]};
  for (std::size_t vertex = 0; vertex < command.vertex_bits.size(); ++vertex) {
    command.vertex_bits[vertex][0] = source.vertex_bits[vertex][0];
    command.vertex_bits[vertex][1] = source.vertex_bits[vertex][1];
  }
  command.fragment_color_bits = {source.fragment_color_bits[0],
                                 source.fragment_color_bits[1],
                                 source.fragment_color_bits[2],
                                 source.fragment_color_bits[3]};
  command.draw_count = source.draw_count;
  command.index_count = source.index_count;
  command.unique_vertices = source.unique_vertices;
  command.primitive_count = source.primitive_count;
  command.clip_primitives = source.clip_primitives;
  command.setup_triangles = source.setup_triangles;
  command.semantic_texel_fetches = source.semantic_texel_fetches;
  command.ia_vertices = source.ia_vertices;
  command.ia_primitives = source.ia_primitives;
  command.vs_invocations = source.vs_invocations;
  command.gs_invocations = source.gs_invocations;
  command.gs_primitives = source.gs_primitives;
  command.clip_invocations = source.clip_invocations;
  command.ps_invocations = source.ps_invocations;
  command.hs_invocations = source.hs_invocations;
  command.ds_invocations = source.ds_invocations;
  command.cs_invocations = source.cs_invocations;
  if (source.framebuffer_rgba8_path && source.framebuffer_rgba8_path[0])
    command.framebuffer_rgba8_path = source.framebuffer_rgba8_path;

  *destination = command;
  return true;
}

int RunModelToFiles(const pvrgpu::stub::Options &options,
                    const std::string &jsonl_path,
                    const std::string &stderr_path,
                    std::string *error) {
  std::ofstream jsonl(jsonl_path, std::ios::out | std::ios::trunc);
  if (!jsonl) {
    if (error)
      *error = "cannot open SystemC API jsonl_path: " + jsonl_path;
    return 1;
  }
  std::ofstream stderr_file;
  if (!stderr_path.empty()) {
    stderr_file.open(stderr_path, std::ios::out | std::ios::trunc);
    if (!stderr_file) {
      if (error)
        *error = "cannot open SystemC API stderr_path: " + stderr_path;
      return 1;
    }
  }

  std::streambuf *old_stdout = std::cout.rdbuf(jsonl.rdbuf());
  std::streambuf *old_stderr =
      stderr_file ? std::cerr.rdbuf(stderr_file.rdbuf()) : nullptr;
  const int result = pvrgpu::stub::RunConfiguredModel(options);
  std::cout.rdbuf(old_stdout);
  if (old_stderr)
    std::cerr.rdbuf(old_stderr);
  jsonl.close();
  if (stderr_file)
    stderr_file.close();

  if (result != 0 && error) {
    *error =
        "SystemC model returned non-zero status: " + std::to_string(result);
  }
  return result;
}

int FlushPendingSubmitLocked(std::string *error) {
  if (!g_pending_submit.valid || g_pending_submit.executed)
    return 0;
  g_pending_submit.executed = true;
  return RunModelToFiles(g_pending_submit.options, g_pending_submit.jsonl_path,
                         g_pending_submit.stderr_path, error);
}

void FlushPendingSubmitAtExit() {
  std::lock_guard<std::mutex> lock(g_bridge_mutex);
  std::string error;
  const int result = FlushPendingSubmitLocked(&error);
  if (result != 0) {
    std::cerr << "PvrGPU SystemC API deferred flush failed: " << error
              << '\n';
  }
}

}  // namespace

extern "C" int pvrgpu_systemc_submit_driver_command(
    const pvrgpu_systemc_submit_info *info, char *error,
    std::size_t error_size) {
  std::lock_guard<std::mutex> lock(g_bridge_mutex);
  std::string message;
  if (!info) {
    CopyError(error, error_size, "missing SystemC API submit info");
    return 2;
  }
  if (info->version != PVRGPU_SYSTEMC_API_VERSION) {
    CopyError(error, error_size, "unsupported SystemC API submit version");
    return 2;
  }
  if (!info->command) {
    CopyError(error, error_size, "missing SystemC API command payload");
    return 2;
  }
  if (!info->jsonl_path || !info->jsonl_path[0]) {
    CopyError(error, error_size, "missing SystemC API jsonl_path");
    return 2;
  }
  if (!info->outdir || !info->outdir[0]) {
    CopyError(error, error_size, "missing SystemC API outdir");
    return 2;
  }

  pvrgpu::stub::Options options;
  options.output_dir = info->outdir;
  if (!SetMemoryMode(info->memory_mode, &options, &message)) {
    CopyError(error, error_size, message);
    return 2;
  }
  if (!CopyCommand(*info->command, &options.driver_command, &message)) {
    CopyError(error, error_size, message);
    return 2;
  }

  PendingSubmit pending;
  pending.options = std::move(options);
  pending.jsonl_path = info->jsonl_path;
  if (info->stderr_path && info->stderr_path[0])
    pending.stderr_path = info->stderr_path;
  pending.valid = true;

  if (!g_atexit_registered) {
    if (std::atexit(FlushPendingSubmitAtExit) != 0) {
      CopyError(error, error_size,
                "cannot register SystemC API deferred flush handler");
      return 1;
    }
    g_atexit_registered = true;
  }

  g_pending_submit = std::move(pending);
  return 0;
}
