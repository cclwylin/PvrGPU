// Persistent-session regression for fixed external texture address reuse.
// Two cached textured jobs use the synthetic command's same
// kGlbenchTextureGpuAddress, but upload different bytes. HostWrite must
// invalidate both TCU and SLC so the second draw cannot reuse the first texel.

#include "pvrgpu_systemc_api.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

[[noreturn]] void Fail(const std::string &message) {
  std::fprintf(stderr, "systemc-api-texture-cache-coherence-test: %s\n",
               message.c_str());
  std::_Exit(EXIT_FAILURE);
}

void SetEnvironment(const char *name, const char *value) {
#if defined(_WIN32)
  if (_putenv_s(name, value ? value : "") != 0)
#else
  if ((value ? setenv(name, value, 1) : unsetenv(name)) != 0)
#endif
    Fail(std::string("cannot configure ") + name);
}

void WriteBytes(const std::filesystem::path &path,
                const std::vector<std::uint8_t> &bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    Fail("cannot create texture sidecar");
  output.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!output)
    Fail("cannot write texture sidecar");
}

std::string ReadText(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    Fail("cannot read model report");
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

std::uint64_t JsonUnsigned(const std::string &report,
                           const std::string &field) {
  const std::string key = "\"" + field + "\":";
  const std::size_t key_position = report.rfind(key);
  if (key_position == std::string::npos)
    Fail("model report is missing " + field);
  const char *begin = report.c_str() + key_position + key.size();
  char *end = nullptr;
  const unsigned long long value = std::strtoull(begin, &end, 10);
  if (end == begin)
    Fail("model report has a non-numeric " + field);
  return static_cast<std::uint64_t>(value);
}

pvrgpu_systemc_driver_command MakeCommand(const std::string &sidecar) {
  pvrgpu_systemc_driver_command command{};
  command.version = PVRGPU_SYSTEMC_API_VERSION;
  command.schema = "pvrgpu.driver-command.v1";
  command.producer = "pvrgpu-gallium-driver";
  command.command = "draw_textured_triangles";
  command.case_name = "persistent-cached-texture-address-reuse";
  command.format = "PIPE_FORMAT_R8G8B8A8_UNORM";
  command.frame = 1;
  command.framebuffer_width = 2;
  command.framebuffer_height = 2;
  command.width = 2;
  command.height = 2;
  command.clear_color_bits[3] = UINT32_C(0x3f800000);

  constexpr std::uint32_t kNegativeOne = UINT32_C(0xbf800000);
  constexpr std::uint32_t kPositiveOne = UINT32_C(0x3f800000);
  const std::uint32_t positions[6][2] = {
      {kNegativeOne, kPositiveOne}, {kNegativeOne, kNegativeOne},
      {kPositiveOne, kPositiveOne}, {kNegativeOne, kNegativeOne},
      {kPositiveOne, kNegativeOne}, {kPositiveOne, kPositiveOne},
  };
  const std::uint32_t texcoords[6][2] = {
      {0, kPositiveOne},
      {0, 0},
      {kPositiveOne, kPositiveOne},
      {0, 0},
      {kPositiveOne, 0},
      {kPositiveOne, kPositiveOne},
  };
  for (std::size_t vertex = 0; vertex < 6; ++vertex) {
    command.vertex_bits[vertex][0] = positions[vertex][0];
    command.vertex_bits[vertex][1] = positions[vertex][1];
    command.texcoord_bits[vertex][0] = texcoords[vertex][0];
    command.texcoord_bits[vertex][1] = texcoords[vertex][1];
  }
  command.texture_width = 2;
  command.texture_height = 2;
  command.texture_rgba8_path = sidecar.c_str();
  return command;
}

std::array<std::uint8_t, 16> RunJob(const std::filesystem::path &root,
                                    const std::filesystem::path &sidecar,
                                    const std::array<std::uint8_t, 4> &texel,
                                    std::uint64_t generation) {
  std::vector<std::uint8_t> texture(2 * 2 * 4);
  for (std::size_t pixel = 0; pixel < 4; ++pixel)
    std::copy(texel.begin(), texel.end(), texture.begin() + pixel * 4);
  WriteBytes(sidecar, texture);

  const std::filesystem::path folder =
      root / ("job-" + std::to_string(generation));
  std::filesystem::create_directory(folder);
  const std::string sidecar_text = sidecar.string();
  const std::string jsonl = (folder / "model.jsonl").string();
  const std::string stderr_path = (folder / "model.stderr.log").string();
  const std::string outdir = folder.string();
  pvrgpu_systemc_driver_command command = MakeCommand(sidecar_text);
  pvrgpu_systemc_submit_info submit{};
  submit.version = PVRGPU_SYSTEMC_API_VERSION;
  submit.command = &command;
  submit.jsonl_path = jsonl.c_str();
  submit.stderr_path = stderr_path.c_str();
  submit.outdir = outdir.c_str();
  submit.memory_mode = "cache";
  submit.submission_generation = generation;

  std::array<char, 512> error{};
  if (pvrgpu_systemc_submit_driver_command(&submit, error.data(),
                                           error.size()) != 0) {
    Fail("cached texture submit failed: " + std::string(error.data()));
  }

  std::array<std::uint8_t, 16> pixels{};
  pvrgpu_systemc_readback_info readback{};
  readback.version = PVRGPU_SYSTEMC_API_VERSION;
  readback.width = 2;
  readback.height = 2;
  readback.bytes_per_pixel = 4;
  readback.pixels = pixels.data();
  readback.pixels_size = pixels.size();
  readback.color_format = "PIPE_FORMAT_R8G8B8A8_UNORM";
  if (pvrgpu_systemc_flush_readback(&readback, error.data(), error.size()) !=
          0 ||
      readback.pixels_written != 1) {
    Fail("cached texture readback failed: " + std::string(error.data()));
  }

  for (std::size_t pixel = 0; pixel < 4; ++pixel) {
    if (!std::equal(texel.begin(), texel.end(), pixels.begin() + pixel * 4)) {
      Fail("persistent cached job returned stale texture bytes at pixel " +
           std::to_string(pixel));
    }
  }
  const std::string report = ReadText(jsonl);
  if (report.find("\"type\":\"done\"") == std::string::npos ||
      report.find("\"type\":\"error\"") != std::string::npos ||
      report.find("\"texture_memory_path\":\"cached\"") == std::string::npos) {
    Fail("cached texture job did not report successful cached execution");
  }
  if (JsonUnsigned(report, "texture_requests") == 0 ||
      JsonUnsigned(report, "texel_fetches") == 0 ||
      JsonUnsigned(report, "tcu_misses") == 0 ||
      JsonUnsigned(report, "slc_misses") == 0 ||
      JsonUnsigned(report, "dram_read_transactions") == 0) {
    Fail("host upload did not force a fresh TCU/SLC/DRAM refill");
  }
  return pixels;
}

} // namespace

int main() {
  SetEnvironment("PVRGPU_TEXTURE_MEMORY_PATH", "cached");
  SetEnvironment("PVRGPU_SYSTEMC_DISABLE_PNG", "1");
  SetEnvironment("PVRGPU_TEXTURE_LOD_MODE", "llvmpipe");

  const auto nonce =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      ("pvrgpu-texture-cache-coherence-" + std::to_string(nonce));
  std::filesystem::create_directories(root);
  const std::filesystem::path sidecar = root / "texture.rgba8";

  const std::array<std::uint8_t, 4> red{255, 0, 0, 255};
  const std::array<std::uint8_t, 4> green{0, 255, 0, 255};
  const auto first = RunJob(root, sidecar, red, 1);
  const auto second = RunJob(root, sidecar, green, 2);
  if (first == second)
    Fail("second cached job repeated the first job's pixels");

  std::error_code error;
  std::filesystem::remove_all(root, error);
  SetEnvironment("PVRGPU_TEXTURE_MEMORY_PATH", nullptr);
  SetEnvironment("PVRGPU_SYSTEMC_DISABLE_PNG", nullptr);
  SetEnvironment("PVRGPU_TEXTURE_LOD_MODE", nullptr);
  std::puts("systemc_api_texture_cache_coherence_test: PASS");
  return EXIT_SUCCESS;
}
