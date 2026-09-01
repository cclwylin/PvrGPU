#include "pvrgpu_systemc_api.h"

#include <png.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <vector>

namespace {

std::filesystem::path g_test_root;

[[noreturn]] void Fail(const std::string &message) {
  std::fprintf(stderr, "systemc-api-bridge-test: %s\n", message.c_str());
  std::_Exit(EXIT_FAILURE);
}

void WriteBytes(const std::filesystem::path &path,
                const std::vector<std::uint8_t> &bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    Fail("cannot create sidecar fixture");
  output.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!output)
    Fail("cannot write sidecar fixture");
}

std::string ReadText(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return {};
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

std::vector<std::uint8_t> ReadPngRgba(const std::filesystem::path &path,
                                      png_uint_32 *width,
                                      png_uint_32 *height) {
  png_image image{};
  image.version = PNG_IMAGE_VERSION;
  if (!png_image_begin_read_from_file(&image, path.string().c_str()))
    Fail("cannot open deferred PNG");
  image.format = PNG_FORMAT_RGBA;
  std::vector<std::uint8_t> pixels(PNG_IMAGE_SIZE(image));
  if (!png_image_finish_read(&image, nullptr, pixels.data(), 0, nullptr)) {
    const std::string message =
        image.message[0] ? image.message : "unknown libpng error";
    png_image_free(&image);
    Fail("cannot decode deferred PNG: " + message);
  }
  *width = image.width;
  *height = image.height;
  png_image_free(&image);
  return pixels;
}

void VerifyDeferredModelAtExit() {
  const std::string jsonl = ReadText(g_test_root / "model.jsonl");
  if (jsonl.find("\"type\":\"done\"") == std::string::npos ||
      jsonl.find("\"driver_command\":\"draw_textured_triangles\"") ==
          std::string::npos ||
      jsonl.find("\"ia_vertices\":6") == std::string::npos ||
      jsonl.find("\"ps_invocations\":6") == std::string::npos) {
    Fail("deferred API execution did not produce the expected JSONL");
  }

  const std::filesystem::path png =
      g_test_root / "out" / "driver_textured_triangles_sample_000001.png";
  std::error_code error;
  if (!std::filesystem::is_regular_file(png, error) || error ||
      std::filesystem::file_size(png, error) < 8 || error) {
    Fail("deferred API execution did not produce a PNG");
  }
  png_uint_32 width = 0;
  png_uint_32 height = 0;
  const std::vector<std::uint8_t> pixels =
      ReadPngRgba(png, &width, &height);
  if (width != 3 || height != 2 || pixels.size() != 3 * 2 * 4)
    Fail("deferred PNG extent is not 3x2 RGBA8");
  const std::uint8_t row0_red = pixels[(0 * width + 1) * 4];
  const std::uint8_t row1_red = pixels[(1 * width + 1) * 4];
  if (!((row0_red == 30 && row1_red == 40) ||
        (row0_red == 40 && row1_red == 30))) {
    Fail("llvmpipe-compatible nearest-boundary interpolation regressed");
  }
  if (std::filesystem::exists(g_test_root / "texture.rgba8"))
    Fail("sidecar deletion fixture was unexpectedly restored");

  std::filesystem::remove_all(g_test_root, error);
  std::puts("systemc-api-bridge-test: PASS");
}

bool SubmitFailsWith(pvrgpu_systemc_submit_info *info,
                     const std::string &expected) {
  std::array<char, 256> error{};
  const int result =
      pvrgpu_systemc_submit_driver_command(info, error.data(), error.size());
  return result != 0 && std::string(error.data()).find(expected) !=
                            std::string::npos;
}

}  // namespace

int main() {
  const auto nonce = std::chrono::high_resolution_clock::now()
                         .time_since_epoch()
                         .count();
  g_test_root = std::filesystem::temp_directory_path() /
                ("pvrgpu-systemc-api-test-" + std::to_string(nonce));
  std::filesystem::create_directories(g_test_root / "out");

  const std::filesystem::path sidecar = g_test_root / "texture.rgba8";
  const std::string sidecar_text = sidecar.string();
  const std::string jsonl_text = (g_test_root / "model.jsonl").string();
  const std::string stderr_text = (g_test_root / "model.stderr.log").string();
  const std::string outdir_text = (g_test_root / "out").string();

  pvrgpu_systemc_driver_command command{};
  command.version = PVRGPU_SYSTEMC_API_VERSION;
  command.schema = "pvrgpu.driver-command.v1";
  command.producer = "pvrgpu-gallium-driver";
  command.command = "draw_textured_triangles";
  command.case_name = "systemc-api-bridge-test";
  command.format = "PIPE_FORMAT_R8G8B8A8_UNORM";
  command.frame = 1;
  command.framebuffer_width = 2;
  command.framebuffer_height = 2;
  command.width = 2;
  command.height = 2;
  command.clear_color_bits[3] = UINT32_C(0x3f800000);
  const std::uint32_t positions[6][2] = {
      {UINT32_C(0xbf800000), UINT32_C(0x3f800000)},
      {UINT32_C(0xbf800000), UINT32_C(0xbf800000)},
      {UINT32_C(0x3f800000), UINT32_C(0x3f800000)},
      {UINT32_C(0xbf800000), UINT32_C(0xbf800000)},
      {UINT32_C(0x3f800000), UINT32_C(0xbf800000)},
      {UINT32_C(0x3f800000), UINT32_C(0x3f800000)},
  };
  const std::uint32_t texcoords[6][2] = {
      {0, UINT32_C(0x3f800000)},
      {0, 0},
      {UINT32_C(0x3f800000), UINT32_C(0x3f800000)},
      {0, 0},
      {UINT32_C(0x3f800000), 0},
      {UINT32_C(0x3f800000), UINT32_C(0x3f800000)},
  };
  for (std::size_t vertex = 0; vertex < 6; ++vertex) {
    command.vertex_bits[vertex][0] = positions[vertex][0];
    command.vertex_bits[vertex][1] = positions[vertex][1];
    command.texcoord_bits[vertex][0] = texcoords[vertex][0];
    command.texcoord_bits[vertex][1] = texcoords[vertex][1];
  }
  command.texture_width = 2;
  command.texture_height = 2;
  command.texture_rgba8_path = sidecar_text.c_str();

  pvrgpu_systemc_submit_info info{};
  info.version = PVRGPU_SYSTEMC_API_VERSION;
  info.command = &command;
  info.jsonl_path = jsonl_text.c_str();
  info.stderr_path = stderr_text.c_str();
  info.outdir = outdir_text.c_str();
  info.memory_mode = "direct";

  command.version = 1;
  if (!SubmitFailsWith(&info, "command version"))
    Fail("API v1 command was not rejected");
  command.version = PVRGPU_SYSTEMC_API_VERSION;

  WriteBytes(sidecar, std::vector<std::uint8_t>(15, UINT8_C(0x11)));
  if (!SubmitFailsWith(&info, "truncated"))
    Fail("truncated sidecar was not rejected");

  WriteBytes(sidecar, std::vector<std::uint8_t>(17, UINT8_C(0x22)));
  if (!SubmitFailsWith(&info, "extra bytes"))
    Fail("oversized sidecar was not rejected");

  command.framebuffer_width = 3;
  command.framebuffer_height = 2;
  command.width = 3;
  command.height = 2;
  command.texture_width = 6;
  command.texture_height = 2;
  std::vector<std::uint8_t> texture;
  texture.reserve(6 * 2 * 4);
  for (std::size_t y = 0; y < 2; ++y) {
    for (std::uint8_t x = 1; x <= 6; ++x) {
      texture.push_back(static_cast<std::uint8_t>(x * 10));
      texture.push_back(0);
      texture.push_back(0);
      texture.push_back(255);
    }
  }
  WriteBytes(sidecar, texture);
  if (std::atexit(VerifyDeferredModelAtExit) != 0)
    Fail("cannot register deferred verifier");

  std::array<char, 256> error{};
  pvrgpu_systemc_driver_command clear{};
  clear.version = PVRGPU_SYSTEMC_API_VERSION;
  clear.schema = "pvrgpu.driver-command.v1";
  clear.producer = "pvrgpu-gallium-driver";
  clear.command = "clear_color";
  clear.case_name = "systemc-api-bridge-test-probe";
  clear.format = "PIPE_FORMAT_R8G8B8A8_UNORM";
  clear.frame = 1;
  clear.width = 1;
  clear.height = 1;
  clear.clear_color_bits[3] = UINT32_C(0x3f800000);
  info.command = &clear;
  if (pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size()) !=
      0) {
    Fail("valid probe clear submit failed: " + std::string(error.data()));
  }

  // Ordinary probe/clear traffic is still last-command-wins. It must never be
  // mistaken for the explicitly gated Ideas ordered PCO sequence.
  info.command = &command;
  error.fill(0);
  const int result =
      pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size());
  if (result != 0)
    Fail("valid API v2 submit failed: " + std::string(error.data()));

  WriteBytes(sidecar, std::vector<std::uint8_t>(48, UINT8_C(0x7f)));
  std::error_code remove_error;
  if (!std::filesystem::remove(sidecar, remove_error) || remove_error)
    Fail("cannot delete sidecar after synchronous submit");
  return EXIT_SUCCESS;
}
