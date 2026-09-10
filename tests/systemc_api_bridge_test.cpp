#include "pvrgpu_systemc_api.h"

#include <png.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
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

void SetDisablePng(const char *value) {
#if defined(_WIN32)
  if (_putenv_s("PVRGPU_SYSTEMC_DISABLE_PNG", value ? value : "") != 0)
#else
  if ((value ? setenv("PVRGPU_SYSTEMC_DISABLE_PNG", value, 1)
             : unsetenv("PVRGPU_SYSTEMC_DISABLE_PNG")) != 0)
#endif
    Fail("cannot configure PNG output fixture");
}

std::string CounterPayload(const std::string &report) {
  const auto start = report.find("\"counters\":{");
  const auto end = report.find('\n', start);
  if (start == std::string::npos || end == std::string::npos)
    Fail("missing complete counter and DrawList payload");
  return report.substr(start, end - start);
}

void TestPngOutput(pvrgpu_systemc_submit_info info) {
  std::array<std::uint8_t, 24> baseline{};
  std::string baseline_counters;
  for (unsigned mode = 0; mode < 3; ++mode) {
    const auto folder = g_test_root / ("png-policy-" + std::to_string(mode));
    std::filesystem::create_directory(folder);
    const std::string jsonl = (folder / "model.jsonl").string();
    const std::string stderr_path = (folder / "model.stderr.log").string();
    const std::string outdir = folder.string();
    info.jsonl_path = jsonl.c_str();
    info.stderr_path = stderr_path.c_str();
    info.outdir = outdir.c_str();
    SetDisablePng(mode == 0 ? nullptr : mode == 1 ? "0" : "1");
    std::array<char, 512> error{};
    if (pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size()) != 0)
      Fail("PNG policy submission failed: " + std::string(error.data()));
    if (mode == 2) {
      // Rejected options must neither replace nor execute this pending draw.
      for (const char *invalid : {"true", "false", "on", "off", "-1", "2", "01", " 1", "1 "}) {
        SetDisablePng(invalid);
        if (!SubmitFailsWith(&info, "invalid PVRGPU_SYSTEMC_DISABLE_PNG"))
          Fail("non-boolean PNG policy was accepted");
      }
#if !defined(_WIN32)
      // Windows _putenv_s removes a variable when passed an empty value.
      SetDisablePng("");
      if (!SubmitFailsWith(&info, "invalid PVRGPU_SYSTEMC_DISABLE_PNG"))
        Fail("empty PNG policy was accepted");
#endif
      SetDisablePng("1");
      info.outdir = "";
      if (!SubmitFailsWith(&info, "missing SystemC API outdir"))
        Fail("PNG suppression bypassed the required output directory");
      info.outdir = outdir.c_str();
    }
    // A deferred submit owns its policy, independent of later environment.
    SetDisablePng(mode == 2 ? "0" : "1");
    std::array<std::uint8_t, 24> pixels{};
    pixels.fill(UINT8_C(0xa5));
    pvrgpu_systemc_readback_info readback{};
    readback.version = PVRGPU_SYSTEMC_API_VERSION;
    readback.width = 3;
    readback.height = 2;
    readback.bytes_per_pixel = 4;
    readback.pixels = pixels.data();
    readback.pixels_size = pixels.size();
    if (pvrgpu_systemc_flush_readback(&readback, error.data(), error.size()) != 0 ||
        readback.pixels_written != 1)
      Fail("PNG policy raw readback failed: " + std::string(error.data()));
    const auto report = ReadText(jsonl);
    if (report.find("\"type\":\"done\"") == std::string::npos ||
        report.find("\"type\":\"error\"") != std::string::npos ||
        report.find("\"ps_invocations\":6") == std::string::npos ||
        report.find("\"framebuffer_dram_readback_bytes\":24") == std::string::npos)
      Fail("PNG policy omitted actual drawing/readback/completion evidence");
    if (mode == 0) {
      baseline = pixels;
      baseline_counters = CounterPayload(report);
      if (pixels[3] != 255 || (pixels[4] != 30 && pixels[4] != 40))
        Fail("PNG policy baseline did not render the real texture");
    } else if (pixels != baseline || CounterPayload(report) != baseline_counters) {
      Fail("PNG policy changed native pixels or dynamic counters");
    }
    const auto png = folder / "driver_textured_triangles_sample_000001.png";
    if (mode == 2) {
      if (std::filesystem::exists(png) || report.find("\"artifact_png\"") != std::string::npos ||
          report.find("@CAPTURE:") != std::string::npos)
        Fail("disabled PNG still created or advertised an artifact");
      for (const auto &entry : std::filesystem::directory_iterator(folder))
        if (entry.path().extension() == ".png")
          Fail("disabled PNG created an unexpected PNG artifact");
    } else {
      png_uint_32 width = 0, height = 0;
      const auto decoded = ReadPngRgba(png, &width, &height);
      if (width != 3 || height != 2 || decoded.size() != pixels.size() ||
          report.find("\"artifact_png\"") == std::string::npos)
        Fail("default/zero PNG policy did not publish a valid artifact");
    }
  }
  SetDisablePng(nullptr);
}

void TestPackedClearReadback() {
  for (const bool bgr : {false, true}) {
    const auto folder = g_test_root / (bgr ? "packed-clear-bgr" : "packed-clear-rgb");
    std::filesystem::create_directory(folder);
    const std::string jsonl = (folder / "model.jsonl").string();
    const std::string stderr_path = (folder / "model.stderr.log").string();
    const std::string outdir = folder.string();
    pvrgpu_systemc_driver_command command{};
    command.version = PVRGPU_SYSTEMC_API_VERSION;
    command.schema = "pvrgpu.driver-command.v1";
    command.producer = "pvrgpu-gallium-driver";
    command.command = "clear_color";
    command.case_name = "generic-packed-clear";
    command.format = bgr ? "PIPE_FORMAT_B10G10R10A2_UNORM"
                         : "PIPE_FORMAT_R10G10B10A2_UNORM";
    command.frame = 1;
    command.width = 3;
    command.height = 2;
    const std::array<float, 4> color{1.0F / 1023, 257.0F / 1023,
                                    769.0F / 1023, 2.0F / 3};
    std::memcpy(command.clear_color_bits, color.data(), sizeof(color));
    pvrgpu_systemc_submit_info submit{};
    submit.version = PVRGPU_SYSTEMC_API_VERSION;
    submit.command = &command;
    submit.jsonl_path = jsonl.c_str();
    submit.stderr_path = stderr_path.c_str();
    submit.outdir = outdir.c_str();
    submit.memory_mode = "direct";
    std::array<char, 512> error{};
    if (pvrgpu_systemc_submit_driver_command(&submit, error.data(), error.size()) != 0)
      Fail("packed clear submission: " + std::string(error.data()));
    std::array<std::uint32_t, 6> words{};
    words.fill(UINT32_C(0xdeadbeef));
    pvrgpu_systemc_readback_info readback{};
    readback.version = PVRGPU_SYSTEMC_API_VERSION;
    readback.width = 3;
    readback.height = 2;
    readback.bytes_per_pixel = 4;
    readback.sample_count = 1;
    readback.layer_count = 1;
    readback.pixels = reinterpret_cast<std::uint8_t *>(words.data());
    readback.pixels_size = sizeof(words);
    if (pvrgpu_systemc_flush_readback(&readback, error.data(), error.size()) != 0 ||
        readback.pixels_written != 1)
      Fail("packed clear readback: " + std::string(error.data()));
    // These exact physical codes cannot survive a round trip through RGBA8.
    const std::uint32_t expected = (bgr ? 769U : 1U) | (257U << 10) |
        ((bgr ? 1U : 769U) << 20) | (2U << 30);
    for (const auto word : words)
      if (word != expected)
        Fail("packed clear lost low bits or channel identity: got=" +
             std::to_string(word) + " expected=" + std::to_string(expected));
    const auto log = ReadText(folder / "model.jsonl");
    if (log.find("\"type\":\"done\"") == std::string::npos ||
        log.find("\"type\":\"error\"") != std::string::npos)
      Fail("packed clear missing successful native model completion");
    png_uint_32 width = 0, height = 0;
    const auto rgba = ReadPngRgba(folder / "driver_clear_color_sample_000001.png", &width, &height);
    if (width != 3 || height != 2 || rgba.size() != 24)
      Fail("packed clear display extent");
    const std::array<std::uint8_t, 4> expected_display{0, 64, 192, 170};
    for (std::size_t pixel = 0; pixel < words.size(); ++pixel)
      for (std::size_t channel = 0; channel < 4; ++channel)
        if (rgba[pixel * 4 + channel] != expected_display[channel])
          Fail("packed clear display must decode physical words without changing raw readback");
  }
}

}  // namespace

int main() {
  SetDisablePng(nullptr);
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
  command.depth_enable = 1;
  command.depth_write = 1;
  command.depth_func = 3;
  command.depth_clear_bits = UINT32_C(0x3f800000);
  command.depth_format = 276;

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

  // Register the verifier before the bridge's first submit registers its
  // deferred flusher; preserve the original atexit ordering contract below.
  TestPngOutput(info);
  TestPackedClearReadback();

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
