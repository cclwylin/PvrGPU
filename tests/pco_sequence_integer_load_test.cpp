// Regression coverage for wide integer attachment LOADs in native PCO
// sequences.  The second draw aliases the first draw's colour attachment, so
// Submitter must read width * height * bytes_per_pixel from DRAM before PBE
// validates and consumes that LOAD.  RG32 and RGBA32 are respectively eight
// and sixteen bytes per pixel; treating either as RGBA8 truncates the payload.

#include "pvrgpu_systemc_api.h"
#include "shader/pco_iss.h"
#include "../model_stub/model_types.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <vector>

namespace {

constexpr std::uint32_t kWidth = 4;
constexpr std::uint32_t kHeight = 4;
constexpr std::uint32_t kFloatOne = UINT32_C(0x3f800000);
constexpr std::uint32_t kNewAttachment = UINT32_MAX;

[[noreturn]] void Fail(const std::string &message) {
  std::fprintf(stderr, "pco-sequence-integer-load-test: %s\n",
               message.c_str());
  std::_Exit(EXIT_FAILURE);
}

std::string ReadText(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return {};
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::vector<std::uint8_t> VertexBytes(
    const std::vector<std::array<float, 4>> &vertices) {
  std::vector<std::uint8_t> bytes(vertices.size() * sizeof(vertices.front()));
  std::memcpy(bytes.data(), vertices.data(), bytes.size());
  return bytes;
}

std::vector<std::uint8_t> FragmentBinary(std::uint32_t channels) {
  std::vector<std::uint8_t> binary =
      pvrgpu::stub::VaryingsOneFragmentPcoBinary();
  if (channels == 2) {
    // FITRP + WDF occupy the first sixteen bytes and each PIXOUT MBYP is
    // eight.  Retaining the first two exports yields PIXOUT mask 0x3; move
    // the instruction-group end bit from the removed fourth export to the
    // retained second one.
    binary.resize(32);
    binary[26] |= UINT8_C(0x80);
  } else if (channels != 4) {
    Fail("unsupported integer channel count in fixture");
  }
  return binary;
}

pvrgpu_systemc_driver_command MakeDraw(
    const char *case_name, const char *format,
    const std::vector<std::uint8_t> &vertices,
    const std::vector<std::uint8_t> &vertex_pco,
    const std::vector<std::uint8_t> &fragment_pco,
    std::uint32_t channels, std::uint32_t source_ordinal) {
  pvrgpu_systemc_driver_command draw{};
  draw.version = PVRGPU_SYSTEMC_API_VERSION;
  draw.schema = "pvrgpu.driver-command.v1";
  draw.producer = "pvrgpu-gallium-driver";
  draw.command = "draw_pco_triangles";
  draw.case_name = case_name;
  draw.format = format;
  draw.frame = 1;
  draw.framebuffer_width = kWidth;
  draw.framebuffer_height = kHeight;
  draw.width = kWidth;
  draw.height = kHeight;
  draw.clear_color_bits[3] = kFloatOne;

  draw.raw_vertex_data = vertices.data();
  draw.raw_vertex_data_size = vertices.size();
  draw.vertex_stride = 4U * sizeof(float);
  draw.vertex_count = static_cast<std::uint32_t>(
      vertices.size() / draw.vertex_stride);
  draw.instance_count = 1;
  draw.primitive_mode = 4;
  draw.render_target_count = 1;
  draw.vertex_attribute_count = 1;
  draw.vertex_attribute_components[0] = 4;

  draw.vertex_pco = vertex_pco.data();
  draw.vertex_pco_size = vertex_pco.size();
  draw.fragment_pco = fragment_pco.data();
  draw.fragment_pco_size = fragment_pco.size();
  draw.vertex_pco_abi.temps = 4;
  draw.vertex_pco_abi.vertex_inputs = 4;
  draw.vertex_pco_abi.vertex_outputs = 8;
  draw.fragment_pco_abi.temps = 4;
  draw.fragment_pco_abi.coefficients = 20;
  draw.position_output_count = 4;
  draw.varying_output_start = 4;
  draw.varying_output_count = 4;
  draw.fragment_position_count = 4;
  draw.fragment_varying_start = 4;
  draw.fragment_varying_count = 16;
  draw.fragment_output_mask[0] = (1U << channels) - 1U;

  const std::array<float, 3> viewport = {
      static_cast<float>(kWidth) * 0.5F,
      static_cast<float>(kHeight) * 0.5F,
      0.5F,
  };
  std::memcpy(draw.viewport_scale_bits, viewport.data(), sizeof(viewport));
  std::memcpy(draw.viewport_translate_bits, viewport.data(), sizeof(viewport));
  draw.half_pixel_center = 1;
  draw.depth_clip_near = 1;
  draw.depth_clip_far = 1;
  draw.sample_mask = UINT32_MAX;
  draw.color_mask = 0x0f;
  draw.dither = 1;
  draw.depth_attachment_source_command_index = kNewAttachment;
  draw.color_attachment_source_command_index = source_ordinal;
  draw.blend_rgb_equation = PVRGPU_SYSTEMC_PCO_BLEND_EQUATION_ADD;
  draw.blend_alpha_equation = PVRGPU_SYSTEMC_PCO_BLEND_EQUATION_ADD;
  draw.blend_source_rgb_factor = PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE;
  draw.blend_destination_rgb_factor = PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ZERO;
  draw.blend_source_alpha_factor = PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE;
  draw.blend_destination_alpha_factor = PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ZERO;
  return draw;
}

struct SequenceResult {
  std::vector<std::uint8_t> pixels;
  std::vector<std::uint8_t> depth;
  std::string counters;
};

SequenceResult SubmitAndFlushSequence(
    const std::filesystem::path &root, const char *case_name,
    const char *format, std::uint32_t channels,
    bool rgba8 = false, bool expect_png = false) {
  const std::vector<std::uint8_t> fullscreen = VertexBytes({
      {-1.0F, -1.0F, 0.0F, 1.0F},
      {1.0F, -1.0F, 0.0F, 1.0F},
      {-1.0F, 1.0F, 0.0F, 1.0F},
      {-1.0F, 1.0F, 0.0F, 1.0F},
      {1.0F, -1.0F, 0.0F, 1.0F},
      {1.0F, 1.0F, 0.0F, 1.0F},
  });
  const std::vector<std::uint8_t> lower_left = VertexBytes({
      {-1.0F, -1.0F, 0.0F, 1.0F},
      {1.0F, -1.0F, 0.0F, 1.0F},
      {-1.0F, 1.0F, 0.0F, 1.0F},
  });
  const std::vector<std::uint8_t> vertex_pco =
      pvrgpu::stub::VaryingsOneVertexPcoBinary();
  const std::vector<std::uint8_t> fragment_pco = FragmentBinary(channels);

  std::array<pvrgpu_systemc_driver_command, 2> draws = {
      MakeDraw(case_name, format, fullscreen, vertex_pco, fragment_pco,
               channels, kNewAttachment),
      MakeDraw(case_name, format, lower_left, vertex_pco, fragment_pco,
               channels, 0),
  };
  if (rgba8) {
    for (unsigned i = 0; i < draws.size(); ++i) {
      draws[i].depth_format = pvrgpu::stub::kDriverPcoDepthFormatZ24X8Unorm;
      draws[i].depth_enable = 1;
      draws[i].depth_write = 1;
      draws[i].depth_func = 3; // LEQUAL: both actual draws pass at z = 0.5.
      draws[i].depth_clear_bits = kFloatOne;
      draws[i].depth_attachment_source_command_index = i == 0 ? kNewAttachment : 0;
    }
  }

  pvrgpu_systemc_driver_command sequence{};
  sequence.version = PVRGPU_SYSTEMC_API_VERSION;
  sequence.schema = "pvrgpu.driver-command.v1";
  sequence.producer = "pvrgpu-gallium-driver";
  sequence.command = "draw_pco_sequence";
  sequence.case_name = case_name;
  sequence.format = format;
  sequence.frame = 1;
  sequence.framebuffer_width = kWidth;
  sequence.framebuffer_height = kHeight;
  sequence.width = kWidth;
  sequence.height = kHeight;
  sequence.draw_count = draws.size();
  sequence.ia_vertices = draws[0].vertex_count + draws[1].vertex_count;
  sequence.ia_primitives = 3;
  sequence.clip_invocations = sequence.ia_vertices;
  sequence.pco_sequence_command_count = draws.size();
  sequence.pco_sequence_commands = draws.data();

  const std::filesystem::path run_root = root / case_name;
  std::filesystem::create_directories(run_root / "out");
  const std::string jsonl = (run_root / "model.jsonl").string();
  const std::string stderr_log = (run_root / "model.stderr.log").string();
  const std::string outdir = (run_root / "out").string();
  pvrgpu_systemc_submit_info info{};
  info.version = PVRGPU_SYSTEMC_API_VERSION;
  info.command = &sequence;
  info.jsonl_path = jsonl.c_str();
  info.stderr_path = stderr_log.c_str();
  info.outdir = outdir.c_str();
  info.memory_mode = "direct";

  std::array<char, 512> error{};
  if (pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size()) !=
      0) {
    Fail(std::string(case_name) + " submit failed: " + error.data());
  }

  const std::uint32_t bytes_per_pixel = rgba8 ? 4U : channels * sizeof(std::uint32_t);
  std::vector<std::uint8_t> pixels(kWidth * kHeight * bytes_per_pixel,
                                   UINT8_C(0xa5));
  pvrgpu_systemc_readback_info readback{};
  readback.version = PVRGPU_SYSTEMC_API_VERSION;
  readback.width = kWidth;
  readback.height = kHeight;
  readback.bytes_per_pixel = bytes_per_pixel;
  readback.pixels = pixels.data();
  readback.pixels_size = pixels.size();
  if (pvrgpu_systemc_flush_readback(&readback, error.data(), error.size()) !=
      0) {
    Fail(std::string(case_name) + " flush failed: " + error.data() +
         "\n" + ReadText(stderr_log));
  }
  if (readback.pixels_written != 1)
    Fail(std::string(case_name) + " published no pixels");

  // The first draw covers the whole target and writes z=0,w=1.  The second
  // covers only one triangle; therefore every untouched pixel proves that the
  // complete wide LOAD, not a fresh clear, survived the alias.
  for (std::size_t pixel = 0; pixel < kWidth * kHeight; ++pixel) {
    std::array<std::uint32_t, 4> words{};
    std::memcpy(words.data(), pixels.data() + pixel * bytes_per_pixel,
                bytes_per_pixel);
    const std::size_t x = pixel % kWidth;
    const std::size_t y = pixel / kWidth;
    const float expected_x =
        static_cast<float>(2 * static_cast<int>(x) + 1 -
                           static_cast<int>(kWidth)) /
        static_cast<float>(kWidth);
    const float expected_y =
        static_cast<float>(2 * static_cast<int>(y) + 1 -
                           static_cast<int>(kHeight)) /
        static_cast<float>(kHeight);
    if (rgba8) {
      // The real varyings shader emits x,y,z,w. These sample centers quantize
      // to exact UNORM8 codes without an external image or production packer.
      const std::array<std::uint8_t, 4> codes{0, 0, 64, 191};
      if (pixels[pixel * 4] != codes[x] ||
          pixels[pixel * 4 + 1] != codes[y] ||
          pixels[pixel * 4 + 2] != 0 || pixels[pixel * 4 + 3] != 255)
        Fail(std::string(case_name) + " RGBA8 sequence pixel mismatch");
      continue;
    }
    if (words[0] != FloatBits(expected_x) ||
        words[1] != FloatBits(expected_y)) {
      Fail(std::string(case_name) +
           " did not preserve an untouched aliased texel");
    }
    if (channels == 4 && (words[2] != 0 || words[3] != kFloatOne)) {
      Fail(std::string(case_name) +
           " returned a truncated/corrupt RGBA32 texel");
    }
  }

  const std::string report = ReadText(jsonl);
  if (report.find("\"driver_command_sequence_length\":2") ==
          std::string::npos ||
      report.find("\"type\":\"done\"") == std::string::npos) {
    Fail(std::string(case_name) + " did not complete a two-draw sequence");
  }
  const auto counters_begin = report.find("\"counters\":{");
  const auto counters_end = report.find('\n', counters_begin);
  if (counters_begin == std::string::npos || counters_end == std::string::npos)
    Fail("sequence did not publish complete counters");
  SequenceResult result{pixels, {}, report.substr(counters_begin, counters_end - counters_begin)};
  if (rgba8) {
    result.depth.assign(kWidth * kHeight * 4, UINT8_C(0xa5));
    readback.attachment = UINT32_MAX;
    readback.depth_format = pvrgpu::stub::kDriverPcoDepthFormatZ24X8Unorm;
    readback.pixels = result.depth.data();
    readback.pixels_size = result.depth.size();
    readback.pixels_written = 0;
    if (pvrgpu_systemc_flush_readback(&readback, error.data(), error.size()) != 0 ||
        readback.pixels_written != 1 ||
        result.depth == std::vector<std::uint8_t>(result.depth.size(), UINT8_C(0xa5)))
      Fail("RGBA8 sequence did not publish actual depth readback: " + std::string(error.data()));
    if (report.find("\"ps_invocations\":0,") != std::string::npos ||
        report.find("\"type\":\"error\"") != std::string::npos)
      Fail("RGBA8 sequence omitted native fragment work or reported an error");
    const auto png = run_root / "out" / "driver_pco_triangles_sample_000001.png";
    if (std::filesystem::exists(png) != expect_png ||
        (report.find("\"artifact_png\"") != std::string::npos) != expect_png)
      Fail("sequence PNG output did not follow the requested policy");
    if (expect_png && std::filesystem::file_size(png) < 8)
      Fail("default sequence PNG is empty");
  }
  return result;
}

void SetDisablePng(const char *value) {
#if defined(_WIN32)
  if (_putenv_s("PVRGPU_SYSTEMC_DISABLE_PNG", value ? value : "") != 0)
#else
  if ((value ? setenv("PVRGPU_SYSTEMC_DISABLE_PNG", value, 1)
             : unsetenv("PVRGPU_SYSTEMC_DISABLE_PNG")) != 0)
#endif
    Fail("cannot configure sequence PNG output");
}

}  // namespace

int main() {
  const auto nonce =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      ("pvrgpu-pco-sequence-integer-load-test-" + std::to_string(nonce));
  std::filesystem::create_directories(root);

  SetDisablePng(nullptr);
  SubmitAndFlushSequence(root, "sequence-rg32ui-load",
                                    "PIPE_FORMAT_R32G32_UINT", 2);
  SubmitAndFlushSequence(root, "sequence-rgba32ui-load",
                                    "PIPE_FORMAT_R32G32B32A32_UINT", 4);
  const auto with_png = SubmitAndFlushSequence(root, "sequence-rgba8-png",
      "PIPE_FORMAT_R8G8B8A8_UNORM", 4, true, true);
  SetDisablePng("1");
  const auto no_png = SubmitAndFlushSequence(root, "sequence-rgba8-no-png",
      "PIPE_FORMAT_R8G8B8A8_UNORM", 4, true, false);
  SetDisablePng(nullptr);
  if (with_png.pixels != no_png.pixels || with_png.depth != no_png.depth ||
      with_png.counters != no_png.counters)
    Fail("PNG suppression changed actual sequence color/depth/counters");

  std::error_code error;
  std::filesystem::remove_all(root, error);
  std::puts("pco-sequence-integer-load-test: PASS");
  return EXIT_SUCCESS;
}
