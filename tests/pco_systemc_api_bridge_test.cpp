#include "pvrgpu_systemc_api.h"
#include "common/functional_types.h"
#include "shader/pco_iss.h"
#include "../model_stub/model_types.h"

#include <png.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

std::filesystem::path g_test_root;

[[noreturn]] void Fail(const std::string &message) {
  std::fprintf(stderr, "pco-systemc-api-bridge-test: %s\n", message.c_str());
  std::_Exit(EXIT_FAILURE);
}

std::string ReadText(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return {};
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

std::uint64_t Fnv1a64(const std::vector<std::uint8_t> &bytes) {
  std::uint64_t hash = UINT64_C(14695981039346656037);
  for (const std::uint8_t byte : bytes) {
    hash ^= byte;
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

std::vector<std::uint8_t> ReadRgbaPng(const std::filesystem::path &path,
                                      std::uint32_t *width,
                                      std::uint32_t *height) {
  png_image image{};
  image.version = PNG_IMAGE_VERSION;
  const std::string path_text = path.string();
  if (!png_image_begin_read_from_file(&image, path_text.c_str()))
    Fail("cannot read conditionals PNG header: " + std::string(image.message));
  image.format = PNG_FORMAT_RGBA;
  std::vector<std::uint8_t> rgba(PNG_IMAGE_SIZE(image));
  if (!png_image_finish_read(&image, nullptr, rgba.data(), 0, nullptr)) {
    const std::string message = image.message;
    png_image_free(&image);
    Fail("cannot decode conditionals PNG: " + message);
  }
  *width = image.width;
  *height = image.height;
  png_image_free(&image);
  return rgba;
}

void VerifyDeferredModelAtExit() {
  const std::string jsonl = ReadText(g_test_root / "model.jsonl");
  if (jsonl.find("\"type\":\"done\"") == std::string::npos ||
      jsonl.find("\"driver_command\":\"draw_pco_triangles\"") ==
          std::string::npos ||
      jsonl.find("\"ia_vertices\":6144") == std::string::npos ||
      jsonl.find("\"ia_primitives\":2048") == std::string::npos ||
      jsonl.find("\"vs_invocations\":6144") == std::string::npos ||
      jsonl.find("\"c_invocations\":2048") == std::string::npos ||
      jsonl.find("\"c_primitives\":2048") == std::string::npos ||
      jsonl.find("\"ps_invocations\":2006") == std::string::npos ||
      jsonl.find("\"setup_triangles\":2048") == std::string::npos) {
    Fail("deferred PCO API execution did not preserve copied payloads");
  }
  const std::filesystem::path png =
      g_test_root / "out" / "driver_pco_triangles_sample_000001.png";
  std::error_code error;
  if (!std::filesystem::is_regular_file(png, error) || error ||
      std::filesystem::file_size(png, error) < 8 || error) {
    Fail("deferred PCO API execution did not produce a PNG");
  }
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  const std::vector<std::uint8_t> rgba = ReadRgbaPng(png, &width, &height);
  if (width != 80 || height != 60 || rgba.size() != 80U * 60U * 4U ||
      Fnv1a64(rgba) != UINT64_C(0xf11e741976cf66c4)) {
    Fail("conditionals SystemC PNG differs from the decoded Golden RGBA");
  }
  std::filesystem::remove_all(g_test_root, error);
  std::puts("pco-systemc-api-bridge-test: PASS");
}

void StoreFloat(std::vector<std::uint8_t> *bytes, std::size_t offset,
                float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  std::memcpy(bytes->data() + offset, &bits, sizeof(bits));
}

void VerifyIdeasTopologyExpansion() {
  using namespace pvrgpu::stub;
  constexpr std::uint32_t kStride = 4U * sizeof(float);
  constexpr std::uint32_t kVertexCount = 6;
  const float position[kVertexCount][3] = {
      {-1.0F, -1.0F, 0.0F},
      {0.0F, -1.0F, 0.0F},
      {-0.5F, 0.0F, 0.0F},
      {-0.5F, 0.0F, 0.0F},
      {0.5F, 0.0F, 0.0F},
      {1.0F, 1.0F, 0.0F},
  };
  DriverCommand command;
  command.test_case = "ideas";
  command.vertex_stride = kStride;
  command.vertex_count = kVertexCount;
  command.primitive_mode = 5;
  command.raw_vertex_data.resize(kVertexCount * kStride);
  for (std::uint32_t vertex = 0; vertex < kVertexCount; ++vertex) {
    for (std::uint32_t component = 0; component < 3; ++component) {
      StoreFloat(&command.raw_vertex_data,
                 vertex * kStride + component * sizeof(float),
                 position[vertex][component]);
    }
    // The duplicate-position records deliberately retain different payload
    // bits so elimination is based on raster position, not whole-record
    // equality.
    StoreFloat(&command.raw_vertex_data,
               vertex * kStride + 3U * sizeof(float),
               static_cast<float>(vertex + 1U));
  }

  const DriverPcoTopologyExpansion expansion =
      ExpandDriverPcoTopology(command);
  if (expansion.input_primitives != 4 ||
      expansion.emitted_primitives != 4 ||
      expansion.duplicate_position_primitives != 2 ||
      expansion.vertices.size() != 12U * kStride) {
    Fail("Ideas strip degenerate accounting changed");
  }
  static constexpr std::uint32_t kExpectedVertices[12] = {
      0, 1, 2, 2, 1, 3, 2, 3, 4, 4, 3, 5};
  for (std::size_t occurrence = 0; occurrence < 12; ++occurrence) {
    const std::size_t source = kExpectedVertices[occurrence] * kStride;
    const std::size_t destination = occurrence * kStride;
    if (!std::equal(command.raw_vertex_data.begin() + source,
                    command.raw_vertex_data.begin() + source + kStride,
                    expansion.vertices.begin() + destination)) {
      Fail("Ideas strip expansion skipped or reordered a live PCO vertex");
    }
  }
}

void VerifyGenericFloat2StripExpansion() {
  using namespace pvrgpu::stub;
  DriverCommand command;
  command.test_case = "shadow.mask";
  command.vertex_stride = 2U * sizeof(float);
  command.vertex_count = 4;
  command.primitive_mode = 5;
  command.raw_vertex_data.resize(command.vertex_stride * command.vertex_count);
  const float positions[4][2] = {
      {-1.0F, -1.0F}, {1.0F, -1.0F}, {-1.0F, 1.0F}, {1.0F, 1.0F}};
  std::memcpy(command.raw_vertex_data.data(), positions, sizeof(positions));
  const DriverPcoTopologyExpansion expansion =
      ExpandDriverPcoTopology(command);
  if (expansion.input_primitives != 2 ||
      expansion.emitted_primitives != 2 ||
      expansion.duplicate_position_primitives != 0 ||
      expansion.vertices.size() != 6U * command.vertex_stride) {
    Fail("generic float2 strip expansion inferred an Ideas layout");
  }
}

void VerifyDepthAttachmentFormats() {
  using namespace pvrgpu::stub;
  if (DepthAttachmentBytesPerPixel(kDriverPcoDepthFormatZ16Unorm) != 2 ||
      DepthAttachmentBytesPerPixel(kDriverPcoDepthFormatZ24X8Unorm) != 4 ||
      DepthAttachmentBytesPerPixel(kDriverPcoDepthFormatZ32Unorm) != 4 ||
      EncodeDepthAttachmentUnorm(0.5F,
                                 kDriverPcoDepthFormatZ16Unorm) != 0x8000U ||
      EncodeDepthAttachmentUnorm(0.5F,
                                 kDriverPcoDepthFormatZ24X8Unorm) !=
          0x800000U ||
      EncodeDepthAttachmentUnorm(0.5F,
                                 kDriverPcoDepthFormatZ32Unorm) !=
          0x80000000U ||
      EncodeDepthAttachmentUnorm(1.0F,
                                 kDriverPcoDepthFormatZ16Unorm) != 0xffffU ||
      EncodeDepthAttachmentUnorm(1.0F,
                                 kDriverPcoDepthFormatZ24X8Unorm) !=
          0xffffffU ||
      EncodeDepthAttachmentUnorm(1.0F,
                                 kDriverPcoDepthFormatZ32Unorm) != UINT32_MAX) {
    Fail("native depth UNORM format conversion changed");
  }
  const float z16 = DecodeDepthAttachmentUnorm(
      0x1234U, kDriverPcoDepthFormatZ16Unorm);
  if (EncodeDepthAttachmentUnorm(z16,
                                 kDriverPcoDepthFormatZ16Unorm) != 0x1234U) {
    Fail("Z16 attachment LOAD round-trip changed");
  }
  bool padding_rejected = false;
  try {
    (void)DecodeDepthAttachmentUnorm(
        0x01000000U, kDriverPcoDepthFormatZ24X8Unorm);
  } catch (const std::runtime_error &) {
    padding_rejected = true;
  }
  if (!padding_rejected)
    Fail("Z24X8 nonzero padding was accepted");
  const auto verify_untouched_load = [](std::uint32_t format,
                                        const auto &values) {
    std::vector<std::uint32_t> encoded(values.begin(), values.end());
    const std::vector<std::uint8_t> bytes =
        EncodeDepthAttachmentUnormBytes(encoded, format);
    if (EncodeDepthAttachmentUnormBytes(
            DecodeDepthAttachmentUnormBytes(bytes, format), format) != bytes) {
      Fail("untouched depth attachment LOAD pixel changed on commit");
    }
  };
  verify_untouched_load(kDriverPcoDepthFormatZ16Unorm,
                        std::array<std::uint32_t, 3>{0U, 0x1234U, 0xffffU});
  verify_untouched_load(
      kDriverPcoDepthFormatZ24X8Unorm,
      std::array<std::uint32_t, 3>{0U, 0x654321U, 0xffffffU});
  verify_untouched_load(
      kDriverPcoDepthFormatZ32Unorm,
      std::array<std::uint32_t, 3>{0U, 0x12345678U, UINT32_MAX});
}

} // namespace

int main() {
  using namespace pvrgpu::stub;
  static_assert(PVRGPU_SYSTEMC_API_VERSION == 8U,
                "native sequence bridge test requires API-v8");
  static_assert(PVRGPU_SYSTEMC_MAX_TEXTURE_MIP_LEVELS == 15U);
  static_assert(kDriverPcoMaximumTextureMipLevels == 15U);
  static_assert(kMaximumTextureMipLevels == 15U);
  static_assert(
      sizeof(pvrgpu_systemc_pco_sequence_texture::mip) /
              sizeof(pvrgpu_systemc_pco_texture_mip) ==
          PVRGPU_SYSTEMC_MAX_TEXTURE_MIP_LEVELS,
      "SystemC API texture ABI does not expose all 15 mip slots");
  static_assert(
      sizeof(void *) != 8U ||
          sizeof(pvrgpu_systemc_pco_sequence_texture) == 336U,
      "64-bit SystemC API-v8 sequence texture ABI size changed");
  static_assert(
      std::tuple_size<decltype(DriverPcoSampledTexture{}.mip)>::value ==
          kDriverPcoMaximumTextureMipLevels,
      "owned model texture ABI does not expose all 15 mip slots");
  static_assert(
      sizeof(TextureResource::mip) / sizeof(TextureMipLevel) ==
          kMaximumTextureMipLevels,
      "functional texture payload does not expose all 15 mip slots");
  VerifyIdeasTopologyExpansion();
  VerifyGenericFloat2StripExpansion();
  VerifyDepthAttachmentFormats();
  const auto nonce = std::chrono::high_resolution_clock::now()
                         .time_since_epoch()
                         .count();
  g_test_root = std::filesystem::temp_directory_path() /
                ("pvrgpu-pco-systemc-api-test-" + std::to_string(nonce));
  std::filesystem::create_directories(g_test_root / "out");

  constexpr std::size_t kVertexCount = 6144;
  constexpr std::size_t kStride = 12;
  std::vector<std::uint8_t> vertices(kVertexCount * kStride);
  // Exact GLBench conditionals 32x32 cell stream: two non-indexed float3
  // triangles per cell.  This exercises real clipping, BACK/CW conversion,
  // 2006 fragment invocations and fragment-coordinate PCO execution.
  std::size_t vertex = 0;
  for (unsigned i = 0; i < 32; ++i) {
    for (unsigned j = 0; j < 32; ++j) {
      const double side = 0.136875;
      const double pitch = 0.156875;
      const float ax = static_cast<float>(-2.5 + i * pitch);
      const float ay = static_cast<float>(2.5 - j * pitch);
      const float bx = ax;
      const float by = static_cast<float>(ay - side);
      const float cx = static_cast<float>(ax + side);
      const float cy = ay;
      const float dx = cx;
      const float dy = by;
      const float cell[6][3] = {
          {ax, ay, 0.0F}, {bx, by, 0.0F}, {cx, cy, 0.0F},
          {bx, by, 0.0F}, {dx, dy, 0.0F}, {cx, cy, 0.0F},
      };
      for (const auto &position : cell) {
        for (std::size_t component = 0; component < 3; ++component)
          StoreFloat(&vertices, vertex * kStride + component * 4,
                     position[component]);
        ++vertex;
      }
    }
  }
  if (vertex != kVertexCount)
    Fail("conditionals vertex fixture count changed");
  std::vector<std::uint8_t> vertex_pco = ConditionalsVertexPcoBinary();
  std::vector<std::uint8_t> fragment_pco = ConditionalsFragmentPcoBinary();
  std::vector<std::uint32_t> vertex_shared = {
      UINT32_C(0x3fa646e0), 0, 0, 0,
      0, UINT32_C(0x3fddb3d6), 0, 0,
      0, 0, UINT32_C(0xbf804010), UINT32_C(0xbf800000),
      0, 0, UINT32_C(0x40408020), UINT32_C(0x40a00000),
  };
  std::vector<std::uint32_t> fragment_shared = {
      UINT32_C(0x3f800000), 0, UINT32_C(0xbf800000),
      UINT32_C(0x42700000),
  };

  pvrgpu_systemc_driver_command command{};
  command.version = PVRGPU_SYSTEMC_API_VERSION;
  command.schema = "pvrgpu.driver-command.v1";
  command.producer = "pvrgpu-gallium-driver";
  command.command = "draw_pco_triangles";
  command.case_name = "glmark2.conditionals";
  command.format = "PIPE_FORMAT_R8G8B8A8_UNORM";
  command.frame = 1;
  command.framebuffer_width = 80;
  command.framebuffer_height = 60;
  command.width = 80;
  command.height = 60;
  command.clear_color_bits[3] = UINT32_C(0x3f800000);
  command.raw_vertex_data = vertices.data();
  command.raw_vertex_data_size = vertices.size();
  command.vertex_stride = kStride;
  command.vertex_count = kVertexCount;
  command.instance_count = 1;
  command.primitive_mode = 4;
  command.vertex_pco = vertex_pco.data();
  command.vertex_pco_size = vertex_pco.size();
  command.fragment_pco = fragment_pco.data();
  command.fragment_pco_size = fragment_pco.size();
  command.vertex_shared = vertex_shared.data();
  command.vertex_shared_count = vertex_shared.size();
  command.fragment_shared = fragment_shared.data();
  command.fragment_shared_count = fragment_shared.size();
  command.vertex_pco_abi.temps = 10;
  command.vertex_pco_abi.vertex_inputs = 4;
  command.vertex_pco_abi.vertex_outputs = 4;
  command.vertex_pco_abi.shareds = 16;
  command.vertex_pco_abi.push_constant_count = 16;
  command.fragment_pco_abi.temps = 4;
  command.fragment_pco_abi.shareds = 4;
  command.fragment_pco_abi.push_constant_count = 4;
  command.position_output_count = 4;
  // gl_FragCoord is a hardware/window input, not a linked VS varying.
  command.fragment_position_count = 0;
  command.viewport_scale_bits[0] = UINT32_C(0x42200000);
  command.viewport_scale_bits[1] = UINT32_C(0x41f00000);
  command.viewport_scale_bits[2] = UINT32_C(0x3f000000);
  std::copy(std::begin(command.viewport_scale_bits),
            std::end(command.viewport_scale_bits),
            std::begin(command.viewport_translate_bits));
  command.cull_face = 2;
  command.half_pixel_center = 1;
  command.depth_clip_near = 1;
  command.depth_clip_far = 1;
  command.sample_mask = UINT32_MAX;
  command.color_mask = 0x0f;
  command.dither = 1;
  command.depth_enable = 1;
  command.depth_write = 1;
  command.depth_func = 3;
  command.depth_clear_bits = UINT32_C(0x3f800000);
  command.depth_format = 1;
  const auto set_pco_resolution = [&](std::uint32_t width,
                                      std::uint32_t height) {
    command.framebuffer_width = width;
    command.framebuffer_height = height;
    command.width = width;
    command.height = height;
    const std::array<float, 3> viewport = {
        static_cast<float>(width) * 0.5F,
        static_cast<float>(height) * 0.5F,
        0.5F,
    };
    static_assert(sizeof(viewport) == sizeof(command.viewport_scale_bits));
    std::memcpy(command.viewport_scale_bits, viewport.data(),
                sizeof(command.viewport_scale_bits));
    std::memcpy(command.viewport_translate_bits, viewport.data(),
                sizeof(command.viewport_translate_bits));
  };

  const std::string jsonl = (g_test_root / "model.jsonl").string();
  const std::string stderr_log = (g_test_root / "model.stderr.log").string();
  const std::string outdir = (g_test_root / "out").string();
  pvrgpu_systemc_submit_info info{};
  info.version = PVRGPU_SYSTEMC_API_VERSION;
  info.command = &command;
  info.jsonl_path = jsonl.c_str();
  info.stderr_path = stderr_log.c_str();
  info.outdir = outdir.c_str();
  info.memory_mode = "direct";

  std::array<char, 256> error{};
  info.version = PVRGPU_SYSTEMC_API_VERSION - 1U;
  if (pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size()) ==
          0 ||
      std::string(error.data()).find("submit version") == std::string::npos) {
    Fail("previous-version submit envelope was not rejected");
  }
  info.version = PVRGPU_SYSTEMC_API_VERSION;
  command.version = PVRGPU_SYSTEMC_API_VERSION - 1U;
  error.fill(0);
  if (pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size()) ==
          0 ||
      std::string(error.data()).find("command version") == std::string::npos) {
    Fail("previous-version command was not rejected before the API-v8 tail");
  }
  command.version = PVRGPU_SYSTEMC_API_VERSION;
  error.fill(0);

  set_pco_resolution(640, 480);
  if (pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size()) ==
          0 ||
      std::string(error.data()).find("80x60 or 800x600") ==
          std::string::npos) {
    Fail("unsupported 640x480 PCO API resolution was not rejected");
  }
  set_pco_resolution(800, 600);
  command.width = 80;
  error.fill(0);
  if (pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size()) ==
          0 ||
      std::string(error.data()).find("framebuffer-sized") ==
          std::string::npos) {
    Fail("non-framebuffer-sized PCO API viewport extent was not rejected");
  }
  set_pco_resolution(800, 600);
  command.viewport_translate_bits[0] = UINT32_C(0x42200000);
  error.fill(0);
  if (pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size()) ==
          0 ||
      std::string(error.data()).find("ABI/viewport metadata") ==
          std::string::npos) {
    Fail("80x60 viewport state was accepted for an 800x600 PCO API command");
  }
  set_pco_resolution(800, 600);
  if (std::atexit(VerifyDeferredModelAtExit) != 0)
    Fail("cannot register deferred verifier");
  error.fill(0);
  if (pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size()) !=
      0) {
    Fail("valid 800x600 generic PCO API submit was rejected: " +
         std::string(error.data()));
  }
  set_pco_resolution(80, 60);

  using AbiMember =
      std::uint32_t pvrgpu_systemc_pco_stage_abi::*;
  static constexpr std::array<AbiMember, 8> kAbiMembers = {
      &pvrgpu_systemc_pco_stage_abi::temps,
      &pvrgpu_systemc_pco_stage_abi::vertex_inputs,
      &pvrgpu_systemc_pco_stage_abi::vertex_outputs,
      &pvrgpu_systemc_pco_stage_abi::coefficients,
      &pvrgpu_systemc_pco_stage_abi::shareds,
      &pvrgpu_systemc_pco_stage_abi::push_constant_start,
      &pvrgpu_systemc_pco_stage_abi::push_constant_count,
      &pvrgpu_systemc_pco_stage_abi::entry_offset,
  };
  const auto expect_abi_rejected = [&](pvrgpu_systemc_pco_stage_abi &abi,
                                       const char *stage) {
    for (const AbiMember member : kAbiMembers) {
      ++(abi.*member);
      error.fill(0);
      if (pvrgpu_systemc_submit_driver_command(&info, error.data(),
                                               error.size()) == 0 ||
          std::string(error.data()).find("ABI/viewport metadata") ==
              std::string::npos) {
        Fail(std::string("wrong ") + stage +
             " PCO ABI field was not rejected");
      }
      --(abi.*member);
    }
  };

  // The legacy single-draw contract keeps all existing profiles texture-free and accepts sampled
  // bytes only with the exact current GLMark2 texture transport profile.
  std::uint8_t stray_texture_byte = 0;
  command.sampled_texture_count = 1;
  command.sampled_texture_bytes = &stray_texture_byte;
  command.sampled_texture_bytes_size = 1;
  error.fill(0);
  if (pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size()) ==
          0 ||
      std::string(error.data()).find("untextured profile") ==
          std::string::npos) {
    Fail("sampled texture state on conditionals was not rejected");
  }
  command.sampled_texture_count = 0;
  command.sampled_texture_bytes = nullptr;
  command.sampled_texture_bytes_size = 0;

  const pvrgpu_systemc_driver_command conditionals_command = command;
  const auto make_sequence_draw = [&]() {
    pvrgpu_systemc_driver_command draw = conditionals_command;
    draw.case_name = "bridge.sequence.boundary.draw";
    draw.vertex_pco_abi.vertex_outputs = 5;
    draw.fragment_pco_abi.coefficients = 8;
    draw.position_output_start = 0;
    draw.position_output_count = 4;
    draw.varying_output_start = 4;
    draw.varying_output_count = 1;
    draw.fragment_position_start = 0;
    draw.fragment_position_count = 4;
    draw.fragment_varying_start = 4;
    draw.fragment_varying_count = 4;
    draw.depth_format = kDriverPcoDepthFormatZ24X8Unorm;
    draw.color_attachment_source_command_index =
        PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR;
    draw.depth_attachment_source_command_index =
        PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR;
    draw.blend_rgb_equation = PVRGPU_SYSTEMC_PCO_BLEND_EQUATION_ADD;
    draw.blend_alpha_equation = PVRGPU_SYSTEMC_PCO_BLEND_EQUATION_ADD;
    draw.blend_source_rgb_factor = PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE;
    draw.blend_destination_rgb_factor = PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ZERO;
    draw.blend_source_alpha_factor = PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE;
    draw.blend_destination_alpha_factor = PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ZERO;
    return draw;
  };
  const auto expect_sequence_rejected =
      [&](std::array<pvrgpu_systemc_driver_command, 2> draws,
          const char *expected_error, const char *description) {
        pvrgpu_systemc_driver_command sequence{};
        sequence.version = PVRGPU_SYSTEMC_API_VERSION;
        sequence.schema = "pvrgpu.driver-command.v1";
        sequence.producer = "pvrgpu-gallium-driver";
        sequence.command = "draw_pco_sequence";
        sequence.case_name = "bridge.sequence.boundary";
        sequence.format = "PIPE_FORMAT_R8G8B8A8_UNORM";
        sequence.framebuffer_width = 80;
        sequence.framebuffer_height = 60;
        sequence.width = 80;
        sequence.height = 60;
        sequence.pco_sequence_command_count = draws.size();
        sequence.pco_sequence_commands = draws.data();
        info.command = &sequence;
        error.fill(0);
        if (pvrgpu_systemc_submit_driver_command(&info, error.data(),
                                                 error.size()) == 0 ||
            std::string(error.data()).find(expected_error) ==
                std::string::npos) {
          Fail(std::string(description) + " was not rejected: " +
               error.data());
        }
        info.command = &command;
      };

  {
    std::array<pvrgpu_systemc_driver_command, 2> draws = {
        make_sequence_draw(), make_sequence_draw()};
    draws[0].color_attachment_source_command_index = 0;
    expect_sequence_rejected(draws, "raster/resource state",
                             "self-referencing color attachment alias");
  }
  {
    std::array<pvrgpu_systemc_driver_command, 2> draws = {
        make_sequence_draw(), make_sequence_draw()};
    draws[0].depth_attachment_source_command_index = 1;
    expect_sequence_rejected(draws, "raster/resource state",
                             "future depth attachment alias");
  }
  {
    std::array<pvrgpu_systemc_driver_command, 2> draws = {
        make_sequence_draw(), make_sequence_draw()};
    draws[1].color_attachment_source_command_index = 0;
    draws[1].framebuffer_width = 64;
    draws[1].width = 64;
    expect_sequence_rejected(draws, "alias format/extent mismatch",
                             "different-size color attachment alias");
  }
  {
    std::array<pvrgpu_systemc_driver_command, 2> draws = {
        make_sequence_draw(), make_sequence_draw()};
    draws[0].depth_format = kDriverPcoDepthFormatZ32Unorm;
    draws[1].depth_attachment_source_command_index = 0;
    expect_sequence_rejected(draws, "alias format/extent mismatch",
                             "different-format depth attachment alias");
  }
  {
    std::array<pvrgpu_systemc_driver_command, 2> draws = {
        make_sequence_draw(), make_sequence_draw()};
    draws[0].blend_source_rgb_factor =
        PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ZERO;
    expect_sequence_rejected(draws, "raster/resource state",
                             "noncanonical disabled blend state");
  }

  std::vector<std::uint8_t> texture_vertices(36U * 32U, 0);
  std::vector<std::uint8_t> texture_bytes(512U * 512U * 4U, UINT8_C(0x5a));
  std::vector<std::uint32_t> texture_vertex_shared(32U, 0);
  std::vector<std::uint32_t> texture_fragment_shared(20U, 0);
  command.raw_vertex_data = texture_vertices.data();
  command.raw_vertex_data_size = texture_vertices.size();
  command.vertex_stride = 32;
  command.vertex_count = 36;
  command.vertex_shared = texture_vertex_shared.data();
  command.vertex_shared_count = texture_vertex_shared.size();
  command.fragment_shared = texture_fragment_shared.data();
  command.fragment_shared_count = texture_fragment_shared.size();
  command.vertex_pco_abi = {};
  command.vertex_pco_abi.temps = 10;
  command.vertex_pco_abi.vertex_inputs = 12;
  command.vertex_pco_abi.vertex_outputs = 7;
  command.vertex_pco_abi.shareds = 32;
  command.vertex_pco_abi.push_constant_count = 32;
  command.fragment_pco_abi = {};
  command.fragment_pco_abi.temps = 4;
  command.fragment_pco_abi.coefficients = 16;
  command.fragment_pco_abi.shareds = 20;
  command.position_output_start = 0;
  command.position_output_count = 4;
  command.fragment_position_start = 0;
  command.fragment_position_count = 4;
  command.varying_output_start = 4;
  command.varying_output_count = 3;
  command.fragment_varying_start = 4;
  command.fragment_varying_count = 12;
  command.sampled_texture_count = 1;
  command.sampled_texture_bytes = texture_bytes.data();
  command.sampled_texture_bytes_size = texture_bytes.size();
  command.sampled_texture_width = 512;
  command.sampled_texture_height = 512;
  command.sampled_texture_row_pitch = 2048;
  command.sampled_texture_format = "PIPE_FORMAT_R8G8B8X8_UNORM";
  command.sampled_texture_mip_count = 1;

  command.depth_func = 0;
  error.fill(0);
  if (pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size()) ==
          0 ||
      std::string(error.data()).find("raster/depth metadata") ==
          std::string::npos) {
    Fail("texture PCO ABI did not reach the raster validation gate");
  }
  command.depth_func = 3;
  command.sampled_texture_row_pitch = 2044;
  error.fill(0);
  if (pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size()) ==
          0 ||
      std::string(error.data()).find("sampled texture payload") ==
          std::string::npos) {
    Fail("non-tight texture row pitch was not rejected");
  }
  command = conditionals_command;

  // The public register ABI accepts the complete modeled TEMP bank, rather
  // than a value hard-coded for today's Ideas lighting shader (TEMP36).
  // Use a non-root Ideas command so the subsequent conditionals submit can
  // replace it under the ordinary deferred last-command-wins contract.
  static_assert(kPcoTemporaryCount == 64,
                "bridge boundary test tracks the modeled TEMP bank");
  std::vector<std::uint8_t> ideas_vertices(12U * 16U, 0);
  std::vector<std::uint32_t> ideas_vertex_shared(32U, 0);
  std::vector<std::uint32_t> ideas_fragment_shared(4U, 0);
  command.case_name = "ideas.temp-boundary";
  command.raw_vertex_data = ideas_vertices.data();
  command.raw_vertex_data_size = ideas_vertices.size();
  command.vertex_stride = 16;
  command.vertex_count = 12;
  command.primitive_mode = 6;
  command.vertex_shared = ideas_vertex_shared.data();
  command.vertex_shared_count = ideas_vertex_shared.size();
  command.fragment_shared = ideas_fragment_shared.data();
  command.fragment_shared_count = ideas_fragment_shared.size();
  command.vertex_pco_abi = {};
  command.vertex_pco_abi.temps = kPcoTemporaryCount;
  command.vertex_pco_abi.vertex_inputs = 4;
  command.vertex_pco_abi.vertex_outputs = 4;
  command.vertex_pco_abi.shareds = 32;
  command.vertex_pco_abi.push_constant_count = 32;
  command.fragment_pco_abi = {};
  command.fragment_pco_abi.temps = kPcoTemporaryCount;
  command.fragment_pco_abi.shareds = 4;
  command.fragment_pco_abi.push_constant_count = 4;
  command.position_output_start = 0;
  command.position_output_count = 4;
  command.fragment_position_start = 0;
  command.fragment_position_count = 0;
  command.varying_output_start = 0;
  command.varying_output_count = 0;
  command.fragment_varying_start = 0;
  command.fragment_varying_count = 0;
  command.cull_face = 0;
  command.depth_enable = 0;
  command.depth_write = 0;
  command.depth_func = 0;
  set_pco_resolution(800, 600);
  error.fill(0);
  if (pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size()) !=
      0) {
    Fail("valid 800x600 Ideas PCO API submit was rejected: " +
         std::string(error.data()));
  }
  set_pco_resolution(640, 480);
  error.fill(0);
  if (pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size()) ==
          0 ||
      std::string(error.data()).find("80x60 or 800x600") ==
          std::string::npos) {
    Fail("unsupported 640x480 Ideas PCO API resolution was not rejected");
  }
  set_pco_resolution(80, 60);
  error.fill(0);
  if (pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size()) !=
      0) {
    Fail("TEMP64 Ideas API submit was rejected: " +
         std::string(error.data()));
  }
  command.vertex_pco_abi.temps = kPcoTemporaryCount + 1U;
  error.fill(0);
  if (pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size()) ==
          0 ||
      std::string(error.data()).find("vertex ABI exceeds model bounds") ==
          std::string::npos) {
    Fail("TEMP65 vertex ABI was not rejected at the bridge boundary");
  }
  command.vertex_pco_abi.temps = kPcoTemporaryCount;
  command.fragment_pco_abi.temps = kPcoTemporaryCount + 1U;
  error.fill(0);
  if (pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size()) ==
          0 ||
      std::string(error.data()).find("fragment ABI exceeds model bounds") ==
          std::string::npos) {
    Fail("TEMP65 fragment ABI was not rejected at the bridge boundary");
  }
  command = conditionals_command;

  expect_abi_rejected(command.vertex_pco_abi, "vertex");
  expect_abi_rejected(command.fragment_pco_abi, "fragment");
  error.fill(0);
  command.fragment_position_count = 4;
  if (pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size()) ==
          0 ||
      std::string(error.data()).find("ABI/viewport metadata") ==
          std::string::npos) {
    Fail("interpolated fragment-position API linkage was not rejected");
  }
  command.fragment_position_count = 0;
  error.fill(0);
  --command.vertex_pco_size;
  if (pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size()) ==
          0 ||
      std::string(error.data()).find("shader binary/profile") ==
          std::string::npos) {
    Fail("truncated PCO binary was not rejected synchronously");
  }
  ++command.vertex_pco_size;
  error.fill(0);
  if (pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size()) !=
      0) {
    Fail("valid PCO API-v8 submit failed: " + std::string(error.data()));
  }

  // The bridge contract promises that none of these producer allocations are
  // consulted by the deferred atexit run.
  std::fill(vertices.begin(), vertices.end(), UINT8_C(0xa5));
  std::fill(vertex_pco.begin(), vertex_pco.end(), UINT8_C(0x5a));
  std::fill(fragment_pco.begin(), fragment_pco.end(), UINT8_C(0xc3));
  std::fill(vertex_shared.begin(), vertex_shared.end(), UINT32_C(0xdeadbeef));
  std::fill(fragment_shared.begin(), fragment_shared.end(),
            UINT32_C(0xfeedface));
  vertices.clear();
  vertex_pco.clear();
  fragment_pco.clear();
  vertex_shared.clear();
  fragment_shared.clear();
  command = {};
  return EXIT_SUCCESS;
}
