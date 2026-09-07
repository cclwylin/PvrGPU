// A render pass resumed after a framebuffer switch must LOAD the driver's
// saved attachment, including pixels no fragment touches.  Exercise the public
// bridge, deferred payload ownership, Submitter DRAM transfer and PBE together.
#include "pvrgpu_systemc_api.h"
#include "shader/pco_iss.h"

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
#include <string>
#include <system_error>
#include <vector>

namespace {

constexpr std::uint32_t kWidth = 4;
constexpr std::uint32_t kHeight = 4;

[[noreturn]] void Fail(const std::string &message) {
  std::fprintf(stderr, "pco-initial-color-load-test: %s\n", message.c_str());
  std::_Exit(EXIT_FAILURE);
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

struct Fixture {
  std::vector<std::uint8_t> vertices;
  std::vector<std::uint8_t> vertex_pco =
      pvrgpu::stub::VaryingsOneVertexPcoBinary();
  std::vector<std::uint8_t> fragment_pco =
      pvrgpu::stub::VaryingsOneFragmentPcoBinary();
  std::vector<std::uint8_t> initial;
  pvrgpu_systemc_driver_command draw{};
  pvrgpu_systemc_driver_command sequence{};

  Fixture(const char *format, unsigned channels, bool normalized,
          bool clipped = false) {
    // This public shader consumes x/y and supplies clip z=0,w=1 itself.
    // Move all vertices past the right clip plane; changing input z cannot
    // clip this shader because its vec2 input ignores that component.
    const float clip_offset = clipped ? 4.0F : 0.0F;
    const std::array<std::array<float, 4>, 3> positions = {{
        {-1.0F + clip_offset, -1.0F, 0.0F, 1.0F},
        {1.0F + clip_offset, -1.0F, 0.0F, 1.0F},
        {-1.0F + clip_offset, 1.0F, 0.0F, 1.0F},
    }};
    vertices.resize(sizeof(positions));
    std::memcpy(vertices.data(), positions.data(), vertices.size());
    if (channels != 4) {
      // FITRP + WDF take 16 bytes, followed by one eight-byte PIXOUT per
      // component.  Move the end bit onto the last retained export.
      fragment_pco.resize(16U + 8U * channels);
      fragment_pco[16U + 8U * (channels - 1U) + 2U] |= UINT8_C(0x80);
    }
    const unsigned bpp = normalized ? 4U : channels * 4U;
    initial.resize(kWidth * kHeight * bpp);
    for (std::size_t byte = 0; byte < initial.size(); ++byte)
      initial[byte] = static_cast<std::uint8_t>((byte * 37U + 19U) % 251U);

    draw.version = PVRGPU_SYSTEMC_API_VERSION;
    draw.schema = "pvrgpu.driver-command.v1";
    draw.producer = "pvrgpu-gallium-driver";
    draw.command = "draw_pco_triangles";
    draw.case_name = "external-attachment-load";
    draw.format = format;
    draw.frame = 1;
    draw.framebuffer_width = draw.width = kWidth;
    draw.framebuffer_height = draw.height = kHeight;
    draw.clear_color_bits[3] = FloatBits(1.0F);
    draw.raw_vertex_data = vertices.data();
    draw.raw_vertex_data_size = vertices.size();
    draw.vertex_stride = 4U * sizeof(float);
    draw.vertex_count = 3;
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
    const std::array<float, 3> viewport = {2.0F, 2.0F, 0.5F};
    std::memcpy(draw.viewport_scale_bits, viewport.data(), sizeof(viewport));
    std::memcpy(draw.viewport_translate_bits, viewport.data(), sizeof(viewport));
    draw.half_pixel_center = 1;
    draw.depth_clip_near = 1;
    draw.depth_clip_far = 1;
    draw.sample_mask = UINT32_MAX;
    draw.color_mask = 0x0f;
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
    draw.initial_color_attachment_bytes = initial.data();
    draw.initial_color_attachment_bytes_size = initial.size();

    sequence.version = PVRGPU_SYSTEMC_API_VERSION;
    sequence.schema = draw.schema;
    sequence.producer = draw.producer;
    sequence.command = "draw_pco_sequence";
    sequence.case_name = draw.case_name;
    sequence.format = format;
    sequence.frame = 1;
    sequence.framebuffer_width = sequence.width = kWidth;
    sequence.framebuffer_height = sequence.height = kHeight;
    sequence.draw_count = 1;
    sequence.ia_vertices = 3;
    sequence.ia_primitives = 1;
    sequence.clip_invocations = 1;
    sequence.pco_sequence_command_count = 1;
    sequence.pco_sequence_commands = &draw;
  }
};

struct Submission {
  std::string jsonl;
  std::string stderr_log;
  std::string outdir;
  pvrgpu_systemc_submit_info info{};

  Submission(const std::filesystem::path &root,
             const pvrgpu_systemc_driver_command *command) {
    std::filesystem::create_directories(root / "out");
    jsonl = (root / "model.jsonl").string();
    stderr_log = (root / "model.stderr.log").string();
    outdir = (root / "out").string();
    info.version = PVRGPU_SYSTEMC_API_VERSION;
    info.command = command;
    info.jsonl_path = jsonl.c_str();
    info.stderr_path = stderr_log.c_str();
    info.outdir = outdir.c_str();
    info.memory_mode = "direct";
  }
};

void VerifyInitialLoad(const std::filesystem::path &root,
                       const char *name, const char *format,
                       unsigned channels, bool normalized,
                       bool clipped = false, bool implicit_single_target = false) {
  Fixture fixture(format, channels, normalized, clipped);
  if (implicit_single_target)
    fixture.draw.render_target_count = 0;
  const std::vector<std::uint8_t> expected = fixture.initial;
  Submission submission(root / name, &fixture.sequence);
  std::array<char, 512> error{};
  if (pvrgpu_systemc_submit_driver_command(
          &submission.info, error.data(), error.size()) != 0)
    Fail(std::string(name) + " submit failed: " + error.data());

  // Submission is deferred.  The caller is allowed to reuse its payload as
  // soon as submit returns; a borrowed pointer would now load 0xcc everywhere.
  std::fill(fixture.initial.begin(), fixture.initial.end(), UINT8_C(0xcc));
  const unsigned bpp = normalized ? 4U : channels * 4U;
  std::vector<std::uint8_t> pixels(expected.size(), UINT8_C(0xa5));
  pvrgpu_systemc_readback_info readback{};
  readback.version = PVRGPU_SYSTEMC_API_VERSION;
  readback.width = kWidth;
  readback.height = kHeight;
  readback.bytes_per_pixel = bpp;
  readback.pixels = pixels.data();
  readback.pixels_size = pixels.size();
  if (pvrgpu_systemc_flush_readback(&readback, error.data(), error.size()) != 0)
    Fail(std::string(name) + " flush failed: " + error.data());
  if (readback.pixels_written != 1)
    Fail(std::string(name) + " published no pixels");
  if (clipped) {
    std::ifstream input(submission.jsonl);
    const std::string report{std::istreambuf_iterator<char>(input),
                             std::istreambuf_iterator<char>()};
    if (report.find("\"ps_invocations\":0") == std::string::npos ||
        report.find("\"pbe_fragment_writes\":0") == std::string::npos)
      Fail(std::string(name) + " unexpectedly shaded a fragment");
  }

  for (unsigned y = 0; y < kHeight; ++y) {
    for (unsigned x = 0; x < kWidth; ++x) {
      const std::size_t offset = (y * kWidth + x) * bpp;
      // Leave the triangle's diagonal out of the assertion: either edge-rule
      // ownership is legal for this fixture, but strictly exterior texels
      // must survive byte-for-byte, across every integer component.
      if ((clipped || x + y > 3U) &&
          !std::equal(expected.begin() + offset,
                      expected.begin() + offset + bpp,
                      pixels.begin() + offset))
        Fail(std::string(name) + " lost an untouched initial texel");
    }
  }
  if (!clipped) {
    // The lower-left sample is strictly inside the triangle.  Checking its
    // shader output ensures success is not merely returning the input bytes.
    if (normalized) {
      const std::array<std::uint8_t, 4> shaded = {0, 0, 0, 255};
      if (!std::equal(shaded.begin(), shaded.end(), pixels.begin()))
        Fail(std::string(name) + " did not shade a covered RGBA8 texel");
    } else {
      const std::array<std::uint32_t, 4> shaded = {
          FloatBits(-0.75F), FloatBits(-0.75F), 0U, FloatBits(1.0F)};
      if (std::memcmp(pixels.data(), shaded.data(), bpp) != 0)
        Fail(std::string(name) + " did not shade a covered integer texel");
    }
  }
}

void VerifyRejectedPayloads(const std::filesystem::path &root) {
  const auto reject = [&](Fixture &fixture, const char *reason) {
    Submission submission(root / reason, &fixture.sequence);
    std::array<char, 512> error{};
    if (pvrgpu_systemc_submit_driver_command(
            &submission.info, error.data(), error.size()) == 0)
      Fail(std::string(reason) + " was accepted");
    if (std::string(error.data()).find("initial color attachment") ==
        std::string::npos)
      Fail(std::string(reason) + " was rejected for a different reason: " +
           error.data());
  };
  {
    Fixture fixture("PIPE_FORMAT_R8G8B8A8_UNORM", 4, true);
    fixture.draw.initial_color_attachment_bytes = nullptr;
    reject(fixture, "missing-pointer");
  }
  {
    Fixture fixture("PIPE_FORMAT_R8G8B8A8_UNORM", 4, true);
    fixture.draw.initial_color_attachment_bytes_size = 0;
    reject(fixture, "missing-size");
  }
  {
    Fixture fixture("PIPE_FORMAT_R32G32B32A32_UINT", 4, false);
    fixture.draw.initial_color_attachment_bytes_size -= 1;
    reject(fixture, "truncated-wide-payload");
  }
  {
    Fixture fixture("PIPE_FORMAT_R32G32_UINT", 2, false);
    fixture.initial.push_back(0);
    fixture.draw.initial_color_attachment_bytes = fixture.initial.data();
    fixture.draw.initial_color_attachment_bytes_size = fixture.initial.size();
    reject(fixture, "oversized-payload");
  }
  {
    Fixture fixture("PIPE_FORMAT_R8G8B8A8_UNORM", 4, true);
    fixture.draw.render_target_count = 2;
    reject(fixture, "external-load-with-mrt");
  }
  {
    Fixture fixture("PIPE_FORMAT_R8G8B8A8_UNORM", 4, true);
    std::array<pvrgpu_systemc_driver_command, 2> draws = {
        fixture.draw, fixture.draw};
    draws[0].initial_color_attachment_bytes = nullptr;
    draws[0].initial_color_attachment_bytes_size = 0;
    draws[1].color_attachment_source_command_index = 0;
    fixture.sequence.draw_count = 2;
    fixture.sequence.ia_vertices = 6;
    fixture.sequence.ia_primitives = 2;
    fixture.sequence.clip_invocations = 2;
    fixture.sequence.pco_sequence_command_count = 2;
    fixture.sequence.pco_sequence_commands = draws.data();
    reject(fixture, "alias-and-external-load");
  }
}

}  // namespace

int main() {
  const auto nonce =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      ("pvrgpu-pco-initial-color-load-test-" + std::to_string(nonce));
  VerifyRejectedPayloads(root);
  VerifyInitialLoad(root, "rgba8", "PIPE_FORMAT_R8G8B8A8_UNORM", 4, true);
  VerifyInitialLoad(root, "r32ui", "PIPE_FORMAT_R32_UINT", 1, false);
  VerifyInitialLoad(root, "rg32ui", "PIPE_FORMAT_R32G32_UINT", 2, false);
  VerifyInitialLoad(root, "rgba32ui", "PIPE_FORMAT_R32G32B32A32_UINT", 4, false);
  VerifyInitialLoad(root, "fully-clipped", "PIPE_FORMAT_R8G8B8A8_UNORM",
                    4, true, true);
  VerifyInitialLoad(root, "implicit-single-target", "PIPE_FORMAT_R32G32_UINT",
                    2, false, false, true);
  std::error_code error;
  std::filesystem::remove_all(root, error);
  std::puts("pco-initial-color-load-test: PASS");
  return EXIT_SUCCESS;
}
