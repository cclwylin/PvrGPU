// A render pass resumed after a framebuffer switch must LOAD the driver's
// saved attachment, including pixels no fragment touches.  Exercise the public
// bridge, deferred payload ownership, Submitter DRAM transfer and PBE together.
#include "pvrgpu_systemc_api.h"
#include "shader/pco_iss.h"
#include "pco_uniform_buffer_fixtures.h"

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
                       bool clipped = false, bool implicit_single_target = false,
                       unsigned reserved_temps = 0) {
  Fixture fixture(format, channels, normalized, clipped);
  if (reserved_temps != 0) {
    fixture.draw.vertex_pco_abi.temps = reserved_temps;
    fixture.draw.fragment_pco_abi.temps = reserved_temps;
  }
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

void VerifyOversizedViewport(const std::filesystem::path &root, bool translated) {
  Fixture fixture("PIPE_FORMAT_R32G32B32A32_FLOAT", 4, false);
  fixture.draw.width = fixture.draw.height = 8;
  fixture.sequence.width = fixture.sequence.height = 8;
  fixture.draw.viewport_scale_bits[0] = FloatBits(4.0F);
  fixture.draw.viewport_scale_bits[1] = FloatBits(4.0F);
  const float offset_x = translated ? 6.0F : 4.0F;
  const float offset_y = translated ? 5.0F : 4.0F;
  fixture.draw.viewport_translate_bits[0] = FloatBits(offset_x);
  fixture.draw.viewport_translate_bits[1] = FloatBits(offset_y);
  Submission submission(root / (translated ? "viewport-translated" : "viewport-large"),
                        &fixture.sequence);
  std::array<char, 512> error{};
  if (pvrgpu_systemc_submit_driver_command(&submission.info,
          error.data(), error.size()) != 0)
    Fail(std::string("oversized viewport submit: ") + error.data());
  std::vector<std::uint8_t> pixels(fixture.initial.size());
  pvrgpu_systemc_readback_info readback{};
  readback.version = PVRGPU_SYSTEMC_API_VERSION;
  readback.width = kWidth;
  readback.height = kHeight;
  readback.bytes_per_pixel = 16;
  readback.pixels = pixels.data();
  readback.pixels_size = pixels.size();
  if (pvrgpu_systemc_flush_readback(&readback, error.data(), error.size()) != 0 ||
      readback.pixels_written != 1)
    Fail(std::string("oversized viewport readback: ") + error.data());
  for (unsigned y = 0; y < kHeight; ++y) {
    for (unsigned x = 0; x < kWidth; ++x) {
      const std::size_t offset = (y * kWidth + x) * 16;
      const bool covered = !translated || (x >= 2 && y >= 1);
      if (!covered) {
        if (std::memcmp(pixels.data() + offset,
                        fixture.initial.data() + offset, 16) != 0)
          Fail("translated oversized viewport damaged an exterior LOAD pixel");
        continue;
      }
      const std::array<std::uint32_t, 4> expected = {
          FloatBits((static_cast<float>(x) + 0.5F - offset_x) / 4.0F),
          FloatBits((static_cast<float>(y) + 0.5F - offset_y) / 4.0F),
          0, FloatBits(1.0F)};
      if (std::memcmp(pixels.data() + offset, expected.data(), 16) != 0)
        Fail("oversized viewport was resized instead of clipped to its framebuffer");
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

void VerifyUniformBufferRejections(const std::filesystem::path &root) {
  const auto reject = [&](auto mutate, const char *name) {
    Fixture fixture("PIPE_FORMAT_R32G32B32A32_FLOAT", 4, false);
    std::array<std::uint8_t, 16> bytes{};
    std::array<std::uint32_t, 4> shared = {0, 0, 16, 0};
    std::array<pvrgpu_systemc_pco_uniform_buffer, 2> buffers = {{
        {PVRGPU_SYSTEMC_PCO_SHADER_STAGE_FRAGMENT, 0, bytes.data(), bytes.size()},
        {PVRGPU_SYSTEMC_PCO_SHADER_STAGE_FRAGMENT, 0, bytes.data(), bytes.size()},
    }};
    fixture.draw.fragment_shared = shared.data();
    fixture.draw.fragment_shared_count = shared.size();
    fixture.draw.fragment_pco_abi.shareds = shared.size();
    fixture.draw.fragment_pco_abi.uniform_buffer_descriptor_count = 1;
    fixture.draw.fragment_pco_abi.push_constant_start = 4;
    fixture.draw.uniform_buffers = buffers.data();
    fixture.draw.uniform_buffer_count = 1;
    mutate(fixture.draw, buffers, shared);
    Submission submission(root / name, &fixture.sequence);
    std::array<char, 512> error{};
    if (pvrgpu_systemc_submit_driver_command(&submission.info,
            error.data(), error.size()) == 0 ||
        std::string(error.data()).find("uniform buffer") == std::string::npos)
      Fail(std::string(name) + " did not reject the invalid UBO: " + error.data());
  };
  reject([](auto &d, auto &, auto &) { d.uniform_buffers = nullptr; }, "ubo-list-null");
  reject([](auto &d, auto &, auto &) { d.uniform_buffer_count = 0; }, "ubo-list-count");
  reject([](auto &d, auto &, auto &) { d.uniform_buffer_count = 2; }, "ubo-duplicate");
  reject([](auto &, auto &b, auto &) { b[0].stage = 2; }, "ubo-stage");
  reject([](auto &, auto &b, auto &) { b[0].block_index = 1; }, "ubo-block");
  reject([](auto &, auto &b, auto &) { b[0].bytes = nullptr; }, "ubo-bytes-null");
  reject([](auto &, auto &b, auto &) { b[0].bytes_size = 0; }, "ubo-bytes-zero");
  reject([](auto &, auto &b, auto &) { b[0].bytes_size = 65537; }, "ubo-bytes-large");
  reject([](auto &, auto &, auto &s) { s[2] = 12; }, "ubo-size-mismatch");
  reject([](auto &, auto &, auto &s) { s[0] = 128; }, "ubo-prepatched-address");
  reject([](auto &, auto &, auto &s) { s[3] = 4; }, "ubo-dynamic-offset");
  reject([](auto &d, auto &, auto &) { d.fragment_pco_abi.push_constant_start = 0; },
         "ubo-push-overlap");
}

void VerifyDeferredUniformBuffers(const std::filesystem::path &root,
                                  const char *memory_mode) {
  Fixture fixture("PIPE_FORMAT_R32G32B32A32_FLOAT", 4, false);
  // This fragment shader has no varying inputs; use a position-only VS so
  // the fixed-function linkage does not declare an unused varying route.
  fixture.vertex_pco = pvrgpu::stub::AttributeFetchVertexPcoBinary();
  fixture.draw.vertex_pco = fixture.vertex_pco.data();
  fixture.draw.vertex_pco_size = fixture.vertex_pco.size();
  fixture.draw.vertex_pco_abi.vertex_outputs = 4;
  fixture.draw.varying_output_count = 0;
  auto fragment_pco = pvrgpu::stub::test::UniformBufferFixture(false, 4);
  std::array<pvrgpu_systemc_driver_command, 2> draws = {fixture.draw, fixture.draw};
  const std::array<std::array<float, 4>, 2> expected = {{
      {-0.75F, 2.5F, 17.25F, 1.0F}, {0.375F, -3.25F, 50.0F, 0.5F},
  }};
  std::array<std::array<std::uint8_t, 32>, 2> bytes{};
  std::array<std::array<std::uint32_t, 5>, 2> shared = {{
      {0, 0, 32, 0, 16}, {0, 0, 32, 0, 16},
  }};
  std::array<pvrgpu_systemc_pco_uniform_buffer, 2> buffers{};
  for (unsigned index = 0; index < 2; ++index) {
    // A runtime byte offset reaches the last legal vec4 of the bound range.
    std::memcpy(bytes[index].data() + 16, expected[index].data(), 16);
    buffers[index] = {PVRGPU_SYSTEMC_PCO_SHADER_STAGE_FRAGMENT, 0,
                      bytes[index].data(), bytes[index].size()};
    auto &draw = draws[index];
    draw.fragment_pco = fragment_pco.data();
    draw.fragment_pco_size = fragment_pco.size();
    draw.fragment_pco_abi.temps = 4;
    draw.fragment_pco_abi.coefficients = 4;
    draw.fragment_pco_abi.shareds = 5;
    draw.fragment_pco_abi.push_constant_start = 4;
    draw.fragment_pco_abi.push_constant_count = 1;
    draw.fragment_pco_abi.uniform_buffer_descriptor_count = 1;
    draw.fragment_shared = shared[index].data();
    draw.fragment_shared_count = shared[index].size();
    draw.fragment_varying_count = 0;
    draw.uniform_buffers = &buffers[index];
    draw.uniform_buffer_count = 1;
    draw.scissor = 1;
    draw.scissor_x = index * 2;
    draw.scissor_width = 2;
    draw.scissor_height = kHeight;
    if (index != 0) {
      draw.color_attachment_source_command_index = 0;
      draw.initial_color_attachment_bytes = nullptr;
      draw.initial_color_attachment_bytes_size = 0;
    }
  }
  fixture.sequence.pco_sequence_commands = draws.data();
  fixture.sequence.pco_sequence_command_count = 2;
  fixture.sequence.draw_count = 2;
  fixture.sequence.ia_vertices = 6;
  fixture.sequence.ia_primitives = 2;
  fixture.sequence.clip_invocations = 2;
  Submission submission(root / (std::string("ubo-deferred-") + memory_mode),
                        &fixture.sequence);
  submission.info.memory_mode = memory_mode;
  std::array<char, 512> error{};
  if (pvrgpu_systemc_submit_driver_command(&submission.info,
          error.data(), error.size()) != 0)
    Fail(std::string("UBO submit: ") + error.data());
  // Neither draw may borrow the caller's descriptor, push data or UBO bytes.
  for (unsigned index = 0; index < 2; ++index) {
    bytes[index].fill(0xcc);
    shared[index].fill(0xffffffff);
    buffers[index].bytes = nullptr;
  }
  std::vector<std::uint8_t> pixels(kWidth * kHeight * 16);
  pvrgpu_systemc_readback_info readback{};
  readback.version = PVRGPU_SYSTEMC_API_VERSION;
  readback.width = kWidth;
  readback.height = kHeight;
  readback.bytes_per_pixel = 16;
  readback.pixels = pixels.data();
  readback.pixels_size = pixels.size();
  if (pvrgpu_systemc_flush_readback(&readback, error.data(), error.size()) != 0 ||
      readback.pixels_written != 1)
    Fail(std::string("UBO readback: ") + error.data());
  for (unsigned index = 0; index < 2; ++index) {
    if (std::memcmp(pixels.data() + index * 2U * 16U,
                    expected[index].data(), 16) != 0)
      Fail("UBO per-draw snapshot/LD returned another binding's bytes");
  }
}

}  // namespace

int main(int argc, char **argv) {
  const auto nonce =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      ("pvrgpu-pco-initial-color-load-test-" + std::to_string(nonce));
  // The shared runtime's memory mode is fixed at SystemC elaboration. Each
  // additional mode therefore gets its own test process, not a mode switch.
  if (argc == 2 && std::string(argv[1]) == "ubo-bypass") {
    VerifyDeferredUniformBuffers(root, "bypass");
  } else if (argc == 2 && std::string(argv[1]) == "ubo-cache") {
    VerifyDeferredUniformBuffers(root, "cache");
  } else if (argc == 1) {
    VerifyRejectedPayloads(root);
    VerifyUniformBufferRejections(root);
    VerifyInitialLoad(root, "rgba8", "PIPE_FORMAT_R8G8B8A8_UNORM", 4, true);
    VerifyInitialLoad(root, "r32ui", "PIPE_FORMAT_R32_UINT", 1, false);
    VerifyInitialLoad(root, "rg32ui", "PIPE_FORMAT_R32G32_UINT", 2, false);
    VerifyInitialLoad(root, "rgba32ui", "PIPE_FORMAT_R32G32B32A32_UINT", 4, false);
    VerifyInitialLoad(root, "rgba32f", "PIPE_FORMAT_R32G32B32A32_FLOAT", 4, false);
    // Real shader execution with a larger resource reservation: the public
    // ABI and Submitter must retain 65/256 rather than narrowing to uint8_t.
    // Actual accesses to high TEMP indices are covered by the ISS tests.
    VerifyInitialLoad(root, "temp65-abi", "PIPE_FORMAT_R32G32B32A32_FLOAT",
                      4, false, false, false, 65);
    VerifyInitialLoad(root, "temp256-abi", "PIPE_FORMAT_R32G32B32A32_FLOAT",
                      4, false, false, false, 256);
    VerifyInitialLoad(root, "fully-clipped", "PIPE_FORMAT_R8G8B8A8_UNORM",
                      4, true, true);
    VerifyInitialLoad(root, "implicit-single-target", "PIPE_FORMAT_R32G32_UINT",
                      2, false, false, true);
    VerifyOversizedViewport(root, false);
    VerifyOversizedViewport(root, true);
    VerifyDeferredUniformBuffers(root, "direct");
  } else {
    Fail("unknown test mode");
  }
  std::error_code error;
  std::filesystem::remove_all(root, error);
  std::puts("pco-initial-color-load-test: PASS");
  return EXIT_SUCCESS;
}
