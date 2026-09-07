// API-v20 depth/stencil LOAD and readback across separate native submissions.
// A spatially varying attachment contains depths both before and behind the
// triangle, so neither a uniform clear nor a copied input can satisfy the test.
#include "pvrgpu_systemc_api.h"
#include "shader/pco_iss.h"
#include "pco_depth_feedback_fixture.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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

constexpr unsigned kWidth = 4;
constexpr unsigned kHeight = 4;
constexpr unsigned kZ16 = 268;
constexpr unsigned kZ24S8 = 272;
constexpr unsigned kZ32F = 271;
constexpr unsigned kZ32FS8 = 279;

[[noreturn]] void Fail(const std::string &message) {
  std::fprintf(stderr, "pco-initial-depth-load-test: %s\n", message.c_str());
  std::_Exit(EXIT_FAILURE);
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

unsigned NativeBytes(unsigned format) {
  return format == kZ16 ? 2U : format == kZ32FS8 ? 8U : 4U;
}

bool HasStencil(unsigned format) { return format == kZ24S8 || format == kZ32FS8; }

std::uint32_t NativeDepth(float depth, unsigned format) {
  if (format == kZ32F || format == kZ32FS8)
    return FloatBits(depth);
  const double maximum = format == kZ16 ? 65535.0 : 16777215.0;
  return static_cast<std::uint32_t>(std::nearbyint(depth * maximum));
}

void StoreDepth(std::vector<std::uint8_t> &bytes, std::size_t pixel,
                unsigned format, std::uint32_t depth) {
  const unsigned bpp = NativeBytes(format);
  const unsigned depth_bytes = format == kZ16 ? 2U : format == kZ24S8 ? 3U : 4U;
  for (unsigned byte = 0; byte < depth_bytes; ++byte)
    bytes[pixel * bpp + byte] = static_cast<std::uint8_t>(depth >> (byte * 8));
}

struct Fixture {
  std::vector<std::uint8_t> vertices;
  std::vector<std::uint8_t> vertex_pco = pvrgpu::stub::VaryingsOneVertexPcoBinary();
  std::vector<std::uint8_t> fragment_pco = pvrgpu::stub::VaryingsOneFragmentPcoBinary();
  std::vector<std::uint8_t> initial;
  pvrgpu_systemc_driver_command draw{};
  pvrgpu_systemc_driver_command sequence{};

  Fixture(unsigned format, unsigned samples) {
    const std::array<std::array<float, 4>, 3> positions = {{
        {-1, -1, 0, 1}, {1, -1, 0, 1}, {-1, 1, 0, 1}}};
    vertices.resize(sizeof(positions));
    std::memcpy(vertices.data(), positions.data(), vertices.size());
    initial.resize(kWidth * kHeight * samples * NativeBytes(format), 0);
    for (unsigned pixel = 0; pixel < kWidth * kHeight * samples; ++pixel) {
      const float depth = (pixel % 2 ? 0.75F : 0.125F) +
                          static_cast<float>(pixel % 4) * 0.03125F;
      StoreDepth(initial, pixel, format, NativeDepth(depth, format));
      if (HasStencil(format))
        initial[pixel * NativeBytes(format) + (format == kZ24S8 ? 3U : 4U)] =
            static_cast<std::uint8_t>(pixel * 3 + 11);
    }
    draw.version = PVRGPU_SYSTEMC_API_VERSION;
    draw.schema = "pvrgpu.driver-command.v1";
    draw.producer = "pvrgpu-gallium-driver";
    draw.command = "draw_pco_triangles";
    draw.case_name = "external-depth-continuity";
    draw.format = "PIPE_FORMAT_R8G8B8A8_UNORM";
    draw.frame = 1;
    draw.framebuffer_width = draw.width = kWidth;
    draw.framebuffer_height = draw.height = kHeight;
    draw.clear_color_bits[3] = FloatBits(1);
    draw.raw_vertex_data = vertices.data();
    draw.raw_vertex_data_size = vertices.size();
    draw.vertex_stride = 4 * sizeof(float);
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
    draw.fragment_output_mask[0] = 0x0f;
    const std::array<float, 3> viewport = {2, 2, 0.5F};
    std::memcpy(draw.viewport_scale_bits, viewport.data(), sizeof(viewport));
    std::memcpy(draw.viewport_translate_bits, viewport.data(), sizeof(viewport));
    draw.half_pixel_center = 1;
    draw.depth_clip_near = draw.depth_clip_far = 1;
    draw.sample_mask = UINT32_MAX;
    draw.raster_samples = samples;
    draw.multisample = samples > 1;
    draw.color_mask = 0x0f;
    draw.depth_enable = draw.depth_write = 1;
    draw.depth_func = 1; // PIPE_FUNC_LESS.
    draw.depth_clear_bits = FloatBits(1);
    draw.depth_format = format;
    draw.stencil_enable = HasStencil(format);
    for (unsigned face = 0; face < 2; ++face) {
      draw.stencil_func[face] = 7; // PIPE_FUNC_ALWAYS.
      draw.stencil_pass_op[face] = 3; // PIPE_STENCIL_OP_INCR.
      draw.stencil_value_mask[face] = draw.stencil_write_mask[face] = 0xff;
    }
    draw.color_attachment_source_command_index = PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR;
    draw.depth_attachment_source_command_index = PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR;
    draw.blend_rgb_equation = draw.blend_alpha_equation = PVRGPU_SYSTEMC_PCO_BLEND_EQUATION_ADD;
    draw.blend_source_rgb_factor = draw.blend_source_alpha_factor = PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE;
    draw.blend_destination_rgb_factor = draw.blend_destination_alpha_factor = PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ZERO;
    SetInitial();
    sequence.version = PVRGPU_SYSTEMC_API_VERSION;
    sequence.schema = draw.schema;
    sequence.producer = draw.producer;
    sequence.command = "draw_pco_sequence";
    sequence.case_name = draw.case_name;
    sequence.format = draw.format;
    sequence.frame = 1;
    sequence.framebuffer_width = sequence.width = kWidth;
    sequence.framebuffer_height = sequence.height = kHeight;
    sequence.draw_count = 1;
    sequence.ia_vertices = 3;
    sequence.ia_primitives = sequence.clip_invocations = 1;
    sequence.pco_sequence_command_count = 1;
    sequence.pco_sequence_commands = &draw;
  }

  void SetInitial() {
    draw.initial_depth_attachment_bytes = initial.data();
    draw.initial_depth_attachment_bytes_size = initial.size();
  }
};

struct Submission {
  std::string jsonl, stderr_log, outdir;
  pvrgpu_systemc_submit_info info{};
  Submission(const std::filesystem::path &root, const pvrgpu_systemc_driver_command *command) {
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

void VerifyContinuity(const std::filesystem::path &root, unsigned format, unsigned samples,
                      bool shader_depth = false) {
  Fixture fixture(format, samples);
  std::array<std::uint32_t, 1> depth_shared{};
  if (shader_depth) {
    fixture.fragment_pco = DepthFeedbackFragmentFixture();
    fixture.draw.fragment_pco = fixture.fragment_pco.data();
    fixture.draw.fragment_pco_size = fixture.fragment_pco.size();
    fixture.draw.fragment_shared = depth_shared.data();
    fixture.draw.fragment_shared_count = depth_shared.size();
    fixture.draw.fragment_pco_abi.shareds = depth_shared.size();
  }
  const auto original = fixture.initial;
  for (unsigned pass = 0; pass < 2; ++pass) {
    const std::string name = (shader_depth ? "shader-" : "") +
                             std::to_string(format) + "-s" +
                             std::to_string(samples) + "-pass" + std::to_string(pass);
    const float incoming_depth = pass == 0 ? 0.5F : 0.375F;
    // Geometry depth would reject the high-depth input pixels, while the
    // shader's real DEPTHF value passes.  This distinguishes late-Z from
    // merely preserving/exporting a shader depth without using it to test.
    fixture.draw.viewport_translate_bits[2] = FloatBits(shader_depth ? 0.875F : incoming_depth);
    depth_shared[0] = FloatBits(incoming_depth);
    fixture.SetInitial();
    const auto before = fixture.initial;
    Submission submission(root / name, &fixture.sequence);
    std::array<char, 512> error{};
    if (pvrgpu_systemc_submit_driver_command(&submission.info, error.data(), error.size()) != 0)
      Fail(name + " submit: " + error.data());
    std::fill(fixture.initial.begin(), fixture.initial.end(), UINT8_C(0xdd));
    std::vector<std::uint8_t> result(before.size(), UINT8_C(0xa5));
    pvrgpu_systemc_readback_info readback{};
    readback.version = PVRGPU_SYSTEMC_API_VERSION;
    readback.width = kWidth;
    readback.height = kHeight;
    readback.bytes_per_pixel = NativeBytes(format);
    readback.attachment = UINT32_MAX;
    readback.depth_format = format;
    readback.sample_count = samples;
    readback.pixels = result.data();
    readback.pixels_size = result.size();
    if (pvrgpu_systemc_flush_readback(&readback, error.data(), error.size()) != 0 ||
        readback.pixels_written != 1)
      Fail(name + " depth flush: " + error.data());
    if (shader_depth) {
      std::ifstream log(submission.jsonl);
      const std::string evidence{std::istreambuf_iterator<char>(log),
                                 std::istreambuf_iterator<char>()};
      if (evidence.find("\"depthf\":1") == std::string::npos)
        Fail(name + " missing decoded DEPTHF evidence");
    }
    for (unsigned y = 0; y < kHeight; ++y) {
      for (unsigned x = 0; x < kWidth; ++x) {
        // Exclude the diagonal: every sample of strict interior/exterior
        // pixels has unambiguous coverage for each supported sample pattern.
        if (x + y == 3)
          continue;
        for (unsigned sample = 0; sample < samples; ++sample) {
          const unsigned pixel = (y * kWidth + x) * samples + sample;
          auto expected = before;
          if (x + y < 3 && pixel % 2) {
            StoreDepth(expected, pixel, format, NativeDepth(incoming_depth, format));
            if (HasStencil(format)) {
              const unsigned offset = pixel * NativeBytes(format) + (format == kZ24S8 ? 3U : 4U);
              expected[offset] = static_cast<std::uint8_t>(std::min<unsigned>(before[offset] + 1U, 255U));
            }
          }
          const unsigned offset = pixel * NativeBytes(format);
          if (!std::equal(expected.begin() + offset, expected.begin() + offset + NativeBytes(format),
                          result.begin() + offset))
            Fail(name + " mismatch at pixel " + std::to_string(x) + "," +
                 std::to_string(y) + " sample " + std::to_string(sample));
        }
      }
    }
    if (result == original)
      Fail(name + " returned the input without executing depth/stencil updates");
    fixture.initial = std::move(result); // Actual output becomes next submission's input.
  }
}

void VerifyRejectedPayloads(const std::filesystem::path &root) {
  for (unsigned mutation = 0; mutation < 4; ++mutation) {
    Fixture fixture(kZ32FS8, 1);
    if (mutation == 0) fixture.draw.initial_depth_attachment_bytes = nullptr;
    if (mutation == 1) fixture.draw.initial_depth_attachment_bytes_size = 0;
    if (mutation == 2) --fixture.draw.initial_depth_attachment_bytes_size;
    if (mutation == 3) fixture.draw.depth_format = 0;
    Submission submission(root / ("reject-" + std::to_string(mutation)), &fixture.sequence);
    std::array<char, 512> error{};
    if (pvrgpu_systemc_submit_driver_command(&submission.info, error.data(), error.size()) == 0)
      Fail("invalid initial depth payload was accepted");
    const char *field = mutation == 3 ? "depth_format" : "initial depth";
    if (std::string(error.data()).find(field) == std::string::npos)
      Fail(std::string("invalid depth payload rejected for another reason: ") + error.data());
  }
}

void VerifyRejectedMultisampleDependencies(const std::filesystem::path &root) {
  for (const unsigned samples : {2U, 4U, 8U, 16U}) {
    for (const unsigned source : {
             PVRGPU_SYSTEMC_PCO_TEXTURE_PREVIOUS_COLOR_ATTACHMENT,
             PVRGPU_SYSTEMC_PCO_TEXTURE_PREVIOUS_DEPTH_ATTACHMENT}) {
      Fixture fixture(kZ32F, samples);
      // Z32_UNORM is the existing sampled-depth attachment contract.  The
      // imported float bytes are irrelevant: this is rejected before execute.
      fixture.draw.depth_format = 270U;
      std::array<pvrgpu_systemc_driver_command, 2> draws = {
          fixture.draw, fixture.draw};
      draws[1].initial_depth_attachment_bytes = nullptr;
      draws[1].initial_depth_attachment_bytes_size = 0;
      draws[1].color_attachment_source_command_index = 0;
      draws[1].depth_attachment_source_command_index = 0;
      // Metadata is otherwise a valid single-sample 2D attachment alias.
      // Its width*4 pitch would treat pixel 0 sample 1 as pixel 1 on MSAA.
      std::array<std::uint32_t, 20> descriptor{};
      draws[1].sampled_texture_count = 1;
      draws[1].fragment_shared = descriptor.data();
      draws[1].fragment_shared_count = descriptor.size();
      draws[1].fragment_pco_abi.shareds = descriptor.size();
      pvrgpu_systemc_pco_sequence_texture texture{};
      texture.source = source;
      texture.stage = PVRGPU_SYSTEMC_PCO_SHADER_STAGE_FRAGMENT;
      texture.format = source == PVRGPU_SYSTEMC_PCO_TEXTURE_PREVIOUS_COLOR_ATTACHMENT
                           ? "PIPE_FORMAT_R8G8B8A8_UNORM"
                           : "PIPE_FORMAT_Z32_UNORM";
      texture.declared_bytes_size = kWidth * kHeight * 4;
      texture.mip_count = 1;
      texture.mip[0] = {kWidth, kHeight, kWidth * 4, 0};
      texture.normalized_coordinates = 1;
      fixture.sequence.draw_count = 2;
      fixture.sequence.ia_vertices = 6;
      fixture.sequence.ia_primitives = fixture.sequence.clip_invocations = 2;
      fixture.sequence.pco_sequence_command_count = draws.size();
      fixture.sequence.pco_sequence_commands = draws.data();
      fixture.sequence.pco_sequence_texture_count = 1;
      fixture.sequence.pco_sequence_textures = &texture;
      const std::string name = "reject-dependency-" + std::to_string(source) +
                               "-s" + std::to_string(samples);
      Submission submission(root / name, &fixture.sequence);
      std::array<char, 512> error{};
      if (pvrgpu_systemc_submit_driver_command(&submission.info, error.data(), error.size()) == 0)
        Fail(name + " accepted an ordinary texture alias of MSAA storage");
      if (std::string(error.data()).find("single-sample producer") == std::string::npos)
        Fail(name + " rejected for another reason: " + error.data());
    }
  }
}

} // namespace

int main() {
  const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  const auto root = std::filesystem::temp_directory_path() /
      ("pvrgpu-pco-initial-depth-load-test-" + std::to_string(nonce));
  VerifyRejectedPayloads(root);
  VerifyRejectedMultisampleDependencies(root);
  for (const unsigned format : {kZ16, kZ24S8, kZ32F, kZ32FS8})
    for (const unsigned samples : {1U, 4U}) {
      VerifyContinuity(root, format, samples);
      VerifyContinuity(root, format, samples, true);
    }
  std::error_code error;
  std::filesystem::remove_all(root, error);
  std::puts("pco-initial-depth-load-test: PASS");
  return EXIT_SUCCESS;
}
