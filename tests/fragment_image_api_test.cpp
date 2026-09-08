// SPDX-License-Identifier: MIT
// Graphics API -> native VS/FS -> USC image RMW -> physical GPU readback.
#include "pvrgpu_systemc_api.h"
#include "pco_fragment_image_api_fixtures.h"
#include "shader_images.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {
using namespace pvrgpu::stub;
unsigned checks;
void Check(bool condition, const std::string &message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}
void Word(std::vector<std::uint8_t> &bytes, std::size_t offset, std::uint32_t value) {
  Check(offset <= bytes.size() && bytes.size() - offset >= 4, "test word extent");
  std::memcpy(bytes.data() + offset, &value, 4);
}
std::uint32_t Word(const std::vector<std::uint8_t> &bytes, std::size_t offset) {
  std::uint32_t word;
  Check(offset <= bytes.size() && bytes.size() - offset >= 4, "test word extent");
  std::memcpy(&word, bytes.data() + offset, 4);
  return word;
}

struct Fixture {
  unsigned kind, count;
  std::vector<std::uint8_t> vs, fs;
  std::array<float, 12> vertices{-1,-1,0,1, 3,-1,0,1, -1,3,0,1};
  std::vector<std::uint8_t> backing = std::vector<std::uint8_t>(256);
  std::vector<std::uint32_t> shared;
  std::array<pvrgpu_systemc_shader_image, 2> images{};
  pvrgpu_systemc_varying_binding no_varyings{};
  pvrgpu_systemc_driver_command draw{};

  explicit Fixture(unsigned fixture_kind = 1)
      : kind(fixture_kind), count(kind == 3 ? 2 : 1),
        vs(FragmentImageApiBinary(kind, false)), fs(FragmentImageApiBinary(kind, true)),
        shared(count * 8 + 4) {
    for (std::size_t i = 0; i < backing.size(); ++i) backing[i] = std::uint8_t(17 * i + 3);
    Word(backing, 88, 0);
    for (unsigned slot = 0; slot < count; ++slot)
      images[slot] = {slot, PVRGPU_SYSTEMC_SHADER_IMAGE_R32_UINT, 3,
         UINT64_C(0x123456789abc), backing.data(), backing.size(),
         64, 3, 2, 1, 20, 40, 4};
    shared[count * 8] = shared[count * 8 + 1] = shared[count * 8 + 2] = 1;
    shared[count * 8 + 3] = 2;
    draw.version = PVRGPU_SYSTEMC_API_VERSION;
    draw.command = "draw_pco_triangles";
    draw.case_name = "native.fragment_image_api";
    draw.format = "PIPE_FORMAT_R8G8B8A8_UNORM";
    draw.frame = 1;
    draw.width = draw.height = draw.framebuffer_width = draw.framebuffer_height = 1;
    draw.primitive_mode = 4;
    draw.vertex_count = 3;
    draw.instance_count = 1;
    draw.vertex_stride = 16;
    draw.raw_vertex_data = reinterpret_cast<const std::uint8_t *>(vertices.data());
    draw.raw_vertex_data_size = sizeof(vertices);
    draw.vertex_attribute_count = 1;
    draw.vertex_attribute_components[0] = 4;
    draw.vertex_pco = vs.data(); draw.vertex_pco_size = vs.size();
    draw.fragment_pco = fs.data(); draw.fragment_pco_size = fs.size();
    draw.vertex_pco_abi = FragmentImageApiAbi(kind, false);
    draw.fragment_pco_abi = FragmentImageApiAbi(kind, true);
    draw.fragment_shared = shared.data(); draw.fragment_shared_count = shared.size();
    draw.position_output_count = draw.varying_output_start = 4;
    draw.fragment_position_count = draw.fragment_pco_abi.coefficients;
    draw.fragment_varying_start = draw.fragment_pco_abi.coefficients;
    draw.varying_bindings = &no_varyings;
    draw.fragment_output_mask[0] = 15;
    draw.viewport_scale_bits[0] = draw.viewport_scale_bits[1] = draw.viewport_scale_bits[2] = 0x3f000000;
    std::copy_n(draw.viewport_scale_bits, 3, draw.viewport_translate_bits);
    draw.half_pixel_center = draw.depth_clip_near = draw.depth_clip_far = 1;
    draw.sample_mask = UINT32_MAX; draw.color_mask = 15;
    draw.blend_source_rgb_factor = draw.blend_source_alpha_factor = PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE;
    draw.color_attachment_source_command_index = draw.depth_attachment_source_command_index =
        PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR;
    draw.fragment_images = images.data(); draw.fragment_image_count = count;
    draw.fragment_image_descriptor_count = count;
    draw.fragment_image_read_mask = draw.fragment_image_write_mask = (1U << count) - 1;
    draw.fragment_early_tests = kind == 2;
    RefreshDescriptors();
  }
  void RefreshDescriptors() {
    for (unsigned slot = 0; slot < count; ++slot) {
      const auto &i = images[slot];
      const std::array<std::uint32_t, 8> descriptor{0,0,i.depth,i.layer_stride,
                                                  i.width,i.height,i.row_stride,i.texel_bytes};
      std::copy(descriptor.begin(), descriptor.end(), shared.begin() + 8 * slot);
    }
  }
  void ReleaseBorrowedPayloads() {
    std::fill(vs.begin(), vs.end(), 0);
    std::fill(fs.begin(), fs.end(), 0);
    vertices.fill(99);
    std::fill(shared.begin(), shared.end(), UINT32_MAX);
    std::fill(backing.begin(), backing.end(), 0x5c);
    images = {}; draw = {};
  }
};

struct Api {
  std::filesystem::path directory;
  std::string jsonl, outdir, stderr_path;
  pvrgpu_systemc_driver_command sequence{};
  pvrgpu_systemc_submit_info info{};
  std::array<char, 2048> error{};
  std::uint64_t generation = 100;
  explicit Api(const char *mode) {
    directory = std::filesystem::temp_directory_path() /
        ("pvrgpu-fs-image-api29-" + std::to_string(
          std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory / "out");
    jsonl = (directory / "model.jsonl").string();
    outdir = (directory / "out").string();
    stderr_path = (directory / "model.stderr.log").string();
    sequence.version = PVRGPU_SYSTEMC_API_VERSION;
    sequence.command = "draw_pco_sequence";
    sequence.case_name = "native.fragment_image_api";
    sequence.format = "PIPE_FORMAT_R8G8B8A8_UNORM";
    sequence.frame = 1;
    sequence.width = sequence.height = sequence.framebuffer_width = sequence.framebuffer_height = 1;
    info.version = PVRGPU_SYSTEMC_API_VERSION; info.command = &sequence;
    info.jsonl_path = jsonl.c_str(); info.outdir = outdir.c_str();
    info.stderr_path = stderr_path.c_str(); info.memory_mode = mode;
  }
  int Submit(pvrgpu_systemc_driver_command *draws, unsigned count = 1) {
    sequence.pco_sequence_commands = draws;
    sequence.pco_sequence_command_count = count;
    info.submission_generation = ++generation;
    error.fill(0);
    return pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size());
  }
  std::vector<std::uint8_t> Read(std::uint64_t token, std::size_t size) {
    std::vector<std::uint8_t> bytes(size, 0xcc);
    pvrgpu_systemc_shader_image_readback read{PVRGPU_SYSTEMC_API_VERSION, generation,
        token, bytes.data(), bytes.size(), 0};
    const auto status = pvrgpu_systemc_flush_shader_image(&read, error.data(), error.size());
    Check(status == 0 && read.data_written == 1, std::string("native FS image readback: ") + error.data());
    return bytes;
  }
  void RejectReadbacks(std::uint64_t token, std::uint64_t stale) {
    for (unsigned bad = 0; bad < 7; ++bad) {
      std::vector<std::uint8_t> bytes(256, 0xcd);
      const auto original = bytes;
      pvrgpu_systemc_shader_image_readback read{PVRGPU_SYSTEMC_API_VERSION, generation,
          token, bytes.data(), bytes.size(), 1};
      if (bad == 0) read.version--;
      if (bad == 1) read.resource_token++;
      if (bad == 2) read.submission_generation = 0;
      if (bad == 3) read.submission_generation = stale;
      if (bad == 4) read.bytes_size--;
      if (bad == 5) read.bytes = nullptr;
      if (bad == 6) read.resource_token = 0;
      Check(pvrgpu_systemc_flush_shader_image(&read, error.data(), error.size()) == 2 &&
            (bad == 0 || read.data_written == 0) && error[0] && bytes == original,
            std::string("invalid/stale image readback ") + std::to_string(bad));
    }
  }
};

void RejectBoundaries(Api &api) {
  for (unsigned bad = 0; bad < 25; ++bad) {
    Fixture f(3);
    const auto original = f.backing;
    if (bad == 0) f.draw.fragment_image_descriptor_count = 33;
    if (bad == 1) f.draw.fragment_images = nullptr;
    if (bad == 2) f.draw.fragment_image_count = 33;
    if (bad == 3) f.images[0].resource_token = 0;
    if (bad == 4) f.images[0].image_slot = 2;
    if (bad == 5) f.images[1].image_slot = 0;
    if (bad == 6) f.images[0].format = 2;
    if (bad == 7) f.images[0].access = 1;
    if (bad == 8) f.images[0].offset++;
    if (bad == 9) f.images[0].row_stride = 8;
    if (bad == 10) f.images[0].layer_stride = 24;
    if (bad == 11) f.images[0].texel_bytes = 8;
    if (bad == 12) f.images[0].width = 0;
    if (bad == 13) f.images[0].height = UINT32_MAX;
    if (bad == 14) f.images[0].bytes_size = 4;
    if (bad == 15) f.images[1].bytes_size--;
    if (bad == 16) f.shared[0] = 4;
    if (bad == 17) f.shared[4]++;
    if (bad == 18) f.draw.fragment_image_read_mask = 4;
    if (bad == 19) f.draw.fragment_image_descriptor_start = 4;
    if (bad == 20) f.draw.fragment_pco_abi.push_constant_start--;
    if (bad == 21) f.draw.fragment_early_tests = 2;
    if (bad == 22) { f.images[0].depth = 2; f.RefreshDescriptors(); }
    if (bad == 23) { f.images[0].access = 2; f.draw.fragment_image_read_mask &= ~1U; }
    if (bad == 24) f.images[0].bytes = nullptr;
    Check(api.Submit(&f.draw) == 2 && api.error[0] && f.backing == original,
          std::string("bad fragment image boundary ") + std::to_string(bad) + ": " + api.error.data());
  }
  Fixture first, second;
  Word(second.backing, 88, 77);
  second.draw.color_attachment_source_command_index = 0;
  std::array<pvrgpu_systemc_driver_command, 2> changed{first.draw, second.draw};
  Check(api.Submit(changed.data(), 2) == 2 && api.error[0],
        "cross-command immutable image alias mismatch must reject before execution");
  api.sequence.fragment_images = first.images.data();
  api.sequence.fragment_image_count = 1;
  Check(api.Submit(&first.draw) == 2 && api.error[0],
        "logical sequence wrapper cannot silently discard an image payload");
  api.sequence.fragment_images = nullptr; api.sequence.fragment_image_count = 0;
#if !defined(_WIN32)
  // Version is the only field known before validating the ABI. An old or
  // future caller may not own storage for any of the API29 tail fields.
  const auto page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
  auto *mapping = static_cast<std::uint8_t *>(mmap(nullptr, page * 2, PROT_NONE,
      MAP_PRIVATE | MAP_ANON, -1, 0));
  Check(mapping != MAP_FAILED && mprotect(mapping, page, PROT_READ | PROT_WRITE) == 0,
        "readback version guard allocation");
  auto *prefix = mapping + page - alignof(pvrgpu_systemc_shader_image_readback);
  for (const std::uint32_t version : {PVRGPU_SYSTEMC_API_VERSION - 1U,
                                     PVRGPU_SYSTEMC_API_VERSION + 1U}) {
    std::memcpy(prefix, &version, sizeof(version));
    Check(pvrgpu_systemc_flush_shader_image(
          reinterpret_cast<pvrgpu_systemc_shader_image_readback *>(prefix),
          api.error.data(), api.error.size()) == 2 &&
          std::string(api.error.data()).find("version") != std::string::npos,
          "wrong readback version accessed an unavailable API tail");
  }
  munmap(mapping, page * 2);
#endif
}

void Run(Api &api, unsigned scenario) {
  Fixture f(scenario == 1 || scenario == 2 ? 3 : scenario == 5 ? 2 : 1);
  const auto token = f.images[0].resource_token;
  auto expected = f.backing;
  bool visible = true;
  unsigned operations = f.count;
  if (scenario == 2) {
    f.images[1].offset = 128;
    Word(f.backing, 152, 0);
    expected = f.backing;
  }
  if (scenario == 3) { f.shared[f.count * 8] = UINT32_MAX; operations = 0; }
  if (scenario == 4 || scenario == 5) {
    f.draw.depth_enable = 1; f.draw.depth_func = 1;
    f.draw.depth_format = kDriverPcoDepthFormatZ32Float;
    f.draw.depth_clear_bits = 0; // Fragment .5 fails LESS against 0.
    visible = false;
    if (scenario == 5) operations = 0; // Genuine early_fragment_tests shader.
  }
  if (scenario == 6) { Word(f.backing, 88, UINT32_MAX); expected = f.backing; }
  const auto stale = api.generation;
  if (operations) {
    Word(expected, 88, Word(expected, 88) + 1);
    if (f.count == 2) {
      const auto target = f.images[1].offset + 24;
      Word(expected, target, Word(expected, target) + 2);
    }
  }
  Check(api.Submit(&f.draw) == 0, std::string("submit fragment image: ") + api.error.data());
  f.ReleaseBorrowedPayloads();
  const auto actual = api.Read(token, expected.size());
  Check(actual == expected, "GPU atomic result / offset / row padding / alias bytes mismatch");
  std::array<std::uint8_t, 4> pixel{};
  pvrgpu_systemc_readback_info read{};
  read.version = PVRGPU_SYSTEMC_API_VERSION; read.width = read.height = 1;
  read.bytes_per_pixel = 4; read.pixels = pixel.data(); read.pixels_size = pixel.size();
  Check(pvrgpu_systemc_flush_readback(&read, api.error.data(), api.error.size()) == 0 && read.pixels_written,
        std::string("fragment image framebuffer: ") + api.error.data());
  const unsigned red = scenario == 1 || scenario == 6 ? 255 : 0;
  Check(pixel == std::array<std::uint8_t,4>{std::uint8_t(visible ? red : 0),0,0,
                                          std::uint8_t(visible ? 255 : 0)},
        "atomic old value or early/late depth visibility mismatch");
  api.RejectReadbacks(token, stale);
}

void RunSequence(Api &api) {
  Fixture first, second;
  const auto token = first.images[0].resource_token;
  second.draw.color_attachment_source_command_index = 0;
  std::array<pvrgpu_systemc_driver_command, 2> draws{first.draw, second.draw};
  auto expected = first.backing;
  Word(expected, 88, 2);
  Check(api.Submit(draws.data(), draws.size()) == 0,
        std::string("same-resource sequence: ") + api.error.data());
  first.ReleaseBorrowedPayloads(); second.ReleaseBorrowedPayloads(); draws = {};
  Check(api.Read(token, expected.size()) == expected,
        "later physical draw must see first draw's GPU atomic without restoring host snapshot");
}

void TestValidator() {
  Fixture fixture(3);
  DriverCommand command;
  static_assert(sizeof(command.fragment_pco_abi) == sizeof(fixture.draw.fragment_pco_abi));
  std::memcpy(&command.fragment_pco_abi, &fixture.draw.fragment_pco_abi,
              sizeof(command.fragment_pco_abi));
  command.fragment_shared = fixture.shared;
  command.fragment_image_descriptor_count = 2;
  command.fragment_image_read_mask = command.fragment_image_write_mask = 3;
  for (unsigned slot = 0; slot < 2; ++slot) {
    const auto &source = fixture.images[slot];
    DriverShaderImage image;
    image.image_slot = slot; image.format = source.format; image.access = source.access;
    image.resource_token = source.resource_token; image.bytes = fixture.backing;
    image.offset = source.offset; image.width = source.width; image.height = source.height;
    image.depth = source.depth; image.row_stride = source.row_stride;
    image.layer_stride = source.layer_stride; image.texel_bytes = source.texel_bytes;
    command.fragment_images.push_back(std::move(image));
  }
  std::string error;
  Check(ValidateDriverShaderImages(command, &error), "validator accepts native image aliases");
  for (unsigned bad = 0; bad < 14; ++bad) {
    auto malformed = command;
    auto &image = malformed.fragment_images[0];
    if (bad == 0) image.resource_token = 0;
    if (bad == 1) image.width = UINT32_MAX;
    if (bad == 2) image.offset = UINT64_MAX;
    if (bad == 3) image.bytes[0] ^= 1;
    if (bad == 4) malformed.fragment_shared[0] = 4;
    if (bad == 5) image.image_slot = 1;
    if (bad == 6) malformed.fragment_image_descriptor_count = 33;
    if (bad == 7) malformed.fragment_image_read_mask = 4;
    if (bad == 8) malformed.fragment_early_tests = 2;
    if (bad == 9) { image.depth = 2; malformed.fragment_shared[2] = 2; }
    if (bad == 10) { image.access = 2; malformed.fragment_image_read_mask = 2; }
    if (bad == 11) image.row_stride = 8;
    if (bad == 12) image.layer_stride = 8;
    if (bad == 13) image.bytes.resize(4);
    error.clear();
    Check(!ValidateDriverShaderImages(malformed, &error) && !error.empty(),
          "image validator malformed boundary " + std::to_string(bad));
  }
}
} // namespace

int main(int argc, char **argv) {
  try {
    TestValidator();
#ifndef PVRGPU_FRAGMENT_IMAGE_VALIDATOR_ONLY
    const char *mode = argc > 1 ? argv[1] : "cache";
    Api api(mode);
    RejectBoundaries(api);
    for (unsigned scenario = 0; scenario < 7; ++scenario) Run(api, scenario);
    RunSequence(api);
    std::ifstream input(api.jsonl);
    const std::string report{std::istreambuf_iterator<char>(input), {}};
    Check(report.find("\"pool_leaks\":0") != std::string::npos &&
          report.find("\"type\":\"error\"") == std::string::npos,
          "actual graphics model did not complete with balanced memory ownership");
    std::istringstream lines(report);
    std::string line;
    unsigned graphics_submissions = 0;
    while (std::getline(lines, line)) {
      if (line.find("\"type\":\"counter\"") == std::string::npos) continue;
      ++graphics_submissions;
      Check(line.find("\"driver_command\":\"draw_pco_sequence\"") != std::string::npos &&
            line.find("\"fragment_pco_opcodes\":") != std::string::npos &&
            line.find("\"atomic32\":") != std::string::npos &&
            line.find("\"cs_invocations\":0,") != std::string::npos,
            "image atomics must be native fragment instructions, not compute dispatch");
    }
    Check(graphics_submissions == 8, "all native graphics image submissions emitted evidence");
    std::cout << "fragment image API " << mode << ": PASS " << checks << " checks\n";
    std::cout << "fragment image API artifacts: " << api.directory << '\n';
#else
    (void)argc; (void)argv;
    std::cout << "fragment image validator: PASS " << checks << " checks\n";
#endif
  } catch (const std::exception &error) {
    std::cerr << "fragment image API FAIL: " << error.what() << '\n';
    return 1;
  }
}
