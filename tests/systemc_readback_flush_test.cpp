// What this test is for.
//
// The model used to be deferred to `atexit`, because SystemC elaborates once
// per process and `RunConfiguredModel` built its module set as locals:
//
//     Error: (E529) insert module failed: elaboration done
//
// One process, one simulation, and it had to be last -- so a `glReadPixels`
// between two draws could only ever see the driver's own CPU clear, and every
// dEQP image comparison failed with missing pixels and never a wrong one.
//
// This exercises the arrangement that replaces it: elaborate once, `sc_start()`
// per flush, and hand the pixels back.  Two flushes with different clear
// colours prove both halves -- the second run happens at all, and it is the
// second command's result that comes back, not the first one's.

#include "pvrgpu_systemc_api.h"
#include "pvrgpu_systemc_compute_api.h"
#include "pco_compute_fixtures.h"

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

std::filesystem::path g_test_root;

[[noreturn]] void Fail(const std::string &message) {
  std::fprintf(stderr, "systemc-readback-flush-test: %s\n", message.c_str());
  std::_Exit(EXIT_FAILURE);
}

std::string ReadText(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return {};
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

std::size_t CountOccurrences(const std::string &haystack,
                             const std::string &needle) {
  std::size_t count = 0;
  for (std::size_t at = haystack.find(needle); at != std::string::npos;
       at = haystack.find(needle, at + needle.size())) {
    ++count;
  }
  return count;
}

constexpr std::uint32_t kFloatOne = UINT32_C(0x3f800000);
constexpr std::uint32_t kWidth = 2;
constexpr std::uint32_t kHeight = 2;

// Real native SSBO LD/WDF/ST, interleaved with graphics in the same elaborated
// session. Compute must not consume the last framebuffer or reset its owner.
void SubmitComputeCopy(std::uint32_t seed) {
  std::array<std::uint32_t, 30> input{}, output{};
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = seed + static_cast<std::uint32_t>(i * 17U);
  output.fill(UINT32_C(0xdeadbeef));
  std::array<pvrgpu_systemc_compute_resource, 2> resources{{
      {reinterpret_cast<std::uint8_t *>(input.data()), sizeof(input)},
      {reinterpret_cast<std::uint8_t *>(output.data()), sizeof(output)},
  }};
  std::array<pvrgpu_systemc_compute_binding, 2> bindings{{
      {PVRGPU_SYSTEMC_COMPUTE_STORAGE_BUFFER, 0, 0,
       PVRGPU_SYSTEMC_COMPUTE_ACCESS_READ, 0, sizeof(input)},
      {PVRGPU_SYSTEMC_COMPUTE_STORAGE_BUFFER, 1, 1,
       PVRGPU_SYSTEMC_COMPUTE_ACCESS_WRITE, 0, sizeof(output)},
  }};
  pvrgpu_systemc_compute_dispatch dispatch{};
  dispatch.version = PVRGPU_SYSTEMC_COMPUTE_API_VERSION;
  const auto &binary = pvrgpu::stub::ComputePcoFixture(1);
  dispatch.binary = binary.data();
  dispatch.binary_size = binary.size();
  dispatch.grid[0] = dispatch.grid[1] = dispatch.grid[2] = 1;
  const std::uint32_t local[] = {3, 2, 5};
  for (unsigned i = 0; i < 3; ++i)
    dispatch.block[i] = dispatch.abi.local_size[i] = local[i];
  dispatch.abi.stage.temps = 6;
  dispatch.abi.stage.vertex_inputs = 1;
  dispatch.abi.stage.shareds = dispatch.abi.stage.push_constant_start = 8;
  dispatch.abi.local_invocation_index_count = 1;
  dispatch.abi.storage_buffer_descriptor_count = 2;
  dispatch.abi.storage_buffer_used_mask = 3;
  dispatch.abi.storage_buffer_read_mask = 1;
  dispatch.abi.storage_buffer_write_mask = 2;
  dispatch.resources = resources.data();
  dispatch.resource_count = resources.size();
  dispatch.bindings = bindings.data();
  dispatch.binding_count = bindings.size();
  pvrgpu_systemc_compute_stats stats{};
  std::array<char, 512> error{};
  if (pvrgpu_systemc_submit_compute(&dispatch, &stats, error.data(),
                                    error.size()) != 0)
    Fail(std::string("interleaved native compute failed: ") + error.data());
  if (input != output || stats.workgroups != 1 || stats.invocations != 30 ||
      stats.load_instructions != 30 || stats.store_instructions != 30 ||
      stats.readback_bytes != sizeof(output) || stats.pool_allocations == 0 ||
      stats.pool_allocations != stats.pool_releases)
    Fail("interleaved native compute result/accounting mismatch");
}

// Submits one opaque clear of the given colour and returns the pixels the
// model left in DRAM for it.
std::array<std::uint8_t, kWidth * kHeight * 4>
SubmitAndFlushClear(pvrgpu_systemc_submit_info *info,
                    pvrgpu_systemc_driver_command *clear,
                    const char *case_name, std::uint32_t red,
                    std::uint32_t green, std::uint32_t blue) {
  clear->case_name = case_name;
  clear->clear_color_bits[0] = red;
  clear->clear_color_bits[1] = green;
  clear->clear_color_bits[2] = blue;
  clear->clear_color_bits[3] = kFloatOne;

  std::array<char, 512> error{};
  if (pvrgpu_systemc_submit_driver_command(info, error.data(), error.size()) !=
      0) {
    Fail(std::string("submit failed: ") + error.data());
  }

  std::array<std::uint8_t, kWidth * kHeight * 4> pixels{};
  pvrgpu_systemc_readback_info readback{};
  readback.version = PVRGPU_SYSTEMC_API_VERSION;
  readback.width = kWidth;
  readback.height = kHeight;
  readback.bytes_per_pixel = 4;
  readback.pixels = pixels.data();
  readback.pixels_size = pixels.size();
  error.fill(0);
  if (pvrgpu_systemc_flush_readback(&readback, error.data(), error.size()) !=
      0) {
    Fail(std::string("flush failed: ") + error.data());
  }
  if (readback.pixels_written != 1)
    Fail("flush published no pixels for a submitted clear");
  return pixels;
}

void RequireOpaqueColor(
    const std::array<std::uint8_t, kWidth * kHeight * 4> &pixels,
    std::uint8_t red, std::uint8_t green, std::uint8_t blue,
    const std::string &what) {
  for (std::uint32_t pixel = 0; pixel < kWidth * kHeight; ++pixel) {
    const std::uint8_t *texel = pixels.data() + pixel * 4U;
    if (texel[0] != red || texel[1] != green || texel[2] != blue ||
        texel[3] != 255) {
      Fail(what + ": pixel " + std::to_string(pixel) + " is " +
           std::to_string(texel[0]) + ',' + std::to_string(texel[1]) + ',' +
           std::to_string(texel[2]) + ',' + std::to_string(texel[3]));
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  const auto nonce =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  g_test_root = std::filesystem::temp_directory_path() /
                ("pvrgpu-systemc-readback-test-" + std::to_string(nonce));
  std::filesystem::create_directories(g_test_root / "out");

  const std::string jsonl_text = (g_test_root / "model.jsonl").string();
  const std::string stderr_text = (g_test_root / "model.stderr.log").string();
  const std::string outdir_text = (g_test_root / "out").string();

  // Run both elaboration orders in separate CTest processes.
  if (argc == 2 && std::string(argv[1]) == "compute-first")
    SubmitComputeCopy(100);

  pvrgpu_systemc_driver_command clear{};
  clear.version = PVRGPU_SYSTEMC_API_VERSION;
  clear.schema = "pvrgpu.driver-command.v1";
  clear.producer = "pvrgpu-gallium-driver";
  clear.command = "clear_color";
  clear.format = "PIPE_FORMAT_R8G8B8A8_UNORM";
  clear.frame = 1;
  clear.width = kWidth;
  clear.height = kHeight;

  pvrgpu_systemc_submit_info info{};
  info.version = PVRGPU_SYSTEMC_API_VERSION;
  info.command = &clear;
  info.jsonl_path = jsonl_text.c_str();
  info.stderr_path = stderr_text.c_str();
  info.outdir = outdir_text.c_str();
  info.memory_mode = "direct";

  // A flush before anything is submitted is not an error; it simply has no
  // pixels to hand over, and the caller keeps whatever it already had.
  std::array<std::uint8_t, kWidth * kHeight * 4> untouched{};
  untouched.fill(UINT8_C(0x5a));
  pvrgpu_systemc_readback_info idle{};
  idle.version = PVRGPU_SYSTEMC_API_VERSION;
  idle.width = kWidth;
  idle.height = kHeight;
  idle.bytes_per_pixel = 4;
  idle.pixels = untouched.data();
  idle.pixels_size = untouched.size();
  std::array<char, 512> error{};
  if (pvrgpu_systemc_flush_readback(&idle, error.data(), error.size()) != 0)
    Fail(std::string("idle flush failed: ") + error.data());
  if (idle.pixels_written != 0)
    Fail("idle flush claimed to publish pixels");
  for (const std::uint8_t byte : untouched) {
    if (byte != UINT8_C(0x5a))
      Fail("idle flush overwrote the caller's pixels");
  }

  const auto first = SubmitAndFlushClear(&info, &clear,
                                         "systemc-readback-flush-test-red",
                                         kFloatOne, 0, 0);
  RequireOpaqueColor(first, 255, 0, 0, "first readback");
  SubmitComputeCopy(200);

  // The second submission is what the old arrangement could not do at all: the
  // model has already run once, and a second elaboration is refused.  It only
  // works because the module set outlived the first flush.
  const auto second = SubmitAndFlushClear(&info, &clear,
                                          "systemc-readback-flush-test-green",
                                          0, kFloatOne, 0);
  RequireOpaqueColor(second, 0, 255, 0, "second readback");

  /*
   * A clear only proves the model ran.  This drives a real draw through the
   * whole pipeline -- VDM, vertex fetch, shading, clip, tile, ISP, PBE, DRAM
   * readback -- on a third flush, which is the case that used to be
   * impossible: the modules are the ones elaborated for the first clear.  The
   * triangle covers the lower-left half of the surface on a blue clear, so a
   * red pixel there is the model's own rasterization coming back.
   */
  pvrgpu_systemc_driver_command triangle{};
  triangle.version = PVRGPU_SYSTEMC_API_VERSION;
  triangle.schema = "pvrgpu.driver-command.v1";
  triangle.producer = "pvrgpu-gallium-driver";
  triangle.command = "draw_triangle";
  triangle.case_name = "systemc-readback-flush-test-triangle";
  triangle.format = "PIPE_FORMAT_R8G8B8A8_UNORM";
  triangle.frame = 1;
  triangle.width = kWidth;
  triangle.height = kHeight;
  triangle.clear_color_bits[2] = kFloatOne;
  triangle.clear_color_bits[3] = kFloatOne;
  const std::uint32_t kFloatMinusOne = UINT32_C(0xbf800000);
  const std::uint32_t corners[3][2] = {
      {kFloatMinusOne, kFloatMinusOne},
      {kFloatOne, kFloatMinusOne},
      {kFloatMinusOne, kFloatOne},
  };
  for (std::size_t vertex = 0; vertex < 3; ++vertex) {
    triangle.vertex_bits[vertex][0] = corners[vertex][0];
    triangle.vertex_bits[vertex][1] = corners[vertex][1];
  }
  triangle.fragment_color_bits[0] = kFloatOne;
  triangle.fragment_color_bits[3] = kFloatOne;

  info.command = &triangle;
  error.fill(0);
  if (pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size()) !=
      0) {
    Fail(std::string("triangle submit failed: ") + error.data());
  }
  // This dispatch must flush the queued triangle before using the shared
  // memory service, and later graphics readback must still return that draw.
  SubmitComputeCopy(300);
  std::array<std::uint8_t, kWidth * kHeight * 4> drawn{};
  pvrgpu_systemc_readback_info drawn_readback{};
  drawn_readback.version = PVRGPU_SYSTEMC_API_VERSION;
  drawn_readback.width = kWidth;
  drawn_readback.height = kHeight;
  drawn_readback.bytes_per_pixel = 4;
  drawn_readback.pixels = drawn.data();
  drawn_readback.pixels_size = drawn.size();
  error.fill(0);
  if (pvrgpu_systemc_flush_readback(&drawn_readback, error.data(),
                                    error.size()) != 0) {
    Fail(std::string("triangle flush failed: ") + error.data());
  }
  if (drawn_readback.pixels_written != 1)
    Fail("triangle flush published no pixels");
  bool any_red = false;
  bool any_clear = false;
  for (std::uint32_t pixel = 0; pixel < kWidth * kHeight; ++pixel) {
    const std::uint8_t *texel = drawn.data() + pixel * 4U;
    if (texel[3] != 255)
      Fail("triangle readback is not opaque");
    if (texel[0] == 255 && texel[1] == 0 && texel[2] == 0)
      any_red = true;
    else if (texel[0] == 0 && texel[1] == 0 && texel[2] == 255)
      any_clear = true;
    else
      Fail("triangle readback has a pixel that is neither the draw nor the "
           "clear");
  }
  if (!any_red)
    Fail("a draw on a later flush produced no rasterized pixels");
  if (!any_clear)
    Fail("a draw on a later flush covered the whole surface");
  SubmitComputeCopy(400);

  info.command = &clear;

  /*
   * A readback with nothing new submitted hands back what the last one
   * produced rather than running it again.  A pass writing several colour
   * attachments is read one attachment at a time and the first of those reads
   * is what runs it, so the later ones have to find the same result waiting.
   * That the submission is not re-run is what the JSONL record count below
   * establishes: three flushes, three record sets.
   */
  std::array<std::uint8_t, kWidth * kHeight * 4> repeat{};
  repeat.fill(UINT8_C(0x17));
  pvrgpu_systemc_readback_info again{};
  again.version = PVRGPU_SYSTEMC_API_VERSION;
  again.width = kWidth;
  again.height = kHeight;
  again.bytes_per_pixel = 4;
  again.pixels = repeat.data();
  again.pixels_size = repeat.size();
  error.fill(0);
  if (pvrgpu_systemc_flush_readback(&again, error.data(), error.size()) != 0)
    Fail(std::string("repeat flush failed: ") + error.data());
  if (again.pixels_written != 1)
    Fail("repeat readback did not hand back the executed submission's pixels");
  if (repeat != drawn)
    Fail("repeat readback published something other than the last result");

  /* An attachment the pass never wrote publishes nothing rather than the
   * first one's pixels. */
  std::array<std::uint8_t, kWidth * kHeight * 4> absent{};
  absent.fill(UINT8_C(0x17));
  pvrgpu_systemc_readback_info second_target{};
  second_target.version = PVRGPU_SYSTEMC_API_VERSION;
  second_target.width = kWidth;
  second_target.height = kHeight;
  second_target.bytes_per_pixel = 4;
  second_target.attachment = 1;
  second_target.pixels = absent.data();
  second_target.pixels_size = absent.size();
  error.fill(0);
  if (pvrgpu_systemc_flush_readback(&second_target, error.data(),
                                    error.size()) != 0) {
    Fail(std::string("second-attachment readback failed: ") + error.data());
  }
  if (second_target.pixels_written != 0)
    Fail("a colour attachment the pass never wrote published pixels");
  if (absent != std::array<std::uint8_t, kWidth * kHeight * 4>{
                   [] {
                     std::array<std::uint8_t, kWidth * kHeight * 4> filled{};
                     filled.fill(UINT8_C(0x17));
                     return filled;
                   }()}) {
    Fail("an unwritten attachment's readback touched the destination");
  }

  // Each flush reports for itself, so the stream carries one complete record
  // set per flush rather than a single set held back until exit.
  const std::string jsonl = ReadText(g_test_root / "model.jsonl");
  if (CountOccurrences(jsonl, "\"type\":\"done\"") != 3 ||
      CountOccurrences(jsonl, "\"type\":\"hello\"") != 3 ||
      CountOccurrences(jsonl, "\"type\":\"counter\"") != 3) {
    Fail("three flushes did not leave three record sets in the JSONL stream");
  }

  std::error_code remove_error;
  std::filesystem::remove_all(g_test_root, remove_error);
  std::puts("systemc-readback-flush-test: PASS");
  return EXIT_SUCCESS;
}
