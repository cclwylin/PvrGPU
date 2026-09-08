// API4 -> native compiler binaries -> independent ComputeShader -> GPU memory.
#include "pvrgpu_systemc_compute_api.h"
#include "pco_compute_image_fixtures.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {
using namespace pvrgpu::stub;
unsigned checks = 0;
void Check(bool ok, const std::string &message) {
  ++checks;
  if (!ok) throw std::runtime_error(message);
}
void Word(std::vector<std::uint8_t> &bytes, std::size_t offset, std::uint32_t value) {
  Check(offset <= bytes.size() && 4 <= bytes.size() - offset, "test word extent");
  std::memcpy(bytes.data() + offset, &value, 4);
}
struct Fixture {
  static constexpr std::size_t image_offset = 64, buffer_offset = 2048;
  unsigned kind, width, height, stride, lanes, invocations;
  std::vector<std::uint8_t> backing;
  std::array<std::uint32_t,4> push{0x12345670,0,0,0};
  pvrgpu_systemc_compute_resource resource{};
  pvrgpu_systemc_compute_binding buffer{};
  std::vector<pvrgpu_systemc_compute_image_binding> images;
  Fixture(unsigned fixture_kind, bool out_of_bounds = false)
      : kind(fixture_kind), lanes(ComputeImagePcoAbi(kind).local_size[0]), backing(8192) {
    width = out_of_bounds ? lanes : 2 * lanes;
    height = out_of_bounds ? 1 : 3;
    stride = width * 4 + 12;
    invocations = 2 * lanes * height;
    for (std::size_t i = 0; i < backing.size(); ++i) backing[i] = i * 17U + 3U;
    for (unsigned y = 0; y < height; ++y)
      for (unsigned x = 0; x < width; ++x)
        Word(backing, image_offset + y * stride + x * 4,
             UINT32_C(0x80000000) + 37U * (y * width + x));
    for (unsigned i = 0; i < invocations; ++i)
      Word(backing, buffer_offset + 4U * i, UINT32_C(0x12340000) + 13U * i);
    resource = {backing.data(), backing.size()};
    const auto a = ComputeImagePcoAbi(kind);
    buffer = {1,0,0, ((a.storage_buffer_read_mask & 1U) ? 1U : 0U) |
                      ((a.storage_buffer_write_mask & 1U) ? 2U : 0U),
              buffer_offset, 4U * invocations * (kind == 32 || kind == 36 ? 4U : 1U)};
    images.push_back({1,0, ((a.image_read_mask & 2U) ? 1U : 0U) |
                           ((a.image_write_mask & 2U) ? 2U : 0U), 1,
                      image_offset, (height - 1U) * stride + width * 4U,
                      width,height,stride,0});
    if (kind == 38) {
      images.push_back(images[0]);
      images.back().slot = 3; images.back().access = 1;
    }
  }
  pvrgpu_systemc_compute_dispatch Dispatch(unsigned mode) {
    const auto a = ComputeImagePcoAbi(kind);
    pvrgpu_systemc_compute_dispatch d{};
    d.version = PVRGPU_SYSTEMC_COMPUTE_API_VERSION;
    static_assert(sizeof(d.abi.stage) == sizeof(a.stage));
    std::memcpy(&d.abi.stage, &a.stage, sizeof(a.stage));
#define COPY(field) d.abi.field = a.field
    COPY(local_invocation_index_start); COPY(local_invocation_index_count);
    COPY(workgroup_id_start); COPY(workgroup_id_count);
    COPY(num_workgroups_start); COPY(num_workgroups_count);
    COPY(storage_buffer_descriptor_start); COPY(storage_buffer_descriptor_count);
    COPY(uniform_buffer_used_mask); COPY(storage_buffer_used_mask);
    COPY(storage_buffer_read_mask); COPY(storage_buffer_write_mask);
    COPY(shared_memory_bytes); COPY(scratch_bytes);
    COPY(shared_memory_descriptor_start); COPY(shared_memory_descriptor_count);
    COPY(image_descriptor_start); COPY(image_descriptor_count);
    COPY(image_used_mask); COPY(image_read_mask); COPY(image_write_mask);
#undef COPY
    for (unsigned axis = 0; axis < 3; ++axis)
      d.abi.local_size[axis] = d.block[axis] = a.local_size[axis];
    d.grid[0] = 2; d.grid[1] = height; d.grid[2] = 1;
    const auto &binary = ComputeImagePcoFixture(kind);
    d.binary = binary.data(); d.binary_size = binary.size();
    d.push_words = push.data(); d.push_word_count = a.stage.push_constant_count;
    d.resources = &resource; d.resource_count = 1;
    d.bindings = &buffer; d.binding_count = kind == 37 ? 0 : 1;
    d.images = images.data(); d.image_count = images.size();
    d.memory_mode = mode;
    return d;
  }
};

void Run(unsigned kind, unsigned mode, bool oob) {
  Fixture fixture(kind, oob);
  auto d = fixture.Dispatch(mode);
  auto expected = fixture.backing;
  unsigned valid = 0;
  for (unsigned y = 0; y < fixture.height; ++y) {
    for (unsigned x = 0; x < 2 * fixture.lanes; ++x) {
      const unsigned linear = y * fixture.width + x;
      const unsigned image_x = kind == 36 ? x - 1U : x;
      const bool inside = image_x < fixture.width;
      valid += inside;
      const auto old = inside ? UINT32_C(0x80000000) + 37U * (y * fixture.width + image_x) : 0U;
      const auto operand = UINT32_C(0x12340000) + 13U * linear;
      if (kind == 32 || kind == 36) {
        const auto offset = Fixture::buffer_offset + linear * 16U;
        Word(expected, offset, old); Word(expected, offset + 4, 0);
        Word(expected, offset + 8, 0); Word(expected, offset + 12, 1);
      } else {
        if (inside)
          Word(expected, Fixture::image_offset + y * fixture.stride + image_x * 4,
               kind == 34 ? old + operand : kind == 37 ? linear + fixture.push[0] : operand);
        if (kind == 34) Word(expected, Fixture::buffer_offset + linear * 4U, old);
      }
    }
  }
  pvrgpu_systemc_compute_stats stats{};
  std::array<char,512> error{};
  const auto status = pvrgpu_systemc_submit_compute(&d, &stats, error.data(), error.size());
  Check(status == 0, "native image " + std::to_string(kind) + ": " + error.data());
  Check(stats.workgroups == 2 * fixture.height && stats.invocations == fixture.invocations,
        "image native workgroup/invocation counts");
  Check(stats.pool_allocations && stats.pool_allocations == stats.pool_releases,
        "image API ownership");
  Check(stats.readback_bytes == fixture.backing.size(), "aliased image/SSBO backing readback must happen once");
  for (std::size_t i = 0; i < expected.size(); ++i)
    Check(fixture.backing[i] == expected[i], "image raw byte / padding / alias mismatch at " + std::to_string(i));
  if (kind == 32 || kind == 36)
    Check(stats.load_instructions == valid && stats.store_instructions == fixture.invocations,
          "OOB image load issued a DMA or lost vector result stores");
  if (kind == 34)
    Check(stats.atomic_instructions == valid, "OOB image atomic performed an RMW");
}

void RejectBoundaries(unsigned mode) {
  for (unsigned bad = 0; bad < 15; ++bad) {
    Fixture fixture(34);
    auto d = fixture.Dispatch(mode);
    const auto original = fixture.backing;
    if (bad == 0) d.abi.image_descriptor_start = 0;
    if (bad == 1) d.abi.image_descriptor_count = 33;
    if (bad == 2) d.abi.image_read_mask = 4;
    if (bad == 3) d.abi.stage.push_constant_start -= 8;
    if (bad == 4) d.images = nullptr;
    if (bad == 5) d.image_count = 33;
    if (bad == 6) fixture.images[0].access = 1;
    if (bad == 7) fixture.images[0].format = 2;
    if (bad == 8) fixture.images[0].offset++;
    if (bad == 9) fixture.images[0].bytes_size = 4;
    if (bad == 10) fixture.images[0].row_stride_bytes = fixture.width * 4 - 4;
    if (bad == 11) fixture.images[0].width = 0;
    if (bad == 12) fixture.images[0].reserved = 1;
    if (bad == 13) fixture.images[0].resource_index = 1;
    if (bad == 14) d.abi.shared_memory_bytes = 4; // private descriptor overlaps image suffix
    pvrgpu_systemc_compute_stats stats{};
    std::array<char,512> error{};
    Check(pvrgpu_systemc_submit_compute(&d, &stats, error.data(), error.size()) != 0 && error[0],
          "invalid image boundary accepted");
    Check(fixture.backing == original && stats.workgroups == 0 && stats.readback_bytes == 0,
          "invalid image boundary ran model or changed caller memory");
  }
#if !defined(_WIN32)
  const auto page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
  auto *mapping = static_cast<std::uint8_t *>(mmap(nullptr, page * 2, PROT_NONE,
                                                 MAP_PRIVATE | MAP_ANON, -1, 0));
  Check(mapping != MAP_FAILED && mprotect(mapping, page, PROT_READ | PROT_WRITE) == 0,
        "old image API guard mapping");
  auto *version = mapping + page - alignof(pvrgpu_systemc_compute_dispatch);
  const std::uint32_t old = 3;
  std::memcpy(version, &old, 4);
  std::array<char,128> error{};
  Check(pvrgpu_systemc_submit_compute(reinterpret_cast<pvrgpu_systemc_compute_dispatch *>(version),
          nullptr, error.data(), error.size()) != 0 && std::string(error.data()).find("version") != std::string::npos,
        "API3 inaccessible image tail was read");
  munmap(mapping, page * 2);
#endif
}
} // namespace

int main(int argc, char **argv) {
  try {
    static_assert(PVRGPU_SYSTEMC_COMPUTE_API_VERSION == 4);
    const unsigned mode = argc > 1 ? static_cast<unsigned>(std::stoul(argv[1])) : 0;
    Check(mode <= 2, "image memory mode");
    for (unsigned epoch = 0; epoch < 2; ++epoch)
      for (unsigned kind = 32; kind <= 38; ++kind) Run(kind, mode, false);
    for (unsigned kind : {32U,33U,34U,36U,37U}) Run(kind, mode, true);
    RejectBoundaries(mode);
    std::cout << "compute image API native: PASS " << checks << " checks\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
