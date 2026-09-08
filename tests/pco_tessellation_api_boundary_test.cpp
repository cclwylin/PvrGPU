// SPDX-License-Identifier: MIT
// API-envelope and native-decoder tests only. Every submission is rejected at
// a named validation gate; no draw is queued and no SystemC model is run.
#include "pvrgpu_systemc_api.h"
#include "pco_tessellation_compiler_fixtures.h"
#include "pco_geometry_compiler_fixtures.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {
unsigned checks = 0;
constexpr const char *kLateGate = "unsupported: blend";
constexpr const char *kExecutable = "tessellation executable size/pointer";
constexpr const char *kExtent = "tessellation patch vertex extent";
constexpr const char *kLayout = "tessellation patch storage layout";
constexpr const char *kAbi = "tessellation native stage register/descriptor ABI";
constexpr const char *kLink = "tessellation stage/topology/input linkage";
constexpr const char *kUboEntry = "uniform buffer stage/index/bytes is invalid";
void Check(bool value, const std::string &message) {
  ++checks;
  if (!value) throw std::runtime_error(message);
}

struct Fixture {
  std::vector<std::uint8_t> vs = Tess0VertexPco(), tcs = kTess0tcs,
      tes = kTess0tes, fs = Tess0FragmentPco();
  std::vector<std::uint8_t> gs = pvrgpu::stub::GeometryCompilerFixture(0);
  std::array<std::array<std::uint32_t, 256>, 5> shared{};
  std::array<pvrgpu_systemc_pco_uniform_buffer, 75> ubos{};
  std::vector<std::uint8_t> uniform_bytes =
      std::vector<std::uint8_t>(PVRGPU_SYSTEMC_MAX_UNIFORM_BUFFER_BYTES, 0x5a);
  pvrgpu_systemc_tessellation tess{};
  pvrgpu_systemc_driver_command draw{};
  Fixture() {
    tess.control_pco = tcs.data(); tess.control_pco_size = tcs.size();
    tess.evaluation_pco = tes.data(); tess.evaluation_pco_size = tes.size();
    tess.control_shared = shared[3].data(); tess.control_shared_count = 8;
    tess.evaluation_shared = shared[4].data(); tess.evaluation_shared_count = 4;
    tess.control_abi = {6,3,0,0,8,8,0,0,8,0};
    tess.evaluation_abi = {2,5,4,0,4,4,0,0,4,0};
    tess.input_vertices = tess.output_vertices = tess.vertices_per_instance = 1;
    tess.input_stride_dwords = tess.output_vertex_stride_dwords = 4;
    tess.per_vertex_offset_dwords = 6; tess.patch_stride_dwords = 10;
    draw.version = PVRGPU_SYSTEMC_API_VERSION;
    draw.command = "draw_pco_triangles";
    draw.case_name = "native.tessellation.api.boundary";
    draw.format = "PIPE_FORMAT_R8G8B8A8_UNORM";
    draw.frame = 1;
    draw.framebuffer_width = draw.width = draw.framebuffer_height = draw.height = 16;
    draw.primitive_mode = 14;
    draw.vertex_count = draw.instance_count = 1;
    draw.vertex_pco = vs.data(); draw.vertex_pco_size = vs.size();
    draw.fragment_pco = fs.data(); draw.fragment_pco_size = fs.size();
    draw.vertex_pco_abi = {0,0,4,0,0,0,0,0,0,0};
    draw.fragment_pco_abi = {1,0,0,4,0,0,0,0,0,0};
    draw.position_output_count = draw.varying_output_start = 4;
    draw.fragment_position_count = draw.fragment_varying_start = 4;
    draw.fragment_output_mask[0] = 15;
    draw.viewport_scale_bits[0] = draw.viewport_scale_bits[1] = 0x41000000;
    draw.viewport_scale_bits[2] = 0x3f000000;
    std::memcpy(draw.viewport_translate_bits, draw.viewport_scale_bits, sizeof(draw.viewport_scale_bits));
    draw.half_pixel_center = draw.depth_clip_near = draw.depth_clip_far = 1;
    draw.sample_mask = UINT32_MAX; draw.color_mask = 15;
    draw.blend_source_rgb_factor = draw.blend_source_alpha_factor = PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE;
    draw.color_attachment_source_command_index = draw.depth_attachment_source_command_index = PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR;
    draw.tessellation = &tess;
    draw.blend_enable = 2;
  }
  Fixture(const Fixture &) = delete;
  Fixture &operator=(const Fixture &) = delete;
  pvrgpu_systemc_pco_stage_abi &Abi(unsigned stage) {
    switch (stage) {
    case 0: return draw.vertex_pco_abi;
    case 1: return draw.fragment_pco_abi;
    case 2: return draw.geometry_pco_abi;
    case 3: return tess.control_abi;
    default: return tess.evaluation_abi;
    }
  }
  void SharedCount(unsigned stage, std::uint32_t count) {
    Abi(stage).shareds = count;
    switch (stage) {
    case 0: draw.vertex_shared = shared[0].data(); draw.vertex_shared_count = count; break;
    case 1: draw.fragment_shared = shared[1].data(); draw.fragment_shared_count = count; break;
    case 2: draw.geometry_shared = shared[2].data(); draw.geometry_shared_count = count; break;
    case 3: tess.control_shared_count = count; break;
    default: tess.evaluation_shared_count = count; break;
    }
  }
  void Geometry() {
    // A separate real GS envelope exercises public UBO stage 2. Combining GS
    // with TCS/TES is deliberately unsupported, not a legal five-stage draw.
    draw.tessellation = nullptr; draw.primitive_mode = 0;
    draw.geometry_pco = gs.data(); draw.geometry_pco_size = gs.size();
    draw.geometry_shared = shared[2].data(); draw.geometry_shared_count = 4;
    draw.geometry_pco_abi = {4,2,8,0,4,4,0,0,4,0};
    draw.geometry_input_primitive_vertices = draw.geometry_max_vertices = draw.geometry_invocations = 1;
    draw.geometry_input_stride_dwords = 4; draw.geometry_vertices_per_instance = 1;
  }
  void DescriptorCount(unsigned stage, std::uint32_t count) {
    auto &a = Abi(stage);
    const auto prefix = stage == 3 ? 8u : stage >= 2 ? 4u : 0u;
    a.uniform_buffer_descriptor_start = prefix;
    a.uniform_buffer_descriptor_count = count;
    a.push_constant_start = prefix + count * 4;
    a.push_constant_count = 0;
    SharedCount(stage, a.push_constant_start);
  }
  void AddUbo(unsigned stage, unsigned block, std::size_t bytes) {
    auto &buffer = ubos[draw.uniform_buffer_count++];
    buffer = {stage, block, uniform_bytes.data(), bytes};
    draw.uniform_buffers = ubos.data();
    const auto word = Abi(stage).uniform_buffer_descriptor_start + block * 4;
    shared[stage][word + 2] = static_cast<std::uint32_t>(bytes);
  }
};

struct Submission {
  std::filesystem::path root;
  std::string jsonl, outdir, stderr_path;
  pvrgpu_systemc_driver_command sequence{};
  pvrgpu_systemc_submit_info info{};
  explicit Submission(std::filesystem::path path) : root(std::move(path)),
      jsonl((root / "model.jsonl").string()), outdir((root / "out").string()),
      stderr_path((root / "model.stderr.log").string()) {
    sequence.version = PVRGPU_SYSTEMC_API_VERSION;
    sequence.command = "draw_pco_sequence";
    sequence.case_name = "native.tessellation.api.boundary";
    sequence.format = "PIPE_FORMAT_R8G8B8A8_UNORM";
    sequence.frame = 1;
    sequence.framebuffer_width = sequence.width = sequence.framebuffer_height = sequence.height = 16;
    info.version = PVRGPU_SYSTEMC_API_VERSION; info.command = &sequence;
    info.jsonl_path = jsonl.c_str(); info.outdir = outdir.c_str();
    info.stderr_path = stderr_path.c_str(); info.memory_mode = "cache";
    info.submission_generation = 1;
  }
  void Call(const pvrgpu_systemc_submit_info &input, const char *expected, const std::string &label) {
    std::array<char, 1024> error{};
    const int status = pvrgpu_systemc_submit_driver_command(&input, error.data(), error.size());
    Check(status == 2, label + " must reject before model submission, status=" + std::to_string(status));
    Check(std::string(error.data()).find(expected) != std::string::npos,
          label + " rejected at wrong gate: " + error.data());
    Check(!std::filesystem::exists(root), label + " must not create model output");
  }
  void Reject(const pvrgpu_systemc_driver_command &draw, const char *expected, const std::string &label) {
    sequence.pco_sequence_commands = &draw; sequence.pco_sequence_command_count = 1;
    Call(info, expected, label);
  }
  void AfterUbo(const pvrgpu_systemc_driver_command &draw, const char *expected, const std::string &label) {
    // UBO entries and canonical descriptors are validated AFTER the first
    // draw's blend gate. A second, blend-invalid sentinel proves the first
    // completed those checks without letting any sequence reach the model.
    Fixture sentinel;
    std::array<pvrgpu_systemc_driver_command, 2> draws = {draw, sentinel.draw};
    draws[0].blend_enable = 0;
    sequence.pco_sequence_commands = draws.data(); sequence.pco_sequence_command_count = 2;
    Call(info, expected, label);
  }
};

// The bytes end exactly at a no-access page. Pointer/count validation must
// precede reading a versioned tail or a descriptor past its stated extent.
class GuardedBytes {
 public:
  explicit GuardedBytes(std::size_t size) : size_(size) {
#if defined(_WIN32)
    SYSTEM_INFO info{}; GetSystemInfo(&info); page_ = info.dwPageSize;
    mapping_ = VirtualAlloc(nullptr, page_ * 2, MEM_COMMIT | MEM_RESERVE, PAGE_NOACCESS);
    DWORD previous = 0;
    Check(mapping_ && VirtualProtect(mapping_, page_, PAGE_READWRITE, &previous), "allocate guard pages");
#else
    const auto page = sysconf(_SC_PAGESIZE);
    Check(page > 0, "query guard page size"); page_ = static_cast<std::size_t>(page);
    mapping_ = mmap(nullptr, page_ * 2, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
    Check(mapping_ != MAP_FAILED && mprotect(mapping_, page_, PROT_READ | PROT_WRITE) == 0,
          "allocate guard pages");
#endif
    Check(size_ <= page_, "guarded payload fits one page");
    bytes_ = static_cast<std::uint8_t *>(mapping_) + page_ - size_;
    std::memset(bytes_, 0, size_);
  }
  ~GuardedBytes() {
#if defined(_WIN32)
    VirtualFree(mapping_, 0, MEM_RELEASE);
#else
    munmap(mapping_, page_ * 2);
#endif
  }
  GuardedBytes(const GuardedBytes &) = delete;
  GuardedBytes &operator=(const GuardedBytes &) = delete;
  std::uint8_t *data() { return bytes_; }
 private:
  void *mapping_ = nullptr;
  std::uint8_t *bytes_ = nullptr;
  std::size_t page_ = 0, size_ = 0;
};

void VerifyVersions(Submission &submit) {
  static_assert(PVRGPU_SYSTEMC_API_VERSION == 26);
  constexpr auto previous_size = offsetof(pvrgpu_systemc_driver_command, tessellation);
  static_assert(previous_size % alignof(pvrgpu_systemc_driver_command) == 0);
  GuardedBytes previous(previous_size);
  const std::uint32_t old_version = 24;
  std::memcpy(previous.data(), &old_version, sizeof(old_version));
  const auto &old = *reinterpret_cast<const pvrgpu_systemc_driver_command *>(previous.data());
  auto info = submit.info; info.command = &old;
  submit.Call(info, "command version", "API24 short top-level command before API25 tail");
  submit.Reject(old, "version=24 expected=26", "API24 short nested command before API25 tail");
  GuardedBytes api25(offsetof(pvrgpu_systemc_driver_command, stream_output));
  const std::uint32_t api25_version = 25;
  std::memcpy(api25.data(), &api25_version, sizeof(api25_version));
  const auto &old25 = *reinterpret_cast<const pvrgpu_systemc_driver_command *>(api25.data());
  info.command = &old25;
  submit.Call(info, "command version", "API25 short command before API26 stream output tail");
  submit.Reject(old25, "version=25 expected=26", "API25 short nested command before API26 tail");
  Fixture fixture;
  auto full = fixture.draw; full.version = 24;
  submit.Reject(full, "version=24 expected=26", "full allocation with old nested version");
  info.command = &full;
  submit.Call(info, "command version", "full allocation with old top-level version");
  for (auto version : {0u,24u,25u,27u,UINT32_MAX}) {
    info = submit.info; info.version = version;
    submit.Call(info, "submit version", "invalid submit-info version " + std::to_string(version));
    full.version = version;
    submit.Reject(full, "version=", "invalid nested version " + std::to_string(version));
  }
  GuardedBytes complete(sizeof(pvrgpu_systemc_driver_command));
  std::memcpy(complete.data(), &fixture.draw, sizeof(fixture.draw));
  submit.Reject(*reinterpret_cast<const pvrgpu_systemc_driver_command *>(complete.data()),
                kLateGate, "full API25 command reads no byte beyond its tail");
  GuardedBytes payload(sizeof(pvrgpu_systemc_tessellation));
  std::memcpy(payload.data(), &fixture.tess, sizeof(fixture.tess));
  full = fixture.draw;
  full.tessellation = reinterpret_cast<const pvrgpu_systemc_tessellation *>(payload.data());
  submit.Reject(full, kLateGate, "complete Tessellation payload ends at a guard page");
}

template <typename Mutate>
void Probe(Submission &submit, Mutate mutate, const char *expected, const std::string &name) {
  Fixture f; mutate(f); submit.Reject(f.draw, expected, name);
}
void VerifyMetadata(Submission &submit) {
  for (bool evaluation : {false,true}) {
    const std::string name = evaluation ? "TES " : "TCS ";
    Probe(submit, [&](auto &f) { (evaluation ? f.tess.evaluation_pco : f.tess.control_pco) = nullptr; },
          kExecutable, name + "missing binary pointer");
    for (std::size_t size : {std::size_t{0},std::size_t{(1u << 24) + 1},SIZE_MAX})
      Probe(submit, [&](auto &f) { (evaluation ? f.tess.evaluation_pco_size : f.tess.control_pco_size) = size; },
            kExecutable, name + "binary size " + std::to_string(size));
  }
  // These are storage-envelope limits, not assertions that padded bytes are
  // executable. The separate decoder tests below use real compiler binaries.
  std::vector<std::uint8_t> largest_binary(1u << 24);
  for (bool evaluation : {false,true})
    Probe(submit, [&](auto &f) {
      (evaluation ? f.tess.evaluation_pco : f.tess.control_pco) = largest_binary.data();
      (evaluation ? f.tess.evaluation_pco_size : f.tess.control_pco_size) = largest_binary.size();
    }, kLateGate, "16 MiB binary storage-envelope maximum");
  for (auto member : {&pvrgpu_systemc_tessellation::input_vertices,
                     &pvrgpu_systemc_tessellation::output_vertices}) {
    for (auto value : {0u,33u,UINT32_MAX})
      Probe(submit, [&](auto &f) { f.tess.*member = value; }, kExtent, "patch vertex bound");
    for (auto value : {1u,32u})
      Probe(submit, [&](auto &f) {
        f.tess.*member = value;
        f.tess.patch_stride_dwords = f.tess.per_vertex_offset_dwords + f.tess.output_vertices * 4;
      }, kLateGate, "legal patch vertex endpoint");
  }
  Probe(submit, [](auto &f) { f.tess.vertices_per_instance = 0; }, kExtent, "zero per-instance extent");
  for (auto value : {0u,3u,65u,UINT32_MAX})
    Probe(submit, [&](auto &f) { f.tess.input_stride_dwords = value; }, kLayout, "VS patch stride bound");
  for (auto value : {4u,64u})
    Probe(submit, [&](auto &f) { f.tess.input_stride_dwords = f.draw.vertex_pco_abi.vertex_outputs = value; },
          kLateGate, "legal VS patch stride endpoint");
  for (auto value : {65u,UINT32_MAX})
    Probe(submit, [&](auto &f) { f.tess.output_vertex_stride_dwords = value; }, kLayout, "TCS vertex stride bound");
  for (auto value : {0u,1u,64u})
    Probe(submit, [&](auto &f) { f.tess.output_vertex_stride_dwords = value; f.tess.patch_stride_dwords = 6 + value; },
          kLateGate, "TCS stride endpoint including levels-only patch");
  for (auto value : {0u,5u,135u,UINT32_MAX})
    Probe(submit, [&](auto &f) { f.tess.per_vertex_offset_dwords = value; }, kLayout, "patch-varying prefix bound");
  for (auto value : {6u,134u})
    Probe(submit, [&](auto &f) { f.tess.per_vertex_offset_dwords = value; f.tess.patch_stride_dwords = value + 4; },
          kLateGate, "legal patch-varying prefix endpoint");
  for (auto value : {0u,9u,11u,UINT32_MAX})
    Probe(submit, [&](auto &f) { f.tess.patch_stride_dwords = value; }, kLayout, "exact total patch DWORD extent");
  Probe(submit, [](auto &f) {
    f.tess.output_vertices = 32; f.tess.output_vertex_stride_dwords = 64;
    f.tess.per_vertex_offset_dwords = 134; f.tess.patch_stride_dwords = 134 + 32 * 64;
  }, kLateGate, "maximum complete patch allocation");
  for (auto member : {&pvrgpu_systemc_tessellation::domain,&pvrgpu_systemc_tessellation::spacing,
                     &pvrgpu_systemc_tessellation::clockwise,&pvrgpu_systemc_tessellation::point_mode}) {
    const auto maximum = member == &pvrgpu_systemc_tessellation::domain ||
        member == &pvrgpu_systemc_tessellation::spacing ? 2u : 1u;
    for (auto value : {maximum + 1,UINT32_MAX})
      Probe(submit, [&](auto &f) { f.tess.*member = value; }, "domain/spacing/winding/point mode", "invalid enum endpoint");
    for (unsigned value = 0; value <= maximum; ++value)
      Probe(submit, [&](auto &f) { f.tess.*member = value; }, kLateGate, "legal Tessellation enum");
  }
  for (auto value : {0u,1u,UINT32_MAX})
    Probe(submit, [&](auto &f) { f.tess.control_barrier_count = value; }, kLateGate,
          "barrier count is compiler diagnostic metadata, not patch allocation");
  Probe(submit, [](auto &f) { f.draw.primitive_mode = 4; }, kLink, "Tessellation requires PATCHES");
  Probe(submit, [](auto &f) { f.draw.geometry_pco = f.gs.data(); f.draw.geometry_pco_size = f.gs.size(); },
        kLink, "GS plus Tessellation is explicitly unsupported");
  Probe(submit, [](auto &f) { f.draw.vertex_pco_abi.vertex_outputs = 8; }, kLink, "VS/TCS input stride linkage");
  Probe(submit, [](auto &f) { f.tess.vertices_per_instance = 2; }, kLink, "incomplete transported instance span");
  Probe(submit, [](auto &f) { f.draw.render_target_count = 2; }, "geometry MRT", "Tessellation MRT fail-closed");
}

void VerifyStageAbis(Submission &submit) {
  for (unsigned stage : {3u,4u}) {
    const auto prefix = stage == 3 ? 8u : 4u;
    const auto vi = stage == 3 ? 3u : 5u;
    for (auto value : {0u,256u})
      Probe(submit, [&](auto &f) { f.Abi(stage).temps = value; }, kLateGate, "legal TEMP endpoint");
    Probe(submit, [&](auto &f) { f.Abi(stage).temps = 257; }, kAbi, "TEMP exceeds 256");
    for (auto value : {0u,vi - 1,vi + 1,UINT32_MAX})
      Probe(submit, [&](auto &f) { f.Abi(stage).vertex_inputs = value; }, kAbi, "exact stage-specific VI count");
    for (auto value : stage == 3 ? std::vector<unsigned>{1,4,64,UINT32_MAX} : std::vector<unsigned>{0,3,65,UINT32_MAX})
      Probe(submit, [&](auto &f) { f.Abi(stage).vertex_outputs = value; }, kAbi, "stage-specific VO bounds");
    if (stage == 4)
      for (auto value : {4u,64u})
        Probe(submit, [&](auto &f) { f.Abi(stage).vertex_outputs = value; f.draw.varying_output_count = value - 4; },
              kLateGate, "TES output bank endpoint with matching raster linkage");
    for (auto member : {&pvrgpu_systemc_pco_stage_abi::coefficients,&pvrgpu_systemc_pco_stage_abi::entry_offset})
      for (auto value : {1u,UINT32_MAX})
        Probe(submit, [&](auto &f) { f.Abi(stage).*member = value; }, kAbi, "forbidden stage ABI field");
    Probe(submit, [&](auto &f) { (stage == 3 ? f.tess.control_shared : f.tess.evaluation_shared) = nullptr; },
          kAbi, "missing shared storage");
    for (auto value : {0u,prefix - 1,prefix + 1,UINT32_MAX})
      Probe(submit, [&](auto &f) { (stage == 3 ? f.tess.control_shared_count : f.tess.evaluation_shared_count) = value; },
            kAbi, "SHARED pointer/count/ABI disagreement");
    for (auto value : {prefix - 1,257u,UINT32_MAX})
      Probe(submit, [&](auto &f) { f.SharedCount(stage, value); }, kAbi, "SHARED bounded prefix and maximum");
    for (auto value : {0u,prefix - 1,prefix + 1,UINT32_MAX})
      Probe(submit, [&](auto &f) { f.Abi(stage).uniform_buffer_descriptor_start = value; }, kAbi, "UBO descriptor start follows patch descriptors");
    for (auto value : {16u,UINT32_MAX})
      Probe(submit, [&](auto &f) { f.Abi(stage).uniform_buffer_descriptor_count = value; }, kAbi, "UBO stage count bound");
    Probe(submit, [&](auto &f) { f.Abi(stage).push_constant_start = prefix - 1; }, kAbi, "push prefix overlaps patch descriptor");
    Probe(submit, [&](auto &f) { f.Abi(stage).push_constant_count = 1; }, kAbi, "push suffix exceeds SHARED");
    Probe(submit, [&](auto &f) { f.Abi(stage).push_constant_start = UINT32_MAX; f.Abi(stage).push_constant_count = 1; },
          kAbi, "push arithmetic cannot wrap");
    for (auto count : {0u,1u,15u})
      for (bool maximum_push : {false,true})
        Probe(submit, [&](auto &f) {
          f.DescriptorCount(stage, count);
          if (maximum_push) {
            f.Abi(stage).push_constant_count = 256 - f.Abi(stage).push_constant_start;
            f.SharedCount(stage, 256);
          }
        }, kLateGate, "legal UBO descriptors plus exact CB0 suffix");
    for (unsigned word = 0; word < prefix; ++word)
      Probe(submit, [&](auto &f) { f.shared[stage][word] = UINT32_MAX; }, "must be unrelocated", "every patch descriptor word must be zero");
    GuardedBytes words(prefix * sizeof(std::uint32_t));
    Probe(submit, [&](auto &f) {
      (stage == 3 ? f.tess.control_shared : f.tess.evaluation_shared) = reinterpret_cast<const std::uint32_t *>(words.data());
    }, kLateGate, "minimal stage SHARED allocation ends at a guard page");
  }
}

void VerifyDrawExtents(Submission &submit) {
  static_assert(PVRGPU_SYSTEMC_MAX_TESSELLATION_INPUT_VERTICES == 131072);
  static_assert(PVRGPU_SYSTEMC_MAX_TESSELLATION_PATCHES == 4096);
  for (auto occurrences : {4096u,131072u})
    Probe(submit, [&](auto &f) {
      f.draw.vertex_count = f.tess.vertices_per_instance = occurrences;
      f.tess.input_vertices = occurrences == 4096 ? 1 : 32;
    }, kLateGate, "exact 4096 patch maximum");
  Probe(submit, [](auto &f) {
    f.draw.vertex_count = 131072; f.tess.vertices_per_instance = f.tess.input_vertices = 32;
  }, kLateGate, "4096 flattened instances at maximum input occurrence count");
  for (auto occurrences : {4097u,131072u})
    Probe(submit, [&](auto &f) { f.draw.vertex_count = f.tess.vertices_per_instance = occurrences; },
          "patch count exceeds", "patch budget independent of input budget");
  for (auto occurrences : {131073u,UINT32_MAX})
    Probe(submit, [&](auto &f) {
      f.draw.vertex_count = f.tess.vertices_per_instance = occurrences; f.tess.input_vertices = 32;
    }, "input occurrence count exceeds", "input occurrence allocation bound");
  // GL ignores an incomplete patch at each instance tail. It is NOT an
  // invalid API envelope, including when the draw has no complete patch.
  for (auto occurrences : {1u,31u,33u,63u})
    Probe(submit, [&](auto &f) {
      f.draw.vertex_count = f.tess.vertices_per_instance = occurrences; f.tess.input_vertices = 32;
    }, kLateGate, "legal incomplete patch tail " + std::to_string(occurrences));
  Probe(submit, [](auto &f) {
    f.draw.vertex_count = 62; f.tess.vertices_per_instance = 31; f.tess.input_vertices = 32;
  }, kLateGate, "incomplete patches do not combine across two instances");
  std::vector<std::uint8_t> indices(131073, 0);
  for (auto occurrences : {31u,33u,131072u,131073u})
    Probe(submit, [&](auto &f) {
      f.draw.indexed = 1; f.draw.index_size = 1; f.draw.raw_index_data = indices.data();
      f.draw.index_count = f.tess.vertices_per_instance = occurrences;
      f.draw.raw_index_data_size = occurrences; f.tess.input_vertices = 32;
    }, occurrences <= 131072 ? kLateGate : "input occurrence count exceeds",
          "indexed patch budget uses index occurrences, not unique vertex count");
}

void VerifyUniformBuffers(Submission &submit) {
  for (auto count : {76u,UINT32_MAX})
    Probe(submit, [&](auto &f) { f.draw.uniform_buffers = f.ubos.data(); f.draw.uniform_buffer_count = count; },
          "uniform buffer payload list", "five-stage aggregate UBO list bound");
  Probe(submit, [](auto &f) { f.draw.uniform_buffers = f.ubos.data(); }, "uniform buffer payload list", "non-null empty UBO list");
  Probe(submit, [](auto &f) { f.draw.uniform_buffer_count = 1; }, "uniform buffer payload list", "missing UBO list pointer");
  Probe(submit, [](auto &f) { f.draw.uniform_buffers = f.ubos.data(); f.draw.uniform_buffer_count = 75; },
        kLateGate, "75-entry list envelope fits all five stage namespaces");
  const auto probe = [&](const auto &mutate, const char *expected, const std::string &name, unsigned stage) {
    Fixture f;
    if (stage == 2) f.Geometry();
    f.DescriptorCount(stage, 15); f.AddUbo(stage, 14, 4);
    mutate(f); submit.AfterUbo(f.draw, expected, name + " stage=" + std::to_string(stage));
  };
  for (unsigned stage = 0; stage < 5; ++stage) {
    probe([](auto &) {}, kLateGate, "stage-local UBO14 and sparse zero descriptors", stage);
    for (std::size_t bytes : {std::size_t{1},std::size_t{65536}})
      probe([&](auto &f) {
        f.ubos[0].bytes_size = bytes;
        f.shared[stage][f.Abi(stage).uniform_buffer_descriptor_start + 14 * 4 + 2] = static_cast<std::uint32_t>(bytes);
      }, kLateGate, "legal exact UBO byte endpoint", stage);
    for (std::size_t bytes : {std::size_t{0},std::size_t{65537},SIZE_MAX})
      probe([&](auto &f) { f.ubos[0].bytes_size = bytes; }, kUboEntry, "UBO byte bound", stage);
    probe([](auto &f) { f.ubos[0].bytes = nullptr; }, kUboEntry, "missing UBO bytes", stage);
    for (auto block : {15u,UINT32_MAX})
      probe([&](auto &f) { f.ubos[0].block_index = block; }, kUboEntry, "UBO block index bound", stage);
    probe([](auto &f) { f.ubos[1] = f.ubos[0]; f.draw.uniform_buffer_count = 2; },
          "stage/block index is duplicated", "duplicate stage-local UBO", stage);
    probe([&](auto &f) { f.DescriptorCount(stage, 14); }, "payload block exceeds its stage descriptor range",
          "payload outside declared descriptors", stage);
    for (unsigned component : {0u,1u,3u})
      probe([&](auto &f) { f.shared[stage][f.Abi(stage).uniform_buffer_descriptor_start + 14 * 4 + component] = 1; },
            "descriptor address/size/dynamic offset is not canonical", "unrelocated UBO address/reserved word", stage);
    probe([&](auto &f) { f.shared[stage][f.Abi(stage).uniform_buffer_descriptor_start + 14 * 4 + 2] = 3; },
          "descriptor address/size/dynamic offset is not canonical", "descriptor byte extent disagrees with payload", stage);
    probe([](auto &f) { f.draw.uniform_buffers = nullptr; f.draw.uniform_buffer_count = 0; },
          "descriptor address/size/dynamic offset is not canonical", "nonzero descriptor extent needs actual bytes", stage);
  }
  for (auto stage : {5u,UINT32_MAX})
    probe([&](auto &f) { f.ubos[0].stage = stage; }, kUboEntry, "unknown UBO stage", 3);
  for (bool geometry : {false,true}) {
    Fixture f;
    if (geometry) f.Geometry();
    const auto stages = geometry ? std::vector<unsigned>{0,1,2} : std::vector<unsigned>{0,1,3,4};
    for (auto stage : stages) {
      f.DescriptorCount(stage, 15);
      for (unsigned block = 0; block < 15; ++block) f.AddUbo(stage, block, 4);
    }
    submit.AfterUbo(f.draw, kLateGate, geometry ? "45 GS/VS/FS UBOs" : "60 VS/TCS/TES/FS UBOs with distinct stage namespaces");
  }
  Fixture missing_stage;
  missing_stage.ubos[0] = {2,0,missing_stage.uniform_bytes.data(),4};
  missing_stage.draw.uniform_buffers = missing_stage.ubos.data(); missing_stage.draw.uniform_buffer_count = 1;
  submit.AfterUbo(missing_stage.draw, "payload block exceeds its stage descriptor range",
                  "five API stage tags do not permit a phantom GS binding in a Tessellation draw");
}

void VerifyBinaryStages() {
  static_assert(PVRGPU_SYSTEMC_PCO_SHADER_STAGE_TESS_CONTROL == 3);
  static_assert(PVRGPU_SYSTEMC_PCO_SHADER_STAGE_TESS_EVALUATION == 4);
  const auto probe = [](unsigned stage, const std::vector<std::uint8_t> &bytes, bool valid, const std::string &name) {
    std::array<char, 512> error{};
    const int status = pvrgpu_systemc_can_execute_pco_binary(stage, bytes.data(), bytes.size(), error.data(), error.size());
    Check(status == (valid ? 0 : 2), name + ": " + error.data());
    if (!valid) Check(error[0] != 0, name + " requires a diagnostic");
  };
  const std::array<const std::vector<std::uint8_t> *, 5> control = {&kTess0tcs,&kTess1tcs,&kTess2tcs,&kTess3tcs,&kTess4tcs};
  const std::array<const std::vector<std::uint8_t> *, 5> evaluation = {&kTess0tes,&kTess1tes,&kTess2tes,&kTess3tes,&kTess4tes};
  for (unsigned kind = 0; kind < control.size(); ++kind) {
    probe(3, *control[kind], true, "real native TCS compiler fixture " + std::to_string(kind));
    probe(4, *evaluation[kind], true, "real native TES compiler fixture " + std::to_string(kind));
    probe(4, *control[kind], false, "TCS store/NOP.end cannot masquerade as TES");
    probe(3, *evaluation[kind], false, "TES UVSW cannot masquerade as TCS");
  }
  for (unsigned stage : {0u,1u,2u,5u,UINT32_MAX})
    probe(stage, kTess0tcs, false, "native TCS rejects wrong public stage " + std::to_string(stage));
  for (unsigned stage : {1u,5u,UINT32_MAX})
    probe(stage, kTess0tes, false, "native TES rejects incompatible public stage " + std::to_string(stage));
  // VS/GS/TES share the UVSW ISA, so a TES stream may also be decodable by
  // another vertex-output decoder. Stage ownership comes from real compiler
  // metadata and the distinct native task ABI, not a binary fingerprint.
  for (unsigned stage : {3u,4u}) {
    const auto &native = stage == 3 ? kTess0tcs : kTess0tes;
    probe(stage, {}, false, "empty tessellation binary");
    auto truncated = native; truncated.pop_back();
    probe(stage, truncated, false, "truncated final native group");
    auto trailing = native; trailing.push_back(0);
    probe(stage, trailing, false, "bytes following native END");
    std::array<char, 2> error = {'x','x'};
    Check(pvrgpu_systemc_can_execute_pco_binary(stage, nullptr, native.size(), error.data(), error.size()) == 2,
          "binary size cannot substitute for an actual pointer");
    Check(error.back() == 0, "short diagnostic buffer is terminated");
    Check(pvrgpu_systemc_can_execute_pco_binary(stage, nullptr, 1, nullptr, 0) == 2,
          "optional diagnostic pointer is safe");
    GuardedBytes bytes(native.size());
    std::memcpy(bytes.data(), native.data(), native.size());
    std::array<char, 512> diagnostic{};
    Check(pvrgpu_systemc_can_execute_pco_binary(stage, bytes.data(), native.size(), diagnostic.data(), diagnostic.size()) == 0,
          std::string("native stage decoder reads no byte beyond binary extent: ") + diagnostic.data());
  }
}
} // namespace

int main() {
  try {
    const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    Submission submit(std::filesystem::temp_directory_path() / ("pvrgpu-tess-api25-boundary-" + std::to_string(nonce)));
    Check(!std::filesystem::exists(submit.root), "boundary output path starts absent");
    VerifyVersions(submit);
    VerifyMetadata(submit);
    VerifyStageAbis(submit);
    VerifyDrawExtents(submit);
    VerifyUniformBuffers(submit);
    VerifyBinaryStages();
    Check(!std::filesystem::exists(submit.root), "all boundary probes produce no model output");
    std::cout << "native Tessellation API25 boundary " << checks << " checks PASS (no model submissions)\n";
    return 0;
  } catch (const std::exception &failure) {
    std::cerr << "pco-tessellation-api-boundary-test: " << failure.what() << '\n';
    return 1;
  }
}
