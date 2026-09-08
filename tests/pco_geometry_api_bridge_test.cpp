// SPDX-License-Identifier: MIT
#include "pvrgpu_systemc_api.h"
#include "pco_geometry_fixtures.h"
#include "pco_geometry_compiler_fixtures.h"
#include "pco_geometry_stats_checks.h"
#include "shader/pco_iss.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
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
std::uint32_t checks = 0;
void Check(bool condition, const std::string &message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

struct Fixture {
  std::vector<std::uint8_t> gs = pvrgpu::stub::GeometryNativeLoadFixture();
  std::vector<std::uint8_t> vs = pvrgpu::stub::ConditionalsVertexPcoBinary();
  std::vector<std::uint8_t> fs = pvrgpu::stub::ConditionalsFragmentPcoBinary();
  std::array<float, 12> vertices = {0,0,0,1, 0,0,0,1, 0,0,0,1};
  std::array<std::uint32_t, 16> vs_shared = {
      0x3f800000,0,0,0, 0,0x3f800000,0,0,
      0,0,0x3f800000,0, 0,0,0,0x3f800000};
  std::array<std::uint32_t, 4> fs_shared = {0x3f800000,0,0,0x3f800000};
  std::array<std::uint32_t, 256> gs_shared{};
  pvrgpu_systemc_driver_command Draw() const {
    pvrgpu_systemc_driver_command d{};
    d.version = PVRGPU_SYSTEMC_API_VERSION;
    d.command = "draw_pco_triangles";
    d.case_name = "bridge.geometry.boundary.draw";
    d.format = "PIPE_FORMAT_R8G8B8A8_UNORM";
    d.framebuffer_width = d.width = 16;
    d.framebuffer_height = d.height = 16;
    d.raw_vertex_data = reinterpret_cast<const std::uint8_t *>(vertices.data());
    d.raw_vertex_data_size = sizeof(vertices);
    d.vertex_stride = 16; d.vertex_count = 3; d.instance_count = 1;
    d.vertex_pco = vs.data(); d.vertex_pco_size = vs.size();
    d.fragment_pco = fs.data(); d.fragment_pco_size = fs.size();
    d.vertex_shared = vs_shared.data(); d.vertex_shared_count = vs_shared.size();
    d.fragment_shared = fs_shared.data(); d.fragment_shared_count = fs_shared.size();
    d.vertex_pco_abi = {10,4,4,0,16,0,16,0,0,0};
    d.fragment_pco_abi = {4,0,0,4,4,0,4,0,0,0};
    d.geometry_pco = gs.data(); d.geometry_pco_size = gs.size();
    d.geometry_shared = gs_shared.data(); d.geometry_shared_count = 4;
    d.geometry_pco_abi = {4,2,4,0,4,4,0,0,4,0};
    d.geometry_input_primitive_vertices = 1;
    d.geometry_max_vertices = 1; d.geometry_invocations = 1;
    d.geometry_input_stride_dwords = 4; d.geometry_vertices_per_instance = 3;
    d.position_output_count = d.varying_output_start = 4;
    d.fragment_position_count = d.fragment_varying_start = 4;
    d.viewport_scale_bits[0] = d.viewport_scale_bits[1] = 0x41000000;
    d.viewport_scale_bits[2] = 0x3f000000;
    std::memcpy(d.viewport_translate_bits, d.viewport_scale_bits, sizeof(d.viewport_scale_bits));
    d.half_pixel_center = d.depth_clip_near = d.depth_clip_far = 1;
    d.sample_mask = UINT32_MAX; d.color_mask = 15;
    d.blend_source_rgb_factor = d.blend_source_alpha_factor = PVRGPU_SYSTEMC_PCO_BLEND_FACTOR_ONE;
    d.color_attachment_source_command_index = d.depth_attachment_source_command_index = PVRGPU_SYSTEMC_ATTACHMENT_NEW_CLEAR;
    // A later, named gate distinguishes successful GS ABI validation from an
    // earlier rejection. These boundary probes never enqueue model work.
    d.blend_enable = 2;
    return d;
  }
};

struct Submission {
  std::string jsonl, outdir;
  pvrgpu_systemc_driver_command sequence{};
  pvrgpu_systemc_submit_info info{};
  explicit Submission(const std::filesystem::path &root)
      : jsonl((root / "model.jsonl").string()), outdir((root / "out").string()) {
    sequence.version = PVRGPU_SYSTEMC_API_VERSION;
    sequence.command = "draw_pco_sequence";
    sequence.case_name = "bridge.geometry.boundary";
    sequence.format = "PIPE_FORMAT_R8G8B8A8_UNORM";
    sequence.framebuffer_width = sequence.width = 16;
    sequence.framebuffer_height = sequence.height = 16;
    sequence.pco_sequence_command_count = 1;
    info.version = PVRGPU_SYSTEMC_API_VERSION; info.command = &sequence;
    info.jsonl_path = jsonl.c_str(); info.outdir = outdir.c_str(); info.memory_mode = "cache";
  }
  void Rejected(const pvrgpu_systemc_driver_command &draw, const char *expected, const std::string &label) {
    sequence.pco_sequence_commands = &draw;
    std::array<char, 512> error{};
    const int status = pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size());
    Check(status == 2, label + " must be rejected, status=" + std::to_string(status));
    Check(std::string(error.data()).find(expected) != std::string::npos,
          label + " rejected at wrong gate: " + error.data());
  }
};

void VerifyBinaryStages() {
  static_assert(PVRGPU_SYSTEMC_PCO_SHADER_STAGE_GEOMETRY == 2);
  const auto check_binary = [](std::uint32_t stage, const std::vector<std::uint8_t> &bytes, bool valid, const std::string &name) {
    std::array<char, 512> error{};
    const int status = pvrgpu_systemc_can_execute_pco_binary(stage, bytes.data(), bytes.size(), error.data(), error.size());
    Check(status == (valid ? 0 : 2), name + ": " + error.data());
    if (!valid) Check(error[0] != 0, name + " needs a diagnostic");
  };
  const auto native = pvrgpu::stub::GeometryNativeLoadFixture();
  check_binary(2, native, true, "native LD/EMIT/CUT/ENDTASK is GS API stage 2");
  check_binary(0, native, false, "GS cannot use the VS decoder");
  check_binary(1, native, false, "GS cannot use the FS decoder");
  check_binary(3, native, false, "internal enum 3 is not the public GS stage");
  check_binary(UINT32_MAX, native, false, "unknown stage");
  check_binary(2, {}, false, "empty GS binary");
  auto truncated = native; truncated.pop_back();
  check_binary(2, truncated, false, "truncated native ENDTASK");
  auto trailing = native; trailing.push_back(0);
  check_binary(2, trailing, false, "bytes after native ENDTASK");
  for (unsigned kind = 0; kind != 10; ++kind)
    check_binary(2, pvrgpu::stub::GeometryCompilerFixture(kind), true,
                 "genuine GS compiler fixture " + std::to_string(kind));
  std::array<char, 2> error = {'x','x'};
  Check(pvrgpu_systemc_can_execute_pco_binary(2, nullptr, native.size(), error.data(), error.size()) == 2,
        "missing binary pointer must fail");
  Check(error.back() == 0, "short error buffer is NUL terminated");
  Check(pvrgpu_systemc_can_execute_pco_binary(2, nullptr, 1, nullptr, 0) == 2,
        "missing diagnostic storage is safe");
}

void VerifyPreviousVersionGuard(Submission &submit) {
  static_assert(PVRGPU_SYSTEMC_API_VERSION == 25);
  constexpr auto previous_size = offsetof(pvrgpu_systemc_driver_command, geometry_pco);
  static_assert(previous_size % alignof(pvrgpu_systemc_driver_command) == 0);
#if defined(_WIN32)
  SYSTEM_INFO system_info{}; GetSystemInfo(&system_info);
  const auto page_size = static_cast<std::size_t>(system_info.dwPageSize);
  void *pages = VirtualAlloc(nullptr, page_size * 2, MEM_COMMIT | MEM_RESERVE, PAGE_NOACCESS);
  DWORD old_protection = 0;
  Check(pages && VirtualProtect(pages, page_size, PAGE_READWRITE, &old_protection), "allocate API23 guard");
#else
  const long queried_size = sysconf(_SC_PAGESIZE);
  Check(queried_size > 0, "query guard page size");
  const auto page_size = static_cast<std::size_t>(queried_size);
  void *pages = mmap(nullptr, page_size * 2, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
  Check(pages != MAP_FAILED && mprotect(pages, page_size, PROT_READ | PROT_WRITE) == 0, "allocate API23 guard");
#endif
  Check(previous_size <= page_size, "API23 command fits guard page");
  auto *bytes = static_cast<std::uint8_t *>(pages) + page_size - previous_size;
  std::memset(bytes, 0, previous_size);
  const std::uint32_t old_version = 23;
  std::memcpy(bytes, &old_version, sizeof(old_version));
  const auto &old_command = *reinterpret_cast<const pvrgpu_systemc_driver_command *>(bytes);
  auto info = submit.info; info.command = &old_command;
  std::array<char, 512> error{};
  Check(pvrgpu_systemc_submit_driver_command(&info, error.data(), error.size()) == 2 &&
            std::string(error.data()).find("command version") != std::string::npos,
        "top-level API23 short command is rejected before API24 tail access");
  submit.Rejected(old_command, "version=23 expected=25", "nested guarded API23 command");
#if defined(_WIN32)
  Check(VirtualFree(pages, 0, MEM_RELEASE) != 0, "release API23 guard");
#else
  Check(munmap(pages, page_size * 2) == 0, "release API23 guard");
#endif
}

void VerifyGeometryPayload(Fixture &fixture, Submission &submit) {
  constexpr const char *geometry_error = "geometry ABI, topology or unrelocated input descriptor";
  const auto reject = [&](const auto &mutate, const std::string &name, const char *expected = geometry_error) {
    auto draw = fixture.Draw(); mutate(draw); submit.Rejected(draw, expected, name);
  };
  reject([](auto &d) { d.geometry_pco = nullptr; }, "absent GS executable pointer");
  reject([](auto &d) { d.geometry_pco_size = 0; }, "GS payload without binary", "geometry payload without executable");
  reject([](auto &d) { d.geometry_pco_size = SIZE_MAX; }, "unbounded GS binary size");
  reject([](auto &d) { d.geometry_shared = nullptr; }, "absent GS SHARED pointer");
  reject([](auto &d) { d.geometry_shared_count = 3; }, "truncated GS SHARED count");
  reject([](auto &d) { d.geometry_shared_count = 5; }, "oversized GS SHARED count");
  reject([](auto &d) { d.geometry_pco_abi = {}; }, "missing complete GS ABI");
  reject([](auto &d) { d.render_target_count = 2; d.fragment_output_mask[0] = 15; },
         "GS sparse MRT without per-target LOAD", "geometry MRT");
  reject([](auto &d) { d.render_target_count = 2;
                       d.fragment_output_mask[0] = d.fragment_output_mask[1] = 15; },
         "GS dense MRT also needs per-target LOAD for untouched pixels", "geometry MRT");
  reject([](auto &d) { d.geometry_pco = d.vertex_pco; d.geometry_pco_size = d.vertex_pco_size;
                       d.geometry_pco_abi = d.vertex_pco_abi; }, "VS executable ABI cannot stand in for GS");
  for (auto field : {&pvrgpu_systemc_driver_command::geometry_invocations,
                     &pvrgpu_systemc_driver_command::geometry_input_stride_dwords,
                     &pvrgpu_systemc_driver_command::geometry_input_primitive_vertices,
                     &pvrgpu_systemc_driver_command::geometry_vertices_per_instance})
    reject([&](auto &d) { d.*field = 0; }, "missing required GS layout field");
  for (std::uint32_t count : {5U,7U,UINT32_MAX})
    reject([&](auto &d) { d.geometry_input_primitive_vertices = count; }, "invalid primitive input arity");
  for (std::uint32_t count : {33U,UINT32_MAX})
    reject([&](auto &d) { d.geometry_invocations = count; }, "GS invocation bound");
  for (std::uint32_t count : {257U,UINT32_MAX})
    reject([&](auto &d) { d.geometry_max_vertices = count; }, "GS emission bound");
  for (std::uint32_t count : {3U,5U,65U,UINT32_MAX})
    reject([&](auto &d) { d.geometry_input_stride_dwords = count; }, "VS to GS stride mismatch");
  reject([](auto &d) { d.geometry_vertices_per_instance = 2; }, "nonintegral instance occurrence extent");
  for (std::uint32_t topology : {1U,2U,4U,6U,UINT32_MAX})
    reject([&](auto &d) { d.geometry_output_primitive = topology; }, "invalid GS output topology");
  reject([](auto &d) { d.geometry_pco_abi.temps = 257; }, "TEMP overflow");
  reject([](auto &d) { d.geometry_pco_abi.coefficients = 1; }, "GS coefficient bank forbidden");
  for (std::uint32_t count : {0U,1U,3U,64U})
    reject([&](auto &d) { d.geometry_pco_abi.vertex_inputs = count; }, "GS needs exact two system-value inputs");
  for (std::uint32_t count : {0U,3U,65U,UINT32_MAX})
    reject([&](auto &d) { d.geometry_pco_abi.vertex_outputs = count; }, "GS output bank bounds");
  for (std::uint32_t start : {0U,3U,5U,UINT32_MAX})
    reject([&](auto &d) { d.geometry_pco_abi.uniform_buffer_descriptor_start = start; }, "UBO must follow primitive descriptor");
  reject([](auto &d) { d.geometry_pco_abi.uniform_buffer_descriptor_count = 16; }, "UBO descriptor overflow");
  reject([](auto &d) { d.geometry_pco_abi.push_constant_start = 0; }, "push prefix overlaps primitive descriptor");
  reject([](auto &d) { d.geometry_pco_abi.push_constant_count = 1; }, "push suffix exceeds SHARED");
  reject([](auto &d) { d.geometry_pco_abi.entry_offset = 4; }, "nonzero GS entry offset");
  for (std::uint32_t count : {3U,257U,UINT32_MAX})
    reject([&](auto &d) { d.geometry_pco_abi.shareds = d.geometry_shared_count = count; }, "SHARED bank bound");
  for (unsigned component = 0; component != 4; ++component) {
    fixture.gs_shared[component] = UINT32_MAX;
    submit.Rejected(fixture.Draw(), geometry_error, "primitive descriptor must be unrelocated/canonical");
    fixture.gs_shared[component] = 0;
  }
  reject([](auto &d) { d.geometry_layer_output_count = 2; }, "layer must be scalar");
  reject([](auto &d) { d.geometry_primitive_id_output_count = 2; }, "primitive ID must be scalar");
  reject([](auto &d) { d.geometry_layer_output_start = UINT32_MAX; d.geometry_layer_output_count = 1; }, "layer addition cannot wrap");
  reject([](auto &d) { d.geometry_primitive_id_output_start = UINT32_MAX; d.geometry_primitive_id_output_count = 1; }, "primitive ID addition cannot wrap");
  reject([](auto &d) { d.geometry_layer_output_start = 4; d.geometry_layer_output_count = 1; }, "layer outside output bank");
  reject([](auto &d) { d.geometry_primitive_id_output_start = 4; d.geometry_primitive_id_output_count = 1; }, "primitive ID outside output bank");
  // Do not pretend these are draws: the intentional later blend error proves
  // only that valid GS envelopes survived every ABI/topology guard above it.
  for (std::uint32_t maximum : {0U,1U,256U}) {
    auto draw = fixture.Draw(); draw.geometry_max_vertices = maximum;
    submit.Rejected(draw, "unsupported: blend", "legal GS maximum including zero");
  }
  for (std::uint32_t inputs : {1U,2U,3U,4U,6U}) {
    auto draw = fixture.Draw(); draw.geometry_input_primitive_vertices = inputs;
    submit.Rejected(draw, "unsupported: blend", "legal GS input layout arity");
  }
  for (std::uint32_t topology : {0U,3U,5U}) {
    auto draw = fixture.Draw(); draw.geometry_output_primitive = topology;
    submit.Rejected(draw, "unsupported: blend", "legal GS output topology");
  }
  for (std::uint32_t temps : {0U,256U}) {
    auto draw = fixture.Draw(); draw.geometry_pco_abi.temps = temps;
    submit.Rejected(draw, "unsupported: blend", "legal GS TEMP envelope boundary");
  }
}

void VerifyZeroAttributePayload(Fixture &fixture, Submission &submit) {
  const auto make = [&]() {
    auto draw = fixture.Draw();
    draw.vertex_stride = 0; draw.raw_vertex_data = nullptr; draw.raw_vertex_data_size = 0;
    draw.vertex_pco_abi.vertex_inputs = 0;
    return draw;
  };
  submit.Rejected(make(), "unsupported: blend", "canonical GS draw with no attributes and no VBO");
  const auto reject = [&](const auto &mutate, const char *expected, const char *label) {
    auto draw = make(); mutate(draw); submit.Rejected(draw, expected, label);
  };
  reject([](auto &d) { d.vertex_attribute_count = 1; }, "vertex stride is below", "zero stride cannot carry an attribute");
  reject([](auto &d) { d.vertex_pco_abi.vertex_inputs = 4; }, "vertex stride is below", "VTXIN reads require actual vertex input storage");
  reject([](auto &d) { d.vertex_stride = 4; }, "vertex data is absent", "nonzero stride requires VBO bytes");
  reject([&](auto &d) { d.raw_vertex_data = reinterpret_cast<const std::uint8_t *>(fixture.vertices.data()); },
         "vertex stride is below", "no-VBO pointer must be null");
  reject([](auto &d) { d.raw_vertex_data_size = 4; }, "vertex stride is below", "no-VBO byte extent must be zero");
  reject([](auto &d) { d.vertex_count = 0; }, "vertex count is zero", "no attributes does not mean no draw occurrences");
  reject([](auto &d) { d.first_vertex = 1; }, "first vertex is not zero", "no-VBO first occurrence remains canonical");
  reject([](auto &d) { d.instance_count = 2; }, "instance count is not one", "no-VBO instance transport remains canonical");
}

void VerifyAcceptedNativePipeline(Fixture &fixture, Submission &submit,
                                  const std::filesystem::path &root, unsigned kind) {
  // Genuine three-stage PCO binaries from generate_geometry.c KIND 12. The
  // position-only VS and varying-color FS were linked against GS KIND 0;
  // these are compiler output bytes, never hand-authored shader behavior.
  std::vector<std::uint8_t> vs = {
    0x58,0xa0,0x06,0x08,0x00,0x80,0x04,0x00,0x00,0x30,0xf3,0xff,
    0xff,0xff,0xff,0xff,0x44,0xa0,0x80,0x05,0x00,0x00,0x00,0xff};
  std::vector<std::uint8_t> gs = pvrgpu::stub::GeometryCompilerFixture(0);
  std::vector<std::uint8_t> fs = {
    0x56,0xa0,0x00,0xb0,0x04,0xc4,0x40,0x10,0xc0,0x40,0x00,0xff,
    0x02,0x80,0x6a,0xff,0x34,0x8a,0x00,0x87,0x40,0x00,0x00,0x20,
    0x34,0x8a,0x00,0x87,0x41,0x00,0x00,0x21,0x34,0x8a,0x00,0x87,
    0x42,0x00,0x00,0x22,0x34,0x8a,0x80,0x87,0x43,0x00,0x00,0x23};
  const bool no_attributes = kind >= 15;
  const bool zero_emit = kind == 14 || kind == 17 || kind == 18;
  const bool no_fragment_output = kind == 18 || kind == 19;
  // KIND 14 is a true max_vertices=0 shader, KIND 15 is a no-input/no-output
  // VS followed by a GS that generates its own position, and KIND 17 combines
  // that no-op VS with the true zero-Emit GS. All bytes came from the compiler.
  if (no_attributes)
    vs = {0x44,0xa0,0x80,0x05,0x00,0x00,0x00,0xff};
  if (zero_emit)
    gs = {0x44,0xa0,0x80,0x04,0x00,0x00,0x00,0xff};
  else if (no_attributes)
    gs = {
      0x55,0xa0,0x00,0x08,0x00,0x80,0x00,0x00,0x00,0x30,0x55,0xa0,
      0x00,0x08,0x01,0x80,0x00,0x00,0x00,0x30,0x55,0xa0,0x00,0x08,
      0x02,0x80,0x00,0x00,0x00,0x30,0x55,0xa0,0x00,0x08,0x03,0x80,
      0x01,0x00,0x00,0x30,0x44,0xa0,0x00,0x01,0x00,0x00,0x00,0xff,
      0x44,0xa0,0x00,0x02,0x00,0x00,0x00,0xff,0x44,0xa0,0x80,0x04,
      0x00,0x00,0x00,0xff};
  if (kind != 12)
    fs = {
      0x35,0x8a,0x00,0x87,0x8c,0x01,0x00,0x00,0x00,0x20,0x35,0x8a,
      0x00,0x87,0x8b,0x01,0x00,0x00,0x00,0x21,0x86,0x92,0x40,0x13,
      0x00,0x00,0x40,0x3f,0x00,0x00,0x40,0xff,0x34,0x8a,0x00,0x87,
      0x40,0x00,0x00,0x22,0x38,0x8a,0x80,0x87,0x80,0x01,0x00,0x00,
      0x00,0x23,0xf3,0xff,0xff,0xff,0xff,0xff};
  if (no_fragment_output) fs = pvrgpu::stub::GeometryEmptyFragmentFixture();
  std::array<std::uint32_t, 4> shared{};
  std::array<float, 4> position = {0,0,0,1};
  std::array<std::uint8_t, 16 * 16 * 4> initial{};
  for (unsigned y=0;y<16;++y) for (unsigned x=0;x<16;++x) {
    const unsigned pixel=(y*16+x)*4;
    initial[pixel]=static_cast<std::uint8_t>(x*13+y*7);
    initial[pixel+1]=static_cast<std::uint8_t>(x*3+y*17);
    initial[pixel+2]=static_cast<std::uint8_t>((x^y)*11);
    initial[pixel+3]=static_cast<std::uint8_t>(19+x*5+y*3);
  }
  const auto expected_initial=initial;
  auto draw = fixture.Draw();
  draw.blend_enable = 0;
  draw.case_name = submit.sequence.case_name;
  draw.frame = submit.sequence.frame = 1;
  draw.vertex_attribute_count = 1; draw.vertex_attribute_components[0] = 4;
  draw.raw_vertex_data = reinterpret_cast<const std::uint8_t *>(position.data());
  draw.raw_vertex_data_size = sizeof(position);
  draw.vertex_count = draw.geometry_vertices_per_instance = 1;
  draw.vertex_pco = vs.data(); draw.vertex_pco_size = vs.size();
  draw.geometry_pco = gs.data(); draw.geometry_pco_size = gs.size();
  draw.fragment_pco = fs.data(); draw.fragment_pco_size = fs.size();
  draw.vertex_shared = draw.fragment_shared = nullptr;
  draw.vertex_shared_count = draw.fragment_shared_count = 0;
  draw.geometry_shared = shared.data();
  draw.vertex_pco_abi = {0,4,4,0,0,0,0,0,0,0};
  draw.geometry_pco_abi = {4,2,8,0,4,4,0,0,4,0};
  draw.fragment_pco_abi = {4,0,0,20,0,0,0,0,0,0};
  draw.varying_output_count = 4; draw.fragment_varying_count = 16;
  draw.fragment_output_mask[0] = 15;
  if (kind != 12) {
    draw.geometry_pco_abi = {0,2,4,0,4,4,0,0,4,0};
    draw.fragment_pco_abi = {1,0,0,4,0,0,0,0,0,0};
    draw.varying_output_count = draw.fragment_varying_count = 0;
  }
  if (zero_emit) draw.geometry_max_vertices = 0;
  if (no_fragment_output) {
    draw.fragment_pco_abi.temps = 0;
    draw.fragment_output_mask[0] = 0;
    draw.initial_color_attachment_bytes = initial.data();
    draw.initial_color_attachment_bytes_size = initial.size();
  }
  if (no_attributes) {
    draw.vertex_pco_abi.vertex_inputs = 0;
    draw.vertex_attribute_count = draw.vertex_attribute_components[0] = 0;
    draw.vertex_stride = 0; draw.raw_vertex_data = nullptr; draw.raw_vertex_data_size = 0;
  }
  draw.clear_color_bits[0] = draw.clear_color_bits[2] = draw.clear_color_bits[3] = 0x3f800000;
  submit.sequence.clear_color_bits[0] = submit.sequence.clear_color_bits[2] = submit.sequence.clear_color_bits[3] = 0x3f800000;
  std::filesystem::create_directories(root / "out");
  submit.sequence.pco_sequence_commands = &draw;
  submit.info.submission_generation = 100 + kind;
  std::array<char, 1024> error{};
  Check(pvrgpu_systemc_submit_driver_command(&submit.info, error.data(), error.size()) == 0,
        std::string("accept genuine three-stage GS pipeline: ") + error.data());
  // The API promises an immutable copy before returning. Destroy every
  // borrowed executable, register and input payload before native execution.
  std::fill(vs.begin(), vs.end(), 0);
  std::fill(gs.begin(), gs.end(), 0);
  std::fill(fs.begin(), fs.end(), 0);
  shared.fill(UINT32_MAX); position.fill(0); initial.fill(0);
  draw = {};
  VerifyGeometryStats(submit.info.submission_generation, 1, zero_emit ? 0 : 1, Check);
  std::array<std::uint8_t, 16 * 16 * 4> pixels{};
  pvrgpu_systemc_readback_info readback{};
  readback.version = PVRGPU_SYSTEMC_API_VERSION;
  readback.width = readback.height = 16; readback.bytes_per_pixel = 4;
  readback.pixels = pixels.data(); readback.pixels_size = pixels.size();
  error.fill(0);
  const int status = pvrgpu_systemc_flush_readback(&readback, error.data(), error.size());
  Check(status == 0, std::string("execute copied genuine GS pipeline: ") + error.data());
  Check(readback.pixels_written == 1, "native GS draw produced readback");
  unsigned shader_pixels = 0;
  for (std::size_t pixel = 0; pixel < 16 * 16; ++pixel) {
    const auto *rgba = pixels.data() + pixel * 4;
    if (no_fragment_output) {
      Check(std::equal(rgba,rgba+4,expected_initial.data()+pixel*4),
            "empty FS preserves every nonuniform LOAD byte rather than recreating the clear");
      continue;
    }
    const bool clear = rgba[0] == 255 && rgba[1] == 0 && rgba[2] == 255 && rgba[3] == 255;
    const bool shader_color = kind == 12
        ? rgba[0] == 0 && rgba[1] == 0 && rgba[2] == 0 && rgba[3] == 255
        : rgba[0] == 64 && rgba[1] == 128 && rgba[2] == 191 && rgba[3] == 255;
    Check(clear || (!zero_emit && shader_color), "only modeled clear or shader-written point pixels exist");
    shader_pixels += shader_color;
  }
  Check(shader_pixels == (zero_emit || no_fragment_output ? 0U : 1U), "native GS/FS output determines exact color-write count");
  std::ifstream log(submit.jsonl, std::ios::binary);
  const std::string json{std::istreambuf_iterator<char>(log), std::istreambuf_iterator<char>()};
  for (const char *evidence : {"\"gs_invocations\":1", "\"gs_input_write_bytes\":16",
                               "\"pool_bytes_in_flight\":0",
                               "\"pool_leaks\":0"})
    Check(json.find(evidence) != std::string::npos, std::string("native GS/pool evidence missing: ") + evidence);
  Check(json.find(zero_emit ? "\"gs_primitives\":0" : "\"gs_primitives\":1") != std::string::npos,
        "zero-Emit GS complete primitive counter remains zero");
  Check(json.find(zero_emit ? "\"gs_emitted_vertices\":0" : "\"gs_emitted_vertices\":1") != std::string::npos,
        "actual GS snapshot counter matches emission");
  Check(json.find(kind == 12 ? "\"gs_input_read_bytes\":16" : "\"gs_input_read_bytes\":0") != std::string::npos,
        "only native GS LD contributes input reads");
  if (no_attributes)
    for (const char *evidence : {"\"vertex_attribute_bytes\":0", "\"vertex_attribute_fetches\":0"})
      Check(json.find(evidence) != std::string::npos, std::string("no fabricated attribute fetch: ") + evidence);
  if (no_fragment_output) {
    for (const char *evidence : {"\"pbe_color_reads\":0", "\"pbe_blended_fragments\":0",
                                 "\"pbe_fragment_writes\":0", "\"memory_mode\":\"cache\"",
                                 "\"nop\":1", "\"fs_alu_instructions\":0"})
      Check(json.find(evidence) != std::string::npos, std::string("empty FS produces no color access: ") + evidence);
    Check(json.find(zero_emit ? "\"ps_invocations\":0" : "\"ps_invocations\":1") != std::string::npos,
          "empty FS executes for emitted geometry, not for zero-Emit GS");
  }
  std::filesystem::remove_all(root);
  std::cout << "native Geometry API pipeline " << kind << " PASS\n";
}
} // namespace

int main() {
  try {
    const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const auto root = std::filesystem::temp_directory_path() / ("pvrgpu-gs-api24-" + std::to_string(nonce));
    Fixture fixture; Submission submit(root);
    VerifyBinaryStages();
    VerifyPreviousVersionGuard(submit);
    VerifyGeometryPayload(fixture, submit);
    VerifyZeroAttributePayload(fixture, submit);
    Check(!std::filesystem::exists(root), "rejected envelopes never create deferred model output");
    for (unsigned kind : {12U,14U,15U,17U,18U,19U}) {
      const auto pipeline_root = root / ("pipeline-" + std::to_string(kind));
      Submission pipeline_submit(pipeline_root);
      VerifyAcceptedNativePipeline(fixture, pipeline_submit, pipeline_root, kind);
    }
    std::filesystem::remove(root);
    std::cout << "native Geometry API24 " << checks << " checks PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "pco-geometry-api-bridge-test: " << error.what() << '\n';
    return 1;
  }
}
