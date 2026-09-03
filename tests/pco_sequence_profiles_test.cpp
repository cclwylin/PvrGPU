#include "pco_sequence_profiles.h"

#include "model_types.h"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using pvrgpu::stub::DriverPcoSequenceSupported;
using pvrgpu::stub::DriverPcoTerrainExternalPayloadHashMatches;
using pvrgpu::stub::DriverPcoTerrainFragmentBinaryHashMatches;
using pvrgpu::stub::Options;

void Check(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

Options SequenceEnvelope(const char *case_name, std::uint32_t width = 80,
                         std::uint32_t height = 60) {
  Options options;
  auto &logical = options.driver_command;
  logical.enabled = true;
  logical.schema = "pvrgpu.driver-command.v1";
  logical.producer = "pvrgpu-gallium-driver";
  logical.command = "draw_pco_sequence";
  logical.test_case = case_name;
  logical.frame = 1;
  logical.framebuffer_width = width;
  logical.framebuffer_height = height;
  logical.width = width;
  logical.height = height;
  logical.format = "PIPE_FORMAT_R8G8B8A8_UNORM";
  return options;
}

void TestRejectsNonSequenceAndClearsStaleError() {
  Options options;
  std::string error = "stale";
  Check(!DriverPcoSequenceSupported(options, &error),
        "default options unexpectedly accepted as a PCO sequence");
  Check(error == "command is not a native PCO sequence",
        "non-sequence rejection did not replace the stale reason");
  Check(!DriverPcoSequenceSupported(options, nullptr),
        "null error sink changed non-sequence validation");
}

void TestRejectsUnknownProfile() {
  Options options = SequenceEnvelope("unknown.capture.1");
  std::string error;
  Check(!DriverPcoSequenceSupported(options, &error),
        "unknown PCO sequence profile unexpectedly accepted");
  Check(error == "native PCO sequence profile is unsupported",
        "unknown profile rejection reason changed");
}

void TestProfileCardinalityDispatch() {
  struct Case {
    const char *name;
    std::size_t draws;
  };
  static constexpr std::array<Case, 3> cases = {{
      {"refract.refract.capture.1", 2},
      {"shadow.shadow.capture.1", 3},
      {"terrain.terrain.capture.1", 8},
  }};
  static constexpr std::array<std::array<std::uint32_t, 2>, 2> resolutions = {{
      {80, 60},
      {800, 600},
  }};
  for (const Case &profile : cases) {
    for (const auto &resolution : resolutions) {
      Options options =
          SequenceEnvelope(profile.name, resolution[0], resolution[1]);
      std::string error;
      Check(!DriverPcoSequenceSupported(options, &error),
            "empty physical sequence unexpectedly accepted");
      Check(error.find(std::to_string(profile.draws) + " physical draws") !=
                std::string::npos,
            "profile did not dispatch to its exact physical cardinality");
    }
  }
}

void TestUnsupportedResolutionFailsClosed() {
  static constexpr std::array<const char *, 3> cases = {
      "refract.refract.capture.1",
      "shadow.shadow.capture.1",
      "terrain.terrain.capture.1",
  };
  static constexpr std::array<std::array<std::uint32_t, 2>, 3> unsupported = {{
      {1, 1},
      {640, 480},
      {800, 60},
  }};
  for (const char *case_name : cases) {
    for (const auto &resolution : unsupported) {
      Options options =
          SequenceEnvelope(case_name, resolution[0], resolution[1]);
      std::string error;
      Check(!DriverPcoSequenceSupported(options, &error),
            "unsupported sequence resolution unexpectedly accepted");
      Check(error.find("logical command envelope is invalid") !=
                std::string::npos,
            "unsupported sequence resolution escaped the envelope gate");
    }

    Options mixed = SequenceEnvelope(case_name, 800, 600);
    mixed.driver_command.width = 80;
    mixed.driver_command.height = 60;
    std::string error;
    Check(!DriverPcoSequenceSupported(mixed, &error),
          "mixed logical/framebuffer sequence extents unexpectedly accepted");
    Check(error.find("logical command envelope is invalid") !=
              std::string::npos,
          "mixed logical/framebuffer sequence extents escaped the envelope gate");
  }
}

struct LogicalCounterCase {
  const char *case_name;
  const char *profile_name;
  std::uint32_t width;
  std::uint32_t height;
  std::size_t physical_draws;
  std::uint64_t drawlists;
  std::uint64_t ia_vertices;
  std::uint64_t ia_primitives;
  std::uint64_t clip_primitives;
  std::uint64_t ps_invocations;
  std::uint64_t texel_fetches;
  bool terrain_clear;
};

void SetLogicalCounters(Options *options, const LogicalCounterCase &fixture) {
  auto &logical = options->driver_command;
  logical.clear_color_bits = fixture.terrain_clear ?
      std::array<std::uint32_t, 4>{
          UINT32_C(0x3f533333), UINT32_C(0x3f3e147b),
          UINT32_C(0x3f1e6666), UINT32_C(0x3f800000)} :
      std::array<std::uint32_t, 4>{0, 0, 0, UINT32_C(0x3f800000)};
  logical.draw_count = fixture.drawlists;
  logical.ia_vertices = fixture.ia_vertices;
  logical.ia_primitives = fixture.ia_primitives;
  logical.vs_invocations = fixture.ia_vertices;
  logical.clip_invocations = fixture.ia_primitives;
  logical.clip_primitives = fixture.clip_primitives;
  logical.ps_invocations = fixture.ps_invocations;
  logical.setup_triangles = fixture.clip_primitives;
  logical.semantic_texel_fetches = fixture.texel_fetches;
}

void TestLogicalCountersByResolution() {
  static constexpr std::array<LogicalCounterCase, 6> fixtures = {{
      {"refract.refract.capture.1", "Refract", 80, 60, 2, 9, 417996,
       139332, 108310, 15675, 130272, false},
      {"refract.refract.capture.1", "Refract", 800, 600, 2, 12, 417996,
       139332, 108310, 1568167, 8261280, false},
      {"shadow.shadow.capture.1", "Shadow", 80, 60, 3, 3, 43036, 14346,
       14349, 3463, 2104, false},
      {"shadow.shadow.capture.1", "Shadow", 800, 600, 3, 3, 43036, 14346,
       14349, 348735, 168960, false},
      {"terrain.terrain.capture.1", "Terrain", 80, 60, 8, 42, 393258,
       131086, 25496, 329330, 5413856, true},
      {"terrain.terrain.capture.1", "Terrain", 800, 600, 8, 51, 393258,
       131086, 25496, 2658615, 91927520, true},
  }};

  for (const LogicalCounterCase &fixture : fixtures) {
    Options options =
        SequenceEnvelope(fixture.case_name, fixture.width, fixture.height);
    SetLogicalCounters(&options, fixture);
    options.driver_commands.resize(fixture.physical_draws);
    std::string error;
    Check(!DriverPcoSequenceSupported(options, &error),
          "zeroed physical sequence unexpectedly accepted");
    const std::string expected_physical_error =
        std::string(fixture.profile_name) +
        " PCO draw 0 has invalid command envelope";
    Check(error == expected_physical_error,
          "canonical logical counters did not reach physical validation");

    const auto expect_counter_rejected = [&](const char *description) {
      error.clear();
      Check(!DriverPcoSequenceSupported(options, &error), description);
      const std::string expected =
          std::string(fixture.profile_name) +
          " PCO logical counters are invalid";
      Check(error == expected,
            "mutated sequence counter escaped the resolution profile");
    };

    ++options.driver_command.draw_count;
    expect_counter_rejected("mutated sequence drawlist count was accepted");
    --options.driver_command.draw_count;
    ++options.driver_command.ps_invocations;
    expect_counter_rejected("mutated sequence PS count was accepted");
    --options.driver_command.ps_invocations;
    ++options.driver_command.semantic_texel_fetches;
    expect_counter_rejected("mutated sequence texel count was accepted");
  }
}

void TestKnownProfileEnvelopeFailsClosed() {
  Options options = SequenceEnvelope("shadow.shadow.capture.1");
  options.driver_commands.resize(3);
  std::string error;
  Check(!DriverPcoSequenceSupported(options, &error),
        "zeroed Shadow physical draws unexpectedly accepted");
  Check(error == "Shadow PCO logical counters are invalid",
        "Shadow validation did not reject missing logical counters first");

  options = SequenceEnvelope("terrain.terrain.capture.1");
  options.driver_command.clear_color_bits = {
      UINT32_C(0x3f533333), UINT32_C(0x3f3e147b),
      UINT32_C(0x3f1e6666), UINT32_C(0x3f800000)};
  options.driver_command.draw_count = 42;
  options.driver_command.ia_vertices = 393258;
  options.driver_command.ia_primitives = 131086;
  options.driver_command.vs_invocations = 393258;
  options.driver_command.clip_invocations = 131086;
  options.driver_command.clip_primitives = 25496;
  options.driver_command.ps_invocations = 329330;
  options.driver_command.setup_triangles = 25496;
  options.driver_command.semantic_texel_fetches = 5413856;
  options.driver_commands.resize(8);
  error.clear();
  Check(!DriverPcoSequenceSupported(options, &error),
        "zeroed Terrain physical draws unexpectedly accepted");
  Check(error == "Terrain PCO draw 0 has invalid command envelope",
        "Terrain did not validate physical state before acceptance");
}

void TestTerrainFullMipPayloadFingerprintsFailClosed() {
  struct Fingerprint {
    std::size_t texture_index;
    std::uint64_t hash;
  };
  static constexpr std::array<Fingerprint, 4> fingerprints = {{
      {2, UINT64_C(0xa69ccd9838551cb3)},
      {3, UINT64_C(0x777443d6a3c0ceeb)},
      {4, UINT64_C(0xd510ff3e570680dd)},
      {6, UINT64_C(0x3964257e9bde4861)},
  }};
  for (const Fingerprint &fingerprint : fingerprints) {
    Check(DriverPcoTerrainExternalPayloadHashMatches(
              fingerprint.texture_index, fingerprint.hash),
          "canonical Terrain full-mip payload fingerprint was rejected");
    Check(!DriverPcoTerrainExternalPayloadHashMatches(
               fingerprint.texture_index, fingerprint.hash ^ UINT64_C(1)),
          "mutated Terrain full-mip payload fingerprint was accepted");
  }
  Check(!DriverPcoTerrainExternalPayloadHashMatches(0, fingerprints[0].hash),
        "previous-attachment slot accepted an external fingerprint");
  Check(!DriverPcoTerrainExternalPayloadHashMatches(7, fingerprints[0].hash),
        "out-of-range Terrain texture slot was accepted");
}

void TestTerrainResolutionSpecificFragmentBinariesFailClosed() {
  struct Fixture {
    std::size_t draw_index;
    std::uint64_t hash_80x60;
    std::uint64_t hash_800x600;
  };
  static constexpr std::array<Fixture, 3> fixtures = {{
      {3, UINT64_C(0x956d5ea59737b66f), UINT64_C(0x6ad4537c64c80942)},
      {6, UINT64_C(0xab0dfc14e6aa5116), UINT64_C(0x1d6737c7f69c0953)},
      {7, UINT64_C(0xd0b9eb8de7e641d2), UINT64_C(0xb41e711d1ef41b5a)},
  }};

  for (const Fixture &fixture : fixtures) {
    Check(fixture.hash_80x60 != fixture.hash_800x600,
          "Terrain resolution-specific binaries unexpectedly collapsed");
    Check(DriverPcoTerrainFragmentBinaryHashMatches(
              80, 60, fixture.draw_index, fixture.hash_80x60),
          "Terrain 80x60 fragment binary was rejected");
    Check(DriverPcoTerrainFragmentBinaryHashMatches(
              800, 600, fixture.draw_index, fixture.hash_800x600),
          "Terrain 800x600 fragment binary was rejected");
    Check(!DriverPcoTerrainFragmentBinaryHashMatches(
               80, 60, fixture.draw_index, fixture.hash_800x600),
          "Terrain 80x60 profile accepted the 800x600 fragment binary");
    Check(!DriverPcoTerrainFragmentBinaryHashMatches(
               800, 600, fixture.draw_index, fixture.hash_80x60),
          "Terrain 800x600 profile accepted the 80x60 fragment binary");
    Check(!DriverPcoTerrainFragmentBinaryHashMatches(
               800, 600, fixture.draw_index,
               fixture.hash_800x600 ^ UINT64_C(1)),
          "mutated Terrain fragment binary fingerprint was accepted");
  }
  Check(!DriverPcoTerrainFragmentBinaryHashMatches(
             640, 480, fixtures[0].draw_index, fixtures[0].hash_80x60),
        "unsupported Terrain resolution accepted a fragment binary");
  Check(!DriverPcoTerrainFragmentBinaryHashMatches(
             800, 600, 8, fixtures[0].hash_800x600),
        "out-of-range Terrain draw accepted a fragment binary");
}

}  // namespace

int main() {
  try {
    TestRejectsNonSequenceAndClearsStaleError();
    TestRejectsUnknownProfile();
    TestProfileCardinalityDispatch();
    TestUnsupportedResolutionFailsClosed();
    TestLogicalCountersByResolution();
    TestKnownProfileEnvelopeFailsClosed();
    TestTerrainFullMipPayloadFingerprintsFailClosed();
    TestTerrainResolutionSpecificFragmentBinariesFailClosed();
    std::cout << "pco_sequence_profiles_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "pco_sequence_profiles_test: FAIL: " << error.what()
              << '\n';
    return 1;
  }
}
