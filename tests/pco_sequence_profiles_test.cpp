#include "pco_sequence_profiles.h"

#include "model_types.h"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using pvrgpu::stub::DriverPcoSequenceSupported;
using pvrgpu::stub::DriverPcoTerrainExternalPayloadHashMatches;
using pvrgpu::stub::Options;

void Check(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

Options SequenceEnvelope(const char *case_name) {
  Options options;
  auto &logical = options.driver_command;
  logical.enabled = true;
  logical.schema = "pvrgpu.driver-command.v1";
  logical.producer = "pvrgpu-gallium-driver";
  logical.command = "draw_pco_sequence";
  logical.test_case = case_name;
  logical.frame = 1;
  logical.framebuffer_width = 80;
  logical.framebuffer_height = 60;
  logical.width = 80;
  logical.height = 60;
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
  for (const Case &profile : cases) {
    Options options = SequenceEnvelope(profile.name);
    std::string error;
    Check(!DriverPcoSequenceSupported(options, &error),
          "empty physical sequence unexpectedly accepted");
    Check(error.find(std::to_string(profile.draws) + " physical draws") !=
              std::string::npos,
          "profile did not dispatch to its exact physical cardinality");
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

}  // namespace

int main() {
  try {
    TestRejectsNonSequenceAndClearsStaleError();
    TestRejectsUnknownProfile();
    TestProfileCardinalityDispatch();
    TestKnownProfileEnvelopeFailsClosed();
    TestTerrainFullMipPayloadFingerprintsFailClosed();
    std::cout << "pco_sequence_profiles_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "pco_sequence_profiles_test: FAIL: " << error.what()
              << '\n';
    return 1;
  }
}
