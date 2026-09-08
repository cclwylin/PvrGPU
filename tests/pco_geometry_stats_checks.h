// SPDX-License-Identifier: MIT
#pragma once

#include "pvrgpu_systemc_api.h"

#include <array>
#include <cstdint>
#include <string>

// Invoke only after submission of genuine native PCO work and before pixel
// readback. The first statistics request itself must execute that submission;
// later readbacks must reuse its result without executing/counting it again.
template <typename Check>
void VerifyGeometryStats(std::uint64_t generation,
                         std::uint64_t expected_invocations,
                         std::uint64_t expected_primitives,
                         Check check,
                         std::uint64_t expected_inputs = 1) {
  std::array<char, 1024> error{};
  pvrgpu_systemc_graphics_stats stats{};
  stats.version = PVRGPU_SYSTEMC_API_VERSION;
  stats.submission_generation = generation;
  check(pvrgpu_systemc_flush_graphics_stats(&stats, error.data(), error.size()) == 0,
        std::string("native GS statistics flush: ") + error.data());
  check(stats.submission_generation == generation && stats.physical_submissions == 1,
        "statistics belong to exactly the requested native submission");
  check(stats.ia_primitives == expected_inputs &&
            stats.gs_invocations == expected_invocations &&
            stats.gs_primitives == expected_primitives &&
            stats.primitives_generated == expected_primitives,
        "query uses complete native GS output, including zero, not input count");
  auto repeated = stats;
  repeated.physical_submissions = repeated.primitives_generated = UINT64_MAX;
  check(pvrgpu_systemc_flush_graphics_stats(&repeated, error.data(), error.size()) == 0,
        "completed native statistics can be reread without execution");
  check(repeated.physical_submissions == stats.physical_submissions &&
            repeated.primitives_generated == stats.primitives_generated &&
            repeated.ia_primitives == stats.ia_primitives &&
            repeated.gs_primitives == stats.gs_primitives &&
            repeated.gs_invocations == stats.gs_invocations,
        "statistics reread is idempotent");
  auto unrelated = stats;
  unrelated.submission_generation ^= UINT64_C(0x8000000000000000);
  check(pvrgpu_systemc_flush_graphics_stats(&unrelated, error.data(), error.size()) == 2,
        "unrelated submission cannot consume another context statistics");
  auto invalid = stats;
  invalid.version = PVRGPU_SYSTEMC_API_VERSION - 1;
  check(pvrgpu_systemc_flush_graphics_stats(&invalid, error.data(), error.size()) == 2,
        "older statistics ABI is rejected");
  invalid = stats; invalid.submission_generation = 0;
  check(pvrgpu_systemc_flush_graphics_stats(&invalid, nullptr, 0) == 2,
        "zero generation is not an owned submission");
  check(pvrgpu_systemc_flush_graphics_stats(nullptr, nullptr, 0) == 2,
        "null statistics request is rejected safely");
}
