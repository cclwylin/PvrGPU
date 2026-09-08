// SPDX-License-Identifier: MIT
#include "graphics_stats.h"

#include <iostream>
#include <stdexcept>

int main() {
  unsigned checks = 0;
  const auto check = [&](bool value) {
    ++checks;
    if (!value) throw std::runtime_error("graphics statistics assertion failed");
  };
  try {
    pvrgpu::stub::ModelGraphicsStats stats;
    stats.Add(false, 8, 0, 0);
    check(stats.physical_submissions == 1 && stats.primitives_generated == 8);
    stats.Add(true, 8, 24, 8);
    check(stats.primitives_generated == 32 && stats.ia_primitives == 16 &&
          stats.gs_primitives == 24 && stats.gs_invocations == 8);
    stats.Add(true, 7, 0, 21);
    check(stats.primitives_generated == 32 && stats.ia_primitives == 23 &&
          stats.gs_invocations == 29 && stats.physical_submissions == 3);
    // Tessellation counts its generated primitives, not input patch count,
    // and never charges fixed output to the independent Geometry Shader.
    stats.Add(false, 2, 0, 0, true, 18);
    check(stats.primitives_generated == 50 && stats.ia_primitives == 25 &&
          stats.gs_primitives == 24 && stats.gs_invocations == 29 &&
          stats.physical_submissions == 4);
    stats.Add(false, 7, 0, 0, true, 0);
    check(stats.primitives_generated == 50 && stats.ia_primitives == 32 &&
          stats.gs_primitives == 24 && stats.gs_invocations == 29 &&
          stats.physical_submissions == 5);
    // If both stages execute, only final GS output enters GENERATED. A GS
    // that emits nothing must not fall back to its tessellated input count.
    stats.Add(true, 1, 3, 11, true, 128);
    check(stats.primitives_generated == 53 && stats.ia_primitives == 33 &&
          stats.gs_primitives == 27 && stats.gs_invocations == 40);
    stats.Add(true, 1, 0, 128, true, 128);
    check(stats.primitives_generated == 53 && stats.ia_primitives == 34 &&
          stats.gs_primitives == 27 && stats.gs_invocations == 168);
    // Disabled-stage metadata cannot override the real input/last stage.
    stats.Add(false, 4, 0, 0, false, UINT64_MAX);
    check(stats.primitives_generated == 57 && stats.ia_primitives == 38);
    stats.Add(true, 1, 2, 1, true, UINT64_MAX);
    check(stats.primitives_generated == 59 && stats.gs_primitives == 29 &&
          stats.gs_invocations == 169);
    // This accumulator is deliberately additive per physical submission;
    // generation-owned flush idempotence belongs at the bridge, not here.
    auto repeated = pvrgpu::stub::ModelGraphicsStats{};
    repeated.Add(false, 1, 0, 0, true, 9);
    repeated.Add(false, 1, 0, 0, true, 9);
    check(repeated.physical_submissions == 2 && repeated.primitives_generated == 18 &&
          repeated.ia_primitives == 2 && repeated.gs_primitives == 0 && repeated.gs_invocations == 0);
    const auto same = [](const auto &a, const auto &b) {
      return a.physical_submissions == b.physical_submissions &&
             a.primitives_generated == b.primitives_generated &&
             a.ia_primitives == b.ia_primitives && a.gs_primitives == b.gs_primitives &&
             a.gs_invocations == b.gs_invocations;
    };
    auto tess_overflow = pvrgpu::stub::ModelGraphicsStats{};
    tess_overflow.Add(false, 1, 0, 0, true, UINT64_MAX);
    const auto tess_before = tess_overflow;
    // Repeating a refused overflow is state-idempotent: no partial increments
    // of physical submissions, IA, or stage-local statistics can escape.
    for (unsigned retry = 0; retry < 3; ++retry) {
      bool rejected = false;
      try { tess_overflow.Add(false, 1, 0, 0, true, 1); }
      catch (const std::overflow_error &) { rejected = true; }
      check(rejected && same(tess_overflow, tess_before));
    }
    auto culled_at_maximum = tess_before;
    culled_at_maximum.Add(false, 1, 0, 0, true, 0);
    check(culled_at_maximum.primitives_generated == UINT64_MAX &&
          culled_at_maximum.physical_submissions == 2 && culled_at_maximum.ia_primitives == 2);
    // Both late IA overflow and physical-submission overflow leave the chosen
    // tessellation GENERATED contribution entirely unpublished.
    for (auto field : {&pvrgpu::stub::ModelGraphicsStats::physical_submissions,
                       &pvrgpu::stub::ModelGraphicsStats::ia_primitives}) {
      auto overflowing = repeated;
      overflowing.*field = UINT64_MAX;
      const auto before = overflowing;
      bool rejected = false;
      try { overflowing.Add(false, 1, 0, 0, true, 100); }
      catch (const std::overflow_error &) { rejected = true; }
      check(rejected && same(overflowing, before));
    }
    // Every counter is uint64 and overflow is transactional, not saturation.
    for (auto field : {&pvrgpu::stub::ModelGraphicsStats::physical_submissions,
                       &pvrgpu::stub::ModelGraphicsStats::primitives_generated,
                       &pvrgpu::stub::ModelGraphicsStats::ia_primitives,
                       &pvrgpu::stub::ModelGraphicsStats::gs_primitives,
                       &pvrgpu::stub::ModelGraphicsStats::gs_invocations}) {
      auto overflowing = stats;
      overflowing.*field = UINT64_MAX;
      const auto before = overflowing;
      bool rejected = false;
      try { overflowing.Add(true, 1, 1, 1); }
      catch (const std::overflow_error &) { rejected = true; }
      check(rejected);
      check(overflowing.physical_submissions == before.physical_submissions &&
            overflowing.primitives_generated == before.primitives_generated &&
            overflowing.ia_primitives == before.ia_primitives &&
            overflowing.gs_primitives == before.gs_primitives &&
            overflowing.gs_invocations == before.gs_invocations);
    }
    std::cout << "graphics statistics: " << checks << " checks PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
