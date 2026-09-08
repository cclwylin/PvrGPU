#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace pvrgpu::stub {

// Raw counters collected from completed physical SystemC transactions. This
// is separate from the JSON presentation/normalization of captured workloads.
struct ModelGraphicsStats {
  std::uint64_t physical_submissions = 0;
  std::uint64_t primitives_generated = 0;
  std::uint64_t ia_primitives = 0;
  std::uint64_t gs_primitives = 0;
  std::uint64_t gs_invocations = 0;
  std::uint64_t stream_output_primitives_written = 0;
  std::uint64_t stream_output_primitives_storage_needed = 0;

  void Add(bool geometry_enabled, std::uint64_t input_primitives,
           std::uint64_t output_primitives, std::uint64_t invocations,
           bool tessellation_enabled = false, std::uint64_t tessellation_primitives = 0,
           std::uint64_t stream_output_written = 0, std::uint64_t stream_output_needed = 0) {
    const auto checked = [](std::uint64_t a, std::uint64_t b) {
      if (b > std::numeric_limits<std::uint64_t>::max() - a)
        throw std::overflow_error("native graphics statistics overflow");
      return a + b;
    };
    // Preflight every sum before publishing any of this transaction.
    ModelGraphicsStats next;
    next.physical_submissions = checked(physical_submissions, 1);
    next.primitives_generated = checked(
        primitives_generated, geometry_enabled ? output_primitives
            : tessellation_enabled ? tessellation_primitives : input_primitives);
    next.ia_primitives = checked(ia_primitives, input_primitives);
    next.gs_primitives = checked(gs_primitives, output_primitives);
    next.gs_invocations = checked(gs_invocations, invocations);
    next.stream_output_primitives_written = checked(stream_output_primitives_written,
                                                    stream_output_written);
    next.stream_output_primitives_storage_needed = checked(stream_output_primitives_storage_needed,
                                                           stream_output_needed);
    *this = next;
  }
};

} // namespace pvrgpu::stub
