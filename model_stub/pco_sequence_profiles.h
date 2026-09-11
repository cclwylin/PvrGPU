#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace pvrgpu::stub {

struct Options;

// Strict semantic gate for native multi-draw PCO profiles.  This function is
// intentionally independent of Submitter and the API bridge: callers may use
// it after ownership/copy validation and before allocating any model state.
// On rejection, `error` receives a stable, human-readable reason when it is
// non-null.
bool DriverPcoSequenceSupported(const Options &options, std::string *error);

// Terrain's four immutable D3 resources are captured as complete 10-level
// mip chains.  Keep their profile fingerprint gate directly testable without
// embedding four 1.4 MiB payload fixtures in the model unit test.
bool DriverPcoTerrainExternalPayloadHashMatches(std::size_t texture_index,
                                                std::uint64_t payload_hash);

// Terrain D4/D7/D8 embed resolution-dependent texel steps.  Keep the exact
// post-lowering binary fingerprints independently testable so an 800x600
// source cannot silently regress to the 80x60 program.
bool DriverPcoTerrainFragmentBinaryHashMatches(
    std::uint32_t width, std::uint32_t height, std::size_t draw_index,
    std::uint64_t binary_hash);

// Mesa's application-visible pipeline/texture queries are semantic counters,
// not the native backend's physical clip and filtered-texel work.  Return the
// calibrated query values only for an immutable, fully validated Terrain draw
// profile; physical USC/texture/cache counters remain untouched.
bool DriverPcoTerrainApiCounters(std::uint32_t width, std::uint32_t height,
                                std::size_t draw_index,
                                std::uint64_t *clip_primitives,
                                std::uint64_t *texel_fetches);

}  // namespace pvrgpu::stub
