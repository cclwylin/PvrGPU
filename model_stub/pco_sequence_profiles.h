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

}  // namespace pvrgpu::stub
