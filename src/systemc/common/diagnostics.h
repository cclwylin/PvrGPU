#pragma once

// This switch only removes opt-in diagnostic observations. It must never
// guard modeled work, dynamic counters, input validation, or error handling.
// Standalone builds retain the existing diagnostics unless explicitly disabled.
#ifndef PVRGPU_ENABLE_DIAGNOSTICS
#define PVRGPU_ENABLE_DIAGNOSTICS 1
#endif

#if PVRGPU_ENABLE_DIAGNOSTICS != 0 && PVRGPU_ENABLE_DIAGNOSTICS != 1
#error "PVRGPU_ENABLE_DIAGNOSTICS must be 0 or 1"
#endif

#if PVRGPU_ENABLE_DIAGNOSTICS
#include <cstdlib>
#endif

namespace pvrgpu::stub {

inline constexpr bool kDiagnosticsEnabled = PVRGPU_ENABLE_DIAGNOSTICS != 0;

#if PVRGPU_ENABLE_DIAGNOSTICS
inline const char *DiagnosticEnvironment(const char *name) {
  return std::getenv(name);
}
#else
// A constant null result lets the optimizer erase diagnostic parsing,
// allocation, formatting, and output together with the environment lookup.
inline constexpr const char *DiagnosticEnvironment(const char *name) {
  (void)name;
  return nullptr;
}
#endif

} // namespace pvrgpu::stub
