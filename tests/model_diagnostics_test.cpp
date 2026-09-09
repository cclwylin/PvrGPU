#include "common/diagnostics.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

int main() {
  using pvrgpu::stub::DiagnosticEnvironment;
  using pvrgpu::stub::kDiagnosticsEnabled;
  constexpr const char *name = "PVRGPU_DIAGNOSTICS_UNIT_TEST_ONLY";
  if (unsetenv(name) != 0 || DiagnosticEnvironment(name) != nullptr) {
    std::cerr << "absent diagnostic variable did not return null\n";
    return 1;
  }
  for (const char *value : {"", "1", "invalid-coordinate"}) {
    if (setenv(name, value, 1) != 0) {
      std::cerr << "could not set test environment\n";
      return 1;
    }
    const char *observed = DiagnosticEnvironment(name);
    if (kDiagnosticsEnabled
            ? observed == nullptr || std::strcmp(observed, value) != 0
            : observed != nullptr) {
      std::cerr << "diagnostic compile-time switch changed its contract\n";
      return 1;
    }
  }
  if (unsetenv(name) != 0)
    return 1;
#if !PVRGPU_ENABLE_DIAGNOSTICS
  static_assert(DiagnosticEnvironment("any-name") == nullptr);
#endif
  std::cout << "PASS diagnostics=" << kDiagnosticsEnabled << '\n';
  return 0;
}
