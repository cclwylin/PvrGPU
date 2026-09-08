#pragma once

#include "rdc_runner/native_report.h"

#include <string>

namespace pvrgpu::rdc {

struct ReplayRangeAudit {
  bool ok = false;
  bool native_work = false;
  NativeReport report;
  std::string error;
};

// The caller must supply ONLY bytes produced within the same replay scope,
// bounded after synchronization. This parser cannot establish file freshness
// or associate previously saved bytes with a new execution on its own.
// A valid no-work scope leaves report empty, rather than manufacturing counters.
// Generic API submissions include clears: they are not application draw counts,
// and several submissions may legitimately produce one coalesced model report.
// Explicit accepted graphics/nonzero decoded compute diagnostics must have a
// completion of the same work class; mere draw observations are not acceptance.
ReplayRangeAudit AuditNativeReplayRange(const std::string &model_jsonl,
                                       const std::string &driver_events);

// Empty means no observed driver error; it is NOT proof of completed native
// work. Use AuditNativeReplayRange for that. In raw-transfer scopes, also refuse
// any draw/dispatch/native API event, even a zero-count draw. State bindings,
// synchronization and raw resource transfers remain legal. This does not replace
// ValidateInitialCopyAudit's stricter initial-restoration allowlist/flush gate.
std::string AuditNativeDriverEvents(const std::string &driver_events,
                                    bool raw_transfers_only = false);

}  // namespace pvrgpu::rdc
