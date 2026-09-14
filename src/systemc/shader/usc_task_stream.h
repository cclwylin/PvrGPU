// USC task stream: the stage-neutral shader execution vocabulary shared by the
// vertex, geometry and fragment executors (Phase 1 of the unified shader; see
// docs/USC_TASK_STREAM_PHASE1.md). This is a plain library, not an sc_module:
// modules keep their FIFOs and own the timing waits, and use these helpers so
// issue grouping, texture request identity, stream timing and partial-render
// accounting have one definition.
#pragma once

#include "common/functional_types.h"
#include "common/reference_uarch.h"
#include "shader/pco_iss.h"

#include <cstdint>

namespace pvrgpu::stub {

struct PipelineState;

enum class UscStage : std::uint8_t { kVertex, kGeometry, kFragment };

// Lanes issued to the cluster in groups of kReferenceUarch.usc_issue_lanes.
struct UscIssuePlan {
  std::uint64_t lanes = 0;
  std::uint64_t groups = 0;
  static UscIssuePlan ForLanes(std::uint64_t lanes);
};

// Work a streaming stage performed, in the units its rates are stated in.
struct UscStreamWork {
  std::uint64_t accepted_inputs = 0;
  std::uint64_t invocations = 0;
  std::uint64_t emitted_vertices = 0;
  std::uint64_t exported_primitives = 0;
};

// Periods, in multiples of the stream period T, per unit of work.
struct UscStreamRates {
  std::uint32_t per_accepted_input = 0;
  std::uint32_t per_invocation = 0;
  std::uint32_t per_emitted_vertex = 0;
  std::uint32_t per_exported_primitive = 0;
};

// Geometry stream throughput: accept 1 primitive/T, 1 invocation/2T,
// emit 1 vertex/T, export 1 primitive/2T.
inline constexpr UscStreamRates kGeometryStreamRates = {1, 2, 1, 2};

// kGeometryStreamCapacityPrimitives, unless PVRGPU_GEOMETRY_STREAM_CAPACITY
// overrides it for partial-render equivalence validation.
std::uint64_t GeometryStreamCapacity();

std::uint64_t UscStreamCycles(const UscStreamWork &work,
                              const UscStreamRates &rates);

// The one PCO SMP -> TextureUnit request conversion. Identity fields
// (lane/request/quad) are the caller's; stage policy rejects operands that
// stage's texture path does not implement.
TextureSampleRequest MakeUscTextureRequest(UscStage stage,
                                           const PcoTextureRequest &issued);

// ---- Partial render (bounded geometry output) --------------------------------

// Folds one partial render's counters into the running total. Everything is
// additive except identities (frame, functional_frame, drawlists), gauges
// (pool bytes: maximum), the draw's final attachment commit (pixel data
// master transactions/bytes and framebuffer readback bytes: last render; the
// intermediate commits remain visible in the DRAM counters) and the
// per-submission derived total recomputed by FinalizePartialRenderCounters.
void AccumulatePartialRenderCounters(CounterTxn &total, const CounterTxn &batch);
void FinalizePartialRenderCounters(CounterTxn &counters);
CounterTxn PartialRenderCounterIdentity(const CounterTxn &counters);

// Executed (not program) shader statistics are additive across batches.
void AccumulateShaderExecutions(DrawListShaderStats &total,
                                const DrawListShaderStats &batch);

// Turns a completed partial render into the next one's initial attachment
// state: committed colour/depth become LOAD, inherited clears are dropped (they
// already landed), and every downstream intermediate payload is released so the
// raster stages see a fresh draw. The draw's original LOAD evidence was saved
// by BeginPartialRender and is restored by FinishPartialRender.
void BeginPartialRender(MemoryPool &pool, PipelineState &state);
void ChainPartialRender(MemoryPool &pool, PipelineState &state);
void FinishPartialRender(MemoryPool &pool, PipelineState &state);

}  // namespace pvrgpu::stub
