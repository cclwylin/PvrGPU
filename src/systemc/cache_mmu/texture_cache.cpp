// TCU（Texture Cache Unit）SystemC controller implementation. FIFO arrival
// wakes the SC_THREAD; one timed completion event represents controller service
// without a clock-driven loop. CacheArray owns cache-line data/tag/LRU state
// and MemoryPool owns bulk texture payload bytes.
#include "cache_mmu/texture_cache.h"

#include "common/functional_types.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pvrgpu::stub {
namespace {

inline constexpr std::uint64_t kTextureCacheAccessCycles = 1;
inline constexpr std::size_t kTextureBatchScratchBytes = 16;

void AddChecked(std::uint64_t &target, std::uint64_t amount,
                const char *field) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - target)
    throw std::overflow_error(std::string("TCU counter overflow: ") + field);
  target += amount;
}

bool SamePipeline(const PipelineTxn &left, const PipelineTxn &right) {
  return left.frame == right.frame && left.sequence == right.sequence &&
         left.state.slot == right.state.slot &&
         left.state.generation == right.state.generation;
}

bool SameHandle(PoolHandle left, PoolHandle right) {
  return left.slot == right.slot && left.generation == right.generation;
}

std::size_t TextureResponseRouteIndex(MemoryResponseRoute route) {
  switch (route) {
  case MemoryResponseRoute::kTextureVertex:
    return 0;
  case MemoryResponseRoute::kTextureFragment:
    return 1;
  case MemoryResponseRoute::kTextureCompute:
    return 2;
  case MemoryResponseRoute::kTextureGeometry:
    return 3;
  case MemoryResponseRoute::kTextureTessellationControl:
    return 4;
  case MemoryResponseRoute::kTextureTessellationEvaluation:
    return 5;
  case MemoryResponseRoute::kNone:
    break;
  }
  throw std::runtime_error("TCU batch has no shader-stage response route");
}

PipelineStage PendingStageForRoute(MemoryResponseRoute route) {
  switch (route) {
  case MemoryResponseRoute::kTextureVertex:
    return PipelineStage::kVertexTexturePending;
  case MemoryResponseRoute::kTextureFragment:
    return PipelineStage::kFragmentTexturePending;
  case MemoryResponseRoute::kTextureCompute:
    return PipelineStage::kComputeTexturePending;
  case MemoryResponseRoute::kTextureGeometry:
    return PipelineStage::kGeometryTexturePending;
  case MemoryResponseRoute::kTextureTessellationControl:
    return PipelineStage::kTessellationControlTexturePending;
  case MemoryResponseRoute::kTextureTessellationEvaluation:
    return PipelineStage::kTessellationEvaluationTexturePending;
  case MemoryResponseRoute::kNone:
    break;
  }
  throw std::runtime_error("TCU batch has no shader-stage response route");
}

std::uint64_t SampleDelayCycles(const CacheStats &tcu_delta,
                                const MemoryAccessStats &lower_stats) {
  std::uint64_t cycles =
      tcu_delta.line_accesses * kTextureCacheAccessCycles;
  AddChecked(cycles, MemoryAccessDelayCycles(lower_stats), "sample_delay");
  return cycles;
}

std::uint64_t AccountSample(PipelineState &state, const CacheStats &tcu_delta,
                            const MemoryAccessStats &lower_stats,
                            bool integrated_memory) {
  AddChecked(state.counters.tcu_line_accesses, tcu_delta.line_accesses,
             "line_accesses");
  AddChecked(state.counters.tcu_read_accesses, tcu_delta.read_accesses,
             "read_accesses");
  AddChecked(state.counters.tcu_hits, tcu_delta.hits, "hits");
  AddChecked(state.counters.tcu_misses, tcu_delta.misses, "misses");
  AddChecked(state.counters.tcu_evictions, tcu_delta.evictions, "evictions");
  AddChecked(state.counters.tcu_writebacks, tcu_delta.writebacks,
             "writebacks");
  AddChecked(state.counters.tcu_bypassed, tcu_delta.bypassed, "bypassed");
  const std::uint64_t tcu_cycles =
      tcu_delta.line_accesses * kTextureCacheAccessCycles;
  AddChecked(state.counters.tcu_cycles, tcu_cycles, "cycles");

  std::uint64_t wait_cycles = tcu_cycles;
  if (integrated_memory) {
    ApplyMemoryAccessStats(state.counters, lower_stats);
    AddChecked(wait_cycles, MemoryAccessDelayCycles(lower_stats),
               "wait_cycles");
    AddChecked(state.counters.texture_cycles, wait_cycles, "texture_cycles");
    switch (state.stage) {
    case PipelineStage::kFragmentTexturePending:
      AddChecked(state.counters.renderer_cycles, wait_cycles,
                 "renderer_cycles");
      break;
    case PipelineStage::kVertexTexturePending:
    case PipelineStage::kGeometryTexturePending:
    case PipelineStage::kTessellationControlTexturePending:
    case PipelineStage::kTessellationEvaluationTexturePending:
      AddChecked(state.counters.tiler_cycles, wait_cycles, "tiler_cycles");
      break;
    case PipelineStage::kComputeTexturePending:
      break;
    default:
      throw std::runtime_error("TCU received texture work at an invalid stage");
    }
  } else {
    // The legacy fully event-driven lower chain accounts its own latency in
    // renderer_cycles. Preserve that contract for standalone cache tests.
    AddChecked(state.counters.renderer_cycles, tcu_cycles, "renderer_cycles");
  }
  return wait_cycles;
}

std::size_t ValidateTxn(const MemoryTxn &txn, const MemoryPool &pool) {
  if (txn.client != MemoryClient::kTextureCache)
    throw std::invalid_argument(
        "TextureCache received a transaction for another cache client");
  if (txn.batch_control != MemoryBatchControl::kNone)
    throw std::invalid_argument(
        "TextureCache generic port received a batched transaction");
  if (txn.bytes == 0)
    throw std::invalid_argument(
        "TextureCache received a zero-byte transaction");
  if (txn.bytes > std::numeric_limits<std::size_t>::max())
    throw std::overflow_error("TextureCache transaction is too large");

  switch (txn.operation) {
  case MemoryOperation::kRead:
    break;
  case MemoryOperation::kWrite: {
    if (!HasPoolHandle(txn.payload))
      throw std::invalid_argument(
          "TextureCache write transaction has no MemoryPool payload");
    const auto &payload = pool.Read(txn.payload);
    if (txn.payload_format == MemoryPayloadFormat::kLinearBytes &&
        payload.size() < txn.bytes) {
      throw std::invalid_argument(
          "TextureCache linear write payload is shorter than transaction");
    }
    if (txn.payload_format == MemoryPayloadFormat::kCacheLineWrites &&
        (payload.empty() || payload.size() % sizeof(DramLineWrite) != 0)) {
      throw std::invalid_argument(
          "TextureCache cache-line payload is not a DramLineWrite array");
    }
    break;
  }
  default:
    throw std::invalid_argument("TextureCache received an invalid operation");
  }

  switch (txn.payload_format) {
  case MemoryPayloadFormat::kLinearBytes:
  case MemoryPayloadFormat::kCacheLineWrites:
    break;
  default:
    throw std::invalid_argument(
        "TextureCache received an invalid payload format");
  }
  return static_cast<std::size_t>(txn.bytes);
}

} // namespace

TextureCache::TextureCache(sc_core::sc_module_name name, MemoryPool &pool,
                           bool cache_bypass, GpuMemorySystem *memory)
    : sc_module(name), pool_(pool), memory_(memory),
      cache_(TcuCacheConfig(), cache_bypass) {
  if (memory_) {
    memory_->SetTextureCacheInvalidator(
        [this](std::uint64_t address, std::size_t bytes) {
          InvalidateRange(address, bytes);
        });
  }
  SC_THREAD(Run);
  SC_THREAD(SampleRun);
}

TextureCache::~TextureCache() {
  if (memory_)
    memory_->SetTextureCacheInvalidator({});
}

std::uint64_t TextureCache::SetCacheBypass(bool bypass) {
  return cache_.SetBypass(bypass);
}

void TextureCache::InvalidateRange(std::uint64_t address, std::size_t bytes) {
  (void)cache_.InvalidateRange(address, bytes);
}

void TextureCache::Run() {
  while (true) {
    const MemoryTxn txn = input.read();
    const std::size_t bytes = ValidateTxn(txn, pool_);
    last_delta_ = cache_.AccessRange(txn.address, bytes,
                                     txn.operation == MemoryOperation::kWrite);
    WaitForCycles(kTextureCacheAccessCycles);
    output.write(txn);
  }
}

void TextureCache::SampleRun() {
  if (sample_input.size() == 0 || sample_output.size() == 0)
    return;
  if (!memory_ && (lower_request.size() == 0 || lower_response.size() == 0))
    return;
  const std::size_t line_bytes = cache_.config().line_size_bytes;
  while (true) {
    const MemoryTxn request = sample_input->read();
    if (memory_ && memory_->mode() != MemoryMode::kCache) {
      throw std::runtime_error(
          "TCU cached texture path requires cache memory mode");
    }
    const bool batch_member =
        request.batch_control == MemoryBatchControl::kTextureMember;
    const bool batch_end =
        request.batch_control == MemoryBatchControl::kTextureEnd;
    if (!batch_member && !batch_end &&
        request.batch_control != MemoryBatchControl::kNone) {
      throw std::runtime_error("TCU received an invalid batch control");
    }
    if (batch_end) {
      if (!memory_ || request.client != MemoryClient::kTextureCache ||
          request.operation != MemoryOperation::kRead ||
          request.payload_format != MemoryPayloadFormat::kLinearBytes ||
          request.address != 0 || request.bytes != 0 ||
          !HasPoolHandle(request.payload) ||
          pool_.Read(request.payload).size() != kTextureBatchScratchBytes) {
        throw std::runtime_error("TCU received an invalid sample batch end");
      }
      const std::size_t route_index =
          TextureResponseRouteIndex(request.response_route);
      PendingSampleBatch &pending = pending_sample_batches_[route_index];
      if (pending.active && !SamePipeline(pending.pipeline, request.pipeline)) {
        throw std::runtime_error("TCU sample batch end identity mismatch");
      }
      if ((pending.active &&
           (!SameHandle(pending.scratch, request.payload) ||
            pending.member_count != request.request_id)) ||
          (!pending.active && request.request_id != 0)) {
        throw std::runtime_error("TCU sample batch end token/count mismatch");
      }

      const CacheStats tcu_delta = pending.active ? pending.tcu : CacheStats{};
      const MemoryAccessStats lower_stats =
          pending.active ? pending.lower : MemoryAccessStats{};
      const std::uint64_t service_cycles =
          pending.active ? pending.service_cycles : 0;
      PipelineState state = LoadPipelineState(pool_, request.pipeline.state);
      if (state.stage != PendingStageForRoute(request.response_route)) {
        throw std::runtime_error("TCU batch route does not match pipeline stage");
      }
      const std::uint64_t wait_cycles =
          AccountSample(state, tcu_delta, lower_stats, true);
      if (wait_cycles != service_cycles) {
        throw std::logic_error("TCU batch service accounting mismatch");
      }
      StorePipelineState(pool_, request.pipeline.state, state);
      pending = {};
      last_delta_ = tcu_delta;
      // Members return data immediately so the TPU can keep issuing taps, but
      // every member reserved its latency on one global ordered TCU service
      // timeline. An end transaction is a FIFO fence: it cannot acknowledge
      // before any cache/memory operation that arrived ahead of it, including
      // work from another shader stage.
      const sc_core::sc_time now = sc_core::sc_time_stamp();
      if (sample_service_ready_ > now)
        sc_core::wait(sample_service_ready_ - now);
      sample_output->write(request);
      continue;
    }

    const bool supported_bytes =
        request.bytes == 1 || request.bytes == 2 || request.bytes == 4 ||
        request.bytes == 8 || request.bytes == 16;
    if (request.client != MemoryClient::kTextureCache ||
        request.operation != MemoryOperation::kRead ||
        request.payload_format != MemoryPayloadFormat::kLinearBytes ||
        !supported_bytes ||
        (batch_member
             ? (!memory_ ||
                request.response_route == MemoryResponseRoute::kNone ||
                !HasPoolHandle(request.payload) ||
                pool_.Read(request.payload).size() !=
                    kTextureBatchScratchBytes)
             : HasPoolHandle(request.payload))) {
      throw std::runtime_error("TCU received an invalid texel read request");
    }
    if (request.bytes - 1 >
        std::numeric_limits<std::uint64_t>::max() - request.address) {
      throw std::overflow_error("TCU texel read address overflow");
    }
    PendingSampleBatch *pending_batch = nullptr;
    if (batch_member) {
      pending_batch = &pending_sample_batches_[TextureResponseRouteIndex(
          request.response_route)];
      if (!pending_batch->active) {
        pending_batch->active = true;
        pending_batch->pipeline = request.pipeline;
        pending_batch->scratch = request.payload;
      } else if (!SamePipeline(pending_batch->pipeline, request.pipeline)) {
        throw std::runtime_error("TCU interleaved two batches on one route");
      } else if (!SameHandle(pending_batch->scratch, request.payload)) {
        throw std::runtime_error("TCU sample batch scratch token changed");
      }
    } else {
      if (request.response_route != MemoryResponseRoute::kNone)
        (void)TextureResponseRouteIndex(request.response_route);
      if (std::any_of(pending_sample_batches_.begin(),
                      pending_sample_batches_.end(),
                      [](const PendingSampleBatch &pending) {
                        return pending.active;
                      })) {
        throw std::runtime_error(
            "TCU legacy request interleaved with an active sample batch");
      }
    }
    MemoryAccessStats lower_stats;
    const CacheLineRead lower_read = [&](std::uint64_t address,
                                         std::size_t bytes) {
      if (memory_) {
        CacheLineData data(bytes);
        lower_stats += memory_->ReadInto(address, data.data(), data.size(),
                                         MemoryClient::kTextureCache);
        return data;
      }
      MemoryTxn miss = request;
      miss.address = address;
      miss.bytes = bytes;
      miss.payload = {};
      miss.batch_control = MemoryBatchControl::kNone;
      lower_request->write(miss);
      const MemoryTxn response = lower_response->read();
      if (response.pipeline.frame != request.pipeline.frame ||
          response.pipeline.sequence != request.pipeline.sequence ||
          response.pipeline.state.slot != request.pipeline.state.slot ||
          response.pipeline.state.generation !=
              request.pipeline.state.generation ||
          response.request_id != request.request_id ||
          response.address != address || response.bytes != bytes ||
          response.client != MemoryClient::kTextureCache ||
          response.operation != MemoryOperation::kRead ||
          response.payload_format != MemoryPayloadFormat::kLinearBytes ||
          !HasPoolHandle(response.payload)) {
        throw std::runtime_error("TCU received an invalid SLC fill response");
      }
      CacheLineData data =
          LoadArray<std::uint8_t>(pool_, response.payload);
      pool_.Release(response.payload);
      if (data.size() != bytes)
        throw std::runtime_error("TCU SLC fill byte count mismatch");
      return data;
    };

    const std::size_t texel_bytes = static_cast<std::size_t>(request.bytes);
    std::array<std::uint8_t, kTextureBatchScratchBytes> texel{};
    CacheStats tcu_delta;
    std::size_t copied = 0;
    while (copied < texel_bytes) {
      const std::uint64_t current = request.address + copied;
      const std::uint64_t line_address = current - current % line_bytes;
      const std::size_t line_offset =
          static_cast<std::size_t>(current - line_address);
      const std::size_t chunk =
          std::min(texel_bytes - copied, line_bytes - line_offset);
      const CacheLineAccess access = cache_.ReadLineInto(
          line_address, line_offset, texel.data() + copied, chunk, lower_read,
          {});
      tcu_delta += access.delta;
      copied += chunk;
    }
    last_delta_ = tcu_delta;

    MemoryTxn response = request;
    if (batch_member) {
      const std::uint64_t member_cycles =
          SampleDelayCycles(tcu_delta, lower_stats);
      AddChecked(pending_batch->member_count, 1, "batch_member_count");
      AddChecked(pending_batch->service_cycles, member_cycles,
                 "batch_service_cycles");
      pending_batch->tcu += tcu_delta;
      pending_batch->lower += lower_stats;
      sample_service_ready_ =
          std::max(sample_service_ready_, sc_core::sc_time_stamp());
      sample_service_ready_ += sc_core::sc_time(
          static_cast<double>(member_cycles), sc_core::SC_NS);
      std::vector<std::uint8_t> &scratch = pool_.Write(request.payload);
      std::copy_n(texel.begin(), texel_bytes, scratch.begin());
      // The caller owns and reuses this handle for every member in the batch.
      // No per-tap allocation, state copy, or wait is performed here.
      sample_output->write(response);
      continue;
    }

    PipelineState state = LoadPipelineState(pool_, request.pipeline.state);
    const std::uint64_t wait_cycles =
        AccountSample(state, tcu_delta, lower_stats, memory_ != nullptr);
    StorePipelineState(pool_, request.pipeline.state, state);
    response.payload = pool_.Allocate(texel_bytes);
    std::copy_n(texel.begin(), texel_bytes,
                pool_.Write(response.payload).begin());
    WaitForCycles(wait_cycles);
    sample_output->write(response);
  }
}

} // namespace pvrgpu::stub
