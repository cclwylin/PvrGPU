#include "geometry/tessellator.h"

#include "common/tessellation_state.h"
#include "memory/gpu_memory_system.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pvrgpu::stub {

namespace {
class OwnedPayload {
 public:
  OwnedPayload(MemoryPool &pool, std::size_t bytes)
      : pool_(pool), handle_(pool.Allocate(bytes)) {}
  ~OwnedPayload() { if (HasPoolHandle(handle_)) pool_.Release(handle_); }
  OwnedPayload(const OwnedPayload &) = delete;
  OwnedPayload &operator=(const OwnedPayload &) = delete;
  template <class T> T *data() {
    return reinterpret_cast<T *>(pool_.Write(handle_).data());
  }
  PoolHandle handle() const { return handle_; }
  PoolHandle Publish() { const auto result = handle_; handle_ = {}; return result; }
 private:
  MemoryPool &pool_;
  PoolHandle handle_;
};
}

Tessellator::Tessellator(sc_core::sc_module_name name, MemoryPool &pool,
                         GpuMemorySystem *memory)
    : sc_module(name), pool_(pool), memory_(memory) { SC_THREAD(Run); }

void Tessellator::Execute(PipelineState &state) {
  auto records = LoadArray<TessellationState>(pool_, state.tessellation_state);
  if (records.size() != 1 || state.stage != PipelineStage::kVertexShaded ||
      !memory_ || state.memory_mode != memory_->mode())
    throw std::runtime_error("tessellator pipeline/state/memory contract is invalid");
  auto &tessellation = records[0];
  if (tessellation.phase != TessellationPhase::kControlComplete ||
      !tessellation.output_address || tessellation.output_address % 4 ||
      !tessellation.domain_address || tessellation.domain_address % 8 ||
      tessellation.domain_address > std::numeric_limits<std::uint64_t>::max() -
          kTessellationMaxDrawPoints * sizeof(TessellationDomainPoint) ||
      HasPoolHandle(tessellation.domain_points) ||
      HasPoolHandle(tessellation.domain_indices))
    throw std::runtime_error("tessellator phase/address/ownership contract is invalid");
  auto patches = LoadArray<TessellationPatch>(pool_, tessellation.patches);
  if (patches.size() > kTessellationMaxPatches)
    throw std::runtime_error("tessellator patch count exceeds draw bound");
  const auto output_bytes = patches.size() * kTessellationPatchAddressStride;
  const auto domain_bytes = kTessellationMaxDrawPoints * sizeof(TessellationDomainPoint);
  if (tessellation.output_address > std::numeric_limits<std::uint64_t>::max() - output_bytes ||
      (tessellation.output_address < tessellation.domain_address + domain_bytes &&
       tessellation.domain_address < tessellation.output_address + output_bytes))
    throw std::runtime_error("tessellator level/domain regions overlap or overflow");
  OwnedPayload patch_points(pool_, kTessellationMaxPoints * sizeof(TessellationDomainPoint));
  OwnedPayload patch_indices(pool_, kTessellationMaxIndices * sizeof(std::uint32_t));
  const TessellationStorage storage{patch_points.data<TessellationDomainPoint>(),
      kTessellationMaxPoints, patch_indices.data<std::uint32_t>(), kTessellationMaxIndices};
  std::vector<TessellationDomainPoint> points;
  std::vector<std::uint32_t> indices;
  for (std::size_t patch_index = 0; patch_index < patches.size(); ++patch_index) {
    auto &patch = patches[patch_index];
    if (!patch.output_address || patch.output_address % 4 ||
        patch.output_address != tessellation.output_address + patch_index * kTessellationPatchAddressStride ||
        patch.output_address > std::numeric_limits<std::uint64_t>::max() - 24 ||
        patch.point_count || patch.index_count || patch.domain_address)
      throw std::runtime_error("tessellator patch address/empty-domain contract is invalid");
    TessellationRequest request;
    request.domain = tessellation.domain;
    request.spacing = tessellation.spacing;
    request.clockwise = tessellation.clockwise;
    request.point_mode = tessellation.point_mode;
    const auto levels = memory_->Read(patch.output_address, 24, MemoryClient::kTessellator);
    if (levels.data.size() != 24)
      throw std::runtime_error("tessellator level read completion size is invalid");
    std::memcpy(request.outer, levels.data.data(), 16);
    std::memcpy(request.inner, levels.data.data() + 16, 8);
    ApplyMemoryAccessStats(state.counters, levels.stats);
    state.counters.tessellation_level_read_bytes += 24;
    WaitForCycles(MemoryAccessDelayCycles(levels.stats));
    TessellationResult result;
    const auto status = TessellatePatch(request, storage, result);
    if (status != TessellationStatus::kSuccess)
      throw std::runtime_error(std::string("tessellator: ") + TessellationStatusName(status));
    if (result.point_count > kTessellationMaxDrawPoints - points.size() ||
        result.index_count > kTessellationMaxDrawIndices - indices.size())
      throw std::runtime_error("tessellator generated domain exceeds draw storage bound");
    patch.point_start = static_cast<std::uint32_t>(points.size());
    patch.point_count = result.point_count;
    patch.index_start = static_cast<std::uint32_t>(indices.size());
    patch.index_count = result.index_count;
    patch.primitive_size = result.primitive_size;
    patch.domain_address = tessellation.domain_address + points.size() * sizeof(TessellationDomainPoint);
    if (result.point_count) {
      const auto bytes = result.point_count * sizeof(TessellationDomainPoint);
      const auto write = memory_->Write(patch.domain_address, storage.points,
                                        bytes, MemoryClient::kTessellator);
      ApplyMemoryAccessStats(state.counters, write);
      state.counters.tessellation_domain_write_bytes += bytes;
      WaitForCycles(MemoryAccessDelayCycles(write));
      points.insert(points.end(), storage.points, storage.points + result.point_count);
      indices.insert(indices.end(), storage.indices, storage.indices + result.index_count);
    }
    ++state.counters.tessellation_patches;
    state.counters.tessellation_primitives += result.index_count / result.primitive_size;
  }
  OwnedPayload all_points(pool_, points.size() * sizeof(TessellationDomainPoint));
  OwnedPayload all_indices(pool_, indices.size() * sizeof(std::uint32_t));
  if (!points.empty()) std::memcpy(all_points.data<TessellationDomainPoint>(), points.data(),
                                 points.size() * sizeof(TessellationDomainPoint));
  if (!indices.empty()) std::memcpy(all_indices.data<std::uint32_t>(), indices.data(),
                                  indices.size() * sizeof(std::uint32_t));
  StoreArray(pool_, tessellation.patches, patches);
  tessellation.domain_points = all_points.handle();
  tessellation.domain_indices = all_indices.handle();
  tessellation.phase = TessellationPhase::kDomainComplete;
  StoreArray(pool_, state.tessellation_state, records);
  (void)all_points.Publish();
  (void)all_indices.Publish();
}

void Tessellator::Run() {
  for (;;) {
    PipelineTxn transaction;
    while (!input.nb_read(transaction)) wait(input.data_written_event());
    auto state = LoadPipelineState(pool_, transaction.state);
    if (HasPoolHandle(state.tessellation_state)) {
      try { Execute(state); }
      catch (...) {
        StorePipelineState(pool_, transaction.state, state);
        throw;
      }
      StorePipelineState(pool_, transaction.state, state);
    }
    while (!output.nb_write(transaction)) wait(output.data_read_event());
  }
}

}  // namespace pvrgpu::stub
