#pragma once

#include "common/pipeline_state.h"
#include <systemc>
#include <algorithm>
#include <functional>
#include <stdexcept>

namespace pvrgpu::stub {
using TessellationTextureRequestPort = sc_core::sc_port<sc_core::sc_fifo_out_if<PipelineTxn>, 0,
    sc_core::SC_ZERO_OR_MORE_BOUND>;
using TessellationTextureResponsePort = sc_core::sc_port<sc_core::sc_fifo_in_if<PipelineTxn>, 0,
    sc_core::SC_ZERO_OR_MORE_BOUND>;

inline bool SameTessellationSample(const TextureSampleRequest &a, const TextureSampleRequest &b) {
  return std::equal(std::begin(a.coordinates), std::end(a.coordinates), std::begin(b.coordinates)) &&
      std::equal(std::begin(a.texture_state), std::end(a.texture_state), std::begin(b.texture_state)) &&
      std::equal(std::begin(a.sampler_state), std::end(a.sampler_state), std::begin(b.sampler_state)) &&
      a.shader_stage == b.shader_stage && a.descriptor_set == b.descriptor_set &&
      a.coordinate_count == 2 && a.component_count == 4 && a.dimension == 2 &&
      a.binding == 0 && a.normalized == 1 && a.fcnorm == 1 && a.data_request == 0 &&
      a.shader_lane_index == 0 && a.request_id == 0 && a.quad_id == 0 && a.quad_lane == 0 && a.sample_id == 0 &&
      a.texture_address_lo == 0 && a.texture_address_hi == 0 &&
      a.sample_index == 0 && a.sample_index_present == 0 && a.gather == 0 &&
      a.spatial_offsets[0] == 0 && a.spatial_offsets[1] == 0 && a.spatial_offsets[2] == 0 &&
      a.explicit_lod == b.explicit_lod && a.explicit_lod_present == b.explicit_lod_present &&
      a.lod_bias == 0 && a.lod_bias_present == 0;
}

// A selected native task lane blocks on one real TPU transaction. The ISS
// keeps these response words pending until its following WDF; no fragment
// quad, alternate shader stage, or host texture evaluation is involved.
inline void SampleTessellationTexture(MemoryPool &pool, PipelineState &state,
    const PipelineTxn &txn, ShaderStage stage, const PcoTextureRequest &issued,
    std::uint32_t *response, TessellationTextureRequestPort &output,
    TessellationTextureResponsePort &input) {
  const bool control = stage == ShaderStage::kTessellationControl;
  if ((!control && stage != ShaderStage::kTessellationEvaluation) || !response ||
      !output.size() || !input.size() || !UsesTextureSampling(state, stage) ||
      state.stage != PipelineStage::kVertexShaded ||
      HasPoolHandle(state.texture_sample_requests) || HasPoolHandle(state.texture_sample_responses))
    throw std::runtime_error("tessellation SMP FIFO/state ownership mismatch");
  if (issued.coordinate_count != 2 || issued.component_count != 4 || issued.dimension != 2 ||
      issued.normalized != 1 || issued.fcnorm != 1 || issued.binding || issued.data_request ||
      issued.coordinates[2] || issued.texture_address_lo || issued.texture_address_hi ||
      issued.sample_index || issued.sample_index_present || issued.gather ||
      issued.spatial_offsets[0] || issued.spatial_offsets[1] || issued.spatial_offsets[2] ||
      issued.lod_bias || issued.lod_bias_present || issued.explicit_lod_present > 1 ||
      (!issued.explicit_lod_present && issued.explicit_lod))
    throw std::runtime_error("unsupported tessellation SMP request class");
  TextureSampleRequest request;
  request.shader_stage = stage;
  std::copy_n(issued.coordinates.begin(), 3, request.coordinates);
  std::copy_n(issued.texture_state.begin(), 4, request.texture_state);
  std::copy_n(issued.sampler_state.begin(), 4, request.sampler_state);
  request.coordinate_count = 2; request.component_count = 4; request.dimension = 2;
  request.normalized = 1; request.fcnorm = 1;
  request.descriptor_set = issued.descriptor_set;
  request.explicit_lod = issued.explicit_lod;
  request.explicit_lod_present = issued.explicit_lod_present;
  const auto pending = control ? PipelineStage::kTessellationControlTexturePending : PipelineStage::kTessellationEvaluationTexturePending;
  const auto ready = control ? PipelineStage::kTessellationControlTextureSamplesReady : PipelineStage::kTessellationEvaluationTextureSamplesReady;
  state.texture_sample_requests = StoreNewArray(pool, std::vector<TextureSampleRequest>{request});
  state.stage = pending;
  StorePipelineState(pool, txn.state, state);
  output->write(txn);
  const auto completion = input->read();
  if (completion.state.slot != txn.state.slot || completion.state.generation != txn.state.generation ||
      completion.sequence != txn.sequence || completion.frame != txn.frame)
    throw std::runtime_error("tessellation SMP completion identity mismatch");
  state = LoadPipelineState(pool, txn.state);
  RequireStage(state.stage, ready, "tessellation SMP");
  const auto requests = LoadArray<TextureSampleRequest>(pool, state.texture_sample_requests);
  const auto responses = LoadArray<TextureSampleResponse>(pool, state.texture_sample_responses);
  if (requests.size() != 1 || responses.size() != 1 || !SameTessellationSample(requests[0], request) ||
      responses[0].shader_stage != stage || responses[0].shader_lane_index || responses[0].request_id)
    throw std::runtime_error("tessellation SMP completion payload mismatch");
  std::copy_n(responses[0].rgba, 4, response);
  pool.Release(state.texture_sample_requests); state.texture_sample_requests = {};
  pool.Release(state.texture_sample_responses); state.texture_sample_responses = {};
  state.stage = PipelineStage::kVertexShaded;
}
} // namespace pvrgpu::stub
