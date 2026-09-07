// Focused FragmentFrontend regression. A texture-sampling draw whose complete
// half-stamp was rejected by ISP must not launch helper-only USC quads.

#include "common/functional_types.h"
#include "common/pipeline_state.h"
#include "fragment/fragment_frontend.h"

#include <systemc>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace pvrgpu::stub;

void Check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error("FragmentFrontend test failed: " + message);
}

} // namespace

int sc_main(int, char **) {
  try {
    MemoryPool pool;
    PipelineState state;
    state.width = 4;
    state.height = 2;
    state.sequence = 1;
    state.functional_case = FunctionalCase::kFillTexNearest;
    state.stage = PipelineStage::kVisibilityReady;
    state.raster_state.sample_count = 1;

    ParameterTriangle parameter;
    parameter.key.api_primitive_id = 7;
    parameter.key.submit_ordinal = 1;
    parameter.rasterizable = 1;
    state.parameter_triangles =
        StoreNewArray(pool, std::vector<ParameterTriangle>{parameter});

    FragmentCandidate rejected;
    rejected.x = 0;
    rejected.y = 0;
    rejected.primitive_id = parameter.key.api_primitive_id;
    rejected.parameter_index = 0;
    rejected.submit_ordinal = parameter.key.submit_ordinal;
    rejected.depth = 0.5F;
    rejected.sample_mask = 1;
    rejected.visibility = FragmentVisibility::kRejected;
    state.fragment_candidates =
        StoreNewArray(pool, std::vector<FragmentCandidate>{rejected});
    state.active_fragment_invocations = 0;

    const PoolHandle state_handle = pool.Allocate(sizeof(PipelineState));
    StorePipelineState(pool, state_handle, state);
    const PipelineTxn txn{state_handle, 1, state.sequence};

    sc_core::sc_fifo<PipelineTxn> input("input", 1);
    sc_core::sc_fifo<PipelineTxn> output("output", 1);
    FragmentFrontend frontend("fragment_frontend", pool);
    frontend.input(input);
    frontend.output(output);

    input.write(txn);
    sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_NS));
    sc_core::sc_start(sc_core::SC_ZERO_TIME);

    PipelineTxn completed;
    Check(output.nb_read(completed) && completed.sequence == state.sequence,
          "rejected texture draw did not complete");
    const PipelineState result = LoadPipelineState(pool, state_handle);
    Check(result.stage == PipelineStage::kFragmentsReady,
          "completion stage");
    Check(result.active_fragment_invocations == 0 &&
              result.counters.ps_invocations == 0,
          "rejected candidates launched visible invocations");
    Check(result.fragment_shader_lane_count == 0 &&
              result.fragment_groups == 0,
          "rejected-only half-stamp launched helper quads");
    Check(LoadArray<FragmentInvocation>(pool, result.fragment_invocations)
                  .empty() &&
              LoadArray<FragmentShaderLane>(pool,
                                            result.fragment_shader_lanes)
                  .empty() &&
              LoadArray<FragmentQuad>(pool, result.fragment_quads).empty(),
          "rejected-only texture payload is not empty");

    ReleaseFunctionalPayloads(pool, result);
    pool.Release(state_handle);

    PipelineState multisample;
    multisample.width = multisample.height = 1;
    multisample.sequence = 2;
    multisample.functional_case = FunctionalCase::kFillTexNearest;
    multisample.stage = PipelineStage::kVisibilityReady;
    multisample.raster_state.sample_count = 16;
    multisample.active_fragment_invocations = 2;
    std::vector<ParameterTriangle> parameters(2, parameter);
    parameters[1].key.api_primitive_id = 8;
    parameters[1].key.submit_ordinal = 2;
    std::vector<FragmentCandidate> visible(2, rejected);
    for (std::uint32_t primitive = 0; primitive < 2; ++primitive) {
      visible[primitive].primitive_id = parameters[primitive].key.api_primitive_id;
      visible[primitive].parameter_index = primitive;
      visible[primitive].submit_ordinal = parameters[primitive].key.submit_ordinal;
      visible[primitive].sample_mask = primitive == 0 ? 0x00ffU : 0xff00U;
      visible[primitive].visibility = FragmentVisibility::kVisible;
    }
    multisample.parameter_triangles = StoreNewArray(pool, parameters);
    multisample.fragment_candidates = StoreNewArray(pool, visible);
    const PoolHandle msaa_handle = pool.Allocate(sizeof(PipelineState));
    StorePipelineState(pool, msaa_handle, multisample);
    input.write(PipelineTxn{msaa_handle, 2, multisample.sequence});
    sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_NS));
    sc_core::sc_start(sc_core::SC_ZERO_TIME);
    Check(output.nb_read(completed) && completed.sequence == 2,
          "disjoint multisample owners did not complete");
    const PipelineState msaa_result = LoadPipelineState(pool, msaa_handle);
    const auto msaa_invocations =
        LoadArray<FragmentInvocation>(pool, msaa_result.fragment_invocations);
    const auto msaa_lanes =
        LoadArray<FragmentShaderLane>(pool, msaa_result.fragment_shader_lanes);
    Check(msaa_invocations.size() == 2 && msaa_lanes.size() == 2 &&
              msaa_result.counters.ps_invocations == 2,
          "MSAA frontend did not preserve pixel-frequency shading");
    for (std::size_t primitive = 0; primitive < 2; ++primitive) {
      Check(msaa_invocations[primitive].sample_mask == visible[primitive].sample_mask &&
                msaa_lanes[primitive].sample_mask == visible[primitive].sample_mask,
            "MSAA frontend lost 16-bit sample coverage");
    }
    ReleaseFunctionalPayloads(pool, msaa_result);
    pool.Release(msaa_handle);
    Check(pool.bytes_in_flight() == 0 &&
              pool.allocations() == pool.releases(),
          "MemoryPool balance");
    std::cout << "fragment_frontend_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "fragment_frontend_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
