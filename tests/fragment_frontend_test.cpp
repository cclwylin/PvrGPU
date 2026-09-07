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
