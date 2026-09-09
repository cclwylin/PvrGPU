// Focused FragmentFrontend regression. A texture-sampling draw whose complete
// 2x2 quad was rejected by ISP must not launch helper-only USC quads, even
// when the adjacent child of its 4x2 half-stamp remains visible.

#include "common/functional_types.h"
#include "common/pipeline_state.h"
#include "fragment/fragment_frontend.h"

#include <systemc>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace pvrgpu::stub;

static_assert(sizeof(FragmentShaderLane) == 64);
static_assert(offsetof(FragmentShaderLane, front_facing) == 41);
static_assert(offsetof(FragmentShaderLane, depth) == 44);

void Check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error("FragmentFrontend test failed: " + message);
}

} // namespace

int sc_main(int, char **) {
  try {
    // Genuine generic driver FS ABI always reserves CF0..3 for position,
    // even with no user varying. Derivative-only quads need that existing
    // ParameterBuffer/PDS route just as texture-only quads do.
    PipelineState derivative_layout;
    derivative_layout.functional_case = FunctionalCase::kDriverPcoTriangles;
    derivative_layout.position_output_count = 4;
    derivative_layout.fragment_position_count = 4;
    derivative_layout.fragment_varying_start = 4;
    derivative_layout.fragment_pco_abi.coefficients = 4;
    Check(!UsesShaderVaryings(derivative_layout) &&
              !UsesFragmentQuadLanes(derivative_layout) &&
              VaryingCoefficientDwordCount(derivative_layout) == 0,
          "legacy no-texture/no-derivative route changed");
    derivative_layout.fragment_program_summary.uses_derivatives = 1;
    Check(UsesShaderVaryings(derivative_layout) &&
              UsesFragmentQuadLanes(derivative_layout) &&
              VaryingVectorCount(derivative_layout) == 0 &&
              VaryingCoefficientSetCount(derivative_layout) == 1 &&
              VaryingCoefficientDwordCount(derivative_layout) == 4,
          "derivative-only shader lost its declared position plane");
    for (unsigned mutation = 0; mutation < 4; ++mutation) {
      auto invalid = derivative_layout;
      if (mutation == 0) invalid.fragment_position_start = 1;
      if (mutation == 1) invalid.fragment_position_count = 0;
      if (mutation == 2) invalid.fragment_position_count = 8;
      if (mutation == 3) invalid.fragment_varying_start = 0;
      Check(VaryingCoefficientDwordCount(invalid) == 0,
            "derivative-only route accepted malformed position layout");
    }
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
    parameters[1].front_facing = 1;
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
      Check(msaa_invocations[primitive].front_facing == parameters[primitive].front_facing &&
                msaa_lanes[primitive].front_facing == parameters[primitive].front_facing,
            "front/back multisample primitives retain distinct normalized facing");
    }
    ReleaseFunctionalPayloads(pool, msaa_result);
    pool.Release(msaa_handle);
    for (unsigned samples : {1U, 2U, 4U, 8U, 16U}) {
      for (bool helpers : {false, true}) {
        PipelineState sampled;
        sampled.width = 4; sampled.height = 2;
        sampled.sequence = 30 + samples * 2 + helpers;
        sampled.functional_case = FunctionalCase::kDriverPcoTriangles;
        sampled.stage = PipelineStage::kVisibilityReady;
        sampled.raster_state.sample_count = samples;
        sampled.raster_state.sample_frequency = 1;
        sampled.fragment_program_summary.uses_derivatives = helpers;
        sampled.active_fragment_invocations = 1;
        auto p = parameter; p.depth_plane_valid = 1;
        p.front_facing = samples & 1U;
        auto candidate = rejected;
        candidate.visibility = FragmentVisibility::kVisible;
        candidate.sample_mask = (1U << samples) - 1U;
        if (samples > 2) candidate.sample_mask &= ~2U;
        for (unsigned sample = 0; sample < samples; ++sample)
          candidate.sample_depth[sample] = static_cast<float>(sample + 1) / 32.0F;
        const unsigned expected = samples > 2 ? samples - 1 : samples;
        sampled.parameter_triangles = StoreNewArray(pool, std::vector<ParameterTriangle>{p});
        sampled.fragment_candidates = StoreNewArray(pool, std::vector<FragmentCandidate>{candidate});
        const auto handle = pool.Allocate(sizeof(PipelineState));
        StorePipelineState(pool, handle, sampled);
        input.write(PipelineTxn{handle, static_cast<std::uint32_t>(sampled.sequence), sampled.sequence});
        sc_core::sc_start(sc_core::sc_time(1000, sc_core::SC_NS));
        Check(output.nb_read(completed) && completed.sequence == sampled.sequence,
              "sample-frequency frontend did not complete");
        const auto done = LoadPipelineState(pool, handle);
        const auto invocations = LoadArray<FragmentInvocation>(pool, done.fragment_invocations);
        const auto quads = LoadArray<FragmentQuad>(pool, done.fragment_quads);
        Check(invocations.size() == expected && done.active_fragment_invocations == expected &&
                  done.counters.ps_invocations == expected &&
                  done.fragment_shader_lane_count == expected * (helpers ? 4U : 1U),
              "sample-frequency shading did not create one native invocation per covered sample");
        std::uint32_t seen_samples = 0;
        for (const auto &invocation : invocations) {
          Check(invocation.sample_id < samples &&
                    invocation.sample_mask == (1U << invocation.sample_id) &&
                    invocation.depth == candidate.sample_depth[invocation.sample_id] &&
                    (seen_samples & invocation.sample_mask) == 0,
                "sample invocation lost unique ID/coverage/depth");
          seen_samples |= invocation.sample_mask;
        }
        Check(seen_samples == candidate.sample_mask, "sample invocation coverage union changed");
        if (helpers) {
          const auto lanes = LoadArray<FragmentShaderLane>(pool, done.fragment_shader_lanes);
          for (const auto &quad : quads) {
            Check(quad.sample_id < samples && (candidate.sample_mask & (1U << quad.sample_id)),
                  "helper quad belongs to an uncovered sample");
            for (unsigned lane = 0; lane < 4; ++lane) {
              const auto &work = lanes.at(quad.invocation_indices[lane]);
              Check(work.sample_id == quad.sample_id && work.quad_lane == lane &&
                        work.sample_mask == (work.helper ? 0U : 1U << quad.sample_id),
                    "helper quad mixed sample IDs or invented covered helper samples");
              Check(work.front_facing == p.front_facing,
                    "covered and uncovered helper lanes inherit their actual primitive facing");
            }
          }
        }
        ReleaseFunctionalPayloads(pool, done); pool.Release(handle);
      }
    }
    for (unsigned stage = 0; stage < 4; ++stage) {
      PipelineState sampled;
      sampled.width = 4; sampled.height = 2; sampled.sequence = 3 + stage;
      sampled.functional_case = FunctionalCase::kDriverPcoTriangles;
      sampled.stage = PipelineStage::kVisibilityReady;
      sampled.vertex_sampled_texture_count = stage == 0;
      sampled.sampled_texture_count = stage == 1;
      sampled.geometry_sampled_texture_count = stage == 2;
      sampled.fragment_program_summary.uses_derivatives = stage == 3;
      sampled.active_fragment_invocations = 1;
      auto p = parameter; p.depth_plane_valid = 1;
      auto candidate = rejected; candidate.visibility = FragmentVisibility::kVisible;
      sampled.parameter_triangles = StoreNewArray(pool, std::vector<ParameterTriangle>{p});
      sampled.fragment_candidates = StoreNewArray(pool, std::vector<FragmentCandidate>{candidate});
      const auto handle = pool.Allocate(sizeof(PipelineState));
      StorePipelineState(pool, handle, sampled);
      input.write(PipelineTxn{handle, 3 + stage, sampled.sequence});
      sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_NS));
      Check(output.nb_read(completed) && completed.sequence == sampled.sequence,
            "stage-local texture frontend did not complete");
      const auto done = LoadPipelineState(pool, handle);
      Check(done.active_fragment_invocations == 1 && done.counters.ps_invocations == 1 &&
            done.fragment_shader_lane_count == (stage == 1 || stage == 3 ? 4U : 1U) &&
            HasPoolHandle(done.fragment_shader_lanes) == (stage == 1 || stage == 3),
            "fragment SMP and derivatives need helper quads; VS/GS samplers remain independent");
      ReleaseFunctionalPayloads(pool, done); pool.Release(handle);
    }

    struct ExpectedQuad {
      std::uint32_t parameter_index;
      std::uint32_t quad_id;
      std::uint8_t sample_id;
      std::uint8_t coverage;
    };
    std::uint32_t fixture_sequence = 100;
    auto candidate_at = [&](std::uint32_t x, std::uint32_t y,
                            std::uint32_t primitive, std::uint16_t samples,
                            bool is_visible = true) {
      auto candidate = rejected;
      candidate.x = x;
      candidate.y = y;
      candidate.parameter_index = primitive;
      candidate.primitive_id = parameters.at(primitive).key.api_primitive_id;
      candidate.submit_ordinal = parameters.at(primitive).key.submit_ordinal;
      candidate.sample_mask = samples;
      candidate.visibility = is_visible ? FragmentVisibility::kVisible
                                        : FragmentVisibility::kRejected;
      for (unsigned sample = 0; sample < 16; ++sample)
        candidate.sample_depth[sample] = static_cast<float>(sample + 1) / 32.0F;
      return candidate;
    };
    auto check_quad_fixture = [&](const std::string &name, unsigned width,
                                  unsigned height, unsigned samples,
                                  bool sample_frequency,
                                  const std::vector<FragmentCandidate> &candidates,
                                  const std::vector<ExpectedQuad> &expected) {
      PipelineState fixture;
      fixture.width = width;
      fixture.height = height;
      fixture.sequence = fixture_sequence++;
      fixture.functional_case = FunctionalCase::kDriverPcoTriangles;
      fixture.stage = PipelineStage::kVisibilityReady;
      fixture.raster_state.sample_count = samples;
      fixture.raster_state.sample_frequency = sample_frequency;
      fixture.sampled_texture_count = 1;
      fixture.fragment_program_summary.uses_derivatives = 1;
      auto fixture_parameters = parameters;
      for (auto &p : fixture_parameters)
        p.depth_plane_valid = 1;
      for (const auto &candidate : candidates)
        fixture.active_fragment_invocations +=
            candidate.visibility == FragmentVisibility::kVisible;
      fixture.parameter_triangles = StoreNewArray(pool, fixture_parameters);
      fixture.fragment_candidates = StoreNewArray(pool, candidates);
      const auto handle = pool.Allocate(sizeof(PipelineState));
      StorePipelineState(pool, handle, fixture);
      input.write(PipelineTxn{handle, static_cast<std::uint32_t>(fixture.sequence), fixture.sequence});
      sc_core::sc_start(sc_core::sc_time(1000, sc_core::SC_NS));
      Check(output.nb_read(completed) && completed.sequence == fixture.sequence,
            name + ": frontend did not complete");
      const auto done = LoadPipelineState(pool, handle);
      const auto quads = LoadArray<FragmentQuad>(pool, done.fragment_quads);
      const auto lanes = LoadArray<FragmentShaderLane>(pool, done.fragment_shader_lanes);
      const auto invocations = LoadArray<FragmentInvocation>(pool, done.fragment_invocations);
      Check(done.stage == PipelineStage::kFragmentsReady &&
                quads.size() == expected.size() &&
                done.fragment_groups == expected.size() &&
                lanes.size() == 4 * expected.size() &&
                done.fragment_shader_lane_count == lanes.size(),
            name + ": only nonempty 2x2 quads may issue four shader lanes");
      std::vector<bool> seen_lanes(lanes.size(), false);
      std::vector<bool> seen_invocations(invocations.size(), false);
      unsigned covered_count = 0;
      for (const auto &wanted : expected) {
        const auto it = std::find_if(quads.begin(), quads.end(), [&](const auto &quad) {
          return quad.parameter_index == wanted.parameter_index &&
                 quad.quad_id == wanted.quad_id && quad.sample_id == wanted.sample_id;
        });
        Check(it != quads.end(), name + ": missing primitive/quad/sample identity");
        const auto &quad = *it;
        Check(wanted.coverage != 0 && quad.coverage_mask == wanted.coverage &&
                  quad.write_mask == wanted.coverage &&
                  quad.helper_mask == (0xfU ^ wanted.coverage),
              name + ": visible coverage and required helper masks changed");
        const unsigned quads_x = (width + 1) / 2;
        const unsigned quad_x = (wanted.quad_id % quads_x) * 2;
        const unsigned quad_y = (wanted.quad_id / quads_x) * 2;
        for (unsigned lane = 0; lane < 4; ++lane) {
          const unsigned index = quad.invocation_indices[lane];
          Check(index < lanes.size() && !seen_lanes[index], name + ": shader lane alias");
          seen_lanes[index] = true;
          const auto &work = lanes[index];
          const bool covered = (wanted.coverage & (1U << lane)) != 0;
          Check(work.x == quad_x + lane % 2 && work.y == quad_y + lane / 2 &&
                    work.quad_lane == lane && work.quad_id == wanted.quad_id &&
                    work.parameter_index == wanted.parameter_index &&
                    work.sample_id == wanted.sample_id && work.helper == !covered &&
                    work.primitive_id == fixture_parameters[wanted.parameter_index].key.api_primitive_id &&
                    work.submit_ordinal == quad.submit_ordinal,
                name + ": lane coordinates, identity or helper state changed");
          if (covered) {
            ++covered_count;
            const auto visible_index = work.visible_invocation_index;
            Check(visible_index < invocations.size() && !seen_invocations[visible_index],
                  name + ": visible invocation alias");
            seen_invocations[visible_index] = true;
            const auto &visible_work = invocations[visible_index];
            Check(work.x < width && work.y < height &&
                      visible_work.x == work.x && visible_work.y == work.y &&
                      visible_work.parameter_index == work.parameter_index &&
                      visible_work.sample_id == work.sample_id &&
                      work.sample_mask == visible_work.sample_mask &&
                      (!sample_frequency || work.sample_mask == (1U << work.sample_id)),
                  name + ": covered lane lost its unique invocation/sample");
          } else {
            Check(work.sample_mask == 0 &&
                      work.visible_invocation_index == kInvalidFragmentInvocationIndex,
                  name + ": helper invented visible work or sample coverage");
          }
        }
      }
      Check(covered_count == invocations.size() &&
                done.active_fragment_invocations == covered_count &&
                done.counters.ps_invocations == covered_count &&
                std::all_of(seen_lanes.begin(), seen_lanes.end(), [](bool seen) { return seen; }) &&
                std::all_of(seen_invocations.begin(), seen_invocations.end(), [](bool seen) { return seen; }),
            name + ": unmatched or lost shader/visible work");
      ReleaseFunctionalPayloads(pool, done);
      pool.Release(handle);
    };
    check_quad_fixture("left child only", 4, 2, 1, false,
                       {candidate_at(0, 0, 0, 1)}, {{0, 0, 0, 0x1}});
    check_quad_fixture("right child only", 4, 2, 1, false,
                       {candidate_at(3, 1, 0, 1)}, {{0, 1, 0, 0x8}});
    check_quad_fixture("both children", 4, 2, 1, false,
                       {candidate_at(0, 0, 0, 1), candidate_at(1, 1, 0, 1),
                        candidate_at(2, 0, 0, 1)},
                       {{0, 0, 0, 0x9}, {0, 1, 0, 0x1}});
    check_quad_fixture("children belong to different primitives", 4, 2, 1, false,
                       {candidate_at(0, 0, 0, 1), candidate_at(3, 1, 1, 1)},
                       {{0, 0, 0, 0x1}, {1, 1, 0, 0x8}});
    check_quad_fixture("children belong to different samples", 4, 2, 4, true,
                       {candidate_at(0, 0, 0, 0x2), candidate_at(3, 1, 0, 0x8)},
                       {{0, 0, 1, 0x1}, {0, 1, 3, 0x8}});
    check_quad_fixture("different primitive and sample identities", 4, 2, 4, true,
                       {candidate_at(0, 0, 0, 0x5), candidate_at(3, 1, 1, 0xa)},
                       {{0, 0, 0, 0x1}, {0, 0, 2, 0x1},
                        {1, 1, 1, 0x8}, {1, 1, 3, 0x8}});
    check_quad_fixture("odd viewport retains derivative helpers", 3, 3, 1, false,
                       {candidate_at(2, 2, 0, 1)}, {{0, 3, 0, 0x1}});
    check_quad_fixture("ISP rejected adjacent child", 4, 2, 1, false,
                       {candidate_at(0, 0, 0, 1), candidate_at(2, 0, 0, 1, false)},
                       {{0, 0, 0, 0x1}});
    check_quad_fixture("ISP rejected entire primitive beside visible primitive", 4, 2, 1, false,
                       {candidate_at(0, 0, 0, 1, false), candidate_at(2, 0, 1, 1)},
                       {{1, 1, 0, 0x1}});
    check_quad_fixture("ISP rejected all samples in both children", 4, 2, 4, true,
                       {candidate_at(0, 0, 0, 0xf, false), candidate_at(2, 0, 1, 0xf, false)},
                       {});
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
