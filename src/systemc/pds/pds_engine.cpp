// PdsEngine：PDS = Programmable Data Sequencer（可程式化資料定序器）。
// 它執行 fragment coefficient-loading work：每個 primitive/2x2 quad task
// 依 ParameterBuffer range 把 A/B/C/PAD raw dwords 複製到 USC coefficient
// bank，並發布獨立 task descriptor。FIFO 只傳 PipelineState handle；bulk
// coefficient/task data 留在 MemoryPool，採 event-driven SC_THREAD。
#include "pds/pds_engine.h"

#include "common/functional_types.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pvrgpu::stub {

PdsEngine::PdsEngine(sc_core::sc_module_name name, MemoryPool &pool)
    : sc_module(name), pool_(pool) {
  SC_THREAD(Run);
}

void PdsEngine::Run() {
  while (true) {
    const PipelineTxn txn = input.read();
    PipelineState state = LoadPipelineState(pool_, txn.state);
    RequireStage(state.stage, PipelineStage::kFragmentsReady, name());
    if (!IsRasterFunctionalCase(state.functional_case))
      throw std::runtime_error("PDS received an unsupported case");
    if (!HasPoolHandle(state.fragment_invocations) ||
        !HasPoolHandle(state.fragment_quads) ||
        !HasPoolHandle(state.parameter_triangles) ||
        HasPoolHandle(state.usc_fragment_tasks) ||
        HasPoolHandle(state.usc_coefficient_banks)) {
      throw std::runtime_error("PDS input/output payload ownership is invalid");
    }
    if (state.counters.pds_coefficient_tasks != 0 ||
        state.counters.pds_douti_issues != 0 ||
        state.counters.usc_coefficient_load_bytes != 0) {
      throw std::runtime_error("PDS counters were populated before PDS");
    }

    const std::vector<FragmentInvocation> invocations =
        LoadArray<FragmentInvocation>(pool_, state.fragment_invocations);
    const bool texture_case = IsTextureFamily(state.functional_case);
    std::vector<FragmentShaderLane> shader_lanes;
    if (texture_case) {
      if (!HasPoolHandle(state.fragment_shader_lanes))
        throw std::runtime_error("PDS texture case has no shader lanes");
      shader_lanes = LoadArray<FragmentShaderLane>(
          pool_, state.fragment_shader_lanes);
      if (shader_lanes.size() != state.fragment_shader_lane_count)
        throw std::runtime_error("PDS texture shader-lane count mismatch");
    }
    const std::vector<FragmentQuad> quads =
        LoadArray<FragmentQuad>(pool_, state.fragment_quads);
    const std::vector<ParameterTriangle> parameters =
        LoadArray<ParameterTriangle>(pool_, state.parameter_triangles);
    if (invocations.size() != state.active_fragment_invocations ||
        quads.size() != state.fragment_groups) {
      throw std::runtime_error("PDS fragment work counts are inconsistent");
    }

    const bool varying_case = UsesShaderVaryings(state.functional_case);
    std::vector<ParameterCoefficientSet> parameter_coefficients;
    if (varying_case) {
      if (!HasPoolHandle(state.shader_varying_bindings) ||
          !HasPoolHandle(state.parameter_coefficients)) {
        throw std::runtime_error(
            "PDS varying case has no linkage/coefficient payload");
      }
      const std::vector<ShaderVaryingBinding> bindings =
          LoadArray<ShaderVaryingBinding>(pool_,
                                          state.shader_varying_bindings);
      const std::uint32_t varying_count =
          VaryingVectorCount(state.functional_case);
      if (varying_count == 0 || bindings.size() != varying_count) {
        throw std::runtime_error(
            "PDS varying linkage count is invalid");
      }
      for (std::size_t index = 0; index < bindings.size(); ++index) {
        if (!IsExactVaryingBinding(state.functional_case, bindings[index],
                                   index)) {
          throw std::runtime_error("PDS varying linkage is not exact");
        }
      }
      parameter_coefficients = LoadArray<ParameterCoefficientSet>(
          pool_, state.parameter_coefficients);
      if (state.counters.parameter_coefficient_sets !=
              parameter_coefficients.size() ||
          state.counters.parameter_write_bytes !=
              static_cast<std::uint64_t>(parameter_coefficients.size()) *
                  sizeof(ParameterCoefficientSet)) {
        throw std::runtime_error(
            "PDS parameter coefficient counters are inconsistent");
      }
    } else if (HasPoolHandle(state.shader_varying_bindings) ||
               HasPoolHandle(state.parameter_coefficients) ||
               state.counters.parameter_coefficient_sets != 0 ||
               state.counters.parameter_write_bytes != 0) {
      throw std::runtime_error(
          "PDS solid-color case has unexpected coefficient state");
    }

    std::size_t expected_parameter_set = 0;
    for (const ParameterTriangle &parameter : parameters) {
      if (parameter.first_coefficient_set != expected_parameter_set ||
          parameter.front_facing > 1 || parameter.rasterizable > 1 ||
          parameter.face_culled > 1 ||
          (parameter.face_culled != 0 && parameter.rasterizable != 0)) {
        throw std::runtime_error(
            "PDS parameter metadata/ranges are invalid");
      }
      const std::uint16_t expected_count =
          varying_case && parameter.rasterizable
              ? static_cast<std::uint16_t>(
                    VaryingCoefficientSetCount(state.functional_case))
              : 0;
      if (parameter.coefficient_set_count != expected_count ||
          parameter.reserved[0] != 0 || parameter.reserved[1] != 0 ||
          parameter.reserved[2] != 0) {
        throw std::runtime_error(
            "PDS parameter coefficient metadata is invalid");
      }
      expected_parameter_set += parameter.coefficient_set_count;
      if (expected_parameter_set > parameter_coefficients.size()) {
        throw std::runtime_error(
            "PDS parameter coefficient range is out of bounds");
      }
    }
    if (expected_parameter_set != parameter_coefficients.size()) {
      throw std::runtime_error(
          "PDS parameter coefficient payload has trailing sets");
    }

    std::vector<UscFragmentTask> tasks;
    std::vector<std::uint32_t> usc_coefficient_banks;
    tasks.reserve(quads.size());
    const std::uint32_t coefficient_dwords =
        VaryingCoefficientDwordCount(state.functional_case);
    if (varying_case) {
      if (coefficient_dwords == 0 ||
          coefficient_dwords > std::numeric_limits<std::uint16_t>::max()) {
        throw std::runtime_error("PDS varying coefficient count is invalid");
      }
      if (quads.size() >
          std::numeric_limits<std::size_t>::max() /
              coefficient_dwords) {
        throw std::overflow_error("PDS USC coefficient bank size overflow");
      }
      usc_coefficient_banks.reserve(
          quads.size() * coefficient_dwords);
    }

    for (std::size_t quad_index = 0; quad_index < quads.size(); ++quad_index) {
      const FragmentQuad &quad = quads[quad_index];
      const std::uint8_t active_mask = static_cast<std::uint8_t>(
          quad.coverage_mask | quad.helper_mask);
      if (quad.parameter_index >= parameters.size() ||
          (!texture_case &&
           (quad.coverage_mask == 0 || quad.helper_mask != 0)) ||
          (texture_case &&
           (active_mask == 0 ||
            (quad.coverage_mask & quad.helper_mask) != 0)) ||
          quad.write_mask != quad.coverage_mask || quad.reserved != 0) {
        throw std::runtime_error("PDS received invalid FragmentQuad metadata");
      }
      const ParameterTriangle &parameter = parameters[quad.parameter_index];
      if (!parameter.rasterizable || parameter.face_culled ||
          parameter.key.submit_ordinal != quad.submit_ordinal) {
        throw std::runtime_error("PDS FragmentQuad lost primitive identity");
      }
      for (std::size_t lane = 0; lane < 4; ++lane) {
        const bool covered = (quad.coverage_mask & (1U << lane)) != 0;
        const bool active = (active_mask & (1U << lane)) != 0;
        const std::uint32_t invocation_index = quad.invocation_indices[lane];
        if (active ==
            (invocation_index == kInvalidFragmentInvocationIndex)) {
          throw std::runtime_error("PDS FragmentQuad lane mapping is invalid");
        }
        if (!active)
          continue;
        if (texture_case) {
          if (invocation_index >= shader_lanes.size())
            throw std::runtime_error("PDS shader lane is out of bounds");
          const FragmentShaderLane &shader_lane =
              shader_lanes[invocation_index];
          if (shader_lane.parameter_index != quad.parameter_index ||
              shader_lane.quad_id != quad.quad_id ||
              shader_lane.submit_ordinal != quad.submit_ordinal ||
              shader_lane.quad_lane != lane ||
              shader_lane.helper != static_cast<std::uint8_t>(!covered) ||
              (covered && shader_lane.visible_invocation_index >=
                              invocations.size()) ||
              (!covered && shader_lane.visible_invocation_index !=
                               kInvalidFragmentInvocationIndex)) {
            throw std::runtime_error("PDS quad/shader-lane identity mismatch");
          }
        } else {
          if (invocation_index >= invocations.size())
            throw std::runtime_error("PDS invocation index is out of bounds");
          const FragmentInvocation &invocation = invocations[invocation_index];
          if (invocation.parameter_index != quad.parameter_index ||
              invocation.quad_id != quad.quad_id ||
              invocation.submit_ordinal != quad.submit_ordinal ||
              invocation.quad_lane != lane) {
            throw std::runtime_error("PDS quad/invocation identity mismatch");
          }
        }
      }

      UscFragmentTask task;
      if (quad_index > std::numeric_limits<std::uint32_t>::max())
        throw std::overflow_error("PDS fragment task index overflow");
      task.fragment_quad_index = static_cast<std::uint32_t>(quad_index);
      if (usc_coefficient_banks.size() >
          std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("PDS USC coefficient offset overflow");
      }
      task.first_coefficient_dword =
          static_cast<std::uint32_t>(usc_coefficient_banks.size());
      if (varying_case) {
        if (usc_coefficient_banks.size() >
            std::numeric_limits<std::uint32_t>::max() -
                coefficient_dwords) {
          throw std::overflow_error("PDS USC coefficient range overflow");
        }
        task.coefficient_dword_count =
            static_cast<std::uint16_t>(coefficient_dwords);
        const std::size_t first = parameter.first_coefficient_set;
        for (std::size_t set = 0; set < parameter.coefficient_set_count;
             ++set) {
          const ParameterCoefficientSet &coefficient =
              parameter_coefficients[first + set];
          if (coefficient.pad != 0)
            throw std::runtime_error("PDS coefficient PAD dword is not zero");
          usc_coefficient_banks.push_back(coefficient.a);
          usc_coefficient_banks.push_back(coefficient.b);
          usc_coefficient_banks.push_back(coefficient.c);
          usc_coefficient_banks.push_back(coefficient.pad);
        }
      }
      tasks.push_back(task);
    }

    state.usc_fragment_tasks = StoreNewArray(pool_, tasks);
    state.usc_coefficient_banks =
        StoreNewArray(pool_, usc_coefficient_banks);
    if (varying_case) {
      state.counters.pds_coefficient_tasks = tasks.size();
      state.counters.pds_douti_issues =
          static_cast<std::uint64_t>(tasks.size()) * 2;
      state.counters.usc_coefficient_load_bytes =
          static_cast<std::uint64_t>(usc_coefficient_banks.size()) *
          sizeof(std::uint32_t);
      if (usc_coefficient_banks.size() !=
          tasks.size() * coefficient_dwords) {
        throw std::runtime_error("PDS coefficient copy count is inconsistent");
      }
    }
    state.stage = PipelineStage::kPdsReady;
    StorePipelineState(pool_, txn.state, state);
    output.write(txn);
  }
}

}  // namespace pvrgpu::stub
