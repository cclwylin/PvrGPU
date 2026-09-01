// UscSlot：模擬 shader task 取得 USC execution slot 的 issue 階段。
// USC（Unified Shading Cluster，統一著色叢集）是 PowerVR 執行 vertex、fragment
// 與 compute shader 的可程式單元。vertex work 使用 four-lane issue
// group；fragment work 必須先通過 PDS，並使用其 spatial 2×2 quad task
// 與 coefficient-bank range（含 partial quad）。模組以
// FIFO（First-In, First-Out）傳遞 MemoryPool state handle，採單次事件延遲
// 而非逐週期 clock。
#include "shader/usc_slot.h"

#include "common/functional_types.h"
#include "common/pipeline_state.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pvrgpu::stub {

UscSlot::UscSlot(sc_core::sc_module_name name, MemoryPool &pool,
                 ShaderStage stage)
    : sc_module(name), pool_(pool), stage_(stage) {
  SC_THREAD(Run);
}

void UscSlot::Run() {
  while (true) {
    const PipelineTxn txn = input.read();
    PipelineState state = LoadPipelineState(pool_, txn.state);

    if (!IsRasterFunctionalCase(state.functional_case))
      throw std::runtime_error("USC slot received an unsupported case");

    const bool texture_fragment =
        stage_ == ShaderStage::kFragment &&
        UsesTextureSampling(state);
    const std::uint64_t lanes = stage_ == ShaderStage::kVertex
                                    ? state.counters.vs_invocations
                                    : texture_fragment
                                          ? state.fragment_shader_lane_count
                                          : state.counters.ps_invocations;
    const std::uint64_t groups = stage_ == ShaderStage::kVertex
                                     ? state.vertex_groups
                                     : state.fragment_groups;

    if (stage_ == ShaderStage::kVertex) {
      RequireStage(state.stage, PipelineStage::kVertexDecoded, name());
      if (!HasPoolHandle(state.vertex_lanes))
        throw std::runtime_error("vertex USC slot has no vertex lanes");
    } else {
      RequireStage(state.stage, PipelineStage::kPdsReady, name());
      if (!HasPoolHandle(state.fragment_invocations) ||
          !HasPoolHandle(state.fragment_quads) ||
          !HasPoolHandle(state.usc_fragment_tasks) ||
          !HasPoolHandle(state.usc_coefficient_banks)) {
        throw std::runtime_error(
            "fragment USC slot has incomplete PDS work payloads");
      }
      const std::vector<FragmentInvocation> invocations =
          LoadArray<FragmentInvocation>(pool_, state.fragment_invocations);
      std::vector<FragmentShaderLane> shader_lanes;
      if (texture_fragment) {
        if (!HasPoolHandle(state.fragment_shader_lanes))
          throw std::runtime_error(
              "fragment USC texture task has no shader lanes");
        shader_lanes = LoadArray<FragmentShaderLane>(
            pool_, state.fragment_shader_lanes);
        if (shader_lanes.size() != lanes)
          throw std::runtime_error(
              "fragment USC texture shader-lane payload mismatch");
      }
      const std::vector<FragmentQuad> quads =
          LoadArray<FragmentQuad>(pool_, state.fragment_quads);
      const std::vector<UscFragmentTask> tasks =
          LoadArray<UscFragmentTask>(pool_, state.usc_fragment_tasks);
      const std::vector<std::uint32_t> coefficient_banks =
          LoadArray<std::uint32_t>(pool_, state.usc_coefficient_banks);
      if (!texture_fragment && invocations.size() != lanes)
        throw std::runtime_error("fragment USC invocation payload mismatch");
      if (invocations.size() != state.active_fragment_invocations) {
        throw std::runtime_error(
            "fragment USC lane count does not match active pixels");
      }
      if (quads.size() != groups || tasks.size() != groups)
        throw std::runtime_error("fragment USC PDS task/group count mismatch");

      const bool varying_case = UsesShaderVaryings(state);
      const std::uint32_t varying_coefficient_dwords =
          VaryingCoefficientDwordCount(state);
      if (varying_case &&
          (varying_coefficient_dwords == 0 ||
           varying_coefficient_dwords >
               std::numeric_limits<std::uint16_t>::max())) {
        throw std::runtime_error(
            "fragment USC varying coefficient count is invalid");
      }
      std::size_t expected_coefficient_dword = 0;
      for (std::size_t task_index = 0; task_index < tasks.size();
           ++task_index) {
        const UscFragmentTask &task = tasks[task_index];
        if (task.fragment_quad_index != task_index || task.reserved != 0 ||
            task.fragment_quad_index >= quads.size() ||
            task.first_coefficient_dword != expected_coefficient_dword) {
          throw std::runtime_error(
              "fragment USC task descriptor is invalid");
        }
        const std::uint16_t expected_count =
            varying_case
                ? static_cast<std::uint16_t>(varying_coefficient_dwords)
                : 0;
        if (task.coefficient_dword_count != expected_count)
          throw std::runtime_error(
              "fragment USC task coefficient count is invalid");
        expected_coefficient_dword += task.coefficient_dword_count;
        if (expected_coefficient_dword > coefficient_banks.size())
          throw std::runtime_error(
              "fragment USC coefficient range is out of bounds");
        const FragmentQuad &quad = quads[task.fragment_quad_index];
        const std::uint8_t active_mask = static_cast<std::uint8_t>(
            quad.coverage_mask | quad.helper_mask);
        if ((!texture_fragment &&
             (quad.coverage_mask == 0 || quad.helper_mask != 0)) ||
            (texture_fragment &&
             (active_mask == 0 ||
              (quad.coverage_mask & quad.helper_mask) != 0)) ||
            quad.write_mask != quad.coverage_mask || quad.reserved != 0) {
          throw std::runtime_error("fragment USC received invalid quad masks");
        }
      }
      if (expected_coefficient_dword != coefficient_banks.size())
        throw std::runtime_error(
            "fragment USC coefficient bank has trailing dwords");
      if (varying_case) {
        if (state.counters.pds_coefficient_tasks != tasks.size() ||
            state.counters.pds_douti_issues != tasks.size() * 2 ||
            state.counters.usc_coefficient_load_bytes !=
                coefficient_banks.size() * sizeof(std::uint32_t)) {
          throw std::runtime_error(
              "fragment USC coefficient counters are inconsistent");
        }
      } else if (!coefficient_banks.empty() ||
                 state.counters.pds_coefficient_tasks != 0 ||
                 state.counters.pds_douti_issues != 0 ||
                 state.counters.usc_coefficient_load_bytes != 0) {
        throw std::runtime_error(
            "fragment USC solid-color task has coefficient state");
      }
    }

    if (stage_ == ShaderStage::kVertex) {
      if (lanes == 0 || groups == 0 ||
          groups != CeilDivide(lanes, kReferenceUarch.usc_issue_lanes)) {
        throw std::runtime_error(
            "vertex USC slot received an invalid lane/group count");
      }
    } else {
      if ((lanes == 0) != (groups == 0) ||
          (lanes != 0 &&
           (groups < CeilDivide(lanes, kReferenceUarch.usc_issue_lanes) ||
            groups > lanes))) {
        throw std::runtime_error(
            "fragment USC slot received an invalid lane/quad count");
      }
    }

    const std::uint64_t cycles =
        groups == 0
            ? 0
            : kReferenceUarch.usc_slot_base_cycles +
                  CeilDivide(groups, kReferenceUarch.usc_groups_per_slot_batch);

    state.counters.usc_slot_cycles += cycles;
    if (stage_ == ShaderStage::kVertex) {
      state.counters.tiler_cycles += cycles;
      state.stage = PipelineStage::kVertexIssued;
    } else {
      state.counters.renderer_cycles += cycles;
      state.stage = PipelineStage::kFragmentIssued;
    }

    WaitForCycles(cycles);
    StorePipelineState(pool_, txn.state, state);
    output.write(txn);
  }
}

} // namespace pvrgpu::stub
