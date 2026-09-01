// PcoDecoder fetches real PowerVR PCO bytes from MemoryPool, validates the
// public variable-length instruction-group encoding, and serializes semantic
// instructions for the USC ISS (Instruction Set Simulator). PCO is Mesa's
// public backend name; no unverified expansion is asserted. Unknown opcode,
// register, flag, padding or stage fails closed. It also classifies each
// decoded DrawList program into static ALU/Tex/Memory counts. The SystemC
// module is event-driven and its FIFO carries only the PipelineState handle.
#include "shader/pco_decoder.h"

#include "common/functional_types.h"
#include "common/pipeline_state.h"
#include "shader/pco_iss.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pvrgpu::stub {

PcoDecoder::PcoDecoder(sc_core::sc_module_name name, MemoryPool &pool,
                       ShaderStage stage)
    : sc_module(name), pool_(pool), stage_(stage) {
  SC_THREAD(Run);
}

void PcoDecoder::Run() {
  while (true) {
    const PipelineTxn txn = input.read();
    PipelineState state = LoadPipelineState(pool_, txn.state);
    const bool varying_case = UsesShaderVaryings(state);
    const bool driver_pco_triangles =
        IsDriverPcoTrianglesCase(state.functional_case);
    if (!IsSolidColorRasterCase(state.functional_case) && !varying_case)
      throw std::runtime_error("PCO decoder received an unsupported case");

    const PoolHandle code_handle = stage_ == ShaderStage::kVertex
                                       ? state.vertex_code
                                       : state.fragment_code;
    if (!HasPoolHandle(code_handle))
      throw std::runtime_error("PCO decoder received no shader binary");
    if (stage_ == ShaderStage::kVertex) {
      if (state.stage != PipelineStage::kVertexFetched &&
          state.stage != PipelineStage::kVertexPdsReady) {
        RequireStage(state.stage, PipelineStage::kVertexPdsReady, name());
      }
    } else {
      RequireStage(state.stage, PipelineStage::kParameterBufferReady, name());
    }

    const std::vector<std::uint8_t> binary =
        LoadArray<std::uint8_t>(pool_, code_handle);
    const PcoDecodedProgram decoded = DecodePcoProgram(stage_, binary);
    if (decoded.instructions.empty() ||
        decoded.summary.instruction_count != decoded.instructions.size()) {
      throw std::runtime_error(
          "PCO decoder produced an empty/inconsistent program");
    }

    if (stage_ == ShaderStage::kVertex) {
      if (!HasPoolHandle(state.vertex_attribute_bindings)) {
        throw std::runtime_error(
            "vertex PCO decoder received no attribute-binding contract");
      }
      const std::vector<VertexAttributeBinding> bindings =
          LoadArray<VertexAttributeBinding>(pool_,
                                            state.vertex_attribute_bindings);
      std::uint32_t available_vertex_inputs = 0;
      std::uint32_t driver_readable_vertex_inputs = 0;
      std::uint32_t declared_vertex_input_count = 0;
      for (const VertexAttributeBinding &binding : bindings) {
        const std::size_t first = binding.destination_register;
        const std::size_t count = binding.destination_components;
        if (count == 0 || first >= kPcoVertexInputCount ||
            count > kPcoVertexInputCount - first) {
          throw std::runtime_error(
              "vertex attribute binding has an invalid VTXIN range");
        }
        for (std::size_t component = 0; component < count; ++component) {
          available_vertex_inputs |= static_cast<std::uint32_t>(
              UINT32_C(1) << (first + component));
        }
        if (driver_pco_triangles) {
          if (binding.source_components == 0 ||
              binding.source_components > binding.destination_components) {
            throw std::runtime_error(
                "driver PCO attribute has an invalid source width");
          }
          for (std::size_t component = 0;
               component < binding.source_components; ++component) {
            driver_readable_vertex_inputs |= static_cast<std::uint32_t>(
                UINT32_C(1) << (first + component));
          }
          declared_vertex_input_count = std::max(
              declared_vertex_input_count,
              static_cast<std::uint32_t>(first + count));
        }
      }
      if ((decoded.summary.vertex_input_mask & ~available_vertex_inputs) != 0) {
        throw std::runtime_error(
            "vertex PCO reads VTXIN registers absent from attribute bindings");
      }
      if (driver_pco_triangles &&
          (decoded.summary.vertex_input_mask != driver_readable_vertex_inputs ||
           (state.vertex_pco_abi.vertex_inputs != 0 &&
            state.vertex_pco_abi.vertex_inputs !=
                declared_vertex_input_count))) {
        throw std::runtime_error(
            "driver PCO vertex-input mask does not match attribute bindings");
      }
    }

    if (!HasPoolHandle(state.drawlist_stats))
      throw std::runtime_error("PCO decoder received no DrawList statistics");
    std::vector<DrawListStats> drawlists =
        LoadArray<DrawListStats>(pool_, state.drawlist_stats);
    if (drawlists.size() != 1 || drawlists[0].drawlist_index != 0)
      throw std::runtime_error("PCO decoder requires DrawList 0");
    DrawListShaderStats &shader_stats = stage_ == ShaderStage::kVertex
                                            ? drawlists[0].vertex
                                            : drawlists[0].fragment;
    if (shader_stats.program_recorded != 0)
      throw std::runtime_error("PCO program statistics were decoded twice");
    const PcoInstructionCounts program_counts =
        CountPcoInstructions(decoded.instructions, false);
    shader_stats.program_groups = decoded.summary.group_count;
    shader_stats.program_instructions = decoded.summary.instruction_count;
    shader_stats.program_alu_instructions = program_counts.alu;
    shader_stats.program_tex_instructions = program_counts.texture;
    shader_stats.program_memory_instructions = program_counts.memory;
    shader_stats.program_recorded = 1;

    if (stage_ == ShaderStage::kVertex) {
      std::uint32_t expected_output_count = 4;
      if (driver_pco_triangles) {
        if (state.position_output_start != 0 ||
            state.position_output_count == 0 ||
            state.position_output_count >= 64 ||
            (state.varying_output_count != 0 &&
             (state.varying_output_start != state.position_output_count ||
              state.varying_output_count >=
                  64 - state.varying_output_start))) {
          throw std::runtime_error(
              "driver PCO vertex-output linkage is invalid");
        }
        expected_output_count = state.position_output_count;
        if (state.varying_output_count != 0) {
          expected_output_count =
              state.varying_output_start + state.varying_output_count;
        }
      } else if (varying_case) {
        expected_output_count = VaryingVertexOutputDwordCount(state);
      }
      if (expected_output_count == 0 || expected_output_count >= 64)
        throw std::runtime_error(
            "vertex PCO expected VTXOUT range is invalid");
      const std::uint64_t expected_output_mask =
          (UINT64_C(1) << expected_output_count) - 1;
      const std::uint64_t required_position_mask =
          driver_pco_triangles
              ? (UINT64_C(1) << state.position_output_count) - 1
              : expected_output_mask;
      if ((decoded.summary.vertex_output_mask & required_position_mask) !=
              required_position_mask ||
          (decoded.summary.vertex_output_mask & ~expected_output_mask) != 0 ||
          (driver_pco_triangles && state.vertex_pco_abi.vertex_outputs != 0 &&
           state.vertex_pco_abi.vertex_outputs != expected_output_count) ||
          !decoded.summary.ends_task) {
        throw std::runtime_error(
            "vertex PCO output mask does not match shader linkage: decoded=" +
            std::to_string(decoded.summary.vertex_output_mask) +
            " required_position=" +
            std::to_string(required_position_mask) +
            " declared=" + std::to_string(expected_output_mask) +
            " abi_outputs=" +
            std::to_string(state.vertex_pco_abi.vertex_outputs) +
            " ends_task=" + std::to_string(decoded.summary.ends_task));
      }
      state.vertex_program_summary = decoded.summary;
      state.vertex_instructions = StoreNewArray(pool_, decoded.instructions);
      state.vertex_groups = CeilDivide(state.counters.vs_invocations,
                                       kReferenceUarch.usc_issue_lanes);
      state.stage = PipelineStage::kVertexDecoded;
    } else {
      if (decoded.summary.pixel_output_mask != 0x0f) {
        throw std::runtime_error(
            "solid-color fragment PCO must write PIXOUT0..3");
      }
      state.fragment_program_summary = decoded.summary;
      state.fragment_instructions = StoreNewArray(pool_, decoded.instructions);
      state.fragment_early_hsr_safe = decoded.summary.early_hsr_safe;
      state.stage = PipelineStage::kFragmentDecoded;
    }

    const std::uint64_t cycles =
        kReferenceUarch.pco_decode_base_cycles +
        CeilDivide(decoded.summary.group_count,
                   kReferenceUarch.pco_groups_per_decode_batch);
    if (decoded.summary.instruction_count >
        std::numeric_limits<std::uint64_t>::max() -
            state.counters.pco_instructions) {
      throw std::overflow_error("PCO instruction counter overflow");
    }
    state.counters.pco_instructions += decoded.summary.instruction_count;
    state.counters.pco_decode_cycles += cycles;
    if (stage_ == ShaderStage::kVertex)
      state.counters.tiler_cycles += cycles;
    else
      state.counters.renderer_cycles += cycles;
    StoreArray(pool_, state.drawlist_stats, drawlists);

    WaitForCycles(cycles);
    StorePipelineState(pool_, txn.state, state);
    output.write(txn);
  }
}

} // namespace pvrgpu::stub
