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

#include <limits>
#include <stdexcept>
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
    const bool varying_case = UsesShaderVaryings(state.functional_case);
    if (!IsSolidColorRasterCase(state.functional_case) && !varying_case)
      throw std::runtime_error("PCO decoder received an unsupported case");

    const PoolHandle code_handle = stage_ == ShaderStage::kVertex
                                       ? state.vertex_code
                                       : state.fragment_code;
    if (!HasPoolHandle(code_handle))
      throw std::runtime_error("PCO decoder received no shader binary");
    if (stage_ == ShaderStage::kVertex) {
      RequireStage(state.stage, PipelineStage::kVertexFetched, name());
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
      }
      if ((decoded.summary.vertex_input_mask & ~available_vertex_inputs) != 0) {
        throw std::runtime_error(
            "vertex PCO reads VTXIN registers absent from attribute bindings");
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
      const std::uint32_t expected_output_count =
          varying_case
              ? VaryingVertexOutputDwordCount(state.functional_case)
              : 4;
      if (expected_output_count == 0 || expected_output_count >= 64)
        throw std::runtime_error(
            "vertex PCO expected VTXOUT range is invalid");
      const std::uint64_t expected_output_mask =
          (UINT64_C(1) << expected_output_count) - 1;
      if (decoded.summary.vertex_output_mask != expected_output_mask ||
          !decoded.summary.ends_task) {
        throw std::runtime_error(
            "vertex PCO output mask does not match shader linkage");
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
