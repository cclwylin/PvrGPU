// SPDX-License-Identifier: MIT
// Exercise the actual FIFO decoder's VTXOUT layout envelope, not only the
// standalone ISA decoder. Undefined intervening exports remain legal.
#include "common/pipeline_state.h"
#include "common/tessellation_state.h"
#include "shader/pco_decoder.h"
#include "shader/pco_iss.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace pvrgpu::stub;
namespace {
unsigned checks = 0;
void Check(bool condition, const char *message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

std::vector<std::uint8_t> HighExportBinary(unsigned output_count) {
  // Relocate only the destination of the compiler fixture's repeat-four
  // UVSW.WRITE at group byte 48. Position still occupies VTXOUT0..3; its
  // second export now ends at the real register-bank boundary.
  auto binary = VaryingsOneVertexPcoBinary();
  Check(binary.size() == 72 && binary[48] == 0x58 && binary[51] == 0x08 &&
        binary[52] == 4, "known native fixture destination field");
  binary[52] = static_cast<std::uint8_t>(output_count - 4);
  const auto program = DecodePcoProgram(ShaderStage::kVertex, binary);
  const auto mask = (UINT64_C(15) << (output_count - 4)) | 15;
  Check(program.summary.vertex_output_mask == mask && program.summary.ends_task,
        "native UVSW encoding writes position and the high four DWORDs");
  const auto execution = ExecuteVertexPco(program.summary, program.instructions,
      std::vector<std::uint32_t>{0xbf000000, 0x3e800000});
  Check(execution.written_mask == mask && execution.emitted && execution.ended_task,
        "native export execution reaches the declared high output register");
  for (unsigned c = 0; c < 4; ++c)
    Check(execution.outputs[output_count - 4 + c] == execution.outputs[c],
          "relocated UVSW exports raw computed position DWORDs");
  return binary;
}

PipelineTxn Make(MemoryPool &pool, unsigned outputs, bool point_size) {
  PipelineState state;
  state.functional_case = FunctionalCase::kDriverPcoTriangles;
  state.stage = PipelineStage::kVertexPdsReady;
  state.driver_describes_attributes = 1;
  state.vertex_pco_abi.vertex_inputs = 4;
  state.vertex_pco_abi.vertex_outputs = outputs;
  state.vertex_pco_abi.temps = 4;
  state.position_output_start = 0; state.position_output_count = 4;
  state.raster_state.point_size_output_start = point_size ? 4 : 0;
  state.raster_state.point_size_output_count = point_size ? 1 : 0;
  state.varying_output_start = 4 + (point_size ? 1 : 0);
  state.varying_output_count = outputs - state.varying_output_start;
  state.fragment_position_count = state.fragment_varying_start = 4;
  state.fragment_pco_abi.coefficients = 4;
  state.driver_varying_bindings_explicit = 1;
  state.driver_varying_binding_count = 0; // TF-only outputs, no FS linkage
  state.counters.vs_invocations = 1;
  state.vertex_code = StoreNewArray(pool, HighExportBinary(std::min(outputs, 64U)));
  VertexAttributeBinding binding;
  binding.source_components = 2; binding.destination_components = 4;
  state.vertex_attribute_bindings = StoreNewArray(pool, std::vector<VertexAttributeBinding>{binding});
  state.drawlist_stats = StoreNewArray(pool, std::vector<DrawListStats>(1));
  PipelineTxn txn; txn.sequence = outputs * 2 + point_size;
  txn.state = pool.Allocate(sizeof(state)); StorePipelineState(pool, txn.state, state);
  return txn;
}

std::vector<std::uint8_t> FragmentAtomicBinary() {
  std::vector<std::uint8_t> bytes;
  const auto append = [&](std::initializer_list<std::uint8_t> group) {
    bytes.insert(bytes.end(), group.begin(), group.end());
  };
  const auto move = [&](std::uint32_t value, std::uint8_t temporary) {
    append({0x86, 0x92, 0x40, 0x13,
            static_cast<std::uint8_t>(value),
            static_cast<std::uint8_t>(value >> 8U),
            static_cast<std::uint8_t>(value >> 16U),
            static_cast<std::uint8_t>(value >> 24U),
            0x00, 0x00, static_cast<std::uint8_t>(0x40U + temporary),
            0xff});
  };
  constexpr std::uint64_t address = UINT64_C(0x12345678000);
  move(static_cast<std::uint32_t>(address), 1);
  move(static_cast<std::uint32_t>(address >> 32U), 2);
  move(7, 3);
  append({0x65, 0xa0, 0x00, 0xe5, 0x00, 0x03, 0x41, 0x42, 0x00,
          0xff});
  append({0x02, 0x80, 0x6a, 0xff});
  append({0x04, 0x80, 0xee, 0x00, 0xf2, 0xff, 0xff, 0xff});
  return bytes;
}

PipelineTxn MakeFragmentStorageAtomic(MemoryPool &pool) {
  PipelineState state;
  state.functional_case = FunctionalCase::kDriverPcoTriangles;
  state.stage = PipelineStage::kParameterBufferReady;
  state.fragment_code = StoreNewArray(pool, FragmentAtomicBinary());
  state.drawlist_stats = StoreNewArray(pool, std::vector<DrawListStats>(1));
  state.raster_state.shader_writes_memory = 1;
  state.graphics_storage[1] = {0, 1, 1, 1, 1};
  state.graphics_buffer_resources =
      StoreNewArray(pool, std::vector<ShaderBufferResource>(1));
  state.graphics_buffer_ranges[1] =
      StoreNewArray(pool, std::vector<ShaderBufferRange>(1));
  PipelineTxn txn;
  txn.sequence = 1;
  txn.state = pool.Allocate(sizeof(state));
  StorePipelineState(pool, txn.state, state);
  return txn;
}
void Retire(MemoryPool &pool, PipelineTxn txn) {
  ReleaseFunctionalPayloads(pool, LoadPipelineState(pool, txn.state));
  pool.Release(txn.state);
}
}  // namespace

int sc_main(int argc, char **argv) {
  try {
    const std::string mode = argc > 1 ? argv[1] : "valid";
    Check(mode == "valid" || mode == "reject65" ||
              mode == "reject65-point-size" ||
              mode == "fragment-storage-atomic",
          "recognized boundary test mode");
    MemoryPool pool;
    const bool fragment_storage_atomic = mode == "fragment-storage-atomic";
    PcoDecoder decoder(fragment_storage_atomic ? "fragment_decoder"
                                               : "vertex_decoder",
                       pool, fragment_storage_atomic ? ShaderStage::kFragment
                                                     : ShaderStage::kVertex);
    sc_core::sc_fifo<PipelineTxn> input("input", 1), output("output", 1);
    decoder.input(input); decoder.output(output);
    if (fragment_storage_atomic) {
      const auto txn = MakeFragmentStorageAtomic(pool);
      Check(input.nb_write(txn),
            "fragment storage atomic enters the actual decoder FIFO");
      sc_core::sc_start(sc_core::sc_time(1, sc_core::SC_US));
      PipelineTxn done;
      Check(output.nb_read(done) && done.sequence == txn.sequence,
            "fragment storage atomic decoder completion retains sequence");
      const auto state = LoadPipelineState(pool, done.state);
      Check(state.stage == PipelineStage::kFragmentDecoded &&
                HasPoolHandle(state.fragment_instructions) &&
                state.fragment_program_summary.instruction_count == 6,
            "fragment storage backing is an observable native atomic contract");
      Retire(pool, done);
    } else if (mode != "valid") {
      const auto txn = Make(pool, 65, mode == "reject65-point-size");
      Check(input.nb_write(txn), "invalid metadata enters actual decoder FIFO");
      bool rejected = false;
      try { sc_core::sc_start(sc_core::sc_time(1, sc_core::SC_US)); }
      catch (const std::exception &error) {
        rejected = std::string(error.what()).find("driver PCO vertex-output linkage is invalid") != std::string::npos;
      }
      Check(rejected, "65 DWORDs reject specifically at the output layout boundary");
      PipelineTxn done;
      Check(!output.nb_read(done), "rejected metadata is never published downstream");
      Check(!HasPoolHandle(LoadPipelineState(pool, txn.state).vertex_instructions),
            "rejection does not publish decoded instruction ownership");
      Retire(pool, txn);
    } else {
      for (unsigned outputs : {63U, 64U}) for (bool point_size : {false, true}) {
        const auto txn = Make(pool, outputs, point_size);
        Check(input.nb_write(txn), "valid metadata enters actual decoder FIFO");
        sc_core::sc_start(sc_core::sc_time(1, sc_core::SC_US));
        PipelineTxn done;
        Check(output.nb_read(done) && done.sequence == txn.sequence,
              "63/64 DWORD decoder completion retains sequence");
        const auto state = LoadPipelineState(pool, done.state);
        Check(state.stage == PipelineStage::kVertexDecoded &&
              state.vertex_pco_abi.vertex_outputs == outputs,
              "full bank is accepted without shrinking ABI or changing stage");
        Check(state.vertex_program_summary.vertex_output_mask ==
                  ((UINT64_C(15) << (outputs - 4)) | 15),
              "decoded high-register mask survives pool transport");
        Check(LoadArray<DrawListStats>(pool, state.drawlist_stats)[0].vertex.program_recorded &&
              HasPoolHandle(state.vertex_instructions), "program statistics and native instructions are published");
        Retire(pool, done); sc_core::sc_start(sc_core::SC_ZERO_TIME);
      }
    }
    Check(pool.bytes_in_flight() == 0 && pool.allocations() == pool.releases(),
          "all decoder payloads released on success and refusal");
    std::cout << "pco_decoder_boundary_test " << mode << ": PASS " << checks << " checks\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << " after " << checks << " checks\n"; return 1;
  }
  return 0;
}
