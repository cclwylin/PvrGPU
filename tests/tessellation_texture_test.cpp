// Stage-local native TCS/TES -> TPU FIFO transport. Each argv negative is a
// separate SystemC simulation; no captured shader outputs or pixel oracles.
#include "common/tessellation_state.h"
#include "shader/tessellation_texture.h"
#include "texture/texture_unit.h"
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

using namespace pvrgpu::stub;
namespace {
void Check(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
std::uint32_t Bits(float value) { std::uint32_t bits; std::memcpy(&bits, &value, 4); return bits; }
void Word64(std::vector<std::uint32_t> &words, unsigned offset, std::uint64_t value) {
  words.at(offset) = value; words.at(offset + 1) = value >> 32U;
}
struct Client : sc_core::sc_module {
  TessellationTextureRequestPort control_out{"control_out"}, evaluation_out{"evaluation_out"};
  TessellationTextureResponsePort control_in{"control_in"}, evaluation_in{"evaluation_in"};
  MemoryPool &pool;
  GpuMemorySystem &memory;
  std::string mode;
  bool passed = false;
  Client(sc_core::sc_module_name name, MemoryPool &p, GpuMemorySystem &m, std::string value)
      : sc_module(name), pool(p), memory(m), mode(std::move(value)) { SC_THREAD(Run); }
  void Run() {
    for (unsigned epoch = 0; epoch < 2; ++epoch) for (unsigned stage = 0; stage < 2; ++stage) {
      PipelineState state;
      state.functional_case = FunctionalCase::kDriverPcoTriangles;
      state.memory_mode = memory.mode();
      state.stage = PipelineStage::kVertexShaded;
      state.tessellation_control_sampled_texture_count = stage == 0 ? 2 : 0;
      state.tessellation_evaluation_sampled_texture_count = stage == 1 ? 2 : 0;
      TessellationState tess;
      tess.phase = stage == 0 ? TessellationPhase::kSubmitted : TessellationPhase::kDomainComplete;
      const unsigned prefix = stage == 0 ? 8 : 4;
      auto &abi = stage == 0 ? tess.control_abi : tess.evaluation_abi;
      abi.shareds = abi.push_constant_start = abi.uniform_buffer_descriptor_start = prefix + 40;
      std::vector<std::uint32_t> shared(abi.shareds);
      std::vector<TextureResource> resources;
      std::vector<SamplerState> samplers;
      for (unsigned set = 0; set < 2; ++set) {
        const std::uint64_t address = UINT64_C(0x30000000) + stage * 0x10000 + set * 0x1000;
        TextureResource resource;
        resource.gpu_address = address; resource.byte_size = (16 + 4 + 1) * 16;
        resource.format = TextureFormat::kRgba32Float;
        resource.mip_count = 3; resource.descriptor_set = set;
        resource.mip[0] = {4, 4, 64, 0}; resource.mip[1] = {2, 2, 32, 256}; resource.mip[2] = {1, 1, 16, 320};
        std::vector<std::uint8_t> bytes(resource.byte_size);
        for (unsigned level = 0; level < 3; ++level) {
          const std::array<float, 4> color{float(100 * stage + 10 * set + level), -2.0F, 7.5F, 0.25F};
          const auto &mip = resource.mip[level];
          for (unsigned offset = mip.offset_bytes; offset < mip.offset_bytes + mip.height * mip.row_pitch_bytes; offset += 16)
            std::memcpy(bytes.data() + offset, color.data(), 16);
        }
        memory.HostWrite(address, bytes.data(), bytes.size());
        SamplerState sampler;
        sampler.descriptor_set = set;
        sampler.min_filter = sampler.mag_filter = sampler.mip_filter = TextureFilter::kLinear;
        sampler.wrap_u = sampler.wrap_v = TextureWrapMode::kClampToEdge;
        sampler.max_lod_u4_6 = 128;
        const unsigned base = prefix + 20 * set;
        const std::uint64_t image0 = UINT64_C(4) | (UINT64_C(3) << 5U) | (UINT64_C(2) << 8U) |
            (UINT64_C(1) << 11U) | (UINT64_C(61) << 27U) | (UINT64_C(3) << 34U) | (UINT64_C(3) << 48U);
        const std::uint64_t image1 = ((address >> 2U) << 16U) | (UINT64_C(3) << 60U) | (UINT64_C(1) << 15U) | 3U;
        const std::uint64_t sampler0 = UINT64_C(4095) | (UINT64_C(128) << 23U) |
            (UINT64_C(2) << 33U) | (UINT64_C(1) << 36U) | (UINT64_C(1) << 38U) |
            (UINT64_C(1) << 40U) | (UINT64_C(2) << 41U);
        Word64(shared, base, image0); Word64(shared, base + 2, image1);
        shared[base + 4] = resource.byte_size;
        Word64(shared, base + 8, sampler0);
        Word64(shared, base + 16, sampler0 | (UINT64_C(1) << 36U) | (UINT64_C(1) << 38U));
        resources.push_back(resource); samplers.push_back(sampler);
      }
      if (mode == "reject-descriptor") abi.uniform_buffer_descriptor_start -= 1;
      if (mode == "reject-shadow") shared[prefix + 12] = 1;
      if (mode == "reject-count") state.tessellation_control_sampled_texture_count = 1;
      (stage == 0 ? tess.control_shared : tess.evaluation_shared) = StoreNewArray(pool, shared);
      (stage == 0 ? tess.control_texture_resources : tess.evaluation_texture_resources) = StoreNewArray(pool, resources);
      (stage == 0 ? tess.control_sampler_states : tess.evaluation_sampler_states) = StoreNewArray(pool, samplers);
      state.tessellation_state = StoreNewArray(pool, std::vector<TessellationState>{tess});
      PipelineTxn txn;
      txn.state = pool.Allocate(sizeof(state)); txn.sequence = epoch * 2 + stage;
      StorePipelineState(pool, txn.state, state);
      const auto kind = stage == 0 ? ShaderStage::kTessellationControl : ShaderStage::kTessellationEvaluation;
      auto &output = stage == 0 ? control_out : evaluation_out;
      auto &input = stage == 0 ? control_in : evaluation_in;
      const std::array<float, 7> levels{0.0F, -4.0F, 0.0F, 0.5F, 1.0F, 2.0F, 9.0F};
      for (unsigned round = 0; round < 2; ++round) for (unsigned set = 0; set < 2; ++set)
        for (unsigned sample = 0; sample < levels.size(); ++sample) {
          const float level = levels[sample];
          PcoTextureRequest request;
          request.coordinates = {Bits(0.375F), Bits(0.625F), 0};
          request.coordinate_count = request.dimension = 2; request.component_count = 4;
          request.normalized = 1; request.descriptor_set = set;
          // First sample is genuine no-derivative regular sampling; other samples carry raw dynamic LOD.
          request.explicit_lod_present = sample != 0;
          request.explicit_lod = request.explicit_lod_present ? Bits(level) : 0;
          std::copy_n(shared.begin() + prefix + 20 * set, 4, request.texture_state.begin());
          std::copy_n(shared.begin() + prefix + 20 * set + 8, 4, request.sampler_state.begin());
          if (mode == "reject-gather") request.gather = 1;
          if (mode == "reject-dimension") request.dimension = 3;
          std::uint32_t response[4]{};
          const auto before = state.counters;
          SampleTessellationTexture(pool, state, txn,
              mode == "reject-stage" ? ShaderStage::kGeometry : kind, request, response, output, input);
          const float expected_level = level < 0 ? 0 : level > 2 ? 2 : level;
          Check(response[0] == Bits(float(100 * stage + 10 * set) + expected_level) &&
                response[1] == Bits(-2) && response[2] == Bits(7.5F) && response[3] == Bits(0.25F),
                "stage-local mip sample returned wrong raw words");
          Check(state.stage == PipelineStage::kVertexShaded && !HasPoolHandle(state.texture_sample_requests) &&
                !HasPoolHandle(state.texture_sample_responses), "SMP did not restore ownership");
          const auto taps = state.counters.texel_fetches - before.texel_fetches;
          Check(state.counters.texture_requests == before.texture_requests + 1 && (taps == 4 || taps == 8),
                "SMP request/tap accounting mismatch");
          if (memory.mode() == MemoryMode::kDirect)
            Check(state.counters.memory_direct_read_bytes - before.memory_direct_read_bytes == 16 * taps,
                  "SMP bypassed complete direct texel reads");
          else if (memory.mode() == MemoryMode::kBypass)
            Check(state.counters.dram_read_bytes - before.dram_read_bytes == 16 * taps &&
                  state.counters.dram_read_transactions - before.dram_read_transactions == taps,
                  "SMP bypass mode did not issue each real texel read");
          else {
            Check(state.counters.slc_read_accesses - before.slc_read_accesses == taps,
                  "SMP cache lookup count mismatch");
            if (round == 1) Check(state.counters.dram_read_bytes == before.dram_read_bytes,
                                  "warm repeated texture unexpectedly reread DRAM");
          }
          Check((stage == 0 ? state.tessellation_control_texture_request_count : state.tessellation_evaluation_texture_request_count) ==
                  state.counters.texture_requests && state.fragment_texture_request_count == 0 &&
                  state.geometry_texture_request_count == 0 && state.vertex_texture_request_count == 0,
                "tessellation traffic leaked into another shader stage");
        }
      ReleaseFunctionalPayloads(pool, state);
      pool.Release(txn.state);
      Check(pool.bytes_in_flight() == 0 && pool.allocations() == pool.releases(),
            "tessellation transport resource ownership is unbalanced");
    }
    passed = true; sc_core::sc_stop();
  }
};
struct BadResponder : sc_core::sc_module {
  sc_core::sc_fifo_in<PipelineTxn> input{"input"};
  sc_core::sc_fifo_out<PipelineTxn> output{"output"};
  explicit BadResponder(sc_core::sc_module_name name) : sc_module(name) { SC_THREAD(Run); }
  void Run() { auto txn = input.read(); ++txn.sequence; output.write(txn); }
};
}
int sc_main(int argc, char **argv) {
  const std::string mode = argc > 1 ? argv[1] : "cache";
  MemoryPool pool;
  GpuMemorySystem memory(mode == "direct" ? MemoryMode::kDirect : mode == "bypass" ? MemoryMode::kBypass : MemoryMode::kCache);
  sc_core::sc_fifo<PipelineTxn> dummy_in("dummy_in", 1), dummy_out("dummy_out", 1);
  sc_core::sc_fifo<PipelineTxn> control_in("control_in", 1), control_out("control_out", 1);
  sc_core::sc_fifo<PipelineTxn> evaluation_in("evaluation_in", 1), evaluation_out("evaluation_out", 1);
  Client client("client", pool, memory, mode);
  client.control_out(control_in); client.control_in(control_out);
  client.evaluation_out(evaluation_in); client.evaluation_in(evaluation_out);
  TextureUnit unit("unit", pool, &memory);
  unit.input(dummy_in); unit.output(dummy_out);
  std::unique_ptr<BadResponder> bad;
  if (mode == "reject-identity") {
    bad = std::make_unique<BadResponder>("bad"); bad->input(control_in); bad->output(control_out);
  } else {
    unit.tessellation_control_sample_input(control_in); unit.tessellation_control_sample_output(control_out);
  }
  unit.tessellation_evaluation_sample_input(evaluation_in); unit.tessellation_evaluation_sample_output(evaluation_out);
  try {
    sc_core::sc_start(sc_core::sc_time(1, sc_core::SC_MS));
    Check(mode.rfind("reject-", 0) != 0 && client.passed, "test did not reach its expected terminal condition");
  } catch (const std::exception &e) {
    const std::string error = e.what();
    const std::string expected = mode == "reject-identity" ? "completion identity mismatch" :
        mode == "reject-count" ? "resource/request count mismatch" :
        mode == "reject-descriptor" || mode == "reject-shadow" ? "tessellation descriptor/state mismatch" :
        mode == "reject-stage" ? "FIFO/state ownership mismatch" :
        mode == "reject-gather" || mode == "reject-dimension" ? "unsupported tessellation SMP request class" : "";
    if (expected.empty() || error.find(expected) == std::string::npos) {
      std::cerr << error << '\n'; return 1;
    }
  }
  std::cout << "tessellation texture transport " << mode << " PASS\n";
  return 0;
}
