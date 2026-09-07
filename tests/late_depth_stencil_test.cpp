// Real coverage flows ISP -> FragmentFrontend, then explicit shader outputs
// flow PBE -> DRAM. Raster Z deliberately disagrees with shader Z so early
// testing or opaque HSR cannot accidentally produce the expected attachment.
#include "common/functional_types.h"
#include "common/msaa.h"
#include "common/pipeline_state.h"
#include "fragment/fragment_frontend.h"
#include "fragment/isp.h"
#include "fragment/pbe.h"
#include "fragment/pbe_write_back.h"
#include "memory/gpu_memory_system.h"

#include <systemc>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace pvrgpu::stub;

void Check(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

ParameterTriangle Triangle(std::uint32_t index) {
  ParameterTriangle result;
  result.key.submit_ordinal = index + 1;
  result.key.api_primitive_id = index;
  result.front_facing = index == 2 ? 0 : 1;
  result.rasterizable = 1;
  result.max_x = result.max_y = 1;
  const std::array<std::array<std::int64_t, 2>, 3> vertices = {{
      {-2 * kSubpixelScale, -2 * kSubpixelScale},
      {4 * kSubpixelScale, -2 * kSubpixelScale},
      {-2 * kSubpixelScale, 4 * kSubpixelScale}}};
  for (std::size_t edge = 0; edge < 3; ++edge) {
    const auto &a = vertices[edge];
    const auto &b = vertices[(edge + 1) % 3];
    const auto dx = b[0] - a[0];
    const auto dy = b[1] - a[1];
    result.edge[edge] = {-dy, dx, dy * a[0] - dx * a[1],
        static_cast<std::uint8_t>(dy < 0 || (dy == 0 && dx > 0)), {}};
  }
  result.signed_area = 36LL * kSubpixelScale * kSubpixelScale;
  result.depth_plane[2] = FloatBits(0.1F);
  result.depth_plane_valid = 1;
  return result;
}

void RunCase(MemoryPool &pool, GpuMemorySystem &memory,
             sc_core::sc_fifo<PipelineTxn> &isp_input,
             sc_core::sc_fifo<PipelineTxn> &frontend_output,
             sc_core::sc_fifo<PipelineTxn> &pbe_input,
             sc_core::sc_fifo<PipelineTxn> &completion,
             std::uint64_t sequence, std::uint32_t samples,
             std::uint32_t format, int clamp_case) {
  PipelineState state;
  state.width = state.height = 1;
  state.sequence = sequence;
  state.functional_case = FunctionalCase::kDriverPcoTriangles;
  state.stage = PipelineStage::kTilesScheduled;
  state.memory_mode = MemoryMode::kDirect;
  state.raster_state.sample_count = samples;
  state.raster_state.shader_writes_depth = 1;
  state.fragment_early_hsr_safe = 1;
  state.raster_state.depth.test_enable = 1;
  state.raster_state.depth.write_enable = 1;
  state.raster_state.depth.compare_op = clamp_case
      ? DepthCompareOp::kAlways : DepthCompareOp::kLess;
  state.raster_state.stencil.test_enable = 1;
  state.raster_state.stencil.front.pass_op = StencilOp::kIncrementClamp;
  state.raster_state.stencil.front.depth_fail_op = StencilOp::kDecrementClamp;
  state.raster_state.stencil.back.pass_op = StencilOp::kIncrementClamp;
  state.raster_state.stencil.back.depth_fail_op = StencilOp::kInvert;
  state.capture_depth_attachment = 1;
  state.depth_attachment_format = format;
  state.depth_attachment_gpu_address = kDriverPcoSequenceDepthAddressBase;
  state.scheduled_tiles = 1;
  state.parameter_triangles = StoreNewArray(pool,
      std::vector<ParameterTriangle>{Triangle(0), Triangle(1), Triangle(2)});
  state.tile_records = StoreNewArray(pool,
      std::vector<TileRecord>{{0, 0, 1, 1, 0, 3}});
  state.tile_primitive_refs = StoreNewArray(pool,
      std::vector<TilePrimitiveRef>{{0, 0, 1}, {1, 0, 2}, {2, 0, 3}});
  std::vector<std::uint32_t> initial(samples);
  const std::vector<std::uint8_t> initial_stencil(samples, 7);
  for (std::uint32_t sample = 0; sample < samples; ++sample)
    initial[sample] = EncodeDepthAttachmentUnorm(sample % 2 ? 0.5F : 0.3F, format);
  const auto bytes = EncodeDepthAttachmentUnormBytes(initial, format,
      DepthAttachmentHasStencil(format) ? &initial_stencil : nullptr);
  state.depth_attachment_load = StoreNewArray(pool, bytes);
  state.depth_attachment_load_enable = 1;
  state.depth_attachment_load_bytes = bytes.size();
  const PoolHandle handle = pool.Allocate(sizeof(PipelineState));
  StorePipelineState(pool, handle, state);
  const PipelineTxn txn{handle, static_cast<std::uint32_t>(sequence), sequence};
  isp_input.write(txn);
  sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_NS));
  PipelineTxn received;
  Check(frontend_output.nb_read(received), "late-Z frontend completion");
  state = LoadPipelineState(pool, handle);
  Check(state.active_fragment_invocations == 3 && state.counters.ps_invocations == 3,
        "ISP must preserve every late-depth primitive for shader execution");
  Check(state.counters.depth_tested_fragments == 0 &&
            state.counters.stencil_tested_fragments == 0 &&
            state.counters.hsr_rejected_fragments == 0 &&
            !state.depth_attachment_ready,
        "ISP/frontend must neither test nor publish unfinished late-Z state");
  Check(LoadArray<std::uint32_t>(pool, state.isp_depth_attachment) == initial,
        "early stage must leave the initial depth plane unchanged");
  const auto invocations = LoadArray<FragmentInvocation>(pool, state.fragment_invocations);
  std::vector<FragmentOutput> outputs;
  for (const auto &invocation : invocations) {
    Check(invocation.sample_mask == RasterSampleMask(samples),
          "late-Z primitive must retain its actual full sample coverage");
    FragmentOutput result;
    result.primitive_id = invocation.primitive_id;
    result.parameter_index = invocation.parameter_index;
    result.submit_ordinal = invocation.submit_ordinal;
    result.depth_written = 1;
    result.depth = clamp_case ? (clamp_case < 0 ? -2.0F : 2.0F)
        : invocation.primitive_id == 0 ? 0.9F
        : invocation.primitive_id == 1 ? 0.4F : 0.6F;
    result.written_mask[0] = 0xf;
    result.pixel_output[invocation.primitive_id] = FloatBits(1.0F);
    result.pixel_output[3] = FloatBits(1.0F);
    outputs.push_back(result);
  }
  state.fragment_outputs = StoreNewArray(pool, outputs);
  state.stage = PipelineStage::kTextureComplete;
  StorePipelineState(pool, handle, state);
  pbe_input.write(txn);
  sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_NS));
  Check(completion.nb_read(received), "late-Z PBE/DRAM completion");
  state = LoadPipelineState(pool, handle);
  Check(state.depth_attachment_ready == 1 && state.framebuffer_from_dram == 1,
        "late-Z final depth and color must pass through DRAM");
  const auto final_bytes = LoadArray<std::uint8_t>(pool, state.depth_attachment);
  std::vector<std::uint8_t> final_stencil;
  const auto final_depth = DecodeDepthAttachmentUnormBytes(final_bytes, format,
      DepthAttachmentHasStencil(format) ? &final_stencil : nullptr);
  const auto color = LoadArray<std::uint8_t>(pool, state.dram_framebuffer);
  for (std::uint32_t sample = 0; sample < samples; ++sample) {
    const float expected_depth = clamp_case ? (clamp_case < 0 ? 0.0F : 1.0F)
        : sample % 2 ? 0.4F : 0.3F;
    Check(final_depth[sample] == EncodeDepthAttachmentUnorm(expected_depth, format),
          "late-Z must compare/store shader depth in the native format");
    if (DepthAttachmentHasStencil(format))
      Check(final_stencil[sample] == (clamp_case ? 10 : sample % 2 ? 248 : 250),
            "late-Z stencil must apply the correct facing and depth outcome");
    Check(color[sample * 4] == 0 &&
              color[sample * 4 + 1] == (!clamp_case && sample % 2 ? 255 : 0) &&
              color[sample * 4 + 2] == (clamp_case ? 255 : 0),
          "late-Z rejected shader outputs must not write color");
  }
  Check(memory.backing().Read(state.depth_attachment_gpu_address, final_bytes.size()) ==
            final_bytes, "depth readback must agree with authoritative DRAM");
  Check(state.counters.depth_tested_fragments == 3 * samples &&
            state.counters.depth_written_fragments ==
                (clamp_case ? 3 * samples : samples / 2) &&
            state.counters.ps_invocations == 3,
        "late-Z counters separate sample tests/writes from shader invocations");
  ReleaseFunctionalPayloads(pool, state);
  pool.Release(handle);
}
} // namespace

int sc_main(int, char **) {
  try {
    MemoryPool pool;
    GpuMemorySystem memory(MemoryMode::kDirect);
    sc_core::sc_fifo<PipelineTxn> input("input", 1), isp_out("isp_out", 1),
        front_out("front_out", 1), pbe_in("pbe_in", 1), pbe_out("pbe_out", 1),
        completion("completion", 1);
    Isp isp("isp", pool);
    FragmentFrontend frontend("frontend", pool);
    Pbe pbe("pbe", pool);
    PbeWriteBack writeback("writeback", pool, &memory);
    isp.input(input); isp.output(isp_out);
    frontend.input(isp_out); frontend.output(front_out);
    pbe.input(pbe_in); pbe.output(pbe_out);
    writeback.input(pbe_out); writeback.completion(completion);
    std::uint64_t sequence = 0;
    for (std::uint32_t samples : {1U, 4U, 16U})
      for (std::uint32_t format : {kDriverPcoDepthFormatZ16Unorm,
              kDriverPcoDepthFormatZ24UnormS8Uint, kDriverPcoDepthFormatZ32Float,
              kDriverPcoDepthFormatZ32FloatS8X24Uint})
        for (int clamp_case : {-1, 0, 1})
          RunCase(pool, memory, input, front_out, pbe_in, completion,
                  ++sequence, samples, format, clamp_case);
    Check(pool.bytes_in_flight() == 0 && pool.allocations() == pool.releases(),
          "late-Z MemoryPool balance");
    std::cout << "late_depth_stencil_test: PASS (" << sequence << " cases)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "late_depth_stencil_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
