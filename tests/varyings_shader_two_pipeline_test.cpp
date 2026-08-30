/*
 * GLBench varyings_shader_2 complete event-driven SystemC pipeline test.
 *
 * Two smooth vec4 values travel from public PCO VTXOUT4..11 through real
 * clipping, the Parameter Buffer's 1/W and varying/W planes, PDS coefficient
 * banks, fragment USC FITRP/WDF/FADD execution, PBE, enabled SLC and fixed
 * latency DRAM readback. FIFO links carry only MemoryPool handles. The test
 * derives every payload size from the two linkage records and independently
 * checks the final DRAM RGBA bytes; it does not inject pixels or counters.
 */
#include "cache_mmu/slc.h"
#include "common/functional_types.h"
#include "common/pipeline_state.h"
#include "fragment/pbe_write_back.h"
#include "fragment/fragment_frontend.h"
#include "fragment/isp.h"
#include "fragment/pbe.h"
#include "fragment/tile_scheduler.h"
#include "geometry/clip_cull.h"
#include "geometry/parameter_buffer.h"
#include "geometry/tiler.h"
#include "geometry/vdm.h"
#include "geometry/vertex_fetch.h"
#include "memory/dram_model.h"
#include "pds/pds_engine.h"
#include "pds/vertex_pds_engine.h"
#include "shader/pco_decoder.h"
#include "shader/pco_iss.h"
#include "shader/usc_cluster.h"
#include "shader/usc_slot.h"
#include "submitter.h"
#include "texture/texture_unit.h"

#include <systemc>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace pvrgpu::stub;

constexpr std::uint32_t kWidth = 64;
constexpr std::uint32_t kHeight = 64;
constexpr std::size_t kVertexCount = 25;
constexpr std::size_t kIndexCount = 96;
constexpr std::size_t kTriangleCount = 32;
constexpr std::size_t kInvocationCount = 4096;
constexpr std::size_t kFragmentTaskCount = 1152;

void Check(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(
        "varyings_shader_two pipeline test failed: " + message);
  }
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::uint8_t FloatToUnorm8(float value) {
  const float clamped = std::clamp(value, 0.0F, 1.0F);
  return static_cast<std::uint8_t>(
      std::floor(static_cast<double>(clamped) * 255.0 + 0.5));
}

bool SameHandle(PoolHandle left, PoolHandle right) {
  return left.slot == right.slot && left.generation == right.generation;
}

void CheckVertexPath(const MemoryPool &pool, const PipelineState &state) {
  Check(state.vertex_program_summary.vertex_input_mask == UINT32_C(0x3) &&
            state.vertex_program_summary.vertex_output_mask ==
                UINT64_C(0xfff) &&
            state.vertex_program_summary.ends_task == 1,
        "vertex PCO consumes VTXIN0..1 and emits VTXOUT0..11");

  const std::vector<ShaderVaryingBinding> bindings =
      LoadArray<ShaderVaryingBinding>(pool, state.shader_varying_bindings);
  Check(bindings.size() == VaryingVectorCount(state.functional_case),
        "two VS-to-FS bindings");
  for (std::size_t index = 0; index < bindings.size(); ++index)
    Check(IsExactVaryingBinding(state.functional_case, bindings[index], index),
          "contiguous smooth vec4 linkage");

  const std::vector<VertexBufferResource> resources =
      LoadArray<VertexBufferResource>(pool, state.vertex_buffer_resources);
  Check(resources.size() == 1, "one GLBench VBO");
  const std::vector<float> vertices =
      LoadArray<float>(pool, resources.front().data);
  Check(vertices.size() == 2 * kVertexCount, "4x4 float2 lattice");
  const std::vector<VertexLane> lanes =
      LoadArray<VertexLane>(pool, state.vertex_lanes);
  const std::vector<VertexLaneRef> refs =
      LoadArray<VertexLaneRef>(pool, state.vertex_lane_refs);
  Check(lanes.size() == kVertexCount && refs.size() == kIndexCount,
        "post-transform cache lane counts");

  std::vector<std::uint32_t> lane_vertex(
      lanes.size(), std::numeric_limits<std::uint32_t>::max());
  for (const VertexLaneRef &ref : refs) {
    Check(ref.lane_index < lanes.size() && ref.vertex_index < kVertexCount,
          "lane reference range");
    std::uint32_t &mapped = lane_vertex[ref.lane_index];
    if (mapped == std::numeric_limits<std::uint32_t>::max())
      mapped = ref.vertex_index;
    else
      Check(mapped == ref.vertex_index, "lane retains vertex identity");
  }

  for (std::size_t lane_index = 0; lane_index < lanes.size(); ++lane_index) {
    const std::uint32_t vertex = lane_vertex[lane_index];
    Check(vertex != std::numeric_limits<std::uint32_t>::max(),
          "every lane is referenced");
    const float clip[4] = {vertices[2 * vertex], vertices[2 * vertex + 1],
                           0.0F, 1.0F};
    const VertexLane &lane = lanes[lane_index];
    Check(lane.emitted == 1 && lane.ended == 1,
          "vertex task emits and ends");
    for (std::size_t component = 0; component < 4; ++component) {
      Check(lane.vertex_output[component] == FloatBits(clip[component]),
            "clip-position VTXOUT is exact");
      const std::uint32_t half = FloatBits(clip[component] * 0.5F);
      Check(lane.vertex_output[4 + component] == half &&
                lane.vertex_output[8 + component] == half,
            "v1 and v2 contain c/2");
    }
    for (std::size_t output = 12;
         output < kPcoVertexOutputRegisterCount; ++output)
      Check(lane.vertex_output[output] == 0,
            "VS writes no output above VTXOUT11");
  }
}

void CheckParameterAndPdsPath(const MemoryPool &pool,
                              const PipelineState &state) {
  const std::size_t sets_per_triangle =
      VaryingCoefficientSetCount(state.functional_case);
  const std::size_t dwords_per_task =
      VaryingCoefficientDwordCount(state.functional_case);
  Check(sets_per_triangle == 9 && dwords_per_task == 36,
        "two vec4 linkages derive nine sets and 36 dwords");

  const std::vector<RasterTriangle> triangles =
      LoadArray<RasterTriangle>(pool, state.raster_triangles);
  const std::vector<std::uint32_t> raster_outputs =
      LoadArray<std::uint32_t>(pool, state.raster_vertex_outputs);
  Check(triangles.size() == kTriangleCount &&
            raster_outputs.size() == kTriangleCount * 3 * 12,
        "ClipCull preserves twelve VTXOUT dwords per triangle vertex");
  for (std::size_t index = 0; index < triangles.size(); ++index) {
    const RasterTriangle &triangle = triangles[index];
    Check(triangle.key.submit_ordinal == index &&
              triangle.rasterizable == 1 && triangle.face_culled == 0 &&
              triangle.vertex_output_stride_dwords == 12 &&
              triangle.first_vertex_output_dword == index * 3 * 12,
          "raster triangle identity and VTXOUT range");
  }

  const std::vector<ParameterTriangle> parameters =
      LoadArray<ParameterTriangle>(pool, state.parameter_triangles);
  const std::vector<ParameterCoefficientSet> coefficients =
      LoadArray<ParameterCoefficientSet>(pool, state.parameter_coefficients);
  const std::size_t coefficient_set_count =
      kTriangleCount * sets_per_triangle;
  Check(parameters.size() == kTriangleCount &&
            coefficients.size() == coefficient_set_count,
        "one nine-set Parameter Buffer range per triangle");
  for (std::size_t index = 0; index < parameters.size(); ++index) {
    const ParameterTriangle &parameter = parameters[index];
    const std::size_t first = index * sets_per_triangle;
    Check(parameter.first_coefficient_set == first &&
              parameter.coefficient_set_count == sets_per_triangle &&
              parameter.rasterizable == 1 && parameter.face_culled == 0,
          "parameter coefficient range");
    const ParameterCoefficientSet &reciprocal_w = coefficients[first];
    Check((reciprocal_w.a & UINT32_C(0x7fffffff)) == 0 &&
              (reciprocal_w.b & UINT32_C(0x7fffffff)) == 0 &&
              reciprocal_w.c == FloatBits(1.0F),
          "shared position-W plane is one");
    for (std::size_t component = 0; component < 4; ++component) {
      const ParameterCoefficientSet &v1 = coefficients[first + 1 + component];
      const ParameterCoefficientSet &v2 = coefficients[first + 5 + component];
      Check(v1.a == v2.a && v1.b == v2.b && v1.c == v2.c &&
                v1.pad == 0 && v2.pad == 0,
            "independent v1/v2 coefficient planes contain equal c/2 data");
    }
  }
  Check(state.counters.parameter_coefficient_sets == coefficient_set_count &&
            state.counters.parameter_write_bytes ==
                coefficient_set_count * sizeof(ParameterCoefficientSet),
        "Parameter Buffer traffic is payload-derived");

  const std::vector<FragmentInvocation> invocations =
      LoadArray<FragmentInvocation>(pool, state.fragment_invocations);
  const std::vector<FragmentQuad> quads =
      LoadArray<FragmentQuad>(pool, state.fragment_quads);
  const std::vector<UscFragmentTask> tasks =
      LoadArray<UscFragmentTask>(pool, state.usc_fragment_tasks);
  const std::vector<std::uint32_t> copied =
      LoadArray<std::uint32_t>(pool, state.usc_coefficient_banks);
  Check(invocations.size() == kInvocationCount &&
            quads.size() == kFragmentTaskCount &&
            tasks.size() == kFragmentTaskCount,
        "visible pixels and primitive/quad tasks");
  Check(!SameHandle(state.parameter_coefficients,
                    state.usc_coefficient_banks),
        "PDS bank is independently owned");
  Check(copied.size() == tasks.size() * dwords_per_task,
        "PDS copies 36 coefficient dwords per task");
  for (std::size_t task_index = 0; task_index < tasks.size(); ++task_index) {
    const UscFragmentTask &task = tasks[task_index];
    Check(task.fragment_quad_index == task_index &&
              task.first_coefficient_dword == task_index * dwords_per_task &&
              task.coefficient_dword_count == dwords_per_task &&
              task.reserved == 0,
          "PDS descriptor has exact dynamic coefficient range");
    const ParameterTriangle &parameter =
        parameters[quads[task.fragment_quad_index].parameter_index];
    std::size_t copied_offset = task.first_coefficient_dword;
    for (std::size_t set = 0; set < parameter.coefficient_set_count; ++set) {
      const ParameterCoefficientSet &source =
          coefficients[parameter.first_coefficient_set + set];
      Check(copied[copied_offset++] == source.a &&
                copied[copied_offset++] == source.b &&
                copied[copied_offset++] == source.c &&
                copied[copied_offset++] == source.pad,
            "PDS copies raw A/B/C/PAD exactly");
    }
  }
  Check(state.counters.pds_coefficient_tasks == tasks.size() &&
            state.counters.pds_douti_issues == tasks.size() * 2 &&
            state.counters.usc_coefficient_load_bytes ==
                copied.size() * sizeof(std::uint32_t),
        "PDS and USC coefficient traffic counters");
}

void CheckShaderAndDramPath(const MemoryPool &pool,
                            const PipelineState &state) {
  const PcoDecodedProgram vertex =
      Decode(ShaderStage::kVertex, VaryingsTwoVertexPcoBinary());
  const PcoDecodedProgram fragment =
      Decode(ShaderStage::kFragment, VaryingsTwoFragmentPcoBinary());
  const PcoInstructionCounts vertex_dynamic =
      CountPcoInstructions(vertex.instructions, true);
  const PcoInstructionCounts fragment_dynamic =
      CountPcoInstructions(fragment.instructions, true);
  const std::vector<DrawListStats> drawlists =
      LoadArray<DrawListStats>(pool, state.drawlist_stats);
  Check(drawlists.size() == 1, "one DrawList");
  const DrawListShaderStats &vs = drawlists.front().vertex;
  const DrawListShaderStats &fs = drawlists.front().fragment;
  Check(vs.program_recorded == 1 && vs.executions_recorded == 1 &&
            vs.invocations == kVertexCount &&
            vs.executed_alu_instructions == vertex_dynamic.alu * kVertexCount &&
            vs.executed_tex_instructions ==
                vertex_dynamic.texture * kVertexCount &&
            vs.executed_memory_instructions ==
                vertex_dynamic.memory * kVertexCount,
        "DrawList VS totals derive from executed PCO");
  Check(fs.program_recorded == 1 && fs.executions_recorded == 1 &&
            fs.invocations == kInvocationCount &&
            fs.executed_alu_instructions ==
                fragment_dynamic.alu * kInvocationCount &&
            fs.executed_tex_instructions ==
                fragment_dynamic.texture * kInvocationCount &&
            fs.executed_memory_instructions ==
                fragment_dynamic.memory * kInvocationCount,
        "DrawList FS totals derive from executed PCO");
  Check(state.counters.vs_alu_instructions ==
                vs.executed_alu_instructions &&
            state.counters.vs_tex_instructions ==
                vs.executed_tex_instructions &&
            state.counters.vs_memory_instructions ==
                vs.executed_memory_instructions &&
            state.counters.fs_alu_instructions ==
                fs.executed_alu_instructions &&
            state.counters.fs_tex_instructions ==
                fs.executed_tex_instructions &&
            state.counters.fs_memory_instructions ==
                fs.executed_memory_instructions,
        "aggregate shader counters equal DrawList totals");

  const std::vector<FragmentOutput> outputs =
      LoadArray<FragmentOutput>(pool, state.fragment_outputs);
  Check(outputs.size() == kInvocationCount,
        "fragment USC emits every visible pixel");
  std::vector<std::uint8_t> pixel_seen(kInvocationCount, 0);
  for (const FragmentOutput &output : outputs) {
    Check(output.x < kWidth && output.y < kHeight &&
              output.written_mask == 0x0f,
          "fragment output identity");
    const std::size_t pixel = output.y * kWidth + output.x;
    Check(pixel_seen[pixel]++ == 0, "HSR shades each opaque pixel once");
    const float expected_x =
        static_cast<float>(2 * static_cast<int>(output.x) - 63) / 64.0F;
    const float expected_y =
        static_cast<float>(2 * static_cast<int>(output.y) - 63) / 64.0F;
    Check(output.pixel_output[0] == FloatBits(expected_x) &&
              output.pixel_output[1] == FloatBits(expected_y) &&
              output.pixel_output[2] == FloatBits(0.0F) &&
              output.pixel_output[3] == FloatBits(1.0F),
          "two interpolated c/2 vectors sum to exact c");
  }

  Check(state.framebuffer_from_dram == 1 &&
            HasPoolHandle(state.dram_framebuffer) &&
            !HasPoolHandle(state.pbe_framebuffer) &&
            !HasPoolHandle(state.slc_writeback_lines),
        "only DRAM readback remains published");
  const std::vector<std::uint8_t> framebuffer =
      LoadArray<std::uint8_t>(pool, state.dram_framebuffer);
  Check(framebuffer.size() == kInvocationCount * 4,
        "DRAM readback has exact RGBA8 size");
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      const float expected_x =
          static_cast<float>(2 * static_cast<int>(x) - 63) / 64.0F;
      const float expected_y =
          static_cast<float>(2 * static_cast<int>(y) - 63) / 64.0F;
      const std::size_t offset =
          (static_cast<std::size_t>(y) * kWidth + x) * 4;
      Check(framebuffer[offset] == FloatToUnorm8(expected_x) &&
                framebuffer[offset + 1] == FloatToUnorm8(expected_y) &&
                framebuffer[offset + 2] == 0 &&
                framebuffer[offset + 3] == 255,
            "DRAM RGBA is independently reconstructed from sample centers");
    }
  }
  const std::uint64_t line_count =
      CeilDivide(framebuffer.size(), kDramLineWriteBytes);
  Check(state.cache_bypass == 0 &&
            state.counters.slc_line_accesses == line_count &&
            state.counters.slc_write_accesses == line_count &&
            state.counters.slc_writebacks == line_count &&
            state.counters.slc_bypassed == 0 &&
            state.counters.dram_write_transactions == line_count &&
            state.counters.dram_read_transactions == 1 &&
            state.counters.dram_read_bytes == framebuffer.size() &&
            state.counters.framebuffer_dram_readback_bytes ==
                framebuffer.size(),
        "enabled SLC writes lines and PNG source is a DRAM read request");
}

} // namespace

int sc_main(int, char **) {
  try {
    MemoryPool pool;
    Options options;
    options.frames = 1;
    options.width = kWidth;
    options.height = kHeight;
    options.test_case = "varyings_shader_2";
    options.cache_bypass = false;

    sc_core::sc_fifo<PipelineTxn> submit_to_vdm("submit_to_vdm", 1);
    sc_core::sc_fifo<PipelineTxn> vdm_to_fetch("vdm_to_fetch", 1);
    sc_core::sc_fifo<PipelineTxn> fetch_to_pds("fetch_to_pds", 1);
    sc_core::sc_fifo<PipelineTxn> vs_pds_to_decoder("vs_pds_to_decoder", 1);
    sc_core::sc_fifo<PipelineTxn> vs_decoder_to_slot("vs_decoder_to_slot", 1);
    sc_core::sc_fifo<PipelineTxn> vs_slot_to_cluster("vs_slot_to_cluster", 1);
    sc_core::sc_fifo<PipelineTxn> vs_cluster_to_clip("vs_cluster_to_clip", 1);
    sc_core::sc_fifo<PipelineTxn> clip_to_tiler("clip_to_tiler", 1);
    sc_core::sc_fifo<PipelineTxn> tiler_to_parameter("tiler_to_parameter", 1);
    sc_core::sc_fifo<PipelineTxn> parameter_to_fs_decoder(
        "parameter_to_fs_decoder", 1);
    sc_core::sc_fifo<PipelineTxn> fs_decoder_to_scheduler(
        "fs_decoder_to_scheduler", 1);
    sc_core::sc_fifo<PipelineTxn> scheduler_to_isp("scheduler_to_isp", 1);
    sc_core::sc_fifo<PipelineTxn> isp_to_frontend("isp_to_frontend", 1);
    sc_core::sc_fifo<PipelineTxn> frontend_to_pds("frontend_to_pds", 1);
    sc_core::sc_fifo<PipelineTxn> pds_to_fs_slot("pds_to_fs_slot", 1);
    sc_core::sc_fifo<PipelineTxn> fs_slot_to_cluster("fs_slot_to_cluster", 1);
    sc_core::sc_fifo<PipelineTxn> cluster_to_texture("cluster_to_texture", 1);
    sc_core::sc_fifo<PipelineTxn> texture_to_pbe("texture_to_pbe", 1);
    sc_core::sc_fifo<PipelineTxn> pbe_to_pixel_dm("pbe_to_pixel_dm", 1);
    sc_core::sc_fifo<MemoryTxn> pixel_dm_to_slc("pixel_dm_to_slc", 1);
    sc_core::sc_fifo<MemoryTxn> slc_to_dram("slc_to_dram", 1);
    sc_core::sc_fifo<PipelineTxn> completed_fifo("completed_fifo", 1);

    Submitter submitter("submitter", pool, options);
    Vdm vdm("vdm", pool);
    VertexFetch fetch("vertex_fetch", pool);
    PcoDecoder vertex_decoder("vertex_pco_decoder", pool,
                              ShaderStage::kVertex);
    UscSlot vertex_slot("vertex_usc_slot", pool, ShaderStage::kVertex);
    UscCluster vertex_cluster("vertex_usc_cluster", pool,
                              ShaderStage::kVertex);
    ClipCull clip("clip_cull", pool);
    Tiler tiler("tiler", pool);
    ParameterBuffer parameter("parameter_buffer", pool);
    PcoDecoder fragment_decoder("fragment_pco_decoder", pool,
                                ShaderStage::kFragment);
    TileScheduler scheduler("tile_scheduler", pool);
    Isp isp("isp", pool);
    FragmentFrontend frontend("fragment_frontend", pool);
    PdsEngine pds("pds", pool);
    VertexPdsEngine vertex_pds("vertex_pds", pool);
    UscSlot fragment_slot("fragment_usc_slot", pool,
                          ShaderStage::kFragment);
    UscCluster fragment_cluster("fragment_usc_cluster", pool,
                                ShaderStage::kFragment);
    TextureUnit texture("texture_unit", pool);
    Pbe pbe("pbe", pool);
    PbeWriteBack pixel_dm("pbe_write_back", pool);
    Slc slc("slc", pool, false);
    DramModel dram("dram_model", pool);

    submitter.output(submit_to_vdm);
    vdm.input(submit_to_vdm);
    vdm.output(vdm_to_fetch);
    fetch.input(vdm_to_fetch);
    fetch.output(fetch_to_pds);
    vertex_pds.input(fetch_to_pds);
    vertex_pds.output(vs_pds_to_decoder);
    vertex_decoder.input(vs_pds_to_decoder);
    vertex_decoder.output(vs_decoder_to_slot);
    vertex_slot.input(vs_decoder_to_slot);
    vertex_slot.output(vs_slot_to_cluster);
    vertex_cluster.input(vs_slot_to_cluster);
    vertex_cluster.output(vs_cluster_to_clip);
    clip.input(vs_cluster_to_clip);
    clip.output(clip_to_tiler);
    tiler.input(clip_to_tiler);
    tiler.output(tiler_to_parameter);
    parameter.input(tiler_to_parameter);
    parameter.output(parameter_to_fs_decoder);
    fragment_decoder.input(parameter_to_fs_decoder);
    fragment_decoder.output(fs_decoder_to_scheduler);
    scheduler.input(fs_decoder_to_scheduler);
    scheduler.output(scheduler_to_isp);
    isp.input(scheduler_to_isp);
    isp.output(isp_to_frontend);
    frontend.input(isp_to_frontend);
    frontend.output(frontend_to_pds);
    pds.input(frontend_to_pds);
    pds.output(pds_to_fs_slot);
    fragment_slot.input(pds_to_fs_slot);
    fragment_slot.output(fs_slot_to_cluster);
    fragment_cluster.input(fs_slot_to_cluster);
    fragment_cluster.output(cluster_to_texture);
    texture.input(cluster_to_texture);
    texture.output(texture_to_pbe);
    pbe.input(texture_to_pbe);
    pbe.output(pbe_to_pixel_dm);
    pixel_dm.input(pbe_to_pixel_dm);
    pixel_dm.output(pixel_dm_to_slc);
    slc.input(pixel_dm_to_slc);
    slc.output(slc_to_dram);
    dram.input(slc_to_dram);
    dram.output(completed_fifo);

    sc_core::sc_start(sc_core::sc_time(2, sc_core::SC_MS));

    PipelineTxn completed;
    Check(completed_fifo.nb_read(completed) && completed.frame == 1 &&
              completed.sequence == 1,
          "complete FIFO/MemoryTxn route reaches DRAM");
    const PipelineState result = LoadPipelineState(pool, completed.state);
    Check(result.stage == PipelineStage::kFramebufferReady &&
              result.functional_case == FunctionalCase::kVaryingsShaderTwo &&
              result.width == kWidth && result.height == kHeight &&
              result.cache_bypass == 0,
          "Gate 13 reaches framebuffer-ready with cache enabled");
    Check(result.counters.ia_vertices == kIndexCount &&
              result.counters.ia_primitives == kTriangleCount &&
              result.counters.vs_invocations == kVertexCount &&
              result.counters.ps_invocations == kInvocationCount &&
              result.counters.drawlists == 1,
          "geometry and shader invocation counters are work-derived");

    CheckVertexPath(pool, result);
    CheckParameterAndPdsPath(pool, result);
    CheckShaderAndDramPath(pool, result);

    ReleaseFunctionalPayloads(pool, result);
    pool.Release(completed.state);
    Check(pool.bytes_in_flight() == 0 &&
              pool.allocations() == pool.releases(),
          "MemoryPool ownership is balanced");
    std::cout << "varyings_shader_two_pipeline_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "varyings_shader_two_pipeline_test: FAIL: " << error.what()
              << '\n';
    return 1;
  }
}
