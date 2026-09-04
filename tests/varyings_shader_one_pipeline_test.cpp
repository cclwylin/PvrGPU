/*
 * GLBench varyings_shader_1 full event-driven SystemC pipeline test.
 *
 * The exact public PowerVR PCO vertex and fragment programs traverse the
 * production hardware route from Submitter through VDM (Vertex Data Master),
 * vertex fetch/USC (Unified Shading Cluster), clip/setup/tiling, Parameter
 * Buffer, ISP (Image Synthesis Processor), PDS (Programmable Data Sequencer),
 * and fragment USC.  FIFO links carry only MemoryPool handles.  The test
 * proves linkage, coefficient construction/copying, spatial quad identity,
 * FITRP interpolation, instruction accounting, and pool ownership without a
 * software-raster or shader-result shortcut.
 */
#include "common/functional_types.h"
#include "common/pipeline_state.h"
#include "fragment/fragment_frontend.h"
#include "fragment/isp.h"
#include "fragment/tile_scheduler.h"
#include "geometry/clip_cull.h"
#include "geometry/parameter_buffer.h"
#include "geometry/tiler.h"
#include "geometry/vdm.h"
#include "geometry/vertex_fetch.h"
#include "pds/pds_engine.h"
#include "pds/vertex_pds_engine.h"
#include "shader/pco_decoder.h"
#include "shader/usc_cluster.h"
#include "shader/usc_slot.h"
#include "submitter.h"

#include <systemc>

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
constexpr std::size_t kCoefficientSetCount = 160;
constexpr std::size_t kInvocationCount = 4096;
constexpr std::size_t kFragmentTaskCount = 1152;
constexpr std::size_t kCopiedCoefficientDwordCount = 23040;

void Check(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(
        "varyings_shader_one pipeline test failed: " + message);
  }
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool SameHandle(PoolHandle left, PoolHandle right) {
  return left.slot == right.slot && left.generation == right.generation;
}

void CheckVertexLinkage(const MemoryPool &pool, const PipelineState &state) {
  Check(state.vertex_program_summary.vertex_input_mask == UINT32_C(0x3),
        "vertex PCO reads exactly VTXIN0..1");
  Check(state.vertex_program_summary.vertex_output_mask == UINT64_C(0xff),
        "vertex PCO writes exactly VTXOUT0..7");
  Check(state.vertex_program_summary.ends_task == 1,
        "vertex PCO emits and ends every task");

  const std::vector<ShaderVaryingBinding> bindings =
      LoadArray<ShaderVaryingBinding>(pool, state.shader_varying_bindings);
  Check(bindings.size() == 1, "one VS-to-FS linkage binding");
  const ShaderVaryingBinding &binding = bindings.front();
  Check(binding.vertex_output_base == 4 &&
            binding.coefficient_set_base == 1 &&
            binding.w_coefficient_set == 0 && binding.component_count == 4 &&
            binding.interpolation == InterpolationMode::kSmooth &&
            binding.reserved[0] == 0 && binding.reserved[1] == 0,
        "exact smooth VTXOUT4..7 to coefficient-set 1..4 linkage");

  const std::vector<VertexBufferResource> resources =
      LoadArray<VertexBufferResource>(pool, state.vertex_buffer_resources);
  Check(resources.size() == 1, "one GLBench VBO resource");
  const std::vector<float> vertices =
      LoadArray<float>(pool, resources.front().data);
  Check(vertices.size() == 2 * kVertexCount, "4x4 float2 lattice has 25 vertices");

  const std::vector<VertexLane> lanes =
      LoadArray<VertexLane>(pool, state.vertex_lanes);
  const std::vector<VertexLaneRef> refs =
      LoadArray<VertexLaneRef>(pool, state.vertex_lane_refs);
  Check(lanes.size() == kVertexCount, "one shaded lane per unique lattice vertex");
  Check(refs.size() == kIndexCount, "one lane reference per index occurrence");

  std::vector<std::uint32_t> lane_vertex(
      lanes.size(), std::numeric_limits<std::uint32_t>::max());
  for (const VertexLaneRef &ref : refs) {
    Check(ref.lane_index < lanes.size() && ref.vertex_index < kVertexCount,
          "lane reference range");
    std::uint32_t &mapped_vertex = lane_vertex[ref.lane_index];
    if (mapped_vertex == std::numeric_limits<std::uint32_t>::max()) {
      mapped_vertex = ref.vertex_index;
    } else {
      Check(mapped_vertex == ref.vertex_index,
            "post-transform cache lane keeps vertex identity");
    }
  }

  for (std::size_t lane_index = 0; lane_index < lanes.size(); ++lane_index) {
    const std::uint32_t vertex_index = lane_vertex[lane_index];
    Check(vertex_index != std::numeric_limits<std::uint32_t>::max(),
          "every shaded lane is referenced");
    const VertexLane &lane = lanes[lane_index];
    const std::uint32_t expected[4] = {
        FloatBits(vertices[2 * vertex_index]),
        FloatBits(vertices[2 * vertex_index + 1]), FloatBits(0.0F),
        FloatBits(1.0F)};
    Check(lane.emitted == 1 && lane.ended == 1,
          "every vertex lane completed UVSW emit/end-task");
    for (std::size_t component = 0; component < 4; ++component) {
      Check(lane.vertex_output[component] == expected[component],
            "VTXOUT0..3 contains exact clip position");
      Check(lane.vertex_output[4 + component] == expected[component],
            "VTXOUT4..7 contains the exact v1 varying");
    }
    for (std::size_t component = 8;
         component < kPcoVertexOutputRegisterCount; ++component) {
      Check(lane.vertex_output[component] == 0,
            "vertex PCO does not write outside VTXOUT0..7");
    }
  }
}

void CheckGeometryAndParameters(const MemoryPool &pool,
                                const PipelineState &state) {
  const std::vector<RasterTriangle> triangles =
      LoadArray<RasterTriangle>(pool, state.raster_triangles);
  const std::vector<std::uint32_t> raster_outputs =
      LoadArray<std::uint32_t>(pool, state.raster_vertex_outputs);
  Check(triangles.size() == kTriangleCount,
        "4x4 lattice emits exactly 32 triangles");
  Check(raster_outputs.size() == kTriangleCount * 3 * 8,
        "clip/setup preserves eight VTXOUT dwords per triangle vertex");
  for (std::size_t index = 0; index < triangles.size(); ++index) {
    const RasterTriangle &triangle = triangles[index];
    Check(triangle.key.submit_ordinal == index &&
              triangle.rasterizable == 1 && triangle.face_culled == 0 &&
              triangle.vertex_output_stride_dwords == 8 &&
              triangle.first_vertex_output_dword == index * 3 * 8,
          "raster triangle identity and VTXOUT range");
  }

  const std::vector<ParameterTriangle> parameters =
      LoadArray<ParameterTriangle>(pool, state.parameter_triangles);
  const std::vector<ParameterCoefficientSet> coefficients =
      LoadArray<ParameterCoefficientSet>(pool, state.parameter_coefficients);
  Check(parameters.size() == kTriangleCount,
        "one ParameterBuffer record per raster triangle");
  Check(coefficients.size() == kCoefficientSetCount,
        "32 triangles times five coefficient sets");
  for (std::size_t index = 0; index < parameters.size(); ++index) {
    const ParameterTriangle &parameter = parameters[index];
    Check(parameter.key.submit_ordinal == index &&
              parameter.rasterizable == 1 && parameter.face_culled == 0 &&
              parameter.first_coefficient_set == index * 5 &&
              parameter.coefficient_set_count == 5,
          "parameter identity and five-set range");
    const ParameterCoefficientSet &reciprocal_w = coefficients[index * 5];
    Check((reciprocal_w.a & UINT32_C(0x7fffffff)) == 0 &&
              (reciprocal_w.b & UINT32_C(0x7fffffff)) == 0 &&
              reciprocal_w.c == FloatBits(1.0F),
          "position-W coefficient plane is exactly one (signed zero is valid)");
  }
  for (const ParameterCoefficientSet &coefficient : coefficients)
    Check(coefficient.pad == 0, "every public A/B/C/PAD set has zero PAD");

  Check(state.counters.c_invocations == kTriangleCount &&
            state.counters.c_primitives == kTriangleCount &&
            state.counters.setup_triangles == kTriangleCount,
        "clip and setup counters come from all 32 triangles");
  Check(state.counters.parameter_coefficient_sets == kCoefficientSetCount &&
            state.counters.parameter_write_bytes ==
                kCoefficientSetCount * sizeof(ParameterCoefficientSet),
        "ParameterBuffer coefficient traffic counters");
}

void CheckPdsWork(const MemoryPool &pool, const PipelineState &state) {
  Check(!SameHandle(state.parameter_coefficients,
                    state.usc_coefficient_banks),
        "PDS USC coefficient bank does not alias ParameterBuffer storage");

  const std::vector<ParameterTriangle> parameters =
      LoadArray<ParameterTriangle>(pool, state.parameter_triangles);
  const std::vector<ParameterCoefficientSet> source =
      LoadArray<ParameterCoefficientSet>(pool, state.parameter_coefficients);
  const std::vector<FragmentInvocation> invocations =
      LoadArray<FragmentInvocation>(pool, state.fragment_invocations);
  const std::vector<FragmentQuad> quads =
      LoadArray<FragmentQuad>(pool, state.fragment_quads);
  const std::vector<UscFragmentTask> tasks =
      LoadArray<UscFragmentTask>(pool, state.usc_fragment_tasks);
  const std::vector<std::uint32_t> copied =
      LoadArray<std::uint32_t>(pool, state.usc_coefficient_banks);

  Check(invocations.size() == kInvocationCount &&
            state.active_fragment_invocations == kInvocationCount &&
            state.counters.ps_invocations == kInvocationCount,
        "one visible fragment invocation per surface pixel");
  Check(quads.size() == kFragmentTaskCount &&
            tasks.size() == kFragmentTaskCount &&
            state.fragment_groups == kFragmentTaskCount,
        "1152 primitive/2x2 spatial quads and PDS tasks");
  Check(copied.size() == kCopiedCoefficientDwordCount,
        "PDS copied 1152 times 20 coefficient dwords");

  std::size_t covered_lane_count = 0;
  for (std::size_t task_index = 0; task_index < tasks.size(); ++task_index) {
    const UscFragmentTask &task = tasks[task_index];
    Check(task.fragment_quad_index == task_index &&
              task.first_coefficient_dword == task_index * 20 &&
              task.coefficient_dword_count == 20 && task.reserved == 0,
          "PDS task identity, offset, and coefficient count");
    const FragmentQuad &quad = quads[task.fragment_quad_index];
    Check(quad.parameter_index < parameters.size() &&
              quad.coverage_mask != 0 && quad.helper_mask == 0 &&
              quad.write_mask == quad.coverage_mask && quad.reserved == 0,
          "PDS quad masks and parameter range");
    const ParameterTriangle &parameter = parameters[quad.parameter_index];
    Check(parameter.key.submit_ordinal == quad.submit_ordinal,
          "PDS task preserves primitive submit identity");

    std::size_t copied_offset = task.first_coefficient_dword;
    for (std::size_t set = 0; set < parameter.coefficient_set_count; ++set) {
      const ParameterCoefficientSet &expected =
          source[parameter.first_coefficient_set + set];
      Check(copied[copied_offset++] == expected.a &&
                copied[copied_offset++] == expected.b &&
                copied[copied_offset++] == expected.c &&
                copied[copied_offset++] == expected.pad,
            "PDS copied each A/B/C/PAD dword exactly");
    }
    Check(copied_offset == task.first_coefficient_dword + 20,
          "PDS task copied exactly five coefficient sets");

    for (std::size_t lane = 0; lane < 4; ++lane) {
      const bool covered = (quad.coverage_mask & (1U << lane)) != 0;
      const std::uint32_t invocation_index = quad.invocation_indices[lane];
      if (!covered) {
        Check(invocation_index == kInvalidFragmentInvocationIndex,
              "uncovered quad lane has no invocation");
        continue;
      }
      Check(invocation_index < invocations.size(),
            "covered quad lane invocation range");
      const FragmentInvocation &invocation = invocations[invocation_index];
      Check(invocation.parameter_index == quad.parameter_index &&
                invocation.quad_id == quad.quad_id &&
                invocation.quad_lane == lane &&
                invocation.submit_ordinal == quad.submit_ordinal,
            "quad-to-invocation identity survives PDS");
      ++covered_lane_count;
    }
  }
  Check(covered_lane_count == kInvocationCount,
        "quad lane maps contain all 4096 invocations exactly once");

  Check(state.counters.pds_coefficient_tasks == kFragmentTaskCount &&
            state.counters.pds_douti_issues == 2 * kFragmentTaskCount &&
            state.counters.usc_coefficient_load_bytes ==
                kCopiedCoefficientDwordCount * sizeof(std::uint32_t),
        "PDS DOUTI and USC coefficient traffic counters");
}

void CheckShaderStatistics(const MemoryPool &pool,
                           const PipelineState &state) {
  Check(state.counters.vs_alu_instructions == 100 &&
            state.counters.vs_tex_instructions == 0 &&
            state.counters.vs_memory_instructions == 225,
        "VS dynamic ALU/Tex/Memory totals are 100/0/225");
  Check(state.counters.fs_alu_instructions == 20480 &&
            state.counters.fs_tex_instructions == 0 &&
            state.counters.fs_memory_instructions == 0,
        "FS dynamic ALU/Tex/Memory totals are 20480/0/0");

  const std::vector<DrawListStats> drawlists =
      LoadArray<DrawListStats>(pool, state.drawlist_stats);
  Check(drawlists.size() == 1 && drawlists.front().drawlist_index == 0,
        "exactly DrawList 0 is reported");
  const DrawListShaderStats &vs = drawlists.front().vertex;
  const DrawListShaderStats &fs = drawlists.front().fragment;
  Check(vs.program_recorded == 1 && vs.executions_recorded == 1 &&
            vs.invocations == kVertexCount &&
            vs.executed_alu_instructions == 100 &&
            vs.executed_tex_instructions == 0 &&
            vs.executed_memory_instructions == 225,
        "DrawList VS totals match executed public PCO");
  Check(fs.program_recorded == 1 && fs.executions_recorded == 1 &&
            fs.invocations == kInvocationCount &&
            fs.executed_alu_instructions == 20480 &&
            fs.executed_tex_instructions == 0 &&
            fs.executed_memory_instructions == 0,
        "DrawList FS totals match FITRP/WDF/MBYP execution");
}

void CheckFragmentOutputs(const MemoryPool &pool,
                          const PipelineState &state) {
  const std::vector<FragmentInvocation> invocations =
      LoadArray<FragmentInvocation>(pool, state.fragment_invocations);
  const std::vector<FragmentOutput> outputs =
      LoadArray<FragmentOutput>(pool, state.fragment_outputs);
  Check(outputs.size() == kInvocationCount &&
            invocations.size() == outputs.size(),
        "USC emitted one fragment output per invocation");

  std::vector<std::uint8_t> pixel_seen(kInvocationCount, 0);
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    const FragmentInvocation &invocation = invocations[index];
    const FragmentOutput &output = outputs[index];
    Check(output.x == invocation.x && output.y == invocation.y &&
              output.primitive_id == invocation.primitive_id &&
              output.parameter_index == invocation.parameter_index &&
              output.submit_ordinal == invocation.submit_ordinal &&
              output.written_mask[0] == 0x0f &&
              output.render_target_count == 1,
          "fragment USC output preserves invocation identity");
    Check(output.x < kWidth && output.y < kHeight,
          "fragment output coordinate is on the surface");
    const std::size_t pixel = output.y * kWidth + output.x;
    Check(pixel_seen[pixel] == 0,
          "HSR and top-left coverage shade each pixel only once");
    pixel_seen[pixel] = 1;

    const float expected_x =
        static_cast<float>(2 * static_cast<int>(output.x) - 63) / 64.0F;
    const float expected_y =
        static_cast<float>(2 * static_cast<int>(output.y) - 63) / 64.0F;
    Check(output.pixel_output[0] == FloatBits(expected_x) &&
              output.pixel_output[1] == FloatBits(expected_y) &&
              output.pixel_output[2] == FloatBits(0.0F) &&
              output.pixel_output[3] == FloatBits(1.0F),
          "FITRP.PIXEL produces exact smooth v1 at the sample center");
  }
  for (const std::uint8_t seen : pixel_seen)
    Check(seen == 1, "all 64x64 pixels have exactly one USC output");
}

} // namespace

int sc_main(int, char **) {
  try {
    MemoryPool pool;
    Options options;
    options.frames = 1;
    options.width = kWidth;
    options.height = kHeight;
    options.test_case = "varyings_shader_1";
    options.cache_bypass = false;

    sc_core::sc_fifo<PipelineTxn> submit_to_vdm("submit_to_vdm", 1);
    sc_core::sc_fifo<PipelineTxn> vdm_to_fetch("vdm_to_fetch", 1);
    sc_core::sc_fifo<PipelineTxn> fetch_to_pds("fetch_to_pds", 1);
    sc_core::sc_fifo<PipelineTxn> vs_pds_to_decoder("vs_pds_to_decoder", 1);
    sc_core::sc_fifo<PipelineTxn> vs_decoder_to_slot(
        "vs_decoder_to_slot", 1);
    sc_core::sc_fifo<PipelineTxn> vs_slot_to_cluster(
        "vs_slot_to_cluster", 1);
    sc_core::sc_fifo<PipelineTxn> vs_cluster_to_clip(
        "vs_cluster_to_clip", 1);
    sc_core::sc_fifo<PipelineTxn> clip_to_tiler("clip_to_tiler", 1);
    sc_core::sc_fifo<PipelineTxn> tiler_to_parameter(
        "tiler_to_parameter", 1);
    sc_core::sc_fifo<PipelineTxn> parameter_to_fs_decoder(
        "parameter_to_fs_decoder", 1);
    sc_core::sc_fifo<PipelineTxn> fs_decoder_to_scheduler(
        "fs_decoder_to_scheduler", 1);
    sc_core::sc_fifo<PipelineTxn> scheduler_to_isp("scheduler_to_isp", 1);
    sc_core::sc_fifo<PipelineTxn> isp_to_frontend("isp_to_frontend", 1);
    sc_core::sc_fifo<PipelineTxn> frontend_to_pds("frontend_to_pds", 1);
    sc_core::sc_fifo<PipelineTxn> pds_to_fs_slot("pds_to_fs_slot", 1);
    sc_core::sc_fifo<PipelineTxn> fs_slot_to_cluster(
        "fs_slot_to_cluster", 1);
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
    fragment_cluster.output(completed_fifo);

    sc_core::sc_start(sc_core::sc_time(1, sc_core::SC_MS));

    PipelineTxn completed;
    Check(completed_fifo.nb_read(completed) && completed.frame == 1 &&
              completed.sequence == 1,
          "full FIFO route completes frame one");
    Check(submitter.fifo_stalls() == 0, "single-frame submit did not stall");
    const PipelineState result = LoadPipelineState(pool, completed.state);
    Check(result.stage == PipelineStage::kFragmentShaded,
          "fragment USC completion stage");
    Check(result.functional_case == FunctionalCase::kVaryingsShaderOne &&
              result.width == kWidth && result.height == kHeight &&
              result.sequence == 1 && result.cache_bypass == 0,
          "exact options and cache-bypass-off state reach completion");
    Check(result.counters.ia_vertices == kIndexCount &&
              result.counters.ia_primitives == kTriangleCount &&
              result.counters.vs_invocations == kVertexCount &&
              result.counters.drawlists == 1,
          "VDM and post-transform reuse counters");

    CheckVertexLinkage(pool, result);
    CheckGeometryAndParameters(pool, result);
    CheckPdsWork(pool, result);
    CheckShaderStatistics(pool, result);
    CheckFragmentOutputs(pool, result);

    ReleaseFunctionalPayloads(pool, result);
    pool.Release(completed.state);
    Check(pool.bytes_in_flight() == 0 &&
              pool.allocations() == pool.releases(),
          "MemoryPool allocations and releases are balanced");

    std::cout << "varyings_shader_one_pipeline_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "varyings_shader_one_pipeline_test: FAIL: " << error.what()
              << '\n';
    return 1;
  }
}
