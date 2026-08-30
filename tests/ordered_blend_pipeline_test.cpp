/*
 * Ordered translucent fragment-pipeline integration test.
 *
 * Two overlapping primitives are seeded at the Parameter Buffer boundary and
 * traverse fragment PCO decode, TileScheduler, ISP/HSR, FragmentFrontend, USC
 * slot/cluster ISS, TPU bypass, and PBE. The shader is an exact Mesa-generated
 * public PowerVR PCO red/alpha=0.5 binary. Correct output proves blending keeps
 * both fragments instead of applying opaque HSR and shading the pixel once.
 */
#include "common/functional_types.h"
#include "common/pipeline_state.h"
#include "fragment/fragment_frontend.h"
#include "fragment/isp.h"
#include "fragment/pbe.h"
#include "fragment/tile_scheduler.h"
#include "pds/pds_engine.h"
#include "shader/pco_decoder.h"
#include "shader/pco_iss.h"
#include "shader/usc_cluster.h"
#include "shader/usc_slot.h"
#include "texture/texture_unit.h"

#include <systemc>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace pvrgpu::stub;

void Check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error("ordered blend pipeline test failed: " + message);
}

ParameterTriangle FullCoverTriangle(std::uint64_t ordinal,
                                    std::uint32_t primitive_id) {
  ParameterTriangle triangle;
  triangle.key.submit_ordinal = ordinal;
  triangle.key.api_primitive_id = primitive_id;
  triangle.front_facing = 1;
  triangle.rasterizable = 1;
  triangle.signed_area = 512LL * 512LL;
  triangle.min_x = 0;
  triangle.min_y = 0;
  triangle.max_x = 1;
  triangle.max_y = 1;
  triangle.window_z[0] = 0.5F;
  triangle.window_z[1] = 0.5F;
  triangle.window_z[2] = 0.5F;
  // Counter-clockwise vertices (0,0), (2,0), (0,2), in 24.8-style fixed
  // point. The center of pixel (0,0) is strictly inside every edge.
  triangle.edge[0] = {0, 512, 0, 1, {}};
  triangle.edge[1] = {-512, -512, 512LL * 512LL, 0, {}};
  triangle.edge[2] = {512, 0, 0, 0, {}};
  return triangle;
}

void ConfigureBlend(RasterState &raster) {
  raster.clear_color[0] = 0.0F;
  raster.clear_color[1] = 0.0F;
  raster.clear_color[2] = 1.0F;
  raster.clear_color[3] = 1.0F;
  raster.sample_count = 1;
  raster.blend.enable = 1;
  raster.blend.rgb_equation = BlendEquation::kAdd;
  raster.blend.alpha_equation = BlendEquation::kAdd;
  raster.blend.source_rgb_factor = BlendFactor::kSourceAlpha;
  raster.blend.destination_rgb_factor = BlendFactor::kOneMinusSourceAlpha;
  raster.blend.source_alpha_factor = BlendFactor::kSourceAlpha;
  raster.blend.destination_alpha_factor = BlendFactor::kOneMinusSourceAlpha;
}

} // namespace

int sc_main(int, char **) {
  try {
    MemoryPool pool;
    PipelineState state;
    state.width = 1;
    state.height = 1;
    state.sequence = 1;
    state.functional_case = FunctionalCase::kFillSolidBlended;
    state.stage = PipelineStage::kParameterBufferReady;
    ConfigureBlend(state.raster_state);
    state.parameter_triangles = StoreNewArray(
        pool, std::vector<ParameterTriangle>{FullCoverTriangle(1, 0),
                                             FullCoverTriangle(2, 1)});
    state.tile_records =
        StoreNewArray(pool, std::vector<TileRecord>{{0, 0, 1, 1, 0, 2}});
    state.tile_primitive_refs = StoreNewArray(
        pool, std::vector<TilePrimitiveRef>{{0, 0, 1}, {1, 0, 2}});
    state.fragment_code = StoreNewArray(
        pool, FillSolidRedHalfAlphaFragmentPcoBinary());
    state.drawlist_stats =
        StoreNewArray(pool, std::vector<DrawListStats>{{}});
    state.counters.drawlists = 1;

    const PoolHandle state_handle = pool.Allocate(sizeof(PipelineState));
    StorePipelineState(pool, state_handle, state);
    const PipelineTxn txn{state_handle, 1, 1};

    sc_core::sc_fifo<PipelineTxn> decode_in("decode_in", 1);
    sc_core::sc_fifo<PipelineTxn> decode_to_scheduler("decode_to_scheduler", 1);
    sc_core::sc_fifo<PipelineTxn> scheduler_to_isp("scheduler_to_isp", 1);
    sc_core::sc_fifo<PipelineTxn> isp_to_frontend("isp_to_frontend", 1);
    sc_core::sc_fifo<PipelineTxn> frontend_to_pds("frontend_to_pds", 1);
    sc_core::sc_fifo<PipelineTxn> pds_to_slot("pds_to_slot", 1);
    sc_core::sc_fifo<PipelineTxn> slot_to_cluster("slot_to_cluster", 1);
    sc_core::sc_fifo<PipelineTxn> cluster_to_texture("cluster_to_texture", 1);
    sc_core::sc_fifo<PipelineTxn> texture_to_pbe("texture_to_pbe", 1);
    sc_core::sc_fifo<PipelineTxn> pbe_out("pbe_out", 1);

    PcoDecoder decoder("fragment_pco_decoder", pool, ShaderStage::kFragment);
    TileScheduler scheduler("tile_scheduler", pool);
    Isp isp("isp", pool);
    FragmentFrontend frontend("fragment_frontend", pool);
    PdsEngine pds("pds", pool);
    UscSlot slot("fragment_usc_slot", pool, ShaderStage::kFragment);
    UscCluster cluster("fragment_usc_cluster", pool, ShaderStage::kFragment);
    TextureUnit texture("texture_unit", pool);
    Pbe pbe("pbe", pool);
    decoder.input(decode_in);
    decoder.output(decode_to_scheduler);
    scheduler.input(decode_to_scheduler);
    scheduler.output(scheduler_to_isp);
    isp.input(scheduler_to_isp);
    isp.output(isp_to_frontend);
    frontend.input(isp_to_frontend);
    frontend.output(frontend_to_pds);
    pds.input(frontend_to_pds);
    pds.output(pds_to_slot);
    slot.input(pds_to_slot);
    slot.output(slot_to_cluster);
    cluster.input(slot_to_cluster);
    cluster.output(cluster_to_texture);
    texture.input(cluster_to_texture);
    texture.output(texture_to_pbe);
    pbe.input(texture_to_pbe);
    pbe.output(pbe_out);

    decode_in.write(txn);
    sc_core::sc_start(sc_core::sc_time(100, sc_core::SC_NS));
    sc_core::sc_start(sc_core::SC_ZERO_TIME);

    PipelineTxn completed;
    Check(pbe_out.nb_read(completed) && completed.sequence == 1,
          "FIFO completion");
    const PipelineState result = LoadPipelineState(pool, state_handle);
    Check(result.stage == PipelineStage::kPbeComplete, "PBE completion stage");
    Check(result.counters.fragment_candidates == 2 &&
              result.counters.hsr_rejected_fragments == 0 &&
              result.counters.covered_pixels == 1 &&
              result.counters.ps_invocations == 2,
          "ISP preserves both translucent fragments");
    Check(result.fragment_groups == 2 && result.counters.usc_groups == 2,
          "one USC quad per primitive");
    Check(result.counters.fs_alu_instructions == 8,
          "two four-instruction fragment ISS executions");
    Check(result.counters.pbe_color_reads == 2 &&
              result.counters.pbe_blended_fragments == 2 &&
              result.counters.pbe_fragment_writes == 2,
          "two PBE destination RMW operations");
    Check(LoadArray<std::uint8_t>(pool, result.pbe_framebuffer) ==
              std::vector<std::uint8_t>({192, 0, 63, 159}),
          "two red half-alpha stores over opaque blue");

    ReleaseFunctionalPayloads(pool, result);
    pool.Release(state_handle);
    Check(pool.bytes_in_flight() == 0 && pool.allocations() == pool.releases(),
          "MemoryPool balanced");
    std::cout << "ordered_blend_pipeline_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ordered_blend_pipeline_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
