// PvrGPU event-driven counter transport stub entry point.
//
// SystemC module declarations and implementations intentionally live in
// one-module/one-header/one-source pairs. This file contains only sc_main and
// top-level elaboration/binding.

#include "cache_mmu/mixed_cache.h"
#include "cache_mmu/mmu_bif.h"
#include "cache_mmu/slc.h"
#include "cache_mmu/texture_cache.h"
#include "cache_mmu/usc_l2_cache.h"
#include "common/functional_types.h"
#include "common/reference_uarch.h"
#include "common/shader_stage.h"
#include "compression/image_compression.h"
#include "data_master/compute_data_master.h"
#include "data_master/domain_data_master.h"
#include "data_master/pixel_data_master.h"
#include "fragment/pbe_write_back.h"
#include "data_master/two_d_data_master.h"
#include "driver_command.h"
#include "firmware/firmware_scheduler.h"
#include "fragment/fragment_frontend.h"
#include "fragment/isp.h"
#include "fragment/pbe.h"
#include "fragment/tile_scheduler.h"
#include "geometry/clip_cull.h"
#include "geometry/parameter_buffer.h"
#include "geometry/tiler.h"
#include "geometry/vdm.h"
#include "geometry/vertex_fetch.h"
#include "host/control_register_bus.h"
#include "host/soc_bus_interface.h"
#include "host/xpu_interface.h"
#include "json_reporter.h"
#include "memory/gpu_memory_system.h"
#include "memory/mem_fabric.h"
#include "memory/dram_model.h"
#include "memory/on_chip_fabric.h"
#include "memory_pool.h"
#include "model_runner.h"
#include "model_types.h"
#include "pds/pds_engine.h"
#include "pds/vertex_pds_engine.h"
#include "shader/pco_decoder.h"
#include "shader/usc_cluster.h"
#include "shader/usc_slot.h"
#include "submitter.h"
#include "texture/texture_unit.h"

#include <systemc>

#include <iostream>
#include <string>

namespace pvrgpu::stub {

bool ConfigureDriverCommandOptions(Options *options, std::string *error) {
  if (!options || !error)
    return false;
  if (!options->driver_command.enabled && options->driver_command_path.empty())
    return true;
  if (!options->driver_command_path.empty()) {
    if (!LoadDriverCommand(options->driver_command_path,
                           &options->driver_command, error)) {
      return false;
    }
  }
  if (!options->driver_command.enabled) {
    *error = "driver command is not populated";
    return false;
  }

  options->frames = 1;
  if (options->driver_command.command == "draw_triangle") {
    options->width = options->driver_command.width;
    options->height = options->driver_command.height;
    options->test_case = "driver_triangle_solid";
  } else if (options->driver_command.command == "draw_indexed_quad") {
    options->width = options->driver_command.framebuffer_width;
    options->height = options->driver_command.framebuffer_height;
    options->test_case = "driver_indexed_quad";
  } else if (options->driver_command.command == "draw_primitive_sequence") {
    options->width = options->driver_command.width;
    options->height = options->driver_command.height;
    const FunctionalCase command_case =
        FunctionalCaseFromName(options->driver_command.test_case);
    options->test_case = IsRasterFunctionalCase(command_case)
                             ? options->driver_command.test_case
                             : "driver_clear_color";
  } else {
    options->width = options->driver_command.width;
    options->height = options->driver_command.height;
    options->test_case = "driver_clear_color";
  }
  return true;
}

}  // namespace pvrgpu::stub

int pvrgpu::stub::RunConfiguredModel(pvrgpu::stub::Options options) {
  using pvrgpu::stub::ClipCull;
  using pvrgpu::stub::ComputeDataMaster;
  using pvrgpu::stub::ControlRegisterBus;
  using pvrgpu::stub::DomainDataMaster;
  using pvrgpu::stub::DramModel;
  using pvrgpu::stub::FirmwareScheduler;
  using pvrgpu::stub::FragmentFrontend;
  using pvrgpu::stub::FunctionalCase;
  using pvrgpu::stub::FunctionalCaseFromName;
  using pvrgpu::stub::GpuMemorySystem;
  using pvrgpu::stub::IsAttributeFetchFamily;
  using pvrgpu::stub::IsRasterFunctionalCase;
  using pvrgpu::stub::IsTriangleSetupFamily;
  using pvrgpu::stub::IsTextureFamily;
  using pvrgpu::stub::IsVaryingsFamily;
  using pvrgpu::stub::Isp;
  using pvrgpu::stub::ImageCompression;
  using pvrgpu::stub::JsonEscape;
  using pvrgpu::stub::JsonReporter;
  using pvrgpu::stub::kSchema;
  using pvrgpu::stub::LoadDriverCommand;
  using pvrgpu::stub::MemFabric;
  using pvrgpu::stub::MemoryPool;
  using pvrgpu::stub::MemoryTxn;
  using pvrgpu::stub::MixedCache;
  using pvrgpu::stub::MmuBif;
  using pvrgpu::stub::OnChipFabric;
  using pvrgpu::stub::ParameterBuffer;
  using pvrgpu::stub::Pbe;
  using pvrgpu::stub::PcoDecoder;
  using pvrgpu::stub::PdsEngine;
  using pvrgpu::stub::PbeWriteBack;
  using pvrgpu::stub::PixelDataMaster;
  using pvrgpu::stub::PipelineTxn;
  using pvrgpu::stub::VertexPdsEngine;
  using pvrgpu::stub::RequiresBackCcwFaceCull;
  using pvrgpu::stub::ShaderStage;
  using pvrgpu::stub::Slc;
  using pvrgpu::stub::SocBusInterface;
  using pvrgpu::stub::Submitter;
  using pvrgpu::stub::TextureCache;
  using pvrgpu::stub::TextureUnit;
  using pvrgpu::stub::Tiler;
  using pvrgpu::stub::TileScheduler;
  using pvrgpu::stub::TwoDDataMaster;
  using pvrgpu::stub::UscCluster;
  using pvrgpu::stub::UscL2Cache;
  using pvrgpu::stub::UscSlot;
  using pvrgpu::stub::Vdm;
  using pvrgpu::stub::VertexFetch;
  using pvrgpu::stub::XpuInterface;

  std::string command_error;
  if (!pvrgpu::stub::ConfigureDriverCommandOptions(&options, &command_error)) {
    std::cerr << command_error << '\n';
    return 2;
  }
  const FunctionalCase functional_case =
      FunctionalCaseFromName(options.test_case);
  if (!IsRasterFunctionalCase(functional_case)) {
    std::cerr << "Unsupported SystemC functional case: " << options.test_case
              << " (supported: fill_solid, fill_solid_blended, "
                 "fill_solid_depth_neq, fill_solid_depth_never, "
                 "triangle_setup, triangle_setup_all_culled, "
                 "triangle_setup_half_culled, attribute_fetch_shader, "
                 "attribute_fetch_shader_2_attr, "
                 "attribute_fetch_shader_4_attr, "
                 "attribute_fetch_shader_8_attr, varyings_shader_1, "
                 "varyings_shader_2, varyings_shader_4, "
                 "varyings_shader_8, fill_tex_nearest, "
                 "fill_tex_bilinear, fill_tex_trilinear_linear_01, "
                 "fill_tex_trilinear_linear_04, "
                 "fill_tex_trilinear_linear_05, driver_clear_color, "
                 "driver_triangle_solid, driver_indexed_quad, "
                 "draw_primitive_sequence)\n";
    return 2;
  }
  const bool is_blended =
      functional_case == FunctionalCase::kFillSolidBlended;
  const bool is_driver_triangle =
      functional_case == FunctionalCase::kDriverTriangleSolid;
  const bool is_driver_indexed_quad =
      functional_case == FunctionalCase::kDriverIndexedQuad;
  const bool is_driver_primitive_sequence =
      options.driver_command.enabled &&
      options.driver_command.command == "draw_primitive_sequence";
  const bool is_triangle_setup = IsTriangleSetupFamily(functional_case);
  const bool is_triangle_setup_all_culled =
      functional_case == FunctionalCase::kTriangleSetupAllCulled;
  const bool is_triangle_setup_half_culled =
      functional_case == FunctionalCase::kTriangleSetupHalfCulled;
  const bool is_attribute_fetch = IsAttributeFetchFamily(functional_case);
  const bool is_varyings = IsVaryingsFamily(functional_case);
  const bool is_texture = IsTextureFamily(functional_case);
  const bool is_bilinear =
      functional_case == FunctionalCase::kFillTexBilinear;
  const bool is_trilinear_linear_01 =
      functional_case == FunctionalCase::kFillTexTrilinearLinear01;
  const bool is_trilinear_linear_04 =
      functional_case == FunctionalCase::kFillTexTrilinearLinear04;
  const bool is_trilinear_linear_05 =
      functional_case == FunctionalCase::kFillTexTrilinearLinear05;
  const bool is_varyings_multi =
      is_varyings && functional_case != FunctionalCase::kVaryingsShaderOne;
  const bool is_two_attribute_fetch =
      functional_case == FunctionalCase::kAttributeFetchShaderTwoAttribute;
  const bool is_four_attribute_fetch =
      functional_case == FunctionalCase::kAttributeFetchShaderFourAttribute;
  const bool is_eight_attribute_fetch =
      functional_case == FunctionalCase::kAttributeFetchShaderEightAttribute;
  const char *mode =
      options.driver_command.enabled
          ? (is_driver_indexed_quad
                 ? "pvrgpu-driver-draw-indexed-quad-phase7"
                 : is_driver_triangle
                    ? "pvrgpu-driver-draw-triangle-phase2"
                    : is_driver_primitive_sequence
                       ? "pvrgpu-driver-draw-primitive-sequence-phase9"
                       : "pvrgpu-driver-clear-color-phase1")
          :
      is_trilinear_linear_05
          ? "systemc-functional-fill-texture-trilinear-linear-05"
          : is_trilinear_linear_04
          ? "systemc-functional-fill-texture-trilinear-linear-04"
          : is_trilinear_linear_01
          ? "systemc-functional-fill-texture-trilinear-linear-01"
          : is_bilinear
          ? "systemc-functional-fill-texture-bilinear"
          : is_texture
          ? "systemc-functional-fill-texture-nearest"
          : is_varyings
          ? "systemc-functional-varyings-shader"
          : is_attribute_fetch
          ? "systemc-functional-attribute-fetch-shader"
          : is_triangle_setup_half_culled
          ? "systemc-functional-triangle-setup-half-culled"
          : is_triangle_setup_all_culled
          ? "systemc-functional-triangle-setup-all-culled"
          : is_triangle_setup
          ? "systemc-functional-triangle-setup"
          : is_blended
          ? "systemc-functional-fill-blended"
          : functional_case == FunctionalCase::kFillSolid
                ? "systemc-functional-fill-solid"
                : "systemc-functional-fill-depth";

  std::cout << "{\"protocol\":\"pvrgpu-jsonl\",\"version\":1"
            << ",\"schema\":\"" << kSchema
            << "\",\"type\":\"hello\",\"backend\":\"pvrgpu\""
            << ",\"mode\":\"" << mode << "\""
            << ",\"renderer\":\"PvrGPU SystemC functional "
            << JsonEscape(options.test_case) << "\""
            << ",\"functional_scope\":\"" << JsonEscape(options.test_case)
            << "-pco-iss-v1\"";
  if (options.driver_command.enabled) {
    const pvrgpu::stub::DriverCommand &command = options.driver_command;
    std::cout << ",\"command_source\":\"pvrgpu-gallium-driver-command\""
              << ",\"driver_command_ingest\":true"
              << ",\"driver_command_schema\":\""
              << JsonEscape(command.schema) << "\""
              << ",\"driver_command_producer\":\""
              << JsonEscape(command.producer) << "\""
              << ",\"driver_command\":\"" << JsonEscape(command.command)
              << "\""
              << ",\"driver_command_case\":\""
              << JsonEscape(command.test_case) << "\""
              << ",\"driver_command_format\":\""
              << JsonEscape(command.format) << "\""
              << ",\"driver_command_width\":" << command.width
              << ",\"driver_command_height\":" << command.height
              << ",\"driver_command_framebuffer_width\":"
              << command.framebuffer_width
              << ",\"driver_command_framebuffer_height\":"
              << command.framebuffer_height;
  } else {
    std::cout << ",\"command_source\":\"builtin-glbench-fixture\"";
  }
  std::cout << ",\"shader_binary\":\"mesa-pco-public-encoding\""
            << ",\"pco_subset\":\""
            << (is_texture
                    ? "fmul-fitrp-wdf-smp-mbyp-uvsw-texture"
                    : is_varyings_multi
                    ? "fmul-fitrp-wdf-fadd-mbyp-uvsw-varying"
                    : is_varyings ? "fitrp-wdf-mbyp-uvsw-varying"
                    : is_two_attribute_fetch || is_four_attribute_fetch ||
                        is_eight_attribute_fetch
                    ? "fadd-mbyp-uvsw-attribute-fetch"
                    : is_attribute_fetch
                    ? "mbyp-uvsw-attribute-fetch"
                    : is_driver_indexed_quad
                    ? "mbyp-uvsw-driver-indexed-quad"
                    : is_driver_triangle
                    ? "mbyp-uvsw-driver-triangle"
                    : is_triangle_setup ? "mbyp-uvsw-triangle-setup"
                                        : "mbyp-uvsw-fill-solid")
            << "\""
            << ",\"mesa_version\":\"26.2.1\""
            << ",\"mesa_commit\":\"da14d65e4499e66468094be52bff9ea0915a695e\""
            << ",\"reference_uarch\":\"pvrgpu-ref-v1\""
            << ",\"uarch_provenance\":\"assumed\""
            << ",\"timing_provenance\":\"uncalibrated\""
            << ",\"cache_bypass\":"
            << (options.cache_bypass ? "true" : "false")
            << ",\"memory_mode\":\""
            << pvrgpu::stub::MemoryModeName(options.memory_mode) << "\""
            << ",\"cache_simulated\":"
            << (options.memory_mode == pvrgpu::stub::MemoryMode::kCache
                    ? "true"
                    : "false")
            << ",\"cache_policy\":\"set-associative-write-back-"
               "write-allocate-true-lru\""
            << ",\"mcu_cache\":{\"capacity_bytes\":"
            << pvrgpu::stub::kMcuCacheConfig.capacity_bytes
            << ",\"line_bytes\":"
            << pvrgpu::stub::kMcuCacheConfig.line_size_bytes
            << ",\"ways\":" << pvrgpu::stub::kMcuCacheConfig.ways
            << ",\"banks\":" << pvrgpu::stub::kMcuCacheConfig.banks << '}'
            << ",\"tcu_cache\":{\"capacity_bytes\":"
            << pvrgpu::stub::kTcuCacheConfig.capacity_bytes
            << ",\"line_bytes\":"
            << pvrgpu::stub::kTcuCacheConfig.line_size_bytes
            << ",\"ways\":" << pvrgpu::stub::kTcuCacheConfig.ways
            << ",\"banks\":" << pvrgpu::stub::kTcuCacheConfig.banks << '}'
            << ",\"slc_cache\":{\"capacity_bytes\":"
            << pvrgpu::stub::kSlcCacheConfig.capacity_bytes
            << ",\"line_bytes\":"
            << pvrgpu::stub::kSlcCacheConfig.line_size_bytes
            << ",\"ways\":" << pvrgpu::stub::kSlcCacheConfig.ways
            << ",\"banks\":" << pvrgpu::stub::kSlcCacheConfig.banks << '}'
            << ",\"usc_l2_cache\":{\"capacity_bytes\":"
            << pvrgpu::stub::kUscL2CacheConfig.capacity_bytes
            << ",\"line_bytes\":"
            << pvrgpu::stub::kUscL2CacheConfig.line_size_bytes
            << ",\"ways\":" << pvrgpu::stub::kUscL2CacheConfig.ways
            << ",\"banks\":" << pvrgpu::stub::kUscL2CacheConfig.banks << '}'
            << ",\"dram_model\":\"unified-fixed-latency-backing-store\""
            << ",\"dram_fixed_latency_cycles\":"
            << pvrgpu::stub::kDramFixedLatencyCycles
            << ",\"framebuffer_source\":\"dram-readback\""
            << ",\"blend_state\":{"
            << "\"enabled\":"
            << (is_blended ? "true" : "false")
            << ",\"rgb_equation\":\"add\""
            << ",\"alpha_equation\":\"add\""
            << ",\"source_rgb_factor\":\""
            << (is_blended ? "source-alpha" : "one") << "\""
            << ",\"destination_rgb_factor\":\""
            << (is_blended ? "one-minus-source-alpha" : "zero") << "\""
            << ",\"source_alpha_factor\":\""
            << (is_blended ? "source-alpha" : "one") << "\""
            << ",\"destination_alpha_factor\":\""
            << (is_blended ? "one-minus-source-alpha" : "zero") << "\"}"
            << ",\"effective_early_hsr\":"
            << (is_blended ? "false" : "true")
            << ",\"face_cull_state\":{"
            << "\"enabled\":"
            << (RequiresBackCcwFaceCull(functional_case) ? "true" : "false")
            << ",\"mode\":\"back\""
            << ",\"front_face\":\"counter-clockwise\"}"
            << ",\"tile_width\":" << pvrgpu::stub::kReferenceUarch.tile_width
            << ",\"tile_height\":" << pvrgpu::stub::kReferenceUarch.tile_height
            << ",\"warning\":\"";
  if (options.driver_command.enabled) {
    if (options.driver_command.command == "draw_indexed_quad") {
      std::cout
          << "Phase7 driver-command indexed-quad path; command is generated "
             "by the Mesa Gallium pvrgpu driver skeleton from a validated "
             "RDC indexed triangle-list quad; SystemC emits a DRAM readback "
             "framebuffer at framebuffer_width/height while reporting "
             "frame-level draw batch counters from viewport width/height "
             "metadata; assumed "
             "uncalibrated uArch timing; framebuffer published only from DRAM "
             "readback";
    } else if (options.driver_command.command == "draw_triangle") {
      std::cout
          << "Phase2 driver-command draw-triangle path; command is generated "
             "by the Mesa Gallium pvrgpu driver skeleton; non-indexed Gallium "
             "draw is lowered to a canonical internal indexed triangle for "
             "the SystemC raster pipeline; assumed uncalibrated uArch timing; "
             "framebuffer published only from DRAM readback";
    } else if (options.driver_command.command == "draw_primitive_sequence") {
      std::cout
          << "Phase9 driver-command primitive-sequence path; command is "
             "generated by the Mesa Gallium pvrgpu driver skeleton from "
             "validated point/line/strip/fan API draw metadata; SystemC "
             "publishes a DRAM readback framebuffer while reporting "
             "API-visible primitive counters from command metadata; assumed "
             "uncalibrated uArch timing; framebuffer published only from DRAM "
             "readback";
    } else {
      std::cout
          << "Phase1 driver-command clear-color path; command is generated by "
             "the Mesa Gallium pvrgpu driver skeleton; implemented through the "
             "validated depth-never clear framebuffer path until the dedicated "
             "PrvGPU clear engine is split out; assumed uncalibrated uArch "
             "timing; framebuffer published only from DRAM readback";
    }
  } else {
    std::cout
        << "function-correct supported raster case; real public PCO subset; "
           "built-in command fixture; no Mesa command ingest; assumed "
           "uncalibrated uArch timing; framebuffer published only from DRAM "
           "readback";
  }
  std::cout << "\""
            << ",\"workload\":\"" << JsonEscape(options.test_case) << "\""
            << ",\"frames\":" << options.frames << "}\n";
  std::cout.flush();

  MemoryPool pool;
  GpuMemorySystem memory(options.memory_mode);

  // DXTP-aligned structural placeholders. They elaborate as distinct SystemC
  // modules but intentionally have no ports, process, timing, or functional
  // connection until their FIFO transaction contracts are defined.
  SocBusInterface soc_bus_interface("soc_bus_interface");
  XpuInterface xpu_interface("xpu_interface");
  ControlRegisterBus control_register_bus("control_register_bus");
  FirmwareScheduler firmware_scheduler("firmware_scheduler");
  ComputeDataMaster compute_data_master("compute_data_master");
  DomainDataMaster domain_data_master("domain_data_master");
  PixelDataMaster pixel_data_master("pixel_data_master");
  TwoDDataMaster two_d_data_master("two_d_data_master");
  ImageCompression image_compression("image_compression");

  // MMU/fabric modules remain structural placeholders. MCU, TCU and USC-L2
  // bind idle traffic because active clients now use the shared GpuMemorySystem
  // API for DRAM backing plus optional SLC simulation.
  MmuBif mmu_bif("mmu_bif");
  MixedCache mixed_cache("mixed_cache", pool, options.cache_bypass);
  TextureCache texture_cache("texture_cache", pool, options.cache_bypass);
  UscL2Cache usc_l2_cache("usc_l2_cache", pool, options.cache_bypass);
  OnChipFabric on_chip_fabric("on_chip_fabric");
  MemFabric mem_fabric("mem_fabric");

  const int fifo_depth =
      static_cast<int>(pvrgpu::stub::kReferenceUarch.fifo_depth);
  sc_core::sc_fifo<PipelineTxn> submit_to_vdm("submit_to_vdm", fifo_depth);
  sc_core::sc_fifo<PipelineTxn> vdm_to_vertex_fetch("vdm_to_vertex_fetch",
                                                    fifo_depth);
  sc_core::sc_fifo<PipelineTxn> vertex_fetch_to_pds(
      "vertex_fetch_to_pds", fifo_depth);
  sc_core::sc_fifo<PipelineTxn> vertex_pds_to_decoder(
      "vertex_pds_to_decoder", fifo_depth);
  sc_core::sc_fifo<PipelineTxn> vertex_decoder_to_slot("vertex_decoder_to_slot",
                                                       fifo_depth);
  sc_core::sc_fifo<PipelineTxn> vertex_slot_to_cluster("vertex_slot_to_cluster",
                                                       fifo_depth);
  sc_core::sc_fifo<PipelineTxn> vertex_cluster_to_clip("vertex_cluster_to_clip",
                                                       fifo_depth);
  sc_core::sc_fifo<PipelineTxn> clip_to_tiler("clip_to_tiler", fifo_depth);
  sc_core::sc_fifo<PipelineTxn> tiler_to_parameter_buffer(
      "tiler_to_parameter_buffer", fifo_depth);
  sc_core::sc_fifo<PipelineTxn> parameter_buffer_to_fragment_decoder(
      "parameter_buffer_to_fragment_decoder", fifo_depth);
  sc_core::sc_fifo<PipelineTxn> fragment_decoder_to_tile_scheduler(
      "fragment_decoder_to_tile_scheduler", fifo_depth);
  sc_core::sc_fifo<PipelineTxn> tile_scheduler_to_isp("tile_scheduler_to_isp",
                                                      fifo_depth);
  sc_core::sc_fifo<PipelineTxn> isp_to_fragment_frontend(
      "isp_to_fragment_frontend", fifo_depth);
  sc_core::sc_fifo<PipelineTxn> fragment_frontend_to_pds(
      "fragment_frontend_to_pds", fifo_depth);
  sc_core::sc_fifo<PipelineTxn> pds_to_fragment_slot(
      "pds_to_fragment_slot", fifo_depth);
  sc_core::sc_fifo<PipelineTxn> fragment_slot_to_cluster(
      "fragment_slot_to_cluster", fifo_depth);
  sc_core::sc_fifo<PipelineTxn> fragment_cluster_to_texture(
      "fragment_cluster_to_texture", fifo_depth);
  sc_core::sc_fifo<PipelineTxn> fragment_cluster_to_texture_samples(
      "fragment_cluster_to_texture_samples", fifo_depth);
  sc_core::sc_fifo<PipelineTxn> texture_samples_to_fragment_cluster(
      "texture_samples_to_fragment_cluster", fifo_depth);
  sc_core::sc_fifo<PipelineTxn> texture_to_pbe("texture_to_pbe", fifo_depth);
  sc_core::sc_fifo<PipelineTxn> pbe_to_pbe_write_back(
      "pbe_to_pbe_write_back", fifo_depth);
  sc_core::sc_fifo<PipelineTxn> dram_to_reporter("dram_to_reporter",
                                                 fifo_depth);

  sc_core::sc_fifo<MemoryTxn> idle_mcu_input("idle_mcu_input", fifo_depth);
  sc_core::sc_fifo<MemoryTxn> idle_mcu_output("idle_mcu_output", fifo_depth);
  sc_core::sc_fifo<MemoryTxn> idle_tcu_input("idle_tcu_input", fifo_depth);
  sc_core::sc_fifo<MemoryTxn> idle_tcu_output("idle_tcu_output", fifo_depth);
  sc_core::sc_fifo<MemoryTxn> idle_usc_l2_input("idle_usc_l2_input",
                                                fifo_depth);
  sc_core::sc_fifo<MemoryTxn> idle_usc_l2_output("idle_usc_l2_output",
                                                 fifo_depth);

  Submitter submitter("submitter", pool, options, &memory);
  Vdm vdm("vdm", pool, &memory);
  VertexFetch vertex_fetch("vertex_fetch", pool, &memory);
  PcoDecoder vertex_decoder("vertex_pco_decoder", pool, ShaderStage::kVertex);
  UscSlot vertex_slot("vertex_usc_slot", pool, ShaderStage::kVertex);
  UscCluster vertex_cluster("vertex_usc_cluster", pool, ShaderStage::kVertex);
  ClipCull clip_cull("clip_cull", pool);
  Tiler tiler("tiler", pool);
  ParameterBuffer parameter_buffer("parameter_buffer", pool, &memory);
  TileScheduler tile_scheduler("tile_scheduler", pool, &memory);
  Isp isp("isp", pool, &memory);
  FragmentFrontend fragment_frontend("fragment_frontend", pool, &memory);
  PdsEngine pds_engine("pds_engine", pool, &memory);
  VertexPdsEngine vertex_pds_engine("vertex_pds_engine", pool);
  PcoDecoder fragment_decoder("fragment_pco_decoder", pool,
                              ShaderStage::kFragment);
  UscSlot fragment_slot("fragment_usc_slot", pool, ShaderStage::kFragment);
  UscCluster fragment_cluster("fragment_usc_cluster", pool,
                              ShaderStage::kFragment);
  TextureUnit texture_unit("texture_unit", pool, &memory);
  Pbe pbe("pbe", pool);
  PbeWriteBack pbe_write_back("pbe_write_back", pool, &memory);
  JsonReporter reporter("json_reporter", options, pool);

  mixed_cache.input(idle_mcu_input);
  mixed_cache.output(idle_mcu_output);
  texture_cache.input(idle_tcu_input);
  texture_cache.output(idle_tcu_output);
  usc_l2_cache.input(idle_usc_l2_input);
  usc_l2_cache.output(idle_usc_l2_output);

  submitter.output(submit_to_vdm);
  vdm.input(submit_to_vdm);
  vdm.output(vdm_to_vertex_fetch);
  vertex_fetch.input(vdm_to_vertex_fetch);
  vertex_fetch.output(vertex_fetch_to_pds);
  vertex_pds_engine.input(vertex_fetch_to_pds);
  vertex_pds_engine.output(vertex_pds_to_decoder);
  vertex_decoder.input(vertex_pds_to_decoder);
  vertex_decoder.output(vertex_decoder_to_slot);
  vertex_slot.input(vertex_decoder_to_slot);
  vertex_slot.output(vertex_slot_to_cluster);
  vertex_cluster.input(vertex_slot_to_cluster);
  vertex_cluster.output(vertex_cluster_to_clip);
  clip_cull.input(vertex_cluster_to_clip);
  clip_cull.output(clip_to_tiler);
  tiler.input(clip_to_tiler);
  tiler.output(tiler_to_parameter_buffer);
  parameter_buffer.input(tiler_to_parameter_buffer);
  parameter_buffer.output(parameter_buffer_to_fragment_decoder);
  fragment_decoder.input(parameter_buffer_to_fragment_decoder);
  fragment_decoder.output(fragment_decoder_to_tile_scheduler);
  tile_scheduler.input(fragment_decoder_to_tile_scheduler);
  tile_scheduler.output(tile_scheduler_to_isp);
  isp.input(tile_scheduler_to_isp);
  isp.output(isp_to_fragment_frontend);
  fragment_frontend.input(isp_to_fragment_frontend);
  fragment_frontend.output(fragment_frontend_to_pds);
  pds_engine.input(fragment_frontend_to_pds);
  pds_engine.output(pds_to_fragment_slot);
  fragment_slot.input(pds_to_fragment_slot);
  fragment_slot.output(fragment_slot_to_cluster);
  fragment_cluster.input(fragment_slot_to_cluster);
  fragment_cluster.texture_request_output(
      fragment_cluster_to_texture_samples);
  fragment_cluster.texture_response_input(
      texture_samples_to_fragment_cluster);
  fragment_cluster.output(fragment_cluster_to_texture);
  texture_unit.sample_input(fragment_cluster_to_texture_samples);
  texture_unit.sample_output(texture_samples_to_fragment_cluster);
  texture_unit.input(fragment_cluster_to_texture);
  texture_unit.output(texture_to_pbe);
  pbe.input(texture_to_pbe);
  pbe.output(pbe_to_pbe_write_back);
  pbe_write_back.input(pbe_to_pbe_write_back);
  pbe_write_back.completion(dram_to_reporter);
  reporter.input(dram_to_reporter);

  sc_core::sc_start();
  return !reporter.failed() && pool.allocations() == pool.releases() &&
                 pool.bytes_in_flight() == 0
             ? 0
             : 1;
}

int sc_main(int argc, char **argv) {
  pvrgpu::stub::Options options;
  if (!pvrgpu::stub::ParseOptions(argc, argv, &options))
    return 2;
  return pvrgpu::stub::RunConfiguredModel(options);
}
