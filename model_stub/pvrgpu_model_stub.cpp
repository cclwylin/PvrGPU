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
#include "model_job.h"
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
#include <memory>
#include <string>

namespace pvrgpu::stub {

namespace {

bool IsIdeasPcoSequenceCommand(const DriverCommand &command) {
  const std::string &name = command.test_case;
  return command.command == "draw_pco_triangles" &&
         (name == "ideas" || name.rfind("ideas.", 0) == 0 ||
          name.find(".ideas.") != std::string::npos);
}

bool IdeasDepthStateMatchesOrdinal(const DriverCommand &command,
                                   std::size_t ordinal) {
  const bool depth_enabled =
      ordinal >= kDriverPcoIdeasDepthEnabledFirstCommand &&
      ordinal < kDriverPcoIdeasDepthEnabledEndCommand;
  return command.depth_clear_bits == UINT32_C(0x3f800000) &&
         command.depth_format != 0 &&
         command.depth_enable == (depth_enabled ? 1U : 0U) &&
         command.depth_write == (depth_enabled ? 1U : 0U) &&
         command.depth_func == (depth_enabled ? 3U : 0U);
}

}  // namespace

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
  if (!options->driver_commands.empty()) {
    const bool ideas = IsIdeasPcoSequenceCommand(options->driver_command);
    const bool generic =
        options->driver_command.command == "draw_pco_sequence";
    if ((!ideas && !generic) ||
        options->driver_commands.size() >
            kDriverPcoMaximumSequenceCommands ||
        (ideas &&
         (options->driver_commands.size() !=
              kDriverPcoIdeasSequenceCommands ||
          options->driver_command.draw_count !=
              options->driver_commands.size())) ||
        (generic &&
         (options->driver_commands.empty() ||
          options->driver_commands.size() >
              kDriverPcoMaximumNestedSequenceCommands))) {
      *error = "driver PCO command sequence metadata is invalid";
      return false;
    }
    for (std::size_t ordinal = 0;
         ordinal < options->driver_commands.size(); ++ordinal) {
      const DriverCommand &command = options->driver_commands[ordinal];
      if (!command.enabled || command.command != "draw_pco_triangles" ||
          command.test_case != options->driver_command.test_case ||
          command.format != options->driver_command.format ||
          (ideas &&
           (command.framebuffer_width !=
                options->driver_command.framebuffer_width ||
            command.framebuffer_height !=
                options->driver_command.framebuffer_height ||
            command.width != options->driver_command.width ||
            command.height != options->driver_command.height ||
            command.clear_color_bits !=
                options->driver_command.clear_color_bits ||
            !IdeasDepthStateMatchesOrdinal(command, ordinal)))) {
        *error = "driver PCO command sequence members are incompatible";
        return false;
      }
    }
    if (generic &&
        (options->driver_commands.back().framebuffer_width !=
             options->driver_command.framebuffer_width ||
         options->driver_commands.back().framebuffer_height !=
             options->driver_command.framebuffer_height ||
         options->driver_commands.back().width !=
             options->driver_command.width ||
         options->driver_commands.back().height !=
             options->driver_command.height)) {
      *error = "driver PCO sequence final target is incompatible";
      return false;
    }
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
  } else if (options->driver_command.command == "draw_textured_triangles") {
    options->width = options->driver_command.framebuffer_width;
    options->height = options->driver_command.framebuffer_height;
    options->test_case = "driver_textured_triangles";
  } else if (options->driver_command.command == "draw_pco_triangles" ||
             options->driver_command.command == "draw_pco_sequence") {
    options->width = options->driver_command.framebuffer_width;
    options->height = options->driver_command.framebuffer_height;
    options->test_case = "driver_pco_triangles";
  } else {
    options->width = options->driver_command.width;
    options->height = options->driver_command.height;
    options->test_case = "driver_clear_color";
  }
  return true;
}

}  // namespace pvrgpu::stub

namespace pvrgpu::stub {

/*
 * Validate the flush's options and announce it on the counter stream.
 *
 * This is per flush, not per process: a model that stays alive across
 * readbacks still describes each submission it is given.
 */
int AnnounceModelConfiguration(pvrgpu::stub::Options &options) {
  // A sequence carries its per-draw state in the nested commands; the root
  // command is only the summary.  Report the first draw that enables the
  // stencil test, because a sequence commonly opens with draws that do not and
  // reporting those says nothing about the state the test runs under.
  const pvrgpu::stub::DriverCommand &first_draw = [&options]()
      -> const pvrgpu::stub::DriverCommand & {
    for (const pvrgpu::stub::DriverCommand &draw : options.driver_commands) {
      if (draw.stencil_enable != 0)
        return draw;
    }
    return options.driver_commands.empty() ? options.driver_command
                                           : options.driver_commands.front();
  }();
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
                 "driver_textured_triangles, driver_pco_triangles)\n";
    return 2;
  }
  const bool is_blended =
      functional_case == FunctionalCase::kFillSolidBlended;
  const bool is_driver_triangle =
      functional_case == FunctionalCase::kDriverTriangleSolid;
  const bool is_driver_indexed_quad =
      functional_case == FunctionalCase::kDriverIndexedQuad;
  const bool is_driver_textured_triangles =
      functional_case == FunctionalCase::kDriverTexturedTriangles;
  const bool is_driver_pco_triangles =
      functional_case == FunctionalCase::kDriverPcoTriangles;
  const bool is_driver_pco_varying =
      is_driver_pco_triangles &&
      options.driver_command.varying_output_count != 0;
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
                 : is_driver_pco_triangles
                    ? "pvrgpu-driver-draw-pco-triangles"
                 : is_driver_textured_triangles
                    ? "pvrgpu-driver-draw-textured-triangles-phase10"
                 : is_driver_triangle
                    ? "pvrgpu-driver-draw-triangle-phase2"
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
    if (command.command == "draw_textured_triangles") {
      std::cout << ",\"driver_texture_width\":" << command.texture_width
                << ",\"driver_texture_height\":" << command.texture_height;
    } else if (command.command == "draw_pco_triangles") {
      std::cout << ",\"driver_first_vertex\":" << command.first_vertex
                << ",\"driver_vertex_count\":" << command.vertex_count
                << ",\"driver_vertex_stride\":" << command.vertex_stride
                << ",\"driver_vertex_pco_bytes\":"
                << command.vertex_pco.size()
                << ",\"driver_fragment_pco_bytes\":"
                << command.fragment_pco.size()
                << ",\"driver_varying_output_start\":"
                << command.varying_output_start
                << ",\"driver_varying_output_count\":"
                << command.varying_output_count
                << ",\"driver_fragment_varying_start\":"
                << command.fragment_varying_start
                << ",\"driver_fragment_varying_count\":"
                << command.fragment_varying_count;
      if (!options.driver_commands.empty()) {
        std::cout << ",\"driver_command_sequence_length\":"
                  << options.driver_commands.size();
      }
    } else if (command.command == "draw_pco_sequence") {
      std::uint64_t resource_count = 0;
      for (const pvrgpu::stub::DriverCommand &physical :
           options.driver_commands) {
        resource_count += physical.sampled_textures.size();
      }
      std::cout << ",\"driver_command_sequence_length\":"
                << options.driver_commands.size()
                << ",\"driver_command_sequence_resources\":"
                << resource_count;
    }
  } else {
    std::cout << ",\"command_source\":\"builtin-glbench-fixture\"";
  }
  std::cout << ",\"shader_binary\":\"mesa-pco-public-encoding\""
            << ",\"pco_subset\":\""
            << (is_driver_pco_varying
                    ? "driver-pco-dynamic-smooth-varying"
                    : is_driver_pco_triangles
                    ? "conditionals-public-pco"
                    : is_texture
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
              << ((RequiresBackCcwFaceCull(functional_case) ||
                   is_driver_textured_triangles || is_driver_pco_triangles)
                    ? "true"
                    : "false")
            << ",\"mode\":\"back\""
            << ",\"front_face\":\""
              << ((is_driver_textured_triangles || is_driver_pco_triangles)
                      ? "clockwise"
                      : "counter-clockwise")
            << "\"}"
            << ",\"stencil_state\":{"
            // A sequence keeps its per-draw state in the nested commands; the
            // root is only the summary, so report the first draw that has one.
            << "\"enabled\":"
              << (first_draw.stencil_enable != 0 ? "true" : "false")
            << ",\"depth_format\":" << first_draw.depth_format
            << ",\"clear\":" << first_draw.stencil_clear
            << ",\"front_func\":" << first_draw.stencil_func[0]
            << ",\"front_value_mask\":" << first_draw.stencil_value_mask[0]
            << ",\"front_write_mask\":" << first_draw.stencil_write_mask[0]
            << ",\"front_ref\":" << first_draw.stencil_ref[0]
            << ",\"inherited_clears\":" << first_draw.attachment_clears.size()
            << ",\"sequence_clears\":" << [&options]() {
                 std::size_t total = 0;
                 for (const pvrgpu::stub::DriverCommand &draw :
                      options.driver_commands) {
                   total += draw.attachment_clears.size();
                 }
                 return total;
               }()
            << "}"
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
    } else if (options.driver_command.command == "draw_pco_triangles" ||
               options.driver_command.command == "draw_pco_sequence") {
      if (!options.driver_commands.empty()) {
        std::cout
            << "Ordered native driver-command PCO sequence; every owned "
               "draw executes through VertexFetch and the USC ISS, and the "
               "logical report aggregates physical DrawList evidence; "
               "assumed uncalibrated uArch timing; final-subdraw framebuffer "
               "published only from DRAM readback";
      } else {
      std::cout
          << "Driver-command non-indexed triangle-list path; the raw float3 "
             "position or interleaved float3 position/normal VBO, real Mesa "
             "PCO VS/FS binaries, dynamic stage ABI/shared registers, and "
             "generic smooth-varying linkage were synchronously copied from "
             "the Gallium API v6 submission and execute through VertexFetch "
             "and the USC ISS; "
             "assumed uncalibrated uArch timing; framebuffer published only "
             "from DRAM readback";
      }
    } else if (options.driver_command.command == "draw_textured_triangles") {
      std::cout
          << "Driver-command textured triangle-list path; six non-indexed "
             "position/texcoord occurrences and an exact RGBA8 sidecar are "
             "consumed by the public fill_tex_nearest PCO VS/FS through a "
             "single-mip normalized clamp-to-edge sampler; assumed "
             "uncalibrated uArch timing; framebuffer published only from "
             "DRAM readback";
    } else if (options.driver_command.command == "draw_triangle") {
      std::cout
          << "Phase2 driver-command draw-triangle path; command is generated "
             "by the Mesa Gallium pvrgpu driver skeleton; non-indexed Gallium "
             "draw is lowered to a canonical internal indexed triangle for "
             "the SystemC raster pipeline; assumed uncalibrated uArch timing; "
             "framebuffer published only from DRAM readback";
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
  return 0;
}

} // namespace pvrgpu::stub

namespace pvrgpu::stub {
namespace {

int ModelFifoDepth() {
  return static_cast<int>(pvrgpu::stub::kReferenceUarch.fifo_depth);
}

/*
 * The elaborated model.
 *
 * SystemC elaborates once per process.  Building the module set as locals and
 * calling `sc_start()` once meant a second submission was answered with
 *
 *     Error: (E529) insert module failed: elaboration done
 *
 * so the whole simulation had to be deferred to `atexit` -- one process, one
 * run, and it had to be last.  That is why a readback came back black: the
 * draw had not happened yet.
 *
 * `sc_start()`, on the other hand, may be called as often as you like, as long
 * as you elaborate once and do not `sc_stop()`.  Control returns to the caller
 * each time and the simulation resumes with its state intact.  So the module
 * and FIFO set lives here instead, outliving any one flush.  The wiring is
 * unchanged; only where the objects live has moved.  This is also the more
 * faithful arrangement: the model stays alive between draws rather than being
 * rebuilt, which is what hardware does.
 */
class ModelSession {
public:
  explicit ModelSession(MemoryMode memory_mode, bool cache_bypass);

  MemoryMode memory_mode() const { return memory_mode_; }

  // Runs one flush.  Returns 0 when the reporter published the job's records
  // and the MemoryPool balanced, and non-zero otherwise, with `error` set.
  int Run(const Options &options, ModelFramebuffer *framebuffer,
          std::string *error);

  // True once a flush has failed.  A failure can leave a process part-way
  // through a submission it will never finish, and there is no way to tell a
  // SystemC process to start over, so the session refuses further work rather
  // than reporting a later flush against a pipeline in an unknown state.
  bool poisoned() const { return poisoned_; }

  void Shutdown();

private:
  MemoryMode memory_mode_;
  bool cache_bypass_;
  bool started_ = false;
  bool stopped_ = false;
  bool poisoned_ = false;

  MemoryPool pool;
  GpuMemorySystem memory;
  ModelJob job;

  // DXTP-aligned structural placeholders. They elaborate as distinct SystemC
  // modules but intentionally have no ports, process, timing, or functional
  // connection until their FIFO transaction contracts are defined.
  SocBusInterface soc_bus_interface{"soc_bus_interface"};
  XpuInterface xpu_interface{"xpu_interface"};
  ControlRegisterBus control_register_bus{"control_register_bus"};
  FirmwareScheduler firmware_scheduler{"firmware_scheduler"};
  ComputeDataMaster compute_data_master{"compute_data_master"};
  DomainDataMaster domain_data_master{"domain_data_master"};
  PixelDataMaster pixel_data_master{"pixel_data_master"};
  TwoDDataMaster two_d_data_master{"two_d_data_master"};
  ImageCompression image_compression{"image_compression"};

  // MMU/fabric modules remain structural placeholders. MCU, TCU and USC-L2
  // bind idle traffic because active clients now use the shared GpuMemorySystem
  // API for DRAM backing plus optional SLC simulation.
  MmuBif mmu_bif{"mmu_bif"};
  MixedCache mixed_cache{"mixed_cache", pool, cache_bypass_};
  TextureCache texture_cache{"texture_cache", pool, cache_bypass_};
  UscL2Cache usc_l2_cache{"usc_l2_cache", pool, cache_bypass_};
  OnChipFabric on_chip_fabric{"on_chip_fabric"};
  MemFabric mem_fabric{"mem_fabric"};

  sc_core::sc_fifo<PipelineTxn> submit_to_vdm{"submit_to_vdm",
                                              ModelFifoDepth()};
  sc_core::sc_fifo<PipelineTxn> vdm_to_vertex_fetch{"vdm_to_vertex_fetch",
                                                    ModelFifoDepth()};
  sc_core::sc_fifo<PipelineTxn> vertex_fetch_to_pds{"vertex_fetch_to_pds",
                                                    ModelFifoDepth()};
  sc_core::sc_fifo<PipelineTxn> vertex_pds_to_decoder{"vertex_pds_to_decoder",
                                                      ModelFifoDepth()};
  sc_core::sc_fifo<PipelineTxn> vertex_decoder_to_slot{"vertex_decoder_to_slot",
                                                       ModelFifoDepth()};
  sc_core::sc_fifo<PipelineTxn> vertex_slot_to_cluster{"vertex_slot_to_cluster",
                                                       ModelFifoDepth()};
  sc_core::sc_fifo<PipelineTxn> vertex_cluster_to_texture_samples{
      "vertex_cluster_to_texture_samples", ModelFifoDepth()};
  sc_core::sc_fifo<PipelineTxn> texture_samples_to_vertex_cluster{
      "texture_samples_to_vertex_cluster", ModelFifoDepth()};
  sc_core::sc_fifo<PipelineTxn> vertex_cluster_to_clip{"vertex_cluster_to_clip",
                                                       ModelFifoDepth()};
  sc_core::sc_fifo<PipelineTxn> clip_to_tiler{"clip_to_tiler",
                                              ModelFifoDepth()};
  sc_core::sc_fifo<PipelineTxn> tiler_to_parameter_buffer{
      "tiler_to_parameter_buffer", ModelFifoDepth()};
  sc_core::sc_fifo<PipelineTxn> parameter_buffer_to_fragment_decoder{
      "parameter_buffer_to_fragment_decoder", ModelFifoDepth()};
  sc_core::sc_fifo<PipelineTxn> fragment_decoder_to_tile_scheduler{
      "fragment_decoder_to_tile_scheduler", ModelFifoDepth()};
  sc_core::sc_fifo<PipelineTxn> tile_scheduler_to_isp{"tile_scheduler_to_isp",
                                                      ModelFifoDepth()};
  sc_core::sc_fifo<PipelineTxn> isp_to_fragment_frontend{
      "isp_to_fragment_frontend", ModelFifoDepth()};
  sc_core::sc_fifo<PipelineTxn> fragment_frontend_to_pds{
      "fragment_frontend_to_pds", ModelFifoDepth()};
  sc_core::sc_fifo<PipelineTxn> pds_to_fragment_slot{"pds_to_fragment_slot",
                                                     ModelFifoDepth()};
  sc_core::sc_fifo<PipelineTxn> fragment_slot_to_cluster{
      "fragment_slot_to_cluster", ModelFifoDepth()};
  sc_core::sc_fifo<PipelineTxn> fragment_cluster_to_texture{
      "fragment_cluster_to_texture", ModelFifoDepth()};
  sc_core::sc_fifo<PipelineTxn> fragment_cluster_to_texture_samples{
      "fragment_cluster_to_texture_samples", ModelFifoDepth()};
  sc_core::sc_fifo<PipelineTxn> texture_samples_to_fragment_cluster{
      "texture_samples_to_fragment_cluster", ModelFifoDepth()};
  sc_core::sc_fifo<PipelineTxn> texture_to_pbe{"texture_to_pbe",
                                               ModelFifoDepth()};
  sc_core::sc_fifo<PipelineTxn> pbe_to_pbe_write_back{"pbe_to_pbe_write_back",
                                                      ModelFifoDepth()};
  sc_core::sc_fifo<PipelineTxn> dram_to_reporter{"dram_to_reporter",
                                                 ModelFifoDepth()};

  sc_core::sc_fifo<MemoryTxn> idle_mcu_input{"idle_mcu_input",
                                             ModelFifoDepth()};
  sc_core::sc_fifo<MemoryTxn> idle_mcu_output{"idle_mcu_output",
                                              ModelFifoDepth()};
  sc_core::sc_fifo<MemoryTxn> idle_tcu_input{"idle_tcu_input",
                                             ModelFifoDepth()};
  sc_core::sc_fifo<MemoryTxn> idle_tcu_output{"idle_tcu_output",
                                              ModelFifoDepth()};
  sc_core::sc_fifo<MemoryTxn> idle_usc_l2_input{"idle_usc_l2_input",
                                                ModelFifoDepth()};
  sc_core::sc_fifo<MemoryTxn> idle_usc_l2_output{"idle_usc_l2_output",
                                                 ModelFifoDepth()};

  sc_core::sc_event sequence_completion;
  Options elaboration_options_;
  Submitter submitter{"submitter",     pool,
                      elaboration_options_, &memory,
                      &sequence_completion, &job};
  Vdm vdm{"vdm", pool, &memory};
  VertexFetch vertex_fetch{"vertex_fetch", pool, &memory};
  PcoDecoder vertex_decoder{"vertex_pco_decoder", pool, ShaderStage::kVertex};
  UscSlot vertex_slot{"vertex_usc_slot", pool, ShaderStage::kVertex};
  UscCluster vertex_cluster{"vertex_usc_cluster", pool, ShaderStage::kVertex};
  ClipCull clip_cull{"clip_cull", pool};
  Tiler tiler{"tiler", pool};
  ParameterBuffer parameter_buffer{"parameter_buffer", pool, &memory};
  TileScheduler tile_scheduler{"tile_scheduler", pool, &memory};
  Isp isp{"isp", pool, &memory};
  FragmentFrontend fragment_frontend{"fragment_frontend", pool, &memory};
  PdsEngine pds_engine{"pds_engine", pool, &memory};
  VertexPdsEngine vertex_pds_engine{"vertex_pds_engine", pool};
  PcoDecoder fragment_decoder{"fragment_pco_decoder", pool,
                              ShaderStage::kFragment};
  UscSlot fragment_slot{"fragment_usc_slot", pool, ShaderStage::kFragment};
  UscCluster fragment_cluster{"fragment_usc_cluster", pool,
                              ShaderStage::kFragment};
  TextureUnit texture_unit{"texture_unit", pool, &memory};
  Pbe pbe{"pbe", pool};
  PbeWriteBack pbe_write_back{"pbe_write_back", pool, &memory};
  JsonReporter reporter{"json_reporter", elaboration_options_, pool,
                        &sequence_completion, &job};
};

ModelSession::ModelSession(MemoryMode memory_mode, bool cache_bypass)
    : memory_mode_(memory_mode), cache_bypass_(cache_bypass),
      memory(memory_mode) {
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
  vertex_cluster.texture_request_output(vertex_cluster_to_texture_samples);
  vertex_cluster.texture_response_input(texture_samples_to_vertex_cluster);
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
  texture_unit.vertex_sample_input(vertex_cluster_to_texture_samples);
  texture_unit.vertex_sample_output(texture_samples_to_vertex_cluster);
  texture_unit.input(fragment_cluster_to_texture);
  texture_unit.output(texture_to_pbe);
  pbe.input(texture_to_pbe);
  pbe.output(pbe_to_pbe_write_back);
  pbe_write_back.input(pbe_to_pbe_write_back);
  pbe_write_back.completion(dram_to_reporter);
  reporter.input(dram_to_reporter);
}

int ModelSession::Run(const Options &options, ModelFramebuffer *framebuffer,
                      std::string *error) {
  const auto fail = [error](const std::string &message) {
    if (error)
      *error = message;
    return 1;
  };
  if (stopped_)
    return fail("SystemC model has been stopped");
  if (poisoned_)
    return fail("SystemC model is not usable after a failed flush");

  const std::uint64_t allocations_before = pool.allocations();
  const std::uint64_t releases_before = pool.releases();

  // Every process must reach its first `wait` before any work is queued.
  // Getting this the other way round is what cost the first job: the
  // submitter wrote into a FIFO nobody was reading yet.
  if (!started_) {
    sc_core::sc_start(sc_core::SC_ZERO_TIME);
    started_ = true;
  }

  job.Begin(options);
  job.start.notify(sc_core::SC_ZERO_TIME);
  while (!job.complete && sc_core::sc_pending_activity())
    sc_core::sc_start(sc_core::sc_time_to_pending_activity());
  job.running = false;

  const bool balanced =
      pool.allocations() - allocations_before ==
          pool.releases() - releases_before &&
      pool.bytes_in_flight() == 0;
  if (!job.complete) {
    poisoned_ = true;
    return fail(job.failed && !job.error.empty()
                    ? job.error
                    : "SystemC model went idle before the flush completed");
  }
  if (job.failed) {
    poisoned_ = true;
    return fail(job.error.empty() ? "SystemC model flush failed" : job.error);
  }
  if (!balanced) {
    poisoned_ = true;
    return fail("SystemC model flush leaked MemoryPool payloads");
  }
  if (framebuffer) {
    framebuffer->pixels = job.framebuffer;
    framebuffer->width = job.framebuffer_width;
    framebuffer->height = job.framebuffer_height;
  }
  return 0;
}

void ModelSession::Shutdown() {
  if (stopped_ || !started_)
    return;
  // After `sc_stop()` no `sc_start()` will run again, so this is teardown
  // only.  The processes are parked in `wait`, which is a clean place to stop.
  sc_core::sc_stop();
  stopped_ = true;
}

// One process, one elaboration -- so one session, created on the first flush
// and reused by every flush after it.
std::unique_ptr<ModelSession> g_session;

} // namespace
} // namespace pvrgpu::stub

int pvrgpu::stub::RunConfiguredModel(pvrgpu::stub::Options options,
                                     pvrgpu::stub::ModelFramebuffer *out) {
  const int announced = pvrgpu::stub::AnnounceModelConfiguration(options);
  if (announced != 0)
    return announced;

  if (!pvrgpu::stub::g_session) {
    pvrgpu::stub::g_session = std::make_unique<pvrgpu::stub::ModelSession>(
        options.memory_mode, options.cache_bypass);
  } else if (pvrgpu::stub::g_session->memory_mode() != options.memory_mode) {
    // The caches and the DRAM backing are elaborated, so the memory mode is
    // fixed for the life of the process.  Say so rather than silently running
    // the flush against the mode the first submission asked for.
    std::cerr << "SystemC model memory mode cannot change after elaboration\n";
    return 1;
  }

  std::string error;
  const int result =
      pvrgpu::stub::g_session->Run(options, out, &error);
  if (result != 0 && !error.empty())
    std::cerr << error << '\n';
  return result;
}

void pvrgpu::stub::ShutdownConfiguredModel() {
  if (!pvrgpu::stub::g_session)
    return;
  pvrgpu::stub::g_session->Shutdown();
}

int sc_main(int argc, char **argv) {
  pvrgpu::stub::Options options;
  if (!pvrgpu::stub::ParseOptions(argc, argv, &options))
    return 2;
  const int result = pvrgpu::stub::RunConfiguredModel(options);
  pvrgpu::stub::ShutdownConfiguredModel();
  return result;
}
