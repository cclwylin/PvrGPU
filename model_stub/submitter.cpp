// Submitter（工作提交器）module 的實作。
// 產生目前通過 gate 的內建 GLBench fixtures：Fill.Solid 的四頂點
// triangle strip、TriangleSetup family 的 128×128 indexed lattice，以及
// AttributeFetchShader case 1/2/4/8 的精確 64×64 indexed lattice，以及
// VaryingsShader case 1/2/4/8 的 spacing=1/4、4×4 fullscreen indexed lattice。所有
// GLBench attributes 共用同一個 tightly-packed float2 VBO；VertexFetch 依
// binding 將真實 x/y 放入 VTXIN，GLES z/w default 由公開 PCO shader
// lowering 實作。需要 face
// culling 的 case 明確啟用 BACK/CCW；HalfCulled 使用 pinned GLBench srand(0)
// mixed-winding index stream。每個 DrawList 都使用 Mesa 產生的真實公開 PCO
// shader binary；
// 大型 vertex/index/pipeline payload 留在 MemoryPool，output FIFO 只傳
// PipelineTxn handle 與 frame/sequence metadata。
#include "submitter.h"

#include "common/functional_types.h"
#include "common/glbench_triangle_fixture.h"
#include "common/glbench_texture_fixture.h"
#include "common/pipeline_state.h"
#include "shader/pco_iss.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pvrgpu::stub {
namespace {

inline constexpr std::uint64_t kBuiltinVertexBufferGpuAddress =
    UINT64_C(0x20000000);
inline constexpr std::uint64_t kBuiltinTexcoordBufferGpuAddress =
    UINT64_C(0x21000000);

float FloatFromBits(std::uint32_t bits) {
  float value = 0.0F;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

bool DriverTriangleFragmentColorSupported(const DriverCommand &command) {
  static constexpr std::array<std::uint32_t, 4> kOpaqueRed = {
      UINT32_C(0x3f800000), 0U, 0U, UINT32_C(0x3f800000)};
  return command.fragment_color_bits == kOpaqueRed;
}

bool DriverIndexedQuadCommandSupported(const DriverCommand &command) {
  return command.draw_count != 0 && command.index_count == 6 &&
         command.unique_vertices == 4 && command.primitive_count == 2;
}

std::vector<float> DriverTriangleFloat2Vertices(
    const DriverCommand &command) {
  std::vector<float> vertices;
  vertices.reserve(6);
  for (const auto &vertex : command.vertex_bits) {
    vertices.push_back(FloatFromBits(vertex[0]));
    vertices.push_back(FloatFromBits(vertex[1]));
  }
  return vertices;
}

VertexBufferResource StoreFloat2VertexBuffer(MemoryPool &pool,
                                             const std::vector<float> &values,
                                             std::uint64_t gpu_address) {
  if (values.empty() || values.size() % 2 != 0 ||
      values.size() > std::numeric_limits<std::uint32_t>::max() /
                          sizeof(float)) {
    throw std::runtime_error("Submitter float2 VBO size is invalid");
  }
  VertexBufferResource resource;
  resource.data = StoreNewArray(pool, values);
  resource.gpu_address = gpu_address;
  resource.byte_size =
      static_cast<std::uint32_t>(values.size() * sizeof(float));
  return resource;
}

VertexAttributeBinding MakeFloat2Binding(std::uint32_t buffer_index,
                                         std::uint16_t destination_register,
                                         std::uint8_t destination_components) {
  VertexAttributeBinding binding;
  binding.buffer_index = buffer_index;
  binding.offset_bytes = 0;
  binding.stride_bytes = 2U * sizeof(float);
  binding.destination_register = destination_register;
  binding.component_type = VertexComponentType::kFloat32;
  binding.source_components = 2;
  binding.destination_components = destination_components;
  binding.normalized = 0;
  binding.integer = 0;
  binding.instance_divisor = 0;
  return binding;
}

} // namespace

Submitter::Submitter(sc_core::sc_module_name name, MemoryPool &pool,
                     const Options &options)
    : sc_module(name), pool_(pool), options_(options) {
  SC_THREAD(Run);
}

void Submitter::Run() {
  const bool driver_command = options_.driver_command.enabled;
  const bool driver_clear_command =
      driver_command && options_.driver_command.command == "clear_color";
  const bool driver_triangle_command =
      driver_command && options_.driver_command.command == "draw_triangle";
  const bool driver_indexed_quad_command =
      driver_command && options_.driver_command.command == "draw_indexed_quad";
  const bool driver_primitive_sequence_command =
      driver_command &&
      options_.driver_command.command == "draw_primitive_sequence";
  const FunctionalCase functional_case =
      FunctionalCaseFromName(options_.test_case);
  if (!IsRasterFunctionalCase(functional_case))
    throw std::runtime_error("Submitter received an unsupported GLBench case");
  const std::uint32_t command_framebuffer_width =
      driver_indexed_quad_command
          ? options_.driver_command.framebuffer_width
          : options_.driver_command.width;
  const std::uint32_t command_framebuffer_height =
      driver_indexed_quad_command
          ? options_.driver_command.framebuffer_height
          : options_.driver_command.height;
  if (driver_command &&
      (options_.frames != 1 ||
       options_.width != command_framebuffer_width ||
       options_.height != command_framebuffer_height ||
       (driver_clear_command &&
        functional_case != FunctionalCase::kDriverClearColor) ||
       (driver_triangle_command &&
        functional_case != FunctionalCase::kDriverTriangleSolid) ||
       (driver_indexed_quad_command &&
        functional_case != FunctionalCase::kDriverIndexedQuad) ||
       (driver_primitive_sequence_command &&
        functional_case != FunctionalCase::kDriverClearColor) ||
       (!driver_clear_command && !driver_triangle_command &&
        !driver_indexed_quad_command &&
        !driver_primitive_sequence_command))) {
    throw std::runtime_error(
        "Submitter driver command options do not match the one-frame command");
  }
  if (driver_triangle_command &&
      !DriverTriangleFragmentColorSupported(options_.driver_command)) {
    throw std::runtime_error(
        "Submitter driver triangle currently supports only opaque red fragments");
  }
  if (driver_indexed_quad_command &&
      !DriverIndexedQuadCommandSupported(options_.driver_command)) {
    throw std::runtime_error(
        "Submitter driver indexed quad command fields are unsupported");
  }

  for (unsigned frame = 1; frame <= options_.frames; ++frame) {
    PipelineState state;
    state.width = options_.width;
    state.height = options_.height;
    state.workload_class = WorkloadClass(options_.test_case);
    state.sequence = frame;
    state.functional_case = functional_case;
    state.stage = PipelineStage::kSubmitted;
    state.cache_bypass = options_.cache_bypass ? 1U : 0U;
    state.raster_state.sample_count = 1;
    const bool triangle_setup = IsTriangleSetupFamily(functional_case);
    const bool attribute_fetch = IsAttributeFetchFamily(functional_case);
    const bool varyings = IsVaryingsFamily(functional_case);
    const bool texture_case = IsTextureFamily(functional_case);
    const bool shader_varyings = UsesShaderVaryings(functional_case);
    const std::uint32_t varying_count = VaryingVectorCount(functional_case);
    const std::uint32_t attribute_count =
        functional_case == FunctionalCase::kAttributeFetchShaderEightAttribute
            ? 8U
            : functional_case ==
                  FunctionalCase::kAttributeFetchShaderFourAttribute
                  ? 4U
                  : functional_case ==
                        FunctionalCase::kAttributeFetchShaderTwoAttribute
                        ? 2U
                        : 1U;
    const bool indexed_triangle =
        IsIndexedTriangleRasterCase(functional_case);
    const bool driver_clear = driver_clear_command;
    const bool driver_triangle = driver_triangle_command;
    const bool driver_indexed_quad = driver_indexed_quad_command;
    const bool driver_primitive_sequence = driver_primitive_sequence_command;
    const bool driver_clear_like = driver_clear || driver_primitive_sequence;
    const bool depth_case =
        driver_clear_like ||
        functional_case == FunctionalCase::kFillSolidDepthNotEqual ||
        functional_case == FunctionalCase::kFillSolidDepthNever;
    state.raster_state.depth.test_enable = depth_case ? 1U : 0U;
    state.raster_state.depth.write_enable = depth_case ? 1U : 0U;
    state.raster_state.depth.compare_op =
        (driver_clear_like ||
         functional_case == FunctionalCase::kFillSolidDepthNever)
            ? DepthCompareOp::kNever
            : DepthCompareOp::kNotEqual;
    if (functional_case == FunctionalCase::kFillSolidBlended) {
      state.raster_state.blend.enable = 1;
      state.raster_state.blend.rgb_equation = BlendEquation::kAdd;
      state.raster_state.blend.alpha_equation = BlendEquation::kAdd;
      state.raster_state.blend.source_rgb_factor = BlendFactor::kSourceAlpha;
      state.raster_state.blend.destination_rgb_factor =
          BlendFactor::kOneMinusSourceAlpha;
      state.raster_state.blend.source_alpha_factor =
          BlendFactor::kSourceAlpha;
      state.raster_state.blend.destination_alpha_factor =
          BlendFactor::kOneMinusSourceAlpha;
    }
    if (indexed_triangle) {
      state.raster_state.clear_color[0] = 0.0F;
      state.raster_state.clear_color[1] = 1.0F;
      state.raster_state.clear_color[2] = 0.0F;
      state.raster_state.clear_color[3] = 1.0F;
    }
    if (RequiresBackCcwFaceCull(functional_case)) {
      state.raster_state.face_cull.enable = 1;
      state.raster_state.face_cull.mode = CullFaceMode::kBack;
      state.raster_state.face_cull.front_face =
          FrontFaceWinding::kCounterClockwise;
    }
    if (driver_clear_like || driver_triangle || driver_indexed_quad) {
      for (std::size_t component = 0; component < 4; ++component) {
        state.raster_state.clear_color[component] =
            FloatFromBits(options_.driver_command.clear_color_bits[component]);
      }
    }
    state.counters.frame = frame;
    state.counters.functional_frame = 1;

    std::vector<float> vertex_buffer;
    GlbenchFillTextureFixture texture_fixture;
    if (texture_case) {
      texture_fixture = MakeGlbenchFillTextureFixture(functional_case);
    }
    if (driver_triangle) {
      vertex_buffer = DriverTriangleFloat2Vertices(options_.driver_command);
      const std::vector<std::uint16_t> indices = {0, 1, 2};
      state.draw.topology = PrimitiveTopology::kTriangleList;
      state.draw.first_index = 0;
      state.draw.index_count = static_cast<std::uint32_t>(indices.size());
      state.draw.base_vertex = 0;
      state.draw.index_format = IndexFormat::kUint16;
      state.vertex_indices = StoreNewArray(pool_, indices);
    } else if (driver_indexed_quad) {
      vertex_buffer = {
          -1.0F, -1.0F,
          -1.0F, 1.0F,
          1.0F,  -1.0F,
          1.0F,  1.0F,
      };
      const std::vector<std::uint16_t> indices = {0, 2, 1, 1, 2, 3};
      state.draw.topology = PrimitiveTopology::kTriangleList;
      state.draw.first_index = 0;
      state.draw.index_count = static_cast<std::uint32_t>(indices.size());
      state.draw.base_vertex = 0;
      state.draw.index_format = IndexFormat::kUint16;
      state.vertex_indices = StoreNewArray(pool_, indices);
    } else if (indexed_triangle) {
      const GlbenchTriangleMeshShape &mesh =
          varyings ? kGlbenchVaryingsMesh
                   : attribute_fetch ? kGlbenchAttributeFetchMesh
                                     : kGlbenchTriangleSetupMesh;
      // Official varying cases call CreateLattice(size=1/4, shift=1)
      // independently of viewport size. Passing the 4×4 mesh dimensions as
      // the helper's coordinate denominator preserves exactly {-1,-.5,0,.5,1}.
      const std::uint32_t lattice_width = varyings ? mesh.width : state.width;
      const std::uint32_t lattice_height =
          varyings ? mesh.height : state.height;
      vertex_buffer = MakeGlbenchTriangleFloat2Vertices(
          lattice_width, lattice_height, mesh);
      const GlbenchTriangleWindingPattern winding =
          functional_case == FunctionalCase::kTriangleSetupHalfCulled
              ? GlbenchTriangleWindingPattern::kSrandZeroHalfCulled
              : GlbenchTriangleWindingPattern::kAllClockwise;
      const std::vector<std::uint16_t> indices =
          MakeGlbenchTriangleIndices(winding, mesh);
      state.draw.topology = PrimitiveTopology::kTriangleList;
      state.draw.first_index = 0;
      state.draw.index_count = static_cast<std::uint32_t>(indices.size());
      state.draw.base_vertex = 0;
      state.draw.index_format = IndexFormat::kUint16;
      state.vertex_indices = StoreNewArray(pool_, indices);
    } else if (texture_case) {
      vertex_buffer = texture_fixture.positions;
      state.draw.topology = PrimitiveTopology::kTriangleStrip;
      state.draw.first_vertex = 0;
      state.draw.vertex_count = 4;
    } else {
      vertex_buffer = {
          -1.0F, -1.0F,
          1.0F,  -1.0F,
          -1.0F, 1.0F,
          1.0F,  1.0F,
      };
      state.draw.topology = PrimitiveTopology::kTriangleStrip;
      state.draw.first_vertex = 0;
      state.draw.vertex_count = 4;
    }
    const VertexBufferResource vertex_resource =
        StoreFloat2VertexBuffer(pool_, vertex_buffer,
                                kBuiltinVertexBufferGpuAddress);
    std::vector<VertexBufferResource> vertex_resources{vertex_resource};
    if (texture_case) {
      vertex_resources.push_back(StoreFloat2VertexBuffer(
          pool_, texture_fixture.texture_coordinates,
          kBuiltinTexcoordBufferGpuAddress));
    }
    state.vertex_buffer_resources = StoreNewArray(pool_, vertex_resources);
    std::vector<VertexAttributeBinding> bindings;
    if (attribute_fetch && attribute_count > 1) {
      // GLBench binds every cN to the same VBO object. PVI supplies two
      // float32 components per input in consecutive VTXIN registers.
      bindings.reserve(attribute_count);
      for (std::uint32_t attribute = 0; attribute < attribute_count;
           ++attribute) {
        bindings.push_back(MakeFloat2Binding(0,
            static_cast<std::uint16_t>(attribute * 2U), 2));
      }
    } else if (texture_case) {
      bindings = {
          MakeFloat2Binding(0, 0, 2),
          MakeFloat2Binding(1, 2, 2),
      };
    } else {
      bindings = {MakeFloat2Binding(0, 0, 4)};
    }
    state.vertex_attribute_bindings = StoreNewArray(pool_, bindings);
    if (shader_varyings) {
      if (varying_count == 0)
        throw std::runtime_error("Submitter varying count is invalid");
      std::vector<ShaderVaryingBinding> linkages;
      linkages.reserve(varying_count);
      for (std::uint32_t varying = 0; varying < varying_count; ++varying) {
        ShaderVaryingBinding linkage;
        linkage.vertex_output_base = static_cast<std::uint16_t>(
            4U + varying * kVaryingVectorComponentCount);
        linkage.coefficient_set_base = static_cast<std::uint16_t>(
            1U + varying * kVaryingVectorComponentCount);
        linkage.w_coefficient_set = 0;
        linkage.component_count = texture_case
                                      ? 2U
                                      : kVaryingVectorComponentCount;
        linkage.interpolation = InterpolationMode::kSmooth;
        if (!IsExactVaryingBinding(functional_case, linkage, varying))
          throw std::runtime_error("Submitter varying linkage is invalid");
        linkages.push_back(linkage);
      }
      state.shader_varying_bindings =
          StoreNewArray(pool_, linkages);
    }
    if (texture_case) {
      texture_fixture.resource.data =
          StoreNewArray(pool_, texture_fixture.texture_bytes);
      state.texture_resources = StoreNewArray(
          pool_, std::vector<TextureResource>{texture_fixture.resource});
      state.sampler_states = StoreNewArray(
          pool_, std::vector<SamplerState>{texture_fixture.sampler});
      state.vertex_shared_registers = StoreNewArray(
          pool_, std::vector<ShaderSharedRegister>{
                     {texture_fixture.vertex_scale_bits}});
      state.fragment_shared_registers = StoreNewArray(
          pool_, std::vector<std::uint32_t>(
                     texture_fixture.fragment_shared.begin(),
                     texture_fixture.fragment_shared.end()));
    }
    state.vertex_code = StoreNewArray(
        pool_, texture_case ? FillTexNearestVertexPcoBinary()
                    : functional_case == FunctionalCase::kVaryingsShaderEight
                    ? VaryingsEightVertexPcoBinary()
                    : functional_case == FunctionalCase::kVaryingsShaderFour
                          ? VaryingsFourVertexPcoBinary()
                          : functional_case == FunctionalCase::kVaryingsShaderTwo
                          ? VaryingsTwoVertexPcoBinary()
                          : varyings ? VaryingsOneVertexPcoBinary()
                        : attribute_count == 8
                   ? AttributeFetchEightAttributeVertexPcoBinary()
                   : attribute_count == 4
                         ? AttributeFetchFourAttributeVertexPcoBinary()
                         : attribute_count == 2
                         ? AttributeFetchTwoAttributeVertexPcoBinary()
                         : attribute_fetch ? AttributeFetchVertexPcoBinary()
                                           : FillSolidVertexPcoBinary());
    state.fragment_code = StoreNewArray(
        pool_, texture_case ? FillTexNearestFragmentPcoBinary()
                    : functional_case == FunctionalCase::kVaryingsShaderEight
                    ? VaryingsEightFragmentPcoBinary()
                    : functional_case == FunctionalCase::kVaryingsShaderFour
                          ? VaryingsFourFragmentPcoBinary()
                          : functional_case == FunctionalCase::kVaryingsShaderTwo
                          ? VaryingsTwoFragmentPcoBinary()
                          : varyings ? VaryingsOneFragmentPcoBinary()
                        : attribute_fetch
                   ? AttributeFetchGrayFragmentPcoBinary()
                   : driver_indexed_quad ? FillSolidBlackFragmentPcoBinary()
                   : driver_triangle
                         ? FillSolidFragmentPcoBinary()
                   : functional_case == FunctionalCase::kTriangleSetupHalfCulled
                   ? TriangleSetupCyanFragmentPcoBinary()
                   : triangle_setup ? TriangleSetupOrangeFragmentPcoBinary()
                                    : FillSolidFragmentPcoBinary());
    state.drawlist_stats = StoreNewArray(pool_, std::vector<DrawListStats>{{}});

    const PoolHandle handle = pool_.Allocate(sizeof(PipelineState));
    if (output.num_free() == 0)
      fifo_stalls_++;
    state.counters.fifo_stall_events = fifo_stalls_;
    StorePipelineState(pool_, handle, state);
    output.write({handle, frame, frame});
  }
}

} // namespace pvrgpu::stub
