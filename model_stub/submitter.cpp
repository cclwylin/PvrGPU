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

std::uint32_t LoadLittleEndianU32(const std::uint8_t *bytes) {
  return static_cast<std::uint32_t>(bytes[0]) |
         (static_cast<std::uint32_t>(bytes[1]) << 8U) |
         (static_cast<std::uint32_t>(bytes[2]) << 16U) |
         (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

const std::vector<std::uint8_t> &MesaPocFragmentBinary(
    const MesaPocCommand &command) {
  // These public PCO programs materialize constants directly. Recheck the
  // real Mesa bytes at the selection point so an unrecognized uniform cannot
  // silently receive a fixture shader.
  static constexpr std::array<std::uint8_t, 16> kRed = {
      0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
  };
  static constexpr std::array<std::uint8_t, 16> kOrange = {
      0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x3f,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
  };
  static constexpr std::array<std::uint8_t, 16> kCyan = {
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3f,
      0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x80, 0x3f,
  };
  if (command.fragment_constants.empty() &&
      command.test_case.rfind("attribute_fetch_shader", 0) == 0) {
    return AttributeFetchGrayFragmentPcoBinary();
  }
  if (command.fragment_constants.empty() &&
      command.test_case == "varyings_shader_1") {
    return VaryingsOneFragmentPcoBinary();
  }
  if (command.fragment_constants.empty() &&
      command.test_case == "varyings_shader_2") {
    return VaryingsTwoFragmentPcoBinary();
  }
  if (command.fragment_constants.empty() &&
      command.test_case == "varyings_shader_4") {
    return VaryingsFourFragmentPcoBinary();
  }
  if (command.fragment_constants.empty() &&
      command.test_case == "varyings_shader_8") {
    return VaryingsEightFragmentPcoBinary();
  }
  if (command.fragment_constants.empty() &&
      command.test_case.rfind("fill_tex", 0) == 0) {
    return FillTexNearestFragmentPcoBinary();
  }
  if (command.fragment_constants.size() == kRed.size() &&
      std::equal(command.fragment_constants.begin(),
                 command.fragment_constants.end(), kRed.begin())) {
    return FillSolidFragmentPcoBinary();
  }
  if (command.fragment_constants.size() == kOrange.size() &&
      std::equal(command.fragment_constants.begin(),
                 command.fragment_constants.end(), kOrange.begin())) {
    return TriangleSetupOrangeFragmentPcoBinary();
  }
  if (command.fragment_constants.size() == kCyan.size() &&
      std::equal(command.fragment_constants.begin(),
                 command.fragment_constants.end(), kCyan.begin())) {
    return TriangleSetupCyanFragmentPcoBinary();
  }
  throw std::runtime_error(
      "Mesa POC fragment constants do not match an enabled public PCO");
}

const std::vector<std::uint8_t> &MesaPocVertexBinary(
    const MesaPocCommand &command) {
  if (command.test_case.rfind("attribute_fetch_shader", 0) == 0) {
    switch (command.vertex_elements) {
    case 1:
      return AttributeFetchVertexPcoBinary();
    case 2:
      return AttributeFetchTwoAttributeVertexPcoBinary();
    case 4:
      return AttributeFetchFourAttributeVertexPcoBinary();
    case 8:
      return AttributeFetchEightAttributeVertexPcoBinary();
    default:
      throw std::runtime_error(
          "Mesa POC attribute count has no enabled public PCO");
    }
  }
  if (command.test_case == "varyings_shader_1")
    return VaryingsOneVertexPcoBinary();
  if (command.test_case == "varyings_shader_2")
    return VaryingsTwoVertexPcoBinary();
  if (command.test_case == "varyings_shader_4")
    return VaryingsFourVertexPcoBinary();
  if (command.test_case == "varyings_shader_8")
    return VaryingsEightVertexPcoBinary();
  if (command.test_case.rfind("fill_tex", 0) == 0)
    return FillTexNearestVertexPcoBinary();
  return FillSolidVertexPcoBinary();
}

BlendFactor MesaPocBlendFactor(const std::string &factor) {
  if (factor == "PIPE_BLENDFACTOR_SRC_ALPHA")
    return BlendFactor::kSourceAlpha;
  if (factor == "PIPE_BLENDFACTOR_INV_SRC_ALPHA")
    return BlendFactor::kOneMinusSourceAlpha;
  throw std::runtime_error("Mesa POC blend factor is unsupported: " + factor);
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
  const bool mesa_poc = options_.mesa_command.enabled;
  const FunctionalCase functional_case =
      FunctionalCaseFromName(options_.test_case);
  if (!IsRasterFunctionalCase(functional_case))
    throw std::runtime_error("Submitter received an unsupported GLBench case");
  const bool mesa_poc_case_supported =
      functional_case == FunctionalCase::kFillSolid ||
      functional_case == FunctionalCase::kFillSolidDepthNever ||
      functional_case == FunctionalCase::kFillSolidDepthNotEqual ||
      functional_case == FunctionalCase::kFillSolidBlended ||
      functional_case == FunctionalCase::kTriangleSetup ||
      functional_case == FunctionalCase::kTriangleSetupAllCulled ||
      functional_case == FunctionalCase::kTriangleSetupHalfCulled ||
      functional_case == FunctionalCase::kAttributeFetchShader ||
      functional_case == FunctionalCase::kAttributeFetchShaderTwoAttribute ||
      functional_case == FunctionalCase::kAttributeFetchShaderFourAttribute ||
      functional_case == FunctionalCase::kAttributeFetchShaderEightAttribute ||
      functional_case == FunctionalCase::kVaryingsShaderOne ||
      functional_case == FunctionalCase::kVaryingsShaderTwo ||
      functional_case == FunctionalCase::kVaryingsShaderFour ||
      functional_case == FunctionalCase::kVaryingsShaderEight ||
      functional_case == FunctionalCase::kFillTexNearest ||
      functional_case == FunctionalCase::kFillTexBilinear ||
      functional_case == FunctionalCase::kFillTexTrilinearLinear01 ||
      functional_case == FunctionalCase::kFillTexTrilinearLinear04 ||
      functional_case == FunctionalCase::kFillTexTrilinearLinear05;
  if (mesa_poc &&
      (!mesa_poc_case_supported ||
       options_.frames != 1 ||
       options_.test_case != options_.mesa_command.test_case ||
       options_.width != options_.mesa_command.width ||
       options_.height != options_.mesa_command.height)) {
    throw std::runtime_error(
        "Submitter Mesa POC options do not match the one-frame command");
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
    const bool depth_case =
        functional_case == FunctionalCase::kFillSolidDepthNotEqual ||
        functional_case == FunctionalCase::kFillSolidDepthNever;
    state.raster_state.depth.test_enable = depth_case ? 1U : 0U;
    state.raster_state.depth.write_enable = depth_case ? 1U : 0U;
    state.raster_state.depth.compare_op =
        functional_case == FunctionalCase::kFillSolidDepthNever
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
    if (mesa_poc) {
      const MesaPocCommand &command = options_.mesa_command;
      state.raster_state.depth.test_enable =
          command.depth_enabled ? 1U : 0U;
      state.raster_state.depth.write_enable = command.depth_write ? 1U : 0U;
      state.raster_state.depth.compare_op =
          static_cast<DepthCompareOp>(command.depth_func);
      state.raster_state.blend.enable = command.blend_enabled ? 1U : 0U;
      if (command.blend_enabled) {
        if (command.blend_rgb_func != "PIPE_BLEND_ADD" ||
            command.blend_alpha_func != "PIPE_BLEND_ADD") {
          throw std::runtime_error("Mesa POC blend equation is unsupported");
        }
        state.raster_state.blend.rgb_equation = BlendEquation::kAdd;
        state.raster_state.blend.alpha_equation = BlendEquation::kAdd;
        state.raster_state.blend.source_rgb_factor =
            MesaPocBlendFactor(command.blend_rgb_src_factor);
        state.raster_state.blend.destination_rgb_factor =
            MesaPocBlendFactor(command.blend_rgb_dst_factor);
        state.raster_state.blend.source_alpha_factor =
            MesaPocBlendFactor(command.blend_alpha_src_factor);
        state.raster_state.blend.destination_alpha_factor =
            MesaPocBlendFactor(command.blend_alpha_dst_factor);
      }
      state.raster_state.face_cull.enable = command.cull_face != 0 ? 1U : 0U;
      if (command.cull_face == 1)
        state.raster_state.face_cull.mode = CullFaceMode::kFront;
      else if (command.cull_face == 2)
        state.raster_state.face_cull.mode = CullFaceMode::kBack;
      else if (command.cull_face == 3)
        state.raster_state.face_cull.mode = CullFaceMode::kFrontAndBack;
      else if (command.cull_face != 0)
        throw std::runtime_error("Mesa POC cull-face value is unsupported");
      // Gallium's lower-left framebuffer convention reverses the winding bit
      // relative to the GLES/API-space convention consumed by this model.
      state.raster_state.face_cull.front_face =
          command.front_ccw ? FrontFaceWinding::kClockwise
                            : FrontFaceWinding::kCounterClockwise;
      for (std::size_t component = 0; component < 4; ++component) {
        state.raster_state.clear_color[component] =
            FloatFromBits(command.clear_color_bits[component]);
      }
    }
    state.counters.frame = frame;
    state.counters.functional_frame = 1;

    std::vector<float> vertex_buffer;
    GlbenchFillTextureFixture texture_fixture;
    if (texture_case) {
      texture_fixture = MakeGlbenchFillTextureFixture(functional_case);
      if (mesa_poc) {
        const MesaPocCommand &command = options_.mesa_command;
        if (command.vertex_float2_1.empty() ||
            command.vertex_constants.size() != sizeof(std::uint32_t) ||
            command.texture_bytes.size() != texture_fixture.texture_bytes.size()) {
          throw std::runtime_error(
              "Submitter Mesa texture command payload is incomplete");
        }
        texture_fixture.texture_coordinates = command.vertex_float2_1;
        texture_fixture.texture_bytes = command.texture_bytes;
        texture_fixture.vertex_scale_bits =
            LoadLittleEndianU32(command.vertex_constants.data());
      }
    }
    if (mesa_poc) {
      vertex_buffer = options_.mesa_command.vertex_float2;
      if (options_.mesa_command.indexed) {
        state.draw.topology = PrimitiveTopology::kTriangleList;
        state.draw.first_index = 0;
        state.draw.index_count = options_.mesa_command.draw_count;
        state.draw.base_vertex = 0;
        state.draw.index_format = IndexFormat::kUint16;
        state.vertex_indices =
            StoreNewArray(pool_, options_.mesa_command.indices);
      } else {
        state.draw.topology = PrimitiveTopology::kTriangleStrip;
        state.draw.first_vertex = 0;
        state.draw.vertex_count = options_.mesa_command.draw_count;
      }
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
    if (mesa_poc && attribute_fetch) {
      bindings.reserve(options_.mesa_command.vertex_elements);
      for (std::uint32_t attribute = 0;
           attribute < options_.mesa_command.vertex_elements; ++attribute) {
        bindings.push_back(MakeFloat2Binding(
            0, static_cast<std::uint16_t>(attribute * 2U), 2));
      }
    } else if (attribute_fetch && attribute_count > 1) {
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
    if (mesa_poc) {
      for (VertexAttributeBinding &binding : bindings) {
        binding.offset_bytes = options_.mesa_command.vertex_offset;
        binding.stride_bytes = options_.mesa_command.vertex_stride;
      }
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
        pool_, mesa_poc ? MesaPocVertexBinary(options_.mesa_command)
                    : texture_case ? FillTexNearestVertexPcoBinary()
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
        pool_, mesa_poc ? MesaPocFragmentBinary(options_.mesa_command)
                    : texture_case ? FillTexNearestFragmentPcoBinary()
                    : functional_case == FunctionalCase::kVaryingsShaderEight
                    ? VaryingsEightFragmentPcoBinary()
                    : functional_case == FunctionalCase::kVaryingsShaderFour
                          ? VaryingsFourFragmentPcoBinary()
                          : functional_case == FunctionalCase::kVaryingsShaderTwo
                          ? VaryingsTwoFragmentPcoBinary()
                          : varyings ? VaryingsOneFragmentPcoBinary()
                        : attribute_fetch
                   ? AttributeFetchGrayFragmentPcoBinary()
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
