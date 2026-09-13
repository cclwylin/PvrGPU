#include "../model_stub/graphics_shader_buffers.h"
#include "../model_stub/shader_images.h"
#include "../model_stub/uniform_buffers.h"
#include "shader/geometry_iss.h"
#include "shader/tessellation_iss.h"
#include "shader/usc_shader_buffer_memory.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace pvrgpu::stub;

namespace {

[[noreturn]] void Fail(const char *message) {
  std::fprintf(stderr, "graphics-shader-buffer-test: %s\n", message);
  std::exit(EXIT_FAILURE);
}

void Check(bool condition, const char *message) {
  if (!condition)
    Fail(message);
}

template <typename F> void Reject(F action, const char *message) {
  bool rejected = false;
  try {
    action();
  } catch (const std::exception &) {
    rejected = true;
  }
  Check(rejected, message);
}

DriverCommand Fixture() {
  DriverCommand command;
  command.graphics_buffer_resources = {{UINT64_C(0xabc),
                                        std::vector<std::uint8_t>(16, 0x5a)}};
  const std::array<DriverPcoShaderStage, 5> stages{
      DriverPcoShaderStage::kVertex,
      DriverPcoShaderStage::kFragment,
      DriverPcoShaderStage::kGeometry,
      DriverPcoShaderStage::kTessellationControl,
      DriverPcoShaderStage::kTessellationEvaluation};
  const std::array<std::uint32_t, 5> system_dwords{0, 0, 4, 8, 4};
  for (std::size_t stage = 0; stage < stages.size(); ++stage) {
    command.graphics_storage[stage] = {system_dwords[stage], 1, 1, 1, 1};
    command.graphics_buffer_bindings.push_back(
        {stages[stage], 0, 0, 3, 0, 16});
  }
  command.vertex_shared = command.fragment_shared = {0, 0, 16, 0};
  command.geometry_shared.assign(8, 0);
  command.geometry_shared[6] = 16;
  command.tessellation.control_shared.assign(12, 0);
  command.tessellation.control_shared[10] = 16;
  command.tessellation.evaluation_shared.assign(8, 0);
  command.tessellation.evaluation_shared[6] = 16;
  command.vertex_pco_abi.shareds = command.fragment_pco_abi.shareds = 4;
  command.geometry_pco_abi.shareds = 8;
  command.tessellation.control_abi.shareds = 12;
  command.tessellation.evaluation_abi.shareds = 8;
  command.geometry_pco_abi.uniform_buffer_descriptor_start = 4;
  command.tessellation.control_abi.uniform_buffer_descriptor_start = 8;
  command.tessellation.evaluation_abi.uniform_buffer_descriptor_start = 4;
  command.vertex_pco_abi.push_constant_start = 4;
  command.fragment_pco_abi.push_constant_start = 4;
  command.geometry_pco_abi.push_constant_start = 8;
  command.tessellation.control_abi.push_constant_start = 12;
  command.tessellation.evaluation_abi.push_constant_start = 8;
  return command;
}

void TestDescriptorLayoutPermutations() {
  DriverGraphicsDescriptorLayout layout;
  DriverPcoStageAbi abi;
  DriverStorageBufferAbi storage{24, 1, 1, 1, 0};
  abi.uniform_buffer_descriptor_start = 20;
  abi.uniform_buffer_descriptor_count = 1;
  abi.push_constant_start = 28;
  abi.push_constant_count = 3;
  abi.shareds = 31;
  Check(ResolveDriverGraphicsDescriptorLayout(abi, 0, 1, 0, storage,
                                               &layout) &&
            layout.texture_end == 20 && layout.uniform_end == 24 &&
            layout.storage_end == 28,
        "texture/UBO/storage/push layout is canonical");

  storage.descriptor_start = 32;
  abi.push_constant_start = 36;
  abi.push_constant_count = 2;
  abi.shareds = 38;
  Check(ResolveDriverGraphicsDescriptorLayout(abi, 0, 1, 1, storage,
                                               &layout) &&
            layout.image_start == 24 && layout.image_end == 32 &&
            layout.storage_end == 36,
        "fragment image precedes storage and push constants");

  storage.descriptor_start = 28;
  abi.uniform_buffer_descriptor_start = 24;
  abi.push_constant_start = 32;
  abi.push_constant_count = 0;
  abi.shareds = 32;
  Check(ResolveDriverGraphicsDescriptorLayout(abi, 4, 1, 0, storage,
                                               &layout),
        "geometry system/texture/UBO/storage layout is canonical");

  abi = {};
  storage = {20, 1, 1, 1, 0};
  abi.push_constant_start = abi.shareds = 24;
  Check(ResolveDriverGraphicsDescriptorLayout(abi, 0, 1, 0, storage,
                                               &layout) &&
            layout.uniform_end == 20 && layout.storage_end == 24,
        "VS texture/zero-UBO/storage layout ignores the empty UBO start");
  storage.descriptor_start = 28;
  abi.push_constant_start = abi.shareds = 32;
  Check(ResolveDriverGraphicsDescriptorLayout(abi, 0, 1, 1, storage,
                                               &layout) &&
            layout.image_start == 20 && layout.storage_end == 32,
        "FS texture/zero-UBO/image/storage layout is canonical");

  abi.uniform_buffer_descriptor_start = 20;
  Check(!ResolveDriverGraphicsDescriptorLayout(abi, 0, 1, 1, storage),
        "VS/FS zero-UBO descriptor start remains canonically zero");

  abi = {};
  storage = {};
  abi.shareds = 20;
  Check(ResolveDriverGraphicsDescriptorLayout(abi, 0, 1, 0, storage,
                                               &layout) &&
            layout.texture_end == 20 && layout.storage_end == 20,
        "PCO {0,0} push range after a texture prefix is an empty window");
  abi.shareds = 60;
  Check(ResolveDriverGraphicsDescriptorLayout(abi, 0, 3, 0, storage),
        "PCO {0,0} push range after three textures is an empty window");
  abi.shareds = 24;
  Check(!ResolveDriverGraphicsDescriptorLayout(abi, 0, 1, 0, storage),
        "PCO {0,0} push range cannot leave shared words past the prefix");

  abi.uniform_buffer_descriptor_start = 24;
  storage = {28, 1, 1, 1, 0};
  abi.uniform_buffer_descriptor_count = 1;
  abi.push_constant_start = abi.shareds = 32;
  auto bad_storage = storage;
  bad_storage.descriptor_start = 24;
  Check(!ResolveDriverGraphicsDescriptorLayout(abi, 4, 1, 0, bad_storage),
        "storage cannot overlap a geometry UBO descriptor");
  bad_storage = storage;
  bad_storage.used_mask = 2;
  Check(!ResolveDriverGraphicsDescriptorLayout(abi, 4, 1, 0, bad_storage),
        "storage masks stay within the descriptor count");
}

void TestLegacyFragmentTextureLayout() {
  // glmark2 texture: draw_pco_triangles with one legacy fragment texture, its
  // 20-dword descriptor as the whole FS shared bank and no push constants.
  DriverCommand command;
  command.sampled_texture_count = 1;
  command.fragment_shared.assign(20, 0);
  command.fragment_pco_abi.temps = 8;
  command.fragment_pco_abi.coefficients = 16;
  command.fragment_pco_abi.shareds = 20;
  DriverGraphicsDescriptorLayout layout;
  Check(ResolveDriverGraphicsDescriptorLayout(command, 1, &layout) &&
            layout.texture_end == 20 && layout.storage_end == 20,
        "legacy fragment texture reserves its descriptor prefix");
  std::string error;
  Check(ValidateDriverUniformBuffers(command, &error), error.c_str());

  command.sampled_texture_count = 0;
  Check(!ResolveDriverGraphicsDescriptorLayout(command, 1),
        "untextured FS cannot carry a descriptor-sized shared bank");
}

void TestTransportContract() {
  std::string error;
  Check(ValidateDriverGraphicsShaderBuffers(DriverCommand{}, &error),
        "empty transport is valid");
  const auto fixture = Fixture();
  Check(ValidateDriverGraphicsShaderBuffers(fixture, &error), error.c_str());

  auto unbound = fixture;
  unbound.graphics_buffer_bindings.erase(
      unbound.graphics_buffer_bindings.begin());
  unbound.vertex_shared[2] = 0;
  Check(ValidateDriverGraphicsShaderBuffers(unbound, &error),
        "a used but unbound graphics SSBO remains a canonical null hole");

  auto size_only = fixture;
  size_only.graphics_storage[0].read_mask = 0;
  size_only.graphics_storage[0].write_mask = 0;
  size_only.graphics_buffer_bindings[0].access = 0;
  Check(ValidateDriverGraphicsShaderBuffers(size_only, &error),
        "size-query-only descriptor needs no memory permission");

  auto bad = fixture;
  bad.graphics_buffer_bindings[4].access = 1;
  Check(!ValidateDriverGraphicsShaderBuffers(bad, &error) &&
            error.find("permissions") != std::string::npos,
        "TES write permission is enforced");
  bad = fixture;
  bad.graphics_storage[1].write_mask = 0;
  Check(!ValidateDriverGraphicsShaderBuffers(bad, &error) &&
            error.find("exactly match") != std::string::npos,
        "binding cannot escalate fragment WRITE beyond the ABI mask");
  bad = fixture;
  bad.graphics_storage[0].read_mask = 0;
  bad.graphics_buffer_bindings[0].access = 3;
  Check(!ValidateDriverGraphicsShaderBuffers(bad, &error) &&
            error.find("exactly match") != std::string::npos,
        "binding cannot retain extra vertex READ permission");
  bad = fixture;
  bad.graphics_buffer_bindings[2].offset = 4;
  bad.graphics_buffer_bindings[2].bytes_size = 16;
  Check(!ValidateDriverGraphicsShaderBuffers(bad, &error) &&
            error.find("exceeds") != std::string::npos,
        "GS view is bounded by the whole resource");
  bad = fixture;
  bad.graphics_storage[3].used_mask = 0;
  Check(!ValidateDriverGraphicsShaderBuffers(bad, &error),
        "TCS binding outside the use mask is rejected");
  bad = fixture;
  bad.graphics_buffer_resources.push_back(bad.graphics_buffer_resources[0]);
  Check(!ValidateDriverGraphicsShaderBuffers(bad, &error) &&
            error.find("duplicated") != std::string::npos,
        "whole-resource alias token is unique");
  bad = fixture;
  bad.fragment_shared[2] = 12;
  Check(!ValidateDriverGraphicsShaderBuffers(bad, &error) &&
            error.find("canonical") != std::string::npos,
        "descriptor extent cannot diverge from its bound view");
}

void TestFiveStageAtomicCallbacks() {
  constexpr std::uint64_t address = UINT64_C(0x12345678000);
  const std::array<MemoryClient, 5> clients{
      MemoryClient::kVertexShader, MemoryClient::kFragmentShader,
      MemoryClient::kGeometryShader, MemoryClient::kTessellationControl,
      MemoryClient::kTessellationEvaluation};
  for (const auto mode : {MemoryMode::kDirect, MemoryMode::kBypass,
                          MemoryMode::kCache}) {
    std::array<std::uint32_t, 4> words{10, 20, 30, 40};
    GpuMemorySystem memory(mode);
    memory.HostWrite(address, words.data(), sizeof(words));
    const ShaderBufferResource resource{
        UINT64_C(0xabc), address, sizeof(words), 3, 0, {0, 1}};
    const ShaderBufferRange range{address, sizeof(words), 3, 0};
    std::uint32_t expected = words[0];
    for (std::size_t stage = 0; stage < clients.size(); ++stage) {
      UscShaderBufferMemory buffers(&memory, mode, {resource}, {range},
                                    clients[stage]);
      std::uint32_t visible = 0;
      UscShaderBufferMemory::Read(&buffers, address, 1, &visible);
      Check(visible == expected, "later stage observes earlier atomic writeback");
      const std::uint32_t operand = static_cast<std::uint32_t>(stage + 1);
      const auto old = UscShaderBufferMemory::Atomic32(
          &buffers, PcoOpcode::kAtomicAdd32, address, operand);
      Check(old == expected, "atomic callback returns the old value");
      expected += operand;
      Check(buffers.atomics() == 1, "atomic callback is counted once");
      const auto readback = buffers.Readback(resource);
      std::uint32_t value = 0;
      std::memcpy(&value, readback.data(), sizeof(value));
      Check(value == expected, "atomic writeback reaches the shared backing");
    }

    UscShaderBufferMemory read_only(
        &memory, mode, {resource}, {{address, sizeof(words), 1, 0}},
        MemoryClient::kVertexShader);
    Reject([&] {
      UscShaderBufferMemory::Atomic32(&read_only, PcoOpcode::kAtomicAdd32,
                                      address, 1);
    }, "read-only stage view blocks atomic writeback");
    Reject([&] {
      UscShaderBufferMemory::Atomic32(&read_only, PcoOpcode::kAtomicCompSwap,
                                      address, 1);
    }, "compare-swap cannot enter the one-operand graphics ABI");
    std::uint32_t robust_output = 0xccccccccU;
    UscShaderBufferMemory::Read(&read_only, address + sizeof(words), 1,
                                &robust_output);
    Check(robust_output == 0,
          "an SSBO address outside the stage view loads zero");

    UscShaderBufferMemory write_only(
        &memory, mode, {resource}, {{address, sizeof(words), 2, 0}},
        MemoryClient::kFragmentShader);
    Reject([&] {
      std::uint32_t output = 0;
      UscShaderBufferMemory::Read(&write_only, address, 1, &output);
    }, "write-only stage view blocks shader loads");
    const std::uint32_t replacement = UINT32_C(0x11223344);
    UscShaderBufferMemory::Write(&write_only, address + 4, 1, &replacement);

    // A native vec load/store is bounded per DWORD.  Preserve the prefix at
    // the end of the selected view, suppress its tail, and do not touch the
    // adjacent bytes in the backing BO.  OOB atomics return zero and perform
    // no RMW or traffic.
    std::array<std::uint32_t, 4> robust_words{1, 2, 3, 0xfeedfaceU};
    GpuMemorySystem robust_memory(mode);
    robust_memory.HostWrite(address, robust_words.data(), sizeof(robust_words));
    const ShaderBufferResource robust_resource{
        UINT64_C(0xdef), address, sizeof(robust_words), 3, 0, {0, 1}};
    UscShaderBufferMemory robust_buffers(
        &robust_memory, mode, {robust_resource}, {{address, 12, 3, 0}},
        MemoryClient::kFragmentShader);
    std::array<std::uint32_t, 2> vector{0xccccccccU, 0xccccccccU};
    UscShaderBufferMemory::Read(&robust_buffers, address + 8, 2,
                                vector.data());
    Check(vector[0] == 3 && vector[1] == 0,
          "partial SSBO load preserves its in-range DWORD and zeroes its tail");
    const std::array<std::uint32_t, 2> vector_store{
        UINT32_C(0x01020304), UINT32_C(0xaabbccdd)};
    UscShaderBufferMemory::Write(&robust_buffers, address + 8, 2,
                                 vector_store.data());
    Check(UscShaderBufferMemory::Atomic32(
              &robust_buffers, PcoOpcode::kAtomicAdd32, address + 12, 9) == 0 &&
              robust_buffers.atomics() == 0,
          "OOB SSBO atomic returns zero without executing an RMW");
    const auto robust_readback = robust_buffers.Readback(robust_resource);
    std::array<std::uint32_t, 4> robust_actual{};
    std::memcpy(robust_actual.data(), robust_readback.data(),
                sizeof(robust_actual));
    Check(robust_actual[0] == 1 && robust_actual[1] == 2 &&
              robust_actual[2] == vector_store[0] &&
              robust_actual[3] == robust_words[3],
          "partial/OOB SSBO operations touched bytes beyond the bound view");

    struct AtomicCase {
      PcoOpcode opcode;
      std::uint32_t old_value;
      std::uint32_t operand;
      std::uint32_t next_value;
    };
    const std::array<AtomicCase, 10> atomic_cases{{
        {PcoOpcode::kAtomicAdd32, 9, 4, 13},
        {PcoOpcode::kAtomicSub32, 9, 4, 5},
        {PcoOpcode::kAtomicExchange32, 9, 4, 4},
        {PcoOpcode::kAtomicUnsignedMin32, 9, 4, 4},
        {PcoOpcode::kAtomicSignedMin32, UINT32_C(0xfffffff9), 4,
         UINT32_C(0xfffffff9)},
        {PcoOpcode::kAtomicUnsignedMax32, 9, 14, 14},
        {PcoOpcode::kAtomicSignedMax32, UINT32_C(0xfffffff9), 4, 4},
        {PcoOpcode::kAtomicAnd32, 0xf3, 0x5a, 0x52},
        {PcoOpcode::kAtomicOr32, 0x31, 0x84, 0xb5},
        {PcoOpcode::kAtomicXor32, 0xf0, 0x5a, 0xaa},
    }};
    for (const auto &test : atomic_cases) {
      GpuMemorySystem atomic_memory(mode);
      atomic_memory.HostWrite(address, &test.old_value,
                              sizeof(test.old_value));
      UscShaderBufferMemory atomic_buffers(
          &atomic_memory, mode, {resource}, {range},
          MemoryClient::kFragmentShader);
      Check(UscShaderBufferMemory::Atomic32(
                &atomic_buffers, test.opcode, address, test.operand) ==
                test.old_value,
            "native atomic operation returns its old DWORD");
      std::uint32_t next = 0;
      UscShaderBufferMemory::Read(&atomic_buffers, address, 1, &next);
      Check(next == test.next_value,
            "native atomic operation writes its exact integer result");
    }
  }
}

std::vector<std::uint8_t> AtomicProgram(ShaderStage stage,
                                        bool vertex_registers) {
  std::vector<std::uint8_t> bytes;
  const auto append = [&](std::initializer_list<std::uint8_t> group) {
    bytes.insert(bytes.end(), group.begin(), group.end());
  };
  if (!vertex_registers) {
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
  } else {
    // Captured native I_ATOMIC grammar with both operands encoded through
    // I_ONE_LO's extended VTXIN bank: VI1..3 -> VI4.
    append({0x67, 0xa0, 0x00, 0xe5, 0x00, 0x03, 0x81, 0x04, 0x00,
            0x84, 0x04, 0x00, 0x00, 0xff});
  }
  append({0x02, 0x80, 0x6a, 0xff});
  if (stage == ShaderStage::kFragment ||
      stage == ShaderStage::kTessellationControl)
    append({0x04, 0x80, 0xee, 0x00, 0xf2, 0xff, 0xff, 0xff});
  else
    append({0x44, 0xa0, 0x80, 0x05, 0x00, 0x00, 0x00, 0xff});
  return bytes;
}

void CheckAtomicDecode(const PcoDecodedProgram &program,
                       bool vertex_registers) {
  const auto found = std::find_if(program.instructions.begin(),
                                  program.instructions.end(), [](const auto &i) {
    return i.opcode == PcoOpcode::kAtomicAdd32;
  });
  Check(found != program.instructions.end(), "raw stage program decodes atomic ADD32");
  Check(found->source.bank == (vertex_registers
                                   ? PcoRegisterBank::kVertexInput
                                   : PcoRegisterBank::kTemporary) &&
            found->source.index == 1 && found->source1.index == 2 &&
            found->source2.index == 3 && found->output_index ==
                (vertex_registers ? 4U : 2U) &&
            found->target == (vertex_registers
                                  ? PcoWriteTarget::kVertexInput
                                  : PcoWriteTarget::kTemporary),
        "atomic source triplet and response bank preserve the encoded grammar");
}

void TestFiveStageRawAtomicExecution() {
  constexpr std::uint64_t address = UINT64_C(0x12345678000);
  constexpr std::uint32_t initial = 41;
  constexpr std::uint32_t operand = 7;
  const ShaderBufferResource resource{
      UINT64_C(0xabc), address, 16, 3, 0, {0, 1}};
  const ShaderBufferRange range{address, 16, 3, 0};
  const auto service = [&](GpuMemorySystem &memory, MemoryClient client) {
    return UscShaderBufferMemory(&memory, MemoryMode::kDirect, {resource},
                                 {range}, client);
  };
  const auto reset = [&](GpuMemorySystem &memory) {
    const std::array<std::uint32_t, 4> words{initial, 0, 0, 0};
    memory.HostWrite(address, words.data(), sizeof(words));
  };
  const auto verify = [&](UscShaderBufferMemory &buffers) {
    std::uint32_t value = 0;
    UscShaderBufferMemory::Read(&buffers, address, 1, &value);
    Check(value == initial + operand && buffers.atomics() == 1,
          "raw stage ISS executes one bounded atomic writeback");
  };

  GpuMemorySystem memory(MemoryMode::kDirect);
  reset(memory);
  auto vertex_buffers = service(memory, MemoryClient::kVertexShader);
  const auto vertex = DecodePcoProgram(
      ShaderStage::kVertex, AtomicProgram(ShaderStage::kVertex, true));
  CheckAtomicDecode(vertex, true);
  PcoVertexExecutionContext vertex_context;
  vertex_context.memory_atomic32 = UscShaderBufferMemory::Atomic32;
  vertex_context.memory_user_data = &vertex_buffers;
  const auto vertex_result = ExecuteVertexPco(
      vertex.summary, vertex.instructions,
      {0, static_cast<std::uint32_t>(address),
       static_cast<std::uint32_t>(address >> 32U), operand, 0},
      vertex_context);
  Check(vertex_result.ended_task && vertex_result.emitted,
        "vertex raw atomic reaches native ENDTASK");
  verify(vertex_buffers);

  reset(memory);
  auto fragment_buffers = service(memory, MemoryClient::kFragmentShader);
  const auto fragment = DecodePcoProgram(
      ShaderStage::kFragment, AtomicProgram(ShaderStage::kFragment, false));
  CheckAtomicDecode(fragment, false);
  PcoFragmentExecutionContext fragment_context;
  fragment_context.memory_atomic32 = UscShaderBufferMemory::Atomic32;
  fragment_context.image_memory_user_data = &fragment_buffers;
  const auto fragment_result = ExecuteFragmentPco(
      fragment.summary, fragment.instructions, fragment_context);
  Check(!fragment_result.suspended && fragment_result.native_steps == 6,
        "fragment raw atomic completes through WDF");
  verify(fragment_buffers);
  Reject([&] {
    DecodePcoProgram(ShaderStage::kFragment,
                     AtomicProgram(ShaderStage::kVertex, true));
  }, "fragment decoder keeps atomic operands TEMP-only");

  reset(memory);
  auto geometry_buffers = service(memory, MemoryClient::kGeometryShader);
  const auto geometry = DecodeGeometryPcoProgram(
      AtomicProgram(ShaderStage::kGeometry, true));
  CheckAtomicDecode(geometry, true);
  DriverPcoStageAbi geometry_abi;
  geometry_abi.vertex_inputs = 5;
  geometry_abi.vertex_outputs = 4;
  geometry_abi.shareds = 4;
  geometry_abi.uniform_buffer_descriptor_start = 4;
  geometry_abi.push_constant_start = 4;
  ValidateGeometryProgram(geometry, geometry_abi);
  auto geometry_task = MakeGeometryTask(
      geometry_abi, std::vector<std::uint32_t>(4), 0, 0);
  geometry_task.inputs[1] = static_cast<std::uint32_t>(address);
  geometry_task.inputs[2] = static_cast<std::uint32_t>(address >> 32U);
  geometry_task.inputs[3] = operand;
  geometry_task.inputs_written |= UINT64_C(0xe);
  GeometryExecutionCallbacks geometry_callbacks;
  geometry_callbacks.user_data = &geometry_buffers;
  geometry_callbacks.atomic32 = UscShaderBufferMemory::Atomic32;
  geometry_callbacks.emit = [](void *, const std::uint32_t *, std::uint32_t,
                               std::uint64_t) {};
  geometry_callbacks.finish = [](void *) {};
  GeometryExecutionStats geometry_stats;
  while (!geometry_task.ended)
    StepGeometryTask(geometry, geometry_abi, geometry_task,
                     geometry_callbacks, geometry_stats);
  Check(geometry_task.inputs[4] == initial,
        "geometry WDF writes atomic old value to VI4");
  verify(geometry_buffers);

  for (const auto stage : {ShaderStage::kTessellationControl,
                           ShaderStage::kTessellationEvaluation}) {
    reset(memory);
    const auto client = stage == ShaderStage::kTessellationControl
                            ? MemoryClient::kTessellationControl
                            : MemoryClient::kTessellationEvaluation;
    auto buffers = service(memory, client);
    const auto program = DecodeTessellationPcoProgram(
        stage, AtomicProgram(stage, true));
    CheckAtomicDecode(program, true);
    DriverPcoStageAbi abi;
    abi.vertex_inputs = 5;
    abi.vertex_outputs = stage == ShaderStage::kTessellationControl ? 0 : 4;
    abi.shareds = stage == ShaderStage::kTessellationControl ? 8 : 4;
    abi.uniform_buffer_descriptor_start = abi.shareds;
    abi.push_constant_start = abi.shareds;
    ValidateTessellationProgram(program, abi);
    std::array<std::uint32_t, 3> coordinates{};
    auto task = stage == ShaderStage::kTessellationControl
        ? MakeTessellationControlTask(
              abi, std::vector<std::uint32_t>(abi.shareds), 0, 1, 1)
        : MakeTessellationEvaluationTask(
              abi, std::vector<std::uint32_t>(abi.shareds), 0, 1,
              &coordinates, 1);
    auto &lane = task.lanes[0];
    lane.inputs[1] = static_cast<std::uint32_t>(address);
    lane.inputs[2] = static_cast<std::uint32_t>(address >> 32U);
    lane.inputs[3] = operand;
    lane.inputs_written |= UINT64_C(0xe);
    TessellationMemoryCallbacks callbacks;
    callbacks.user_data = &buffers;
    callbacks.atomic32 = UscShaderBufferMemory::Atomic32;
    TessellationExecutionStats stats;
    while (!task.ended)
      StepTessellationTask(program, abi, task, callbacks, stats);
    Check(lane.inputs[4] == initial,
          "tessellation WDF writes atomic old value to VI4");
    verify(buffers);
  }
}

void TestTessellationVtxinLoadResponse() {
  constexpr std::uint64_t address = UINT64_C(0x12345678000);
  const std::array<std::uint32_t, 4> response{
      UINT32_C(0x11223344), UINT32_C(0x55667788),
      UINT32_C(0x99aabbcc), UINT32_C(0xddeeff00)};
  const auto program_bytes = [](ShaderStage stage) {
    std::vector<std::uint8_t> bytes{
        // Actual extended I_ONE_UP response form: SH0..1 -> VI0..3.
        0x67, 0xa0, 0x00, 0xf1, 0x10, 0x00, 0x80, 0x08, 0x00,
        0x80, 0x04, 0x00, 0x00, 0xff,
        0x02, 0x80, 0x6a, 0xff};
    const std::initializer_list<std::uint8_t> end =
        stage == ShaderStage::kTessellationControl
            ? std::initializer_list<std::uint8_t>{
                  0x04, 0x80, 0xee, 0x00, 0xf2, 0xff, 0xff, 0xff}
            : std::initializer_list<std::uint8_t>{
                  0x44, 0xa0, 0x80, 0x05, 0x00, 0x00, 0x00, 0xff};
    bytes.insert(bytes.end(), end.begin(), end.end());
    return bytes;
  };
  struct LoadRecord {
    std::uint64_t address = 0;
    const std::array<std::uint32_t, 4> *response = nullptr;
    unsigned reads = 0;
  } record{address, &response, 0};
  TessellationMemoryCallbacks callbacks;
  callbacks.user_data = &record;
  callbacks.read = [](void *opaque, std::uint64_t requested,
                      std::uint32_t count, std::uint32_t *destination) {
    auto &state = *static_cast<LoadRecord *>(opaque);
    Check(requested == state.address && count == state.response->size(),
          "tessellation VTXIN LD preserves address and burst width");
    std::copy(state.response->begin(), state.response->end(), destination);
    ++state.reads;
  };
  for (const auto stage : {ShaderStage::kTessellationControl,
                           ShaderStage::kTessellationEvaluation}) {
    record.reads = 0;
    const auto program = DecodeTessellationPcoProgram(stage,
                                                       program_bytes(stage));
    const auto &load = program.instructions.front();
    Check(load.opcode == PcoOpcode::kBufferLoad &&
              load.target == PcoWriteTarget::kVertexInput &&
              load.output_index == 0 && load.component_count == 4,
          "TCS/TES decoder retains captured VTXIN LD response");
    DriverPcoStageAbi abi;
    abi.vertex_inputs = 5;
    abi.vertex_outputs = stage == ShaderStage::kTessellationControl ? 0 : 4;
    abi.shareds = stage == ShaderStage::kTessellationControl ? 8 : 4;
    abi.uniform_buffer_descriptor_start = abi.shareds;
    abi.push_constant_start = abi.shareds;
    ValidateTessellationProgram(program, abi);
    std::vector<std::uint32_t> shared(abi.shareds);
    shared[0] = static_cast<std::uint32_t>(address);
    shared[1] = static_cast<std::uint32_t>(address >> 32U);
    std::array<std::uint32_t, 3> coordinates{};
    auto task = stage == ShaderStage::kTessellationControl
        ? MakeTessellationControlTask(abi, shared, 0, 1, 1)
        : MakeTessellationEvaluationTask(abi, shared, 0, 1,
                                         &coordinates, 1);
    TessellationExecutionStats stats;
    StepTessellationTask(program, abi, task, callbacks, stats);
    Check(task.lanes[0].pending_target == PcoWriteTarget::kVertexInput &&
              task.lanes[0].pending_count == 4 && record.reads == 1,
          "TCS/TES keeps VTXIN response pending until WDF");
    StepTessellationTask(program, abi, task, callbacks, stats);
    for (std::size_t word = 0; word < response.size(); ++word)
      Check(task.lanes[0].inputs[word] == response[word],
            "TCS/TES WDF commits every LD word to VTXIN");
    Check(!task.lanes[0].temporary_written.test(0),
          "TCS/TES VTXIN LD does not alias TEMP");
    while (!task.ended)
      StepTessellationTask(program, abi, task, callbacks, stats);
  }
  Reject([&] {
    DecodePcoProgram(ShaderStage::kFragment,
                     program_bytes(ShaderStage::kTessellationControl));
  }, "fragment decoder keeps LD response TEMP-only");
}

std::vector<PcoInstruction> ShortShadowPattern() {
  std::vector<PcoInstruction> instructions(5);
  auto &sample = instructions[0];
  sample.opcode = PcoOpcode::kTextureSample;
  sample.target = PcoWriteTarget::kTemporary;
  sample.output_index = 0;
  sample.source = {PcoRegisterBank::kTemporary, 0};
  sample.source1 = {PcoRegisterBank::kShared, 0};
  sample.source_count = 2;
  sample.component_count = 4;

  instructions[1].opcode = PcoOpcode::kWaitDataFence;

  auto &marker = instructions[2];
  marker.opcode = PcoOpcode::kMoveImmediate;
  marker.target = PcoWriteTarget::kTemporary;
  marker.output_index = 10;
  marker.immediate = 255;
  marker.source_count = 0;

  auto &test = instructions[3];
  test.opcode = PcoOpcode::kBooleanCompare;
  test.target = PcoWriteTarget::kTemporary;
  test.output_index = 11;
  test.source = {PcoRegisterBank::kShared, 12};
  test.source1 = {PcoRegisterBank::kTemporary, 10};
  test.source_count = 2;
  test.comparison_test_op = 4;   // TST.E
  test.comparison_test_type = 5; // TST.U32

  auto &select = instructions[4];
  select.opcode = PcoOpcode::kTestConditionalSelect;
  select.target = PcoWriteTarget::kTemporary;
  select.output_index = 12;
  select.source = {PcoRegisterBank::kTemporary, 11};
  select.source1 = {PcoRegisterBank::kTemporary, 20};
  select.source2 = {PcoRegisterBank::kTemporary, 0};
  select.source_count = 3;
  return instructions;
}

void TestTextureAnnotationRegisterSpansAndBarriers() {
  {
    auto instructions = ShortShadowPattern();
    AnnotatePcoTextureMetadataForTesting(instructions);
    Check(instructions[0].texture_shadow_compare == 1 &&
              instructions[0].texture_shadow_reference.bank ==
                  PcoRegisterBank::kTemporary &&
              instructions[0].texture_shadow_reference.index == 20,
          "canonical short shadow marker recovers Dref");
  }

  {
    auto instructions = ShortShadowPattern();
    PcoInstruction load;
    load.opcode = PcoOpcode::kBufferLoad;
    load.target = PcoWriteTarget::kTemporary;
    load.output_index = 19;
    load.component_count = 2;
    load.repeat_count = 1;
    instructions.insert(instructions.begin() + 2, load);
    AnnotatePcoTextureMetadataForTesting(instructions);
    Check(instructions[0].texture_shadow_compare == 0,
          "non-base burst LD clobber blocks shadow Dref recovery");
  }

  {
    auto instructions = ShortShadowPattern();
    PcoInstruction repeated;
    repeated.opcode = PcoOpcode::kFloatMultiply;
    repeated.target = PcoWriteTarget::kTemporary;
    repeated.output_index = 19;
    repeated.source = {PcoRegisterBank::kTemporary, 19};
    repeated.source1 = {PcoRegisterBank::kSpecial, 0};
    repeated.source_count = 2;
    repeated.repeat_count = 2;
    instructions.insert(instructions.begin() + 2, repeated);
    AnnotatePcoTextureMetadataForTesting(instructions);
    Check(instructions[0].texture_shadow_compare == 1,
          "in-place repeated ALU retains the tracked non-base lane");

    instructions = ShortShadowPattern();
    repeated.source = {PcoRegisterBank::kTemporary, 30};
    instructions.insert(instructions.begin() + 2, repeated);
    AnnotatePcoTextureMetadataForTesting(instructions);
    Check(instructions[0].texture_shadow_compare == 0,
          "repeated ALU without the tracked source lane is a clobber");
  }

  for (const auto barrier : {PcoOpcode::kBranch,
                             PcoOpcode::kBranchConditional,
                             PcoOpcode::kConditionalMask}) {
    auto instructions = ShortShadowPattern();
    PcoInstruction control;
    control.opcode = barrier;
    instructions.insert(instructions.begin() + 2, control);
    AnnotatePcoTextureMetadataForTesting(instructions);
    Check(instructions[0].texture_shadow_compare == 0,
          "shadow annotation cannot cross a branch or conditional mask");
  }

  {
    auto instructions = ShortShadowPattern();
    auto reference_copy = PcoInstruction{};
    reference_copy.opcode = PcoOpcode::kFloatAdd;
    reference_copy.target = PcoWriteTarget::kTemporary;
    reference_copy.output_index = 21;
    reference_copy.source = {PcoRegisterBank::kTemporary, 20};
    reference_copy.source1 = {PcoRegisterBank::kSpecial, 0};
    reference_copy.source_count = 2;
    instructions.insert(instructions.begin() + 2, reference_copy);
    instructions.insert(instructions.begin() + 3, 4, PcoInstruction{});
    instructions.back().source1 = {PcoRegisterBank::kTemporary, 21};
    AnnotatePcoTextureMetadataForTesting(instructions);
    Check(instructions[0].texture_shadow_compare == 1,
          "canonical long shadow marker recovers copied Dref");

    instructions = ShortShadowPattern();
    instructions.insert(instructions.begin() + 2, reference_copy);
    instructions.insert(instructions.begin() + 3, 4, PcoInstruction{});
    instructions[4].opcode = PcoOpcode::kConditionalMask;
    instructions.back().source1 = {PcoRegisterBank::kTemporary, 21};
    AnnotatePcoTextureMetadataForTesting(instructions);
    Check(instructions[0].texture_shadow_compare == 0,
          "long shadow fast path checks every middle control barrier");

    instructions = ShortShadowPattern();
    instructions.insert(instructions.begin() + 2, reference_copy);
    instructions.insert(instructions.begin() + 3, 4, PcoInstruction{});
    instructions[4].opcode = PcoOpcode::kTextureSample;
    instructions[4].target = PcoWriteTarget::kTemporary;
    instructions[4].output_index = 20;
    instructions[4].component_count = 4;
    instructions[4].repeat_count = 1;
    instructions.back().source1 = {PcoRegisterBank::kTemporary, 21};
    AnnotatePcoTextureMetadataForTesting(instructions);
    Check(instructions[0].texture_shadow_compare == 0,
          "four-DWORD SMP response clobber blocks copied Dref recovery");
  }

  {
    std::vector<PcoInstruction> instructions(2);
    instructions[0].opcode = PcoOpcode::kMoveImmediate;
    instructions[0].target = PcoWriteTarget::kTemporary;
    instructions[0].output_index = 2;
    instructions[0].immediate = 0;
    instructions[0].source_count = 0;
    instructions[1].opcode = PcoOpcode::kTextureSample;
    instructions[1].source = {PcoRegisterBank::kTemporary, 0};
    instructions[1].texture_dimension = 2;
    instructions[1].texture_address_offset = 1;
    instructions[1].texture_lod_bias = 1;
    AnnotatePcoTextureMetadataForTesting(instructions);
    Check(instructions[1].texture_lod_bias == 0,
          "canonical zero TAO padding is annotated");

    PcoInstruction load;
    load.opcode = PcoOpcode::kBufferLoad;
    load.target = PcoWriteTarget::kTemporary;
    load.output_index = 1;
    load.component_count = 2;
    instructions.insert(instructions.begin() + 1, load);
    instructions[2].texture_lod_bias = 1;
    AnnotatePcoTextureMetadataForTesting(instructions);
    Check(instructions[2].texture_lod_bias == 1,
          "non-base burst LD clobber blocks zero-TAO annotation");

    instructions.erase(instructions.begin() + 1);
    PcoInstruction sample;
    sample.opcode = PcoOpcode::kTextureSample;
    sample.target = PcoWriteTarget::kTemporary;
    sample.output_index = 1;
    sample.component_count = 4;
    sample.repeat_count = 1;
    instructions.insert(instructions.begin() + 1, sample);
    instructions[2].texture_lod_bias = 1;
    AnnotatePcoTextureMetadataForTesting(instructions);
    Check(instructions[2].texture_lod_bias == 1,
          "non-base SMP response clobber blocks zero-TAO annotation");

    instructions.erase(instructions.begin() + 1);
    PcoInstruction mask;
    mask.opcode = PcoOpcode::kConditionalMask;
    instructions.insert(instructions.begin() + 1, mask);
    instructions[2].texture_lod_bias = 1;
    AnnotatePcoTextureMetadataForTesting(instructions);
    Check(instructions[2].texture_lod_bias == 1,
          "conditional mask blocks zero-TAO annotation");
  }
}

void TestVertexAtomicBeforeTextureContinuation() {
  constexpr std::uint64_t address = UINT64_C(0x12345678000);
  std::array<std::uint32_t, 4> words{41, 0, 0, 0};
  GpuMemorySystem memory(MemoryMode::kDirect);
  memory.HostWrite(address, words.data(), sizeof(words));
  const ShaderBufferResource resource{
      UINT64_C(0xabc), address, sizeof(words), 3, 0, {0, 1}};
  const ShaderBufferRange range{address, sizeof(words), 3, 0};
  UscShaderBufferMemory buffers(&memory, MemoryMode::kDirect, {resource},
                                {range}, MemoryClient::kVertexShader);

  auto program = DecodePcoProgram(
      ShaderStage::kVertex, AtomicProgram(ShaderStage::kVertex, true));
  auto &instructions = program.instructions;
  const auto insert_at = instructions.begin() + 2;
  PcoInstruction coordinate_x;
  coordinate_x.opcode = PcoOpcode::kMoveImmediate;
  coordinate_x.target = PcoWriteTarget::kTemporary;
  coordinate_x.output_index = 0;
  coordinate_x.source_count = 0;
  PcoInstruction coordinate_y = coordinate_x;
  coordinate_y.output_index = 1;
  PcoInstruction sample;
  sample.opcode = PcoOpcode::kTextureSample;
  sample.target = PcoWriteTarget::kTemporary;
  sample.output_index = 8;
  sample.source = {PcoRegisterBank::kTemporary, 0};
  sample.source1 = {PcoRegisterBank::kShared, 0};
  sample.source2 = {PcoRegisterBank::kShared, 8};
  sample.source_count = 3;
  sample.component_count = 4;
  PcoInstruction wait;
  wait.opcode = PcoOpcode::kWaitDataFence;
  wait.target = PcoWriteTarget::kNone;
  wait.source_count = 0;
  instructions.insert(insert_at, {coordinate_x, coordinate_y, sample, wait});
  for (std::size_t index = 0; index < instructions.size(); ++index) {
    instructions[index].group_index = static_cast<std::uint16_t>(index);
    instructions[index].binary_offset =
        static_cast<std::uint32_t>(index * 8U + 1U);
    instructions[index].end_group = index + 1U == instructions.size();
  }
  program.summary.instruction_count = instructions.size();
  program.summary.group_count = instructions.size();
  program.summary.binary_size = instructions.size() * 8U;

  PcoVertexExecutionContext context;
  context.shared_count = 20;
  context.memory_atomic32 = UscShaderBufferMemory::Atomic32;
  context.memory_user_data = &buffers;
  const auto first = ExecuteVertexPco(
      program.summary, instructions,
      {0, static_cast<std::uint32_t>(address),
       static_cast<std::uint32_t>(address >> 32U), 7, 0},
      context);
  Check(first.suspended && first.continuation.vertex_inputs[4] == 41 &&
            buffers.atomics() == 1,
        "atomic/WDF state survives the following texture suspension");
  const std::array<std::uint32_t, 4> response{1, 2, 3, 4};
  const auto done = ResumeVertexPco(
      program.summary, instructions, first.continuation, response, nullptr,
      &buffers, UscShaderBufferMemory::Atomic32);
  Check(done.ended_task && done.emitted && buffers.atomics() == 1,
        "vertex continuation reconstructs atomic request/WDF exactly once");
}

} // namespace

int main() {
  TestDescriptorLayoutPermutations();
  TestLegacyFragmentTextureLayout();
  TestTransportContract();
  TestFiveStageAtomicCallbacks();
  TestFiveStageRawAtomicExecution();
  TestTessellationVtxinLoadResponse();
  TestTextureAnnotationRegisterSpansAndBarriers();
  TestVertexAtomicBeforeTextureContinuation();
  std::puts("graphics-shader-buffer-test: PASS");
  return EXIT_SUCCESS;
}
