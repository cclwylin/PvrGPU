// Decoded-instruction unit fixtures for TCS/TES texture request and WDF state.
// These hand-constructed instructions are not claimed to be compiler binaries.
// The callback is a transport oracle, not a TextureUnit or prepared image.
#include "shader/tessellation_iss.h"
#include "pco_tessellation_texture_fixtures.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace pvrgpu::stub;
namespace {
unsigned checks = 0;
void Check(bool ok, const char *message) {
  ++checks;
  if (!ok) throw std::runtime_error(message);
}
template<class F> void Reject(F fn, const char *expected) {
  try { fn(); }
  catch (const std::runtime_error &error) {
    Check(std::string(error.what()).find(expected) != std::string::npos,
          "rejection must name the intended tessellation guard");
    return;
  }
  throw std::runtime_error("missing expected tessellation rejection");
}
std::uint32_t Bits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}
float Float(std::uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}
unsigned System(ShaderStage stage) { return stage == ShaderStage::kTessellationControl ? 8 : 4; }
DriverPcoStageAbi Abi(ShaderStage stage, unsigned textures = 1) {
  DriverPcoStageAbi abi;
  abi.temps = 8;
  abi.vertex_inputs = stage == ShaderStage::kTessellationControl ? 3 : 5;
  abi.vertex_outputs = stage == ShaderStage::kTessellationControl ? 0 : 4;
  abi.uniform_buffer_descriptor_start = System(stage) + textures * 20;
  abi.uniform_buffer_descriptor_count = 1;
  abi.push_constant_start = abi.uniform_buffer_descriptor_start + 4;
  abi.push_constant_count = 4;
  abi.shareds = abi.push_constant_start + 4;
  return abi;
}
PcoDecodedProgram Program(ShaderStage stage, bool lod, unsigned descriptor = 0) {
  PcoDecodedProgram program;
  program.summary.stage = stage;
  program.summary.ends_task = 1;
  PcoInstruction sample;
  sample.opcode = PcoOpcode::kTextureSample;
  sample.target = PcoWriteTarget::kTemporary;
  sample.source = {PcoRegisterBank::kTemporary, 0};
  sample.source1 = {PcoRegisterBank::kShared, static_cast<std::uint16_t>(System(stage) + descriptor * 20)};
  sample.source2 = {PcoRegisterBank::kShared, static_cast<std::uint16_t>(sample.source1.index + 8)};
  sample.source_count = 3;
  sample.repeat_count = 1;
  sample.output_index = 4;
  sample.component_count = 4;
  sample.texture_dimension = 2;
  sample.texture_fcnorm = 1;
  sample.texture_lod_replace = lod;
  PcoInstruction wdf;
  wdf.opcode = PcoOpcode::kWaitDataFence;
  wdf.target = PcoWriteTarget::kNone;
  wdf.source_count = 0;
  wdf.repeat_count = 1;
  PcoInstruction end;
  end.opcode = stage == ShaderStage::kTessellationControl ? PcoOpcode::kNop : PcoOpcode::kUvsWriteEmitEndTask;
  end.target = stage == ShaderStage::kTessellationControl ? PcoWriteTarget::kNone : PcoWriteTarget::kVertexOutput;
  end.source = {PcoRegisterBank::kTemporary, 4};
  end.source_count = stage == ShaderStage::kTessellationControl ? 0 : 1;
  end.repeat_count = stage == ShaderStage::kTessellationControl ? 1 : 4;
  end.end_group = 1;
  program.instructions = {sample, wdf, end};
  program.summary.instruction_count = program.summary.group_count = 3;
  if (stage == ShaderStage::kTessellationEvaluation) program.summary.vertex_output_mask = 15;
  return program;
}
TessellationTaskState Task(ShaderStage stage, const DriverPcoStageAbi &abi, unsigned lanes) {
  std::vector<std::uint32_t> shared(abi.shareds);
  for (unsigned i = System(stage); i < abi.shareds; ++i) shared[i] = 0x4d000000 + i;
  std::array<std::array<std::uint32_t, 3>, 32> coordinates{};
  auto task = stage == ShaderStage::kTessellationControl ?
      MakeTessellationControlTask(abi, shared, 17, 3, lanes) :
      MakeTessellationEvaluationTask(abi, shared, 17, 3, coordinates.data(), lanes);
  for (unsigned i = 0; i < lanes; ++i) {
    task.lanes[i].temporaries[0] = Bits(float(i + 1) / 32);
    task.lanes[i].temporaries[1] = Bits(-float(i + 1) / 16);
    task.lanes[i].temporaries[2] = Bits(float(i % 5) - 0.5f);
    for (unsigned c = 0; c < 3; ++c) task.lanes[i].temporary_written.set(c);
    for (unsigned c = 4; c < 8; ++c) task.lanes[i].temporaries[c] = 0xc0decafe;
  }
  return task;
}
struct Samples {
  std::vector<PcoTextureRequest> requests;
  static std::array<std::uint32_t, 4> Response(const PcoTextureRequest &r) {
    return {r.coordinates[0], r.coordinates[1], r.explicit_lod_present ? r.explicit_lod : Bits(7),
            r.texture_state[0] ^ r.sampler_state[0]};
  }
  static void Sample(void *opaque, const PcoTextureRequest &r, std::uint32_t *words) {
    auto &self = *static_cast<Samples *>(opaque);
    self.requests.push_back(r);
    const auto response = Response(r);
    std::copy(response.begin(), response.end(), words);
  }
};
void RequestsAndFence() {
  for (auto stage : {ShaderStage::kTessellationControl, ShaderStage::kTessellationEvaluation})
    for (bool lod : {false, true}) for (unsigned textures : {1U, 8U})
      for (unsigned count : {1U, 7U, 32U}) {
        auto abi = Abi(stage, textures);
        auto program = Program(stage, lod, textures - 1);
        ValidateTessellationProgram(program, abi);
        auto task = Task(stage, abi, count);
        const auto shared_before = task.shared;
        Samples samples;
        TessellationExecutionStats stats;
        TessellationMemoryCallbacks callback;
        callback.user_data = &samples;
        callback.sample = Samples::Sample;
        StepTessellationTask(program, abi, task, callback, stats);
        Check(samples.requests.size() == count && stats.texture_instructions == count,
              "one synchronous sample per real active lane");
        Check(stats.memory_instructions == 0 && stats.load_instructions == 0 && stats.store_instructions == 0,
              "SMP has separate texture counters, consistent with GS");
        for (unsigned lane = 0; lane < count; ++lane) {
          const auto &r = samples.requests[lane];
          Check(r.coordinates[0] == task.lanes[lane].temporaries[0] &&
                    r.coordinates[1] == task.lanes[lane].temporaries[1] && r.coordinates[2] == 0,
                "raw per-lane UV bits survive callback transport");
          Check(r.explicit_lod_present == lod && r.explicit_lod == (lod ? task.lanes[lane].temporaries[2] : 0),
                "dynamic positive/negative explicit LOD bits are not replaced");
          Check(r.coordinate_count == 2 && r.dimension == 2 && r.component_count == 4 && r.normalized == 1 &&
                    r.fcnorm == 1 && r.descriptor_set == textures - 1 && r.binding == 0 && r.data_request == 0 &&
                    !r.gather && !r.lod_bias_present && !r.sample_index_present && !r.texture_address_lo && !r.texture_address_hi &&
                    r.spatial_offsets == std::array<std::int32_t, 3>{}, "canonical bounded texture request");
          const auto base = System(stage) + (textures - 1) * 20;
          for (unsigned c = 0; c < 4; ++c) {
            Check(r.texture_state[c] == task.shared[base + c] && r.sampler_state[c] == task.shared[base + 8 + c],
                  "descriptor namespace follows system prefix and never UBO/CB0");
            Check(task.lanes[lane].temporaries[4 + c] == 0xc0decafe && !task.lanes[lane].temporary_written.test(4 + c),
                  "SMP response is unavailable before WDF");
          }
          Check(task.lanes[lane].pending_operation == 3 && task.lanes[lane].pending_count == 4,
                "SMP has explicit pending response kind");
        }
        StepTessellationTask(program, abi, task, callback, stats);
        for (unsigned lane = 0; lane < count; ++lane) {
          const auto words = Samples::Response(samples.requests[lane]);
          for (unsigned c = 0; c < 4; ++c)
            Check(task.lanes[lane].temporaries[4 + c] == words[c] && task.lanes[lane].temporary_written.test(4 + c),
                  "matching WDF publishes exact four response words");
          Check(!task.lanes[lane].pending_operation && !task.lanes[lane].pending_count, "WDF resolves pending SMP once");
        }
        while (!task.ended) StepTessellationTask(program, abi, task, callback, stats);
        Check(samples.requests.size() == count && task.shared == shared_before, "no repeated SMP or shared/UBO/push mutation");
        if (stage == ShaderStage::kTessellationEvaluation)
          for (unsigned lane = 0; lane < count; ++lane) for (unsigned c = 0; c < 4; ++c)
            Check(task.lanes[lane].outputs[c] == Samples::Response(samples.requests[lane])[c], "TES exports native callback results after WDF");
      }
}
void Predication() {
  for (auto stage : {ShaderStage::kTessellationControl, ShaderStage::kTessellationEvaluation})
    for (unsigned condition : {0U, 1U, 2U, 3U}) {
      auto abi = Abi(stage);
      auto program = Program(stage, true);
      program.instructions[0].exec_cnd = condition;
      ValidateTessellationProgram(program, abi);
      auto task = Task(stage, abi, 7);
      std::array<bool, 7> selected{};
      unsigned expected = 0;
      for (unsigned lane = 0; lane < 7; ++lane) {
        task.lanes[lane].execution_predicate = lane % 3 != 0;
        task.lanes[lane].predicate = lane % 2 == 0;
        selected[lane] = condition == 2 || (task.lanes[lane].execution_predicate &&
            (condition == 0 || (condition == 1 ? task.lanes[lane].predicate != 0 : !task.lanes[lane].predicate)));
        expected += selected[lane];
      }
      Samples samples;
      TessellationMemoryCallbacks cb{&samples, nullptr, nullptr, Samples::Sample};
      TessellationExecutionStats stats;
      StepTessellationTask(program, abi, task, cb, stats);
      Check(samples.requests.size() == expected && stats.texture_instructions == expected, "inactive/predicate-false lanes never sample");
      StepTessellationTask(program, abi, task, cb, stats);
      for (unsigned lane = 0; lane < 7; ++lane) for (unsigned c = 4; c < 8; ++c)
        Check(task.lanes[lane].temporary_written.test(c) == selected[lane], "WDF writes only lanes that really issued SMP");
    }
}
void HighTemporaryBoundary() {
  for (auto stage : {ShaderStage::kTessellationControl, ShaderStage::kTessellationEvaluation}) {
    auto abi = Abi(stage);
    abi.temps = kPcoTemporaryCount;
    auto program = Program(stage, true);
    program.instructions[0].source.index = kPcoTemporaryCount - 3;
    program.instructions[0].output_index = kPcoTemporaryCount - 4;
    if (stage == ShaderStage::kTessellationEvaluation) program.instructions[2].source.index = kPcoTemporaryCount - 4;
    ValidateTessellationProgram(program, abi);
    auto task = Task(stage, abi, 1);
    for (unsigned c = 0; c < 3; ++c) {
      task.lanes[0].temporaries[kPcoTemporaryCount - 3 + c] = task.lanes[0].temporaries[c];
      task.lanes[0].temporary_written.set(kPcoTemporaryCount - 3 + c);
    }
    Samples samples; TessellationMemoryCallbacks cb{&samples, nullptr, nullptr, Samples::Sample};
    TessellationExecutionStats stats;
    while (!task.ended) StepTessellationTask(program, abi, task, cb, stats);
    Check(samples.requests.size() == 1, "full TEMP file boundary request uses one real callback");
    for (unsigned c = 0; c < 4; ++c)
      Check(task.lanes[0].temporaries[kPcoTemporaryCount - 4 + c] == Samples::Response(samples.requests[0])[c],
            "overlapping upper TEMP payload/response is safe until WDF");
    auto bad = program; ++bad.instructions[0].source.index;
    Reject([&] { ValidateTessellationProgram(bad, abi); }, "native SMP source/response/descriptor layout");
    bad = program; ++bad.instructions[0].output_index;
    Reject([&] { ValidateTessellationProgram(bad, abi); }, "native SMP source/response/descriptor layout");
  }
}
void Mutations() {
  for (auto stage : {ShaderStage::kTessellationControl, ShaderStage::kTessellationEvaluation}) {
    const auto abi = Abi(stage);
    const auto original = Program(stage, true);
    std::vector<std::function<void(PcoInstruction &)>> mutations = {
      [](auto &i) { i.source_count = 2; }, [](auto &i) { i.repeat_count = 2; },
      [](auto &i) { i.data_request = 1; }, [](auto &i) { i.component_count = 3; },
      [](auto &i) { i.end_group = 1; }, [](auto &i) { i.target = PcoWriteTarget::kVertexInput; },
      [](auto &i) { i.source.bank = PcoRegisterBank::kShared; }, [](auto &i) { i.source.index = 6; },
      [](auto &i) { i.output_index = 5; }, [](auto &i) { i.source1.bank = PcoRegisterBank::kTemporary; },
      [](auto &i) { --i.source1.index; }, [](auto &i) { ++i.source1.index; },
      [](auto &i) { i.source1.index += 20; i.source2.index += 20; },
      [](auto &i) { i.source2.bank = PcoRegisterBank::kTemporary; }, [](auto &i) { ++i.source2.index; },
      [](auto &i) { i.texture_dimension = 3; }, [](auto &i) { i.texture_fcnorm = 2; },
      [](auto &i) { i.texture_fcnorm = 0; },
      [](auto &i) { i.texture_address_offset = 1; }, [](auto &i) { i.texture_non_normalized_coords = 1; },
      [](auto &i) { i.texture_sample_index_present = 1; }, [](auto &i) { i.texture_spatial_offset_present = 1; },
    };
    for (const auto &mutate : mutations) {
      auto program = original;
      mutate(program.instructions[0]);
      Reject([&] { ValidateTessellationProgram(program, abi); }, "native SMP source/response/descriptor layout");
      auto task = Task(stage, abi, 1);
      Samples samples;
      TessellationMemoryCallbacks cb{&samples, nullptr, nullptr, Samples::Sample};
      TessellationExecutionStats stats;
      Reject([&] { StepTessellationTask(program, abi, task, cb, stats); }, "native SMP source/response/descriptor layout");
      Check(samples.requests.empty(), "invalid native metadata must fail before callback");
    }
    for (const auto &mutate : std::vector<std::function<void(PcoInstruction &)>>{
        [](auto &i) { i.texture_lod_replace = 2; }, [](auto &i) { i.texture_gather = 1; },
        [](auto &i) { i.texture_lod_bias = 1; }}) {
      auto program = original;
      mutate(program.instructions[0]);
      Reject([&] { ValidateTessellationProgram(program, abi); }, "texture LOD mode");
      auto task = Task(stage, abi, 1); TessellationExecutionStats stats;
      Reject([&] { StepTessellationTask(program, abi, task, {}, stats); }, "texture LOD mode");
    }
    for (unsigned descriptor : {System(stage) - 1, System(stage) + 1, System(stage) + 180}) {
      auto bad = abi;
      bad.uniform_buffer_descriptor_start = descriptor;
      bad.push_constant_start = descriptor + 4;
      bad.shareds = bad.push_constant_start + 4;
      Reject([&] { ValidateTessellationProgram(original, bad); }, "register/descriptor/push ABI");
    }
    const auto none = Abi(stage, 0);
    Reject([&] { ValidateTessellationProgram(original, none); }, "native SMP source/response/descriptor layout");
    auto interrupted = original;
    interrupted.instructions.insert(interrupted.instructions.begin() + 1, original.instructions.back());
    ++interrupted.summary.instruction_count;
    Reject([&] { ValidateTessellationProgram(interrupted, abi); }, "not followed by WDF");
    for (unsigned c = 0; c < 3; ++c) {
      auto task = Task(stage, abi, 1);
      task.lanes[0].temporary_written.words[0] &= ~(UINT64_C(1) << c);
      Samples samples; TessellationMemoryCallbacks cb{&samples, nullptr, nullptr, Samples::Sample};
      TessellationExecutionStats stats;
      Reject([&] { StepTessellationTask(original, abi, task, cb, stats); }, "TEMP read before write");
      Check(samples.requests.empty(), "missing UV/LOD fails before callback");
    }
    auto task = Task(stage, abi, 1); TessellationExecutionStats stats;
    Reject([&] { StepTessellationTask(original, abi, task, {}, stats); }, "no TextureUnit request callback");
    for (bool bad_count : {false, true}) {
      task = Task(stage, abi, 1); stats = {};
      Samples samples; TessellationMemoryCallbacks cb{&samples, nullptr, nullptr, Samples::Sample};
      StepTessellationTask(original, abi, task, cb, stats);
      if (bad_count) task.lanes[0].pending_count = 3; else task.lanes[0].pending_output = 5;
      Reject([&] { StepTessellationTask(original, abi, task, cb, stats); }, "pending response span");
    }
    task = Task(stage, abi, 1); stats = {};
    Samples samples; TessellationMemoryCallbacks cb{&samples, nullptr, nullptr, Samples::Sample};
    StepTessellationTask(original, abi, task, cb, stats);
    task.lanes[0].pending_operation = 4;
    Reject([&] { StepTessellationTask(original, abi, task, cb, stats); }, "unknown pending memory operation");
    Check(task.lanes[0].pending_operation == 4 && !task.lanes[0].temporary_written.test(4), "unknown pending response is not cleared or published");
    task = Task(stage, abi, 1); stats = {};
    StepTessellationTask(original, abi, task, cb, stats);
    task.instruction_index = 2;
    Reject([&] { StepTessellationTask(original, abi, task, cb, stats); }, "requires its immediate WDF");
    task = Task(stage, abi, 1); stats = {};
    StepTessellationTask(original, abi, task, cb, stats);
    task.lanes[0].pending_output = 0; // In bounds, but not the issued request.
    Reject([&] { StepTessellationTask(original, abi, task, cb, stats); }, "pending response span");
    auto no_request = original;
    no_request.instructions[0] = original.instructions.back();
    task = Task(stage, abi, 1); stats = {};
    StepTessellationTask(original, abi, task, cb, stats);
    Reject([&] { StepTessellationTask(no_request, abi, task, cb, stats); }, "no matching request before WDF");
  }
}
struct CompilerMemory {
  std::array<std::uint32_t, 12> input{};
  std::array<std::uint32_t, 34> output{};
  std::array<bool, 34> written{};
  std::vector<PcoTextureRequest> requests;
  bool control = true;
  static void Read(void *opaque, std::uint64_t address, std::uint32_t count, std::uint32_t *words) {
    auto &m = *static_cast<CompilerMemory *>(opaque);
    const bool patch = address >= UINT64_C(0x8100002000);
    const auto base = patch ? UINT64_C(0x8100002000) : UINT64_C(0x8100001000);
    const auto size = patch ? m.output.size() : m.input.size();
    Check(address >= base && (address - base) % 4 == 0 && (address - base) / 4 <= size &&
              count <= size - (address - base) / 4, "real compiler LD stays in supplied input/patch memory");
    const auto first = (address - base) / 4;
    for (unsigned c = 0; c < count; ++c) {
      if (patch) Check(m.written[first + c], "real barrier/cross-invocation LD reads an actual prior native store");
      words[c] = patch ? m.output[first + c] : m.input[first + c];
    }
  }
  static void Write(void *opaque, std::uint64_t address, std::uint32_t count, const std::uint32_t *words) {
    auto &m = *static_cast<CompilerMemory *>(opaque);
    const auto base = UINT64_C(0x8100002000);
    Check(address >= base && (address - base) % 4 == 0 && (address - base) / 4 <= m.output.size() &&
              count <= m.output.size() - (address - base) / 4, "real compiler ST stays in native patch output");
    const auto first = (address - base) / 4;
    for (unsigned c = 0; c < count; ++c) { m.output[first + c] = words[c]; m.written[first + c] = true; }
  }
  static std::array<std::uint32_t, 4> Response(const PcoTextureRequest &r) {
    return {Bits(Float(r.coordinates[0]) + .5f), Bits(Float(r.coordinates[1]) + .25f),
            Bits(r.explicit_lod_present ? Float(r.explicit_lod) + .5f : .75f), Bits(1)};
  }
  static void Sample(void *opaque, const PcoTextureRequest &r, std::uint32_t *words) {
    auto &m = *static_cast<CompilerMemory *>(opaque);
    m.requests.push_back(r);
    // Constant red for TCS keeps all concurrent level stores equal. Their
    // per-invocation UV/LOD requests are still independently verified below.
    const auto response = m.control ? std::array<std::uint32_t, 4>{Bits(1.25f), Bits(2), Bits(3), Bits(4)} : Response(r);
    std::copy(response.begin(), response.end(), words);
  }
};
void CompilerFixtures() {
  // Genuine complete TCS/TES binaries, with input-derived patch/UV/LOD and
  // callback transport oracles. This is not a real TPU texel/filter test.
  for (bool explicit_lod : {false, true}) {
    const auto &tes_bytes = explicit_lod ? kTessTextureExplicitTes : kTessTextureImplicitTes;
    const auto control = DecodeTessellationPcoProgram(ShaderStage::kTessellationControl, kTessTextureTcs);
    const auto evaluation = DecodeTessellationPcoProgram(ShaderStage::kTessellationEvaluation, tes_bytes);
    DriverPcoStageAbi tcs, tes;
    tcs.temps = 14; tcs.vertex_inputs = 3; tcs.shareds = 28;
    tcs.uniform_buffer_descriptor_start = tcs.push_constant_start = 28;
    tes.temps = 20; tes.vertex_inputs = 5; tes.vertex_outputs = 8; tes.shareds = 24;
    tes.uniform_buffer_descriptor_start = tes.push_constant_start = 24;
    ValidateTessellationProgram(control, tcs);
    ValidateTessellationProgram(evaluation, tes);
    for (const auto *program : {&control, &evaluation}) {
      const bool is_control = program->summary.stage == ShaderStage::kTessellationControl;
      const PcoInstruction *sample = nullptr;
      unsigned count = 0;
      for (const auto &i : program->instructions) if (i.opcode == PcoOpcode::kTextureSample) { sample = &i; ++count; }
      Check(count == 1 && sample && sample->texture_fcnorm == 1 && sample->texture_lod_replace == (is_control || explicit_lod), "true compiler FCNORM SMP ordinary/dynamic LOD decoded");
      Check(sample->source1.index == (is_control ? 8 : 4) && sample->source2.index == (is_control ? 16 : 12),
            "true compiler stage-local descriptor namespace");
      const auto &original = is_control ? kTessTextureTcs : tes_bytes;
      for (unsigned mutation = 0; mutation < 4; ++mutation) {
        auto bad = original;
        if (mutation == 0) bad[sample->binary_offset] |= 8; // unsupported DRC1
        if (mutation == 1) bad[sample->binary_offset + 1] |= 128; // reserved extended backend
        if (mutation == 2) bad[sample->binary_offset + 1] &= ~12U; // CHAN1 is not ordinary count4
        if (mutation == 3) bad[sample->binary_offset] &= ~16U; // raw/integer response is outside initial FCNORM contract
        Reject([&] { DecodeTessellationPcoProgram(program->summary.stage, bad); }, "SMP");
      }
    }
    for (unsigned epoch : {0U, 1U, 3U}) {
      CompilerMemory memory;
      for (unsigned i = 0; i < 12; ++i) memory.input[i] = Bits(float(i + epoch + 1) / 32);
      memory.output.fill(0xc0decafe);
      std::vector<std::uint32_t> shared{0x1000, 0x81, 48, 0, 0x2000, 0x81, 136, 0};
      shared.resize(28);
      for (unsigned c = 8; c < shared.size(); ++c) shared[c] = 0x5a000000 + c;
      const auto task_shared = shared;
      auto task = MakeTessellationControlTask(tcs, shared, epoch, 3, 3);
      TessellationMemoryCallbacks cb{&memory, CompilerMemory::Read, CompilerMemory::Write, CompilerMemory::Sample};
      TessellationExecutionStats stats;
      while (!task.ended) StepTessellationTask(control, tcs, task, cb, stats);
      Check(stats.texture_instructions == 3 && memory.requests.size() == 3, "true TCS issues one SMP per output invocation");
      for (unsigned i = 0; i < 3; ++i) {
        const auto &r = memory.requests[i];
        Check(r.coordinates[0] == Bits(.125f + i * .25f) && r.coordinates[1] == Bits(.5f) &&
                  r.explicit_lod_present == 1 && r.explicit_lod == Bits(float(i)), "true TCS computes dynamic input-derived UV and LOD");
        for (unsigned c = 0; c < 4; ++c)
          Check(r.texture_state[c] == task_shared[8 + c] && r.sampler_state[c] == task_shared[16 + c], "TCS actual descriptor words reach callback");
      }
      for (unsigned c = 0; c < 6; ++c) Check(memory.output[c] == Bits(1.25f), "real TCS stores returned red to six tessellation levels");
      for (unsigned c = 0; c < 4; ++c) Check(memory.output[6 + c] == Bits(float(c + 1) * .25f), "actual invocation-zero patch varying");
      for (unsigned vertex = 0; vertex < 3; ++vertex) for (unsigned c = 0; c < 4; ++c) {
        Check(memory.output[10 + vertex * 8 + c] == memory.input[vertex * 4 + c], "actual per-invocation native positions");
        Check(memory.output[14 + vertex * 8 + c] == memory.input[((vertex + 1) % 3) * 4 + c], "real barrier retains cross-invocation data dependence");
      }
      for (unsigned count : {1U, 7U, 32U}) {
        memory.control = false; memory.requests.clear();
        std::array<std::array<std::uint32_t, 3>, 32> coordinates{};
        for (unsigned lane = 0; lane < count; ++lane) {
          const float u = float(lane % 4) * .125f, v = float(lane % 3) * .125f;
          coordinates[lane] = {Bits(u), Bits(v), Bits(1 - u - v)};
        }
        std::vector<std::uint32_t> te_shared{0x2000, 0x81, 136, 0};
        te_shared.resize(24);
        for (unsigned c = 4; c < te_shared.size(); ++c) te_shared[c] = 0x6b000000 + c;
        auto te = MakeTessellationEvaluationTask(tes, te_shared, epoch, 3, coordinates.data(), count);
        TessellationExecutionStats te_stats;
        while (!te.ended) StepTessellationTask(evaluation, tes, te, cb, te_stats);
        Check(te_stats.texture_instructions == count && memory.requests.size() == count && te_stats.emit_instructions == count,
              "true TES samples and emits once per supplied domain point");
        for (unsigned lane = 0; lane < count; ++lane) {
          const auto &r = memory.requests[lane];
          Check(r.coordinates[0] == coordinates[lane][0] && r.coordinates[1] == coordinates[lane][1] &&
                    r.explicit_lod_present == explicit_lod && r.explicit_lod == (explicit_lod ? Bits(float(epoch)) : 0),
                "true TES preserves domain coordinates and dynamic primitive-ID LOD");
          const auto response = CompilerMemory::Response(r);
          for (unsigned c = 0; c < 4; ++c) {
            Check(r.texture_state[c] == te_shared[4 + c] && r.sampler_state[c] == te_shared[12 + c], "TES slot0 does not borrow TCS slot0 state");
            const float position = std::fma(Float(memory.input[8 + c]), Float(coordinates[lane][2]),
                std::fma(Float(memory.input[4 + c]), Float(coordinates[lane][1]), Float(memory.input[c]) * Float(coordinates[lane][0])));
            Check(te.lanes[lane].outputs[c] == Bits(position), "real TES dyadic interpolated position stays exact");
            Check(te.lanes[lane].outputs[4 + c] == Bits((float(c + 1) * .25f + Float(memory.input[4 + c])) * Float(response[c])),
                  "real TES multiplies native patch data by its returned four sample words");
          }
        }
      }
    }
  }
}
} // namespace
int main() {
  try {
    RequestsAndFence(); Predication(); HighTemporaryBoundary(); Mutations(); CompilerFixtures();
    std::cout << "native tessellation texture ISS decoded-unit and true-compiler fixtures: PASS " << checks << " checks\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << " after " << checks << " checks\n";
    return 1;
  }
}
