#include "shader/usc_shader_image_memory.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>

using namespace pvrgpu::stub;
extern "C" void test_native_fragment_images(void);
static unsigned checks;
static void Require(bool value, const char *message) {
  ++checks; if (!value) throw std::runtime_error(message);
}
static std::uint32_t Bits(float value) {
  std::uint32_t word; std::memcpy(&word, &value, 4); return word;
}

extern "C" void check_native_fragment_image(const std::uint8_t *bytes,
    std::size_t size, unsigned shared_count, unsigned cb0_start,
    unsigned image_count, unsigned kind) {
  const auto program = DecodePcoProgram(ShaderStage::kFragment,
      std::vector<std::uint8_t>(bytes, bytes + size));
  Require(!program.summary.early_hsr_safe, "image atomic must be HSR-unsafe");
  const unsigned image_start = kind == 4 ? 20 : 0;
  Require(cb0_start == image_start + 8 * image_count && shared_count == cb0_start + 4,
          "fragment image descriptor/CB0 layout is wrong");
  for (const auto mode : {MemoryMode::kDirect, MemoryMode::kBypass, MemoryMode::kCache}) {
    GpuMemorySystem memory(mode);
    constexpr std::uint64_t address = UINT64_C(0x740000000000);
    std::array<std::uint32_t, 24> words;
    words.fill(0xdeadbeef);
    words[4] = 10; words[5] = 20; words[8] = 30; words[9] = UINT32_MAX - 1;
    memory.HostWrite(address, words.data(), sizeof(words));
    std::vector<ShaderImageResource> images;
    for (unsigned slot = 0; slot < image_count; ++slot) {
      ShaderImageResource image;
      image.resource_token = 23; image.gpu_address = address; image.bytes = sizeof(words);
      image.offset = 16; image.image_slot = slot; image.format = 1; image.access = 3;
      image.width = 2; image.height = 2; image.depth = 1;
      image.row_stride = 16; image.layer_stride = 32; image.texel_bytes = 4;
      images.push_back(image);
    }
    UscShaderImageMemory adapter(&memory, mode, images);
    PcoFragmentExecutionContext context;
    context.shared_count = shared_count;
    context.memory_atomic32 = UscShaderImageMemory::Atomic32;
    context.image_memory_user_data = &adapter;
    for (unsigned slot = 0; slot < image_count; ++slot) {
      const std::array<std::uint32_t, 8> descriptor{
          std::uint32_t(address + 16), std::uint32_t((address + 16) >> 32),
          1, 32, 2, 2, 16, 4};
      std::copy(descriptor.begin(), descriptor.end(), context.shared_registers.begin() + image_start + 8 * slot);
    }
    const auto invoke = [&](std::uint32_t x, std::uint32_t y, bool helper) {
      const auto previous_atomics = adapter.atomics();
      context.shared_registers[cb0_start] = x;
      context.shared_registers[cb0_start + 1] = y;
      context.shared_registers[cb0_start + 2] = 3;
      context.shared_registers[cb0_start + 3] = 5;
      context.memory_side_effects_enabled = helper ? 0 : 1;
      auto result = ExecuteFragmentPco(program.summary, program.instructions, context);
      auto visited = result.executed_instruction_count;
      if (kind == 4) {
        Require(result.suspended && result.texture_request_valid,
                "image/texture program did not reach its native SMP checkpoint");
        const auto atomics = adapter.atomics();
        auto resume = context;
        resume.continuation = result.continuation;
        resume.texture_response_valid = 1;
        resume.texture_response = {Bits(.125f), Bits(.25f), Bits(.5f), Bits(1.f)};
        result = ExecuteFragmentPco(program.summary, program.instructions, resume);
        visited += result.executed_instruction_count;
        Require(adapter.atomics() == atomics, "texture continuation replayed a fragment image atomic");
      }
      Require(result.native_steps == visited &&
              result.executed_instructions.memory == adapter.atomics() - previous_atomics &&
              result.executed_instructions.texture == (kind == 4 ? 1 : 0),
              "native image/helper/OOB/SMP counters do not match actual requests");
      return result;
    };
    for (unsigned epoch = 0; epoch < 3; ++epoch) {
      for (unsigned y = 0; y < 2; ++y) for (unsigned x = 0; x < 2; ++x) {
        const unsigned index = 4 + y * 4 + x;
        const auto old = words[index];
        const auto result = invoke(x, y, false);
        Require(!result.suspended && !result.discarded && result.written_mask == 15,
                "native fragment atomic did not finish its PIXOUT program");
        Require(result.pixel_outputs[0] == Bits(kind == 0 ? 1.f :
            float(old + (kind == 3 ? 3 : 0)) + (kind == 4 ? .125f : 0.f)),
                "native atomic old value/WDF result is wrong");
        words[index] += kind == 3 ? 8 : 3;
      }
      const auto atomics = adapter.atomics();
      invoke(0, 0, true);
      invoke(UINT32_MAX, 0, false);
      invoke(0, UINT32_MAX, false);
      invoke(2, 0, false);
      invoke(0, 2, false);
      Require(adapter.atomics() == atomics, "helper/out-of-bounds image atomic modified memory");
      const auto readback = adapter.Readback(images[0]);
      Require(readback.size() == sizeof(words) &&
              std::memcmp(readback.data(), words.data(), sizeof(words)) == 0,
              "image atomic result or untouched backing padding is wrong");
    }
    Require(adapter.atomics() == 12 * image_count, "native image atomic count is wrong");
    bool rejected = false;
    try { UscShaderImageMemory::Atomic32(&adapter, PcoOpcode::kAtomicAdd32, address + 24, 1); }
    catch (const std::runtime_error &) { rejected = true; }
    Require(rejected, "padding outside image view accepted an atomic");
    rejected = false;
    context.memory_side_effects_enabled = 2;
    try { ExecuteFragmentPco(program.summary, program.instructions, context); }
    catch (const std::runtime_error &) { rejected = true; }
    Require(rejected, "noncanonical helper memory flag was accepted");
  }
}

int main() {
  try { test_native_fragment_images(); }
  catch (const std::exception &error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
  std::printf("native fragment image compiler/ISS/GPU memory: PASS (%u checks)\n", checks);
}
