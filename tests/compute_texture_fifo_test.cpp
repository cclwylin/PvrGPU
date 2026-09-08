// Genuine compute PCO SMP/WDF -> independent ComputeShader -> TPU -> GPU memory.
// No vertex/fragment executor or framebuffer exists in this test harness.
#include "common/pipeline_state.h"
#include "data_master/compute_data_master.h"
#include "memory/gpu_memory_system.h"
#include "pco_compute_texture_fixtures.h"
#include "shader/compute_shader.h"
#include "texture/texture_unit.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
using namespace pvrgpu::stub;
unsigned checks = 0;
void Check(bool ok, const char *message) {
  ++checks;
  if (!ok) throw std::runtime_error(message);
}
template <class T> PoolHandle Store(MemoryPool &pool, const T &value) {
  auto h = pool.Allocate(sizeof(value));
  std::memcpy(pool.Write(h).data(), &value, sizeof(value));
  return h;
}
template <class T> T Load(MemoryPool &pool, PoolHandle h) {
  T value;
  Check(pool.Read(h).size() == sizeof(value), "POD size");
  std::memcpy(&value, pool.Read(h).data(), sizeof(value));
  return value;
}
std::uint32_t Bits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, 4);
  return bits;
}

void Run(MemoryPool &pool, GpuMemorySystem &memory,
         sc_core::sc_fifo<ComputeDispatchTxn> &input,
         sc_core::sc_fifo<ComputeDispatchTxn> &output,
         unsigned shape, unsigned format, unsigned epoch, unsigned sequence) {
  const auto abi = ComputeTexturePcoAbi(300 + shape);
  const auto &binary = ComputeTexturePcoFixture(300 + shape);
  const auto decoded = DecodeComputePcoProgram(binary);
  ValidateComputeProgram(decoded, abi);
  Check(std::count_if(decoded.instructions.begin(), decoded.instructions.end(),
      [](const auto &i) { return i.opcode == PcoOpcode::kTextureSample; }) == 1,
      "fixture did not encode native SMP");
  auto invalid = abi;
  invalid.sampled_texture_count = 0;
  bool rejected = false;
  try { ValidateComputeProgram(decoded, invalid); } catch (const std::exception &) { rejected = true; }
  Check(rejected, "compute accepted a mismatched texture descriptor ABI");

  constexpr std::uint64_t texture_address = UINT64_C(0x3000000000);
  constexpr std::uint64_t output_address = UINT64_C(0x1000000000000);
  const unsigned layers = shape == 2 ? 6 : shape ? 2 : 1;
  const unsigned texel_bytes = format == 0 ? 16 : format == 1 ? 8 : 4;
  std::vector<std::uint8_t> bytes(16 * layers * texel_bytes);
  std::vector<std::array<float,4>> expected_texels(16 * layers);
  const unsigned halves[] = {0, 0x3000, 0x3400, 0x3600, 0x3800, 0x3900, 0x3a00, 0x3b00};
  for (unsigned layer = 0; layer < layers; ++layer)
    for (unsigned y = 0; y < 4; ++y)
      for (unsigned x = 0; x < 4; ++x) {
        const unsigned texel = layer * 16 + y * 4 + x;
        const unsigned values[] = {(x + epoch) % 4, y, layer, 7};
        auto &expected = expected_texels[texel];
        for (unsigned c = 0; c < 4; ++c) {
          if (format < 2) {
            expected[c] = c == 3 ? 1.0F : values[c] * 0.125F;
            if (format == 0) {
              auto bits = Bits(expected[c]);
              std::memcpy(bytes.data() + texel * texel_bytes + 4 * c, &bits, 4);
            } else {
              const std::uint16_t half = c == 3 ? 0x3c00 : halves[values[c]];
              std::memcpy(bytes.data() + texel * texel_bytes + 2 * c, &half, 2);
            }
          } else {
            const unsigned scale = format == 2 ? 255 : 127;
            const unsigned raw = c == 3 ? scale : values[c] * 16;
            bytes[texel * 4 + c] = raw;
            expected[c] = static_cast<float>(raw) / scale;
          }
        }
      }
  memory.HostWrite(texture_address, bytes.data(), bytes.size());
  const std::array<std::uint32_t,16> poison{};
  memory.HostWrite(output_address, poison.data(), sizeof(poison));
  std::vector<std::uint32_t> shared(abi.stage.shareds, 0);
  const auto set64 = [&](unsigned offset, std::uint64_t value) {
    shared[offset] = static_cast<std::uint32_t>(value);
    shared[offset + 1] = static_cast<std::uint32_t>(value >> 32);
  };
  const unsigned rogue_format[] = {61, 28, 12, 13};
  set64(0, (layers > 1 ? 1U : 4U) | (UINT64_C(3) << 5) | (UINT64_C(2) << 8) |
      (UINT64_C(1) << 11) | (static_cast<std::uint64_t>(rogue_format[format]) << 27) |
      (UINT64_C(3) << 34) | (UINT64_C(3) << 48));
  set64(2, (layers > 1 ? 1U | ((layers - 1U) << 4) : UINT64_C(3) | (UINT64_C(1) << 60)) |
      ((texture_address >> 2) << 16));
  shared[4] = shape == 3 ? 16 * texel_bytes : bytes.size();
  const auto sampler_word = UINT64_C(4095) | (UINT64_C(2) << 33) |
      (UINT64_C(2) << 41) | (UINT64_C(2) << 56);
  set64(8, sampler_word);
  set64(16, sampler_word | (UINT64_C(1) << 36) | (UINT64_C(1) << 38));
  set64(20, output_address);
  shared[22] = sizeof(poison);
  std::array<unsigned,4> sampled_texels{};
  for (unsigned lane = 0; lane < 4; ++lane) {
    float xyz[] = {(lane + 0.5F) / 4, (lane + 0.5F) / 4,
                   shape == 3 ? static_cast<float>(lane % 2) : (lane % 2 + 0.5F) / 2};
    if (shape == 2) {
      xyz[0] = lane == 0 ? 1 : lane == 1 ? -1 : 0;
      xyz[1] = lane == 2 ? 1 : lane == 3 ? -1 : 0;
      xyz[2] = 0;
      sampled_texels[lane] = lane * 16 + 10;
    } else sampled_texels[lane] = (shape ? lane % 2 * 16 : 0) + lane * 5;
    for (unsigned c = 0; c < 3; ++c) shared[24 + lane * 4 + c] = Bits(xyz[c]);
  }
  TextureResource resource;
  resource.gpu_address = texture_address;
  resource.byte_size = bytes.size(); resource.mip_count = 1;
  const TextureFormat formats[] = {TextureFormat::kRgba32Float, TextureFormat::kRgba16Float,
      TextureFormat::kRgba8Unorm, TextureFormat::kRgba8Snorm};
  resource.format = formats[format]; resource.layer_count = layers;
  resource.dimension_type = shape == 1 ? TextureDimensionType::k3D :
      shape == 2 ? TextureDimensionType::kCube : shape == 3 ? TextureDimensionType::k2DArray : TextureDimensionType::k2D;
  resource.mip[0] = {4,4,4 * texel_bytes,0};
  SamplerState sampler;
  sampler.wrap_u = sampler.wrap_v = TextureWrapMode::kClampToEdge;
  PipelineState texture;
  texture.memory_mode = memory.mode(); texture.compute_pco_abi = abi.stage;
  texture.compute_sampled_texture_count = 1;
  texture.compute_shared_registers = StoreNewArray(pool, shared);
  texture.compute_texture_resources = StoreNewArray(pool, std::vector<TextureResource>{resource});
  texture.compute_sampler_states = StoreNewArray(pool, std::vector<SamplerState>{sampler});
  ComputeDispatchState dispatch;
  dispatch.abi = abi; dispatch.grid = {1,1,1}; dispatch.sequence = sequence;
  dispatch.code = StoreNewArray(pool, binary);
  dispatch.shared_registers = texture.compute_shared_registers;
  dispatch.texture_state = Store(pool, texture);
  dispatch.buffer_ranges = StoreNewArray(pool, std::vector<ComputeBufferRange>{
      {output_address, sizeof(poison), kComputeAccessWrite, 0, 1}});
  const auto handle = Store(pool, dispatch);
  Check(input.nb_write({handle, sequence}), "dispatch FIFO was not empty");
  ComputeDispatchTxn done;
  bool received = false;
  while (!(received = output.nb_read(done)) && sc_core::sc_pending_activity())
    sc_core::sc_start(sc_core::sc_time_to_pending_activity());
  Check(received && done.sequence == sequence, "compute SMP dispatch failed to complete");
  const auto final = Load<ComputeDispatchState>(pool, handle);
  if (final.failed) throw std::runtime_error(final.error.data());
  Check(final.stats.invocations == 4 && final.counters.vs_invocations == 0 &&
        final.counters.ps_invocations == 0, "compute invoked graphics stage");
  const auto complete_texture = LoadPipelineState(pool, dispatch.texture_state);
  Check(complete_texture.compute_texture_request_count == 4 &&
        complete_texture.compute_texel_fetch_count == 4, "native SMP bypassed TPU");
  Check(!HasPoolHandle(complete_texture.texture_sample_requests) &&
        !HasPoolHandle(complete_texture.texture_sample_responses), "SMP/WDF payload not released");
  const auto readback = memory.Readback(output_address, sizeof(poison), MemoryClient::kComputeReadback);
  std::array<float,16> actual{};
  std::memcpy(actual.data(), readback.data.data(), sizeof(actual));
  for (unsigned lane = 0; lane < 4; ++lane)
    for (unsigned c = 0; c < 4; ++c)
      Check(std::fabs(actual[4 * lane + c] - expected_texels[sampled_texels[lane]][c]) < 0.000001F,
            "native compute TextureUnit float/normalized result mismatch");
  for (const auto h : {dispatch.code, dispatch.shared_registers, dispatch.buffer_ranges,
      dispatch.texture_state, texture.compute_texture_resources, texture.compute_sampler_states, handle}) pool.Release(h);
  Check(pool.allocations() == pool.releases() && pool.bytes_in_flight() == 0,
        "compute texture retained a pool payload");
}
} // namespace

int sc_main(int argc, char **argv) {
  try {
    const unsigned mode = argc > 1 ? std::stoul(argv[1]) : 0;
    Check(mode <= 2, "memory mode");
    MemoryPool pool;
    GpuMemorySystem memory(static_cast<MemoryMode>(mode));
    sc_core::sc_fifo<ComputeDispatchTxn> input("dispatch",1), done("done",1);
    sc_core::sc_fifo<ComputeWorkgroupTxn> groups("groups",1), group_done("group_done",1);
    sc_core::sc_fifo<ComputeMemoryTxn> requests("requests",1), responses("responses",1);
    sc_core::sc_fifo<PipelineTxn> samples("samples",1), sampled("sampled",1), unused_in("unused_in",1), unused_out("unused_out",1);
    ComputeDataMaster cdm("compute_data_master",pool,memory);
    ComputeShader shader("compute_shader",pool);
    TextureUnit texture("texture_unit",pool,&memory);
    cdm.input(input); cdm.completion(done); cdm.workgroup_output(groups); cdm.workgroup_completion(group_done);
    cdm.memory_input(requests); cdm.memory_output(responses);
    shader.input(groups); shader.output(group_done); shader.memory_request_output(requests); shader.memory_response_input(responses);
    shader.texture_request_output(samples); shader.texture_response_input(sampled);
    texture.compute_sample_input(samples); texture.compute_sample_output(sampled);
    texture.input(unused_in); texture.output(unused_out);
    sc_core::sc_start(sc_core::SC_ZERO_TIME);
    unsigned sequence = 0;
    for (unsigned epoch = 0; epoch < 2; ++epoch)
      for (unsigned shape = 0; shape < 4; ++shape)
        for (unsigned format = 0; format < 4; ++format)
          Run(pool,memory,input,done,shape,format,epoch,++sequence);
    Check(samples.num_available() == 0 && sampled.num_available() == 0,
          "compute texture FIFO did not drain");
    std::cout << "compute texture native FIFO: PASS " << checks << " checks\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}
