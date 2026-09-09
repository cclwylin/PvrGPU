// Native raw Z24 / RGBA32Float component-zero gather. Run each negative argv case in its
// own process: a fail-closed SystemC exception terminates that simulation.
// No capture data, expected-image bytes, or shader-side gather emulation.
#include "common/pipeline_state.h"
#include "texture/texture_unit.h"

#include <systemc>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace pvrgpu::stub;
using Point = std::array<float, 2>;

static_assert(sizeof(TextureSampleRequest) == 128,
              "gather must consume only the former reserved byte");
static_assert(offsetof(TextureSampleRequest, gather) ==
              offsetof(TextureSampleRequest, sample_index_present) + 1);

void Check(bool ok, const std::string &message) {
  if (!ok)
    throw std::runtime_error("texture gather test: " + message);
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void StoreWord(std::vector<std::uint32_t> &words, std::size_t at,
               std::uint64_t value) {
  words.at(at) = static_cast<std::uint32_t>(value);
  words.at(at + 1) = static_cast<std::uint32_t>(value >> 32U);
}

std::uint32_t EncodedDepth(std::uint32_t x, std::uint32_t y,
                           std::uint32_t width, std::uint32_t layer = 0) {
  // Asymmetric, nonconstant depth and nonzero stencil detect tap permutation,
  // interpolation, treating response words as RGBA, and stencil contamination.
  return ((y * width + x) * UINT32_C(7919) + UINT32_C(0x123456) +
          layer * UINT32_C(0x100000)) &
         UINT32_C(0xffffff);
}

std::uint32_t DepthBits(std::uint32_t x, std::uint32_t y, std::uint32_t width,
                        std::uint32_t layer = 0) {
  return FloatBits(static_cast<float>(
      static_cast<double>(EncodedDepth(x, y, width, layer)) / 16777215.0));
}

std::array<float, 4> FloatTexel(std::uint32_t x, std::uint32_t y,
                              std::uint32_t width, std::uint32_t layer = 0) {
  const float red[] = {-2.0F, 0.0F, 5.5F, 1234567.0F, -8192.125F,
                       1.0e20F, -1.0e-20F, 0.9375F};
  return {{red[(y * width + x) % 8U] + layer * 4096.0F, 17.25F, -9.5F, 0.375F}};
}

void CheckAxis() {
  struct Case { float s; std::uint32_t extent, first, second; };
  const Case cases[] = {
      {0.5F, 4, 1, 2}, {0.0F, 4, 0, 0}, {1.0F, 4, 3, 3},
      {-0.25F, 4, 0, 0}, {1.25F, 4, 3, 3},
      {0.125F, 4, 0, 1}, {0.375F, 4, 1, 2},
      {std::nextafter(0.125F, 0.0F), 4, 0, 1}, // bits 0x3dffffff
      {std::nextafter(std::nextafter(0.125F, 0.0F), 0.0F), 4, 0, 0},
      {std::nextafter(0.125F, 1.0F), 4, 0, 1},
      {std::nextafter(0.375F, 0.0F), 4, 0, 1},
      {std::nextafter(0.375F, 1.0F), 4, 1, 2},
      {std::nextafter(0.875F, 0.0F), 4, 2, 3},
      {std::nextafter(0.875F, 1.0F), 4, 3, 3},
      {0.5F, 3, 1, 2}, {0.5F, 5, 2, 3},
      {0.0F, 1, 0, 0}, {0.5F, 1, 0, 0}, {1.0F, 1, 0, 0},
      {-std::numeric_limits<float>::max(), 7, 0, 0},
      {std::numeric_limits<float>::max(), 7, 6, 6},
  };
  for (const auto &c : cases)
    Check(ComputeTextureGatherClampToEdge(c.s, c.extent) ==
              std::array<std::uint32_t, 2>{{c.first, c.second}},
          "independently clamped gather axis");
  for (float value : {std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()}) {
    bool rejected = false;
    try { (void)ComputeTextureGatherClampToEdge(value, 4); }
    catch (const std::runtime_error &) { rejected = true; }
    Check(rejected, "nonfinite axis rejected");
  }
  bool rejected = false;
  try { (void)ComputeTextureGatherClampToEdge(0.5F, 0); }
  catch (const std::runtime_error &) { rejected = true; }
  Check(rejected, "empty axis rejected");
}

// Independent reference for the declared finite-precision addressing policy.
// Mesa lp_bld_sample_soa.c PIPE_TEX_WRAP_CLAMP_TO_EDGE gather rounds the
// product and +/-0.5 separately, then truncates. Force each rounding via
// a double expression converted to materialized binary32, not the production
// helper or an ideal-floor oracle. No captured pixel values participate.
std::array<std::uint32_t, 2> ReferenceAxis(float coordinate,
                                         std::uint32_t extent, bool gather) {
  const auto clamp = [extent](double value) {
    return static_cast<std::uint32_t>(std::max(0.0,
        std::min(static_cast<double>(extent - 1U), value)));
  };
  if (!gather) {
    const auto index = clamp(std::floor(static_cast<double>(coordinate) * extent));
    return {{index, index}};
  }
  const auto round_binary32 = [](double value) {
    const volatile float rounded = static_cast<float>(value);
    return rounded;
  };
  const double normalized = std::max(0.0, std::min(1.0, double(coordinate)));
  const float scaled = round_binary32(normalized * extent);
  return {{clamp(std::trunc(round_binary32(double(scaled) - 0.5))),
           clamp(std::trunc(round_binary32(double(scaled) + 0.5)))}};
}

std::array<std::uint32_t, 4> Expected(Point point, std::uint32_t width,
                                    std::uint32_t height, bool gather,
                                    bool rgba_float, std::uint32_t layer = 0) {
  const auto xs = ReferenceAxis(point[0], width, gather);
  const auto ys = ReferenceAxis(point[1], height, gather);
  std::array<std::uint32_t, 4> result{};
  for (std::size_t tap = 0; tap < (gather ? 4U : 1U); ++tap) {
    const auto x = xs[tap & 1U];
    const auto y = ys[tap >> 1U];
    result[tap] = rgba_float ? FloatBits(FloatTexel(x, y, width, layer)[0])
                             : DepthBits(x, y, width, layer);
  }
  if (!gather && rgba_float) {
    const auto rgba = FloatTexel(xs[0], ys[0], width, layer);
    for (std::size_t c = 0; c < 4; ++c) result[c] = FloatBits(rgba[c]);
  } else if (!gather) {
    result[3] = FloatBits(1.0F);
  }
  return result;
}

void Run(const std::string &mode) {
  const bool negative = mode.rfind("reject-", 0) == 0;
  const bool array = mode.rfind("array-", 0) == 0 ||
                     mode.rfind("reject-array-", 0) == 0;
  const std::string without_array = mode.rfind("array-", 0) == 0 ? mode.substr(6) : mode;
  const bool rgba_float = without_array.rfind("rgba-", 0) == 0 ||
                          mode.rfind("reject-rgba-", 0) == 0;
  const std::string configuration = without_array.rfind("rgba-", 0) == 0 ? without_array.substr(5) : without_array;
  const bool min_linear = configuration == "linear" || configuration == "min-linear";
  const bool mag_linear = configuration == "linear" || configuration == "mag-linear";
  const std::uint32_t width = configuration == "1x1" ? 1U : configuration == "odd" ? 17U : 4U;
  const std::uint32_t height = configuration == "1x1" ? 1U : configuration == "odd" ? 7U : 4U;
  const std::uint32_t texel_bytes = rgba_float ? 16U : 4U;
  const std::uint32_t layers = array ? 4U : 1U;
  const std::uint32_t samples = mode == "reject-array-msaa" ? 2U : 1U;
  const bool mipped = mode == "reject-array-mips";
  const std::uint32_t layer_stride = width * height * texel_bytes * samples;
  const std::uint32_t mip1_width = std::max(1U, width / 2U);
  const std::uint32_t mip1_height = std::max(1U, height / 2U);
  constexpr std::uint64_t base = UINT64_C(0x30000000);
  MemoryPool pool;
  const MemoryMode memory_mode = configuration == "direct" ? MemoryMode::kDirect
      : configuration == "bypass" ? MemoryMode::kBypass : MemoryMode::kCache;
  GpuMemorySystem memory(memory_mode);
  std::vector<std::uint8_t> bytes(layer_stride * layers +
      (mipped ? mip1_width * mip1_height * texel_bytes * layers : 0U));
  for (std::uint32_t layer = 0; layer < layers; ++layer)
   for (std::uint32_t y = 0; y < height; ++y)
    for (std::uint32_t x = 0; x < width; ++x) {
      if (rgba_float) {
        const auto rgba = FloatTexel(x, y, width, layer);
        std::memcpy(bytes.data() + layer * layer_stride + (y * width + x) * texel_bytes,
                    rgba.data(), sizeof(rgba));
        continue;
      }
      const std::uint32_t packed = EncodedDepth(x, y, width, layer) |
                                  ((UINT32_C(0xa5) ^ x ^ y) << 24U);
      for (std::size_t byte = 0; byte < 4; ++byte)
        bytes[layer * layer_stride + (y * width + x) * 4U + byte] =
            static_cast<std::uint8_t>(packed >> (byte * 8U));
    }
  memory.HostWrite(base, bytes.data(), bytes.size());
  TextureResource resource;
  resource.gpu_address = base;
  resource.byte_size = bytes.size();
  resource.format = rgba_float ? TextureFormat::kRgba32Float : TextureFormat::kZ24UnormS8Uint;
  resource.mip_count = mipped ? 2 : 1;
  resource.sample_count = samples;
  resource.dimension_type = array ? TextureDimensionType::k2DArray : TextureDimensionType::k2D;
  resource.layer_count = layers;
  resource.mip[0] = {width, height, width * texel_bytes * samples, 0};
  if (mipped)
    resource.mip[1] = {mip1_width, mip1_height, mip1_width * texel_bytes,
                       layer_stride * layers};
  SamplerState sampler;
  sampler.wrap_u = sampler.wrap_v = TextureWrapMode::kClampToEdge;
  sampler.min_filter = min_linear ? TextureFilter::kLinear : TextureFilter::kNearest;
  sampler.mag_filter = mag_linear ? TextureFilter::kLinear : TextureFilter::kNearest;
  if (configuration == "lod-window") {
    sampler.min_lod_u4_6 = 128;
    sampler.max_lod_u4_6 = 256;
    sampler.mip_filter = TextureFilter::kLinear;
  }
  std::vector<std::uint32_t> shared(kFillTexNearestSharedDwordCount);
  const std::uint64_t channels = rgba_float
      ? (UINT64_C(3) << 5U) | (UINT64_C(2) << 8U) | (UINT64_C(1) << 11U) | (UINT64_C(61) << 27U)
      : (UINT64_C(4) << 5U) | (UINT64_C(5) << 8U) | (UINT64_C(5) << 11U) | (UINT64_C(22) << 27U);
  const std::uint64_t image0 = (array ? UINT64_C(1) : UINT64_C(4)) | channels |
      (static_cast<std::uint64_t>(width - 1U) << 34U) |
      (static_cast<std::uint64_t>(height - 1U) << 48U) |
      (samples == 2 ? UINT64_C(1) << 62U : 0U);
  const std::uint64_t image1 = ((base >> 2U) << 16U) |
      (array ? static_cast<std::uint64_t>(resource.mip_count) |
                   (mipped ? UINT64_C(1) << 15U : 0U) |
                   (static_cast<std::uint64_t>(layers - 1) << 4U)
             : (UINT64_C(1) << 60U) | (width - 1U));
  StoreWord(shared, 0, image0);
  StoreWord(shared, 2, image1);
  shared[4] = layer_stride;
  shared[7] = 0x100U;  // UNORM pack metadata is valid without shadow compare.
  // Compare metadata is consumed by native shader ALU, never by raw TPU
  // gather. Every compare function must leave these four raw tap bits intact.
  shared[12] = array ? 3U : 0U;
  if (configuration.rfind("compare-", 0) == 0)
    shared[12] = static_cast<std::uint32_t>(std::stoul(configuration.substr(8)));
  std::uint64_t sampler0 = UINT64_C(4095) | (UINT64_C(2) << 33U) |
      (UINT64_C(2) << 41U) |
      (static_cast<std::uint64_t>(sampler.min_lod_u4_6) << 13U) |
      (static_cast<std::uint64_t>(sampler.max_lod_u4_6) << 23U) |
      (static_cast<std::uint64_t>(mag_linear) << 36U) |
      (static_cast<std::uint64_t>(min_linear) << 38U) |
      (static_cast<std::uint64_t>(sampler.mip_filter == TextureFilter::kLinear) << 40U);
  if (mode == "reject-repeat") {
    sampler0 &= ~(UINT64_C(7) << 33U);
    sampler.wrap_u = TextureWrapMode::kRepeat;
  }
  StoreWord(shared, 8, sampler0);
  StoreWord(shared, 16, sampler0 | (UINT64_C(1) << 36U) | (UINT64_C(1) << 38U));
  if (mode == "reject-red-swizzle" || mode == "reject-rgba-red-swizzle")
    shared[0] |= UINT32_C(1) << 14U;
  if (mode == "reject-gather-descriptor") shared[17] ^= UINT32_C(1) << 4U;
  if (mode == "reject-gather-word1") shared[18] = 1U;
  if (mode == "reject-compare") shared[12] = 8U;
  if (mode == "reject-array-stride") shared[4] += 4U;
  if (mode == "reject-array-depth") resource.layer_count = 3U;
  if (mode == "reject-base") sampler.base_mip_level = 1U;
  if (mode == "reject-layers") resource.layer_count = 2U;
  if (mode == "reject-dimension") resource.dimension_type = TextureDimensionType::k3D;
  if (mode == "reject-format") {
    resource.format = TextureFormat::kRgba8Unorm;
    const std::uint64_t rgba = (image0 & ~((UINT64_C(0x7f) << 27U) |
        (UINT64_C(0xfff) << 5U))) | (UINT64_C(12) << 27U) |
        (UINT64_C(3) << 5U) | (UINT64_C(2) << 8U) | (UINT64_C(1) << 11U);
    StoreWord(shared, 0, rgba);
  }
  PipelineState state;
  state.memory_mode = memory_mode;
  state.functional_case = FunctionalCase::kDriverPcoTriangles;
  state.stage = PipelineStage::kFragmentTexturePending;
  state.sampled_texture_count = 1;
  state.fragment_pco_abi.shareds = shared.size();
  state.texture_resources = StoreNewArray(pool, std::vector<TextureResource>{resource});
  state.sampler_states = StoreNewArray(pool, std::vector<SamplerState>{sampler});
  state.fragment_shared_registers = StoreNewArray(pool, shared);
  const PoolHandle handle = pool.Allocate(sizeof(PipelineState));
  const PipelineTxn txn{handle, 1, 1};
  sc_core::sc_fifo<PipelineTxn> input("input", 1), output("output", 1);
  sc_core::sc_fifo<PipelineTxn> requests("requests", 1), responses("responses", 1);
  TextureUnit texture("texture", pool, &memory);
  texture.input(input);
  texture.output(output);
  texture.sample_input(requests);
  texture.sample_output(responses);
  std::set<std::uint64_t> resident_lines;

  const auto batch = [&](const std::vector<Point> &points, bool gather) {
    std::vector<TextureSampleRequest> values(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
      auto &r = values[i];
      r.shader_lane_index = i;
      r.request_id = i;
      r.quad_id = i / 4U;
      r.quad_lane = i % 4U;
      r.coordinates[0] = FloatBits(points[i][0]);
      r.coordinates[1] = FloatBits(points[i][1]);
      r.coordinate_count = 2;
      r.component_count = 4;
      r.dimension = 2;
      r.normalized = 1;
      r.explicit_lod_present = 1;
      r.gather = gather;
      if (array) {
        const auto address = base + (i % layers) * layer_stride;
        r.texture_address_lo = static_cast<std::uint32_t>(address);
        r.texture_address_hi = static_cast<std::uint32_t>(address >> 32U);
      }
      std::copy_n(shared.begin(), 4, r.texture_state);
      std::copy_n(shared.begin() + (gather ? 16 : 8), 4, r.sampler_state);
    }
    auto &r = values[0];
    if (mode == "reject-flag") r.gather = 2;
    if (mode == "reject-implicit") r.explicit_lod_present = 0;
    if (mode == "reject-lod") r.explicit_lod = FloatBits(1.0F);
    if (mode == "reject-negative-zero-lod") r.explicit_lod = 0x80000000U;
    if (mode == "reject-normalized") r.normalized = 0;
    if (mode == "reject-fcnorm") r.fcnorm = 0;
    if (mode == "reject-component") r.component_count = 1;
    if (mode == "reject-coordinate-count") r.coordinate_count = 3;
    if (mode == "reject-coordinate-z") r.coordinates[2] = FloatBits(1.0F);
    if (mode == "reject-offset") r.spatial_offsets[0] = 1;
    if (mode == "reject-address") r.texture_address_lo = 4;
    if (mode == "reject-array-unaligned") r.texture_address_lo += 4U;
    if (mode == "reject-array-below-base") r.texture_address_lo -= 4U;
    if (mode == "reject-array-past-last") r.texture_address_lo += layers * layer_stride;
    if (mode == "reject-array-missing-address") r.texture_address_lo = r.texture_address_hi = 0;
    if (mode == "reject-sno") r.sample_index_present = 1;
    if (mode == "reject-bias") r.lod_bias_present = 1;
    if (mode == "reject-stage") r.shader_stage = ShaderStage::kVertex;
    if (mode == "reject-nan") r.coordinates[0] = 0x7fc00000U;
    if (mode == "reject-infinity") r.coordinates[1] = 0x7f800000U;
    if (mode == "reject-sampler-offset")
      std::copy_n(shared.begin() + 8, 4, r.sampler_state);
    if (mode == "reject-image-state") r.texture_state[0] ^= 1U;
    if (mode == "reject-mixed-batch") {
      auto second = r;
      second.shader_lane_index = 1;
      second.request_id = 1;
      second.gather = 0;
      values.push_back(second);
    }
    if (HasPoolHandle(state.texture_sample_requests))
      pool.Release(state.texture_sample_requests);
    if (HasPoolHandle(state.texture_sample_responses)) {
      pool.Release(state.texture_sample_responses);
      state.texture_sample_responses = {};
    }
    const auto before = state.counters;
    state.texture_sample_requests = StoreNewArray(pool, values);
    state.fragment_shader_lane_count = values.size();
    state.stage = PipelineStage::kFragmentTexturePending;
    StorePipelineState(pool, handle, state);
    requests.write(txn);
    bool rejected = false;
    try { sc_core::sc_start(sc_core::sc_time(1, sc_core::SC_MS)); }
    catch (const std::exception &error) {
      if (!negative) throw;
      const std::string detail = error.what();
      const std::map<std::string, std::string> specific_causes = {
          {"reject-flag", "sample batch mixes shader stages, sets or bindings"},
          {"reject-stage", "sample batch mixes shader stages, sets or bindings"},
          {"reject-mixed-batch", "sample batch mixes shader stages, sets or bindings"},
          {"reject-red-swizzle", "unsupported raw Rogue image word0"},
          {"reject-rgba-red-swizzle", "unsupported raw Rogue image word0"},
          {"reject-gather-descriptor", "gather word0 does not match the sampler"},
          {"reject-gather-word1", "gather word1 is not zero"},
          {"reject-sampler-offset", "SMP descriptor state mismatch"},
          {"reject-image-state", "SMP descriptor state mismatch"},
          {"reject-nan", "invalid depth gather coordinate"},
          {"reject-infinity", "invalid depth gather coordinate"},
          {"reject-base", "structured state disagrees with raw descriptor"},
          {"reject-layers", "TEXTYPE/depth disagrees with layer metadata"},
          {"reject-repeat", "unsupported depth gather state"},
          {"reject-compare", "word12 compare operation is invalid"},
          {"reject-array-stride", "word4 is not the image layer size"},
          {"reject-array-depth", "TEXTYPE/depth disagrees with layer metadata"},
          {"reject-array-unaligned", "gather array address is not an exact valid layer"},
          {"reject-array-past-last", "gather array address is not an exact valid layer"},
          {"reject-array-below-base", "array sample address is out of range"},
          {"reject-array-missing-address", "array sample address is out of range"},
          {"reject-array-msaa", "unsupported depth gather state"},
          {"reject-array-mips", "unsupported depth gather state"},
          {"reject-format", "unsupported depth gather state"},
          {"reject-dimension", "unsupported depth gather state"},
      };
      const auto found = specific_causes.find(mode);
      const std::string cause = found == specific_causes.end()
          ? "unsupported depth gather request" : found->second;
      Check(detail.find("TextureUnit") != std::string::npos &&
            detail.find(cause) != std::string::npos,
            "negative case rejected at expected boundary: " + mode +
            " expected=" + cause + " actual=" + detail);
      rejected = true;
    }
    if (negative) {
      Check(rejected, "unsupported request/state did not fail closed: " + mode);
      return;
    }
    PipelineTxn completed;
    Check(responses.nb_read(completed) && completed.state.slot == handle.slot &&
          completed.state.generation == handle.generation && completed.sequence == txn.sequence,
          "response transaction identity");
    state = LoadPipelineState(pool, handle);
    const auto result = LoadArray<TextureSampleResponse>(pool, state.texture_sample_responses);
    Check(result.size() == points.size(), "one response per logical request");
    for (std::size_t i = 0; i < points.size(); ++i) {
      Check(result[i].request_id == i && result[i].shader_lane_index == i &&
            result[i].shader_stage == ShaderStage::kFragment, "response identity");
      const auto expected = Expected(points[i], width, height, gather, rgba_float, i % layers);
      for (std::size_t tap = 0; tap < 4; ++tap)
        Check(result[i].rgba[tap] == expected[tap], "raw depth tap bits at request " +
              std::to_string(i) + " tap " + std::to_string(tap));
    }
    const auto &after = state.counters;
    const std::uint64_t taps = points.size() * (gather ? 4U : 1U);
    Check(after.texture_requests - before.texture_requests == points.size() &&
          after.texel_fetches - before.texel_fetches == taps,
          "four actual texel reads per gather / one ordinary nearest read");
    if (memory_mode == MemoryMode::kDirect) {
      Check(after.memory_direct_read_bytes - before.memory_direct_read_bytes == taps * texel_bytes &&
            after.slc_line_accesses == 0 && after.dram_read_transactions == 0,
            "direct mode records each complete texel byte transfer");
    } else if (memory_mode == MemoryMode::kBypass) {
      Check(after.dram_read_bytes - before.dram_read_bytes == taps * texel_bytes &&
            after.dram_read_transactions - before.dram_read_transactions == taps &&
            after.slc_line_accesses == 0 && after.memory_direct_read_bytes == 0,
            "bypass mode records four separate complete texel DRAM transactions");
    } else {
      std::uint64_t misses = 0;
      for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
        const auto &point = points[point_index];
        const auto xs = ReferenceAxis(point[0], width, gather);
        const auto ys = ReferenceAxis(point[1], height, gather);
        for (unsigned tap = 0; tap < (gather ? 4U : 1U); ++tap) {
          const auto x = xs[tap & 1U];
          const auto y = ys[tap >> 1U];
          const auto line = (base + (point_index % layers) * layer_stride +
              (y * width + x) * texel_bytes) / SlcCacheConfig().line_size_bytes;
          if (resident_lines.insert(line).second) ++misses;
        }
      }
      Check(after.slc_read_accesses - before.slc_read_accesses == taps &&
            after.slc_line_accesses - before.slc_line_accesses == taps &&
            after.slc_misses - before.slc_misses == misses &&
            after.slc_hits - before.slc_hits == taps - misses &&
            after.dram_read_transactions - before.dram_read_transactions == misses &&
            after.dram_read_bytes - before.dram_read_bytes == misses * SlcCacheConfig().line_size_bytes &&
            after.memory_direct_read_bytes == 0,
            "exact cache-line footprint and DRAM traffic across complete texels");
    }
    Check(state.fragment_texture_request_count == after.texture_requests &&
          state.fragment_texel_fetch_count == after.texel_fetches,
          "fragment counters preserve logical versus physical traffic");
  };

  // Four duplicated taps are four cache lookups, including in the warm pass.
  batch({{{-1.0F, -1.0F}}}, true);
  if (negative) return;
  Check(memory_mode != MemoryMode::kCache ||
        (state.counters.slc_misses == 1 && state.counters.slc_hits == 3 &&
        state.counters.dram_read_transactions == 1 &&
        state.counters.dram_read_bytes == SlcCacheConfig().line_size_bytes),
        "cold duplicate taps: one DRAM cache-line fill, three real hits");
  batch({{{-1.0F, -1.0F}}}, true);
  Check(memory_mode != MemoryMode::kCache ||
        (state.counters.slc_misses == 1 && state.counters.slc_hits == 7 &&
        state.counters.dram_read_transactions == 1),
        "warm duplicate taps: four hits, no DRAM read");
  const std::vector<Point> points = {
      {{0.5F, 0.5F}}, {{0.0F, 0.5F}}, {{1.0F, 0.5F}},
      {{0.5F, 0.0F}}, {{0.5F, 1.0F}}, {{0.0F, 0.0F}},
      {{1.0F, 0.0F}}, {{0.0F, 1.0F}}, {{1.0F, 1.0F}},
      {{-0.25F, 0.5F}}, {{1.25F, 0.5F}}, {{0.5F, -0.25F}}, {{0.5F, 1.25F}},
      {{0.5F / width, 0.5F / height}},
      {{1.0F - 0.5F / width, 1.0F - 0.5F / height}},
      {{0.375F, 0.375F}}, {{std::nextafter(0.375F, 0.0F), 0.375F}},
      {{std::nextafter(0.375F, 1.0F), 0.375F}},
      {{std::nextafter(0.125F, 0.0F), 0.5F}},
      {{0.5F, std::nextafter(0.125F, 0.0F)}},
      {{std::nextafter(0.125F, 0.0F), std::nextafter(0.125F, 0.0F)}},
      {{std::nextafter(0.125F, 1.0F), 0.5F}},
      {{0.5F, std::nextafter(0.125F, 1.0F)}},
      {{std::nextafter(0.875F, 0.0F), 0.5F}},
      {{0.5F, std::nextafter(0.875F, 0.0F)}},
      {{std::nextafter(0.875F, 1.0F), 0.5F}},
      {{0.5F, std::nextafter(0.875F, 1.0F)}},
  };
  batch(points, true);
  const auto reads = state.counters.dram_read_transactions;
  batch(points, true);
  Check(memory_mode != MemoryMode::kCache || state.counters.dram_read_transactions == reads,
        "warm full footprint matrix issues no new DRAM reads");
  if (!min_linear && !mag_linear && configuration != "lod-window")
    batch(points, false);  // Same descriptor binding returns ordinary RGBA.
  ReleaseFunctionalPayloads(pool, state);
  pool.Release(handle);
  Check(pool.bytes_in_flight() == 0 && pool.allocations() == pool.releases(),
        "balanced payload ownership");
}
}  // namespace

int sc_main(int argc, char **argv) {
  try {
    CheckAxis();
    Run(argc > 1 ? argv[1] : "default");
    std::cout << "texture_gather_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "texture_gather_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
