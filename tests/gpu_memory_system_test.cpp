#include "memory/gpu_memory_system.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <tuple>
#include <typeinfo>
#include <vector>

// Observe only tightly scoped, already-warm reads. Setup and assertion strings
// are outside the count, so this proves the new hit path avoids heap payloads.
namespace {
bool g_count_allocations = false;
std::size_t g_allocations = 0;
}
void *operator new(std::size_t bytes) {
  if (g_count_allocations) ++g_allocations;
  if (void *memory = std::malloc(bytes ? bytes : 1)) return memory;
  throw std::bad_alloc();
}
void *operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
#if defined(__cpp_sized_deallocation)
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept { std::free(memory); }
#endif

namespace {

using pvrgpu::stub::GpuMemorySystem;
using pvrgpu::stub::MemoryClient;
using pvrgpu::stub::MemoryMode;
using pvrgpu::stub::MemoryAccessStats;

auto Stats(const MemoryAccessStats &s) {
  return std::make_tuple(s.slc.line_accesses, s.slc.read_accesses,
      s.slc.write_accesses, s.slc.hits, s.slc.misses, s.slc.evictions,
      s.slc.writebacks, s.slc.bypassed, s.slc_cycles,
      s.dram_read_transactions, s.dram_write_transactions, s.dram_read_bytes,
      s.dram_write_bytes, s.dram_cycles, s.direct_read_bytes,
      s.direct_write_bytes);
}

void Check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error("gpu memory test failed: " + message);
}

std::vector<std::uint8_t> Pattern(std::size_t bytes, std::uint8_t seed) {
  std::vector<std::uint8_t> result(bytes);
  for (std::size_t index = 0; index < result.size(); ++index)
    result[index] = static_cast<std::uint8_t>(seed + index * 17U);
  return result;
}

void TestDirect() {
  constexpr std::uint64_t kAddress = 0x10000040ULL;
  GpuMemorySystem memory(MemoryMode::kDirect);
  const auto initial = Pattern(300, 3);
  memory.HostWrite(kAddress, initial.data(), initial.size());
  const auto read = memory.Read(kAddress, initial.size(),
                                MemoryClient::kVertexFetch);
  Check(read.data == initial, "direct read is byte exact");
  Check(read.stats.direct_read_bytes == initial.size() &&
            read.stats.slc.line_accesses == 0 &&
            read.stats.dram_read_transactions == 0 &&
            read.stats.slc_cycles == 0 && read.stats.dram_cycles == 0,
        "direct read has functional-only provenance");

  const auto replacement = Pattern(91, 0xa0);
  const auto write = memory.Write(kAddress + 37, replacement.data(),
                                  replacement.size(),
                                  MemoryClient::kParameterWrite);
  Check(write.direct_write_bytes == replacement.size() &&
            write.slc.line_accesses == 0 &&
            write.dram_write_transactions == 0,
        "direct write has no simulated cache/DRAM traffic");
  const auto updated = memory.backing().Read(kAddress, initial.size());
  Check(std::equal(replacement.begin(), replacement.end(),
                   updated.begin() + 37),
        "direct write updates authoritative backing");
}

void TestBypass() {
  constexpr std::uint64_t kAddress = 0x20000000ULL;
  GpuMemorySystem memory(MemoryMode::kBypass);
  const auto initial = Pattern(257, 9);
  memory.HostWrite(kAddress, initial.data(), initial.size());
  const auto read = memory.Read(kAddress + 5, 129,
                                MemoryClient::kIndexFetch);
  Check(std::equal(read.data.begin(), read.data.end(), initial.begin() + 5),
        "bypass read is byte exact");
  Check(read.stats.slc.bypassed == 1 &&
            read.stats.slc.line_accesses == 0 &&
            read.stats.dram_read_transactions == 1 &&
            read.stats.dram_read_bytes == 129 &&
            read.stats.dram_cycles == 1,
        "bypass read retains one DRAM transaction");

  const auto replacement = Pattern(64, 0x33);
  const auto write = memory.Write(kAddress + 128, replacement.data(),
                                  replacement.size(),
                                  MemoryClient::kFramebuffer);
  Check(write.slc.bypassed == 1 &&
            write.dram_write_transactions == 1 &&
            write.dram_write_bytes == replacement.size(),
        "bypass write reaches DRAM without allocating SLC");
  Check(memory.backing().Read(kAddress + 128, replacement.size()) ==
            replacement,
        "bypass write committed authoritative backing");
}

void TestCacheAndFlush() {
  constexpr std::uint64_t kReadAddress = 0x30000000ULL;
  constexpr std::uint64_t kWriteAddress = 0x31000000ULL;
  GpuMemorySystem memory(MemoryMode::kCache);
  const auto initial = Pattern(256, 0x15);
  memory.HostWrite(kReadAddress, initial.data(), initial.size());

  const auto cold = memory.Read(kReadAddress + 16, 32,
                                MemoryClient::kVertexFetch);
  Check(std::equal(cold.data.begin(), cold.data.end(), initial.begin() + 16),
        "cache cold read is byte exact");
  Check(cold.stats.slc.line_accesses == 1 &&
            cold.stats.slc.misses == 1 && cold.stats.slc.hits == 0 &&
            cold.stats.dram_read_transactions == 1 &&
            cold.stats.dram_read_bytes == 128,
        "cache cold read fills one SLC line from DRAM");

  const auto warm = memory.Read(kReadAddress + 32, 16,
                                MemoryClient::kParameterRead);
  Check(warm.stats.slc.line_accesses == 1 && warm.stats.slc.hits == 1 &&
            warm.stats.dram_read_transactions == 0,
        "cache warm read hits shared SLC across clients");

  const auto framebuffer = Pattern(128, 0x80);
  const auto write = memory.Write(kWriteAddress, framebuffer.data(),
                                  framebuffer.size(),
                                  MemoryClient::kFramebuffer);
  Check(write.slc.line_accesses == 1 && write.slc.write_accesses == 1 &&
            write.slc.misses == 1 &&
            write.dram_write_transactions == 0 &&
            !memory.backing().Contains(kWriteAddress, framebuffer.size()),
        "cache write allocates one dirty line without early DRAM write");

  const auto resident = memory.Read(kWriteAddress, framebuffer.size(),
                                    MemoryClient::kFramebufferReadback);
  Check(resident.data == framebuffer && resident.stats.slc.hits == 1 &&
            resident.stats.dram_read_transactions == 0,
        "dirty SLC line is visible before writeback");

  const auto flush = memory.Flush();
  Check(flush.slc.writebacks == 1 &&
            flush.dram_write_transactions == 1 &&
            flush.dram_write_bytes == 128,
        "flush writes one dirty SLC line to DRAM");
  Check(memory.backing().Read(kWriteAddress, framebuffer.size()) == framebuffer,
        "flushed DRAM framebuffer is byte exact");
  const auto second_flush = memory.Flush();
  Check(second_flush.slc.writebacks == 0 &&
            second_flush.dram_write_transactions == 0,
        "second flush does not rewrite clean lines");
}

void TestFramebufferReadback() {
  constexpr std::uint64_t kAddress = 0x40000000ULL;
  GpuMemorySystem memory(MemoryMode::kCache);
  const auto framebuffer = Pattern(192, 0x55);
  const auto write = memory.Write(kAddress, framebuffer.data(),
                                  framebuffer.size(),
                                  MemoryClient::kFramebuffer);
  Check(write.slc.write_accesses == 2 &&
            write.dram_write_transactions == 0 &&
            !memory.backing().Contains(kAddress, framebuffer.size()),
        "cache framebuffer write stays dirty before readback");

  const auto readback = memory.Readback(kAddress, framebuffer.size(),
                                        MemoryClient::kFramebufferReadback);
  Check(readback.data == framebuffer,
        "framebuffer readback is byte exact after SLC flush");
  Check(readback.stats.slc.writebacks == 2 &&
            readback.stats.dram_write_transactions == 2 &&
            readback.stats.dram_read_transactions == 1 &&
            readback.stats.dram_read_bytes == framebuffer.size(),
        "framebuffer readback flushes dirty lines then reads DRAM backing");
  Check(memory.backing().Read(kAddress, framebuffer.size()) == framebuffer,
        "readback leaves authoritative backing populated");
}

void TestFreshPartialStores(MemoryMode mode) {
  constexpr std::uint64_t kAddress = UINT64_C(0x800010000);
  GpuMemorySystem memory(mode);
  std::vector<std::uint8_t> expected(256, 0);
  auto store = [&](std::size_t offset, std::size_t bytes, std::uint8_t seed,
                   MemoryClient client) {
    const auto replacement = Pattern(bytes, seed);
    const auto stats = memory.Write(kAddress + offset, replacement.data(),
                                    replacement.size(), client);
    std::copy(replacement.begin(), replacement.end(), expected.begin() + offset);
    if (mode == MemoryMode::kCache) {
      Check(stats.dram_read_transactions == 0 &&
                stats.dram_write_transactions == 0 &&
                !memory.backing().Contains(kAddress, expected.size()),
            "fresh dirty-line partial updates do not read or initialize DRAM");
    }
  };
  // No HostWrite: fresh output storage begins with a full dirty line, then
  // another client overwrites only an interior span of that resident line.
  store(0, 128, 0x15, MemoryClient::kTessellationControl);
  store(37, 19, 0xb7, MemoryClient::kTessellator);
  auto read = memory.Read(kAddress, 128, MemoryClient::kTessellationEvaluation);
  Check(std::equal(read.data.begin(), read.data.end(), expected.begin()),
        "partial store preserves untouched bytes in fresh full dirty line");
  if (mode == MemoryMode::kCache)
    Check(read.stats.slc.hits == 1 && read.stats.dram_read_transactions == 0,
          "fresh dirty line is shared across all tessellation clients");

  // Two tightly packed patch outputs can share a line. The second line has
  // only partial stores and is still absent from backing when overwritten.
  store(120, 40, 0x6a, MemoryClient::kTessellator);
  store(145, 31, 0xce, MemoryClient::kTessellationControl);
  store(200, 4, 0x28, MemoryClient::kGeometryShader);
  read = memory.Read(kAddress, expected.size(), MemoryClient::kTessellationEvaluation);
  Check(read.data == expected,
        "cross-line and disjoint partial stores preserve every other byte");
  if (mode == MemoryMode::kCache)
    Check(read.stats.slc.hits == 2 && read.stats.dram_read_transactions == 0,
          "both fresh partial output lines remain resident and coherent");
  const auto final = memory.Readback(kAddress, expected.size(), MemoryClient::kFramebufferReadback);
  Check(final.data == expected && memory.backing().Read(kAddress, expected.size()) == expected,
        "partial-store merge survives flush and authoritative readback");
  if (mode == MemoryMode::kCache)
    Check(final.stats.slc.writebacks == 2 && final.stats.dram_write_transactions == 2,
          "fresh partial stores retire as exactly two dirty line writebacks");

  bool rejected = false;
  try { (void)memory.Read(kAddress + 0x10000, 4, MemoryClient::kVertexFetch); }
  catch (const std::runtime_error &) { rejected = true; }
  Check(rejected, "partial-store initialization does not legalize uninitialized reads");
}

void TestReadIntoTrace(MemoryMode mode) {
  constexpr std::uint64_t base = UINT64_C(0x900000000);
  GpuMemorySystem legacy(mode), into(mode);
  auto expected = Pattern(8192, 17);
  legacy.HostWrite(base, expected.data(), expected.size());
  into.HostWrite(base, expected.data(), expected.size());
  const auto read = [&](std::size_t offset, std::size_t bytes,
                        MemoryClient client) {
    const auto old = legacy.Read(base + offset, bytes, client);
    std::vector<std::uint8_t> guarded(bytes + 32, 0xa7);
    const auto fresh = into.ReadInto(base + offset, guarded.data() + 16,
                                     bytes, client);
    Check(Stats(old.stats) == Stats(fresh), "ReadInto matches every read counter");
    Check(std::equal(old.data.begin(), old.data.end(), guarded.begin() + 16) &&
              std::equal(old.data.begin(), old.data.end(), expected.begin() + offset),
          "ReadInto and legacy read match independent updated bytes");
    Check(std::all_of(guarded.begin(), guarded.begin() + 16,
                     [](auto v) { return v == 0xa7; }) &&
              std::all_of(guarded.end() - 16, guarded.end(),
                          [](auto v) { return v == 0xa7; }),
          "ReadInto writes exactly the caller span");
  };
  // All offsets, small texels and whole/cross-line generic payloads. Repeat
  // immediately to cover warm reads without assuming all clients use 16B.
  for (std::size_t offset = 0; offset < 256; ++offset)
    for (const std::size_t bytes : {1U, 4U, 8U, 16U, 127U, 128U, 129U, 257U}) {
      read(offset, bytes, MemoryClient::kTextureCache);
      read(offset, bytes, MemoryClient::kUniformBuffer);
    }
  std::uint32_t random = 0x715af149;
  for (unsigned step = 0; step < 512; ++step) {
    random = random * 1664525U + 1013904223U;
    const std::size_t bytes = 1U + ((random >> 12) % 257U);
    const std::size_t offset = (random >> 3) % (expected.size() - bytes);
    if ((random & 7U) <= 1U) {
      const auto changed = Pattern(bytes, static_cast<std::uint8_t>(random >> 24));
      if (random & 1U) {
        legacy.HostWrite(base + offset, changed.data(), bytes);
        into.HostWrite(base + offset, changed.data(), bytes);
      } else {
        const auto old = legacy.Write(base + offset, changed.data(), bytes,
                                       MemoryClient::kFramebuffer);
        const auto fresh = into.Write(base + offset, changed.data(), bytes,
                                       MemoryClient::kFramebuffer);
        Check(Stats(old) == Stats(fresh), "ReadInto preserves later GPU store stats");
      }
      std::copy(changed.begin(), changed.end(), expected.begin() + offset);
    }
    read(offset, bytes, MemoryClient::kComputeShader);
  }
  Check(Stats(legacy.Flush()) == Stats(into.Flush()), "ReadInto preserves final flush");
  Check(legacy.backing().Read(base, expected.size()) == expected &&
            into.backing().Read(base, expected.size()) == expected,
        "ReadInto preserves authoritative bytes after mixed host/GPU mutation");
}

void TestReadIntoDirtyEviction() {
  constexpr std::uint64_t base = UINT64_C(0xa00000000);
  constexpr std::uint64_t stride = 256U * 8U * 128U;
  GpuMemorySystem legacy(MemoryMode::kCache), into(MemoryMode::kCache);
  for (unsigned tag = 0; tag < 10; ++tag) {
    const auto data = Pattern(128, static_cast<std::uint8_t>(tag * 13));
    Check(Stats(legacy.Write(base + tag * stride, data.data(), data.size(),
                             MemoryClient::kFramebuffer)) ==
              Stats(into.Write(base + tag * stride, data.data(), data.size(),
                                MemoryClient::kFramebuffer)),
          "dirty conflict setup and writebacks match");
    std::array<std::uint8_t, 16> bytes{};
    const auto old = legacy.Read(base + tag * stride + 7, bytes.size(),
                                 MemoryClient::kTextureCache);
    const auto fresh = into.ReadInto(base + tag * stride + 7, bytes.data(),
                                     bytes.size(), MemoryClient::kTextureCache);
    Check(Stats(old.stats) == Stats(fresh) &&
              std::equal(bytes.begin(), bytes.end(), old.data.begin()),
          "dirty resident small read is exact");
  }
  // A miss now evicts a dirty victim and must perform the same lower read
  // before writeback. Subsequent accesses expose any changed LRU choice.
  for (unsigned tag = 0; tag < 10; ++tag) {
    std::array<std::uint8_t, 16> bytes{};
    const auto old = legacy.Read(base + tag * stride + 19, bytes.size(),
                                 MemoryClient::kTextureCache);
    const auto fresh = into.ReadInto(base + tag * stride + 19, bytes.data(),
                                     bytes.size(), MemoryClient::kTextureCache);
    Check(Stats(old.stats) == Stats(fresh) &&
              std::equal(bytes.begin(), bytes.end(), old.data.begin()),
          "cold conflict reads preserve LRU, dirty eviction and memory stats");
  }
  Check(Stats(legacy.Flush()) == Stats(into.Flush()), "conflict final flush matches");
  for (unsigned tag = 0; tag < 10; ++tag)
    Check(legacy.backing().Read(base + tag * stride, 128) ==
              into.backing().Read(base + tag * stride, 128),
          "conflict dirty writebacks preserve bytes");
}

std::string Failure(const std::function<void()> &call) {
  try { call(); }
  catch (const std::exception &error) {
    return std::string(typeid(error).name()) + ':' + error.what();
  }
  throw std::runtime_error("expected read failure");
}

void TestReadIntoFailures(MemoryMode mode) {
  constexpr std::uint64_t base = UINT64_C(0xb00000000);
  GpuMemorySystem legacy(mode), into(mode);
  const auto initial = Pattern(128, 93);
  legacy.HostWrite(base, initial.data(), initial.size());
  into.HostWrite(base, initial.data(), initial.size());
  std::array<std::uint8_t, 64> destination{};
  const auto same_failure = [&](std::uint64_t address, std::size_t bytes,
                                MemoryClient client) {
    const auto before = destination;
    const auto old = Failure([&] { (void)legacy.Read(address, bytes, client); });
    const auto fresh = Failure([&] {
      (void)into.ReadInto(address, destination.data() + 16, bytes, client);
    });
    Check(old == fresh, "ReadInto preserves exception type and message");
    Check(std::equal(destination.begin(), destination.begin() + 16, before.begin()) &&
              std::equal(destination.end() - 16, destination.end(), before.end() - 16),
          "failed ReadInto keeps destination guards");
    Check(Stats(legacy.Flush()) == Stats(into.Flush()), "failed reads preserve flush state");
  };
  same_failure(base, 0, MemoryClient::kTextureCache);
  same_failure(base, 4, MemoryClient::kTextureUpload);
  same_failure(base, 0, MemoryClient::kTextureUpload);
  same_failure(base + 0x10000, 16, MemoryClient::kTextureCache);
  same_failure(base + 120, 16, MemoryClient::kTextureCache);
  same_failure(UINT64_MAX - 3, 8, MemoryClient::kTextureCache);
  const auto null_error = Failure([&] {
    (void)into.ReadInto(base, nullptr, 4, MemoryClient::kTextureCache);
  });
  Check(null_error.find("destination is null") != std::string::npos,
        "ReadInto refuses null destination");
  Check(Failure([&] { (void)into.ReadInto(base, nullptr, 0,
                                        MemoryClient::kTextureUpload); }) ==
            Failure([&] { (void)legacy.Read(base, 0, MemoryClient::kTextureUpload); }),
        "client validation precedes new destination checks");
  const auto old = legacy.Read(base, 16, MemoryClient::kTextureCache);
  const auto fresh = into.ReadInto(base, destination.data(), 16,
                                   MemoryClient::kTextureCache);
  Check(Stats(old.stats) == Stats(fresh) &&
            std::equal(old.data.begin(), old.data.end(), destination.begin()),
        "successful access after failures has identical state and data");
}

void TestReadIntoWarmAllocations() {
  constexpr std::uint64_t base = UINT64_C(0xc00000000);
  GpuMemorySystem memory(MemoryMode::kCache);
  const auto bytes = Pattern(256, 37);
  memory.HostWrite(base, bytes.data(), bytes.size());
  (void)memory.Read(base, bytes.size(), MemoryClient::kTextureCache);
  std::array<std::uint8_t, 16> destination{};
  MemoryAccessStats accumulated;
  g_allocations = 0;
  g_count_allocations = true;
  for (unsigned repeat = 0; repeat < 256; ++repeat)
    accumulated += memory.ReadInto(base + (repeat % 129), destination.data(),
                                    destination.size(), MemoryClient::kTextureCache);
  g_count_allocations = false;
  Check(g_allocations == 0 && accumulated.slc.misses == 0 &&
            accumulated.slc.hits > 256,
        "warm ReadInto, including cross-line texels, allocates no heap payload");
  g_allocations = 0;
  g_count_allocations = true;
  const auto old = memory.Read(base, 16, MemoryClient::kTextureCache);
  g_count_allocations = false;
  Check(g_allocations >= 2 && old.data.size() == 16,
        "allocation observer sees legacy result and whole-line response allocations");
}

}  // namespace

int main() {
  try {
    TestDirect();
    TestBypass();
    TestCacheAndFlush();
    TestFramebufferReadback();
    TestFreshPartialStores(MemoryMode::kDirect);
    TestFreshPartialStores(MemoryMode::kBypass);
    TestFreshPartialStores(MemoryMode::kCache);
    TestReadIntoTrace(MemoryMode::kDirect);
    TestReadIntoTrace(MemoryMode::kBypass);
    TestReadIntoTrace(MemoryMode::kCache);
    TestReadIntoDirtyEviction();
    TestReadIntoFailures(MemoryMode::kDirect);
    TestReadIntoFailures(MemoryMode::kBypass);
    TestReadIntoFailures(MemoryMode::kCache);
    TestReadIntoWarmAllocations();
    std::cout << "gpu_memory_system_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "gpu_memory_system_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
