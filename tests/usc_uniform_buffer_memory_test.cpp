#include "shader/usc_uniform_buffer_memory.h"

#include <array>
#include <iostream>
#include <string>

namespace {
using namespace pvrgpu::stub;

void Check(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(std::string("USC UBO memory test: ") + message);
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

void TestMode(MemoryMode mode) {
  // A nonzero high DWORD catches truncated descriptor addresses. Adjacent
  // ranges share a DRAM page, so a page-presence check cannot pass these tests.
  constexpr std::uint64_t address = UINT64_C(0x12345678000);
  const std::array<std::uint32_t, 8> words{
      0x3f800000, 0x80000000, 0xdeadbeef, 0x40000000,
      0x40400000, 0x40800000, 0x40a00000, 0x40c00000};
  GpuMemorySystem memory(mode);
  memory.HostWrite(address, words.data(), sizeof(words));
  UscUniformBufferMemory vertex(&memory, mode, {{address, 16, 0, 0}});
  UscUniformBufferMemory fragment(&memory, mode, {{address + 16, 16, 0, 0}});
  UscUniformBufferMemory adjacent(
      &memory, mode, {{address, 16, 0, 0}, {address + 16, 16, 1, 4}});
  std::array<std::uint32_t, 4> output{};
  UscUniformBufferMemory::Read(&vertex, address, 4, output.data());
  Check(std::equal(output.begin(), output.end(), words.begin()),
        "four raw DWORDs preserve floating/integer bits");
  UscUniformBufferMemory::Read(&vertex, address + 12, 1, output.data());
  Check(output[0] == words[3], "last bound DWORD is readable");
  UscUniformBufferMemory::Read(&fragment, address + 16, 4, output.data());
  Check(std::equal(output.begin(), output.end(), words.begin() + 4),
        "fragment range is independent");
  Reject([&] { UscUniformBufferMemory::Read(&vertex, address + 16, 1, output.data()); },
         "another stage's same-page memory is forbidden");
  Reject([&] { UscUniformBufferMemory::Read(&fragment, address, 1, output.data()); },
         "preceding stage range is forbidden");
  Reject([&] { UscUniformBufferMemory::Read(&adjacent, address + 12, 2, output.data()); },
         "one LD cannot cross adjacent block ranges");
  Reject([&] { UscUniformBufferMemory::Read(&vertex, address + 1, 1, output.data()); },
         "unaligned LD is rejected");
  Reject([&] { UscUniformBufferMemory::Read(&vertex, address, 0, output.data()); },
         "empty LD is rejected");
  Reject([&] { UscUniformBufferMemory::Read(&vertex, address, 17, output.data()); },
         "oversized LD is rejected");
  Reject([&] { UscUniformBufferMemory::Read(&vertex, address, 1, nullptr); },
         "missing response buffer is rejected");
  Reject([&] { UscUniformBufferMemory::Read(nullptr, address, 1, output.data()); },
         "missing context is rejected");
  Reject([&] { UscUniformBufferMemory invalid(nullptr, mode, {{address, 16, 0, 0}}); },
         "nonempty resources require memory service");
  const MemoryMode wrong_mode = mode == MemoryMode::kCache
      ? MemoryMode::kDirect : MemoryMode::kCache;
  Reject([&] { UscUniformBufferMemory invalid(&memory, wrong_mode, {{address, 16, 0, 0}}); },
         "memory mode mismatch is rejected");
  Reject([&] { UscUniformBufferMemory invalid(&memory, mode, {{address, 16, 0, 0}, {address + 16, 16, 0, 4}}); },
         "duplicate block index is rejected");
  Reject([&] { UscUniformBufferMemory invalid(&memory, mode, {{address, 20, 0, 0}, {address + 16, 16, 1, 4}}); },
         "overlapping ranges are rejected");
  Reject([&] { UscUniformBufferMemory invalid(&memory, mode, {{UINT64_MAX - 3, 16, 0, 0}}); },
         "resource address overflow is rejected");
  UscUniformBufferMemory empty(&memory, mode, {});
  Reject([&] { UscUniformBufferMemory::Read(&empty, address, 1, output.data()); },
         "unbound slot cannot read allocated DRAM");
  Check(empty.stats().dram_read_transactions == 0, "rejected LD does not issue traffic");
  const auto &stats = vertex.stats();
  if (mode == MemoryMode::kDirect) {
    Check(stats.direct_read_bytes == 20 && MemoryAccessDelayCycles(stats) == 0,
          "direct mode retains bytes without modeled latency");
  } else if (mode == MemoryMode::kBypass) {
    Check(stats.dram_read_transactions == 2 && stats.dram_read_bytes == 20 &&
              MemoryAccessDelayCycles(stats) == 2,
          "bypass mode counts each actual LD");
  } else {
    Check(stats.slc.read_accesses == 2 && stats.slc.misses == 1 &&
              stats.slc.hits == 1 && stats.dram_read_transactions == 1 &&
              MemoryAccessDelayCycles(stats) == 3,
          "cache mode has real cold miss and warm hit");
  }
  CounterTxn counters;
  ApplyMemoryAccessStats(counters, stats);
  Check(counters.dram_read_transactions == stats.dram_read_transactions &&
            counters.memory_direct_read_bytes == stats.direct_read_bytes,
        "LD traffic reaches public counters");

  const std::array<std::uint32_t, 16> wide_words{
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0xdeadbeef};
  constexpr std::uint64_t wide_address = address + 128;
  memory.HostWrite(wide_address, wide_words.data(), sizeof(wide_words));
  UscUniformBufferMemory wide(&memory, mode, {{wide_address, 64, 0, 0}});
  std::array<std::uint32_t, 16> wide_output{};
  for (std::uint32_t count = 1; count <= 16; ++count) {
    wide_output.fill(0);
    UscUniformBufferMemory::Read(&wide, wide_address, count, wide_output.data());
    Check(std::equal(wide_output.begin(), wide_output.begin() + count,
                     wide_words.begin()), "all native burst lengths preserve data");
    if (count < 16)
      Check(wide_output[count] == 0, "LD writes only its declared response span");
  }
  Reject([&] { UscUniformBufferMemory::Read(&wide, wide_address + 4, 16, wide_output.data()); },
         "16-DWORD load still obeys the exact bound range");
}
} // namespace

int main() {
  try {
    for (auto mode : {MemoryMode::kDirect, MemoryMode::kBypass, MemoryMode::kCache})
      TestMode(mode);
    std::cout << "USC UBO memory tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
