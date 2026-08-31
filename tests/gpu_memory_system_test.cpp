#include "memory/gpu_memory_system.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using pvrgpu::stub::GpuMemorySystem;
using pvrgpu::stub::MemoryClient;
using pvrgpu::stub::MemoryMode;

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

}  // namespace

int main() {
  try {
    TestDirect();
    TestBypass();
    TestCacheAndFlush();
    TestFramebufferReadback();
    std::cout << "gpu_memory_system_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "gpu_memory_system_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
