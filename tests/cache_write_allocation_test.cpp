// SPDX-License-Identifier: MIT
// Host-allocation optimization regression. GPU bytes, all counters, replacement
// decisions and lower-callback ordering remain identical to the v5 baseline.
#include "memory/gpu_memory_system.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
std::uint64_t allocations = 0;
bool count_allocations = false;
}
void *operator new(std::size_t bytes) {
  if (count_allocations) ++allocations;
  if (void *pointer = std::malloc(bytes ? bytes : 1)) return pointer;
  throw std::bad_alloc();
}
void *operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void *pointer) noexcept { std::free(pointer); }
void operator delete[](void *pointer) noexcept { std::free(pointer); }

namespace {
using namespace pvrgpu::stub;
unsigned checks = 0;
void Check(bool value, const std::string &label) {
  ++checks;
  if (!value) throw std::runtime_error(label);
}
std::array<std::uint64_t, 8> Fields(const CacheStats &s) {
  return {s.line_accesses,s.read_accesses,s.write_accesses,s.hits,s.misses,
          s.evictions,s.writebacks,s.bypassed};
}
std::array<std::uint64_t, 16> Fields(const MemoryAccessStats &s) {
  return {s.slc.line_accesses,s.slc.read_accesses,s.slc.write_accesses,s.slc.hits,
          s.slc.misses,s.slc.evictions,s.slc.writebacks,s.slc.bypassed,
          s.slc_cycles,s.dram_read_transactions,s.dram_write_transactions,
          s.dram_read_bytes,s.dram_write_bytes,s.dram_cycles,s.direct_read_bytes,
          s.direct_write_bytes};
}
CacheLineAccess Store(CacheArray &cache, std::uint64_t address,
                      const CacheLineData &bytes, const CacheLineRead &read,
                      const CacheLineWrite &write, bool return_data) {
#ifdef PVRGPU_CACHE_WRITE_BASELINE
  auto result = cache.WriteLine(address, bytes, read, write);
  if (!return_data) result.data.clear();
  return result;
#else
  return cache.WriteLine(address, bytes, read, write, return_data);
#endif
}
void TestCacheEquivalence() {
  using Event = std::pair<std::uint64_t, CacheLineData>;
  for (bool bypass : {false, true}) {
    CacheArray retained({"retained",256,16,2,2}, bypass);
    CacheArray omitted({"omitted",256,16,2,2}, bypass);
    std::array<std::vector<std::uint8_t>,2> backing;
    std::array<std::vector<Event>,2> events;
    std::array<CacheLineRead,2> read;
    std::array<CacheLineWrite,2> write;
    for (unsigned i = 0; i < 2; ++i) {
      backing[i].resize(4096);
      for (unsigned j = 0; j < 4096; ++j) backing[i][j] = j * 17U + 3U;
      read[i] = [&,i](std::uint64_t address, std::size_t bytes) {
        CacheLineData result(backing[i].begin() + address, backing[i].begin() + address + bytes);
        events[i].emplace_back(address | (UINT64_C(1) << 63), result);
        return result;
      };
      write[i] = [&,i](std::uint64_t address, const CacheLineData &bytes) {
        events[i].emplace_back(address, bytes);
        std::copy(bytes.begin(), bytes.end(), backing[i].begin() + address);
      };
    }
    std::uint32_t random = 0x73581024;
    for (unsigned step = 0; step < 700; ++step) {
      random = random * 1664525U + 1013904223U;
      const auto address = std::uint64_t((random >> 16) % 256) * 16;
      const bool is_write = step % 4 != 0;
      CacheLineData data(16, static_cast<std::uint8_t>(step));
      const auto a = is_write ? Store(retained,address,data,read[0],write[0],true)
                              : retained.ReadLine(address,read[0],write[0]);
      const auto b = is_write ? Store(omitted,address,data,read[1],write[1],false)
                              : omitted.ReadLine(address,read[1],write[1]);
      Check(a.line_address == b.line_address && a.bank == b.bank && a.set == b.set &&
            a.way == b.way && a.hit == b.hit && a.bypassed == b.bypassed &&
            Fields(a.delta) == Fields(b.delta), "write response policy changed cache access/LRU/counters");
      Check(is_write ? a.data == data && b.data.empty() : a.data == b.data,
            "write response policy changed read/write data contract");
      if (step % 71 == 0) {
        Check(retained.Flush(write[0]) == omitted.Flush(write[1]), "flush count changed");
        const bool next_bypass = step % 142 == 0;
        Check(retained.SetBypass(next_bypass,write[0]) == omitted.SetBypass(next_bypass,write[1]),
              "bypass transition writebacks changed");
      }
      Check(events[0] == events[1] && backing[0] == backing[1] &&
            Fields(retained.stats()) == Fields(omitted.stats()),
            "lower traffic, payload, ordering or cumulative cache counters changed");
    }
  }
  for (bool return_data : {false,true}) {
    CacheArray cache({"throw",32,16,2,1});
    const CacheLineData old(16,0xa5), replacement(16,0x5a);
    Store(cache,0,old,{}, {},return_data);
    Store(cache,16,old,{}, {},return_data);
    std::uint64_t victim = UINT64_MAX;
    const CacheLineWrite fail = [&](std::uint64_t address, const CacheLineData &bytes) {
      victim = address;
      Check(bytes == old, "throwing callback must receive original dirty victim bytes");
      throw std::runtime_error("lower write failure");
    };
    bool threw = false;
    const auto before = cache.stats();
    try { Store(cache,32,replacement,{},fail,return_data); }
    catch (const std::runtime_error &error) { threw = std::string(error.what()) == "lower write failure"; }
    const auto delta = cache.stats() - before;
    Check(threw && victim == 0 && delta.line_accesses == 1 && delta.misses == 1 &&
          delta.evictions == 1 && delta.writebacks == 0, "failed write changed exception/counter order");
    // Retry before any read touch: the exact same LRU victim must be selected.
    Store(cache,32,replacement,{},[&](auto address,const auto &bytes) {
      Check(address == 0 && bytes == old, "failed write mutated cache tag/data/LRU");
    },return_data);
    Check(cache.ReadLine(32).data == replacement && cache.ReadLine(16).data == old,
          "retry after callback failure did not preserve resident lines");
    const auto stable = Fields(cache.stats());
    for (unsigned invalid = 0; invalid < 2; ++invalid) {
      bool rejected = false;
      try { Store(cache,invalid ? 1 : 0,invalid ? old : CacheLineData(15),{}, {},return_data); }
      catch (const std::invalid_argument &) { rejected = true; }
      Check(rejected && Fields(cache.stats()) == stable,
            "malformed write must reject before counters or cache mutation");
    }
  }
}

void TestMemoryTrace() {
  constexpr std::uint64_t base = UINT64_C(0x900000000);
  constexpr unsigned size = 4U * 1024U * 1024U;
  for (const auto mode : {MemoryMode::kDirect, MemoryMode::kBypass, MemoryMode::kCache}) {
    GpuMemorySystem memory(mode);
    std::vector<std::uint8_t> expected(size), payload(16384);
    for (unsigned i = 0; i < size; ++i) expected[i] = i * 37U + 11U;
    memory.HostWrite(base,expected.data(),expected.size());
    MemoryAccessStats total;
    std::uint64_t hash = UINT64_C(14695981039346656037);
    const auto record = [&](const MemoryAccessStats &stats) {
      total += stats;
      for (auto field : Fields(stats)) for (unsigned byte = 0; byte < 8; ++byte) {
        hash ^= (field >> (byte * 8)) & 255U;
        hash *= UINT64_C(1099511628211);
      }
    };
    // More than SLC capacity, followed by conflicting and unaligned writes.
    record(memory.Write(base,expected.data(),expected.size(),MemoryClient::kFramebuffer));
    std::uint32_t random = 0x71529384;
    for (unsigned step = 0; step < 1800; ++step) {
      random = random * 1664525U + 1013904223U;
      unsigned bytes = 1U + (random >> 17) % payload.size();
      unsigned offset = (random >> 1) % (size - payload.size());
      if (step % 3 == 0) { offset &= ~127U; bytes = ((bytes - 1) & ~127U) + 128; }
      for (unsigned i = 0; i < bytes; ++i) payload[i] = step * 7U + i * 19U;
      if (step % 19 == 0) memory.HostWrite(base + offset,payload.data(),bytes);
      else record(memory.Write(base + offset,payload.data(),bytes,
                               step % 2 ? MemoryClient::kFramebuffer : MemoryClient::kFragmentImage));
      std::copy_n(payload.begin(),bytes,expected.begin() + offset);
      if (step % 7 == 0) {
        const auto result = memory.Read(base + offset,bytes,MemoryClient::kTextureCache);
        record(result.stats);
        Check(std::equal(result.data.begin(),result.data.end(),expected.begin() + offset),
              "partial/full GPU store or host invalidation changed bytes");
      }
      if (step % 113 == 0) record(memory.Flush());
    }
    const auto final = memory.Readback(base,size,MemoryClient::kShaderImageReadback);
    record(final.stats);
    Check(final.data == expected, "GPU/DRAM whole backing trace mismatch");
    // Frozen pre-optimization v5 implementation, same deterministic operations.
    // Hash covers every field of every operation in order, not just totals.
    constexpr std::array<std::uint64_t,3> baseline_hashes = {
        UINT64_C(10240765984881413160),UINT64_C(6964219422181272517),
        UINT64_C(8070626494531387353)};
    constexpr std::array<std::array<std::uint64_t,16>,3> baseline_totals = {{
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,6254565,18361126},
        {0,0,0,0,0,0,0,1965,0,259,1706,6254565,18361126,1965,0,0},
        {163066,18509,144557,72118,90948,71920,132641,0,163066,2062,
         132641,4458112,16978048,134703,0,0}}};
    const unsigned index = static_cast<unsigned>(mode);
    Check(hash == baseline_hashes[index] && Fields(total) == baseline_totals[index],
          "GPU traffic/counter trace differs from unoptimized baseline");
    std::cout << "memory trace mode=" << static_cast<unsigned>(mode) << " hash=" << hash << " counters=";
    for (auto field : Fields(total)) std::cout << field << ',';
    std::cout << '\n';
  }
}

void Benchmark() {
  CacheArray cache({"allocation",256,128,2,1});
  CacheLineData line(128,0x5a);
  allocations = 0;
  count_allocations = true;
  for (unsigned i = 0; i < 20000; ++i)
    (void)Store(cache,(i % 3) * 128,line,{}, {},false);
  count_allocations = false;
  const auto cache_allocations = allocations;
  GpuMemorySystem memory(MemoryMode::kCache);
  CacheLineData bulk(4U * 1024U * 1024U,0x37);
  memory.HostWrite(UINT64_C(0xa00000000),bulk.data(),bulk.size());
  (void)memory.Write(UINT64_C(0xa00000000),bulk.data(),bulk.size(),MemoryClient::kFramebuffer);
  allocations = 0;
  const auto begin = std::chrono::steady_clock::now();
  count_allocations = true;
  for (unsigned i = 0; i < 16; ++i)
    (void)memory.Write(UINT64_C(0xa00000000),bulk.data(),bulk.size(),MemoryClient::kFramebuffer);
  count_allocations = false;
  const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  std::cout << "host allocations: cache=" << cache_allocations << " GPU-64MiB=" << allocations
            << " elapsed_seconds=" << elapsed << '\n';
#ifndef PVRGPU_CACHE_WRITE_BASELINE
  Check(cache_allocations == 0, "full-line cache stores must reuse resident storage without response copies");
  Check(allocations == 16, "each multi-line GPU Write must reuse one scratch allocation");
#endif
}
}
int main() {
  try {
    TestCacheEquivalence(); TestMemoryTrace(); Benchmark();
    std::cout << "cache write allocation: PASS " << checks << " checks\n";
  } catch (const std::exception &error) {
    count_allocations = false;
    std::cerr << "cache write allocation FAIL: " << error.what() << '\n';
    return 1;
  }
}
