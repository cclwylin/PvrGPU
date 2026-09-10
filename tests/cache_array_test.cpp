/*
 * Unit tests for the non-SystemC CacheArray helper: profile geometry,
 * bank/set mapping, write-back/write-allocate, true LRU, bypass, range access,
 * flush and fail-fast input validation.
 */
#include "cache_mmu/cache_array.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <typeinfo>
#include <utility>
#include <vector>

namespace {

using pvrgpu::stub::CacheArray;
using pvrgpu::stub::CacheArrayConfig;
using pvrgpu::stub::CacheLineData;
using pvrgpu::stub::CacheStats;
using pvrgpu::stub::CacheLineAccess;
using pvrgpu::stub::McuCacheConfig;
using pvrgpu::stub::SlcCacheConfig;
using pvrgpu::stub::TcuCacheConfig;
using pvrgpu::stub::UscL2CacheConfig;

void Check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error("test check failed: " + message);
}

template <typename Function>
void ExpectFailure(Function &&function, const std::string &description) {
  try {
    function();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error("expected failure: " + description);
}

CacheLineData Pattern(std::size_t size, std::uint8_t value) {
  return CacheLineData(size, value);
}

auto Stats(const CacheStats &s) {
  return std::make_tuple(s.line_accesses, s.read_accesses, s.write_accesses,
      s.hits, s.misses, s.evictions, s.writebacks, s.bypassed);
}

void SameAccess(const CacheLineAccess &old, const CacheLineAccess &fresh) {
  Check(std::tie(old.line_address, old.bank, old.set, old.way, old.hit, old.bypassed) ==
            std::tie(fresh.line_address, fresh.bank, fresh.set, fresh.way,
                     fresh.hit, fresh.bypassed) && Stats(old.delta) == Stats(fresh.delta),
        "range read preserves all access metadata and counters");
  Check(fresh.data.empty(), "range read has no owned response payload");
}

void TestReferenceProfiles() {
  CacheArray mcu(McuCacheConfig());
  CacheArray tcu(TcuCacheConfig());
  CacheArray slc(SlcCacheConfig());
  CacheArray usc_l2(UscL2CacheConfig());

  Check(mcu.line_count() == 384 && mcu.sets_per_bank() == 24,
        "MCU is 24 KB, 64 B, four-bank, four-way");
  Check(tcu.line_count() == 384 && tcu.sets_per_bank() == 24,
        "TCU is 24 KB, 64 B, four-bank, four-way");
  Check(slc.line_count() == 16384 && slc.sets_per_bank() == 256,
        "SLC is 2 MB, 128 B, eight-bank, eight-way");
  Check(usc_l2.line_count() == 128 && usc_l2.sets_per_bank() == 32,
        "USC L2 is 8 KB, 64 B, one-bank, four-way");
  Check(!mcu.bypass() && !tcu.bypass() && !slc.bypass() && !usc_l2.bypass(),
        "cache bypass defaults off");
}

void TestConfigValidation() {
  ExpectFailure([] { CacheArray cache({"", 1024, 64, 1, 1}); },
                "empty profile name");
  ExpectFailure([] { CacheArray cache({"bad", 0, 64, 1, 1}); },
                "zero capacity");
  ExpectFailure([] { CacheArray cache({"bad", 1024, 0, 1, 1}); },
                "zero line size");
  ExpectFailure([] { CacheArray cache({"bad", 1024, 48, 1, 1}); },
                "non-power-of-two line size");
  ExpectFailure([] { CacheArray cache({"bad", 1024, 64, 3, 1}); },
                "non-power-of-two ways");
  ExpectFailure([] { CacheArray cache({"bad", 1024, 64, 1, 3}); },
                "non-power-of-two banks");
  ExpectFailure([] { CacheArray cache({"bad", 1000, 64, 1, 1}); },
                "partial cache line");
  ExpectFailure([] { CacheArray cache({"bad", 64, 64, 2, 1}); },
                "capacity smaller than one set");
  ExpectFailure([] { CacheArray cache({"bad", 320, 64, 2, 2}); },
                "line count not divisible by ways times banks");
}

void TestBankMappingAndRangeDelta() {
  CacheArray cache({"mapping", 256, 16, 2, 2});
  const auto a0 = cache.ReadLine(0);
  const auto a1 = cache.ReadLine(16);
  const auto a2 = cache.ReadLine(32);
  Check(a0.bank == 0 && a0.set == 0, "line zero maps to bank zero");
  Check(a1.bank == 1 && a1.set == 0,
        "consecutive line interleaves to bank one");
  Check(a2.bank == 0 && a2.set == 1,
        "third line advances set within bank zero");

  cache.ResetStats();
  const auto first = cache.AccessRange(63, 34, false);
  Check(first.line_accesses == 4 && first.read_accesses == 4,
        "unaligned range touches exactly four lines");
  Check(first.hits == 0 && first.misses == 4,
        "range delta reports four cold misses");
  const auto second = cache.AccessRange(63, 34, false);
  Check(second.line_accesses == 4 && second.hits == 4 && second.misses == 0,
        "second range reports four hits in its own delta");

  Check(cache.AccessRange(0, 0, true).line_accesses == 0,
        "empty range performs no access");
  ExpectFailure([&] { (void)cache.AccessRange(UINT64_MAX - 3, 8, false); },
                "wrapping address range");
}

void TestTrueLru() {
  // Two sets, two ways.  0, 32 and 64 all map to set zero.
  CacheArray cache({"lru", 64, 16, 2, 1});
  (void)cache.ReadLine(0);
  (void)cache.ReadLine(32);
  Check(cache.ReadLine(0).hit, "touch makes address zero MRU");
  const auto replacement = cache.ReadLine(64);
  Check(!replacement.hit && replacement.delta.evictions == 1,
        "third tag evicts the true LRU line");
  Check(cache.ReadLine(0).hit, "MRU line survives replacement");
  Check(!cache.ReadLine(32).hit, "untouched line was the LRU victim");
}

void TestWriteAllocateEvictionAndFlush() {
  CacheArray cache({"writeback", 32, 16, 1, 1});
  std::map<std::uint64_t, CacheLineData> memory;
  memory.emplace(0, Pattern(16, 0x10));
  memory.emplace(32, Pattern(16, 0x20));
  std::uint64_t lower_reads = 0;
  std::vector<std::pair<std::uint64_t, CacheLineData>> lower_writes;

  const auto read = [&](std::uint64_t address, std::size_t size) {
    ++lower_reads;
    Check(size == 16, "lower read line size");
    return memory.at(address);
  };
  const auto write = [&](std::uint64_t address, const CacheLineData &data) {
    lower_writes.emplace_back(address, data);
    memory[address] = data;
  };

  const CacheLineData dirty_data = Pattern(16, 0xa5);
  const auto allocated = cache.WriteLine(0, dirty_data, read, write);
  Check(!allocated.hit && allocated.delta.misses == 1 && lower_reads == 0,
        "full-line write miss allocates without a lower read");
  Check(lower_writes.empty(), "write-back defers lower write");
  Check(cache.ReadLine(0, read, write).data == dirty_data,
        "read hit observes resident dirty data");

  const auto evict = cache.ReadLine(32, read, write);
  Check(lower_reads == 1, "read miss fetches exactly one lower line");
  Check(evict.delta.evictions == 1 && evict.delta.writebacks == 1,
        "conflict miss writes back dirty victim");
  Check(lower_writes.size() == 1 && lower_writes[0].first == 0 &&
            lower_writes[0].second == dirty_data,
        "dirty eviction preserves victim address and bytes");

  (void)cache.WriteLine(32, Pattern(16, 0x5a), read, write);
  const std::uint64_t first_flush = cache.Flush(write);
  Check(first_flush == 1 && lower_writes.size() == 2,
        "flush writes each dirty line once");
  Check(cache.Flush(write) == 0 && lower_writes.size() == 2,
        "second flush does not rewrite clean lines");
}

void TestBypass() {
  CacheArray cache({"bypass", 64, 16, 2, 1}, true);
  std::uint64_t reads = 0;
  std::uint64_t writes = 0;
  const auto read = [&](std::uint64_t, std::size_t size) {
    ++reads;
    return Pattern(size, 0x33);
  };
  const auto write = [&](std::uint64_t, const CacheLineData &) { ++writes; };

  Check(cache.ReadLine(0, read, write).bypassed,
        "constructor enables requested bypass");
  Check(cache.ReadLine(0, read, write).bypassed,
        "bypassed reads do not allocate");
  Check(cache.WriteLine(0, Pattern(16, 0x44), read, write).bypassed,
        "bypassed write goes directly lower");
  Check(reads == 2 && writes == 1, "bypass directly accesses lower storage");
  Check(cache.stats().bypassed == 3 && cache.stats().hits == 0 &&
            cache.stats().misses == 0,
        "bypass is counted separately from hits and misses");

  Check(cache.SetBypass(false, write) == 0, "disable empty bypass cache");
  (void)cache.WriteLine(0, Pattern(16, 0x77), read, write);
  Check(cache.SetBypass(true, write) == 1,
        "enabling bypass flushes one dirty resident line");
  Check(cache.SetBypass(false, write) == 0, "disable bypass again");
  Check(!cache.ReadLine(0, read, write).hit,
        "bypass transition invalidated prior cache contents");
}

void TestLineValidationAndResetStats() {
  CacheArray cache({"validation", 64, 16, 2, 1});
  ExpectFailure([&] { (void)cache.ReadLine(1); }, "unaligned line address");
  ExpectFailure([&] { (void)cache.WriteLine(0, Pattern(15, 0)); },
                "incorrect write line size");
  ExpectFailure(
      [&] {
        (void)cache.ReadLine(
            0, [](std::uint64_t, std::size_t) { return Pattern(15, 0); });
      },
      "incorrect lower fill size");

  (void)cache.ReadLine(0);
  cache.ResetStats();
  Check(cache.stats().line_accesses == 0, "ResetStats clears counters");
  Check(cache.ReadLine(0).hit, "ResetStats preserves resident lines");
}

void TestReadLineInto(bool bypass) {
  CacheArray legacy({"into", 64, 16, 2, 1}, bypass);
  CacheArray into({"into", 64, 16, 2, 1}, bypass);
  using Store = std::map<std::uint64_t, CacheLineData>;
  Store old_memory, new_memory;
  using Events = std::vector<std::pair<char, std::uint64_t>>;
  Events old_events, new_events;
  for (std::uint64_t address = 0; address < 160; address += 16) {
    CacheLineData line(16);
    for (std::size_t i = 0; i < line.size(); ++i)
      line[i] = static_cast<std::uint8_t>(address + 3 * i);
    old_memory[address] = new_memory[address] = line;
  }
  const auto old_read = [&](std::uint64_t a, std::size_t n) {
    Check(n == 16, "legacy lower size");
    old_events.emplace_back('R', a); return old_memory.at(a);
  };
  const auto new_read = [&](std::uint64_t a, std::size_t n) {
    Check(n == 16, "range lower size");
    new_events.emplace_back('R', a); return new_memory.at(a);
  };
  const auto old_write = [&](std::uint64_t a, const CacheLineData &d) {
    old_events.emplace_back('W', a); old_memory[a] = d;
  };
  const auto new_write = [&](std::uint64_t a, const CacheLineData &d) {
    new_events.emplace_back('W', a); new_memory[a] = d;
  };
  const auto read = [&](std::uint64_t address, std::size_t offset,
                        std::size_t bytes) {
    std::array<std::uint8_t, 32> guarded;
    guarded.fill(0x9b);
    const auto old = legacy.ReadLine(address, old_read, old_write);
    const auto fresh = into.ReadLineInto(address, offset, guarded.data() + 8,
                                         bytes, new_read, new_write);
    SameAccess(old, fresh);
    Check(std::equal(old.data.begin() + offset, old.data.begin() + offset + bytes,
                     guarded.begin() + 8), "range data equals legacy slice");
    Check(std::all_of(guarded.begin(), guarded.begin() + 8,
                     [](auto v) { return v == 0x9b; }) &&
              std::all_of(guarded.begin() + 8 + bytes, guarded.end(),
                          [](auto v) { return v == 0x9b; }),
          "range copy respects exact caller span");
    Check(old_events == new_events && old_memory == new_memory &&
              Stats(legacy.stats()) == Stats(into.stats()),
          "range read preserves lower callback sequence, bytes and lifetime stats");
  };
  for (std::size_t offset = 0; offset < 16; ++offset)
    for (std::size_t bytes = 1; bytes <= 16 - offset; ++bytes) {
      read((offset % 5) * 32, offset, bytes);
      read((offset % 5) * 32, offset, bytes);
    }
  for (const std::uint64_t address : {0U, 32U, 64U, 0U, 96U, 128U}) {
    const auto data = Pattern(16, static_cast<std::uint8_t>(address + 97));
    const auto a = legacy.WriteLine(address, data, old_read, old_write, false);
    const auto b = into.WriteLine(address, data, new_read, new_write, false);
    SameAccess(a, b);
    read(address, 3, 9);
    read((address + 32) % 160, 1, 15);
  }
  Check(legacy.Flush(old_write) == into.Flush(new_write) &&
            old_events == new_events && old_memory == new_memory &&
            Stats(legacy.stats()) == Stats(into.stats()),
        "range read preserves dirty flush and LRU-induced writebacks");
  Check(legacy.InvalidateRange(0, 80, old_write) ==
            into.InvalidateRange(0, 80, new_write), "range invalidation matches");
  old_memory[0] = new_memory[0] = Pattern(16, 0xe3);
  read(0, 0, 16);
  Check(legacy.SetBypass(!bypass, old_write) == into.SetBypass(!bypass, new_write),
        "range cache bypass transition matches");
  read(32, 15, 1);
}

void TestReadLineIntoFailures() {
  CacheArray legacy({"failure", 16, 16, 1, 1});
  CacheArray into({"failure", 16, 16, 1, 1});
  std::array<std::uint8_t, 16> destination;
  destination.fill(0x63);
  const auto original = destination;
  const auto failure = [](const auto &call) {
    try { call(); }
    catch (const std::exception &error) {
      return std::string(typeid(error).name()) + ':' + error.what();
    }
    throw std::runtime_error("expected cache read failure");
  };
  const auto compare = [&](std::uint64_t address,
                           const pvrgpu::stub::CacheLineRead &read,
                           const pvrgpu::stub::CacheLineWrite &write) {
    const auto a = failure([&] { (void)legacy.ReadLine(address, read, write); });
    const auto b = failure([&] {
      (void)into.ReadLineInto(address, 5, destination.data(), 4, read, write);
    });
    Check(a == b && Stats(legacy.stats()) == Stats(into.stats()),
          "range read matches lower failure type, message and counters");
    Check(destination == original, "failed single-line read leaves destination unchanged");
  };
  compare(1, {}, {});
  compare(0, [](std::uint64_t, std::size_t) { return Pattern(15, 0); }, {});
  compare(0, [](std::uint64_t, std::size_t) -> CacheLineData {
    throw std::runtime_error("lower read sentinel");
  }, {});
  (void)legacy.WriteLine(0, Pattern(16, 0x75));
  (void)into.WriteLine(0, Pattern(16, 0x75));
  compare(16, {}, [](std::uint64_t, const CacheLineData &) {
    throw std::runtime_error("dirty writeback sentinel");
  });
  const auto old = legacy.ReadLine(0);
  const auto fresh = into.ReadLineInto(0, 0, destination.data(), 16);
  SameAccess(old, fresh);
  Check(std::equal(destination.begin(), destination.end(), old.data.begin()) && fresh.hit,
        "failed dirty eviction preserves old resident line");

  const auto before = Stats(into.stats());
  ExpectFailure([&] { (void)into.ReadLineInto(0, 0, nullptr, 1); }, "null destination");
  ExpectFailure([&] { (void)into.ReadLineInto(0, 0, destination.data(), 0); }, "zero span");
  ExpectFailure([&] { (void)into.ReadLineInto(0, 16, destination.data(), 1); }, "past line");
  ExpectFailure([&] { (void)into.ReadLineInto(0, 15, destination.data(), 2); }, "cross-line span");
  ExpectFailure([&] { (void)into.ReadLineInto(0, SIZE_MAX, destination.data(), 1); }, "huge offset");
  ExpectFailure([&] { (void)into.ReadLineInto(0, 1, destination.data(), SIZE_MAX); }, "huge span");
  Check(before == Stats(into.stats()), "invalid destination/span causes no cache accesses");
}

} // namespace

int main() {
  try {
    TestReferenceProfiles();
    TestConfigValidation();
    TestBankMappingAndRangeDelta();
    TestTrueLru();
    TestWriteAllocateEvictionAndFlush();
    TestBypass();
    TestLineValidationAndResetStats();
    TestReadLineInto(false);
    TestReadLineInto(true);
    TestReadLineIntoFailures();
    std::cout << "cache_array_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "cache_array_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
