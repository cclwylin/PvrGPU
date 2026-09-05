/*
 * Functional tag/data-array implementation for CacheArray.  Timing, DRAM
 * latency and SystemC events belong to the owning cache controller; this file
 * only implements address mapping, write-back/write-allocate and true LRU.
 */
#include "cache_mmu/cache_array.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace pvrgpu::stub {
namespace {

bool IsPowerOfTwo(std::size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

void AddChecked(std::uint64_t &left, std::uint64_t right, const char *field) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left)
    throw std::overflow_error(std::string("CacheStats overflow: ") + field);
  left += right;
}

} // namespace

CacheStats operator-(const CacheStats &after, const CacheStats &before) {
  if (after.line_accesses < before.line_accesses ||
      after.read_accesses < before.read_accesses ||
      after.write_accesses < before.write_accesses ||
      after.hits < before.hits || after.misses < before.misses ||
      after.evictions < before.evictions ||
      after.writebacks < before.writebacks ||
      after.bypassed < before.bypassed) {
    throw std::logic_error("CacheStats delta underflow");
  }
  return {
      after.line_accesses - before.line_accesses,
      after.read_accesses - before.read_accesses,
      after.write_accesses - before.write_accesses,
      after.hits - before.hits,
      after.misses - before.misses,
      after.evictions - before.evictions,
      after.writebacks - before.writebacks,
      after.bypassed - before.bypassed,
  };
}

CacheStats &operator+=(CacheStats &left, const CacheStats &right) {
  AddChecked(left.line_accesses, right.line_accesses, "line_accesses");
  AddChecked(left.read_accesses, right.read_accesses, "read_accesses");
  AddChecked(left.write_accesses, right.write_accesses, "write_accesses");
  AddChecked(left.hits, right.hits, "hits");
  AddChecked(left.misses, right.misses, "misses");
  AddChecked(left.evictions, right.evictions, "evictions");
  AddChecked(left.writebacks, right.writebacks, "writebacks");
  AddChecked(left.bypassed, right.bypassed, "bypassed");
  return left;
}

CacheArray::CacheArray(CacheArrayConfig config, bool bypass)
    : config_(config), bypass_(bypass) {
  if (config_.name.empty())
    throw std::invalid_argument("CacheArray config name must not be empty");
  if (config_.capacity_bytes == 0)
    throw std::invalid_argument("CacheArray capacity_bytes must be non-zero");
  if (!IsPowerOfTwo(config_.line_size_bytes))
    throw std::invalid_argument(
        "CacheArray line_size_bytes must be a non-zero power of two");
  if (!IsPowerOfTwo(config_.ways))
    throw std::invalid_argument(
        "CacheArray ways must be a non-zero power of two");
  if (!IsPowerOfTwo(config_.banks))
    throw std::invalid_argument(
        "CacheArray banks must be a non-zero power of two");
  if (config_.capacity_bytes % config_.line_size_bytes != 0)
    throw std::invalid_argument(
        "CacheArray capacity must contain whole cache lines");

  line_count_ = config_.capacity_bytes / config_.line_size_bytes;
  if (config_.ways > std::numeric_limits<std::size_t>::max() / config_.banks) {
    throw std::invalid_argument("CacheArray ways * banks overflows size_t");
  }
  const std::size_t lines_per_set_group = config_.ways * config_.banks;
  if (line_count_ < lines_per_set_group ||
      line_count_ % lines_per_set_group != 0) {
    throw std::invalid_argument(
        "CacheArray line count must be divisible by ways * banks");
  }
  sets_per_bank_ = line_count_ / lines_per_set_group;

  if (config_.banks >
      std::numeric_limits<std::size_t>::max() / sets_per_bank_) {
    throw std::invalid_argument("CacheArray set count overflows size_t");
  }
  sets_.resize(config_.banks * sets_per_bank_);
  for (auto &set : sets_) {
    set.resize(config_.ways);
    for (auto &line : set)
      line.data.resize(config_.line_size_bytes, 0);
  }
}

CacheArray::AddressFields
CacheArray::DecodeAddress(std::uint64_t line_address) const {
  if (line_address % config_.line_size_bytes != 0)
    throw std::invalid_argument("CacheArray line address is not aligned");

  const std::uint64_t line_number = line_address / config_.line_size_bytes;
  AddressFields fields;
  fields.line_address = line_address;
  fields.bank = static_cast<std::size_t>(line_number % config_.banks);
  const std::uint64_t bank_line = line_number / config_.banks;
  fields.set = static_cast<std::size_t>(bank_line % sets_per_bank_);
  fields.tag = bank_line / sets_per_bank_;
  return fields;
}

std::uint64_t CacheArray::ReconstructAddress(std::uint64_t tag,
                                             std::size_t bank,
                                             std::size_t set) const {
  const std::uint64_t bank_line = tag * sets_per_bank_ + set;
  const std::uint64_t line_number = bank_line * config_.banks + bank;
  return line_number * config_.line_size_bytes;
}

std::size_t CacheArray::FindWay(const AddressFields &fields) const {
  const std::size_t set_index = fields.bank * sets_per_bank_ + fields.set;
  const auto &set = sets_[set_index];
  for (std::size_t way = 0; way < set.size(); ++way) {
    if (set[way].valid && set[way].tag == fields.tag)
      return way;
  }
  return CacheLineAccess::kNoCacheIndex;
}

std::size_t CacheArray::ChooseVictim(const AddressFields &fields) const {
  const std::size_t set_index = fields.bank * sets_per_bank_ + fields.set;
  const auto &set = sets_[set_index];
  for (std::size_t way = 0; way < set.size(); ++way) {
    if (!set[way].valid)
      return way;
  }

  std::size_t victim = 0;
  for (std::size_t way = 1; way < set.size(); ++way) {
    if (set[way].lru_rank > set[victim].lru_rank)
      victim = way;
  }
  return victim;
}

void CacheArray::Touch(std::size_t set_index, std::size_t way) {
  auto &set = sets_[set_index];
  Line &touched = set.at(way);
  const std::size_t old_rank = touched.valid ? touched.lru_rank : set.size();
  for (std::size_t other_way = 0; other_way < set.size(); ++other_way) {
    Line &other = set[other_way];
    if (other_way != way && other.valid && other.lru_rank < old_rank)
      ++other.lru_rank;
  }
  touched.lru_rank = 0;
}

CacheLineData CacheArray::ReadLower(std::uint64_t line_address,
                                    const CacheLineRead &lower_read) const {
  CacheLineData data = lower_read
                           ? lower_read(line_address, config_.line_size_bytes)
                           : CacheLineData(config_.line_size_bytes, 0);
  if (data.size() != config_.line_size_bytes)
    throw std::runtime_error(
        "CacheArray lower read returned an incorrectly sized cache line");
  return data;
}

void CacheArray::WriteLower(std::uint64_t line_address,
                            const CacheLineData &data,
                            const CacheLineWrite &lower_write) const {
  if (data.size() != config_.line_size_bytes)
    throw std::logic_error("CacheArray resident line has an invalid size");
  if (lower_write)
    lower_write(line_address, data);
}

CacheLineAccess CacheArray::AccessLine(std::uint64_t line_address,
                                       bool is_write,
                                       const CacheLineData *write_data,
                                       const CacheLineRead &lower_read,
                                       const CacheLineWrite &lower_write,
                                       bool return_data) {
  const AddressFields fields = DecodeAddress(line_address);
  if (write_data && write_data->size() != config_.line_size_bytes)
    throw std::invalid_argument(
        "CacheArray WriteLine data size must equal line_size_bytes");

  const CacheStats before = stats_;
  ++stats_.line_accesses;
  if (is_write)
    ++stats_.write_accesses;
  else
    ++stats_.read_accesses;

  CacheLineAccess result;
  result.line_address = fields.line_address;

  if (bypass_) {
    ++stats_.bypassed;
    result.bypassed = true;
    if (is_write) {
      const CacheLineData &data =
          write_data ? *write_data : CacheLineData(config_.line_size_bytes, 0);
      WriteLower(line_address, data, lower_write);
      if (return_data)
        result.data = data;
    } else {
      result.data = ReadLower(line_address, lower_read);
      if (!return_data)
        result.data.clear();
    }
    result.delta = stats_ - before;
    return result;
  }

  result.bank = fields.bank;
  result.set = fields.set;
  const std::size_t set_index = fields.bank * sets_per_bank_ + fields.set;
  std::size_t way = FindWay(fields);
  if (way != CacheLineAccess::kNoCacheIndex) {
    ++stats_.hits;
    result.hit = true;
  } else {
    ++stats_.misses;

    // A full-line store supplies every byte, so write-allocate can install it
    // without a synchronous lower read.  Read misses still fetch before any
    // set mutation, preserving cache state if a lower callback fails.
    CacheLineData fill = is_write && write_data
                             ? *write_data
                             : ReadLower(line_address, lower_read);
    way = ChooseVictim(fields);
    Line &victim = sets_[set_index][way];
    if (victim.valid) {
      ++stats_.evictions;
      if (victim.dirty) {
        const std::uint64_t victim_address =
            ReconstructAddress(victim.tag, fields.bank, fields.set);
        WriteLower(victim_address, victim.data, lower_write);
        ++stats_.writebacks;
      }
    }
    Touch(set_index, way);
    victim.valid = true;
    victim.dirty = false;
    victim.tag = fields.tag;
    victim.data = std::move(fill);
  }

  Line &line = sets_[set_index][way];
  Touch(set_index, way);
  if (is_write) {
    line.dirty = true;
    if (write_data)
      line.data = *write_data;
  }

  result.way = way;
  if (return_data)
    result.data = line.data;
  result.delta = stats_ - before;
  return result;
}

CacheLineAccess CacheArray::ReadLine(std::uint64_t line_address,
                                     const CacheLineRead &lower_read,
                                     const CacheLineWrite &lower_write) {
  return AccessLine(line_address, false, nullptr, lower_read, lower_write,
                    true);
}

CacheLineAccess CacheArray::WriteLine(std::uint64_t line_address,
                                      const CacheLineData &data,
                                      const CacheLineRead &lower_read,
                                      const CacheLineWrite &lower_write) {
  return AccessLine(line_address, true, &data, lower_read, lower_write, true);
}

CacheStats CacheArray::AccessRange(std::uint64_t address, std::size_t bytes,
                                   bool is_write) {
  if (bytes == 0)
    return {};
  if (bytes - 1 > std::numeric_limits<std::uint64_t>::max() - address)
    throw std::overflow_error("CacheArray access range wraps address space");

  const CacheStats before = stats_;
  const std::uint64_t first_line = address - address % config_.line_size_bytes;
  const std::uint64_t final_address = address + bytes - 1;
  const std::uint64_t final_line =
      final_address - final_address % config_.line_size_bytes;

  std::uint64_t line_address = first_line;
  while (true) {
    (void)AccessLine(line_address, is_write, nullptr, {}, {}, false);
    if (line_address == final_line)
      break;
    line_address += config_.line_size_bytes;
  }
  return stats_ - before;
}

std::uint64_t CacheArray::Flush(const CacheLineWrite &lower_write) {
  std::uint64_t flushed = 0;
  for (std::size_t bank = 0; bank < config_.banks; ++bank) {
    for (std::size_t set = 0; set < sets_per_bank_; ++set) {
      auto &cache_set = sets_[bank * sets_per_bank_ + set];
      for (Line &line : cache_set) {
        if (!line.valid || !line.dirty)
          continue;
        WriteLower(ReconstructAddress(line.tag, bank, set), line.data,
                   lower_write);
        line.dirty = false;
        ++stats_.writebacks;
        ++flushed;
      }
    }
  }
  return flushed;
}

void CacheArray::InvalidateAll() noexcept {
  for (auto &set : sets_) {
    for (Line &line : set) {
      line.valid = false;
      line.dirty = false;
      line.tag = 0;
      line.lru_rank = 0;
      std::fill(line.data.begin(), line.data.end(), 0);
    }
  }
}

std::uint64_t CacheArray::InvalidateRange(std::uint64_t address,
                                          std::size_t bytes,
                                          const CacheLineWrite &lower_write) {
  if (bytes == 0 || bypass_)
    return 0;
  if (bytes > std::numeric_limits<std::uint64_t>::max() - address)
    throw std::overflow_error("CacheArray invalidate range overflows");
  const std::uint64_t line_bytes = config_.line_size_bytes;
  const std::uint64_t end = address + bytes;
  std::uint64_t flushed = 0;
  for (std::uint64_t line_address = address - address % line_bytes;
       line_address < end; line_address += line_bytes) {
    const AddressFields fields = DecodeAddress(line_address);
    const std::size_t way = FindWay(fields);
    if (way == CacheLineAccess::kNoCacheIndex)
      continue;
    Line &line = sets_[fields.bank * sets_per_bank_ + fields.set][way];
    // Clean before dropping: a line the GPU dirtied may hold bytes outside the
    // range the host is replacing, and they belong in DRAM either way.
    if (line.dirty) {
      WriteLower(line_address, line.data, lower_write);
      ++flushed;
    }
    line.valid = false;
    line.dirty = false;
    line.tag = 0;
    line.lru_rank = 0;
    std::fill(line.data.begin(), line.data.end(), 0);
  }
  return flushed;
}

std::uint64_t CacheArray::SetBypass(bool bypass,
                                    const CacheLineWrite &lower_write) {
  if (bypass_ == bypass)
    return 0;
  std::uint64_t flushed = 0;
  if (bypass) {
    flushed = Flush(lower_write);
    InvalidateAll();
  }
  bypass_ = bypass;
  return flushed;
}

} // namespace pvrgpu::stub
