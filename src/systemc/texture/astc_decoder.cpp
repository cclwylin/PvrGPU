#include "texture/astc_decoder.h"

#include <algorithm>
#include <array>
#include <cstddef>

namespace pvrgpu::stub {
namespace {

// Rogue TEXSTATE FORMAT_COMPRESSED ASTC values, in the order texstate.xml
// declares them.  The table is the enum, not a guess: index is the encoded
// value, so a value outside it is not ASTC.
constexpr AstcBlockFootprint kRogueAstcFootprints[] = {
    {4, 4},   {5, 4},  {5, 5},   {6, 5},   {6, 6},   {8, 5},  {8, 6},
    {8, 8},   {10, 5}, {10, 6},  {10, 8},  {10, 10}, {12, 10}, {12, 12},
};

// The spec's ceilings, and the reason every fixed-size array below is the
// size it is.  A block holds at most 64 weights and 18 colour endpoint
// values; the largest footprint is 12x12.
constexpr std::size_t kMaxWeights = 64;
constexpr std::size_t kMaxEndpointValues = 18;
constexpr std::size_t kMaxTexels = 144;
constexpr std::size_t kMaxPartitions = 4;

// One little-endian bit field out of the 128-bit block.  ASTC numbers its
// bits from the low end of the first byte, which is how the block arrives in
// memory, so no byte swapping happens anywhere in this file.
std::uint32_t BlockBits(const std::uint8_t block[16], unsigned first,
                        unsigned last) {
  std::uint32_t value = 0;
  for (unsigned bit = first; bit <= last; ++bit) {
    const unsigned byte = bit >> 3U;
    const unsigned offset = bit & 7U;
    if ((block[byte] >> offset) & 1U)
      value |= 1U << (bit - first);
  }
  return value;
}

std::uint32_t ReverseBits(std::uint32_t value, unsigned count) {
  std::uint32_t reversed = 0;
  for (unsigned bit = 0; bit < count; ++bit)
    reversed |= ((value >> bit) & 1U) << (count - 1U - bit);
  return reversed;
}

// Widen a small unsigned field by repeating its bits, which is how ASTC
// unquantises a value stored as plain bits: the pattern is tiled from the
// top so that an all-ones input stays all-ones.
std::uint32_t BitReplicate(std::uint32_t value, unsigned source_bits,
                           unsigned destination_bits) {
  std::uint32_t result = 0;
  const int step = static_cast<int>(source_bits);
  for (int shift = static_cast<int>(destination_bits) - step; shift > -step;
       shift -= step)
    result |= shift >= 0 ? value << shift : value >> -shift;
  return result;
}

/*
 * Integer Sequence Encoding.
 *
 * ASTC packs both the weights and the colour endpoints as a sequence of
 * integers whose range need not be a power of two.  A range of the form
 * 3*2^n is stored as a trit plus n bits, 5*2^n as a quint plus n bits, and
 * 2^n as plain bits; five trits (or three quints) share one packed field so
 * that no bits are wasted.
 */
enum class IseMode { kPlainBit, kTrit, kQuint };

struct IseParams {
  IseMode mode = IseMode::kPlainBit;
  unsigned bits = 0;
};

unsigned IseSequenceBits(const IseParams &params, unsigned count) {
  switch (params.mode) {
  case IseMode::kTrit:
    return (count * 8U + 4U) / 5U + count * params.bits;
  case IseMode::kQuint:
    return (count * 7U + 2U) / 3U + count * params.bits;
  case IseMode::kPlainBit:
    break;
  }
  return count * params.bits;
}

// The colour endpoints are not given a range: they take the widest range
// whose sequence still fits the bits the block has left over after the
// weights and the configuration.  Widest wins, and among equals the search
// simply steps down until one fits.
IseParams IseParamsForBudget(unsigned available_bits, unsigned count) {
  int trit_bits = 6;
  int quint_bits = 5;
  int plain_bits = 8;
  while (true) {
    const long trit_range = trit_bits > 0 ? (3L << trit_bits) - 1 : -1;
    const long quint_range = quint_bits > 0 ? (5L << quint_bits) - 1 : -1;
    const long plain_range = plain_bits > 0 ? (1L << plain_bits) - 1 : -1;
    const long widest = std::max(std::max(trit_range, quint_range), plain_range);
    if (widest < 0)
      return IseParams{IseMode::kPlainBit, 1};
    if (widest == trit_range) {
      const IseParams params{IseMode::kTrit, static_cast<unsigned>(trit_bits)};
      if (IseSequenceBits(params, count) <= available_bits)
        return params;
      --trit_bits;
    } else if (widest == quint_range) {
      const IseParams params{IseMode::kQuint, static_cast<unsigned>(quint_bits)};
      if (IseSequenceBits(params, count) <= available_bits)
        return params;
      --quint_bits;
    } else {
      const IseParams params{IseMode::kPlainBit,
                             static_cast<unsigned>(plain_bits)};
      if (IseSequenceBits(params, count) <= available_bits)
        return params;
      --plain_bits;
    }
  }
}

// One decoded ISE value: the low bits `m`, the trit or quint `tq` it shares
// with its neighbours, and the value `v` the two combine into.  Unquantising
// needs all three, so the decode keeps them apart.
struct IseValue {
  std::uint32_t m = 0;
  std::uint32_t tq = 0;
  std::uint32_t v = 0;
};

// Sequential bit access into the block.  The colour endpoints are read
// forwards from a low bit; the weights are read backwards from bit 127,
// each field bit-reversed, because ASTC grows the two sequences towards each
// other from opposite ends of the block.
class BlockBitReader {
public:
  BlockBitReader(const std::uint8_t *block, int start, int length, bool forward)
      : block_(block), start_(start), length_(length), forward_(forward) {}

  std::uint32_t Next(unsigned count) {
    // Past the end of the sequence the block reads as zero: a sequence may
    // ask for bits its declared length does not cover.
    if (count == 0 || index_ >= length_)
      return 0;
    const int end = index_ + static_cast<int>(count);
    const int from_block = std::min(length_, end) - index_;
    const int low = index_;
    const int high = index_ + from_block - 1;
    index_ = end;
    if (forward_) {
      return BlockBits(block_, static_cast<unsigned>(start_ + low),
                       static_cast<unsigned>(start_ + high));
    }
    return ReverseBits(BlockBits(block_, static_cast<unsigned>(start_ - high),
                                 static_cast<unsigned>(start_ - low)),
                       static_cast<unsigned>(from_block));
  }

private:
  const std::uint8_t *block_;
  int start_;
  int length_;
  bool forward_;
  int index_ = 0;
};

// Five trits out of the eight-bit field they share.  This is the spec's own
// bit manipulation rather than a 256-entry table: the table would be a
// remembered answer, the manipulation is the rule that produces it.
void DecodeTrits(std::uint32_t packed, std::uint32_t out[5]) {
  std::uint32_t c = 0;
  std::uint32_t t3 = 0;
  std::uint32_t t4 = 0;
  if (((packed >> 2U) & 7U) == 7U) {
    c = (((packed >> 5U) & 7U) << 2U) | (packed & 3U);
    t3 = 2;
    t4 = 2;
  } else {
    c = packed & 0x1fU;
    if (((packed >> 5U) & 3U) == 3U) {
      t4 = 2;
      t3 = (packed >> 7U) & 1U;
    } else {
      t4 = (packed >> 7U) & 1U;
      t3 = (packed >> 5U) & 3U;
    }
  }

  std::uint32_t t0 = 0;
  std::uint32_t t1 = 0;
  std::uint32_t t2 = 0;
  if ((c & 3U) == 3U) {
    const std::uint32_t c3 = (c >> 3U) & 1U;
    t2 = 2;
    t1 = (c >> 4U) & 1U;
    t0 = (c3 << 1U) | (((c >> 2U) & 1U) & (c3 ^ 1U));
  } else if (((c >> 2U) & 3U) == 3U) {
    t2 = 2;
    t1 = 2;
    t0 = c & 3U;
  } else {
    const std::uint32_t c1 = (c >> 1U) & 1U;
    t2 = (c >> 4U) & 1U;
    t1 = (c >> 2U) & 3U;
    t0 = (c1 << 1U) | ((c & 1U) & (c1 ^ 1U));
  }

  out[0] = t0;
  out[1] = t1;
  out[2] = t2;
  out[3] = t3;
  out[4] = t4;
}

// Three quints out of the seven-bit field they share.
void DecodeQuints(std::uint32_t packed, std::uint32_t out[3]) {
  std::uint32_t q0 = 0;
  std::uint32_t q1 = 0;
  std::uint32_t q2 = 0;
  if (((packed >> 1U) & 3U) == 3U && ((packed >> 5U) & 3U) == 0U) {
    const std::uint32_t low = packed & 1U;
    q2 = (low << 2U) | (((((packed >> 4U) & 1U) & (low ^ 1U))) << 1U) |
         (((packed >> 3U) & 1U) & (low ^ 1U));
    q1 = 4;
    q0 = 4;
  } else {
    std::uint32_t c = 0;
    if (((packed >> 1U) & 3U) == 3U) {
      q2 = 4;
      c = (((packed >> 3U) & 3U) << 3U) | (((~(packed >> 5U)) & 3U) << 1U) |
          (packed & 1U);
    } else {
      q2 = (packed >> 5U) & 3U;
      c = packed & 0x1fU;
    }
    if ((c & 7U) == 5U) {
      q1 = 4;
      q0 = (c >> 3U) & 3U;
    } else {
      q1 = (c >> 3U) & 3U;
      q0 = c & 7U;
    }
  }

  out[0] = q0;
  out[1] = q1;
  out[2] = q2;
}

void DecodeIseTritBlock(IseValue *dst, unsigned count, BlockBitReader &reader,
                        unsigned bits) {
  std::uint32_t m[5] = {};
  m[0] = reader.Next(bits);
  const std::uint32_t t01 = reader.Next(2);
  m[1] = reader.Next(bits);
  std::uint32_t t23 = reader.Next(2);
  m[2] = reader.Next(bits);
  std::uint32_t t4 = reader.Next(1);
  m[3] = reader.Next(bits);
  std::uint32_t t56 = reader.Next(2);
  m[4] = reader.Next(bits);
  std::uint32_t t7 = reader.Next(1);

  // A partial block's missing values contribute no trit bits.
  if (count < 2)
    t23 = 0;
  if (count < 3)
    t4 = 0;
  if (count < 4)
    t56 = 0;
  if (count < 5)
    t7 = 0;

  std::uint32_t trits[5] = {};
  DecodeTrits((t7 << 7U) | (t56 << 5U) | (t4 << 4U) | (t23 << 2U) | t01, trits);
  for (unsigned index = 0; index < count; ++index) {
    dst[index].m = m[index];
    dst[index].tq = trits[index];
    dst[index].v = (trits[index] << bits) + m[index];
  }
}

void DecodeIseQuintBlock(IseValue *dst, unsigned count, BlockBitReader &reader,
                         unsigned bits) {
  std::uint32_t m[3] = {};
  m[0] = reader.Next(bits);
  const std::uint32_t q012 = reader.Next(3);
  m[1] = reader.Next(bits);
  std::uint32_t q34 = reader.Next(2);
  m[2] = reader.Next(bits);
  std::uint32_t q56 = reader.Next(2);

  if (count < 2)
    q34 = 0;
  if (count < 3)
    q56 = 0;

  std::uint32_t quints[3] = {};
  DecodeQuints((q56 << 5U) | (q34 << 3U) | q012, quints);
  for (unsigned index = 0; index < count; ++index) {
    dst[index].m = m[index];
    dst[index].tq = quints[index];
    dst[index].v = (quints[index] << bits) + m[index];
  }
}

void DecodeIse(IseValue *dst, unsigned count, BlockBitReader &reader,
               const IseParams &params) {
  switch (params.mode) {
  case IseMode::kTrit: {
    const unsigned blocks = (count + 4U) / 5U;
    for (unsigned block = 0; block < blocks; ++block) {
      const unsigned in_block =
          block == blocks - 1U ? count - 5U * (blocks - 1U) : 5U;
      DecodeIseTritBlock(&dst[5U * block], in_block, reader, params.bits);
    }
    return;
  }
  case IseMode::kQuint: {
    const unsigned blocks = (count + 2U) / 3U;
    for (unsigned block = 0; block < blocks; ++block) {
      const unsigned in_block =
          block == blocks - 1U ? count - 3U * (blocks - 1U) : 3U;
      DecodeIseQuintBlock(&dst[3U * block], in_block, reader, params.bits);
    }
    return;
  }
  case IseMode::kPlainBit:
    break;
  }
  for (unsigned index = 0; index < count; ++index) {
    dst[index].m = reader.Next(params.bits);
    dst[index].v = dst[index].m;
  }
}

/*
 * The block mode: bits [10:0] say how large the weight grid is, what range
 * its weights are stored in, and whether the block carries a second plane.
 * Several encodings are reserved; a block that uses one has no defined
 * contents, so the mode reports an error and the block becomes error colour.
 */
struct BlockMode {
  bool error = true;
  bool void_extent = false;
  bool dual_plane = false;
  unsigned weight_width = 0;
  unsigned weight_height = 0;
  IseParams weights;
};

BlockMode DecodeBlockMode(std::uint32_t mode_bits) {
  const auto bits = [mode_bits](unsigned first, unsigned last) {
    return (mode_bits >> first) & ((1U << (last - first + 1U)) - 1U);
  };
  const auto bit = [mode_bits](unsigned index) {
    return (mode_bits >> index) & 1U;
  };

  BlockMode mode;
  if (bits(0, 8) == 0x1fcU) {
    mode.error = false;
    mode.void_extent = true;
    return mode;
  }
  if ((bits(0, 1) == 0U && bits(6, 8) == 7U) || bits(0, 3) == 0U)
    return mode;  // Reserved encoding.

  std::uint32_t r = 0;
  if (bits(0, 1) == 0U) {
    r = (bit(3) << 2U) | (bit(2) << 1U) | bit(4);
    const std::uint32_t high = bits(7, 8);
    if (high == 3U) {
      mode.weight_width = bit(5) != 0U ? 10U : 6U;
      mode.weight_height = bit(5) != 0U ? 6U : 10U;
    } else {
      const std::uint32_t a = bits(5, 6);
      switch (high) {
      case 0:
        mode.weight_width = 12;
        mode.weight_height = a + 2U;
        break;
      case 1:
        mode.weight_width = a + 2U;
        mode.weight_height = 12;
        break;
      default:
        mode.weight_width = a + 6U;
        mode.weight_height = bits(9, 10) + 6U;
        break;
      }
    }
  } else {
    r = (bit(1) << 2U) | (bit(0) << 1U) | bit(4);
    const std::uint32_t low = bits(2, 3);
    const std::uint32_t a = bits(5, 6);
    if (low == 3U) {
      const std::uint32_t b = bit(7);
      const bool swapped = bit(8) != 0U;
      mode.weight_width = swapped ? b + 2U : a + 2U;
      mode.weight_height = swapped ? a + 2U : b + 6U;
    } else {
      const std::uint32_t b = bits(7, 8);
      switch (low) {
      case 0:
        mode.weight_width = b + 4U;
        mode.weight_height = a + 2U;
        break;
      case 1:
        mode.weight_width = b + 8U;
        mode.weight_height = a + 2U;
        break;
      default:
        mode.weight_width = a + 2U;
        mode.weight_height = b + 8U;
        break;
      }
    }
  }

  // The 12x2 and 2x12 grids spend bits 9 and 10 on the grid itself, so those
  // modes carry neither the high range bit nor a second plane.
  const bool grid_took_high_bits = bits(0, 1) == 0U && bits(7, 8) == 2U;
  const bool high_range = grid_took_high_bits ? false : bit(9) != 0U;
  mode.dual_plane = grid_took_high_bits ? false : bit(10) != 0U;

  // r selects the weight range; the high bit shifts the whole table up.
  if (high_range) {
    switch (r) {
    case 2: mode.weights = {IseMode::kQuint, 1}; break;
    case 3: mode.weights = {IseMode::kTrit, 2}; break;
    case 4: mode.weights = {IseMode::kPlainBit, 4}; break;
    case 5: mode.weights = {IseMode::kQuint, 2}; break;
    case 6: mode.weights = {IseMode::kTrit, 3}; break;
    case 7: mode.weights = {IseMode::kPlainBit, 5}; break;
    default: return mode;  // r < 2 is the reserved encoding rejected above.
    }
  } else {
    switch (r) {
    case 2: mode.weights = {IseMode::kPlainBit, 1}; break;
    case 3: mode.weights = {IseMode::kTrit, 0}; break;
    case 4: mode.weights = {IseMode::kPlainBit, 2}; break;
    case 5: mode.weights = {IseMode::kQuint, 0}; break;
    case 6: mode.weights = {IseMode::kTrit, 1}; break;
    case 7: mode.weights = {IseMode::kPlainBit, 3}; break;
    default: return mode;
    }
  }

  mode.error = false;
  return mode;
}

unsigned WeightCount(const BlockMode &mode) {
  return mode.weight_width * mode.weight_height * (mode.dual_plane ? 2U : 1U);
}

// A colour endpoint mode's class (mode/4) says how many values it reads.
unsigned EndpointValueCount(std::uint32_t endpoint_mode) {
  return (endpoint_mode / 4U + 1U) * 2U;
}

bool EndpointModeIsHdr(std::uint32_t endpoint_mode) {
  return endpoint_mode == 2U || endpoint_mode == 3U || endpoint_mode == 7U ||
         endpoint_mode == 11U || endpoint_mode == 14U || endpoint_mode == 15U;
}

// The colour endpoint modes.  One partition states its mode outright; more
// than one packs a class shared by all partitions plus two low bits each,
// and the low bits that do not fit beside the class spill into the bits just
// above the weight data.
void DecodeEndpointModes(std::uint32_t *modes, const std::uint8_t block[16],
                         unsigned partitions, int extra_bits_start) {
  if (partitions == 1U) {
    modes[0] = BlockBits(block, 13, 16);
    return;
  }
  const std::uint32_t selector = BlockBits(block, 23, 24);
  if (selector == 0U) {
    const std::uint32_t shared = BlockBits(block, 25, 28);
    for (unsigned index = 0; index < partitions; ++index)
      modes[index] = shared;
    return;
  }
  for (unsigned index = 0; index < partitions; ++index) {
    const std::uint32_t mode_class =
        selector - (BlockBits(block, 25 + index, 25 + index) != 0U ? 0U : 1U);
    const unsigned low0_index = partitions + 2U * index;
    const unsigned low1_index = partitions + 2U * index + 1U;
    const auto spilled = [&](unsigned bit_index) {
      const unsigned position =
          bit_index < 4U
              ? 25U + bit_index
              : static_cast<unsigned>(extra_bits_start) + bit_index - 4U;
      return BlockBits(block, position, position);
    };
    modes[index] =
        (mode_class << 2U) | (spilled(low1_index) << 1U) | spilled(low0_index);
  }
}

// Unquantise the colour endpoint values to eight bits.  A trit or quint
// range multiplies the trit/quint by a range-specific constant and folds the
// low bits back in; a plain-bit range simply replicates.
void UnquantiseEndpoints(std::uint32_t *dst, const IseValue *values,
                         unsigned count, const IseParams &params) {
  if (params.mode == IseMode::kPlainBit) {
    for (unsigned index = 0; index < count; ++index)
      dst[index] = BitReplicate(values[index].v, params.bits, 8);
    return;
  }

  const unsigned range =
      params.bits * 2U - (params.mode == IseMode::kTrit ? 2U : 1U);
  static constexpr std::uint32_t kRangeConstant[11] = {204, 113, 93, 54, 44, 26,
                                                       22,  13,  11, 6,  5};
  const std::uint32_t constant = kRangeConstant[range];

  for (unsigned index = 0; index < count; ++index) {
    const std::uint32_t m = values[index].m;
    const std::uint32_t a = (m >> 0U) & 1U;
    const std::uint32_t b = (m >> 1U) & 1U;
    const std::uint32_t c = (m >> 2U) & 1U;
    const std::uint32_t d = (m >> 3U) & 1U;
    const std::uint32_t e = (m >> 4U) & 1U;
    const std::uint32_t f = (m >> 5U) & 1U;

    const std::uint32_t sign = a == 0U ? 0U : (1U << 9U) - 1U;
    std::uint32_t offset = 0;
    switch (range) {
    case 0:
    case 1:
      offset = 0;
      break;
    case 2:
      offset = (b << 8U) | (b << 4U) | (b << 2U) | (b << 1U);
      break;
    case 3:
      offset = (b << 8U) | (b << 3U) | (b << 2U);
      break;
    case 4:
      offset = (c << 8U) | (b << 7U) | (c << 3U) | (b << 2U) | (c << 1U) | b;
      break;
    case 5:
      offset = (c << 8U) | (b << 7U) | (c << 2U) | (b << 1U) | c;
      break;
    case 6:
      offset = (d << 8U) | (c << 7U) | (b << 6U) | (d << 2U) | (c << 1U) | b;
      break;
    case 7:
      offset = (d << 8U) | (c << 7U) | (b << 6U) | (d << 1U) | c;
      break;
    case 8:
      offset = (e << 8U) | (d << 7U) | (c << 6U) | (b << 5U) | (e << 1U) | d;
      break;
    case 9:
      offset = (e << 8U) | (d << 7U) | (c << 6U) | (b << 5U) | e;
      break;
    default:
      offset = (f << 8U) | (e << 7U) | (d << 6U) | (c << 5U) | (b << 4U) | f;
      break;
    }

    dst[index] = (((values[index].tq * constant + offset) ^ sign) >> 2U) |
                 (sign & 0x80U);
  }
}

// Unquantise the weights to the 0..64 scale the interpolation uses.
void UnquantiseWeights(std::uint32_t *dst, const IseValue *values,
                       unsigned count, const IseParams &params) {
  if (params.mode == IseMode::kPlainBit) {
    for (unsigned index = 0; index < count; ++index)
      dst[index] = BitReplicate(values[index].v, params.bits, 6);
  } else {
    const unsigned range =
        params.bits * 2U + (params.mode == IseMode::kQuint ? 1U : 0U);
    if (range < 2U) {
      // A bare trit or quint is small enough to state outright.
      static constexpr std::uint32_t kTritMap[3] = {0, 32, 63};
      static constexpr std::uint32_t kQuintMap[5] = {0, 16, 32, 47, 63};
      for (unsigned index = 0; index < count; ++index) {
        dst[index] = range == 0U ? kTritMap[values[index].v % 3U]
                                 : kQuintMap[values[index].v % 5U];
      }
    } else {
      static constexpr std::uint32_t kRangeConstant[5] = {50, 28, 23, 13, 11};
      const std::uint32_t constant = kRangeConstant[range - 2U];
      for (unsigned index = 0; index < count; ++index) {
        const std::uint32_t m = values[index].m;
        const std::uint32_t a = (m >> 0U) & 1U;
        const std::uint32_t b = (m >> 1U) & 1U;
        const std::uint32_t c = (m >> 2U) & 1U;
        const std::uint32_t sign = a == 0U ? 0U : (1U << 7U) - 1U;
        std::uint32_t offset = 0;
        switch (range) {
        case 2:
        case 3:
          offset = 0;
          break;
        case 4:
          offset = (b << 6U) | (b << 2U) | b;
          break;
        case 5:
          offset = (b << 6U) | (b << 1U);
          break;
        default:
          offset = (c << 6U) | (b << 5U) | (c << 1U) | b;
          break;
        }
        dst[index] = (((values[index].tq * constant + offset) ^ sign) >> 2U) |
                     (sign & 0x20U);
      }
    }
  }

  // The scale is 0..64 but the encoding only reaches 63, so the top half is
  // stretched by one to put the midpoint exactly at 32.
  for (unsigned index = 0; index < count; ++index)
    dst[index] += dst[index] > 32U ? 1U : 0U;
}

// A weight per texel out of the weight grid, which is generally smaller than
// the footprint.  The grid is stretched over the block in 1/16ths and read
// bilinearly; at the last texel of each axis the fraction is zero, so the
// out-of-grid neighbours it names always carry weight zero.
struct TexelWeights {
  std::uint32_t plane[2] = {0, 0};
};

void InfillWeights(TexelWeights *dst, const std::uint32_t *weights,
                   unsigned block_width, unsigned block_height,
                   const BlockMode &mode) {
  const unsigned planes = mode.dual_plane ? 2U : 1U;
  const std::uint32_t scale_x = (1024U + block_width / 2U) / (block_width - 1U);
  const std::uint32_t scale_y =
      (1024U + block_height / 2U) / (block_height - 1U);

  for (unsigned texel_y = 0; texel_y < block_height; ++texel_y) {
    for (unsigned texel_x = 0; texel_x < block_width; ++texel_x) {
      const std::uint32_t grid_x =
          (scale_x * texel_x * (mode.weight_width - 1U) + 32U) >> 6U;
      const std::uint32_t grid_y =
          (scale_y * texel_y * (mode.weight_height - 1U) + 32U) >> 6U;
      const std::uint32_t left = grid_x >> 4U;
      const std::uint32_t top = grid_y >> 4U;
      const std::uint32_t fraction_x = grid_x & 0xfU;
      const std::uint32_t fraction_y = grid_y & 0xfU;

      const std::uint32_t weight_11 = (fraction_x * fraction_y + 8U) >> 4U;
      const std::uint32_t weight_10 = fraction_y - weight_11;
      const std::uint32_t weight_01 = fraction_x - weight_11;
      const std::uint32_t weight_00 = 16U - fraction_x - fraction_y + weight_11;

      const std::uint32_t index_00 = top * mode.weight_width + left;
      const std::uint32_t index_01 = index_00 + 1U;
      const std::uint32_t index_10 = index_00 + mode.weight_width;
      const std::uint32_t index_11 = index_10 + 1U;

      for (unsigned plane = 0; plane < planes; ++plane) {
        // The neighbour indices can leave the grid, but only where their
        // weight is zero, so wrapping them into the array is safe.
        const auto at = [&](std::uint32_t grid_index) {
          return weights[(grid_index * planes + plane) & 0x3fU];
        };
        dst[texel_y * block_width + texel_x].plane[plane] =
            (at(index_00) * weight_00 + at(index_01) * weight_01 +
             at(index_10) * weight_10 + at(index_11) * weight_11 + 8U) >>
            4U;
      }
    }
  }
}

std::int32_t ClampChannel(std::int32_t value) {
  return std::min(std::max(value, 0), 255);
}

// The two values of a signed endpoint pair share a byte: the high bit of the
// second belongs to the first, and the first is a six-bit two's-complement
// delta.
void BitTransferSigned(std::int32_t &value, std::int32_t &base) {
  base >>= 1;
  base |= value & 0x80;
  value >>= 1;
  value &= 0x3f;
  if ((value & 0x20) != 0)
    value -= 0x40;
}

struct Endpoint {
  std::int32_t channel[4] = {0, 0, 0, 255};
};

// Blue contraction: an endpoint pair whose sum runs the wrong way is stored
// with red and green folded towards blue, and is expanded back here.
Endpoint BlueContract(std::int32_t r, std::int32_t g, std::int32_t b,
                      std::int32_t a) {
  Endpoint result;
  result.channel[0] = (r + b) >> 1;
  result.channel[1] = (g + b) >> 1;
  result.channel[2] = b;
  result.channel[3] = a;
  return result;
}

Endpoint Clamped(const Endpoint &endpoint) {
  Endpoint result;
  for (unsigned index = 0; index < 4; ++index)
    result.channel[index] = ClampChannel(endpoint.channel[index]);
  return result;
}

Endpoint MakeEndpoint(std::int32_t r, std::int32_t g, std::int32_t b,
                      std::int32_t a) {
  Endpoint result;
  result.channel[0] = r;
  result.channel[1] = g;
  result.channel[2] = b;
  result.channel[3] = a;
  return result;
}

struct EndpointPair {
  Endpoint low;
  Endpoint high;
};

/*
 * The LDR colour endpoint modes.  Six of the sixteen modes are HDR; in LDR
 * decode mode their partitions produce the error colour, so this decodes
 * their values no further than counting them.
 */
void DecodeEndpoints(EndpointPair *dst, const std::uint32_t *values,
                     const std::uint32_t *modes, unsigned partitions) {
  unsigned consumed = 0;
  for (unsigned partition = 0; partition < partitions; ++partition) {
    const std::uint32_t mode = modes[partition];
    const std::uint32_t *v = &values[consumed];
    consumed += EndpointValueCount(mode);
    EndpointPair &pair = dst[partition];
    pair = EndpointPair{};

    switch (mode) {
    case 0:
      pair.low = MakeEndpoint(v[0], v[0], v[0], 255);
      pair.high = MakeEndpoint(v[1], v[1], v[1], 255);
      break;

    case 1: {
      const std::uint32_t low = (v[0] >> 2U) | (((v[1] >> 6U) & 3U) << 6U);
      const std::uint32_t high =
          std::min<std::uint32_t>(255U, low + (v[1] & 0x3fU));
      pair.low = MakeEndpoint(low, low, low, 255);
      pair.high = MakeEndpoint(high, high, high, 255);
      break;
    }

    case 4:
      pair.low = MakeEndpoint(v[0], v[0], v[0], v[2]);
      pair.high = MakeEndpoint(v[1], v[1], v[1], v[3]);
      break;

    case 5: {
      std::int32_t v0 = static_cast<std::int32_t>(v[0]);
      std::int32_t v1 = static_cast<std::int32_t>(v[1]);
      std::int32_t v2 = static_cast<std::int32_t>(v[2]);
      std::int32_t v3 = static_cast<std::int32_t>(v[3]);
      BitTransferSigned(v1, v0);
      BitTransferSigned(v3, v2);
      pair.low = Clamped(MakeEndpoint(v0, v0, v0, v2));
      pair.high = Clamped(MakeEndpoint(v0 + v1, v0 + v1, v0 + v1, v2 + v3));
      break;
    }

    case 6:
      pair.low = MakeEndpoint((v[0] * v[3]) >> 8U, (v[1] * v[3]) >> 8U,
                              (v[2] * v[3]) >> 8U, 255);
      pair.high = MakeEndpoint(v[0], v[1], v[2], 255);
      break;

    case 8:
      if (v[1] + v[3] + v[5] >= v[0] + v[2] + v[4]) {
        pair.low = MakeEndpoint(v[0], v[2], v[4], 255);
        pair.high = MakeEndpoint(v[1], v[3], v[5], 255);
      } else {
        pair.low = BlueContract(v[1], v[3], v[5], 255);
        pair.high = BlueContract(v[0], v[2], v[4], 255);
      }
      break;

    case 9: {
      std::int32_t v0 = static_cast<std::int32_t>(v[0]);
      std::int32_t v1 = static_cast<std::int32_t>(v[1]);
      std::int32_t v2 = static_cast<std::int32_t>(v[2]);
      std::int32_t v3 = static_cast<std::int32_t>(v[3]);
      std::int32_t v4 = static_cast<std::int32_t>(v[4]);
      std::int32_t v5 = static_cast<std::int32_t>(v[5]);
      BitTransferSigned(v1, v0);
      BitTransferSigned(v3, v2);
      BitTransferSigned(v5, v4);
      if (v1 + v3 + v5 >= 0) {
        pair.low = Clamped(MakeEndpoint(v0, v2, v4, 255));
        pair.high = Clamped(MakeEndpoint(v0 + v1, v2 + v3, v4 + v5, 255));
      } else {
        pair.low = Clamped(BlueContract(v0 + v1, v2 + v3, v4 + v5, 255));
        pair.high = Clamped(BlueContract(v0, v2, v4, 255));
      }
      break;
    }

    case 10:
      pair.low = MakeEndpoint((v[0] * v[3]) >> 8U, (v[1] * v[3]) >> 8U,
                              (v[2] * v[3]) >> 8U, v[4]);
      pair.high = MakeEndpoint(v[0], v[1], v[2], v[5]);
      break;

    case 12:
      if (v[1] + v[3] + v[5] >= v[0] + v[2] + v[4]) {
        pair.low = MakeEndpoint(v[0], v[2], v[4], v[6]);
        pair.high = MakeEndpoint(v[1], v[3], v[5], v[7]);
      } else {
        pair.low = Clamped(BlueContract(v[1], v[3], v[5], v[7]));
        pair.high = Clamped(BlueContract(v[0], v[2], v[4], v[6]));
      }
      break;

    case 13: {
      std::int32_t v0 = static_cast<std::int32_t>(v[0]);
      std::int32_t v1 = static_cast<std::int32_t>(v[1]);
      std::int32_t v2 = static_cast<std::int32_t>(v[2]);
      std::int32_t v3 = static_cast<std::int32_t>(v[3]);
      std::int32_t v4 = static_cast<std::int32_t>(v[4]);
      std::int32_t v5 = static_cast<std::int32_t>(v[5]);
      std::int32_t v6 = static_cast<std::int32_t>(v[6]);
      std::int32_t v7 = static_cast<std::int32_t>(v[7]);
      BitTransferSigned(v1, v0);
      BitTransferSigned(v3, v2);
      BitTransferSigned(v5, v4);
      BitTransferSigned(v7, v6);
      if (v1 + v3 + v5 >= 0) {
        pair.low = Clamped(MakeEndpoint(v0, v2, v4, v6));
        pair.high = Clamped(MakeEndpoint(v0 + v1, v2 + v3, v4 + v5, v6 + v7));
      } else {
        pair.low = Clamped(BlueContract(v0 + v1, v2 + v3, v4 + v5, v6 + v7));
        pair.high = Clamped(BlueContract(v0, v2, v4, v6));
      }
      break;
    }

    default:
      // Modes 2, 3, 7, 11, 14 and 15 are HDR.  In LDR decode mode their
      // texels are the error colour, so their values are never read.
      break;
    }
  }
}

// The partition pattern generator.  Which partition a texel belongs to is
// not stored: it is computed from the ten-bit seed by this hash, so a block
// spends ten bits where a per-texel table would cost far more.
std::uint32_t Hash52(std::uint32_t value) {
  std::uint32_t p = value;
  p ^= p >> 15U;
  p -= p << 17U;
  p += p << 7U;
  p += p << 4U;
  p ^= p >> 5U;
  p += p << 16U;
  p ^= p >> 7U;
  p ^= p >> 3U;
  p ^= p << 6U;
  p ^= p >> 17U;
  return p;
}

unsigned TexelPartition(std::uint32_t seed_in, std::uint32_t texel_x,
                        std::uint32_t texel_y, unsigned partitions,
                        bool small_block) {
  // A small footprint doubles its coordinates so that the same patterns
  // cover it at the same scale as a large one.
  const std::uint32_t x = small_block ? texel_x << 1U : texel_x;
  const std::uint32_t y = small_block ? texel_y << 1U : texel_y;
  const std::uint32_t seed = seed_in + 1024U * (partitions - 1U);
  const std::uint32_t random = Hash52(seed);

  // Eight of the twelve seeds the generator defines; the other four only
  // ever multiply the third coordinate, which a two-dimensional footprint
  // does not have.
  std::uint32_t seeds[8] = {
      (random >> 0U) & 0xfU,  (random >> 4U) & 0xfU,  (random >> 8U) & 0xfU,
      (random >> 12U) & 0xfU, (random >> 16U) & 0xfU, (random >> 20U) & 0xfU,
      (random >> 24U) & 0xfU, (random >> 28U) & 0xfU,
  };
  for (std::uint32_t &value : seeds)
    value = (value * value) & 0xffU;

  const unsigned shift_a = (seed & 2U) != 0U ? 4U : 5U;
  const unsigned shift_b = partitions == 3U ? 6U : 5U;
  const unsigned shift_1 = (seed & 1U) != 0U ? shift_a : shift_b;
  const unsigned shift_2 = (seed & 1U) != 0U ? shift_b : shift_a;

  seeds[0] >>= shift_1;
  seeds[1] >>= shift_2;
  seeds[2] >>= shift_1;
  seeds[3] >>= shift_2;
  seeds[4] >>= shift_1;
  seeds[5] >>= shift_2;
  seeds[6] >>= shift_1;
  seeds[7] >>= shift_2;

  const std::uint32_t a =
      0x3fU & (seeds[0] * x + seeds[1] * y + (random >> 14U));
  const std::uint32_t b =
      0x3fU & (seeds[2] * x + seeds[3] * y + (random >> 10U));
  const std::uint32_t c =
      partitions >= 3U ? 0x3fU & (seeds[4] * x + seeds[5] * y + (random >> 6U))
                       : 0U;
  const std::uint32_t d =
      partitions >= 4U ? 0x3fU & (seeds[6] * x + seeds[7] * y + (random >> 2U))
                       : 0U;

  if (a >= b && a >= c && a >= d)
    return 0;
  if (b >= c && b >= d)
    return 1;
  return c >= d ? 2U : 3U;
}

// The ASTC error colour: opaque magenta, in whichever encoding the image
// stores.  sRGB stores it directly, so both come out as 0xff, 0, 0xff, 0xff.
constexpr std::array<std::uint8_t, 4> kErrorColour = {0xff, 0x00, 0xff, 0xff};

void FillErrorColour(AstcDecodedBlock *out) {
  for (std::uint32_t index = 0; index < out->footprint.texel_count(); ++index)
    out->texels[index] = kErrorColour;
  out->error_colour = true;
}

/*
 * The 16-bit interpolation result as an eight-bit texel.
 *
 * The LDR decode defines the linear result as a value in [0,1]: 0xffff is
 * 1.0 and everything else is c/65536.  An eight-bit output takes the top
 * eight bits of that result, for linear and sRGB alike: this is what Mesa's
 * CPU decoder (util/texcompress_astc.cpp Block::write_decoded, output_unorm8)
 * stores, and llvmpipe samples that decode because it advertises no ASTC
 * support.  Rounding ((c * 255 + 32768) >> 16) instead is one-sided in the
 * low range -- for texels Mesa stores as <= 8 it is +1 in half of the
 * interpolated cases and never lower -- and flips alpha tests against the
 * llvmpipe reference at soft leaf edges (GL5 Draw 185).  0xffff still maps
 * to 0xff, and the sRGB endpoints were expanded with 0x80 in the low byte
 * precisely so that their top byte is the encoded value.
 */
std::uint8_t ResultToUnorm8(std::uint32_t value, bool srgb) {
  (void)srgb;
  return static_cast<std::uint8_t>((value & 0xff00U) >> 8U);
}

bool DecodeVoidExtent(const std::uint8_t block[16], bool srgb,
                      AstcDecodedBlock *out) {
  const std::uint32_t min_s = BlockBits(block, 12, 24);
  const std::uint32_t max_s = BlockBits(block, 25, 37);
  const std::uint32_t min_t = BlockBits(block, 38, 50);
  const std::uint32_t max_t = BlockBits(block, 51, 63);
  const bool no_extent = min_s == 0x1fffU && max_s == 0x1fffU &&
                         min_t == 0x1fffU && max_t == 0x1fffU;
  const bool hdr = BlockBits(block, 9, 9) != 0U;

  /*
   * Bits 10 and 11 are reserved and an encoder sets both, but the decode has
   * no case for them: the spec's error conditions for a void extent are the
   * two below and nothing else, so a block with them clear still names a
   * colour and this returns it.
   *
   * An HDR void extent has no LDR meaning, and an extent whose minimum is
   * not below its maximum describes no region at all.
   */
  if (hdr || (!no_extent && (min_s >= max_s || min_t >= max_t))) {
    FillErrorColour(out);
    return true;
  }

  const std::array<std::uint32_t, 4> channels = {
      BlockBits(block, 64, 79),
      BlockBits(block, 80, 95),
      BlockBits(block, 96, 111),
      BlockBits(block, 112, 127),
  };
  std::array<std::uint8_t, 4> texel{};
  for (std::size_t channel = 0; channel < 4; ++channel)
    texel[channel] = ResultToUnorm8(channels[channel], srgb);
  for (std::uint32_t index = 0; index < out->footprint.texel_count(); ++index)
    out->texels[index] = texel;
  return true;
}

} // namespace

AstcBlockFootprint AstcFootprintForRogueFormat(std::uint32_t rogue_format) {
  constexpr std::size_t count =
      sizeof(kRogueAstcFootprints) / sizeof(kRogueAstcFootprints[0]);
  if (rogue_format >= count)
    return AstcBlockFootprint{};
  return kRogueAstcFootprints[rogue_format];
}

bool DecodeAstcBlock(const std::uint8_t block[16],
                     AstcBlockFootprint footprint,
                     bool srgb,
                     AstcDecodedBlock *out,
                     const char **out_refusal) {
  static const char *ignored = nullptr;
  if (out_refusal == nullptr)
    out_refusal = &ignored;
  *out_refusal = nullptr;

  if (block == nullptr || out == nullptr) {
    *out_refusal = "missing block or destination";
    return false;
  }
  if (!footprint.valid() || footprint.texel_count() > out->texels.size()) {
    *out_refusal = "block footprint is not one this decoder was given";
    return false;
  }

  out->footprint = footprint;
  out->texels = {};
  out->error_colour = false;

  const unsigned block_width = footprint.width;
  const unsigned block_height = footprint.height;

  const BlockMode mode = DecodeBlockMode(BlockBits(block, 0, 10));
  if (mode.error) {
    FillErrorColour(out);
    return true;
  }
  if (mode.void_extent)
    return DecodeVoidExtent(block, srgb, out);

  const unsigned weight_count = WeightCount(mode);
  const unsigned weight_bits = IseSequenceBits(mode.weights, weight_count);
  const unsigned partitions = BlockBits(block, 11, 12) + 1U;

  /*
   * The weight grid has to fit inside the footprint, its sequence has to fit
   * the space the format reserves for it, and a four-partition block has no
   * room for a second plane.  A block that breaks any of these is not a
   * block this format can express.
   */
  if (weight_count > kMaxWeights || weight_bits > 96U || weight_bits < 24U ||
      mode.weight_width > block_width || mode.weight_height > block_height ||
      (partitions == 4U && mode.dual_plane)) {
    FillErrorColour(out);
    return true;
  }

  // What is left of the 128 bits after the weights and the configuration is
  // what the colour endpoints have to fit into.
  const bool single_cem = partitions == 1U || BlockBits(block, 23, 24) == 0U;
  const unsigned config_bits =
      (partitions == 1U ? 17U : single_cem ? 29U : 25U + 3U * partitions) +
      (mode.dual_plane ? 2U : 0U);
  const int endpoint_bits =
      128 - static_cast<int>(weight_bits) - static_cast<int>(config_bits);
  const int extra_bits_start =
      127 - static_cast<int>(weight_bits) -
      (single_cem ? -1 : partitions == 4U ? 7 : partitions == 3U ? 4 : 1);

  std::uint32_t endpoint_modes[kMaxPartitions] = {};
  DecodeEndpointModes(endpoint_modes, block, partitions, extra_bits_start);

  unsigned endpoint_value_count = 0;
  for (unsigned partition = 0; partition < partitions; ++partition)
    endpoint_value_count += EndpointValueCount(endpoint_modes[partition]);

  // Eighteen values is the format's ceiling, and each value needs at least
  // 13/5 bits even in the narrowest range there is.
  if (endpoint_value_count > kMaxEndpointValues || endpoint_bits <= 0 ||
      static_cast<unsigned>(endpoint_bits) <
          (13U * endpoint_value_count + 4U) / 5U) {
    FillErrorColour(out);
    return true;
  }

  EndpointPair endpoints[kMaxPartitions];
  {
    const IseParams params = IseParamsForBudget(
        static_cast<unsigned>(endpoint_bits), endpoint_value_count);
    IseValue encoded[kMaxEndpointValues] = {};
    BlockBitReader reader(block, partitions == 1U ? 17 : 29, endpoint_bits,
                          /*forward=*/true);
    DecodeIse(encoded, endpoint_value_count, reader, params);
    std::uint32_t unquantised[kMaxEndpointValues] = {};
    UnquantiseEndpoints(unquantised, encoded, endpoint_value_count, params);
    DecodeEndpoints(endpoints, unquantised, endpoint_modes, partitions);
  }

  TexelWeights texel_weights[kMaxTexels];
  {
    IseValue encoded[kMaxWeights] = {};
    BlockBitReader reader(block, 127, static_cast<int>(weight_bits),
                          /*forward=*/false);
    DecodeIse(encoded, weight_count, reader, mode.weights);
    std::uint32_t unquantised[kMaxWeights] = {};
    UnquantiseWeights(unquantised, encoded, weight_count, mode.weights);
    InfillWeights(texel_weights, unquantised, block_width, block_height, mode);
  }

  // The colour component selector names the channel the second plane drives.
  const int selected_component =
      mode.dual_plane
          ? static_cast<int>(
                BlockBits(block, static_cast<unsigned>(extra_bits_start - 2),
                          static_cast<unsigned>(extra_bits_start - 1)))
          : -1;
  const std::uint32_t partition_seed =
      partitions > 1U ? BlockBits(block, 13, 22) : 0U;
  const bool small_block = block_width * block_height < 31U;

  bool hdr_partition[kMaxPartitions] = {};
  for (unsigned partition = 0; partition < partitions; ++partition)
    hdr_partition[partition] = EndpointModeIsHdr(endpoint_modes[partition]);

  for (unsigned texel_y = 0; texel_y < block_height; ++texel_y) {
    for (unsigned texel_x = 0; texel_x < block_width; ++texel_x) {
      const unsigned texel_index = texel_y * block_width + texel_x;
      const unsigned partition =
          partitions == 1U ? 0U
                           : TexelPartition(partition_seed, texel_x, texel_y,
                                            partitions, small_block);

      // An HDR endpoint has no LDR value; the decode mode, not the block,
      // decides that, so this is the error colour rather than a refusal.
      if (hdr_partition[partition]) {
        out->texels[texel_index] = kErrorColour;
        out->error_colour = true;
        continue;
      }

      const Endpoint &low = endpoints[partition].low;
      const Endpoint &high = endpoints[partition].high;
      for (int channel = 0; channel < 4; ++channel) {
        const std::uint32_t e0 =
            static_cast<std::uint32_t>(low.channel[channel]);
        const std::uint32_t e1 =
            static_cast<std::uint32_t>(high.channel[channel]);
        const std::uint32_t c0 = (e0 << 8U) | (srgb ? 0x80U : e0);
        const std::uint32_t c1 = (e1 << 8U) | (srgb ? 0x80U : e1);
        const std::uint32_t weight =
            texel_weights[texel_index]
                .plane[selected_component == channel ? 1 : 0];
        const std::uint32_t result =
            (c0 * (64U - weight) + c1 * weight + 32U) / 64U;
        out->texels[texel_index][channel] = ResultToUnorm8(result, srgb);
      }
    }
  }

  return true;
}

} // namespace pvrgpu::stub
