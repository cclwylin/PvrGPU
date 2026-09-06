#include "texture/astc_decoder.h"

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

// UNORM16 to UNORM8, rounded the way the ASTC LDR decode does: the result is
// the nearest 8-bit value, not a truncation, so 0xffff stays 255 and 0x0080
// does not fall to zero.
std::uint8_t Unorm16ToUnorm8(std::uint32_t value) {
  return static_cast<std::uint8_t>((value * 255U + 32895U) >> 16U);
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

  /*
   * A void-extent block is identified by its low nine bits.  Everything else
   * -- weight grids, partitions, dual planes -- is a block type this decoder
   * has not implemented, and saying so is the whole of its answer.
   */
  if (BlockBits(block, 0, 8) != 0x1fcU) {
    *out_refusal = "block is not a void-extent block";
    return false;
  }
  if (BlockBits(block, 9, 9) != 0U) {
    *out_refusal = "HDR void-extent block";
    return false;
  }
  /*
   * Bits 10 and 11 are reserved and the spec requires both set.  A block with
   * them clear is either corrupt or from a profile this does not implement;
   * either way it is not something to decode past.
   */
  if (BlockBits(block, 10, 11) != 0x3U) {
    *out_refusal = "void-extent reserved bits are not set";
    return false;
  }

  /*
   * The LDR void-extent colour is four UNORM16 channels.  sRGB is not decoded
   * here: the texture unit applies the transfer function after the texel
   * leaves this decoder, exactly as it does for an uncompressed sRGB image,
   * so this returns the stored channels either way.
   */
  (void)srgb;
  const std::array<std::uint32_t, 4> channels = {
      BlockBits(block, 64, 79),
      BlockBits(block, 80, 95),
      BlockBits(block, 96, 111),
      BlockBits(block, 112, 127),
  };

  out->footprint = footprint;
  out->texels = {};
  std::array<std::uint8_t, 4> texel{};
  for (std::size_t component = 0; component < 4; ++component)
    texel[component] = Unorm16ToUnorm8(channels[component]);
  for (std::uint32_t index = 0; index < footprint.texel_count(); ++index)
    out->texels[index] = texel;
  return true;
}

} // namespace pvrgpu::stub
