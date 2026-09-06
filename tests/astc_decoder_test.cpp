// AstcDecoder 的單元測試。
//
// 區塊是在這裡照 ASTC 規格「組」出來的，不是從任何 capture 取來的固定
// 位元組。測試因此檢查的是解碼器讀懂了規格，而不是它記住了某個檔案。
#include "texture/astc_decoder.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int g_failures = 0;

void Check(bool condition, const std::string &what) {
  if (condition)
    return;
  std::fprintf(stderr, "astc_decoder_test: FAIL: %s\n", what.c_str());
  ++g_failures;
}

// 依 ASTC 規格把一個 LDR void-extent 區塊寫進 16 個 byte。
//
//   bits  8..0   = 0b111111100  區塊身分
//   bit      9   = 0            LDR
//   bits 11..10  = 0b11         保留位元，規格要求為 1
//   bits 63..12  = void extent 的座標範圍
//   bits 79..64  = R, 95..80 = G, 111..96 = B, 127..112 = A（皆 UNORM16）
std::array<std::uint8_t, 16> MakeVoidExtentBlock(std::uint16_t r,
                                                 std::uint16_t g,
                                                 std::uint16_t b,
                                                 std::uint16_t a,
                                                 bool hdr = false,
                                                 bool reserved_set = true) {
  std::array<std::uint8_t, 16> block{};
  const auto set_bits = [&block](unsigned first, unsigned last,
                                 std::uint32_t value) {
    for (unsigned bit = first; bit <= last; ++bit) {
      if ((value >> (bit - first)) & 1U)
        block[bit >> 3U] |= static_cast<std::uint8_t>(1U << (bit & 7U));
    }
  };
  set_bits(0, 8, 0x1fcU);
  set_bits(9, 9, hdr ? 1U : 0U);
  set_bits(10, 11, reserved_set ? 0x3U : 0x0U);
  // "無座標" 的 void extent：四個範圍欄位全為 1。
  set_bits(12, 24, 0x1fffU);
  set_bits(25, 37, 0x1fffU);
  set_bits(38, 50, 0x1fffU);
  set_bits(51, 63, 0x1fffU);
  set_bits(64, 79, r);
  set_bits(80, 95, g);
  set_bits(96, 111, b);
  set_bits(112, 127, a);
  return block;
}

} // namespace

int main() {
  using pvrgpu::stub::AstcBlockFootprint;
  using pvrgpu::stub::AstcDecodedBlock;
  using pvrgpu::stub::AstcFootprintForRogueFormat;
  using pvrgpu::stub::DecodeAstcBlock;

  // Rogue FORMAT_COMPRESSED 的 ASTC 值對應到 texstate.xml 宣告的 footprint。
  Check(AstcFootprintForRogueFormat(0).width == 4 &&
            AstcFootprintForRogueFormat(0).height == 4,
        "rogue format 0 is ASTC 4x4");
  Check(AstcFootprintForRogueFormat(5).width == 8 &&
            AstcFootprintForRogueFormat(5).height == 5,
        "rogue format 5 is ASTC 8x5");
  Check(AstcFootprintForRogueFormat(13).width == 12 &&
            AstcFootprintForRogueFormat(13).height == 12,
        "rogue format 13 is ASTC 12x12");
  Check(!AstcFootprintForRogueFormat(14).valid(),
        "rogue format 14 is not ASTC");
  Check(!AstcFootprintForRogueFormat(68).valid(),
        "rogue format 68 is ETC2, not ASTC");

  // 一個實心顏色的 void extent 覆蓋整個 footprint。
  {
    const auto block = MakeVoidExtentBlock(0xffff, 0x8000, 0x0000, 0xffff);
    AstcDecodedBlock decoded;
    const char *refusal = nullptr;
    const bool ok = DecodeAstcBlock(block.data(), AstcBlockFootprint{4, 4},
                                    false, &decoded, &refusal);
    Check(ok, std::string("4x4 void extent decodes") +
                  (refusal ? std::string(": ") + refusal : ""));
    if (ok) {
      Check(decoded.footprint.texel_count() == 16, "4x4 covers 16 texels");
      bool uniform = true;
      for (std::uint32_t index = 0; index < 16; ++index) {
        uniform = uniform && decoded.texels[index][0] == 255 &&
                  decoded.texels[index][1] == 128 &&
                  decoded.texels[index][2] == 0 &&
                  decoded.texels[index][3] == 255;
      }
      Check(uniform, "every texel is the block's colour");
    }
  }

  // UNORM16 到 UNORM8 是取最近值，不是截斷：0xffff 必須是 255，而
  // 0x0080 不能掉成 0。
  {
    const auto block = MakeVoidExtentBlock(0x0080, 0x00ff, 0x0100, 0x7fff);
    AstcDecodedBlock decoded;
    const char *refusal = nullptr;
    Check(DecodeAstcBlock(block.data(), AstcBlockFootprint{12, 12}, false,
                          &decoded, &refusal),
          "12x12 void extent decodes");
    Check(decoded.footprint.texel_count() == 144, "12x12 covers 144 texels");
    Check(decoded.texels[143][0] == 0, "0x0080 rounds to 0");
    Check(decoded.texels[143][1] == 1, "0x00ff rounds to 1");
    Check(decoded.texels[143][2] == 1, "0x0100 rounds to 1");
    // 32767/65535 * 255 = 127.498, so the nearest 8-bit value is 127.
    Check(decoded.texels[143][3] == 127, "0x7fff rounds to 127");
  }

  // 未實作的區塊型別要具名拒絕，不能猜一個顏色。
  {
    std::array<std::uint8_t, 16> block{};
    block[0] = 0x12;  // 低九位元不是 0x1fc
    AstcDecodedBlock decoded;
    const char *refusal = nullptr;
    Check(!DecodeAstcBlock(block.data(), AstcBlockFootprint{4, 4}, false,
                           &decoded, &refusal),
          "a weighted block is declined");
    Check(refusal != nullptr &&
              std::string(refusal).find("void-extent") != std::string::npos,
          "the refusal names the block type");
  }
  {
    const auto block = MakeVoidExtentBlock(0, 0, 0, 0xffff, /*hdr=*/true);
    AstcDecodedBlock decoded;
    const char *refusal = nullptr;
    Check(!DecodeAstcBlock(block.data(), AstcBlockFootprint{4, 4}, false,
                           &decoded, &refusal),
          "an HDR void extent is declined");
    Check(refusal != nullptr && std::string(refusal).find("HDR") !=
                                    std::string::npos,
          "the refusal names HDR");
  }
  {
    const auto block = MakeVoidExtentBlock(0, 0, 0, 0xffff, /*hdr=*/false,
                                           /*reserved_set=*/false);
    AstcDecodedBlock decoded;
    const char *refusal = nullptr;
    Check(!DecodeAstcBlock(block.data(), AstcBlockFootprint{4, 4}, false,
                           &decoded, &refusal),
          "clear reserved bits are declined");
  }
  // footprint 必須是解碼器被交代的那一個，而不是它自己猜的。
  {
    const auto block = MakeVoidExtentBlock(0xffff, 0xffff, 0xffff, 0xffff);
    AstcDecodedBlock decoded;
    const char *refusal = nullptr;
    Check(!DecodeAstcBlock(block.data(), AstcBlockFootprint{0, 0}, false,
                           &decoded, &refusal),
          "an empty footprint is declined");
  }

  if (g_failures != 0) {
    std::fprintf(stderr, "astc_decoder_test: %d failure(s)\n", g_failures);
    return EXIT_FAILURE;
  }
  std::printf("astc_decoder_test: all checks passed\n");
  return EXIT_SUCCESS;
}
