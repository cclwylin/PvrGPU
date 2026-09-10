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

void SetBits(std::array<std::uint8_t, 16> &block, unsigned first, unsigned last,
             std::uint32_t value) {
  for (unsigned bit = first; bit <= last; ++bit) {
    if ((value >> (bit - first)) & 1U)
      block[bit >> 3U] |= static_cast<std::uint8_t>(1U << (bit & 7U));
  }
}

// 依 ASTC 規格把一個 void-extent 區塊寫進 16 個 byte。
//
//   bits  8..0   = 0b111111100  區塊身分
//   bit      9   = 0            LDR
//   bits 11..10  = 0b11         保留位元，編碼器一律寫 1
//   bits 63..12  = void extent 的座標範圍
//   bits 79..64  = R, 95..80 = G, 111..96 = B, 127..112 = A（皆 UNORM16）
std::array<std::uint8_t, 16> MakeVoidExtentBlock(std::uint16_t r,
                                                 std::uint16_t g,
                                                 std::uint16_t b,
                                                 std::uint16_t a,
                                                 bool hdr = false,
                                                 bool reserved_set = true) {
  std::array<std::uint8_t, 16> block{};
  SetBits(block, 0, 8, 0x1fcU);
  SetBits(block, 9, 9, hdr ? 1U : 0U);
  SetBits(block, 10, 11, reserved_set ? 0x3U : 0x0U);
  // "無座標" 的 void extent：四個範圍欄位全為 1。
  SetBits(block, 12, 24, 0x1fffU);
  SetBits(block, 25, 37, 0x1fffU);
  SetBits(block, 38, 50, 0x1fffU);
  SetBits(block, 51, 63, 0x1fffU);
  SetBits(block, 64, 79, r);
  SetBits(block, 80, 95, g);
  SetBits(block, 96, 111, b);
  SetBits(block, 112, 127, a);
  return block;
}

/*
 * 一個 4x4、單一 partition、CEM 8（LDR RGB direct）的加權區塊。
 *
 * block mode 0x42 的欄位：bits[1:0]=0b10 所以走第二組編碼，bits[3:2]=0 選
 * 「寬 = b+4、高 = a+2」，a=bits[6:5]=2、b=bits[8:7]=0，得到 4x4 的權重格
 * ——和 footprint 一樣大，所以 infill 是一對一。r=(bit1<<2)|(bit0<<1)|bit4
 * = 4 且 bit9(h)=0，權重範圍就是 2 個純位元（0..3）；16 個權重共 32 bits，
 * 落在規格要求的 24..96 之間。
 *
 * 端點：1 個 partition 的 CEM 寫在 bits[16:13]，端點資料自 bit 17 起。權重
 * 用掉 32 bits、設定用掉 17 bits，端點還剩 79 bits 給 6 個值，最寬的範圍
 * 因此是 8 個純位元 —— 端點就是寫進去的那個 byte，不需要反量化。
 *
 * 權重從 bit 127 往下讀且每個欄位位元反轉，所以第 i 個權重的兩個位元是
 * bit[127-2i]（低位）與 bit[126-2i]（高位）。
 */
std::array<std::uint8_t, 16> MakeWeightedBlock(
    const std::array<std::uint8_t, 6> &endpoint_values,
    const std::array<std::uint8_t, 16> &weights) {
  std::array<std::uint8_t, 16> block{};
  SetBits(block, 0, 10, 0x42U);
  SetBits(block, 11, 12, 0U);   // 1 個 partition
  SetBits(block, 13, 16, 8U);   // CEM 8: LDR RGB direct
  for (unsigned value = 0; value < 6; ++value)
    SetBits(block, 17 + 8 * value, 24 + 8 * value, endpoint_values[value]);
  for (unsigned index = 0; index < 16; ++index) {
    SetBits(block, 127 - 2 * index, 127 - 2 * index, weights[index] & 1U);
    SetBits(block, 126 - 2 * index, 126 - 2 * index,
            (weights[index] >> 1U) & 1U);
  }
  return block;
}

bool TexelIs(const pvrgpu::stub::AstcDecodedBlock &decoded, unsigned index,
             int r, int g, int b, int a) {
  return decoded.texels[index][0] == r && decoded.texels[index][1] == g &&
         decoded.texels[index][2] == b && decoded.texels[index][3] == a;
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
      Check(!decoded.error_colour, "a valid void extent is not error colour");
      bool uniform = true;
      for (std::uint32_t index = 0; index < 16; ++index)
        uniform = uniform && TexelIs(decoded, index, 255, 128, 0, 255);
      Check(uniform, "every texel is the block's colour");
    }
  }

  // UNORM16 到 UNORM8 取高 8 位元（截斷），與 Mesa util/texcompress_astc.cpp
  // 的 output_unorm8 路徑（llvmpipe 取樣的 CPU decode）一致：0xffff 必須是
  // 255，0x00ff 是 0 而不是四捨五入成 1。
  {
    const auto block = MakeVoidExtentBlock(0x0080, 0x00ff, 0x0100, 0x7fff);
    AstcDecodedBlock decoded;
    const char *refusal = nullptr;
    Check(DecodeAstcBlock(block.data(), AstcBlockFootprint{12, 12}, false,
                          &decoded, &refusal),
          "12x12 void extent decodes");
    Check(decoded.footprint.texel_count() == 144, "12x12 covers 144 texels");
    Check(decoded.texels[143][0] == 0, "0x0080 truncates to 0");
    Check(decoded.texels[143][1] == 0, "0x00ff truncates to 0, not 1");
    Check(decoded.texels[143][2] == 1, "0x0100 truncates to 1");
    Check(decoded.texels[143][3] == 127, "0x7fff truncates to 127");
  }
  {
    const auto block = MakeVoidExtentBlock(0xffff, 0x01ff, 0x0000, 0xfeff);
    AstcDecodedBlock decoded;
    const char *refusal = nullptr;
    Check(DecodeAstcBlock(block.data(), AstcBlockFootprint{4, 4}, false,
                          &decoded, &refusal),
          "4x4 void extent decodes (truncation extremes)");
    Check(decoded.texels[15][0] == 255, "0xffff is 255");
    Check(decoded.texels[15][1] == 1, "0x01ff truncates to 1 (rounding gave 2)");
    Check(decoded.texels[15][2] == 0, "0x0000 is 0");
    Check(decoded.texels[15][3] == 254, "0xfeff truncates to 254 (rounding gave 255)");
  }

  // sRGB 的 void extent 取 UNORM16 的高 8 位元。
  {
    const auto block = MakeVoidExtentBlock(0xffff, 0x80ff, 0x0100, 0x7f00);
    AstcDecodedBlock decoded;
    Check(DecodeAstcBlock(block.data(), AstcBlockFootprint{4, 4}, true,
                          &decoded, nullptr),
          "sRGB void extent decodes");
    Check(TexelIs(decoded, 0, 255, 128, 1, 127),
          "sRGB void extent takes the high byte of each channel");
  }

  /*
   * 加權區塊。權重 0 的 texel 必須正好是低端點，權重 3（反量化後是滿刻度
   * 的 64）必須正好是高端點 —— 這一條同時驗到 block mode、端點解碼、ISE
   * 反量化與 infill；任何一段錯了，兩端就不會落在端點上。
   */
  {
    // CEM 8 的六個值是 r0,r1,g0,g1,b0,b1。
    const std::array<std::uint8_t, 6> endpoints = {0, 255, 0, 128, 0, 64};
    std::array<std::uint8_t, 16> weights{};
    for (unsigned index = 0; index < 16; ++index)
      weights[index] = index < 8 ? 0U : 3U;
    const auto block = MakeWeightedBlock(endpoints, weights);

    AstcDecodedBlock decoded;
    const char *refusal = nullptr;
    const bool ok = DecodeAstcBlock(block.data(), AstcBlockFootprint{4, 4},
                                    false, &decoded, &refusal);
    Check(ok, std::string("a weighted 4x4 block decodes") +
                  (refusal ? std::string(": ") + refusal : ""));
    Check(!decoded.error_colour, "a valid weighted block is not error colour");
    bool low_half = true;
    bool high_half = true;
    for (unsigned index = 0; index < 8; ++index)
      low_half = low_half && TexelIs(decoded, index, 0, 0, 0, 255);
    for (unsigned index = 8; index < 16; ++index)
      high_half = high_half && TexelIs(decoded, index, 255, 128, 64, 255);
    Check(low_half, "weight 0 gives the low endpoint");
    Check(high_half, "weight 3 gives the high endpoint");
  }

  // 同一個區塊在 sRGB 下走另一條端點展開（低位元組 0x80），端點本身仍是
  // 那個 byte。
  {
    const std::array<std::uint8_t, 6> endpoints = {0, 255, 0, 128, 0, 64};
    std::array<std::uint8_t, 16> weights{};
    for (unsigned index = 0; index < 16; ++index)
      weights[index] = 3U;
    const auto block = MakeWeightedBlock(endpoints, weights);
    AstcDecodedBlock decoded;
    Check(DecodeAstcBlock(block.data(), AstcBlockFootprint{4, 4}, true,
                          &decoded, nullptr),
          "a weighted sRGB block decodes");
    Check(TexelIs(decoded, 0, 255, 128, 64, 255),
          "sRGB weight 3 gives the high endpoint");
  }

  /*
   * 無效的編碼不是拒絕，是規格規定的 error colour（不透明洋紅）。硬體照樣
   * 會吐出 texel，模型也照樣吐。
   */
  {
    // block mode 的 bits[3:0] 全為 0 是保留編碼。
    std::array<std::uint8_t, 16> block{};
    AstcDecodedBlock decoded;
    const char *refusal = nullptr;
    Check(DecodeAstcBlock(block.data(), AstcBlockFootprint{4, 4}, false,
                          &decoded, &refusal),
          "a reserved block mode still decodes");
    Check(decoded.error_colour, "a reserved block mode is error colour");
    Check(TexelIs(decoded, 0, 255, 0, 255, 255),
          "the error colour is opaque magenta");
  }
  {
    const auto block = MakeVoidExtentBlock(0, 0, 0, 0xffff, /*hdr=*/true);
    AstcDecodedBlock decoded;
    Check(DecodeAstcBlock(block.data(), AstcBlockFootprint{4, 4}, false,
                          &decoded, nullptr),
          "an HDR void extent still decodes");
    Check(decoded.error_colour && TexelIs(decoded, 0, 255, 0, 255, 255),
          "an HDR void extent is error colour in LDR mode");
  }
  {
    // 保留位元 10、11 沒被設起來並不是規格列出的錯誤條件，區塊照樣有顏色。
    const auto block = MakeVoidExtentBlock(0xffff, 0, 0, 0xffff,
                                           /*hdr=*/false,
                                           /*reserved_set=*/false);
    AstcDecodedBlock decoded;
    Check(DecodeAstcBlock(block.data(), AstcBlockFootprint{4, 4}, false,
                          &decoded, nullptr),
          "clear reserved bits still decode");
    Check(!decoded.error_colour && TexelIs(decoded, 0, 255, 0, 0, 255),
          "clear reserved bits keep the void extent colour");
  }
  // footprint 必須是解碼器被交代的那一個，而不是它自己猜的。
  {
    const auto block = MakeVoidExtentBlock(0xffff, 0xffff, 0xffff, 0xffff);
    AstcDecodedBlock decoded;
    const char *refusal = nullptr;
    Check(!DecodeAstcBlock(block.data(), AstcBlockFootprint{0, 0}, false,
                           &decoded, &refusal),
          "an empty footprint is declined");
    Check(refusal != nullptr &&
              std::string(refusal).find("footprint") != std::string::npos,
          "the refusal names the footprint");
  }

  if (g_failures != 0) {
    std::fprintf(stderr, "astc_decoder_test: %d failure(s)\n", g_failures);
    return EXIT_FAILURE;
  }
  std::printf("astc_decoder_test: all checks passed\n");
  return EXIT_SUCCESS;
}
