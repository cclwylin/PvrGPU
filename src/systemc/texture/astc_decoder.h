// PowerVR TPU 內 ASTC 解碼器的資料路徑。真實硬體在 texture unit 裡直接
// 解 ASTC 區塊：TPU 從 TCU 取回 128-bit 的壓縮區塊，解出 footprint 內的
// texel，再交給 filter。把這件事放在 driver 或讓 Mesa 在 CPU 上先解壓，
// 等於把模型該算出來的答案交給它，模型就無法證明硬體有這個能力。
//
// 這裡只有「一個區塊 → 一組 texel」這件事。定址、mip 與 filter 仍屬
// TextureUnit；區塊的取得也還是走 TCU，所以 cache 與 DRAM 的計數不變。
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace pvrgpu::stub {

// 一個 ASTC block footprint。ASTC 的區塊一律 128 bits，footprint 決定
// 它覆蓋多少 texel，因此也決定每個 texel 的位元率。
struct AstcBlockFootprint {
  std::uint32_t width = 0;
  std::uint32_t height = 0;

  bool valid() const { return width != 0 && height != 0; }
  std::uint32_t texel_count() const { return width * height; }
};

// Rogue TEXSTATE FORMAT_COMPRESSED 的 ASTC 值（0..13）對應的 footprint。
// 回傳 width==0 表示這個值不是 ASTC。
AstcBlockFootprint AstcFootprintForRogueFormat(std::uint32_t rogue_format);

// 一個已解碼的 ASTC 區塊：footprint 內每個 texel 的 RGBA8。
struct AstcDecodedBlock {
  AstcBlockFootprint footprint;
  // 最大 footprint 是 12x12 = 144 texel。
  std::array<std::array<std::uint8_t, 4>, 144> texels{};
  // 這個區塊裡有沒有 texel 走了規格的 error colour 路徑（無效編碼，或
  // LDR 模式下的 HDR 區塊）。解碼仍然成功 —— error colour 是規格規定的
  // 輸出，不是失敗。
  bool error_colour = false;
};

// 解一個 128-bit ASTC 區塊，走 LDR 解碼模式（GLES 的
// GL_KHR_texture_compression_astc_ldr）。二維 footprint 的整個 LDR profile
// 都在這裡：block mode、weight grid 與其 infill、ISE（trit / quint / 純
// 位元）、partition pattern、十個 LDR colour endpoint mode、dual plane 與
// colour component selector。
//
// 區塊「無效」不是拒絕。規格對每一種無效編碼都規定了輸出 —— error
// colour（洋紅）—— HDR 區塊在 LDR 模式下也是同一條路。硬體照樣會吐出
// texel，所以這裡也照樣吐；能讓這個函式回 false 的只有「這根本不是一次
// 合法的呼叫」：空指標，或一個不屬於這個解碼器的 footprint。
//
// `srgb` 選的是輸出的轉換，不是要不要做 sRGB。sRGB 區塊在內插時就用不同
// 的端點展開（低位元組是 0x80 而不是端點自己），輸出取 16-bit 結果的高
// 8 位元；transfer function 由 TextureUnit 在 filter 之前施加，和未壓縮的
// sRGB 影像走同一條路。
bool DecodeAstcBlock(const std::uint8_t block[16],
                     AstcBlockFootprint footprint,
                     bool srgb,
                     AstcDecodedBlock *out,
                     const char **out_refusal);

} // namespace pvrgpu::stub
