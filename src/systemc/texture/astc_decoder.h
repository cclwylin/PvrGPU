// PowerVR TPU 內 ASTC 解碼器的資料路徑。真實硬體在 texture unit 裡直接
// 解 ASTC 區塊：TPU 從 TCU 取回 128-bit 的壓縮區塊，解出 footprint 內的
// texel，再交給 filter。把這件事放在 driver 或讓 Mesa 在 CPU 上先解壓，
// 等於把模型該算出來的答案交給它，模型就無法證明硬體有這個能力。
//
// 這裡只有「一個區塊 → 一組 texel」這件事。定址、mip 與 filter 仍屬
// TextureUnit；區塊的取得也還是走 TCU，所以 cache 與 DRAM 的計數不變。
// 包住它的 sc_module 與 TextureUnit 的接線隨壓縮紋理真的送到模型時一起
// 加進這一組檔案 —— 在 driver 宣告 ASTC、Mesa 停止 CPU 解壓之前，沒有
// 任何壓縮資料會走到這裡，先接線只會是無人執行的程式碼。
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
};

// 解一個 128-bit ASTC LDR 區塊。
//
// 目前只解 void-extent 區塊：整個 footprint 是同一個顏色，區塊自己用
// 低 9 位元標示身分，不需要權重或 partition。其他區塊型別會被具名拒絕
// 而不是猜一個顏色出來 —— 拒絕會少一個 PASS，編一個答案則是錯的。
//
// 成功時回傳 true 並填滿 `out`；失敗時回傳 false 並把 `*out_refusal`
// 指向一個說明「哪一種區塊、哪個欄位」的靜態字串。
bool DecodeAstcBlock(const std::uint8_t block[16],
                     AstcBlockFootprint footprint,
                     bool srgb,
                     AstcDecodedBlock *out,
                     const char **out_refusal);

} // namespace pvrgpu::stub
