# GL5 Normal capture 1 debug 交接（2026-09-10，第二輪）

本文件記錄 `gl_5_normal_capture_1.rdc` 的實際除錯狀態，供下一個 session
直接接續。流程必須遵守 [MANHATTAN_DEBUG_WORKFLOW.md](MANHATTAN_DEBUG_WORKFLOW.md)
與 [DRAWLIST_SNAPSHOT_REPLAY.md](DRAWLIST_SNAPSHOT_REPLAY.md)。

## 先讀這一段

- Capture 共有 **601 個零起算 DrawList：Draw 0–600**。最後一筆是 Draw 600，
  action event 12898，完整 range 結束於 event 12912。
- **上一輪的 Draw 176 blocker 已解除**。400-DWORD VS CB0 改走 read-only UBO/DMA
  descriptor 的候選修正已用 capture 實際重播驗證，Draw 176、177 的 D24 attachment
  與 llvmpipe 逐 byte 相同。
- 本輪把嚴格 ordered-prefix 前沿從 **Draw 175 推進到 Draw 196**，其後只剩一個
  1-half-ULP 等級的 RGBA16F 差異（Draw 197，158 px），D24 一路到 Draw 211 都相同。
- 過程中找到並修正 **五個 model 與 llvmpipe 的算術偏差**（setup vertex order、
  texture LOD 的 log2、ASTC unorm8 量化、texture float lerp 的 FMA、USC 的
  log2/exp2 多項式）。每一項都不是「讓畫面看起來對」，而是把 model 對齊
  llvmpipe 實際執行的運算。
- **repository 已從 iCloud 搬到 `/Users/linwanyi/Downloads/_Codex/PvrGPU`**。舊文件與
  凍結的 `build_private_driver.rb` 仍寫死 iCloud 路徑，直接用會 abort。
- **一個嚴重的操作事故**：在已通過 capture 驗證的 build 目錄下執行整包
  `cmake --build`，把凍結的 bridge v27 dylib 覆蓋掉了。checkpoint 沒有被污染
  （fail-closed 擋下了接續），但該 artifact 已無法重現。詳見「事故與教訓」。
- 目前 **另一個 session 正在同一個 working tree 上做 speed tuning**（09:01–09:28
  修改了 cache/memory/USC/texture 檔案）。接手前務必確認 tree 狀態。

## 固定輸入與 repository 狀態

| 項目 | 值 |
| --- | --- |
| Repository | `/Users/linwanyi/Downloads/_Codex/PvrGPU`（已非 iCloud） |
| Branch / HEAD | `main` / `30a444c7bdfa9f6cf3145a48e4ff6a9e5b29589d`（未 commit） |
| Capture | `/Users/linwanyi/Downloads/_Codex/GPU_TestPatterns/7.gl_5_normal/recorder/trace/gl_5_normal_capture_1.rdc` |
| Capture bytes / SHA-256 | `140738057` / `6e60585e5623470b966528976c4398d7f0df03e28023b78e1331ef40bab964b6` |
| Debug root | `/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/out/runs/gl5normal-debug.zKWINY` |
| Action census | `reference-phase1/actions.tsv` |
| Gallium counters | `reference-phase1/GalliumDrawLists.csv` |
| Replay GLES override | `3.2` |

## 結論等級

沿用上一輪的區分，不得混用：

- **Strict execution PASS**：replay 返回、API errors 為 0、native submit 與完成一一
  對應、readback audit 通過。
- **Attachment exact**：指定 attachment 的格式、subresource 與 raw bytes 都與
  llvmpipe 相同。
- **Checkpoint save/manifest PASS**：`replay.json`、`snapshot/state.bin`、
  `snapshot/manifest.json` 齊全並原子提交。
- **Restore-proven checkpoint**：後續新程序真的把它當 `--resume` 載入並通過
  restore/recapture compare。
- 單元測試通過、兩張空圖相同、draw 被跳過，都不能升格為 capture PASS。

## 目前前沿

| 邊界 | PvrGPU | 與 llvmpipe 比較 | 結論 |
| --- | --- | --- | --- |
| Draw 0–68 | fresh native strict PASS | D24 Resource 2697 相同 | exact |
| Draw 69–151 | 83 個邊界各自從 fresh D68 resume，全部 strict PASS、driver errors 0 | 76 個有 llvmpipe 對照的邊界 D24 全部相同；Resource 2693 在 D100/D151 也相同 | exact |
| Draw 175 | strict PASS | D24 相同 | exact |
| Draw 176 / event 1863 | **1 submit / 1 completion，API 0，restore verified**；counter 記錄 `draw_array_primitive_cb0_dma stage=0 source_dwords=400 native_ubo_slot=0` | D24 blob 相同 | **上一輪 blocker 已修復** |
| Draw 177 | strict PASS，同樣走 CB0 DMA | D24 相同 | exact |
| Draw 184 | fresh native strict PASS | 全 inventory 比對：176/176 texture、549/549 buffer 相同 | exact |
| Draw 185 / event 1986 | strict PASS | D24 相同（`94140146…`） | **上一輪未知的第一個 mismatch，已修復** |
| Draw 186–194 | 9 個邊界各自從 exact D185 resume，全部 strict PASS | D24 全部相同；D194 = `30eb0b67…`，與交接文件記載的 reference 一致 | exact |
| Draw 195 / event 2153 | strict PASS，1 submit/1 done | D24 **與 RGBA16F Resource 2708 都相同** | **目前 strict frontier；第一個通過驗證的 RGBA16F color attachment 邊界** |
| Draw 196 | strict PASS | D24 與 RGBA16F 都相同 | exact |
| Draw 197–199 | strict PASS、driver errors 0 | D24 相同；RGBA16F 在 197 差 158 px（1 half-ULP 為主） | 見 blocker A |
| Draw 211 | strict PASS，D24 相同 | RGBA16F 有 1184/1048576 px（0.11%）不同，2013/2040 個 half 恰好差 1 ULP | **下一個 blocker（精度）** |
| Draw 227 | **replay 失敗，無 snapshot** | 未比較 | **第二個 blocker（功能缺口）** |
| Draw 600 / range end 12912 | 未執行 | 未執行 | 最終目標 |

llvmpipe 參考鏈已完整建立到終點：`chain-ref-d211 … chain-ref-d595`（每 16 draws
一個節點）與 `chain-ref-d600-llvmpipe`（after=600/12912，restore verified）。
native 只要能通過下面兩個 blocker，就有逐節點可比的對照。

## 本輪修正的六個問題

### 1. Draw 176：400-DWORD VS CB0 走 read-only UBO/DMA descriptor（driver）

上一輪已有候選修正但未經 capture 驗證，且留下一個稽核缺口：
`pvrgpu_lower_large_cb0()` 直接把 `nir_intrinsic_base()` 放進 `uint64_t`，沒有在
helper 內拒絕負值或做 checked arithmetic。graphics caller 有前置的 prefix proof，
compute caller 沒有，`base=-1` 配 direct offset 1 可能 wrap 成 0。

已在 `src/gallium/drivers/pvrgpu/pvrgpu_pco.c` 的 shared helper 內補上：signed base
檢查、base+offset 的 checked add、`last * 16` 的 32-bit byte-address 上界、以及
span 檢查；並在 `tests/pvrgpu_compute_compiler_test.c` 補三個 compute 負例
（`native_large_cb0_dma_invalid`：base=-1、direct offset 溢位、indirect range 溢位）。

### 2. Draw 185：Mesa setup vertex order（model，geometry）

llvmpipe 的 `lp_setup_tri.c` 只有在 24.8 定點面積為負時才交換 v0/v1
（`rotate_fixed_position_01`）；面積為正時保持輸入順序。model 的
`ClipCull::BuildRasterTriangle` 無條件假設「已交換」的順序 `{1,0,2}`／clipper fan
`{2,1,0}`，那只有在 back-face culling 會濾掉正面積三角形時才總是成立。

Draw 183–185 關閉了 face culling（雙面樹葉 shadow card），兩種 winding 都會進 setup，
於是每個三角形的交換與否都不同。因為 `lp_state_setup.c` 把 depth plane 的
a0/dadx/dady 與 polygon-offset slope 都錨定在 setup v0，錨錯會產生整個三角形一致的
sub-code 偏移，在 D24 上呈現為 ±1 code。

修正：以 `QuantizedArea` 的符號（llvmpipe 面積的相反數）決定是否交換，
順序改為 `{1,0,2}`/`{2,1,0}`（負面積）或 `{0,1,2}`/`{1,2,0}`（正面積）。
`tests/driver_front_face_test.cpp` 加入檢查：未被 cull 的三角形，其記錄的 setup
順序在 llvmpipe 慣例下面積必須為正。

效果：Draw 185 co-written 像素的 13289 個 ±1..4 code 差異全部歸零。

### 3. Draw 185：trilinear LOD 的 log2（model，texture）

commit `57a191f`（2026-09-06）為了消除 dEQP 的 `*_mipmap_linear` QualityWarning，
把 LOD 從 llvmpipe 的分段線性 `lp_build_fast_log2` 改成精確 `std::log2`。但
llvmpipe 的 `lp_build_lod_selector` 一般路徑就是 `0.5 * fast_log2(rho^2)`
（`gallivm_perf` 預設 0，`RHO_APPROX`／`BRILINEAR` 都是 opt-in），其中段誤差約
0.086 正是 reference 實際的 mip blend weight。

GL5 的樹葉 shadow draw 對一張 minified trilinear ASTC 貼圖做 alpha test
（threshold 0.021），精確 log2 會把混合後的 alpha 推過門檻，造成覆蓋翻轉。

修正：`SelectTextureLod` 與 `SelectTextureBiasedLod` 改回 `TextureFastLog2`。
**取捨**：dEQP 的 `*_mipmap_linear` 可能重新出現 QualityWarning（依 57a191f 的
說明，那些案例仍通過低精度標準）。capture replay 的 oracle 是 llvmpipe，兩者衝突時
以 llvmpipe 為準。

### 4. Draw 185：ASTC 16-bit → unorm8 量化（model，texture）

llvmpipe 不支援 ASTC（`lp_screen.c` 回報 false），Mesa 用
`util/texcompress_astc.cpp` 在 CPU 解碼成 unorm8，取高 8 位元（`c >> 8`）。
model 自己解 ASTC（driver 有 advertise ASTC），但用四捨五入
`(c * 255 + 32768) >> 16`。對 Mesa 值 ≤ 8 的 texel，model 有一半機率 +1、從不更低，
剛好偏在 alpha test 門檻那一段。

修正：`ResultToUnorm8` 對 linear 與 sRGB 一律取高 8 位元。
`tests/astc_decoder_test.cpp` 的量化案例改為截斷語意並補上極值案例。

### 5. Draw 196：texture float lerp 缺少 FMA（model，texture）

llvmpipe 的 `lp_build_lerp_simple` 對所有非 8-bit-unorm 的 datapath 走浮點分支，
而該分支是 `lp_build_mad(x, v1 - v0, v0)` → `lp_build_fmuladd`，一個會在本機融合的
`llvm.fmuladd`（ISP depth plane 先前已依賴同一個融合行為）。model 的
`LerpTextureFloat` 用的是分離的乘與加，會多一次捨入。

RGBA16F 走的正是浮點 datapath（8-bit 定點路徑只服務 8-bit unorm），所以差異出現在
bloom chain。修正為 `std::fma(weight, second - first, first)` 後，Draw 196 的
RGBA16F 變成 exact，第一個差異從 Draw 197 開始，差異量從 1184 px 降到 159 px。

### 6. Draw 197–199：USC 超越函數用 libm 而非 llvmpipe 的多項式（model，shader）

Draw 197 起換了一個 fragment shader，其 counter 是 **`ps_tex=0`、`ps_sfu=6`** ——
完全沒有 texture 運算，只有六個 special-function 運算，正是三個 component 各做一次
`pow()`（gallivm 把 `fpow` lower 成 `exp2(y * log2(x))`）的特徵。

gallivm 不用 libm：`lp_bld_arit.c` 以 minimax 多項式實作
（`EXP_POLY_DEGREE 5`、`LOG_POLY_DEGREE 4`），而且 `lp_build_polynomial` 是把級數
拆成 x² 的偶次與奇次兩半、各自用 FMA 收斂，不是 Horner。model 的
`FloatLog2Bits`／`FloatExp2Bits` 呼叫 `std::log2`／`std::exp2`。

修正：在 `src/systemc/shader/pco_iss.cpp` 內以 `LlvmpipeLog2`／`LlvmpipeExp2`
忠實重現 `lp_build_log2_approx`／`lp_build_exp2`，包含係數先捨入成 binary32、
exp2 的 `[-126.99999, 128]` clamp 與 exponent-field 構造、log2 的
`y = (m-1)/(m+1)` 代換。原有的 inf／NaN／zero／負值 guard 保留在呼叫端，
那些已經符合 NIR flog2 的參考行為。

獨立驗證：log2 相對 libm 的最大相對誤差約 `4.5e-5`（degree-4 fit 的預期量級），
exp2 約 `2.6e-7`；`pow(0.5, 0.45)` 在第 8 位數分岔，正是 binary32 的 1 ULP。
NIR 對應也已核對：`nir_op_fexp2` → `lp_build_exp2`，`nir_op_flog2` →
`lp_build_log2_safe`，兩者都是上述多項式。

**實測結果：這個修正沒有解掉 Draw 197**，它只讓 4194304 個 half 之中的 1 個改變。
原因是多項式誤差（約 `4.5e-5`）低於 half-float 的解析度（約 `5e-4`），存進
RGBA16F 之後幾乎看不出來。修正仍然保留 —— 它本來就是正確的對齊，而且整個 frame
有 372／623 筆 draw 使用特殊函數，之後的 RGBA8 與更高精度 target 會需要它。
Draw 195／196（前綴中唯一使用 SFU 的兩筆）重跑後仍 exact，沒有回歸。

### 修正後的 unit 測試

以下都對本輪 bridge 執行過並通過：`astc-decoder-test`、`texture-filter-test`、
`texture-unit-test`、`texture-bias-test`（9772 checks）、
`texture-lod-invariance-test`（62915 checks）、`texture-lod-independent-test`、
`texture-cube-array-test`（20114 checks）、`driver-front-face-test`（42030 checks／
99 cases）、`geometry-frontend-test`（894 checks）、`isp-depth-attachment-test`。

Mesa 側（driver 修正）：`run_mesa_pco_ubo_unit.sh shared-budget`（38 cases／322
checks）、`ubo`、`run_mesa_resource_unit.sh ubo`（6105 checks）、
`run_mesa_pco_compute_unit.sh`、`run_mesa_pco_fragment_image_unit.sh`（1072 checks）、
`run_mesa_pco_texture_unit.sh`、`run_mesa_resource_unit.sh push-map`（334231 checks）。

**已知無關失敗**（兩者在本輪修改前後都失敗，且在 bridge v28、v29 這兩個
不含相關修改的 build 上同樣失敗，因此不是本輪造成的）：

- `usc-cluster-texture-continuation-test`：`fragment USC varying coefficient
  count is invalid`。該測試與 `src/systemc/shader/usc_cluster.cpp` 屬於並行
  speed-tuning session。
- `pco-iss-test`：`VS/FS unary ABS/NEG preserve signed zero, infinity, NaN and
  subnormals`。同樣在 v28／v29 上就失敗。

接手時若要判斷某個測試失敗是否為自己造成，最快的方法是拿一個不含該修改的
既有 bridge build 目錄，用 `cmake --build <dir> --target <test>` 重建同一個
測試再跑一次。

## 下一個 blocker（依序）

### A. Draw 197–199：RGBA16F 還差 158 px

Draw 195、196 的 RGBA16F 已 exact。Draw 197 差 158 px（4194304 個 half 中的 214 個，
其中 196 個恰好差 1 個 half-ULP），幾乎只在 RGB，alpha 幾乎不動。
Draw 198、199 是把它往後帶。

這幾筆共用一個 fragment shader（`reference-phase1/shaders/ps-nir-000003c800000001.nir`，
ps_src `e3a6c1c1839f035d`）：`ps_tex=0`、`ps_sfu=6`、`ps_alu=32`、`ps_mem=15`，
668493 個 invocation。它的六個 SFU 運算已從 NIR 逐一對上：
一個 `frsq`、一個 `fsqrt`、三個 `fdiv`、一個 `fpow`。

已排除的候選：

- **rcp／rsqrt／div／sqrt**：llvmpipe 沒有用 SSE 近似。`lp_build_rcp` 的快速路徑寫成
  `if (false && …)`、`lp_build_rsqrt` 寫成 `if (0 && …)`，兩者都落到精確的
  fdiv 與 `1/sqrt`；`nir_op_fdiv` 直接對到 `lp_build_div`。與 model 現行做法相同。
- **pow 的展開方式**：PCO 的 `pco_nir.c` 設 `.lower_fpow = true`，所以 NIR 在進入
  model 之前就把 `fpow(x,y)` 展開成 `exp2(y * log2(x))`；llvmpipe 的
  `lp_build_pow` 是 `exp2(log2_safe(x) * y)`。乘法可交換且精確捨入，兩者組合方式
  相同（llvmpipe 另外多一個 `x == 0 → 0` 的 select，只影響 x 為零的情形）。
- **log2／exp2 本身**：改成 llvmpipe 多項式後只動了 1 個 half（理由見上）。

**剩下的假說**：half-float 存檔之前有一個 binary32 等級的 1 ULP 差異。數量級吻合 ——
把約 4M 個值各擾動一個 float ULP，只有落在 half 進位邊界一個 float ULP 內的會翻轉，
約 `4M × 2⁻¹³ ≈ 500`，與觀察到的 214 同一個量級。

**建議的下一步**：SFU 已經全部排除，所以差異應該在該 shader 的 32 個一般 ALU
運算、varying 的透視內插（這個 FS 讀 `VARYING_SLOT_VAR0`／`VAR1`），或最後寫入
RGBA16F 的 float→half 轉換。最有效率的做法是對其中一個差異像素做單點
instrument，把 model 與 llvmpipe 在 `%10`、`%19`、`%24`、`%34`、`%38` 這幾個中間值
上的 binary32 位元逐一比對，而不是繼續猜測。

### B. Draw ~216 起：mixed render targets（功能缺口）

Draw 215 起進入 1920×1080，Draw 216 之後是 4 個 color attachment 的 deferred
G-buffer。driver 以 `mixed_render_targets` 拒絕（`suffix-d227-pvrgpu-v27` 的
counter 有 14 次 `draw_array_primitive_gate_declined`）。

原因在 `pvrgpu_context.c:870`：只要各 attachment 格式不一致
（`pvrgpu_framebuffer_has_mixed_color_formats`），每個格式都必須在
`pvrgpu_color_formats.h` 的白名單內，而該白名單只有三個格式
（`R8G8B8A8_UNORM`、`R10G10B10A2_UNORM`、`B10G10R10A2_UNORM`）。

從 `chain-ref-d227-llvmpipe` 的 snapshot 解出這個 pass 全部的 1920×1080 texture
（signature 的 u64 佈局是 type／target／internalformat／width／height）：

| Resource | internalformat | bytes/px |
| --- | --- | --- |
| 214, 215, 216, 223 | `GL_RGBA8` (0x8058) | 4 |
| 217 | `GL_R16F` (0x822d) | 2 |
| 218 | `GL_RG16F` (0x822f) | 4 |
| 220 | `GL_R11F_G11F_B10F` (0x8c3a) | 4 |
| 219 | `GL_DEPTH_COMPONENT24` | 4 |
| 2679 | `GL_RGB8` | 3 |

所以這是典型的 deferred G-buffer：RGBA8 之外還有 R16F／RG16F／R11F_G11F_B10F，
三者都不在白名單內。工作項目是擴充 driver 的 explicit color format 清單與 model
端對應的 color transport／PBE 編碼，而不是精度調整。實際綁定的是哪四個，
要在能執行該 draw 之後從 `set_framebuffer_state` 的 counter 確認。

### C. 同一段內的 compute dispatch（功能缺口）

在一次 96×512 `R16G16B16A16_FLOAT` readback 之後有一個 compute dispatch，
被 `compute_launch_unsupported reason=compute_compile` 拒絕，訊息是
`pvrgpu_pco.c:6029` 的 OR 條件：
`shared_size > 32 KiB || scratch_size || num_images > 32 || num_textures > 上限`。
需要先確定是哪一個條件觸發（訊息本身不區分），再決定是放寬上限還是補功能。

## 凍結 runtime 與 artifact

| 元件 | 路徑 | SHA-256／狀態 |
| --- | --- | --- |
| 新 native Mesa driver（本輪 CB0 DMA） | `…/gl5normal-debug.zKWINY/driver-graphics-cb0-dma-v25/install` | Gallium `4ecd62b064fa33e47fc7233704ea023c8083c4327eb18126dfe17598aabb1d3b` |
| 舊 native Mesa v22 | `…/driver-vs-split-v22/install` | Gallium `fb3944339d6e2acb5b71ba216e4ed07938cb6ff8745d5dcdf917dd4d178bcc4b` |
| llvmpipe reference prefix | `…/driver-depth-only-v19/install` | Gallium `e531e5d1d8c1913cea046dbdcd643bd8381f29fefdcc434cee0943c28172b751` |
| 私有 driver builder（已重定位） | `…/gl5normal-debug.zKWINY/build_private_driver_v2.rb` | `f51aac7046e4847ddb2f897ac7a547fe3ef29e55b63fdc60a9f9f7a2057fee62` |
| 私有 driver builder（原版，仍寫死 iCloud 路徑） | `…/build_private_driver.rb` | `5e30b296a9ecd751d05fa585e3e5dc2aaf174570b0b6b56ac2d46dbdf3279caa` |
| Snapshot RenderDoc | `…/tmp/renderdoc-gl5normal-d24-codec.v2/lib/librenderdoc-gl5normal-snapshot-rgb9e5-d24-codec.dylib` | `4b355ae47a124e42787d47f9050a5342b136c43f6f3a4ff4f150e5d57ad7d2a7` |
| DrawList player | `…/tmp/pvrgpu-drawlist-player.XHq5Ng/pvrgpu-drawlist-player` | `2ba54e6a65768a063cf071f356835c3c6aa267c693eb56f360588098c7a522cf` |

`build_private_driver_v2.rb` 只把 repository 的 source／include 參數從舊 iCloud
路徑重寫到現在的路徑；凍結的 compiler archive、link input 與 receipt 都維持原
絕對路徑與 SHA，並在 legacy 路徑仍存在時直接 abort 以避免歧義。

### Bridge 版本

| 版本 | 內容 | 狀態 |
| --- | --- | --- |
| v24 exact | 上一輪的 non-finite clip 修正 | `9d0c1477e5d9be8765453cc34c7225560aa84fdfdbf62b8471441e7f465dcec8` |
| v25 | 加入 setup vertex order 修正 | `2318517ced6ab1d123a153f127085c4fda9c850d56b19bc5ab2b5b44c124be00` |
| v26 | 加入 fast_log2 LOD（中間版，未用於驗證） | `2d3b3044918bb472…` |
| v27 | v26 + ASTC 截斷；**Draw 184/185 exact 就是用它驗證的** | 驗證時 `9061b7161c65e9808291670d8fa2df9f2de7c21db42e0b6f6a8a4e4d188325d5`，**檔案已被覆蓋** |
| v28 | 從目前 tree（含並行 session 的 perf 修改）重建；D185／D194／D195 全部重現 exact | `04e2e16911d4c1151d47deefe46e714d9e1ac61545d64e4f5251a668a38e36d9` |
| v29 | v28 + texture float lerp 改用 FMA；Draw 196 RGBA16F exact | `c369608d0f03e3fb9a3e42393b49301d19e0a3462526b4dbac0c655dedf6609e` |
| v30 | v29 + USC log2/exp2 改用 llvmpipe 多項式；Draw 196 仍 exact | `8e93ac94aff23906d20aaaebc5b7a71d3f7f83c0aa4d9baaa4124c62383f9014` |

## 事故與教訓

在 `bridge-fastlog2-astc-v27/build` 下執行了整包 `cmake --build`（原意是跑完整
ctest 迴歸），它把已通過 capture 驗證的 bridge dylib 重新連結成含並行 session
09:01 之後 perf 修改的版本。

- **沒有 checkpoint 被污染**：所有 manifest 記錄的都是 `9061b716…`，
  `run_drawlist_replay.py` 以 `snapshot runtime changed: bridge` 拒絕接續。
  這正是 fail-closed 設計要達到的效果。
- **但該 artifact 無法重現**：working tree 已經前進，磁碟上沒有副本。
- **教訓**：驗證用的 build 目錄只能用具名 target 建置
  （`cmake --build <dir> --target <name>…`），永遠不要在裡面執行不帶 `--target`
  的整包建置；迴歸測試要另開 build 目錄。

## 並行 session 警告

09:01–09:28 之間另一個 session 修改了下列檔案（allocation 迴避／效能重構，
對應未追蹤的 `docs/CARCHASE_CPU_PROFILE_20260910.md`、
`docs/CARCHASE_SPEED_ROUND1_20260910.md`、`docs/MANHATTAN_SPEED_ROUND1_20260910.md`）：

```text
src/systemc/cache_mmu/cache_array.{cpp,h}
src/systemc/memory/gpu_memory_system.{cpp,h}
src/systemc/shader/usc_cluster.cpp
src/systemc/texture/texture_unit.cpp
tests/{cache_array,gpu_memory_system,usc_cluster_texture_continuation}_test.cpp
```

它們看起來是行為保持的（inline buffer 取代 vector、重用 host capacity），但沒有
被本輪的 capture 驗證涵蓋。bridge v28 的重新驗證同時也是對這批修改的交叉檢查：
如果 D185/D194/D195 仍然逐 byte 相同，就等於證明它們對這個 workload 行為不變。

## 本輪的可重跑命令

環境（每個 code block 都可獨立執行）：

```bash
REPO='/Users/linwanyi/Downloads/_Codex/PvrGPU'
RUN='/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/out/runs/gl5normal-debug.zKWINY'
CAPTURE='/Users/linwanyi/Downloads/_Codex/GPU_TestPatterns/7.gl_5_normal/recorder/trace/gl_5_normal_capture_1.rdc'
PLAYER='/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/pvrgpu-drawlist-player.XHq5Ng/pvrgpu-drawlist-player'
RDOC='/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/renderdoc-gl5normal-d24-codec.v2/lib/librenderdoc-gl5normal-snapshot-rgb9e5-d24-codec.dylib'
NEW_MESA="$RUN/driver-graphics-cb0-dma-v25/install"
LP_MESA="$RUN/driver-depth-only-v19/install"
BRIDGE="$RUN/bridge-fastlog2-astc-v28/build/lib/libpvrgpu_systemc_bridge.dylib"
```

從 capture 初始狀態建立 native checkpoint（換 `--through-draw` 即可）：

```bash
cd "$REPO"
python3 script/run_drawlist_replay.py "$CAPTURE" \
  --player "$PLAYER" --renderdoc-lib "$RDOC" --mesa-prefix "$NEW_MESA" \
  --backend pvrgpu --bridge "$BRIDGE" --gles-version 3.2 \
  --through-draw 185 --outdir "$RUN/<new-name>" --timeout 3600
```

llvmpipe 參考（同樣的 boundary，換 backend 與 prefix，不要 `--bridge`）：

```bash
cd "$REPO"
python3 script/run_drawlist_replay.py "$CAPTURE" \
  --player "$PLAYER" --renderdoc-lib "$RDOC" --mesa-prefix "$LP_MESA" \
  --backend llvmpipe --gles-version 3.2 \
  --through-draw 185 --outdir "$RUN/<new-name>" --timeout 3600
```

逐節點稽核（status／submit-done／driver errors／attachment blob）：

```bash
cd "$REPO"; OUT="$RUN/<name>"
jq -c '{status,after_draw,after_event,resumed_from_event,context_finished,api_errors,snapshot_restore_verified}' "$OUT/replay.json"
printf 'submit=%s done=%s\n' \
  "$(grep -c 'event=systemc_api_submit' "$OUT/driver-counter.txt")" \
  "$(grep -c 'event=systemc_api_done' "$OUT/driver-counter.txt")"
! grep -q 'command_error\|record_error\|unsupported_draw\|gate_declined\|not_lowerable' "$OUT/driver-counter.txt"
python3 tools/inspect_drawlist_snapshot.py --resource 2697 "$OUT/snapshot/state.bin"
```

`inspect_drawlist_snapshot.py` 只驗證外層 envelope。要宣稱 raw exact，取
blob 的 offset 256 起、`1024*1024*4` bytes（D24）或 `1024*1024*8` bytes
（RGBA16F）比較；該 offset 已用交接文件記載的 `ccce0006…`／`41b7cd29…`
兩個 golden 校準過。

## 2026-09-10 dEQP L2／texture-functions 深測更新

### 執行範圍與最終結果

本輪先從 L2 停在 6 個 image Fail，修正後 6/6 通過；接著完整探索
`dEQP-GLES3.functional.shaders.texture_functions.*`。此族群共 609 cases，包含
489 個有效執行路徑與 120 個 invalid/compiler-negative cases，並以 8 shards、每 case
獨立 process 執行，避免 GL／driver state 在 cases 間洩漏。

探索階段一共找到 11 個 Fail，分成三個根因（4 + 2 + 5）。三個修正完成後又用最新
driver 與 bridge 從零完整重跑一次；正式基線是：

```text
total=609  Pass=609  Fail=0  NotSupported=0  Warning=0  crash=0
```

可稽核 artifacts：

```text
/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/out/runs/deqp_debug/texture_functions_deep_20260910_2130/
  caselists/all_interleaved.txt       # 609 個 exact case names
  full_clean/summary.tsv              # 合併後 609 rows、609 unique、全 Pass
  full_clean/shard_0..7/              # 每 case 的 QPA、run log、driver/SystemC artifacts
  array_tao_regression/summary.tsv    # 4 個原 Fail + 3 個正反向 guard，7/7 Pass
  array_shadow_grad_offset_regression/summary.tsv  # 2/2 Pass
  shadow_bias_regression/summary.tsv  # 5 個原 Fail + 3 個 guard，8/8 Pass
```

### 深測找到並修正的三個根因

1. **Vertex 2D-array `textureOffset` 被誤判為 shader bias（4 cases）**
   - Mesa 的 public SMP encoding 在 TAO 存在時重載 `LODM=BIAS`：fragment `txb`
     的 PPLOD word 是真實 bias；non-fragment array `textureOffset` 的同一位置只是
     TAO address pair 前的 zero-PPLOD padding。
   - `src/systemc/shader/pco_iss.cpp` 現在以 shader stage 消歧義。Vertex TAO 保留
     padding 並解出 layer address；fragment BIAS+TAO 仍消費真實 bias。
   - 原 Fail：fixed／float／int／uint 四種
     `textureoffset.*sampler2darray*_vertex`，修後全部 Pass。

2. **2D-array nearest shadow 的 admission 過窄（2 cases）**
   - `textureGradOffset()` 已由既有 NIR lowering 形成 explicit LOD，array layer 已由
     TAO 運送，depth compare 也已在 shader ALU 執行，但 driver gate 只准 2D/cube。
   - `pvrgpu_pco.c` 與 `pvrgpu_context.c` 僅新增 2D-array nearest shadow；沒有開放
     cube-array、linear PCF 或 sparse。
   - Vertex／fragment 的 `texturegradoffset.sampler2darrayshadow` 均已 Pass。

3. **Fragment shadow `texture(..., bias)` 被 admission 排除（5 cases）**
   - Native PPLOD.BIAS 與 shadow compare 原本各自已實作，只有 `biased_sample`
     人工排除了 `is_shadow`。
   - 現在只准 fragment scalar-f32 bias 與既有 nearest shadow target 的交集。
   - 2D／cube、projected／offset 五個原 Fail 均已 Pass；另以一般 non-shadow bias、
     2D shadow、cube shadow 三個 guard 確認沒有回歸。

上述 Gallium 修改同時套用到 repository source-of-truth 與實際 Meson runtime mirror：

```text
/Users/linwanyi/Downloads/_Codex/PvrGPU/src/gallium/drivers/pvrgpu/
/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/mesa-poc/src/src/gallium/drivers/pvrgpu/
```

### L2 首批六個 Fail 的前置修正

在完整 texture-functions 深測前，L2 首批六個錯誤已修正並 6/6 回歸通過：fragment
dual-phase `O_MBYP2` decode/parallel move、3D explicit LOD、2D-array BIAS+TAO、以及
nearest cube shadow admission。合併結果：

```text
/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/out/runs/deqp_debug/l2_first6_regression_20260910_2050/summary.tsv
```

單元回歸也已用目前 source 執行：

```text
script/run_mesa_pco_texture_unit.sh       PASS
script/run_mesa_pco_texture_bias_unit.sh  PASS（normal 9800 checks；ASan/UBSan 9772 checks）
```

`pco-iss-test` 的本輪新增 MBYP2／BIAS+TAO tests 位於 main 最前方並已通過；完整 target
之後仍會停在 dirty worktree 既有的 `ABS/NEG preserve signed zero...` 失敗，與本輪
texture 修正無關。完整 Mesa `meson compile` 也仍有既有測試編譯問題：
`pvrgpu_pco_packed_uniforms_test.c` 呼叫已新增 `bool *exceeds_budget` 參數的函式時少傳
一個 argument；本輪只建置並安裝實際的 `libgallium-26.2.1.dylib` target。

### 後續建議

`functional.shaders.texture_functions.*` 已 clean。若要再擴大 texture 覆蓋，下一層是
完整 `dEQP-GLES3.functional.*texture*`（discovery 共 7,423 cases），不要重跑本節的
609-case 核心族群。Manhattan C regression 仍應使用本文件前段凍結的 Capture/Play
路徑；一般 `script/run_regression.sh` 預設要求
`build/bin/pvrgpu-rdc-player`，目前該檔不存在，因此先前嘗試只是在 setup 階段失敗，
不是 render regression 結果。

## 2026-09-10：Manhattan replay 記憶體 retention 修正

Manhattan／Car Chase 的 2-worker full replay 由使用者中止；兩筆結果的 exit code 143
都表示人工 `SIGTERM`，不可當成渲染回歸失敗。中止前 Manhattan 約執行 43 分鐘，
RSS 為 6,332,280 KiB，`vmmap` physical footprint 約 7.9 GiB。同期已完成的 model
sequence 卻持續回報 `pool_leaks=0`、`pool_bytes_in_flight=0`；因此不是 live payload
ownership leak。

根因在 `model_stub/memory_pool.cpp`：`MemoryPool::Release()` 原本只對 payload vector
呼叫 `clear()`。slot 雖可重用，卻保留歷史最大 capacity；Manhattan 各 sequence 的
allocation shape 不同，數萬次配置後，大型 framebuffer／fragment payload capacity
會散落並留在許多 free slots，使 RSS 遠高於當下 live high-water。

目前修正為：容量不超過 64 KiB 的小型 control payload 繼續保留重用；超過 64 KiB
則在最後一個 reference 釋放時以 empty-vector swap 確實釋放 backing storage。新增
`MemoryPool::capacity_bytes()` 供精確單元驗證，並加入 `memory-pool-unit`，涵蓋大型
payload、保留的小型 payload 與由小型 slot 升級成大型 payload 的情況。

驗證結果：

```text
memory-pool-unit                 PASS
pco-systemc-api-bridge-unit      PASS
isp-depth-attachment-unit        PASS
fragment-frontend-unit           PASS
ASan/UBSan memory-pool test      PASS
git diff --check                 PASS
```

修正版 bridge 已建置於：

```text
/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/build/lib/libpvrgpu_systemc_bridge.dylib
SHA-256 094b3418471e68bb0db01e1bd218d0bc57bb02c952437195743cf1c48597fde1
```

上述 capacity-retention 修正已在後述 Release full-frame regression 完成實測；本節
當時留下的「尚未重跑」限制已解除。完整結果、實際載入的 bridge 與 memory samples
路徑見下一節。

## 2026-09-10：GLES31 multisample 2D-array 與 Release Manhattan 回歸

### `samples_N.use_texture_*_2d_array` 根因與修正

目標 regex：

```text
^dEQP-GLES31\.functional\.texture\.multisample\.samples_[0-9]+\.use_texture_(color|int|uint|depth)_2d_array$
```

初始矩陣共 40 cases：裝置支援的 samples 1/2/3/4/8 共 20 個全 Fail；samples
10/12/13/16/64 共 20 個正確回報 NotSupported。20 個 Fail 都是同一支 1024-byte
fragment PCO 在 byte 849 被 `invalid generic 2D texture request` gate 拒絕。

完整 PCO dump 證明合法的 2D-array `txf_ms` 編碼同時帶有 TAO、NNCOORDS、SNO 與
`LODM=BIAS`。Mesa 在 TAO address pair 前必須插入 zero PPLOD word；該 word 對
multisample fetch 是 padding，不是 shader texture bias。舊 decoder 只靠 fragment
stage 判斷，因而把它誤標成 `texture_lod_bias=1`，再與 SNO 的合法性規則衝突。

`src/systemc/shader/pco_iss.cpp` 現在用 SNO 消歧義：fragment BIAS+TAO 的真實 bias
仍照常保留；SNO+TAO 則把 PPLOD 當 zero padding，並由既有 address-override 路徑
驗證與消費。`tests/pco_multisample_texture_test.cpp` 新增明確 assertion，要求所有
native multisample array fixtures 都不產生 LOD-bias semantic flag。

驗證：

```text
pco-multisample-texture-unit  PASS (2112 checks)
texture-multisample-unit      PASS
memory-pool-unit              PASS
pco-systemc-api-bridge-unit   PASS
isp-depth-attachment-unit     PASS
fragment-frontend-unit        PASS
git diff --check              PASS
```

完整 live dEQP 結果是支援的 **20 Pass**、超限的 **20 NotSupported**，且所有 case
的 driver counter 均沒有 decode error、unsupported draw 或 validation error：

```text
/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/out/runs/deqp_debug/gles31_msaa_2d_array_fixed_20260910/
```

### Release/O3 Manhattan correctness 與 memory

獨立 Release build 使用 Homebrew LLVM 22.1.8，compile command 已核對含
`-O3 -DNDEBUG` 與 `PVRGPU_ENABLE_DIAGNOSTICS=0`：

```text
/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/build-release-memory-llvm/
lib/libpvrgpu_systemc_bridge.dylib
SHA-256 3401949dcf41c4720819955ee590c282b75c06d66771fb7d2bcfd59a8e963b6f
```

正式 run 前以 `lsof` 驗證 player 實際載入上述 Release bridge、正式 RenderDoc 與
目前 Mesa prefix。完整 231 draw-action replay 於 **717.59 秒（11 分 57.59 秒）**
完成，regression 為 PASS；比先前相近但不同 frozen runtime 的 820.20 秒歷史值快
約 102.61 秒，但 runtime/helper 不完全相同，這只能作趨勢參考，不能作嚴格 A/B。

correctness 證據：model `frame.png` 與 player final attachment PNG 都是 1920x1080、
SHA-256 都為 `185a89f710e0d9c639c770987a50da6a444a37d358f72e6f3102d7ec94df8897`；
API errors=0、model/bridge exit 0、219 drawlists、36 graphics submissions、5 compute
dispatches、pool allocations/releases 平衡且 `pool_leaks=0`。

memory 每 5 秒取 RSS、每約 60 秒取 `vmmap -summary`：

```text
RSS peak                 12,125,416 KiB (~11.56 GiB)
last sampled RSS          6,374,648 KiB (~6.08 GiB)
physical-footprint peak   7.2 GiB
settled physical footprint (439-627 s)  2.6 GiB
```

RSS 包含 allocator 可重用／非駐留虛擬頁，判斷實體壓力應看 physical footprint。
本次 footprint 在中段工作峰值後回落並連續四次維持 2.6 GiB；舊版中止前則在
約 7.9 GiB 且持續偏高，因此 full-frame 實測支持 large-slot capacity release 有效。

完整 regression、memory samples 與 correctness artifacts：

```text
/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/out/runs/rdc_regression/manhattan_release_memory_20260910_final_v2/
```

## 2026-09-11：SMP gate 說明與 Release Car Chase 回歸

### Multisample 2D-array 修正不是 workaround

`pco_iss.cpp` 的兩個既有 dimension gate 刻意維持不變：decode admission 仍要求
SNO/NNCOORDS request 的 `dimension == 2`，`HasCanonicalTextureFields()` 也維持同一
canonical contract。這是正確語意，不是漏改：`sampler2DMSArray` 的 array layer 已由
Mesa 降成 TAO base-address offset，TPU request 本身只有二維座標，因此不應把它改成
三維 request 或放寬 gate。

實際 bug 位於 gate 之前的 semantic decode：TAO+SNO 的 zero PPLOD padding 被誤認成
真實 texture bias。現在 `lod_bias` 以 SNO 消歧義，並在上述兩個 gate 各補一段註解，
說明為何 array request 實際仍以 `dimension=2` 通過。這是 decoder contract 修正；沒有
改寫輸入 dimension、清除錯誤、跳過 draw 或依 case name 分支。

### Car Chase 首次失敗與 RenderDoc runtime 修復

第一次 Release/O3 full run 使用 ES 3.1，於 898.67 秒後被嚴格 GL 檢查拒絕：
`Formal replay: GL error 1282`。driver counter 直接顯示第一個 tessellation draw 的
`patch_vertices=3`；實際載入的正式 RenderDoc source 又把
`glPatchParameteriEXT/OES` 分派到 point serializer。這是先前已修過、但後來正式 build
沒有包含的 RenderDoc regression，不是 PvrGPU shader/model regression。

已把 repository 中兩個一般化 patch 重新套到實際 RenderDoc checkout 並重建：

- `renderdoc-mesa-patch-parameter-alias.patch`：EXT/OES alias 改走 patch serializer。
- `renderdoc-mesa-discard-multiview-attachment.patch`：depth-only FBO 不再對不存在的
  COLOR_ATTACHMENT0 發出 texture-only multiview query。

alias 6 項與 discard/multiview 7 項 source/mock regression 全過。重建後正式 library：

```text
/Users/linwanyi/Downloads/_Codex/Working/build/renderdoc-mesa/build-homebrew-x86_64/lib/librenderdoc.dylib
SHA-256 1129e74a28ccc79ab000748eb9a6f1d0fff4ef9be31823b38e1c24a2d605c7b5
```

中間曾強制 ES 3.2 執行；19 分鐘仍停在 `CaptureFile::OpenCapture` 的 captured-debug-message
處理，且會重入先前已知的 independent color-mask capability 問題，因此由本 session
以 SIGTERM 中止。Car Chase capture metadata 雖是 ES 3.2，現行 PvrGPU 已驗證的 replay
契約是 **ES 3.1 + tessellation extensions**；不能把這次人工中止當成渲染失敗。

### Release/O3 Car Chase 正式結果與 memory

最終以修正版 RenderDoc、現行 Release bridge、ES 3.1 extension 路徑完整重播 event
1–6649，於 **1295.66 秒（21 分 35.66 秒）PASS**。第一筆 counter 明確記錄
`set_patch_vertices patch_vertices=16`；全程沒有 `unsupported_draw` 或
`draw_not_lowerable`。

嚴格證據：395 個 RenderDoc draw actions、56 份 graphics reports、78 graphics
submissions、5 compute dispatches；model aggregate 為 400 drawlists。model 56 組
hello/counter/done 完整，32,014,371 次 pool allocations/releases 完全平衡，
`pool_leaks=0`。player/model stderr 都是空檔，final receipt 為 API errors=0、player/model
exit 0。`frame.png` 與 player completed-attachment PNG 都是 1920x1080，SHA-256 同為
`f8255136ca0b543edc896cb707c4da0121cde35b61fd2a7c5a77e04647b13433`。

每 5 秒 RSS 與每約 60 秒 macOS physical-footprint 結果：

```text
RSS peak                  9,573,616 KiB (~9.13 GiB)
last pre-exit RSS         7,720,624 KiB (~7.36 GiB)
physical-footprint peak   3,774 MB (~3.69 GiB)
last physical footprint   2,194 MB (~2.14 GiB)
max model pool high-water 948,157,934 bytes (~904 MiB)
```

physical footprint 在工作峰值後回落到約 2.1 GiB，pool ownership 也完全平衡；沒有
capacity-retention leak 的型態。RSS 峰值低於同一 Release bridge 的 Manhattan
11.56 GiB，但兩個 capture 的 allocation shape 不同，只能分別看壓力，不是性能 A/B。

與既有 llvmpipe event-6649 reference 比較仍不是 bit-exact：2,073,600 pixels 中
1,385,774 不同（66.829379%），RGB MAE 2.071891/255，單 channel 最大差 181。相較舊
native full run 記錄的 69.5387%、RGB MAE 2.2054、最大 214 有改善，但仍必須把
**strict replay PASS** 與 **reference attachment exact** 分開報告。

完整 artifacts：

```text
/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/out/runs/rdc_regression/carchase_release_memory_20260911_fixed_gles31/
```

## 下一階段：改成硬體級 render-pass-wide TBDR

目前的 `draw_pco_sequence` 雖然能一次攜帶多個 draw，但 submitter 在每個 draw 送進
pipeline 後仍等待 `sequence_completion_`，因此現行排程實際上是：

```text
draw 0: tiler -> renderer -> PBE -> wait
draw 1: tiler -> renderer -> PBE -> wait
...
```

這只能稱為 **draw-level two phase**；不能把它描述成真正硬體的、整個 render pass
共用 tile lists 的 TBDR。Manhattan 的 219 個 model drawlists 也不是先全部完成 tiling
才開始 tile rendering；該次 run 被切成 36 個 graphics submissions，而且 submission
內仍採上述逐 draw 完成模式。現行路徑依靠 attachment LOAD/STORE 在 draw 之間維持
結果，功能上可通過 regression，但會造成重複 framebuffer/depth traffic、失去跨 draw
HSR 機會，memory/time/counter 行為也不等同硬體。

後續重構的明確目標：

1. 建立 compatible render pass/sequence 共用的 tile-list 與 tile-local attachment state。
2. Phase 1 依 draw order 執行所有 geometry/tiler，將 primitive 累積進共同 per-tile lists。
3. 所有 tiling 完成後設置 pass-wide barrier，再啟動 Phase 2。
4. Phase 2 每個 tile 依原始 draw/primitive order 執行 depth、stencil、blend，最後只做一次
   PBE/store；不得藉重排破壞 API ordering。
5. 只在 framebuffer/attachment 不相容、resolve/readback、compute/resource hazard、明確
   flush，或 tile/parameter memory 壓力需要 partial render 時切斷 render pass。
6. 加入 bounded tile-list/parameter-buffer spill，避免「先吞全部 draw」反而造成無上限
   peak memory。
7. 完成後重跑 dEQP texture/depth/stencil/blend、Manhattan 與 Car Chase，除 correctness
   外比較 graphics submission、PBE 次數、HSR、bandwidth、RSS/physical footprint 與時間。

對應程式內的 `TODO(pvrgpu-tbdr)` 位於 `model_stub/submitter.cpp` 的逐 draw
`sequence_completion_` wait 前。移除該 wait 本身不是修正；必須先完成以上共享
render-pass state、barrier、hazard splitting 與記憶體上限機制。

### 真實 TBDR 的 debugability 是實作前置條件

render-pass-wide TBDR 會把錯誤產生點與畫面出錯點隔開：某個 draw 在 Phase 1 寫壞
tile list，可能到累積完後的 Phase 2 才顯示錯誤，且後續 draw 還可能覆蓋或遮住問題。
因此不能在新路徑完成後刪除現行逐 draw 排程；兩條模式都必須保留：

- `draw-serialized`：現行慢速 reference/debug 模式，可逐 draw 停止及二分定位。
- `render-pass-tbdr`：硬體級正式模式，用於正確性、效能與記憶體 regression。

切換必須是一般性的 runtime/debug option，不得依 capture 名稱、draw index 或測試名稱
選擇渲染答案。開始硬體級 TBDR 重構時，應同步完成以下工具，不能留到最後才補：

1. 可在指定 draw 後強制 partial render/flush，以二分出第一個污染狀態的 draw。
2. 可 dump 每個 draw 對 per-tile list 的 delta，以及指定 tile 的 draw/primitive ordering。
3. Capture/Play 能分別保存並重播 Phase 1 與 Phase 2；snapshot 必須包含 shared tile
   lists、parameter buffer、tile-local attachments、draw ordinal 與 phase/barrier 狀態。
4. 每個 render pass 記錄 attachment LOAD/STORE、barrier、partial render 與 split reason。
5. 提供相同輸入下 `draw-serialized` 對 `render-pass-tbdr` 的指定 draw、tile、color/depth/
   stencil attachment 比對，先找出第一個 divergence，再進入元件級 trace。
6. debug 強制 flush 不得冒充正式 PASS；正式 regression 必須回到未強制切割的
   `render-pass-tbdr` 模式再次驗證。

## 最重要的禁止事項

- 不得跳過失敗 draw，或把 skipped/unsupported draw 後未改變的 attachment 算成 PASS。
- 不得把 llvmpipe state、pixels 或 counters 注入 PvrGPU。
- 不得跨新舊 Mesa binary 強行載入 checkpoint。
- **不得在驗證用的 build 目錄執行不帶 `--target` 的 `cmake --build`。**
- 不得覆寫正在使用或已凍結的 driver、bridge、RenderDoc、player、snapshot 目錄。
- 不得以 hash／capture 名稱／draw index 選擇渲染答案。
- 不得因為 dEQP 的品質警告而讓 model 偏離 llvmpipe 實際執行的運算；capture replay
  的 oracle 是 llvmpipe。

## 2026-09-11 glmark2 WIP 修復與回歸

這一輪把 glmark2 WIP 收斂完成，沒有重新啟用 formal player 明確清除的
`PVRGPU_RDC_CASE_NAME` 捷徑。修正內容如下：

- `conditionals`、`lit_mesh`、`texture` 三個專用 PCO compiler 明確宣告
  `fragment_output_mask[0] = 0xf`，emit command 時完整傳輸八個 render-target
  mask。不要在 generic color compiler 裡重跑 `nir_shader_gather_info()`；那會把
  dEQP multisample texture query 人工建立的 `num_textures` metadata 清成零。
- `draw_textured_triangles` 命令新增 canonical `depth_state`，driver、文字 parser、
  SystemC API deep-copy、submitter capture 與 reporter publish 全部接通。這是
  effect2d 的真實修正：model 現在回傳 framebuffer boundary 所要求的 depth plane，
  沒有略過 boundary 或降低驗證強度。
- 合法 Mesa depth format 的 single-draw PCO 同樣 materialize/publish depth plane，
  修復 build、bump、conditionals、shading、texture 的 boundary failure。
- runner 先清除父程序遺留的 output width/height，再只在 explicit/manifest extent
  存在時注入正式 replay；process test 同時覆蓋有 manifest 與無 manifest 的污染環境。

最終正式回歸（2 workers、cache mode、每案 timeout 900 秒）為 **10/10 PASS**：

```text
/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/out/runs/rdc_regression/glmark2_final_10of10_20260911/
total=10 passed=10 failed=0 timeout=0 total_time=578.32s
terrain=483.95s
```

terrain 在 300 秒會 timeout，但單獨與完整 2-worker 回歸都在約 481–484 秒 PASS。
原因是 formal player 會清除 case name，八個 terrain draw 走通用 native PCO/USC ISS，
而不是 capture-name 專用的八-draw sequence。這符合目前「不用 case 名稱選答案」的
規則，但成本高：最終 run 的最大 model pool high-water 為
`1,055,172,624` bytes（約 1006 MiB），觀察到 player RSS 約 1.61 GiB。後續若要降
terrain 時間／記憶體，應優化通用 texture/fragment payload，而不是恢復 case-name
捷徑。

驗證通過：Mesa `pvrgpu_pco_lowering`、`fragment-output-mask-unit`、
`driver-command-unit`、`systemc-api-bridge-unit`、`pco-systemc-api-bridge-unit`、
`rdc-native-runner-process-test`，以及 9 個 runner 靜態契約 unittest。最後沒有殘留
`pvrgpu` 或 `pvrgpu-rdc-player` 程序。
