# Tessellation 原生實作與 dEQP 驗證

2026-09-08，原生 TCS／固定功能 tessellator／TES 已接通。**尚未 all pass，也尚未追平 llvmpipe。** 下列數字只計完整執行的測試輪次。

## 完整 406-case 結果

範圍為 stock CTS 的 `dEQP-GLES31.functional.tessellation.*`，包含所有失敗與不支援案例。

| 執行版本 | Pass | Fail | NotSupported |
| --- | ---: | ---: | ---: |
| PvrGPU 原始基線 | 42 | 358 | 6 |
| PvrGPU native-v1 | 87 | 313 | 6 |
| PvrGPU native-v2（本輪最終） | 89 | 311 | 6 |
| llvmpipe 基線 | 406 | 0 | 0 |

原有 42 個 Pass 全部保留；其中 37 個是 query/API/link、5 個是 negative compile，不能當作 native Tessellation 執行證據。native-v1 新增 45 個 native Pass，native-v2 再修復兩個 unequal patch-size 案例。**不依賴 Transform Feedback 的 47 個繪圖案例現為 47/47 Pass**，全部有真實 TCS／TES execution、modeled memory traffic 與完成的 framebuffer 驗證。

`native-v1` 的 313 Fail 分為：

- 311 個案例需要尚未實作的 Transform Feedback buffer writeback/readback。
- 2 個 unequal patch-size 案例（5→10、10→5）在 TCS signed multiply-high／arithmetic-shift 原生解碼被拒絕；該輪並未執行 Tess draw，不能誤判為 TES 浮點或 raster 誤差。

native-v2 的 signed 指令解碼／執行已修復上述兩例；目前剩下的 311 Fail 全部涉及 Transform Feedback，不能把僅有繪圖支援當成整組完成。

6 個 NotSupported 為 ES 3.2 negative compile 案例；CTS 在實際 context 的 ES 3.2 支援檢查停止，尚未建立 shader 或送出 SystemC 工作。PvrGPU 的 ESSL 310 與 independent blending／texture-buffer 等 ES 3.2 要求尚有缺口，不能只提高版本宣告來把這六例變成 Pass。llvmpipe 能通過，故仍是差距。完整 QPA 狀態一致不等於所有輸出 bytes 都與 llvmpipe 相同；下述固定功能 differential 才是明確的逐位元比較。

## 架構與演算法

graphics chain 現為 VS → `TessellationControlShader` → `Tessellator` → `TessellationEvaluationShader` → `GeometryShader` → ClipCull。TCS、固定功能與 TES 各有自己的 `.h/.cpp`、event-driven `SC_THREAD`、bounded POD/PoolHandle FIFO。沒有對應 stage 時才原樣 bypass，enabled stage 不呼叫 VS/FS/CS shader executor。

Mesa compiler 保留真正的 `MESA_SHADER_TESS_CTRL`／`MESA_SHADER_TESS_EVAL`。TCS 執行 native LD/ST/WDF/NOP.end，TES 執行 LD/WDF/UVSW；共用純 PCO ALU/encoding helper，不使用 host NIR/GLSL 解譯、LLVM JIT、預先算好的頂點或測試名稱分支。

固定功能採用同 pinned Mesa/llvmpipe 使用的 Microsoft MIT tessellator，保留授權；涵蓋 triangle、quad、isoline，equal／fractional-even／fractional-odd spacing、winding、point mode，以及負值／NaN discard 和最大值 clamp。triangle W 按參考實作的 `(1-U)-V` 順序產生。

TCS input scratch、per-vertex／per-patch output 與六個 levels 均經 `GpuMemorySystem` 實際 LD/ST；固定功能讀取真實 levels、寫入 domain UV，TES 再從相同 GPU address 讀取 UV 與 patch outputs。pool 只承載受控 payload／索引，不取代 GPU memory traffic。TCS task 逐 instruction group 鎖步，TES 每 task 最多 32 個 domain points。

API25 深拷貝 TCS/TES binary、shared 與五階段 UBO snapshots；API24 在讀取新增 tail 前拒絕。上限是每 patch 1–32 個 input/output control points、每 draw 4,096 patches、131,072 input occurrences、1,048,576 domain points、6,291,456 indices。每 instance 不足一個 patch 的尾端不產生 primitive。完整 ABI 見 [driver command contract](PVRGPU_DRIVER_COMMAND.md)。

`hs_invocations` 計完成的 patches，`tcs_invocations` 計實際 control lanes，`ds_invocations` 計實際 evaluation points；`primitives_generated` 計 fixed/TES 產生的完整 primitives，不以 input patches 或 GS counters 代替。

## 本輪修正

- Compiler tess-level array scalarization：最初只有一個真實 ST，poison memory／native write mask 抓到其餘五個 level 未寫入；在 generic linking 前正規化 level arrays，現在產生六個不同地址的 ST。
- 共用 partial-store cache bug：新資料可能只在 dirty SLC、尚無 DRAM backing；後續 partial store 原本錯誤清空同一 line 的先前資料。現在先查 resident cache，只在真正的未初始化 miss 才建立零底稿。direct／bypass／cache 都驗證 exact readback。
- 共用 `glFrontFace` 分類：native driver 的 window-Y reflection 必須同時交換 CW 與 CCW，原先將兩者折疊成同一方向。修正後 winding 4/4 Pass；另以普通 VS 與 TES metadata、兩種 viewport 符號、所有 cull modes、front/back stencil 的完整路徑回歸。
- Signed integer lowering：增加真正 IMADD64.s high 與 ASR_TWB，保留 signed multiply、高 DWORD addend、shift count 與 sign extension；非支援 modifier／OLCHK 仍拒絕。所有 native validator、Step 與純 ALU 入口驗證旗標值域及 opcode，VS/FS 拒絕此 native-task-only 旗標。實際 unequal patch-size binaries 納入 native ISS fixtures，並用 signed／unsigned 獨立數學 oracle 驗證。

## 分層驗證與證據

以下 checks 不是 dEQP cases，不能加進 406 的 Pass 數。

最終中央 full build 完成；CTest **108/108 Pass**（`ctest-final.log`），Python **83/83 Pass**（`python-final.log`）。新增完整 draw API 在 direct／bypass／cache 各自真實執行、驗證借用 payload 被覆寫後仍可正確 readback；另測試不足一個 patch 時零 HS/TCS/DS 與零生成 primitive，完整 clear attachment 與 pool balance 不變。

| 驗證 | 結果 |
| --- | --- |
| Fixed-function invariant checks | 41,897,828 checks／5,489 patches Pass，ASan/UBSan |
| 直接使用原 Mesa source 的 differential | 41,919,784 checks／5,489 patches Pass；UV bits、counts、indices 與 winding 全部 exact，ASan/UBSan |
| Fixed SystemC module | 3,844 checks／108 draws／324 patches Pass；direct、bypass、cache，另 7 個負例各 7 checks |
| Native TCS/TES ISS（含 signed lowering） | 191,376 checks Pass，ASan/UBSan；真實 compiler binaries、poison storage、dynamic indexing／UBO、task width 1/7/32 |
| TCS→fixed→TES module chain | 11,023 checks／18 draws／36 patches Pass，ASan/UBSan；另 8 個拒絕案例各 6 checks |
| Native TCS/TES compiler | 177 checks Pass（97 功能 + 80 fixture I/O） |
| API25 boundary harness | 932 checks Pass，亦有 ASan/UBSan harness；中央 bridge 本身非 sanitizer build |
| Signed decoded-metadata 契約 | 4,372 checks Pass，ASan/UBSan；CS/GS/TCS/TES 的 validator／Step／純 ALU 拒絕非法旗標，另有 VS/FS 279 次逐指令拒絕回歸 |
| Driver front-face／culling／stencil | 22,018 checks／64 cases Pass，ASan/UBSan |
| Graphics statistics helper | 26 checks Pass，ASan/UBSan |

`native-v1` 獨立稽核確認三輪共 1,218 份 QPA 的精確 CasePath/result、相同 CTS binary／flags、renderer 與每例 DYLD 實際載入路徑，各輪 runtime 前後 hash 不變。45 個新增 native Pass 的 HS/TCS/DS、ALU/MEM、DrawList 與 driver completion 均一致，domain write/read = DS×8 bytes、level read = HS×24 bytes，GS counters 為零；全輪 pool allocation/release = 8,027/8,027，leaks 與 bytes-in-flight 均零。

最終 `native-v2-audit` 再逐案核對 Tess 四輪（406×4）及 Geometry 三輪（207×3），validation errors=0。47/47 native render Pass 均符合上述執行與 memory/counter 契約；全輪 HS=371、TCS=1,306、DS=44,330，pool allocation/release = 8,073/8,073，零洩漏。兩個新增 unequal patch-size Pass 各有 HS=1、DS=36、generated primitives=50，TCS 分別為 10 與 5 lanes，沒有用固定 primitive count 代替真實 execution。

Geometry 完整 207 案 `pvrgpu-tess-regression-v1`、`pvrgpu-tess-regression-v2` 皆維持原 GS `native-v3` 的 168 Pass／29 Fail／10 NotSupported，各輪 runtime hash 穩定。這不是對所有 FBO、MSAA、SSBO、atomic 或 Compute 完整 dEQP groups 的重跑聲明。

Stock CTS SHA256：`9ac9ca1efae4c5194d59663e22af6c455575c1a7b795bd0024b89fc083ab128d`。
406-case manifest SHA256：`ed3bcff94f604b7ab2375a411651e2ec2608542d56e6654f63014f6b63bb6e22`。
兩個 Mesa checkout 皆基於 `da14d65e4499e66468094be52bff9ea0915a695e`；PvrGPU 有 native GS/Tess compiler patches，llvmpipe 僅保留先前 GS zero-output statistics patch，不是相同 source tree。

完整 artifacts 位於工作區外：

`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/out/runs/deqp_groups/tessellation_v1/`

`run_baseline.py` 保存 runner；各 `*-baseline/`、`pvrgpu-native-v1/`、`pvrgpu-native-v2/` 保存 immutable receipt 與逐例原始結果；`native-v1-audit.py/.json/.md`、`native-v2-audit.py/.json/.md` 為可重現的獨立稽核，`fixed-tessellator-validation.md`、`native-shader-validation.md`、`front-face-and-signed-isa-validation.md` 保存專項 sanitizer／參考原始碼證據。不可用後續磁碟上的 runtime hash 取代歷史 receipt。

## 未完成範圍

下一個主要缺口是 native Transform Feedback：必須保留 TF-only TES exports，以獨立 modeled-memory writer 寫回真正原生輸出、正確處理 buffer ranges／append／whole-primitive overflow，並以完成 generation 更新 readback 與 written/needed queries。不能以 `primitives_generated`、host shader 結果或空 buffer 冒充 TF 支援。

目前 Tess+GS combined、TCS/TES texture、SSBO/image/atomic、indirect、primitive restart 與 Tess MRT 亦未建立完整支援；相關路徑維持明確拒絕，不宣稱完整 GLES conformance。
