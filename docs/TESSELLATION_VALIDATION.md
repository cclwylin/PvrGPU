# Tessellation 原生實作與 dEQP 驗證

2026-09-08，原生 TCS／固定功能 tessellator／TES 與 TES Transform Feedback 已接通。最新 **中央 API27 完整 406 案為 400 Pass／6 NotSupported／0 Fail／0 NoResult**，獨立逐案稽核通過；6 個 ES 3.2 capability 差距仍未追平 llvmpipe。下列歷史與最終中央 runtime 各有自己的 immutable receipt，不互相繼承結果。

## 完整 406-case 結果

範圍為 stock CTS 的 `dEQP-GLES31.functional.tessellation.*`，包含所有失敗與不支援案例。

| 執行版本 | Pass | Fail | NoResult | NotSupported |
| --- | ---: | ---: | ---: | ---: |
| PvrGPU 原始基線（歷史） | 42 | 358 | 0 | 6 |
| PvrGPU native-v1（歷史，尚無 TES TF） | 87 | 313 | 0 | 6 |
| PvrGPU native-v2（歷史，尚無 TES TF） | 89 | 311 | 0 | 6 |
| PvrGPU tes-tf-native-v1（歷史 private 第一輪） | 379 | 18 | 3 | 6 |
| PvrGPU tes-tf-native-v2（歷史 private API26 最終） | 400 | 0 | 0 | 6 |
| PvrGPU final-central-v1（最新中央 API27） | 400 | 0 | 0 | 6 |
| llvmpipe 基線（歷史比較） | 406 | 0 | 0 | 0 |

歷史 native-v2 保留原有 42 個 Pass；其中 37 個是 query/API/link、5 個是 negative compile，不能當作 native Tessellation 執行證據。歷史 native-v1 新增 45 個 native Pass，native-v2 再修復兩個 unequal patch-size 案例。當時不依賴 Transform Feedback 的 47 個繪圖案例為 47/47 Pass，全部有真實 TCS／TES execution、modeled memory traffic 與完成的 framebuffer 驗證。

歷史 `native-v1` 的 313 Fail 分為：

- 311 個案例需要尚未實作的 Transform Feedback buffer writeback/readback。
- 2 個 unequal patch-size 案例（5→10、10→5）在 TCS signed multiply-high／arithmetic-shift 原生解碼被拒絕；該輪並未執行 Tess draw，不能誤判為 TES 浮點或 raster 誤差。

歷史 native-v2 的 signed 指令解碼／執行已修復上述兩例，當時剩餘 311 Fail 全部涉及 Transform Feedback。本輪 TES TF 與下述三項修正後，完整重跑已不再有 Fail／NoResult。獨立稽核逐案確認 406 個精確 CasePath、每案一份 QPA 結果、無 timeout 或 unsupported event，以及 runtime 執行前後 hash 不變；不能把上述歷史失敗分類當作目前缺口。

6 個 NotSupported 為 ES 3.2 negative compile 案例；CTS 在實際 context 的 ES 3.2 支援檢查停止，尚未建立 shader 或送出 SystemC 工作。PvrGPU 的 ESSL 310 與 independent blending／texture-buffer 等 ES 3.2 要求尚有缺口，不能只提高版本宣告來把這六例變成 Pass。llvmpipe 能通過，故仍是差距。完整 QPA 狀態一致不等於所有輸出 bytes 都與 llvmpipe 相同；下述固定功能 differential 才是明確的逐位元比較。

## 最終中央 API27 驗證

`final-central-v1` 使用中央 graphics API27／Compute API4 coherent runtime，3 個 workers，於 **2026-09-08 13:20:23–13:24:35（臺灣時間）**完整執行，耗時 252.388 秒。406 案各有精確 CasePath、完整 QPA 與 exit code 0；runtime 前後 SHA-256 相同。這是新的中央實測，不是沿用 private API26 的結果。

| 案例性質 | 數量 | 原生證據 |
| --- | ---: | --- |
| TES Transform Feedback | 311 Pass | 真正 TES bindings、StreamOutput readback、TCS／TES 指令及 query／QPA 驗證 |
| 不依賴 TF 的 Tessellation 繪圖 | 47 Pass | 真正 TCS／TES、DrawList／memory counters 與 framebuffer 驗證 |
| Query／API／negative compile | 42 Pass | 保留 API 驗證，不冒充 shader 執行 |
| ES 3.2 negative compile capability gate | 6 NotSupported | 無 shader/model 工作；與 llvmpipe 的 406 Pass 仍有差距 |

新稽核 `audit-final-tessellation.py` 獨立核對 stock discovery、406-case manifest、相同 CTS binary、原始 QPA、每案 command 的中央 library 路徑、預先指定的 frozen runtime hashes，以及每筆 DrawList 的實際 TCS／TES ALU／memory／invocation totals。巢狀 `user_defined_io.negative.*` 另外分類，不沿用舊 runner 的 render 標籤。

全組實際 **TCS invocations=85,345、TES invocations=15,740,928、patches=59,834、tessellation primitives=23,441,200**；TCS 並未以 patch 數代替。逐筆 counter／done 完整對應，所有 pool allocations=releases、pool leaks=0、bytes-in-flight=0；unsupported events、model errors、timeout 與 audit issues 均為 0。NotSupported 仍如實保留，不改成 Pass。

最終中央 SHA-256：

- SystemC bridge：`1e3a1106af02305ae5010317271cc3b9bf2bb6dc123fe4154a84bfa915c46566`
- Mesa libgallium：`bc0f893d46f7df83d54bec97a295d90aae091a2195b42cab11365be3d73bbc48`
- Audit script：`3d87046e46511a55f22c1f07a94b483bad72638251315232fce1b0da8214d38e`

證據在外部工作目錄 `out/runs/deqp_groups/tessellation_v1/`：`pvrgpu-final-central-v1/runtime-receipt.json`、其下逐案 `results.qpa`／`result.json`／`driver-counter.txt`／`systemc.jsonl`，以及 `final-central-v1-run.log`、`final-central-v1-audit.json`、`final-central-v1-audit.log`。歷史 `pvrgpu-tes-tf-native-v2/` 與 `llvmpipe-baseline/` 保持不變。

相同 frozen runtime 的重跑方法（使用新 label，避免覆寫既有證據；runtime 若更新，必須先取得對應的新 hashes）：

```sh
tess_work="${PVRGPU_WORK_ROOT:-$HOME/Downloads/_Codex/Working/PvrGPU}"
tess_base="$tess_work/out/runs/deqp_groups/tessellation_v1"
tess_label=final-central-recheck-v1
python3 "$tess_base/run_baseline.py" --output-label "$tess_label" \
  --backends pvrgpu --workers 3 \
  --bridge "$tess_work/build/lib/libpvrgpu_systemc_bridge.dylib" \
  --pvrgpu-prefix "$tess_work/tmp/pvrgpu-mesa-install"
python3 "$tess_base/audit-final-tessellation.py" \
  --run "$tess_base/pvrgpu-$tess_label" --work-root "$tess_work" \
  --expected-mesa-sha256 bc0f893d46f7df83d54bec97a295d90aae091a2195b42cab11365be3d73bbc48 \
  --expected-bridge-sha256 1e3a1106af02305ae5010317271cc3b9bf2bb6dc123fe4154a84bfa915c46566 \
  --output "$tess_base/$tess_label-audit.json"
```

## 架構與演算法

graphics chain 現為 VS → `TessellationControlShader` → `Tessellator` → `TessellationEvaluationShader` → `GeometryShader` → ClipCull。TCS、固定功能與 TES 各有自己的 `.h/.cpp`、event-driven `SC_THREAD`、bounded POD/PoolHandle FIFO。沒有對應 stage 時才原樣 bypass，enabled stage 不呼叫 VS/FS/CS shader executor。

Mesa compiler 保留真正的 `MESA_SHADER_TESS_CTRL`／`MESA_SHADER_TESS_EVAL`。TCS 執行 native LD/ST/WDF/NOP.end，TES 執行 LD/WDF/UVSW；共用純 PCO ALU/encoding helper，不使用 host NIR/GLSL 解譯、LLVM JIT、預先算好的頂點或測試名稱分支。

固定功能採用同 pinned Mesa/llvmpipe 使用的 Microsoft MIT tessellator，保留授權；涵蓋 triangle、quad、isoline，equal／fractional-even／fractional-odd spacing、winding、point mode，以及負值／NaN discard 和最大值 clamp。triangle W 按參考實作的 `(1-U)-V` 順序產生。

TCS input scratch、per-vertex／per-patch output 與六個 levels 均經 `GpuMemorySystem` 實際 LD/ST；固定功能讀取真實 levels、寫入 domain UV，TES 再從相同 GPU address 讀取 UV 與 patch outputs。pool 只承載受控 payload／索引，不取代 GPU memory traffic。TCS task 逐 instruction group 鎖步，TES 每 task 最多 32 個 domain points。

API25 起深拷貝 TCS/TES binary、shared 與五階段 UBO snapshots；API26 再承載 owned StreamOutput targets／bindings 與明確 last-stage varying mapping。舊版在讀取新增 tail 前拒絕。上限是每 patch 1–32 個 input/output control points、每 draw 4,096 patches、131,072 input occurrences、1,048,576 domain points、6,291,456 indices。每 instance 不足一個 patch 的尾端不產生 primitive。完整 ABI 見 [driver command contract](PVRGPU_DRIVER_COMMAND.md)。

`hs_invocations` 計完成的 patches，`tcs_invocations` 計實際 control lanes，`ds_invocations` 計實際 evaluation points；`primitives_generated` 計 fixed/TES 產生的完整 primitives，不以 input patches 或 GS counters 代替。

## 本輪 TES Transform Feedback 修正

- Compiler 保留 TF-only TES exports 及原 Gallium/NIR location，映射至真正 native UVSW physical DWORD；FS 未讀取的輸出不配置 interpolation coefficients。FS 真正讀取的部分使用 explicit last-stage bindings，不把輸出 ordinal 當成 physical register。
- 獨立 StreamOutput module 在 TES EvaluationComplete 後使用實際 TES lanes、primitive refs 與 padding，經 modeled GPU memory 寫入；沿用 owned target snapshots、generation／target identity、buffer ranges、append cursors、whole-primitive overflow 與 written/needed query 回讀。沒有 host shader evaluation、預設答案或以 generated count 取代 buffer 寫入。
- 18 個 `user_defined_io.per_vertex_block.*` 的真正 TES group（byte 1770：`56b24041028040004c488304`）為 `LOGICAL.AND r12,r8 -> vi3`。Task-stage decoder 接受有界 writable VTXIN，fragment／graphics-export destination 仍拒絕。PCO allocator 會使用對齊的備用 VTXIN，因此 TCS/TES ABI 保留實際 compiler metadata，接受固定 3／5 DWORD system-value prefix 至 64 DWORD 上限；僅初始化固定 prefix，備用字首次原生寫入前的讀取仍拒絕，不無條件擴大或 round ABI。定點 18 案與最終全組皆 Pass。
- 3 個 ParameterBuffer NoResult 是浮點 area 變為零、但量化 raster area 非零時，常數平面的 `0 * Inf` 產生 NaN。僅對三個有限且相同的輸入建立 derivatives=0、origin=value；非恆定分支保留 llvmpipe float arithmetic 與 finite gate。沒有 epsilon cull、double fallback、刪除 primitive 或修改 TES／TF 原始資料。
- Gallium query 失敗時，Mesa frontend 原本丟失 terminal failure 並持續 `get_query_result(wait=true)`，造成無窮等待。可重建的 `third_party/mesa-26.2.1-failed-query-state.patch` 記錄 Failed、退休 query resource 並平衡 active count；結果 getter 保留 GL_OUT_OF_MEMORY 且不改寫 caller／query-buffer memory，同物件重新 Begin 可重用。未回傳成功零 query 冒充正常執行。Post-OOM 退休是此實作的錯誤處理策略，不宣稱跨實作的 OOM 後狀態保證。

private API26 原始證據目錄：

`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/out/runs/deqp_groups/tessellation_v1/`

完整 406 案 receipt／QPA 在 `pvrgpu-tes-tf-native-v2/`；`tes-tf-native-v2-audit.json` 與 `audit-tes-tf-v2.py` 保存逐案獨立稽核；`tes-tf-validation.md` 保存重現指令與私有 runtime 邊界；`failed-query-validation.md` 保存錯誤路徑驗證。`pvrgpu-tes-tf-native-v1/` 保留第一輪 379P／18F／3NoResult／6NS，不覆寫歷史結果。

最終 private bridge SHA256：`4ee6f3fc6ece7243b055d9e1720810fc4c1155d610e59fee42449b7626ffa5c9`。
最終 private Mesa Gallium SHA256：`c00b009002c71d1770321692fbe5b6eb20e15fe90f69675aefb643f92a5d1595`。
對應 runtime 在相鄰 `tess_and_private_v1/`；EGL/GLES 是獨立複本，不透過 shared-install symlink。歷史私有 linker 必須使用匹配的 API26 headers／objects，不能混用後續 API27 objects。

該 private 輪次補充 checks（不是 dEQP case 數）：native TCS/TES ISS **191,387 checks Pass**，含實際 AND group、writable VI5、uninitialized-read／ABI bounds／非法 bank 負例（`../tess_and_private_v1/unit-test.log`）；TES compiler **241 checks Pass**（`tes-tf-private-v1/test.log`）；真 Mesa query frontend **23 checks Pass、3 objects destroyed、0 result polls**（`tes-tf-private-v1/frontend-test.log`）。真 EGL/GLES failed-submit probe 亦 Pass，驗證 untouched TF buffer／query params、重用與刪除（`failed-query-native-v1/`）。中央 API27 的新完整結果與 immutable receipt 見本頁最終驗證節。

## 歷史 native-v1／native-v2 修正

- Compiler tess-level array scalarization：最初只有一個真實 ST，poison memory／native write mask 抓到其餘五個 level 未寫入；在 generic linking 前正規化 level arrays，現在產生六個不同地址的 ST。
- 共用 partial-store cache bug：新資料可能只在 dirty SLC、尚無 DRAM backing；後續 partial store 原本錯誤清空同一 line 的先前資料。現在先查 resident cache，只在真正的未初始化 miss 才建立零底稿。direct／bypass／cache 都驗證 exact readback。
- 共用 `glFrontFace` 分類：native driver 的 window-Y reflection 必須同時交換 CW 與 CCW，原先將兩者折疊成同一方向。修正後 winding 4/4 Pass；另以普通 VS 與 TES metadata、兩種 viewport 符號、所有 cull modes、front/back stencil 的完整路徑回歸。
- Signed integer lowering：增加真正 IMADD64.s high 與 ASR_TWB，保留 signed multiply、高 DWORD addend、shift count 與 sign extension；非支援 modifier／OLCHK 仍拒絕。所有 native validator、Step 與純 ALU 入口驗證旗標值域及 opcode，VS/FS 拒絕此 native-task-only 旗標。實際 unequal patch-size binaries 納入 native ISS fixtures，並用 signed／unsigned 獨立數學 oracle 驗證。

## 歷史 native-v1／native-v2 分層驗證與證據

以下 checks 不是 dEQP cases，不能加進 406 的 Pass 數。

該歷史輪次中央 full build 完成；CTest **108/108 Pass**（`ctest-final.log`），Python **83/83 Pass**（`python-final.log`）。當時新增完整 draw API 在 direct／bypass／cache 各自真實執行、驗證借用 payload 被覆寫後仍可正確 readback；另測試不足一個 patch 時零 HS/TCS/DS 與零生成 primitive，完整 clear attachment 與 pool balance 不變。這些不是本輪中央 API27 的測試數字。

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

TES Transform Feedback 已通過 private API26 與本輪中央 API27 各自獨立的完整 406 案回歸，已不是原先 311 Fail 的缺口；仍有 6 個 ES 3.2 negative compile NotSupported，與 llvmpipe 的 406 Pass 不同。

目前 Tess+GS combined、TCS/TES texture、SSBO/image/atomic、primitive restart 與 Tess MRT 亦未建立完整支援；相關路徑維持明確拒絕，不宣稱完整 GLES conformance。Driver 已有 bounded single-command indirect 入口與獨立 VS TF 驗證，但本次 Tess406 不等於完整 Tessellation indirect matrix 驗證；multi-draw／indirect-count／count-from-stream-output 仍不支援。
