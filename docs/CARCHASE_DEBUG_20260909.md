# GFXBench Car Chase capture 1 debug

## 本次進度摘要

- 最新driver17/model15 run `native-v17-depth17-s32-full6649`已完成
  Draw0–394共395個action ranges，以及trailing event6649；exit0、沒有timeout，
  bounded native evidence audit通過、errors為空，已輸出1920×1080最終PNG。
  這是functional ordered replay完成，不是畫面bit-exact PASS；零instance
  indirect no-op也包含在395個actions內，不能全部稱為有GPU工作的draw。
- Draw360的兩個阻擋均已修正：CB0只傳送實際使用的91個DWORD，將總SH需求
  從272壓至255；一般non-shadow CubeArray depth view轉為既有RGBA32F
  transport。Actual17直接確認slot4是Z32_UNORM fallback，已完成360–369。
  不過這十個draw的PS/SMP皆0（無coverage或全部depth rejected）；新增FS
  texture sampling路徑的執行正確性由獨立live fixtures證明，不能混稱capture證據。
- 最終PNG與llvmpipe有1441954／2073600個像素不同（69.5387%），RGB平均
  絕對差2.2054／255、最大214／255；不可只看平均值忽略少數大差異。
  本輪44份可對照附件中5份raw完全相同。先前Draw171／175的color／D24
  差異仍未解決，下一階段是逐draw數值／coverage correctness。
- 本輪每32 draws讀回附件，所有ordered ranges／finish／API觀察照常執行。
  沒有使用whole-state checkpoint，也沒有跳過前段draw。Car Chase的D16
  renderbuffer完整snapshot codec仍未實作，現有Manhattan checkpoint不可套用。
- 未改原capture、未更新預設Mesa prefix、未commit／push。下文保留各階段
  的歷史邊界與證據限制；較早的「尚未完成」不代表最新run仍在執行。

### 較早階段摘要（歷史紀錄）

- 已修正並重播驗證 RenderDoc 的 EXT／OES patch-parameter alias 分派錯誤；
  patch 與 6 項 source regression 留在 repository，實際 library 使用私人 build。
- 新增一般化 `gl_VertexID` 輸入傳遞，並修正零 sample 奇異三角形的平面建立，
  PvrGPU 已以新版同步 GL observer、GLES3.1 完成 Draw 0–155（event1903），
  共156 ordered ranges／157乾淨 API scopes；llvmpipe 也完成相同邊界。
  有 native submission／completion 及 D16／D24 讀回紀錄，但圖片仍有差異。
  尚未完成全部 395 actions。
- Draw 0–27 的 28 張 native 深度圖與 VertexID 修正前逐 byte 相同。
  llvmpipe 對照仍有幾何／深度差異，不能標成 bit-exact PASS。
- llvmpipe 被動記錄原本 TES 執行的輸出，確認第一個 coverage 差異不是
  transform-feedback 診斷重跑造成；精確的 shader stage／運算根因仍待定位。
- Draw 39 的平面例外已排除：只對確定不涵蓋任何啟用 sample 的三角形保留
  inactive placeholder，沒有略過 draw。Draw 39–53 的 D16 與 llvmpipe
  完全相同；前 39 張 native raw 也與修正前一致。
- Draw 75 的零 attributes VS、未綁定 FS color outputs、depth-only mask 及
  原生 `textureGather` 已接通。一般 gather 336 與半 texel 邊界 576 live draws
  在 binary32 定址版全部符合 llvmpipe。Draw 77 的 branch／texture
  continuation TEMP 誤拒絕也已修正，保留必要狀態檢查。Draw156 的新阻擋是
  混合格式 MRT（API31 已實作，進行 live 整合驗證）；TCS／TES texture
  sampling 已通過真 GL 測試。Draw102 在 GLES3.1 的 color0 已逐 byte 符合
  llvmpipe，修正 GLES3.2 override 下的 color-mask state 還原問題。
  未提交／push Git。

### 新發現：Draw102 清色失效與既有觀察缺口

Metadata-only capture extraction 確認原 texture ResourceId438 mip0/layer1
的3,686,400 bytes 與 native Draw102、103 color0 完全一致：SHA256
`f6c525eefd17f73b589284cccf5e184d0123d6c94c2f057dab39fbdf9dba1865`。
事件873–877只有 invalidate DEPTH、attach layer1、clearColor(0)、clear
COLOR|DEPTH，沒有捕捉到把 color mask 關掉的呼叫。Native 沒有把這層清掉。

獨立小型 GLES fixture 重現：在 native 強制 GLES3.2 下，RT0–3 的 indexed
GL_COLOR_WRITEMASK 查詢回1280而不改寫輸出，glColorMaski 卻成功；以零初始化
的儲存陣列還原便得到全關 mask。llvmpipe 同流程零錯誤且還原全開。這是
RenderDoc 的 core-version capability promotion 與 native 不具備的
independent-blend capability 的交互問題；不可只宣告 cap 來掩蓋缺少的實作。
Fixture 只查詢／設定狀態，不是 draw 或 native replay PASS。
證據：WORK `tmp/carchase-color102-audit.Qavn9X/`，其 `capability-live.2VWO4L/`。

RenderDoc 的 GLRenderState Fetch/Apply 會內部清除 sticky GL errors；既有
frozen bounded helper 未在 ReplayEventRange 外掛同步 GLDebugScope。因此
歷史 `evidence_audit_ok=true` 只代表原本 driver/model/readback 檢查通過，
不能證明區間內完全沒有 GL API errors。保留原 artifact，不改寫舊結果；
建立新 observer helper，並檢查 alias 修正後 GLES3.1 的實際相容性。

### 更新：嚴格 GL 觀察與 Draw155 對照

新版 helper 使用既有 `GLDebugScope`，完整 preflight10946個 captured calls，
在 initial-state restoration 與每個 actual replay→finish→readback 區間記錄
同步 API errors、same-context 與 observer restoration。Bootstrap 邊界無錯誤。
`native-model7-strict-gles31-prefix155` 與 LP `strict-ovr-prefix155` 各156 ranges、
157乾淨 scopes；helper 的 observer/readback source bytes 完全相同。

llvmpipe 嚴格測試另外抓出 RenderDoc depth-only FBO discard 的 bug：原碼在沒有
COLOR_ATTACHMENT0 的 FBO 上仍查詢 NUM_VIEWS_OVR。新增一般化 patch 先找實際
texture attachment（colors→depth→stencil）再查詢，保留 extension guard／view
count，沒有清除或忽略錯誤。新7項回歸與原 alias6項通過，333個受保護輸入不變。
新私人 RenderDoc SHA256：
`979a494d4d1ab7a090649b45414c54490841d44401ac676ad55e8d3d49df5e05`。

Draw102 原生 color0 現為全0，與 llvmpipe 相同。0–101 的 native raw 不變；
102–150 與舊3.2 run 的變化僅為color，depth未改變。新對照共228對 raw，
25對完全相同，其餘仍需逐像素追查，不能標成畫面 PASS。Draw155 的兩張PNG
均可見天空／地形的圓形中間附件，不是最後畫面。

獨立報告：debug 工作目錄 `strict-gles31-prefix155-comparison.json`，SHA256
`9ee97b799f773f7bbced86baf8da0edcf0d62ee62eab3677d1b8845b827954c0`。
仍使用 compile-only OpenCapture initialization；這不是完整 native initial-copy
認證，也不是正式 whole-state replay checkpoint。

## 固定輸入

- Capture：`/Users/linwanyi/Downloads/_Codex/GPU_TestPatterns/5.gl_4/recorder/trace/gl_4_capture_1.rdc`
- SHA256：`7c10e7ba74f0ca4dbd452d741644d72b35a0bddd11844875d48e212167934867`
- `BenchScope-public/scripts/lib/gfxbench-target.sh` 明列 `gl_4` 為 Car Chase；runtime verification 對應 `car_chase/scene_4.xml`。
- Capture 初始化資料記錄 OpenGL ES 3.2。RenderDoc census 有 395 個 Drawcall actions，最後 Draw 394 為 event 6629；event 6649 是 End of Capture，兩者邊界不同。不能與錄製報告的 429 個 Gallium drawlists 混為一談。
- 原生診斷工作目錄：`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/carchase-debug.t3s25m`
- llvmpipe／RenderDoc 診斷工作目錄：`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/carchase-reference.morh1s`

本次沒有修改原 capture、預設 Mesa prefix 或既有 frozen runtime。

## 第一個根因：RenderDoc 把 tessellation API alias 分派到 point API

第一個 draw 的真實事件順序：

| Event | Capture API | 參數 |
| --- | --- | --- |
| 31 | `glPatchParameteriEXT` | `GL_PATCH_VERTICES = 16` |
| 32 | `glDrawElements` | `GL_PATCHES`，33824 個 `UNSIGNED_SHORT` indices |

Pinned RenderDoc 的 `renderdoc/driver/gl/gl_driver.cpp` 錯把
`glPatchParameteriEXT`／`glPatchParameteriOES` 放在
`Serialise_glPointParameteri` 的 switch case 群組，而非
`Serialise_glPatchParameteri`。

因此重播發出 `glPointParameteri(GL_PATCH_VERTICES, 16)`，Mesa 回報
`GL_INVALID_OPERATION`，而 patch size 留在預設值 3。
同步 GL debug callback 的 backtrace 明確經過
`WrappedOpenGL::Serialise_glPointParameteri`，不是附件讀回工具或
`RenderState::ApplyState` 額外發出的呼叫。

這同時解釋了兩個現象：

1. llvmpipe 與 PvrGPU 都留下 GL error 1282，附件讀回稽核拒絕繼續。
2. 原生 driver 看到 11274 個完整三頂點 patches，超過 4096 的容量限制；
   正確的十六頂點 patch 數其實是 `33824 / 16 = 2114`，未超出限制。

把 GLES override 從 3.1 改成 capture 的 3.2，仍可重現相同錯誤，
證明不能只靠版本 override 修好分派問題。

修正記錄在
[`renderdoc-mesa-patch-parameter-alias.patch`](../third_party/renderdoc-mesa-patch-parameter-alias.patch)：
兩個 alias 與 core API 使用相同的 patch serializer；point serializer
保持原樣。沒有省略 captured calls、清除錯誤以取得 PASS，或依 capture 名稱分支。

## 作廢的中間實驗

`native-draw0` 保存原始容量拒絕；`native-capacity-draw0` 曾暫時擴大容量，
完成一次 native submission，但仍被錯誤的 patch size 與 GL error 污染。
該次 11274 HS／45096 DS、clipping 和附件結果都**不是有效 Car Chase reference**。

在查明 API alias 根因後，已完整撤回這次容量、位址配置及相關測試的修改，
保留原本 4096 patch 上限。私人 `driver-o2g`、`runtime` 及其測試紀錄僅為
作廢實驗，不可當作最後修正版本或完整 frame 通過證據。

容量實驗的 24 項普通 CTest 通過，不等於修好 capture。
另一次 sanitizer 實驗記錄於
`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/tess-capacity-sanitizers.Uzy1St/RECEIPT.md`；
其中 API boundary fixture 的既有未對齊 struct-reference 問題與此次 alias
修正分開處理，未變更該 fixture。

## 後續驗證順序

### Alias 修正後的第一筆實際結果

`native-alias-fixed-draw0` 使用原本的 O3 driver／Contains bridge，
搭配修正後的 RenderDoc 和 capture 相符的 GLES 3.2 override。
`fixed-prefix0` 是使用同一 replay library 的 llvmpipe 對照。

| Counter | PvrGPU | llvmpipe |
| --- | ---: | ---: |
| IA vertices | 33824 | 33824 |
| IA primitives／HS patches | 2114 | 2114 |
| DS invocations | 8456 | 8456 |
| Clipping input primitives | 4228 | 4228 |
| Clipping output／setup triangles | 163 | 163 |
| VS invocations | 24168 | 24644 |
| PS invocations | 8074 | 8306 |

原生有一筆提交及其完成報告，pool bytes in flight 為 0；附件狀態查詢的
preexisting GL error 為 0。VS／PS 計數差異與實際附件比較必須分開：

- Native tessellation vertex cache 每個 instance 重設；llvmpipe 的
  `draw_pt_vsplit` 每至多 1024 indices 重設。兩者的 256-entry cache
  因此可能有不同 VS invocation 數，不能直接當作 vertex 輸出錯誤。
- Native 有 8565 個 covered primitive-fragment candidates，260 個 depth
  rejects，8305 次成功 depth writes；另有 231 個後來被覆蓋的 owner
  被 opaque HSR 移除，最後才執行 8074 次 FS。
- llvmpipe 按逐 primitive early depth 後的 lanes 計 PS，所以與 native
  最後可見片段數的差異有計數語意因素。比較 8306 與 native 的 8305
  depth passes 仍有差 1，需 raw D16 判斷；aggregate 差 1 不代表只錯一像素。

第一筆是 960×960、單樣本 D16 depth-only pass。舊 bounded helper 只支援
D24 的直接深度讀回，因此早期 prefix 稽核在 snapshot scope 失敗，
原因為 `unsupported_depth_contract_requires_single_2D_D24_sample`。
這不是已比對通過的深度圖，也不是完整 frame 通過。

修正後 RenderDoc library SHA256：
`b1fed12fd6a6a856d4932d8ba8f8cc7856fb6a7d38838400e546fec9c92239f8`；
helper SHA256：
`02ab2cd87638c9cdf05b671a35273c2916e3da3367bf369f5f019406a9f667dd`。
完整來源／link inputs／保護檔案 receipt 在 reference 工作目錄的
`renderdoc-fixed/build-receipt.json`。原始及共享 libraries 未被修改。

原始碼回歸測試：

```sh
python3 -B -m unittest tests/test_renderdoc_patch_parameter_alias.py -v
```

6 項通過，包括 core／EXT／OES 同 serializer、point API 保持不變、
以及錯誤／遺漏／重複 alias 的負向測試；這是原始碼測試，不代替 replay。

### D16 array slice 直接讀回與第一個實際差異

實際 depth attachment 是 4-layer `Texture2DArray` 的 mip 0／layer 0，
不是單一 2D texture。新版私人 helper 同時驗證 live FBO 的 texture object、
mip、layer、非 layered attachment、internal format 與層數，再從當前已完成
的 replay FBO 直接執行 `glReadPixels(GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT)`。
它保存並還原 read FBO、正確 texture target 的 binding、PBO 與 PACK state；
不插入額外 shader draw，也不以 llvmpipe 資料替代原生輸出。

- Helper：reference 工作目錄下的 `helper-depth-array/bounded-player`。
- Helper SHA256：`895ab76d8b4ec9b6a2171226e30556832daaeac951235482482a7a4080c67944`。
- Header SHA256：`4e1ba121d6cc171f5e4dbf424faac87bd8a789a65d10d905694f1feba090a42e`。
- Native：debug 工作目錄下的 `native-alias-array-draw0`。
- llvmpipe：reference 工作目錄下的 `array-prefix0` 與 `array-prefix4`。
- 每張 raw 為 1,843,200 bytes，960×960、D16 little-endian、bottom-left origin、
  row stride 1920；每次只讀所選一層。

Native Draw 0 與 llvmpipe Draw 0–4 的 bounded evidence／readback 稽核皆通過，
GL error 欄位全為 0。llvmpipe Draw 0 單獨執行與 prefix 4 中的 Draw 0
raw bytes 完全相同。這證明診斷讀回有效，**不代表 native 畫面相符**。
新版 helper 的 56 個 mock 測試在 ASan／UBSan、warnings-as-errors 下通過，
記錄於 reference 工作目錄的 `helper-array-tests.0g8r2V`；mock 也不代替實際 replay。

Draw 0 的逐 D16 比較（`draw0-depth-comparison.json`）：

| 項目 | 結果 |
| --- | ---: |
| 全部像素 | 921600 |
| 完全相同 | 921573 |
| 不同 | 27 |
| Native − llvmpipe = −1 | 5 |
| Native − llvmpipe = +1 | 21 |
| Native 為 clear、llvmpipe 非 clear | 1 |

較大的差異位於 bottom-left 座標 `(19,331)`：native 為 `65535`，
llvmpipe 為 `48600`。其餘 26 個差異為 ±1 個 D16 單位；此時尚未證明
是 shader、clip／viewport、coverage 或 depth quantization 的哪一層造成，
不能直接稱為正常容差，也不能把第一筆 draw 標示為像素一致。

Native raw SHA256：`a55a2438648fc21809ad1040929d0e978414d95dce89301ed13d84b6df74417f`。
llvmpipe raw SHA256：`75abc92384ce28db4458ff4d244cfcd64ac28befb50a9f117c4ae798035c3c6c`。
debug 工作目錄的 `compare_depth.rb` 驗證 capture、backend、draw／event、
readback 契約和 raw 長度後才計算差異。clear／non-clear mask 只用於定位，
不可當成一般性的 coverage 計數器。

同一工具組重跑第一筆 ordered native prefix：

```sh
bash /Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/carchase-debug.t3s25m/run_native_prefix.sh 0
```

此 launcher 使用原本 frozen O3 driver／Contains bridge、修正後 RenderDoc、
新版 array helper 和 GLES 3.2，每次建立新的輸出目錄。它使用顯式
`--compile-only-init`；不能將這份 bounded 診斷視為完整原生 initial-copy 認證。

### 繼續順序

1. 固定已驗證的 array readback 和修正後 RenderDoc，不再使用錯誤 patch-size
   重播或容量實驗 runtime。
2. 從 `(19,331)` 附近的實際 post-tessellation／clip／window coordinates
   與 coverage edge equations 追最早差異；另外追查 26 個 D16 ±1 差異。
3. 診斷 trace 必須保持原生 raw 輸出不變；若修改執行語意，新增一般化回歸測試，
   再回到同一完整 ordered prefix 比較。
4. 成功返回、原生完成、附件相符分開報告。
   只在第一個差異定位後擴大 ordered prefix，保留 clear、compute、barrier 等事件。

## 第一個像素差異的幾何追蹤

私人 trace bridge 在
`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/carchase-parameter-trace.QGtYPk`。
builder 先以原 link inputs 重連，確認結果與原 Contains bridge **逐 byte 相同**，
再只替換私人 overlay 的 ParameterBuffer object。修改僅增加
`PVRGPU_PARAMETER_DEBUG_ALL=1` gate 與 depth-plane 位元輸出；保留原 O3、
diagnostics-off、API 30、4096 patch 容量及其餘原生執行邏輯。

- Trace bridge SHA256：`a35418db1911e0aea646d81685fe4fe60341651cb27a7f1403197c358a653403`。
- Trace run：debug 工作目錄的 `native-parameter-trace-draw0`。
- Trace run exit 0、bounded evidence 稽核通過，raw D16 SHA256 與原本 native
  `a55a2438…` 完全相同；診斷沒有改變這筆輸出。
- `inspect_parameter_pixel.rb` 從實際 window float bits 重建 8-bit subpixel
  edge values；它只做離線分析，產物不送回 simulator。
- 原有 trace 區塊位於 `!rasterizable` early-continue 之後，因此不涵蓋被
  剔除或 zero-area 的候選；不能把沒有日誌視為沒有產生 primitive。

在 `(19.5,331.5)`，最接近的原生三角形為 parameter index 162、
API patch primitive 1884、clip piece 0。三條 fixed edge 值為
`529770, 736150, -448`，因此不涵蓋這個 sample；若只外插該 plane，
D16 值為 `48600`，與 llvmpipe 在該像素的實際值相同。
這個外插值是定位資料，不是實際 native depth write。

llvmpipe 的私人 post-TES probe 位於 reference 工作目錄的
`posttes-probe.dGMpZj`。診斷前的 live FBO D16 與原 reference 逐 byte 相同，
接著 `GetPostVSData(..., GSOut)` 以 transform feedback **重新執行** shader，
取得 12684 個展開頂點／4228 個三角形、stride 36、456624 raw bytes。
額外工作有獨立 diagnostic counter marker，不得混入原 draw counters；
它不是被動擷取原本 shader 輸出，也不能用作 native 輸入。
`mesh.json` 中兩個 uint8 欄位的文字序列化瑕疵保留原檔，另有明確標示的
`mesh-normalized.json`；實際 raw buffer／positions TSV 未改動。

目標對應 TF triangle 3768（patch 1884 的第一個 triangle）。比較原始 clip
position 位元，一個 corner 完全相同，另外兩個 corner 部分分量差 1–4 ULP。
這是 TF 診斷下量到的幾何差，不足以排除 TF relink 對 shader codegen 的影響。

另已從 frozen llvmpipe dylib 的實際反組譯確認：TES 的 CPU post-VS viewport
使用分開的 reciprocal／multiply／multiply／add，native 原始 TES vertex
則使用 viewport FMA。這是確定的運算順序差，但把同一組 native clip bits
改用分開乘加，目標 edge 仍為 `-448`，**不能單獨歸因或宣稱已修好這個像素**。
完整四組交叉計算在 probe 目錄的 `viewport-four-combinations.json`：
native clip bits 使用兩種 viewport 都不涵蓋 sample；LP TF clip bits 使用
兩種 viewport 都涵蓋，最後一條 edge 為 `838`。因此這一點不是 edge tie，
也不能靠改 inclusive fill rule 修正。

下一個幾何檢查點是區分原本 llvmpipe TES 與 TF 診斷的輸出，再向前比對
VS／TCS／TES 的實際輸入及 PCO 運算。此階段沒有調整 coverage tolerance，
沒有修改 shader 或模型執行語意。

## D16 小差異的離線歸因檢查

`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/carchase-depth-conversion-analysis.ZHasYe`
保存 script、26 份逐像素分析和 `report.json`。先用實際 native trace 的
coverage、depth planes、兩次 ordered binary32 FMA 與 D16 RNE，重建 26 個
共同非 clear 差異像素；26／26 都與 native raw 完全一致。

接著保持同一組 native geometry／coverage／depth float，只在離線計算中
改用 llvmpipe 的 D16 magic-float encoding。16／26 個值可重現 reference，
其餘 10 個仍不同。這只表示轉換差在數值上足以解釋 16 點，**不是已證明
這 16 點的 llvmpipe depth float 與 native 相同**；也未檢查原本相同像素
是否會被該替代轉換改壞。沒有產生替代影像或修改 simulator 的 encoding。

剩餘 10 點（bottom-left）：`(105,254)`、`(135,271)`、`(67,274)`、
`(102,277)`、`(101,282)`、`(115,288)`、`(112,290)`、`(76,303)`、
`(71,306)`、`(25,326)`。另有 `(19,331)` 的 coverage 差異獨立追蹤，
不能以 depth encoding 解釋不存在的 sample。

## Ordered prefix Draw 0–4

定位第一筆差異的 primitive 後，保留其未解決記錄，使用原本非 trace frozen
runtime 執行完整前綴到 event 51。Native 輸出在 debug 工作目錄的
`prefix.kVQ1i9/replay`，對照為 reference 的 `array-prefix4`。
五筆皆有一筆 graphics submission／report、成功同步及 verified D16 readback。
比較結果在 debug 工作目錄的 `prefix4-depth-comparison.json`。

| 零起算 Draw | Event | 不同像素數 | 最大 D16 差 | Clear／non-clear mask 不同 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 32 | 27 | 16935 | 1 |
| 1 | 37 | 36 | 1 | 0 |
| 2 | 42 | 40 | 1 | 0 |
| 3 | 47 | 40 | 1 | 0 |
| 4 | 51 | 40 | 1 | 0 |

上述比較的是每筆 draw 後累積附件，不能把每列當成該筆新增錯誤數。
Draw 0 的 raw hash 跨 standalone、trace、prefix 4 三種執行保持一致。
後續覆蓋不代表第一筆差異被修好；目前只有深度 pass 前綴，沒有完整 Car Chase
frame 的最終 color PNG 或 PASS 證據。

完整流程遵循 [`MANHATTAN_DEBUG_WORKFLOW.md`](MANHATTAN_DEBUG_WORKFLOW.md)。
目前記錄不得解讀為完整 Car Chase frame 已完成或通過。

## 第二個阻擋：Draw 28 的 gl_VertexID

Draw 28／event 193 是 6 個 u16 indices、3 個 instances 的 draw。實際 VS NIR
同時讀取 `load_vertex_id` 與 `load_instance_id`；VertexID 0／1／2 的分支
選擇 billboard corner，並非索引越界。舊 driver 在 index rebasing 的
system-value guard 拒絕，compiler 也沒有允許該 intrinsic。

修正位於 `pvrgpu_context.c` 和 `pvrgpu_pco.c`：

- 真正的 vertex attributes 後，按 InstanceID、VertexID 順序保留各一個
  vec4-aligned VTXIN slot，以 R32_UINT 傳遞一個原始整數 DWORD。
- VertexID 使用原 API 有效 index（raw index 加 signed baseVertex），
  非 indexed draw 使用 first 加 vertex offset；不使用 compact rebased index
  或 instance-expanded stream offset。InstanceID 不包含 baseInstance。
- 一般 VS／FS 與 extended geometry pipelines 共用 slot allocator；兩處均以
  **原始 NIR** 的 system-values mask 配置，避免 DCE 移除 InstanceID 後，
  compiler 的 VertexID slot 與 driver packing 錯位。
- 保留 VertexIDZeroBase／BaseVertex 的既有拒絕與容量檢查；沒有新增 ISS
  特殊指令、capture 名稱分支或 reference input。

私人 candidate v2：debug 工作目錄的 `driver-vertex-id-v2/install`，
Gallium dylib SHA256：
`19327023c7df28ddfc89e830c2230aba1bb16523a3f0454e666f44cc542156f2`。
Build 僅寫入新的私人目錄，未覆蓋 shared Mesa prefix。模型仍使用原 Contains
bridge，API 30／4096 patches；driver 是 O2-g 診斷 build，不能作 O3 速度結論。

### VertexID 一般化驗證

`tests/test_pvrgpu_vertex_system_values.py` 的 3 項測試通過，包含 306 個
mask／DCE subset／slot-capacity 案例，並驗證兩個實際 compiler call sites。
`tests/vertex_id_live_probe.c` 另外執行真正 EGL/GLES draw：

- VertexID-only 與 VertexID＋InstanceID 兩種 shader。
- Direct／indirect arrays（first=4），u8／u16／u32 indexed draws。
- 重複、非零最小 indices；baseVertex 0／+2／−3，1／3 instances。
- 以 API 輸入計算 oracle，驗證 TF 中的 IDs、attributes、position、query
  以及 prefix／range／tail guards，未使用準備好的 llvmpipe 結果。

llvmpipe 與 candidate v2 各 56 cases、14,336 DWORD、16,321 checks 通過，
QUERY／WORD transcript 完全一致。Native 有 56 次實際提交與完成，零 driver
errors；舊 native 作負向控制，確實因 `unsupported NIR intrinsic load_vertex_id`
失敗。Native stdout 多一段精確確認的 SystemC shutdown 訊息，未改寫 raw logs。

完整記錄：
`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/vertex-id-live.4lP3Nu/verified-result.json`。

### Draw 0–38 的前綴證據

- 修正前成功控制組：debug 工作目錄的 `prefix.TCn3u1/replay`（0–27）。
- VertexID v1 成功 native：`native-vertex-id-v1-prefix38`（0–38）；
  v1 與 v2 的差別是原始 NIR mask 的 ABI 穩定性修正。
- 獨立 llvmpipe：reference 工作目錄的
  `prefix38-independent.aji1wT/replay`。
- 比較報告：debug 工作目錄的 `vertex-id-v1-prefix38-comparison.json`。

前 28 張 native raw 與舊控制組逐 byte 相同。39 張都是 D16 960×960，
但 Draw 20／29／38 開始不同 array layer，不能把 clear 後的零差異解讀為
前面差異已修復。代表性累積附件差異：

| Draw | Event | Array slice | 不同像素 | 最大 D16 差 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 32 | 0 | 27 | 16935 |
| 19 | 129 | 0 | 190 | 18665 |
| 20 | 139 | 1 | 0 | 0 |
| 27 | 183 | 1 | 924 | 22461 |
| 28 | 193 | 1 | 911 | 22461 |
| 29 | 203 | 2 | 0 | 0 |
| 37 | 251 | 2 | 468 | 1 |
| 38 | 261 | 3 | 0 | 0 |

## 被動 TES 證據：排除 TF 重跑的擾動

Reference 工作目錄的 `passive-tes.KAI3Um/REPORT.md` 記錄私人 llvmpipe
logging-only hook：在原本 `llvm_tes_run` 後、viewport／clipping 前，記錄
patch 1884 的 domain coordinates、indices、原始 clip-position bits。
未修改 NIR／JIT shader，未重新執行 TF；logging 保存浮點環境及 errno。

Control relink 經相同 install-name／rpath 處理後重現 frozen llvmpipe dylib。
私人 trace run 的原本 Draw 0 D16 與全部 counter rows 仍逐 byte 相同，
零 GL errors。Trace 的 4 個 domain vertices、展開後 6 個 vertices 與先前 TF
診斷完全一致。因此之前 1–4 ULP 的 clip-position 差是實際原生／llvmpipe
差異，不是 TF 診斷引入；仍不能單憑此斷定是 VS、TCS 或 TES 的哪個運算。

`(19.5,331.5)` 的 native edge 為 `529770,736150,-448`，原本 llvmpipe
TES bits 對應 edge 為 `529770,734990,838`。兩組分別套用 fused 或 separate
viewport arithmetic，涵蓋與否不變。下一步要往前比對實際 shader inputs／outputs，
而不是改 fill rule、放寬 tolerance 或用 depth encoding 掩蓋 coverage 差。

## Draw 39 的奇異平面阻擋

`native-vertex-id-v2-prefix74` 使用 candidate v2，在 Draw 39／event 265 的
glFinish 發生 ParameterBuffer 非有限 llvmpipe depth-plane 例外並 abort。
座標約為 `(478.912903,499.939392)`、`(478.919250,499.949097)`、
`(478.906555,499.929688)`，depth 並非常數；binary32 area 為零，但量化
fixed-point area 非零。此 run 整體 native audit 失敗，不能截取其前綴宣稱 PASS。

Reference `array-prefix74` 已成功完成 75 個 audited ranges：Draw 0–73 為
D16 shadow-array layers，Draw 74 為 D24 960×540。格式比較器
`compare_depth_formats.rb` 嚴格區分 D16 與 D24 high-24 depth units／raw low8
encoding，只接受兩邊都成功完成且稽核通過的 runs；34 項正負向測試通過，
也完整重現既有 39 張 D16 的 663 個舊比較欄位。

修正只在 plane 非有限時，使用實際 fixed edges、fill convention、
framebuffer／scissor、sample count／mask／positions，證明沒有任何啟用 sample
會被涵蓋，才允許不產生 fragments 的 identity placeholder。可能涵蓋 sample
或存在其他無效 payload 時仍須報錯；不得略過整筆 draw 或捏造可見深度平面。

### 奇異平面修正與實際驗證

`parameter_buffer.cpp` 以專用 `NonFiniteDriverPlane` 例外區分數值失敗與
其他 payload／overflow／state 錯誤。證明無 sample 後保留原 primitive key、
front／face／line／window Z 與 coefficient offset，清空 depth-plane metadata、
coefficients count 和半開 bbox；回滾該三角形已部分寫入的 coefficient vector。
Tiler 已建立的 references 不刪除、不重新編號；ISP 不會進入空 bbox 的迴圈。
原本可建立有限 plane 的情況完全不進入此分支，包括既有常數 attribute 的
零 float-area 測試。沒有修改 ISP、PDS、sampling table 或浮點容差。

私人 O3／diagnostics-off bridge：
`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/carchase-empty-sample.SbvyZx/lib/libpvrgpu_systemc_empty_sample.dylib`
SHA256：`f5f9a0a6ab0abe75732989f4a05e045d1c0d970da0214e6e334dc4d40f2b1961`。
Builder 的 control relink 與原 Contains bridge 逐 byte 相同，只替換 production
ParameterBuffer object。API 30、4096 patches、其餘模型 source／link inputs
及 public headers 保持原 identity。CMake 只新增 24 個 test registrations，
未重新產生 frozen build metadata。

`tests/parameter_buffer_perspective_test.cpp` 新增 24 個獨立 process 案例：
實際形狀／合成 depth、獨立的有 coverage 奇異三角形負控制、全部 sample counts、
MSAA disabled／mask／scissor／framebuffer／fill-rule tie、錯誤 metadata，
以及 active–inactive–active 的 partial-coefficient rollback。
連同原本兩個 perspective tests，普通建置 **26／26**、ASan＋UBSan **26／26**
通過；SystemC 是未 instrument 的既有 shared library，未宣稱全依賴 sanitizer。
完整命令在 `tmp/parameter-empty-plane.oMWkq8/RECEIPT.md`（Working/PvrGPU 下）。

另外 25 個既有鄰近 CTests 中 **24／25** 通過，涵蓋實際 PB→ISP→PDS、
MSAA depth、front-face／clip／viewport、native depth LOAD/readback 與
bypass／cache UBO DRAM 路徑。失敗的 `ordered-blend-pipeline-unit` 未連入
ParameterBuffer：HEAD fixture 有 fragment code，卻未初始化 fragment output
mask，PBE 因 `mask=15 expected=0` 拒絕；未順便改寫此無關 fixture。

成功 replay：debug 工作目錄的 `native-empty-sample-prefix39` 和
`native-empty-sample-prefix74`，兩者 exit 0／完整 evidence audit 通過。
比較報告分別為 `empty-sample-prefix39-comparison.json`、
`empty-sample-prefix74-comparison.json`。Draw 39 的 D16 raw SHA256
（native 與 llvmpipe 相同）：
`a0d34b1a2ae39c6ff23a714f0133bc09c39720fffa99252da9c657e77ed0cc1d`。

| Draw | Depth attachment | 比較結果 |
| --- | --- | --- |
| 39–53 | D16，layer 3，960×960 | 每張 raw 完全相同 |
| 54–73 | D16，layer 3，960×960 | 每張最大差 1 D16 單位；Draw 73 有 158 像素不同 |
| 74 | D24，layer 0，960×540 | 106480 像素不同，最大差 7 D24 單位 |

這仍是每筆 draw 後的累積附件，並非 per-primitive error counter。
深度相近不等於完整畫面已正確；先前 layer 0–2 的差異仍未解決。

## Draw 75：沒有 attributes 的 procedural VS

`native-empty-sample-prefix83` 和固定 NIR dump 的 `native-nir75`，均在
Draw 75／event 463 因 `vertex_elements=0 vertex_buffers=0` 的舊 layout gate
失敗。它們是失敗 runs，不取其前 75 筆當成另一份通過證據。

真正綁定的 VS 對應 `nir75.VRvtJI/0666-vertex.nir`：沒有 shader inputs，
`load_vertex_id` 索引四個常數 vec4，輸出全螢幕 strip 的 position。
這不是缺失 vertex resource；GLES 允許 shader 自行產生頂點。

Candidate v3 讓空 real-attribute layout 通過 state gate，再依實際 NIR
`inputs_read == 0` 驗證；非空 layout 仍要求 buffers。系統值仍以實際
R32_UINT input transport 供給，不捏造 attribute 或跳過 VS。普通 pipeline
若連 system-value slots 也沒有，原有零 packed-words 限制仍會拒絕。
`tests/vertex_id_live_probe.c` 擴充為 112 cases，加入零 real attributes
的 VertexID-only／VertexID＋InstanceID procedural shader；oracle 仍由 API
inputs 計算。此節的候選 replay／live-test 結果完成後另列。

### 零 attribute live-test 與候選 v3

`driver-zero-attributes-v3/install` 的 Gallium SHA256：
`c0b44fe928fca63d9d0bf337ac1f0da8bb94bac9724340be9ac2fa0ab9457911`。
擴充 probe 在 llvmpipe／v3 各 **112 cases**、28,672 DWORD、32,637 checks
通過，所有 QUERY／WORD transcript 完全一致，原生 112 次提交／完成、零 errors
及 leaks。v2 負向控制先完成前 56 個舊案例，再於第 57 個 procedural draw
以 `no_vertex_layout, attributes=0` 明確拒絕。原始 logs 保留，僅允許精確的
SystemC 結束訊息差異。

證據：`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/vertex-id-zero-live.Kq1Dj3/verified-result.json`。
Probe source SHA256：`814488a26cba15c0f80172b9b1cf74ed8efd464b9efc17b5d9ebaf3206330f85`。

`native-v3-new-helper-prefix74` 以 v3 和新版 renderbuffer helper 完成成功的
0–74 prefix，75 張 raw 與 v2／舊 helper 的 `native-empty-sample-prefix74`
全部逐 byte 相同。比較報告為 `v3-new-helper-prefix74-comparison.json`。

v3 的 `native-zero-attributes-prefix83` 已越過 Draw 75 的 vertex gate，
但在 FS compile 以 `FS writes unsupported outputs 0xff1` 拒絕；仍是失敗 run。
真正綁定的 FS 為 `nir75.VRvtJI/0665-fragment.nir`，含 RenderDoc 的
`DiscardUBOData`、flags、discard、gl_FragDepth 及 DATA0–7。當時 framebuffer
只有 depth，因此高位 color outputs 未綁定。此處不能略過 RenderDoc utility
draw；需在 compiler clone 中一般化處理不會輸出到 framebuffer 的 stores，
並保留 output 內部讀回、depth／discard／memory side effects。

## 深度 renderbuffer 的讀回工具

Draw 84／event 588 的實際 depth object type 為 GL_RENDERBUFFER（36161），
不是 texture。舊 helper 在其上查詢 texture mip／layer，導致 GL_INVALID_ENUM；
reference `prefix84-independent.JkXuIF/replay` 保留為失敗 run。

新 private helper `helper-depth-renderbuffer/bounded-player` 先分辨 live object
type。Renderbuffer 路徑要求 metadata mip0／slice0／單層／單樣本，並核對 live
WIDTH／HEIGHT／INTERNAL_FORMAT／SAMPLES；保存並還原 renderbuffer binding，
不發出 texture-only queries，再以相同直接 glReadPixels 讀回 D16／D24。
原本 FBO／PBO／PACK／context 與四個 GL error 欄位的驗證不變。

Helper SHA256：`04e4c3e47cdcc2432da6d8dd4ebc4fcfbbcea2b5d0a03e4e9a8de92bf8594d89`；
header SHA256：`69c17c3ed6eb0b323cb150e7b494ddec07524be74ea80df1ced2943cf245b21f`。
舊 helpers、reference 和 shared libraries 均未覆寫。

Reference `prefix84-renderbuffer.Ng24mY/replay` 成功完成 85 個 ordered ranges、
86 個 verified attachments。前 84 張 depth raw 與所有舊 metadata 欄位不變。
Draw 84 的 D16 renderbuffer 為 960×960，color0 為 RGBA8 960×960，raw／PNG
均驗證成功；該 color PNG 視覺上全黑，不應當成完整 Car Chase 畫面。
原始 Gallium CSV 是 **94 rows／85 range markers**：Draw 75–83 各有兩個
Gallium rows。前 93 rows 與成功的 `prefix83-independent.kfekO9/replay` 相同，
全部 94 rows 與舊 failed84 run 的已返回執行 counters 相同。

`compare_depth_attachments.rb` 另存為新版 comparator，支援 texture 與
renderbuffer 的 per-run object identity 驗證；不比較不同 contexts 的 GL 名稱。
Renderbuffer 必須來自上述 frozen helper 並具備完整新 metadata。89 項正負向
測試通過；既有 39／75 draw 的比較欄位完全重現。它仍拒絕整體執行失敗或
未完成的 run，沒有把讀回失敗改當成容差。

## Draw 75：unbound outputs 與原生 gather

v4 `driver-unbound-outputs-v4/install` 的 Gallium SHA256 為
`d434da34cb4dd558c0dd2d1dd5a1f1abfc86fb77a5a45856c9a10ec794a581e2`。
新增 `pvrgpu_prune_unbound_fragment_outputs`，在兩條 compiler 路徑的 FS
clone 上，將完全落在未綁定 DATA 範圍的變數降為 shader temporary。保留 stores、
內部 loads、depth/discard/memory effects；拒絕跨綁定邊界的 arrays、dual-source、
pointer aliases 和混合 lowered IO，原始 application shader 不變。

- Real NIR：21 cases／169 checks 通過，包括重新綁定、讀回後供 depth 使用、
  side effects、resource counts 與負向控制。ASan／UBSan fixture、NIR core／
  validator 同樣通過；不是完整 Mesa sanitizer coverage。
- Live llvmpipe／v4：12 passes／2215 checks，MRT4→1→4→1、depth-only、
  discard 及 image atomic 均符合獨立輸入 oracle；native 12 submits/reports，
  零錯誤。沒有 GL/GLES version override。v3 在第一個 MRT1 draw 正確拒絕。
- 證據：`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/fragment-output-pruning.Lf72EX/RECEIPT.md`
  及 `tmp/fragment-output-public-live.tmff57/verified-result.json`（相同 WORK root）。

v4 replay `native-unbound-outputs-prefix84` 越過 utility shader compile gate，
仍在 Draw 75 的實際 shader 編譯以 `unsupported texture operation (op=tg4)`
失敗，整份 run 的 audit 為 false，不能宣稱 Draw 75 已完成。
`nir75.VRvtJI/0764-fragment.nir` 使用 sampler2D Z24 depth、component0 gather，
對四個值取 max 寫 gl_FragDepth，並有五個普通 ±1 texel offset samples。

Pinned PCO 已有真正 `tg4 → SMP.RAWDATA → [2,3,1,0] swizzle` lowering。
正實作的 bounded path 為 fragment、2D、non-array/non-shadow、float32、
component0、無 gather offsets、單層 Z24 clamp-edge。component1–3 原生回傳
8/12/16 words，超過目前四-word ABI，保持拒絕。
Sampler descriptor 使用既有 SH+16 gather block，不是普通 sample 的 SH+8。
四個 raw taps 都經既有 TPU/cache/memory 模型，不讀取 reference 圖。

Gather 的 base-level、忽略 min/mag filter 及 GL tap order 依
[GLSL ES 3.20 §8.9.4](https://registry.khronos.org/OpenGL/specs/es/3.2/GLSL_ES_Specification_3.20.pdf)。
NIR `tg4_offsets` 的預設值是四個 neighbor positions，不是四組零 offsets；
driver gate 使用 Mesa 的 `nir_tex_instr_has_explicit_tg4_offsets` 判斷。
新的 real-NIR admission test 有 86 checks，普通和 ASan／UBSan fixture 均通過
（已建置的 NIR archives 未 instrument）。v5、v5b 保留為未成功候選，不能用來重播。

## 完整 llvmpipe frame reference

`fullframe-reference-end6649.X7CSES/replay` 完成全部 395 個 ordered draw ranges
及真正尾段 events 6630–6649，exit 0／audit 通過。最後 draw 本身仍是
Draw 394／event 6629；final snapshot 的 event 6649 表示讀回邊界。
原 attachment 為 B5G6R5_UNORM，直接 RGBA8 readback，1920×1080。
PNG 為 `draw-394-event-6649-color0.png`，SHA256：
`c4cc9afe73ecb6e4958c9c827f240b7202eb0b5ba46636416610ea049ff1d77a`。
Raw SHA256：`2d50e3efb5f9e0ea0a96cf57afe337e12b0eb011a71b4df6224315631aaf0eac`。

442 個 Gallium rows／387 markers 不等於 395 actions：8 個 zero-instance
draw 沒有 Gallium row，RenderDoc utility work 也會增加 rows。
final raw／PNG／CSV／Report.md 與停在最後 draw 的獨立 run byte-identical；
這不是一般性的 post-tail GL state 比較。此圖僅作 llvmpipe reference，未餵給
native，也不是 PvrGPU 完整 frame 的結果或效能測量。

## Gather 接通後的整合驗證

### 真正 PCO 指令與資料路徑

`tools/pco-fixtures/generate_texture_gather.c` 使用 pinned PCO 產生真正的
component0／component1 gather 及普通 textureLod control，沒有手工拼指令。
Component0 binary 為 80 bytes，SHA256：
`9f2dffd537025589e20b844084de75b9e0de57b0671474e1346df5bdeff4d6af`；
與 live GLES probe 經 driver compiler 產生的 binary 完全一致。
SMP backend bytes `F4 52 90` 為 CHAN1／RAWDATA／REPLACE／PPLOD；
四個 response words 經既有 WDF continuation，shader 自行執行 tap-order swizzle。

ISS／USC 以明確 gather flag 保留這個語意，同步涵蓋 prepared-program hash、
續跑 request equality、same-PC batch 與 fragment-only 驗證。普通 sample 仍用
SH+8，gather 用 SH+16；component1–3 等不符合四-word ABI 的模式維持拒絕。
117 項真 binary、mutation、WDF once-only tests 通過，ASan／UBSan 亦通過。
既有 multisample/offset、prepared fragment、texture sequence regression 分別
2112／1302／14987 checks 通過。

### Z24X8 的既有 canonical transport

第一個 live Z24 probe 在 model-v2 被格式 gate 拒絕。不是 capture 資料缺失：
`pvrgpu_capture_generic_sequence_texture` 已有的 `float_depth_view` 路徑會將
Z24X8、Z16 等非 D24S8 depth view，套用 view swizzle 後傳成 RGBA32Float。
因此 TPU gather 支援兩種明確宣告的格式：Z24/S8 red 及 RGBA32Float red。
後者每 tap 讀完整 16 bytes，以既有 decoder 取 red；不需要依 producer 或
capture 名稱判斷，也沒有把 reference depth 放入記憶體。

43 個 TPU 正負向案例涵蓋兩種格式、1×1／奇數尺寸、filters、LOD clamp、
重複 taps、16-byte transfers、cache/direct/bypass；ordinary texture-unit 也通過。
舊 exact-floor 版本的 43 項 sanitizer 結果及 source/binaries 另行保留。
Geometry/compute 的普通 texture pipelines 5 項通過；2187 項真正 native GS/CS
instruction mutations 確認非 fragment gather 在執行前即拒絕。

### Depth-only color-mask 一致性

model-v2 的 capture Draw75 下一個錯誤為 `decoded=0 expected=15`。
實際 FS 只有 gl_FragDepth，但一般 compiler 路徑曾無條件宣告每個 transport
target 有 color outputs。v6 改為依 `outputs_written` 保留真正的空 color mask，
與既有 GS compiler 路徑一致。Decoder／TPU bypass／PBE 共用
`HasExplicitFragmentOutputMasks`；有 native code 時 mask 0 是明確的零輸出，
不是 legacy vec4 預設。PBE 原有 depth feedback 和 color LOAD 行為不變。

v6 Gallium SHA256：`3ebb9d9c91a9c6ca241e9929ce8389739d93b194d2bbd0107efef689ea6b74e1`。
實際 compiler matrix 65 項 metadata、16 項 ISS checks 通過；8 個 PCO binary
與 v5c 完全相同，只有錯誤的 output metadata 改正。完全空、無 side effect
的 ordinary FS 仍為既有拒絕案例，沒有順便擴大支援。

`fragment_depth_only_live_probe.c` 的 llvmpipe／native 各 8 draws／460 checks
通過：128 個 D32F 值一致、128 個已初始化 color pixels 保留，原生 8 次實際
提交及 model reports 完整，零 errors／leaks。v5c 搭同一 model 在第一 draw
仍因錯誤 mask 拒絕，作為負向控制。共享 mask helper 另有 512 checks。
證據：`tmp/fragment-depth-only-live.bDQeAl/verification.json` 與
`tmp/carchase-output-contract.QIPsf6/validated/receipt.json`（WORK root）。

### 第一份成功的 native Draw75

`native-gather-v3-prefix75` 使用 v6 driver／model-gather-v3，完整完成 76 個
ordered ranges、exit 0／evidence audit 通過。前 75 張 depth raw 與
`native-v3-new-helper-prefix74` 全部 byte-identical。
Draw75 為 480×270、D24 mip1，129600 pixels 中 39077 個不同，最大差 6 個
D24 單位，沒有 native-only/reference-only nonmax pixels。這不是 pixel PASS；
前一層的深度差異仍可能傳遞，尚未把每個差異歸因。
比較報告：`gather-v3-prefix75-comparison.json`。

Native Draw75 SHA256：`135fa35daefde624c9266fe039f442cfd4f4c6090961fda256904575961484f4`。
llvmpipe：`9e889099fa8c314b8805808415c8c35151fbbeac736e799b70e747fdeb868eb0`。

### 額外的 half-texel finite-precision 回歸

一般 live probe 336 draws／16561 checks 在 v6/model-v3 全部相符，且有
336 組 submit／hello／counter／done，不以 driver 的 done 訊息單獨當成功。
另加的 576-draw boundary probe 在 llvmpipe 通過，v3 在第 413 draw 如期
暴露差異：4×4、s bits `0x3dffffff`（.125 前一個 float），LP taps 為 0,1，
ideal double floor 的 v3 為 0,0。8 cases／64 words 受影響，舊失敗輸出保留。

Pinned llvmpipe 的 clamp-edge gather 分別 round binary32 `s*extent` 與
`+/-0.5` 後取整。新版明確採相同 binary32 定址，先處理 clamp-edge outliers
以防 huge finite 值溢位，再獨立計算兩個 tap，沒有改普通 texture filtering。
此版本的 frozen live/capture replay 結果如下。

### Binary32 model-v4 的完成證據

`model-gather-v4/build/lib/libpvrgpu_systemc_bridge.dylib` SHA256：
`8a7ac71867b6f692d5f004796c2f004d60636ea294f5c54e4afe6254b2e2f7e1`。
這是全量重新建置的 O3／NDEBUG、diagnostics OFF 模型，沒有沿用不同內部
request type 的舊 objects。公開 API 仍為 30，primitive capacity 仍為 4096。

搭配 frozen v6 driver 的獨立 live 驗證：

| 測試 | 實際 native submits／完整 model reports | 結果 |
| --- | ---: | --- |
| 原始 gather | 336 | 16561 checks、5376 output words 與 llvmpipe 一致 |
| 額外 half-texel／huge finite 邊界 | 576 | 28322 checks、9216 output words 一致 |
| VertexID／零 attributes 回歸 | 112 | 32637 checks、28672 output words 一致 |
| Unbound fragment outputs 回歸 | 12 | 2215 checks、一致 |

共 1036 次 native submissions，逐項核對輸入 oracle、完整 LP transcript、
submit／hello／counter／done，零 GL／driver errors 與 pool leaks。所有新 runs
均未使用 capability/version override；舊 VertexID 的 LP reference 曾用 GLES3.1
override，此限制保留記錄。Z24 live input 經既有 canonical RGBA32F transport；
raw D24S8 的覆蓋由 TPU unit tests 提供，不能混為同一種測試。

驗證報告位於 WORK 下：

- `tmp/texture-gather-live-v3.eJuL1F/verified-result-candidate-v6-model-v4.json`
- `tmp/texture-gather-boundary.AFrJ75/verified-candidate-v6-model-v4.json`
- `tmp/gather-model-control-regression.quprqy/verified-result.json`
- `tmp/texture-gather.jTGD42/V4_RECEIPT.md`：44／44 普通及 ASan／UBSan
  TPU tests，並保存舊 exact-floor 對新 boundary assertion 的負向控制。

另外在 frozen model-v4 source/build 執行 gather／ordinary texture／ISS gather／
output-mask 的 registered CTest，46／46 通過。不是全專案 CTest PASS。

### Draw 76 與下一個 continuation 阻擋點

`native-gather-v4-prefix76` 完整完成 77 個 ordered ranges、exit 0／audit true。
Draw 0–75 的 76 份 depth raw 與成功的 `native-gather-v3-prefix75` byte-identical。
新 Draw 76／event 470 是 mip2 D24、240×135；32400 pixels 中 15101 個深度值
不同，最大差 4 個 D24 單位，沒有 native-only/reference-only nonmax pixels。
這是有界前綴完成證據，不是 pixel PASS 或 full frame。

Draw76 native SHA256：`efa0d6f9ac902f9318d4ef586977a7b31537a0065c45d1bf6dd2e0e75bf57ce5`；
LP SHA256：`1b217eca7acc76063f6944076ee083eff4c35b490f8c204164d757da71b79b97`。
完整比較：`gather-v4-prefix76-comparison.json`。LP reference 仍為成功的
`prefix84-renderbuffer.Ng24mY/replay`，Draw 77–84 明列為尚未比較的尾段。

較長的 `native-gather-v4-prefix84` 在 Draw 77／event 477 終止：
`texture fragment continuation TEMP state is inconsistent at PC 147 mask=0 pixels=0/0`。
此處 `mask=0` 是驗證 Boolean，不是「沒有 TEMP registers」。整份 run 的 audit
為 false，不能截取其成功片段代替另跑的 completed prefix76。
目前正在核對 forward branch 跳過的 TEMP writes 與 continuation 的 must-write
判定；需以真 PCO 指令、多分支正負向 tests 與實際 replay 證明修正。

需要完整 shader binary 時，可以在同一個 bounded launcher 前加
`PCO_DEBUG_PRINT=fs,binary,internal PCO_COLOR=off`，使用編譯器既有的只讀
print path。`internal` 不可省略：一般 driver compiler 將其 FS clone 標為
internal。這些 print flags 不改 `PCO_DEBUG` 的 compilation/validation policy。
不要使用 driver error 裡以省略號截斷的 `pco=` 欄位重建完整 binary。

`native-pco-internal-binary-prefix76` exit 0／audit true，77 張 depth raw、
逐 range 的完整 counters、drawlist_stats 及 shader fingerprints 全部與未加
print 的 v4 prefix76 相同。`extract_pco_binary.rb` 驗證每行 hex offset 連續後，
擷取 GLSL249 的 1696 bytes；兩次真正 compiler 輸出 byte-identical，SHA256：
`d8bdee6009f295e1c6bfc2d098fb65c87afbb1691c208197f72fcab249b27b9b`。
Binary 與來源行數／hash receipt 位於 debug 工作目錄 `actual-glsl249-pco/`。

另有成功的 LP `prefix101-renderbuffer`：102 個 ordered ranges、exit 0／audit
true，前 86 個 raw attachments 與 LP prefix84 完全相同。它是後續對照資料，
不是 native 已到 Draw 101 的證據。

### Control-flow aware TEMP continuation 修正

根因已用獨立的真 PCO if/else＋textureLod fixture 重現：336-byte shader 的
false branch 在 PC23 續跑時，被線性 prefix 檢查錯誤要求 skipped THEN block
的 r5–r8 也必須已寫入；true branch PC12 沒有這個問題。兩者真正保存的
TEMP mask 都是 `0x21f`，修正沒有補造缺少的暫存器值。

新增 `BuildFragmentTemporaryRequirements`，對 `(PC, execution-mask active)`
建立 reachable dataflow，在控制流程匯合處取 must-written intersection。
CND 同時分析可能的 active/inactive 後繼，native BR 條件選擇對應 edges，
有 backedge 時求固定點；DMA 結果直到對應 WDF 才加入已寫入集合。
Prepared program 在建立時快取 immutable proof，沒有逐 lane 重播 shader
或新增 GPU memory reads；raw vector API 在 resume 時重算。

只改 TEMP 的「必須已寫」判定。既有 may-write 上限、PIXOUT／P0／depth／
signature／response／counter 檢查保留，不宣稱所有 continuation fields 已
改成完整 CFG 驗證。現行 validator 拒絕 SMP／derivative 的非零 exec_cnd，
所以合法 suspension 必定來自 active lane。Derivative cache eligibility
從真正 opcodes 決定，不依賴未驗證的 summary hint。

`pco_branch_continuation_test.cpp`（CTest `pco-branch-continuation-unit`）以
真 compiler fixture 做 202 checks：兩個分支、非零 condition、raw/prepared
一致性、必要 TEMP 缺漏、非法額外 TEMP、WDF 前過早 response publication、
checkpoint／signature／depth／PIXOUT／counters mutations，以及 derivative
summary hint 清零時的兩條 API parity。已通過短回歸與獨立 code review；
完整模型 replay 結果另列。

另外建立兩個不可混淆的 live controls（WORK 下）：

- `tmp/odd-depth-reduction-live.SgHgj0`：generic 1672-byte variant，舊模型
  在 Y-odd case 的 PC145 重現錯誤。
- `tmp/odd-depth-reduction-capture.u09MlM`：保留 capture NIR 中額外 `+0.5`
  的 variant，真正編出的 FS 與 capture **1696 bytes 完全相同**，SHA 為
  上述 `d8bdee...`，舊模型精確重現 PC147。四組 8×8／8×9／9×8／9×9
  D32F input 減為 4×4，LP 510 checks 通過。這是相同 shader binary 的獨立
  inputs，不是相同 capture resource state；不使用 reference 圖作 native input。

早先將 capture NIR 的 `+0.5` 猜測為 Mesa 自動加入的說法未獲證明，已撤回；
以上以真正編譯出的 bytes 作區分。Monotonic depth input 本身不足以證明
conditional extra samples 沒被略過，還要比對由 branch/quad layout 獨立推算
的 texture instruction/request 與 texel-fetch 次數。

### Model-v5：實際 replay 越過 Draw 77

Frozen full model：`model-continuation-v5/build/lib/libpvrgpu_systemc_bridge.dylib`，
SHA256 `2c292c328643083031e59e07d3c6640e1c0f222aaa5f5f7337a88349244ef533`；
搭配相同 v6 driver。新 branch CTest 在這份 frozen source/build 也通過。
ISS 普通回歸合計 27575 checks（2112 multisample、14987 sequence、1302
prepared、117 gather、9057 native loops），另有新增 202 checks 及其
ASan／UBSan run、真 capture binary synthetic-lane 58 checks 通過。
詳見 WORK `tmp/carchase-continuation-cfg.1rBzmT/verification.json`。

兩個 odd-size live probes 在 v5 均完成四組 case，depth words 與 LP 相同；
事先推算的 texture requests／FS texture instructions `[24,36,36,47]`，
texel fetches `[96,108,108,119]` 全部精確符合。普通 gather／boundary／
VertexID／unbound outputs 再跑通過，加上這 8 次 odd-size draws，共 1044 次
實際 native submissions／完整 reports，零 GL／driver errors、零 pool leaks。
來源 `tests/fragment_texture_branch_live_probe.c` 與 frozen capture-identical
probe byte-identical；六組摘要位於 WORK
`tmp/continuation-model-control-regression.RY3l2F/six-suite-summary.json`。

`native-continuation-v5-prefix84` 成功完成 85 個 ordered ranges、exit 0／audit
true。獨立比較確認前 77 張 depth raw 與完整 per-range counter records
（只排除輸出檔名，含 virtual_time_ns／drawlist_stats）與 v4 prefix76 不變。

| Draw | Depth 不同 pixels | 最大 depth-unit 差 |
| --- | ---: | ---: |
| 77 | 3745 | 5 D24 |
| 78 | 917 | 6 D24 |
| 79 | 223 | 7 D24 |
| 80 | 51 | 7 D24 |
| 81 | 9 | 8 D24 |
| 82–83 | 0 | 0 |
| 84 | 0 | 0 |

Draw84 color raw 的 3686400 bytes 亦與 LP 相同，SHA256
`0c660f2bd3eff3150dd0040789abe2291613b9af319df870203d4f77a4913a5f`。
必須保留解讀限制：Draw82／83／84 的 depth 兩邊全為 maximum/clear，
Draw84 color 兩邊全為 RGBA(0,0,0,0)，不能以此宣稱可見場景已畫出。
比較報告 `continuation-v5-prefix84-independent-comparison.json`。

接著 `native-continuation-v5-prefix101` 成功完成 102 個 ordered ranges，
每 draw depth/color readback 均返回，exit 0／audit true。獨立比較確認前 86
個 attachment raws（0–84 depth 加 color84）及全部 85 份 counter records
與成功的 prefix84 不變。新 85–101 都已有非均勻、非零的 color output。

第一個非 clear 的 Draw85／event608，兩邊各 6234 個 nonmax depth pixels，
D16 有 7 pixels 不同、最大 1 unit；color 有 3648 pixels／6821 channel bytes
不同，最大 channel 差 21。Draw101 的 D16 有 1870 pixels 不同、最大 2 units，
兩邊各 672099 個 nonmax pixels；color 有 326215 pixels／816829 bytes 不同，
最大 channel 差 42。這些差異保留追查，不套 tolerance 或視為通過。
Root 已檢視兩邊 Draw101 PNG：均為有內容的圓形中間 render target，仍不是
完整 Car Chase 最終畫面。完整 report：
`continuation-v5-prefix101-independent-comparison.json`，SHA256
`d709e9b811390b1bb93aa4d14480aef9ec1b600b60873542f456b013c900730c`。

目前 frozen runtime 的重跑入口是 `run_native_prefix_v5.sh`，預設 Draw84，
可傳入 0–394 的 draw index。它每次建立新 output directory、從 capture
起點 ordered replay，**不是**先前討論的正規 GPU snapshot/replay 機制。
舊 `run_native_prefix.sh` 保留最初版本的歷史 runtime，不是目前延續入口。

### 完成 Draw150，與 Draw156 的新阻擋

`native-continuation-v5-prefix150` 和 LP `prefix150-renderbuffer` 各自完成
151 個 ordered ranges，最後 event1801，整體 audit true。獨立比較全部 218
attachment pairs（151 depth／67 color）；native 前102 draws 的120份 raw
及全部102份 per-range counter records 與成功的 prefix101 不變。

新49份 depth 的 maximum/nonmaximum mask 相同，最大差2 D16 units；這不等於
一般 primitive coverage PASS。Draw102/event898 color0 改為 mip0/slice1，
LP 全為 RGBA0，native 有753953個不同 pixels／3015812個不同 channel bytes，
最大差255；該 draw 的 depth 完全相同。Draw150 color 有737685個不同 pixels，
需分開追查附件／layer／clear 狀態，不把差異直接歸咎於取樣精度。
報告 WORK `tmp/prefix150-independent-comparison.gPHOVZ/review.md` 及
`comparison.json`（SHA256 `5a64d03eb6b7e7c2ffd27c63ae4fe8205bb296710284d99f2e16aecc8d3f3ad3`）。

`native-continuation-v5-prefix175` 則在 Draw156/event1958 被
`tessellation_texture` gate 拒絕，整體 audit false，不當成已完成 prefix175。
與既有 NIR creation ordinal 對上：GLSL195 TCS 有4筆真實 `txl` 用於 Hi-Z
culling，TES 有1筆 `tex`，alpha 用於 displacement。不是 dead metadata，
不能只拿掉 gate；需把各 stage descriptors、ISS／WDF、TPU／memory path 一起接通。

### TCS／TES texture 路徑整合

Compiler 支援有界的 normalized、nonshadow、single-sample 2D `tex`／`txl`，
每 stage 最多8組 descriptors，TCS/TES 的 descriptor prefix 分別從 SHARED8／4
開始，每組20 dwords，再接 UBO／CB0。Gather、array、shadow、bias／derivative
等尚未支援的模式保留拒絕。公共 API struct layout/version30不變，使用既有
TCS／TES stage enums，不假冒 GS 或 CPU 直接算 texture。

- 真 NIR compiler tests：477 checks；既有20個零 texture binaries 逐 byte 不變。
- ISS：既有191,387、新增20,026 checks 均通過 normal 與 ASan/UBSan；包含真
  compiler TCS/TES binaries、動態 LOD、barrier、pending SMP 僅在 WDF 發布，
  不活躍 lane 不發送、錯誤 response／unknown pending state 負向測試。
- Module／TPU 整合：真 TCS→fixed tessellator→TES 6 draws、4,573 checks，
  cache/direct/bypass 與 implicit/explicit TES LOD；focused CTests 37/37。
  這些不是 capture Draw156 通過，也不等於完整 GL driver 路徑已驗證。

Private driver v7b SHA256
`c4afc8ad0e576aa4775d923cc9380dfe0d3b51be46031bfb832058cc2c601d01`。
`native-driver-v7a-model-v5-tess-binary-prefix156` 整體失敗：實際156在
`mixed_render_targets` gate 即拒絕，未進 TCS/TES compiler，不能當成 binary
decode failure 或完成156。Metadata 顯示 FBO446 四個 color targets 依序是
RGBA8、RGB10_A2、RGB10_A2、RGBA8，1920×1080，D24。現有 public capsule 的
單一 format 不能表示它；不能移除 guard 後把10-bit附件默默降成8-bit。

另以 alias-fixed v6/model-v5、GLES3.1 成功跑到 Draw0，depth raw SHA256
`a55a2438648fc21809ad1040929d0e978414d95dce89301ed13d84b6df74417f`
與原3.2版本相同；這只是第一個 draw 的相容性檢查，仍待新版 observer 的
嚴格區間驗證，並不代表 GLES3.1 已通過完整 capture。

後續修正 driver command UBO prefix 及 shared tessellation payload 的兩個
consumer gates，完整 driver v7c／model v7 已通過真 GLES fixture：8 draws、
803檢查、745筆 typed records 與 llvmpipe 完全一致；24TCS＋24TES SMP，
48texel fetches，pools平衡。獨立 frozen model7 focused CTests38/38通過。
此測試包含動態LOD、獨立stage sampler、TF與RGBA32UI floatBitsToUint輸出，
並非以 reference pixels 代替 native shader。

- driver7c SHA256：`8dc476c1459cb47cb25b1b93ae4bad185a5088f17f4bc508a5f346889c61b02e`
- model7 SHA256：`f761b7dd55308059028be1394a9ca1bdd1557964c1723a857fcc74fae2a0422b`
- WORK `tmp/tessellation-texture-live.Ldu2rQ/LIVE_RECEIPT.md`
- WORK `tmp/model7-focused-verification.KVIeJ4/REPORT.md`

### API31 混合 MRT：實作與驗證中

新增每個attachment的明確格式vector；目前只允許RGBA8、RGB10_A2、BGR10_A2
四byte normalized組合，最多4targets。count0保留既有同格式路徑；正count必須
完整覆蓋targets、target0與common format相同、inactive pointers為NULL。
Model深拷貝格式與LOAD資料，alias比對完整格式vector，PBE逐target選codec，
readback驗證實際格式，depth要求color_format為NULL。沒有10bit→8bit替換。

GS／TES MRT只在已驗證explicit vector下開放，legacy guard保留。Recorder入口
也檢查target數與mixed格式，避免observable fallback繞過generic admission後
對formats[4]越界。不同API版本在讀取新增tail前拒絕。

已通過PBE混合200scenarios／81347checks、12個mixed／legacy負向模式與
既有13PBE CTests的ASan/UBSan檢查。Mesa搬運測試包含mixed LOAD11340、
framebuffer boundary27138、texture snapshot1163與command120checks。
真正GL mixed MRT／API31 integration仍在進行，尚未宣稱capture156通過。

更新：真GL mixed MRT已通過8 draws／3178 checks；521筆raw typed records
與llvmpipe逐byte一致（SHA256
`c98a1ec45ed907dea9313296a39243764dc66746f1c8b46c9bfe5903eda37920`）。
RT1/2實際查詢及使用RGBA／UNSIGNED_INT_2_10_10_10_REV，沒有降至RGBA8。
包含ordinary VS與真正TCS/TES各4draws、LOAD/scissor/masks/additive blend、
缺少RT3 export及clear後alpha-only覆寫；12TCS／12TES、72PS，所有8筆counter
保有正確4target格式vector，PDM/DRAM讀回bytes與pool平衡符合獨立預期。
WORK `tmp/mixed-mrt-live.XwNiM5/LIVE_RECEIPT.md`；reviewed-report SHA256
`bff718e11f59ecffd2f4e5d98dd0fc4cda4f83f65b5a936dd567bdcc085248d8`。
最初wrapper錯找hello而非counter欄位，原始false報告保留；只修正離線驗證，
沒有重跑或改動native輸出。完整16組Mesa resource單元測試也已全通過，
artifact `tmp/pvrgpu-mesa-unit.OGg49C`。

新frozen candidate（不覆寫既有runtime）：

- driver8 `driver-mixed-mrt-v8/install`，SHA256 `069e8ca48768cc8516bd0e51208111ad3ac4c31b912e32d73f53734fa917d293`
- model8b `model-mixed-mrt-v8b/build/lib/libpvrgpu_systemc_bridge.dylib`，SHA256 `2d61c5f87e49c07584f31fdc48d88da477fa4c1129b809fd483ad7cd404547c5`
- model8為缺少depth color_format NULL檢查的中間版本，不作最終API31契約認證。

model8b focused67/67 CTests通過（8.52s、無skip）；新API458 assertions、
format contract3779、parser48，保留原38項tessellation focused與既有geometry／
layered／color-depth-integer LOAD／舊API guard-page測試。5個test-only當前版本
static_assert更新成31，舊版本輸入與guard-page大小保持原樣，私人編譯副本只
允許這些精確差異；frozen source和bridge均不改動。494個保護hash一致。
WORK `tmp/model8b-focused-api31.Ggf244/REPORT.md`。

### 已完成真實 Draw156

`native-model8b-strict-gles31-prefix156` 實際完成0–156，共157ordered ranges，
event1958返回、driver/model audit true、最後GL scope無錯誤。這次沒有再被
tessellation_texture或mixed_render_targets擋住。156讀回1920×1080 D24與
四個color附件；目前color1/2仍為canonicalRGBA8診斷讀回，尚不能以該檔證明
packed10bit全部位元相符。獨立完整scope／前155輸出不變稽核及無損10bit
helper驗證正在進行；接著使用同一frozenruntime重播到175。

細分156的實際工作：624patches、1872TCS invocations、7488次TCS texture
sampling；此draw所有patches被剔除，TES及FS invocations為0。所以不能把
capture156的成功返回說成「已實際執行TES取樣」；TES路徑目前由獨立GL fixture
證明。需以相同llvmpipe boundary確認剔除結果，再往後檢查實際TES/FS draw。
該range另含framebuffer切換所完成的前一筆submission，統計時必須分開。

獨立156稽核完成：candidate與LP各157ranges／158乾淨API scopes、10946
preflight calls；1039個受保護輸入不變。與model7 prefix155比較，228份raw及
snapshot metadata全部一致；159份舊model reports僅pool_high_water_bytes增加
16（新的state metadata大小），其餘functional counters與driver compute counters／
command順序不變。156五份canonical附件均符合LP，但color1/2仍非packed證明。
報告 `strict-prefix156-certificate.lcmuk8/report.json`，SHA256
`3ac0720c444392708c9ed354318cce1059834c2bb3c0e74ad7900ec1f7414914`。

更新：嚴格llvmpipe prefix156也已完成，其最後實際draw為CSV row205／marker157，
IA1872、624patches、HS624、DS0、PS0，確認此draw剔除結果一致。marker157
另含19個內部mip/blit draw，不能拿marker總和與原生單一draw比較。

### 已完成 Draw175，開始 frame 結尾重播

`native-model8b-strict-gles31-prefix175` 成功完成176ranges、event2493，
整體audit true、最後同步GL scope乾淨。175 color0已可見1920×1080岩石地形，
仍是中間G-buffer，不是最後frame。171實際DS9014／TEStex9014、PS153631／
FStex916336；172、173也有非零TES／FS工作，並非只有全剔除draw的成功返回。

獨立175證明：177乾淨API scopes、328verified snapshots；前233raw／metadata、
161model reports（只正規化artifact_png路徑）、compute counters及command序列
全部不變，915保護檔一致。`strict-prefix175-certificate.Urpp9t/report.json`
SHA256 `2d04a5f653ee94a47c7d0b15df5f816a4d3b7d038fbe3ce806ea40e3f51f93af`。

新的私人 helper `REF/helper-packed10-ovr.BzcEha/bounded-player`（REF為本文件
固定llvmpipe工作目錄），SHA256
`eb1c1b5b2efc3ac7078f165efed75e91fd9a9345b4452f105119755457cf08e1`。
增加live internal-format查詢；RGB10_A2只接受實際advertised的RGBA／
UNSIGNED_INT_2_10_10_10_REV讀回，raw保持32bit與明確endianness。PNG從獨立
display copy轉換，不改寫raw；observer、ordered replay、D16/D24路徑不變。

Helper mock576scenarios／477715checks通過；LP/native真GL各8draws、3530checks，
32次附件讀回、512raw words符合獨立oracle且兩backend一致。每次驗證dirty
readFBO／PBO／PACK狀態還原及同步GL observer，所有native work／pools正常。

以此helper和driver8/model8b從原capture重播到EndCapture event6649的嘗試：
`native-model8b-packed-full6649` 已停止，在Draw292／event4486編譯拒絕，
整體returncode1、audit false。Draw0–291已完成，但未執行其後draws；
尾端event6649與最後draw394/event6629不同，不能宣稱全frame完成。

Draw292具四個vertex attributes、五個FS textures、四個mixed MRT；VS CB0
384bytes（96 DWORDs）、FS CB0 80bytes，另有VS/FS外部UBO binding1。
目前一般VS/FS compiler會把完整bound CB0計入shared-register預算，再於
allocator保留完整buffer；VS上限96，descriptor prefix亦占同一bank。
新GLES3.1 compile-only initialization＋真正Draw0小範圍探測已取得正確NIR：
兩次762個shader creation與1051個normalized create/bind events完整相同。
Draw292為ordinal703VS／704FS、GLSL190，不採用舊3.2的768份dump索引。
VS最高load確實抵達word95，不能裁短CB0；它宣告一個UBO但沒有UBO讀取，
FS則真的有UBO讀取。修正方向為只移除已證明未使用的descriptor，而非提高
硬體上限。證明 `nir292-packed.9dU4gw/verified-shaders.json` SHA256
`37db883dc763a7bf74af7914139b4230412e05e3c78a9241d0b91d21b5f2eae9`。

已實作一般VS/FS shared-budget修正：只有舊配置超過實際96／256上限時，
才回收已證明未引用的UBO／CB0尾端；保留原binding／word索引，不改shader
值或硬體上限。dynamic UBO保留全部，`get_ubo_size`視為引用；FS有image時
保留已固定的UBO descriptor前綴，避免改壞image位移。32個真NIR案例正常／
ASan+UBSan各399checks，舊版10個rescue negatives；16份既有binary／ABI不變，
四組既有compiler suites與Meson零參數模式通過。WORK
`tmp/cb0-prefix-expanded.1UxTmx/REPORT.md`。

真GL小測另發現recorder仍用原NIR declared-count要求完全相同，第一版driver9
因此在`stage=uniform_buffers`拒絕；不是GPU執行PASS。補上僅一般VS/FS接受
compiled prefix≤declared的transport contract，其他GS／TCS／TES與其相鄰stage
保持exact-count。最終6095項UBO snapshot／descriptor checks正常與ASan＋UBSan
通過。Sanitizer另找出既有測試把31筆傳給僅30筆storage的過期fixture：全stage
上限已為75，31不再先被count gate拒絕。修正測試為76筆拒絕，另以足量storage
驗證31筆內的bad-entry；沒有以production繞過隱藏越界。

完成重建driver9b SHA256
`4255cc2536cc7cba02e5b178461e51ca704593e1a5af18bab574ad21c16802f3`，
沿用未修改的model8b。真GL fixture已完成8draws／545checks，96word CB0＋
未讀VS UBO，以及92word CB0＋真正VS/FS UBO都符合獨立整數oracle及LP；
8個model reports pools平衡。原wrapper把8clear＋8draw API完成數誤當應為8，
保留其false結果，另做58項offline分類稽核PASS，不重跑或改寫原輸出。
新`native-v9b-packed-full6649`已完成Draw0–330，在331／event5327拒絕，
整體returncode1／audit false。剩餘64draws與tail尚未完成，不宣稱frame完成。

0–291獨立不變性證明：908raw＋metadata、296完整model report triplets、
300graphics／clear commands與兩次compute counters完全相同；292ranges／
293API scopes乾淨。2410檔案與6個bounded log prefixes雜湊不變。Driver log
剩餘159筆差異只為已逐項對照原碼`%p`的跨process host addresses，保留原報告
及獨立pointer supplement，沒有把numeric差異隱藏。證明
`v9b-prefix291-certificate.EqqggW/report.json` SHA256
`4e7d45127724cf00c447ad87bd1994d820dd6fd039e0c61d078529fc796e2bab`。
此證明範圍只到291，不能推論native與LP畫面相同或完整frame通過。

Draw331 trusted NIR為739VS／740TCS／741TES／742FS，GLSL198；與失敗run
762個creation signatures及1051個normalized initialization events完全對應。
原TCS四個／TES七個UBO讀取均為static scalar block0、32-bit vec4，僅byte
offset動態。Private compiler-only重現指出preprocess後TCS實際為vec16；既有
PCO `trans_load_buffer` 與native `UscUniformBufferMemory`／tess ISS已有完整
16DWORD LD／WDF路徑及精確resource-range檢查。已完成最小gate修正：只接受
32-bit且NIR合法的1/2/3/4/5/8/16component讀取，get_ubo_size保持scalar32。
沒有提高硬體容量或新增dynamic-block／host shader捷徑。六組compiler tests
正常與ASan＋UBSan皆通過，wide UBO427checks／舊9b三stage拒絕23checks；
TCS／TES／GS產生真正LD16，normal／san binary完全相同。Sanitizer只instrument
production pco.c及test TUs，既有Mesa／NIR／PCO archives未instrument，沒有
宣稱全Mesa leak證明。WORK `tmp/wide-ubo-validation.H7zWRl/RECEIPT.md`。

凍結driver10 SHA256
`1a82f659e97a6b12f258cd266acef14134be0fca5e89cfb1761d26189f7bd9e9`，
pco.c SHA256 `06bda52c750f0ee0465a570e81efa91c02884191065a2382570b23848a1a3109`。
`tests/tessellation_wide_ubo_live_probe.c`真正GL24draws／2384checks在LP與
native完全符合獨立oracle。涵蓋TCS-only、TES-only、both，動態matrix offset、
UBO rebinding／range offsets、全部16words、TF guards與RGBA32UI全像素。
另開print-only compiler診斷，逐draw確認實際TCS／TES LD16，24整份counter／
done objects與未開診斷run完全相同。Original wrapper誤預期24integer clears
也產生24SystemC提交；實際clear已填authoritative storage而不走normalized
legacy clear capsule，僅24真draw API pairs。保留false artifacts與55checks
獨立稽核：WORK `tmp/wide-ubo-live.fjSeYQ/REVIEWED_RECEIPT.md`，report SHA256
`33c1ff7b8bb9443e9b1f25a52cce4f82f35950f8b595bf397d0ede8b4ff9200a`。
`native-v10-packed-full6649`已完成Draw0–334，在335／event5407拒絕：
`color primitive varyings are unsupported: vs=0x1f00000001 fs=0x1f01000000`。
整體returncode1／audit false，tail未完成。Draw331實際driver instance_count=3，
有完整native draw_pco_sequence submit／done；RDDrawcall census的instances=0
不能替代實際captured-call／driver draw info。

v9b→v10的0–330不變性證明：331ranges／332API scopes，1103份附件與335完整
model report triplets完全相同；339graphics／clear submissions與2compute
counters相同。2881檔案與6bounded prefixes未變。`v10-prefix330-certificate.JzlKtS/`
report SHA256 `4e2a1a0e329bfb478f75c06c14a4ecff92134bd48b6b1936f29df2d4b8d3c32c`。
此證明不包含331後續或full-frame完成。

llvmpipe `REF/strict-packed10-full6649` 已完成396ordered ranges（395 draws＋
tail）、397乾淨API scopes、1269typed snapshots，其中394份完整packed10。
前0–175共328raw／PNG與既有prefix完全一致；輸入runtime hashes未變。
證明 `REF/helper-packed10-ovr.BzcEha/full6649-verification.json` SHA256
`9cf23cbaf3e02de36f85860a316a07851193b5ccd66afb14862d3f7cf103f035`。
此為獨立reference認證，不是native-versus-reference PASS。

針對native已完成的指定scopes，新的無損比較及獨立C++解碼交叉核對確認：
Draw156全部五附件完全相同；Draw171至少一個MRT不同的pixels為148183，
Draw175為419090。D24最大差92006units，並非僅微小量化誤差。
報告 `packed-delta3.h680kr/report.json` SHA256
`b0aa9010033f554d47ad4d9961db2bc3468a9ff9bec420e696e7cd5f4008b486`。
選定48份檔案與finalized journal prefixes未變，未認證失敗native run的全局完成。

接續精確逐draw查核：Draw156–170的75份附件全部與LP full reference相同，
此pass第一個差異為Draw171／event2381。兩邊IA4716／1572、HS1572、DS9014、
clip-input11527相同，clip/setup output LP8263、native8238；需追TES輸出與幾何
邊界，不可只歸因MRT量化。`first-packed-delta.eeXPSO/report.json` SHA256
`b8d6e2130d4892a74a6e188174441512913ccf3ada59ece46c77a646a678a207`。

原始LP Draw171被動trace已完成：1572個patch、985個zero-domain culls、
587個live patches、9014原TES domain vertices、34581 indices／11527triangles。
保存原TCS三個output vertices的semantic-labelled slots、factors、domain及
preclip position raw bits；沒有relink shader或TF替代原工作。僅私人LP
`draw_draw_tess.c.o`改變；unmodified control重現原library SHA。診斷prefix171
的308附件及221counter rows完全不變，172ranges／173scopes乾淨。
`REF/passive-tes171.RDOQz3/passive-tes.jsonl` SHA256
`4992e1e4a463107f14ba0e376c8545d3244045534f899eae0c55793cdab1e694`。
Native對應的私人三object被動logger已完成prefix171；只複製原callback
payload與已產生domain／task outputs，不額外呼叫GpuMemory或更改counter／
cache。308raw＋metadata完全不變，包含overlapping observed／range的45678
model records、3337948numeric fields完全相同；這些累加數不是unique GPU work。
全部1572patches的domain counts／indices相同；2112個used factor words、
1589個UV points及9014個position vectors有差異。

第一個used factor差異為patch184 outer[1]：native `3f8c8e5c`、LP
`3f8c8f7c`。該patch三個vertices全部39個語義words完全相同，且native TCS
原始輸入到輸出沒有改值。TCS實際PCO的mv乘加順序與原NIR的unfused順序，
離線各自完整重現這兩個factor；該非零factor的dataflow不依賴texture結果。
仍需區分數值精度差異與不合法的compiler／ISS行為，不以pixel差異直接判bug。

TES patch184／point0使用完全相同corner UV與vertex1語義資料。原生sample
alpha為`3f074ba4`；依觀察到的shared constants與融合矩陣鏈可重現全部native
clip words。離線將該alpha經真正_Float16 RNE轉為`3f074000`，再依原NIR
unfused計算，則完整重現LP四個clip words：`4183d780 bf029a00 41c3c928
41c56068`。這只是可重現的precision假說，未證明原LP實際sample bits，
不能據此直接更改TPU結果或注入LP資料。WORK `tmp/tes184-numerics.F7E2EV/`
保留原始與half-control source／輸入／outputs／hashes；沒有GPU重播接受修改值。

後續原LP內建`GALLIVM_DEBUG=tgsi`取證已確認：同source_blake3的GLSL193 TES
實際NIR是`16x4(float16)txl`，再`f2f32`取alpha；lowp sampler決定結果精度，
不因UV為highp改變。Pinned LP在F16C使用toward-zero轉換，不是一般RNE；
前述patch184特例在兩種rounding下結果相同。診斷prefix171全308raw／metadata
及221counter rows不變，172ranges／173scopes乾淨。證明
`REF/tes171-precision-dump.pKW6i6/REPORT.md`，receipt SHA256
`889444db698a68bca1abb12f712d3c7efd6b6b2cc3e4f5db8eb05c4c2dae4d1f`。

Offline擴大到全部1809個相同corner UV：observed native alpha＋實際解碼的
native displacement／FMA順序重現1809／1809原生clip；同alpha經F16C RTZ再
按原NIRunfused計算重現1809／1809LP clip（各7236words）。RNE control僅
1011／1809，raw alpha＋NIR與RTZ＋native順序皆0／1809。1189個全13word
輸入完全相同的subset亦全符合。這是stage精度／編譯政策差異的定位，不是
全9014domain points或最終pixels的等價證明，也不能據此全域强制TPU FP16。
獨立資料WORK `tmp/tes-corner-numerics.93OS8A/results-rtz.json`。

GLES3.1修正後、同一嚴格prefix155的精確差異：

| Draw | Color不同pixels／921600 | RGBA最大差 | D16不同pixels／最大差 |
| --- | ---: | --- | --- |
| 102 | 0 | 0/0/0/0 | 0/0 |
| 103 | 2616 | 10/6/3/23 | 6/1 |
| 120 | 6179 | 14/7/4/21 | 31/1 |
| 138 | 838 | 8/5/3/10 | 4/1 |
| 155 | 187438 | 14/13/8/24 | 121/1 |

這些是實際raw差異，未套容差或宣稱PASS；D16 max/nonmax mask相同也不是
primitive coverage證明。報告在debug工作目錄 `strict-color-delta5.1rL7vm`。

### Draw335：逐片段 FrontFacing 的原生接法

完整762個shader creations與1051個初始化create/bind事件確認：Draw335／
event5407使用VS731／FS732、GLSL230。FS為flat uint32 FACE load_deref，
再經b2b1決定DATA2法線正負；這不是VS缺少一個user varying。實際indexed
triangles有3個instances，culling關閉。不能以action metadata的instances=0
推論沒有工作，也不能只放寬varying mask或固定成正面。

Compiler只改private cloned FS：將受驗證的scalar FACE或load_front_face
改為真正PCO BACK_FACE／FACE_ORIENT special44等於0；原uint32布林讀取
保留canonical32-bit true表示。模型已在clip/cull分類套用API winding，故
採既有PCO front-face op的NOP等價形式，不再增加winding uniform，也不
改動UBO／descriptor／CB0 shared ABI。無FACE的shader保持原路徑。

ISS要求有效、canonical的raster-facing context；值來自原始primitive，
包含covered與helper lanes，SMP／derivative suspension與resume會保存並
比對identity。FragmentShaderLane重用原reserved byte，不增加payload大小；
PDS利用原本已載入的ParameterTriangle檢查，沒有新增GPU memory read。

新driver11 SHA256
`9592f930723b349d09d7b5601f05c2c8802c751da431de92b7a779c9c780d92c`，
初版model9 SHA256
`6c005261d6328d81416a4cce22ef178bdce858a612f602a3faf5e97ca1d60638`。
`tests/front_facing_live_probe.c`在LP及此native各48draw／12401checks通過，
包含正反頂點順序、glFrontFace、X反射、oversized clipping、disabled／FRONT／
BACK culling與背面texture branch。舊driver10首draw在FACE varying guard
拒絕，沒有model完成紀錄。證據WORK `tmp/front-face-live.vVZlB7/`。

另增小三角形helper probe，每個有效draw有15個可見pixels、8個quads及
17個未覆蓋helpers，floatBitsToUint逐bit驗證dFdx／dFdy為1.0。LP48draw
通過；初版model9首draw暴露既有derivative-only／no-user-varying的position
coefficient preparation缺口，故此初版helper run是FAIL，不是helper支援通過。
修正方向是讓此合法shader走既有position coefficient流程，保留所有layout／
ABI比對。model9b加入predicate後，因導數分類直到Fragment PcoDecoder才
建立，Submitter／ClipCull／ParameterBuffer先前仍未準備position payload，
故helper首draw改在PDS正確拒絕；不是已修好。新版model9c由Submitter先從
自己持有的實際FS bytes只推導uses_derivatives規劃bit，正式PcoDecoder仍
讀取原生code、解碼並比對此bit。不能用host預解碼虛增model instruction／
timing或替代正式memory fetch。原本沒有導數的shader仍保持原payload分支。
完整管線小測試已涵蓋front、back與中途分類更改拒絕；實際GL與CarChase
結果另記，不改寫舊失敗artifact。
證據WORK `tmp/front-face-helper-bits-live.xJUk41/`。

Compiler最終272 normal／272 ASan+UBSan checks，23個不支援／不合法形狀
拒絕；generic／GS／TES三條管線共18份genuine binaries。No-FACE三份binary
及ABI與driver10逐byte一致；七組compiler regressions正常及sanitizer皆通過。
範圍僅pvrgpu_pco.c與tests被instrument，Mesa／NIR／PCO archives未被instrument。
完整receipt WORK `tmp/front-face-compiler.5FhyjP/RECEIPT.md`。

Fullscreen model9→9b的stdout、48份model triplets全部數值／shader fingerprints
及final command逐byte相同；獨立驗證共640次SMP、2048visible fragments。
Clipping後每個有效texture draw有20個per-primitive quads，因此背面80次SMP，
不是原始64可見pixels的64次；多出的16 lanes是helpers。Full attachment的
pbe_pixels_written在culled draw仍為64，coverage應以ps_invocations／
covered_pixels／pbe_fragment_writes的0值判斷。驗證器沒有忽略未知stdout／
JSON錯誤，也沒有把driver API done單獨當成model成功。
WORK `tmp/front-face-live-audit.eLawlo/REPORT.md`與`V9B.md`。

新版model9c SHA256
`938a28ba7fdfc356d1f41c27047d60000ccdcd4a5521f71fed5126428c4eb0dc`。
兩組真GL fixture的`candidate-v9c`各48draw／12401checks通過，helper包含
逐bit導數與全部coverage／facing oracle；每組48份完整model completion，
pool allocation／release平衡。這是model9c的小型整合驗證，不是其CarChase
全frame認證。當時的CarChase ordered run固定model9b；後續runtime另建
新目錄與receipt，不混用建置中的runtime，也不把舊失敗目錄改寫成成功。

本次重跑CLI／driver-tree／RenderDoc alias＋OVR source tests共49項通過。
第一次OVR mock compile選到系統c++而缺少標準array header，屬測試工具鏈
選擇錯誤；明確使用既有LLVM clang++後全部通過，沒有修改測試或production
來忽略該錯誤。

### Draw335後續：VS shift重用VTXIN暫存器

driver11/model9b完成0–334 scopes，但Draw335尚未提交model：FS FrontFacing
編譯已成功，接著VS native decoder在byte46回報`shift destination must be
temporary`。原始error中完整group為`56 d2 40 00 02 80 a8 80 87 86 04 ff`；
其中extended destination `86 04`是bank2 VTXIN6，而非未知register bank。
可信VS731 NIR使用`(gl_InstanceID + instance_offset) << 7`計算128-byte
instance UBO stride。Pinned PCO register allocator本來允許SSA使用VTXIN
工作空間；現有vertex ISS寫入與依賴追蹤也支援，缺口在shift decoder只准TEMP。
修正範圍是讓vertex stage的shift接受已有限制的VTXIN，其他stage／bank保持
原拒絕。完整重建的1544-byte真NIR編譯結果可解碼140groups，與error保留
的全部259-byte前綴一致；這是compiler重建，不是假稱取得完整actual binary。

0–334獨立不變性證明：1123組raw附件及typed metadata完全一致，336個
API scopes乾淨；332個有native work的scopes含336份完整model triplets，
所有counter／shader fingerprint／virtual timestamps維持原樣（PNG目錄除外）。
Draw332／333／334實際解出的indirect instance_count皆為0，各有
`draw_indirect_empty reason=zero_work`且沒有native submission/report，明確
保留為no-op。6426個host pointer差異皆逐一記錄，沒有其它driver記錄差異。
證明WORK `tmp/carchase-debug.t3s25m/v11-prefix334-certificate.QQnY7N/REPORT.md`。

修正完成後，genuine shader unit各523 normal／ASan+UBSan checks通過，
舊model9c object仍對相同shader在byte46拒絕；9組既有ISS suites共18次
normal／sanitizer回歸通過。完整140groups／22個VS輸出及實際UBO位址
以獨立輸入oracle驗證。WORK `tmp/vertex-shift-iss.FWJ77L/REPORT.md`。

新版model10 SHA256
`1586f3dc299eb586952c487e1cd49d21b77fccd339d2e0db036f1ef995063f77`。
48個fullscreen FrontFacing、48個精確derivative/helper與16個packed
attribute＋UBO/TF真GL draws全部通過。最後一組在舊model9c也通過，
所以只是功能控制，不是假稱成功重現VTXIN decoder缺口；原wrapper期待
失敗而回false的receipt保留，獨立稽核另記。WORK
`tmp/vertex-shift-live-audit.tLdcpf/REPORT.md`。

實際`native-v11-shift10-s32-full6649`已完成0–353，Draw335 VS 1544bytes／
FNV `e348dc0343bb3d20`、FS1888bytes／FNV `a1a33955da292214`有原生執行
報告。這次每32draws取附件而非每draw取附件，所有ordered工作仍執行。
Draw354失敗，故整體`evidence_audit_ok=false`；不能以prefix成功覆寫它。
重建303 VS的完整FNV為`a2c034cefca60984`，與上述actual VS不同；因此
只聲明相同1544-byte長度、保存的259-byte前綴及失敗group吻合，不聲明
完整shader byte-identical。新actual完成證據與小fixture語意證據分開保存。

獨立prefix353稽核：354 finalized ranges／355 clean API scopes；346個
native-work scopes、8個zero-instance no-ops；350個graphics model reports、
355個graphics API pairs（350draw＋5clear）及2個compute pairs。前335
所有model records／counter／virtual time／shader fingerprint維持不變；
11個interval32保留邊界的37組raw及metadata完全相同，不推論未保存邊界。
Driver log因較少讀回而少69組sampler rebind pairs，差異逐項保留，沒有
假稱整份driver log相同。WORK
`tmp/carchase-debug.t3s25m/shift10-prefix353-audit-v2.69ydN9/REPORT.md`。

### Draw354：SSAO shader的原生FTB指令

新邊界event5803為4頂點triangle strip、960×540 R8_UNORM單RT、無depth。
Chronological shader creation／binding需追蹤pointer重用的generation，
不能用全stream第一次相同pointer：實際VS729、FS730／GLSL111，而非已
釋放的FS622。來源0730 NIR有7次`ufind_msb`，用於SSAO取樣mip選擇。

目前實際failure為FS size10600、byte2216
`unsupported bitwise-immediate instruction-group header`。編譯器全圖
重建的10600bytes符合actual error保存的259-byte前綴；其byte2216為
`45 94 40 20 80 40 00 6a 00 6a`，PCO自有disassembler確認FTB s2=r42、
W1=r42。現有origin1分派只識別MOVI的W0形式，並非捕捉檔壞掉。
完整binary仍標示為重建，沒有假稱是actual全量擷取。

新增真GL `tests/find_msb_live_probe.c`：unsigned／signed每個bit位置，
zero與負值，24個radius的findMSB→clamped LOD→實際texture sampling，
以及scalar R8輸出，共112draws／7423checks。LP全部通過；model10於
首draw FS byte40重現相同header拒絕，0份model completion。該小fixture
error保存完整112-byte actual FS，作為快速原生decoder負控制；FTB修正
後的結果另記。WORK `tmp/find-msb-live.WcbGG3/`。

FTB修正已限定於fragment bitwise phase0、DA4、W1-only、S2來源及TEMP目的；
zero回0xffffffff，其餘回最高set-bit index。保留其他stage／count-op／
modifier／不合法metadata拒絕。Genuine80-byte與actual112-byte shader
的raw／prepared execution、位元邊界、隨機輸入、一次ALU計數與跨SMP/WDF
保存FTB結果，共9107 normal／9107 ASan+UBSan checks通過。完整重建FS
可解碼872groups／7FTB，但不把解碼成功當成完整capture執行成功。

獨立完整O3／NDEBUG／diagnostics-off model11已建置，沒有fast-math，SHA256
`224edb29f1f56720488165ccd0789ca0e65aa043d59a044cfff7d01ab503884c`。
`find-msb-live.WcbGG3/candidate-v11`實際112draws／7423checks全部通過，
112份native model completion及API pairs，零driver errors／pool leaks；
結果與LP獨立CPU oracle一致，包括R8通道預設值。CarChase新run另存
`native-v11-ftb11-s32-full6649`，最終完成0–354，下一節另記新邊界。

Draw352／event5766最後保留的完整MRT中間圖已有地形／岩石／植被，但
raw與LP仍不完全相同：color0有606992個不同pixels，color1有325570、
color2有1154728、color3有94850、D24有1287630（分母各2073600）。
color0最大RGBA code delta為130／82／43／217；D24最大184304。
這是同格式、同API邊界的精確比較，不是PNG視覺相近即PASS，也不是判定
第一個差異在Draw352；前段Draw171已發現精度／運算順序差異。
WORK `tmp/carchase-debug.t3s25m/draw352-comparison.zSfpZf/report.json`，SHA256
`3521f9b4af9333d0d7cba5277d5ca9f61193cf50e0377fd5c207ff9358917729`。

### Draw354完成；Draw355的array shadow gather

Driver11/model11的actual354完成：FS10600 bytes、FNV `1f373f568c69d634`，
static FTB7／SMP9，PS518400、FS ALU292251614、texture requests3208168。
完整replay仍失敗：Draw355／event5829的FS `tg4 dim=2d array=1 shadow=1`
在texture/sampler slot2被compiler admission拒絕。該失敗scope先有一個
成功clear_color API/model triplet；沒有Draw355的draw_pco completion。

獨立稽核確認355 finalized scopes0–354、356 clean API scopes；347個
native-work ranges加8個zero-instance no-ops。前0–353所有model records／
counters一致（僅PNG路徑重定位），12個保留邊界共42份typed/raw附件、
每run306892800 bytes完全相同。Draw353有一項非indexed DrawArrays的
未指定index_bias log值不同，明確保留；實際rebasing只在index_size非0
時使用該值。沒有宣稱完整driver log相同。Draw354本次未保存raw附件，
不能聲稱SSAO像素與llvmpipe相同。WORK
`tmp/carchase-debug.t3s25m/ftb11-prefix354-audit.vyvOGq/REPORT.md`，
report.json SHA256
`b8cdd8d5232b9c2f5d8b9558e2c85fbf39928dfa43ff193a26cb75b83fef73d7`。

Actual355使用D16 960×960×4 array、單一level、CLAMP_TO_EDGE，sampler
min/mag LINEAR、COMPARE_REF_TO_TEXTURE／LEQUAL。原PCO已能將layer
round/clamp後轉成TAO地址，取得四個raw taps，再由shader比較；不能
在TPU改成比較插值後的depth，也不應以此放行一般linear shadow sampling。

新增 `tests/texture_gather_array_shadow_live_probe.c`：D16/D24、三層
不同內容、nearest/linear兩組filters、八種compare、14組UV/layer/reference，
共448draws／16791checks。LP全部通過；舊driver11/model11首draw拒絕，
此前48個D16 source upload words已精確驗證。獨立Mesa binary32 oracle
重現全部545行transcript；未涵蓋half-layer ties、非0view base或巨大layer。
新版driver12/model12的candidate-v12已通過全部448draws／16791checks，
545行typed transcript與LP一致，448份native model completion，無driver
errors；actual CarChase是否通過355仍由獨立full replay決定。
WORK `tmp/array-shadow-gather-live.7e5V2P/`、
`tmp/array-shadow-gather-review.rry66l/REPORT.md`。

修正分成三層，沒有新增ABI或在host/TPU預先比較深度：

- Compiler：真正PCO保持四個raw taps，FS-only 2D/2DArray shadow red
  gather；matching statictexture/samplerslot、scalar comparator、無offset。
  新`has_shadow_gather`只負責保守路由，`shadow_gather_only`另證明所有
  matching-slot採樣均為支援形狀。219 normal／san checks、六組既有compiler
  entry regressions，以及80/1608/1168-byte portable binaries完全一致。
- Driver：有shadow gather即強制專用state guard，不經舊nearest/query
  捷徑；MSAA及compare NONE也拒絕。限定depth red X、單一exposed mip、
  clamp-edge、normalized、無anisotropy。特別修正nearest 2D常數swizzle
  可能繞過guard的入口，因GL gather是在comparison後套常數swizzle，
  不能比較canonicalized常數depth冒充。實際context snapshot6116checks
  normal／ASan+UBSan通過，包含mixed ordinary shadow、nearest ZERO/ONE/Y、
  repeat、多mip及有效單sample MS拒絕；mip1＋array layer1..2重定位逐word驗證。
- ISS／TPU：只增加gather的genuine0x91 TAO；array地址必須恰為
  base＋layer×stride且在界內，不能floor錯位地址或clamp越界地址。TPU仍
  經實際cache／L2／DRAM回四個raw taps，由shader ALU執行比較。
  81項normal及81項sanitized focused tests、22組ISS回歸通過；真array
  bytecode21526checks，真plain2D shadow1348checks，均normal／san通過。

Frozen driver12 SHA256
`f65e96bce14e75245b3a1011c50ae32a7239ab1b48754b137b6fcb6350ce6f31`；
完整重建O3／NDEBUG／diagnostics-off model12 SHA256
`a099cf01a2bbca1dfc91073c4fa8df2a3a6a878ca71f2b285c1781f0c02b3db4`。
WORK `tmp/carchase-debug.t3s25m/{driver-shadow-gather-v12,model-array-shadow-gather-v12}/receipt.json`。
新實際capture replay為`native-v12-shadow12-s32-full6649`，已結束並完成0–357；
仍全range replay，沒有whole-state checkpoint或跳過prefix。

最終context證据WORK `tmp/shadow-gather-snapshot-v3.v5oh3r/receipt.json`，
SHA256 `05607ced953ef87a8b4cff3bcc148245b4a9852c20106958d6d78c98454a567d`。
先前5939／6080checks的中間receipts保留但不是最終guard認證。
一般`script/run_mesa_resource_unit.sh texture`也已實際通過6116checks，
連結當前production PCO分類器而非stub。舊選用real-NIR測試入口同步補上
共用static resolver的擷取，實際啟用Mesa後normal／san各86checks通過。

新版raw非shadow gather live336draws／16561checks與半texel邊界576draws／
28322checks均exit0，輸出與其先前LP結果一致；完整counter稽核另列報告。
這些小測試不是完整frame correctness認證。

獨立array-native稽核WORK `tmp/array-shadow-gather-native-audit.luRBHq/REPORT.md`
確認1344份ordered model records／448API pairs，每draw8次SMP、32 taps，
共3584 requests／14336 taps；NEVER/ALWAYS也實際取樣。report.json SHA256
`a7993be28c787b2c5bcb6e4f54a52a19543d83b06e6b2bb0ef661eb5591ce448`。
私有plain2D shadow GL fixture在WORK `tmp/plain-shadow-gather-live.w4zQCP/`
另驗證真sampler2DShadow/vec2，LP/native各448draws／16655checks；保存
全部7168個actual RGBA32UI words，逐word符合獨立oracle與LP，448原生
submissions完成且無leaks。其audit.json SHA256
`65102f5f3873132c9897b25e138b45e19c657a1b5df15ee6081addcc0355d6ec`。
這是plain2D整合證據，沒有拿array unit的重建binary冒充actual GL binary。

### Draw357 完成與 Draw358 的下一個原生邊界

Frozen12完整run停於Draw358／event5899，exit1，沒有timeout。
獨立prefix報告 `tmp/carchase-debug.t3s25m/shadow12-prefix357-audit.jymu14/REPORT.md`：
0–357共358 finalized ranges、359 clean API scopes，350有native工作、
8為明確zero-instance no-op。0–354所有model欄位（除artifact路徑）不變；
42份保留typed raw附件合計306,892,800 bytes逐byte不變。355–357皆有
真native draw completion，但interval32沒有保留這三個draw的附件，
不能宣稱這三張圖與LP相同。358僅完成clear_color的7,045-byte model
triplet後失敗，不是完整draw，也不是整個scope零工作。

實際358的FS透過762 create signatures、1051 initialization create/bind
與generation-aware pointer追蹤，確認為0672-fragment.nir／GLSL120；
證據 `tmp/carchase-debug.t3s25m/nir358-cubearray.WTOBcs/REPORT.md`。
它有9個真texture slots、CB0 224 bytes／56 DWORD及1個UBO descriptor：
9×20＋56＋4＝240，沒有超過FS256。第一個拒絕是舊cap8，下一個是
CubeArray resource/view target。CPU-present guard本身位於generic lowering
之後，必須保留；新增較早的gate-declined日誌，顯示真正原因。

`static_envmaps`的NIR slot6對應app unit8／captured CubeArray resource0。
Pinned Mesa未完整texture的nonshadow fallback會建立RGBA8 1×1×6，
與現有view候選一致，但舊log只記首view，沒有閉合slot6 pointer／bytes。
不能硬編黑色、忽略sampler或跳過shader分支。API32／cap12與明確kind4
CubeArray正在小測試驗證中；VS96／FS256與既有unsupported gates不放寬。

### Draw358 修正：API32 的12 bindings與CubeArray

已建立新的frozen `driver-cube-array-v13`／`model-cube-array-v13`；
driver SHA256 `fe60f2488da8ca0164abbc87dfdf88fecc20ac58468242192c1706414be9c60e`，
完整獨立O3／NDEBUG／diagnostics0 model SHA256
`48d4b8ff64653c87a53ca726af3d2730c15163c89c0d458610e399278940539a`。
API32區分新的descriptor容量與kind4语義，public struct欄位／大小沒有改；
舊API版本仍先拒絕，不讀可能不存在的tail。VS96／FS256保持原值。

Model `tmp/cube-array-model.LFmaGw/REPORT.md` 的最終結果：100／100
normal及100／100 ASan＋UBSan相關ISS／TPU CTests通過，最終API另1次
sanitized執行通過。8個current API32可執行檔／13次回歸全部連結exact
model13通過；新增176checks，包括22個malformed／old-version拒絕、
真CubeArray與12-slot＋UBO SH256 jobs及submit後caller資料改寫的deep-copy
驗證。真textureSize的尺寸／cube數／mip亦由raw與prepared ISS執行驗證。
最終verification JSON SHA256
`f13fc690768356574a795e8f4a41fc6725299a63eac677905fa5809a0fd5af87`。

Compiler count、driver arrays、ISS descriptor sets、TPU residency及sequence
總容量一致引用共用cap12。CubeArray仅放行FS普通tex／txl及size query；
shadow／gather／offset／gradient／bias／dynamic binding不順帶開放。
Snapshot限制uncompressed color、square、single-sample、整個6-face cube
view，可重定位first cube／mip；當前driver中間raw格式最多2046完整faces。
Raw descriptor的depth＝cube_count−1，word4＝base-level一面stride。
TPU從真TAO解析cube base，face／seam／corner均保留cube base，仍走
正常texture memory path，沒有copy reference pixel或硬編dummy texture。

Compiler `tmp/many-textures-final.GAfkbt/REPORT.md`：402 normal＋402 sanitized
checks，四組舊compiler回歸normal／san通過，真8-texture旧新bytecode逐byte
相同。Context `tmp/cube-array-snapshot-final.gqKGE9/`：6410 normal＋6410
ASan／UBSan checks，包含非零cube/mip view、完整face transport、拒絕
partial cube／depth／compressed／MS／錯stage，及slot11有效／空sampler／
空view與slot12越界；失敗不產生snapshot。Native-present guard另258checks
保持通過。這些是resource snapshot單測，不是whole-state checkpoint。

真GL `tmp/many-textures-live.NFbBgJ/REPORT.md`：8／9／12 bindings共32draws，
每一binding逐一修改並保留真CB0＋UBO；2048個actual output words完全符合
独立整數oracle與LP。舊12runtime僅完成前9個8-binding controls，9-binding
拒絕；那9個controls的新舊model counter全部欄位相同。新版12-binding
實際FS1424bytes／SMP12；沒有少算sampler以通過限制。

真CubeArray `tmp/cube-array-live.Q9SsI0/`：348 draw都native完成，192 nearest
cases精確符合input-derived oracle；LP對照共210／5568 words不同：48words
為constant linear/seam的1ULP，162words來自12個spatial implicit cases，
最大絕對差0.02246248722076416。另 `tmp/cube-array-spatial-control.eMCZ89/`
用普通samplerCube保留同source／XYZ／gradient，LP與native各192words均
逐word等於各自CubeArray輸出；因此這12cases是既有Cube LOD數值差異，
沒有新增CubeArray額外delta。仍不得稱為整體LP pixel-parity PASS。

歷史控制 `tmp/cube-array-live.Q9SsI0/HISTORICAL-CUBE.md` 另用完全相同probe
在frozen driver12／model12執行普通Cube12cases：192words及全部stdout
與13逐byte相同，model欄位只除PNG路徑後全部相同，確認此spatial差異
在CubeArray新增前已存在。最終snapshot receipt SHA256
`add6390133de7dcfb7e2e12402d335fe6b49d7b97bbd84fdca9c7f4efb7c3e43`；
public `run_mesa_resource_unit.sh texture`也已通過6410checks。

完整FS358離線重建 `tmp/fragment358-reconstruct-final.ZB7Dqn/REPORT.md`：
319個FS定義、27個declarations、控制流程及兩個preserve:sz annotation
與可信0672文字相同；caller NIR不變。真PCO9560bytes／TEMP70／SH240／
CF12，915groups全部解碼。Binary SHA256
`c9a3aecff103405bff5f2d381a09bbabea61bf88390844c084cacf6b884f41f3`。
這不是captured binary identity，也未證明dynamic execution或pixel結果。

新實際ordered run為`native-v13-cube13-s32-full6649`，已結束exit1、無timeout。
獨立 `tmp/carchase-debug.t3s25m/cube13-prefix357-audit.QcKiAo/REPORT.md`
確認0–357全部model欄位（除PNG路徑）與42份raw／306,892,800 bytes不變。
只有非indexed353–357的index_bias記錄值不同，保留而非當pointer抹除。
沒有使用Car Chase whole-state checkpoint，也沒有跳過prefix。

358實際slot6已由新snapshot counter閉合到唯一view/resource generation，
RGBA8 1×1×6、24bytes；初始化upload只記第一texel0,0,0,255，不代表已存下
全部24bytes。Clear及draw各有API submit/done，但真正finish失敗，retained
model tail只有2468bytes的一個draw hello，無358 counter／done。不能把
API queued/returned當成模型完成；也不能宣稱retained clear triplet存在。
失敗為TPU implicit LOD的quad identity，尚不知確切descriptor／PC；
正在建立只加失敗資訊、不改成功運算的model14以分辨實際觸發site。
原FS的slot5與CubeArray slot6都有分支內implicit sample，沒有branch後
共同implicit site；一般scheduler提前送partialquad的假說不能直接當作
Car358根因。必須保留不同PC／quad／loop epoch界線，不填假helper座標。

### Draw358 的 LOD dependency 縮小測試

捕捉來源與 driver transport 查核指出：分支內 NIR slot5（GLSL名稱
`texture_unit7`，實際 app unit5）使用 texture55／sampler1929；sampler
為 LINEAR／LINEAR、no mip、MIN/MAX_LOD0，texture有12層ASTC5×5。
但 driver刻意把no-mip編成raw nearest mip／LOD window0..16（U4.6，
即0..0.25），所以原本僅靠mip_count>1的判斷仍要求quad derivatives。
不能直接把captured GL maxLOD0當作model maxLOD0。

`SelectTextureLod`先clamp lambda，`SelectTextureLevels`在active clamp
下用nearbyint選nearest mip。相同min/mag、nearest mip、有效window
max<0.5且兩端實際selector都選base0時，所有可達lambda都選同一層、
同一filter；這是與texture內容／UV無關的依賴性證明。仍保留真UV、
取樣請求、bilinear taps與cache／L2／DRAM，不補不存在的helper。
不同min/mag、linear mip或較寬window仍需quad；ambient FE_UPWARD
若使端點選到mip1，也不套用此簡化。

新 `tests/texture_lod_independent_live_probe.c` 使用四層不同的RGBA32F
8×8→1×1、變動UV、repeat wrap、nonzero sampler覆寫texture state，
以及四種branch門檻，輸出RGBA32UI。LP已通過16draws／1343checks；
每個word由輸入資料獨立計算nearest／bilinear base-mip oracle。
完全未改成功路徑的diagnostic model14，SHA256
`1c18c5d909d893306c1c9a3c4309190d11240fae2e08e4a7eabc8d8dba384bb9`，
第一個full-quad draw符合oracle，下一個partial-quad draw確實abort：
18requests、4mips、minmag0/0、window0..16、PC13。此舊版失敗是
專用小測試，不是Car358 actual PC已被定位；尚待修正版及實際capture。
證據在WORK `tmp/lod-independent-live.yINTTc/`。此時沒有改USC排程。

新的frozen `model-lod-invariant-v15` 已實作上述有界predicate，完整獨立
O3／NDEBUG／diagnostics0 build SHA256
`f2cd648286a2c848fd1976137bdf4f247f9553497171ee6dcaff64281bbac243`。
相較model14，production只新增pure dependency helper、TPU呼叫及
failure-only mip_filter欄位；driver13、compiler與USC沒有改動。
新native GL同一probe亦16draws／1343checks通過，1024words與LP及
獨立oracle逐word一致，16組model counter/done、API completion與
balanced pool皆存在。`tmp/lod-independent-model.7RihPG/normal-seven.xml`
的7/7 focused CTests通過：selector62771checks、真TPU2448checks／
96batches，含direct/bypass/cache、nearest/bilinear、完整與稀疏lane；
5種真正LOD-dependent partial-quad依然明確拒絕。後續test-only補強及
normal／sanitized舊回歸另記，不混成frozen15 source原已包含。

實際Car Chase新run為`native-v15-lod15-s32-full6649`；仍從native原始
capture依序執行，這不是從Car Chase whole-state checkpoint接續。
此run已結束exit1，完成0–359，停在360的新compiler budget問題。
358／359均有實際draw h/c/d、成功finish及clean observer scopes。
358：PS/PBE2073600、SMP12514200、texel18909600、FS9560bytes／
FNV`a4ebfc22710c4a58`；359：PS725784、SMP743292、texel5946336、
FS528bytes／FNV`b59f4ff26bdc2d49`。兩者皆balanced pool。
間隔32沒有要求這兩個draw的附件讀回，不能宣稱它們已與LP像素相等。

最終current-predicate回歸已完成108/108 normal及108/108 ASan＋UBSan，
沒有skip；pure selector62915checks、真TPU2682checks／105batches。
新增test-only bias policy／off-center bilinear九組亦通過；production
與frozen15 source完全相同。最終receipt
`tmp/lod-independent-model.7RihPG/verification.json` SHA256
`8aff9493b24deeb6b4a2246c2d7beeb3d97012fc5e11205d9126e624c140a8c4`。
Live獨立Fraction oracle另確認全部1024words、完整stdout、16有序
triplets／API pairs；192真SMP requests、480texel reads，linear四taps
沒有被省略。舊14第一draw三筆model records不作任何normalization
亦與15完全一致。報告`tmp/lod-independent-audit.Fl9G6Q/REPORT.md`。

實際15的bounded0–160已獨立確認所有model欄位（只除PNG路徑）、
12份typedraw／58,060,800 bytes、metadata與state完全不變；只認證
成功範圍，不把仍在寫入的全run當成完成。證據
`tmp/carchase-debug.t3s25m/lod15-prefix160-audit.uieOUE/report.json`
SHA256 `4dca65b197f0d1e53811ee32dd69323462bf83bd4c9b4e68928aaa4a95f809a2`。

後續完整0–357 prefix亦已確認：所有model欄位（只除PNG路徑）與
42份typedraw／306,892,800bytes不變；只有原樣保留的4421個pointer
差異及非indexed353–357的5個index_bias log值不同。全prefix證書
`tmp/carchase-debug.t3s25m/lod15-prefix357-audit.lx6yIs/report.json`
SHA256 `d7c92ddc75305b22d737f33336c5c8cfd3e1205b5afd895af936f80e962d88b3`。
尾段另見同目錄`TAIL.md`與`tail-report.json`，沒有把完整frame標為PASS。

### Draw360：CB0 sparse DWORD packing

實際event5974先執行的clear已完成，terminal model tail是clear h/c/d：
PS0、PBE2073600、pool766807/766807、零洩漏。接著480個u16indices／
160triangles的draw在編譯前期被budget拒絕，沒有draw API submit或
draw model hello，也沒有成功finish／range。不能把整range說成零GPU工作。
762份original shader signatures／1051份init create-bind records閉合
到VS711／FS712、GLSL224；不是只依可能重用的host pointer判定。

FS綁定CB0108DWORD、8texture descriptors160DWORD、UBO1×4DWORD，
共272>256；VS64+4=68<96。現有prefix trimming仍需CB0高水位105，
所以269仍超限。獨立逐instruction分析FS712的26個load_uniform，
真正source-word union為91：0..11、16..86、88..92、96、100、104。
其中唯一dynamic load是base23/range4，四個可能scalar92/96/100/104
全部保留；不是依本次輸入index或branch刪掉可能使用的資料。
緊密排列後160+4+91=255，保持原FS256限制。

已完成修正：僅舊prefix策略仍超限時，compiler建立升序source DWORD
map，先按原始bound展開既有dynamic SSA，再重寫packed push-constant
byte offsets與4-byte alignment。Map放driver-private owned binary，
stable ABI／API32不增欄；context在enqueue前逐word bit-copy成原本的
flat shared payload，不更動texture／UBO描述符或增加DMA。
此前能fit的程式維持空map與原bytecode。已freeze driver16，Gallium SHA256
`1883b4259943dc3d50b210b590a5e115d4b9639e9a16814b109c5f870a42cc03`；
model15不變。context僅對非空map記錄source/count/shared span及map／
materialized payload FNV，提交與完成證據另行檢查；hash本身不代表執行。

`tests/pvrgpu_push_constant_map_test.c`直接測actual context reader：
334231 normal＋334231 ASan/UBSan checks通過，包括count256、source512、
未對齊user/resource backing、raw特殊bits、生命週期、原完整DWORD補零、
map/count/order/extent錯誤於任何寫入前拒絕。前版333713check證據保留；
加入map counter後的最終freeze亦重跑normal／sanitizer通過，證據WORK
`tmp/push-map-transport-freeze.3D8t7O/receipt.json`；前一版
`tmp/push-map-transport-final.GlkI6u/receipt.json`保留不改寫。
全部17組Mesa resource unit亦通過，artifact `tmp/pvrgpu-mesa-unit.SCWDqT`。
`tests/packed_cb0_live_probe.c`的LP112draw／9335checks通過；同probe在
old driver13/model15完成56fit controls後，在first pressure case拒絕。
新driver16/model15完成112draw／9335checks，獨立oracle核對7168words
完全相同；112組API submit/done與model hello/counter/done完整、零洩漏。
前56fit controls沒有map event，整份model紀錄與old13完全相同。
後56pressure cases的真NIR source184、map14、shared164..177及每次
packed payload hash均符合原輸入（含全部四個dynamic array candidates）。
Root artifact `tmp/packed-cb0-live.yg2pcw/`；獨立證明
`tmp/packed-cb0-live-audit.ZC9sZF/REPORT.md`，certificate SHA256
`0d6e75af97bf2c795ffdc493cc186f94bca59bab5ad6f1ff1125de5fa60fed05`。
不得將old failed完整run稱為PASS，也不能把此fixture當作actual Draw360。

舊shared-budget regression還揭露unlowered UBO deref在新的packing
放行後會到達原本不支援的PCO路徑；compiler已在早期拒絕實際live
UBO load/store_deref，保留合法load_ubo/get_ubo_size、一般IO/local deref
及無害UBO宣告。原32-case shared-budget正常／sanitizer各534checks通過。
新compiler matrix各1425checks通過，其餘frontface146、wideUBO35、
shadow gather219、many textures376均於normal／sanitizer通過。
真正編出的5組PCO由raw/prepared ISS執行：各64109checks、640executions、
7680SMP requests；此ISS unit的texture callback是合成輸入，不作TPU證明。
獨立final review `tmp/sparse-cb0-iss.Jyr1qu/REPORT-v2.md`。

完整原VS711／FS712 source-operation NIR reconstruction亦已編譯：
FS11776bytes、TEMP66／SH255／CF88、push164/91，1077groups完整generic
decode通過；VS1392bytes、SH68、push4/64。原始body／declarations／
precision及FP preserve flags保留，caller未修改；不宣稱與actual shader
binary相同，也不是ISS prepare或執行結果。證據
`tmp/fragment360-reconstruct.ZtVXls/REPORT.md`。正式driver16/model15
ordered replay另存`native-v16-packed16-s32-full6649`，已結束於新的CubeArray
texture gate。Actual360 map index FNV為a0c7a6c41a2f0252，符合原完整union；
packed payload FNV ca8f884b038e0b02僅為觀測（完整CB0未dump）。

同樣有壓力的後續三組完整material sources（361/364、362/363/365/366、
367–369）亦以原printed body／declarations／FP flags重建並真compile：
FS皆map91/source108/SH255、各1077groups完整generic decode通過。
370–394其餘postprocess僅做source／budget盤點，沒有將未執行的shader
標為PASS。證據`tmp/tail-material-preflight.zCQOne/REPORT.md`。

獨立prefix0–160稽核已確認driver16對driver13/model15：165份完整graphics
model reports（僅排除artifact_png路徑）、169graphics API pairs、2compute
pairs一致，162乾淨observer scopes；12份typed raw共58060800bytes及
metadata逐byte相同，零packed-map events。Helper本來的
`native_execution_verified:false`在兩版都保留：frozen helper在line524
固定輸出此literal，非動態檢驗失敗；真正`range_native_work_verified`
另由line177的`audit.native_work`產生。不使用此literal flag判定完成；
另以actual hello/counter/done、API pairs、balanced pools、finish及observer
建立完成證據。artifact `packed16-prefix-audit.KISszq/PREFIX160.md`，
machine certificate仍保留舊schema字串的命名瑕疵，未覆寫原證據。

最終prefix0–359 certificate已完成：360finalized scopes／361乾淨API scopes、
356model reports／364graphics API pairs／2compute pairs；352actual-work
及8zero-indirect scopes。所有model fields（只排除PNG路徑）、42raw共
306892800bytes及metadata相同。4453個host pointer差異明列；353–358另有
6項nonindexed index_bias診斷值差異，保留不當pointer正規化。
`packed16-prefix-audit.KISszq/REPORT.md`，final certificate SHA256
`66f70e77ab2379880329c35bf5cb6d6f670a6d6fd359f33fea9e31a46e633e6f`。

### Draw360：non-shadow CubeArray depth view

Captured Program1855的static_envmaps是logical sampler4→GL unit7，
e5968綁CubeArray texture0、e5969明確sampler0；同程式的shadow2DArray
為logical7→GL unit4。Pinned Mesa把shadow bit按logical index7建立，
卻在texstate fallback選擇用application unit7檢查，指向depth fallback。
初始化確有Z32_UNORM1×1×6／swizzleX/X/X/ONE與RGBA8兩種CubeArray。
但driver16拒絕前未記錄slot4的實際pointer，所以depth來源仍標為source
inference；不可把推論說成已讀到該slot的完整bytes。Draw358無shadow
sampler、CubeArray在unit8，實際使用RGBA8 fallback。

一般noncompare depth CubeArray是正當的取樣路徑。Driver17讓depth-bearing
CubeArray（含Z24S8）沿用現有unpack_z_float／view swizzle→RGBA32F
transport；descriptor依真正16B storage texel重建face stride。Stencil-only、
shadow compare、compressed blocks、非法samples／mips／非整cube仍拒絕。
其他target的Z24S8 raw路徑不改；compiler／model／API32皆不改。
Depth-only新counter記錄source/storage format、swizzle／compare及pointer，
讓下一輪實際capture能閉合來源。沒有修改Mesa、fallback值或原shader。

Driver17 Gallium SHA256
`eaa698618540f0bc82d829c961d80798002f1135493e290fbb7bf3cfc27f2f94`。
`tmp/cube-depth-snapshot.haecG9/receipt.json`的normal與ASan/UBSan各
252656checks通過；6種depth格式×mip/cube base×4swizzles逐DWORD確認，
含source不變、descriptorstride、compare/stencil負例。全17組resource unit
通過於`tmp/pvrgpu-mesa-unit.c8p0dY`。Packed-CB0 live112draw／9335checks
也在driver17再通過（`tmp/packed-cb0-live.yg2pcw/native-v17`）。

`tests/cube_array_depth_live_probe.c`在LP／driver17各288draw、17541checks：
D16/D24/D32F、2cubes×6faces×3mips、nearest／linear、implicit／explicit LOD；
756個source depth texels完整raw讀回相同。Old16在第一個D16 draw明確拒絕，
無draw model submission。新driver17全部實際完成，288組model triplets與
API pairs完整、pool平衡。4608output words中576與LP有1ULP差異，最大
5.96e-8，均符合預先設定的depth oracle絕對容差2e-7；不宣稱bitexact。
最終fixture取face center，不以此證明新seam算法。Artifact
`tmp/cube-array-depth-live.n5TzVH`；先前off-center 1×1-mip oracle錯誤的
嘗試另存`sElKjR`而非宣稱模型錯誤。獨立audit已完成於
`tmp/cube-depth-live-audit.uDuiNi/report.json`，SHA256
`b5aaf6f210c8ac8904cf50e4ee2210d4d7c3ed6d5d3130145d3fd92caa573825`：
整份stdout／known suffix、source texels、288depth snapshot/source-format records、
actual model/API順序皆驗證；2304SMP、5760texel taps、1152visible PS，
native最大input-oracle誤差2.8871e-8。Live LP自然提供3.2、native3.1，
兩者沒有version override；actual capture繼續使用經驗證的3.1 override。
Actual17已完成全frame，直接runtime證據見後面的「driver17完整frame」段落。

同一問題另以合法8-active-sampler GLES fixture直接重現，不修改capture：
logical Cube4→appunit7或8；logical shadow7→appunit4，其餘units皆相異。
真正NIR確認index順序；program各使用Cube tex／txl，其他7samplers皆參與
可觀測輸出。LP兩種unit皆2draw通過；old16在unit7的Z32 fallback拒絕而
unit8的RGBA8完成；new17兩種皆完成、32output words與LP逐bit相同。
unit7直接以resource→view→slot4 counter閉合Z32_UNORM→RGBA32F96B、
compareNONE；unit8保留RGBA8/24B，old16/new17整份model JSONL相同。
這證明pinned Mesa的internal fallback index-space mismatch；不宣稱GL
規定一定要用depth fallback，也未注入fallback像素。
原報告`tmp/cube-array-fallback-live.axwsqe/REPORT.md`；獨立稽核
`tmp/cube-fallback-live-audit.4oVKJN/REPORT.md`，certificate SHA256
`885b8cad382565730650521970bb40c285ce0f15432e42810016b5a9d0319a7f`。
Actual capture slot4已由driver17的新counter直接閉合，詳見下節。

### 尾段 compute：完整原始 GLSL 的小型實跑預檢

`tmp/tail-compute-live.S0c4VY/REPORT.md`：original516/520 GLSL bytes不改，
LP與driver16/model15各2dispatch、192invocations、131checks通過。
這是合成texture／buffer／UBO輸入的獨立測試，不是capture資源替換；
516只跑1個16×8 workgroup，520保留1個64-lane workgroup。
兩dispatch間只有真正shader-storage barrier，沒有CPU map/read/write。
first SSBO的63個未寫DWORD、output.w及1024B UBO皆bit-exact保留，
second CS真正消費first CS寫回的word0。沒有測其後graphics UBO消費。

Native真PCO fingerprint與先前原GLSL compile-only完全相同：
516為10368B/FNV57f68a3b65e0bae5、1536SMP/6144taps；
520為6736B/FNVca74f490f41f71b4、0SMP。兩組compute_api_submit/done
與buffer snapshot/readback完整，pool各5460/5460、1035/1035零洩漏。
此compute API沒有model JSONL，不能宣稱不存在的model triplets。

Numeric oracle tolerance在執行前即固定：sum5、average0.003、exposure0.03。
Native／LP並非bit-exact：sum −1349.76367188／−1346.17639160，
average0.415300726891／0.416271805763，exposure−1.46725571156／
−1.47167968750；差異保留，不以此猜定precision根因或聲稱capture畫面通過。
verification SHA256
`c7af02b030efd2176dafb346d9bf0cf7e3f51c65ccdee58af8a818b88912aeca`。

### driver17完整frame：實際完成與畫面比對分開判讀

正式artifact：WORK
`tmp/carchase-debug.t3s25m/native-v17-depth17-s32-full6649`。
`progress.json`記錄exit0、`timed_out=false`、native audit成功；395個ordered
action ranges執行至Draw394/event6629，再完成trailing6649及最終readback。
固定driver SHA256
`eaa698618540f0bc82d829c961d80798002f1135493e290fbb7bf3cfc27f2f94`，
model15 SHA256
`f2cd648286a2c848fd1976137bdf4f247f9553497171ee6dcaff64281bbac243`。
helper、RenderDoc、capture與參數全部固定於run的`receipt.json`。

仍採compile-only OpenCapture initialization，初始capture資源還原後才啟用
native ordered replay。`initialization_failure_count=772`屬明確分隔的loading
scan，不當成actual replay failure，也不把這次說成完整native initial-copy認證。
Helper既有`range_returned.native_execution_verified=false`為literal欄位；
真正每range model/API/finish證據須看獨立audit，不因工具exit0就放寬標準。

獨立完整structural audit `depth17-prefix-audit.stkCuU/full6649.json` SHA256
`b302821bdb316c967f3840822b71394d5f44c78ea529d7ca36d46e3f038f2e5e`：
395 ordered draw ranges＋6630–6649 trailing range，397個乾淨同步API scopes；
387個有graphics工作的range、8個經decode確認的zero-work indirect。
391組model header/counter/done、400個physical graphics submissions、
413組graphics API pairs，5組compute API pairs（Draw84×2、370×2、383×1）；
所有model／compute pool balance通過，391個`@CAPTURE` markers，兩stderr為空。
395個helper literal false與772個initial-scan failure仍明確保留，未清除或改寫。

獨立prefix audit在`tmp/carchase-debug.t3s25m/depth17-prefix-audit.stkCuU`：
0–359與driver16的全部model內容（除輸出path）及42份raw、306892800bytes
不變，metadata同樣符合。`prefix359.json` SHA256
`e57da1eba00aa1c9f95545a7f7fcfbf21b3eb15538b378097c1a685e538350ca`。
`draw360.json` SHA256
`9abe397cf1d6f8423182e5a4cd5ed95f6be6a3d054714903d86f26c85a27cd2d`。

Actual360 slot4的source view/resource直接記錄Z32_UNORM24B、1×1×6faces、
RRR1 swizzle、compareNONE，轉為RGBA32F96B；不是對capture資源注入值。
CB0 map108→91/shared255/index-FNV`a0c7a6c41a2f0252`、payload-FNV
`ca8f884b038e0b02`。VS711/FS712原source identity閉合。Draw360有IA480、
VS306、setup160，但24093個fragment candidates全部depth rejected，故PS0、
SMP0；360–369全部都是有native graphics工作的draw，但PS0。
不可宣稱actual360執行了新的FS texture path；live288draw與unit7/8 fallback
fixture則有真正SMP執行與像素oracle。後面不同draw使用不同shader，不能統稱FS712。

Draw370範圍內兩次compute真正完成：grid8×8/block16×8為8192invocations，
98304texture requests／393216texel taps、256B readback；第二次grid1/block64
為64invocations、16B readback。兩pool各348936／348936與1035／1035。
第二次compute後的fragment UBO直接記錄4DWORD
`3d33afbc 3d33afbc 405319bf 3f800000`。
Compute actual-submit只有binary size／counter，PCO fingerprints只記錄於
compile-only初始化；不可把size對應誤稱為完整pointer-level source identity證明。
Draw370本身另有518400PS／PBE fragment writes，519840texture requests。
`tail370.json` SHA256
`70b80e57b8461b3f16f1ca6c7dcea48d8885710d274e5a34c7a697de3c9d6190`。

Draw383範圍內的第三個尾段compute也真正完成：binary568B、grid1/block128，
128invocations／128loads，0stores／0atomics／0texture requests，4B readback，
pool141／141。Captured Program547→shader548、UseProgram6563／Dispatch6570
與後續同buffer的UBO3綁定對上；VS／FS直接讀到DWORD0。這份capture的此路徑
並未走到texture sampling或atomic increment，不能用完成dispatch來宣稱這兩種
指令已受此capture測到。PCO fingerprint同樣只有initialization記錄。

最終nativePNG為run內`draw-394-event-6649-color0.png`，1920×1080。
llvmpipe reference仍為`tmp/carchase-reference.morh1s/strict-packed10-full6649`。
兩者B5G6R5 attachment的raw都是`GL_RGBA/UNSIGNED_BYTE` canonical RGBA8
讀回，**不是packed565 bytes**；row_stride7680、bottom-left、8294400B。
Native raw SHA256
`fed53ff416f52ca89977c858f46a3953c69f1d92041c8b7d6d5a2a53e0397d2f`；
native PNG SHA256
`c4d7183218bb0e539d78b62790d9c69fc76110b72e5571681b47161f36b1d8ef`；
LP raw SHA256
`2d50e3efb5f9e0ea0a96cf57afe337e12b0eb011a71b4df6224315631aaf0eac`。

比對script與JSON存於`tmp/carchase-final17-compare.wFDqSz`，44份common
snapshots先核對subresource/format/readback metadata與size，共323481600B／backend，
5份raw相同；不是全部395draw都有snapshot的宣稱。最終兩邊PNG解碼皆與
各自raw垂直翻轉完全相等，alpha皆255；RGBA compare得到：

- 差異像素1441954／2073600（69.5386767%）；不是bit-exact。
- RGB平均絕對差2.20541281／255，RMSE4.00254512／255，最大214／255。
- 409個像素的任一RGB差>16，213個>32，104個>64，16個>128。
- 最大差例子，左上原點(x325,y435)：native(132,190,222,255)，
  LP(8,8,8,255)。大差異不能用平均值、格式量化或未知precision原因直接豁免。

`comparison.json` SHA256
`efc90e5ad37a201cf9621a4ae1ab4b67ce37e22a8eed3a3d09349bdf7e36905c`。
另一agent以獨立Pillow decoder、integer error sums與92個root input hashes
重算，4580checks通過；沒有重新覆寫root comparison。五份raw-exact附件皆在
Draw160，不表示其他draw相同。獨立報告`tmp/carchase-final17-image-audit.a3r0k5/REPORT.md`，
receipt SHA256
`d09062388c516a21a442b2fc0addc801f710e6199cea357ec7fe2f205d8c87fc`。
收尾再跑public driver-tree 16項與snapshot/replay CLI 20項全部通過，
`git diff --check`乾淨；這不替代前述compiler、transport、真GLES與actual replay測試。

目視兩邊最終frame都有天空、地形、岩石與植被，皆未見車輛；這只描述此次
capture的兩份replay輸出，不宣稱與原裝置capture framebuffer已逐pixel吻合。
功能阻擋已清除、frame已畫出，後續仍需依序追查第一個可重現的數值／coverage差異。
此為debug／同步稽核run，不當作無debug最佳性能量測。

### 正規 snapshot／replay 的實際使用限制

Repo 已有 `docs/DRAWLIST_SNAPSHOT_REPLAY.md` 所述的正規 whole-state snapshot；
上面的 bounded launcher 不是該入口。這次補上 `run_drawlist_replay.py`
`--gles-version {3.1,3.2}`（預設3.1），runtime identity v2 固定該選項；
舊 v1 只可在3.1接續。
20項 CLI／manifest process-contract tests 通過，並非 Car Chase checkpoint 通過。

獨立查核找到既有 Manhattan native147→148 的真實、相容 checkpoint，沒有
Car Chase checkpoint。兩者不能跨 capture 使用。目前 snapshot RenderDoc 未含
本次 patch-parameter alias 修正；alias-fixed library 也沒有 snapshot exports。
此外 Car Chase 在 Draw155 邊界仍有960×960 D16 renderbuffer；snapshot V1
及其 underlying RenderDoc codec 都尚未支援 lossless renderbuffer save／restore。
不能移除 guard、略過 renderbuffer 或只存 PNG 冒充 checkpoint。

完整 compatibility 調查在 WORK `tmp/carchase-snapshot-audit.MtxADU/REPORT.md`。
在相容 library 與全資源 codec 驗證前，保留 bounded 診斷。建立可用 checkpoint
後只接續 native 自己的 state；更換 driver 仍須重新建立 prefix，不能用
`--allow-model-change` 放寬 driver identity。

後續完整395 draw endpoints／10946 chunks唯讀稽核：唯一RB437在event1908
被invalidate後雖不再使用，仍註冊並附著於FBO436，直到tail6649皆在V1全資源
inventory內，因此不存在只靠改存到較晚boundary就能避開的機會。
最小通用修正需D16 single-sample RB signature、raw capture／preflight／restore
及FBO texture-or-renderbuffer reference驗證；不能只增加raw讀回或把RB冒充
texture。需真Save/Load raw roundtrip、依賴下一draw、不良資料拒絕與沒有額外
native draw的證明。保持本次驗證過的GLES3.1，不因capture header改回3.2。
完整hook與測試清單：WORK `tmp/carchase-snapshot-boundary.v6JbCL/REPORT.md`。

D16 transport可行性已用真正GL證明：新
`tests/d16_renderbuffer_transport_live_probe.c`在LP及native driver8/model8b各
789788checks／2draws通過。256²及257²兩種配置包含全部65536個D16值；
保存自己RB的raw、確實覆寫，再用相同D16 staging texture depth-only blit還原，
透過兩個原有FBO aliases逐word讀回一致。dirty PBO／pack／unpack／FBO／
scissor／ANGLE reverse-row狀態還原，下一個GL_LESS depth draw也符合整數oracle。
保存／還原範圍沒有draw、PCO編譯或SystemC提交；首次blit建立2個未bind的
dormant NIR objects，不能宣稱零CPU compiler活動。此測試未整合snapshot codec，
也沒有CarChase whole-state checkpoint。證明WORK
`tmp/d16-rb-transport.vyV99b/verified-transport.json` SHA256
`8db3b91badad68f51d9fe89318f74b811dcd0c110d597ea876faf66d40e63d3d`。
