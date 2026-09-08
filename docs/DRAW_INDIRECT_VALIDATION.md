# dEQP Draw-indirect 驗證：2026-09-08

範圍是 stock GLES31 `functional.draw_indirect.*` 221 項與
`stress.draw_indirect.*` 23 項，共 244 項；不含不同功能的 indirect compute
dispatch。完整執行使用固定 pbuffer 256×256、`rgba8888d24s8ms0`、cache-mode
SystemC 與停用 shader cache。未修改 CTS、比較 tolerance 或能力宣告。

## 最終結果（修正版）

| 完整群組 | 案例數 | Pass | Fail | NotSupported | Warning / NoResult |
| --- | ---: | ---: | ---: | ---: | ---: |
| `functional.draw_indirect.*` | 221 | 203 | 0 | 18 | 0 |
| `stress.draw_indirect.*` | 23 | 23 | 0 | 0 | 0 |
| 合計 | 244 | 226 | 0 | 18 | 0 |

同一個 frozen candidate runtime 於臺灣時間 2026-09-08 13:49–14:24 完整
執行，沒有 timeout、未驗證的 Pass、unsupported draw 或模型錯誤；
runtime SHA 前後相同。llvmpipe 同一 stock CTS 基線亦為 226 Pass /
18 NotSupported。18 個能力跳過均是 CTS 檢查
`GL_MAX_SHADER_STORAGE_BLOCK_SIZE is too small for vertex attrib buffers`。
這是測項狀態一致，不是 244 項全 Pass，也不是全部畫面逐位元等同 llvmpipe。

獨立完整稽核將 226 Pass 分為 **202 個原生影像測試、11 個原生 no-crash
測試、13 個預期 GL error 測試**；不把 API/negative 當成 shader 執行。
另有 3,373 項逐案交叉檢查通過：244 案狀態與 18 案 NS 原因均與 llvmpipe
相同，213 個原生案例實際載入同一 bridge；81 次 Compute、24,795 次
graphics submit/done 配對，model/Compute memory-pool ownership 完全平衡。

修正前完整基線是 204 Pass / 19 Fail / 18 NotSupported /
1 CompatibilityWarning / 2 NoResult（600 秒 watchdog）。19 Fail、
1 Warning、2 watchdog 均轉 Pass；另外 3 個原本含拒絕 draw 的 raw Pass
現在也通過原生稽核。前後逐案差異稽核 6,546 checks 全通過，18 NS 不變。

## 修正與邊界

1. **Signed baseVertex**：先複製真正 EBO，按原始 unsigned index 處理既有
   POINTS restart，再對 referenced min/max range 做 rebase。以 int64 驗證
   effective source index；保留索引寬度、順序、reuse 與 instance 邊界。
   不再直接拒絕合法負值，也不讀取未引用的低位頂點。`gl_VertexID`／相關
   system-value ABI 尚未支援，仍明確拒絕，不能把 rebased index 當成原值。
2. **Constant/default attribute**：保留 Gallium 的真正零 stride。
   `glVertexAttrib*` current/default 值不隨頂點遞增；GL API 的緊密排列
   stride-zero shorthand 早已由 Mesa state tracker 正規化，不受影響。
3. **有界 EBO fetch**：採 llvmpipe `DRAW_GET_IDX` 的處理方式，超出完整
   元素範圍時提供 raw index 0，再執行 restart/baseVertex 與原生 pipeline。
   不縮短 draw count、不把整個 draw 改成 no-op，也不讀取部分尾端元素。
   缺少 backing、非法型別或非法 effective source range 仍失敗。

三項修正都只處理命令／索引／attribute 輸入；shader、Transform Feedback、
query 與 framebuffer 仍來自真正 SystemC 原生執行。ComputeShader、
GeometryShader、Tessellation 的獨立 SystemC 模組及 API 均未變更。
multi-draw／indirect-count、indirect compute 等既有能力邊界不因此擴張。

修正前已確認的 19 個 Fail 包括 14 個 default-attribute、3 個明確負
baseVertex 與 random.30／39；stress random.33 的 CompatibilityWarning
也是負 baseVertex。另有 3 個 raw Pass 含拒絕 draw：兩個 stress EBO
越界與 functional random.40。修正版以上案例均原生執行、零 unsupported。
random.30／39 的格式轉換由 Mesa u_vbuf 先解碼真正 indirect buffer，
driver 接收的仍是 signed base -2；不能因 PvrGPU decoder event 為零就
推論未執行 indirect API。這兩例原生 VS 分別為 2／4，random.40 為 2。

llvmpipe 對照基底為本機 pinned Mesa
`da14d65e4499e66468094be52bff9ea0915a695e` 的 draw module：
`draw_private.h` 的 `DRAW_GET_IDX`、`draw_context.c` 的 `draw_set_indexes`，
以及 `draw_pt_vsplit_tmp.h` 的有界 slow path。零 stride 對照
`state_tracker/st_atom_array.cpp` 的 `st_setup_current`。

## Pass 的意義

- Stress 12 個 `unaligned_data.random.*` 有 reference-renderer 像素比較；
  compatibility warning 也必須另外列出，不能算成 Pass。
- Stress 11 個 invalid-input 案例只在 draw + finish 後通過；沒有像素
  oracle，甚至不檢查 draw 後 GL error。這些案例額外核對原生 VS、IA、
  driver/model completion，但仍只宣稱 stock no-crash 測試通過。
- 測試沒有開啟 robust context；越界資料的畫面不是跨實作的規範 oracle。
  採用 llvmpipe 的有界策略，不等於取得完整 robustness conformance。
- Functional 的 13 個 negative 案例是預期 GL error 的 API 驗證，不要求
  shader 執行。其餘案例有 reference-image 或黃綠覆蓋率比較。
- 不把 NotSupported 算成 Pass，不將 QPA Pass 自動等同原生工作已執行。
  JSONL counter 值位於 `record.counters`；稽核也檢查 schema、submit/done、
  memory-pool ownership、driver/model errors 與實際 renderer。

修正前 runner 曾將 `native_vs_invocations` 摘要讀自錯誤 JSON 層級，導致
該欄一律為零；原始 QPA、driver/model logs 沒有更動。獨立稽核直接重讀
nested counters，不能把舊摘要的零值當成原生執行情況。新版 runner 已修正。

## 獨立原生回歸

`tests/indirect_draw_vertex_fetch_probe.c` 的 88 個情境涵蓋 u8/u16/u32、
合法負／正 baseVertex 的等效索引、DrawArrays、disabled default／自訂
attribute、明確 stride=0，以及 EBO 部分／全部越界。以完整 VS position、
attribute、instance ID 的 guarded TF bytes 與 query 驗證。

- 新 probe：14,080 DWORD、88 queries，PvrGPU 與 llvmpipe 精確相同。
- 既有 CPU／TF／Compute 命令 producer probe：2,560 DWORD、20 queries
  精確相同。合計 108 個情境、16,640 DWORD、108 queries。
- 原始 stdout 只容許既知 SystemC 正常停模訊息；完整 QUERY／WORD
  transcript 逐行比較，保留 raw diff。未忽略任何資料差異。
- ASan/UBSan：index-fetch 1,488 checks、vertex rebase 274,708 checks、
  實際 Gallium attribute/index reader 18,514 checks 全通過。
- 中央 CTest 130/130；Python 83/83。
- 相同 candidate runtime 完整回歸 Geometry207：197 Pass / 10 NotSupported，
  Tessellation406：400 Pass / 6 NotSupported；均無 Fail、NoResult 或 timeout。
  Geometry 獨立稽核 8,549 checks；Tessellation 的 400 Pass 分為 311 個
  native TES TF、47 個 native tessellation render 與 42 個 API/negative。
  模型 ownership 平衡，逐案核對真正載入的 candidate Mesa 與 bridge。

讀取 `config/local.env` 的路徑設定後，可重現測試：

```sh
bash script/run_mesa_resource_unit.sh vertex
bash script/run_mesa_indirect_draw_probe.sh \
  --pvrgpu-prefix /external/draw_indirect_v1/candidate-prefix \
  --output /external/new-indirect-probe
ctest --test-dir "$PVRGPU_BUILD_DIR" \
  -R '^(index-fetch|vertex-fetch|indirect-draw|point-restart)-unit$' \
  --output-on-failure
```

Live probe 入口亦接受 `--pvrgpu-prefix`、`--llvmpipe-prefix`、`--bridge`、
`--cc`、`--timeout`；只編譯 probe，不 build/install Mesa 或 SystemC。
必須選擇 repo 外的新輸出目錄，不覆寫舊證據。
上面的 `/external/draw_indirect_v1/candidate-prefix` 是 placeholder；
本輪實際 candidate 位於下列 artifacts 根目錄。若未指定 prefix，入口
改用本機設定；執行前應確認其 runtime SHA 與欲重現的版本相同。

## 大型分段 draw 的逾時修復

`drawelements_separate_grid_1000x1000_drawcount_5000` 在舊版 600 秒
watchdog 時只有 1,970 次 draw 完成，第 1,971 次已提交但尚未完成；
修正版 392.427 秒完成全部 5,000 次，低於原本的 600 秒門檻。

兩版的兩個 Compute producer 完成計數逐欄一致；共同前 1,970 個完成
draw 的 native IA/VS/attribute-fetch/readback 計數逐 draw 相同，前
1,971 個 decoded commands 亦相同。修正版全案完成 600 萬 IA vertices、
200 萬 IA primitives、401 萬 native VS invocations，沒有 skip 或改 count。

共同 1,970 次 draw 的 **host packed bytes** 從 25,001,443,360 降至
151,485,120；這是 driver 輸入打包量，不是 GPU memory traffic。
兩輪 workers／watchdog 設定不同，elapsed time 不宣稱為控制變因 benchmark。
完整欄位、來源 SHA 與比較方法保存於 `performance-note.json`／`.md`。

## 執行證據

修正前、修正版完整 Draw-indirect 與 candidate graphics 回歸皆已完成。

Artifacts 根目錄：
`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/out/runs/deqp_groups/draw_indirect_v1/`

- `pvrgpu-baseline/`：修正前完整基線，保留原始失敗與 receipt。
- `pvrgpu-baseline-audit-final/baseline-delta.{json,md}`：完整前後差異；
  `baseline-prefix/` 保存安裝修正版前的舊 runtime。
- `llvmpipe-baseline/`：244 項，226 Pass / 18 NotSupported / 0 Fail。
- `pvrgpu-native-v1/`：隔離 candidate runtime 的完整重跑。
- `pvrgpu-native-v1-audit-final/`：獨立完整清單、schema、原生工作與 oracle
  分類稽核，重讀原始 logs，不修改 case 結果。
- `native_probe_candidate_v2/`：108 情境的兩後端原生逐字比較。
- `native_probe_repo_candidate_v3/`：repo 內可重現入口的最終驗證，含
  嚴格 driver/model schema、實際 loader record、TF/query transcript 比對。
- `graphics-regression/`：Geometry207 與 Tessellation406 的完整回歸、
  `geometry-audit.json`／`tessellation-audit.json` 的獨立原生稽核。

| Runtime | SHA-256 |
| --- | --- |
| stock CTS | `9ac9ca1efae4c5194d59663e22af6c455575c1a7b795bd0024b89fc083ab128d` |
| 修正前 PvrGPU Mesa | `bc0f893d46f7df83d54bec97a295d90aae091a2195b42cab11365be3d73bbc48` |
| Candidate PvrGPU Mesa | `af5bbe7da91c2a7214a250dc86a13d2f231b24ad8034cef215152cd47131c7ac` |
| 安裝後 PvrGPU Mesa | `c6cb3dc5e4a3a860e866b2b194b5364b9a1747f85b8a6ceb4d4712f0f5348881` |
| SystemC bridge（未變） | `1e3a1106af02305ae5010317271cc3b9bf2bb6dc123fe4154a84bfa915c46566` |
| llvmpipe Mesa | `6f931b70f03c30e3e65fb41f48332a80e2accc686ceee6b700f8138979fd2463` |

修正前 runtime 與 candidate 位於不同 prefix，測試前後核對 SHA。
各 CTS case 保存 command、QPA、driver counters、SystemC JSONL 與 DYLD
載入路徑；不得將不同 runtime 的子集合拼成一個完整組結果。

完整 CTS 的 hash 是 candidate `af5bbe7d…`。標準 Meson 安裝後改寫
Mach-O `LC_ID_DYLIB` 並移除兩個建置用 `LC_RPATH`，得到 `c6cb3dc5…`；
`install-binary-equivalence-v2.json` 證明所有其他 load commands、所有
程式／資料 sections 及 load-command 區域之後的 bytes 完全相同，
動態依賴仍是相同的 absolute paths。安裝後預設路徑另跑完整 108 個
native probe 情境，仍為 16,640 DWORD／108 queries 精確相同，保存於
`native_probe_installed_final/`；不將此 smoke 稱為另一輪完整 244 項 CTS。
此外 `pvrgpu-installed-smoke/` 以預設安裝路徑重跑原來 19 Fail、1 Warning
與 3 個未驗證 Pass 對應的 23 個案例，全部 Pass、零 unsupported／錯誤，
runtime 前後不變。所有完整組結果仍只引用其原來的 frozen runtime。
