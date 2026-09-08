# GLES3 Transform Feedback 驗證

最終 native_v2 完整執行 stock `dEQP-GLES3.functional.transform_feedback.*` 的 1320 個案例：1212 Pass、108 NotSupported、0 Fail、0 NoResult，與 llvmpipe 的每個案例狀態完全相同。Tessellation 的 Transform Feedback 問題另行處理，以下結果不代表 Tessellation 通過。

## Stock CTS 範圍與結果

CTS：`opengl-cts-4.6.8.1-0-g067e8832315e79817ede1c4863804e440f5d1c80`。由 GLES3 全部 48300 個案例發現並選出 1320 個精確名稱；執行時每個 process 僅接收一個精確案例。案例清單 SHA-256：`3d6038213387699099213c5a1cdef5159db371a5b2e90945be6e5b560553ad98`。

| 子群組 | 案例數 | native_v2 Pass | native_v2 NotSupported |
| --- | ---: | ---: | ---: |
| position | 6 | 6 | 0 |
| point_size | 6 | 6 | 0 |
| basic_types | 378 | 378 | 0 |
| array | 378 | 378 | 0 |
| array_element | 378 | 270 | 108 |
| interpolation | 54 | 54 | 0 |
| random | 60 | 60 | 0 |
| random_full_array_capture | 60 | 60 | 0 |
| 合計 | 1320 | 1212 | 108 |

| 執行 | Pass | Fail | NoResult | NotSupported |
| --- | ---: | ---: | ---: | ---: |
| PvrGPU baseline | 0 | 1102 | 110 | 108 |
| llvmpipe baseline | 1212 | 0 | 0 | 108 |
| PvrGPU native_v1 | 1212 | 0 | 0 | 108 |
| PvrGPU native_v2（最終） | 1212 | 0 | 0 | 108 |

Baseline 於 2026-09-08 09:36:45–09:41:00（Asia/Taipei）執行；native_v1 於同日 09:54:43–10:06:20 執行。上述每次執行的 CTS、Mesa library 與 SystemC bridge SHA-256 在執行前後一致。native_v1 與 llvmpipe 的 1320 個逐案例狀態完全相同；1212 個 Pass 的 123624 筆 model JSON records 中，unsupported events、driver/model errors、pool leaks、Fatal lines 均為零；所有 process exit code 為零且無 timeout。108 個 NotSupported 為 array_element matrix 超過實作的 vertex-attribute 上限，CTS 在 link 失敗後檢查宣告的 implementation limits；不是跳過或 waiver。

最終 native_v2 使用正式 runner 於 2026-09-08 10:13:54–10:25:29（Asia/Taipei）重跑，process exit code 0、runtime hashes 前後相同。1320 個精確案例逐一對照 llvmpipe，mismatches 為空；1212 個 Pass 合計 123624 筆 model JSON records，process failures、unsupported events、driver/model errors、Fatal lines 與 pool leaks 全部為零。

本次全組紀錄位於外部 work root 的 `out/runs/deqp_groups/transform_feedback_v1`：最終結果在 `pvrgpu-native_v2/runtime-receipt.json`、`summary.tsv`、`integrity-summary.json`，逐案例對照在 `comparison-native_v2.json`；每案例另存 QPA、driver events 與 SystemC JSONL。

最終 runtime SHA-256：

- CTS：`641797fa7c04e4ee0bb5115a4dcdf57f8e5098f478cfcb60a71055141d7e6961`。
- Mesa libgallium：`33014caffdc535f69572eb57907d77ceae49082ba8e78c1034a628c577998d02`。
- SystemC bridge：`2f28a8511590426ad7e70ea23c59f3cf325b0a2af9392106478af40e8b60300c`。
- 正式 runner：`cdef4c1fc5201cd4b89ceedefe36978e6f3ae36fb10e3034fa31ab7943e19a96`。

## 修復與實際語意

原先 TF target 僅被記錄，TF buffer map 時 draw 尚未提交，`GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN` 固定為零。現在由獨立 SystemC `StreamOutput` module 讀取原生 PCO shader 的 VTXOUT，按完整 primitive 預先檢查全部 target 範圍，再寫入 modeled GPU memory；driver 依 generation/resource token 回讀 buffer 並維持 pause/resume append cursor。written 與 storage-needed 使用獨立計數，overflow 不會留下半個 primitive。

Compiler 保留 TF-only varying，並提供明確的 packed output/coefficient/interpolation mapping。移除 ABI 分配後的第二次 `pco_link_nir`，避免再次 compact varying 而使既有 output DWORD 範圍失效。VS flat output 使用原始 provoking vertex 的 raw bits，經 clipping、line/point expansion 與 winding normalization 後仍保留整數及 NaN payload。Centroid FITRP.PIXEL 按原生指令語意執行。

每個可執行 CTS 案例最多包含 10 個 iteration：元素數 1/2/3/4/123、pause/resume、開始即 paused、多 draw append、buffer guard，以及 TF 開啟/關閉時的 rendering 比較。例 `position.points_separate` 最後一輪 active draw 元素數 217+171+147+55 得到 query 590，paused draw 的 152 個元素不計入；buffer 回寫與 TF 開啟/關閉 rendering 比較均通過。

本輪沒有 host shader evaluator、expected-result 注入或 CTS case-name 判斷；模型處理的 shader 輸出來自原生 PCO 執行。

目前支援範圍為 VS、stream 0 的 Transform Feedback；GS/TES feedback 仍保留明確拒絕閘門，`rasterizer_discard` 尚未支援。Centroid FITRP.PIXEL 目前涵蓋 single-sample rasterization；raster sample count 大於 1 的 centroid 指令仍明確拒絕。上述 1320 案例的全組結果不應外推到這些能力。

## Regression

- 最終 runtime 安裝且全組重跑後的完整 SystemC CTest：117/117 PASS，14.82 秒，`ctest-final.log`；前一輪 `ctest-native-v2.log` 亦為 117/117 PASS。
- StreamOutput module：direct/bypass/cache 各 10066 checks PASS。
- StreamOutput API boundary 與原生 capture：direct/bypass/cache 各 58 checks PASS。
- Graphics statistics：33 checks PASS，含 written/storage-needed 獨立累計與 transactional overflow。
- Decoder：VTXOUT 63/64 的 FIFO 路徑接受，65 明確拒絕；valid 50 checks、reject-65 13 checks、reject-65-point-size 13 checks PASS。
- Geometry frontend：288 checks PASS，涵蓋 VS/GS flat raw DWORD、原始 provoking vertex 被裁切、points/lines/triangles、雙 winding，以及 smooth NaN 的 clip-stage 傳遞。
- 真實 EGL/GLES 3.0 instancing probe：LINE_STRIP、LINE_LOOP、TRIANGLE_STRIP、TRIANGLE_FAN 各 2 instances，分別測 non-indexed 與 uint8 indexed，共 8 案例；PvrGPU 與 llvmpipe 均 8/8 PASS。每案完整 512-byte TF buffer（含未寫入尾端 sentinel），合計 4096 bytes / 1024 DWORD 與 8 個 primitive query 的觀察結果完全相同；`observed.txt` SHA-256 為 `4d0907c5cc1c4d130a8f89a6a0e64a6a936bd4d9bce7d08c3763eee2a6ddecd1`。source、編譯/執行參數和 runtime hashes 保存於 `instanced_live_v3/VALIDATION.md`。
- Driver query/topology unit：PASS，含提交失敗時 query 不得偽報成功零值、全新空 query interval、四種 connected topology 的 non-indexed/uint8/uint16/uint32、instance 邊界、strip winding、fan anchor、loop closure、index upcast、bounds 與乘法 overflow。
- PCO lowering unit：PASS，含 TF sparse array、scalar、PointSize physical output mapping 與 mixed flat/smooth explicit linkage。

## 重跑

使用 `script/run_deqp_transform_feedback.py`。`--work-root` 預設為 `PVRGPU_WORK_ROOT`，未設定時使用目前使用者的 `~/Downloads/_Codex/Working/PvrGPU`；不含固定帳號路徑。可用 `--output-root`、`--cts-binary`、`--bridge`、`--pvrgpu-prefix`、`--llvmpipe-prefix` 指定實際建置位置。

```sh
tf_work="${PVRGPU_WORK_ROOT:-$HOME/Downloads/_Codex/Working/PvrGPU}"
python3 script/run_deqp_transform_feedback.py \
  --work-root "$tf_work" \
  --output-root "$tf_work/out/runs/deqp_groups/transform_feedback" \
  --backends pvrgpu llvmpipe --output-label baseline --workers 3
```

已有固定 llvmpipe 基準時，只重跑 PvrGPU 並逐案例比較狀態：

```sh
tf_work="${PVRGPU_WORK_ROOT:-$HOME/Downloads/_Codex/Working/PvrGPU}"
python3 script/run_deqp_transform_feedback.py \
  --work-root "$tf_work" \
  --output-root "$tf_work/out/runs/deqp_groups/transform_feedback_v1" \
  --reference-run "$tf_work/out/runs/deqp_groups/transform_feedback_v1/llvmpipe-baseline" \
  --backends pvrgpu --output-label recheck_01 --workers 3
```

`--prepare-only` 僅做 stock discovery 與 1320 案例清單驗證。`--case-filter` 可縮小診斷案例，但實際執行仍使用精確名稱。`--debug-pco-nir` 額外保存原生 NIR/PCO 診斷。

每個 output label 建立新的結果目錄，不覆寫舊結果。Runner 保存完整 command/environment、runtime hashes 與 script hash；發現 CTS Fail/NoResult、執行中 runtime 變更、Pass 案例的 unsupported/model-error evidence 或 backend 狀態不一致時回傳非零。`integrity-summary.json` 與 `comparison-<label>.json` 分別保存模型證據檢查和逐案例 backend 對照。
