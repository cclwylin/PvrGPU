# RDC Counter Pass/Fail UI

這是一個獨立的 Pass/Fail 工具，不會把 Mesa 或 SystemC 載入 UI
process。它會掃描指定目錄下的 `.rdc`，依序以獨立 process 執行
llvmpipe Golden 與 PvrGPU，將兩邊結果正規化成固定 17 欄 counter
text，比較解碼後的 PNG，完成後產生 `report.md`。

PNG 是 PASS gate。當 backend 宣告該 selected replay 有 framebuffer color
output 時，兩邊 PNG 必須都存在、可解碼、尺寸相同，且 RGBA8 每個
pixel/channel 完全相同；只有具備明確 no-color evidence 的 case 才能將
PNG 標為 `SKIP`。

## 啟動圖形介面

先用 CMake 建置兩個 native backend：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --target llvmpipe pvrgpu
```

目前的圖形介面仍是保留的 PySide6 entry point。建立 Python runtime 後
直接啟動，不經過 shell wrapper：

```bash
python3 -m venv "$HOME/Downloads/_Codex/Working/PvrGPU/venv"
"$HOME/Downloads/_Codex/Working/PvrGPU/venv/bin/python" \
  -m pip install -r requirements-ui.txt
PVRGPU_BUILD_DIR="$PWD/build" \
  "$HOME/Downloads/_Codex/Working/PvrGPU/venv/bin/python" \
  tools/rdc_counter_ui.py
```

如果未設定 `PVRGPU_UI_VENV`，checked-in default 是
`$HOME/Downloads/_Codex/Working/PvrGPU/venv`。安裝完成後也可以直接執行；
若 PATH 上的 `python3` 沒有 PySide6，entry point 會自動切到該 venv：

```bash
cd tools
./rdc_counter_ui.py
```

需要指定其他 interpreter 時，設定 `PVRGPU_UI_PYTHON`。

macOS/Linux 的 backend 沒有 `.exe` suffix。Windows 由 CMake 自動產生
`llvmpipe.exe` 與 `pvrgpu.exe`；文件中的 `build/bin/...` runner 路徑在
Windows 請加上 `.exe`，不應在 macOS/Linux 強制改名。

在 UI 中選擇：

- RDC directory：要遞迴掃描 `.rdc` 的輸入目錄；掃描不跟隨 directory symlink。
- Output root：每次執行的 timestamped result directory 要放置的父目錄。
- llvmpipe runner：worker 預設使用 `$PVRGPU_BUILD_DIR/bin/llvmpipe`，也就是 `RDC -> RenderDoc player -> Mesa/Gallium llvmpipe`。
- PvrGPU runner：UI 預設使用 `$PVRGPU_BUILD_DIR/bin/pvrgpu`，也就是 `RDC -> RenderDoc player -> Mesa/Gallium pvrgpu -> PvrGPU SystemC`。

開始後，UI 會顯示目前檔案、Golden/PvrGPU/counter/PNG 階段及累計
PASS/FAIL。取消會終止目前 runner，尚未執行的 RDC 也會在最終報告中
標示為未執行／取消。完成後可由 UI 開啟該次執行的 `report.md`。
`Quit` 會關閉視窗；若 worker 還在執行，會先取消目前 replay 再退出。

結果表格預設使用 `Fail` filter，只顯示最終結果為 FAIL 的 RDC，方便直接 debug。可切換成 `Pass` 只看通過項目，或 `All` 顯示整批掃描到的所有 RDC；切換 filter 不會重新執行測試。

Golden artifact 仍會依 RDC SHA-256 保存於 output root，但預設每次都重新
執行 llvmpipe，避免 Mesa/player 修正後誤用舊 counter 或 PNG。只有在確認
llvmpipe executable、Mesa runtime 與 replay policy 完全未變時，CLI 才可
明確加上 `--reuse-golden-cache`。cache hit 時報告的 Golden 欄會顯示
`CACHED`，case artifact 也會留下 `golden/cache-hit.txt`；UI 預設不啟用
這個 opt-in。

## Native Qt UI follow-up

完整 native Qt/C++ UI 是下一階段 target，尚未在目前 CMake build 中宣告；
它需要正式 Qt 6 C++ SDK，不能只依賴 PySide6 wheel。該 UI 仍應保持
thin process orchestrator，並補上 recursive/non-recursive 切換、per-backend
timeout、單一 case 重跑，以及開啟 case directory/counter diff/PNG 等
artifact 的操作。這些功能不得把 Mesa 或 SystemC 移入 UI process。

## 命令列執行

```bash
python3 tools/rdc_counter_report.py
```

不帶參數時，輸入目錄預設為
`${PVRGPU_RDC_ROOT}`；若未設定該環境變數，則使用
`~/Downloads/_Codex/Working/GPU_TestPatterns/1.GLBench`。也可明確覆寫：

```bash
python3 tools/rdc_counter_report.py \
  --rdc-dir "/path/to/rdc-directory" \
  --output-root "/path/to/result-root" \
  --golden-runner build/bin/llvmpipe \
  --pvrgpu-runner build/bin/pvrgpu
```

`--rdc-dir` 與 `--output-root` 都可省略；output root 預設為：

```text
${PVRGPU_WORK_ROOT}/out/rdc-counter-report
```

每次執行都會在 output root 下建立新的 `rdc-counter-<UTC timestamp>-<pid>/`，不會覆寫先前報告。

需要讓另一個程式（例如 UI）讀取即時進度時，加上 `--json`：

```bash
python3 tools/rdc_counter_report.py \
  --rdc-dir "/path/to/rdc-directory" \
  --output-root "/path/to/result-root" \
  --golden-runner build/bin/llvmpipe \
  --pvrgpu-runner build/bin/pvrgpu \
  --json
```

此時 stdout 是一行一筆的 JSONL event。runner 的 stdout/stderr 會保存在各 RDC 的 artifact 目錄，不會混入 event stream。

其他可測試或診斷用選項可由下列命令查看：

```bash
python3 tools/rdc_counter_report.py --help
```

其中包括 `--manifest`、`--golden-runner`、`--pvrgpu-runner`、
`--timeout-seconds` 及 `--require-manifest`。兩個 runner 都是 native
executable；不再接受或尋找 shell adapter。

命令列退出碼：

- `0`：找到的 RDC 全部 PASS。
- `1`：至少一個 RDC FAIL，或目錄內沒有 `.rdc`。
- `2`：manifest、runner、輸入目錄等全域設定錯誤，無法開始執行。
- `130`：使用者取消。

單一 RDC 失敗不會停止整批；工具仍會執行後續檔案並寫出最終報告。只有全域設定錯誤或使用者取消會提前停止 backend 執行。

## RDC 與 manifest 的映射

每個找到的 RDC 都會先計算 SHA-256。若 digest 對應到 `config/rdc-glbench-v1.tsv` 的 frozen manifest row，工具會沿用該 row 的 case、width、height 與 canonical filename。

若 SHA-256 不在 frozen manifest 中，工具不會直接 FAIL；它會先嘗試讀取附近的 dEQP `recorder/manifest.txt` 取得正式 case name。若沒有 dEQP manifest，則由 RDC 檔名推導安全 case name。這讓已 capture 的 dEQP 或其他 RDC 可以進入真正的 Golden/PvrGPU replay。需要舊的 strict 行為時，命令列可加 `--require-manifest`。

因此輸入目錄的階層可以任意安排，檔案也可位於多層子目錄。若內容已被重新命名且命中 frozen manifest，工具會在該次 artifact 中以 manifest 的 canonical filename 建立 staging link，再交給 backend。

## PASS/FAIL 規則

Golden backend 使用 counter-enabled Mesa llvmpipe，輸出一個 frame 的 `Report.md`。PvrGPU backend 輸出一個 `pvrgpu.counter.v1` JSONL stream。兩者會各自正規化為固定順序的 17 行 `name=integer` text：

```text
ia_vertices
ia_primitives
vs_invocations
gs_invocations
gs_primitives
c_invocations
c_primitives
ps_invocations
hs_invocations
ds_invocations
cs_invocations
ts_invocations
ms_invocations
ms_primitives
drawlists
setup_triangles
texel_fetches
```

PASS 必須同時滿足：兩份正規化 counter text 完全相同，以及應比較的
PNG 解碼後 RGBA8 完全相同。runner exit failure、timeout、缺少或無效的
Golden report、PvrGPU protocol failure、MemoryPool leak、counter
mismatch、缺少應存在的 PNG、PNG decode/dimension/pixel mismatch 都是
FAIL；每個 failure stage 與原因都會寫進 `result.json` 和最終
`report.md`。

## 產物

典型輸出如下：

```text
rdc-counter-<timestamp>-<pid>/
├── report.md
├── run.json
└── cases/
    └── 0001-<case>-<short-sha>/
        ├── input/
        │   └── <canonical-name>.rdc
        ├── result.json
        ├── counter_golden.txt
        ├── counter_pvrgpu.txt
        ├── counter_diff.txt          # 只有 counter mismatch 才有
        ├── golden.png                 # Golden framebuffer（有 color output 時）
        ├── png_diff.txt               # PNG missing/decode/pixel mismatch 時
        ├── png_skip.txt               # 只有明確 no-color evidence 時
        ├── golden/
        │   ├── backend-result.json   # llvmpipe stage/reason、SHA、log pointers
        │   ├── Report.md             # Golden backend report
        │   ├── stdout.log
        │   ├── stderr.log
        │   └── player-output/
        └── pvrgpu/
            ├── backend-result.json   # pvrgpu stage/reason、SHA、log pointers
            ├── stdout.jsonl
            ├── stderr.log
            ├── driver-command.txt
            ├── driver-counter.txt
            ├── player.stdout.log
            ├── player.stderr.log
            ├── model.stdout.jsonl
            ├── model.stderr.log
            ├── player-png/
            └── png/
golden-cache/
└── <rdc-sha256>/
    ├── counter_golden.txt
    ├── golden.png
    └── metadata.json
```

最外層小寫 `report.md` 是整批 PASS/FAIL 報告；`golden/Report.md` 是單次 llvmpipe replay 的原始 counter report，兩者用途不同。`run.json` 保存同一批結果的 machine-readable summary；每個 case 的 `result.json` 保存該檔案的 stage、reason 與 artifact 相對路徑。

Debug 時先由 case 的 `result.json` 找到失敗 stage，再看對應 backend 的
`backend-result.json`。後者會固定記錄輸入 RDC SHA-256、child exit code、
reason 與 stdout/stderr 相對路徑；PvrGPU 的 `runner.txt` 另保存實際解析到的
Mesa prefix、RenderDoc root、trace draw-action 來源與 SystemC bridge，避免
只看到最終 counter mismatch 卻無法重現 runtime。

## Native single-RDC backends

兩個 backend 可獨立執行，且使用相同的必要參數：

```bash
build/bin/llvmpipe \
  FILE.rdc \
  --case CASE \
  --width WIDTH \
  --height HEIGHT \
  --outdir DIRECTORY

build/bin/pvrgpu \
  FILE.rdc \
  --case CASE \
  --width WIDTH \
  --height HEIGHT \
  --outdir DIRECTORY
```

直接操作時可使用上面的 positional `FILE.rdc`。為了與 directory worker
相容，兩個 backend 也接受等價的 `--rdc FILE.rdc`；worker 使用這個形式
並為每一邊傳入各自的 `--outdir`。

`llvmpipe` 驗證 renderer，並在 `DIRECTORY` 下保存原始
`Report.md`、replay PNG 與 log。`pvrgpu` 保存
`pvrgpu.counter.v1` JSONL、model PNG、driver/model log 與必要的中間
artifact。單一 backend 不負責跨 backend 比較；directory worker 或 UI
在兩邊都完成後正規化 counter 並執行 PNG PASS gate。

若直接執行 `pvrgpu FILE.rdc` 而未提供 `--trace-draw-actions`，runner 會先
做一次 metadata probe，從 player log 取得真實 draw-action 數，再執行正式
PvrGPU replay；probe log 與 PNG 會保留供 debug。directory worker 已從
llvmpipe 階段取得該數值，因此正常 batch 不需額外 probe。

可用下列環境變數覆寫本機 runtime：

- `PVRGPU_RENDERDOC_MESA_ROOT`
- `PVRGPU_RDC_PLAYER`
- `PVRGPU_LLVMPIPE_MESA_PREFIX`
- `PVRGPU_MESA_PVRGPU_PREFIX`
- `PVRGPU_MODEL_STUB`
- `PVRGPU_SYSTEMC_API_LIB`

一般情況應在 `config/local.env` 設定 machine-specific 路徑，讓 UI 與 CLI 共用相同 runtime。
