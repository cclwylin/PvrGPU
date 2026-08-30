# RDC Counter Pass/Fail UI

這是一個獨立的 Pass/Fail 工具，不會啟動或整合 PvrGPU debug UI。它會遞迴掃描指定目錄下的 `.rdc`，依序執行 Golden 與 PvrGPU，將兩邊結果正規化成固定 17 欄 counter text，完成全部檔案後產生 `report.md`。

PNG 可能由 backend 產生並保留為 artifact，但不參與 PASS/FAIL。

## 啟動圖形介面

第一次使用先建立 UI runtime：

```bash
./scripts/setup-ui.sh
```

之後直接啟動獨立視窗：

```bash
./scripts/run-rdc-counter-ui.sh
```

在 UI 中選擇：

- RDC directory：要遞迴掃描的輸入目錄。
- Output root：每次執行的 timestamped result directory 要放置的父目錄。
- PvrGPU runner：單一 RDC 的 PvrGPU runner script/executable；預設為 `scripts/run-rdc-pvrgpu-driver-systemc.sh`，也就是 `RDC -> RenderDoc player -> Mesa/Gallium pvrgpu -> PvrGPU SystemC`。UI 會顯式傳給 backend，不會使用舊的 trace-capsule default。

開始後，UI 會顯示目前檔案、Golden/PvrGPU/compare 階段及累計 PASS/FAIL。取消會終止目前 runner，尚未執行的 RDC 也會在最終報告中標示為失敗。完成後可由 UI 開啟該次執行的 `report.md`。`Quit` 會關閉視窗；若 worker 還在執行，會先終止目前 replay 再退出。

結果表格預設使用 `Fail` filter，只顯示最終結果為 FAIL 的 RDC，方便直接 debug。可切換成 `Pass` 只看通過項目，或 `All` 顯示整批掃描到的所有 RDC；切換 filter 不會重新執行測試。

Golden counter 會依 RDC SHA-256 自動 cache 在 output root 下。若同一個 RDC 之前已成功產生過可解析的 `counter_golden.txt`，下一次 UI/CLI 執行會直接複製 cached Golden counter，跳過 llvmpipe replay，只重跑 PvrGPU 與 compare。cache hit 時 UI/報告的 Golden 欄會顯示 `CACHED` 而不是 `PASS`，case artifact 也會留下 `golden/cache-hit.txt`。

## 命令列執行

```bash
./scripts/run-rdc-counter-report.sh \
  --rdc-dir "/path/to/rdc-directory" \
  --output-root "/path/to/result-root" \
  --pvrgpu-runner "/path/to/current-pvrgpu-runner"
```

`--rdc-dir` 是必要參數。`--output-root` 可省略；預設為：

```text
${PVRGPU_WORK_ROOT}/out/rdc-counter-report
```

每次執行都會在 output root 下建立新的 `rdc-counter-<UTC timestamp>-<pid>/`，不會覆寫先前報告。

需要讓另一個程式（例如 UI）讀取即時進度時，加上 `--json`：

```bash
./scripts/run-rdc-counter-report.sh \
  --rdc-dir "/path/to/rdc-directory" \
  --output-root "/path/to/result-root" \
  --pvrgpu-runner "/path/to/current-pvrgpu-runner" \
  --json
```

此時 stdout 是一行一筆的 JSONL event。runner 的 stdout/stderr 會保存在各 RDC 的 artifact 目錄，不會混入 event stream。

其他可測試或診斷用選項可由下列命令查看：

```bash
./scripts/run-rdc-counter-report.sh --help
```

其中包括 `--manifest`、`--golden-runner`、`--pvrgpu-runner`、`--timeout-seconds` 及 `--require-manifest`。`--pvrgpu-runner` 必須顯式指定；舊的 trace-capsule runner 已移除，避免誤跑 adapter。

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

只有兩份正規化 counter text 完全相同才是 PASS。runner exit failure、缺少或無效的 Golden report、PvrGPU protocol failure、MemoryPool leak，或 counter mismatch 都是 FAIL；每個 failure stage 與原因都會寫進 `result.json` 和最終 `report.md`。

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
        ├── golden/
        │   ├── Report.md             # Golden backend report
        │   ├── stdout.log
        │   ├── stderr.log
        │   └── player-output/
        └── pvrgpu/
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
    └── metadata.json
```

最外層小寫 `report.md` 是整批 PASS/FAIL 報告；`golden/Report.md` 是單次 llvmpipe replay 的原始 counter report，兩者用途不同。`run.json` 保存同一批結果的 machine-readable summary；每個 case 的 `result.json` 保存該檔案的 stage、reason 與 artifact 相對路徑。

## Golden adapter

production Golden adapter 也可單獨執行：

```bash
./scripts/run-rdc-golden-counter.sh \
  --rdc FILE.rdc \
  --case CASE \
  --width WIDTH \
  --height HEIGHT \
  --outdir DIRECTORY
```

它透過 BenchScope `play_glbench_rdc.sh` 及 counter-enabled llvmpipe runtime replay 一次，驗證產物非空且 renderer 包含 `llvmpipe`，再將生成的 report 複製成 `DIRECTORY/Report.md`。它不比較 PNG、不跑 PvrGPU，也不做第二次 cold replay。

可用下列環境變數覆寫本機 runtime：

- `PVRGPU_BENCHSCOPE_ROOT`
- `PVRGPU_RENDERDOC_MESA_ROOT`
- `PVRGPU_LLVMPIPE_MESA_PREFIX`

一般情況應在 `config/local.env` 設定 machine-specific 路徑，讓 UI 與 CLI 共用相同 runtime。
