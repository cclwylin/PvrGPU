# `script/` — dynamic-link dEQP entry point

| 檔案 | 用途 |
| --- | --- |
| `run_deqp_dynamic.sh` | 執行引擎：把已 build 好的 dEQP binary、PCO driver、PvrGPU bridge 在 runtime 串起來 |
| `deqp_dynamic_ui.py` | PySide6 桌面前端：預設選單 → 即時 run status → 最後的 dashboard |
| `deqp_4level_ui.py` | 同一個引擎的四層回歸前端：L1–L4 × 30 組目錄、等距取樣、分片、合併與分組統計（見 [docs/dEQP_4level.md](../docs/dEQP_4level.md)）|
| `run_deqp_group_sample.sh` | 取 `tools/deqp_groups.py` 其中一組的樣本跑，列出 tally 與每個失敗的 QPA 理由（`--list` 列出 24 組）|
| `run_deqp_transform_feedback.py` | 精確執行 stock GLES3 Transform Feedback 1320 案例，保存 runtime 雜湊並核對 llvmpipe 逐案狀態及原生模型記錄（見 [驗證報告](../docs/TRANSFORM_FEEDBACK_VALIDATION.md)）|

**dEQP 一律走這條路。** `script/run_regression.sh` 不跑 dEQP：`2.dEQP` 的 RDC
capture 已經退役（見 `tools/run_rdc_regression.py` 的 `RETIRED_SUITES`）。那
1809 個 pattern 從來沒有記錄過 pass/fail 基準，所以其中出現失敗也無法歸因於當
次改動；而這裡的 runner 回報的是 dEQP 主程式自己對每個 case 的
Pass/Fail/NotSupported。RDC regression 留給有基準的 GLBench 與 glmark2。

## 讀一個 PASS 之前

這裡跑出來的 PASS/FAIL 只有在**結果是 PCO driver 與 PvrGPU model 真的算出來的**
前提下才有意義。專案的第一條規範是：不得以測試名稱、capture 名稱或 workload
名稱決定 counter、像素或某個 draw 是否執行。完整規範見
[根目錄 README](../README.md) 與 [PvrGPU.md §3.5](../PvrGPU.md)。

實務上，讀這個 runner 的輸出時：

- `NotSupported` 是誠實的結果，不是缺陷。dEQP 自己跳過驅動未宣告的能力
  （例如 wide line / wide point），這與模型算錯是兩回事。
- **失敗的形狀比失敗的數量重要。** `results.qpa` 說「N missing pixels，
  0 incorrectly filled」代表回讀根本沒看到 draw；同時出現 missing 與
  incorrectly filled 則代表回讀通了，剩下的是光柵化或狀態差異——那是不同的、
  而且比較好的問題。
- 一個案例從 FAIL 變 PASS 時，要能指出是**哪一段實作**讓它變的。如果指不出來，
  它多半不是被修好的。

## `run_deqp_dynamic.sh`

Runs a stock upstream dEQP module binary against the **already-built** PCO
driver and the **already-built** PvrGPU SystemC model. Nothing is compiled,
relinked, or rebuilt; the three artifacts meet only at run time.

```
deqp-<module>                          dEQP project build, unmodified
  │  DYLD_LIBRARY_PATH → <mesa-prefix>/lib
  ▼
libEGL.1.dylib / libGLESv2.2.dylib     Mesa prefix carrying the PCO driver
  │  GALLIUM_DRIVER=pvrgpu             (gallium pvrgpu + PowerVR PCO compiler)
  ▼
dlopen(PVRGPU_SYSTEMC_API_LIB)         pvrgpu_cmd.c resolves the bridge at run
  ▼                                    time — see pvrgpu_submit_systemc_api()
libpvrgpu_systemc_bridge.dylib         PvrGPU build, SystemC model
```

This is the dynamic counterpart of `pvrgpu-deqp`, which folds the same three
pieces into one statically linked executable (`README.md` → *Live dEQP Runner*).
Use this script when the dEQP binary and the driver are built separately and
have to stay separate.

### Quick start

```bash
# 1. Preflight only: resolve every path, check architectures, run nothing.
./script/run_deqp_dynamic.sh --check --print-env

# 2. One exact case.
./script/run_deqp_dynamic.sh -c dEQP-GLES2.functional.prerequisite.clear_color

# 3. Prove the run really loaded the PCO driver and the bridge, not system GL.
./script/run_deqp_dynamic.sh -c dEQP-EGL.functional.create_context.rgb565_no_depth_no_stencil \
    --verify-link

# 4. A list of cases, one fresh process each.
printf '%s\n' dEQP-EGL.functional.create_context.rgb565_no_depth_no_stencil \
              dEQP-EGL.functional.create_context.rgb888_depth_no_stencil > /tmp/cases.txt
./script/run_deqp_dynamic.sh --caselist /tmp/cases.txt --keep-going
```

### Where the paths come from

Everything is read from `config/local.env` first, then overridden by the
environment, then by flags:

| Variable | Meaning |
| --- | --- |
| `PVRGPU_MESA_PVRGPU_PREFIX` | Mesa prefix with the PCO/pvrgpu Gallium driver (`--mesa-prefix`) |
| `PVRGPU_SYSTEMC_API_LIB` | `libpvrgpu_systemc_bridge.dylib` (`--systemc-lib`) |
| `PVRGPU_BUILD_DIR` | PvrGPU build dir, used as the bridge fallback |
| `PVRGPU_DEQP_PROJECT_DIR` | dEQP project; its `out/deqp-build-<arch>.env` supplies the CMake build dir (`--deqp-project`) |
| `PVRGPU_DEQP_BUILD_DIR` | dEQP CMake build dir directly (`--deqp-build-dir`) |
| `PVRGPU_OUTPUT_ROOT` | run output root (`--output-dir`) |

The module binary is `<build>/modules/<module>/deqp-<module>`; the module is
inferred from the case prefix (`dEQP-EGL.` → `egl`, `dEQP-GLES31.` → `gles31`,
and so on). The dEQP project builds only `deqp-gles31` by default — build the
module you need there (`SELECTED_BUILD_TARGETS=deqp-gles2`) or point
`--deqp-binary` at an existing one.

### Contract

One exact case per process, the same rule `pvrgpu-deqp` and
`tools/deqp_live_ui.py` follow: the SystemC bridge defers simulation until
process exit and keeps only its latest submitted command. Wildcards and
comma-separated case lists are rejected; a `--caselist` is expanded here and
each case gets its own fresh process.

### Output

```
<output-dir>/
  summary.tsv                       case, status, exit code, QPA path
  cases/<case-slug>/
    run.txt                         schema=pvrgpu.deqp-dynamic-run.v1 manifest
    results.qpa                     dEQP result log
    run.log                         stdout + stderr
    link.txt                        loaded GL/driver libraries (--verify-link)
    driver-command.txt
    driver-counter.txt
    systemc.jsonl
    systemc.stderr.log
    systemc/                        model artifacts (PNG, dumps)
```

### Preflight checks

`--check` fails loudly rather than running a misleading test when:

* the Mesa prefix has no `libEGL.1.dylib` / `libGLESv2.2.dylib`;
* the SystemC bridge dylib is missing;
* the dEQP runner, the driver, and the bridge are not all built for the host
  architecture (`lipo -archs`);

and warns when the runner has no dynamic `libEGL`/`libGLESv2` dependency (a
statically linked runner such as `pvrgpu-deqp` cannot be redirected by
`DYLD_LIBRARY_PATH`), or when no `pvrgpu` symbol text is found under the Mesa
prefix (that prefix probably does not carry the PCO driver).

### Note on scope

The driver advertises ES 3.1 by default, so GLES3/GLES31 groups start; set
`PVRGPU_DISABLE_ES3=1` to get the ES2-only surface back (the catalog and both
UIs follow that switch). Image-comparison cases still report Fail because the
bridge only simulates at process exit and never writes the model framebuffer
back for `glReadPixels` — read the per-case `systemc/*.png` for the model's
actual output.


---

## `deqp_dynamic_ui.py` — 桌面前端

`tools/deqp_live_ui.py` 的動態連結版本。它不自己實作任何 runtime wiring，
只是驅動上面那支 script（`--emit-events`），所以 UI 跟 CLI 永遠跑同一條路徑。

```bash
python3 script/deqp_dynamic_ui.py
```

沒有 PySide6 時會自動用 `PVRGPU_UI_PYTHON` / `PVRGPU_UI_VENV` 重新啟動，
規則與 `tools/deqp_live_ui.py` 相同。

### 左側：預先可選的項目

* **選擇要跑的項目** — 四種模式
  * `Group 目錄（24 組）`：直接用 `tools/deqp_groups.py` 的同一份目錄，
    blocked 的組以 ⛔ 標記並顯示原因；按 Run 會先透過同一條 wiring 做
    caselist discovery，再用 `filter_exact_cases()` 過濾，最後一個 case
    一個 process 跑。「最多 cases」可先跑子集合。
  * `常用單一 case`：EGL / GLES2 / GLES31 的 smoke case 快捷。
  * `自訂 case`：手動輸入一個 exact case。
  * `Caselist 檔案`：自己準備的清單。
* **路徑與環境** — PCO driver prefix、SystemC bridge、dEQP binary 或 build dir、
  output root，留空就沿用 `config/local.env`；`Preflight (--check)` 會跑
  `--check --print-env` 並把結果顯示成 ✅ / ❌。
* **執行參數** — GL config、surface type、size、log-images、timeout、
  `--verify-link`、`--keep-going`。

### 右側：run status 與 dashboard

* 上方固定區：目前 case、進度條、`Total / Pass / Fail / Skip / Warn / Elapsed` 六格,
  右上角是 **複製診斷資訊**。
* `Run status`：每個 case 一列（#、case、status、exit、duration、artifacts），
  status 依 pass/fail/skip/warning 上色；artifacts 欄顯示 `systemc✓ / png / cmd / link`
  代表這個 case 真的有產生模型輸出。雙擊可打開該 case 目錄。
* `Log`：script 的完整輸出（事件行已被解析掉）。
* `Dashboard`：跑完自動切過來 —— 結果分佈長條、這次實際串到的東西
  （module / runner / archive / PCO driver / bridge / arch / output）、
  最慢的 10 個 case、未通過清單，以及開啟 `summary.tsv` 與輸出目錄的按鈕。
* `Artifacts`：整個輸出目錄的檔案列表，雙擊開啟。

### 複製診斷資訊

右上角那個按鈕把一次執行收成一份純文字：選了什麼、UI 裡填的路徑、runner 回報
的實際串接、完整的執行參數、每個 case 一行的結果表，然後是**每個未通過 case
的細節** —— case 目錄、實際下的命令、`systemc.jsonl` 裡的 done/error 行、
`stderr.log` 的尾巴、有沒有產出 PNG —— 最後接上 UI log 的尾巴。

按下去會同時做兩件事：複製到剪貼簿,以及寫成這次 run 目錄裡的
`diagnostics.txt`（剪貼簿內容的第一行就是那個檔案路徑）。要貼給別人看就用
剪貼簿；如果對方本來就讀得到這台機器的檔案，給檔案路徑更好 —— 報告為了好貼
會截斷（最多 `DIAGNOSTICS_MAX_CASES` 個失敗 case、`DIAGNOSTICS_LOG_LINES`
行 log），但 run 目錄裡什麼都沒少。

還沒跑過也可以按，那會得到一份只有設定與路徑的報告，適合用來問「我這樣設定
對不對」。

### 版面與配色

預設是淺色（`DEFAULT_APPEARANCE = "light"`）。選單列的
**外觀 → 淺色 / 深色 / 跟隨系統** 可以改；「跟隨系統」會偵測桌面 palette，
深色桌面套 `DARK_THEME`、淺色套 `LIGHT_THEME`。

切換即時生效（palette、stylesheet、表格狀態顏色一起換），選擇存在 QSettings
裡，下次開啟沿用。三種配色都用 Fusion style 統一 combo popup、spinbox、
scrollbar 等原生元件，所以不會出現白底白字。

左欄是「caption 在上、控制項在下」的堆疊式版面（不是兩欄 grid），
所以長路徑不會把 label 欄擠掉；設定區可捲動，`Run / Cancel / Quit` 固定在
左欄底部，永遠看得到。`Quit` 會關掉視窗並存下路徑設定；如果還有工作在跑，
會先問要不要中止。Group 與常用 case 的下拉選單用
`AdjustToMinimumContentsLengthWithIcon`，避免最長的那一項決定整個側欄寬度。

### UI 與 script 的分工

UI 只做三件事：組參數、解析 `PVRGPU_DYN {json}` 事件、呈現。
`resolved / check_ok / discovery_started / discovery_finished / run_start /
case_start / case_end / run_end` 這幾個事件就是兩者之間的全部介面，
所以 CLI 單獨跑出來的結果與 UI 完全一致，CI 也可以只用 script。


## `deqp_4level_ui.py` — 四層回歸前端

`deqp_dynamic_ui.py` 一次選一個 group 或一個 case；這一支選的是
[docs/dEQP_4level.md](../docs/dEQP_4level.md) 定義的**層**。

```bash
python3 script/deqp_4level_ui.py
```

四層都涵蓋全部 30 組，差別只在每組取幾條：L1 每組 `min(N, 100)`、L2
`min(N, 400)`、L3 全跑、L4 是四個 module 列舉出的全部案例。取樣用文件第 4.2 節
的等距取法 `c[i*len(c)//n]`，不是整數 stride——後者在 N 小於配額兩倍時會退化成
「取前 N 條」，第 19 組因此會整批漏掉 `indirect_dispatch`。

- **分片**：L1–L3 預設 8 片，round-robin 切，每片自己的 `--output-dir`。
  驅動是附加寫入 counter 檔的，兩片共用一個目錄會讀到交錯的舊事件。
  L4 強制 1 片並開啟存圖，這是文件刻意的規定。
- **gl-config 例外**：第 22、30 組的 `multisample.default_framebuffer.` 自動
  切到 `rgba8888d24s8ms4`，並分到獨立的 caselist 與輸出目錄。
- **30 組目錄**寫在這支 UI 裡，不在 `tools/deqp_groups.py`（那裡還是舊的 24
  組）。表格與文件的三個層總數（2534 / 7904 / 30491）在 import 時互相檢查，
  對不上就直接拋例外，不會安靜地跑成另一份回歸。
- **計畫分頁**會把文件記的 N 與這台機器 discovery 到的條數並排；不一致代表 CTS
  換過，取樣落在另一批案例上。

每次 run 的目錄裡有 `plan.json`（這次的計畫）、`commands.sh`（等效的手動指令，
可以直接貼進終端機重跑同一批 case）、合併後的 `summary.tsv`、以及
`stats.txt`（每組的 pass/fail 與分片的 exit code）。
