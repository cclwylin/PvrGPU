# dEQP 四層回歸測試速查表

四層回歸的分層定義、30 組測試目錄，以及每一組可以直接複製執行的指令。
用途是查詢：想測某個功能時，直接在本文件找到那一組，複製指令執行。

時間預算來自 456 個既有 run、6,349 條案例的實測耗時，不是估計值：
平均 0.66 秒、中位數 0.38 秒、p90 1.46 秒。8 分片實測加速 6.2 倍。
表內 CPU／wall 時間沿用原量測作上限參考；新的 L4-only 規則會縮短
L1/L2/L3，待完成一輪新版實測後再更新時間欄。

**實作狀態**：`script/deqp_4level_ui.py` 已實作本文件的分層、30 組目錄、
長測項隔離與分片。`tools/deqp_groups.py` 仍是舊的 24 組目錄；文件中標示
「可直接執行」的指令仍可從 discovery 檔案產生相同 caselist。

---

## 一、四層定義

| 層 | 名稱 | 案例 | CPU 時間 | 8 分片 wall | 預算 | 用途 |
|---|---|---|---|---|---|---|
| L1 | Very Fast Regression | 2,474 | 20 min | 2.7 min | 5 min | 每次 commit 前 |
| L2 | Fast Regression | 7,750 | 54 min | 8.6 min | 15 min | 修完一個 subsystem |
| L3 | Detail Regression | 30,224 | 4.0 h | 40 min | 1 h | milestone、過夜 |
| L4 | Full Regression | 90,360 | 12.5 h | 見下 | 25 h | 版本釋出、認證前 |

### 分層規則

| 層 | 每組取樣規則 |
|---|---|
| L1 | 先移除 L4-only 長測項，再每組 `min(N, 100)` 條等距取樣；第 11 組 compressed 蓋在 30 |
| L2 | 先移除 L4-only 長測項，再每組 `min(N, 400)` 條等距取樣；第 11 組 compressed 蓋在 120 |
| L3 | 30 組目錄扣除 L4-only 長測項後全跑 |
| L4 | 四個 module 列舉出的全部案例，單行程、開啟存圖 |

L1 到 L3 必須分片才進得了預算。**L4 刻意不分片**：25 小時是單行程加存圖的數字，
認證等級的 run 要順序固定、單一 QPA、不跨片干擾。L4 另外要帶跑 `ctest` 與 `rdc_counter_report.py`。

第 11 組 compressed 被蓋住是因為它每條 2.78 秒，是所有組裡最貴的，
滿額會吃掉 L1 四分之一的預算。

### L4-only 長測項

L1、L2、L3 一律先排除下列 267 條；L4 仍完整執行，單案例與自訂 caselist
除錯也不套用此過濾。這是固定的 case-prefix 規則，不會因上一次執行速度而
悄悄改變下一次 caselist。

| 組 | L4-only family | 案例 |
|---|---|---:|
| 18 | `stress.long_shaders.*`、`stress.long_running_shaders.*` | 40 |
| 20 | `functional.draw_indirect.compute_interop.large.*`、`stress.draw_indirect.{drawarrays,drawelements}.data_over_bounds_with_primcount` | 66 |
| 22 | `functional.multisample.default_framebuffer.constancy_*`、`functional.texture.multisample.samples_8.sample_mask_*` | 9 |
| 28 | `functional.image_load_store.{2d_array,3d,cube}.atomic.*` | 102 |
| 29 | `functional.synchronization.inter_call.with_memory_barrier.*`、`without_memory_barrier.{image_atomic_dispatch_100_calls_128x128_invocations,ssbo_atomic_dispatch_100_calls_32k_invocations}` | 30 |
| 30 | `functional.multisample.{default_framebuffer,fbo_4_samples,fbo_8_samples,fbo_max_samples}.constancy_*` | 20 |

這些 family 包含無界 shader loop，或在 PvrGPU SystemC 實測會反覆耗用數分鐘；
移出快速層不代表 Pass、Skip 或刪除，只是改由 L4／單獨長測批次執行。

### 每層涵蓋幾組

L1 到 L3 涵蓋 29 個一般組；第 18 組的 40 條全是長測項，只在 L4。
其餘五組仍保留一般案例，只有上表的長 family 移到 L4。

| 層 | 涵蓋組數 | 該層已 100% 覆蓋 | 仍在取樣 | 案例 |
|---|---|---|---|---|
| L1 | 29 | 7 組 | 22 組 | 2,474 |
| L2 | 29 | 13 組 | 16 組 | 7,750 |
| L3 | 29 | 29 組 | 0 | 30,224 |
| L4 | 全部四個 module | — | — | 90,360 |

**L1 已 100% 覆蓋一般案例的 7 組**（案例數不足 100，全取）：

| # | 組名 | 案例 |
|---|---|---|
| 1 | EGL Create context | 22 |
| 3 | EGL Robustness | 60 |
| 4 | Color clear | 19 |
| 7 | Instancing | 45 |
| 16 | Stress draw | 74 |
| 17 | Stress memory | 80 |
| 30 | Basic MSAA（扣除 20 條 L4-only） | 44 |

**L2 再多 6 組**（合計 13 組已 100% 覆蓋一般案例）：

| # | 組名 | 案例 |
|---|---|---|
| 2 | EGL Image | 226 |
| 8 | Rasterization | 108 |
| 19 | Compute | 195 |
| 20 | Draw indirect（扣除 66 條 L4-only） | 178 |
| 23 | Geometry shading | 207 |
| 29 | Sync + atomic counter（扣除 30 條 L4-only） | 372 |

對這 13 組的一般案例而言 **L2 和 L3 是同一件事**。長 family 仍須跑 L4。

**只有 L3 才跑得完一般案例的 15 組**是第 5、6、9、10、12、13、14、15、21、22、
24、25、26、27、28 組。其中第 25 組 shader operator 6,478 條差距最大，L2 只看得到 6%。
第 11 組 compressed 兩層都被蓋住，要全跑只有 L3。

各層配額皆套在排除 L4-only 後的 pool；若剩餘案例少於配額（例如第 20、29、30 組），就全取一般案例。

---

## 二、環境準備

所有指令都先做這三行：

```bash
cd "/Users/linwanyi/Library/Mobile Documents/com~apple~CloudDocs/Codex/PvrGPU" && set -a && source config/local.env && set +a && export DISC="$PVRGPU_OUTPUT_ROOT/deqp_groups/discovery" && echo "$DISC"
```

discovery 檔案是列舉快取，三個檔案分別是 `egl.txt`、`gles3.txt`、`gles31.txt`。
如果不存在或 dEQP 重建過，重新產生：

```bash
script/run_deqp_dynamic.sh --module gles31 --discover --caselist-out "$DISC/gles31.txt"
```

---

## 三、30 組目錄

`selector` 欄可直接當 grep 樣式使用（見第四節）。`N` 是該組全部案例數。

| # | 組名 | Suite | selector | N | L1 | L2 |
|---|---|---|---|---|---|---|
| 1 | EGL Create context | EGL | `functional.create_context.` | 22 | 22 | 22 |
| 2 | EGL Image | EGL | `functional.image.` | 226 | 100 | 226 |
| 3 | EGL Robustness | EGL | `functional.robustness.` | 60 | 60 | 60 |
| 4 | Color clear | GLES3 | `functional.color_clear.` | 19 | 19 | 19 |
| 5 | FBO | GLES3 | `functional.fbo.` | 2077 | 100 | 400 |
| 6 | Fragment ops | GLES3 | `functional.fragment_ops.` | 3176 | 100 | 400 |
| 7 | Instancing | GLES3 | `functional.instanced.` | 45 | 45 | 45 |
| 8 | Rasterization | GLES3 | `functional.rasterization.` | 108 | 100 | 108 |
| 9 | Texture functions + derivate | GLES3 | `functional.shaders.texture_functions.`<br>`functional.shaders.derivate.` | 1221 | 100 | 400 |
| 10 | Shader built-in functions | GLES3 | `functional.shaders.builtin_functions.` | 1730 | 100 | 400 |
| 11 | Compressed textures | GLES3 | `functional.texture.compressed.` | 322 | 30 | 120 |
| 12 | Texture filtering | GLES3 | `functional.texture.filtering.` | 1124 | 100 | 400 |
| 13 | Transform feedback | GLES3 | `functional.transform_feedback.` | 1320 | 100 | 400 |
| 14 | UBO | GLES3 | `functional.ubo.` | 2357 | 100 | 400 |
| 15 | Vertex arrays | GLES3 | `functional.vertex_arrays.` | 1005 | 100 | 400 |
| 16 | Stress draw | GLES3 | `stress.draw.` | 74 | 74 | 74 |
| 17 | Stress memory | GLES3 | `stress.memory.` | 80 | 80 | 80 |
| 18 | Stress shaders | GLES3 | `stress.long_shaders.`<br>`stress.long_running_shaders.` | 40 | 0 | 0 |
| 19 | Compute | GLES31 | `functional.compute.` | 195 | 100 | 195 |
| 20 | Draw indirect | GLES31 | `functional.draw_indirect.`<br>`stress.draw_indirect.` | 244 | 100 | 178 |
| 21 | SSBO | GLES31 | `functional.ssbo.` | 2061 | 100 | 400 |
| 22 | Multisample | GLES31 | `functional.texture.multisample.`<br>`functional.shaders.sample_variables.`<br>`functional.shaders.multisample_interpolation.`<br>`functional.sample_shading.`<br>`functional.multisample.` | 595 | 100 | 400 |
| 23 | Geometry shading | GLES31 | `functional.geometry_shading.` | 207 | 100 | 207 |
| 24 | Tessellation | GLES31 | `functional.tessellation.` | 406 | 100 | 400 |
| 25 | Shader operator | GLES3 | `functional.shaders.operator.` | 6478 | 100 | 400 |
| 26 | Shader matrix | GLES3 | `functional.shaders.matrix.` | 2646 | 100 | 400 |
| 27 | Texture wrap | GLES3 | `functional.texture.wrap.` | 1440 | 100 | 400 |
| 28 | Image load/store | GLES31 | `functional.image_load_store.` | 747 | 100 | 400 |
| 29 | Sync + atomic counter | GLES31 | `functional.synchronization.`<br>`functional.atomic_counter.` | 402 | 100 | 372 |
| 30 | Basic MSAA | GLES3 | `functional.multisample.` | 64 | 44 | 44 |

本表只記結構，不記通過率。各組當下的狀態看該次 run 的 `summary.tsv`
（統計指令見 4.5），不在本文件維護。

---

## 四、指令速查

### 4.1 跑某一組的某一層（可直接執行）

三步：grep 出全組 caselist、依配額 stride 取樣、執行。
以下用第 19 組 compute 的 L3（全跑）當範例：

```bash
grep -E '^dEQP-GLES31\.functional\.compute\.' "$DISC/gles31.txt" > /tmp/g19_L3.txt && wc -l /tmp/g19_L3.txt
```

```bash
script/run_deqp_dynamic.sh --caselist /tmp/g19_L3.txt --keep-going --surface-type pbuffer --size 256x256 --gl-config rgba8888d24s8ms0 --log-images disable --output-dir "$PVRGPU_OUTPUT_ROOT/deqp_groups/g19_L3"
```

### 4.2 依配額取樣（L1 / L2）

取第 19 組的 L1 配額 100 條。取樣要跨越所有子群，不能取字母序的前 N 條：

```bash
python3 -c "import sys;c=[l.strip() for l in open(sys.argv[1]) if l.strip()];n=min(int(sys.argv[2]),len(c));print('\n'.join(c[i*len(c)//n] for i in range(n)))" /tmp/g19_L3.txt 100 > /tmp/g19_L1.txt
```

**不要用整數 stride**（`c[::len(c)//n]`）。當 N 小於配額兩倍時 `len(c)//n` 會退化成 1，取樣就變回「前 N 條」，正是要避免的取法。
第 19 組取 100 條時整數 stride 完全漏掉 `indirect_dispatch` 那 18 條；
上面的等距取法給出 basic 22、shared_var 69、indirect_dispatch 9，與母體比例一致。
受影響的是 N 介於配額與兩倍配額之間的組：L1 配額 100 時是第 2、8、19、23 組，
L2 配額 400 時是第 22、24、28 組。

取樣前要先排除上節的 L4-only family；UI 產生的 `commands.sh` 會自動加入
同一組 `grep -vE` 規則。手動產生整層 caselist 時也必須先過濾，不能在取樣後
才刪除，否則取樣分布與 UI 不一致。

### 4.3 單一案例除錯（可直接執行）

```bash
script/run_deqp_dynamic.sh -c dEQP-GLES31.functional.compute.basic.ubo_to_ssbo_single_group --log-images enable --output-dir "$PVRGPU_OUTPUT_ROOT/debug/compute1"
```

存圖與 shader 原始碼要進 QPA 時加 `--log-images enable`。
崩潰時查 PCO：`meson configure -Db_ndebug=false` 在 `build/mesa-pvrgpu`，
再設 `NIR_DEBUG=validate`，一次執行就會指出是哪個 pass 產生了壞 NIR。
`PVRGPU_DEBUG_PCO_NIR=1` 會傾印 driver 交給 PCO 的 shader。

### 4.4 八分片平行執行

L1 到 L3 都要靠這個才進得了預算。round-robin 切法讓貴的案例平均散開：

```bash
rm -f /tmp/shard_* && awk -v n=8 '{ f = "/tmp/shard_" (NR % n); print >> f }' /tmp/g19_L3.txt && for i in 0 1 2 3 4 5 6 7; do script/run_deqp_dynamic.sh --caselist /tmp/shard_$i --keep-going --output-dir "$PVRGPU_OUTPUT_ROOT/deqp_groups/g19_L3/shard_$i" & done; wait
```

每片必須有自己的 `--output-dir`。driver 是**附加**寫入 counter 檔的，
兩片寫進同一個目錄會互相污染，讀到交錯的舊事件是最容易導致誤判的陷阱。

合併結果：

```bash
cat "$PVRGPU_OUTPUT_ROOT"/deqp_groups/g19_L3/shard_*/summary.tsv | awk -F'\t' 'NR==1 || $1!="case"' > "$PVRGPU_OUTPUT_ROOT/deqp_groups/g19_L3/summary.tsv"
```

### 4.5 統計與失敗原因

```bash
awk -F'\t' 'NR>1 { c[$2]++ } END { for (k in c) printf "%-18s %d\n", k, c[k] }' "$PVRGPU_OUTPUT_ROOT/deqp_groups/g19_L3/summary.tsv" | sort
```

列出每個失敗案例的 QPA 原因：

```bash
awk -F'\t' 'NR>1 && $2!="Pass" && $2!="NotSupported" { print $5 }' "$PVRGPU_OUTPUT_ROOT/deqp_groups/g19_L3/summary.tsv" | while read d; do grep -o '<Result StatusCode="[^"]*">[^<]*' "$d/results.qpa" 2>/dev/null | sed 's/.*">//' | head -1; done | sort | uniq -c | sort -rn
```

### 4.6 現有的 24 組取樣器

第 1 到 24 組（舊目錄）可以用現成的包裝腳本，它自己處理 discovery 快取與
stride 取樣，並印出每個失敗的 QPA 原因：

```bash
script/run_deqp_group_sample.sh --list
```

```bash
script/run_deqp_group_sample.sh 19 100 g19_L1
```

注意舊目錄第 19 組的 selector 是 `compute.basic.*`（41 條），不是本文件的
`compute.*`（195 條）。第 8、9、20、22 組同樣尚未放寬。

這個腳本內部用的是整數 stride（`matched[::step]`），有 4.2 說的退化問題，
要求的條數超過該組案例數一半時會變成取前 N 條。改用 4.1 加 4.2 的手動流程可以避開。

### 4.7 跑整層

```bash
script/run_deqp_level.sh       # 預設 Level 1
script/run_deqp_level.sh --1
script/run_deqp_level.sh --2
script/run_deqp_level.sh --3
script/run_deqp_level.sh --4
```

腳本自動載入 `config/local.env`，命令列參數優先於環境變數與設定檔。
使用一般 `python3`，不需要 PySide6；可用 `PVRGPU_DEQP_LEVEL_PYTHON` 指定 Python。
UI 與命令列共用 `script/deqp_4level_catalog.py` 的目錄、長測項過濾、等距取樣與 ms4 例外。

L1–L3 預設最多 8 個 worker；`--shards 4` 可調整。L4 固定單一 worker，依序執行
四個 module 的完整列舉清單，開啟存圖。所有層都由 `run_deqp_dynamic.sh` 逐案例啟動新程序。
單案例預設不限時；可加 `--timeout 900` 設定 900 秒上限。

```bash
script/run_deqp_level.sh --dry-run                # 僅用快取產生 L1 計畫，不執行測試
script/run_deqp_level.sh --2 --shards 4 --timeout 900
script/run_deqp_level.sh --3 --refresh-discovery   # 重列舉後執行 L3
```

列舉快取預設在 `$PVRGPU_OUTPUT_ROOT/deqp_groups/discovery`；缺少 module 快取時自動列舉。
`--dry-run` 需要已存在的快取。可用 `--discovery-dir` 指定另一份快取。
每輪建立新的 `$PVRGPU_OUTPUT_ROOT/deqp_4level/<時間>_L<層>_<識別碼>` 目錄；
`--output-dir` 可指定尚未存在的結果目錄。

結果含 `plan.json`、`caselists/all.txt`、每片 caselist／log、合併的 `summary.tsv` 與 `result.json`。
Fail、程序非零退出、缺少結果或分片失敗都回傳非零；NotSupported 與 Warning 保留原始狀態。
Ctrl-C 會停止 runner 與其 dEQP 子程序，保留已完成結果。
此腳本負責 L4 的 dEQP 部分；發版要求的 `ctest` 與 RDC 回歸仍另外執行。

---

## 五、gl-config 例外

預設一律用 `rgba8888d24s8ms0`，也就是預設 framebuffer 的多重取樣關閉。
**兩組例外**，它們的 `default_framebuffer` 子群需要 ms4，否則測不到東西：

| 組 | 子群 | 案例 | 需要的 config |
|---|---|---|---|
| 22 | `multisample.default_framebuffer.` | 8 | `rgba8888d24s8ms4` |
| 30 | `multisample.default_framebuffer.` | 16 | `rgba8888d24s8ms4` |

同組其餘案例自己指定取樣數，ms0 不影響。範例：

```bash
grep -E '^dEQP-GLES3\.functional\.multisample\.default_framebuffer\.' "$DISC/gles3.txt" > /tmp/g30_df.txt && script/run_deqp_dynamic.sh --caselist /tmp/g30_df.txt --keep-going --gl-config rgba8888d24s8ms4 --output-dir "$PVRGPU_OUTPUT_ROOT/deqp_groups/g30_df"
```

`run_deqp_level.sh` 會依 module 與 gl-config 分桶，每桶使用各自的 caselist 與結果目錄。

---

## 六、常查的六組展開範例

### 第 19 組 Compute（L3 全跑，195 條）

```bash
grep -E '^dEQP-GLES31\.functional\.compute\.' "$DISC/gles31.txt" > /tmp/g19.txt && script/run_deqp_dynamic.sh --caselist /tmp/g19.txt --keep-going --output-dir "$PVRGPU_OUTPUT_ROOT/deqp_groups/g19"
```

子群：`basic` 41、`shared_var` 136、`indirect_dispatch` 18。
與第 21 組 SSBO 共用 shader 寫入記憶體的路徑，兩組綁在一起動。

### 第 21 組 SSBO（L2 配額 400 條）

```bash
grep -E '^dEQP-GLES31\.functional\.ssbo\.' "$DISC/gles31.txt" > /tmp/g21_all.txt && python3 -c "import sys;c=[l.strip() for l in open(sys.argv[1]) if l.strip()];n=min(int(sys.argv[2]),len(c));print('\n'.join(c[i*len(c)//n] for i in range(n)))" /tmp/g21_all.txt 400 > /tmp/g21_L2.txt && script/run_deqp_dynamic.sh --caselist /tmp/g21_L2.txt --keep-going --output-dir "$PVRGPU_OUTPUT_ROOT/deqp_groups/g21_L2"
```

子群：`layout` 2007（97%，測 std430 成員位址與 stride）、`atomic` 48、
`array_length` 6。layout 幾乎不是繪圖測試，考的是編譯器與 driver 的佈局約定。

### 第 22 組 Multisample（L3 全跑，595 條）

```bash
grep -E '^dEQP-GLES31\.functional\.(texture\.multisample|shaders\.sample_variables|shaders\.multisample_interpolation|sample_shading|multisample)\.' "$DISC/gles31.txt" > /tmp/g22.txt && script/run_deqp_dynamic.sh --caselist /tmp/g22.txt --keep-going --output-dir "$PVRGPU_OUTPUT_ROOT/deqp_groups/g22"
```

先跑第 30 組（基本 MSAA 光柵化）再跑這組，依賴順序是
基本 MSAA → sample 內建變數 → multisample texture。

### 第 20 組 Draw indirect（L2/L3 一般案例全跑，178 條）

```bash
grep -E '^dEQP-GLES31\.(functional|stress)\.draw_indirect\.' "$DISC/gles31.txt" | grep -vE '^dEQP-GLES31\.(functional\.draw_indirect\.compute_interop\.large\.|stress\.draw_indirect\.(drawarrays|drawelements)\.data_over_bounds_with_primcount$)' > /tmp/g20.txt && script/run_deqp_dynamic.sh --caselist /tmp/g20.txt --keep-going --output-dir "$PVRGPU_OUTPUT_ROOT/deqp_groups/g20"
```

221 條 functional 裡有 79 條 `compute_interop` 綁在第 19 組上，
其中 64 條 `compute_interop.large` 只在 L4；其餘 142 條只需要 driver 從緩衝區
讀 draw 參數，不需要 compute。

### 第 5 組 FBO（L2 配額 400 條）

```bash
grep -E '^dEQP-GLES3\.functional\.fbo\.' "$DISC/gles3.txt" > /tmp/g5_all.txt && python3 -c "import sys;c=[l.strip() for l in open(sys.argv[1]) if l.strip()];n=min(int(sys.argv[2]),len(c));print('\n'.join(c[i*len(c)//n] for i in range(n)))" /tmp/g5_all.txt 400 > /tmp/g5_L2.txt && script/run_deqp_dynamic.sh --caselist /tmp/g5_L2.txt --keep-going --output-dir "$PVRGPU_OUTPUT_ROOT/deqp_groups/g5_L2"
```

分層依賴：逐格式 clear 值、逐格式 PIXOUT 打包、逐格式（含整數）貼圖取樣、
多 FBO 編排，再來是 blit 的翻轉縮放轉換 32 條。

### 第 6 組 Fragment ops（L3 全跑，3176 條，建議分片）

```bash
grep -E '^dEQP-GLES3\.functional\.fragment_ops\.' "$DISC/gles3.txt" > /tmp/g6.txt && rm -f /tmp/g6shard_* && awk -v n=8 '{ f = "/tmp/g6shard_" (NR % n); print >> f }' /tmp/g6.txt && for i in 0 1 2 3 4 5 6 7; do script/run_deqp_dynamic.sh --caselist /tmp/g6shard_$i --keep-going --output-dir "$PVRGPU_OUTPUT_ROOT/deqp_groups/g6/shard_$i" & done; wait
```

單行程約 46 分鐘，分片後約 7 分鐘。`depth_stencil` 621 條每條 2.14 秒，是整個目錄裡第二貴的子群。

---

## 七、除錯流程

1. **單獨重現**。用 4.3 的單案例指令，加 `--log-images enable`。
   絕不要重複寫進同一個輸出目錄，driver 附加寫入 counter 檔，
   舊事件與新事件交錯過，讀錯行是最常見的誤判來源。
2. **讀 QPA**。比對 expected 與 actual 影像、shader 原始碼、GL 呼叫順序。
3. **對照 llvmpipe**。同一個 Mesa prefix 下設 `GALLIUM_DRIVER=llvmpipe`
   跑同一條案例。如果 llvmpipe 也是同樣的 skip 或 fail，那不是 PvrGPU 的缺陷。
4. **看 gate 訊息裡的數字**。fail-closed 的檢查如果沒印出數字就先補上數字，
   不要猜。把捆在一起的布林條件拆開，讓它自己說是哪個欄位拒絕的。
5. **修一個環節就重跑，讀新的錯誤**，不要預測下一個環節。
   長修復鏈幾乎每一步都由前一步的錯誤訊息指名。

---

## 八、待實作清單

| 項目 | 內容 |
|---|---|
| `tools/deqp_groups.py` | 24 組擴成 30 組；放寬第 8、19、20、22 組 selector；第 9 組換掉重複的 scissor；第 23、24 組移除 blocked 旗標 |
| `tests/deqp/caselists/L*.txt` | 生成後的 caselist 進版控。dEQP 一旦重建，列舉順序會變，stride 取到的就是另一批案例，回歸比對失效 |
| `tests/deqp/baselines/L*.tsv` | case 對 expected status。現在只有案例數漂移檢查，沒有結果比對 |
| `script/run_deqp_level.sh` baseline 比對 | 分片、執行、合併、失敗非零退出與 gl-config 分桶已完成；尚待加入逐案例 baseline diff |
| 層級包含關係 | L2 要定義成 L1 加 stride 偏移的補集，否則「L2 掛了但 L1 過」沒有意義 |
| `tests/test_deqp_live_ui_source.py`<br>`tests/deqp_live_ui_smoke.py` | 兩處對 24 的斷言 |
| `docs/dEQP_GLES_Test_Execution_Matrix.xlsx` | 6 個不存在的 pattern（見第九節） |

---

## 九、xlsx 矩陣裡不存在的 pattern

對照 xlsx 時會誤判 harness 少跑東西，實際是 xlsx 的名字錯了：

| xlsx 寫的 | 實際存在的 |
|---|---|
| `dEQP-GLES3.functional.scissor.*` | `dEQP-GLES3.functional.fragment_ops.scissor.*` |
| `dEQP-GLES3.functional.instancing.*` | `dEQP-GLES3.functional.instanced.*` |
| `dEQP-GLES3.stress.shaders.*` | `dEQP-GLES3.stress.long_shaders.*` 與 `stress.long_running_shaders.*` |
| `dEQP-GLES31.functional.robustness.*` | 不在這四個 module 裡，屬 openglcts 的 KHR-GLES31，未編譯 |
| `./deqp-gles32`、`dEQP-GLES32.functional.*` | GLES 3.2 功能測試住在 gles31 module 底下 |
| `gles32-master.txt` | 這棵樹只有 openglcts 的 KHR 版本，未編譯 |
