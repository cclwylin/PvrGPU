# Car Chase CPU profile — 2026-09-10

本報告分析 host CPU 的成本，不把 GPU 虛擬 cycles、ALU 指令數或 texture taps 當成 CPU 時間。此次只做量測與診斷，沒有修改 simulator、driver 或 shader 語義，也沒有 commit / push。

## 結論與優先順序

**首要瓶頸是 simulator 的資料搬運與 heap 管理，不是 RenderDoc，也不是先前的 backing `Contains` 掃描。** 建議第一輪先做 USC scratch capacity 重用與 TPU `ReadInto`，再處理 ISS resume-PC 靜態 metadata；ASTC decode reuse / bit reader 排在下一輪。先逐項 A/B，不預先承諾倍數。

37 份有效報告共 **411,243 個主執行緒 stack snapshots**。以下為互斥的 observed exclusive 分類，不是 wall-time 分配：

| CPU 工作類別 | 樣本數 | 主執行緒取樣占比 |
|---|---:|---:|
| memcpy / memmove / 清零 | 94,741 | 23.04% |
| heap 配置／釋放，包括 allocator 下的 madvise | 86,322 | 20.99% |
| Shader ISS 本體、暫存器操作與驗證 | 66,154 | 16.09% |
| 模型 cache / DRAM 實作 | 39,847 | 9.69% |
| ASTC 解碼演算法，不含另列的搬運／配置 | 23,288 | 5.66% |
| Memory checked counter 累加 | 18,454 | 4.49% |
| 其餘類別 | 82,437 | 20.05% |

資料搬運＋heap 管理合計 **44.03%**。按 caller 再拆解，全部仍使用 411,243 作分母：

| 呼叫路徑 | copy / zero | heap 配置／釋放 |
|---|---:|---:|
| USC graphics | 17.23% | 7.37% |
| TPU | 1.57% | 11.66% |
| Simulator 全部路徑 | 22.50% | 20.64% |

此表最後一列包含前兩列，不可相加。`-O3` inlining 下能確認 module/caller 成本，但不能僅從 `memmove` 的符號直接判定所有 samples 都屬於某一個 source object copy；原始碼位置是要驗證的候選，不是完成了 line-level attribution。

補充判讀：

- Strict helper-only 為 2.46%，其中只對全域 copy/zero 貢獻約 0.10 個百分點、對 heap 約 0.11 個百分點；已排除 helper 底下真正呼叫到的 model / Mesa / RenderDoc descendants。`diagnosticObserve` 的 inclusive share 不能全部當 helper overhead。
- 全域 caller context 中 simulator 約 95.85%，RenderDoc 約 0.24%、Mesa/driver/compiler 約 1.33%。Helper 的 full driver-log read 約 1.58%，model report full-read 約 0.42%；可改成增量解析作為低優先的工具優化，但不是主要 frame 瓶頸。
- `Contains`/backing range validation 713 samples，約 0.173%；舊 Manhattan profile 的 54% 問題不是這次瓶頸。
- ComputeShader + ComputeDataMaster caller contexts 共 869 samples，約 0.211%；不是本 capture 的第一優先。不能把此 share 或 compute API envelope 當精確 compute wall time。
- 運行期間 process 大多約 96–97% CPU，其他取樣執行緒是等待路徑，實際主負載接近一顆 CPU core。這不代表可直接平行所有 draw：資源依賴、cache/counter 順序與 SystemC 狀態需要另外設計。

### 不同 draw 的瓶頸不同

經 visibility containment 驗證的 phase windows 中，Draw 102–352：ISS 約 9.41%、ASTC 約 12.46%；Draw 353–359：ISS 約 30.30%、copy/zero 約 24.85%、heap 約 19.16%。Draw 370–382 的 9 個純 phase windows（102,461 samples）則是 heap 約 28.78%、copy/zero 約 25.98%、ISS 約 14.07%、ASTC 0 observed samples。

單獨落在 Draw 358 內的 sample-024（11,564 samples）：ISS 類別 45.48%，`ExecuteFragmentPcoValidated` 精確函式 self 35.65%，copy/zero 22.34%，heap 11.71%；其中 USC copy 21.52%。這支持針對重 ALU draw 繼續改善 ISS，同時仍要處理 continuation 搬運。這些是 pure-window 的取樣比例，不是該 draw 全時間的精確百分比，也不能拿多個重疊 phase 相加。

## 完成狀況與耗時

完整 replay 正常結束：**1097.055 秒，即 18 分 17 秒**。這是 profiler 加上 strict helper 的 process wall time，包含最多約一輪 polling 的退出觀察延遲，不是「完全拔掉 audit」後的最佳速度。

- 395 個 draw ranges、1 個尾段到 event 6649；397 個 GL API observation scopes 乾淨。
- 391 組 graphics model reports，400 個 physical graphics submissions；413 組 graphics API pairs，其中 391 draw commands、22 clears；另有 5 組 compute。這些數字的定義不同，不能互換。
- 原生工作量、所有非 `artifact_png` 的 model 欄位、aggregate counters、compute 欄位、ordered boundaries 均與 frozen baseline 完全一致；所有 pool allocations/releases 平衡。
- 最終 1920×1080 raw 與 PNG 都與 baseline byte-exact。Raw 是 canonical RGBA8 readback，8,294,400 bytes，不是把 RGB565 附件直接當作 packed565 bytes。
- 兩份 runtime stderr 為空；compile-only initialization 的既有拒絕紀錄與 baseline 相同，不宣稱整份 driver log 都無 error/decline。
- 輸出差異符合設定：44 份附件 snapshots 降成 1 份；391 個 per-model PNG artifact 欄位／capture markers 關閉。工作量沒有減少。
- 這證明量測沒有改變既有 native 結果，**不是宣稱與 llvmpipe pixel-exact 或 Car Chase 全項 correctness PASS**。

共 38 次 sampler 嘗試：0–36 成功產出報告，36 是程序退出前的較短窗口；最後 37 在程序退出邊界 attach 失敗（exit 255，無報告），保留其 stderr 與 manifest，不納入樣本分母。Replay 自身 exit 0、未 timeout、所有 pinned inputs 的 SHA-256 不變。

### 最耗時的 draw

| Draw index | Event | Range elapsed 估計 | 可見時間界限 |
|---:|---:|---:|---:|
| 379 | 6507 | 72.867 s | 72.815–72.919 s |
| 358 | 5899 | 72.165 s | 72.112–72.218 s |
| 293 | 4518 | 65.337 s | 65.282–65.391 s |
| 381 | 6533 | 64.885 s | 64.831–64.940 s |
| 380 | 6522 | 61.610 s | 61.555–61.665 s |
| 357 | 5850 | 61.312 s | 61.259–61.365 s |
| 356 | 5844 | 59.003 s | 58.952–59.054 s |
| 115 | 1116 | 52.953 s | 52.900–53.006 s |

### 各區間的 range 耗時合計

| Draw 區間 | Range 數 | 各 range 秒數相加 |
|---|---:|---:|
| 0–101 | 102 | 76.753 s |
| 102–352 | 251 | 401.148 s |
| 353–359 | 7 | 279.166 s |
| 360–369 | 10 | 5.344 s |
| 370–382 | 13 | 315.614 s |
| 383–394 | 12 | 6.232 s |

395 個 draw ranges 合計 1084.257 s。初始化的可見時間界限是 8.295–8.352 s；尾段 range 估計 0.110 s。不可把 range totals 或 API totals 再加到 process wall，也不可把短 range 的 0 秒估計理解為零工作。

## 量測條件

- Capture：`/Users/linwanyi/Downloads/_Codex/GPU_TestPatterns/5.gl_4/recorder/trace/gl_4_capture_1.rdc`，SHA-256 `7c10e7ba74f0ca4dbd452d741644d72b35a0bddd11844875d48e212167934867`。
- 執行所有 ordered draw ranges 0–394，以及最後 event 6649 的 trailing range；Draw 編號是 helper 的 zero-based RenderDoc action index，不能直接當作內部 physical DrawList 編號。
- 使用上次完整 Car Chase 成功執行的 frozen driver17 / model15 / strict helper / RenderDoc。不是目前 checkout 中尚未提交的 GL5 修正版。
- Model：Release、`-O3 -DNDEBUG`、`PVRGPU_ENABLE_DIAGNOSTICS=0`、沒有 fast-math。Gallium private driver 的 compile commands 是 `-O2`，不能稱為整套都 `-O3`。
- 關閉逐 model PNG；只讀回最後附件並輸出最終 PNG，保留 counters。strict helper 的逐 range GL 檢查、`glFinish`、model evidence 複製及 inline audit 仍在量測範圍內。
- OpenCapture 使用既有 compile-only loading，之後恢復 capture 初始資源，再執行完整 native ordered replay；沒有用 llvmpipe 資源替代，也沒有跳 draw。
- 沒有使用不完整的 attachment snapshot 當 whole-state replay checkpoint；此次沒有可供 Car Chase/D16 使用的完整 checkpoint。
- Intel i5-10600，6 physical / 12 logical cores，32 GiB RAM。未停止其他使用者程序；系統背景負載與 profiler 本身可能影響耗時。

目前 repo 位於 `/Users/linwanyi/Downloads/_Codex/PvrGPU`；既有 runtime / reference / 本次大型結果仍在平行的 `Working/PvrGPU` 下。

## 分析方法與解讀限制

一個實際 replay process，使用 macOS `sample`，間隔 2 ms；第一個窗口要求 5 秒，其餘要求 30 秒，直至 replay 結束。保留原始 call tree、sampler 狀態與輸入 hash。

CPU 分類的分母是主執行緒觀察到的 stack snapshots，**不是精確 CPU cycle 或整個 frame 的 wall-time 百分比**。其他執行緒另外檢查，不能把等待中的執行緒加進分母。Exclusive 分類以每個節點減去直接子節點的樣本數計算；每個樣本只屬於一個 mechanism。Inclusive function / caller tree 有重疊，不可相加。

約每 50 ms 讀取完整追加的 journal / native API 行，保存 byte offset 及可見時間上下界。Draw 耗時估計是完成與開始可見區間的中點差，不是修改 helper 插入的精確執行時鐘；同一輪觀察到的短 draw 可能估計為零。Native API 以實際 driver byte offsets 對應 range，而不是猜測 cached range label。

單一 draw 的 pure sample window 必須整個窗口落在 `range_start.upper` 與 `range_end.lower` 之間；native API 的純窗口另外做相同檢查。只看窗口前後同名 draw 不足以證明 pure。窗口程序耗時也包含 symbolication，不當成實際採樣秒數。

逐 draw elapsed 包含 replay、同步 finish、helper 檢查與 evidence 複製。群組秒數是各 range 耗時估計相加，不包含 range 之間的空檔、初始化與最後輸出等，也不等於 process wall。最終獨立 audit 在 replay 計時完成之後才執行。此次 profiler 耗時不能直接與舊 unprofiled 或不同 readback/audit 設定的數據宣稱加速比。

重要：graphics `systemc_api_submit/done` 只包住 enqueue API，pending graphics 在後續 finish/readback 才真正執行，不能拿其間隔當 graphics 執行時間。Compute API 是同步 envelope，但也可能先 flush 前面的 graphics，不是純 compute CPU 時間。這些行在 counter logger 中每次寫完都 close file；其 byte span 用來證明順序及範圍，不取代 draw envelope 或 sampled call tree。

## 原始碼對應的優化方向

### TPU：小量讀取避免 heap allocation 與整條 cache line 複製

`src/systemc/texture/texture_unit.cpp:2109` 每個 texel tap 呼叫 `GpuMemorySystem::Read`。`src/systemc/memory/gpu_memory_system.cpp:165` 為結果建立 vector，`:198` 取得 cache line，`:201` 再複製需要的片段；`src/systemc/cache_mmu/cache_array.cpp:274` 命中時仍複製整條 cache line。

候選修改是 caller-owned 固定小 buffer / `ReadInto`，及 cache-line range-copy API。目標是移除 host 容器成本，不是省略 texture access。每個 tap 的 memory transaction、LRU、hit/miss、eviction/writeback、counter、跨 line 行為及拒絕條件都必須不變。

### USC：重用 scratch buffer，減少 continuation / context 整批搬運

`src/systemc/shader/usc_cluster.cpp:1452` 每個 texture batch 建立多個 vector；`:1485` 存入 pool；`:1514` 取回 requests、responses 與 continuations；`:1530` 複製三份 pending arrays；`:1599` 還有 per-lane execution context copy。

先做容器 capacity 重用，再有針對性地減少全量拷貝。不能只把所有物件改成借用 reference：跨 FIFO / wait 的存活期、request identity、generation、helper/derivative lane、divergence、ownership 與失敗清理必須保留。

`MemoryPool::Release` 已保留 vector capacity，因此不能把每一次 pool allocation 都說成新的 heap allocation；仍要區分 resize/清零、實際配置與資料搬運。

### Fragment ISS：把 immutable resume-PC 資料提前整理

`src/systemc/shader/pco_iss.cpp:9460` 的 prepared program 已快取 validation、signature 與 must-write CFG requirements，不是每個 fragment 都從頭 decode。

剩餘候選在 `:9672` 的 prefix scan，以及 `:9723` 的 enclosing-backedge scan。可在 immutable prepared program 建立時整理每個 resume PC 的靜態 metadata；真正 continuation 的 masks、predicate、pending response、loop state 與身份檢查仍須執行，拒絕語義不能放寬。

### ASTC：擴大正確的 decode reuse，改善 bit reader

`src/systemc/texture/texture_unit.cpp:2080` 現有快取只保存單一 request 內最近一次解碼的 16-byte block，下一個 lane/request 會重新開始。

可評估有限大小的 host-only decoded-block memo：key 必須包含完整 16 bytes、block footprint、sRGB 模式；查詢必須放在既有每個 tap 的模型 memory read、response validation 與 counter 更新之後。相同位址但內容改變必須 miss，hash 命中必須驗證全部 key；保留錯誤顏色與拒絕語義，不快取錯誤路徑以逃過檢查。尚未量到跨 request reuse hit rate，不能預估倍數。

`src/systemc/texture/astc_decoder.cpp` 的 `BlockBits`、`BlockBitReader::Next` 可考慮 bounded word/shift/mask 實作，保留截斷、bit order、reverse、跨 word 與非法範圍行為；`InfillWeights` 的 geometry/coefficients 可考慮按 footprint/grid 預先計算，維持整數 rounding。

目前 checkout 的 ASTC output rounding 與 texture LOD log2 跟 frozen model15 有 correctness 差異；這些尚未納入本次效能量測，不能視為已驗證的加速修改。

### 次要候選與暫不優先事項

- `src/systemc/common/functional_types.cpp:40` 的 `SrgbChannelToLinear(uint8_t)` 輸入只有 256 種，現有非線性分支使用 double `pow`。可評估 exact 256-entry table，驗證全部輸入的 float bits / rounding contract；不是改成近似曲線。
- `GpuMemorySystem` 的 checked counter accumulation 有 CPU 成本，但不能刪除 counters 或 overflow 檢查來宣稱等價加速。
- `src/systemc/shader/compute_shader.cpp:293` 每個 step 的 task state ReadPod/WritePod 是獨立候選；是否優先必須由這一個 capture 的實測 compute 成本決定，不能只看結構很大。
- JIT 不會自動消除 TPU 解碼、模型 cache/DRAM、USC 搬運與 allocator 成本。本次沒有實作 JIT，也沒有可支持 JIT 倍數的 A/B 數據。

## 後續 A/B 驗證流程

1. 固定 capture、driver/model/helper、編譯選項、環境、輸出、audit 政策與 action boundary，先建立不開 profiler 的基準。
2. 一次只改一個 host optimization，先跑針對性 unit / invariant tests，再比較完整 frame；沒有經驗證的 whole-state checkpoint 時，不可從 draw 148 等中間 action 空跑來代替。
3. 要求 native API / 工作量 counters / cache與DRAM事件 / pool balance 不變，最終 canonical raw 必須與同版 baseline bit-exact；PNG 另驗證。不能只憑 exit 0 或 PNG 看起來相似。
4. ASTC 額外覆蓋各 footprint、linear/sRGB、合法與錯誤 block、cold/warm cache、同址改寫；USC / ISS 額外覆蓋分支、loop、derivative/helper lane 與 continuation 拒絕案例。
5. 在相同背景條件交錯 A/B 多次，報告每次值、中位數及差異；重新 profile 看瓶頸是否移轉。不要以本次 stack share 直接承諾加速倍數。

## 結果位置與固定輸入

- 量測根目錄：`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/carchase-cpu-profile.u9hy6Z`。`manifest.json` 記錄 PID、命令、每次 sampler 結果、執行前後 hash；`observations.jsonl` 保存時間界限與原始事件；`timing.json` 保存所有 range/API 的對應及估計。
- CPU 分析：`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/carchase-profile-analysis.dHEBOS/final/`，含 `profile.json`、`profile.md`、`context-breakdown.json`、分析 receipt / tests；人讀說明以 `REPORT-v3.md` 為準。前版的函式 prefix/lambda self 標籤、helper 名稱與 model/driver 編譯選項標籤已訂正，統計資料不變。同層 `final-conditioned/groups.json` 保存 visibility-certified phase / draw 分組。早期 `pass-*` 是部分進度，不可替代 final 統計。
- 原始 replay：上述目錄的 `replay/`；最後畫面 `draw-394-event-6649-color0.png`；最終 raw SHA-256 `fed53ff416f52ca89977c858f46a3953c69f1d92041c8b7d6d5a2a53e0397d2f`，PNG SHA-256 `c4d7183218bb0e539d78b62790d9c69fc76110b72e5571681b47161f36b1d8ef`。
- 獨立工作量／影像 audit：`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/carchase-profile-audit.usr5D4`。`native-audit.json` 與 `report.json` 包含逐範圍與欄位比對。這是在 replay 完成計時後產生。
- Frozen baseline：`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/carchase-debug.t3s25m/native-v17-depth17-s32-full6649`。
- Bridge：`carchase-debug.t3s25m/model-lod-invariant-v15/build/lib/libpvrgpu_systemc_bridge.dylib`；SHA-256 `f2cd648286a2c848fd1976137bdf4f247f9553497171ee6dcaff64281bbac243`。
- Gallium：`carchase-debug.t3s25m/driver-cube-depth-v17/install/lib/libgallium-26.2.1.dylib`；SHA-256 `eaa698618540f0bc82d829c961d80798002f1135493e290fbb7bf3cfc27f2f94`。
- Helper SHA-256 `eb1c1b5b2efc3ac7078f165efed75e91fd9a9345b4452f105119755457cf08e1`；RenderDoc SHA-256 `979a494d4d1ab7a090649b45414c54490841d44401ac676ad55e8d3d49df5e05`；完整絕對路徑見 `replay/receipt.json`。

使用 `profile_run.py` 的方法再次量測時，須建立新的輸出目錄，不覆寫此次證據；分析程式與 runtime 是不同產物，分析成功不取代 replay / hash / native audit 成功。
