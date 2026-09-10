# PvrGPU Speed Tune

更新日期：2026-09-10。

目的：以量測決定 simulator 的效能優化順序，在不改變 native shader、GPU 工作量、memory/cache 語意及最後畫面的前提下，減少 host CPU 的額外成本。

## 1. 已完成的基準與優化

測試 pattern：`gl_manhattan31_capture_1.rdc`。

| 版本 | 完整 frame 時間 |
|---|---:|
| Release O3、診斷關閉、保留 counters 與最後 PNG | 1714.257843 秒（28 分 34.26 秒） |
| 加入 backing 範圍驗證快取 | 814.086443 秒（13 分 34.09 秒） |

本次前後比較約 **2.11 倍加速、耗時減少 52.51%**。這是單次未掛 profiler 的前後對照，不是多次量測中位數；計時含程序啟動、最後 readback／PNG 與 teardown，排除建置、檔案 hash 與事後稽核。

已完成的 `DramAddressSpace::Contains` 優化：

- 固定 64 槽，只快取完整驗證成功的 page span；未命中仍做原本逐頁檢查。
- 不快取失敗；零長度、overflow、缺頁與實際讀寫的檢查保留。
- 一般 Write 只新增頁或更新內容，既有成功證明仍有效。
- Copy／move／reset 正確清除舊證明；未來若新增移除頁面或 remap，必須加入失效處理。
- 這是 host 有效性驗證快取，不是 texture 取樣結果快取，也沒有省略 GPU memory transaction。

相關提交：`3e46844`（診斷／PNG 開關）、`4f72708`（backing 驗證快取）。

驗證：41 項 CTest、267,536 次差分檢查與 ASan／UBSan 通過。完整 231 個 draw ranges 到 event 8299，219 graphics、14 clears、5 compute、12 zero-work draw actions 均與原 native baseline 一致；48,086 個數值欄位與最後 raw／PNG 完全相同。這不是與 llvmpipe pixel-exact 或 API conformance 的宣告。

## 2. 下一步先量新版 CPU profile

先前的 profile 是「加入 Contains 快取之前」的版本，共 61 個有效取樣窗口、678,279 個主執行緒樣本：

| 工作 | 舊版樣本占比 |
|---|---:|
| Backing 範圍驗證 Contains | 54.81% |
| 記憶體複製／清零 | 12.43% |
| Heap 配置／釋放 | 10.32% |
| 可見 ISS 本體／暫存器／驗證 | 7.89% |
| 其他 memory／cache／DRAM 模型工作 | 6.60% |

這些是 sampled-stack 占比，不是精確 CPU cycles，也不是現在 814 秒版本的占比。不能直接刪掉 Contains 的分母後，把其餘比例當作新量測結果；O3 inline 也會讓部分工作歸在可見父函式。

下一輪應先對 `4f72708` 或明確記錄的後續版本重新取樣，再按新的熱點排序。取樣執行與效能 benchmark 分開；不能拿帶 profiler 的時間當最佳效能數據。

## 3. 第一個修改候選：TPU 小量讀取避免配置與整條 line 複製

本節保留前期設計；Car Chase 第一輪已實作 `ReadInto`，結果見第 9 節。

### 現況

每次 texture tap 讀取 4／8／16 bytes，會建立回傳用的 vector。SLC hit 時，`CacheArray::ReadLine` 仍把整條 128-byte cache line 複製出來，之後再抽出需要的 bytes。

### 建議

- 增加 caller-owned `ReadInto` 類型介面，TPU 一般 texel 路徑使用固定大小 buffer。
- Cache 層提供指定 byte range 的複製，避免為了少量資料先建立整條 line 的副本。
- 其他大小或特殊格式使用正確的通用路徑，不把「最多 16 bytes」當作所有 memory client 的限制。

### 必須保留

- 每個 tap 的順序、每次 cache lookup／LRU 更新、hit／miss、dirty eviction／writeback 與 counters。
- 跨 cache line 讀取、位址 overflow、缺頁、錯誤回傳與既有例外順序。
- ASTC／壓縮格式、特殊格式及不同 memory mode 的正確行為。

不可透過快取最終取樣答案、略過 SLC 或合併原本分開的 GPU 存取，來取得不同工作量的「加速」。這一項改動範圍相對集中，適合作為新版 profile 後的第一個實作候選。

## 4. 第二個候選：減少 USC 暫停／恢復的資料搬運

Car Chase 第一輪已實作下列第 1 階段 capacity 重用；第 2／3 階段尚未實作，結果見第 9 節。

### 現況

Shader 等待 texture 回應時，會建立 batch vectors，把 continuation 複製至 MemoryPool，再載入 saved vectors；回應處理又把整批 pending arrays 複製成 `next_*`，並複製各 lane 的 execution context。

Continuation 包含 temporary registers 與 loop state；execution context 還包含其他輸入資料。每次 texture round-trip 都做這些工作，會增加 memcpy、清零與 heap 成本。

### 建議分階段實作

1. 先把 scratch vectors 移出內層迴圈，重用容量，保留現有完整驗證。
2. 再評估只更新實際受影響的 lanes，減少整批 `pending_* → next_*` 複製。
3. 將 immutable program／輸入資料與 mutable continuation 分離，避免隨每次 resume 複製不變內容。

不能直接移除 request／response identity、generation、排序及 continuation 比對。必須測試 branch divergence、quad derivatives、helper lanes、錯誤時的狀態與 payload 清理、FIFO 生命週期；跨 wait 或 pool 重新配置時，不得留下失效指標。

## 5. 第三個候選：預先計算靜態 resume 驗證資訊

### 現況

`ExecuteFragmentPcoValidated` 恢復 continuation 時，會掃描指令前綴及相關 branch／loop 區間，推導合法的 temporary written masks、pending request 等條件。

### 建議

評估在 program prepare 時，依 resume PC 預先計算只取決於 owned immutable program 的驗證資訊，存於 prepared program 的私有表格。執行時查詢這些 metadata，再驗證實際 continuation，不反覆掃相同指令。不可相信外部提供的「已驗證」旗標或僅憑 hash 判定安全。

保留 runtime predicate、實際 written masks、pending request、program identity、resume PC、loop state 與輸出邊界檢查。不能把「靜態上可能寫入」誤當成「這次執行必定寫入」，也不能移除非法 continuation 的拒絕行為。

必須保留原本的拒絕時機與順序：預先計算若發現無效前綴，可記錄為 metadata，但不能提前拒絕原本未使用或不可達的 checkpoint。

這不是每個 pixel 都重新 decode shader 的問題；目前 `PcoPreparedFragmentProgram` 已將 `ValidateFragmentExecutionProgram` 與 `FragmentProgramSignature` 提前處理。優化目標是 resume 時仍重複推導的部分，不是再做一次已完成的 prepare 優化。

## 6. 後續再評估

- **Compute task 搬運**：每個 instruction group 目前 `ReadPod → StepComputeTask → WritePod` 搬移整份 task。若新版 profile 仍顯示此處顯著，再研究穩定的 resident task storage；公平排程、mutex／barrier 可見性及 checkpoint 狀態必須保留。
- **JIT／SIMD**：在剩餘 ISS 成本有明確量測後再決定範圍。需保留指令語意、浮點精度、動態 counters、memory 邊界與同步，不能先承諾整個 frame 幾倍加速。
- **完整 Texture Cache → L2 路徑**：屬於模型精度與架構完善，與 host 效能優化分開驗證。目前實際 texture 讀取使用共享 GpuMemorySystem／SLC／DRAM backing，不是完整的獨立 Texture Cache、USC-L2 模組鏈。補齊硬體層次不等於 simulator 一定更快。

## 7. 每輪優化流程與完成門檻

1. 凍結 baseline：記錄 capture、driver、helper、runtime hashes、編譯 flags、環境、memory mode 與初始化方式。
2. 量 CPU profile，定位實際熱點；區分 graphics、compute 與 mixed ranges，不以 GPU virtual cycles 代替 host CPU 比例。
3. 一次只改一類成本，補上正常、邊界與拒絕案例；必要時使用差分 oracle／sanitizers。
4. 在相同條件下，不掛 profiler 重跑完整 frame；不得跳過 draw／compute，不以已知 capture／shader 特判，不替換結果或 counters。
5. 執行結束後，獨立比對完整工作量、counters、raw／PNG；稽核不放進 timed replay。
6. 同時報告完整 frame 與主要區段時間。Compute 與 graphics 共用的範圍不能當作單一 shader stage 的倍率。
7. 記錄單次或多次量測方法、限制與結果；再量 profile 決定下一步。更快但語意／counters 不同的版本，不算通過此輪純效能優化。

前期建議順序：**新版 profile → TPU ReadInto → 完整 A/B → USC 搬運 → 靜態 resume metadata → 再評估 JIT**。已完成項目與新版順序見第 9 節；尚未實作的項目不寫成已完成成果。

## 8. 程式定位與本機證據

下列是本機絕對路徑；在其他 checkout 請依檔名及函式名稱重新定位，行號可能隨修改移動。

- [Backing 範圍驗證快取](</Users/linwanyi/Library/Mobile Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/memory/dram_address_space.cpp:107>)
- [TPU texel 讀取](</Users/linwanyi/Library/Mobile Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/texture/texture_unit.cpp:1874>)
- [GpuMemorySystem::Read](</Users/linwanyi/Library/Mobile Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/memory/gpu_memory_system.cpp:158>)
- [Cache line 回傳複製](</Users/linwanyi/Library/Mobile Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/cache_mmu/cache_array.cpp:273>)
- [USC continuation／pending state 搬運](</Users/linwanyi/Library/Mobile Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/shader/usc_cluster.cpp:1484>)
- [Resume 驗證](</Users/linwanyi/Library/Mobile Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/shader/pco_iss.cpp:9441>)
- [Compute task 搬運](</Users/linwanyi/Library/Mobile Documents/com~apple~CloudDocs/Codex/PvrGPU/src/systemc/shader/compute_shader.cpp:287>)
- [舊版 CPU profile](/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/manhattan-cpu-profile.xpSzNB/PROFILE.md)
- [Contains 優化完整結果與重跑方式](/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/manhattan-contains-candidate.tX5mpL/RESULT.md)
- [Native counters／影像稽核](/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/manhattan-contains-candidate.tX5mpL/final-native-audit/report.json)

本機 tmp 目錄不是永久發佈位置；若搬移或清理，需一併保存 provenance、原始量測與稽核證據。此文件的效能數字不得在更換 runtime 後直接沿用。

## 9. Car Chase 第一輪：ReadInto＋USC scratch（2026-09-10）

本輪依 [Car Chase 新版 CPU profile](docs/CARCHASE_CPU_PROFILE_20260910.md) 實作 TPU 小量 `ReadInto` 與 USC texture-round scratch capacity 重用。只動六個 simulator production files，沒有改 PCO driver/compiler、API/ABI、ISS 或 shader binary；工作目錄原有其他修改保留。

| 完整 frame，未掛 profiler | Process wall |
|---|---:|
| Frozen Car Chase 原版 | 1009.543620 秒（16 分 49.54 秒） |
| 原版＋ReadInto／USC scratch | 790.485868 秒（13 分 10.49 秒） |

此單次 A→B 觀察到 **耗時減少 21.70%、1.277 倍速度**，節省 219.057753 秒。這是兩項一起套用的效果，沒有分別量出 TPU／USC 各自倍率；baseline 曾短暫與小型編譯／離線自測重疊，不稱為乾淨重複中位數或最佳數據。

兩邊 model 都是 Release O3、diagnostics OFF、無 fast-math；沿用 O2 driver17。只輸出最後 PNG、保留 counters，但 strict helper 的逐 range 檢查／finish／inline audit 仍保留。完整 draw 0–394＋尾段6649，不用不完整 snapshot 跳過前段。所有 model 欄位／counters 與最終 raw／PNG 都與原 native baseline 相同，不代表 llvmpipe pixel-exact。

Driver 日誌保留 19 筆非 indexed draw 未使用的 `index_bias` 診斷差異，已逐筆用 frozen source 確認；strict log equality 仍為 MISMATCH，沒有改 logger 或放寬比較器欄位。不能宣稱整份 driver log 完全相同，詳見完整報告。

Normal CTest 166/167 通過，剩下一項 tessellation ISS 舊負測試已在未修改的 frozen 原版重現；memory/cache ASan＋UBSan 和 USC 47/47 UBSan 通過。更完整的來源、測試、量測限制與 native 稽核記錄見 [第一輪優化報告](docs/CARCHASE_SPEED_ROUND1_20260910.md)。量測期間未 commit／push。

下一步先量這個候選的新 CPU profile，再決定 ISS resume 靜態 metadata、更多 USC copy reduction、ASTC decode reuse 或 JIT 的優先順序；不能用優化前的樣本占比直接估計下一輪倍率。

## 10. Manhattan：同一組 ReadInto＋USC scratch 量測（2026-09-10）

沿用第 9 節同一組 frozen A/B bridge、driver17、RenderDoc 與 strict helper，對 `gl_manhattan31_capture_1.rdc` 重跑完整 231 個 ordered draw ranges＋tail 至 event 8299，未掛 profiler。兩邊均已有 Contains 快取，B 另外包含 ReadInto＋USC scratch。

| 完整 frame | Process wall |
|---|---:|
| 本輪 frozen 原版 A | 875.458080 秒（14 分 35.46 秒） |
| 同版＋ReadInto／USC scratch B | 820.198838 秒（13 分 40.20 秒） |

單次 A→B **節省 55.259242 秒，耗時減少 6.31%，速度為 1.06737 倍**，不是最佳值或重複中位數。A/B 計時期間沒有並行本任務的 build、unit test、大型稽核或 CPU sampler。此 capture 的改善小於 Car Chase 的 21.70%；尚未量此候選的新 CPU profile，不能據此確定剩餘瓶頸或拆分 TPU／USC 各自貢獻。

兩邊皆 exit 0，全部 1080 個 pinned inputs 前後未變；219 組 model triplets、233 組 graphics API pairs（含 14 clears）與 5 組 compute pairs 完整。所有 model/counter、compute 欄位及最後 raw／PNG 精確相同。

嚴格 driver log 比較仍保留 **21 項差異**：16 項非 indexed draw 未使用的 `index_bias` 診斷值，以及 5 項 Draw0 host `cbuf0` 指標位址，皆逐筆以 frozen source 確認。未改 logger、未放寬 comparator；完整 log 嚴格結果仍是 MISMATCH，不能冒稱整份 log 完全相同，也不代表 llvmpipe pixel-exact。

第 1 節歷史 **814.086443 秒**使用較舊的 driver/bridge/RenderDoc 與不同 helper 稽核政策，不能拿來當這輪 B 的直接基準。該歷史結果保留，但本輪優化效果只由上述新 A/B 計算。

完整 pins、計時邊界、初始化／zero-work 驗證、差異來源與重跑方式見 [Manhattan 量測報告](docs/MANHATTAN_SPEED_ROUND1_20260910.md)。量測期間沒有新增 production 修改、改動 PCO driver、覆寫預設 runtime 或 commit/push；沿用前輪 build/test 證據，新增的是此 capture 的完整 A/B 量測及獨立稽核。
