# Car Chase simulator 優化：ReadInto 與 USC scratch reuse

日期：2026-09-10。第一輪 host-only 優化完成：此對完整 frame 從 1009.54 秒降為 790.49 秒，耗時減少 21.70%。所有 model 欄位、counters 與最終 raw／PNG 一致；driver 日誌仍有 19 筆已逐項解釋的既有非 indexed 診斷差異，strict log equality 未通過，詳見下方。

## 本輪實作範圍

只改六個 simulator production files：

- `src/systemc/memory/gpu_memory_system.{h,cpp}`：新增 caller-owned `ReadInto`。
- `src/systemc/cache_mmu/cache_array.{h,cpp}`：新增 `ReadLineInto`，命中時只複製要求的 line span。
- `src/systemc/texture/texture_unit.cpp`：目前接受的 2／4／8／16-byte texel 與 16-byte ASTC block 使用 inline buffer。
- `src/systemc/shader/usc_cluster.cpp`：同一 resident chunk 的 texture rounds 重用 scratch vector capacity。

不改 PCO driver/compiler、driver↔model API/ABI、PCO ISS、ASTC decoder、texture filter/LOD。工作目錄原有 GL5／driver／正確性修改保留；量測期間沒有 commit 或 push。

### TPU / memory

舊 `Read` 和 `ReadLine` 保持原樣，作為獨立差分 oracle。新增介面仍經過原 cache access path，保留每個 tap 的 lookup、LRU、hit/miss、dirty eviction/writeback、memory counters、cycles、缺頁／範圍拒絕。

Warm cache hit 不再建立 owned read payload 或複製完整 128-byte line；資料直接複製到 caller 的小 buffer。跨 line 仍按原順序逐 line 存取，不合併 transaction。Direct/bypass 與 cold fills 仍保留 backing vector，**不是所有 memory mode 零配置**。

固定 16-byte 上限只屬於目前 TPU 已接受的格式，不是通用 `ReadInto` 的長度限制。FIFO 路徑仍取得 owned payload，沒有借用跨 SystemC wait 的 cache／pool 指標。ASTC 的每次 modeled read、payload validation 與現有單-request decode cache 均保留。

### USC

batch lanes/requests/continuations、回傳的 owned arrays 與 `next_*` buffers 移到 texture loop 外。只有實際發生 SMP 才 reserve；每輪 clear/resize/assignment 重用容量。成功完成驗證、pool release 與 state publication 後，才以 swap 交換 pending/next。

仍完整拷貝 transactional pending state，並逐 lane 執行原有 identity、generation、response order、continuation、helper/derivative 檢查。這輪只減少 host heap growth 與重複初始化，**沒有消除全部 continuation/context memcpy**，也沒有改 modeled pool allocation/release。

## 編譯與來源隔離

效能候選從 frozen model15 的 528-file source snapshot 複製，只覆蓋上述六檔；沒有將目前 dirty checkout 整份混入。獨立來源檢查確認六檔精確差異及 61 個 bridge/memory-core compile commands，除 source/build 絕對路徑外與 baseline 相同。

- Model：Release，`-O3 -DNDEBUG`，diagnostics OFF，無 fast-math。
- Gallium driver：沿用 frozen driver17，`-O2`，沒有重建或修改。
- Compiler：LLVM Clang 22.1.8。
- Baseline bridge SHA-256：`f2cd648286a2c848fd1976137bdf4f247f9553497171ee6dcaff64281bbac243`。
- Candidate bridge SHA-256：`75463f6584384d57e094c507ba2e0a540a33d6aeb3727ff92b7fa11754032ac5`。
- Driver SHA-256：`eaa698618540f0bc82d829c961d80798002f1135493e290fbb7bf3cfc27f2f94`。

目前 repo 的 43 個 tracked `src/gallium` 檔案，優化前後 aggregate SHA-256 相同：`f8b11626b6c46dfabe44d77324b1e3e5b33ca224def75a0e1c35e7ffa79e1992`。這是保留原有 dirty driver bytes 的證明，不是聲稱整個 worktree 沒有 driver diff。

## 針對性測試

測試使用目前 checkout 的獨立 Release build，與 frozen runtime 的完整 frame A/B 分開解讀。

- 最後一輪 CTest：**166/167 通過**。涵蓋 memory/cache、TPU direct/cache/bypass、gather/array-shadow/cube-array/multisample、USC continuation/residency、prepared fragment/branch、derivative pipeline、ASTC、compute/tessellation texture。
- Memory/cache standalone ASan＋UBSan：兩項通過，無診斷。未啟用 LeakSanitizer，不宣稱 leak sanitizer coverage。
- USC continuation/residency UBSan：47/47 通過；包含真正的 4→2→4 FIFO batch、10/16-round sequence、request limit/boundary、四種 warm scratch payload corruption 及失敗 cleanup。
- Warm ReadInto allocation observer：256 次讀取（包含跨 line）零 heap allocation；舊 Read positive control 能偵測到配置。
- 差分測試保留舊 API oracle，涵蓋三種模式的 bytes/statistics、混合讀寫、HostWrite invalidate、dirty eviction、flush、失敗後狀態與 guards。

測試修正：既有 derivative-only harness 錯把合法 position-CF4 ABI/task/bank 清為 CF0，且缺少進入 UscSlot 前的 post-PDS counters。本輪只修正 fixture 並增加 CF4、tasks=1、DOUTI=2、load bytes=16 的斷言，原有 derivative/helper output 與零 texture traffic 斷言保留。兩次中間失敗的 XML 保留；沒有修改 production 防呆。

尚未通過項目：`tessellation-texture-iss-unit` 在 6472 checks 後缺少預期拒絕。使用未修改的 frozen model15 測試、headers 與 bridge 已逐字重現。測試仍將 9 textures 的 ABI mutation 當成非法，但既有上限已是 12；本輪不改此不相關測試／ISS，不將它列為通過，也不宣稱全專案 CTest 綠燈。

## 完整 frame A/B 條件

Capture：`GPU_TestPatterns/5.gl_4/recorder/trace/gl_4_capture_1.rdc`，SHA-256 `7c10e7ba74f0ca4dbd452d741644d72b35a0bddd11844875d48e212167934867`。

相同 frozen driver/helper/RenderDoc，GLES 3.1、compile-only initialization、shader cache OFF。完整 ordered draw 0–394，再跑尾段 event 6649；沒有 snapshot resume、跳 draw 或替換 llvmpipe 資源。

`PVRGPU_SYSTEMC_DISABLE_PNG=1`，只最後 attachment readback/PNG，保留 counters。兩邊都不掛 profiler／journal observer；strict helper 的逐 range GL observer、finish、evidence copies 和 inline audit 仍保留。

計時從 helper `Popen` 到 `wait` 完成，包含初始化、執行、最後 readback/PNG、teardown；排除 build、setup/hash 與事後 audit。不是純 shader 時間，也不是完全移除稽核後的最佳速度。

| 計時範圍 | 原版 | ReadInto＋USC scratch |
|---|---:|---:|
| Process wall | 1009.543620 s（16 分 49.54 秒） | 790.485868 s（13 分 10.49 秒） |
| Child user CPU | 879.777767 s | 704.568022 s |
| Child system CPU | 128.765234 s | 84.593416 s |

這一對觀察到少 **219.057753 秒（3 分 39.06 秒）**，耗時減少 **21.6987%**，速度為 **1.2771 倍**。兩邊 exit 0、未 timeout、pinned inputs 不變。Child CPU 包含 helper 及其正常子程序，不是 model-only CPU。

第一個 A/B pair 的 baseline 曾短暫與小型 O1 unit compile/test、舊 profile 的離線自測重疊；沒有 concurrent full build 或 sampler。保留此限制，不能稱作安靜環境的重複中位數或最佳數據。兩個優化一起套用，本輪只能量到組合效果，不能拆成 TPU 和 USC 各自的倍率。

候選計時前已結束所有本輪編譯、測試與大型分析；計時中僅低頻讀取既有 journal 尾端以回報 draw 進度，沒有 sampler／高頻 observer。Journal 不含可用的逐 draw 執行時鐘，因此這輪不宣稱有精確逐 draw 的 A/B 耗時；原 profile 的區段時間不能替代此次數據。

目前工作目錄的六個 production files 與實際執行的 candidate snapshot 逐 byte 相同。完整 frame 驗證針對 frozen Car Chase correctness baseline 加上這六檔；不是把尚未提交的其他 GL5 修改整份納入 full-frame 驗證。

程式修改已在 repo；本輪只建置私有驗證版，**沒有覆寫現有 Mesa install prefix 或預設 build/runtime**。要重跑已驗證的候選，可使用新的輸出目錄（不要覆寫本次 receipts）：

```sh
speed_round=/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/carchase-speed-round1.RxlvhF
speed_repeat=$(mktemp -d /Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/carchase-speed-repeat.XXXXXX)
python3 -B "$speed_round/run_unprofiled.py" \
  --out "$speed_repeat/candidate" \
  --bridge "$speed_round/candidate-model/build/lib/libpvrgpu_systemc_bridge.dylib" \
  --source-receipt "$speed_round/candidate-model/receipt.json"
```

這個命令會重新跑完整 frame；若要再做 baseline，使用另一個 fresh `--out` 並省略 bridge/source-receipt，不能把單獨候選重跑稱作新的成對 A/B。

## 完整輸出與稽核

最終 1920×1080 canonical RGBA8 raw（8,294,400 bytes）與 PNG 的 SHA-256 各自相同：

- Raw：`fed53ff416f52ca89977c858f46a3953c69f1d92041c8b7d6d5a2a53e0397d2f`。
- PNG：`c4d7183218bb0e539d78b62790d9c69fc76110b72e5571681b47161f36b1d8ef`。

離線稽核在兩邊 replay 結束後執行，確認：

- 395 個 ordered draw ranges＋尾段6649，397 個 clean GL API scopes。
- 391 組 graphics model hello/counter/done，400 個 physical graphics submissions、413 組 graphics API pairs（391 draws＋22 clears）、5 組 compute API pairs。
- 8 個 zero-work indirect draw scopes 如實保留：332、333、334、336、337、339、341、342。
- **所有 model 欄位零差異**，包含 virtual time、shader fingerprints/opcodes、pool allocations/releases、cache/DRAM 與 stage counters；aggregate、compute、ordered boundaries、最後 command 與 metadata 也零差異。所有 pool 平衡。
- 兩份 runtime stderr 為空；compile-only initialization 的既有 772 筆 legacy-policy／1132 筆較寬 diagnostic event rows 與原版相同，不宣稱整個 loading log 無拒絕。

這是與原 native baseline 相同，不是與 llvmpipe pixel-exact 或 Car Chase 全部 correctness PASS 的宣告；只有最後一張附件，不宣稱所有中間附件逐 pixel 相同。

### 保留的 driver 日誌差異

Strict comparator 只正規化明列 host pointers 與輸出路徑，仍留下 19 筆 `index_bias` 差異，分布在 Draw 353–358、370–382。全部是 `draw_array_primitive`，且 `index_size=0`、`user_indices=0`、`index_res=0x0`，沒有 indexed draw 或其他非指標欄位差異。

逐筆來源核對：frozen driver17 只有 indexed 分支使用 bias；非 indexed vertex packing 使用 `draw.start + v`。Mesa DrawArrays 只初始化 start/count、未初始化無意義的 index_bias，但既有 logger 無條件印出它。這是舊記錄器的不穩定欄位，沒有改變實際 model 工作；本輪沒有修改 logger/driver，也不推測每個數字的精確 stack 來源。

原 `strict-v2/report.json` 的 `A_B_MISMATCH` 和 `timing.equivalent_workload_verified=false` **保持原樣**。另附逐筆 source classification，限定為「model/counters/final images exact、完整 driver log equality=false」，不將 strict 報告改成 PASS，不忽略未來其他 index_bias 差異。

稽核工具另修正一次路徑正規化 bug：原程式把 replay 路徑 `candidate` 誤替換到 `candidate-model` bridge 路徑。新的私有 v2 只改成先判斷精確 bridge 值，再按完整路徑 component 判斷 run；4 個 collision／真實環境差異負控制測試通過。原工具與早期 refusal 證據未覆寫，model/driver 欄位檢查未放寬。

## 本機證據位置

- 本輪根目錄：`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/carchase-speed-round1.RxlvhF`。含 pinned launcher/builder、`baseline/`、`candidate-model/`、current-build 和 `regression-final.xml`。
- Source certificate：`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/carchase-round1-baseline-audit.3bposu/candidate-source-scope.json`。
- 完整 A/B 稽核：`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/carchase-round1-pair-audit.Ggaa1k`，含 `strict-v2/report.json` 與 `index-bias-source-classification.json`。
- Memory/cache 測試：`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/read-into-unit.mzECXO/REPORT.md` 和 `SANITIZER_REPORT.md`。
- USC UBSan：`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/usc-scratch-ubsan-final.wv3zRS`。
- 既有 tessellation 負測試對照：`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/tess-texture-stale-bound.fUxhTJ/REPORT.md`。
- 不修改歷史量測：[原始 CPU profile 報告](CARCHASE_CPU_PROFILE_20260910.md)。其中 1097.055 秒是帶 profiler 的執行，不能拿來當這輪 unprofiled baseline。

上述 tmp 路徑不是永久發佈位置；搬移或清理前須保存 receipts、原始輸出與 audit 證據。

## 下一步

先對這個候選重新量 CPU profile，再決定更多 USC copy reduction、ISS resume 靜態 metadata、ASTC decode reuse 或 JIT 的次序。本輪沒有分項 A/B 或新版 profiler，不把舊版 stack share 當成剩餘瓶頸的精確比例，也不預估 JIT 倍數。
