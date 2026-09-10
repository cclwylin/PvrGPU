# Manhattan：ReadInto＋USC scratch 完整 frame 量測

日期：2026-09-10。完整 frame 量測完成：原版 A 875.458080 秒，候選 B 820.198838 秒；此對觀察到耗時減少 6.31%，節省約 55 秒。所有 model/counter、compute 與最後 raw／PNG 精確一致；嚴格 driver log 比較仍保留 21 項已逐筆來源確認的診斷／host 位址差異，詳見下方。

## 本次比較的版本

沿用 [Car Chase 第一輪](CARCHASE_SPEED_ROUND1_20260910.md) 的同一組 frozen runtime，只更換 capture 與最後 event。兩邊都已包含先前的 backing `Contains` 範圍驗證快取；B 另外包含 TPU caller-owned `ReadInto` 和 USC texture-round scratch capacity 重用。

- A bridge SHA-256：`f2cd648286a2c848fd1976137bdf4f247f9553497171ee6dcaff64281bbac243`。
- B bridge SHA-256：`75463f6584384d57e094c507ba2e0a540a33d6aeb3727ff92b7fa11754032ac5`。
- Driver17 SHA-256：`eaa698618540f0bc82d829c961d80798002f1135493e290fbb7bf3cfc27f2f94`，兩邊相同。
- Packed10 strict helper SHA-256：`eb1c1b5b2efc3ac7078f165efed75e91fd9a9345b4452f105119755457cf08e1`。
- RenderDoc SHA-256：`979a494d4d1ab7a090649b45414c54490841d44401ac676ad55e8d3d49df5e05`。

既有 source-scope certificate 證明兩個 model 的 528-file source snapshot 只有六個 simulator production files 不同，61 個 bridge/memory-core compile commands 除 source/build 路徑外相同。量測期間不新增 production 修改、不重建／覆寫 driver、Mesa prefix 或預設 runtime，也未 commit/push。

先前 Manhattan 的 **814.086443 秒**是另一組較舊的 driver/bridge/RenderDoc，以及只做事後 audit 的 fast helper。共同的 145 個 source receipt 項目已有 35 項不同，因此只能保留為歷史結果，不能拿它與此次 B 直接計算 ReadInto／USC 的效果。

## 固定工作量與計時方式

Capture：`/Users/linwanyi/Downloads/_Codex/GPU_TestPatterns/4.gl_31_manhattan/recorder/trace/gl_manhattan31_capture_1.rdc`。

SHA-256：`82247f3d148039fd33082c3ee7c64975e6b42959b2068a0f955f4eabcc8e81ff`。

執行完整 231 個 draw actions（index 0–230）。最後 draw 的 event 是 8276，命令使用 **`--event 8299`**，另包含 8277–8299 的 trailing range；不能以 `--draw 230` 替代完整 frame。Draw action index 不是 physical submission 數。

- Model Release `-O3 -DNDEBUG`，diagnostics OFF，無 fast-math；固定 driver 是 `-O2`。
- GLES 3.1、surfaceless、shader cache OFF，沿用 compile-only OpenCapture loading 後恢復初始資源，再執行 native ordered ranges。
- `PVRGPU_SYSTEMC_DISABLE_PNG=1`，只關掉 per-model PNG；helper 仍保存最後附件 raw／PNG。沒有 every-draw snapshot、checkpoint resume 或跳 draw。
- 保留 counters、實際 model 檢查、strict helper 的逐 range API observer、finish、evidence copies 與 inline native audit。
- 不掛 profiler，也沒有高頻 journal observer。僅低頻讀既有 journal 尾端以回報進度。
- A 完成退出後才啟動 B；計時期間不並行本任務的 build、unit test 或大型稽核。不停止其他使用者程序，故不保證整台主機完全無背景負載。

Process wall 使用 monotonic clock，從 helper `Popen` 到 `wait` 完成，包含初始化、最後 readback／PNG 與 teardown；排除 setup/hash、build 與事後 audit。啟動後一次很小的 receipt save/print 仍在計時內。Child user/system CPU 包含 helper 及其正常子程序，不是純 model CPU。

這輪是一次 A→B 比較，不是中位數、最佳值或統計信賴區間；兩項優化一起套用，不能分拆成各自的加速倍率。

## 量測結果

| 完整 frame | Process wall | Child user CPU | Child system CPU |
|---|---:|---:|---:|
| A：原版 | 875.458080 s | 708.212535 s | 166.175380 s |
| B：ReadInto＋USC scratch | 820.198838 s | 639.805556 s | 172.956016 s |

由 **14 分 35.46 秒降至 13 分 40.20 秒**，省 **55.259242 秒**，耗時減少 **6.3120%**，速度為 **1.06737 倍**。A/B 皆正常 exit 0、未 timeout，全部 pinned inputs 前後一致。

此 capture 的整體改善小於同輪 Car Chase 的 21.70%；不能把一個 workload 的加速倍率套到另一個 workload。Child user CPU 下降，但 system CPU 略增；本輪沒有新增 CPU profile，不據此確定歸因剩餘瓶頸，也沒有分別量出 TPU／USC 的獨立貢獻。

兩邊最後 1920×1080 canonical RGBA8 raw（8,294,400 bytes）與 PNG 各自完全相同：

- Raw SHA-256：`84d7341f968997d0cdcafca60664934d832f646049cfaacba2273d5f041f59b3`。
- PNG SHA-256：`ca8372f4ab857f3b3c238c6c383fda2fcc30797828f6a9769ae289e54fe63199`。

這兩個 image hashes 也與歷史 814 秒版本相同；但 runtime/helper/稽核政策不同，所以歷史時間仍不是本輪的可比基準。

## 獨立稽核結果

兩邊實際完成 231 個 ordered draw ranges＋tail 至 event 8299，共 233 個乾淨 API scopes。獨立稽核確認：

- 219 組 model hello/counter/done、219 個 physical graphics draw submissions；233 組 graphics API submit/done（含 14 clears）及 5 組 compute submit/done 完整。
- 所有 model 欄位、virtual cycles、cache/DRAM、pool allocations/releases、shader fingerprints/opcodes、aggregate counters 與 compute 欄位精確一致，最後 raw／PNG byte-for-byte 相同。
- Pools 收支平衡，snapshot／shutdown audits 乾淨，helper/model runtime stderr 為空。
- 12 個 empty scopes 由 capture action counts、實際 range counts 及零 model/graphics/compute 工作共同證明。Draw136–141 為 vertices=0、instances=1；Draw206–211 為 vertices=4、instances=0。不能一律豁免 zero-count action：Draw205 的 range 仍含實際前置工作，必須驗證。
- Compile-only initialization 另行比對：兩邊同為 382 個 legacy-policy matches、538 筆較廣義的診斷列（218 record errors、151 not-lowerable、164 unsupported draw、5 unsupported compute）。這些初始化 declines 不是實際 replay ranges 的執行失敗。

完整 driver log 的嚴格結果仍是 **`A_B_MISMATCH`**，原報告的 `timing.equivalent_workload_verified=false` 不改寫。以下 21 項差異由另一份 source-classification 記錄逐筆說明，不修改 comparator，也不新增忽略欄位：

1. **16 筆 `draw_array_primitive.index_bias`**：Draw143–145、151、219–230。每筆實際記錄皆為非 indexed（index_size=0，無 index resource/user indices）。固定 driver17 的 logger 無條件印出 Mesa DrawArrays 未指定的欄位，但只有 indexed path 使用它；nonindexed vertex selection 使用 `draw.start + v`。重用已 pinned 的 driver source 證據，另核對這次每筆實際記錄；沒有追蹤該未指定值的具體 stack 起源。
2. **5 筆 Draw0 `framebuffer_readback_declined.cbuf0`**：固定 `pvrgpu_resource.c` 證明這是以 `%p` 印出的 host `pipe_resource` 指標。每邊五筆皆維持同一 nonnull attachment pointer，`cbuf1` 為 null，requested `res` 與 attachment 不同；它不在原 comparator 的 named host-pointer allowlist，因此仍列為嚴格差異。

除此之外，在原有的明列 host-pointer／輸出路徑正規化下沒有其他差異。結論是「觀察到的 functional workload、model/counters 與最後影像一致」，不是整份 driver log 嚴格 PASS；也不是所有中間影像、llvmpipe pixel-exact 或 API conformance 的宣告。Helper 原有 hardcoded `native_execution_verified:false` 沒有修改，完成性由實際 range/API/model 證據確認。

稽核器準備時，v1 原本只接受 Car Chase indirect zero-work 證據，因而拒絕 Manhattan 的上述 12 個直接 zero-count scopes。原始拒絕與 v1 都保留；v2 在 B 啟動前固定，只允許有 pinned capture counts 及實際零工作證據的這 12 個 scopes，其他 missing work 照樣拒絕。32 個離線測試（含負向案例）與明確標為 `SELF_CHECK_NOT_A_B` 的 A 自我檢查通過後，才執行正式 B 與實際 A/B 比對。

沿用上一輪已驗證的 build/test 證據，本輪不重跑也不冒稱新增 unit/sanitizer coverage；這次新增的是 Manhattan 完整 frame 量測與等價比對。

## 本機證據與重跑

本輪根目錄：`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/manhattan-speed-round1.h5S77T`。

- Launcher：`run_unprofiled.py`，SHA-256 `16f3bd7a4ecfd1c4d8a774c7f2e8e3e59b0e76a641a0d12b40829b7350609125`。
- 啟動前兩個 variants 的 `--check-only` 均通過，各核對 1080 個 pinned inputs。Baseline/candidate receipt 保存完整 pins、環境、command 與 process timing。
- Source-scope certificate：`/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/carchase-round1-baseline-audit.3bposu/candidate-source-scope.json`，SHA-256 `c2365f331a29dc3eb4db0292608b28c32cebb74d3eec58bf0d3c8fb6a6318b1a`。
- [獨立稽核摘要](/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/manhattan-round1-audit.kgumJl/REPORT.md)。同目錄保存 frozen v2 auditor、32 項離線測試與 self-check 證據。
- [嚴格 A/B report](/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/manhattan-round1-audit.kgumJl/actual-pair-v2/report.json)，SHA-256 `19c64d29abd3756ca4e263cdfe87121b75a1aa62fbe9a1a8246443b7412f0e5b`。
- [21 項差異的來源分類](/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/manhattan-round1-audit.kgumJl/actual-pair-v2/diagnostic-classification.json)，SHA-256 `ca610d03cd6e52fb949dc34c75dfeceffed3eb1954b4671c9034a70878caf938`。
- Frozen v2 auditor SHA-256：`9368b648a19b4783c30ce84c5e08432d4e703f49151eb2581e93a6d4fae9f952`。全程固定的 capture-boundaries-v2 metadata SHA-256：`2b2d67595cd7a24236606b48ab1b60a818a55b6afdcbb574f96c2b122477da54`。

需要新的完整 A/B 時，兩個命令依序執行；先確認 A 成功完成才執行 B。使用新的輸出目錄，不覆寫這次證據：

```sh
manhattan_round=/Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/manhattan-speed-round1.h5S77T
manhattan_repeat=$(mktemp -d /Users/linwanyi/Downloads/_Codex/Working/PvrGPU/tmp/manhattan-speed-repeat.XXXXXX)
python3 -B "$manhattan_round/run_unprofiled.py" --variant baseline --out "$manhattan_repeat/baseline"
python3 -B "$manhattan_round/run_unprofiled.py" --variant candidate --out "$manhattan_repeat/candidate"
```

每個命令都真的重跑完整 frame；可以加 `--check-only` 只做非執行的 preflight。tmp 路徑不是永久發佈位置，搬移或清理前須保存 runtime、receipts、原始輸出和稽核證據。
