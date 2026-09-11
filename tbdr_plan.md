# PvrGPU Render-Pass TBDR 實作計畫

狀態：規劃中（已依實作程式碼複查修訂）
更新日期：2026-09-11

## 1. 目標與模式契約

PvrGPU 必須永久保留兩條公開排程路徑，兩者使用相同的 shader、raster、depth/stencil、
blend 與 PBE 語意；差別只能是 draw/tile 的排程和中間狀態生命週期。

| 模式 | 用途 | 排程 | 可否作為效能結果 |
|---|---|---|---|
| `draw-serialized` | debug、reference、逐 draw 定位 | 每個 draw 完整執行 tiler → renderer → PBE，完成後才送下一個 draw | 否 |
| `render-pass-tbdr` | HW mode、performance mode、正式硬體行為 regression | compatible render pass 內先完成所有 Phase 1，再逐 tile 完成 Phase 2 | 是 |

**不可更改的政策：**

- Debug mode 預設且強制使用 `draw-serialized`，以保留逐 draw 停止、Capture/Play、
  snapshot 和二分定位能力。
- HW mode 或 performance mode 預設且強制使用 `render-pass-tbdr`。
- 在 HW/performance run 中若 TBDR 遇到不支援的狀態，必須明確報錯或以有紀錄的
  render-pass split/partial render 處理；不得靜默退回 `draw-serialized` 後仍宣稱為
  HW/performance 結果。
- 為了診斷，允許操作者顯式將同一 workload 改跑 `draw-serialized`，但報告必須標為
  debug/reference，不能納入效能比較。
- 不得依 capture 名稱、測試名稱、draw index 或 framebuffer hash 選擇模式或渲染答案。
- Pass-builder bring-up 可以提供 internal/test-only 的 dry-run：只計算 pass grouping、
  split reason 和預期 LOAD/STORE，不改變實際 serialized 排程。它不是第三個公開 mode，
  必須在 report 標成 `pass_builder_dry_run=true` 且永遠回報「非效能結果」。
- 在 Phase 3 通過完成條件之前，`render-pass-tbdr` 一律回報 unavailable；不得讓
  `render-pass-tbdr` 這個名稱在底層跑 serialized backend——那正是本節禁止的 hidden
  fallback，計畫自身也適用。

建議提供一個一般性的設定入口，例如：

```text
--tbdr-mode=draw-serialized
--tbdr-mode=render-pass-tbdr
PVRGPU_TBDR_MODE=draw-serialized|render-pass-tbdr
```

CLI/runner 的 execution intent 應做顯式映射：

```text
debug/reference     -> draw-serialized
hw/performance      -> render-pass-tbdr
```

若 CLI 與環境變數同時存在，CLI 優先；啟動時必須把最後選定的模式寫入 log、counter
report 和 replay manifest。

## 2. 現況

目前 `draw_pco_sequence` 可以攜帶多個 draw，但
`model_stub/submitter.cpp` 在每個 draw 將 `PipelineState` 送入 pipeline 後，仍等待
`sequence_completion_` 才送下一個 draw：

```text
draw 0: Phase 1 -> Phase 2 -> PBE -> wait
draw 1: Phase 1 -> Phase 2 -> PBE -> wait
...
```

因此現況是 **draw-level two phase**，不是硬體級 render-pass-wide TBDR。它依靠 draw
之間的 attachment LOAD/STORE 維持結果，功能可正確，但會重複讀寫 framebuffer/depth、
失去跨 draw 的 hidden-surface removal 機會，memory、bandwidth、counter 與時間行為也不
代表真正 TBDR。

`model_stub/submitter.cpp` 現有 `TODO(pvrgpu-tbdr)` 所在的 wait（`submitter.cpp:3056`
註解、`:3064` 的 `wait(*sequence_completion_)`）是 debug/reference 模式的核心同步點。
不能只刪除此 wait；必須先建立共享 render-pass 狀態、Phase 1/2 barrier、hazard
splitting 和有上限的 parameter/tile memory。

### 2.1 從現行 code path 確認的重複成本

重複 traffic 不是推論，是現行資料流的直接後果：

- `submitter.cpp:1711`：每個 alias 前一張 attachment 的 draw，對每個 render target
  做一次**整張 color 的 `Readback`**，depth 亦同，且發生在 submitter thread 內、
  送出該 draw 之前。
- `isp.cpp:251`：ISP 以 `width * height * samples * layers` 配置**整張 surface** 的
  depth/encoded-depth/stencil/coverage plane，並整張解碼 LOAD 內容。
- PBE/PbeWriteBack 在 draw 結束時整張寫回。

1920×1080 RGBA8 的單張 attachment 約 8.29 MB（約 7.91 MiB），但不能把現行成本簡化成
固定的「每 draw 一讀一寫」：PbeWriteBack 寫入 DRAM 後還會為發布/驗證做整張 Readback，
下一個 alias draw 的 submitter 又會做一次 Readback；MRT、depth/stencil、MSAA 和 cache
行為還會增加或改變 transaction。這裡是 code-path estimate，不是已量測 bandwidth。
Phase 0 必須從 counters 建立每個 workload 的實測 bytes/transactions baseline。

因此「移除 wait」本身不會降低 traffic：真正的成本在 per-draw 全畫面 LOAD/STORE、額外
Readback 與全畫面 plane 配置，這才是重構標的。

參考數量級：handoff 記錄 Manhattan 為 219 個 model drawlist、36 個 graphics
submission，平均約 6 draw/submission。這只能描述目前 batching，不能直接推導 TBDR 的
收益上限：有些 submission 可能因真正 hazard/flush 切開，有些可能只是 payload budget
或目前 transport 邊界。Phase 0 必須先輸出每次 split reason census，再判斷哪些邊界在
硬體 render pass 中仍不可跨越；§9 的目標值以實測 traffic 和該 census 為準。

### 2.2 已具備、可直接沿用的基礎

- `model_types.h:663` `ResolveSequenceAttachmentAddresses` 已實作 attachment identity
  鏈接（`color_attachment_source_command_index` / `depth_attachment_source_command_index`）。
  color alias 會比對 format、extent、render-target count、sample count 和 layer count，
  可作為 §5 pass builder 的起點。
- 上述 resolver 不是完整 compatibility engine：depth alias 目前只確認來源 draw 在前且
  有非零 address，尚未比對 depth format、extent、sample count、layer count；它也沒有
  first-class LOAD/STORE op、resolve 或 general resource hazard。Phase 1 必須補齊這些條件，
  不能把現有 address resolver 直接視為 render-pass 判定。
- `isp.cpp:365` 的主迴圈已經是 `for (tile) for (ordered primitive ref)` 的 tile-major
  形狀。共用 tile list 之後迴圈結構不必改寫；改的是每個 ref 必須查出「自己那個 draw」
  的 raster/shader/resource state，而不是沿用單一 `PipelineState` 欄位。
- tile list 已經用 `MemoryPool` handle 傳遞，§4「不得持有 host pointer」是既有慣例的
  延續，不是新規則。

目前沒有 first-class `RenderPassState` 或明確的 LOAD/STORE-op state machine；但既有
attachment source、initial payload、clear 與 readback 語意可以沿用。§5 的 pass builder
需要新建，attachment 基礎則是擴充而不是完全從零重做。

## 3. 目標硬體流程

一個 compatible render pass 的正式流程如下：

```text
BeginRenderPass
  LOAD/CLEAR attachments once

  Phase 1: geometry + tiler
    draw 0 -> append primitive refs to shared per-tile lists
    draw 1 -> append primitive refs to shared per-tile lists
    ...

  Render-pass-wide barrier

  Phase 2: tile renderer
    tile 0 -> execute refs in original draw/primitive order -> STORE if required
    tile 1 -> execute refs in original draw/primitive order -> STORE if required
    ...
EndRenderPass
```

Phase 2 可以 tile-major 執行，但每個 tile 內必須維持 API 可觀察的 draw order、primitive
order、depth/stencil、discard、blend、sample mask 和 side-effect ordering。任何重排最佳化
都必須另外證明不改變 GLES 語意，不屬於第一版範圍。

### 3.1 跨 draw HSR 延後為獨立最佳化

現況：`isp.cpp:171` 的 `opaque_early_hsr` 是**整個 draw 一個 bool**——只要該 draw
blend enable、shader 可能 discard、shader 寫記憶體，或需要 late depth/stencil，就整批
關閉 early HSR。這在「一個 tile list 只含一個 draw」時是安全且足夠的簡化，但不能直接
套到混合多個 draw state 的共享 tile list。

第一版 `render-pass-tbdr` 必須以 correctness-first 實作：每個 tile 內所有 candidate 嚴格
依 API draw/primitive order 執行，**不做跨 draw candidate 淘汰**。即使先不做跨 draw HSR，
pass-wide attachment LOAD/STORE 與 tile-local working set 仍能帶來主要 bandwidth/memory
收益。這可以把排程正確性與進階最佳化分開驗證。

跨 draw HSR 在核心 TBDR、Capture/Play 和 differential debug 穩定後，另以獨立 phase
加入。屆時不能只用 opaque／punch-through／translucent 三分類；至少要納入：

- shader image/SSBO/atomic 等不可省略的 side effect；
- discard、custom depth、late depth/stencil 造成的 order dependency；
- blend、logic/color operation 與 destination read；
- 每個 MRT 的 fragment output mask 與 color write mask 是否完整覆寫；
- sample mask、sample-frequency shading、alpha-to-coverage 與 per-sample coverage；
- depth/stencil write 對後續 candidate 的可觀察影響。

只有在「省略被遮蔽 candidate 的 shader、color、depth/stencil 和 memory side effect 都不可
被 API 觀察」時才能淘汰。判定必須以 attachment 和 sample 為單位，不能只以 pixel 或單一
draw-level bool 近似。任何保守禁止 HSR 的原因都要輸出 counter，但這是最佳化統計，不是
serialized fallback；只要仍走 pass-wide Phase 1/2，該 run 仍是合法 TBDR。

驗收條件：啟用 HSR 前後都與 `draw-serialized` bit-exact；discard、late depth/stencil、
blend、MRT masked output、sample mask 和 shader side-effect cases 必須分別覆蓋，HSR
rejection 與各類 conservative-disable counter 分開報告。

### 3.2 32×32 tile fragment lifecycle 與 tile-local framebuffer（未來工作）

現況只有前半段 tile 化：Tiler 已建立 row-major 32×32 `TileRecord` 與 ordered primitive
refs，ISP 也以 `for (tile) for (primitive ref)` 走訪；但 `TileScheduler` 仍只為整個 draw
送出一個 pipeline token，ISP 最後把所有 tile 的 candidates flatten 成 draw-wide payload，
FragmentFrontend／PDS／USC 再以整個 draw 的 invocation／2×2 quad arrays 執行。另一方面，
ISP 的 depth/stencil/coverage 與 PBE 的 color working surface 都仍按完整
`width × height × layers × samples` 配置，PbeWriteBack 也以整張 attachment 寫回。因此目前
不能把「32×32 binning／tile-major ISP」稱為「32×32 fragment lifecycle」或「32×32
framebuffer」。

正式 `render-pass-tbdr` 必須補齊下列兩個彼此相連、但分階段落地的契約：

1. **32×32 tile-bounded fragment lifecycle。** Phase 1 barrier 後，由 coordinator 或等價
   排程狀態以有上限的 tile work descriptor 驅動 fragment path。descriptor 至少帶
   pass/generation/tile ID、global tile rect、primitive-ref range，以及 immutable draw-state
   reference。FragmentFrontend、PDS、USC、TextureUnit 與 PBE 必須能讓一個 tile 的
   candidates、2×2 quads、shader lanes、continuations 和 outputs 在該 tile 完成後釋放，
   不得以 pass/draw-wide flat arrays 掩蓋無上限的 transient working set。2×2 quad 與
   4×2 half-stamp 必須保留 global framebuffer coordinates、helper-lane/derivative 語意；
   tile 內仍依原始 draw/primitive order 執行。實作不得為 1080p 直接複製約 60×34 份完整
   `PipelineState`；共享 state 的 ownership、in-flight tile 上限與 completion aggregation
   必須 first-class。
2. **32×32 tile-local framebuffer。** 每個 tile 的 color/depth/stencil/coverage working set
   依有效 tile rect、layers、samples、MRT 數和各 attachment bytes-per-pixel 配置；右／下
   邊界 tile 的有效範圍必須顯式記錄。tile 開始時對每個 attachment 做一次所需的
   LOAD/CLEAR，該 tile 的所有 draw refs 完成後才由 PBE STORE/scatter 到完整 attachment 的
   GPU address（正確處理 row pitch、layer、sample 與 MRT）。最終 Phase 4 TBDR path 不得再
   配置 pass-owned full-frame color/depth/stencil staging surface；完整 framebuffer 只在
   pass end 的 presentation、API readback 或 Capture/Play 明確要求時 materialize，不能在
   每個 draw 後重建或 readback。

落地順序固定為：Phase 3 先建立 tile work/range、真正的 tile-bounded shader lifecycle 與
per-tile completion，允許暫時沿用 pass-owned full-frame attachment backing；Phase 4 再把
attachment working storage 和 transport 改成 tile-local。不得以「把 draw-wide array 加上
tile offset」作為最終完成狀態，也不得在 Phase 4 只切割 PBE write 而保留 full-frame ISP/PBE
working surface。

驗收至少覆蓋跨 `x=31/32`、`y=31/32` 的 primitive、空 tile、非 32 倍數的 framebuffer
邊界、單 tile 大量 overdraw、MSAA/sample-frequency、derivative/helper lanes、MRT、layered
attachment、blend、discard 與 late depth/stencil。兩種排程必須 bit-exact；report 必須能
證明 fragment transient 與 attachment working-set high-water 受 in-flight tile budget 約束，
且同一 compatible pass 沒有 per-draw full-frame LOAD/STORE/Readback。

## 4. 核心資料結構

新增一個共享的 `RenderPassState`（名稱可依既有命名調整），至少包含：

- render-pass ID、submission ID、目前 phase 和 barrier generation；
- framebuffer extent、layers、samples、render-target count；
- 每個 color/depth/stencil attachment 的 identity、format、LOAD/STORE、clear value；
- attachment 的 DRAM backing 與 tile-local backing；
- shared per-tile list，entry 至少帶 draw ordinal、primitive ordinal 和所需 pipeline
  state；未來啟用 §3.1 的 HSR 時再加可證明安全的 per-ref metadata；
- shared parameter buffer，以及 vertex/varying/coefficient payload 的有效生命週期；
- 每個 draw 的 immutable raster/shader/resource state reference；
- tile-list/parameter-buffer 已用量、high-water、budget 和 spill/partial-render generation；
- split reason、partial-render reason 和 counters；
- snapshot/Capture/Play 所需的 deterministic ordering/version 資訊。

不能讓 tile-list entry 持有 host pointer。跨 Phase 1/2 的資料必須使用可驗證、可序列化、
有明確 ownership 的 `MemoryPool` handle 或等價 ID；所有引用直到該 render pass 的 Phase 2
完成後才能釋放。

## 5. Render-pass compatibility 與切割規則

只有 framebuffer 與附件狀態相容的 draws 才能合併。第一版採保守規則，遇到下列事件
必須結束目前 pass 或執行有紀錄的 partial render：

| 事件 | 必要動作 |
|---|---|
| framebuffer、extent、layer、sample count 或 attachment format/identity 改變 | split |
| attachment 需要 resolve、readback、map 或成為後續 texture/image input | split + STORE/barrier |
| texture feedback、image/SSBO/atomic 或其他讀寫 hazard | split；除非已有正式 hazard model |
| compute dispatch 或 graphics/compute resource dependency | split + memory barrier |
| API 明確 flush/finish 或 capture checkpoint | partial render/split，並記錄原因 |
| tile-list 或 parameter-buffer 達到 budget | deterministic partial render |
| 單純改 shader、viewport、scissor、blend、depth/stencil 等 per-draw state | 不 split；記在 draw state |

Clear 若能正確表示成 pass-local tile operation，可以留在 pass 內；第一階段無法證明時先
保守 split。每次 split 都要有 enum reason 和前後 draw ordinal，不能只留自由文字。

## 6. 記憶體上限與 partial render

真正 TBDR 不能無限制「先吞下全部 draw」。實作前先定義可設定的硬上限：

- shared tile-list bytes；
- parameter/coefficient bytes；
- retained per-draw state bytes；
- tile-local attachment bytes；
- 整個 render pass 的 pool high-water。

超過上限時，先完成目前已 bin 的 deterministic draw prefix，STORE 必要 attachments，接著
以 LOAD 延續下一個 generation。partial render 必須保持結果正確，並輸出：

- budget 類型、limit、requested bytes、high-water；
- 截止的 draw ordinal；
- 已完成的 tile/generation；
- 額外 LOAD/STORE/PBE 與 bandwidth counters。

可先實作 bounded partial render，再考慮把 tile-list/parameter buffer spill 到 modeled DRAM。
禁止用 host 無上限容器掩蓋硬體容量問題。

## 7. Debug、Capture/Play 與 Snapshot

Debugability 是 TBDR 的前置需求，不是最後補上的工具。

### 7.1 `draw-serialized` reference mode

- 保留目前每 draw 的 `sequence_completion_` wait。
- 支援在指定 draw 前/後停止。
- 支援現有 drawlist snapshot、區間 replay 與 Capture/Play。
- 作為 TBDR divergence 的 reference 排程，但不是外部像素 oracle；兩條路徑仍須遵守同一
  PvrGPU 語意。

### 7.2 `render-pass-tbdr` debug controls

- 可在指定 draw 後強制 partial render，用二分法找第一個污染狀態的 draw。
- 可 dump 每個 draw 對指定 tile list 的 delta。
- 可 dump 指定 tile 內的 draw/primitive ordering。
- 可在 Phase 1 barrier 前後建立 checkpoint。
- Capture/Play 可分別保存與重播 Phase 1、Phase 2。
- Snapshot 必須包含 shared tile lists、parameter buffer、retained draw states、tile-local
  color/depth/stencil、draw ordinal、phase、barrier generation 和 memory budget 狀態。
- 每個 pass 記錄 LOAD/STORE、barrier、split、partial render 和原因。
- 提供相同輸入下 `draw-serialized` 與 `render-pass-tbdr` 的指定 draw/tile/color/depth/
  stencil 比對，回報第一個 divergence。

使用強制 partial render 得到的正確畫面只能當診斷證據。正式 HW/performance regression
必須回到沒有 debug 強制切割的 `render-pass-tbdr` 再跑一次。

## 8. 分階段實作

用語澄清：本節的 **Phase 0–8** 指實作階段；§3、§4、§7 出現的 **Phase 1／Phase 2** 指
TBDR 硬體流程的 geometry/tiler 階段與 tile renderer 階段。兩者不同，勿混用。

### Phase 0：凍結正確性基線

- 先完成目前 glmark2 regression 修復，清理或提交既有 WIP，避免 TBDR 變更和舊 regression
  混在一起。
- 保存 dEQP L1/L2、texture 深度測試、GLBench、glmark2、Manhattan、Car Chase 的命令、
  artifacts、runtime identity、時間與 memory baseline。
- **新增 submission-boundary reason instrumentation。** 這不是「保存既有資料」：現在
  repo 內沒有任何 `split_reason` / `flush_reason` / `submit_reason` 欄位，邊界原因完全
  沒有被記錄。實際產生邊界的來源至少有兩類，都必須各自帶 enum reason：
  - driver 內 16 個 `pvrgpu_flush_current_color_attachments` 呼叫點（state change、
    clear、map/readback、frontbuffer 等成因不同）；
  - `pvrgpu_cmd.c:2660` 的 `PVRGPU_SYSTEMC_MAX_PCO_SEQUENCE_COMMANDS`（256）硬上限，
    這是 transport budget，不是硬體 render-pass 邊界。
  此項在關鍵路徑上：Phase 1 的 census 與 §9 的目標值都依賴它，不得延後。
- 加入 mode 欄位與解析；此階段只允許 `draw-serialized` 執行，
  `render-pass-tbdr` 必須回報 unavailable，不能暗中指向 serialized backend 或假裝是
  硬體模式。

完成條件：baseline 可重現；report 能明確辨識 execution mode；**每一個 graphics
submission 邊界都帶 enum reason**，census 能把該 workload 的 drawlist 數收斂到
submission 數的每一次切割逐一歸類（例如 Manhattan 的 219 → 36），並區分「硬體 render
pass 中仍不可跨越」與「僅為目前 transport/payload 限制」兩類。後者的比例決定 §9 第 4
條的門檻值，該門檻在本階段結束時釘死並寫回本文件。

### Phase 1：Render-pass builder、split engine 與管線拓撲決策

- 從 `draw_pco_sequence` 建立 compatible pass groups，以既有
  `ResolveSequenceAttachmentAddresses` 作為 color identity 的起點，並補上 depth format、
  extent、sample、layer、subresource 及完整 hazard 判定。
- **收緊 depth alias 必須先 census 再 enforce。** 現行 depth alias 只檢查來源 draw 在前
  且 address 非零，完全沒有比對 format/extent/sample/layer；補齊之後，目前能通過的
  capture 可能開始被判定不相容而多切 pass，甚至被 reject。順序必須是：先以 report-only
  模式列出哪些既有 capture 會新增 split 或被拒、原因為何，確認每一筆都是舊檢查過鬆而
  非新邏輯錯誤，才切換為 enforce。否則這會在 Phase 1 表現成「TBDR 造成 regression」，
  但實際是既有驗證不足。
- 實作 first-class attachment identity、LOAD/STORE、resolve 與 hazard 判定。
- 產生 deterministic pass ID、draw ordinal、split reason。
- 提供 internal/test-only pass-builder dry-run，只記錄預測的 groups、split reasons 和
  LOAD/STORE；實際仍由 `draw-serialized` 執行，`render-pass-tbdr` 仍回報 unavailable。

**本階段必須產出的架構決策（決定 Phase 2–3 是增量還是重寫）：**

`pvrgpu_model_stub.cpp:896` 起是一條靜態連接的線性 FIFO 鏈，geometry 與 fragment 串在
同一條上，因此一個 draw 的 fragment 工作緊接其 geometry 工作由同一個 token 完成。
Phase 2 開工前必須做一個小型 spike，比較至少下列四種方案並把決策寫回本文件：

| 方案 | 作法 | 代價 |
|---|---|---|
| A. 雙拓撲 | 另建 binner path（終止於 parameter buffer）與 tile fan-out path（fragment_decoder..pbe），barrier 在中間 | elaboration 與 module 實體加倍，但兩條路徑可獨立驗證，commit 可增量 |
| B. Mode-aware module | 現有 module 依 mode 改變行為 | 不加實體，但 ISP、FragmentFrontend、PDS、USC slot/cluster、TextureUnit、PBE 全部變成共用可變模組，§10 的「可獨立驗證 commit」不再成立 |
| C. Boundary coordinator/router（優先評估） | 在 ParameterBuffer 與 FragmentDecoder 間插入 pass accumulator、barrier 和 mux；serialized 模式 passthrough，TBDR 模式累積後再送入既有 fragment path | 不複製整條 pipeline，但需建立 per-ref draw-state table，並讓必要的 downstream stages 透過穩定介面取得自己的 state |
| C′. Coordinator + tile-major／draw-minor token（C 的低風險子集） | 同樣是 coordinator，但 barrier 後以 **tile 為外層、draw 為內層**發 token：每個 (tile × draw) 一個 token，downstream 每個 token 仍只看到單一 `PipelineState` | fragment 側幾乎不需改動——不必建 per-ref draw-state table；只需 PBE 支援「暫不 STORE」，並讓 depth/stencil/color working surface 存活於 `RenderPassState` 跨 token。代價是 token 數由 draws 變成 tiles×draws（1080p 約 60×34 tile），spike 必須量測 token/pool churn 與 FIFO backpressure 是否可接受 |

方案 C／C′ 都不是預先保證可行；spike 必須驗證 SystemC static port/elaboration、
backpressure、completion 和 snapshot ownership。C′ 保留 tile-major 外層迴圈，因此不會
擋住 Phase 4 把 working surface 改成 tile-local，可視為「先用 C′ 通過 Phase 3，再視需要
演進到 C」的路徑；但它把成本轉移到 token 數量，spike 要量的就是這個 overhead。若 C 與
C′ 都驗證失敗，再選 A 或 B，不能為了維持預設結論而繞過模型限制。

完成條件：分組/split unit tests 完整；dry-run 開關不改變 `draw-serialized` pixels、
counters 或 ordering；depth alias 收緊已完成 report-only census 且每筆差異可解釋；
pass builder 預測的 group/split 與 Phase 0 記錄的既有 submission 邊界能逐一對照，說明
每個差異是「TBDR 可合併」還是「仍不可跨越」；拓撲方案已定案並記錄證據與理由。

### Phase 2：共享 Phase 1 binning

- 將各 draw 的 geometry/tiler 結果附加到 shared per-tile lists。
- tile entry 保存 draw/primitive ordinal 和 immutable state reference。
- 延長所有 Phase 2 需要 payload 的生命週期。
- 加入 tile-list delta dump、budget accounting 和 Phase 1 snapshot。

完成條件：只跑 Phase 1 時，tile membership/order 可對照逐 draw 結果；無 leak、無 dangling
handle，snapshot round-trip exact。

### Phase 3：Pass-wide barrier 與 tile-major Phase 2

- 所有 Phase 1 工作完成後才開啟 Phase 2。
- 依 §3.2 建立有上限的 tile work/range 與 completion aggregation；fragment candidates、
  2×2 quads、PDS/USC tasks、texture continuations 和 outputs 必須具有真正的 per-tile
  lifecycle。此階段可保留 full-frame attachment backing，但不能再把 draw/pass-wide flat
  fragment arrays 當作 `render-pass-tbdr` 的最終實作。
- 每個 tile 依原始 draw/primitive order 執行 ISP/fragment/depth/stencil/blend；此階段禁止
  跨 draw HSR，所有有 API 可觀察效果的工作都維持 in-order。
- 允許先使用 pass-owned full-frame color/depth/stencil working surfaces，以降低第一次
  排程重構的風險；但 attachment 必須在 pass begin LOAD/CLEAR 一次、pass end STORE 一次，
  不能保留 per-draw 全畫面 LOAD/STORE。
- 把現行 per-draw PBE completion 改為 per-pass/per-generation completion。
- `draw-serialized` 保留原 wait；`render-pass-tbdr` 走新的 barrier/state machine。
- 在進入複雜 workload 前完成 Phase 2 checkpoint、指定 tile/draw ordering dump，以及
  兩模式第一個 divergence 的最小比較工具。

本階段同時驗收排程正確與 per-draw attachment traffic 消失；尚未要求 working surface
本身變成 tile-local，因此 peak host/model-pool memory 可能仍高，留待 Phase 4 解決。

完成條件：基礎 triangle、depth/stencil、blend、discard、MSAA、MRT、layered attachment
在 `draw-serialized` 與 `render-pass-tbdr` 之間 bit-exact；同一 compatible pass 只有一次
logical LOAD/CLEAR 和 STORE；沒有 serialized fallback。達成後才可把 `render-pass-tbdr`
標為 available。

### Phase 4：Tile-local attachment 儲存與 cycle model

這是整個重構最大的一塊 code change，獨立成階段。Phase 3 已降低 DRAM traffic；本階段
進一步把 full-frame working set 降為 tile-local，取得硬體級 memory footprint。

- 把 ISP 的整張 surface plane（`isp.cpp:251` 的 encoded depth/depth/stencil/coverage）
  改為 tile working set；FragmentFrontend、PBE、PbeWriteBack 同步改為 tile-local。
- 依 §3.2 實作真正的 32×32 color/depth/stencil/coverage backing：邊界 tile 記錄 valid
  rect，PBE 以正確 row pitch/layer/sample/MRT 位址把 tile STORE/scatter 到完整 attachment；
  只有 presentation、API readback 或 Capture/Play 可以在 pass end materialize 完整畫面。
- 把 Phase 3 的 pass-level full-frame LOAD/STORE transport 改成 modeled per-tile LOAD/STORE；
  `submitter.cpp:1711` 的 per-draw 全畫面 `Readback` 不得重新出現。
- 重新定義 reference uArch cycle model：現行 `WaitForCycles(kReferenceUarch...)` 的常數
  代表舊的 per-draw 工作量，tile-major 後必須依明確的 service rate、latency、transaction
  與 contention 方程重新推導，並版本化。virtual GPU cycles 是硬體模型指標；host wall
  time 是 simulator 執行成本，兩者必須分開報告，不能用 host wall time 反校硬體 cycles。

完成條件：per-tile working set 有可量測上限且符合預期（不是只看畫面正確）；
tile-local working-set high-water 相對 Phase 3 明確下降；LOAD/STORE bytes 符合逐 tile
推導；cycle model 有可解釋的方程，並在 report 標記版本。

### Phase 5：Bounded memory 與 partial render

- 實作可設定 budget 與 deterministic partial render。
- 驗證跨 generation 的 LOAD/STORE、depth/stencil 和 blend continuity。
- 加入 memory high-water、spill、extra PBE/bandwidth counters。

完成條件：壓低 budget 的 stress test 可穩定觸發多次 partial render，輸出仍與無切割結果
一致，peak memory 不超出容許誤差。

### Phase 6：完整 Capture/Play 與 differential debug

- Phase 1/Phase 2 checkpoint、snapshot schema/version、reject guards。
- 指定 tile/draw 的兩模式 differential 工具。
- 強制 split/partial render 的 debug controls 和 manifest audit。

完成條件：可以從錯誤 framebuffer 自動縮小到第一個 divergent pass、tile、draw 和 phase。

### Phase 7：可選的跨 draw HSR 最佳化

- 核心 `render-pass-tbdr` 在 HSR 關閉時已是合法 HW/performance 模式；本階段只能改善
  工作量，不能改變其模式身分或 attachment 語意。
- 依 §3.1 建立可證明安全的 per-ref/per-attachment/per-sample metadata。
- 有 shader memory side effect、late depth/stencil、discard、MRT masked output、partial
  color mask、blend 或不完整 sample coverage 時先採保守禁止，之後逐類證明並開放。
- 每類 rejection 與 conservative-disable 都要有獨立 counter，且能由 Capture/Play 重現。

完成條件：HSR 開/關與 `draw-serialized` 三方 bit-exact；side-effect/order-sensitive tests
完整；任何無法證明安全的 case 都只禁用 HSR，不得退回 serialized 排程。

### Phase 8：Regression 與 performance sign-off

- dEQP：texture、multisample、depth、stencil、blend、discard、MRT、layered、barrier/hazard。
- GLBench 與 glmark2 全套。
- Manhattan 與 Car Chase correctness + memory + performance。
- 同時比較 `draw-serialized` 和 `render-pass-tbdr`，但只有後者列入 HW/performance 成績。
- capture replay 的像素 oracle 仍是 llvmpipe：`render-pass-tbdr` 相對 llvmpipe 的既有
  差異不得擴大，且 strict replay PASS 與 reference attachment exact 分開報告。
- HSR 可關閉後獨立 sign-off；若開啟，須另外提供 Phase 7 的 correctness 與收益報告。

## 9. 測試與驗收指標

Correctness 報告至少包含：

- API/model/bridge error 與 unsupported/fallback count；
- framebuffer exact/diff、depth/stencil diff；
- draw/pass/split/partial-render 數量；
- shader/resource hazard audit；
- snapshot/Capture/Play runtime identity 與 schema version。

Performance/memory 報告至少包含（**預期設定**：跨 draw HSR 延後至 Phase 7，因此 Phase
3–4 減少的是 attachment traffic 與 working-set footprint，**shading 工作量完全沒有減少**；
virtual GPU frame time 可能幾乎不動，甚至因 tile 排程 overhead 略為變差。Phase 3–4 不以
time 作為收益指標，frame time 的改善主要來自 Phase 7）：

- host wall time、virtual GPU/model cycles、每 frame 指標，並標記所用的 reference uArch
  cycle model 版本；host wall time 只代表 simulator 成本，Phase 4 前後的 model cycles
  也因版本不同而不可直接比較；
- graphics submissions、render passes、Phase 1/2 次數；
- PBE/attachment LOAD/STORE 次數與估算 bytes；
- binned primitives、tile refs；
- HSR/early-Z rejection；Phase 7 啟用跨 draw HSR 時，另列每種 rejection 與因 side
  effect/order/coverage/output-mask 而 conservative-disable 的次數；
- tile-list/parameter-buffer high-water；
- process RSS、physical footprint、model pool high-water；
- partial-render/spill 次數與原因。

正式 sign-off 的必要條件：

1. `render-pass-tbdr` 不得發生 hidden serialized fallback。
2. 正確性不得比既有 `draw-serialized` baseline 退步。
3. pool ownership 平衡，長時間 run 無持續成長型 leak。
4. Manhattan 與 Car Chase 的 PBE/attachment traffic 下降幅度達到門檻。該門檻不是
   「明顯下降」這種主觀敘述：依 §2.1 與 Phase 0 完成條件，它在 Phase 0 結束時由實測
   traffic baseline 與 split-reason census（可跨越邊界的比例）推導出具體數值並寫回本
   文件，之後才據以驗收。
5. 在沒有 memory pressure 的 case，同一 compatible pass 不應逐 draw STORE/LOAD。
6. 壓力 case 能以 bounded partial render 完成，而不是 OOM 或無上限吃 memory。
7. 相對 llvmpipe 的既有像素差異未擴大；virtual GPU time 所依據的 cycle model 已重校
   且註明版本，host wall time 未冒充硬體效能。
8. 若啟用 Phase 7 HSR，HSR 開/關及兩種排程結果 bit-exact；所有 conservative-disable
   原因已計數並可解釋，未被計入 HSR 收益。

## 10. 預估工程量與提交策略

原型（Phase 1–3）預估約 15–25 個檔案、2k–4k 行，此估計經程式碼複查後維持。

完整版（含穩健 hazard model、tile-local 儲存、bounded memory、Capture/Play、snapshot、
differential debug 和完整 regression）**上修為約 40–60 個檔案、8k–14k 行**。原先
30–50 檔／5k–10k 行的估計低估了兩塊：

- **測試 churn**：`tests/` 現有 248 個檔、約 47k 行 C++，其中 17 個直接斷言 tile/PBE
  counter。counter 語意由 per-draw 變為 per-pass/per-tile 後，這部分的改動量本身就
  接近 2k 行，且無法延後到最後補。
- **Report schema**：`json_reporter.cpp` 本身 2897 行，需同時承載 mode 欄位、pass/split/
  partial-render 計數、HSR 分類與 cycle model 版本，並保留舊 schema 的相容或版本遷移。

另外，主要重寫對象的既有規模供排程參考：`submitter.cpp` 3070 行、`json_reporter.cpp`
2897 行、`texture_unit.cpp` 2927 行、`usc_cluster.cpp` 1917 行、`pbe.cpp` 810 行、
`isp.cpp` 670 行、`fragment_frontend.cpp` 426 行。

這不是單一 patch，應依上述 phase 分成可獨立驗證的 commits。

建議提交順序：

1. submission-boundary reason instrumentation 與 traffic baseline（Phase 0 的前置條件，
   §9 第 4 條門檻由此推導）；
2. 兩種公開 mode contract、internal pass-builder dry-run、logging/report schema；
3. pass builder/split engine（depth alias 先 census 後 enforce），並以 spike 定案管線
   拓撲方案；
4. shared Phase 1 state、最小 tile delta 與 snapshot；
5. correctness-first barrier + in-order tile-major Phase 2 + pass-level LOAD/STORE；
6. tile-local attachment 儲存與 cycle model 重建；
7. bounded partial render；
8. 完整 snapshot/Capture/Play 與 differential tools；
9. 可選的、獨立驗證的跨 draw HSR；
10. performance tuning 與完整 regression artifacts。

任何 phase 若尚未通過自己的完成條件，不得靠 capture-specific workaround 進入下一階段。
